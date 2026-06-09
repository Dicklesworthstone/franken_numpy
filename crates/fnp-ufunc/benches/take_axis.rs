//! take(axis) A/B: old serial nested gather (extend_from_slice rows) vs the
//! shipped parallel copy_from_slice block/row build.
//!
//! "new" calls the real `UFuncArray::take` (production path). "old" replicates
//! the previous behaviour: a serial `for o { for ri { extend_from_slice } }`
//! gather. Both produce bit-identical output.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;
use std::hint::black_box;

fn old_take_axis(values: &[f64], shape: &[usize], ax: usize, idx: &[i64]) -> Vec<f64> {
    let inner: usize = shape[ax + 1..].iter().product();
    let outer: usize = shape[..ax].iter().product();
    let axis_len = shape[ax] as i64;
    let resolved: Vec<usize> = idx
        .iter()
        .map(|&i| (if i < 0 { i + axis_len } else { i }) as usize)
        .collect();
    let src_stride = shape[ax] * inner;
    let mut out = Vec::with_capacity(outer * resolved.len() * inner);
    for o in 0..outer {
        for &ri in &resolved {
            let base = o * src_stride + ri * inner;
            out.extend_from_slice(&values[base..base + inner]);
        }
    }
    out
}

fn bench_take_axis(c: &mut Criterion) {
    // (shape, axis, num_indices): axis 0 (single outer block, parallel over
    // rows), middle axis (many outer blocks, inner>1), last axis (inner==1).
    let cases: &[(Vec<usize>, usize, usize)] = &[
        (vec![8192, 1024], 0, 4096),
        (vec![1024, 512, 16], 1, 256),
        (vec![4096, 2048], 0, 2048),
    ];
    for (shape, ax, nidx) in cases {
        let n: usize = shape.iter().product();
        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 - 1.0).collect();
        let arr = UFuncArray::new(shape.clone(), data.clone(), DType::F64).unwrap();
        // Deterministic mix of forward, duplicate, and negative indices.
        let axis_len = shape[*ax] as i64;
        let idx: Vec<i64> = (0..*nidx)
            .map(|k| {
                let v = ((k as i64) * 2654435761) % axis_len;
                if k % 5 == 0 { v - axis_len } else { v }
            })
            .collect();
        let got = arr.take(&idx, Some(*ax as isize)).unwrap();
        assert_eq!(
            got.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            old_take_axis(&data, shape, *ax, &idx)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let mut group = c.benchmark_group(format!("take_{shape:?}_ax{ax}_n{nidx}"));
        group.bench_with_input(BenchmarkId::new("old_serial", n), &n, |b, _| {
            b.iter(|| black_box(old_take_axis(black_box(&data), shape, *ax, &idx)))
        });
        group.bench_with_input(BenchmarkId::new("copyslice_par", n), &n, |b, _| {
            b.iter(|| black_box(arr.take(black_box(&idx), Some(*ax as isize)).unwrap()))
        });
        group.finish();
    }
}

criterion_group!(benches, bench_take_axis);
criterion_main!(benches);
