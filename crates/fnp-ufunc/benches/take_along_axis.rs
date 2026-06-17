//! take_along_axis A/B: old serial per-element decode+gather vs the shipped
//! parallel odometer gather.
//!
//! "new" calls the real `UFuncArray::take_along_axis` (production path). "old"
//! replicates the previous behaviour: a serial `for flat { decode; gather }`
//! loop. Both produce bit-identical output.

#![allow(clippy::needless_range_loop)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;
use std::hint::black_box;

fn c_strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

// Faithful replica of the PRE-CHANGE production loop: full `ndim`-division
// decode, per-element finite + bounds validation, and the `source_indices` Vec
// that the old path always materialised (used only for sidecar reindex, but
// built unconditionally even for the common no-sidecar f64 case).
fn old_take_along(
    values: &[f64],
    shape: &[usize],
    idx_vals: &[f64],
    idx_shape: &[usize],
    ax: usize,
) -> Vec<f64> {
    let ndim = shape.len();
    let strides = c_strides(shape);
    let idx_strides = c_strides(idx_shape);
    let axis_len = shape[ax] as i64;
    let total: usize = idx_shape.iter().product();
    let mut out = Vec::with_capacity(total);
    let mut source_indices = Vec::with_capacity(total);
    for flat in 0..total {
        let mut rem = flat;
        let mut src_flat = 0usize;
        for d in 0..ndim {
            let coord = rem / idx_strides[d];
            rem %= idx_strides[d];
            if d == ax {
                let idx_f = idx_vals[flat];
                assert!(idx_f.is_finite());
                let idx = idx_f as i64;
                let resolved = if idx < 0 { idx + axis_len } else { idx };
                assert!(resolved >= 0 && resolved < axis_len);
                src_flat += resolved as usize * strides[d];
            } else {
                let c = if shape[d] == 1 { 0 } else { coord };
                src_flat += c * strides[d];
            }
        }
        out.push(values[src_flat]);
        source_indices.push(src_flat);
    }
    std::hint::black_box(&source_indices);
    out
}

fn bench_take_along(c: &mut Criterion) {
    // (src_shape, idx_shape, axis): a size sweep from cache-resident (where the
    // O(1) decode dominates) up to DRAM-bound (where the random gather floors
    // the win). take_along_axis is most often applied to argsort output on
    // moderate arrays — the cache-resident regime is the representative case.
    let cases: &[(Vec<usize>, Vec<usize>, usize)] = &[
        (vec![1024, 1024], vec![1024, 1024], 1),
        (vec![1024, 1024], vec![1024, 1024], 0),
        (vec![512, 512, 4], vec![512, 512, 4], 1),
        (vec![4096, 4096], vec![4096, 4096], 1),
        (vec![256, 256, 256], vec![256, 256, 256], 1),
    ];
    for (src_shape, idx_shape, ax) in cases {
        let n: usize = src_shape.iter().product();
        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.25 - 3.0).collect();
        let total: usize = idx_shape.iter().product();
        let axis_len = src_shape[*ax];
        let idx_vals: Vec<f64> = (0..total)
            .map(|k| ((k * 2654435761) % axis_len) as f64)
            .collect();
        let arr = UFuncArray::new(src_shape.clone(), data.clone(), DType::F64).unwrap();
        let idx = UFuncArray::new(idx_shape.clone(), idx_vals.clone(), DType::I64).unwrap();
        let got = arr.take_along_axis(&idx, *ax as isize).unwrap();
        assert_eq!(
            got.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            old_take_along(&data, src_shape, &idx_vals, idx_shape, *ax)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let mut group = c.benchmark_group(format!("take_along_{src_shape:?}_ax{ax}"));
        group.bench_with_input(BenchmarkId::new("old_serial", total), &total, |b, _| {
            b.iter(|| {
                black_box(old_take_along(
                    black_box(&data),
                    src_shape,
                    &idx_vals,
                    idx_shape,
                    *ax,
                ))
            })
        });
        group.bench_with_input(BenchmarkId::new("par_odometer", total), &total, |b, _| {
            b.iter(|| black_box(arr.take_along_axis(black_box(&idx), *ax as isize).unwrap()))
        });
        group.finish();
    }
}

criterion_group!(benches, bench_take_along);
criterion_main!(benches);
