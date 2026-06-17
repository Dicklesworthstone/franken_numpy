//! broadcast_to A/B: old per-element coordinate-decomposition gather vs the
//! shipped parallel chunked-odometer gather.
//!
//! "new" calls the real `UFuncArray::broadcast_to` (production path). "old"
//! replicates the previous behaviour: a flat scan decomposing every output index
//! with `ndim` integer divisions, then gathering with broadcast-aware strides.
//! Both produce bit-identical output.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;
use std::hint::black_box;

fn old_bcast(values: &[f64], src_shape: &[usize], target: &[usize]) -> Vec<f64> {
    let ndim = target.len();
    let mut padded = vec![1usize; ndim - src_shape.len()];
    padded.extend_from_slice(src_shape);
    let mut src_strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        src_strides[d] = src_strides[d + 1] * padded[d + 1];
    }
    let mut out_strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        out_strides[d] = out_strides[d + 1] * target[d + 1];
    }
    let total: usize = target.iter().product();
    let mut out = Vec::with_capacity(total);
    for f in 0..total {
        let mut rem = f;
        let mut src = 0usize;
        for d in 0..ndim {
            let coord = rem / out_strides[d];
            rem %= out_strides[d];
            if padded[d] > 1 {
                src += coord * src_strides[d];
            }
        }
        out.push(values[src]);
    }
    out
}

fn bench_broadcast(c: &mut Criterion) {
    let cases: &[(Vec<usize>, Vec<usize>)] = &[
        (vec![1, 4096], vec![4096, 4096]),        // leading broadcast
        (vec![4096, 1], vec![4096, 4096]),        // trailing broadcast
        (vec![256, 1, 256], vec![256, 256, 256]), // interior broadcast
    ];
    for (src_shape, target) in cases {
        let sn: usize = src_shape.iter().product();
        let data: Vec<f64> = (0..sn).map(|i| (i as f64) * 0.5 - 1.0).collect();
        let arr = UFuncArray::new(src_shape.clone(), data.clone(), DType::F64).unwrap();
        let got = arr.broadcast_to(target).unwrap();
        assert_eq!(
            got.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            old_bcast(&data, src_shape, target)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let n: usize = target.iter().product();
        let mut group = c.benchmark_group(format!("broadcast_{src_shape:?}_to_{target:?}"));
        group.bench_with_input(BenchmarkId::new("old_gather", n), &n, |b, _| {
            b.iter(|| black_box(old_bcast(black_box(&data), src_shape, target)))
        });
        group.bench_with_input(BenchmarkId::new("odometer_par", n), &n, |b, _| {
            b.iter(|| black_box(arr.broadcast_to(black_box(target)).unwrap()))
        });
        group.finish();
    }
}

criterion_group!(benches, bench_broadcast);
criterion_main!(benches);
