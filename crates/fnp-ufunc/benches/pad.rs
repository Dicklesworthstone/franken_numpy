//! pad gather-mode A/B: old per-element coordinate-decomposition serial gather vs
//! the shipped precomputed-axis-map parallel row gather (pad_gather_by_axis_maps).
//!
//! "new" calls the real `UFuncArray::pad_reflect` / `pad_edge` (production path).
//! "old" replicates the previous behaviour: a flat scan decomposing every output
//! index with `ndim` divisions and applying the mode's branchy index logic per
//! element. Both produce bit-identical output.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;
use std::hint::black_box;

fn reflect_index(idx: isize, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let period = 2 * (n as isize - 1);
    if period == 0 {
        return 0;
    }
    let mut i = idx.rem_euclid(period);
    if i >= n as isize {
        i = period - i;
    }
    i as usize
}

fn old_reflect(values: &[f64], shape: &[usize], pad: &[(usize, usize)]) -> Vec<f64> {
    let ndim = shape.len();
    let out_shape: Vec<usize> = shape.iter().zip(pad).map(|(&s, &(b, a))| s + b + a).collect();
    let mut src_strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        src_strides[d] = src_strides[d + 1] * shape[d + 1];
    }
    let mut out_strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
    }
    let total: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(total);
    for out_flat in 0..total {
        let mut rem = out_flat;
        let mut src = 0usize;
        for d in 0..ndim {
            let oi = rem / out_strides[d];
            rem %= out_strides[d];
            let si = reflect_index(oi as isize - pad[d].0 as isize, shape[d]);
            src += si * src_strides[d];
        }
        out.push(values[src]);
    }
    out
}

fn bench_pad(c: &mut Criterion) {
    let cases: &[(Vec<usize>, Vec<(usize, usize)>)] = &[
        (vec![4096, 4096], vec![(64, 64), (64, 64)]),
        (vec![256, 256, 256], vec![(8, 8), (8, 8), (8, 8)]),
    ];
    for (shape, pad) in cases {
        let n: usize = shape.iter().product();
        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 - 1.0).collect();
        let arr = UFuncArray::new(shape.clone(), data.clone(), DType::F64).unwrap();
        let got = arr.pad_reflect(pad).unwrap();
        assert_eq!(
            got.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            old_reflect(&data, shape, pad)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let out_n: usize = got.shape().iter().product();
        let mut group = c.benchmark_group(format!("pad_reflect_{shape:?}"));
        group.bench_with_input(BenchmarkId::new("old_gather", out_n), &out_n, |b, _| {
            b.iter(|| black_box(old_reflect(black_box(&data), shape, pad)))
        });
        group.bench_with_input(BenchmarkId::new("axis_map_par", out_n), &out_n, |b, _| {
            b.iter(|| black_box(arr.pad_reflect(black_box(pad)).unwrap()))
        });
        group.finish();
    }
}

criterion_group!(benches, bench_pad);
criterion_main!(benches);
