//! pad-constant A/B: old per-element coordinate-decomposition scatter vs the
//! shipped parallel row scatter (copy_from_slice into a constant-filled output).
//!
//! "new" calls the real `UFuncArray::pad` (production path). "old" replicates the
//! previous behaviour: a flat scan over source elements decomposing each into a
//! multi-index and scattering one cell at a time. Both produce bit-identical out.

#![allow(clippy::needless_range_loop, clippy::type_complexity)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;
use std::hint::black_box;

fn old_pad(values: &[f64], shape: &[usize], pad: &[(usize, usize)], constant: f64) -> Vec<f64> {
    let ndim = shape.len();
    let out_shape: Vec<usize> = shape
        .iter()
        .zip(pad)
        .map(|(&s, &(b, a))| s + b + a)
        .collect();
    let mut src_strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        src_strides[d] = src_strides[d + 1] * shape[d + 1];
    }
    let mut out_strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
    }
    let total: usize = out_shape.iter().product();
    let src_count: usize = shape.iter().product();
    let mut out = vec![constant; total];
    for flat in 0..src_count {
        let mut rem = flat;
        let mut of = 0usize;
        for d in 0..ndim {
            let i = rem / src_strides[d];
            rem %= src_strides[d];
            of += (i + pad[d].0) * out_strides[d];
        }
        out[of] = values[flat];
    }
    out
}

fn bench_pad_constant(c: &mut Criterion) {
    let cases: &[(Vec<usize>, Vec<(usize, usize)>)] = &[
        (vec![4096, 4096], vec![(16, 16), (16, 16)]),
        (vec![256, 256, 256], vec![(4, 4), (4, 4), (4, 4)]),
    ];
    for (shape, pad) in cases {
        let n: usize = shape.iter().product();
        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 - 1.0).collect();
        let arr = UFuncArray::new(shape.clone(), data.clone(), DType::F64).unwrap();
        let got = arr.pad(pad, 0.0).unwrap();
        assert_eq!(
            got.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            old_pad(&data, shape, pad, 0.0)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let out_n: usize = got.shape().iter().product();
        let mut group = c.benchmark_group(format!("pad_constant_{shape:?}"));
        group.bench_with_input(BenchmarkId::new("old_scatter", out_n), &out_n, |b, _| {
            b.iter(|| black_box(old_pad(black_box(&data), shape, pad, 0.0)))
        });
        group.bench_with_input(BenchmarkId::new("row_par", out_n), &out_n, |b, _| {
            b.iter(|| black_box(arr.pad(black_box(pad), 0.0).unwrap()))
        });
        group.finish();
    }
}

criterion_group!(benches, bench_pad_constant);
criterion_main!(benches);
