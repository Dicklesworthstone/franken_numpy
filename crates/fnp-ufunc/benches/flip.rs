//! flip(axis) A/B: old element-by-element in-place swap vs the shipped parallel
//! copy_from_slice row build.
//!
//! "new" calls the real `UFuncArray::flip` (production path). "old" replicates the
//! previous behaviour: clone then swap mirror elements one at a time. Both produce
//! bit-identical output.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;
use std::hint::black_box;

fn old_flip(values: &[f64], shape: &[usize], ax: usize) -> Vec<f64> {
    let inner: usize = shape[ax + 1..].iter().product();
    let outer: usize = shape[..ax].iter().product();
    let axis_len = shape[ax];
    let mut out = values.to_vec();
    for o in 0..outer {
        for k in 0..axis_len / 2 {
            let rk = axis_len - 1 - k;
            for i in 0..inner {
                let a = o * axis_len * inner + k * inner + i;
                let b = o * axis_len * inner + rk * inner + i;
                out.swap(a, b);
            }
        }
    }
    out
}

fn old_rot90_k2(arr: &UFuncArray) -> UFuncArray {
    arr.flip(Some(0)).unwrap().flip(Some(1)).unwrap()
}

fn bench_rot90_k2(c: &mut Criterion) {
    let shape = vec![768, 512];
    let n: usize = shape.iter().product();
    let data: Vec<f64> = (0..n)
        .map(|i| match i % 8 {
            0 => -0.0,
            1 => f64::from_bits(0x7ff8_0000_0000_0042),
            _ => (i as f64) * 0.5 - 1.0,
        })
        .collect();
    let arr = UFuncArray::new(shape, data, DType::F64).unwrap();

    let old = old_rot90_k2(&arr);
    let new = arr.rot90(2).unwrap();
    assert_eq!(new.shape(), old.shape());
    assert_eq!(
        new.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        old.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );

    let mut group = c.benchmark_group("rot90_rank2_k2");
    group.bench_with_input(BenchmarkId::new("old_two_flips", n), &n, |b, _| {
        b.iter(|| black_box(old_rot90_k2(black_box(&arr))))
    });
    group.bench_with_input(BenchmarkId::new("one_reverse", n), &n, |b, _| {
        b.iter(|| black_box(arr.rot90(black_box(2)).unwrap()))
    });
    group.finish();
}

fn bench_flip(c: &mut Criterion) {
    // (shape, axis): axis 0 (single outer block), middle axis (many blocks),
    // last axis (inner==1).
    let cases: &[(Vec<usize>, usize)] = &[
        (vec![4096, 4096], 0),
        (vec![64, 256, 256], 1),
        (vec![4096, 4096], 1),
    ];
    for (shape, ax) in cases {
        let n: usize = shape.iter().product();
        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 - 1.0).collect();
        let arr = UFuncArray::new(shape.clone(), data.clone(), DType::F64).unwrap();
        let got = arr.flip(Some(*ax as isize)).unwrap();
        assert_eq!(
            got.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            old_flip(&data, shape, *ax)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let mut group = c.benchmark_group(format!("flip_{shape:?}_ax{ax}"));
        group.bench_with_input(BenchmarkId::new("old_swap", n), &n, |b, _| {
            b.iter(|| black_box(old_flip(black_box(&data), shape, *ax)))
        });
        group.bench_with_input(BenchmarkId::new("copyslice_par", n), &n, |b, _| {
            b.iter(|| black_box(arr.flip(black_box(Some(*ax as isize))).unwrap()))
        });
        group.finish();
    }
}

criterion_group!(benches, bench_rot90_k2, bench_flip);
criterion_main!(benches);
