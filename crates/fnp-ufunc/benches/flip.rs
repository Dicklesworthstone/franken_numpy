//! flip(axis) A/B: old element-by-element in-place swap vs the shipped parallel
//! copy_from_slice row build.
//!
//! "new" calls the real `UFuncArray::flip` (production path). "old" replicates the
//! previous behaviour: clone then swap mirror elements one at a time. Both produce
//! bit-identical output.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Duration;

/// Frozen replica of the CURRENT singleton-axis flip kernel: zero-filled
/// output plus the generic block copy (which degenerates to one-element
/// parallel chunks for `[outer, 1]` axis 1), exactly as `flip_axis_build`
/// runs it today. The identity-clone candidate must reproduce it bit-for-bit.
#[inline(never)]
fn former_flip_singleton_values(
    src: &[f64],
    outer: usize,
    axis_len: usize,
    inner: usize,
) -> Vec<f64> {
    let n = outer * axis_len * inner;
    let mut out = vec![0.0f64; n];
    if n == 0 {
        return out;
    }
    let block = axis_len * inner;
    let do_block = |(out_blk, in_blk): (&mut [f64], &[f64])| {
        for k in 0..axis_len {
            let rk = axis_len - 1 - k;
            out_blk[k * inner..k * inner + inner]
                .copy_from_slice(&in_blk[rk * inner..rk * inner + inner]);
        }
    };
    const FLIP_PAR_MIN: usize = 1 << 15;
    if outer >= 2 && n >= FLIP_PAR_MIN && rayon::current_num_threads() >= 2 {
        out.par_chunks_mut(block)
            .zip(src.par_chunks(block))
            .for_each(do_block);
    } else if n >= FLIP_PAR_MIN && rayon::current_num_threads() >= 2 {
        out.par_chunks_mut(inner.max(1))
            .enumerate()
            .for_each(|(k, out_row)| {
                let rk = axis_len - 1 - k;
                out_row.copy_from_slice(&src[rk * inner..rk * inner + inner]);
            });
    } else {
        out.chunks_mut(block)
            .zip(src.chunks(block))
            .for_each(do_block);
    }
    out
}

fn bench_flip_singleton_axis(c: &mut Criterion) {
    const OUTER: usize = 131_071;
    let data: Vec<f64> = (0..OUTER)
        .map(|i| f64::from_bits(0x3ff0_0000_0000_0000 ^ (i as u64).wrapping_mul(0x9e37)))
        .collect();
    let arr = UFuncArray::new(vec![OUTER, 1], data.clone(), DType::F64).unwrap();

    let former = former_flip_singleton_values(&data, OUTER, 1, 1);
    let public = arr.flip(Some(1)).unwrap();
    assert_eq!(public.shape(), &[OUTER, 1]);
    assert!(
        public
            .values()
            .iter()
            .zip(&former)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    // .319 retry protocol: 20 samples and a 2 s window on a warm pinned
    // worker, floor predeclared in the bead/ledger.
    let mut group = c.benchmark_group("flip_singleton_axis");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(OUTER as u64));
    group.bench_function("former_axis_build_kernel", |bench| {
        bench.iter(|| black_box(former_flip_singleton_values(black_box(&data), OUTER, 1, 1)))
    });
    group.bench_function("candidate_identity_clone", |bench| {
        bench.iter(|| black_box(arr.flip(black_box(Some(1))).unwrap()))
    });
    group.finish();
}

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

fn old_flip_axes2(arr: &UFuncArray) -> UFuncArray {
    arr.flip(Some(0)).unwrap().flip(Some(1)).unwrap()
}

fn bench_flip_axes2(c: &mut Criterion) {
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

    let old = old_flip_axes2(&arr);
    let new = arr.flip_axes(&[0, 1]).unwrap();
    assert_eq!(new.shape(), old.shape());
    assert_eq!(
        new.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        old.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );

    let mut group = c.benchmark_group("flip_axes_rank2_two_axes");
    group.bench_with_input(BenchmarkId::new("old_two_flips", n), &n, |b, _| {
        b.iter(|| black_box(old_flip_axes2(black_box(&arr))))
    });
    group.bench_with_input(BenchmarkId::new("half_turn_route", n), &n, |b, _| {
        b.iter(|| black_box(arr.flip_axes(black_box(&[0, 1])).unwrap()))
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

criterion_group!(
    benches,
    bench_rot90_k2,
    bench_flip_axes2,
    bench_flip,
    bench_flip_singleton_axis
);
criterion_main!(benches);
