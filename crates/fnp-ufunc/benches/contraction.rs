use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;
use std::hint::black_box;

fn make_2d(m: usize, n: usize) -> UFuncArray {
    let values: Vec<f64> = (0..m * n).map(|i| (i as f64) * 0.5 + 1.0).collect();
    UFuncArray::new(vec![m, n], values, DType::F64).unwrap()
}

fn make_3d(a: usize, b: usize, c: usize) -> UFuncArray {
    let values: Vec<f64> = (0..a * b * c).map(|i| (i as f64) * 0.25 + 0.5).collect();
    UFuncArray::new(vec![a, b, c], values, DType::F64).unwrap()
}

/// `ij,jk->ik` — the canonical matmul-shaped contraction (n^3 inner iterations).
fn bench_einsum_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("einsum_ij_jk_ik");
    for size in [32usize, 64, 96, 128].iter() {
        let a = make_2d(*size, *size);
        let b = make_2d(*size, *size);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| UFuncArray::einsum("ij,jk->ik", black_box(&[&a, &b])).unwrap())
        });
    }
    group.finish();
}

/// `ijk,ikl->ijl` — batched contraction (4-D iteration space).
fn bench_einsum_batched(c: &mut Criterion) {
    let mut group = c.benchmark_group("einsum_ijk_ikl_ijl");
    for size in [16usize, 24, 32].iter() {
        let a = make_3d(*size, *size, *size);
        let b = make_3d(*size, *size, *size);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| UFuncArray::einsum("ijk,ikl->ijl", black_box(&[&a, &b])).unwrap())
        });
    }
    group.finish();
}

/// `ij,ij->` — full reduction to a scalar.
fn bench_einsum_reduce(c: &mut Criterion) {
    let mut group = c.benchmark_group("einsum_ij_ij_scalar");
    for size in [128usize, 256, 512].iter() {
        let a = make_2d(*size, *size);
        let b = make_2d(*size, *size);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| UFuncArray::einsum("ij,ij->", black_box(&[&a, &b])).unwrap())
        });
    }
    group.finish();
}

/// `tensordot(A, B, axes=1)` over square matrices — GEMM-shaped contraction.
fn bench_tensordot(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensordot_axes1");
    for size in [64usize, 128, 256].iter() {
        let a = make_2d(*size, *size);
        let b = make_2d(*size, *size);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| black_box(&a).tensordot(black_box(&b), 1).unwrap())
        });
    }
    group.finish();
}

/// `inner(A, B)` over square matrices — contracts the last axis (B^T-shaped).
fn bench_inner(c: &mut Criterion) {
    let mut group = c.benchmark_group("inner_lastaxis");
    for size in [64usize, 128, 256].iter() {
        let a = make_2d(*size, *size);
        let b = make_2d(*size, *size);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| black_box(&a).inner(black_box(&b)).unwrap())
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_einsum_matmul,
    bench_einsum_batched,
    bench_einsum_reduce,
    bench_tensordot,
    bench_inner
);
criterion_main!(benches);
