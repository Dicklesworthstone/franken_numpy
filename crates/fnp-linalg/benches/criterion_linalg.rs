//! Criterion benchmarks for fnp-linalg.
//!
//! Measures performance baselines for core linear algebra operations:
//! - solve_nxn: linear system solving
//! - det_nxn: determinant computation
//! - inv_nxn: matrix inversion
//! - cholesky_nxn: Cholesky decomposition
//! - qr_nxn: QR decomposition
//! - svd_nxn: singular value decomposition
//! - eigvalsh_nxn: symmetric eigenvalues
//! - matrix_norm_frobenius: Frobenius norm
//!
//! Finding: fnp-linalg (10,120 LOC) had ZERO benchmarks despite containing
//! performance-critical numerical algorithms.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_linalg::{
    batch_cholesky, batch_eigvalsh, batch_inv, cholesky_nxn, det_nxn, eigvalsh_nxn, inv_nxn,
    matrix_norm_frobenius, qr_nxn, solve_nxn, svd_nxn,
};
use std::hint::black_box;

fn generate_spd_matrix(n: usize) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = if i == j {
                (n + 1) as f64
            } else {
                1.0 / ((i as f64 - j as f64).abs() + 1.0)
            };
        }
    }
    a
}

fn generate_random_matrix(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n * n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
        })
        .collect()
}

fn generate_invertible_matrix(n: usize) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = if i == j {
                (n * 2) as f64
            } else {
                ((i + j) % 5) as f64 * 0.1
            };
        }
    }
    a
}

fn bench_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_nxn");

    for n in [16, 32, 64, 128, 256] {
        let a = generate_invertible_matrix(n);
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        group.bench_with_input(BenchmarkId::new("size", n), &n, |bench, _| {
            bench.iter(|| {
                let result = solve_nxn(black_box(&a), black_box(&b), n);
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_det(c: &mut Criterion) {
    let mut group = c.benchmark_group("det_nxn");

    for n in [16, 32, 64, 128, 256] {
        let a = generate_random_matrix(n, 42);

        group.bench_with_input(BenchmarkId::new("size", n), &n, |bench, _| {
            bench.iter(|| {
                let result = det_nxn(black_box(&a), n);
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_inv(c: &mut Criterion) {
    let mut group = c.benchmark_group("inv_nxn");

    for n in [16, 32, 64, 128, 256] {
        let a = generate_invertible_matrix(n);

        group.bench_with_input(BenchmarkId::new("size", n), &n, |bench, _| {
            bench.iter(|| {
                let result = inv_nxn(black_box(&a), n);
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_cholesky(c: &mut Criterion) {
    let mut group = c.benchmark_group("cholesky_nxn");

    for n in [16, 32, 64, 128, 256] {
        let a = generate_spd_matrix(n);

        group.bench_with_input(BenchmarkId::new("size", n), &n, |bench, _| {
            bench.iter(|| {
                let result = cholesky_nxn(black_box(&a), n);
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_qr(c: &mut Criterion) {
    let mut group = c.benchmark_group("qr_nxn");

    for n in [16, 32, 64, 128] {
        let a = generate_random_matrix(n, 123);

        group.bench_with_input(BenchmarkId::new("size", n), &n, |bench, _| {
            bench.iter(|| {
                let result = qr_nxn(black_box(&a), n);
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_svd(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd_nxn");

    for n in [16, 32, 64, 128] {
        let a = generate_random_matrix(n, 456);

        group.bench_with_input(BenchmarkId::new("size", n), &n, |bench, _| {
            bench.iter(|| {
                let result = svd_nxn(black_box(&a), n);
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_eigvalsh(c: &mut Criterion) {
    let mut group = c.benchmark_group("eigvalsh_nxn");

    for n in [16, 32, 64, 128] {
        let a = generate_spd_matrix(n);

        group.bench_with_input(BenchmarkId::new("size", n), &n, |bench, _| {
            bench.iter(|| {
                let result = eigvalsh_nxn(black_box(&a), n);
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_norm_frobenius(c: &mut Criterion) {
    let mut group = c.benchmark_group("norm_frobenius");

    for n in [64, 128, 256, 512, 1024] {
        let a = generate_random_matrix(n, 789);

        group.bench_with_input(BenchmarkId::new("size", n), &n, |bench, _| {
            bench.iter(|| {
                let result = matrix_norm_frobenius(black_box(&a), n);
                black_box(result)
            });
        });
    }

    group.finish();
}

// Stacked-matrix ("batched") workloads: NumPy-style leading batch dims. These
// loop over many independent matrices, the embarrassingly-parallel hot path
// exercised by np.linalg.{inv,eigvalsh,cholesky} on (..., n, n) arrays.

fn generate_batch_spd(batch: usize, n: usize) -> (Vec<f64>, Vec<usize>) {
    let mat_size = n * n;
    let mut data = Vec::with_capacity(batch * mat_size);
    for b in 0..batch {
        // Per-lane diagonal perturbation keeps every matrix distinct so the
        // benchmark measures real work rather than a single cached result.
        let bump = (b % 7) as f64 * 0.25;
        for i in 0..n {
            for j in 0..n {
                data.push(if i == j {
                    (n + 1) as f64 + bump
                } else {
                    1.0 / ((i as f64 - j as f64).abs() + 1.0)
                });
            }
        }
    }
    (data, vec![batch, n, n])
}

fn generate_batch_invertible(batch: usize, n: usize) -> (Vec<f64>, Vec<usize>) {
    let mat_size = n * n;
    let mut data = Vec::with_capacity(batch * mat_size);
    for b in 0..batch {
        let bump = (b % 5) as f64 * 0.5;
        for i in 0..n {
            for j in 0..n {
                data.push(if i == j {
                    (n * 2) as f64 + bump
                } else {
                    ((i + j) % 5) as f64 * 0.1
                });
            }
        }
    }
    (data, vec![batch, n, n])
}

fn bench_batch_inv(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_inv");

    // Compute-bound per-lane sizes (n >= 128): O(n^3) work dominates fixed
    // per-call overhead, the regime where lane parallelism pays.
    for (batch, n) in [(64usize, 128usize), (16, 256)] {
        let (data, shape) = generate_batch_invertible(batch, n);
        let id = format!("{batch}x{n}x{n}");
        group.bench_with_input(BenchmarkId::new("shape", id), &shape, |bench, shape| {
            bench.iter(|| {
                let result = batch_inv(black_box(&data), black_box(shape));
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_batch_eigvalsh(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_eigvalsh");

    for (batch, n) in [(64usize, 128usize), (16, 256)] {
        let (data, shape) = generate_batch_spd(batch, n);
        let id = format!("{batch}x{n}x{n}");
        group.bench_with_input(BenchmarkId::new("shape", id), &shape, |bench, shape| {
            bench.iter(|| {
                let result = batch_eigvalsh(black_box(&data), black_box(shape));
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_batch_cholesky(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_cholesky");

    for (batch, n) in [(64usize, 128usize), (16, 256)] {
        let (data, shape) = generate_batch_spd(batch, n);
        let id = format!("{batch}x{n}x{n}");
        group.bench_with_input(BenchmarkId::new("shape", id), &shape, |bench, shape| {
            bench.iter(|| {
                let result = batch_cholesky(black_box(&data), black_box(shape));
                black_box(result)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_solve,
    bench_det,
    bench_inv,
    bench_cholesky,
    bench_qr,
    bench_svd,
    bench_eigvalsh,
    bench_norm_frobenius,
    bench_batch_inv,
    bench_batch_eigvalsh,
    bench_batch_cholesky,
);

criterion_main!(benches);
