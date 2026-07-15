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
    batch_cholesky, batch_det, batch_eigvalsh, batch_inv, batch_matrix_norm, batch_slogdet,
    batch_trace, cholesky_nxn, cholesky_nxn_general_control, complex_matmul, cond_nxn, det_nxn,
    det_nxn_general_control, eigvalsh_nxn, inv_nxn, kron_nxn, matrix_norm_frobenius,
    matrix_norm_nxn, matrix_power_nxn, multi_dot, qr_mxn, qr_nxn,
    sbr_stage1_dense_to_band_lower_nxn, slogdet_nxn, slogdet_nxn_general_control, solve_nxn,
    svd_mxn_full, svd_nxn,
};
use std::hint::black_box;
use std::time::Duration;

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

fn generate_spd_tridiagonal_matrix(n: usize) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        a[i * n + i] = 2.0;
        if i + 1 < n {
            a[i * n + i + 1] = -1.0;
            a[(i + 1) * n + i] = -1.0;
        }
    }
    a
}

fn generate_descending_diagonal_matrix(n: usize) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        a[i * n + i] = (n - i) as f64 + 0.25;
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

fn generate_upper_triangular_matrix(n: usize) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for row in 0..n {
        for col in row..n {
            a[row * n + col] = if row == col {
                (row % 17 + 1) as f64 * 0.25
            } else {
                ((row * 31 + col * 17) % 23) as f64 * 0.125 - 1.25
            };
        }
    }
    a
}

fn generate_upper_trapezoidal_matrix(m: usize, n: usize) -> Vec<f64> {
    let mut a = vec![0.0; m * n];
    for row in 0..m.min(n) {
        for col in row..n {
            a[row * n + col] = if row == col {
                (row % 19 + 1) as f64 * 0.25
            } else {
                ((row * 29 + col * 13) % 29) as f64 * 0.125 - 1.5
            };
        }
    }
    a
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

    // Through n=1024: the factorizations route the trailing update through the
    // blocked packed GEMM there, so this is the regime where the remaining gap to
    // OpenBLAS getrf (the AVX2-no-FMA microkernel, bead 8vdtg) and any future
    // sub-cubic GEMM (Strassen, crossover ~n>=4096) actually show up.
    for n in [16, 64, 128, 256, 512, 768, 1024] {
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

fn bench_det_exact_upper_triangular(c: &mut Criterion) {
    let n = 256usize;
    let mut matrix = vec![0.0; n * n];
    for row in 0..n {
        let magnitude = 1.0 + (row % 7) as f64 * 0.001;
        matrix[row * n + row] = if row % 2 == 0 { magnitude } else { -magnitude };
        for col in (row + 1)..n {
            matrix[row * n + col] = ((row * 17 + col * 13) % 97) as f64 / 101.0 - 0.4;
        }
    }

    let former = det_nxn_general_control(&matrix, n).expect("general determinant");
    let candidate = det_nxn(&matrix, n).expect("structured determinant");
    assert_eq!(candidate.to_bits(), former.to_bits());

    let mut group = c.benchmark_group("det_exact_upper_triangular_256");
    group.bench_function("former_partial_pivot_lu", |bench| {
        bench.iter(|| black_box(det_nxn_general_control(black_box(&matrix), n).unwrap()))
    });
    group.bench_function("structured_diagonal_product", |bench| {
        bench.iter(|| black_box(det_nxn(black_box(&matrix), n).unwrap()))
    });
    group.finish();
}

fn bench_det_exact_lower_triangular_no_pivot(c: &mut Criterion) {
    let n = 256usize;
    let mut matrix = vec![0.0; n * n];
    for row in 0..n {
        let magnitude = 2.0 + (row % 7) as f64 * 0.001;
        matrix[row * n + row] = if row % 2 == 0 { magnitude } else { -magnitude };
        for col in 0..row {
            matrix[row * n + col] = ((row * 17 + col * 13) % 97) as f64 / 101.0 - 0.4;
        }
    }

    let former = det_nxn_general_control(&matrix, n).expect("general determinant");
    let candidate = det_nxn(&matrix, n).expect("structured determinant");
    assert_eq!(candidate.to_bits(), former.to_bits());

    let mut group = c.benchmark_group("det_exact_lower_triangular_no_pivot_256");
    group.bench_function("former_partial_pivot_lu", |bench| {
        bench.iter(|| black_box(det_nxn_general_control(black_box(&matrix), n).unwrap()))
    });
    group.bench_function("structured_diagonal_product", |bench| {
        bench.iter(|| black_box(det_nxn(black_box(&matrix), n).unwrap()))
    });
    group.finish();
}

fn bench_slogdet_exact_upper_triangular(c: &mut Criterion) {
    let n = 256usize;
    let mut matrix = vec![0.0; n * n];
    for row in 0..n {
        let magnitude = 1.0 + (row % 7) as f64 * 0.001;
        matrix[row * n + row] = if row % 2 == 0 { magnitude } else { -magnitude };
        for col in (row + 1)..n {
            matrix[row * n + col] = ((row * 17 + col * 13) % 97) as f64 / 101.0 - 0.4;
        }
    }

    let former = slogdet_nxn_general_control(&matrix, n).expect("general sign and log determinant");
    let candidate = slogdet_nxn(&matrix, n).expect("structured sign and log determinant");
    assert_eq!(candidate.0.to_bits(), former.0.to_bits());
    assert_eq!(candidate.1.to_bits(), former.1.to_bits());

    let mut group = c.benchmark_group("slogdet_exact_upper_triangular_256");
    group.bench_function("former_partial_pivot_lu", |bench| {
        bench.iter(|| black_box(slogdet_nxn_general_control(black_box(&matrix), n).unwrap()))
    });
    group.bench_function("structured_diagonal_log_fold", |bench| {
        bench.iter(|| black_box(slogdet_nxn(black_box(&matrix), n).unwrap()))
    });
    group.finish();
}

fn bench_slogdet_exact_lower_triangular_no_pivot(c: &mut Criterion) {
    let n = 256usize;
    let mut matrix = vec![0.0; n * n];
    for row in 0..n {
        let magnitude = 2.0 + (row % 7) as f64 * 0.001;
        matrix[row * n + row] = if row % 2 == 0 { magnitude } else { -magnitude };
        for col in 0..row {
            matrix[row * n + col] = ((row * 17 + col * 13) % 97) as f64 / 101.0 - 0.4;
        }
    }

    let former = slogdet_nxn_general_control(&matrix, n).expect("general sign and log determinant");
    let candidate = slogdet_nxn(&matrix, n).expect("structured sign and log determinant");
    assert_eq!(candidate.0.to_bits(), former.0.to_bits());
    assert_eq!(candidate.1.to_bits(), former.1.to_bits());

    let mut group = c.benchmark_group("slogdet_exact_lower_triangular_no_pivot_256");
    group.bench_function("former_partial_pivot_lu", |bench| {
        bench.iter(|| black_box(slogdet_nxn_general_control(black_box(&matrix), n).unwrap()))
    });
    group.bench_function("structured_diagonal_log_fold", |bench| {
        bench.iter(|| black_box(slogdet_nxn(black_box(&matrix), n).unwrap()))
    });
    group.finish();
}

fn bench_inv(c: &mut Criterion) {
    let mut group = c.benchmark_group("inv_nxn");

    for n in [16, 32, 64, 128, 256, 512, 768, 1024] {
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

    // Add 512/768: above CHOL_MID_MIN the blocked Cholesky's serial diagonal/panel
    // factorization dominates (the trailing GEMM is already a small fraction), the
    // regime that bounds the gap to OpenBLAS potrf.
    for n in [16, 32, 64, 128, 256, 512, 768] {
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

fn bench_cholesky_exact_diagonal(c: &mut Criterion) {
    let n = 256usize;
    let a = generate_descending_diagonal_matrix(n);
    let former = cholesky_nxn_general_control(&a, n).expect("former Cholesky control");
    let candidate = cholesky_nxn(&a, n).expect("diagonal Cholesky candidate");
    assert_eq!(former.len(), candidate.len());
    for (index, (&lhs, &rhs)) in former.iter().zip(&candidate).enumerate() {
        assert_eq!(
            lhs.to_bits(),
            rhs.to_bits(),
            "diagonal Cholesky output {index} changed bits"
        );
    }

    let mut group = c.benchmark_group("cholesky_exact_diagonal_256");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_secs(1));
    group.bench_function("former_general_control", |bench| {
        bench.iter(|| black_box(cholesky_nxn_general_control(black_box(&a), n).unwrap()));
    });
    group.bench_function("exact_diagonal_candidate", |bench| {
        bench.iter(|| black_box(cholesky_nxn(black_box(&a), n).unwrap()));
    });
    group.finish();
}

fn bench_cholesky_exact_tridiagonal(c: &mut Criterion) {
    let n = 256usize;
    let a = generate_spd_tridiagonal_matrix(n);
    let former = cholesky_nxn_general_control(&a, n).expect("former Cholesky control");
    let candidate = cholesky_nxn(&a, n).expect("tridiagonal Cholesky candidate");
    assert_eq!(former.len(), candidate.len());
    for (index, (&lhs, &rhs)) in former.iter().zip(&candidate).enumerate() {
        assert_eq!(
            lhs.to_bits(),
            rhs.to_bits(),
            "tridiagonal Cholesky output {index} changed bits"
        );
    }

    let mut group = c.benchmark_group("cholesky_exact_tridiagonal_256");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.bench_function("former_general_control", |bench| {
        bench.iter(|| black_box(cholesky_nxn_general_control(black_box(&a), n).unwrap()));
    });
    group.bench_function("exact_tridiagonal_candidate", |bench| {
        bench.iter(|| black_box(cholesky_nxn(black_box(&a), n).unwrap()));
    });
    group.finish();
}

fn bench_qr(c: &mut Criterion) {
    let mut group = c.benchmark_group("qr_nxn");

    for n in [16, 32, 64, 128, 256, 512] {
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

fn qr_upper_triangular_former(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut q = vec![0.0; n * n];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }
    let mut r = a.to_vec();
    let mut v = vec![0.0; n];
    let mut d = vec![0.0; n];
    let mut f_vec = vec![0.0; n];

    for k in 0..n {
        let mut col_norm_sq = 0.0;
        for i in k..n {
            col_norm_sq += r[i * n + k] * r[i * n + k];
        }
        let col_norm = col_norm_sq.sqrt();
        if col_norm == 0.0 {
            continue;
        }

        let sign = if r[k * n + k] >= 0.0 { 1.0 } else { -1.0 };
        for i in k..n {
            v[i] = r[i * n + k];
        }
        v[k] += sign * col_norm;
        let v_norm_sq: f64 = v[k..].iter().map(|x| x * x).sum();
        if v_norm_sq == 0.0 {
            continue;
        }

        let scale = 2.0 / v_norm_sq;
        for dj in d[k..n].iter_mut() {
            *dj = 0.0;
        }
        for i in k..n {
            let vi = v[i];
            let row = &r[i * n + k..i * n + n];
            for (dj, &rij) in d[k..n].iter_mut().zip(row.iter()) {
                *dj += vi * rij;
            }
        }
        for (fj, &dj) in f_vec[k..n].iter_mut().zip(d[k..n].iter()) {
            *fj = scale * dj;
        }
        for i in k..n {
            let vi = v[i];
            let row = &mut r[i * n + k..i * n + n];
            for (rij, &fj) in row.iter_mut().zip(f_vec[k..n].iter()) {
                *rij -= fj * vi;
            }
        }

        for i in 0..n {
            let mut dot = 0.0;
            for j in k..n {
                dot += q[i * n + j] * v[j];
            }
            let factor = scale * dot;
            for j in k..n {
                q[i * n + j] -= factor * v[j];
            }
        }
    }

    (q, r)
}

fn bench_qr_exact_upper_triangular(c: &mut Criterion) {
    let n = 256usize;
    let a = generate_upper_triangular_matrix(n);
    let former = qr_upper_triangular_former(&a, n);
    let candidate = qr_nxn(&a, n).expect("upper-triangular QR candidate");
    for (name, lhs, rhs) in [
        ("Q", former.0.as_slice(), candidate.0.as_slice()),
        ("R", former.1.as_slice(), candidate.1.as_slice()),
    ] {
        assert_eq!(lhs.len(), rhs.len());
        for (index, (&old, &new)) in lhs.iter().zip(rhs).enumerate() {
            assert_eq!(
                old.to_bits(),
                new.to_bits(),
                "upper-triangular QR {name}[{index}] changed bits"
            );
        }
    }

    let mut group = c.benchmark_group("qr_exact_upper_triangular_256");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.bench_function("former_unblocked", |bench| {
        bench.iter(|| black_box(qr_upper_triangular_former(black_box(&a), n)));
    });
    group.bench_function("scalar_active_row", |bench| {
        bench.iter(|| black_box(qr_nxn(black_box(&a), n).unwrap()));
    });
    group.finish();
}

fn qr_upper_trapezoidal_former(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut q = vec![0.0; m * m];
    for i in 0..m {
        q[i * m + i] = 1.0;
    }
    let mut r = a.to_vec();
    let mut v = vec![0.0; m];

    for col in 0..m.min(n) {
        let mut col_norm_sq = 0.0;
        for i in col..m {
            col_norm_sq += r[i * n + col] * r[i * n + col];
        }
        let col_norm = col_norm_sq.sqrt();
        if col_norm == 0.0 {
            continue;
        }

        let sign = if r[col * n + col] >= 0.0 { 1.0 } else { -1.0 };
        for vi in &mut v[..col] {
            *vi = 0.0;
        }
        for (i, vi) in v[col..m].iter_mut().enumerate() {
            *vi = r[(i + col) * n + col];
        }
        v[col] += sign * col_norm;
        let v_norm_sq: f64 = v[col..].iter().map(|x| x * x).sum();
        if v_norm_sq == 0.0 {
            continue;
        }

        let scale = 2.0 / v_norm_sq;
        for j in col..n {
            let mut dot = 0.0;
            for i in col..m {
                dot += v[i] * r[i * n + j];
            }
            let factor = scale * dot;
            for i in col..m {
                r[i * n + j] -= factor * v[i];
            }
        }

        for i in 0..m {
            let mut dot = 0.0;
            for j in col..m {
                dot += q[i * m + j] * v[j];
            }
            let factor = scale * dot;
            for j in col..m {
                q[i * m + j] -= factor * v[j];
            }
        }
    }

    (q, r)
}

fn bench_qr_exact_upper_trapezoidal(c: &mut Criterion) {
    let (m, n) = (256usize, 128usize);
    let a = generate_upper_trapezoidal_matrix(m, n);
    let former = qr_upper_trapezoidal_former(&a, m, n);
    let candidate = qr_mxn(&a, m, n).expect("upper-trapezoidal QR candidate");
    for (name, lhs, rhs) in [
        ("Q", former.0.as_slice(), candidate.0.as_slice()),
        ("R", former.1.as_slice(), candidate.1.as_slice()),
    ] {
        assert_eq!(lhs.len(), rhs.len());
        for (index, (&old, &new)) in lhs.iter().zip(rhs).enumerate() {
            assert_eq!(
                old.to_bits(),
                new.to_bits(),
                "upper-trapezoidal QR {name}[{index}] changed bits"
            );
        }
    }

    let mut group = c.benchmark_group("qr_exact_upper_trapezoidal_256x128");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.bench_function("former_rectangular", |bench| {
        bench.iter(|| {
            black_box(qr_upper_trapezoidal_former(
                black_box(&a),
                black_box(m),
                black_box(n),
            ))
        });
    });
    group.bench_function("scalar_active_row", |bench| {
        bench.iter(|| {
            black_box(qr_mxn(black_box(&a), black_box(m), black_box(n)).unwrap())
        });
    });
    group.finish();
}

fn bench_svd(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd_nxn");

    for n in [16, 32, 64, 128, 256, 512] {
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

fn bench_svd_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd_mxn_full");

    for n in [128usize, 256, 512] {
        let a = generate_random_matrix(n, 456);

        group.bench_with_input(BenchmarkId::new("size", n), &n, |bench, &n| {
            bench.iter(|| {
                let result = svd_mxn_full(black_box(&a), black_box(n), black_box(n));
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_eigvalsh(c: &mut Criterion) {
    let mut group = c.benchmark_group("eigvalsh_nxn");

    for n in [16, 32, 64, 128, 256, 512, 1024] {
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

fn bench_eigvalsh_diagonal(c: &mut Criterion) {
    let mut group = c.benchmark_group("eigvalsh_diagonal_nxn");

    for n in [128usize, 256, 512] {
        let a = generate_descending_diagonal_matrix(n);

        group.bench_with_input(BenchmarkId::new("size", n), &n, |bench, &n| {
            bench.iter(|| {
                let result = eigvalsh_nxn(black_box(&a), black_box(n));
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_eigvalsh_tridiagonal(c: &mut Criterion) {
    let mut group = c.benchmark_group("eigvalsh_tridiagonal_nxn");

    for n in [128usize, 256, 512] {
        let a = generate_spd_tridiagonal_matrix(n);

        group.bench_with_input(BenchmarkId::new("size", n), &n, |bench, &n| {
            bench.iter(|| {
                let result = eigvalsh_nxn(black_box(&a), black_box(n));
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_sbr_stage1(c: &mut Criterion) {
    let mut group = c.benchmark_group("sbr_stage1_band_nxn");

    for n in [512usize, 1024] {
        let a = generate_spd_matrix(n);

        group.bench_with_input(BenchmarkId::new("size", n), &n, |bench, _| {
            bench.iter(|| {
                let result = sbr_stage1_dense_to_band_lower_nxn(black_box(&a), n);
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

fn bench_matrix_norm_orders(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_norm_nxn_orders");

    // Public np.linalg.norm-style orders that should be memory-bandwidth bound:
    // fro/inf scan rows, while 1/-1 used to stride columns through row-major data.
    for n in [128usize, 256, 512, 1024] {
        let a = generate_random_matrix(n, 0x4E4F_524D_4F52_4445);

        group.bench_with_input(BenchmarkId::new("one", n), &n, |bench, &n| {
            bench.iter(|| {
                let result = matrix_norm_nxn(black_box(&a), black_box(n), black_box(n), "1");
                black_box(result)
            });
        });
        group.bench_with_input(BenchmarkId::new("neg_one", n), &n, |bench, &n| {
            bench.iter(|| {
                let result = matrix_norm_nxn(black_box(&a), black_box(n), black_box(n), "-1");
                black_box(result)
            });
        });
        group.bench_with_input(BenchmarkId::new("fro", n), &n, |bench, &n| {
            bench.iter(|| {
                let result = matrix_norm_nxn(black_box(&a), black_box(n), black_box(n), "fro");
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_cond(c: &mut Criterion) {
    let mut group = c.benchmark_group("cond_nxn");

    // cond(A) is the main public linalg consumer that needs only ordered
    // singular values from the SVD pipeline, not U/Vt reconstruction.
    for n in [64usize, 128, 256, 512] {
        let a = generate_invertible_matrix(n);

        group.bench_with_input(BenchmarkId::new("size", n), &n, |bench, &n| {
            bench.iter(|| {
                let result = cond_nxn(black_box(&a), black_box(n));
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
    for (batch, n) in [
        (8192usize, 8usize),
        (2048, 16),
        (512, 32),
        (256, 48),
        (64, 128),
        (16, 256),
    ] {
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

    for (batch, n) in [
        (2000usize, 16usize),
        (1000, 32),
        (500, 64),
        (64, 128),
        (16, 256),
    ] {
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

fn bench_complex_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_matmul");

    // Interleaved complex n*n*n GEMM (each operand is 2*n*n reals); all dims >=
    // the parallel threshold so the rayon row-partition path runs.
    let gen_interleaved = |len: usize, seed: u64| -> Vec<f64> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
            })
            .collect()
    };
    for n in [128usize, 256, 512] {
        let a = gen_interleaved(2 * n * n, 0xCAFE_F00D_1234_5678);
        let b = gen_interleaved(2 * n * n, 0xDEAD_BEEF_9876_5432);
        group.bench_with_input(BenchmarkId::new("size", n), &n, |bench, &n| {
            bench.iter(|| {
                let result = complex_matmul(black_box(&a), black_box(&b), n, n, n);
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_multi_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_dot");

    // Two-matrix multi_dot dispatches straight to the rectangular GEMM
    // (mat_mul_rect). Square n*n*n inputs keep all three dims >= the parallel
    // threshold so the rayon row-partition path runs.
    for n in [128usize, 256, 512] {
        let a = generate_random_matrix(n, 0x1234_5678_9ABC_DEF0);
        let b = generate_random_matrix(n, 0x0FED_CBA9_8765_4321);
        group.bench_with_input(BenchmarkId::new("size", n), &n, |bench, &n| {
            bench.iter(|| {
                let result = multi_dot(black_box(&[(a.as_slice(), n, n), (b.as_slice(), n, n)]));
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_matrix_power(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_power_nxn");

    // matrix_power(A, 3) performs three full n*n*n GEMMs via the internal
    // mat_mul_flat kernel — the compute-bound regime (n >= 128) where
    // intra-matrix row parallelism dominates the per-call dispatch cost.
    for n in [128usize, 256, 512, 1024] {
        let a = generate_random_matrix(n, 0x9E37_79B9_7F4A_7C15);
        group.bench_with_input(BenchmarkId::new("size", n), &n, |bench, &n| {
            bench.iter(|| {
                let result = matrix_power_nxn(black_box(&a), black_box(n), black_box(3));
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_kron(c: &mut Criterion) {
    let mut group = c.benchmark_group("kron_nxn");
    let n = 64usize;
    let p = 4usize;
    let a = generate_random_matrix(n, 0x4b52_4f4e);
    let mut b = vec![0.0f64; p * p];
    for (i, row) in b.chunks_mut(p).enumerate() {
        if let Some(slot) = row.get_mut(i) {
            *slot = 1.0;
        }
    }

    group.bench_function("kron_64x64_4x4_eye", |bench| {
        bench.iter(|| {
            let result = kron_nxn(black_box(&a), n, n, black_box(&b), p, p);
            black_box(result)
        });
    });

    let large_n = 128usize;
    let a_nonnegative: Vec<f64> = generate_random_matrix(large_n, 0x4b52_4f4e_4641_5354)
        .into_iter()
        .map(f64::abs)
        .collect();
    group.bench_function("kron_128x128_4x4_eye_nonnegative_fast_path", |bench| {
        bench.iter(|| {
            let result = kron_nxn(
                black_box(&a_nonnegative),
                large_n,
                large_n,
                black_box(&b),
                p,
                p,
            );
            black_box(result)
        });
    });

    group.finish();
}

fn bench_batch_trace(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_trace");

    for (batch, n) in [(4096usize, 8usize), (1024, 32)] {
        let mat_size = n * n;
        let data: Vec<f64> = (0..batch * mat_size)
            .map(|idx| ((idx % 251) as f64 - 125.0) * 0.125)
            .collect();
        let shape = [batch, n, n];
        let id = format!("{batch}x{n}x{n}");
        group.bench_with_input(BenchmarkId::new("shape", id), &shape, |bench, shape| {
            bench.iter(|| {
                let result = batch_trace(black_box(&data), black_box(shape));
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_batch_det_slogdet(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_det_slogdet");

    // Real np.linalg.det/slogdet workloads often stack many small independent
    // matrices. These rows isolate allocator/scratch overhead from cubic work.
    for (batch, n) in [(8192usize, 4usize), (2048, 8)] {
        let (data, shape) = generate_batch_invertible(batch, n);
        let id = format!("{batch}x{n}x{n}");
        group.bench_with_input(BenchmarkId::new("det", id.clone()), &shape, |bench, shape| {
            bench.iter(|| {
                let result = batch_det(black_box(&data), black_box(shape));
                black_box(result)
            });
        });
        group.bench_with_input(BenchmarkId::new("slogdet", id), &shape, |bench, shape| {
            bench.iter(|| {
                let result = batch_slogdet(black_box(&data), black_box(shape));
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_batch_matrix_norm_fro(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_matrix_norm_fro");

    for (batch, m, n) in [(4096usize, 8usize, 8usize), (1024, 32, 32)] {
        let mat_size = m * n;
        let data: Vec<f64> = (0..batch * mat_size)
            .map(|idx| ((idx % 251) as f64 - 125.0) * 0.125)
            .collect();
        let shape = [batch, m, n];
        let id = format!("{batch}x{m}x{n}");
        group.bench_with_input(BenchmarkId::new("shape", id), &shape, |bench, shape| {
            bench.iter(|| {
                let result = batch_matrix_norm(black_box(&data), black_box(shape), black_box("fro"));
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_batch_matrix_norm_row_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_matrix_norm_row_sum");

    for ord in ["inf", "-inf"] {
        for (batch, m, n) in [(4096usize, 8usize, 8usize), (1024, 32, 32)] {
            let mat_size = m * n;
            let data: Vec<f64> = (0..batch * mat_size)
                .map(|idx| ((idx % 251) as f64 - 125.0) * 0.125)
                .collect();
            let shape = [batch, m, n];
            let id = format!("{ord}_{batch}x{m}x{n}");
            group.bench_with_input(BenchmarkId::new("shape", id), &shape, |bench, shape| {
                bench.iter(|| {
                    let result =
                        batch_matrix_norm(black_box(&data), black_box(shape), black_box(ord));
                    black_box(result)
                });
            });
        }
    }

    group.finish();
}

fn bench_batch_matrix_norm_column_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_matrix_norm_column_sum");

    for ord in ["1", "-1"] {
        for (batch, m, n) in [(4096usize, 8usize, 8usize), (1024, 32, 32)] {
            let mat_size = m * n;
            let data: Vec<f64> = (0..batch * mat_size)
                .map(|idx| ((idx % 251) as f64 - 125.0) * 0.125)
                .collect();
            let shape = [batch, m, n];
            let id = format!("{ord}_{batch}x{m}x{n}");
            group.bench_with_input(BenchmarkId::new("shape", id), &shape, |bench, shape| {
                bench.iter(|| {
                    let result =
                        batch_matrix_norm(black_box(&data), black_box(shape), black_box(ord));
                    black_box(result)
                });
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_complex_matmul,
    bench_multi_dot,
    bench_matrix_power,
    bench_kron,
    bench_solve,
    bench_det,
    bench_det_exact_upper_triangular,
    bench_det_exact_lower_triangular_no_pivot,
    bench_slogdet_exact_upper_triangular,
    bench_slogdet_exact_lower_triangular_no_pivot,
    bench_inv,
    bench_cholesky,
    bench_cholesky_exact_diagonal,
    bench_cholesky_exact_tridiagonal,
    bench_qr,
    bench_qr_exact_upper_triangular,
    bench_qr_exact_upper_trapezoidal,
    bench_svd,
    bench_svd_full,
    bench_eigvalsh,
    bench_eigvalsh_diagonal,
    bench_eigvalsh_tridiagonal,
    bench_sbr_stage1,
    bench_norm_frobenius,
    bench_matrix_norm_orders,
    bench_cond,
    bench_batch_inv,
    bench_batch_eigvalsh,
    bench_batch_cholesky,
    bench_batch_trace,
    bench_batch_det_slogdet,
    bench_batch_matrix_norm_fro,
    bench_batch_matrix_norm_row_sum,
    bench_batch_matrix_norm_column_sum,
);

criterion_main!(benches);
