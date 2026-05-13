use fnp_linalg::{cholesky_nxn, cholesky_solve_multi};
use std::time::Instant;

fn matmul_axb(a: &[f64], x: &[f64], n: usize, m: usize) -> Vec<f64> {
    let mut result = vec![0.0; n * m];
    for row in 0..n {
        for col in 0..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[row * n + k] * x[k * m + col];
            }
            result[row * m + col] = sum;
        }
    }
    result
}

fn make_spd_matrix(n: usize) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let val = 1.0 / (1.0 + (i as f64 - j as f64).abs());
            a[i * n + j] = val;
            a[j * n + i] = val;
        }
        a[i * n + i] += n as f64;
    }
    a
}

#[test]
fn cholesky_solve_multi_correctness_identity() {
    let n = 10;
    let m = 5;
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        l[i * n + i] = 1.0;
    }
    let b: Vec<f64> = (0..(n * m)).map(|i| (i as f64) * 0.1).collect();

    let x = cholesky_solve_multi(&l, &b, n, m).unwrap();

    for i in 0..(n * m) {
        assert!(
            (x[i] - b[i]).abs() < 1e-12,
            "Identity solve failed at index {i}: got {} expected {}",
            x[i],
            b[i]
        );
    }
}

#[test]
fn cholesky_solve_multi_correctness_spd() {
    let n = 20;
    let m = 10;
    let a = make_spd_matrix(n);
    let l = cholesky_nxn(&a, n).expect("Cholesky decomposition should succeed");
    let b: Vec<f64> = (0..(n * m)).map(|i| ((i % 7) as f64) - 3.0).collect();

    let x = cholesky_solve_multi(&l, &b, n, m).unwrap();
    let ax = matmul_axb(&a, &x, n, m);

    for i in 0..(n * m) {
        assert!(
            (ax[i] - b[i]).abs() < 1e-9,
            "A*X != B at index {i}: got {} expected {}",
            ax[i],
            b[i]
        );
    }
}

#[test]
fn cholesky_solve_multi_correctness_single_rhs() {
    let a = [4.0, 2.0, 2.0, 5.0];
    let l = cholesky_nxn(&a, 2).unwrap();
    let b = [1.0, 2.0];

    let x = cholesky_solve_multi(&l, &b, 2, 1).unwrap();
    let ax = matmul_axb(&a, &x, 2, 1);

    for i in 0..2 {
        assert!((ax[i] - b[i]).abs() < 1e-12, "Single RHS: A*X != B at {i}");
    }
}

#[test]
fn cholesky_solve_multi_perf_baseline() {
    let n = 100;
    let m = 1000;
    let a = make_spd_matrix(n);
    let l = cholesky_nxn(&a, n).expect("Cholesky should succeed");
    let b = vec![1.0; n * m];

    let start = Instant::now();
    let x = cholesky_solve_multi(&l, &b, n, m).unwrap();
    let duration = start.elapsed();

    assert_eq!(x.len(), n * m, "Output should have correct size");
    let ax = matmul_axb(&a, &x, n, m);
    let max_err = ax
        .iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(max_err < 1e-8, "Solution error too large: {max_err}");

    eprintln!("cholesky_solve_multi (n={n}, m={m}) took: {duration:?}, max_err: {max_err:.2e}");
}
