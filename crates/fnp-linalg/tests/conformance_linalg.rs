//! NumPy oracle conformance tests for fnp-linalg.
//!
//! These tests verify that our linear algebra implementations produce results
//! that match NumPy's np.linalg module within floating-point tolerance.

use fnp_linalg::{
    cholesky_nxn, det_2x2, det_nxn, inv_2x2, inv_nxn, qr_mxn, qr_nxn, slogdet_2x2, slogdet_nxn,
    solve_2x2, solve_nxn, svd_mxn_full,
};
use std::process::Command;

const NUMPY_LINALG_ORACLE: &str = r#"
import numpy as np
import sys

case = sys.argv[1]

def emit_vec(name, arr):
    flat = arr.ravel(order='C')
    values = ",".join(f"{float(v):.17g}" for v in flat)
    print(f"{name}={values}")

def emit_scalar(name, val):
    print(f"{name}={float(val):.17g}")

# Test matrices
A_2x2 = np.array([[4.0, 7.0], [2.0, 6.0]])
b_2 = np.array([1.0, 2.0])

A_3x3 = np.array([[6.0, 1.0, 1.0], [4.0, -2.0, 5.0], [2.0, 8.0, 7.0]])
b_3 = np.array([1.0, 2.0, 3.0])

A_4x4 = np.array([
    [4.0, 2.0, 3.0, 1.0],
    [2.0, 5.0, 1.0, 2.0],
    [3.0, 1.0, 6.0, 3.0],
    [1.0, 2.0, 3.0, 4.0]
])
b_4 = np.array([1.0, 2.0, 3.0, 4.0])

# Symmetric positive definite for Cholesky
SPD_3x3 = np.array([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]])
SPD_4x4 = np.array([
    [18.0, 22.0,  54.0,  42.0],
    [22.0, 70.0,  86.0,  62.0],
    [54.0, 86.0, 174.0, 134.0],
    [42.0, 62.0, 134.0, 106.0]
])

# For SVD / QR
M_3x2 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
M_2x3 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

if case == "solve_2x2":
    x = np.linalg.solve(A_2x2, b_2)
    emit_vec("x", x)
elif case == "det_2x2":
    d = np.linalg.det(A_2x2)
    emit_scalar("det", d)
elif case == "slogdet_2x2":
    sign, logdet = np.linalg.slogdet(A_2x2)
    emit_scalar("sign", sign)
    emit_scalar("logdet", logdet)
elif case == "inv_2x2":
    inv = np.linalg.inv(A_2x2)
    emit_vec("inv", inv)
elif case == "solve_3x3":
    x = np.linalg.solve(A_3x3, b_3)
    emit_vec("x", x)
elif case == "det_3x3":
    d = np.linalg.det(A_3x3)
    emit_scalar("det", d)
elif case == "slogdet_3x3":
    sign, logdet = np.linalg.slogdet(A_3x3)
    emit_scalar("sign", sign)
    emit_scalar("logdet", logdet)
elif case == "inv_3x3":
    inv = np.linalg.inv(A_3x3)
    emit_vec("inv", inv)
elif case == "solve_4x4":
    x = np.linalg.solve(A_4x4, b_4)
    emit_vec("x", x)
elif case == "det_4x4":
    d = np.linalg.det(A_4x4)
    emit_scalar("det", d)
elif case == "inv_4x4":
    inv = np.linalg.inv(A_4x4)
    emit_vec("inv", inv)
elif case == "cholesky_3x3":
    L = np.linalg.cholesky(SPD_3x3)
    emit_vec("L", L)
elif case == "cholesky_4x4":
    L = np.linalg.cholesky(SPD_4x4)
    emit_vec("L", L)
elif case == "qr_3x3":
    Q, R = np.linalg.qr(A_3x3)
    emit_vec("Q", Q)
    emit_vec("R", R)
elif case == "qr_3x2":
    Q, R = np.linalg.qr(M_3x2)
    emit_vec("Q", Q)
    emit_vec("R", R)
elif case == "svd_3x2":
    U, s, Vh = np.linalg.svd(M_3x2, full_matrices=True)
    emit_vec("U", U)
    emit_vec("s", s)
    emit_vec("Vh", Vh)
elif case == "svd_2x3":
    U, s, Vh = np.linalg.svd(M_2x3, full_matrices=True)
    emit_vec("U", U)
    emit_vec("s", s)
    emit_vec("Vh", Vh)
else:
    raise AssertionError(f"unknown case {case}")
"#;

fn run_numpy_oracle(case: &str) -> std::collections::HashMap<String, Vec<f64>> {
    let output = Command::new("python3")
        .args(["-c", NUMPY_LINALG_ORACLE, case])
        .output()
        .expect("python3 should be available");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("NumPy oracle failed for case '{case}': {stderr}");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut result = std::collections::HashMap::new();

    for line in stdout.lines() {
        if let Some((key, values_str)) = line.split_once('=') {
            let values: Vec<f64> = values_str
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|s| s.parse().expect("valid f64"))
                .collect();
            result.insert(key.to_string(), values);
        }
    }

    result
}

fn assert_vec_close(name: &str, got: &[f64], expected: &[f64], rtol: f64, atol: f64) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{name}: length mismatch: got {}, expected {}",
        got.len(),
        expected.len()
    );

    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        let tol = atol + rtol * e.abs();
        assert!(
            diff <= tol || (g.is_nan() && e.is_nan()),
            "{name}[{i}]: got {g}, expected {e}, diff {diff} > tol {tol}"
        );
    }
}

fn assert_scalar_close(name: &str, got: f64, expected: f64, rtol: f64, atol: f64) {
    let diff = (got - expected).abs();
    let tol = atol + rtol * expected.abs();
    assert!(
        diff <= tol || (got.is_nan() && expected.is_nan()),
        "{name}: got {got}, expected {expected}, diff {diff} > tol {tol}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 2x2 operations
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn conformance_solve_2x2() {
    let a = [[4.0, 7.0], [2.0, 6.0]];
    let b = [1.0, 2.0];
    let x = solve_2x2(a, b).expect("solve_2x2");

    let oracle = run_numpy_oracle("solve_2x2");
    let expected = oracle.get("x").expect("oracle x");
    assert_vec_close("solve_2x2", &x, expected, 1e-10, 1e-12);
}

#[test]
fn conformance_det_2x2() {
    let a = [[4.0, 7.0], [2.0, 6.0]];
    let det = det_2x2(a).expect("det_2x2");

    let oracle = run_numpy_oracle("det_2x2");
    let expected = oracle.get("det").expect("oracle det")[0];
    assert_scalar_close("det_2x2", det, expected, 1e-10, 1e-12);
}

#[test]
fn conformance_slogdet_2x2() {
    let a = [[4.0, 7.0], [2.0, 6.0]];
    let (sign, logdet) = slogdet_2x2(a).expect("slogdet_2x2");

    let oracle = run_numpy_oracle("slogdet_2x2");
    let expected_sign = oracle.get("sign").expect("oracle sign")[0];
    let expected_logdet = oracle.get("logdet").expect("oracle logdet")[0];
    assert_scalar_close("slogdet_2x2 sign", sign, expected_sign, 1e-10, 1e-12);
    assert_scalar_close("slogdet_2x2 logdet", logdet, expected_logdet, 1e-10, 1e-12);
}

#[test]
fn conformance_inv_2x2() {
    let a = [[4.0, 7.0], [2.0, 6.0]];
    let inv = inv_2x2(a).expect("inv_2x2");
    let inv_flat: Vec<f64> = inv.iter().flat_map(|row| row.iter().copied()).collect();

    let oracle = run_numpy_oracle("inv_2x2");
    let expected = oracle.get("inv").expect("oracle inv");
    assert_vec_close("inv_2x2", &inv_flat, expected, 1e-10, 1e-12);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3x3 operations
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn conformance_solve_3x3() {
    let a: Vec<f64> = vec![6.0, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0];
    let b: Vec<f64> = vec![1.0, 2.0, 3.0];
    let x = solve_nxn(&a, &b, 3).expect("solve_3x3");

    let oracle = run_numpy_oracle("solve_3x3");
    let expected = oracle.get("x").expect("oracle x");
    assert_vec_close("solve_3x3", &x, expected, 1e-10, 1e-12);
}

#[test]
fn conformance_det_3x3() {
    let a: Vec<f64> = vec![6.0, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0];
    let det = det_nxn(&a, 3).expect("det_3x3");

    let oracle = run_numpy_oracle("det_3x3");
    let expected = oracle.get("det").expect("oracle det")[0];
    assert_scalar_close("det_3x3", det, expected, 1e-10, 1e-12);
}

#[test]
fn conformance_slogdet_3x3() {
    let a: Vec<f64> = vec![6.0, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0];
    let (sign, logdet) = slogdet_nxn(&a, 3).expect("slogdet_3x3");

    let oracle = run_numpy_oracle("slogdet_3x3");
    let expected_sign = oracle.get("sign").expect("oracle sign")[0];
    let expected_logdet = oracle.get("logdet").expect("oracle logdet")[0];
    assert_scalar_close("slogdet_3x3 sign", sign, expected_sign, 1e-10, 1e-12);
    assert_scalar_close("slogdet_3x3 logdet", logdet, expected_logdet, 1e-10, 1e-12);
}

#[test]
fn conformance_inv_3x3() {
    let a: Vec<f64> = vec![6.0, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0];
    let inv = inv_nxn(&a, 3).expect("inv_3x3");

    let oracle = run_numpy_oracle("inv_3x3");
    let expected = oracle.get("inv").expect("oracle inv");
    assert_vec_close("inv_3x3", &inv, expected, 1e-10, 1e-12);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4x4 operations
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn conformance_solve_4x4() {
    let a: Vec<f64> = vec![
        4.0, 2.0, 3.0, 1.0, 2.0, 5.0, 1.0, 2.0, 3.0, 1.0, 6.0, 3.0, 1.0, 2.0, 3.0, 4.0,
    ];
    let b: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let x = solve_nxn(&a, &b, 4).expect("solve_4x4");

    let oracle = run_numpy_oracle("solve_4x4");
    let expected = oracle.get("x").expect("oracle x");
    assert_vec_close("solve_4x4", &x, expected, 1e-10, 1e-12);
}

#[test]
fn conformance_det_4x4() {
    let a: Vec<f64> = vec![
        4.0, 2.0, 3.0, 1.0, 2.0, 5.0, 1.0, 2.0, 3.0, 1.0, 6.0, 3.0, 1.0, 2.0, 3.0, 4.0,
    ];
    let det = det_nxn(&a, 4).expect("det_4x4");

    let oracle = run_numpy_oracle("det_4x4");
    let expected = oracle.get("det").expect("oracle det")[0];
    assert_scalar_close("det_4x4", det, expected, 1e-9, 1e-10);
}

#[test]
fn conformance_inv_4x4() {
    let a: Vec<f64> = vec![
        4.0, 2.0, 3.0, 1.0, 2.0, 5.0, 1.0, 2.0, 3.0, 1.0, 6.0, 3.0, 1.0, 2.0, 3.0, 4.0,
    ];
    let inv = inv_nxn(&a, 4).expect("inv_4x4");

    let oracle = run_numpy_oracle("inv_4x4");
    let expected = oracle.get("inv").expect("oracle inv");
    assert_vec_close("inv_4x4", &inv, expected, 1e-9, 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// Cholesky decomposition
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn conformance_cholesky_3x3() {
    let spd: Vec<f64> = vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
    let l = cholesky_nxn(&spd, 3).expect("cholesky_3x3");

    let oracle = run_numpy_oracle("cholesky_3x3");
    let expected = oracle.get("L").expect("oracle L");
    assert_vec_close("cholesky_3x3", &l, expected, 1e-10, 1e-12);
}

#[test]
fn conformance_cholesky_4x4() {
    let spd: Vec<f64> = vec![
        18.0, 22.0, 54.0, 42.0, 22.0, 70.0, 86.0, 62.0, 54.0, 86.0, 174.0, 134.0, 42.0, 62.0,
        134.0, 106.0,
    ];
    let l = cholesky_nxn(&spd, 4).expect("cholesky_4x4");

    let oracle = run_numpy_oracle("cholesky_4x4");
    let expected = oracle.get("L").expect("oracle L");
    assert_vec_close("cholesky_4x4", &l, expected, 1e-10, 1e-12);
}

// ─────────────────────────────────────────────────────────────────────────────
// QR decomposition
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn conformance_qr_3x3() {
    let a: Vec<f64> = vec![6.0, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0];
    let (q, r) = qr_nxn(&a, 3).expect("qr_3x3");

    let oracle = run_numpy_oracle("qr_3x3");
    let _expected_q = oracle.get("Q").expect("oracle Q");
    let _expected_r = oracle.get("R").expect("oracle R");

    // QR can have sign ambiguity in columns, so check Q*R = A instead
    let mut qr_product = vec![0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                qr_product[i * 3 + j] += q[i * 3 + k] * r[k * 3 + j];
            }
        }
    }
    assert_vec_close("qr_3x3 Q*R", &qr_product, &a, 1e-10, 1e-12);

    // Also verify R is upper triangular (zeros below diagonal)
    for i in 0..3 {
        for j in 0..i {
            assert!(
                r[i * 3 + j].abs() < 1e-10,
                "R should be upper triangular: R[{i},{j}] = {}",
                r[i * 3 + j]
            );
        }
    }
}

#[test]
fn conformance_qr_3x2() {
    let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (q, r) = qr_mxn(&a, 3, 2).expect("qr_3x2");

    // Our qr_mxn returns Q as m×m (full) and R as m×n
    let m = 3;
    let n = 2;
    assert_eq!(q.len(), m * m, "Q should be 3x3 = 9 elements (full)");
    assert_eq!(r.len(), m * n, "R should be 3x2 = 6 elements");

    // Verify Q*R = A
    let mut qr_product = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            for k in 0..m {
                qr_product[i * n + j] += q[i * m + k] * r[k * n + j];
            }
        }
    }
    assert_vec_close("qr_3x2 Q*R", &qr_product, &a, 1e-10, 1e-12);

    // Verify Q is orthogonal: Q^T * Q = I
    let mut qtq = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            for k in 0..m {
                qtq[i * m + j] += q[k * m + i] * q[k * m + j];
            }
        }
    }
    let identity: Vec<f64> = (0..m * m)
        .map(|idx| if idx / m == idx % m { 1.0 } else { 0.0 })
        .collect();
    assert_vec_close("qr_3x2 Q^T*Q", &qtq, &identity, 1e-10, 1e-12);
}

// ─────────────────────────────────────────────────────────────────────────────
// SVD decomposition
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn conformance_svd_3x2() {
    let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (u, s, vt) = svd_mxn_full(&a, 3, 2).expect("svd_3x2");

    let oracle = run_numpy_oracle("svd_3x2");
    let expected_s = oracle.get("s").expect("oracle s");

    // Singular values should match (they're unique up to ordering)
    let mut got_s = s.clone();
    let mut exp_s = expected_s.clone();
    got_s.sort_by(|a, b| b.partial_cmp(a).unwrap());
    exp_s.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert_vec_close("svd_3x2 singular values", &got_s, &exp_s, 1e-10, 1e-12);

    // Verify U * diag(S) * Vt = A (reconstruction)
    // U is m×m (3×3), S has k=min(m,n)=2 entries, Vt is n×n (2×2)
    let m = 3;
    let n = 2;
    let k = s.len();
    let mut reconstructed = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            for l in 0..k {
                // U[i,l] * S[l] * Vt[l,j]
                // U is m×m row-major: U[i,l] = u[i*m + l]
                // Vt is n×n row-major: Vt[l,j] = vt[l*n + j]
                reconstructed[i * n + j] += u[i * m + l] * s[l] * vt[l * n + j];
            }
        }
    }
    assert_vec_close("svd_3x2 U*S*Vt", &reconstructed, &a, 1e-10, 1e-12);
}

#[test]
fn conformance_svd_2x3() {
    let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (u, s, vt) = svd_mxn_full(&a, 2, 3).expect("svd_2x3");

    let oracle = run_numpy_oracle("svd_2x3");
    let expected_s = oracle.get("s").expect("oracle s");

    // Singular values should match
    let mut got_s = s.clone();
    let mut exp_s = expected_s.clone();
    got_s.sort_by(|a, b| b.partial_cmp(a).unwrap());
    exp_s.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert_vec_close("svd_2x3 singular values", &got_s, &exp_s, 1e-10, 1e-12);

    // Verify reconstruction: U * diag(S) * Vt = A
    // U is m×m (2×2), S has k=min(m,n)=2 entries, Vt is n×n (3×3)
    let m = 2;
    let n = 3;
    let k = s.len();
    let mut reconstructed = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            for l in 0..k {
                // U[i,l] * S[l] * Vt[l,j]
                // U is m×m row-major: U[i,l] = u[i*m + l]
                // Vt is n×n row-major: Vt[l,j] = vt[l*n + j]
                reconstructed[i * n + j] += u[i * m + l] * s[l] * vt[l * n + j];
            }
        }
    }
    assert_vec_close("svd_2x3 U*S*Vt", &reconstructed, &a, 1e-10, 1e-12);
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge cases: Identity matrices
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn conformance_det_identity_2x2() {
    let identity = [[1.0, 0.0], [0.0, 1.0]];
    let det = det_2x2(identity).expect("det identity 2x2");
    assert_scalar_close("det identity 2x2", det, 1.0, 1e-14, 1e-14);
}

#[test]
fn conformance_det_identity_3x3() {
    let identity: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let det = det_nxn(&identity, 3).expect("det identity 3x3");
    assert_scalar_close("det identity 3x3", det, 1.0, 1e-14, 1e-14);
}

#[test]
fn conformance_inv_identity_2x2() {
    let identity = [[1.0, 0.0], [0.0, 1.0]];
    let inv = inv_2x2(identity).expect("inv identity 2x2");
    let inv_flat: Vec<f64> = inv.iter().flat_map(|row| row.iter().copied()).collect();
    let expected = vec![1.0, 0.0, 0.0, 1.0];
    assert_vec_close("inv identity 2x2", &inv_flat, &expected, 1e-14, 1e-14);
}

#[test]
fn conformance_inv_identity_3x3() {
    let identity: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let inv = inv_nxn(&identity, 3).expect("inv identity 3x3");
    assert_vec_close("inv identity 3x3", &inv, &identity, 1e-14, 1e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge cases: Negative determinant
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn conformance_det_negative_2x2() {
    // This matrix has det = 1*4 - 2*3 = -2
    let a = [[1.0, 2.0], [3.0, 4.0]];
    let det = det_2x2(a).expect("det negative 2x2");
    assert_scalar_close("det negative 2x2", det, -2.0, 1e-14, 1e-14);
}

#[test]
fn conformance_slogdet_negative_2x2() {
    // Matrix with negative determinant
    let a = [[1.0, 2.0], [3.0, 4.0]];
    let (sign, logdet) = slogdet_2x2(a).expect("slogdet negative 2x2");
    // det = -2, so sign = -1, logdet = ln(2) ≈ 0.693
    assert_scalar_close("slogdet negative 2x2 sign", sign, -1.0, 1e-14, 1e-14);
    assert_scalar_close("slogdet negative 2x2 logdet", logdet, 2.0_f64.ln(), 1e-10, 1e-12);
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge cases: Singular matrices (det = 0)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn conformance_det_singular_2x2() {
    // Linearly dependent rows: second row = 2 * first row
    let a = [[1.0, 2.0], [2.0, 4.0]];
    let det = det_2x2(a).expect("det singular 2x2");
    assert_scalar_close("det singular 2x2", det, 0.0, 1e-14, 1e-14);
}

#[test]
fn conformance_det_singular_3x3() {
    // Linearly dependent rows: third row = first + second
    let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 7.0, 9.0];
    let det = det_nxn(&a, 3).expect("det singular 3x3");
    assert!(det.abs() < 1e-10, "det singular 3x3 should be ~0, got {det}");
}

#[test]
fn conformance_inv_singular_2x2_returns_error() {
    let a = [[1.0, 2.0], [2.0, 4.0]];
    let result = inv_2x2(a);
    assert!(
        result.is_err(),
        "inv of singular matrix should return error"
    );
}

#[test]
fn conformance_solve_singular_2x2_returns_error() {
    let a = [[1.0, 2.0], [2.0, 4.0]];
    let b = [1.0, 2.0];
    let result = solve_2x2(a, b);
    assert!(
        result.is_err(),
        "solve with singular matrix should return error"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge cases: Diagonal matrices
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn conformance_det_diagonal_2x2() {
    let a = [[3.0, 0.0], [0.0, 5.0]];
    let det = det_2x2(a).expect("det diagonal 2x2");
    assert_scalar_close("det diagonal 2x2", det, 15.0, 1e-14, 1e-14);
}

#[test]
fn conformance_inv_diagonal_2x2() {
    let a = [[2.0, 0.0], [0.0, 4.0]];
    let inv = inv_2x2(a).expect("inv diagonal 2x2");
    let inv_flat: Vec<f64> = inv.iter().flat_map(|row| row.iter().copied()).collect();
    let expected = vec![0.5, 0.0, 0.0, 0.25];
    assert_vec_close("inv diagonal 2x2", &inv_flat, &expected, 1e-14, 1e-14);
}

#[test]
fn conformance_cholesky_identity_3x3() {
    let identity: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let l = cholesky_nxn(&identity, 3).expect("cholesky identity 3x3");
    // Cholesky of identity should be identity
    assert_vec_close("cholesky identity 3x3", &l, &identity, 1e-14, 1e-14);
}

#[test]
fn conformance_cholesky_diagonal_3x3() {
    // Diagonal SPD matrix: diag(4, 9, 16)
    let diag: Vec<f64> = vec![4.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 16.0];
    let l = cholesky_nxn(&diag, 3).expect("cholesky diagonal 3x3");
    // Cholesky of diagonal should be sqrt of diagonal
    let expected: Vec<f64> = vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
    assert_vec_close("cholesky diagonal 3x3", &l, &expected, 1e-14, 1e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge cases: Very small and large values
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn conformance_det_small_values_2x2() {
    let a = [[1e-100, 0.0], [0.0, 1e-100]];
    let det = det_2x2(a).expect("det small values 2x2");
    assert_scalar_close("det small values 2x2", det, 1e-200, 1e-10, 1e-210);
}

#[test]
fn conformance_det_large_values_2x2() {
    let a = [[1e50, 0.0], [0.0, 1e50]];
    let det = det_2x2(a).expect("det large values 2x2");
    assert_scalar_close("det large values 2x2", det, 1e100, 1e-10, 1e90);
}

#[test]
fn conformance_slogdet_large_values_2x2() {
    // For very large determinants, slogdet avoids overflow
    let a = [[1e150, 0.0], [0.0, 1e150]];
    let (sign, logdet) = slogdet_2x2(a).expect("slogdet large values 2x2");
    // det = 1e300, sign = 1, logdet = ln(1e300) = 300 * ln(10) ≈ 690.78
    assert_scalar_close("slogdet large values sign", sign, 1.0, 1e-14, 1e-14);
    let expected_logdet = 300.0 * 10.0_f64.ln();
    assert_scalar_close("slogdet large values logdet", logdet, expected_logdet, 1e-10, 1e-8);
}
