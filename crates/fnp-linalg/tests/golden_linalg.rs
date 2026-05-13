//! Golden artifact tests for fnp-linalg native implementations.
//!
//! These tests verify linear algebra operations against pre-computed expected
//! values, enabling regression testing without requiring numpy oracle.

use fnp_linalg::{
    cholesky_nxn, det_2x2, det_nxn, eigvalsh_nxn, inv_2x2, inv_nxn, qr_nxn, solve_2x2, solve_nxn,
    svd_nxn,
};

const EPSILON: f64 = 1e-10;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPSILON || (a.is_nan() && b.is_nan())
}

fn vec_approx_eq(a: &[f64], b: &[f64], eps: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < eps)
}

// ─────────────────────────────────────────────────────────────────────────────
// 2x2 Determinant
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_det_2x2_identity() {
    let matrix = [[1.0, 0.0], [0.0, 1.0]];
    let det = det_2x2(matrix).unwrap();
    assert!(approx_eq(det, 1.0), "det(I) should be 1, got {det}");
}

#[test]
fn golden_det_2x2_simple() {
    let matrix = [[3.0, 8.0], [4.0, 6.0]];
    let det = det_2x2(matrix).unwrap();
    let expected = 3.0 * 6.0 - 8.0 * 4.0;
    assert!(
        approx_eq(det, expected),
        "det should be {expected}, got {det}"
    );
}

#[test]
fn golden_det_2x2_singular() {
    let matrix = [[1.0, 2.0], [2.0, 4.0]];
    let det = det_2x2(matrix).unwrap();
    assert!(
        approx_eq(det, 0.0),
        "singular matrix det should be 0, got {det}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// NxN Determinant
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_det_3x3() {
    let matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0];
    let det = det_nxn(&matrix, 3).unwrap();
    let expected = -3.0;
    assert!(
        approx_eq(det, expected),
        "det should be {expected}, got {det}"
    );
}

#[test]
fn golden_det_4x4() {
    let matrix = [
        1.0, 0.0, 2.0, -1.0, 3.0, 0.0, 0.0, 5.0, 2.0, 1.0, 4.0, -3.0, 1.0, 0.0, 5.0, 0.0,
    ];
    let det = det_nxn(&matrix, 4).unwrap();
    let expected = 30.0;
    assert!(
        (det - expected).abs() < 1e-8,
        "det should be {expected}, got {det}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 2x2 Inverse
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_inv_2x2_identity() {
    let matrix = [[1.0, 0.0], [0.0, 1.0]];
    let inv = inv_2x2(matrix).unwrap();
    assert!(approx_eq(inv[0][0], 1.0) && approx_eq(inv[0][1], 0.0));
    assert!(approx_eq(inv[1][0], 0.0) && approx_eq(inv[1][1], 1.0));
}

#[test]
fn golden_inv_2x2_simple() {
    let matrix = [[4.0, 7.0], [2.0, 6.0]];
    let inv = inv_2x2(matrix).unwrap();
    let expected = [[0.6, -0.7], [-0.2, 0.4]];
    assert!(approx_eq(inv[0][0], expected[0][0]));
    assert!(approx_eq(inv[0][1], expected[0][1]));
    assert!(approx_eq(inv[1][0], expected[1][0]));
    assert!(approx_eq(inv[1][1], expected[1][1]));
}

// ─────────────────────────────────────────────────────────────────────────────
// NxN Inverse
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_inv_3x3() {
    let matrix = [1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0];
    let inv = inv_nxn(&matrix, 3).unwrap();
    let expected = [-24.0, 18.0, 5.0, 20.0, -15.0, -4.0, -5.0, 4.0, 1.0];
    assert!(vec_approx_eq(&inv, &expected, 1e-8), "inv mismatch");
}

// ─────────────────────────────────────────────────────────────────────────────
// 2x2 Solve
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_solve_2x2_simple() {
    let lhs = [[2.0, 1.0], [5.0, 7.0]];
    let rhs = [11.0, 13.0];
    let x = solve_2x2(lhs, rhs).unwrap();
    let ax0 = lhs[0][0] * x[0] + lhs[0][1] * x[1];
    let ax1 = lhs[1][0] * x[0] + lhs[1][1] * x[1];
    assert!(
        approx_eq(ax0, rhs[0]),
        "Ax[0] should be {}, got {ax0}",
        rhs[0]
    );
    assert!(
        approx_eq(ax1, rhs[1]),
        "Ax[1] should be {}, got {ax1}",
        rhs[1]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// NxN Solve
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_solve_3x3() {
    let a = [3.0, 1.0, -1.0, 2.0, 4.0, 1.0, -1.0, 2.0, 5.0];
    let b = [4.0, 1.0, 1.0];
    let x = solve_nxn(&a, &b, 3).unwrap();
    let mut ax = vec![0.0; 3];
    for i in 0..3 {
        for j in 0..3 {
            ax[i] += a[i * 3 + j] * x[j];
        }
    }
    assert!(vec_approx_eq(&ax, &b, 1e-8), "Ax should equal b");
}

// ─────────────────────────────────────────────────────────────────────────────
// Cholesky
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_cholesky_3x3() {
    let a = [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
    let l = cholesky_nxn(&a, 3).unwrap();
    let expected = [2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0];
    assert!(vec_approx_eq(&l, &expected, 1e-8), "Cholesky L mismatch");
}

#[test]
fn golden_cholesky_identity() {
    let a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let l = cholesky_nxn(&a, 3).unwrap();
    assert!(vec_approx_eq(&l, &a, 1e-10), "Cholesky of I should be I");
}

// ─────────────────────────────────────────────────────────────────────────────
// QR Decomposition
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_qr_3x3_orthogonality() {
    let a = [12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0];
    let (q, _r) = qr_nxn(&a, 3).unwrap();
    let mut qtq = vec![0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                qtq[i * 3 + j] += q[k * 3 + i] * q[k * 3 + j];
            }
        }
    }
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    assert!(
        vec_approx_eq(&qtq, &identity, 1e-8),
        "Q should be orthogonal (Q^T Q = I)"
    );
}

#[test]
fn golden_qr_3x3_reconstruction() {
    let a = [12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0];
    let (q, r) = qr_nxn(&a, 3).unwrap();
    let mut qr = vec![0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                qr[i * 3 + j] += q[i * 3 + k] * r[k * 3 + j];
            }
        }
    }
    assert!(vec_approx_eq(&qr, &a, 1e-8), "QR should reconstruct A");
}

// ─────────────────────────────────────────────────────────────────────────────
// SVD Singular Values
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_svd_3x3_identity() {
    let a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let s = svd_nxn(&a, 3).unwrap();
    let expected = [1.0, 1.0, 1.0];
    assert!(
        vec_approx_eq(&s, &expected, 1e-8),
        "SVD of I should have singular values [1,1,1]"
    );
}

#[test]
fn golden_svd_3x3_diagonal() {
    let a = [3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0];
    let mut s = svd_nxn(&a, 3).unwrap();
    s.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let expected = [3.0, 2.0, 1.0];
    assert!(
        vec_approx_eq(&s, &expected, 1e-8),
        "SVD of diag([3,2,1]) should be [3,2,1]"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Eigenvalues (symmetric)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_eigvalsh_2x2() {
    let a = [2.0, 1.0, 1.0, 2.0];
    let mut eig = eigvalsh_nxn(&a, 2).unwrap();
    eig.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let expected = [1.0, 3.0];
    assert!(
        vec_approx_eq(&eig, &expected, 1e-8),
        "eigenvalues should be [1, 3]"
    );
}

#[test]
fn golden_eigvalsh_3x3_identity() {
    let a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let eig = eigvalsh_nxn(&a, 3).unwrap();
    let expected = [1.0, 1.0, 1.0];
    assert!(
        vec_approx_eq(&eig, &expected, 1e-8),
        "eigenvalues of I should be [1,1,1]"
    );
}

#[test]
fn golden_eigvalsh_3x3_symmetric() {
    let a = [2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0];
    let mut eig = eigvalsh_nxn(&a, 3).unwrap();
    eig.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sqrt2 = std::f64::consts::SQRT_2;
    let expected = [2.0 - sqrt2, 2.0, 2.0 + sqrt2];
    assert!(vec_approx_eq(&eig, &expected, 1e-8), "eigenvalues mismatch");
}
