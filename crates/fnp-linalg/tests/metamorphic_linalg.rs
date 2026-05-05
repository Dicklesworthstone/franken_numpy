//! Metamorphic relation tests for fnp-linalg.
//!
//! Tests mathematical invariants that must hold for correct linear algebra
//! implementations. These catch bugs without needing oracle values.
//!
//! Key relations tested:
//! - Inverse: inv(inv(A)) = A, A @ inv(A) = I
//! - Determinant: det(inv(A)) = 1/det(A), det(A@B) = det(A)*det(B)
//! - Solve: solve(A, A@x) = x
//! - QR: Q @ R = A, Q^T @ Q = I
//! - SVD: U @ S @ V^T = A
//! - Cholesky: L @ L^T = A
//! - Trace: trace(A+B) = trace(A) + trace(B)
//! - Norm: ||A||_F^2 = trace(A^T @ A)

use fnp_linalg::{
    cholesky_nxn, det_2x2, det_nxn, inv_2x2, inv_nxn, matrix_norm_frobenius,
    qr_nxn, solve_2x2, solve_nxn, svd_nxn, trace_nxn,
};

const EPSILON: f64 = 1e-10;
const LOOSE_EPSILON: f64 = 1e-6;

fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() < eps
}

fn approx_eq_vec(a: &[f64], b: &[f64], eps: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y, eps))
}

fn mat_mul_2x2(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

fn mat_vec_2x2(a: [[f64; 2]; 2], v: [f64; 2]) -> [f64; 2] {
    [
        a[0][0] * v[0] + a[0][1] * v[1],
        a[1][0] * v[0] + a[1][1] * v[1],
    ]
}

fn mat_mul_nxn(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn mat_vec_nxn(a: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            result[i] += a[i * n + j] * v[j];
        }
    }
    result
}

fn identity_nxn(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

fn transpose_nxn(a: &[f64], n: usize) -> Vec<f64> {
    let mut t = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            t[j * n + i] = a[i * n + j];
        }
    }
    t
}

// ─────────────────────────────────────────────────────────────────────────────
// Inverse metamorphic relations
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_inv_inv_is_identity_2x2() {
    let matrices = [
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, -1.0], [2.0, 3.0]],
        [[0.5, 0.25], [0.125, 0.75]],
        [[10.0, 0.0], [0.0, 10.0]],
    ];

    for a in &matrices {
        let inv_a = inv_2x2(*a).unwrap();
        let inv_inv_a = inv_2x2(inv_a).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    approx_eq(a[i][j], inv_inv_a[i][j], LOOSE_EPSILON),
                    "inv(inv(A)) != A for matrix {:?}\ngot {:?}",
                    a,
                    inv_inv_a
                );
            }
        }
    }
}

#[test]
fn mr_a_times_inv_a_is_identity_2x2() {
    let matrices = [
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, -1.0], [2.0, 3.0]],
        [[1.5, 0.5], [0.5, 1.5]],
    ];

    let identity = [[1.0, 0.0], [0.0, 1.0]];

    for a in &matrices {
        let inv_a = inv_2x2(*a).unwrap();
        let product = mat_mul_2x2(*a, inv_a);

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    approx_eq(product[i][j], identity[i][j], LOOSE_EPSILON),
                    "A @ inv(A) != I for matrix {:?}\ngot {:?}",
                    a,
                    product
                );
            }
        }
    }
}

#[test]
fn mr_inv_inv_is_identity_nxn() {
    let matrices: Vec<(Vec<f64>, usize)> = vec![
        (vec![4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0], 3),
        (
            vec![
                5.0, 1.0, 0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 1.0, 2.0,
            ],
            4,
        ),
    ];

    for (a, n) in &matrices {
        let inv_a = inv_nxn(a, *n).unwrap();
        let inv_inv_a = inv_nxn(&inv_a, *n).unwrap();

        assert!(
            approx_eq_vec(a, &inv_inv_a, LOOSE_EPSILON),
            "inv(inv(A)) != A for {}x{} matrix",
            n,
            n
        );
    }
}

#[test]
fn mr_a_times_inv_a_is_identity_nxn() {
    let matrices: Vec<(Vec<f64>, usize)> = vec![
        (vec![4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0], 3),
        (
            vec![
                2.0, 1.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0, 1.0, 5.0,
            ],
            4,
        ),
    ];

    for (a, n) in &matrices {
        let inv_a = inv_nxn(a, *n).unwrap();
        let product = mat_mul_nxn(a, &inv_a, *n);
        let identity = identity_nxn(*n);

        assert!(
            approx_eq_vec(&product, &identity, LOOSE_EPSILON),
            "A @ inv(A) != I for {}x{} matrix",
            n,
            n
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Determinant metamorphic relations
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_det_inv_is_reciprocal_2x2() {
    let matrices = [
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, -1.0], [2.0, 3.0]],
        [[2.0, 0.0], [0.0, 3.0]],
    ];

    for a in &matrices {
        let det_a = det_2x2(*a).unwrap();
        let inv_a = inv_2x2(*a).unwrap();
        let det_inv_a = det_2x2(inv_a).unwrap();

        assert!(
            approx_eq(det_inv_a, 1.0 / det_a, LOOSE_EPSILON),
            "det(inv(A)) != 1/det(A) for {:?}\ndet(A)={}, det(inv(A))={}",
            a,
            det_a,
            det_inv_a
        );
    }
}

#[test]
fn mr_det_product_is_product_det_2x2() {
    let pairs = [
        ([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]),
        ([[2.0, 0.0], [0.0, 3.0]], [[1.0, 1.0], [1.0, 2.0]]),
    ];

    for (a, b) in &pairs {
        let det_a = det_2x2(*a).unwrap();
        let det_b = det_2x2(*b).unwrap();
        let product = mat_mul_2x2(*a, *b);
        let det_product = det_2x2(product).unwrap();

        assert!(
            approx_eq(det_product, det_a * det_b, LOOSE_EPSILON),
            "det(A@B) != det(A)*det(B)\ndet(A@B)={}, det(A)*det(B)={}",
            det_product,
            det_a * det_b
        );
    }
}

#[test]
fn mr_det_inv_is_reciprocal_nxn() {
    let matrices: Vec<(Vec<f64>, usize)> = vec![
        (vec![4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0], 3),
    ];

    for (a, n) in &matrices {
        let det_a = det_nxn(a, *n).unwrap();
        let inv_a = inv_nxn(a, *n).unwrap();
        let det_inv_a = det_nxn(&inv_a, *n).unwrap();

        assert!(
            approx_eq(det_inv_a, 1.0 / det_a, LOOSE_EPSILON),
            "det(inv(A)) != 1/det(A) for {}x{}\ndet(A)={}, det(inv(A))={}",
            n,
            n,
            det_a,
            det_inv_a
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Solve metamorphic relations
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_solve_ax_equals_b_2x2() {
    let cases = [
        ([[1.0, 2.0], [3.0, 4.0]], [5.0, 11.0]),
        ([[2.0, 1.0], [1.0, 3.0]], [4.0, 5.0]),
        ([[5.0, -1.0], [2.0, 3.0]], [4.0, 7.0]),
    ];

    for (a, b) in &cases {
        let x = solve_2x2(*a, *b).unwrap();
        let ax = mat_vec_2x2(*a, x);

        assert!(
            approx_eq(ax[0], b[0], LOOSE_EPSILON) && approx_eq(ax[1], b[1], LOOSE_EPSILON),
            "A @ solve(A, b) != b\nA={:?}, b={:?}, x={:?}, Ax={:?}",
            a,
            b,
            x,
            ax
        );
    }
}

#[test]
fn mr_solve_a_ax_recovers_x_nxn() {
    let matrices: Vec<(Vec<f64>, usize)> = vec![
        (vec![4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0], 3),
        (
            vec![
                5.0, 1.0, 0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 1.0, 2.0,
            ],
            4,
        ),
    ];

    for (a, n) in &matrices {
        let x_original: Vec<f64> = (1..=*n).map(|i| i as f64).collect();
        let b = mat_vec_nxn(a, &x_original, *n);
        let x_solved = solve_nxn(a, &b, *n).unwrap();

        assert!(
            approx_eq_vec(&x_original, &x_solved, LOOSE_EPSILON),
            "solve(A, A@x) != x for {}x{}\noriginal: {:?}\nsolved: {:?}",
            n,
            n,
            x_original,
            x_solved
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QR decomposition metamorphic relations
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_qr_reconstructs_original() {
    let matrices: Vec<(Vec<f64>, usize)> = vec![
        (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0], 3),
        (vec![4.0, 2.0, 2.0, 5.0, 1.0, 3.0, 6.0], 2),
    ];

    for (a, n) in &matrices {
        if a.len() != n * n {
            continue;
        }
        let (q, r) = qr_nxn(a, *n).unwrap();
        let qr = mat_mul_nxn(&q, &r, *n);

        assert!(
            approx_eq_vec(a, &qr, LOOSE_EPSILON),
            "Q @ R != A for {}x{}\nA: {:?}\nQR: {:?}",
            n,
            n,
            a,
            qr
        );
    }
}

#[test]
fn mr_q_is_orthogonal() {
    let matrices: Vec<(Vec<f64>, usize)> = vec![
        (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0], 3),
    ];

    for (a, n) in &matrices {
        let (q, _) = qr_nxn(a, *n).unwrap();
        let qt = transpose_nxn(&q, *n);
        let qtq = mat_mul_nxn(&qt, &q, *n);
        let identity = identity_nxn(*n);

        assert!(
            approx_eq_vec(&qtq, &identity, LOOSE_EPSILON),
            "Q^T @ Q != I for {}x{}\nQ^T @ Q: {:?}",
            n,
            n,
            qtq
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cholesky decomposition metamorphic relations
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_cholesky_reconstructs_original() {
    // Symmetric positive definite matrices
    let matrices: Vec<(Vec<f64>, usize)> = vec![
        (vec![4.0, 2.0, 2.0, 5.0], 2),
        (vec![4.0, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 4.0], 3),
    ];

    for (a, n) in &matrices {
        let l = cholesky_nxn(a, *n).unwrap();
        let lt = transpose_nxn(&l, *n);
        let llt = mat_mul_nxn(&l, &lt, *n);

        assert!(
            approx_eq_vec(a, &llt, LOOSE_EPSILON),
            "L @ L^T != A for {}x{}\nA: {:?}\nL@L^T: {:?}",
            n,
            n,
            a,
            llt
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SVD metamorphic relations
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_svd_singular_values_are_positive() {
    let matrices: Vec<(Vec<f64>, usize)> = vec![
        (vec![1.0, 2.0, 3.0, 4.0], 2),
        (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3),
    ];

    for (a, n) in &matrices {
        let s = svd_nxn(a, *n).unwrap();

        for (i, &val) in s.iter().enumerate() {
            assert!(
                val >= -EPSILON,
                "Singular value {} is negative: {} for {}x{}",
                i,
                val,
                n,
                n
            );
        }
    }
}

#[test]
fn mr_svd_singular_values_sorted_descending() {
    let matrices: Vec<(Vec<f64>, usize)> = vec![
        (vec![1.0, 2.0, 3.0, 4.0], 2),
        (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3),
    ];

    for (a, n) in &matrices {
        let s = svd_nxn(a, *n).unwrap();

        for i in 1..s.len() {
            assert!(
                s[i - 1] >= s[i] - EPSILON,
                "Singular values not sorted: s[{}]={} < s[{}]={}",
                i - 1,
                s[i - 1],
                i,
                s[i]
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Trace metamorphic relations
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_trace_is_additive() {
    let n = 3;
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let sum: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

    let trace_a = trace_nxn(&a, n).unwrap();
    let trace_b = trace_nxn(&b, n).unwrap();
    let trace_sum = trace_nxn(&sum, n).unwrap();

    assert!(
        approx_eq(trace_sum, trace_a + trace_b, EPSILON),
        "trace(A+B) != trace(A) + trace(B)\ntrace(A+B)={}, trace(A)+trace(B)={}",
        trace_sum,
        trace_a + trace_b
    );
}

#[test]
fn mr_trace_is_sum_of_diagonal() {
    let matrices: Vec<(Vec<f64>, usize)> = vec![
        (vec![1.0, 2.0, 3.0, 4.0], 2),
        (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3),
    ];

    for (a, n) in &matrices {
        let trace = trace_nxn(a, *n).unwrap();
        let diag_sum: f64 = (0..*n).map(|i| a[i * n + i]).sum();

        assert!(
            approx_eq(trace, diag_sum, EPSILON),
            "trace != sum of diagonal for {}x{}\ntrace={}, diag_sum={}",
            n,
            n,
            trace,
            diag_sum
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Norm metamorphic relations
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_frobenius_norm_squared_equals_trace_ata() {
    let matrices: Vec<(Vec<f64>, usize)> = vec![
        (vec![1.0, 2.0, 3.0, 4.0], 2),
        (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3),
    ];

    for (a, n) in &matrices {
        let norm_f = matrix_norm_frobenius(a, *n).unwrap();
        let at = transpose_nxn(a, *n);
        let ata = mat_mul_nxn(&at, a, *n);
        let trace_ata = trace_nxn(&ata, *n).unwrap();

        assert!(
            approx_eq(norm_f * norm_f, trace_ata, LOOSE_EPSILON),
            "||A||_F^2 != trace(A^T @ A)\n||A||_F^2={}, trace(A^T A)={}",
            norm_f * norm_f,
            trace_ata
        );
    }
}

#[test]
fn mr_frobenius_norm_is_sqrt_sum_squares() {
    let matrices: Vec<(Vec<f64>, usize)> = vec![
        (vec![1.0, 2.0, 3.0, 4.0], 2),
        (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3),
    ];

    for (a, n) in &matrices {
        let norm_f = matrix_norm_frobenius(a, *n).unwrap();
        let sum_sq: f64 = a.iter().map(|x| x * x).sum();
        let expected = sum_sq.sqrt();

        assert!(
            approx_eq(norm_f, expected, EPSILON),
            "||A||_F != sqrt(sum of squares)\n||A||_F={}, sqrt(sum)={}",
            norm_f,
            expected
        );
    }
}

#[test]
fn mr_norm_is_homogeneous() {
    let n = 3;
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let c = 2.5;
    let ca: Vec<f64> = a.iter().map(|x| c * x).collect();

    let norm_a = matrix_norm_frobenius(&a, n).unwrap();
    let norm_ca = matrix_norm_frobenius(&ca, n).unwrap();

    assert!(
        approx_eq(norm_ca, c.abs() * norm_a, EPSILON),
        "||cA|| != |c| * ||A||\n||cA||={}, |c|*||A||={}",
        norm_ca,
        c.abs() * norm_a
    );
}
