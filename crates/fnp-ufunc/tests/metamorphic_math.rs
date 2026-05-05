//! Metamorphic tests for mathematical operations.
//!
//! These tests verify mathematical invariants that MUST hold regardless of
//! specific input values. When an oracle (expected output) is unavailable,
//! metamorphic relations between inputs/outputs provide correctness evidence.

use fnp_ufunc::{UFuncArray, UnaryOp, BinaryOp};
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;
const TRIG_EPSILON: f64 = 1e-14;

fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps || (a.is_nan() && b.is_nan())
}

fn arr_approx_eq(a: &UFuncArray, b: &UFuncArray, eps: f64) -> bool {
    let a_data = a.values();
    let b_data = b.values();
    a_data.len() == b_data.len() && a_data.iter().zip(b_data.iter()).all(|(x, y)| approx_eq(*x, *y, eps))
}

fn scalar_approx_eq(a: &UFuncArray, expected: f64, eps: f64) -> bool {
    let data = a.values();
    data.len() == 1 && approx_eq(data[0], expected, eps)
}

fn from_vec(v: Vec<f64>) -> UFuncArray {
    UFuncArray::from_vec(v)
}

// ─────────────────────────────────────────────────────────────────────────────
// MR Category 1: Pythagorean Identity (sin²x + cos²x = 1)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_pythagorean_identity_single_values() {
    let test_values = [0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0, PI, 1.5, 2.7, -0.5];

    for x in test_values {
        let arr = from_vec(vec![x]);
        let sin_x = arr.elementwise_unary(UnaryOp::Sin);
        let cos_x = arr.elementwise_unary(UnaryOp::Cos);
        let sin_sq = sin_x.elementwise_unary(UnaryOp::Square);
        let cos_sq = cos_x.elementwise_unary(UnaryOp::Square);
        let sum = sin_sq.elementwise_binary(&cos_sq, BinaryOp::Add).unwrap();

        assert!(
            scalar_approx_eq(&sum, 1.0, TRIG_EPSILON),
            "sin²({x}) + cos²({x}) should equal 1, got {:?}",
            sum.values()
        );
    }
}

#[test]
fn mr_pythagorean_identity_array() {
    let values: Vec<f64> = (0..100).map(|i| i as f64 * 0.1 - 5.0).collect();
    let arr = from_vec(values);

    let sin_x = arr.elementwise_unary(UnaryOp::Sin);
    let cos_x = arr.elementwise_unary(UnaryOp::Cos);
    let sin_sq = sin_x.elementwise_unary(UnaryOp::Square);
    let cos_sq = cos_x.elementwise_unary(UnaryOp::Square);
    let sum = sin_sq.elementwise_binary(&cos_sq, BinaryOp::Add).unwrap();

    let ones = from_vec(vec![1.0; 100]);
    assert!(
        arr_approx_eq(&sum, &ones, TRIG_EPSILON),
        "sin²(x) + cos²(x) should equal 1 for all elements"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// MR Category 2: Odd/Even Function Properties
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_sin_odd_function() {
    let values: Vec<f64> = (1..50).map(|i| i as f64 * 0.1).collect();
    let pos = from_vec(values.clone());
    let neg = from_vec(values.iter().map(|x| -x).collect());

    let sin_pos = pos.elementwise_unary(UnaryOp::Sin);
    let sin_neg = neg.elementwise_unary(UnaryOp::Sin);
    let neg_sin_pos = sin_pos.elementwise_unary(UnaryOp::Negative);

    assert!(
        arr_approx_eq(&sin_neg, &neg_sin_pos, TRIG_EPSILON),
        "sin(-x) should equal -sin(x)"
    );
}

#[test]
fn mr_cos_even_function() {
    let values: Vec<f64> = (1..50).map(|i| i as f64 * 0.1).collect();
    let pos = from_vec(values.clone());
    let neg = from_vec(values.iter().map(|x| -x).collect());

    let cos_pos = pos.elementwise_unary(UnaryOp::Cos);
    let cos_neg = neg.elementwise_unary(UnaryOp::Cos);

    assert!(
        arr_approx_eq(&cos_pos, &cos_neg, TRIG_EPSILON),
        "cos(-x) should equal cos(x)"
    );
}

#[test]
fn mr_sinh_odd_function() {
    let values: Vec<f64> = (1..30).map(|i| i as f64 * 0.1).collect();
    let pos = from_vec(values.clone());
    let neg = from_vec(values.iter().map(|x| -x).collect());

    let sinh_pos = pos.elementwise_unary(UnaryOp::Sinh);
    let sinh_neg = neg.elementwise_unary(UnaryOp::Sinh);
    let neg_sinh_pos = sinh_pos.elementwise_unary(UnaryOp::Negative);

    assert!(
        arr_approx_eq(&sinh_neg, &neg_sinh_pos, EPSILON),
        "sinh(-x) should equal -sinh(x)"
    );
}

#[test]
fn mr_cosh_even_function() {
    let values: Vec<f64> = (1..30).map(|i| i as f64 * 0.1).collect();
    let pos = from_vec(values.clone());
    let neg = from_vec(values.iter().map(|x| -x).collect());

    let cosh_pos = pos.elementwise_unary(UnaryOp::Cosh);
    let cosh_neg = neg.elementwise_unary(UnaryOp::Cosh);

    assert!(
        arr_approx_eq(&cosh_pos, &cosh_neg, EPSILON),
        "cosh(-x) should equal cosh(x)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// MR Category 3: Hyperbolic Identity (cosh²x - sinh²x = 1)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_hyperbolic_identity() {
    let values: Vec<f64> = (-20..=20).map(|i| i as f64 * 0.25).collect();
    let arr = from_vec(values);

    let sinh_x = arr.elementwise_unary(UnaryOp::Sinh);
    let cosh_x = arr.elementwise_unary(UnaryOp::Cosh);
    let sinh_sq = sinh_x.elementwise_unary(UnaryOp::Square);
    let cosh_sq = cosh_x.elementwise_unary(UnaryOp::Square);
    let diff = cosh_sq.elementwise_binary(&sinh_sq, BinaryOp::Sub).unwrap();

    let ones = from_vec(vec![1.0; 41]);
    assert!(
        arr_approx_eq(&diff, &ones, EPSILON),
        "cosh²(x) - sinh²(x) should equal 1"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// MR Category 4: Inverse Relations (exp/log, degrees/radians)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_exp_log_inverse() {
    let values: Vec<f64> = (1..50).map(|i| i as f64 * 0.1).collect();
    let original = from_vec(values.clone());

    let exp_x = original.elementwise_unary(UnaryOp::Exp);
    let log_exp_x = exp_x.elementwise_unary(UnaryOp::Log);

    assert!(
        arr_approx_eq(&log_exp_x, &original, EPSILON),
        "log(exp(x)) should equal x"
    );
}

#[test]
fn mr_log_exp_inverse() {
    let values: Vec<f64> = (1..50).map(|i| i as f64 * 0.1 + 0.01).collect();
    let original = from_vec(values.clone());

    let log_x = original.elementwise_unary(UnaryOp::Log);
    let exp_log_x = log_x.elementwise_unary(UnaryOp::Exp);

    assert!(
        arr_approx_eq(&exp_log_x, &original, EPSILON),
        "exp(log(x)) should equal x for x > 0"
    );
}

#[test]
fn mr_degrees_radians_inverse() {
    let degrees: Vec<f64> = (0..=360).step_by(15).map(|d| d as f64).collect();
    let original = from_vec(degrees);

    let radians = original.elementwise_unary(UnaryOp::Radians);
    let back_to_degrees = radians.elementwise_unary(UnaryOp::Degrees);

    assert!(
        arr_approx_eq(&back_to_degrees, &original, EPSILON),
        "degrees(radians(x)) should equal x"
    );
}

#[test]
fn mr_radians_degrees_inverse() {
    let radians: Vec<f64> = (0..20).map(|i| i as f64 * PI / 10.0).collect();
    let original = from_vec(radians);

    let degrees = original.elementwise_unary(UnaryOp::Degrees);
    let back_to_radians = degrees.elementwise_unary(UnaryOp::Radians);

    assert!(
        arr_approx_eq(&back_to_radians, &original, EPSILON),
        "radians(degrees(x)) should equal x"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// MR Category 5: Additive Properties
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_exp_additive_property() {
    let a_vals: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
    let b_vals: Vec<f64> = (0..20).map(|i| (20 - i) as f64 * 0.05).collect();

    let a = from_vec(a_vals.clone());
    let b = from_vec(b_vals.clone());

    let exp_a = a.elementwise_unary(UnaryOp::Exp);
    let exp_b = b.elementwise_unary(UnaryOp::Exp);
    let exp_a_times_exp_b = exp_a.elementwise_binary(&exp_b, BinaryOp::Mul).unwrap();

    let a_plus_b = a.elementwise_binary(&b, BinaryOp::Add).unwrap();
    let exp_a_plus_b = a_plus_b.elementwise_unary(UnaryOp::Exp);

    assert!(
        arr_approx_eq(&exp_a_times_exp_b, &exp_a_plus_b, EPSILON),
        "exp(a) * exp(b) should equal exp(a+b)"
    );
}

#[test]
fn mr_log_multiplicative_property() {
    let a_vals: Vec<f64> = (1..20).map(|i| i as f64 * 0.5 + 0.1).collect();
    let b_vals: Vec<f64> = (1..20).map(|i| (20 - i) as f64 * 0.3 + 0.1).collect();

    let a = from_vec(a_vals);
    let b = from_vec(b_vals);

    let log_a = a.elementwise_unary(UnaryOp::Log);
    let log_b = b.elementwise_unary(UnaryOp::Log);
    let log_a_plus_log_b = log_a.elementwise_binary(&log_b, BinaryOp::Add).unwrap();

    let a_times_b = a.elementwise_binary(&b, BinaryOp::Mul).unwrap();
    let log_a_times_b = a_times_b.elementwise_unary(UnaryOp::Log);

    assert!(
        arr_approx_eq(&log_a_plus_log_b, &log_a_times_b, EPSILON),
        "log(a) + log(b) should equal log(a*b)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// MR Category 6: Reduction Permutation Invariance
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_sum_permutation_invariant() {
    let original = vec![1.5, 2.7, 3.1, 4.9, 5.2, 6.8, 7.3, 8.1, 9.4, 10.6];
    let permuted = vec![5.2, 1.5, 9.4, 3.1, 7.3, 2.7, 10.6, 4.9, 8.1, 6.8];

    let arr_orig = from_vec(original);
    let arr_perm = from_vec(permuted);

    let sum_orig = arr_orig.reduce_sum(None, false).unwrap();
    let sum_perm = arr_perm.reduce_sum(None, false).unwrap();

    assert!(
        arr_approx_eq(&sum_orig, &sum_perm, EPSILON),
        "sum(permute(x)) should equal sum(x)"
    );
}

#[test]
fn mr_prod_permutation_invariant() {
    let original = vec![1.1, 1.2, 1.3, 1.4, 1.5];
    let permuted = vec![1.4, 1.1, 1.5, 1.2, 1.3];

    let arr_orig = from_vec(original);
    let arr_perm = from_vec(permuted);

    let prod_orig = arr_orig.reduce_prod(None, false).unwrap();
    let prod_perm = arr_perm.reduce_prod(None, false).unwrap();

    assert!(
        arr_approx_eq(&prod_orig, &prod_perm, EPSILON),
        "prod(permute(x)) should equal prod(x)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// MR Category 7: Idempotent Operations
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_abs_idempotent() {
    let values: Vec<f64> = (-10..=10).map(|i| i as f64 * 0.7).collect();
    let arr = from_vec(values);

    let abs_once = arr.elementwise_unary(UnaryOp::Abs);
    let abs_twice = abs_once.elementwise_unary(UnaryOp::Abs);

    assert!(
        arr_approx_eq(&abs_once, &abs_twice, EPSILON),
        "abs(abs(x)) should equal abs(x)"
    );
}

#[test]
fn mr_floor_idempotent() {
    let values: Vec<f64> = (-10..=10).map(|i| i as f64 * 0.7 + 0.3).collect();
    let arr = from_vec(values);

    let floor_once = arr.elementwise_unary(UnaryOp::Floor);
    let floor_twice = floor_once.elementwise_unary(UnaryOp::Floor);

    assert!(
        arr_approx_eq(&floor_once, &floor_twice, EPSILON),
        "floor(floor(x)) should equal floor(x)"
    );
}

#[test]
fn mr_ceil_idempotent() {
    let values: Vec<f64> = (-10..=10).map(|i| i as f64 * 0.7 + 0.3).collect();
    let arr = from_vec(values);

    let ceil_once = arr.elementwise_unary(UnaryOp::Ceil);
    let ceil_twice = ceil_once.elementwise_unary(UnaryOp::Ceil);

    assert!(
        arr_approx_eq(&ceil_once, &ceil_twice, EPSILON),
        "ceil(ceil(x)) should equal ceil(x)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// MR Category 8: Double Negation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_double_negation() {
    let values: Vec<f64> = (-20..=20).map(|i| i as f64 * 0.5).collect();
    let original = from_vec(values);

    let neg_once = original.elementwise_unary(UnaryOp::Negative);
    let neg_twice = neg_once.elementwise_unary(UnaryOp::Negative);

    assert!(
        arr_approx_eq(&neg_twice, &original, EPSILON),
        "-(-x) should equal x"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// MR Category 9: Sqrt/Square Partial Inverse
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_sqrt_square_inverse_positive() {
    let values: Vec<f64> = (1..=50).map(|i| i as f64 * 0.1).collect();
    let original = from_vec(values);

    let sqrt_x = original.elementwise_unary(UnaryOp::Sqrt);
    let sqrt_sq = sqrt_x.elementwise_unary(UnaryOp::Square);

    assert!(
        arr_approx_eq(&sqrt_sq, &original, EPSILON),
        "square(sqrt(x)) should equal x for x >= 0"
    );
}

#[test]
fn mr_square_sqrt_abs() {
    let values: Vec<f64> = (-25..=25).map(|i| i as f64 * 0.2).collect();
    let arr = from_vec(values);

    let squared = arr.elementwise_unary(UnaryOp::Square);
    let sqrt_squared = squared.elementwise_unary(UnaryOp::Sqrt);
    let abs_original = arr.elementwise_unary(UnaryOp::Abs);

    assert!(
        arr_approx_eq(&sqrt_squared, &abs_original, EPSILON),
        "sqrt(x²) should equal |x|"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// MR Category 10: Cbrt/Cube Inverse
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_cbrt_cube_inverse() {
    let values: Vec<f64> = (-20..=20).map(|i| i as f64 * 0.3).collect();
    let original = from_vec(values);

    let cbrt_x = original.elementwise_unary(UnaryOp::Cbrt);
    let cubed = cbrt_x.elementwise_binary(&cbrt_x, BinaryOp::Mul).unwrap();
    let cubed = cubed.elementwise_binary(&cbrt_x, BinaryOp::Mul).unwrap();

    assert!(
        arr_approx_eq(&cubed, &original, 1e-9),
        "cbrt(x)³ should equal x"
    );
}
