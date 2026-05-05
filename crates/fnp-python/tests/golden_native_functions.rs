//! Golden artifact tests for native Rust implementations.
//!
//! These tests verify exact numerical output against hardcoded expected values,
//! catching regressions in native implementations without requiring numpy.
//! Values were captured from verified-correct implementations.

use fnp_dtype::DType;
use fnp_ufunc::{UFuncArray, UnaryOp};

const EPSILON: f64 = 1e-12;

fn assert_close(actual: f64, expected: f64, name: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff < EPSILON,
        "{name}: expected {expected}, got {actual}, diff {diff}"
    );
}

fn assert_vec_close(actual: &[f64], expected: &[f64], name: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{name}: length mismatch {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff < EPSILON,
            "{name}[{i}]: expected {e}, got {a}, diff {diff}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Window functions
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_bartlett_8() {
    let result = UFuncArray::bartlett(8);
    let expected = [
        0.0,
        0.2857142857142857,
        0.5714285714285714,
        0.8571428571428571,
        0.8571428571428571,
        0.5714285714285714,
        0.2857142857142857,
        0.0,
    ];
    assert_vec_close(result.values(), &expected, "bartlett(8)");
}

#[test]
fn golden_hanning_8() {
    let result = UFuncArray::hanning(8);
    let expected = [
        0.0,
        0.1882550990706332,
        0.6112604669781572,
        0.9504844339512095,
        0.9504844339512095,
        0.6112604669781572,
        0.1882550990706332,
        0.0,
    ];
    assert_vec_close(result.values(), &expected, "hanning(8)");
}

#[test]
fn golden_hamming_8() {
    let result = UFuncArray::hamming(8);
    let expected = [
        0.08,
        0.25319469114498255,
        0.6423596296199047,
        0.9544456792351128,
        0.9544456792351128,
        0.6423596296199047,
        0.25319469114498255,
        0.08,
    ];
    assert_vec_close(result.values(), &expected, "hamming(8)");
}

#[test]
fn golden_blackman_8() {
    let result = UFuncArray::blackman(8);
    let expected = [
        0.0,
        0.09045342435412804,
        0.4591829575459636,
        0.9203636180999081,
        0.9203636180999081,
        0.4591829575459636,
        0.09045342435412804,
        0.0,
    ];
    assert_vec_close(result.values(), &expected, "blackman(8)");
}

#[test]
fn golden_kaiser_8_beta14() {
    let result = UFuncArray::kaiser(8, 14.0);
    // Golden values from our native implementation (uses bessel_i0 polynomial approx)
    let expected = [
        7.726866835270368e-06,
        0.017964074282044187,
        0.27277209015009414,
        0.8708037664339706,
        0.8708037664339706,
        0.27277209015009414,
        0.017964074282044187,
        7.726866835270368e-06,
    ];
    // Kaiser uses bessel_i0 polynomial approximation - allow 1e-7 tolerance
    for (i, (&a, &e)) in result.values().iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff < 1e-7,
            "kaiser(8,14)[{i}]: expected {e}, got {a}, diff {diff}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bessel function i0
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_i0_values() {
    let input = UFuncArray::new(vec![6], vec![0.0, 0.5, 1.0, 2.0, 5.0, 10.0], DType::F64).unwrap();
    let result = input.i0();
    let expected = [
        1.0,                    // i0(0) = 1 exactly
        1.0634833707413234,     // i0(0.5)
        1.2660658480342601,     // i0(1) - our polynomial approx
        2.2795853023360673,     // i0(2)
        27.239871823604442,     // i0(5)
        2815.7166284662544,     // i0(10)
    ];
    for (i, (&a, &e)) in result.values().iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let rel_diff = diff / e.abs().max(1e-10);
        assert!(
            rel_diff < 1e-6,
            "i0[{i}]: expected {e}, got {a}, rel_diff {rel_diff}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sinc function
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_sinc_values() {
    // numpy.sinc uses normalized sinc: sinc(x) = sin(pi*x)/(pi*x)
    let input = UFuncArray::new(
        vec![7],
        vec![0.0, 0.5, 1.0, -1.0, 2.0, 0.25, std::f64::consts::PI],
        DType::F64,
    )
    .unwrap();
    let result = input.sinc();
    let expected = [
        1.0,                                // sinc(0) = 1 by definition
        0.6366197723675814,                 // sinc(0.5) = 2/pi
        3.898171832519376e-17,              // sinc(1) ≈ 0
        3.898171832519376e-17,              // sinc(-1) ≈ 0
        -3.898171832519376e-17,             // sinc(2) ≈ 0
        0.9003163161571061,                 // sinc(0.25)
        -0.04359862862918773,               // sinc(pi) from our impl
    ];
    for (i, (&a, &e)) in result.values().iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        // For values near zero, use absolute tolerance
        let tol = if e.abs() < 1e-10 { 1e-10 } else { e.abs() * 1e-8 };
        assert!(
            diff < tol,
            "sinc[{i}]: expected {e}, got {a}, diff {diff}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Trig functions via UnaryOp
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_sin_cos_tan() {
    let pi = std::f64::consts::PI;
    let input = UFuncArray::new(vec![4], vec![0.0, pi / 6.0, pi / 4.0, pi / 3.0], DType::F64).unwrap();

    let sin_result = input.elementwise_unary(UnaryOp::Sin);
    let cos_result = input.elementwise_unary(UnaryOp::Cos);
    let tan_result = input.elementwise_unary(UnaryOp::Tan);

    let sin_expected = [0.0, 0.5, 0.7071067811865476, 0.8660254037844386];
    let cos_expected = [1.0, 0.8660254037844387, 0.7071067811865476, 0.5000000000000001];
    let tan_expected = [0.0, 0.5773502691896257, 0.9999999999999999, 1.7320508075688767];

    assert_vec_close(sin_result.values(), &sin_expected, "sin");
    assert_vec_close(cos_result.values(), &cos_expected, "cos");
    assert_vec_close(tan_result.values(), &tan_expected, "tan");
}

#[test]
fn golden_exp_log() {
    let input = UFuncArray::new(vec![5], vec![0.0, 1.0, 2.0, std::f64::consts::E, 10.0], DType::F64).unwrap();

    let exp_result = input.elementwise_unary(UnaryOp::Exp);
    let log_result = input.elementwise_unary(UnaryOp::Log);

    let exp_expected = [
        1.0,
        std::f64::consts::E,
        7.38905609893065,
        15.154262241479259,
        22026.465794806718,
    ];
    let log_expected = [
        f64::NEG_INFINITY,
        0.0,
        0.6931471805599453,
        1.0,
        2.302585092994046,
    ];

    assert_vec_close(exp_result.values(), &exp_expected, "exp");
    // log(0) is -inf, handle separately
    assert!(log_result.values()[0].is_infinite() && log_result.values()[0].is_sign_negative());
    assert_vec_close(&log_result.values()[1..], &log_expected[1..], "log");
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_window_edge_cases() {
    // M=0 should return empty array
    assert!(UFuncArray::bartlett(0).values().is_empty());
    assert!(UFuncArray::hanning(0).values().is_empty());
    assert!(UFuncArray::hamming(0).values().is_empty());
    assert!(UFuncArray::blackman(0).values().is_empty());
    assert!(UFuncArray::kaiser(0, 1.0).values().is_empty());

    // M=1 should return [1.0]
    assert_eq!(UFuncArray::bartlett(1).values(), &[1.0]);
    assert_eq!(UFuncArray::hanning(1).values(), &[1.0]);
    assert_eq!(UFuncArray::hamming(1).values(), &[1.0]);
    assert_eq!(UFuncArray::blackman(1).values(), &[1.0]);
    assert_eq!(UFuncArray::kaiser(1, 1.0).values(), &[1.0]);
}

#[test]
fn golden_i0_edge_cases() {
    // i0(0) = 1 exactly
    let zero = UFuncArray::new(vec![1], vec![0.0], DType::F64).unwrap();
    assert_eq!(zero.i0().values()[0], 1.0);

    // i0 is even: i0(-x) = i0(x)
    let pos = UFuncArray::new(vec![1], vec![3.5], DType::F64).unwrap();
    let neg = UFuncArray::new(vec![1], vec![-3.5], DType::F64).unwrap();
    assert_eq!(pos.i0().values()[0], neg.i0().values()[0]);
}
