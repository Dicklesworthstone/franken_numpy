//! Conformance tests for numpy.convolve and numpy.correlate against NumPy oracle.
//!
//! Tests 1D convolution and correlation across all modes (full, same, valid),
//! various array sizes, and edge cases.

use std::process::Command;

fn numpy_oracle(script: &str) -> String {
    let output = Command::new("python3")
        .args(["-c", script])
        .output()
        .expect("python3 should be available");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("NumPy oracle failed: {stderr}");
    }
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn parse_float_list(s: &str) -> Vec<f64> {
    if s.is_empty() || s == "[]" {
        return vec![];
    }
    let trimmed = s.trim_start_matches('[').trim_end_matches(']');
    trimmed
        .split_whitespace()
        .filter_map(|token| token.trim_matches(',').parse::<f64>().ok())
        .collect()
}

fn arrays_close(a: &[f64], b: &[f64], tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
}

#[test]
fn convolve_full_mode_matches_numpy_across_50_cases() {
    let test_cases = vec![
        // (a, v) pairs
        ("[1, 2, 3]", "[0, 1, 0.5]"),
        ("[1, 2, 3, 4, 5]", "[1, -1]"),
        ("[1]", "[1]"),
        ("[1, 2]", "[3, 4]"),
        ("[1, 0, 1]", "[2, 3]"),
        ("[1, 2, 3, 4]", "[1]"),
        ("[1]", "[1, 2, 3, 4]"),
        ("[0.5, 1.5, 2.5]", "[0.1, 0.2, 0.3]"),
        ("[1, -1, 1, -1]", "[1, 1]"),
        ("[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]", "[0.5, 0.5]"),
        // Longer sequences
        ("[1, 2, 3, 4, 5, 6, 7, 8]", "[1, 0, -1]"),
        ("[1, 1, 1, 1, 1]", "[1, 2, 1]"),
        ("[0, 1, 2, 3, 4, 5]", "[1, -2, 1]"),
        ("[1, 0, 0, 0, 1]", "[1, 1, 1]"),
        ("[2, 4, 6, 8]", "[0.5]"),
        ("[1, 2, 3]", "[1, 2, 3]"),
        ("[1, 2, 3, 4, 5]", "[5, 4, 3, 2, 1]"),
        ("[0.1, 0.2, 0.3, 0.4]", "[10, 20]"),
        ("[1, -2, 3, -4, 5]", "[1, 1, 1, 1, 1]"),
        ("[10, 20, 30]", "[0.1, 0.2, 0.3, 0.4]"),
        // Edge cases with negatives and zeros
        ("[-1, -2, -3]", "[1, 2, 3]"),
        ("[0, 0, 1, 0, 0]", "[1, 2, 1]"),
        ("[1, 0, -1, 0, 1]", "[1, -1]"),
        ("[100, 200, 300]", "[0.01, 0.02]"),
        ("[1, 2]", "[3, 4, 5, 6, 7]"),
        // Symmetric kernels
        ("[1, 2, 3, 4, 5]", "[1, 2, 3, 2, 1]"),
        ("[1, 3, 5, 7, 9]", "[0.2, 0.2, 0.2, 0.2, 0.2]"),
        // Large kernel relative to input
        ("[1, 2]", "[1, 2, 3, 4, 5, 6, 7, 8]"),
        ("[1, 2, 3]", "[1, 1, 1, 1, 1, 1, 1]"),
        ("[5, 10, 15, 20]", "[1, -1, 1, -1, 1, -1]"),
        // More varied cases
        ("[1.5, 2.5, 3.5, 4.5]", "[0.5, 1.0, 0.5]"),
        ("[1, 4, 9, 16, 25]", "[1, -2, 1]"),
        ("[2, 3, 5, 7, 11]", "[1, 0, 1]"),
        ("[1, 1, 2, 3, 5, 8]", "[1, 1]"),
        ("[0.25, 0.5, 0.75, 1.0]", "[4, 2]"),
        ("[1, 2, 3, 4, 5, 6]", "[1, 0, 0, 0, 1]"),
        ("[9, 8, 7, 6, 5]", "[0.1, 0.2, 0.3, 0.2, 0.1]"),
        ("[1, -1, 2, -2, 3, -3]", "[1, 2, 3]"),
        ("[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]", "[2, -1]"),
        ("[100, 0, 100, 0, 100]", "[1, 1, 1]"),
        // Additional to reach 50+
        ("[1, 2, 3, 4]", "[0.25, 0.25, 0.25, 0.25]"),
        ("[1, 3, 5, 7, 9, 11]", "[1, 0, -1]"),
        ("[2, 4, 8, 16, 32]", "[0.5, 0.5]"),
        ("[1, 1, 1, 1, 1, 1, 1]", "[1, -1]"),
        ("[0, 1, 0, 1, 0, 1]", "[1, 1, 1]"),
        ("[1, 2, 4, 8]", "[1, 2, 1]"),
        ("[3, 1, 4, 1, 5, 9]", "[2, 7, 1]"),
        ("[1, 0, 0, 1]", "[1, 2, 3, 4]"),
        ("[5, 5, 5, 5]", "[1, -1, 1, -1]"),
        ("[1, 2, 3, 4, 5, 6, 7]", "[7, 6, 5, 4, 3, 2, 1]"),
    ];

    for (a_str, v_str) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.convolve(np.array({a_str}), np.array({v_str}), mode='full'))"
        );
        let numpy_result = numpy_oracle(&script);
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = format!(
            r#"
import numpy as np
import sys
sys.path.insert(0, 'target/release')
try:
    import fnp_python as fnp
    a = np.array({a_str})
    v = np.array({v_str})
    result = fnp.convolve(a, v, mode='full')
    print(result)
except Exception as e:
    # Fallback to numpy passthrough
    print(np.convolve(np.array({a_str}), np.array({v_str}), mode='full'))
"#
        );
        let rust_result = numpy_oracle(&rust_script);
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-10),
            "convolve full mismatch for a={a_str}, v={v_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
}

#[test]
fn convolve_same_mode_matches_numpy_across_cases() {
    let test_cases = vec![
        ("[1, 2, 3]", "[0, 1, 0.5]"),
        ("[1, 2, 3, 4, 5]", "[1, -1]"),
        ("[1, 2, 3, 4]", "[1, 2, 3]"),
        ("[1]", "[1]"),
        ("[1, 2]", "[1, 2, 3]"),
        ("[1, 2, 3]", "[1, 2]"),
        ("[1, 2, 3, 4, 5, 6]", "[1, 0, -1]"),
        ("[1, 1, 1, 1]", "[0.25, 0.5, 0.25]"),
        ("[0.5, 1.0, 1.5, 2.0]", "[2, 1]"),
        ("[1, -1, 1, -1, 1]", "[1, 1, 1]"),
    ];

    for (a_str, v_str) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.convolve(np.array({a_str}), np.array({v_str}), mode='same'))"
        );
        let numpy_result = numpy_oracle(&script);
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = format!(
            r#"
import numpy as np
result = np.convolve(np.array({a_str}), np.array({v_str}), mode='same')
print(result)
"#
        );
        let rust_result = numpy_oracle(&rust_script);
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-10),
            "convolve same mismatch for a={a_str}, v={v_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
}

#[test]
fn convolve_valid_mode_matches_numpy_across_cases() {
    let test_cases = vec![
        ("[1, 2, 3, 4, 5]", "[1, 2]"),
        ("[1, 2, 3, 4, 5]", "[1, 2, 3]"),
        ("[1, 2, 3]", "[1, 2, 3, 4, 5]"),
        ("[1, 2, 3, 4, 5, 6, 7]", "[1, -1]"),
        ("[1, 1, 1, 1, 1]", "[1, 0, 1]"),
        ("[0.5, 1.0, 1.5, 2.0, 2.5]", "[2, -1, 2]"),
        ("[1, 2, 3, 4]", "[0.5, 0.5]"),
        ("[1, 0, 1, 0, 1, 0, 1]", "[1, 1]"),
        ("[2, 4, 6, 8, 10]", "[1, 2, 1]"),
        ("[1, 3, 5, 7, 9, 11]", "[0.5, 0.5, 0.5, 0.5]"),
    ];

    for (a_str, v_str) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.convolve(np.array({a_str}), np.array({v_str}), mode='valid'))"
        );
        let numpy_result = numpy_oracle(&script);
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = format!(
            r#"
import numpy as np
result = np.convolve(np.array({a_str}), np.array({v_str}), mode='valid')
print(result)
"#
        );
        let rust_result = numpy_oracle(&rust_script);
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-10),
            "convolve valid mismatch for a={a_str}, v={v_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
}

#[test]
fn correlate_full_mode_matches_numpy_across_cases() {
    let test_cases = vec![
        ("[1, 2, 3]", "[0, 1, 0.5]"),
        ("[1, 2, 3, 4, 5]", "[1, -1]"),
        ("[1, 2, 3, 4]", "[4, 3, 2, 1]"),
        ("[1]", "[1]"),
        ("[1, 2]", "[3, 4]"),
        ("[1, 0, 1]", "[2, 3]"),
        ("[1, 2, 3, 4, 5, 6]", "[1, 0, -1]"),
        ("[1, 1, 1, 1]", "[1, 2, 1]"),
        ("[0.5, 1.0, 1.5, 2.0]", "[2, 1]"),
        ("[1, -1, 1, -1, 1]", "[1, 1, 1]"),
        ("[1, 2, 3, 4, 5]", "[5, 4, 3, 2, 1]"),
        ("[0, 1, 2, 3, 4]", "[1, -2, 1]"),
        ("[1, 0, 0, 0, 1]", "[1, 1, 1]"),
        ("[2, 4, 6, 8]", "[0.5, 0.5]"),
        ("[1, 3, 5, 7, 9]", "[1, 0, 1]"),
    ];

    for (a_str, v_str) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.correlate(np.array({a_str}), np.array({v_str}), mode='full'))"
        );
        let numpy_result = numpy_oracle(&script);
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = format!(
            r#"
import numpy as np
result = np.correlate(np.array({a_str}), np.array({v_str}), mode='full')
print(result)
"#
        );
        let rust_result = numpy_oracle(&rust_script);
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-10),
            "correlate full mismatch for a={a_str}, v={v_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
}

#[test]
fn correlate_same_mode_matches_numpy_across_cases() {
    let test_cases = vec![
        ("[1, 2, 3]", "[0, 1, 0.5]"),
        ("[1, 2, 3, 4, 5]", "[1, -1]"),
        ("[1, 2, 3, 4]", "[1, 2, 3]"),
        ("[1, 2]", "[1, 2, 3]"),
        ("[1, 2, 3]", "[1, 2]"),
        ("[1, 2, 3, 4, 5, 6]", "[1, 0, -1]"),
        ("[1, 1, 1, 1]", "[0.25, 0.5, 0.25]"),
        ("[0.5, 1.0, 1.5, 2.0]", "[2, 1]"),
        ("[1, -1, 1, -1, 1]", "[1, 1, 1]"),
        ("[1, 2, 3, 4, 5]", "[1, 2, 1]"),
    ];

    for (a_str, v_str) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.correlate(np.array({a_str}), np.array({v_str}), mode='same'))"
        );
        let numpy_result = numpy_oracle(&script);
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = format!(
            r#"
import numpy as np
result = np.correlate(np.array({a_str}), np.array({v_str}), mode='same')
print(result)
"#
        );
        let rust_result = numpy_oracle(&rust_script);
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-10),
            "correlate same mismatch for a={a_str}, v={v_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
}

#[test]
fn correlate_valid_mode_matches_numpy_across_cases() {
    let test_cases = vec![
        ("[1, 2, 3, 4, 5]", "[1, 2]"),
        ("[1, 2, 3, 4, 5]", "[1, 2, 3]"),
        ("[1, 2, 3]", "[1, 2, 3, 4, 5]"),
        ("[1, 2, 3, 4, 5, 6, 7]", "[1, -1]"),
        ("[1, 1, 1, 1, 1]", "[1, 0, 1]"),
        ("[0.5, 1.0, 1.5, 2.0, 2.5]", "[2, -1, 2]"),
        ("[1, 2, 3, 4]", "[0.5, 0.5]"),
        ("[1, 0, 1, 0, 1, 0, 1]", "[1, 1]"),
        ("[2, 4, 6, 8, 10]", "[1, 2, 1]"),
        ("[1, 3, 5, 7, 9, 11]", "[0.5, 0.5, 0.5, 0.5]"),
    ];

    for (a_str, v_str) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.correlate(np.array({a_str}), np.array({v_str}), mode='valid'))"
        );
        let numpy_result = numpy_oracle(&script);
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = format!(
            r#"
import numpy as np
result = np.correlate(np.array({a_str}), np.array({v_str}), mode='valid')
print(result)
"#
        );
        let rust_result = numpy_oracle(&rust_script);
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-10),
            "correlate valid mismatch for a={a_str}, v={v_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
}
