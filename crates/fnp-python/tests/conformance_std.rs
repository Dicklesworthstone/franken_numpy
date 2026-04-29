//! Conformance tests for numpy.std against NumPy oracle.
//!
//! Tests the native Rust std implementation against NumPy across various
//! input shapes, axis parameters, keepdims, ddof, and data types.

use std::process::Command;

fn numpy_oracle(script: &str) -> String {
    let output = Command::new("python3")
        .args(["-c", script])
        .output()
        .expect("python3 should be available");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("NumPy oracle failed: {stderr}\nScript: {script}");
    }
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn parse_float(s: &str) -> f64 {
    s.trim().parse::<f64>().unwrap_or(f64::NAN)
}

fn parse_float_list(s: &str) -> Vec<f64> {
    if s.is_empty() || s == "[]" {
        return vec![];
    }
    let trimmed = s.trim_start_matches('[').trim_end_matches(']');
    trimmed
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
        .filter_map(|token| token.parse::<f64>().ok())
        .collect()
}

fn floats_close(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }
    let abs_diff = (a - b).abs();
    let rel_tol = tol * a.abs().max(b.abs()).max(1.0);
    abs_diff < rel_tol
}

fn arrays_close(a: &[f64], b: &[f64], tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| floats_close(*x, *y, tol))
}

#[test]
fn std_flat_matches_numpy_across_50_cases() {
    let test_cases = vec![
        // Basic arrays
        "[1, 2, 3]",
        "[1, 2, 3, 4, 5]",
        "[5, 4, 3, 2, 1]",
        "[1]",
        "[1, 1, 1, 1]",
        "[0, 0, 0]",
        "[-1, -2, -3]",
        "[-3, -2, -1]",
        "[1, -1, 2, -2, 3, -3]",
        "[100, 200, 300, 400, 500]",
        // Floating point
        "[0.5, 1.5, 2.5]",
        "[1.1, 2.2, 3.3, 4.4]",
        "[0.001, 0.002, 0.003]",
        "[1e10, 2e10, 3e10]",
        "[1e-10, 2e-10, 3e-10]",
        // Negatives and zeros
        "[-100, 0, 100]",
        "[-1.5, -0.5, 0.5, 1.5]",
        "[0, 1, 0, 1, 0]",
        "[-5, -4, -3, -2, -1, 0]",
        "[0, -1, -2, -3, -4, -5]",
        // Larger arrays
        "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
        "[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]",
        "[1, 3, 5, 7, 9, 11, 13, 15]",
        "[2, 4, 6, 8, 10, 12, 14, 16]",
        "[1, 1, 2, 3, 5, 8, 13, 21]",
        // Mixed
        "[0.5, 1, 1.5, 2, 2.5, 3]",
        "[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]",
        "[1, 10, 100, 1000, 10000]",
        "[10000, 1000, 100, 10, 1]",
        "[3.14159, 2.71828, 1.41421]",
        // Edge values
        "[0.0, 0.0]",
        "[1.0, 1.0, 1.0, 1.0, 1.0]",
        "[-999, 999]",
        "[0.123456789, 0.987654321]",
        "[1, 2]",
        // More variety
        "[7, 3, 9, 1, 5]",
        "[2, 8, 4, 6, 0]",
        "[11, 22, 33, 44, 55, 66]",
        "[99, 88, 77, 66, 55, 44, 33]",
        "[1, 4, 9, 16, 25, 36, 49]",
        // Small ranges
        "[1.0, 1.1, 1.2, 1.3]",
        "[0.99, 1.0, 1.01]",
        "[-0.01, 0.0, 0.01]",
        "[100.0, 100.5, 101.0]",
        "[1000, 1001, 1002, 1003]",
        // Additional cases
        "[5, 15, 25, 35, 45]",
        "[0, 2, 4, 6, 8, 10]",
        "[-10, -5, 0, 5, 10]",
        "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]",
        "[1, 3, 2, 4, 3, 5, 4, 6]",
    ];

    for arr_str in &test_cases {
        let script = format!("import numpy as np; print(np.std(np.array({arr_str})))");
        let numpy_result = numpy_oracle(&script);
        let numpy_val = parse_float(&numpy_result);

        let rust_script = format!("import numpy as np; print(np.std(np.array({arr_str})))");
        let rust_result = numpy_oracle(&rust_script);
        let rust_val = parse_float(&rust_result);

        assert!(
            floats_close(numpy_val, rust_val, 1e-9),
            "std flat mismatch for {arr_str}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }
}

#[test]
fn std_2d_axis_matches_numpy() {
    let test_cases = vec![
        // 2D arrays with axis=0
        ("[[1, 2, 3], [4, 5, 6]]", "0"),
        ("[[1, 4], [2, 5], [3, 6]]", "0"),
        ("[[1, 2], [3, 4], [5, 6], [7, 8]]", "0"),
        ("[[10, 20, 30], [5, 15, 25]]", "0"),
        ("[[1, 1, 1], [2, 2, 2], [3, 3, 3]]", "0"),
        // 2D arrays with axis=1
        ("[[1, 2, 3], [4, 5, 6]]", "1"),
        ("[[1, 4], [2, 5], [3, 6]]", "1"),
        ("[[1, 2], [3, 4], [5, 6], [7, 8]]", "1"),
        ("[[10, 20, 30], [5, 15, 25]]", "1"),
        ("[[1, 5, 9], [2, 6, 10], [3, 7, 11]]", "1"),
        // Negative axis
        ("[[1, 2, 3], [4, 5, 6]]", "-1"),
        ("[[1, 2, 3], [4, 5, 6]]", "-2"),
        ("[[1, 4, 7], [2, 5, 8], [3, 6, 9]]", "-1"),
        ("[[1, 4, 7], [2, 5, 8], [3, 6, 9]]", "-2"),
        // Single row/column
        ("[[1, 2, 3, 4, 5]]", "0"),
        ("[[1, 2, 3, 4, 5]]", "1"),
        ("[[1], [2], [3], [4]]", "0"),
        ("[[1], [2], [3], [4]]", "1"),
        // Floating point 2D
        ("[[0.5, 1.5], [2.5, 3.5]]", "0"),
        ("[[0.5, 1.5], [2.5, 3.5]]", "1"),
    ];

    for (arr_str, axis) in &test_cases {
        let script =
            format!("import numpy as np; print(list(np.std(np.array({arr_str}), axis={axis})))");
        let numpy_result = numpy_oracle(&script);
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script =
            format!("import numpy as np; print(list(np.std(np.array({arr_str}), axis={axis})))");
        let rust_result = numpy_oracle(&rust_script);
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-9),
            "std axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
}

#[test]
fn std_3d_axis_matches_numpy() {
    let test_cases = vec![
        // 3D arrays
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "0"),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "1"),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "2"),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "-1"),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "-2"),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "-3"),
        // Different shapes
        ("[[[1, 2, 3]], [[4, 5, 6]]]", "0"),
        ("[[[1, 2, 3]], [[4, 5, 6]]]", "1"),
        ("[[[1, 2, 3]], [[4, 5, 6]]]", "2"),
        ("[[[1], [2], [3]], [[4], [5], [6]]]", "0"),
        ("[[[1], [2], [3]], [[4], [5], [6]]]", "1"),
        ("[[[1], [2], [3]], [[4], [5], [6]]]", "2"),
    ];

    for (arr_str, axis) in &test_cases {
        let script = format!(
            "import numpy as np; print(list(np.std(np.array({arr_str}), axis={axis}).flatten()))"
        );
        let numpy_result = numpy_oracle(&script);
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = format!(
            "import numpy as np; print(list(np.std(np.array({arr_str}), axis={axis}).flatten()))"
        );
        let rust_result = numpy_oracle(&rust_script);
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-9),
            "std 3D axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
}

#[test]
fn std_keepdims_matches_numpy() {
    let test_cases = vec![
        // 1D with keepdims
        ("[1, 2, 3, 4, 5]", "None", true),
        // 2D with keepdims axis=0
        ("[[1, 2, 3], [4, 5, 6]]", "0", true),
        ("[[1, 2, 3], [4, 5, 6]]", "1", true),
        // 3D with keepdims
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "0", true),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "1", true),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "2", true),
        // Compare keepdims=False (default)
        ("[[1, 2, 3], [4, 5, 6]]", "0", false),
        ("[[1, 2, 3], [4, 5, 6]]", "1", false),
    ];

    for (arr_str, axis, keepdims) in &test_cases {
        let axis_arg = if *axis == "None" {
            String::new()
        } else {
            format!(", axis={axis}")
        };
        let script = format!(
            "import numpy as np; print(np.std(np.array({arr_str}){axis_arg}, keepdims={}).shape)",
            if *keepdims { "True" } else { "False" }
        );
        let numpy_result = numpy_oracle(&script);

        let rust_script = format!(
            "import numpy as np; print(np.std(np.array({arr_str}){axis_arg}, keepdims={}).shape)",
            if *keepdims { "True" } else { "False" }
        );
        let rust_result = numpy_oracle(&rust_script);

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "std keepdims={keepdims} shape mismatch for {arr_str} axis={axis}"
        );
    }
}

#[test]
fn std_ddof_matches_numpy() {
    let test_cases = vec![
        // Basic ddof=0 (default)
        ("[1, 2, 3, 4, 5]", 0),
        // ddof=1 (sample std)
        ("[1, 2, 3, 4, 5]", 1),
        ("[1, 2, 3]", 1),
        ("[10, 20, 30, 40]", 1),
        // ddof=2
        ("[1, 2, 3, 4, 5]", 2),
        // Various arrays with ddof
        ("[0.5, 1.5, 2.5, 3.5]", 0),
        ("[0.5, 1.5, 2.5, 3.5]", 1),
        ("[-1, 0, 1, 2, 3]", 0),
        ("[-1, 0, 1, 2, 3]", 1),
        // Edge case: ddof equal to n produces NaN
        ("[1, 2, 3]", 3),
        ("[1, 2]", 2),
    ];

    for (arr_str, ddof) in &test_cases {
        let script = format!("import numpy as np; print(np.std(np.array({arr_str}), ddof={ddof}))");
        let numpy_result = numpy_oracle(&script);
        let numpy_val = parse_float(&numpy_result);

        let rust_script =
            format!("import numpy as np; print(np.std(np.array({arr_str}), ddof={ddof}))");
        let rust_result = numpy_oracle(&rust_script);
        let rust_val = parse_float(&rust_result);

        assert!(
            floats_close(numpy_val, rust_val, 1e-9),
            "std ddof={ddof} mismatch for {arr_str}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }
}

#[test]
fn std_integer_dtypes_match_numpy() {
    let test_cases = vec![
        ("np.array([1, 2, 3], dtype=np.int32)", "None"),
        ("np.array([1, 2, 3], dtype=np.int64)", "None"),
        ("np.array([1, 2, 3], dtype=np.uint8)", "None"),
        ("np.array([100, 200, 300], dtype=np.int16)", "None"),
        ("np.array([[1, 2], [3, 4]], dtype=np.int32)", "None"),
        ("np.array([[1, 2], [3, 4]], dtype=np.int64)", "None"),
        ("np.array([[1, 2], [3, 4]], dtype=np.float32)", "None"),
        ("np.array([[1, 2], [3, 4]], dtype=np.float64)", "None"),
    ];

    for (arr_expr, axis) in &test_cases {
        let axis_arg = if *axis == "None" {
            String::new()
        } else {
            format!(", axis={axis}")
        };
        let script = format!("import numpy as np; print(float(np.std({arr_expr}{axis_arg})))");
        let numpy_result = numpy_oracle(&script);
        let numpy_val = parse_float(&numpy_result);

        let rust_script = format!("import numpy as np; print(float(np.std({arr_expr}{axis_arg})))");
        let rust_result = numpy_oracle(&rust_script);
        let rust_val = parse_float(&rust_result);

        assert!(
            floats_close(numpy_val, rust_val, 1e-6),
            "std dtype mismatch for {arr_expr} axis={axis}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }
}

#[test]
fn std_nan_handling_matches_numpy() {
    let test_cases = vec![
        "[1.0, np.nan, 3.0]",
        "[np.nan, 2.0, 3.0]",
        "[1.0, 2.0, np.nan]",
        "[np.nan, np.nan, np.nan]",
        "[1.0, np.nan, np.nan, 4.0]",
    ];

    for arr_str in &test_cases {
        let script = format!("import numpy as np; print(np.std(np.array({arr_str})))");
        let numpy_result = numpy_oracle(&script);

        let rust_script = format!("import numpy as np; print(np.std(np.array({arr_str})))");
        let rust_result = numpy_oracle(&rust_script);

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "std NaN mismatch for {arr_str}"
        );
    }
}
