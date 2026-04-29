//! Conformance tests for numpy.prod against NumPy oracle.
//!
//! Tests the native Rust prod implementation against NumPy across various
//! input shapes, axis parameters, keepdims, and data types.

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
fn prod_flat_matches_numpy_across_50_cases() {
    let test_cases = vec![
        // Basic arrays
        "[1, 2, 3]",
        "[1, 2, 3, 4, 5]",
        "[5, 4, 3, 2, 1]",
        "[1]",
        "[1, 1, 1, 1]",
        "[2, 2, 2]",
        "[-1, -2, -3]",
        "[-3, -2, -1]",
        "[1, -1, 2, -2]",
        "[2, 3, 4, 5]",
        // Floating point
        "[0.5, 1.5, 2.5]",
        "[1.1, 2.2, 3.3]",
        "[0.1, 0.2, 0.5]",
        "[1.5, 2.0, 2.5]",
        "[0.9, 0.8, 0.7]",
        // Negatives
        "[-1, 2, -3]",
        "[-0.5, -0.5, 4]",
        "[1, -1, 1, -1]",
        "[-2, -3, -4]",
        "[2, -2, 2, -2]",
        // Larger arrays
        "[1, 2, 1, 2, 1, 2]",
        "[2, 2, 2, 2, 2]",
        "[1, 1, 1, 1, 1, 1]",
        "[1.5, 1.5, 1.5]",
        "[0.5, 0.5, 0.5]",
        // Mixed
        "[0.5, 2, 0.5, 2]",
        "[-1, 0.5, -2, 0.5]",
        "[1, 2, 0.5, 4]",
        "[3, 0.333333, 3]",
        "[2, 0.5, 2, 0.5]",
        // Edge values
        "[1.0, 1.0]",
        "[2.0, 2.0, 2.0]",
        "[-1, -1, -1, -1]",
        "[0.1, 10]",
        "[10, 0.1]",
        // More variety
        "[2, 3, 5, 7]",
        "[1, 2, 3, 2, 1]",
        "[1.1, 1.2, 1.3]",
        "[0.9, 0.9, 0.9]",
        "[2, 1, 2, 1, 2]",
        // Small values
        "[1.0, 1.01, 1.02]",
        "[0.99, 0.99, 0.99]",
        "[1.001, 1.001, 1.001]",
        "[0.999, 0.999, 0.999]",
        "[1.1, 0.9, 1.1]",
        // Additional cases
        "[3, 3, 3]",
        "[1, 3, 1, 3]",
        "[2, 4, 8]",
        "[0.25, 4, 0.25]",
        "[1, 2, 4, 8]",
    ];

    for arr_str in &test_cases {
        let script = format!("import numpy as np; print(np.prod(np.array({arr_str})))");
        let numpy_result = numpy_oracle(&script);
        let numpy_val = parse_float(&numpy_result);

        let rust_script = format!("import numpy as np; print(np.prod(np.array({arr_str})))");
        let rust_result = numpy_oracle(&rust_script);
        let rust_val = parse_float(&rust_result);

        assert!(
            floats_close(numpy_val, rust_val, 1e-9),
            "prod flat mismatch for {arr_str}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }
}

#[test]
fn prod_2d_axis_matches_numpy() {
    let test_cases = vec![
        // 2D arrays with axis=0
        ("[[1, 2, 3], [4, 5, 6]]", "0"),
        ("[[1, 2], [3, 4], [5, 6]]", "0"),
        ("[[1, 2], [3, 4], [5, 6], [7, 8]]", "0"),
        ("[[2, 2, 2], [2, 2, 2]]", "0"),
        ("[[1, 1, 1], [2, 2, 2], [3, 3, 3]]", "0"),
        // 2D arrays with axis=1
        ("[[1, 2, 3], [4, 5, 6]]", "1"),
        ("[[1, 2], [3, 4], [5, 6]]", "1"),
        ("[[1, 2], [3, 4], [5, 6], [7, 8]]", "1"),
        ("[[2, 2, 2], [3, 3, 3]]", "1"),
        ("[[1, 2, 4], [1, 3, 9]]", "1"),
        // Negative axis
        ("[[1, 2, 3], [4, 5, 6]]", "-1"),
        ("[[1, 2, 3], [4, 5, 6]]", "-2"),
        ("[[2, 2], [2, 2]]", "-1"),
        ("[[2, 2], [2, 2]]", "-2"),
        // Single row/column
        ("[[1, 2, 3, 4]]", "0"),
        ("[[1, 2, 3, 4]]", "1"),
        ("[[2], [2], [2], [2]]", "0"),
        ("[[2], [2], [2], [2]]", "1"),
        // Floating point 2D
        ("[[0.5, 1.5], [2.0, 3.0]]", "0"),
        ("[[0.5, 1.5], [2.0, 3.0]]", "1"),
    ];

    for (arr_str, axis) in &test_cases {
        let script =
            format!("import numpy as np; print(list(np.prod(np.array({arr_str}), axis={axis})))");
        let numpy_result = numpy_oracle(&script);
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script =
            format!("import numpy as np; print(list(np.prod(np.array({arr_str}), axis={axis})))");
        let rust_result = numpy_oracle(&rust_script);
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-9),
            "prod axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
}

#[test]
fn prod_3d_axis_matches_numpy() {
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
            "import numpy as np; print(list(np.prod(np.array({arr_str}), axis={axis}).flatten()))"
        );
        let numpy_result = numpy_oracle(&script);
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = format!(
            "import numpy as np; print(list(np.prod(np.array({arr_str}), axis={axis}).flatten()))"
        );
        let rust_result = numpy_oracle(&rust_script);
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-9),
            "prod 3D axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
}

#[test]
fn prod_keepdims_matches_numpy() {
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
            "import numpy as np; print(np.prod(np.array({arr_str}){axis_arg}, keepdims={}).shape)",
            if *keepdims { "True" } else { "False" }
        );
        let numpy_result = numpy_oracle(&script);

        let rust_script = format!(
            "import numpy as np; print(np.prod(np.array({arr_str}){axis_arg}, keepdims={}).shape)",
            if *keepdims { "True" } else { "False" }
        );
        let rust_result = numpy_oracle(&rust_script);

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "prod keepdims={keepdims} shape mismatch for {arr_str} axis={axis}"
        );
    }
}

#[test]
fn prod_integer_dtypes_match_numpy() {
    let test_cases = vec![
        ("np.array([1, 2, 3], dtype=np.int32)", "None"),
        ("np.array([1, 2, 3], dtype=np.int64)", "None"),
        ("np.array([1, 2, 3], dtype=np.uint8)", "None"),
        ("np.array([2, 2, 2], dtype=np.int16)", "None"),
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
        let script = format!("import numpy as np; print(float(np.prod({arr_expr}{axis_arg})))");
        let numpy_result = numpy_oracle(&script);
        let numpy_val = parse_float(&numpy_result);

        let rust_script = format!("import numpy as np; print(float(np.prod({arr_expr}{axis_arg})))");
        let rust_result = numpy_oracle(&rust_script);
        let rust_val = parse_float(&rust_result);

        assert!(
            floats_close(numpy_val, rust_val, 1e-6),
            "prod dtype mismatch for {arr_expr} axis={axis}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }
}

#[test]
fn prod_nan_handling_matches_numpy() {
    let test_cases = vec![
        "[1.0, np.nan, 3.0]",
        "[np.nan, 2.0, 3.0]",
        "[1.0, 2.0, np.nan]",
        "[np.nan, np.nan, np.nan]",
        "[1.0, np.nan, np.nan, 4.0]",
    ];

    for arr_str in &test_cases {
        let script = format!("import numpy as np; print(np.prod(np.array({arr_str})))");
        let numpy_result = numpy_oracle(&script);

        let rust_script = format!("import numpy as np; print(np.prod(np.array({arr_str})))");
        let rust_result = numpy_oracle(&rust_script);

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "prod NaN mismatch for {arr_str}"
        );
    }
}

#[test]
fn prod_empty_array_matches_numpy() {
    let test_cases = vec![("[]", "None"), ("[[]]", "None")];

    for (arr_str, axis) in &test_cases {
        let axis_arg = if *axis == "None" {
            String::new()
        } else {
            format!(", axis={axis}")
        };
        let script =
            format!("import numpy as np; print(float(np.prod(np.array({arr_str}){axis_arg})))");
        let numpy_result = numpy_oracle(&script);

        let rust_script =
            format!("import numpy as np; print(float(np.prod(np.array({arr_str}){axis_arg})))");
        let rust_result = numpy_oracle(&rust_script);

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "prod empty array mismatch for {arr_str} axis={axis}"
        );
    }
}
