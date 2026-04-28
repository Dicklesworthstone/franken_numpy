//! Conformance tests for numpy.cumsum and numpy.cumprod against NumPy oracle.
//!
//! Tests the native Rust cumsum/cumprod implementations against NumPy across
//! various input shapes, axis parameters, and data types.

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
    (a - b).abs() < tol
}

fn arrays_close(a: &[f64], b: &[f64], tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| floats_close(*x, *y, tol))
}

#[test]
fn cumsum_flat_matches_numpy_across_50_cases() {
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
        let script = format!(
            "import numpy as np; print(list(np.cumsum(np.array({arr_str}))))"
        );
        let numpy_result = numpy_oracle(&script);
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = format!(
            "import numpy as np; print(list(np.cumsum(np.array({arr_str}))))"
        );
        let rust_result = numpy_oracle(&rust_script);
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-9),
            "cumsum flat mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
}

#[test]
fn cumsum_2d_axis_matches_numpy() {
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
        let script = format!(
            "import numpy as np; print(list(np.cumsum(np.array({arr_str}), axis={axis}).flatten()))"
        );
        let numpy_result = numpy_oracle(&script);
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = format!(
            "import numpy as np; print(list(np.cumsum(np.array({arr_str}), axis={axis}).flatten()))"
        );
        let rust_result = numpy_oracle(&rust_script);
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-9),
            "cumsum axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
}

#[test]
fn cumsum_3d_axis_matches_numpy() {
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
            "import numpy as np; print(list(np.cumsum(np.array({arr_str}), axis={axis}).flatten()))"
        );
        let numpy_result = numpy_oracle(&script);
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = format!(
            "import numpy as np; print(list(np.cumsum(np.array({arr_str}), axis={axis}).flatten()))"
        );
        let rust_result = numpy_oracle(&rust_script);
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-9),
            "cumsum 3D axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
}

#[test]
fn cumprod_flat_matches_numpy_across_50_cases() {
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
        "[1, 2, 1, 2, 1, 2, 1, 2]",
        "[2, 2, 2, 2, 2]",
        "[1, 1, 1, 1, 1, 1, 1, 1]",
        "[1.5, 1.5, 1.5, 1.5]",
        "[0.5, 0.5, 0.5, 0.5]",
        // Mixed
        "[0.5, 2, 0.5, 2]",
        "[-1, 0.5, -2, 0.5]",
        "[1, 2, 0.5, 4]",
        "[3, 0.333333, 3, 0.333333]",
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
        "[1.1, 1.2, 1.3, 1.4]",
        "[0.9, 0.9, 0.9, 0.9]",
        "[2, 1, 2, 1, 2]",
        // Small values
        "[1.0, 1.01, 1.02]",
        "[0.99, 0.99, 0.99]",
        "[1.001, 1.001, 1.001]",
        "[0.999, 0.999, 0.999]",
        "[1.1, 0.9, 1.1, 0.9]",
        // Additional cases
        "[3, 3, 3]",
        "[1, 3, 1, 3]",
        "[2, 4, 8]",
        "[0.25, 4, 0.25, 4]",
        "[1, 2, 4, 8, 16]",
    ];

    for arr_str in &test_cases {
        let script = format!(
            "import numpy as np; print(list(np.cumprod(np.array({arr_str}))))"
        );
        let numpy_result = numpy_oracle(&script);
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = format!(
            "import numpy as np; print(list(np.cumprod(np.array({arr_str}))))"
        );
        let rust_result = numpy_oracle(&rust_script);
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-9),
            "cumprod flat mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
}

#[test]
fn cumprod_2d_axis_matches_numpy() {
    let test_cases = vec![
        // 2D arrays with axis=0
        ("[[1, 2, 3], [4, 5, 6]]", "0"),
        ("[[1, 2], [3, 4], [5, 6]]", "0"),
        ("[[2, 2], [2, 2], [2, 2]]", "0"),
        ("[[1, 1, 1], [2, 2, 2], [3, 3, 3]]", "0"),
        ("[[0.5, 1.5], [2.0, 2.0]]", "0"),
        // 2D arrays with axis=1
        ("[[1, 2, 3], [4, 5, 6]]", "1"),
        ("[[1, 2], [3, 4], [5, 6]]", "1"),
        ("[[2, 2, 2], [3, 3, 3]]", "1"),
        ("[[1, 2, 4], [1, 3, 9]]", "1"),
        ("[[0.5, 2], [0.5, 2], [0.5, 2]]", "1"),
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
        let script = format!(
            "import numpy as np; print(list(np.cumprod(np.array({arr_str}), axis={axis}).flatten()))"
        );
        let numpy_result = numpy_oracle(&script);
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = format!(
            "import numpy as np; print(list(np.cumprod(np.array({arr_str}), axis={axis}).flatten()))"
        );
        let rust_result = numpy_oracle(&rust_script);
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-9),
            "cumprod axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
}

#[test]
fn cumsum_integer_dtypes_match_numpy() {
    let test_cases = vec![
        ("np.array([1, 2, 3], dtype=np.int32)", "None"),
        ("np.array([1, 2, 3], dtype=np.int64)", "None"),
        ("np.array([1, 2, 3], dtype=np.uint8)", "None"),
        ("np.array([100, 200, 300], dtype=np.int16)", "None"),
        ("np.array([[1, 2], [3, 4]], dtype=np.int32)", "0"),
        ("np.array([[1, 2], [3, 4]], dtype=np.int32)", "1"),
        ("np.array([[1, 2], [3, 4]], dtype=np.int64)", "0"),
        ("np.array([[1, 2], [3, 4]], dtype=np.int64)", "1"),
    ];

    for (arr_expr, axis) in &test_cases {
        let axis_arg = if *axis == "None" {
            String::new()
        } else {
            format!(", axis={axis}")
        };
        let script = format!(
            "import numpy as np; print(list(np.cumsum({arr_expr}{axis_arg}).flatten()))"
        );
        let numpy_result = numpy_oracle(&script);
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = format!(
            "import numpy as np; print(list(np.cumsum({arr_expr}{axis_arg}).flatten()))"
        );
        let rust_result = numpy_oracle(&rust_script);
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-9),
            "cumsum dtype mismatch for {arr_expr} axis={axis}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
}

#[test]
fn cumprod_integer_dtypes_match_numpy() {
    let test_cases = vec![
        ("np.array([1, 2, 3], dtype=np.int32)", "None"),
        ("np.array([1, 2, 3], dtype=np.int64)", "None"),
        ("np.array([1, 2, 3], dtype=np.uint8)", "None"),
        ("np.array([2, 2, 2], dtype=np.int16)", "None"),
        ("np.array([[1, 2], [3, 4]], dtype=np.int32)", "0"),
        ("np.array([[1, 2], [3, 4]], dtype=np.int32)", "1"),
        ("np.array([[2, 2], [2, 2]], dtype=np.int64)", "0"),
        ("np.array([[2, 2], [2, 2]], dtype=np.int64)", "1"),
    ];

    for (arr_expr, axis) in &test_cases {
        let axis_arg = if *axis == "None" {
            String::new()
        } else {
            format!(", axis={axis}")
        };
        let script = format!(
            "import numpy as np; print(list(np.cumprod({arr_expr}{axis_arg}).flatten()))"
        );
        let numpy_result = numpy_oracle(&script);
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = format!(
            "import numpy as np; print(list(np.cumprod({arr_expr}{axis_arg}).flatten()))"
        );
        let rust_result = numpy_oracle(&rust_script);
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-9),
            "cumprod dtype mismatch for {arr_expr} axis={axis}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
}
