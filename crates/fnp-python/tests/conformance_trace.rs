//! Conformance tests for numpy.trace against NumPy oracle.
//!
//! Tests the native Rust trace implementation against NumPy across various
//! input shapes, offset parameters, axis parameters, and data types.

use std::process::Command;

fn numpy_oracle(script: &str) -> Result<String, String> {
    let output = Command::new("python3")
        .args(["-c", script])
        .output()
        .map_err(|error| format!("python3 should be available: {error}\nScript: {script}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("NumPy oracle failed: {stderr}\nScript: {script}"));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn fnp_trace_script(body: String) -> String {
    let library_name = format!(
        "{}fnp_python{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    );
    let module_path = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.join(&library_name)))
        .unwrap_or_else(|| library_name.into());
    let module_literal = format!("{module_path:?}");
    format!(
        "import importlib.util\n\
         import numpy as np\n\
         spec = importlib.util.spec_from_file_location('fnp_python', {module_literal})\n\
         fnp = importlib.util.module_from_spec(spec)\n\
         spec.loader.exec_module(fnp)\n\
         {body}"
    )
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
    (a - b).abs() < tol
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
fn trace_2d_default_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        // Identity matrices
        "[[1, 0], [0, 1]]",
        "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]",
        "[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]",
        // Diagonal matrices
        "[[2, 0], [0, 3]]",
        "[[1, 0, 0], [0, 2, 0], [0, 0, 3]]",
        "[[5, 0, 0, 0], [0, 4, 0, 0], [0, 0, 3, 0], [0, 0, 0, 2]]",
        // General matrices
        "[[1, 2], [3, 4]]",
        "[[1, 2, 3], [4, 5, 6], [7, 8, 9]]",
        "[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]",
        // Non-square matrices
        "[[1, 2, 3], [4, 5, 6]]",
        "[[1, 2], [3, 4], [5, 6]]",
        "[[1, 2, 3, 4], [5, 6, 7, 8]]",
        "[[1, 2], [3, 4], [5, 6], [7, 8]]",
        // Floating point
        "[[1.5, 2.5], [3.5, 4.5]]",
        "[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]",
        "[[1.1, 2.2], [3.3, 4.4]]",
        // Negative values
        "[[-1, -2], [-3, -4]]",
        "[[-1, 2], [3, -4]]",
        "[[1, -2, 3], [-4, 5, -6], [7, -8, 9]]",
        // Mixed
        "[[0, 1], [2, 0]]",
        "[[10, 20], [30, 40]]",
        "[[100, 0], [0, 200]]",
        // Single element
        "[[5]]",
        "[[0]]",
        "[[-3]]",
        // Large values
        "[[1000, 2000], [3000, 4000]]",
        "[[1e10, 0], [0, 2e10]]",
        // Small values
        "[[0.001, 0.002], [0.003, 0.004]]",
        "[[1e-10, 0], [0, 2e-10]]",
        // Zeros
        "[[0, 0], [0, 0]]",
        "[[0, 1, 2], [3, 0, 5], [6, 7, 0]]",
        // More variety
        "[[7, 3], [9, 1]]",
        "[[2, 8, 4], [6, 0, 2], [1, 3, 5]]",
        "[[11, 22], [33, 44]]",
        "[[1, 4, 9], [16, 25, 36], [49, 64, 81]]",
        // Integer sequences
        "[[1, 2, 3], [4, 5, 6], [7, 8, 9]]",
        "[[9, 8, 7], [6, 5, 4], [3, 2, 1]]",
        // Symmetric
        "[[1, 2, 3], [2, 4, 5], [3, 5, 6]]",
        "[[5, 2], [2, 5]]",
        // Triangular-like
        "[[1, 2, 3], [0, 4, 5], [0, 0, 6]]",
        "[[1, 0, 0], [2, 3, 0], [4, 5, 6]]",
        // Wide rectangles
        "[[1, 2, 3, 4, 5]]",
        "[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]",
        // Tall rectangles
        "[[1], [2], [3], [4], [5]]",
        "[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]",
        // More floating point
        "[[3.14159, 2.71828], [1.41421, 1.73205]]",
        "[[0.5, 1.5, 2.5], [3.5, 4.5, 5.5], [6.5, 7.5, 8.5]]",
        // Edge cases
        "[[999, 1], [1, 999]]",
        "[[0.123456789, 0], [0, 0.987654321]]",
    ];

    for arr_str in &test_cases {
        let script = format!("import numpy as np; print(np.trace(np.array({arr_str})))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_float(&numpy_result);

        let rust_script = fnp_trace_script(format!("print(fnp.trace(np.array({arr_str})))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_float(&rust_result);

        assert!(
            floats_close(numpy_val, rust_val, 1e-9),
            "trace default mismatch for {arr_str}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }
    Ok(())
}

#[test]
fn trace_offset_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        // Positive offset (super-diagonal)
        ("[[1, 2, 3], [4, 5, 6], [7, 8, 9]]", 1),
        ("[[1, 2, 3], [4, 5, 6], [7, 8, 9]]", 2),
        (
            "[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]",
            1,
        ),
        (
            "[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]",
            2,
        ),
        (
            "[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]",
            3,
        ),
        // Negative offset (sub-diagonal)
        ("[[1, 2, 3], [4, 5, 6], [7, 8, 9]]", -1),
        ("[[1, 2, 3], [4, 5, 6], [7, 8, 9]]", -2),
        (
            "[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]",
            -1,
        ),
        (
            "[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]",
            -2,
        ),
        (
            "[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]",
            -3,
        ),
        // Non-square with offset
        ("[[1, 2, 3], [4, 5, 6]]", 1),
        ("[[1, 2, 3], [4, 5, 6]]", 2),
        ("[[1, 2, 3], [4, 5, 6]]", -1),
        ("[[1, 2], [3, 4], [5, 6]]", 1),
        ("[[1, 2], [3, 4], [5, 6]]", -1),
        ("[[1, 2], [3, 4], [5, 6]]", -2),
        // Zero offset (main diagonal)
        ("[[1, 2], [3, 4]]", 0),
        ("[[1, 2, 3], [4, 5, 6], [7, 8, 9]]", 0),
        // Floating point with offset
        ("[[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]]", 1),
        ("[[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]]", -1),
    ];

    for (arr_str, offset) in &test_cases {
        let script =
            format!("import numpy as np; print(np.trace(np.array({arr_str}), offset={offset}))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_float(&numpy_result);

        let rust_script = fnp_trace_script(format!(
            "print(fnp.trace(np.array({arr_str}), offset={offset}))"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_float(&rust_result);

        assert!(
            floats_close(numpy_val, rust_val, 1e-9),
            "trace offset={offset} mismatch for {arr_str}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }
    Ok(())
}

#[test]
fn trace_3d_axis_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        // 3D arrays with different axis pairs
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", 0, 1, 2),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", 0, 0, 1),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", 0, 0, 2),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", 0, 1, 0),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", 0, 2, 0),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", 0, 2, 1),
        // Different shapes
        (
            "[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]",
            0,
            0,
            1,
        ),
        (
            "[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]",
            0,
            0,
            2,
        ),
        (
            "[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]",
            0,
            1,
            2,
        ),
        // Negative axis
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", 0, -2, -1),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", 0, -1, -2),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", 0, 0, -1),
    ];

    for (arr_str, offset, axis1, axis2) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.trace(np.array({arr_str}), offset={offset}, axis1={axis1}, axis2={axis2}).flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = fnp_trace_script(format!(
            "print(fnp.trace(np.array({arr_str}), offset={offset}, axis1={axis1}, axis2={axis2}).flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-9),
            "trace 3D axis1={axis1}, axis2={axis2}, offset={offset} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
    Ok(())
}

#[test]
fn trace_integer_dtypes_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([[1, 2], [3, 4]], dtype=np.int32)", 0),
        ("np.array([[1, 2], [3, 4]], dtype=np.int64)", 0),
        ("np.array([[1, 2], [3, 4]], dtype=np.uint8)", 0),
        ("np.array([[100, 200], [300, 400]], dtype=np.int16)", 0),
        (
            "np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)",
            1,
        ),
        (
            "np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)",
            -1,
        ),
        (
            "np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)",
            0,
        ),
        (
            "np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)",
            0,
        ),
    ];

    for (arr_expr, offset) in &test_cases {
        let script = format!("import numpy as np; print(np.trace({arr_expr}, offset={offset}))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_float(&numpy_result);

        let rust_script =
            fnp_trace_script(format!("print(fnp.trace({arr_expr}, offset={offset}))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_float(&rust_result);

        assert!(
            floats_close(numpy_val, rust_val, 1e-9),
            "trace dtype mismatch for {arr_expr} offset={offset}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }
    Ok(())
}

#[test]
fn trace_nan_handling_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "[[1.0, np.nan], [3.0, 4.0]]",
        "[[np.nan, 2.0], [3.0, np.nan]]",
        "[[np.nan, np.nan], [np.nan, np.nan]]",
        "[[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]]",
    ];

    for arr_str in &test_cases {
        let script = format!("import numpy as np; print(np.trace(np.array({arr_str})))");
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_trace_script(format!("print(fnp.trace(np.array({arr_str})))"));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "trace NaN mismatch for {arr_str}"
        );
    }
    Ok(())
}
