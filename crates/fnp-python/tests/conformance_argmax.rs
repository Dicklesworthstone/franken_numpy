//! Conformance tests for numpy.argmax against NumPy oracle.
//!
//! Tests the native Rust argmax implementation against NumPy across various
//! input shapes, axis parameters, and data types.

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

fn fnp_argmax_script(body: String) -> String {
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

fn parse_int(s: &str) -> i64 {
    s.trim().parse::<i64>().unwrap_or(-1)
}

fn parse_int_list(s: &str) -> Vec<i64> {
    if s.is_empty() || s == "[]" {
        return vec![];
    }
    let trimmed = s.trim_start_matches('[').trim_end_matches(']');
    trimmed
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
        .filter_map(|token| token.parse::<i64>().ok())
        .collect()
}

#[test]
fn argmax_flat_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        // Basic arrays - max at various positions
        "[1, 3, 2]",
        "[1, 2, 3, 4, 5]",
        "[5, 4, 3, 2, 1]",
        "[1]",
        "[1, 1, 1, 1]",
        "[0, 0, 0]",
        "[-1, -2, -3]",
        "[-3, -2, -1]",
        "[1, -1, 2, -2, 3, -3]",
        "[100, 500, 200, 400, 300]",
        // Floating point
        "[0.5, 2.5, 1.5]",
        "[1.1, 4.4, 2.2, 3.3]",
        "[0.001, 0.003, 0.002]",
        "[1e10, 3e10, 2e10]",
        "[1e-10, 3e-10, 2e-10]",
        // Negatives and zeros
        "[100, 0, -100]",
        "[-1.5, -0.5, 1.5, 0.5]",
        "[0, 1, 0, 1, 0]",
        "[-5, -4, 0, -2, -1, -3]",
        "[0, -1, -2, -3, -4, -5]",
        // Larger arrays
        "[1, 2, 3, 4, 10, 6, 7, 8, 9, 5]",
        "[1, 9, 8, 7, 6, 5, 4, 3, 2, 10]",
        "[1, 3, 5, 7, 9, 15, 13, 11]",
        "[2, 4, 16, 8, 10, 12, 14, 6]",
        "[1, 21, 2, 3, 5, 8, 13, 0]",
        // Mixed
        "[0.5, 1, 3.0, 2, 2.5, 1.5]",
        "[-2.5, -1.5, -0.5, 2.5, 1.5, 0.5]",
        "[1, 10, 10000, 1000, 100]",
        "[1, 1000, 100, 10, 10000]",
        "[3.14159, 2.71828, 1.41421]",
        // Edge values
        "[0.0, 0.0]",
        "[1.0, 1.5, 1.0, 1.0, 1.0]",
        "[999, -999]",
        "[0.123456789, 0.987654321]",
        "[1, 2]",
        // More variety
        "[7, 3, 9, 1, 5]",
        "[2, 8, 4, 6, 0]",
        "[11, 66, 33, 44, 55, 22]",
        "[33, 88, 77, 66, 55, 44, 99]",
        "[1, 4, 49, 16, 25, 36, 9]",
        // Small ranges
        "[1.0, 1.3, 1.1, 1.2]",
        "[0.99, 1.01, 1.00]",
        "[-0.01, 0.01, 0.0]",
        "[100.0, 100.5, 101.0]",
        "[1000, 1003, 1001, 1002]",
        // Additional cases
        "[5, 15, 25, 45, 35]",
        "[0, 2, 10, 6, 8, 4]",
        "[-10, 10, 0, 5, -5]",
        "[0.1, 0.2, 1.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.3]",
        "[1, 3, 2, 6, 3, 5, 4, 3]",
    ];

    for arr_str in &test_cases {
        let script = format!("import numpy as np; print(np.argmax(np.array({arr_str})))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_int(&numpy_result);

        let rust_script = fnp_argmax_script(format!("print(fnp.argmax(np.array({arr_str})))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_int(&rust_result);

        assert_eq!(
            numpy_val, rust_val,
            "argmax flat mismatch for {arr_str}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }

    Ok(())
}

#[test]
fn argmax_2d_axis_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        // 2D arrays with axis=0
        ("[[1, 5, 3], [4, 2, 6]]", "0"),
        ("[[4, 1], [2, 5], [3, 6]]", "0"),
        ("[[1, 8], [3, 4], [5, 6], [7, 2]]", "0"),
        ("[[20, 5, 30], [10, 15, 25]]", "0"),
        ("[[1, 3, 2], [3, 1, 2], [2, 2, 3]]", "0"),
        // 2D arrays with axis=1
        ("[[1, 3, 2], [4, 6, 5]]", "1"),
        ("[[1, 4], [5, 2], [3, 6]]", "1"),
        ("[[1, 2], [3, 4], [5, 6], [7, 8]]", "1"),
        ("[[10, 30, 20], [25, 15, 5]]", "1"),
        ("[[1, 9, 5], [2, 10, 6], [3, 11, 7]]", "1"),
        // Negative axis
        ("[[1, 3, 2], [4, 6, 5]]", "-1"),
        ("[[1, 3, 2], [4, 6, 5]]", "-2"),
        ("[[1, 7, 4], [2, 8, 5], [3, 9, 6]]", "-1"),
        ("[[1, 7, 4], [2, 8, 5], [3, 9, 6]]", "-2"),
        // Single row/column
        ("[[1, 5, 3, 2, 4]]", "0"),
        ("[[1, 5, 3, 2, 4]]", "1"),
        ("[[3], [1], [4], [2]]", "0"),
        ("[[3], [1], [4], [2]]", "1"),
        // Floating point 2D
        ("[[0.5, 3.5], [2.5, 1.5]]", "0"),
        ("[[0.5, 3.5], [2.5, 1.5]]", "1"),
    ];

    for (arr_str, axis) in &test_cases {
        let script =
            format!("import numpy as np; print(list(np.argmax(np.array({arr_str}), axis={axis})))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_int_list(&numpy_result);

        let rust_script = fnp_argmax_script(format!(
            "print(list(fnp.argmax(np.array({arr_str}), axis={axis})))"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_int_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "argmax axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn argmax_3d_axis_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        // 3D arrays
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "0"),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "1"),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "2"),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "-1"),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "-2"),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "-3"),
        // Different shapes
        ("[[[1, 5, 3]], [[4, 2, 6]]]", "0"),
        ("[[[4, 2, 6]], [[1, 5, 3]]]", "1"),
        ("[[[1, 3, 2]], [[4, 6, 5]]]", "2"),
        ("[[[6], [4], [5]], [[1], [3], [2]]]", "0"),
        ("[[[1], [3], [2]], [[6], [4], [5]]]", "1"),
        ("[[[1], [3], [2]], [[6], [4], [5]]]", "2"),
    ];

    for (arr_str, axis) in &test_cases {
        let script = format!(
            "import numpy as np; print(list(np.argmax(np.array({arr_str}), axis={axis}).flatten()))"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_int_list(&numpy_result);

        let rust_script = fnp_argmax_script(format!(
            "print(list(fnp.argmax(np.array({arr_str}), axis={axis}).flatten()))"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_int_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "argmax 3D axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn argmax_integer_dtypes_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([1, 3, 2], dtype=np.int32)", "None"),
        ("np.array([1, 3, 2], dtype=np.int64)", "None"),
        ("np.array([1, 3, 2], dtype=np.uint8)", "None"),
        ("np.array([100, 300, 200], dtype=np.int16)", "None"),
        ("np.array([[1, 4], [3, 2]], dtype=np.int32)", "None"),
        ("np.array([[1, 4], [3, 2]], dtype=np.int64)", "None"),
        ("np.array([[1, 4], [3, 2]], dtype=np.float32)", "None"),
        ("np.array([[1, 4], [3, 2]], dtype=np.float64)", "None"),
    ];

    for (arr_expr, axis) in &test_cases {
        let axis_arg = if *axis == "None" {
            String::new()
        } else {
            format!(", axis={axis}")
        };
        let script = format!("import numpy as np; print(int(np.argmax({arr_expr}{axis_arg})))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_int(&numpy_result);

        let rust_script =
            fnp_argmax_script(format!("print(int(fnp.argmax({arr_expr}{axis_arg})))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_int(&rust_result);

        assert_eq!(
            numpy_val, rust_val,
            "argmax dtype mismatch for {arr_expr} axis={axis}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }

    Ok(())
}

#[test]
fn argmax_first_occurrence_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "[1, 1, 1]",
        "[1, 2, 2, 1]",
        "[1, 3, 2, 3, 1]",
        "[5, 5, 5, 5, 5]",
        "[[3, 2], [3, 1]]",
    ];

    for arr_str in &test_cases {
        let script = format!("import numpy as np; print(np.argmax(np.array({arr_str})))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_int(&numpy_result);

        let rust_script = fnp_argmax_script(format!("print(fnp.argmax(np.array({arr_str})))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_int(&rust_result);

        assert_eq!(
            numpy_val, rust_val,
            "argmax first occurrence mismatch for {arr_str}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }

    Ok(())
}
