//! Conformance tests for numpy.argmin against NumPy oracle.
//!
//! Tests the native Rust argmin implementation against NumPy across various
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

fn fnp_argmin_script(body: String) -> String {
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
fn argmin_flat_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        // Basic arrays - min at various positions
        "[3, 1, 2]",
        "[1, 2, 3, 4, 5]",
        "[5, 4, 3, 2, 1]",
        "[1]",
        "[1, 1, 1, 1]",
        "[0, 0, 0]",
        "[-1, -2, -3]",
        "[-3, -2, -1]",
        "[1, -1, 2, -2, 3, -3]",
        "[100, 200, 50, 400, 500]",
        // Floating point
        "[2.5, 0.5, 1.5]",
        "[1.1, 2.2, 0.5, 4.4]",
        "[0.003, 0.001, 0.002]",
        "[3e10, 1e10, 2e10]",
        "[2e-10, 1e-10, 3e-10]",
        // Negatives and zeros
        "[-100, 0, 100]",
        "[-1.5, -2.0, 0.5, 1.5]",
        "[0, -1, 0, 1, 0]",
        "[-5, -6, -3, -2, -1, 0]",
        "[0, -1, -2, -5, -4, -3]",
        // Larger arrays
        "[5, 2, 3, 4, 1, 6, 7, 8, 9, 10]",
        "[10, 9, 8, 7, 6, 1, 4, 3, 2, 5]",
        "[11, 3, 5, 7, 1, 9, 13, 15]",
        "[2, 4, 6, 0, 10, 12, 14, 16]",
        "[21, 1, 2, 3, 5, 8, 13, 0]",
        // Mixed
        "[0.5, 0.1, 1.5, 2, 2.5, 3]",
        "[-2.5, -3.0, -0.5, 0.5, 1.5, 2.5]",
        "[1, 10, 100, 0, 10000]",
        "[10000, 1000, 100, 10, 1]",
        "[3.14159, 1.41421, 2.71828]",
        // Edge values
        "[0.0, 0.0]",
        "[1.0, 1.0, 0.5, 1.0, 1.0]",
        "[-999, 999]",
        "[0.987654321, 0.123456789]",
        "[2, 1]",
        // More variety
        "[7, 3, 9, 1, 5]",
        "[2, 0, 4, 6, 8]",
        "[11, 22, 33, 0, 55, 66]",
        "[99, 88, 77, 66, 55, 33, 44]",
        "[49, 4, 9, 16, 1, 36, 25]",
        // Small ranges
        "[1.3, 1.0, 1.1, 1.2]",
        "[1.01, 0.99, 1.00]",
        "[0.0, -0.01, 0.01]",
        "[101.0, 100.0, 100.5]",
        "[1003, 1000, 1001, 1002]",
        // Additional cases
        "[45, 5, 15, 25, 35]",
        "[10, 2, 4, 0, 8, 6]",
        "[-10, -5, 0, 5, 10]",
        "[1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]",
        "[6, 3, 2, 4, 1, 5, 4, 3]",
    ];

    for arr_str in &test_cases {
        let script = format!("import numpy as np; print(np.argmin(np.array({arr_str})))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_int(&numpy_result);

        let rust_script = fnp_argmin_script(format!("print(fnp.argmin(np.array({arr_str})))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_int(&rust_result);

        assert_eq!(
            numpy_val, rust_val,
            "argmin flat mismatch for {arr_str}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }

    Ok(())
}

#[test]
fn argmin_2d_axis_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        // 2D arrays with axis=0
        ("[[4, 2, 6], [1, 5, 3]]", "0"),
        ("[[1, 4], [2, 3], [0, 6]]", "0"),
        ("[[8, 2], [3, 4], [5, 1], [7, 6]]", "0"),
        ("[[10, 5, 30], [20, 15, 25]]", "0"),
        ("[[3, 1, 2], [1, 3, 2], [2, 2, 1]]", "0"),
        // 2D arrays with axis=1
        ("[[3, 1, 2], [6, 4, 5]]", "1"),
        ("[[4, 1], [2, 5], [6, 3]]", "1"),
        ("[[2, 1], [4, 3], [6, 5], [8, 7]]", "1"),
        ("[[30, 10, 20], [15, 25, 5]]", "1"),
        ("[[9, 1, 5], [10, 2, 6], [11, 3, 7]]", "1"),
        // Negative axis
        ("[[3, 1, 2], [6, 4, 5]]", "-1"),
        ("[[3, 1, 2], [6, 4, 5]]", "-2"),
        ("[[7, 1, 4], [8, 2, 5], [9, 3, 6]]", "-1"),
        ("[[7, 1, 4], [8, 2, 5], [9, 3, 6]]", "-2"),
        // Single row/column
        ("[[5, 1, 3, 2, 4]]", "0"),
        ("[[5, 1, 3, 2, 4]]", "1"),
        ("[[1], [3], [2], [4]]", "0"),
        ("[[1], [3], [2], [4]]", "1"),
        // Floating point 2D
        ("[[3.5, 0.5], [2.5, 1.5]]", "0"),
        ("[[3.5, 0.5], [2.5, 1.5]]", "1"),
    ];

    for (arr_str, axis) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.argmin(np.array({arr_str}), axis={axis}).tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_int_list(&numpy_result);

        let rust_script = fnp_argmin_script(format!(
            "print(fnp.argmin(np.array({arr_str}), axis={axis}).tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_int_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "argmin axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn argmin_3d_axis_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        // 3D arrays
        ("[[[8, 2], [3, 4]], [[5, 6], [7, 1]]]", "0"),
        ("[[[1, 8], [3, 4]], [[5, 2], [7, 6]]]", "1"),
        ("[[[1, 2], [4, 3]], [[6, 5], [8, 7]]]", "2"),
        ("[[[2, 1], [4, 3]], [[6, 5], [8, 7]]]", "-1"),
        ("[[[1, 2], [4, 3]], [[5, 6], [8, 7]]]", "-2"),
        ("[[[5, 2], [3, 4]], [[1, 6], [7, 8]]]", "-3"),
        // Different shapes
        ("[[[4, 2, 6]], [[1, 5, 3]]]", "0"),
        ("[[[1, 5, 3]], [[4, 2, 6]]]", "1"),
        ("[[[3, 1, 2]], [[6, 4, 5]]]", "2"),
        ("[[[1], [3], [2]], [[6], [4], [5]]]", "0"),
        ("[[[1], [3], [2]], [[6], [4], [5]]]", "1"),
        ("[[[1], [3], [2]], [[6], [4], [5]]]", "2"),
    ];

    for (arr_str, axis) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.argmin(np.array({arr_str}), axis={axis}).flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_int_list(&numpy_result);

        let rust_script = fnp_argmin_script(format!(
            "print(fnp.argmin(np.array({arr_str}), axis={axis}).flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_int_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "argmin 3D axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn argmin_integer_dtypes_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([3, 1, 2], dtype=np.int32)", "None"),
        ("np.array([3, 1, 2], dtype=np.int64)", "None"),
        ("np.array([3, 1, 2], dtype=np.uint8)", "None"),
        ("np.array([300, 100, 200], dtype=np.int16)", "None"),
        ("np.array([[4, 2], [1, 3]], dtype=np.int32)", "None"),
        ("np.array([[4, 2], [1, 3]], dtype=np.int64)", "None"),
        ("np.array([[4, 2], [1, 3]], dtype=np.float32)", "None"),
        ("np.array([[4, 2], [1, 3]], dtype=np.float64)", "None"),
    ];

    for (arr_expr, axis) in &test_cases {
        let axis_arg = if *axis == "None" {
            String::new()
        } else {
            format!(", axis={axis}")
        };
        let script = format!("import numpy as np; print(int(np.argmin({arr_expr}{axis_arg})))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_int(&numpy_result);

        let rust_script =
            fnp_argmin_script(format!("print(int(fnp.argmin({arr_expr}{axis_arg})))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_int(&rust_result);

        assert_eq!(
            numpy_val, rust_val,
            "argmin dtype mismatch for {arr_expr} axis={axis}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }

    Ok(())
}

#[test]
fn argmin_first_occurrence_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "[1, 1, 1]",
        "[2, 1, 1, 3]",
        "[3, 1, 2, 1, 4]",
        "[5, 5, 5, 5, 5]",
        "[[1, 2], [1, 3]]",
    ];

    for arr_str in &test_cases {
        let script = format!("import numpy as np; print(np.argmin(np.array({arr_str})))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_int(&numpy_result);

        let rust_script = fnp_argmin_script(format!("print(fnp.argmin(np.array({arr_str})))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_int(&rust_result);

        assert_eq!(
            numpy_val, rust_val,
            "argmin first occurrence mismatch for {arr_str}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }

    Ok(())
}

#[test]
fn argmin_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_argmin_script(
        r#"
x = np.float64(5.0)
fnp_result = fnp.argmin(x)
np_result = np.argmin(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "argmin scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn argmin_complex() -> Result<(), String> {
    let script = fnp_argmin_script(
        r#"
a = np.array([3+1j, 1-1j, 2+2j], dtype=np.complex128)
fnp_result = fnp.argmin(a)
np_result = np.argmin(a)
print(fnp_result == np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "argmin complex should match numpy");
    Ok(())
}

#[test]
fn argmin_with_nan() -> Result<(), String> {
    let script = fnp_argmin_script(
        r#"
a = np.array([1.0, np.nan, 3.0])
fnp_result = fnp.argmin(a)
np_result = np.argmin(a)
print(fnp_result == np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "argmin nan handling should match numpy"
    );
    Ok(())
}

#[test]
fn argmin_empty_array_raises_valueerror() -> Result<(), String> {
    let script = fnp_argmin_script(
        r#"
empty = np.array([])
fnp_raised = False
np_raised = False
try:
    fnp.argmin(empty)
except ValueError:
    fnp_raised = True
except Exception:
    pass
try:
    np.argmin(empty)
except ValueError:
    np_raised = True
except Exception:
    pass
print(fnp_raised == np_raised == True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "argmin of empty array should raise ValueError"
    );
    Ok(())
}
