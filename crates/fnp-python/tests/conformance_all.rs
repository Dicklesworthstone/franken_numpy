//! Conformance tests for numpy.all against NumPy oracle.
//!
//! Tests the native Rust all implementation against NumPy across various
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

fn fnp_all_script(body: String) -> String {
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

fn parse_bool(s: &str) -> bool {
    s.trim() == "True"
}

fn parse_bool_list(s: &str) -> Vec<bool> {
    if s.is_empty() || s == "[]" {
        return vec![];
    }
    let trimmed = s.trim_start_matches('[').trim_end_matches(']');
    trimmed
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
        .map(|token| token.trim() == "True")
        .collect()
}

#[test]
fn all_flat_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        // All true arrays
        "[1, 2, 3]",
        "[1, 1, 1, 1, 1]",
        "[True, True, True]",
        "[1]",
        "[100, 200, 300]",
        // Arrays with false
        "[1, 0, 1]",
        "[0, 0, 0]",
        "[True, False, True]",
        "[False, False, False]",
        "[1, 2, 0, 4, 5]",
        // Mixed boolean/int
        "[1, 2, 3, 4, 5]",
        "[5, 4, 3, 2, 1]",
        "[-1, -2, -3]",
        "[-3, -2, -1]",
        "[1, -1, 2, -2, 3, -3]",
        // Floating point
        "[0.5, 1.5, 2.5]",
        "[1.1, 2.2, 3.3, 4.4]",
        "[0.0, 1.0, 2.0]",
        "[1.0, 0.0, 2.0]",
        "[0.001, 0.002, 0.003]",
        // Zeros mixed
        "[0, 1, 0, 1, 0]",
        "[1, 1, 0, 1, 1]",
        "[0, 0, 0, 0, 1]",
        "[1, 0, 0, 0, 0]",
        "[0]",
        // Larger arrays
        "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
        "[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]",
        "[1, 1, 1, 1, 1, 0, 1, 1, 1, 1]",
        "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
        "[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]",
        // Edge cases
        "[0.0, 0.0]",
        "[1.0, 1.0, 1.0, 1.0, 1.0]",
        "[-999, 999]",
        "[0.0]",
        "[1, 2]",
        // More variety
        "[7, 3, 9, 1, 5]",
        "[2, 8, 4, 6, 0]",
        "[11, 22, 33, 44, 55, 66]",
        "[99, 88, 0, 66, 55, 44, 33]",
        "[1, 4, 9, 16, 25, 36, 49]",
        // Small values
        "[1.0, 1.1, 1.2, 1.3]",
        "[0.99, 1.0, 1.01]",
        "[0.0, 0.0, 0.0]",
        "[100.0, 100.5, 101.0]",
        "[1000, 1001, 1002, 1003]",
        // Additional cases
        "[5, 15, 25, 35, 45]",
        "[0, 2, 4, 6, 8, 10]",
        "[-10, -5, 0, 5, 10]",
        "[0.1, 0.2, 0.3, 0.4, 0.5]",
        "[1, 3, 2, 4, 0, 5, 4, 6]",
    ];

    for arr_str in &test_cases {
        let script = format!("import numpy as np; print(np.all(np.array({arr_str})))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_bool(&numpy_result);

        let rust_script = fnp_all_script(format!("print(fnp.all(np.array({arr_str})))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_bool(&rust_result);

        assert_eq!(
            numpy_val, rust_val,
            "all flat mismatch for {arr_str}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }

    Ok(())
}

#[test]
fn all_2d_axis_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        // 2D arrays with axis=0
        ("[[1, 1, 1], [1, 1, 1]]", "0"),
        ("[[1, 0, 1], [1, 1, 1]]", "0"),
        ("[[1, 1], [0, 1], [1, 1]]", "0"),
        ("[[0, 0], [1, 1], [1, 1], [1, 1]]", "0"),
        ("[[1, 1, 1], [0, 0, 0], [1, 1, 1]]", "0"),
        // 2D arrays with axis=1
        ("[[1, 1, 1], [1, 0, 1]]", "1"),
        ("[[1, 1], [0, 1], [1, 1]]", "1"),
        ("[[1, 0], [1, 1], [0, 1], [1, 1]]", "1"),
        ("[[1, 1, 1], [0, 0, 0]]", "1"),
        ("[[1, 1, 0], [1, 1, 1], [0, 1, 1]]", "1"),
        // Negative axis
        ("[[1, 1, 1], [1, 0, 1]]", "-1"),
        ("[[1, 1, 1], [1, 1, 1]]", "-2"),
        ("[[1, 0, 1], [1, 1, 1], [1, 1, 0]]", "-1"),
        ("[[1, 0, 1], [1, 1, 1], [1, 1, 0]]", "-2"),
        // Single row/column
        ("[[1, 1, 1, 1, 1]]", "0"),
        ("[[1, 1, 0, 1, 1]]", "1"),
        ("[[1], [1], [1], [1]]", "0"),
        ("[[1], [0], [1], [1]]", "1"),
        // All false
        ("[[0, 0], [0, 0]]", "0"),
        ("[[0, 0], [0, 0]]", "1"),
    ];

    for (arr_str, axis) in &test_cases {
        let script =
            format!("import numpy as np; print(np.all(np.array({arr_str}), axis={axis}).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_bool_list(&numpy_result);

        let rust_script = fnp_all_script(format!(
            "print(fnp.all(np.array({arr_str}), axis={axis}).tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_bool_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "all axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn all_3d_axis_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        // 3D arrays
        ("[[[1, 1], [1, 1]], [[1, 1], [1, 1]]]", "0"),
        ("[[[1, 1], [0, 1]], [[1, 1], [1, 1]]]", "1"),
        ("[[[1, 0], [1, 1]], [[1, 1], [1, 1]]]", "2"),
        ("[[[1, 1], [1, 1]], [[1, 1], [1, 0]]]", "-1"),
        ("[[[1, 1], [1, 1]], [[0, 1], [1, 1]]]", "-2"),
        ("[[[1, 1], [1, 1]], [[1, 1], [1, 1]]]", "-3"),
        // Different shapes
        ("[[[1, 1, 1]], [[1, 1, 1]]]", "0"),
        ("[[[1, 1, 0]], [[1, 1, 1]]]", "1"),
        ("[[[1, 0, 1]], [[1, 1, 1]]]", "2"),
        ("[[[1], [1], [1]], [[1], [0], [1]]]", "0"),
        ("[[[1], [1], [0]], [[1], [1], [1]]]", "1"),
        ("[[[1], [1], [1]], [[1], [1], [1]]]", "2"),
    ];

    for (arr_str, axis) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.all(np.array({arr_str}), axis={axis}).flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_bool_list(&numpy_result);

        let rust_script = fnp_all_script(format!(
            "print(fnp.all(np.array({arr_str}), axis={axis}).flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_bool_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "all 3D axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn all_integer_dtypes_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([1, 2, 3], dtype=np.int32)", "None"),
        ("np.array([1, 0, 3], dtype=np.int64)", "None"),
        ("np.array([1, 1, 1], dtype=np.uint8)", "None"),
        ("np.array([0, 0, 0], dtype=np.int16)", "None"),
        ("np.array([[1, 1], [1, 1]], dtype=np.int32)", "None"),
        ("np.array([[1, 0], [1, 1]], dtype=np.int64)", "None"),
        (
            "np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)",
            "None",
        ),
        (
            "np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float64)",
            "None",
        ),
    ];

    for (arr_expr, axis) in &test_cases {
        let axis_arg = if *axis == "None" {
            String::new()
        } else {
            format!(", axis={axis}")
        };
        let script = format!("import numpy as np; print(np.all({arr_expr}{axis_arg}))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_bool(&numpy_result);

        let rust_script = fnp_all_script(format!("print(fnp.all({arr_expr}{axis_arg}))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_bool(&rust_result);

        assert_eq!(
            numpy_val, rust_val,
            "all dtype mismatch for {arr_expr} axis={axis}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }

    Ok(())
}

#[test]
fn all_empty_array_matches_numpy() -> Result<(), String> {
    let test_cases = vec![("[]", "None")];

    for (arr_str, axis) in &test_cases {
        let axis_arg = if *axis == "None" {
            String::new()
        } else {
            format!(", axis={axis}")
        };
        let script = format!("import numpy as np; print(np.all(np.array({arr_str}){axis_arg}))");
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_all_script(format!("print(fnp.all(np.array({arr_str}){axis_arg}))"));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "all empty array mismatch for {arr_str} axis={axis}"
        );
    }

    Ok(())
}

#[test]
fn all_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_all_script(
        r#"
x = np.float64(5.0)
fnp_result = fnp.all(x)
np_result = np.all(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "all scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn all_complex_dtype_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        // All nonzero complex
        "[1+1j, 2-1j, 3+2j]",
        "[0.5+0.5j, 1.5-1.5j]",
        // Contains zero complex
        "[1+1j, 0+0j, 3+2j]",
        "[0+0j, 0+0j]",
        // Single element
        "[1+1j]",
        "[0+0j]",
    ];

    for arr_str in &test_cases {
        let script =
            format!("import numpy as np; print(np.all(np.array({arr_str}, dtype=np.complex128)))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_bool(&numpy_result);

        let rust_script = fnp_all_script(format!(
            "print(fnp.all(np.array({arr_str}, dtype=np.complex128)))"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_bool(&rust_result);

        assert_eq!(
            numpy_val, rust_val,
            "all complex mismatch for {arr_str}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }

    Ok(())
}

#[test]
fn all_special_values_match_numpy() -> Result<(), String> {
    let script = fnp_all_script(
        r#"
tests = []
# NaN is truthy (nonzero)
a = np.array([np.nan, 1.0, 2.0])
tests.append(fnp.all(a) == np.all(a))

# inf is truthy
a = np.array([np.inf, 1.0])
tests.append(fnp.all(a) == np.all(a))

# -inf is truthy
a = np.array([-np.inf, 1.0])
tests.append(fnp.all(a) == np.all(a))

# Mixed special values (all truthy)
a = np.array([np.nan, np.inf, -np.inf, 1.0])
tests.append(fnp.all(a) == np.all(a))

# Special value with zero (falsy)
a = np.array([np.nan, 0.0, 1.0])
tests.append(fnp.all(a) == np.all(a))

# Only nan
a = np.array([np.nan])
tests.append(fnp.all(a) == np.all(a))

# Only inf
a = np.array([np.inf])
tests.append(fnp.all(a) == np.all(a))

print(all(tests))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "all special values should match numpy"
    );
    Ok(())
}

#[test]
fn all_negative_zero_matches_numpy() -> Result<(), String> {
    let script = fnp_all_script(
        r#"
# -0.0 is falsy (zero)
a = np.array([-0.0])
fnp_result = fnp.all(a)
np_result = np.all(a)
print(fnp_result == np_result and fnp_result == False)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "all negative zero should match numpy"
    );
    Ok(())
}
