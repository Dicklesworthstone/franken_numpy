//! Conformance tests for numpy.sum against NumPy oracle.
//!
//! Tests the native Rust sum implementation against NumPy across various
//! input shapes, axis parameters, keepdims, and data types.

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

fn fnp_sum_script(body: String) -> String {
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
fn sum_flat_matches_numpy_across_50_cases() -> Result<(), String> {
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
        let script = format!("import numpy as np; print(np.sum(np.array({arr_str})))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_float(&numpy_result);

        let rust_script = fnp_sum_script(format!("print(fnp.sum(np.array({arr_str})))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_float(&rust_result);

        assert!(
            floats_close(numpy_val, rust_val, 1e-9),
            "sum flat mismatch for {arr_str}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }
    Ok(())
}

#[test]
fn sum_2d_axis_matches_numpy() -> Result<(), String> {
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
            format!("import numpy as np; print(np.sum(np.array({arr_str}), axis={axis}).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = fnp_sum_script(format!(
            "print(fnp.sum(np.array({arr_str}), axis={axis}).tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-9),
            "sum axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
    Ok(())
}

#[test]
fn sum_3d_axis_matches_numpy() -> Result<(), String> {
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
            "import numpy as np; print(np.sum(np.array({arr_str}), axis={axis}).flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = fnp_sum_script(format!(
            "print(fnp.sum(np.array({arr_str}), axis={axis}).flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-9),
            "sum 3D axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
    Ok(())
}

#[test]
fn sum_keepdims_matches_numpy() -> Result<(), String> {
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
            "import numpy as np; print(np.sum(np.array({arr_str}){axis_arg}, keepdims={}).shape)",
            if *keepdims { "True" } else { "False" }
        );
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_sum_script(format!(
            "print(fnp.sum(np.array({arr_str}){axis_arg}, keepdims={}).shape)",
            if *keepdims { "True" } else { "False" }
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "sum keepdims={keepdims} shape mismatch for {arr_str} axis={axis}"
        );
    }
    Ok(())
}

#[test]
fn sum_unknown_keyword_matches_numpy_error() -> Result<(), String> {
    let numpy_script = r#"import numpy as np
try:
    print(np.sum(np.array([1, 2, 3]), unexpected_kw=1))
except Exception as exc:
    print(f'{type(exc).__name__}:{exc}')"#;
    let numpy_result = numpy_oracle(numpy_script)?;

    let rust_script = fnp_sum_script(
        r#"try:
    print(fnp.sum(np.array([1, 2, 3]), unexpected_kw=1))
except Exception as exc:
    print(f'{type(exc).__name__}:{exc}')"#
            .to_string(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert!(
        numpy_result.starts_with("TypeError:"),
        "NumPy should reject unknown sum keyword, got {numpy_result}"
    );
    assert_eq!(numpy_result, rust_result);
    Ok(())
}

#[test]
fn sum_integer_dtypes_match_numpy() -> Result<(), String> {
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
        let script = format!("import numpy as np; print(float(np.sum({arr_expr}{axis_arg})))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_float(&numpy_result);

        let rust_script = fnp_sum_script(format!("print(float(fnp.sum({arr_expr}{axis_arg})))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_float(&rust_result);

        assert!(
            floats_close(numpy_val, rust_val, 1e-6),
            "sum dtype mismatch for {arr_expr} axis={axis}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }
    Ok(())
}

#[test]
fn sum_nan_handling_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "[1.0, np.nan, 3.0]",
        "[np.nan, 2.0, 3.0]",
        "[1.0, 2.0, np.nan]",
        "[np.nan, np.nan, np.nan]",
        "[1.0, np.nan, np.nan, 4.0]",
    ];

    for arr_str in &test_cases {
        let script = format!("import numpy as np; print(np.sum(np.array({arr_str})))");
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_sum_script(format!("print(fnp.sum(np.array({arr_str})))"));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "sum NaN mismatch for {arr_str}"
        );
    }
    Ok(())
}

#[test]
fn sum_empty_array_matches_numpy() -> Result<(), String> {
    let test_cases = vec![("[]", "None"), ("[[]]", "None")];

    for (arr_str, axis) in &test_cases {
        let axis_arg = if *axis == "None" {
            String::new()
        } else {
            format!(", axis={axis}")
        };
        let script =
            format!("import numpy as np; print(float(np.sum(np.array({arr_str}){axis_arg})))");
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_sum_script(format!(
            "print(float(fnp.sum(np.array({arr_str}){axis_arg})))"
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "sum empty array mismatch for {arr_str} axis={axis}"
        );
    }
    Ok(())
}

#[test]
fn sum_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
x = np.float64(5.0)
fnp_result = fnp.sum(x)
np_result = np.sum(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "sum scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn sum_complex() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
z = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex128)
fnp_result = fnp.sum(z)
np_result = np.sum(z)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sum complex should match numpy");
    Ok(())
}

#[test]
fn sum_complex_axis() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
z = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
fnp_result = fnp.sum(z, axis=0)
np_result = np.sum(z, axis=0)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sum complex axis=0 should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Error behavior tests
// ─────────────────────────────────────────────────────────────────────────────

fn classify_error(script: &str) -> String {
    let output = std::process::Command::new("python3")
        .args(["-c", script])
        .output()
        .expect("python3 should be available");
    if output.status.success() {
        "ok".to_string()
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("AxisError") || stderr.contains("axis") {
            "AxisError".to_string()
        } else if stderr.contains("ValueError") {
            "ValueError".to_string()
        } else {
            format!("other: {}", stderr.lines().last().unwrap_or(""))
        }
    }
}

#[test]
fn sum_axis_out_of_bounds_raises_axiserror() {
    let fnp_err = classify_error(&fnp_sum_script(
        r#"
a = fnp.arange(12).reshape(3, 4)
fnp.sum(a, axis=5)
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(12).reshape(3, 4)
np.sum(a, axis=5)
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "sum with out-of-bounds axis should raise same error as numpy"
    );
}

#[test]
fn sum_inf_handling_matches_numpy() -> Result<(), String> {
    let inf_cases = [
        "[1.0, np.inf, 3.0]",
        "[np.inf, 2.0, 3.0]",
        "[-np.inf, np.inf]",
        "[np.inf, np.inf]",
        "[-np.inf, -np.inf]",
    ];

    for arr_str in &inf_cases {
        let np_script =
            format!("import numpy as np; print(repr(np.sum(np.array({arr_str}))))");
        let np_output = numpy_oracle(&np_script)?;

        let fnp_script = fnp_sum_script(format!(
            "print(repr(fnp.sum(np.array({arr_str}))))"
        ));
        let fnp_output = numpy_oracle(&fnp_script)?;

        assert_eq!(
            fnp_output.trim(),
            np_output.trim(),
            "sum inf mismatch for {arr_str}"
        );
    }
    Ok(())
}

#[test]
fn sum_with_out_parameter() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
a = np.array([[1, 2], [3, 4]])
out = np.empty((2,), dtype=np.int64)
fnp_result = fnp.sum(a, axis=0, out=out)
np_out = np.empty((2,), dtype=np.int64)
np_result = np.sum(a, axis=0, out=np_out)
# Check both result and that out was modified
print(np.array_equal(fnp_result, np_result) and np.array_equal(out, np_out))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sum with out parameter should match numpy");
    Ok(())
}

#[test]
fn sum_with_where_parameter() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
mask = np.array([True, False, True, False, True])
fnp_result = fnp.sum(a, where=mask)
np_result = np.sum(a, where=mask)
print(fnp_result == np_result == 9)  # 1 + 3 + 5
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sum with where parameter should match numpy");
    Ok(())
}

#[test]
fn sum_with_initial_parameter() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
fnp_result = fnp.sum(a, initial=10)
np_result = np.sum(a, initial=10)
print(fnp_result == np_result == 25)  # 10 + 1+2+3+4+5
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sum with initial parameter should match numpy");
    Ok(())
}

#[test]
fn sum_signed_zero_parity() -> Result<(), String> {
    // Test signed-zero behavior for parallel operation safety proofs.
    // IEEE 754: 0.0 + 0.0 = 0.0, -0.0 + -0.0 = -0.0, 0.0 + -0.0 = 0.0
    let script = fnp_sum_script(
        r#"
# Signed-zero sum semantics
tests = [
    ([0.0, 0.0], False),      # 0.0 + 0.0 = 0.0 (positive)
    ([-0.0, -0.0], True),     # -0.0 + -0.0 = -0.0 (negative)
    ([0.0, -0.0], False),     # 0.0 + -0.0 = 0.0 (positive - IEEE 754 rule)
    ([-0.0, 0.0], False),     # -0.0 + 0.0 = 0.0 (positive)
    ([-0.0, -0.0, -0.0], True), # Multiple -0.0 sum
]
all_pass = True
for values, expected_signbit in tests:
    arr = np.array(values)
    fnp_result = fnp.sum(arr)
    np_result = np.sum(arr)
    if np.signbit(fnp_result) != np.signbit(np_result):
        print(f"FAIL: sum({values}) fnp signbit={np.signbit(fnp_result)} np signbit={np.signbit(np_result)}")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sum signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn sum_accumulation_stability() -> Result<(), String> {
    // Test that sum accumulation order matches NumPy
    let script = fnp_sum_script(
        r#"
# Large values that could suffer from accumulation order issues
a = np.array([1e16, 1.0, -1e16])
fnp_result = fnp.sum(a)
np_result = np.sum(a)

# Also test with axis reduction
b = np.array([[1e16, 1.0], [-1e16, 2.0]])
fnp_axis = fnp.sum(b, axis=0)
np_axis = np.sum(b, axis=0)

axis_match = np.allclose(fnp_axis, np_axis)
scalar_match = np.isclose(fnp_result, np_result) or (fnp_result == np_result)
print(scalar_match and axis_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sum accumulation stability should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn sum_negative_axis() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
fnp_result_m1 = fnp.sum(a, axis=-1)
np_result_m1 = np.sum(a, axis=-1)
fnp_result_m2 = fnp.sum(a, axis=-2)
np_result_m2 = np.sum(a, axis=-2)
print(np.array_equal(fnp_result_m1, np_result_m1) and np.array_equal(fnp_result_m2, np_result_m2))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sum with negative axis should match numpy");
    Ok(())
}

#[test]
fn sum_tuple_axis() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
fnp_result_02 = fnp.sum(a, axis=(0, 2))
np_result_02 = np.sum(a, axis=(0, 2))
fnp_result_12 = fnp.sum(a, axis=(1, 2))
np_result_12 = np.sum(a, axis=(1, 2))
print(
    fnp_result_02.shape == np_result_02.shape,
    fnp_result_12.shape == np_result_12.shape,
    np.array_equal(fnp_result_02, np_result_02),
    np.array_equal(fnp_result_12, np_result_12)
)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True True True True"),
        "sum with tuple axis should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn sum_axis_none_flatten() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
fnp_result = fnp.sum(a, axis=None)
np_result = np.sum(a, axis=None)
print(fnp_result == np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sum with axis=None should flatten and match numpy");
    Ok(())
}
