//! Conformance tests for numpy.var against NumPy oracle.
//!
//! Tests the native Rust var implementation against NumPy across various
//! input shapes, axis parameters, keepdims, ddof, and data types.

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

fn fnp_var_script(body: String) -> String {
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

fn indent_python(body: &str) -> String {
    body.lines().map(|line| format!("    {line}\n")).collect()
}

fn var_outcome_body(body: &str) -> String {
    let indented = indent_python(body);
    r#"import json

def normalize(value):
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [normalize(item) for item in value]}
    if isinstance(value, np.ndarray):
        return {
            "kind": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if np.isscalar(value):
        scalar_type = type(value).__name__
        scalar_dtype = str(value.dtype) if hasattr(value, "dtype") else None
        scalar_value = value.item() if hasattr(value, "item") else value
        return {
            "kind": "scalar",
            "type": scalar_type,
            "dtype": scalar_dtype,
            "value": scalar_value,
        }
    return {"kind": "object", "type": type(value).__name__, "repr": repr(value)}

try:
__BODY__    payload = {"status": "ok", "result": normalize(result)}
    if "out" in locals():
        payload["out"] = normalize(out)
        payload["result_is_out"] = result is out
    print(json.dumps(payload, sort_keys=True, default=str))
except Exception as exc:
    message = str(exc).splitlines()[0] if str(exc) else ""
    print(json.dumps(
        {"status": "err", "type": type(exc).__name__, "message": message},
        sort_keys=True,
        default=str,
    ))
"#
    .replace("__BODY__", &indented)
}

fn numpy_var_outcome_script(body: &str) -> String {
    format!(
        "import numpy as np\n\
         MODULE = np\n\
         {}",
        var_outcome_body(body)
    )
}

fn fnp_var_outcome_script(body: &str) -> String {
    fnp_var_script(format!("MODULE = fnp\n{}", var_outcome_body(body)))
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
fn var_python_container_keyword_outcomes_match_numpy() -> Result<(), String> {
    let cases = [
        (
            "list input scalar",
            "result = MODULE.var([1, 2, 3])",
        ),
        (
            "tuple input axis keepdims",
            "result = MODULE.var(((1, 2, 3), (4, 5, 6)), axis=1, keepdims=True)",
        ),
        (
            "dtype keyword",
            "result = MODULE.var(np.array([1, 2, 3], dtype=np.int16), dtype=np.float32)",
        ),
        (
            "ddof keyword",
            "result = MODULE.var([1.0, 2.0, 3.0, 4.0], ddof=1)",
        ),
        (
            "where keyword",
            "result = MODULE.var(
    np.array([[1.0, 2.0], [3.0, 4.0]]),
    where=np.array([[True, False], [False, True]]),
)",
        ),
        (
            "out forwarding",
            "out = np.empty((2,), dtype=np.float64)
result = MODULE.var(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=0, out=out)",
        ),
        (
            "axis error type",
            "result = MODULE.var([1, 2, 3], axis=2)",
        ),
    ];

    for (name, body) in cases {
        let numpy_result = numpy_oracle(&numpy_var_outcome_script(body))?;
        let fnp_result = numpy_oracle(&fnp_var_outcome_script(body))?;

        assert_eq!(
            fnp_result, numpy_result,
            "var outcome mismatch for {name}\nnumpy: {numpy_result}\nfnp:   {fnp_result}"
        );
    }
    Ok(())
}

#[test]
fn var_flat_matches_numpy_across_50_cases() -> Result<(), String> {
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
        let script = format!("import numpy as np; print(np.var(np.array({arr_str})))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_float(&numpy_result);

        let rust_script = fnp_var_script(format!("print(fnp.var(np.array({arr_str})))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_float(&rust_result);

        assert!(
            floats_close(numpy_val, rust_val, 1e-9),
            "var flat mismatch for {arr_str}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }
    Ok(())
}

#[test]
fn var_2d_axis_matches_numpy() -> Result<(), String> {
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
            format!("import numpy as np; print(np.var(np.array({arr_str}), axis={axis}).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = fnp_var_script(format!(
            "print(fnp.var(np.array({arr_str}), axis={axis}).tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-9),
            "var axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
    Ok(())
}

#[test]
fn var_3d_axis_matches_numpy() -> Result<(), String> {
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
            "import numpy as np; print(np.var(np.array({arr_str}), axis={axis}).flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = fnp_var_script(format!(
            "print(fnp.var(np.array({arr_str}), axis={axis}).flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-9),
            "var 3D axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
    Ok(())
}

#[test]
fn var_keepdims_matches_numpy() -> Result<(), String> {
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
            "import numpy as np; print(np.var(np.array({arr_str}){axis_arg}, keepdims={}).shape)",
            if *keepdims { "True" } else { "False" }
        );
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_var_script(format!(
            "print(fnp.var(np.array({arr_str}){axis_arg}, keepdims={}).shape)",
            if *keepdims { "True" } else { "False" }
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "var keepdims={keepdims} shape mismatch for {arr_str} axis={axis}"
        );
    }
    Ok(())
}

#[test]
fn var_ddof_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        // Basic ddof=0 (default)
        ("[1, 2, 3, 4, 5]", 0),
        // ddof=1 (sample variance)
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
        let script = format!("import numpy as np; print(np.var(np.array({arr_str}), ddof={ddof}))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_float(&numpy_result);

        let rust_script =
            fnp_var_script(format!("print(fnp.var(np.array({arr_str}), ddof={ddof}))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_float(&rust_result);

        assert!(
            floats_close(numpy_val, rust_val, 1e-9),
            "var ddof={ddof} mismatch for {arr_str}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }
    Ok(())
}

#[test]
fn var_non_integral_ddof_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("[1.0, 2.0, 3.0]", "-1"),
        ("[1.0, 2.0, 3.0]", "0.5"),
        ("[1.0, 2.0, 3.0]", "1.5"),
        ("[2.0, 4.0, 8.0, 16.0]", "np.float64(0.5)"),
    ];

    for (arr_str, ddof) in &test_cases {
        let script = format!("import numpy as np; print(np.var(np.array({arr_str}), ddof={ddof}))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_float(&numpy_result);

        let rust_script =
            fnp_var_script(format!("print(fnp.var(np.array({arr_str}), ddof={ddof}))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_float(&rust_result);

        assert!(
            floats_close(numpy_val, rust_val, 1e-9),
            "var ddof={ddof} mismatch for {arr_str}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }
    Ok(())
}

#[test]
fn var_integer_dtypes_match_numpy() -> Result<(), String> {
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
        let script = format!("import numpy as np; print(float(np.var({arr_expr}{axis_arg})))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_float(&numpy_result);

        let rust_script = fnp_var_script(format!("print(float(fnp.var({arr_expr}{axis_arg})))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_float(&rust_result);

        assert!(
            floats_close(numpy_val, rust_val, 1e-6),
            "var dtype mismatch for {arr_expr} axis={axis}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }
    Ok(())
}

#[test]
fn var_nan_handling_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "[1.0, np.nan, 3.0]",
        "[np.nan, 2.0, 3.0]",
        "[1.0, 2.0, np.nan]",
        "[np.nan, np.nan, np.nan]",
        "[1.0, np.nan, np.nan, 4.0]",
    ];

    for arr_str in &test_cases {
        let script = format!("import numpy as np; print(np.var(np.array({arr_str})))");
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_var_script(format!("print(fnp.var(np.array({arr_str})))"));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "var NaN mismatch for {arr_str}"
        );
    }
    Ok(())
}

#[test]
fn var_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_var_script(
        r#"
x = np.float64(5.0)
fnp_result = fnp.var(x)
np_result = np.var(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "var scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn var_complex() -> Result<(), String> {
    let script = fnp_var_script(
        r#"
z = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex128)
fnp_result = fnp.var(z)
np_result = np.var(z)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "var complex should match numpy");
    Ok(())
}

#[test]
fn var_inf_handling_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "[1.0, np.inf, 3.0]",
        "[-np.inf, 2.0, 3.0]",
        "[1.0, np.inf, -np.inf]",
        "[np.inf, np.inf, np.inf]",
    ];

    for arr_str in &test_cases {
        let script = format!("import numpy as np; print(np.var(np.array({arr_str})))");
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_var_script(format!("print(fnp.var(np.array({arr_str})))"));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "var inf mismatch for {arr_str}"
        );
    }
    Ok(())
}

#[test]
fn var_out_parameter_matches_numpy() -> Result<(), String> {
    let script = fnp_var_script(
        r#"
a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
fnp_out = np.empty((3,))
np_out = np.empty((3,))
fnp.var(a, axis=0, out=fnp_out)
np.var(a, axis=0, out=np_out)
print(np.allclose(fnp_out, np_out))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "var out parameter should match numpy"
    );
    Ok(())
}

#[test]
fn var_empty_array_returns_nan() -> Result<(), String> {
    let script = fnp_var_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
empty = np.array([])
fnp_result = fnp.var(empty)
np_result = np.var(empty)
print(np.isnan(fnp_result) and np.isnan(np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "var of empty array should return nan"
    );
    Ok(())
}

#[test]
fn var_single_element_is_zero() -> Result<(), String> {
    let script = fnp_var_script(
        r#"
single = np.array([5.0])
fnp_result = fnp.var(single)
np_result = np.var(single)
print(fnp_result == np_result == 0.0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "var of single element should be 0");
    Ok(())
}

#[test]
fn var_std_multiaxis_trailing_matches_numpy() -> Result<(), String> {
    // Exercises the native multi-axis trailing var/std fold (axis a tuple resolving
    // to the contiguous trailing axes) against numpy bit-exactly (atol=0,
    // equal_nan=True) incl dtype/shape: var and std, ddof 0/1, keepdims, reversed
    // axis order (variance is symmetric), 2-D axis=(0,1), a non-trailing axis
    // fallthrough, and a NaN block (which must defer + match numpy's NaN).
    let script = fnp_var_script(
        r#"
def same(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.allclose(a, b, rtol=0, atol=0, equal_nan=True)

s3 = np.linspace(-4.0, 6.0, 4 * 5 * 6, dtype=np.float64).reshape(4, 5, 6)
s4 = np.linspace(-2.0, 3.0, 2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 3, 4, 5)
m2 = np.linspace(-1.0, 2.0, 7 * 8, dtype=np.float64).reshape(7, 8)
nanblk = np.array([[[1.0, np.nan], [3.0, 4.0]], [[1.0, 2.0], [2.0, 5.0]]], dtype=np.float64)
ok = True
cases = [
    (s3, (-2, -1), 0, False, False),
    (s3, (-2, -1), 1, True, False),
    (s3, (-1, -2), 0, False, False),
    (s4, (-3, -2, -1), 0, False, False),
    (s4, (-2, -1), 0, False, True),
    (m2, (0, 1), 1, False, False),
    (s3, (0, 1), 0, False, False),
    (nanblk, (-2, -1), 0, False, False),
]
for arr, axis, ddof, keepdims, use_std in cases:
    if use_std:
        f = fnp.std(arr, axis=axis, ddof=ddof, keepdims=keepdims)
        n = np.std(arr, axis=axis, ddof=ddof, keepdims=keepdims)
    else:
        f = fnp.var(arr, axis=axis, ddof=ddof, keepdims=keepdims)
        n = np.var(arr, axis=axis, ddof=ddof, keepdims=keepdims)
    if not same(f, n):
        print("FAIL", axis, ddof, keepdims, use_std, np.asarray(f), np.asarray(n)); ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "multi-axis trailing var/std parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn var_std_axis0_first_axis_matches_numpy() -> Result<(), String> {
    // Exercises the native first-axis (axis=0) streaming two-pass var/std fold against
    // numpy bit-exactly (atol=0, equal_nan=True) incl dtype/shape: var and std, ddof
    // 0/1, keepdims, negative axis index, 3-D axis=0, NaN/Inf columns (propagate, no
    // defer), an M<=ddof defer case (numpy NaN + warning), and a wide/tall mix.
    let script = fnp_var_script(
        r#"
import warnings
def same(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.allclose(a, b, rtol=0, atol=0, equal_nan=True)

rng = np.random.default_rng(17)
m2 = rng.standard_normal((1000, 257))
tall = rng.standard_normal((50000, 16))
wide = rng.standard_normal((97, 1024))
s3 = rng.standard_normal((64, 9, 7))
nanm = rng.standard_normal((40, 8)); nanm[3, 2] = np.nan; nanm[10, 5] = np.inf
small = rng.standard_normal((1, 6))  # M=1: var ddof=1 -> numpy NaN + warning (defer)
ok = True
cases = [
    (m2, 0, 0, False, False),
    (m2, 0, 1, True, False),
    (m2, -2, 0, False, False),
    (tall, 0, 0, False, True),
    (wide, 0, 1, False, False),
    (s3, 0, 0, False, False),
    (nanm, 0, 0, False, False),
    (small, 0, 1, False, False),
]
for arr, axis, ddof, keepdims, use_std in cases:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if use_std:
            f = fnp.std(arr, axis=axis, ddof=ddof, keepdims=keepdims)
            n = np.std(arr, axis=axis, ddof=ddof, keepdims=keepdims)
        else:
            f = fnp.var(arr, axis=axis, ddof=ddof, keepdims=keepdims)
            n = np.var(arr, axis=axis, ddof=ddof, keepdims=keepdims)
    if not same(f, n):
        print("FAIL", axis, ddof, keepdims, use_std, np.asarray(f), np.asarray(n)); ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "axis=0 first-axis var/std parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn int_var_std_via_f64_conversion_bit_exact_matches_numpy() -> Result<(), String> {
    // Integer/bool var and std convert once to f64 and ride the whole f64
    // kernel family; numpy sums with dtype=f64 (elements convert BEFORE any
    // arithmetic), so the conversion is byte-exact UNCONDITIONALLY - pinned
    // here including 2^62-scale values. dtype= overrides and small inputs
    // keep the delegate.
    let script = fnp_var_script(
        r#"
import time
rng = np.random.default_rng(173)
verdicts = []
M = rng.integers(-1000, 1000, (2048, 1024))
for fname in ("var", "std"):
    ff = getattr(fnp, fname); nf = getattr(np, fname)
    for kw in [dict(), dict(axis=1), dict(axis=0), dict(axis=-1), dict(axis=1, ddof=1), dict(axis=0, keepdims=True)]:
        r = ff(M, **kw); e = nf(M, **kw)
        ra = np.asarray(r, dtype=np.float64); ea = np.asarray(e, dtype=np.float64)
        if ra.shape != ea.shape or ra.tobytes() != ea.tobytes():
            verdicts.append(f"FAIL {fname} {kw}")
# 3-D middle axis + widths + bool
M3 = rng.integers(-100, 100, (64, 256, 256)).astype(np.int32)
if np.asarray(fnp.var(M3, axis=1)).tobytes() != np.asarray(np.var(M3, axis=1)).tobytes():
    verdicts.append("FAIL int32 3-D mid-axis")
B = rng.random((2048, 1024)) > 0.7
if np.asarray(fnp.std(B, axis=1)).tobytes() != np.asarray(np.std(B, axis=1)).tobytes():
    verdicts.append("FAIL bool std")
# HUGE values: conversion is unconditional (numpy converts pre-arithmetic too)
H = rng.integers(-2**62, 2**62, (512, 1024))
if np.asarray(fnp.var(H, axis=1)).tobytes() != np.asarray(np.var(H, axis=1)).tobytes():
    verdicts.append("FAIL huge-value bytes")
# dtype= override + small inputs keep the delegate
if np.asarray(fnp.var(M, axis=1, dtype=np.float32)).tobytes() != np.asarray(np.var(M, axis=1, dtype=np.float32)).tobytes():
    verdicts.append("FAIL dtype-override delegate")
S = rng.integers(-100, 100, (10, 10))
if np.asarray(fnp.var(S, axis=1)).tobytes() != np.asarray(np.var(S, axis=1)).tobytes():
    verdicts.append("FAIL small delegate")

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

W = rng.integers(-1000, 1000, (4096, 4096))
tn = best(lambda: np.var(W, axis=1)); tf = best(lambda: fnp.var(W, axis=1))
print(f"VAR_INT_AX1_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
tn = best(lambda: np.std(W, axis=0)); tf = best(lambda: fnp.std(W, axis=0))
print(f"STD_INT_AX0_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces VAR/STD_INT_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "int var/std via f64 conversion must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn int_nanvar_nanstd_route_to_var_std_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy's nanvar/nanstd short-circuit non-float dtypes straight to
    // var/std; fnp now mirrors that routing for int/bool (whose var/std int
    // conversion arm applies). Extra kwargs keep the delegate.
    let script = fnp_var_script(
        r#"
import time
rng = np.random.default_rng(181)
verdicts = []
M = rng.integers(-1000, 1000, (2048, 1024))
for fname in ("nanvar", "nanstd"):
    ff = getattr(fnp, fname); nf = getattr(np, fname)
    for kw in [dict(), dict(axis=1), dict(axis=0), dict(axis=1, ddof=1), dict(axis=0, keepdims=True)]:
        r = ff(M, **kw); e = nf(M, **kw)
        ra = np.asarray(r, dtype=np.float64); ea = np.asarray(e, dtype=np.float64)
        if ra.shape != ea.shape or ra.tobytes() != ea.tobytes():
            verdicts.append(f"FAIL {fname} {kw}")
B = rng.random((2048, 1024)) > 0.7
if np.asarray(fnp.nanstd(B, axis=1)).tobytes() != np.asarray(np.nanstd(B, axis=1)).tobytes():
    verdicts.append("FAIL bool nanstd")
H = rng.integers(-2**62, 2**62, (512, 1024))
if np.asarray(fnp.nanvar(H, axis=1)).tobytes() != np.asarray(np.nanvar(H, axis=1)).tobytes():
    verdicts.append("FAIL huge-value nanvar")
# dtype override keeps delegate
if np.asarray(fnp.nanvar(M, axis=1, dtype=np.float32)).tobytes() != np.asarray(np.nanvar(M, axis=1, dtype=np.float32)).tobytes():
    verdicts.append("FAIL dtype-override delegate")

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

W = rng.integers(-1000, 1000, (4096, 4096))
tn = best(lambda: np.nanvar(W, axis=1)); tf = best(lambda: fnp.nanvar(W, axis=1))
print(f"NANVAR_INT_AX1_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces NANVAR_INT_AX1_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "int nanvar/nanstd routing must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
#[ignore = "PARITY GAP (ISA-dependent): flat f64 nansum/var/std byte parity vs numpy depends on the WORKER's numpy SIMD build - one gate worker read all-True at every size, another read False from n=131072 (nansum) / ~2M + all N-D flats (var/std). numpy's pairwise-sum leaf structure varies with vector width (AVX-512 vs AVX2 partial accumulators), so base_sum_simd matches one ISA and sub-ULP-diverges on others - the transcendental ISA-divergence class, sum edition. Fix requires the ISA-gate treatment (worker_isa_probe grid); see dtype-gap-audit memory 2026-07-13."]
fn f64_var_flat_byte_parity_probe_vs_numpy() -> Result<(), String> {
    // PROBE preserved for the ISA-gate investigation: nansum rows isolate the
    // shared pairwise sum core; var/std rows add the sqr-dev pass; 2-D/3-D
    // rows cover flattened N-D routing.
    let script = fnp_var_script(
        r#"
rng = np.random.default_rng(241)
rows = []
for n in [7, 100, 128, 129, 1000, 4096, 131072, 2_000_000, 2_097_152, 4_000_001]:
    a = rng.standard_normal(n) * 7
    s_ok = np.float64(fnp.nansum(a)).tobytes() == np.float64(np.nansum(a)).tobytes()
    v_ok = np.float64(fnp.var(a)).tobytes() == np.float64(np.var(a)).tobytes()
    sd_ok = np.float64(fnp.std(a)).tobytes() == np.float64(np.std(a)).tobytes()
    rows.append((n, s_ok, v_ok, sd_ok))
# 2-D CONTIGUOUS flat (axis=None) f64: distinguishes kernel-vs-routing -
# the converted-int gate failure was a 2-D flat input
M2 = rng.standard_normal((2048, 1024)) * 7
rows.append(("2Dflat", np.float64(fnp.nansum(M2)).tobytes() == np.float64(np.nansum(M2)).tobytes(),
             np.float64(fnp.var(M2)).tobytes() == np.float64(np.var(M2)).tobytes(),
             np.float64(fnp.std(M2)).tobytes() == np.float64(np.std(M2)).tobytes()))
M3 = rng.standard_normal((64, 128, 128))
rows.append(("3Dflat", True,
             np.float64(fnp.var(M3)).tobytes() == np.float64(np.var(M3)).tobytes(),
             np.float64(fnp.std(M3)).tobytes() == np.float64(np.std(M3)).tobytes()))
for r in rows:
    print("PROBE", r)
bad = [r for r in rows if not (r[1] and r[2] and r[3])]
print([] if not bad else bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}");
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "[]",
        "flat f64 var/std/nansum must be bit-identical to numpy: {result}"
    );
    Ok(())
}
