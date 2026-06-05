//! Conformance tests for numpy comparison operations against NumPy oracle.
//!
//! Tests equal, not_equal, less, less_equal, greater, greater_equal.

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

fn fnp_script(body: String) -> String {
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

#[test]
fn equal_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1, 2, 3, 4])
x2 = np.array([1, 3, 3, 5])
result = fnp.equal(x1, x2)
expected = np.equal(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "equal basic should match numpy");
    Ok(())
}

#[test]
fn not_equal_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1, 2, 3, 4])
x2 = np.array([1, 3, 3, 5])
result = fnp.not_equal(x1, x2)
expected = np.not_equal(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "not_equal basic should match numpy");
    Ok(())
}

#[test]
fn less_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1, 2, 3, 4])
x2 = np.array([2, 2, 2, 2])
result = fnp.less(x1, x2)
expected = np.less(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "less basic should match numpy");
    Ok(())
}

#[test]
fn less_equal_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1, 2, 3, 4])
x2 = np.array([2, 2, 2, 2])
result = fnp.less_equal(x1, x2)
expected = np.less_equal(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "less_equal basic should match numpy");
    Ok(())
}

#[test]
fn greater_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1, 2, 3, 4])
x2 = np.array([2, 2, 2, 2])
result = fnp.greater(x1, x2)
expected = np.greater(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "greater basic should match numpy");
    Ok(())
}

#[test]
fn greater_equal_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1, 2, 3, 4])
x2 = np.array([2, 2, 2, 2])
result = fnp.greater_equal(x1, x2)
expected = np.greater_equal(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "greater_equal basic should match numpy"
    );
    Ok(())
}

#[test]
fn equal_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(3.0)
x2 = np.float64(3.0)
fnp_result = fnp.equal(x1, x2)
np_result = np.equal(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "equal scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn less_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(2.0)
x2 = np.float64(3.0)
fnp_result = fnp.less(x1, x2)
np_result = np.less(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "less scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn equal_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z1 = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
z2 = np.array([1+1j, 2+3j, 3+3j], dtype=np.complex128)
fnp_result = fnp.equal(z1, z2)
np_result = np.equal(z1, z2)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "equal complex should match numpy");
    Ok(())
}

#[test]
fn not_equal_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z1 = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
z2 = np.array([1+1j, 2+3j, 3+3j], dtype=np.complex128)
fnp_result = fnp.not_equal(z1, z2)
np_result = np.not_equal(z1, z2)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "not_equal complex should match numpy"
    );
    Ok(())
}

#[test]
fn comparison_with_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1.0, np.nan, np.nan, 2.0])
x2 = np.array([1.0, np.nan, 2.0, np.nan])
tests_pass = True
for func_name in ['equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(x1, x2)
    np_result = np_func(x1, x2)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "comparison with nan should match numpy"
    );
    Ok(())
}

#[test]
fn comparison_with_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1.0, np.inf, -np.inf, np.inf])
x2 = np.array([np.inf, np.inf, -np.inf, -np.inf])
tests_pass = True
for func_name in ['equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(x1, x2)
    np_result = np_func(x1, x2)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "comparison with inf should match numpy"
    );
    Ok(())
}

#[test]
fn equal_negative_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([0.0, -0.0, 0.0])
x2 = np.array([-0.0, 0.0, 0.0])
fnp_result = fnp.equal(x1, x2)
np_result = np.equal(x1, x2)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "equal with negative zero should match numpy"
    );
    Ok(())
}

#[test]
fn comparison_broadcasting() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([[1, 2, 3], [4, 5, 6]])
x2 = np.array([2, 2, 2])
tests_pass = True
for func_name in ['equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(x1, x2)
    np_result = np_func(x1, x2)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "comparison broadcasting should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge case tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn comparison_empty_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([], dtype=np.float64)
x2 = np.array([], dtype=np.float64)
tests_pass = True
for func_name in ['equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(x1, x2)
    np_result = np_func(x1, x2)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
    tests_pass = tests_pass and (fnp_result.shape == np_result.shape)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "comparison empty arrays should match numpy"
    );
    Ok(())
}

#[test]
fn comparison_single_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([2.0])
x2 = np.array([3.0])
tests_pass = True
for func_name in ['equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(x1, x2)
    np_result = np_func(x1, x2)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "comparison single element should match numpy"
    );
    Ok(())
}

#[test]
fn comparison_signed_zeros() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([0.0, -0.0, 0.0, -0.0])
x2 = np.array([0.0, 0.0, -0.0, -0.0])
tests_pass = True
for func_name in ['equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(x1, x2)
    np_result = np_func(x1, x2)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "comparison signed zeros should match numpy"
    );
    Ok(())
}

#[test]
fn comparison_mixed_dtypes() -> Result<(), String> {
    let script = fnp_script(
        r#"
x_int = np.array([1, 2, 3, 4])
x_float = np.array([1.0, 2.5, 3.0, 3.5])
tests_pass = True
for func_name in ['equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(x_int, x_float)
    np_result = np_func(x_int, x_float)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "comparison mixed dtypes should match numpy"
    );
    Ok(())
}

#[test]
fn comparison_boolean_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([True, True, False, False])
x2 = np.array([True, False, True, False])
tests_pass = True
for func_name in ['equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(x1, x2)
    np_result = np_func(x1, x2)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "comparison boolean arrays should match numpy"
    );
    Ok(())
}

#[test]
fn comparison_nan_vs_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([np.nan, np.nan, np.inf, -np.inf])
x2 = np.array([np.inf, -np.inf, np.nan, np.nan])
tests_pass = True
for func_name in ['equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(x1, x2)
    np_result = np_func(x1, x2)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "comparison nan vs inf should match numpy"
    );
    Ok(())
}

#[test]
fn comparison_subnormal_numbers() -> Result<(), String> {
    let script = fnp_script(
        r#"
import sys
tiny = sys.float_info.min
subnormal = tiny / 2.0
x1 = np.array([subnormal, subnormal, tiny, 0.0])
x2 = np.array([0.0, tiny, subnormal, subnormal])
tests_pass = True
for func_name in ['equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(x1, x2)
    np_result = np_func(x1, x2)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "comparison subnormal numbers should match numpy"
    );
    Ok(())
}
