//! Conformance tests for comparison and logical operations.
//!
//! Tests: equal, not_equal, greater, greater_equal, less, less_equal,
//! logical_and, logical_or, logical_xor, logical_not.

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

// ─────────────────────────────────────────────────────────────────────────────
// equal
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn equal_basic_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([1.0, 3.0, 3.0, 5.0])
result = fnp.equal(x, y)
expected = np.equal(x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "equal basic arrays should match numpy");
    Ok(())
}

#[test]
fn equal_with_broadcast() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([[1.0, 2.0], [3.0, 4.0]])
y = np.array([1.0, 4.0])
result = fnp.equal(x, y)
expected = np.equal(x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "equal with broadcast should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// not_equal
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn not_equal_basic_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([1.0, 3.0, 3.0, 5.0])
result = fnp.not_equal(x, y)
expected = np.not_equal(x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "not_equal basic arrays should match numpy");
    Ok(())
}

#[test]
fn not_equal_with_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, np.nan, 3.0])
y = np.array([1.0, np.nan, 4.0])
result = fnp.not_equal(x, y)
expected = np.not_equal(x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "not_equal with nan should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// greater
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn greater_basic_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 5.0, 3.0, 4.0])
y = np.array([2.0, 3.0, 3.0, 5.0])
result = fnp.greater(x, y)
expected = np.greater(x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "greater basic arrays should match numpy");
    Ok(())
}

#[test]
fn greater_scalar_broadcast() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
result = fnp.greater(x, 3.0)
expected = np.greater(x, 3.0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "greater scalar broadcast should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// greater_equal
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn greater_equal_basic_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 5.0, 3.0, 4.0])
y = np.array([2.0, 3.0, 3.0, 5.0])
result = fnp.greater_equal(x, y)
expected = np.greater_equal(x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "greater_equal basic arrays should match numpy");
    Ok(())
}

#[test]
fn greater_equal_scalar_broadcast() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
result = fnp.greater_equal(x, 3.0)
expected = np.greater_equal(x, 3.0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "greater_equal scalar broadcast should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// less
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn less_basic_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 5.0, 3.0, 4.0])
y = np.array([2.0, 3.0, 3.0, 5.0])
result = fnp.less(x, y)
expected = np.less(x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "less basic arrays should match numpy");
    Ok(())
}

#[test]
fn less_scalar_broadcast() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
result = fnp.less(x, 3.0)
expected = np.less(x, 3.0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "less scalar broadcast should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// less_equal
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn less_equal_basic_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 5.0, 3.0, 4.0])
y = np.array([2.0, 3.0, 3.0, 5.0])
result = fnp.less_equal(x, y)
expected = np.less_equal(x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "less_equal basic arrays should match numpy");
    Ok(())
}

#[test]
fn less_equal_scalar_broadcast() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
result = fnp.less_equal(x, 3.0)
expected = np.less_equal(x, 3.0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "less_equal scalar broadcast should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// logical_and
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn logical_and_basic_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([True, True, False, False])
y = np.array([True, False, True, False])
result = fnp.logical_and(x, y)
expected = np.logical_and(x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logical_and basic should match numpy");
    Ok(())
}

#[test]
fn logical_and_numeric_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 0.0, 5.0, 0.0])
y = np.array([2.0, 3.0, 0.0, 0.0])
result = fnp.logical_and(x, y)
expected = np.logical_and(x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logical_and numeric arrays should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// logical_or
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn logical_or_basic_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([True, True, False, False])
y = np.array([True, False, True, False])
result = fnp.logical_or(x, y)
expected = np.logical_or(x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logical_or basic should match numpy");
    Ok(())
}

#[test]
fn logical_or_numeric_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 0.0, 5.0, 0.0])
y = np.array([2.0, 3.0, 0.0, 0.0])
result = fnp.logical_or(x, y)
expected = np.logical_or(x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logical_or numeric arrays should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// logical_xor
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn logical_xor_basic_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([True, True, False, False])
y = np.array([True, False, True, False])
result = fnp.logical_xor(x, y)
expected = np.logical_xor(x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logical_xor basic should match numpy");
    Ok(())
}

#[test]
fn logical_xor_numeric_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 0.0, 5.0, 0.0])
y = np.array([2.0, 3.0, 0.0, 0.0])
result = fnp.logical_xor(x, y)
expected = np.logical_xor(x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logical_xor numeric arrays should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// logical_not
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn logical_not_basic_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([True, False, True, False])
result = fnp.logical_not(x)
expected = np.logical_not(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logical_not basic should match numpy");
    Ok(())
}

#[test]
fn logical_not_numeric_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 0.0, 5.0, -3.0, 0.0])
result = fnp.logical_not(x)
expected = np.logical_not(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logical_not numeric arrays should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn comparison_with_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.inf, -np.inf, 1.0, np.inf])
y = np.array([np.inf, np.inf, np.inf, 1.0])
result_eq = fnp.equal(x, y)
result_gt = fnp.greater(x, y)
result_lt = fnp.less(x, y)
expected_eq = np.equal(x, y)
expected_gt = np.greater(x, y)
expected_lt = np.less(x, y)
print(np.array_equal(result_eq, expected_eq) and np.array_equal(result_gt, expected_gt) and np.array_equal(result_lt, expected_lt))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "comparison with inf should match numpy");
    Ok(())
}

#[test]
fn comparison_2d_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = np.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]])
result = fnp.greater_equal(x, y)
expected = np.greater_equal(x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "comparison 2d arrays should match numpy");
    Ok(())
}
