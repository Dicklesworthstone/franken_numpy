//! Conformance tests for numpy isclose against NumPy oracle.

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
// isclose basic
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn isclose_exact_equal() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 2.0, 3.0])
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose exact equal should match numpy"
    );
    Ok(())
}

#[test]
fn isclose_within_tolerance() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0 + 1e-9, 2.0 + 1e-9, 3.0 + 1e-9])
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose within tolerance should match numpy"
    );
    Ok(())
}

#[test]
fn isclose_outside_tolerance() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.1, 2.1, 3.1])
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose outside tolerance should match numpy"
    );
    Ok(())
}

#[test]
fn isclose_custom_rtol() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.01, 2.02, 3.03])
result = fnp.isclose(a, b, rtol=0.02)
expected = np.isclose(a, b, rtol=0.02)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose custom rtol should match numpy"
    );
    Ok(())
}

#[test]
fn isclose_custom_atol() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0.0, 0.0, 0.0])
b = np.array([1e-9, 1e-7, 1e-5])
result = fnp.isclose(a, b, atol=1e-6)
expected = np.isclose(a, b, atol=1e-6)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose custom atol should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// isclose NaN handling
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn isclose_nan_default() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0])
b = np.array([1.0, np.nan, 3.0])
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose nan default should match numpy"
    );
    Ok(())
}

#[test]
fn isclose_nan_equal_nan_true() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0])
b = np.array([1.0, np.nan, 3.0])
result = fnp.isclose(a, b, equal_nan=True)
expected = np.isclose(a, b, equal_nan=True)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose equal_nan=True should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// isclose infinity handling
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn isclose_inf_equal() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([np.inf, -np.inf, 1.0])
b = np.array([np.inf, -np.inf, 1.0])
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose inf equal should match numpy"
    );
    Ok(())
}

#[test]
fn isclose_inf_not_equal() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([np.inf, -np.inf])
b = np.array([-np.inf, np.inf])
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose inf not equal should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// isclose broadcasting
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn isclose_broadcast_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array(1.0)
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose broadcast scalar should match numpy"
    );
    Ok(())
}

#[test]
fn isclose_broadcast_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([1.0, 2.0])
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose broadcast 2d should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// isclose dtype handling
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn isclose_int_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype='int32')
b = np.array([1, 2, 3], dtype='int32')
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose int arrays should match numpy"
    );
    Ok(())
}

#[test]
fn isclose_mixed_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype='int32')
b = np.array([1.0, 2.0, 3.0], dtype='float64')
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose mixed dtype should match numpy"
    );
    Ok(())
}

#[test]
fn isclose_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(1.0)
y = np.float64(1.0)
fnp_result = fnp.isclose(x, y)
np_result = np.isclose(x, y)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "isclose scalar return type should match numpy: {result}"
    );
    Ok(())
}
