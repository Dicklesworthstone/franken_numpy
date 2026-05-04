//! Conformance tests for numpy allclose against NumPy oracle.

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
// allclose basic
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn allclose_exact_equal() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 2.0, 3.0])
result = fnp.allclose(a, b)
expected = np.allclose(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose exact equal should match numpy");
    Ok(())
}

#[test]
fn allclose_within_tolerance() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0 + 1e-9, 2.0 + 1e-9, 3.0 + 1e-9])
result = fnp.allclose(a, b)
expected = np.allclose(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose within tolerance should match numpy");
    Ok(())
}

#[test]
fn allclose_outside_tolerance() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.1, 2.0, 3.0])
result = fnp.allclose(a, b)
expected = np.allclose(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose outside tolerance should match numpy");
    Ok(())
}

#[test]
fn allclose_custom_rtol() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.01, 2.02, 3.03])
result = fnp.allclose(a, b, rtol=0.02)
expected = np.allclose(a, b, rtol=0.02)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose custom rtol should match numpy");
    Ok(())
}

#[test]
fn allclose_custom_atol() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0.0, 0.0, 0.0])
b = np.array([1e-9, 1e-9, 1e-9])
result = fnp.allclose(a, b, atol=1e-6)
expected = np.allclose(a, b, atol=1e-6)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose custom atol should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// allclose NaN handling
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn allclose_nan_default() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0])
b = np.array([1.0, np.nan, 3.0])
result = fnp.allclose(a, b)
expected = np.allclose(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose nan default should match numpy");
    Ok(())
}

#[test]
fn allclose_nan_equal_nan_true() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0])
b = np.array([1.0, np.nan, 3.0])
result = fnp.allclose(a, b, equal_nan=True)
expected = np.allclose(a, b, equal_nan=True)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose equal_nan=True should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// allclose infinity handling
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn allclose_inf_equal() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([np.inf, -np.inf, 1.0])
b = np.array([np.inf, -np.inf, 1.0])
result = fnp.allclose(a, b)
expected = np.allclose(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose inf equal should match numpy");
    Ok(())
}

#[test]
fn allclose_inf_not_equal() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([np.inf, 1.0])
b = np.array([-np.inf, 1.0])
result = fnp.allclose(a, b)
expected = np.allclose(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose inf not equal should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// allclose broadcasting
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn allclose_broadcast_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 1.0, 1.0])
b = np.array(1.0)
result = fnp.allclose(a, b)
expected = np.allclose(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose broadcast scalar should match numpy");
    Ok(())
}

#[test]
fn allclose_broadcast_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1.0, 2.0], [1.0, 2.0]])
b = np.array([1.0, 2.0])
result = fnp.allclose(a, b)
expected = np.allclose(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose broadcast 2d should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// allclose dtype handling
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn allclose_int_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype='int32')
b = np.array([1, 2, 3], dtype='int32')
result = fnp.allclose(a, b)
expected = np.allclose(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose int arrays should match numpy");
    Ok(())
}

#[test]
fn allclose_mixed_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype='int32')
b = np.array([1.0, 2.0, 3.0], dtype='float64')
result = fnp.allclose(a, b)
expected = np.allclose(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose mixed dtype should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// allclose returns scalar boolean
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn allclose_returns_python_bool() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 2.0, 3.0])
result = fnp.allclose(a, b)
# Should be a Python bool, not numpy.bool_
print(type(result).__name__ in ('bool', 'numpy.bool_', 'bool_'))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose should return bool type");
    Ok(())
}
