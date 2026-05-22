//! Conformance tests for numpy around/round against NumPy oracle.

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
// around basic
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn around_default_decimals() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.4, 1.5, 1.6, 2.5, -1.5])
result = fnp.around(a)
expected = np.around(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "around default should match numpy");
    Ok(())
}

#[test]
fn around_positive_decimals() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.234, 2.567, 3.891])
result = fnp.around(a, decimals=2)
expected = np.around(a, decimals=2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "around decimals=2 should match numpy"
    );
    Ok(())
}

#[test]
fn around_negative_decimals() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1234.5, 2567.5, 3891.5])
result = fnp.around(a, decimals=-2)
expected = np.around(a, decimals=-2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "around decimals=-2 should match numpy"
    );
    Ok(())
}

#[test]
fn around_integer_input() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4], dtype='int32')
result = fnp.around(a, decimals=2)
expected = np.around(a, decimals=2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "around integer input should match numpy"
    );
    Ok(())
}

#[test]
fn around_2d_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1.234, 2.567], [3.891, 4.123]])
result = fnp.around(a, decimals=1)
expected = np.around(a, decimals=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "around 2d should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// round (alias for around)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn round_matches_around() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.234, 2.567, 3.891])
around_result = fnp.around(a, decimals=2)
round_result = fnp.round(a, decimals=2)
print(np.array_equal(around_result, round_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "round should match around");
    Ok(())
}

#[test]
fn round_default() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.4, 1.5, 2.5, 3.5])
result = fnp.round(a)
expected = np.round(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "round default should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn around_nan_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([np.nan, np.inf, -np.inf, 1.5])
result = fnp.around(a)
expected = np.around(a)
print(np.allclose(result, expected, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "around nan/inf should match numpy");
    Ok(())
}

#[test]
fn around_preserves_dtype_float32() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.234, 2.567], dtype='float32')
result = fnp.around(a, decimals=1)
expected = np.around(a, decimals=1)
print(result.dtype == expected.dtype and np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "around should preserve float32 dtype"
    );
    Ok(())
}

#[test]
fn around_scalar_input() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.around(np.array(1.567), decimals=2)
expected = np.around(1.567, decimals=2)
print(np.isclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "around scalar should match numpy");
    Ok(())
}

#[test]
fn around_bankers_rounding() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Test banker's rounding (round half to even)
a = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
result = fnp.around(a)
expected = np.around(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "around should use banker's rounding");
    Ok(())
}

#[test]
fn around_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(5.5)
fnp_result = fnp.around(x)
np_result = np.around(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "around scalar return type should match numpy: {result}"
    );
    Ok(())
}
