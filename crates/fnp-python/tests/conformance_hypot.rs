//! Conformance tests for numpy.hypot against NumPy oracle.
//!
//! Tests hypot (hypotenuse calculation).

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
fn hypot_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([3.0, 5.0, 8.0])
x2 = np.array([4.0, 12.0, 15.0])
result = fnp.hypot(x1, x2)
expected = np.hypot(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hypot basic should match numpy");
    Ok(())
}

#[test]
fn hypot_broadcasting() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([[1, 2], [3, 4]])
x2 = np.array([5, 6])
result = fnp.hypot(x1, x2)
expected = np.hypot(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "hypot broadcasting should match numpy"
    );
    Ok(())
}

#[test]
fn hypot_with_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([np.inf, -np.inf, 1.0])
x2 = np.array([1.0, 1.0, np.inf])
result = fnp.hypot(x1, x2)
expected = np.hypot(x1, x2)
print(np.array_equal(result, expected) or all(np.isinf(result) == np.isinf(expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "hypot with infinity should match numpy"
    );
    Ok(())
}

#[test]
fn hypot_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(3.0)
x2 = np.float64(4.0)
fnp_result = fnp.hypot(x1, x2)
np_result = np.hypot(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "hypot scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn hypot_with_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([np.nan, 1.0, np.nan])
x2 = np.array([1.0, np.nan, np.nan])
result = fnp.hypot(x1, x2)
expected = np.hypot(x1, x2)
print(np.allclose(result, expected, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hypot with nan should match numpy");
    Ok(())
}

#[test]
fn hypot_with_zeros() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([0.0, 0.0, 3.0, -0.0])
x2 = np.array([0.0, 4.0, 0.0, -0.0])
result = fnp.hypot(x1, x2)
expected = np.hypot(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hypot with zeros should match numpy");
    Ok(())
}

#[test]
fn hypot_negative_inputs() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([-3.0, -5.0, 3.0])
x2 = np.array([4.0, -12.0, -4.0])
result = fnp.hypot(x1, x2)
expected = np.hypot(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hypot negative inputs should match numpy");
    Ok(())
}

#[test]
fn hypot_large_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1e154, 1e200, 1e-154])
x2 = np.array([1e154, 1e200, 1e-154])
result = fnp.hypot(x1, x2)
expected = np.hypot(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hypot large values should match numpy");
    Ok(())
}
