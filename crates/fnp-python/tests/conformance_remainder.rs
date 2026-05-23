//! Conformance tests for numpy remainder/fmod against NumPy oracle.
//!
//! Tests remainder and fmod functions.

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
fn remainder_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([7, 8, 9, 10])
x2 = np.array([3, 3, 3, 3])
result = fnp.remainder(x1, x2)
expected = np.remainder(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "remainder basic should match numpy"
    );
    Ok(())
}

#[test]
fn remainder_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([-7, -8, 7, 8])
x2 = np.array([3, 3, -3, -3])
result = fnp.remainder(x1, x2)
expected = np.remainder(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "remainder negative should match numpy"
    );
    Ok(())
}

#[test]
fn fmod_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([7.5, 8.5, 9.5])
x2 = np.array([2.5, 2.5, 2.5])
result = fnp.fmod(x1, x2)
expected = np.fmod(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fmod basic should match numpy");
    Ok(())
}

#[test]
fn fmod_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([-7.5, 7.5, -7.5, 7.5])
x2 = np.array([2.5, -2.5, -2.5, 2.5])
result = fnp.fmod(x1, x2)
expected = np.fmod(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "fmod negative should match numpy"
    );
    Ok(())
}

#[test]
fn remainder_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(7.0)
x2 = np.float64(3.0)
fnp_result = fnp.remainder(x1, x2)
np_result = np.remainder(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "remainder scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn fmod_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(7.0)
x2 = np.float64(3.0)
fnp_result = fnp.fmod(x1, x2)
np_result = np.fmod(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "fmod scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn remainder_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([np.inf, -np.inf, np.nan, 1.0, 0.0])
x2 = np.array([2.0, 2.0, 2.0, np.nan, 2.0])
result = fnp.remainder(x1, x2)
expected = np.remainder(x1, x2)
print(np.allclose(result, expected, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "remainder special values should match numpy");
    Ok(())
}

#[test]
fn fmod_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([np.inf, -np.inf, np.nan, 1.0, 0.0])
x2 = np.array([2.0, 2.0, 2.0, np.nan, 2.0])
result = fnp.fmod(x1, x2)
expected = np.fmod(x1, x2)
print(np.allclose(result, expected, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fmod special values should match numpy");
    Ok(())
}

#[test]
fn remainder_divide_by_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    x1 = np.array([1.0, 2.0, 0.0])
    x2 = np.array([0.0, 0.0, 0.0])
    result = fnp.remainder(x1, x2)
    expected = np.remainder(x1, x2)
    # Both should produce NaN for division by zero
    print(np.allclose(result, expected, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "remainder divide by zero should match numpy");
    Ok(())
}

#[test]
fn fmod_divide_by_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    x1 = np.array([1.0, 2.0, 0.0])
    x2 = np.array([0.0, 0.0, 0.0])
    result = fnp.fmod(x1, x2)
    expected = np.fmod(x1, x2)
    print(np.allclose(result, expected, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fmod divide by zero should match numpy");
    Ok(())
}

#[test]
fn remainder_broadcasting() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([[7, 8], [9, 10]])
x2 = np.array([3, 4])
result = fnp.remainder(x1, x2)
expected = np.remainder(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "remainder broadcasting should match numpy");
    Ok(())
}
