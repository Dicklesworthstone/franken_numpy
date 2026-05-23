//! Conformance tests for numpy.fmin against NumPy oracle.
//!
//! Tests fmin (element-wise minimum, ignoring NaNs).

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
fn fmin_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1.0, 3.0, 5.0])
x2 = np.array([2.0, 2.0, 6.0])
result = fnp.fmin(x1, x2)
expected = np.fmin(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fmin basic should match numpy");
    Ok(())
}

#[test]
fn fmin_ignores_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1.0, np.nan, 3.0])
x2 = np.array([2.0, 2.0, np.nan])
result = fnp.fmin(x1, x2)
expected = np.fmin(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "fmin should ignore NaN and return non-NaN value like numpy"
    );
    Ok(())
}

#[test]
fn fmin_both_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([np.nan])
x2 = np.array([np.nan])
result = fnp.fmin(x1, x2)
expected = np.fmin(x1, x2)
print(np.isnan(result[0]) and np.isnan(expected[0]))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "fmin with both NaN should return NaN like numpy"
    );
    Ok(())
}

#[test]
fn fmin_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(3.0)
x2 = np.float64(5.0)
fnp_result = fnp.fmin(x1, x2)
np_result = np.fmin(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "fmin scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn fmin_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z1 = np.array([1+1j, 5+5j, 2+2j], dtype=np.complex128)
z2 = np.array([3+3j, 2+2j, 4+4j], dtype=np.complex128)
fnp_result = fnp.fmin(z1, z2)
np_result = np.fmin(z1, z2)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fmin complex should match numpy");
    Ok(())
}

#[test]
fn fmin_with_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1.0, np.inf, -np.inf, np.inf])
x2 = np.array([np.inf, 1.0, np.inf, -np.inf])
result = fnp.fmin(x1, x2)
expected = np.fmin(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fmin with inf should match numpy");
    Ok(())
}

#[test]
fn fmin_broadcasting() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([[1.0, 2.0], [3.0, 4.0]])
x2 = np.array([2.5, 2.5])
result = fnp.fmin(x1, x2)
expected = np.fmin(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fmin broadcasting should match numpy");
    Ok(())
}

#[test]
fn fmin_negative_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([0.0, -0.0, 0.0])
x2 = np.array([-0.0, 0.0, -0.0])
result = fnp.fmin(x1, x2)
expected = np.fmin(x1, x2)
# fmin(0, -0) and fmin(-0, 0) behavior - check they match
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fmin negative zero should match numpy");
    Ok(())
}
