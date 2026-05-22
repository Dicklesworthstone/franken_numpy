//! Conformance tests for numpy.sign and numpy.signbit against NumPy oracle.
//!
//! Tests sign and signbit functions.

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
fn sign_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-5, -1, 0, 1, 5])
result = fnp.sign(x)
expected = np.sign(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sign basic should match numpy");
    Ok(())
}

#[test]
fn sign_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
result = fnp.sign(x)
expected = np.sign(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sign float should match numpy");
    Ok(())
}

#[test]
fn signbit_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1.0, 0.0, 1.0, -0.0])
result = fnp.signbit(x)
expected = np.signbit(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "signbit basic should match numpy"
    );
    Ok(())
}

#[test]
fn signbit_with_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.nan, -np.nan, np.inf, -np.inf])
result = fnp.signbit(x)
expected = np.signbit(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "signbit with nan/inf should match numpy"
    );
    Ok(())
}

#[test]
fn sign_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(-5.0)
fnp_result = fnp.sign(x)
np_result = np.sign(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "sign scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn signbit_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(-5.0)
fnp_result = fnp.signbit(x)
np_result = np.signbit(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "signbit scalar return type should match numpy: {result}"
    );
    Ok(())
}
