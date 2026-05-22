//! Conformance tests for numpy logical operations against NumPy oracle.
//!
//! Tests logical_and, logical_or, logical_not, logical_xor.

use std::io::Write;
use std::process::{Command, Stdio};

fn numpy_oracle(script: &str) -> Result<String, String> {
    let mut child = Command::new("python3")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("python3 should be available: {error}\nScript: {script}"))?;

    child
        .stdin
        .as_mut()
        .ok_or_else(|| format!("python3 stdin pipe should be available\nScript: {script}"))?
        .write_all(script.as_bytes())
        .map_err(|error| {
            format!("failed to write Python oracle script: {error}\nScript: {script}")
        })?;

    let output = child
        .wait_with_output()
        .map_err(|error| format!("failed to wait for Python oracle: {error}\nScript: {script}"))?;
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
fn logical_and_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([True, True, False, False])
x2 = np.array([True, False, True, False])
result = fnp.logical_and(x1, x2)
expected = np.logical_and(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "logical_and basic should match numpy"
    );
    Ok(())
}

#[test]
fn logical_or_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([True, True, False, False])
x2 = np.array([True, False, True, False])
result = fnp.logical_or(x1, x2)
expected = np.logical_or(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logical_or basic should match numpy");
    Ok(())
}

#[test]
fn logical_not_basic() -> Result<(), String> {
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
    assert_eq!(
        result.trim(),
        "True",
        "logical_not basic should match numpy"
    );
    Ok(())
}

#[test]
fn logical_xor_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([True, True, False, False])
x2 = np.array([True, False, True, False])
result = fnp.logical_xor(x1, x2)
expected = np.logical_xor(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "logical_xor basic should match numpy"
    );
    Ok(())
}

#[test]
fn logical_and_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.bool_(True)
x2 = np.bool_(False)
fnp_result = fnp.logical_and(x1, x2)
np_result = np.logical_and(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "logical_and scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn logical_not_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.bool_(True)
fnp_result = fnp.logical_not(x)
np_result = np.logical_not(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "logical_not scalar return type should match numpy: {result}"
    );
    Ok(())
}
