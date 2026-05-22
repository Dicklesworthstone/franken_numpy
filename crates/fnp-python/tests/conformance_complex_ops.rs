//! Conformance tests for numpy complex number operations against NumPy oracle.
//!
//! Tests real, imag, conj (complex number operations).

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
fn real_complex_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+2j, 3+4j, 5+6j])
result = fnp.real(z)
expected = np.real(z)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "real complex array should match numpy"
    );
    Ok(())
}

#[test]
fn imag_complex_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+2j, 3+4j, 5+6j])
result = fnp.imag(z)
expected = np.imag(z)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "imag complex array should match numpy"
    );
    Ok(())
}

#[test]
fn conj_complex_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+2j, 3+4j, 5+6j])
result = fnp.conj(z)
expected = np.conj(z)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "conj complex array should match numpy"
    );
    Ok(())
}

#[test]
fn real_real_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 3.0])
result = fnp.real(x)
expected = np.real(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "real on real array should match numpy"
    );
    Ok(())
}

#[test]
fn real_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.complex128(1+2j)
fnp_result = fnp.real(z)
np_result = np.real(z)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "real scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn imag_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.complex128(1+2j)
fnp_result = fnp.imag(z)
np_result = np.imag(z)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "imag scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn conj_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.complex128(1+2j)
fnp_result = fnp.conj(z)
np_result = np.conj(z)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "conj scalar return type should match numpy: {result}"
    );
    Ok(())
}
