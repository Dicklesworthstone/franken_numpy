//! Conformance tests for numpy.angle against NumPy oracle.
//!
//! Tests angle (return the angle of a complex number).

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
fn angle_complex_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+1j, 1-1j, -1+1j, -1-1j])
result = fnp.angle(z)
expected = np.angle(z)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "angle complex array should match numpy"
    );
    Ok(())
}

#[test]
fn angle_deg() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+1j, 1-1j])
result = fnp.angle(z, deg=True)
expected = np.angle(z, deg=True)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "angle with deg=True should match numpy"
    );
    Ok(())
}

#[test]
fn angle_real_input() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1.0, -1.0, 0.0])
result = fnp.angle(z)
expected = np.angle(z)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "angle with real input should match numpy"
    );
    Ok(())
}

#[test]
fn angle_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.complex128(1+1j)
fnp_result = fnp.angle(z)
np_result = np.angle(z)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "angle scalar return type should match numpy: {result}"
    );
    Ok(())
}
