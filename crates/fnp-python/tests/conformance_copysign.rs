//! Conformance tests for numpy.copysign against NumPy oracle.
//!
//! Tests copysign (copy sign of a number).

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
fn copysign_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1.0, -1.0, 2.0, -2.0])
x2 = np.array([1.0, 1.0, -1.0, -1.0])
result = fnp.copysign(x1, x2)
expected = np.copysign(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "copysign basic should match numpy");
    Ok(())
}

#[test]
fn copysign_signed_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1.0, -1.0])
x2 = np.array([-0.0, 0.0])
result = fnp.copysign(x1, x2)
expected = np.copysign(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "copysign with signed zero should match numpy"
    );
    Ok(())
}

#[test]
fn copysign_broadcasting() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([[1.0, 2.0], [3.0, 4.0]])
x2 = np.array([-1.0, 1.0])
result = fnp.copysign(x1, x2)
expected = np.copysign(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "copysign broadcasting should match numpy"
    );
    Ok(())
}

#[test]
fn copysign_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(-3.0)
x2 = np.float64(1.0)
fnp_result = fnp.copysign(x1, x2)
np_result = np.copysign(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "copysign scalar return type should match numpy: {result}"
    );
    Ok(())
}
