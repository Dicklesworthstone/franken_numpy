//! Conformance tests for numpy absolute value functions against NumPy oracle.
//!
//! Tests fabs and absolute functions.

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
fn fabs_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
result = fnp.fabs(x)
expected = np.fabs(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fabs basic should match numpy");
    Ok(())
}

#[test]
fn fabs_signed_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-0.0, 0.0])
result = fnp.fabs(x)
expected = np.fabs(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "fabs signed zero should match numpy"
    );
    Ok(())
}

#[test]
fn absolute_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-5, -1, 0, 1, 5])
result = fnp.absolute(x)
expected = np.absolute(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "absolute basic should match numpy"
    );
    Ok(())
}

#[test]
fn absolute_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([3+4j, -3+4j, 3-4j])
result = fnp.absolute(x)
expected = np.absolute(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "absolute complex should match numpy"
    );
    Ok(())
}

#[test]
fn fabs_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(-5.0)
fnp_result = fnp.fabs(x)
np_result = np.fabs(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "fabs scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn absolute_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.int64(-5)
fnp_result = fnp.absolute(x)
np_result = np.absolute(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "absolute scalar return type should match numpy: {result}"
    );
    Ok(())
}
