//! Conformance tests for numpy.logaddexp and numpy.logaddexp2 against NumPy oracle.
//!
//! Tests logaddexp (log(exp(x1) + exp(x2))) and logaddexp2 (log2(2^x1 + 2^x2)).

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
fn logaddexp_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([0.0, 1.0, 2.0])
x2 = np.array([1.0, 2.0, 3.0])
result = fnp.logaddexp(x1, x2)
expected = np.logaddexp(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "logaddexp basic should match numpy"
    );
    Ok(())
}

#[test]
fn logaddexp_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([-1.0, -2.0, -3.0])
x2 = np.array([-0.5, -1.0, -2.0])
result = fnp.logaddexp(x1, x2)
expected = np.logaddexp(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "logaddexp with negative values should match numpy"
    );
    Ok(())
}

#[test]
fn logaddexp_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([0.0, -np.inf])
x2 = np.array([np.inf, 0.0])
result = fnp.logaddexp(x1, x2)
expected = np.logaddexp(x1, x2)
print(np.array_equal(result, expected) or all(np.isinf(result) == np.isinf(expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "logaddexp with infinity should match numpy"
    );
    Ok(())
}

#[test]
fn logaddexp2_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([0.0, 1.0, 2.0])
x2 = np.array([1.0, 2.0, 3.0])
result = fnp.logaddexp2(x1, x2)
expected = np.logaddexp2(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "logaddexp2 basic should match numpy"
    );
    Ok(())
}

#[test]
fn logaddexp_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(1.0)
x2 = np.float64(2.0)
fnp_result = fnp.logaddexp(x1, x2)
np_result = np.logaddexp(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "logaddexp scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn logaddexp2_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(1.0)
x2 = np.float64(2.0)
fnp_result = fnp.logaddexp2(x1, x2)
np_result = np.logaddexp2(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "logaddexp2 scalar return type should match numpy: {result}"
    );
    Ok(())
}
