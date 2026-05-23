//! Conformance tests for numpy.ldexp against NumPy oracle.
//!
//! Tests ldexp (load exponent).

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
fn ldexp_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 3.0])
exp = np.array([2, 3, 4])
result = fnp.ldexp(x, exp)
expected = np.ldexp(x, exp)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ldexp basic should match numpy");
    Ok(())
}

#[test]
fn ldexp_negative_exp() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([8.0, 16.0, 32.0])
exp = np.array([-1, -2, -3])
result = fnp.ldexp(x, exp)
expected = np.ldexp(x, exp)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ldexp with negative exponents should match numpy"
    );
    Ok(())
}

#[test]
fn ldexp_broadcasting() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([[1.0, 2.0], [3.0, 4.0]])
exp = np.array([1, 2])
result = fnp.ldexp(x, exp)
expected = np.ldexp(x, exp)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ldexp broadcasting should match numpy"
    );
    Ok(())
}

#[test]
fn ldexp_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(2.0)
exp = np.int32(3)
fnp_result = fnp.ldexp(x, exp)
np_result = np.ldexp(x, exp)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "ldexp scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn ldexp_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0])
exp = np.array([1, 1, 1, 1, 1])
result = fnp.ldexp(x, exp)
expected = np.ldexp(x, exp)
print(np.allclose(result, expected, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ldexp special values should match numpy");
    Ok(())
}

#[test]
fn ldexp_overflow() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 1.0])
exp = np.array([1024, 2000])
result = fnp.ldexp(x, exp)
expected = np.ldexp(x, exp)
print(np.allclose(result, expected) or np.all(np.isinf(result) == np.isinf(expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ldexp overflow should match numpy");
    Ok(())
}

#[test]
fn ldexp_underflow() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 1.0])
exp = np.array([-1024, -2000])
result = fnp.ldexp(x, exp)
expected = np.ldexp(x, exp)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ldexp underflow should match numpy");
    Ok(())
}

#[test]
fn ldexp_zero_exponent() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 2.0, -3.0, 0.5])
exp = np.array([0, 0, 0, 0])
result = fnp.ldexp(x, exp)
expected = np.ldexp(x, exp)
print(np.allclose(result, expected) and np.allclose(result, x))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ldexp zero exponent should match numpy");
    Ok(())
}
