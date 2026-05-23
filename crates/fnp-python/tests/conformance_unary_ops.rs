//! Conformance tests for numpy unary operations against NumPy oracle.
//!
//! Tests positive, negative, reciprocal functions.

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
fn positive_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-3, -2, -1, 0, 1, 2, 3])
result = fnp.positive(x)
expected = np.positive(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "positive basic should match numpy"
    );
    Ok(())
}

#[test]
fn negative_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-3, -2, -1, 0, 1, 2, 3])
result = fnp.negative(x)
expected = np.negative(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "negative basic should match numpy"
    );
    Ok(())
}

#[test]
fn reciprocal_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 4.0, 5.0])
result = fnp.reciprocal(x)
expected = np.reciprocal(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "reciprocal basic should match numpy"
    );
    Ok(())
}

#[test]
fn negative_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(5.0)
fnp_result = fnp.negative(x)
np_result = np.negative(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "negative scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn reciprocal_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(5.0)
fnp_result = fnp.reciprocal(x)
np_result = np.reciprocal(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "reciprocal scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn negative_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+2j, -3+4j, 5-6j], dtype=np.complex128)
fnp_result = fnp.negative(z)
np_result = np.negative(z)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "negative complex should match numpy");
    Ok(())
}

#[test]
fn positive_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+2j, -3+4j, 5-6j], dtype=np.complex128)
fnp_result = fnp.positive(z)
np_result = np.positive(z)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "positive complex should match numpy");
    Ok(())
}

#[test]
fn reciprocal_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+1j, 2-1j, 3+0j], dtype=np.complex128)
fnp_result = fnp.reciprocal(z)
np_result = np.reciprocal(z)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "reciprocal complex should match numpy");
    Ok(())
}

#[test]
fn positive_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0])
fnp_result = fnp.positive(x)
np_result = np.positive(x)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "positive special values should match numpy");
    Ok(())
}

#[test]
fn negative_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0])
fnp_result = fnp.negative(x)
np_result = np.negative(x)
# Check values and sign bits for zeros
value_match = np.allclose(fnp_result, np_result, equal_nan=True)
sign_match = np.array_equal(np.signbit(fnp_result), np.signbit(np_result))
print(value_match and sign_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "negative special values should match numpy");
    Ok(())
}

#[test]
fn negative_zero_sign() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.0, -0.0])
fnp_result = fnp.negative(x)
np_result = np.negative(x)
# negative(0.0) should be -0.0, negative(-0.0) should be 0.0
sign_match = np.array_equal(np.signbit(fnp_result), np.signbit(np_result))
print(sign_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "negative zero sign should match numpy");
    Ok(())
}

#[test]
fn positive_preserves_sign() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.0, -0.0, 1.0, -1.0])
fnp_result = fnp.positive(x)
np_result = np.positive(x)
# positive should preserve sign
sign_match = np.array_equal(np.signbit(fnp_result), np.signbit(np_result))
value_match = np.array_equal(fnp_result, np_result)
print(sign_match and value_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "positive should preserve sign");
    Ok(())
}
