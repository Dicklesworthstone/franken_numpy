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

#[test]
fn sign_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+1j, -2+2j, 0+0j, 3-4j], dtype=np.complex128)
fnp_result = fnp.sign(z)
np_result = np.sign(z)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sign complex should match numpy");
    Ok(())
}

#[test]
fn sign_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0])
fnp_result = fnp.sign(x)
np_result = np.sign(x)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sign special values should match numpy");
    Ok(())
}

#[test]
fn signbit_negative_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.0, -0.0])
fnp_result = fnp.signbit(x)
np_result = np.signbit(x)
# signbit(-0.0) should be True, signbit(0.0) should be False
print(np.array_equal(fnp_result, np_result) and fnp_result[0] == False and fnp_result[1] == True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "signbit negative zero should match numpy");
    Ok(())
}

#[test]
fn sign_integer_dtypes() -> Result<(), String> {
    let script = fnp_script(
        r#"
tests_pass = True
for dtype in [np.int8, np.int16, np.int32, np.int64]:
    x = np.array([-128, -1, 0, 1, 127], dtype=dtype)
    fnp_result = fnp.sign(x)
    np_result = np.sign(x)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sign integer dtypes should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge case tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sign_empty_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([], dtype=np.float64)
fnp_result = fnp.sign(x)
np_result = np.sign(x)
print(np.array_equal(fnp_result, np_result) and fnp_result.shape == np_result.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sign empty array should match numpy");
    Ok(())
}

#[test]
fn signbit_empty_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([], dtype=np.float64)
fnp_result = fnp.signbit(x)
np_result = np.signbit(x)
print(np.array_equal(fnp_result, np_result) and fnp_result.shape == np_result.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "signbit empty array should match numpy");
    Ok(())
}

#[test]
fn sign_single_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
tests_pass = True
for val in [-5.0, -0.0, 0.0, 5.0]:
    x = np.array([val])
    fnp_result = fnp.sign(x)
    np_result = np.sign(x)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sign single element should match numpy");
    Ok(())
}

#[test]
fn sign_unsigned_integers() -> Result<(), String> {
    let script = fnp_script(
        r#"
tests_pass = True
for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
    x = np.array([0, 1, 127, 255], dtype=dtype)
    fnp_result = fnp.sign(x)
    np_result = np.sign(x)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sign unsigned integers should match numpy");
    Ok(())
}

#[test]
fn sign_subnormal_numbers() -> Result<(), String> {
    let script = fnp_script(
        r#"
import sys
tiny = sys.float_info.min
subnormal = tiny / 2.0
x = np.array([subnormal, -subnormal, tiny, -tiny, 0.0])
fnp_result = fnp.sign(x)
np_result = np.sign(x)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sign subnormal numbers should match numpy");
    Ok(())
}

#[test]
fn signbit_subnormal_numbers() -> Result<(), String> {
    let script = fnp_script(
        r#"
import sys
tiny = sys.float_info.min
subnormal = tiny / 2.0
x = np.array([subnormal, -subnormal, tiny, -tiny])
fnp_result = fnp.signbit(x)
np_result = np.signbit(x)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "signbit subnormal numbers should match numpy");
    Ok(())
}
