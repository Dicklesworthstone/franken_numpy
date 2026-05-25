//! Conformance tests for numpy sqrt, square, and cbrt against NumPy oracle.
//!
//! Tests sqrt, square, and cbrt functions.

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
fn sqrt_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
result = fnp.sqrt(x)
expected = np.sqrt(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sqrt basic should match numpy");
    Ok(())
}

#[test]
fn sqrt_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1.0+0j, -4.0+0j])
result = fnp.sqrt(x)
expected = np.sqrt(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sqrt complex should match numpy");
    Ok(())
}

#[test]
fn square_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-3, -2, -1, 0, 1, 2, 3])
result = fnp.square(x)
expected = np.square(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "square basic should match numpy");
    Ok(())
}

#[test]
fn cbrt_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-8.0, -1.0, 0.0, 1.0, 8.0, 27.0])
result = fnp.cbrt(x)
expected = np.cbrt(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cbrt basic should match numpy");
    Ok(())
}

#[test]
fn sqrt_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(4.0)
fnp_result = fnp.sqrt(x)
np_result = np.sqrt(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "sqrt scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn square_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(3.0)
fnp_result = fnp.square(x)
np_result = np.square(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "square scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn cbrt_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(8.0)
fnp_result = fnp.cbrt(x)
np_result = np.cbrt(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "cbrt scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn sqrt_negative_returns_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([-1.0, -4.0, -9.0])
fnp_result = fnp.sqrt(a)
np_result = np.sqrt(a)
# Both should return NaN for negative inputs
print(np.all(np.isnan(fnp_result)) and np.all(np.isnan(np_result)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sqrt of negative should return NaN");
    Ok(())
}

#[test]
fn sqrt_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0.0, np.inf, np.nan])
fnp_result = fnp.sqrt(a)
np_result = np.sqrt(a)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sqrt special values should match numpy"
    );
    Ok(())
}

#[test]
fn sqrt_negative_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0.0, -0.0])
fnp_result = fnp.sqrt(a)
np_result = np.sqrt(a)
# Check both value and sign bit
value_match = np.allclose(fnp_result, np_result)
sign_match = np.array_equal(np.signbit(fnp_result), np.signbit(np_result))
print(value_match and sign_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sqrt negative zero should match numpy"
    );
    Ok(())
}

#[test]
fn cbrt_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0])
fnp_result = fnp.cbrt(a)
np_result = np.cbrt(a)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cbrt special values should match numpy"
    );
    Ok(())
}

#[test]
fn square_overflow() -> Result<(), String> {
    let script = fnp_script(
        r#"
fmax = np.finfo(np.float64).max
a = np.array([fmax ** 0.5 * 1.1, -(fmax ** 0.5 * 1.1)])
fnp_result = fnp.square(a)
np_result = np.square(a)
# Both should overflow to inf
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "square overflow should match numpy");
    Ok(())
}

#[test]
fn square_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0])
fnp_result = fnp.square(a)
np_result = np.square(a)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "square special values should match numpy"
    );
    Ok(())
}

#[test]
fn square_signed_zero_parity() -> Result<(), String> {
    let script = fnp_script(
        r#"
# square signed-zero: (-0.0)^2 = +0.0 per IEEE 754 (negative * negative = positive)
tests = [0.0, -0.0]
all_pass = True
for x in tests:
    fnp_result = fnp.square(np.float64(x))
    np_result = np.square(np.float64(x))
    fnp_sign = np.signbit(fnp_result)
    np_sign = np.signbit(np_result)
    if fnp_sign != np_sign:
        print(f"FAIL: square({x})")
        print(f"  fnp result={fnp_result} signbit={fnp_sign}")
        print(f"  np result={np_result} signbit={np_sign}")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "square signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn cbrt_signed_zero_parity() -> Result<(), String> {
    let script = fnp_script(
        r#"
# cbrt signed-zero: cbrt(-0.0) = -0.0 (odd function preserves sign)
tests = [0.0, -0.0]
all_pass = True
for x in tests:
    fnp_result = fnp.cbrt(np.float64(x))
    np_result = np.cbrt(np.float64(x))
    fnp_sign = np.signbit(fnp_result)
    np_sign = np.signbit(np_result)
    if fnp_sign != np_sign:
        print(f"FAIL: cbrt({x})")
        print(f"  fnp result={fnp_result} signbit={fnp_sign}")
        print(f"  np result={np_result} signbit={np_sign}")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cbrt signed-zero parity should match numpy: {result}"
    );
    Ok(())
}
