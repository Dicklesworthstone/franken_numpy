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

#[test]
fn fabs_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0])
fnp_result = fnp.fabs(x)
np_result = np.fabs(x)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fabs special values should match numpy");
    Ok(())
}

#[test]
fn absolute_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0])
fnp_result = fnp.absolute(x)
np_result = np.absolute(x)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "absolute special values should match numpy");
    Ok(())
}

#[test]
fn absolute_integer_dtypes() -> Result<(), String> {
    let script = fnp_script(
        r#"
tests_pass = True
for dtype in [np.int8, np.int16, np.int32, np.int64]:
    x = np.array([-128, -1, 0, 1, 127], dtype=dtype)
    fnp_result = fnp.absolute(x)
    np_result = np.absolute(x)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "absolute integer dtypes should match numpy");
    Ok(())
}

#[test]
fn abs_alias_matches_absolute() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-5, -1, 0, 1, 5])
fnp_abs = fnp.abs(x)
fnp_absolute = fnp.absolute(x)
print(np.array_equal(fnp_abs, fnp_absolute))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "abs should be alias for absolute");
    Ok(())
}

#[test]
fn abs_signed_zero_parity() -> Result<(), String> {
    let script = fnp_script(
        r#"
# abs signed-zero: abs(-0.0) = 0.0 (positive, not negative zero)
tests = [0.0, -0.0]
all_pass = True
for x in tests:
    fnp_result = fnp.abs(np.float64(x))
    np_result = np.abs(np.float64(x))
    fnp_sign = np.signbit(fnp_result)
    np_sign = np.signbit(np_result)
    if fnp_sign != np_sign:
        print(f"FAIL: abs({x}) signbit fnp={fnp_sign} np={np_sign}")
        all_pass = False
    if fnp_result != np_result:
        print(f"FAIL: abs({x}) value mismatch")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "abs signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Complex number edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn absolute_complex_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([np.inf + 1j, 1 + np.inf*1j, np.inf + np.inf*1j])
fnp_result = fnp.absolute(z)
np_result = np.absolute(z)
print(np.all(np.isinf(fnp_result) == np.isinf(np_result)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "absolute of complex inf should match numpy");
    Ok(())
}

#[test]
fn absolute_complex_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([np.nan + 1j, 1 + np.nan*1j, np.nan + np.nan*1j])
fnp_result = fnp.absolute(z)
np_result = np.absolute(z)
print(np.all(np.isnan(fnp_result) == np.isnan(np_result)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "absolute of complex nan should match numpy");
    Ok(())
}

#[test]
fn real_imag_conj_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+2j, 3+4j, -1-2j])
tests = []
tests.append(np.array_equal(fnp.real(a), np.real(a)))
tests.append(np.array_equal(fnp.imag(a), np.imag(a)))
tests.append(np.array_equal(fnp.conj(a), np.conj(a)))
print(all(tests))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "real/imag/conj should match numpy");
    Ok(())
}

#[test]
fn real_imag_on_real_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
tests = []
tests.append(np.array_equal(fnp.real(a), np.real(a)))
tests.append(np.array_equal(fnp.imag(a), np.imag(a)))
print(all(tests))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "real/imag on real array should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// More edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fabs_empty_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([], dtype=np.float64)
fnp_result = fnp.fabs(x)
np_result = np.fabs(x)
print(np.array_equal(fnp_result, np_result) and fnp_result.shape == np_result.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fabs empty array should match numpy");
    Ok(())
}

#[test]
fn absolute_empty_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([], dtype=np.float64)
fnp_result = fnp.absolute(x)
np_result = np.absolute(x)
print(np.array_equal(fnp_result, np_result) and fnp_result.shape == np_result.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "absolute empty array should match numpy");
    Ok(())
}

#[test]
fn fabs_subnormal_numbers() -> Result<(), String> {
    let script = fnp_script(
        r#"
import sys
tiny = sys.float_info.min
subnormal = tiny / 2.0
x = np.array([subnormal, -subnormal, tiny, -tiny])
fnp_result = fnp.fabs(x)
np_result = np.fabs(x)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fabs subnormal numbers should match numpy");
    Ok(())
}

#[test]
fn absolute_large_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
import sys
x = np.array([sys.float_info.max, -sys.float_info.max, sys.float_info.max / 2, -sys.float_info.max / 2])
fnp_result = fnp.absolute(x)
np_result = np.absolute(x)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "absolute large values should match numpy");
    Ok(())
}

#[test]
fn absolute_complex_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([0+0j, 0-0j, -0+0j, -0-0j], dtype=np.complex128)
fnp_result = fnp.absolute(z)
np_result = np.absolute(z)
print(np.array_equal(fnp_result, np_result) and np.all(fnp_result == 0.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "absolute of complex zero should match numpy");
    Ok(())
}
