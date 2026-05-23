//! Conformance tests for numpy.modf against NumPy oracle.
//!
//! Tests modf (return fractional and integral parts of an array).

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
fn modf_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.5, 2.7, -3.3, 4.0])
fnp_frac, fnp_int = fnp.modf(x)
np_frac, np_int = np.modf(x)
print(np.allclose(fnp_frac, np_frac) and np.array_equal(fnp_int, np_int))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "modf basic should match numpy");
    Ok(())
}

#[test]
fn modf_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1.5, -2.7, -3.9])
fnp_frac, fnp_int = fnp.modf(x)
np_frac, np_int = np.modf(x)
print(np.allclose(fnp_frac, np_frac) and np.array_equal(fnp_int, np_int))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "modf with negative values should match numpy"
    );
    Ok(())
}

#[test]
fn modf_integers() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 3.0, -4.0])
fnp_frac, fnp_int = fnp.modf(x)
np_frac, np_int = np.modf(x)
print(np.allclose(fnp_frac, np_frac) and np.array_equal(fnp_int, np_int))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "modf with integers should match numpy"
    );
    Ok(())
}

#[test]
fn modf_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(3.5)
fnp_frac, fnp_int = fnp.modf(x)
np_frac, np_int = np.modf(x)
frac_type_match = type(fnp_frac).__name__ == type(np_frac).__name__
int_type_match = type(fnp_int).__name__ == type(np_int).__name__
print(frac_type_match and int_type_match, fnp_frac, fnp_int, np_frac, np_int)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "modf scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn modf_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0])
fnp_frac, fnp_int = fnp.modf(x)
np_frac, np_int = np.modf(x)
frac_match = np.allclose(fnp_frac, np_frac, equal_nan=True)
int_match = np.allclose(fnp_int, np_int, equal_nan=True)
print(frac_match and int_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "modf special values should match numpy");
    Ok(())
}

#[test]
fn modf_signed_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.0, -0.0])
fnp_frac, fnp_int = fnp.modf(x)
np_frac, np_int = np.modf(x)
# Check both values and sign bits
value_match = np.allclose(fnp_frac, np_frac) and np.allclose(fnp_int, np_int)
sign_match = np.array_equal(np.signbit(fnp_frac), np.signbit(np_frac)) and np.array_equal(np.signbit(fnp_int), np.signbit(np_int))
print(value_match and sign_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "modf signed zero should match numpy");
    Ok(())
}

#[test]
fn modf_small_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.1, 0.01, 0.001, -0.1, -0.01])
fnp_frac, fnp_int = fnp.modf(x)
np_frac, np_int = np.modf(x)
print(np.allclose(fnp_frac, np_frac) and np.allclose(fnp_int, np_int))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "modf small values should match numpy");
    Ok(())
}

#[test]
fn modf_large_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1e15, 1e16, -1e15, 1e15 + 0.5])
fnp_frac, fnp_int = fnp.modf(x)
np_frac, np_int = np.modf(x)
print(np.allclose(fnp_frac, np_frac) and np.allclose(fnp_int, np_int))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "modf large values should match numpy");
    Ok(())
}
