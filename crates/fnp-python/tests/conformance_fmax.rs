//! Conformance tests for numpy.fmax against NumPy oracle.
//!
//! Tests fmax (element-wise maximum, ignoring NaNs).

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
fn fmax_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1.0, 3.0, 5.0])
x2 = np.array([2.0, 2.0, 6.0])
result = fnp.fmax(x1, x2)
expected = np.fmax(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fmax basic should match numpy");
    Ok(())
}

#[test]
fn fmax_ignores_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1.0, np.nan, 3.0])
x2 = np.array([2.0, 2.0, np.nan])
result = fnp.fmax(x1, x2)
expected = np.fmax(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "fmax should ignore NaN and return non-NaN value like numpy"
    );
    Ok(())
}

#[test]
fn fmax_both_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([np.nan])
x2 = np.array([np.nan])
result = fnp.fmax(x1, x2)
expected = np.fmax(x1, x2)
print(np.isnan(result[0]) and np.isnan(expected[0]))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "fmax with both NaN should return NaN like numpy"
    );
    Ok(())
}

#[test]
fn fmax_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(3.0)
x2 = np.float64(5.0)
fnp_result = fnp.fmax(x1, x2)
np_result = np.fmax(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "fmax scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn fmax_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z1 = np.array([1+1j, 5+5j, 2+2j], dtype=np.complex128)
z2 = np.array([3+3j, 2+2j, 4+4j], dtype=np.complex128)
fnp_result = fnp.fmax(z1, z2)
np_result = np.fmax(z1, z2)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fmax complex should match numpy");
    Ok(())
}

#[test]
fn fmax_with_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1.0, np.inf, -np.inf, np.inf])
x2 = np.array([np.inf, 1.0, np.inf, -np.inf])
result = fnp.fmax(x1, x2)
expected = np.fmax(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fmax with inf should match numpy");
    Ok(())
}

#[test]
fn fmax_broadcasting() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([[1.0, 2.0], [3.0, 4.0]])
x2 = np.array([2.5, 2.5])
result = fnp.fmax(x1, x2)
expected = np.fmax(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "fmax broadcasting should match numpy"
    );
    Ok(())
}

#[test]
fn fmax_negative_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([0.0, -0.0, 0.0])
x2 = np.array([-0.0, 0.0, -0.0])
result = fnp.fmax(x1, x2)
expected = np.fmax(x1, x2)
# fmax(0, -0) and fmax(-0, 0) behavior - check they match
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "fmax negative zero should match numpy"
    );
    Ok(())
}

#[test]
fn fmax_signed_zero_tie_selection_parity() -> Result<(), String> {
    // Critical for parallel operation safety: when comparing +0.0 and -0.0,
    // the selected fmax and its sign bit must match NumPy exactly.
    //
    // FINDING: fnp.fmax returns x1 when values are equal (sign of first arg)
    //          np.fmax returns x2 when values are equal (sign of second arg)
    let script = fnp_script(
        r#"
# Test all combinations of signed zero comparisons
x1 = np.array([0.0, -0.0, 0.0, -0.0])
x2 = np.array([0.0, 0.0, -0.0, -0.0])
fnp_result = fnp.fmax(x1, x2)
np_result = np.fmax(x1, x2)

# Check both value and sign bit match
values_match = np.array_equal(fnp_result, np_result)
signs_match = np.array_equal(np.signbit(fnp_result), np.signbit(np_result))
print(f"fnp signbit: {np.signbit(fnp_result)}")
print(f"np signbit:  {np.signbit(np_result)}")
print(f"values={values_match} signs={signs_match}")
print(values_match and signs_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().ends_with("True"),
        "fmax signed-zero tie selection must match numpy sign bits exactly: {result}"
    );
    Ok(())
}
