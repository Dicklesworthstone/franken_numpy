//! Conformance tests for numpy.maximum against NumPy oracle.
//!
//! Tests maximum (element-wise maximum).

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
fn maximum_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1.0, 3.0, 5.0])
x2 = np.array([2.0, 2.0, 6.0])
result = fnp.maximum(x1, x2)
expected = np.maximum(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "maximum basic should match numpy");
    Ok(())
}

#[test]
fn maximum_with_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1.0, np.nan, 3.0])
x2 = np.array([2.0, 2.0, np.nan])
result = fnp.maximum(x1, x2)
expected = np.maximum(x1, x2)
# NaN propagates
match = all((np.isnan(r) and np.isnan(e)) or r == e for r, e in zip(result.flat, expected.flat))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "maximum with NaN should propagate NaN like numpy"
    );
    Ok(())
}

#[test]
fn maximum_broadcasting() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([[1, 2], [3, 4]])
x2 = np.array([2, 3])
result = fnp.maximum(x1, x2)
expected = np.maximum(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "maximum broadcasting should match numpy"
    );
    Ok(())
}

#[test]
fn maximum_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(3.0)
x2 = np.float64(5.0)
fnp_result = fnp.maximum(x1, x2)
np_result = np.maximum(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "maximum scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn maximum_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z1 = np.array([1+1j, 5+5j, 2+2j], dtype=np.complex128)
z2 = np.array([3+3j, 2+2j, 4+4j], dtype=np.complex128)
fnp_result = fnp.maximum(z1, z2)
np_result = np.maximum(z1, z2)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "maximum complex should match numpy");
    Ok(())
}

#[test]
fn maximum_with_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1.0, np.inf, -np.inf, np.inf])
x2 = np.array([np.inf, 1.0, np.inf, -np.inf])
result = fnp.maximum(x1, x2)
expected = np.maximum(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "maximum with inf should match numpy");
    Ok(())
}

#[test]
fn maximum_negative_zero_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([0.0, -0.0, 0.0])
x2 = np.array([-0.0, 0.0, -0.0])
fnp_result = fnp.maximum(x1, x2)
np_result = np.maximum(x1, x2)
# Check value equality (0.0 == -0.0), but sign bit behavior may differ
value_match = np.array_equal(fnp_result, np_result)
print(value_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "maximum negative zero values should match numpy");
    Ok(())
}

#[test]
fn maximum_all_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([np.nan, np.nan])
x2 = np.array([np.nan, 1.0])
fnp_result = fnp.maximum(x1, x2)
np_result = np.maximum(x1, x2)
# NaN propagates, so maximum(nan, anything) = nan
print(np.array_equal(np.isnan(fnp_result), np.isnan(np_result)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "maximum all nan should match numpy");
    Ok(())
}

#[test]
fn maximum_signed_zero_tie_selection_parity() -> Result<(), String> {
    // Critical for parallel operation safety: when comparing +0.0 and -0.0,
    // the selected maximum and its sign bit must match NumPy exactly.
    // This is a stronger test than value equality (0.0 == -0.0).
    //
    // FINDING: fnp.maximum returns x1 when values are equal (sign of first arg)
    //          np.maximum returns x2 when values are equal (sign of second arg)
    //          e.g., maximum(-0.0, 0.0): fnp → -0.0, np → 0.0
    let script = fnp_script(
        r#"
# Test all combinations of signed zero comparisons
x1 = np.array([0.0, -0.0, 0.0, -0.0])
x2 = np.array([0.0, 0.0, -0.0, -0.0])
fnp_result = fnp.maximum(x1, x2)
np_result = np.maximum(x1, x2)

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
        "maximum signed-zero tie selection must match numpy sign bits exactly: {result}"
    );
    Ok(())
}

#[test]
fn maximum_nan_propagation_parity() -> Result<(), String> {
    // Critical for parallel operation safety: NaN propagation must be deterministic.
    let script = fnp_script(
        r#"
import numpy as np
# Test NaN in both positions
x1 = np.array([np.nan, 1.0, np.nan, np.inf])
x2 = np.array([1.0, np.nan, np.nan, np.nan])
fnp_result = fnp.maximum(x1, x2)
np_result = np.maximum(x1, x2)

# Both NaN positions and non-NaN values must match
nan_mask_match = np.array_equal(np.isnan(fnp_result), np.isnan(np_result))
non_nan_match = np.allclose(
    fnp_result[~np.isnan(fnp_result)],
    np_result[~np.isnan(np_result)]
) if not np.all(np.isnan(fnp_result)) else True
print(nan_mask_match and non_nan_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "maximum NaN propagation must match numpy exactly"
    );
    Ok(())
}
