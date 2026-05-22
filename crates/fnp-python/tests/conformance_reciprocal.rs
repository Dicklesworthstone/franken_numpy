//! Conformance tests for numpy.reciprocal against NumPy oracle.
//!
//! Tests the native Rust reciprocal implementation against NumPy.

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

// ─────────────────────────────────────────────────────────────────────────────
// reciprocal
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn reciprocal_float64_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 4.0, 5.0, 10.0])
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
        "reciprocal float64 basic should match numpy"
    );
    Ok(())
}

#[test]
fn reciprocal_float32_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 4.0, 5.0, 10.0], dtype=np.float32)
result = fnp.reciprocal(x)
expected = np.reciprocal(x)
print(np.allclose(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "reciprocal float32 should match numpy dtype and values"
    );
    Ok(())
}

#[test]
fn reciprocal_int64_integer_division() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 3, 4, 5], dtype=np.int64)
result = fnp.reciprocal(x)
expected = np.reciprocal(x)
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "reciprocal int64 should use integer division"
    );
    Ok(())
}

#[test]
fn reciprocal_negative_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1.0, -2.0, -4.0, -5.0])
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
        "reciprocal negative values should match numpy"
    );
    Ok(())
}

#[test]
fn reciprocal_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
x = np.array([0.0, np.inf, -np.inf, np.nan])
result = fnp.reciprocal(x)
expected = np.reciprocal(x)
# Check inf/nan handling
match = (np.isinf(result[0]) and np.isinf(expected[0]) and
         result[1] == expected[1] and
         result[2] == expected[2] and
         np.isnan(result[3]) and np.isnan(expected[3]))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "reciprocal special values should match numpy"
    );
    Ok(())
}

#[test]
fn reciprocal_2d_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([[1.0, 2.0], [4.0, 8.0]])
result = fnp.reciprocal(x)
expected = np.reciprocal(x)
print(np.allclose(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "reciprocal 2d array should match numpy"
    );
    Ok(())
}

#[test]
fn reciprocal_empty_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([], dtype=np.float64)
result = fnp.reciprocal(x)
expected = np.reciprocal(x)
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "reciprocal empty array should match numpy"
    );
    Ok(())
}

#[test]
fn reciprocal_large_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1e100, 1e-100, -1e100, -1e-100])
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
        "reciprocal large values should match numpy"
    );
    Ok(())
}

#[test]
fn reciprocal_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(2.0)
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
