//! Conformance tests for numpy.floor_divide against NumPy oracle.
//!
//! Tests floor_divide (element-wise floor division).

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
fn floor_divide_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([7, 8, 9, 10])
x2 = np.array([3, 3, 3, 3])
result = fnp.floor_divide(x1, x2)
expected = np.floor_divide(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "floor_divide basic should match numpy"
    );
    Ok(())
}

#[test]
fn floor_divide_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([7.5, 8.5, 9.5])
x2 = np.array([2.5, 2.5, 2.5])
result = fnp.floor_divide(x1, x2)
expected = np.floor_divide(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "floor_divide float should match numpy"
    );
    Ok(())
}

#[test]
fn floor_divide_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([-7, -8, 7, 8])
x2 = np.array([3, 3, -3, -3])
result = fnp.floor_divide(x1, x2)
expected = np.floor_divide(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "floor_divide negative should match numpy"
    );
    Ok(())
}

#[test]
fn floor_divide_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(7.0)
x2 = np.float64(3.0)
fnp_result = fnp.floor_divide(x1, x2)
np_result = np.floor_divide(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "floor_divide scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn floor_divide_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([np.inf, -np.inf, 1.0, np.nan])
x2 = np.array([2.0, 2.0, np.inf, 1.0])
fnp_result = fnp.floor_divide(x1, x2)
np_result = np.floor_divide(x1, x2)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "floor_divide special values should match numpy");
    Ok(())
}

#[test]
fn floor_divide_by_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
x1 = np.array([1.0, -1.0, 0.0])
x2 = np.array([0.0, 0.0, 0.0])
fnp_result = fnp.floor_divide(x1, x2)
np_result = np.floor_divide(x1, x2)
# Check inf/nan results match
print(np.allclose(fnp_result, np_result, equal_nan=True) or
      all((np.isinf(f) == np.isinf(n) and np.isnan(f) == np.isnan(n))
          for f, n in zip(fnp_result.flat, np_result.flat)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "floor_divide by zero should match numpy");
    Ok(())
}

#[test]
fn floor_divide_broadcasting() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([[10.0, 20.0], [30.0, 40.0]])
x2 = np.array([3.0, 4.0])
fnp_result = fnp.floor_divide(x1, x2)
np_result = np.floor_divide(x1, x2)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "floor_divide broadcasting should match numpy");
    Ok(())
}
