//! Conformance tests for numpy.logaddexp and numpy.logaddexp2 against NumPy oracle.
//!
//! Tests the native Rust implementations against NumPy.
//!
//! logaddexp(x1, x2) = log(exp(x1) + exp(x2)) - numerically stable
//! logaddexp2(x1, x2) = log2(2**x1 + 2**x2) - base-2 version

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
// logaddexp
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn logaddexp_basic_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([0.0, 1.0, 2.0, 3.0])
x2 = np.array([0.0, 1.0, 2.0, 3.0])
result = fnp.logaddexp(x1, x2)
expected = np.logaddexp(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logaddexp basic values should match numpy");
    Ok(())
}

#[test]
fn logaddexp_different_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([0.0, 1.0, 2.0, 10.0])
x2 = np.array([1.0, 2.0, 3.0, 0.0])
result = fnp.logaddexp(x1, x2)
expected = np.logaddexp(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logaddexp different values should match numpy");
    Ok(())
}

#[test]
fn logaddexp_large_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([700.0, 500.0, 1000.0])
x2 = np.array([700.0, 500.0, 999.0])
result = fnp.logaddexp(x1, x2)
expected = np.logaddexp(x1, x2)
print(np.allclose(result, expected, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logaddexp large values should match numpy");
    Ok(())
}

#[test]
fn logaddexp_negative_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([-1.0, -10.0, -100.0])
x2 = np.array([-2.0, -20.0, -99.0])
result = fnp.logaddexp(x1, x2)
expected = np.logaddexp(x1, x2)
print(np.allclose(result, expected, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logaddexp negative values should match numpy");
    Ok(())
}

#[test]
fn logaddexp_mixed_signs() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([-10.0, 10.0, -5.0, 5.0])
x2 = np.array([10.0, -10.0, 5.0, -5.0])
result = fnp.logaddexp(x1, x2)
expected = np.logaddexp(x1, x2)
print(np.allclose(result, expected, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logaddexp mixed signs should match numpy");
    Ok(())
}

#[test]
fn logaddexp_inf_handling() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
x1 = np.array([np.inf, -np.inf, 0.0, np.inf])
x2 = np.array([0.0, 0.0, np.inf, np.inf])
result = fnp.logaddexp(x1, x2)
expected = np.logaddexp(x1, x2)
def check(a, b):
    if np.isnan(a) and np.isnan(b):
        return True
    if np.isinf(a) and np.isinf(b):
        return np.sign(a) == np.sign(b)
    return np.allclose([a], [b])
print(all(check(result[i], expected[i]) for i in range(len(result))))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logaddexp inf handling should match numpy");
    Ok(())
}

#[test]
fn logaddexp_nan_handling() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
x1 = np.array([np.nan, 0.0, np.nan])
x2 = np.array([0.0, np.nan, np.nan])
result = fnp.logaddexp(x1, x2)
expected = np.logaddexp(x1, x2)
print(np.array_equal(np.isnan(result), np.isnan(expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logaddexp nan handling should match numpy");
    Ok(())
}

#[test]
fn logaddexp_broadcast() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([[1.0, 2.0], [3.0, 4.0]])
x2 = np.array([0.5, 1.5])
result = fnp.logaddexp(x1, x2)
expected = np.logaddexp(x1, x2)
print(np.allclose(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logaddexp broadcast should match numpy");
    Ok(())
}

#[test]
fn logaddexp_very_large_difference() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1000.0, -1000.0, 500.0])
x2 = np.array([0.0, 0.0, -500.0])
result = fnp.logaddexp(x1, x2)
expected = np.logaddexp(x1, x2)
print(np.allclose(result, expected, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logaddexp very large difference should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// logaddexp2
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn logaddexp2_basic_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([0.0, 1.0, 2.0, 3.0])
x2 = np.array([0.0, 1.0, 2.0, 3.0])
result = fnp.logaddexp2(x1, x2)
expected = np.logaddexp2(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logaddexp2 basic values should match numpy");
    Ok(())
}

#[test]
fn logaddexp2_different_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([0.0, 1.0, 2.0, 10.0])
x2 = np.array([1.0, 2.0, 3.0, 0.0])
result = fnp.logaddexp2(x1, x2)
expected = np.logaddexp2(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logaddexp2 different values should match numpy");
    Ok(())
}

#[test]
fn logaddexp2_large_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1000.0, 500.0, 1020.0])
x2 = np.array([1000.0, 500.0, 1019.0])
result = fnp.logaddexp2(x1, x2)
expected = np.logaddexp2(x1, x2)
print(np.allclose(result, expected, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logaddexp2 large values should match numpy");
    Ok(())
}

#[test]
fn logaddexp2_negative_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([-1.0, -10.0, -100.0])
x2 = np.array([-2.0, -20.0, -99.0])
result = fnp.logaddexp2(x1, x2)
expected = np.logaddexp2(x1, x2)
print(np.allclose(result, expected, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logaddexp2 negative values should match numpy");
    Ok(())
}

#[test]
fn logaddexp2_mixed_signs() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([-10.0, 10.0, -5.0, 5.0])
x2 = np.array([10.0, -10.0, 5.0, -5.0])
result = fnp.logaddexp2(x1, x2)
expected = np.logaddexp2(x1, x2)
print(np.allclose(result, expected, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logaddexp2 mixed signs should match numpy");
    Ok(())
}

#[test]
fn logaddexp2_inf_handling() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
x1 = np.array([np.inf, -np.inf, 0.0, np.inf])
x2 = np.array([0.0, 0.0, np.inf, np.inf])
result = fnp.logaddexp2(x1, x2)
expected = np.logaddexp2(x1, x2)
def check(a, b):
    if np.isnan(a) and np.isnan(b):
        return True
    if np.isinf(a) and np.isinf(b):
        return np.sign(a) == np.sign(b)
    return np.allclose([a], [b])
print(all(check(result[i], expected[i]) for i in range(len(result))))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logaddexp2 inf handling should match numpy");
    Ok(())
}

#[test]
fn logaddexp2_nan_handling() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
x1 = np.array([np.nan, 0.0, np.nan])
x2 = np.array([0.0, np.nan, np.nan])
result = fnp.logaddexp2(x1, x2)
expected = np.logaddexp2(x1, x2)
print(np.array_equal(np.isnan(result), np.isnan(expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logaddexp2 nan handling should match numpy");
    Ok(())
}

#[test]
fn logaddexp2_broadcast() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([[1.0, 2.0], [3.0, 4.0]])
x2 = np.array([0.5, 1.5])
result = fnp.logaddexp2(x1, x2)
expected = np.logaddexp2(x1, x2)
print(np.allclose(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logaddexp2 broadcast should match numpy");
    Ok(())
}

#[test]
fn logaddexp2_empty_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([], dtype=np.float64)
x2 = np.array([], dtype=np.float64)
result = fnp.logaddexp2(x1, x2)
expected = np.logaddexp2(x1, x2)
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logaddexp2 empty array should match numpy");
    Ok(())
}

#[test]
fn logaddexp_empty_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([], dtype=np.float64)
x2 = np.array([], dtype=np.float64)
result = fnp.logaddexp(x1, x2)
expected = np.logaddexp(x1, x2)
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logaddexp empty array should match numpy");
    Ok(())
}
