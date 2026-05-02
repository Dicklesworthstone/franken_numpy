//! Conformance tests for numpy.isposinf and numpy.isneginf against NumPy oracle.
//!
//! Tests the native Rust implementations against NumPy.

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
// isposinf
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn isposinf_basic_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.inf, -np.inf, 0.0, 1.0, -1.0, np.nan])
result = fnp.isposinf(x)
expected = np.isposinf(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isposinf basic values should match numpy");
    Ok(())
}

#[test]
fn isposinf_all_positive_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.inf, np.inf, np.inf])
result = fnp.isposinf(x)
expected = np.isposinf(x)
print(np.array_equal(result, expected) and np.all(result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isposinf all positive inf should be True");
    Ok(())
}

#[test]
fn isposinf_negative_inf_is_false() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-np.inf, -np.inf, -np.inf])
result = fnp.isposinf(x)
expected = np.isposinf(x)
print(np.array_equal(result, expected) and not np.any(result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isposinf for negative inf should be False");
    Ok(())
}

#[test]
fn isposinf_finite_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, -1.0, 0.0, 1e308, -1e308, 1e-308])
result = fnp.isposinf(x)
expected = np.isposinf(x)
print(np.array_equal(result, expected) and not np.any(result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isposinf for finite values should be False");
    Ok(())
}

#[test]
fn isposinf_nan_is_false() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.nan, np.nan, np.nan])
result = fnp.isposinf(x)
expected = np.isposinf(x)
print(np.array_equal(result, expected) and not np.any(result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isposinf for NaN should be False");
    Ok(())
}

#[test]
fn isposinf_2d_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([[np.inf, -np.inf], [1.0, np.nan]])
result = fnp.isposinf(x)
expected = np.isposinf(x)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isposinf 2d array should match numpy");
    Ok(())
}

#[test]
fn isposinf_empty_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([], dtype=np.float64)
result = fnp.isposinf(x)
expected = np.isposinf(x)
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isposinf empty array should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// isneginf
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn isneginf_basic_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.inf, -np.inf, 0.0, 1.0, -1.0, np.nan])
result = fnp.isneginf(x)
expected = np.isneginf(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isneginf basic values should match numpy");
    Ok(())
}

#[test]
fn isneginf_all_negative_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-np.inf, -np.inf, -np.inf])
result = fnp.isneginf(x)
expected = np.isneginf(x)
print(np.array_equal(result, expected) and np.all(result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isneginf all negative inf should be True");
    Ok(())
}

#[test]
fn isneginf_positive_inf_is_false() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.inf, np.inf, np.inf])
result = fnp.isneginf(x)
expected = np.isneginf(x)
print(np.array_equal(result, expected) and not np.any(result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isneginf for positive inf should be False");
    Ok(())
}

#[test]
fn isneginf_finite_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, -1.0, 0.0, 1e308, -1e308, 1e-308])
result = fnp.isneginf(x)
expected = np.isneginf(x)
print(np.array_equal(result, expected) and not np.any(result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isneginf for finite values should be False");
    Ok(())
}

#[test]
fn isneginf_nan_is_false() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.nan, np.nan, np.nan])
result = fnp.isneginf(x)
expected = np.isneginf(x)
print(np.array_equal(result, expected) and not np.any(result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isneginf for NaN should be False");
    Ok(())
}

#[test]
fn isneginf_2d_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([[np.inf, -np.inf], [1.0, np.nan]])
result = fnp.isneginf(x)
expected = np.isneginf(x)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isneginf 2d array should match numpy");
    Ok(())
}

#[test]
fn isneginf_empty_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([], dtype=np.float64)
result = fnp.isneginf(x)
expected = np.isneginf(x)
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isneginf empty array should match numpy");
    Ok(())
}

#[test]
fn isposinf_isneginf_complement() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.inf, -np.inf, 0.0, 1.0, np.nan])
posinf = fnp.isposinf(x)
neginf = fnp.isneginf(x)
isinf = np.isinf(x)
# For any x, isinf(x) == isposinf(x) | isneginf(x)
complement = np.logical_or(posinf, neginf)
print(np.array_equal(complement, isinf))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isposinf | isneginf should equal isinf");
    Ok(())
}

#[test]
fn isposinf_isneginf_mutually_exclusive() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.inf, -np.inf, 0.0, 1.0, np.nan, 1e308, -1e308])
posinf = fnp.isposinf(x)
neginf = fnp.isneginf(x)
# isposinf and isneginf should never both be True for the same element
overlap = np.logical_and(posinf, neginf)
print(not np.any(overlap))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isposinf and isneginf should be mutually exclusive");
    Ok(())
}
