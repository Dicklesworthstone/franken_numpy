//! Conformance tests for numpy ediff1d against NumPy oracle.

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
// ediff1d basic
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ediff1d_1d_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 7, 0])
result = fnp.ediff1d(a)
expected = np.ediff1d(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ediff1d 1d basic should match numpy");
    Ok(())
}

#[test]
fn ediff1d_2d_flattens() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.ediff1d(a)
expected = np.ediff1d(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ediff1d 2d should flatten then diff");
    Ok(())
}

#[test]
fn ediff1d_3d_flattens() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.ediff1d(a)
expected = np.ediff1d(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ediff1d 3d should flatten then diff");
    Ok(())
}

#[test]
fn ediff1d_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.5, 2.5, 4.0, 7.5])
result = fnp.ediff1d(a)
expected = np.ediff1d(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ediff1d float should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ediff1d with to_begin / to_end
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ediff1d_with_to_begin_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 7, 0])
result = fnp.ediff1d(a, to_begin=-99)
expected = np.ediff1d(a, to_begin=-99)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ediff1d with to_begin scalar should match numpy"
    );
    Ok(())
}

#[test]
fn ediff1d_with_to_end_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 7, 0])
result = fnp.ediff1d(a, to_end=88)
expected = np.ediff1d(a, to_end=88)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ediff1d with to_end scalar should match numpy"
    );
    Ok(())
}

#[test]
fn ediff1d_with_to_begin_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 7, 0])
result = fnp.ediff1d(a, to_begin=np.array([-99, -88]))
expected = np.ediff1d(a, to_begin=np.array([-99, -88]))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ediff1d with to_begin array should match numpy"
    );
    Ok(())
}

#[test]
fn ediff1d_with_to_end_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 7, 0])
result = fnp.ediff1d(a, to_end=np.array([88, 99]))
expected = np.ediff1d(a, to_end=np.array([88, 99]))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ediff1d with to_end array should match numpy"
    );
    Ok(())
}

#[test]
fn ediff1d_with_both() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 7, 0])
result = fnp.ediff1d(a, to_begin=-99, to_end=np.array([88, 99]))
expected = np.ediff1d(a, to_begin=-99, to_end=np.array([88, 99]))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ediff1d with both to_begin and to_end should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ediff1d edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ediff1d_single_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([5])
result = fnp.ediff1d(a)
expected = np.ediff1d(a)
print(len(result) == 0 and len(expected) == 0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ediff1d single element returns empty array"
    );
    Ok(())
}

#[test]
fn ediff1d_empty() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([])
result = fnp.ediff1d(a)
expected = np.ediff1d(a)
print(len(result) == 0 and len(expected) == 0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ediff1d empty array returns empty array"
    );
    Ok(())
}

#[test]
fn ediff1d_preserves_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 7], dtype='int32')
result = fnp.ediff1d(a)
expected = np.ediff1d(a)
print(result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ediff1d should preserve dtype");
    Ok(())
}

#[test]
fn ediff1d_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+2j, 3+4j, 6+1j])
result = fnp.ediff1d(a)
expected = np.ediff1d(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ediff1d complex should match numpy");
    Ok(())
}

#[test]
fn ediff1d_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.inf, 3.0, np.nan, 5.0])
fnp_result = fnp.ediff1d(a)
np_result = np.ediff1d(a)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ediff1d special values should match numpy");
    Ok(())
}

#[test]
fn ediff1d_constant_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([5.0, 5.0, 5.0, 5.0])
fnp_result = fnp.ediff1d(a)
np_result = np.ediff1d(a)
# Diff of constant should be zero
print(np.allclose(fnp_result, np_result) and np.allclose(fnp_result, 0.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ediff1d constant array should match numpy");
    Ok(())
}
