//! Conformance tests for numpy shape manipulation: squeeze, expand_dims, swapaxes.
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
// squeeze
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn squeeze_remove_all_1_dims() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[[1, 2, 3]]])  # shape (1, 1, 3)
result = fnp.squeeze(a)
expected = np.squeeze(a)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "squeeze remove all 1 dims should match numpy");
    Ok(())
}

#[test]
fn squeeze_specific_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[[1, 2, 3]]])  # shape (1, 1, 3)
result = fnp.squeeze(a, axis=0)
expected = np.squeeze(a, axis=0)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "squeeze specific axis should match numpy");
    Ok(())
}

#[test]
fn squeeze_no_1_dims() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.squeeze(a)
expected = np.squeeze(a)
print(np.array_equal(result, expected) and result.shape == a.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "squeeze no 1 dims should be no-op");
    Ok(())
}

#[test]
fn squeeze_negative_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1], [2], [3]])  # shape (3, 1)
result = fnp.squeeze(a, axis=-1)
expected = np.squeeze(a, axis=-1)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "squeeze negative axis should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// expand_dims
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn expand_dims_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.expand_dims(a, axis=0)
expected = np.expand_dims(a, axis=0)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "expand_dims axis=0 should match numpy");
    Ok(())
}

#[test]
fn expand_dims_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.expand_dims(a, axis=1)
expected = np.expand_dims(a, axis=1)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "expand_dims axis=1 should match numpy");
    Ok(())
}

#[test]
fn expand_dims_negative_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.expand_dims(a, axis=-1)
expected = np.expand_dims(a, axis=-1)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "expand_dims negative axis should match numpy");
    Ok(())
}

#[test]
fn expand_dims_2d_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.expand_dims(a, axis=1)
expected = np.expand_dims(a, axis=1)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "expand_dims 2d array should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// swapaxes
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn swapaxes_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.swapaxes(a, 0, 1)
expected = np.swapaxes(a, 0, 1)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "swapaxes 2d should match numpy");
    Ok(())
}

#[test]
fn swapaxes_3d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.swapaxes(a, 0, 2)
expected = np.swapaxes(a, 0, 2)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "swapaxes 3d should match numpy");
    Ok(())
}

#[test]
fn swapaxes_negative_axes() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.swapaxes(a, -1, -2)
expected = np.swapaxes(a, -1, -2)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "swapaxes negative axes should match numpy");
    Ok(())
}

#[test]
fn swapaxes_same_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.swapaxes(a, 0, 0)
expected = np.swapaxes(a, 0, 0)
print(np.array_equal(result, expected) and np.array_equal(result, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "swapaxes same axis should be identity");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn expand_dims_squeeze_roundtrip() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
expanded = fnp.expand_dims(a, axis=0)
result = fnp.squeeze(expanded)
print(np.array_equal(result, a) and result.shape == a.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "expand_dims then squeeze should be identity");
    Ok(())
}

#[test]
fn swapaxes_twice_is_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
swapped = fnp.swapaxes(a, 0, 1)
result = fnp.swapaxes(swapped, 0, 1)
print(np.array_equal(result, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "swapaxes twice should be identity");
    Ok(())
}

#[test]
fn swapaxes_transpose_equivalence_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
swap_result = fnp.swapaxes(a, 0, 1)
transpose_result = fnp.transpose(a)
print(np.array_equal(swap_result, transpose_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "swapaxes(0,1) should equal transpose for 2d");
    Ok(())
}
