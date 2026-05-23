//! Conformance tests for numpy.concatenate, append, insert, delete against NumPy oracle.
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
// concatenate
// ──────────────���──────────────────────────────────────────────────────────────

#[test]
fn concatenate_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.concatenate([a, b])
expected = np.concatenate([a, b])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "concatenate 1d should match numpy");
    Ok(())
}

#[test]
fn concatenate_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = fnp.concatenate([a, b], axis=0)
expected = np.concatenate([a, b], axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "concatenate 2d axis=0 should match numpy"
    );
    Ok(())
}

#[test]
fn concatenate_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = fnp.concatenate([a, b], axis=1)
expected = np.concatenate([a, b], axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "concatenate 2d axis=1 should match numpy"
    );
    Ok(())
}

#[test]
fn concatenate_multiple_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2])
b = np.array([3, 4])
c = np.array([5, 6])
result = fnp.concatenate([a, b, c])
expected = np.concatenate([a, b, c])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "concatenate multiple arrays should match numpy"
    );
    Ok(())
}

#[test]
fn concatenate_axis_none() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
result = fnp.concatenate([a, b], axis=None)
expected = np.concatenate([a, b], axis=None)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "concatenate axis=None should flatten and match numpy"
    );
    Ok(())
}

#[test]
fn concatenate_negative_axis_3d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(12, dtype=np.int16).reshape(2, 3, 2)
b = np.arange(12, 24, dtype=np.int16).reshape(2, 3, 2)
result = fnp.concatenate([a, b], axis=-1)
expected = np.concatenate([a, b], axis=-1)
print(result.dtype == expected.dtype and result.shape == expected.shape and np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "concatenate negative axis should match numpy"
    );
    Ok(())
}

#[test]
fn concatenate_empty_axis_preserves_shape_and_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.empty((2, 0), dtype=np.float32)
b = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
result = fnp.concatenate([a, b, a], axis=1)
expected = np.concatenate([a, b, a], axis=1)
print(result.dtype == expected.dtype and result.shape == expected.shape and np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "concatenate empty axis operands should match numpy"
    );
    Ok(())
}

#[test]
fn concatenate_structured_arrays_preserves_fields() -> Result<(), String> {
    let script = fnp_script(
        r#"
dtype = np.dtype([("left", "i4"), ("right", "f8")])
a = np.array([(1, 1.5), (2, 2.5)], dtype=dtype)
b = np.array([(3, 3.5)], dtype=dtype)
result = fnp.concatenate([a, b])
expected = np.concatenate([a, b])
print(result.dtype == expected.dtype and result.shape == expected.shape and np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "concatenate structured arrays should match numpy"
    );
    Ok(())
}

// ��────────────────────────────────────────────────────────────────────────────
// append
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn append_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.append(a, [4, 5, 6])
expected = np.append(a, [4, 5, 6])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "append 1d should match numpy");
    Ok(())
}

#[test]
fn append_single_value() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.append(a, 4)
expected = np.append(a, 4)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "append single value should match numpy"
    );
    Ok(())
}

#[test]
fn append_2d_no_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.append(a, [[5, 6]])
expected = np.append(a, [[5, 6]])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "append 2d no axis should flatten and match numpy"
    );
    Ok(())
}

#[test]
fn append_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.append(a, [[5, 6]], axis=0)
expected = np.append(a, [[5, 6]], axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "append 2d axis=0 should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// insert
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn insert_single_value() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4])
result = fnp.insert(a, 2, 99)
expected = np.insert(a, 2, 99)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "insert single value should match numpy"
    );
    Ok(())
}

#[test]
fn insert_multiple_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4])
result = fnp.insert(a, 2, [98, 99])
expected = np.insert(a, 2, [98, 99])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "insert multiple values should match numpy"
    );
    Ok(())
}

#[test]
fn insert_at_beginning() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.insert(a, 0, 0)
expected = np.insert(a, 0, 0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "insert at beginning should match numpy"
    );
    Ok(())
}

#[test]
fn insert_at_end() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.insert(a, 3, 4)
expected = np.insert(a, 3, 4)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "insert at end should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// delete
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn delete_single_index() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.delete(a, 2)
expected = np.delete(a, 2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "delete single index should match numpy"
    );
    Ok(())
}

#[test]
fn delete_multiple_indices() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.delete(a, [1, 3])
expected = np.delete(a, [1, 3])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "delete multiple indices should match numpy"
    );
    Ok(())
}

#[test]
fn delete_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4], [5, 6]])
result = fnp.delete(a, 1, axis=0)
expected = np.delete(a, 1, axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "delete 2d axis=0 should match numpy");
    Ok(())
}

#[test]
fn delete_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.delete(a, 1, axis=1)
expected = np.delete(a, 1, axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "delete 2d axis=1 should match numpy");
    Ok(())
}

#[test]
fn delete_negative_index() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.delete(a, -1)
expected = np.delete(a, -1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "delete negative index should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Complex dtype tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn concatenate_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j], dtype=np.complex128)
b = np.array([3+2j, 4-2j], dtype=np.complex128)
fnp_result = fnp.concatenate([a, b])
np_result = np.concatenate([a, b])
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "concatenate complex should match numpy"
    );
    Ok(())
}

#[test]
fn append_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j], dtype=np.complex128)
b = np.array([3+2j, 4-2j], dtype=np.complex128)
fnp_result = fnp.append(a, b)
np_result = np.append(a, b)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "append complex should match numpy");
    Ok(())
}

#[test]
fn insert_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j, 3+2j], dtype=np.complex128)
fnp_result = fnp.insert(a, 1, 9+9j)
np_result = np.insert(a, 1, 9+9j)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "insert complex should match numpy");
    Ok(())
}

#[test]
fn delete_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j, 3+2j, 4-2j], dtype=np.complex128)
fnp_result = fnp.delete(a, 1)
np_result = np.delete(a, 1)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "delete complex should match numpy");
    Ok(())
}
