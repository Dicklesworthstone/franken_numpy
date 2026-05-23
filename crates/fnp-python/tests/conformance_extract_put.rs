//! Conformance tests for numpy.extract, put, place, putmask against NumPy oracle.
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
// extract
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn extract_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([False, True, False, True, True])
arr = np.array([1, 2, 3, 4, 5])
result = fnp.extract(condition, arr)
expected = np.extract(condition, arr)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "extract basic should match numpy");
    Ok(())
}

#[test]
fn extract_2d_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([[True, False], [False, True]])
arr = np.array([[1, 2], [3, 4]])
result = fnp.extract(condition, arr)
expected = np.extract(condition, arr)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "extract 2d array should match numpy");
    Ok(())
}

#[test]
fn extract_all_false() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([False, False, False])
arr = np.array([1, 2, 3])
result = fnp.extract(condition, arr)
expected = np.extract(condition, arr)
print(np.array_equal(result, expected) and len(result) == 0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "extract all false should return empty array"
    );
    Ok(())
}

#[test]
fn extract_all_true() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, True, True])
arr = np.array([1, 2, 3])
result = fnp.extract(condition, arr)
expected = np.extract(condition, arr)
print(np.array_equal(result, expected) and np.array_equal(result, arr))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "extract all true should return all elements"
    );
    Ok(())
}

#[test]
fn extract_float_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False, True])
arr = np.array([1.5, 2.5, 3.5])
result = fnp.extract(condition, arr)
expected = np.extract(condition, arr)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "extract float array should match numpy"
    );
    Ok(())
}

#[test]
fn extract_string_payload_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False, True])
arr = np.array(["alpha", "beta", "gamma"])
result = fnp.extract(condition, arr)
expected = np.extract(condition, arr)
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "extract should preserve NumPy string payload behavior"
    );
    Ok(())
}

#[test]
fn extract_string_condition_truthiness_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array(["", "x", "0"])
arr = np.array([10, 20, 30])
result = fnp.extract(condition, arr)
expected = np.extract(condition, arr)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "extract should match NumPy string condition truthiness"
    );
    Ok(())
}

#[test]
fn extract_object_condition_truthiness_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([object(), None, 1], dtype=object)
arr = np.array([10, 20, 30])
result = fnp.extract(condition, arr)
expected = np.extract(condition, arr)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "extract should match NumPy object condition truthiness"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// put
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn put_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
fnp.put(a, [0, 2], [10, 30])
b = np.array([1, 2, 3, 4, 5])
np.put(b, [0, 2], [10, 30])
print(np.array_equal(a, b))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "put basic should match numpy");
    Ok(())
}

#[test]
fn put_single_value() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
fnp.put(a, [0, 2, 4], [99])
b = np.array([1, 2, 3, 4, 5])
np.put(b, [0, 2, 4], [99])
print(np.array_equal(a, b))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "put single value should broadcast");
    Ok(())
}

#[test]
fn put_negative_indices() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
fnp.put(a, [-1, -2], [50, 40])
b = np.array([1, 2, 3, 4, 5])
np.put(b, [-1, -2], [50, 40])
print(np.array_equal(a, b))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "put negative indices should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// place
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn place_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
mask = np.array([True, False, True, False, True])
fnp.place(a, mask, [10, 30, 50])
b = np.array([1, 2, 3, 4, 5])
np.place(b, mask, [10, 30, 50])
print(np.array_equal(a, b))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "place basic should match numpy");
    Ok(())
}

#[test]
fn place_cycle_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
mask = np.array([True, True, True, True, True])
fnp.place(a, mask, [10, 20])  # cycles: 10, 20, 10, 20, 10
b = np.array([1, 2, 3, 4, 5])
np.place(b, mask, [10, 20])
print(np.array_equal(a, b))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "place should cycle values");
    Ok(())
}

#[test]
fn place_2d_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
mask = np.array([[True, False], [False, True]])
fnp.place(a, mask, [10, 40])
b = np.array([[1, 2], [3, 4]])
np.place(b, mask, [10, 40])
print(np.array_equal(a, b))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "place 2d array should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// putmask
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn putmask_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
mask = np.array([True, False, True, False, True])
fnp.putmask(a, mask, [10, 30, 50])
b = np.array([1, 2, 3, 4, 5])
np.putmask(b, mask, [10, 30, 50])
print(np.array_equal(a, b))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "putmask basic should match numpy");
    Ok(())
}

#[test]
fn putmask_scalar_value() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
mask = np.array([True, False, True, False, True])
fnp.putmask(a, mask, [99])
b = np.array([1, 2, 3, 4, 5])
np.putmask(b, mask, [99])
print(np.array_equal(a, b))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "putmask scalar value should broadcast"
    );
    Ok(())
}

#[test]
fn putmask_2d_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
mask = np.array([[True, False], [False, True]])
fnp.putmask(a, mask, [10, 40])
b = np.array([[1, 2], [3, 4]])
np.putmask(b, mask, [10, 40])
print(np.array_equal(a, b))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "putmask 2d array should match numpy");
    Ok(())
}

#[test]
fn putmask_all_false() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
mask = np.array([False, False, False, False, False])
original = a.copy()
fnp.putmask(a, mask, [99])
print(np.array_equal(a, original))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "putmask all false should not modify array"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn extract_where_equivalence() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False, True, False, True])
arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
extract_result = fnp.extract(condition, arr)
# where in index mode + indexing = extract
indices = fnp.where(condition)[0]
where_result = arr[indices]
print(np.array_equal(extract_result, where_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "extract should be equivalent to where + indexing"
    );
    Ok(())
}

#[test]
fn extract_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False, True, False])
arr = np.array([1+1j, 2-1j, 3+2j, 4-2j], dtype=np.complex128)
fnp_result = fnp.extract(condition, arr)
np_result = np.extract(condition, arr)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "extract complex should match numpy");
    Ok(())
}

#[test]
fn place_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
arr1 = np.array([1+1j, 2-1j, 3+2j, 4-2j], dtype=np.complex128)
arr2 = np.array([1+1j, 2-1j, 3+2j, 4-2j], dtype=np.complex128)
mask = np.array([True, False, True, False])
vals = np.array([9+9j, 8+8j], dtype=np.complex128)
fnp.place(arr1, mask, vals)
np.place(arr2, mask, vals)
print(np.array_equal(arr1, arr2))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "place complex should match numpy");
    Ok(())
}

#[test]
fn putmask_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
arr1 = np.array([1+1j, 2-1j, 3+2j, 4-2j], dtype=np.complex128)
arr2 = np.array([1+1j, 2-1j, 3+2j, 4-2j], dtype=np.complex128)
mask = np.array([True, False, True, False])
vals = np.array([9+9j, 8+8j], dtype=np.complex128)
fnp.putmask(arr1, mask, vals)
np.putmask(arr2, mask, vals)
print(np.array_equal(arr1, arr2))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "putmask complex should match numpy");
    Ok(())
}

#[test]
fn put_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
arr1 = np.array([1+1j, 2-1j, 3+2j, 4-2j], dtype=np.complex128)
arr2 = np.array([1+1j, 2-1j, 3+2j, 4-2j], dtype=np.complex128)
indices = [0, 2]
vals = np.array([9+9j, 8+8j], dtype=np.complex128)
fnp.put(arr1, indices, vals)
np.put(arr2, indices, vals)
print(np.array_equal(arr1, arr2))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "put complex should match numpy");
    Ok(())
}
