//! Conformance tests for numpy asarray and asanyarray against NumPy oracle.
//!
//! Tests asarray, asanyarray, fromstring, frombuffer.

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
// asarray
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn asarray_from_list() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.asarray([1, 2, 3])
expected = np.asarray([1, 2, 3])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "asarray from list should match numpy");
    Ok(())
}

#[test]
fn asarray_from_nested_list() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.asarray([[1, 2], [3, 4]])
expected = np.asarray([[1, 2], [3, 4]])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "asarray from nested list should match numpy");
    Ok(())
}

#[test]
fn asarray_with_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.asarray([1, 2, 3], dtype='float64')
expected = np.asarray([1, 2, 3], dtype='float64')
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "asarray with dtype should match numpy");
    Ok(())
}

#[test]
fn asarray_from_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.asarray(42)
expected = np.asarray(42)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "asarray from scalar should match numpy");
    Ok(())
}

#[test]
fn asarray_from_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.asarray(a)
expected = np.asarray(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "asarray from array should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// asanyarray
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn asanyarray_from_list() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.asanyarray([1, 2, 3])
expected = np.asanyarray([1, 2, 3])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "asanyarray from list should match numpy");
    Ok(())
}

#[test]
fn asanyarray_with_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.asanyarray([1, 2, 3], dtype='float32')
expected = np.asanyarray([1, 2, 3], dtype='float32')
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "asanyarray with dtype should match numpy");
    Ok(())
}

#[test]
fn asanyarray_from_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.asanyarray(a)
expected = np.asanyarray(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "asanyarray from array should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// fromstring
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fromstring_with_sep() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.fromstring('1 2 3 4', sep=' ')
expected = np.fromstring('1 2 3 4', sep=' ')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fromstring with sep should match numpy");
    Ok(())
}

#[test]
fn fromstring_with_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.fromstring('1 2 3 4', sep=' ', dtype='int32')
expected = np.fromstring('1 2 3 4', sep=' ', dtype='int32')
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fromstring with dtype should match numpy");
    Ok(())
}

#[test]
fn fromstring_comma_sep() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.fromstring('1,2,3,4', sep=',')
expected = np.fromstring('1,2,3,4', sep=',')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fromstring comma sep should match numpy");
    Ok(())
}

#[test]
fn fromstring_with_count() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.fromstring('1 2 3 4 5', sep=' ', count=3)
expected = np.fromstring('1 2 3 4 5', sep=' ', count=3)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fromstring with count should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn asarray_asanyarray_equivalence_for_lists() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = [[1, 2, 3], [4, 5, 6]]
asarray_result = fnp.asarray(data)
asanyarray_result = fnp.asanyarray(data)
print(np.array_equal(asarray_result, asanyarray_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "asarray and asanyarray should be equivalent for lists");
    Ok(())
}

#[test]
fn asarray_preserves_existing_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.asarray(a)
# For same dtype, asarray should return the input
print(result.shape == a.shape and result.dtype == a.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "asarray should preserve existing array");
    Ok(())
}
