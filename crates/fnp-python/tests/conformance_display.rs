//! Conformance tests for numpy display/string functions against NumPy oracle.
//!
//! Tests array_repr, array_str.

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
// array_repr
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn array_repr_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.array_repr(a)
expected = np.array_repr(a)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "array_repr 1d should match numpy");
    Ok(())
}

#[test]
fn array_repr_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.array_repr(a)
expected = np.array_repr(a)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "array_repr 2d should match numpy");
    Ok(())
}

#[test]
fn array_repr_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.5, 2.5, 3.5])
result = fnp.array_repr(a)
expected = np.array_repr(a)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "array_repr float should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// array_str
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn array_str_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.array_str(a)
expected = np.array_str(a)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "array_str 1d should match numpy");
    Ok(())
}

#[test]
fn array_str_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.array_str(a)
expected = np.array_str(a)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "array_str 2d should match numpy");
    Ok(())
}

#[test]
fn array_str_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.5, 2.5, 3.5])
result = fnp.array_str(a)
expected = np.array_str(a)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "array_str float should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn array_repr_contains_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.array_repr(a)
# array_repr should include "array(" prefix
print("array(" in result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "array_repr should contain 'array('");
    Ok(())
}

#[test]
fn array_str_vs_repr_different() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
str_result = fnp.array_str(a)
repr_result = fnp.array_repr(a)
# str and repr have different formats
print("array(" in repr_result and "array(" not in str_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "array_str and array_repr should differ"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// array2string and display configuration
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn array2string_printoptions_and_set_printoptions_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1.23456, 2000.0], [np.nan, -0.0]])
original = np.get_printoptions()
try:
    with np.printoptions(precision=2, suppress=True, linewidth=32):
        expected = np.array2string(a, separator=", ")
    with fnp.printoptions(precision=2, suppress=True, linewidth=32):
        result = fnp.array2string(a, separator=", ")

    fnp.set_printoptions(precision=3, threshold=4, edgeitems=1)
    ours = {key: np.get_printoptions()[key] for key in ["precision", "threshold", "edgeitems"]}
    np.set_printoptions(**original)
    np.set_printoptions(precision=3, threshold=4, edgeitems=1)
    expected_options = {
        key: np.get_printoptions()[key]
        for key in ["precision", "threshold", "edgeitems"]
    }
    print(result == expected and ours == expected_options)
finally:
    np.set_printoptions(**original)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "array2string/printoptions/set_printoptions should match numpy"
    );
    Ok(())
}

#[test]
fn setbufsize_and_get_include_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
original_size = np.setbufsize(8192)
try:
    ours_old_size = fnp.setbufsize(16384)
    np.setbufsize(8192)
    expected_old_size = np.setbufsize(16384)
    include_match = fnp.get_include() == np.get_include()
    print(ours_old_size == expected_old_size and include_match)
finally:
    np.setbufsize(original_size)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "setbufsize/get_include should match numpy"
    );
    Ok(())
}
