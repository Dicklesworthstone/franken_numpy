//! Conformance tests for numpy memory utility functions against NumPy oracle.
//!
//! Tests may_share_memory, shares_memory, result_type.

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
// may_share_memory
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn may_share_memory_same_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.may_share_memory(a, a)
expected = np.may_share_memory(a, a)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "may_share_memory same array should match numpy"
    );
    Ok(())
}

#[test]
fn may_share_memory_different_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.may_share_memory(a, b)
expected = np.may_share_memory(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "may_share_memory different arrays should match numpy"
    );
    Ok(())
}

#[test]
fn may_share_memory_view() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
b = a[1:4]  # view of a
result = fnp.may_share_memory(a, b)
expected = np.may_share_memory(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "may_share_memory view should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// shares_memory
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn shares_memory_same_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.shares_memory(a, a)
expected = np.shares_memory(a, a)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "shares_memory same array should match numpy"
    );
    Ok(())
}

#[test]
fn shares_memory_different_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.shares_memory(a, b)
expected = np.shares_memory(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "shares_memory different arrays should match numpy"
    );
    Ok(())
}

#[test]
fn shares_memory_view() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
b = a[1:4]  # view of a
result = fnp.shares_memory(a, b)
expected = np.shares_memory(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "shares_memory view should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// result_type
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn result_type_int_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2], dtype='int32')
b = np.array([1.0, 2.0], dtype='float64')
result = fnp.result_type(a, b)
expected = np.result_type(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "result_type int+float should match numpy"
    );
    Ok(())
}

#[test]
fn result_type_dtypes() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.result_type('float32', 'float64')
expected = np.result_type('float32', 'float64')
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "result_type dtypes should match numpy"
    );
    Ok(())
}

#[test]
fn result_type_scalars() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.result_type(3, 3.0)
expected = np.result_type(3, 3.0)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "result_type scalars should match numpy"
    );
    Ok(())
}

#[test]
fn result_type_multiple() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2], dtype='int8')
b = np.array([1, 2], dtype='int16')
c = np.array([1, 2], dtype='int32')
result = fnp.result_type(a, b, c)
expected = np.result_type(a, b, c)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "result_type multiple should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn shares_vs_may_share_same() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
# For same array, both should be True
shares = fnp.shares_memory(a, a)
may_share = fnp.may_share_memory(a, a)
print(shares == True and may_share == True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "both should be True for same array");
    Ok(())
}

#[test]
fn shares_vs_may_share_different() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
# For different arrays, both should be False
shares = fnp.shares_memory(a, b)
may_share = fnp.may_share_memory(a, b)
print(shares == False and may_share == False)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "both should be False for different arrays"
    );
    Ok(())
}
