//! Conformance tests for numpy bitwise shift functions against NumPy oracle.
//!
//! Tests bitwise_left_shift, bitwise_right_shift, bitwise_count.

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
// bitwise_left_shift
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn bitwise_left_shift_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 8], dtype='int64')
result = fnp.bitwise_left_shift(a, 1)
expected = np.left_shift(a, 1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "bitwise_left_shift basic should match numpy");
    Ok(())
}

#[test]
fn bitwise_left_shift_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 8], dtype='int64')
b = np.array([1, 2, 3, 4], dtype='int64')
result = fnp.bitwise_left_shift(a, b)
expected = np.left_shift(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "bitwise_left_shift array should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// bitwise_right_shift
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn bitwise_right_shift_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([16, 32, 64, 128], dtype='int64')
result = fnp.bitwise_right_shift(a, 1)
expected = np.right_shift(a, 1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "bitwise_right_shift basic should match numpy");
    Ok(())
}

#[test]
fn bitwise_right_shift_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([16, 32, 64, 128], dtype='int64')
b = np.array([1, 2, 3, 4], dtype='int64')
result = fnp.bitwise_right_shift(a, b)
expected = np.right_shift(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "bitwise_right_shift array should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// bitwise_count
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn bitwise_count_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 7, 15, 255], dtype='uint8')
result = fnp.bitwise_count(a)
expected = np.bitwise_count(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "bitwise_count basic should match numpy");
    Ok(())
}

#[test]
fn bitwise_count_int64() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 3, 7, 15], dtype='int64')
result = fnp.bitwise_count(a)
expected = np.bitwise_count(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "bitwise_count int64 should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn left_right_shift_inverse() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 8], dtype='int64')
shifted = fnp.bitwise_left_shift(a, 2)
back = fnp.bitwise_right_shift(shifted, 2)
print(np.array_equal(a, back))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "left then right shift should restore original");
    Ok(())
}

#[test]
fn bitwise_count_power_of_2() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Powers of 2 have exactly one bit set
powers = np.array([1, 2, 4, 8, 16, 32, 64], dtype='int64')
counts = fnp.bitwise_count(powers)
print(np.all(counts == 1))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "powers of 2 should have count 1");
    Ok(())
}
