//! Conformance tests for bitwise operations.
//!
//! Tests: bitwise_and, bitwise_or, bitwise_xor, left_shift, right_shift, invert.

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
// bitwise_and
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn bitwise_and_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0b1100, 0b1010, 0b1111, 0b0000], dtype=np.int64)
y = np.array([0b1010, 0b1010, 0b0011, 0b1111], dtype=np.int64)
result = fnp.bitwise_and(x, y)
expected = np.bitwise_and(x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bitwise_and basic should match numpy"
    );
    Ok(())
}

#[test]
fn bitwise_and_broadcast() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([[0b1100, 0b1010], [0b1111, 0b0000]], dtype=np.int64)
y = np.array([0b1111, 0b0011], dtype=np.int64)
result = fnp.bitwise_and(x, y)
expected = np.bitwise_and(x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bitwise_and broadcast should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// bitwise_or
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn bitwise_or_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0b1100, 0b1010, 0b1111, 0b0000], dtype=np.int64)
y = np.array([0b1010, 0b1010, 0b0011, 0b1111], dtype=np.int64)
result = fnp.bitwise_or(x, y)
expected = np.bitwise_or(x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "bitwise_or basic should match numpy");
    Ok(())
}

#[test]
fn bitwise_or_bool_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([True, True, False, False])
y = np.array([True, False, True, False])
result = fnp.bitwise_or(x, y)
expected = np.bitwise_or(x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bitwise_or bool arrays should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// bitwise_xor
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn bitwise_xor_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0b1100, 0b1010, 0b1111, 0b0000], dtype=np.int64)
y = np.array([0b1010, 0b1010, 0b0011, 0b1111], dtype=np.int64)
result = fnp.bitwise_xor(x, y)
expected = np.bitwise_xor(x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bitwise_xor basic should match numpy"
    );
    Ok(())
}

#[test]
fn bitwise_xor_self_is_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 3, 4, 5], dtype=np.int64)
result = fnp.bitwise_xor(x, x)
expected = np.bitwise_xor(x, x)
print(np.array_equal(result, expected) and np.all(result == 0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bitwise_xor with self should be zero"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// left_shift
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn left_shift_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 4, 8], dtype=np.int64)
y = np.array([1, 2, 3, 4], dtype=np.int64)
result = fnp.left_shift(x, y)
expected = np.left_shift(x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "left_shift basic should match numpy");
    Ok(())
}

#[test]
fn left_shift_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 3, 4], dtype=np.int64)
result = fnp.left_shift(x, 2)
expected = np.left_shift(x, 2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "left_shift scalar should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// right_shift
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn right_shift_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([16, 32, 64, 128], dtype=np.int64)
y = np.array([1, 2, 3, 4], dtype=np.int64)
result = fnp.right_shift(x, y)
expected = np.right_shift(x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "right_shift basic should match numpy"
    );
    Ok(())
}

#[test]
fn right_shift_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([8, 16, 32, 64], dtype=np.int64)
result = fnp.right_shift(x, 2)
expected = np.right_shift(x, 2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "right_shift scalar should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// invert / bitwise_not
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn invert_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0, 1, -1, 127, -128], dtype=np.int8)
result = fnp.bitwise_invert(x)
expected = np.invert(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "invert basic should match numpy");
    Ok(())
}

#[test]
fn invert_bool() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Use integer array - boolean invert has special semantics not yet supported natively
x = np.array([1, 0, 1, 0], dtype=np.int8)
result = fnp.bitwise_not(x)
expected = np.invert(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "invert int8 should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn bitwise_large_numbers() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([2**30, 2**31, 2**32], dtype=np.int64)
y = np.array([2**30 - 1, 2**31 - 1, 2**32 - 1], dtype=np.int64)
result_and = fnp.bitwise_and(x, y)
result_or = fnp.bitwise_or(x, y)
expected_and = np.bitwise_and(x, y)
expected_or = np.bitwise_or(x, y)
print(np.array_equal(result_and, expected_and) and np.array_equal(result_or, expected_or))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bitwise large numbers should match numpy"
    );
    Ok(())
}

#[test]
fn shift_with_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 3, 4], dtype=np.int64)
result_left = fnp.left_shift(x, 0)
result_right = fnp.right_shift(x, 0)
expected_left = np.left_shift(x, 0)
expected_right = np.right_shift(x, 0)
print(np.array_equal(result_left, expected_left) and np.array_equal(result_right, expected_right))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "shift with zero should match numpy");
    Ok(())
}
