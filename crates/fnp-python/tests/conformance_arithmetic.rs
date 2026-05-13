//! Conformance tests for numpy arithmetic operations against NumPy oracle.
//!
//! Tests add, subtract, multiply, divide, negative, absolute, mod, remainder.

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
// add
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn add_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
result = fnp.add(a, b)
expected = np.add(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "add basic should match numpy");
    Ok(())
}

#[test]
fn add_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4])
result = fnp.add(a, 10)
expected = np.add(a, 10)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "add scalar should match numpy");
    Ok(())
}

#[test]
fn add_broadcast() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([10, 20])
result = fnp.add(a, b)
expected = np.add(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "add broadcast should match numpy");
    Ok(())
}

#[test]
fn add_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.5, 2.5, 3.5])
b = np.array([0.1, 0.2, 0.3])
result = fnp.add(a, b)
expected = np.add(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "add float should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// subtract
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn subtract_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([10, 20, 30, 40])
b = np.array([1, 2, 3, 4])
result = fnp.subtract(a, b)
expected = np.subtract(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "subtract basic should match numpy");
    Ok(())
}

#[test]
fn subtract_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([10, 20, 30])
result = fnp.subtract(a, 5)
expected = np.subtract(a, 5)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "subtract scalar should match numpy");
    Ok(())
}

#[test]
fn subtract_negative_result() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([5, 5, 5])
result = fnp.subtract(a, b)
expected = np.subtract(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "subtract negative result should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// multiply
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn multiply_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
result = fnp.multiply(a, b)
expected = np.multiply(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "multiply basic should match numpy");
    Ok(())
}

#[test]
fn multiply_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4])
result = fnp.multiply(a, 3)
expected = np.multiply(a, 3)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "multiply scalar should match numpy");
    Ok(())
}

#[test]
fn multiply_by_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4])
result = fnp.multiply(a, 0)
expected = np.multiply(a, 0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "multiply by zero should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// divide
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn divide_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([10.0, 20.0, 30.0])
b = np.array([2.0, 4.0, 5.0])
result = fnp.divide(a, b)
expected = np.divide(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "divide basic should match numpy");
    Ok(())
}

#[test]
fn divide_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([10.0, 20.0, 30.0])
result = fnp.divide(a, 2)
expected = np.divide(a, 2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "divide scalar should match numpy");
    Ok(())
}

#[test]
fn divide_by_zero_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
a = np.array([1.0, -1.0, 0.0])
b = np.array([0.0, 0.0, 0.0])
result = fnp.divide(a, b)
expected = np.divide(a, b)
# Both should have same inf/nan pattern
print(np.array_equal(np.isinf(result), np.isinf(expected)) and
      np.array_equal(np.isnan(result), np.isnan(expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "divide by zero should match numpy inf/nan pattern"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// negative
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn negative_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, -2, 3, -4])
result = fnp.negative(a)
expected = np.negative(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "negative basic should match numpy");
    Ok(())
}

#[test]
fn negative_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.5, -2.5, 0.0])
result = fnp.negative(a)
expected = np.negative(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "negative float should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// absolute / abs
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn absolute_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([-1, -2, 3, -4, 5])
result = fnp.absolute(a)
expected = np.absolute(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "absolute basic should match numpy");
    Ok(())
}

#[test]
fn absolute_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([-1.5, -2.5, 0.0, 3.5])
result = fnp.absolute(a)
expected = np.absolute(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "absolute float should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// mod / remainder
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mod_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([10, 20, 30, 40])
b = np.array([3, 7, 8, 9])
result = fnp.mod(a, b)
expected = np.mod(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "mod basic should match numpy");
    Ok(())
}

#[test]
fn mod_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([-10, 10, -10, 10])
b = np.array([3, 3, -3, -3])
result = fnp.mod(a, b)
expected = np.mod(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "mod negative should match numpy");
    Ok(())
}

#[test]
fn remainder_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([10, 20, 30])
b = np.array([3, 7, 8])
result = fnp.remainder(a, b)
expected = np.remainder(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "remainder basic should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// floor_divide
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn floor_divide_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([10, 20, 30])
b = np.array([3, 7, 8])
result = fnp.floor_divide(a, b)
expected = np.floor_divide(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "floor_divide basic should match numpy"
    );
    Ok(())
}

#[test]
fn floor_divide_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([-10, 10, -10, 10])
b = np.array([3, 3, -3, -3])
result = fnp.floor_divide(a, b)
expected = np.floor_divide(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "floor_divide negative should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn add_subtract_inverse() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
# (a + b) - b = a
result = fnp.subtract(fnp.add(a, b), b)
print(np.array_equal(result, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "add then subtract should return original"
    );
    Ok(())
}

#[test]
fn multiply_divide_inverse() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.array([2.0, 3.0, 4.0, 5.0])
# (a * b) / b = a
result = fnp.divide(fnp.multiply(a, b), b)
print(np.allclose(result, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "multiply then divide should return original"
    );
    Ok(())
}

#[test]
fn negative_double_inverse() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, -2, 3, -4])
# -(-a) = a
result = fnp.negative(fnp.negative(a))
print(np.array_equal(result, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "double negative should return original"
    );
    Ok(())
}

#[test]
fn floor_divide_mod_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([10, 23, 37, 45])
b = np.array([3, 7, 8, 9])
# a = (a // b) * b + (a % b)
reconstructed = fnp.add(fnp.multiply(fnp.floor_divide(a, b), b), fnp.mod(a, b))
print(np.array_equal(reconstructed, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "floor_divide and mod should reconstruct original"
    );
    Ok(())
}
