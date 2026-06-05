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

#[test]
fn arithmetic_scalar_return_type_matches_numpy() -> Result<(), String> {
    let binary_funcs = [
        "add",
        "subtract",
        "multiply",
        "divide",
        "floor_divide",
        "true_divide",
        "mod",
    ];
    for func in binary_funcs {
        let script = fnp_script(format!(
            r#"
x = np.float64(10.0)
y = np.float64(3.0)
fnp_result = fnp.{func}(x, y)
np_result = np.{func}(x, y)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        ));
        let result = numpy_oracle(&script)?;
        assert!(
            result.trim().starts_with("True"),
            "{func} scalar return type should match numpy: {result}"
        );
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Integer overflow behavior
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn add_int64_overflow_wraps() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([np.iinfo(np.int64).max], dtype=np.int64)
b = np.array([1], dtype=np.int64)
result = fnp.add(a, b)
expected = np.add(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "int64 overflow should wrap like numpy"
    );
    Ok(())
}

#[test]
fn multiply_int64_overflow_wraps() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([np.iinfo(np.int64).max], dtype=np.int64)
b = np.array([2], dtype=np.int64)
result = fnp.multiply(a, b)
expected = np.multiply(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "int64 multiply overflow should wrap like numpy"
    );
    Ok(())
}

#[test]
fn subtract_int64_underflow_wraps() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([np.iinfo(np.int64).min], dtype=np.int64)
b = np.array([1], dtype=np.int64)
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
        "int64 underflow should wrap like numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Signed-zero parity tests (critical for parallel operation safety)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn subtract_signed_zero_parity() -> Result<(), String> {
    let script = fnp_script(
        r#"
# IEEE 754 signed-zero subtract semantics
# 0.0 - 0.0 = 0.0, -0.0 - (-0.0) = 0.0, 0.0 - (-0.0) = 0.0, -0.0 - 0.0 = -0.0
tests = []
cases = [
    (0.0, 0.0),
    (-0.0, -0.0),
    (0.0, -0.0),
    (-0.0, 0.0),
    (1.0, 1.0),    # 1.0 - 1.0 = 0.0 (positive zero)
    (-1.0, -1.0),  # -1.0 - (-1.0) = 0.0
]
for a, b in cases:
    fnp_result = fnp.subtract(np.float64(a), np.float64(b))
    np_result = np.subtract(np.float64(a), np.float64(b))
    # Check both value and sign bit
    value_match = fnp_result == np_result or (np.isnan(fnp_result) and np.isnan(np_result))
    sign_match = np.signbit(fnp_result) == np.signbit(np_result)
    tests.append(value_match and sign_match)
print(all(tests))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "subtract signed zero parity should match numpy"
    );
    Ok(())
}

#[test]
fn add_signed_zero_parity() -> Result<(), String> {
    let script = fnp_script(
        r#"
# IEEE 754 signed-zero add semantics
# 0.0 + 0.0 = 0.0, -0.0 + (-0.0) = -0.0, 0.0 + (-0.0) = 0.0, -0.0 + 0.0 = 0.0
tests = []
cases = [
    (0.0, 0.0),
    (-0.0, -0.0),
    (0.0, -0.0),
    (-0.0, 0.0),
]
for a, b in cases:
    fnp_result = fnp.add(np.float64(a), np.float64(b))
    np_result = np.add(np.float64(a), np.float64(b))
    # Check both value and sign bit
    value_match = fnp_result == np_result
    sign_match = np.signbit(fnp_result) == np.signbit(np_result)
    tests.append(value_match and sign_match)
print(all(tests))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "add signed zero parity should match numpy"
    );
    Ok(())
}

#[test]
fn multiply_signed_zero_parity() -> Result<(), String> {
    let script = fnp_script(
        r#"
# IEEE 754 signed-zero multiply semantics
# Sign of result is XOR of operand signs
tests = []
cases = [
    (0.0, 1.0),    # 0.0 * 1.0 = 0.0
    (-0.0, 1.0),   # -0.0 * 1.0 = -0.0
    (0.0, -1.0),   # 0.0 * -1.0 = -0.0
    (-0.0, -1.0),  # -0.0 * -1.0 = 0.0
    (0.0, 0.0),
    (-0.0, -0.0),
    (0.0, -0.0),
    (-0.0, 0.0),
]
for a, b in cases:
    fnp_result = fnp.multiply(np.float64(a), np.float64(b))
    np_result = np.multiply(np.float64(a), np.float64(b))
    value_match = fnp_result == np_result
    sign_match = np.signbit(fnp_result) == np.signbit(np_result)
    tests.append(value_match and sign_match)
print(all(tests))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "multiply signed zero parity should match numpy"
    );
    Ok(())
}

#[test]
fn divide_signed_zero_parity() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
# IEEE 754 signed-zero divide semantics
tests = []
cases = [
    (0.0, 1.0),    # 0.0 / 1.0 = 0.0
    (-0.0, 1.0),   # -0.0 / 1.0 = -0.0
    (0.0, -1.0),   # 0.0 / -1.0 = -0.0
    (-0.0, -1.0),  # -0.0 / -1.0 = 0.0
    (1.0, np.inf),  # 1.0 / inf = 0.0
    (-1.0, np.inf), # -1.0 / inf = -0.0
    (1.0, -np.inf), # 1.0 / -inf = -0.0
    (-1.0, -np.inf), # -1.0 / -inf = 0.0
]
for a, b in cases:
    fnp_result = fnp.divide(np.float64(a), np.float64(b))
    np_result = np.divide(np.float64(a), np.float64(b))
    value_match = fnp_result == np_result or (np.isnan(fnp_result) and np.isnan(np_result))
    sign_match = np.signbit(fnp_result) == np.signbit(np_result)
    tests.append(value_match and sign_match)
print(all(tests))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "divide signed zero parity should match numpy"
    );
    Ok(())
}

#[test]
fn arithmetic_special_values_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
tests = []
# inf arithmetic
cases = [
    ('add', np.inf, 1.0, np.inf),
    ('add', -np.inf, 1.0, -np.inf),
    ('add', np.inf, -np.inf, np.nan),
    ('subtract', np.inf, 1.0, np.inf),
    ('subtract', np.inf, np.inf, np.nan),
    ('multiply', np.inf, 2.0, np.inf),
    ('multiply', np.inf, -2.0, -np.inf),
    ('multiply', np.inf, 0.0, np.nan),
    ('divide', np.inf, 2.0, np.inf),
    ('divide', 1.0, 0.0, np.inf),
    ('divide', -1.0, 0.0, -np.inf),
    ('divide', 0.0, 0.0, np.nan),
]
for op, a, b, expected in cases:
    fnp_func = getattr(fnp, op)
    np_func = getattr(np, op)
    fnp_result = fnp_func(np.float64(a), np.float64(b))
    np_result = np_func(np.float64(a), np.float64(b))
    if np.isnan(expected):
        match = np.isnan(fnp_result) and np.isnan(np_result)
    elif np.isinf(expected):
        match = np.isinf(fnp_result) and np.isinf(np_result) and np.sign(fnp_result) == np.sign(np_result)
    else:
        match = fnp_result == np_result
    tests.append(match)
print(all(tests))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "arithmetic special values inf should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// dtype promotion tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn add_dtype_promotion_int_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype=np.int32)
b = np.array([1.5, 2.5, 3.5], dtype=np.float32)
fnp_result = fnp.add(a, b)
np_result = np.add(a, b)
print(fnp_result.dtype == np_result.dtype, fnp_result.dtype, np_result.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "add dtype promotion int32+float32 should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn add_dtype_promotion_int_sizes() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype=np.int8)
b = np.array([1, 2, 3], dtype=np.int64)
fnp_result = fnp.add(a, b)
np_result = np.add(a, b)
print(fnp_result.dtype == np_result.dtype, fnp_result.dtype, np_result.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "add dtype promotion int8+int64 should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn add_dtype_promotion_float_sizes() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0], dtype=np.float32)
b = np.array([1.0], dtype=np.float64)
fnp_result = fnp.add(a, b)
np_result = np.add(a, b)
print(fnp_result.dtype == np_result.dtype, fnp_result.dtype, np_result.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "add dtype promotion float32+float64 should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn add_dtype_promotion_float_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0], dtype=np.float64)
b = np.array([1+0j], dtype=np.complex128)
fnp_result = fnp.add(a, b)
np_result = np.add(a, b)
print(fnp_result.dtype == np_result.dtype, fnp_result.dtype, np_result.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "add dtype promotion float64+complex128 should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn add_dtype_promotion_unsigned_signed() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1], dtype=np.uint8)
b = np.array([1], dtype=np.int8)
fnp_result = fnp.add(a, b)
np_result = np.add(a, b)
print(fnp_result.dtype == np_result.dtype, fnp_result.dtype, np_result.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "add dtype promotion uint8+int8 should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn add_dtype_promotion_bool_int() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([True, False])
b = np.array([1, 2])
fnp_result = fnp.add(a, b)
np_result = np.add(a, b)
print(fnp_result.dtype == np_result.dtype, fnp_result.dtype, np_result.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "add dtype promotion bool+int should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn multiply_dtype_promotion_int_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype=np.int32)
b = np.array([1.5, 2.5, 3.5], dtype=np.float32)
fnp_result = fnp.multiply(a, b)
np_result = np.multiply(a, b)
print(fnp_result.dtype == np_result.dtype, fnp_result.dtype, np_result.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "multiply dtype promotion int32*float32 should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn divide_dtype_promotion_int_int() -> Result<(), String> {
    let script = fnp_script(
        r#"
# integer division promotes to float64
a = np.array([10, 20, 30], dtype=np.int32)
b = np.array([3, 4, 5], dtype=np.int32)
fnp_result = fnp.divide(a, b)
np_result = np.divide(a, b)
print(fnp_result.dtype == np_result.dtype, fnp_result.dtype, np_result.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "divide dtype promotion int32/int32 should match numpy: {result}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// out parameter tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn add_out_parameter_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
out = np.zeros(3)
fnp_result = fnp.add(a, b, out=out)
np_out = np.zeros(3)
np_result = np.add(a, b, out=np_out)
print(fnp_result is out, np_result is np_out)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True True",
        "add out parameter should return same object"
    );
    Ok(())
}

#[test]
fn add_out_parameter_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
fnp_out = np.zeros(3)
np_out = np.zeros(3)
fnp.add(a, b, out=fnp_out)
np.add(a, b, out=np_out)
print(np.array_equal(fnp_out, np_out))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "add out parameter values should match numpy"
    );
    Ok(())
}

#[test]
fn multiply_out_parameter() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
fnp_out = np.zeros(3)
np_out = np.zeros(3)
fnp.multiply(a, b, out=fnp_out)
np.multiply(a, b, out=np_out)
print(np.array_equal(fnp_out, np_out))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "multiply out parameter should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Broadcasting edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn add_broadcast_0d_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array(5)  # 0-d scalar array
b = np.array([1, 2, 3])
fnp_result = fnp.add(a, b)
np_result = np.add(a, b)
print(fnp_result.shape == np_result.shape, np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True True"),
        "add broadcast 0-d + 1-d should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn add_broadcast_empty_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([]).reshape(0, 3)  # (0, 3)
b = np.array([1, 2, 3])  # (3,)
fnp_result = fnp.add(a, b)
np_result = np.add(a, b)
print(fnp_result.shape == np_result.shape, np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True True"),
        "add broadcast with empty array should match numpy: {result}"
    );
    Ok(())
}
