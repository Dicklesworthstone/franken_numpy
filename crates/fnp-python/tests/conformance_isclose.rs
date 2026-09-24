//! Conformance tests for numpy isclose against NumPy oracle.

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

mod support;
use support::fnp_script;

// ─────────────────────────────────────────────────────────────────────────────
// isclose basic
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn isclose_exact_equal() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 2.0, 3.0])
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose exact equal should match numpy"
    );
    Ok(())
}

#[test]
fn isclose_within_tolerance() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0 + 1e-9, 2.0 + 1e-9, 3.0 + 1e-9])
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose within tolerance should match numpy"
    );
    Ok(())
}

#[test]
fn isclose_outside_tolerance() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.1, 2.1, 3.1])
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose outside tolerance should match numpy"
    );
    Ok(())
}

#[test]
fn isclose_custom_rtol() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.01, 2.02, 3.03])
result = fnp.isclose(a, b, rtol=0.02)
expected = np.isclose(a, b, rtol=0.02)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose custom rtol should match numpy"
    );
    Ok(())
}

#[test]
fn isclose_custom_atol() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0.0, 0.0, 0.0])
b = np.array([1e-9, 1e-7, 1e-5])
result = fnp.isclose(a, b, atol=1e-6)
expected = np.isclose(a, b, atol=1e-6)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose custom atol should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// isclose NaN handling
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn isclose_nan_default() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0])
b = np.array([1.0, np.nan, 3.0])
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose nan default should match numpy"
    );
    Ok(())
}

#[test]
fn isclose_nan_equal_nan_true() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0])
b = np.array([1.0, np.nan, 3.0])
result = fnp.isclose(a, b, equal_nan=True)
expected = np.isclose(a, b, equal_nan=True)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose equal_nan=True should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// isclose infinity handling
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn isclose_inf_equal() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([np.inf, -np.inf, 1.0])
b = np.array([np.inf, -np.inf, 1.0])
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose inf equal should match numpy"
    );
    Ok(())
}

#[test]
fn isclose_inf_not_equal() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([np.inf, -np.inf])
b = np.array([-np.inf, np.inf])
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose inf not equal should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// isclose broadcasting
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn isclose_broadcast_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array(1.0)
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose broadcast scalar should match numpy"
    );
    Ok(())
}

#[test]
fn isclose_broadcast_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([1.0, 2.0])
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose broadcast 2d should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// isclose dtype handling
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn isclose_int_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype='int32')
b = np.array([1, 2, 3], dtype='int32')
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose int arrays should match numpy"
    );
    Ok(())
}

#[test]
fn isclose_mixed_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype='int32')
b = np.array([1.0, 2.0, 3.0], dtype='float64')
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose mixed dtype should match numpy"
    );
    Ok(())
}

#[test]
fn isclose_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(1.0)
y = np.float64(1.0)
fnp_result = fnp.isclose(x, y)
np_result = np.isclose(x, y)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "isclose scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn isclose_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j, 3+2j], dtype=np.complex128)
b = np.array([1+1j, 2-1j, 3+2j], dtype=np.complex128)
fnp_result = fnp.isclose(a, b)
np_result = np.isclose(a, b)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isclose complex should match numpy");
    Ok(())
}

/// isclose/allclose must follow numpy's NEP 50 scalar rules and its non-float inputs:
/// - a Python float/int `b` is WEAK: against a float32 array numpy computes `|x - b|` and the
///   tolerance comparison in float32, so `isclose(f32_array, 0.1, rtol=0, atol=0)` matches the
///   element equal to float32(0.1); fnp computed in float64 and answered all-False;
/// - a float32/float16 scalar `a` with a Python float `b` likewise stays in float32/float16;
/// - two MaskedArrays give a MaskedArray; a timedelta64 `atol` stays a timedelta; a negative
///   tolerance brings numpy's `| (x == y)` term (numpy's own TestIsclose).
///
/// Controls: float64 arrays and a numpy float64 scalar (strong) keep matching. Outcome =
/// result type, dtype, values and mask, or exception type.
#[test]
fn isclose_follows_nep50_scalars_and_numpy_input_kinds() -> Result<(), String> {
    let script = fnp_script(
        r#"
def outcome(fn):
    try:
        r = fn()
        mask = np.ma.getmaskarray(r).tolist() if isinstance(r, np.ma.MaskedArray) else None
        return ("ok", type(r).__name__, str(getattr(r, "dtype", "")), np.asarray(r).tolist(), mask)
    except Exception as exc:
        return ("err", type(exc).__name__)
a32 = np.array([1.0, 0.1, 3.0], np.float32)
a64 = np.array([1.0, 0.1, 3.0])
td = np.array([1, 2], dtype="m8[ns]")
cases = [
    lambda m: m.isclose(a32, 0.1, rtol=0, atol=0),
    lambda m: m.isclose(a32, 1.0 + 1e-8, rtol=0, atol=0),
    lambda m: m.isclose(a32, 3, rtol=0, atol=0),
    lambda m: m.isclose(a32, 0.1),
    lambda m: m.isclose(a32, np.float64(0.1), rtol=0, atol=0),
    lambda m: m.isclose(np.float32(0.1), 0.1, rtol=0, atol=0),
    lambda m: m.isclose(np.array([0.1], np.float16), 0.1, rtol=0, atol=0),
    lambda m: m.allclose(np.float32(0.1), 0.1, rtol=0, atol=0),
    lambda m: m.isclose(a64, 0.1, rtol=0, atol=0),
    lambda m: m.isclose(a64, np.float32(0.1), rtol=1e-9, atol=0),
    lambda m: m.isclose(np.ma.array([1.0, 2.0, 3.0], mask=[0, 1, 0]), np.ma.array([1.0, 5.0, 3.1], mask=[0, 0, 1])),
    lambda m: m.isclose(td, np.array([1, 3], dtype="m8[ns]"), atol=np.timedelta64(1, "ns")),
    lambda m: m.isclose(1.0, 1.0, rtol=-1),
    lambda m: m.isclose([1.0, 2.0], [1.1, 2.0], rtol=[0.2, 0.0]),
]
bad = [i for i, c in enumerate(cases) if outcome(lambda: c(fnp)) != outcome(lambda: c(np))]
print(bad if bad else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "True",
        "isclose NEP 50 / input-kind surface must match numpy: {result}"
    );
    Ok(())
}
