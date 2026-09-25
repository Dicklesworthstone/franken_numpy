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

/// Two float32 arrays with Python-float tolerances: numpy evaluates `|x - y|` and
/// `float32(atol) + float32(rtol) * |y|` in float32 (the tolerances are weak under NEP 50),
/// rounding at each ufunc. fnp's float32 isclose/allclose kernels widened to float64 and
/// disagreed at the tolerance boundary (2 of 200,000 near-boundary pairs at rtol=1e-3, seed 0).
/// The sweep places x within +-0.1% of the tolerance boundary of y.
#[test]
fn f32_array_pairs_use_numpys_float32_tolerance_arithmetic() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(0)
bad = []
for rtol, atol in ((1e-5, 1e-8), (1e-3, 0.0), (0.0, 1e-3), (1e-6, 1e-7)):
    y = rng.uniform(-10, 10, 200000).astype(np.float32)
    tol = atol + rtol * np.abs(y.astype(np.float64))
    x = (y.astype(np.float64) + tol * rng.choice([-1, 1], y.size) * rng.uniform(0.999, 1.001, y.size)).astype(np.float32)
    got, want = fnp.isclose(x, y, rtol=rtol, atol=atol), np.isclose(x, y, rtol=rtol, atol=atol)
    if got.dtype != want.dtype or not np.array_equal(got, want):
        bad.append(f"isclose rtol={rtol} atol={atol}: {int((got != want).sum())} mismatches")
    # allclose over slices that are all-close in numpy, so a disagreement cannot hide behind False
    close_idx = np.flatnonzero(want)[:5000]
    if bool(fnp.allclose(x[close_idx], y[close_idx], rtol=rtol, atol=atol)) != bool(np.allclose(x[close_idx], y[close_idx], rtol=rtol, atol=atol)):
        bad.append(f"allclose rtol={rtol} atol={atol}")
print(bad if bad else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "True",
        "float32 isclose/allclose tolerance arithmetic must match numpy: {result}"
    );
    Ok(())
}

/// A Python-int tolerance must reach numpy as an int. `parse_close_args` reads `atol=0` as
/// the f64 `0.0` for the native kernels, and every route that declined then called numpy with
/// that float: `0.0 + rtol * |td|` cannot add a float to a timedelta64, so
/// `isclose(td, td, atol=0)` raised where numpy returns True (numpy's own
/// TestIsclose::test_timedelta), and `rtol=0, atol=0` on Decimal objects failed in
/// `float * Decimal` before numpy's own isfinite TypeError. A decline now hands numpy the
/// caller's arguments. 5 of the 13 cases failed before the fix.
///
/// Controls: float operands with int tolerances, an int array against a scalar, and a
/// MaskedArray keep matching. Outcome = result type, dtype, shape and bytes, or exception
/// type and message.
#[test]
fn close_declines_hand_numpy_the_callers_int_tolerances() -> Result<(), String> {
    let script = fnp_script(
        r#"
from decimal import Decimal

def outcome(call):
    try:
        r = call()
    except Exception as ex:
        return (type(ex).__name__, str(ex))
    a = np.asarray(r)
    return (type(r).__name__, a.dtype.str, a.shape, a.tobytes())

td = np.array([[1, 2, 3, "NaT"]], dtype="m8[ns]")
dec = np.array([Decimal("1.5"), Decimal("2.25")], dtype=object)
f = np.array([1.0, 2.0, np.nan])
cases = {
    # A Python-int tolerance stays an int in numpy's arithmetic: `0 + rtol * |td|` is a
    # timedelta, where a float 0.0 cannot be added to one.
    "td atol=0": lambda m: m.isclose(td, td, atol=0, equal_nan=True),
    "td atol=0 all": lambda m: m.allclose(td, td, atol=0, equal_nan=True),
    "td atol=m8": lambda m: m.isclose(td, td, atol=np.timedelta64(1, "ns"), equal_nan=True),
    "td scalar rtol=0 atol=0": lambda m: m.isclose(np.timedelta64(1, "s"), np.timedelta64(2, "s"), atol=0, rtol=0),
    "td default tol": lambda m: m.isclose(td, td),
    # int * Decimal and int + Decimal work; float * Decimal raises.
    "decimal rtol=0 atol=0": lambda m: m.isclose(dec, dec, rtol=0, atol=0),
    "decimal rtol=0 atol=0 all": lambda m: m.allclose(dec, dec, rtol=0, atol=0),
    "decimal default tol": lambda m: m.isclose(dec, dec),
    # Float operands with int tolerances keep the native answer.
    "float atol=0": lambda m: m.isclose(f, f + 1e-9, atol=0),
    "float rtol=0 atol=1": lambda m: m.isclose(f, f + 0.5, rtol=0, atol=1, equal_nan=True),
    "float all atol=0": lambda m: m.allclose(f[:2], f[:2] * (1 + 1e-7), atol=0),
    "int arr scalar atol=0": lambda m: m.isclose(np.arange(4), 2, atol=0),
    "masked": lambda m: m.isclose(np.ma.array([1.0, 2.0], mask=[0, 1]), np.ma.array([1.0, 5.0]), atol=0),
}
bad = []
for name, case in cases.items():
    ours, theirs = outcome(lambda: case(fnp)), outcome(lambda: case(np))
    if ours != theirs:
        bad.append(f"{name}: fnp={str(ours)[:160]} numpy={str(theirs)[:160]}")
print(len(cases), bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "13 []",
        "isclose/allclose declines must hand numpy the caller's tolerances: {result}"
    );
    Ok(())
}
