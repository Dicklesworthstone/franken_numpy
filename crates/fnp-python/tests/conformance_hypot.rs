//! Conformance tests for numpy.hypot against NumPy oracle.
//!
//! Tests hypot (hypotenuse calculation).

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

#[test]
fn hypot_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([3.0, 5.0, 8.0])
x2 = np.array([4.0, 12.0, 15.0])
result = fnp.hypot(x1, x2)
expected = np.hypot(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hypot basic should match numpy");
    Ok(())
}

#[test]
fn hypot_broadcasting() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([[1, 2], [3, 4]])
x2 = np.array([5, 6])
result = fnp.hypot(x1, x2)
expected = np.hypot(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "hypot broadcasting should match numpy"
    );
    Ok(())
}

#[test]
fn hypot_with_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([np.inf, -np.inf, 1.0])
x2 = np.array([1.0, 1.0, np.inf])
result = fnp.hypot(x1, x2)
expected = np.hypot(x1, x2)
print(np.array_equal(result, expected) or all(np.isinf(result) == np.isinf(expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "hypot with infinity should match numpy"
    );
    Ok(())
}

#[test]
fn hypot_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(3.0)
x2 = np.float64(4.0)
fnp_result = fnp.hypot(x1, x2)
np_result = np.hypot(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "hypot scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn hypot_with_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([np.nan, 1.0, np.nan])
x2 = np.array([1.0, np.nan, np.nan])
result = fnp.hypot(x1, x2)
expected = np.hypot(x1, x2)
print(np.allclose(result, expected, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hypot with nan should match numpy");
    Ok(())
}

#[test]
fn hypot_with_zeros() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([0.0, 0.0, 3.0, -0.0])
x2 = np.array([0.0, 4.0, 0.0, -0.0])
result = fnp.hypot(x1, x2)
expected = np.hypot(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hypot with zeros should match numpy");
    Ok(())
}

#[test]
fn hypot_negative_inputs() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([-3.0, -5.0, 3.0])
x2 = np.array([4.0, -12.0, -4.0])
result = fnp.hypot(x1, x2)
expected = np.hypot(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "hypot negative inputs should match numpy"
    );
    Ok(())
}

#[test]
fn hypot_large_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1e154, 1e200, 1e-154])
x2 = np.array([1e154, 1e200, 1e-154])
result = fnp.hypot(x1, x2)
expected = np.hypot(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "hypot large values should match numpy"
    );
    Ok(())
}

#[test]
fn hypot_signed_zero_parity() -> Result<(), String> {
    let script = fnp_script(
        r#"
# hypot signed-zero parity
# hypot(0, 0) = 0.0 (always positive, per IEEE 754)
# hypot(-0, -0) = 0.0 (always positive)
tests = [
    (0.0, 0.0),
    (-0.0, 0.0),
    (0.0, -0.0),
    (-0.0, -0.0),
]
all_pass = True
for x1, x2 in tests:
    fnp_result = fnp.hypot(np.float64(x1), np.float64(x2))
    np_result = np.hypot(np.float64(x1), np.float64(x2))
    fnp_sign = np.signbit(fnp_result)
    np_sign = np.signbit(np_result)
    if fnp_sign != np_sign:
        print(f"FAIL: hypot({x1}, {x2})")
        print(f"  fnp result={fnp_result} signbit={fnp_sign}")
        print(f"  np result={np_result} signbit={np_sign}")
        all_pass = False
    if fnp_result != np_result:
        print(f"FAIL: hypot({x1}, {x2}) value mismatch: fnp={fnp_result} np={np_result}")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "hypot signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

/// The native float32 route (from 2^16 elements) calls numpy's own `hypotf`, so every cell must
/// match numpy's bytes AND its events under each errstate: overflow and underflow come off the
/// FE status word and are replayed on a numpy witness pair, a signaling NaN ("invalid", no
/// witness) defers, quiet NaNs keep their payloads. A spy counting `np.hypot`'s ARRAY calls
/// proves the route answers the plain 2^16 + 37 and 2^20 + 3 cells itself and leaves 2^15 to
/// numpy, so the byte comparison is not passing by delegating everything.
#[test]
fn hypot_float32_route_matches_numpy_bytes_and_events() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
def outcome(f, a, b, mode):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with np.errstate(all=mode):
                r = f(a, b)
            got = (r.dtype.str, r.shape, r.tobytes())
        except Exception as exc:
            got = ("raise", type(exc).__name__, str(exc))
    return got, sorted(str(w.message) for w in caught)
def delegations(a, b):
    real, calls = np.hypot, []
    def spy(*args, **kwargs):
        calls.append(isinstance(args[0], np.ndarray))
        return real(*args, **kwargs)
    np.hypot = spy
    try:
        fnp.hypot(a, b)
    finally:
        np.hypot = real
    return sum(calls)
def bits(x):
    return x if isinstance(x, int) else int(np.array([x], np.float32).view(np.uint32)[0])
info = np.finfo(np.float32)
specials = {
    "overflow": [(info.max, info.max), (-3e38, 3e38)],
    "underflow": [(info.smallest_subnormal, info.smallest_subnormal), (1e-30, -1e-30)],
    "nonfinite": [(np.inf, np.nan), (np.nan, -np.inf), (-np.inf, 2.0), (3.0, np.inf)],
    "nan payloads": [(0x7fc00001, 0xffc00002), (0xffc00002, 1.0), (2.0, 0x7fc00001)],
    "signaling nan": [(0x7fa00000, 1.0)],
    "signed zeros": [(-0.0, -0.0), (0.0, -3.0), (-0.0, 0.0)],
}
rng = np.random.default_rng(41)
cells, bad = 0, []
for n in (1 << 15, (1 << 16) + 37, (1 << 20) + 3):
    a0 = (rng.standard_normal(n) * 1e3).astype(np.float32)
    b0 = (rng.standard_normal(n) * 1e3).astype(np.float32)
    for label, pairs in {"plain": [], **specials}.items():
        a, b = a0.copy(), b0.copy()
        if pairs:
            a.view(np.uint32)[-len(pairs):] = [bits(p[0]) for p in pairs]
            b.view(np.uint32)[-len(pairs):] = [bits(p[1]) for p in pairs]
        for mode in ("warn", "raise", "ignore"):
            cells += 1
            if outcome(fnp.hypot, a, b, mode) != outcome(np.hypot, a, b, mode):
                bad.append(f"n={n} {label} {mode}")
    expected = 1 if n < 1 << 16 else 0
    if delegations(a0, b0) != expected:
        bad.append(f"n={n} delegations != {expected}")
a = (rng.standard_normal(1 << 17) * 1e3).astype(np.float32)
b = (rng.standard_normal(1 << 17) * 1e3).astype(np.float32)
layouts = {
    "2-D": (a.reshape(256, 512), b.reshape(256, 512)),
    "broadcast row": (a.reshape(256, 512), b[:512]),
    "strided": (a[::2], b[::2]),
    "mixed float64": (a, b.astype(np.float64)),
    "big-endian": (a.astype(">f4"), b.astype(">f4")),
    "fortran": (np.asfortranarray(a.reshape(256, 512)), np.asfortranarray(b.reshape(256, 512))),
}
for label, (x, y) in layouts.items():
    for mode in ("warn", "raise", "ignore"):
        cells += 1
        if outcome(fnp.hypot, x, y, mode) != outcome(np.hypot, x, y, mode):
            bad.append(f"{label} {mode}")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (cells, bad) = result.trim().split_once(' ').unwrap_or(("0", &result));
    assert_eq!(cells, "81", "cell table drifted: {result}");
    assert_eq!(
        bad, "[]",
        "float32 hypot must match numpy's bytes and events: {result}"
    );
    Ok(())
}
