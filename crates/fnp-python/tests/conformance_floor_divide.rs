//! Conformance tests for numpy.floor_divide against NumPy oracle.
//!
//! Tests floor_divide (element-wise floor division).

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
fn floor_divide_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([7, 8, 9, 10])
x2 = np.array([3, 3, 3, 3])
result = fnp.floor_divide(x1, x2)
expected = np.floor_divide(x1, x2)
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
fn floor_divide_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([7.5, 8.5, 9.5])
x2 = np.array([2.5, 2.5, 2.5])
result = fnp.floor_divide(x1, x2)
expected = np.floor_divide(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "floor_divide float should match numpy"
    );
    Ok(())
}

#[test]
fn floor_divide_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([-7, -8, 7, 8])
x2 = np.array([3, 3, -3, -3])
result = fnp.floor_divide(x1, x2)
expected = np.floor_divide(x1, x2)
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

#[test]
fn floor_divide_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(7.0)
x2 = np.float64(3.0)
fnp_result = fnp.floor_divide(x1, x2)
np_result = np.floor_divide(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "floor_divide scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn floor_divide_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([np.inf, -np.inf, 1.0, np.nan])
x2 = np.array([2.0, 2.0, np.inf, 1.0])
fnp_result = fnp.floor_divide(x1, x2)
np_result = np.floor_divide(x1, x2)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "floor_divide special values should match numpy"
    );
    Ok(())
}

#[test]
fn floor_divide_by_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
x1 = np.array([1.0, -1.0, 0.0])
x2 = np.array([0.0, 0.0, 0.0])
fnp_result = fnp.floor_divide(x1, x2)
np_result = np.floor_divide(x1, x2)
# Check inf/nan results match
print(np.allclose(fnp_result, np_result, equal_nan=True) or
      all((np.isinf(f) == np.isinf(n) and np.isnan(f) == np.isnan(n))
          for f, n in zip(fnp_result.flat, np_result.flat)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "floor_divide by zero should match numpy"
    );
    Ok(())
}

#[test]
fn floor_divide_broadcasting() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([[10.0, 20.0], [30.0, 40.0]])
x2 = np.array([3.0, 4.0])
fnp_result = fnp.floor_divide(x1, x2)
np_result = np.floor_divide(x1, x2)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "floor_divide broadcasting should match numpy"
    );
    Ok(())
}

#[test]
fn floor_divide_signed_zero_parity() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
# floor_divide signed-zero: 0 // x preserves sign rules
# 0 // positive = 0, 0 // negative = -0
# -0 // positive = -0, -0 // negative = 0
tests = [
    (0.0, 1.0), (0.0, -1.0),
    (-0.0, 1.0), (-0.0, -1.0),
]
all_pass = True
for x1, x2 in tests:
    fnp_result = fnp.floor_divide(np.float64(x1), np.float64(x2))
    np_result = np.floor_divide(np.float64(x1), np.float64(x2))
    fnp_sign = np.signbit(fnp_result)
    np_sign = np.signbit(np_result)
    if fnp_sign != np_sign:
        print(f"FAIL: floor_divide({x1}, {x2})")
        print(f"  fnp result={fnp_result} signbit={fnp_sign}")
        print(f"  np result={np_result} signbit={np_sign}")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "floor_divide signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

/// The native float64 / float32 floor_divide route at 2^15 (numpy's call), 2^16 + 37 (its call
/// floor, pooled and ragged) and 2^20 + 3. Each runs plain and with one set in the LAST chunk -
/// zero divisors, infinite and NaN operands, an overflowing quotient, near-exact multiples
/// (where `floor(a / b)` overshoots numpy by one), signed zeros and subnormals - under
/// errstate(all=) warn / raise / ignore. Bytes, dtype, shape and every warning are compared.
#[test]
fn floor_divide_float_route_matches_numpy_bytes_and_events_at_every_floor() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
def outcome(m, a, b, mode):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with np.errstate(all=mode):
                r = m.floor_divide(a, b)
            got = (r.dtype.str, r.shape, r.tobytes())
        except Exception as exc:
            got = ("raise", type(exc).__name__, str(exc))
    return got, sorted(str(w.message) for w in caught)
rng = np.random.default_rng(37)
cells, bad = 0, []
for dt in (np.float64, np.float32):
    info = np.finfo(dt)
    near = [(dt(k) * dt(b), dt(b)) for k in (5, 7, 9) for b in (0.1, 0.3, -0.7)]
    specials = {
        "zero divisor": [(1.0, 0.0), (-1.0, 0.0), (0.0, 0.0), (2.0, -0.0)],
        "nonfinite": [(np.inf, 3.0), (3.0, np.inf), (-3.0, np.inf), (np.nan, 2.0), (2.0, np.nan)],
        "overflow": [(info.max, 0.5), (-info.max, 0.25)],
        "near multiples": near,
        "signs subnormal": [(-0.0, 3.0), (0.0, -3.0), (info.smallest_subnormal, 1.0), (-info.smallest_subnormal, 3.0)],
    }
    for n in (1 << 15, (1 << 16) + 37, (1 << 20) + 3):
        a0 = (rng.standard_normal(n) * 50).astype(dt)
        b0 = rng.uniform(0.1, 7.0, n).astype(dt) * rng.choice([-1, 1], n).astype(dt)
        for label, pairs in {"plain": [], **specials}.items():
            a, b = a0.copy(), b0.copy()
            if pairs:
                a[-len(pairs):] = [p[0] for p in pairs]
                b[-len(pairs):] = [p[1] for p in pairs]
            for mode in ("warn", "raise", "ignore"):
                cells += 1
                if outcome(fnp, a, b, mode) != outcome(np, a, b, mode):
                    bad.append(f"{np.dtype(dt).name} n={n} {label} {mode}")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (cells, bad) = result.trim().split_once(' ').unwrap_or(("0", &result));
    assert_eq!(cells, "108", "cell table drifted: {result}");
    assert_eq!(
        bad, "[]",
        "floor_divide must match numpy's bytes and events: {result}"
    );
    Ok(())
}
