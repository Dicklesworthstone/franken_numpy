//! Conformance tests for numpy.logaddexp and numpy.logaddexp2 against NumPy oracle.
//!
//! Tests logaddexp (log(exp(x1) + exp(x2))) and logaddexp2 (log2(2^x1 + 2^x2)).

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
fn logaddexp_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([0.0, 1.0, 2.0])
x2 = np.array([1.0, 2.0, 3.0])
result = fnp.logaddexp(x1, x2)
expected = np.logaddexp(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logaddexp basic should match numpy");
    Ok(())
}

#[test]
fn logaddexp_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([-1.0, -2.0, -3.0])
x2 = np.array([-0.5, -1.0, -2.0])
result = fnp.logaddexp(x1, x2)
expected = np.logaddexp(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "logaddexp with negative values should match numpy"
    );
    Ok(())
}

#[test]
fn logaddexp_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([0.0, -np.inf])
x2 = np.array([np.inf, 0.0])
result = fnp.logaddexp(x1, x2)
expected = np.logaddexp(x1, x2)
print(np.array_equal(result, expected) or all(np.isinf(result) == np.isinf(expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "logaddexp with infinity should match numpy"
    );
    Ok(())
}

#[test]
fn logaddexp2_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([0.0, 1.0, 2.0])
x2 = np.array([1.0, 2.0, 3.0])
result = fnp.logaddexp2(x1, x2)
expected = np.logaddexp2(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logaddexp2 basic should match numpy");
    Ok(())
}

#[test]
fn logaddexp_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(1.0)
x2 = np.float64(2.0)
fnp_result = fnp.logaddexp(x1, x2)
np_result = np.logaddexp(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "logaddexp scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn logaddexp2_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(1.0)
x2 = np.float64(2.0)
fnp_result = fnp.logaddexp2(x1, x2)
np_result = np.logaddexp2(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "logaddexp2 scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn logaddexp_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([np.nan, 1.0, np.nan])
x2 = np.array([1.0, np.nan, np.nan])
result = fnp.logaddexp(x1, x2)
expected = np.logaddexp(x1, x2)
print(np.allclose(result, expected, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logaddexp nan should match numpy");
    Ok(())
}

#[test]
fn logaddexp_neg_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([-np.inf, -np.inf, 0.0])
x2 = np.array([-np.inf, 0.0, -np.inf])
result = fnp.logaddexp(x1, x2)
expected = np.logaddexp(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "logaddexp neg inf should match numpy"
    );
    Ok(())
}

#[test]
fn logaddexp2_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([np.inf, -np.inf, np.nan, 0.0])
x2 = np.array([0.0, 0.0, 0.0, np.inf])
result = fnp.logaddexp2(x1, x2)
expected = np.logaddexp2(x1, x2)
print(np.allclose(result, expected, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "logaddexp2 special values should match numpy"
    );
    Ok(())
}

#[test]
fn logaddexp_broadcasting() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([[1.0, 2.0], [3.0, 4.0]])
x2 = np.array([0.0, 1.0])
result = fnp.logaddexp(x1, x2)
expected = np.logaddexp(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "logaddexp broadcasting should match numpy"
    );
    Ok(())
}

#[test]
fn logaddexp_large_difference() -> Result<(), String> {
    let script = fnp_script(
        r#"
# When one value dominates, result should be close to max
x1 = np.array([1000.0, -1000.0])
x2 = np.array([1.0, 1.0])
result = fnp.logaddexp(x1, x2)
expected = np.logaddexp(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "logaddexp large difference should match numpy"
    );
    Ok(())
}

/// The native float32 routes (from 2^16 elements) call numpy's own `expf` / `exp2f` and
/// `log1pf`, so every cell must match numpy's bytes AND its events under each errstate. numpy's
/// loop raises "invalid" for a quiet NaN operand (a signaling compare) where Rust's compare does
/// not: a route that trusted the FE status word alone answers those cells silently. Underflow
/// past a gap of ~87 (logaddexp) or ~126 (logaddexp2) and the `MAX - (-MAX)` overflow come off
/// the status word and are replayed on a numpy witness pair. A spy counting numpy's ARRAY calls
/// (a witness passes two floats) proves the route answers the plain 2^16 + 37 and 2^20 + 3
/// cells itself, which underflow: sigma-20 operands put ~0.2% of their gaps past 87.
#[test]
fn logaddexp_float32_routes_match_numpy_bytes_and_events() -> Result<(), String> {
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
def delegations(name, a, b):
    real, calls = getattr(np, name), []
    def spy(*args):
        calls.append(isinstance(args[0], np.ndarray))
        return real(*args)
    setattr(np, name, spy)
    try:
        getattr(fnp, name)(a, b)
    finally:
        setattr(np, name, real)
    return sum(calls)
def bits(x):
    return x if isinstance(x, int) else int(np.array([x], np.float32).view(np.uint32)[0])
info = np.finfo(np.float32)
inf = np.inf
specials = {
    "equal infinities": [(inf, inf), (-inf, -inf), (inf, -inf), (-inf, inf), (-inf, 3.0)],
    "quiet nan": [(0x7fc00001, 1.0), (1.0, 0xffc00002), (0x7fc00001, 0xffc00002)],
    "signaling nan": [(0x7fa00000, 1.0)],
    "underflow": [(0.0, -140.0), (-200.0, 0.0)],
    "wide gap": [(50.0, 0.0), (0.0, -60.0)],
    "extremes": [(info.max, info.max), (info.max, -info.max), (-0.0, 0.0)],
}
rng = np.random.default_rng(43)
cells, bad = 0, []
for name in ("logaddexp", "logaddexp2"):
    for n in (1 << 15, (1 << 16) + 37, (1 << 20) + 3):
        a0 = (rng.standard_normal(n) * 20).astype(np.float32)
        b0 = (rng.standard_normal(n) * 20).astype(np.float32)
        for label, pairs in {"plain": [], **specials}.items():
            a, b = a0.copy(), b0.copy()
            if pairs:
                a.view(np.uint32)[-len(pairs):] = [bits(p[0]) for p in pairs]
                b.view(np.uint32)[-len(pairs):] = [bits(p[1]) for p in pairs]
            for mode in ("warn", "raise", "ignore"):
                cells += 1
                got = outcome(getattr(fnp, name), a, b, mode)
                if got != outcome(getattr(np, name), a, b, mode):
                    bad.append(f"{name} n={n} {label} {mode}")
        expected = 1 if n < 1 << 16 else 0
        if delegations(name, a0, b0) != expected:
            bad.append(f"{name} n={n} delegations != {expected}")
    a = (rng.standard_normal(1 << 17) * 20).astype(np.float32)
    b = (rng.standard_normal(1 << 17) * 20).astype(np.float32)
    layouts = {
        "2-D": (a.reshape(256, 512), b.reshape(256, 512)),
        "broadcast row": (a.reshape(256, 512), b[:512]),
        "strided": (a[::2], b[::2]),
        "mixed float64": (a, b.astype(np.float64)),
        "big-endian": (a.astype(">f4"), b.astype(">f4")),
    }
    for label, (x, y) in layouts.items():
        for mode in ("warn", "raise", "ignore"):
            cells += 1
            got = outcome(getattr(fnp, name), x, y, mode)
            if got != outcome(getattr(np, name), x, y, mode):
                bad.append(f"{name} {label} {mode}")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (cells, bad) = result.trim().split_once(' ').unwrap_or(("0", &result));
    assert_eq!(cells, "156", "cell table drifted: {result}");
    assert_eq!(
        bad, "[]",
        "float32 logaddexp / logaddexp2 must match numpy's bytes and events: {result}"
    );
    Ok(())
}
