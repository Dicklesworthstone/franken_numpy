//! Conformance tests for numpy.divmod against NumPy oracle.
//!
//! Tests the native Rust divmod implementation against NumPy.

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
// divmod
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn divmod_float64_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([7.0, 8.0, 9.0, 10.0])
y = np.array([3.0, 3.0, 3.0, 3.0])
q, r = fnp.divmod(x, y)
qe, re = np.divmod(x, y)
print(np.allclose(q, qe) and np.allclose(r, re))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "divmod float64 basic should match numpy"
    );
    Ok(())
}

/// Near an exact multiple `a / b` rounds up, so a `floor(a / b)` quotient overshoots numpy's
/// fmod-based one by 1 (numpy gh-6127; `test_float_remainder_roundoff`). The old native kernels
/// did exactly that in up to 39% of cells, which `allclose` above cannot see. Byte equality
/// across every call form, plus numpy's FP-error surface: `divmod(inf, inf)` is "invalid" and
/// `divmod(4, tiny)` "overflow".
#[test]
fn divmod_float64_is_byte_exact_near_multiples_and_raises_like_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
rng = np.random.default_rng(1)
k = rng.integers(1, 200, 4000).astype(np.float64)
b = rng.choice([6e-8, -6e-8, 0.1, -0.1, 3.3, 1e-300, 7.0], 4000)
a = k * b
forms = [(a, b), (np.array(a[0]), np.array(b[0])), (a, np.array(6e-8)),
         (a[:60].reshape(-1, 1), b[:50].reshape(1, -1)), (float(a[3]), float(b[3])),
         (a, 6e-8), (a[::2], b[::2]), (np.tile(a, 80), np.tile(b, 80))]
bad = []
naive_wrong = int(np.sum(np.floor(a / b) != np.divmod(a, b)[0]))
for i, (x, y) in enumerate(forms):
    r, e = fnp.divmod(x, y), np.divmod(x, y)
    for p, q in zip(r, e):
        if type(p) is not type(q) or np.shape(p) != np.shape(q) or np.asarray(p).tobytes() != np.asarray(q).tobytes():
            bad.append(("form", i))
def outcome(mod, x, y, **err):
    with np.errstate(**err), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            r = mod.divmod(x, y)
        except FloatingPointError as ex:
            return ("raise", str(ex))
        return ("ok", [np.asarray(v).tobytes() for v in r], sorted(str(m.message) for m in w))
tiny = np.finfo(np.float64).tiny
cases = [((np.array(np.inf), np.array(np.inf)), dict(invalid="raise")),
         ((np.array([np.inf, 1.0]), np.array([np.inf, 2.0])), dict(invalid="raise")),
         ((np.array(4.0), np.array(tiny)), dict(over="raise", invalid="ignore")),
         ((np.full(300_000, 4.0), np.full(300_000, tiny)), dict(over="raise", invalid="ignore")),
         ((np.full(8, 4.0), np.full(8, tiny)), dict(all="warn")),
         ((np.array(1.0), np.array(0.0)), dict(divide="raise", invalid="ignore")),
         ((np.array(np.inf), np.array(0.0)), dict(divide="raise", invalid="ignore"))]
for i, (args, err) in enumerate(cases):
    if outcome(fnp, *args, **err) != outcome(np, *args, **err):
        bad.append(("errstate", i))
print(naive_wrong, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (naive_wrong, bad) = result.trim().split_once(' ').unwrap_or(("0", &result));
    // Negative control: the inputs must sit in the regime where floor(a / b) is wrong.
    assert!(
        naive_wrong.parse::<usize>().unwrap_or(0) >= 100,
        "inputs no longer exercise the roundoff regime: {result}"
    );
    assert_eq!(bad, "[]", "divmod differs from numpy: {result}");
    Ok(())
}

#[test]
fn divmod_negative_dividend() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-7.0, -8.0, -9.0, -10.0])
y = np.array([3.0, 3.0, 3.0, 3.0])
q, r = fnp.divmod(x, y)
qe, re = np.divmod(x, y)
print(np.allclose(q, qe) and np.allclose(r, re))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "divmod negative dividend should match numpy"
    );
    Ok(())
}

#[test]
fn divmod_negative_divisor() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([7.0, 8.0, 9.0, 10.0])
y = np.array([-3.0, -3.0, -3.0, -3.0])
q, r = fnp.divmod(x, y)
qe, re = np.divmod(x, y)
print(np.allclose(q, qe) and np.allclose(r, re))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "divmod negative divisor should match numpy"
    );
    Ok(())
}

#[test]
fn divmod_both_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-7.0, -8.0, -9.0, -10.0])
y = np.array([-3.0, -3.0, -3.0, -3.0])
q, r = fnp.divmod(x, y)
qe, re = np.divmod(x, y)
print(np.allclose(q, qe) and np.allclose(r, re))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "divmod both negative should match numpy"
    );
    Ok(())
}

#[test]
fn divmod_int64() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([7, 8, 9, 10], dtype=np.int64)
y = np.array([3, 3, 3, 3], dtype=np.int64)
q, r = fnp.divmod(x, y)
qe, re = np.divmod(x, y)
print(np.array_equal(q, qe) and np.array_equal(r, re))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "divmod int64 should match numpy");
    Ok(())
}

#[test]
fn divmod_broadcast() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([[7.0, 8.0], [9.0, 10.0]])
y = np.array([3.0, 4.0])
q, r = fnp.divmod(x, y)
qe, re = np.divmod(x, y)
print(np.allclose(q, qe) and np.allclose(r, re))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "divmod broadcast should match numpy");
    Ok(())
}

#[test]
fn divmod_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
x = np.array([1.0, 0.0, np.inf, np.nan])
y = np.array([0.0, 0.0, 1.0, 1.0])
q, r = fnp.divmod(x, y)
qe, re = np.divmod(x, y)
# Check inf/nan handling element by element
def check_special(a, b):
    if np.isnan(a) and np.isnan(b):
        return True
    if np.isinf(a) and np.isinf(b):
        return np.sign(a) == np.sign(b)
    return np.allclose([a], [b])
q_match = all(check_special(q[i], qe[i]) for i in range(len(q)))
r_match = all(check_special(r[i], re[i]) for i in range(len(r)))
print(q_match and r_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "divmod special values should match numpy"
    );
    Ok(())
}

#[test]
fn divmod_exact_division() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([6.0, 9.0, 12.0, 15.0])
y = np.array([3.0, 3.0, 3.0, 3.0])
q, r = fnp.divmod(x, y)
qe, re = np.divmod(x, y)
# Remainder should be 0 for exact division
print(np.allclose(q, qe) and np.allclose(r, re) and np.allclose(r, 0.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "divmod exact division should match numpy"
    );
    Ok(())
}

#[test]
fn divmod_signed_zero_signbits() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
x = np.array([6.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0])
y = np.array([-3.0, -3.0, 3.0, -3.0, -np.inf, np.inf, -np.inf])
q, r = fnp.divmod(x, y)
qe, re = np.divmod(x, y)
value_match = (
    np.array_equal(q, qe, equal_nan=True) and
    np.array_equal(r, re, equal_nan=True)
)
sign_match = (
    np.array_equal(np.signbit(q), np.signbit(qe)) and
    np.array_equal(np.signbit(r), np.signbit(re))
)
print(value_match and sign_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "divmod signed zero signbits should match numpy"
    );
    Ok(())
}

#[test]
fn divmod_negative_inf_divisor() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
x = np.array([1.0, -1.0, 0.0])
y = np.array([-np.inf, -np.inf, -np.inf])
q, r = fnp.divmod(x, y)
qe, re = np.divmod(x, y)
def check_special(a, b):
    if np.isnan(a) and np.isnan(b):
        return True
    if np.isinf(a) and np.isinf(b):
        return np.sign(a) == np.sign(b)
    return np.allclose([a], [b])
q_match = all(check_special(q[i], qe[i]) for i in range(len(q)))
r_match = all(check_special(r[i], re[i]) for i in range(len(r)))
print(q_match and r_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "divmod negative inf divisor should match numpy"
    );
    Ok(())
}

#[test]
fn divmod_large_numbers() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1e300, 1e200, 1e100])
y = np.array([1e100, 1e50, 1e10])
q, r = fnp.divmod(x, y)
qe, re = np.divmod(x, y)
# Use relative tolerance for large numbers
print(np.allclose(q, qe, rtol=1e-10) and np.allclose(r, re, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "divmod large numbers should match numpy"
    );
    Ok(())
}

#[test]
fn divmod_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = "import numpy as np; x = np.float64(7.0); y = np.float64(3.0); q, r = np.divmod(x, y); print(type(q).__name__, type(r).__name__, q, r)";
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script("x = np.float64(7.0); y = np.float64(3.0); q, r = fnp.divmod(x, y); print(type(q).__name__, type(r).__name__, q, r)".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "divmod scalar return type mismatch\nnumpy: {numpy_result}\nfnp: {rust_result}"
    );

    Ok(())
}
