//! Conformance tests for numpy percentile, quantile, median, ptp against NumPy oracle.
//!
//! Tests percentile, quantile, median, ptp.

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

fn indent_python(body: &str) -> String {
    body.lines().map(|line| format!("    {line}\n")).collect()
}

fn outcome_body(body: &str) -> String {
    let indented = indent_python(body);
    r#"import json

def normalize(value):
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [normalize(item) for item in value]}
    if isinstance(value, np.ndarray):
        return {
            "kind": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if np.isscalar(value):
        scalar_type = type(value).__name__
        scalar_dtype = str(value.dtype) if hasattr(value, "dtype") else None
        scalar_value = value.item() if hasattr(value, "item") else value
        return {
            "kind": "scalar",
            "type": scalar_type,
            "dtype": scalar_dtype,
            "value": scalar_value,
        }
    return {"kind": "object", "type": type(value).__name__, "repr": repr(value)}

try:
__BODY__    payload = {"status": "ok", "result": normalize(result)}
    if "out" in locals():
        payload["out"] = normalize(out)
        payload["result_is_out"] = result is out
    print(json.dumps(payload, sort_keys=True, default=str))
except Exception as exc:
    message = str(exc).splitlines()[0] if str(exc) else ""
    print(json.dumps(
        {"status": "err", "type": type(exc).__name__, "message": message},
        sort_keys=True,
        default=str,
    ))
"#
    .replace("__BODY__", &indented)
}

fn numpy_outcome_script(body: &str) -> String {
    format!(
        "import numpy as np\n\
         MODULE = np\n\
         {}",
        outcome_body(body)
    )
}

fn fnp_outcome_script(body: &str) -> String {
    fnp_script(format!("MODULE = fnp\n{}", outcome_body(body)))
}

#[test]
fn percentile_quantile_median_keyword_outcomes_match_numpy() -> Result<(), String> {
    let cases = [
        (
            "percentile list scalar q",
            "result = MODULE.percentile([1, 2, 3, 4], 50)",
        ),
        (
            "percentile q sequence method keepdims",
            "result = MODULE.percentile(
    np.array([[1, 2, 3], [4, 5, 6]]),
    [25, 75],
    axis=1,
    method='nearest',
    keepdims=True,
)",
        ),
        (
            "percentile out forwarding",
            "out = np.empty((2,), dtype=np.float64)
result = MODULE.percentile(np.array([[1.0, 2.0], [3.0, 4.0]]), 50, axis=0, out=out)",
        ),
        (
            "quantile tuple q sequence axis",
            "result = MODULE.quantile(((1, 2, 3), (4, 5, 6)), [0.25, 0.75], axis=0)",
        ),
        (
            "quantile method fallback",
            "result = MODULE.quantile(np.array([1, 2, 3, 4]), 0.5, method='lower')",
        ),
        (
            "median tuple axis keepdims",
            "result = MODULE.median(((1, 3, 2), (6, 4, 5)), axis=1, keepdims=True)",
        ),
        (
            "median out forwarding",
            "out = np.empty((2,), dtype=np.float64)
result = MODULE.median(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=0, out=out)",
        ),
        (
            "median axis error type",
            "result = MODULE.median([1, 2, 3], axis=2)",
        ),
    ];

    for (name, body) in cases {
        let numpy_result = numpy_oracle(&numpy_outcome_script(body))?;
        let fnp_result = numpy_oracle(&fnp_outcome_script(body))?;

        assert_eq!(
            fnp_result, numpy_result,
            "percentile/quantile/median outcome mismatch for {name}\n\
             numpy: {numpy_result}\nfnp:   {fnp_result}"
        );
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// percentile
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn percentile_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
result = fnp.percentile(a, 50)
expected = np.percentile(a, 50)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "percentile basic should match numpy");
    Ok(())
}

/// A NaN lane's median / percentile / quantile is the NaN numpy's partition leaves last, payload
/// and sign included. The native kernels returned the canonical NaN (0x7ff8000000000000).
#[test]
fn median_percentile_quantile_return_numpys_nan_payload() -> Result<(), String> {
    let script = fnp_script(
        r#"
pos = np.array([0x7FF8000000000123], dtype=np.uint64).view(np.float64)[0]
neg = np.array([0xFFF8000000000456], dtype=np.uint64).view(np.float64)[0]
canonical = np.array([0x7FF8000000000000], dtype=np.uint64)
bad = []
cells = 0
payload_results = 0
for n in (5, 1000, 200_000):
    base = np.linspace(-3.0, 7.0, n)
    one = base.copy(); one[n // 2] = pos
    two = one.copy(); two[1] = neg
    grid = base.reshape(-1, 5).copy() if n % 5 == 0 else None
    for arr, label in ((one, "one payload"), (two, "two payloads")):
        calls = [("median", (arr,), {}), ("percentile", (arr, 50), {}), ("quantile", (arr, 0.25), {}),
                 ("percentile", (arr, [10, 90]), {}), ("quantile", (arr, [0.5, 0.75]), {})]
        if grid is not None:
            g = grid.copy(); g.flat[n // 2] = pos
            calls += [("median", (g,), {"axis": 1}), ("percentile", (g, 50), {"axis": 0}),
                      ("quantile", (g, 0.5), {"axis": -1, "keepdims": True})]
        for name, args, kw in calls:
            cells += 1
            r = getattr(fnp, name)(*args, **kw); e = getattr(np, name)(*args, **kw)
            r_bits = np.asarray(r, dtype=np.float64).view(np.uint64)
            e_bits = np.asarray(e, dtype=np.float64).view(np.uint64)
            payload_results += bool(np.any((e_bits != canonical[0]) & np.isnan(np.asarray(e, dtype=np.float64))))
            if type(r) is not type(e) or r_bits.shape != e_bits.shape or not np.array_equal(r_bits, e_bits):
                bad.append((name, n, label, kw))
print(cells, payload_results, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut fields = result.trim().splitn(3, ' ');
    let cells: usize = fields.next().unwrap_or("").parse().unwrap_or(0);
    let payload_results: usize = fields.next().unwrap_or("").parse().unwrap_or(0);
    assert!(cells >= 30, "cell table drifted: {result}");
    // Negative control: numpy must actually return non-canonical NaNs here, or a kernel that
    // canonicalises would pass.
    assert!(
        payload_results * 2 >= cells,
        "too few cells where numpy returns a payload NaN: {result}"
    );
    assert_eq!(
        fields.next().unwrap_or(""),
        "[]",
        "NaN payload differs from numpy: {result}"
    );
    Ok(())
}

#[test]
fn percentile_quantile_large_bounded_integer_scalar_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(20260708)
n = (1 << 20) + 33
cases = [
    ("i64", rng.integers(-500, 500, n, dtype=np.int64), 12.5, 0.125),
    ("u16", rng.integers(0, 30000, n, dtype=np.uint16), 75.0, 0.75),
]
ok = True
for label, a, p, q in cases:
    got_p = fnp.percentile(a, p)
    exp_p = np.percentile(a, p)
    got_q = fnp.quantile(a, q)
    exp_q = np.quantile(a, q)
    for op, got, exp in [("percentile", got_p, exp_p), ("quantile", got_q, exp_q)]:
        if str(np.asarray(got).dtype) != str(np.asarray(exp).dtype):
            print(("dtype", label, op, str(np.asarray(got).dtype), str(np.asarray(exp).dtype)))
            ok = False
        if not np.array_equal(np.asarray(got), np.asarray(exp)):
            print(("value", label, op, repr(got), repr(exp)))
            ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "large bounded integer percentile/quantile scalar paths should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn percentile_multiple() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
result = fnp.percentile(a, [25, 50, 75])
expected = np.percentile(a, [25, 50, 75])
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "percentile multiple should match numpy"
    );
    Ok(())
}

#[test]
fn percentile_2d_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.percentile(a, 50, axis=0)
expected = np.percentile(a, 50, axis=0)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "percentile 2d axis should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// quantile
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn quantile_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
result = fnp.quantile(a, 0.5)
expected = np.quantile(a, 0.5)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "quantile basic should match numpy");
    Ok(())
}

#[test]
fn quantile_multiple() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
result = fnp.quantile(a, [0.25, 0.5, 0.75])
expected = np.quantile(a, [0.25, 0.5, 0.75])
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "quantile multiple should match numpy"
    );
    Ok(())
}

#[test]
fn quantile_2d_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.quantile(a, 0.5, axis=1)
expected = np.quantile(a, 0.5, axis=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "quantile 2d axis should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// median
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn median_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 3, 2, 5, 4])
result = fnp.median(a)
expected = np.median(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "median basic should match numpy");
    Ok(())
}

#[test]
fn median_even_count() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4])
result = fnp.median(a)
expected = np.median(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "median even count should match numpy"
    );
    Ok(())
}

#[test]
fn median_2d_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.median(a, axis=1)
expected = np.median(a, axis=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "median 2d axis should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ptp (peak-to-peak)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ptp_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
result = fnp.ptp(a)
expected = np.ptp(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ptp basic should match numpy");
    Ok(())
}

#[test]
fn ptp_2d_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 5, 3], [2, 8, 1]])
result = fnp.ptp(a, axis=1)
expected = np.ptp(a, axis=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ptp 2d axis should match numpy");
    Ok(())
}

#[test]
fn ptp_2d_all() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 5, 3], [2, 8, 1]])
result = fnp.ptp(a)
expected = np.ptp(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ptp 2d all should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn percentile_50_equals_median() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 3, 2, 5, 4])
p50 = fnp.percentile(a, 50)
med = fnp.median(a)
print(np.allclose(p50, med))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "percentile 50 should equal median");
    Ok(())
}

#[test]
fn quantile_05_equals_percentile_50() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
q = fnp.quantile(a, 0.5)
p = fnp.percentile(a, 50)
print(np.allclose(q, p))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "quantile 0.5 should equal percentile 50"
    );
    Ok(())
}

#[test]
fn ptp_equals_max_minus_min() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
ptp_val = fnp.ptp(a)
manual = np.max(a) - np.min(a)
print(np.allclose(ptp_val, manual))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ptp should equal max - min");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge case tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn percentile_boundary_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
p0 = fnp.percentile(a, 0)
p100 = fnp.percentile(a, 100)
np_p0 = np.percentile(a, 0)
np_p100 = np.percentile(a, 100)
print(np.allclose(p0, np_p0) and np.allclose(p100, np_p100))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "percentile 0/100 should match numpy");
    Ok(())
}

#[test]
fn quantile_boundary_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
q0 = fnp.quantile(a, 0.0)
q1 = fnp.quantile(a, 1.0)
np_q0 = np.quantile(a, 0.0)
np_q1 = np.quantile(a, 1.0)
print(np.allclose(q0, np_q0) and np.allclose(q1, np_q1))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "quantile 0/1 should match numpy");
    Ok(())
}

#[test]
fn median_single_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([42.0])
result = fnp.median(a)
expected = np.median(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "median single element should match numpy"
    );
    Ok(())
}

#[test]
fn percentile_nan_handling() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
result = fnp.percentile(a, 50)
expected = np.percentile(a, 50)
# Both should return nan
print(np.isnan(result) == np.isnan(expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "percentile nan handling should match numpy"
    );
    Ok(())
}

#[test]
fn median_nan_handling() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0])
result = fnp.median(a)
expected = np.median(a)
# Both should return nan
print(np.isnan(result) == np.isnan(expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "median nan handling should match numpy"
    );
    Ok(())
}

#[test]
fn ptp_single_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([42.0])
result = fnp.ptp(a)
expected = np.ptp(a)
# ptp of single element should be 0
print(np.allclose(result, expected) and result == 0.0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ptp single element should be 0");
    Ok(())
}

#[test]
fn ptp_nan_propagation() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0])
result = fnp.ptp(a)
expected = np.ptp(a)
# Both should return nan
print(np.isnan(result) == np.isnan(expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ptp nan propagation should match numpy"
    );
    Ok(())
}

#[test]
fn percentile_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
fnp_result = fnp.percentile(a, 50)
np_result = np.percentile(a, 50)
print(type(fnp_result).__name__ == type(np_result).__name__)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim() == "True",
        "percentile scalar return type should match numpy: {result}"
    );
    Ok(())
}

// Array-q WITH an axis is delegated to numpy (no native multi-q-axis path) — the delegation must be
// byte-identical AND must happen BEFORE the whole-array extract (a perf fix: the wasted 32MB copy made
// percentile([25,50,75], axis=1) a 0.64x loss). This locks in the byte-exact parity across q-forms/axes.
#[test]
fn percentile_quantile_array_q_with_axis_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import numpy as np
ok = True
rng = np.random.default_rng(20260701)
m = rng.standard_normal((400, 500))
t = rng.standard_normal((40, 50, 30))
for q in ([25, 50, 75], [0, 100], [33.3, 66.6], np.array([10.0, 90.0])):
    for ax in (0, 1, -1):
        if not np.array_equal(np.asarray(fnp.percentile(m, q, axis=ax)),
                              np.percentile(m, q, axis=ax), equal_nan=True):
            ok = False
        qq = np.asarray(q) / 100.0
        if not np.array_equal(np.asarray(fnp.quantile(m, qq, axis=ax)),
                              np.quantile(m, qq, axis=ax), equal_nan=True):
            ok = False
    if not np.array_equal(np.asarray(fnp.percentile(t, q, axis=1)),
                          np.percentile(t, q, axis=1), equal_nan=True):
        ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "percentile/quantile array-q with axis must delegate byte-identically to numpy: {result}"
    );
    Ok(())
}

/// numpy computes the quantile family in `q`'s own type. An object `q` keeps object
/// arithmetic: `np.quantile([1, 2], Fraction(1, 2))` is `Fraction(3, 2)`, and fnp returned
/// the float 1.5 (numpy's own TestQuantile::test_quantile_gh_29003_Fraction). A `Decimal` `q`
/// gives a `Decimal`, `method='nearest'` with a `Fraction` raises in numpy, and a float32 `q`
/// scales in float32 in `percentile`. The native kernels compute in float64, so every `q`
/// that is not float64 or integer is numpy's. 18 of the 68 cells failed before the fix
/// (numpy 2.4.3); 0 fail after, on numpy 2.4.3 and 2.3.5.
///
/// Controls: Python-float, int, float64-array and 0-d q keep matching, and so do float16 /
/// float32 / object DATA (gated separately).
#[test]
fn quantile_family_computes_in_qs_own_type() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
from fractions import Fraction
from decimal import Decimal

def outcome(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            r = call()
            if isinstance(r, (np.ndarray, np.generic)):
                a = np.asarray(r)
                data = repr(a.tolist()) if a.dtype == object else a.tobytes()
                got = ("ok", type(r).__name__, a.dtype.str, a.shape, data)
            else:
                got = ("ok", type(r).__name__, repr(r))
        except Exception as ex:
            got = (type(ex).__name__, str(ex))
    return got + (sorted({w.category.__name__ for w in caught}),)

cases = {}
for fn in ("quantile", "percentile", "nanquantile", "nanpercentile"):
    scale = 100 if "percentile" in fn else 1
    cases[f"{fn} Fraction(1)"] = lambda m, fn=fn, s=scale: getattr(m, fn)([1, 2], Fraction(1) * s)
    cases[f"{fn} Fraction(1/2)"] = lambda m, fn=fn, s=scale: getattr(m, fn)([1, 2], Fraction(1, 2) * s)
    cases[f"{fn} Fraction list"] = lambda m, fn=fn, s=scale: getattr(m, fn)([1, 2, 3], [Fraction(1, 3) * s, Fraction(2, 3) * s])
    cases[f"{fn} Decimal"] = lambda m, fn=fn, s=scale: getattr(m, fn)([1, 2], Decimal("0.5") * s)
    cases[f"{fn} float q"] = lambda m, fn=fn, s=scale: getattr(m, fn)([1, 2], 0.5 * s)
    cases[f"{fn} int q"] = lambda m, fn=fn, s=scale: getattr(m, fn)([1, 2], 1 * s)
    cases[f"{fn} f32 q"] = lambda m, fn=fn, s=scale: getattr(m, fn)(np.arange(5.0), np.float32(0.3) * s)
    cases[f"{fn} f16 data"] = lambda m, fn=fn, s=scale: getattr(m, fn)(np.arange(50_001, dtype=np.float16), 0.999 * s)
    cases[f"{fn} f32 data"] = lambda m, fn=fn, s=scale: getattr(m, fn)(np.arange(11, dtype=np.float32), 0.35 * s)
    cases[f"{fn} object data"] = lambda m, fn=fn, s=scale: getattr(m, fn)(np.array([Fraction(1), Fraction(2)], dtype=object), 0.5 * s)
    cases[f"{fn} q array"] = lambda m, fn=fn, s=scale: getattr(m, fn)(np.arange(10.0), np.array([0.1, 0.9]) * s)
    cases[f"{fn} q 0-d f32"] = lambda m, fn=fn, s=scale: getattr(m, fn)(np.arange(10.0), np.array(0.25, np.float32) * s)
    cases[f"{fn} q>1"] = lambda m, fn=fn, s=scale: getattr(m, fn)([1, 2], 1.5 * s)
    cases[f"{fn} q nan"] = lambda m, fn=fn, s=scale: getattr(m, fn)([1, 2], float("nan"))
    cases[f"{fn} q complex"] = lambda m, fn=fn, s=scale: getattr(m, fn)([1, 2], 0.5j)
    cases[f"{fn} method weibull"] = lambda m, fn=fn, s=scale: getattr(m, fn)(np.arange(10.0), 0.3 * s, method="weibull")
    cases[f"{fn} method nearest Fraction"] = lambda m, fn=fn, s=scale: getattr(m, fn)([1, 2, 3], Fraction(1, 2) * s, method="nearest")
bad = []
for name, case in cases.items():
    ours, theirs = outcome(lambda: case(fnp)), outcome(lambda: case(np))
    if ours != theirs:
        bad.append(f"{name}: fnp={str(ours)[:150]} numpy={str(theirs)[:150]}")
print(len(cases), bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "68 []",
        "the quantile family must compute in q's own type as numpy does: {result}"
    );
    Ok(())
}

/// Explicit spellings of numpy's defaults that fnp read differently:
/// - `cov(x, bias=None)`: numpy sets `ddof = 1 if bias == 0 else 0` - EQUALITY with 0 - so an
///   explicit None is the BIASED estimate (1.5 for [1, 2.5, 4]); a truthiness read of a
///   defaulted `Option` gave the unbiased 2.25;
/// - `corrcoef` of one variable is numpy's `c / c` on a 0-d covariance, a numpy SCALAR
///   (`np.float64(1.0)`), where fnp returned a 0-d ndarray - in the DEFAULT call too;
/// - `select(..., default=None)` is an object fill in numpy (`[1, None, 3]`), where fnp read the
///   explicit None as the omitted 0 (`[1, 0, 3]`);
/// - a one-variable `corrcoef` is exactly 1.0 in numpy and was 0.9999999999999998 here.
///
/// Float VALUES compare to 9 decimals: numpy's 2-D covariance is a BLAS `dot` whose FMA and
/// blocking bits the no-FMA Gram kernel does not reproduce (the accepted matmul tolerance
/// class); type, dtype and shape compare exactly. 5 of the 12 cells failed before the fix
/// (numpy 2.4.3); 0 after, on numpy 2.4.3 and 2.3.5.
#[test]
fn cov_corrcoef_select_explicit_defaults_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings

def outcome(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            r = call()
            a = np.asarray(r)
            data = repr(a.tolist()) if a.dtype == object else np.round(a, 9).tobytes()
            got = ("ok", type(r).__name__, a.dtype.str, a.shape, data)
        except Exception as ex:
            got = (type(ex).__name__, str(ex))
    return got + (sorted({w.category.__name__ for w in caught}),)

x = np.array([1.0, 2.5, 4.0])
m = np.array([[1.0, 2.0, 4.0], [0.5, 1.5, 1.0]])
cases = {
    "cov bias=None": lambda n: n.cov(x, bias=None),
    "cov bias=0": lambda n: n.cov(x, bias=0),
    "cov bias=1": lambda n: n.cov(x, bias=1),
    "cov bias=False": lambda n: n.cov(x, bias=False),
    "cov 2-D bias=None": lambda n: n.cov(m, bias=None),
    "corrcoef 1-D": lambda n: n.corrcoef(x),
    "corrcoef 1-D rowvar=False": lambda n: n.corrcoef(x, rowvar=False),
    "corrcoef 2-D": lambda n: n.corrcoef(m),
    "corrcoef x, x": lambda n: n.corrcoef(x, x),
    "select default=None": lambda n: n.select([np.array([True, False, True])], [np.array([1, 2, 3])], default=None),
    "select default=0": lambda n: n.select([np.array([True, False, True])], [np.array([1, 2, 3])], default=0),
    "select omitted": lambda n: n.select([np.array([True, False, True])], [np.array([1, 2, 3])]),
}
bad = []
for name, case in cases.items():
    ours, theirs = outcome(lambda: case(fnp)), outcome(lambda: case(np))
    if ours != theirs:
        bad.append(f"{name}: fnp={str(ours)[:150]} numpy={str(theirs)[:150]}")
print(len(cases), bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "12 []",
        "cov/corrcoef/select explicit defaults must match numpy: {result}"
    );
    Ok(())
}

/// Many-q percentile / quantile / nanpercentile / nanquantile below the parallel floor (n < 2^19):
/// bytes against numpy for 1-1000 q over normal and duplicate-heavy data, keepdims included. The
/// serial route cloned the input and ran a quickselect PER q (90x numpy at n=1e5 with 4096 q);
/// it now selects every needed rank in one buffer. Duplicate-heavy data puts many q on the
/// same order statistic and adjacent (lo, lo + 1) pairs across recursion boundaries.
#[test]
fn many_q_percentile_family_matches_numpy_bytes_below_the_parallel_floor() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.simplefilter("ignore")
rng = np.random.default_rng(11)
def outcome(call):
    try:
        v = call()
    except Exception as ex:
        return (type(ex).__name__,)
    a = np.asarray(v)
    return (type(v).__name__, a.dtype.str, a.shape, a.tobytes())
cells, bad = 0, []
for n in (1, 2, 3, 10, 1000, 20000):
    for label, data in (("normal", rng.standard_normal(n)), ("dup", rng.integers(0, 7, n).astype("f8"))):
        with_nan = data.copy()
        if n > 2:
            with_nan[::3] = np.nan
        for nq in (1, 2, 5, 101, 1000):
            q = np.linspace(0, 100, nq) if nq > 1 else [37.0]
            calls = {
                "percentile": lambda m: m.percentile(data, q),
                "quantile": lambda m: m.quantile(data, np.asarray(q) / 100),
                "percentile keepdims": lambda m: m.percentile(data, q, keepdims=True),
                "nanpercentile": lambda m: m.nanpercentile(with_nan, q),
                "nanquantile": lambda m: m.nanquantile(with_nan, np.asarray(q) / 100),
            }
            for name, call in calls.items():
                cells += 1
                ours, theirs = outcome(lambda: call(fnp)), outcome(lambda: call(np))
                if ours != theirs:
                    bad.append(f"{name} {label} n={n} nq={nq}")
print(cells, bad[:8])
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "300 []",
        "many-q percentile family differs from numpy: {result}"
    );
    Ok(())
}

/// median / nanmedian along an axis on both sides of the lane floor (2^18 elements) and through
/// both routes: the in-place read of a float64 C-contiguous operand's last axis, and the extract
/// copy for everything else. A strided, Fortran-ordered or big-endian operand read in place would
/// return other lanes or byte-swapped values, so those cases are the negative controls. NaN lanes
/// are numpy's (payload, and the all-NaN warning).
#[test]
fn median_lanes_match_numpy_across_the_lane_floor_and_both_routes() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
rng = np.random.default_rng(2027)
bad, cells = [], 0
def same(label, ours, theirs):
    global cells
    cells += 1
    x, y = np.asarray(ours), np.asarray(theirs)
    if type(ours) is not type(theirs) or x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
        bad.append(label)
for rows, lane in ((64, 64), (256, 511), (512, 512), (3, 100_000), (2048, 1024)):
    a = rng.standard_normal((rows, lane))
    an = a.copy()
    an[::7, ::5] = np.nan
    a_nan_lane = a.copy()
    a_nan_lane[1, 3] = np.nan
    tag = f"{rows}x{lane}"
    for name, fn in [
        ("median ax1", lambda m: m.median(a, axis=1)),
        ("median ax-1", lambda m: m.median(a, axis=-1)),
        ("median ax1 keepdims", lambda m: m.median(a, axis=1, keepdims=True)),
        ("median 3d ax2", lambda m: m.median(a.reshape(1, rows, lane), axis=2)),
        ("median ax0", lambda m: m.median(a, axis=0)),
        ("median strided", lambda m: m.median(a[:, ::2], axis=1)),
        ("median F order", lambda m: m.median(np.asfortranarray(a), axis=1)),
        ("median big-endian", lambda m: m.median(a.astype(">f8"), axis=1)),
        ("median nan lane", lambda m: m.median(a_nan_lane, axis=1)),
        ("nanmedian ax1", lambda m: m.nanmedian(an, axis=1)),
        ("nanmedian ax-1 keepdims", lambda m: m.nanmedian(an, axis=-1, keepdims=True)),
        ("nanmedian ax0", lambda m: m.nanmedian(an, axis=0)),
        ("nanmedian big-endian", lambda m: m.nanmedian(an.astype(">f8"), axis=1)),
    ]:
        same(f"{name} {tag}", fn(fnp), fn(np))
v = rng.standard_normal(300_001)
same("median 1-d axis 0", fnp.median(v, axis=0), np.median(v, axis=0))
same("nanmedian 1-d axis -1", fnp.nanmedian(v, axis=-1), np.nanmedian(v, axis=-1))
all_nan = rng.standard_normal((8, 64))
all_nan[2] = np.nan
with warnings.catch_warnings(record=True) as ours_w:
    warnings.simplefilter("always")
    ours = fnp.nanmedian(all_nan, axis=1)
with warnings.catch_warnings(record=True) as theirs_w:
    warnings.simplefilter("always")
    theirs = np.nanmedian(all_nan, axis=1)
same("nanmedian all-NaN lane", ours, theirs)
if [w.category for w in ours_w] != [w.category for w in theirs_w]:
    bad.append("nanmedian all-NaN lane warnings")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "68 []",
        "median / nanmedian lanes differ from numpy: {result}"
    );
    Ok(())
}

/// A 1-D `cov` operand is one variable, so it takes the (1, n) Gram's route decision: numpy's
/// BLAS answers it from 200k observations. It skipped that gate and ran the native Gram at every
/// size (5.1x numpy at 2^20) with bits off numpy's in the last place; above the floor it must be
/// numpy's bytes exactly, and below it within DIV-COV-GRAM-NO-FMA's documented 1e-12.
#[test]
fn one_dimensional_cov_takes_the_gram_route_decision() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(3)
cells, bad = 0, []
for n in (250_000, 1 << 20):
    a = rng.standard_normal(n) * 3 + 1
    for kw in ({}, {"rowvar": False}, {"ddof": 0}, {"bias": True}):
        cells += 1
        ours, theirs = np.asarray(fnp.cov(a, **kw)), np.asarray(np.cov(a, **kw))
        if (ours.dtype, ours.shape, ours.tobytes()) != (theirs.dtype, theirs.shape, theirs.tobytes()):
            bad.append(f"n={n} {kw}: {ours!r} vs {theirs!r}")
for n in (10, 1000, 150_000):
    a = rng.standard_normal(n) * 3 + 1
    cells += 1
    ours, theirs = np.asarray(fnp.cov(a)), np.asarray(np.cov(a))
    if ours.dtype != theirs.dtype or ours.shape != theirs.shape or not np.allclose(ours, theirs, rtol=1e-12, atol=0):
        bad.append(f"n={n} beyond 1e-12: {ours!r} vs {theirs!r}")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "11 []",
        "1-D cov differs from numpy: {result}"
    );
    Ok(())
}
