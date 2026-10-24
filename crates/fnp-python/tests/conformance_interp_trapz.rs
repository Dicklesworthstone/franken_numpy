//! Conformance tests for numpy interp and trapz functions against NumPy oracle.
//!
//! Tests interp, trapz (using np.trapezoid for NumPy 2.x compatibility).

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

fn outcome_body(setup: &str, call_expr: &str) -> String {
    // A `\` line-continuation eats the SOURCE indentation, so the Python indent must
    // be injected via {I4}/{I8} placeholders (substituted after the eaten whitespace);
    // writing the indent as plain source spaces would emit a flat script -> the numpy
    // oracle raises IndentationError and every case fails (was a harness bug).
    format!(
        "{setup}\n\
         def outcome(op):\n\
         {I4}try:\n\
         {I8}value = {call_expr}\n\
         {I8}arr = np.asarray(value)\n\
         {I8}print('ok')\n\
         {I8}print(type(value).__name__)\n\
         {I8}print(str(arr.dtype))\n\
         {I8}print(tuple(arr.shape))\n\
         {I8}print(repr(arr.tolist()))\n\
         {I4}except Exception as exc:\n\
         {I8}print('err')\n\
         {I8}print(type(exc).__name__)\n\
         outcome(op)",
        I4 = "    ",
        I8 = "        ",
    )
}

fn numpy_outcome_script(function_expr: &str, setup: &str, call_expr: &str) -> String {
    format!(
        "import numpy as np\nop = {function_expr}\n{}",
        outcome_body(setup, call_expr)
    )
}

fn fnp_outcome_script(function_name: &str, setup: &str, call_expr: &str) -> String {
    fnp_script(format!(
        "op = fnp.{function_name}\n{}",
        outcome_body(setup, call_expr)
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// interp
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn interp_python_container_keyword_surfaces_match_numpy() -> Result<(), String> {
    let cases = [
        (
            "scalar tuple inputs with left/right keywords",
            "",
            "op(0.0, (1, 2, 3), (10, 20, 30), left=-5, right=99)",
        ),
        (
            "list inputs preserve ndarray metadata",
            "",
            "op([0.0, 1.5, 3.0], [1, 2, 3], [10, 20, 30], left=-1, right=100)",
        ),
        (
            "period keyword delegates angular interpolation",
            "",
            "op([0, 90, 270, 360], [0, 180, 360], [0.0, 1.0, 0.0], period=360)",
        ),
        (
            "tuple probe with ndarray xp fp",
            "xp = np.array([0.0, 2.0, 4.0])\nfp = np.array([0.0, 20.0, 40.0])",
            "op((1.0, 3.0), xp, fp)",
        ),
        ("missing fp error type", "", "op([0.0], [0.0])"),
        (
            "xp fp length mismatch error type",
            "",
            "op([0.0, 1.0], [0.0, 1.0], [10.0])",
        ),
    ];

    for (label, setup, call_expr) in cases {
        let numpy_result = numpy_oracle(&numpy_outcome_script("np.interp", setup, call_expr))?;
        let rust_result = numpy_oracle(&fnp_outcome_script("interp", setup, call_expr))?;

        assert_eq!(
            numpy_result, rust_result,
            "interp Python-container keyword surface mismatch for {label}"
        );
    }

    Ok(())
}

#[test]
fn interp_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
xp = [1, 2, 3]
fp = [3, 2, 0]
result = fnp.interp(2.5, xp, fp)
expected = np.interp(2.5, xp, fp)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "interp basic should match numpy");
    Ok(())
}

#[test]
fn interp_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
xp = [1, 2, 3, 4, 5]
fp = [10, 20, 30, 40, 50]
x = [1.5, 2.5, 3.5]
result = fnp.interp(x, xp, fp)
expected = np.interp(x, xp, fp)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "interp array should match numpy");
    Ok(())
}

#[test]
fn interp_outside_bounds() -> Result<(), String> {
    let script = fnp_script(
        r#"
xp = [1, 2, 3]
fp = [10, 20, 30]
x = [0, 4]  # outside bounds
result = fnp.interp(x, xp, fp)
expected = np.interp(x, xp, fp)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "interp outside bounds should match numpy"
    );
    Ok(())
}

#[test]
fn interp_with_left_right() -> Result<(), String> {
    let script = fnp_script(
        r#"
xp = [1, 2, 3]
fp = [10, 20, 30]
x = [0, 4]
result = fnp.interp(x, xp, fp, left=-1, right=-1)
expected = np.interp(x, xp, fp, left=-1, right=-1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "interp with left/right should match numpy"
    );
    Ok(())
}

#[test]
fn interp_single_point() -> Result<(), String> {
    let script = fnp_script(
        r#"
xp = [0, 1, 2, 3, 4]
fp = [0, 1, 4, 9, 16]
result = fnp.interp(1.5, xp, fp)
expected = np.interp(1.5, xp, fp)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "interp single point should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// trapz
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn trapz_python_container_keyword_surfaces_match_numpy() -> Result<(), String> {
    let cases = [
        ("list y default scalar", "", "op([1, 2, 3, 4])"),
        (
            "tuple y with x list",
            "",
            "op((1, 2, 3, 4), x=[0, 1, 3, 6])",
        ),
        (
            "nested list axis zero",
            "",
            "op([[1, 2, 3], [4, 5, 6]], axis=0)",
        ),
        (
            "nested tuple axis one dx keyword",
            "",
            "op(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), dx=0.5, axis=1)",
        ),
        (
            "ndarray y with broadcast x spacing",
            "y = np.array([[1.0, 2.0, 4.0], [2.0, 3.0, 5.0]])\nx = np.array([0.0, 0.5, 2.0])",
            "op(y, x=x, axis=-1)",
        ),
        ("axis type error parity", "", "op([1, 2, 3], axis='bad')"),
    ];

    for (label, setup, call_expr) in cases {
        let numpy_result = numpy_oracle(&numpy_outcome_script("np.trapezoid", setup, call_expr))?;
        let rust_result = numpy_oracle(&fnp_outcome_script("trapz", setup, call_expr))?;

        assert_eq!(
            numpy_result, rust_result,
            "trapz Python-container keyword surface mismatch for {label}"
        );
    }

    Ok(())
}

#[test]
fn trapz_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
y = [1, 2, 3, 4]
result = fnp.trapz(y)
expected = np.trapezoid(y)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "trapz basic should match numpy");
    Ok(())
}

#[test]
fn trapz_with_x() -> Result<(), String> {
    let script = fnp_script(
        r#"
y = [1, 2, 3, 4]
x = [0, 1, 2, 3]
result = fnp.trapz(y, x)
expected = np.trapezoid(y, x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "trapz with x should match numpy");
    Ok(())
}

#[test]
fn trapz_with_dx() -> Result<(), String> {
    let script = fnp_script(
        r#"
y = [1, 2, 3, 4]
result = fnp.trapz(y, dx=0.5)
expected = np.trapezoid(y, dx=0.5)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "trapz with dx should match numpy");
    Ok(())
}

#[test]
fn trapz_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
y = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.trapz(y, axis=0)
expected = np.trapezoid(y, axis=0)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "trapz 2d axis=0 should match numpy");
    Ok(())
}

#[test]
fn trapz_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
y = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.trapz(y, axis=1)
expected = np.trapezoid(y, axis=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "trapz 2d axis=1 should match numpy");
    Ok(())
}

#[test]
fn trapz_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
y = np.array([0.0, 1.0, 1.0, 0.0])
result = fnp.trapz(y)
expected = np.trapezoid(y)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "trapz float should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn interp_exact_points() -> Result<(), String> {
    let script = fnp_script(
        r#"
xp = [1, 2, 3, 4, 5]
fp = [10, 20, 30, 40, 50]
# Interpolating at exact points should return exact values
result = fnp.interp(xp, xp, fp)
print(np.allclose(result, fp))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "interp at exact points should return exact values"
    );
    Ok(())
}

#[test]
fn trapz_rectangle_integration() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Trapezoidal rule on constant function should give width * height
y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])  # constant 5
x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # width 4
result = fnp.trapz(y, x)
# Should be 5 * 4 = 20
print(np.allclose(result, 20.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "trapz of constant should equal width * height"
    );
    Ok(())
}

#[test]
fn interp_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(1.5)
xp = [1, 2, 3]
fp = [10, 20, 30]
fnp_result = fnp.interp(x, xp, fp)
np_result = np.interp(x, xp, fp)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "interp scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn int_trapezoid_via_f64_conversion_matches_numpy() -> Result<(), String> {
    // Integer y/x within +-2^51 convert once to f64 and ride the existing
    // paths. The fnp f64 FAST path is an allclose-level surface by design
    // (different summation order, ~1e-14) - int rows assert the same
    // tolerance the f64 surface already carries; DELEGATED forms (huge
    // values, non-contig) stay byte-exact since the conversion is
    // value-transparent to numpy's own chain in range.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(163)
verdicts = []
# half-range-safe values per width (in-dtype pairwise adds cannot wrap)
for dt, lo, hi in [(np.int64, -1000, 1000), (np.int32, -1000, 1000), (np.int16, -1000, 1000), (np.uint8, 0, 100)]:
    y = rng.integers(lo, hi, 4_000_000).astype(dt)
    r = fnp.trapezoid(y); e = np.trapezoid(y)
    if not np.allclose(np.asarray(r, dtype=np.float64), np.asarray(e, dtype=np.float64), rtol=1e-12, atol=1e-6):
        verdicts.append(f"FAIL 1-D {dt.__name__}")
# FULL-range uint8 can wrap in-dtype -> must DELEGATE, byte-exact
yw = rng.integers(0, 256, 2_000_000).astype(np.uint8)
if np.asarray(fnp.trapezoid(yw)).tobytes() != np.asarray(np.trapezoid(yw)).tobytes():
    verdicts.append("FAIL full-range uint8 delegate bytes")
# int y with int x coordinates
y = rng.integers(-1000, 1000, 2_000_000)
x = np.arange(2_000_000) * 2
r = fnp.trapezoid(y, x); e = np.trapezoid(y, x)
if not np.allclose(float(r), float(e), rtol=1e-12, atol=1e-6):
    verdicts.append("FAIL int x coords")
# dx scalar
r = fnp.trapezoid(y, dx=0.5); e = np.trapezoid(y, dx=0.5)
if not np.allclose(float(r), float(e), rtol=1e-12, atol=1e-6):
    verdicts.append("FAIL dx scalar")
# 2-D axis
y2 = rng.integers(-1000, 1000, (2048, 1024))
for ax in (0, 1):
    r = fnp.trapezoid(y2, axis=ax); e = np.trapezoid(y2, axis=ax)
    if not np.allclose(r, e, rtol=1e-12, atol=1e-6):
        verdicts.append(f"FAIL 2-D ax={ax}")
# huge values keep the delegate: BYTE-exact
big = rng.integers(2**60, 2**62, 300_000)
if np.asarray(fnp.trapezoid(big)).tobytes() != np.asarray(np.trapezoid(big)).tobytes():
    verdicts.append("FAIL huge-value delegate bytes")

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

W = rng.integers(-1000, 1000, 16_000_000)
tn = best(lambda: np.trapezoid(W)); tf = best(lambda: fnp.trapezoid(W))
print(f"TRAPEZOID_INT_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces TRAPEZOID_INT_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last, "True",
        "int trapezoid via f64 conversion must match numpy: {result}"
    );
    Ok(())
}

/// trapezoid over ten dtypes (narrow signed/unsigned ints included) at (40,) and (256, 300), with
/// y alone, an integer x (random and reversed), dx, a float x, a float y over an integer x, and
/// axis=0 - byte-compared with numpy (140 cells). numpy takes diff(x) in x's integer dtype (a
/// decreasing unsigned x wraps) and multiplies d * (y[1:] + y[:-1]) in the promoted integer
/// dtype (a narrow product overflows). Before the fix (bead .8) an integer x was converted to
/// float64 alongside y and the route skipped both wraps: uint16 / uint8 / uint64 y with an
/// integer x at (256, 300) differed in every output (by up to 7.6e22 for uint64). The test
/// above never reached it: its only integer x is an increasing int64 arange.
#[test]
fn trapezoid_with_integer_x_keeps_numpys_integer_wraparound() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(12)
bad = []
cells = 0
for dt in (np.uint16, np.int16, np.uint8, np.int8, np.int32, np.uint32, np.uint64, np.int64, np.float32, np.float64):
    for shape in ((40,), (256, 300)):
        y = rng.integers(0, 50, shape).astype(dt)
        x = rng.integers(0, 50, shape).astype(dt)
        variants = (("y", (y,), {}), ("y,x", (y,), {"x": x}), ("y,xrev", (y,), {"x": x[::-1].copy()}),
                    ("y,dx", (y,), {"dx": 0.5}), ("yf,x", (y.astype(float),), {"x": x}),
                    ("y,xf", (y,), {"x": x.astype(float)}),
                    ("y,x,axis0", (y,), {"x": x, "axis": 0} if len(shape) > 1 else {"x": x}))
        for label, args, kw in variants:
            cells += 1
            r, s = np.asarray(fnp.trapezoid(*args, **kw)), np.asarray(np.trapezoid(*args, **kw))
            if r.dtype != s.dtype or r.shape != s.shape or r.tobytes() != s.tobytes():
                bad.append(f"{dt.__name__} {shape} {label}")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (cells, bad) = result.trim().split_once(' ').unwrap_or(("0", &result));
    assert_eq!(cells, "140", "cell table drifted: {result}");
    assert_eq!(bad, "[]", "trapezoid must match numpy bytes: {result}");
    Ok(())
}

/// An integer `y` along its last axis is read natively with numpy's IN-DTYPE pair sums, which
/// wrap: every width at full range, 1-D and N-D, from 2 elements through the pool floor, plus the
/// shapes the route declines. And `dx`'s TYPE is numpy's (NEP 50): a numpy scalar or 0-d array
/// is strong (float32 `y` with `np.float64(0.1)` answers float64), a Python int keeps an integer
/// `y`'s product in its dtype (int16 wraps at `dx=3`; `dx=10**6` is numpy's OverflowError). Before
/// the fix 33 of these 194 cells differed - every one a `dx` of those types, which the parser
/// had turned into a Python float for every delegate. Byte-compared, result type included.
#[test]
fn trapezoid_integer_y_wraps_in_dtype_and_dx_keeps_its_type() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.simplefilter("ignore")
rng = np.random.default_rng(29)
bad = []
cells = 0
def check(label, y, **kw):
    global cells
    cells += 1
    try:
        s = np.trapezoid(y, **kw)
    except Exception as exc:
        s = ("raise", type(exc).__name__)
    try:
        r = fnp.trapezoid(y, **kw)
    except Exception as exc:
        r = ("raise", type(exc).__name__)
    if isinstance(s, tuple) or isinstance(r, tuple):
        if not (isinstance(s, tuple) and isinstance(r, tuple) and r == s):
            bad.append(label)
        return
    ra, sa = np.asarray(r), np.asarray(s)
    if type(r) is not type(s) or ra.dtype != sa.dtype or ra.shape != sa.shape or ra.tobytes() != sa.tobytes():
        bad.append(label)
for dt in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64):
    info = np.iinfo(dt)
    for shape in ((2,), (257,), (70_001,), (1 << 21 | 3,), (3, 2), (64, 300), (2, 3, 129), (1024, 2049)):
        if np.prod(shape) > 1 << 20 and dt not in (np.int16, np.uint64):
            continue
        y = rng.integers(info.min, info.max, shape, dtype=dt, endpoint=True)
        check(f"{dt.__name__} {shape}", y)
        check(f"{dt.__name__} {shape} dx=0.37", y, dx=0.37)
y16 = rng.integers(-30000, 30000, 5000).astype(np.int16)
check("int16 dx=inf", y16, dx=float("inf"))
check("int16 dx=nan", y16, dx=float("nan"))
y2 = rng.integers(-30000, 30000, (300, 7)).astype(np.int16)
check("int16 axis0", y2, axis=0)
check("int16 strided", y16[::3])
check("int16 one", y16[:1])
check("int16 empty rows", np.zeros((0, 5), dtype=np.int16))
big16 = np.tile(np.array([20000, 19000, -20000, 15000], dtype=np.int16), 1 << 15)
y32 = (rng.random(1000) * 100).astype(np.float32)
y64 = rng.random(1000) * 100
for label, y in (("big16", big16), ("big16 2d", big16.reshape(64, -1)), ("f32", y32), ("f64", y64),
                 ("f32 2d", y32.reshape(10, 100)), ("i16", y16)):
    for dx in (3, 1, 2.5, True, 10**6, -(2**30), np.float64(0.1), np.float32(0.1), np.float16(0.5),
               np.int64(3), np.int8(3), np.longdouble(0.1), np.array(0.1), np.array(0.1, dtype=np.float32)):
        check(f"{label} dx={dx!r}", y, dx=dx)
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (cells, bad) = result.trim().split_once(' ').unwrap_or(("0", &result));
    assert_eq!(cells, "194", "cell table drifted: {result}");
    assert_eq!(
        bad, "[]",
        "trapezoid must match numpy's type and bytes: {result}"
    );
    Ok(())
}
