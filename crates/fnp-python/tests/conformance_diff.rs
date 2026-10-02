//! Conformance tests for numpy.diff against NumPy oracle.
//!
//! Tests diff (calculate n-th discrete difference).

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
fn diff_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 7, 11])
result = fnp.diff(a)
expected = np.diff(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diff basic should match numpy");
    Ok(())
}

#[test]
fn diff_n2() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 7, 11])
result = fnp.diff(a, n=2)
expected = np.diff(a, n=2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diff with n=2 should match numpy");
    Ok(())
}

#[test]
fn diff_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.diff(a, axis=0)
expected = np.diff(a, axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diff 2D axis=0 should match numpy");
    Ok(())
}

#[test]
fn diff_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.diff(a, axis=1)
expected = np.diff(a, axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diff 2D axis=1 should match numpy");
    Ok(())
}

#[test]
fn diff_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.5, 4.5, 7.0])
result = fnp.diff(a)
expected = np.diff(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diff float should match numpy");
    Ok(())
}

#[test]
fn diff_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+1j, 3+4j, 6+9j], dtype=np.complex128)
fnp_result = fnp.diff(z)
np_result = np.diff(z)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diff complex should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Error behavior tests
// ─────────────────────────────────────────────────────────────────────────────

fn classify_error(script: &str) -> String {
    let output = std::process::Command::new("python3")
        .args(["-c", script])
        .output()
        .expect("python3 should be available");
    if output.status.success() {
        "ok".to_string()
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("AxisError") || stderr.contains("axis") {
            "AxisError".to_string()
        } else if stderr.contains("ValueError") {
            "ValueError".to_string()
        } else {
            format!("other: {}", stderr.lines().last().unwrap_or(""))
        }
    }
}

#[test]
fn diff_axis_out_of_bounds_raises_axiserror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(12).reshape(3, 4)
fnp.diff(a, axis=5)
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(12).reshape(3, 4)
np.diff(a, axis=5)
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "diff with out-of-bounds axis should raise same error as numpy"
    );
}

#[test]
fn diff_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.inf, 3.0, np.nan, 5.0])
fnp_result = fnp.diff(a)
np_result = np.diff(a)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diff special values should match numpy"
    );
    Ok(())
}

#[test]
fn diff_constant_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([5.0, 5.0, 5.0, 5.0])
fnp_result = fnp.diff(a)
np_result = np.diff(a)
# Diff of constant should be zero
print(np.allclose(fnp_result, np_result) and np.allclose(fnp_result, 0.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diff constant array should match numpy"
    );
    Ok(())
}

#[test]
fn diff_two_elements() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 5.0])
fnp_result = fnp.diff(a)
np_result = np.diff(a)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diff two elements should match numpy"
    );
    Ok(())
}

#[test]
fn diff_with_prepend() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 7])
fnp_result = fnp.diff(a, prepend=0)
np_result = np.diff(a, prepend=0)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diff with prepend should match numpy"
    );
    Ok(())
}

#[test]
fn diff_with_append() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 7])
fnp_result = fnp.diff(a, append=20)
np_result = np.diff(a, append=20)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diff with append should match numpy");
    Ok(())
}

#[test]
fn diff_signed_zero_parity() -> Result<(), String> {
    // Test signed-zero behavior for diff (a[i+1] - a[i])
    // diff computes subtraction, so signed-zero rules apply
    let script = fnp_script(
        r#"
# Signed-zero diff semantics (subtraction rules)
# 0.0 - 0.0 = 0.0, -0.0 - (-0.0) = 0.0, 0.0 - (-0.0) = 0.0, -0.0 - 0.0 = -0.0
tests = [
    ([0.0, 0.0, 0.0], [False, False]),      # diff: [0, 0]
    ([-0.0, -0.0, -0.0], [False, False]),   # diff: [0, 0]
    ([0.0, -0.0], [True]),                  # diff: [-0] (0-(-0) = 0? or -0-0?)
]
all_pass = True
for values, expected_signbits in tests:
    arr = np.array(values)
    fnp_result = fnp.diff(arr)
    np_result = np.diff(arr)
    fnp_signs = np.signbit(fnp_result).tolist()
    np_signs = np.signbit(np_result).tolist()
    if fnp_signs != np_signs:
        print(f"FAIL: diff({values}) fnp signbit={fnp_signs} np signbit={np_signs}")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diff signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn diff_f16_1d_parallel_matches_numpy() -> Result<(), String> {
    // The native candidate widens each adjacent binary16 pair to binary32,
    // subtracts, and narrows once. Exercise byte parity plus every warning
    // class that must stay on NumPy's delegated path.
    let script = fnp_script(
        r#"
import time
import warnings
verdicts = []
rng = np.random.default_rng(20260714)
n = 1 << 21
a = (rng.standard_normal(n) * 2).astype(np.float16)

def same(tag, arr, **kwargs):
    with warnings.catch_warnings(record=True) as wf:
        warnings.simplefilter("always")
        r = fnp.diff(arr, **kwargs)
    with warnings.catch_warnings(record=True) as wn:
        warnings.simplefilter("always")
        e = np.diff(arr, **kwargs)
    if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
        verdicts.append(f"FAIL bytes {tag}")
    if [(w.category.__name__, str(w.message)) for w in wf] != [(w.category.__name__, str(w.message)) for w in wn]:
        verdicts.append(f"FAIL warnings {tag}")

same("random", a)
edges = np.array([0.0, -0.0, 6e-8, -6e-8, 65504.0, 65472.0,
                  -65504.0, 0.5, np.inf, 1.0, -np.inf, -1.0,
                  2048.0, 2050.0], dtype=np.float16)
same("edge-cycle", np.tile(edges, n // edges.size + 1)[:n])
for tag, pair in (
    ("nan", (np.float16(np.nan), np.float16(1.0))),
    ("finite-overflow", (np.float16(-65504.0), np.float16(65504.0))),
    ("positive-inf-invalid", (np.float16(np.inf), np.float16(np.inf))),
    ("negative-inf-invalid", (np.float16(-np.inf), np.float16(-np.inf))),
):
    h = a.copy()
    h[500000:500002] = pair
    same(tag, h)
same("below-gate", a[:4096])
same("2d-delegate", a[:1 << 20].reshape(1024, 1024), axis=1)
same("n2-delegate", a, n=2)

def best(fn, reps=5):
    fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return min(samples)

timed = (rng.standard_normal(8_000_000) * 2).astype(np.float16)
tn = best(lambda: np.diff(timed))
tf = best(lambda: fnp.diff(timed))
print(f"DIFF_F16_8M_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}");
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "True",
        "f16 1-D diff must match NumPy bytes and warnings: {result}"
    );
    Ok(())
}

/// Integer and float32 `diff` goes to the pool from 16 MiB of INPUT (`diff_typed`), not from 2^21
/// elements: an int8 of 2^22 stays serial, an int8 of 2^24 and an int32 / float32 of 2^22 fan
/// out. Chunking must not change a bit on either side of that floor - wrapping subtraction across
/// chunk seams (int8 -128 - 1), both axes of a 2-D operand, and the n - 1 boundary.
#[test]
fn integer_diff_matches_numpy_on_both_sides_of_the_byte_floor() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(20261010)
bad = []
count = 0
def check(a, **kw):
    global count
    count += 1
    ours, theirs = fnp.diff(a, **kw), np.diff(a, **kw)
    if ours.dtype != theirs.dtype or ours.shape != theirs.shape or ours.tobytes() != theirs.tobytes():
        bad.append((a.dtype.name, a.shape, kw))
for dtype, sizes in [(np.int8, [1 << 22, (1 << 24) + 1]), (np.uint8, [1 << 22, 1 << 24]),
                     (np.int16, [1 << 22, 1 << 23]), (np.int32, [(1 << 22) - 1, 1 << 22]),
                     (np.float32, [1 << 22])]:
    for n in sizes:
        if np.dtype(dtype).kind == "f":
            a = rng.standard_normal(n).astype(dtype)
        else:
            info = np.iinfo(dtype)
            a = rng.integers(info.min, info.max, n, dtype=dtype, endpoint=True)
        check(a)
check(np.tile(np.array([127, -128], dtype=np.int8), 1 << 23))
m = rng.integers(-128, 127, (4096, 4096), dtype=np.int8)
check(m, axis=0); check(m, axis=1)
print(bad if bad else True, count)
"#
        .into(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 12");
    Ok(())
}

/// diff, all and any on complex and float16 operands the native routes do not take are numpy's
/// own call: value, dtype, shape, result type and ownership. Their copy-based tails answered
/// float16 diff (and float16 all/any along an axis) with reshaped views - 18 of these 87 cells
/// differed - and never beat numpy (complex diff 3.07x at 64 elements).
#[test]
fn complex_and_float16_diff_all_any_are_numpys() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(106)
bad, cells = [], 0
def outcome(fn):
    try:
        v = fn()
    except Exception as exc:
        return ("raise", type(exc).__name__)
    a = np.asarray(v)
    return (type(v).__name__, a.dtype.str, a.shape, a.tobytes(), a.flags.owndata if isinstance(v, np.ndarray) else None)
def check(label, call):
    global cells
    cells += 1
    if outcome(lambda: call(np)) != outcome(lambda: call(fnp)):
        bad.append(label)
for dt in ("complex64", "complex128", "float16"):
    for shp in ((2,), (64,), (4096,), (8, 8), (64, 64)):
        if np.dtype(dt).kind == "c":
            a = (rng.standard_normal(shp) + 1j * rng.standard_normal(shp)).astype(dt)
        else:
            a = rng.standard_normal(shp).astype(dt)
        a.flat[::7] = 0
        check(f"diff {dt} {shp}", lambda m, a=a: m.diff(a))
        check(f"diff {dt} {shp} n=2", lambda m, a=a: m.diff(a, n=2))
        for name in ("all", "any"):
            check(f"{name} {dt} {shp}", lambda m, a=a, name=name: getattr(m, name)(a))
            if len(shp) > 1:
                check(f"diff {dt} {shp} axis0 {name}", lambda m, a=a: m.diff(a, axis=0))
                check(f"{name} {dt} {shp} axis1", lambda m, a=a, name=name: getattr(m, name)(a, axis=1))
check("all list", lambda m: m.all([1, 2, 0]))
check("any list", lambda m: m.any([0, 0.0, 1j]))
check("diff list", lambda m: m.diff([1 + 1j, 2, 5j]))
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (cells, bad) = result.trim().split_once(' ').unwrap_or(("0", &result));
    assert_eq!(cells, "87", "cell table drifted: {result}");
    assert_eq!(
        bad, "[]",
        "complex / float16 diff, all, any must be numpy's: {result}"
    );
    Ok(())
}
