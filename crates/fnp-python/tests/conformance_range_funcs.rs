//! Conformance tests for numpy range functions against NumPy oracle.
//!
//! Tests arange, linspace, logspace, geomspace.

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
fn range_functions_keyword_and_fallback_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def attempt(call_fn, *args, **kwargs):
    try:
        return ("ok", call_fn(*args, **kwargs))
    except Exception as exc:
        return ("err", type(exc).__name__)

def values_match(actual, expected):
    if isinstance(actual, tuple) or isinstance(expected, tuple):
        if not isinstance(actual, tuple) or not isinstance(expected, tuple):
            return False
        if len(actual) != len(expected):
            return False
        return all(values_match(a, e) for a, e in zip(actual, expected))
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if str(actual_array.dtype) != str(expected_array.dtype):
        return False
    if tuple(actual_array.shape) != tuple(expected_array.shape):
        return False
    if bool(actual_array.flags["WRITEABLE"]) != bool(expected_array.flags["WRITEABLE"]):
        return False
    if actual_array.dtype.kind in "fc":
        return bool(np.allclose(actual_array, expected_array, equal_nan=True))
    return bool(np.array_equal(actual_array, expected_array))

cases = [
    ("arange dtype keyword", "arange", lambda: ((1, 5), {"dtype": np.float32})),
    ("arange device none keyword", "arange", lambda: ((5,), {"device": None})),
    ("arange like ndarray keyword", "arange", lambda: ((3,), {"like": np.array([], dtype=np.int8)})),
    ("arange zero step error", "arange", lambda: ((1, 5, 0), {})),
    ("linspace retstep tuple", "linspace", lambda: ((0, 1), {"num": 5, "retstep": True})),
    (
        "linspace array endpoints dtype axis",
        "linspace",
        lambda: (([0, 10], [1, 20]), {"num": 3, "axis": 1, "dtype": np.float32}),
    ),
    ("linspace negative num error", "linspace", lambda: ((0, 1), {"num": -1})),
    (
        "logspace base dtype keyword",
        "logspace",
        lambda: ((0, 3), {"num": 4, "base": 2.0, "dtype": np.float32}),
    ),
    (
        "logspace array endpoints axis keyword",
        "logspace",
        lambda: (([0, 1], [2, 3]), {"num": 3, "axis": -1}),
    ),
    ("geomspace negative endpoints fallback", "geomspace", lambda: ((-1, -1000), {"num": 4})),
    (
        "geomspace array endpoints axis keyword",
        "geomspace",
        lambda: (([1, 10], [100, 1000]), {"num": 3, "axis": 1}),
    ),
    ("geomspace negative num error", "geomspace", lambda: ((1, 10), {"num": -2})),
]

ok = True
for label, name, factory in cases:
    args, kwargs = factory()
    actual = attempt(getattr(fnp, name), *args, **kwargs)
    args, kwargs = factory()
    expected = attempt(getattr(np, name), *args, **kwargs)
    if actual[0] != expected[0]:
        print(label)
        print(actual[0])
        print(expected[0])
        ok = False
    elif actual[0] == "err" and actual[1] != expected[1]:
        print(label)
        print(actual)
        print(expected)
        ok = False
    elif actual[0] == "ok" and not values_match(actual[1], expected[1]):
        print(label)
        print(np.asarray(actual[1]).dtype if not isinstance(actual[1], tuple) else "tuple")
        print(np.asarray(expected[1]).dtype if not isinstance(expected[1], tuple) else "tuple")
        ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "range-function keyword/fallback surfaces should match numpy: {result}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// arange
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn arange_stop_only() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.arange(10)
expected = np.arange(10)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "arange stop only should match numpy");
    Ok(())
}

#[test]
fn arange_start_stop() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.arange(2, 10)
expected = np.arange(2, 10)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "arange start stop should match numpy"
    );
    Ok(())
}

#[test]
fn arange_with_step() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.arange(0, 10, 2)
expected = np.arange(0, 10, 2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "arange with step should match numpy");
    Ok(())
}

#[test]
fn arange_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.arange(0.5, 5.5, 0.5)
expected = np.arange(0.5, 5.5, 0.5)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "arange float should match numpy");
    Ok(())
}

#[test]
fn arange_negative_step() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.arange(10, 0, -1)
expected = np.arange(10, 0, -1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "arange negative step should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// linspace
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn linspace_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.linspace(0, 10, 5)
expected = np.linspace(0, 10, 5)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "linspace basic should match numpy");
    Ok(())
}

#[test]
fn linspace_endpoint_false() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.linspace(0, 10, 5, endpoint=False)
expected = np.linspace(0, 10, 5, endpoint=False)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "linspace endpoint=False should match numpy"
    );
    Ok(())
}

#[test]
fn linspace_single_point() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.linspace(5, 5, 1)
expected = np.linspace(5, 5, 1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "linspace single point should match numpy"
    );
    Ok(())
}

#[test]
fn linspace_many_points() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.linspace(0, 1, 100)
expected = np.linspace(0, 1, 100)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "linspace many points should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// logspace
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn logspace_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.logspace(0, 2, 5)  # 10^0 to 10^2
expected = np.logspace(0, 2, 5)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logspace basic should match numpy");
    Ok(())
}

#[test]
fn logspace_custom_base() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.logspace(0, 3, 4, base=2)  # 2^0 to 2^3
expected = np.logspace(0, 3, 4, base=2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "logspace custom base should match numpy"
    );
    Ok(())
}

#[test]
fn logspace_endpoint_false() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.logspace(0, 2, 5, endpoint=False)
expected = np.logspace(0, 2, 5, endpoint=False)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "logspace endpoint=False should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// geomspace
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn geomspace_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.geomspace(1, 1000, 4)
expected = np.geomspace(1, 1000, 4)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "geomspace basic should match numpy");
    Ok(())
}

#[test]
fn geomspace_endpoint_false() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.geomspace(1, 1000, 4, endpoint=False)
expected = np.geomspace(1, 1000, 4, endpoint=False)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "geomspace endpoint=False should match numpy"
    );
    Ok(())
}

#[test]
fn geomspace_small_range() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.geomspace(0.1, 10, 5)
expected = np.geomspace(0.1, 10, 5)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "geomspace small range should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn linspace_arange_equivalence() -> Result<(), String> {
    let script = fnp_script(
        r#"
# linspace(0, 9, 10) should equal arange(10) for integers
linspace_result = fnp.linspace(0, 9, 10)
arange_result = fnp.arange(10).astype(float)
print(np.allclose(linspace_result, arange_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "linspace should match arange for equivalent params"
    );
    Ok(())
}

#[test]
fn logspace_geomspace_relationship() -> Result<(), String> {
    let script = fnp_script(
        r#"
# logspace(a, b, n) should equal geomspace(10^a, 10^b, n)
logspace_result = fnp.logspace(1, 3, 5)
geomspace_result = fnp.geomspace(10, 1000, 5)
print(np.allclose(logspace_result, geomspace_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "logspace should relate to geomspace via powers"
    );
    Ok(())
}

/// numpy broadcasts an ARRAY `base` against the samples (`logspace(0, 2, 3, base=[2, 10])` is
/// 3x2); `base: f64` rejected it with TypeError before the delegate could see it (numpy's own
/// TestLogspace::test_base_array). Scalar bases, including numpy scalars, and an explicit
/// `base=None` (numpy's TypeError) must still match.
#[test]
fn logspace_array_base_broadcasts_like_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def outcome(fn):
    try:
        r = fn()
        return ("ok", type(r).__name__, str(r.dtype), r.shape, np.round(r, 10).tolist())
    except Exception as exc:
        return ("err", type(exc).__name__)
cases = [
    lambda m: m.logspace(0, 2, 3, base=np.array([2.0, 10.0])),
    lambda m: m.logspace(0, 2, 3, base=[2, 10]),
    lambda m: m.logspace(0, 2, 3, base=[2, 10], axis=1),
    lambda m: m.logspace(0, 2, 3, base=np.array([[2.0], [3.0]])),
    lambda m: m.logspace(0, 2, 3, base=2.0),
    lambda m: m.logspace(0, 2, 3, base=np.float32(3.0)),
    lambda m: m.logspace(0, 2, 3),
    lambda m: m.logspace(0, 2, 3, base=None),
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
        "logspace with array base must match numpy: {result}"
    );
    Ok(())
}

/// `np.arange(0, 1, step=0.1)` is an everyday call. fnp's passthrough declared only
/// dtype/device/like, so `step=` (and `start=`/`stop=`) were a TypeError (numpy's own
/// TestDateTime::test_datetime_arange under the drop-in harness). Every keyword form must
/// match numpy, datetime ranges included.
#[test]
fn arange_accepts_start_stop_step_keywords() -> Result<(), String> {
    let script = fnp_script(
        r#"
def outcome(fn):
    try:
        r = fn()
        return ("ok", str(r.dtype), r.tolist())
    except Exception as exc:
        return ("err", type(exc).__name__)
cases = [
    lambda m: m.arange(0, 10, step=3),
    lambda m: m.arange(0.0, 1.0, step=0.25),
    lambda m: m.arange(start=2, stop=8),
    lambda m: m.arange(5, step=2),
    lambda m: m.arange("2010-01-01", "2010-01-05", step=2, dtype="M8[D]"),
    lambda m: m.arange(1, 4, dtype=np.float32),
    lambda m: m.arange(0, 5, step=0),
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
        "arange keyword forms must match numpy: {result}"
    );
    Ok(())
}
