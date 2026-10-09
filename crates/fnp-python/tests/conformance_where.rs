//! Conformance tests for numpy.where against NumPy oracle.
//!
//! Tests the native Rust where implementation against NumPy.
//!
//! np.where has two modes:
//! - where(condition): returns tuple of indices where condition is True (like nonzero)
//! - where(condition, x, y): returns x where condition is True, y otherwise

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
fn where_python_container_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def clean(value):
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clean(item) for item in value)
    return value

def normalize(value):
    if isinstance(value, tuple):
        arrays = []
        for item in value:
            array = np.asarray(item)
            arrays.append((str(array.dtype), tuple(array.shape), clean(array.tolist())))
        return ("tuple", len(value), arrays)
    array = np.asarray(value)
    return ("array", type(value).__name__, str(array.dtype), tuple(array.shape), clean(array.tolist()))

def where_outcome(where_fn, *args, **kwargs):
    try:
        return ("ok", normalize(where_fn(*args, **kwargs)))
    except Exception as exc:
        return ("err", type(exc).__name__, str(exc))

cases = [
    ("one arg list condition", lambda: (([False, True, False, True],), {})),
    ("one arg nested tuple condition", lambda: ((((0, 1), (2, 0)),), {})),
    ("list condition scalar choices", lambda: (([True, False, True], 1, 0), {})),
    (
        "tuple condition tuple choices",
        lambda: (((True, False, True), (1.5, 2.5, 3.5), (10.5, 20.5, 30.5)), {}),
    ),
    (
        "nested list string choices",
        lambda: (([[True, False], [False, True]], [["a", "b"], ["c", "d"]], "fallback"), {}),
    ),
    (
        "object none choices",
        lambda: (([True, False, True], None, np.array([1, 2, 3], dtype=object)), {}),
    ),
    (
        "broadcast scalar condition",
        lambda: ((True, np.array([1, 2, 3]), np.array([10, 20, 30])), {}),
    ),
    (
        "condition kwargs",
        lambda: ((), {"condition": [True, False, True], "x": [1, 2, 3], "y": [10, 20, 30]}),
    ),
    ("partial args error", lambda: (([True, False], [1, 2]), {})),
]

ok = True
for label, factory in cases:
    args, kwargs = factory()
    actual = where_outcome(fnp.where, *args, **kwargs)
    args, kwargs = factory()
    expected = where_outcome(np.where, *args, **kwargs)
    if actual != expected:
        print(label)
        print(actual)
        print(expected)
        ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "where Python-container surfaces should match numpy: {result}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// where(condition, x, y) - selection mode
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn where_basic_selection() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False, True, False])
x = np.array([1, 2, 3, 4])
y = np.array([10, 20, 30, 40])
result = fnp.where(condition, x, y)
expected = np.where(condition, x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "where basic selection should match numpy"
    );
    Ok(())
}

#[test]
fn where_float_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False, True, False])
x = np.array([1.5, 2.5, 3.5, 4.5])
y = np.array([10.5, 20.5, 30.5, 40.5])
result = fnp.where(condition, x, y)
expected = np.where(condition, x, y)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "where float values should match numpy"
    );
    Ok(())
}

#[test]
fn where_2d_selection() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([[True, False], [False, True]])
x = np.array([[1, 2], [3, 4]])
y = np.array([[10, 20], [30, 40]])
result = fnp.where(condition, x, y)
expected = np.where(condition, x, y)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "where 2d selection should match numpy"
    );
    Ok(())
}

#[test]
fn where_broadcast_condition() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False])
x = np.array([[1, 2], [3, 4]])
y = np.array([[10, 20], [30, 40]])
result = fnp.where(condition, x, y)
expected = np.where(condition, x, y)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "where broadcast condition should match numpy"
    );
    Ok(())
}

#[test]
fn where_scalar_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False, True])
x = np.array([1, 1, 1])
y = np.array([0, 0, 0])
result = fnp.where(condition, x, y)
expected = np.where(condition, x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "where scalar values should match numpy"
    );
    Ok(())
}

#[test]
fn where_python_scalar_choices_preserve_numpy_weak_promotion() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False])
cases = [
    (1, np.array([2, 3], dtype=np.int8)),
    (np.array([1, 2], dtype=np.int8), 3),
    (1, np.array([2, 3], dtype=np.uint8)),
    (np.array([1, 2], dtype=np.uint8), 3),
    (1.0, np.array([2, 3], dtype=np.float32)),
    (np.array([1, 2], dtype=np.float32), 3.0),
    (np.array([1, 2], dtype=np.float16), 3.0),
]
outcomes = []
for x, y in cases:
    result = fnp.where(condition, x, y)
    expected = np.where(condition, x, y)
    outcomes.append(
        np.array_equal(result, expected)
        and result.shape == expected.shape
        and result.dtype == expected.dtype
    )
print(all(outcomes))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "where Python scalar choices should use NumPy weak scalar promotion"
    );
    Ok(())
}

#[test]
fn where_explicit_none_choice_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False, True, False])
y = np.array([1, 2, 3, 4])
left_none = fnp.where(condition, None, y)
left_expected = np.where(condition, None, y)
right_none = fnp.where(condition, y, None)
right_expected = np.where(condition, y, None)
both_none = fnp.where(condition, None, None)
both_expected = np.where(condition, None, None)
print(
    np.array_equal(left_none, left_expected)
    and left_none.dtype == left_expected.dtype
    and np.array_equal(right_none, right_expected)
    and right_none.dtype == right_expected.dtype
    and np.array_equal(both_none, both_expected)
    and both_none.dtype == both_expected.dtype
)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "where explicit None choices should match numpy object selection"
    );
    Ok(())
}

#[test]
fn where_rejects_positional_only_keywords() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False])
x = np.array([1, 2])
y = np.array([10, 20])

def error_type(call):
    try:
        call()
        return "OK"
    except Exception as exc:
        return type(exc).__name__

cases = [
    (
        error_type(lambda: fnp.where(condition=condition)),
        error_type(lambda: np.where(condition=condition)),
    ),
    (
        error_type(lambda: fnp.where(condition, x=x, y=y)),
        error_type(lambda: np.where(condition, x=x, y=y)),
    ),
]
print(all(ours == theirs == "TypeError" for ours, theirs in cases))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "where positional-only keyword rejection should match numpy"
    );
    Ok(())
}

#[test]
fn where_invalid_positional_arity_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False])
x = np.array([1, 2])
y = np.array([10, 20])

def error_surface(call):
    try:
        call()
        return ("OK", "")
    except Exception as exc:
        return (type(exc).__name__, str(exc))

cases = [
    (
        error_surface(lambda: fnp.where(condition, x)),
        error_surface(lambda: np.where(condition, x)),
    ),
    (
        error_surface(lambda: fnp.where(condition, x, y, 99)),
        error_surface(lambda: np.where(condition, x, y, 99)),
    ),
]
print(all(ours == theirs and ours[0] != "OK" for ours, theirs in cases))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "where invalid positional arity errors should match numpy"
    );
    Ok(())
}

#[test]
fn where_ndarray_subclass_dispatch_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
class WhereOverride(np.ndarray):
    def __array_function__(self, func, types, args, kwargs):
        if func is np.where:
            return "OVERRIDE"
        return NotImplemented

def subclass(values):
    return np.asarray(values).view(WhereOverride)

condition = np.array([True, False])
x = np.array([1, 2])
y = np.array([10, 20])

cases = [
    (fnp.where(subclass([True, False])), np.where(subclass([True, False]))),
    (fnp.where(subclass([True, False]), x, y), np.where(subclass([True, False]), x, y)),
    (fnp.where(condition, subclass([1, 2]), y), np.where(condition, subclass([1, 2]), y)),
    (fnp.where(condition, x, subclass([10, 20])), np.where(condition, x, subclass([10, 20]))),
]
print(all(ours == theirs == "OVERRIDE" for ours, theirs in cases))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "where should honor ndarray subclass __array_function__ dispatch"
    );
    Ok(())
}

#[test]
fn where_all_true() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, True, True])
x = np.array([1, 2, 3])
y = np.array([10, 20, 30])
result = fnp.where(condition, x, y)
expected = np.where(condition, x, y)
print(np.array_equal(result, expected) and np.array_equal(result, x))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "where all true should return x");
    Ok(())
}

#[test]
fn where_all_false() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([False, False, False])
x = np.array([1, 2, 3])
y = np.array([10, 20, 30])
result = fnp.where(condition, x, y)
expected = np.where(condition, x, y)
print(np.array_equal(result, expected) and np.array_equal(result, y))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "where all false should return y");
    Ok(())
}

#[test]
fn where_empty_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([], dtype=bool)
x = np.array([], dtype=np.float64)
y = np.array([], dtype=np.float64)
result = fnp.where(condition, x, y)
expected = np.where(condition, x, y)
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "where empty array should match numpy"
    );
    Ok(())
}

#[test]
fn where_numeric_condition() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([0, 1, 2, 0, -1])  # 0 is False, non-zero is True
x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 20, 30, 40, 50])
result = fnp.where(condition, x, y)
expected = np.where(condition, x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "where numeric condition should match numpy"
    );
    Ok(())
}

#[test]
fn where_preserves_numpy_dtype_promotion_matrix() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False, True, False])
cases = [
    (
        np.array([1, 2, 3, 4], dtype=np.int8),
        np.array([10, 20, 30, 40], dtype=np.int8),
    ),
    (
        np.array([1, 2, 3, 4], dtype=np.float32),
        np.array([10, 20, 30, 40], dtype=np.float32),
    ),
    (
        np.array([1, 2, 3, 4], dtype=np.int16),
        np.array([10, 20, 30, 40], dtype=np.uint16),
    ),
]
outcomes = []
for x, y in cases:
    result = fnp.where(condition, x, y)
    expected = np.where(condition, x, y)
    outcomes.append(
        np.array_equal(result, expected)
        and result.shape == expected.shape
        and result.dtype == expected.dtype
    )
print(all(outcomes))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "where should preserve NumPy dtype promotion for narrow numeric choices"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// where(condition) - index mode (like nonzero)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn where_index_mode_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([False, True, False, True, True])
result = fnp.where(condition)
expected = np.where(condition)
print(len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "where index mode 1d should match numpy"
    );
    Ok(())
}

#[test]
fn where_index_mode_scalar_error_surface_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def error_surface(call):
    try:
        call()
        return ("OK", "")
    except Exception as exc:
        return (type(exc).__name__, str(exc))

cases = [
    True,
    False,
    np.array(True),
    np.array(False),
    np.array(1),
    np.array(0),
]
print(all(error_surface(lambda cond=cond: fnp.where(cond)) == error_surface(lambda cond=cond: np.where(cond)) for cond in cases))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "where scalar-condition error surface should match numpy"
    );
    Ok(())
}

#[test]
fn where_index_mode_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([[True, False], [False, True]])
result = fnp.where(condition)
expected = np.where(condition)
print(len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "where index mode 2d should match numpy"
    );
    Ok(())
}

#[test]
fn where_index_mode_all_false() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([False, False, False])
result = fnp.where(condition)
expected = np.where(condition)
print(len(result) == len(expected) and all(len(r) == 0 for r in result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "where index mode all false should return empty indices"
    );
    Ok(())
}

#[test]
fn where_index_mode_all_true() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, True, True])
result = fnp.where(condition)
expected = np.where(condition)
print(len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "where index mode all true should return all indices"
    );
    Ok(())
}

#[test]
fn where_with_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False, True])
x = np.array([np.nan, 2.0, np.nan])
y = np.array([10.0, 20.0, 30.0])
result = fnp.where(condition, x, y)
expected = np.where(condition, x, y)
# NaN comparison needs special handling
match = all(
    (np.isnan(r) and np.isnan(e)) or (r == e)
    for r, e in zip(result.flat, expected.flat)
)
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "where with NaN should match numpy");
    Ok(())
}

#[test]
fn where_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False, True])
x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
y = np.array([4+4j, 5+5j, 6+6j], dtype=np.complex128)
fnp_result = fnp.where(condition, x, y)
np_result = np.where(condition, x, y)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "where complex should match numpy");
    Ok(())
}

#[test]
fn where_f32_parallel_large_bit_exact_matches_numpy() -> Result<(), String> {
    // Above the 1<<24-byte gate (n*4 bytes) the f32 arr/arr select runs the parallel
    // raw-slice path through a uint32 view. The blend is a verbatim bit-pattern copy,
    // so NaN/-0.0/+-inf must be selected byte-for-byte from whichever side wins.
    let script = fnp_script(
        r#"
n = (1 << 22) + 65
cond = (np.arange(n) % 2 == 0)
x = np.linspace(-2000.5, 2000.5, n, dtype=np.float32) * np.float32(2.0)
y = np.linspace(2000.5, -2000.5, n, dtype=np.float32) + np.float32(1.0)
x[0] = np.float32(np.nan)    # cond True  -> NaN from x
y[1] = np.float32(-0.0)      # cond False -> -0.0 from y
x[2] = np.float32(np.inf)    # cond True  -> +inf from x
y[3] = np.float32(-np.inf)   # cond False -> -inf from y
actual = fnp.where(cond, x, y)
expected = np.where(cond, x, y)
print(
    actual.dtype == expected.dtype
    and actual.shape == expected.shape
    and actual.tobytes() == expected.tobytes()
)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "large f32 where parallel path must be bit-identical to numpy"
    );
    Ok(())
}

#[test]
fn where_i32_parallel_large_bit_exact_matches_numpy() -> Result<(), String> {
    // 4-byte int select takes the same parallel raw-slice path as f32 (above the
    // 1<<24-byte gate). The blend is a verbatim copy, so extreme/min/max int values
    // must be selected exactly from whichever side the bool condition picks.
    let script = fnp_script(
        r#"
n = (1 << 22) + 65
cond = (np.arange(n) % 2 == 0)
x = (np.arange(n, dtype=np.int32) * np.int32(3)) - np.int32(7)
y = np.int32(11) - (np.arange(n, dtype=np.int32) * np.int32(2))
x[0] = np.iinfo(np.int32).max     # cond True  -> INT32_MAX from x
y[1] = np.iinfo(np.int32).min     # cond False -> INT32_MIN from y
x[2] = np.int32(0)
y[3] = np.iinfo(np.int32).max
actual = fnp.where(cond, x, y)
expected = np.where(cond, x, y)
print(
    actual.dtype == expected.dtype
    and actual.shape == expected.shape
    and actual.tobytes() == expected.tobytes()
)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "large i32 where parallel path must be bit-identical to numpy"
    );
    Ok(())
}

/// where(condition, x, y) either side of its per-dtype gate (numpy's call below each x dtype's
/// crossover, always for complex128 and for bool against a scalar): nine x dtypes x seven sizes
/// 1,023 .. 2^18, against a scalar, an array, and a broadcast 2-D form - dtype, shape, bytes and
/// warnings.
#[test]
fn where_matches_numpy_either_side_of_its_dtype_gate() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
rng = np.random.default_rng(112)
bad, cells = [], 0
def outcome(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            v = fn()
        except Exception as exc:
            return ("raise", type(exc).__name__, str(exc)[:80])
        a = np.asarray(v)
        return (type(v).__name__, a.dtype.str, a.shape, a.tobytes(), sorted({str(w.message)[:50] for w in caught}))
def check(label, call):
    global cells
    cells += 1
    if outcome(lambda: call(np)) != outcome(lambda: call(fnp)):
        bad.append(label)
for dt in ("float64", "float32", "int16", "uint8", "complex64", "complex128", "bool", "int64", "float16"):
    for n in (1023, 1024, 4095, 4096, 8192, 16384, 1 << 18):
        c = rng.random(n) < 0.5
        if dt == "bool":
            x = rng.random(n) < 0.5
        elif dt.startswith("complex"):
            x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(dt)
        else:
            x = (rng.standard_normal(n) * 50).astype(dt)
        y = x[::-1].copy()
        check(f"where {dt} {n} a,0", lambda m: m.where(c, x, 0))
        check(f"where {dt} {n} a,b", lambda m: m.where(c, x, y))
        check(f"where {dt} {n} 2-D", lambda m: m.where(c.reshape(1, -1), x.reshape(1, -1), y[:1]))
# dtypes the gate cannot classify by descriptor pointer: it reads their attributes instead.
for dt in (">f8", ">i4", "M8[ns]", "m8[s]", "U3"):
    for n in (1023, 1024, 4095, 4096, 8192, 16384, 1 << 18):
        c = rng.random(n) < 0.5
        if dt[0] in "Mm":
            x = rng.integers(-10**9, 10**9, n).astype(dt)
            x[5] = np.array("NaT").astype(dt)
        elif dt[0] == "U":
            x = rng.integers(0, 999, n).astype(dt)
        else:
            x = (rng.standard_normal(n) * 50).astype(dt)
        y = x[::-1].copy()
        if dt[0] not in "MmU":
            check(f"where {dt} {n} a,0", lambda m: m.where(c, x, 0))
        check(f"where {dt} {n} a,b", lambda m: m.where(c, x, y))
        check(f"where {dt} {n} 2-D", lambda m: m.where(c.reshape(1, -1), x.reshape(1, -1), y[:1]))
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (cells, bad) = result.trim().split_once(' ').unwrap_or(("0", &result));
    assert_eq!(cells, "273", "cell table drifted: {result}");
    assert_eq!(bad, "[]", "where must match numpy: {result}");
    Ok(())
}

/// `where(cond, x, y)` through the branch-free selects (`SelectBits`) and the in-place strided
/// float64 read (`f64_where_strided_1d`): 13 dtypes by every width the select dispatches on, x and
/// y contiguous, `[::2]`, `[::-1]`, a column, F-ordered and transposed 2-D, against random,
/// all-True and all-False masks and float64 / int scalar branches, at sizes either side of every
/// gate (complex128 now from 32,768) and the f64 pool floor. NaN payloads and -0.0 planted in the
/// operands make the byte comparison a verbatim-copy check; result type, dtype, shape, strides and
/// bytes compared. A select that swaps the branches, reads a stride wrong or normalises a NaN
/// fails it; the old branchy loops passed too, slower.
#[test]
fn where_select_layouts_and_widths_match_numpy_bytes() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(2026)
cells, bad = 0, []
def outcome(fn):
    try:
        r = fn()
        a = np.asarray(r)
        return ("ok", type(r).__name__, a.dtype.str, a.shape, a.strides, a.tobytes())
    except Exception as e:
        return ("raise", type(e).__name__, str(e))
def check(label, call):
    global cells
    cells += 1
    if outcome(lambda: call(fnp)) != outcome(lambda: call(np)):
        bad.append(label)
def values(dt, shape):
    if dt == "?":
        return rng.random(shape) < 0.5
    if dt[0] == "c":
        v = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dt)
    elif dt[0] == "M":
        return rng.integers(-10**9, 10**9, shape).astype(dt)
    else:
        v = (rng.standard_normal(shape) * 50).astype(dt)
    if dt in ("f8", "f4", "f2"):
        flat = v.reshape(-1)
        flat[1] = -0.0
        bits = {"f8": ("u8", 0x7FF8000000000123), "f4": ("u4", 0x7FC00123),
                "f2": ("u2", 0x7E01)}[dt]
        flat[2] = np.array([bits[1]], bits[0]).view(dt)[0]
    return v
dtypes = ("f8", "f4", "f2", "i8", "i4", "i2", "i1", "u8", "u1", "?", "c16", "c8", "M8[ns]")
for dt in dtypes:
    for n in (5, 4099, 16385, 40001, (1 << 21) + 3):
        if n > (1 << 17) and dt not in ("f8", "c16", "u1"):
            continue
        masks = {"random": rng.random(n) < 0.5, "all True": np.ones(n, bool),
                 "all False": np.zeros(n, bool)}
        twice = values(dt, 2 * n)
        tall = values(dt, (n, 3))
        sides = {
            "contig": (values(dt, n), values(dt, n)),
            "[::2]": (twice[::2], values(dt, 2 * n)[::2]),
            "[::-1]": (values(dt, n)[::-1], values(dt, n)[::-1]),
            "column": (tall[:, 1], values(dt, (n, 3))[:, 2]),
            "contig, [::2]": (values(dt, n), twice[1::2]),
        }
        for mask_label, c in masks.items():
            for side_label, (x, y) in sides.items():
                if mask_label != "random" and side_label != "contig":
                    continue
                check(f"{dt} {n} {mask_label} {side_label}", lambda m: m.where(c, x, y))
            x = sides["contig"][0]
            if dt[0] in "fiu":
                check(f"{dt} {n} {mask_label} x, 0.0", lambda m: m.where(c, x, 0.0))
                check(f"{dt} {n} {mask_label} 7, x", lambda m: m.where(c, 7, x))
                check(f"{dt} {n} {mask_label} [::2], 0.0", lambda m: m.where(c, twice[::2], 0.0))
    side = 160
    c2 = rng.random((side, side)) < 0.5
    a2, b2 = values(dt, (side, side)), values(dt, (side, side))
    check(f"{dt} F 2-D", lambda m: m.where(c2, np.asfortranarray(a2), np.asfortranarray(b2)))
    check(f"{dt} transposed", lambda m: m.where(c2, a2.T, b2.T))
    check(f"{dt} F cond", lambda m: m.where(np.asfortranarray(c2), a2, b2))
print(cells, bad[:12])
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (cells, bad) = result.trim().split_once(' ').unwrap_or(("0", &result));
    assert_eq!(cells, "766", "cell table drifted: {result}");
    assert_eq!(bad, "[]", "where must match numpy: {result}");
    Ok(())
}
