//! Conformance tests for native np.zeros and np.ones implementations.

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
// zeros
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn zeros_1d_default_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.zeros(5)
np_result = np.zeros(5)
print(np.array_equal(fnp_result, np_result) and fnp_result.dtype == np_result.dtype)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "zeros 1d default dtype mismatch");
    Ok(())
}

#[test]
fn zeros_2d_tuple_shape() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.zeros((3, 4))
np_result = np.zeros((3, 4))
print(np.array_equal(fnp_result, np_result) and fnp_result.shape == (3, 4))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "zeros 2d tuple shape mismatch");
    Ok(())
}

#[test]
fn zeros_with_int_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.zeros((2, 3), dtype=np.int32)
np_result = np.zeros((2, 3), dtype=np.int32)
print(np.array_equal(fnp_result, np_result) and fnp_result.dtype == np.int32)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "zeros with int dtype mismatch");
    Ok(())
}

#[test]
fn zeros_with_bool_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.zeros(4, dtype=bool)
np_result = np.zeros(4, dtype=bool)
print(np.array_equal(fnp_result, np_result) and fnp_result.dtype == np.bool_)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "zeros with bool dtype mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ones
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ones_1d_default_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.ones(5)
np_result = np.ones(5)
print(np.array_equal(fnp_result, np_result) and fnp_result.dtype == np_result.dtype)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "ones 1d default dtype mismatch");
    Ok(())
}

#[test]
fn ones_2d_tuple_shape() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.ones((3, 4))
np_result = np.ones((3, 4))
print(np.array_equal(fnp_result, np_result) and fnp_result.shape == (3, 4))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "ones 2d tuple shape mismatch");
    Ok(())
}

#[test]
fn ones_with_int_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.ones((2, 3), dtype=np.int64)
np_result = np.ones((2, 3), dtype=np.int64)
print(np.array_equal(fnp_result, np_result) and fnp_result.dtype == np.int64)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "ones with int dtype mismatch");
    Ok(())
}

#[test]
fn ones_with_float32_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.ones(10, dtype=np.float32)
np_result = np.ones(10, dtype=np.float32)
print(np.array_equal(fnp_result, np_result) and fnp_result.dtype == np.float32)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "ones with float32 dtype mismatch");
    Ok(())
}

#[test]
fn ones_3d_shape() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.ones((2, 3, 4))
np_result = np.ones((2, 3, 4))
print(np.array_equal(fnp_result, np_result) and fnp_result.shape == (2, 3, 4))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "ones 3d shape mismatch");
    Ok(())
}

#[test]
fn zeros_empty_shape() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.zeros((0,))
np_result = np.zeros((0,))
print(np.array_equal(fnp_result, np_result) and fnp_result.shape == (0,))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "zeros empty shape mismatch");
    Ok(())
}

#[test]
fn ones_matches_numpy_across_dtypes_shapes_orders_and_errors() -> Result<(), String> {
    // A small `ones` is filled natively - numpy's own `empty(shape, dtype)` filled with numpy's
    // own one for the builtin numeric dtypes - and everything else stays numpy's. Every
    // observable must match: type, dtype (incl. byte order), shape, strides, flags, the bytes
    // themselves (float16 / complex ones, a bool True), and the errors of bad shapes and dtypes.
    let script = fnp_script(
        r#"
def describe(r):
    return (type(r).__name__, r.dtype.str, r.dtype.char, r.shape, r.strides, bool(r.flags.owndata),
            bool(r.flags.writeable), bool(r.flags.c_contiguous), bool(r.flags.f_contiguous),
            r.tobytes() if r.dtype.kind not in "OUSV" else repr(r.tolist()))
def outcome(fn, *args, **kwargs):
    try:
        return ("ok", describe(fn(*args, **kwargs)))
    except Exception as exc:
        return ("raise", type(exc).__name__, str(exc))
dtypes = [None, float, int, bool, complex, "?", "b", "B", "h", "H", "i", "I", "l", "L", "q", "Q",
          "e", "f", "d", "F", "D", "g", ">f8", "<i4", "U3", "S2", object, "M8[D]", "m8[s]",
          [("a", "i4"), ("b", "f8")], np.float64, np.dtype("int16")]
shapes = [0, 5, (2, 3), (), (0, 4), (3, 1, 2), 4095, 10000, [2, 2], np.int64(3)]
cells, bad = 0, []
for dtype in dtypes:
    for shape in shapes:
        for order in (None, "C", "F"):
            kwargs = {} if dtype is None else {"dtype": dtype}
            if order is not None:
                kwargs["order"] = order
            cells += 1
            if outcome(fnp.ones, shape, **kwargs) != outcome(np.ones, shape, **kwargs):
                bad.append(f"ones({shape!r}, {kwargs})")
for args, kwargs in [((-1,), {}), (((2, -3),), {}), (("a",), {}), ((2.5,), {}), ((3,), {"dtype": "not a dtype"}),
                     ((3,), {"order": "Z"}), ((3,), {"like": np.empty(0)})]:
    cells += 1
    if outcome(fnp.ones, *args, **kwargs) != outcome(np.ones, *args, **kwargs):
        bad.append(f"ones{args} {kwargs}")
print(cells, bad[:8])
"#
        .into(),
    );
    let out = numpy_oracle(&script)?;
    let (cells, bad) = out.trim().split_once(' ').unwrap_or(("0", &out));
    assert_eq!(bad, "[]", "ones must match numpy: {out}");
    assert_eq!(cells, "967", "cell table drifted: {out}");
    Ok(())
}

/// A small `full` without `dtype=` of a Python float / int (int64 range) / bool fill is numpy's
/// own `empty` filled with the value's bytes (`try_native_small_full`). Every observable must
/// stay numpy's (type, dtype and its char, shape, strides, flags, bytes incl. NaN and -0.0,
/// warnings, errors) across the fills the route declines (out-of-range ints, numpy scalars,
/// complex, strings, None, lists, 0-d arrays), shapes (empty, 0-d, lists, numpy ints, negative,
/// float, past the size cap) and keywords (dtype=None / "f4", order C / F / K, like=None).
#[test]
fn small_full_matches_numpy_across_fills_shapes_and_orders() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
def outcome(fn, *a, **k):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            r = fn(*a, **k); x = np.asarray(r)
            res = ("ok", type(r).__name__, x.dtype.str, x.dtype.char, x.shape, x.strides,
                   bool(x.flags.c_contiguous), bool(x.flags.f_contiguous),
                   x.tobytes() if x.dtype != object else repr(x.tolist()))
        except Exception as e:
            res = ("raise", type(e).__name__, str(e))
    return res + (tuple(sorted((x.category.__name__, str(x.message)) for x in w)),)
fills = [3.5, -0.0, float("nan"), float("inf"), 1e308, 7, -7, 0, 2**62, 2**63, -2**63, 2**70, True, False,
         np.float64(2.0), np.float32(2.0), np.int64(3), np.int32(3), 1 + 2j, "x", None, [1, 2], np.array(1.5),
         np.bool_(True)]
shapes = [0, 1, 7, 64, 4096, 4097, 10000, (2, 3), (0, 4), (), (3, 1, 2), [2, 2], np.int64(3), -1, (2, -3), 2.5,
          (4096, 2)]
cells, bad = 0, []
for fill in fills:
    for shape in shapes:
        for kw in ({}, {"dtype": None}, {"order": "C"}, {"order": "F"}, {"order": "K"}, {"dtype": "f4"},
                   {"like": None}):
            cells += 1
            if outcome(fnp.full, shape, fill, **kw) != outcome(np.full, shape, fill, **kw):
                bad.append((repr(fill), repr(shape), kw))
print(cells, bad[:8])
"#
        .into(),
    );
    assert_eq!(numpy_oracle(&script)?, "2856 []");
    Ok(())
}
