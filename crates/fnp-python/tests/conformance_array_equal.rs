//! Conformance tests for numpy.array_equal and numpy.array_equiv against NumPy oracle.

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
// array_equal basic
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn array_equal_identical_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = [1, 2, 3]
b = [1, 2, 3]
fnp_result = fnp.array_equal(a, b)
np_result = np.array_equal(a, b)
print(fnp_result == np_result == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal identical 1d mismatch");
    Ok(())
}

#[test]
fn array_equal_different_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = [1, 2, 3]
b = [1, 2, 4]
fnp_result = fnp.array_equal(a, b)
np_result = np.array_equal(a, b)
print(fnp_result == np_result == False)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal different values mismatch");
    Ok(())
}

#[test]
fn array_equal_different_shapes() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = [1, 2, 3]
b = [1, 2, 3, 4]
fnp_result = fnp.array_equal(a, b)
np_result = np.array_equal(a, b)
print(fnp_result == np_result == False)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal different shapes mismatch");
    Ok(())
}

#[test]
fn array_equal_2d_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[1, 2], [3, 4]])
fnp_result = fnp.array_equal(a, b)
np_result = np.array_equal(a, b)
print(fnp_result == np_result == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal 2d arrays mismatch");
    Ok(())
}

#[test]
fn array_equal_empty_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([])
b = np.array([])
fnp_result = fnp.array_equal(a, b)
np_result = np.array_equal(a, b)
print(fnp_result == np_result == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal empty arrays mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// array_equal with equal_nan
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn array_equal_nan_default() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0])
b = np.array([1.0, np.nan, 3.0])
fnp_result = fnp.array_equal(a, b)
np_result = np.array_equal(a, b)
print(fnp_result == np_result == False)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal nan default mismatch");
    Ok(())
}

#[test]
fn array_equal_nan_true() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0])
b = np.array([1.0, np.nan, 3.0])
fnp_result = fnp.array_equal(a, b, equal_nan=True)
np_result = np.array_equal(a, b, equal_nan=True)
print(fnp_result == np_result == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal nan=True mismatch");
    Ok(())
}

#[test]
fn array_equal_inf_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.inf, -np.inf])
b = np.array([1.0, np.inf, -np.inf])
fnp_result = fnp.array_equal(a, b)
np_result = np.array_equal(a, b)
print(fnp_result == np_result == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal inf values mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// array_equal dtypes
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn array_equal_int_float_same_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype=np.int32)
b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
fnp_result = fnp.array_equal(a, b)
np_result = np.array_equal(a, b)
print(fnp_result == np_result)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal int/float mismatch");
    Ok(())
}

#[test]
fn array_equal_bool_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([True, False, True])
b = np.array([True, False, True])
fnp_result = fnp.array_equal(a, b)
np_result = np.array_equal(a, b)
print(fnp_result == np_result == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal bool arrays mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// array_equiv
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn array_equiv_identical() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = [1, 2, 3]
b = [1, 2, 3]
fnp_result = fnp.array_equiv(a, b)
np_result = np.array_equiv(a, b)
print(fnp_result == np_result == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equiv identical mismatch");
    Ok(())
}

#[test]
fn array_equiv_broadcastable() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [1, 2]])
b = np.array([1, 2])
fnp_result = fnp.array_equiv(a, b)
np_result = np.array_equiv(a, b)
print(fnp_result == np_result == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equiv broadcastable mismatch");
    Ok(())
}

#[test]
fn array_equiv_not_broadcastable() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([1, 2])
fnp_result = fnp.array_equiv(a, b)
np_result = np.array_equiv(a, b)
print(fnp_result == np_result == False)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equiv not broadcastable mismatch");
    Ok(())
}

#[test]
fn array_equiv_scalar_broadcast() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([5, 5, 5])
b = 5
fnp_result = fnp.array_equiv(a, b)
np_result = np.array_equiv(a, b)
print(fnp_result == np_result == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equiv scalar broadcast mismatch");
    Ok(())
}

#[test]
fn array_equiv_different_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([1, 2, 4])
fnp_result = fnp.array_equiv(a, b)
np_result = np.array_equiv(a, b)
print(fnp_result == np_result == False)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equiv different values mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn array_equal_scalar_inputs() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.array_equal(5, 5)
np_result = np.array_equal(5, 5)
print(fnp_result == np_result == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal scalar inputs mismatch");
    Ok(())
}

#[test]
fn array_equal_nested_lists() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = [[1, 2], [3, 4]]
b = [[1, 2], [3, 4]]
fnp_result = fnp.array_equal(a, b)
np_result = np.array_equal(a, b)
print(fnp_result == np_result == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal nested lists mismatch");
    Ok(())
}

#[test]
fn array_equiv_empty_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([])
b = np.array([])
fnp_result = fnp.array_equiv(a, b)
np_result = np.array_equiv(a, b)
print(fnp_result == np_result == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equiv empty arrays mismatch");
    Ok(())
}

#[test]
fn array_equal_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j, 3+2j], dtype=np.complex128)
b = np.array([1+1j, 2-1j, 3+2j], dtype=np.complex128)
fnp_result = fnp.array_equal(a, b)
np_result = np.array_equal(a, b)
print(fnp_result == np_result)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal complex should match numpy");
    Ok(())
}

#[test]
fn array_equiv_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j], dtype=np.complex128)
b = np.array([1+1j, 2-1j], dtype=np.complex128)
fnp_result = fnp.array_equiv(a, b)
np_result = np.array_equiv(a, b)
print(fnp_result == np_result)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equiv complex should match numpy");
    Ok(())
}

/// array_equal / array_equiv across nine dtypes and four sizes: equal pairs, pairs whose LAST
/// element differs (a byte compare that stopped short, or compared the wrong length, fails these),
/// an array against a scalar, a 2-D against a 1-D view, NaN with and without equal_nan, mixed
/// dtypes, F / strided layouts, lists, 0-d, broadcasting, and ma.allequal. Integer / bool pairs
/// are a memcmp, every other declined ndarray is numpy's; the type and value of the verdict must
/// be numpy's.
#[test]
fn array_equal_and_equiv_match_numpy_across_dtypes_scalars_and_layouts() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.simplefilter("ignore")
rng = np.random.default_rng(104)
bad, cells = [], 0
def outcome(fn):
    try:
        v = fn()
    except Exception as exc:
        return ("raise", type(exc).__name__)
    return (type(v).__name__, bool(v))
def check(label, call):
    global cells
    cells += 1
    if outcome(lambda: call(np)) != outcome(lambda: call(fnp)):
        bad.append(label)
for dt in ("uint8", "int16", "int64", "float16", "float32", "float64", "complex64", "complex128", "bool"):
    for n in (1, 64, 4097, 1 << 18):
        if dt == "bool":
            a = rng.random(n) < 0.5
        elif np.dtype(dt).kind == "c":
            a = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(dt)
        else:
            a = (rng.standard_normal(n) * 50).astype(dt)
        b, c = a.copy(), a.copy()
        if n > 1:
            c[-1] = (not c[-1]) if dt == "bool" else c[-2] + 1
        for name in ("array_equal", "array_equiv"):
            check(f"{name} {dt} {n} same", lambda m, a=a, b=b, name=name: getattr(m, name)(a, b))
            check(f"{name} {dt} {n} last differs", lambda m, a=a, c=c, name=name: getattr(m, name)(a, c))
            check(f"{name} {dt} {n} vs scalar", lambda m, a=a, name=name: getattr(m, name)(a, a.flat[0]))
            check(f"{name} {dt} {n} 2d vs 1d", lambda m, a=a, name=name: getattr(m, name)(a.reshape(1, -1), a))
        if np.dtype(dt).kind in "fc":
            an = a.copy()
            an[0] = np.nan
            check(f"array_equal {dt} {n} nan", lambda m, an=an: m.array_equal(an, an.copy()))
            check(f"array_equal {dt} {n} nan equal_nan", lambda m, an=an: m.array_equal(an, an.copy(), equal_nan=True))
d = rng.standard_normal((32, 16))
for label, call in (("int8 vs int64", lambda m: m.array_equal(np.arange(5, dtype=np.int8), np.arange(5))),
                    ("F vs C", lambda m: m.array_equal(np.asfortranarray(d), d)),
                    ("strided", lambda m: m.array_equal(d[:, ::2], d[:, ::2].copy())),
                    ("lists", lambda m: m.array_equal([1, 2], [1, 2])),
                    ("list vs array", lambda m: m.array_equal([1, 2], np.array([1, 2]))),
                    ("0-d vs scalar", lambda m: m.array_equal(np.array(3), 3)),
                    ("array vs None", lambda m: m.array_equal(d, None)),
                    ("str vs array", lambda m: m.array_equal("x", d)),
                    ("array vs Python int", lambda m: m.array_equal(d[0], 1)),
                    ("equiv broadcast", lambda m: m.array_equiv(np.ones((3, 1)), np.ones(3))),
                    ("equiv int broadcast", lambda m: m.array_equiv(np.ones((3, 4), np.int16), np.ones(4, np.int16))),
                    ("ma.allequal same", lambda m: m.ma.allequal(d.ravel(), d.ravel().copy())),
                    ("ma.allequal differs", lambda m: m.ma.allequal(d.ravel(), d.ravel() + (np.arange(d.size) == d.size - 1))),
                    ("ma.allequal nan", lambda m: m.ma.allequal(np.r_[np.nan, d.ravel()], np.r_[np.nan, d.ravel()]))):
    check(label, call)
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (cells, bad) = result.trim().split_once(' ').unwrap_or(("0", &result));
    assert_eq!(cells, "342", "cell table drifted: {result}");
    assert_eq!(
        bad, "[]",
        "array_equal / array_equiv must match numpy: {result}"
    );
    Ok(())
}
