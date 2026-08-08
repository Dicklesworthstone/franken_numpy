//! Conformance tests for numpy.sign and numpy.signbit against NumPy oracle.
//!
//! Tests sign and signbit functions.

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

fn fnp_script(body: String) -> String {
    let library_name = format!(
        "{}fnp_python{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    );
    let module_path = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.join(&library_name)))
        .unwrap_or_else(|| library_name.into());
    let module_literal = format!("{module_path:?}");
    format!(
        "import importlib.util\n\
         import numpy as np\n\
         spec = importlib.util.spec_from_file_location('fnp_python', {module_literal})\n\
         fnp = importlib.util.module_from_spec(spec)\n\
         spec.loader.exec_module(fnp)\n\
         {body}"
    )
}

// The signbit surface scripts below close with a provenance line (host + numpy
// version) after the parity verdict, so a red run names the environment it was
// measured in instead of leaving the reader to guess which worker ran it.
fn split_verdict(output: &str) -> (&str, &str) {
    let mut lines = output.trim().lines().rev();
    let provenance = lines.next().unwrap_or("").trim();
    let verdict = lines.next().unwrap_or("").trim();
    (verdict, provenance)
}

#[test]
fn sign_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-5, -1, 0, 1, 5])
result = fnp.sign(x)
expected = np.sign(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sign basic should match numpy");
    Ok(())
}

#[test]
fn sign_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
result = fnp.sign(x)
expected = np.sign(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sign float should match numpy");
    Ok(())
}

#[test]
fn signbit_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1.0, 0.0, 1.0, -0.0])
result = fnp.signbit(x)
expected = np.signbit(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "signbit basic should match numpy");
    Ok(())
}

#[test]
fn signbit_with_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.nan, -np.nan, np.inf, -np.inf])
result = fnp.signbit(x)
expected = np.signbit(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "signbit with nan/inf should match numpy"
    );
    Ok(())
}

#[test]
fn sign_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(-5.0)
fnp_result = fnp.sign(x)
np_result = np.sign(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "sign scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn signbit_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(-5.0)
fnp_result = fnp.signbit(x)
np_result = np.signbit(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "signbit scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn sign_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+1j, -2+2j, 0+0j, 3-4j], dtype=np.complex128)
fnp_result = fnp.sign(z)
np_result = np.sign(z)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sign complex should match numpy");
    Ok(())
}

#[test]
fn sign_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0])
fnp_result = fnp.sign(x)
np_result = np.sign(x)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sign special values should match numpy"
    );
    Ok(())
}

#[test]
fn signbit_negative_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.0, -0.0])
fnp_result = fnp.signbit(x)
np_result = np.signbit(x)
# signbit(-0.0) should be True, signbit(0.0) should be False
print(np.array_equal(fnp_result, np_result) and fnp_result[0] == False and fnp_result[1] == True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "signbit negative zero should match numpy"
    );
    Ok(())
}

#[test]
fn sign_integer_dtypes() -> Result<(), String> {
    let script = fnp_script(
        r#"
tests_pass = True
for dtype in [np.int8, np.int16, np.int32, np.int64]:
    x = np.array([-128, -1, 0, 1, 127], dtype=dtype)
    fnp_result = fnp.sign(x)
    np_result = np.sign(x)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sign integer dtypes should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge case tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sign_empty_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([], dtype=np.float64)
fnp_result = fnp.sign(x)
np_result = np.sign(x)
print(np.array_equal(fnp_result, np_result) and fnp_result.shape == np_result.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sign empty array should match numpy");
    Ok(())
}

#[test]
fn signbit_empty_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([], dtype=np.float64)
fnp_result = fnp.signbit(x)
np_result = np.signbit(x)
print(np.array_equal(fnp_result, np_result) and fnp_result.shape == np_result.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "signbit empty array should match numpy"
    );
    Ok(())
}

#[test]
fn sign_single_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
tests_pass = True
for val in [-5.0, -0.0, 0.0, 5.0]:
    x = np.array([val])
    fnp_result = fnp.sign(x)
    np_result = np.sign(x)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sign single element should match numpy"
    );
    Ok(())
}

#[test]
fn sign_unsigned_integers() -> Result<(), String> {
    let script = fnp_script(
        r#"
tests_pass = True
for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
    x = np.array([0, 1, 127, 255], dtype=dtype)
    fnp_result = fnp.sign(x)
    np_result = np.sign(x)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sign unsigned integers should match numpy"
    );
    Ok(())
}

#[test]
fn sign_subnormal_numbers() -> Result<(), String> {
    let script = fnp_script(
        r#"
import sys
tiny = sys.float_info.min
subnormal = tiny / 2.0
x = np.array([subnormal, -subnormal, tiny, -tiny, 0.0])
fnp_result = fnp.sign(x)
np_result = np.sign(x)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sign subnormal numbers should match numpy"
    );
    Ok(())
}

#[test]
fn signbit_subnormal_numbers() -> Result<(), String> {
    let script = fnp_script(
        r#"
import sys
tiny = sys.float_info.min
subnormal = tiny / 2.0
x = np.array([subnormal, -subnormal, tiny, -tiny])
fnp_result = fnp.signbit(x)
np_result = np.signbit(x)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "signbit subnormal numbers should match numpy"
    );
    Ok(())
}

// fnp.signbit keeps a native one-argument fast path and delegates every other
// arity/keyword surface to numpy. This walks that boundary: out= as keyword,
// positional, tuple, None, a strided view, a dtype numpy is allowed to cast
// into, where= partial writes, and the two shapes that must raise. Each case
// compares fnp's outcome against numpy's outcome computed in the same
// interpreter, so nothing is pinned to a build.
#[test]
fn signbit_out_keyword_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import platform

def keyword_bool(fn):
    x = np.array([-2.0, -0.0, 0.0, 3.5], dtype=np.float64)
    out = np.ones(4, dtype=bool)
    result = fn(x, out=out)
    return (result is out, out.dtype.str, tuple(out.shape), out.tolist())

def positional_bool(fn):
    x = np.array([-2.0, -0.0, 0.0, 3.5], dtype=np.float64)
    out = np.ones(4, dtype=bool)
    result = fn(x, out)
    return (result is out, out.dtype.str, tuple(out.shape), out.tolist())

def tuple_out(fn):
    x = np.array([-2.0, -0.0, 0.0, 3.5], dtype=np.float64)
    out = np.ones(4, dtype=bool)
    result = fn(x, out=(out,))
    return (result is out, out.tolist())

def none_out(fn):
    x = np.array([-2.0, -0.0, 0.0, 3.5], dtype=np.float64)
    result = fn(x, out=None)
    return (str(result.dtype), result.tolist())

def float_out(fn):
    # numpy casts the bool result into a float out buffer; it must not refuse.
    x = np.array([-2.0, -0.0, 0.0, 3.5], dtype=np.float64)
    out = np.full(4, 9.0, dtype=np.float64)
    result = fn(x, out=out)
    return (result is out, out.dtype.str, out.tolist())

def strided_out(fn):
    # Writing through a non-contiguous view must leave the untouched slots alone.
    x = np.array([-2.0, -0.0, 0.0, 3.5], dtype=np.float64)
    big = np.zeros(8, dtype=bool)
    fn(x, out=big[::2])
    return (big.tolist(),)

def where_partial(fn):
    x = np.array([-2.0, -0.0, 0.0, 3.5], dtype=np.float64)
    out = np.zeros(4, dtype=bool)
    result = fn(x, out=out, where=np.array([True, False, True, False]))
    return (result is out, out.tolist())

def int_input_out(fn):
    x = np.array([-2, -1, 0, 3], dtype=np.int32)
    out = np.ones(4, dtype=bool)
    result = fn(x, out=out)
    return (result is out, out.tolist())

def float16_input_out(fn):
    x = np.array([-2.0, -0.0, 0.0, 3.5], dtype=np.float16)
    out = np.ones(4, dtype=bool)
    result = fn(x, out=out)
    return (result is out, out.tolist())

def two_dimensional_out(fn):
    x = np.array([[-1.0, 2.0], [0.0, -0.0]], dtype=np.float64)
    out = np.ones((2, 2), dtype=bool)
    result = fn(x, out=out)
    return (result is out, out.tolist())

def bad_out_shape(fn):
    x = np.array([-2.0, -0.0, 0.0, 3.5], dtype=np.float64)
    out = np.ones(2, dtype=bool)
    result = fn(x, out=out)
    return (result is out, out.tolist())

def complex_input_out(fn):
    x = np.array([1 + 2j, -1 - 2j], dtype=np.complex128)
    out = np.ones(2, dtype=bool)
    result = fn(x, out=out)
    return (result is out, out.tolist())

def extra_positional(fn):
    x = np.array([-2.0, 3.5], dtype=np.float64)
    out = np.ones(2, dtype=bool)
    result = fn(x, out, None)
    return (result is out, out.tolist())

cases = [
    ("keyword bool out", keyword_bool),
    ("positional bool out", positional_bool),
    ("tuple out", tuple_out),
    ("out=None", none_out),
    ("float out buffer", float_out),
    ("strided out view", strided_out),
    ("where= partial write", where_partial),
    ("int32 input with out", int_input_out),
    ("float16 input with out", float16_input_out),
    ("2-D out", two_dimensional_out),
    ("bad out shape", bad_out_shape),
    ("complex input with out", complex_input_out),
    ("extra positional argument", extra_positional),
]

def outcome(fn, case):
    try:
        return ("ok",) + case(fn)
    except Exception as exc:
        return ("err", type(exc).__name__)

ok = True
for label, case in cases:
    actual = outcome(fnp.signbit, case)
    expected = outcome(np.signbit, case)
    if actual != expected:
        print(label)
        print(actual)
        print(expected)
        ok = False
print(ok)
print("oracle", platform.node(), np.__version__)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (verdict, provenance) = split_verdict(&result);
    assert_eq!(
        verdict, "True",
        "signbit out keyword surfaces should match numpy ({provenance}): {result}"
    );
    Ok(())
}

// The one-argument fast path is not one path: it branches on dtype (f64 / f32 /
// f16 bit test / constant-False for unsigned and bool / delegate for signed int)
// and on C-contiguity. This walks every branch, since a value defect in one of
// them is invisible to the float64-contiguous tests above.
#[test]
fn signbit_native_dtype_and_layout_grid_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import platform

def described(value):
    array = np.asarray(value)
    return (str(array.dtype), tuple(array.shape), array.tolist())

cases = []
for dtype in [np.float64, np.float32, np.float16]:
    cases.append((f"{np.dtype(dtype).name} values",
                  np.array([-2.0, -0.0, 0.0, 3.5, np.inf, -np.inf, np.nan], dtype=dtype)))
for dtype in [np.int8, np.int16, np.int32, np.int64]:
    cases.append((f"{np.dtype(dtype).name} values", np.array([-3, -1, 0, 1, 7], dtype=dtype)))
for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
    cases.append((f"{np.dtype(dtype).name} values", np.array([0, 1, 7], dtype=dtype)))
cases.append(("bool values", np.array([True, False, True])))
cases.append(("float64 2-D", np.array([[-1.0, 2.0], [0.0, -0.0]], dtype=np.float64)))
cases.append(("float64 transposed view", np.array([[-1.0, 2.0], [0.0, -0.0]], dtype=np.float64).T))
cases.append(("float32 strided view", np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32)[::2]))
cases.append(("int32 strided view", np.array([-1, 2, -3, 4], dtype=np.int32)[::2]))
cases.append(("float64 empty", np.array([], dtype=np.float64)))
cases.append(("uint8 empty", np.array([], dtype=np.uint8)))
cases.append(("python float scalar", -0.0))
cases.append(("python int scalar", -7))
cases.append(("nested list", [[-1.0, 0.0], [2.0, -3.0]]))

ok = True
for label, x in cases:
    try:
        actual = ("ok",) + described(fnp.signbit(x))
    except Exception as exc:
        actual = ("err", type(exc).__name__)
    try:
        expected = ("ok",) + described(np.signbit(x))
    except Exception as exc:
        expected = ("err", type(exc).__name__)
    if actual != expected:
        print(label)
        print(actual)
        print(expected)
        ok = False
print(ok)
print("oracle", platform.node(), np.__version__)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (verdict, provenance) = split_verdict(&result);
    assert_eq!(
        verdict, "True",
        "signbit dtype/layout grid should match numpy ({provenance}): {result}"
    );
    Ok(())
}
