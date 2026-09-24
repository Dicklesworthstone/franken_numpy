//! Conformance tests for numpy asarray and asanyarray against NumPy oracle.
//!
//! Tests asarray, asanyarray, fromstring, frombuffer.

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
// asarray
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn asarray_from_list() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.asarray([1, 2, 3])
expected = np.asarray([1, 2, 3])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "asarray from list should match numpy"
    );
    Ok(())
}

#[test]
fn asarray_from_nested_list() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.asarray([[1, 2], [3, 4]])
expected = np.asarray([[1, 2], [3, 4]])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "asarray from nested list should match numpy"
    );
    Ok(())
}

#[test]
fn asarray_with_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.asarray([1, 2, 3], dtype='float64')
expected = np.asarray([1, 2, 3], dtype='float64')
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "asarray with dtype should match numpy"
    );
    Ok(())
}

#[test]
fn asarray_from_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.asarray(42)
expected = np.asarray(42)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "asarray from scalar should match numpy"
    );
    Ok(())
}

#[test]
fn asarray_from_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.asarray(a)
expected = np.asarray(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "asarray from array should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// asanyarray
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn asanyarray_from_list() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.asanyarray([1, 2, 3])
expected = np.asanyarray([1, 2, 3])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "asanyarray from list should match numpy"
    );
    Ok(())
}

#[test]
fn asanyarray_with_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.asanyarray([1, 2, 3], dtype='float32')
expected = np.asanyarray([1, 2, 3], dtype='float32')
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "asanyarray with dtype should match numpy"
    );
    Ok(())
}

#[test]
fn asanyarray_from_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.asanyarray(a)
expected = np.asanyarray(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "asanyarray from array should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// fromstring
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fromstring_with_sep() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.fromstring('1 2 3 4', sep=' ')
expected = np.fromstring('1 2 3 4', sep=' ')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "fromstring with sep should match numpy"
    );
    Ok(())
}

#[test]
fn fromstring_with_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.fromstring('1 2 3 4', sep=' ', dtype='int32')
expected = np.fromstring('1 2 3 4', sep=' ', dtype='int32')
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "fromstring with dtype should match numpy"
    );
    Ok(())
}

#[test]
fn fromstring_comma_sep() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.fromstring('1,2,3,4', sep=',')
expected = np.fromstring('1,2,3,4', sep=',')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "fromstring comma sep should match numpy"
    );
    Ok(())
}

#[test]
fn fromstring_with_count() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.fromstring('1 2 3 4 5', sep=' ', count=3)
expected = np.fromstring('1 2 3 4 5', sep=' ', count=3)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "fromstring with count should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn asarray_asanyarray_equivalence_for_lists() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = [[1, 2, 3], [4, 5, 6]]
asarray_result = fnp.asarray(data)
asanyarray_result = fnp.asanyarray(data)
print(np.array_equal(asarray_result, asanyarray_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "asarray and asanyarray should be equivalent for lists"
    );
    Ok(())
}

#[test]
fn asarray_preserves_existing_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.asarray(a)
# For same dtype, asarray should return the input
print(result.shape == a.shape and result.dtype == a.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "asarray should preserve existing array"
    );
    Ok(())
}

#[test]
fn asarray_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = [1+1j, 2-1j, 3+2j]
fnp_result = fnp.asarray(a, dtype=np.complex128)
np_result = np.asarray(a, dtype=np.complex128)
print(np.array_equal(fnp_result, np_result) and fnp_result.dtype == np_result.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "asarray complex should match numpy");
    Ok(())
}

/// numpy's own TestArrayConstruction::test_array_signature, over fnp's constructors: every one
/// reports a signature with its array argument first (`object` for `array`, `a` otherwise) as a
/// required positional-or-keyword parameter, a `dtype` parameter, and at least 3 parameters.
/// `fnp.array` is a `*args, **kwargs` passthrough and reported just those two.
#[test]
fn constructor_signatures_satisfy_numpys_array_signature_test() -> Result<(), String> {
    let script = fnp_script(
        r#"
import inspect
bad = []
for name in ("array", "asarray", "asanyarray", "ascontiguousarray", "asfortranarray"):
    try:
        sig = inspect.signature(getattr(fnp, name))
    except (TypeError, ValueError) as exc:
        bad.append((name, type(exc).__name__))
        continue
    arg0 = "object" if name == "array" else "a"
    params = sig.parameters
    if not (len(params) >= 3 and arg0 in params and "dtype" in params
            and params[arg0].default is inspect.Parameter.empty
            and params[arg0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD):
        bad.append((name, str(sig)))
print(bad if bad else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "True",
        "constructor signatures must satisfy numpy's test: {result}"
    );
    Ok(())
}

/// `asarray`/`asanyarray` VIEW an operand that exports the buffer protocol or an array
/// interface, as numpy does. fnp extracted every non-ndarray operand into a fresh array, so
/// `asarray(memoryview(a))` / `bytearray` / `array.array` / ctypes / `__array_interface__`
/// objects came back as COPIES: writes through the result never reached the source (numpy's
/// own TestNewBufferProtocol::_check_roundtrip). Lists and tuples are the copy controls.
#[test]
fn asarray_views_buffer_exporters_like_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import array, ctypes
class ArrayInterface:
    def __init__(self):
        self.base = np.arange(3.0)
        self.__array_interface__ = self.base.__array_interface__
class ReadOnlyStruct:
    # numpy's own TestFlags::test_readonly_flag_protocols: a view keeps the read-only flag.
    def __init__(self):
        self.base = np.arange(10)
        self.base.flags.writeable = False
        self.__array_struct__ = self.base.__array_struct__
def sources():
    return [("memoryview", memoryview(np.arange(4.0))), ("bytearray", bytearray(b"abcd")),
            ("array.array", array.array("d", [1.0, 2.0])), ("ctypes", (ctypes.c_double * 3)(1, 2, 3)),
            ("array_interface", ArrayInterface()), ("readonly_array_struct", ReadOnlyStruct()),
            ("list", [1.0, 2.0]), ("tuple", (1, 2, 3))]
def outcome(m, name, src):
    y = getattr(m, name)(src)
    shares = False if isinstance(src, (list, tuple)) else np.shares_memory(y, np.asarray(src))
    return y.flags.owndata, y.flags.writeable, y.dtype.str, y.shape, shares
bad = []
for name in ("asarray", "asanyarray"):
    for (label, src), (_, src2) in zip(sources(), sources()):
        if outcome(fnp, name, src) != outcome(np, name, src2):
            bad.append((name, label))
print(bad if bad else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "True",
        "asarray must view buffer exporters like numpy: {result}"
    );
    Ok(())
}
