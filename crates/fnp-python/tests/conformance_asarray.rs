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

/// `asarray` of an ndarray SUBCLASS is a base-class VIEW in numpy (memmap: `asarray(fp).base is
/// fp`), and the requested dtype is matched by OBJECT: an equal-named dtype carrying metadata is
/// a view with it, a byte-swapped '>i4' asked for as int32 is a native-order cast. fnp rebuilt
/// subclasses natively - a COPY, so writes never reached the source - and matched dtypes by name,
/// returning the '>i4' array itself (numpy's test_memmap::test_view,
/// test_array_coercion::test_dtype_identity). 4 of these 12 cells failed on a8d9a337.
#[test]
fn asarray_views_subclasses_and_matches_dtypes_by_object() -> Result<(), String> {
    let script = fnp_script(
        r#"
import tempfile
tmp = tempfile.NamedTemporaryFile()
fp = np.memmap(tmp, dtype="f4", shape=(3, 4), mode="w+")
meta = np.dtype("i", metadata={"spam": True})
def outcome(f):
    try:
        return ("ok", repr(f()))
    except Exception as ex:
        return (type(ex).__name__, str(ex)[:80])
cases = {
    "same dtype object is a": lambda m: (lambda a: m.asarray(a, dtype="i") is a)(np.array([1, 2], dtype="i")),
    "metadata dtype is a view": lambda m: (lambda a: (lambda r: (r is a, r.base is a, r.dtype.metadata))(m.asarray(a, dtype=meta)))(np.array([1, 2], dtype="i")),
    ">i4 as int32 casts": lambda m: (lambda a: (lambda r: (r is a, r.dtype.str, r.tolist()))(m.asarray(a, dtype=np.int32)))(np.arange(3, dtype=">i4")),
    "memmap is viewed": lambda m: (lambda r: (type(r).__name__, r.base is fp, np.shares_memory(r, fp)))(m.asarray(fp)),
    "subclass writes through": lambda m: (lambda s: (m.asarray(s).__setitem__((0, 0), 9.0), s[0, 0]))(np.arange(4.0).view(np.matrix)),
    "subclass view type": lambda m: type(m.asarray(np.arange(4.0).view(np.matrix))).__name__,
    "asanyarray keeps subclass": lambda m: (lambda s: m.asanyarray(s) is s)(np.arange(4.0).view(np.matrix)),
    "copy=True drops metadata": lambda m: m.asarray(np.arange(2, dtype="i"), dtype=meta, copy=True).dtype.metadata,
    "list takes metadata dtype": lambda m: m.asarray([1, 2], dtype=meta).dtype.metadata,
    "exact ndarray is a": lambda m: (lambda a: m.asarray(a) is a)(np.arange(3.0)),
    "order F copies": lambda m: (lambda a: (lambda r: (r is a, r.flags.f_contiguous))(m.asarray(a, order="F")))(np.arange(6.0).reshape(2, 3)),
    "copy=False subclass": lambda m: (lambda r: np.shares_memory(r, fp))(m.asarray(fp, copy=False)),
}
bad = [f"{k}: fnp={outcome(lambda: f(fnp))} numpy={outcome(lambda: f(np))}" for k, f in cases.items()
       if outcome(lambda: f(fnp)) != outcome(lambda: f(np))]
print(len(cases), bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last, "12 []",
        "asarray must view and match dtypes as numpy does: {result}"
    );
    Ok(())
}

/// `fromstring` / `fromfile` with a non-whitespace `sep`: numpy 2.x raises "string or file could
/// not be read to its end due to unmatched data" for an empty token ("1xx2", "x1x2", "x") and reads
/// a separator followed only by whitespace its own way ("1x2x\n"); fnp's tokenizer dropped empty
/// tokens and answered [1, 2] (numpy's test_longdouble::test_fromstring_empty / _missing). 14 of
/// these 44 cells failed on a8d9a337.
#[test]
fn fromstring_and_fromfile_refuse_unmatched_separators_like_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import tempfile, warnings
scratch = tempfile.NamedTemporaryFile(mode="w+")  # removed when closed
path = scratch.name
def outcome(f):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return ("ok", f().tolist())
        except Exception as ex:
            return (type(ex).__name__, str(ex)[:60])
def from_file(m, s, sep):
    with open(path, "w") as out:
        out.write(s)
    return m.fromfile(path, sep=sep)
texts = ["1x2x", "x1x2", " 1 x 2 ", "1x2x\n", "1xx2", "xxxxx", "", "1", "1x", "x", "1, 2", "1,,2",
         "1 , 2 ,", " , 1", "1x2xabc", "1x 2x3", "1\nx2", "1 2 3", "1  2\n3 ", "1 2 a", "1x2x3", "1,2,3,"]
bad, cells = [], 0
for s in texts:
    sep = "," if "," in s else ("x" if "x" in s or s in ("", "1") else " ")
    for label, f in (("fromstring", lambda m: m.fromstring(s, sep=sep)), ("fromfile", lambda m: from_file(m, s, sep))):
        cells += 1
        ours, theirs = outcome(lambda: f(fnp)), outcome(lambda: f(np))
        if ours != theirs:
            bad.append(f"{label} {s!r}: fnp={ours} numpy={theirs}")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last, "44 []",
        "fromstring/fromfile separators must match numpy: {result}"
    );
    Ok(())
}
