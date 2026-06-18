//! Conformance tests for numpy.compress, choose, diagonal against NumPy oracle.
//!
//! Tests the native Rust implementations against NumPy.

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

// ─────────────────────────────────────────────────────────────────────────────
// compress
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compress_python_container_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def compress_outcome(fn, condition, a, **kwargs):
    try:
        result = fn(condition, a, **kwargs)
        arr = np.asarray(result)
        return ("ok", type(result).__name__, str(arr.dtype), tuple(arr.shape), arr.tolist())
    except Exception as exc:
        return ("err", type(exc).__name__, str(exc))

cases = [
    ("empty condition", lambda: ([], [1, 2, 3], {})),
    ("list condition list payload", lambda: ([True, False, True], [10, 20, 30], {})),
    ("integer condition truthiness", lambda: ([1, 0, 2], [10, 20, 30], {})),
    (
        "bool ndarray condition",
        lambda: (np.array([False, True, True], dtype=np.bool_), np.array([1, 2, 3], dtype=np.int16), {}),
    ),
    ("tuple condition tuple payload", lambda: ((False, True, True), (1.5, 2.5, 3.5), {})),
    ("shorter condition truncates", lambda: ([True, False], [1, 2, 3, 4], {})),
    ("nested list axis zero", lambda: ([True, False, True], [[1, 2], [3, 4], [5, 6]], {"axis": 0})),
    ("nested list axis one", lambda: ([False, True], [[1, 2], [3, 4], [5, 6]], {"axis": 1})),
    ("nested list axis minus one", lambda: ([True, False], [[1, 2], [3, 4], [5, 6]], {"axis": -1})),
    ("empty axis result", lambda: ([False, False], [[1, 2], [3, 4]], {"axis": 1})),
    ("string payload list", lambda: ([True, False, True], ["alpha", "beta", "gamma"], {})),
    ("axis mismatch error", lambda: ([True, False, True], [[1, 2], [3, 4]], {"axis": 1})),
]

ok = True
for label, factory in cases:
    condition, a, kwargs = factory()
    actual = compress_outcome(fnp.compress, condition, a, **kwargs)
    condition, a, kwargs = factory()
    expected = compress_outcome(np.compress, condition, a, **kwargs)
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
        "compress Python-container surfaces should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn compress_1d_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([False, True, False, True, True])
a = np.array([1, 2, 3, 4, 5])
result = fnp.compress(condition, a)
expected = np.compress(condition, a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "compress 1d basic should match numpy"
    );
    Ok(())
}

#[test]
fn compress_2d_no_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False, True])
a = np.array([[1, 2], [3, 4], [5, 6]])
result = fnp.compress(condition, a)
expected = np.compress(condition, a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "compress 2d no axis should match numpy"
    );
    Ok(())
}

#[test]
fn compress_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False, True])
a = np.array([[1, 2], [3, 4], [5, 6]])
result = fnp.compress(condition, a, axis=0)
expected = np.compress(condition, a, axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "compress 2d axis=0 should match numpy"
    );
    Ok(())
}

#[test]
fn compress_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False])
a = np.array([[1, 2], [3, 4], [5, 6]])
result = fnp.compress(condition, a, axis=1)
expected = np.compress(condition, a, axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "compress 2d axis=1 should match numpy"
    );
    Ok(())
}

#[test]
fn compress_all_false() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([False, False, False])
a = np.array([1, 2, 3])
result = fnp.compress(condition, a)
expected = np.compress(condition, a)
print(np.array_equal(result, expected) and len(result) == 0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "compress all false should return empty"
    );
    Ok(())
}

#[test]
fn compress_string_payload_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False, True])
a = np.array(["alpha", "beta", "gamma"])
result = fnp.compress(condition, a)
expected = np.compress(condition, a)
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "compress should preserve NumPy string payload behavior"
    );
    Ok(())
}

#[test]
fn compress_string_condition_truthiness_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array(["", "x", "0"])
a = np.array([10, 20, 30])
result = fnp.compress(condition, a)
expected = np.compress(condition, a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "compress should match NumPy string condition truthiness"
    );
    Ok(())
}

#[test]
fn compress_object_condition_truthiness_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([object(), None, 1], dtype=object)
a = np.array([10, 20, 30])
result = fnp.compress(condition, a)
expected = np.compress(condition, a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "compress should match NumPy object condition truthiness"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// choose
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn choose_python_container_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def choose_outcome(fn, a, choices, **kwargs):
    try:
        result = fn(a, choices, **kwargs)
        arr = np.asarray(result)
        return ("ok", type(result).__name__, str(arr.dtype), tuple(arr.shape), arr.tolist())
    except Exception as exc:
        return ("err", type(exc).__name__, str(exc))

cases = [
    ("list indices list choices", lambda: ([0, 1, 2, 1], [[10, 10, 10, 10], [20, 20, 20, 20], [30, 30, 30, 30]], {})),
    ("tuple indices tuple choices", lambda: ((0, 1, 0), ((1.5, 2.5, 3.5), (10.5, 20.5, 30.5)), {})),
    ("nested list choices", lambda: ([[0, 1], [1, 0]], [[[1, 2], [3, 4]], [[10, 20], [30, 40]]], {})),
    ("wrap mode", lambda: ([-1, 0, 3], [[10, 10, 10], [20, 20, 20], [30, 30, 30]], {"mode": "wrap"})),
    ("clip mode", lambda: ([-1, 0, 3], [[10, 10, 10], [20, 20, 20], [30, 30, 30]], {"mode": "clip"})),
    ("raise mode error", lambda: ([0, 3], [[10, 10], [20, 20]], {})),
]

ok = True
for label, factory in cases:
    a, choices, kwargs = factory()
    actual = choose_outcome(fnp.choose, a, choices, **kwargs)
    a, choices, kwargs = factory()
    expected = choose_outcome(np.choose, a, choices, **kwargs)
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
        "choose Python-container surfaces should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn choose_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 2, 1, 0])
choices = [np.array([10, 10, 10, 10, 10]), np.array([20, 20, 20, 20, 20]), np.array([30, 30, 30, 30, 30])]
result = fnp.choose(a, choices)
expected = np.choose(a, choices)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "choose basic should match numpy");
    Ok(())
}

#[test]
fn choose_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[0, 1], [1, 0]])
choices = [np.array([[1, 2], [3, 4]]), np.array([[10, 20], [30, 40]])]
result = fnp.choose(a, choices)
expected = np.choose(a, choices)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "choose 2d should match numpy");
    Ok(())
}

#[test]
fn choose_float_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 0, 1])
choices = [np.array([1.5, 2.5, 3.5, 4.5]), np.array([10.5, 20.5, 30.5, 40.5])]
result = fnp.choose(a, choices)
expected = np.choose(a, choices)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "choose float arrays should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// diagonal
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn diagonal_python_container_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def diagonal_outcome(fn, a, **kwargs):
    try:
        result = fn(a, **kwargs)
        arr = np.asarray(result)
        return ("ok", type(result).__name__, str(arr.dtype), tuple(arr.shape), arr.tolist())
    except Exception as exc:
        return ("err", type(exc).__name__, str(exc))

cases = [
    ("list 2d", lambda: ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], {})),
    ("tuple offset positive", lambda: (((1, 2, 3), (4, 5, 6), (7, 8, 9)), {"offset": 1})),
    ("list offset negative", lambda: ([[1, 2, 3, 4], [5, 6, 7, 8]], {"offset": -1})),
    ("nested list custom axes", lambda: (np.arange(24).reshape(2, 3, 4).tolist(), {"axis1": 0, "axis2": 2})),
    ("scalar error", lambda: (5, {})),
    ("repeated axis error", lambda: ([[1, 2], [3, 4]], {"axis1": 0, "axis2": 0})),
]

ok = True
for label, factory in cases:
    a, kwargs = factory()
    actual = diagonal_outcome(fnp.diagonal, a, **kwargs)
    a, kwargs = factory()
    expected = diagonal_outcome(np.diagonal, a, **kwargs)
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
        "diagonal Python-container surfaces should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn diagonal_2d_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.diagonal(a)
expected = np.diagonal(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diagonal 2d basic should match numpy"
    );
    Ok(())
}

#[test]
fn diagonal_offset_positive() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.diagonal(a, offset=1)
expected = np.diagonal(a, offset=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diagonal offset=1 should match numpy"
    );
    Ok(())
}

#[test]
fn diagonal_offset_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.diagonal(a, offset=-1)
expected = np.diagonal(a, offset=-1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diagonal offset=-1 should match numpy"
    );
    Ok(())
}

#[test]
fn diagonal_non_square() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
result = fnp.diagonal(a)
expected = np.diagonal(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diagonal non-square should match numpy"
    );
    Ok(())
}

#[test]
fn diagonal_3d_default_axes() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.diagonal(a)
expected = np.diagonal(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diagonal 3d default axes should match numpy"
    );
    Ok(())
}

#[test]
fn diagonal_3d_custom_axes() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.diagonal(a, axis1=0, axis2=2)
expected = np.diagonal(a, axis1=0, axis2=2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diagonal 3d custom axes should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compress_extract_equivalence() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False, True, False, True])
a = np.array([1, 2, 3, 4, 5])
compress_result = fnp.compress(condition, a)
extract_result = fnp.extract(condition, a)
print(np.array_equal(compress_result, extract_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "compress and extract should be equivalent for 1d"
    );
    Ok(())
}

#[test]
fn compress_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j, 3+2j, 4-2j], dtype=np.complex128)
condition = [True, False, True, False]
fnp_result = fnp.compress(condition, a)
np_result = np.compress(condition, a)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "compress complex should match numpy");
    Ok(())
}

#[test]
fn choose_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
choices = [np.array([1+1j, 2+2j], dtype=np.complex128),
           np.array([3+3j, 4+4j], dtype=np.complex128)]
a = [0, 1]
fnp_result = fnp.choose(a, choices)
np_result = np.choose(a, choices)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "choose complex should match numpy");
    Ok(())
}

/// Locks the zero-copy gather fast path for `numpy.compress` with axis=None
/// (`try_zerocopy_f64_compress`) to bit-exact parity. compress copies selected
/// values verbatim, so parity must hold at the IEEE-754 bit level (signed zero,
/// nan, inf). Compares sha256 of raw output bytes across same-length and shorter
/// conditions and extreme values — the bool-condition + f64-array zero-copy path.
#[test]
fn compress_zerocopy_f64_bit_exact_matches_numpy() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE
rng = np.random.default_rng(20260605)
chunks = []
for n in [1000, 100003]:
    x = rng.standard_normal(n)
    chunks.append(np.asarray(mod.compress(x > 0.1, x)).tobytes())
    chunks.append(np.asarray(mod.compress(x[:n // 3] > 0, x)).tobytes())
xe = np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float64)
chunks.append(np.asarray(mod.compress(np.array([True, True, False, True, True]), xe)).tobytes())
print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_script(body.replace("MODULE", "fnp")))?;
    let numpy_hash = numpy_oracle(&format!(
        "import numpy as np\n{}",
        body.replace("MODULE", "np")
    ))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "zero-copy compress must be bit-identical to numpy (sha256 of raw output bytes)"
    );
    Ok(())
}

/// Locks the zero-copy f64 per-axis compress (try_zerocopy_f64_compress_axis):
/// deterministic 2-D and 3-D float64 arrays compressed along several axes with
/// full / shorter / all-true / all-false conditions, byte-identical to
/// numpy.compress plus a sha256 golden. Elements move verbatim, so it is
/// bit-identical to numpy and the prior extract path.
#[test]
fn compress_f64_axis_matches_numpy_bytes_and_golden() -> Result<(), String> {
    let script = fnp_script(
        r#"
import hashlib
s = 0x2545F4914F6CDD1D
def nxt():
    global s
    s = (s * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
    return s
A = np.empty((130, 71), dtype=np.float64)
for i in range(130):
    for j in range(71):
        A[i, j] = ((nxt() >> 11) / (1 << 53)) * 8.0 - 4.0
B = np.empty((11, 13, 9), dtype=np.float64)
for x in np.ndindex(11, 13, 9):
    B[x] = ((nxt() >> 11) / (1 << 53)) * 6.0 - 3.0
def mask(n, kind):
    if kind == 'half':  return (np.arange(n) % 2 == 0)
    if kind == 'short': return (np.arange(n * 2 // 3) % 3 != 0)
    if kind == 'all':   return np.ones(n, bool)
    if kind == 'none':  return np.zeros(n, bool)
h = hashlib.sha256()
allmatch = True
specs = [(A, 0), (A, 1), (B, 0), (B, 1), (B, 2)]
for (arr, ax) in specs:
    for kind in ('half', 'short', 'all', 'none'):
        c = mask(arr.shape[ax], kind)
        r = np.asarray(fnp.compress(c, arr, axis=ax))
        e = np.compress(c, arr, axis=ax)
        if r.shape != e.shape or r.dtype != e.dtype or r.tobytes() != e.tobytes():
            allmatch = False
        h.update(r.tobytes())
print(allmatch)
print(h.hexdigest())
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.lines();
    assert_eq!(
        lines.next().unwrap_or("").trim(),
        "True",
        "per-axis f64 compress must be byte-identical to numpy.compress"
    );
    assert_eq!(
        lines.next().unwrap_or("").trim(),
        "699d33622306a9a32dbcf94b5596dffe824018d6d06f9b8c94c4a65375e9f770",
        "per-axis f64 compress golden sha256 drifted"
    );
    Ok(())
}
