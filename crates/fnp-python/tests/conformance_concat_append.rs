//! Conformance tests for numpy.concatenate, append, insert, delete against NumPy oracle.
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
// concatenate
// ──────────────���──────────────────────────────────────────────────────────────

#[test]
fn concatenate_python_container_and_keyword_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def clean(value):
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value

def normalize(value):
    array = np.asarray(value)
    return (str(array.dtype), tuple(array.shape), clean(array.tolist()))

def outcome(concat_fn, *args, **kwargs):
    try:
        return ("ok", normalize(concat_fn(*args, **kwargs)))
    except Exception as exc:
        return ("err", type(exc).__name__)

cases = [
    ("tuple of Python lists", lambda: ((((1, 2), (3, 4)),), {})),
    (
        "axis none nested lists",
        lambda: (([np.array([[1, 2], [3, 4]]), [[5, 6]]],), {"axis": None}),
    ),
    (
        "negative axis tuple arrays",
        lambda: (((np.arange(6).reshape(2, 3), np.arange(6, 12).reshape(2, 3)),), {"axis": -1}),
    ),
    (
        "mixed ndarray and list",
        lambda: (([np.array([1, 2], dtype=np.int16), [3, 4]],), {}),
    ),
    (
        "dtype casting unsafe",
        lambda: (([np.array([1.25, 2.75]), np.array([3.5])],), {"dtype": np.int64, "casting": "unsafe"}),
    ),
    (
        "object string fallback",
        lambda: (([np.array(["a", "b"], dtype=object), ["c"]],), {}),
    ),
    ("empty sequence error", lambda: (([],), {})),
    ("scalar entries error", lambda: (([1, 2],), {})),
    (
        "invalid axis error",
        lambda: (([np.array([1, 2]), np.array([3, 4])],), {"axis": 2}),
    ),
    (
        "out dtype conflict error",
        lambda: (
            ([np.array([1, 2]), np.array([3, 4])],),
            {"out": np.empty(4, dtype=np.int64), "dtype": np.float64},
        ),
    ),
]

ok = True
for label, factory in cases:
    args, kwargs = factory()
    actual = outcome(fnp.concatenate, *args, **kwargs)
    args, kwargs = factory()
    expected = outcome(np.concatenate, *args, **kwargs)
    if actual != expected:
        print(label)
        print(actual)
        print(expected)
        ok = False

def out_contract(concat_fn):
    out = np.empty(4, dtype=np.float64)
    result = concat_fn([np.array([1.5, 2.5]), [3.5, 4.5]], out=out)
    return (normalize(result), normalize(out), result is out)

actual_out = out_contract(fnp.concatenate)
expected_out = out_contract(np.concatenate)
if actual_out != expected_out:
    print("out contract")
    print(actual_out)
    print(expected_out)
    ok = False

print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "concatenate Python-container and keyword surfaces should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn concatenate_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.concatenate([a, b])
expected = np.concatenate([a, b])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "concatenate 1d should match numpy");
    Ok(())
}

#[test]
fn concatenate_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = fnp.concatenate([a, b], axis=0)
expected = np.concatenate([a, b], axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "concatenate 2d axis=0 should match numpy"
    );
    Ok(())
}

#[test]
fn concatenate_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = fnp.concatenate([a, b], axis=1)
expected = np.concatenate([a, b], axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "concatenate 2d axis=1 should match numpy"
    );
    Ok(())
}

#[test]
fn concatenate_multiple_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2])
b = np.array([3, 4])
c = np.array([5, 6])
result = fnp.concatenate([a, b, c])
expected = np.concatenate([a, b, c])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "concatenate multiple arrays should match numpy"
    );
    Ok(())
}

#[test]
fn concatenate_axis_none() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
result = fnp.concatenate([a, b], axis=None)
expected = np.concatenate([a, b], axis=None)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "concatenate axis=None should flatten and match numpy"
    );
    Ok(())
}

#[test]
fn concatenate_negative_axis_3d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(12, dtype=np.int16).reshape(2, 3, 2)
b = np.arange(12, 24, dtype=np.int16).reshape(2, 3, 2)
result = fnp.concatenate([a, b], axis=-1)
expected = np.concatenate([a, b], axis=-1)
print(result.dtype == expected.dtype and result.shape == expected.shape and np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "concatenate negative axis should match numpy"
    );
    Ok(())
}

#[test]
fn concatenate_empty_axis_preserves_shape_and_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.empty((2, 0), dtype=np.float32)
b = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
result = fnp.concatenate([a, b, a], axis=1)
expected = np.concatenate([a, b, a], axis=1)
print(result.dtype == expected.dtype and result.shape == expected.shape and np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "concatenate empty axis operands should match numpy"
    );
    Ok(())
}

#[test]
fn concatenate_structured_arrays_preserves_fields() -> Result<(), String> {
    let script = fnp_script(
        r#"
dtype = np.dtype([("left", "i4"), ("right", "f8")])
a = np.array([(1, 1.5), (2, 2.5)], dtype=dtype)
b = np.array([(3, 3.5)], dtype=dtype)
result = fnp.concatenate([a, b])
expected = np.concatenate([a, b])
print(result.dtype == expected.dtype and result.shape == expected.shape and np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "concatenate structured arrays should match numpy"
    );
    Ok(())
}

// ��────────────────────────────────────────────────────────────────────────────
// append
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn append_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.append(a, [4, 5, 6])
expected = np.append(a, [4, 5, 6])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "append 1d should match numpy");
    Ok(())
}

#[test]
fn append_single_value() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.append(a, 4)
expected = np.append(a, 4)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "append single value should match numpy"
    );
    Ok(())
}

#[test]
fn append_2d_no_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.append(a, [[5, 6]])
expected = np.append(a, [[5, 6]])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "append 2d no axis should flatten and match numpy"
    );
    Ok(())
}

#[test]
fn append_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.append(a, [[5, 6]], axis=0)
expected = np.append(a, [[5, 6]], axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "append 2d axis=0 should match numpy");
    Ok(())
}

#[test]
fn append_insert_delete_python_container_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def clean(value):
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value

def normalize(value):
    array = np.asarray(value)
    return (str(array.dtype), tuple(array.shape), clean(array.tolist()))

def outcome(call_fn, *args, **kwargs):
    try:
        return ("ok", normalize(call_fn(*args, **kwargs)))
    except Exception as exc:
        return ("err", type(exc).__name__)

cases = [
    (
        "append same dtype flat fast path",
        "append",
        lambda: ((np.array([1, 2], dtype=np.int16), np.array([3, 4], dtype=np.int16)), {}),
    ),
    (
        "append Python list promotion",
        "append",
        lambda: ((np.array([1, 2], dtype=np.int16), [3, 4]), {}),
    ),
    (
        "append nested list axis 0",
        "append",
        lambda: ((np.array([[1, 2], [3, 4]]), [[5, 6]]), {"axis": 0}),
    ),
    (
        "append nested list axis 1",
        "append",
        lambda: ((np.array([[1], [2]]), [[3], [4]]), {"axis": 1}),
    ),
    (
        "insert flattened list positions",
        "insert",
        lambda: (([1, 2, 3], [1, 3], [10, 11]), {}),
    ),
    (
        "insert axis scalar broadcast",
        "insert",
        lambda: ((np.array([[1, 2], [3, 4]]), 1, 99), {"axis": 1}),
    ),
    (
        "delete scalar index Python list",
        "delete",
        lambda: (([1, 2, 3, 4], 1), {}),
    ),
    (
        "delete list indices",
        "delete",
        lambda: ((np.arange(6), [0, 2, 5]), {}),
    ),
    (
        "delete slice object",
        "delete",
        lambda: ((np.arange(6), slice(1, None, 2)), {}),
    ),
    (
        "delete bool mask",
        "delete",
        lambda: ((np.array([1, 2, 3, 4]), np.array([True, False, True, False])), {}),
    ),
    (
        "append invalid axis error",
        "append",
        lambda: ((np.array([1, 2]), [3]), {"axis": 1}),
    ),
    (
        "insert out of bounds error",
        "insert",
        lambda: (([1, 2], 5, 9), {}),
    ),
    (
        "delete out of bounds error",
        "delete",
        lambda: (([1, 2], 5), {}),
    ),
]

ok = True
for label, name, factory in cases:
    args, kwargs = factory()
    actual = outcome(getattr(fnp, name), *args, **kwargs)
    args, kwargs = factory()
    expected = outcome(getattr(np, name), *args, **kwargs)
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
        "append/insert/delete Python-container surfaces should match numpy: {result}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// insert
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn insert_single_value() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4])
result = fnp.insert(a, 2, 99)
expected = np.insert(a, 2, 99)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "insert single value should match numpy"
    );
    Ok(())
}

#[test]
fn insert_multiple_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4])
result = fnp.insert(a, 2, [98, 99])
expected = np.insert(a, 2, [98, 99])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "insert multiple values should match numpy"
    );
    Ok(())
}

#[test]
fn insert_at_beginning() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.insert(a, 0, 0)
expected = np.insert(a, 0, 0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "insert at beginning should match numpy"
    );
    Ok(())
}

#[test]
fn insert_at_end() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.insert(a, 3, 4)
expected = np.insert(a, 3, 4)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "insert at end should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// delete
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn delete_single_index() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.delete(a, 2)
expected = np.delete(a, 2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "delete single index should match numpy"
    );
    Ok(())
}

#[test]
fn delete_multiple_indices() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.delete(a, [1, 3])
expected = np.delete(a, [1, 3])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "delete multiple indices should match numpy"
    );
    Ok(())
}

#[test]
fn delete_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4], [5, 6]])
result = fnp.delete(a, 1, axis=0)
expected = np.delete(a, 1, axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "delete 2d axis=0 should match numpy");
    Ok(())
}

#[test]
fn delete_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.delete(a, 1, axis=1)
expected = np.delete(a, 1, axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "delete 2d axis=1 should match numpy");
    Ok(())
}

#[test]
fn delete_negative_index() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.delete(a, -1)
expected = np.delete(a, -1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "delete negative index should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Complex dtype tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn concatenate_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j], dtype=np.complex128)
b = np.array([3+2j, 4-2j], dtype=np.complex128)
fnp_result = fnp.concatenate([a, b])
np_result = np.concatenate([a, b])
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "concatenate complex should match numpy"
    );
    Ok(())
}

#[test]
fn append_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j], dtype=np.complex128)
b = np.array([3+2j, 4-2j], dtype=np.complex128)
fnp_result = fnp.append(a, b)
np_result = np.append(a, b)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "append complex should match numpy");
    Ok(())
}

#[test]
fn insert_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j, 3+2j], dtype=np.complex128)
fnp_result = fnp.insert(a, 1, 9+9j)
np_result = np.insert(a, 1, 9+9j)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "insert complex should match numpy");
    Ok(())
}

#[test]
fn delete_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j, 3+2j, 4-2j], dtype=np.complex128)
fnp_result = fnp.delete(a, 1)
np_result = np.delete(a, 1)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "delete complex should match numpy");
    Ok(())
}

/// Locks the zero-copy axis-0 concatenate fast path
/// (`try_zerocopy_f64_concatenate_axis0`) to bit-exact parity with numpy. The
/// concatenated buffers are copied verbatim, so parity must hold at the IEEE-754
/// bit level (signed zero, nan, inf). Compares the sha256 of raw output bytes
/// across 1-D, 2-D, and 3-D axis-0 concatenations and extreme values.
#[test]
fn concatenate_axis0_zerocopy_f64_bit_exact_matches_numpy() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE
rng = np.random.default_rng(20260605)
chunks = []
chunks.append(np.asarray(mod.concatenate([rng.standard_normal(1000), rng.standard_normal(2000)])).tobytes())
chunks.append(np.asarray(mod.concatenate([rng.standard_normal((100, 50)), rng.standard_normal((200, 50))], 0)).tobytes())
chunks.append(np.asarray(mod.concatenate([rng.standard_normal((2, 3, 4)), rng.standard_normal((5, 3, 4))], 0)).tobytes())
xe = np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float64)
chunks.append(np.asarray(mod.concatenate([xe, xe * 2], 0)).tobytes())
print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_script(body.replace("MODULE", "fnp")))?;
    let numpy_hash = numpy_oracle(&format!(
        "import numpy as np\n{}",
        body.replace("MODULE", "np")
    ))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "zero-copy axis-0 concatenate must be bit-identical to numpy (sha256 of raw output bytes)"
    );
    Ok(())
}

/// Locks the general (any-axis) zero-copy concatenate fast path
/// (`try_zerocopy_f64_concatenate`) to bit-exact parity with numpy for non-zero
/// axes. Covers 2-D axis-1, 3-D axis-1 and axis-2, and extreme values.
#[test]
fn concatenate_any_axis_zerocopy_f64_bit_exact_matches_numpy() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE
rng = np.random.default_rng(20260605)
chunks = []
chunks.append(np.asarray(mod.concatenate([rng.standard_normal((100, 50)), rng.standard_normal((100, 30))], 1)).tobytes())
chunks.append(np.asarray(mod.concatenate([rng.standard_normal((2, 30, 4)), rng.standard_normal((2, 30, 6))], 2)).tobytes())
chunks.append(np.asarray(mod.concatenate([rng.standard_normal((2, 30, 4)), rng.standard_normal((2, 5, 4))], 1)).tobytes())
xe = np.array([[0.0, -0.0, np.inf], [-np.inf, np.nan, 1e308]], dtype=np.float64)
chunks.append(np.asarray(mod.concatenate([xe, xe * 2], 1)).tobytes())
print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_script(body.replace("MODULE", "fnp")))?;
    let numpy_hash = numpy_oracle(&format!(
        "import numpy as np\n{}",
        body.replace("MODULE", "np")
    ))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "zero-copy any-axis concatenate must be bit-identical to numpy (sha256 of raw output bytes)"
    );
    assert_eq!(
        fnp_hash, "e3ed79c8bc8ae9a723fa04f0dcd23d47d37a81a4792420eb1c43ad6fb8d1746d",
        "golden sha256 of any-axis concatenate raw output bytes"
    );
    Ok(())
}

// The native parallel concatenate block copy (large output, >= gate) must be byte-identical to numpy
// across dtypes, axes, input counts, and call forms (positional axis, axis kwarg, no axis). concatenate
// moves whole elements verbatim, so the parallel disjoint-block byte copy is exact.
#[test]
fn concatenate_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE
rng = np.random.default_rng(20260701)
chunks = []
for dtn in ["float64", "float32", "int64", "int32", "int16", "int8", "uint32", "complex128", "complex64", "bool"]:
    dt = np.dtype(dtn)
    def mk(shp):
        if dt.kind == "f":
            return (rng.standard_normal(shp) * 1.5).astype(dt)
        if dt.kind == "c":
            return (rng.standard_normal(shp) + 1j * rng.standard_normal(shp)).astype(dt)
        if dt.kind == "b":
            return rng.integers(0, 2, shp).astype(dt)
        info = np.iinfo(dt)
        return rng.integers(info.min // 2, info.max // 2, shp).astype(dt)
    a = mk((1200, 1200)); b = mk((1200, 1200)); c = mk((600, 1200))
    chunks.append(np.ascontiguousarray(mod.concatenate([a, b])).tobytes())           # no axis (0)
    chunks.append(np.ascontiguousarray(mod.concatenate([a, b, c], 0)).tobytes())      # positional axis
    chunks.append(np.ascontiguousarray(mod.concatenate([a, b], axis=1)).tobytes())    # axis kwarg
    chunks.append(np.ascontiguousarray(mod.concatenate([a, b], axis=-1)).tobytes())
    chunks.append(np.ascontiguousarray(mod.vstack([a, b])).tobytes())
    chunks.append(np.ascontiguousarray(mod.hstack([a, b])).tobytes())
    t = mk((60, 400, 60)); u = mk((60, 250, 60))
    chunks.append(np.ascontiguousarray(mod.concatenate([t, u], axis=1)).tobytes())    # 3-D non-trivial outer
print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_script(body.replace("MODULE", "fnp")))?;
    let numpy_hash = numpy_oracle(&format!(
        "import numpy as np\n{}",
        body.replace("MODULE", "np")
    ))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "native parallel concatenate must be bit-identical to numpy (sha256 of raw output bytes)"
    );
    Ok(())
}

#[test]
fn insert_nd_axis_none_flat_view_matches_numpy() -> Result<(), String> {
    // np.insert(..., axis=None) flattens the input in C order before inserting.
    // This locks the N-D input-form fast path to byte-exact parity while keeping
    // explicit-axis and non-C-contiguous inputs on NumPy's existing semantics.
    let script = fnp_script(
        r#"
import time

verdicts = []

def outcome(fn, *args, **kwargs):
    try:
        out = np.asarray(fn(*args, **kwargs))
        return ("ok", str(out.dtype), tuple(out.shape), out.tobytes())
    except Exception as exc:
        return ("err", type(exc).__name__)

def ab(name, arr, obj, value, **kwargs):
    ours = outcome(fnp.insert, arr, obj, value, **kwargs)
    theirs = outcome(np.insert, arr, obj, value, **kwargs)
    if ours != theirs:
        verdicts.append(f"FAIL {name}")

base = np.array(
    [[0.0, -0.0, 1.5], [np.inf, -np.inf, np.nan]],
    dtype=np.float64,
)
for idx in (0, 3, base.size, -1, -base.size):
    ab(f"2-D index {idx}", base, idx, -0.0)
ab("3-D", np.arange(60, dtype=np.float64).reshape(3, 4, 5), 17, 99.25)
ab("0-D", np.array(7.0, dtype=np.float64), 1, 8.0)
ab("F-contiguous defer", np.asfortranarray(np.arange(24.0).reshape(4, 6)), 7, 2.5)
ab("axis 0 unchanged", np.arange(24.0).reshape(4, 6), 2, 2.5, axis=0)
ab("axis -1 unchanged", np.arange(24.0).reshape(4, 6), 2, 2.5, axis=-1)
ab("out of bounds", base, base.size + 1, 2.5)

large = np.arange(8_000_000, dtype=np.float64).reshape(2000, 4000)

def best(fn, reps=3):
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e3)
    return min(samples)

numpy_ms = best(lambda: np.insert(large, large.size // 2, -3.25))
fnp_ms = best(lambda: fnp.insert(large, large.size // 2, -3.25))
print(f"INSERT_ND_AXIS_NONE_AB numpy_ms={numpy_ms:.3f} fnp_ms={fnp_ms:.3f} ratio={numpy_ms / fnp_ms:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}");
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "True",
        "N-D axis=None scalar insert must be byte-identical to NumPy: {result}"
    );
    Ok(())
}
