//! Conformance tests for numpy take and put functions against NumPy oracle.
//!
//! Tests take, take_along_axis, put_along_axis.

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
// take
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn take_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([4, 3, 5, 7, 6, 8])
indices = np.array([0, 1, 4])
result = fnp.take(a, indices)
expected = np.take(a, indices)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "take basic should match numpy");
    Ok(())
}

#[test]
fn take_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4], [5, 6]])
result = fnp.take(a, [0, 2], axis=0)
expected = np.take(a, [0, 2], axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "take 2d should match numpy");
    Ok(())
}

#[test]
fn take_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.take(a, [0, 2], axis=1)
expected = np.take(a, [0, 2], axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "take axis=1 should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// take_along_axis
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn take_along_axis_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[10, 20, 30], [40, 50, 60]])
ai = np.array([[0], [2]])
result = fnp.take_along_axis(a, ai, axis=1)
expected = np.take_along_axis(a, ai, axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "take_along_axis basic should match numpy"
    );
    Ok(())
}

#[test]
fn take_along_axis_argsort() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[3, 1, 2], [6, 4, 5]])
ai = np.argsort(a, axis=1)
result = fnp.take_along_axis(a, ai, axis=1)
expected = np.take_along_axis(a, ai, axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "take_along_axis argsort should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// put_along_axis
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn put_along_axis_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[10, 20, 30], [40, 50, 60]])
ai = np.array([[0], [2]])
fnp.put_along_axis(a, ai, 99, axis=1)
b = np.array([[10, 20, 30], [40, 50, 60]])
np.put_along_axis(b, ai, 99, axis=1)
print(np.array_equal(a, b))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "put_along_axis basic should match numpy"
    );
    Ok(())
}

#[test]
fn put_along_axis_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[10, 20, 30], [40, 50, 60]])
ai = np.array([[0, 1], [1, 2]])
values = np.array([[1, 2], [3, 4]])
fnp.put_along_axis(a, ai, values, axis=1)
b = np.array([[10, 20, 30], [40, 50, 60]])
np.put_along_axis(b, ai, values, axis=1)
print(np.array_equal(a, b))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "put_along_axis values should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn take_equals_indexing() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([4, 3, 5, 7, 6, 8])
indices = np.array([0, 1, 4])
take_result = fnp.take(a, indices)
index_result = a[indices]
print(np.array_equal(take_result, index_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "take should equal direct indexing");
    Ok(())
}

#[test]
fn take_along_axis_sorts() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[3, 1, 2], [6, 4, 5]])
ai = np.argsort(a, axis=1)
sorted_a = fnp.take_along_axis(a, ai, axis=1)
# Result should be sorted along axis 1
print(np.all(sorted_a[:, :-1] <= sorted_a[:, 1:]))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "take_along_axis with argsort should sort"
    );
    Ok(())
}

#[test]
fn take_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j, 3+2j, 4-2j], dtype=np.complex128)
indices = [0, 2, 3]
fnp_result = fnp.take(a, indices)
np_result = np.take(a, indices)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "take complex should match numpy");
    Ok(())
}

#[test]
fn put_along_axis_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
arr1 = np.array([[1+1j, 2-1j], [3+2j, 4-2j]], dtype=np.complex128)
arr2 = np.array([[1+1j, 2-1j], [3+2j, 4-2j]], dtype=np.complex128)
indices = np.array([[0], [1]])
values = np.array([[9+9j], [8+8j]], dtype=np.complex128)
fnp.put_along_axis(arr1, indices, values, axis=1)
np.put_along_axis(arr2, indices, values, axis=1)
print(np.array_equal(arr1, arr2))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "put_along_axis complex should match numpy"
    );
    Ok(())
}

#[test]
fn take_along_axis_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
arr = np.array([[1+1j, 2-1j, 3+2j], [4-2j, 5+1j, 6-1j]], dtype=np.complex128)
indices = np.array([[0, 2], [1, 0]])
fnp_result = fnp.take_along_axis(arr, indices, axis=1)
np_result = np.take_along_axis(arr, indices, axis=1)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "take_along_axis complex should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// float-index acceptance (numpy 2.4.3 still accepts Python float sequences as
// take/put indices, truncating toward zero, while rejecting float ndarrays)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn take_float_list_indices_truncate() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(5)
fnp_result = fnp.take(a, [1.9, 2.9, -1.0])
np_result = np.take(a, [1.9, 2.9, -1.0])
print(repr(fnp_result.tolist()) == repr(np_result.tolist()) and repr(np_result.tolist()) == '[1, 2, 4]')
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "take should truncate Python float indices toward zero like numpy"
    );
    Ok(())
}

#[test]
fn take_float_ndarray_indices_rejected() -> Result<(), String> {
    // numpy refuses to *cast* a float ndarray to intp; fnp must also raise.
    let script = fnp_script(
        r#"
a = np.arange(5)
raised = False
try:
    fnp.take(a, np.array([1.0, 2.0]))
except (TypeError, ValueError):
    raised = True
print(raised)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "take should reject float ndarray indices like numpy"
    );
    Ok(())
}

#[test]
fn take_float_nan_index_raises_value_error() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(5)
def err(fn):
    try:
        fn()
        return None
    except Exception as exc:
        return type(exc).__name__
print(err(lambda: fnp.take(a, [float('nan')])) == err(lambda: np.take(a, [float('nan')])) == 'ValueError')
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "take float NaN index should raise ValueError like numpy"
    );
    Ok(())
}

#[test]
fn take_float_inf_index_raises_overflow_error() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(5)
def err(fn):
    try:
        fn()
        return None
    except Exception as exc:
        return type(exc).__name__
print(err(lambda: fnp.take(a, [float('inf')])) == err(lambda: np.take(a, [float('inf')])) == 'OverflowError')
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "take float infinity index should raise OverflowError like numpy"
    );
    Ok(())
}

#[test]
fn put_float_list_indices_truncate() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_arr = np.arange(5)
np_arr = np.arange(5)
fnp.put(fnp_arr, [1.0, 2.9], [9, 8])
np.put(np_arr, [1.0, 2.9], [9, 8])
print(np.array_equal(fnp_arr, np_arr) and np_arr.tolist() == [0, 9, 8, 3, 4])
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "put should truncate Python float indices toward zero like numpy"
    );
    Ok(())
}

/// Locks the zero-copy flat-gather fast path for `numpy.take` (axis=None,
/// mode="raise", `try_zerocopy_f64_take`) to bit-exact parity. take copies the
/// gathered values verbatim, so parity must hold at the IEEE-754 bit level
/// (signed zero, nan, inf). Compares sha256 of raw output bytes across negative
/// (wraparound) indices, multi-D index arrays (output takes the index shape), and
/// extreme values — the f64-array + int64-index zero-copy path.
#[test]
fn take_zerocopy_f64_bit_exact_matches_numpy() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE
rng = np.random.default_rng(20260605)
chunks = []
for n in [1000, 100003]:
    x = rng.standard_normal(n)
    idx = rng.integers(-n, n, size=n // 3)
    chunks.append(np.asarray(mod.take(x, idx)).tobytes())
x = rng.standard_normal(1000)
chunks.append(np.asarray(mod.take(x, rng.integers(0, 1000, size=(5, 7)))).tobytes())
xe = np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float64)
chunks.append(np.asarray(mod.take(xe, np.array([4, 3, 2, 1, 0, -1, -5]))).tobytes())
print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_script(body.replace("MODULE", "fnp")))?;
    let numpy_hash = numpy_oracle(&format!(
        "import numpy as np\n{}",
        body.replace("MODULE", "np")
    ))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "zero-copy take must be bit-identical to numpy (sha256 of raw output bytes)"
    );
    Ok(())
}
