//! Conformance tests for numpy.trace against NumPy oracle.
//!
//! Tests trace (sum along diagonal).

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
fn trace_square_matrix() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.trace(a)
expected = np.trace(a)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "trace square matrix should match numpy"
    );
    Ok(())
}

#[test]
fn trace_rectangular_matrix() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
result = fnp.trace(a)
expected = np.trace(a)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "trace rectangular matrix should match numpy"
    );
    Ok(())
}

#[test]
fn trace_with_offset() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.trace(a, offset=1)
expected = np.trace(a, offset=1)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "trace with offset should match numpy"
    );
    Ok(())
}

#[test]
fn trace_negative_offset() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.trace(a, offset=-1)
expected = np.trace(a, offset=-1)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "trace with negative offset should match numpy"
    );
    Ok(())
}

#[test]
fn trace_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
fnp_result = fnp.trace(a)
np_result = np.trace(a)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "trace scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn trace_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1+1j, 2], [3, 4-1j]], dtype=np.complex128)
fnp_result = fnp.trace(a)
np_result = np.trace(a)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "trace complex should match numpy");
    Ok(())
}

#[test]
fn trace_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[np.inf, 1.0], [2.0, np.nan]])
fnp_result = fnp.trace(a)
np_result = np.trace(a)
# inf + nan = nan
print(np.isnan(fnp_result) and np.isnan(np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "trace special values should match numpy"
    );
    Ok(())
}

#[test]
fn trace_1x1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[5.0]])
fnp_result = fnp.trace(a)
np_result = np.trace(a)
print(fnp_result == np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "trace 1x1 should match numpy");
    Ok(())
}

#[test]
fn trace_large_offset() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Offset larger than matrix size - should return 0
a = np.array([[1, 2], [3, 4]])
fnp_result = fnp.trace(a, offset=5)
np_result = np.trace(a, offset=5)
print(fnp_result == np_result == 0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "trace large offset should match numpy"
    );
    Ok(())
}

#[test]
fn trace_3d_batched() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
fnp_result = fnp.trace(a)
np_result = np.trace(a)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "trace 3d batched should match numpy");
    Ok(())
}

/// Integer traces by dtype x shape x offset x layout: scalar TYPE, dtype and bytes against numpy.
/// Contiguous 2-D int64 / uint64 matrices now gather the diagonal from the buffer and fold it with
/// wrapping adds (numpy's add.reduce; 5.3x slower before, via `diagonal()` + the generic
/// extract). The ±2^62 values make the fold WRAP, which a float or saturating accumulator gets
/// wrong; 'q' / 'Q' matrices must come back as np.longlong / np.ulonglong, which every native
/// route answered as np.int64 / np.uint64 (64 cells before this test).
#[test]
fn trace_integer_matrices_match_numpy_type_and_wrapping_bytes() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(11)
def outcome(call):
    try:
        v = call()
    except Exception as ex:
        return (type(ex).__name__,)
    return (type(v).__name__, np.asarray(v).dtype.str, np.asarray(v).tobytes())
cells, bad = 0, []
for dt in ("i8", "u8", "q", "Q", "i4", "?"):
    for shape in ((5, 5), (3, 7), (7, 3), (0, 4), (4, 0), (1, 1), (64, 64)):
        if dt in ("u8", "Q"):
            m = (rng.integers(0, 2**63, shape, dtype=np.uint64) * np.uint64(3)).astype(dt)
        elif dt == "?":
            m = rng.integers(0, 2, shape).astype(dt)
        else:
            m = rng.integers(-(2**62), 2**62, shape).astype(dt)
        calls = {f"off={off}": (lambda mod, m=m, off=off: mod.trace(m, offset=off)) for off in (-8, -2, -1, 0, 1, 3, 8)}
        calls["T"] = lambda mod, m=m: mod.trace(m.T)
        calls["strided"] = lambda mod, m=m: mod.trace(m[::2, ::2])
        for label, call in calls.items():
            cells += 1
            ours, theirs = outcome(lambda: call(fnp)), outcome(lambda: call(np))
            if ours != theirs:
                bad.append(f"{dt} {shape} {label}: fnp={ours[:2]} numpy={theirs[:2]}")
print(cells, len(bad), bad[:6])
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "378 0 []",
        "integer trace differs from numpy: {result}"
    );
    Ok(())
}
