//! Conformance tests for numpy atleast_*d and broadcast functions against NumPy oracle.
//!
//! Tests atleast_1d, atleast_2d, atleast_3d, broadcast_to, broadcast_arrays.

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

#[test]
fn atleast_broadcast_python_container_and_keyword_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
class SubArray(np.ndarray):
    pass

def clean(value):
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value

def normalize_array(value):
    array = np.asarray(value)
    return (
        type(value).__name__,
        str(array.dtype),
        tuple(array.shape),
        clean(array.tolist()),
        bool(array.flags["WRITEABLE"]),
    )

def normalize(value):
    if isinstance(value, (list, tuple)):
        return ("sequence", [normalize_array(item) for item in value])
    return ("array", normalize_array(value))

def outcome(call_fn, *args, **kwargs):
    try:
        return ("ok", normalize(call_fn(*args, **kwargs)))
    except Exception as exc:
        return ("err", type(exc).__name__)

sub = np.arange(3).view(SubArray)
cases = [
    (
        "atleast_1d mixed multiple",
        "atleast_1d",
        lambda: ((1, [2, 3], np.array(None, dtype=object)), {}),
    ),
    (
        "atleast_2d tuple input",
        "atleast_2d",
        lambda: ((((1, 2, 3), (4, 5, 6)),), {}),
    ),
    (
        "atleast_3d object list",
        "atleast_3d",
        lambda: ((np.array(["a", "b"], dtype=object),), {}),
    ),
    ("broadcast_to Python list shape list", "broadcast_to", lambda: (([1, 2, 3], [2, 3]), {})),
    (
        "broadcast_to subclass subok",
        "broadcast_to",
        lambda: ((sub, (2, 3)), {"subok": True}),
    ),
    (
        "broadcast_arrays mixed Python list and ndarray",
        "broadcast_arrays",
        lambda: (([1, 2, 3], np.arange(3).reshape(1, 3)), {}),
    ),
    (
        "broadcast_arrays subclass subok",
        "broadcast_arrays",
        lambda: ((sub, np.arange(3).reshape(1, 3)), {"subok": True}),
    ),
    ("broadcast_to incompatible error", "broadcast_to", lambda: (([1, 2, 3], (2, 2)), {})),
    (
        "broadcast_arrays incompatible error",
        "broadcast_arrays",
        lambda: ((np.ones((2, 3)), np.ones((4,))), {}),
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
        "atleast/broadcast Python-container and keyword surfaces should match numpy: {result}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// atleast_1d
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn atleast_1d_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.atleast_1d(42)
expected = np.atleast_1d(42)
print(np.array_equal(result, expected) and result.ndim >= 1)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "atleast_1d scalar should match numpy"
    );
    Ok(())
}

#[test]
fn atleast_1d_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.atleast_1d(a)
expected = np.atleast_1d(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "atleast_1d 1d should match numpy");
    Ok(())
}

#[test]
fn atleast_1d_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.atleast_1d(a)
expected = np.atleast_1d(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "atleast_1d 2d should match numpy");
    Ok(())
}

#[test]
fn atleast_1d_multiple() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.atleast_1d(1, [2, 3], [[4, 5]])
expected = np.atleast_1d(1, [2, 3], [[4, 5]])
print(len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "atleast_1d multiple should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// atleast_2d
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn atleast_2d_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.atleast_2d(42)
expected = np.atleast_2d(42)
print(np.array_equal(result, expected) and result.ndim >= 2)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "atleast_2d scalar should match numpy"
    );
    Ok(())
}

#[test]
fn atleast_2d_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.atleast_2d(a)
expected = np.atleast_2d(a)
print(np.array_equal(result, expected) and result.ndim >= 2)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "atleast_2d 1d should match numpy");
    Ok(())
}

#[test]
fn atleast_2d_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.atleast_2d(a)
expected = np.atleast_2d(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "atleast_2d 2d should match numpy");
    Ok(())
}

#[test]
fn atleast_2d_multiple() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.atleast_2d(1, [2, 3], [[4, 5]])
expected = np.atleast_2d(1, [2, 3], [[4, 5]])
print(len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "atleast_2d multiple should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// atleast_3d
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn atleast_3d_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.atleast_3d(42)
expected = np.atleast_3d(42)
print(np.array_equal(result, expected) and result.ndim >= 3)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "atleast_3d scalar should match numpy"
    );
    Ok(())
}

#[test]
fn atleast_3d_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.atleast_3d(a)
expected = np.atleast_3d(a)
print(np.array_equal(result, expected) and result.ndim >= 3)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "atleast_3d 1d should match numpy");
    Ok(())
}

#[test]
fn atleast_3d_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.atleast_3d(a)
expected = np.atleast_3d(a)
print(np.array_equal(result, expected) and result.ndim >= 3)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "atleast_3d 2d should match numpy");
    Ok(())
}

#[test]
fn atleast_3d_3d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(8).reshape(2, 2, 2)
result = fnp.atleast_3d(a)
expected = np.atleast_3d(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "atleast_3d 3d should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// broadcast_to
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn broadcast_to_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.broadcast_to(a, (3, 3))
expected = np.broadcast_to(a, (3, 3))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "broadcast_to basic should match numpy"
    );
    Ok(())
}

#[test]
fn broadcast_to_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.broadcast_to(5, (3, 4))
expected = np.broadcast_to(5, (3, 4))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "broadcast_to scalar should match numpy"
    );
    Ok(())
}

#[test]
fn broadcast_to_column() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1], [2], [3]])
result = fnp.broadcast_to(a, (3, 4))
expected = np.broadcast_to(a, (3, 4))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "broadcast_to column should match numpy"
    );
    Ok(())
}

#[test]
fn broadcast_to_row() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3, 4]])
result = fnp.broadcast_to(a, (3, 4))
expected = np.broadcast_to(a, (3, 4))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "broadcast_to row should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// broadcast_arrays
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn broadcast_arrays_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([[1], [2], [3]])
result = fnp.broadcast_arrays(a, b)
expected = np.broadcast_arrays(a, b)
print(len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "broadcast_arrays basic should match numpy"
    );
    Ok(())
}

#[test]
fn broadcast_arrays_three() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([[1], [2]])
c = np.array([[[1]]])
result = fnp.broadcast_arrays(a, b, c)
expected = np.broadcast_arrays(a, b, c)
print(len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "broadcast_arrays three should match numpy"
    );
    Ok(())
}

#[test]
fn broadcast_arrays_same_shape() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = fnp.broadcast_arrays(a, b)
expected = np.broadcast_arrays(a, b)
print(len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "broadcast_arrays same shape should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn atleast_hierarchy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = 42  # scalar
r1 = fnp.atleast_1d(a)
r2 = fnp.atleast_2d(a)
r3 = fnp.atleast_3d(a)
# each level adds a dimension
print(r1.ndim >= 1 and r2.ndim >= 2 and r3.ndim >= 3)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "atleast functions should increase dimensions"
    );
    Ok(())
}

#[test]
fn broadcast_to_shape_preserved() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
target_shape = (5, 3)
result = fnp.broadcast_to(a, target_shape)
print(result.shape == target_shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "broadcast_to should produce target shape"
    );
    Ok(())
}

#[test]
fn broadcast_arrays_shapes_match() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([[1], [2], [3]])
results = fnp.broadcast_arrays(a, b)
# all results should have the same shape
shapes = [r.shape for r in results]
print(len(set(shapes)) == 1)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "broadcast_arrays results should have same shape"
    );
    Ok(())
}

#[test]
fn broadcast_to_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j], dtype=np.complex128)
fnp_result = fnp.broadcast_to(a, (3, 2))
np_result = np.broadcast_to(a, (3, 2))
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "broadcast_to complex should match numpy"
    );
    Ok(())
}

#[test]
fn atleast_1d_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array(1+1j, dtype=np.complex128)
fnp_result = fnp.atleast_1d(a)
np_result = np.atleast_1d(a)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "atleast_1d complex should match numpy"
    );
    Ok(())
}
