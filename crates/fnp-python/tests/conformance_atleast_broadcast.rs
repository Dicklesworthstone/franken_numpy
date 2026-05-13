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
