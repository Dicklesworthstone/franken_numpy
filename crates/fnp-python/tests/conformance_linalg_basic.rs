//! Conformance tests for numpy basic linear algebra operations against NumPy oracle.
//!
//! Tests dot, matmul, inner, outer, cross, tensordot.

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
// dot
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn dot_1d_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.dot(a, b)
expected = np.dot(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "dot 1d-1d should match numpy");
    Ok(())
}

#[test]
fn dot_2d_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([1, 2])
result = fnp.dot(a, b)
expected = np.dot(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "dot 2d-1d should match numpy");
    Ok(())
}

#[test]
fn dot_2d_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = fnp.dot(a, b)
expected = np.dot(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "dot 2d-2d should match numpy");
    Ok(())
}

#[test]
fn dot_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.5, 2.5, 3.5])
b = np.array([0.5, 1.5, 2.5])
result = fnp.dot(a, b)
expected = np.dot(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "dot float should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// matmul
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn matmul_2d_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = fnp.matmul(a, b)
expected = np.matmul(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "matmul 2d-2d should match numpy");
    Ok(())
}

#[test]
fn matmul_1d_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2])
b = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.matmul(a, b)
expected = np.matmul(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "matmul 1d-2d should match numpy");
    Ok(())
}

#[test]
fn matmul_2d_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([1, 2, 3])
result = fnp.matmul(a, b)
expected = np.matmul(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "matmul 2d-1d should match numpy");
    Ok(())
}

#[test]
fn matmul_3d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(12).reshape(2, 3, 2)
b = np.arange(8).reshape(2, 2, 2)
result = fnp.matmul(a, b)
expected = np.matmul(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "matmul 3d should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// inner
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn inner_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.inner(a, b)
expected = np.inner(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "inner 1d should match numpy");
    Ok(())
}

#[test]
fn inner_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = fnp.inner(a, b)
expected = np.inner(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "inner 2d should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// outer
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn outer_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5])
result = fnp.outer(a, b)
expected = np.outer(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "outer 1d should match numpy");
    Ok(())
}

#[test]
fn outer_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([5, 6])
result = fnp.outer(a, b)
expected = np.outer(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "outer 2d should match numpy (flattens input)");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// cross
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cross_3d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.cross(a, b)
expected = np.cross(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cross 3d should match numpy");
    Ok(())
}

#[test]
fn cross_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2])
b = np.array([3, 4])
result = fnp.cross(a, b)
expected = np.cross(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cross 2d should match numpy");
    Ok(())
}

#[test]
fn cross_multiple() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[7, 8, 9], [10, 11, 12]])
result = fnp.cross(a, b)
expected = np.cross(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cross multiple should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// tensordot
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn tensordot_default() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Default axes=2 contracts last 2 axes of a with first 2 axes of b
a = np.arange(12).reshape(2, 3, 2)
b = np.arange(12).reshape(3, 2, 2)
result = fnp.tensordot(a, b)
expected = np.tensordot(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tensordot default should match numpy");
    Ok(())
}

#[test]
fn tensordot_axes_1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
b = np.arange(6).reshape(3, 2)
result = fnp.tensordot(a, b, axes=1)
expected = np.tensordot(a, b, axes=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tensordot axes=1 should match numpy");
    Ok(())
}

#[test]
fn tensordot_axes_0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(4).reshape(2, 2)
b = np.arange(4).reshape(2, 2)
result = fnp.tensordot(a, b, axes=0)
expected = np.tensordot(a, b, axes=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tensordot axes=0 should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn dot_matmul_equivalence_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
dot_result = fnp.dot(a, b)
matmul_result = fnp.matmul(a, b)
print(np.array_equal(dot_result, matmul_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "dot and matmul should be equivalent for 2d arrays");
    Ok(())
}

#[test]
fn inner_dot_equivalence_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
inner_result = fnp.inner(a, b)
dot_result = fnp.dot(a, b)
print(inner_result == dot_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "inner and dot should be equivalent for 1d arrays");
    Ok(())
}

#[test]
fn cross_antisymmetric() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
cross_ab = fnp.cross(a, b)
cross_ba = fnp.cross(b, a)
print(np.array_equal(cross_ab, -cross_ba))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cross product should be antisymmetric");
    Ok(())
}
