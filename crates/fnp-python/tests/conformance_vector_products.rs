//! Conformance tests for numpy vector/matrix product functions against NumPy oracle.
//!
//! Tests cross, outer, inner, kron.

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
// cross (cross product)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cross_3d_vectors() -> Result<(), String> {
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
    assert_eq!(result.trim(), "True", "cross 3d vectors should match numpy");
    Ok(())
}

#[test]
fn cross_2d_vectors() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2])
b = np.array([3, 4])
result = fnp.cross(a, b)
expected = np.cross(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cross 2d vectors should match numpy");
    Ok(())
}

#[test]
fn cross_batch_3d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[7, 8, 9], [1, 2, 3]])
result = fnp.cross(a, b)
expected = np.cross(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cross batch 3d should match numpy");
    Ok(())
}

#[test]
fn cross_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.5, 2.5, 3.5])
b = np.array([4.5, 5.5, 6.5])
result = fnp.cross(a, b)
expected = np.cross(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cross float should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// outer (outer product)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn outer_basic() -> Result<(), String> {
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
    assert_eq!(result.trim(), "True", "outer basic should match numpy");
    Ok(())
}

#[test]
fn outer_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([0.5, 1.5, 2.5])
result = fnp.outer(a, b)
expected = np.outer(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "outer float should match numpy");
    Ok(())
}

#[test]
fn outer_2d_flattened() -> Result<(), String> {
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
    assert_eq!(
        result.trim(),
        "True",
        "outer 2d flattened should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// inner (inner product)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn inner_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.inner(a, b)
expected = np.inner(a, b)
print(np.allclose(result, expected))
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
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "inner 2d should match numpy");
    Ok(())
}

#[test]
fn inner_broadcast() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([1, 2, 3])
result = fnp.inner(a, b)
expected = np.inner(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "inner broadcast should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// kron (Kronecker product)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn kron_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5])
result = fnp.kron(a, b)
expected = np.kron(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "kron 1d should match numpy");
    Ok(())
}

#[test]
fn kron_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[0, 5], [6, 7]])
result = fnp.kron(a, b)
expected = np.kron(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "kron 2d should match numpy");
    Ok(())
}

#[test]
fn kron_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.eye(2)
b = np.eye(3)
result = fnp.kron(a, b)
expected = np.kron(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "kron identity should match numpy");
    Ok(())
}

#[test]
fn kron_mixed_dims() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2])
b = np.array([[1, 2], [3, 4]])
result = fnp.kron(a, b)
expected = np.kron(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "kron mixed dims should match numpy");
    Ok(())
}
