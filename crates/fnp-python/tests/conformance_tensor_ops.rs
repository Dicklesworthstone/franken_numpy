//! Conformance tests for numpy tensor operations against NumPy oracle.
//!
//! Tests tensorsolve, tensorinv.

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
// tensorsolve
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn tensorsolve_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.eye(2*3*4).reshape((2*3, 4, 2, 3, 4))
b = np.random.randn(2*3, 4)
result = fnp.tensorsolve(a, b)
expected = np.linalg.tensorsolve(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tensorsolve basic should match numpy");
    Ok(())
}

#[test]
fn tensorsolve_with_axes() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
a = np.eye(24).reshape((4, 6, 8, 3))
b = np.random.randn(8, 3)
result = fnp.tensorsolve(a, b, axes=[2, 3])
expected = np.linalg.tensorsolve(a, b, axes=[2, 3])
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tensorsolve with axes should match numpy");
    Ok(())
}

#[test]
fn tensorsolve_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([1.0, 0.0])
result = fnp.tensorsolve(a, b)
expected = np.linalg.tensorsolve(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tensorsolve 2d should match numpy");
    Ok(())
}

#[test]
fn tensorsolve_verify_solution() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.eye(2*3).reshape((2, 3, 2, 3))
b = np.ones((2, 3))
x = fnp.tensorsolve(a, b)
reconstructed = np.tensordot(a, x, axes=2)
print(np.allclose(reconstructed, b))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tensorsolve solution should satisfy equation");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// tensorinv
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn tensorinv_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.eye(24).reshape((4, 6, 8, 3))
result = fnp.tensorinv(a)
expected = np.linalg.tensorinv(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tensorinv basic should match numpy");
    Ok(())
}

#[test]
fn tensorinv_ind1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.eye(12).reshape((12, 3, 4))
result = fnp.tensorinv(a, ind=1)
expected = np.linalg.tensorinv(a, ind=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tensorinv ind=1 should match numpy");
    Ok(())
}

#[test]
fn tensorinv_ind3() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.eye(24).reshape((2, 3, 4, 24))
result = fnp.tensorinv(a, ind=3)
expected = np.linalg.tensorinv(a, ind=3)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tensorinv ind=3 should match numpy");
    Ok(())
}

#[test]
fn tensorinv_verify_inverse() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.eye(12).reshape((4, 3, 12))
ainv = fnp.tensorinv(a, ind=2)
product = np.tensordot(ainv, a, 2)
# Should be identity-like
print(np.allclose(product, np.eye(12).reshape((12, 12))))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tensorinv inverse should be correct");
    Ok(())
}

#[test]
fn tensorinv_shape_check() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.eye(24).reshape((4, 6, 8, 3))
result = fnp.tensorinv(a)
expected = np.linalg.tensorinv(a)
print(result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tensorinv shape should match numpy");
    Ok(())
}

#[test]
fn tensorinv_float32() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.eye(12, dtype='float64').reshape((4, 3, 4, 3))
result = fnp.tensorinv(a)
expected = np.linalg.tensorinv(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tensorinv float64 should match numpy");
    Ok(())
}
