//! Conformance tests for numpy.reshape and numpy.ravel against NumPy oracle.
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
// reshape
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn reshape_1d_to_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6])
result = fnp.reshape(a, (2, 3))
expected = np.reshape(a, (2, 3))
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "reshape 1d to 2d should match numpy");
    Ok(())
}

#[test]
fn reshape_2d_to_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.reshape(a, (6,))
expected = np.reshape(a, (6,))
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "reshape 2d to 1d should match numpy");
    Ok(())
}

#[test]
fn reshape_with_minus1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6])
result = fnp.reshape(a, (2, -1))
expected = np.reshape(a, (2, -1))
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "reshape with -1 should match numpy");
    Ok(())
}

#[test]
fn reshape_3d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24)
result = fnp.reshape(a, (2, 3, 4))
expected = np.reshape(a, (2, 3, 4))
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "reshape 3d should match numpy");
    Ok(())
}

#[test]
fn reshape_order_c() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.reshape(a, (3, 2), order='C')
expected = np.reshape(a, (3, 2), order='C')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "reshape order=C should match numpy");
    Ok(())
}

#[test]
fn reshape_order_f() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.reshape(a, (3, 2), order='F')
expected = np.reshape(a, (3, 2), order='F')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "reshape order=F should match numpy");
    Ok(())
}

#[test]
fn reshape_float_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.5, 2.5, 3.5, 4.5])
result = fnp.reshape(a, (2, 2))
expected = np.reshape(a, (2, 2))
print(np.allclose(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "reshape float array should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ravel
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ravel_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.ravel(a)
expected = np.ravel(a)
print(np.array_equal(result, expected) and result.ndim == 1)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ravel 2d should match numpy");
    Ok(())
}

#[test]
fn ravel_3d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.ravel(a)
expected = np.ravel(a)
print(np.array_equal(result, expected) and result.ndim == 1)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ravel 3d should match numpy");
    Ok(())
}

#[test]
fn ravel_already_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.ravel(a)
expected = np.ravel(a)
print(np.array_equal(result, expected) and np.array_equal(result, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ravel already 1d should match numpy");
    Ok(())
}

#[test]
fn ravel_order_c() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.ravel(a, order='C')
expected = np.ravel(a, order='C')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ravel order=C should match numpy");
    Ok(())
}

#[test]
fn ravel_order_f() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.ravel(a, order='F')
expected = np.ravel(a, order='F')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ravel order=F should match numpy");
    Ok(())
}

#[test]
fn ravel_float_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1.5, 2.5], [3.5, 4.5]])
result = fnp.ravel(a)
expected = np.ravel(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ravel float array should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ravel_reshape_minus1_equivalence() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
ravel_result = fnp.ravel(a)
reshape_result = fnp.reshape(a, (-1,))
print(np.array_equal(ravel_result, reshape_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ravel should equal reshape(-1)");
    Ok(())
}
