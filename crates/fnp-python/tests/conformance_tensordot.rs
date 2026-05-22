//! Conformance tests for numpy.tensordot against NumPy oracle.
//!
//! Tests tensordot (tensor dot product).

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
fn tensordot_axes_int() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(60.).reshape(3, 4, 5)
b = np.arange(24.).reshape(4, 3, 2)
result = fnp.tensordot(a, b, axes=([1, 0], [0, 1]))
expected = np.tensordot(a, b, axes=([1, 0], [0, 1]))
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tensordot with axes tuple should match numpy"
    );
    Ok(())
}

#[test]
fn tensordot_axes_0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
b = np.arange(6).reshape(2, 3)
result = fnp.tensordot(a, b, axes=0)
expected = np.tensordot(a, b, axes=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tensordot with axes=0 (outer product) should match numpy"
    );
    Ok(())
}

#[test]
fn tensordot_axes_1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
b = np.arange(12).reshape(3, 4)
result = fnp.tensordot(a, b, axes=1)
expected = np.tensordot(a, b, axes=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tensordot with axes=1 (matrix product) should match numpy"
    );
    Ok(())
}

#[test]
fn tensordot_axes_2() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
b = np.arange(24).reshape(3, 4, 2)
result = fnp.tensordot(a, b, axes=2)
expected = np.tensordot(a, b, axes=2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tensordot with axes=2 should match numpy"
    );
    Ok(())
}

#[test]
fn tensordot_scalar_result() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
fnp_result = fnp.tensordot(a, b, axes=1)
np_result = np.tensordot(a, b, axes=1)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "tensordot scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn tensordot_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1+1j, 2]], dtype=np.complex128)
b = np.array([[1], [2+1j]], dtype=np.complex128)
fnp_result = fnp.tensordot(a, b, axes=1)
np_result = np.tensordot(a, b, axes=1)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tensordot complex should match numpy");
    Ok(())
}
