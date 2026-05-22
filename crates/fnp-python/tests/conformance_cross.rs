//! Conformance tests for numpy.cross against NumPy oracle.
//!
//! Tests cross (cross product of two vectors).

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
    assert_eq!(
        result.trim(),
        "True",
        "cross 3D vectors should match numpy"
    );
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
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cross 2D vectors should match numpy (returns scalar)"
    );
    Ok(())
}

#[test]
fn cross_batch_3d() -> Result<(), String> {
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
    assert_eq!(
        result.trim(),
        "True",
        "cross batch 3D should match numpy"
    );
    Ok(())
}

#[test]
fn cross_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = fnp.cross(a, b)
expected = np.cross(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cross float should match numpy"
    );
    Ok(())
}

#[test]
fn cross_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
b = np.array([4+1j, 5+2j, 6+3j], dtype=np.complex128)
fnp_result = fnp.cross(a, b)
np_result = np.cross(a, b)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cross complex should match numpy");
    Ok(())
}
