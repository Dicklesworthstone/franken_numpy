//! Conformance tests for numpy.kron against NumPy oracle.
//!
//! Tests kron (Kronecker product).

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
fn kron_2d_matrices() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = fnp.kron(a, b)
expected = np.kron(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "kron 2D matrices should match numpy"
    );
    Ok(())
}

#[test]
fn kron_1d_vectors() -> Result<(), String> {
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
    assert_eq!(
        result.trim(),
        "True",
        "kron 1D vectors should match numpy"
    );
    Ok(())
}

#[test]
fn kron_mixed_shapes() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([1, 2, 3])
result = fnp.kron(a, b)
expected = np.kron(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "kron with mixed shapes should match numpy"
    );
    Ok(())
}

#[test]
fn kron_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([[0.5, 1.5], [2.5, 3.5]])
result = fnp.kron(a, b)
expected = np.kron(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "kron float should match numpy"
    );
    Ok(())
}
