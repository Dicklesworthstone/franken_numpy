//! Conformance tests for numpy.matmul against NumPy oracle.
//!
//! Tests matmul (matrix multiplication).

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
fn matmul_2d_matrices() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
b = np.arange(6).reshape(3, 2)
result = fnp.matmul(a, b)
expected = np.matmul(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "matmul 2D matrices should match numpy"
    );
    Ok(())
}

#[test]
fn matmul_vector_matrix() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.arange(6).reshape(3, 2)
result = fnp.matmul(a, b)
expected = np.matmul(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "matmul vector-matrix should match numpy"
    );
    Ok(())
}

#[test]
fn matmul_matrix_vector() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
b = np.array([1.0, 2.0, 3.0])
result = fnp.matmul(a, b)
expected = np.matmul(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "matmul matrix-vector should match numpy"
    );
    Ok(())
}

#[test]
fn matmul_batch() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
b = np.arange(16).reshape(2, 4, 2)
result = fnp.matmul(a, b)
expected = np.matmul(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "matmul batch should match numpy"
    );
    Ok(())
}

#[test]
fn matmul_operator() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
b = np.arange(6).reshape(3, 2)
result = fnp.matmul(a, b)
expected = a @ b
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "matmul should match @ operator"
    );
    Ok(())
}
