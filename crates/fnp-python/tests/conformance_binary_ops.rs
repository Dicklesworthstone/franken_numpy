//! Conformance tests for numpy basic binary operations against NumPy oracle.
//!
//! Tests add, subtract, multiply, divide functions.

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
fn add_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1, 2, 3, 4])
x2 = np.array([5, 6, 7, 8])
result = fnp.add(x1, x2)
expected = np.add(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "add basic should match numpy");
    Ok(())
}

#[test]
fn subtract_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([5, 6, 7, 8])
x2 = np.array([1, 2, 3, 4])
result = fnp.subtract(x1, x2)
expected = np.subtract(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "subtract basic should match numpy"
    );
    Ok(())
}

#[test]
fn multiply_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1, 2, 3, 4])
x2 = np.array([2, 3, 4, 5])
result = fnp.multiply(x1, x2)
expected = np.multiply(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "multiply basic should match numpy"
    );
    Ok(())
}

#[test]
fn divide_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([8.0, 9.0, 10.0, 12.0])
x2 = np.array([2.0, 3.0, 5.0, 4.0])
result = fnp.divide(x1, x2)
expected = np.divide(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "divide basic should match numpy");
    Ok(())
}

#[test]
fn add_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(3.0)
x2 = np.float64(5.0)
fnp_result = fnp.add(x1, x2)
np_result = np.add(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "add scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn subtract_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(5.0)
x2 = np.float64(3.0)
fnp_result = fnp.subtract(x1, x2)
np_result = np.subtract(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "subtract scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn multiply_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(3.0)
x2 = np.float64(5.0)
fnp_result = fnp.multiply(x1, x2)
np_result = np.multiply(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "multiply scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn divide_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(10.0)
x2 = np.float64(2.0)
fnp_result = fnp.divide(x1, x2)
np_result = np.divide(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "divide scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn add_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z1 = np.array([1+2j, 3+4j], dtype=np.complex128)
z2 = np.array([5+6j, 7+8j], dtype=np.complex128)
fnp_result = fnp.add(z1, z2)
np_result = np.add(z1, z2)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "add complex should match numpy");
    Ok(())
}

#[test]
fn subtract_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z1 = np.array([5+6j, 7+8j], dtype=np.complex128)
z2 = np.array([1+2j, 3+4j], dtype=np.complex128)
fnp_result = fnp.subtract(z1, z2)
np_result = np.subtract(z1, z2)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "subtract complex should match numpy");
    Ok(())
}

#[test]
fn multiply_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z1 = np.array([1+2j, 3+4j], dtype=np.complex128)
z2 = np.array([5+6j, 7+8j], dtype=np.complex128)
fnp_result = fnp.multiply(z1, z2)
np_result = np.multiply(z1, z2)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "multiply complex should match numpy");
    Ok(())
}

#[test]
fn divide_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z1 = np.array([5+10j, 15+20j], dtype=np.complex128)
z2 = np.array([1+2j, 3+4j], dtype=np.complex128)
fnp_result = fnp.divide(z1, z2)
np_result = np.divide(z1, z2)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "divide complex should match numpy");
    Ok(())
}
