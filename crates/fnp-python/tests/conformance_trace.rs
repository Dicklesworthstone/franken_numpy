//! Conformance tests for numpy.trace against NumPy oracle.
//!
//! Tests trace (sum along diagonal).

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
fn trace_square_matrix() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.trace(a)
expected = np.trace(a)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "trace square matrix should match numpy"
    );
    Ok(())
}

#[test]
fn trace_rectangular_matrix() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
result = fnp.trace(a)
expected = np.trace(a)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "trace rectangular matrix should match numpy"
    );
    Ok(())
}

#[test]
fn trace_with_offset() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.trace(a, offset=1)
expected = np.trace(a, offset=1)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "trace with offset should match numpy"
    );
    Ok(())
}

#[test]
fn trace_negative_offset() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.trace(a, offset=-1)
expected = np.trace(a, offset=-1)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "trace with negative offset should match numpy"
    );
    Ok(())
}

#[test]
fn trace_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
fnp_result = fnp.trace(a)
np_result = np.trace(a)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "trace scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn trace_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1+1j, 2], [3, 4-1j]], dtype=np.complex128)
fnp_result = fnp.trace(a)
np_result = np.trace(a)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "trace complex should match numpy");
    Ok(())
}

#[test]
fn trace_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[np.inf, 1.0], [2.0, np.nan]])
fnp_result = fnp.trace(a)
np_result = np.trace(a)
# inf + nan = nan
print(np.isnan(fnp_result) and np.isnan(np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "trace special values should match numpy"
    );
    Ok(())
}

#[test]
fn trace_1x1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[5.0]])
fnp_result = fnp.trace(a)
np_result = np.trace(a)
print(fnp_result == np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "trace 1x1 should match numpy");
    Ok(())
}

#[test]
fn trace_large_offset() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Offset larger than matrix size - should return 0
a = np.array([[1, 2], [3, 4]])
fnp_result = fnp.trace(a, offset=5)
np_result = np.trace(a, offset=5)
print(fnp_result == np_result == 0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "trace large offset should match numpy"
    );
    Ok(())
}

#[test]
fn trace_3d_batched() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
fnp_result = fnp.trace(a)
np_result = np.trace(a)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "trace 3d batched should match numpy");
    Ok(())
}
