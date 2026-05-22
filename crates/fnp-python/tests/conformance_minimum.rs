//! Conformance tests for numpy.minimum against NumPy oracle.
//!
//! Tests minimum (element-wise minimum).

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
fn minimum_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1.0, 3.0, 5.0])
x2 = np.array([2.0, 2.0, 6.0])
result = fnp.minimum(x1, x2)
expected = np.minimum(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "minimum basic should match numpy");
    Ok(())
}

#[test]
fn minimum_with_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1.0, np.nan, 3.0])
x2 = np.array([2.0, 2.0, np.nan])
result = fnp.minimum(x1, x2)
expected = np.minimum(x1, x2)
# NaN propagates
match = all((np.isnan(r) and np.isnan(e)) or r == e for r, e in zip(result.flat, expected.flat))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "minimum with NaN should propagate NaN like numpy"
    );
    Ok(())
}

#[test]
fn minimum_broadcasting() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([[1, 2], [3, 4]])
x2 = np.array([2, 3])
result = fnp.minimum(x1, x2)
expected = np.minimum(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "minimum broadcasting should match numpy"
    );
    Ok(())
}

#[test]
fn minimum_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(3.0)
x2 = np.float64(5.0)
fnp_result = fnp.minimum(x1, x2)
np_result = np.minimum(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "minimum scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn minimum_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z1 = np.array([1+1j, 5+5j, 2+2j], dtype=np.complex128)
z2 = np.array([3+3j, 2+2j, 4+4j], dtype=np.complex128)
fnp_result = fnp.minimum(z1, z2)
np_result = np.minimum(z1, z2)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "minimum complex should match numpy");
    Ok(())
}
