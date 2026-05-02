//! Conformance tests for numpy closeness comparison functions against NumPy oracle.
//!
//! Tests allclose, isclose.

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
// allclose
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn allclose_equal() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 2.0, 3.0])
result = fnp.allclose(a, b)
expected = np.allclose(a, b)
print(result == expected == True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose equal should match numpy");
    Ok(())
}

#[test]
fn allclose_close() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0 + 1e-9, 2.0 + 1e-9, 3.0 + 1e-9])
result = fnp.allclose(a, b)
expected = np.allclose(a, b)
print(result == expected == True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose close should match numpy");
    Ok(())
}

#[test]
fn allclose_not_close() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.1, 2.1, 3.1])
result = fnp.allclose(a, b)
expected = np.allclose(a, b)
print(result == expected == False)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose not close should match numpy");
    Ok(())
}

#[test]
fn allclose_custom_rtol() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.05, 2.1, 3.15])
result = fnp.allclose(a, b, rtol=0.1)
expected = np.allclose(a, b, rtol=0.1)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose custom rtol should match numpy");
    Ok(())
}

#[test]
fn allclose_custom_atol() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.01, 2.01, 3.01])
result = fnp.allclose(a, b, atol=0.1)
expected = np.allclose(a, b, atol=0.1)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose custom atol should match numpy");
    Ok(())
}

#[test]
fn allclose_with_nan_default() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0])
b = np.array([1.0, np.nan, 3.0])
result = fnp.allclose(a, b)
expected = np.allclose(a, b)
# By default, nan != nan, so allclose returns False
print(result == expected == False)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose with nan default should match numpy");
    Ok(())
}

#[test]
fn allclose_with_nan_equal_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0])
b = np.array([1.0, np.nan, 3.0])
result = fnp.allclose(a, b, equal_nan=True)
expected = np.allclose(a, b, equal_nan=True)
print(result == expected == True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose equal_nan=True should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// isclose
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn isclose_equal() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 2.0, 3.0])
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isclose equal should match numpy");
    Ok(())
}

#[test]
fn isclose_mixed() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0 + 1e-9, 2.5, 3.0])  # second element differs
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isclose mixed should match numpy");
    Ok(())
}

#[test]
fn isclose_custom_rtol() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.05, 2.1, 3.5])
result = fnp.isclose(a, b, rtol=0.1)
expected = np.isclose(a, b, rtol=0.1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isclose custom rtol should match numpy");
    Ok(())
}

#[test]
fn isclose_custom_atol() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.001, 2.05, 3.2])
result = fnp.isclose(a, b, atol=0.1)
expected = np.isclose(a, b, atol=0.1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isclose custom atol should match numpy");
    Ok(())
}

#[test]
fn isclose_with_nan_default() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0])
b = np.array([1.0, np.nan, 3.0])
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isclose with nan default should match numpy");
    Ok(())
}

#[test]
fn isclose_with_nan_equal_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0])
b = np.array([1.0, np.nan, 3.0])
result = fnp.isclose(a, b, equal_nan=True)
expected = np.isclose(a, b, equal_nan=True)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isclose equal_nan=True should match numpy");
    Ok(())
}

#[test]
fn isclose_with_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.inf, -np.inf])
b = np.array([1.0, np.inf, -np.inf])
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isclose with inf should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn allclose_isclose_relationship() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0 + 1e-9, 2.0 + 1e-9, 3.0 + 1e-9])
# allclose should be True iff all elements of isclose are True
allclose_result = fnp.allclose(a, b)
isclose_result = fnp.isclose(a, b)
print(allclose_result == np.all(isclose_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose should equal all(isclose)");
    Ok(())
}

#[test]
fn isclose_symmetry() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.1, 2.0, 3.3])
result_ab = fnp.isclose(a, b)
result_ba = fnp.isclose(b, a)
print(np.array_equal(result_ab, result_ba))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isclose should be symmetric");
    Ok(())
}
