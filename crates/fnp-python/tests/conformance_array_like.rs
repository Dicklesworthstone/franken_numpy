//! Conformance tests for numpy *_like array creation functions against NumPy oracle.
//!
//! Tests empty_like, ones_like, zeros_like, full, full_like.

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
// empty_like
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn empty_like_shape_matches() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.empty_like(a)
expected = np.empty_like(a)
print(result.shape == expected.shape and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "empty_like shape should match numpy");
    Ok(())
}

#[test]
fn empty_like_with_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.empty_like(a, dtype='float64')
expected = np.empty_like(a, dtype='float64')
print(result.shape == expected.shape and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "empty_like with dtype should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ones_like
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ones_like_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.ones_like(a)
expected = np.ones_like(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ones_like should match numpy");
    Ok(())
}

#[test]
fn ones_like_with_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.ones_like(a, dtype='float64')
expected = np.ones_like(a, dtype='float64')
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ones_like with dtype should match numpy");
    Ok(())
}

#[test]
fn ones_like_3d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.ones_like(a)
expected = np.ones_like(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ones_like 3d should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// zeros_like
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn zeros_like_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.zeros_like(a)
expected = np.zeros_like(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "zeros_like should match numpy");
    Ok(())
}

#[test]
fn zeros_like_with_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.zeros_like(a, dtype='float64')
expected = np.zeros_like(a, dtype='float64')
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "zeros_like with dtype should match numpy");
    Ok(())
}

#[test]
fn zeros_like_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.5, 2.5, 3.5])
result = fnp.zeros_like(a)
expected = np.zeros_like(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "zeros_like float should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// full
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn full_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.full((2, 3), 7)
expected = np.full((2, 3), 7)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "full should match numpy");
    Ok(())
}

#[test]
fn full_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.full((3, 4), 3.14)
expected = np.full((3, 4), 3.14)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "full float should match numpy");
    Ok(())
}

#[test]
fn full_with_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.full((2, 2), 5, dtype='float32')
expected = np.full((2, 2), 5, dtype='float32')
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "full with dtype should match numpy");
    Ok(())
}

#[test]
fn full_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.full(5, 42)
expected = np.full(5, 42)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "full 1d should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// full_like
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn full_like_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.full_like(a, 99)
expected = np.full_like(a, 99)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "full_like should match numpy");
    Ok(())
}

#[test]
fn full_like_with_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.full_like(a, 3.14, dtype='float64')
expected = np.full_like(a, 3.14, dtype='float64')
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "full_like with dtype should match numpy");
    Ok(())
}

#[test]
fn full_like_float_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.5, 2.5, 3.5])
result = fnp.full_like(a, -1.0)
expected = np.full_like(a, -1.0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "full_like float array should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ones_like_equals_full_like_one() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
ones = fnp.ones_like(a)
full_ones = fnp.full_like(a, 1)
print(np.array_equal(ones, full_ones))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ones_like should equal full_like(a, 1)");
    Ok(())
}

#[test]
fn zeros_like_equals_full_like_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
zeros = fnp.zeros_like(a)
full_zeros = fnp.full_like(a, 0)
print(np.array_equal(zeros, full_zeros))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "zeros_like should equal full_like(a, 0)");
    Ok(())
}

#[test]
fn like_functions_preserve_shape() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
ones = fnp.ones_like(a)
zeros = fnp.zeros_like(a)
full = fnp.full_like(a, 7)
print(ones.shape == a.shape and zeros.shape == a.shape and full.shape == a.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "*_like functions should preserve shape");
    Ok(())
}
