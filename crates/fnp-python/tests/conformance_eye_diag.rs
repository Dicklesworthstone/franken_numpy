//! Conformance tests for numpy.eye, identity, diag, diagflat against NumPy oracle.
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
// eye
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn eye_square() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.eye(4)
expected = np.eye(4)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "eye square should match numpy");
    Ok(())
}

#[test]
fn eye_rectangular() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.eye(3, 5)
expected = np.eye(3, 5)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "eye rectangular should match numpy");
    Ok(())
}

#[test]
fn eye_positive_k() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.eye(4, k=1)
expected = np.eye(4, k=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "eye positive k should match numpy");
    Ok(())
}

#[test]
fn eye_negative_k() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.eye(4, k=-1)
expected = np.eye(4, k=-1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "eye negative k should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// identity
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn identity_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.identity(4)
expected = np.identity(4)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "identity basic should match numpy");
    Ok(())
}

#[test]
fn identity_small() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.identity(1)
expected = np.identity(1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "identity 1x1 should match numpy");
    Ok(())
}

#[test]
fn identity_large() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.identity(100)
expected = np.identity(100)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "identity 100x100 should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// diag
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn diag_extract_main() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.diag(a)
expected = np.diag(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diag extract main diagonal should match numpy");
    Ok(())
}

#[test]
fn diag_extract_k1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.diag(a, k=1)
expected = np.diag(a, k=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diag extract k=1 should match numpy");
    Ok(())
}

#[test]
fn diag_extract_k_neg1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.diag(a, k=-1)
expected = np.diag(a, k=-1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diag extract k=-1 should match numpy");
    Ok(())
}

#[test]
fn diag_construct() -> Result<(), String> {
    let script = fnp_script(
        r#"
v = np.array([1, 2, 3])
result = fnp.diag(v)
expected = np.diag(v)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diag construct from 1d should match numpy");
    Ok(())
}

#[test]
fn diag_construct_k1() -> Result<(), String> {
    let script = fnp_script(
        r#"
v = np.array([1, 2, 3])
result = fnp.diag(v, k=1)
expected = np.diag(v, k=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diag construct with k=1 should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// diagflat
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn diagflat_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
v = np.array([1, 2, 3])
result = fnp.diagflat(v)
expected = np.diagflat(v)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diagflat 1d should match numpy");
    Ok(())
}

#[test]
fn diagflat_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
v = np.array([[1, 2], [3, 4]])
result = fnp.diagflat(v)
expected = np.diagflat(v)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diagflat 2d (flattens) should match numpy");
    Ok(())
}

#[test]
fn diagflat_with_k() -> Result<(), String> {
    let script = fnp_script(
        r#"
v = np.array([1, 2, 3])
result = fnp.diagflat(v, k=1)
expected = np.diagflat(v, k=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diagflat with k should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn eye_identity_equivalence() -> Result<(), String> {
    let script = fnp_script(
        r#"
eye_result = fnp.eye(5)
identity_result = fnp.identity(5)
print(np.array_equal(eye_result, identity_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "eye(n) should equal identity(n)");
    Ok(())
}

#[test]
fn diag_diagflat_1d_equivalence() -> Result<(), String> {
    let script = fnp_script(
        r#"
v = np.array([1, 2, 3])
diag_result = fnp.diag(v)
diagflat_result = fnp.diagflat(v)
print(np.array_equal(diag_result, diagflat_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diag(1d) should equal diagflat(1d)");
    Ok(())
}

#[test]
fn diag_roundtrip() -> Result<(), String> {
    let script = fnp_script(
        r#"
v = np.array([1, 2, 3])
matrix = fnp.diag(v)
extracted = fnp.diag(matrix)
print(np.array_equal(v, extracted))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diag(diag(v)) should return v for 1d input");
    Ok(())
}
