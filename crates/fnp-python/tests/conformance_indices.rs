//! Conformance tests for numpy index generation functions against NumPy oracle.
//!
//! Tests indices, diag_indices, tril_indices, triu_indices, fill_diagonal, copyto.

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
// indices
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn indices_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.indices((2, 3))
expected = np.indices((2, 3))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "indices 2d should match numpy");
    Ok(())
}

#[test]
fn indices_3d() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.indices((2, 3, 4))
expected = np.indices((2, 3, 4))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "indices 3d should match numpy");
    Ok(())
}

#[test]
fn indices_with_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.indices((3, 3), dtype='float64')
expected = np.indices((3, 3), dtype='float64')
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "indices with dtype should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// diag_indices
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn diag_indices_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.diag_indices(3)
expected = np.diag_indices(3)
print(all(np.array_equal(r, e) for r, e in zip(result, expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diag_indices basic should match numpy"
    );
    Ok(())
}

#[test]
fn diag_indices_with_ndim() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.diag_indices(3, ndim=3)
expected = np.diag_indices(3, ndim=3)
print(all(np.array_equal(r, e) for r, e in zip(result, expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diag_indices with ndim should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// tril_indices / triu_indices
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn tril_indices_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.tril_indices(4)
expected = np.tril_indices(4)
print(all(np.array_equal(r, e) for r, e in zip(result, expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tril_indices basic should match numpy"
    );
    Ok(())
}

#[test]
fn tril_indices_with_k() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.tril_indices(4, k=1)
expected = np.tril_indices(4, k=1)
print(all(np.array_equal(r, e) for r, e in zip(result, expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tril_indices with k should match numpy"
    );
    Ok(())
}

#[test]
fn triu_indices_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.triu_indices(4)
expected = np.triu_indices(4)
print(all(np.array_equal(r, e) for r, e in zip(result, expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "triu_indices basic should match numpy"
    );
    Ok(())
}

#[test]
fn triu_indices_with_k() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.triu_indices(4, k=-1)
expected = np.triu_indices(4, k=-1)
print(all(np.array_equal(r, e) for r, e in zip(result, expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "triu_indices with k should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// fill_diagonal
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fill_diagonal_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.zeros((3, 3))
b = np.zeros((3, 3))
fnp.fill_diagonal(a, 5)
np.fill_diagonal(b, 5)
print(np.array_equal(a, b))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "fill_diagonal basic should match numpy"
    );
    Ok(())
}

#[test]
fn fill_diagonal_array_val() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.zeros((3, 3))
b = np.zeros((3, 3))
fnp.fill_diagonal(a, [1, 2, 3])
np.fill_diagonal(b, [1, 2, 3])
print(np.array_equal(a, b))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "fill_diagonal array val should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// copyto
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn copyto_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
dst = np.zeros(5)
src = np.array([1, 2, 3, 4, 5])
fnp.copyto(dst, src)
print(np.array_equal(dst, src))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "copyto basic should match numpy");
    Ok(())
}

#[test]
fn copyto_with_where() -> Result<(), String> {
    let script = fnp_script(
        r#"
dst = np.zeros(5)
src = np.array([1, 2, 3, 4, 5])
mask = np.array([True, False, True, False, True])
fnp.copyto(dst, src, where=mask)
expected = np.zeros(5)
np.copyto(expected, src, where=mask)
print(np.array_equal(dst, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "copyto with where should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn diag_indices_use_for_diagonal() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(16).reshape(4, 4)
idx = fnp.diag_indices(4)
diag_values = a[idx]
expected = np.array([0, 5, 10, 15])
print(np.array_equal(diag_values, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diag_indices should extract diagonal"
    );
    Ok(())
}

#[test]
fn tril_triu_cover_all() -> Result<(), String> {
    let script = fnp_script(
        r#"
n = 3
tril = fnp.tril_indices(n)
triu = fnp.triu_indices(n, k=1)  # exclude diagonal
# Total should cover entire matrix
total = len(tril[0]) + len(triu[0])
print(total == n * n)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tril + triu should cover all elements"
    );
    Ok(())
}
