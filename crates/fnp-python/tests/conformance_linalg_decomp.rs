//! Conformance tests for numpy linalg decomposition operations against NumPy oracle.
//!
//! Tests qr, cholesky, eigh, eigvalsh, svdvals, inv, solve, lstsq, cond, multi_dot.

use std::io::Write;
use std::process::{Command, Stdio};

fn numpy_oracle(script: &str) -> Result<String, String> {
    let mut child = Command::new("python3")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("python3 should be available: {error}\nScript: {script}"))?;
    child
        .stdin
        .as_mut()
        .ok_or_else(|| format!("python3 stdin should be available\nScript: {script}"))?
        .write_all(script.as_bytes())
        .map_err(|error| {
            format!("failed to write NumPy oracle script: {error}\nScript: {script}")
        })?;
    let output = child
        .wait_with_output()
        .map_err(|error| format!("failed to wait for NumPy oracle: {error}\nScript: {script}"))?;
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
// qr
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn qr_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
fnp_q, fnp_r = fnp.linalg.qr(a)
np_q, np_r = np.linalg.qr(a)
q_close = np.allclose(np.abs(fnp_q), np.abs(np_q))
r_close = np.allclose(np.abs(fnp_r), np.abs(np_r))
print(q_close and r_close)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "qr basic should match numpy");
    Ok(())
}

#[test]
fn qr_square() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=np.float64)
fnp_q, fnp_r = fnp.linalg.qr(a)
np_q, np_r = np.linalg.qr(a)
# Check reconstruction
fnp_recon = fnp_q @ fnp_r
np_recon = np_q @ np_r
print(np.allclose(fnp_recon, np_recon))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "qr square reconstruction should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// cholesky
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cholesky_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Positive definite matrix
a = np.array([[4, 2], [2, 5]], dtype=np.float64)
fnp_l = fnp.linalg.cholesky(a)
np_l = np.linalg.cholesky(a)
print(np.allclose(fnp_l, np_l))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cholesky basic should match numpy");
    Ok(())
}

#[test]
fn cholesky_reconstruction() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=np.float64)
fnp_l = fnp.linalg.cholesky(a)
np_l = np.linalg.cholesky(a)
# L @ L.T should equal a
fnp_recon = fnp_l @ fnp_l.T
print(np.allclose(fnp_recon, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cholesky reconstruction should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// eigh
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn eigh_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Symmetric matrix
a = np.array([[1, 2], [2, 4]], dtype=np.float64)
fnp_vals, fnp_vecs = fnp.linalg.eigh(a)
np_vals, np_vecs = np.linalg.eigh(a)
vals_close = np.allclose(fnp_vals, np_vals)
# Eigenvectors can differ by sign
vecs_close = np.allclose(np.abs(fnp_vecs), np.abs(np_vecs))
print(vals_close and vecs_close)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "eigh basic should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// eigvalsh
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn eigvalsh_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [2, 4]], dtype=np.float64)
fnp_vals = fnp.linalg.eigvalsh(a)
np_vals = np.linalg.eigvalsh(a)
print(np.allclose(fnp_vals, np_vals))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "eigvalsh basic should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// svdvals
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn svdvals_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
fnp_s = fnp.linalg.svdvals(a)
np_s = np.linalg.svdvals(a)
print(np.allclose(fnp_s, np_s))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "svdvals basic should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// inv
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn inv_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]], dtype=np.float64)
fnp_inv = fnp.linalg.inv(a)
np_inv = np.linalg.inv(a)
print(np.allclose(fnp_inv, np_inv))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "inv basic should match numpy");
    Ok(())
}

#[test]
fn inv_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]], dtype=np.float64)
fnp_inv = fnp.linalg.inv(a)
# a @ inv(a) should equal identity
product = a @ fnp_inv
print(np.allclose(product, np.eye(2)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "inv should produce identity when multiplied"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// solve
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn solve_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[3, 1], [1, 2]], dtype=np.float64)
b = np.array([9, 8], dtype=np.float64)
fnp_x = fnp.linalg.solve(a, b)
np_x = np.linalg.solve(a, b)
print(np.allclose(fnp_x, np_x))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "solve basic should match numpy");
    Ok(())
}

#[test]
fn solve_verify() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[3, 1], [1, 2]], dtype=np.float64)
b = np.array([9, 8], dtype=np.float64)
fnp_x = fnp.linalg.solve(a, b)
# a @ x should equal b
print(np.allclose(a @ fnp_x, b))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "solve should satisfy a @ x = b");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// lstsq
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn lstsq_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 1], [1, 2], [1, 3]], dtype=np.float64)
b = np.array([1, 2, 2], dtype=np.float64)
fnp_result = fnp.linalg.lstsq(a, b, rcond=None)
np_result = np.linalg.lstsq(a, b, rcond=None)
print(np.allclose(fnp_result[0], np_result[0]))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "lstsq basic should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// cond
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cond_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]], dtype=np.float64)
fnp_c = fnp.linalg.cond(a)
np_c = np.linalg.cond(a)
print(np.allclose(fnp_c, np_c))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cond basic should match numpy");
    Ok(())
}

#[test]
fn cond_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Identity matrix has condition number 1
a = np.eye(3)
fnp_c = fnp.linalg.cond(a)
np_c = np.linalg.cond(a)
print(np.allclose(fnp_c, 1.0) and np.allclose(np_c, 1.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cond of identity should be 1");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// multi_dot
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn multi_dot_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.array([[9, 10], [11, 12]])
fnp_result = fnp.linalg.multi_dot([a, b, c])
np_result = np.linalg.multi_dot([a, b, c])
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "multi_dot basic should match numpy");
    Ok(())
}

#[test]
fn multi_dot_chain() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Test that multi_dot gives same result as sequential dots
a = np.random.randn(10, 20)
b = np.random.randn(20, 5)
c = np.random.randn(5, 15)
d = np.random.randn(15, 3)
fnp_result = fnp.linalg.multi_dot([a, b, c, d])
sequential = a @ b @ c @ d
print(np.allclose(fnp_result, sequential))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "multi_dot should equal sequential multiplication"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar return type tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cond_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
fnp_result = fnp.linalg.cond(a)
np_result = np.linalg.cond(a)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "linalg.cond scalar return type should match numpy: {result}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge case tests: singular/ill-conditioned matrices
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cond_singular_matrix() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Singular matrix - condition number should be inf
a = np.array([[1, 2], [2, 4]], dtype=np.float64)
fnp_c = fnp.linalg.cond(a)
np_c = np.linalg.cond(a)
print(np.isinf(fnp_c) == np.isinf(np_c))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cond of singular matrix should be inf"
    );
    Ok(())
}

#[test]
fn svdvals_rank_deficient() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Rank-deficient matrix - should have a zero singular value
a = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]], dtype=np.float64)
fnp_s = fnp.linalg.svdvals(a)
np_s = np.linalg.svdvals(a)
# Both should have same number of zero (or near-zero) singular values
fnp_near_zero = np.sum(fnp_s < 1e-10)
np_near_zero = np.sum(np_s < 1e-10)
print(fnp_near_zero == np_near_zero)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "svdvals should identify rank deficiency"
    );
    Ok(())
}

#[test]
fn lstsq_overdetermined() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Overdetermined system (more equations than unknowns)
a = np.array([[1], [2], [3]], dtype=np.float64)
b = np.array([1, 2, 4], dtype=np.float64)
fnp_x, _, _, _ = fnp.linalg.lstsq(a, b, rcond=None)
np_x, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
print(np.allclose(fnp_x, np_x))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "lstsq overdetermined should match numpy"
    );
    Ok(())
}

#[test]
fn lstsq_underdetermined() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Underdetermined system (more unknowns than equations)
a = np.array([[1, 2, 3]], dtype=np.float64)
b = np.array([6], dtype=np.float64)
fnp_x, _, _, _ = fnp.linalg.lstsq(a, b, rcond=None)
np_x, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
# Both should find a solution that satisfies the equation
print(np.allclose(a @ fnp_x, b))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "lstsq underdetermined should satisfy equation"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Complex matrix tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn inv_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1+1j, 2], [3, 4-1j]], dtype=np.complex128)
fnp_inv = fnp.linalg.inv(a)
np_inv = np.linalg.inv(a)
print(np.allclose(fnp_inv, np_inv))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "inv complex should match numpy");
    Ok(())
}

#[test]
fn solve_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1+1j, 2], [3, 4-1j]], dtype=np.complex128)
b = np.array([5+2j, 6-1j], dtype=np.complex128)
fnp_x = fnp.linalg.solve(a, b)
np_x = np.linalg.solve(a, b)
print(np.allclose(fnp_x, np_x))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "solve complex should match numpy");
    Ok(())
}

#[test]
fn eigh_hermitian() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Hermitian matrix
a = np.array([[2, 1+1j], [1-1j, 3]], dtype=np.complex128)
fnp_vals, fnp_vecs = fnp.linalg.eigh(a)
np_vals, np_vecs = np.linalg.eigh(a)
# Eigenvalues of Hermitian matrix are real
vals_close = np.allclose(fnp_vals, np_vals)
print(vals_close)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "eigh hermitian should match numpy");
    Ok(())
}

#[test]
fn svdvals_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1+1j, 2], [3, 4-1j], [5+2j, 6]], dtype=np.complex128)
fnp_s = fnp.linalg.svdvals(a)
np_s = np.linalg.svdvals(a)
print(np.allclose(fnp_s, np_s))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "svdvals complex should match numpy");
    Ok(())
}
