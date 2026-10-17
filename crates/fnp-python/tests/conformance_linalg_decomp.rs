//! Conformance tests for numpy linalg decomposition operations against NumPy oracle.
//!
//! Tests qr, cholesky, eigh, eigvalsh, svdvals, inv, solve, lstsq, cond, multi_dot.

use std::io::Write;
use std::process::Stdio;

fn numpy_oracle(script: &str) -> Result<String, String> {
    let mut child = support::python_command()
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

mod support;
use support::fnp_script;

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

#[test]
fn qr_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1+1j, 2], [3, 4-1j], [5+2j, 6]], dtype=np.complex128)
fnp_q, fnp_r = fnp.linalg.qr(a)
np_q, np_r = np.linalg.qr(a)
# Check reconstruction
fnp_recon = fnp_q @ fnp_r
print(np.allclose(fnp_recon, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "qr complex reconstruction should match"
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

#[test]
fn cholesky_stacked_small_spd_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
raw = np.arange(32, dtype=np.float64).reshape(2, 4, 4)
a = raw @ np.swapaxes(raw, -1, -2) + 5000.0 * np.eye(4)
fnp_l = fnp.linalg.cholesky(a)
np_l = np.linalg.cholesky(a)
print(np.allclose(fnp_l, np_l))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "stacked 4x4 cholesky should match numpy"
    );
    Ok(())
}

#[test]
fn cholesky_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.eye(5)
fnp_l = fnp.linalg.cholesky(a)
np_l = np.linalg.cholesky(a)
# Cholesky of identity is identity
print(np.allclose(fnp_l, np_l) and np.allclose(fnp_l, np.eye(5)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cholesky of identity should be identity"
    );
    Ok(())
}

#[test]
fn cholesky_diagonal() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Diagonal positive definite matrix
a = np.diag([4.0, 9.0, 16.0, 25.0])
fnp_l = fnp.linalg.cholesky(a)
np_l = np.linalg.cholesky(a)
# Should be diagonal with sqrt of original diagonal
print(np.allclose(fnp_l, np_l))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cholesky diagonal should match numpy"
    );
    Ok(())
}

#[test]
fn cholesky_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Hermitian positive definite matrix
a = np.array([[4, 1+1j], [1-1j, 3]], dtype=np.complex128)
fnp_l = fnp.linalg.cholesky(a)
np_l = np.linalg.cholesky(a)
print(np.allclose(fnp_l, np_l))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cholesky complex should match numpy");
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

#[test]
fn eigh_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.eye(3)
fnp_vals, fnp_vecs = fnp.linalg.eigh(a)
np_vals, np_vecs = np.linalg.eigh(a)
# Identity eigenvalues are all 1
vals_close = np.allclose(fnp_vals, np.ones(3))
print(vals_close)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "eigh identity eigenvalues should be 1"
    );
    Ok(())
}

#[test]
fn eigh_diagonal() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.diag([1.0, 3.0, 2.0])
fnp_vals, _ = fnp.linalg.eigh(a)
np_vals, _ = np.linalg.eigh(a)
# Eigenvalues of diagonal are the diagonal elements (sorted)
print(np.allclose(np.sort(fnp_vals), np.sort(np_vals)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "eigh diagonal should match numpy");
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

#[test]
fn eigvalsh_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.eye(4)
fnp_vals = fnp.linalg.eigvalsh(a)
np_vals = np.linalg.eigvalsh(a)
# Identity has all eigenvalues = 1
print(np.allclose(fnp_vals, np.ones(4)) and np.allclose(np_vals, np.ones(4)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "eigvalsh identity should have eigenvalues 1"
    );
    Ok(())
}

#[test]
fn eigvalsh_diagonal() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Diagonal matrix eigenvalues are the diagonal elements
a = np.diag([1.0, 2.0, 3.0, 4.0])
fnp_vals = fnp.linalg.eigvalsh(a)
np_vals = np.linalg.eigvalsh(a)
print(np.allclose(np.sort(fnp_vals), np.sort(np_vals)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "eigvalsh diagonal should match numpy"
    );
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

// ─────────────────────────────────────────────────────────────────────────────
// Batched (stacked) matrix tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn inv_batched() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Stack of 3 invertible 2x2 matrices
a = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 9]],
    [[2, 1], [1, 3]]
], dtype=np.float64)
fnp_inv = fnp.linalg.inv(a)
np_inv = np.linalg.inv(a)
print(np.allclose(fnp_inv, np_inv))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "inv batched should match numpy");
    Ok(())
}

#[test]
fn solve_batched() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Stack of 2 systems
a = np.array([
    [[3, 1], [1, 2]],
    [[4, 2], [1, 3]]
], dtype=np.float64)
b = np.array([
    [9, 8],
    [10, 7]
], dtype=np.float64)
fnp_x = fnp.linalg.solve(a, b)
np_x = np.linalg.solve(a, b)
print(np.allclose(fnp_x, np_x))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "solve batched should match numpy");
    Ok(())
}

#[test]
fn qr_batched() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
], dtype=np.float64)
fnp_q, fnp_r = fnp.linalg.qr(a)
np_q, np_r = np.linalg.qr(a)
# Check reconstruction
fnp_recon = np.einsum('...ij,...jk->...ik', fnp_q, fnp_r)
np_recon = np.einsum('...ij,...jk->...ik', np_q, np_r)
print(np.allclose(fnp_recon, np_recon))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "qr batched reconstruction should match numpy"
    );
    Ok(())
}

/// The same matrix as a nested list, a tuple of tuples and an ndarray must give the same bytes.
/// Each 2-D linalg route used to gate its numpy delegation on an exact float ndarray, so a list
/// fell through to the native kernel: eigh returned eigenvector columns of the opposite sign,
/// and inv/cholesky/eigvalsh/svdvals/matrix_power/solve/det/slogdet differed in the last bits
/// (bead rc0923 .12). The sizes cross the old native floors: solve's 104 and det/slogdet's 832.
/// Where a route delegates to numpy, the answer must also be numpy's own bytes; float32
/// tensorinv must keep its dtype.
#[test]
fn linalg_answer_does_not_depend_on_the_operand_container() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(12)
DELEGATED = {"inv", "det", "slogdet", "solve", "cholesky", "eigh", "eigvalsh", "svdvals",
             "matrix_power"}
def outcome(fn, args):
    try:
        r = fn(*args)
    except Exception as ex:
        return (type(ex).__name__, str(ex))
    parts = tuple(r) if isinstance(r, tuple) else (r,)
    return tuple((np.asarray(p).dtype.str, np.asarray(p).shape, np.asarray(p).tobytes()) for p in parts)
bad = []
for n in (3, 8, 17, 110, 840):
    g = rng.random((n, n)) + n * np.eye(n)
    spd = g @ g.T
    b = rng.random(n)
    ops = {"det": (spd,), "slogdet": (spd,)} if n == 840 else {
        "inv": (g,), "det": (g,), "slogdet": (g,), "solve": (g, b), "cholesky": (spd,),
        "eigh": (spd,), "eigvalsh": (spd,), "svdvals": (g,), "matrix_power": (g, 3),
        "pinv": (g,), "tensorinv": (g, 1), "lstsq": (g, b, None), "matrix_rank": (g,),
        "cond": (g,), "norm": (g,), "qr": (g,), "svd": (g,), "eig": (g,), "eigvals": (g,)}
    for name, args in ops.items():
        fnp_fn = getattr(fnp.linalg, name)
        as_array = outcome(fnp_fn, args)
        for container in (lambda x: x.tolist(), lambda x: tuple(map(tuple, x.tolist()))):
            wrapped = tuple(container(x) if isinstance(x, np.ndarray) and x.ndim == 2 else x for x in args)
            if outcome(fnp_fn, wrapped) != as_array:
                bad.append(f"{name} n={n} {type(wrapped[0]).__name__}")
        if name in DELEGATED and as_array != outcome(getattr(np.linalg, name), args):
            bad.append(f"{name} n={n} differs from numpy")
a32 = (rng.random((4, 4)) + 4 * np.eye(4)).astype(np.float32)
if fnp.linalg.tensorinv(a32, 1).dtype != np.linalg.tensorinv(a32, 1).dtype:
    bad.append("tensorinv float32 dtype")
print(bad or "OK")
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "OK",
        "a linalg answer changed with the operand's container"
    );
    Ok(())
}

/// deadlock-audit-rc0923-epic-71qy3.12. Per-function routing table (native / conditional / delegate)
/// across all 18 `fnp.fft.*` entry points, all 28 `fnp.linalg.*` callables, and the `fnp.solve_triangular`
/// extension, verified by monkeypatching recording spies onto `numpy.fft.*`, `numpy.linalg.*` and
/// `numpy.matmul`. Includes two negative controls proving that flipping a `native` entry to `delegate`
/// or a `delegate` entry to `native` fails the check.
#[test]
fn fft_and_linalg_python_routing_table_and_negative_controls() -> Result<(), String> {
    let script = fnp_script(
        r#"
counts = {}
for mod_name, mod in (("fft", np.fft), ("linalg", np.linalg)):
    for name in list(getattr(mod, "__all__", dir(mod))):
        obj = getattr(mod, name, None)
        if callable(obj) and not isinstance(obj, type):
            key = f"{mod_name}.{name}"
            counts[key] = 0
            def make_spy(orig, k):
                def spy(*a, **kw):
                    counts[k] += 1
                    return orig(*a, **kw)
                return spy
            setattr(mod, name, make_spy(obj, key))

orig_np_matmul = np.matmul
def spy_np_matmul(*a, **kw):
    counts["linalg.matmul"] += 1
    return orig_np_matmul(*a, **kw)
np.matmul = spy_np_matmul

rng = np.random.default_rng(7)
x1 = rng.standard_normal(16)
x2 = rng.standard_normal((8, 8))
xc1 = x1 + 1j * rng.standard_normal(16)
xr1 = np.fft.rfft(x1)
xh1 = np.fft.ihfft(x1)
xr2 = np.fft.rfft2(x2)
xrn = np.fft.rfftn(x2)

g8 = rng.standard_normal((8, 8)) + 8 * np.eye(8)
spd8 = g8 @ g8.T
b8 = rng.standard_normal(8)
g64 = rng.standard_normal((64, 64)) + 64 * np.eye(64)
tall = rng.standard_normal((64, 4))
btall = rng.standard_normal(64)
i64_64 = np.arange(1, 64 * 64 + 1, dtype=np.int64).reshape(64, 64)
stack_g = rng.standard_normal((16, 4, 4)) + 4 * np.eye(4)
stack_spd = stack_g @ stack_g.transpose(0, 2, 1) + 4 * np.eye(4)
stack_3 = rng.standard_normal((16, 3, 3)) + 3 * np.eye(3)
stack_spd3 = stack_3 @ stack_3.transpose(0, 2, 1) + 3 * np.eye(3)
stack_256 = rng.standard_normal((256, 4, 4)) + 4 * np.eye(4)
stack_b256 = rng.standard_normal((256, 4, 1))
c8 = g8.astype(np.complex128)
v3a = rng.standard_normal((4, 3))
v3b = rng.standard_normal((4, 3))

ROUTING_TABLE = [
    ("fft", "fftfreq", "native", [((16, 0.1), {})], []),
    ("fft", "rfftfreq", "delegate", [], [((16, 0.1), {})]),
    ("fft", "fft", "delegate", [], [((xc1,), {})]),
    ("fft", "ifft", "delegate", [], [((xc1,), {})]),
    ("fft", "rfft", "delegate", [], [((x1,), {})]),
    ("fft", "irfft", "delegate", [], [((xr1,), {})]),
    ("fft", "hfft", "delegate", [], [((xh1,), {})]),
    ("fft", "ihfft", "delegate", [], [((x1,), {})]),
    ("fft", "fft2", "delegate", [], [((x2,), {})]),
    ("fft", "ifft2", "delegate", [], [((x2,), {})]),
    ("fft", "fftn", "delegate", [], [((x2,), {})]),
    ("fft", "ifftn", "delegate", [], [((x2,), {})]),
    ("fft", "rfft2", "delegate", [], [((x2,), {})]),
    ("fft", "irfft2", "delegate", [], [((xr2,), {})]),
    ("fft", "rfftn", "delegate", [], [((x2,), {})]),
    ("fft", "irfftn", "delegate", [], [((xrn,), {})]),
    ("fft", "fftshift", "delegate", [], [((x1,), {})]),
    ("fft", "ifftshift", "delegate", [], [((x1,), {})]),
    ("linalg", "matmul", "conditional", [((i64_64, i64_64), {})], [((c8, c8), {})]),
    ("linalg", "solve", "conditional", [((stack_256, stack_b256), {})], [((g8, b8), {})]),
    ("linalg", "inv", "conditional", [((stack_g,), {})], [((g8,), {})]),
    ("linalg", "cholesky", "conditional", [((stack_spd3,), {})], [((spd8,), {})]),
    ("linalg", "eigvalsh", "conditional", [((stack_spd,), {})], [((spd8,), {})]),
    ("linalg", "svdvals", "conditional", [((stack_g,), {})], [((g8,), {})]),
    ("linalg", "pinv", "conditional", [((g8,), {}), ((stack_g,), {})], [((g64,), {})]),
    ("linalg", "matrix_rank", "conditional", [((g8,), {}), ((stack_g,), {})], [((c8,), {})]),
    ("linalg", "cond", "conditional", [((stack_g,), {})], [((g8,), {})]),
    ("linalg", "norm", "conditional", [((stack_g, 2, (-2, -1)), {})], [((g8,), {})]),
    ("linalg", "matrix_norm", "conditional", [((stack_g,), {"ord": "nuc"})], [((g8,), {})]),
    ("linalg", "lstsq", "conditional", [((tall, btall, None), {})], [((c8, b8.astype(np.complex128), None), {})]),
    ("linalg", "matrix_power", "conditional", [((i64_64, 3), {})], [((g8, 3), {})]),
    ("linalg", "multi_dot", "conditional", [(([i64_64, i64_64],), {})], [(([g8, g8],), {})]),
    ("linalg", "det", "delegate", [], [((g8,), {}), ((stack_g,), {})]),
    ("linalg", "slogdet", "delegate", [], [((g8,), {}), ((stack_g,), {})]),
    ("linalg", "eigh", "delegate", [], [((spd8,), {}), ((stack_spd,), {})]),
    ("linalg", "eig", "delegate", [], [((g8,), {}), ((stack_g,), {})]),
    ("linalg", "eigvals", "delegate", [], [((g8,), {}), ((stack_g,), {})]),
    ("linalg", "svd", "delegate", [], [((g8,), {}), ((stack_g,), {})]),
    ("linalg", "qr", "delegate", [], [((g8,), {}), ((stack_g,), {})]),
    ("linalg", "tensorsolve", "delegate", [], [((g8.reshape(2, 4, 2, 4), b8.reshape(2, 4)), {})]),
    ("linalg", "tensorinv", "delegate", [], [((g8, 1), {})]),
    ("linalg", "vector_norm", "delegate", [], [((b8,), {})]),
    ("linalg", "matrix_transpose", "delegate", [], [((g8,), {})]),
    ("linalg", "vecdot", "delegate", [], [((b8, b8), {})]),
    ("linalg", "outer", "delegate", [], [((b8, b8), {})]),
    ("linalg", "tensordot", "delegate", [], [((g8, g8), {})]),
    ("linalg", "diagonal", "delegate", [], [((g8,), {})]),
    ("linalg", "trace", "delegate", [], [((g8,), {})]),
    ("linalg", "cross", "delegate", [], [((v3a, v3b), {})]),
]

def verify_table(table):
    errors = []
    for mod_name, fn_name, kind, nat_probes, del_probes in table:
        key = f"{mod_name}.{fn_name}"
        mod = getattr(fnp, mod_name)
        fn = getattr(mod, fn_name)
        if kind == "native" and (not nat_probes or del_probes):
            errors.append(f"{key}: invalid native probe spec")
        if kind == "delegate" and (nat_probes or not del_probes):
            errors.append(f"{key}: invalid delegate probe spec")
        if kind == "conditional" and (not nat_probes or not del_probes):
            errors.append(f"{key}: invalid conditional probe spec")
        for args, kw in nat_probes:
            counts[key] = 0
            fn(*args, **kw)
            if counts[key] != 0:
                errors.append(f"{key} ({kind}): expected native (0 hits), got {counts[key]} delegate hits")
        for args, kw in del_probes:
            counts[key] = 0
            fn(*args, **kw)
            if counts[key] == 0:
                errors.append(f"{key} ({kind}): expected delegate (>0 hits), got 0 hits")
    return errors

errs = verify_table(ROUTING_TABLE)
# Also verify fnp.solve_triangular extension executes natively with 0 numpy.linalg hits
for k in counts:
    counts[k] = 0
tri_res = fnp.solve_triangular(np.tril(g8), b8, lower=True)
if any(counts.values()) or not np.allclose(np.tril(g8) @ tri_res, b8):
    errs.append("solve_triangular extension failed native verification")

flipped1 = list(ROUTING_TABLE)
flipped1[0] = ("fft", "fftfreq", "delegate", [], [((16, 0.1), {})])
neg1 = verify_table(flipped1)

flipped2 = [
    ("linalg", "qr", "native", [((g8,), {})], []) if (m, n) == ("linalg", "qr") else row
    for row in ROUTING_TABLE
    for m, n in [(row[0], row[1])]
]
neg2 = verify_table(flipped2)

if errs:
    print("FAIL_VERIFY", errs)
elif not neg1 or not neg2:
    print("FAIL_NEG", neg1, neg2)
else:
    print("OK")
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "OK",
        "fft/linalg routing table or negative control failed: {result}"
    );
    Ok(())
}

/// deadlock-audit-41n96 / DIV-BATCHED-LINALG-NO-LAPACK. The stacked routes that stay native - inv,
/// solve, eigvalsh, cholesky (k <= 3), pinv, svdvals, cond, matrix_rank, norm / matrix_norm ord 2,
/// -2 and 'nuc', each inside its measured win gate - return numpy's type, dtype, shape and layout
/// and values within the ledger row's norm-wise bounds (C = 32): B1 `max|f - n| <= C * cond2 * k *
/// eps * max|n|` per lane for inv / solve / pinv / cholesky / cond / norm -2, B3 `max|f - n| <= C *
/// k * eps * ||A||2` for eigvalsh / svdvals / norm 2 / 'nuc', ranks equal on generic input. The
/// bounds must also FAIL a wrong answer: numpy's own result perturbed by 1e-7 relative, and a
/// transposed inverse, are checked against the same bound and must exceed it. Outside the row
/// everything is numpy's byte for byte: qr / svd / eig / eigvals / matrix_power stacks, and float32
/// and complex128 stacks of every route. A nested-list stack takes the ndarray's gate (it used to
/// bypass every gate into the native kernel).
#[test]
fn stacked_native_linalg_stays_within_the_divergence_bounds() -> Result<(), String> {
    let script = fnp_script(
        r#"
EPS, C = np.finfo(np.float64).eps, 32.0
rng = np.random.default_rng(43)
def well(b, k):
    return rng.standard_normal((b, k, k)) + k * np.eye(k)
def ill(b, k):
    u = np.linalg.qr(rng.standard_normal((b, k, k)))[0]
    v = np.linalg.qr(rng.standard_normal((b, k, k)))[0]
    return u @ (np.logspace(0, -6, k)[None, :, None] * v)
def spd(a):
    return a @ a.transpose(0, 2, 1) + a.shape[-1] * np.eye(a.shape[-1])
def lane_ratio_b1(f, n, a):
    kappa = np.linalg.cond(a, 2)
    k = a.shape[-1]
    err = np.abs(f - n).reshape(len(a), -1).max(axis=1)
    scale = np.abs(n).reshape(len(a), -1).max(axis=1)
    return float(np.max(err / np.maximum(C * kappa * k * EPS * scale, 1e-300)))
def lane_ratio_b3(f, n, a):
    k = a.shape[-1]
    err = np.abs(f - n).reshape(len(a), -1).max(axis=1)
    return float(np.max(err / np.maximum(C * k * EPS * np.linalg.norm(a, 2, axis=(-2, -1)), 1e-300)))
def same_meta(f, n):
    return (type(f) is type(n) and f.dtype == n.dtype and f.shape == n.shape
            and f.flags.c_contiguous == n.flags.c_contiguous)
bad, worst = [], {}
cells = [(1, 2), (16, 3), (16, 4), (1024, 4), (64, 8), (16, 6), (16, 16), (64, 24), (16, 32)]
for b, k in cells:
    for kind, a in (("well", well(b, k)), ("ill", ill(b, k))):
        rhs = rng.standard_normal((b, k))
        checks = [
            ("inv", lambda m: fnp.linalg.inv(m), lambda m: np.linalg.inv(m), "b1"),
            ("solve", lambda m: fnp.linalg.solve(m, rhs[..., None]), lambda m: np.linalg.solve(m, rhs[..., None]), "b1"),
            ("pinv", lambda m: fnp.linalg.pinv(m), lambda m: np.linalg.pinv(m), "b1"),
            ("cond", lambda m: fnp.linalg.cond(m), lambda m: np.linalg.cond(m), "b1"),
            ("norm-2", lambda m: fnp.linalg.norm(m, -2, axis=(-2, -1)), lambda m: np.linalg.norm(m, -2, axis=(-2, -1)), "b1"),
            ("svdvals", lambda m: fnp.linalg.svdvals(m), lambda m: np.linalg.svdvals(m), "b3"),
            ("norm2", lambda m: fnp.linalg.norm(m, 2, axis=(-2, -1)), lambda m: np.linalg.norm(m, 2, axis=(-2, -1)), "b3"),
            ("nuc", lambda m: fnp.linalg.matrix_norm(m, ord="nuc"), lambda m: np.linalg.matrix_norm(m, ord="nuc"), "b3"),
        ]
        if kind == "well":
            s = spd(a)
            checks.append(("eigvalsh", lambda m, s=s: fnp.linalg.eigvalsh(s), lambda m, s=s: np.linalg.eigvalsh(s), "b3s"))
            if k <= 3:
                checks.append(("cholesky", lambda m, s=s: fnp.linalg.cholesky(s), lambda m, s=s: np.linalg.cholesky(s), "b1s"))
        for name, ours_fn, theirs_fn, bound in checks:
            ours, theirs = ours_fn(a), theirs_fn(a)
            if not same_meta(ours, theirs):
                bad.append(f"{name} ({b},{k},{k}) {kind}: meta {type(ours).__name__} {ours.dtype} {ours.shape}")
                continue
            ref = spd(a) if bound.endswith("s") else a
            ratio = (lane_ratio_b1 if bound.startswith("b1") else lane_ratio_b3)(ours, theirs, ref)
            worst[name] = max(worst.get(name, 0.0), ratio)
            if ratio > 1.0:
                bad.append(f"{name} ({b},{k},{k}) {kind}: {ratio:.3g} x bound")
        ranks_f, ranks_n = fnp.linalg.matrix_rank(a), np.linalg.matrix_rank(a)
        if not (same_meta(ranks_f, ranks_n) and np.array_equal(ranks_f, ranks_n)):
            bad.append(f"matrix_rank ({b},{k},{k}) {kind}")
# The bounds discriminate: a 1e-7 relative perturbation and a transposed inverse both fail B1.
a = well(16, 4)
n_inv = np.linalg.inv(a)
if lane_ratio_b1(n_inv * (1 + 1e-7), n_inv, a) <= 1.0:
    bad.append("B1 does not reject a 1e-7 relative error")
if lane_ratio_b1(n_inv.transpose(0, 2, 1), n_inv, a) <= 1.0:
    bad.append("B1 does not reject a transposed inverse")
# Everything outside the row is numpy's, byte for byte: the delegated routes, and float32 /
# complex128 stacks of every route.
def outcome(fn, *args):
    try:
        r = fn(*args)
    except Exception as ex:
        return (type(ex).__name__, str(ex))
    parts = tuple(r) if isinstance(r, tuple) else (r,)
    return tuple((type(p).__name__, np.asarray(p).dtype.str, np.asarray(p).shape, np.asarray(p).tobytes())
                 for p in parts)
for b, k in ((16, 4), (64, 16), (16, 32)):
    base = well(b, k)
    for name, args in (("qr", (base,)), ("svd", (base,)), ("eig", (base,)), ("eigvals", (base,)),
                       ("matrix_power", (base, 3))):
        if outcome(getattr(fnp.linalg, name), *args) != outcome(getattr(np.linalg, name), *args):
            bad.append(f"{name} ({b},{k},{k}) differs from numpy")
    for dtype in (np.float32, np.complex128):
        m = base.astype(dtype)
        for name in ("inv", "det", "slogdet", "pinv", "svdvals", "eigvalsh", "matrix_rank", "cond"):
            arg = (m + m.conj().transpose(0, 2, 1)) if name == "eigvalsh" else m
            if outcome(getattr(fnp.linalg, name), arg) != outcome(getattr(np.linalg, name), arg):
                bad.append(f"{name} {np.dtype(dtype).name} ({b},{k},{k}) differs from numpy")
# A nested-list stack takes the ndarray's route: the same bytes either way.
for b, k in ((300, 20), (64, 32), (2, 4)):
    m = well(b, k)
    for name in ("inv", "svdvals", "pinv"):
        f = getattr(fnp.linalg, name)
        if f(m).tobytes() != f(m.tolist()).tobytes():
            bad.append(f"{name} ({b},{k},{k}): list and ndarray differ")
for name, ratio in sorted(worst.items()):
    print(f"WORST {name} {ratio:.3g}")
print("BAD", len(bad))
for line in bad[:30]:
    print("BADLINE", line)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    for line in result.lines() {
        eprintln!("{line}");
    }
    assert!(
        result.lines().any(|line| line == "BAD 0"),
        "stacked native linalg left its divergence bounds:\n{result}"
    );
    Ok(())
}

/// deadlock-audit-41n96. Stacked det / slogdet / eigh / tensorinv are numpy's: the native batch
/// kernels returned WRONG answers - det 0.0 and slogdet (0, -inf) for nonsingular, badly scaled
/// lanes (diag(1e10, 1, 1e-10) is 1.0 in numpy, [[0, 2^-30], [2^30, 0]] is -1.0), eigenvector
/// columns with flipped signs, NaN for an off-diagonal inf - so these must now equal numpy byte for
/// byte, the scaled cases included (fails on v0.4.0). A non-finite stack is numpy's in inv too
/// (the native LU spread one inf over the whole lane).
#[test]
fn stacked_det_slogdet_eigh_tensorinv_and_nonfinite_inv_are_numpys() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(41)
def outcome(fn, *args):
    try:
        r = fn(*args)
    except Exception as ex:
        return (type(ex).__name__, str(ex))
    parts = tuple(r) if isinstance(r, tuple) else (r,)
    return tuple((type(p).__name__, np.asarray(p).dtype.str, np.asarray(p).shape, np.asarray(p).tobytes())
                 for p in parts)
bad = []
scaled = np.stack([np.diag([1e10, 1.0, 1e-10]),
                   (rng.standard_normal((3, 3)) + 3 * np.eye(3)) * np.array([1e8, 1.0, 1e-8])[:, None]])
anti = np.array([[[0.0, 2.0**-30], [2.0**30, 0.0]]] * 4)
g = rng.standard_normal((16, 6, 6)) + 6 * np.eye(6)
sym = g + g.transpose(0, 2, 1)
inf_stack = rng.standard_normal((2, 3, 3)) + 3 * np.eye(3)
inf_stack[1, 0, 1] = np.inf
cases = [("det", (scaled,)), ("det", (anti,)), ("det", (g,)), ("det", (inf_stack,)),
         ("slogdet", (scaled,)), ("slogdet", (anti,)), ("slogdet", (g,)), ("slogdet", (inf_stack,)),
         ("eigh", (sym,)), ("eigh", (sym, "U")), ("inv", (inf_stack,)),
         ("tensorinv", (rng.standard_normal((4, 6, 24)) + 0.0, 2))]
for name, args in cases:
    ours, theirs = outcome(getattr(fnp.linalg, name), *args), outcome(getattr(np.linalg, name), *args)
    if ours != theirs:
        bad.append(f"{name} {np.asarray(args[0]).shape}: fnp={str(ours)[:100]} numpy={str(theirs)[:100]}")
if fnp.linalg.det(scaled)[0] != 1.0 or fnp.linalg.slogdet(anti)[0][0] != -1.0:
    bad.append("scaled det / anti slogdet sign not numpy's")
print(bad or "OK")
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "OK",
        "stacked det/slogdet/eigh/tensorinv and a non-finite inv stack must be numpy's"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Error behavior tests
// ─────────────────────────────────────────────────────────────────────────────

fn classify_error(script: &str) -> String {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut child = Command::new("python3")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("python3 should be available");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(script.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    if output.status.success() {
        "ok".to_string()
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("LinAlgError") {
            "LinAlgError".to_string()
        } else if stderr.contains("ValueError") {
            "ValueError".to_string()
        } else {
            format!("other: {}", stderr.lines().last().unwrap_or(""))
        }
    }
}

#[test]
fn eig_non_square_raises_linalgerror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(6).reshape(2, 3).astype(float)
fnp.linalg.eig(a)
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3).astype(float)
np.linalg.eig(a)
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "eig on non-square matrix should raise same error as numpy"
    );
}

#[test]
fn qr_empty_raises_valueerror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.array([]).reshape(0, 3)
fnp.linalg.qr(a)
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.array([]).reshape(0, 3)
np.linalg.qr(a)
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "qr on empty array should raise same error as numpy"
    );
}
