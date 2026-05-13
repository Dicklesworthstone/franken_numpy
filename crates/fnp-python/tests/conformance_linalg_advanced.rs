//! Conformance tests for advanced numpy.linalg functions against NumPy oracle.
//!
//! Tests pinv, eigvals, slogdet, matrix_rank, matrix_power, svd.
//!
//! Finding: These 6 linalg functions are exposed but had ZERO conformance tests.

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
// pinv (pseudo-inverse)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn pinv_square_invertible() -> Result<(), String> {
    let script = fnp_script(
        r#"
A = np.array([[1, 2], [3, 4]], dtype=np.float64)
fnp_pinv = fnp.pinv(A)
np_pinv = np.linalg.pinv(A)
print(np.allclose(fnp_pinv, np_pinv, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "pinv of invertible matrix should match numpy"
    );
    Ok(())
}

#[test]
fn pinv_rectangular() -> Result<(), String> {
    let script = fnp_script(
        r#"
A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
fnp_pinv = fnp.pinv(A)
np_pinv = np.linalg.pinv(A)
print(np.allclose(fnp_pinv, np_pinv, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "pinv of rectangular matrix should match numpy"
    );
    Ok(())
}

#[test]
fn pinv_singular() -> Result<(), String> {
    let script = fnp_script(
        r#"
A = np.array([[1, 2], [2, 4]], dtype=np.float64)  # rank 1
fnp_pinv = fnp.pinv(A)
np_pinv = np.linalg.pinv(A)
print(np.allclose(fnp_pinv, np_pinv, rtol=1e-8))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "pinv of singular matrix should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// eigvals
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn eigvals_symmetric() -> Result<(), String> {
    let script = fnp_script(
        r#"
A = np.array([[2, 1], [1, 2]], dtype=np.float64)
fnp_eig = np.sort(fnp.eigvals(A).real)
np_eig = np.sort(np.linalg.eigvals(A).real)
print(np.allclose(fnp_eig, np_eig, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "eigvals of symmetric matrix should match numpy"
    );
    Ok(())
}

#[test]
fn eigvals_diagonal() -> Result<(), String> {
    let script = fnp_script(
        r#"
A = np.diag([1.0, 2.0, 3.0, 4.0])
fnp_eig = np.sort(fnp.eigvals(A).real)
np_eig = np.sort(np.linalg.eigvals(A).real)
print(np.allclose(fnp_eig, np_eig, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "eigvals of diagonal matrix should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// slogdet (sign and log determinant)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn slogdet_positive_det() -> Result<(), String> {
    let script = fnp_script(
        r#"
A = np.array([[2, 1], [1, 3]], dtype=np.float64)
fnp_sign, fnp_logdet = fnp.slogdet(A)
np_sign, np_logdet = np.linalg.slogdet(A)
print(fnp_sign == np_sign and np.allclose(fnp_logdet, np_logdet, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "slogdet with positive det should match numpy"
    );
    Ok(())
}

#[test]
fn slogdet_negative_det() -> Result<(), String> {
    let script = fnp_script(
        r#"
A = np.array([[1, 2], [3, 4]], dtype=np.float64)  # det = -2
fnp_sign, fnp_logdet = fnp.slogdet(A)
np_sign, np_logdet = np.linalg.slogdet(A)
print(fnp_sign == np_sign and np.allclose(fnp_logdet, np_logdet, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "slogdet with negative det should match numpy"
    );
    Ok(())
}

#[test]
fn slogdet_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
A = np.eye(4)
fnp_sign, fnp_logdet = fnp.slogdet(A)
np_sign, np_logdet = np.linalg.slogdet(A)
print(fnp_sign == np_sign and np.allclose(fnp_logdet, np_logdet, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "slogdet of identity should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// matrix_rank
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn matrix_rank_full_rank() -> Result<(), String> {
    let script = fnp_script(
        r#"
A = np.array([[1, 2], [3, 4]], dtype=np.float64)
fnp_rank = fnp.matrix_rank(A)
np_rank = np.linalg.matrix_rank(A)
print(fnp_rank == np_rank)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "matrix_rank of full rank matrix should match numpy"
    );
    Ok(())
}

#[test]
fn matrix_rank_deficient() -> Result<(), String> {
    let script = fnp_script(
        r#"
A = np.array([[1, 2], [2, 4]], dtype=np.float64)  # rank 1
fnp_rank = fnp.matrix_rank(A)
np_rank = np.linalg.matrix_rank(A)
print(fnp_rank == np_rank)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "matrix_rank of rank-deficient matrix should match numpy"
    );
    Ok(())
}

#[test]
fn matrix_rank_zero_matrix() -> Result<(), String> {
    let script = fnp_script(
        r#"
A = np.zeros((3, 3))
fnp_rank = fnp.matrix_rank(A)
np_rank = np.linalg.matrix_rank(A)
print(fnp_rank == np_rank == 0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "matrix_rank of zero matrix should be 0"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// matrix_power
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn matrix_power_positive() -> Result<(), String> {
    let script = fnp_script(
        r#"
A = np.array([[1, 2], [3, 4]], dtype=np.float64)
fnp_pow = fnp.matrix_power(A, 3)
np_pow = np.linalg.matrix_power(A, 3)
print(np.allclose(fnp_pow, np_pow, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "matrix_power with positive exponent should match numpy"
    );
    Ok(())
}

#[test]
fn matrix_power_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
A = np.array([[1, 2], [3, 4]], dtype=np.float64)
fnp_pow = fnp.matrix_power(A, 0)
np_pow = np.linalg.matrix_power(A, 0)
print(np.allclose(fnp_pow, np_pow, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "matrix_power with exponent 0 should be identity"
    );
    Ok(())
}

#[test]
fn matrix_power_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
A = np.array([[1, 2], [3, 4]], dtype=np.float64)
fnp_pow = fnp.matrix_power(A, -1)
np_pow = np.linalg.matrix_power(A, -1)
print(np.allclose(fnp_pow, np_pow, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "matrix_power with -1 should be inverse"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// svd
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn svd_square() -> Result<(), String> {
    let script = fnp_script(
        r#"
A = np.array([[1, 2], [3, 4]], dtype=np.float64)
fnp_u, fnp_s, fnp_vh = fnp.svd(A)
np_u, np_s, np_vh = np.linalg.svd(A)
# Singular values should match exactly
s_match = np.allclose(fnp_s, np_s, rtol=1e-10)
# U @ diag(s) @ Vh should reconstruct A
fnp_recon = fnp_u @ np.diag(fnp_s) @ fnp_vh
np_recon = np_u @ np.diag(np_s) @ np_vh
recon_match = np.allclose(fnp_recon, np_recon, rtol=1e-10) and np.allclose(fnp_recon, A, rtol=1e-10)
print(s_match and recon_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "svd of square matrix should reconstruct original"
    );
    Ok(())
}

#[test]
fn svd_rectangular_wide() -> Result<(), String> {
    let script = fnp_script(
        r#"
A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
fnp_u, fnp_s, fnp_vh = fnp.svd(A)
np_u, np_s, np_vh = np.linalg.svd(A)
# Singular values should match
print(np.allclose(fnp_s, np_s, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "svd singular values of wide matrix should match numpy"
    );
    Ok(())
}

#[test]
fn svd_rectangular_tall() -> Result<(), String> {
    let script = fnp_script(
        r#"
A = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
fnp_u, fnp_s, fnp_vh = fnp.svd(A)
np_u, np_s, np_vh = np.linalg.svd(A)
# Singular values should match
print(np.allclose(fnp_s, np_s, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "svd singular values of tall matrix should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn pinv_times_original_is_identity_ish() -> Result<(), String> {
    let script = fnp_script(
        r#"
A = np.array([[1, 2], [3, 4]], dtype=np.float64)
Apinv = fnp.pinv(A)
# For full rank square matrix, A @ pinv(A) ≈ I
result = A @ Apinv
print(np.allclose(result, np.eye(2), rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "A @ pinv(A) should be identity for invertible A"
    );
    Ok(())
}

#[test]
fn matrix_power_one_is_original() -> Result<(), String> {
    let script = fnp_script(
        r#"
A = np.array([[1, 2], [3, 4]], dtype=np.float64)
fnp_pow = fnp.matrix_power(A, 1)
print(np.allclose(fnp_pow, A, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "matrix_power(A, 1) should equal A");
    Ok(())
}
