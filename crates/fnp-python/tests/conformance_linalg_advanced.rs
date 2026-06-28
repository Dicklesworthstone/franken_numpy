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

// ─────────────────────────────────────────────────────────────────────────────
// Complex matrix tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn pinv_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1+1j, 2], [3, 4-1j], [5+2j, 6]], dtype=np.complex128)
fnp_pinv = fnp.pinv(a)
np_pinv = np.linalg.pinv(a)
print(np.allclose(fnp_pinv, np_pinv))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "pinv complex should match numpy");
    Ok(())
}

#[test]
fn tensorinv_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Create a tensor that can be inverted: shape (2,3,6) with ind=2 means 2*3=6
a = np.arange(36, dtype=np.complex128).reshape(2, 3, 6) + 1j
# Make it more invertible by adding scaled identity-like structure
for i in range(6):
    a.flat[i * 7] += 10
fnp_result = fnp.tensorinv(a, ind=2)
np_result = np.linalg.tensorinv(a, ind=2)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tensorinv complex should match numpy"
    );
    Ok(())
}

#[test]
fn solve_triangular_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
import scipy.linalg
a = np.array([[2+1j, 0, 0], [1, 3-1j, 0], [2, 1, 4+1j]], dtype=np.complex128)
b = np.array([1+1j, 2-1j, 3], dtype=np.complex128)
fnp_result = fnp.solve_triangular(a, b, lower=True)
sp_result = scipy.linalg.solve_triangular(a, b, lower=True)
print(np.allclose(fnp_result, sp_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "solve_triangular complex should match scipy"
    );
    Ok(())
}

#[test]
fn eigvals_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1+1j, 2], [3, 4-1j]], dtype=np.complex128)
fnp_vals = fnp.eigvals(a)
np_vals = np.linalg.eigvals(a)
# Eigenvalues may be in different order, so compare sorted
fnp_sorted = np.sort_complex(fnp_vals)
np_sorted = np.sort_complex(np_vals)
print(np.allclose(fnp_sorted, np_sorted))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "eigvals complex should match numpy");
    Ok(())
}

#[test]
fn svd_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1+1j, 2], [3, 4-1j], [5+2j, 6]], dtype=np.complex128)
fnp_u, fnp_s, fnp_vh = fnp.svd(a)
np_u, np_s, np_vh = np.linalg.svd(a)
# Singular values should match
print(np.allclose(fnp_s, np_s))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "svd complex singular values should match numpy"
    );
    Ok(())
}

#[test]
fn slogdet_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1+1j, 2], [3, 4-1j]], dtype=np.complex128)
fnp_sign, fnp_logdet = fnp.slogdet(a)
np_sign, np_logdet = np.linalg.slogdet(a)
print(np.allclose(fnp_logdet, np_logdet))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "slogdet complex logdet should match numpy"
    );
    Ok(())
}

#[test]
fn svd_empty_rows() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([]).reshape(0, 3)
fnp_u, fnp_s, fnp_vh = fnp.svd(a)
np_u, np_s, np_vh = np.linalg.svd(a)
# Shapes should match
shape_ok = fnp_u.shape == np_u.shape and fnp_s.shape == np_s.shape and fnp_vh.shape == np_vh.shape
print(shape_ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "svd empty rows shapes should match numpy"
    );
    Ok(())
}

#[test]
fn pinv_empty() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([]).reshape(0, 3)
fnp_result = fnp.pinv(a)
np_result = np.linalg.pinv(a)
print(fnp_result.shape == np_result.shape and np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "pinv empty should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Error behavior tests
// ─────────────────────────────────────────────────────────────────────────────

fn classify_error(script: &str) -> String {
    let output = std::process::Command::new("python3")
        .args(["-c", script])
        .output()
        .expect("python3 should be available");
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
fn matrix_power_non_square_raises_linalgerror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(6).reshape(2, 3).astype(float)
fnp.linalg.matrix_power(a, 2)
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3).astype(float)
np.linalg.matrix_power(a, 2)
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "matrix_power on non-square should raise same error as numpy"
    );
}

#[test]
fn eigvals_non_square_raises_linalgerror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(6).reshape(2, 3).astype(float)
fnp.linalg.eigvals(a)
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3).astype(float)
np.linalg.eigvals(a)
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "eigvals on non-square should raise same error as numpy"
    );
}

#[test]
fn int_matrix_power_native_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy integer matrix_power = repeated naive int matmul (no BLAS). The native
    // binary-exp parallel GEMM must be byte-identical (Z/2^w ring assoc) incl. overflow
    // wrap, across powers and int widths.
    let script = fnp_script(
        r#"
rng = np.random.default_rng(19)
ok = True
for dt in [np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32]:
    M = rng.integers(-3, 4, (96, 96)).astype(dt)
    for p in [2, 3, 5, 8, 13]:
        r = fnp.matrix_power(M, p); e = np.linalg.matrix_power(M, p)
        ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# explicit overflow wrap (int64, values grow fast)
M = rng.integers(100000, 200000, (80, 80)).astype(np.int64)
for p in [2, 4, 7]:
    ok = ok and fnp.matrix_power(M, p).tobytes() == np.linalg.matrix_power(M, p).tobytes()
# n==1 and n==0 still match (delegated paths)
M = rng.integers(-5, 5, (70, 70)).astype(np.int64)
ok = ok and fnp.matrix_power(M, 1).tobytes() == np.linalg.matrix_power(M, 1).tobytes()
ok = ok and fnp.matrix_power(M, 0).tobytes() == np.linalg.matrix_power(M, 0).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native integer matrix_power must be bit-identical to numpy: {result}"
    );
    Ok(())
}
