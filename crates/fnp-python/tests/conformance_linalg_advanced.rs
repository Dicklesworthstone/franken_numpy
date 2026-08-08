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

#[test]
fn matrix_power_bool_bitpacked_bit_exact_matches_numpy() -> Result<(), String> {
    // Bool A^n is OR-AND reachability; the binary-exp driver over the bitpacked
    // bool GEMM must be byte-identical to numpy for every parenthesization-
    // independent case (semiring associativity), across powers, densities, and
    // the n<=1 / small-dim / non-square delegates.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(41)
verdicts = []
for dens in [0.002, 0.02, 0.3]:
    for dim in [64, 97, 200]:
        A = rng.random((dim, dim)) < dens
        for p in [2, 3, 5, 12, 13]:
            r = fnp.matrix_power(A, p)
            e = np.linalg.matrix_power(A, p)
            if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
                verdicts.append(f"FAIL dens={dens} dim={dim} p={p}")
# delegates: n=0 identity, n=1 alias-semantics, small dim, non-square error
A = rng.random((96, 96)) > 0.9
if fnp.matrix_power(A, 0).tobytes() != np.linalg.matrix_power(A, 0).tobytes():
    verdicts.append("FAIL n=0")
if fnp.matrix_power(A, 1).tobytes() != np.linalg.matrix_power(A, 1).tobytes():
    verdicts.append("FAIL n=1")
S = rng.random((16, 16)) > 0.5
if fnp.matrix_power(S, 4).tobytes() != np.linalg.matrix_power(S, 4).tobytes():
    verdicts.append("FAIL small-dim delegate")
try:
    fnp.matrix_power(rng.random((8, 9)) > 0.5, 2)
    verdicts.append("FAIL non-square must raise")
except Exception:
    pass

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

W = rng.random((1024, 1024)) < 0.005
tn = best(lambda: np.linalg.matrix_power(W, 13))
tf = best(lambda: fnp.matrix_power(W, 13))
print(f"MATPOW_BOOL_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces MATPOW_BOOL_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last, "True",
        "bool matrix_power must be bit-identical to numpy incl. delegates: {result}"
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

/// `solve_triangular` on a complex lower-triangular system.
///
/// This was the ONLY test in the crate that took scipy as its oracle, and on any
/// worker without scipy it failed with ModuleNotFoundError — a permanent red that
/// trains people to ignore the shard (bead deadlock-audit-o1p3g).
///
/// It is fixed in two parts, because "don't fail" and "don't silently skip" are
/// different requirements and only doing the first is the vacuous-green trap:
///
///  1. THE ASSERTION NEVER DEGRADES TO NOTHING. NumPy — this project's declared
///     oracle — can decide this case on its own: for an exactly lower-triangular
///     `a`, `np.linalg.solve(a, b)` is the same solution as a triangular solve.
///     Verified against forward substitution and against scipy, all three agree.
///     So the numpy arm always runs and always asserts, scipy or no scipy.
///  2. THE SCIPY CROSS-CHECK IS OPTIONAL AND ITS ABSENCE IS ANNOUNCED. When scipy
///     is missing the script emits a `SKIPPED_ORACLE` line naming exactly what
///     was not compared, and the Rust side re-emits it as a banner on stderr.
///
/// Honest limitation: libtest captures output for PASSING tests, so the banner is
/// visible under `--nocapture`, in any run where the shard fails for another
/// reason, and to a log scan grepping `SKIPPED_ORACLE`. That token is the durable
/// signal — it is deliberately distinctive so coverage loss is greppable rather
/// than invisible.
#[test]
fn solve_triangular_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
bad = []

def check(name, a, b, **kw):
    got = fnp.solve_triangular(a, b, **kw)
    # NumPy is the declared oracle and settles a triangular system by itself, so
    # long as the matrix handed to it is the triangle the call names.
    tri = np.tril(a) if kw.get('lower') else np.triu(a)
    if kw.get('unit_diagonal'):
        tri = tri.copy()
        np.fill_diagonal(tri, 1)
    want = np.linalg.solve(tri, b)
    if not np.allclose(got, want):
        bad.append((name, repr(got), repr(want)))
    return got

a = np.array([[2+1j, 0, 0], [1, 3-1j, 0], [2, 1, 4+1j]], dtype=np.complex128)
b = np.array([1+1j, 2-1j, 3], dtype=np.complex128)
fnp_result = check('lower 1-D', a, b, lower=True)

# The rest of the substitution's surface, each of which a naive implementation
# gets wrong in a different way.
u = np.array([[2+1j, 5, 1], [0, 3-1j, 2], [0, 0, 4+1j]], dtype=np.complex128)
check('upper 1-D', u, b, lower=False)
check('lower unit_diagonal', a, b, lower=True, unit_diagonal=True)
check('upper unit_diagonal', u, b, lower=False, unit_diagonal=True)
# Multiple right-hand sides: catches a column/row indexing slip.
B2 = np.array([[1+1j, 2], [2-1j, 0], [3, 1-3j]], dtype=np.complex128)
check('lower 2-D rhs', a, B2, lower=True)
check('upper 2-D rhs', u, B2, lower=False)
# The OPPOSITE triangle must be IGNORED, not read: poisoning it must not move
# the answer. A solver that reads the whole matrix fails here.
poisoned = a.copy(); poisoned[0, 2] = 999 - 7j; poisoned[0, 1] = -42j
check('lower ignores upper triangle', poisoned, b, lower=True)
# complex64 must stay complex64 on the way out.
a32 = a.astype(np.complex64); b32 = b.astype(np.complex64)
r32 = fnp.solve_triangular(a32, b32, lower=True)
if r32.dtype != np.complex64:
    bad.append(('complex64 dtype', str(r32.dtype), 'complex64'))
if not np.allclose(r32, np.linalg.solve(np.tril(a32), b32), rtol=1e-5, atol=1e-6):
    bad.append(('complex64 value', repr(r32), 'numpy solve'))
# A zero diagonal is singular and must raise, not return inf/nan.
sing = a.copy(); sing[1, 1] = 0
try:
    fnp.solve_triangular(sing, b, lower=True)
    bad.append(('singular', 'returned normally', 'LinAlgError'))
except Exception as exc:
    if 'LinAlgError' not in type(exc).__name__:
        bad.append(('singular', type(exc).__name__, 'LinAlgError'))

print('True' if not bad else f'MISMATCH {bad}')

try:
    import scipy.linalg
except ImportError as exc:
    print(f"SKIPPED_ORACLE scipy.linalg.solve_triangular ({type(exc).__name__}: {exc}) "
          f"- NOT cross-checked: scipy's lower=True triangular solve. The numpy arm above "
          f"DID run and did assert; only the second opinion is missing.")
else:
    sp_result = scipy.linalg.solve_triangular(a, b, lower=True)
    print("SCIPY_CROSSCHECK " + ("ok" if np.allclose(fnp_result, sp_result) else "MISMATCH"))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.lines();
    let numpy_arm = lines.next().unwrap_or_default().trim();
    let scipy_arm = lines.next().unwrap_or_default().trim();

    assert_eq!(
        numpy_arm, "True",
        "solve_triangular complex must match numpy's own solve; output: {result}"
    );
    if let Some(note) = scipy_arm.strip_prefix("SKIPPED_ORACLE") {
        eprintln!("SKIPPED_ORACLE [solve_triangular_complex]{note}");
    } else {
        assert_eq!(
            scipy_arm, "SCIPY_CROSSCHECK ok",
            "scipy cross-check ran and disagreed; output: {result}"
        );
    }
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

#[test]
fn f16_multi_dot_three_chain_bit_exact() -> Result<(), String> {
    // f16 3-array multi_dot: numpy's _multi_dot_three order rule (documented
    // cost arithmetic, byte-stable on numpy 2.2.4/2.3.5/2.4.3/2.4.6) + matmul
    // pairs. fnp replicates the rule and routes both pairs through the
    // shipped byte-matched f16 matmul kernel. Shapes exercise BOTH orders,
    // MR tails, and the below-gate fallback.
    let script = fnp_script(
        r#"
verdicts = []
rng = np.random.default_rng(20260715)
for (p0, p1, p2, p3) in ((96, 96, 96, 96), (24, 8, 400, 12), (12, 400, 8, 24), (130, 64, 96, 70), (65, 33, 129, 17)):
    a = (rng.standard_normal((p0, p1)) * 0.3).astype(np.float16)
    b = (rng.standard_normal((p1, p2)) * 0.3).astype(np.float16)
    c = (rng.standard_normal((p2, p3)) * 0.3).astype(np.float16)
    r = fnp.multi_dot([a, b, c]); e = np.linalg.multi_dot([a, b, c])
    if r.dtype != e.dtype or r.shape != e.shape:
        verdicts.append(f"FAIL ({p0},{p1},{p2},{p3}) shape/dtype")
    elif not bool(((r.view(np.uint16) == e.view(np.uint16)) | (np.isnan(r) & np.isnan(e))).all()):
        verdicts.append(f"FAIL ({p0},{p1},{p2},{p3}) bytes")
# below-gate + 1-D endpoints + 4-array + f64 all defer byte-exactly
sm = (rng.standard_normal((8, 8)) * 0.3).astype(np.float16)
if fnp.multi_dot([sm, sm, sm]).tobytes() != np.linalg.multi_dot([sm, sm, sm]).tobytes():
    verdicts.append("FAIL below-gate")
v = (rng.standard_normal(96) * 0.3).astype(np.float16)
m1 = (rng.standard_normal((96, 96)) * 0.3).astype(np.float16)
if fnp.multi_dot([v, m1, m1]).tobytes() != np.linalg.multi_dot([v, m1, m1]).tobytes():
    verdicts.append("FAIL 1-D endpoint defer")
if fnp.multi_dot([m1, m1, m1, m1]).tobytes() != np.linalg.multi_dot([m1, m1, m1, m1]).tobytes():
    verdicts.append("FAIL 4-array defer")
d64 = rng.standard_normal((96, 96))
if fnp.multi_dot([d64, d64, d64]).tobytes() != np.linalg.multi_dot([d64, d64, d64]).tobytes():
    verdicts.append("FAIL f64 defer")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "f16 multi_dot 3-chain must be bit-identical via the order rule + matmul pairs: {result}"
    );
    Ok(())
}

#[test]
fn matrix_power_noncontig_base_bit_exact_matches_numpy() -> Result<(), String> {
    // Transposed/strided int and bool bases now contiguate post-gate and route
    // the native binary-exp GEMM chain; results must be byte-identical to
    // numpy across dtypes, powers, and layouts (F-order, strided views).
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(127)
verdicts = []
A = rng.integers(-10, 10, (256, 256))
cases = [
    ("A.T int64", np.ascontiguousarray(A).T),
    ("F-order int64", np.asfortranarray(A)),
    ("strided int64", rng.integers(-10, 10, (256, 512))[:, ::2]),
]
for name, M in cases:
    for p in (2, 3, 5, 13):
        r = fnp.matrix_power(M, p)
        e = np.linalg.matrix_power(M, p)
        if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
            verdicts.append(f"FAIL {name} p={p}")
Bb = (rng.random((256, 256)) < 0.01)
Bt = np.ascontiguousarray(Bb).T
for p in (2, 5, 13):
    if fnp.matrix_power(Bt, p).tobytes() != np.linalg.matrix_power(Bt, p).tobytes():
        verdicts.append(f"FAIL bool A.T p={p}")

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

W = rng.integers(-10, 10, (512, 512))
Wt = np.ascontiguousarray(W).T
tn = best(lambda: np.linalg.matrix_power(Wt, 5))
tf = best(lambda: fnp.matrix_power(Wt, 5))
print(f"MATPOW_INT_NC_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces MATPOW_INT_NC_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last, "True",
        "non-contiguous-base matrix_power must be bit-identical to numpy: {result}"
    );
    Ok(())
}

// numpy.linalg.diagonal / trace / outer / tensordot share a NAME with a
// top-level numpy function but are NOT the same function - they are Array-API
// variants. diagonal and trace take the LAST TWO axes where np.diagonal and
// np.trace take axes 0 and 1, so aliasing the top-level wrapper under the
// linalg name returned the wrong array for ndim > 2 while looking correct for
// 2-D. This pins all four against numpy.linalg, pins the top-level four as
// UNCHANGED, and - critically - asserts the two conventions actually DIFFER on
// the 3-D input used, without which the whole test could pass on an alias.
#[test]
fn linalg_namespace_functions_are_not_the_toplevel_ones() -> Result<(), String> {
    let script = fnp_script(
        r#"
import platform

cube = np.arange(24).reshape(2, 3, 4)
square = np.arange(9).reshape(3, 3)
vec = np.array([1.0, 2.0, 3.0])
mat = np.ones((2, 2))

def described(value):
    array = np.asarray(value)
    return (str(array.dtype), tuple(array.shape), array.tolist())

def linalg_diagonal_3d(module):
    return described(module.linalg.diagonal(cube))

def linalg_diagonal_offset(module):
    return described(module.linalg.diagonal(cube, offset=1))

def linalg_diagonal_2d(module):
    return described(module.linalg.diagonal(square))

def linalg_diagonal_positional_offset(module):
    # numpy makes offset keyword-only, so this must raise on both sides.
    return described(module.linalg.diagonal(cube, 1))

def linalg_diagonal_axis1(module):
    return described(module.linalg.diagonal(cube, axis1=0))

def linalg_trace_3d(module):
    return described(module.linalg.trace(cube))

def linalg_trace_offset(module):
    return described(module.linalg.trace(cube, offset=1))

def linalg_trace_out(module):
    return described(module.linalg.trace(cube, out=None))

def linalg_trace_axes(module):
    return described(module.linalg.trace(cube, axis1=0, axis2=1))

def linalg_outer_1d(module):
    return described(module.linalg.outer(vec, vec))

def linalg_outer_2d_rejected(module):
    # np.outer flattens 2-D input; np.linalg.outer REQUIRES 1-D.
    return described(module.linalg.outer(mat, mat))

def linalg_outer_out(module):
    return described(module.linalg.outer(vec, vec, out=None))

def linalg_tensordot_keyword(module):
    return described(module.linalg.tensordot(mat, mat, axes=2))

def linalg_tensordot_positional(module):
    # numpy makes axes keyword-only under linalg.
    return described(module.linalg.tensordot(mat, mat, 2))

def linalg_matrix_rank_capital(module):
    return described(module.linalg.matrix_rank(A=square))

def linalg_matrix_rank_lowercase(module):
    return described(module.linalg.matrix_rank(a=square))

# The top-level four must be untouched by the linalg fix.
def toplevel_diagonal(module):
    return described(module.diagonal(cube))

def toplevel_trace(module):
    return described(module.trace(cube))

def toplevel_outer_2d(module):
    return described(module.outer(mat, mat))

def toplevel_tensordot_positional(module):
    return described(module.tensordot(mat, mat, 2))

cases = [
    ("linalg.diagonal 3-D", linalg_diagonal_3d),
    ("linalg.diagonal offset=", linalg_diagonal_offset),
    ("linalg.diagonal 2-D", linalg_diagonal_2d),
    ("linalg.diagonal positional offset", linalg_diagonal_positional_offset),
    ("linalg.diagonal axis1=", linalg_diagonal_axis1),
    ("linalg.trace 3-D", linalg_trace_3d),
    ("linalg.trace offset=", linalg_trace_offset),
    ("linalg.trace out=", linalg_trace_out),
    ("linalg.trace axis1=/axis2=", linalg_trace_axes),
    ("linalg.outer 1-D", linalg_outer_1d),
    ("linalg.outer 2-D rejected", linalg_outer_2d_rejected),
    ("linalg.outer out=", linalg_outer_out),
    ("linalg.tensordot axes= keyword", linalg_tensordot_keyword),
    ("linalg.tensordot positional axes", linalg_tensordot_positional),
    ("linalg.matrix_rank A=", linalg_matrix_rank_capital),
    ("linalg.matrix_rank a=", linalg_matrix_rank_lowercase),
    ("top-level diagonal 3-D", toplevel_diagonal),
    ("top-level trace 3-D", toplevel_trace),
    ("top-level outer 2-D", toplevel_outer_2d),
    ("top-level tensordot positional", toplevel_tensordot_positional),
]

def outcome(module, call):
    try:
        return ("ok", call(module))
    except Exception as exc:
        return ("err", type(exc).__name__)

ok = True
for label, call in cases:
    actual = outcome(fnp, call)
    expected = outcome(np, call)
    if actual != expected:
        print(label)
        print(actual)
        print(expected)
        ok = False

# Guard against the whole test passing on an alias: the linalg and top-level
# conventions MUST differ on this input, or every case above is vacuous.
if np.array_equal(np.linalg.diagonal(cube), np.diagonal(cube)):
    print("PRECONDITION LOST: linalg.diagonal and np.diagonal agree on the 3-D input")
    ok = False
if np.array_equal(np.linalg.trace(cube), np.trace(cube)):
    print("PRECONDITION LOST: linalg.trace and np.trace agree on the 3-D input")
    ok = False

print(ok)
print("oracle", platform.node(), np.__version__)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.trim().lines().rev();
    let provenance = lines.next().unwrap_or("").trim();
    let verdict = lines.next().unwrap_or("").trim();
    assert_eq!(
        verdict, "True",
        "linalg namespace functions should match numpy.linalg ({provenance}): {result}"
    );
    Ok(())
}
