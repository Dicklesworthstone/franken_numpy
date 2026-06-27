//! Linear algebra operations for FrankenNumPy.
//!
//! This crate provides NumPy-compatible linear algebra functions including
//! matrix decompositions, solvers, and eigenvalue computations.
//!
//! # Functions
//!
//! - **Decompositions**: `svd`, `qr`, `cholesky`, `lu`
//! - **Solvers**: `solve`, `lstsq`, `pinv`
//! - **Eigenvalues**: `eig`, `eigh`, `eigvals`, `eigvalsh`
//! - **Norms**: `norm`, `cond`, `matrix_rank`
//! - **Determinants**: `det`, `slogdet`
//!
//! # Memory Safety
//!
//! Uses `#![forbid(unsafe_code)]` - implemented in safe Rust.

#![forbid(unsafe_code)]
#![feature(portable_simd)]

use core::fmt;
use rayon::prelude::*;

/// Result of `lstsq_svd`: `(x, residuals, rank, singular_values)`.
pub type LstsqResult = (Vec<f64>, Vec<f64>, usize, Vec<f64>);

pub const LINALG_PACKET_ID: &str = "FNP-P2C-008";
/// Maximum recursive depth when searching for an optimal rcond tolerance.
/// Prevents stack overflow in degenerate tolerance-tuning scenarios.
pub const MAX_TOLERANCE_SEARCH_DEPTH: usize = 128;
/// Maximum retry attempts for backend solver revalidation after a transient failure.
/// Covers scenarios like temporary numerical instability from ill-conditioned inputs.
pub const MAX_BACKEND_REVALIDATION_ATTEMPTS: usize = 64;
/// Maximum shape compatibility checks for batch operations. Prevents O(n^2) broadcast
/// validation from hanging on arrays with extremely many batch dimensions.
pub const MAX_BATCH_SHAPE_CHECKS: usize = 2_000_000;

/// Iteration limit coefficient for SVD bidiagonal QR convergence: max_iters = coeff * n * n.
pub const SVD_QR_ITERATION_COEFF: usize = 100;
/// Large full-SVD threshold where reconstructing U from A·V·Σ⁻¹ is cheaper
/// than accumulating every left Householder/Givens rotation into an m×m matrix.
#[cfg(not(test))]
const SVD_RECONSTRUCT_LEFT_MIN_DIM: usize = 128;
#[cfg(test)]
const SVD_RECONSTRUCT_LEFT_MIN_DIM: usize = 32;
/// Minimum remaining trailing width for the fused two-sided SVD update. Below
/// this size the simpler scalar sweeps win by avoiding the extra branch surface.
const SVD_FUSED_TWO_SIDED_MIN_TRAIL: usize = 16;
const SVD_FUSED_TWO_SIDED_REGISTER_BLOCK: usize = 4;
/// Large-square full-SVD threshold where phase-1 right reflectors are accumulated
/// into Vt by compact-WY panels instead of one scalar reflector at a time.
const SVD_RIGHT_VT_BLOCK_MIN_DIM: usize = 512;
const SVD_RIGHT_VT_PANEL_NB: usize = 128;
/// Iteration limit coefficient for eigenvalue/Schur QR convergence: max_iters = coeff * n * n.
pub const EIGEN_QR_ITERATION_COEFF: usize = 60;
/// Maximum iterations for matrix square root (Denman-Beavers).
pub const SQRTM_MAX_ITERATIONS: usize = 50;
/// Maximum Taylor series terms for matrix exponential / logarithm.
pub const MATRIX_FUNC_TAYLOR_TERMS: usize = 30;
/// Maximum scaling iterations for matrix logarithm.
pub const LOGM_MAX_SCALING_ITERATIONS: usize = 20;

pub const LINALG_PACKET_REASON_CODES: [&str; 10] = [
    "linalg_shape_contract_violation",
    "linalg_solver_singularity",
    "linalg_cholesky_contract_violation",
    "linalg_qr_mode_invalid",
    "linalg_svd_nonconvergence",
    "linalg_spectral_convergence_failed",
    "linalg_lstsq_tuple_contract_violation",
    "linalg_norm_det_rank_policy_violation",
    "linalg_backend_bridge_invalid",
    "linalg_policy_unknown_metadata",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinAlgRuntimeMode {
    Strict,
    Hardened,
}

impl LinAlgRuntimeMode {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Hardened => "hardened",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QrMode {
    Reduced,
    Complete,
    R,
    Raw,
}

impl QrMode {
    pub fn from_mode_token(mode: &str) -> Result<Self, LinAlgError> {
        match mode {
            "reduced" => Ok(Self::Reduced),
            "complete" => Ok(Self::Complete),
            "r" => Ok(Self::R),
            "raw" => Ok(Self::Raw),
            _ => Err(LinAlgError::QrModeInvalid),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VectorNormOrder {
    Zero,
    One,
    Two,
    Inf,
    NegInf,
    P(f64),
}

impl VectorNormOrder {
    pub fn from_token(token: &str) -> Result<Self, LinAlgError> {
        match token.trim().to_ascii_lowercase().as_str() {
            "0" => Ok(Self::Zero),
            "1" => Ok(Self::One),
            "2" => Ok(Self::Two),
            "inf" | "+inf" => Ok(Self::Inf),
            "-inf" => Ok(Self::NegInf),
            other => {
                if let Ok(p) = other.parse::<f64>() {
                    Ok(Self::P(p))
                } else {
                    Err(LinAlgError::NormDetRankPolicyViolation(
                        "unsupported vector norm order token",
                    ))
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixNormOrder {
    Fro,
    One,
    NegOne,
    Inf,
    NegInf,
    Two,
    NegTwo,
    Nuclear,
}

impl MatrixNormOrder {
    pub fn from_token(token: &str) -> Result<Self, LinAlgError> {
        match token.trim().to_ascii_lowercase().as_str() {
            "fro" | "f" => Ok(Self::Fro),
            "1" => Ok(Self::One),
            "-1" => Ok(Self::NegOne),
            "inf" | "+inf" => Ok(Self::Inf),
            "-inf" => Ok(Self::NegInf),
            "2" => Ok(Self::Two),
            "-2" => Ok(Self::NegTwo),
            "nuc" => Ok(Self::Nuclear),
            _ => Err(LinAlgError::NormDetRankPolicyViolation(
                "unsupported matrix norm order token",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinAlgError {
    ShapeContractViolation(&'static str),
    SolverSingularity,
    CholeskyContractViolation(&'static str),
    QrModeInvalid,
    SvdNonConvergence,
    SpectralConvergenceFailed,
    LstsqTupleContractViolation(&'static str),
    NormDetRankPolicyViolation(&'static str),
    BackendBridgeInvalid(&'static str),
    PolicyUnknownMetadata(&'static str),
}

impl LinAlgError {
    #[must_use]
    pub fn reason_code(&self) -> &'static str {
        match self {
            Self::ShapeContractViolation(_) => "linalg_shape_contract_violation",
            Self::SolverSingularity => "linalg_solver_singularity",
            Self::CholeskyContractViolation(_) => "linalg_cholesky_contract_violation",
            Self::QrModeInvalid => "linalg_qr_mode_invalid",
            Self::SvdNonConvergence => "linalg_svd_nonconvergence",
            Self::SpectralConvergenceFailed => "linalg_spectral_convergence_failed",
            Self::LstsqTupleContractViolation(_) => "linalg_lstsq_tuple_contract_violation",
            Self::NormDetRankPolicyViolation(_) => "linalg_norm_det_rank_policy_violation",
            Self::BackendBridgeInvalid(_) => "linalg_backend_bridge_invalid",
            Self::PolicyUnknownMetadata(_) => "linalg_policy_unknown_metadata",
        }
    }
}

impl fmt::Display for LinAlgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SolverSingularity => write!(f, "solve/inv rejected singular matrix"),
            Self::QrModeInvalid => write!(f, "qr mode is not one of reduced|complete|r|raw"),
            Self::SvdNonConvergence => write!(f, "svd did not converge"),
            Self::SpectralConvergenceFailed => write!(f, "spectral decomposition did not converge"),
            Self::ShapeContractViolation(msg)
            | Self::CholeskyContractViolation(msg)
            | Self::LstsqTupleContractViolation(msg)
            | Self::NormDetRankPolicyViolation(msg)
            | Self::BackendBridgeInvalid(msg)
            | Self::PolicyUnknownMetadata(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for LinAlgError {}

/// Full SVD result: (U, singular values, Vt).
pub type SvdFullResult = (Vec<f64>, Vec<f64>, Vec<f64>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QrOutputShapes {
    pub q_shape: Option<Vec<usize>>,
    pub r_shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvdOutputShapes {
    pub u_shape: Vec<usize>,
    pub s_shape: Vec<usize>,
    pub vh_shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LstsqOutputShapes {
    pub x_shape: Vec<usize>,
    pub residuals_shape: Vec<usize>,
    pub rank_upper_bound: usize,
    pub singular_values_shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qr2x2Result {
    pub q: Option<[[f64; 2]; 2]>,
    pub r: [[f64; 2]; 2],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Svd2x2Result {
    pub u: [[f64; 2]; 2],
    pub singular_values: [f64; 2],
    pub vt: [[f64; 2]; 2],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Lstsq2x2Result {
    pub solution: [f64; 2],
    pub residual_sum_squares: f64,
    pub rank: usize,
    pub singular_values: [f64; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinAlgLogRecord {
    pub ts_utc: String,
    pub suite_id: String,
    pub test_id: String,
    pub packet_id: String,
    pub fixture_id: String,
    pub mode: LinAlgRuntimeMode,
    pub seed: u64,
    pub input_digest: String,
    pub output_digest: String,
    pub env_fingerprint: String,
    pub artifact_refs: Vec<String>,
    pub duration_ms: u64,
    pub outcome: String,
    pub reason_code: String,
}

impl LinAlgLogRecord {
    #[must_use]
    pub fn is_replay_complete(&self) -> bool {
        if self.ts_utc.trim().is_empty()
            || self.suite_id.trim().is_empty()
            || self.test_id.trim().is_empty()
            || self.packet_id.trim().is_empty()
            || self.fixture_id.trim().is_empty()
            || self.input_digest.trim().is_empty()
            || self.output_digest.trim().is_empty()
            || self.env_fingerprint.trim().is_empty()
            || self.reason_code.trim().is_empty()
        {
            return false;
        }

        if self.packet_id != LINALG_PACKET_ID {
            return false;
        }

        if self.outcome != "pass" && self.outcome != "fail" {
            return false;
        }

        if self.artifact_refs.is_empty()
            || self
                .artifact_refs
                .iter()
                .any(|artifact| artifact.trim().is_empty())
        {
            return false;
        }

        LINALG_PACKET_REASON_CODES
            .iter()
            .any(|code| *code == self.reason_code)
    }
}

pub fn validate_matrix_shape(shape: &[usize]) -> Result<(usize, usize), LinAlgError> {
    if shape.len() < 2 {
        return Err(LinAlgError::ShapeContractViolation(
            "linalg input must be at least 2D",
        ));
    }

    let rows = shape[shape.len() - 2];
    let cols = shape[shape.len() - 1];
    if rows == 0 || cols == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "matrix rows/cols must be non-zero",
        ));
    }

    let batch_rank = shape.len() - 2;
    let batch_lanes = match batch_rank {
        0 => 1usize,
        1 => shape[0],
        2 => shape[0]
            .checked_mul(shape[1])
            .ok_or(LinAlgError::ShapeContractViolation(
                "batch lane multiplication overflowed",
            ))?,
        _ => shape[..batch_rank]
            .iter()
            .copied()
            .try_fold(1usize, |acc, dim| acc.checked_mul(dim))
            .ok_or(LinAlgError::ShapeContractViolation(
                "batch lane multiplication overflowed",
            ))?,
    };

    if batch_lanes > MAX_BATCH_SHAPE_CHECKS {
        return Err(LinAlgError::ShapeContractViolation(
            "batch lanes exceeded bounded validation budget",
        ));
    }

    Ok((rows, cols))
}

pub fn validate_square_matrix(shape: &[usize]) -> Result<usize, LinAlgError> {
    let (rows, cols) = validate_matrix_shape(shape)?;
    if rows != cols {
        return Err(LinAlgError::ShapeContractViolation(
            "square matrix required for solve/inv/cholesky",
        ));
    }
    Ok(rows)
}

pub fn solve_2x2(lhs: [[f64; 2]; 2], rhs: [f64; 2]) -> Result<[f64; 2], LinAlgError> {
    let det = lhs[0][0] * lhs[1][1] - lhs[0][1] * lhs[1][0];
    if det == 0.0 {
        return Err(LinAlgError::SolverSingularity);
    }

    let inv_det = 1.0 / det;
    let x0 = (rhs[0] * lhs[1][1] - lhs[0][1] * rhs[1]) * inv_det;
    let x1 = (lhs[0][0] * rhs[1] - rhs[0] * lhs[1][0]) * inv_det;
    Ok([x0, x1])
}

fn validate_finite_matrix_2x2(matrix: [[f64; 2]; 2]) -> Result<(), LinAlgError> {
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "matrix entries must be finite for norm/det/rank/pinv operations",
        ));
    }
    Ok(())
}

pub fn det_2x2(matrix: [[f64; 2]; 2]) -> Result<f64, LinAlgError> {
    Ok(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])
}

pub fn slogdet_2x2(matrix: [[f64; 2]; 2]) -> Result<(f64, f64), LinAlgError> {
    let flat = [matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]];
    let (lu, _, sign) = match lu_decompose_for_det(&flat, 2) {
        Ok(parts) => parts,
        Err(LinAlgError::SolverSingularity) => return Ok((0.0, f64::NEG_INFINITY)),
        Err(err) => return Err(err),
    };
    let mut det_sign = sign;
    let mut log_abs_det = 0.0;
    for i in 0..2 {
        let diag = lu[i * 2 + i];
        if diag.is_nan() {
            return Ok((det_sign, f64::NAN));
        }
        if diag < 0.0 {
            det_sign = -det_sign;
            log_abs_det += (-diag).ln();
        } else if diag > 0.0 {
            log_abs_det += diag.ln();
        } else {
            return Ok((0.0, f64::NEG_INFINITY));
        }
    }
    Ok((det_sign, log_abs_det))
}

pub fn inv_2x2(matrix: [[f64; 2]; 2]) -> Result<[[f64; 2]; 2], LinAlgError> {
    let det = det_2x2(matrix)?;
    if det == 0.0 {
        return Err(LinAlgError::SolverSingularity);
    }

    let inv_det = 1.0 / det;
    Ok([
        [matrix[1][1] * inv_det, -matrix[0][1] * inv_det],
        [-matrix[1][0] * inv_det, matrix[0][0] * inv_det],
    ])
}

/// LU decomposition with partial pivoting.  PA = LU where P is a row
/// permutation, L is unit-lower-triangular, and U is upper-triangular.
/// Returns (lu, perm, sign) with L and U packed into one flat row-major
/// buffer.  `perm[i]` records the original row index that ended up at
/// position i after pivoting; `sign` is +1 or -1.
fn lu_decompose_inner(
    a: &[f64],
    n: usize,
    reject_non_finite: bool,
) -> Result<(Vec<f64>, Vec<usize>, f64), LinAlgError> {
    if Some(a.len()) != n.checked_mul(n) || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "LU input must be n*n with n > 0",
        ));
    }
    if reject_non_finite && a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::SolverSingularity);
    }

    let matrix_max_abs = a.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let singularity_threshold = if matrix_max_abs.is_finite() {
        (n as f64) * f64::EPSILON * matrix_max_abs
    } else {
        0.0
    };

    // Large finite systems use the blocked right-looking factorization, which
    // routes the trailing-submatrix update through the cache-blocked packed GEMM
    // (compute-bound) instead of the memory-bound rank-1 sweep. Identical pivot
    // sequence to the unblocked path; the GEMM re-associates the trailing update,
    // so results match within tolerance (LU is not bit-reproducible — conformance
    // is tolerance-based, like NumPy's LAPACK). NaN/Inf inputs (the det path with
    // reject_non_finite=false) keep the unblocked loop and its LAPACK NaN-pivot
    // passthrough behaviour.
    if n >= LU_BLOCK_MIN && !a.iter().any(|v| !v.is_finite()) {
        return lu_decompose_blocked(a, n, singularity_threshold);
    }

    let mut lu = vec![0.0f64; n * n];
    let mut perm: Vec<usize> = vec![0; n];
    let sign = lu_factor_unblocked_into(a, n, singularity_threshold, &mut lu, &mut perm)?;
    Ok((lu, perm, sign))
}

/// Core unblocked right-looking LU with partial pivoting, writing into caller-owned
/// `lu` (n*n) and `perm` (n) scratch buffers (so a batched solve can REUSE them across
/// lanes instead of allocating per call). `lu` is overwritten with a copy of `a` first;
/// `perm` is reset to the identity. Returns the permutation sign. The arithmetic is
/// identical to the previous inline loop in `lu_decompose_inner`, so every caller
/// (det/inv/slogdet/solve) is byte-for-byte unchanged (locked by their golden tests).
fn lu_factor_unblocked_into(
    a: &[f64],
    n: usize,
    singularity_threshold: f64,
    lu: &mut [f64],
    perm: &mut [usize],
) -> Result<f64, LinAlgError> {
    lu.copy_from_slice(a);
    for (i, p) in perm.iter_mut().enumerate() {
        *p = i;
    }
    let mut sign = 1.0_f64;

    for k in 0..n {
        // partial-pivot search
        let mut max_val = lu[k * n + k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let val = lu[i * n + k].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }

        if max_val <= singularity_threshold {
            return Err(LinAlgError::SolverSingularity);
        }

        if max_row != k {
            for j in 0..n {
                lu.swap(k * n + j, max_row * n + j);
            }
            perm.swap(k, max_row);
            sign = -sign;
        }

        let pivot = lu[k * n + k];

        // LAPACK dgetrf compatibility: if pivot is NaN, it doesn't divide or eliminate,
        // it just leaves the trailing matrix alone and proceeds. This produces the exact
        // same garbage LU factorization that NumPy/SciPy return for matrices with NaNs.
        if pivot.is_nan() {
            continue;
        }

        // Rank-1 trailing-submatrix update (serial; memory-bound).
        for i in (k + 1)..n {
            let factor = lu[i * n + k] / pivot;
            lu[i * n + k] = factor;
            for j in (k + 1)..n {
                let u_val = lu[k * n + j];
                lu[i * n + j] -= factor * u_val;
            }
        }
    }

    Ok(sign)
}

#[inline]
fn lu_factor_for_det_into(
    a: &[f64],
    n: usize,
    lu: &mut [f64],
    perm: &mut [usize],
) -> Result<f64, LinAlgError> {
    let matrix_max_abs = a.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let singularity_threshold = if matrix_max_abs.is_finite() {
        (n as f64) * f64::EPSILON * matrix_max_abs
    } else {
        0.0
    };
    lu_factor_unblocked_into(a, n, singularity_threshold, lu, perm)
}

// Engage blocked LU once the parallel trailing GEMM beats the unblocked rank-1 sweep.
// Fresh same-load Criterion A/B (RAYON_NUM_THREADS=16, det_nxn) shows the blocked
// level-3 path already wins from n=512 — the old 896 cutoff left the whole 512..896
// band on the memory-bound serial sweep:
//   n=512  serial 8.77ms -> blocked 6.35ms  (1.38x)
//   n=640  serial 16.67  -> blocked 11.01   (1.51x)
//   n=768  serial 28.82  -> blocked 16.51   (1.75x)
//   n=896  serial 45.75  -> blocked 23.76   (1.93x)
// The win grows with n (the trailing GEMM amortizes the serial panel/extraction
// overhead). Cut over at 512, where the margin (1.38x) is already well clear of
// worker noise; below 512 the panel overhead makes the unblocked sweep competitive
// (n=384 was only ~1.15x). Pivot sequence is identical, so results match within
// tolerance (LU is not bit-reproducible — conformance is tolerance-based like LAPACK).
const LU_BLOCK_MIN: usize = 512;
// A 64-wide panel cuts the serial pivot/panel fraction at n=512 enough to beat
// the 128-wide GEMM-parallel panel. The trailing update may run serial at this k,
// but the smaller panel keeps more total work in cache and wins on the det/LU
// mid-size gate.
const LU_PANEL_NB: usize = 64;

// Right-looking blocked LU with partial pivoting (LAPACK dgetrf shape). For each
// column panel of width nb: factor the panel (full-column pivoting, row swaps
// across the whole matrix, rank-1 updates confined to the panel), triangular-
// solve the U12 block-row against the unit-lower L11, then update the trailing
// submatrix A22 -= L21·U12 with the packed GEMM. The pivot sequence is identical
// to the unblocked loop (each column is fully updated before its pivot search),
// so the factorization is numerically equivalent up to the GEMM's re-association
// (tolerance — never bit-exact). Caller guarantees all-finite input.
fn lu_decompose_blocked(
    a: &[f64],
    n: usize,
    singularity_threshold: f64,
) -> Result<(Vec<f64>, Vec<usize>, f64), LinAlgError> {
    let mut lu = a.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();
    let mut sign = 1.0_f64;

    let mut jb = 0;
    while jb < n {
        let bw = LU_PANEL_NB.min(n - jb);
        let panel_end = jb + bw;

        // (1) Panel factorization: columns [jb, panel_end).
        for k in jb..panel_end {
            let mut max_val = lu[k * n + k].abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                let val = lu[i * n + k].abs();
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }
            if max_val <= singularity_threshold {
                return Err(LinAlgError::SolverSingularity);
            }
            if max_row != k {
                for j in 0..n {
                    lu.swap(k * n + j, max_row * n + j);
                }
                perm.swap(k, max_row);
                sign = -sign;
            }
            let pivot = lu[k * n + k];
            // Multipliers + rank-1 update confined to the panel columns.
            for i in (k + 1)..n {
                let factor = lu[i * n + k] / pivot;
                lu[i * n + k] = factor;
                for j in (k + 1)..panel_end {
                    let u_val = lu[k * n + j];
                    lu[i * n + j] -= factor * u_val;
                }
            }
        }

        let trail = n - panel_end;
        if trail == 0 {
            break;
        }

        // (2) U12 = L11^{-1} · A12 (forward substitution, unit-lower L11). For each
        // panel row r, subtract the already-solved rows above it within the panel.
        for r in jb..panel_end {
            let (head, tail) = lu.split_at_mut(r * n);
            let row_r = &mut tail[0..n];
            for p in jb..r {
                let lrp = row_r[p];
                if lrp != 0.0 {
                    let row_p = &head[p * n..p * n + n];
                    for j in panel_end..n {
                        row_r[j] -= lrp * row_p[j];
                    }
                }
            }
        }

        // (3) Trailing update A22 -= L21·U12 via the packed GEMM.
        let mut l21 = vec![0.0f64; trail * bw];
        for i in 0..trail {
            let src = (panel_end + i) * n + jb;
            l21[i * bw..i * bw + bw].copy_from_slice(&lu[src..src + bw]);
        }
        let mut u12 = vec![0.0f64; bw * trail];
        for i in 0..bw {
            let src = (jb + i) * n + panel_end;
            u12[i * trail..i * trail + trail].copy_from_slice(&lu[src..src + trail]);
        }
        let target_start = panel_end * n + panel_end;
        packed_gemm_sub_assign_strided(&l21, &u12, trail, bw, trail, n, &mut lu[target_start..]);

        jb = panel_end;
    }

    Ok((lu, perm, sign))
}

fn lu_decompose(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<usize>, f64), LinAlgError> {
    lu_decompose_inner(a, n, true)
}

fn lu_decompose_for_det(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<usize>, f64), LinAlgError> {
    lu_decompose_inner(a, n, false)
}

fn diagonal_nan_cutoff(a: &[f64], n: usize) -> Option<usize> {
    let mut max_diag_nan = None;
    for row in 0..n {
        for col in 0..n {
            let value = a[row * n + col];
            if value.is_nan() {
                if row != col {
                    return None;
                }
                max_diag_nan = Some(row);
            }
        }
    }
    max_diag_nan
}

fn fill_diagonal_nans_with_one(a: &[f64], n: usize) -> Vec<f64> {
    let mut sanitized = a.to_vec();
    for i in 0..n {
        let diag = i * n + i;
        if sanitized[diag].is_nan() {
            sanitized[diag] = 1.0;
        }
    }
    sanitized
}

/// Forward-substitution (Ly = Pb) then back-substitution (Ux = y).
fn lu_forward_back(lu: &[f64], perm: &[usize], b: &[f64], n: usize) -> Vec<f64> {
    let mut x: Vec<f64> = perm.iter().map(|&p| b[p]).collect();

    // forward (L has unit diagonal)
    for i in 1..n {
        for j in 0..i {
            let l_ij = lu[i * n + j];
            x[i] -= l_ij * x[j];
        }
    }

    // back
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            let u_ij = lu[i * n + j];
            x[i] -= u_ij * x[j];
        }
        x[i] /= lu[i * n + i];
    }

    x
}

/// Forward-substitution then back-substitution for multiple right-hand sides.
// Engage blocked TRSM at this size; below it the row-by-row substitution wins.
const TRSM_BLOCK_MIN: usize = 768;
const TRSM_PANEL_NB: usize = 128;

// Blocked multi-RHS forward+back substitution (LAPACK dtrsm shape) for A·X = B
// given the LU factor. Forward (unit-lower L) processes row-blocks top-down:
// solve the nb×nb diagonal block, then update the trailing rows
// X[be..] -= L21·X[block] with the packed GEMM. Back (upper U) processes
// row-blocks bottom-up: solve the diagonal block, then update the rows above
// X[..ib] -= U12·X[block]. The GEMMs re-associate the substitution sums, so the
// result matches the row-by-row path within tolerance (triangular solve is not
// bit-reproducible; conformance is tolerance-based like NumPy's LAPACK).
fn lu_forward_back_multi_blocked(
    lu: &[f64],
    perm: &[usize],
    b: &[f64],
    n: usize,
    m: usize,
) -> Vec<f64> {
    let mut x = vec![0.0f64; n * m];
    for i in 0..n {
        let p_i = perm[i];
        x[i * m..i * m + m].copy_from_slice(&b[p_i * m..p_i * m + m]);
    }

    // Forward: L·Y = P·B, L unit-lower.
    let mut ib = 0;
    while ib < n {
        let be = (ib + TRSM_PANEL_NB).min(n);
        for i in ib..be {
            let (head, tail) = x.split_at_mut(i * m);
            let row_i = &mut tail[0..m];
            for j in ib..i {
                let lij = lu[i * n + j];
                if lij != 0.0 {
                    let row_j = &head[j * m..j * m + m];
                    for col in 0..m {
                        row_i[col] -= lij * row_j[col];
                    }
                }
            }
        }
        let bw = be - ib;
        let trail = n - be;
        if trail > 0 {
            let mut l21 = vec![0.0f64; trail * bw];
            for i in 0..trail {
                let src = (be + i) * n + ib;
                l21[i * bw..i * bw + bw].copy_from_slice(&lu[src..src + bw]);
            }
            let mut xb = vec![0.0f64; bw * m];
            for i in 0..bw {
                xb[i * m..i * m + m].copy_from_slice(&x[(ib + i) * m..(ib + i) * m + m]);
            }
            packed_gemm_sub_assign(&l21, &xb, trail, bw, m, &mut x[be * m..n * m]);
        }
        ib = be;
    }

    // Back: U·X = Y, U upper with non-unit diagonal.
    let mut be = n;
    while be > 0 {
        let ib = be.saturating_sub(TRSM_PANEL_NB);
        for i in (ib..be).rev() {
            let (head, tail) = x.split_at_mut((i + 1) * m);
            let row_i = &mut head[i * m..i * m + m];
            for j in (i + 1)..be {
                let uij = lu[i * n + j];
                if uij != 0.0 {
                    let row_j = &tail[(j - i - 1) * m..(j - i - 1) * m + m];
                    for col in 0..m {
                        row_i[col] -= uij * row_j[col];
                    }
                }
            }
            let uii = lu[i * n + i];
            for cell in row_i.iter_mut().take(m) {
                *cell /= uii;
            }
        }
        if ib > 0 {
            let bw = be - ib;
            let mut u12 = vec![0.0f64; ib * bw];
            for i in 0..ib {
                let src = i * n + ib;
                u12[i * bw..i * bw + bw].copy_from_slice(&lu[src..src + bw]);
            }
            let mut xb = vec![0.0f64; bw * m];
            for i in 0..bw {
                xb[i * m..i * m + m].copy_from_slice(&x[(ib + i) * m..(ib + i) * m + m]);
            }
            packed_gemm_sub_assign(&u12, &xb, ib, bw, m, &mut x[..ib * m]);
        }
        be = ib;
    }

    x
}

fn lu_forward_back_multi(lu: &[f64], perm: &[usize], b: &[f64], n: usize, m: usize) -> Vec<f64> {
    // Many right-hand sides (e.g. inv solves m=n columns of the identity): the
    // O(n^2·m) forward/back substitution becomes the dominant cost once the LU is
    // blocked, so route it through the blocked-TRSM path (panel solve + packed
    // GEMM trailing update). Gated on n and m large enough for the GEMM to win.
    if n >= TRSM_BLOCK_MIN && m >= MATMUL_PARALLEL_MIN_DIM {
        return lu_forward_back_multi_blocked(lu, perm, b, n, m);
    }

    let mut x = vec![0.0; n * m];

    // Apply permutation (Pb)
    for i in 0..n {
        let p_i = perm[i];
        for col in 0..m {
            x[i * m + col] = b[p_i * m + col];
        }
    }

    // Forward substitution (L has unit diagonal): Lx = Pb
    for i in 1..n {
        for j in 0..i {
            let l_ij = lu[i * n + j];
            for col in 0..m {
                x[i * m + col] -= l_ij * x[j * m + col];
            }
        }
    }

    // Back substitution: Ux = y
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            let u_ij = lu[i * n + j];
            for col in 0..m {
                x[i * m + col] -= u_ij * x[j * m + col];
            }
        }
        let u_ii = lu[i * n + i];
        for col in 0..m {
            x[i * m + col] /= u_ii;
        }
    }

    x
}

/// Solve Ax = b for an NxN system via LU decomposition with partial pivoting.
/// `a` is n*n row-major, `b` has length n.
pub fn solve_nxn(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, LinAlgError> {
    if b.len() != n {
        return Err(LinAlgError::ShapeContractViolation(
            "solve_nxn: rhs length must equal n",
        ));
    }
    if let Some(last_nan_diag) = diagonal_nan_cutoff(a, n) {
        let sanitized = fill_diagonal_nans_with_one(a, n);
        let mut solved = solve_nxn(&sanitized, b, n)?;
        for value in &mut solved[..=last_nan_diag] {
            *value = f64::NAN;
        }
        return Ok(solved);
    }
    if a.iter().any(|value| value.is_nan()) {
        return Ok(vec![f64::NAN; n]);
    }
    let (lu, perm, _) = lu_decompose_for_det(a, n)?;
    Ok(lu_forward_back(&lu, &perm, b, n))
}

/// Vector-RHS solve that REUSES caller-owned `lu` (n*n) and `perm` (n) scratch
/// buffers instead of allocating them per call — for a batched small-N solve this
/// eliminates the two largest per-lane allocations. The all-finite path runs the
/// same unblocked LU (`lu_factor_unblocked_into`) + forward/back as `solve_nxn`, so
/// it is byte-for-byte identical; any non-finite input (NaN-diagonal handling,
/// all-NaN, Inf) defers to `solve_nxn` for exact edge-case semantics. Intended for
/// `n < LU_BLOCK_MIN` (the regime where `solve_nxn` also uses the unblocked LU);
/// above that the caller keeps the regular path.
fn solve_nxn_into_out(
    a: &[f64],
    b: &[f64],
    n: usize,
    lu: &mut [f64],
    perm: &mut [usize],
    out: &mut [f64],
) -> Result<(), LinAlgError> {
    if b.len() != n || out.len() != n {
        return Err(LinAlgError::ShapeContractViolation(
            "solve_nxn_into_out: rhs and out length must equal n",
        ));
    }
    // Non-finite inputs (NaN-diagonal handling, all-NaN, Inf) defer to solve_nxn for
    // byte-identical edge-case semantics; the common all-finite path runs the same
    // unblocked LU + forward/back as solve_nxn but writes the solution straight into
    // `out`, reusing the caller's lu/perm scratch — zero per-lane allocation.
    if a.iter().any(|value| !value.is_finite()) {
        let solved = solve_nxn(a, b, n)?;
        out.copy_from_slice(&solved);
        return Ok(());
    }
    let matrix_max_abs = a.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let singularity_threshold = (n as f64) * f64::EPSILON * matrix_max_abs;
    lu_factor_unblocked_into(a, n, singularity_threshold, lu, perm)?;
    // Forward/back substitution into `out` (byte-identical to lu_forward_back's
    // x-allocating form: x[i] = b[perm[i]], then unit-lower forward, then upper back).
    for i in 0..n {
        out[i] = b[perm[i]];
    }
    for i in 1..n {
        for j in 0..i {
            out[i] -= lu[i * n + j] * out[j];
        }
    }
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            out[i] -= lu[i * n + j] * out[j];
        }
        out[i] /= lu[i * n + i];
    }
    Ok(())
}

/// Multiple-RHS sibling of `solve_nxn_into_out`: solve `A·X = B` (B is `n*m`,
/// row-major) writing the `n*m` solution directly into `out`, reusing the caller's
/// lu/perm scratch. Byte-identical to `solve_nxn_multi`'s unblocked path (the only
/// path reachable for `n < LU_BLOCK_MIN`, since the blocked TRSM needs
/// `n >= TRSM_BLOCK_MIN`); non-finite inputs defer to `solve_nxn_multi`.
fn solve_nxn_multi_into_out(
    a: &[f64],
    b: &[f64],
    n: usize,
    m: usize,
    lu: &mut [f64],
    perm: &mut [usize],
    out: &mut [f64],
) -> Result<(), LinAlgError> {
    if b.len() != n * m || out.len() != n * m {
        return Err(LinAlgError::ShapeContractViolation(
            "solve_nxn_multi_into_out: b and out length must equal n*m",
        ));
    }
    if a.iter().any(|value| !value.is_finite()) {
        let solved = solve_nxn_multi(a, b, n, m)?;
        out.copy_from_slice(&solved);
        return Ok(());
    }
    let matrix_max_abs = a.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let singularity_threshold = (n as f64) * f64::EPSILON * matrix_max_abs;
    lu_factor_unblocked_into(a, n, singularity_threshold, lu, perm)?;
    // Permutation (Pb) + unit-lower forward + upper back, into `out` (byte-identical
    // to lu_forward_back_multi's unblocked form).
    for i in 0..n {
        let p_i = perm[i];
        for col in 0..m {
            out[i * m + col] = b[p_i * m + col];
        }
    }
    for i in 1..n {
        for j in 0..i {
            let l_ij = lu[i * n + j];
            for col in 0..m {
                out[i * m + col] -= l_ij * out[j * m + col];
            }
        }
    }
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            let u_ij = lu[i * n + j];
            for col in 0..m {
                out[i * m + col] -= u_ij * out[j * m + col];
            }
        }
        let u_ii = lu[i * n + i];
        for col in 0..m {
            out[i * m + col] /= u_ii;
        }
    }
    Ok(())
}

#[inline]
fn det_from_lu_diagonal(lu: &[f64], n: usize, sign: f64) -> f64 {
    let mut det = sign;
    for i in 0..n {
        det *= lu[i * n + i];
    }
    det
}

#[inline]
fn slogdet_from_lu_diagonal(lu: &[f64], n: usize, sign: f64) -> (f64, f64) {
    let mut det_sign = sign;
    let mut log_abs_det = 0.0;
    for i in 0..n {
        let diag = lu[i * n + i];
        if diag.is_nan() {
            return (det_sign, f64::NAN);
        }
        if diag < 0.0 {
            det_sign = -det_sign;
            log_abs_det += (-diag).ln();
        } else if diag > 0.0 {
            log_abs_det += diag.ln();
        } else {
            return (0.0, f64::NEG_INFINITY);
        }
    }
    (det_sign, log_abs_det)
}

/// Determinant of an NxN matrix (flat row-major).  Returns 0.0 for singular
/// matrices instead of erroring.
pub fn det_nxn(a: &[f64], n: usize) -> Result<f64, LinAlgError> {
    if Some(a.len()) != n.checked_mul(n) {
        return Err(LinAlgError::ShapeContractViolation(
            "det_nxn: input length must equal n*n",
        ));
    }
    if n == 0 {
        return Ok(1.0);
    }

    match lu_decompose_for_det(a, n) {
        Ok((lu, _, sign)) => Ok(det_from_lu_diagonal(&lu, n, sign)),
        Err(LinAlgError::SolverSingularity) => Ok(0.0),
        Err(e) => Err(e),
    }
}

/// Sign and log-absolute-determinant for an NxN matrix.
pub fn slogdet_nxn(a: &[f64], n: usize) -> Result<(f64, f64), LinAlgError> {
    if Some(a.len()) != n.checked_mul(n) {
        return Err(LinAlgError::ShapeContractViolation(
            "slogdet_nxn: input length must equal n*n",
        ));
    }
    if n == 0 {
        return Ok((1.0, 0.0));
    }

    match lu_decompose_for_det(a, n) {
        Ok((lu, _, sign)) => Ok(slogdet_from_lu_diagonal(&lu, n, sign)),
        Err(LinAlgError::SolverSingularity) => Ok((0.0, f64::NEG_INFINITY)),
        Err(e) => Err(e),
    }
}

#[inline]
fn det_nxn_unblocked_with_scratch(
    a: &[f64],
    n: usize,
    lu: &mut [f64],
    perm: &mut [usize],
) -> Result<f64, LinAlgError> {
    match lu_factor_for_det_into(a, n, lu, perm) {
        Ok(sign) => Ok(det_from_lu_diagonal(lu, n, sign)),
        Err(LinAlgError::SolverSingularity) => Ok(0.0),
        Err(e) => Err(e),
    }
}

#[inline]
fn slogdet_nxn_unblocked_with_scratch(
    a: &[f64],
    n: usize,
    lu: &mut [f64],
    perm: &mut [usize],
) -> Result<(f64, f64), LinAlgError> {
    match lu_factor_for_det_into(a, n, lu, perm) {
        Ok(sign) => Ok(slogdet_from_lu_diagonal(lu, n, sign)),
        Err(LinAlgError::SolverSingularity) => Ok((0.0, f64::NEG_INFINITY)),
        Err(e) => Err(e),
    }
}

/// Inverse of an NxN matrix via LU decomposition.  Returns n*n flat
/// row-major.
pub fn inv_nxn(a: &[f64], n: usize) -> Result<Vec<f64>, LinAlgError> {
    if let Some(last_nan_diag) = diagonal_nan_cutoff(a, n) {
        let sanitized = fill_diagonal_nans_with_one(a, n);
        let mut inverse = inv_nxn(&sanitized, n)?;
        for row in 0..=last_nan_diag {
            for col in 0..n {
                inverse[row * n + col] = f64::NAN;
            }
        }
        return Ok(inverse);
    }
    if a.iter().any(|value| value.is_nan()) {
        return Ok(vec![f64::NAN; n * n]);
    }
    let (lu, perm, _) = lu_decompose_for_det(a, n)?;
    // For small unblocked inverses, solve against the natural identity first:
    // L^-1 is unit-lower-triangular, so the forward pass can skip x -= l*0 terms.
    // Larger sizes keep the dense multi-RHS path, where fixed-width vector loops
    // and the blocked TRSM crossover dominate the triangular sparsity savings.
    const INV_SPARSE_MAX_N: usize = 48;
    if n <= INV_SPARSE_MAX_N {
        Ok(inv_from_lu_unblocked(&lu, &perm, n))
    } else {
        let mut eye = vec![0.0; n * n];
        for i in 0..n {
            eye[i * n + i] = 1.0;
        }
        Ok(lu_forward_back_multi(&lu, &perm, &eye, n, n))
    }
}

/// Inverse A^-1 from packed LU + row permutation, exploiting identity-RHS
/// sparsity. Solving against natural I yields M = U^-1 L^-1, then A^-1 = M P.
fn inv_from_lu_unblocked(lu: &[f64], perm: &[usize], n: usize) -> Vec<f64> {
    let mut x = vec![0.0; n * n];
    for i in 0..n {
        x[i * n + i] = 1.0;
    }
    for i in 1..n {
        for j in 0..i {
            let l_ij = lu[i * n + j];
            let (row_i, row_j) = two_rows_mut(&mut x, i, j, n);
            for col in 0..=j {
                row_i[col] -= l_ij * row_j[col];
            }
        }
    }
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            let u_ij = lu[i * n + j];
            let (row_i, row_j) = two_rows_mut(&mut x, i, j, n);
            for col in 0..n {
                row_i[col] -= u_ij * row_j[col];
            }
        }
        let u_ii = lu[i * n + i];
        let row_i = &mut x[i * n..i * n + n];
        for col in 0..n {
            row_i[col] /= u_ii;
        }
    }
    let mut tmp = vec![0.0; n];
    for i in 0..n {
        tmp.copy_from_slice(&x[i * n..i * n + n]);
        let row_i = &mut x[i * n..i * n + n];
        for k in 0..n {
            row_i[perm[k]] = tmp[k];
        }
    }
    x
}

#[inline]
fn two_rows_mut(x: &mut [f64], a: usize, b: usize, n: usize) -> (&mut [f64], &[f64]) {
    debug_assert_ne!(a, b);
    if a < b {
        let (lo, hi) = x.split_at_mut(b * n);
        (&mut lo[a * n..a * n + n], &hi[..n])
    } else {
        let (lo, hi) = x.split_at_mut(a * n);
        (&mut hi[..n], &lo[b * n..b * n + n])
    }
}

/// Write `A^-1` (n*n) directly into `out`, reusing the caller's lu/perm scratch — no
/// per-lane lu/perm/eye/result allocation. Byte-identical to `inv_nxn`'s finite path
/// (LU + solve against the identity): the permutation of the identity RHS is
/// `out[i][col] = I[perm[i]][col] = (perm[i]==col)`, so no eye buffer is needed; the
/// forward/back is `lu_forward_back_multi`'s unblocked form (the only path for
/// `n < LU_BLOCK_MIN`). Non-finite inputs defer to `inv_nxn` for exact NaN-diagonal
/// semantics.
fn inv_nxn_into_out(
    a: &[f64],
    n: usize,
    lu: &mut [f64],
    perm: &mut [usize],
    out: &mut [f64],
) -> Result<(), LinAlgError> {
    if out.len() != n * n {
        return Err(LinAlgError::ShapeContractViolation(
            "inv_nxn_into_out: out length must equal n*n",
        ));
    }
    if a.iter().any(|value| !value.is_finite()) {
        let inv = inv_nxn(a, n)?;
        out.copy_from_slice(&inv);
        return Ok(());
    }
    let matrix_max_abs = a.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let singularity_threshold = (n as f64) * f64::EPSILON * matrix_max_abs;
    lu_factor_unblocked_into(a, n, singularity_threshold, lu, perm)?;
    // Pb where b = identity: out[i][col] = (perm[i] == col) ? 1 : 0.
    for i in 0..n {
        let p_i = perm[i];
        for col in 0..n {
            out[i * n + col] = if p_i == col { 1.0 } else { 0.0 };
        }
    }
    // Forward (unit-lower L), then back (upper U), n columns at a time.
    for i in 1..n {
        for j in 0..i {
            let l_ij = lu[i * n + j];
            for col in 0..n {
                out[i * n + col] -= l_ij * out[j * n + col];
            }
        }
    }
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            let u_ij = lu[i * n + j];
            for col in 0..n {
                out[i * n + col] -= u_ij * out[j * n + col];
            }
        }
        let u_ii = lu[i * n + i];
        for col in 0..n {
            out[i * n + col] /= u_ii;
        }
    }
    Ok(())
}

/// Solve A*X = B for multiple right-hand sides given the LU factor.
/// `lu` and `perm` are from `lu_decompose`. `b` is n*m (row-major).
pub fn lu_solve_multi(
    lu: &[f64],
    perm: &[usize],
    b: &[f64],
    n: usize,
    m: usize,
) -> Result<Vec<f64>, LinAlgError> {
    if Some(lu.len()) != n.checked_mul(n) || Some(b.len()) != n.checked_mul(m) || n == 0 || m == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "lu_solve_multi: LU must be n*n, B must be n*m, with n,m > 0",
        ));
    }

    let mut x = vec![0.0; n * m];

    // Process each column of B
    for col in 0..m {
        // Forward substitution (Ly = Pb)
        // We do this in-place in the output buffer for this column
        for i in 0..n {
            let mut sum = b[perm[i] * m + col];
            for j in 0..i {
                sum -= lu[i * n + j] * x[j * m + col];
            }
            x[i * m + col] = sum;
        }

        // Backward substitution (Ux = y)
        for i in (0..n).rev() {
            let mut sum = x[i * m + col];
            for j in (i + 1)..n {
                sum -= lu[i * n + j] * x[j * m + col];
            }
            x[i * m + col] = sum / lu[i * n + i];
        }
    }

    Ok(x)
}

/// LU factorization of an NxN matrix with partial pivoting.
/// Returns `(lu, perm, sign)`:
///   - `lu`: packed LU factors in n*n flat row-major (L is unit-lower-triangular,
///     U is upper-triangular, stored in the same buffer)
///   - `perm`: row permutation vector (perm[i] = original row at position i)
///   - `sign`: +1.0 or -1.0 (parity of permutation)
///
/// Matches `scipy.linalg.lu_factor` semantics.
pub fn lu_factor_nxn(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<usize>, f64), LinAlgError> {
    lu_decompose(a, n)
}

/// Solve a linear system using a pre-computed LU factorization.
/// `lu` and `perm` are the outputs of `lu_factor_nxn`.
/// `b` is the right-hand side vector of length n.
///
/// Matches `scipy.linalg.lu_solve` semantics.
pub fn lu_solve(lu: &[f64], perm: &[usize], b: &[f64], n: usize) -> Result<Vec<f64>, LinAlgError> {
    if Some(lu.len()) != n.checked_mul(n) || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "lu_solve: LU buffer must be n*n with n > 0",
        ));
    }
    if perm.len() != n {
        return Err(LinAlgError::ShapeContractViolation(
            "lu_solve: permutation length must equal n",
        ));
    }
    if b.len() != n {
        return Err(LinAlgError::ShapeContractViolation(
            "lu_solve: rhs length must equal n",
        ));
    }
    if b.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "rhs entries must be finite for lu_solve",
        ));
    }

    Ok(lu_forward_back(lu, perm, b, n))
}

/// Solve AX = B where B is an n*m matrix (multiple right-hand sides).
/// `a` is n*n row-major, `b` is n*m row-major.
/// Returns the n*m solution matrix X in row-major order.
///
/// Matches `numpy.linalg.solve` semantics when B is 2-D.
pub fn solve_nxn_multi(a: &[f64], b: &[f64], n: usize, m: usize) -> Result<Vec<f64>, LinAlgError> {
    if Some(a.len()) != n.checked_mul(n) || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "solve_nxn_multi: A must be n*n with n > 0",
        ));
    }
    if Some(b.len()) != n.checked_mul(m) {
        return Err(LinAlgError::ShapeContractViolation(
            "solve_nxn_multi: B must be n*m",
        ));
    }
    if let Some(last_nan_diag) = diagonal_nan_cutoff(a, n) {
        let sanitized = fill_diagonal_nans_with_one(a, n);
        let mut solved = solve_nxn_multi(&sanitized, b, n, m)?;
        for row in 0..=last_nan_diag {
            for col in 0..m {
                solved[row * m + col] = f64::NAN;
            }
        }
        return Ok(solved);
    }
    if a.iter().any(|value| value.is_nan()) {
        return Ok(vec![f64::NAN; n * m]);
    }
    let (lu, perm, _) = lu_decompose_for_det(a, n)?;

    Ok(lu_forward_back_multi(&lu, &perm, b, n, m))
}

/// Solve a triangular linear system.
/// `a` is n*n row-major triangular matrix, `b` has length n.
/// If `lower` is true, solves Lx = b (forward substitution).
/// If `lower` is false, solves Ux = b (back substitution).
/// If `unit_diagonal` is true, the diagonal of A is assumed to be all 1s.
///
/// Matches `scipy.linalg.solve_triangular` semantics.
pub fn solve_triangular(
    a: &[f64],
    b: &[f64],
    n: usize,
    lower: bool,
    unit_diagonal: bool,
) -> Result<Vec<f64>, LinAlgError> {
    if Some(a.len()) != n.checked_mul(n) || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "solve_triangular: A must be n*n with n > 0",
        ));
    }
    if b.len() != n {
        return Err(LinAlgError::ShapeContractViolation(
            "solve_triangular: rhs length must equal n",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) || b.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "entries must be finite for solve_triangular",
        ));
    }

    let mut x = b.to_vec();

    if lower {
        // Forward substitution: Lx = b
        for i in 0..n {
            for j in 0..i {
                x[i] -= a[i * n + j] * x[j];
            }
            if unit_diagonal {
                // diagonal assumed to be 1
            } else {
                let diag = a[i * n + i];
                if diag == 0.0 {
                    return Err(LinAlgError::SolverSingularity);
                }
                x[i] /= diag;
            }
        }
    } else {
        // Back substitution: Ux = b
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                x[i] -= a[i * n + j] * x[j];
            }
            if unit_diagonal {
                // diagonal assumed to be 1
            } else {
                let diag = a[i * n + i];
                if diag == 0.0 {
                    return Err(LinAlgError::SolverSingularity);
                }
                x[i] /= diag;
            }
        }
    }

    Ok(x)
}

/// Cholesky decomposition for NxN positive-definite matrix.
/// Returns the lower-triangular factor L such that A = L L^T.
/// `a` is n*n row-major.
// Blocked Cholesky engages at this dimension (same crossover rationale as the
// blocked LU); below it the unblocked dot-product factorization wins. Panel width
// >= the GEMM parallel gate (128) so the trailing-update GEMM runs parallel.
// The blocked Cholesky factors each nb-wide diagonal block with an UNBLOCKED scalar
// dot-product kernel (step 1 of cholesky_blocked), costing O(nb^3/3) scalar flops per
// panel. A wide panel therefore serializes a large chunk of scalar work that the
// (parallel, register-tiled) trailing GEMM/SYRK cannot recover. The previous 128-wide
// large-n panel left Cholesky 1.6-2.2x behind numpy at n=896-2048; profiling (low
// load, vs numpy) shows narrow panels win across the board:
//   n=896/1024   nb=32 -> 1.04 / 1.73   (nb=128 gave 1.62 / 2.19)
//   n=1536/2048  nb=64 -> 1.35 / 1.40   (nb=32 drifts to 1.47 / 1.50, nb=128 ~1.53)
//   n in [384,640) nb=64 (prior mid tuning; 512-ish regime likes a wider panel)
const CHOL_MID_MIN: usize = 128;

// One width policy for every blocked size. Narrow below the large-n crossover (keeps
// the scalar diagonal block tiny); 64 once the matrix is large enough that the wider
// rank-64 trailing GEMM amortizes its extra diagonal cost.
const fn cholesky_panel_width(n: usize) -> usize {
    if n >= 1280 {
        64
    } else if n >= 640 {
        32
    } else if n >= 384 {
        64
    } else {
        32
    }
}

// Column-block width for the block-triangular SYRK trailing update. A multiple of
// PACKED_NR so each block stays on the register-tiled GEMM path; >= 128 so the
// strided sub-assign GEMM clears its parallel threshold on the leading (tall)
// blocks. The per-diagonal-block upper-triangle waste is ~SYRK_COL_BLOCK/trail of
// the update flops, so a moderate width keeps the triangular saving near 2x.
const SYRK_COL_BLOCK: usize = 256;
const SYRK_MID_COL_BLOCK: usize = 64;
const SYRK_MID_TRIANGULAR_MIN_TRAIL: usize = 384;

// Minimum trailing-panel height for the parallel dtrsm panel solve. Below this the
// rayon dispatch + L11 packing cost exceeds the per-row solve work, so the small
// trailing panels (and every panel of a small matrix) keep the serial row scan.
const PAR_TRSM_MIN_TRAIL: usize = 384;

#[inline(always)]
fn cholesky_dot_add_ordered(lhs: &[f64], rhs: &[f64]) -> f64 {
    debug_assert_eq!(lhs.len(), rhs.len());
    let mut sum = 0.0;
    let mut k = 0;
    while k + 4 <= lhs.len() {
        sum += lhs[k] * rhs[k];
        sum += lhs[k + 1] * rhs[k + 1];
        sum += lhs[k + 2] * rhs[k + 2];
        sum += lhs[k + 3] * rhs[k + 3];
        k += 4;
    }
    while k < lhs.len() {
        sum += lhs[k] * rhs[k];
        k += 1;
    }
    sum
}

// Right-looking blocked Cholesky (LAPACK dpotrf shape). For each width-nb column
// panel: factor the nb×nb diagonal block (unblocked), solve the panel below it
// (L21 = A21·L11^{-T}), then update the trailing block A22 -= L21·L21^T with the
// cache-blocked packed GEMM. Numerically equivalent to the unblocked dot-product
// factorization up to the GEMM's re-association (tolerance — Cholesky is not
// bit-reproducible). Caller guarantees finite input.
fn cholesky_blocked(a: &[f64], n: usize, panel_nb: usize) -> Result<Vec<f64>, LinAlgError> {
    let mut l = vec![0.0f64; n * n]; // output (lower triangular)
    let mut work = a.to_vec(); // trailing submatrix, updated in place (lower read)
    let mut jb = 0;
    while jb < n {
        let bw = panel_nb.min(n - jb);
        let pend = jb + bw;

        // (1) Factor the diagonal block A11 [jb,pend)×[jb,pend) into L11.
        for i in jb..pend {
            for j in jb..=i {
                let mut sum = work[i * n + j];
                for k in jb..j {
                    sum -= l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    if sum <= 0.0 {
                        return Err(LinAlgError::CholeskyContractViolation(
                            "matrix is not positive definite",
                        ));
                    }
                    l[i * n + j] = sum.sqrt();
                } else {
                    l[i * n + j] = sum / l[j * n + j];
                }
            }
        }

        let trail = n - pend;
        if trail == 0 {
            break;
        }

        // (2) Panel below the diagonal: L21 = A21·L11^{-T} (a triangular solve /
        // dtrsm). Each trailing row solves independently against the finalized
        // diagonal block L11, so for a tall trailing panel the rows run in parallel
        // (the old serial row scan was the dominant cost vs numpy's dtrsm). L11 (bw×bw
        // lower) and the reciprocal of its diagonal are packed once into small
        // cache-resident buffers so the hot inner dot reads contiguous, read-only
        // memory instead of striding across `l` by n per term. For a short trailing
        // panel the rayon dispatch + packing cost exceeds the work, so we keep the
        // original serial scan (no small-matrix regression).
        let mut l21 = vec![0.0f64; trail * bw];
        if trail >= PAR_TRSM_MIN_TRAIL {
            let mut l11 = vec![0.0f64; bw * bw];
            let mut inv_diag = vec![0.0f64; bw];
            for r in 0..bw {
                let row = (jb + r) * n + jb;
                l11[r * bw..r * bw + r + 1].copy_from_slice(&l[row..row + r + 1]);
                inv_diag[r] = 1.0 / l[(jb + r) * n + (jb + r)];
            }
            l[pend * n..n * n]
                .par_chunks_mut(n)
                .zip(l21.par_chunks_mut(bw))
                .enumerate()
                .for_each(|(ti, (lrow, l21row))| {
                    let i = pend + ti;
                    let wrow = &work[i * n..i * n + n];
                    for r in 0..bw {
                        let j = jb + r;
                        let mut sum = wrow[j];
                        let solved = &lrow[jb..jb + r]; // this row's solved L21 entries
                        let l11r = &l11[r * bw..r * bw + r];
                        for (&a, &b) in solved.iter().zip(l11r) {
                            sum -= a * b;
                        }
                        let value = sum * inv_diag[r];
                        lrow[j] = value;
                        l21row[r] = value;
                    }
                });
        } else {
            for i in pend..n {
                let l21_base = (i - pend) * bw;
                for j in jb..pend {
                    let mut sum = work[i * n + j];
                    for k in jb..j {
                        sum -= l[i * n + k] * l[j * n + k];
                    }
                    let value = sum / l[j * n + j];
                    l[i * n + j] = value;
                    l21[l21_base + (j - jb)] = value;
                }
            }
        }

        // (3) Trailing update A22 -= L21·L21^T. This is a symmetric rank-bw update
        // (SYRK): only A22's LOWER triangle is needed, since Cholesky never reads the
        // strict upper triangle of `work` (steps 1 and 2 only touch work[i*n+j] with
        // j <= i). We exploit this by tiling A22 into column blocks and updating only
        // the rows at or below each block: a lower cell (i,j), i >= j, lands in exactly
        // one column block (whose columns contain j) whose row range (>= the block's
        // first column >= ... actually >= c0 <= j <= i) contains i, so every lower cell
        // gets its full bw-deep update exactly once. This halves the trailing-update
        // flops versus a full trail×trail GEMM (matching LAPACK dpotrf's dsyrk) and
        // subtracts directly into `work` with no product buffer, transpose, or subtract
        // pass. Strict-upper entries inside diagonal blocks are written stale but never
        // read. Wide panels use the generic strided-GEMM parallel gate. Mid panels use
        // smaller lower-triangular column tiles once the trailing matrix is wide enough:
        // each lower cell still receives one ascending-k dot product, but strict-upper
        // cells are not computed or written.
        if bw >= MATMUL_PARALLEL_MIN_DIM && trail >= SYRK_COL_BLOCK {
            let mut bblk = vec![0.0f64; bw * SYRK_COL_BLOCK];
            let mut c0 = 0;
            while c0 < trail {
                let cbw = SYRK_COL_BLOCK.min(trail - c0);
                let rows = trail - c0; // rows c0..trail (lower-triangular: i >= c0)
                // b-block = (l21[c0..c0+cbw, :])^T, shape bw×cbw row-major.
                for r in 0..cbw {
                    let arow = (c0 + r) * bw;
                    for k in 0..bw {
                        bblk[k * cbw + r] = l21[arow + k];
                    }
                }
                let a_block = &l21[c0 * bw..trail * bw]; // rows c0..trail, shape rows×bw
                let dst = (pend + c0) * n + (pend + c0);
                packed_gemm_sub_assign_strided(
                    a_block,
                    &bblk[..bw * cbw],
                    rows,
                    bw,
                    cbw,
                    n,
                    &mut work[dst..],
                );
                c0 += cbw;
            }
        } else if trail >= PAR_TRSM_MIN_TRAIL && rayon::current_num_threads() >= 2 {
            // Thin-panel parallel SYRK. A rank-bw trailing update (here bw = the mid-size
            // panel, < MATMUL_PARALLEL_MIN_DIM) never trips packed_gemm's k>=128 parallel
            // gate, so the small-panel `packed_gemm` above runs the entire trail×trail
            // update on ONE core — the dominant cost and the whole mid-size (128<=n<896)
            // cholesky vs-numpy gap (numpy's dpotrf parallelizes its dsyrk). Instead fan
            // out over the LARGE trailing-row dimension: each trailing row i subtracts,
            // for every lower-triangle column j<=i, the bw-term dot accumulated in k order
            // — BYTE-IDENTICAL to the packed_gemm product subtract (same per-cell k-order;
            // only the lower triangle is ever read downstream, so the strict upper is left
            // untouched exactly as the old full-GEMM result was never read there). Disjoint
            // output rows give unsafe-free rayon over contiguous row chunks.
            work[pend * n..n * n]
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(i, wrow)| {
                    let li = &l21[i * bw..i * bw + bw];
                    for j in 0..=i {
                        let lj = &l21[j * bw..j * bw + bw];
                        let mut s = 0.0f64;
                        for k in 0..bw {
                            s += li[k] * lj[k];
                        }
                        wrow[pend + j] -= s;
                    }
                });
        } else if trail >= SYRK_MID_TRIANGULAR_MIN_TRAIL {
            // Single-core fallback (rayon thread count < 2, so the parallel thin-panel
            // branch above is skipped). Rather than drop to the full trail×trail GEMM
            // (which computes the discarded strict-upper triangle = ~2x flops + a product
            // buffer + a separate subtract pass), tile A22 into SYRK_MID_COL_BLOCK
            // lower-triangular column blocks and subtract each directly into `work` via
            // the packed microkernel, computing only the lower cells. ~5x over the
            // full-GEMM path at n=895 on one core; byte-identical (same ascending-k
            // per-cell dot — the cholesky_mid_panel_256/512 goldens hold for both this
            // and the parallel branch).
            let mut bblk = vec![0.0f64; bw * SYRK_MID_COL_BLOCK];
            let mut c0 = 0;
            while c0 < trail {
                let cbw = SYRK_MID_COL_BLOCK.min(trail - c0);
                let rows = trail - c0;
                for r in 0..cbw {
                    let arow = (c0 + r) * bw;
                    for k in 0..bw {
                        bblk[k * cbw + r] = l21[arow + k];
                    }
                }
                let a_block = &l21[c0 * bw..trail * bw];
                let dst = (pend + c0) * n + (pend + c0);
                packed_gemm_sub_assign_strided(
                    a_block,
                    &bblk[..bw * cbw],
                    rows,
                    bw,
                    cbw,
                    n,
                    &mut work[dst..],
                );
                c0 += cbw;
            }
        } else {
            // Small-panel path: full trail×trail GEMM, subtract into the lower triangle.
            let mut l21t = vec![0.0f64; bw * trail];
            for i in 0..trail {
                for k in 0..bw {
                    l21t[k * trail + i] = l21[i * bw + k];
                }
            }
            let g = packed_gemm(&l21, &l21t, trail, bw, trail);
            for i in 0..trail {
                let dst = (pend + i) * n + pend;
                for (cell, &gij) in work[dst..dst + trail]
                    .iter_mut()
                    .zip(&g[i * trail..i * trail + trail])
                {
                    *cell -= gij;
                }
            }
        }

        jb = pend;
    }

    Ok(l)
}

pub fn cholesky_nxn(a: &[f64], n: usize) -> Result<Vec<f64>, LinAlgError> {
    if Some(a.len()) != n.checked_mul(n) || n == 0 {
        return Err(LinAlgError::CholeskyContractViolation(
            "cholesky_nxn: input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::CholeskyContractViolation(
            "cholesky requires finite entries",
        ));
    }

    if n >= CHOL_MID_MIN {
        return cholesky_blocked(a, n, cholesky_panel_width(n));
    }

    let use_ordered_dot = (16..=32).contains(&n);
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        let row_i = i * n;
        for j in 0..=i {
            let row_j = j * n;
            let sum = if use_ordered_dot {
                cholesky_dot_add_ordered(&l[row_i..row_i + j], &l[row_j..row_j + j])
            } else {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[row_i + k] * l[row_j + k];
                }
                sum
            };
            if i == j {
                let diag = a[row_i + i] - sum;
                if diag <= 0.0 {
                    return Err(LinAlgError::CholeskyContractViolation(
                        "matrix is not positive definite",
                    ));
                }
                l[row_i + j] = diag.sqrt();
            } else {
                l[row_i + j] = (a[row_i + j] - sum) / l[row_j + j];
            }
        }
    }
    Ok(l)
}

/// Write the lower Cholesky factor `L` (n*n) directly into `out`, with no per-lane
/// allocation. `out` MUST be pre-zeroed (the upper triangle is never written, matching
/// `cholesky_nxn`'s zeroed buffer). Byte-identical to `cholesky_nxn`'s unblocked path
/// (the only path for `n < CHOL_MID_MIN`): same direct formula, reading the
/// already-written lower-triangle entries from `out` instead of an owned `l`. Returns
/// the same finite / not-positive-definite errors.
fn cholesky_nxn_into_out(a: &[f64], n: usize, out: &mut [f64]) -> Result<(), LinAlgError> {
    if out.len() != n * n {
        return Err(LinAlgError::CholeskyContractViolation(
            "cholesky_nxn_into_out: out length must equal n*n",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::CholeskyContractViolation(
            "cholesky requires finite entries",
        ));
    }
    let use_ordered_dot = (16..=32).contains(&n);
    for i in 0..n {
        let row_i = i * n;
        for j in 0..=i {
            let row_j = j * n;
            let sum = if use_ordered_dot {
                cholesky_dot_add_ordered(&out[row_i..row_i + j], &out[row_j..row_j + j])
            } else {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += out[row_i + k] * out[row_j + k];
                }
                sum
            };
            if i == j {
                let diag = a[row_i + i] - sum;
                if diag <= 0.0 {
                    return Err(LinAlgError::CholeskyContractViolation(
                        "matrix is not positive definite",
                    ));
                }
                out[row_i + j] = diag.sqrt();
            } else {
                out[row_i + j] = (a[row_i + j] - sum) / out[row_j + j];
            }
        }
    }
    Ok(())
}

/// Solve A*x = b given the lower Cholesky factor L where A = L*L^T.
///
/// Two-step forward/back substitution:
///   1. Solve L*y = b  (forward substitution)
///   2. Solve L^T*x = y (backward substitution)
///
/// `l` is the n×n lower-triangular Cholesky factor (row-major, n*n elements).
/// `b` is the n-element right-hand side.
pub fn cholesky_solve(l: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, LinAlgError> {
    if Some(l.len()) != n.checked_mul(n) || b.len() != n || n == 0 {
        return Err(LinAlgError::CholeskyContractViolation(
            "cholesky_solve: L must be n*n, b must be n, with n > 0",
        ));
    }
    if l.iter().chain(b.iter()).any(|v| !v.is_finite()) {
        return Err(LinAlgError::CholeskyContractViolation(
            "cholesky_solve requires finite entries",
        ));
    }
    // Check for zero diagonal (singular L)
    for i in 0..n {
        if l[i * n + i] == 0.0 {
            return Err(LinAlgError::SolverSingularity);
        }
    }

    // Forward substitution: L*y = b
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i * n + j] * y[j];
        }
        y[i] = sum / l[i * n + i];
    }

    // Backward substitution: L^T*x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j * n + i] * x[j]; // L^T[i,j] = L[j,i]
        }
        x[i] = sum / l[i * n + i];
    }
    Ok(x)
}

/// Solve A*X = B for multiple right-hand sides given the Cholesky factor L.
///
/// `l` is n×n, `b` is n×m (row-major). Returns n×m solution matrix.
pub fn cholesky_solve_multi(
    l: &[f64],
    b: &[f64],
    n: usize,
    m: usize,
) -> Result<Vec<f64>, LinAlgError> {
    if Some(l.len()) != n.checked_mul(n) || Some(b.len()) != n.checked_mul(m) || n == 0 || m == 0 {
        return Err(LinAlgError::CholeskyContractViolation(
            "cholesky_solve_multi: L must be n*n, B must be n*m, with n,m > 0",
        ));
    }

    let mut x = b.to_vec();

    // Forward substitution: L*Y = B
    for i in 0..n {
        for j in 0..i {
            let l_ij = l[i * n + j];
            for col in 0..m {
                x[i * m + col] -= l_ij * x[j * m + col];
            }
        }
        let l_ii = l[i * n + i];
        for col in 0..m {
            x[i * m + col] /= l_ii;
        }
    }

    // Backward substitution: L^T*X = Y
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            let l_ji = l[j * n + i];
            for col in 0..m {
                x[i * m + col] -= l_ji * x[j * m + col]; // L^T[i,j] = L[j,i]
            }
        }
        let l_ii = l[i * n + i];
        for col in 0..m {
            x[i * m + col] /= l_ii;
        }
    }

    Ok(x)
}

/// Solve the tensor equation `a x = b` for x.
///
/// Equivalent to `numpy.linalg.tensorsolve`. Reshapes `a` and `b` so that
/// the equation becomes a standard linear system, solves it, then reshapes back.
///
/// `a_shape` and `b_shape` are the shapes of the operands, `a_data` and `b_data`
/// are the flat row-major data buffers.
pub fn tensorsolve(
    a_data: &[f64],
    a_shape: &[usize],
    b_data: &[f64],
    b_shape: &[usize],
) -> Result<(Vec<f64>, Vec<usize>), LinAlgError> {
    // The equation a x = b: a has shape (b_shape..., x_shape...) and
    // x has shape x_shape such that contracting the last len(x_shape) dims of a
    // with x produces b.
    let b_ndim = b_shape.len();
    if a_shape.len() <= b_ndim {
        return Err(LinAlgError::ShapeContractViolation(
            "tensorsolve: a must have more dimensions than b",
        ));
    }

    // Verify leading dims of a match b_shape
    for (i, (&a_dim, &b_dim)) in a_shape.iter().zip(b_shape.iter()).enumerate() {
        if a_dim != b_dim {
            return Err(LinAlgError::ShapeContractViolation(
                "tensorsolve: leading dimensions of a must match shape of b",
            ));
        }
        let _ = i; // suppress unused warning
    }

    // x_shape is the trailing dims of a after b_shape
    let x_shape = &a_shape[b_ndim..];
    let n: usize = fnp_ndarray::element_count(b_shape)
        .map_err(|_| LinAlgError::ShapeContractViolation("product overflow"))?;
    if b_data.len() != n {
        return Err(LinAlgError::ShapeContractViolation(
            "tensorsolve: b_data length must equal product of b_shape",
        ));
    }
    let m: usize = fnp_ndarray::element_count(x_shape)
        .map_err(|_| LinAlgError::ShapeContractViolation("product overflow"))?; // product of x_shape

    if n != m {
        return Err(LinAlgError::ShapeContractViolation(
            "tensorsolve: reshaped system must be square (prod(b_shape) == prod(x_shape))",
        ));
    }

    if Some(a_data.len()) != n.checked_mul(m) {
        return Err(LinAlgError::ShapeContractViolation(
            "tensorsolve: a_data length must be prod(a_shape)",
        ));
    }

    // Now solve the n×n system
    Ok((solve_nxn(a_data, b_data, n)?, x_shape.to_vec()))
}

/// Compute the inverse of an N-dimensional array.
///
/// Equivalent to `numpy.linalg.tensorinv(a, ind)`. The first `ind` axes of `a`
/// are "output" axes, remaining axes are "input" axes. The product of output axes
/// must equal the product of input axes, forming a square matrix to invert.
pub fn tensorinv(
    a_data: &[f64],
    a_shape: &[usize],
    ind: usize,
) -> Result<(Vec<f64>, Vec<usize>), LinAlgError> {
    if ind == 0 || ind >= a_shape.len() {
        return Err(LinAlgError::ShapeContractViolation(
            "tensorinv: ind must be > 0 and < ndim(a)",
        ));
    }
    let output_shape = &a_shape[..ind];
    let input_shape = &a_shape[ind..];
    let n: usize = fnp_ndarray::element_count(output_shape)
        .map_err(|_| LinAlgError::ShapeContractViolation("product overflow"))?;
    let m: usize = fnp_ndarray::element_count(input_shape)
        .map_err(|_| LinAlgError::ShapeContractViolation("product overflow"))?;

    if n != m {
        return Err(LinAlgError::ShapeContractViolation(
            "tensorinv: product of first ind dims must equal product of remaining dims",
        ));
    }
    if Some(a_data.len()) != n.checked_mul(n) {
        return Err(LinAlgError::ShapeContractViolation(
            "tensorinv: data length does not match shape",
        ));
    }

    let inv = inv_nxn(a_data, n)?;

    // Output shape is (input_shape..., output_shape...)
    let mut result_shape = input_shape.to_vec();
    result_shape.extend_from_slice(output_shape);

    Ok((inv, result_shape))
}

/// QR decomposition via Householder reflections for NxN matrix.
/// Returns (q, r) as flat row-major n*n buffers.
// Blocked QR (compact-WY, GEMM trailing update) engages at this size; below it
// the unblocked two-pass Householder loop wins. Current profile-backed crossover
// includes n=512, where the compact-WY path amortizes panel/GEMM work.
const QR_BLOCK_MIN: usize = 512;
const QR_PANEL_NB: usize = 128;

// Blocked Householder QR via the compact-WY representation (LAPACK dgeqrf +
// dlarft/dlarfb shape). For each width-nb column panel: factor the panel
// unblocked (storing the reflector vectors V and coefficients tau), build the
// nb×nb upper-triangular T so that H_jb·…·H_{pend-1} = I − V·T·V^T, then apply
// that single block reflector to the trailing R columns and to Q as GEMMs
// (I − V·T·V^T)·R_trail and Q·(I − V·T·V^T). Same reflector/sign convention as
// the unblocked path, so Q and R match it within tolerance — never bit-exact
// (Householder QR is not bit-reproducible; conformance is tolerance-based like
// NumPy's LAPACK). Caller guarantees finite input.
fn qr_blocked(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), LinAlgError> {
    let mut q = vec![0.0f64; n * n];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }
    let mut r = a.to_vec();

    let mut jb = 0;
    while jb < n {
        let nb = QR_PANEL_NB.min(n - jb);
        let pend = jb + nb;
        let h = n - jb; // active row height (reflectors touch rows jb..n)

        // V stored as h×nb (row r-th = global row jb+r); column t = reflector v_{jb+t}.
        let mut vv = vec![0.0f64; h * nb];
        let mut taus = vec![0.0f64; nb];
        // Scratch for the cache-friendly two-pass reflector apply within the panel.
        let mut dpanel = vec![0.0f64; nb];

        // (1) Panel factorization — unblocked within the panel columns [jb, pend).
        for t in 0..nb {
            let k = jb + t;
            let mut col_norm_sq = 0.0;
            for i in k..n {
                col_norm_sq += r[i * n + k] * r[i * n + k];
            }
            let col_norm = col_norm_sq.sqrt();
            if col_norm == 0.0 {
                continue; // null reflector (tau=0, v=0)
            }
            let sign = if r[k * n + k] >= 0.0 { 1.0 } else { -1.0 };
            for i in k..n {
                vv[(i - jb) * nb + t] = r[i * n + k];
            }
            vv[(k - jb) * nb + t] += sign * col_norm;
            let mut v_norm_sq = 0.0;
            for i in k..n {
                let x = vv[(i - jb) * nb + t];
                v_norm_sq += x * x;
            }
            if v_norm_sq == 0.0 {
                continue;
            }
            let scale = 2.0 / v_norm_sq;
            taus[t] = scale;
            // Apply H_k to the panel columns [k, pend) (the trailing columns
            // [pend, n) are deferred to the block GEMM in step 3). Two-pass,
            // row-contiguous form instead of striding r[i*n+j] down each column:
            // pass 1 sums d[j] = Σ_i vv[i]·r[i][j] in i-ascending order, pass 2
            // applies the identical (scale·d[j])·vv[i] per element — bit-for-bit
            // the same result as the per-column walk, but streams r by rows.
            let width = pend - k;
            for dj in dpanel[..width].iter_mut() {
                *dj = 0.0;
            }
            for i in k..n {
                let vi = vv[(i - jb) * nb + t];
                let row = &r[i * n + k..i * n + pend];
                for (dj, &rij) in dpanel[..width].iter_mut().zip(row.iter()) {
                    *dj += vi * rij;
                }
            }
            for dj in dpanel[..width].iter_mut() {
                *dj *= scale;
            }
            for i in k..n {
                let vi = vv[(i - jb) * nb + t];
                let row = &mut r[i * n + k..i * n + pend];
                for (rij, &dj) in row.iter_mut().zip(dpanel[..width].iter()) {
                    *rij -= dj * vi;
                }
            }
        }

        // (2) Build T (nb×nb upper triangular), forward direction.
        let mut tm = vec![0.0f64; nb * nb];
        for t in 0..nb {
            tm[t * nb + t] = taus[t];
            if taus[t] == 0.0 {
                continue;
            }
            // col[i] = -tau_t · (V[:,i]·V[:,t]) for i in 0..t.
            let mut col = vec![0.0f64; t];
            for (i, ci) in col.iter_mut().enumerate() {
                let mut dot = 0.0;
                for row in 0..h {
                    dot += vv[row * nb + i] * vv[row * nb + t];
                }
                *ci = -taus[t] * dot;
            }
            // T[0:t, t] = T[0:t, 0:t] · col (T upper triangular).
            for i in 0..t {
                let mut s = 0.0;
                for l in i..t {
                    s += tm[i * nb + l] * col[l];
                }
                tm[i * nb + t] = s;
            }
        }

        // V^T (nb×h), reused by steps 3 and 4.
        let mut vt = vec![0.0f64; nb * h];
        for row in 0..h {
            for t in 0..nb {
                vt[t * h + row] = vv[row * nb + t];
            }
        }

        // (3) Trailing R is applied on the LEFT in increasing-k order, i.e.
        // H_{pend-1}·…·H_jb = (I − V·T·V^T)^T = I − V·T^T·V^T, so use T^T here
        // (Q, applied on the right in forward order, uses T in step 4).
        let mut tmt = vec![0.0f64; nb * nb];
        for i in 0..nb {
            for j in 0..nb {
                tmt[i * nb + j] = tm[j * nb + i];
            }
        }
        let trail = n - pend;
        if trail > 0 {
            let mut rt = vec![0.0f64; h * trail];
            for row in 0..h {
                let src = (jb + row) * n + pend;
                rt[row * trail..row * trail + trail].copy_from_slice(&r[src..src + trail]);
            }
            let w1 = packed_gemm(&vt, &rt, nb, h, trail); // V^T·R_trail
            let w2 = packed_gemm(&tmt, &w1, nb, nb, trail); // T^T·W1
            let vw2 = packed_gemm(&vv, &w2, h, nb, trail); // V·W2
            for row in 0..h {
                let dst = (jb + row) * n + pend;
                for (cell, &x) in r[dst..dst + trail]
                    .iter_mut()
                    .zip(&vw2[row * trail..row * trail + trail])
                {
                    *cell -= x;
                }
            }
        }

        // (4) Q := Q·(I − V·T·V^T) = Q − ((Q_active·V)·T)·V^T.
        let mut qa = vec![0.0f64; n * h];
        for i in 0..n {
            let src = i * n + jb;
            qa[i * h..i * h + h].copy_from_slice(&q[src..src + h]);
        }
        let qv = packed_gemm(&qa, &vv, n, h, nb); // Q_active·V
        let qvt = packed_gemm(&qv, &tm, n, nb, nb); // ·T
        let upd = packed_gemm(&qvt, &vt, n, nb, h); // ·V^T
        for i in 0..n {
            let dst = i * n + jb;
            for (cell, &x) in q[dst..dst + h].iter_mut().zip(&upd[i * h..i * h + h]) {
                *cell -= x;
            }
        }

        jb = pend;
    }

    Ok((q, r))
}

pub fn qr_nxn(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), LinAlgError> {
    if Some(a.len()) != n.checked_mul(n) || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "qr_nxn: input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "matrix entries must be finite for QR",
        ));
    }

    if n >= QR_BLOCK_MIN {
        return qr_blocked(a, n);
    }

    // Start with Q = I, R = A
    let mut q = vec![0.0; n * n];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }
    let mut r = a.to_vec();

    let mut v = vec![0.0; n];
    // Scratch for the cache-friendly left Householder transform of R (see below).
    let mut d = vec![0.0; n];
    let mut f_vec = vec![0.0; n];
    for k in 0..n {
        // Extract column k below diagonal
        let mut col_norm_sq = 0.0;
        for i in k..n {
            col_norm_sq += r[i * n + k] * r[i * n + k];
        }
        let col_norm = col_norm_sq.sqrt();
        if col_norm == 0.0 {
            continue;
        }

        // Householder vector v = x + sign(x_k)*||x||*e_k
        let sign = if r[k * n + k] >= 0.0 { 1.0 } else { -1.0 };
        for i in k..n {
            v[i] = r[i * n + k];
        }
        v[k] += sign * col_norm;
        let v_norm_sq: f64 = v[k..].iter().map(|x| x * x).sum();
        if v_norm_sq == 0.0 {
            continue;
        }

        // Apply H = I - 2*v*v^T/||v||^2 to R as two row-contiguous passes instead
        // of the naive per-column walk (which strode r[i*n+j] down each column
        // with stride n). Bit-exact: pass 1 sums d[j] = Σ_i v[i]·r[i][j] in the
        // same i-ascending order, pass 2 applies the identical (scale·d[j])·v[i]
        // product grouping per element.
        let scale = 2.0 / v_norm_sq;
        for dj in d[k..n].iter_mut() {
            *dj = 0.0;
        }
        for i in k..n {
            let vi = v[i];
            let row = &r[i * n + k..i * n + n];
            for (dj, &rij) in d[k..n].iter_mut().zip(row.iter()) {
                *dj += vi * rij;
            }
        }
        for (fj, &dj) in f_vec[k..n].iter_mut().zip(d[k..n].iter()) {
            *fj = scale * dj;
        }
        // Pass 2: R[i, k..n] -= (scale·d[j])·v[i] (row-contiguous, serial — already
        // cache-optimal as a two-pass transform).
        for i in k..n {
            let vi = v[i];
            let row = &mut r[i * n + k..i * n + n];
            for (rij, &fj) in row.iter_mut().zip(f_vec[k..n].iter()) {
                *rij -= fj * vi;
            }
        }

        // Accumulate Q = Q * H.
        for i in 0..n {
            let mut dot = 0.0;
            for j in k..n {
                dot += q[i * n + j] * v[j];
            }
            let factor = scale * dot;
            for j in k..n {
                q[i * n + j] -= factor * v[j];
            }
        }
    }

    Ok((q, r))
}

/// QR decomposition of an m×n rectangular matrix (Householder reflections).
///
/// Input `a` is row-major with shape (m, n). Returns `(Q, R)` where:
/// - Q is m×m orthogonal matrix (row-major, m*m elements)
/// - R is m×n upper-trapezoidal matrix (row-major, m*n elements)
///
/// This is the "complete" mode (full Q). For `reduced` mode, the caller
/// can take only the first n columns of Q and first n rows of R.
pub fn qr_mxn(a: &[f64], m: usize, n: usize) -> Result<(Vec<f64>, Vec<f64>), LinAlgError> {
    if Some(a.len()) != m.checked_mul(n) || m == 0 || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "qr_mxn: input must be m*n with m,n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "matrix entries must be finite for QR",
        ));
    }

    // Q = I_m, R = copy of A
    let mut q = vec![0.0; m * m];
    for i in 0..m {
        q[i * m + i] = 1.0;
    }
    let mut r = a.to_vec();
    let k = m.min(n);

    // Householder vector v
    let mut v = vec![0.0; m];
    for col in 0..k {
        // Compute norm of column `col` below the diagonal
        let mut col_norm_sq = 0.0;
        for i in col..m {
            col_norm_sq += r[i * n + col] * r[i * n + col];
        }
        let col_norm = col_norm_sq.sqrt();
        if col_norm == 0.0 {
            continue;
        }

        let sign = if r[col * n + col] >= 0.0 { 1.0 } else { -1.0 };
        for vi in &mut v[..col] {
            *vi = 0.0;
        }
        for (i, vi) in v[col..m].iter_mut().enumerate() {
            *vi = r[(i + col) * n + col];
        }
        v[col] += sign * col_norm;
        let v_norm_sq: f64 = v[col..].iter().map(|x| x * x).sum();
        if v_norm_sq == 0.0 {
            continue;
        }

        // Apply H = I - 2*v*v^T/||v||^2 to R (only columns col..n)
        let scale = 2.0 / v_norm_sq;
        for j in col..n {
            let mut dot = 0.0;
            for i in col..m {
                dot += v[i] * r[i * n + j];
            }
            let factor = scale * dot;
            for i in col..m {
                r[i * n + j] -= factor * v[i];
            }
        }

        // Accumulate Q = Q * H (Q is m×m, so we update all m rows)
        for i in 0..m {
            let mut dot = 0.0;
            for j in col..m {
                dot += q[i * m + j] * v[j];
            }
            let factor = scale * dot;
            for j in col..m {
                q[i * m + j] -= factor * v[j];
            }
        }
    }

    Ok((q, r))
}

/// SVD of an m×n rectangular matrix (singular values only).
///
/// Input `a` is row-major with shape (m, n). Returns singular values in
/// descending order.
///
/// Uses the Golub-Kahan bidiagonalization approach:
/// 1. Bidiagonalize A using Householder reflections
/// 2. Apply implicit QR (Golub-Reinsch) to find singular values
///
/// For the full decomposition (U, S, V^T), use `svd_mxn_full`.
pub fn svd_mxn(a: &[f64], m: usize, n: usize) -> Result<Vec<f64>, LinAlgError> {
    if Some(a.len()) != m.checked_mul(n) || m == 0 || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "svd_mxn: input must be m*n with m,n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "matrix entries must be finite for SVD",
        ));
    }

    let sigmas = svd_bidiag_values(a, m, n)?;
    Ok(sigmas)
}

/// Full SVD of an m×n rectangular matrix: returns (U, S, Vt).
///
/// - U: m×m orthogonal matrix (row-major, m*m elements)
/// - S: min(m,n) singular values in descending order
/// - Vt: n×n orthogonal matrix (row-major, n*n elements)
///
/// Uses Golub-Kahan bidiagonalization + implicit QR (Golub-Reinsch):
/// 1. Householder bidiagonalization: A = U1 * B * V1^T
/// 2. Implicit QR on bidiagonal B to diagonalize: B = U2 * S * V2^T
/// 3. Final: U = U1*U2, Vt = V2^T * V1^T
pub fn svd_mxn_full(a: &[f64], m: usize, n: usize) -> Result<SvdFullResult, LinAlgError> {
    if Some(a.len()) != m.checked_mul(n) || m == 0 || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "svd_mxn_full: input must be m*n with m,n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "matrix entries must be finite for SVD",
        ));
    }

    svd_bidiag_full(a, m, n)
}

// ── Golub-Kahan Bidiagonalization SVD ────────────────────────────────────

/// Compute the smaller singular value of a 2×2 upper triangular matrix
/// `[[f, g], [0, h]]`. Used for the Wilkinson shift in implicit QR.
fn svd_shift_2x2(f: f64, g: f64, h: f64) -> f64 {
    let fh = f * h;
    if fh.abs() == 0.0 {
        return 0.0;
    }
    let s = f * f + g * g + h * h;
    let disc = (s * s - 4.0 * fh * fh).max(0.0).sqrt();
    // σ_min² = 2·f²h² / (s + √(s²−4f²h²))  [numerically stable form]
    let sigma_min_sq = 2.0 * fh * fh / (s + disc);
    sigma_min_sq.sqrt()
}

fn try_store_orthonormal_column(
    u: &mut [f64],
    m: usize,
    col: usize,
    candidate: &mut [f64],
    tol: f64,
) -> bool {
    for prev in 0..col {
        let mut dot = 0.0;
        for row in 0..m {
            dot += candidate[row] * u[row * m + prev];
        }
        for row in 0..m {
            candidate[row] -= dot * u[row * m + prev];
        }
    }

    let norm = candidate
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm <= tol {
        return false;
    }

    for row in 0..m {
        u[row * m + col] = candidate[row] / norm;
    }
    true
}

fn store_orthonormal_column(
    u: &mut [f64],
    m: usize,
    col: usize,
    seed: &[f64],
    tol: f64,
) -> Result<(), LinAlgError> {
    let mut candidate = seed.to_vec();
    if try_store_orthonormal_column(u, m, col, &mut candidate, tol) {
        return Ok(());
    }

    for basis in 0..m {
        candidate.fill(0.0);
        candidate[basis] = 1.0;
        if try_store_orthonormal_column(u, m, col, &mut candidate, tol) {
            return Ok(());
        }
    }

    Err(LinAlgError::SvdNonConvergence)
}

fn reconstruct_u_from_vt(
    a: &[f64],
    m: usize,
    n: usize,
    sigmas: &[f64],
    vt: &[f64],
) -> Result<Vec<f64>, LinAlgError> {
    let k = sigmas.len();
    let mut v = vec![0.0f64; n * k];
    for col in 0..k {
        for row in 0..n {
            v[row * k + col] = vt[col * n + row];
        }
    }

    let av = packed_gemm(a, &v, m, n, k);
    let sigma_max = sigmas.first().copied().unwrap_or(0.0);
    let tol = f64::EPSILON * (m.max(n) as f64) * sigma_max.max(1.0) * 8.0;
    let mut u = vec![0.0; m * m];
    let mut seed = vec![0.0; m];
    let mut candidate_norm_sq = vec![0.0; k];
    for row in 0..m {
        let av_row = &av[row * k..row * k + k];
        let u_row = &mut u[row * m..row * m + k];
        for col in 0..k {
            let sigma = sigmas[col];
            if sigma > tol {
                let candidate = av_row[col] / sigma;
                u_row[col] = candidate;
                candidate_norm_sq[col] += candidate * candidate;
            }
        }
    }
    for col in 0..k {
        seed.fill(0.0);
        let sigma = sigmas[col];
        if sigma > tol {
            let norm = candidate_norm_sq[col].sqrt();
            if norm.is_finite() && norm > tol {
                for row in 0..m {
                    u[row * m + col] /= norm;
                }
                continue;
            }
            for row in 0..m {
                seed[row] = u[row * m + col];
            }
        }
        store_orthonormal_column(&mut u, m, col, &seed, tol)?;
    }

    for col in k..m {
        seed.fill(0.0);
        store_orthonormal_column(&mut u, m, col, &seed, tol)?;
    }

    Ok(u)
}

struct SvdRightVtPanel {
    start: usize,
    len: usize,
    v: Vec<f64>,
    taus: Vec<f64>,
}

impl SvdRightVtPanel {
    fn new(n: usize) -> Self {
        Self {
            start: 0,
            len: 0,
            v: vec![0.0f64; n * SVD_RIGHT_VT_PANEL_NB],
            taus: vec![0.0f64; SVD_RIGHT_VT_PANEL_NB],
        }
    }

    fn push(&mut self, vt: &mut [f64], n: usize, reflector_col: usize, w_house: &[f64], tau: f64) {
        if self.len == SVD_RIGHT_VT_PANEL_NB {
            self.flush(vt, n);
        }
        if self.len == 0 {
            self.start = reflector_col;
            self.v.fill(0.0);
            self.taus.fill(0.0);
        }

        let slot = self.len;
        self.taus[slot] = tau;
        for (row, &value) in w_house
            .iter()
            .enumerate()
            .take(n)
            .skip(reflector_col + 1)
        {
            self.v[(row - self.start) * SVD_RIGHT_VT_PANEL_NB + slot] = value;
        }
        self.len += 1;
    }

    fn flush(&mut self, vt: &mut [f64], n: usize) {
        if self.len == 0 {
            return;
        }
        flush_svd_right_vt_panel(vt, n, self.start, self.len, &self.v, &self.taus);
        self.len = 0;
    }
}

fn flush_svd_right_vt_panel(
    vt: &mut [f64],
    n: usize,
    panel_start: usize,
    panel_len: usize,
    panel_v: &[f64],
    panel_taus: &[f64],
) {
    if panel_len == 0 {
        return;
    }

    let h = n - panel_start;
    let mut vv = vec![0.0f64; h * panel_len];
    for row in 0..h {
        for col in 0..panel_len {
            vv[row * panel_len + col] = panel_v[row * SVD_RIGHT_VT_PANEL_NB + col];
        }
    }

    let mut tm = vec![0.0f64; panel_len * panel_len];
    for t in 0..panel_len {
        let tau = panel_taus[t];
        tm[t * panel_len + t] = tau;
        if tau == 0.0 {
            continue;
        }
        let mut col = vec![0.0f64; t];
        for (i, ci) in col.iter_mut().enumerate() {
            let mut dot = 0.0;
            for row in 0..h {
                dot += vv[row * panel_len + i] * vv[row * panel_len + t];
            }
            *ci = -tau * dot;
        }
        for i in 0..t {
            let mut s = 0.0;
            for l in i..t {
                s += tm[i * panel_len + l] * col[l];
            }
            tm[i * panel_len + t] = s;
        }
    }

    let mut vtrans = vec![0.0f64; panel_len * h];
    for row in 0..h {
        for col in 0..panel_len {
            vtrans[col * h + row] = vv[row * panel_len + col];
        }
    }

    let mut tmt = vec![0.0f64; panel_len * panel_len];
    for i in 0..panel_len {
        for j in 0..panel_len {
            tmt[i * panel_len + j] = tm[j * panel_len + i];
        }
    }

    let mut vt_active = vec![0.0f64; h * n];
    for row in 0..h {
        let src = (panel_start + row) * n;
        vt_active[row * n..row * n + n].copy_from_slice(&vt[src..src + n]);
    }

    let w1 = packed_gemm(&vtrans, &vt_active, panel_len, h, n);
    let w2 = packed_gemm(&tmt, &w1, panel_len, panel_len, n);
    let update = packed_gemm(&vv, &w2, h, panel_len, n);
    for row in 0..h {
        let dst = (panel_start + row) * n;
        for (cell, &delta) in vt[dst..dst + n]
            .iter_mut()
            .zip(&update[row * n..row * n + n])
        {
            *cell -= delta;
        }
    }
}

fn svd_via_jacobi_full(a: &[f64], m: usize, n: usize) -> Result<SvdFullResult, LinAlgError> {
    let k = m.min(n);
    let mut b = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let a_norm = a.iter().map(|value| value * value).sum::<f64>().sqrt();
    let ortho_tol = f64::EPSILON * (m as f64) * a_norm.max(1.0) * a_norm.max(1.0) * 16.0;
    let mut converged = false;
    for _sweep in 0..SVD_QR_ITERATION_COEFF {
        let mut changed = false;
        for p in 0..n.saturating_sub(1) {
            for q in (p + 1)..n {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;
                for row in 0..m {
                    let bp = b[row * n + p];
                    let bq = b[row * n + q];
                    alpha += bp * bp;
                    beta += bq * bq;
                    gamma += bp * bq;
                }

                if gamma.abs() <= ortho_tol.max(f64::EPSILON * (alpha * beta).sqrt()) {
                    continue;
                }

                changed = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta == 0.0 {
                    1.0
                } else {
                    zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;

                for row in 0..m {
                    let bp = b[row * n + p];
                    let bq = b[row * n + q];
                    b[row * n + p] = c * bp - s * bq;
                    b[row * n + q] = s * bp + c * bq;
                }
                for row in 0..n {
                    let vp = v[row * n + p];
                    let vq = v[row * n + q];
                    v[row * n + p] = c * vp - s * vq;
                    v[row * n + q] = s * vp + c * vq;
                }
            }
        }

        if !changed {
            converged = true;
            break;
        }
    }

    if !converged {
        for p in 0..n.saturating_sub(1) {
            for q in (p + 1)..n {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;
                for row in 0..m {
                    let bp = b[row * n + p];
                    let bq = b[row * n + q];
                    alpha += bp * bp;
                    beta += bq * bq;
                    gamma += bp * bq;
                }
                if gamma.abs() > ortho_tol.max(f64::EPSILON * (alpha * beta).sqrt()) {
                    return Err(LinAlgError::SvdNonConvergence);
                }
            }
        }
    }

    let mut sigma_by_col = vec![0.0; n];
    for col in 0..n {
        let mut sigma_sq = 0.0;
        for row in 0..m {
            let value = b[row * n + col];
            sigma_sq += value * value;
        }
        sigma_by_col[col] = sigma_sq.sqrt();
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&lhs, &rhs| {
        sigma_by_col[rhs]
            .partial_cmp(&sigma_by_col[lhs])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sigma_max = order.first().map(|&idx| sigma_by_col[idx]).unwrap_or(0.0);
    let tol = f64::EPSILON * (m.max(n) as f64) * sigma_max.max(1.0) * 8.0;

    let mut sigmas = vec![0.0; k];
    let mut vt = vec![0.0; n * n];
    for (new_col, &old_col) in order.iter().enumerate() {
        for row in 0..n {
            vt[new_col * n + row] = v[row * n + old_col];
        }
        if new_col < k {
            sigmas[new_col] = sigma_by_col[old_col];
        }
    }

    let mut u = vec![0.0; m * m];
    for col in 0..k {
        let sigma = sigmas[col];
        let mut seed = vec![0.0; m];
        if sigma > tol {
            for row in 0..m {
                seed[row] = b[row * n + order[col]] / sigma;
            }
        }
        store_orthonormal_column(&mut u, m, col, &seed, tol)?;
    }

    for col in k..m {
        store_orthonormal_column(&mut u, m, col, &vec![0.0; m], tol)?;
    }

    Ok((u, sigmas, vt))
}

fn svd_bidiag_full_with_max_iters(
    a: &[f64],
    m: usize,
    n: usize,
    max_iter: usize,
) -> Result<SvdFullResult, LinAlgError> {
    if m < n {
        let mut at = vec![0.0; n * m];
        for i in 0..m {
            for j in 0..n {
                at[j * m + i] = a[i * n + j];
            }
        }
        let (u_t, sigmas, vt_t) = svd_bidiag_full_with_max_iters(&at, n, m, max_iter)?;
        let u = transpose_mat(&vt_t, m, m);
        let vt = transpose_mat(&u_t, n, n);
        return Ok((u, sigmas, vt));
    }

    match svd_bidiag_qr_full(a, m, n, max_iter) {
        Ok(result) => Ok(result),
        Err(LinAlgError::SvdNonConvergence) => svd_via_jacobi_full(a, m, n),
        Err(err) => Err(err),
    }
}

/// Core SVD implementation using Golub-Kahan bidiagonalization + Golub-Reinsch QR.
///
/// Returns (U, sigmas, Vt) where U is m×m, sigmas has min(m,n) entries, Vt is n×n.
fn svd_bidiag_full(a: &[f64], m: usize, n: usize) -> Result<SvdFullResult, LinAlgError> {
    let k = m.min(n);
    svd_bidiag_full_with_max_iters(a, m, n, SVD_QR_ITERATION_COEFF * k * k)
}

#[inline]
fn svd_apply_fused_two_sided_row_pair(
    row0_tail: &mut [f64],
    row1_tail: &mut [f64],
    vi0: f64,
    vi1: f64,
    lh_tail: &[f64],
    w_tail: &[f64],
    right_scale: f64,
) {
    debug_assert_eq!(row0_tail.len(), row1_tail.len());
    debug_assert_eq!(row0_tail.len(), lh_tail.len());
    debug_assert_eq!(row0_tail.len(), w_tail.len());

    let tail_len = row0_tail.len();
    let blocked_len =
        (tail_len / SVD_FUSED_TWO_SIDED_REGISTER_BLOCK) * SVD_FUSED_TWO_SIDED_REGISTER_BLOCK;
    let mut dot0 = 0.0;
    let mut dot1 = 0.0;
    let mut offset = 0;
    while offset < blocked_len {
        row0_tail[offset] -= lh_tail[offset] * vi0;
        dot0 += row0_tail[offset] * w_tail[offset];
        row1_tail[offset] -= lh_tail[offset] * vi1;
        dot1 += row1_tail[offset] * w_tail[offset];
        row0_tail[offset + 1] -= lh_tail[offset + 1] * vi0;
        dot0 += row0_tail[offset + 1] * w_tail[offset + 1];
        row1_tail[offset + 1] -= lh_tail[offset + 1] * vi1;
        dot1 += row1_tail[offset + 1] * w_tail[offset + 1];
        row0_tail[offset + 2] -= lh_tail[offset + 2] * vi0;
        dot0 += row0_tail[offset + 2] * w_tail[offset + 2];
        row1_tail[offset + 2] -= lh_tail[offset + 2] * vi1;
        dot1 += row1_tail[offset + 2] * w_tail[offset + 2];
        row0_tail[offset + 3] -= lh_tail[offset + 3] * vi0;
        dot0 += row0_tail[offset + 3] * w_tail[offset + 3];
        row1_tail[offset + 3] -= lh_tail[offset + 3] * vi1;
        dot1 += row1_tail[offset + 3] * w_tail[offset + 3];
        offset += SVD_FUSED_TWO_SIDED_REGISTER_BLOCK;
    }
    for offset in blocked_len..tail_len {
        row0_tail[offset] -= lh_tail[offset] * vi0;
        dot0 += row0_tail[offset] * w_tail[offset];
        row1_tail[offset] -= lh_tail[offset] * vi1;
        dot1 += row1_tail[offset] * w_tail[offset];
    }

    let f0 = right_scale * dot0;
    let f1 = right_scale * dot1;
    let mut offset = 0;
    while offset < blocked_len {
        row0_tail[offset] -= f0 * w_tail[offset];
        row1_tail[offset] -= f1 * w_tail[offset];
        row0_tail[offset + 1] -= f0 * w_tail[offset + 1];
        row1_tail[offset + 1] -= f1 * w_tail[offset + 1];
        row0_tail[offset + 2] -= f0 * w_tail[offset + 2];
        row1_tail[offset + 2] -= f1 * w_tail[offset + 2];
        row0_tail[offset + 3] -= f0 * w_tail[offset + 3];
        row1_tail[offset + 3] -= f1 * w_tail[offset + 3];
        offset += SVD_FUSED_TWO_SIDED_REGISTER_BLOCK;
    }
    for offset in blocked_len..tail_len {
        row0_tail[offset] -= f0 * w_tail[offset];
        row1_tail[offset] -= f1 * w_tail[offset];
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn svd_apply_fused_two_sided_row_quad(
    row0_tail: &mut [f64],
    row1_tail: &mut [f64],
    row2_tail: &mut [f64],
    row3_tail: &mut [f64],
    vis: [f64; 4],
    lh_tail: &[f64],
    w_tail: &[f64],
    right_scale: f64,
) {
    debug_assert_eq!(row0_tail.len(), row1_tail.len());
    debug_assert_eq!(row0_tail.len(), row2_tail.len());
    debug_assert_eq!(row0_tail.len(), row3_tail.len());
    debug_assert_eq!(row0_tail.len(), lh_tail.len());
    debug_assert_eq!(row0_tail.len(), w_tail.len());

    let tail_len = row0_tail.len();
    let blocked_len =
        (tail_len / SVD_FUSED_TWO_SIDED_REGISTER_BLOCK) * SVD_FUSED_TWO_SIDED_REGISTER_BLOCK;
    let mut dots = [0.0; 4];
    let mut offset = 0;
    while offset < blocked_len {
        for lane in 0..SVD_FUSED_TWO_SIDED_REGISTER_BLOCK {
            let idx = offset + lane;
            let lh = lh_tail[idx];
            let w = w_tail[idx];
            row0_tail[idx] -= lh * vis[0];
            dots[0] += row0_tail[idx] * w;
            row1_tail[idx] -= lh * vis[1];
            dots[1] += row1_tail[idx] * w;
            row2_tail[idx] -= lh * vis[2];
            dots[2] += row2_tail[idx] * w;
            row3_tail[idx] -= lh * vis[3];
            dots[3] += row3_tail[idx] * w;
        }
        offset += SVD_FUSED_TWO_SIDED_REGISTER_BLOCK;
    }
    for idx in blocked_len..tail_len {
        let lh = lh_tail[idx];
        let w = w_tail[idx];
        row0_tail[idx] -= lh * vis[0];
        dots[0] += row0_tail[idx] * w;
        row1_tail[idx] -= lh * vis[1];
        dots[1] += row1_tail[idx] * w;
        row2_tail[idx] -= lh * vis[2];
        dots[2] += row2_tail[idx] * w;
        row3_tail[idx] -= lh * vis[3];
        dots[3] += row3_tail[idx] * w;
    }

    let fs = [
        right_scale * dots[0],
        right_scale * dots[1],
        right_scale * dots[2],
        right_scale * dots[3],
    ];
    let mut offset = 0;
    while offset < blocked_len {
        for lane in 0..SVD_FUSED_TWO_SIDED_REGISTER_BLOCK {
            let idx = offset + lane;
            let w = w_tail[idx];
            row0_tail[idx] -= fs[0] * w;
            row1_tail[idx] -= fs[1] * w;
            row2_tail[idx] -= fs[2] * w;
            row3_tail[idx] -= fs[3] * w;
        }
        offset += SVD_FUSED_TWO_SIDED_REGISTER_BLOCK;
    }
    for idx in blocked_len..tail_len {
        let w = w_tail[idx];
        row0_tail[idx] -= fs[0] * w;
        row1_tail[idx] -= fs[1] * w;
        row2_tail[idx] -= fs[2] * w;
        row3_tail[idx] -= fs[3] * w;
    }
}

fn svd_bidiag_values_with_max_iters(
    a: &[f64],
    m: usize,
    n: usize,
    max_iter: usize,
) -> Result<Vec<f64>, LinAlgError> {
    if m < n {
        let mut at = vec![0.0; n * m];
        for i in 0..m {
            for j in 0..n {
                at[j * m + i] = a[i * n + j];
            }
        }
        return svd_bidiag_values_with_max_iters(&at, n, m, max_iter);
    }

    match svd_bidiag_qr_values(a, m, n, max_iter) {
        Ok(sigmas) => Ok(sigmas),
        Err(LinAlgError::SvdNonConvergence) => {
            let (_, sigmas, _) = svd_via_jacobi_full(a, m, n)?;
            Ok(sigmas)
        }
        Err(err) => Err(err),
    }
}

fn svd_bidiag_values(a: &[f64], m: usize, n: usize) -> Result<Vec<f64>, LinAlgError> {
    let k = m.min(n);
    svd_bidiag_values_with_max_iters(a, m, n, SVD_QR_ITERATION_COEFF * k * k)
}

#[inline]
fn sort_singular_values_descending_in_place(values: &mut [f64]) {
    values.sort_by(|a, b| {
        b.partial_cmp(a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Core QR-phase SVD implementation for the bidiagonalized matrix.
///
/// Requires `m >= n`; transpose handling and fallback are done by the wrapper.
fn svd_bidiag_qr_full(
    a: &[f64],
    m: usize,
    n: usize,
    max_iter: usize,
) -> Result<SvdFullResult, LinAlgError> {
    let k = m.min(n);

    // Now m >= n. Bidiagonalize A into U1 * B * V1^T
    // B is n×n upper bidiagonal: diagonal d[0..n], superdiagonal e[0..n-1]
    let mut work = a.to_vec(); // m×n working copy
    let mut d = vec![0.0; n]; // diagonal
    let mut e = vec![0.0; n.saturating_sub(1)]; // superdiagonal
    let reconstruct_left = m >= SVD_RECONSTRUCT_LEFT_MIN_DIM && n >= SVD_RECONSTRUCT_LEFT_MIN_DIM;
    let block_right_vt =
        reconstruct_left && m == n && n >= SVD_RIGHT_VT_BLOCK_MIN_DIM && max_iter > 0;

    // U1^T = transpose of the product of left Householder reflections. Keeping
    // this accumulator transposed makes both Householder and QR left rotations
    // row-contiguous while preserving the same scalar recurrence.
    let mut u_t = if reconstruct_left {
        Vec::new()
    } else {
        let mut identity = vec![0.0; m * m];
        for i in 0..m {
            identity[i * m + i] = 1.0;
        }
        identity
    };
    // V1 = product of right Householder reflections (n×n)
    let mut vt = vec![0.0; n * n];
    for i in 0..n {
        vt[i * n + i] = 1.0;
    }
    let mut right_vt_panel = if block_right_vt {
        Some(SvdRightVtPanel::new(n))
    } else {
        None
    };

    let mut v_house = vec![0.0; m];
    let mut w_house = vec![0.0; n];
    // Scratch for the cache-friendly two-pass left Householder transform.
    let mut lh_dot = vec![0.0; n];
    let mut lh_f = vec![0.0; n];
    // Scratch for the cache-friendly two-pass right Householder Vt accumulation.
    let mut rh_dot = vec![0.0; n];
    let mut rh_f = vec![0.0; n];
    // Scratch for transposed-U left Householder accumulation.
    let mut uh_dot = if reconstruct_left {
        Vec::new()
    } else {
        vec![0.0; m]
    };
    let mut uh_f = if reconstruct_left {
        Vec::new()
    } else {
        vec![0.0; m]
    };

    for j in 0..n {
        let mut fused_two_sided = false;
        // Left Householder: zero out column j below diagonal
        for vi in &mut v_house[..j] {
            *vi = 0.0;
        }
        let col_norm = {
            let mut s = 0.0;
            for (i, vi) in v_house.iter_mut().enumerate().take(m).skip(j) {
                let value = work[i * n + j];
                *vi = value;
                s += value * value;
            }
            s.sqrt()
        };
        if col_norm > 0.0 {
            let sign = if work[j * n + j] >= 0.0 { 1.0 } else { -1.0 };
            v_house[j] += sign * col_norm;
            let v_norm_sq: f64 = v_house[j..].iter().map(|x| x * x).sum();
            if v_norm_sq > 0.0 {
                let scale = 2.0 / v_norm_sq;
                let can_fuse_two_sided = reconstruct_left
                    && j + 2 <= n
                    && m - j >= SVD_FUSED_TWO_SIDED_MIN_TRAIL
                    && n - j > SVD_FUSED_TWO_SIDED_MIN_TRAIL;
                if can_fuse_two_sided {
                    // Large full-SVD reconstructs U from A*V/S, so the left
                    // reflector only has to update the active work panel. Fuse
                    // the left and right trailing updates: materialize the pivot
                    // row to build the right reflector, then write rows j+1..m
                    // directly to their post-left/post-right state. Each scalar
                    // `left_value` is computed with the same multiply/subtract
                    // sequence and each right-dot still walks columns ascending.
                    for x in lh_dot[j..n].iter_mut() {
                        *x = 0.0;
                    }
                    let lh_tail_len = n - j;
                    let lh_blocked_len = (lh_tail_len / SVD_FUSED_TWO_SIDED_REGISTER_BLOCK)
                        * SVD_FUSED_TWO_SIDED_REGISTER_BLOCK;
                    let mut lh_offset = 0;
                    while lh_offset < lh_blocked_len {
                        let col = j + lh_offset;
                        let mut dot0 = 0.0;
                        let mut dot1 = 0.0;
                        let mut dot2 = 0.0;
                        let mut dot3 = 0.0;
                        for (i, &vi) in v_house.iter().enumerate().take(m).skip(j) {
                            let row_base = i * n + col;
                            dot0 += vi * work[row_base];
                            dot1 += vi * work[row_base + 1];
                            dot2 += vi * work[row_base + 2];
                            dot3 += vi * work[row_base + 3];
                        }
                        lh_dot[col] = dot0;
                        lh_dot[col + 1] = dot1;
                        lh_dot[col + 2] = dot2;
                        lh_dot[col + 3] = dot3;
                        lh_offset += SVD_FUSED_TWO_SIDED_REGISTER_BLOCK;
                    }
                    for col in (j + lh_blocked_len)..n {
                        let mut dot = 0.0;
                        for (i, &vi) in v_house.iter().enumerate().take(m).skip(j) {
                            dot += vi * work[i * n + col];
                        }
                        lh_dot[col] = dot;
                    }
                    for (fc, &dc) in lh_f[j..n].iter_mut().zip(lh_dot[j..n].iter()) {
                        *fc = scale * dc;
                    }

                    let pivot_base = j * n;
                    let pivot_vi = v_house[j];
                    for (w, &fc) in work[pivot_base + j..pivot_base + n]
                        .iter_mut()
                        .zip(lh_f[j..n].iter())
                    {
                        *w -= fc * pivot_vi;
                    }
                    d[j] = work[pivot_base + j];

                    for wi in &mut w_house[..=j] {
                        *wi = 0.0;
                    }
                    let row_norm = {
                        let mut s = 0.0;
                        for (col, wi) in w_house.iter_mut().enumerate().take(n).skip(j + 1) {
                            let value = work[pivot_base + col];
                            *wi = value;
                            s += value * value;
                        }
                        s.sqrt()
                    };
                    if row_norm > 0.0 {
                        let sign = if work[pivot_base + j + 1] >= 0.0 {
                            1.0
                        } else {
                            -1.0
                        };
                        w_house[j + 1] += sign * row_norm;
                        let w_norm_sq: f64 = w_house[(j + 1)..].iter().map(|x| x * x).sum();
                        if w_norm_sq > 0.0 {
                            let right_scale = 2.0 / w_norm_sq;
                            let mut pivot_dot = 0.0;
                            for col in (j + 1)..n {
                                pivot_dot += work[pivot_base + col] * w_house[col];
                            }
                            let pivot_f = right_scale * pivot_dot;
                            for col in (j + 1)..n {
                                work[pivot_base + col] -= pivot_f * w_house[col];
                            }
                            e[j] = work[pivot_base + j + 1];

                            let lh_tail = &lh_f[j + 1..n];
                            let w_tail = &w_house[j + 1..n];
                            let mut row = j + 1;
                            while row + 3 < m {
                                let rows = &mut work[row * n..(row + 4) * n];
                                let (row0, rows) = rows.split_at_mut(n);
                                let (row1, rows) = rows.split_at_mut(n);
                                let (row2, row3) = rows.split_at_mut(n);
                                svd_apply_fused_two_sided_row_quad(
                                    &mut row0[j + 1..n],
                                    &mut row1[j + 1..n],
                                    &mut row2[j + 1..n],
                                    &mut row3[j + 1..n],
                                    [
                                        v_house[row],
                                        v_house[row + 1],
                                        v_house[row + 2],
                                        v_house[row + 3],
                                    ],
                                    lh_tail,
                                    w_tail,
                                    right_scale,
                                );
                                row += 4;
                            }
                            while row + 1 < m {
                                let row0_base = row * n;
                                let row1_base = (row + 1) * n;
                                let (head, tail) = work.split_at_mut(row1_base);
                                let row0_tail = &mut head[row0_base + j + 1..row0_base + n];
                                let row1_tail = &mut tail[j + 1..n];
                                svd_apply_fused_two_sided_row_pair(
                                    row0_tail,
                                    row1_tail,
                                    v_house[row],
                                    v_house[row + 1],
                                    lh_tail,
                                    w_tail,
                                    right_scale,
                                );
                                row += 2;
                            }
                            if row < m {
                                let vi = v_house[row];
                                let row_base = row * n;
                                let row_tail = &mut work[row_base + j + 1..row_base + n];
                                let tail_len = row_tail.len();
                                let blocked_len = (tail_len / SVD_FUSED_TWO_SIDED_REGISTER_BLOCK)
                                    * SVD_FUSED_TWO_SIDED_REGISTER_BLOCK;
                                let mut dot = 0.0;
                                let mut offset = 0;
                                while offset < blocked_len {
                                    row_tail[offset] -= lh_tail[offset] * vi;
                                    dot += row_tail[offset] * w_tail[offset];
                                    row_tail[offset + 1] -= lh_tail[offset + 1] * vi;
                                    dot += row_tail[offset + 1] * w_tail[offset + 1];
                                    row_tail[offset + 2] -= lh_tail[offset + 2] * vi;
                                    dot += row_tail[offset + 2] * w_tail[offset + 2];
                                    row_tail[offset + 3] -= lh_tail[offset + 3] * vi;
                                    dot += row_tail[offset + 3] * w_tail[offset + 3];
                                    offset += SVD_FUSED_TWO_SIDED_REGISTER_BLOCK;
                                }
                                for offset in blocked_len..tail_len {
                                    row_tail[offset] -= lh_tail[offset] * vi;
                                    dot += row_tail[offset] * w_tail[offset];
                                }
                                let f = right_scale * dot;
                                let mut offset = 0;
                                while offset < blocked_len {
                                    row_tail[offset] -= f * w_tail[offset];
                                    row_tail[offset + 1] -= f * w_tail[offset + 1];
                                    row_tail[offset + 2] -= f * w_tail[offset + 2];
                                    row_tail[offset + 3] -= f * w_tail[offset + 3];
                                    offset += SVD_FUSED_TWO_SIDED_REGISTER_BLOCK;
                                }
                                for offset in blocked_len..tail_len {
                                    row_tail[offset] -= f * w_tail[offset];
                                }
                            }

                            if block_right_vt {
                                if let Some(panel) = right_vt_panel.as_mut() {
                                    panel.push(&mut vt, n, j, &w_house, right_scale);
                                }
                            } else {
                                // Accumulate into Vt: Vt = (I - scale*w*w^T) * Vt.
                                for x in rh_dot.iter_mut() {
                                    *x = 0.0;
                                }
                                for row in (j + 1)..n {
                                    let wr = w_house[row];
                                    let vt_row = &vt[row * n..row * n + n];
                                    for (dot, &value) in rh_dot.iter_mut().zip(vt_row.iter()) {
                                        *dot += wr * value;
                                    }
                                }
                                for (fc, &dot) in rh_f.iter_mut().zip(rh_dot.iter()) {
                                    *fc = right_scale * dot;
                                }
                                for row in (j + 1)..n {
                                    let wr = w_house[row];
                                    let vt_row = &mut vt[row * n..row * n + n];
                                    for (value, &fc) in vt_row.iter_mut().zip(rh_f.iter()) {
                                        *value -= fc * wr;
                                    }
                                }
                            }
                            fused_two_sided = true;
                        }
                    }

                    if !fused_two_sided {
                        for (i, &vi) in v_house.iter().enumerate().take(m).skip(j + 1) {
                            let row = &mut work[i * n + j + 1..i * n + n];
                            for (w, &fc) in row.iter_mut().zip(lh_f[(j + 1)..n].iter()) {
                                *w -= fc * vi;
                            }
                        }
                    }
                } else {
                    // Apply to work (left) as two row-contiguous passes (bit-exact)
                    // instead of striding work[i*n+col] down each column: pass 1 sums
                    // lh_dot[col] = Σ_i v_house[i]·work[i][col] in i-ascending order,
                    // pass 2 applies the identical (scale·lh_dot[col])·v_house[i]
                    // product per element.
                    for x in lh_dot[j..n].iter_mut() {
                        *x = 0.0;
                    }
                    for i in j..m {
                        let vi = v_house[i];
                        let row = &work[i * n + j..i * n + n];
                        for (x, &w) in lh_dot[j..n].iter_mut().zip(row.iter()) {
                            *x += vi * w;
                        }
                    }
                    for (fc, &dc) in lh_f[j..n].iter_mut().zip(lh_dot[j..n].iter()) {
                        *fc = scale * dc;
                    }
                    for i in j..m {
                        let vi = v_house[i];
                        let row = &mut work[i * n + j..i * n + n];
                        for (w, &fc) in row.iter_mut().zip(lh_f[j..n].iter()) {
                            *w -= fc * vi;
                        }
                    }
                }
                if !reconstruct_left && !fused_two_sided {
                    // Accumulate into U^T: U^T = (I - scale*v*v^T) * U^T.
                    // This is the exact transpose of U = U * H. For each output
                    // element the dot still runs reflector indices in ascending
                    // order, and the update uses the same scale*dot then *v scalar
                    // sequence as the row-major U path.
                    for x in uh_dot.iter_mut() {
                        *x = 0.0;
                    }
                    for i in j..m {
                        let vi = v_house[i];
                        let u_row = &u_t[i * m..i * m + m];
                        for (dot, &value) in uh_dot.iter_mut().zip(u_row.iter()) {
                            *dot += vi * value;
                        }
                    }
                    for (fc, &dot) in uh_f.iter_mut().zip(uh_dot.iter()) {
                        *fc = scale * dot;
                    }
                    for i in j..m {
                        let vi = v_house[i];
                        let u_row = &mut u_t[i * m..i * m + m];
                        for (value, &fc) in u_row.iter_mut().zip(uh_f.iter()) {
                            *value -= fc * vi;
                        }
                    }
                }
            }
        }
        if !fused_two_sided {
            d[j] = work[j * n + j];
        }

        // Right Householder: zero out row j to the right of superdiagonal
        if !fused_two_sided && j + 2 <= n {
            for wi in &mut w_house[..=j] {
                *wi = 0.0;
            }
            let row_norm = {
                let mut s = 0.0;
                for (col, wi) in w_house.iter_mut().enumerate().take(n).skip(j + 1) {
                    let value = work[j * n + col];
                    *wi = value;
                    s += value * value;
                }
                s.sqrt()
            };
            if row_norm > 0.0 {
                let sign = if work[j * n + j + 1] >= 0.0 {
                    1.0
                } else {
                    -1.0
                };
                w_house[j + 1] += sign * row_norm;
                let w_norm_sq: f64 = w_house[(j + 1)..].iter().map(|x| x * x).sum();
                if w_norm_sq > 0.0 {
                    let scale = 2.0 / w_norm_sq;
                    // Apply to work (right): work = work * (I - scale*w*w^T)
                    for row in j..m {
                        let mut dot = 0.0;
                        for col in (j + 1)..n {
                            dot += work[row * n + col] * w_house[col];
                        }
                        let f = scale * dot;
                        for col in (j + 1)..n {
                            work[row * n + col] -= f * w_house[col];
                        }
                    }
                    if block_right_vt {
                        if let Some(panel) = right_vt_panel.as_mut() {
                            panel.push(&mut vt, n, j, &w_house, scale);
                        }
                    } else {
                        // Accumulate into Vt: Vt = (I - scale*w*w^T) * Vt.
                        // Keep each column's row-ascending dot order, but compute and
                        // apply all columns through row-contiguous Vt slices.
                        for x in rh_dot.iter_mut() {
                            *x = 0.0;
                        }
                        for row in (j + 1)..n {
                            let wr = w_house[row];
                            let vt_row = &vt[row * n..row * n + n];
                            for (dot, &value) in rh_dot.iter_mut().zip(vt_row.iter()) {
                                *dot += wr * value;
                            }
                        }
                        for (fc, &dot) in rh_f.iter_mut().zip(rh_dot.iter()) {
                            *fc = scale * dot;
                        }
                        for row in (j + 1)..n {
                            let wr = w_house[row];
                            let vt_row = &mut vt[row * n..row * n + n];
                            for (value, &fc) in vt_row.iter_mut().zip(rh_f.iter()) {
                                *value -= fc * wr;
                            }
                        }
                    }
                }
            }
            if j < n - 1 {
                e[j] = work[j * n + j + 1];
            }
        } else if !fused_two_sided && j < n - 1 {
            e[j] = work[j * n + j + 1];
        }
    }
    if let Some(panel) = right_vt_panel.as_mut() {
        panel.flush(&mut vt, n);
    }

    // Phase 2: Golub-Reinsch implicit QR iteration directly on the bidiagonal.
    // Works with (d, e) without forming B^T*B, avoiding condition-number squaring.
    // Left rotations accumulate into U (column operations), right into Vt (row operations).
    //
    // The left Givens rotations touch two COLUMNS of U. Because phase 1 already
    // kept U transposed, each rotation below touches two contiguous rows of u_t.
    let eps_mach = f64::EPSILON;

    let mut converged = false;
    for _iter in 0..max_iter {
        // Deflation: set small superdiagonal elements to zero
        for i in 0..e.len() {
            if e[i].abs() <= eps_mach * (d[i].abs() + d[i + 1].abs()) {
                e[i] = 0.0;
            }
        }

        // Find the largest unreduced block [lo..=hi]
        let mut hi = n - 1;
        while hi > 0 && e[hi - 1] == 0.0 {
            hi -= 1;
        }
        if hi == 0 {
            converged = true;
            break; // Fully converged
        }
        let mut lo = hi - 1;
        while lo > 0 && e[lo - 1] != 0.0 {
            lo -= 1;
        }

        // Compute Wilkinson shift: smallest singular value of trailing 2×2 of B
        // [[d[hi-1], e[hi-1]], [0, d[hi]]]
        let shift = svd_shift_2x2(d[hi - 1], e[hi - 1], d[hi]);

        // Initialize bulge chase: f = (d[lo]²-shift²)/d[lo], g = e[lo]
        let (mut f, mut g) = if d[lo].abs() > 0.0 {
            let sign_d = if d[lo] >= 0.0 { 1.0 } else { -1.0 };
            ((d[lo].abs() - shift) * (sign_d + shift / d[lo]), e[lo])
        } else {
            (0.0, e[lo])
        };

        // Bulge chase (LAPACK dbdsqr pattern)
        for kk in lo..hi {
            // Right Givens rotation: zero g in [f, g]
            let r = f.hypot(g);
            let (cs, sn) = if r > 0.0 { (f / r, g / r) } else { (1.0, 0.0) };

            if kk > lo {
                e[kk - 1] = r;
            }

            f = cs * d[kk] + sn * e[kk];
            e[kk] = cs * e[kk] - sn * d[kk];
            g = sn * d[kk + 1];
            d[kk + 1] *= cs;

            // Accumulate right rotation into Vt (rows kk, kk+1)
            for col in 0..n {
                let t1 = vt[kk * n + col];
                let t2 = vt[(kk + 1) * n + col];
                vt[kk * n + col] = cs * t1 + sn * t2;
                vt[(kk + 1) * n + col] = -sn * t1 + cs * t2;
            }

            // Left Givens rotation: zero g in [f, g]
            let r = f.hypot(g);
            let (cs, sn) = if r > 0.0 { (f / r, g / r) } else { (1.0, 0.0) };

            d[kk] = r;
            f = cs * e[kk] + sn * d[kk + 1];
            d[kk + 1] = cs * d[kk + 1] - sn * e[kk];

            if kk + 1 < hi {
                g = sn * e[kk + 1];
                e[kk + 1] *= cs;
            }

            if !reconstruct_left {
                // Accumulate left rotation into U's columns kk, kk+1, applied as a
                // contiguous rotation of rows kk, kk+1 of the transpose u_t.
                let (head, tail) = u_t.split_at_mut((kk + 1) * m);
                let row_k = &mut head[kk * m..kk * m + m];
                let row_k1 = &mut tail[..m];
                for r in 0..m {
                    let t1 = row_k[r];
                    let t2 = row_k1[r];
                    row_k[r] = cs * t1 + sn * t2;
                    row_k1[r] = -sn * t1 + cs * t2;
                }
            }
        }
        e[hi - 1] = f;
    }

    if !converged {
        return Err(LinAlgError::SvdNonConvergence);
    }

    // Make all singular values non-negative
    for j in 0..n {
        if d[j] < 0.0 {
            d[j] = -d[j];
            for col in 0..n {
                vt[j * n + col] = -vt[j * n + col];
            }
        }
    }

    // Sort descending by singular value and reorder U columns / Vt rows
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| d[b].partial_cmp(&d[a]).unwrap_or(std::cmp::Ordering::Equal));

    let sigmas: Vec<f64> = order.iter().take(k).map(|&i| d[i]).collect();

    let mut sorted_vt = vec![0.0; n * n];
    for (new_idx, &old_idx) in order.iter().enumerate() {
        for col in 0..n {
            sorted_vt[new_idx * n + col] = vt[old_idx * n + col];
        }
    }

    let u = if reconstruct_left {
        reconstruct_u_from_vt(a, m, n, &sigmas, &sorted_vt)?
    } else {
        // Transpose the accumulated rotations directly into sorted U columns.
        // This preserves the same stable singular-value ordering as the former
        // transpose-then-clone reorder, while avoiding an extra m*m clone/copy.
        let mut u = vec![0.0; m * m];
        for (new_idx, &old_idx) in order.iter().enumerate() {
            for row in 0..m {
                u[row * m + new_idx] = u_t[old_idx * m + row];
            }
        }
        for col in n..m {
            for row in 0..m {
                u[row * m + col] = u_t[col * m + row];
            }
        }
        u
    };

    Ok((u, sigmas, sorted_vt))
}

fn svd_bidiag_qr_values(
    a: &[f64],
    m: usize,
    n: usize,
    max_iter: usize,
) -> Result<Vec<f64>, LinAlgError> {
    let k = m.min(n);

    let mut work = a.to_vec();
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n.saturating_sub(1)];
    let mut v_house = vec![0.0; m];
    let mut w_house = vec![0.0; n];
    // Scratch for the cache-friendly two-pass left Householder transform.
    let mut lh_dot = vec![0.0; n];
    let mut lh_f = vec![0.0; n];

    for j in 0..n {
        for vi in &mut v_house[..j] {
            *vi = 0.0;
        }
        let col_norm = {
            let mut s = 0.0;
            for (i, vi) in v_house.iter_mut().enumerate().take(m).skip(j) {
                let value = work[i * n + j];
                *vi = value;
                s += value * value;
            }
            s.sqrt()
        };
        if col_norm > 0.0 {
            let sign = if work[j * n + j] >= 0.0 { 1.0 } else { -1.0 };
            v_house[j] += sign * col_norm;
            let v_norm_sq: f64 = v_house[j..].iter().map(|x| x * x).sum();
            if v_norm_sq > 0.0 {
                let scale = 2.0 / v_norm_sq;
                // Two row-contiguous passes (bit-exact) instead of striding
                // work[i*n+col] down each column with stride n: pass 1 sums
                // lh_dot[col] = Σ_i v_house[i]·work[i][col] in i-ascending order,
                // pass 2 applies the identical (scale·lh_dot[col])·v_house[i]
                // product per element.
                for x in lh_dot[j..n].iter_mut() {
                    *x = 0.0;
                }
                for i in j..m {
                    let vi = v_house[i];
                    let row = &work[i * n + j..i * n + n];
                    for (x, &w) in lh_dot[j..n].iter_mut().zip(row.iter()) {
                        *x += vi * w;
                    }
                }
                for (fc, &dc) in lh_f[j..n].iter_mut().zip(lh_dot[j..n].iter()) {
                    *fc = scale * dc;
                }
                for i in j..m {
                    let vi = v_house[i];
                    let row = &mut work[i * n + j..i * n + n];
                    for (w, &fc) in row.iter_mut().zip(lh_f[j..n].iter()) {
                        *w -= fc * vi;
                    }
                }
            }
        }
        d[j] = work[j * n + j];

        if j + 2 <= n {
            for wi in &mut w_house[..=j] {
                *wi = 0.0;
            }
            let row_norm = {
                let mut s = 0.0;
                for (col, wi) in w_house.iter_mut().enumerate().take(n).skip(j + 1) {
                    let value = work[j * n + col];
                    *wi = value;
                    s += value * value;
                }
                s.sqrt()
            };
            if row_norm > 0.0 {
                let sign = if work[j * n + j + 1] >= 0.0 {
                    1.0
                } else {
                    -1.0
                };
                w_house[j + 1] += sign * row_norm;
                let w_norm_sq: f64 = w_house[(j + 1)..].iter().map(|x| x * x).sum();
                if w_norm_sq > 0.0 {
                    let scale = 2.0 / w_norm_sq;
                    let w_tail = &w_house[(j + 1)..n];
                    for row in j..m {
                        let mut dot = 0.0;
                        let row_tail_start = row * n + j + 1;
                        let row_tail_end = row * n + n;
                        let row_tail = &mut work[row_tail_start..row_tail_end];
                        for (&value, &weight) in row_tail.iter().zip(w_tail.iter()) {
                            dot += value * weight;
                        }
                        let f = scale * dot;
                        for (value, &weight) in row_tail.iter_mut().zip(w_tail.iter()) {
                            *value -= f * weight;
                        }
                    }
                }
            }
            if j < n - 1 {
                e[j] = work[j * n + j + 1];
            }
        } else if j < n - 1 {
            e[j] = work[j * n + j + 1];
        }
    }

    let eps_mach = f64::EPSILON;
    let mut converged = false;
    for _iter in 0..max_iter {
        for i in 0..e.len() {
            if e[i].abs() <= eps_mach * (d[i].abs() + d[i + 1].abs()) {
                e[i] = 0.0;
            }
        }

        let mut hi = n - 1;
        while hi > 0 && e[hi - 1] == 0.0 {
            hi -= 1;
        }
        if hi == 0 {
            converged = true;
            break;
        }
        let mut lo = hi - 1;
        while lo > 0 && e[lo - 1] != 0.0 {
            lo -= 1;
        }

        let shift = svd_shift_2x2(d[hi - 1], e[hi - 1], d[hi]);
        let (mut f, mut g) = if d[lo].abs() > 0.0 {
            let sign_d = if d[lo] >= 0.0 { 1.0 } else { -1.0 };
            ((d[lo].abs() - shift) * (sign_d + shift / d[lo]), e[lo])
        } else {
            (0.0, e[lo])
        };

        for kk in lo..hi {
            let r = f.hypot(g);
            let (cs, sn) = if r > 0.0 { (f / r, g / r) } else { (1.0, 0.0) };

            if kk > lo {
                e[kk - 1] = r;
            }

            f = cs * d[kk] + sn * e[kk];
            e[kk] = cs * e[kk] - sn * d[kk];
            g = sn * d[kk + 1];
            d[kk + 1] *= cs;

            let r = f.hypot(g);
            let (cs, sn) = if r > 0.0 { (f / r, g / r) } else { (1.0, 0.0) };

            d[kk] = r;
            f = cs * e[kk] + sn * d[kk + 1];
            d[kk + 1] = cs * d[kk + 1] - sn * e[kk];

            if kk + 1 < hi {
                g = sn * e[kk + 1];
                e[kk + 1] *= cs;
            }
        }
        e[hi - 1] = f;
    }

    if !converged {
        return Err(LinAlgError::SvdNonConvergence);
    }

    for value in &mut d {
        if *value < 0.0 {
            *value = -*value;
        }
    }

    sort_singular_values_descending_in_place(&mut d);
    d.truncate(k);
    Ok(d)
}

/// Transpose an m×n matrix (row-major) to n×m.
fn transpose_mat(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut t = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            t[j * m + i] = a[i * n + j];
        }
    }
    t
}

/// Frobenius norm of an NxN matrix (flat row-major).
pub fn matrix_norm_frobenius(a: &[f64], n: usize) -> Result<f64, LinAlgError> {
    if Some(a.len()) != n.checked_mul(n) || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "matrix_norm: input must be n*n with n > 0",
        ));
    }

    let mut sum = 0.0;
    for &value in a {
        if !value.is_finite() {
            return Err(LinAlgError::NormDetRankPolicyViolation(
                "matrix entries must be finite for norm",
            ));
        }
        sum += value * value;
    }
    Ok(sum.sqrt())
}

const MATRIX_NORM_CACHE_LINEAR_COLUMN_SUM_MIN_ELEMS: usize = 4096;
const MATRIX_NORM_COLUMN_SUM_SIMD_MIN_COLS: usize = 256;
const MATRIX_NORM_COLUMN_SUM_SIMD_LANES: usize = 8;
const MATRIX_NORM_COLUMN_SUM_STACK_MIN_COLS: usize = 512;
const MATRIX_NORM_COLUMN_SUM_STACK_MAX_COLS: usize = 1024;

fn matrix_norm_spectral_precheck(a: &[f64]) -> Result<Option<f64>, LinAlgError> {
    if a.iter().any(|value| value.is_nan()) {
        return Err(LinAlgError::SvdNonConvergence);
    }
    if a.iter().any(|value| value.is_infinite()) {
        return Ok(Some(f64::NAN));
    }
    Ok(None)
}

fn matrix_norm_column_sum_strided(a: &[f64], m: usize, n: usize, use_min: bool) -> f64 {
    let mut selected = if use_min { f64::INFINITY } else { 0.0 };
    for j in 0..n {
        let mut col_sum = 0.0;
        for i in 0..m {
            let value = a[i * n + j];
            if value.is_nan() {
                return f64::NAN;
            }
            col_sum += value.abs();
        }
        selected = if use_min {
            selected.min(col_sum)
        } else {
            selected.max(col_sum)
        };
    }
    selected
}

fn matrix_norm_column_sum_cache_linear_select(col_sums: &[f64], use_min: bool) -> f64 {
    if use_min {
        col_sums.iter().copied().fold(f64::INFINITY, f64::min)
    } else {
        col_sums.iter().copied().fold(0.0_f64, f64::max)
    }
}

fn matrix_norm_column_sum_cache_linear_fill(
    a: &[f64],
    n: usize,
    use_min: bool,
    col_sums: &mut [f64],
) -> f64 {
    for row in a.chunks_exact(n) {
        for (sum, &value) in col_sums.iter_mut().zip(row) {
            if value.is_nan() {
                return f64::NAN;
            }
            *sum += value.abs();
        }
    }
    matrix_norm_column_sum_cache_linear_select(col_sums, use_min)
}

#[inline(never)]
fn matrix_norm_column_sum_cache_linear_fill_simd(
    a: &[f64],
    n: usize,
    use_min: bool,
    col_sums: &mut [f64],
) -> f64 {
    if n >= MATRIX_NORM_COLUMN_SUM_SIMD_MIN_COLS {
        use std::simd::Simd;
        use std::simd::num::SimdFloat;

        type Lane = Simd<f64, MATRIX_NORM_COLUMN_SUM_SIMD_LANES>;
        let simd_cols = n / MATRIX_NORM_COLUMN_SUM_SIMD_LANES * MATRIX_NORM_COLUMN_SUM_SIMD_LANES;
        let mut row_blocks = a.chunks_exact(n * 4);
        for rows in &mut row_blocks {
            let (row0, rows) = rows.split_at(n);
            let (row1, rows) = rows.split_at(n);
            let (row2, row3) = rows.split_at(n);
            let mut col = 0;
            while col < simd_cols {
                let sums =
                    Lane::from_slice(&col_sums[col..col + MATRIX_NORM_COLUMN_SUM_SIMD_LANES]);
                let values0 =
                    Lane::from_slice(&row0[col..col + MATRIX_NORM_COLUMN_SUM_SIMD_LANES]).abs();
                let values1 =
                    Lane::from_slice(&row1[col..col + MATRIX_NORM_COLUMN_SUM_SIMD_LANES]).abs();
                let values2 =
                    Lane::from_slice(&row2[col..col + MATRIX_NORM_COLUMN_SUM_SIMD_LANES]).abs();
                let values3 =
                    Lane::from_slice(&row3[col..col + MATRIX_NORM_COLUMN_SUM_SIMD_LANES]).abs();
                (((sums + values0) + values1) + values2 + values3)
                    .copy_to_slice(&mut col_sums[col..col + MATRIX_NORM_COLUMN_SUM_SIMD_LANES]);
                col += MATRIX_NORM_COLUMN_SUM_SIMD_LANES;
            }
            for col in simd_cols..n {
                let sum = &mut col_sums[col];
                let value0 = row0[col];
                let value1 = row1[col];
                let value2 = row2[col];
                let value3 = row3[col];
                if value0.is_nan() || value1.is_nan() || value2.is_nan() || value3.is_nan() {
                    return f64::NAN;
                }
                *sum += value0.abs();
                *sum += value1.abs();
                *sum += value2.abs();
                *sum += value3.abs();
            }
        }
        for row in row_blocks.remainder().chunks_exact(n) {
            let mut col = 0;
            while col < simd_cols {
                let sums =
                    Lane::from_slice(&col_sums[col..col + MATRIX_NORM_COLUMN_SUM_SIMD_LANES]);
                let values =
                    Lane::from_slice(&row[col..col + MATRIX_NORM_COLUMN_SUM_SIMD_LANES]).abs();
                (sums + values)
                    .copy_to_slice(&mut col_sums[col..col + MATRIX_NORM_COLUMN_SUM_SIMD_LANES]);
                col += MATRIX_NORM_COLUMN_SUM_SIMD_LANES;
            }
            for (sum, &value) in col_sums[simd_cols..n].iter_mut().zip(&row[simd_cols..]) {
                if value.is_nan() {
                    return f64::NAN;
                }
                *sum += value.abs();
            }
        }
        if col_sums[..n].iter().any(|value| value.is_nan()) {
            return f64::NAN;
        }
        return matrix_norm_column_sum_cache_linear_select(col_sums, use_min);
    }
    matrix_norm_column_sum_cache_linear_fill(a, n, use_min, col_sums)
}

fn matrix_norm_column_sum_cache_linear_scalar(a: &[f64], n: usize, use_min: bool) -> f64 {
    if (MATRIX_NORM_COLUMN_SUM_STACK_MIN_COLS..=MATRIX_NORM_COLUMN_SUM_STACK_MAX_COLS).contains(&n)
    {
        let mut col_sums = [0.0_f64; MATRIX_NORM_COLUMN_SUM_STACK_MAX_COLS];
        matrix_norm_column_sum_cache_linear_fill(a, n, use_min, &mut col_sums[..n])
    } else {
        let mut col_sums = vec![0.0_f64; n];
        matrix_norm_column_sum_cache_linear_fill(a, n, use_min, &mut col_sums)
    }
}

fn matrix_norm_column_sum_cache_linear_simd(a: &[f64], n: usize, use_min: bool) -> f64 {
    if (MATRIX_NORM_COLUMN_SUM_STACK_MIN_COLS..=MATRIX_NORM_COLUMN_SUM_STACK_MAX_COLS).contains(&n)
    {
        let mut col_sums = [0.0_f64; MATRIX_NORM_COLUMN_SUM_STACK_MAX_COLS];
        matrix_norm_column_sum_cache_linear_fill_simd(a, n, use_min, &mut col_sums[..n])
    } else {
        let mut col_sums = vec![0.0_f64; n];
        matrix_norm_column_sum_cache_linear_fill_simd(a, n, use_min, &mut col_sums)
    }
}

fn matrix_norm_column_sum(a: &[f64], m: usize, n: usize, use_min: bool) -> f64 {
    if a.len() >= MATRIX_NORM_CACHE_LINEAR_COLUMN_SUM_MIN_ELEMS {
        if n >= MATRIX_NORM_COLUMN_SUM_SIMD_MIN_COLS {
            matrix_norm_column_sum_cache_linear_simd(a, n, use_min)
        } else {
            matrix_norm_column_sum_cache_linear_scalar(a, n, use_min)
        }
    } else {
        matrix_norm_column_sum_strided(a, m, n, use_min)
    }
}

fn matrix_norm_row_sum(a: &[f64], n: usize, use_min: bool) -> f64 {
    let mut selected = if use_min { f64::INFINITY } else { 0.0 };
    for row in a.chunks_exact(n) {
        let mut row_sum = 0.0;
        for &value in row {
            if value.is_nan() {
                return f64::NAN;
            }
            row_sum += value.abs();
        }
        selected = if use_min {
            selected.min(row_sum)
        } else {
            selected.max(row_sum)
        };
    }
    selected
}

#[inline]
fn matrix_norm_frobenius_unchecked_at(a: &[f64], base: usize, len: usize) -> f64 {
    let mut sum = 0.0;
    for &value in &a[base..base + len] {
        sum += value * value;
    }
    sum.sqrt()
}

/// General NxN matrix norm (np.linalg.norm for matrices).
/// Supports: "fro" (Frobenius), "1" (max column sum), "inf" (max row sum),
/// "2" (spectral, i.e. largest singular value), "nuc" (nuclear/trace norm).
pub fn matrix_norm_nxn(a: &[f64], m: usize, n: usize, ord: &str) -> Result<f64, LinAlgError> {
    if Some(a.len()) != m.checked_mul(n) || m == 0 || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "matrix_norm_nxn: input must be m*n with m,n > 0",
        ));
    }
    match ord {
        "fro" => Ok(matrix_norm_frobenius_unchecked_at(a, 0, a.len())),
        "1" => {
            // Large row-major matrices make strided column walks cache-hostile;
            // scan rows once while preserving the per-column addition order.
            Ok(matrix_norm_column_sum(a, m, n, false))
        }
        "-1" => {
            Ok(matrix_norm_column_sum(a, m, n, true))
        }
        "inf" => {
            Ok(matrix_norm_row_sum(a, n, false))
        }
        "-inf" => {
            Ok(matrix_norm_row_sum(a, n, true))
        }
        "2" => {
            // Spectral norm = largest singular value (works on MxN)
            if let Some(value) = matrix_norm_spectral_precheck(a)? {
                return Ok(value);
            }
            let sigmas = svd_mxn(a, m, n)?;
            Ok(sigmas.first().copied().unwrap_or(0.0))
        }
        "-2" => {
            // Smallest singular value (works on MxN)
            if let Some(value) = matrix_norm_spectral_precheck(a)? {
                return Ok(value);
            }
            let sigmas = svd_mxn(a, m, n)?;
            Ok(sigmas.last().copied().unwrap_or(0.0))
        }
        "nuc" => {
            // Nuclear norm = sum of singular values (works on MxN)
            if let Some(value) = matrix_norm_spectral_precheck(a)? {
                return Ok(value);
            }
            let sigmas = svd_mxn(a, m, n)?;
            Ok(sigmas.iter().sum())
        }
        _ => Err(LinAlgError::NormDetRankPolicyViolation(
            "unknown norm order; use fro, 1, -1, inf, -inf, 2, -2, or nuc",
        )),
    }
}

#[inline]
fn trace_nxn_unchecked_at(a: &[f64], base: usize, n: usize) -> f64 {
    let stride = n + 1;
    let mut idx = base;
    let mut sum = 0.0;
    for _ in 0..n {
        sum += a[idx];
        idx += stride;
    }
    sum
}

/// Trace of an NxN flat matrix (sum of diagonal elements).
pub fn trace_nxn(a: &[f64], n: usize) -> Result<f64, LinAlgError> {
    if Some(a.len()) != n.checked_mul(n) {
        return Err(LinAlgError::ShapeContractViolation(
            "trace_nxn: input must be n*n",
        ));
    }
    Ok(trace_nxn_unchecked_at(a, 0, n))
}

fn validate_matrix_rank_tol(tol: Option<f64>) -> Result<(), LinAlgError> {
    if let Some(tol) = tol
        && (!tol.is_finite() || tol < 0.0)
    {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "tol must be finite and >= 0",
        ));
    }
    Ok(())
}

fn resolve_matrix_rank_threshold(sigma_max: f64, m: usize, n: usize, tol: Option<f64>) -> f64 {
    tol.unwrap_or_else(|| sigma_max * (m.max(n) as f64) * f64::EPSILON)
}

fn count_matrix_rank(sigmas: &[f64], threshold: f64) -> usize {
    sigmas.iter().filter(|&&sigma| sigma > threshold).count()
}

/// Matrix rank via SVD for MxN matrices with NumPy-compatible tolerance semantics.
///
/// When `tol` is `None`, the threshold matches NumPy's default:
/// `sigma_max * max(m, n) * eps`.
/// When `tol` is `Some`, it is treated as an absolute threshold.
pub fn matrix_rank_mxn_tol(
    a: &[f64],
    m: usize,
    n: usize,
    tol: Option<f64>,
) -> Result<usize, LinAlgError> {
    if Some(a.len()) != m.checked_mul(n) || m == 0 || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "matrix_rank_mxn: input must be m*n with m,n > 0",
        ));
    }
    validate_matrix_rank_tol(tol)?;
    let has_nan = a.iter().any(|value| value.is_nan());
    let has_inf = a.iter().any(|value| value.is_infinite());
    if has_nan {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "matrix entries must be finite for rank",
        ));
    }
    if has_inf {
        return Ok(0);
    }

    let sigmas = svd_mxn(a, m, n)?;
    let sigma_max = sigmas.first().copied().unwrap_or(0.0);
    if sigma_max == 0.0 {
        return Ok(0);
    }
    let threshold = resolve_matrix_rank_threshold(sigma_max, m, n, tol);
    Ok(count_matrix_rank(&sigmas, threshold))
}

pub fn matrix_rank_nxn_tol(a: &[f64], n: usize, tol: Option<f64>) -> Result<usize, LinAlgError> {
    matrix_rank_mxn_tol(a, n, n, tol)
}

/// Matrix rank via SVD for MxN matrices.
/// Returns the number of singular values above `rcond * sigma_max`.
pub fn matrix_rank_mxn(a: &[f64], m: usize, n: usize, rcond: f64) -> Result<usize, LinAlgError> {
    if Some(a.len()) != m.checked_mul(n) || m == 0 || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "matrix_rank_mxn: input must be m*n with m,n > 0",
        ));
    }
    if !rcond.is_finite() || rcond < 0.0 {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "rcond must be finite and >= 0",
        ));
    }
    let has_nan = a.iter().any(|value| value.is_nan());
    let has_inf = a.iter().any(|value| value.is_infinite());
    if has_nan {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "matrix entries must be finite for rank",
        ));
    }
    if has_inf {
        return Ok(0);
    }

    let sigmas = svd_mxn(a, m, n)?;
    let sigma_max = sigmas.first().copied().unwrap_or(0.0);
    if sigma_max == 0.0 {
        return Ok(0);
    }
    let threshold = sigma_max * rcond;
    Ok(count_matrix_rank(&sigmas, threshold))
}

pub fn matrix_rank_nxn(a: &[f64], n: usize, rcond: f64) -> Result<usize, LinAlgError> {
    matrix_rank_mxn(a, n, n, rcond)
}

/// Compute singular values of an NxN matrix via QR iteration on A^T A.
/// Returns singular values in descending order.
pub fn svd_nxn(a: &[f64], n: usize) -> Result<Vec<f64>, LinAlgError> {
    svd_mxn(a, n, n)
}

// ── Eigenvalue infrastructure ─────────────────────────────────────────

// Symmetric tridiagonalization switches from the unblocked (BLAS-2, rank-1
// reflector) reduction to the blocked (BLAS-3 / packed_gemm) `tridiag_reduce_blocked`
// at this size. The blocked kernel handles BOTH the values-only AND the
// Q-accumulating path and is faster for both once n >= ~64 (the old 384 threshold
// left eigvalsh/eigh on the memory-bound unblocked loop for 64<=n<384 — measured
// 1.3-1.9x slower than blocked, the dominant eigvalsh-vs-numpy gap there). Same
// factorization up to GEMM re-association (allclose parity; verified recon<1e-9 and
// allclose-vs-numpy for eigvalsh+eigh across n=64..383). Panel width below.
const TRIDIAG_BLOCK_MIN: usize = 64;
const TRIDIAG_PANEL_NB: usize = 64;
const SBR_STAGE1_BAND_WIDTH: usize = 96;
const SBR_STAGE1_PANEL_NB: usize = 128;
type SbrStage1Result = (Vec<f64>, Vec<f64>, Vec<f64>);
const TRIDIAG_SERIAL_ROWDOT_MIN_N: usize = 192;
const TRIDIAG_SERIAL_ROWDOT_MAX_N: usize = 384;
// Minimum trailing-block height for the parallel symmetric panel matvec (u = A·v).
// Each matvec is only O(h²) work split across rows, so the rayon dispatch +
// work-stealing setup (~tens of µs, worse under machine load) dominates until h is
// large; measured break-even is past n=1024. Gated high so only the big early
// matvecs of a large reduction parallelize — every matvec of an n<=1024 matrix
// stays serial (bit-identical perf, no regression), and the late short matvecs of
// a large matrix also stay serial.
const TRIDIAG_MATVEC_PAR_MIN: usize = 1024;

fn tridiag_symmetric_matvec_serial(
    work: &[f64],
    n: usize,
    start: usize,
    v: &[f64],
    u: &mut [f64],
) {
    debug_assert_eq!(work.len(), n * n);
    debug_assert_eq!(v.len(), n);
    debug_assert_eq!(u.len(), n);

    if (TRIDIAG_SERIAL_ROWDOT_MIN_N..TRIDIAG_SERIAL_ROWDOT_MAX_N).contains(&n) {
        let v_tail = &v[start..n];
        for i in start..n {
            let row = &work[i * n + start..i * n + n];
            let mut ui = 0.0;
            let mut offset = 0usize;
            while offset + 4 <= v_tail.len() {
                ui += row[offset] * v_tail[offset];
                ui += row[offset + 1] * v_tail[offset + 1];
                ui += row[offset + 2] * v_tail[offset + 2];
                ui += row[offset + 3] * v_tail[offset + 3];
                offset += 4;
            }
            while offset < v_tail.len() {
                ui += row[offset] * v_tail[offset];
                offset += 1;
            }
            u[i] = ui;
        }
        return;
    }

    for ui in &mut u[start..n] {
        *ui = 0.0;
    }
    for i in start..n {
        let vi = v[i];
        let row = &work[i * n..i * n + n];
        let mut ui = u[i] + row[i] * vi;
        let mut l = i + 1;
        while l + 4 <= n {
            let a = row[l];
            ui += a * v[l];
            u[l] += a * vi;
            let a = row[l + 1];
            ui += a * v[l + 1];
            u[l + 1] += a * vi;
            let a = row[l + 2];
            ui += a * v[l + 2];
            u[l + 2] += a * vi;
            let a = row[l + 3];
            ui += a * v[l + 3];
            u[l + 3] += a * vi;
            l += 4;
        }
        while l < n {
            let a = row[l];
            ui += a * v[l];
            u[l] += a * vi;
            l += 1;
        }
        u[i] = ui;
    }
}

// Blocked symmetric tridiagonalization, values only (dsytrd/dlatrd shape). For
// each width-nb panel, reduce nb columns producing reflectors V and vectors
// W (w = tau·A·v − (tau²·vᵀAv/2)·v), computed from symmetric matvecs against the
// panel-start trailing block plus rank-2 corrections from the prior panel
// reflectors; then update the trailing block ONCE with the symmetric rank-2k
// update A22 -= V·Wᵀ + W·Vᵀ via the packed GEMM. This roughly halves the
// trailing-matrix memory traffic of the unblocked per-column left+right sweep
// (which is DRAM-bound). Same reflectors as the unblocked path → tolerance
// equivalent (reassociated updates; never bit-exact). Returns (d, e).
fn tridiag_reduce_blocked(a: &[f64], n: usize, accumulate_q: bool) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut work = a.to_vec();
    let mut d = vec![0.0f64; n];
    let mut e = vec![0.0f64; n - 1];
    let mut q = if accumulate_q {
        let mut qq = vec![0.0f64; n * n];
        for i in 0..n {
            qq[i * n + i] = 1.0;
        }
        qq
    } else {
        Vec::new()
    };

    let mut jb = 0;
    while jb < n - 2 {
        let pend = (jb + TRIDIAG_PANEL_NB).min(n - 2);
        let nb = pend - jb;
        let h = n - jb;
        let mut vv = vec![0.0f64; h * nb]; // vv[(i-jb)*nb + t] = v_{jb+t}[i]
        let mut ww = vec![0.0f64; h * nb];
        let mut taus = vec![0.0f64; nb];
        let mut col = vec![0.0f64; n];
        let mut u = vec![0.0f64; n];
        let mut vcol = vec![0.0f64; n]; // contiguous copy of reflector v (panel col t)

        for t in 0..nb {
            let j = jb + t;
            let jr = j - jb;
            // Corrected column j (apply prior panel reflectors locally).
            for i in j..n {
                let ir = i - jb;
                let mut s = work[i * n + j];
                for q in 0..t {
                    s -= vv[ir * nb + q] * ww[jr * nb + q] + ww[ir * nb + q] * vv[jr * nb + q];
                }
                col[i] = s;
            }
            d[j] = col[j];
            let mut cns = 0.0;
            for &c in col.iter().take(n).skip(j + 1) {
                cns += c * c;
            }
            let col_norm = cns.sqrt();
            if col_norm < f64::EPSILON * col[j].abs().max(1.0) {
                e[j] = col[j + 1];
                continue; // null reflector (vv/ww column t stay 0)
            }
            let sign = if col[j + 1] >= 0.0 { 1.0 } else { -1.0 };
            for i in (j + 1)..n {
                vv[(i - jb) * nb + t] = col[i];
            }
            vv[(j + 1 - jb) * nb + t] += sign * col_norm;
            let mut vns = 0.0;
            for i in (j + 1)..n {
                let x = vv[(i - jb) * nb + t];
                vns += x * x;
            }
            if vns == 0.0 {
                e[j] = col[j + 1];
                continue;
            }
            let tau = 2.0 / vns;
            taus[t] = tau;
            e[j] = -sign * col_norm;
            // u = A_current·v (rows [j+1,n)): A_ps·v − V·(Wᵀv) − W·(Vᵀv).
            // The A_ps·v symmetric matvec is the O(h²)-per-column dominant cost of
            // the panel reduction (dlatrd). Each u[i] is an independent dot of work
            // row i with v, so it parallelizes across rows; gather v contiguously
            // first so the inner dot is a contiguous row·vector. Identical
            // ascending-l summation order per row → bit-exact vs the serial scan.
            for (idx, vc) in vcol[(j + 1)..n].iter_mut().enumerate() {
                *vc = vv[(j + 1 + idx - jb) * nb + t];
            }
            if n - (j + 1) >= TRIDIAG_MATVEC_PAR_MIN {
                let (work_ref, vcol_ref) = (&work, &vcol);
                u[(j + 1)..n]
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(idx, ui)| {
                        let i = j + 1 + idx;
                        let row = &work_ref[i * n + (j + 1)..i * n + n];
                        let vc = &vcol_ref[(j + 1)..n];
                        let mut s = 0.0;
                        for (&w, &vl) in row.iter().zip(vc.iter()) {
                            s += w * vl;
                        }
                        *ui = s;
                    });
            } else {
                tridiag_symmetric_matvec_serial(&work, n, j + 1, &vcol, &mut u);
            }
            for q in 0..t {
                let (mut wtv, mut vtv) = (0.0, 0.0);
                for l in (j + 1)..n {
                    let lr = l - jb;
                    let vl = vv[lr * nb + t];
                    wtv += ww[lr * nb + q] * vl;
                    vtv += vv[lr * nb + q] * vl;
                }
                for (i, ui) in u.iter_mut().enumerate().take(n).skip(j + 1) {
                    let ir = i - jb;
                    *ui -= vv[ir * nb + q] * wtv + ww[ir * nb + q] * vtv;
                }
            }
            for ui in u.iter_mut().take(n).skip(j + 1) {
                *ui *= tau;
            }
            let mut vu = 0.0;
            for i in (j + 1)..n {
                vu += vv[(i - jb) * nb + t] * u[i];
            }
            let alpha = tau * vu / 2.0;
            for i in (j + 1)..n {
                ww[(i - jb) * nb + t] = u[i] - alpha * vv[(i - jb) * nb + t];
            }
        }

        // Symmetric rank-2k trailing update A22 -= V·Wᵀ + W·Vᵀ.
        // The panel width is 64, below packed_gemm's k-parallel cutoff, so the
        // old two-GEMM materialization ran the dominant update serially and then
        // streamed two trail×trail temporaries back into A22. The fused helper
        // keeps the same k-ascending rank-2k formula while parallelizing across
        // independent output rows and avoiding both materialized products.
        let trail = n - pend;
        if trail > 0 {
            let off = pend - jb;
            let mut vtr = vec![0.0f64; trail * nb];
            let mut wtr = vec![0.0f64; trail * nb];
            for i in 0..trail {
                for t in 0..nb {
                    vtr[i * nb + t] = vv[(off + i) * nb + t];
                    wtr[i * nb + t] = ww[(off + i) * nb + t];
                }
            }
            sbr_apply_symmetric_rank2k_update(&mut work, n, pend, trail, nb, &vtr, &wtr);
        }

        // Q := Q·(H_jb·…·H_{pend-1}) = Q·(I − V·T·Vᵀ) via the compact-WY block
        // reflector applied with GEMMs (same technique as the blocked QR Q build).
        if accumulate_q {
            // T (nb×nb upper triangular), forward direction.
            let mut tm = vec![0.0f64; nb * nb];
            for c in 0..nb {
                tm[c * nb + c] = taus[c];
                if taus[c] == 0.0 {
                    continue;
                }
                let mut col = vec![0.0f64; c];
                for (i, ci) in col.iter_mut().enumerate() {
                    let mut dot = 0.0;
                    for row in 0..h {
                        dot += vv[row * nb + i] * vv[row * nb + c];
                    }
                    *ci = -taus[c] * dot;
                }
                for i in 0..c {
                    let mut s = 0.0;
                    for l in i..c {
                        s += tm[i * nb + l] * col[l];
                    }
                    tm[i * nb + c] = s;
                }
            }
            let mut vt = vec![0.0f64; nb * h];
            for row in 0..h {
                for c in 0..nb {
                    vt[c * h + row] = vv[row * nb + c];
                }
            }
            // Q_active = Q[:, jb..n] (n×h).
            let mut qa = vec![0.0f64; n * h];
            for i in 0..n {
                let src = i * n + jb;
                qa[i * h..i * h + h].copy_from_slice(&q[src..src + h]);
            }
            let qv = packed_gemm(&qa, &vv, n, h, nb); // Q_active·V
            let qvt = packed_gemm(&qv, &tm, n, nb, nb); // ·T
            let upd = packed_gemm(&qvt, &vt, n, nb, h); // ·Vᵀ
            for i in 0..n {
                let dst = i * n + jb;
                for (cell, &x) in q[dst..dst + h].iter_mut().zip(&upd[i * h..i * h + h]) {
                    *cell -= x;
                }
            }
        }

        jb = pend;
    }

    // Corner (last 2×2 block, fully updated by the final rank-2k update).
    d[n - 2] = work[(n - 2) * n + (n - 2)];
    d[n - 1] = work[(n - 1) * n + (n - 1)];
    e[n - 2] = work[(n - 2) * n + (n - 1)];
    (d, e, q)
}

fn pack_lower_band(work: &[f64], n: usize, band_width: usize) -> Vec<f64> {
    let mut band = vec![0.0f64; n * (band_width + 1)];
    for col in 0..n {
        let rows = (n - col).min(band_width + 1);
        for delta in 0..rows {
            band[col * (band_width + 1) + delta] = work[(col + delta) * n + col];
        }
    }
    band
}

#[cfg(test)]
fn unpack_lower_band(band: &[f64], n: usize, band_width: usize) -> Vec<f64> {
    let mut work = vec![0.0f64; n * n];
    for col in 0..n {
        let rows = (n - col).min(band_width + 1);
        for delta in 0..rows {
            let row = col + delta;
            let value = band[col * (band_width + 1) + delta];
            work[row * n + col] = value;
            work[col * n + row] = value;
        }
    }
    work
}

fn sbr_active_times_v(
    work: &[f64],
    n: usize,
    active: usize,
    h: usize,
    nb: usize,
    vv: &[f64],
) -> Vec<f64> {
    let mut out = vec![0.0f64; h * nb];
    let parallel =
        h >= MATMUL_PARALLEL_MIN_DIM && nb >= PACKED_NR && rayon::current_num_threads() >= 2;
    if parallel {
        out.par_chunks_mut(nb).enumerate().for_each(|(row, dst)| {
            let src = &work[(active + row) * n + active..(active + row) * n + active + h];
            for k in 0..h {
                let aik = src[k];
                let vrow = &vv[k * nb..k * nb + nb];
                for (cell, &v) in dst.iter_mut().zip(vrow) {
                    *cell += aik * v;
                }
            }
        });
    } else {
        for row in 0..h {
            let src = &work[(active + row) * n + active..(active + row) * n + active + h];
            let dst = &mut out[row * nb..row * nb + nb];
            for k in 0..h {
                let aik = src[k];
                let vrow = &vv[k * nb..k * nb + nb];
                for (cell, &v) in dst.iter_mut().zip(vrow) {
                    *cell += aik * v;
                }
            }
        }
    }
    out
}

fn sbr_apply_cross_wy_update(
    work: &mut [f64],
    n: usize,
    active: usize,
    h: usize,
    nb: usize,
    cvt: &[f64],
    vt: &[f64],
) {
    if active == 0 || h == 0 {
        return;
    }
    debug_assert_eq!(work.len(), n * n);
    debug_assert_eq!(cvt.len(), active * nb);
    debug_assert_eq!(vt.len(), nb * h);

    packed_gemm_sub_assign_strided(cvt, vt, active, nb, h, n, &mut work[active..]);
    for row in 0..active {
        let src = row * n + active;
        for col in 0..h {
            work[(active + col) * n + row] = work[src + col];
        }
    }
}

fn sbr_apply_symmetric_rank2k_update(
    work: &mut [f64],
    n: usize,
    active: usize,
    h: usize,
    nb: usize,
    vv: &[f64],
    w: &[f64],
) {
    const COL_TILE: usize = 32;
    let parallel = h >= MATMUL_PARALLEL_MIN_DIM && rayon::current_num_threads() >= 2;
    let update_row = |row_idx: usize, row: &mut [f64]| {
        let vi = &vv[row_idx * nb..row_idx * nb + nb];
        let wi = &w[row_idx * nb..row_idx * nb + nb];
        let mut j0 = row_idx;
        while j0 < h {
            let j_end = (j0 + COL_TILE).min(h);
            for j in j0..j_end {
                let vj = &vv[j * nb..j * nb + nb];
                let wj = &w[j * nb..j * nb + nb];
                let mut delta = 0.0f64;
                for k in 0..nb {
                    delta += vi[k] * wj[k] + wi[k] * vj[k];
                }
                row[active + j] -= delta;
            }
            j0 = j_end;
        }
    };

    let active_rows = &mut work[active * n..];
    if parallel {
        active_rows
            .par_chunks_mut(n)
            .take(h)
            .enumerate()
            .for_each(|(row_idx, row)| update_row(row_idx, row));
    } else {
        for (row_idx, row) in active_rows.chunks_mut(n).take(h).enumerate() {
            update_row(row_idx, row);
        }
    }

    for row_idx in 0..h {
        let src = (active + row_idx) * n + active;
        for col_idx in (row_idx + 1)..h {
            let value = work[src + col_idx];
            work[(active + col_idx) * n + active + row_idx] = value;
        }
    }
}

fn sbr_stage1_dense_to_band_impl(
    a: &[f64],
    n: usize,
    accumulate_q: bool,
) -> Result<SbrStage1Result, LinAlgError> {
    if Some(a.len()) != n.checked_mul(n) || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "sbr_stage1_dense_to_band_lower_nxn: input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "matrix entries must be finite for SBR stage-1",
        ));
    }

    let band_width = SBR_STAGE1_BAND_WIDTH.min(n - 1);
    let mut work = a.to_vec();
    let mut q = if accumulate_q {
        let mut qq = vec![0.0f64; n * n];
        for i in 0..n {
            qq[i * n + i] = 1.0;
        }
        qq
    } else {
        Vec::new()
    };

    let mut jb = 0usize;
    while jb + band_width < n {
        let active = jb + band_width;
        let h = n - active;
        if h == 0 {
            break;
        }
        let nb = SBR_STAGE1_PANEL_NB.min(n - band_width - jb).min(h);
        if nb == 0 {
            break;
        }

        let mut panel = vec![0.0f64; h * nb];
        for row in 0..h {
            let src = (active + row) * n + jb;
            panel[row * nb..row * nb + nb].copy_from_slice(&work[src..src + nb]);
        }

        let mut vv = vec![0.0f64; h * nb];
        let mut taus = vec![0.0f64; nb];
        let mut dpanel = vec![0.0f64; nb];
        for t in 0..nb {
            let mut col_norm_sq = 0.0;
            for row in t..h {
                let x = panel[row * nb + t];
                col_norm_sq += x * x;
            }
            let col_norm = col_norm_sq.sqrt();
            if col_norm < f64::EPSILON {
                continue;
            }
            let sign = if panel[t * nb + t] >= 0.0 { 1.0 } else { -1.0 };
            for row in t..h {
                vv[row * nb + t] = panel[row * nb + t];
            }
            vv[t * nb + t] += sign * col_norm;
            let mut v_norm_sq = 0.0;
            for row in t..h {
                let x = vv[row * nb + t];
                v_norm_sq += x * x;
            }
            if v_norm_sq == 0.0 {
                continue;
            }
            let tau = 2.0 / v_norm_sq;
            taus[t] = tau;

            let width = nb - t;
            for dj in &mut dpanel[..width] {
                *dj = 0.0;
            }
            for row in t..h {
                let vi = vv[row * nb + t];
                let src = row * nb + t;
                for (dj, &cell) in dpanel[..width].iter_mut().zip(panel[src..src + width].iter()) {
                    *dj += vi * cell;
                }
            }
            for dj in &mut dpanel[..width] {
                *dj *= tau;
            }
            for row in t..h {
                let vi = vv[row * nb + t];
                let dst = row * nb + t;
                for (cell, &dj) in panel[dst..dst + width].iter_mut().zip(dpanel[..width].iter()) {
                    *cell -= dj * vi;
                }
            }
        }

        let mut tm = vec![0.0f64; nb * nb];
        for t in 0..nb {
            tm[t * nb + t] = taus[t];
            if taus[t] == 0.0 {
                continue;
            }
            let mut col = vec![0.0f64; t];
            for (i, ci) in col.iter_mut().enumerate() {
                let mut dot = 0.0;
                for row in 0..h {
                    dot += vv[row * nb + i] * vv[row * nb + t];
                }
                *ci = -taus[t] * dot;
            }
            for i in 0..t {
                let mut sum = 0.0;
                for l in i..t {
                    sum += tm[i * nb + l] * col[l];
                }
                tm[i * nb + t] = sum;
            }
        }

        let mut tmt = vec![0.0f64; nb * nb];
        for row in 0..nb {
            for col in 0..nb {
                tmt[row * nb + col] = tm[col * nb + row];
            }
        }
        let mut vt = vec![0.0f64; nb * h];
        for row in 0..h {
            for col in 0..nb {
                vt[col * h + row] = vv[row * nb + col];
            }
        }

        if active > 0 {
            let mut cross = vec![0.0f64; active * h];
            for row in 0..active {
                let src = row * n + active;
                cross[row * h..row * h + h].copy_from_slice(&work[src..src + h]);
            }
            let cv = packed_gemm(&cross, &vv, active, h, nb);
            let cvt = packed_gemm(&cv, &tm, active, nb, nb);
            sbr_apply_cross_wy_update(&mut work, n, active, h, nb, &cvt, &vt);
        }

        let av = sbr_active_times_v(&work, n, active, h, nb, &vv);
        let mut w = packed_gemm(&av, &tm, h, nb, nb);
        let vtw = packed_gemm(&vt, &w, nb, h, nb);
        let corr = packed_gemm(&tmt, &vtw, nb, nb, nb);
        let vcorr = packed_gemm(&vv, &corr, h, nb, nb);
        for (wi, &ci) in w.iter_mut().zip(&vcorr) {
            *wi -= 0.5 * ci;
        }
        sbr_apply_symmetric_rank2k_update(&mut work, n, active, h, nb, &vv, &w);

        if accumulate_q {
            let mut qa = vec![0.0f64; n * h];
            for row in 0..n {
                let src = row * n + active;
                qa[row * h..row * h + h].copy_from_slice(&q[src..src + h]);
            }
            let qv = packed_gemm(&qa, &vv, n, h, nb);
            let qvt = packed_gemm(&qv, &tm, n, nb, nb);
            let upd = packed_gemm(&qvt, &vt, n, nb, h);
            for row in 0..n {
                let dst = row * n + active;
                for (cell, &x) in q[dst..dst + h].iter_mut().zip(&upd[row * h..row * h + h]) {
                    *cell -= x;
                }
            }
        }

        jb += nb;
    }

    for row in 0..n {
        for col in 0..row {
            if row - col > band_width {
                work[row * n + col] = 0.0;
                work[col * n + row] = 0.0;
            } else {
                let value = 0.5 * (work[row * n + col] + work[col * n + row]);
                work[row * n + col] = value;
                work[col * n + row] = value;
            }
        }
    }

    let band = pack_lower_band(&work, n, band_width);
    Ok((work, band, q))
}

#[doc(hidden)]
pub fn sbr_stage1_dense_to_band_lower_nxn(a: &[f64], n: usize) -> Result<Vec<f64>, LinAlgError> {
    let (_work, band, _q) = sbr_stage1_dense_to_band_impl(a, n, false)?;
    Ok(band)
}

#[doc(hidden)]
pub fn sbr_stage1_band_width() -> usize {
    SBR_STAGE1_BAND_WIDTH
}

fn tridiag_reduce_impl(a: &[f64], n: usize, accumulate_q: bool) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if n >= TRIDIAG_BLOCK_MIN {
        return tridiag_reduce_blocked(a, n, accumulate_q);
    }
    let mut work = a.to_vec();
    let mut q = if accumulate_q {
        let mut q = vec![0.0; n * n];
        for i in 0..n {
            q[i * n + i] = 1.0;
        }
        q
    } else {
        Vec::new()
    };

    let mut v = vec![0.0; n];
    // Scratch for the cache-friendly left Householder transform (see below).
    let mut d = vec![0.0; n];
    let mut f_vec = vec![0.0; n];
    for j in 0..n.saturating_sub(2) {
        // Householder to zero column j below row j+1
        let col_norm = {
            let mut s = 0.0;
            for i in (j + 1)..n {
                s += work[i * n + j] * work[i * n + j];
            }
            s.sqrt()
        };
        if col_norm < f64::EPSILON * work[j * n + j].abs().max(1.0) {
            continue;
        }

        let sign = if work[(j + 1) * n + j] >= 0.0 {
            1.0
        } else {
            -1.0
        };
        for vi in &mut v[..=j] {
            *vi = 0.0;
        }
        for (idx, vi) in v[(j + 1)..n].iter_mut().enumerate() {
            *vi = work[(j + 1 + idx) * n + j];
        }
        v[j + 1] += sign * col_norm;

        let v_norm_sq: f64 = v[(j + 1)..].iter().map(|x| x * x).sum();
        if v_norm_sq == 0.0 {
            continue;
        }
        let scale = 2.0 / v_norm_sq;

        // Similarity: work = H * work * H  where H = I - scale·v·v^T
        // Left: work = H * work. Done as two row-contiguous passes instead of the
        // naive per-column walk (which strode `work[i*n+col]` down each column
        // with stride n — a cache line per step). Bit-exact: pass 1 sums each
        // d[col] = Σ_i v[i]·work[i][col] in the same i-ascending order, and pass 2
        // applies the identical `(scale·d[col])·v[i]` product grouping per element.
        for dc in d.iter_mut() {
            *dc = 0.0;
        }
        for i in (j + 1)..n {
            let vi = v[i];
            let row = &work[i * n..i * n + n];
            for (dc, &w) in d.iter_mut().zip(row.iter()) {
                *dc += vi * w;
            }
        }
        for (fc, &dc) in f_vec.iter_mut().zip(d.iter()) {
            *fc = scale * dc;
        }
        for i in (j + 1)..n {
            let vi = v[i];
            let row = &mut work[i * n..i * n + n];
            for (w, &fc) in row.iter_mut().zip(f_vec.iter()) {
                *w -= fc * vi;
            }
        }
        // Right: work = work * H
        for row in 0..n {
            let mut dot = 0.0;
            for i in (j + 1)..n {
                dot += v[i] * work[row * n + i];
            }
            let f = scale * dot;
            for i in (j + 1)..n {
                work[row * n + i] -= f * v[i];
            }
        }
        // Accumulate: Q = Q * H
        if accumulate_q {
            for row in 0..n {
                let mut dot = 0.0;
                for i in (j + 1)..n {
                    dot += v[i] * q[row * n + i];
                }
                let f = scale * dot;
                for i in (j + 1)..n {
                    q[row * n + i] -= f * v[i];
                }
            }
        }
    }

    let d: Vec<f64> = (0..n).map(|i| work[i * n + i]).collect();
    let e: Vec<f64> = (0..n.saturating_sub(1))
        .map(|i| work[i * n + (i + 1)])
        .collect();
    (d, e, q)
}

/// Reduce symmetric n×n matrix to tridiagonal form via Householder similarity
/// transformations.  Returns `(diagonal, off_diagonal, Q)` where `A = Q T Q^T`,
/// `T` has diagonal `d` and off-diagonal `e`, and `Q` is orthogonal (n×n row-major).
fn tridiag_reduce(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    tridiag_reduce_impl(a, n, true)
}

fn tridiag_reduce_values(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let (d, e, _) = tridiag_reduce_impl(a, n, false);
    (d, e)
}

/// Implicit QR iteration on symmetric tridiagonal `(d, e)` to find eigenvalues.
/// Modifies `d` and `e` in-place; eigenvalues end up on `d`.  If `q` is `Some`,
/// accumulates the rotation product into the n×n matrix (for eigenvectors).
fn tridiag_eig_qr(d: &mut [f64], e: &mut [f64], q: Option<&mut [f64]>, n: usize) {
    let eps = f64::EPSILON;
    let max_iter = EIGEN_QR_ITERATION_COEFF * n * n;

    // Each implicit-QR Givens rotation right-multiplies Q (Q := Q·G), updating two
    // COLUMNS — a stride-n walk down the rows that thrashes cache (the dominant
    // cost of eigh-with-eigenvectors: ~83 MFLOP/s at n=1024). Accumulate into Q
    // TRANSPOSED instead, so each rotation updates two contiguous ROWS; transpose
    // back at the end. Identical arithmetic per element (c·t1 + s·t2) — bit-exact.
    let mut qt: Option<Vec<f64>> = q.as_deref().map(|qq| {
        let mut t = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                t[i * n + j] = qq[j * n + i];
            }
        }
        t
    });

    for _iter in 0..max_iter {
        // Deflation: set small off-diagonals to zero
        for i in 0..e.len() {
            if e[i].abs() <= eps * (d[i].abs() + d[i + 1].abs()) {
                e[i] = 0.0;
            }
        }

        // Find largest unreduced block [lo..=hi]
        let mut hi = n - 1;
        while hi > 0 && e[hi - 1] == 0.0 {
            hi -= 1;
        }
        if hi == 0 {
            break; // Fully converged
        }
        let mut lo = hi - 1;
        while lo > 0 && e[lo - 1] != 0.0 {
            lo -= 1;
        }

        // Wilkinson shift from trailing 2×2  [[d[hi-1], e[hi-1]], [e[hi-1], d[hi]]]
        let delta = (d[hi - 1] - d[hi]) / 2.0;
        let shift = if delta == 0.0 {
            d[hi] - e[hi - 1].abs()
        } else {
            let sign = if delta >= 0.0 { 1.0 } else { -1.0 };
            d[hi]
                - e[hi - 1] * e[hi - 1]
                    / (delta + sign * (delta * delta + e[hi - 1] * e[hi - 1]).sqrt())
        };

        // Implicit QR step via Givens similarity chase on symmetric tridiagonal.
        // Each Givens G_k on (k, k+1) is applied as G^T * T * G (similarity),
        // which preserves eigenvalues and the trace.
        let mut x = d[lo] - shift;
        let mut z = e[lo];
        let mut bulge = 0.0;

        for kk in lo..hi {
            // Givens to zero z in [x, z]
            let r = x.hypot(z);
            let (c, s) = if r > 0.0 { (x / r, z / r) } else { (1.0, 0.0) };

            if kk > lo {
                e[kk - 1] = r; // was the previous off-diagonal / bulge resultant
            }

            // Similarity on 2×2 block (kk, kk+1): T := G^T T G
            let dk = d[kk];
            let dk1 = d[kk + 1];
            let ek = e[kk];

            d[kk] = c * c * dk + 2.0 * c * s * ek + s * s * dk1;
            d[kk + 1] = s * s * dk - 2.0 * c * s * ek + c * c * dk1;
            e[kk] = c * s * (dk1 - dk) + (c * c - s * s) * ek;

            if kk + 1 < hi {
                bulge = s * e[kk + 1];
                e[kk + 1] *= c;
            }

            // Next iteration: chase the bulge
            x = e[kk];
            z = bulge;

            // Accumulate rotation into Q^T (rows kk, kk+1 — contiguous).
            if let Some(ref mut qt) = qt {
                let (lo_row, hi_row) = qt.split_at_mut((kk + 1) * n);
                let row_kk = &mut lo_row[kk * n..kk * n + n];
                let row_kk1 = &mut hi_row[0..n];
                for col in 0..n {
                    let t1 = row_kk[col];
                    let t2 = row_kk1[col];
                    row_kk[col] = c * t1 + s * t2;
                    row_kk1[col] = -s * t1 + c * t2;
                }
            }
        }
    }

    // Transpose the accumulated Q^T back into the caller's Q.
    if let (Some(qq), Some(t)) = (q, qt.as_ref()) {
        for i in 0..n {
            for j in 0..n {
                qq[i * n + j] = t[j * n + i];
            }
        }
    }
}

// Robust scaled Euclidean length of a 2-vector, without libm `hypot`'s full
// correctly-rounded special-case machinery. Used in the eigenvalues-ONLY symmetric
// tridiagonal QR chase, where `hypot` is called O(n^2) times and dominates runtime.
// Overflow-safe (only the larger magnitude is squared after scaling) and accurate to
// ~1 ulp, which is well within the eigvalsh allclose contract.
#[inline(always)]
fn scaled_hypot(x: f64, z: f64) -> f64 {
    let ax = x.abs();
    let az = z.abs();
    let (hi, lo) = if ax >= az { (ax, az) } else { (az, ax) };
    if hi == 0.0 {
        0.0
    } else {
        let t = lo / hi;
        hi * (1.0 + t * t).sqrt()
    }
}

/// Eigenvalues-ONLY implicit-QR iteration on a symmetric tridiagonal `(d, e)`.
/// Identical Wilkinson-shift bulge chase as [`tridiag_eig_qr`] with `q = None`, but
/// the per-rotation length uses [`scaled_hypot`] instead of libm `hypot` and there is
/// no eigenvector-accumulation branch in the hot loop. Eigenvalues end up on `d`.
/// Result matches the eigenvector path to allclose (residual-checked by tests).
fn tridiag_eigvals_qr(d: &mut [f64], e: &mut [f64], n: usize) {
    if n <= 1 {
        return;
    }
    let eps = f64::EPSILON;
    let max_iter = EIGEN_QR_ITERATION_COEFF * n * n;

    for _iter in 0..max_iter {
        // Deflation: set small off-diagonals to zero.
        for i in 0..e.len() {
            if e[i].abs() <= eps * (d[i].abs() + d[i + 1].abs()) {
                e[i] = 0.0;
            }
        }

        // Largest unreduced trailing block [lo..=hi].
        let mut hi = n - 1;
        while hi > 0 && e[hi - 1] == 0.0 {
            hi -= 1;
        }
        if hi == 0 {
            break;
        }
        let mut lo = hi - 1;
        while lo > 0 && e[lo - 1] != 0.0 {
            lo -= 1;
        }

        // Wilkinson shift from the trailing 2×2.
        let delta = (d[hi - 1] - d[hi]) / 2.0;
        let shift = if delta == 0.0 {
            d[hi] - e[hi - 1].abs()
        } else {
            let sign = if delta >= 0.0 { 1.0 } else { -1.0 };
            d[hi]
                - e[hi - 1] * e[hi - 1]
                    / (delta + sign * (delta * delta + e[hi - 1] * e[hi - 1]).sqrt())
        };

        let mut x = d[lo] - shift;
        let mut z = e[lo];
        let mut bulge = 0.0;

        for kk in lo..hi {
            let r = scaled_hypot(x, z);
            let (c, s) = if r > 0.0 { (x / r, z / r) } else { (1.0, 0.0) };

            if kk > lo {
                e[kk - 1] = r;
            }

            let dk = d[kk];
            let dk1 = d[kk + 1];
            let ek = e[kk];

            d[kk] = c * c * dk + 2.0 * c * s * ek + s * s * dk1;
            d[kk + 1] = s * s * dk - 2.0 * c * s * ek + c * c * dk1;
            e[kk] = c * s * (dk1 - dk) + (c * c - s * s) * ek;

            if kk + 1 < hi {
                bulge = s * e[kk + 1];
                e[kk + 1] *= c;
            }

            x = e[kk];
            z = bulge;
        }
    }
}

/// Reduce general n×n matrix to upper Hessenberg form via Householder similarity.
/// Returns `(H, Q)` where `A = Q H Q^T`, `H` is upper Hessenberg (n×n row-major).
fn hessenberg_reduce(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut h = a.to_vec();
    let mut q = vec![0.0; n * n];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }

    let mut v = vec![0.0; n];
    // Scratch for the cache-friendly two-pass left Householder transform.
    let mut dbuf = vec![0.0; n];
    let mut f_vec = vec![0.0; n];
    for j in 0..n.saturating_sub(2) {
        // Householder to zero column j below row j+1 (entries j+2..n)
        let col_norm = {
            let mut s = 0.0;
            for i in (j + 1)..n {
                s += h[i * n + j] * h[i * n + j];
            }
            s.sqrt()
        };
        if col_norm < f64::EPSILON {
            continue;
        }

        let sign = if h[(j + 1) * n + j] >= 0.0 { 1.0 } else { -1.0 };
        for vi in &mut v[..=j] {
            *vi = 0.0;
        }
        for (i, vi) in v[(j + 1)..n].iter_mut().enumerate() {
            *vi = h[(i + j + 1) * n + j];
        }
        v[j + 1] += sign * col_norm;

        let v_norm_sq: f64 = v[(j + 1)..].iter().map(|x| x * x).sum();
        if v_norm_sq == 0.0 {
            continue;
        }
        let scale = 2.0 / v_norm_sq;

        // Left: H = P * H. Two row-contiguous passes instead of the naive
        // per-column walk (which strode `h[i*n+col]` down each column with stride
        // n — a cache line per step). Bit-exact: pass 1 sums each
        // d[col] = Σ_i v[i]·h[i][col] in the same i-ascending order, pass 2 applies
        // the identical (scale·d[col])·v[i] product grouping per element.
        for dc in dbuf.iter_mut() {
            *dc = 0.0;
        }
        for i in (j + 1)..n {
            let vi = v[i];
            let row = &h[i * n..i * n + n];
            for (dc, &hv) in dbuf.iter_mut().zip(row.iter()) {
                *dc += vi * hv;
            }
        }
        for (fc, &dc) in f_vec.iter_mut().zip(dbuf.iter()) {
            *fc = scale * dc;
        }
        for i in (j + 1)..n {
            let vi = v[i];
            let row = &mut h[i * n..i * n + n];
            for (hv, &fc) in row.iter_mut().zip(f_vec.iter()) {
                *hv -= fc * vi;
            }
        }
        // Right: H = H * P
        for row in h.chunks_mut(n) {
            let row_tail = &mut row[(j + 1)..n];
            let v_tail = &v[(j + 1)..n];
            let mut dot = 0.0;
            for (&vi, &hi) in v_tail.iter().zip(row_tail.iter()) {
                dot += vi * hi;
            }
            let f = scale * dot;
            for (hi, &vi) in row_tail.iter_mut().zip(v_tail.iter()) {
                *hi -= f * vi;
            }
        }
        // Q = Q * P
        for row in q.chunks_mut(n) {
            let row_tail = &mut row[(j + 1)..n];
            let v_tail = &v[(j + 1)..n];
            let mut dot = 0.0;
            for (&vi, &qi) in v_tail.iter().zip(row_tail.iter()) {
                dot += vi * qi;
            }
            let f = scale * dot;
            for (qi, &vi) in row_tail.iter_mut().zip(v_tail.iter()) {
                *qi -= f * vi;
            }
        }
    }

    (h, q)
}

/// Reduce a general n x n matrix to upper Hessenberg form when Schur vectors
/// are not needed. The H updates intentionally match `hessenberg_reduce`; only
/// the independent Q accumulation is skipped.
fn hessenberg_reduce_values(a: &[f64], n: usize) -> Vec<f64> {
    let mut h = a.to_vec();

    let mut v = vec![0.0; n];
    // Scratch for the cache-friendly two-pass left Householder transform.
    let mut dbuf = vec![0.0; n];
    let mut f_vec = vec![0.0; n];
    for j in 0..n.saturating_sub(2) {
        // Householder to zero column j below row j+1 (entries j+2..n)
        let col_norm = {
            let mut s = 0.0;
            for i in (j + 1)..n {
                s += h[i * n + j] * h[i * n + j];
            }
            s.sqrt()
        };
        if col_norm < f64::EPSILON {
            continue;
        }

        let sign = if h[(j + 1) * n + j] >= 0.0 { 1.0 } else { -1.0 };
        for vi in &mut v[..=j] {
            *vi = 0.0;
        }
        for (i, vi) in v[(j + 1)..n].iter_mut().enumerate() {
            *vi = h[(i + j + 1) * n + j];
        }
        v[j + 1] += sign * col_norm;

        let v_norm_sq: f64 = v[(j + 1)..].iter().map(|x| x * x).sum();
        if v_norm_sq == 0.0 {
            continue;
        }
        let scale = 2.0 / v_norm_sq;

        for dc in dbuf.iter_mut() {
            *dc = 0.0;
        }
        for i in (j + 1)..n {
            let vi = v[i];
            let row = &h[i * n..i * n + n];
            for (dc, &hv) in dbuf.iter_mut().zip(row.iter()) {
                *dc += vi * hv;
            }
        }
        for (fc, &dc) in f_vec.iter_mut().zip(dbuf.iter()) {
            *fc = scale * dc;
        }
        for i in (j + 1)..n {
            let vi = v[i];
            let row = &mut h[i * n..i * n + n];
            for (hv, &fc) in row.iter_mut().zip(f_vec.iter()) {
                *hv -= fc * vi;
            }
        }
        // Right: H = H * P
        for row in h.chunks_mut(n) {
            let row_tail = &mut row[(j + 1)..n];
            let v_tail = &v[(j + 1)..n];
            let mut dot = 0.0;
            for (&vi, &hi) in v_tail.iter().zip(row_tail.iter()) {
                dot += vi * hi;
            }
            let f = scale * dot;
            for (hi, &vi) in row_tail.iter_mut().zip(v_tail.iter()) {
                *hi -= f * vi;
            }
        }
    }

    h
}

/// Explicit single-shift QR iteration on upper Hessenberg form with deflation.
/// Converges to quasi-upper-triangular (real Schur) form.
/// If `z` is `Some`, accumulates the Schur vectors.
fn hessenberg_qr_iter(h: &mut [f64], mut z: Option<&mut [f64]>, n: usize) {
    let eps = f64::EPSILON;
    let max_iter = EIGEN_QR_ITERATION_COEFF * n * n;
    let mut p = n; // active upper bound (exclusive of converged tail)
    // Stagnation tracking: force an exceptional shift if the active block has not
    // deflated for several iterations (guards rare double-shift cycling).
    let mut since_defl = 0usize;
    let mut last_p = n;

    for _iter in 0..max_iter {
        if p <= 1 {
            break;
        }

        // Deflation from bottom: check if h[p-1, p-2] ≈ 0
        while p > 1
            && h[(p - 1) * n + (p - 2)].abs()
                <= eps * (h[(p - 2) * n + (p - 2)].abs() + h[(p - 1) * n + (p - 1)].abs())
        {
            h[(p - 1) * n + (p - 2)] = 0.0;
            p -= 1;
        }
        if p <= 1 {
            break;
        }

        // Find start of active unreduced block
        let mut lo = p - 1;
        while lo > 0
            && h[lo * n + (lo - 1)].abs()
                > eps * (h[(lo - 1) * n + (lo - 1)].abs() + h[lo * n + lo].abs())
        {
            lo -= 1;
        }
        if lo > 0 {
            h[lo * n + (lo - 1)] = 0.0;
        }

        if p - lo <= 1 {
            continue; // 1x1 block, already converged
        }

        // An isolated 2×2 active block with COMPLEX eigenvalues is already in
        // real-Schur form (a complex-conjugate pair). The single-shift QR can
        // never drive its subdiagonal to zero, so without deflating it here the
        // iteration burns the entire max_iter budget — the cause of the
        // pathological O(n²)-iteration slowdown on matrices with complex spectra.
        // Real 2×2 blocks are left to the normal QR step, which splits them into
        // two 1×1 blocks (the eigenvector path relies on that triangular form).
        if p - lo == 2 {
            let a11 = h[(p - 2) * n + (p - 2)];
            let a12 = h[(p - 2) * n + (p - 1)];
            let a21 = h[(p - 1) * n + (p - 2)];
            let a22 = h[(p - 1) * n + (p - 1)];
            let tr = a11 + a22;
            let dt = a11 * a22 - a12 * a21;
            if tr * tr - 4.0 * dt < 0.0 {
                p -= 2;
                continue;
            }
        }

        // Track stagnation: reset the counter whenever the block deflated.
        if p < last_p {
            since_defl = 0;
            last_p = p;
        } else {
            since_defl += 1;
        }

        // --- Francis double-shift implicit QR step on active block [lo, p) ---
        // The two shifts are the eigenvalues of the trailing 2×2, used as a
        // conjugate pair in real arithmetic (Golub & Van Loan, Alg. 7.5.1). This
        // gives cubic convergence and resolves complex pairs without the
        // single-shift stagnation that made eigvals pathologically slow.
        let m = p - 1;
        let (s, t) = if since_defl > 0 && since_defl.is_multiple_of(10) {
            // Exceptional shift to break a rare cycle: complex shifts of magnitude
            // ~the local subdiagonal scale.
            let sa = h[m * n + (m - 1)].abs()
                + if m >= lo + 2 { h[(m - 1) * n + (m - 2)].abs() } else { 0.0 };
            (1.5 * sa, sa * sa)
        } else {
            let tr = h[(p - 2) * n + (p - 2)] + h[(p - 1) * n + (p - 1)];
            let dt = h[(p - 2) * n + (p - 2)] * h[(p - 1) * n + (p - 1)]
                - h[(p - 2) * n + (p - 1)] * h[(p - 1) * n + (p - 2)];
            (tr, dt)
        };

        // First column of (H - λ1 I)(H - λ2 I) = H² - sH + tI (entries x, y, z).
        let h00 = h[lo * n + lo];
        let h10 = h[(lo + 1) * n + lo];
        let h11 = h[(lo + 1) * n + (lo + 1)];
        let h01 = h[lo * n + (lo + 1)];
        let mut x = h00 * h00 + h01 * h10 - s * h00 + t;
        let mut y = h10 * (h00 + h11 - s);
        let mut zz = if lo + 2 <= m { h10 * h[(lo + 2) * n + (lo + 1)] } else { 0.0 };

        let mut k = lo;
        while k <= p - 2 {
            let nr = if k + 2 <= m { 3 } else { 2 };
            let a0 = x;
            let a1 = y;
            let a2 = if nr == 3 { zz } else { 0.0 };
            let norm2 = a0 * a0 + a1 * a1 + a2 * a2;
            if norm2 > 0.0 {
                let norm = norm2.sqrt();
                let alpha = if a0 >= 0.0 { -norm } else { norm };
                let denom = a0 - alpha;
                let v1 = a1 / denom;
                let v2 = if nr == 3 { a2 / denom } else { 0.0 };
                let tau = (alpha - a0) / alpha; // H = I − tau·v·vᵀ, v = (1, v1, v2)

                // Left: P·H on rows k..k+nr, columns [colstart, n).
                let colstart = if k > lo { k - 1 } else { lo };
                if nr == 3 {
                    let row0_base = k * n;
                    let row1_base = (k + 1) * n;
                    let row2_base = (k + 2) * n;
                    let (before_row2, from_row2) = h.split_at_mut(row2_base);
                    let row2 = &mut from_row2[..n];
                    let (before_row1, from_row1) = before_row2.split_at_mut(row1_base);
                    let row1 = &mut from_row1[..n];
                    let row0 = &mut before_row1[row0_base..row0_base + n];
                    for j in colstart..n {
                        let w0 = row0[j];
                        let w1 = row1[j];
                        let w2 = row2[j];
                        let td = tau * (w0 + v1 * w1 + v2 * w2);
                        row0[j] = w0 - td;
                        row1[j] = w1 - td * v1;
                        row2[j] = w2 - td * v2;
                    }
                } else {
                    let row0_base = k * n;
                    let row1_base = (k + 1) * n;
                    let (before_row1, from_row1) = h.split_at_mut(row1_base);
                    let row1 = &mut from_row1[..n];
                    let row0 = &mut before_row1[row0_base..row0_base + n];
                    for j in colstart..n {
                        let w0 = row0[j];
                        let w1 = row1[j];
                        let td = tau * (w0 + v1 * w1);
                        row0[j] = w0 - td;
                        row1[j] = w1 - td * v1;
                    }
                }
                // Right: H·P on columns k..k+nr, rows [0, rowend).
                let rowend = (k + nr + 1).min(p);
                for row in h.chunks_mut(n).take(rowend) {
                    let w0 = row[k];
                    let w1 = row[k + 1];
                    if nr == 3 {
                        let w2 = row[k + 2];
                        let td = tau * (w0 + v1 * w1 + v2 * w2);
                        row[k] = w0 - td;
                        row[k + 1] = w1 - td * v1;
                        row[k + 2] = w2 - td * v2;
                    } else {
                        let td = tau * (w0 + v1 * w1);
                        row[k] = w0 - td;
                        row[k + 1] = w1 - td * v1;
                    }
                }
                // Schur-vector accumulation: Z·P on columns k..k+nr, all rows.
                if let Some(ref mut z) = z {
                    for row in z.chunks_mut(n) {
                        let w0 = row[k];
                        let w1 = row[k + 1];
                        if nr == 3 {
                            let w2 = row[k + 2];
                            let td = tau * (w0 + v1 * w1 + v2 * w2);
                            row[k] = w0 - td;
                            row[k + 1] = w1 - td * v1;
                            row[k + 2] = w2 - td * v2;
                        } else {
                            let td = tau * (w0 + v1 * w1);
                            row[k] = w0 - td;
                            row[k + 1] = w1 - td * v1;
                        }
                    }
                }
            }
            // Recompute the bulge from column k for the next reflector.
            if k < p - 2 {
                x = h[(k + 1) * n + k];
                y = h[(k + 2) * n + k];
                zz = if k + 3 <= m { h[(k + 3) * n + k] } else { 0.0 };
            }
            k += 1;
        }
    }
}

/// Compute eigenvalues of a symmetric NxN matrix via QR iteration.
/// Returns eigenvalues in ascending order (NumPy convention).
/// The matrix must be symmetric; behavior is undefined for non-symmetric input.
pub fn eigvalsh_nxn(a: &[f64], n: usize) -> Result<Vec<f64>, LinAlgError> {
    if Some(a.len()) != n.checked_mul(n) || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "eigvalsh_nxn: input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::SpectralConvergenceFailed);
    }

    Ok(eigvalsh_finite_nxn(a, n))
}

fn eigvalsh_finite_nxn(a: &[f64], n: usize) -> Vec<f64> {
    debug_assert_eq!(Some(a.len()), n.checked_mul(n));
    debug_assert!(n > 0);
    debug_assert!(a.iter().all(|v| v.is_finite()));

    let (mut d, mut e) = exact_symmetric_tridiagonal_values(a, n)
        .unwrap_or_else(|| tridiag_reduce_values(a, n));
    tridiag_eigvals_qr(&mut d, &mut e, n);

    d.sort_by(|a, b| a.total_cmp(b));
    d
}

fn exact_symmetric_tridiagonal_values(a: &[f64], n: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut d = vec![0.0f64; n];
    let mut e = vec![0.0f64; n.saturating_sub(1)];
    for row in 0..n {
        d[row] = a[row * n + row];
        if row + 1 < n {
            let upper = a[row * n + row + 1];
            if upper != a[(row + 1) * n + row] {
                return None;
            }
            e[row] = upper;
        }
        for col in (row + 2)..n {
            if a[row * n + col] != 0.0 || a[col * n + row] != 0.0 {
                return None;
            }
        }
    }
    Some((d, e))
}

/// Compute eigenvalues and eigenvectors of a symmetric NxN matrix via QR iteration.
/// Returns (eigenvalues, eigenvectors_flat) where eigenvectors are stored as columns
/// in a row-major n*n array. Eigenvalues are in ascending order (NumPy convention).
pub fn eigh_nxn(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), LinAlgError> {
    if Some(a.len()) != n.checked_mul(n) || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "eigh_nxn: input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::SpectralConvergenceFailed);
    }

    let (mut d, mut e, mut q) = tridiag_reduce(a, n);
    tridiag_eig_qr(&mut d, &mut e, Some(&mut q), n);

    // Sort eigenvalues ascending (NumPy convention) and permute eigenvectors accordingly
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap_or(std::cmp::Ordering::Equal));

    let sorted_eigenvalues: Vec<f64> = indices.iter().map(|&i| d[i]).collect();
    let mut sorted_v = vec![0.0; n * n];
    for (col_out, &col_in) in indices.iter().enumerate() {
        for row in 0..n {
            sorted_v[row * n + col_out] = q[row * n + col_in];
        }
    }

    Ok((sorted_eigenvalues, sorted_v))
}

/// Infinity-norm of an n×n row-major matrix: max row sum of absolute values.
/// Used internally for scale-relative threshold computation.
fn matrix_inf_norm(a: &[f64], n: usize) -> f64 {
    let mut max_row_sum = 0.0f64;
    for i in 0..n {
        let row_sum: f64 = (0..n).map(|j| a[i * n + j].abs()).sum();
        max_row_sum = max_row_sum.max(row_sum);
    }
    max_row_sum
}

/// Extract eigenvalues from quasi-upper-triangular (real Schur) form.
/// Returns interleaved `[re0, im0, re1, im1, ...]`.
fn extract_schur_eigenvalues(m: &[f64], n: usize) -> Vec<f64> {
    // Scale-relative threshold for detecting 2x2 blocks in Schur form.
    // Matches LAPACK's approach: compare sub-diagonal to eps * local diagonal magnitude.
    let mat_norm = matrix_inf_norm(m, n).max(f64::MIN_POSITIVE);
    let block_eps = f64::EPSILON * mat_norm;
    let mut eigenvalues = Vec::with_capacity(n * 2);
    let mut i = 0;
    while i < n {
        if i + 1 < n && m[(i + 1) * n + i].abs() > block_eps {
            let a11 = m[i * n + i];
            let a12 = m[i * n + (i + 1)];
            let a21 = m[(i + 1) * n + i];
            let a22 = m[(i + 1) * n + (i + 1)];
            let trace = a11 + a22;
            let det = a11 * a22 - a12 * a21;
            let disc = trace * trace - 4.0 * det;
            if disc < 0.0 {
                let real = trace / 2.0;
                let imag = (-disc).sqrt() / 2.0;
                eigenvalues.push(real);
                eigenvalues.push(imag);
                eigenvalues.push(real);
                eigenvalues.push(-imag);
            } else {
                let sqrt_disc = disc.sqrt();
                eigenvalues.push((trace + sqrt_disc) / 2.0);
                eigenvalues.push(0.0);
                eigenvalues.push((trace - sqrt_disc) / 2.0);
                eigenvalues.push(0.0);
            }
            i += 2;
        } else {
            eigenvalues.push(m[i * n + i]);
            eigenvalues.push(0.0);
            i += 1;
        }
    }
    eigenvalues
}

/// Eigenvalues of a general (possibly non-symmetric) NxN matrix via QR iteration.
/// Returns eigenvalues as interleaved (re, im) pairs: [re0, im0, re1, im1, ...].
/// For real eigenvalues, the imaginary part is 0.
/// The QR iteration converges to a quasi-upper-triangular (real Schur) form;
/// 1x1 diagonal blocks give real eigenvalues, 2x2 blocks give complex conjugate pairs.
pub fn eig_nxn(a: &[f64], n: usize) -> Result<Vec<f64>, LinAlgError> {
    if Some(a.len()) != n.checked_mul(n) || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "eig_nxn: input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(spectral_non_finite_input_error());
    }

    // Hessenberg reduction + implicit shifted QR
    let mut h = hessenberg_reduce_values(a, n);
    hessenberg_qr_iter(&mut h, None, n);

    // Extract eigenvalues from quasi-upper-triangular (real Schur) form
    let eigenvalues = extract_schur_eigenvalues(&h, n);
    Ok(eigenvalues)
}

/// Schur decomposition of a general square matrix (scipy.linalg.schur).
///
/// Returns `(T, Z)` where `A = Z * T * Z^T`, `T` is quasi-upper-triangular
/// (real Schur form: 1x1 and 2x2 blocks on diagonal), and `Z` is orthogonal.
/// Both returned as row-major flat arrays of length `n*n`.
pub fn schur_nxn(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), LinAlgError> {
    if Some(a.len()) != n.checked_mul(n) || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "schur_nxn: input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::SpectralConvergenceFailed);
    }

    // Hessenberg reduction + implicit shifted QR with Schur vector accumulation
    let (mut h, mut q) = hessenberg_reduce(a, n);
    hessenberg_qr_iter(&mut h, Some(&mut q), n);

    Ok((h, q))
}

/// Cross product of two 3-element vectors (np.cross for 3-D).
///
/// Returns `a × b = [a1*b2 - a2*b1, a2*b0 - a0*b2, a0*b1 - a1*b0]`.
pub fn cross_product(a: &[f64], b: &[f64]) -> Result<Vec<f64>, LinAlgError> {
    match (a.len(), b.len()) {
        (2, 2) => {
            // 2D cross product: returns scalar (as 1-element vec)
            // np.cross([a0, a1], [b0, b1]) = a0*b1 - a1*b0
            Ok(vec![a[0] * b[1] - a[1] * b[0]])
        }
        (2, 3) => {
            // Treat 2D as 3D with z=0
            Ok(vec![a[1] * b[2], -a[0] * b[2], a[0] * b[1] - a[1] * b[0]])
        }
        (3, 2) => {
            // Treat 2D as 3D with z=0
            Ok(vec![-a[2] * b[1], a[2] * b[0], a[0] * b[1] - a[1] * b[0]])
        }
        (3, 3) => Ok(vec![
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]),
        _ => Err(LinAlgError::ShapeContractViolation(
            "cross_product: inputs must have 2 or 3 elements",
        )),
    }
}

fn kron_identity_rhs_nonnegative_fast_path(
    a: &[f64],
    m: usize,
    n: usize,
    b: &[f64],
    p: usize,
    q: usize,
) -> Option<Vec<f64>> {
    if p != q {
        return None;
    }

    let one_bits = 1.0f64.to_bits();
    let positive_zero_bits = 0.0f64.to_bits();
    for row in 0..p {
        for col in 0..q {
            let expected = if row == col {
                one_bits
            } else {
                positive_zero_bits
            };
            if b[row * q + col].to_bits() != expected {
                return None;
            }
        }
    }
    if !a
        .iter()
        .all(|value| value.is_finite() && value.is_sign_positive())
    {
        return None;
    }

    let out_cols = n * q;
    let mut result = vec![0.0; m * p * out_cols];
    let fill_identity_row = |(r, row): (usize, &mut [f64])| {
        let i = r / p;
        let k = r % p;
        let a_base = i * n;
        for j in 0..n {
            row[j * p + k] = a[a_base + j];
        }
    };
    const KRON_PAR_MIN: usize = 1 << 18;
    if result.len() >= KRON_PAR_MIN && rayon::current_num_threads() >= 2 {
        result
            .par_chunks_mut(out_cols)
            .enumerate()
            .for_each(fill_identity_row);
    } else {
        result
            .chunks_mut(out_cols)
            .enumerate()
            .for_each(fill_identity_row);
    }
    Some(result)
}

/// Kronecker product of two matrices (np.kron).
///
/// Given `a` of shape `(m, n)` and `b` of shape `(p, q)`,
/// returns a matrix of shape `(m*p, n*q)` as a row-major flat array.
pub fn kron_nxn(
    a: &[f64],
    m: usize,
    n: usize,
    b: &[f64],
    p: usize,
    q: usize,
) -> Result<Vec<f64>, LinAlgError> {
    if Some(a.len()) != m.checked_mul(n) || Some(b.len()) != p.checked_mul(q) {
        return Err(LinAlgError::ShapeContractViolation(
            "kron_nxn: input size mismatch",
        ));
    }
    let out_rows = m * p;
    let out_cols = n * q;
    let out_count = out_rows * out_cols;
    if out_count == 0 {
        return Ok(Vec::new());
    }
    if let Some(identity_result) = kron_identity_rhs_nonnegative_fast_path(a, m, n, b, p, q) {
        return Ok(identity_result);
    }
    let mut result = vec![0.0; out_count];

    // Each output row `R = i*p + k` (so `i = R/p`, `k = R%p`) is an independent
    // function of a-row `i` and b-row `k`: `result[R][j*q + l] = a[i*n+j] *
    // b[k*q+l]`. The old nested loops also wrote each cell from a single product
    // with no accumulation, so filling the disjoint contiguous output rows across
    // the rayon pool is bit-for-bit identical to the serial order. The inner scaled
    // copy of a contiguous b-row auto-vectorizes (vs the old strided cell writes).
    let fill_row = |(r, row): (usize, &mut [f64])| {
        let i = r / p;
        let k = r % p;
        let a_base = i * n;
        let b_row = &b[k * q..k * q + q];
        for j in 0..n {
            let a_val = a[a_base + j];
            let dst = &mut row[j * q..j * q + q];
            for l in 0..q {
                dst[l] = a_val * b_row[l];
            }
        }
    };
    const KRON_PAR_MIN: usize = 1 << 18;
    if out_count >= KRON_PAR_MIN && rayon::current_num_threads() >= 2 {
        result
            .par_chunks_mut(out_cols)
            .enumerate()
            .for_each(fill_row);
    } else {
        result.chunks_mut(out_cols).enumerate().for_each(fill_row);
    }
    Ok(result)
}

/// Optimal multi-matrix multiplication (np.linalg.multi_dot).
///
/// Takes a list of matrices (as flat row-major arrays with their dimensions)
/// and finds the optimal parenthesization to minimize total scalar multiplications.
/// Each entry is `(data, rows, cols)`.
pub fn multi_dot(
    matrices: &[(&[f64], usize, usize)],
) -> Result<(Vec<f64>, usize, usize), LinAlgError> {
    if matrices.is_empty() {
        return Err(LinAlgError::ShapeContractViolation(
            "multi_dot: need at least one matrix",
        ));
    }
    if matrices.len() == 1 {
        return Ok((matrices[0].0.to_vec(), matrices[0].1, matrices[0].2));
    }
    if matrices.len() == 2 {
        let (a, m, k1) = matrices[0];
        let (b, k2, n) = matrices[1];
        if k1 != k2 {
            return Err(LinAlgError::ShapeContractViolation(
                "multi_dot: inner dimension mismatch",
            ));
        }
        let c = mat_mul_rect(a, b, m, k1, n);
        return Ok((c, m, n));
    }

    let count = matrices.len();
    // Dimensions: matrices[i] is dims[i] x dims[i+1]
    let mut dims = Vec::with_capacity(count + 1);
    dims.push(matrices[0].1);
    for (i, &(_, rows, cols)) in matrices.iter().enumerate() {
        if i > 0 && rows != dims[i] {
            return Err(LinAlgError::ShapeContractViolation(
                "multi_dot: inner dimension mismatch",
            ));
        }
        dims.push(cols);
    }

    // Dynamic programming for optimal parenthesization
    let mut cost = vec![vec![0u64; count]; count];
    let mut split = vec![vec![0usize; count]; count];
    for len in 2..=count {
        for i in 0..=count - len {
            let j = i + len - 1;
            cost[i][j] = u64::MAX;
            for k in i..j {
                let c = cost[i][k]
                    + cost[k + 1][j]
                    + (dims[i] as u64) * (dims[k + 1] as u64) * (dims[j + 1] as u64);
                if c < cost[i][j] {
                    cost[i][j] = c;
                    split[i][j] = k;
                }
            }
        }
    }

    // Recursively multiply using optimal order
    fn multiply_range(
        matrices: &[(&[f64], usize, usize)],
        split: &[Vec<usize>],
        i: usize,
        j: usize,
    ) -> (Vec<f64>, usize, usize) {
        if i == j {
            return (matrices[i].0.to_vec(), matrices[i].1, matrices[i].2);
        }
        let k = split[i][j];
        let (a, m, ka) = multiply_range(matrices, split, i, k);
        let (b, _kb, n) = multiply_range(matrices, split, k + 1, j);
        let c = mat_mul_rect(&a, &b, m, ka, n);
        (c, m, n)
    }

    let (result, rows, cols) = multiply_range(matrices, &split, 0, count - 1);
    Ok((result, rows, cols))
}

/// Eigenvalues AND eigenvectors of a general (non-symmetric) matrix (np.linalg.eig).
///
/// Returns `(eigenvalues, eigenvectors)` where:
/// - eigenvalues: interleaved `[re0, im0, re1, im1, ...]` (length `2*n`)
/// - eigenvectors: column-major interleaved complex matrix of size `n x n`,
///   stored as `[re(v[0,0]), im(v[0,0]), re(v[1,0]), im(v[1,0]), ..., re(v[0,1]), ...]`
///   Total length `2*n*n`. Column `j` is the eigenvector for eigenvalue `j`.
///
/// For real eigenvalues the imaginary parts are zero.
/// For complex conjugate pairs the two eigenvectors are also conjugate.
pub fn eig_nxn_full(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), LinAlgError> {
    if Some(a.len()) != n.checked_mul(n) || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "eig_nxn_full: input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(spectral_non_finite_input_error());
    }

    // Hessenberg reduction + implicit shifted QR accumulating Schur vectors
    let (mut h, mut z) = hessenberg_reduce(a, n);
    hessenberg_qr_iter(&mut h, Some(&mut z), n);

    // Extract eigenvalues from quasi-upper-triangular Schur form
    let eigenvalues = extract_schur_eigenvalues(&h, n);

    // Scale-relative thresholds for block detection and denominator checks.
    let h_norm = matrix_inf_norm(&h, n).max(f64::MIN_POSITIVE);
    let block_eps = f64::EPSILON * h_norm;
    let denom_eps = f64::EPSILON * h_norm;
    let denom_sq_eps = denom_eps * denom_eps;

    // Build eigenvectors of T via back-substitution, then transform by Z.
    // For each eigenvalue λ, solve (T - λI)x = 0 by back-substitution on the
    // quasi-upper-triangular Schur form, then compute v = Z * x.
    let mut vr_re = vec![0.0; n * n]; // eigenvectors of T in columns (real part)
    let mut vr_im = vec![0.0; n * n]; // eigenvectors of T in columns (imag part)

    let mut col = 0;
    while col < n {
        if col + 1 < n && h[(col + 1) * n + col].abs() > block_eps {
            // 2x2 block → complex conjugate eigenvalue pair
            let lam_re = eigenvalues[2 * col];
            let lam_im = eigenvalues[2 * col + 1];

            // Start with unit vector at block position
            vr_re[col * n + col] = 1.0;
            vr_im[col * n + col] = 0.0;
            // Second component of the 2x2 Schur block eigenvector
            let a11 = h[col * n + col];
            let a12 = h[col * n + (col + 1)];
            // (a11 - λ)*x1 + a12*x2 = 0, x1=1 → x2 = -(a11-λ)/a12
            let denom = a12;
            if denom.abs() > denom_eps {
                vr_re[(col + 1) * n + col] = -(a11 - lam_re) / denom;
                vr_im[(col + 1) * n + col] = lam_im / denom;
            }

            // Back-substitute through rows above the block
            for j in (0..col).rev() {
                // Check if row j is part of a 2x2 block
                if j > 0 && h[j * n + (j - 1)].abs() > block_eps {
                    continue; // handled as part of the 2x2 block starting at j-1
                }
                // 1x1 row: solve (T[j,j] - λ)*x[j] = -sum_{k=j+1..col+2} T[j,k]*x[k]
                let mut sum_re = 0.0;
                let mut sum_im = 0.0;
                for k in (j + 1)..=(col + 1).min(n - 1) {
                    sum_re += h[j * n + k] * vr_re[k * n + col] - 0.0 * vr_im[k * n + col];
                    sum_im += h[j * n + k] * vr_im[k * n + col];
                }
                let d_re = h[j * n + j] - lam_re;
                let d_im = -lam_im;
                let d_sq = d_re * d_re + d_im * d_im;
                if d_sq > denom_sq_eps {
                    // x[j] = -sum / (T[j,j]-λ), complex division
                    vr_re[j * n + col] = -(sum_re * d_re + sum_im * d_im) / d_sq;
                    vr_im[j * n + col] = -(sum_im * d_re - sum_re * d_im) / d_sq;
                }
            }
            // Conjugate eigenvector for the second eigenvalue
            for row in 0..n {
                vr_re[row * n + (col + 1)] = vr_re[row * n + col];
                vr_im[row * n + (col + 1)] = -vr_im[row * n + col];
            }
            col += 2;
        } else {
            // 1x1 block → real eigenvalue
            let lam = eigenvalues[2 * col];
            vr_re[col * n + col] = 1.0;
            // Back-substitute: for j = col-1 down to 0
            for j in (0..col).rev() {
                // Check if row j is part of a 2x2 block
                if j > 0 && h[j * n + (j - 1)].abs() > block_eps {
                    continue; // handled as 2x2 block
                }
                let mut sum = 0.0;
                for k in (j + 1)..=col {
                    sum += h[j * n + k] * vr_re[k * n + col];
                }
                let denom = h[j * n + j] - lam;
                if denom.abs() > denom_eps {
                    vr_re[j * n + col] = -sum / denom;
                } else {
                    // Near-singular: use sign-preserving epsilon to avoid sign flip
                    let safe_denom = if denom >= 0.0 { denom_eps } else { -denom_eps };
                    vr_re[j * n + col] = -sum / safe_denom;
                }
            }
            col += 1;
        }
    }

    // Transform: V = Z * VR (apply Schur vectors)
    let mut eigvecs_re = vec![0.0; n * n];
    let mut eigvecs_im = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            let mut re = 0.0;
            let mut im = 0.0;
            for k in 0..n {
                re += z[i * n + k] * vr_re[k * n + j];
                im += z[i * n + k] * vr_im[k * n + j];
            }
            eigvecs_re[i * n + j] = re;
            eigvecs_im[i * n + j] = im;
        }
    }

    // Normalize each eigenvector column
    for j in 0..n {
        let mut norm_sq = 0.0;
        for i in 0..n {
            let re = eigvecs_re[i * n + j];
            let im = eigvecs_im[i * n + j];
            norm_sq += re * re + im * im;
        }
        let norm = norm_sq.sqrt();
        if norm > f64::MIN_POSITIVE {
            for i in 0..n {
                eigvecs_re[i * n + j] /= norm;
                eigvecs_im[i * n + j] /= norm;
            }
        }
    }

    // Interleave into output format: [re(v[0,0]), im(v[0,0]), re(v[1,0]), im(v[1,0]), ...]
    let mut eigvecs = Vec::with_capacity(2 * n * n);
    for j in 0..n {
        for i in 0..n {
            eigvecs.push(eigvecs_re[i * n + j]);
            eigvecs.push(eigvecs_im[i * n + j]);
        }
    }

    Ok((eigenvalues, eigvecs))
}

/// Condition number of a matrix (np.linalg.cond).
/// Uses the ratio of largest to smallest singular value (2-norm condition number).
pub fn cond_nxn(a: &[f64], n: usize) -> Result<f64, LinAlgError> {
    cond_p_nxn(a, n, None)
}

#[inline]
fn is_exact_symmetric_nxn(a: &[f64], n: usize) -> bool {
    for i in 0..n {
        for j in (i + 1)..n {
            if a[i * n + j] != a[j * n + i] {
                return false;
            }
        }
    }
    true
}

fn symmetric_cond_from_eigvalsh(a: &[f64], n: usize, reciprocal: bool) -> f64 {
    let eigvals = eigvalsh_finite_nxn(a, n);
    let mut sigma_min = f64::INFINITY;
    let mut sigma_max = 0.0f64;
    for eig in eigvals {
        let sigma = eig.abs();
        sigma_min = sigma_min.min(sigma);
        sigma_max = sigma_max.max(sigma);
    }

    if reciprocal {
        if sigma_max == 0.0 {
            f64::INFINITY
        } else {
            sigma_min / sigma_max
        }
    } else if sigma_min == 0.0 {
        f64::INFINITY
    } else {
        sigma_max / sigma_min
    }
}

/// Condition number for rectangular MxN matrices (np.linalg.cond).
/// Only the 2-norm is supported for non-square matrices.
/// Returns sigma_max / sigma_min from SVD.
pub fn cond_mxn(a: &[f64], m: usize, n: usize) -> Result<f64, LinAlgError> {
    if m == 0 || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "cond is not defined on empty arrays",
        ));
    }
    if Some(a.len()) != m.checked_mul(n) {
        return Err(LinAlgError::ShapeContractViolation(
            "cond_mxn: input must be m*n",
        ));
    }
    let has_nan = a.iter().any(|value| value.is_nan());
    if has_nan {
        // NumPy raises "SVD did not converge" for NaN inputs
        return Err(LinAlgError::SvdNonConvergence);
    }
    let has_inf = a.iter().any(|value| value.is_infinite());
    if has_inf {
        return Ok(f64::INFINITY);
    }
    let sigmas = svd_mxn(a, m, n)?;
    let sigma_max = sigmas.first().copied().unwrap_or(0.0);
    let sigma_min = sigmas.last().copied().unwrap_or(0.0);
    if sigma_min == 0.0 {
        return Ok(f64::INFINITY);
    }
    Ok(sigma_max / sigma_min)
}

/// Condition number with explicit norm order (np.linalg.cond with p parameter).
///
/// `p`: `None` defaults to 2-norm. Supported: `"1"`, `"-1"`, `"2"`, `"-2"`,
/// `"inf"`, `"-inf"`, `"fro"`.
/// For p=2 or p=-2, uses SVD (sigma_max/sigma_min).
/// For other p, computes `norm(A, p) * norm(inv(A), p)`.
pub fn cond_p_nxn(a: &[f64], n: usize, p: Option<&str>) -> Result<f64, LinAlgError> {
    if n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "cond is not defined on empty arrays",
        ));
    }
    let ord = p.unwrap_or("2");
    let has_nan = a.iter().any(|value| value.is_nan());
    let has_inf = a.iter().any(|value| value.is_infinite());
    if Some(a.len()) != n.checked_mul(n) {
        if n == 0 {
            return Err(LinAlgError::ShapeContractViolation(
                "cond is not defined on empty arrays",
            ));
        }
        if !a.len().is_multiple_of(n) {
            return Err(LinAlgError::ShapeContractViolation(
                "cond_p_nxn: input must be n*n or m*n with n > 0",
            ));
        }
        let m = a.len() / n;
        if m == 0 {
            return Err(LinAlgError::ShapeContractViolation(
                "cond is not defined on empty arrays",
            ));
        }
        if ord != "2" && ord != "-2" {
            return Err(LinAlgError::ShapeContractViolation(
                "cond: non-square matrices only support p=None, '2', or '-2'",
            ));
        }
        if has_inf {
            return Ok(f64::INFINITY);
        }
        let sigmas = svd_mxn(a, m, n)?;
        let sigma_max = sigmas.first().copied().unwrap_or(0.0);
        let sigma_min = sigmas.last().copied().unwrap_or(0.0);
        if ord == "-2" {
            if sigma_max == 0.0 {
                return Ok(f64::INFINITY);
            }
            return Ok(sigma_min / sigma_max);
        }
        if sigma_min == 0.0 {
            return Ok(f64::INFINITY);
        }
        return Ok(sigma_max / sigma_min);
    }
    match ord {
        "2" => {
            if has_inf {
                return Ok(f64::INFINITY);
            }
            if !has_nan && is_exact_symmetric_nxn(a, n) {
                return Ok(symmetric_cond_from_eigvalsh(a, n, false));
            }
            let sigmas = svd_nxn(a, n)?;
            let sigma_max = sigmas.first().copied().unwrap_or(0.0);
            let sigma_min = sigmas.last().copied().unwrap_or(0.0);
            if sigma_min == 0.0 {
                return Ok(f64::INFINITY);
            }
            Ok(sigma_max / sigma_min)
        }
        "-2" => {
            if has_inf {
                return Ok(f64::INFINITY);
            }
            if !has_nan && is_exact_symmetric_nxn(a, n) {
                return Ok(symmetric_cond_from_eigvalsh(a, n, true));
            }
            let sigmas = svd_nxn(a, n)?;
            let sigma_max = sigmas.first().copied().unwrap_or(0.0);
            let sigma_min = sigmas.last().copied().unwrap_or(0.0);
            if sigma_max == 0.0 {
                return Ok(f64::INFINITY);
            }
            Ok(sigma_min / sigma_max)
        }
        "1" | "-1" | "inf" | "-inf" | "fro" => {
            if has_nan {
                return Ok(f64::NAN);
            }
            let norm_a = matrix_norm_nxn(a, n, n, ord)?;
            // NumPy computes cond(A, p) = norm(A, p) * norm(inv(A), p) under
            // `errstate(all="ignore")`: a singular A makes inv(A) blow up, so
            // the condition number is +inf rather than an error. NumPy also
            // maps a nan result produced from a finite input to +inf.
            match inv_nxn(a, n) {
                Ok(a_inv) => {
                    let norm_inv = matrix_norm_nxn(&a_inv, n, n, ord)?;
                    let r = norm_a * norm_inv;
                    Ok(if r.is_nan() { f64::INFINITY } else { r })
                }
                Err(LinAlgError::SolverSingularity) => Ok(f64::INFINITY),
                Err(other) => Err(other),
            }
        }
        _ => Err(LinAlgError::ShapeContractViolation(
            "cond: p must be one of None, '1', '-1', '2', '-2', 'inf', '-inf', 'fro'",
        )),
    }
}

/// Matrix exponentiation: compute A^p for integer p (np.linalg.matrix_power).
/// Uses repeated squaring. p can be negative (requires invertible matrix).
pub fn matrix_power_nxn(a: &[f64], n: usize, p: i64) -> Result<Vec<f64>, LinAlgError> {
    if Some(a.len()) != n.checked_mul(n) || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "matrix_power_nxn: input must be n*n with n > 0",
        ));
    }

    if p == 0 {
        // A^0 = I
        let mut eye = vec![0.0; n * n];
        for i in 0..n {
            eye[i * n + i] = 1.0;
        }
        return Ok(eye);
    }

    let base = if p < 0 { inv_nxn(a, n)? } else { a.to_vec() };

    // The square-and-multiply ladder seeds `result = I` and folds in `result @ cur`
    // at each set bit. The FIRST such fold is `I @ cur`, which equals `cur` exactly
    // when `cur` is finite (1.0*x + Σ 0.0*y = x, bit-for-bit) — so we can skip that
    // GEMM and clone `cur` instead. The old code only elided it for ODD powers (the
    // lowest bit set), leaving a wasted `I @ cur` GEMM on every EVEN power — e.g.
    // A² ran two GEMMs (A@A then I@A²) when one suffices. Tracking whether `result`
    // is still the identity elides that first GEMM for ANY power. Gated on
    // finite-nonzero entries: `0.0*∞`/`0.0*NaN` make `I @ cur` differ from `cur`, so
    // a base with a zero or non-finite entry keeps the explicit identity seed to
    // preserve numpy's NaN/Inf propagation. The odd-power GEMM schedule is unchanged
    // (golden-pinned), so this is bit-identical there and on the elided even powers.
    let elide_identity = p > 0 && base.iter().all(|&v| v.is_finite() && v != 0.0);
    if elide_identity {
        let mut exp = p.unsigned_abs();
        let mut cur = base;
        let mut result: Vec<f64> = Vec::new();
        let mut result_is_identity = true;
        while exp > 0 {
            if exp & 1 == 1 {
                if result_is_identity {
                    result = cur.clone();
                    result_is_identity = false;
                } else {
                    result = mat_mul_flat(&result, &cur, n);
                }
            }
            exp >>= 1;
            if exp > 0 {
                cur = mat_mul_flat(&cur, &cur, n);
            }
        }
        return Ok(result);
    }

    let mut exp = p.unsigned_abs();
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        result[i * n + i] = 1.0; // identity
    }
    let mut cur = base;

    while exp > 0 {
        if exp & 1 == 1 {
            result = mat_mul_flat(&result, &cur, n);
        }
        exp >>= 1;
        if exp > 0 {
            cur = mat_mul_flat(&cur, &cur, n);
        }
    }
    Ok(result)
}

/// Matrix exponential via scaling-and-squaring with Taylor series.
///
/// Mimics `scipy.linalg.expm(A)`. Computes e^A for an NxN matrix.
/// Uses scaling-and-squaring: scale A by 2^(-s) so the norm is small,
/// compute exp via truncated Taylor series, then square s times.
pub fn expm_nxn(a: &[f64], n: usize) -> Result<Vec<f64>, LinAlgError> {
    if Some(a.len()) != n.checked_mul(n) || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "expm_nxn: input must be n*n with n > 0",
        ));
    }
    if let Some(result) = expm_non_finite_compat(a, n) {
        return Ok(result);
    }

    // 1-norm of matrix
    let norm1 = {
        let mut max_col = 0.0f64;
        for j in 0..n {
            let col_sum: f64 = (0..n).map(|i| a[i * n + j].abs()).sum();
            if col_sum > max_col {
                max_col = col_sum;
            }
        }
        max_col
    };

    // Determine scaling factor s such that ||A/2^s|| < 1
    let s = if norm1 > 1.0 {
        (norm1.log2().ceil().max(0.0) as u32) + 1
    } else {
        0
    };

    // Scale: A_s = A / 2^s
    let scale = 2.0f64.powi(s as i32);
    let a_s: Vec<f64> = a.iter().map(|&v| v / scale).collect();

    // Taylor series: exp(A_s) = I + A_s + A_s^2/2! + A_s^3/3! + ...
    let mut result = vec![0.0; n * n];
    // Initialize to identity
    for i in 0..n {
        result[i * n + i] = 1.0;
    }

    // term = A_s^k / k!
    let mut term = vec![0.0; n * n];
    for i in 0..n {
        term[i * n + i] = 1.0; // identity = A^0 / 0!
    }

    for k in 1..=MATRIX_FUNC_TAYLOR_TERMS {
        // term = term * A_s / k
        let new_term = mat_mul_flat(&term, &a_s, n);
        let inv_k = 1.0 / k as f64;
        for i in 0..n * n {
            term[i] = new_term[i] * inv_k;
        }
        // result += term
        for i in 0..n * n {
            result[i] += term[i];
        }
        // Adaptive convergence: stop when term is negligible relative to result
        let term_norm: f64 = term.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let result_norm = matrix_inf_norm(&result, n).max(f64::MIN_POSITIVE);
        if term_norm < f64::EPSILON * result_norm {
            break;
        }
    }

    // Undo scaling: square s times
    for _ in 0..s {
        result = mat_mul_flat(&result, &result, n);
    }

    Ok(result)
}

/// Matrix square root via Denman-Beavers iteration.
///
/// Mimics `scipy.linalg.sqrtm(A)`. Computes X such that X @ X = A.
/// Uses the Denman-Beavers iteration which converges quadratically:
///   Y_{k+1} = 0.5 * (Y_k + Z_k^{-1})
///   Z_{k+1} = 0.5 * (Z_k + Y_k^{-1})
/// with Y_0 = A, Z_0 = I, and Y_∞ = A^{1/2}.
pub fn sqrtm_nxn(a: &[f64], n: usize) -> Result<Vec<f64>, LinAlgError> {
    if Some(a.len()) != n.checked_mul(n) || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "sqrtm_nxn: input must be n*n with n > 0",
        ));
    }
    if let Some(result) = sqrtm_non_finite_compat(a, n) {
        return Ok(result);
    }

    let mut y = a.to_vec();
    let mut z = vec![0.0; n * n];
    for i in 0..n {
        z[i * n + i] = 1.0; // Z_0 = I
    }

    for _ in 0..SQRTM_MAX_ITERATIONS {
        let z_inv = inv_nxn(&z, n)?;
        let y_inv = inv_nxn(&y, n)?;

        let mut y_new = vec![0.0; n * n];
        let mut z_new = vec![0.0; n * n];
        for i in 0..n * n {
            y_new[i] = 0.5 * (y[i] + z_inv[i]);
            z_new[i] = 0.5 * (z[i] + y_inv[i]);
        }

        // Relative convergence: ||Y_new - Y||_inf / max(||Y||_inf, 1)
        let diff: f64 = y_new
            .iter()
            .zip(y.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        let y_norm = matrix_inf_norm(&y, n).max(1.0);
        y = y_new;
        z = z_new;
        if diff < f64::EPSILON * 64.0 * y_norm {
            break;
        }
    }

    Ok(y)
}

/// Matrix logarithm via inverse scaling-and-squaring.
///
/// Mimics `scipy.linalg.logm(A)`. Computes X such that expm(X) = A.
/// Uses repeated square roots to bring the matrix close to I, then
/// applies the series log(I + X) ≈ X - X²/2 + X³/3 - ...
pub fn logm_nxn(a: &[f64], n: usize) -> Result<Vec<f64>, LinAlgError> {
    if Some(a.len()) != n.checked_mul(n) || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "logm_nxn: input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::SpectralConvergenceFailed);
    }

    // Inverse scaling: compute A^{1/2^s} until close to identity
    let mut m = a.to_vec();
    let mut s = 0u32;
    let max_s = u32::try_from(LOGM_MAX_SCALING_ITERATIONS).unwrap_or(100);

    let mut identity = vec![0.0; n * n];
    for i in 0..n {
        identity[i * n + i] = 1.0;
    }

    // Keep taking square roots until ||M - I|| is small
    while s < max_s {
        let diff: f64 = m
            .iter()
            .zip(identity.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        if diff < 0.5 {
            break;
        }
        m = sqrtm_nxn(&m, n)?;
        s += 1;
    }

    // Now M ≈ I + X where X is small
    // Compute X = M - I
    let mut x = vec![0.0; n * n];
    for i in 0..n * n {
        x[i] = m[i] - identity[i];
    }

    // log(I + X) via Taylor series: X - X²/2 + X³/3 - X⁴/4 + ...
    let mut result = vec![0.0; n * n];
    let mut x_power = x.clone(); // X^1
    for k in 1..=MATRIX_FUNC_TAYLOR_TERMS {
        let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
        let coeff = sign / k as f64;
        for i in 0..n * n {
            result[i] += coeff * x_power[i];
        }
        x_power = mat_mul_flat(&x_power, &x, n);

        // Adaptive convergence: stop when term is negligible relative to result
        let term_norm: f64 = x_power.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let result_norm = matrix_inf_norm(&result, n).max(f64::MIN_POSITIVE);
        if term_norm < f64::EPSILON * result_norm {
            break;
        }
    }

    // Undo scaling: multiply by 2^s
    let scale = 2.0f64.powi(s as i32);
    for v in &mut result {
        *v *= scale;
    }

    Ok(result)
}

/// General matrix function: compute f(A) for an NxN matrix.
///
/// Uses Schur decomposition: A = Z * T * Z^T where Z is orthogonal.
/// For matrices with real eigenvalues on the diagonal of T, applies f to each
/// diagonal element and reconstructs: f(A) = Z * f(T) * Z^T.
///
/// Only supports matrices whose Schur form is (quasi-)upper triangular with
/// real diagonal entries (i.e., all eigenvalues are real).
///
/// Equivalent to `scipy.linalg.funm(A, func)`.
pub fn funm_nxn(a: &[f64], n: usize, f: impl Fn(f64) -> f64) -> Result<Vec<f64>, LinAlgError> {
    if Some(a.len()) != n.checked_mul(n) || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "funm_nxn: input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::SpectralConvergenceFailed);
    }

    // Schur decomposition: A = Z * T * Z^T, Z orthogonal
    let (t, z) = schur_nxn(a, n)?;

    // Check that T is diagonal (off-diagonal entries negligible)
    for i in 0..n {
        for j in 0..n {
            if i != j && t[i * n + j].abs() > 1e-10 {
                return Err(LinAlgError::SpectralConvergenceFailed);
            }
        }
    }

    // Apply f to diagonal of T (the eigenvalues)
    let mut ft = vec![0.0; n * n];
    for i in 0..n {
        ft[i * n + i] = f(t[i * n + i]);
    }

    // Reconstruct: f(A) = Z * f(T) * Z^T
    let z_ft = mat_mul_flat(&z, &ft, n);
    // Z^T (transpose of Z)
    let mut z_t = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            z_t[i * n + j] = z[j * n + i];
        }
    }
    Ok(mat_mul_flat(&z_ft, &z_t, n))
}

/// Polar decomposition of an NxN matrix: A = U * P.
///
/// `U` is a unitary (orthogonal) matrix and `P` is a positive semi-definite
/// symmetric matrix. Computed via SVD: if A = Us * S * Vt, then U = Us * Vt
/// and P = V * S * Vt.
///
/// Returns `(u_flat, p_flat)` each of length n*n.
///
/// Equivalent to `scipy.linalg.polar(a)`.
pub fn polar_nxn(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), LinAlgError> {
    if Some(a.len()) != n.checked_mul(n) || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "polar_nxn: input must be n*n with n > 0",
        ));
    }

    // Get full SVD: A = Us * diag(S) * Vt
    let (u_s, s_vals, vt) = svd_mxn_full(a, n, n)?;

    // U = Us * Vt
    let u_mat = mat_mul_flat(&u_s, &vt, n);

    // P = V * diag(S) * Vt
    // First compute V (transpose of Vt)
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            v[i * n + j] = vt[j * n + i];
        }
    }

    // V * diag(S) — multiply columns of V by singular values
    let mut v_s = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            v_s[i * n + j] = v[i * n + j] * s_vals[j];
        }
    }

    // P = V_S * Vt
    let p_mat = mat_mul_flat(&v_s, &vt, n);

    Ok((u_mat, p_mat))
}

/// NxN least-squares solve: minimize ||Ax - b||_2 using normal equations.
/// Returns the solution vector x.
/// NxN least-squares solve: minimize ||Ax - b||_2 using SVD for numerical stability.
///
/// Matches `numpy.linalg.lstsq` semantics. Returns `(x, residuals, rank, singular_values)`.
pub fn lstsq_svd(
    a: &[f64],
    b: &[f64],
    m: usize,
    n: usize,
    rcond: f64,
) -> Result<LstsqResult, LinAlgError> {
    if Some(a.len()) != m.checked_mul(n) || b.len() != m || m == 0 || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "lstsq_svd: a must be m*n, b must be m",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) || b.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "entries must be finite for lstsq",
        ));
    }

    let (u, s, vt) = svd_mxn_full(a, m, n)?;
    let k = s.len();
    let sigma_max = s.first().copied().unwrap_or(0.0);
    let threshold = if rcond < 0.0 {
        f64::EPSILON * (m.max(n) as f64) * sigma_max
    } else {
        rcond * sigma_max
    };

    let mut s_inv = vec![0.0; k];
    let mut rank = 0;
    for i in 0..k {
        if s[i] > threshold {
            s_inv[i] = 1.0 / s[i];
            rank += 1;
        }
    }

    // x = V * S^+ * U^T * b
    // 1. utb = U^T * b  (m x m) * (m x 1) -> (m x 1)
    let mut utb = vec![0.0; m];
    for i in 0..m {
        let mut sum = 0.0;
        for j in 0..m {
            sum += u[j * m + i] * b[j]; // U^T[i,j] = U[j,i]
        }
        utb[i] = sum;
    }

    // 2. siutb = S^+ * utb  (n x m) * (m x 1) -> (n x 1)
    let mut siutb = vec![0.0; n];
    for i in 0..k {
        siutb[i] = s_inv[i] * utb[i];
    }

    // 3. x = V * siutb = Vt^T * siutb  (n x n) * (n x 1) -> (n x 1)
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += vt[j * n + i] * siutb[j]; // V[i,j] = Vt^T[i,j] = Vt[j,i]
        }
        x[i] = sum;
    }

    // Residuals (only if full rank and m > n)
    let mut residuals = Vec::new();
    if rank == n && m > n {
        let sum_sq: f64 = utb[n..m].iter().map(|v| v * v).sum();
        residuals.push(sum_sq);
    }

    Ok((x, residuals, rank, s))
}

pub fn lstsq_nxn(a: &[f64], b: &[f64], m: usize, n: usize) -> Result<Vec<f64>, LinAlgError> {
    let (x, _, _, _) = lstsq_svd(a, b, m, n, -1.0)?;
    Ok(x)
}

/// Pseudoinverse of an MxN matrix (np.linalg.pinv) via SVD.
pub fn pinv_mxn(a: &[f64], m: usize, n: usize, rcond: f64) -> Result<Vec<f64>, LinAlgError> {
    if Some(a.len()) != m.checked_mul(n) || m == 0 || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "pinv_mxn: a must be m*n with m,n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "entries must be finite for pinv",
        ));
    }

    let (u, s, vt) = svd_mxn_full(a, m, n)?;
    let k = s.len();
    let sigma_max = s.first().copied().unwrap_or(0.0);
    let threshold = pinv_singular_value_cutoff(sigma_max, m, n, rcond);

    let mut s_inv = vec![0.0; k];
    for i in 0..k {
        if s[i] > threshold {
            s_inv[i] = 1.0 / s[i];
        }
    }

    // A^+ = V * S^+ * U^T
    // result_{i,j} = sum_k Vt_{k,i} * s_inv[k] * U_{j,k}
    let mut result = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut sum = 0.0;
            for kk in 0..k {
                if s_inv[kk] > 0.0 {
                    sum += vt[kk * n + i] * s_inv[kk] * u[j * m + kk];
                }
            }
            result[i * m + j] = sum;
        }
    }
    Ok(result)
}

pub fn pinv_nxn(a: &[f64], m: usize, n: usize) -> Result<Vec<f64>, LinAlgError> {
    pinv_mxn(a, m, n, -1.0)
}

fn resolve_pinv_tolerance_aliases(
    rcond: Option<f64>,
    rtol: Option<Option<f64>>,
) -> Result<f64, LinAlgError> {
    if rcond.is_some() && rtol.is_some() {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "`rtol` and `rcond` can't be both set.",
        ));
    }

    Ok(match rtol {
        Some(Some(value)) => value,
        Some(None) => -1.0,
        None => rcond.unwrap_or(-1.0),
    })
}

fn pinv_singular_value_cutoff(sigma_max: f64, m: usize, n: usize, rcond: f64) -> f64 {
    if rcond.is_nan() {
        f64::INFINITY
    } else if rcond < 0.0 {
        f64::EPSILON * (m.max(n) as f64) * sigma_max
    } else {
        rcond * sigma_max
    }
}

pub fn pinv_mxn_with_tolerance_aliases(
    a: &[f64],
    m: usize,
    n: usize,
    rcond: Option<f64>,
    rtol: Option<Option<f64>>,
) -> Result<Vec<f64>, LinAlgError> {
    let resolved_rcond = resolve_pinv_tolerance_aliases(rcond, rtol)?;
    pinv_mxn(a, m, n, resolved_rcond)
}

fn hermitian_from_lower_triangle(a: &[f64], n: usize) -> Vec<f64> {
    let mut hermitian = vec![0.0; n * n];
    for row in 0..n {
        for col in 0..=row {
            let value = a[row * n + col];
            hermitian[row * n + col] = value;
            hermitian[col * n + row] = value;
        }
    }
    hermitian
}

/// Pseudoinverse of a real Hermitian NxN matrix (NumPy `pinv(..., hermitian=True)` semantics).
///
/// NumPy uses only the lower triangle when `hermitian=True`; the upper triangle is ignored.
pub fn pinv_hermitian_nxn(a: &[f64], n: usize, rcond: f64) -> Result<Vec<f64>, LinAlgError> {
    if Some(a.len()) != n.checked_mul(n) || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "pinv_hermitian_nxn: input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "entries must be finite for pinv",
        ));
    }

    let hermitian = hermitian_from_lower_triangle(a, n);
    let (eigenvalues, eigenvectors) = eigh_nxn(&hermitian, n)?;
    let sigma_max = eigenvalues
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let threshold = pinv_singular_value_cutoff(sigma_max, n, n, rcond);

    let mut inverted = vec![0.0; n];
    for (index, eigenvalue) in eigenvalues.iter().copied().enumerate() {
        if eigenvalue.abs() > threshold {
            inverted[index] = 1.0 / eigenvalue;
        }
    }

    let mut result = vec![0.0; n * n];
    for row in 0..n {
        for col in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                if inverted[k] != 0.0 {
                    sum += eigenvectors[row * n + k] * inverted[k] * eigenvectors[col * n + k];
                }
            }
            result[row * n + col] = sum;
        }
    }
    Ok(result)
}

pub fn pinv_hermitian_nxn_with_tolerance_aliases(
    a: &[f64],
    n: usize,
    rcond: Option<f64>,
    rtol: Option<Option<f64>>,
) -> Result<Vec<f64>, LinAlgError> {
    let resolved_rcond = resolve_pinv_tolerance_aliases(rcond, rtol)?;
    pinv_hermitian_nxn(a, n, resolved_rcond)
}

/// Helper: flat NxN matrix multiply C = A * B (row-major).
/// Minimum dimension at which the square GEMM parallelizes across the rayon
/// pool. Below this the O(n³) work is too small to amortize thread dispatch, so
/// the serial ikj loop wins. At n >= 128 the per-row work (n^2
/// fused-multiply-adds) and the n rows give the pool enough independent,
/// compute-bound tasks in the measured Criterion lane.
const MATMUL_PARALLEL_MIN_DIM: usize = 128;
const MATMUL_ROW_BLOCK: usize = 4;

const PACKED_MR: usize = 4;
const PACKED_NR: usize = 8;

// Cache-blocked, B-packed GEMM serial kernel: A(m×k)·B(k×n) accumulated into a
// pre-zeroed `out` (m×n). Ported from fnp-ufunc's matmul_accumulate_serial
// (e509860c): NC column-panel blocking keeps a ~256 KiB B panel L2-resident, the
// panel is packed into a contiguous kk-major micropanel so the hot B read is a
// sequential (prefetchable) stream instead of the stride-n / DRAM-restreamed
// access of the old ikj kernel, and an MR×NR register tile accumulates the full
// k in ascending order. Every output element sums k in the SAME ascending order
// as the naive ikj loop, so the result is BIT-IDENTICAL (locked by the
// mat_mul_*_row_parallel_matches_serial_reference_and_golden_sha256 tests).
fn packed_gemm_serial(a: &[f64], b: &[f64], m: usize, k: usize, n: usize, out: &mut [f64]) {
    let m_full = m - m % PACKED_MR;
    let n_full = n - n % PACKED_NR;
    let nc = {
        let cols = (256 * 1024) / (k.max(1) * core::mem::size_of::<f64>());
        (cols / PACKED_NR).max(1) * PACKED_NR
    };
    let mut bp = vec![0.0f64; k * PACKED_NR];
    let mut jc = 0;
    while jc < n_full {
        let jc_end = (jc + nc).min(n_full);
        let mut j0 = jc;
        while j0 < jc_end {
            for kk in 0..k {
                bp[kk * PACKED_NR..kk * PACKED_NR + PACKED_NR]
                    .copy_from_slice(&b[kk * n + j0..kk * n + j0 + PACKED_NR]);
            }
            let mut i0 = 0;
            while i0 < m_full {
                let mut acc = [[0.0f64; PACKED_NR]; PACKED_MR];
                for kk in 0..k {
                    let brow = &bp[kk * PACKED_NR..kk * PACKED_NR + PACKED_NR];
                    for (ii, row) in acc.iter_mut().enumerate() {
                        let av = a[(i0 + ii) * k + kk];
                        for (slot, &bv) in row.iter_mut().zip(brow) {
                            *slot += av * bv;
                        }
                    }
                }
                for (ii, row) in acc.iter().enumerate() {
                    let base = (i0 + ii) * n + j0;
                    for (slot, &v) in out[base..base + PACKED_NR].iter_mut().zip(row) {
                        *slot += v;
                    }
                }
                i0 += PACKED_MR;
            }
            j0 += PACKED_NR;
        }
        jc += nc;
    }
    // Remainder columns for the full row blocks, then all remainder rows.
    for i in 0..m_full {
        packed_row_tail(a, b, out, i, k, n, n_full);
    }
    for i in m_full..m {
        packed_row_tail(a, b, out, i, k, n, 0);
    }
}

// out[i, j0..n] += sum_k a[i,k]*b[k,j], summed in ascending k (bit-exact tail).
fn packed_row_tail(a: &[f64], b: &[f64], out: &mut [f64], i: usize, k: usize, n: usize, j0: usize) {
    let a_base = i * k;
    let o_base = i * n;
    for j in j0..n {
        let mut s = 0.0f64;
        for kk in 0..k {
            s += a[a_base + kk] * b[kk * n + j];
        }
        out[o_base + j] += s;
    }
}

// Band-parallel driver shared by the square and rectangular wrappers: split the
// output rows into bands (several per thread for work-stealing balance), each
// band runs the packed serial kernel on its disjoint row slice. Per-row k-order
// is unchanged, so the parallel result is bit-identical to the serial kernel.
fn packed_gemm(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    let parallel = m >= MATMUL_PARALLEL_MIN_DIM
        && k >= MATMUL_PARALLEL_MIN_DIM
        && n >= MATMUL_PARALLEL_MIN_DIM
        && rayon::current_num_threads() >= 2;
    if parallel {
        let threads = rayon::current_num_threads();
        // MR-aligned band height: a few bands per thread for work-stealing, each a
        // whole number of register tiles so every band stays on the fast
        // register-tiled path (a non-aligned band leaves a remainder row that
        // falls to the slow scalar tail — measured ~4x slower overall on
        // matrix_power and the blocked-factorization GEMMs).
        let band_rows = (m.div_ceil(threads * 4).div_ceil(PACKED_MR).max(1)) * PACKED_MR;
        c.par_chunks_mut(band_rows * n)
            .enumerate()
            .for_each(|(bi, c_band)| {
                let row_start = bi * band_rows;
                let rows = c_band.len() / n;
                let a_band = &a[row_start * k..row_start * k + rows * k];
                packed_gemm_serial(a_band, b, rows, k, n, c_band);
            });
    } else {
        packed_gemm_serial(a, b, m, k, n, &mut c);
    }
    c
}

fn packed_gemm_sub_assign(a: &[f64], b: &[f64], m: usize, k: usize, n: usize, target: &mut [f64]) {
    debug_assert_eq!(target.len(), m * n);
    let parallel = m >= MATMUL_PARALLEL_MIN_DIM
        && k >= MATMUL_PARALLEL_MIN_DIM
        && n >= MATMUL_PARALLEL_MIN_DIM
        && rayon::current_num_threads() >= 2;
    if parallel {
        let threads = rayon::current_num_threads();
        let band_rows = (m.div_ceil(threads * 4).div_ceil(PACKED_MR).max(1)) * PACKED_MR;
        target
            .par_chunks_mut(band_rows * n)
            .enumerate()
            .for_each(|(bi, target_band)| {
                let row_start = bi * band_rows;
                let rows = target_band.len() / n;
                let a_band = &a[row_start * k..row_start * k + rows * k];
                packed_gemm_sub_assign_serial(a_band, b, rows, k, n, target_band);
            });
    } else {
        packed_gemm_sub_assign_serial(a, b, m, k, n, target);
    }
}

fn packed_gemm_sub_assign_strided(
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    n: usize,
    row_stride: usize,
    target: &mut [f64],
) {
    if m == 0 || n == 0 {
        return;
    }
    debug_assert!(row_stride >= n);
    debug_assert!(target.len() >= (m - 1) * row_stride + n);
    let parallel = m >= MATMUL_PARALLEL_MIN_DIM
        && k >= MATMUL_PARALLEL_MIN_DIM
        && n >= MATMUL_PARALLEL_MIN_DIM
        && rayon::current_num_threads() >= 2;
    if parallel {
        let threads = rayon::current_num_threads();
        let band_rows = (m.div_ceil(threads * 4).div_ceil(PACKED_MR).max(1)) * PACKED_MR;
        target
            .par_chunks_mut(band_rows * row_stride)
            .enumerate()
            .for_each(|(bi, target_band)| {
                let row_start = bi * band_rows;
                if row_start >= m {
                    return;
                }
                let rows = (m - row_start).min(band_rows);
                let a_band = &a[row_start * k..row_start * k + rows * k];
                packed_gemm_sub_assign_strided_serial(
                    a_band,
                    b,
                    rows,
                    k,
                    n,
                    row_stride,
                    target_band,
                );
            });
    } else {
        packed_gemm_sub_assign_strided_serial(a, b, m, k, n, row_stride, target);
    }
}

fn packed_gemm_sub_assign_serial(
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    n: usize,
    target: &mut [f64],
) {
    let m_full = m - m % PACKED_MR;
    let n_full = n - n % PACKED_NR;
    let nc = {
        let cols = (256 * 1024) / (k.max(1) * core::mem::size_of::<f64>());
        (cols / PACKED_NR).max(1) * PACKED_NR
    };
    let mut bp = vec![0.0f64; k * PACKED_NR];
    let mut jc = 0;
    while jc < n_full {
        let jc_end = (jc + nc).min(n_full);
        let mut j0 = jc;
        while j0 < jc_end {
            for kk in 0..k {
                bp[kk * PACKED_NR..kk * PACKED_NR + PACKED_NR]
                    .copy_from_slice(&b[kk * n + j0..kk * n + j0 + PACKED_NR]);
            }
            let mut i0 = 0;
            while i0 < m_full {
                let mut acc = [[0.0f64; PACKED_NR]; PACKED_MR];
                for kk in 0..k {
                    let brow = &bp[kk * PACKED_NR..kk * PACKED_NR + PACKED_NR];
                    for (ii, row) in acc.iter_mut().enumerate() {
                        let av = a[(i0 + ii) * k + kk];
                        for (slot, &bv) in row.iter_mut().zip(brow) {
                            *slot += av * bv;
                        }
                    }
                }
                for (ii, row) in acc.iter().enumerate() {
                    let base = (i0 + ii) * n + j0;
                    for (slot, &v) in target[base..base + PACKED_NR].iter_mut().zip(row) {
                        *slot -= v;
                    }
                }
                i0 += PACKED_MR;
            }
            j0 += PACKED_NR;
        }
        jc += nc;
    }
    for i in 0..m_full {
        packed_row_tail_sub_assign(a, b, target, i, k, n, n_full);
    }
    for i in m_full..m {
        packed_row_tail_sub_assign(a, b, target, i, k, n, 0);
    }
}

fn packed_row_tail_sub_assign(
    a: &[f64],
    b: &[f64],
    target: &mut [f64],
    i: usize,
    k: usize,
    n: usize,
    j0: usize,
) {
    let a_base = i * k;
    let o_base = i * n;
    for j in j0..n {
        let mut s = 0.0f64;
        for kk in 0..k {
            s += a[a_base + kk] * b[kk * n + j];
        }
        target[o_base + j] -= s;
    }
}

fn packed_gemm_sub_assign_strided_serial(
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    n: usize,
    row_stride: usize,
    target: &mut [f64],
) {
    let m_full = m - m % PACKED_MR;
    let n_full = n - n % PACKED_NR;
    let nc = {
        let cols = (256 * 1024) / (k.max(1) * core::mem::size_of::<f64>());
        (cols / PACKED_NR).max(1) * PACKED_NR
    };
    let mut bp = vec![0.0f64; k * PACKED_NR];
    let mut jc = 0;
    while jc < n_full {
        let jc_end = (jc + nc).min(n_full);
        let mut j0 = jc;
        while j0 < jc_end {
            for kk in 0..k {
                bp[kk * PACKED_NR..kk * PACKED_NR + PACKED_NR]
                    .copy_from_slice(&b[kk * n + j0..kk * n + j0 + PACKED_NR]);
            }
            let mut i0 = 0;
            while i0 < m_full {
                let mut acc = [[0.0f64; PACKED_NR]; PACKED_MR];
                for kk in 0..k {
                    let brow = &bp[kk * PACKED_NR..kk * PACKED_NR + PACKED_NR];
                    for (ii, row) in acc.iter_mut().enumerate() {
                        let av = a[(i0 + ii) * k + kk];
                        for (slot, &bv) in row.iter_mut().zip(brow) {
                            *slot += av * bv;
                        }
                    }
                }
                for (ii, row) in acc.iter().enumerate() {
                    let base = (i0 + ii) * row_stride + j0;
                    for (slot, &v) in target[base..base + PACKED_NR].iter_mut().zip(row) {
                        *slot -= v;
                    }
                }
                i0 += PACKED_MR;
            }
            j0 += PACKED_NR;
        }
        jc += nc;
    }
    for i in 0..m_full {
        let a_base = i * k;
        let o_base = i * row_stride;
        for j in n_full..n {
            let mut s = 0.0f64;
            for kk in 0..k {
                s += a[a_base + kk] * b[kk * n + j];
            }
            target[o_base + j] -= s;
        }
    }
    for i in m_full..m {
        let a_base = i * k;
        let o_base = i * row_stride;
        for j in 0..n {
            let mut s = 0.0f64;
            for kk in 0..k {
                s += a[a_base + kk] * b[kk * n + j];
            }
            target[o_base + j] -= s;
        }
    }
}

fn mat_mul_flat(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    // Square GEMM via the cache-blocked, B-packed kernel (band-parallel over
    // output rows). Bit-identical to the previous ikj kernel — same ascending-k
    // per-element reduction — but keeps B L2-resident instead of re-streaming it
    // from DRAM per row block (~2-4.5x at n>=1024).
    packed_gemm(a, b, n, n, n)
}

fn is_diagonal_matrix_flat(a: &[f64], n: usize) -> bool {
    (0..n).all(|i| (0..n).all(|j| i == j || a[i * n + j] == 0.0))
}

fn is_upper_triangular_flat(a: &[f64], n: usize) -> bool {
    (0..n).all(|i| (0..i).all(|j| a[i * n + j] == 0.0))
}

fn is_lower_triangular_flat(a: &[f64], n: usize) -> bool {
    (0..n).all(|i| ((i + 1)..n).all(|j| a[i * n + j] == 0.0))
}

fn expm_non_finite_compat(a: &[f64], n: usize) -> Option<Vec<f64>> {
    if !a.iter().any(|value| !value.is_finite()) {
        return None;
    }

    if is_diagonal_matrix_flat(a, n) {
        let mut result = vec![0.0; n * n];
        for i in 0..n {
            result[i * n + i] = a[i * n + i].exp();
        }
        return Some(result);
    }

    if is_upper_triangular_flat(a, n) {
        let mut result = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                result[i * n + j] = f64::NAN;
            }
        }
        return Some(result);
    }

    if is_lower_triangular_flat(a, n) {
        let mut result = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                result[i * n + j] = f64::NAN;
            }
        }
        return Some(result);
    }

    Some(vec![f64::NAN; n * n])
}

fn sqrtm_upper_triangular(a: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0; n * n];

    for i in 0..n {
        result[i * n + i] = a[i * n + i].sqrt();
    }

    for span in 1..n {
        for i in 0..(n - span) {
            let j = i + span;
            let mut numerator = a[i * n + j];
            for k in (i + 1)..j {
                numerator -= result[i * n + k] * result[k * n + j];
            }
            let denom = result[i * n + i] + result[j * n + j];
            result[i * n + j] = numerator / denom;
        }
    }

    result
}

fn sqrtm_non_finite_compat(a: &[f64], n: usize) -> Option<Vec<f64>> {
    if !a.iter().any(|value| !value.is_finite()) {
        return None;
    }

    if is_upper_triangular_flat(a, n) {
        return Some(sqrtm_upper_triangular(a, n));
    }

    Some(vec![f64::NAN; n * n])
}

/// Rectangular matrix multiply: A (m×k) × B (k×n) → C (m×n).
fn mat_mul_rect(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    // Rectangular GEMM via the same cache-blocked, B-packed band-parallel kernel
    // as mat_mul_flat. Bit-identical to the previous ikj kernel (same ascending-k
    // per-element reduction); keeps B L2-resident rather than DRAM-restreamed.
    packed_gemm(a, b, m, k, n)
}

fn singular_values_2x2(matrix: [[f64; 2]; 2]) -> Result<[f64; 2], LinAlgError> {
    validate_finite_matrix_2x2(matrix)?;
    let a = matrix[0][0];
    let b = matrix[0][1];
    let c = matrix[1][0];
    let d = matrix[1][1];

    let trace = a.mul_add(a, b.mul_add(b, c.mul_add(c, d * d)));
    let det = a * d - b * c;
    let mut disc = trace.mul_add(trace, -4.0 * det * det);
    if disc < 0.0 && disc.abs() <= f64::EPSILON * 32.0 {
        disc = 0.0;
    }
    if disc < 0.0 || !disc.is_finite() {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "singular-value discriminant became invalid",
        ));
    }

    let sqrt_disc = disc.sqrt();
    let lambda_max = ((trace + sqrt_disc) * 0.5).max(0.0);
    let lambda_min = ((trace - sqrt_disc) * 0.5).max(0.0);
    Ok([lambda_max.sqrt(), lambda_min.sqrt()])
}

fn validate_finite_spectral_matrix_2x2(matrix: [[f64; 2]; 2]) -> Result<(), LinAlgError> {
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(spectral_non_finite_input_error());
    }
    Ok(())
}

fn spectral_non_finite_input_error() -> LinAlgError {
    LinAlgError::NormDetRankPolicyViolation("Array must not contain infs or NaNs")
}

fn real_eigenvalues_2x2(matrix: [[f64; 2]; 2]) -> Result<[f64; 2], LinAlgError> {
    let trace = matrix[0][0] + matrix[1][1];
    let det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    let mut disc = trace.mul_add(trace, -4.0 * det);
    if disc < 0.0 && disc.abs() <= f64::EPSILON * 32.0 {
        disc = 0.0;
    }
    if disc < 0.0 || !disc.is_finite() {
        return Err(LinAlgError::SpectralConvergenceFailed);
    }

    let sqrt_disc = disc.sqrt();
    Ok([(trace + sqrt_disc) * 0.5, (trace - sqrt_disc) * 0.5])
}

pub fn matrix_rank_2x2(matrix: [[f64; 2]; 2], rcond: f64) -> Result<usize, LinAlgError> {
    validate_tolerance_policy(rcond, 0)?;
    let flat = [matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]];
    if flat.iter().any(|value| value.is_nan()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "matrix entries must be finite for rank",
        ));
    }
    if flat.iter().any(|value| value.is_infinite()) {
        return Ok(0);
    }
    let singular_values = svd_mxn(&flat, 2, 2)?;
    let sigma_max = singular_values[0];
    if sigma_max == 0.0 {
        return Ok(0);
    }

    let threshold = sigma_max * rcond;
    Ok(count_matrix_rank(&singular_values, threshold))
}

pub fn matrix_rank_2x2_tol(matrix: [[f64; 2]; 2], tol: Option<f64>) -> Result<usize, LinAlgError> {
    validate_matrix_rank_tol(tol)?;
    let flat = [matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]];
    if flat.iter().any(|value| value.is_nan()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "matrix entries must be finite for rank",
        ));
    }
    if flat.iter().any(|value| value.is_infinite()) {
        return Ok(0);
    }
    let singular_values = svd_mxn(&flat, 2, 2)?;
    let sigma_max = singular_values[0];
    if sigma_max == 0.0 {
        return Ok(0);
    }
    let threshold = resolve_matrix_rank_threshold(sigma_max, 2, 2, tol);
    Ok(count_matrix_rank(&singular_values, threshold))
}

pub fn pinv_2x2(matrix: [[f64; 2]; 2], rcond: f64) -> Result<[[f64; 2]; 2], LinAlgError> {
    let flat = [matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]];
    let pinv = pinv_mxn(&flat, 2, 2, rcond)?;
    Ok([[pinv[0], pinv[1]], [pinv[2], pinv[3]]])
}

pub fn pinv_2x2_with_tolerance_aliases(
    matrix: [[f64; 2]; 2],
    rcond: Option<f64>,
    rtol: Option<Option<f64>>,
) -> Result<[[f64; 2]; 2], LinAlgError> {
    let resolved_rcond = resolve_pinv_tolerance_aliases(rcond, rtol)?;
    pinv_2x2(matrix, resolved_rcond)
}

pub fn vector_norm(values: &[f64], ord: Option<VectorNormOrder>) -> Result<f64, LinAlgError> {
    let order = ord.unwrap_or(VectorNormOrder::Two);
    if values.is_empty() {
        // NumPy reduces over an empty vector as follows:
        //   * ord=-inf  -> min(|x|) over empty has no identity, so it raises.
        //   * negative finite ord -> sum(|x|^ord)=0, then 0^(1/ord) = +inf (1/ord < 0).
        //   * every non-negative order (0, 1, 2, +inf, positive p) -> 0.0.
        return match order {
            VectorNormOrder::NegInf => Err(LinAlgError::NormDetRankPolicyViolation(
                "negative infinity vector norm is undefined for empty inputs",
            )),
            VectorNormOrder::P(p) if p < 0.0 => Ok(f64::INFINITY),
            _ => Ok(0.0),
        };
    }

    let result = match order {
        VectorNormOrder::Zero => values.iter().filter(|v| **v != 0.0).count() as f64,
        VectorNormOrder::One => values.iter().map(|v| v.abs()).sum(),
        VectorNormOrder::Two => values.iter().map(|v| v * v).sum::<f64>().sqrt(),
        VectorNormOrder::Inf => {
            if values.iter().any(|value| value.is_nan()) {
                f64::NAN
            } else {
                values.iter().map(|v| v.abs()).fold(0.0, f64::max)
            }
        }
        VectorNormOrder::NegInf => {
            if values.iter().any(|value| value.is_nan()) {
                f64::NAN
            } else {
                values.iter().map(|v| v.abs()).fold(f64::INFINITY, f64::min)
            }
        }
        VectorNormOrder::P(p) => {
            if (p - 1.0).abs() < f64::EPSILON {
                values.iter().map(|v| v.abs()).sum()
            } else if (p - 2.0).abs() < f64::EPSILON {
                values.iter().map(|v| v * v).sum::<f64>().sqrt()
            } else {
                values
                    .iter()
                    .map(|v| v.abs().powf(p))
                    .sum::<f64>()
                    .powf(1.0 / p)
            }
        }
    };
    Ok(result)
}

pub fn matrix_norm_2x2(
    matrix: [[f64; 2]; 2],
    ord: Option<MatrixNormOrder>,
) -> Result<f64, LinAlgError> {
    let order = ord.unwrap_or(MatrixNormOrder::Fro);
    let flat = [matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]];
    let has_nan = flat.iter().any(|value| value.is_nan());
    let has_inf = flat.iter().any(|value| value.is_infinite());
    let result = match order {
        MatrixNormOrder::Fro => {
            let mut sum_sq = 0.0;
            for row in matrix {
                for value in row {
                    sum_sq += value * value;
                }
            }
            sum_sq.sqrt()
        }
        MatrixNormOrder::One => {
            if has_nan {
                return Ok(f64::NAN);
            }
            let col0 = matrix[0][0].abs() + matrix[1][0].abs();
            let col1 = matrix[0][1].abs() + matrix[1][1].abs();
            col0.max(col1)
        }
        MatrixNormOrder::NegOne => {
            if has_nan {
                return Ok(f64::NAN);
            }
            let col0 = matrix[0][0].abs() + matrix[1][0].abs();
            let col1 = matrix[0][1].abs() + matrix[1][1].abs();
            col0.min(col1)
        }
        MatrixNormOrder::Inf => {
            if has_nan {
                return Ok(f64::NAN);
            }
            let row0 = matrix[0][0].abs() + matrix[0][1].abs();
            let row1 = matrix[1][0].abs() + matrix[1][1].abs();
            row0.max(row1)
        }
        MatrixNormOrder::NegInf => {
            if has_nan {
                return Ok(f64::NAN);
            }
            let row0 = matrix[0][0].abs() + matrix[0][1].abs();
            let row1 = matrix[1][0].abs() + matrix[1][1].abs();
            row0.min(row1)
        }
        MatrixNormOrder::Two => {
            if has_nan {
                return Err(LinAlgError::SvdNonConvergence);
            }
            if has_inf {
                return Ok(f64::NAN);
            }
            singular_values_2x2(matrix)?[0]
        }
        MatrixNormOrder::NegTwo => {
            if has_nan {
                return Err(LinAlgError::SvdNonConvergence);
            }
            if has_inf {
                return Ok(f64::NAN);
            }
            singular_values_2x2(matrix)?[1]
        }
        MatrixNormOrder::Nuclear => {
            if has_nan {
                return Err(LinAlgError::SvdNonConvergence);
            }
            if has_inf {
                return Ok(f64::NAN);
            }
            let singular_values = singular_values_2x2(matrix)?;
            singular_values[0] + singular_values[1]
        }
    };
    Ok(result)
}

pub fn eigvals_2x2(matrix: [[f64; 2]; 2], converged: bool) -> Result<[f64; 2], LinAlgError> {
    if !converged {
        return Err(LinAlgError::SpectralConvergenceFailed);
    }
    validate_finite_spectral_matrix_2x2(matrix)?;
    real_eigenvalues_2x2(matrix)
}

pub fn eigh_2x2(
    matrix: [[f64; 2]; 2],
    uplo: &str,
    converged: bool,
) -> Result<([f64; 2], [[f64; 2]; 2]), LinAlgError> {
    validate_spectral_branch(uplo, converged)?;
    validate_finite_spectral_matrix_2x2(matrix)?;

    let symmetric = match uplo {
        "L" => [[matrix[0][0], matrix[1][0]], [matrix[1][0], matrix[1][1]]],
        "U" => [[matrix[0][0], matrix[0][1]], [matrix[0][1], matrix[1][1]]],
        _ => return Err(LinAlgError::SpectralConvergenceFailed),
    };

    let mut eigenvalues = real_eigenvalues_2x2(symmetric)?;
    if eigenvalues[0] > eigenvalues[1] {
        eigenvalues.swap(0, 1);
    }

    let mut eigenvectors = [[0.0_f64; 2]; 2];
    let a = symmetric[0][0];
    let b = symmetric[0][1];
    let d = symmetric[1][1];

    let lambda0 = eigenvalues[0];
    let mut v0 = if b.abs() > f64::EPSILON {
        [b, lambda0 - a]
    } else if (a - lambda0).abs() <= (d - lambda0).abs() {
        [1.0, 0.0]
    } else {
        [0.0, 1.0]
    };
    let v0_norm = v0[0].hypot(v0[1]);
    if !v0_norm.is_finite() || v0_norm <= f64::EPSILON {
        return Err(LinAlgError::SpectralConvergenceFailed);
    }
    v0[0] /= v0_norm;
    v0[1] /= v0_norm;

    let v1 = [-v0[1], v0[0]];
    eigenvectors[0][0] = v0[0];
    eigenvectors[1][0] = v0[1];
    eigenvectors[0][1] = v1[0];
    eigenvectors[1][1] = v1[1];

    Ok((eigenvalues, eigenvectors))
}

fn validate_cholesky_uplo(uplo: &str) -> Result<(), LinAlgError> {
    if uplo == "L" || uplo == "U" {
        Ok(())
    } else {
        Err(LinAlgError::CholeskyContractViolation(
            "cholesky uplo must be L or U",
        ))
    }
}

pub fn cholesky_2x2(matrix: [[f64; 2]; 2], uplo: &str) -> Result<[[f64; 2]; 2], LinAlgError> {
    validate_cholesky_uplo(uplo)?;

    let (a, b, d) = match uplo {
        "L" => (matrix[0][0], matrix[1][0], matrix[1][1]),
        "U" => (matrix[0][0], matrix[0][1], matrix[1][1]),
        _ => return Err(LinAlgError::CholeskyContractViolation("invalid uplo")),
    };

    if !a.is_finite() || !b.is_finite() || !d.is_finite() {
        return Err(LinAlgError::CholeskyContractViolation(
            "cholesky requires finite selected-triangle entries",
        ));
    }
    if a <= 0.0 {
        return Err(LinAlgError::CholeskyContractViolation(
            "cholesky requires positive leading principal minor",
        ));
    }

    let l11 = a.sqrt();
    let l21 = b / l11;
    let schur = d - l21 * l21;
    if !schur.is_finite() || schur <= 0.0 {
        return Err(LinAlgError::CholeskyContractViolation(
            "matrix is not positive definite on selected triangle",
        ));
    }
    let l22 = schur.sqrt();

    let result = match uplo {
        "L" => [[l11, 0.0], [l21, l22]],
        "U" => [[l11, l21], [0.0, l22]],
        _ => return Err(LinAlgError::CholeskyContractViolation("invalid uplo")),
    };
    Ok(result)
}

pub fn validate_cholesky_diagonal(diagonal: &[f64]) -> Result<(), LinAlgError> {
    if diagonal.is_empty() {
        return Err(LinAlgError::CholeskyContractViolation(
            "cholesky diagonal cannot be empty",
        ));
    }
    if diagonal
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(LinAlgError::CholeskyContractViolation(
            "cholesky requires strictly positive finite diagonal",
        ));
    }
    Ok(())
}

pub fn qr_output_shapes(shape: &[usize], mode: QrMode) -> Result<QrOutputShapes, LinAlgError> {
    let (m, n) = validate_matrix_shape(shape)?;
    let k = m.min(n);

    let output = match mode {
        QrMode::Reduced => QrOutputShapes {
            q_shape: Some(vec![m, k]),
            r_shape: vec![k, n],
        },
        QrMode::Complete => QrOutputShapes {
            q_shape: Some(vec![m, m]),
            r_shape: vec![m, n],
        },
        QrMode::R => QrOutputShapes {
            q_shape: None,
            r_shape: vec![k, n],
        },
        QrMode::Raw => QrOutputShapes {
            q_shape: Some(vec![n, m]),
            r_shape: vec![k],
        },
    };
    Ok(output)
}

pub fn qr_2x2(matrix: [[f64; 2]; 2], mode: QrMode) -> Result<Qr2x2Result, LinAlgError> {
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(LinAlgError::QrModeInvalid);
    }
    if mode == QrMode::Raw {
        return Err(LinAlgError::QrModeInvalid);
    }

    let c1 = [matrix[0][0], matrix[1][0]];
    let c2 = [matrix[0][1], matrix[1][1]];

    let mut q1 = [1.0_f64, 0.0_f64];
    let r11 = c1[0].hypot(c1[1]);
    if r11 > 0.0 {
        q1 = [c1[0] / r11, c1[1] / r11];
    }

    let r12 = q1[0].mul_add(c2[0], q1[1] * c2[1]);
    let u2 = [c2[0] - r12 * q1[0], c2[1] - r12 * q1[1]];
    let mut q2 = [-q1[1], q1[0]];
    let mut r22 = u2[0].hypot(u2[1]);
    if r22 > 0.0 {
        q2 = [u2[0] / r22, u2[1] / r22];
    } else {
        r22 = q2[0].mul_add(c2[0], q2[1] * c2[1]);
    }

    if r22 < 0.0 {
        q2[0] = -q2[0];
        q2[1] = -q2[1];
        r22 = -r22;
    }

    let q = [[q1[0], q2[0]], [q1[1], q2[1]]];
    let r = [[r11, r12], [0.0, r22]];

    let q_out = match mode {
        QrMode::Reduced | QrMode::Complete => Some(q),
        QrMode::R => None,
        QrMode::Raw => return Err(LinAlgError::QrModeInvalid),
    };
    Ok(Qr2x2Result { q: q_out, r })
}

pub fn svd_2x2(matrix: [[f64; 2]; 2], converged: bool) -> Result<Svd2x2Result, LinAlgError> {
    if !converged {
        return Err(LinAlgError::SvdNonConvergence);
    }
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(LinAlgError::SvdNonConvergence);
    }

    let a = matrix[0][0];
    let b = matrix[0][1];
    let c = matrix[1][0];
    let d = matrix[1][1];

    // Right-singular vectors come from eigendecomposition of A^T A.
    let m00 = a.mul_add(a, c * c);
    let m01 = a.mul_add(b, c * d);
    let m11 = b.mul_add(b, d * d);
    let trace = m00 + m11;
    let det = m00 * m11 - m01 * m01;
    let mut disc = trace.mul_add(trace, -4.0 * det);
    if disc < 0.0 && disc.abs() <= f64::EPSILON * 32.0 {
        disc = 0.0;
    }
    if disc < 0.0 || !disc.is_finite() {
        return Err(LinAlgError::SvdNonConvergence);
    }

    let sqrt_disc = disc.sqrt();
    let lambda_1 = ((trace + sqrt_disc) * 0.5).max(0.0);
    let lambda_2 = ((trace - sqrt_disc) * 0.5).max(0.0);
    let sigma_1 = lambda_1.sqrt();
    let sigma_2 = lambda_2.sqrt();
    if !sigma_1.is_finite() || !sigma_2.is_finite() {
        return Err(LinAlgError::SvdNonConvergence);
    }

    let mut v1 = if m01.abs() > f64::EPSILON {
        [m01, lambda_1 - m00]
    } else if m00 >= m11 {
        [1.0, 0.0]
    } else {
        [0.0, 1.0]
    };
    let v1_norm = v1[0].hypot(v1[1]);
    if !v1_norm.is_finite() || v1_norm <= f64::EPSILON {
        return Err(LinAlgError::SvdNonConvergence);
    }
    v1[0] /= v1_norm;
    v1[1] /= v1_norm;
    let v2 = [-v1[1], v1[0]];
    let vectors = [v1, v2];

    let singular_values = [sigma_1, sigma_2];
    let mut u_cols = [[0.0_f64; 2]; 2];
    if sigma_1 <= 0.0 {
        u_cols[0] = [1.0, 0.0];
        u_cols[1] = [0.0, 1.0];
    } else {
        for idx in 0..2 {
            let sigma = singular_values[idx];
            let v = vectors[idx];

            let mut u = if sigma > 0.0 {
                [
                    a.mul_add(v[0], b * v[1]) / sigma,
                    c.mul_add(v[0], d * v[1]) / sigma,
                ]
            } else {
                [-u_cols[0][1], u_cols[0][0]]
            };

            if idx == 1 && sigma > 0.0 {
                let proj = u_cols[0][0].mul_add(u[0], u_cols[0][1] * u[1]);
                u[0] -= proj * u_cols[0][0];
                u[1] -= proj * u_cols[0][1];
            }

            let norm = u[0].hypot(u[1]);
            if !norm.is_finite() || norm <= 0.0 {
                if idx == 0 {
                    return Err(LinAlgError::SvdNonConvergence);
                }
                u = [-u_cols[0][1], u_cols[0][0]];
            } else {
                u[0] /= norm;
                u[1] /= norm;
            }
            u_cols[idx] = u;
        }
    }

    let u = [[u_cols[0][0], u_cols[1][0]], [u_cols[0][1], u_cols[1][1]]];
    let vt = [
        [vectors[0][0], vectors[0][1]],
        [vectors[1][0], vectors[1][1]],
    ];

    Ok(Svd2x2Result {
        u,
        singular_values,
        vt,
    })
}

pub fn svd_output_shapes(
    shape: &[usize],
    full_matrices: bool,
    converged: bool,
) -> Result<SvdOutputShapes, LinAlgError> {
    if !converged {
        return Err(LinAlgError::SvdNonConvergence);
    }

    let (m, n) = validate_matrix_shape(shape)?;
    let k = m.min(n);

    let (u_shape, vh_shape) = if full_matrices {
        (vec![m, m], vec![n, n])
    } else {
        (vec![m, k], vec![k, n])
    };

    Ok(SvdOutputShapes {
        u_shape,
        s_shape: vec![k],
        vh_shape,
    })
}

pub fn validate_spectral_branch(uplo: &str, converged: bool) -> Result<(), LinAlgError> {
    if uplo != "L" && uplo != "U" {
        return Err(LinAlgError::SpectralConvergenceFailed);
    }
    if !converged {
        return Err(LinAlgError::SpectralConvergenceFailed);
    }
    Ok(())
}

pub fn lstsq_output_shapes(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
) -> Result<LstsqOutputShapes, LinAlgError> {
    if lhs_shape.len() != 2 {
        return Err(LinAlgError::LstsqTupleContractViolation(
            "lhs must be 2D for lstsq",
        ));
    }
    let m = lhs_shape[0];
    let n = lhs_shape[1];
    if m == 0 || n == 0 {
        return Err(LinAlgError::LstsqTupleContractViolation(
            "lhs dimensions must be non-zero",
        ));
    }

    let rhs_cols = match rhs_shape {
        [rows] => {
            if *rows != m {
                return Err(LinAlgError::LstsqTupleContractViolation(
                    "rhs rows must equal lhs rows",
                ));
            }
            1usize
        }
        [rows, cols] => {
            if *rows != m || *cols == 0 {
                return Err(LinAlgError::LstsqTupleContractViolation(
                    "rhs shape must be (m,) or (m,k>0)",
                ));
            }
            *cols
        }
        _ => {
            return Err(LinAlgError::LstsqTupleContractViolation(
                "rhs must be 1D or 2D for lstsq",
            ));
        }
    };

    let x_shape = if rhs_shape.len() == 1 {
        vec![n]
    } else {
        vec![n, rhs_cols]
    };
    let residuals_shape = if m > n { vec![rhs_cols] } else { Vec::new() };

    Ok(LstsqOutputShapes {
        x_shape,
        residuals_shape,
        rank_upper_bound: m.min(n),
        singular_values_shape: vec![m.min(n)],
    })
}

pub fn lstsq_2x2(
    lhs: [[f64; 2]; 2],
    rhs: [f64; 2],
    rcond: f64,
) -> Result<Lstsq2x2Result, LinAlgError> {
    if !rcond.is_finite() || rcond < 0.0 {
        return Err(LinAlgError::LstsqTupleContractViolation(
            "rcond must be finite and >= 0 for lstsq_2x2",
        ));
    }
    if lhs.iter().flatten().any(|value| !value.is_finite()) {
        return Err(LinAlgError::LstsqTupleContractViolation(
            "lhs entries must be finite for lstsq_2x2",
        ));
    }
    if rhs.iter().any(|value| !value.is_finite()) {
        return Err(LinAlgError::LstsqTupleContractViolation(
            "rhs entries must be finite for lstsq_2x2",
        ));
    }

    let pinv = pinv_2x2(lhs, rcond)
        .map_err(|_| LinAlgError::LstsqTupleContractViolation("lstsq_2x2 pinv route failed"))?;
    let solution = [
        pinv[0][0].mul_add(rhs[0], pinv[0][1] * rhs[1]),
        pinv[1][0].mul_add(rhs[0], pinv[1][1] * rhs[1]),
    ];

    let residual = [
        lhs[0][0].mul_add(solution[0], lhs[0][1] * solution[1]) - rhs[0],
        lhs[1][0].mul_add(solution[0], lhs[1][1] * solution[1]) - rhs[1],
    ];
    let residual_sum_squares = residual[0].mul_add(residual[0], residual[1] * residual[1]);

    let rank = matrix_rank_2x2(lhs, rcond).map_err(|_| {
        LinAlgError::LstsqTupleContractViolation("lstsq_2x2 rank evaluation failed")
    })?;
    let singular_values = singular_values_2x2(lhs).map_err(|_| {
        LinAlgError::LstsqTupleContractViolation("lstsq_2x2 singular-value evaluation failed")
    })?;

    Ok(Lstsq2x2Result {
        solution,
        residual_sum_squares,
        rank,
        singular_values,
    })
}

pub fn validate_tolerance_policy(rcond: f64, search_depth: usize) -> Result<(), LinAlgError> {
    if !rcond.is_finite() || rcond < 0.0 {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "rcond must be finite and >= 0",
        ));
    }
    if search_depth > MAX_TOLERANCE_SEARCH_DEPTH {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "search depth exceeded tolerance budget",
        ));
    }
    Ok(())
}

pub fn validate_backend_bridge(
    backend_supported: bool,
    revalidation_attempts: usize,
) -> Result<(), LinAlgError> {
    if !backend_supported {
        return Err(LinAlgError::BackendBridgeInvalid(
            "backend bridge is unsupported",
        ));
    }
    if revalidation_attempts > MAX_BACKEND_REVALIDATION_ATTEMPTS {
        return Err(LinAlgError::BackendBridgeInvalid(
            "backend bridge revalidation budget exceeded",
        ));
    }
    Ok(())
}

pub fn validate_policy_metadata(mode: &str, class: &str) -> Result<(), LinAlgError> {
    let mode = mode.trim();
    let class = class.trim();
    let known_mode = mode == "strict" || mode == "hardened";
    let known_class = class == "known_compatible"
        || class == "known_compatible_low_risk"
        || class == "known_compatible_high_risk"
        || class == "known_incompatible"
        || class == "known_incompatible_semantics"
        || class == "unknown"
        || class == "unknown_semantics";

    if !known_mode || !known_class {
        return Err(LinAlgError::PolicyUnknownMetadata(
            "unknown mode/class metadata rejected fail-closed",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Batched linalg: stacked matrix support (NumPy-style leading batch dims)
// ---------------------------------------------------------------------------

/// Compute the total number of matrices in the batch (product of leading dims).
fn batch_count(shape: &[usize]) -> Result<usize, LinAlgError> {
    if shape.len() <= 2 {
        Ok(1)
    } else {
        fnp_ndarray::element_count(&shape[..shape.len() - 2])
            .map_err(|_| LinAlgError::ShapeContractViolation("product overflow"))
    }
}

/// Validate that shape has at least 2 dimensions and extract (batch_count, m, n).
fn parse_batched_shape(shape: &[usize]) -> Result<(usize, usize, usize), LinAlgError> {
    if shape.len() < 2 {
        return Err(LinAlgError::ShapeContractViolation(
            "batched linalg: input must have at least 2 dimensions",
        ));
    }
    let m = shape[shape.len() - 2];
    let n = shape[shape.len() - 1];
    let batch = batch_count(shape)?;
    Ok((batch, m, n))
}

/// Validate that shape represents stacked square matrices.
fn parse_batched_square(shape: &[usize]) -> Result<(usize, usize), LinAlgError> {
    let (batch, m, n) = parse_batched_shape(shape)?;
    if m != n || m == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "batched linalg: last two dimensions must be equal and non-zero",
        ));
    }
    Ok((batch, n))
}

/// Minimum *total* size (sum of scalar f64 elements across the whole batch)
/// above which the independent per-lane kernels are worth dispatching across the
/// rayon pool. The earlier gate keyed on *per-lane* size (≈ n ≥ 128), on the
/// theory that the allocator would contend across threads for small matrices —
/// but the system allocator's per-thread arenas make concurrent small allocs
/// cheap, and a large batch of SMALL matrices is still a large pile of fully
/// independent O(n³) work. Measured serial→parallel: 2.5x @ n=4 (batch 524 288),
/// 9.3x @ n=8, 14x @ n=16, 20.4x @ n=32 — all of which the per-lane gate kept
/// serial. Gating on total work parallelizes those (rayon work-stealing amortizes
/// per-task overhead over the whole batch) while still keeping a 2-element batch
/// of tiny matrices serial.
const BATCH_PARALLEL_MIN_TOTAL_ELEMS: usize = 1 << 14;

/// Decide whether a batch of `batch` matrices, each `per_lane_elems` scalars,
/// should run across the rayon pool: at least two lanes, at least two worker
/// threads, and enough *total* work across the batch to amortize scheduling.
/// This is strictly more permissive than the old per-lane gate (since
/// `batch ≥ 2`, `batch·per_lane ≥ 2·per_lane`), so no previously-parallel case
/// regresses.
#[inline]
fn batch_should_parallelize(batch: usize, per_lane_elems: usize) -> bool {
    batch >= 2
        && rayon::current_num_threads() >= 2
        && batch.saturating_mul(per_lane_elems) >= BATCH_PARALLEL_MIN_TOTAL_ELEMS
}

/// Run an independent per-lane kernel `f` over `0..batch`, collecting results in
/// strict lane order. The lanes are fully independent (each touches a disjoint
/// sub-matrix and shares no mutable state), so the parallel path produces
/// bit-for-bit identical output to the serial path. Error behavior matches the
/// serial loop exactly: the parallel branch first collects every lane's
/// `Result` in order, then folds them serially so the returned error is the one
/// from the lowest-indexed failing lane — identical to a serial `?`-loop.
fn batch_map_lanes<T, F>(batch: usize, per_lane_elems: usize, f: F) -> Result<Vec<T>, LinAlgError>
where
    T: Send,
    F: Fn(usize) -> Result<T, LinAlgError> + Send + Sync,
{
    if batch_should_parallelize(batch, per_lane_elems) {
        let lanes: Vec<Result<T, LinAlgError>> = (0..batch).into_par_iter().map(f).collect();
        lanes.into_iter().collect()
    } else {
        (0..batch).map(f).collect()
    }
}

/// Batched matrix inversion: inv on (..., n, n) → (..., n, n).
pub fn batch_inv(data: &[f64], shape: &[usize]) -> Result<Vec<f64>, LinAlgError> {
    let (batch, n) = parse_batched_square(shape)?;
    let mat_size = n * n;
    if Some(data.len()) != batch.checked_mul(mat_size) {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_inv: data length does not match shape",
        ));
    }
    // Tiny matrices: write each inverse DIRECTLY into the flat output through
    // per-thread reusable (lu, perm) scratch — no per-lane lu/perm/eye/result alloc,
    // no Vec<Vec>, no flatten. inv is n*n (m=n RHS columns), so like matrix-RHS solve
    // its per-lane O(n^3) compute overtakes the alloc savings beyond small n; gate to
    // the regime where the win is clear. Byte-identical to per-lane inv_nxn.
    // NO-SHIP 2026-06-21 (BlackThrush): raising this gate to 128 to also direct-write
    // n=16..64 did NOT fix the moderate-batch loss — a SERIAL A/B (RAYON=1, numpy loop
    // already serial) is a stable 2.3-2.5x at n=16/32/64, i.e. the native inv_nxn
    // per-lane kernel is ~2.3x slower than LAPACK getri (KERNEL wall, not alloc; same
    // class as batch_cholesky). The parallel "wins" were load noise. Real fix = SIMD/
    // blocked inv kernel (bit-exactness risk) or delegate moderate-batch to numpy.
    const INV_SCRATCH_MAX_N: usize = 16;
    if n < INV_SCRATCH_MAX_N {
        let mut result = vec![0.0f64; batch * mat_size];
        if batch_should_parallelize(batch, mat_size) {
            use std::sync::Mutex;
            let first_err: Mutex<Option<(usize, LinAlgError)>> = Mutex::new(None);
            result.par_chunks_mut(mat_size).enumerate().for_each_init(
                || (vec![0.0f64; mat_size], vec![0usize; n]),
                |(lu, perm), (idx, out_chunk)| {
                    let a_sub = &data[idx * mat_size..(idx + 1) * mat_size];
                    if let Err(e) = inv_nxn_into_out(a_sub, n, lu, perm, out_chunk) {
                        let mut slot = first_err.lock().unwrap();
                        let replace = match slot.as_ref() {
                            None => true,
                            Some((i, _)) => idx < *i,
                        };
                        if replace {
                            *slot = Some((idx, e));
                        }
                    }
                },
            );
            if let Some((_, e)) = first_err.into_inner().unwrap() {
                return Err(e);
            }
        } else {
            let mut lu = vec![0.0f64; mat_size];
            let mut perm = vec![0usize; n];
            for (idx, out_chunk) in result.chunks_mut(mat_size).enumerate() {
                let a_sub = &data[idx * mat_size..(idx + 1) * mat_size];
                inv_nxn_into_out(a_sub, n, &mut lu, &mut perm, out_chunk)?;
            }
        }
        return Ok(result);
    }
    let lanes = batch_map_lanes(batch, mat_size, |b| {
        inv_nxn(&data[b * mat_size..(b + 1) * mat_size], n)
    })?;
    let mut result = Vec::with_capacity(batch * mat_size);
    for lane in &lanes {
        result.extend_from_slice(lane);
    }
    Ok(result)
}

/// Batched determinant: det on (..., n, n) → (...).
pub fn batch_det(data: &[f64], shape: &[usize]) -> Result<Vec<f64>, LinAlgError> {
    let (batch, n) = parse_batched_square(shape)?;
    let mat_size = n * n;
    if Some(data.len()) != batch.checked_mul(mat_size) {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_det: data length does not match shape",
        ));
    }
    // Small stacked determinants are dominated by per-lane LU/perm allocation in
    // realistic np.linalg.det/slogdet batches. Reuse caller-owned scratch per
    // worker and write scalar outputs directly; arithmetic and singular handling
    // remain the same unblocked LU path used by det_nxn for n < LU_BLOCK_MIN.
    const DET_SCRATCH_MAX_N: usize = 16;
    if n < DET_SCRATCH_MAX_N {
        let mut result = vec![0.0f64; batch];
        if batch_should_parallelize(batch, mat_size) {
            use std::sync::Mutex;
            let first_err: Mutex<Option<(usize, LinAlgError)>> = Mutex::new(None);
            result.par_iter_mut().enumerate().for_each_init(
                || (vec![0.0f64; mat_size], vec![0usize; n]),
                |(lu, perm), (idx, out)| {
                    let a_sub = &data[idx * mat_size..(idx + 1) * mat_size];
                    match det_nxn_unblocked_with_scratch(a_sub, n, lu, perm) {
                        Ok(det) => *out = det,
                        Err(e) => {
                            let mut slot = first_err.lock().unwrap();
                            let replace = match slot.as_ref() {
                                None => true,
                                Some((i, _)) => idx < *i,
                            };
                            if replace {
                                *slot = Some((idx, e));
                            }
                        }
                    }
                },
            );
            if let Some((_, e)) = first_err.into_inner().unwrap() {
                return Err(e);
            }
        } else {
            let mut lu = vec![0.0f64; mat_size];
            let mut perm = vec![0usize; n];
            for (idx, out) in result.iter_mut().enumerate() {
                let a_sub = &data[idx * mat_size..(idx + 1) * mat_size];
                *out = det_nxn_unblocked_with_scratch(a_sub, n, &mut lu, &mut perm)?;
            }
        }
        return Ok(result);
    }
    batch_map_lanes(batch, mat_size, |b| {
        det_nxn(&data[b * mat_size..(b + 1) * mat_size], n)
    })
}

/// Batched slogdet: slogdet on (..., n, n) → (signs: ..., log_abs_dets: ...).
pub fn batch_slogdet(data: &[f64], shape: &[usize]) -> Result<(Vec<f64>, Vec<f64>), LinAlgError> {
    let (batch, n) = parse_batched_square(shape)?;
    let mat_size = n * n;
    if Some(data.len()) != batch.checked_mul(mat_size) {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_slogdet: data length does not match shape",
        ));
    }
    const DET_SCRATCH_MAX_N: usize = 16;
    if n < DET_SCRATCH_MAX_N {
        let mut signs = vec![0.0f64; batch];
        let mut logabsdets = vec![0.0f64; batch];
        if batch_should_parallelize(batch, mat_size) {
            use std::sync::Mutex;
            let first_err: Mutex<Option<(usize, LinAlgError)>> = Mutex::new(None);
            signs
                .par_iter_mut()
                .zip(logabsdets.par_iter_mut())
                .enumerate()
                .for_each_init(
                    || (vec![0.0f64; mat_size], vec![0usize; n]),
                    |(lu, perm), (idx, (sign_out, log_out))| {
                        let a_sub = &data[idx * mat_size..(idx + 1) * mat_size];
                        match slogdet_nxn_unblocked_with_scratch(a_sub, n, lu, perm) {
                            Ok((sign, logabsdet)) => {
                                *sign_out = sign;
                                *log_out = logabsdet;
                            }
                            Err(e) => {
                                let mut slot = first_err.lock().unwrap();
                                let replace = match slot.as_ref() {
                                    None => true,
                                    Some((i, _)) => idx < *i,
                                };
                                if replace {
                                    *slot = Some((idx, e));
                                }
                            }
                        }
                    },
                );
            if let Some((_, e)) = first_err.into_inner().unwrap() {
                return Err(e);
            }
        } else {
            let mut lu = vec![0.0f64; mat_size];
            let mut perm = vec![0usize; n];
            for idx in 0..batch {
                let a_sub = &data[idx * mat_size..(idx + 1) * mat_size];
                let (sign, logabsdet) =
                    slogdet_nxn_unblocked_with_scratch(a_sub, n, &mut lu, &mut perm)?;
                signs[idx] = sign;
                logabsdets[idx] = logabsdet;
            }
        }
        return Ok((signs, logabsdets));
    }
    let lanes = batch_map_lanes(batch, mat_size, |b| {
        slogdet_nxn(&data[b * mat_size..(b + 1) * mat_size], n)
    })?;
    let mut signs = Vec::with_capacity(batch);
    let mut logabsdets = Vec::with_capacity(batch);
    for (sign, logabsdet) in lanes {
        signs.push(sign);
        logabsdets.push(logabsdet);
    }
    Ok((signs, logabsdets))
}

/// Batched solve: solve on A(..., n, n) × b(..., n) → x(..., n).
pub fn batch_solve(
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
    vector_rhs: bool,
) -> Result<Vec<f64>, LinAlgError> {
    let (a_batch, n) = parse_batched_square(a_shape)?;
    if b_shape.is_empty() {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_solve: b must have at least 1 dimension",
        ));
    }

    let (b_batch, rhs_cols, rhs_width) = if vector_rhs {
        if b_shape[b_shape.len() - 1] != n {
            return Err(LinAlgError::ShapeContractViolation(
                "batch_solve: vector b length must match n",
            ));
        }
        let b_batch = if b_shape.len() <= 1 {
            1
        } else {
            fnp_ndarray::element_count(&b_shape[..b_shape.len() - 1])
                .map_err(|_| LinAlgError::ShapeContractViolation("product overflow"))?
        };
        (b_batch, 1, n)
    } else {
        if b_shape.len() < 2 || b_shape[b_shape.len() - 2] != n {
            return Err(LinAlgError::ShapeContractViolation(
                "batch_solve: matrix b penultimate dimension must match n",
            ));
        }
        let rhs_cols = b_shape[b_shape.len() - 1];
        let rhs_width = n
            .checked_mul(rhs_cols)
            .ok_or(LinAlgError::ShapeContractViolation(
                "batch_solve: rhs width overflow",
            ))?;
        let b_batch = if b_shape.len() <= 2 {
            1
        } else {
            fnp_ndarray::element_count(&b_shape[..b_shape.len() - 2])
                .map_err(|_| LinAlgError::ShapeContractViolation("product overflow"))?
        };
        (b_batch, rhs_cols, rhs_width)
    };

    let batch = a_batch.max(b_batch);
    if (a_batch != 1 && a_batch != batch) || (b_batch != 1 && b_batch != batch) {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_solve: batch dimensions are not broadcastable",
        ));
    }
    let mat_size = n * n;
    if Some(a.len()) != a_batch.checked_mul(mat_size) {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_solve: A data length does not match shape",
        ));
    }
    if Some(b.len()) != b_batch.checked_mul(rhs_width) {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_solve: B data length does not match shape",
        ));
    }

    // Matrix-RHS does m columns of substitution per lane, so its per-lane compute
    // (~O(n²·m)) overtakes the constant per-lane alloc cost at a much smaller n than
    // vector-RHS: same-worker A/B shows matrix-RHS 3.3x @ n=3, 1.3-1.6x @ n=8, but
    // only ~neutral (0.9-1.1x, noise) at n>=16. Gate the matrix scratch path to the
    // small-n regime where it clearly wins; vector-RHS stays unrestricted (neutral-to
    // -positive all the way to LU_BLOCK_MIN since its alloc savings never go negative).
    const MATRIX_RHS_SCRATCH_MAX_N: usize = 16;
    if n < LU_BLOCK_MIN && (vector_rhs || n < MATRIX_RHS_SCRATCH_MAX_N) {
        // Small/medium batched solve (vector OR matrix RHS): write each lane's solution
        // DIRECTLY into the flat output through per-thread reusable (lu, perm) scratch —
        // no per-lane Vec, no Vec<Vec> collection, no flatten, ZERO per-lane allocation
        // (the regime where the per-lane lu+perm+X allocs dwarf the O(n^3) compute; the
        // matrix-RHS X is n*rhs_cols, so eliminating it helps even more). The LU +
        // forward/back is byte-for-byte identical to per-lane solve_nxn[_multi]
        // (the *_into_out helpers share lu_factor_unblocked_into and the same
        // substitution; n < LU_BLOCK_MIN keeps the unblocked TRSM). The lowest-indexed
        // failing lane's error is returned, matching the batch_map_lanes ordering.
        let mut result = vec![0.0f64; batch * rhs_width];
        let lane_inputs = |idx: usize| {
            let a_idx = if a_batch == 1 { 0 } else { idx };
            let b_idx = if b_batch == 1 { 0 } else { idx };
            (
                &a[a_idx * mat_size..(a_idx + 1) * mat_size],
                &b[b_idx * rhs_width..(b_idx + 1) * rhs_width],
            )
        };
        let solve_into = |a_sub: &[f64],
                          b_sub: &[f64],
                          lu: &mut [f64],
                          perm: &mut [usize],
                          out: &mut [f64]|
         -> Result<(), LinAlgError> {
            if vector_rhs {
                solve_nxn_into_out(a_sub, b_sub, n, lu, perm, out)
            } else {
                solve_nxn_multi_into_out(a_sub, b_sub, n, rhs_cols, lu, perm, out)
            }
        };
        if batch_should_parallelize(batch, mat_size + rhs_width) {
            use std::sync::Mutex;
            let first_err: Mutex<Option<(usize, LinAlgError)>> = Mutex::new(None);
            result.par_chunks_mut(rhs_width).enumerate().for_each_init(
                || (vec![0.0f64; mat_size], vec![0usize; n]),
                |(lu, perm), (idx, out_chunk)| {
                    let (a_sub, b_sub) = lane_inputs(idx);
                    if let Err(e) = solve_into(a_sub, b_sub, lu, perm, out_chunk) {
                        let mut slot = first_err.lock().unwrap();
                        let replace = match slot.as_ref() {
                            None => true,
                            Some((i, _)) => idx < *i,
                        };
                        if replace {
                            *slot = Some((idx, e));
                        }
                    }
                },
            );
            if let Some((_, e)) = first_err.into_inner().unwrap() {
                return Err(e);
            }
        } else {
            let mut lu = vec![0.0f64; mat_size];
            let mut perm = vec![0usize; n];
            for (idx, out_chunk) in result.chunks_mut(rhs_width).enumerate() {
                let (a_sub, b_sub) = lane_inputs(idx);
                solve_into(a_sub, b_sub, &mut lu, &mut perm, out_chunk)?;
            }
        }
        return Ok(result);
    }

    let lanes = batch_map_lanes(batch, mat_size + rhs_width, |idx| {
        let a_idx = if a_batch == 1 { 0 } else { idx };
        let b_idx = if b_batch == 1 { 0 } else { idx };
        let a_sub = &a[a_idx * mat_size..(a_idx + 1) * mat_size];
        let b_sub = &b[b_idx * rhs_width..(b_idx + 1) * rhs_width];
        if vector_rhs {
            solve_nxn(a_sub, b_sub, n)
        } else {
            solve_nxn_multi(a_sub, b_sub, n, rhs_cols)
        }
    })?;
    let mut result = Vec::with_capacity(batch * rhs_width);
    for x in &lanes {
        result.extend_from_slice(x);
    }
    Ok(result)
}

/// Batched eigenvalues: eigvalsh on (..., n, n) → (..., n).
pub fn batch_eigvalsh(data: &[f64], shape: &[usize]) -> Result<Vec<f64>, LinAlgError> {
    let (batch, n) = parse_batched_square(shape)?;
    let mat_size = n * n;
    if Some(data.len()) != batch.checked_mul(mat_size) {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_eigvalsh: data length does not match shape",
        ));
    }
    let lanes = batch_map_lanes(batch, mat_size, |b| {
        eigvalsh_nxn(&data[b * mat_size..(b + 1) * mat_size], n)
    })?;
    let mut result = Vec::with_capacity(batch * n);
    for eigvals in &lanes {
        result.extend_from_slice(eigvals);
    }
    Ok(result)
}

/// Batched eigh: eigh on (..., n, n) → (eigenvalues (..., n), eigenvectors (..., n, n)).
pub fn batch_eigh(data: &[f64], shape: &[usize]) -> Result<(Vec<f64>, Vec<f64>), LinAlgError> {
    let (batch, n) = parse_batched_square(shape)?;
    let mat_size = n * n;
    if Some(data.len()) != batch.checked_mul(mat_size) {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_eigh: data length does not match shape",
        ));
    }
    let lanes = batch_map_lanes(batch, mat_size, |b| {
        eigh_nxn(&data[b * mat_size..(b + 1) * mat_size], n)
    })?;
    let mut eigenvalues = Vec::with_capacity(batch * n);
    let mut eigenvectors = Vec::with_capacity(batch * mat_size);
    for (vals, vecs) in &lanes {
        eigenvalues.extend_from_slice(vals);
        eigenvectors.extend_from_slice(vecs);
    }
    Ok((eigenvalues, eigenvectors))
}

/// Batched eig: eig on (..., n, n) → eigenvalues as interleaved (re,im) (..., n*2).
pub fn batch_eig(data: &[f64], shape: &[usize]) -> Result<Vec<f64>, LinAlgError> {
    let (batch, n) = parse_batched_square(shape)?;
    let mat_size = n * n;
    if Some(data.len()) != batch.checked_mul(mat_size) {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_eig: data length does not match shape",
        ));
    }
    let lanes = batch_map_lanes(batch, mat_size, |b| {
        eig_nxn(&data[b * mat_size..(b + 1) * mat_size], n)
    })?;
    let mut result = Vec::with_capacity(batch * n * 2);
    for eigvals in &lanes {
        result.extend_from_slice(eigvals);
    }
    Ok(result)
}

/// Batched SVD (singular values only): svd on (..., m, n) → (..., min(m,n)).
pub fn batch_svd(data: &[f64], shape: &[usize]) -> Result<Vec<f64>, LinAlgError> {
    let (batch, m, n) = parse_batched_shape(shape)?;
    let mat_size = m * n;
    let k = m.min(n);
    if Some(data.len()) != batch.checked_mul(mat_size) {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_svd: data length does not match shape",
        ));
    }
    let lanes = batch_map_lanes(batch, mat_size, |b| {
        svd_mxn(&data[b * mat_size..(b + 1) * mat_size], m, n)
    })?;
    let mut result = Vec::with_capacity(batch * k);
    for sigmas in &lanes {
        result.extend_from_slice(sigmas);
    }
    Ok(result)
}

/// Batched full SVD: svd on (..., m, n) → (U (..., m, m), S (..., min(m,n)), Vt (..., n, n)).
pub fn batch_svd_full(data: &[f64], shape: &[usize]) -> Result<SvdFullResult, LinAlgError> {
    let (batch, m, n) = parse_batched_shape(shape)?;
    let mat_size = m * n;
    let k = m.min(n);
    if Some(data.len()) != batch.checked_mul(mat_size) {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_svd_full: data length does not match shape",
        ));
    }
    let lanes = batch_map_lanes(batch, mat_size, |b| {
        svd_mxn_full(&data[b * mat_size..(b + 1) * mat_size], m, n)
    })?;
    let mut all_u = Vec::with_capacity(batch * m * m);
    let mut all_s = Vec::with_capacity(batch * k);
    let mut all_vt = Vec::with_capacity(batch * n * n);
    for (u, s, vt) in &lanes {
        all_u.extend_from_slice(u);
        all_s.extend_from_slice(s);
        all_vt.extend_from_slice(vt);
    }
    Ok((all_u, all_s, all_vt))
}

/// Batched pseudoinverse: pinv on (..., m, n) → (..., n, m), one parallel lane
/// per stacked matrix. The Moore-Penrose pseudoinverse is unique (independent of
/// the SVD sign/phase convention), so this is parity-safe at allclose level.
pub fn batch_pinv(
    data: &[f64],
    shape: &[usize],
    rcond: Option<f64>,
    rtol: Option<Option<f64>>,
) -> Result<Vec<f64>, LinAlgError> {
    let (batch, m, n) = parse_batched_shape(shape)?;
    let mat_size = m * n;
    if Some(data.len()) != batch.checked_mul(mat_size) {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_pinv: data length does not match shape",
        ));
    }
    let resolved_rcond = resolve_pinv_tolerance_aliases(rcond, rtol)?;
    let lanes = batch_map_lanes(batch, mat_size, |b| {
        pinv_mxn(&data[b * mat_size..(b + 1) * mat_size], m, n, resolved_rcond)
    })?;
    let mut result = Vec::with_capacity(batch * n * m);
    for lane in &lanes {
        result.extend_from_slice(lane);
    }
    Ok(result)
}

/// Batched QR: qr on (..., m, n) → (Q (..., m, m), R (..., m, n)).
pub fn batch_qr(data: &[f64], shape: &[usize]) -> Result<(Vec<f64>, Vec<f64>), LinAlgError> {
    let (batch, m, n) = parse_batched_shape(shape)?;
    let mat_size = m * n;
    if Some(data.len()) != batch.checked_mul(mat_size) {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_qr: data length does not match shape",
        ));
    }
    let lanes = batch_map_lanes(batch, mat_size, |b| {
        qr_mxn(&data[b * mat_size..(b + 1) * mat_size], m, n)
    })?;
    let mut all_q = Vec::with_capacity(batch * m * m);
    let mut all_r = Vec::with_capacity(batch * m * n);
    for (q, r) in &lanes {
        all_q.extend_from_slice(q);
        all_r.extend_from_slice(r);
    }
    Ok((all_q, all_r))
}

/// Batched Cholesky: cholesky on (..., n, n) → (..., n, n).
pub fn batch_cholesky(data: &[f64], shape: &[usize]) -> Result<Vec<f64>, LinAlgError> {
    let (batch, n) = parse_batched_square(shape)?;
    let mat_size = n * n;
    if Some(data.len()) != batch.checked_mul(mat_size) {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_cholesky: data length does not match shape",
        ));
    }
    // Small matrices: write each L DIRECTLY into the flat (pre-zeroed) output — no
    // per-lane Vec, no Vec<Vec>, no flatten. cholesky writes L in place with no
    // scratch, so this needs no per-thread buffers at all. Byte-identical to per-lane
    // cholesky_nxn (the unblocked formula reachable for n < CHOL_MID_MIN). Gated to
    // the small/mid-n regime where alloc-elimination still wins before blocked
    // Cholesky takes over at CHOL_MID_MIN.
    const CHOL_DIRECT_WRITE_MAX_N: usize = 64;
    if n <= CHOL_DIRECT_WRITE_MAX_N {
        let mut result = vec![0.0f64; batch * mat_size];
        if batch_should_parallelize(batch, mat_size) {
            use std::sync::Mutex;
            let first_err: Mutex<Option<(usize, LinAlgError)>> = Mutex::new(None);
            result.par_chunks_mut(mat_size).enumerate().for_each(|(idx, out_chunk)| {
                let a_sub = &data[idx * mat_size..(idx + 1) * mat_size];
                if let Err(e) = cholesky_nxn_into_out(a_sub, n, out_chunk) {
                    let mut slot = first_err.lock().unwrap();
                    let replace = match slot.as_ref() {
                        None => true,
                        Some((i, _)) => idx < *i,
                    };
                    if replace {
                        *slot = Some((idx, e));
                    }
                }
            });
            if let Some((_, e)) = first_err.into_inner().unwrap() {
                return Err(e);
            }
        } else {
            for (idx, out_chunk) in result.chunks_mut(mat_size).enumerate() {
                let a_sub = &data[idx * mat_size..(idx + 1) * mat_size];
                cholesky_nxn_into_out(a_sub, n, out_chunk)?;
            }
        }
        return Ok(result);
    }
    let lanes = batch_map_lanes(batch, mat_size, |b| {
        cholesky_nxn(&data[b * mat_size..(b + 1) * mat_size], n)
    })?;
    let mut result = Vec::with_capacity(batch * mat_size);
    for chol in &lanes {
        result.extend_from_slice(chol);
    }
    Ok(result)
}

/// Batched matrix norm: norm on (..., m, n) → (...).
pub fn batch_matrix_norm(
    data: &[f64],
    shape: &[usize],
    ord: &str,
) -> Result<Vec<f64>, LinAlgError> {
    let (batch, m, n) = parse_batched_shape(shape)?;
    let mat_size = m * n;
    if Some(data.len()) != batch.checked_mul(mat_size) {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_matrix_norm: data length does not match shape",
        ));
    }
    if ord == "fro" {
        if batch > 0 && (m == 0 || n == 0) {
            return Err(LinAlgError::ShapeContractViolation(
                "matrix_norm_nxn: input must be m*n with m,n > 0",
            ));
        }
        let norm_lane = |b: usize| matrix_norm_frobenius_unchecked_at(data, b * mat_size, mat_size);
        if batch_should_parallelize(batch, mat_size) {
            return Ok((0..batch).into_par_iter().map(norm_lane).collect());
        }
        return Ok((0..batch).map(norm_lane).collect());
    }
    if ord == "1" || ord == "-1" {
        if batch > 0 && (m == 0 || n == 0) {
            return Err(LinAlgError::ShapeContractViolation(
                "matrix_norm_nxn: input must be m*n with m,n > 0",
            ));
        }
        let use_min = ord == "-1";
        let norm_lane = |b: usize| {
            let base = b * mat_size;
            matrix_norm_column_sum(&data[base..base + mat_size], m, n, use_min)
        };
        if batch_should_parallelize(batch, mat_size) {
            return Ok((0..batch).into_par_iter().map(norm_lane).collect());
        }
        return Ok((0..batch).map(norm_lane).collect());
    }
    if ord == "inf" || ord == "-inf" {
        if batch > 0 && (m == 0 || n == 0) {
            return Err(LinAlgError::ShapeContractViolation(
                "matrix_norm_nxn: input must be m*n with m,n > 0",
            ));
        }
        let use_min = ord == "-inf";
        let norm_lane = |b: usize| {
            let base = b * mat_size;
            matrix_norm_row_sum(&data[base..base + mat_size], n, use_min)
        };
        if batch_should_parallelize(batch, mat_size) {
            return Ok((0..batch).into_par_iter().map(norm_lane).collect());
        }
        return Ok((0..batch).map(norm_lane).collect());
    }
    batch_map_lanes(batch, mat_size, |b| {
        matrix_norm_nxn(&data[b * mat_size..(b + 1) * mat_size], m, n, ord)
    })
}

/// Batched matrix rank: rank on (..., m, n) → (...).
pub fn batch_matrix_rank(
    data: &[f64],
    shape: &[usize],
    rcond: f64,
) -> Result<Vec<usize>, LinAlgError> {
    let (batch, m, n) = parse_batched_shape(shape)?;
    if m != n {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_matrix_rank: currently requires square matrices",
        ));
    }
    let mat_size = n * n;
    if Some(data.len()) != batch.checked_mul(mat_size) {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_matrix_rank: data length does not match shape",
        ));
    }
    batch_map_lanes(batch, mat_size, |b| {
        matrix_rank_nxn(&data[b * mat_size..(b + 1) * mat_size], n, rcond)
    })
}

/// Batched trace: trace on (..., n, n) → (...).
pub fn batch_trace(data: &[f64], shape: &[usize]) -> Result<Vec<f64>, LinAlgError> {
    let (batch, n) = parse_batched_square(shape)?;
    let mat_size = n * n;
    if Some(data.len()) != batch.checked_mul(mat_size) {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_trace: data length does not match shape",
        ));
    }
    let trace_lane = |b: usize| trace_nxn_unchecked_at(data, b * mat_size, n);
    if batch_should_parallelize(batch, mat_size) {
        Ok((0..batch).into_par_iter().map(trace_lane).collect())
    } else {
        Ok((0..batch).map(trace_lane).collect())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Complex-valued linear algebra operations
// ────────────────────────────────────────────────────────────────────────────
//
// Complex numbers are stored as interleaved (re, im) pairs in flat Vec<f64>.
// A complex n×n matrix has 2·n·n f64 values.
// Entry (i,j) real part: data[2*(i*n+j)], imaginary part: data[2*(i*n+j)+1].

/// Complex number multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
#[inline]
fn cmul(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    (ar * br - ai * bi, ar * bi + ai * br)
}

/// Complex conjugate multiply: conj(a)*b = (ar-ai·i)(br+bi·i)
#[inline]
fn cmul_conj_a(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    (ar * br + ai * bi, ar * bi - ai * br)
}

/// Complex modulus squared: |z|² = re² + im²
#[inline]
fn cabs2(re: f64, im: f64) -> f64 {
    re * re + im * im
}

/// Complex division: (a+bi)/(c+di)
#[inline]
fn cdiv(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    let denom = cabs2(br, bi);
    ((ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom)
}

/// LU decomposition with partial pivoting for complex matrices.
/// Input: interleaved (re,im) flat array of length 2·n·n.
/// Returns (lu_interleaved, perm, sign_re, sign_im).
fn complex_lu_decompose(
    a: &[f64],
    n: usize,
) -> Result<(Vec<f64>, Vec<usize>, f64, f64), LinAlgError> {
    if Some(a.len()) != n.checked_mul(n).and_then(|x| x.checked_mul(2)) || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "complex LU input must be 2*n*n with n > 0",
        ));
    }
    for v in a {
        if !v.is_finite() {
            return Err(LinAlgError::NormDetRankPolicyViolation(
                "complex matrix entries must be finite for LU",
            ));
        }
    }

    let matrix_max_abs = a
        .chunks_exact(2)
        .map(|c| cabs2(c[0], c[1]).sqrt())
        .fold(0.0_f64, f64::max);
    let singularity_threshold = (n as f64) * f64::EPSILON * matrix_max_abs;

    let mut lu = a.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();
    let mut sign_re = 1.0_f64;
    let mut sign_im = 0.0_f64;

    for k in 0..n {
        // partial-pivot: find row with max |entry| in column k
        let mut max_abs = cabs2(lu[2 * (k * n + k)], lu[2 * (k * n + k) + 1]).sqrt();
        let mut max_row = k;
        for i in (k + 1)..n {
            let abs_val = cabs2(lu[2 * (i * n + k)], lu[2 * (i * n + k) + 1]).sqrt();
            if abs_val > max_abs {
                max_abs = abs_val;
                max_row = i;
            }
        }

        if max_abs <= singularity_threshold {
            return Err(LinAlgError::SolverSingularity);
        }

        if max_row != k {
            for j in 0..n {
                lu.swap(2 * (k * n + j), 2 * (max_row * n + j));
                lu.swap(2 * (k * n + j) + 1, 2 * (max_row * n + j) + 1);
            }
            perm.swap(k, max_row);
            // swap negates determinant sign
            let (sr, si) = cmul(sign_re, sign_im, -1.0, 0.0);
            sign_re = sr;
            sign_im = si;
        }

        let pivot_re = lu[2 * (k * n + k)];
        let pivot_im = lu[2 * (k * n + k) + 1];
        // Rank-1 trailing update — same independent-rows structure as the real
        // lu_decompose_inner: each trailing row reads the unchanged pivot row and
        // writes its own disjoint (interleaved) row, one cmul-subtract per element,
        // so parallelizing across rows is BIT-IDENTICAL to the serial loop.
        for i in (k + 1)..n {
            let (fr, fi) = cdiv(
                lu[2 * (i * n + k)],
                lu[2 * (i * n + k) + 1],
                pivot_re,
                pivot_im,
            );
            lu[2 * (i * n + k)] = fr;
            lu[2 * (i * n + k) + 1] = fi;
            for j in (k + 1)..n {
                let ur = lu[2 * (k * n + j)];
                let ui = lu[2 * (k * n + j) + 1];
                let (pr, pi) = cmul(fr, fi, ur, ui);
                lu[2 * (i * n + j)] -= pr;
                lu[2 * (i * n + j) + 1] -= pi;
            }
        }
    }

    Ok((lu, perm, sign_re, sign_im))
}

/// Forward-substitution then back-substitution for complex LU system.
fn complex_lu_forward_back(lu: &[f64], perm: &[usize], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0; 2 * n];
    for i in 0..n {
        let p = perm[i];
        x[2 * i] = b[2 * p];
        x[2 * i + 1] = b[2 * p + 1];
    }

    // forward (L has unit diagonal)
    for i in 1..n {
        for j in 0..i {
            let lr = lu[2 * (i * n + j)];
            let li = lu[2 * (i * n + j) + 1];
            let (pr, pi) = cmul(lr, li, x[2 * j], x[2 * j + 1]);
            x[2 * i] -= pr;
            x[2 * i + 1] -= pi;
        }
    }

    // back
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            let ur = lu[2 * (i * n + j)];
            let ui = lu[2 * (i * n + j) + 1];
            let (pr, pi) = cmul(ur, ui, x[2 * j], x[2 * j + 1]);
            x[2 * i] -= pr;
            x[2 * i + 1] -= pi;
        }
        let (dr, di) = cdiv(
            x[2 * i],
            x[2 * i + 1],
            lu[2 * (i * n + i)],
            lu[2 * (i * n + i) + 1],
        );
        x[2 * i] = dr;
        x[2 * i + 1] = di;
    }

    x
}

/// Solve Ax = b for a complex NxN system via LU decomposition.
/// `a` is 2·n·n interleaved, `b` is 2·n interleaved.
pub fn complex_solve_nxn(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, LinAlgError> {
    if Some(b.len()) != n.checked_mul(2) {
        return Err(LinAlgError::ShapeContractViolation(
            "complex_solve_nxn: rhs length must equal 2*n",
        ));
    }
    for v in b {
        if !v.is_finite() {
            return Err(LinAlgError::NormDetRankPolicyViolation(
                "complex rhs entries must be finite",
            ));
        }
    }

    let (lu, perm, _, _) = complex_lu_decompose(a, n)?;
    Ok(complex_lu_forward_back(&lu, &perm, b, n))
}

/// Determinant of a complex NxN matrix.
/// Returns (det_re, det_im).
pub fn complex_det_nxn(a: &[f64], n: usize) -> Result<(f64, f64), LinAlgError> {
    if Some(a.len()) != n.checked_mul(n).and_then(|x| x.checked_mul(2)) || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "complex_det_nxn: input must be 2*n*n with n > 0",
        ));
    }

    match complex_lu_decompose(a, n) {
        Ok((lu, _, sign_re, sign_im)) => {
            let mut dr = sign_re;
            let mut di = sign_im;
            for i in 0..n {
                let ur = lu[2 * (i * n + i)];
                let ui = lu[2 * (i * n + i) + 1];
                let (nr, ni) = cmul(dr, di, ur, ui);
                dr = nr;
                di = ni;
            }
            Ok((dr, di))
        }
        Err(LinAlgError::SolverSingularity) => Ok((0.0, 0.0)),
        Err(e) => Err(e),
    }
}

/// Inverse of a complex NxN matrix via LU decomposition.
/// Returns 2·n·n interleaved flat row-major.
pub fn complex_inv_nxn(a: &[f64], n: usize) -> Result<Vec<f64>, LinAlgError> {
    let (lu, perm, _, _) = complex_lu_decompose(a, n)?;
    let mut inv = vec![0.0; 2 * n * n];

    for col in 0..n {
        let mut e_col = vec![0.0; 2 * n];
        e_col[2 * col] = 1.0;
        let x = complex_lu_forward_back(&lu, &perm, &e_col, n);
        for row in 0..n {
            inv[2 * (row * n + col)] = x[2 * row];
            inv[2 * (row * n + col) + 1] = x[2 * row + 1];
        }
    }

    Ok(inv)
}

/// Cholesky factorization for Hermitian positive-definite complex matrix.
/// A = L·L^H where L is lower triangular.
/// Input: 2·n·n interleaved. Output: 2·n·n interleaved lower-triangular L.
pub fn complex_cholesky_nxn(a: &[f64], n: usize) -> Result<Vec<f64>, LinAlgError> {
    if Some(a.len()) != n.checked_mul(n).and_then(|x| x.checked_mul(2)) || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "complex_cholesky: input must be 2*n*n with n > 0",
        ));
    }
    for v in a {
        if !v.is_finite() {
            return Err(LinAlgError::NormDetRankPolicyViolation(
                "complex matrix entries must be finite for Cholesky",
            ));
        }
    }

    let mut l = vec![0.0; 2 * n * n];

    for i in 0..n {
        // Diagonal: L[i,i] = sqrt(A[i,i] - sum_{k<i} |L[i,k]|²)
        let mut diag_re = a[2 * (i * n + i)];
        for k in 0..i {
            diag_re -= cabs2(l[2 * (i * n + k)], l[2 * (i * n + k) + 1]);
        }
        // Diagonal of Hermitian PD matrix must be real and positive
        if diag_re <= 0.0 {
            return Err(LinAlgError::CholeskyContractViolation(
                "complex_cholesky: matrix is not Hermitian positive-definite",
            ));
        }
        let lii = diag_re.sqrt();
        l[2 * (i * n + i)] = lii;
        // imaginary part of diagonal is zero for Hermitian matrix
        l[2 * (i * n + i) + 1] = 0.0;

        // Off-diagonal: L[j,i] = (A[j,i] - sum_{k<i} L[j,k]*conj(L[i,k])) / L[i,i]
        for j in (i + 1)..n {
            let mut sr = a[2 * (j * n + i)];
            let mut si = a[2 * (j * n + i) + 1];
            for k in 0..i {
                let (pr, pi) = cmul(
                    l[2 * (j * n + k)],
                    l[2 * (j * n + k) + 1],
                    l[2 * (i * n + k)],
                    -l[2 * (i * n + k) + 1], // conjugate
                );
                sr -= pr;
                si -= pi;
            }
            l[2 * (j * n + i)] = sr / lii;
            l[2 * (j * n + i) + 1] = si / lii;
        }
    }

    Ok(l)
}

/// QR decomposition for complex m×n matrix (m ≥ 1, n ≥ 1) using Householder.
/// Input: 2·m·n interleaved. Returns (Q: 2·m·m, R: 2·m·n).
pub fn complex_qr_mxn(a: &[f64], m: usize, n: usize) -> Result<(Vec<f64>, Vec<f64>), LinAlgError> {
    if Some(a.len()) != m.checked_mul(n).and_then(|x| x.checked_mul(2)) || m == 0 || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "complex_qr: input must be 2*m*n with m,n > 0",
        ));
    }
    for v in a {
        if !v.is_finite() {
            return Err(LinAlgError::NormDetRankPolicyViolation(
                "complex matrix entries must be finite for QR",
            ));
        }
    }

    let mut r = a.to_vec();
    // Q starts as identity
    let mut q = vec![0.0; 2 * m * m];
    for i in 0..m {
        q[2 * (i * m + i)] = 1.0;
    }

    let k = m.min(n);
    for col in 0..k {
        // Build Householder vector for column `col`, rows col..m
        let sub_len = m - col;
        let mut v = vec![0.0; 2 * sub_len];
        for i in 0..sub_len {
            v[2 * i] = r[2 * ((col + i) * n + col)];
            v[2 * i + 1] = r[2 * ((col + i) * n + col) + 1];
        }

        // Compute norm of v
        let mut norm_sq = 0.0;
        for i in 0..sub_len {
            norm_sq += cabs2(v[2 * i], v[2 * i + 1]);
        }
        let norm = norm_sq.sqrt();

        if norm < f64::MIN_POSITIVE {
            continue;
        }

        // Phase: e^{iθ} where θ = arg(v[0])
        let v0_abs = cabs2(v[0], v[1]).sqrt();
        let (phase_re, phase_im) = if v0_abs > f64::MIN_POSITIVE {
            (v[0] / v0_abs, v[1] / v0_abs)
        } else {
            (1.0, 0.0)
        };

        // v[0] += phase * norm
        let (shift_re, shift_im) = cmul(phase_re, phase_im, norm, 0.0);
        v[0] += shift_re;
        v[1] += shift_im;

        // Normalize v
        let mut v_norm_sq = 0.0;
        for i in 0..sub_len {
            v_norm_sq += cabs2(v[2 * i], v[2 * i + 1]);
        }
        let v_norm = v_norm_sq.sqrt();
        if v_norm < f64::MIN_POSITIVE {
            continue;
        }
        for i in 0..sub_len {
            v[2 * i] /= v_norm;
            v[2 * i + 1] /= v_norm;
        }

        // Apply Householder H = I - 2*v*v^H to R columns
        for j in col..n {
            // dot = v^H · R[col:, j]
            let mut dot_re = 0.0;
            let mut dot_im = 0.0;
            for i in 0..sub_len {
                let (pr, pi) = cmul_conj_a(
                    v[2 * i],
                    v[2 * i + 1],
                    r[2 * ((col + i) * n + j)],
                    r[2 * ((col + i) * n + j) + 1],
                );
                dot_re += pr;
                dot_im += pi;
            }
            // R[col+i, j] -= 2 * v[i] * dot
            for i in 0..sub_len {
                let (pr, pi) = cmul(v[2 * i], v[2 * i + 1], dot_re, dot_im);
                r[2 * ((col + i) * n + j)] -= 2.0 * pr;
                r[2 * ((col + i) * n + j) + 1] -= 2.0 * pi;
            }
        }

        // Apply Householder to Q: Q = Q · H = Q - 2*(Q·v)·v^H
        for i in 0..m {
            // dot = sum_k Q[i, col+k] * v[k]
            let mut dot_re = 0.0;
            let mut dot_im = 0.0;
            for kk in 0..sub_len {
                let (pr, pi) = cmul(
                    q[2 * (i * m + col + kk)],
                    q[2 * (i * m + col + kk) + 1],
                    v[2 * kk],
                    v[2 * kk + 1],
                );
                dot_re += pr;
                dot_im += pi;
            }
            // Q[i, col+k] -= 2 * dot * conj(v[k])
            for kk in 0..sub_len {
                let (pr, pi) = cmul(dot_re, dot_im, v[2 * kk], -v[2 * kk + 1]);
                q[2 * (i * m + col + kk)] -= 2.0 * pr;
                q[2 * (i * m + col + kk) + 1] -= 2.0 * pi;
            }
        }
    }

    Ok((q, r))
}

/// Frobenius norm of a complex matrix (2·m·n interleaved).
pub fn complex_matrix_norm_frobenius(a: &[f64]) -> f64 {
    let mut sum = 0.0;
    for chunk in a.chunks_exact(2) {
        sum += cabs2(chunk[0], chunk[1]);
    }
    sum.sqrt()
}

/// Trace of a complex NxN matrix. Returns (re, im).
pub fn complex_trace_nxn(a: &[f64], n: usize) -> Result<(f64, f64), LinAlgError> {
    if Some(a.len()) != n.checked_mul(n).and_then(|x| x.checked_mul(2)) || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "complex_trace: input must be 2*n*n with n > 0",
        ));
    }
    let mut re = 0.0;
    let mut im = 0.0;
    for i in 0..n {
        re += a[2 * (i * n + i)];
        im += a[2 * (i * n + i) + 1];
    }
    Ok((re, im))
}

/// Matrix-vector multiply for complex: y = A*x.
/// A is 2·m·n interleaved, x is 2·n interleaved, result is 2·m interleaved.
pub fn complex_matvec(a: &[f64], x: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut y = vec![0.0; 2 * m];
    for i in 0..m {
        for j in 0..n {
            let (pr, pi) = cmul(
                a[2 * (i * n + j)],
                a[2 * (i * n + j) + 1],
                x[2 * j],
                x[2 * j + 1],
            );
            y[2 * i] += pr;
            y[2 * i + 1] += pi;
        }
    }
    y
}

/// Matrix-matrix multiply for complex: C = A*B.
/// A is 2·m·k interleaved, B is 2·k·n interleaved, result is 2·m·n interleaved.
/// Minimum FLOP count (m·k·n) before the interleaved-complex GEMM fans out across
/// row bands; below this the work is too small to amortize thread dispatch.
const COMPLEX_MATMUL_PARALLEL_MIN_FLOPS: usize = 1 << 18;
const COMPLEX_PACKED_NR: usize = 8;
const COMPLEX_PACKED_MIN_FLOPS: usize = 1 << 23;

fn complex_matmul_stream_row_block4(
    a: &[f64],
    b: &[f64],
    k: usize,
    n: usize,
    row_start: usize,
    c: &mut [f64],
) {
    if c.len() == MATMUL_ROW_BLOCK * 2 * n {
        let (c0, rest) = c.split_at_mut(2 * n);
        let (c1, rest) = rest.split_at_mut(2 * n);
        let (c2, c3) = rest.split_at_mut(2 * n);
        let a0 = &a[2 * row_start * k..2 * row_start * k + 2 * k];
        let a1 = &a[2 * (row_start + 1) * k..2 * (row_start + 1) * k + 2 * k];
        let a2 = &a[2 * (row_start + 2) * k..2 * (row_start + 2) * k + 2 * k];
        let a3 = &a[2 * (row_start + 3) * k..2 * (row_start + 3) * k + 2 * k];

        for p in 0..k {
            let b_row = &b[2 * p * n..2 * p * n + 2 * n];
            let (a0r, a0i) = (a0[2 * p], a0[2 * p + 1]);
            let (a1r, a1i) = (a1[2 * p], a1[2 * p + 1]);
            let (a2r, a2i) = (a2[2 * p], a2[2 * p + 1]);
            let (a3r, a3i) = (a3[2 * p], a3[2 * p + 1]);
            for j in 0..n {
                let (br, bi) = (b_row[2 * j], b_row[2 * j + 1]);
                let (p0r, p0i) = cmul(a0r, a0i, br, bi);
                let (p1r, p1i) = cmul(a1r, a1i, br, bi);
                let (p2r, p2i) = cmul(a2r, a2i, br, bi);
                let (p3r, p3i) = cmul(a3r, a3i, br, bi);
                c0[2 * j] += p0r;
                c0[2 * j + 1] += p0i;
                c1[2 * j] += p1r;
                c1[2 * j + 1] += p1i;
                c2[2 * j] += p2r;
                c2[2 * j + 1] += p2i;
                c3[2 * j] += p3r;
                c3[2 * j + 1] += p3i;
            }
        }
    } else {
        let rows = c.len() / (2 * n);
        for ii in 0..rows {
            let i = row_start + ii;
            let c_row = &mut c[ii * 2 * n..ii * 2 * n + 2 * n];
            let a_row = &a[2 * i * k..2 * i * k + 2 * k];
            for p in 0..k {
                let (ar, ai) = (a_row[2 * p], a_row[2 * p + 1]);
                let b_row = &b[2 * p * n..2 * p * n + 2 * n];
                for j in 0..n {
                    let (pr, pi) = cmul(ar, ai, b_row[2 * j], b_row[2 * j + 1]);
                    c_row[2 * j] += pr;
                    c_row[2 * j + 1] += pi;
                }
            }
        }
    }
}

fn complex_matmul_packed_tile4(
    a: &[f64],
    bp: &[f64],
    k: usize,
    n: usize,
    row_start: usize,
    j0: usize,
    c: &mut [f64],
) {
    let (c0, rest) = c.split_at_mut(2 * n);
    let (c1, rest) = rest.split_at_mut(2 * n);
    let (c2, c3) = rest.split_at_mut(2 * n);
    let a0 = &a[2 * row_start * k..2 * row_start * k + 2 * k];
    let a1 = &a[2 * (row_start + 1) * k..2 * (row_start + 1) * k + 2 * k];
    let a2 = &a[2 * (row_start + 2) * k..2 * (row_start + 2) * k + 2 * k];
    let a3 = &a[2 * (row_start + 3) * k..2 * (row_start + 3) * k + 2 * k];
    let mut acc_r = [[0.0f64; COMPLEX_PACKED_NR]; MATMUL_ROW_BLOCK];
    let mut acc_i = [[0.0f64; COMPLEX_PACKED_NR]; MATMUL_ROW_BLOCK];

    for p in 0..k {
        let brow = &bp[2 * p * COMPLEX_PACKED_NR..2 * (p + 1) * COMPLEX_PACKED_NR];
        let (a0r, a0i) = (a0[2 * p], a0[2 * p + 1]);
        let (a1r, a1i) = (a1[2 * p], a1[2 * p + 1]);
        let (a2r, a2i) = (a2[2 * p], a2[2 * p + 1]);
        let (a3r, a3i) = (a3[2 * p], a3[2 * p + 1]);
        for j in 0..COMPLEX_PACKED_NR {
            let (br, bi) = (brow[2 * j], brow[2 * j + 1]);
            let (p0r, p0i) = cmul(a0r, a0i, br, bi);
            let (p1r, p1i) = cmul(a1r, a1i, br, bi);
            let (p2r, p2i) = cmul(a2r, a2i, br, bi);
            let (p3r, p3i) = cmul(a3r, a3i, br, bi);
            acc_r[0][j] += p0r;
            acc_i[0][j] += p0i;
            acc_r[1][j] += p1r;
            acc_i[1][j] += p1i;
            acc_r[2][j] += p2r;
            acc_i[2][j] += p2i;
            acc_r[3][j] += p3r;
            acc_i[3][j] += p3i;
        }
    }

    for j in 0..COMPLEX_PACKED_NR {
        let dst = 2 * (j0 + j);
        c0[dst] += acc_r[0][j];
        c0[dst + 1] += acc_i[0][j];
        c1[dst] += acc_r[1][j];
        c1[dst + 1] += acc_i[1][j];
        c2[dst] += acc_r[2][j];
        c2[dst + 1] += acc_i[2][j];
        c3[dst] += acc_r[3][j];
        c3[dst + 1] += acc_i[3][j];
    }
}

fn complex_matmul_band(a: &[f64], b: &[f64], k: usize, n: usize, row_start: usize, c: &mut [f64]) {
    let rows = c.len() / (2 * n);
    let row_full = rows - rows % MATMUL_ROW_BLOCK;
    let n_full = n - n % COMPLEX_PACKED_NR;
    let mut bp = vec![0.0f64; 2 * k * COMPLEX_PACKED_NR];

    let mut j0 = 0;
    while j0 < n_full {
        for p in 0..k {
            bp[2 * p * COMPLEX_PACKED_NR..2 * (p + 1) * COMPLEX_PACKED_NR]
                .copy_from_slice(&b[2 * (p * n + j0)..2 * (p * n + j0 + COMPLEX_PACKED_NR)]);
        }
        let mut ii = 0;
        while ii < row_full {
            let c_block = &mut c[ii * 2 * n..(ii + MATMUL_ROW_BLOCK) * 2 * n];
            complex_matmul_packed_tile4(a, &bp, k, n, row_start + ii, j0, c_block);
            ii += MATMUL_ROW_BLOCK;
        }
        for ii in row_full..rows {
            let i = row_start + ii;
            let c_row = &mut c[ii * 2 * n..ii * 2 * n + 2 * n];
            let a_row = &a[2 * i * k..2 * i * k + 2 * k];
            for p in 0..k {
                let (ar, ai) = (a_row[2 * p], a_row[2 * p + 1]);
                let brow = &bp[2 * p * COMPLEX_PACKED_NR..2 * (p + 1) * COMPLEX_PACKED_NR];
                for j in 0..COMPLEX_PACKED_NR {
                    let (pr, pi) = cmul(ar, ai, brow[2 * j], brow[2 * j + 1]);
                    let dst = 2 * (j0 + j);
                    c_row[dst] += pr;
                    c_row[dst + 1] += pi;
                }
            }
        }
        j0 += COMPLEX_PACKED_NR;
    }

    for ii in 0..rows {
        let i = row_start + ii;
        let c_row = &mut c[ii * 2 * n..ii * 2 * n + 2 * n];
        let a_row = &a[2 * i * k..2 * i * k + 2 * k];
        for p in 0..k {
            let (ar, ai) = (a_row[2 * p], a_row[2 * p + 1]);
            let b_row = &b[2 * p * n..2 * p * n + 2 * n];
            for j in n_full..n {
                let (pr, pi) = cmul(ar, ai, b_row[2 * j], b_row[2 * j + 1]);
                c_row[2 * j] += pr;
                c_row[2 * j + 1] += pi;
            }
        }
    }
}

pub fn complex_matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; 2 * m * n];
    if c.is_empty() {
        return c;
    }

    // Cache-banded four-row microkernel: each band packs B's complex column
    // tiles once, then feeds several four-row A tiles before advancing. Every
    // c[i,j] still accumulates p in 0..k ascending order through the same
    // `cmul` operation as the serial reference; tails use the same p sequence.
    let flops = m.saturating_mul(k).saturating_mul(n);
    let use_packed = flops >= COMPLEX_PACKED_MIN_FLOPS && n >= COMPLEX_PACKED_NR && m >= 4;
    if use_packed
        && rayon::current_num_threads() >= 2
        && flops >= COMPLEX_MATMUL_PARALLEL_MIN_FLOPS
        && m >= 8
    {
        let threads = rayon::current_num_threads();
        let band_rows =
            (m.div_ceil(threads * 2).div_ceil(MATMUL_ROW_BLOCK).max(2)) * MATMUL_ROW_BLOCK;
        c.par_chunks_mut(band_rows * 2 * n)
            .enumerate()
            .for_each(|(block, c_block)| {
                complex_matmul_band(a, b, k, n, block * band_rows, c_block);
            });
    } else if use_packed {
        complex_matmul_band(a, b, k, n, 0, &mut c);
    } else if rayon::current_num_threads() >= 2
        && flops >= COMPLEX_MATMUL_PARALLEL_MIN_FLOPS
        && m >= 8
    {
        c.par_chunks_mut(MATMUL_ROW_BLOCK * 2 * n)
            .enumerate()
            .for_each(|(block, c_block)| {
                complex_matmul_stream_row_block4(a, b, k, n, block * MATMUL_ROW_BLOCK, c_block);
            });
    } else {
        complex_matmul_stream_row_block4(a, b, k, n, 0, &mut c);
    }
    c
}

/// Conjugate transpose (Hermitian transpose) of a complex m×n matrix.
/// Returns 2·n·m interleaved.
pub fn complex_conjugate_transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut ah = vec![0.0; 2 * n * m];
    for i in 0..m {
        for j in 0..n {
            ah[2 * (j * m + i)] = a[2 * (i * n + j)];
            ah[2 * (j * m + i) + 1] = -a[2 * (i * n + j) + 1];
        }
    }
    ah
}

#[cfg(test)]
mod tests {
    #[test]
    fn batch_inv_small_matrices_parallel_matches_serial_bits() {
        // A large batch of small (n < 128) matrices now takes the parallel path via
        // the total-work gate (it stayed serial under the old per-lane gate). The
        // lanes are independent and collected in order, so the result must be
        // bit-for-bit identical to a serial per-lane inversion. n=8, batch large
        // enough that total elems >= BATCH_PARALLEL_MIN_TOTAL_ELEMS.
        let n = 8usize;
        let ms = n * n;
        let batch = 4096usize; // total = 4096*64 = 262144 >= 1<<14
        let mat: Vec<f64> = (0..batch * ms)
            .map(|i| {
                let cell = i % ms;
                let (r, c) = (cell / n, cell % n);
                if r == c {
                    n as f64 + 1.0 + ((i / ms) % 5) as f64
                } else {
                    (((i % 13) as f64) - 6.0) * 0.1
                }
            })
            .collect();
        let shape = [batch, n, n];
        let parallel = super::batch_inv(&mat, &shape).expect("batch_inv");
        let mut serial = Vec::with_capacity(batch * ms);
        for b in 0..batch {
            serial.extend_from_slice(&super::inv_nxn(&mat[b * ms..(b + 1) * ms], n).unwrap());
        }
        assert_eq!(parallel.len(), serial.len());
        for (i, (p, s)) in parallel.iter().zip(&serial).enumerate() {
            assert_eq!(p.to_bits(), s.to_bits(), "lane-flattened index {i} diverged");
        }
    }

    use super::{
        LINALG_PACKET_ID,
        LINALG_PACKET_REASON_CODES,
        LinAlgError,
        LinAlgLogRecord,
        LinAlgRuntimeMode,
        MAX_BACKEND_REVALIDATION_ATTEMPTS,
        MAX_BATCH_SHAPE_CHECKS,
        MAX_TOLERANCE_SEARCH_DEPTH,
        MatrixNormOrder,
        QrMode,
        VectorNormOrder,
        // Batched linalg
        batch_cholesky,
        batch_det,
        batch_eig,
        batch_eigvalsh,
        batch_inv,
        batch_matrix_norm,
        batch_qr,
        batch_slogdet,
        batch_solve,
        batch_svd,
        batch_trace,
        cholesky_2x2,
        cholesky_nxn,
        cholesky_solve,
        cholesky_solve_multi,
        cmul,
        // Complex linalg
        complex_cholesky_nxn,
        complex_conjugate_transpose,
        complex_det_nxn,
        complex_inv_nxn,
        complex_matmul,
        complex_matrix_norm_frobenius,
        complex_matvec,
        complex_qr_mxn,
        complex_solve_nxn,
        complex_trace_nxn,
        cond_mxn,
        cond_nxn,
        cond_p_nxn,
        cross_product,
        det_2x2,
        det_nxn,
        eig_nxn,
        eig_nxn_full,
        eigh_2x2,
        eigh_nxn,
        eigvals_2x2,
        eigvalsh_nxn,
        expm_nxn,
        funm_nxn,
        inv_2x2,
        inv_nxn,
        kron_nxn,
        logm_nxn,
        lstsq_2x2,
        lstsq_nxn,
        lstsq_output_shapes,
        lstsq_svd,
        lu_factor_nxn,
        lu_solve,
        mat_mul_flat,
        mat_mul_rect,
        matrix_norm_2x2,
        matrix_norm_frobenius,
        matrix_norm_nxn,
        matrix_power_nxn,
        matrix_rank_2x2,
        matrix_rank_2x2_tol,
        matrix_rank_mxn_tol,
        matrix_rank_nxn,
        matrix_rank_nxn_tol,
        multi_dot,
        pinv_2x2,
        pinv_2x2_with_tolerance_aliases,
        pinv_hermitian_nxn,
        pinv_hermitian_nxn_with_tolerance_aliases,
        pinv_mxn_with_tolerance_aliases,
        pinv_nxn,
        polar_nxn,
        qr_2x2,
        qr_mxn,
        qr_nxn,
        qr_output_shapes,
        schur_nxn,
        slogdet_2x2,
        slogdet_nxn,
        solve_2x2,
        solve_nxn,
        solve_nxn_multi,
        solve_triangular,
        sqrtm_nxn,
        svd_2x2,
        svd_bidiag_full_with_max_iters,
        svd_mxn,
        svd_mxn_full,
        svd_nxn,
        svd_output_shapes,
        tensorinv,
        tensorsolve,
        trace_nxn,
        tridiag_reduce,
        tridiag_reduce_values,
        validate_backend_bridge,
        validate_cholesky_diagonal,
        validate_matrix_shape,
        validate_policy_metadata,
        validate_spectral_branch,
        validate_square_matrix,
        validate_tolerance_policy,
        vector_norm,
    };
    use sha2::{Digest, Sha256};
    use std::process::Command;

    fn packet008_artifacts() -> Vec<String> {
        vec![
            "artifacts/phase2c/FNP-P2C-008/contract_table.md".to_string(),
            "artifacts/phase2c/FNP-P2C-008/unit_property_evidence.json".to_string(),
        ]
    }

    fn approx_equal(lhs: f64, rhs: f64, tol: f64) -> bool {
        (lhs - rhs).abs() <= tol
    }

    #[test]
    fn batch_lanes_parallel_match_serial_reference_and_golden_sha256() {
        // Per-lane size (n*n = 128*128 = 16_384) crosses
        // BATCH_PARALLEL_MIN_LANE_ELEMS so the rayon lane path actually runs on
        // a multi-core worker. The proof itself is threading-independent: each
        // lane runs the identical scalar kernel on a disjoint sub-matrix and
        // results are assembled in lane order, so the parallel output must
        // equal a hand-written serial reference bit-for-bit.
        let batch = 16usize;
        let n = 128usize;
        let mat_size = n * n;
        let mut data = Vec::with_capacity(batch * mat_size);
        for b in 0..batch {
            let bump = (b % 9) as f64 * 0.5;
            for i in 0..n {
                for j in 0..n {
                    // Symmetric, strongly diagonally dominant => SPD (valid for
                    // cholesky/eigvalsh) and well-conditioned (valid for inv).
                    let off = 1.0 / ((i as f64 - j as f64).abs() + 1.0);
                    data.push(if i == j { n as f64 + bump + 2.0 } else { off });
                }
            }
        }
        let shape = vec![batch, n, n];

        // Explicit serial reference: same scalar kernel, lane-ordered assembly.
        let mut inv_ref = Vec::with_capacity(batch * mat_size);
        let mut chol_ref = Vec::with_capacity(batch * mat_size);
        let mut eig_ref = Vec::with_capacity(batch * n);
        for b in 0..batch {
            let sub = &data[b * mat_size..(b + 1) * mat_size];
            inv_ref.extend_from_slice(&inv_nxn(sub, n).unwrap());
            chol_ref.extend_from_slice(&cholesky_nxn(sub, n).unwrap());
            eig_ref.extend_from_slice(&eigvalsh_nxn(sub, n).unwrap());
        }

        let inv_out = batch_inv(&data, &shape).unwrap();
        let chol_out = batch_cholesky(&data, &shape).unwrap();
        let eig_out = batch_eigvalsh(&data, &shape).unwrap();

        assert_eq!(inv_out.len(), inv_ref.len());
        assert_eq!(chol_out.len(), chol_ref.len());
        assert_eq!(eig_out.len(), eig_ref.len());
        for (a, b) in inv_out.iter().zip(&inv_ref) {
            assert_eq!(a.to_bits(), b.to_bits(), "batch_inv lane drifted");
        }
        for (a, b) in chol_out.iter().zip(&chol_ref) {
            assert_eq!(a.to_bits(), b.to_bits(), "batch_cholesky lane drifted");
        }
        for (a, b) in eig_out.iter().zip(&eig_ref) {
            assert_eq!(a.to_bits(), b.to_bits(), "batch_eigvalsh lane drifted");
        }

        // Defense against blind golden re-pins: before trusting the digest,
        // prove the *actual* outputs still solve their defining equations to
        // ~machine epsilon. A future kernel refactor that changes the bits but
        // keeps these residuals tiny is a benign last-ULP drift (re-pin the
        // digest); one that blows a residual is a real regression (do NOT
        // re-pin). These checks are threading- and load-independent.
        let mut max_inv_resid = 0.0f64; // ||A·inv - I||_max over all lanes
        let mut max_chol_resid = 0.0f64; // ||L·Lᵀ - A||_max over all lanes
        let mut max_eig_resid = 0.0f64; // |Σλ - trace(A)| over all lanes
        for b in 0..batch {
            let a_sub = &data[b * mat_size..(b + 1) * mat_size];
            let inv_sub = &inv_out[b * mat_size..(b + 1) * mat_size];
            let chol_sub = &chol_out[b * mat_size..(b + 1) * mat_size];
            let eig_sub = &eig_out[b * n..(b + 1) * n];
            for i in 0..n {
                for j in 0..n {
                    let mut ainv = 0.0;
                    let mut llt = 0.0; // L is lower-triangular row-major: L[r*n+k]
                    for k in 0..n {
                        ainv += a_sub[i * n + k] * inv_sub[k * n + j];
                        llt += chol_sub[i * n + k] * chol_sub[j * n + k];
                    }
                    let eye = if i == j { 1.0 } else { 0.0 };
                    max_inv_resid = max_inv_resid.max((ainv - eye).abs());
                    max_chol_resid = max_chol_resid.max((llt - a_sub[i * n + j]).abs());
                }
            }
            let mut trace = 0.0;
            let mut eig_sum = 0.0;
            for i in 0..n {
                trace += a_sub[i * n + i];
                eig_sum += eig_sub[i];
            }
            max_eig_resid = max_eig_resid.max((eig_sum - trace).abs());
        }
        // Tolerances sit ~3 orders above the observed ~1e-13 residuals, so a
        // genuine regression trips them long before the digest assert fires.
        assert!(
            max_inv_resid < 1e-10,
            "A·inv residual {max_inv_resid:e} exceeds tolerance — real inv regression, do not re-pin"
        );
        assert!(
            max_chol_resid < 1e-9,
            "L·Lᵀ residual {max_chol_resid:e} exceeds tolerance — real cholesky regression, do not re-pin"
        );
        assert!(
            max_eig_resid < 1e-9,
            "Σλ vs trace residual {max_eig_resid:e} exceeds tolerance — real eigvalsh regression, do not re-pin"
        );

        // Golden SHA-256 over the concatenated little-endian output bits pins
        // the exact numeric result against future refactors.
        let mut hasher = Sha256::new();
        for v in inv_out.iter().chain(&chol_out).chain(&eig_out) {
            hasher.update(v.to_bits().to_le_bytes());
        }
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        // Golden re-pinned 2026-06-19: eigvalsh's eigenvalues-only QR chase moved to
        // `scaled_hypot` (benign ~1 ulp/rotation), shifting the eig_out bits. Verified
        // benign by the residual asserts above (all ~1e-13, far inside tolerance) and
        // the per-lane parallel==serial bit-identity asserts. Prior pin
        // 34aaf43d (2026-06-17) predated the scaled_hypot lever; c4213c22 (2026-06-15)
        // predated the blocked-tridiag lever.
        assert_eq!(
            digest, "5fa28f4d0c627ca64d3f5da7f1f355dc690878377898af89ebbf278bea9f2b86",
            "batch parallel golden digest drifted: {digest}"
        );
    }

    #[test]
    fn mat_mul_flat_row_parallel_matches_serial_reference_and_golden_sha256() {
        // matrix_power_nxn(a, 3) drives the internal square GEMM (mat_mul_flat).
        // At n = 128 (>= MATMUL_PARALLEL_MIN_DIM), the local rayon pool below
        // forces the row-partition path even if the process-global pool is
        // single-threaded. Because each output row is computed from a disjoint
        // slice of `c` with the identical k/j accumulation order as the serial
        // loop, the parallel result must equal a hand-written serial reference
        // bit-for-bit.
        let n = 128usize;

        // Deterministic LCG fill (same generator as the criterion bench).
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let a: Vec<f64> = (0..n * n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
            })
            .collect();

        // Serial reference: the pre-parallelization ikj kernel, plus an exact
        // replica of matrix_power_nxn's repeated-squaring schedule for p = 3.
        fn serial_gemm(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
            let mut c = vec![0.0; n * n];
            for i in 0..n {
                for k in 0..n {
                    let a_ik = a[i * n + k];
                    for j in 0..n {
                        c[i * n + j] += a_ik * b[k * n + j];
                    }
                }
            }
            c
        }
        let serial_ref = {
            let mut exp = 3u64;
            let mut result = vec![0.0; n * n];
            for i in 0..n {
                result[i * n + i] = 1.0;
            }
            let mut cur = a.clone();
            while exp > 0 {
                if exp & 1 == 1 {
                    result = serial_gemm(&result, &cur, n);
                }
                exp >>= 1;
                if exp > 0 {
                    cur = serial_gemm(&cur, &cur, n);
                }
            }
            result
        };

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("build local rayon pool");
        let parallel_out = pool.install(|| matrix_power_nxn(&a, n, 3).unwrap());
        assert_eq!(parallel_out.len(), serial_ref.len());
        for (p, s) in parallel_out.iter().zip(&serial_ref) {
            assert_eq!(
                p.to_bits(),
                s.to_bits(),
                "mat_mul_flat row-parallel drifted"
            );
        }

        // Golden SHA-256 over the little-endian output bits pins the exact
        // numeric result against future refactors.
        let mut hasher = Sha256::new();
        for v in &parallel_out {
            hasher.update(v.to_bits().to_le_bytes());
        }
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "99adb17b3a1e6490bc5d3dc5d19e0ec5fd5bfb185384bd3e75f1647061f5920e",
            "mat_mul_flat parallel golden digest drifted"
        );
    }

    #[test]
    fn packed_gemm_sub_assign_matches_materialized_product_sha256() {
        let (m, k, n) = (137usize, 130usize, 139usize);
        let mut state: u64 = 0xB10C_AB1E_5EED_0101;
        let mut fill = |len: usize| -> Vec<f64> {
            (0..len)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
                })
                .collect()
        };
        let a = fill(m * k);
        let b = fill(k * n);
        let seed_target = fill(m * n);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("build local rayon pool");
        let product = pool.install(|| super::packed_gemm(&a, &b, m, k, n));
        let mut materialized = seed_target.clone();
        for (cell, &value) in materialized.iter_mut().zip(&product) {
            *cell -= value;
        }

        let mut fused = seed_target;
        pool.install(|| super::packed_gemm_sub_assign(&a, &b, m, k, n, &mut fused));
        assert_eq!(fused.len(), materialized.len());
        for (fused_value, materialized_value) in fused.iter().zip(&materialized) {
            assert_eq!(
                fused_value.to_bits(),
                materialized_value.to_bits(),
                "fused GEMM subtract changed materialized-product bits"
            );
        }

        let mut hasher = Sha256::new();
        hasher.update(m.to_le_bytes());
        hasher.update(k.to_le_bytes());
        hasher.update(n.to_le_bytes());
        for value in &fused {
            hasher.update(value.to_bits().to_le_bytes());
        }
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "3042ee3757dbb9aec00c16b0932ea42dd3501b500a773675a142dd3709fe310f",
            "packed_gemm_sub_assign golden digest drifted: {digest}"
        );
    }

    #[test]
    fn packed_gemm_sub_assign_strided_matches_materialized_product_sha256() {
        let (m, k, n, row_stride) = (137usize, 130usize, 139usize, 173usize);
        let mut state: u64 = 0x5172_1DED_5EED_0101;
        let mut fill = |len: usize| -> Vec<f64> {
            (0..len)
                .map(|_| {
                    state = state
                        .wrapping_mul(2862933555777941757)
                        .wrapping_add(3037000493);
                    ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
                })
                .collect()
        };
        let a = fill(m * k);
        let b = fill(k * n);
        let target_len = (m - 1) * row_stride + n;
        let seed_target = fill(target_len);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("build local rayon pool");
        let product = pool.install(|| super::packed_gemm(&a, &b, m, k, n));
        let mut materialized = seed_target.clone();
        for i in 0..m {
            let dst = i * row_stride;
            let prod_row = &product[i * n..i * n + n];
            for (cell, &value) in materialized[dst..dst + n].iter_mut().zip(prod_row) {
                *cell -= value;
            }
        }

        let mut fused = seed_target;
        pool.install(|| {
            super::packed_gemm_sub_assign_strided(&a, &b, m, k, n, row_stride, &mut fused)
        });
        assert_eq!(fused.len(), materialized.len());
        for (idx, (fused_value, materialized_value)) in
            fused.iter().zip(&materialized).enumerate()
        {
            assert_eq!(
                fused_value.to_bits(),
                materialized_value.to_bits(),
                "strided fused GEMM subtract changed materialized-product bits at index {idx}"
            );
        }

        let mut hasher = Sha256::new();
        hasher.update(m.to_le_bytes());
        hasher.update(k.to_le_bytes());
        hasher.update(n.to_le_bytes());
        hasher.update(row_stride.to_le_bytes());
        for value in &fused {
            hasher.update(value.to_bits().to_le_bytes());
        }
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "8ec7fce06fab4782db37a4e023dae9f750e5f454d2a33e2363bb3de542973c9b",
            "packed_gemm_sub_assign_strided golden digest drifted: {digest}"
        );
    }

    #[test]
    fn mat_mul_rect_row_parallel_matches_serial_reference_and_golden_sha256() {
        // multi_dot of two matrices dispatches straight to mat_mul_rect. Use a
        // genuinely rectangular shape with every dim >= MATMUL_PARALLEL_MIN_DIM
        // so the rayon row-partition path runs and the rectangular a/b/c index
        // arithmetic is exercised. Row-disjoint output + identical p/j order =>
        // bit-for-bit identical to a serial reference.
        let (m, k, n) = (130usize, 128usize, 129usize);
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        let mut fill = |len: usize| -> Vec<f64> {
            (0..len)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
                })
                .collect()
        };
        let a = fill(m * k);
        let b = fill(k * n);

        // Serial reference: the pre-parallelization ikj rectangular kernel.
        let mut serial_ref = vec![0.0; m * n];
        for i in 0..m {
            for p in 0..k {
                let a_ip = a[i * k + p];
                for j in 0..n {
                    serial_ref[i * n + j] += a_ip * b[p * n + j];
                }
            }
        }

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("build local rayon pool");
        let (out, out_m, out_n) =
            pool.install(|| multi_dot(&[(a.as_slice(), m, k), (b.as_slice(), k, n)]).unwrap());
        assert_eq!((out_m, out_n), (m, n));
        assert_eq!(out.len(), serial_ref.len());
        for (p, s) in out.iter().zip(&serial_ref) {
            assert_eq!(
                p.to_bits(),
                s.to_bits(),
                "mat_mul_rect row-parallel drifted"
            );
        }

        let mut hasher = Sha256::new();
        for v in &out {
            hasher.update(v.to_bits().to_le_bytes());
        }
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "9016722803dc4256abe1d21b2d31c0da7a85afde89098848195f1a52cf677897",
            "mat_mul_rect parallel golden digest drifted"
        );
    }

    #[test]
    fn complex_matmul_row_parallel_matches_serial_reference_and_golden_sha256() {
        // Interleaved complex GEMM with all dims >= MATMUL_PARALLEL_MIN_DIM so
        // the rayon row-partition path runs. Row-disjoint output + preserved
        // per-element p-reduction order => bit-for-bit identical to serial.
        let (m, k, n) = (128usize, 128usize, 128usize);
        let mut state: u64 = 0xCAFE_F00D_1234_5678;
        let mut fill = |len: usize| -> Vec<f64> {
            (0..len)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
                })
                .collect()
        };
        let a = fill(2 * m * k);
        let b = fill(2 * k * n);

        // Serial reference: the pre-parallelization ijp kernel.
        let mut serial_ref = vec![0.0_f64; 2 * m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sr = 0.0f64;
                let mut si = 0.0f64;
                for p in 0..k {
                    let (ar, ai) = (a[2 * (i * k + p)], a[2 * (i * k + p) + 1]);
                    let (br, bi) = (b[2 * (p * n + j)], b[2 * (p * n + j) + 1]);
                    let (pr, pi) = cmul(ar, ai, br, bi);
                    sr += pr;
                    si += pi;
                }
                serial_ref[2 * (i * n + j)] = sr;
                serial_ref[2 * (i * n + j) + 1] = si;
            }
        }

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("build local rayon pool");
        let out = pool.install(|| complex_matmul(&a, &b, m, k, n));
        assert_eq!(out.len(), serial_ref.len());
        for (p, s) in out.iter().zip(&serial_ref) {
            assert_eq!(
                p.to_bits(),
                s.to_bits(),
                "complex_matmul row-parallel drifted"
            );
        }

        let mut hasher = Sha256::new();
        for v in &out {
            hasher.update(v.to_bits().to_le_bytes());
        }
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "b2ce594b8b5b6da625364c07b32ac1a8c4bc09275b0291c425497ceb44ab2eb6",
            "complex_matmul parallel golden digest drifted"
        );
    }

    #[test]
    fn complex_matmul_packed_path_matches_serial_reference_and_golden_sha256() {
        // Dimensions deliberately cross COMPLEX_PACKED_MIN_FLOPS and include
        // multiple packed B panels, while keeping the serial reference bounded.
        let (m, k, n) = (64usize, 256usize, 512usize);
        let mut state: u64 = 0xF00D_BAAD_D15C_A11E;
        let mut fill = |len: usize| -> Vec<f64> {
            (0..len)
                .map(|idx| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    match idx % 17 {
                        0 => 0.0,
                        1 => -0.0,
                        _ => ((state >> 33) as f64) / (u32::MAX as f64) - 0.5,
                    }
                })
                .collect()
        };
        let a = fill(2 * m * k);
        let b = fill(2 * k * n);

        let mut serial_ref = vec![0.0_f64; 2 * m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sr = 0.0f64;
                let mut si = 0.0f64;
                for p in 0..k {
                    let (ar, ai) = (a[2 * (i * k + p)], a[2 * (i * k + p) + 1]);
                    let (br, bi) = (b[2 * (p * n + j)], b[2 * (p * n + j) + 1]);
                    let (pr, pi) = cmul(ar, ai, br, bi);
                    sr += pr;
                    si += pi;
                }
                serial_ref[2 * (i * n + j)] = sr;
                serial_ref[2 * (i * n + j) + 1] = si;
            }
        }

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("build local rayon pool");
        let out = pool.install(|| complex_matmul(&a, &b, m, k, n));
        assert_eq!(out.len(), serial_ref.len());
        for (p, s) in out.iter().zip(&serial_ref) {
            assert_eq!(
                p.to_bits(),
                s.to_bits(),
                "complex_matmul packed path drifted"
            );
        }

        let mut hasher = Sha256::new();
        for v in &out {
            hasher.update(v.to_bits().to_le_bytes());
        }
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "71af9c98bd94b4c56cf5375e77c1d081316283aa448cfd47f615ec8f5c13d3f8",
            "complex_matmul packed-path golden digest drifted: {digest}"
        );
    }

    fn legacy_validate_matrix_shape(shape: &[usize]) -> Result<(usize, usize), LinAlgError> {
        if shape.len() < 2 {
            return Err(LinAlgError::ShapeContractViolation(
                "linalg input must be at least 2D",
            ));
        }

        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];
        if rows == 0 || cols == 0 {
            return Err(LinAlgError::ShapeContractViolation(
                "matrix rows/cols must be non-zero",
            ));
        }

        let batch_lanes = shape[..shape.len() - 2]
            .iter()
            .copied()
            .try_fold(1usize, |acc, dim| acc.checked_mul(dim))
            .ok_or(LinAlgError::ShapeContractViolation(
                "batch lane multiplication overflowed",
            ))?;

        if batch_lanes > MAX_BATCH_SHAPE_CHECKS {
            return Err(LinAlgError::ShapeContractViolation(
                "batch lanes exceeded bounded validation budget",
            ));
        }

        Ok((rows, cols))
    }

    #[test]
    fn lstsq_rank_deficient_repro() {
        // A = [[1, 1], [1, 1]], b = [2, 2]
        // Rank 1, exact solution x = [1, 1]
        let a = [1.0, 1.0, 1.0, 1.0];
        let b = [2.0, 2.0];
        let x = lstsq_nxn(&a, &b, 2, 2).expect("SVD lstsq should handle rank-deficient");
        // Solution should satisfy A*x = b. [1, 1] * [1, 1]^T = 2.
        assert!((x[0] + x[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn pinv_rank_deficient_repro() {
        // A = [[1, 1], [1, 1]]
        // pinv(A) = [[0.25, 0.25], [0.25, 0.25]]
        let a = [1.0, 1.0, 1.0, 1.0];
        let pinv = pinv_nxn(&a, 2, 2).expect("SVD pinv should handle rank-deficient");
        for &val in &pinv {
            assert!((val - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn reason_code_registry_matches_packet_contract() {
        assert_eq!(
            LINALG_PACKET_REASON_CODES,
            [
                "linalg_shape_contract_violation",
                "linalg_solver_singularity",
                "linalg_cholesky_contract_violation",
                "linalg_qr_mode_invalid",
                "linalg_svd_nonconvergence",
                "linalg_spectral_convergence_failed",
                "linalg_lstsq_tuple_contract_violation",
                "linalg_norm_det_rank_policy_violation",
                "linalg_backend_bridge_invalid",
                "linalg_policy_unknown_metadata",
            ]
        );
    }

    #[test]
    fn matrix_shape_accepts_batched_square_shapes() {
        assert_eq!(
            validate_matrix_shape(&[8, 16, 3, 3]).expect("batched matrix"),
            (3, 3)
        );
        assert_eq!(
            validate_square_matrix(&[4, 4]).expect("square matrix"),
            4usize
        );
    }

    #[test]
    fn matrix_shape_rejects_invalid_rank_or_budget() {
        let err = validate_matrix_shape(&[4]).expect_err("rank<2 should fail");
        assert_eq!(err.reason_code(), "linalg_shape_contract_violation");

        let huge_batch = [MAX_TOLERANCE_SEARCH_DEPTH * 20_000usize, 2usize, 2usize];
        let err = validate_matrix_shape(&huge_batch).expect_err("batch budget should fail");
        assert_eq!(err.reason_code(), "linalg_shape_contract_violation");
    }

    #[test]
    fn matrix_shape_fast_paths_are_isomorphic_with_legacy_path() {
        let fixtures: [&[usize]; 13] = [
            &[2, 2],
            &[4, 4],
            &[1, 2, 2],
            &[8, 2, 2],
            &[7, 9, 3, 3],
            &[3, 5, 7, 11, 2, 2],
            &[4],
            &[0, 2],
            &[2, 0],
            &[MAX_BATCH_SHAPE_CHECKS + 1, 2, 2],
            &[usize::MAX, usize::MAX, 2, 2],
            &[usize::MAX, 1, 2, 2],
            &[1, usize::MAX, 2, 2],
        ];

        for shape in fixtures {
            let baseline = legacy_validate_matrix_shape(shape);
            let optimized = validate_matrix_shape(shape);
            assert_eq!(optimized, baseline, "shape={shape:?}");
        }
    }

    #[test]
    fn square_validation_rejects_non_square() {
        let err = validate_square_matrix(&[2, 3]).expect_err("non-square should fail");
        assert_eq!(err.reason_code(), "linalg_shape_contract_violation");
    }

    #[test]
    fn solve_2x2_reconstructs_rhs_across_seed_grid() {
        for seed in 1_u32..=256_u32 {
            let alpha = f64::from((seed % 19) + 2);
            let beta = f64::from((seed % 11) + 3);
            let matrix = [[alpha + 5.0, 1.0], [1.0, beta + 4.0]];
            let expected = [f64::from(seed) / 7.0, f64::from(seed) / 11.0];
            let rhs = [
                matrix[0][0] * expected[0] + matrix[0][1] * expected[1],
                matrix[1][0] * expected[0] + matrix[1][1] * expected[1],
            ];

            let solved = solve_2x2(matrix, rhs).expect("non-singular solve");
            assert!(
                approx_equal(solved[0], expected[0], 1e-10),
                "fixture_id=UP-008-solve-inv-contract seed={seed} lhs={} rhs={}",
                solved[0],
                expected[0]
            );
            assert!(
                approx_equal(solved[1], expected[1], 1e-10),
                "fixture_id=UP-008-solve-inv-contract seed={seed} lhs={} rhs={}",
                solved[1],
                expected[1]
            );
        }
    }

    #[test]
    fn solve_2x2_rejects_singular_system() {
        let err = solve_2x2([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])
            .expect_err("singular matrix should fail");
        assert_eq!(err.reason_code(), "linalg_solver_singularity");
    }

    #[test]
    fn det_and_slogdet_are_deterministic() {
        let matrix = [[4.0, 7.0], [2.0, 6.0]];
        let det = det_2x2(matrix).expect("det");
        assert!(approx_equal(det, 10.0, 1e-12));

        let (sign, log_abs_det) = slogdet_2x2(matrix).expect("slogdet");
        assert!(approx_equal(sign, 1.0, 1e-12));
        assert!(approx_equal(log_abs_det, 10.0_f64.ln(), 1e-12));

        let singular = [[1.0, 2.0], [2.0, 4.0]];
        let (sign, log_abs_det) = slogdet_2x2(singular).expect("singular slogdet");
        assert_eq!(sign, 0.0);
        assert_eq!(log_abs_det, f64::NEG_INFINITY);
    }

    #[test]
    fn det_and_slogdet_match_numpy_non_finite_semantics() {
        let nan_diag = [[f64::NAN, 1.0], [2.0, 3.0]];
        assert!(det_2x2(nan_diag).expect("nan det").is_nan());
        let (sign, log_abs_det) = slogdet_2x2(nan_diag).expect("nan slogdet");
        assert!(approx_equal(sign, 1.0, 1e-12));
        assert!(log_abs_det.is_nan());

        let nan_offdiag = [[1.0, f64::NAN], [2.0, 3.0]];
        assert!(det_2x2(nan_offdiag).expect("nan det offdiag").is_nan());
        let (sign, log_abs_det) = slogdet_2x2(nan_offdiag).expect("nan slogdet offdiag");
        assert!(approx_equal(sign, -1.0, 1e-12));
        assert!(log_abs_det.is_nan());

        let inf_diag = [[f64::INFINITY, 1.0], [2.0, 3.0]];
        assert!(det_2x2(inf_diag).expect("inf det").is_infinite());
        let (sign, log_abs_det) = slogdet_2x2(inf_diag).expect("inf slogdet");
        assert!(approx_equal(sign, 1.0, 1e-12));
        assert!(log_abs_det.is_infinite() && log_abs_det.is_sign_positive());

        let inf_offdiag = [[1.0, f64::INFINITY], [2.0, 3.0]];
        assert!(det_2x2(inf_offdiag).expect("inf det offdiag").is_infinite());
        let (sign, log_abs_det) = slogdet_2x2(inf_offdiag).expect("inf slogdet offdiag");
        assert!(approx_equal(sign, -1.0, 1e-12));
        assert!(log_abs_det.is_infinite() && log_abs_det.is_sign_positive());

        let nan_nxn = [f64::NAN, 1.0, 0.0, 0.0, 3.0, 1.0, 2.0, 0.0, 3.0];
        assert!(det_nxn(&nan_nxn, 3).expect("nan det nxn").is_nan());
        let (sign, log_abs_det) = slogdet_nxn(&nan_nxn, 3).expect("nan slogdet nxn");
        assert!(approx_equal(sign, 1.0, 1e-12));
        assert!(log_abs_det.is_nan());

        let inf_nxn = [f64::INFINITY, 1.0, 0.0, 0.0, 3.0, 1.0, 2.0, 0.0, 3.0];
        assert!(det_nxn(&inf_nxn, 3).expect("inf det nxn").is_infinite());
        let (sign, log_abs_det) = slogdet_nxn(&inf_nxn, 3).expect("inf slogdet nxn");
        assert!(approx_equal(sign, 1.0, 1e-12));
        assert!(log_abs_det.is_infinite() && log_abs_det.is_sign_positive());
    }

    #[test]
    fn inv_2x2_matches_identity_reconstruction() {
        let matrix = [[4.0, 7.0], [2.0, 6.0]];
        let inv = inv_2x2(matrix).expect("inverse");
        let m00 = matrix[0][0].mul_add(inv[0][0], matrix[0][1] * inv[1][0]);
        let m01 = matrix[0][0].mul_add(inv[0][1], matrix[0][1] * inv[1][1]);
        let m10 = matrix[1][0].mul_add(inv[0][0], matrix[1][1] * inv[1][0]);
        let m11 = matrix[1][0].mul_add(inv[0][1], matrix[1][1] * inv[1][1]);
        assert!(approx_equal(m00, 1.0, 1e-12));
        assert!(approx_equal(m01, 0.0, 1e-12));
        assert!(approx_equal(m10, 0.0, 1e-12));
        assert!(approx_equal(m11, 1.0, 1e-12));

        let err = inv_2x2([[1.0, 2.0], [2.0, 4.0]]).expect_err("singular inverse");
        assert_eq!(err.reason_code(), "linalg_solver_singularity");
    }

    #[test]
    fn matrix_rank_2x2_detects_rank_profiles() {
        let full_rank = matrix_rank_2x2([[3.0, 1.0], [2.0, 4.0]], 1e-12).expect("rank");
        assert_eq!(full_rank, 2);

        let rank_one = matrix_rank_2x2([[1.0, 2.0], [2.0, 4.0]], 1e-12).expect("rank");
        assert_eq!(rank_one, 1);

        let rank_zero = matrix_rank_2x2([[0.0, 0.0], [0.0, 0.0]], 1e-12).expect("rank");
        assert_eq!(rank_zero, 0);
    }

    #[test]
    fn matrix_rank_matches_numpy_non_finite_semantics() {
        assert_eq!(
            matrix_rank_2x2([[f64::INFINITY, 1.0], [2.0, 3.0]], 1e-12).expect("inf rank"),
            0
        );
        assert_eq!(
            matrix_rank_nxn(&[f64::INFINITY, 1.0, 2.0, 3.0], 2, 1e-12).expect("inf rank nxn"),
            0
        );

        let err = matrix_rank_2x2([[f64::NAN, 1.0], [2.0, 3.0]], 1e-12).expect_err("nan rank");
        assert_eq!(err.reason_code(), "linalg_norm_det_rank_policy_violation");
        let err = matrix_rank_nxn(&[f64::NAN, 1.0, 2.0, 3.0], 2, 1e-12).expect_err("nan rank nxn");
        assert_eq!(err.reason_code(), "linalg_norm_det_rank_policy_violation");
    }

    #[test]
    fn matrix_rank_tol_matches_numpy_default_and_explicit_cases() {
        let diag2_small = [1.0, 0.0, 0.0, 1e-18];
        assert_eq!(
            matrix_rank_2x2_tol([[1.0, 0.0], [0.0, 1e-18]], None).unwrap(),
            1
        );
        assert_eq!(
            matrix_rank_2x2_tol([[1.0, 0.0], [0.0, 1e-18]], Some(0.0)).unwrap(),
            2
        );
        assert_eq!(
            matrix_rank_2x2_tol([[1.0, 0.0], [0.0, 1e-18]], Some(1e-18)).unwrap(),
            1
        );
        assert_eq!(
            matrix_rank_mxn_tol(&diag2_small, 2, 2, Some(1e-17)).unwrap(),
            1
        );

        let diag3 = [1.0, 0.0, 0.0, 0.0, 1e-14, 0.0, 0.0, 0.0, 1e-16];
        assert_eq!(matrix_rank_nxn_tol(&diag3, 3, None).unwrap(), 2);
        assert_eq!(matrix_rank_nxn_tol(&diag3, 3, Some(0.0)).unwrap(), 3);
        assert_eq!(matrix_rank_nxn_tol(&diag3, 3, Some(1e-13)).unwrap(), 1);
    }

    #[test]
    fn matrix_rank_tol_rejects_invalid_tolerances() {
        let err =
            matrix_rank_2x2_tol([[1.0, 0.0], [0.0, 1.0]], Some(-1.0)).expect_err("negative tol");
        assert_eq!(err.reason_code(), "linalg_norm_det_rank_policy_violation");

        let err =
            matrix_rank_nxn_tol(&[1.0, 0.0, 0.0, 1.0], 2, Some(f64::NAN)).expect_err("nan tol");
        assert_eq!(err.reason_code(), "linalg_norm_det_rank_policy_violation");
    }

    #[test]
    fn pinv_2x2_full_rank_and_rank_deficient_paths() {
        let matrix = [[4.0, 7.0], [2.0, 6.0]];
        let pinv = pinv_2x2(matrix, 1e-12).expect("pinv");
        let inv = inv_2x2(matrix).expect("inv");
        for row in 0..2 {
            for col in 0..2 {
                assert!(approx_equal(pinv[row][col], inv[row][col], 1e-10));
            }
        }

        let rank_def = [[1.0, 2.0], [2.0, 4.0]];
        let pinv_rank_def = pinv_2x2(rank_def, 1e-12).expect("rank-def pinv");
        let expected = [[1.0 / 25.0, 2.0 / 25.0], [2.0 / 25.0, 4.0 / 25.0]];
        for row in 0..2 {
            for col in 0..2 {
                assert!(approx_equal(
                    pinv_rank_def[row][col],
                    expected[row][col],
                    1e-10
                ));
            }
        }
    }

    #[test]
    fn pinv_tolerance_aliases_match_numpy_conflict_and_cutoff_semantics() {
        let matrix = [[1.0, 0.0], [0.0, 1e-12]];

        let default = pinv_2x2_with_tolerance_aliases(matrix, None, None).expect("default");
        let explicit_none =
            pinv_2x2_with_tolerance_aliases(matrix, None, Some(None)).expect("rtol none");
        assert!(approx_equal(default[0][0], 1.0, 1e-12));
        assert!(approx_equal(default[1][1], 1e12, 1.0));
        assert_eq!(default, explicit_none);

        let cutoff =
            pinv_2x2_with_tolerance_aliases(matrix, None, Some(Some(1e-9))).expect("rtol cutoff");
        assert!(approx_equal(cutoff[0][0], 1.0, 1e-12));
        assert!(approx_equal(cutoff[1][1], 0.0, 1e-12));

        let negative = pinv_2x2(matrix, -1.0).expect("negative rcond");
        assert_eq!(negative, default);

        let nan_rcond = pinv_2x2(matrix, f64::NAN).expect("nan rcond");
        assert!(approx_equal(nan_rcond[0][0], 0.0, 1e-12));
        assert!(approx_equal(nan_rcond[1][1], 0.0, 1e-12));

        let err = pinv_2x2_with_tolerance_aliases(matrix, Some(1e-9), Some(None))
            .expect_err("both rcond and rtol");
        assert_eq!(format!("{err}"), "`rtol` and `rcond` can't be both set.");

        let mxn_default =
            pinv_mxn_with_tolerance_aliases(&[1.0, 0.0, 0.0, 1e-12], 2, 2, None, None)
                .expect("mxn default");
        assert!(approx_equal(mxn_default[0], 1.0, 1e-12));
        assert!(approx_equal(mxn_default[3], 1e12, 1.0));
    }

    #[test]
    fn pinv_hermitian_uses_lower_triangle_like_numpy() {
        let matrix = [2.0, 4.0, 1.0, 2.0];
        let pinv = pinv_hermitian_nxn(&matrix, 2, -1.0).expect("hermitian pinv");
        let expected = [2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0];
        for (index, (&got, &want)) in pinv.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_equal(got, want, 1e-12),
                "pinv[{index}] mismatch: got {got}, want {want}"
            );
        }

        let cutoff =
            pinv_hermitian_nxn_with_tolerance_aliases(&[1.0, 0.0, 0.0, 1e-12], 2, Some(1e-9), None)
                .expect("hermitian cutoff");
        assert!(approx_equal(cutoff[0], 1.0, 1e-12));
        assert!(approx_equal(cutoff[3], 0.0, 1e-12));
    }

    #[test]
    fn rank_and_pinv_still_reject_non_finite_inputs() {
        let err = pinv_2x2([[f64::INFINITY, 0.0], [0.0, 1.0]], 1e-12).expect_err("inf matrix");
        assert_eq!(err.reason_code(), "linalg_norm_det_rank_policy_violation");
    }

    #[test]
    fn norm_order_token_parsers_are_fail_closed() {
        assert_eq!(
            VectorNormOrder::from_token("1").expect("vector one"),
            VectorNormOrder::One
        );
        assert_eq!(
            VectorNormOrder::from_token("-inf").expect("vector neginf"),
            VectorNormOrder::NegInf
        );
        assert_eq!(
            MatrixNormOrder::from_token("fro").expect("matrix fro"),
            MatrixNormOrder::Fro
        );
        assert_eq!(
            MatrixNormOrder::from_token("nuc").expect("matrix nuc"),
            MatrixNormOrder::Nuclear
        );

        let err = VectorNormOrder::from_token("hostile").expect_err("vector token should fail");
        assert_eq!(err.reason_code(), "linalg_norm_det_rank_policy_violation");
        let err = MatrixNormOrder::from_token("hostile").expect_err("matrix token should fail");
        assert_eq!(err.reason_code(), "linalg_norm_det_rank_policy_violation");
    }

    #[test]
    fn vector_norm_orders_match_first_wave_contracts() {
        let values = [3.0, -4.0];
        assert!(approx_equal(
            vector_norm(&values, None).expect("default l2"),
            5.0,
            1e-12
        ));
        assert!(approx_equal(
            vector_norm(&values, Some(VectorNormOrder::One)).expect("l1"),
            7.0,
            1e-12
        ));
        assert!(approx_equal(
            vector_norm(&values, Some(VectorNormOrder::Inf)).expect("inf"),
            4.0,
            1e-12
        ));
        assert!(approx_equal(
            vector_norm(&values, Some(VectorNormOrder::NegInf)).expect("-inf"),
            3.0,
            1e-12
        ));

        assert!(approx_equal(
            vector_norm(&[], None).expect("empty default"),
            0.0,
            1e-12
        ));
        let err = vector_norm(&[], Some(VectorNormOrder::NegInf)).expect_err("empty -inf");
        assert_eq!(err.reason_code(), "linalg_norm_det_rank_policy_violation");

        // Empty input with a negative finite order matches NumPy: 0^(1/ord) = +inf.
        // np.linalg.norm([], -1) == np.linalg.norm([], -2) == np.linalg.norm([], -0.5) == inf
        assert_eq!(
            vector_norm(&[], Some(VectorNormOrder::P(-1.0))).expect("empty p=-1"),
            f64::INFINITY
        );
        assert_eq!(
            vector_norm(&[], Some(VectorNormOrder::P(-2.0))).expect("empty p=-2"),
            f64::INFINITY
        );
        assert_eq!(
            vector_norm(&[], Some(VectorNormOrder::P(-0.5))).expect("empty p=-0.5"),
            f64::INFINITY
        );
        // Positive finite order over empty stays 0.0, like NumPy.
        assert_eq!(
            vector_norm(&[], Some(VectorNormOrder::P(3.0))).expect("empty p=3"),
            0.0
        );
    }

    #[test]
    fn matrix_norm_orders_match_first_wave_contracts() {
        let matrix = [[1.0, 2.0], [3.0, 4.0]];
        assert!(approx_equal(
            matrix_norm_2x2(matrix, None).expect("default fro"),
            5.477225575051661,
            1e-12
        ));
        assert!(approx_equal(
            matrix_norm_2x2(matrix, Some(MatrixNormOrder::One)).expect("one"),
            6.0,
            1e-12
        ));
        assert!(approx_equal(
            matrix_norm_2x2(matrix, Some(MatrixNormOrder::Inf)).expect("inf"),
            7.0,
            1e-12
        ));
        assert!(approx_equal(
            matrix_norm_2x2(matrix, Some(MatrixNormOrder::Two)).expect("two"),
            5.464985704219043,
            1e-12
        ));
        assert!(approx_equal(
            matrix_norm_2x2(matrix, Some(MatrixNormOrder::NegTwo)).expect("-two"),
            0.36596619062625746,
            1e-12
        ));
        assert!(approx_equal(
            matrix_norm_2x2(matrix, Some(MatrixNormOrder::Nuclear)).expect("nuclear"),
            5.8309518948453,
            1e-12
        ));
    }

    #[test]
    fn vector_norm_matches_numpy_non_finite_semantics() {
        assert!(
            vector_norm(&[f64::NAN, 1.0], None)
                .expect("vector nan default")
                .is_nan()
        );
        assert!(
            vector_norm(&[f64::NAN, 1.0], Some(VectorNormOrder::One))
                .expect("vector nan l1")
                .is_nan()
        );
        assert!(
            vector_norm(&[f64::NAN, 1.0], Some(VectorNormOrder::Two))
                .expect("vector nan l2")
                .is_nan()
        );
        assert!(
            vector_norm(&[f64::NAN, 1.0], Some(VectorNormOrder::Inf))
                .expect("vector nan inf")
                .is_nan()
        );
        assert!(
            vector_norm(&[f64::NAN, 1.0], Some(VectorNormOrder::NegInf))
                .expect("vector nan neginf")
                .is_nan()
        );
        assert!(approx_equal(
            vector_norm(&[f64::NAN, 1.0], Some(VectorNormOrder::Zero)).expect("vector nan zero"),
            2.0,
            1e-12
        ));

        assert!(
            vector_norm(&[f64::INFINITY, 1.0], None)
                .expect("vector inf default")
                .is_infinite()
        );
        assert!(
            vector_norm(&[f64::INFINITY, 1.0], Some(VectorNormOrder::One))
                .expect("vector inf l1")
                .is_infinite()
        );
        assert!(
            vector_norm(&[f64::INFINITY, 1.0], Some(VectorNormOrder::Two))
                .expect("vector inf l2")
                .is_infinite()
        );
        assert!(
            vector_norm(&[f64::INFINITY, 1.0], Some(VectorNormOrder::Inf))
                .expect("vector inf inf")
                .is_infinite()
        );
        assert!(approx_equal(
            vector_norm(&[f64::INFINITY, 1.0], Some(VectorNormOrder::NegInf))
                .expect("vector inf neginf"),
            1.0,
            1e-12
        ));
        assert!(approx_equal(
            vector_norm(&[f64::INFINITY, 1.0], Some(VectorNormOrder::Zero))
                .expect("vector inf zero"),
            2.0,
            1e-12
        ));
    }

    #[test]
    fn matrix_norm_matches_numpy_non_finite_semantics() {
        let nan_matrix = [[f64::NAN, 1.0], [2.0, 3.0]];
        assert!(matrix_norm_2x2(nan_matrix, None).expect("nan fro").is_nan());
        assert!(
            matrix_norm_2x2(nan_matrix, Some(MatrixNormOrder::One))
                .expect("nan one")
                .is_nan()
        );
        assert!(
            matrix_norm_2x2(nan_matrix, Some(MatrixNormOrder::NegOne))
                .expect("nan neg one")
                .is_nan()
        );
        assert!(
            matrix_norm_2x2(nan_matrix, Some(MatrixNormOrder::Inf))
                .expect("nan inf")
                .is_nan()
        );
        assert!(
            matrix_norm_2x2(nan_matrix, Some(MatrixNormOrder::NegInf))
                .expect("nan neg inf")
                .is_nan()
        );
        let err = matrix_norm_2x2(nan_matrix, Some(MatrixNormOrder::Two)).expect_err("nan two");
        assert_eq!(err.reason_code(), "linalg_svd_nonconvergence");
        let err =
            matrix_norm_2x2(nan_matrix, Some(MatrixNormOrder::Nuclear)).expect_err("nan nuclear");
        assert_eq!(err.reason_code(), "linalg_svd_nonconvergence");

        let inf_matrix = [[f64::INFINITY, 1.0], [2.0, 3.0]];
        assert!(
            matrix_norm_2x2(inf_matrix, None)
                .expect("inf fro")
                .is_infinite()
        );
        assert!(
            matrix_norm_2x2(inf_matrix, Some(MatrixNormOrder::One))
                .expect("inf one")
                .is_infinite()
        );
        assert!(approx_equal(
            matrix_norm_2x2(inf_matrix, Some(MatrixNormOrder::NegOne)).expect("inf neg one"),
            4.0,
            1e-12
        ));
        assert!(
            matrix_norm_2x2(inf_matrix, Some(MatrixNormOrder::Inf))
                .expect("inf inf")
                .is_infinite()
        );
        assert!(approx_equal(
            matrix_norm_2x2(inf_matrix, Some(MatrixNormOrder::NegInf)).expect("inf neg inf"),
            5.0,
            1e-12
        ));
        assert!(
            matrix_norm_2x2(inf_matrix, Some(MatrixNormOrder::Two))
                .expect("inf two")
                .is_nan()
        );
        assert!(
            matrix_norm_2x2(inf_matrix, Some(MatrixNormOrder::NegTwo))
                .expect("inf neg two")
                .is_nan()
        );
        assert!(
            matrix_norm_2x2(inf_matrix, Some(MatrixNormOrder::Nuclear))
                .expect("inf nuclear")
                .is_nan()
        );
    }

    #[test]
    fn matrix_norm_nxn_matches_numpy_non_finite_semantics() {
        let nan_matrix = [f64::NAN, 1.0, 2.0, 3.0];
        assert!(
            matrix_norm_nxn(&nan_matrix, 2, 2, "fro")
                .expect("nan fro")
                .is_nan()
        );
        assert!(
            matrix_norm_nxn(&nan_matrix, 2, 2, "1")
                .expect("nan one")
                .is_nan()
        );
        assert!(
            matrix_norm_nxn(&nan_matrix, 2, 2, "-1")
                .expect("nan neg one")
                .is_nan()
        );
        assert!(
            matrix_norm_nxn(&nan_matrix, 2, 2, "inf")
                .expect("nan inf")
                .is_nan()
        );
        assert!(
            matrix_norm_nxn(&nan_matrix, 2, 2, "-inf")
                .expect("nan neg inf")
                .is_nan()
        );
        let err = matrix_norm_nxn(&nan_matrix, 2, 2, "2").expect_err("nan two");
        assert_eq!(err.reason_code(), "linalg_svd_nonconvergence");
        let err = matrix_norm_nxn(&nan_matrix, 2, 2, "nuc").expect_err("nan nuclear");
        assert_eq!(err.reason_code(), "linalg_svd_nonconvergence");

        let inf_matrix = [f64::INFINITY, 1.0, 2.0, 3.0];
        assert!(
            matrix_norm_nxn(&inf_matrix, 2, 2, "fro")
                .expect("inf fro")
                .is_infinite()
        );
        assert!(
            matrix_norm_nxn(&inf_matrix, 2, 2, "1")
                .expect("inf one")
                .is_infinite()
        );
        assert!(approx_equal(
            matrix_norm_nxn(&inf_matrix, 2, 2, "-1").expect("inf neg one"),
            4.0,
            1e-12
        ));
        assert!(
            matrix_norm_nxn(&inf_matrix, 2, 2, "inf")
                .expect("inf inf")
                .is_infinite()
        );
        assert!(approx_equal(
            matrix_norm_nxn(&inf_matrix, 2, 2, "-inf").expect("inf neg inf"),
            5.0,
            1e-12
        ));
        assert!(
            matrix_norm_nxn(&inf_matrix, 2, 2, "2")
                .expect("inf two")
                .is_nan()
        );
        assert!(
            matrix_norm_nxn(&inf_matrix, 2, 2, "-2")
                .expect("inf neg two")
                .is_nan()
        );
        assert!(
            matrix_norm_nxn(&inf_matrix, 2, 2, "nuc")
                .expect("inf nuclear")
                .is_nan()
        );
    }

    #[test]
    fn eigvals_2x2_preserves_trace_and_determinant_relations() {
        let matrix = [[4.0, 2.0], [1.0, 3.0]];
        let eigvals = eigvals_2x2(matrix, true).expect("eigvals");
        let trace = eigvals[0] + eigvals[1];
        let det = eigvals[0] * eigvals[1];
        assert!(approx_equal(trace, 7.0, 1e-12));
        assert!(approx_equal(det, 10.0, 1e-12));
    }

    #[test]
    fn eigh_2x2_returns_orthonormal_eigenvectors() {
        let matrix = [[2.0, 1.0], [1.0, 2.0]];
        let (eigvals, eigvecs) = eigh_2x2(matrix, "L", true).expect("eigh");
        assert!(approx_equal(eigvals[0], 1.0, 1e-12));
        assert!(approx_equal(eigvals[1], 3.0, 1e-12));

        let dot = eigvecs[0][0] * eigvecs[0][1] + eigvecs[1][0] * eigvecs[1][1];
        assert!(approx_equal(dot, 0.0, 1e-12));
        let n0 = eigvecs[0][0].hypot(eigvecs[1][0]);
        let n1 = eigvecs[0][1].hypot(eigvecs[1][1]);
        assert!(approx_equal(n0, 1.0, 1e-12));
        assert!(approx_equal(n1, 1.0, 1e-12));

        for col in 0..2 {
            let lambda = eigvals[col];
            let v = [eigvecs[0][col], eigvecs[1][col]];
            let av = [
                matrix[0][0].mul_add(v[0], matrix[0][1] * v[1]),
                matrix[1][0].mul_add(v[0], matrix[1][1] * v[1]),
            ];
            assert!(approx_equal(av[0], lambda * v[0], 1e-10));
            assert!(approx_equal(av[1], lambda * v[1], 1e-10));
        }
    }

    #[test]
    fn eigh_2x2_respects_uplo_branch_choice() {
        let matrix = [[2.0, 100.0], [1.0, 2.0]];
        let (eigvals_l, _) = eigh_2x2(matrix, "L", true).expect("L");
        let (eigvals_u, _) = eigh_2x2(matrix, "U", true).expect("U");

        assert!(approx_equal(eigvals_l[0], 1.0, 1e-12));
        assert!(approx_equal(eigvals_l[1], 3.0, 1e-12));
        assert!(approx_equal(eigvals_u[0], -98.0, 1e-10));
        assert!(approx_equal(eigvals_u[1], 102.0, 1e-10));
    }

    #[test]
    fn spectral_kernels_fail_closed_for_invalid_inputs() {
        let err = eigvals_2x2([[1.0, 2.0], [3.0, 4.0]], false).expect_err("non-converged");
        assert_eq!(err.reason_code(), "linalg_spectral_convergence_failed");

        let err = eigvals_2x2([[f64::NAN, 0.0], [0.0, 1.0]], true).expect_err("nan matrix");
        assert_eq!(format!("{err}"), "Array must not contain infs or NaNs");

        let err = eigvals_2x2([[0.0, -1.0], [1.0, 0.0]], true).expect_err("complex spectrum");
        assert_eq!(err.reason_code(), "linalg_spectral_convergence_failed");

        let err = eigh_2x2([[1.0, 0.0], [0.0, 1.0]], "X", true).expect_err("invalid uplo");
        assert_eq!(err.reason_code(), "linalg_spectral_convergence_failed");
    }

    #[test]
    fn eig_non_finite_inputs_match_numpy_error_surface() {
        let matrix = [f64::NAN, 0.0, 0.0, 1.0];

        let err = eig_nxn(&matrix, 2).expect_err("eigvals nan matrix");
        assert_eq!(format!("{err}"), "Array must not contain infs or NaNs");

        let err = eig_nxn_full(&matrix, 2).expect_err("eig nan matrix");
        assert_eq!(format!("{err}"), "Array must not contain infs or NaNs");
    }

    #[test]
    fn cholesky_2x2_lower_and_upper_reconstruct_matrix() {
        let matrix = [[4.0, 1.0], [1.0, 3.0]];

        let lower = cholesky_2x2(matrix, "L").expect("lower");
        let ll_t = [
            [
                lower[0][0].mul_add(lower[0][0], lower[0][1] * lower[0][1]),
                lower[0][0].mul_add(lower[1][0], lower[0][1] * lower[1][1]),
            ],
            [
                lower[1][0].mul_add(lower[0][0], lower[1][1] * lower[0][1]),
                lower[1][0].mul_add(lower[1][0], lower[1][1] * lower[1][1]),
            ],
        ];
        assert!(approx_equal(ll_t[0][0], 4.0, 1e-12));
        assert!(approx_equal(ll_t[0][1], 1.0, 1e-12));
        assert!(approx_equal(ll_t[1][0], 1.0, 1e-12));
        assert!(approx_equal(ll_t[1][1], 3.0, 1e-12));

        let upper = cholesky_2x2(matrix, "U").expect("upper");
        let u_tu = [
            [
                upper[0][0].mul_add(upper[0][0], upper[1][0] * upper[1][0]),
                upper[0][0].mul_add(upper[0][1], upper[1][0] * upper[1][1]),
            ],
            [
                upper[0][1].mul_add(upper[0][0], upper[1][1] * upper[1][0]),
                upper[0][1].mul_add(upper[0][1], upper[1][1] * upper[1][1]),
            ],
        ];
        assert!(approx_equal(u_tu[0][0], 4.0, 1e-12));
        assert!(approx_equal(u_tu[0][1], 1.0, 1e-12));
        assert!(approx_equal(u_tu[1][0], 1.0, 1e-12));
        assert!(approx_equal(u_tu[1][1], 3.0, 1e-12));
    }

    #[test]
    fn cholesky_2x2_uses_selected_triangle_only() {
        let lower_only = [[4.0, 999.0], [1.0, 3.0]];
        let lower = cholesky_2x2(lower_only, "L").expect("lower");
        assert!(approx_equal(lower[0][0], 2.0, 1e-12));
        assert!(approx_equal(lower[1][0], 0.5, 1e-12));

        let upper_only = [[4.0, 1.0], [999.0, 3.0]];
        let upper = cholesky_2x2(upper_only, "U").expect("upper");
        assert!(approx_equal(upper[0][0], 2.0, 1e-12));
        assert!(approx_equal(upper[0][1], 0.5, 1e-12));
    }

    #[test]
    fn cholesky_2x2_fail_closed_for_invalid_inputs() {
        let err = cholesky_2x2([[1.0, 2.0], [2.0, 1.0]], "L").expect_err("non-pd");
        assert_eq!(err.reason_code(), "linalg_cholesky_contract_violation");

        let err = cholesky_2x2([[4.0, 1.0], [1.0, 3.0]], "X").expect_err("bad uplo");
        assert_eq!(err.reason_code(), "linalg_cholesky_contract_violation");

        let err = cholesky_2x2([[f64::NAN, 0.0], [0.0, 1.0]], "U").expect_err("non-finite");
        assert_eq!(err.reason_code(), "linalg_cholesky_contract_violation");
    }

    #[test]
    fn cholesky_diagonal_contract_is_enforced() {
        validate_cholesky_diagonal(&[4.0, 3.0, 2.0]).expect("pd diagonal should pass");
        let err = validate_cholesky_diagonal(&[3.0, 0.0]).expect_err("zero diagonal should fail");
        assert_eq!(err.reason_code(), "linalg_cholesky_contract_violation");
    }

    #[test]
    fn qr_mode_shapes_are_deterministic() {
        let reduced = qr_output_shapes(&[5, 3], QrMode::Reduced).expect("reduced");
        assert_eq!(reduced.q_shape, Some(vec![5, 3]));
        assert_eq!(reduced.r_shape, vec![3, 3]);

        let complete = qr_output_shapes(&[5, 3], QrMode::Complete).expect("complete");
        assert_eq!(complete.q_shape, Some(vec![5, 5]));
        assert_eq!(complete.r_shape, vec![5, 3]);

        let just_r = qr_output_shapes(&[5, 3], QrMode::R).expect("r mode");
        assert_eq!(just_r.q_shape, None);
        assert_eq!(just_r.r_shape, vec![3, 3]);

        let err = QrMode::from_mode_token("hostile_mode").expect_err("mode should fail");
        assert_eq!(err.reason_code(), "linalg_qr_mode_invalid");
    }

    #[test]
    fn qr_2x2_reduced_reconstructs_input_and_is_orthonormal() {
        let matrix = [[1.0, 2.0], [3.0, 4.0]];
        let out = qr_2x2(matrix, QrMode::Reduced).expect("qr reduced");
        let q = out.q.expect("reduced has q");
        let r = out.r;

        let qtq = [
            [
                q[0][0].mul_add(q[0][0], q[1][0] * q[1][0]),
                q[0][0].mul_add(q[0][1], q[1][0] * q[1][1]),
            ],
            [
                q[0][1].mul_add(q[0][0], q[1][1] * q[1][0]),
                q[0][1].mul_add(q[0][1], q[1][1] * q[1][1]),
            ],
        ];
        assert!(approx_equal(qtq[0][0], 1.0, 1e-12));
        assert!(approx_equal(qtq[0][1], 0.0, 1e-12));
        assert!(approx_equal(qtq[1][0], 0.0, 1e-12));
        assert!(approx_equal(qtq[1][1], 1.0, 1e-12));

        let qr = [
            [
                q[0][0].mul_add(r[0][0], q[0][1] * r[1][0]),
                q[0][0].mul_add(r[0][1], q[0][1] * r[1][1]),
            ],
            [
                q[1][0].mul_add(r[0][0], q[1][1] * r[1][0]),
                q[1][0].mul_add(r[0][1], q[1][1] * r[1][1]),
            ],
        ];
        assert!(approx_equal(qr[0][0], matrix[0][0], 1e-10));
        assert!(approx_equal(qr[0][1], matrix[0][1], 1e-10));
        assert!(approx_equal(qr[1][0], matrix[1][0], 1e-10));
        assert!(approx_equal(qr[1][1], matrix[1][1], 1e-10));
    }

    #[test]
    fn qr_2x2_r_mode_and_complete_mode_contracts() {
        let matrix = [[1.0, 2.0], [3.0, 4.0]];
        let reduced = qr_2x2(matrix, QrMode::Reduced).expect("reduced");
        let complete = qr_2x2(matrix, QrMode::Complete).expect("complete");
        let r_only = qr_2x2(matrix, QrMode::R).expect("r");

        assert_eq!(complete.q, reduced.q);
        assert_eq!(complete.r, reduced.r);
        assert!(r_only.q.is_none());
        assert_eq!(r_only.r, reduced.r);
        assert!(approx_equal(r_only.r[1][0], 0.0, 1e-12));
    }

    #[test]
    fn qr_2x2_handles_rank_deficient_input() {
        let matrix = [[1.0, 2.0], [2.0, 4.0]];
        let out = qr_2x2(matrix, QrMode::Reduced).expect("rank-def qr");
        assert!(approx_equal(out.r[1][1], 0.0, 1e-10));

        let q = out.q.expect("q");
        let qr = [
            [
                q[0][0].mul_add(out.r[0][0], q[0][1] * out.r[1][0]),
                q[0][0].mul_add(out.r[0][1], q[0][1] * out.r[1][1]),
            ],
            [
                q[1][0].mul_add(out.r[0][0], q[1][1] * out.r[1][0]),
                q[1][0].mul_add(out.r[0][1], q[1][1] * out.r[1][1]),
            ],
        ];
        assert!(approx_equal(qr[0][0], matrix[0][0], 1e-10));
        assert!(approx_equal(qr[0][1], matrix[0][1], 1e-10));
        assert!(approx_equal(qr[1][0], matrix[1][0], 1e-10));
        assert!(approx_equal(qr[1][1], matrix[1][1], 1e-10));
    }

    #[test]
    fn qr_2x2_fail_closed_for_non_finite_and_raw_mode() {
        let err = qr_2x2([[f64::NAN, 0.0], [0.0, 1.0]], QrMode::Reduced).expect_err("nan");
        assert_eq!(err.reason_code(), "linalg_qr_mode_invalid");

        let err = qr_2x2([[1.0, 0.0], [0.0, 1.0]], QrMode::Raw).expect_err("raw");
        assert_eq!(err.reason_code(), "linalg_qr_mode_invalid");
    }

    #[test]
    fn svd_2x2_reconstructs_and_orders_singular_values() {
        let matrix = [[3.0, 1.0], [1.0, 3.0]];
        let out = svd_2x2(matrix, true).expect("svd");
        assert!(out.singular_values[0] >= out.singular_values[1]);
        assert!(out.singular_values[1] >= 0.0);

        let u = out.u;
        let vt = out.vt;
        let s = out.singular_values;

        let utu = [
            [
                u[0][0].mul_add(u[0][0], u[1][0] * u[1][0]),
                u[0][0].mul_add(u[0][1], u[1][0] * u[1][1]),
            ],
            [
                u[0][1].mul_add(u[0][0], u[1][1] * u[1][0]),
                u[0][1].mul_add(u[0][1], u[1][1] * u[1][1]),
            ],
        ];
        assert!(approx_equal(utu[0][0], 1.0, 1e-10));
        assert!(approx_equal(utu[0][1], 0.0, 1e-10));
        assert!(approx_equal(utu[1][0], 0.0, 1e-10));
        assert!(approx_equal(utu[1][1], 1.0, 1e-10));

        let vvt = [
            [
                vt[0][0].mul_add(vt[0][0], vt[0][1] * vt[0][1]),
                vt[0][0].mul_add(vt[1][0], vt[0][1] * vt[1][1]),
            ],
            [
                vt[1][0].mul_add(vt[0][0], vt[1][1] * vt[0][1]),
                vt[1][0].mul_add(vt[1][0], vt[1][1] * vt[1][1]),
            ],
        ];
        assert!(approx_equal(vvt[0][0], 1.0, 1e-10));
        assert!(approx_equal(vvt[0][1], 0.0, 1e-10));
        assert!(approx_equal(vvt[1][0], 0.0, 1e-10));
        assert!(approx_equal(vvt[1][1], 1.0, 1e-10));

        let us = [
            [u[0][0] * s[0], u[0][1] * s[1]],
            [u[1][0] * s[0], u[1][1] * s[1]],
        ];
        let recon = [
            [
                us[0][0].mul_add(vt[0][0], us[0][1] * vt[1][0]),
                us[0][0].mul_add(vt[0][1], us[0][1] * vt[1][1]),
            ],
            [
                us[1][0].mul_add(vt[0][0], us[1][1] * vt[1][0]),
                us[1][0].mul_add(vt[0][1], us[1][1] * vt[1][1]),
            ],
        ];
        assert!(approx_equal(recon[0][0], matrix[0][0], 1e-10));
        assert!(approx_equal(recon[0][1], matrix[0][1], 1e-10));
        assert!(approx_equal(recon[1][0], matrix[1][0], 1e-10));
        assert!(approx_equal(recon[1][1], matrix[1][1], 1e-10));
    }

    #[test]
    fn svd_2x2_handles_rank_deficient_and_fail_closed_paths() {
        let rank_def = [[1.0, 2.0], [2.0, 4.0]];
        let out = svd_2x2(rank_def, true).expect("rank-def svd");
        assert!(approx_equal(out.singular_values[1], 0.0, 1e-10));

        let err = svd_2x2([[f64::NAN, 0.0], [0.0, 1.0]], true).expect_err("nan");
        assert_eq!(err.reason_code(), "linalg_svd_nonconvergence");

        let err = svd_2x2([[1.0, 0.0], [0.0, 1.0]], false).expect_err("non-converged");
        assert_eq!(err.reason_code(), "linalg_svd_nonconvergence");
    }

    #[test]
    fn svd_shapes_match_full_and_reduced_contracts() {
        let reduced = svd_output_shapes(&[6, 4], false, true).expect("reduced svd");
        assert_eq!(reduced.u_shape, vec![6, 4]);
        assert_eq!(reduced.s_shape, vec![4]);
        assert_eq!(reduced.vh_shape, vec![4, 4]);

        let full = svd_output_shapes(&[6, 4], true, true).expect("full svd");
        assert_eq!(full.u_shape, vec![6, 6]);
        assert_eq!(full.s_shape, vec![4]);
        assert_eq!(full.vh_shape, vec![4, 4]);

        let err = svd_output_shapes(&[6, 4], false, false).expect_err("non-convergence");
        assert_eq!(err.reason_code(), "linalg_svd_nonconvergence");
    }

    #[test]
    fn spectral_branch_is_fail_closed() {
        validate_spectral_branch("L", true).expect("L branch should pass");
        validate_spectral_branch("U", true).expect("U branch should pass");
        let err = validate_spectral_branch("X", true).expect_err("unknown branch");
        assert_eq!(err.reason_code(), "linalg_spectral_convergence_failed");
    }

    #[test]
    fn lstsq_tuple_shapes_cover_vector_and_matrix_rhs() {
        let vector_rhs = lstsq_output_shapes(&[5, 3], &[5]).expect("vector rhs");
        assert_eq!(vector_rhs.x_shape, vec![3]);
        assert_eq!(vector_rhs.residuals_shape, vec![1]);
        assert_eq!(vector_rhs.rank_upper_bound, 3);
        assert_eq!(vector_rhs.singular_values_shape, vec![3]);

        let matrix_rhs = lstsq_output_shapes(&[5, 3], &[5, 2]).expect("matrix rhs");
        assert_eq!(matrix_rhs.x_shape, vec![3, 2]);
        assert_eq!(matrix_rhs.residuals_shape, vec![2]);

        let err = lstsq_output_shapes(&[5, 3], &[4, 2]).expect_err("mismatch rows");
        assert_eq!(err.reason_code(), "linalg_lstsq_tuple_contract_violation");
    }

    #[test]
    fn lstsq_2x2_runtime_outputs_match_contract() {
        let lhs = [[3.0, 1.0], [1.0, 2.0]];
        let rhs = [5.0, 5.0];
        let out = lstsq_2x2(lhs, rhs, 1e-12).expect("lstsq runtime");
        assert!(approx_equal(out.solution[0], 1.0, 1e-12));
        assert!(approx_equal(out.solution[1], 2.0, 1e-12));
        assert!(approx_equal(out.residual_sum_squares, 0.0, 1e-12));
        assert_eq!(out.rank, 2);
        assert!(out.singular_values[0] >= out.singular_values[1]);

        let rank_def = [[1.0, 2.0], [2.0, 4.0]];
        let rank_def_rhs = [3.0, 6.0];
        let rank_def_out = lstsq_2x2(rank_def, rank_def_rhs, 1e-12).expect("rank-def");
        assert!(approx_equal(rank_def_out.solution[0], 0.6, 1e-10));
        assert!(approx_equal(rank_def_out.solution[1], 1.2, 1e-10));
        assert!(approx_equal(rank_def_out.residual_sum_squares, 0.0, 1e-10));
        assert_eq!(rank_def_out.rank, 1);
    }

    #[test]
    fn lstsq_2x2_reports_residual_for_inconsistent_rhs() {
        let lhs = [[1.0, 2.0], [2.0, 4.0]];
        let rhs = [1.0, 0.0];
        let out = lstsq_2x2(lhs, rhs, 1e-12).expect("inconsistent");
        assert!(out.residual_sum_squares > 0.1);
        assert_eq!(out.rank, 1);
    }

    #[test]
    fn lstsq_2x2_fail_closed_policy_checks() {
        let err = lstsq_2x2([[1.0, 0.0], [0.0, 1.0]], [1.0, 2.0], -1.0).expect_err("rcond");
        assert_eq!(err.reason_code(), "linalg_lstsq_tuple_contract_violation");

        let err =
            lstsq_2x2([[f64::INFINITY, 0.0], [0.0, 1.0]], [1.0, 2.0], 1e-12).expect_err("lhs");
        assert_eq!(err.reason_code(), "linalg_lstsq_tuple_contract_violation");

        let err = lstsq_2x2([[1.0, 0.0], [0.0, 1.0]], [f64::NAN, 2.0], 1e-12).expect_err("rhs");
        assert_eq!(err.reason_code(), "linalg_lstsq_tuple_contract_violation");
    }

    #[test]
    fn tolerance_policy_enforces_bounds() {
        for depth in [0usize, 1, 8, 32, 64, MAX_TOLERANCE_SEARCH_DEPTH] {
            validate_tolerance_policy(1e-6, depth).expect("depth within budget");
        }
        let err = validate_tolerance_policy(-1.0, 1).expect_err("negative rcond");
        assert_eq!(err.reason_code(), "linalg_norm_det_rank_policy_violation");

        let err = validate_tolerance_policy(1e-6, MAX_TOLERANCE_SEARCH_DEPTH + 1)
            .expect_err("search-depth overflow");
        assert_eq!(err.reason_code(), "linalg_norm_det_rank_policy_violation");
    }

    #[test]
    fn backend_bridge_enforces_support_and_budget() {
        validate_backend_bridge(true, 0).expect("baseline backend pass");
        validate_backend_bridge(true, MAX_BACKEND_REVALIDATION_ATTEMPTS).expect("edge budget pass");

        let unsupported =
            validate_backend_bridge(false, 0).expect_err("unsupported backend must fail");
        assert_eq!(unsupported.reason_code(), "linalg_backend_bridge_invalid");

        let overflow = validate_backend_bridge(true, MAX_BACKEND_REVALIDATION_ATTEMPTS + 1)
            .expect_err("revalidation budget overflow");
        assert_eq!(overflow.reason_code(), "linalg_backend_bridge_invalid");
    }

    #[test]
    fn policy_metadata_is_fail_closed_for_unknowns() {
        validate_policy_metadata("strict", "known_compatible_low_risk").expect("known strict");
        validate_policy_metadata("hardened", "known_incompatible_semantics")
            .expect("known hardened");
        validate_policy_metadata("strict", "known_compatible").expect("legacy known compatible");
        validate_policy_metadata("hardened", "known_incompatible")
            .expect("legacy known incompatible");
        validate_policy_metadata("strict", "unknown").expect("legacy unknown semantics");
        validate_policy_metadata(" strict ", " known_compatible_low_risk ")
            .expect("whitespace-padded metadata should normalize");

        let err = validate_policy_metadata("weird", "known_compatible_low_risk")
            .expect_err("unknown mode should fail");
        assert_eq!(err.reason_code(), "linalg_policy_unknown_metadata");
    }

    #[test]
    fn packet008_log_record_is_replay_complete() {
        let record = LinAlgLogRecord {
            ts_utc: "2026-02-16T00:00:00Z".to_string(),
            suite_id: "fnp-linalg::tests".to_string(),
            test_id: "UP-008-solve-inv-contract".to_string(),
            packet_id: LINALG_PACKET_ID.to_string(),
            fixture_id: "UP-008-solve-inv-contract".to_string(),
            mode: LinAlgRuntimeMode::Strict,
            seed: 8008,
            input_digest: "sha256:input".to_string(),
            output_digest: "sha256:output".to_string(),
            env_fingerprint: "fnp-linalg-unit-tests".to_string(),
            artifact_refs: packet008_artifacts(),
            duration_ms: 1,
            outcome: "pass".to_string(),
            reason_code: "linalg_solver_singularity".to_string(),
        };
        assert!(record.is_replay_complete());
    }

    #[test]
    fn packet008_log_record_rejects_missing_fields() {
        let record = LinAlgLogRecord {
            ts_utc: String::new(),
            suite_id: String::new(),
            test_id: String::new(),
            packet_id: "wrong-packet".to_string(),
            fixture_id: String::new(),
            mode: LinAlgRuntimeMode::Hardened,
            seed: 9001,
            input_digest: String::new(),
            output_digest: String::new(),
            env_fingerprint: String::new(),
            artifact_refs: vec![String::new()],
            duration_ms: 0,
            outcome: "unknown".to_string(),
            reason_code: String::new(),
        };
        assert!(!record.is_replay_complete());
    }

    #[test]
    fn packet008_reason_codes_round_trip_into_logs() {
        for (idx, reason_code) in LINALG_PACKET_REASON_CODES.iter().enumerate() {
            let seed = u64::try_from(idx).expect("small index");
            let record = LinAlgLogRecord {
                ts_utc: "2026-02-16T00:00:00Z".to_string(),
                suite_id: "fnp-linalg::tests".to_string(),
                test_id: format!("UP-008-{idx}"),
                packet_id: LINALG_PACKET_ID.to_string(),
                fixture_id: format!("UP-008-{idx}"),
                mode: LinAlgRuntimeMode::Strict,
                seed: 10_000 + seed,
                input_digest: "sha256:input".to_string(),
                output_digest: "sha256:output".to_string(),
                env_fingerprint: "fnp-linalg-unit-tests".to_string(),
                artifact_refs: packet008_artifacts(),
                duration_ms: 1,
                outcome: "pass".to_string(),
                reason_code: (*reason_code).to_string(),
            };
            assert!(record.is_replay_complete());
            assert_eq!(record.reason_code, *reason_code);
        }
    }

    #[test]
    #[ignore = "perf A/B bench; run with --release -- --ignored --nocapture"]
    fn batch_solve_vector_scratch_ab_bench() {
        use rayon::prelude::*;
        use std::time::Instant;
        for &n in &[3usize, 8, 16, 32] {
            let batch = (1usize << 22) / (n * n);
            let ms = n * n;
            let a: Vec<f64> = (0..batch * ms)
                .map(|i| {
                    let cell = i % ms;
                    let (r, c) = (cell / n, cell % n);
                    if r == c { n as f64 + 2.0 } else { ((i % 11) as f64 - 5.0) * 0.1 }
                })
                .collect();
            let b: Vec<f64> = (0..batch * n).map(|i| (i % 17) as f64 - 8.0).collect();
            let a_shape = [batch, n, n];
            let b_shape = [batch, n];
            // OLD: the original batch_solve shape — per-lane solve_nxn (lu+perm+x alloc
            // every lane) collected into Vec<Vec> then flattened into one buffer.
            let old = || -> usize {
                let lanes: Vec<Vec<f64>> = (0..batch)
                    .into_par_iter()
                    .map(|i| super::solve_nxn(&a[i * ms..(i + 1) * ms], &b[i * n..(i + 1) * n], n).unwrap())
                    .collect();
                let mut out = Vec::with_capacity(batch * n);
                for x in &lanes {
                    out.extend_from_slice(x);
                }
                out.len()
            };
            // NEW: batch_solve scratch fast path (reused lu/perm per thread).
            let new = || -> usize {
                super::batch_solve(&a, &a_shape, &b, &b_shape, true).unwrap().len()
            };
            let _ = old();
            let _ = new();
            let t = Instant::now();
            std::hint::black_box(old());
            let old_ms = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            std::hint::black_box(new());
            let new_ms = t.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "batch_solve n={n:2} batch={batch:8}: per-lane={old_ms:8.2}ms scratch={new_ms:8.2}ms speedup={:.2}x",
                old_ms / new_ms
            );
        }
    }

    #[test]
    fn batch_solve_vector_scratch_matches_per_lane_solve_nxn_bits() {
        // The vector-RHS batch_solve scratch fast path (reused lu/perm buffers) must
        // be BYTE-IDENTICAL to an independent per-lane solve_nxn — equality vs that
        // unchanged reference is the isomorphism proof that the buffer reuse never
        // alters the LU arithmetic.
        for &n in &[2usize, 3, 5, 8, 16] {
            let batch = 4096usize; // total >= gate so the parallel path is exercised
            let ms = n * n;
            let a: Vec<f64> = (0..batch * ms)
                .map(|i| {
                    let cell = i % ms;
                    let (r, c) = (cell / n, cell % n);
                    if r == c {
                        n as f64 + 2.0 + ((i / ms) % 5) as f64
                    } else {
                        (((i % 11) as f64) - 5.0) * 0.1
                    }
                })
                .collect();
            let b: Vec<f64> = (0..batch * n)
                .map(|i| (((i % 17) as f64) - 8.0) * 0.25)
                .collect();
            let a_shape = [batch, n, n];
            let b_shape = [batch, n];
            let got = super::batch_solve(&a, &a_shape, &b, &b_shape, true).expect("batch_solve");
            // Independent per-lane reference via the unchanged solve_nxn.
            let mut reference = Vec::with_capacity(batch * n);
            for lane in 0..batch {
                let x = super::solve_nxn(&a[lane * ms..(lane + 1) * ms], &b[lane * n..(lane + 1) * n], n)
                    .expect("solve_nxn");
                reference.extend_from_slice(&x);
            }
            assert_eq!(got.len(), reference.len());
            for (i, (g, r)) in got.iter().zip(&reference).enumerate() {
                assert_eq!(g.to_bits(), r.to_bits(), "n={n} lane-flat index {i} diverged");
            }
        }
    }

    #[test]
    fn batch_solve_matrix_scratch_matches_per_lane_solve_nxn_multi_bits() {
        // Matrix-RHS zero-alloc path must be BYTE-IDENTICAL to per-lane solve_nxn_multi.
        for &(n, m) in &[(2usize, 3usize), (3, 2), (5, 4), (8, 8), (16, 5)] {
            let batch = 2048usize;
            let ms = n * n;
            let rw = n * m;
            let a: Vec<f64> = (0..batch * ms)
                .map(|i| {
                    let cell = i % ms;
                    let (r, c) = (cell / n, cell % n);
                    if r == c {
                        n as f64 + 2.0 + ((i / ms) % 5) as f64
                    } else {
                        (((i % 11) as f64) - 5.0) * 0.1
                    }
                })
                .collect();
            let b: Vec<f64> = (0..batch * rw)
                .map(|i| (((i % 19) as f64) - 9.0) * 0.2)
                .collect();
            let a_shape = [batch, n, n];
            let b_shape = [batch, n, m];
            let got = super::batch_solve(&a, &a_shape, &b, &b_shape, false).expect("batch_solve");
            let mut reference = Vec::with_capacity(batch * rw);
            for lane in 0..batch {
                let x = super::solve_nxn_multi(
                    &a[lane * ms..(lane + 1) * ms],
                    &b[lane * rw..(lane + 1) * rw],
                    n,
                    m,
                )
                .expect("solve_nxn_multi");
                reference.extend_from_slice(&x);
            }
            assert_eq!(got.len(), reference.len());
            for (i, (g, r)) in got.iter().zip(&reference).enumerate() {
                assert_eq!(g.to_bits(), r.to_bits(), "n={n} m={m} index {i} diverged");
            }
        }
    }

    #[test]
    #[ignore = "perf A/B bench; run with --release -- --ignored --nocapture"]
    fn batch_solve_matrix_scratch_ab_bench() {
        use rayon::prelude::*;
        use std::time::Instant;
        for &(n, m) in &[(3usize, 3usize), (8, 8), (16, 8), (32, 8)] {
            let batch = (1usize << 22) / (n * n + n * m);
            let ms = n * n;
            let rw = n * m;
            let a: Vec<f64> = (0..batch * ms)
                .map(|i| {
                    let cell = i % ms;
                    let (r, c) = (cell / n, cell % n);
                    if r == c { n as f64 + 2.0 } else { ((i % 11) as f64 - 5.0) * 0.1 }
                })
                .collect();
            let b: Vec<f64> = (0..batch * rw).map(|i| (i % 19) as f64 - 9.0).collect();
            let a_shape = [batch, n, n];
            let b_shape = [batch, n, m];
            let old = || -> usize {
                let lanes: Vec<Vec<f64>> = (0..batch)
                    .into_par_iter()
                    .map(|i| {
                        super::solve_nxn_multi(&a[i * ms..(i + 1) * ms], &b[i * rw..(i + 1) * rw], n, m)
                            .unwrap()
                    })
                    .collect();
                let mut out = Vec::with_capacity(batch * rw);
                for x in &lanes {
                    out.extend_from_slice(x);
                }
                out.len()
            };
            let new = || -> usize {
                super::batch_solve(&a, &a_shape, &b, &b_shape, false).unwrap().len()
            };
            let _ = old();
            let _ = new();
            let t = Instant::now();
            std::hint::black_box(old());
            let old_ms = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            std::hint::black_box(new());
            let new_ms = t.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "batch_solve_multi n={n:2} m={m:2} batch={batch:8}: per-lane={old_ms:8.2}ms scratch={new_ms:8.2}ms speedup={:.2}x",
                old_ms / new_ms
            );
        }
    }

    #[test]
    fn solve_nxn_3x3_system() {
        // A x = b  where x = [2, 3, -1]
        let a = [2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0];
        let b = [8.0, -11.0, -3.0];
        let x = solve_nxn(&a, &b, 3).expect("3x3 solve");
        assert!(approx_equal(x[0], 2.0, 1e-10));
        assert!(approx_equal(x[1], 3.0, 1e-10));
        assert!(approx_equal(x[2], -1.0, 1e-10));
    }

    #[test]
    fn solve_nxn_rejects_singular() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = [1.0, 2.0, 3.0];
        let err = solve_nxn(&a, &b, 3).expect_err("singular");
        assert_eq!(err.reason_code(), "linalg_solver_singularity");
    }

    #[test]
    fn det_nxn_scalar_and_3x3() {
        // 1x1
        let d1 = det_nxn(&[5.0], 1).expect("1x1 det");
        assert!(approx_equal(d1, 5.0, 1e-12));

        // 3x3: det([[6,1,1],[4,-2,5],[2,8,7]]) = -306
        let a3 = [6.0, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0];
        let d3 = det_nxn(&a3, 3).expect("3x3 det");
        assert!(approx_equal(d3, -306.0, 1e-8));

        // singular
        let a_sing = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let d_sing = det_nxn(&a_sing, 3).expect("singular det");
        assert!(approx_equal(d_sing, 0.0, 1e-10));
    }

    #[test]
    fn det_empty_matrix_returns_one() {
        let d0 = det_nxn(&[], 0).expect("empty det");
        assert_eq!(d0, 1.0);
    }

    #[test]
    fn slogdet_empty_matrix_returns_sign_one_logdet_zero() {
        let (sign, logdet) = slogdet_nxn(&[], 0).expect("empty slogdet");
        assert_eq!(sign, 1.0);
        assert_eq!(logdet, 0.0);
    }

    #[test]
    fn batch_inv_scratch_matches_per_lane_inv_nxn_bits() {
        // Zero-alloc batch_inv must be BYTE-IDENTICAL to per-lane inv_nxn.
        for &n in &[2usize, 3, 5, 8, 15] {
            let batch = 2048usize;
            let ms = n * n;
            let a: Vec<f64> = (0..batch * ms)
                .map(|i| {
                    let cell = i % ms;
                    let (r, c) = (cell / n, cell % n);
                    if r == c {
                        n as f64 + 2.0 + ((i / ms) % 5) as f64
                    } else {
                        (((i % 11) as f64) - 5.0) * 0.1
                    }
                })
                .collect();
            let shape = [batch, n, n];
            let got = super::batch_inv(&a, &shape).expect("batch_inv");
            let mut reference = Vec::with_capacity(batch * ms);
            for lane in 0..batch {
                let inv = super::inv_nxn(&a[lane * ms..(lane + 1) * ms], n).expect("inv_nxn");
                reference.extend_from_slice(&inv);
            }
            assert_eq!(got.len(), reference.len());
            for (i, (g, r)) in got.iter().zip(&reference).enumerate() {
                assert_eq!(g.to_bits(), r.to_bits(), "n={n} index {i} diverged");
            }
        }
    }

    #[test]
    fn batch_cholesky_scratch_matches_per_lane_cholesky_nxn_bits() {
        // Zero-alloc batch_cholesky must be BYTE-IDENTICAL to per-lane cholesky_nxn.
        for &n in &[2usize, 3, 5, 8, 15, 16, 32, 64] {
            let batch = if n >= 64 { 128usize } else { 2048usize };
            let ms = n * n;
            // Symmetric positive-definite per lane: A = M·Mᵀ + diag boost. Build a
            // diagonally-dominant symmetric matrix directly.
            let a: Vec<f64> = (0..batch * ms)
                .map(|i| {
                    let cell = i % ms;
                    let (r, c) = (cell / n, cell % n);
                    if r == c {
                        n as f64 * 4.0 + ((i / ms) % 5) as f64
                    } else {
                        let lo = r.min(c);
                        let hi = r.max(c);
                        (((lo * 7 + hi) % 9) as f64 - 4.0) * 0.1
                    }
                })
                .collect();
            let shape = [batch, n, n];
            let got = super::batch_cholesky(&a, &shape).expect("batch_cholesky");
            let mut reference = Vec::with_capacity(batch * ms);
            for lane in 0..batch {
                let l = super::cholesky_nxn(&a[lane * ms..(lane + 1) * ms], n).expect("cholesky_nxn");
                reference.extend_from_slice(&l);
            }
            assert_eq!(got.len(), reference.len());
            for (i, (g, r)) in got.iter().zip(&reference).enumerate() {
                assert_eq!(g.to_bits(), r.to_bits(), "n={n} index {i} diverged");
            }
        }
    }

    #[test]
    #[ignore = "perf A/B bench; run with --release -- --ignored --nocapture"]
    fn batch_cholesky_scratch_ab_bench() {
        use rayon::prelude::*;
        use std::time::Instant;
        for &n in &[3usize, 8, 12, 16] {
            let batch = (1usize << 22) / (n * n);
            let ms = n * n;
            let a: Vec<f64> = (0..batch * ms)
                .map(|i| {
                    let cell = i % ms;
                    let (r, c) = (cell / n, cell % n);
                    if r == c {
                        n as f64 * 4.0
                    } else {
                        let (lo, hi) = (r.min(c), r.max(c));
                        ((lo * 7 + hi) % 9) as f64 * 0.1 - 0.4
                    }
                })
                .collect();
            let shape = [batch, n, n];
            let old = || -> usize {
                let lanes: Vec<Vec<f64>> = (0..batch)
                    .into_par_iter()
                    .map(|i| super::cholesky_nxn(&a[i * ms..(i + 1) * ms], n).unwrap())
                    .collect();
                let mut out = Vec::with_capacity(batch * ms);
                for x in &lanes {
                    out.extend_from_slice(x);
                }
                out.len()
            };
            let new = || -> usize { super::batch_cholesky(&a, &shape).unwrap().len() };
            let _ = old();
            let _ = new();
            let t = Instant::now();
            std::hint::black_box(old());
            let old_ms = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            std::hint::black_box(new());
            let new_ms = t.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "batch_chol n={n:2} batch={batch:8}: per-lane={old_ms:8.2}ms scratch={new_ms:8.2}ms speedup={:.2}x",
                old_ms / new_ms
            );
        }
    }

    #[test]
    #[ignore = "perf A/B bench; run with --release -- --ignored --nocapture"]
    fn batch_inv_scratch_ab_bench() {
        use rayon::prelude::*;
        use std::time::Instant;
        for &n in &[3usize, 8, 12, 16] {
            let batch = (1usize << 22) / (n * n);
            let ms = n * n;
            let a: Vec<f64> = (0..batch * ms)
                .map(|i| {
                    let cell = i % ms;
                    let (r, c) = (cell / n, cell % n);
                    if r == c { n as f64 + 2.0 } else { ((i % 11) as f64 - 5.0) * 0.1 }
                })
                .collect();
            let shape = [batch, n, n];
            let old = || -> usize {
                let lanes: Vec<Vec<f64>> = (0..batch)
                    .into_par_iter()
                    .map(|i| super::inv_nxn(&a[i * ms..(i + 1) * ms], n).unwrap())
                    .collect();
                let mut out = Vec::with_capacity(batch * ms);
                for x in &lanes {
                    out.extend_from_slice(x);
                }
                out.len()
            };
            let new = || -> usize { super::batch_inv(&a, &shape).unwrap().len() };
            let _ = old();
            let _ = new();
            let t = Instant::now();
            std::hint::black_box(old());
            let old_ms = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            std::hint::black_box(new());
            let new_ms = t.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "batch_inv n={n:2} batch={batch:8}: per-lane={old_ms:8.2}ms scratch={new_ms:8.2}ms speedup={:.2}x",
                old_ms / new_ms
            );
        }
    }

    #[test]
    fn inv_nxn_identity_reconstruction() {
        let a = [2.0, 1.0, 0.0, 0.0, 3.0, 1.0, 1.0, 0.0, 2.0];
        let inv = inv_nxn(&a, 3).expect("3x3 inv");

        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += a[i * 3 + k] * inv[k * 3 + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_equal(sum, expected, 1e-10),
                    "A*A^-1 [{i}][{j}] = {sum}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn inv_nxn_sparse_identity_path_matches_dense_lu_solve_bits() {
        for &n in &[16usize, 32] {
            let mut a = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    a[i * n + j] = if i == j {
                        (n * 2) as f64
                    } else {
                        ((i + j) % 5) as f64 * 0.1
                    };
                }
            }
            let (lu, perm, _) = super::lu_decompose_for_det(&a, n).expect("lu");
            let mut eye = vec![0.0; n * n];
            for i in 0..n {
                eye[i * n + i] = 1.0;
            }
            let dense = super::lu_forward_back_multi(&lu, &perm, &eye, n, n);
            let sparse = super::inv_nxn(&a, n).expect("sparse identity inv");
            assert_eq!(sparse.len(), dense.len());
            for (idx, (got, want)) in sparse.iter().zip(&dense).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "n={n} sparse inverse bit drift at {idx}: got {got:?}, want {want:?}"
                );
            }
        }
    }

    #[test]
    fn inv_nxn_rejects_singular() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let err = inv_nxn(&a, 3).expect_err("singular inv");
        assert_eq!(err.reason_code(), "linalg_solver_singularity");
    }

    #[test]
    fn slogdet_nxn_agrees_with_det() {
        let a = [6.0, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0];
        let det = det_nxn(&a, 3).expect("det");
        let (sign, log_abs) = slogdet_nxn(&a, 3).expect("slogdet");
        assert!(approx_equal(sign * log_abs.exp(), det, 1e-8));

        // singular
        let a_sing = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let (sign, log_abs) = slogdet_nxn(&a_sing, 3).expect("singular slogdet");
        assert_eq!(sign, 0.0);
        assert_eq!(log_abs, f64::NEG_INFINITY);
    }

    #[test]
    fn blocked_lu_band_matches_serial_reference_within_tolerance() {
        // n in [LU_BLOCK_MIN, 896) now routes det/inv/slogdet/solve through the blocked
        // level-3 LU (was the serial unblocked sweep before the threshold was lowered to
        // 512). The blocked path re-associates the trailing update via GEMM, so it matches
        // the serial reference within tolerance (not bit-exact) — exactly numpy/LAPACK's
        // own contract. Verify at n=600 (mid-band): the determinant matches an independent
        // serial LU, and inv reconstructs the identity.
        // Serial reference returns (sign, log|det|) — det itself overflows f64 at n=600.
        fn serial_slogdet_reference(a: &[f64], n: usize) -> (f64, f64) {
            let mut lu = a.to_vec();
            let mut sign = 1.0_f64;
            let max_abs = a.iter().map(|v| v.abs()).fold(0.0, f64::max);
            let thr = (n as f64) * f64::EPSILON * max_abs;
            for k in 0..n {
                let mut max_val = lu[k * n + k].abs();
                let mut max_row = k;
                for i in (k + 1)..n {
                    let v = lu[i * n + k].abs();
                    if v > max_val {
                        max_val = v;
                        max_row = i;
                    }
                }
                assert!(max_val > thr, "reference singular pivot");
                if max_row != k {
                    for j in 0..n {
                        lu.swap(k * n + j, max_row * n + j);
                    }
                    sign = -sign;
                }
                let pivot = lu[k * n + k];
                for i in (k + 1)..n {
                    let factor = lu[i * n + k] / pivot;
                    for j in (k + 1)..n {
                        let u = lu[k * n + j];
                        lu[i * n + j] -= factor * u;
                    }
                }
            }
            let mut log_abs = 0.0;
            for i in 0..n {
                let d = lu[i * n + i];
                if d < 0.0 {
                    sign = -sign;
                }
                log_abs += d.abs().ln();
            }
            (sign, log_abs)
        }

        let n = 600;
        assert!(
            (crate::LU_BLOCK_MIN..896).contains(&n),
            "n must be inside the newly-blocked band"
        );
        let mut a = vec![0.0f64; n * n];
        let mut state = 0x00c0_ffee_1234_5678_u64;
        for v in a.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
        }
        for i in 0..n {
            a[i * n + i] += n as f64; // diagonally dominant → well-conditioned
        }

        let (blocked_sign, blocked_log) = slogdet_nxn(&a, n).expect("blocked slogdet");
        let (ref_sign, ref_log) = serial_slogdet_reference(&a, n);
        assert_eq!(blocked_sign, ref_sign, "slogdet sign must match reference");
        let rel = (blocked_log - ref_log).abs() / ref_log.abs().max(1.0);
        assert!(
            rel < 1e-9,
            "blocked log|det| {blocked_log} vs serial reference {ref_log} (rel {rel:e})"
        );

        // inv · A ≈ I (max abs deviation tiny).
        let inv = inv_nxn(&a, n).expect("blocked inv");
        let mut max_dev = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += inv[i * n + k] * a[k * n + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                max_dev = max_dev.max((s - expected).abs());
            }
        }
        assert!(max_dev < 1e-9, "inv·A deviates from I by {max_dev:e}");
    }

    #[test]
    fn solve_and_inv_nxn_propagate_infinite_inputs() {
        let inf_matrix = [f64::INFINITY, 1.0, 0.0, 0.0, 3.0, 1.0, 2.0, 0.0, 3.0];
        let inf_solve = solve_nxn(&inf_matrix, &[1.0, 2.0, 3.0], 3).expect("inf solve");
        assert!(approx_equal(inf_solve[0], 0.0, 1e-12));
        assert!(approx_equal(inf_solve[1], 0.3333333333333333, 1e-12));
        assert!(approx_equal(inf_solve[2], 1.0, 1e-12));

        let inf_inv = inv_nxn(&inf_matrix, 3).expect("inf inv");
        assert!(approx_equal(inf_inv[0], 0.0, 1e-12));
        assert!(approx_equal(inf_inv[1], 0.0, 1e-12));
        assert!(approx_equal(inf_inv[2], 0.0, 1e-12));
        assert!(approx_equal(inf_inv[3], 0.0, 1e-12));
        assert!(approx_equal(inf_inv[4], 0.3333333333333333, 1e-12));
        assert!(approx_equal(inf_inv[5], -0.1111111111111111, 1e-12));
        assert!(approx_equal(inf_inv[6], 0.0, 1e-12));
        assert!(approx_equal(inf_inv[7], 0.0, 1e-12));
        assert!(approx_equal(inf_inv[8], 0.3333333333333333, 1e-12));
    }

    #[test]
    fn solve_and_inv_nxn_match_numpy_diagonal_nan_parity() {
        let diag00 = [f64::NAN, 1.0, 0.0, 0.0, 3.0, 1.0, 2.0, 0.0, 3.0];
        let solved = solve_nxn(&diag00, &[1.0, 2.0, 3.0], 3).expect("diag00 solve");
        assert!(solved[0].is_nan());
        assert!(approx_equal(solved[1], 0.45454545454545453, 1e-12));
        assert!(approx_equal(solved[2], 0.6363636363636364, 1e-12));

        let inverse = inv_nxn(&diag00, 3).expect("diag00 inv");
        assert!(inverse[0].is_nan());
        assert!(inverse[1].is_nan());
        assert!(inverse[2].is_nan());
        assert!(approx_equal(inverse[3], 0.18181818181818182, 1e-12));
        assert!(approx_equal(inverse[4], 0.2727272727272727, 1e-12));
        assert!(approx_equal(inverse[5], -0.09090909090909091, 1e-12));
        assert!(approx_equal(inverse[6], -0.5454545454545454, 1e-12));
        assert!(approx_equal(inverse[7], 0.18181818181818182, 1e-12));
        assert!(approx_equal(inverse[8], 0.2727272727272727, 1e-12));

        let diag11 = [1.0, 2.0, 0.0, 0.0, f64::NAN, 1.0, 2.0, 0.0, 3.0];
        let solved = solve_nxn(&diag11, &[1.0, 2.0, 3.0], 3).expect("diag11 solve");
        assert!(solved[0].is_nan());
        assert!(solved[1].is_nan());
        assert!(approx_equal(solved[2], 1.2857142857142858, 1e-12));

        let inverse = inv_nxn(&diag11, 3).expect("diag11 inv");
        for value in &inverse[..6] {
            assert!(value.is_nan());
        }
        assert!(approx_equal(inverse[6], -0.2857142857142857, 1e-12));
        assert!(approx_equal(inverse[7], 0.5714285714285714, 1e-12));
        assert!(approx_equal(inverse[8], 0.14285714285714285, 1e-12));
    }

    #[test]
    fn solve_and_inv_nxn_match_numpy_off_diagonal_nan_parity() {
        let offdiag = [1.0, f64::NAN, 0.0, 0.0, 3.0, 1.0, 2.0, 0.0, 3.0];
        let solved = solve_nxn(&offdiag, &[1.0, 2.0, 3.0], 3).expect("offdiag solve");
        assert!(solved.iter().all(|value| value.is_nan()));

        let inverse = inv_nxn(&offdiag, 3).expect("offdiag inv");
        assert!(inverse.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn solve_nxn_allows_non_finite_rhs() {
        let a = [2.0, 1.0, 0.0, 0.0, 3.0, 1.0, 1.0, 0.0, 2.0];

        let nan_rhs = solve_nxn(&a, &[f64::NAN, 2.0, 3.0], 3).expect("nan rhs solve");
        assert!(nan_rhs.iter().all(|value| value.is_nan()));

        let inf_rhs = solve_nxn(&a, &[f64::INFINITY, 2.0, 3.0], 3).expect("inf rhs solve");
        assert!(inf_rhs.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn solve_nxn_multi_propagates_non_finite_inputs() {
        let inf_matrix = [f64::INFINITY, 1.0, 0.0, 0.0, 3.0, 1.0, 2.0, 0.0, 3.0];
        let rhs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let solved = solve_nxn_multi(&inf_matrix, &rhs, 3, 2).expect("inf matrix multi solve");
        let expected = [
            0.0,
            0.0,
            0.4444444444444444,
            0.6666666666666666,
            1.6666666666666667,
            2.0,
        ];
        for (actual, expected) in solved.iter().zip(expected) {
            assert!(approx_equal(*actual, expected, 1e-12));
        }

        let finite_matrix = [2.0, 1.0, 0.0, 0.0, 3.0, 1.0, 1.0, 0.0, 2.0];
        let non_finite_rhs = [f64::NAN, 1.0, 2.0, 3.0, 4.0, 5.0];
        let solved =
            solve_nxn_multi(&finite_matrix, &non_finite_rhs, 3, 2).expect("nan rhs multi solve");
        assert!(solved[0].is_nan());
        assert!(approx_equal(solved[1], 0.38461538461538464, 1e-12));
        assert!(solved[2].is_nan());
        assert!(approx_equal(solved[3], 0.23076923076923078, 1e-12));
        assert!(solved[4].is_nan());
        assert!(approx_equal(solved[5], 2.3076923076923075, 1e-12));
    }

    #[test]
    fn cholesky_nxn_reconstructs_matrix() {
        // 3x3 positive definite: [[4,2,1],[2,5,3],[1,3,6]]
        let a = [4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0];
        let l = cholesky_nxn(&a, 3).expect("3x3 cholesky");
        // Verify L is lower triangular
        assert!(approx_equal(l[1], 0.0, 1e-12));
        assert!(approx_equal(l[2], 0.0, 1e-12));
        assert!(approx_equal(l[5], 0.0, 1e-12));
        // Verify L * L^T = A
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += l[i * 3 + k] * l[j * 3 + k];
                }
                assert!(
                    approx_equal(sum, a[i * 3 + j], 1e-10),
                    "L*L^T [{i}][{j}] = {sum}, expected {}",
                    a[i * 3 + j]
                );
            }
        }
    }

    #[test]
    fn cholesky_nxn_index_hoist_preserves_scalar_reference_bits() -> Result<(), String> {
        fn scalar_reference(a: &[f64], n: usize) -> Result<Vec<f64>, LinAlgError> {
            let mut l = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..=i {
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += l[i * n + k] * l[j * n + k];
                    }
                    if i == j {
                        let diag = a[i * n + i] - sum;
                        if diag <= 0.0 {
                            return Err(LinAlgError::CholeskyContractViolation(
                                "matrix is not positive definite",
                            ));
                        }
                        l[i * n + j] = diag.sqrt();
                    } else {
                        l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
                    }
                }
            }
            Ok(l)
        }

        let n = 12usize;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let distance = i.abs_diff(j) as f64;
                a[i * n + j] = if i == j {
                    32.0 + i as f64 * 0.25
                } else {
                    1.0 / (distance + 2.0)
                };
            }
        }

        let expected = scalar_reference(&a, n).map_err(|err| err.to_string())?;
        let actual = cholesky_nxn(&a, n).map_err(|err| err.to_string())?;
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "row-base index hoist changed Cholesky output bits"
        );

        let mut hasher = Sha256::new();
        for value in &actual {
            hasher.update(value.to_bits().to_le_bytes());
        }
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "287798e558ddd98292d5eb5546085eb3c41c9981d2da152dd84b87c7d586ad50",
            "Cholesky output bit pattern must remain fixed"
        );
        Ok(())
    }

    #[test]
    fn cholesky_nxn_rejects_non_pd() {
        let a = [1.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let err = cholesky_nxn(&a, 3).expect_err("non-pd");
        assert_eq!(err.reason_code(), "linalg_cholesky_contract_violation");
    }

    #[test]
    fn qr_nxn_reconstructs_and_is_orthogonal() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0];
        let (q, r) = qr_nxn(&a, 3).expect("3x3 qr");

        // Q^T Q should be identity
        for i in 0..3 {
            for j in 0..3 {
                let mut dot = 0.0;
                for k in 0..3 {
                    dot += q[k * 3 + i] * q[k * 3 + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_equal(dot, expected, 1e-10),
                    "Q^T*Q [{i}][{j}] = {dot}, expected {expected}"
                );
            }
        }

        // Q * R should reconstruct A
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += q[i * 3 + k] * r[k * 3 + j];
                }
                assert!(
                    approx_equal(sum, a[i * 3 + j], 1e-10),
                    "Q*R [{i}][{j}] = {sum}, expected {}",
                    a[i * 3 + j]
                );
            }
        }
    }

    #[test]
    fn qr_nxn_reused_householder_workspace_preserves_reference_bits() -> Result<(), String> {
        fn allocation_reference(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), LinAlgError> {
            if Some(a.len()) != n.checked_mul(n) || n == 0 {
                return Err(LinAlgError::ShapeContractViolation(
                    "qr_nxn: input must be n*n with n > 0",
                ));
            }
            if a.iter().any(|v| !v.is_finite()) {
                return Err(LinAlgError::NormDetRankPolicyViolation(
                    "matrix entries must be finite for QR",
                ));
            }

            let mut q = vec![0.0; n * n];
            for i in 0..n {
                q[i * n + i] = 1.0;
            }
            let mut r = a.to_vec();

            for k in 0..n {
                let mut col_norm_sq = 0.0;
                for i in k..n {
                    col_norm_sq += r[i * n + k] * r[i * n + k];
                }
                let col_norm = col_norm_sq.sqrt();
                if col_norm == 0.0 {
                    continue;
                }

                let mut v = vec![0.0; n];
                let sign = if r[k * n + k] >= 0.0 { 1.0 } else { -1.0 };
                for i in k..n {
                    v[i] = r[i * n + k];
                }
                v[k] += sign * col_norm;
                let v_norm_sq: f64 = v[k..].iter().map(|x| x * x).sum();
                if v_norm_sq == 0.0 {
                    continue;
                }

                let scale = 2.0 / v_norm_sq;
                for j in k..n {
                    let mut dot = 0.0;
                    for i in k..n {
                        dot += v[i] * r[i * n + j];
                    }
                    let factor = scale * dot;
                    for i in k..n {
                        r[i * n + j] -= factor * v[i];
                    }
                }

                for i in 0..n {
                    let mut dot = 0.0;
                    for j in k..n {
                        dot += q[i * n + j] * v[j];
                    }
                    let factor = scale * dot;
                    for j in k..n {
                        q[i * n + j] -= factor * v[j];
                    }
                }
            }

            Ok((q, r))
        }

        let n = 8usize;
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let centered = ((state >> 33) as f64) / (u32::MAX as f64) - 0.5;
                a[i * n + j] = if i == j {
                    centered + 2.0 + i as f64 * 0.125
                } else {
                    centered
                };
            }
        }

        let (expected_q, expected_r) =
            allocation_reference(&a, n).map_err(|err| err.to_string())?;
        let (actual_q, actual_r) = qr_nxn(&a, n).map_err(|err| err.to_string())?;
        for (label, actual, expected) in [
            ("Q", actual_q.as_slice(), expected_q.as_slice()),
            ("R", actual_r.as_slice(), expected_r.as_slice()),
        ] {
            assert_eq!(
                actual
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "{label} bits changed when reusing Householder workspace"
            );
        }

        let mut hasher = Sha256::new();
        for value in actual_q.iter().chain(actual_r.iter()) {
            hasher.update(value.to_bits().to_le_bytes());
        }
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "613184a2d10d3d5d1a19a0dfc5f4d785817fb6e324c4a99e08f5d533bb752c7f",
            "QR Q/R bit pattern must remain fixed"
        );
        Ok(())
    }

    #[test]
    fn matrix_norm_frobenius_matches_expected() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let norm = matrix_norm_frobenius(&a, 2).expect("frobenius");
        let expected = (1.0 + 4.0 + 9.0 + 16.0_f64).sqrt();
        assert!(approx_equal(norm, expected, 1e-12));
    }

    fn matrix_norm_frobenius_two_pass_reference(a: &[f64], n: usize) -> Result<f64, LinAlgError> {
        if Some(a.len()) != n.checked_mul(n) || n == 0 {
            return Err(LinAlgError::ShapeContractViolation(
                "matrix_norm: input must be n*n with n > 0",
            ));
        }
        if a.iter().any(|v| !v.is_finite()) {
            return Err(LinAlgError::NormDetRankPolicyViolation(
                "matrix entries must be finite for norm",
            ));
        }
        Ok(a.iter().map(|v| v * v).sum::<f64>().sqrt())
    }

    #[test]
    fn matrix_norm_fused_scan_preserves_reference_bits_and_sha256() -> Result<(), String> {
        let n = 17;
        let a: Vec<f64> = (0..n * n)
            .map(|i| (((i * 37) % 101) as f64 - 50.0) / 8.0)
            .collect();

        let expected = matrix_norm_frobenius_two_pass_reference(&a, n)
            .map_err(|err| format!("reference norm failed: {err}"))?;
        let actual =
            matrix_norm_frobenius(&a, n).map_err(|err| format!("fused norm failed: {err}"))?;
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "fused norm must preserve the old finite-input summation bits"
        );

        let mut hasher = Sha256::new();
        hasher.update(actual.to_bits().to_le_bytes());
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "9aff026d01bf0d8d28edf16d190ff6a55feab83362d5d4de6f7c480bba9820ae",
            "fused norm golden digest drifted"
        );
        Ok(())
    }

    #[test]
    fn matrix_norm_frobenius_fused_scan_preserves_non_finite_rejection() {
        for non_finite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut a = [1.0, 2.0, 3.0, 4.0];
            a[2] = non_finite;
            assert!(matches!(
                matrix_norm_frobenius(&a, 2),
                Err(LinAlgError::NormDetRankPolicyViolation(
                    "matrix entries must be finite for norm"
                ))
            ));
        }
    }

    #[test]
    fn matrix_rank_nxn_detects_profiles() {
        // Full rank 3x3
        let a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let rank = matrix_rank_nxn(&a, 3, 1e-10).expect("identity rank");
        assert_eq!(rank, 3);

        // Rank 2 (third row = first + second)
        let b = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let rank = matrix_rank_nxn(&b, 3, 1e-10).expect("rank-2");
        assert_eq!(rank, 2);

        // Rank 0 (all zeros)
        let z = [0.0; 9];
        let rank = matrix_rank_nxn(&z, 3, 1e-10).expect("zero rank");
        assert_eq!(rank, 0);
    }

    #[test]
    fn svd_nxn_identity_singular_values() {
        // SVD of 3x3 identity: singular values = [1, 1, 1]
        let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let sigmas = svd_nxn(&eye, 3).expect("identity svd");
        assert_eq!(sigmas.len(), 3);
        for s in &sigmas {
            assert!((*s - 1.0).abs() < 1e-6, "sigma={s}");
        }
    }

    #[test]
    fn svd_nxn_diagonal_matrix() {
        // SVD of diag(3, 2, 1): singular values = [3, 2, 1]
        let d = [3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0];
        let sigmas = svd_nxn(&d, 3).expect("diag svd");
        assert!((sigmas[0] - 3.0).abs() < 1e-6);
        assert!((sigmas[1] - 2.0).abs() < 1e-6);
        assert!((sigmas[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn eigvalsh_nxn_symmetric_3x3() {
        // Symmetric matrix: [[2, 1, 0], [1, 3, 1], [0, 1, 2]]
        // Known eigenvalues approx: 4.0, 2.0, 1.0
        let a = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let eigvals = eigvalsh_nxn(&a, 3).expect("eigvalsh");
        assert_eq!(eigvals.len(), 3);
        // Eigenvalues of this matrix: 1, 2, 4 (ascending, NumPy convention)
        // x^3 - 7x^2 + 14x - 8 = (x-1)(x-2)(x-4)
        assert!((eigvals[0] - 1.0).abs() < 1e-6, "eig0={}", eigvals[0]);
        assert!((eigvals[1] - 2.0).abs() < 1e-6, "eig1={}", eigvals[1]);
        assert!((eigvals[2] - 4.0).abs() < 1e-6, "eig2={}", eigvals[2]);
    }

    #[test]
    fn exact_symmetric_tridiagonal_values_accepts_only_exact_band() {
        let tri = [
            2.0, -1.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0,
            0.0, -1.0, 2.0,
        ];
        let (d, e) = super::exact_symmetric_tridiagonal_values(&tri, 4).expect("tridiagonal");
        assert_eq!(d, vec![2.0, 2.0, 2.0, 2.0]);
        assert_eq!(e, vec![-1.0, -1.0, -1.0]);

        let mut dense = tri;
        dense[2] = f64::MIN_POSITIVE;
        assert!(super::exact_symmetric_tridiagonal_values(&dense, 4).is_none());

        let mut asymmetric = tri;
        asymmetric[4] = -0.5;
        assert!(super::exact_symmetric_tridiagonal_values(&asymmetric, 4).is_none());
    }

    #[test]
    fn eigvalsh_exact_tridiagonal_matches_dense_reduction_fallback() {
        for &n in &[8usize, 32] {
            let mut tri = vec![0.0; n * n];
            for i in 0..n {
                tri[i * n + i] = 2.0;
                if i + 1 < n {
                    tri[i * n + i + 1] = -1.0;
                    tri[(i + 1) * n + i] = -1.0;
                }
            }

            let fast = eigvalsh_nxn(&tri, n).expect("fast tridiagonal eigvalsh");

            let mut forced_dense = tri.clone();
            forced_dense[2] = f64::MIN_POSITIVE;
            forced_dense[2 * n] = f64::MIN_POSITIVE;
            let dense = eigvalsh_nxn(&forced_dense, n).expect("dense fallback eigvalsh");

            for (idx, (a, b)) in fast.iter().zip(dense.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= 1.0e-10,
                    "n={n} idx={idx} fast={a} dense={b}"
                );
            }
        }
    }

    #[test]
    fn eigvalsh_repeated_eigenvalues() {
        // Matrix with repeated eigenvalue: diag(3, 3, 5)
        let a = [3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0];
        let eigvals = eigvalsh_nxn(&a, 3).expect("eigvalsh repeated");
        let mut sorted = eigvals.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        assert!((sorted[0] - 5.0).abs() < 1e-10, "eig0={}", sorted[0]);
        assert!((sorted[1] - 3.0).abs() < 1e-10, "eig1={}", sorted[1]);
        assert!((sorted[2] - 3.0).abs() < 1e-10, "eig2={}", sorted[2]);
    }

    #[test]
    fn eigvalsh_10x10_residual_check() {
        // Build symmetric 10x10 from A = M + M^T where M has known structure
        let n = 10;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = (i + 1) as f64 * 2.0; // diagonal: 2, 4, 6, ..., 20
            if i + 1 < n {
                a[i * n + (i + 1)] = 1.0;
                a[(i + 1) * n + i] = 1.0;
            }
        }
        let (eigvals, eigvecs) = eigh_nxn(&a, n).expect("eigh 10x10");
        // Check ||A*v - λ*v|| < eps * ||A|| for each eigenpair
        let a_norm: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        for j in 0..n {
            // A * v_j
            let mut av = vec![0.0; n];
            for i in 0..n {
                for k in 0..n {
                    av[i] += a[i * n + k] * eigvecs[k * n + j];
                }
            }
            // λ * v_j
            let lam = eigvals[j];
            let mut residual_sq = 0.0;
            for i in 0..n {
                let diff = av[i] - lam * eigvecs[i * n + j];
                residual_sq += diff * diff;
            }
            let residual = residual_sq.sqrt();
            assert!(
                residual < 1e-8 * a_norm,
                "eigpair {j}: residual {residual}, a_norm {a_norm}"
            );
        }
    }

    #[test]
    fn eig_nxn_complex_eigenvalues_rotation() {
        // 2x2 rotation matrix: eigenvalues e^(±iθ) = cos(θ) ± i*sin(θ)
        let theta = std::f64::consts::FRAC_PI_4; // 45 degrees
        let c = theta.cos();
        let s = theta.sin();
        let a = [c, -s, s, c];
        let eigs = eig_nxn(&a, 2).expect("eig rotation");
        assert_eq!(eigs.len(), 4); // 2 eigenvalues × (re, im)
        // Both real parts should be cos(θ)
        assert!((eigs[0] - c).abs() < 1e-10, "re0={}, expected {c}", eigs[0]);
        assert!((eigs[2] - c).abs() < 1e-10, "re1={}, expected {c}", eigs[2]);
        // Imaginary parts should be ±sin(θ)
        assert!(
            (eigs[1].abs() - s).abs() < 1e-10,
            "im0={}, expected ±{s}",
            eigs[1]
        );
        assert!(
            (eigs[3].abs() - s).abs() < 1e-10,
            "im1={}, expected ±{s}",
            eigs[3]
        );
        // Should be conjugate pair
        assert!(
            (eigs[1] + eigs[3]).abs() < 1e-10,
            "not conjugate: im0={}, im1={}",
            eigs[1],
            eigs[3]
        );
    }

    #[test]
    fn eig_nxn_general_output_matches_golden_sha256() {
        let n = 24usize;
        let mut seed = 0x58d4_9017_2ad3_6cabu64;
        let matrix: Vec<f64> = (0..n * n)
            .map(|idx| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let random = ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
                let row = idx / n;
                let col = idx % n;
                if row == col {
                    random + (row as f64 + 1.0) * 0.03125
                } else {
                    random
                }
            })
            .collect();
        let eigenvalues = eig_nxn(&matrix, n).expect("eig_nxn deterministic general matrix");
        let mut digest = Sha256::new();
        for value in &eigenvalues {
            digest.update(value.to_bits().to_le_bytes());
        }
        let hex: String = digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert_eq!(
            hex, "b18b30e93428dc2b8d8fad2a4c97893b7ee9eeec595acab6e5d4c82bee703b33",
            "eig_nxn general-matrix golden digest drifted"
        );
    }

    #[test]
    fn hessenberg_reduce_values_matches_full_hessenberg_bits() {
        for &(n, seed0) in &[
            (1usize, 0x12ab_34cd_55aa_0000u64),
            (2usize, 0x12ab_34cd_55aa_0004u64),
            (3usize, 0x12ab_34cd_55aa_0001u64),
            (8usize, 0x12ab_34cd_55aa_0002u64),
            (24usize, 0x12ab_34cd_55aa_0003u64),
        ] {
            let mut seed = seed0;
            let matrix: Vec<f64> = (0..n * n)
                .map(|idx| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let random = ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
                    let row = idx / n;
                    let col = idx % n;
                    if row == col {
                        random + (row as f64 + 1.0) * 0.0625
                    } else {
                        random + ((row as isize - col as isize) as f64) * 0.001
                    }
                })
                .collect();
            let (full_h, _q) = super::hessenberg_reduce(&matrix, n);
            let values_h = super::hessenberg_reduce_values(&matrix, n);
            assert_eq!(values_h.len(), full_h.len());
            for (idx, (full, values)) in full_h.iter().zip(&values_h).enumerate() {
                assert_eq!(
                    values.to_bits(),
                    full.to_bits(),
                    "H bit drift at n={n} flat index {idx}"
                );
            }
        }
    }

    #[test]
    fn eig_nxn_full_residual_check() {
        // 4x4 non-symmetric matrix with known structure
        // A has real eigenvalues at 1, 2, 3, 4
        let a = [
            1.0, 0.5, 0.0, 0.0, 0.0, 2.0, 0.3, 0.0, 0.0, 0.0, 3.0, 0.7, 0.0, 0.0, 0.0, 4.0,
        ];
        let n = 4;
        let (eigs, vecs) = eig_nxn_full(&a, n).expect("eig_full 4x4");
        let a_norm: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();

        for j in 0..n {
            let lam_re = eigs[2 * j];
            let lam_im = eigs[2 * j + 1];

            // Extract eigenvector column j (complex)
            let mut v_re = vec![0.0; n];
            let mut v_im = vec![0.0; n];
            for i in 0..n {
                v_re[i] = vecs[j * n * 2 + i * 2];
                v_im[i] = vecs[j * n * 2 + i * 2 + 1];
            }

            // Compute A * v (real matrix × complex vector)
            let mut av_re = vec![0.0; n];
            let mut av_im = vec![0.0; n];
            for i in 0..n {
                for k in 0..n {
                    av_re[i] += a[i * n + k] * v_re[k];
                    av_im[i] += a[i * n + k] * v_im[k];
                }
            }

            // Compute λ * v
            let mut lv_re = vec![0.0; n];
            let mut lv_im = vec![0.0; n];
            for i in 0..n {
                lv_re[i] = lam_re * v_re[i] - lam_im * v_im[i];
                lv_im[i] = lam_re * v_im[i] + lam_im * v_re[i];
            }

            // Residual ||A*v - λ*v||
            let mut residual_sq = 0.0;
            for i in 0..n {
                residual_sq += (av_re[i] - lv_re[i]).powi(2) + (av_im[i] - lv_im[i]).powi(2);
            }
            let residual = residual_sq.sqrt();
            assert!(
                residual < 1e-8 * a_norm,
                "eigpair {j}: λ=({lam_re},{lam_im}), residual={residual}"
            );
        }
    }

    #[test]
    fn schur_20x20_reconstruction() {
        // 20x20 upper Hessenberg matrix
        let n = 20;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = (i as f64 + 1.0) * 1.5;
            if i + 1 < n {
                a[i * n + (i + 1)] = 0.8;
                a[(i + 1) * n + i] = 0.4;
            }
        }
        let (t, z) = schur_nxn(&a, n).expect("schur 20x20");
        // Reconstruct: A ≈ Z * T * Z^T
        let mut recon = vec![0.0; n * n];
        // zt = Z * T
        let mut zt = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    zt[i * n + j] += z[i * n + k] * t[k * n + j];
                }
            }
        }
        // recon = zt * Z^T
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    recon[i * n + j] += zt[i * n + k] * z[j * n + k];
                }
            }
        }
        let a_norm: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mut err_sq = 0.0;
        for i in 0..(n * n) {
            err_sq += (a[i] - recon[i]).powi(2);
        }
        let err = err_sq.sqrt();
        assert!(
            err < 1e-10 * a_norm,
            "Schur reconstruction error: {err} (norm={a_norm})"
        );
    }

    #[test]
    fn eigvalsh_50x50_tridiagonal() {
        // 50x50 symmetric tridiagonal with known eigenvalue bounds
        let n = 50;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = 2.0;
            if i + 1 < n {
                a[i * n + (i + 1)] = -1.0;
                a[(i + 1) * n + i] = -1.0;
            }
        }
        let eigvals = eigvalsh_nxn(&a, n).expect("eigvalsh 50x50");
        assert_eq!(eigvals.len(), n);
        // Known eigenvalues: 2 - 2*cos(k*π/(n+1)) for k=1..n
        // All should be in (0, 4)
        let mut sorted = eigvals.clone();
        sorted.sort_by(f64::total_cmp);
        for (k, &val) in sorted.iter().enumerate() {
            let expected =
                2.0 - 2.0 * ((k + 1) as f64 * std::f64::consts::PI / (n + 1) as f64).cos();
            assert!(
                (val - expected).abs() < 1e-8,
                "eigval {k}: got {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn eigvalsh_values_only_reduction_matches_full_reduce_bits() {
        let n = 64;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = if i == j {
                    (n + 1) as f64
                } else {
                    1.0 / ((i as f64 - j as f64).abs() + 1.0)
                };
            }
        }

        let (full_d, full_e, _) = tridiag_reduce(&a, n);
        let (values_d, values_e) = tridiag_reduce_values(&a, n);
        assert_eq!(
            values_d
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            full_d
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "values-only tridiagonal diagonal must match full-Q reduction bits"
        );
        assert_eq!(
            values_e
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            full_e
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "values-only tridiagonal off-diagonal must match full-Q reduction bits"
        );

        let eigvals = eigvalsh_nxn(&a, n).expect("eigvalsh profile matrix");
        let mut hasher = Sha256::new();
        for value in &eigvals {
            hasher.update(value.to_bits().to_le_bytes());
        }
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        // Re-pinned after the eigenvalues-only QR chase moved to `scaled_hypot`
        // (faster than libm `hypot`, ~1 ulp different per Givens rotation → a benign
        // shift of the last bits of the converged eigenvalues). The values-only and
        // full-Q reductions still agree bitwise above, every eigvalsh residual/known-
        // value test still passes, and the output stream stays deterministic across
        // builds (scalar IEEE div/sqrt/mul, no FMA under +avx2). This pins the new
        // public eigvalsh output stream.
        assert_eq!(
            digest,
            "8918728120e12aa8cc5442a80d0a14a05b3669793ae13425a340d9602344dbb2"
        );
    }

    #[test]
    fn tridiag_eigvals_qr_matches_eig_qr_to_allclose() {
        // The values-only kernel must agree with the (proven) eigenvector-path QR to
        // allclose across odd sizes and clustered/spread spectra.
        for &n in &[7usize, 33, 64, 128] {
            let mut s = 0x9e3779b97f4a7c15u64 ^ (n as u64).wrapping_mul(0x100000001b3);
            let mut next = || {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            };
            let mut d_ref = vec![0.0f64; n];
            let mut e_ref = vec![0.0f64; n - 1];
            for v in d_ref.iter_mut() {
                *v = next() * 4.0;
            }
            for v in e_ref.iter_mut() {
                *v = next();
            }
            let (mut d_old, mut e_old) = (d_ref.clone(), e_ref.clone());
            super::tridiag_eig_qr(&mut d_old, &mut e_old, None, n);
            d_old.sort_by(f64::total_cmp);
            let (mut d_new, mut e_new) = (d_ref.clone(), e_ref.clone());
            super::tridiag_eigvals_qr(&mut d_new, &mut e_new, n);
            d_new.sort_by(f64::total_cmp);
            let max_diff = d_old
                .iter()
                .zip(&d_new)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(
                max_diff < 1e-9,
                "n={n}: values-only QR diverged from eigvec QR by {max_diff}"
            );
        }
    }

    #[test]
    #[ignore = "perf timing; run with --release -- --ignored --nocapture"]
    fn tridiag_eigvals_qr_perf_report() {
        use std::time::Instant;
        for &n in &[256usize, 512, 768] {
            // Reduce a deterministic dense symmetric matrix to tridiagonal once.
            let mut s = 0x1234_5678u64 ^ (n as u64);
            let mut a = vec![0.0f64; n * n];
            for i in 0..n {
                for j in i..n {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let v = ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
                    a[i * n + j] = v;
                    a[j * n + i] = v;
                }
            }
            let (d0, e0) = super::tridiag_reduce_values(&a, n);
            let timed = |old: bool| -> f64 {
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let (mut d, mut e) = (d0.clone(), e0.clone());
                    let t = Instant::now();
                    if old {
                        super::tridiag_eig_qr(&mut d, &mut e, None, n);
                    } else {
                        super::tridiag_eigvals_qr(&mut d, &mut e, n);
                    }
                    best = best.min(t.elapsed().as_secs_f64() * 1e3);
                }
                best
            };
            // interleave OLD(libm hypot) vs NEW(scaled_hypot) to share machine state
            let (mut t_old, mut t_new) = (f64::MAX, f64::MAX);
            for _ in 0..3 {
                t_new = t_new.min(timed(false));
                t_old = t_old.min(timed(true));
            }
            println!(
                "n={n:5} QR_OLD={t_old:8.3}ms  QR_NEW={t_new:8.3}ms  QRspeedup={:.2}x",
                t_old / t_new
            );
        }
    }

    #[test]
    fn eigh_nxn_eigenvectors_reconstruct() {
        // Symmetric 3x3 identity: eigvals = [1,1,1], eigvecs = I
        let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let (eigvals, eigvecs) = eigh_nxn(&eye, 3).expect("eigh identity");
        for e in &eigvals {
            assert!((*e - 1.0).abs() < 1e-6, "eig={e}");
        }
        // Eigenvectors should be orthonormal
        for col in 0..3 {
            let mut norm_sq = 0.0;
            for row in 0..3 {
                norm_sq += eigvecs[row * 3 + col] * eigvecs[row * 3 + col];
            }
            assert!(
                (norm_sq - 1.0).abs() < 1e-6,
                "eigvec col {col} norm^2={norm_sq}"
            );
        }
    }

    #[test]
    fn eigh_nxn_reconstructs_matrix() {
        // A = V * diag(eigenvalues) * V^T for symmetric A
        let a = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let n = 3;
        let (eigvals, v) = eigh_nxn(&a, n).expect("eigh");
        // Reconstruct: A' = V * diag(eigvals) * V^T
        let mut reconstructed = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += v[i * n + k] * eigvals[k] * v[j * n + k];
                }
                reconstructed[i * n + j] = sum;
            }
        }
        for i in 0..n * n {
            assert!(
                (reconstructed[i] - a[i]).abs() < 1e-6,
                "reconstruct[{i}]={}, expected {}",
                reconstructed[i],
                a[i]
            );
        }
    }

    #[test]
    fn cond_nxn_identity_is_one() {
        let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let c = cond_nxn(&eye, 3).expect("cond identity");
        assert!((c - 1.0).abs() < 1e-6, "cond(I)={c}");
    }

    #[test]
    fn cond_nxn_singular_is_infinity() {
        // Singular matrix: row 3 = row 1 + row 2
        let a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let c = cond_nxn(&a, 3).expect("cond singular");
        assert!(
            c.is_infinite(),
            "cond of singular matrix should be inf, got {c}"
        );
    }

    #[test]
    fn cond_mxn_rectangular() {
        // 2x3 matrix [[1,2,3],[4,5,6]]
        // np.linalg.cond(a) = 12.302245504069202
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let c = cond_mxn(&a, 2, 3).expect("cond 2x3");
        assert!(
            (c - 12.302245504069202).abs() < 1e-6,
            "cond(2x3)={c}, expected 12.302245"
        );
    }

    #[test]
    fn cond_mxn_nan_raises_svd_error() {
        // NumPy raises LinAlgError: SVD did not converge for NaN inputs
        let a = [1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0];
        let err = cond_mxn(&a, 2, 3).expect_err("NaN should fail");
        assert_eq!(err.reason_code(), "linalg_svd_nonconvergence");
    }

    #[test]
    fn matrix_power_nxn_identity() {
        let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        // I^5 = I
        let result = matrix_power_nxn(&eye, 3, 5).expect("I^5");
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((result[i * 3 + j] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn matrix_power_nxn_squared() {
        // A = [[1,1],[0,1]], A^2 = [[1,2],[0,1]]
        let a = [1.0, 1.0, 0.0, 1.0];
        let sq = matrix_power_nxn(&a, 2, 2).expect("A^2");
        assert!((sq[0] - 1.0).abs() < 1e-10);
        assert!((sq[1] - 2.0).abs() < 1e-10);
        assert!((sq[2] - 0.0).abs() < 1e-10);
        assert!((sq[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn matrix_power_nxn_negative() {
        // A^(-1) * A should give identity
        let a = [2.0, 1.0, 1.0, 1.0];
        let a_inv = matrix_power_nxn(&a, 2, -1).expect("A^-1");
        let product = mat_mul_flat(&a, &a_inv, 2);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[i * 2 + j] - expected).abs() < 1e-10,
                    "A*A^-1[{i},{j}]={}",
                    product[i * 2 + j]
                );
            }
        }
    }

    #[test]
    fn matrix_power_initial_identity_elision_matches_old_schedule_sha256() {
        fn old_schedule(a: &[f64], n: usize, p: u64) -> Vec<f64> {
            let mut exp = p;
            let mut result = vec![0.0; n * n];
            for i in 0..n {
                result[i * n + i] = 1.0;
            }
            let mut cur = a.to_vec();
            while exp > 0 {
                if exp & 1 == 1 {
                    result = mat_mul_flat(&result, &cur, n);
                }
                exp >>= 1;
                if exp > 0 {
                    cur = mat_mul_flat(&cur, &cur, n);
                }
            }
            result
        }

        let n = 32usize;
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let a: Vec<f64> = (0..n * n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                0.25 + ((state >> 33) as f64) / (u32::MAX as f64)
            })
            .collect();

        let actual = matrix_power_nxn(&a, n, 3).expect("A^3");
        let expected = old_schedule(&a, n, 3);
        assert_eq!(actual.len(), expected.len());
        for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "matrix_power optimized schedule drifted at {index}"
            );
        }

        let mut hasher = Sha256::new();
        for v in &actual {
            hasher.update(v.to_bits().to_le_bytes());
        }
        let digest = hasher.finalize();
        let digest = digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "4cc72d5780b2d27b5e8e544630012f09bb268738be641b8481476fa7e15541a3",
            "matrix_power finite-nonzero golden digest drifted"
        );
    }

    #[test]
    fn matrix_power_identity_elision_falls_back_for_zero_and_nan_bits() {
        fn old_schedule(a: &[f64], n: usize, p: u64) -> Vec<f64> {
            let mut exp = p;
            let mut result = vec![0.0; n * n];
            for i in 0..n {
                result[i * n + i] = 1.0;
            }
            let mut cur = a.to_vec();
            while exp > 0 {
                if exp & 1 == 1 {
                    result = mat_mul_flat(&result, &cur, n);
                }
                exp >>= 1;
                if exp > 0 {
                    cur = mat_mul_flat(&cur, &cur, n);
                }
            }
            result
        }

        let cases = [
            [1.0, -0.0, 2.0, 3.0],
            [1.0, f64::NAN, 2.0, 3.0],
            [1.0, f64::INFINITY, 2.0, 3.0],
        ];
        for a in cases {
            let actual = matrix_power_nxn(&a, 2, 3).expect("fallback A^3");
            let expected = old_schedule(&a, 2, 3);
            for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    e.to_bits(),
                    "matrix_power fallback drifted at {index}"
                );
            }
        }
    }

    #[test]
    fn lstsq_nxn_exact_system() {
        // 3x2 system with exact solution: A*x = b
        // A = [[1,0],[0,1],[1,1]], b = [1,2,3]
        // Normal equations: A^T A = [[2,1],[1,2]], A^T b = [4,5]
        // Solution: x = [1, 2]
        let a = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let b = [1.0, 2.0, 3.0];
        let x = lstsq_nxn(&a, &b, 3, 2).expect("lstsq exact");
        assert!((x[0] - 1.0).abs() < 1e-6, "x[0]={}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-6, "x[1]={}", x[1]);
    }

    #[test]
    fn pinv_nxn_reconstructs_identity() {
        // For square invertible matrix, pinv = inv
        let a = [2.0, 1.0, 1.0, 1.0];
        let a_pinv = pinv_nxn(&a, 2, 2).expect("pinv 2x2");
        // A * A+ should be I
        let product = mat_mul_flat(&a, &a_pinv, 2);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[i * 2 + j] - expected).abs() < 1e-6,
                    "A*A+[{i},{j}]={}",
                    product[i * 2 + j]
                );
            }
        }
    }

    #[test]
    fn trace_nxn_identity() {
        let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert!((trace_nxn(&eye, 3).unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn trace_nxn_general() {
        // trace([[1,2],[3,4]]) = 1 + 4 = 5
        let a = [1.0, 2.0, 3.0, 4.0];
        assert!((trace_nxn(&a, 2).unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn matrix_norm_nxn_frobenius() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let f = matrix_norm_nxn(&a, 2, 2, "fro").unwrap();
        let expected = (1.0 + 4.0 + 9.0 + 16.0_f64).sqrt();
        assert!((f - expected).abs() < 1e-10, "fro={f}, expected {expected}");
    }

    #[test]
    fn matrix_norm_nxn_one_and_inf() {
        // A = [[1, -2], [3, 4]]
        // 1-norm = max col sum = max(|1|+|3|, |-2|+|4|) = max(4, 6) = 6
        // inf-norm = max row sum = max(|1|+|-2|, |3|+|4|) = max(3, 7) = 7
        let a = [1.0, -2.0, 3.0, 4.0];
        let n1 = matrix_norm_nxn(&a, 2, 2, "1").unwrap();
        assert!((n1 - 6.0).abs() < 1e-10, "1-norm={n1}");
        let ni = matrix_norm_nxn(&a, 2, 2, "inf").unwrap();
        assert!((ni - 7.0).abs() < 1e-10, "inf-norm={ni}");
    }

    fn matrix_norm_column_sum_strided_reference(
        a: &[f64],
        m: usize,
        n: usize,
        use_min: bool,
    ) -> f64 {
        let mut selected = if use_min { f64::INFINITY } else { 0.0 };
        for j in 0..n {
            let mut col_sum = 0.0;
            for i in 0..m {
                let value = a[i * n + j];
                if value.is_nan() {
                    return f64::NAN;
                }
                col_sum += value.abs();
            }
            selected = if use_min {
                selected.min(col_sum)
            } else {
                selected.max(col_sum)
            };
        }
        selected
    }

    #[test]
    fn matrix_norm_column_reduction_matches_strided_reference_bits() {
        for (m, n) in [(67, 73), (9, 512)] {
            let a: Vec<f64> = (0..m * n)
                .map(|i| (((i * 29 + 17) % 113) as f64 - 56.0) / 13.0)
                .collect();
            assert!(a.len() >= super::MATRIX_NORM_CACHE_LINEAR_COLUMN_SUM_MIN_ELEMS);

            for (ord, use_min) in [("1", false), ("-1", true)] {
                let expected = matrix_norm_column_sum_strided_reference(&a, m, n, use_min);
                let actual = matrix_norm_nxn(&a, m, n, ord).expect("matrix norm");
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "cache-linear {ord} reduction must preserve the former strided sum bits for {m}x{n}"
                );
            }

            let mut with_nan = a.clone();
            with_nan[n + 5] = f64::NAN;
            assert!(matrix_norm_nxn(&with_nan, m, n, "1").unwrap().is_nan());
            assert!(matrix_norm_nxn(&with_nan, m, n, "-1").unwrap().is_nan());
            assert!(matrix_norm_nxn(&with_nan, m, n, "inf").unwrap().is_nan());
            assert!(matrix_norm_nxn(&with_nan, m, n, "-inf").unwrap().is_nan());
        }
    }

    #[test]
    fn matrix_norm_nxn_spectral() {
        // Identity: spectral norm = 1
        let eye = [1.0, 0.0, 0.0, 1.0];
        let s = matrix_norm_nxn(&eye, 2, 2, "2").unwrap();
        assert!((s - 1.0).abs() < 1e-6, "spectral(I)={s}");
    }

    #[test]
    fn vector_norm_zero_counts_nonzero() {
        let v = [1.0, 0.0, -3.0, 0.0, 5.0];
        let n = vector_norm(&v, Some(VectorNormOrder::Zero)).unwrap();
        assert!((n - 3.0).abs() < 1e-10);
    }

    #[test]
    fn vector_norm_arbitrary_p() {
        let v = [1.0, 2.0, 3.0];
        let n3 = vector_norm(&v, Some(VectorNormOrder::P(3.0))).unwrap();
        let expected = (1.0_f64 + 8.0 + 27.0).powf(1.0 / 3.0);
        assert!(
            (n3 - expected).abs() < 1e-10,
            "3-norm={n3}, expected {expected}"
        );
    }

    #[test]
    fn vector_norm_p_token_parsing() {
        let ord = VectorNormOrder::from_token("3").unwrap();
        assert_eq!(ord, VectorNormOrder::P(3.0));
        let ord0 = VectorNormOrder::from_token("0").unwrap();
        assert_eq!(ord0, VectorNormOrder::Zero);
        let ord_half = VectorNormOrder::from_token("0.5").unwrap();
        assert_eq!(ord_half, VectorNormOrder::P(0.5));
    }

    #[test]
    fn matrix_norm_nxn_neg_one_and_neg_inf() {
        // A = [[1, -2], [3, 4]]
        // -1-norm = min col sum = min(|1|+|3|, |-2|+|4|) = min(4, 6) = 4
        // -inf-norm = min row sum = min(|1|+|-2|, |3|+|4|) = min(3, 7) = 3
        let a = [1.0, -2.0, 3.0, 4.0];
        let n_neg1 = matrix_norm_nxn(&a, 2, 2, "-1").unwrap();
        assert!((n_neg1 - 4.0).abs() < 1e-10, "-1-norm={n_neg1}");
        let n_neginf = matrix_norm_nxn(&a, 2, 2, "-inf").unwrap();
        assert!((n_neginf - 3.0).abs() < 1e-10, "-inf-norm={n_neginf}");
    }

    #[test]
    fn matrix_norm_nxn_neg_two() {
        // Identity: smallest singular value = 1
        let eye = [1.0, 0.0, 0.0, 1.0];
        let s = matrix_norm_nxn(&eye, 2, 2, "-2").unwrap();
        assert!((s - 1.0).abs() < 1e-6, "-2-norm(I)={s}");
    }

    #[test]
    fn matrix_norm_nxn_rectangular_spectral() {
        // 2x3 matrix [[1,2,3],[4,5,6]] - works in NumPy
        // np.linalg.norm(a, 2) = 9.508032000695724
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let s2 = matrix_norm_nxn(&a, 2, 3, "2").unwrap();
        assert!(
            (s2 - 9.508032000695724).abs() < 1e-6,
            "2-norm(2x3)={s2}, expected 9.508032"
        );
    }

    #[test]
    fn matrix_norm_nxn_rectangular_nuclear() {
        // 2x3 matrix [[1,2,3],[4,5,6]]
        // np.linalg.norm(a, 'nuc') = 10.280901636369208
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let nuc = matrix_norm_nxn(&a, 2, 3, "nuc").unwrap();
        assert!(
            (nuc - 10.280901636369208).abs() < 1e-6,
            "nuc-norm(2x3)={nuc}, expected 10.280901"
        );
    }

    #[test]
    fn matrix_norm_nxn_rectangular_neg_two() {
        // 2x3 matrix [[1,2,3],[4,5,6]]
        // np.linalg.norm(a, -2) = 0.7728696356734838
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let s_neg2 = matrix_norm_nxn(&a, 2, 3, "-2").unwrap();
        assert!(
            (s_neg2 - 0.7728696356734838).abs() < 1e-6,
            "-2-norm(2x3)={s_neg2}, expected 0.772869"
        );
    }

    #[test]
    fn matrix_norm_2x2_neg_one_and_neg_inf() {
        let m = [[1.0, -2.0], [3.0, 4.0]];
        let n_neg1 = matrix_norm_2x2(m, Some(MatrixNormOrder::NegOne)).unwrap();
        assert!((n_neg1 - 4.0).abs() < 1e-10, "-1-norm={n_neg1}");
        let n_neginf = matrix_norm_2x2(m, Some(MatrixNormOrder::NegInf)).unwrap();
        assert!((n_neginf - 3.0).abs() < 1e-10, "-inf-norm={n_neginf}");
    }

    #[test]
    fn matrix_norm_order_token_neg_variants() {
        assert_eq!(
            MatrixNormOrder::from_token("-1").unwrap(),
            MatrixNormOrder::NegOne
        );
        assert_eq!(
            MatrixNormOrder::from_token("-inf").unwrap(),
            MatrixNormOrder::NegInf
        );
    }

    #[test]
    fn eig_nxn_symmetric_gives_real_eigenvalues() {
        // Symmetric 2x2: [[2, 1], [1, 2]], eigenvalues = 3, 1
        let a = [2.0, 1.0, 1.0, 2.0];
        let eigs = eig_nxn(&a, 2).unwrap();
        // Should have 2 eigenvalues with ~0 imaginary parts
        assert_eq!(eigs.len(), 4); // [re0, im0, re1, im1]
        assert!(eigs[1].abs() < 1e-6, "im0={}", eigs[1]); // real
        assert!(eigs[3].abs() < 1e-6, "im1={}", eigs[3]); // real
        let mut vals = [eigs[0], eigs[2]];
        vals.sort_by(|a, b| b.total_cmp(a));
        assert!((vals[0] - 3.0).abs() < 1e-6, "eig0={}", vals[0]);
        assert!((vals[1] - 1.0).abs() < 1e-6, "eig1={}", vals[1]);
    }

    #[test]
    fn eig_nxn_diagonal_matrix() {
        let a = [5.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0];
        let eigs = eig_nxn(&a, 3).unwrap();
        assert_eq!(eigs.len(), 6);
        let mut reals: Vec<f64> = (0..3).map(|i| eigs[i * 2]).collect();
        reals.sort_by(|a, b| b.total_cmp(a));
        assert!((reals[0] - 5.0).abs() < 1e-6);
        assert!((reals[1] - 3.0).abs() < 1e-6);
        assert!((reals[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn eig_nxn_rotation_gives_complex_eigenvalues() {
        // 90-degree rotation: [[0, -1], [1, 0]], eigenvalues = ±i
        let a = [0.0, -1.0, 1.0, 0.0];
        let eigs = eig_nxn(&a, 2).unwrap();
        assert_eq!(eigs.len(), 4);
        // Both eigenvalues should have real part ~0 and imaginary parts ±1
        assert!(eigs[0].abs() < 1e-6, "re0={}", eigs[0]);
        assert!(eigs[2].abs() < 1e-6, "re1={}", eigs[2]);
        let mut imags = [eigs[1].abs(), eigs[3].abs()];
        imags.sort_by(f64::total_cmp);
        assert!((imags[0] - 1.0).abs() < 1e-6, "im magnitude={}", imags[0]);
        assert!((imags[1] - 1.0).abs() < 1e-6, "im magnitude={}", imags[1]);
    }

    #[test]
    fn eig_nxn_full_symmetric_eigenvectors() {
        // Symmetric 2x2: [[2, 1], [1, 2]], eigenvalues 3 and 1
        let a = [2.0, 1.0, 1.0, 2.0];
        let (eigs, vecs) = eig_nxn_full(&a, 2).unwrap();
        assert_eq!(eigs.len(), 4);
        assert_eq!(vecs.len(), 8); // 2 columns, 2 rows, 2 (re/im) = 8

        // Each eigenvalue should be real
        assert!(eigs[1].abs() < 1e-6, "im0={}", eigs[1]);
        assert!(eigs[3].abs() < 1e-6, "im1={}", eigs[3]);

        // Verify A*v ≈ lambda*v for each eigenvector
        for col in 0..2 {
            let lam_re = eigs[col * 2];
            let v_re = [vecs[col * 4], vecs[col * 4 + 2]]; // row 0, row 1 real parts
            // A*v
            let av0 = a[0] * v_re[0] + a[1] * v_re[1];
            let av1 = a[2] * v_re[0] + a[3] * v_re[1];
            assert!(
                (av0 - lam_re * v_re[0]).abs() < 1e-4,
                "A*v[0]={av0}, lam*v[0]={}",
                lam_re * v_re[0]
            );
            assert!(
                (av1 - lam_re * v_re[1]).abs() < 1e-4,
                "A*v[1]={av1}, lam*v[1]={}",
                lam_re * v_re[1]
            );
        }
    }

    #[test]
    fn eig_nxn_full_diagonal() {
        // Diagonal 3x3: eigenvalues are the diagonal entries
        let a = [5.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0];
        let (eigs, vecs) = eig_nxn_full(&a, 3).unwrap();
        assert_eq!(eigs.len(), 6);
        assert_eq!(vecs.len(), 18); // 3 cols * 3 rows * 2

        // Each eigenvector should be a unit basis vector (up to sign)
        for col in 0..3 {
            let lam = eigs[col * 2];
            // Verify A*v ≈ lam*v
            for row in 0..3 {
                let v_re = vecs[col * 6 + row * 2]; // col * (n*2) + row * 2
                let mut av = 0.0;
                for k in 0..3 {
                    av += a[row * 3 + k] * vecs[col * 6 + k * 2];
                }
                assert!(
                    (av - lam * v_re).abs() < 1e-4,
                    "col={col} row={row} av={av} lam*v={}",
                    lam * v_re
                );
            }
        }
    }

    #[test]
    fn eig_nxn_full_rotation_complex_eigenvectors() {
        // 90-degree rotation: eigenvalues ±i
        let a = [0.0, -1.0, 1.0, 0.0];
        let (eigs, vecs) = eig_nxn_full(&a, 2).unwrap();
        assert_eq!(eigs.len(), 4);
        assert_eq!(vecs.len(), 8);

        // Eigenvalues should be purely imaginary
        assert!(eigs[0].abs() < 1e-6, "re0={}", eigs[0]);
        assert!(eigs[2].abs() < 1e-6, "re1={}", eigs[2]);
        assert!((eigs[1].abs() - 1.0).abs() < 1e-6);
        assert!((eigs[3].abs() - 1.0).abs() < 1e-6);

        // Eigenvectors should be non-zero and normalized
        for col in 0..2 {
            let mut norm_sq = 0.0;
            for row in 0..2 {
                let re = vecs[col * 4 + row * 2];
                let im = vecs[col * 4 + row * 2 + 1];
                norm_sq += re * re + im * im;
            }
            assert!(
                (norm_sq - 1.0).abs() < 1e-4,
                "eigenvector {col} not normalized: norm²={norm_sq}"
            );
        }
    }

    #[test]
    fn eig_nxn_full_rejects_empty() {
        let result = eig_nxn_full(&[], 0);
        assert!(result.is_err());
    }

    // ── Extreme-scale regression tests (scale-relative threshold validation) ──

    #[test]
    fn eig_large_scale_matrix() {
        // Matrix with entries ~1e10. Absolute 1e-10 threshold would misclassify.
        let scale = 1e10;
        let a = [3.0 * scale, 1.0 * scale, 1.0 * scale, 2.0 * scale];
        let eigvals = eig_nxn(&a, 2).expect("large-scale eig");
        // Eigenvalues of [[3s, s], [s, 2s]] = s * eigenvalues of [[3,1],[1,2]]
        // Eigenvalues of [[3,1],[1,2]]: (5 ± sqrt(5))/2 ≈ 3.618, 1.382
        let e0 = eigvals[0]; // real part
        let e1 = eigvals[2];
        assert!(
            (e0 / scale - 3.618).abs() < 0.01 || (e1 / scale - 3.618).abs() < 0.01,
            "eigenvalue ~3.618*1e10 not found: e0={e0}, e1={e1}"
        );
    }

    #[test]
    fn eig_small_scale_matrix() {
        // Matrix with entries ~1e-12.
        let scale = 1e-12;
        let a = [2.0 * scale, 0.0, 0.0, 5.0 * scale];
        let eigvals = eig_nxn(&a, 2).expect("small-scale eig");
        let e0 = eigvals[0];
        let e1 = eigvals[2];
        let (lo, hi) = if e0 < e1 { (e0, e1) } else { (e1, e0) };
        assert!((lo / scale - 2.0).abs() < 1e-6, "small eigenvalue: {lo}");
        assert!((hi / scale - 5.0).abs() < 1e-6, "large eigenvalue: {hi}");
    }

    #[test]
    fn sqrtm_large_scale() {
        // sqrtm of a scaled identity: sqrtm(k*I) = sqrt(k)*I
        let k = 1e8;
        let a = [k, 0.0, 0.0, k];
        let s = sqrtm_nxn(&a, 2).expect("sqrtm large scale");
        let expected = k.sqrt();
        assert!(
            (s[0] - expected).abs() < expected * 1e-10,
            "sqrtm[0,0]={}, expected {expected}",
            s[0]
        );
        assert!(s[1].abs() < 1e-5, "sqrtm off-diag should be ~0");
    }

    #[test]
    fn expm_near_zero_matrix() {
        // expm of near-zero matrix should be near identity
        let eps = 1e-15;
        let a = [eps, 0.0, 0.0, eps];
        let e = expm_nxn(&a, 2).expect("expm near zero");
        assert!((e[0] - 1.0).abs() < 1e-12, "expm[0,0]={}", e[0]);
        assert!((e[3] - 1.0).abs() < 1e-12, "expm[1,1]={}", e[3]);
    }

    // Random symmetric positive-definite matrix: A = B·B^T + n·I.
    fn chol_spd(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed | 1;
        let b: Vec<f64> = (0..n * n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect();
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += b[i * n + k] * b[j * n + k];
                }
                a[i * n + j] = acc;
            }
            a[i * n + i] += n as f64;
        }
        a
    }

    // Inline unblocked dot-product Cholesky (pre-blocking reference).
    fn cholesky_unblocked_ref(a: &[f64], n: usize) -> Vec<f64> {
        let mut l = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut sum = a[i * n + j];
                for k in 0..j {
                    sum -= l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    l[i * n + j] = sum.sqrt();
                } else {
                    l[i * n + j] = sum / l[j * n + j];
                }
            }
        }
        l
    }

    fn qr_rand(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed | 1;
        (0..n * n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect()
    }

    // Old stride-n column accumulation (pre-transpose) for the same-process A/B.
    fn tridiag_eig_qr_stridn_ref(d: &mut [f64], e: &mut [f64], q: &mut [f64], n: usize) {
        let eps = f64::EPSILON;
        let max_iter = super::EIGEN_QR_ITERATION_COEFF * n * n;
        for _iter in 0..max_iter {
            for i in 0..e.len() {
                if e[i].abs() <= eps * (d[i].abs() + d[i + 1].abs()) {
                    e[i] = 0.0;
                }
            }
            let mut hi = n - 1;
            while hi > 0 && e[hi - 1] == 0.0 {
                hi -= 1;
            }
            if hi == 0 {
                break;
            }
            let mut lo = hi - 1;
            while lo > 0 && e[lo - 1] != 0.0 {
                lo -= 1;
            }
            let delta = (d[hi - 1] - d[hi]) / 2.0;
            let shift = if delta == 0.0 {
                d[hi] - e[hi - 1].abs()
            } else {
                let sign = if delta >= 0.0 { 1.0 } else { -1.0 };
                d[hi] - e[hi - 1] * e[hi - 1] / (delta + sign * (delta * delta + e[hi - 1] * e[hi - 1]).sqrt())
            };
            let mut x = d[lo] - shift;
            let mut z = e[lo];
            let mut bulge = 0.0;
            for kk in lo..hi {
                let r = x.hypot(z);
                let (c, s) = if r > 0.0 { (x / r, z / r) } else { (1.0, 0.0) };
                if kk > lo {
                    e[kk - 1] = r;
                }
                let dk = d[kk];
                let dk1 = d[kk + 1];
                let ek = e[kk];
                d[kk] = c * c * dk + 2.0 * c * s * ek + s * s * dk1;
                d[kk + 1] = s * s * dk - 2.0 * c * s * ek + c * c * dk1;
                e[kk] = c * s * (dk1 - dk) + (c * c - s * s) * ek;
                if kk + 1 < hi {
                    bulge = s * e[kk + 1];
                    e[kk + 1] *= c;
                }
                x = e[kk];
                z = bulge;
                for row in 0..n {
                    let t1 = q[row * n + kk];
                    let t2 = q[row * n + kk + 1];
                    q[row * n + kk] = c * t1 + s * t2;
                    q[row * n + kk + 1] = -s * t1 + c * t2;
                }
            }
        }
    }

    #[test]
    #[ignore = "perf timing; run with --release -- --ignored --nocapture"]
    fn tridiag_eig_qr_accum_speedup() {
        use std::time::Instant;
        for &n in &[512usize, 1024] {
            // Build a symmetric matrix, reduce to tridiagonal + Householder Q.
            let a = chol_spd(n, 0x99);
            let (d0, e0, q0) = super::tridiag_reduce(&a, n);
            let med = |mut xs: Vec<f64>| {
                xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                xs[xs.len() / 2]
            };
            let it = if n <= 512 { 3 } else { 2 };
            let (mut to, mut tn) = (Vec::new(), Vec::new());
            for _ in 0..it {
                let (mut d, mut e, mut q) = (d0.clone(), e0.clone(), q0.clone());
                let t = Instant::now();
                tridiag_eig_qr_stridn_ref(&mut d, &mut e, &mut q, n);
                to.push(t.elapsed().as_secs_f64() * 1e3);
                let (mut d2, mut e2, mut q2) = (d0.clone(), e0.clone(), q0.clone());
                let t = Instant::now();
                super::tridiag_eig_qr(&mut d2, &mut e2, Some(&mut q2), n);
                tn.push(t.elapsed().as_secs_f64() * 1e3);
                // Results must match (bit-exact accumulation, transposed storage).
                let maxq = q
                    .iter()
                    .zip(&q2)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f64, f64::max);
                assert!(maxq < 1e-12, "Q mismatch {maxq:e}");
            }
            let (o, nn) = (med(to), med(tn));
            println!("n={n:5} stride-n={o:9.2}ms transposed={nn:9.2}ms speedup={:.2}x", o / nn);
        }
    }

    #[test]
    #[ignore = "profiling; run with --release -- --ignored --nocapture"]
    fn eig_eigh_phase_profile() {
        use std::time::Instant;
        fn rnd(n: usize, seed: u64) -> Vec<f64> {
            let mut s = seed | 1;
            (0..n * n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
                })
                .collect()
        }
        // eig (non-symmetric): hessenberg_reduce + hessenberg_qr_iter (no Z).
        {
            let n = 512usize;
            let a = rnd(n, 0x77);
            let t = Instant::now();
            let (mut h, _q) = super::hessenberg_reduce(&a, n);
            let t_hess = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            super::hessenberg_qr_iter(&mut h, None, n);
            let t_qr = t.elapsed().as_secs_f64() * 1e3;
            println!("eig n={n}: hessenberg={t_hess:.1}ms qr_iter={t_qr:.1}ms");
        }
        // eigh (with vectors): tridiag_reduce (with Q) + tridiag_eig_qr (with Q).
        {
            let n = 1024usize;
            let a = chol_spd(n, 0x88);
            let t = Instant::now();
            let (mut d, mut e, mut q) = super::tridiag_reduce(&a, n);
            let t_red = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            super::tridiag_eig_qr(&mut d, &mut e, Some(&mut q), n);
            let t_qr = t.elapsed().as_secs_f64() * 1e3;
            println!("eigh n={n}: tridiag_reduce(Q)={t_red:.1}ms tridiag_qr(Q)={t_qr:.1}ms");
        }
    }

    // Old single-shift explicit-QR iteration (with the complex-2×2 deflation),
    // eigenvalues only, for the same-process A/B against the double-shift.
    fn hessenberg_qr_iter_singleshift_ref(h: &mut [f64], n: usize) {
        let eps = f64::EPSILON;
        let max_iter = super::EIGEN_QR_ITERATION_COEFF * n * n;
        let mut p = n;
        let mut cos_vals = vec![0.0; n];
        let mut sin_vals = vec![0.0; n];
        for iter in 0..max_iter {
            if p <= 1 {
                break;
            }
            while p > 1
                && h[(p - 1) * n + (p - 2)].abs()
                    <= eps * (h[(p - 2) * n + (p - 2)].abs() + h[(p - 1) * n + (p - 1)].abs())
            {
                h[(p - 1) * n + (p - 2)] = 0.0;
                p -= 1;
            }
            if p <= 1 {
                break;
            }
            let mut lo = p - 1;
            while lo > 0
                && h[lo * n + (lo - 1)].abs()
                    > eps * (h[(lo - 1) * n + (lo - 1)].abs() + h[lo * n + lo].abs())
            {
                lo -= 1;
            }
            if lo > 0 {
                h[lo * n + (lo - 1)] = 0.0;
            }
            if p - lo <= 1 {
                continue;
            }
            if p - lo == 2 {
                let a11 = h[(p - 2) * n + (p - 2)];
                let a12 = h[(p - 2) * n + (p - 1)];
                let a21 = h[(p - 1) * n + (p - 2)];
                let a22 = h[(p - 1) * n + (p - 1)];
                if (a11 + a22) * (a11 + a22) - 4.0 * (a11 * a22 - a12 * a21) < 0.0 {
                    p -= 2;
                    continue;
                }
            }
            let a11 = h[(p - 2) * n + (p - 2)];
            let a12 = h[(p - 2) * n + (p - 1)];
            let a21 = h[(p - 1) * n + (p - 2)];
            let a22 = h[(p - 1) * n + (p - 1)];
            let trace = a11 + a22;
            let det = a11 * a22 - a12 * a21;
            let disc = trace * trace - 4.0 * det;
            let mu = if disc >= 0.0 {
                let sd = disc.sqrt();
                let l1 = (trace + sd) / 2.0;
                let l2 = (trace - sd) / 2.0;
                if (l1 - a22).abs() < (l2 - a22).abs() { l1 } else { l2 }
            } else if iter % 10 == 0 {
                a22 + h[(p - 1) * n + (p - 2)].abs()
            } else {
                trace / 2.0
            };
            for i in lo..p {
                h[i * n + i] -= mu;
            }
            for k in lo..(p - 1) {
                let ff = h[k * n + k];
                let gg = h[(k + 1) * n + k];
                let r = ff.hypot(gg);
                let (c, s) = if r > 0.0 { (ff / r, gg / r) } else { (1.0, 0.0) };
                cos_vals[k] = c;
                sin_vals[k] = s;
                for j in k..n {
                    let t1 = c * h[k * n + j] + s * h[(k + 1) * n + j];
                    h[(k + 1) * n + j] = -s * h[k * n + j] + c * h[(k + 1) * n + j];
                    h[k * n + j] = t1;
                }
            }
            for k in lo..(p - 1) {
                let c = cos_vals[k];
                let s = sin_vals[k];
                let row_end = (k + 2).min(p);
                for i in 0..row_end {
                    let t1 = c * h[i * n + k] + s * h[i * n + k + 1];
                    h[i * n + k + 1] = -s * h[i * n + k] + c * h[i * n + k + 1];
                    h[i * n + k] = t1;
                }
            }
            for i in lo..p {
                h[i * n + i] += mu;
            }
        }
    }

    #[test]
    #[ignore = "perf timing; run with --release -- --ignored --nocapture"]
    fn hessenberg_qr_double_vs_single_shift() {
        use std::time::Instant;
        fn rnd(n: usize, seed: u64) -> Vec<f64> {
            let mut s = seed | 1;
            (0..n * n)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
                })
                .collect()
        }
        for &n in &[256usize, 512] {
            let a = rnd(n, 0x77);
            let (h0, _q) = super::hessenberg_reduce(&a, n);
            let med = |mut xs: Vec<f64>| {
                xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                xs[xs.len() / 2]
            };
            let it = 3;
            let (mut to, mut tn) = (Vec::new(), Vec::new());
            // eigenvalue sets must match (both reach the same real-Schur form).
            let mut hs = h0.clone();
            hessenberg_qr_iter_singleshift_ref(&mut hs, n);
            let mut hd = h0.clone();
            super::hessenberg_qr_iter(&mut hd, None, n);
            let mut es = super::extract_schur_eigenvalues(&hs, n);
            let mut ed = super::extract_schur_eigenvalues(&hd, n);
            es.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ed.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let maxd = es.iter().zip(&ed).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
            assert!(maxd < 1e-6, "eigenvalue mismatch {maxd:e} (n={n})");
            for _ in 0..it {
                let mut h1 = h0.clone();
                let t = Instant::now();
                hessenberg_qr_iter_singleshift_ref(&mut h1, n);
                to.push(t.elapsed().as_secs_f64() * 1e3);
                let mut h2 = h0.clone();
                let t = Instant::now();
                super::hessenberg_qr_iter(&mut h2, None, n);
                tn.push(t.elapsed().as_secs_f64() * 1e3);
            }
            println!("n={n:5} single-shift={:9.2}ms double-shift={:9.2}ms speedup={:.2}x", med(to.clone()), med(tn.clone()), med(to) / med(tn));
        }
    }

    // Old Hessenberg reduction with the stride-n column-walk left apply, for the
    // same-process A/B. (H only; Q not needed for the timing comparison.)
    fn hessenberg_reduce_stridn_ref(a: &[f64], n: usize) -> Vec<f64> {
        let mut h = a.to_vec();
        let mut q = vec![0.0f64; n * n];
        for i in 0..n {
            q[i * n + i] = 1.0;
        }
        let mut v = vec![0.0f64; n];
        for j in 0..n.saturating_sub(2) {
            let cn = {
                let mut s = 0.0;
                for i in (j + 1)..n {
                    s += h[i * n + j] * h[i * n + j];
                }
                s.sqrt()
            };
            if cn < f64::EPSILON {
                continue;
            }
            let sign = if h[(j + 1) * n + j] >= 0.0 { 1.0 } else { -1.0 };
            for vi in &mut v[..=j] {
                *vi = 0.0;
            }
            for (i, vi) in v[(j + 1)..n].iter_mut().enumerate() {
                *vi = h[(i + j + 1) * n + j];
            }
            v[j + 1] += sign * cn;
            let vns: f64 = v[(j + 1)..].iter().map(|x| x * x).sum();
            if vns == 0.0 {
                continue;
            }
            let scale = 2.0 / vns;
            for col in 0..n {
                let mut dot = 0.0;
                for i in (j + 1)..n {
                    dot += v[i] * h[i * n + col];
                }
                let f = scale * dot;
                for i in (j + 1)..n {
                    h[i * n + col] -= f * v[i];
                }
            }
            for row in 0..n {
                let mut dot = 0.0;
                for i in (j + 1)..n {
                    dot += v[i] * h[row * n + i];
                }
                let f = scale * dot;
                for i in (j + 1)..n {
                    h[row * n + i] -= f * v[i];
                }
            }
            for row in 0..n {
                let mut dot = 0.0;
                for i in (j + 1)..n {
                    dot += v[i] * q[row * n + i];
                }
                let f = scale * dot;
                for i in (j + 1)..n {
                    q[row * n + i] -= f * v[i];
                }
            }
        }
        h
    }

    #[test]
    #[ignore = "perf timing; run with --release -- --ignored --nocapture"]
    fn hessenberg_left_apply_speedup() {
        use std::time::Instant;
        fn rnd(n: usize, seed: u64) -> Vec<f64> {
            let mut s = seed | 1;
            (0..n * n)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
                })
                .collect()
        }
        for &n in &[512usize, 1024] {
            let a = rnd(n, 0x33);
            // bit-exactness of the new contiguous left apply vs the stride-n ref.
            let hs = hessenberg_reduce_stridn_ref(&a, n);
            let (hn, _q) = super::hessenberg_reduce(&a, n);
            let maxd = hs.iter().zip(&hn).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
            assert!(maxd == 0.0, "Hessenberg not bit-exact: {maxd:e} (n={n})");
            let med = |mut xs: Vec<f64>| {
                xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                xs[xs.len() / 2]
            };
            let it = 3;
            let (mut to, mut tn) = (Vec::new(), Vec::new());
            for _ in 0..it {
                let t = Instant::now();
                let r = hessenberg_reduce_stridn_ref(&a, n);
                std::hint::black_box(&r);
                to.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                let r = super::hessenberg_reduce(&a, n);
                std::hint::black_box(&r);
                tn.push(t.elapsed().as_secs_f64() * 1e3);
            }
            println!("n={n:5} stride-n={:9.2}ms contiguous={:9.2}ms speedup={:.2}x", med(to.clone()), med(tn.clone()), med(to) / med(tn));
        }
    }

    // Misaligned-band GEMM driver (pre-fix band_rows, same serial kernel) for A/B.
    fn packed_gemm_misaligned(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        use rayon::prelude::*;
        let mut c = vec![0.0; m * n];
        let threads = rayon::current_num_threads();
        if m >= 128 && k >= 128 && n >= 128 && threads >= 2 {
            let band_rows = m.div_ceil(threads * 4).max(1);
            c.par_chunks_mut(band_rows * n).enumerate().for_each(|(bi, c_band)| {
                let row_start = bi * band_rows;
                let rows = c_band.len() / n;
                let a_band = &a[row_start * k..row_start * k + rows * k];
                super::packed_gemm_serial(a_band, b, rows, k, n, c_band);
            });
        } else {
            super::packed_gemm_serial(a, b, m, k, n, &mut c);
        }
        c
    }

    #[test]
    #[ignore = "perf timing; run with --release -- --ignored --nocapture"]
    fn packed_gemm_band_alignment_speedup() {
        use std::time::Instant;
        fn rnd(rows: usize, cols: usize, seed: u64) -> Vec<f64> {
            let mut s = seed | 1;
            (0..rows * cols)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
                })
                .collect()
        }
        for &n in &[512usize, 1024] {
            let a = rnd(n, n, 1);
            let b = rnd(n, n, 2);
            // bit-exactness of aligned vs misaligned (same serial kernel).
            let ca = super::packed_gemm(&a, &b, n, n, n);
            let cm = packed_gemm_misaligned(&a, &b, n, n, n);
            let maxd = ca.iter().zip(&cm).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max);
            assert!(maxd == 0.0, "GEMM not bit-exact across band split: {maxd:e}");
            let med = |mut xs: Vec<f64>| {
                xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                xs[xs.len() / 2]
            };
            let it = 5;
            let (mut to, mut tn) = (Vec::new(), Vec::new());
            for _ in 0..it {
                let t = Instant::now();
                let r = packed_gemm_misaligned(&a, &b, n, n, n);
                std::hint::black_box(&r);
                to.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                let r = super::packed_gemm(&a, &b, n, n, n);
                std::hint::black_box(&r);
                tn.push(t.elapsed().as_secs_f64() * 1e3);
            }
            println!("n={n:5} misaligned={:9.2}ms aligned={:9.2}ms speedup={:.2}x", med(to.clone()), med(tn.clone()), med(to) / med(tn));
        }
    }

    #[test]
    fn eig_nxn_resolves_complex_pairs_fast() {
        // Block-diagonal: a 2×2 rotation (eigenvalues ±i) plus a real eigenvalue 2.
        // Before the 2×2-deflation fix the single-shift QR looped forever on the
        // complex pair; this must now return {±i, 2} promptly and correctly.
        let a = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0];
        let ev = super::eig_nxn(&a, 3).expect("eig");
        // ev is interleaved (re, im) per eigenvalue.
        let mut pairs: Vec<(f64, f64)> = ev.chunks_exact(2).map(|c| (c[0], c[1])).collect();
        pairs.sort_by(|x, y| {
            x.0.partial_cmp(&y.0)
                .unwrap()
                .then(x.1.partial_cmp(&y.1).unwrap())
        });
        // Expected sorted by (re, im): (0,-1), (0,1), (2,0).
        assert!((pairs[0].0 - 0.0).abs() < 1e-9 && (pairs[0].1 + 1.0).abs() < 1e-9, "{pairs:?}");
        assert!((pairs[1].0 - 0.0).abs() < 1e-9 && (pairs[1].1 - 1.0).abs() < 1e-9, "{pairs:?}");
        assert!((pairs[2].0 - 2.0).abs() < 1e-9 && (pairs[2].1 - 0.0).abs() < 1e-9, "{pairs:?}");
    }

    // Inline unblocked two-sided tridiagonalization WITH Q accumulation, for the
    // same-process A/B.
    fn tridiag_reduce_unblocked_q_ref(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut work = a.to_vec();
        let mut q = vec![0.0f64; n * n];
        for i in 0..n {
            q[i * n + i] = 1.0;
        }
        let mut v = vec![0.0f64; n];
        let mut dbuf = vec![0.0f64; n];
        let mut f_vec = vec![0.0f64; n];
        for j in 0..n.saturating_sub(2) {
            let cn = {
                let mut s = 0.0;
                for i in (j + 1)..n {
                    s += work[i * n + j] * work[i * n + j];
                }
                s.sqrt()
            };
            if cn < f64::EPSILON * work[j * n + j].abs().max(1.0) {
                continue;
            }
            let sign = if work[(j + 1) * n + j] >= 0.0 { 1.0 } else { -1.0 };
            for vi in &mut v[..=j] {
                *vi = 0.0;
            }
            for (idx, vi) in v[(j + 1)..n].iter_mut().enumerate() {
                *vi = work[(j + 1 + idx) * n + j];
            }
            v[j + 1] += sign * cn;
            let vns: f64 = v[(j + 1)..].iter().map(|x| x * x).sum();
            if vns == 0.0 {
                continue;
            }
            let scale = 2.0 / vns;
            for dc in dbuf.iter_mut() {
                *dc = 0.0;
            }
            for i in (j + 1)..n {
                let vi = v[i];
                let row = &work[i * n..i * n + n];
                for (dc, &w) in dbuf.iter_mut().zip(row.iter()) {
                    *dc += vi * w;
                }
            }
            for (fc, &dc) in f_vec.iter_mut().zip(dbuf.iter()) {
                *fc = scale * dc;
            }
            for i in (j + 1)..n {
                let vi = v[i];
                let row = &mut work[i * n..i * n + n];
                for (w, &fc) in row.iter_mut().zip(f_vec.iter()) {
                    *w -= fc * vi;
                }
            }
            for row in 0..n {
                let mut dot = 0.0;
                for i in (j + 1)..n {
                    dot += v[i] * work[row * n + i];
                }
                let f = scale * dot;
                for i in (j + 1)..n {
                    work[row * n + i] -= f * v[i];
                }
            }
            for row in 0..n {
                let mut dot = 0.0;
                for i in (j + 1)..n {
                    dot += v[i] * q[row * n + i];
                }
                let f = scale * dot;
                for i in (j + 1)..n {
                    q[row * n + i] -= f * v[i];
                }
            }
        }
        let d: Vec<f64> = (0..n).map(|i| work[i * n + i]).collect();
        let e: Vec<f64> = (0..n - 1).map(|i| work[i * n + i + 1]).collect();
        (d, e, q)
    }

    #[test]
    #[ignore = "perf timing; run with --release -- --ignored --nocapture"]
    fn blocked_tridiag_q_speedup() {
        use std::time::Instant;
        for &n in &[512usize, 1024] {
            let a = chol_spd(n, 0x1234);
            let med = |mut xs: Vec<f64>| {
                xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                xs[xs.len() / 2]
            };
            let it = if n <= 512 { 3 } else { 2 };
            let (mut to, mut tn) = (Vec::new(), Vec::new());
            for _ in 0..it {
                let t = Instant::now();
                let r = tridiag_reduce_unblocked_q_ref(&a, n);
                std::hint::black_box(&r);
                to.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                let r2 = super::tridiag_reduce_blocked(&a, n, true);
                std::hint::black_box(&r2);
                tn.push(t.elapsed().as_secs_f64() * 1e3);
            }
            println!("n={n:5} unblocked-Q={:9.2}ms blocked-Q={:9.2}ms speedup={:.2}x", med(to.clone()), med(tn.clone()), med(to) / med(tn));
        }
    }

    #[test]
    fn blocked_tridiag_with_q_reconstructs() {
        // n >= TRIDIAG_BLOCK_MIN -> tridiag_reduce routes to the blocked with-Q
        // path. Verify A = Q·T·Q^T (T tridiagonal from d,e) and Q orthogonal.
        for &n in &[400usize, 512] {
            let a = chol_spd(n, 0x61 + n as u64);
            let (d, e, q) = super::tridiag_reduce(&a, n);
            assert_eq!(q.len(), n * n, "Q must be accumulated");
            let mut tqt = vec![0.0f64; n * n];
            for k in 0..n {
                for j in 0..n {
                    let mut s = d[k] * q[j * n + k];
                    if k > 0 {
                        s += e[k - 1] * q[j * n + (k - 1)];
                    }
                    if k + 1 < n {
                        s += e[k] * q[j * n + (k + 1)];
                    }
                    tqt[k * n + j] = s;
                }
            }
            let mut max_recon = 0.0f64;
            let mut max_orth = 0.0f64;
            for i in 0..n {
                for j in 0..n {
                    let mut r = 0.0;
                    let mut o = 0.0;
                    for k in 0..n {
                        r += q[i * n + k] * tqt[k * n + j];
                        o += q[k * n + i] * q[k * n + j];
                    }
                    max_recon = max_recon.max((r - a[i * n + j]).abs() / (1.0 + a[i * n + j].abs()));
                    let t = if i == j { 1.0 } else { 0.0 };
                    max_orth = max_orth.max((o - t).abs());
                }
            }
            assert!(max_recon < 1e-9, "Q·T·Q^T=A err {max_recon:e} (n={n})");
            assert!(max_orth < 1e-9, "Q orthogonality err {max_orth:e} (n={n})");
        }
    }

    #[test]
    fn tridiag_symmetric_matvec_serial_matches_full_row_dot_bits() {
        for (n, start) in [(96usize, 13usize), (224, 17)] {
            let mut a = vec![0.0f64; n * n];
            for i in 0..n {
                for j in i..n {
                    let value = if i == j {
                        (n + i + 1) as f64
                    } else {
                        ((i * 131 + j * 17 + 7) % 97) as f64 / 97.0 - 0.5
                    };
                    a[i * n + j] = value;
                    a[j * n + i] = value;
                }
            }
            let mut v = vec![0.0f64; n];
            for (i, vi) in v.iter_mut().enumerate().skip(start) {
                *vi = ((i * 19 + 3) % 43) as f64 / 23.0 - 0.8;
            }
            let mut half = vec![123.0f64; n];
            let mut full = vec![123.0f64; n];
            super::tridiag_symmetric_matvec_serial(&a, n, start, &v, &mut half);
            for i in start..n {
                let row = &a[i * n + start..i * n + n];
                let mut s = 0.0;
                for (offset, &entry) in row.iter().enumerate() {
                    s += entry * v[start + offset];
                }
                full[i] = s;
            }
            for i in start..n {
                assert_eq!(
                    half[i].to_bits(),
                    full[i].to_bits(),
                    "n={n} row {i} drifted"
                );
            }
        }
    }

    #[test]
    fn blocked_tridiag_matches_unblocked() {
        // n below TRIDIAG_BLOCK_MIN -> tridiag_reduce_impl takes the unblocked
        // path; compare its (d,e) to the directly-called blocked kernel.
        for &n in &[128usize, 160, 200] {
            let a = chol_spd(n, 0x31 + n as u64); // symmetric
            let (db, eb, _) = super::tridiag_reduce_blocked(&a, n, false);
            let (du, eu, _) = super::tridiag_reduce_impl(&a, n, false);
            let mut max_d = 0.0f64;
            let mut max_e = 0.0f64;
            for i in 0..n {
                max_d = max_d.max((db[i] - du[i]).abs() / (1.0 + du[i].abs()));
            }
            for i in 0..n - 1 {
                max_e = max_e.max((eb[i].abs() - eu[i].abs()).abs() / (1.0 + eu[i].abs()));
            }
            assert!(max_d < 1e-8, "blocked tridiag d err {max_d:e} (n={n})");
            assert!(max_e < 1e-8, "blocked tridiag e err {max_e:e} (n={n})");
        }
    }

    #[test]
    fn blocked_tridiag_parallel_matvec_matches_unblocked_large() {
        // n=1152 > TRIDIAG_MATVEC_PAR_MIN, so the leading panel matvecs (u = A·v,
        // h up to 1151 >= 1024) run on the PARALLEL path. Compare the resulting
        // (d,e) tridiagonal against the explicit serial unblocked reference to prove
        // the parallelized symmetric matvec still produces a correct reduction.
        let n = 1152usize;
        let a = chol_spd(n, 0xa5); // dense symmetric (dense reflectors exercise matvec)
        let (db, eb, _) = super::tridiag_reduce_blocked(&a, n, false);
        let (du, eu, _) = tridiag_reduce_unblocked_q_ref(&a, n);
        let mut max_d = 0.0f64;
        let mut max_e = 0.0f64;
        for i in 0..n {
            max_d = max_d.max((db[i] - du[i]).abs() / (1.0 + du[i].abs()));
        }
        for i in 0..n - 1 {
            max_e = max_e.max((eb[i].abs() - eu[i].abs()).abs() / (1.0 + eu[i].abs()));
        }
        assert!(max_d < 1e-8, "parallel-matvec tridiag d err {max_d:e}");
        assert!(max_e < 1e-8, "parallel-matvec tridiag e err {max_e:e}");
    }

    #[test]
    fn tridiag_rank2k_fused_update_preserves_spectra_and_golden_sha256() {
        let n = 160usize;
        let mut cases = Vec::new();
        cases.push(chol_spd(n, 0x2130_0001));

        let mut repeated = vec![0.0f64; n * n];
        for i in 0..n {
            repeated[i * n + i] = if i < n / 2 { 2.0 } else { 9.0 };
            if i + 1 < n {
                repeated[i * n + i + 1] = 0.125;
                repeated[(i + 1) * n + i] = 0.125;
            }
        }
        cases.push(repeated);

        let mut indefinite = vec![0.0f64; n * n];
        for i in 0..n {
            indefinite[i * n + i] = i as f64 - (n as f64 / 2.0);
            for j in (i + 1)..n {
                let value = ((i * 19 + j * 37 + 11) % 29) as f64 / 47.0 - 0.3;
                indefinite[i * n + j] = value;
                indefinite[j * n + i] = value;
            }
        }
        cases.push(indefinite);

        let mut hasher = Sha256::new();
        for (case_idx, a) in cases.iter().enumerate() {
            let (mut ref_d, mut ref_e) = tridiag_unblocked_ref(a, n);
            super::tridiag_eig_qr(&mut ref_d, &mut ref_e, None, n);
            ref_d.sort_by(|a, b| a.total_cmp(b));

            let mut eigvals = super::eigvalsh_nxn(a, n).expect("blocked eigvalsh");
            eigvals.sort_by(|a, b| a.total_cmp(b));
            for (idx, (&lhs, &rhs)) in ref_d.iter().zip(&eigvals).enumerate() {
                let err = (lhs - rhs).abs() / (1.0 + lhs.abs().max(rhs.abs()));
                assert!(
                    err < 2e-8,
                    "case {case_idx} eigval {idx} drifted: reference {lhs} vs blocked {rhs}, rel {err:e}"
                );
            }
            for value in eigvals {
                hasher.update(value.to_bits().to_le_bytes());
            }
        }

        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        // Re-pinned 2026-06-19: eigvalsh's eigenvalues-only QR chase moved to
        // `scaled_hypot` (benign ~1 ulp/rotation). The correctness gate above
        // (blocked eigvalsh vs unblocked-ref QR, rel err < 2e-8) still passes for all
        // three cases, so this only shifts the last bits of the output stream.
        assert_eq!(
            digest, "d8a5154cdf2b005605b832840983ece912dac6252c0d6b59452f47256b8cb2f8",
            "fused rank-2k tridiagonalization golden digest drifted: {digest}"
        );
    }

    #[test]
    fn symmetric_rank2k_triangular_update_matches_full_reference_and_golden_sha256() {
        let active = 5usize;
        let h = 160usize;
        let n = active + h + 3;
        let mut hasher = Sha256::new();

        for &nb in &[64usize, 128] {
            let mut work = vec![0.0f64; n * n];
            for row in 0..n {
                for col in row..n {
                    let value = ((row * 17 + col * 29 + nb + 7) % 43) as f64 / 59.0 - 0.35;
                    work[row * n + col] = value;
                    work[col * n + row] = value;
                }
            }

            let mut vv = vec![0.0f64; h * nb];
            let mut ww = vec![0.0f64; h * nb];
            for row in 0..h {
                for col in 0..nb {
                    vv[row * nb + col] = ((row * 13 + col * 19 + nb + 3) % 37) as f64 / 47.0 - 0.25;
                    ww[row * nb + col] = ((row * 23 + col * 11 + nb + 5) % 31) as f64 / 41.0 - 0.2;
                }
            }

            let mut full = work.clone();
            for row_idx in 0..h {
                let vi = &vv[row_idx * nb..row_idx * nb + nb];
                let wi = &ww[row_idx * nb..row_idx * nb + nb];
                for col_idx in 0..h {
                    let vj = &vv[col_idx * nb..col_idx * nb + nb];
                    let wj = &ww[col_idx * nb..col_idx * nb + nb];
                    let mut delta = 0.0f64;
                    for k in 0..nb {
                        delta += vi[k] * wj[k] + wi[k] * vj[k];
                    }
                    full[(active + row_idx) * n + active + col_idx] -= delta;
                }
            }

            let mut triangular = work;
            super::sbr_apply_symmetric_rank2k_update(
                &mut triangular,
                n,
                active,
                h,
                nb,
                &vv,
                &ww,
            );
            for row in 0..h {
                for col in 0..h {
                    let idx = (active + row) * n + active + col;
                    assert_eq!(
                        triangular[idx].to_bits(),
                        full[idx].to_bits(),
                        "rank2k active block drifted at nb={nb}, row={row}, col={col}"
                    );
                    hasher.update(triangular[idx].to_bits().to_le_bytes());
                }
            }
        }

        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "47e81b08104eb087d27fd21a6674cdbdfea002a91b0c59fceb72d7052aec66b4",
            "triangular rank-2k golden digest drifted: {digest}"
        );
    }

    #[test]
    fn sbr_cross_wy_strided_update_matches_materialized_reference_and_golden_sha256() {
        let mut hasher = Sha256::new();
        for &(active, h, nb) in &[(8usize, 17usize, 8usize), (96, 160, 64), (140, 192, 128)] {
            let n = active + h + 5;
            let mut work = vec![0.0f64; n * n];
            for row in 0..n {
                for col in row..n {
                    let value = ((row * 31 + col * 17 + active + h + nb) % 53) as f64 / 67.0 - 0.37;
                    work[row * n + col] = value;
                    work[col * n + row] = value;
                }
            }
            let cvt: Vec<f64> = (0..active * nb)
                .map(|idx| ((idx * 19 + active + nb) % 47) as f64 / 59.0 - 0.29)
                .collect();
            let vt: Vec<f64> = (0..nb * h)
                .map(|idx| ((idx * 23 + h + nb) % 61) as f64 / 71.0 - 0.31)
                .collect();

            let mut materialized = work.clone();
            let upd = super::packed_gemm(&cvt, &vt, active, nb, h);
            for row in 0..active {
                let dst = row * n + active;
                for col in 0..h {
                    let value = materialized[dst + col] - upd[row * h + col];
                    materialized[dst + col] = value;
                    materialized[(active + col) * n + row] = value;
                }
            }

            let mut strided = work;
            super::sbr_apply_cross_wy_update(&mut strided, n, active, h, nb, &cvt, &vt);
            assert_eq!(strided.len(), materialized.len());
            for (idx, (&lhs, &rhs)) in strided.iter().zip(&materialized).enumerate() {
                assert_eq!(
                    lhs.to_bits(),
                    rhs.to_bits(),
                    "SBR cross WY direct update drifted at flat index {idx}, active={active}, h={h}, nb={nb}"
                );
                hasher.update(lhs.to_bits().to_le_bytes());
            }
        }

        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "f3e6bf77ab1fed8c1bffa2cb8853843fca6556a4ba408f4dfbf6bf16ddd21187",
            "SBR cross WY strided update golden digest drifted: {digest}"
        );
    }

    fn q_t_a_q(a: &[f64], q: &[f64], n: usize) -> Vec<f64> {
        let mut aq = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += a[i * n + k] * q[k * n + j];
                }
                aq[i * n + j] = s;
            }
        }
        let mut out = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += q[k * n + i] * aq[k * n + j];
                }
                out[i * n + j] = s;
            }
        }
        out
    }

    fn assert_stage1_band_oracle(a: &[f64], n: usize) {
        let bandwidth = super::sbr_stage1_band_width().min(n - 1);
        let (dense_band, lower_band, q) =
            super::sbr_stage1_dense_to_band_impl(a, n, true).expect("SBR stage-1");
        assert_eq!(dense_band.len(), n * n);
        assert_eq!(q.len(), n * n);
        let unpacked = super::unpack_lower_band(&lower_band, n, bandwidth);
        for (idx, (&dense, &packed)) in dense_band.iter().zip(&unpacked).enumerate() {
            assert_eq!(dense.to_bits(), packed.to_bits(), "compact band drift at flat index {idx}");
        }

        let mut max_outside = 0.0f64;
        let mut max_sym = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                if i.abs_diff(j) > bandwidth {
                    max_outside = max_outside.max(dense_band[i * n + j].abs());
                }
                max_sym = max_sym.max((dense_band[i * n + j] - dense_band[j * n + i]).abs());
            }
        }
        assert_eq!(
            max_outside.to_bits(),
            0.0f64.to_bits(),
            "outside-band entries must be exact zero"
        );
        assert_eq!(
            max_sym.to_bits(),
            0.0f64.to_bits(),
            "band matrix must stay exactly symmetric"
        );

        let oracle = q_t_a_q(a, &q, n);
        let mut max_recon = 0.0f64;
        let mut max_orth = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                max_recon = max_recon.max(
                    (oracle[i * n + j] - dense_band[i * n + j]).abs()
                        / (1.0 + dense_band[i * n + j].abs()),
                );
                let mut dot = 0.0;
                for k in 0..n {
                    dot += q[k * n + i] * q[k * n + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                max_orth = max_orth.max((dot - expected).abs());
            }
        }
        assert!(max_recon < 2e-8, "Q^T A Q != band, rel err {max_recon:e}");
        assert!(max_orth < 2e-10, "Q orthogonality err {max_orth:e}");

        let original = eigvalsh_nxn(a, n).expect("original eigvalsh");
        let reduced = eigvalsh_nxn(&dense_band, n).expect("band eigvalsh");
        for (idx, (&lhs, &rhs)) in original.iter().zip(&reduced).enumerate() {
            let err = (lhs - rhs).abs() / (1.0 + lhs.abs().max(rhs.abs()));
            assert!(err < 2e-8, "eigval {idx} drifted: {lhs} vs {rhs}, rel {err:e}");
        }
    }

    #[test]
    fn sbr_stage1_band_q_oracle_preserves_symmetric_spectra() {
        let n = 128usize;

        let spd = chol_spd(n, 0x5196);
        assert_stage1_band_oracle(&spd, n);

        let mut repeated = vec![0.0f64; n * n];
        for i in 0..n {
            repeated[i * n + i] = if i < n / 2 { 3.0 } else { 7.0 };
            if i + 1 < n {
                repeated[i * n + i + 1] = 0.125;
                repeated[(i + 1) * n + i] = 0.125;
            }
        }
        assert_stage1_band_oracle(&repeated, n);

        let mut indefinite = vec![0.0f64; n * n];
        for i in 0..n {
            indefinite[i * n + i] = i as f64 - (n as f64 / 2.0);
            for j in (i + 1)..n {
                let value = ((i * 17 + j * 31 + 5) % 19) as f64 / 37.0 - 0.25;
                indefinite[i * n + j] = value;
                indefinite[j * n + i] = value;
            }
        }
        assert_stage1_band_oracle(&indefinite, n);
    }

    #[test]
    fn sbr_stage1_compact_band_golden_sha256() {
        let n = 128usize;
        let a = chol_spd(n, 0x5b12_0001);
        let (_dense_band, lower_band, _q) =
            super::sbr_stage1_dense_to_band_impl(&a, n, false).expect("SBR stage-1");
        let mut hasher = Sha256::new();
        for value in &lower_band {
            hasher.update(value.to_bits().to_le_bytes());
        }
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "a746956cc26746ad23415eb96d7692f844ab574c0fc186975bd073d7ef967ed6",
            "SBR stage-1 compact-band golden digest drifted: {digest}"
        );
    }

    #[test]
    #[ignore = "perf timing; run with --release -- --ignored --nocapture"]
    fn blocked_tridiag_speedup_report() {
        use std::time::Instant;
        for &n in &[512usize, 1024, 2048] {
            let a = chol_spd(n, 0x1234);
            let it = if n <= 1024 { 3 } else { 2 };
            let med = |mut xs: Vec<f64>| {
                xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                xs[xs.len() / 2]
            };
            let mut tu = Vec::new();
            let mut tb = Vec::new();
            for _ in 0..it {
                let t = Instant::now();
                let r = super::tridiag_reduce_impl(&a, n, false); // unblocked? no — gated
                std::hint::black_box(&r);
                tu.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                let r = super::tridiag_reduce_blocked(&a, n, false);
                std::hint::black_box(&r);
                tb.push(t.elapsed().as_secs_f64() * 1e3);
            }
            // tridiag_reduce_impl routes to blocked for n>=384, so the "unblocked"
            // timing above is actually blocked; recompute a true unblocked time by
            // calling the inline reference.
            let mut tu2 = Vec::new();
            for _ in 0..it {
                let t = Instant::now();
                let r = tridiag_unblocked_ref(&a, n);
                std::hint::black_box(&r);
                tu2.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let (u, b) = (med(tu2), med(tb));
            let _ = med(tu);
            println!("n={n:5} unblocked={u:9.2}ms blocked={b:9.2}ms speedup={:.2}x", u / b);
        }
    }

    // Inline unblocked values-only tridiagonalization (matches tridiag_reduce_impl
    // with accumulate_q=false) for the A/B.
    fn tridiag_unblocked_ref(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut work = a.to_vec();
        let mut v = vec![0.0f64; n];
        let mut d = vec![0.0f64; n];
        let mut f_vec = vec![0.0f64; n];
        for j in 0..n.saturating_sub(2) {
            let cn = {
                let mut s = 0.0;
                for i in (j + 1)..n {
                    s += work[i * n + j] * work[i * n + j];
                }
                s.sqrt()
            };
            if cn < f64::EPSILON * work[j * n + j].abs().max(1.0) {
                continue;
            }
            let sign = if work[(j + 1) * n + j] >= 0.0 { 1.0 } else { -1.0 };
            for vi in &mut v[..=j] {
                *vi = 0.0;
            }
            for (idx, vi) in v[(j + 1)..n].iter_mut().enumerate() {
                *vi = work[(j + 1 + idx) * n + j];
            }
            v[j + 1] += sign * cn;
            let vns: f64 = v[(j + 1)..].iter().map(|x| x * x).sum();
            if vns == 0.0 {
                continue;
            }
            let scale = 2.0 / vns;
            for dc in d.iter_mut() {
                *dc = 0.0;
            }
            for i in (j + 1)..n {
                let vi = v[i];
                let row = &work[i * n..i * n + n];
                for (dc, &w) in d.iter_mut().zip(row.iter()) {
                    *dc += vi * w;
                }
            }
            for (fc, &dc) in f_vec.iter_mut().zip(d.iter()) {
                *fc = scale * dc;
            }
            for i in (j + 1)..n {
                let vi = v[i];
                let row = &mut work[i * n..i * n + n];
                for (w, &fc) in row.iter_mut().zip(f_vec.iter()) {
                    *w -= fc * vi;
                }
            }
            for row in 0..n {
                let mut dot = 0.0;
                for i in (j + 1)..n {
                    dot += v[i] * work[row * n + i];
                }
                let f = scale * dot;
                for i in (j + 1)..n {
                    work[row * n + i] -= f * v[i];
                }
            }
        }
        let dd: Vec<f64> = (0..n).map(|i| work[i * n + i]).collect();
        let ee: Vec<f64> = (0..n - 1).map(|i| work[i * n + i + 1]).collect();
        (dd, ee)
    }

    #[test]
    fn blocked_qr_reconstructs_and_matches_unblocked() {
        // n below QR_BLOCK_MIN -> call the blocked kernel directly; compare to the
        // unblocked path. Verify Q·R = A, Q orthogonal, and blocked == unblocked.
        for &n in &[160usize, 200, 256] {
            let a = qr_rand(n, 0x41 + n as u64);
            let (qb, rb) = super::qr_blocked(&a, n).expect("blocked qr");
            let (qu, ru) = super::qr_nxn(&a, n).expect("unblocked qr");
            let mut max_recon = 0.0f64;
            let mut max_orth = 0.0f64;
            let mut max_q = 0.0f64;
            let mut max_r = 0.0f64;
            for i in 0..n {
                for j in 0..n {
                    // Q·R
                    let mut s = 0.0;
                    for k in 0..n {
                        s += qb[i * n + k] * rb[k * n + j];
                    }
                    max_recon = max_recon.max((s - a[i * n + j]).abs() / (1.0 + a[i * n + j].abs()));
                    // Q^T·Q
                    let mut o = 0.0;
                    for k in 0..n {
                        o += qb[k * n + i] * qb[k * n + j];
                    }
                    let target = if i == j { 1.0 } else { 0.0 };
                    max_orth = max_orth.max((o - target).abs());
                    max_q = max_q.max((qb[i * n + j] - qu[i * n + j]).abs() / (1.0 + qu[i * n + j].abs()));
                    max_r = max_r.max((rb[i * n + j] - ru[i * n + j]).abs() / (1.0 + ru[i * n + j].abs()));
                }
            }
            assert!(max_recon < 1e-9, "Q·R=A err {max_recon:e} (n={n})");
            assert!(max_orth < 1e-9, "Q orthogonality err {max_orth:e} (n={n})");
            assert!(max_q < 1e-9, "blocked vs unblocked Q err {max_q:e} (n={n})");
            assert!(max_r < 1e-9, "blocked vs unblocked R err {max_r:e} (n={n})");
        }
    }

    #[test]
    fn qr_nxn_serial_matches_reference_and_golden_sha256() {
        // The unblocked qr_nxn path (n < QR_BLOCK_MIN) applies each reflector to R
        // as a cache-friendly two-pass row-contiguous transform and accumulates Q
        // per row. Both must be BYTE-IDENTICAL to the naive per-column Householder
        // reference (same i-ascending dot order, same per-element products). n =
        // 200 and 256 stay on the unblocked path under the shipped threshold.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("build local rayon pool");
        for &n in &[200usize, 256] {
            let a = qr_rand(n, 0x77 + n as u64);
            let (q, r) = pool.install(|| super::qr_nxn(&a, n).expect("qr"));
            let (qref, rref) = qr_unblocked_ref(&a, n);
            for (p, s) in q.iter().zip(&qref) {
                assert_eq!(p.to_bits(), s.to_bits(), "Q drifted from serial ref (n={n})");
            }
            for (p, s) in r.iter().zip(&rref) {
                assert_eq!(p.to_bits(), s.to_bits(), "R drifted from serial ref (n={n})");
            }
        }

        // Golden SHA-256 over the n=256 Q‖R output bits pins the exact numeric
        // result against future refactors of the Householder kernel.
        let n = 256usize;
        let a = qr_rand(n, 0x77 + n as u64);
        let (q, r) = pool.install(|| super::qr_nxn(&a, n).expect("qr"));
        let mut hasher = Sha256::new();
        for v in q.iter().chain(r.iter()) {
            hasher.update(v.to_bits().to_le_bytes());
        }
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "9b6c201de83d8db509f597f0aa1ccab6b3386b7b00d003a78af269d8fbcb1617",
            "qr_nxn serial golden digest drifted"
        );
    }

    #[test]
    fn qr_nxn_blocked_path_matches_serial_within_tolerance_and_golden_sha256() {
        // n = 768 >= QR_BLOCK_MIN, so `qr_nxn` now dispatches to the compact-WY
        // blocked path. Householder QR is not bit-reproducible across blockings
        // (LAPACK isn't either), so behaviour parity is tolerance-based: the
        // blocked Q and R must agree with the serial Householder reference to
        // ~1e-9 relative. The golden SHA-256 then pins the exact blocked bits so
        // future kernel edits can't silently drift the result.
        let n = 768usize;
        let a = qr_rand(n, 0x53 + n as u64);
        let (qb, rb) = super::qr_nxn(&a, n).expect("blocked qr via qr_nxn");
        let (qref, rref) = qr_unblocked_ref(&a, n);
        let mut max_q = 0.0f64;
        let mut max_r = 0.0f64;
        for (p, s) in qb.iter().zip(&qref) {
            max_q = max_q.max((p - s).abs() / (1.0 + s.abs()));
        }
        for (p, s) in rb.iter().zip(&rref) {
            max_r = max_r.max((p - s).abs() / (1.0 + s.abs()));
        }
        assert!(max_q < 1e-9, "blocked Q vs serial ref err {max_q:e}");
        assert!(max_r < 1e-9, "blocked R vs serial ref err {max_r:e}");

        let mut hasher = Sha256::new();
        for v in qb.iter().chain(rb.iter()) {
            hasher.update(v.to_bits().to_le_bytes());
        }
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "95caa242da11e5c573d75f23419e994556fe06443af250f8dba1fc58c1af0227",
            "qr_nxn blocked-path golden digest drifted"
        );
    }

    #[test]
    fn qr_nxn_blocked_512_path_matches_serial_within_tolerance_and_golden_sha256() {
        // n = 512 is the lowered compact-WY crossover. QR is not bit-reproducible
        // across blockings, so the behavior proof is LAPACK-style tolerance
        // equivalence to the serial Householder reference plus a golden lock on
        // the exact blocked output bits for this newly routed size.
        let n = 512usize;
        let a = qr_rand(n, 0x53 + n as u64);
        let (qb, rb) = super::qr_nxn(&a, n).expect("blocked qr via qr_nxn");
        let (qref, rref) = qr_unblocked_ref(&a, n);
        let mut max_q = 0.0f64;
        let mut max_r = 0.0f64;
        for (p, s) in qb.iter().zip(&qref) {
            max_q = max_q.max((p - s).abs() / (1.0 + s.abs()));
        }
        for (p, s) in rb.iter().zip(&rref) {
            max_r = max_r.max((p - s).abs() / (1.0 + s.abs()));
        }
        assert!(max_q < 1e-9, "blocked 512 Q vs serial ref err {max_q:e}");
        assert!(max_r < 1e-9, "blocked 512 R vs serial ref err {max_r:e}");

        let mut hasher = Sha256::new();
        for v in qb.iter().chain(rb.iter()) {
            hasher.update(v.to_bits().to_le_bytes());
        }
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "a74134dcdca6dd5da9f40c65d41a6c726782524a7e4cc9266ffe6c40c1165219",
            "qr_nxn blocked 512 golden digest drifted: {digest}"
        );
    }

    #[test]
    #[ignore = "perf timing; run with --release -- --ignored --nocapture"]
    fn blocked_qr_speedup_report() {
        use std::time::Instant;
        // Compares a serial Householder reference vs the compact-WY `qr_blocked`
        // GEMM path, same process (load-independent — cross-worker criterion A/B
        // is not). `qr_unblocked_ref` is the naive per-column serial form, ~2x
        // slower than `qr_nxn`'s real two-pass path, so the printed ratio is an
        // upper bound; the real two-pass crossover (measured separately against
        // `super::qr_nxn` for n < old threshold) is ~768, which is QR_BLOCK_MIN.
        for &n in &[512usize, 768, 1024, 1536] {
            let a = qr_rand(n, 0x1234);
            let it = if n <= 1024 { 5 } else { 3 };
            let med = |mut xs: Vec<f64>| {
                xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                xs[xs.len() / 2]
            };
            let mut tu = Vec::new();
            let mut tb = Vec::new();
            for _ in 0..it {
                let t = Instant::now();
                let r = qr_unblocked_ref(&a, n);
                std::hint::black_box(&r);
                tu.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                let r = super::qr_blocked(&a, n).unwrap();
                std::hint::black_box(&r);
                tb.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let (u, b) = (med(tu), med(tb));
            println!("n={n:5} unblocked={u:9.2}ms blocked={b:9.2}ms speedup={:.2}x", u / b);
        }
    }

    // Inline unblocked Householder QR (matches qr_nxn's algorithm) for the A/B.
    fn qr_unblocked_ref(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut q = vec![0.0f64; n * n];
        for i in 0..n {
            q[i * n + i] = 1.0;
        }
        let mut r = a.to_vec();
        let mut v = vec![0.0f64; n];
        for k in 0..n {
            let mut cns = 0.0;
            for i in k..n {
                cns += r[i * n + k] * r[i * n + k];
            }
            let cn = cns.sqrt();
            if cn == 0.0 {
                continue;
            }
            let sign = if r[k * n + k] >= 0.0 { 1.0 } else { -1.0 };
            for i in k..n {
                v[i] = r[i * n + k];
            }
            v[k] += sign * cn;
            let vns: f64 = v[k..].iter().map(|x| x * x).sum();
            if vns == 0.0 {
                continue;
            }
            let scale = 2.0 / vns;
            for j in k..n {
                let mut dot = 0.0;
                for i in k..n {
                    dot += v[i] * r[i * n + j];
                }
                let f = scale * dot;
                for i in k..n {
                    r[i * n + j] -= f * v[i];
                }
            }
            for i in 0..n {
                let mut dot = 0.0;
                for j in k..n {
                    dot += q[i * n + j] * v[j];
                }
                let f = scale * dot;
                for j in k..n {
                    q[i * n + j] -= f * v[j];
                }
            }
        }
        (q, r)
    }

    #[test]
    fn blocked_trsm_matches_unblocked_and_solves() {
        // Blocked multi-RHS solve (inv-shaped, m=n). Call the blocked TRSM
        // directly (n below TRSM_BLOCK_MIN) and compare to the unblocked path,
        // and verify A·X = B.
        for &n in &[160usize, 200, 256] {
            let m = n;
            let a = lu_spd_like(n, 0x91 + n as u64);
            let (lu, perm, _s) = super::lu_factor_nxn(&a, n).expect("lu");
            let mut bmat = vec![0.0f64; n * m]; // identity RHS (inv)
            for i in 0..n {
                bmat[i * m + i] = 1.0;
            }
            let xb = super::lu_forward_back_multi_blocked(&lu, &perm, &bmat, n, m);
            let xu = super::lu_forward_back_multi(&lu, &perm, &bmat, n, m);
            let mut max_diff = 0.0f64;
            for idx in 0..n * m {
                max_diff = max_diff.max((xb[idx] - xu[idx]).abs() / (1.0 + xu[idx].abs()));
            }
            assert!(max_diff < 1e-9, "blocked vs unblocked TRSM err {max_diff:e} (n={n})");
            // A·X must reconstruct B (identity).
            let mut max_recon = 0.0f64;
            for i in 0..n {
                for col in 0..m {
                    let mut s = 0.0;
                    for k in 0..n {
                        s += a[i * n + k] * xb[k * m + col];
                    }
                    let target = if i == col { 1.0 } else { 0.0 };
                    max_recon = max_recon.max((s - target).abs());
                }
            }
            assert!(max_recon < 1e-7, "A·X=I reconstruction err {max_recon:e} (n={n})");
        }
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn inv_nxn_1024_blocked_trsm_reconstructs_and_hashes() {
        let n = 1024usize;
        let a = lu_spd_like(n, 0x5A17_1024);
        let inverse = super::inv_nxn(&a, n).expect("1024 inverse");

        let mut max_recon = 0.0f64;
        for row in 0..n {
            for col in 0..n {
                let mut sum = 0.0f64;
                for k in 0..n {
                    sum += a[row * n + k] * inverse[k * n + col];
                }
                let target = if row == col { 1.0 } else { 0.0 };
                max_recon = max_recon.max((sum - target).abs());
            }
        }
        assert!(
            max_recon < 1e-7,
            "1024 inverse reconstruction drift {max_recon:e}"
        );

        let mut hasher = Sha256::new();
        hasher.update(n.to_le_bytes());
        for value in &inverse {
            hasher.update(value.to_bits().to_le_bytes());
        }
        hasher.update(max_recon.to_bits().to_le_bytes());
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "c329cbe7cb25210c5c678dc3229349ce0ad56c74c134cf30f75fd507f20793b1",
            "1024 inverse blocked-TRSM golden digest drifted: {digest}"
        );
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn inv_nxn_768_blocked_trsm_reconstructs_and_hashes() {
        let n = 768usize;
        let a = lu_spd_like(n, 0x5A17_0768);
        let inverse = super::inv_nxn(&a, n).expect("768 inverse");

        let mut max_recon = 0.0f64;
        for row in 0..n {
            for col in 0..n {
                let mut sum = 0.0f64;
                for k in 0..n {
                    sum += a[row * n + k] * inverse[k * n + col];
                }
                let target = if row == col { 1.0 } else { 0.0 };
                max_recon = max_recon.max((sum - target).abs());
            }
        }
        assert!(
            max_recon < 1e-7,
            "768 inverse reconstruction drift {max_recon:e}"
        );

        let mut hasher = Sha256::new();
        hasher.update(n.to_le_bytes());
        for value in &inverse {
            hasher.update(value.to_bits().to_le_bytes());
        }
        hasher.update(max_recon.to_bits().to_le_bytes());
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "367d18d0dee0c070249d1e4fc2a92bc64167430edde840baa57ad0fad6e1bd8a",
            "768 inverse blocked-TRSM golden digest drifted: {digest}"
        );
    }

    #[test]
    #[ignore = "perf timing; run with --release -- --ignored --nocapture"]
    fn blocked_trsm_speedup_report() {
        use std::time::Instant;
        for &n in &[1024usize, 1536, 2048] {
            let m = n;
            let a = lu_spd_like(n, 0x1234);
            let (lu, perm, _s) = super::lu_factor_nxn(&a, n).unwrap();
            let mut bmat = vec![0.0f64; n * m];
            for i in 0..n {
                bmat[i * m + i] = 1.0;
            }
            let it = if n <= 1024 { 5 } else { 3 };
            let med = |mut xs: Vec<f64>| {
                xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                xs[xs.len() / 2]
            };
            // Unblocked reference path is reached by calling the blocked function's
            // sibling with the same body — re-run lu_forward_back_multi at a forced
            // small m is wrong; instead time the production entry (blocked) vs an
            // inline unblocked solve.
            let unblocked = |lu: &[f64], perm: &[usize], b: &[f64]| -> Vec<f64> {
                let mut x = vec![0.0f64; n * m];
                for i in 0..n {
                    let p = perm[i];
                    x[i * m..i * m + m].copy_from_slice(&b[p * m..p * m + m]);
                }
                for i in 1..n {
                    let (head, tail) = x.split_at_mut(i * m);
                    let row_i = &mut tail[0..m];
                    for j in 0..i {
                        let lij = lu[i * n + j];
                        let row_j = &head[j * m..j * m + m];
                        for col in 0..m {
                            row_i[col] -= lij * row_j[col];
                        }
                    }
                }
                for i in (0..n).rev() {
                    let (head, tail) = x.split_at_mut((i + 1) * m);
                    let row_i = &mut head[i * m..i * m + m];
                    for j in (i + 1)..n {
                        let uij = lu[i * n + j];
                        let row_j = &tail[(j - i - 1) * m..(j - i - 1) * m + m];
                        for col in 0..m {
                            row_i[col] -= uij * row_j[col];
                        }
                    }
                    let uii = lu[i * n + i];
                    for cell in row_i.iter_mut().take(m) {
                        *cell /= uii;
                    }
                }
                x
            };
            let mut tu = Vec::new();
            let mut tb = Vec::new();
            for _ in 0..it {
                let t = Instant::now();
                let r = unblocked(&lu, &perm, &bmat);
                std::hint::black_box(&r);
                tu.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                let r = super::lu_forward_back_multi_blocked(&lu, &perm, &bmat, n, m);
                std::hint::black_box(&r);
                tb.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let (u, b) = (med(tu), med(tb));
            println!("n={n:5} unblocked={u:9.2}ms blocked={b:9.2}ms speedup={:.2}x", u / b);
        }
    }

    #[test]
    fn blocked_cholesky_reconstructs_and_matches_unblocked() {
        // Exercise every blocked path through the public entry `cholesky_nxn`, which
        // selects the kernel by n: 160/200/256 use the serial mid path (trail < the
        // parallel-dtrsm threshold); 512 uses the mid panel but its tall leading
        // panels cross the parallel-dtrsm threshold; 960/1024 use the large panel
        // (so the trailing update runs the block-triangular SYRK). Verify L·L^T = A
        // and that the blocked result agrees with the unblocked reference (tol —
        // Cholesky is not bit-reproducible, the parallel dtrsm uses a reciprocal-
        // multiply and the SYRK re-associates the trailing update).
        for &n in &[160usize, 200, 256, 512, 960, 1024] {
            let a = chol_spd(n, 0x71 + n as u64);
            let lb = super::cholesky_nxn(&a, n).expect("blocked chol");
            let lr = cholesky_unblocked_ref(&a, n);
            let mut max_recon = 0.0f64;
            let mut max_diff = 0.0f64;
            for i in 0..n {
                for j in 0..=i {
                    let mut s = 0.0;
                    for k in 0..=j {
                        s += lb[i * n + k] * lb[j * n + k];
                    }
                    max_recon = max_recon.max((s - a[i * n + j]).abs() / (1.0 + a[i * n + j].abs()));
                    max_diff =
                        max_diff.max((lb[i * n + j] - lr[i * n + j]).abs() / (1.0 + lr[i * n + j].abs()));
                }
            }
            assert!(max_recon < 1e-9, "blocked L·L^T=A err {max_recon:e} (n={n})");
            assert!(max_diff < 1e-9, "blocked vs unblocked err {max_diff:e} (n={n})");
        }
    }

    #[test]
    fn cholesky_mid_panel_256_output_golden_sha256() {
        let n = 256usize;
        let panel = super::cholesky_panel_width(n);
        assert_eq!(panel, 32);
        let a = chol_spd(n, 0x5A5A_5A5A);
        let l = super::cholesky_nxn(&a, n).expect("mid-panel cholesky");
        let mut max_recon = 0.0f64;
        for i in 0..n {
            for j in 0..=i {
                let mut s = 0.0f64;
                for k in 0..=j {
                    s += l[i * n + k] * l[j * n + k];
                }
                max_recon =
                    max_recon.max((s - a[i * n + j]).abs() / (1.0 + a[i * n + j].abs()));
            }
        }
        assert!(
            max_recon < 1e-9,
            "mid-panel reconstruction drifted: {max_recon:e}"
        );

        let mut hasher = Sha256::new();
        hasher.update(n.to_le_bytes());
        hasher.update(panel.to_le_bytes());
        for value in &l {
            hasher.update(value.to_bits().to_le_bytes());
        }
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "5677abe4016141dfb737c906dc28a8d667526c7e96c5161771033e568c9a0e4e",
            "mid-panel cholesky golden digest drifted: {digest}"
        );
    }

    #[test]
    fn cholesky_mid_panel_512_output_golden_sha256() {
        let n = 512usize;
        let panel = super::cholesky_panel_width(n);
        assert_eq!(panel, 64);
        let a = chol_spd(n, 0xA11C_E512);
        let l = super::cholesky_nxn(&a, n).expect("mid-panel cholesky");
        let mut max_recon = 0.0f64;
        for i in 0..n {
            for j in 0..=i {
                let mut s = 0.0f64;
                for k in 0..=j {
                    s += l[i * n + k] * l[j * n + k];
                }
                max_recon =
                    max_recon.max((s - a[i * n + j]).abs() / (1.0 + a[i * n + j].abs()));
            }
        }
        assert!(
            max_recon < 1e-9,
            "mid-panel reconstruction drifted: {max_recon:e}"
        );

        let mut hasher = Sha256::new();
        hasher.update(n.to_le_bytes());
        hasher.update(panel.to_le_bytes());
        for value in &l {
            hasher.update(value.to_bits().to_le_bytes());
        }
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "2d506e1089d1a9acd6eb6b4666847516b9948ab289fda72c48ed98bc05b9e617",
            "mid-panel 512 cholesky golden digest drifted: {digest}"
        );
    }

    #[test]
    #[ignore = "perf timing; run with --release -- --ignored --nocapture"]
    fn blocked_cholesky_speedup_report() {
        use std::time::Instant;
        for &n in &[1024usize, 1536, 2048] {
            let a = chol_spd(n, 0x1234);
            let it = if n <= 1024 { 5 } else { 3 };
            let med = |mut xs: Vec<f64>| {
                xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                xs[xs.len() / 2]
            };
            let mut tu = Vec::new();
            let mut tb = Vec::new();
            for _ in 0..it {
                let t = Instant::now();
                let r = cholesky_unblocked_ref(&a, n);
                std::hint::black_box(&r);
                tu.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                let r = super::cholesky_nxn(&a, n).unwrap();
                std::hint::black_box(&r);
                tb.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let (u, b) = (med(tu), med(tb));
            println!("n={n:5} unblocked={u:9.2}ms blocked={b:9.2}ms speedup={:.2}x", u / b);
        }
    }

    // Inline unblocked LU (pre-blocking reference) for the A/B and tolerance check.
    fn lu_unblocked_ref(a: &[f64], n: usize) -> (Vec<f64>, Vec<usize>) {
        let mut lu = a.to_vec();
        let mut perm: Vec<usize> = (0..n).collect();
        for k in 0..n {
            let mut max_val = lu[k * n + k].abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                let val = lu[i * n + k].abs();
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }
            if max_row != k {
                for j in 0..n {
                    lu.swap(k * n + j, max_row * n + j);
                }
                perm.swap(k, max_row);
            }
            let pivot = lu[k * n + k];
            for i in (k + 1)..n {
                let factor = lu[i * n + k] / pivot;
                lu[i * n + k] = factor;
                for j in (k + 1)..n {
                    let u_val = lu[k * n + j];
                    lu[i * n + j] -= factor * u_val;
                }
            }
        }
        (lu, perm)
    }

    fn lu_spd_like(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed | 1;
        let mut a: Vec<f64> = (0..n * n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect();
        for i in 0..n {
            a[i * n + i] += n as f64; // diagonally dominant -> finite, stable pivots
        }
        a
    }

    #[test]
    fn blocked_lu_reconstructs_pivoted_matrix() {
        // n >= LU_BLOCK_MIN routes through the blocked path. Verify P·A = L·U to
        // tolerance, and that blocked and unblocked agree to tolerance.
        for &n in &[160usize, 200, 256] {
            let a = lu_spd_like(n, 0x51 + n as u64);
            // Call the blocked kernel directly (these n are below LU_BLOCK_MIN, so
            // the public entry would otherwise take the unblocked path).
            let max_abs = a.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            let thr = (n as f64) * f64::EPSILON * max_abs;
            let (lu, perm, _sign) = super::lu_decompose_blocked(&a, n, thr).expect("blocked lu");
            let (_lu_ref, perm_ref) = lu_unblocked_ref(&a, n);
            assert_eq!(perm, perm_ref, "pivot sequence must match unblocked (n={n})");
            let mut max_err = 0.0f64;
            for i in 0..n {
                for j in 0..n {
                    let mut s = 0.0;
                    for k in 0..n {
                        let lik = if k < i {
                            lu[i * n + k]
                        } else if k == i {
                            1.0
                        } else {
                            0.0
                        };
                        let ukj = if k <= j { lu[k * n + j] } else { 0.0 };
                        s += lik * ukj;
                    }
                    let pa = a[perm[i] * n + j];
                    max_err = max_err.max((s - pa).abs() / (1.0 + pa.abs()));
                }
            }
            assert!(max_err < 1e-9, "blocked P·A=L·U reconstruction err {max_err:e} (n={n})");
        }
    }

    #[test]
    fn lu_panel_64_mid_size_keeps_pivots_logdet_and_golden_sha256() {
        let n = 512;
        let a = lu_spd_like(n, 0x6c75_7061_6e65_6c34);

        let (lu, perm, sign) = super::lu_factor_nxn(&a, n).expect("blocked lu factor");
        let (lu_ref, perm_ref) = lu_unblocked_ref(&a, n);
        assert_eq!(perm, perm_ref, "64-wide panel must preserve pivot order");

        let mut ref_sign = 1.0f64;
        let mut ref_log = 0.0f64;
        for i in 0..n {
            let d = lu_ref[i * n + i];
            if d < 0.0 {
                ref_sign = -ref_sign;
            }
            ref_log += d.abs().ln();
        }
        let mut log = 0.0f64;
        let mut det_sign = sign;
        for i in 0..n {
            let d = lu[i * n + i];
            if d < 0.0 {
                det_sign = -det_sign;
            }
            log += d.abs().ln();
        }
        assert_eq!(det_sign, ref_sign, "determinant sign drifted");
        let rel = (log - ref_log).abs() / ref_log.abs().max(1.0);
        assert!(
            rel < 1e-9,
            "64-wide panel log|det| {log} vs serial reference {ref_log} (rel {rel:e})"
        );

        let mut hasher = Sha256::new();
        hasher.update(n.to_le_bytes());
        hasher.update(sign.to_bits().to_le_bytes());
        for p in &perm {
            hasher.update(p.to_le_bytes());
        }
        for v in &lu {
            hasher.update(v.to_bits().to_le_bytes());
        }
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "0c94d84ca8e58ff7ce27babdac9d584397074a53770050be8151b9482ee586de",
            "64-wide LU panel golden digest drifted: {digest}"
        );
    }

    #[test]
    #[ignore = "perf timing; run with --release -- --ignored --nocapture"]
    fn blocked_lu_speedup_report() {
        use std::time::Instant;
        for &n in &[1024usize, 1536, 2048] {
            let a = lu_spd_like(n, 0x1234);
            let it = if n <= 512 { 5 } else { 3 };
            let med = |mut xs: Vec<f64>| {
                xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                xs[xs.len() / 2]
            };
            let mut tu = Vec::new();
            let mut tb = Vec::new();
            for _ in 0..it {
                let t = Instant::now();
                let r = lu_unblocked_ref(&a, n);
                std::hint::black_box(&r);
                tu.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                let r = super::lu_factor_nxn(&a, n).unwrap();
                std::hint::black_box(&r);
                tb.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let (u, b) = (med(tu), med(tb));
            println!("n={n:5} unblocked={u:9.2}ms blocked={b:9.2}ms speedup={:.2}x", u / b);
        }
    }

    #[test]
    fn lu_factor_and_lu_solve_roundtrip() {
        let a = [2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0];
        let b = [8.0, -11.0, -3.0];
        let (lu, perm, _sign) = lu_factor_nxn(&a, 3).expect("lu_factor");
        let x = lu_solve(&lu, &perm, &b, 3).expect("lu_solve");
        assert!(approx_equal(x[0], 2.0, 1e-10));
        assert!(approx_equal(x[1], 3.0, 1e-10));
        assert!(approx_equal(x[2], -1.0, 1e-10));
    }

    #[test]
    fn lu_factor_rejects_singular() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let err = lu_factor_nxn(&a, 3).expect_err("singular");
        assert_eq!(err.reason_code(), "linalg_solver_singularity");
    }

    #[test]
    fn solve_nxn_multi_matches_column_wise() {
        // A x1 = b1 and A x2 = b2
        let a = [2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0];
        let b1 = [8.0, -11.0, -3.0];
        let b2 = [1.0, 0.0, 0.0];
        // B matrix: columns are b1, b2 (row-major: B[i][j])
        let b_mat = [b1[0], b2[0], b1[1], b2[1], b1[2], b2[2]];
        let x_mat = solve_nxn_multi(&a, &b_mat, 3, 2).expect("multi solve");
        let x1_single = solve_nxn(&a, &b1, 3).expect("single solve 1");
        let x2_single = solve_nxn(&a, &b2, 3).expect("single solve 2");
        for i in 0..3 {
            assert!(
                approx_equal(x_mat[i * 2], x1_single[i], 1e-10),
                "col 0 row {i}: {} vs {}",
                x_mat[i * 2],
                x1_single[i]
            );
            assert!(
                approx_equal(x_mat[i * 2 + 1], x2_single[i], 1e-10),
                "col 1 row {i}: {} vs {}",
                x_mat[i * 2 + 1],
                x2_single[i]
            );
        }
    }

    #[test]
    fn solve_triangular_lower() {
        // L = [[2, 0, 0], [1, 3, 0], [4, 2, 5]]
        let l = [2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 4.0, 2.0, 5.0];
        let b = [4.0, 7.0, 30.0];
        let x = solve_triangular(&l, &b, 3, true, false).expect("lower tri solve");
        // Verify L*x = b
        for i in 0..3 {
            let mut row_sum = 0.0;
            for j in 0..3 {
                row_sum += l[i * 3 + j] * x[j];
            }
            assert!(
                approx_equal(row_sum, b[i], 1e-10),
                "row {i}: {row_sum} vs {}",
                b[i]
            );
        }
    }

    #[test]
    fn solve_triangular_upper() {
        // U = [[3, 1, 2], [0, 4, 1], [0, 0, 2]]
        let u = [3.0, 1.0, 2.0, 0.0, 4.0, 1.0, 0.0, 0.0, 2.0];
        let b = [10.0, 9.0, 4.0];
        let x = solve_triangular(&u, &b, 3, false, false).expect("upper tri solve");
        // Verify U*x = b
        for i in 0..3 {
            let mut row_sum = 0.0;
            for j in 0..3 {
                row_sum += u[i * 3 + j] * x[j];
            }
            assert!(
                approx_equal(row_sum, b[i], 1e-10),
                "row {i}: {row_sum} vs {}",
                b[i]
            );
        }
    }

    #[test]
    fn solve_triangular_unit_diagonal() {
        // L with unit diagonal: [[1, 0, 0], [2, 1, 0], [3, 4, 1]]
        let l = [1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 3.0, 4.0, 1.0];
        let b = [1.0, 4.0, 15.0];
        let x = solve_triangular(&l, &b, 3, true, true).expect("unit diag solve");
        // Verify L*x = b
        for i in 0..3 {
            let mut row_sum = 0.0;
            for j in 0..3 {
                row_sum += l[i * 3 + j] * x[j];
            }
            assert!(
                approx_equal(row_sum, b[i], 1e-10),
                "row {i}: {row_sum} vs {}",
                b[i]
            );
        }
    }

    #[test]
    fn solve_triangular_rejects_singular() {
        // Lower triangular with zero on diagonal
        let l = [1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0];
        let b = [1.0, 2.0, 3.0];
        let err = solve_triangular(&l, &b, 3, true, false).expect_err("singular tri");
        assert_eq!(err.reason_code(), "linalg_solver_singularity");
    }

    // ── Schur decomposition tests ──

    #[test]
    fn schur_diagonal_matrix() {
        // Schur form of a diagonal matrix is itself
        let a = [3.0, 0.0, 0.0, 5.0];
        let (t, z) = schur_nxn(&a, 2).unwrap();
        // T should have eigenvalues on diagonal
        let mut diag = [t[0], t[3]];
        diag.sort_by(|a, b| b.total_cmp(a));
        assert!((diag[0] - 5.0).abs() < 1e-6, "t00={}", diag[0]);
        assert!((diag[1] - 3.0).abs() < 1e-6, "t11={}", diag[1]);

        // Z should be orthogonal: Z * Z^T ≈ I
        let zt = mat_mul_flat(&z, &[z[0], z[2], z[1], z[3]], 2);
        assert!((zt[0] - 1.0).abs() < 1e-6);
        assert!(zt[1].abs() < 1e-6);
        assert!(zt[2].abs() < 1e-6);
        assert!((zt[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn schur_reconstructs_original() {
        // A = Z * T * Z^T
        let a = [1.0, 2.0, 3.0, 4.0];
        let (t, z) = schur_nxn(&a, 2).unwrap();
        // Compute Z * T
        let zt_product = mat_mul_flat(&z, &t, 2);
        // Compute (Z * T) * Z^T
        let z_t = [z[0], z[2], z[1], z[3]]; // transpose
        let reconstructed = mat_mul_flat(&zt_product, &z_t, 2);
        for i in 0..4 {
            assert!(
                (reconstructed[i] - a[i]).abs() < 1e-6,
                "reconstructed[{i}] = {}, expected {}",
                reconstructed[i],
                a[i]
            );
        }
    }

    #[test]
    fn schur_rejects_empty() {
        assert!(schur_nxn(&[], 0).is_err());
    }

    // ── Cross product tests ──

    #[test]
    fn cross_product_standard_basis() {
        // i × j = k
        let result = cross_product(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]).unwrap();
        assert!((result[0]).abs() < 1e-15);
        assert!((result[1]).abs() < 1e-15);
        assert!((result[2] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn cross_product_anticommutative() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let ab = cross_product(&a, &b).unwrap();
        let ba = cross_product(&b, &a).unwrap();
        for i in 0..3 {
            assert!((ab[i] + ba[i]).abs() < 1e-10, "not anticommutative at {i}");
        }
    }

    #[test]
    fn cross_product_self_is_zero() {
        let a = [3.0, -1.0, 4.0];
        let result = cross_product(&a, &a).unwrap();
        for val in &result[..3] {
            assert!(val.abs() < 1e-15);
        }
    }

    #[test]
    fn cross_product_2d() {
        // np.cross([1, 2], [3, 4]) = 1*4 - 2*3 = -2
        let r = cross_product(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        assert_eq!(r.len(), 1);
        assert!((r[0] - (-2.0)).abs() < 1e-12);
    }

    #[test]
    fn cross_product_2d_3d_mixed() {
        // np.cross([1, 2], [3, 4, 5]) = [2*5, -1*5, 1*4-2*3] = [10, -5, -2]
        let r = cross_product(&[1.0, 2.0], &[3.0, 4.0, 5.0]).unwrap();
        assert_eq!(r.len(), 3);
        assert!((r[0] - 10.0).abs() < 1e-12);
        assert!((r[1] - (-5.0)).abs() < 1e-12);
        assert!((r[2] - (-2.0)).abs() < 1e-12);
    }

    #[test]
    fn cross_product_rejects_wrong_size() {
        assert!(cross_product(&[1.0], &[3.0, 4.0]).is_err());
        assert!(cross_product(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]).is_err());
    }

    #[test]
    fn cond_p_frobenius() {
        // Frobenius condition number is ||A||_F * ||A^-1||_F.
        // For I2 this is sqrt(2) * sqrt(2) = 2, matching NumPy.
        let eye = [1.0, 0.0, 0.0, 1.0];
        let c = cond_p_nxn(&eye, 2, Some("fro")).unwrap();
        assert!((c - 2.0).abs() < 1e-10);
    }

    #[test]
    fn cond_p_one_norm() {
        let eye = [1.0, 0.0, 0.0, 1.0];
        let c = cond_p_nxn(&eye, 2, Some("1")).unwrap();
        assert!((c - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cond_p_inf_norm() {
        let eye = [1.0, 0.0, 0.0, 1.0];
        let c = cond_p_nxn(&eye, 2, Some("inf")).unwrap();
        assert!((c - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cond_p_default_matches_cond() {
        let a = [4.0, 7.0, 2.0, 6.0];
        let c1 = cond_nxn(&a, 2).unwrap();
        let c2 = cond_p_nxn(&a, 2, None).unwrap();
        assert!((c1 - c2).abs() < 1e-10);
    }

    #[test]
    fn cond_p_spectral_symmetric_uses_absolute_eigenvalues() {
        let a = [2.0, 0.0, 0.0, -4.0];
        let c = cond_p_nxn(&a, 2, Some("2")).unwrap();
        assert!((c - 2.0).abs() < 1e-12, "cond_2 symmetric diag={c}");

        let reciprocal = cond_p_nxn(&a, 2, Some("-2")).unwrap();
        assert!(
            (reciprocal - 0.5).abs() < 1e-12,
            "cond_-2 symmetric diag={reciprocal}"
        );
    }

    #[test]
    fn cond_p_spectral_symmetric_matches_svd_reference() {
        let a = [
            5.0, 0.25, -0.5, 0.75, 0.25, 4.0, 0.125, -0.25, -0.5, 0.125, 3.0, 0.5,
            0.75, -0.25, 0.5, 2.0,
        ];
        let fast = cond_nxn(&a, 4).unwrap();
        let sigmas = svd_nxn(&a, 4).unwrap();
        let expected = sigmas[0] / sigmas[sigmas.len() - 1];
        assert!(
            (fast - expected).abs() <= 1e-8 * expected.abs().max(1.0),
            "symmetric cond fast path {fast} diverged from SVD reference {expected}"
        );
    }

    #[test]
    fn cond_p_non_spectral_nan_orders_match_numpy() {
        let a = [f64::NAN, 1.0, 2.0, 3.0];
        for ord in ["fro", "1", "-1", "inf", "-inf"] {
            assert!(
                cond_p_nxn(&a, 2, Some(ord))
                    .expect("nan cond should propagate")
                    .is_nan(),
                "order {ord} should propagate NaN",
            );
        }

        cond_p_nxn(&a, 2, None).expect_err("default cond should remain spectral");
        cond_p_nxn(&a, 2, Some("2")).expect_err("2-norm cond should remain spectral");
        cond_p_nxn(&a, 2, Some("-2")).expect_err("-2 cond should remain spectral");
    }

    #[test]
    fn cond_p_singular_non_spectral_orders_are_infinite() {
        // NumPy evaluates cond(A, p) = norm(A, p) * norm(inv(A), p) under
        // errstate(all="ignore"); for a singular (but finite) matrix the
        // result is +inf, not a raised error. Verified against NumPy 2.4.3:
        // np.linalg.cond([[1,2],[2,4]], p) == inf for p in {1,-1,inf,-inf,fro}.
        let singular = [1.0, 2.0, 2.0, 4.0];
        for ord in ["fro", "1", "-1", "inf", "-inf"] {
            let c = cond_p_nxn(&singular, 2, Some(ord))
                .unwrap_or_else(|_| panic!("singular cond order {ord} should not error"));
            assert!(
                c.is_infinite() && c > 0.0,
                "order {ord} on a singular matrix should be +inf, got {c}",
            );
        }
    }

    #[test]
    fn cond_p_spectral_infinite_orders_match_numpy() {
        let a = [f64::INFINITY, 1.0, 2.0, 3.0];
        assert!(
            cond_p_nxn(&a, 2, None)
                .expect("default cond on inf")
                .is_infinite()
        );
        assert!(
            cond_p_nxn(&a, 2, Some("2"))
                .expect("2-norm cond on inf")
                .is_infinite()
        );
        assert!(
            cond_p_nxn(&a, 2, Some("-2"))
                .expect("-2 cond on inf")
                .is_infinite()
        );
    }

    // ── Kronecker product tests ──

    #[test]
    fn kron_parallel_matches_serial_reference_and_golden() {
        use sha2::{Digest, Sha256};
        // The parallel row-fill kron must be BIT-IDENTICAL to the serial nested
        // loop (each cell is a single product, no accumulation), across square and
        // rectangular factors and output sizes crossing the parallel threshold
        // (1<<18). Reference is an independent serial implementation; values include
        // NaN / ±inf / -0.0 to lock the exact product semantics.
        fn serial(a: &[f64], m: usize, n: usize, b: &[f64], p: usize, q: usize) -> Vec<f64> {
            let out_cols = n * q;
            let mut out = vec![0.0f64; m * p * out_cols];
            for i in 0..m {
                for j in 0..n {
                    let av = a[i * n + j];
                    for k in 0..p {
                        for l in 0..q {
                            out[(i * p + k) * out_cols + (j * q + l)] = av * b[k * q + l];
                        }
                    }
                }
            }
            out
        }
        // (m, n, p, q) — square, tall, wide; sizes both below and above the gate.
        let cases: &[(usize, usize, usize, usize)] = &[
            (40, 33, 13, 17),
            (8, 8, 64, 64),
            (100, 1, 1, 100),
            (3, 3, 3, 3),
        ];
        let mut seed = 0x9b1c_4e2f_7a05_d3c8u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            match (seed >> 9) % 19 {
                0 => f64::NAN,
                1 => f64::INFINITY,
                2 => f64::NEG_INFINITY,
                3 => -0.0,
                _ => (seed >> 11) as f64 / (1u64 << 53) as f64 * 20.0 - 10.0,
            }
        };
        let mut digest = Sha256::new();
        for &(m, n, p, q) in cases {
            let a: Vec<f64> = (0..m * n).map(|_| next()).collect();
            let b: Vec<f64> = (0..p * q).map(|_| next()).collect();
            let got = kron_nxn(&a, m, n, &b, p, q).unwrap();
            let want = serial(&a, m, n, &b, p, q);
            assert_eq!(got.len(), want.len(), "len {m}x{n} kron {p}x{q}");
            for (g, w) in got.iter().zip(want.iter()) {
                assert_eq!(g.to_bits(), w.to_bits(), "kron {m}x{n} ⊗ {p}x{q}");
            }
            for v in &got {
                digest.update(v.to_bits().to_le_bytes());
            }
        }
        let hex: String = digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert_eq!(
            hex, "6d16bc3a90ee6673f1f7f200b5955b17bfbe5402416926516eff16920d77efc7",
            "kron parallel golden digest drifted"
        );
    }

    #[test]
    fn kron_identity_rhs_fast_path_matches_dense_reference_and_fallbacks() {
        fn serial(a: &[f64], m: usize, n: usize, b: &[f64], p: usize, q: usize) -> Vec<f64> {
            let out_cols = n * q;
            let mut out = vec![0.0f64; m * p * out_cols];
            for i in 0..m {
                for j in 0..n {
                    let av = a[i * n + j];
                    for k in 0..p {
                        for l in 0..q {
                            out[(i * p + k) * out_cols + (j * q + l)] = av * b[k * q + l];
                        }
                    }
                }
            }
            out
        }

        let a = [0.0, 1.25, 2.5, 3.75, 5.0, 6.25];
        let eye3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let fast = super::kron_identity_rhs_nonnegative_fast_path(&a, 2, 3, &eye3, 3, 3)
            .expect("nonnegative exact identity RHS should specialize");
        let got = super::kron_nxn(&a, 2, 3, &eye3, 3, 3).expect("kron");
        let want = serial(&a, 2, 3, &eye3, 3, 3);
        assert_eq!(fast.len(), want.len());
        assert_eq!(got.len(), want.len());
        for (idx, ((f, g), w)) in fast.iter().zip(&got).zip(&want).enumerate() {
            assert_eq!(f.to_bits(), w.to_bits(), "fast path bit drift at {idx}");
            assert_eq!(g.to_bits(), w.to_bits(), "public kron bit drift at {idx}");
        }

        type FallbackCase<'a> = (&'a [f64], &'a [f64], usize, usize, usize, usize);
        let fallback_cases: &[FallbackCase<'_>] = &[
            (&[-1.0, 2.0], &eye3, 1, 2, 3, 3),
            (&[-0.0, 2.0], &eye3, 1, 2, 3, 3),
            (&[f64::NAN, 2.0], &eye3, 1, 2, 3, 3),
            (&[f64::INFINITY, 2.0], &eye3, 1, 2, 3, 3),
            (&[1.0, 2.0], &[1.0, -0.0, 0.0, 1.0], 1, 2, 2, 2),
        ];
        for &(case_a, case_b, m, n, p, q) in fallback_cases {
            assert!(
                super::kron_identity_rhs_nonnegative_fast_path(case_a, m, n, case_b, p, q)
                    .is_none(),
                "fallback-sensitive case should not specialize"
            );
            let got = super::kron_nxn(case_a, m, n, case_b, p, q).expect("fallback kron");
            let want = serial(case_a, m, n, case_b, p, q);
            for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "fallback bit drift at {idx}");
            }
        }
    }

    #[test]
    fn kron_identity_identity() {
        // I2 ⊗ I2 = I4
        let i2 = [1.0, 0.0, 0.0, 1.0];
        let result = kron_nxn(&i2, 2, 2, &i2, 2, 2).unwrap();
        assert_eq!(result.len(), 16);
        let i4 = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        for i in 0..16 {
            assert!((result[i] - i4[i]).abs() < 1e-15, "i4[{i}] mismatch");
        }
    }

    #[test]
    fn kron_scalar() {
        // [3] ⊗ [1, 2; 3, 4] = [3, 6; 9, 12]
        let a = [3.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        let result = kron_nxn(&a, 1, 1, &b, 2, 2).unwrap();
        assert_eq!(result, vec![3.0, 6.0, 9.0, 12.0]);
    }

    // ── multi_dot tests ──

    #[test]
    fn multi_dot_two_matrices() {
        // Simple 2x2 * 2x2
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let (result, rows, cols) = multi_dot(&[(&a, 2, 2), (&b, 2, 2)]).unwrap();
        assert_eq!(rows, 2);
        assert_eq!(cols, 2);
        // Expected: [[19, 22], [43, 50]]
        assert!((result[0] - 19.0).abs() < 1e-10);
        assert!((result[1] - 22.0).abs() < 1e-10);
        assert!((result[2] - 43.0).abs() < 1e-10);
        assert!((result[3] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn multi_dot_three_matrices() {
        // (2x3) * (3x2) * (2x1) - should use optimal parenthesization
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3x2
        let c = [1.0, 1.0]; // 2x1
        let (result, rows, cols) = multi_dot(&[(&a, 2, 3), (&b, 3, 2), (&c, 2, 1)]).unwrap();
        assert_eq!(rows, 2);
        assert_eq!(cols, 1);

        // Verify by doing it step by step
        let ab = mat_mul_rect(&a, &b, 2, 3, 2);
        let expected = mat_mul_rect(&ab, &c, 2, 2, 1);
        for i in 0..2 {
            assert!(
                (result[i] - expected[i]).abs() < 1e-10,
                "multi_dot[{i}]={}, expected={}",
                result[i],
                expected[i]
            );
        }
    }

    #[test]
    fn multi_dot_single_matrix() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let (result, rows, cols) = multi_dot(&[(&a, 2, 2)]).unwrap();
        assert_eq!(rows, 2);
        assert_eq!(cols, 2);
        assert_eq!(result, a.to_vec());
    }

    #[test]
    fn multi_dot_dimension_mismatch() {
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = [1.0, 2.0, 3.0]; // 1x3
        assert!(multi_dot(&[(&a, 2, 2), (&b, 1, 3)]).is_err());
    }

    // ── Rectangular QR tests ──

    #[test]
    fn qr_mxn_tall_matrix_reconstructs() {
        // 3x2 matrix
        #[rustfmt::skip]
        let a = [
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ];
        let (q, r) = qr_mxn(&a, 3, 2).unwrap();
        // Q is 3x3, R is 3x2
        assert_eq!(q.len(), 9);
        assert_eq!(r.len(), 6);

        // Verify Q * R ≈ A
        for i in 0..3 {
            for j in 0..2 {
                let mut sum: f64 = 0.0;
                for k in 0..3 {
                    sum += q[i * 3 + k] * r[k * 2 + j];
                }
                assert!(
                    (sum - a[i * 2 + j]).abs() < 1e-10,
                    "QR reconstruction mismatch at ({i},{j}): {sum} vs {}",
                    a[i * 2 + j]
                );
            }
        }

        // Verify Q is orthogonal: Q^T * Q ≈ I
        for i in 0..3 {
            for j in 0..3 {
                let mut dot: f64 = 0.0;
                for k in 0..3 {
                    dot += q[k * 3 + i] * q[k * 3 + j];
                }
                let expected: f64 = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "Q orthogonality failed at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn qr_mxn_wide_matrix_reconstructs() {
        // 2x3 matrix
        #[rustfmt::skip]
        let a = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        let (q, r) = qr_mxn(&a, 2, 3).unwrap();
        // Q is 2x2, R is 2x3
        assert_eq!(q.len(), 4);
        assert_eq!(r.len(), 6);

        // Verify Q * R ≈ A
        for i in 0..2 {
            for j in 0..3 {
                let mut sum: f64 = 0.0;
                for k in 0..2 {
                    sum += q[i * 2 + k] * r[k * 3 + j];
                }
                assert!(
                    (sum - a[i * 3 + j]).abs() < 1e-10,
                    "QR reconstruction mismatch at ({i},{j})"
                );
            }
        }
    }

    // ── Rectangular SVD tests ──

    #[test]
    fn svd_mxn_tall_matrix_singular_values() {
        // 3x2 matrix
        #[rustfmt::skip]
        let a = [
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ];
        let sigmas = svd_mxn(&a, 3, 2).unwrap();
        assert_eq!(sigmas.len(), 2);
        // Singular values should be positive and in descending order
        assert!(sigmas[0] >= sigmas[1]);
        assert!(sigmas[1] >= 0.0);
        // Known approximate values for this matrix
        assert!((sigmas[0] - 9.525).abs() < 0.1, "sigma[0]={}", sigmas[0]);
    }

    #[test]
    fn svd_mxn_wide_matrix_singular_values() {
        // 2x3 matrix
        #[rustfmt::skip]
        let a = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        let sigmas = svd_mxn(&a, 2, 3).unwrap();
        assert_eq!(sigmas.len(), 2);
        assert!(sigmas[0] >= sigmas[1]);
    }

    #[test]
    fn svd_mxn_full_reconstructs() {
        // 3x2 matrix
        #[rustfmt::skip]
        let a = [
            1.0, 0.0,
            0.0, 2.0,
            0.0, 0.0,
        ];
        let (u, s, vt) = svd_mxn_full(&a, 3, 2).unwrap();
        assert_eq!(u.len(), 9); // 3x3
        assert_eq!(s.len(), 2); // min(3,2) = 2
        assert_eq!(vt.len(), 4); // 2x2

        // Singular values should be 2 and 1 (descending)
        assert!((s[0] - 2.0).abs() < 1e-8, "s[0]={}", s[0]);
        assert!((s[1] - 1.0).abs() < 1e-8, "s[1]={}", s[1]);

        // Reconstruct: A ≈ U * diag(S) * Vt (using only first min(m,n) columns of U)
        let k = 2;
        for i in 0..3 {
            for j in 0..2 {
                let mut sum: f64 = 0.0;
                for l in 0..k {
                    sum += u[i * 3 + l] * s[l] * vt[l * 2 + j];
                }
                assert!(
                    (sum - a[i * 2 + j]).abs() < 1e-8,
                    "SVD reconstruction mismatch at ({i},{j}): {sum} vs {}",
                    a[i * 2 + j]
                );
            }
        }
    }

    #[test]
    fn svd_mxn_full_identity() {
        let a = [1.0, 0.0, 0.0, 1.0];
        let (u, s, vt) = svd_mxn_full(&a, 2, 2).unwrap();
        assert_eq!(s.len(), 2);
        assert!((s[0] - 1.0).abs() < 1e-10);
        assert!((s[1] - 1.0).abs() < 1e-10);
        // U and Vt should be orthogonal
        for i in 0..2 {
            for j in 0..2 {
                let mut dot_u: f64 = 0.0;
                let mut dot_v: f64 = 0.0;
                for k in 0..2 {
                    dot_u += u[k * 2 + i] * u[k * 2 + j];
                    dot_v += vt[i * 2 + k] * vt[j * 2 + k];
                }
                let expected: f64 = if i == j { 1.0 } else { 0.0 };
                assert!((dot_u - expected).abs() < 1e-10);
                assert!((dot_v - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn svd_mxn_rejects_invalid() {
        assert!(svd_mxn(&[], 0, 0).is_err());
        assert!(svd_mxn(&[1.0, f64::NAN], 1, 2).is_err());
        assert!(qr_mxn(&[], 0, 0).is_err());
    }

    #[test]
    fn svd_values_only_matches_full_reference_bits() {
        let n = 32usize;
        let mut state = 456u64;
        let a: Vec<f64> = (0..(n * n))
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
            })
            .collect();

        let values_only = svd_mxn(&a, n, n).expect("values-only svd");
        let (_, full_reference, _) = super::svd_bidiag_full(&a, n, n).expect("full reference svd");
        assert_eq!(
            values_only
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            full_reference
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "values-only SVD must preserve singular-value bits"
        );

        let mut hasher = Sha256::new();
        for value in &values_only {
            hasher.update(value.to_bits().to_le_bytes());
        }
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "4c7c7aa6cdf4721a02c213c87c18eec2f977fef96cb9ca52b6b79a873f413f7a",
            "values-only SVD singular-value bit pattern must remain fixed"
        );
    }

    #[test]
    fn values_only_svd_in_place_sort_matches_former_index_schedule() {
        fn former_index_schedule(values: &[f64]) -> Vec<u64> {
            let mut order: Vec<usize> = (0..values.len()).collect();
            order.sort_by(|&a, &b| {
                values[b]
                    .partial_cmp(&values[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            order
                .into_iter()
                .map(|index| values[index].to_bits())
                .collect()
        }

        let mut values = vec![
            5.0,
            -0.0,
            3.0,
            f64::INFINITY,
            3.0,
            0.0,
            1.5,
            1.5,
            f64::MIN_POSITIVE,
        ];
        let expected_bits = former_index_schedule(&values);
        super::sort_singular_values_descending_in_place(&mut values);
        let actual_bits = values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();

        assert_eq!(
            actual_bits, expected_bits,
            "in-place values-only SVD sort must preserve the former index-order bits"
        );
    }

    #[test]
    fn svd_full_output_matches_clean_head_digest() {
        let n = 24usize;
        let mut state = 0xCAFE_BABE_D15E_A5E5u64;
        let a: Vec<f64> = (0..(n * n))
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
            })
            .collect();

        let (u, s, vt) = svd_mxn_full(&a, n, n).expect("full svd");
        let mut hasher = Sha256::new();
        for part in [&u[..], &s[..], &vt[..]] {
            hasher.update(part.len().to_le_bytes());
            for value in part {
                hasher.update(value.to_bits().to_le_bytes());
            }
        }
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "4a06f40391aae9cc82d3cb1a711624a8b08e85878d44e9279ae77758fc1c4c9f",
            "full SVD output bit pattern must remain fixed: {digest}"
        );
    }

    #[test]
    fn svd_full_tall_output_matches_clean_head_digest() {
        let m = 32usize;
        let n = 24usize;
        let mut state = 0x51D1_5EED_C0FF_EE11u64;
        let a: Vec<f64> = (0..(m * n))
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
            })
            .collect();

        let (u, s, vt) = svd_mxn_full(&a, m, n).expect("tall full svd");
        let mut hasher = Sha256::new();
        for part in [&u[..], &s[..], &vt[..]] {
            hasher.update(part.len().to_le_bytes());
            for value in part {
                hasher.update(value.to_bits().to_le_bytes());
            }
        }
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "498a91ab1c9f10d9572305acad1516997cc3e29dffbc21fe83c6b04585e870e4",
            "tall full SVD output bit pattern must remain fixed: {digest}"
        );
    }

    #[test]
    fn svd_full_reconstructed_u_fast_path_reconstructs_and_hashes() {
        let n = 32usize;
        let mut state = 0xA17E_F00D_5EED_1234u64;
        let a: Vec<f64> = (0..(n * n))
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
            })
            .collect();

        let (u, s, vt) = svd_mxn_full(&a, n, n).expect("reconstructed-U full svd");
        for pair in s.windows(2) {
            assert!(pair[0] >= pair[1], "singular values must remain descending");
        }

        let mut max_recon = 0.0f64;
        let mut max_u_orth = 0.0f64;
        let mut max_v_orth = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let mut reconstructed = 0.0;
                let mut u_dot = 0.0;
                let mut v_dot = 0.0;
                for k in 0..n {
                    reconstructed += u[i * n + k] * s[k] * vt[k * n + j];
                    u_dot += u[k * n + i] * u[k * n + j];
                    v_dot += vt[i * n + k] * vt[j * n + k];
                }
                max_recon = max_recon.max((reconstructed - a[i * n + j]).abs());
                let target = if i == j { 1.0 } else { 0.0 };
                max_u_orth = max_u_orth.max((u_dot - target).abs());
                max_v_orth = max_v_orth.max((v_dot - target).abs());
            }
        }
        assert!(max_recon < 1e-8, "SVD reconstruction drift {max_recon:e}");
        assert!(max_u_orth < 1e-8, "U orthogonality drift {max_u_orth:e}");
        assert!(max_v_orth < 1e-8, "Vt orthogonality drift {max_v_orth:e}");

        let mut hasher = Sha256::new();
        for part in [&u[..], &s[..], &vt[..]] {
            hasher.update(part.len().to_le_bytes());
            for value in part {
                hasher.update(value.to_bits().to_le_bytes());
            }
        }
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "ed1ac139dac59fe907b7dd9d2a1dccd2ecbbbede840868afe2c9e9e1d41ed793",
            "reconstructed-U full SVD digest must remain fixed: {digest}"
        );
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn svd_full_512_right_vt_panel_reconstructs_and_hashes() {
        let n = 512usize;
        let mut state = 0x5A1D_B10D_5120_0001u64;
        let a: Vec<f64> = (0..(n * n))
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
            })
            .collect();

        let (u, s, vt) = svd_mxn_full(&a, n, n).expect("512 full svd");
        for pair in s.windows(2) {
            assert!(pair[0] >= pair[1], "singular values must remain descending");
        }

        let mut us = u.clone();
        for row in 0..n {
            for col in 0..n {
                us[row * n + col] *= s[col];
            }
        }
        let recon = super::packed_gemm(&us, &vt, n, n, n);
        let mut max_recon = 0.0f64;
        for (actual, expected) in recon.iter().zip(&a) {
            max_recon = max_recon.max((actual - expected).abs());
        }
        assert!(
            max_recon < 1e-8,
            "512 full SVD reconstruction drift {max_recon:e}"
        );

        let mut hasher = Sha256::new();
        for part in [&u[..], &s[..], &vt[..]] {
            hasher.update(part.len().to_le_bytes());
            for value in part {
                hasher.update(value.to_bits().to_le_bytes());
            }
        }
        let digest = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "9e9810981185bb50c0626e61c076ec7bd9ebbde9f6404838a450df81813b6139",
            "512 full SVD right-Vt panel digest drifted: {digest}"
        );
    }

    // ── Rectangular QR tests ─────────────────────────────────────────

    #[test]
    fn qr_mxn_tall_3x2() {
        #[rustfmt::skip]
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (q, r) = qr_mxn(&a, 3, 2).unwrap();
        assert_eq!(q.len(), 9);
        assert_eq!(r.len(), 6);
        let recon = mat_mul_rect(&q, &r, 3, 3, 2);
        for i in 0..6 {
            assert!((recon[i] - a[i]).abs() < 1e-10, "QR recon at {i}");
        }
    }

    #[test]
    fn qr_mxn_wide_2x3() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (q, r) = qr_mxn(&a, 2, 3).unwrap();
        assert_eq!(q.len(), 4);
        assert_eq!(r.len(), 6);
        let recon = mat_mul_rect(&q, &r, 2, 2, 3);
        for i in 0..6 {
            assert!((recon[i] - a[i]).abs() < 1e-10, "QR recon at {i}");
        }
    }

    #[test]
    fn qr_mxn_q_orthogonal() {
        let a = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let (q, _) = qr_mxn(&a, 3, 2).unwrap();
        let mut qtq = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += q[k * 3 + i] * q[k * 3 + j];
                }
                qtq[i * 3 + j] = s;
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                let exp = if i == j { 1.0 } else { 0.0 };
                assert!((qtq[i * 3 + j] - exp).abs() < 1e-10);
            }
        }
    }

    // ── Rectangular SVD tests ────────────────────────────────────────

    #[test]
    fn svd_mxn_tall_singular_values() {
        let a = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let s = svd_mxn(&a, 3, 2).unwrap();
        assert_eq!(s.len(), 2);
        assert!((s[0] - 1.0).abs() < 1e-8);
        assert!((s[1] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn svd_mxn_wide_singular_values() {
        let a = [3.0, 0.0, 0.0, 0.0, 2.0, 0.0];
        let s = svd_mxn(&a, 2, 3).unwrap();
        assert_eq!(s.len(), 2);
        assert!((s[0] - 3.0).abs() < 1e-8);
        assert!((s[1] - 2.0).abs() < 1e-8);
    }

    #[test]
    fn svd_mxn_full_reconstruction() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (u, s, vt) = svd_mxn_full(&a, 3, 2).unwrap();
        assert_eq!(u.len(), 9);
        assert_eq!(s.len(), 2);
        assert_eq!(vt.len(), 4);
        let mut recon = [0.0_f64; 6];
        for i in 0..3 {
            for j in 0..2 {
                let mut sum = 0.0;
                for kk in 0..2 {
                    sum += u[i * 3 + kk] * s[kk] * vt[kk * 2 + j];
                }
                recon[i * 2 + j] = sum;
            }
        }
        for i in 0..6 {
            assert!((recon[i] - a[i]).abs() < 1e-8, "SVD recon at {i}");
        }
    }

    #[test]
    fn svd_fallback_reconstructs_when_qr_budget_is_zero() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (u, s, vt) = svd_bidiag_full_with_max_iters(&a, 3, 2, 0).expect("fallback svd");
        assert_eq!(u.len(), 9);
        assert_eq!(s.len(), 2);
        assert_eq!(vt.len(), 4);

        let mut recon = [0.0; 6];
        for i in 0..3 {
            for j in 0..2 {
                let mut sum = 0.0;
                for kk in 0..2 {
                    sum += u[i * 3 + kk] * s[kk] * vt[kk * 2 + j];
                }
                recon[i * 2 + j] = sum;
            }
        }
        for i in 0..6 {
            assert!((recon[i] - a[i]).abs() < 1e-8, "fallback recon at {i}");
        }
    }

    #[test]
    fn svd_fallback_handles_wide_matrix_when_qr_budget_is_zero() {
        let a = [3.0, 0.0, 0.0, 0.0, 2.0, 0.0];
        let (_u, s, vt) = svd_bidiag_full_with_max_iters(&a, 2, 3, 0).expect("wide fallback svd");
        assert_eq!(s.len(), 2);
        assert_eq!(vt.len(), 9);
        assert!((s[0] - 3.0).abs() < 1e-8, "sigma_0={}", s[0]);
        assert!((s[1] - 2.0).abs() < 1e-8, "sigma_1={}", s[1]);
    }

    #[test]
    fn svd_high_condition_number_preserves_small_singular_values() {
        // 4x3 matrix with cond(A) = 10^12, singular values [1e6, 1.0, 1e-6]
        // Constructed from random orthogonal matrices (np.random.seed(42))
        #[rustfmt::skip]
        let a = [
             379645.7743350327,  -225645.24435488618,  479553.25114605244,
            -178966.65563490757,  106370.25818389589, -226063.73815110396,
            -358826.2181453029,   213270.82822770457, -453254.4589148696,
             184936.09609271045, -109917.40263701396,  233602.18131610617,
        ];
        let (u, s, vt) = svd_mxn_full(&a, 4, 3).unwrap();

        // Singular values must be in descending order
        assert!(s[0] >= s[1] && s[1] >= s[2]);

        // The old A^T*A approach would square the condition number to 10^24,
        // losing small singular values entirely (gave σ₂≈5e-4 instead of 1e-6).
        // Golub-Reinsch QR on the bidiagonal preserves them to ~machine precision.
        assert!(
            (s[0] - 1e6).abs() / 1e6 < 1e-8,
            "sigma_0 = {} (expected 1e6)",
            s[0]
        );
        assert!(
            (s[1] - 1.0).abs() < 1e-8,
            "sigma_1 = {} (expected 1.0)",
            s[1]
        );
        assert!(
            (s[2] - 1e-6).abs() / 1e-6 < 1e-2,
            "sigma_2 = {} (expected 1e-6)",
            s[2]
        );

        // Reconstruction: ||A - U*S*Vt|| / ||A|| < eps * cond
        let m = 4;
        let n = 3;
        let k = 3;
        let mut recon = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += u[i * m + l] * s[l] * vt[l * n + j];
                }
                recon[i * n + j] = sum;
            }
        }
        let a_norm: f64 = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let err: f64 = a
            .iter()
            .zip(recon.iter())
            .map(|(&ai, &ri)| (ai - ri) * (ai - ri))
            .sum::<f64>()
            .sqrt();
        let rel_err = err / a_norm;
        assert!(
            rel_err < 1e-6,
            "Relative reconstruction error = {rel_err:.2e} (must be < 1e-6)"
        );

        // U orthogonality: ||U^T U - I|| < eps
        for i in 0..m {
            for j in 0..m {
                let mut dot = 0.0;
                for r in 0..m {
                    dot += u[r * m + i] * u[r * m + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-6,
                    "U^T*U[{i},{j}] = {dot}, expected {expected}"
                );
            }
        }

        // Vt orthogonality: ||Vt Vt^T - I|| < eps
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for c in 0..n {
                    dot += vt[i * n + c] * vt[j * n + c];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-6,
                    "Vt*Vt^T[{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn svd_fallback_preserves_high_condition_number_singular_values() {
        #[rustfmt::skip]
        let a = [
             379645.7743350327,  -225645.24435488618,  479553.25114605244,
            -178966.65563490757,  106370.25818389589, -226063.73815110396,
            -358826.2181453029,   213270.82822770457, -453254.4589148696,
             184936.09609271045, -109917.40263701396,  233602.18131610617,
        ];
        let (_u, s, _vt) =
            svd_bidiag_full_with_max_iters(&a, 4, 3, 0).expect("high-condition fallback svd");
        assert!(
            (s[0] - 1e6).abs() / 1e6 < 1e-8,
            "fallback sigma_0 = {} (expected 1e6)",
            s[0]
        );
        assert!(
            (s[1] - 1.0).abs() < 1e-8,
            "fallback sigma_1 = {} (expected 1.0)",
            s[1]
        );
        assert!(
            (s[2] - 1e-6).abs() / 1e-6 < 1e-2,
            "fallback sigma_2 = {} (expected 1e-6)",
            s[2]
        );
    }

    // ── expm tests ──

    #[test]
    fn expm_identity_is_e_times_identity() {
        // expm(I) = e*I
        let eye = [1.0, 0.0, 0.0, 1.0];
        let result = expm_nxn(&eye, 2).unwrap();
        let e = std::f64::consts::E;
        assert!((result[0] - e).abs() < 1e-6, "expm(I)[0,0] = e");
        assert!(result[1].abs() < 1e-10, "expm(I)[0,1] = 0");
        assert!(result[2].abs() < 1e-10, "expm(I)[1,0] = 0");
        assert!((result[3] - e).abs() < 1e-6, "expm(I)[1,1] = e");
    }

    #[test]
    fn expm_zero_is_identity() {
        let zero = [0.0; 4];
        let result = expm_nxn(&zero, 2).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!(result[1].abs() < 1e-10);
        assert!(result[2].abs() < 1e-10);
        assert!((result[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn expm_diagonal_matrix() {
        // expm([[a,0],[0,b]]) = [[e^a,0],[0,e^b]]
        let a = [2.0, 0.0, 0.0, 3.0];
        let result = expm_nxn(&a, 2).unwrap();
        assert!((result[0] - 2.0f64.exp()).abs() < 1e-4, "expm diag [0,0]");
        assert!(result[1].abs() < 1e-6, "expm diag [0,1]");
        assert!(result[2].abs() < 1e-6, "expm diag [1,0]");
        assert!((result[3] - 3.0f64.exp()).abs() < 1e-4, "expm diag [1,1]");
    }

    #[test]
    fn expm_matches_scipy_non_finite_semantics() {
        let diag_inf = [f64::INFINITY, 0.0, 0.0, 3.0];
        let result = expm_nxn(&diag_inf, 2).expect("diag inf expm");
        assert!(result[0].is_infinite() && result[0].is_sign_positive());
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 0.0);
        assert!((result[3] - 3.0f64.exp()).abs() < 1e-9);

        let diag_nan = [f64::NAN, 0.0, 0.0, 3.0];
        let result = expm_nxn(&diag_nan, 2).expect("diag nan expm");
        assert!(result[0].is_nan());
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 0.0);
        assert!((result[3] - 3.0f64.exp()).abs() < 1e-9);

        let upper_inf = [1.0, f64::INFINITY, 0.0, 3.0];
        let result = expm_nxn(&upper_inf, 2).expect("upper inf expm");
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 0.0);
        assert!(result[3].is_nan());

        let lower_nan = [1.0, 0.0, f64::NAN, 3.0];
        let result = expm_nxn(&lower_nan, 2).expect("lower nan expm");
        assert!(result[0].is_nan());
        assert_eq!(result[1], 0.0);
        assert!(result[2].is_nan());
        assert!(result[3].is_nan());

        let general_inf = [f64::INFINITY, 1.0, 2.0, 3.0];
        let result = expm_nxn(&general_inf, 2).expect("general inf expm");
        assert!(result.iter().all(|value| value.is_nan()));
    }

    // ── sqrtm tests ──

    #[test]
    fn sqrtm_identity_is_identity() {
        let eye = [1.0, 0.0, 0.0, 1.0];
        let result = sqrtm_nxn(&eye, 2).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!(result[1].abs() < 1e-10);
        assert!(result[2].abs() < 1e-10);
        assert!((result[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn sqrtm_squared_gives_original() {
        // sqrtm(A)^2 ≈ A
        let a = [4.0, 1.0, 2.0, 3.0];
        let s = sqrtm_nxn(&a, 2).unwrap();
        let s2 = mat_mul_flat(&s, &s, 2);
        for i in 0..4 {
            assert!(
                (s2[i] - a[i]).abs() < 1e-8,
                "sqrtm squared at {i}: {} vs {}",
                s2[i],
                a[i]
            );
        }
    }

    #[test]
    fn sqrtm_diagonal() {
        let a = [9.0, 0.0, 0.0, 16.0];
        let result = sqrtm_nxn(&a, 2).unwrap();
        assert!((result[0] - 3.0).abs() < 1e-8);
        assert!(result[1].abs() < 1e-10);
        assert!(result[2].abs() < 1e-10);
        assert!((result[3] - 4.0).abs() < 1e-8);
    }

    #[test]
    fn sqrtm_matches_scipy_non_finite_semantics() {
        let diag_inf = [f64::INFINITY, 0.0, 0.0, 3.0];
        let result = sqrtm_nxn(&diag_inf, 2).expect("diag inf sqrtm");
        assert!(result[0].is_infinite() && result[0].is_sign_positive());
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 0.0);
        assert!((result[3] - 3.0f64.sqrt()).abs() < 1e-9);

        let diag_nan = [f64::NAN, 0.0, 0.0, 3.0];
        let result = sqrtm_nxn(&diag_nan, 2).expect("diag nan sqrtm");
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 0.0);
        assert!((result[3] - 3.0f64.sqrt()).abs() < 1e-9);

        let upper_inf = [1.0, f64::INFINITY, 0.0, 3.0];
        let result = sqrtm_nxn(&upper_inf, 2).expect("upper inf sqrtm");
        assert!((result[0] - 1.0).abs() < 1e-12);
        assert!(result[1].is_infinite() && result[1].is_sign_positive());
        assert_eq!(result[2], 0.0);
        assert!((result[3] - 3.0f64.sqrt()).abs() < 1e-9);

        let upper_diag_nan = [1.0, 0.0, 0.0, f64::NAN];
        let result = sqrtm_nxn(&upper_diag_nan, 2).expect("upper diag nan sqrtm");
        assert!((result[0] - 1.0).abs() < 1e-12);
        assert!(result[1].is_nan());
        assert_eq!(result[2], 0.0);
        assert!(result[3].is_nan());

        let lower_inf = [1.0, 0.0, f64::INFINITY, 3.0];
        let result = sqrtm_nxn(&lower_inf, 2).expect("lower inf sqrtm");
        assert!(result.iter().all(|value| value.is_nan()));

        let general_nan = [f64::NAN, 1.0, 2.0, 3.0];
        let result = sqrtm_nxn(&general_nan, 2).expect("general nan sqrtm");
        assert!(result.iter().all(|value| value.is_nan()));
    }

    // ── logm tests ──

    #[test]
    fn logm_identity_is_zero() {
        let eye = [1.0, 0.0, 0.0, 1.0];
        let result = logm_nxn(&eye, 2).unwrap();
        for (i, &v) in result.iter().enumerate().take(4) {
            assert!(v.abs() < 1e-10, "logm(I) should be zero at {i}");
        }
    }

    #[test]
    fn logm_expm_roundtrip() {
        // logm(expm(A)) ≈ A for small A
        let a = [0.1, 0.05, -0.03, 0.2];
        let ea = expm_nxn(&a, 2).unwrap();
        let la = logm_nxn(&ea, 2).unwrap();
        for i in 0..4 {
            assert!(
                (la[i] - a[i]).abs() < 1e-4,
                "roundtrip at {i}: {} vs {}",
                la[i],
                a[i]
            );
        }
    }

    #[test]
    fn logm_diagonal() {
        let e = std::f64::consts::E;
        let a = [e, 0.0, 0.0, e * e];
        let result = logm_nxn(&a, 2).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-4, "logm diag [0,0]");
        assert!(result[1].abs() < 1e-6, "logm diag [0,1]");
        assert!(result[2].abs() < 1e-6, "logm diag [1,0]");
        assert!((result[3] - 2.0).abs() < 1e-4, "logm diag [1,1]");
    }

    // ── Polar decomposition tests ──

    #[test]
    fn polar_identity() {
        let a = [1.0, 0.0, 0.0, 1.0];
        let (u, p) = polar_nxn(&a, 2).unwrap();
        // Identity: U = I, P = I
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (u[i * 2 + j] - expected).abs() < 1e-10,
                    "polar U identity [{i},{j}]"
                );
                assert!(
                    (p[i * 2 + j] - expected).abs() < 1e-10,
                    "polar P identity [{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn polar_reconstruction() {
        // A = U * P => U * P should reconstruct A
        let a = [2.0, 1.0, 0.5, 3.0];
        let (u, p) = polar_nxn(&a, 2).unwrap();
        let reconstructed = mat_mul_flat(&u, &p, 2);
        for i in 0..4 {
            assert!(
                (reconstructed[i] - a[i]).abs() < 1e-10,
                "polar reconstruction [{i}]"
            );
        }
    }

    #[test]
    fn polar_u_is_orthogonal() {
        let a = [2.0, 1.0, 0.5, 3.0];
        let (u, _p) = polar_nxn(&a, 2).unwrap();
        // U * U^T should be identity
        let mut ut = vec![0.0; 4];
        for i in 0..2 {
            for j in 0..2 {
                ut[i * 2 + j] = u[j * 2 + i];
            }
        }
        let uut = mat_mul_flat(&u, &ut, 2);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (uut[i * 2 + j] - expected).abs() < 1e-10,
                    "U orthogonality [{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn polar_p_is_symmetric() {
        let a = [2.0, 1.0, 0.5, 3.0];
        let (_u, p) = polar_nxn(&a, 2).unwrap();
        // P should be symmetric
        assert!((p[1] - p[2]).abs() < 1e-10, "P symmetry");
    }

    #[test]
    fn polar_p_positive_semidefinite() {
        let a = [2.0, 1.0, 0.5, 3.0];
        let (_u, p) = polar_nxn(&a, 2).unwrap();
        // Eigenvalues of P should be non-negative
        // For 2x2: eigenvalues = (trace +/- sqrt(trace^2 - 4*det)) / 2
        let tr = p[0] + p[3];
        let det = p[0] * p[3] - p[1] * p[2];
        let disc = tr * tr - 4.0 * det;
        assert!(disc >= -1e-10, "P discriminant non-negative");
        let sqrt_disc = disc.max(0.0).sqrt();
        let eig1 = (tr + sqrt_disc) / 2.0;
        let eig2 = (tr - sqrt_disc) / 2.0;
        assert!(eig1 >= -1e-10, "P eigenvalue 1 non-negative");
        assert!(eig2 >= -1e-10, "P eigenvalue 2 non-negative");
    }

    #[test]
    fn polar_3x3() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]; // non-singular
        let (u, p) = polar_nxn(&a, 3).unwrap();
        let reconstructed = mat_mul_flat(&u, &p, 3);
        for i in 0..9 {
            assert!(
                (reconstructed[i] - a[i]).abs() < 1e-8,
                "polar 3x3 reconstruction [{i}]"
            );
        }
    }

    // ── Cholesky solve tests ──

    #[test]
    fn cholesky_solve_identity_system() {
        // A = I, L = I, b = [1,2,3] → x = [1,2,3]
        let l = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b = [1.0, 2.0, 3.0];
        let x = cholesky_solve(&l, &b, 3).unwrap();
        for (i, (&xi, &bi)) in x.iter().zip(b.iter()).enumerate() {
            assert!((xi - bi).abs() < 1e-12, "identity solve at {i}");
        }
    }

    #[test]
    fn cholesky_solve_2x2_pd_system() {
        // A = [[4, 2], [2, 3]], L = [[2, 0], [1, sqrt(2)]]
        // b = [1, 2]
        let a = [4.0, 2.0, 2.0, 3.0];
        let l = cholesky_nxn(&a, 2).unwrap();
        let b = [1.0, 2.0];
        let x = cholesky_solve(&l, &b, 2).unwrap();
        // Verify A*x = b
        let r0 = a[0] * x[0] + a[1] * x[1];
        let r1 = a[2] * x[0] + a[3] * x[1];
        assert!((r0 - b[0]).abs() < 1e-10, "Ax=b check row 0");
        assert!((r1 - b[1]).abs() < 1e-10, "Ax=b check row 1");
    }

    #[test]
    fn cholesky_solve_3x3_pd_system() {
        // 3×3 positive definite matrix
        let a = [4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0];
        let l = cholesky_nxn(&a, 3).unwrap();
        let b = [1.0, -1.0, 2.0];
        let x = cholesky_solve(&l, &b, 3).unwrap();
        // Verify A*x = b
        for i in 0..3 {
            let mut sum = 0.0;
            for j in 0..3 {
                sum += a[i * 3 + j] * x[j];
            }
            assert!((sum - b[i]).abs() < 1e-10, "Ax=b check row {i}");
        }
    }

    #[test]
    fn cholesky_solve_rejects_singular() {
        let l = [1.0, 0.0, 0.0, 0.0]; // zero diagonal
        let b = [1.0, 2.0];
        let err = cholesky_solve(&l, &b, 2).expect_err("singular should fail");
        assert!(matches!(err, LinAlgError::SolverSingularity));
    }

    #[test]
    fn cholesky_solve_multi_roundtrip() {
        let a = [4.0, 2.0, 2.0, 3.0];
        let l = cholesky_nxn(&a, 2).unwrap();
        // B = [[1, 2], [3, 4]] (2×2, two RHS columns)
        let b = [1.0, 2.0, 3.0, 4.0];
        let x = cholesky_solve_multi(&l, &b, 2, 2).unwrap();
        // Verify A*X = B for each column
        for col in 0..2 {
            for row in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += a[row * 2 + k] * x[k * 2 + col];
                }
                let expected = b[row * 2 + col];
                assert!(
                    (sum - expected).abs() < 1e-10,
                    "A*X=B check row={row} col={col}"
                );
            }
        }
    }

    // ── tensorsolve tests ──

    #[test]
    fn tensorsolve_2d_is_standard_solve() {
        // When a is 2D (shape [2,2]) and b is 1D (shape [2]),
        // tensorsolve reduces to standard linear solve.
        let a = [2.0, 1.0, 1.0, 3.0];
        let b = [5.0, 7.0];
        let (x, x_shape) = tensorsolve(&a, &[2, 2], &b, &[2]).unwrap();
        assert_eq!(x_shape, vec![2]);
        // Verify A*x = b
        let r0 = a[0] * x[0] + a[1] * x[1];
        let r1 = a[2] * x[0] + a[3] * x[1];
        assert!((r0 - b[0]).abs() < 1e-10);
        assert!((r1 - b[1]).abs() < 1e-10);
    }

    #[test]
    fn tensorsolve_rejects_non_square() {
        // a_shape = [2, 3], b_shape = [2] → x_shape = [3], prod(b)=2 != prod(x)=3
        let a = [1.0; 6];
        let b = [1.0, 2.0];
        let err = tensorsolve(&a, &[2, 3], &b, &[2]).expect_err("non-square system");
        assert!(matches!(err, LinAlgError::ShapeContractViolation(_)));
    }

    #[test]
    fn tensorsolve_matches_numpy_non_finite_semantics() {
        let inf_matrix = [f64::INFINITY, 1.0, 0.0, 0.0, 3.0, 1.0, 2.0, 0.0, 3.0];
        let (solved, solved_shape) =
            tensorsolve(&inf_matrix, &[3, 3], &[1.0, 2.0, 3.0], &[3]).expect("inf tensorsolve");
        assert_eq!(solved_shape, vec![3]);
        assert!(approx_equal(solved[0], 0.0, 1e-12));
        assert!(approx_equal(solved[1], 0.3333333333333333, 1e-12));
        assert!(approx_equal(solved[2], 1.0, 1e-12));

        let nan_matrix = [f64::NAN, 1.0, 0.0, 0.0, 3.0, 1.0, 2.0, 0.0, 3.0];
        let (solved, solved_shape) =
            tensorsolve(&nan_matrix, &[3, 3], &[1.0, 2.0, 3.0], &[3]).expect("nan tensorsolve");
        assert_eq!(solved_shape, vec![3]);
        assert!(solved[0].is_nan());
        assert!(approx_equal(solved[1], 0.45454545454545453, 1e-12));
        assert!(approx_equal(solved[2], 0.6363636363636364, 1e-12));
    }

    #[test]
    fn tensorsolve_preserves_tensor_output_shape() {
        let mut a = vec![0.0; 64];
        for i in 0..8 {
            a[i * 8 + i] = 1.0;
        }
        let b = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let (x, x_shape) = tensorsolve(&a, &[2, 2, 2, 2, 2, 2], &b, &[2, 2, 2]).unwrap();
        assert_eq!(x_shape, vec![2, 2, 2]);
        assert_eq!(x, b);
    }

    // ── tensorinv tests ──

    #[test]
    fn tensorinv_4d_identity() {
        // 2×2 identity reshaped as (2,2) with ind=1 → n=2, m=2
        let a = [1.0, 0.0, 0.0, 1.0];
        let (inv, result_shape) = tensorinv(&a, &[2, 2], 1).unwrap();
        assert_eq!(result_shape, vec![2, 2]);
        assert!((inv[0] - 1.0).abs() < 1e-10);
        assert!(inv[1].abs() < 1e-10);
        assert!(inv[2].abs() < 1e-10);
        assert!((inv[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn tensorinv_rejects_invalid_ind() {
        let a = [1.0; 4];
        // ind=0 is invalid
        let err = tensorinv(&a, &[2, 2], 0).expect_err("ind=0 should fail");
        assert!(matches!(err, LinAlgError::ShapeContractViolation(_)));
        // ind=2 is >= ndim
        let err = tensorinv(&a, &[2, 2], 2).expect_err("ind=ndim should fail");
        assert!(matches!(err, LinAlgError::ShapeContractViolation(_)));
    }

    #[test]
    fn tensorinv_non_square_rejected() {
        // shape [2, 3] with ind=1: n=2, m=3, not square
        let a = [1.0; 6];
        let err = tensorinv(&a, &[2, 3], 1).expect_err("non-square should fail");
        assert!(matches!(err, LinAlgError::ShapeContractViolation(_)));
    }

    #[test]
    fn tensorinv_matches_numpy_non_finite_semantics() {
        let inf_matrix = [f64::INFINITY, 1.0, 0.0, 0.0, 3.0, 1.0, 2.0, 0.0, 3.0];
        let (inverse, shape) = tensorinv(&inf_matrix, &[3, 3], 1).expect("inf tensorinv");
        assert_eq!(shape, vec![3, 3]);
        assert!(approx_equal(inverse[0], 0.0, 1e-12));
        assert!(approx_equal(inverse[1], 0.0, 1e-12));
        assert!(approx_equal(inverse[2], 0.0, 1e-12));
        assert!(approx_equal(inverse[3], 0.0, 1e-12));
        assert!(approx_equal(inverse[4], 0.3333333333333333, 1e-12));
        assert!(approx_equal(inverse[5], -0.1111111111111111, 1e-12));
        assert!(approx_equal(inverse[6], 0.0, 1e-12));
        assert!(approx_equal(inverse[7], 0.0, 1e-12));
        assert!(approx_equal(inverse[8], 0.3333333333333333, 1e-12));

        let nan_matrix = [f64::NAN, 1.0, 0.0, 0.0, 3.0, 1.0, 2.0, 0.0, 3.0];
        let (inverse, shape) = tensorinv(&nan_matrix, &[3, 3], 1).expect("nan tensorinv");
        assert_eq!(shape, vec![3, 3]);
        assert!(inverse[0].is_nan());
        assert!(inverse[1].is_nan());
        assert!(inverse[2].is_nan());
        assert!(approx_equal(inverse[3], 0.18181818181818182, 1e-12));
        assert!(approx_equal(inverse[4], 0.2727272727272727, 1e-12));
        assert!(approx_equal(inverse[5], -0.09090909090909091, 1e-12));
        assert!(approx_equal(inverse[6], -0.5454545454545454, 1e-12));
        assert!(approx_equal(inverse[7], 0.18181818181818182, 1e-12));
        assert!(approx_equal(inverse[8], 0.2727272727272727, 1e-12));
    }

    #[test]
    fn funm_identity_with_square() {
        // f(x) = x^2 applied to identity should give identity
        let eye = vec![1.0, 0.0, 0.0, 1.0];
        let result = funm_nxn(&eye, 2, |x| x * x).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((result[i * 2 + j] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn funm_diagonal_applies_function() {
        // Diagonal matrix [[2, 0], [0, 3]]: f(x) = x^2 should give [[4, 0], [0, 9]]
        let a = vec![2.0, 0.0, 0.0, 3.0];
        let result = funm_nxn(&a, 2, |x| x * x).unwrap();
        assert!((result[0] - 4.0).abs() < 1e-8);
        assert!(result[1].abs() < 1e-8);
        assert!(result[2].abs() < 1e-8);
        assert!((result[3] - 9.0).abs() < 1e-8);
    }

    #[test]
    fn funm_exp_matches_expm() {
        // funm with exp should approximate expm for a symmetric matrix
        let a = vec![1.0, 0.5, 0.5, 1.0];
        let via_funm = funm_nxn(&a, 2, f64::exp).unwrap();
        let via_expm = expm_nxn(&a, 2).unwrap();
        for i in 0..4 {
            assert!(
                (via_funm[i] - via_expm[i]).abs() < 1e-6,
                "funm[{i}]={} vs expm[{i}]={}",
                via_funm[i],
                via_expm[i],
            );
        }
    }

    // -----------------------------------------------------------------------
    // Batched linalg tests
    // -----------------------------------------------------------------------

    #[test]
    fn batch_inv_3d_stack() {
        // 3 stacked 2x2 identity-like matrices
        let data = vec![
            1.0, 0.0, 0.0, 1.0, // I
            2.0, 0.0, 0.0, 0.5, // diag(2, 0.5)
            1.0, 1.0, 0.0, 1.0, // upper triangular
        ];
        let shape = [3, 2, 2];
        let inv = batch_inv(&data, &shape).expect("batch_inv");
        assert_eq!(inv.len(), 12); // 3 * 2 * 2

        // First: inv(I) = I
        assert!((inv[0] - 1.0).abs() < 1e-12);
        assert!((inv[3] - 1.0).abs() < 1e-12);

        // Second: inv(diag(2, 0.5)) = diag(0.5, 2)
        assert!((inv[4] - 0.5).abs() < 1e-12);
        assert!((inv[7] - 2.0).abs() < 1e-12);

        // Third: inv([[1,1],[0,1]]) = [[1,-1],[0,1]]
        assert!((inv[8] - 1.0).abs() < 1e-12);
        assert!((inv[9] - (-1.0)).abs() < 1e-12);
        assert!((inv[10]).abs() < 1e-12);
        assert!((inv[11] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn batch_det_returns_scalar_per_matrix() {
        // 4 stacked 3x3 matrices (4D shape: 2x2x3x3)
        let n = 3;
        let mut data = Vec::new();
        let expected_dets = [1.0, -1.0, 8.0, 0.0];
        // Identity
        data.extend_from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        // Permutation (det = -1)
        data.extend_from_slice(&[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        // 2*I (det = 8)
        data.extend_from_slice(&[2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]);
        // Singular (det = 0)
        data.extend_from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let shape = [2, 2, n, n]; // 4D batch
        let dets = batch_det(&data, &shape).expect("batch_det");
        assert_eq!(dets.len(), 4);
        for (i, (&got, &expected)) in dets.iter().zip(expected_dets.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-10,
                "det[{i}]: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn batch_det_slogdet_scratch_paths_match_per_lane_reference_bits() {
        fn assert_same_bits_or_nan(got: f64, want: f64, context: &str) {
            if want.is_nan() {
                assert!(got.is_nan(), "{context}: got {got}, expected NaN");
            } else {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{context}: got {got:?}, expected {want:?}"
                );
            }
        }

        for &(batch, n) in &[(5usize, 3usize), (1024, 4)] {
            let mat_size = n * n;
            let mut data = Vec::with_capacity(batch * mat_size);
            for b in 0..batch {
                let mut matrix = vec![0.0f64; mat_size];
                for i in 0..n {
                    for j in 0..n {
                        matrix[i * n + j] = if i == j {
                            (n + 3) as f64 + (b % 5) as f64 * 0.25
                        } else {
                            ((b + i * 7 + j * 11) % 17) as f64 * 0.03125
                        };
                    }
                }
                if b % 29 == 0 && n > 1 {
                    let row0 = matrix[0..n].to_vec();
                    matrix[n..2 * n].copy_from_slice(&row0);
                }
                if b % 31 == 0 {
                    matrix[0] = f64::NAN;
                }
                if b % 37 == 0 && mat_size > 1 {
                    matrix[1] = f64::INFINITY;
                }
                if b % 41 == 0 && mat_size > 2 {
                    matrix[2] = -0.0;
                }
                data.extend_from_slice(&matrix);
            }

            let shape = [batch, n, n];
            let dets = batch_det(&data, &shape).expect("batch det");
            let (signs, logabsdets) = batch_slogdet(&data, &shape).expect("batch slogdet");
            assert_eq!(dets.len(), batch);
            assert_eq!(signs.len(), batch);
            assert_eq!(logabsdets.len(), batch);

            for lane in 0..batch {
                let matrix = &data[lane * mat_size..(lane + 1) * mat_size];
                let want_det = det_nxn(matrix, n).expect("reference det");
                let (want_sign, want_logabsdet) =
                    slogdet_nxn(matrix, n).expect("reference slogdet");
                assert_same_bits_or_nan(dets[lane], want_det, "det scratch path drift");
                assert_same_bits_or_nan(signs[lane], want_sign, "slogdet sign drift");
                assert_same_bits_or_nan(
                    logabsdets[lane],
                    want_logabsdet,
                    "slogdet logabsdet drift",
                );
            }
        }
    }

    #[test]
    fn batch_solve_broadcasts_single_a() {
        // Single 2x2 system, batch of 3 rhs vectors
        let a = vec![2.0, 0.0, 0.0, 3.0]; // diag(2, 3)
        let b = vec![4.0, 9.0, 2.0, 6.0, 6.0, 3.0]; // 3 vectors
        let a_shape = [1, 2, 2]; // batch=1
        let b_shape = [3, 2]; // batch=3
        let x = batch_solve(&a, &a_shape, &b, &b_shape, true).expect("batch_solve");
        assert_eq!(x.len(), 6);
        // x[0] = [2, 3], x[1] = [1, 2], x[2] = [3, 1]
        assert!((x[0] - 2.0).abs() < 1e-12);
        assert!((x[1] - 3.0).abs() < 1e-12);
        assert!((x[2] - 1.0).abs() < 1e-12);
        assert!((x[3] - 2.0).abs() < 1e-12);
        assert!((x[4] - 3.0).abs() < 1e-12);
        assert!((x[5] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn batch_solve_accepts_matrix_rhs() {
        let a = vec![2.0, 0.0, 0.0, 3.0]; // diag(2, 3)
        let b = vec![
            4.0, 2.0, //
            9.0, 6.0,
        ];
        let a_shape = [2, 2];
        let b_shape = [2, 2];

        let x = batch_solve(&a, &a_shape, &b, &b_shape, false).expect("matrix rhs batch_solve");

        assert_eq!(x, vec![2.0, 1.0, 3.0, 2.0]);
    }

    #[test]
    fn batch_eigvalsh_stacked() {
        // 2 stacked 3x3 symmetric matrices
        let a1 = [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0]; // eigs: 5, 3, 2
        let a2 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // eigs: 1, 1, 1
        let mut data = Vec::new();
        data.extend_from_slice(&a1);
        data.extend_from_slice(&a2);
        let shape = [2, 3, 3];
        let eigvals = batch_eigvalsh(&data, &shape).expect("batch_eigvalsh");
        assert_eq!(eigvals.len(), 6); // 2 × 3
        // First matrix: sorted ascending [2, 3, 5] (NumPy convention)
        assert!((eigvals[0] - 2.0).abs() < 1e-10);
        assert!((eigvals[1] - 3.0).abs() < 1e-10);
        assert!((eigvals[2] - 5.0).abs() < 1e-10);
        // Second matrix: [1, 1, 1]
        for (i, eigval) in eigvals.iter().enumerate().skip(3).take(3) {
            assert!((*eigval - 1.0).abs() < 1e-10, "eig[{i}]={eigval}");
        }
    }

    #[test]
    fn batch_svd_stacked_rectangular() {
        // 2 stacked 3x2 matrices
        let a1 = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // sigmas: 1, 1
        let a2 = [3.0, 0.0, 0.0, 4.0, 0.0, 0.0]; // sigmas: 4, 3
        let mut data = Vec::new();
        data.extend_from_slice(&a1);
        data.extend_from_slice(&a2);
        let shape = [2, 3, 2];
        let sigmas = batch_svd(&data, &shape).expect("batch_svd");
        assert_eq!(sigmas.len(), 4); // 2 × min(3,2)
        // First: [1, 1] (sorted descending)
        assert!((sigmas[0] - 1.0).abs() < 1e-10);
        assert!((sigmas[1] - 1.0).abs() < 1e-10);
        // Second: [4, 3]
        assert!((sigmas[2] - 4.0).abs() < 1e-10);
        assert!((sigmas[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn batch_qr_stacked() {
        // 2 stacked 3x2 matrices
        let a1 = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let a2 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let mut data = Vec::new();
        data.extend_from_slice(&a1);
        data.extend_from_slice(&a2);
        let shape = [2, 3, 2];
        let (all_q, all_r) = batch_qr(&data, &shape).expect("batch_qr");
        let m = 3;
        let n = 2;
        // Q is m×m, R is m×n
        assert_eq!(all_q.len(), 2 * m * m);
        assert_eq!(all_r.len(), 2 * m * n);

        // Verify reconstruction: Q*R ≈ A for each
        for b in 0..2 {
            let q = &all_q[b * m * m..(b + 1) * m * m];
            let r = &all_r[b * m * n..(b + 1) * m * n];
            let a_sub = &data[b * m * n..(b + 1) * m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut val = 0.0;
                    for l in 0..m {
                        val += q[i * m + l] * r[l * n + j];
                    }
                    assert!(
                        (val - a_sub[i * n + j]).abs() < 1e-10,
                        "batch {b}, [{i},{j}]: {val} vs {}",
                        a_sub[i * n + j]
                    );
                }
            }
        }
    }

    #[test]
    fn batch_cholesky_stacked() {
        // 2 stacked 2x2 positive definite matrices
        let a1 = [4.0, 2.0, 2.0, 5.0];
        let a2 = [9.0, 0.0, 0.0, 4.0];
        let mut data = Vec::new();
        data.extend_from_slice(&a1);
        data.extend_from_slice(&a2);
        let shape = [2, 2, 2];
        let chol = batch_cholesky(&data, &shape).expect("batch_cholesky");
        assert_eq!(chol.len(), 8); // 2 × 2 × 2
        // Verify L*L^T ≈ A for each
        for b in 0..2 {
            let l = &chol[b * 4..(b + 1) * 4];
            let a_sub = &data[b * 4..(b + 1) * 4];
            for i in 0..2 {
                for j in 0..2 {
                    let mut val = 0.0;
                    for k in 0..2 {
                        val += l[i * 2 + k] * l[j * 2 + k];
                    }
                    assert!(
                        (val - a_sub[i * 2 + j]).abs() < 1e-10,
                        "batch {b}, [{i},{j}]: {val} vs {}",
                        a_sub[i * 2 + j]
                    );
                }
            }
        }
    }

    #[test]
    fn batch_trace_and_slogdet() {
        let data = vec![
            1.0, 0.0, 0.0, 2.0, // diag(1, 2), trace=3, det=2
            3.0, 0.0, 0.0, 4.0, // diag(3, 4), trace=7, det=12
        ];
        let shape = [2, 2, 2];
        let traces = batch_trace(&data, &shape).expect("batch_trace");
        assert_eq!(traces.len(), 2);
        assert!((traces[0] - 3.0).abs() < 1e-12);
        assert!((traces[1] - 7.0).abs() < 1e-12);

        let (signs, logabsdets) = batch_slogdet(&data, &shape).expect("batch_slogdet");
        assert_eq!(signs.len(), 2);
        assert!((signs[0] - 1.0).abs() < 1e-12);
        assert!((signs[1] - 1.0).abs() < 1e-12);
        assert!((logabsdets[0] - 2.0_f64.ln()).abs() < 1e-12);
        assert!((logabsdets[1] - 12.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn batch_eig_complex_pairs() {
        // Stack a rotation matrix (complex eigs) and a diagonal (real eigs)
        let theta = std::f64::consts::FRAC_PI_3; // 60 degrees
        let c = theta.cos();
        let s = theta.sin();
        let mut data = Vec::new();
        data.extend_from_slice(&[c, -s, s, c]); // rotation
        data.extend_from_slice(&[5.0, 0.0, 0.0, 7.0]); // diagonal
        let shape = [2, 2, 2];
        let eigs = batch_eig(&data, &shape).expect("batch_eig");
        assert_eq!(eigs.len(), 8); // 2 matrices × 2 eigenvalues × 2 (re, im)

        // First matrix: eigenvalues cos(θ) ± i*sin(θ)
        assert!((eigs[0] - c).abs() < 1e-10, "re0={}", eigs[0]);
        assert!((eigs[1].abs() - s).abs() < 1e-10, "im0={}", eigs[1]);
        assert!((eigs[2] - c).abs() < 1e-10, "re1={}", eigs[2]);
        assert!((eigs[1] + eigs[3]).abs() < 1e-10, "conjugate");

        // Second matrix: eigenvalues 5, 7 (sorted may vary)
        let re0 = eigs[4];
        let re1 = eigs[6];
        assert!(eigs[5].abs() < 1e-10, "should be real");
        assert!(eigs[7].abs() < 1e-10, "should be real");
        let mut real_eigs = [re0, re1];
        real_eigs.sort_by(|a, b| b.total_cmp(a));
        assert!((real_eigs[0] - 7.0).abs() < 1e-10);
        assert!((real_eigs[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn batch_rejects_1d_input() {
        let data = vec![1.0, 2.0, 3.0];
        let shape = [3]; // 1D - should fail
        assert!(batch_inv(&data, &shape).is_err());
        assert!(batch_det(&data, &shape).is_err());
    }

    // ────────────────────────────────────────────────────────────────
    // Complex linalg tests
    // ────────────────────────────────────────────────────────────────

    #[test]
    fn complex_det_identity() {
        // 2×2 complex identity: [[1+0i, 0+0i], [0+0i, 1+0i]]
        let a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let (dr, di) = complex_det_nxn(&a, 2).expect("det");
        assert!((dr - 1.0).abs() < 1e-12);
        assert!(di.abs() < 1e-12);
    }

    #[test]
    fn complex_det_diagonal() {
        // diag(2+3i, 4-1i) → det = (2+3i)(4-1i) = 11+10i
        let a = [2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, -1.0];
        let (dr, di) = complex_det_nxn(&a, 2).expect("det");
        assert!((dr - 11.0).abs() < 1e-12, "re={dr}");
        assert!((di - 10.0).abs() < 1e-12, "im={di}");
    }

    #[test]
    fn complex_solve_identity() {
        // I·x = [1+2i, 3+4i] → x = [1+2i, 3+4i]
        let a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        let x = complex_solve_nxn(&a, &b, 2).expect("solve");
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 2.0).abs() < 1e-12);
        assert!((x[2] - 3.0).abs() < 1e-12);
        assert!((x[3] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn complex_solve_2x2() {
        // A = [[1+i, 2], [0, 1-i]], b = [3+i, 1]
        // (1-i)x₂ = 1 → x₂ = (1)/(1-i) = (1+i)/2
        // (1+i)x₁ + 2·x₂ = 3+i → (1+i)x₁ = 3+i - (1+i) = 2 → x₁ = 2/(1+i) = 1-i
        let a = [1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0, -1.0];
        let b = [3.0, 1.0, 1.0, 0.0];
        let x = complex_solve_nxn(&a, &b, 2).expect("solve");
        assert!((x[0] - 1.0).abs() < 1e-12, "x1_re={}", x[0]);
        assert!((x[1] - (-1.0)).abs() < 1e-12, "x1_im={}", x[1]);
        assert!((x[2] - 0.5).abs() < 1e-12, "x2_re={}", x[2]);
        assert!((x[3] - 0.5).abs() < 1e-12, "x2_im={}", x[3]);
    }

    #[test]
    fn complex_inv_2x2() {
        // A = [[1+i, 0], [0, 2-i]], det = (1+i)(2-i) = 3+i
        // inv = [[2-i, 0], [0, 1+i]] / (3+i) = [[2-i, 0], [0, 1+i]] · (3-i)/10
        let a = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, -1.0];
        let inv = complex_inv_nxn(&a, 2).expect("inv");

        // A·A⁻¹ should be identity
        let prod = complex_matmul(&a, &inv, 2, 2, 2);
        assert!((prod[0] - 1.0).abs() < 1e-12); // (0,0) re
        assert!(prod[1].abs() < 1e-12); // (0,0) im
        assert!(prod[2].abs() < 1e-12); // (0,1) re
        assert!(prod[3].abs() < 1e-12); // (0,1) im
        assert!(prod[4].abs() < 1e-12); // (1,0) re
        assert!(prod[5].abs() < 1e-12); // (1,0) im
        assert!((prod[6] - 1.0).abs() < 1e-12); // (1,1) re
        assert!(prod[7].abs() < 1e-12); // (1,1) im
    }

    #[test]
    fn complex_inv_general_3x3() {
        // A general complex 3×3 matrix
        let a = [
            1.0, 0.0, 2.0, 1.0, 0.0, -1.0, 0.0, 1.0, 3.0, 0.0, 1.0, 2.0, 2.0, -1.0, 0.0, 0.0, 4.0,
            1.0,
        ];
        let inv = complex_inv_nxn(&a, 3).expect("inv");
        let prod = complex_matmul(&a, &inv, 3, 3, 3);

        // Check identity
        for i in 0..3 {
            for j in 0..3 {
                let re = prod[2 * (i * 3 + j)];
                let im = prod[2 * (i * 3 + j) + 1];
                let exp_re = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (re - exp_re).abs() < 1e-10 && im.abs() < 1e-10,
                    "prod[{i},{j}] = {re}+{im}i, expected {exp_re}"
                );
            }
        }
    }

    #[test]
    fn complex_cholesky_hermitian() {
        // Hermitian PD: A = [[4, 2+i], [2-i, 3]]
        let a = [4.0, 0.0, 2.0, 1.0, 2.0, -1.0, 3.0, 0.0];
        let l = complex_cholesky_nxn(&a, 2).expect("cholesky");

        // L·L^H should equal A
        let lh = complex_conjugate_transpose(&l, 2, 2);
        let prod = complex_matmul(&l, &lh, 2, 2, 2);
        for i in 0..4 {
            assert!(
                (prod[2 * i] - a[2 * i]).abs() < 1e-12
                    && (prod[2 * i + 1] - a[2 * i + 1]).abs() < 1e-12,
                "reconstruction mismatch at entry {i}"
            );
        }
    }

    #[test]
    fn complex_cholesky_3x3_hermitian() {
        // A = [[9, 3-i, 1], [3+i, 5, 2-i], [1, 2+i, 4]]
        let a = [
            9.0, 0.0, 3.0, -1.0, 1.0, 0.0, 3.0, 1.0, 5.0, 0.0, 2.0, -1.0, 1.0, 0.0, 2.0, 1.0, 4.0,
            0.0,
        ];
        let l = complex_cholesky_nxn(&a, 3).expect("cholesky");
        let lh = complex_conjugate_transpose(&l, 3, 3);
        let prod = complex_matmul(&l, &lh, 3, 3, 3);
        for i in 0..9 {
            assert!(
                (prod[2 * i] - a[2 * i]).abs() < 1e-10
                    && (prod[2 * i + 1] - a[2 * i + 1]).abs() < 1e-10,
                "reconstruction mismatch at entry {i}: got {}+{}i, expected {}+{}i",
                prod[2 * i],
                prod[2 * i + 1],
                a[2 * i],
                a[2 * i + 1]
            );
        }
    }

    #[test]
    fn complex_cholesky_rejects_non_pd() {
        // Not positive-definite: [[1, 2+i], [2-i, 1]]
        let a = [1.0, 0.0, 2.0, 1.0, 2.0, -1.0, 1.0, 0.0];
        assert!(complex_cholesky_nxn(&a, 2).is_err());
    }

    #[test]
    fn complex_qr_2x2_reconstruction() {
        // A = [[1+i, 2], [3, 4-i]]
        let a = [1.0, 1.0, 2.0, 0.0, 3.0, 0.0, 4.0, -1.0];
        let (q, r) = complex_qr_mxn(&a, 2, 2).expect("qr");

        // Q·R should equal A
        let prod = complex_matmul(&q, &r, 2, 2, 2);
        for i in 0..4 {
            assert!(
                (prod[2 * i] - a[2 * i]).abs() < 1e-10
                    && (prod[2 * i + 1] - a[2 * i + 1]).abs() < 1e-10,
                "QR reconstruction mismatch at {i}"
            );
        }

        // Q should be unitary: Q^H · Q = I
        let qh = complex_conjugate_transpose(&q, 2, 2);
        let qtq = complex_matmul(&qh, &q, 2, 2, 2);
        for i in 0..2 {
            for j in 0..2 {
                let re = qtq[2 * (i * 2 + j)];
                let im = qtq[2 * (i * 2 + j) + 1];
                let exp = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (re - exp).abs() < 1e-10 && im.abs() < 1e-10,
                    "Q not unitary at ({i},{j}): {re}+{im}i"
                );
            }
        }
    }

    #[test]
    fn complex_qr_3x2_tall() {
        // Tall 3×2 complex matrix
        let a = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let (q, r) = complex_qr_mxn(&a, 3, 2).expect("qr");

        // Q is 3×3, R is 3×2
        assert_eq!(q.len(), 2 * 3 * 3);
        assert_eq!(r.len(), 2 * 3 * 2);

        // Q·R should equal A
        let prod = complex_matmul(&q, &r, 3, 3, 2);
        for i in 0..6 {
            assert!(
                (prod[2 * i] - a[2 * i]).abs() < 1e-10
                    && (prod[2 * i + 1] - a[2 * i + 1]).abs() < 1e-10,
                "QR tall reconstruction mismatch at {i}"
            );
        }
    }

    #[test]
    fn complex_conjugate_transpose_basic() {
        // A = [[1+2i, 3+4i], [5+6i, 7+8i]]
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let ah = complex_conjugate_transpose(&a, 2, 2);
        // A^H = [[1-2i, 5-6i], [3-4i, 7-8i]]
        assert!((ah[0] - 1.0).abs() < 1e-15);
        assert!((ah[1] - (-2.0)).abs() < 1e-15);
        assert!((ah[2] - 5.0).abs() < 1e-15);
        assert!((ah[3] - (-6.0)).abs() < 1e-15);
        assert!((ah[4] - 3.0).abs() < 1e-15);
        assert!((ah[5] - (-4.0)).abs() < 1e-15);
        assert!((ah[6] - 7.0).abs() < 1e-15);
        assert!((ah[7] - (-8.0)).abs() < 1e-15);
    }

    #[test]
    fn complex_matmul_identity() {
        let a = [1.0, 1.0, 2.0, 0.0, 3.0, -1.0, 4.0, 2.0];
        let id = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let prod = complex_matmul(&a, &id, 2, 2, 2);
        for i in 0..8 {
            assert!((prod[i] - a[i]).abs() < 1e-15, "mismatch at {i}");
        }
    }

    #[test]
    fn complex_trace_basic() {
        // [[1+2i, 3], [0, 4-i]] → trace = (1+2i) + (4-i) = 5+i
        let a = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0, -1.0];
        let (tr, ti) = complex_trace_nxn(&a, 2).expect("trace");
        assert!((tr - 5.0).abs() < 1e-15);
        assert!((ti - 1.0).abs() < 1e-15);
    }

    #[test]
    fn complex_frobenius_norm() {
        // [[1+i, 0], [0, 2-i]] → ||A||_F = sqrt(|1+i|² + |2-i|²) = sqrt(2+5) = sqrt(7)
        let a = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, -1.0];
        let norm = complex_matrix_norm_frobenius(&a);
        assert!((norm - 7.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn complex_det_singular() {
        // Singular: [[1+i, 2+2i], [1, 2]]
        let a = [1.0, 1.0, 2.0, 2.0, 1.0, 0.0, 2.0, 0.0];
        let (dr, di) = complex_det_nxn(&a, 2).expect("det");
        assert!(dr.abs() < 1e-10, "det_re={dr}");
        assert!(di.abs() < 1e-10, "det_im={di}");
    }

    #[test]
    fn complex_solve_3x3() {
        // Build a well-conditioned 3×3 complex system, solve, then verify A·x = b
        let a = [
            2.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 0.0, 3.0, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0, -1.0, 4.0,
            0.0,
        ];
        let b = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let x = complex_solve_nxn(&a, &b, 3).expect("solve");

        // Verify: A·x = b
        let ax = complex_matvec(&a, &x, 3, 3);
        for i in 0..3 {
            assert!(
                (ax[2 * i] - b[2 * i]).abs() < 1e-10
                    && (ax[2 * i + 1] - b[2 * i + 1]).abs() < 1e-10,
                "Ax[{i}] = {}+{}i, expected {}+{}i",
                ax[2 * i],
                ax[2 * i + 1],
                b[2 * i],
                b[2 * i + 1]
            );
        }
    }

    // ── NumPy oracle parity tests ──────────────────────────────────────
    //
    // Verify our linalg operations produce numerically identical results
    // to NumPy (within floating-point tolerance).  Values generated from
    // NumPy 2.x with np.set_printoptions(precision=17).

    const ORACLE_TOL: f64 = 1e-12;

    fn oracle_python_bin() -> String {
        std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".to_string())
    }

    fn numpy_oracle_available() -> bool {
        Command::new(oracle_python_bin())
            .args(["-c", "import numpy"])
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    fn numpy_oracle_matrix_rank_tol(a: &[f64], m: usize, n: usize, tol: Option<f64>) -> usize {
        let matrix_arg = format!(
            "[{}]",
            a.iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let tol_arg = tol
            .map(|value| value.to_string())
            .unwrap_or_else(|| "__none__".to_string());
        let script = r#"
import json
import sys
import numpy as np

data = json.loads(sys.argv[1])
m = int(sys.argv[2])
n = int(sys.argv[3])
tol = sys.argv[4]
matrix = np.array(data, dtype=float).reshape((m, n))
if tol == "__none__":
    rank = np.linalg.matrix_rank(matrix)
else:
    rank = np.linalg.matrix_rank(matrix, tol=float(tol))
print(int(rank))
"#;
        let output = Command::new(oracle_python_bin())
            .arg("-c")
            .arg(script)
            .arg(matrix_arg)
            .arg(m.to_string())
            .arg(n.to_string())
            .arg(tol_arg)
            .output()
            .expect("python oracle should launch");
        assert!(
            output.status.success(),
            "NumPy matrix_rank oracle must succeed"
        );
        String::from_utf8(output.stdout)
            .expect("oracle stdout must be utf-8")
            .trim()
            .parse::<usize>()
            .expect("oracle rank")
    }

    #[derive(Debug, Clone, PartialEq)]
    enum PinvOracleOutcome {
        Values(Vec<f64>),
        Error(String),
    }

    fn numpy_oracle_pinv_tolerance_aliases(
        a: &[f64],
        m: usize,
        n: usize,
        rcond: Option<f64>,
        rtol: Option<Option<f64>>,
    ) -> PinvOracleOutcome {
        let matrix_arg = format!(
            "[{}]",
            a.iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let rcond_arg = rcond
            .map(|value| value.to_string())
            .unwrap_or_else(|| "__omitted__".to_string());
        let (rtol_state, rtol_value) = match rtol {
            None => ("omitted".to_string(), String::new()),
            Some(None) => ("none".to_string(), String::new()),
            Some(Some(value)) => ("value".to_string(), value.to_string()),
        };
        let script = r#"
import json
import sys
import numpy as np

data = json.loads(sys.argv[1])
m = int(sys.argv[2])
n = int(sys.argv[3])
rcond_arg = sys.argv[4]
rtol_state = sys.argv[5]
rtol_value = sys.argv[6]
matrix = np.array(data, dtype=float).reshape((m, n))
kwargs = {}
if rcond_arg != "__omitted__":
    kwargs["rcond"] = float(rcond_arg)
if rtol_state == "none":
    kwargs["rtol"] = None
elif rtol_state == "value":
    kwargs["rtol"] = float(rtol_value)
try:
    result = np.linalg.pinv(matrix, **kwargs)
    print("ok\t" + ",".join(str(value) for value in result.reshape(-1).tolist()))
except Exception as exc:
    print("err\t" + str(exc))
"#;
        let output = Command::new(oracle_python_bin())
            .arg("-c")
            .arg(script)
            .arg(matrix_arg)
            .arg(m.to_string())
            .arg(n.to_string())
            .arg(rcond_arg)
            .arg(rtol_state)
            .arg(rtol_value)
            .output()
            .expect("python oracle should launch");
        assert!(output.status.success(), "NumPy pinv oracle must succeed");
        let payload = String::from_utf8(output.stdout).expect("oracle stdout must be utf-8");
        let trimmed = payload.trim();
        let (kind, body) = trimmed
            .split_once('\t')
            .expect("oracle payload must include kind and body");
        match kind {
            "ok" => {
                let values = if body.is_empty() {
                    Vec::new()
                } else {
                    body.split(',')
                        .map(|value| value.parse::<f64>().expect("oracle float"))
                        .collect::<Vec<_>>()
                };
                PinvOracleOutcome::Values(values)
            }
            "err" => PinvOracleOutcome::Error(body.to_string()),
            _ => panic!("unknown oracle payload kind: {kind}"),
        }
    }

    fn numpy_oracle_pinv_hermitian(
        a: &[f64],
        n: usize,
        rcond: Option<f64>,
        rtol: Option<Option<f64>>,
    ) -> PinvOracleOutcome {
        let matrix_arg = format!(
            "[{}]",
            a.iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let rcond_arg = rcond
            .map(|value| value.to_string())
            .unwrap_or_else(|| "__omitted__".to_string());
        let (rtol_state, rtol_value) = match rtol {
            None => ("omitted".to_string(), String::new()),
            Some(None) => ("none".to_string(), String::new()),
            Some(Some(value)) => ("value".to_string(), value.to_string()),
        };
        let script = r#"
import json
import sys
import numpy as np

data = json.loads(sys.argv[1])
n = int(sys.argv[2])
rcond_arg = sys.argv[3]
rtol_state = sys.argv[4]
rtol_value = sys.argv[5]
matrix = np.array(data, dtype=float).reshape((n, n))
kwargs = {"hermitian": True}
if rcond_arg != "__omitted__":
    kwargs["rcond"] = float(rcond_arg)
if rtol_state == "none":
    kwargs["rtol"] = None
elif rtol_state == "value":
    kwargs["rtol"] = float(rtol_value)
try:
    result = np.linalg.pinv(matrix, **kwargs)
    print("ok\t" + ",".join(str(value) for value in result.reshape(-1).tolist()))
except Exception as exc:
    print("err\t" + str(exc))
"#;
        let output = Command::new(oracle_python_bin())
            .arg("-c")
            .arg(script)
            .arg(matrix_arg)
            .arg(n.to_string())
            .arg(rcond_arg)
            .arg(rtol_state)
            .arg(rtol_value)
            .output()
            .expect("python hermitian pinv oracle should launch");
        assert!(
            output.status.success(),
            "NumPy hermitian pinv oracle must succeed"
        );
        let payload = String::from_utf8(output.stdout).expect("oracle stdout must be utf-8");
        let trimmed = payload.trim();
        let (kind, body) = trimmed
            .split_once('\t')
            .expect("oracle payload must include kind and body");
        match kind {
            "ok" => {
                let values = if body.is_empty() {
                    Vec::new()
                } else {
                    body.split(',')
                        .map(|value| value.parse::<f64>().expect("oracle float"))
                        .collect::<Vec<_>>()
                };
                PinvOracleOutcome::Values(values)
            }
            "err" => PinvOracleOutcome::Error(body.to_string()),
            _ => panic!("unknown oracle payload kind: {kind}"),
        }
    }

    fn assert_oracle(label: &str, got: f64, expected: f64) {
        assert!(
            (got - expected).abs() < ORACLE_TOL,
            "{label}: got {got:.17e}, expected {expected:.17e}, diff {:.2e}",
            (got - expected).abs()
        );
    }

    fn assert_oracle_vec(label: &str, got: &[f64], expected: &[f64]) {
        assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < ORACLE_TOL,
                "{label}[{i}]: got {g:.17e}, expected {e:.17e}, diff {:.2e}",
                (g - e).abs()
            );
        }
    }

    fn assert_pinv_oracle_outcome(
        label: &str,
        actual: &PinvOracleOutcome,
        expected: &PinvOracleOutcome,
    ) {
        match (actual, expected) {
            (PinvOracleOutcome::Values(actual), PinvOracleOutcome::Values(expected)) => {
                assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
                for (index, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
                    let tol = ORACLE_TOL * want.abs().max(1.0);
                    assert!(
                        (got - want).abs() <= tol,
                        "{label}[{index}]: got {got:.17e}, expected {want:.17e}, diff {:.2e}",
                        (got - want).abs()
                    );
                }
            }
            (PinvOracleOutcome::Error(actual), PinvOracleOutcome::Error(expected)) => {
                assert_eq!(actual, expected, "{label}: error mismatch");
            }
            _ => panic!("{label}: outcome kind mismatch actual={actual:?} expected={expected:?}"),
        }
    }

    #[test]
    fn numpy_oracle_det_2x2() {
        let a = [[1.0, 2.0], [3.0, 4.0]];
        assert_oracle("det_2x2", det_2x2(a).unwrap(), -2.0);
    }

    #[test]
    fn numpy_oracle_det_3x3() {
        let b = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        assert_oracle("det_3x3", det_nxn(&b, 3).unwrap(), 8.0);
    }

    #[test]
    fn numpy_oracle_inv_2x2() {
        let inv = inv_2x2([[1.0, 2.0], [3.0, 4.0]]).unwrap();
        let flat = [inv[0][0], inv[0][1], inv[1][0], inv[1][1]];
        assert_oracle_vec("inv_2x2", &flat, &[-2.0, 1.0, 1.5, -0.5]);
    }

    #[test]
    fn numpy_oracle_inv_3x3() {
        let b = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let inv = inv_nxn(&b, 3).unwrap();
        assert_oracle_vec(
            "inv_3x3",
            &inv,
            &[0.625, -0.25, 0.125, -0.25, 0.5, -0.25, 0.125, -0.25, 0.625],
        );
    }

    #[test]
    fn numpy_oracle_solve_2x2() {
        let x = solve_2x2([[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0]).unwrap();
        assert_oracle_vec("solve_2x2", &x, &[-4.0, 4.5]);
    }

    #[test]
    fn numpy_oracle_solve_3x3() {
        let a = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let x = solve_nxn(&a, &[1.0, 2.0, 3.0], 3).unwrap();
        assert_oracle_vec("solve_3x3", &x, &[0.5, 0.0, 1.5]);
    }

    #[test]
    fn numpy_oracle_eig_2x2() {
        // eig_nxn_full returns (eigenvalues_interleaved_re_im, eigenvectors)
        let (eigenvalues, _eigenvectors) = eig_nxn_full(&[1.0, 2.0, 3.0, 4.0], 2).unwrap();
        // eigenvalues are interleaved [re0, im0, re1, im1]
        let mut vals_re: Vec<f64> = eigenvalues.chunks(2).map(|c| c[0]).collect();
        vals_re.sort_by(f64::total_cmp);
        assert_oracle("eig[0]", vals_re[0], -0.3722813232690143);
        assert_oracle("eig[1]", vals_re[1], 5.372281323269014);
    }

    #[test]
    fn numpy_oracle_eigvalsh_3x3() {
        let b = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let mut vals = eigvalsh_nxn(&b, 3).unwrap();
        vals.sort_by(f64::total_cmp);
        assert_oracle("eigvalsh[0]", vals[0], 1.0);
        assert_oracle("eigvalsh[1]", vals[1], 2.0);
        assert_oracle("eigvalsh[2]", vals[2], 4.0);
    }

    #[test]
    fn numpy_oracle_svd_2x2() {
        let result = svd_2x2([[1.0, 2.0], [3.0, 4.0]], true).unwrap();
        let mut s = result.singular_values.to_vec();
        s.sort_by(|a, b| b.total_cmp(a));
        assert_oracle("svd_s[0]", s[0], 5.464985704219043);
        assert_oracle("svd_s[1]", s[1], 0.3659661906262575);
    }

    #[test]
    fn numpy_oracle_cholesky_2x2() {
        let l = cholesky_2x2([[4.0, 2.0], [2.0, 3.0]], "L").unwrap();
        let flat = [l[0][0], l[0][1], l[1][0], l[1][1]];
        assert_oracle_vec(
            "cholesky",
            &flat,
            &[2.0, 0.0, 1.0, std::f64::consts::SQRT_2],
        );
    }

    #[test]
    fn numpy_oracle_norm_frobenius() {
        assert_oracle(
            "norm_fro",
            matrix_norm_frobenius(&[1.0, 2.0, 3.0, 4.0], 2).unwrap(),
            5.477225575051661,
        );
    }

    #[test]
    fn numpy_oracle_cond_2x2() {
        let c = cond_nxn(&[1.0, 2.0, 3.0, 4.0], 2).unwrap();
        assert_oracle("cond", c, 14.933034373659265);
    }

    #[test]
    fn numpy_oracle_slogdet_2x2() {
        let (sign, logabsdet) = slogdet_nxn(&[1.0, 2.0, 3.0, 4.0], 2).unwrap();
        assert_oracle("slogdet_sign", sign, -1.0);
        assert_oracle("slogdet_logabsdet", logabsdet, 0.6931471805599455);
    }

    #[test]
    fn numpy_oracle_lstsq() {
        let a = [1.0, 1.0, 1.0, 2.0, 1.0, 3.0];
        let result = lstsq_nxn(&a, &[1.0, 2.0, 2.0], 3, 2).unwrap();
        assert_oracle_vec("lstsq_x", &result, &[0.6666666666666666, 0.5]);
    }

    #[test]
    fn lstsq_negative_rcond_matches_numpy_default_and_none() {
        let a = [1.0, 0.0, 0.0, 2e-16];
        let b = [1.0, 2e-16];

        // NumPy 2.0 reference for this boundary case:
        // - omitted rcond / rcond=None => rank 1, x = [1, 0]
        // - explicit rcond=0.0       => rank 2, x = [1, 1]
        let actual_default = lstsq_svd(&a, &b, 2, 2, -1.0).expect("default rcond");
        assert_oracle_vec("lstsq default x", &actual_default.0, &[1.0, 0.0]);
        assert!(
            actual_default.1.is_empty(),
            "default residuals should be empty"
        );
        assert_eq!(
            actual_default.2, 1,
            "default rank should match NumPy omitted/None"
        );
        assert_oracle_vec(
            "lstsq default singular_values",
            &actual_default.3,
            &[1.0, 2e-16],
        );

        let explicit_zero = lstsq_svd(&a, &b, 2, 2, 0.0).expect("explicit zero rcond");
        assert_oracle_vec("lstsq zero x", &explicit_zero.0, &[1.0, 1.0]);
        assert!(
            explicit_zero.1.is_empty(),
            "explicit-zero residuals should be empty"
        );
        assert_eq!(explicit_zero.2, 2, "explicit zero should keep full rank");
        assert_oracle_vec(
            "lstsq zero singular_values",
            &explicit_zero.3,
            &[1.0, 2e-16],
        );
    }

    #[test]
    fn numpy_oracle_qr_2x2() {
        let result = qr_2x2([[1.0, 2.0], [3.0, 4.0]], QrMode::Reduced).unwrap();
        // QR decomposition sign convention may differ from NumPy (Householder).
        // Verify |R| diagonal matches and R is upper-triangular.
        assert_oracle("qr |R[0,0]|", result.r[0][0].abs(), 3.1622776601683795);
        assert_oracle("qr |R[1,1]|", result.r[1][1].abs(), 0.6324555320336753);
        assert_oracle("qr R[1,0]", result.r[1][0], 0.0); // lower triangle is zero
    }

    #[test]
    fn numpy_oracle_matrix_rank() {
        assert_eq!(matrix_rank_2x2([[1.0, 2.0], [3.0, 4.0]], 1e-10).unwrap(), 2);
        assert_eq!(matrix_rank_2x2([[1.0, 2.0], [2.0, 4.0]], 1e-10).unwrap(), 1);
    }

    #[test]
    fn numpy_oracle_matrix_rank_tol_reference() {
        let diag2_small = [1.0, 0.0, 0.0, 1e-18];
        assert_eq!(matrix_rank_mxn_tol(&diag2_small, 2, 2, None).unwrap(), 1);
        assert_eq!(
            matrix_rank_mxn_tol(&diag2_small, 2, 2, Some(0.0)).unwrap(),
            2
        );
        assert_eq!(
            matrix_rank_mxn_tol(&diag2_small, 2, 2, Some(1e-18)).unwrap(),
            1
        );

        let diag3 = [1.0, 0.0, 0.0, 0.0, 1e-14, 0.0, 0.0, 0.0, 1e-16];
        assert_eq!(matrix_rank_nxn_tol(&diag3, 3, None).unwrap(), 2);
        assert_eq!(matrix_rank_nxn_tol(&diag3, 3, Some(0.0)).unwrap(), 3);
        assert_eq!(matrix_rank_nxn_tol(&diag3, 3, Some(1e-13)).unwrap(), 1);
    }

    #[test]
    fn matrix_rank_tol_matches_live_numpy_oracle_when_available() {
        if !numpy_oracle_available() {
            return;
        }

        let diag2_small = [1.0, 0.0, 0.0, 1e-18];
        for tol in [None, Some(0.0), Some(1e-18), Some(1e-17)] {
            let actual = matrix_rank_mxn_tol(&diag2_small, 2, 2, tol).expect("rank");
            let expected = numpy_oracle_matrix_rank_tol(&diag2_small, 2, 2, tol);
            assert_eq!(actual, expected, "diag2_small tol={tol:?}");
        }

        let diag3 = [1.0, 0.0, 0.0, 0.0, 1e-14, 0.0, 0.0, 0.0, 1e-16];
        for tol in [None, Some(0.0), Some(1e-15), Some(1e-13)] {
            let actual = matrix_rank_nxn_tol(&diag3, 3, tol).expect("rank");
            let expected = numpy_oracle_matrix_rank_tol(&diag3, 3, 3, tol);
            assert_eq!(actual, expected, "diag3 tol={tol:?}");
        }
    }

    #[test]
    fn pinv_tolerance_aliases_match_live_numpy_oracle_when_available() {
        if !numpy_oracle_available() {
            return;
        }

        let matrix = [1.0, 0.0, 0.0, 1e-12];
        let cases = [
            (None, None),
            (None, Some(None)),
            (None, Some(Some(1e-9))),
            (Some(-1.0), None),
            (Some(f64::NAN), None),
            (Some(1e-9), Some(None)),
        ];

        for (rcond, rtol) in cases {
            let expected = numpy_oracle_pinv_tolerance_aliases(&matrix, 2, 2, rcond, rtol);

            let actual_2x2 = match pinv_2x2_with_tolerance_aliases(
                [[matrix[0], matrix[1]], [matrix[2], matrix[3]]],
                rcond,
                rtol,
            ) {
                Ok(values) => PinvOracleOutcome::Values(vec![
                    values[0][0],
                    values[0][1],
                    values[1][0],
                    values[1][1],
                ]),
                Err(err) => PinvOracleOutcome::Error(format!("{err}")),
            };
            assert_pinv_oracle_outcome(
                &format!("2x2 rcond={rcond:?} rtol={rtol:?}"),
                &actual_2x2,
                &expected,
            );

            let actual_mxn = match pinv_mxn_with_tolerance_aliases(&matrix, 2, 2, rcond, rtol) {
                Ok(values) => PinvOracleOutcome::Values(values),
                Err(err) => PinvOracleOutcome::Error(format!("{err}")),
            };
            assert_pinv_oracle_outcome(
                &format!("mxn rcond={rcond:?} rtol={rtol:?}"),
                &actual_mxn,
                &expected,
            );
        }
    }

    #[test]
    fn pinv_hermitian_matches_live_numpy_oracle_when_available() {
        if !numpy_oracle_available() {
            return;
        }

        let matrix = [2.0, 4.0, 1.0, 2.0];
        let cases = [
            (None, None),
            (Some(1e-9), None),
            (None, Some(None)),
            (None, Some(Some(1e-9))),
            (Some(1e-9), Some(None)),
        ];

        for (rcond, rtol) in cases {
            let expected = numpy_oracle_pinv_hermitian(&matrix, 2, rcond, rtol);
            let actual = match pinv_hermitian_nxn_with_tolerance_aliases(&matrix, 2, rcond, rtol) {
                Ok(values) => PinvOracleOutcome::Values(values),
                Err(err) => PinvOracleOutcome::Error(format!("{err}")),
            };
            assert_pinv_oracle_outcome(
                &format!("hermitian rcond={rcond:?} rtol={rtol:?}"),
                &actual,
                &expected,
            );
        }
    }

    #[test]
    fn batch_solve_broadcasts_matrix_rhs_across_stacked_lhs() {
        // Two stacked 2×2 systems: A0 = diag(2,3), A1 = diag(4,5)
        // Shared matrix RHS B = [[1,0],[0,1]] (identity)
        // Expected: X0 = diag(1/2, 1/3), X1 = diag(1/4, 1/5)
        let a = vec![2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0];
        let a_shape = [2, 2, 2];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let b_shape = [2, 2];
        let solved = batch_solve(&a, &a_shape, &b, &b_shape, false).expect("batch_solve");
        assert_eq!(solved.len(), 8); // 2 batches × 2×2
        assert!((solved[0] - 0.5).abs() < 1e-12, "X0[0,0]={}", solved[0]);
        assert!(solved[1].abs() < 1e-12, "X0[0,1]={}", solved[1]);
        assert!(solved[2].abs() < 1e-12, "X0[1,0]={}", solved[2]);
        assert!(
            (solved[3] - 1.0 / 3.0).abs() < 1e-12,
            "X0[1,1]={}",
            solved[3]
        );
        assert!((solved[4] - 0.25).abs() < 1e-12, "X1[0,0]={}", solved[4]);
        assert!(solved[5].abs() < 1e-12, "X1[0,1]={}", solved[5]);
        assert!(solved[6].abs() < 1e-12, "X1[1,0]={}", solved[6]);
        assert!((solved[7] - 0.2).abs() < 1e-12, "X1[1,1]={}", solved[7]);
    }

    #[test]
    fn batch_det_singular_matrix() {
        // Stack with a singular matrix: [[0,0],[0,0]]
        let data = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let dets = batch_det(&data, &[2, 2, 2]).unwrap();
        assert_eq!(dets.len(), 2);
        assert!((dets[0] - 1.0).abs() < 1e-12); // det(I) = 1
        assert!(dets[1].abs() < 1e-12); // det(0) = 0
    }

    #[test]
    fn batch_inv_identity() {
        // Invert a batch of identity matrices
        let data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let inv = batch_inv(&data, &[2, 2, 2]).unwrap();
        assert_eq!(inv.len(), 8);
        // Both should be identity
        for (i, &v) in inv.iter().enumerate() {
            let expected = if i % 4 == 0 || i % 4 == 3 { 1.0 } else { 0.0 };
            assert!(
                (v - expected).abs() < 1e-12,
                "batch_inv[{}] = {}, expected {}",
                i,
                v,
                expected
            );
        }
    }

    #[test]
    fn batch_trace_basic() {
        // [[1,2],[3,4]] trace=5, [[5,6],[7,8]] trace=13
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let traces = batch_trace(&data, &[2, 2, 2]).unwrap();
        assert_eq!(traces.len(), 2);
        assert!((traces[0] - 5.0).abs() < 1e-12);
        assert!((traces[1] - 13.0).abs() < 1e-12);
    }

    #[test]
    fn batch_trace_direct_lane_fill_matches_per_lane_reference_bits() {
        fn reference(data: &[f64], batch: usize, n: usize) -> Vec<f64> {
            let mat_size = n * n;
            (0..batch)
                .map(|b| {
                    trace_nxn(&data[b * mat_size..(b + 1) * mat_size], n)
                        .expect("reference trace")
                })
                .collect()
        }

        for &(batch, n) in &[(3usize, 3usize), (512, 8)] {
            let mat_size = n * n;
            let mut data: Vec<f64> = (0..batch * mat_size)
                .map(|idx| ((idx % 97) as f64 - 48.0) * 0.25)
                .collect();
            for b in 0..batch {
                let base = b * mat_size;
                if b % 17 == 0 {
                    data[base] = f64::NAN;
                }
                if b % 19 == 0 && n > 1 {
                    data[base + n + 1] = -0.0;
                }
            }

            let got = batch_trace(&data, &[batch, n, n]).expect("batch_trace");
            let want = reference(&data, batch, n);
            assert_eq!(got.len(), want.len());
            for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "batch trace bit drift for batch={batch}, n={n}, lane={idx}"
                );
            }
        }
    }

    #[test]
    fn batch_matrix_norm_fro_direct_lane_fill_matches_per_lane_reference_bits() {
        fn reference(data: &[f64], batch: usize, m: usize, n: usize) -> Vec<f64> {
            let mat_size = m * n;
            (0..batch)
                .map(|b| {
                    matrix_norm_nxn(&data[b * mat_size..(b + 1) * mat_size], m, n, "fro")
                        .expect("reference fro norm")
                })
                .collect()
        }

        for &(batch, m, n) in &[(3usize, 3usize, 5usize), (512, 8, 8)] {
            let mat_size = m * n;
            let mut data: Vec<f64> = (0..batch * mat_size)
                .map(|idx| ((idx % 127) as f64 - 63.0) * 0.125)
                .collect();
            for b in 0..batch {
                let base = b * mat_size;
                if b % 17 == 0 {
                    data[base] = f64::NAN;
                }
                if b % 19 == 0 && mat_size > 1 {
                    data[base + 1] = f64::INFINITY;
                }
                if b % 23 == 0 && mat_size > 2 {
                    data[base + 2] = -0.0;
                }
            }

            let got = batch_matrix_norm(&data, &[batch, m, n], "fro").expect("batch fro norm");
            let want = reference(&data, batch, m, n);
            assert_eq!(got.len(), want.len());
            for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "batch fro norm bit drift for batch={batch}, shape={m}x{n}, lane={idx}"
                );
            }
        }

        let empty = batch_matrix_norm(&[], &[0, 0, 8], "fro").expect("empty batch");
        assert!(empty.is_empty());
        assert!(batch_matrix_norm(&[], &[1, 0, 8], "fro").is_err());
    }

    #[test]
    fn batch_matrix_norm_column_sum_direct_lane_fill_matches_per_lane_reference_bits() {
        fn reference(data: &[f64], batch: usize, m: usize, n: usize, ord: &str) -> Vec<f64> {
            let mat_size = m * n;
            (0..batch)
                .map(|b| {
                    matrix_norm_nxn(&data[b * mat_size..(b + 1) * mat_size], m, n, ord)
                        .expect("reference column-sum norm")
                })
                .collect()
        }

        for ord in ["1", "-1"] {
            for &(batch, m, n) in &[(3usize, 3usize, 5usize), (512, 8, 8)] {
                let mat_size = m * n;
                let mut data: Vec<f64> = (0..batch * mat_size)
                    .map(|idx| ((idx % 131) as f64 - 65.0) * 0.15625)
                    .collect();
                for b in 0..batch {
                    let base = b * mat_size;
                    if b % 17 == 0 {
                        data[base] = f64::NAN;
                    }
                    if b % 19 == 0 && mat_size > 1 {
                        data[base + 1] = -0.0;
                    }
                    if b % 23 == 0 && mat_size > n {
                        data[base + n] = 0.0;
                    }
                }

                let got =
                    batch_matrix_norm(&data, &[batch, m, n], ord).expect("batch column-sum norm");
                let want = reference(&data, batch, m, n, ord);
                assert_eq!(got.len(), want.len());
                for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "batch {ord} norm bit drift for batch={batch}, shape={m}x{n}, lane={idx}"
                    );
                }
            }

            let empty = batch_matrix_norm(&[], &[0, 0, 8], ord).expect("empty batch");
            assert!(empty.is_empty());
            assert!(batch_matrix_norm(&[], &[1, 0, 8], ord).is_err());
        }
    }

    #[test]
    fn batch_matrix_norm_row_sum_direct_lane_fill_matches_per_lane_reference_bits() {
        fn reference(data: &[f64], batch: usize, m: usize, n: usize, ord: &str) -> Vec<f64> {
            let mat_size = m * n;
            (0..batch)
                .map(|b| {
                    matrix_norm_nxn(&data[b * mat_size..(b + 1) * mat_size], m, n, ord)
                        .expect("reference row-sum norm")
                })
                .collect()
        }

        for ord in ["inf", "-inf"] {
            for &(batch, m, n) in &[(3usize, 3usize, 5usize), (512, 8, 8)] {
                let mat_size = m * n;
                let mut data: Vec<f64> = (0..batch * mat_size)
                    .map(|idx| ((idx % 113) as f64 - 56.0) * 0.1875)
                    .collect();
                for b in 0..batch {
                    let base = b * mat_size;
                    if b % 17 == 0 {
                        data[base] = f64::NAN;
                    }
                    if b % 19 == 0 && mat_size > 1 {
                        data[base + 1] = -0.0;
                    }
                    if b % 23 == 0 && mat_size > n {
                        data[base + n] = 0.0;
                    }
                }

                let got =
                    batch_matrix_norm(&data, &[batch, m, n], ord).expect("batch row-sum norm");
                let want = reference(&data, batch, m, n, ord);
                assert_eq!(got.len(), want.len());
                for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "batch {ord} norm bit drift for batch={batch}, shape={m}x{n}, lane={idx}"
                    );
                }
            }

            let empty = batch_matrix_norm(&[], &[0, 0, 8], ord).expect("empty batch");
            assert!(empty.is_empty());
            assert!(batch_matrix_norm(&[], &[1, 0, 8], ord).is_err());
        }
    }

    #[test]
    fn batch_det_shape_mismatch() {
        // Non-square last two dims
        assert!(batch_det(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).is_err());
    }

    #[test]
    fn batch_inv_inv_roundtrip() {
        // Inv(Inv(A)) should return A
        let a = vec![2.0, 1.0, 1.0, 3.0];
        let inv1 = batch_inv(&a, &[2, 2]).unwrap();
        let inv2 = batch_inv(&inv1, &[2, 2]).unwrap();
        for (i, (&orig, &back)) in a.iter().zip(inv2.iter()).enumerate() {
            assert!(
                (orig - back).abs() < 1e-10,
                "inv(inv(A))[{}] = {}, expected {}",
                i,
                back,
                orig
            );
        }
    }
}
