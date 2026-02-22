#![forbid(unsafe_code)]

use core::fmt;

pub const LINALG_PACKET_ID: &str = "FNP-P2C-008";
pub const MAX_TOLERANCE_SEARCH_DEPTH: usize = 128;
pub const MAX_BACKEND_REVALIDATION_ATTEMPTS: usize = 64;
pub const MAX_BATCH_SHAPE_CHECKS: usize = 2_000_000;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorNormOrder {
    One,
    Two,
    Inf,
    NegInf,
}

impl VectorNormOrder {
    pub fn from_token(token: &str) -> Result<Self, LinAlgError> {
        match token.trim().to_ascii_lowercase().as_str() {
            "1" => Ok(Self::One),
            "2" => Ok(Self::Two),
            "inf" | "+inf" => Ok(Self::Inf),
            "-inf" => Ok(Self::NegInf),
            _ => Err(LinAlgError::NormDetRankPolicyViolation(
                "unsupported vector norm order token",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixNormOrder {
    Fro,
    One,
    Inf,
    Two,
    NegTwo,
    Nuclear,
}

impl MatrixNormOrder {
    pub fn from_token(token: &str) -> Result<Self, LinAlgError> {
        match token.trim().to_ascii_lowercase().as_str() {
            "fro" | "f" => Ok(Self::Fro),
            "1" => Ok(Self::One),
            "inf" | "+inf" => Ok(Self::Inf),
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
            Self::ShapeContractViolation(msg) => write!(f, "{msg}"),
            Self::SolverSingularity => write!(f, "solve/inv rejected singular matrix"),
            Self::CholeskyContractViolation(msg) => write!(f, "{msg}"),
            Self::QrModeInvalid => write!(f, "qr mode is not one of reduced|complete|r|raw"),
            Self::SvdNonConvergence => write!(f, "svd did not converge"),
            Self::SpectralConvergenceFailed => write!(f, "spectral decomposition did not converge"),
            Self::LstsqTupleContractViolation(msg) => write!(f, "{msg}"),
            Self::NormDetRankPolicyViolation(msg) => write!(f, "{msg}"),
            Self::BackendBridgeInvalid(msg) => write!(f, "{msg}"),
            Self::PolicyUnknownMetadata(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for LinAlgError {}

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
    validate_finite_matrix_2x2(matrix)?;
    Ok(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])
}

pub fn slogdet_2x2(matrix: [[f64; 2]; 2]) -> Result<(f64, f64), LinAlgError> {
    let det = det_2x2(matrix)?;
    if det > 0.0 {
        Ok((1.0, det.ln()))
    } else if det < 0.0 {
        Ok((-1.0, (-det).ln()))
    } else {
        Ok((0.0, f64::NEG_INFINITY))
    }
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
fn lu_decompose(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<usize>, f64), LinAlgError> {
    if a.len() != n * n || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "LU input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "matrix entries must be finite for LU decomposition",
        ));
    }

    let matrix_max_abs = a.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let singularity_threshold = (n as f64) * f64::EPSILON * matrix_max_abs;

    let mut lu = a.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();
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
        for i in (k + 1)..n {
            let factor = lu[i * n + k] / pivot;
            lu[i * n + k] = factor;
            for j in (k + 1)..n {
                let u_val = lu[k * n + j];
                lu[i * n + j] -= factor * u_val;
            }
        }
    }

    Ok((lu, perm, sign))
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

/// Solve Ax = b for an NxN system via LU decomposition with partial pivoting.
/// `a` is n*n row-major, `b` has length n.
pub fn solve_nxn(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, LinAlgError> {
    if b.len() != n {
        return Err(LinAlgError::ShapeContractViolation(
            "solve_nxn: rhs length must equal n",
        ));
    }
    if b.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "rhs entries must be finite for solve",
        ));
    }

    let (lu, perm, _) = lu_decompose(a, n)?;
    Ok(lu_forward_back(&lu, &perm, b, n))
}

/// Determinant of an NxN matrix (flat row-major).  Returns 0.0 for singular
/// matrices instead of erroring.
pub fn det_nxn(a: &[f64], n: usize) -> Result<f64, LinAlgError> {
    if a.len() != n * n || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "det_nxn: input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "matrix entries must be finite for det",
        ));
    }

    match lu_decompose(a, n) {
        Ok((lu, _, sign)) => {
            let mut det = sign;
            for i in 0..n {
                det *= lu[i * n + i];
            }
            Ok(det)
        }
        Err(LinAlgError::SolverSingularity) => Ok(0.0),
        Err(e) => Err(e),
    }
}

/// Sign and log-absolute-determinant for an NxN matrix.
pub fn slogdet_nxn(a: &[f64], n: usize) -> Result<(f64, f64), LinAlgError> {
    if a.len() != n * n || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "slogdet_nxn: input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "matrix entries must be finite for slogdet",
        ));
    }

    match lu_decompose(a, n) {
        Ok((lu, _, sign)) => {
            let mut det_sign = sign;
            let mut log_abs_det = 0.0;
            for i in 0..n {
                let diag = lu[i * n + i];
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
        Err(LinAlgError::SolverSingularity) => Ok((0.0, f64::NEG_INFINITY)),
        Err(e) => Err(e),
    }
}

/// Inverse of an NxN matrix via LU decomposition.  Returns n*n flat
/// row-major.
pub fn inv_nxn(a: &[f64], n: usize) -> Result<Vec<f64>, LinAlgError> {
    let (lu, perm, _) = lu_decompose(a, n)?;
    let mut inv = vec![0.0; n * n];

    for col in 0..n {
        let e_col: Vec<f64> = (0..n).map(|i| if i == col { 1.0 } else { 0.0 }).collect();
        let x = lu_forward_back(&lu, &perm, &e_col, n);
        for (row, &val) in x.iter().enumerate() {
            inv[row * n + col] = val;
        }
    }

    Ok(inv)
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
    if lu.len() != n * n || n == 0 {
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
    if a.len() != n * n || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "solve_nxn_multi: A must be n*n with n > 0",
        ));
    }
    if b.len() != n * m {
        return Err(LinAlgError::ShapeContractViolation(
            "solve_nxn_multi: B must be n*m",
        ));
    }
    if b.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "rhs entries must be finite for solve",
        ));
    }

    let (lu, perm, _) = lu_decompose(a, n)?;

    let mut result = vec![0.0; n * m];
    for col in 0..m {
        let b_col: Vec<f64> = (0..n).map(|row| b[row * m + col]).collect();
        let x_col = lu_forward_back(&lu, &perm, &b_col, n);
        for (row, &val) in x_col.iter().enumerate() {
            result[row * m + col] = val;
        }
    }

    Ok(result)
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
    if a.len() != n * n || n == 0 {
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
pub fn cholesky_nxn(a: &[f64], n: usize) -> Result<Vec<f64>, LinAlgError> {
    if a.len() != n * n || n == 0 {
        return Err(LinAlgError::CholeskyContractViolation(
            "cholesky_nxn: input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::CholeskyContractViolation(
            "cholesky requires finite entries",
        ));
    }

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

/// QR decomposition via Householder reflections for NxN matrix.
/// Returns (q, r) as flat row-major n*n buffers.
pub fn qr_nxn(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), LinAlgError> {
    if a.len() != n * n || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "qr_nxn: input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "matrix entries must be finite for QR",
        ));
    }

    // Start with Q = I, R = A
    let mut q = vec![0.0; n * n];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }
    let mut r = a.to_vec();

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

        // Apply H = I - 2*v*v^T/||v||^2 to R
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

        // Accumulate Q = Q * H
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
    if a.len() != m * n || m == 0 || n == 0 {
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

        // Householder vector v
        let mut v = vec![0.0; m];
        let sign = if r[col * n + col] >= 0.0 { 1.0 } else { -1.0 };
        for i in col..m {
            v[i] = r[i * n + col];
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

/// SVD of an m×n rectangular matrix.
///
/// Input `a` is row-major with shape (m, n). Returns singular values in
/// descending order. Uses the approach: compute A^T*A (n×n), find its
/// eigenvalues via QR iteration, then take square roots.
///
/// For the full decomposition (U, S, V^T), use `svd_mxn_full`.
pub fn svd_mxn(a: &[f64], m: usize, n: usize) -> Result<Vec<f64>, LinAlgError> {
    if a.len() != m * n || m == 0 || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "svd_mxn: input must be m*n with m,n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "matrix entries must be finite for SVD",
        ));
    }

    // Compute A^T * A (n×n)
    let mut ata = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..m {
                sum += a[k * n + i] * a[k * n + j];
            }
            ata[i * n + j] = sum;
        }
    }

    // QR iteration on A^T A
    let mut mat = ata;
    for _ in 0..300 {
        let (q, r) = qr_nxn(&mat, n)?;
        let mut next = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += r[i * n + k] * q[k * n + j];
                }
                next[i * n + j] = sum;
            }
        }
        mat = next;
    }

    let k = m.min(n);
    let mut sigmas: Vec<f64> = (0..k).map(|i| {
        if i < n { mat[i * n + i].max(0.0).sqrt() } else { 0.0 }
    }).collect();
    sigmas.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Ok(sigmas)
}

/// Full SVD of an m×n rectangular matrix: returns (U, S, Vt).
///
/// - U: m×m orthogonal matrix (row-major, m*m elements)
/// - S: min(m,n) singular values in descending order
/// - Vt: n×n orthogonal matrix (row-major, n*n elements)
///
/// Uses the approach: QR-iterate on A^T*A to get V and sigma^2,
/// then recover U = A*V*diag(1/sigma).
pub fn svd_mxn_full(
    a: &[f64],
    m: usize,
    n: usize,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), LinAlgError> {
    if a.len() != m * n || m == 0 || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "svd_mxn_full: input must be m*n with m,n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "matrix entries must be finite for SVD",
        ));
    }

    let k = m.min(n);

    // Compute A^T * A (n×n)
    let mut ata = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for row in 0..m {
                sum += a[row * n + i] * a[row * n + j];
            }
            ata[i * n + j] = sum;
        }
    }

    // QR iteration on A^T A to get eigenvalues and eigenvectors (V)
    let mut mat = ata;
    let mut v_accum = vec![0.0; n * n];
    for i in 0..n {
        v_accum[i * n + i] = 1.0;
    }
    for _ in 0..300 {
        let (q, r) = qr_nxn(&mat, n)?;
        // mat = R * Q
        let mut next = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for kk in 0..n {
                    sum += r[i * n + kk] * q[kk * n + j];
                }
                next[i * n + j] = sum;
            }
        }
        mat = next;
        // Accumulate V = V * Q
        let mut new_v = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for kk in 0..n {
                    sum += v_accum[i * n + kk] * q[kk * n + j];
                }
                new_v[i * n + j] = sum;
            }
        }
        v_accum = new_v;
    }

    // Extract singular values and sort by descending order
    let mut sigma_sq: Vec<(f64, usize)> = (0..n)
        .map(|i| (mat[i * n + i].max(0.0), i))
        .collect();
    sigma_sq.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let sigmas: Vec<f64> = sigma_sq.iter().take(k).map(|(s, _)| s.sqrt()).collect();
    let order: Vec<usize> = sigma_sq.iter().map(|(_, idx)| *idx).collect();

    // Reorder V columns by sorted singular values -> Vt (n×n)
    let mut vt = vec![0.0; n * n];
    for (new_col, &old_col) in order.iter().enumerate() {
        for row in 0..n {
            vt[new_col * n + row] = v_accum[row * n + old_col];
        }
    }

    // Recover U: for each singular value sigma_i > 0, u_i = A * v_i / sigma_i
    // Start with identity for U
    let mut u = vec![0.0; m * m];
    for i in 0..m {
        u[i * m + i] = 1.0;
    }

    for j in 0..k {
        if sigmas[j] > 1e-15 {
            // u_j = A * v_j / sigma_j (v_j is row j of Vt, i.e. column j of V)
            for i in 0..m {
                let mut sum = 0.0;
                for col in 0..n {
                    sum += a[i * n + col] * vt[j * n + col];
                }
                u[i * m + j] = sum / sigmas[j];
            }
        }
    }

    // Orthogonalize remaining columns of U via Gram-Schmidt if m > k
    for j in k..m {
        // u_j is already e_j from identity; orthogonalize against previous columns
        for prev in 0..j {
            let mut dot = 0.0;
            for i in 0..m {
                dot += u[i * m + j] * u[i * m + prev];
            }
            for i in 0..m {
                u[i * m + j] -= dot * u[i * m + prev];
            }
        }
        // Normalize
        let norm: f64 = (0..m).map(|i| u[i * m + j] * u[i * m + j]).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for i in 0..m {
                u[i * m + j] /= norm;
            }
        }
    }

    Ok((u, sigmas, vt))
}

/// Frobenius norm of an NxN matrix (flat row-major).
pub fn matrix_norm_frobenius(a: &[f64], n: usize) -> Result<f64, LinAlgError> {
    if a.len() != n * n || n == 0 {
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

/// General NxN matrix norm (np.linalg.norm for matrices).
/// Supports: "fro" (Frobenius), "1" (max column sum), "inf" (max row sum),
/// "2" (spectral, i.e. largest singular value), "nuc" (nuclear/trace norm).
pub fn matrix_norm_nxn(a: &[f64], m: usize, n: usize, ord: &str) -> Result<f64, LinAlgError> {
    if a.len() != m * n || m == 0 || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "matrix_norm_nxn: input must be m*n with m,n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "matrix entries must be finite for norm",
        ));
    }
    match ord {
        "fro" => Ok(a.iter().map(|v| v * v).sum::<f64>().sqrt()),
        "1" => {
            // Max absolute column sum
            let mut max_col = 0.0_f64;
            for j in 0..n {
                let mut col_sum = 0.0;
                for i in 0..m {
                    col_sum += a[i * n + j].abs();
                }
                max_col = max_col.max(col_sum);
            }
            Ok(max_col)
        }
        "inf" => {
            // Max absolute row sum
            let mut max_row = 0.0_f64;
            for i in 0..m {
                let mut row_sum = 0.0;
                for j in 0..n {
                    row_sum += a[i * n + j].abs();
                }
                max_row = max_row.max(row_sum);
            }
            Ok(max_row)
        }
        "2" => {
            // Spectral norm = largest singular value
            if m != n {
                return Err(LinAlgError::ShapeContractViolation(
                    "spectral norm requires square matrix",
                ));
            }
            let sigmas = svd_nxn(a, n)?;
            Ok(sigmas.first().copied().unwrap_or(0.0))
        }
        "nuc" => {
            // Nuclear norm = sum of singular values
            if m != n {
                return Err(LinAlgError::ShapeContractViolation(
                    "nuclear norm requires square matrix",
                ));
            }
            let sigmas = svd_nxn(a, n)?;
            Ok(sigmas.iter().sum())
        }
        _ => Err(LinAlgError::NormDetRankPolicyViolation(
            "unknown norm order; use fro, 1, inf, 2, or nuc",
        )),
    }
}

/// Trace of an NxN flat matrix (sum of diagonal elements).
pub fn trace_nxn(a: &[f64], n: usize) -> Result<f64, LinAlgError> {
    if a.len() != n * n {
        return Err(LinAlgError::ShapeContractViolation(
            "trace_nxn: input must be n*n",
        ));
    }
    Ok((0..n).map(|i| a[i * n + i]).sum())
}

/// Matrix rank via SVD (uses QR iteration for singular values).
/// Returns the number of singular values above `rcond * sigma_max`.
pub fn matrix_rank_nxn(a: &[f64], n: usize, rcond: f64) -> Result<usize, LinAlgError> {
    if a.len() != n * n || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "matrix_rank_nxn: input must be n*n with n > 0",
        ));
    }
    if !rcond.is_finite() || rcond < 0.0 {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "rcond must be finite and >= 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "matrix entries must be finite for rank",
        ));
    }

    // Compute A^T A, then eigenvalues via QR iteration to get singular values
    let mut ata = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[k * n + i] * a[k * n + j];
            }
            ata[i * n + j] = sum;
        }
    }

    // QR iteration on A^T A to find eigenvalues (which are sigma^2)
    let mut m = ata;
    for _ in 0..200 {
        let (q, r) = qr_nxn(&m, n)?;
        // M = R * Q
        let mut next = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += r[i * n + k] * q[k * n + j];
                }
                next[i * n + j] = sum;
            }
        }
        m = next;
    }

    // Diagonal of converged matrix = eigenvalues of A^T A = sigma^2
    let mut sigmas: Vec<f64> = (0..n).map(|i| m[i * n + i].max(0.0).sqrt()).collect();
    sigmas.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let sigma_max = sigmas.first().copied().unwrap_or(0.0);
    if sigma_max == 0.0 {
        return Ok(0);
    }
    let threshold = sigma_max * rcond;
    Ok(sigmas.iter().filter(|&&s| s > threshold).count())
}

/// Compute singular values of an NxN matrix via QR iteration on A^T A.
/// Returns singular values in descending order.
pub fn svd_nxn(a: &[f64], n: usize) -> Result<Vec<f64>, LinAlgError> {
    if a.len() != n * n || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "svd_nxn: input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "matrix entries must be finite for SVD",
        ));
    }

    // Compute A^T A
    let mut ata = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[k * n + i] * a[k * n + j];
            }
            ata[i * n + j] = sum;
        }
    }

    // QR iteration on A^T A to find eigenvalues (sigma^2)
    let mut m = ata;
    for _ in 0..300 {
        let (q, r) = qr_nxn(&m, n)?;
        let mut next = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += r[i * n + k] * q[k * n + j];
                }
                next[i * n + j] = sum;
            }
        }
        m = next;
    }

    let mut sigmas: Vec<f64> = (0..n).map(|i| m[i * n + i].max(0.0).sqrt()).collect();
    sigmas.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Ok(sigmas)
}

/// Compute eigenvalues of a symmetric NxN matrix via QR iteration.
/// Returns eigenvalues in descending order.
/// The matrix must be symmetric; behavior is undefined for non-symmetric input.
pub fn eigvalsh_nxn(a: &[f64], n: usize) -> Result<Vec<f64>, LinAlgError> {
    if a.len() != n * n || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "eigvalsh_nxn: input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::SpectralConvergenceFailed);
    }

    // QR iteration with Wilkinson shift for faster convergence
    let mut m = a.to_vec();
    for _ in 0..300 {
        let (q, r) = qr_nxn(&m, n)?;
        let mut next = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += r[i * n + k] * q[k * n + j];
                }
                next[i * n + j] = sum;
            }
        }
        m = next;
    }

    let mut eigenvalues: Vec<f64> = (0..n).map(|i| m[i * n + i]).collect();
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Ok(eigenvalues)
}

/// Compute eigenvalues and eigenvectors of a symmetric NxN matrix via QR iteration.
/// Returns (eigenvalues, eigenvectors_flat) where eigenvectors are stored column-major
/// in a flat n*n array. Eigenvalues are in descending order.
pub fn eigh_nxn(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), LinAlgError> {
    if a.len() != n * n || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "eigh_nxn: input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::SpectralConvergenceFailed);
    }

    // Accumulate Q matrices: V = Q1 * Q2 * ... * Qk
    let mut m = a.to_vec();
    let mut v = vec![0.0; n * n];
    // Initialize V = I
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    for _ in 0..300 {
        let (q, r) = qr_nxn(&m, n)?;
        // M = R * Q
        let mut next = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += r[i * n + k] * q[k * n + j];
                }
                next[i * n + j] = sum;
            }
        }
        // V = V * Q
        let mut new_v = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += v[i * n + k] * q[k * n + j];
                }
                new_v[i * n + j] = sum;
            }
        }
        m = next;
        v = new_v;
    }

    let mut eigenvalues: Vec<f64> = (0..n).map(|i| m[i * n + i]).collect();

    // Sort eigenvalues descending and permute eigenvectors accordingly
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        eigenvalues[b]
            .partial_cmp(&eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_eigenvalues: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();
    let mut sorted_v = vec![0.0; n * n];
    for (col_out, &col_in) in indices.iter().enumerate() {
        for row in 0..n {
            sorted_v[row * n + col_out] = v[row * n + col_in];
        }
    }

    eigenvalues = sorted_eigenvalues;
    Ok((eigenvalues, sorted_v))
}

/// Eigenvalues of a general (possibly non-symmetric) NxN matrix via QR iteration.
/// Returns eigenvalues as interleaved (re, im) pairs: [re0, im0, re1, im1, ...].
/// For real eigenvalues, the imaginary part is 0.
/// The QR iteration converges to a quasi-upper-triangular (real Schur) form;
/// 1x1 diagonal blocks give real eigenvalues, 2x2 blocks give complex conjugate pairs.
pub fn eig_nxn(a: &[f64], n: usize) -> Result<Vec<f64>, LinAlgError> {
    if a.len() != n * n || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "eig_nxn: input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::SpectralConvergenceFailed);
    }

    // QR iteration to get real Schur form
    let mut m = a.to_vec();
    for _ in 0..500 {
        let (q, r) = qr_nxn(&m, n)?;
        let mut next = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += r[i * n + k] * q[k * n + j];
                }
                next[i * n + j] = sum;
            }
        }
        m = next;
    }

    // Extract eigenvalues from quasi-upper-triangular form
    let mut eigenvalues = Vec::with_capacity(n * 2);
    let mut i = 0;
    while i < n {
        if i + 1 < n && m[(i + 1) * n + i].abs() > 1e-10 {
            // 2x2 block: eigenvalues are complex conjugate pair
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
            // 1x1 block: real eigenvalue
            eigenvalues.push(m[i * n + i]);
            eigenvalues.push(0.0);
            i += 1;
        }
    }

    Ok(eigenvalues)
}

/// Schur decomposition of a general square matrix (scipy.linalg.schur).
///
/// Returns `(T, Z)` where `A = Z * T * Z^T`, `T` is quasi-upper-triangular
/// (real Schur form: 1x1 and 2x2 blocks on diagonal), and `Z` is orthogonal.
/// Both returned as row-major flat arrays of length `n*n`.
pub fn schur_nxn(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), LinAlgError> {
    if a.len() != n * n || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "schur_nxn: input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::SpectralConvergenceFailed);
    }

    let mut t = a.to_vec();
    let mut z = vec![0.0; n * n];
    for i in 0..n {
        z[i * n + i] = 1.0;
    }

    for _ in 0..500 {
        let (q, r) = qr_nxn(&t, n)?;
        // T = R * Q
        let mut next = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += r[i * n + k] * q[k * n + j];
                }
                next[i * n + j] = sum;
            }
        }
        // Z = Z * Q
        let mut new_z = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += z[i * n + k] * q[k * n + j];
                }
                new_z[i * n + j] = sum;
            }
        }
        t = next;
        z = new_z;
    }

    Ok((t, z))
}

/// Cross product of two 3-element vectors (np.cross for 3-D).
///
/// Returns `a × b = [a1*b2 - a2*b1, a2*b0 - a0*b2, a0*b1 - a1*b0]`.
pub fn cross_product(a: &[f64], b: &[f64]) -> Result<Vec<f64>, LinAlgError> {
    if a.len() != 3 || b.len() != 3 {
        return Err(LinAlgError::ShapeContractViolation(
            "cross_product: both inputs must have exactly 3 elements",
        ));
    }
    Ok(vec![
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
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
    if a.len() != m * n || b.len() != p * q {
        return Err(LinAlgError::ShapeContractViolation(
            "kron_nxn: input size mismatch",
        ));
    }
    let out_rows = m * p;
    let out_cols = n * q;
    let mut result = vec![0.0; out_rows * out_cols];
    for i in 0..m {
        for j in 0..n {
            let a_val = a[i * n + j];
            for k in 0..p {
                for l in 0..q {
                    result[(i * p + k) * out_cols + (j * q + l)] = a_val * b[k * q + l];
                }
            }
        }
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
    if a.len() != n * n || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "eig_nxn_full: input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::SpectralConvergenceFailed);
    }

    // QR iteration accumulating the product of Q matrices
    let mut m = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    for _ in 0..500 {
        let (q, r) = qr_nxn(&m, n)?;
        // M = R * Q
        let mut next = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += r[i * n + k] * q[k * n + j];
                }
                next[i * n + j] = sum;
            }
        }
        // V = V * Q
        let mut new_v = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += v[i * n + k] * q[k * n + j];
                }
                new_v[i * n + j] = sum;
            }
        }
        m = next;
        v = new_v;
    }

    // Extract eigenvalues from quasi-upper-triangular Schur form
    let mut eigenvalues = Vec::with_capacity(n * 2);
    // Build eigenvectors: for real eigenvalues the Schur vector is the eigenvector.
    // For complex conjugate pairs from a 2x2 block we reconstruct the complex eigenvectors.
    let mut eigvecs_re = vec![0.0; n * n];
    let mut eigvecs_im = vec![0.0; n * n];

    let mut i = 0;
    while i < n {
        if i + 1 < n && m[(i + 1) * n + i].abs() > 1e-10 {
            // 2x2 block: complex conjugate eigenvalue pair
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

            // Reconstruct complex eigenvectors from Schur vectors
            // For the 2x2 Schur block, eigenvector in Schur basis:
            //   [1, (lambda - a11)/a12] for the first eigenvalue
            // Then transform back: v_full = V * v_schur
            if disc < 0.0 {
                let imag = (-disc).sqrt() / 2.0;
                // Schur-basis vector: [1, (imag_part) / a12 * j] for first eigenvalue
                // s = (lambda - a11) / a12 = i*imag / a12
                let s_im = imag / a12;
                // First eigenvector (column i): V[:,i] + j * s_im * V[:,i+1]
                // Second eigenvector (column i+1): conjugate
                for row in 0..n {
                    eigvecs_re[row * n + i] = v[row * n + i];
                    eigvecs_im[row * n + i] = s_im * v[row * n + (i + 1)];
                    eigvecs_re[row * n + (i + 1)] = v[row * n + i];
                    eigvecs_im[row * n + (i + 1)] = -s_im * v[row * n + (i + 1)];
                }
            } else {
                // Real distinct eigenvalues from 2x2 block
                let sqrt_disc = disc.sqrt();
                let lam1 = (trace + sqrt_disc) / 2.0;
                // Schur-basis: [1, (lam1-a11)/a12]
                let s = if a12.abs() > 1e-15 {
                    (lam1 - a11) / a12
                } else {
                    0.0
                };
                for row in 0..n {
                    eigvecs_re[row * n + i] = v[row * n + i] + s * v[row * n + (i + 1)];
                    eigvecs_re[row * n + (i + 1)] = v[row * n + i]
                        + ((trace - sqrt_disc) / 2.0 - a11) / a12.max(1e-15) * v[row * n + (i + 1)];
                }
            }
            i += 2;
        } else {
            // 1x1 block: real eigenvalue, Schur vector = eigenvector
            eigenvalues.push(m[i * n + i]);
            eigenvalues.push(0.0);
            for row in 0..n {
                eigvecs_re[row * n + i] = v[row * n + i];
            }
            i += 1;
        }
    }

    // Normalize each eigenvector column
    for col in 0..n {
        let mut norm_sq = 0.0;
        for row in 0..n {
            let re = eigvecs_re[row * n + col];
            let im = eigvecs_im[row * n + col];
            norm_sq += re * re + im * im;
        }
        let norm = norm_sq.sqrt();
        if norm > 1e-15 {
            for row in 0..n {
                eigvecs_re[row * n + col] /= norm;
                eigvecs_im[row * n + col] /= norm;
            }
        }
    }

    // Interleave into output format: [re(v[0,0]), im(v[0,0]), re(v[1,0]), im(v[1,0]), ...]
    let mut eigvecs = Vec::with_capacity(2 * n * n);
    for col in 0..n {
        for row in 0..n {
            eigvecs.push(eigvecs_re[row * n + col]);
            eigvecs.push(eigvecs_im[row * n + col]);
        }
    }

    Ok((eigenvalues, eigvecs))
}

/// Condition number of a matrix (np.linalg.cond).
/// Uses the ratio of largest to smallest singular value (2-norm condition number).
pub fn cond_nxn(a: &[f64], n: usize) -> Result<f64, LinAlgError> {
    let sigmas = svd_nxn(a, n)?;
    let sigma_max = sigmas.first().copied().unwrap_or(0.0);
    let sigma_min = sigmas.last().copied().unwrap_or(0.0);
    if sigma_min == 0.0 {
        return Ok(f64::INFINITY);
    }
    Ok(sigma_max / sigma_min)
}

/// Matrix exponentiation: compute A^p for integer p (np.linalg.matrix_power).
/// Uses repeated squaring. p can be negative (requires invertible matrix).
pub fn matrix_power_nxn(a: &[f64], n: usize, p: i64) -> Result<Vec<f64>, LinAlgError> {
    if a.len() != n * n || n == 0 {
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
        cur = mat_mul_flat(&cur, &cur, n);
        exp >>= 1;
    }
    Ok(result)
}

/// NxN least-squares solve: minimize ||Ax - b||_2 using normal equations.
/// Returns the solution vector x.
pub fn lstsq_nxn(a: &[f64], b: &[f64], m: usize, n: usize) -> Result<Vec<f64>, LinAlgError> {
    if a.len() != m * n || b.len() != m || m == 0 || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "lstsq_nxn: a must be m*n, b must be m",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) || b.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "entries must be finite for lstsq",
        ));
    }

    // Normal equations: A^T A x = A^T b
    let mut ata = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..m {
                sum += a[k * n + i] * a[k * n + j];
            }
            ata[i * n + j] = sum;
        }
    }

    let mut atb = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for k in 0..m {
            sum += a[k * n + i] * b[k];
        }
        atb[i] = sum;
    }

    solve_nxn(&ata, &atb, n)
}

/// Pseudoinverse of an NxN matrix (np.linalg.pinv) via normal equations.
/// Computes A^+ = (A^T A)^{-1} A^T for full column rank matrices.
pub fn pinv_nxn(a: &[f64], m: usize, n: usize) -> Result<Vec<f64>, LinAlgError> {
    if a.len() != m * n || m == 0 || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "pinv_nxn: a must be m*n with m,n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "entries must be finite for pinv",
        ));
    }

    // Compute via lstsq for each column of identity
    let mut result = vec![0.0; n * m];
    for col in 0..m {
        let mut e_col = vec![0.0; m];
        e_col[col] = 1.0;
        let x = lstsq_nxn(a, &e_col, m, n)?;
        for row in 0..n {
            result[row * m + col] = x[row];
        }
    }
    Ok(result)
}

/// Helper: flat NxN matrix multiply C = A * B (row-major).
fn mat_mul_flat(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Rectangular matrix multiply: A (m×k) × B (k×n) → C (m×n).
fn mat_mul_rect(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
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
        return Err(LinAlgError::SpectralConvergenceFailed);
    }
    Ok(())
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
    let singular_values = singular_values_2x2(matrix)?;
    let sigma_max = singular_values[0];
    if sigma_max == 0.0 {
        return Ok(0);
    }

    let threshold = sigma_max * rcond;
    Ok(singular_values
        .iter()
        .filter(|&&sigma| sigma > threshold)
        .count())
}

pub fn pinv_2x2(matrix: [[f64; 2]; 2], rcond: f64) -> Result<[[f64; 2]; 2], LinAlgError> {
    validate_tolerance_policy(rcond, 0)?;
    validate_finite_matrix_2x2(matrix)?;

    let a = matrix[0][0];
    let b = matrix[0][1];
    let c = matrix[1][0];
    let d = matrix[1][1];

    // Right-singular vectors are eigenvectors of A^T A.
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
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "pinv eigendecomposition discriminant became invalid",
        ));
    }

    let sqrt_disc = disc.sqrt();
    let lambda_1 = ((trace + sqrt_disc) * 0.5).max(0.0);
    let lambda_2 = ((trace - sqrt_disc) * 0.5).max(0.0);
    let sigma_1 = lambda_1.sqrt();
    let sigma_2 = lambda_2.sqrt();

    let mut v1 = if m01.abs() > f64::EPSILON {
        [m01, lambda_1 - m00]
    } else if m00 >= m11 {
        [1.0, 0.0]
    } else {
        [0.0, 1.0]
    };
    let v1_norm = (v1[0].mul_add(v1[0], v1[1] * v1[1])).sqrt();
    if !v1_norm.is_finite() || v1_norm <= f64::EPSILON {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "pinv eigenvector normalization failed",
        ));
    }
    v1[0] /= v1_norm;
    v1[1] /= v1_norm;
    let v2 = [-v1[1], v1[0]];
    let sigmas = [sigma_1, sigma_2];
    let vectors = [v1, v2];

    let sigma_max = sigma_1;
    if sigma_max <= f64::EPSILON {
        return Ok([[0.0, 0.0], [0.0, 0.0]]);
    }
    let cutoff = sigma_max * rcond;

    let mut pinv = [[0.0_f64; 2]; 2];
    for idx in 0..2 {
        let sigma = sigmas[idx];
        if sigma <= cutoff {
            continue;
        }

        let v = vectors[idx];
        let av = [a.mul_add(v[0], b * v[1]), c.mul_add(v[0], d * v[1])];
        let u = [av[0] / sigma, av[1] / sigma];
        let inv_sigma = 1.0 / sigma;

        pinv[0][0] += inv_sigma * v[0] * u[0];
        pinv[0][1] += inv_sigma * v[0] * u[1];
        pinv[1][0] += inv_sigma * v[1] * u[0];
        pinv[1][1] += inv_sigma * v[1] * u[1];
    }

    Ok(pinv)
}

pub fn vector_norm(values: &[f64], ord: Option<VectorNormOrder>) -> Result<f64, LinAlgError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "vector norm requires finite entries",
        ));
    }

    let order = ord.unwrap_or(VectorNormOrder::Two);
    if values.is_empty() {
        if matches!(order, VectorNormOrder::NegInf) {
            return Err(LinAlgError::NormDetRankPolicyViolation(
                "negative infinity vector norm is undefined for empty inputs",
            ));
        }
        return Ok(0.0);
    }

    let abs_values = values.iter().map(|value| value.abs());
    let result = match order {
        VectorNormOrder::One => abs_values.sum(),
        VectorNormOrder::Two => abs_values.map(|value| value * value).sum::<f64>().sqrt(),
        VectorNormOrder::Inf => abs_values.fold(0.0, f64::max),
        VectorNormOrder::NegInf => abs_values.fold(f64::INFINITY, f64::min),
    };
    Ok(result)
}

pub fn matrix_norm_2x2(
    matrix: [[f64; 2]; 2],
    ord: Option<MatrixNormOrder>,
) -> Result<f64, LinAlgError> {
    validate_finite_matrix_2x2(matrix)?;

    let order = ord.unwrap_or(MatrixNormOrder::Fro);
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
            let col0 = matrix[0][0].abs() + matrix[1][0].abs();
            let col1 = matrix[0][1].abs() + matrix[1][1].abs();
            col0.max(col1)
        }
        MatrixNormOrder::Inf => {
            let row0 = matrix[0][0].abs() + matrix[0][1].abs();
            let row1 = matrix[1][0].abs() + matrix[1][1].abs();
            row0.max(row1)
        }
        MatrixNormOrder::Two => singular_values_2x2(matrix)?[0],
        MatrixNormOrder::NegTwo => singular_values_2x2(matrix)?[1],
        MatrixNormOrder::Nuclear => {
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
        _ => unreachable!(),
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
        _ => unreachable!(),
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
        QrMode::Raw => unreachable!(),
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
    let known_mode = mode == "strict" || mode == "hardened";
    let known_class = class == "known_compatible_low_risk"
        || class == "known_compatible_high_risk"
        || class == "known_incompatible_semantics"
        || class == "unknown_semantics";

    if !known_mode || !known_class {
        return Err(LinAlgError::PolicyUnknownMetadata(
            "unknown mode/class metadata rejected fail-closed",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        LINALG_PACKET_ID, LINALG_PACKET_REASON_CODES, LinAlgError, LinAlgLogRecord,
        LinAlgRuntimeMode, MAX_BACKEND_REVALIDATION_ATTEMPTS, MAX_BATCH_SHAPE_CHECKS,
        MAX_TOLERANCE_SEARCH_DEPTH, MatrixNormOrder, QrMode, VectorNormOrder, cholesky_2x2,
        cholesky_nxn, cond_nxn, cross_product, det_2x2, det_nxn, eig_nxn, eig_nxn_full, eigh_2x2,
        eigh_nxn, eigvals_2x2, eigvalsh_nxn, inv_2x2, inv_nxn, kron_nxn, lstsq_2x2, lstsq_nxn,
        lstsq_output_shapes, lu_factor_nxn, lu_solve, mat_mul_flat, mat_mul_rect, matrix_norm_2x2,
        matrix_norm_frobenius, matrix_norm_nxn, matrix_power_nxn, matrix_rank_2x2, matrix_rank_nxn,
        multi_dot, pinv_2x2, pinv_nxn, qr_2x2, qr_mxn, qr_nxn, qr_output_shapes, schur_nxn,
        slogdet_2x2, slogdet_nxn, solve_2x2, solve_nxn, solve_nxn_multi, solve_triangular,
        svd_2x2, svd_mxn, svd_mxn_full, svd_nxn,
        svd_output_shapes, trace_nxn, validate_backend_bridge, validate_cholesky_diagonal,
        validate_matrix_shape, validate_policy_metadata, validate_spectral_branch,
        validate_square_matrix, validate_tolerance_policy, vector_norm,
    };

    fn packet008_artifacts() -> Vec<String> {
        vec![
            "artifacts/phase2c/FNP-P2C-008/contract_table.md".to_string(),
            "artifacts/phase2c/FNP-P2C-008/unit_property_evidence.json".to_string(),
        ]
    }

    fn approx_equal(lhs: f64, rhs: f64, tol: f64) -> bool {
        (lhs - rhs).abs() <= tol
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
    fn norm_det_rank_pinv_reject_non_finite_inputs() {
        let err = det_2x2([[f64::NAN, 1.0], [2.0, 3.0]]).expect_err("nan matrix");
        assert_eq!(err.reason_code(), "linalg_norm_det_rank_policy_violation");

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
    fn norm_paths_reject_non_finite_inputs() {
        let err = vector_norm(&[f64::NAN, 1.0], None).expect_err("vector nan");
        assert_eq!(err.reason_code(), "linalg_norm_det_rank_policy_violation");

        let err = matrix_norm_2x2([[f64::INFINITY, 0.0], [0.0, 1.0]], None).expect_err("inf");
        assert_eq!(err.reason_code(), "linalg_norm_det_rank_policy_violation");
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
        assert_eq!(err.reason_code(), "linalg_spectral_convergence_failed");

        let err = eigvals_2x2([[0.0, -1.0], [1.0, 0.0]], true).expect_err("complex spectrum");
        assert_eq!(err.reason_code(), "linalg_spectral_convergence_failed");

        let err = eigh_2x2([[1.0, 0.0], [0.0, 1.0]], "X", true).expect_err("invalid uplo");
        assert_eq!(err.reason_code(), "linalg_spectral_convergence_failed");
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
    fn solve_nxn_rejects_non_finite() {
        let a = [f64::NAN, 1.0, 0.0, 1.0];
        let b = [1.0, 2.0];
        let err = solve_nxn(&a, &b, 2).expect_err("nan matrix");
        assert_eq!(err.reason_code(), "linalg_norm_det_rank_policy_violation");
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
    fn matrix_norm_frobenius_matches_expected() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let norm = matrix_norm_frobenius(&a, 2).expect("frobenius");
        let expected = (1.0 + 4.0 + 9.0 + 16.0_f64).sqrt();
        assert!(approx_equal(norm, expected, 1e-12));
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
        // Eigenvalues of this matrix: 1, 2, 4 (compute via characteristic polynomial)
        // x^3 - 7x^2 + 14x - 8 = (x-1)(x-2)(x-4)
        assert!((eigvals[0] - 4.0).abs() < 1e-6, "eig0={}", eigvals[0]);
        assert!((eigvals[1] - 2.0).abs() < 1e-6, "eig1={}", eigvals[1]);
        assert!((eigvals[2] - 1.0).abs() < 1e-6, "eig2={}", eigvals[2]);
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

    #[test]
    fn matrix_norm_nxn_spectral() {
        // Identity: spectral norm = 1
        let eye = [1.0, 0.0, 0.0, 1.0];
        let s = matrix_norm_nxn(&eye, 2, 2, "2").unwrap();
        assert!((s - 1.0).abs() < 1e-6, "spectral(I)={s}");
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
        vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!((vals[0] - 3.0).abs() < 1e-6, "eig0={}", vals[0]);
        assert!((vals[1] - 1.0).abs() < 1e-6, "eig1={}", vals[1]);
    }

    #[test]
    fn eig_nxn_diagonal_matrix() {
        let a = [5.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0];
        let eigs = eig_nxn(&a, 3).unwrap();
        assert_eq!(eigs.len(), 6);
        let mut reals: Vec<f64> = (0..3).map(|i| eigs[i * 2]).collect();
        reals.sort_by(|a, b| b.partial_cmp(a).unwrap());
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
        imags.sort_by(|a, b| a.partial_cmp(b).unwrap());
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
        diag.sort_by(|a, b| b.partial_cmp(a).unwrap());
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
    fn cross_product_rejects_wrong_size() {
        assert!(cross_product(&[1.0, 2.0], &[3.0, 4.0]).is_err());
    }

    // ── Kronecker product tests ──

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
        let mut qtq = vec![0.0; 9];
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
        let mut recon = vec![0.0; 6];
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
}
