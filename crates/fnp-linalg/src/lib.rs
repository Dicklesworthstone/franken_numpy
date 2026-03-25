#![forbid(unsafe_code)]

use core::fmt;

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
    // Create identity matrix as RHS
    let mut eye = vec![0.0; n * n];
    for i in 0..n {
        eye[i * n + i] = 1.0;
    }
    lu_solve_multi(&lu, &perm, &eye, n, n)
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
    if lu.len() != n * n || b.len() != n * m || n == 0 || m == 0 {
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

/// Solve A*x = b given the lower Cholesky factor L where A = L*L^T.
///
/// Two-step forward/back substitution:
///   1. Solve L*y = b  (forward substitution)
///   2. Solve L^T*x = y (backward substitution)
///
/// `l` is the n×n lower-triangular Cholesky factor (row-major, n*n elements).
/// `b` is the n-element right-hand side.
pub fn cholesky_solve(l: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, LinAlgError> {
    if l.len() != n * n || b.len() != n || n == 0 {
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
    if l.len() != n * n || b.len() != n * m || n == 0 || m == 0 {
        return Err(LinAlgError::CholeskyContractViolation(
            "cholesky_solve_multi: L must be n*n, B must be n*m, with n,m > 0",
        ));
    }

    let mut x = vec![0.0; n * m];

    // Process each column of B
    for col in 0..m {
        // Forward substitution: L*y = b
        // Store y temporarily in x
        for i in 0..n {
            let mut sum = b[i * m + col];
            for j in 0..i {
                sum -= l[i * n + j] * x[j * m + col];
            }
            x[i * m + col] = sum / l[i * n + i];
        }

        // Backward substitution: L^T*x = y
        // We read the 'y' values from x and overwrite them with 'x' values
        for i in (0..n).rev() {
            let mut sum = x[i * m + col];
            for j in (i + 1)..n {
                sum -= l[j * n + i] * x[j * m + col]; // L^T[i,j] = L[j,i]
            }
            x[i * m + col] = sum / l[i * n + i];
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
) -> Result<Vec<f64>, LinAlgError> {
    if a_data.iter().chain(b_data.iter()).any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "tensorsolve requires finite entries",
        ));
    }

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
    let n: usize = b_shape.iter().product();
    if b_data.len() != n {
        return Err(LinAlgError::ShapeContractViolation(
            "tensorsolve: b_data length must equal product of b_shape",
        ));
    }
    let m: usize = x_shape.iter().product(); // product of x_shape

    if n != m {
        return Err(LinAlgError::ShapeContractViolation(
            "tensorsolve: reshaped system must be square (prod(b_shape) == prod(x_shape))",
        ));
    }

    if a_data.len() != n * m {
        return Err(LinAlgError::ShapeContractViolation(
            "tensorsolve: a_data length must be prod(a_shape)",
        ));
    }

    // Now solve the n×n system
    solve_nxn(a_data, b_data, n)
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
    if a_data.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::NormDetRankPolicyViolation(
            "tensorinv requires finite entries",
        ));
    }

    let output_shape = &a_shape[..ind];
    let input_shape = &a_shape[ind..];
    let n: usize = output_shape.iter().product();
    let m: usize = input_shape.iter().product();

    if n != m {
        return Err(LinAlgError::ShapeContractViolation(
            "tensorinv: product of first ind dims must equal product of remaining dims",
        ));
    }
    if a_data.len() != n * n {
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

    let (_, sigmas, _) = svd_bidiag_full(a, m, n)?;
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

/// Core SVD implementation using Golub-Kahan bidiagonalization + Golub-Reinsch QR.
///
/// Returns (U, sigmas, Vt) where U is m×m, sigmas has min(m,n) entries, Vt is n×n.
fn svd_bidiag_full(a: &[f64], m: usize, n: usize) -> Result<SvdFullResult, LinAlgError> {
    let k = m.min(n);

    // Handle m < n by working with A^T and swapping U/V
    if m < n {
        // A^T is n×m row-major
        let mut at = vec![0.0; n * m];
        for i in 0..m {
            for j in 0..n {
                at[j * m + i] = a[i * n + j];
            }
        }
        let (u_t, sigmas, vt_t) = svd_bidiag_full(&at, n, m)?;
        // SVD(A^T) = U_t * S * Vt_t → SVD(A) = Vt_t^T * S * U_t^T
        // So U = Vt_t^T (m×m), Vt = U_t^T (n×n)
        let u = transpose_mat(&vt_t, m, m);
        let vt = transpose_mat(&u_t, n, n);
        return Ok((u, sigmas, vt));
    }

    // Now m >= n. Bidiagonalize A into U1 * B * V1^T
    // B is n×n upper bidiagonal: diagonal d[0..n], superdiagonal e[0..n-1]
    let mut work = a.to_vec(); // m×n working copy
    let mut d = vec![0.0; n]; // diagonal
    let mut e = vec![0.0; n.saturating_sub(1)]; // superdiagonal

    // U1 = product of left Householder reflections (m×m)
    let mut u = vec![0.0; m * m];
    for i in 0..m {
        u[i * m + i] = 1.0;
    }
    // V1 = product of right Householder reflections (n×n)
    let mut vt = vec![0.0; n * n];
    for i in 0..n {
        vt[i * n + i] = 1.0;
    }

    let mut v_house = vec![0.0; m];
    let mut w_house = vec![0.0; n];

    for j in 0..n {
        // Left Householder: zero out column j below diagonal
        let col_norm = {
            let mut s = 0.0;
            for i in j..m {
                s += work[i * n + j] * work[i * n + j];
            }
            s.sqrt()
        };
        if col_norm > 0.0 {
            let sign = if work[j * n + j] >= 0.0 { 1.0 } else { -1.0 };
            for vi in &mut v_house[..j] {
                *vi = 0.0;
            }
            for (i, vi) in v_house[j..m].iter_mut().enumerate() {
                *vi = work[(i + j) * n + j];
            }
            v_house[j] += sign * col_norm;
            let v_norm_sq: f64 = v_house[j..].iter().map(|x| x * x).sum();
            if v_norm_sq > 0.0 {
                let scale = 2.0 / v_norm_sq;
                // Apply to work (left): work = (I - scale*v*v^T) * work
                for col in j..n {
                    let mut dot = 0.0;
                    for i in j..m {
                        dot += v_house[i] * work[i * n + col];
                    }
                    let f = scale * dot;
                    for i in j..m {
                        work[i * n + col] -= f * v_house[i];
                    }
                }
                // Accumulate into U: U = U * (I - scale*v*v^T)
                for row in 0..m {
                    let mut dot = 0.0;
                    for i in j..m {
                        dot += u[row * m + i] * v_house[i];
                    }
                    let f = scale * dot;
                    for i in j..m {
                        u[row * m + i] -= f * v_house[i];
                    }
                }
            }
        }
        d[j] = work[j * n + j];

        // Right Householder: zero out row j to the right of superdiagonal
        if j + 2 <= n {
            let row_norm = {
                let mut s = 0.0;
                for col in (j + 1)..n {
                    s += work[j * n + col] * work[j * n + col];
                }
                s.sqrt()
            };
            if row_norm > 0.0 {
                let sign = if work[j * n + j + 1] >= 0.0 {
                    1.0
                } else {
                    -1.0
                };
                for wi in &mut w_house[..=j] {
                    *wi = 0.0;
                }
                for (idx, wi) in w_house[(j + 1)..n].iter_mut().enumerate() {
                    *wi = work[j * n + (j + 1 + idx)];
                }
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
                    // Accumulate into Vt: Vt = (I - scale*w*w^T) * Vt
                    for col in 0..n {
                        let mut dot = 0.0;
                        for row in (j + 1)..n {
                            dot += w_house[row] * vt[row * n + col];
                        }
                        let f = scale * dot;
                        for row in (j + 1)..n {
                            vt[row * n + col] -= f * w_house[row];
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

    // Phase 2: Golub-Reinsch implicit QR iteration directly on the bidiagonal.
    // Works with (d, e) without forming B^T*B, avoiding condition-number squaring.
    // Left rotations accumulate into U (column operations), right into Vt (row operations).
    let eps_mach = f64::EPSILON;
    let max_iter = SVD_QR_ITERATION_COEFF * n * n;

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

            // Accumulate left rotation into U (columns kk, kk+1)
            for row in 0..m {
                let t1 = u[row * m + kk];
                let t2 = u[row * m + kk + 1];
                u[row * m + kk] = cs * t1 + sn * t2;
                u[row * m + kk + 1] = -sn * t1 + cs * t2;
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

    let u_old = u.clone();
    let vt_old = vt.clone();
    for (new_idx, &old_idx) in order.iter().enumerate() {
        for row in 0..m {
            u[row * m + new_idx] = u_old[row * m + old_idx];
        }
        for col in 0..n {
            vt[new_idx * n + col] = vt_old[old_idx * n + col];
        }
    }

    Ok((u, sigmas, vt))
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
        "-1" => {
            // Min absolute column sum
            let mut min_col = f64::INFINITY;
            for j in 0..n {
                let mut col_sum = 0.0;
                for i in 0..m {
                    col_sum += a[i * n + j].abs();
                }
                min_col = min_col.min(col_sum);
            }
            Ok(min_col)
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
        "-inf" => {
            // Min absolute row sum
            let mut min_row = f64::INFINITY;
            for i in 0..m {
                let mut row_sum = 0.0;
                for j in 0..n {
                    row_sum += a[i * n + j].abs();
                }
                min_row = min_row.min(row_sum);
            }
            Ok(min_row)
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
        "-2" => {
            // Smallest singular value
            if m != n {
                return Err(LinAlgError::ShapeContractViolation(
                    "spectral norm requires square matrix",
                ));
            }
            let sigmas = svd_nxn(a, n)?;
            Ok(sigmas.last().copied().unwrap_or(0.0))
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
            "unknown norm order; use fro, 1, -1, inf, -inf, 2, -2, or nuc",
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

/// Matrix rank via SVD for MxN matrices.
/// Returns the number of singular values above `rcond * sigma_max`.
pub fn matrix_rank_mxn(a: &[f64], m: usize, n: usize, rcond: f64) -> Result<usize, LinAlgError> {
    if a.len() != m * n || m == 0 || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "matrix_rank_mxn: input must be m*n with m,n > 0",
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

    let sigmas = svd_mxn(a, m, n)?;
    let sigma_max = sigmas.first().copied().unwrap_or(0.0);
    if sigma_max == 0.0 {
        return Ok(0);
    }
    let threshold = sigma_max * rcond;
    Ok(sigmas.iter().filter(|&&s| s >= threshold).count())
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

/// Reduce symmetric n×n matrix to tridiagonal form via Householder similarity
/// transformations.  Returns `(diagonal, off_diagonal, Q)` where `A = Q T Q^T`,
/// `T` has diagonal `d` and off-diagonal `e`, and `Q` is orthogonal (n×n row-major).
fn tridiag_reduce(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut work = a.to_vec();
    let mut q = vec![0.0; n * n];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }

    let mut v = vec![0.0; n];
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
        // Left: work = H * work
        for col in 0..n {
            let mut dot = 0.0;
            for i in (j + 1)..n {
                dot += v[i] * work[i * n + col];
            }
            let f = scale * dot;
            for i in (j + 1)..n {
                work[i * n + col] -= f * v[i];
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
    let e: Vec<f64> = (0..n.saturating_sub(1))
        .map(|i| work[i * n + (i + 1)])
        .collect();
    (d, e, q)
}

/// Implicit QR iteration on symmetric tridiagonal `(d, e)` to find eigenvalues.
/// Modifies `d` and `e` in-place; eigenvalues end up on `d`.  If `q` is `Some`,
/// accumulates the rotation product into the n×n matrix (for eigenvectors).
fn tridiag_eig_qr(d: &mut [f64], e: &mut [f64], mut q: Option<&mut [f64]>, n: usize) {
    let eps = f64::EPSILON;
    let max_iter = EIGEN_QR_ITERATION_COEFF * n * n;

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

            // Accumulate rotation into Q (columns kk, kk+1)
            if let Some(ref mut q) = q {
                for row in 0..n {
                    let t1 = q[row * n + kk];
                    let t2 = q[row * n + kk + 1];
                    q[row * n + kk] = c * t1 + s * t2;
                    q[row * n + kk + 1] = -s * t1 + c * t2;
                }
            }
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

        // Left: H = P * H
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
        // Right: H = H * P
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
        // Q = Q * P
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

    (h, q)
}

/// Explicit single-shift QR iteration on upper Hessenberg form with deflation.
/// Converges to quasi-upper-triangular (real Schur) form.
/// If `z` is `Some`, accumulates the Schur vectors.
fn hessenberg_qr_iter(h: &mut [f64], mut z: Option<&mut [f64]>, n: usize) {
    let eps = f64::EPSILON;
    let max_iter = EIGEN_QR_ITERATION_COEFF * n * n;
    let mut p = n; // active upper bound (exclusive of converged tail)

    // QR factorization of H[lo..p, lo..p] using Givens rotations
    let mut cos_vals = vec![0.0; n];
    let mut sin_vals = vec![0.0; n];

    for iter in 0..max_iter {
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

        // Wilkinson shift from trailing 2×2 of active block
        let a11 = h[(p - 2) * n + (p - 2)];
        let a12 = h[(p - 2) * n + (p - 1)];
        let a21 = h[(p - 1) * n + (p - 2)];
        let a22 = h[(p - 1) * n + (p - 1)];
        let trace = a11 + a22;
        let det = a11 * a22 - a12 * a21;
        let disc = trace * trace - 4.0 * det;
        let mu = if disc >= 0.0 {
            let sqrt_disc = disc.sqrt();
            let lam1 = (trace + sqrt_disc) / 2.0;
            let lam2 = (trace - sqrt_disc) / 2.0;
            if (lam1 - a22).abs() < (lam2 - a22).abs() {
                lam1
            } else {
                lam2
            }
        } else {
            // Complex eigenvalue pair: use exceptional shift every 10 iterations
            if iter % 10 == 0 {
                a22 + h[(p - 1) * n + (p - 2)].abs()
            } else {
                trace / 2.0
            }
        };

        // Subtract shift
        for i in lo..p {
            h[i * n + i] -= mu;
        }

        for k in lo..(p - 1) {
            let ff = h[k * n + k];
            let gg = h[(k + 1) * n + k];
            let r = ff.hypot(gg);
            let (c, s) = if r > 0.0 {
                (ff / r, gg / r)
            } else {
                (1.0, 0.0)
            };
            cos_vals[k] = c;
            sin_vals[k] = s;
            // Apply left rotation to rows k, k+1 (columns k..n)
            for j in k..n {
                let t1 = c * h[k * n + j] + s * h[(k + 1) * n + j];
                h[(k + 1) * n + j] = -s * h[k * n + j] + c * h[(k + 1) * n + j];
                h[k * n + j] = t1;
            }
        }

        // Form R * Q (right-multiply by G_lo .. G_{p-2})
        for k in lo..(p - 1) {
            let c = cos_vals[k];
            let s = sin_vals[k];
            // Apply right rotation to columns k, k+1 (rows 0..min(k+2, p))
            let row_end = (k + 2).min(p);
            for i in 0..row_end {
                let t1 = c * h[i * n + k] + s * h[i * n + k + 1];
                h[i * n + k + 1] = -s * h[i * n + k] + c * h[i * n + k + 1];
                h[i * n + k] = t1;
            }
        }

        // Add shift back
        for i in lo..p {
            h[i * n + i] += mu;
        }

        // Accumulate into Z: Z = Z * G_lo * ... * G_{p-2}
        if let Some(ref mut z) = z {
            for k in lo..(p - 1) {
                let c = cos_vals[k];
                let s = sin_vals[k];
                for row in 0..n {
                    let t1 = c * z[row * n + k] + s * z[row * n + k + 1];
                    z[row * n + k + 1] = -s * z[row * n + k] + c * z[row * n + k + 1];
                    z[row * n + k] = t1;
                }
            }
        }
    }
}

/// Compute eigenvalues of a symmetric NxN matrix via QR iteration.
/// Returns eigenvalues in ascending order (NumPy convention).
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

    let (mut d, mut e, _q) = tridiag_reduce(a, n);
    tridiag_eig_qr(&mut d, &mut e, None, n);

    d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(d)
}

/// Compute eigenvalues and eigenvectors of a symmetric NxN matrix via QR iteration.
/// Returns (eigenvalues, eigenvectors_flat) where eigenvectors are stored as columns
/// in a row-major n*n array. Eigenvalues are in ascending order (NumPy convention).
pub fn eigh_nxn(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), LinAlgError> {
    if a.len() != n * n || n == 0 {
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
    if a.len() != n * n || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "eig_nxn: input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::SpectralConvergenceFailed);
    }

    // Hessenberg reduction + implicit shifted QR
    let (mut h, _q) = hessenberg_reduce(a, n);
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
    if a.len() != n * n || n == 0 {
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

/// Matrix exponential via scaling-and-squaring with Taylor series.
///
/// Mimics `scipy.linalg.expm(A)`. Computes e^A for an NxN matrix.
/// Uses scaling-and-squaring: scale A by 2^(-s) so the norm is small,
/// compute exp via truncated Taylor series, then square s times.
pub fn expm_nxn(a: &[f64], n: usize) -> Result<Vec<f64>, LinAlgError> {
    if a.len() != n * n || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "expm_nxn: input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::SpectralConvergenceFailed);
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
        (norm1.log2().ceil() as u32) + 1
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
    if a.len() != n * n || n == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "sqrtm_nxn: input must be n*n with n > 0",
        ));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinAlgError::SpectralConvergenceFailed);
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
    if a.len() != n * n || n == 0 {
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
    let max_s = LOGM_MAX_SCALING_ITERATIONS as u32;

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
    if a.len() != n * n || n == 0 {
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
    if a.len() != n * n || n == 0 {
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
    if a.len() != m * n || b.len() != m || m == 0 || n == 0 {
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
    if a.len() != m * n || m == 0 || n == 0 {
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
    let threshold = if rcond < 0.0 {
        f64::EPSILON * (m.max(n) as f64) * sigma_max
    } else {
        rcond * sigma_max
    };

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

    let result = match order {
        VectorNormOrder::Zero => values.iter().filter(|v| **v != 0.0).count() as f64,
        VectorNormOrder::One => values.iter().map(|v| v.abs()).sum(),
        VectorNormOrder::Two => values.iter().map(|v| v * v).sum::<f64>().sqrt(),
        VectorNormOrder::Inf => values.iter().map(|v| v.abs()).fold(0.0, f64::max),
        VectorNormOrder::NegInf => values.iter().map(|v| v.abs()).fold(f64::INFINITY, f64::min),
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
        MatrixNormOrder::NegOne => {
            let col0 = matrix[0][0].abs() + matrix[1][0].abs();
            let col1 = matrix[0][1].abs() + matrix[1][1].abs();
            col0.min(col1)
        }
        MatrixNormOrder::Inf => {
            let row0 = matrix[0][0].abs() + matrix[0][1].abs();
            let row1 = matrix[1][0].abs() + matrix[1][1].abs();
            row0.max(row1)
        }
        MatrixNormOrder::NegInf => {
            let row0 = matrix[0][0].abs() + matrix[0][1].abs();
            let row1 = matrix[1][0].abs() + matrix[1][1].abs();
            row0.min(row1)
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
fn batch_count(shape: &[usize]) -> usize {
    if shape.len() <= 2 {
        1
    } else {
        shape[..shape.len() - 2].iter().product()
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
    let batch = batch_count(shape);
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

/// Batched matrix inversion: inv on (..., n, n) → (..., n, n).
pub fn batch_inv(data: &[f64], shape: &[usize]) -> Result<Vec<f64>, LinAlgError> {
    let (batch, n) = parse_batched_square(shape)?;
    let mat_size = n * n;
    if data.len() != batch * mat_size {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_inv: data length does not match shape",
        ));
    }
    let mut result = Vec::with_capacity(batch * mat_size);
    for b in 0..batch {
        let sub = &data[b * mat_size..(b + 1) * mat_size];
        let inv = inv_nxn(sub, n)?;
        result.extend_from_slice(&inv);
    }
    Ok(result)
}

/// Batched determinant: det on (..., n, n) → (...).
pub fn batch_det(data: &[f64], shape: &[usize]) -> Result<Vec<f64>, LinAlgError> {
    let (batch, n) = parse_batched_square(shape)?;
    let mat_size = n * n;
    if data.len() != batch * mat_size {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_det: data length does not match shape",
        ));
    }
    let mut result = Vec::with_capacity(batch);
    for b in 0..batch {
        let sub = &data[b * mat_size..(b + 1) * mat_size];
        result.push(det_nxn(sub, n)?);
    }
    Ok(result)
}

/// Batched slogdet: slogdet on (..., n, n) → (signs: ..., log_abs_dets: ...).
pub fn batch_slogdet(data: &[f64], shape: &[usize]) -> Result<(Vec<f64>, Vec<f64>), LinAlgError> {
    let (batch, n) = parse_batched_square(shape)?;
    let mat_size = n * n;
    if data.len() != batch * mat_size {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_slogdet: data length does not match shape",
        ));
    }
    let mut signs = Vec::with_capacity(batch);
    let mut logabsdets = Vec::with_capacity(batch);
    for b in 0..batch {
        let sub = &data[b * mat_size..(b + 1) * mat_size];
        let (sign, logabsdet) = slogdet_nxn(sub, n)?;
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
            b_shape[..b_shape.len() - 1].iter().product()
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
            b_shape[..b_shape.len() - 2].iter().product()
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
    if a.len() != a_batch * mat_size {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_solve: A data length does not match shape",
        ));
    }
    if b.len() != b_batch * rhs_width {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_solve: B data length does not match shape",
        ));
    }

    let mut result = Vec::with_capacity(batch * rhs_width);
    for idx in 0..batch {
        let a_idx = if a_batch == 1 { 0 } else { idx };
        let b_idx = if b_batch == 1 { 0 } else { idx };
        let a_sub = &a[a_idx * mat_size..(a_idx + 1) * mat_size];
        let b_sub = &b[b_idx * rhs_width..(b_idx + 1) * rhs_width];
        if vector_rhs {
            let x = solve_nxn(a_sub, b_sub, n)?;
            result.extend_from_slice(&x);
        } else {
            let x = solve_nxn_multi(a_sub, b_sub, n, rhs_cols)?;
            result.extend_from_slice(&x);
        }
    }
    Ok(result)
}

/// Batched eigenvalues: eigvalsh on (..., n, n) → (..., n).
pub fn batch_eigvalsh(data: &[f64], shape: &[usize]) -> Result<Vec<f64>, LinAlgError> {
    let (batch, n) = parse_batched_square(shape)?;
    let mat_size = n * n;
    if data.len() != batch * mat_size {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_eigvalsh: data length does not match shape",
        ));
    }
    let mut result = Vec::with_capacity(batch * n);
    for b in 0..batch {
        let sub = &data[b * mat_size..(b + 1) * mat_size];
        let eigvals = eigvalsh_nxn(sub, n)?;
        result.extend_from_slice(&eigvals);
    }
    Ok(result)
}

/// Batched eigh: eigh on (..., n, n) → (eigenvalues (..., n), eigenvectors (..., n, n)).
pub fn batch_eigh(data: &[f64], shape: &[usize]) -> Result<(Vec<f64>, Vec<f64>), LinAlgError> {
    let (batch, n) = parse_batched_square(shape)?;
    let mat_size = n * n;
    if data.len() != batch * mat_size {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_eigh: data length does not match shape",
        ));
    }
    let mut eigenvalues = Vec::with_capacity(batch * n);
    let mut eigenvectors = Vec::with_capacity(batch * mat_size);
    for b in 0..batch {
        let sub = &data[b * mat_size..(b + 1) * mat_size];
        let (vals, vecs) = eigh_nxn(sub, n)?;
        eigenvalues.extend_from_slice(&vals);
        eigenvectors.extend_from_slice(&vecs);
    }
    Ok((eigenvalues, eigenvectors))
}

/// Batched eig: eig on (..., n, n) → eigenvalues as interleaved (re,im) (..., n*2).
pub fn batch_eig(data: &[f64], shape: &[usize]) -> Result<Vec<f64>, LinAlgError> {
    let (batch, n) = parse_batched_square(shape)?;
    let mat_size = n * n;
    if data.len() != batch * mat_size {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_eig: data length does not match shape",
        ));
    }
    let mut result = Vec::with_capacity(batch * n * 2);
    for b in 0..batch {
        let sub = &data[b * mat_size..(b + 1) * mat_size];
        let eigvals = eig_nxn(sub, n)?;
        result.extend_from_slice(&eigvals);
    }
    Ok(result)
}

/// Batched SVD (singular values only): svd on (..., m, n) → (..., min(m,n)).
pub fn batch_svd(data: &[f64], shape: &[usize]) -> Result<Vec<f64>, LinAlgError> {
    let (batch, m, n) = parse_batched_shape(shape)?;
    let mat_size = m * n;
    let k = m.min(n);
    if data.len() != batch * mat_size {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_svd: data length does not match shape",
        ));
    }
    let mut result = Vec::with_capacity(batch * k);
    for b in 0..batch {
        let sub = &data[b * mat_size..(b + 1) * mat_size];
        let sigmas = svd_mxn(sub, m, n)?;
        result.extend_from_slice(&sigmas);
    }
    Ok(result)
}

/// Batched full SVD: svd on (..., m, n) → (U (..., m, m), S (..., min(m,n)), Vt (..., n, n)).
pub fn batch_svd_full(data: &[f64], shape: &[usize]) -> Result<SvdFullResult, LinAlgError> {
    let (batch, m, n) = parse_batched_shape(shape)?;
    let mat_size = m * n;
    let k = m.min(n);
    if data.len() != batch * mat_size {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_svd_full: data length does not match shape",
        ));
    }
    let mut all_u = Vec::with_capacity(batch * m * m);
    let mut all_s = Vec::with_capacity(batch * k);
    let mut all_vt = Vec::with_capacity(batch * n * n);
    for b in 0..batch {
        let sub = &data[b * mat_size..(b + 1) * mat_size];
        let (u, s, vt) = svd_mxn_full(sub, m, n)?;
        all_u.extend_from_slice(&u);
        all_s.extend_from_slice(&s);
        all_vt.extend_from_slice(&vt);
    }
    Ok((all_u, all_s, all_vt))
}

/// Batched QR: qr on (..., m, n) → (Q (..., m, m), R (..., m, n)).
pub fn batch_qr(data: &[f64], shape: &[usize]) -> Result<(Vec<f64>, Vec<f64>), LinAlgError> {
    let (batch, m, n) = parse_batched_shape(shape)?;
    let mat_size = m * n;
    if data.len() != batch * mat_size {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_qr: data length does not match shape",
        ));
    }
    let mut all_q = Vec::with_capacity(batch * m * m);
    let mut all_r = Vec::with_capacity(batch * m * n);
    for b in 0..batch {
        let sub = &data[b * mat_size..(b + 1) * mat_size];
        let (q, r) = qr_mxn(sub, m, n)?;
        all_q.extend_from_slice(&q);
        all_r.extend_from_slice(&r);
    }
    Ok((all_q, all_r))
}

/// Batched Cholesky: cholesky on (..., n, n) → (..., n, n).
pub fn batch_cholesky(data: &[f64], shape: &[usize]) -> Result<Vec<f64>, LinAlgError> {
    let (batch, n) = parse_batched_square(shape)?;
    let mat_size = n * n;
    if data.len() != batch * mat_size {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_cholesky: data length does not match shape",
        ));
    }
    let mut result = Vec::with_capacity(batch * mat_size);
    for b in 0..batch {
        let sub = &data[b * mat_size..(b + 1) * mat_size];
        let chol = cholesky_nxn(sub, n)?;
        result.extend_from_slice(&chol);
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
    if data.len() != batch * mat_size {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_matrix_norm: data length does not match shape",
        ));
    }
    let mut result = Vec::with_capacity(batch);
    for b in 0..batch {
        let sub = &data[b * mat_size..(b + 1) * mat_size];
        result.push(matrix_norm_nxn(sub, m, n, ord)?);
    }
    Ok(result)
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
    if data.len() != batch * mat_size {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_matrix_rank: data length does not match shape",
        ));
    }
    let mut result = Vec::with_capacity(batch);
    for b in 0..batch {
        let sub = &data[b * mat_size..(b + 1) * mat_size];
        result.push(matrix_rank_nxn(sub, n, rcond)?);
    }
    Ok(result)
}

/// Batched trace: trace on (..., n, n) → (...).
pub fn batch_trace(data: &[f64], shape: &[usize]) -> Result<Vec<f64>, LinAlgError> {
    let (batch, n) = parse_batched_square(shape)?;
    let mat_size = n * n;
    if data.len() != batch * mat_size {
        return Err(LinAlgError::ShapeContractViolation(
            "batch_trace: data length does not match shape",
        ));
    }
    let mut result = Vec::with_capacity(batch);
    for b in 0..batch {
        let sub = &data[b * mat_size..(b + 1) * mat_size];
        result.push(trace_nxn(sub, n)?);
    }
    Ok(result)
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
    if a.len() != 2 * n * n || n == 0 {
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
    if b.len() != 2 * n {
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
    if a.len() != 2 * n * n || n == 0 {
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
    if a.len() != 2 * n * n || n == 0 {
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
    if a.len() != 2 * m * n || m == 0 || n == 0 {
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
    if a.len() != 2 * n * n || n == 0 {
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
pub fn complex_matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; 2 * m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sr = 0.0;
            let mut si = 0.0;
            for p in 0..k {
                let (pr, pi) = cmul(
                    a[2 * (i * k + p)],
                    a[2 * (i * k + p) + 1],
                    b[2 * (p * n + j)],
                    b[2 * (p * n + j) + 1],
                );
                sr += pr;
                si += pi;
            }
            c[2 * (i * n + j)] = sr;
            c[2 * (i * n + j) + 1] = si;
        }
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
        batch_qr,
        batch_slogdet,
        batch_solve,
        batch_svd,
        batch_trace,
        cholesky_2x2,
        cholesky_nxn,
        cholesky_solve,
        cholesky_solve_multi,
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
        cond_nxn,
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
        lu_factor_nxn,
        lu_solve,
        mat_mul_flat,
        mat_mul_rect,
        matrix_norm_2x2,
        matrix_norm_frobenius,
        matrix_norm_nxn,
        matrix_power_nxn,
        matrix_rank_2x2,
        matrix_rank_nxn,
        multi_dot,
        pinv_2x2,
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
        svd_mxn,
        svd_mxn_full,
        svd_nxn,
        svd_output_shapes,
        tensorinv,
        tensorsolve,
        trace_nxn,
        validate_backend_bridge,
        validate_cholesky_diagonal,
        validate_matrix_shape,
        validate_policy_metadata,
        validate_spectral_branch,
        validate_square_matrix,
        validate_tolerance_policy,
        vector_norm,
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
        // Eigenvalues of this matrix: 1, 2, 4 (ascending, NumPy convention)
        // x^3 - 7x^2 + 14x - 8 = (x-1)(x-2)(x-4)
        assert!((eigvals[0] - 1.0).abs() < 1e-6, "eig0={}", eigvals[0]);
        assert!((eigvals[1] - 2.0).abs() < 1e-6, "eig1={}", eigvals[1]);
        assert!((eigvals[2] - 4.0).abs() < 1e-6, "eig2={}", eigvals[2]);
    }

    #[test]
    fn eigvalsh_repeated_eigenvalues() {
        // Matrix with repeated eigenvalue: diag(3, 3, 5)
        let a = [3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0];
        let eigvals = eigvalsh_nxn(&a, 3).expect("eigvalsh repeated");
        let mut sorted = eigvals.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
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
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
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
            assert!((recon[i] - a[i]).abs() < 1e-8, "SVD recon at {i}");
        }
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
        let x = tensorsolve(&a, &[2, 2], &b, &[2]).unwrap();
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
        real_eigs.sort_by(|a, b| b.partial_cmp(a).unwrap());
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
        vals_re.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_oracle("eig[0]", vals_re[0], -0.3722813232690143);
        assert_oracle("eig[1]", vals_re[1], 5.372281323269014);
    }

    #[test]
    fn numpy_oracle_eigvalsh_3x3() {
        let b = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let mut vals = eigvalsh_nxn(&b, 3).unwrap();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_oracle("eigvalsh[0]", vals[0], 1.0);
        assert_oracle("eigvalsh[1]", vals[1], 2.0);
        assert_oracle("eigvalsh[2]", vals[2], 4.0);
    }

    #[test]
    fn numpy_oracle_svd_2x2() {
        let result = svd_2x2([[1.0, 2.0], [3.0, 4.0]], true).unwrap();
        let mut s = result.singular_values.to_vec();
        s.sort_by(|a, b| b.partial_cmp(a).unwrap());
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
