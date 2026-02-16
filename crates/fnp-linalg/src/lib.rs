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
    if det.abs() <= f64::EPSILON {
        return Err(LinAlgError::SolverSingularity);
    }

    let inv_det = 1.0 / det;
    let x0 = (rhs[0] * lhs[1][1] - lhs[0][1] * rhs[1]) * inv_det;
    let x1 = (lhs[0][0] * rhs[1] - rhs[0] * lhs[1][0]) * inv_det;
    Ok([x0, x1])
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
        LINALG_PACKET_ID, LINALG_PACKET_REASON_CODES, LinAlgLogRecord, LinAlgRuntimeMode,
        MAX_BACKEND_REVALIDATION_ATTEMPTS, MAX_TOLERANCE_SEARCH_DEPTH, QrMode, lstsq_output_shapes,
        qr_output_shapes, solve_2x2, svd_output_shapes, validate_backend_bridge,
        validate_cholesky_diagonal, validate_matrix_shape, validate_policy_metadata,
        validate_spectral_branch, validate_square_matrix, validate_tolerance_policy,
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
}
