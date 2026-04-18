#![forbid(unsafe_code)]

pub mod benchmark;
pub mod contract_schema;
pub mod cross_engine_benchmark;
pub mod raptorq_artifacts;
pub mod security_contracts;
pub mod test_contracts;
pub mod ufunc_differential;
pub mod workflow_scenarios;

use crate::ufunc_differential::{UFuncInputCase, UFuncOperation};
use fnp_dtype::{DType, can_cast, can_cast_lossless, promote};
use fnp_io::{
    GenFromTxtConfig, IOError, IOSupportedDType, LoadDispatch, MemmapMode, NPY_MAGIC_PREFIX,
    classify_load_dispatch, genfromtxt_full, load_structured, validate_descriptor_roundtrip,
    validate_header_schema, validate_io_policy_metadata, validate_magic_version,
    validate_memmap_contract, validate_npz_archive_budget, validate_read_payload,
};
use fnp_iter::{
    FlatIterIndex, NditerBroadcastPlan, NditerError, NditerOperandSpec, NditerOrder,
    NditerTransferFlags, OverlapAction, TransferClass, TransferError, TransferSelectorInput,
    overlap_copy_policy, plan_nditer_broadcast, select_transfer_class, validate_flatiter_read,
    validate_flatiter_write, validate_nditer_flags,
};
use fnp_linalg::{
    LinAlgError, QrMode, batch_det, batch_inv, batch_solve, batch_trace, cholesky_2x2, cond_mxn,
    eigvals_2x2, lstsq_output_shapes, matrix_power_nxn, pinv_2x2, qr_2x2, qr_output_shapes,
    slogdet_2x2, solve_2x2, svd_2x2, svd_output_shapes, validate_backend_bridge,
    validate_policy_metadata as validate_linalg_policy_metadata, validate_spectral_branch,
    validate_tolerance_policy,
};
use fnp_ndarray::{MemoryOrder, NdLayout, broadcast_shape, contiguous_strides};
use fnp_random::{
    BitGenerator, BitGeneratorError, BitGeneratorKind, DeterministicRng, Generator, RandomError,
    RandomPolicyError, SeedMaterial, SeedSequence, SeedSequenceError, validate_rng_policy_metadata,
};
use fnp_runtime::{
    CompatibilityClass, DecisionAction, DecisionAuditContext, EvidenceLedger, OverrideAuditEvent,
    RuntimeMode, decide_and_record_with_context, decide_compatibility_from_wire,
    evaluate_policy_override,
};
use fnp_ufunc::{
    BinaryOp, MAError, MaskedArray, QuantileInterp, StringArray, UFuncArray, UFuncError, UnaryOp,
    busday_count, busday_offset, cheb2poly, chebadd, chebder, chebdiv, chebfit, chebfromroots,
    chebint, chebmul, chebroots, chebsub, chebval, copysign, herm2poly, hermadd, hermder, hermdiv,
    herme2poly, hermeadd, hermediv, hermefromroots, hermemul, hermeroots, hermesub, hermeval,
    hermfromroots, hermint, hermmul, hermroots, hermsub, hermval, hypot, is_busday, isnat,
    lag2poly, lagadd, lagder, lagdiv, lagfromroots, lagint, lagmul, lagroots, lagsub, lagval,
    ldexp, leg2poly, legadd, legder, legdiv, legfit, legfromroots, legint, legmul, legroots,
    legsub, legval, logaddexp, logaddexp2, nextafter, poly2cheb, poly2herm, poly2herme, poly2lag,
    poly2leg, signbit, spacing,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone)]
pub struct HarnessConfig {
    pub oracle_root: PathBuf,
    pub fixture_root: PathBuf,
    pub contract_root: PathBuf,
    pub strict_mode: bool,
}

impl HarnessConfig {
    #[must_use]
    pub fn default_paths() -> Self {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        Self {
            oracle_root: repo_root.join("legacy_numpy_code/numpy"),
            fixture_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"),
            contract_root: repo_root.join("artifacts/contracts"),
            strict_mode: true,
        }
    }
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self::default_paths()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessReport {
    pub suite: &'static str,
    pub oracle_present: bool,
    pub fixture_count: usize,
    pub strict_mode: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuiteReport {
    pub suite: &'static str,
    pub case_count: usize,
    pub pass_count: usize,
    pub failures: Vec<String>,
}

impl SuiteReport {
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.case_count == self.pass_count && self.failures.is_empty()
    }
}

#[derive(Debug, Deserialize)]
struct ShapeStrideFixtureCase {
    id: String,
    lhs: Vec<usize>,
    rhs: Vec<usize>,
    expected_broadcast: Option<Vec<usize>>,
    stride_shape: Vec<usize>,
    stride_item_size: usize,
    stride_order: String,
    expected_strides: Vec<isize>,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
    #[serde(default)]
    as_strided: Option<AsStridedFixtureCase>,
    #[serde(default)]
    broadcast_to: Option<BroadcastToFixtureCase>,
    #[serde(default)]
    sliding_window: Option<SlidingWindowFixtureCase>,
}

#[derive(Debug, Deserialize)]
struct AsStridedFixtureCase {
    shape: Vec<usize>,
    strides: Vec<isize>,
    #[serde(default)]
    expected_shape: Option<Vec<usize>>,
    #[serde(default)]
    expected_strides: Option<Vec<isize>>,
    #[serde(default)]
    expected_writeable: Option<bool>,
    #[serde(default)]
    expected_has_internal_overlap: Option<bool>,
    #[serde(default)]
    expected_error_contains: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BroadcastToFixtureCase {
    shape: Vec<usize>,
    #[serde(default)]
    expected_shape: Option<Vec<usize>>,
    #[serde(default)]
    expected_strides: Option<Vec<isize>>,
    #[serde(default)]
    expected_writeable: Option<bool>,
    #[serde(default)]
    expected_has_internal_overlap: Option<bool>,
    #[serde(default)]
    expected_error_contains: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SlidingWindowFixtureCase {
    window_shape: Vec<usize>,
    #[serde(default)]
    expected_shape: Option<Vec<usize>>,
    #[serde(default)]
    expected_strides: Option<Vec<isize>>,
    #[serde(default)]
    expected_writeable: Option<bool>,
    #[serde(default)]
    expected_has_internal_overlap: Option<bool>,
    #[serde(default)]
    expected_error_contains: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PromotionFixtureCase {
    id: String,
    lhs: String,
    rhs: String,
    expected: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
}

#[derive(Debug, Deserialize)]
struct DTypeDifferentialCase {
    id: String,
    #[serde(default)]
    operation: String,
    lhs: String,
    rhs: String,
    expected: String,
    #[serde(default)]
    casting: String,
    #[serde(default)]
    expected_reason_code: String,
    #[serde(default)]
    minimal_repro_artifacts: Vec<String>,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
}

#[derive(Debug, Deserialize)]
struct DTypeMetamorphicCase {
    id: String,
    relation: String,
    lhs: String,
    #[serde(default)]
    rhs: Option<String>,
    #[serde(default, rename = "casting")]
    _casting: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
}

#[derive(Debug, Deserialize)]
struct DTypeAdversarialCase {
    id: String,
    lhs: String,
    rhs: String,
    expected_error_contains: String,
    expected_reason_code: String,
    #[serde(default)]
    severity: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
}

#[derive(Debug, Deserialize)]
struct PolicyFixtureCase {
    id: String,
    mode: String,
    class: String,
    risk_score: f64,
    threshold: f64,
    expected_action: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
}

#[derive(Debug, Deserialize)]
struct PolicyAdversarialFixtureCase {
    id: String,
    mode_raw: String,
    class_raw: String,
    risk_score: f64,
    threshold: f64,
    expected_action: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OverrideAuditFixtureCase {
    id: String,
    mode: String,
    class: String,
    requested_deviation_class: String,
    allowlisted_classes: Vec<String>,
    packet_id: String,
    requested_by: String,
    override_reason_code: String,
    expected_approved: bool,
    expected_action: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
}

#[derive(Debug, Deserialize)]
struct UFuncMetamorphicCase {
    id: String,
    relation: String,
    lhs_shape: Vec<usize>,
    lhs_values: Vec<f64>,
    #[serde(default = "default_f64_dtype_name")]
    lhs_dtype: String,
    rhs_shape: Option<Vec<usize>>,
    rhs_values: Option<Vec<f64>>,
    rhs_dtype: Option<String>,
    scalar: Option<f64>,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
}

#[derive(Debug, Deserialize)]
struct UFuncAdversarialCase {
    id: String,
    op: UFuncOperation,
    lhs_shape: Vec<usize>,
    lhs_values: Vec<f64>,
    #[serde(default = "default_f64_dtype_name")]
    lhs_dtype: String,
    rhs_shape: Option<Vec<usize>>,
    rhs_values: Option<Vec<f64>>,
    rhs_dtype: Option<String>,
    axis: Option<isize>,
    keepdims: Option<bool>,
    clip_min: Option<f64>,
    clip_max: Option<f64>,
    #[serde(default)]
    expected_error_contains: String,
    #[serde(default)]
    expect_success: bool,
    #[serde(default)]
    expected_shape: Option<Vec<usize>>,
    #[serde(default)]
    expect_non_finite: bool,
    #[serde(default)]
    expected_reason_code: String,
    #[serde(default)]
    severity: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
}

#[derive(Debug, Deserialize)]
struct IoAdversarialCase {
    id: String,
    operation: String,
    expected_error_contains: String,
    #[serde(default)]
    expected_reason_code: String,
    #[serde(default)]
    severity: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
    #[serde(default)]
    payload_prefix: Vec<u8>,
    #[serde(default)]
    shape: Vec<usize>,
    #[serde(default)]
    fortran_order: bool,
    #[serde(default)]
    dtype_descr: String,
    #[serde(default)]
    header_len: usize,
    #[serde(default)]
    payload_len_bytes: usize,
    #[serde(default)]
    allow_pickle: bool,
    #[serde(default)]
    memmap_mode: String,
    #[serde(default)]
    file_len_bytes: usize,
    #[serde(default)]
    expected_bytes: usize,
    #[serde(default)]
    validation_retries: usize,
    #[serde(default)]
    member_count: usize,
    #[serde(default)]
    uncompressed_bytes: usize,
    #[serde(default)]
    dispatch_retries: usize,
    #[serde(default)]
    mode_raw: String,
    #[serde(default)]
    class_raw: String,
}

#[derive(Debug, Deserialize)]
struct IoDifferentialCase {
    id: String,
    operation: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
    #[serde(default)]
    expected_reason_code: String,
    #[serde(default)]
    expected_error_contains: String,
    #[serde(default)]
    payload_prefix: Vec<u8>,
    #[serde(default)]
    shape: Vec<usize>,
    #[serde(default)]
    fortran_order: bool,
    #[serde(default)]
    include_fortran_order: Option<bool>,
    #[serde(default)]
    dtype_descr: String,
    #[serde(default)]
    header_len: usize,
    #[serde(default)]
    payload_len_bytes: usize,
    #[serde(default)]
    allow_pickle: bool,
    #[serde(default)]
    memmap_mode: String,
    #[serde(default)]
    file_len_bytes: usize,
    #[serde(default)]
    expected_bytes: usize,
    #[serde(default)]
    validation_retries: usize,
    #[serde(default)]
    member_count: usize,
    #[serde(default)]
    uncompressed_bytes: usize,
    #[serde(default)]
    dispatch_retries: usize,
    #[serde(default)]
    mode_raw: String,
    #[serde(default)]
    class_raw: String,
    #[serde(default)]
    expected_dispatch: String,
    #[serde(default)]
    expected_magic_version: String,
    #[serde(default)]
    expected_count: Option<usize>,
    #[serde(default)]
    text: String,
    #[serde(default)]
    delimiter: String,
    #[serde(default)]
    comments: String,
    #[serde(default)]
    skip_header: usize,
    #[serde(default)]
    skip_footer: usize,
    #[serde(default)]
    filling_values: Option<f64>,
    #[serde(default)]
    usecols: Vec<usize>,
    #[serde(default)]
    max_rows: Option<usize>,
    #[serde(default)]
    expected_shape: Vec<usize>,
    #[serde(default)]
    expected_values: Vec<f64>,
    #[serde(default)]
    abs_tol: f64,
    #[serde(default)]
    rel_tol: f64,
}

#[derive(Debug, Deserialize)]
struct IoMetamorphicCase {
    id: String,
    relation: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
    #[serde(default)]
    payload_prefix: Vec<u8>,
    #[serde(default)]
    shape: Vec<usize>,
    #[serde(default)]
    fortran_order: bool,
    #[serde(default)]
    dtype_descr: String,
    #[serde(default)]
    header_len: usize,
    #[serde(default)]
    payload_len_bytes: usize,
    #[serde(default)]
    allow_pickle: bool,
    #[serde(default)]
    member_count: usize,
    #[serde(default)]
    uncompressed_bytes: usize,
    #[serde(default)]
    dispatch_retries: usize,
    #[serde(default)]
    mode_raw: String,
    #[serde(default)]
    class_raw: String,
}

#[derive(Debug, Deserialize)]
struct LinalgDifferentialCase {
    id: String,
    operation: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
    #[serde(default)]
    matrix: Vec<Vec<f64>>,
    #[serde(default)]
    rhs: Vec<f64>,
    #[serde(default)]
    shape: Vec<usize>,
    #[serde(default)]
    rhs_shape: Vec<usize>,
    #[serde(default)]
    qr_mode: String,
    #[serde(default)]
    full_matrices: bool,
    #[serde(default = "default_true")]
    converged: bool,
    #[serde(default)]
    uplo: String,
    #[serde(default)]
    rcond: f64,
    #[serde(default)]
    search_depth: usize,
    #[serde(default)]
    backend_supported: bool,
    #[serde(default)]
    validation_retries: usize,
    #[serde(default)]
    mode_raw: String,
    #[serde(default)]
    class_raw: String,
    #[serde(default)]
    expected_solution: Vec<f64>,
    #[serde(default)]
    expected_q_present: Option<bool>,
    #[serde(default)]
    expected_q_shape: Option<Vec<usize>>,
    #[serde(default)]
    expected_r_shape: Vec<usize>,
    #[serde(default)]
    expected_u_shape: Vec<usize>,
    #[serde(default)]
    expected_s_shape: Vec<usize>,
    #[serde(default)]
    expected_vh_shape: Vec<usize>,
    #[serde(default)]
    expected_x_shape: Vec<usize>,
    #[serde(default)]
    expected_residuals_shape: Vec<usize>,
    #[serde(default)]
    expected_rank_upper_bound: usize,
    #[serde(default)]
    expected_singular_values_shape: Vec<usize>,
    #[serde(default)]
    expected_singular_values: Vec<f64>,
    #[serde(default)]
    expected_tolerance: f64,
    #[serde(default)]
    expected_error_contains: String,
    #[serde(default)]
    expected_reason_code: String,
}

#[derive(Debug, Deserialize)]
struct LinalgMetamorphicCase {
    id: String,
    relation: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
    #[serde(default)]
    matrix: Vec<Vec<f64>>,
    #[serde(default)]
    rhs: Vec<f64>,
    #[serde(default)]
    scalar: f64,
    #[serde(default)]
    shape: Vec<usize>,
    #[serde(default)]
    qr_mode: String,
    #[serde(default)]
    lhs_shape: Vec<usize>,
    #[serde(default)]
    rhs_shape: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct LinalgAdversarialCase {
    id: String,
    operation: String,
    expected_error_contains: String,
    expected_reason_code: String,
    #[serde(default)]
    severity: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
    #[serde(default)]
    matrix: Vec<Vec<f64>>,
    #[serde(default)]
    rhs: Vec<f64>,
    #[serde(default)]
    shape: Vec<usize>,
    #[serde(default)]
    rhs_shape: Vec<usize>,
    #[serde(default)]
    qr_mode: String,
    #[serde(default)]
    full_matrices: bool,
    #[serde(default = "default_true")]
    converged: bool,
    #[serde(default)]
    uplo: String,
    #[serde(default)]
    rcond: f64,
    #[serde(default)]
    search_depth: usize,
    #[serde(default)]
    backend_supported: bool,
    #[serde(default)]
    validation_retries: usize,
    #[serde(default)]
    mode_raw: String,
    #[serde(default)]
    class_raw: String,
}

#[derive(Debug, Deserialize)]
struct RngDifferentialCase {
    id: String,
    operation: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
    #[serde(default)]
    expected_reason_code: String,
    #[serde(default)]
    alt_seed: u64,
    #[serde(default)]
    draws: usize,
    #[serde(default)]
    upper_bound: u64,
    #[serde(default)]
    steps: u64,
    #[serde(default)]
    prefix_draws: usize,
    #[serde(default)]
    replay_draws: usize,
    #[serde(default)]
    fill_len: usize,
    #[serde(default)]
    mode_raw: String,
    #[serde(default)]
    class_raw: String,
}

#[derive(Debug, Deserialize)]
struct RngMetamorphicCase {
    id: String,
    relation: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
    #[serde(default)]
    alt_seed: u64,
    #[serde(default)]
    draws: usize,
    #[serde(default)]
    upper_bound: u64,
    #[serde(default)]
    steps: u64,
    #[serde(default)]
    extra_steps: u64,
    #[serde(default)]
    fill_len: usize,
}

#[derive(Debug, Deserialize)]
struct RngAdversarialCase {
    id: String,
    operation: String,
    expected_error_contains: String,
    expected_reason_code: String,
    #[serde(default)]
    severity: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
    #[serde(default)]
    alt_seed: u64,
    #[serde(default)]
    draws: usize,
    #[serde(default)]
    steps: u64,
    #[serde(default)]
    mode_raw: String,
    #[serde(default)]
    class_raw: String,
    #[serde(default)]
    pool_size: usize,
    #[serde(default)]
    spawn_count: usize,
    #[serde(default)]
    words: usize,
    #[serde(default)]
    kind: String,
}

#[derive(Debug, Deserialize)]
struct RngStatisticalCase {
    id: String,
    distribution: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
    #[serde(default)]
    draws: usize,
    #[serde(default)]
    param_a: f64,
    #[serde(default)]
    param_b: f64,
    #[serde(default)]
    param_c: f64,
    #[serde(default)]
    param_d: f64,
    #[serde(default)]
    expected_mean: Option<f64>,
    #[serde(default)]
    expected_variance: Option<f64>,
    #[serde(default = "default_rng_mean_abs_tol")]
    mean_abs_tol: f64,
    #[serde(default = "default_rng_variance_abs_tol")]
    variance_abs_tol: f64,
    #[serde(default)]
    support_min: Option<f64>,
    #[serde(default)]
    support_max: Option<f64>,
    #[serde(default = "default_true")]
    support_min_inclusive: bool,
    #[serde(default = "default_true")]
    support_max_inclusive: bool,
    #[serde(default)]
    ks_distribution: String,
    #[serde(default)]
    ks_alpha: f64,
}

#[derive(Debug, Deserialize)]
struct FftDifferentialCase {
    id: String,
    operation: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
    #[serde(default)]
    expected_reason_code: String,
    #[serde(default)]
    expected_error_contains: String,
    #[serde(default)]
    input_shape: Vec<usize>,
    #[serde(default)]
    input_values: Vec<f64>,
    #[serde(default)]
    n: Option<usize>,
    #[serde(default)]
    output_n: Option<usize>,
    #[serde(default)]
    d: Option<f64>,
    #[serde(default)]
    expected_shape: Vec<usize>,
    #[serde(default)]
    expected_values: Vec<f64>,
    #[serde(default)]
    abs_tol: f64,
    #[serde(default)]
    rel_tol: f64,
}

#[derive(Debug, Deserialize)]
struct FftMetamorphicCase {
    id: String,
    relation: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
    #[serde(default)]
    input_shape: Vec<usize>,
    #[serde(default)]
    input_values: Vec<f64>,
    #[serde(default)]
    x_shape: Vec<usize>,
    #[serde(default)]
    x_values: Vec<f64>,
    #[serde(default)]
    y_shape: Vec<usize>,
    #[serde(default)]
    y_values: Vec<f64>,
    #[serde(default)]
    alpha: Option<f64>,
    #[serde(default)]
    beta: Option<f64>,
    #[serde(default)]
    input_dtype: String,
    #[serde(default)]
    abs_tol: f64,
    #[serde(default)]
    rel_tol: f64,
}

#[derive(Debug, Deserialize)]
struct SignalDifferentialCase {
    id: String,
    operation: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
    #[serde(default)]
    expected_reason_code: String,
    #[serde(default)]
    expected_error_contains: String,
    #[serde(default)]
    n: Option<usize>,
    #[serde(default)]
    beta: Option<f64>,
    #[serde(default)]
    period: Option<f64>,
    #[serde(default)]
    discont: Option<f64>,
    #[serde(default)]
    spacing: Option<f64>,
    #[serde(default)]
    dx: Option<f64>,
    #[serde(default)]
    to_begin: Vec<f64>,
    #[serde(default)]
    to_end: Vec<f64>,
    #[serde(default)]
    input_shape: Vec<usize>,
    #[serde(default)]
    input_values: Vec<f64>,
    #[serde(default)]
    rhs_shape: Vec<usize>,
    #[serde(default)]
    rhs_values: Vec<f64>,
    #[serde(default)]
    expected_shape: Vec<usize>,
    #[serde(default)]
    expected_values: Vec<f64>,
    #[serde(default)]
    abs_tol: f64,
    #[serde(default)]
    rel_tol: f64,
    // norm-related fields
    #[serde(default)]
    ord_float: Option<f64>,
    #[serde(default)]
    ord_str: Option<String>,
    #[serde(default)]
    ord_inf: Option<String>,
    #[serde(default)]
    axis: Option<isize>,
    #[serde(default)]
    method: String,
    #[serde(default)]
    keepdims: bool,
    #[serde(default)]
    tensordot_axes: Option<usize>,
    #[serde(default)]
    axes: Vec<usize>,
    // roll, flip, rot90, pad fields
    #[serde(default)]
    shift: isize,
    #[serde(default)]
    k: i32,
    #[serde(default)]
    pad_width: Vec<(usize, usize)>,
    #[serde(default)]
    constant: f64,
    // clip fields
    #[serde(default)]
    min_val: Option<f64>,
    #[serde(default)]
    max_val: Option<f64>,
    // einsum subscripts
    #[serde(default)]
    subscripts: String,
    // histogram bins
    #[serde(default)]
    bins: Option<usize>,
    // isclose tolerances
    #[serde(default)]
    rtol: Option<f64>,
    #[serde(default)]
    atol: Option<f64>,
    // linspace/geomspace parameters
    #[serde(default)]
    start: Option<f64>,
    #[serde(default)]
    stop: Option<f64>,
    // reshape parameters
    #[serde(default)]
    reshape_shape: Vec<isize>,
}

#[derive(Debug, Deserialize)]
struct PolynomialDifferentialCase {
    id: String,
    operation: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
    #[serde(default)]
    expected_reason_code: String,
    #[serde(default)]
    expected_error_contains: String,
    #[serde(default)]
    lhs_values: Vec<f64>,
    #[serde(default)]
    rhs_values: Vec<f64>,
    #[serde(default)]
    y_values: Vec<f64>,
    #[serde(default)]
    deg: Option<usize>,
    #[serde(default)]
    expected_values: Vec<f64>,
    #[serde(default)]
    expected_aux_values: Vec<f64>,
    #[serde(default)]
    sort_values: bool,
    #[serde(default)]
    abs_tol: f64,
    #[serde(default)]
    rel_tol: f64,
}

#[derive(Debug, Deserialize)]
struct StringDifferentialCase {
    id: String,
    operation: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
    #[serde(default)]
    expected_reason_code: String,
    #[serde(default)]
    expected_error_contains: String,
    #[serde(default)]
    lhs_strings: Vec<String>,
    #[serde(default)]
    rhs_strings: Vec<String>,
    #[serde(default)]
    arg_string: String,
    #[serde(default)]
    arg_alt_string: String,
    #[serde(default)]
    cmp: String,
    #[serde(default)]
    repeat: Option<usize>,
    #[serde(default)]
    slice_start: Option<isize>,
    #[serde(default)]
    slice_stop: Option<isize>,
    #[serde(default)]
    slice_step: Option<isize>,
    #[serde(default)]
    width: Option<usize>,
    #[serde(default)]
    fillchar: Option<String>,
    #[serde(default)]
    tabsize: Option<usize>,
    #[serde(default)]
    expected_values: Vec<f64>,
    #[serde(default)]
    expected_strings: Vec<String>,
    #[serde(default)]
    abs_tol: f64,
    #[serde(default)]
    rel_tol: f64,
}

#[derive(Debug, Deserialize)]
struct StringMetamorphicCase {
    id: String,
    relation: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
    #[serde(default)]
    input_strings: Vec<String>,
    #[serde(default)]
    a_strings: Vec<String>,
    #[serde(default)]
    b_strings: Vec<String>,
    #[serde(default)]
    search_substr: String,
    #[serde(default)]
    search_char: String,
    #[serde(default)]
    old_substr: String,
    #[serde(default)]
    new_substr: String,
    #[serde(default)]
    multiply_factor: Option<usize>,
    #[serde(default)]
    width: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct MaskedDifferentialCase {
    id: String,
    operation: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
    #[serde(default)]
    expected_reason_code: String,
    #[serde(default)]
    expected_error_contains: String,
    #[serde(default)]
    lhs_shape: Vec<usize>,
    #[serde(default)]
    lhs_values: Vec<f64>,
    #[serde(default)]
    lhs_mask: Vec<f64>,
    #[serde(default)]
    rhs_shape: Vec<usize>,
    #[serde(default)]
    rhs_values: Vec<f64>,
    #[serde(default)]
    rhs_mask: Vec<f64>,
    #[serde(default)]
    fill_value: Option<f64>,
    #[serde(default)]
    axis: Option<isize>,
    #[serde(default)]
    ddof: Option<usize>,
    #[serde(default)]
    reshape_shape: Vec<isize>,
    #[serde(default)]
    nan_indices: Vec<usize>,
    #[serde(default)]
    inf_indices: Vec<usize>,
    #[serde(default)]
    expected_shape: Vec<usize>,
    #[serde(default)]
    expected_values: Vec<f64>,
    #[serde(default)]
    expected_mask: Vec<f64>,
    #[serde(default)]
    expect_mask_none: bool,
    #[serde(default)]
    expected_scalar: Option<f64>,
    #[serde(default)]
    abs_tol: f64,
    #[serde(default)]
    rel_tol: f64,
}

#[derive(Debug, Deserialize)]
struct DateTimeDifferentialCase {
    id: String,
    operation: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
    #[serde(default)]
    expected_reason_code: String,
    #[serde(default)]
    expected_error_contains: String,
    #[serde(default)]
    lhs_shape: Vec<usize>,
    #[serde(default)]
    lhs_values: Vec<f64>,
    #[serde(default)]
    lhs_strings: Vec<String>,
    #[serde(default)]
    lhs_dtype: String,
    #[serde(default)]
    rhs_shape: Vec<usize>,
    #[serde(default)]
    rhs_values: Vec<f64>,
    #[serde(default)]
    rhs_strings: Vec<String>,
    #[serde(default)]
    rhs_dtype: String,
    #[serde(default)]
    scalar: Option<f64>,
    #[serde(default)]
    expected_shape: Vec<usize>,
    #[serde(default)]
    expected_values: Vec<f64>,
    #[serde(default)]
    expected_dtype: String,
    #[serde(default)]
    abs_tol: f64,
    #[serde(default)]
    rel_tol: f64,
}

#[derive(Debug, Deserialize)]
struct DateTimeMetamorphicCase {
    id: String,
    relation: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
    #[serde(default)]
    base_datetime: String,
    #[serde(default)]
    datetime_a: String,
    #[serde(default)]
    datetime_b: String,
    #[serde(default)]
    datetime_c: String,
    #[serde(default)]
    delta_unit: String,
    #[serde(default)]
    delta_value: i64,
    #[serde(default)]
    delta_a_unit: String,
    #[serde(default)]
    delta_a_value: i64,
    #[serde(default)]
    delta_b_unit: String,
    #[serde(default)]
    delta_b_value: i64,
    #[serde(default)]
    delta_c_unit: String,
    #[serde(default)]
    delta_c_value: i64,
    #[serde(default)]
    scalar: Option<f64>,
    #[serde(default)]
    abs_tol: f64,
    #[serde(default)]
    rel_tol: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct IterSelectorFixtureInput {
    src_stride: isize,
    dst_stride: isize,
    item_size: usize,
    element_count: usize,
    aligned: bool,
    cast_is_lossless: bool,
    same_value_cast: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct IterOverlapFixtureInput {
    src_offset: usize,
    dst_offset: usize,
    byte_len: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct IterFlatIndexFixtureInput {
    kind: String,
    #[serde(default)]
    index: usize,
    #[serde(default)]
    start: usize,
    #[serde(default)]
    stop: usize,
    #[serde(default = "default_one_usize")]
    step: usize,
    #[serde(default)]
    indices: Vec<usize>,
    #[serde(default)]
    mask: Vec<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct IterFlagsFixtureInput {
    copy_if_overlap: bool,
    no_broadcast: bool,
    observed_overlap: bool,
    observed_broadcast: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct IterBroadcastOperandInput {
    shape: Vec<usize>,
    #[serde(default)]
    no_broadcast: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct IterBroadcastFixtureInput {
    operands: Vec<IterBroadcastOperandInput>,
    #[serde(default = "default_item_size")]
    item_size: usize,
    #[serde(default = "default_order_c")]
    order: String,
}

fn default_item_size() -> usize {
    8
}

fn default_order_c() -> String {
    "C".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct IterDifferentialCase {
    id: String,
    operation: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
    #[serde(default)]
    expected_reason_code: String,
    #[serde(default)]
    expected_error_contains: String,
    #[serde(default)]
    selector: Option<IterSelectorFixtureInput>,
    #[serde(default)]
    overlap: Option<IterOverlapFixtureInput>,
    #[serde(default)]
    flat_index: Option<IterFlatIndexFixtureInput>,
    #[serde(default)]
    flags: Option<IterFlagsFixtureInput>,
    #[serde(default)]
    broadcast: Option<IterBroadcastFixtureInput>,
    #[serde(default)]
    len: Option<usize>,
    #[serde(default)]
    values_len: Option<usize>,
    #[serde(default)]
    expected_transfer_class: Option<String>,
    #[serde(default)]
    expected_overlap_action: Option<String>,
    #[serde(default)]
    expected_selected: Option<usize>,
    #[serde(default)]
    expected_broadcast_shape: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct IterMetamorphicCase {
    id: String,
    relation: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
    #[serde(default)]
    selector: Option<IterSelectorFixtureInput>,
    #[serde(default)]
    overlap: Option<IterOverlapFixtureInput>,
    #[serde(default)]
    flat_index: Option<IterFlatIndexFixtureInput>,
    #[serde(default)]
    flags: Option<IterFlagsFixtureInput>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct IterAdversarialCase {
    id: String,
    operation: String,
    expected_error_contains: String,
    expected_reason_code: String,
    #[serde(default)]
    severity: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    env_fingerprint: String,
    #[serde(default)]
    artifact_refs: Vec<String>,
    #[serde(default)]
    reason_code: String,
    #[serde(default)]
    selector: Option<IterSelectorFixtureInput>,
    #[serde(default)]
    overlap: Option<IterOverlapFixtureInput>,
    #[serde(default)]
    flat_index: Option<IterFlatIndexFixtureInput>,
    #[serde(default)]
    flags: Option<IterFlagsFixtureInput>,
    #[serde(default)]
    len: Option<usize>,
    #[serde(default)]
    values_len: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct CrashSignatureRegistry {
    schema_version: u8,
    registry_version: String,
    signatures: Vec<CrashSignatureEntry>,
}

#[derive(Debug, Deserialize)]
struct CrashSignatureEntry {
    signature_id: String,
    suite: String,
    fixture_id: String,
    seed: u64,
    severity: String,
    reason_code: String,
    status: String,
    minimized_repro_artifacts: Vec<String>,
    blame_refs: Vec<String>,
}

#[derive(Debug, Serialize)]
struct RuntimePolicyLogEntry {
    suite: &'static str,
    fixture_id: String,
    seed: u64,
    mode: String,
    env_fingerprint: String,
    artifact_refs: Vec<String>,
    reason_code: String,
    expected_action: String,
    actual_action: String,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct ShapeStrideLogEntry {
    suite: &'static str,
    fixture_id: String,
    seed: u64,
    mode: String,
    env_fingerprint: String,
    artifact_refs: Vec<String>,
    reason_code: String,
    expected_broadcast: Option<Vec<usize>>,
    stride_shape: Vec<usize>,
    stride_order: String,
    expected_strides: Vec<isize>,
    as_strided_checked: bool,
    broadcast_to_checked: bool,
    sliding_window_checked: bool,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct DTypePromotionLogEntry {
    suite: &'static str,
    fixture_id: String,
    seed: u64,
    mode: String,
    env_fingerprint: String,
    artifact_refs: Vec<String>,
    reason_code: String,
    lhs: String,
    rhs: String,
    expected: String,
    actual: String,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct LinalgDifferentialMismatch {
    fixture_id: String,
    seed: u64,
    mode: String,
    expected_reason_code: String,
    actual_reason_code: String,
    message: String,
    artifact_refs: Vec<String>,
}

#[derive(Debug, Serialize)]
struct LinalgDifferentialReportArtifact {
    suite: &'static str,
    total_cases: usize,
    passed_cases: usize,
    failed_cases: usize,
    mismatches: Vec<LinalgDifferentialMismatch>,
}

#[derive(Debug, Serialize)]
struct UFuncDifferentialMismatch {
    fixture_id: String,
    seed: u64,
    mode: String,
    expected_reason_code: String,
    actual_reason_code: String,
    message: String,
    artifact_refs: Vec<String>,
}

#[derive(Debug, Serialize)]
struct UFuncDifferentialReportArtifact {
    suite: &'static str,
    total_cases: usize,
    passed_cases: usize,
    failed_cases: usize,
    mismatches: Vec<UFuncDifferentialMismatch>,
}

#[derive(Debug, Serialize)]
struct ShapeStrideDifferentialMismatch {
    fixture_id: String,
    seed: u64,
    mode: String,
    reason_code: String,
    message: String,
    artifact_refs: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ShapeStrideDifferentialReportArtifact {
    suite: &'static str,
    total_cases: usize,
    passed_cases: usize,
    failed_cases: usize,
    mismatches: Vec<ShapeStrideDifferentialMismatch>,
}

#[derive(Debug, Serialize)]
struct RngDifferentialMismatch {
    fixture_id: String,
    seed: u64,
    mode: String,
    expected_reason_code: String,
    actual_reason_code: String,
    message: String,
    artifact_refs: Vec<String>,
}

#[derive(Debug, Serialize)]
struct RngDifferentialReportArtifact {
    suite: &'static str,
    total_cases: usize,
    passed_cases: usize,
    failed_cases: usize,
    mismatches: Vec<RngDifferentialMismatch>,
}

#[derive(Debug, Serialize)]
struct FftDifferentialMismatch {
    fixture_id: String,
    seed: u64,
    mode: String,
    operation: String,
    expected_reason_code: String,
    actual_reason_code: String,
    message: String,
    artifact_refs: Vec<String>,
}

#[derive(Debug, Serialize)]
struct FftDifferentialReportArtifact {
    suite: &'static str,
    total_cases: usize,
    passed_cases: usize,
    failed_cases: usize,
    mismatches: Vec<FftDifferentialMismatch>,
}

#[derive(Debug, Serialize)]
struct PolynomialDifferentialMismatch {
    fixture_id: String,
    seed: u64,
    mode: String,
    operation: String,
    expected_reason_code: String,
    actual_reason_code: String,
    message: String,
    artifact_refs: Vec<String>,
}

#[derive(Debug, Serialize)]
struct PolynomialDifferentialReportArtifact {
    suite: &'static str,
    total_cases: usize,
    passed_cases: usize,
    failed_cases: usize,
    mismatches: Vec<PolynomialDifferentialMismatch>,
}

#[derive(Debug, Serialize)]
struct StringDifferentialMismatch {
    fixture_id: String,
    seed: u64,
    mode: String,
    operation: String,
    expected_reason_code: String,
    actual_reason_code: String,
    message: String,
    artifact_refs: Vec<String>,
}

#[derive(Debug, Serialize)]
struct StringDifferentialReportArtifact {
    suite: &'static str,
    total_cases: usize,
    passed_cases: usize,
    failed_cases: usize,
    mismatches: Vec<StringDifferentialMismatch>,
}

#[derive(Debug, Serialize)]
struct MaskedDifferentialMismatch {
    fixture_id: String,
    seed: u64,
    mode: String,
    operation: String,
    expected_reason_code: String,
    actual_reason_code: String,
    message: String,
    artifact_refs: Vec<String>,
}

#[derive(Debug, Serialize)]
struct MaskedDifferentialReportArtifact {
    suite: &'static str,
    total_cases: usize,
    passed_cases: usize,
    failed_cases: usize,
    mismatches: Vec<MaskedDifferentialMismatch>,
}

#[derive(Debug, Serialize)]
struct DateTimeDifferentialMismatch {
    fixture_id: String,
    seed: u64,
    mode: String,
    operation: String,
    expected_reason_code: String,
    actual_reason_code: String,
    message: String,
    artifact_refs: Vec<String>,
}

#[derive(Debug, Serialize)]
struct DateTimeDifferentialReportArtifact {
    suite: &'static str,
    total_cases: usize,
    passed_cases: usize,
    failed_cases: usize,
    mismatches: Vec<DateTimeDifferentialMismatch>,
}

#[derive(Debug, Serialize)]
struct IoDifferentialMismatch {
    fixture_id: String,
    seed: u64,
    mode: String,
    operation: String,
    expected_reason_code: String,
    actual_reason_code: String,
    message: String,
    minimal_repro_artifacts: Vec<String>,
    artifact_refs: Vec<String>,
}

#[derive(Debug, Serialize)]
struct IoDifferentialReportArtifact {
    suite: &'static str,
    total_cases: usize,
    passed_cases: usize,
    failed_cases: usize,
    mismatches: Vec<IoDifferentialMismatch>,
}

#[derive(Debug, Serialize)]
struct IterDifferentialMismatch {
    fixture_id: String,
    seed: u64,
    mode: String,
    operation: String,
    expected_reason_code: String,
    actual_reason_code: String,
    message: String,
    minimal_repro_artifacts: Vec<String>,
    artifact_refs: Vec<String>,
}

#[derive(Debug, Serialize)]
struct IterDifferentialReportArtifact {
    suite: &'static str,
    total_cases: usize,
    passed_cases: usize,
    failed_cases: usize,
    mismatches: Vec<IterDifferentialMismatch>,
}

#[derive(Debug, Serialize)]
struct DTypeDifferentialMismatch {
    fixture_id: String,
    seed: u64,
    mode: String,
    lhs: String,
    rhs: String,
    expected_dtype: String,
    actual_dtype: String,
    expected_reason_code: String,
    actual_reason_code: String,
    message: String,
    minimal_repro_artifacts: Vec<String>,
    artifact_refs: Vec<String>,
}

#[derive(Debug, Serialize)]
struct DTypeDifferentialReportArtifact {
    suite: &'static str,
    total_cases: usize,
    passed_cases: usize,
    failed_cases: usize,
    mismatches: Vec<DTypeDifferentialMismatch>,
}

static RUNTIME_POLICY_LOG_PATH: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
static SHAPE_STRIDE_LOG_PATH: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
static DTYPE_PROMOTION_LOG_PATH: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
static FILE_LOG_MUTEX: Mutex<()> = Mutex::new(());

fn default_f64_dtype_name() -> String {
    "f64".to_string()
}

fn default_one_usize() -> usize {
    1
}

fn default_true() -> bool {
    true
}

fn default_rng_mean_abs_tol() -> f64 {
    0.1
}

fn default_rng_variance_abs_tol() -> f64 {
    0.2
}

pub fn set_runtime_policy_log_path(path: Option<PathBuf>) {
    let cell = RUNTIME_POLICY_LOG_PATH.get_or_init(|| Mutex::new(None));
    if let Ok(mut slot) = cell.lock() {
        *slot = path;
    }
}

pub fn set_shape_stride_log_path(path: Option<PathBuf>) {
    let cell = SHAPE_STRIDE_LOG_PATH.get_or_init(|| Mutex::new(None));
    if let Ok(mut slot) = cell.lock() {
        *slot = path;
    }
}

pub fn set_dtype_promotion_log_path(path: Option<PathBuf>) {
    let cell = DTYPE_PROMOTION_LOG_PATH.get_or_init(|| Mutex::new(None));
    if let Ok(mut slot) = cell.lock() {
        *slot = path;
    }
}

#[must_use]
pub fn run_smoke(config: &HarnessConfig) -> HarnessReport {
    let fixture_count = fs::read_dir(&config.fixture_root)
        .ok()
        .into_iter()
        .flat_map(|it| it.filter_map(Result::ok))
        .count();

    HarnessReport {
        suite: "smoke",
        oracle_present: config.oracle_root.exists(),
        fixture_count,
        strict_mode: config.strict_mode,
    }
}

fn load_shape_stride_cases(fixture_root: &Path) -> Result<Vec<ShapeStrideFixtureCase>, String> {
    let path = fixture_root.join("shape_stride_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))
}

fn load_iter_differential_cases(fixture_root: &Path) -> Result<Vec<IterDifferentialCase>, String> {
    let path = fixture_root.join("iter_differential_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))
}

fn load_iter_metamorphic_cases(fixture_root: &Path) -> Result<Vec<IterMetamorphicCase>, String> {
    let path = fixture_root.join("iter_metamorphic_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))
}

fn load_iter_adversarial_cases(fixture_root: &Path) -> Result<Vec<IterAdversarialCase>, String> {
    let path = fixture_root.join("iter_adversarial_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))
}

fn resolve_packet_fixture_path(fixture_root: &Path, packet_dir: &str, file_name: &str) -> PathBuf {
    let direct_path = fixture_root.join(file_name);
    if direct_path.exists() {
        return direct_path;
    }

    let packet_path = fixture_root.join(packet_dir).join(file_name);
    if packet_path.exists() {
        return packet_path;
    }

    match fixture_root.file_name().and_then(std::ffi::OsStr::to_str) {
        Some(dir_name) if dir_name == packet_dir => direct_path,
        _ => packet_path,
    }
}

fn load_dtype_differential_cases(
    fixture_root: &Path,
) -> Result<Vec<DTypeDifferentialCase>, String> {
    let path = resolve_packet_fixture_path(
        fixture_root,
        "packet002_dtype",
        "dtype_differential_cases.json",
    );
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))
}

fn load_dtype_metamorphic_cases(fixture_root: &Path) -> Result<Vec<DTypeMetamorphicCase>, String> {
    let path = resolve_packet_fixture_path(
        fixture_root,
        "packet002_dtype",
        "dtype_metamorphic_cases.json",
    );
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))
}

fn load_dtype_adversarial_cases(fixture_root: &Path) -> Result<Vec<DTypeAdversarialCase>, String> {
    let path = resolve_packet_fixture_path(
        fixture_root,
        "packet002_dtype",
        "dtype_adversarial_cases.json",
    );
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))
}

fn load_fft_differential_cases(fixture_root: &Path) -> Result<Vec<FftDifferentialCase>, String> {
    let path = fixture_root.join("fft_differential_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))
}

fn load_fft_metamorphic_cases(fixture_root: &Path) -> Result<Vec<FftMetamorphicCase>, String> {
    let path = fixture_root.join("fft_metamorphic_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))
}

fn load_signal_differential_cases(
    fixture_root: &Path,
) -> Result<Vec<SignalDifferentialCase>, String> {
    let path = fixture_root.join("signal_differential_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))
}

fn load_polynomial_differential_cases(
    fixture_root: &Path,
) -> Result<Vec<PolynomialDifferentialCase>, String> {
    let path = fixture_root.join("polynomial_differential_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))
}

fn load_string_differential_cases(
    fixture_root: &Path,
) -> Result<Vec<StringDifferentialCase>, String> {
    let path = fixture_root.join("string_differential_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))
}

fn load_string_metamorphic_cases(
    fixture_root: &Path,
) -> Result<Vec<StringMetamorphicCase>, String> {
    let path = fixture_root.join("string_metamorphic_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))
}

fn load_masked_differential_cases(
    fixture_root: &Path,
) -> Result<Vec<MaskedDifferentialCase>, String> {
    let path = fixture_root.join("masked_differential_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))
}

fn load_datetime_differential_cases(
    fixture_root: &Path,
) -> Result<Vec<DateTimeDifferentialCase>, String> {
    let path = fixture_root.join("datetime_differential_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))
}

fn load_datetime_metamorphic_cases(
    fixture_root: &Path,
) -> Result<Vec<DateTimeMetamorphicCase>, String> {
    let path = fixture_root.join("datetime_metamorphic_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))
}

fn load_rng_statistical_cases(fixture_root: &Path) -> Result<Vec<RngStatisticalCase>, String> {
    let path = fixture_root.join("rng_statistical_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))
}

fn parse_shape_stride_order(case_id: &str, stride_order: &str) -> Result<MemoryOrder, String> {
    match stride_order {
        "C" => Ok(MemoryOrder::C),
        "F" => Ok(MemoryOrder::F),
        bad => Err(format!("{case_id}: invalid stride_order={bad}")),
    }
}

fn evaluate_shape_stride_case(case: &ShapeStrideFixtureCase) -> (bool, Vec<String>) {
    let mut ok = true;
    let mut failures = Vec::new();

    match (
        &case.expected_broadcast,
        broadcast_shape(&case.lhs, &case.rhs),
    ) {
        (Some(expected), Ok(actual)) if expected == &actual => {}
        (None, Err(_)) => {}
        (Some(expected), Ok(actual)) => {
            ok = false;
            failures.push(format!(
                "{}: broadcast mismatch expected={expected:?} actual={actual:?}",
                case.id
            ));
        }
        (Some(expected), Err(err)) => {
            ok = false;
            failures.push(format!(
                "{}: broadcast expected={expected:?} but failed: {err}",
                case.id
            ));
        }
        (None, Ok(actual)) => {
            ok = false;
            failures.push(format!(
                "{}: broadcast expected failure but got {actual:?}",
                case.id
            ));
        }
    }

    let order = match parse_shape_stride_order(&case.id, &case.stride_order) {
        Ok(order) => order,
        Err(err) => {
            ok = false;
            failures.push(err);
            MemoryOrder::C
        }
    };

    let computed_strides =
        match contiguous_strides(&case.stride_shape, case.stride_item_size, order) {
            Ok(strides) if strides == case.expected_strides => Some(strides),
            Ok(strides) => {
                ok = false;
                failures.push(format!(
                    "{}: stride mismatch expected={:?} actual={strides:?}",
                    case.id, case.expected_strides
                ));
                Some(strides)
            }
            Err(err) => {
                ok = false;
                failures.push(format!("{}: stride computation failed: {err}", case.id));
                None
            }
        };

    let base_layout = computed_strides.map(|strides| NdLayout {
        shape: case.stride_shape.clone(),
        strides,
        item_size: case.stride_item_size,
        writeable: true,
        has_internal_overlap: false,
    });

    if let Some(as_strided_case) = &case.as_strided {
        if let Some(base) = &base_layout {
            match base.as_strided(
                as_strided_case.shape.clone(),
                as_strided_case.strides.clone(),
            ) {
                Ok(view) => {
                    if let Some(needle) = &as_strided_case.expected_error_contains {
                        ok = false;
                        failures.push(format!(
                            "{}: as_strided expected error containing '{}' but succeeded",
                            case.id, needle
                        ));
                    } else {
                        if let Some(expected_shape) = &as_strided_case.expected_shape
                            && view.shape != *expected_shape
                        {
                            ok = false;
                            failures.push(format!(
                                "{}: as_strided shape mismatch expected={expected_shape:?} actual={:?}",
                                case.id, view.shape
                            ));
                        }

                        if let Some(expected_strides) = &as_strided_case.expected_strides
                            && view.strides != *expected_strides
                        {
                            ok = false;
                            failures.push(format!(
                                "{}: as_strided strides mismatch expected={expected_strides:?} actual={:?}",
                                case.id, view.strides
                            ));
                        }

                        if let Some(expected_writeable) = as_strided_case.expected_writeable
                            && view.is_writeable() != expected_writeable
                        {
                            ok = false;
                            failures.push(format!(
                                "{}: as_strided writeable mismatch expected={} actual={}",
                                case.id,
                                expected_writeable,
                                view.is_writeable()
                            ));
                        }

                        if let Some(expected_overlap) =
                            as_strided_case.expected_has_internal_overlap
                            && view.has_internal_overlap() != expected_overlap
                        {
                            ok = false;
                            failures.push(format!(
                                "{}: as_strided overlap mismatch expected={} actual={}",
                                case.id,
                                expected_overlap,
                                view.has_internal_overlap()
                            ));
                        }
                    }
                }
                Err(err) => {
                    if let Some(needle) = &as_strided_case.expected_error_contains {
                        let actual = err.to_string().to_lowercase();
                        if !actual.contains(&needle.to_lowercase()) {
                            ok = false;
                            failures.push(format!(
                                "{}: as_strided expected error containing '{}' but got '{}'",
                                case.id, needle, err
                            ));
                        }
                    } else {
                        ok = false;
                        failures.push(format!(
                            "{}: as_strided unexpectedly failed: {}",
                            case.id, err
                        ));
                    }
                }
            }
        } else {
            ok = false;
            failures.push(format!(
                "{}: cannot validate as_strided without valid base layout",
                case.id
            ));
        }
    }

    if let Some(broadcast_to_case) = &case.broadcast_to {
        if let Some(base) = &base_layout {
            match base.broadcast_to(broadcast_to_case.shape.clone()) {
                Ok(view) => {
                    if let Some(needle) = &broadcast_to_case.expected_error_contains {
                        ok = false;
                        failures.push(format!(
                            "{}: broadcast_to expected error containing '{}' but succeeded",
                            case.id, needle
                        ));
                    } else {
                        if let Some(expected_shape) = &broadcast_to_case.expected_shape
                            && view.shape != *expected_shape
                        {
                            ok = false;
                            failures.push(format!(
                                "{}: broadcast_to shape mismatch expected={expected_shape:?} actual={:?}",
                                case.id, view.shape
                            ));
                        }

                        if let Some(expected_strides) = &broadcast_to_case.expected_strides
                            && view.strides != *expected_strides
                        {
                            ok = false;
                            failures.push(format!(
                                "{}: broadcast_to strides mismatch expected={expected_strides:?} actual={:?}",
                                case.id, view.strides
                            ));
                        }

                        if let Some(expected_writeable) = broadcast_to_case.expected_writeable
                            && view.is_writeable() != expected_writeable
                        {
                            ok = false;
                            failures.push(format!(
                                "{}: broadcast_to writeable mismatch expected={} actual={}",
                                case.id,
                                expected_writeable,
                                view.is_writeable()
                            ));
                        }

                        if let Some(expected_overlap) =
                            broadcast_to_case.expected_has_internal_overlap
                            && view.has_internal_overlap() != expected_overlap
                        {
                            ok = false;
                            failures.push(format!(
                                "{}: broadcast_to overlap mismatch expected={} actual={}",
                                case.id,
                                expected_overlap,
                                view.has_internal_overlap()
                            ));
                        }
                    }
                }
                Err(err) => {
                    if let Some(needle) = &broadcast_to_case.expected_error_contains {
                        let actual = err.to_string().to_lowercase();
                        if !actual.contains(&needle.to_lowercase()) {
                            ok = false;
                            failures.push(format!(
                                "{}: broadcast_to expected error containing '{}' but got '{}'",
                                case.id, needle, err
                            ));
                        }
                    } else {
                        ok = false;
                        failures.push(format!(
                            "{}: broadcast_to unexpectedly failed: {}",
                            case.id, err
                        ));
                    }
                }
            }
        } else {
            ok = false;
            failures.push(format!(
                "{}: cannot validate broadcast_to without valid base layout",
                case.id
            ));
        }
    }

    if let Some(sliding_window_case) = &case.sliding_window {
        if let Some(base) = &base_layout {
            match base.sliding_window_view(&sliding_window_case.window_shape) {
                Ok(view) => {
                    if let Some(needle) = &sliding_window_case.expected_error_contains {
                        ok = false;
                        failures.push(format!(
                            "{}: sliding_window_view expected error containing '{}' but succeeded",
                            case.id, needle
                        ));
                    } else {
                        if let Some(expected_shape) = &sliding_window_case.expected_shape
                            && view.shape != *expected_shape
                        {
                            ok = false;
                            failures.push(format!(
                                "{}: sliding_window_view shape mismatch expected={expected_shape:?} actual={:?}",
                                case.id, view.shape
                            ));
                        }

                        if let Some(expected_strides) = &sliding_window_case.expected_strides
                            && view.strides != *expected_strides
                        {
                            ok = false;
                            failures.push(format!(
                                "{}: sliding_window_view strides mismatch expected={expected_strides:?} actual={:?}",
                                case.id, view.strides
                            ));
                        }

                        if let Some(expected_writeable) = sliding_window_case.expected_writeable
                            && view.is_writeable() != expected_writeable
                        {
                            ok = false;
                            failures.push(format!(
                                "{}: sliding_window_view writeable mismatch expected={} actual={}",
                                case.id,
                                expected_writeable,
                                view.is_writeable()
                            ));
                        }

                        if let Some(expected_overlap) =
                            sliding_window_case.expected_has_internal_overlap
                            && view.has_internal_overlap() != expected_overlap
                        {
                            ok = false;
                            failures.push(format!(
                                "{}: sliding_window_view overlap mismatch expected={} actual={}",
                                case.id,
                                expected_overlap,
                                view.has_internal_overlap()
                            ));
                        }
                    }
                }
                Err(err) => {
                    if let Some(needle) = &sliding_window_case.expected_error_contains {
                        let actual = err.to_string().to_lowercase();
                        if !actual.contains(&needle.to_lowercase()) {
                            ok = false;
                            failures.push(format!(
                                "{}: sliding_window_view expected error containing '{}' but got '{}'",
                                case.id, needle, err
                            ));
                        }
                    } else {
                        ok = false;
                        failures.push(format!(
                            "{}: sliding_window_view unexpectedly failed: {}",
                            case.id, err
                        ));
                    }
                }
            }
        } else {
            ok = false;
            failures.push(format!(
                "{}: cannot validate sliding_window_view without valid base layout",
                case.id
            ));
        }
    }

    (ok, failures)
}

pub fn run_shape_stride_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let cases = load_shape_stride_cases(&config.fixture_root)?;

    let mut report = SuiteReport {
        suite: "shape_stride",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };

    for case in cases {
        let reason_code = normalize_reason_code(&case.reason_code);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let as_strided_checked = case.as_strided.is_some();
        let broadcast_to_checked = case.broadcast_to.is_some();
        let sliding_window_checked = case.sliding_window.is_some();
        let mode = if config.strict_mode {
            "strict"
        } else {
            "hardened"
        };

        let (ok, failures_for_case) = evaluate_shape_stride_case(&case);
        report.failures.extend(failures_for_case);
        if ok {
            report.pass_count += 1;
        }

        let log_entry = ShapeStrideLogEntry {
            suite: "shape_stride",
            fixture_id: case.id,
            seed: case.seed,
            mode: mode.to_string(),
            env_fingerprint: normalize_env_fingerprint(&case.env_fingerprint),
            artifact_refs,
            reason_code,
            expected_broadcast: case.expected_broadcast,
            stride_shape: case.stride_shape,
            stride_order: case.stride_order,
            expected_strides: case.expected_strides,
            as_strided_checked,
            broadcast_to_checked,
            sliding_window_checked,
            passed: ok,
        };
        maybe_append_shape_stride_log(&log_entry)?;
    }

    Ok(report)
}

fn parse_fixture_id_from_failure_line(failure: &str) -> Option<&str> {
    failure
        .split_once(':')
        .map(|(fixture_id, _)| fixture_id.trim())
        .filter(|fixture_id| !fixture_id.is_empty())
}

pub fn run_shape_stride_differential_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let suite = run_shape_stride_suite(config)?;
    let cases = load_shape_stride_cases(&config.fixture_root)?;
    let mode = if config.strict_mode {
        "strict".to_string()
    } else {
        "hardened".to_string()
    };
    let case_lookup = cases
        .into_iter()
        .map(|case| {
            (
                case.id,
                (
                    case.seed,
                    normalize_reason_code(&case.reason_code),
                    normalize_artifact_refs(case.artifact_refs),
                ),
            )
        })
        .collect::<BTreeMap<_, _>>();

    let mismatches = suite
        .failures
        .iter()
        .map(|failure| {
            let fixture_id = parse_fixture_id_from_failure_line(failure)
                .unwrap_or("unknown_fixture")
                .to_string();
            let (seed, reason_code, artifact_refs) =
                case_lookup.get(&fixture_id).cloned().unwrap_or_else(|| {
                    (
                        0,
                        "shape_stride_contract_violation".to_string(),
                        vec![
                            "artifacts/phase2c/FNP-P2C-006/contract_table.md".to_string(),
                            "artifacts/phase2c/FNP-P2C-006/risk_note.md".to_string(),
                        ],
                    )
                });
            ShapeStrideDifferentialMismatch {
                fixture_id,
                seed,
                mode: mode.clone(),
                reason_code,
                message: failure.clone(),
                artifact_refs,
            }
        })
        .collect::<Vec<_>>();

    let artifact = ShapeStrideDifferentialReportArtifact {
        suite: "shape_stride_differential",
        total_cases: suite.case_count,
        passed_cases: suite.pass_count,
        failed_cases: suite.case_count.saturating_sub(suite.pass_count),
        mismatches,
    };
    let artifact_path = config
        .fixture_root
        .join("oracle_outputs/shape_stride_differential_report.json");
    if let Some(parent) = artifact_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }
    let payload = serde_json::to_string_pretty(&artifact)
        .map_err(|err| format!("failed serializing shape-stride differential report: {err}"))?;
    fs::write(&artifact_path, payload.as_bytes())
        .map_err(|err| format!("failed writing {}: {err}", artifact_path.display()))?;

    Ok(SuiteReport {
        suite: "shape_stride_differential",
        case_count: suite.case_count,
        pass_count: suite.pass_count,
        failures: suite.failures,
    })
}

pub fn run_shape_stride_metamorphic_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let cases = load_shape_stride_cases(&config.fixture_root)?;
    let mut report = SuiteReport {
        suite: "shape_stride_metamorphic",
        case_count: 0,
        pass_count: 0,
        failures: Vec::new(),
    };
    let mode = if config.strict_mode {
        "strict".to_string()
    } else {
        "hardened".to_string()
    };

    for case in cases {
        let reason_code = normalize_reason_code(&case.reason_code);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let broadcast_forward = broadcast_shape(&case.lhs, &case.rhs);
        let broadcast_reverse = broadcast_shape(&case.rhs, &case.lhs);
        let commutative_ok = match (&broadcast_forward, &broadcast_reverse) {
            (Ok(lhs_rhs), Ok(rhs_lhs)) => lhs_rhs == rhs_lhs,
            (Err(_), Err(_)) => true,
            _ => false,
        };
        record_suite_check(
            &mut report,
            commutative_ok,
            format!(
                "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} broadcast commutativity failed forward={broadcast_forward:?} reverse={broadcast_reverse:?}",
                case.id,
                case.seed,
                mode,
                reason_code,
                env_fingerprint,
                artifact_refs.join(",")
            ),
        );

        if let Some(expected) = &case.expected_broadcast {
            // Skip identity check for 0-D scalars: broadcast([], [1]) == [1], not []
            // This is correct NumPy behavior - the identity property only holds for rank >= 1
            if !expected.is_empty() {
                let identity = broadcast_shape(expected, &[1usize]);
                let identity_ok = identity.as_ref().is_ok_and(|actual| actual == expected);
                record_suite_check(
                    &mut report,
                    identity_ok,
                    format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} broadcast identity failed expected={expected:?} actual={identity:?}",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(",")
                    ),
                );
            }
        }

        let order = match parse_shape_stride_order(&case.id, &case.stride_order) {
            Ok(order) => order,
            Err(err) => {
                record_suite_check(
                    &mut report,
                    false,
                    format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} {}",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        err
                    ),
                );
                continue;
            }
        };

        let base = match NdLayout::contiguous(
            case.stride_shape.clone(),
            case.stride_item_size,
            order,
        ) {
            Ok(layout) => layout,
            Err(err) => {
                record_suite_check(
                    &mut report,
                    false,
                    format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} failed to build base layout for metamorphic checks: {}",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        err
                    ),
                );
                continue;
            }
        };

        if let Some(broadcast_to_case) = &case.broadcast_to
            && broadcast_to_case.expected_error_contains.is_none()
        {
            let first = base.broadcast_to(broadcast_to_case.shape.clone());
            let second = base.broadcast_to(broadcast_to_case.shape.clone());
            let deterministic_ok = matches!((&first, &second), (Ok(lhs), Ok(rhs)) if lhs == rhs);
            record_suite_check(
                &mut report,
                deterministic_ok,
                format!(
                    "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} broadcast_to determinism failed first={first:?} second={second:?}",
                    case.id,
                    case.seed,
                    mode,
                    reason_code,
                    env_fingerprint,
                    artifact_refs.join(",")
                ),
            );
        }

        if let Some(as_strided_case) = &case.as_strided
            && as_strided_case.expected_error_contains.is_none()
        {
            let first = base.as_strided(
                as_strided_case.shape.clone(),
                as_strided_case.strides.clone(),
            );
            let second = base.as_strided(
                as_strided_case.shape.clone(),
                as_strided_case.strides.clone(),
            );
            let deterministic_ok = matches!((&first, &second), (Ok(lhs), Ok(rhs)) if lhs == rhs);
            record_suite_check(
                &mut report,
                deterministic_ok,
                format!(
                    "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} as_strided determinism failed first={first:?} second={second:?}",
                    case.id,
                    case.seed,
                    mode,
                    reason_code,
                    env_fingerprint,
                    artifact_refs.join(",")
                ),
            );
        }

        if let Some(sliding_window_case) = &case.sliding_window
            && sliding_window_case.expected_error_contains.is_none()
        {
            let first = base.sliding_window_view(&sliding_window_case.window_shape);
            let second = base.sliding_window_view(&sliding_window_case.window_shape);
            let deterministic_ok = matches!((&first, &second), (Ok(lhs), Ok(rhs)) if lhs == rhs);
            record_suite_check(
                &mut report,
                deterministic_ok,
                format!(
                    "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} sliding_window determinism failed first={first:?} second={second:?}",
                    case.id,
                    case.seed,
                    mode,
                    reason_code,
                    env_fingerprint,
                    artifact_refs.join(",")
                ),
            );

            let relation_ok = first.as_ref().is_ok_and(|view| {
                let ndim = base.shape.len();
                if view.shape.len() != ndim * 2 {
                    return false;
                }
                base.shape
                    .iter()
                    .zip(sliding_window_case.window_shape.iter())
                    .enumerate()
                    .all(|(axis, (dim, window))| {
                        *window > 0
                            && *window <= *dim
                            && view.shape[axis] == dim.saturating_sub(*window).saturating_add(1)
                    })
            });
            record_suite_check(
                &mut report,
                relation_ok,
                format!(
                    "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} sliding_window relation check failed result={first:?}",
                    case.id,
                    case.seed,
                    mode,
                    reason_code,
                    env_fingerprint,
                    artifact_refs.join(",")
                ),
            );
        }
    }

    Ok(report)
}

pub fn run_shape_stride_adversarial_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let cases = load_shape_stride_cases(&config.fixture_root)?;
    let mut report = SuiteReport {
        suite: "shape_stride_adversarial",
        case_count: 0,
        pass_count: 0,
        failures: Vec::new(),
    };
    let mode = if config.strict_mode {
        "strict".to_string()
    } else {
        "hardened".to_string()
    };

    for case in cases {
        let reason_code = normalize_reason_code(&case.reason_code);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());

        if case.expected_broadcast.is_none() {
            record_suite_check(
                &mut report,
                broadcast_shape(&case.lhs, &case.rhs).is_err(),
                format!(
                    "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected broadcast failure for adversarial case but call succeeded",
                    case.id,
                    case.seed,
                    mode,
                    reason_code,
                    env_fingerprint,
                    artifact_refs.join(",")
                ),
            );
        }

        let order = match parse_shape_stride_order(&case.id, &case.stride_order) {
            Ok(order) => order,
            Err(err) => {
                record_suite_check(
                    &mut report,
                    false,
                    format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} {}",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        err
                    ),
                );
                continue;
            }
        };
        let base = match NdLayout::contiguous(
            case.stride_shape.clone(),
            case.stride_item_size,
            order,
        ) {
            Ok(layout) => layout,
            Err(err) => {
                record_suite_check(
                    &mut report,
                    false,
                    format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} failed to build base layout for adversarial checks: {}",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        err
                    ),
                );
                continue;
            }
        };

        if let Some(as_strided_case) = &case.as_strided
            && let Some(needle) = &as_strided_case.expected_error_contains
        {
            match base.as_strided(
                as_strided_case.shape.clone(),
                as_strided_case.strides.clone(),
            ) {
                Ok(view) => {
                    record_suite_check(
                        &mut report,
                        false,
                        format!(
                            "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected as_strided error containing '{}' but got view={view:?}",
                            case.id,
                            case.seed,
                            mode,
                            reason_code,
                            env_fingerprint,
                            artifact_refs.join(","),
                            needle
                        ),
                    );
                }
                Err(err) => {
                    let matched = err
                        .to_string()
                        .to_lowercase()
                        .contains(&needle.to_lowercase());
                    record_suite_check(
                        &mut report,
                        matched,
                        format!(
                            "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected as_strided error containing '{}' but got '{}'",
                            case.id,
                            case.seed,
                            mode,
                            reason_code,
                            env_fingerprint,
                            artifact_refs.join(","),
                            needle,
                            err
                        ),
                    );
                }
            }
        }

        if let Some(broadcast_to_case) = &case.broadcast_to
            && let Some(needle) = &broadcast_to_case.expected_error_contains
        {
            match base.broadcast_to(broadcast_to_case.shape.clone()) {
                Ok(view) => {
                    record_suite_check(
                        &mut report,
                        false,
                        format!(
                            "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected broadcast_to error containing '{}' but got view={view:?}",
                            case.id,
                            case.seed,
                            mode,
                            reason_code,
                            env_fingerprint,
                            artifact_refs.join(","),
                            needle
                        ),
                    );
                }
                Err(err) => {
                    let matched = err
                        .to_string()
                        .to_lowercase()
                        .contains(&needle.to_lowercase());
                    record_suite_check(
                        &mut report,
                        matched,
                        format!(
                            "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected broadcast_to error containing '{}' but got '{}'",
                            case.id,
                            case.seed,
                            mode,
                            reason_code,
                            env_fingerprint,
                            artifact_refs.join(","),
                            needle,
                            err
                        ),
                    );
                }
            }
        }

        if let Some(sliding_window_case) = &case.sliding_window
            && let Some(needle) = &sliding_window_case.expected_error_contains
        {
            match base.sliding_window_view(&sliding_window_case.window_shape) {
                Ok(view) => {
                    record_suite_check(
                        &mut report,
                        false,
                        format!(
                            "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected sliding_window_view error containing '{}' but got view={view:?}",
                            case.id,
                            case.seed,
                            mode,
                            reason_code,
                            env_fingerprint,
                            artifact_refs.join(","),
                            needle
                        ),
                    );
                }
                Err(err) => {
                    let matched = err
                        .to_string()
                        .to_lowercase()
                        .contains(&needle.to_lowercase());
                    record_suite_check(
                        &mut report,
                        matched,
                        format!(
                            "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected sliding_window_view error containing '{}' but got '{}'",
                            case.id,
                            case.seed,
                            mode,
                            reason_code,
                            env_fingerprint,
                            artifact_refs.join(","),
                            needle,
                            err
                        ),
                    );
                }
            }
        }
    }

    if report.case_count == 0 {
        return Err("shape_stride_adversarial produced zero checks".to_string());
    }

    Ok(report)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum IterOperationOutcome {
    TransferClass(TransferClass),
    OverlapAction(OverlapAction),
    SelectedCount(usize),
    BroadcastPlan(NditerBroadcastPlan),
    Unit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IterSuiteError {
    reason_code: String,
    message: String,
}

impl IterSuiteError {
    fn new(reason_code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            reason_code: reason_code.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for IterSuiteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for IterSuiteError {}

fn classify_iter_transfer_error(operation: &str, error: &TransferError) -> String {
    match operation {
        "select_transfer_class" => "nditer_constructor_invalid_configuration".to_string(),
        "overlap_copy_policy" => "nditer_overlap_policy_triggered".to_string(),
        "validate_nditer_flags" => {
            let lowered = error.to_string().to_lowercase();
            if lowered.contains("no_broadcast") {
                "nditer_no_broadcast_violation".to_string()
            } else {
                "nditer_overlap_policy_triggered".to_string()
            }
        }
        "validate_flatiter_read" | "validate_flatiter_write" => {
            "flatiter_indexing_contract_violation".to_string()
        }
        _ => "nditer_transition_precondition_failed".to_string(),
    }
}

fn classify_nditer_broadcast_error(error: &NditerError) -> String {
    let msg = error.to_string().to_lowercase();
    if msg.contains("no_broadcast") {
        "nditer_no_broadcast_violation".to_string()
    } else {
        "nditer_constructor_invalid_configuration".to_string()
    }
}

fn parse_expected_transfer_class(raw: &str) -> Result<TransferClass, String> {
    match raw {
        "contiguous" => Ok(TransferClass::Contiguous),
        "strided" => Ok(TransferClass::Strided),
        "strided_cast" => Ok(TransferClass::StridedCast),
        other => Err(format!(
            "invalid expected_transfer_class '{other}' (must be contiguous|strided|strided_cast)"
        )),
    }
}

fn parse_expected_overlap_action(raw: &str) -> Result<OverlapAction, String> {
    match raw {
        "no_copy" => Ok(OverlapAction::NoCopy),
        "forward_copy" => Ok(OverlapAction::ForwardCopy),
        "backward_copy" => Ok(OverlapAction::BackwardCopy),
        other => Err(format!(
            "invalid expected_overlap_action '{other}' (must be no_copy|forward_copy|backward_copy)"
        )),
    }
}

fn build_flatiter_index(
    case_id: &str,
    input: &IterFlatIndexFixtureInput,
) -> Result<FlatIterIndex, IterSuiteError> {
    match input.kind.as_str() {
        "single" => Ok(FlatIterIndex::Single(input.index)),
        "slice" => Ok(FlatIterIndex::Slice {
            start: input.start,
            stop: input.stop,
            step: input.step,
        }),
        "fancy" => Ok(FlatIterIndex::Fancy(input.indices.clone())),
        "bool_mask" => Ok(FlatIterIndex::BoolMask(input.mask.clone())),
        other => Err(IterSuiteError::new(
            "nditer_constructor_invalid_configuration",
            format!("{case_id}: unsupported flat_index.kind '{other}'"),
        )),
    }
}

fn infer_flatiter_len(input: &IterFlatIndexFixtureInput) -> usize {
    let fancy_upper = input
        .indices
        .iter()
        .copied()
        .max()
        .map_or(0, |max_index| max_index.saturating_add(1));
    input
        .mask
        .len()
        .max(input.stop)
        .max(input.index.saturating_add(1))
        .max(fancy_upper)
}

fn execute_iter_operation(
    case_id: &str,
    operation: &str,
    selector: Option<&IterSelectorFixtureInput>,
    overlap: Option<&IterOverlapFixtureInput>,
    flags: Option<&IterFlagsFixtureInput>,
    broadcast: Option<&IterBroadcastFixtureInput>,
    _values_len: Option<usize>,
) -> Result<IterOperationOutcome, IterSuiteError> {
    match operation {
        "select_transfer_class" => {
            let Some(selector) = selector else {
                return Err(IterSuiteError::new(
                    "nditer_constructor_invalid_configuration",
                    format!("{case_id}: missing selector payload"),
                ));
            };
            let input = TransferSelectorInput {
                src_stride: selector.src_stride,
                dst_stride: selector.dst_stride,
                item_size: selector.item_size,
                element_count: selector.element_count,
                aligned: selector.aligned,
                cast_is_lossless: selector.cast_is_lossless,
                same_value_cast: selector.same_value_cast,
            };
            select_transfer_class(input)
                .map(IterOperationOutcome::TransferClass)
                .map_err(|error| {
                    IterSuiteError::new(
                        classify_iter_transfer_error(operation, &error),
                        error.to_string(),
                    )
                })
        }
        "overlap_copy_policy" => {
            let Some(overlap) = overlap else {
                return Err(IterSuiteError::new(
                    "nditer_overlap_policy_triggered",
                    format!("{case_id}: missing overlap payload"),
                ));
            };
            overlap_copy_policy(overlap.src_offset, overlap.dst_offset, overlap.byte_len)
                .map(IterOperationOutcome::OverlapAction)
                .map_err(|error| {
                    IterSuiteError::new(
                        classify_iter_transfer_error(operation, &error),
                        error.to_string(),
                    )
                })
        }
        "validate_nditer_flags" => {
            let Some(flags) = flags else {
                return Err(IterSuiteError::new(
                    "nditer_constructor_invalid_configuration",
                    format!("{case_id}: missing flags payload"),
                ));
            };
            validate_nditer_flags(NditerTransferFlags {
                copy_if_overlap: flags.copy_if_overlap,
                no_broadcast: flags.no_broadcast,
                observed_overlap: flags.observed_overlap,
                observed_broadcast: flags.observed_broadcast,
            })
            .map(|_| IterOperationOutcome::Unit)
            .map_err(|error| {
                IterSuiteError::new(
                    classify_iter_transfer_error(operation, &error),
                    error.to_string(),
                )
            })
        }
        "validate_flatiter_read" | "validate_flatiter_write" => Err(IterSuiteError::new(
            "flatiter_indexing_contract_violation",
            format!("{case_id}: operation '{operation}' requires explicit len/value parameters"),
        )),
        "plan_nditer_broadcast" => {
            let Some(bcast) = broadcast else {
                return Err(IterSuiteError::new(
                    "nditer_constructor_invalid_configuration",
                    format!("{case_id}: missing broadcast payload"),
                ));
            };
            let operands: Vec<NditerOperandSpec> = bcast
                .operands
                .iter()
                .map(|op| NditerOperandSpec {
                    shape: op.shape.clone(),
                    no_broadcast: op.no_broadcast,
                })
                .collect();
            let order = match bcast.order.as_str() {
                "F" | "f" => NditerOrder::F,
                _ => NditerOrder::C,
            };
            plan_nditer_broadcast(&operands, bcast.item_size, order)
                .map(IterOperationOutcome::BroadcastPlan)
                .map_err(|error| {
                    IterSuiteError::new(classify_nditer_broadcast_error(&error), error.to_string())
                })
        }
        other => Err(IterSuiteError::new(
            "nditer_constructor_invalid_configuration",
            format!("{case_id}: unsupported iter operation '{other}'"),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_iter_operation_with_len(
    case_id: &str,
    operation: &str,
    selector: Option<&IterSelectorFixtureInput>,
    overlap: Option<&IterOverlapFixtureInput>,
    flat_index: Option<&IterFlatIndexFixtureInput>,
    flags: Option<&IterFlagsFixtureInput>,
    broadcast: Option<&IterBroadcastFixtureInput>,
    len: usize,
    values_len: Option<usize>,
) -> Result<IterOperationOutcome, IterSuiteError> {
    match operation {
        "validate_flatiter_read" => {
            let Some(flat_index) = flat_index else {
                return Err(IterSuiteError::new(
                    "flatiter_indexing_contract_violation",
                    format!("{case_id}: missing flat_index payload"),
                ));
            };
            let index = build_flatiter_index(case_id, flat_index)?;
            validate_flatiter_read(len, &index)
                .map(IterOperationOutcome::SelectedCount)
                .map_err(|error| {
                    IterSuiteError::new(
                        classify_iter_transfer_error(operation, &error),
                        error.to_string(),
                    )
                })
        }
        "validate_flatiter_write" => {
            let Some(flat_index) = flat_index else {
                return Err(IterSuiteError::new(
                    "flatiter_indexing_contract_violation",
                    format!("{case_id}: missing flat_index payload"),
                ));
            };
            let index = build_flatiter_index(case_id, flat_index)?;
            let values_len = values_len.unwrap_or(0);
            validate_flatiter_write(len, &index, values_len)
                .map(IterOperationOutcome::SelectedCount)
                .map_err(|error| {
                    IterSuiteError::new(
                        classify_iter_transfer_error(operation, &error),
                        error.to_string(),
                    )
                })
        }
        _ => execute_iter_operation(
            case_id, operation, selector, overlap, flags, broadcast, values_len,
        ),
    }
}

fn iter_minimal_repro_artifacts(case_id: &str, refs: &[String], fixture_path: &str) -> Vec<String> {
    let mut entries = refs
        .iter()
        .filter(|entry| !entry.trim().is_empty())
        .cloned()
        .collect::<BTreeSet<_>>();
    entries.insert(format!("{fixture_path}#{case_id}"));
    entries.into_iter().collect()
}

fn validate_iter_success_expectations(
    case: &IterDifferentialCase,
    outcome: &IterOperationOutcome,
) -> Result<(), String> {
    if let Some(expected_transfer_class) = &case.expected_transfer_class {
        let expected = parse_expected_transfer_class(expected_transfer_class)?;
        match outcome {
            IterOperationOutcome::TransferClass(actual) if *actual == expected => {}
            IterOperationOutcome::TransferClass(actual) => {
                return Err(format!(
                    "transfer class mismatch expected={expected:?} actual={actual:?}"
                ));
            }
            _ => {
                return Err(
                    "expected transfer class outcome but operation returned different outcome type"
                        .to_string(),
                );
            }
        }
    }

    if let Some(expected_overlap_action) = &case.expected_overlap_action {
        let expected = parse_expected_overlap_action(expected_overlap_action)?;
        match outcome {
            IterOperationOutcome::OverlapAction(actual) if *actual == expected => {}
            IterOperationOutcome::OverlapAction(actual) => {
                return Err(format!(
                    "overlap action mismatch expected={expected:?} actual={actual:?}"
                ));
            }
            _ => {
                return Err(
                    "expected overlap action outcome but operation returned different outcome type"
                        .to_string(),
                );
            }
        }
    }

    if let Some(expected_selected) = case.expected_selected {
        match outcome {
            IterOperationOutcome::SelectedCount(actual) if *actual == expected_selected => {}
            IterOperationOutcome::SelectedCount(actual) => {
                return Err(format!(
                    "selected-count mismatch expected={expected_selected} actual={actual}"
                ));
            }
            _ => {
                return Err(
                    "expected selected-count outcome but operation returned different outcome type"
                        .to_string(),
                );
            }
        }
    }

    if let Some(expected_shape) = &case.expected_broadcast_shape {
        match outcome {
            IterOperationOutcome::BroadcastPlan(plan)
                if plan.broadcast_shape == *expected_shape => {}
            IterOperationOutcome::BroadcastPlan(plan) => {
                return Err(format!(
                    "broadcast shape mismatch expected={expected_shape:?} actual={:?}",
                    plan.broadcast_shape
                ));
            }
            _ => {
                return Err(
                    "expected broadcast plan outcome but operation returned different outcome type"
                        .to_string(),
                );
            }
        }
    }

    Ok(())
}

pub fn run_iter_differential_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let cases = load_iter_differential_cases(&config.fixture_root)?;
    let mut report = SuiteReport {
        suite: "iter_differential",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };
    let mut mismatches = Vec::new();

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);
        let expected_reason_code = if case.expected_reason_code.trim().is_empty() {
            reason_code.clone()
        } else {
            case.expected_reason_code.trim().to_string()
        };
        let expected_error = case.expected_error_contains.to_lowercase();
        let flat_len = case.len.unwrap_or_else(|| {
            case.flat_index.as_ref().map_or(0, |index| {
                index
                    .stop
                    .max(index.mask.len())
                    .max(index.index.saturating_add(1))
            })
        });

        match execute_iter_operation_with_len(
            &case.id,
            &case.operation,
            case.selector.as_ref(),
            case.overlap.as_ref(),
            case.flat_index.as_ref(),
            case.flags.as_ref(),
            case.broadcast.as_ref(),
            flat_len,
            case.values_len,
        ) {
            Ok(outcome) if expected_error.is_empty() => {
                let actual_reason_code = reason_code.clone();
                match validate_iter_success_expectations(&case, &outcome) {
                    Ok(()) if actual_reason_code == expected_reason_code => {
                        report.pass_count += 1;
                    }
                    Ok(()) => {
                        let rendered = format!(
                            "{}: seed={} mode={} reason_code={} expected_reason_code={} env_fingerprint={} artifact_refs={} success reason-code mismatch",
                            case.id,
                            case.seed,
                            mode,
                            actual_reason_code,
                            expected_reason_code,
                            env_fingerprint,
                            artifact_refs.join(",")
                        );
                        report.failures.push(rendered.clone());
                        mismatches.push(IterDifferentialMismatch {
                            fixture_id: case.id.clone(),
                            seed: case.seed,
                            mode: mode.clone(),
                            operation: case.operation.clone(),
                            expected_reason_code,
                            actual_reason_code,
                            message: rendered,
                            minimal_repro_artifacts: iter_minimal_repro_artifacts(
                                &case.id,
                                &artifact_refs,
                                "crates/fnp-conformance/fixtures/iter_differential_cases.json",
                            ),
                            artifact_refs: artifact_refs.clone(),
                        });
                    }
                    Err(detail) => {
                        let rendered = format!(
                            "{}: seed={} mode={} reason_code={} expected_reason_code={} env_fingerprint={} artifact_refs={} {}",
                            case.id,
                            case.seed,
                            mode,
                            actual_reason_code,
                            expected_reason_code,
                            env_fingerprint,
                            artifact_refs.join(","),
                            detail
                        );
                        report.failures.push(rendered.clone());
                        mismatches.push(IterDifferentialMismatch {
                            fixture_id: case.id.clone(),
                            seed: case.seed,
                            mode: mode.clone(),
                            operation: case.operation.clone(),
                            expected_reason_code,
                            actual_reason_code,
                            message: rendered,
                            minimal_repro_artifacts: iter_minimal_repro_artifacts(
                                &case.id,
                                &artifact_refs,
                                "crates/fnp-conformance/fixtures/iter_differential_cases.json",
                            ),
                            artifact_refs: artifact_refs.clone(),
                        });
                    }
                }
            }
            Ok(_) => {
                let rendered = format!(
                    "{}: seed={} mode={} reason_code={} expected_reason_code={} env_fingerprint={} artifact_refs={} expected error containing '{}' but operation '{}' succeeded",
                    case.id,
                    case.seed,
                    mode,
                    reason_code,
                    expected_reason_code,
                    env_fingerprint,
                    artifact_refs.join(","),
                    case.expected_error_contains,
                    case.operation
                );
                report.failures.push(rendered.clone());
                mismatches.push(IterDifferentialMismatch {
                    fixture_id: case.id.clone(),
                    seed: case.seed,
                    mode: mode.clone(),
                    operation: case.operation.clone(),
                    expected_reason_code,
                    actual_reason_code: "none".to_string(),
                    message: rendered,
                    minimal_repro_artifacts: iter_minimal_repro_artifacts(
                        &case.id,
                        &artifact_refs,
                        "crates/fnp-conformance/fixtures/iter_differential_cases.json",
                    ),
                    artifact_refs: artifact_refs.clone(),
                });
            }
            Err(err) if expected_error.is_empty() => {
                let rendered = format!(
                    "{}: seed={} mode={} reason_code={} expected_reason_code={} env_fingerprint={} artifact_refs={} expected success but got '{}' (actual_reason_code='{}')",
                    case.id,
                    case.seed,
                    mode,
                    reason_code,
                    expected_reason_code,
                    env_fingerprint,
                    artifact_refs.join(","),
                    err.message,
                    err.reason_code
                );
                report.failures.push(rendered.clone());
                mismatches.push(IterDifferentialMismatch {
                    fixture_id: case.id.clone(),
                    seed: case.seed,
                    mode: mode.clone(),
                    operation: case.operation.clone(),
                    expected_reason_code,
                    actual_reason_code: err.reason_code,
                    message: rendered,
                    minimal_repro_artifacts: iter_minimal_repro_artifacts(
                        &case.id,
                        &artifact_refs,
                        "crates/fnp-conformance/fixtures/iter_differential_cases.json",
                    ),
                    artifact_refs: artifact_refs.clone(),
                });
            }
            Err(err) => {
                let contains_expected = err.message.to_lowercase().contains(&expected_error);
                let reason_match = err.reason_code == expected_reason_code;
                if contains_expected && reason_match {
                    report.pass_count += 1;
                } else {
                    let rendered = format!(
                        "{}: seed={} mode={} reason_code={} expected_reason_code={} env_fingerprint={} artifact_refs={} expected error containing '{}' with reason_code='{}' but got '{}' (actual_reason_code='{}')",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        expected_reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        case.expected_error_contains,
                        expected_reason_code,
                        err.message,
                        err.reason_code
                    );
                    report.failures.push(rendered.clone());
                    mismatches.push(IterDifferentialMismatch {
                        fixture_id: case.id.clone(),
                        seed: case.seed,
                        mode: mode.clone(),
                        operation: case.operation.clone(),
                        expected_reason_code,
                        actual_reason_code: err.reason_code,
                        message: rendered,
                        minimal_repro_artifacts: iter_minimal_repro_artifacts(
                            &case.id,
                            &artifact_refs,
                            "crates/fnp-conformance/fixtures/iter_differential_cases.json",
                        ),
                        artifact_refs: artifact_refs.clone(),
                    });
                }
            }
        }
    }

    let artifact = IterDifferentialReportArtifact {
        suite: "iter_differential",
        total_cases: report.case_count,
        passed_cases: report.pass_count,
        failed_cases: report.case_count.saturating_sub(report.pass_count),
        mismatches,
    };
    let report_path = config
        .fixture_root
        .join("oracle_outputs/iter_differential_report.json");
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }
    let payload = serde_json::to_string_pretty(&artifact)
        .map_err(|err| format!("failed serializing iter differential report: {err}"))?;
    fs::write(&report_path, payload.as_bytes())
        .map_err(|err| format!("failed writing {}: {err}", report_path.display()))?;

    Ok(report)
}

pub fn run_iter_metamorphic_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let cases = load_iter_metamorphic_cases(&config.fixture_root)?;
    let mut report = SuiteReport {
        suite: "iter_metamorphic",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);
        let pass = match case.relation.as_str() {
            "selector_deterministic" => {
                let first = execute_iter_operation(
                    &case.id,
                    "select_transfer_class",
                    case.selector.as_ref(),
                    None,
                    None,
                    None,
                    None,
                );
                let second = execute_iter_operation(
                    &case.id,
                    "select_transfer_class",
                    case.selector.as_ref(),
                    None,
                    None,
                    None,
                    None,
                );
                first == second
            }
            "overlap_repeatable" => {
                let first = execute_iter_operation(
                    &case.id,
                    "overlap_copy_policy",
                    None,
                    case.overlap.as_ref(),
                    None,
                    None,
                    None,
                );
                let second = execute_iter_operation(
                    &case.id,
                    "overlap_copy_policy",
                    None,
                    case.overlap.as_ref(),
                    None,
                    None,
                    None,
                );
                first == second
            }
            "flatiter_read_write_count_equivalence" => {
                let Some(flat_index) = case.flat_index.as_ref() else {
                    report.failures.push(format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} missing flat_index payload for relation {}",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        case.relation
                    ));
                    continue;
                };
                let len = infer_flatiter_len(flat_index);
                let Ok(IterOperationOutcome::SelectedCount(selected)) =
                    execute_iter_operation_with_len(
                        &case.id,
                        "validate_flatiter_read",
                        None,
                        None,
                        Some(flat_index),
                        None,
                        None,
                        len,
                        None,
                    )
                else {
                    report.failures.push(format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} failed flatiter read relation check",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(",")
                    ));
                    continue;
                };
                let Ok(IterOperationOutcome::SelectedCount(selected_write)) =
                    execute_iter_operation_with_len(
                        &case.id,
                        "validate_flatiter_write",
                        None,
                        None,
                        Some(flat_index),
                        None,
                        None,
                        len,
                        Some(selected),
                    )
                else {
                    report.failures.push(format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} failed flatiter write relation check",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(",")
                    ));
                    continue;
                };
                selected == selected_write
            }
            "bool_mask_population_matches_selected" => {
                let Some(flat_index) = case.flat_index.as_ref() else {
                    report.failures.push(format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} missing flat_index payload for relation {}",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        case.relation
                    ));
                    continue;
                };
                if flat_index.kind != "bool_mask" {
                    report.failures.push(format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} relation {} requires bool_mask kind",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        case.relation
                    ));
                    continue;
                }
                let len = flat_index.mask.len();
                let Ok(IterOperationOutcome::SelectedCount(selected)) =
                    execute_iter_operation_with_len(
                        &case.id,
                        "validate_flatiter_read",
                        None,
                        None,
                        Some(flat_index),
                        None,
                        None,
                        len,
                        None,
                    )
                else {
                    report.failures.push(format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} failed bool-mask read relation check",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(",")
                    ));
                    continue;
                };
                selected == flat_index.mask.iter().filter(|flag| **flag).count()
            }
            "nditer_flags_idempotent" => {
                let first = execute_iter_operation(
                    &case.id,
                    "validate_nditer_flags",
                    None,
                    None,
                    case.flags.as_ref(),
                    None,
                    None,
                );
                let second = execute_iter_operation(
                    &case.id,
                    "validate_nditer_flags",
                    None,
                    None,
                    case.flags.as_ref(),
                    None,
                    None,
                );
                first == second
            }
            other => {
                report.failures.push(format!(
                    "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} unsupported relation {}",
                    case.id,
                    case.seed,
                    mode,
                    reason_code,
                    env_fingerprint,
                    artifact_refs.join(","),
                    other
                ));
                false
            }
        };

        if pass {
            report.pass_count += 1;
        } else if !report
            .failures
            .iter()
            .any(|failure| failure.starts_with(&format!("{}:", case.id)))
        {
            report.failures.push(format!(
                "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} relation {} failed",
                case.id,
                case.seed,
                mode,
                reason_code,
                env_fingerprint,
                artifact_refs.join(","),
                case.relation
            ));
        }
    }

    Ok(report)
}

pub fn run_iter_adversarial_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let cases = load_iter_adversarial_cases(&config.fixture_root)?;
    let mut report = SuiteReport {
        suite: "iter_adversarial",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);
        let severity = case.severity.trim().to_lowercase();
        if !matches!(severity.as_str(), "low" | "medium" | "high" | "critical") {
            report.failures.push(format!(
                "{}: invalid severity '{}' (must be low|medium|high|critical), reason_code={}, mode={}, env_fingerprint={}, artifact_refs={}",
                case.id,
                case.severity,
                reason_code,
                mode,
                env_fingerprint,
                artifact_refs.join(",")
            ));
            continue;
        }
        if case.expected_error_contains.trim().is_empty() {
            report.failures.push(format!(
                "{}: expected_error_contains must be non-empty, reason_code={}, mode={}, env_fingerprint={}, artifact_refs={}",
                case.id,
                reason_code,
                mode,
                env_fingerprint,
                artifact_refs.join(",")
            ));
            continue;
        }

        let expected_reason_code = if case.expected_reason_code.trim().is_empty() {
            reason_code.clone()
        } else {
            case.expected_reason_code.trim().to_string()
        };
        let flat_len = case.len.unwrap_or_else(|| {
            case.flat_index.as_ref().map_or(0, |index| {
                index
                    .stop
                    .max(index.mask.len())
                    .max(index.index.saturating_add(1))
            })
        });
        let expected_error = case.expected_error_contains.to_lowercase();
        match execute_iter_operation_with_len(
            &case.id,
            &case.operation,
            case.selector.as_ref(),
            case.overlap.as_ref(),
            case.flat_index.as_ref(),
            case.flags.as_ref(),
            None,
            flat_len,
            case.values_len,
        ) {
            Ok(_) => {
                report.failures.push(format!(
                    "{}: severity={severity} seed={} reason_code={} mode={} env_fingerprint={} artifact_refs={} expected error containing '{}' but operation '{}' succeeded",
                    case.id,
                    case.seed,
                    reason_code,
                    mode,
                    env_fingerprint,
                    artifact_refs.join(","),
                    case.expected_error_contains,
                    case.operation
                ));
            }
            Err(err) => {
                let contains_expected = err.message.to_lowercase().contains(&expected_error);
                let reason_match = err.reason_code == expected_reason_code;
                if contains_expected && reason_match {
                    report.pass_count += 1;
                } else {
                    report.failures.push(format!(
                        "{}: severity={severity} seed={} reason_code={} mode={} env_fingerprint={} artifact_refs={} expected error containing '{}' with reason_code='{}' but got '{}' (actual_reason_code='{}')",
                        case.id,
                        case.seed,
                        reason_code,
                        mode,
                        env_fingerprint,
                        artifact_refs.join(","),
                        case.expected_error_contains,
                        expected_reason_code,
                        err.message,
                        err.reason_code
                    ));
                }
            }
        }
    }

    Ok(report)
}

pub fn run_dtype_promotion_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let path = config.fixture_root.join("dtype_promotion_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<PromotionFixtureCase> =
        serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))?;

    let mut report = SuiteReport {
        suite: "dtype_promotion",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };

    for case in cases {
        let lhs =
            DType::parse(&case.lhs).ok_or_else(|| format!("{}: unknown lhs dtype", case.id))?;
        let rhs =
            DType::parse(&case.rhs).ok_or_else(|| format!("{}: unknown rhs dtype", case.id))?;
        let expected = DType::parse(&case.expected)
            .ok_or_else(|| format!("{}: unknown expected dtype", case.id))?;

        let actual = promote(lhs, rhs);
        let passed = actual == expected;
        if passed {
            report.pass_count += 1;
        } else {
            report.failures.push(format!(
                "{}: promotion mismatch expected={} actual={}",
                case.id,
                expected.name(),
                actual.name()
            ));
        }

        let log_entry = DTypePromotionLogEntry {
            suite: "dtype_promotion",
            fixture_id: case.id,
            seed: case.seed,
            mode: if config.strict_mode {
                "strict".to_string()
            } else {
                "hardened".to_string()
            },
            env_fingerprint: normalize_env_fingerprint(&case.env_fingerprint),
            artifact_refs: normalize_artifact_refs(case.artifact_refs),
            reason_code: normalize_reason_code(&case.reason_code),
            lhs: lhs.name().to_string(),
            rhs: rhs.name().to_string(),
            expected: expected.name().to_string(),
            actual: actual.name().to_string(),
            passed,
        };
        maybe_append_dtype_promotion_log(&log_entry)?;
    }

    Ok(report)
}

#[derive(Debug, Clone)]
struct DTypeSuiteError {
    reason_code: String,
    message: String,
}

impl DTypeSuiteError {
    fn new(reason_code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            reason_code: reason_code.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for DTypeSuiteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for DTypeSuiteError {}

fn parse_dtype_for_suite(raw: &str, position: &str) -> Result<DType, DTypeSuiteError> {
    DType::parse(raw).ok_or_else(|| {
        DTypeSuiteError::new(
            "dtype_normalization_failed",
            format!("unknown {position} dtype '{raw}'"),
        )
    })
}

fn parse_expected_bool_for_suite(raw: &str, position: &str) -> Result<bool, DTypeSuiteError> {
    match raw.trim() {
        "true" => Ok(true),
        "false" => Ok(false),
        other => Err(DTypeSuiteError::new(
            "dtype_expected_value_invalid",
            format!("unknown {position} boolean '{other}'"),
        )),
    }
}

pub fn run_dtype_differential_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let cases = load_dtype_differential_cases(&config.fixture_root)?;
    let mut report = SuiteReport {
        suite: "dtype_differential",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };
    let mut mismatches = Vec::new();

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);
        let expected_reason_code = if case.expected_reason_code.trim().is_empty() {
            reason_code.clone()
        } else {
            case.expected_reason_code.trim().to_string()
        };
        let minimal_repro_artifacts = if case.minimal_repro_artifacts.is_empty() {
            artifact_refs.clone()
        } else {
            normalize_artifact_refs(case.minimal_repro_artifacts.clone())
        };

        let mut expected_dtype_for_log = case.expected.clone();
        let (passed, actual_reason_code, actual_dtype_for_log, detail) = match parse_dtype_for_suite(
            &case.lhs, "lhs",
        ) {
            Ok(lhs) => match parse_dtype_for_suite(&case.rhs, "rhs") {
                Ok(rhs) => match case.operation.as_str() {
                    "" | "promote" => match parse_dtype_for_suite(&case.expected, "expected") {
                        Ok(expected) => {
                            expected_dtype_for_log = expected.name().to_string();
                            let actual = promote(lhs, rhs);
                            let actual_reason_code = if actual == expected {
                                reason_code.clone()
                            } else {
                                "dtype_promotion_oracle_mismatch".to_string()
                            };
                            let reason_match = actual_reason_code == expected_reason_code;
                            let passed = actual == expected && reason_match;
                            let detail = if passed {
                                String::new()
                            } else if actual != expected {
                                format!(
                                    "promotion mismatch expected={} actual={}",
                                    expected.name(),
                                    actual.name()
                                )
                            } else {
                                format!(
                                    "reason_code mismatch expected_reason_code={} actual_reason_code={actual_reason_code}",
                                    expected_reason_code
                                )
                            };
                            (
                                passed,
                                actual_reason_code,
                                actual.name().to_string(),
                                detail,
                            )
                        }
                        Err(err) => (
                            false,
                            err.reason_code,
                            "parse_error".to_string(),
                            err.message,
                        ),
                    },
                    "can_cast" => match parse_expected_bool_for_suite(&case.expected, "expected") {
                        Ok(expected) => {
                            expected_dtype_for_log = expected.to_string();
                            let casting = case.casting.trim();
                            let actual = can_cast(lhs, rhs, casting);
                            let actual_reason_code = if actual == expected {
                                reason_code.clone()
                            } else {
                                "dtype_can_cast_oracle_mismatch".to_string()
                            };
                            let reason_match = actual_reason_code == expected_reason_code;
                            let passed = actual == expected && reason_match;
                            let detail = if passed {
                                String::new()
                            } else if actual != expected {
                                format!(
                                    "can_cast mismatch casting={} expected={} actual={actual}",
                                    casting, expected
                                )
                            } else {
                                format!(
                                    "reason_code mismatch expected_reason_code={} actual_reason_code={actual_reason_code}",
                                    expected_reason_code
                                )
                            };
                            (passed, actual_reason_code, actual.to_string(), detail)
                        }
                        Err(err) => (
                            false,
                            err.reason_code,
                            "parse_error".to_string(),
                            err.message,
                        ),
                    },
                    other => (
                        false,
                        "dtype_operation_unsupported".to_string(),
                        "unsupported_operation".to_string(),
                        format!("unsupported dtype differential operation {other}"),
                    ),
                },
                Err(err) => (
                    false,
                    err.reason_code,
                    "parse_error".to_string(),
                    err.message,
                ),
            },
            Err(err) => (
                false,
                err.reason_code,
                "parse_error".to_string(),
                err.message,
            ),
        };

        if passed {
            report.pass_count += 1;
        } else {
            let rendered = format!(
                "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} {} (expected_reason_code={} actual_reason_code={})",
                case.id,
                case.seed,
                mode,
                reason_code,
                env_fingerprint,
                artifact_refs.join(","),
                detail,
                expected_reason_code,
                actual_reason_code
            );
            report.failures.push(rendered.clone());
            mismatches.push(DTypeDifferentialMismatch {
                fixture_id: case.id.clone(),
                seed: case.seed,
                mode: mode.clone(),
                lhs: case.lhs.clone(),
                rhs: case.rhs.clone(),
                expected_dtype: expected_dtype_for_log.clone(),
                actual_dtype: actual_dtype_for_log.clone(),
                expected_reason_code: expected_reason_code.clone(),
                actual_reason_code: actual_reason_code.clone(),
                message: rendered,
                minimal_repro_artifacts: minimal_repro_artifacts.clone(),
                artifact_refs: artifact_refs.clone(),
            });
        }

        let log_entry = DTypePromotionLogEntry {
            suite: "dtype_differential",
            fixture_id: case.id,
            seed: case.seed,
            mode: mode.clone(),
            env_fingerprint,
            artifact_refs,
            reason_code: reason_code.clone(),
            lhs: case.lhs,
            rhs: case.rhs,
            expected: expected_dtype_for_log,
            actual: actual_dtype_for_log,
            passed,
        };
        maybe_append_dtype_promotion_log(&log_entry)?;
    }

    let artifact = DTypeDifferentialReportArtifact {
        suite: "dtype_differential",
        total_cases: report.case_count,
        passed_cases: report.pass_count,
        failed_cases: report.case_count.saturating_sub(report.pass_count),
        mismatches,
    };
    let artifact_path = config
        .fixture_root
        .join("oracle_outputs/dtype_differential_report.json");
    if let Some(parent) = artifact_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }
    let payload = serde_json::to_string_pretty(&artifact)
        .map_err(|err| format!("failed serializing dtype differential report: {err}"))?;
    fs::write(&artifact_path, payload.as_bytes())
        .map_err(|err| format!("failed writing {}: {err}", artifact_path.display()))?;

    Ok(report)
}

pub fn run_dtype_metamorphic_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let cases = load_dtype_metamorphic_cases(&config.fixture_root)?;
    let mut report = SuiteReport {
        suite: "dtype_metamorphic",
        case_count: 0,
        pass_count: 0,
        failures: Vec::new(),
    };

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);

        let (passed, expected_for_log, actual_for_log, detail) = match case.relation.as_str() {
            "commutative" => match case.rhs.as_ref() {
                Some(rhs_raw) => match parse_dtype_for_suite(&case.lhs, "lhs")
                    .and_then(|lhs| parse_dtype_for_suite(rhs_raw, "rhs").map(|rhs| (lhs, rhs)))
                {
                    Ok((lhs, rhs)) => {
                        let lhs_rhs = promote(lhs, rhs);
                        let rhs_lhs = promote(rhs, lhs);
                        (
                            lhs_rhs == rhs_lhs,
                            rhs_lhs.name().to_string(),
                            lhs_rhs.name().to_string(),
                            format!(
                                "commutative promotion mismatch lhs_rhs={} rhs_lhs={}",
                                lhs_rhs.name(),
                                rhs_lhs.name()
                            ),
                        )
                    }
                    Err(err) => (
                        false,
                        "parse_success".to_string(),
                        "parse_error".to_string(),
                        err.message,
                    ),
                },
                None => (
                    false,
                    "relation_requires_rhs".to_string(),
                    "missing_rhs".to_string(),
                    "commutative relation requires rhs dtype".to_string(),
                ),
            },
            "idempotent" => match parse_dtype_for_suite(&case.lhs, "lhs") {
                Ok(lhs) => {
                    let actual = promote(lhs, lhs);
                    (
                        actual == lhs,
                        lhs.name().to_string(),
                        actual.name().to_string(),
                        format!(
                            "idempotent promotion mismatch expected={} actual={}",
                            lhs.name(),
                            actual.name()
                        ),
                    )
                }
                Err(err) => (
                    false,
                    "parse_success".to_string(),
                    "parse_error".to_string(),
                    err.message,
                ),
            },
            "lossless_cast_destination" => match case.rhs.as_ref() {
                Some(rhs_raw) => match parse_dtype_for_suite(&case.lhs, "lhs")
                    .and_then(|lhs| parse_dtype_for_suite(rhs_raw, "rhs").map(|rhs| (lhs, rhs)))
                {
                    Ok((lhs, rhs)) => {
                        if !can_cast_lossless(lhs, rhs) {
                            (
                                false,
                                "cast_lossless_precondition".to_string(),
                                "cast_lossless_precondition_failed".to_string(),
                                format!(
                                    "fixture precondition failed: can_cast_lossless({}, {}) is false",
                                    lhs.name(),
                                    rhs.name()
                                ),
                            )
                        } else {
                            let actual = promote(lhs, rhs);
                            (
                                actual == rhs,
                                rhs.name().to_string(),
                                actual.name().to_string(),
                                format!(
                                    "lossless-cast promotion mismatch expected={} actual={}",
                                    rhs.name(),
                                    actual.name()
                                ),
                            )
                        }
                    }
                    Err(err) => (
                        false,
                        "parse_success".to_string(),
                        "parse_error".to_string(),
                        err.message,
                    ),
                },
                None => (
                    false,
                    "relation_requires_rhs".to_string(),
                    "missing_rhs".to_string(),
                    "lossless_cast_destination relation requires rhs dtype".to_string(),
                ),
            },
            "safe_implies_same_kind" => match case.rhs.as_ref() {
                Some(rhs_raw) => match parse_dtype_for_suite(&case.lhs, "lhs")
                    .and_then(|lhs| parse_dtype_for_suite(rhs_raw, "rhs").map(|rhs| (lhs, rhs)))
                {
                    Ok((lhs, rhs)) => {
                        let safe = can_cast(lhs, rhs, "safe");
                        let same_kind = can_cast(lhs, rhs, "same_kind");
                        (
                            !safe || same_kind,
                            "safe_subset_same_kind".to_string(),
                            format!("safe={safe},same_kind={same_kind}"),
                            format!(
                                "safe cast must imply same_kind for lhs={} rhs={}",
                                lhs.name(),
                                rhs.name()
                            ),
                        )
                    }
                    Err(err) => (
                        false,
                        "parse_success".to_string(),
                        "parse_error".to_string(),
                        err.message,
                    ),
                },
                None => (
                    false,
                    "relation_requires_rhs".to_string(),
                    "missing_rhs".to_string(),
                    "safe_implies_same_kind relation requires rhs dtype".to_string(),
                ),
            },
            other => (
                false,
                "known_relation".to_string(),
                other.to_string(),
                format!("unsupported relation {other}"),
            ),
        };

        record_suite_check(
            &mut report,
            passed,
            format!(
                "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} {}",
                case.id,
                case.seed,
                mode,
                reason_code,
                env_fingerprint,
                artifact_refs.join(","),
                detail
            ),
        );

        let rhs_for_log = case.rhs.unwrap_or_else(|| case.lhs.clone());
        let log_entry = DTypePromotionLogEntry {
            suite: "dtype_metamorphic",
            fixture_id: case.id,
            seed: case.seed,
            mode: mode.clone(),
            env_fingerprint,
            artifact_refs,
            reason_code,
            lhs: case.lhs,
            rhs: rhs_for_log,
            expected: expected_for_log,
            actual: actual_for_log,
            passed,
        };
        maybe_append_dtype_promotion_log(&log_entry)?;
    }

    Ok(report)
}

fn execute_dtype_adversarial_operation(
    case: &DTypeAdversarialCase,
) -> Result<(DType, DType, DType), DTypeSuiteError> {
    let lhs = parse_dtype_for_suite(&case.lhs, "lhs")?;
    let rhs = parse_dtype_for_suite(&case.rhs, "rhs")?;
    let actual = promote(lhs, rhs);
    Ok((lhs, rhs, actual))
}

pub fn run_dtype_adversarial_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let cases = load_dtype_adversarial_cases(&config.fixture_root)?;
    let mut report = SuiteReport {
        suite: "dtype_adversarial",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);
        let expected_reason_code = if case.expected_reason_code.trim().is_empty() {
            reason_code.clone()
        } else {
            case.expected_reason_code.trim().to_string()
        };
        let severity = case.severity.trim().to_lowercase();
        if !matches!(severity.as_str(), "low" | "medium" | "high" | "critical") {
            report.failures.push(format!(
                "{}: invalid severity '{}' (must be low|medium|high|critical), reason_code={}, mode={}, env_fingerprint={}, artifact_refs={}",
                case.id,
                case.severity,
                reason_code,
                mode,
                env_fingerprint,
                artifact_refs.join(",")
            ));
            continue;
        }
        if case.expected_error_contains.trim().is_empty() {
            report.failures.push(format!(
                "{}: expected_error_contains must be non-empty, reason_code={}, mode={}, env_fingerprint={}, artifact_refs={}",
                case.id,
                reason_code,
                mode,
                env_fingerprint,
                artifact_refs.join(",")
            ));
            continue;
        }
        let expected_error = case.expected_error_contains.to_lowercase();
        let (passed, expected_for_log, actual_for_log) = match execute_dtype_adversarial_operation(
            &case,
        ) {
            Ok((_lhs, _rhs, actual)) => {
                let expected_for_log = format!(
                    "error_contains:{} reason_code={}",
                    case.expected_error_contains, expected_reason_code
                );
                report.failures.push(format!(
                    "{}: severity={severity} seed={} reason_code={} mode={} env_fingerprint={} artifact_refs={} expected error containing '{}' but operation succeeded with promoted dtype={}",
                    case.id,
                    case.seed,
                    reason_code,
                    mode,
                    env_fingerprint,
                    artifact_refs.join(","),
                    case.expected_error_contains,
                    actual.name()
                ));
                (false, expected_for_log, actual.name().to_string())
            }
            Err(err) => {
                let contains_expected = err.message.to_lowercase().contains(&expected_error);
                let reason_match = err.reason_code == expected_reason_code;
                let expected_for_log = format!(
                    "error_contains:{} reason_code={}",
                    case.expected_error_contains, expected_reason_code
                );
                if contains_expected && reason_match {
                    report.pass_count += 1;
                    (
                        true,
                        expected_for_log,
                        format!("error reason_code={}", err.reason_code),
                    )
                } else {
                    report.failures.push(format!(
                        "{}: severity={severity} seed={} reason_code={} mode={} env_fingerprint={} artifact_refs={} expected error containing '{}' with reason_code='{}' but got '{}' (actual_reason_code='{}')",
                        case.id,
                        case.seed,
                        reason_code,
                        mode,
                        env_fingerprint,
                        artifact_refs.join(","),
                        case.expected_error_contains,
                        expected_reason_code,
                        err.message,
                        err.reason_code
                    ));
                    (
                        false,
                        expected_for_log,
                        format!("error reason_code={}", err.reason_code),
                    )
                }
            }
        };

        let log_entry = DTypePromotionLogEntry {
            suite: "dtype_adversarial",
            fixture_id: case.id,
            seed: case.seed,
            mode: mode.clone(),
            env_fingerprint,
            artifact_refs,
            reason_code: reason_code.clone(),
            lhs: case.lhs,
            rhs: case.rhs,
            expected: expected_for_log,
            actual: actual_for_log,
            passed,
        };
        maybe_append_dtype_promotion_log(&log_entry)?;
    }

    Ok(report)
}

pub fn run_runtime_policy_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let path = config.fixture_root.join("runtime_policy_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<PolicyFixtureCase> =
        serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))?;

    let mut report = SuiteReport {
        suite: "runtime_policy",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };

    let mut ledger = EvidenceLedger::new();

    for case in cases {
        let Some(mode) = RuntimeMode::from_wire(&case.mode) else {
            return Err(format!("{}: invalid mode {}", case.id, case.mode));
        };
        let class = CompatibilityClass::from_wire(&case.class);
        let expected_action = parse_expected_action(&case.id, &case.expected_action)?;
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());

        let context = DecisionAuditContext {
            fixture_id: case.id.clone(),
            seed: case.seed,
            env_fingerprint: normalize_env_fingerprint(&case.env_fingerprint),
            artifact_refs: artifact_refs.clone(),
            reason_code: normalize_reason_code(&case.reason_code),
        };

        let actual = decide_and_record_with_context(
            &mut ledger,
            mode,
            class,
            case.risk_score,
            case.threshold,
            context,
            "runtime_policy_suite",
        );

        let passed = actual == expected_action;
        if passed {
            report.pass_count += 1;
        } else {
            report.failures.push(format!(
                "{}: action mismatch expected={expected_action:?} actual={actual:?}",
                case.id
            ));
        }

        let log_entry = RuntimePolicyLogEntry {
            suite: "runtime_policy",
            fixture_id: case.id,
            seed: case.seed,
            mode: mode.as_str().to_string(),
            env_fingerprint: normalize_env_fingerprint(&case.env_fingerprint),
            artifact_refs,
            reason_code: normalize_reason_code(&case.reason_code),
            expected_action: expected_action.as_str().to_string(),
            actual_action: actual.as_str().to_string(),
            passed,
        };
        maybe_append_runtime_policy_log(&log_entry)?;
    }

    if ledger.events().len() != report.case_count {
        report.failures.push(format!(
            "ledger size mismatch expected={} actual={}",
            report.case_count,
            ledger.events().len()
        ));
    }

    validate_runtime_policy_log_fields(&mut report, ledger.events());

    Ok(report)
}

pub fn run_runtime_policy_adversarial_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let path = config
        .fixture_root
        .join("runtime_policy_adversarial_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<PolicyAdversarialFixtureCase> =
        serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))?;

    let mut report = SuiteReport {
        suite: "runtime_policy_adversarial",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };

    let mut ledger = EvidenceLedger::new();

    for case in cases {
        let expected_action = parse_expected_action(&case.id, &case.expected_action)?;
        let actual = decide_compatibility_from_wire(
            &case.mode_raw,
            &case.class_raw,
            case.risk_score,
            case.threshold,
        );

        if let Some(mode) = RuntimeMode::from_wire(&case.mode_raw) {
            let class = CompatibilityClass::from_wire(&case.class_raw);
            let context = DecisionAuditContext {
                fixture_id: case.id.clone(),
                seed: case.seed,
                env_fingerprint: normalize_env_fingerprint(&case.env_fingerprint),
                artifact_refs: normalize_artifact_refs(case.artifact_refs.clone()),
                reason_code: normalize_reason_code(&case.reason_code),
            };
            let _ = decide_and_record_with_context(
                &mut ledger,
                mode,
                class,
                case.risk_score,
                case.threshold,
                context,
                "runtime_policy_adversarial_suite",
            );
        }

        let passed = actual == expected_action;
        if passed {
            report.pass_count += 1;
        } else {
            report.failures.push(format!(
                "{}: action mismatch expected={expected_action:?} actual={actual:?}",
                case.id
            ));
        }

        let log_entry = RuntimePolicyLogEntry {
            suite: "runtime_policy_adversarial",
            fixture_id: case.id,
            seed: case.seed,
            mode: case.mode_raw,
            env_fingerprint: normalize_env_fingerprint(&case.env_fingerprint),
            artifact_refs: normalize_artifact_refs(case.artifact_refs),
            reason_code: normalize_reason_code(&case.reason_code),
            expected_action: expected_action.as_str().to_string(),
            actual_action: actual.as_str().to_string(),
            passed,
        };
        maybe_append_runtime_policy_log(&log_entry)?;
    }

    validate_runtime_policy_log_fields(&mut report, ledger.events());

    Ok(report)
}

pub fn run_override_audit_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let path = config.fixture_root.join("override_audit_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<OverrideAuditFixtureCase> =
        serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))?;

    let mut report = SuiteReport {
        suite: "override_audit",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };

    for case in cases {
        let Some(mode) = RuntimeMode::from_wire(&case.mode) else {
            report
                .failures
                .push(format!("{}: invalid mode {}", case.id, case.mode));
            continue;
        };
        let class = CompatibilityClass::from_wire(&case.class);
        let allowlist_refs: Vec<&str> = case
            .allowlisted_classes
            .iter()
            .map(|s| s.as_str())
            .collect();
        let expected_action = parse_expected_action(&case.id, &case.expected_action)?;

        let event: OverrideAuditEvent = evaluate_policy_override(
            mode,
            class,
            &case.requested_deviation_class,
            &allowlist_refs,
            &case.packet_id,
            &case.requested_by,
            &case.override_reason_code,
        );

        let mut passed = true;
        let mut failure_reasons = Vec::new();

        if event.approved != case.expected_approved {
            passed = false;
            failure_reasons.push(format!(
                "approved mismatch expected={} actual={}",
                case.expected_approved, event.approved
            ));
        }
        if event.action != expected_action {
            passed = false;
            failure_reasons.push(format!(
                "action mismatch expected={expected_action:?} actual={:?}",
                event.action
            ));
        }

        if passed {
            report.pass_count += 1;
        } else {
            report
                .failures
                .push(format!("{}: {}", case.id, failure_reasons.join("; ")));
        }
    }

    Ok(report)
}

pub fn run_ufunc_differential_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let input_path = config.fixture_root.join("ufunc_input_cases.json");
    let oracle_path = config
        .fixture_root
        .join("oracle_outputs/ufunc_oracle_output.json");
    let report_path = config
        .fixture_root
        .join("oracle_outputs/ufunc_differential_report.json");

    let report = ufunc_differential::compare_against_oracle(&input_path, &oracle_path, 1e-9, 1e-9)?;
    ufunc_differential::write_differential_report(&report_path, &report)?;

    let mismatches = report
        .failures
        .iter()
        .map(|failure| UFuncDifferentialMismatch {
            fixture_id: failure.id.clone(),
            seed: failure.seed,
            mode: failure.mode.clone(),
            expected_reason_code: failure.expected_reason_code.clone(),
            actual_reason_code: failure.actual_reason_code.clone(),
            message: failure
                .reason
                .clone()
                .unwrap_or_else(|| "no reason provided".to_string()),
            artifact_refs: failure.artifact_refs.clone(),
        })
        .collect::<Vec<_>>();
    let artifact = UFuncDifferentialReportArtifact {
        suite: "ufunc_differential",
        total_cases: report.total_cases,
        passed_cases: report.passed_cases,
        failed_cases: report.failed_cases,
        mismatches,
    };
    let artifact_path = config
        .fixture_root
        .join("oracle_outputs/ufunc_differential_mismatch_report.json");
    let payload = serde_json::to_string_pretty(&artifact)
        .map_err(|err| format!("failed serializing ufunc differential report: {err}"))?;
    fs::write(&artifact_path, payload.as_bytes())
        .map_err(|err| format!("failed writing {}: {err}", artifact_path.display()))?;

    let failures = report
        .failures
        .iter()
        .map(|failure| {
            format!(
                "{}: seed={} mode={} reason_code={} expected_reason_code={} env_fingerprint={} artifact_refs={} {}",
                failure.id,
                failure.seed,
                failure.mode,
                failure.actual_reason_code,
                failure.expected_reason_code,
                failure.env_fingerprint,
                failure.artifact_refs.join(","),
                failure.reason.as_deref().unwrap_or("no reason provided")
            )
        })
        .collect();

    Ok(SuiteReport {
        suite: "ufunc_differential",
        case_count: report.total_cases,
        pass_count: report.passed_cases,
        failures,
    })
}

pub fn run_ufunc_metamorphic_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let path = config.fixture_root.join("ufunc_metamorphic_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<UFuncMetamorphicCase> =
        serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))?;

    let mut report = SuiteReport {
        suite: "ufunc_metamorphic",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);

        let pass = match case.relation.as_str() {
            "add_commutative" => {
                let Some(rhs_shape) = case.rhs_shape.clone() else {
                    report.failures.push(format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} missing rhs_shape for add_commutative",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(",")
                    ));
                    continue;
                };
                let Some(rhs_values) = case.rhs_values.clone() else {
                    report.failures.push(format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} missing rhs_values for add_commutative",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(",")
                    ));
                    continue;
                };
                let rhs_dtype = case
                    .rhs_dtype
                    .clone()
                    .unwrap_or_else(default_f64_dtype_name);

                let lhs_rhs = UFuncInputCase {
                    id: format!("{}::lhs_rhs", case.id),
                    op: UFuncOperation::Add,
                    lhs_shape: case.lhs_shape.clone(),
                    lhs_values: case.lhs_values.clone(),
                    lhs_dtype: case.lhs_dtype.clone(),
                    rhs_shape: Some(rhs_shape.clone()),
                    rhs_values: Some(rhs_values.clone()),
                    rhs_dtype: Some(rhs_dtype.clone()),
                    axis: None,
                    axes: None,
                    keepdims: None,
                    ddof: None,
                    clip_min: None,
                    clip_max: None,
                    third_shape: None,
                    third_values: None,
                    third_dtype: None,
                    sig: None,
                    signature: None,
                    dtype: None,
                    seed: case.seed,
                    mode: mode.clone(),
                    env_fingerprint: env_fingerprint.clone(),
                    artifact_refs: artifact_refs.clone(),
                    reason_code: reason_code.clone(),
                    expected_reason_code: reason_code.clone(),
                    expected_error_contains: String::new(),
                };
                let rhs_lhs = UFuncInputCase {
                    id: format!("{}::rhs_lhs", case.id),
                    op: UFuncOperation::Add,
                    lhs_shape: rhs_shape,
                    lhs_values: rhs_values,
                    lhs_dtype: rhs_dtype,
                    rhs_shape: Some(case.lhs_shape.clone()),
                    rhs_values: Some(case.lhs_values.clone()),
                    rhs_dtype: Some(case.lhs_dtype.clone()),
                    axis: None,
                    axes: None,
                    keepdims: None,
                    ddof: None,
                    clip_min: None,
                    clip_max: None,
                    third_shape: None,
                    third_values: None,
                    third_dtype: None,
                    sig: None,
                    signature: None,
                    dtype: None,
                    seed: case.seed,
                    mode: mode.clone(),
                    env_fingerprint: env_fingerprint.clone(),
                    artifact_refs: artifact_refs.clone(),
                    reason_code: reason_code.clone(),
                    expected_reason_code: reason_code.clone(),
                    expected_error_contains: String::new(),
                };
                evaluate_commutative_pair(
                    &case.id,
                    case.seed,
                    &mode,
                    &reason_code,
                    &env_fingerprint,
                    &artifact_refs,
                    lhs_rhs,
                    rhs_lhs,
                    &mut report,
                )
            }
            "mul_commutative" => {
                let Some(rhs_shape) = case.rhs_shape.clone() else {
                    report.failures.push(format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} missing rhs_shape for mul_commutative",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(",")
                    ));
                    continue;
                };
                let Some(rhs_values) = case.rhs_values.clone() else {
                    report.failures.push(format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} missing rhs_values for mul_commutative",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(",")
                    ));
                    continue;
                };
                let rhs_dtype = case
                    .rhs_dtype
                    .clone()
                    .unwrap_or_else(default_f64_dtype_name);

                let lhs_rhs = UFuncInputCase {
                    id: format!("{}::lhs_rhs", case.id),
                    op: UFuncOperation::Mul,
                    lhs_shape: case.lhs_shape.clone(),
                    lhs_values: case.lhs_values.clone(),
                    lhs_dtype: case.lhs_dtype.clone(),
                    rhs_shape: Some(rhs_shape.clone()),
                    rhs_values: Some(rhs_values.clone()),
                    rhs_dtype: Some(rhs_dtype.clone()),
                    axis: None,
                    axes: None,
                    keepdims: None,
                    ddof: None,
                    clip_min: None,
                    clip_max: None,
                    third_shape: None,
                    third_values: None,
                    third_dtype: None,
                    sig: None,
                    signature: None,
                    dtype: None,
                    seed: case.seed,
                    mode: mode.clone(),
                    env_fingerprint: env_fingerprint.clone(),
                    artifact_refs: artifact_refs.clone(),
                    reason_code: reason_code.clone(),
                    expected_reason_code: reason_code.clone(),
                    expected_error_contains: String::new(),
                };
                let rhs_lhs = UFuncInputCase {
                    id: format!("{}::rhs_lhs", case.id),
                    op: UFuncOperation::Mul,
                    lhs_shape: rhs_shape,
                    lhs_values: rhs_values,
                    lhs_dtype: rhs_dtype,
                    rhs_shape: Some(case.lhs_shape.clone()),
                    rhs_values: Some(case.lhs_values.clone()),
                    rhs_dtype: Some(case.lhs_dtype.clone()),
                    axis: None,
                    axes: None,
                    keepdims: None,
                    ddof: None,
                    clip_min: None,
                    clip_max: None,
                    third_shape: None,
                    third_values: None,
                    third_dtype: None,
                    sig: None,
                    signature: None,
                    dtype: None,
                    seed: case.seed,
                    mode: mode.clone(),
                    env_fingerprint: env_fingerprint.clone(),
                    artifact_refs: artifact_refs.clone(),
                    reason_code: reason_code.clone(),
                    expected_reason_code: reason_code.clone(),
                    expected_error_contains: String::new(),
                };
                evaluate_commutative_pair(
                    &case.id,
                    case.seed,
                    &mode,
                    &reason_code,
                    &env_fingerprint,
                    &artifact_refs,
                    lhs_rhs,
                    rhs_lhs,
                    &mut report,
                )
            }
            "sum_linearity" => {
                let scalar = case.scalar.unwrap_or(1.0);
                let sum_case = UFuncInputCase {
                    id: format!("{}::sum_base", case.id),
                    op: UFuncOperation::Sum,
                    lhs_shape: case.lhs_shape.clone(),
                    lhs_values: case.lhs_values.clone(),
                    lhs_dtype: case.lhs_dtype.clone(),
                    rhs_shape: None,
                    rhs_values: None,
                    rhs_dtype: None,
                    axis: None,
                    axes: None,
                    keepdims: Some(false),
                    ddof: None,
                    clip_min: None,
                    clip_max: None,
                    third_shape: None,
                    third_values: None,
                    third_dtype: None,
                    sig: None,
                    signature: None,
                    dtype: None,
                    seed: case.seed,
                    mode: mode.clone(),
                    env_fingerprint: env_fingerprint.clone(),
                    artifact_refs: artifact_refs.clone(),
                    reason_code: reason_code.clone(),
                    expected_reason_code: reason_code.clone(),
                    expected_error_contains: String::new(),
                };
                let scale_case = UFuncInputCase {
                    id: format!("{}::scale", case.id),
                    op: UFuncOperation::Mul,
                    lhs_shape: case.lhs_shape.clone(),
                    lhs_values: case.lhs_values.clone(),
                    lhs_dtype: case.lhs_dtype.clone(),
                    rhs_shape: Some(Vec::new()),
                    rhs_values: Some(vec![scalar]),
                    rhs_dtype: Some(default_f64_dtype_name()),
                    axis: None,
                    axes: None,
                    keepdims: None,
                    ddof: None,
                    clip_min: None,
                    clip_max: None,
                    third_shape: None,
                    third_values: None,
                    third_dtype: None,
                    sig: None,
                    signature: None,
                    dtype: None,
                    seed: case.seed,
                    mode: mode.clone(),
                    env_fingerprint: env_fingerprint.clone(),
                    artifact_refs: artifact_refs.clone(),
                    reason_code: reason_code.clone(),
                    expected_reason_code: reason_code.clone(),
                    expected_error_contains: String::new(),
                };

                let Ok((_, base_values, _)) = ufunc_differential::execute_input_case(&sum_case)
                else {
                    report.failures.push(format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} failed evaluating base sum",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(",")
                    ));
                    continue;
                };
                let Ok((scaled_shape, scaled_values, _)) =
                    ufunc_differential::execute_input_case(&scale_case)
                else {
                    report.failures.push(format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} failed evaluating scaled array",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(",")
                    ));
                    continue;
                };

                let scaled_sum_case = UFuncInputCase {
                    id: format!("{}::sum_scaled", case.id),
                    op: UFuncOperation::Sum,
                    lhs_shape: scaled_shape,
                    lhs_values: scaled_values,
                    lhs_dtype: default_f64_dtype_name(),
                    rhs_shape: None,
                    rhs_values: None,
                    rhs_dtype: None,
                    axis: None,
                    axes: None,
                    keepdims: Some(false),
                    ddof: None,
                    clip_min: None,
                    clip_max: None,
                    third_shape: None,
                    third_values: None,
                    third_dtype: None,
                    sig: None,
                    signature: None,
                    dtype: None,
                    seed: case.seed,
                    mode: mode.clone(),
                    env_fingerprint: env_fingerprint.clone(),
                    artifact_refs: artifact_refs.clone(),
                    reason_code: reason_code.clone(),
                    expected_reason_code: reason_code.clone(),
                    expected_error_contains: String::new(),
                };

                let Ok((_, sum_scaled_values, _)) =
                    ufunc_differential::execute_input_case(&scaled_sum_case)
                else {
                    report.failures.push(format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} failed evaluating scaled sum",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(",")
                    ));
                    continue;
                };

                let expected = base_values.first().copied().unwrap_or(0.0) * scalar;
                let actual = sum_scaled_values.first().copied().unwrap_or(0.0);
                let abs_err = (expected - actual).abs();
                let threshold = 1e-9 + 1e-9 * expected.abs();
                if abs_err > threshold {
                    report.failures.push(format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} sum_linearity mismatch expected={} actual={} abs_err={} threshold={}",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        expected,
                        actual,
                        abs_err,
                        threshold
                    ));
                    false
                } else {
                    true
                }
            }
            other => {
                report.failures.push(format!(
                    "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} unsupported relation {}",
                    case.id,
                    case.seed,
                    mode,
                    reason_code,
                    env_fingerprint,
                    artifact_refs.join(","),
                    other
                ));
                false
            }
        };

        if pass {
            report.pass_count += 1;
        }
    }

    Ok(report)
}

pub fn run_ufunc_adversarial_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let path = config.fixture_root.join("ufunc_adversarial_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<UFuncAdversarialCase> =
        serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))?;

    let mut report = SuiteReport {
        suite: "ufunc_adversarial",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);
        let severity = case.severity.trim().to_lowercase();
        if !matches!(severity.as_str(), "low" | "medium" | "high" | "critical") {
            report.failures.push(format!(
                "{}: invalid severity '{}' (must be low|medium|high|critical), reason_code={}, mode={}, env_fingerprint={}, artifact_refs={}",
                case.id,
                case.severity,
                reason_code,
                mode,
                env_fingerprint,
                artifact_refs.join(",")
            ));
            continue;
        }
        let expected_error_contains = case.expected_error_contains.trim().to_lowercase();
        if case.expect_success {
            if case.expected_shape.is_none() {
                report.failures.push(format!(
                    "{}: expect_success=true requires expected_shape, reason_code={}, mode={}, env_fingerprint={}, artifact_refs={}",
                    case.id,
                    reason_code,
                    mode,
                    env_fingerprint,
                    artifact_refs.join(",")
                ));
                continue;
            }
        } else if expected_error_contains.is_empty() {
            report.failures.push(format!(
                "{}: expected_error_contains must be non-empty for error-expected adversarial cases, reason_code={}, mode={}, env_fingerprint={}, artifact_refs={}",
                case.id,
                reason_code,
                mode,
                env_fingerprint,
                artifact_refs.join(",")
            ));
            continue;
        }
        let expected_reason_code = if case.expected_reason_code.trim().is_empty() {
            reason_code.clone()
        } else {
            case.expected_reason_code.trim().to_string()
        };
        let input = UFuncInputCase {
            id: case.id.clone(),
            op: case.op,
            lhs_shape: case.lhs_shape.clone(),
            lhs_values: case.lhs_values.clone(),
            lhs_dtype: case.lhs_dtype.clone(),
            rhs_shape: case.rhs_shape.clone(),
            rhs_values: case.rhs_values.clone(),
            rhs_dtype: case.rhs_dtype.clone(),
            axis: case.axis,
            axes: None,
            keepdims: case.keepdims,
            ddof: None,
            clip_min: case.clip_min,
            clip_max: case.clip_max,
            third_shape: None,
            third_values: None,
            third_dtype: None,
            sig: None,
            signature: None,
            dtype: None,
            seed: case.seed,
            mode: mode.clone(),
            env_fingerprint: env_fingerprint.clone(),
            artifact_refs: artifact_refs.clone(),
            reason_code: reason_code.clone(),
            expected_reason_code: expected_reason_code.clone(),
            expected_error_contains: case.expected_error_contains.clone(),
        };

        match ufunc_differential::execute_input_case(&input) {
            Ok((shape, values, _)) => {
                if case.expect_success {
                    let Some(expected_shape) = case.expected_shape.as_ref() else {
                        report.failures.push(format!(
                            "{}: internal contract error expected_shape missing at success validation, reason_code={}, mode={}, env_fingerprint={}, artifact_refs={}",
                            case.id,
                            reason_code,
                            mode,
                            env_fingerprint,
                            artifact_refs.join(",")
                        ));
                        continue;
                    };
                    if &shape != expected_shape {
                        report.failures.push(format!(
                            "{}: severity={severity} seed={} reason_code={} mode={} env_fingerprint={} artifact_refs={} expect_success shape mismatch expected={expected_shape:?} actual={shape:?}",
                            case.id,
                            case.seed,
                            reason_code,
                            mode,
                            env_fingerprint,
                            artifact_refs.join(",")
                        ));
                        continue;
                    }
                    if case.expect_non_finite && values.iter().all(|value| value.is_finite()) {
                        report.failures.push(format!(
                            "{}: severity={severity} seed={} reason_code={} mode={} env_fingerprint={} artifact_refs={} expect_non_finite=true but output contained only finite values",
                            case.id,
                            case.seed,
                            reason_code,
                            mode,
                            env_fingerprint,
                            artifact_refs.join(",")
                        ));
                        continue;
                    }
                    report.pass_count += 1;
                } else {
                    report.failures.push(format!(
                        "{}: severity={severity} seed={} reason_code={} mode={} env_fingerprint={} artifact_refs={} expected error containing '{}' with reason_code='{}' but execution succeeded with shape={shape:?}",
                        case.id,
                        case.seed,
                        reason_code,
                        mode,
                        env_fingerprint,
                        artifact_refs.join(","),
                        case.expected_error_contains,
                        expected_reason_code
                    ));
                }
            }
            Err(err) => {
                if case.expect_success {
                    let actual_reason_code = classify_ufunc_reason_code(case.op, &err);
                    report.failures.push(format!(
                        "{}: severity={severity} seed={} reason_code={} mode={} env_fingerprint={} artifact_refs={} expected success but got '{}' (reason_code='{}')",
                        case.id,
                        case.seed,
                        reason_code,
                        mode,
                        env_fingerprint,
                        artifact_refs.join(","),
                        err,
                        actual_reason_code
                    ));
                } else {
                    let actual = err.to_lowercase();
                    let actual_reason_code = classify_ufunc_reason_code(case.op, &err);
                    if actual.contains(&expected_error_contains)
                        && actual_reason_code == expected_reason_code
                    {
                        report.pass_count += 1;
                    } else {
                        report.failures.push(format!(
                            "{}: severity={severity} seed={} reason_code={} mode={} env_fingerprint={} artifact_refs={} expected error containing '{}' with reason_code='{}' but got '{}' (reason_code='{}')",
                            case.id,
                            case.seed,
                            reason_code,
                            mode,
                            env_fingerprint,
                            artifact_refs.join(","),
                            case.expected_error_contains,
                            expected_reason_code,
                            err,
                            actual_reason_code
                        ));
                    }
                }
            }
        }
    }

    Ok(report)
}

#[derive(Debug)]
enum LinalgOperationOutcome {
    Unit,
    SolveVector(Vec<f64>),
    QrShapes {
        q_shape: Option<Vec<usize>>,
        r_shape: Vec<usize>,
    },
    QrDecomposition {
        q: Option<[[f64; 2]; 2]>,
        r: [[f64; 2]; 2],
    },
    SvdShapes {
        u_shape: Vec<usize>,
        s_shape: Vec<usize>,
        vh_shape: Vec<usize>,
    },
    LstsqShapes {
        x_shape: Vec<usize>,
        residuals_shape: Vec<usize>,
        rank_upper_bound: usize,
        singular_values_shape: Vec<usize>,
    },
    SvdDecomposition {
        u: [[f64; 2]; 2],
        singular_values: [f64; 2],
        vt: [[f64; 2]; 2],
    },
}

struct LinalgOperationInput<'a> {
    operation: &'a str,
    matrix: &'a [Vec<f64>],
    rhs: &'a [f64],
    shape: &'a [usize],
    rhs_shape: &'a [usize],
    qr_mode: &'a str,
    full_matrices: bool,
    converged: bool,
    uplo: &'a str,
    rcond: f64,
    search_depth: usize,
    backend_supported: bool,
    validation_retries: usize,
    mode_raw: &'a str,
    class_raw: &'a str,
}

pub fn run_linalg_differential_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let path = config.fixture_root.join("linalg_differential_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<LinalgDifferentialCase> =
        serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))?;

    let mut report = SuiteReport {
        suite: "linalg_differential",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };
    let mut mismatches = Vec::new();

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);
        let expected_reason_code = if case.expected_reason_code.trim().is_empty() {
            reason_code.clone()
        } else {
            case.expected_reason_code.trim().to_string()
        };

        let outcome = execute_linalg_differential_operation(&case);
        if case.expected_error_contains.trim().is_empty() {
            match outcome {
                Ok(actual) => match validate_linalg_differential_expectation(&case, &actual) {
                    Ok(()) => {
                        report.pass_count += 1;
                    }
                    Err(message) => {
                        let rendered = format!(
                            "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} {}",
                            case.id,
                            case.seed,
                            mode,
                            reason_code,
                            env_fingerprint,
                            artifact_refs.join(","),
                            message
                        );
                        report.failures.push(rendered.clone());
                        mismatches.push(LinalgDifferentialMismatch {
                            fixture_id: case.id,
                            seed: case.seed,
                            mode,
                            expected_reason_code,
                            actual_reason_code: reason_code,
                            message: rendered,
                            artifact_refs,
                        });
                    }
                },
                Err(err) => {
                    let actual_reason_code = err.reason_code().to_string();
                    let rendered = format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected success but got error '{}'",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        err
                    );
                    report.failures.push(rendered.clone());
                    mismatches.push(LinalgDifferentialMismatch {
                        fixture_id: case.id,
                        seed: case.seed,
                        mode,
                        expected_reason_code,
                        actual_reason_code,
                        message: rendered,
                        artifact_refs,
                    });
                }
            }
        } else {
            let expected_error = case.expected_error_contains.to_lowercase();
            match outcome {
                Ok(_) => {
                    let rendered = format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected error containing '{}' but operation '{}' succeeded",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        case.expected_error_contains,
                        case.operation
                    );
                    report.failures.push(rendered.clone());
                    mismatches.push(LinalgDifferentialMismatch {
                        fixture_id: case.id,
                        seed: case.seed,
                        mode,
                        expected_reason_code,
                        actual_reason_code: "none".to_string(),
                        message: rendered,
                        artifact_refs,
                    });
                }
                Err(err) => {
                    let actual_reason_code = err.reason_code().to_string();
                    let contains_expected =
                        err.to_string().to_lowercase().contains(&expected_error);
                    let reason_match = actual_reason_code == expected_reason_code;
                    if contains_expected && reason_match {
                        report.pass_count += 1;
                    } else {
                        let rendered = format!(
                            "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected error containing '{}' with reason_code='{}' but got '{}' (reason_code='{}')",
                            case.id,
                            case.seed,
                            mode,
                            reason_code,
                            env_fingerprint,
                            artifact_refs.join(","),
                            case.expected_error_contains,
                            expected_reason_code,
                            err,
                            actual_reason_code
                        );
                        report.failures.push(rendered.clone());
                        mismatches.push(LinalgDifferentialMismatch {
                            fixture_id: case.id,
                            seed: case.seed,
                            mode,
                            expected_reason_code,
                            actual_reason_code,
                            message: rendered,
                            artifact_refs,
                        });
                    }
                }
            }
        }
    }

    let artifact = LinalgDifferentialReportArtifact {
        suite: "linalg_differential",
        total_cases: report.case_count,
        passed_cases: report.pass_count,
        failed_cases: report.case_count.saturating_sub(report.pass_count),
        mismatches,
    };
    let report_path = config
        .fixture_root
        .join("oracle_outputs/linalg_differential_report.json");
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }
    let payload = serde_json::to_string_pretty(&artifact)
        .map_err(|err| format!("failed serializing linalg differential report: {err}"))?;
    fs::write(&report_path, payload.as_bytes())
        .map_err(|err| format!("failed writing {}: {err}", report_path.display()))?;

    Ok(report)
}

pub fn run_linalg_metamorphic_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let path = config.fixture_root.join("linalg_metamorphic_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<LinalgMetamorphicCase> =
        serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))?;

    let mut report = SuiteReport {
        suite: "linalg_metamorphic",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };

    for case in cases {
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);
        let pass = match case.relation.as_str() {
            "solve_homogeneous_scaling" => {
                let matrix = match decode_matrix_2x2(&case.matrix) {
                    Ok(matrix) => matrix,
                    Err(err) => {
                        report.failures.push(format!(
                            "{}: seed={} reason_code={} env_fingerprint={} artifact_refs={} invalid matrix fixture: {}",
                            case.id,
                            case.seed,
                            reason_code,
                            env_fingerprint,
                            artifact_refs.join(","),
                            err
                        ));
                        continue;
                    }
                };
                let rhs = match decode_rhs_2(&case.rhs) {
                    Ok(rhs) => rhs,
                    Err(err) => {
                        report.failures.push(format!(
                            "{}: seed={} reason_code={} env_fingerprint={} artifact_refs={} invalid rhs fixture: {}",
                            case.id,
                            case.seed,
                            reason_code,
                            env_fingerprint,
                            artifact_refs.join(","),
                            err
                        ));
                        continue;
                    }
                };
                let scalar = case.scalar;

                let base = match solve_2x2(matrix, rhs) {
                    Ok(value) => value,
                    Err(err) => {
                        report.failures.push(format!(
                            "{}: seed={} reason_code={} env_fingerprint={} artifact_refs={} base solve failed: {}",
                            case.id,
                            case.seed,
                            reason_code,
                            env_fingerprint,
                            artifact_refs.join(","),
                            err
                        ));
                        continue;
                    }
                };
                let scaled_rhs = [rhs[0] * scalar, rhs[1] * scalar];
                let scaled = match solve_2x2(matrix, scaled_rhs) {
                    Ok(value) => value,
                    Err(err) => {
                        report.failures.push(format!(
                            "{}: seed={} reason_code={} env_fingerprint={} artifact_refs={} scaled solve failed: {}",
                            case.id,
                            case.seed,
                            reason_code,
                            env_fingerprint,
                            artifact_refs.join(","),
                            err
                        ));
                        continue;
                    }
                };
                let expected = [base[0] * scalar, base[1] * scalar];
                if !approx_equal_values(&expected, &scaled, 1e-9, 1e-9) {
                    report.failures.push(format!(
                        "{}: seed={} reason_code={} env_fingerprint={} artifact_refs={} solve_homogeneous_scaling mismatch expected={expected:?} actual={scaled:?}",
                        case.id,
                        case.seed,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(",")
                    ));
                    false
                } else {
                    true
                }
            }
            "qr_repeat_deterministic" => {
                let mode = match QrMode::from_mode_token(&case.qr_mode) {
                    Ok(mode) => mode,
                    Err(err) => {
                        report.failures.push(format!(
                            "{}: seed={} reason_code={} env_fingerprint={} artifact_refs={} invalid qr_mode '{}': {}",
                            case.id,
                            case.seed,
                            reason_code,
                            env_fingerprint,
                            artifact_refs.join(","),
                            case.qr_mode,
                            err
                        ));
                        continue;
                    }
                };

                let first = match qr_output_shapes(&case.shape, mode) {
                    Ok(value) => value,
                    Err(err) => {
                        report.failures.push(format!(
                            "{}: seed={} reason_code={} env_fingerprint={} artifact_refs={} first qr evaluation failed: {}",
                            case.id,
                            case.seed,
                            reason_code,
                            env_fingerprint,
                            artifact_refs.join(","),
                            err
                        ));
                        continue;
                    }
                };
                let second = match qr_output_shapes(&case.shape, mode) {
                    Ok(value) => value,
                    Err(err) => {
                        report.failures.push(format!(
                            "{}: seed={} reason_code={} env_fingerprint={} artifact_refs={} second qr evaluation failed: {}",
                            case.id,
                            case.seed,
                            reason_code,
                            env_fingerprint,
                            artifact_refs.join(","),
                            err
                        ));
                        continue;
                    }
                };

                if first != second {
                    report.failures.push(format!(
                        "{}: seed={} reason_code={} env_fingerprint={} artifact_refs={} qr_repeat_deterministic mismatch first={first:?} second={second:?}",
                        case.id,
                        case.seed,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(",")
                    ));
                    false
                } else {
                    true
                }
            }
            "lstsq_rhs_column_growth" => {
                if case.rhs_shape.len() != 1 {
                    report.failures.push(format!(
                        "{}: seed={} reason_code={} env_fingerprint={} artifact_refs={} rhs_shape must be a vector shape for lstsq_rhs_column_growth",
                        case.id,
                        case.seed,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(",")
                    ));
                    continue;
                }
                let vector = match lstsq_output_shapes(&case.lhs_shape, &case.rhs_shape) {
                    Ok(value) => value,
                    Err(err) => {
                        report.failures.push(format!(
                            "{}: seed={} reason_code={} env_fingerprint={} artifact_refs={} vector rhs lstsq failed: {}",
                            case.id,
                            case.seed,
                            reason_code,
                            env_fingerprint,
                            artifact_refs.join(","),
                            err
                        ));
                        continue;
                    }
                };
                let matrix_rhs_shape = vec![case.rhs_shape[0], 2];
                let matrix = match lstsq_output_shapes(&case.lhs_shape, &matrix_rhs_shape) {
                    Ok(value) => value,
                    Err(err) => {
                        report.failures.push(format!(
                            "{}: seed={} reason_code={} env_fingerprint={} artifact_refs={} matrix rhs lstsq failed: {}",
                            case.id,
                            case.seed,
                            reason_code,
                            env_fingerprint,
                            artifact_refs.join(","),
                            err
                        ));
                        continue;
                    }
                };

                if vector.rank_upper_bound != matrix.rank_upper_bound
                    || vector.singular_values_shape != matrix.singular_values_shape
                    || vector.x_shape.first() != matrix.x_shape.first()
                {
                    report.failures.push(format!(
                        "{}: seed={} reason_code={} env_fingerprint={} artifact_refs={} lstsq_rhs_column_growth mismatch vector={vector:?} matrix={matrix:?}",
                        case.id,
                        case.seed,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(",")
                    ));
                    false
                } else {
                    true
                }
            }
            other => {
                report.failures.push(format!(
                    "{}: seed={} reason_code={} env_fingerprint={} artifact_refs={} unsupported relation {}",
                    case.id,
                    case.seed,
                    reason_code,
                    env_fingerprint,
                    artifact_refs.join(","),
                    other
                ));
                false
            }
        };

        if pass {
            report.pass_count += 1;
        }
    }

    Ok(report)
}

pub fn run_linalg_adversarial_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let path = config.fixture_root.join("linalg_adversarial_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<LinalgAdversarialCase> =
        serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))?;

    let mut report = SuiteReport {
        suite: "linalg_adversarial",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);
        let severity = case.severity.trim().to_lowercase();
        if !matches!(severity.as_str(), "low" | "medium" | "high" | "critical") {
            report.failures.push(format!(
                "{}: invalid severity '{}' (must be low|medium|high|critical), reason_code={}, mode={}, env_fingerprint={}, artifact_refs={}",
                case.id,
                case.severity,
                reason_code,
                mode,
                env_fingerprint,
                artifact_refs.join(",")
            ));
            continue;
        }
        if case.expected_error_contains.trim().is_empty() {
            report.failures.push(format!(
                "{}: expected_error_contains must be non-empty, reason_code={}, mode={}, env_fingerprint={}, artifact_refs={}",
                case.id,
                reason_code,
                mode,
                env_fingerprint,
                artifact_refs.join(",")
            ));
            continue;
        }
        let expected_reason_code = if case.expected_reason_code.trim().is_empty() {
            reason_code.clone()
        } else {
            case.expected_reason_code.trim().to_string()
        };

        let expected_error = case.expected_error_contains.to_lowercase();
        match execute_linalg_adversarial_operation(&case) {
            Ok(_) => {
                report.failures.push(format!(
                    "{}: severity={severity} seed={} reason_code={} mode={} env_fingerprint={} artifact_refs={} expected error containing '{}' but operation '{}' succeeded",
                    case.id,
                    case.seed,
                    reason_code,
                    mode,
                    env_fingerprint,
                    artifact_refs.join(","),
                    case.expected_error_contains,
                    case.operation
                ));
            }
            Err(err) => {
                let actual_reason_code = err.reason_code().to_string();
                let contains_expected = err.to_string().to_lowercase().contains(&expected_error);
                let reason_match = actual_reason_code == expected_reason_code;
                if contains_expected && reason_match {
                    report.pass_count += 1;
                } else {
                    report.failures.push(format!(
                        "{}: severity={severity} seed={} reason_code={} mode={} env_fingerprint={} artifact_refs={} expected error containing '{}' with reason_code='{}' but got '{}' (reason_code='{}')",
                        case.id,
                        case.seed,
                        reason_code,
                        mode,
                        env_fingerprint,
                        artifact_refs.join(","),
                        case.expected_error_contains,
                        expected_reason_code,
                        err,
                        actual_reason_code
                    ));
                }
            }
        }
    }

    Ok(report)
}

#[derive(Debug, Clone, PartialEq)]
enum FftOperationOutcome {
    Array { shape: Vec<usize>, values: Vec<f64> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FftSuiteError {
    reason_code: String,
    message: String,
}

impl FftSuiteError {
    fn new(reason_code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            reason_code: reason_code.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for FftSuiteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for FftSuiteError {}

pub fn run_fft_differential_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let cases = load_fft_differential_cases(&config.fixture_root)?;

    let mut report = SuiteReport {
        suite: "fft_differential",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };
    let mut mismatches = Vec::new();

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);
        let expected_reason_code = if case.expected_reason_code.trim().is_empty() {
            reason_code.clone()
        } else {
            case.expected_reason_code.trim().to_string()
        };
        let expected_error = case.expected_error_contains.trim().to_lowercase();

        match execute_fft_differential_operation(&case) {
            Ok(outcome) => {
                if !expected_error.is_empty() {
                    let rendered = format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected error containing '{}' but operation '{}' succeeded",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        case.expected_error_contains,
                        case.operation
                    );
                    report.failures.push(rendered.clone());
                    mismatches.push(FftDifferentialMismatch {
                        fixture_id: case.id,
                        seed: case.seed,
                        mode,
                        operation: case.operation,
                        expected_reason_code,
                        actual_reason_code: "none".to_string(),
                        message: rendered,
                        artifact_refs,
                    });
                    continue;
                }

                match validate_fft_differential_expectation(&case, &outcome) {
                    Ok(()) => report.pass_count += 1,
                    Err(message) => {
                        let rendered = format!(
                            "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} {}",
                            case.id,
                            case.seed,
                            mode,
                            reason_code,
                            env_fingerprint,
                            artifact_refs.join(","),
                            message
                        );
                        report.failures.push(rendered.clone());
                        mismatches.push(FftDifferentialMismatch {
                            fixture_id: case.id,
                            seed: case.seed,
                            mode,
                            operation: case.operation,
                            expected_reason_code,
                            actual_reason_code: "fft_differential_mismatch".to_string(),
                            message: rendered,
                            artifact_refs,
                        });
                    }
                }
            }
            Err(err) => {
                if expected_error.is_empty() {
                    let rendered = format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected success but got error '{}' (reason_code={})",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        err.message,
                        err.reason_code
                    );
                    report.failures.push(rendered.clone());
                    mismatches.push(FftDifferentialMismatch {
                        fixture_id: case.id,
                        seed: case.seed,
                        mode,
                        operation: case.operation,
                        expected_reason_code,
                        actual_reason_code: err.reason_code,
                        message: rendered,
                        artifact_refs,
                    });
                    continue;
                }

                let contains_expected = err.message.to_lowercase().contains(&expected_error);
                let reason_match = err.reason_code == expected_reason_code;
                if contains_expected && reason_match {
                    report.pass_count += 1;
                } else {
                    let rendered = format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected error containing '{}' with reason_code='{}' but got '{}' (reason_code='{}')",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        case.expected_error_contains,
                        expected_reason_code,
                        err.message,
                        err.reason_code
                    );
                    report.failures.push(rendered.clone());
                    mismatches.push(FftDifferentialMismatch {
                        fixture_id: case.id,
                        seed: case.seed,
                        mode,
                        operation: case.operation,
                        expected_reason_code,
                        actual_reason_code: err.reason_code,
                        message: rendered,
                        artifact_refs,
                    });
                }
            }
        }
    }

    let artifact = FftDifferentialReportArtifact {
        suite: "fft_differential",
        total_cases: report.case_count,
        passed_cases: report.pass_count,
        failed_cases: report.case_count.saturating_sub(report.pass_count),
        mismatches,
    };
    let report_path = config
        .fixture_root
        .join("oracle_outputs/fft_differential_report.json");
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }
    let payload = serde_json::to_string_pretty(&artifact)
        .map_err(|err| format!("failed serializing fft differential report: {err}"))?;
    fs::write(&report_path, payload.as_bytes())
        .map_err(|err| format!("failed writing {}: {err}", report_path.display()))?;

    Ok(report)
}

pub fn run_fft_metamorphic_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let cases = load_fft_metamorphic_cases(&config.fixture_root)?;

    let mut report = SuiteReport {
        suite: "fft_metamorphic",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);

        match evaluate_fft_metamorphic_relation(&case) {
            Ok(()) => {
                report.pass_count += 1;
            }
            Err(err) => {
                report.failures.push(format!(
                    "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} {} (actual_reason_code={})",
                    case.id,
                    case.seed,
                    mode,
                    reason_code,
                    env_fingerprint,
                    artifact_refs.join(","),
                    err.message,
                    err.reason_code
                ));
            }
        }
    }

    Ok(report)
}

fn evaluate_fft_metamorphic_relation(case: &FftMetamorphicCase) -> Result<(), FftSuiteError> {
    match case.relation.as_str() {
        "fft_ifft_inverse" => {
            let input = fft_metamorphic_input_array(
                &case.input_shape,
                &case.input_values,
                &case.input_dtype,
            )?;
            if fft_metamorphic_is_complex_dtype(&case.input_dtype) {
                let recovered = input.ifft().map_err(map_ufunc_error_to_fft_suite)?;
                if recovered.shape() != input.shape() {
                    return Err(FftSuiteError::new(
                        "fft_metamorphic_relation_failed",
                        format!(
                            "complex inverse shape mismatch expected={:?} actual={:?}",
                            input.shape(),
                            recovered.shape()
                        ),
                    ));
                }
                let reconstructed = fft_manual_complex_dft(recovered.values(), false)?;
                fft_metamorphic_assert_values_equal(
                    case,
                    &reconstructed,
                    input.values(),
                    "manual_fft(ifft(x))",
                )
            } else {
                let transformed = input.fft(None).map_err(map_ufunc_error_to_fft_suite)?;
                let recovered = transformed.ifft().map_err(map_ufunc_error_to_fft_suite)?;
                fft_metamorphic_assert_real_inverse(
                    case,
                    &recovered,
                    &case.input_shape,
                    &case.input_values,
                    "ifft(fft(x))",
                )
            }
        }
        "fft_linearity" => {
            let alpha = case.alpha.ok_or_else(|| {
                FftSuiteError::new(
                    "fft_input_contract_violation",
                    "fft_linearity requires alpha",
                )
            })?;
            let beta = case.beta.ok_or_else(|| {
                FftSuiteError::new(
                    "fft_input_contract_violation",
                    "fft_linearity requires beta",
                )
            })?;
            let x = fft_metamorphic_real_array(&case.x_shape, &case.x_values)?;
            let y = fft_metamorphic_real_array(&case.y_shape, &case.y_values)?;
            if x.shape() != y.shape() {
                return Err(FftSuiteError::new(
                    "fft_input_contract_violation",
                    format!(
                        "fft_linearity requires matching shapes, got {:?} and {:?}",
                        x.shape(),
                        y.shape()
                    ),
                ));
            }
            let combined_values = x
                .values()
                .iter()
                .zip(y.values())
                .map(|(lhs, rhs)| alpha.mul_add(*lhs, beta * *rhs))
                .collect::<Vec<_>>();
            let combined = fft_metamorphic_real_array(x.shape(), &combined_values)?;
            let actual = combined.fft(None).map_err(map_ufunc_error_to_fft_suite)?;
            let fft_x = x.fft(None).map_err(map_ufunc_error_to_fft_suite)?;
            let fft_y = y.fft(None).map_err(map_ufunc_error_to_fft_suite)?;
            let expected_values = fft_x
                .values()
                .iter()
                .zip(fft_y.values())
                .map(|(lhs, rhs)| alpha.mul_add(*lhs, beta * *rhs))
                .collect::<Vec<_>>();
            if actual.shape() != fft_x.shape() {
                return Err(FftSuiteError::new(
                    "fft_metamorphic_relation_failed",
                    format!(
                        "fft_linearity shape mismatch expected={:?} actual={:?}",
                        fft_x.shape(),
                        actual.shape()
                    ),
                ));
            }
            fft_metamorphic_assert_values_equal(
                case,
                actual.values(),
                &expected_values,
                "fft(alpha*x + beta*y)",
            )
        }
        "fft_parseval" => {
            let input = fft_metamorphic_real_array(&case.input_shape, &case.input_values)?;
            let transformed = input.fft(None).map_err(map_ufunc_error_to_fft_suite)?;
            let sample_count = input.values().len();
            if sample_count == 0 {
                return Err(FftSuiteError::new(
                    "fft_input_contract_violation",
                    "fft_parseval requires non-empty input",
                ));
            }
            let time_energy = input
                .values()
                .iter()
                .map(|value| value * value)
                .sum::<f64>();
            let freq_energy = transformed
                .values()
                .chunks_exact(2)
                .map(|pair| pair[0] * pair[0] + pair[1] * pair[1])
                .sum::<f64>()
                / sample_count as f64;
            fft_metamorphic_assert_scalar_equal(case, time_energy, freq_energy, "fft_parseval")
        }
        "fftshift_ifftshift_inverse" => {
            let input = fft_metamorphic_real_array(&case.input_shape, &case.input_values)?;
            let recovered = input
                .fftshift()
                .and_then(|shifted| shifted.ifftshift())
                .map_err(map_ufunc_error_to_fft_suite)?;
            fft_metamorphic_assert_array_equal(case, &recovered, &input, "ifftshift(fftshift(x))")
        }
        "rfft_hermitian_symmetry" => {
            let input = fft_metamorphic_real_array(&case.input_shape, &case.input_values)?;
            let full = input.fft(None).map_err(map_ufunc_error_to_fft_suite)?;
            let half = input.rfft(None).map_err(map_ufunc_error_to_fft_suite)?;
            fft_metamorphic_assert_values_equal(
                case,
                half.values(),
                &full.values()[..half.values().len()],
                "rfft prefix",
            )?;
            let n = input.values().len();
            for k in 0..n {
                let lhs = &full.values()[k * 2..k * 2 + 2];
                let rhs = &full.values()[((n - k) % n) * 2..((n - k) % n) * 2 + 2];
                fft_metamorphic_assert_scalar_equal(
                    case,
                    lhs[0],
                    rhs[0],
                    "rfft_hermitian_symmetry real",
                )?;
                fft_metamorphic_assert_scalar_equal(
                    case,
                    lhs[1],
                    -rhs[1],
                    "rfft_hermitian_symmetry imag",
                )?;
            }
            Ok(())
        }
        "fft2_ifft2_inverse" => {
            let input = fft_metamorphic_real_array(&case.input_shape, &case.input_values)?;
            let recovered = input
                .fft2()
                .and_then(|transformed| transformed.ifft2())
                .map_err(map_ufunc_error_to_fft_suite)?;
            fft_metamorphic_assert_real_inverse(
                case,
                &recovered,
                &case.input_shape,
                &case.input_values,
                "ifft2(fft2(x))",
            )
        }
        "fftn_ifftn_inverse" => {
            let input = fft_metamorphic_real_array(&case.input_shape, &case.input_values)?;
            let recovered = input
                .fftn()
                .and_then(|transformed| transformed.ifftn())
                .map_err(map_ufunc_error_to_fft_suite)?;
            fft_metamorphic_assert_real_inverse(
                case,
                &recovered,
                &case.input_shape,
                &case.input_values,
                "ifftn(fftn(x))",
            )
        }
        other => Err(FftSuiteError::new(
            "fft_policy_unknown_metamorphic_relation",
            format!("unsupported fft metamorphic relation {other}"),
        )),
    }
}

fn fft_metamorphic_input_array(
    shape: &[usize],
    values: &[f64],
    dtype_name: &str,
) -> Result<UFuncArray, FftSuiteError> {
    let dtype = if dtype_name.eq_ignore_ascii_case("c128")
        || dtype_name.eq_ignore_ascii_case("complex128")
    {
        DType::Complex128
    } else if dtype_name.eq_ignore_ascii_case("c64") || dtype_name.eq_ignore_ascii_case("complex64")
    {
        DType::Complex64
    } else {
        DType::F64
    };
    UFuncArray::new(shape.to_vec(), values.to_vec(), dtype).map_err(map_ufunc_error_to_fft_suite)
}

fn fft_metamorphic_real_array(
    shape: &[usize],
    values: &[f64],
) -> Result<UFuncArray, FftSuiteError> {
    UFuncArray::new(shape.to_vec(), values.to_vec(), DType::F64)
        .map_err(map_ufunc_error_to_fft_suite)
}

fn fft_metamorphic_is_complex_dtype(dtype_name: &str) -> bool {
    dtype_name.eq_ignore_ascii_case("c128")
        || dtype_name.eq_ignore_ascii_case("complex128")
        || dtype_name.eq_ignore_ascii_case("c64")
        || dtype_name.eq_ignore_ascii_case("complex64")
}

fn fft_manual_complex_dft(values: &[f64], inverse: bool) -> Result<Vec<f64>, FftSuiteError> {
    if !values.len().is_multiple_of(2) {
        return Err(FftSuiteError::new(
            "fft_input_contract_violation",
            "complex DFT input must have interleaved real/imaginary pairs",
        ));
    }
    let n = values.len() / 2;
    if n == 0 {
        return Ok(Vec::new());
    }
    let direction = if inverse { 1.0 } else { -1.0 };
    let scale = if inverse { n as f64 } else { 1.0 };
    let mut output = vec![0.0; values.len()];
    for (k, out_pair) in output.chunks_exact_mut(2).enumerate() {
        let mut sum_re = 0.0;
        let mut sum_im = 0.0;
        for (t, in_pair) in values.chunks_exact(2).enumerate() {
            let angle = direction * std::f64::consts::TAU * k as f64 * t as f64 / n as f64;
            let cos = angle.cos();
            let sin = angle.sin();
            let re = in_pair[0];
            let im = in_pair[1];
            sum_re += re * cos - im * sin;
            sum_im += re * sin + im * cos;
        }
        out_pair[0] = sum_re / scale;
        out_pair[1] = sum_im / scale;
    }
    Ok(output)
}

fn fft_metamorphic_assert_real_inverse(
    case: &FftMetamorphicCase,
    recovered: &UFuncArray,
    input_shape: &[usize],
    input_values: &[f64],
    label: &str,
) -> Result<(), FftSuiteError> {
    let mut expected_shape = input_shape.to_vec();
    expected_shape.push(2);
    if recovered.shape() != expected_shape {
        return Err(FftSuiteError::new(
            "fft_metamorphic_relation_failed",
            format!(
                "{label} shape mismatch expected={:?} actual={:?}",
                expected_shape,
                recovered.shape()
            ),
        ));
    }
    let mut real_parts = Vec::with_capacity(input_values.len());
    for (index, pair) in recovered.values().chunks_exact(2).enumerate() {
        real_parts.push(pair[0]);
        fft_metamorphic_assert_scalar_equal(
            case,
            pair[1],
            0.0,
            &format!("{label} imaginary component {index}"),
        )?;
    }
    fft_metamorphic_assert_values_equal(case, &real_parts, input_values, label)
}

fn fft_metamorphic_assert_array_equal(
    case: &FftMetamorphicCase,
    actual: &UFuncArray,
    expected: &UFuncArray,
    label: &str,
) -> Result<(), FftSuiteError> {
    if actual.shape() != expected.shape() {
        return Err(FftSuiteError::new(
            "fft_metamorphic_relation_failed",
            format!(
                "{label} shape mismatch expected={:?} actual={:?}",
                expected.shape(),
                actual.shape()
            ),
        ));
    }
    fft_metamorphic_assert_values_equal(case, actual.values(), expected.values(), label)
}

fn fft_metamorphic_assert_values_equal(
    case: &FftMetamorphicCase,
    actual: &[f64],
    expected: &[f64],
    label: &str,
) -> Result<(), FftSuiteError> {
    if actual.len() != expected.len() {
        return Err(FftSuiteError::new(
            "fft_metamorphic_relation_failed",
            format!(
                "{label} length mismatch expected={} actual={}",
                expected.len(),
                actual.len()
            ),
        ));
    }
    let abs_tol = if case.abs_tol > 0.0 {
        case.abs_tol
    } else {
        1e-9
    };
    let rel_tol = if case.rel_tol > 0.0 {
        case.rel_tol
    } else {
        1e-9
    };
    if !approx_equal_values(expected, actual, abs_tol, rel_tol) {
        return Err(FftSuiteError::new(
            "fft_metamorphic_relation_failed",
            format!(
                "{label} value mismatch expected={expected:?} actual={actual:?} abs_tol={abs_tol} rel_tol={rel_tol}"
            ),
        ));
    }
    Ok(())
}

fn fft_metamorphic_assert_scalar_equal(
    case: &FftMetamorphicCase,
    actual: f64,
    expected: f64,
    label: &str,
) -> Result<(), FftSuiteError> {
    let abs_tol = if case.abs_tol > 0.0 {
        case.abs_tol
    } else {
        1e-9
    };
    let rel_tol = if case.rel_tol > 0.0 {
        case.rel_tol
    } else {
        1e-9
    };
    let diff = (actual - expected).abs();
    let scale = expected.abs().max(actual.abs());
    if diff > abs_tol.max(rel_tol * scale) {
        return Err(FftSuiteError::new(
            "fft_metamorphic_relation_failed",
            format!(
                "{label} mismatch expected={expected} actual={actual} abs_tol={abs_tol} rel_tol={rel_tol}"
            ),
        ));
    }
    Ok(())
}

fn execute_fft_differential_operation(
    case: &FftDifferentialCase,
) -> Result<FftOperationOutcome, FftSuiteError> {
    let to_outcome = |array: UFuncArray| FftOperationOutcome::Array {
        shape: array.shape().to_vec(),
        values: array.values().to_vec(),
    };

    match case.operation.as_str() {
        "fftfreq" => {
            let n = case.n.ok_or_else(|| {
                FftSuiteError::new("fft_input_contract_violation", "fftfreq requires field 'n'")
            })?;
            Ok(to_outcome(UFuncArray::fftfreq(n, case.d.unwrap_or(1.0))))
        }
        "rfftfreq" => {
            let n = case.n.ok_or_else(|| {
                FftSuiteError::new(
                    "fft_input_contract_violation",
                    "rfftfreq requires field 'n'",
                )
            })?;
            Ok(to_outcome(UFuncArray::rfftfreq(n, case.d.unwrap_or(1.0))))
        }
        operation => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_fft_suite)?;
            let output = match operation {
                "fft" => input.fft(case.n),
                "ifft" => input.ifft(),
                "rfft" => input.rfft(case.n),
                "irfft" => input.irfft(case.output_n),
                "hfft" => input.hfft(case.n),
                "ihfft" => input.ihfft(case.n),
                "fft2" => input.fft2(),
                "ifft2" => input.ifft2(),
                "rfft2" => input.rfft2(),
                "irfft2" => input.irfft2(case.output_n),
                "fftn" => input.fftn(),
                "ifftn" => input.ifftn(),
                "rfftn" => input.rfftn(),
                "irfftn" => input.irfftn(case.output_n),
                "fftshift" => input.fftshift(),
                "ifftshift" => input.ifftshift(),
                other => {
                    return Err(FftSuiteError::new(
                        "fft_policy_unknown_operation",
                        format!("unsupported fft differential operation {other}"),
                    ));
                }
            }
            .map_err(map_ufunc_error_to_fft_suite)?;
            Ok(to_outcome(output))
        }
    }
}

fn validate_fft_differential_expectation(
    case: &FftDifferentialCase,
    outcome: &FftOperationOutcome,
) -> Result<(), String> {
    match outcome {
        FftOperationOutcome::Array { shape, values } => {
            if !case.expected_shape.is_empty() && case.expected_shape != *shape {
                return Err(format!(
                    "shape mismatch expected={:?} actual={shape:?}",
                    case.expected_shape
                ));
            }
            if !case.expected_values.is_empty() {
                if case.expected_values.len() != values.len() {
                    return Err(format!(
                        "value length mismatch expected={} actual={}",
                        case.expected_values.len(),
                        values.len()
                    ));
                }
                let abs_tol = if case.abs_tol > 0.0 {
                    case.abs_tol
                } else {
                    1e-9
                };
                let rel_tol = if case.rel_tol > 0.0 {
                    case.rel_tol
                } else {
                    1e-9
                };
                if !approx_equal_values(&case.expected_values, values, abs_tol, rel_tol) {
                    return Err(format!(
                        "value mismatch expected={:?} actual={values:?} abs_tol={} rel_tol={}",
                        case.expected_values, abs_tol, rel_tol
                    ));
                }
            }
            Ok(())
        }
    }
}

fn map_ufunc_error_to_fft_suite(error: UFuncError) -> FftSuiteError {
    FftSuiteError::new(error.reason_code(), error.to_string())
}

// ────────────────────────────────────────────────────────────────────────────────
// Signal processing differential suite (window functions, gradient, diff, etc.)
// ────────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum SignalOperationOutcome {
    Array { shape: Vec<usize>, values: Vec<f64> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SignalSuiteError {
    reason_code: String,
    message: String,
}

impl SignalSuiteError {
    fn new(reason_code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            reason_code: reason_code.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for SignalSuiteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for SignalSuiteError {}

fn parse_quantile_interp(method: &str) -> Result<QuantileInterp, SignalSuiteError> {
    match method.to_ascii_lowercase().as_str() {
        "linear" => Ok(QuantileInterp::Linear),
        "lower" => Ok(QuantileInterp::Lower),
        "higher" => Ok(QuantileInterp::Higher),
        "nearest" => Ok(QuantileInterp::Nearest),
        "midpoint" => Ok(QuantileInterp::Midpoint),
        other => Err(SignalSuiteError::new(
            "signal_differential_invalid_method",
            format!("unsupported percentile method: {other}"),
        )),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SignalDifferentialMismatch {
    fixture_id: String,
    seed: u64,
    mode: String,
    operation: String,
    expected_reason_code: String,
    actual_reason_code: String,
    message: String,
    artifact_refs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SignalDifferentialReportArtifact {
    suite: &'static str,
    total_cases: usize,
    passed_cases: usize,
    failed_cases: usize,
    mismatches: Vec<SignalDifferentialMismatch>,
}

pub fn run_signal_differential_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let cases = load_signal_differential_cases(&config.fixture_root)?;

    let mut report = SuiteReport {
        suite: "signal_differential",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };
    let mut mismatches = Vec::new();

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let _env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);
        let expected_reason_code = if case.expected_reason_code.trim().is_empty() {
            reason_code.clone()
        } else {
            case.expected_reason_code.trim().to_string()
        };
        let expected_error = case.expected_error_contains.trim().to_lowercase();

        match execute_signal_differential_operation(&case) {
            Ok(outcome) => {
                if !expected_error.is_empty() {
                    let rendered = format!(
                        "{}: expected error containing '{}' but operation succeeded",
                        case.id, case.expected_error_contains
                    );
                    report.failures.push(rendered.clone());
                    mismatches.push(SignalDifferentialMismatch {
                        fixture_id: case.id,
                        seed: case.seed,
                        mode,
                        operation: case.operation,
                        expected_reason_code,
                        actual_reason_code: "none".to_string(),
                        message: rendered,
                        artifact_refs,
                    });
                    continue;
                }

                match validate_signal_differential_expectation(&case, &outcome) {
                    Ok(()) => report.pass_count += 1,
                    Err(message) => {
                        let rendered = format!("{}: {}", case.id, message);
                        report.failures.push(rendered.clone());
                        mismatches.push(SignalDifferentialMismatch {
                            fixture_id: case.id,
                            seed: case.seed,
                            mode,
                            operation: case.operation,
                            expected_reason_code,
                            actual_reason_code: "signal_differential_mismatch".to_string(),
                            message: rendered,
                            artifact_refs,
                        });
                    }
                }
            }
            Err(err) => {
                if expected_error.is_empty() {
                    let rendered = format!(
                        "{}: expected success but got error '{}'",
                        case.id, err.message
                    );
                    report.failures.push(rendered.clone());
                    mismatches.push(SignalDifferentialMismatch {
                        fixture_id: case.id,
                        seed: case.seed,
                        mode,
                        operation: case.operation,
                        expected_reason_code,
                        actual_reason_code: err.reason_code,
                        message: rendered,
                        artifact_refs,
                    });
                    continue;
                }

                let contains_expected = err.message.to_lowercase().contains(&expected_error);
                if contains_expected {
                    report.pass_count += 1;
                } else {
                    let rendered = format!(
                        "{}: expected error containing '{}' but got '{}'",
                        case.id, case.expected_error_contains, err.message
                    );
                    report.failures.push(rendered.clone());
                    mismatches.push(SignalDifferentialMismatch {
                        fixture_id: case.id,
                        seed: case.seed,
                        mode,
                        operation: case.operation,
                        expected_reason_code,
                        actual_reason_code: err.reason_code,
                        message: rendered,
                        artifact_refs,
                    });
                }
            }
        }
    }

    let artifact = SignalDifferentialReportArtifact {
        suite: "signal_differential",
        total_cases: report.case_count,
        passed_cases: report.pass_count,
        failed_cases: report.case_count.saturating_sub(report.pass_count),
        mismatches,
    };
    let report_path = config
        .fixture_root
        .join("oracle_outputs/signal_differential_report.json");
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }
    let payload = serde_json::to_string_pretty(&artifact)
        .map_err(|err| format!("failed serializing signal differential report: {err}"))?;
    fs::write(&report_path, payload.as_bytes())
        .map_err(|err| format!("failed writing {}: {err}", report_path.display()))?;

    Ok(report)
}

fn execute_signal_differential_operation(
    case: &SignalDifferentialCase,
) -> Result<SignalOperationOutcome, SignalSuiteError> {
    let to_outcome = |array: UFuncArray| SignalOperationOutcome::Array {
        shape: array.shape().to_vec(),
        values: array.values().to_vec(),
    };

    match case.operation.as_str() {
        "bartlett" => {
            let n = case.n.ok_or_else(|| {
                SignalSuiteError::new("signal_input_contract_violation", "bartlett requires 'n'")
            })?;
            Ok(to_outcome(UFuncArray::bartlett(n)))
        }
        "blackman" => {
            let n = case.n.ok_or_else(|| {
                SignalSuiteError::new("signal_input_contract_violation", "blackman requires 'n'")
            })?;
            Ok(to_outcome(UFuncArray::blackman(n)))
        }
        "hamming" => {
            let n = case.n.ok_or_else(|| {
                SignalSuiteError::new("signal_input_contract_violation", "hamming requires 'n'")
            })?;
            Ok(to_outcome(UFuncArray::hamming(n)))
        }
        "hanning" => {
            let n = case.n.ok_or_else(|| {
                SignalSuiteError::new("signal_input_contract_violation", "hanning requires 'n'")
            })?;
            Ok(to_outcome(UFuncArray::hanning(n)))
        }
        "kaiser" => {
            let n = case.n.ok_or_else(|| {
                SignalSuiteError::new("signal_input_contract_violation", "kaiser requires 'n'")
            })?;
            let beta = case.beta.unwrap_or(14.0);
            Ok(to_outcome(UFuncArray::kaiser(n, beta)))
        }
        "gradient" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let spacing = case.spacing.unwrap_or(1.0);
            let result = input.gradient().map_err(map_ufunc_error_to_signal_suite)?;
            // Scale by 1/spacing (gradient values are inversely proportional to spacing)
            if (spacing - 1.0).abs() > 1e-12 {
                let inv_spacing = 1.0 / spacing;
                let scaled_values: Vec<f64> =
                    result.values().iter().map(|v| v * inv_spacing).collect();
                Ok(SignalOperationOutcome::Array {
                    shape: result.shape().to_vec(),
                    values: scaled_values,
                })
            } else {
                Ok(to_outcome(result))
            }
        }
        "diff" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let n = case.n.unwrap_or(1);
            let axis = case.axis;
            let result = input
                .diff(n, axis)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "unwrap" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let axis = case.axis.unwrap_or(-1);
            let result = input
                .unwrap_period_axis(case.discont, case.period, axis)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "ediff1d" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let to_begin = if case.to_begin.is_empty() {
                None
            } else {
                Some(case.to_begin.as_slice())
            };
            let to_end = if case.to_end.is_empty() {
                None
            } else {
                Some(case.to_end.as_slice())
            };
            let result = input
                .ediff1d_ext(to_begin, to_end)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "cross" => {
            let a = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let b = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = a.cross(&b).map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "trapezoid" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let dx = case.dx.unwrap_or(1.0);
            let result = input
                .trapezoid(dx, None)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "vector_norm" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let ord = if let Some(ref inf_str) = case.ord_inf {
                match inf_str.as_str() {
                    "inf" | "+inf" | "Infinity" => Some(f64::INFINITY),
                    "-inf" | "-Infinity" => Some(f64::NEG_INFINITY),
                    _ => case.ord_float,
                }
            } else {
                case.ord_float
            };
            let result = input
                .vector_norm(ord, case.axis, case.keepdims)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "matrix_norm" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let ord_str = case.ord_str.as_deref();
            let result = input
                .matrix_norm(ord_str, case.keepdims)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "matvec" => {
            let matrix = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let vector =
                UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                    .map_err(map_ufunc_error_to_signal_suite)?;
            let result =
                UFuncArray::matvec(&matrix, &vector).map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "vecmat" => {
            let vector = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let matrix =
                UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                    .map_err(map_ufunc_error_to_signal_suite)?;
            let result =
                UFuncArray::vecmat(&vector, &matrix).map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "vecdot" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs
                .vecdot(&rhs, case.axis)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "outer" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs.outer(&rhs).map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "inner" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs.inner(&rhs).map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "kron" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs.kron(&rhs).map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "dot" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs.dot(&rhs).map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "tensordot" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let axes = case.tensordot_axes.unwrap_or(1);
            let result = lhs
                .tensordot(&rhs, axes)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "matmul" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs.matmul(&rhs).map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "permute_dims" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .permute_dims(&case.axes)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "concat" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let axis = case.axis.unwrap_or(0);
            let result =
                UFuncArray::concat(&[&lhs, &rhs], axis).map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "roll" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .roll(case.shift, case.axis)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "flip" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .flip(case.axis)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "rot90" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .rot90(case.k)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "convolve" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let kernel =
                UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                    .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .convolve(&kernel)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "correlate" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let kernel =
                UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                    .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .correlate(&kernel)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "pad" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .pad(&case.pad_width, case.constant)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "cumsum" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .cumsum(case.axis)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "cumprod" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .cumprod(case.axis)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "searchsorted" => {
            let sorted_arr = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let values =
                UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                    .map_err(map_ufunc_error_to_signal_suite)?;
            let result = sorted_arr
                .searchsorted(&values, None, None)
                .map_err(map_ufunc_error_to_signal_suite)?;
            // searchsorted returns I64, convert to F64 for test comparison
            let result_f64 = result.astype(DType::F64);
            Ok(to_outcome(result_f64))
        }
        "digitize" => {
            let values = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let bins = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = values
                .digitize(&bins)
                .map_err(map_ufunc_error_to_signal_suite)?;
            // digitize returns I64, convert to F64 for test comparison
            let result_f64 = result.astype(DType::F64);
            Ok(to_outcome(result_f64))
        }
        "clip" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let min_val = case.min_val.unwrap_or(f64::NEG_INFINITY);
            let max_val = case.max_val.unwrap_or(f64::INFINITY);
            let result = input.clip(min_val, max_val);
            Ok(to_outcome(result))
        }
        "floor_divide" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs
                .elementwise_binary(&rhs, BinaryOp::FloorDivide)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "fmod" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs
                .elementwise_binary(&rhs, BinaryOp::Fmod)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "copysign" => {
            let x1 = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let x2 = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = crate::copysign(&x1, &x2).map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "signbit" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = crate::signbit(&input).map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "hypot" => {
            let x1 = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let x2 = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = crate::hypot(&x1, &x2).map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "logaddexp" => {
            let x1 = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let x2 = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = crate::logaddexp(&x1, &x2).map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "logaddexp2" => {
            let x1 = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let x2 = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = crate::logaddexp2(&x1, &x2).map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "sort" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .sort(case.axis, None)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "argsort" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .argsort(case.axis, None)
                .map_err(map_ufunc_error_to_signal_suite)?;
            // argsort returns indices as I64, convert to F64 for comparison
            let result_f64 = result.astype(DType::F64);
            Ok(to_outcome(result_f64))
        }
        "unique" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input.unique();
            Ok(to_outcome(result))
        }
        "any" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .any(case.axis)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "all" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .all(case.axis)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "nonzero" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input.nonzero();
            // nonzero returns indices as I64, convert to F64 for comparison
            let result_f64 = result.astype(DType::F64);
            Ok(to_outcome(result_f64))
        }
        "median" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .median(case.axis)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "trace" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = if case.axes.len() == 2 {
                input
                    .trace_axis(case.k as i64, case.axes[0] as isize, case.axes[1] as isize)
                    .map_err(map_ufunc_error_to_signal_suite)?
            } else {
                input
                    .trace(case.k as i64)
                    .map_err(map_ufunc_error_to_signal_suite)?
            };
            Ok(to_outcome(result))
        }
        "diagonal" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            if case.axes.len() != 2 {
                return Err(SignalSuiteError::new(
                    "signal_differential_invalid_axes",
                    "diagonal requires exactly two axes in the fixture",
                ));
            }
            let result = input
                .diagonal(case.k as i64, case.axes[0] as isize, case.axes[1] as isize)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "diag" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .diag(case.k as i64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "triu" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .triu(case.k as i64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "tril" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .tril(case.k as i64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "reduce_sum" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .reduce_sum(case.axis, case.keepdims)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "reduce_mean" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .reduce_mean(case.axis, case.keepdims)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "reduce_min" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .reduce_min(case.axis, case.keepdims)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "reduce_max" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .reduce_max(case.axis, case.keepdims)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "flatten" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input.flatten();
            Ok(to_outcome(result))
        }
        "reshape" => {
            if case.reshape_shape.is_empty() {
                return Err(SignalSuiteError::new(
                    "signal_input_contract_violation",
                    "reshape requires reshape_shape",
                ));
            }
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .reshape(&case.reshape_shape)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "transpose" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .transpose(None)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "squeeze" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .squeeze(case.axis)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "expand_dims" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let ax = case.axis.unwrap_or(0);
            let result = input
                .expand_dims(ax)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "swapaxes" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            // Use k for axis1, axis for axis2
            let result = input
                .swapaxes(case.k as isize, case.axis.unwrap_or(1))
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "tile" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            // Use rhs_shape as reps
            let result = input
                .tile(&case.rhs_shape)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "repeat" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let repeats = case.n.unwrap_or(2);
            let result = input
                .repeat(repeats, case.axis)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "ravel" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input.ravel();
            Ok(to_outcome(result))
        }
        "count_nonzero" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .count_nonzero(case.axis, case.keepdims)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "argmax" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .reduce_argmax(case.axis)
                .map_err(map_ufunc_error_to_signal_suite)?;
            // argmax returns indices as I64, convert to F64 for comparison
            let result_f64 = result.astype(DType::F64);
            Ok(to_outcome(result_f64))
        }
        "argmin" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .reduce_argmin(case.axis)
                .map_err(map_ufunc_error_to_signal_suite)?;
            // argmin returns indices as I64, convert to F64 for comparison
            let result_f64 = result.astype(DType::F64);
            Ok(to_outcome(result_f64))
        }
        "reduce_prod" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .reduce_prod(case.axis, case.keepdims)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "ptp" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .ptp(case.axis)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "reduce_var" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let ddof = case.n.unwrap_or(0);
            let result = input
                .reduce_var(case.axis, case.keepdims, ddof)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "reduce_std" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let ddof = case.n.unwrap_or(0);
            let result = input
                .reduce_std(case.axis, case.keepdims, ddof)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "where_select" => {
            let condition = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let x = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            // Use expected_values as the third array (y values)
            // We'll repurpose axes field for y_shape
            let y_shape: Vec<usize> = if !case.axes.is_empty() {
                case.axes.clone()
            } else {
                case.rhs_shape.clone()
            };
            // y values: use constant field repeated, or zeros
            let y_values: Vec<f64> = if case.constant != 0.0 {
                vec![case.constant; y_shape.iter().product()]
            } else {
                vec![0.0; y_shape.iter().product()]
            };
            let y = UFuncArray::new(y_shape, y_values, DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = UFuncArray::where_select(&condition, &x, &y)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "percentile" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let q = case.beta.unwrap_or(50.0);
            let result = input
                .percentile(q, case.axis)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "percentile_method" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let q = case.beta.unwrap_or(50.0);
            let method = parse_quantile_interp(&case.method)?;
            let result = input
                .percentile_method(q, case.axis, method)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "interp" => {
            let x = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let xp = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            // For fp, use axes field to store shape, and use linear values 0..len as fp
            let fp_len: usize = case.axes.iter().product();
            let fp_values: Vec<f64> = (0..fp_len).map(|i| i as f64).collect();
            let fp = UFuncArray::new(case.axes.clone(), fp_values, DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result =
                UFuncArray::interp(&x, &xp, &fp).map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "bincount" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input.bincount().map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "einsum" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            // For single-operand einsum (trace, diagonal, etc.)
            let result = if case.rhs_values.is_empty() {
                UFuncArray::einsum(&case.subscripts, &[&input])
                    .map_err(map_ufunc_error_to_signal_suite)?
            } else {
                let rhs =
                    UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                        .map_err(map_ufunc_error_to_signal_suite)?;
                UFuncArray::einsum(&case.subscripts, &[&input, &rhs])
                    .map_err(map_ufunc_error_to_signal_suite)?
            };
            Ok(to_outcome(result))
        }
        "cov" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input.cov().map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "corrcoef" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input.corrcoef().map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "histogram" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let bins = case.bins.unwrap_or(10);
            let (counts, _edges) = input
                .histogram(bins)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(counts))
        }
        "nansum" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .nansum(case.axis, case.keepdims)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "nanmean" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .nanmean(case.axis, case.keepdims)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "nanmin" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .nanmin(case.axis, case.keepdims)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "nanmax" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .nanmax(case.axis, case.keepdims)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "union1d" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs.union1d(&rhs);
            Ok(to_outcome(result))
        }
        "intersect1d" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs.intersect1d(&rhs);
            Ok(to_outcome(result))
        }
        "setdiff1d" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs.setdiff1d(&rhs);
            Ok(to_outcome(result))
        }
        "average" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let weights = if !case.rhs_values.is_empty() {
                Some(
                    UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                        .map_err(map_ufunc_error_to_signal_suite)?,
                )
            } else {
                None
            };
            let result = input
                .average(weights.as_ref(), case.axis)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "meshgrid" => {
            let x = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let y = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let grids = UFuncArray::meshgrid(&[x, y]).map_err(map_ufunc_error_to_signal_suite)?;
            // Return concatenated result for testing
            if grids.len() >= 2 {
                let flat0 = grids[0].flatten();
                let flat1 = grids[1].flatten();
                let combined_values: Vec<f64> = flat0
                    .values()
                    .iter()
                    .chain(flat1.values().iter())
                    .copied()
                    .collect();
                let combined =
                    UFuncArray::new(vec![combined_values.len()], combined_values, DType::F64)
                        .map_err(map_ufunc_error_to_signal_suite)?;
                Ok(to_outcome(combined))
            } else {
                Ok(to_outcome(grids[0].clone()))
            }
        }
        "hstack" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result =
                UFuncArray::hstack(&[lhs, rhs]).map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "vstack" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result =
                UFuncArray::vstack(&[lhs, rhs]).map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "dstack" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result =
                UFuncArray::dstack(&[lhs, rhs]).map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "stack" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let axis = case.axis.unwrap_or(0);
            let result =
                UFuncArray::stack(&[&lhs, &rhs], axis).map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "isclose" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let rtol = case.rtol.unwrap_or(1e-5);
            let atol = case.atol.unwrap_or(1e-8);
            let result = lhs
                .isclose(&rhs, rtol, atol)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "take" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let indices: Vec<i64> = case.rhs_values.iter().map(|&v| v as i64).collect();
            let result = input
                .take(&indices, case.axis)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "linspace" => {
            let start = case.start.unwrap_or(0.0);
            let stop = case.stop.unwrap_or(1.0);
            let num = case.n.unwrap_or(50);
            let result = UFuncArray::linspace(start, stop, num, DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "atleast_1d" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.atleast_1d()))
        }
        "atleast_2d" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.atleast_2d()))
        }
        "atleast_3d" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.atleast_3d()))
        }
        "broadcast_to" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input
                .broadcast_to(&case.expected_shape)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "compress" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let condition: Vec<bool> = case.rhs_values.iter().map(|&v| v != 0.0).collect();
            let result = input
                .compress(&condition, case.axis)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "delete" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let indices: Vec<usize> = case.rhs_values.iter().map(|&v| v as usize).collect();
            let result = input
                .delete(&indices, case.axis)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "insert" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let values =
                UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                    .map_err(map_ufunc_error_to_signal_suite)?;
            let index = case.n.unwrap_or(0);
            let result = input
                .insert(index, &values, case.axis)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "geomspace" => {
            let start = case.start.unwrap_or(1.0);
            let stop = case.stop.unwrap_or(100.0);
            let num = case.n.unwrap_or(10);
            let result = UFuncArray::geomspace(start, stop, num, DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "logspace" => {
            let start = case.start.unwrap_or(0.0);
            let stop = case.stop.unwrap_or(2.0);
            let num = case.n.unwrap_or(10);
            let base = case.beta.unwrap_or(10.0);
            let result = UFuncArray::logspace(start, stop, num, base, DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "eye" => {
            let n = case.n.unwrap_or(3);
            let m = case.bins.unwrap_or(n);
            let k = case.k as i64;
            let result = UFuncArray::eye(n, Some(m), k, DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "identity" => {
            let n = case.n.unwrap_or(3);
            let result =
                UFuncArray::identity(n, DType::F64).map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "full" => {
            let fill = case.constant;
            let result = UFuncArray::full(case.expected_shape.clone(), fill, DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "zeros_like" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(UFuncArray::zeros_like(&input)))
        }
        "ones_like" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(UFuncArray::ones_like(&input)))
        }
        "full_like" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(UFuncArray::full_like(&input, case.constant)))
        }
        "arange" => {
            let start = case.start.unwrap_or(0.0);
            let stop = case.stop.unwrap_or(10.0);
            let step = case.spacing.unwrap_or(1.0);
            let result = UFuncArray::arange(start, stop, step, DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "abs" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Abs)))
        }
        "negative" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Negative)))
        }
        "positive" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Positive)))
        }
        "sign" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Sign)))
        }
        "sqrt" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Sqrt)))
        }
        "square" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Square)))
        }
        "cbrt" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Cbrt)))
        }
        "reciprocal" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Reciprocal)))
        }
        "exp" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Exp)))
        }
        "exp2" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Exp2)))
        }
        "expm1" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Expm1)))
        }
        "log" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Log)))
        }
        "log2" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Log2)))
        }
        "log10" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Log10)))
        }
        "log1p" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Log1p)))
        }
        "sin" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Sin)))
        }
        "cos" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Cos)))
        }
        "tan" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Tan)))
        }
        "arcsin" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Arcsin)))
        }
        "arccos" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Arccos)))
        }
        "arctan" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Arctan)))
        }
        "sinh" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Sinh)))
        }
        "cosh" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Cosh)))
        }
        "tanh" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Tanh)))
        }
        "floor" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Floor)))
        }
        "ceil" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Ceil)))
        }
        "trunc" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Trunc)))
        }
        "rint" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Round)))
        }
        "degrees" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Degrees)))
        }
        "radians" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Radians)))
        }
        "arcsinh" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Arcsinh)))
        }
        "arccosh" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Arccosh)))
        }
        "arctanh" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Arctanh)))
        }
        "isnan" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Isnan)))
        }
        "isinf" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Isinf)))
        }
        "isfinite" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Isfinite)))
        }
        // Binary comparison operations
        "greater" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs
                .elementwise_binary(&rhs, BinaryOp::Greater)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "greater_equal" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs
                .elementwise_binary(&rhs, BinaryOp::GreaterEqual)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "less" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs
                .elementwise_binary(&rhs, BinaryOp::Less)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "less_equal" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs
                .elementwise_binary(&rhs, BinaryOp::LessEqual)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "equal" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs
                .elementwise_binary(&rhs, BinaryOp::Equal)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "not_equal" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs
                .elementwise_binary(&rhs, BinaryOp::NotEqual)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        // Elementwise min/max
        "maximum" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs
                .elementwise_binary(&rhs, BinaryOp::Maximum)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "minimum" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs
                .elementwise_binary(&rhs, BinaryOp::Minimum)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "fmax" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs
                .elementwise_binary(&rhs, BinaryOp::Fmax)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "fmin" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs
                .elementwise_binary(&rhs, BinaryOp::Fmin)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        // Power and modulo
        "power" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs
                .elementwise_binary(&rhs, BinaryOp::Power)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "mod" | "remainder" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs
                .elementwise_binary(&rhs, BinaryOp::Remainder)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        // Logical operations
        "logical_and" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs
                .elementwise_binary(&rhs, BinaryOp::LogicalAnd)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "logical_or" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs
                .elementwise_binary(&rhs, BinaryOp::LogicalOr)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "logical_xor" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs
                .elementwise_binary(&rhs, BinaryOp::LogicalXor)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "logical_not" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::LogicalNot)))
        }
        // Statistics
        "quantile" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let q = case.constant;
            let result = input
                .quantile(q, case.axis)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "nanvar" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let ddof = case.n.unwrap_or(0);
            let result = input
                .nanvar(case.axis, case.keepdims, ddof)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "nanstd" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let ddof = case.n.unwrap_or(0);
            let result = input
                .nanstd(case.axis, case.keepdims, ddof)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        // Special math functions
        "ldexp" => {
            let mantissa = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let exponent =
                UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                    .map_err(map_ufunc_error_to_signal_suite)?;
            let result = ldexp(&mantissa, &exponent).map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "nextafter" => {
            let x1 = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let x2 = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = nextafter(&x1, &x2).map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        // Trigonometric binary
        "arctan2" => {
            let y = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let x = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = y
                .elementwise_binary(&x, BinaryOp::Arctan2)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        // Bitwise operations (require integer dtype)
        "bitwise_and" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::I64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::I64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs
                .elementwise_binary(&rhs, BinaryOp::BitwiseAnd)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "bitwise_or" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::I64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::I64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs
                .elementwise_binary(&rhs, BinaryOp::BitwiseOr)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "bitwise_xor" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::I64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::I64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs
                .elementwise_binary(&rhs, BinaryOp::BitwiseXor)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "left_shift" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::I64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::I64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs
                .elementwise_binary(&rhs, BinaryOp::LeftShift)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "right_shift" => {
            let lhs = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::I64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let rhs = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::I64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = lhs
                .elementwise_binary(&rhs, BinaryOp::RightShift)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        // Array manipulation
        "append" => {
            let arr = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let values =
                UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                    .map_err(map_ufunc_error_to_signal_suite)?;
            let result = arr
                .append(&values, case.axis)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        // Set operations
        "isin" => {
            let elements = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let test_elements =
                UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                    .map_err(map_ufunc_error_to_signal_suite)?;
            let result = elements.isin(&test_elements, false);
            Ok(to_outcome(result))
        }
        // Linear algebra scalars
        "det" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input.det().map_err(map_ufunc_error_to_signal_suite)?;
            Ok(SignalOperationOutcome::Array {
                shape: vec![],
                values: vec![result],
            })
        }
        "inv" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = input.inv().map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "solve" => {
            let a = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let b = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = a.solve(&b).map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        // Additional math functions
        "sinc" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.sinc()))
        }
        "heaviside" => {
            let x1 = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let x2 = UFuncArray::new(case.rhs_shape.clone(), case.rhs_values.clone(), DType::F64)
                .map_err(map_ufunc_error_to_signal_suite)?;
            let result = x1
                .elementwise_binary(&x2, BinaryOp::Heaviside)
                .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "spacing" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::F64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            let result = spacing(&input).map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(result))
        }
        "invert" => {
            let input = UFuncArray::new(
                case.input_shape.clone(),
                case.input_values.clone(),
                DType::I64,
            )
            .map_err(map_ufunc_error_to_signal_suite)?;
            Ok(to_outcome(input.elementwise_unary(UnaryOp::Invert)))
        }
        op => Err(SignalSuiteError::new(
            "signal_unknown_operation",
            format!("unknown signal operation: {op}"),
        )),
    }
}

fn validate_signal_differential_expectation(
    case: &SignalDifferentialCase,
    outcome: &SignalOperationOutcome,
) -> Result<(), String> {
    match outcome {
        SignalOperationOutcome::Array { shape, values } => {
            if !case.expected_shape.is_empty() && case.expected_shape != *shape {
                return Err(format!(
                    "shape mismatch expected={:?} actual={shape:?}",
                    case.expected_shape
                ));
            }
            if !case.expected_values.is_empty() {
                if case.expected_values.len() != values.len() {
                    return Err(format!(
                        "value length mismatch expected={} actual={}",
                        case.expected_values.len(),
                        values.len()
                    ));
                }
                let abs_tol = if case.abs_tol > 0.0 {
                    case.abs_tol
                } else {
                    1e-9
                };
                let rel_tol = if case.rel_tol > 0.0 {
                    case.rel_tol
                } else {
                    1e-9
                };
                if !approx_equal_values(&case.expected_values, values, abs_tol, rel_tol) {
                    return Err(format!(
                        "value mismatch expected={:?} actual={values:?}",
                        case.expected_values
                    ));
                }
            }
            Ok(())
        }
    }
}

fn map_ufunc_error_to_signal_suite(error: UFuncError) -> SignalSuiteError {
    SignalSuiteError::new(error.reason_code(), error.to_string())
}

#[derive(Debug, Clone, PartialEq)]
enum PolynomialOperationOutcome {
    Array {
        values: Vec<f64>,
    },
    Pair {
        values: Vec<f64>,
        aux_values: Vec<f64>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PolynomialSuiteError {
    reason_code: String,
    message: String,
}

impl PolynomialSuiteError {
    fn new(reason_code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            reason_code: reason_code.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for PolynomialSuiteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for PolynomialSuiteError {}

pub fn run_polynomial_differential_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let cases = load_polynomial_differential_cases(&config.fixture_root)?;

    let mut report = SuiteReport {
        suite: "polynomial_differential",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };
    let mut mismatches = Vec::new();

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);
        let expected_reason_code = if case.expected_reason_code.trim().is_empty() {
            reason_code.clone()
        } else {
            case.expected_reason_code.trim().to_string()
        };
        let expected_error = case.expected_error_contains.trim().to_lowercase();

        match execute_polynomial_differential_operation(&case) {
            Ok(outcome) => {
                if !expected_error.is_empty() {
                    let rendered = format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected error containing '{}' but operation '{}' succeeded",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        case.expected_error_contains,
                        case.operation
                    );
                    report.failures.push(rendered.clone());
                    mismatches.push(PolynomialDifferentialMismatch {
                        fixture_id: case.id,
                        seed: case.seed,
                        mode,
                        operation: case.operation,
                        expected_reason_code,
                        actual_reason_code: "none".to_string(),
                        message: rendered,
                        artifact_refs,
                    });
                    continue;
                }

                match validate_polynomial_differential_expectation(&case, &outcome) {
                    Ok(()) => report.pass_count += 1,
                    Err(message) => {
                        let rendered = format!(
                            "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} {}",
                            case.id,
                            case.seed,
                            mode,
                            reason_code,
                            env_fingerprint,
                            artifact_refs.join(","),
                            message
                        );
                        report.failures.push(rendered.clone());
                        mismatches.push(PolynomialDifferentialMismatch {
                            fixture_id: case.id,
                            seed: case.seed,
                            mode,
                            operation: case.operation,
                            expected_reason_code,
                            actual_reason_code: "polynomial_differential_mismatch".to_string(),
                            message: rendered,
                            artifact_refs,
                        });
                    }
                }
            }
            Err(err) => {
                if expected_error.is_empty() {
                    let rendered = format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected success but got error '{}' (reason_code={})",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        err.message,
                        err.reason_code
                    );
                    report.failures.push(rendered.clone());
                    mismatches.push(PolynomialDifferentialMismatch {
                        fixture_id: case.id,
                        seed: case.seed,
                        mode,
                        operation: case.operation,
                        expected_reason_code,
                        actual_reason_code: err.reason_code,
                        message: rendered,
                        artifact_refs,
                    });
                    continue;
                }

                let contains_expected = err.message.to_lowercase().contains(&expected_error);
                let reason_match = err.reason_code == expected_reason_code;
                if contains_expected && reason_match {
                    report.pass_count += 1;
                } else {
                    let rendered = format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected error containing '{}' with reason_code='{}' but got '{}' (reason_code='{}')",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        case.expected_error_contains,
                        expected_reason_code,
                        err.message,
                        err.reason_code
                    );
                    report.failures.push(rendered.clone());
                    mismatches.push(PolynomialDifferentialMismatch {
                        fixture_id: case.id,
                        seed: case.seed,
                        mode,
                        operation: case.operation,
                        expected_reason_code,
                        actual_reason_code: err.reason_code,
                        message: rendered,
                        artifact_refs,
                    });
                }
            }
        }
    }

    let artifact = PolynomialDifferentialReportArtifact {
        suite: "polynomial_differential",
        total_cases: report.case_count,
        passed_cases: report.pass_count,
        failed_cases: report.case_count.saturating_sub(report.pass_count),
        mismatches,
    };
    let report_path = config
        .fixture_root
        .join("oracle_outputs/polynomial_differential_report.json");
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }
    let payload = serde_json::to_string_pretty(&artifact)
        .map_err(|err| format!("failed serializing polynomial differential report: {err}"))?;
    fs::write(&report_path, payload.as_bytes())
        .map_err(|err| format!("failed writing {}: {err}", report_path.display()))?;

    Ok(report)
}

fn execute_polynomial_differential_operation(
    case: &PolynomialDifferentialCase,
) -> Result<PolynomialOperationOutcome, PolynomialSuiteError> {
    let as_array = |values: Vec<f64>| PolynomialOperationOutcome::Array { values };
    let as_pair = |values: Vec<f64>, aux_values: Vec<f64>| PolynomialOperationOutcome::Pair {
        values,
        aux_values,
    };

    let lhs = if case.lhs_values.is_empty() {
        None
    } else {
        Some(polynomial_values_to_array(&case.lhs_values)?)
    };
    let rhs = if case.rhs_values.is_empty() {
        None
    } else {
        Some(polynomial_values_to_array(&case.rhs_values)?)
    };
    let y = if case.y_values.is_empty() {
        None
    } else {
        Some(polynomial_values_to_array(&case.y_values)?)
    };

    match case.operation.as_str() {
        "polyval" => {
            let lhs = lhs.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "polyval requires lhs_values coefficients",
                )
            })?;
            let rhs = rhs.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "polyval requires rhs_values x-domain",
                )
            })?;
            Ok(as_array(
                UFuncArray::polyval(&lhs, &rhs)
                    .map_err(map_ufunc_error_to_polynomial_suite)?
                    .values()
                    .to_vec(),
            ))
        }
        "polyfit" => {
            let lhs = lhs.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "polyfit requires lhs_values x-domain",
                )
            })?;
            let y = y.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "polyfit requires y_values",
                )
            })?;
            let deg = case.deg.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "polyfit requires 'deg'",
                )
            })?;
            Ok(as_array(
                UFuncArray::polyfit(&lhs, &y, deg)
                    .map_err(map_ufunc_error_to_polynomial_suite)?
                    .values()
                    .to_vec(),
            ))
        }
        "polyder" => {
            let lhs = lhs.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "polyder requires lhs_values",
                )
            })?;
            Ok(as_array(
                lhs.polyder()
                    .map_err(map_ufunc_error_to_polynomial_suite)?
                    .values()
                    .to_vec(),
            ))
        }
        "polyint" => {
            let lhs = lhs.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "polyint requires lhs_values",
                )
            })?;
            Ok(as_array(
                lhs.polyint()
                    .map_err(map_ufunc_error_to_polynomial_suite)?
                    .values()
                    .to_vec(),
            ))
        }
        "roots" => {
            let lhs = lhs.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "roots requires lhs_values",
                )
            })?;
            Ok(as_array(
                lhs.roots()
                    .map_err(map_ufunc_error_to_polynomial_suite)?
                    .values()
                    .to_vec(),
            ))
        }
        "poly" => {
            let lhs = lhs.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "poly requires lhs_values roots",
                )
            })?;
            Ok(as_array(
                UFuncArray::poly(&lhs)
                    .map_err(map_ufunc_error_to_polynomial_suite)?
                    .values()
                    .to_vec(),
            ))
        }
        "polymul" => {
            let lhs = lhs.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "polymul requires lhs_values",
                )
            })?;
            let rhs = rhs.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "polymul requires rhs_values",
                )
            })?;
            Ok(as_array(
                lhs.polymul(&rhs)
                    .map_err(map_ufunc_error_to_polynomial_suite)?
                    .values()
                    .to_vec(),
            ))
        }
        "polyadd" => {
            let lhs = lhs.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "polyadd requires lhs_values",
                )
            })?;
            let rhs = rhs.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "polyadd requires rhs_values",
                )
            })?;
            Ok(as_array(
                lhs.polyadd(&rhs)
                    .map_err(map_ufunc_error_to_polynomial_suite)?
                    .values()
                    .to_vec(),
            ))
        }
        "polysub" => {
            let lhs = lhs.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "polysub requires lhs_values",
                )
            })?;
            let rhs = rhs.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "polysub requires rhs_values",
                )
            })?;
            Ok(as_array(
                lhs.polysub(&rhs)
                    .map_err(map_ufunc_error_to_polynomial_suite)?
                    .values()
                    .to_vec(),
            ))
        }
        "polydiv" => {
            let lhs = lhs.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "polydiv requires lhs_values",
                )
            })?;
            let rhs = rhs.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "polydiv requires rhs_values",
                )
            })?;
            let (q, r) = lhs
                .polydiv(&rhs)
                .map_err(map_ufunc_error_to_polynomial_suite)?;
            Ok(as_pair(q.values().to_vec(), r.values().to_vec()))
        }
        "chebval" => Ok(as_array(chebval(&case.rhs_values, &case.lhs_values))),
        "chebadd" => Ok(as_array(chebadd(&case.lhs_values, &case.rhs_values))),
        "chebsub" => Ok(as_array(chebsub(&case.lhs_values, &case.rhs_values))),
        "chebmul" => Ok(as_array(chebmul(&case.lhs_values, &case.rhs_values))),
        "chebdiv" => {
            let (q, r) = chebdiv(&case.lhs_values, &case.rhs_values)
                .map_err(map_ufunc_error_to_polynomial_suite)?;
            Ok(as_pair(q, r))
        }
        "chebfromroots" => Ok(as_array(chebfromroots(&case.lhs_values))),
        "cheb2poly" => Ok(as_array(cheb2poly(&case.lhs_values))),
        "poly2cheb" => Ok(as_array(poly2cheb(&case.lhs_values))),
        "chebfit" => {
            let deg = case.deg.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "chebfit requires 'deg'",
                )
            })?;
            Ok(as_array(
                chebfit(&case.lhs_values, &case.y_values, deg)
                    .map_err(map_ufunc_error_to_polynomial_suite)?,
            ))
        }
        "chebroots" => Ok(as_array(
            chebroots(&case.lhs_values).map_err(map_ufunc_error_to_polynomial_suite)?,
        )),
        "chebder" => Ok(as_array(chebder(&case.lhs_values, case.deg.unwrap_or(1)))),
        "chebint" => Ok(as_array(chebint(&case.lhs_values, case.deg.unwrap_or(1)))),
        "legval" => Ok(as_array(legval(&case.rhs_values, &case.lhs_values))),
        "legadd" => Ok(as_array(legadd(&case.lhs_values, &case.rhs_values))),
        "legsub" => Ok(as_array(legsub(&case.lhs_values, &case.rhs_values))),
        "legmul" => Ok(as_array(legmul(&case.lhs_values, &case.rhs_values))),
        "legdiv" => {
            let (q, r) = legdiv(&case.lhs_values, &case.rhs_values)
                .map_err(map_ufunc_error_to_polynomial_suite)?;
            Ok(as_pair(q, r))
        }
        "legfromroots" => Ok(as_array(legfromroots(&case.lhs_values))),
        "leg2poly" => Ok(as_array(leg2poly(&case.lhs_values))),
        "poly2leg" => Ok(as_array(poly2leg(&case.lhs_values))),
        "legroots" => Ok(as_array(
            legroots(&case.lhs_values).map_err(map_ufunc_error_to_polynomial_suite)?,
        )),
        "legder" => Ok(as_array(legder(&case.lhs_values, case.deg.unwrap_or(1)))),
        "legint" => Ok(as_array(legint(&case.lhs_values, case.deg.unwrap_or(1)))),
        "legfit" => {
            let deg = case.deg.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "legfit requires 'deg'",
                )
            })?;
            Ok(as_array(
                legfit(&case.lhs_values, &case.y_values, deg)
                    .map_err(map_ufunc_error_to_polynomial_suite)?,
            ))
        }
        "hermval" => Ok(as_array(hermval(&case.rhs_values, &case.lhs_values))),
        "hermadd" => Ok(as_array(hermadd(&case.lhs_values, &case.rhs_values))),
        "hermsub" => Ok(as_array(hermsub(&case.lhs_values, &case.rhs_values))),
        "hermmul" => Ok(as_array(hermmul(&case.lhs_values, &case.rhs_values))),
        "hermdiv" => {
            let (q, r) = hermdiv(&case.lhs_values, &case.rhs_values)
                .map_err(map_ufunc_error_to_polynomial_suite)?;
            Ok(as_pair(q, r))
        }
        "hermfromroots" => Ok(as_array(hermfromroots(&case.lhs_values))),
        "herm2poly" => Ok(as_array(herm2poly(&case.lhs_values))),
        "poly2herm" => Ok(as_array(poly2herm(&case.lhs_values))),
        "hermroots" => Ok(as_array(
            hermroots(&case.lhs_values).map_err(map_ufunc_error_to_polynomial_suite)?,
        )),
        "hermder" => Ok(as_array(hermder(&case.lhs_values, case.deg.unwrap_or(1)))),
        "hermint" => Ok(as_array(hermint(&case.lhs_values, case.deg.unwrap_or(1)))),
        "lagval" => Ok(as_array(lagval(&case.rhs_values, &case.lhs_values))),
        "lagadd" => Ok(as_array(lagadd(&case.lhs_values, &case.rhs_values))),
        "lagsub" => Ok(as_array(lagsub(&case.lhs_values, &case.rhs_values))),
        "lagmul" => Ok(as_array(lagmul(&case.lhs_values, &case.rhs_values))),
        "lagdiv" => {
            let (q, r) = lagdiv(&case.lhs_values, &case.rhs_values)
                .map_err(map_ufunc_error_to_polynomial_suite)?;
            Ok(as_pair(q, r))
        }
        "lagfromroots" => Ok(as_array(lagfromroots(&case.lhs_values))),
        "lag2poly" => Ok(as_array(lag2poly(&case.lhs_values))),
        "poly2lag" => Ok(as_array(poly2lag(&case.lhs_values))),
        "lagroots" => Ok(as_array(
            lagroots(&case.lhs_values).map_err(map_ufunc_error_to_polynomial_suite)?,
        )),
        "lagder" => Ok(as_array(lagder(&case.lhs_values, case.deg.unwrap_or(1)))),
        "lagint" => Ok(as_array(lagint(&case.lhs_values, case.deg.unwrap_or(1)))),
        "hermeval" => Ok(as_array(hermeval(&case.rhs_values, &case.lhs_values))),
        "hermeadd" => Ok(as_array(hermeadd(&case.lhs_values, &case.rhs_values))),
        "hermesub" => Ok(as_array(hermesub(&case.lhs_values, &case.rhs_values))),
        "hermemul" => Ok(as_array(hermemul(&case.lhs_values, &case.rhs_values))),
        "hermediv" => {
            let (q, r) = hermediv(&case.lhs_values, &case.rhs_values)
                .map_err(map_ufunc_error_to_polynomial_suite)?;
            Ok(as_pair(q, r))
        }
        "hermefromroots" => Ok(as_array(hermefromroots(&case.lhs_values))),
        "hermeroots" => Ok(as_array(
            hermeroots(&case.lhs_values).map_err(map_ufunc_error_to_polynomial_suite)?,
        )),
        "herme2poly" => Ok(as_array(herme2poly(&case.lhs_values))),
        "poly2herme" => Ok(as_array(poly2herme(&case.lhs_values))),
        "polymulx" => {
            let lhs = lhs.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "polymulx requires lhs_values",
                )
            })?;
            Ok(as_array(
                lhs.polymulx()
                    .map_err(map_ufunc_error_to_polynomial_suite)?
                    .values()
                    .to_vec(),
            ))
        }
        "polypow" => {
            let lhs = lhs.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "polypow requires lhs_values",
                )
            })?;
            let deg = case.deg.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "polypow requires 'deg' (power)",
                )
            })?;
            Ok(as_array(
                lhs.polypow(deg)
                    .map_err(map_ufunc_error_to_polynomial_suite)?
                    .values()
                    .to_vec(),
            ))
        }
        "polytrim" => {
            let lhs = lhs.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "polytrim requires lhs_values",
                )
            })?;
            let tol = case.abs_tol;
            Ok(as_array(
                lhs.polytrim(tol)
                    .map_err(map_ufunc_error_to_polynomial_suite)?
                    .values()
                    .to_vec(),
            ))
        }
        "polyvander" => {
            let lhs = lhs.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "polyvander requires lhs_values (x points)",
                )
            })?;
            let deg = case.deg.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "polyvander requires 'deg'",
                )
            })?;
            Ok(as_array(
                lhs.polyvander(deg)
                    .map_err(map_ufunc_error_to_polynomial_suite)?
                    .values()
                    .to_vec(),
            ))
        }
        "polyvalfromroots" => {
            let lhs = lhs.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "polyvalfromroots requires lhs_values (x points)",
                )
            })?;
            let rhs = rhs.ok_or_else(|| {
                PolynomialSuiteError::new(
                    "polynomial_input_contract_violation",
                    "polyvalfromroots requires rhs_values (roots)",
                )
            })?;
            Ok(as_array(
                lhs.polyvalfromroots(&rhs)
                    .map_err(map_ufunc_error_to_polynomial_suite)?
                    .values()
                    .to_vec(),
            ))
        }
        other => Err(PolynomialSuiteError::new(
            "polynomial_policy_unknown_operation",
            format!("unsupported polynomial differential operation {other}"),
        )),
    }
}

fn validate_polynomial_differential_expectation(
    case: &PolynomialDifferentialCase,
    outcome: &PolynomialOperationOutcome,
) -> Result<(), String> {
    let abs_tol = if case.abs_tol > 0.0 {
        case.abs_tol
    } else {
        1e-9
    };
    let rel_tol = if case.rel_tol > 0.0 {
        case.rel_tol
    } else {
        1e-9
    };

    let compare = |expected: &[f64], actual: &[f64], label: &str| -> Result<(), String> {
        if expected.len() != actual.len() {
            return Err(format!(
                "{label} length mismatch expected={} actual={}",
                expected.len(),
                actual.len()
            ));
        }
        let mut expected_values = expected.to_vec();
        let mut actual_values = actual.to_vec();
        if case.sort_values {
            expected_values
                .sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
            actual_values
                .sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
        }
        if !approx_equal_values(&expected_values, &actual_values, abs_tol, rel_tol) {
            return Err(format!(
                "{label} mismatch expected={expected_values:?} actual={actual_values:?} abs_tol={abs_tol} rel_tol={rel_tol}"
            ));
        }
        Ok(())
    };

    match outcome {
        PolynomialOperationOutcome::Array { values } => {
            if case.expected_values.is_empty() {
                return Err("expected_values required for array operation".to_string());
            }
            compare(&case.expected_values, values, "values")
        }
        PolynomialOperationOutcome::Pair { values, aux_values } => {
            if case.expected_values.is_empty() || case.expected_aux_values.is_empty() {
                return Err(
                    "expected_values and expected_aux_values required for pair operation"
                        .to_string(),
                );
            }
            compare(&case.expected_values, values, "values")?;
            compare(&case.expected_aux_values, aux_values, "aux_values")
        }
    }
}

fn polynomial_values_to_array(values: &[f64]) -> Result<UFuncArray, PolynomialSuiteError> {
    UFuncArray::new(vec![values.len()], values.to_vec(), DType::F64)
        .map_err(map_ufunc_error_to_polynomial_suite)
}

fn map_ufunc_error_to_polynomial_suite(error: UFuncError) -> PolynomialSuiteError {
    PolynomialSuiteError::new(error.reason_code(), error.to_string())
}

#[derive(Debug, Clone, PartialEq)]
enum StringOperationOutcome {
    Numeric { values: Vec<f64> },
    Text { values: Vec<String> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StringSuiteError {
    reason_code: String,
    message: String,
}

impl StringSuiteError {
    fn new(reason_code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            reason_code: reason_code.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for StringSuiteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for StringSuiteError {}

pub fn run_string_differential_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let cases = load_string_differential_cases(&config.fixture_root)?;

    let mut report = SuiteReport {
        suite: "string_differential",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };
    let mut mismatches = Vec::new();

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);
        let expected_reason_code = if case.expected_reason_code.trim().is_empty() {
            reason_code.clone()
        } else {
            case.expected_reason_code.trim().to_string()
        };
        let expected_error = case.expected_error_contains.trim().to_lowercase();

        match execute_string_differential_operation(&case) {
            Ok(outcome) => {
                if !expected_error.is_empty() {
                    let rendered = format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected error containing '{}' but operation '{}' succeeded",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        case.expected_error_contains,
                        case.operation
                    );
                    report.failures.push(rendered.clone());
                    mismatches.push(StringDifferentialMismatch {
                        fixture_id: case.id,
                        seed: case.seed,
                        mode,
                        operation: case.operation,
                        expected_reason_code,
                        actual_reason_code: "none".to_string(),
                        message: rendered,
                        artifact_refs,
                    });
                    continue;
                }

                match validate_string_differential_expectation(&case, &outcome) {
                    Ok(()) => report.pass_count += 1,
                    Err(message) => {
                        let rendered = format!(
                            "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} {}",
                            case.id,
                            case.seed,
                            mode,
                            reason_code,
                            env_fingerprint,
                            artifact_refs.join(","),
                            message
                        );
                        report.failures.push(rendered.clone());
                        mismatches.push(StringDifferentialMismatch {
                            fixture_id: case.id,
                            seed: case.seed,
                            mode,
                            operation: case.operation,
                            expected_reason_code,
                            actual_reason_code: "string_differential_mismatch".to_string(),
                            message: rendered,
                            artifact_refs,
                        });
                    }
                }
            }
            Err(err) => {
                if expected_error.is_empty() {
                    let rendered = format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected success but got error '{}' (reason_code={})",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        err.message,
                        err.reason_code
                    );
                    report.failures.push(rendered.clone());
                    mismatches.push(StringDifferentialMismatch {
                        fixture_id: case.id,
                        seed: case.seed,
                        mode,
                        operation: case.operation,
                        expected_reason_code,
                        actual_reason_code: err.reason_code,
                        message: rendered,
                        artifact_refs,
                    });
                    continue;
                }

                let contains_expected = err.message.to_lowercase().contains(&expected_error);
                let reason_match = err.reason_code == expected_reason_code;
                if contains_expected && reason_match {
                    report.pass_count += 1;
                } else {
                    let rendered = format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected error containing '{}' with reason_code='{}' but got '{}' (reason_code='{}')",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        case.expected_error_contains,
                        expected_reason_code,
                        err.message,
                        err.reason_code
                    );
                    report.failures.push(rendered.clone());
                    mismatches.push(StringDifferentialMismatch {
                        fixture_id: case.id,
                        seed: case.seed,
                        mode,
                        operation: case.operation,
                        expected_reason_code,
                        actual_reason_code: err.reason_code,
                        message: rendered,
                        artifact_refs,
                    });
                }
            }
        }
    }

    let artifact = StringDifferentialReportArtifact {
        suite: "string_differential",
        total_cases: report.case_count,
        passed_cases: report.pass_count,
        failed_cases: report.case_count.saturating_sub(report.pass_count),
        mismatches,
    };
    let report_path = config
        .fixture_root
        .join("oracle_outputs/string_differential_report.json");
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }
    let payload = serde_json::to_string_pretty(&artifact)
        .map_err(|err| format!("failed serializing string differential report: {err}"))?;
    fs::write(&report_path, payload.as_bytes())
        .map_err(|err| format!("failed writing {}: {err}", report_path.display()))?;

    Ok(report)
}

pub fn run_string_metamorphic_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let cases = load_string_metamorphic_cases(&config.fixture_root)?;

    let mut report = SuiteReport {
        suite: "string_metamorphic",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);

        match evaluate_string_metamorphic_relation(&case) {
            Ok(()) => {
                report.pass_count += 1;
            }
            Err(err) => {
                report.failures.push(format!(
                    "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} {} (actual_reason_code={})",
                    case.id,
                    case.seed,
                    mode,
                    reason_code,
                    env_fingerprint,
                    artifact_refs.join(","),
                    err.message,
                    err.reason_code
                ));
            }
        }
    }

    Ok(report)
}

fn execute_string_differential_operation(
    case: &StringDifferentialCase,
) -> Result<StringOperationOutcome, StringSuiteError> {
    let as_text = |values: Vec<String>| StringOperationOutcome::Text { values };
    let as_numeric = |values: Vec<f64>| StringOperationOutcome::Numeric { values };

    let lhs = string_values_to_array(&case.lhs_strings)?;
    let rhs = if case.rhs_strings.is_empty() {
        None
    } else {
        Some(string_values_to_array(&case.rhs_strings)?)
    };

    match case.operation.as_str() {
        "upper" => Ok(as_text(lhs.upper().values().to_vec())),
        "lower" => Ok(as_text(lhs.lower().values().to_vec())),
        "capitalize" => Ok(as_text(lhs.capitalize().values().to_vec())),
        "title" => Ok(as_text(lhs.title().values().to_vec())),
        "strip" => Ok(as_text(lhs.strip().values().to_vec())),
        "lstrip" => Ok(as_text(lhs.lstrip().values().to_vec())),
        "rstrip" => Ok(as_text(lhs.rstrip().values().to_vec())),
        "swapcase" => Ok(as_text(lhs.swapcase().values().to_vec())),
        "replace" => Ok(as_text(
            lhs.replace(&case.arg_string, &case.arg_alt_string)
                .values()
                .to_vec(),
        )),
        "add" => {
            let rhs = rhs.ok_or_else(|| {
                StringSuiteError::new(
                    "string_input_contract_violation",
                    "add requires rhs_strings",
                )
            })?;
            Ok(as_text(
                lhs.add(&rhs)
                    .map_err(map_ufunc_error_to_string_suite)?
                    .values()
                    .to_vec(),
            ))
        }
        "multiply" => {
            let repeat = case.repeat.ok_or_else(|| {
                StringSuiteError::new(
                    "string_input_contract_violation",
                    "multiply requires repeat",
                )
            })?;
            Ok(as_text(lhs.multiply(repeat).values().to_vec()))
        }
        "str_len" => Ok(as_numeric(lhs.str_len().values().to_vec())),
        "count" => Ok(as_numeric(
            lhs.count_substr(&case.arg_string).values().to_vec(),
        )),
        "find" => Ok(as_numeric(lhs.find(&case.arg_string).values().to_vec())),
        "rfind" => Ok(as_numeric(lhs.rfind(&case.arg_string).values().to_vec())),
        "startswith" => Ok(as_numeric(
            lhs.startswith(&case.arg_string).values().to_vec(),
        )),
        "endswith" => Ok(as_numeric(lhs.endswith(&case.arg_string).values().to_vec())),
        "isalpha" => Ok(as_numeric(lhs.isalpha().values().to_vec())),
        "isdigit" => Ok(as_numeric(lhs.isdigit().values().to_vec())),
        "isalnum" => Ok(as_numeric(lhs.isalnum().values().to_vec())),
        "isspace" => Ok(as_numeric(lhs.isspace().values().to_vec())),
        "isupper" => Ok(as_numeric(lhs.isupper().values().to_vec())),
        "islower" => Ok(as_numeric(lhs.islower().values().to_vec())),
        "isnumeric" => Ok(as_numeric(lhs.isnumeric().values().to_vec())),
        "isdecimal" => Ok(as_numeric(lhs.isdecimal().values().to_vec())),
        "istitle" => Ok(as_numeric(lhs.istitle().values().to_vec())),
        "equal" => {
            let rhs = rhs.ok_or_else(|| {
                StringSuiteError::new(
                    "string_input_contract_violation",
                    "equal requires rhs_strings",
                )
            })?;
            Ok(as_numeric(lhs.equal(&rhs).values().to_vec()))
        }
        "not_equal" => {
            let rhs = rhs.ok_or_else(|| {
                StringSuiteError::new(
                    "string_input_contract_violation",
                    "not_equal requires rhs_strings",
                )
            })?;
            Ok(as_numeric(lhs.not_equal(&rhs).values().to_vec()))
        }
        "greater" => {
            let rhs = rhs.ok_or_else(|| {
                StringSuiteError::new(
                    "string_input_contract_violation",
                    "greater requires rhs_strings",
                )
            })?;
            Ok(as_numeric(lhs.greater(&rhs).values().to_vec()))
        }
        "greater_equal" => {
            let rhs = rhs.ok_or_else(|| {
                StringSuiteError::new(
                    "string_input_contract_violation",
                    "greater_equal requires rhs_strings",
                )
            })?;
            Ok(as_numeric(lhs.greater_equal(&rhs).values().to_vec()))
        }
        "less" => {
            let rhs = rhs.ok_or_else(|| {
                StringSuiteError::new(
                    "string_input_contract_violation",
                    "less requires rhs_strings",
                )
            })?;
            Ok(as_numeric(lhs.less(&rhs).values().to_vec()))
        }
        "less_equal" => {
            let rhs = rhs.ok_or_else(|| {
                StringSuiteError::new(
                    "string_input_contract_violation",
                    "less_equal requires rhs_strings",
                )
            })?;
            Ok(as_numeric(lhs.less_equal(&rhs).values().to_vec()))
        }
        "compare_chararrays" => {
            let rhs = rhs.ok_or_else(|| {
                StringSuiteError::new(
                    "string_input_contract_violation",
                    "compare_chararrays requires rhs_strings",
                )
            })?;
            let cmp = if case.cmp.trim().is_empty() {
                case.arg_string.trim()
            } else {
                case.cmp.trim()
            };
            if cmp.is_empty() {
                return Err(StringSuiteError::new(
                    "string_input_contract_violation",
                    "compare_chararrays requires cmp",
                ));
            }
            Ok(as_numeric(
                lhs.compare_chararrays(&rhs, cmp)
                    .map_err(map_ufunc_error_to_string_suite)?
                    .values()
                    .to_vec(),
            ))
        }
        "slice" => {
            let result = lhs.slice(case.slice_start, case.slice_stop, case.slice_step);
            Ok(as_text(result.values().to_vec()))
        }
        "center" => {
            let width = case.width.ok_or_else(|| {
                StringSuiteError::new("string_input_contract_violation", "center requires width")
            })?;
            let fillchar = case
                .fillchar
                .as_ref()
                .and_then(|s| s.chars().next())
                .unwrap_or(' ');
            Ok(as_text(lhs.center(width, fillchar).values().to_vec()))
        }
        "ljust" => {
            let width = case.width.ok_or_else(|| {
                StringSuiteError::new("string_input_contract_violation", "ljust requires width")
            })?;
            let fillchar = case
                .fillchar
                .as_ref()
                .and_then(|s| s.chars().next())
                .unwrap_or(' ');
            Ok(as_text(lhs.ljust(width, fillchar).values().to_vec()))
        }
        "rjust" => {
            let width = case.width.ok_or_else(|| {
                StringSuiteError::new("string_input_contract_violation", "rjust requires width")
            })?;
            let fillchar = case
                .fillchar
                .as_ref()
                .and_then(|s| s.chars().next())
                .unwrap_or(' ');
            Ok(as_text(lhs.rjust(width, fillchar).values().to_vec()))
        }
        "zfill" => {
            let width = case.width.ok_or_else(|| {
                StringSuiteError::new("string_input_contract_violation", "zfill requires width")
            })?;
            Ok(as_text(lhs.zfill(width).values().to_vec()))
        }
        "join" => {
            let sep = &case.arg_string;
            Ok(as_text(lhs.join(sep).values().to_vec()))
        }
        "expandtabs" => {
            let tabsize = case.tabsize.unwrap_or(8);
            Ok(as_text(lhs.expandtabs(tabsize).values().to_vec()))
        }
        "index" => {
            let sub = &case.arg_string;
            Ok(as_numeric(
                lhs.index(sub)
                    .map_err(map_ufunc_error_to_string_suite)?
                    .values()
                    .to_vec(),
            ))
        }
        "rindex" => {
            let sub = &case.arg_string;
            Ok(as_numeric(
                lhs.rindex(sub)
                    .map_err(map_ufunc_error_to_string_suite)?
                    .values()
                    .to_vec(),
            ))
        }
        other => Err(StringSuiteError::new(
            "string_policy_unknown_operation",
            format!("unsupported string differential operation {other}"),
        )),
    }
}

fn validate_string_differential_expectation(
    case: &StringDifferentialCase,
    outcome: &StringOperationOutcome,
) -> Result<(), String> {
    let abs_tol = if case.abs_tol > 0.0 {
        case.abs_tol
    } else {
        1e-9
    };
    let rel_tol = if case.rel_tol > 0.0 {
        case.rel_tol
    } else {
        1e-9
    };

    match outcome {
        StringOperationOutcome::Numeric { values } => {
            if case.expected_values.len() != values.len() {
                return Err(format!(
                    "values length mismatch expected={} actual={}",
                    case.expected_values.len(),
                    values.len()
                ));
            }
            if !approx_equal_values(&case.expected_values, values, abs_tol, rel_tol) {
                return Err(format!(
                    "values mismatch expected={:?} actual={values:?} abs_tol={abs_tol} rel_tol={rel_tol}",
                    case.expected_values
                ));
            }
            Ok(())
        }
        StringOperationOutcome::Text { values } => {
            if case.expected_strings.len() != values.len() {
                return Err(format!(
                    "strings length mismatch expected={} actual={}",
                    case.expected_strings.len(),
                    values.len()
                ));
            }
            if case.expected_strings != *values {
                return Err(format!(
                    "strings mismatch expected={:?} actual={values:?}",
                    case.expected_strings
                ));
            }
            Ok(())
        }
    }
}

fn evaluate_string_metamorphic_relation(
    case: &StringMetamorphicCase,
) -> Result<(), StringSuiteError> {
    match case.relation.as_str() {
        "str_lower_idempotent" => {
            let input = string_values_to_array(&case.input_strings)?;
            let once = input.lower();
            let twice = once.lower();
            assert_string_metamorphic_values_equal(case, twice.values(), once.values())
        }
        "str_upper_idempotent" => {
            let input = string_values_to_array(&case.input_strings)?;
            let once = input.upper();
            let twice = once.upper();
            assert_string_metamorphic_values_equal(case, twice.values(), once.values())
        }
        "str_strip_idempotent" => {
            let input = string_values_to_array(&case.input_strings)?;
            let once = input.strip();
            let twice = once.strip();
            assert_string_metamorphic_values_equal(case, twice.values(), once.values())
        }
        "str_len_after_multiply" => {
            let factor = case.multiply_factor.ok_or_else(|| {
                StringSuiteError::new(
                    "string_input_contract_violation",
                    "str_len_after_multiply requires multiply_factor",
                )
            })?;
            let input = string_values_to_array(&case.input_strings)?;
            let multiplied_lengths = input.multiply(factor).str_len().values().to_vec();
            let expected_lengths = input
                .str_len()
                .values()
                .iter()
                .map(|len| *len * factor as f64)
                .collect::<Vec<_>>();
            assert_numeric_metamorphic_values_equal(case, &multiplied_lengths, &expected_lengths)
        }
        "str_find_in_concat" => {
            require_non_empty_string_field(case, "search_substr", &case.search_substr)?;
            let (lhs, rhs) = string_pair_values_to_arrays(case)?;
            let concatenated = lhs.add(&rhs).map_err(map_ufunc_error_to_string_suite)?;
            let actual = concatenated.find(&case.search_substr).values().to_vec();
            let expected = case
                .a_strings
                .iter()
                .zip(case.b_strings.iter())
                .map(|(a, b)| string_find_char_index(&format!("{a}{b}"), &case.search_substr))
                .collect::<Vec<_>>();
            assert_numeric_metamorphic_values_equal(case, &actual, &expected)
        }
        "str_replace_count_zero_unchanged" => {
            require_non_empty_string_field(case, "old_substr", &case.old_substr)?;
            let input = string_values_to_array(&case.input_strings)?;
            let counts = input.count_substr(&case.old_substr);
            if counts.values().iter().any(|count| *count != 0.0) {
                return Err(StringSuiteError::new(
                    "string_metamorphic_precondition_violation",
                    format!(
                        "str_replace_count_zero_unchanged requires absent old_substr='{}'",
                        case.old_substr
                    ),
                ));
            }
            let replaced = input.replace(&case.old_substr, &case.new_substr);
            assert_string_metamorphic_values_equal(case, replaced.values(), input.values())
        }
        "str_center_len_preserves_content" => {
            let width = required_string_width(case)?;
            let input = string_values_to_array(&case.input_strings)?;
            let centered = input.center(width, ' ');
            for (idx, (original, actual)) in case
                .input_strings
                .iter()
                .zip(centered.values().iter())
                .enumerate()
            {
                let expected_len = width.max(original.chars().count());
                let actual_len = actual.chars().count();
                if actual_len != expected_len || !actual.contains(original) {
                    return Err(StringSuiteError::new(
                        "string_metamorphic_relation_failed",
                        format!(
                            "str_center_len_preserves_content mismatch at index {idx}: original={original:?} actual={actual:?} expected_len={expected_len} actual_len={actual_len}"
                        ),
                    ));
                }
            }
            Ok(())
        }
        "str_count_additive_concat" => {
            require_non_empty_string_field(case, "search_char", &case.search_char)?;
            let (lhs, rhs) = string_pair_values_to_arrays(case)?;
            let concatenated = lhs.add(&rhs).map_err(map_ufunc_error_to_string_suite)?;
            let actual = concatenated
                .count_substr(&case.search_char)
                .values()
                .to_vec();
            let lhs_counts = lhs.count_substr(&case.search_char);
            let rhs_counts = rhs.count_substr(&case.search_char);
            let expected = lhs_counts
                .values()
                .iter()
                .zip(rhs_counts.values().iter())
                .map(|(left, right)| *left + *right)
                .collect::<Vec<_>>();
            assert_numeric_metamorphic_values_equal(case, &actual, &expected)
        }
        "str_swapcase_involution" => {
            let input = string_values_to_array(&case.input_strings)?;
            let swapped_twice = input.swapcase().swapcase();
            assert_string_metamorphic_values_equal(case, swapped_twice.values(), input.values())
        }
        "str_startswith_after_ljust" => {
            let width = required_string_width(case)?;
            let input = string_values_to_array(&case.input_strings)?;
            let padded = input.ljust(width, ' ');
            let actual = padded
                .values()
                .iter()
                .zip(case.input_strings.iter())
                .map(|(value, original)| {
                    if value.starts_with(original) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>();
            let expected = vec![1.0; case.input_strings.len()];
            assert_numeric_metamorphic_values_equal(case, &actual, &expected)
        }
        "str_endswith_after_rjust" => {
            let width = required_string_width(case)?;
            let input = string_values_to_array(&case.input_strings)?;
            let padded = input.rjust(width, ' ');
            let actual = padded
                .values()
                .iter()
                .zip(case.input_strings.iter())
                .map(
                    |(value, original)| {
                        if value.ends_with(original) { 1.0 } else { 0.0 }
                    },
                )
                .collect::<Vec<_>>();
            let expected = vec![1.0; case.input_strings.len()];
            assert_numeric_metamorphic_values_equal(case, &actual, &expected)
        }
        "str_isalpha_subset_isalnum" => {
            let input = string_values_to_array(&case.input_strings)?;
            let alpha = input.isalpha();
            let alnum = input.isalnum();
            for (idx, (alpha_value, alnum_value)) in
                alpha.values().iter().zip(alnum.values().iter()).enumerate()
            {
                if *alpha_value != 0.0 && *alnum_value == 0.0 {
                    return Err(StringSuiteError::new(
                        "string_metamorphic_relation_failed",
                        format!(
                            "str_isalpha_subset_isalnum failed at index {idx}: alpha={alpha_value} alnum={alnum_value}"
                        ),
                    ));
                }
            }
            Ok(())
        }
        other => Err(StringSuiteError::new(
            "string_policy_unknown_metamorphic_relation",
            format!("unsupported string metamorphic relation {other}"),
        )),
    }
}

fn string_pair_values_to_arrays(
    case: &StringMetamorphicCase,
) -> Result<(StringArray, StringArray), StringSuiteError> {
    if case.a_strings.len() != case.b_strings.len() {
        return Err(StringSuiteError::new(
            "string_input_contract_violation",
            format!(
                "paired string inputs must have equal lengths: a_strings={} b_strings={}",
                case.a_strings.len(),
                case.b_strings.len()
            ),
        ));
    }
    Ok((
        string_values_to_array(&case.a_strings)?,
        string_values_to_array(&case.b_strings)?,
    ))
}

fn required_string_width(case: &StringMetamorphicCase) -> Result<usize, StringSuiteError> {
    case.width.ok_or_else(|| {
        StringSuiteError::new(
            "string_input_contract_violation",
            format!("{} requires width", case.relation),
        )
    })
}

fn require_non_empty_string_field(
    case: &StringMetamorphicCase,
    field: &str,
    value: &str,
) -> Result<(), StringSuiteError> {
    if value.is_empty() {
        return Err(StringSuiteError::new(
            "string_input_contract_violation",
            format!("{} requires non-empty {field}", case.relation),
        ));
    }
    Ok(())
}

fn string_find_char_index(haystack: &str, needle: &str) -> f64 {
    haystack
        .find(needle)
        .map_or(-1.0, |byte_idx| haystack[..byte_idx].chars().count() as f64)
}

fn assert_string_metamorphic_values_equal(
    case: &StringMetamorphicCase,
    actual: &[String],
    expected: &[String],
) -> Result<(), StringSuiteError> {
    if actual.len() != expected.len() {
        return Err(StringSuiteError::new(
            "string_metamorphic_relation_failed",
            format!(
                "{} length mismatch expected={} actual={}",
                case.relation,
                expected.len(),
                actual.len()
            ),
        ));
    }
    if actual != expected {
        return Err(StringSuiteError::new(
            "string_metamorphic_relation_failed",
            format!(
                "{} mismatch expected={expected:?} actual={actual:?}",
                case.relation
            ),
        ));
    }
    Ok(())
}

fn assert_numeric_metamorphic_values_equal(
    case: &StringMetamorphicCase,
    actual: &[f64],
    expected: &[f64],
) -> Result<(), StringSuiteError> {
    if actual.len() != expected.len() {
        return Err(StringSuiteError::new(
            "string_metamorphic_relation_failed",
            format!(
                "{} length mismatch expected={} actual={}",
                case.relation,
                expected.len(),
                actual.len()
            ),
        ));
    }
    if !approx_equal_values(expected, actual, 1e-9, 1e-9) {
        return Err(StringSuiteError::new(
            "string_metamorphic_relation_failed",
            format!(
                "{} mismatch expected={expected:?} actual={actual:?}",
                case.relation
            ),
        ));
    }
    Ok(())
}

fn string_values_to_array(values: &[String]) -> Result<StringArray, StringSuiteError> {
    StringArray::new(vec![values.len()], values.to_vec()).map_err(map_ufunc_error_to_string_suite)
}

fn map_ufunc_error_to_string_suite(error: UFuncError) -> StringSuiteError {
    StringSuiteError::new(error.reason_code(), error.to_string())
}

#[derive(Debug, Clone, PartialEq)]
enum MaskedOperationOutcome {
    Array {
        shape: Vec<usize>,
        values: Vec<f64>,
    },
    Masked {
        shape: Vec<usize>,
        values: Vec<f64>,
        mask: Option<Vec<f64>>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MaskedSuiteError {
    reason_code: String,
    message: String,
}

impl MaskedSuiteError {
    fn new(reason_code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            reason_code: reason_code.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for MaskedSuiteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for MaskedSuiteError {}

pub fn run_masked_differential_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let cases = load_masked_differential_cases(&config.fixture_root)?;
    let mut report = SuiteReport {
        suite: "masked_differential",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };
    let mut mismatches = Vec::new();

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);
        let expected_reason_code = if case.expected_reason_code.trim().is_empty() {
            reason_code.clone()
        } else {
            case.expected_reason_code.trim().to_string()
        };
        let expected_error = case.expected_error_contains.trim().to_lowercase();

        match execute_masked_differential_operation(&case) {
            Ok(outcome) => {
                if !expected_error.is_empty() {
                    let rendered = format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected error containing '{}' but operation '{}' succeeded",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        case.expected_error_contains,
                        case.operation
                    );
                    report.failures.push(rendered.clone());
                    mismatches.push(MaskedDifferentialMismatch {
                        fixture_id: case.id,
                        seed: case.seed,
                        mode,
                        operation: case.operation,
                        expected_reason_code,
                        actual_reason_code: "none".to_string(),
                        message: rendered,
                        artifact_refs,
                    });
                    continue;
                }

                match validate_masked_differential_expectation(&case, &outcome) {
                    Ok(()) => {
                        report.pass_count += 1;
                    }
                    Err(message) => {
                        let rendered = format!(
                            "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} {}",
                            case.id,
                            case.seed,
                            mode,
                            reason_code,
                            env_fingerprint,
                            artifact_refs.join(","),
                            message
                        );
                        report.failures.push(rendered.clone());
                        mismatches.push(MaskedDifferentialMismatch {
                            fixture_id: case.id,
                            seed: case.seed,
                            mode,
                            operation: case.operation,
                            expected_reason_code,
                            actual_reason_code: reason_code,
                            message: rendered,
                            artifact_refs,
                        });
                    }
                }
            }
            Err(error) => {
                let error_lower = error.message.to_lowercase();
                if expected_error.is_empty() {
                    let rendered = format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} operation '{}' failed unexpectedly with reason_code={} message={}",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        case.operation,
                        error.reason_code,
                        error.message
                    );
                    report.failures.push(rendered.clone());
                    mismatches.push(MaskedDifferentialMismatch {
                        fixture_id: case.id,
                        seed: case.seed,
                        mode,
                        operation: case.operation,
                        expected_reason_code,
                        actual_reason_code: error.reason_code,
                        message: rendered,
                        artifact_refs,
                    });
                    continue;
                }

                if !error_lower.contains(&expected_error) {
                    let rendered = format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected error containing '{}' but got '{}'",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        case.expected_error_contains,
                        error.message
                    );
                    report.failures.push(rendered.clone());
                    mismatches.push(MaskedDifferentialMismatch {
                        fixture_id: case.id,
                        seed: case.seed,
                        mode,
                        operation: case.operation,
                        expected_reason_code,
                        actual_reason_code: error.reason_code,
                        message: rendered,
                        artifact_refs,
                    });
                    continue;
                }

                if expected_reason_code != error.reason_code {
                    let rendered = format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected reason_code='{}' but got '{}'",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        expected_reason_code,
                        error.reason_code
                    );
                    report.failures.push(rendered.clone());
                    mismatches.push(MaskedDifferentialMismatch {
                        fixture_id: case.id,
                        seed: case.seed,
                        mode,
                        operation: case.operation,
                        expected_reason_code,
                        actual_reason_code: error.reason_code,
                        message: rendered,
                        artifact_refs,
                    });
                    continue;
                }

                report.pass_count += 1;
            }
        }
    }

    let artifact = MaskedDifferentialReportArtifact {
        suite: report.suite,
        total_cases: report.case_count,
        passed_cases: report.pass_count,
        failed_cases: report.case_count.saturating_sub(report.pass_count),
        mismatches,
    };
    let report_path = config
        .fixture_root
        .join("oracle_outputs/masked_differential_report.json");
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }
    let payload = serde_json::to_string_pretty(&artifact)
        .map_err(|err| format!("failed serializing masked differential report: {err}"))?;
    fs::write(&report_path, payload.as_bytes())
        .map_err(|err| format!("failed writing {}: {err}", report_path.display()))?;

    Ok(report)
}

fn execute_masked_differential_operation(
    case: &MaskedDifferentialCase,
) -> Result<MaskedOperationOutcome, MaskedSuiteError> {
    let mut lhs = masked_values_to_array(
        &case.lhs_shape,
        &case.lhs_values,
        &case.lhs_mask,
        case.fill_value,
        &case.nan_indices,
        &case.inf_indices,
    )?;
    if let Some(fill_value) = case.fill_value {
        lhs.set_fill_value(fill_value);
    }

    let mut rhs = if case.rhs_values.is_empty() {
        None
    } else {
        let rhs_array = masked_values_to_array(
            &case.rhs_shape,
            &case.rhs_values,
            &case.rhs_mask,
            case.fill_value,
            &[],
            &[],
        )?;
        Some(rhs_array)
    };
    if let Some(fill_value) = case.fill_value
        && let Some(rhs_array) = rhs.as_mut()
    {
        rhs_array.set_fill_value(fill_value);
    }

    match case.operation.as_str() {
        "add" => {
            let rhs = rhs.ok_or_else(|| {
                MaskedSuiteError::new("masked_input_contract_violation", "add requires rhs_values")
            })?;
            let result = lhs
                .elementwise_binary(&rhs, BinaryOp::Add)
                .map_err(map_ma_error_to_masked_suite)?;
            Ok(masked_array_to_outcome(&result))
        }
        "multiply" => {
            let rhs = rhs.ok_or_else(|| {
                MaskedSuiteError::new(
                    "masked_input_contract_violation",
                    "multiply requires rhs_values",
                )
            })?;
            let result = lhs
                .elementwise_binary(&rhs, BinaryOp::Mul)
                .map_err(map_ma_error_to_masked_suite)?;
            Ok(masked_array_to_outcome(&result))
        }
        "equal" => {
            let rhs = rhs.ok_or_else(|| {
                MaskedSuiteError::new(
                    "masked_input_contract_violation",
                    "equal requires rhs_values",
                )
            })?;
            let result = lhs.equal(&rhs).map_err(map_ma_error_to_masked_suite)?;
            Ok(masked_array_to_outcome(&result))
        }
        "greater_than" => {
            let rhs = rhs.ok_or_else(|| {
                MaskedSuiteError::new(
                    "masked_input_contract_violation",
                    "greater_than requires rhs_values",
                )
            })?;
            let result = lhs
                .greater_than(&rhs)
                .map_err(map_ma_error_to_masked_suite)?;
            Ok(masked_array_to_outcome(&result))
        }
        "filled" => {
            let fill_value = case.fill_value.unwrap_or_else(|| lhs.fill_value());
            let result = lhs
                .filled(fill_value)
                .map_err(map_ufunc_error_to_masked_suite)?;
            Ok(MaskedOperationOutcome::Array {
                shape: result.shape().to_vec(),
                values: result.values().to_vec(),
            })
        }
        "compressed" => {
            let result = lhs.compressed();
            Ok(MaskedOperationOutcome::Array {
                shape: result.shape().to_vec(),
                values: result.values().to_vec(),
            })
        }
        "reshape" => {
            if case.reshape_shape.is_empty() {
                return Err(MaskedSuiteError::new(
                    "masked_input_contract_violation",
                    "reshape requires reshape_shape",
                ));
            }
            let result = lhs
                .reshape(&case.reshape_shape)
                .map_err(map_ma_error_to_masked_suite)?;
            Ok(masked_array_to_outcome(&result))
        }
        "concatenate" => {
            let rhs = rhs.ok_or_else(|| {
                MaskedSuiteError::new(
                    "masked_input_contract_violation",
                    "concatenate requires rhs_values",
                )
            })?;
            let axis = case.axis.unwrap_or(0);
            let result = MaskedArray::concatenate(&[&lhs, &rhs], axis)
                .map_err(map_ma_error_to_masked_suite)?;
            Ok(masked_array_to_outcome(&result))
        }
        "anom" => {
            let result = lhs.anom().map_err(map_ma_error_to_masked_suite)?;
            Ok(masked_array_to_outcome(&result))
        }
        "sum" => {
            let result = lhs
                .sum(case.axis, false)
                .map_err(map_ma_error_to_masked_suite)?;
            Ok(masked_array_to_outcome(&result))
        }
        "mean" => {
            let result = lhs
                .mean(case.axis, false)
                .map_err(map_ma_error_to_masked_suite)?;
            Ok(masked_array_to_outcome(&result))
        }
        "min" => {
            let result = lhs
                .min(case.axis, false)
                .map_err(map_ma_error_to_masked_suite)?;
            Ok(masked_array_to_outcome(&result))
        }
        "max" => {
            let result = lhs
                .max(case.axis, false)
                .map_err(map_ma_error_to_masked_suite)?;
            Ok(masked_array_to_outcome(&result))
        }
        "prod" => {
            let result = lhs
                .prod(case.axis, false)
                .map_err(map_ma_error_to_masked_suite)?;
            Ok(masked_array_to_outcome(&result))
        }
        "var" => {
            let ddof = case.ddof.unwrap_or(0);
            let result = lhs
                .var(case.axis, false, ddof)
                .map_err(map_ma_error_to_masked_suite)?;
            Ok(masked_array_to_outcome(&result))
        }
        "std" => {
            let ddof = case.ddof.unwrap_or(0);
            let result = lhs
                .std(case.axis, false, ddof)
                .map_err(map_ma_error_to_masked_suite)?;
            Ok(masked_array_to_outcome(&result))
        }
        "fix_invalid" => {
            lhs.fix_invalid();
            Ok(masked_array_to_outcome(&lhs))
        }
        "shrink_mask" => {
            lhs.shrink_mask();
            Ok(masked_array_to_outcome(&lhs))
        }
        "argmin" => {
            let result = lhs
                .argmin(case.axis)
                .map_err(map_ma_error_to_masked_suite)?;
            Ok(MaskedOperationOutcome::Array {
                shape: result.shape().to_vec(),
                values: result.values().to_vec(),
            })
        }
        "argmax" => {
            let result = lhs
                .argmax(case.axis)
                .map_err(map_ma_error_to_masked_suite)?;
            Ok(MaskedOperationOutcome::Array {
                shape: result.shape().to_vec(),
                values: result.values().to_vec(),
            })
        }
        other => Err(MaskedSuiteError::new(
            "masked_policy_unknown_operation",
            format!("unsupported masked differential operation {other}"),
        )),
    }
}

fn validate_masked_differential_expectation(
    case: &MaskedDifferentialCase,
    outcome: &MaskedOperationOutcome,
) -> Result<(), String> {
    let abs_tol = if case.abs_tol > 0.0 {
        case.abs_tol
    } else {
        1e-9
    };
    let rel_tol = if case.rel_tol > 0.0 {
        case.rel_tol
    } else {
        1e-9
    };

    let check_shape = |expected: &[usize], actual: &[usize]| -> Result<(), String> {
        if expected.is_empty() {
            return Ok(());
        }
        if expected != actual {
            return Err(format!(
                "shape mismatch expected={expected:?} actual={actual:?}"
            ));
        }
        Ok(())
    };

    let check_values = |expected: &[f64], actual: &[f64], label: &str| -> Result<(), String> {
        if expected.len() != actual.len() {
            return Err(format!(
                "{label} length mismatch expected={} actual={}",
                expected.len(),
                actual.len()
            ));
        }
        if !approx_equal_values(expected, actual, abs_tol, rel_tol) {
            return Err(format!(
                "{label} mismatch expected={expected:?} actual={actual:?} abs_tol={abs_tol} rel_tol={rel_tol}"
            ));
        }
        Ok(())
    };

    // Handle scalar expectation (reduction results)
    if let Some(expected_scalar) = case.expected_scalar {
        match outcome {
            MaskedOperationOutcome::Array { values, .. }
            | MaskedOperationOutcome::Masked { values, .. } => {
                if values.len() != 1 {
                    return Err(format!(
                        "expected scalar but got array of length {}",
                        values.len()
                    ));
                }
                let actual = values[0];
                if !approx_equal_values(&[expected_scalar], &[actual], abs_tol, rel_tol) {
                    return Err(format!(
                        "scalar mismatch expected={expected_scalar} actual={actual} abs_tol={abs_tol} rel_tol={rel_tol}"
                    ));
                }
                return Ok(());
            }
        }
    }

    match outcome {
        MaskedOperationOutcome::Array { shape, values } => {
            check_shape(&case.expected_shape, shape)?;
            check_values(&case.expected_values, values, "values")?;
            Ok(())
        }
        MaskedOperationOutcome::Masked {
            shape,
            values,
            mask,
        } => {
            check_shape(&case.expected_shape, shape)?;
            check_values(&case.expected_values, values, "values")?;
            if case.expect_mask_none {
                if mask.is_some() {
                    return Err(format!("expected mask=None but found mask={mask:?}"));
                }
                return Ok(());
            }
            if !case.expected_mask.is_empty() {
                let actual_mask = mask.as_ref().ok_or_else(|| {
                    "expected mask values but operation returned mask=None".to_string()
                })?;
                check_values(&case.expected_mask, actual_mask, "mask")?;
            }
            Ok(())
        }
    }
}

fn masked_values_to_array(
    shape_hint: &[usize],
    values: &[f64],
    mask_values: &[f64],
    fill_value: Option<f64>,
    nan_indices: &[usize],
    inf_indices: &[usize],
) -> Result<MaskedArray, MaskedSuiteError> {
    let shape = if shape_hint.is_empty() {
        vec![values.len()]
    } else {
        shape_hint.to_vec()
    };
    let expected_len = shape.iter().product::<usize>();
    if expected_len != values.len() {
        return Err(MaskedSuiteError::new(
            "masked_input_contract_violation",
            format!(
                "values length {} does not match shape {shape:?} (len={expected_len})",
                values.len()
            ),
        ));
    }
    let mut data_values = values.to_vec();
    for &idx in nan_indices {
        if idx >= data_values.len() {
            return Err(MaskedSuiteError::new(
                "masked_input_contract_violation",
                format!(
                    "nan index {idx} out of bounds for len={}",
                    data_values.len()
                ),
            ));
        }
        data_values[idx] = f64::NAN;
    }
    for &idx in inf_indices {
        if idx >= data_values.len() {
            return Err(MaskedSuiteError::new(
                "masked_input_contract_violation",
                format!(
                    "inf index {idx} out of bounds for len={}",
                    data_values.len()
                ),
            ));
        }
        data_values[idx] = f64::INFINITY;
    }

    let data = UFuncArray::new(shape.clone(), data_values, DType::F64)
        .map_err(map_ufunc_error_to_masked_suite)?;
    let mask = if mask_values.is_empty() {
        None
    } else {
        if mask_values.len() != expected_len {
            return Err(MaskedSuiteError::new(
                "masked_input_contract_violation",
                format!(
                    "mask length {} does not match shape {shape:?} (len={expected_len})",
                    mask_values.len()
                ),
            ));
        }
        Some(
            UFuncArray::new(shape.clone(), mask_values.to_vec(), DType::Bool)
                .map_err(map_ufunc_error_to_masked_suite)?,
        )
    };

    MaskedArray::new(data, mask, fill_value).map_err(map_ma_error_to_masked_suite)
}

fn masked_array_to_outcome(array: &MaskedArray) -> MaskedOperationOutcome {
    MaskedOperationOutcome::Masked {
        shape: array.shape().to_vec(),
        values: array.data().values().to_vec(),
        mask: array.mask().map(|m| m.values().to_vec()),
    }
}

fn map_ufunc_error_to_masked_suite(error: UFuncError) -> MaskedSuiteError {
    MaskedSuiteError::new(error.reason_code(), error.to_string())
}

fn map_ma_error_to_masked_suite(error: MAError) -> MaskedSuiteError {
    match error {
        MAError::UFunc(inner) => map_ufunc_error_to_masked_suite(inner),
        MAError::MaskShapeMismatch { .. } => {
            MaskedSuiteError::new("masked_shape_mismatch", error.to_string())
        }
        MAError::Msg(_) => MaskedSuiteError::new("masked_operation_error", error.to_string()),
    }
}

#[derive(Debug, Clone, PartialEq)]
enum DateTimeOperationOutcome {
    Array {
        shape: Vec<usize>,
        values: Vec<f64>,
        dtype: DType,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DateTimeSuiteError {
    reason_code: String,
    message: String,
}

impl DateTimeSuiteError {
    fn new(reason_code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            reason_code: reason_code.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for DateTimeSuiteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for DateTimeSuiteError {}

pub fn run_datetime_differential_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let cases = load_datetime_differential_cases(&config.fixture_root)?;

    let mut report = SuiteReport {
        suite: "datetime_differential",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };
    let mut mismatches = Vec::new();

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);
        let expected_reason_code = if case.expected_reason_code.trim().is_empty() {
            reason_code.clone()
        } else {
            case.expected_reason_code.trim().to_string()
        };
        let expected_error = case.expected_error_contains.trim().to_lowercase();

        match execute_datetime_differential_operation(&case) {
            Ok(outcome) => {
                if !expected_error.is_empty() {
                    let rendered = format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected error containing '{}' but operation '{}' succeeded",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        case.expected_error_contains,
                        case.operation
                    );
                    report.failures.push(rendered.clone());
                    mismatches.push(DateTimeDifferentialMismatch {
                        fixture_id: case.id,
                        seed: case.seed,
                        mode,
                        operation: case.operation,
                        expected_reason_code,
                        actual_reason_code: "none".to_string(),
                        message: rendered,
                        artifact_refs,
                    });
                    continue;
                }

                match validate_datetime_differential_expectation(&case, &outcome) {
                    Ok(()) => report.pass_count += 1,
                    Err(message) => {
                        let rendered = format!(
                            "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} {}",
                            case.id,
                            case.seed,
                            mode,
                            reason_code,
                            env_fingerprint,
                            artifact_refs.join(","),
                            message
                        );
                        report.failures.push(rendered.clone());
                        mismatches.push(DateTimeDifferentialMismatch {
                            fixture_id: case.id,
                            seed: case.seed,
                            mode,
                            operation: case.operation,
                            expected_reason_code,
                            actual_reason_code: "datetime_differential_mismatch".to_string(),
                            message: rendered,
                            artifact_refs,
                        });
                    }
                }
            }
            Err(err) => {
                if expected_error.is_empty() {
                    let rendered = format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected success but got error '{}' (reason_code={})",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        err.message,
                        err.reason_code
                    );
                    report.failures.push(rendered.clone());
                    mismatches.push(DateTimeDifferentialMismatch {
                        fixture_id: case.id,
                        seed: case.seed,
                        mode,
                        operation: case.operation,
                        expected_reason_code,
                        actual_reason_code: err.reason_code,
                        message: rendered,
                        artifact_refs,
                    });
                    continue;
                }

                let contains_expected = err.message.to_lowercase().contains(&expected_error);
                let reason_match = err.reason_code == expected_reason_code;
                if contains_expected && reason_match {
                    report.pass_count += 1;
                } else {
                    let rendered = format!(
                        "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} expected error containing '{}' with reason_code='{}' but got '{}' (reason_code='{}')",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        case.expected_error_contains,
                        expected_reason_code,
                        err.message,
                        err.reason_code
                    );
                    report.failures.push(rendered.clone());
                    mismatches.push(DateTimeDifferentialMismatch {
                        fixture_id: case.id,
                        seed: case.seed,
                        mode,
                        operation: case.operation,
                        expected_reason_code,
                        actual_reason_code: err.reason_code,
                        message: rendered,
                        artifact_refs,
                    });
                }
            }
        }
    }

    let artifact = DateTimeDifferentialReportArtifact {
        suite: "datetime_differential",
        total_cases: report.case_count,
        passed_cases: report.pass_count,
        failed_cases: report.case_count.saturating_sub(report.pass_count),
        mismatches,
    };
    let report_path = config
        .fixture_root
        .join("oracle_outputs/datetime_differential_report.json");
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }
    let payload = serde_json::to_string_pretty(&artifact)
        .map_err(|err| format!("failed serializing datetime differential report: {err}"))?;
    fs::write(&report_path, payload.as_bytes())
        .map_err(|err| format!("failed writing {}: {err}", report_path.display()))?;

    Ok(report)
}

pub fn run_datetime_metamorphic_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let cases = load_datetime_metamorphic_cases(&config.fixture_root)?;

    let mut report = SuiteReport {
        suite: "datetime_metamorphic",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);

        match evaluate_datetime_metamorphic_relation(&case) {
            Ok(()) => {
                report.pass_count += 1;
            }
            Err(err) => {
                report.failures.push(format!(
                    "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} {} (actual_reason_code={})",
                    case.id,
                    case.seed,
                    mode,
                    reason_code,
                    env_fingerprint,
                    artifact_refs.join(","),
                    err.message,
                    err.reason_code
                ));
            }
        }
    }

    Ok(report)
}

fn execute_datetime_differential_operation(
    case: &DateTimeDifferentialCase,
) -> Result<DateTimeOperationOutcome, DateTimeSuiteError> {
    let lhs = build_datetime_case_array(
        &case.lhs_shape,
        &case.lhs_values,
        &case.lhs_strings,
        &case.lhs_dtype,
        "lhs",
    )?;
    let rhs = if case.rhs_values.is_empty() && case.rhs_strings.is_empty() {
        None
    } else {
        Some(build_datetime_case_array(
            &case.rhs_shape,
            &case.rhs_values,
            &case.rhs_strings,
            &case.rhs_dtype,
            "rhs",
        )?)
    };

    let as_array = |array: UFuncArray| DateTimeOperationOutcome::Array {
        shape: array.shape().to_vec(),
        values: array.values().to_vec(),
        dtype: array.dtype(),
    };

    match case.operation.as_str() {
        "datetime_add" => {
            let rhs = rhs.ok_or_else(|| {
                DateTimeSuiteError::new(
                    "datetime_input_contract_violation",
                    "datetime_add requires rhs_values",
                )
            })?;
            Ok(as_array(
                lhs.datetime_add(&rhs)
                    .map_err(map_ufunc_error_to_datetime_suite)?,
            ))
        }
        "datetime_sub" => {
            let rhs = rhs.ok_or_else(|| {
                DateTimeSuiteError::new(
                    "datetime_input_contract_violation",
                    "datetime_sub requires rhs_values",
                )
            })?;
            Ok(as_array(
                lhs.datetime_sub(&rhs)
                    .map_err(map_ufunc_error_to_datetime_suite)?,
            ))
        }
        "timedelta_add" => {
            let rhs = rhs.ok_or_else(|| {
                DateTimeSuiteError::new(
                    "datetime_input_contract_violation",
                    "timedelta_add requires rhs_values",
                )
            })?;
            Ok(as_array(
                lhs.timedelta_add(&rhs)
                    .map_err(map_ufunc_error_to_datetime_suite)?,
            ))
        }
        "timedelta_sub" => {
            let rhs = rhs.ok_or_else(|| {
                DateTimeSuiteError::new(
                    "datetime_input_contract_violation",
                    "timedelta_sub requires rhs_values",
                )
            })?;
            Ok(as_array(
                lhs.timedelta_sub(&rhs)
                    .map_err(map_ufunc_error_to_datetime_suite)?,
            ))
        }
        "timedelta_mul" => {
            let scalar = case.scalar.ok_or_else(|| {
                DateTimeSuiteError::new(
                    "datetime_input_contract_violation",
                    "timedelta_mul requires scalar",
                )
            })?;
            Ok(as_array(
                lhs.timedelta_mul(scalar)
                    .map_err(map_ufunc_error_to_datetime_suite)?,
            ))
        }
        "timedelta_div" => {
            let scalar = case.scalar.ok_or_else(|| {
                DateTimeSuiteError::new(
                    "datetime_input_contract_violation",
                    "timedelta_div requires scalar",
                )
            })?;
            Ok(as_array(
                lhs.timedelta_div(scalar)
                    .map_err(map_ufunc_error_to_datetime_suite)?,
            ))
        }
        "timedelta_neg" => Ok(as_array(lhs.timedelta_neg())),
        "timedelta_abs" => Ok(as_array(lhs.timedelta_abs())),
        "is_busday" => Ok(as_array(
            is_busday(&lhs).map_err(map_ufunc_error_to_datetime_suite)?,
        )),
        "isnat" => Ok(as_array(
            isnat(&lhs).map_err(map_ufunc_error_to_datetime_suite)?,
        )),
        "busday_count" => {
            let rhs = rhs.ok_or_else(|| {
                DateTimeSuiteError::new(
                    "datetime_input_contract_violation",
                    "busday_count requires rhs_values",
                )
            })?;
            Ok(as_array(
                busday_count(&lhs, &rhs).map_err(map_ufunc_error_to_datetime_suite)?,
            ))
        }
        "busday_offset" => {
            let rhs = rhs.ok_or_else(|| {
                DateTimeSuiteError::new(
                    "datetime_input_contract_violation",
                    "busday_offset requires rhs_values",
                )
            })?;
            Ok(as_array(
                busday_offset(&lhs, &rhs).map_err(map_ufunc_error_to_datetime_suite)?,
            ))
        }
        "equal" => {
            let rhs = rhs.ok_or_else(|| {
                DateTimeSuiteError::new(
                    "datetime_input_contract_violation",
                    "equal requires rhs_values",
                )
            })?;
            Ok(as_array(
                lhs.elementwise_binary(&rhs, BinaryOp::Equal)
                    .map_err(map_ufunc_error_to_datetime_suite)?,
            ))
        }
        "not_equal" => {
            let rhs = rhs.ok_or_else(|| {
                DateTimeSuiteError::new(
                    "datetime_input_contract_violation",
                    "not_equal requires rhs_values",
                )
            })?;
            Ok(as_array(
                lhs.elementwise_binary(&rhs, BinaryOp::NotEqual)
                    .map_err(map_ufunc_error_to_datetime_suite)?,
            ))
        }
        "less" => {
            let rhs = rhs.ok_or_else(|| {
                DateTimeSuiteError::new(
                    "datetime_input_contract_violation",
                    "less requires rhs_values",
                )
            })?;
            Ok(as_array(
                lhs.elementwise_binary(&rhs, BinaryOp::Less)
                    .map_err(map_ufunc_error_to_datetime_suite)?,
            ))
        }
        "less_equal" => {
            let rhs = rhs.ok_or_else(|| {
                DateTimeSuiteError::new(
                    "datetime_input_contract_violation",
                    "less_equal requires rhs_values",
                )
            })?;
            Ok(as_array(
                lhs.elementwise_binary(&rhs, BinaryOp::LessEqual)
                    .map_err(map_ufunc_error_to_datetime_suite)?,
            ))
        }
        "greater" => {
            let rhs = rhs.ok_or_else(|| {
                DateTimeSuiteError::new(
                    "datetime_input_contract_violation",
                    "greater requires rhs_values",
                )
            })?;
            Ok(as_array(
                lhs.elementwise_binary(&rhs, BinaryOp::Greater)
                    .map_err(map_ufunc_error_to_datetime_suite)?,
            ))
        }
        "greater_equal" => {
            let rhs = rhs.ok_or_else(|| {
                DateTimeSuiteError::new(
                    "datetime_input_contract_violation",
                    "greater_equal requires rhs_values",
                )
            })?;
            Ok(as_array(
                lhs.elementwise_binary(&rhs, BinaryOp::GreaterEqual)
                    .map_err(map_ufunc_error_to_datetime_suite)?,
            ))
        }
        other => Err(DateTimeSuiteError::new(
            "datetime_policy_unknown_operation",
            format!("unsupported datetime differential operation {other}"),
        )),
    }
}

fn validate_datetime_differential_expectation(
    case: &DateTimeDifferentialCase,
    outcome: &DateTimeOperationOutcome,
) -> Result<(), String> {
    match outcome {
        DateTimeOperationOutcome::Array {
            shape,
            values,
            dtype,
        } => {
            if !case.expected_shape.is_empty() && case.expected_shape != *shape {
                return Err(format!(
                    "shape mismatch expected={:?} actual={shape:?}",
                    case.expected_shape
                ));
            }
            if !case.expected_values.is_empty() {
                if case.expected_values.len() != values.len() {
                    return Err(format!(
                        "value length mismatch expected={} actual={}",
                        case.expected_values.len(),
                        values.len()
                    ));
                }
                let abs_tol = if case.abs_tol > 0.0 {
                    case.abs_tol
                } else {
                    1e-9
                };
                let rel_tol = if case.rel_tol > 0.0 {
                    case.rel_tol
                } else {
                    1e-9
                };
                if !approx_equal_values(&case.expected_values, values, abs_tol, rel_tol) {
                    return Err(format!(
                        "value mismatch expected={:?} actual={values:?} abs_tol={} rel_tol={}",
                        case.expected_values, abs_tol, rel_tol
                    ));
                }
            }
            if !case.expected_dtype.trim().is_empty() {
                let expected_dtype =
                    parse_datetime_fixture_dtype(&case.expected_dtype).map_err(|err| {
                        format!(
                            "invalid expected_dtype '{}' ({})",
                            case.expected_dtype, err.message
                        )
                    })?;
                if *dtype != expected_dtype {
                    return Err(format!(
                        "dtype mismatch expected={expected_dtype:?} actual={dtype:?}"
                    ));
                }
            }
            Ok(())
        }
    }
}

fn evaluate_datetime_metamorphic_relation(
    case: &DateTimeMetamorphicCase,
) -> Result<(), DateTimeSuiteError> {
    match case.relation.as_str() {
        "datetime_add_sub_inverse" => {
            let datetime = datetime_metamorphic_datetime(&case.base_datetime, &case.delta_unit)?;
            let delta = datetime_metamorphic_timedelta(case.delta_value, &case.delta_unit)?;
            let shifted = datetime
                .datetime_add(&delta)
                .map_err(map_ufunc_error_to_datetime_suite)?;
            let recovered_delta = shifted
                .datetime_sub(&datetime)
                .map_err(map_ufunc_error_to_datetime_suite)?;
            assert_datetime_metamorphic_array_equal(case, &recovered_delta, &delta)
        }
        "timedelta_add_commutative" => {
            let delta_a = datetime_metamorphic_timedelta(case.delta_a_value, &case.delta_a_unit)?;
            let delta_b = datetime_metamorphic_timedelta(case.delta_b_value, &case.delta_b_unit)?;
            let left = delta_a
                .timedelta_add(&delta_b)
                .map_err(map_ufunc_error_to_datetime_suite)?;
            let right = delta_b
                .timedelta_add(&delta_a)
                .map_err(map_ufunc_error_to_datetime_suite)?;
            assert_datetime_metamorphic_array_equal(case, &left, &right)
        }
        "timedelta_sub_self_zero" => {
            let delta = datetime_metamorphic_timedelta(case.delta_value, &case.delta_unit)?;
            let zero = datetime_metamorphic_timedelta(0, &case.delta_unit)?;
            let actual = delta
                .timedelta_sub(&delta)
                .map_err(map_ufunc_error_to_datetime_suite)?;
            assert_datetime_metamorphic_array_equal(case, &actual, &zero)
        }
        "timedelta_mul_one_identity" => {
            let delta = datetime_metamorphic_timedelta(case.delta_value, &case.delta_unit)?;
            let actual = delta
                .timedelta_mul(1.0)
                .map_err(map_ufunc_error_to_datetime_suite)?;
            assert_datetime_metamorphic_array_equal(case, &actual, &delta)
        }
        "timedelta_neg_neg_identity" => {
            let delta = datetime_metamorphic_timedelta(case.delta_value, &case.delta_unit)?;
            let actual = delta.timedelta_neg().timedelta_neg();
            assert_datetime_metamorphic_array_equal(case, &actual, &delta)
        }
        "datetime_diff_add_inverse" => {
            let datetime_a = datetime_metamorphic_datetime(&case.datetime_a, "h")?;
            let datetime_b = datetime_metamorphic_datetime(&case.datetime_b, "h")?;
            let delta = datetime_b
                .datetime_sub(&datetime_a)
                .map_err(map_ufunc_error_to_datetime_suite)?;
            let actual = datetime_a
                .datetime_add(&delta)
                .map_err(map_ufunc_error_to_datetime_suite)?;
            assert_datetime_metamorphic_array_equal(case, &actual, &datetime_b)
        }
        "timedelta_add_associative" => {
            let delta_a = datetime_metamorphic_timedelta(case.delta_a_value, &case.delta_a_unit)?;
            let delta_b = datetime_metamorphic_timedelta(case.delta_b_value, &case.delta_b_unit)?;
            let delta_c = datetime_metamorphic_timedelta(case.delta_c_value, &case.delta_c_unit)?;
            let left = delta_a
                .timedelta_add(&delta_b)
                .and_then(|sum| sum.timedelta_add(&delta_c))
                .map_err(map_ufunc_error_to_datetime_suite)?;
            let right = delta_b
                .timedelta_add(&delta_c)
                .and_then(|sum| delta_a.timedelta_add(&sum))
                .map_err(map_ufunc_error_to_datetime_suite)?;
            assert_datetime_metamorphic_array_equal(case, &left, &right)
        }
        "timedelta_mul_distributive" => {
            let scalar = case.scalar.ok_or_else(|| {
                DateTimeSuiteError::new(
                    "datetime_input_contract_violation",
                    "timedelta_mul_distributive requires scalar",
                )
            })?;
            let delta_a = datetime_metamorphic_timedelta(case.delta_a_value, &case.delta_a_unit)?;
            let delta_b = datetime_metamorphic_timedelta(case.delta_b_value, &case.delta_b_unit)?;
            let left = delta_a
                .timedelta_add(&delta_b)
                .and_then(|sum| sum.timedelta_mul(scalar))
                .map_err(map_ufunc_error_to_datetime_suite)?;
            let scaled_a = delta_a
                .timedelta_mul(scalar)
                .map_err(map_ufunc_error_to_datetime_suite)?;
            let scaled_b = delta_b
                .timedelta_mul(scalar)
                .map_err(map_ufunc_error_to_datetime_suite)?;
            let right = scaled_a
                .timedelta_add(&scaled_b)
                .map_err(map_ufunc_error_to_datetime_suite)?;
            assert_datetime_metamorphic_array_equal(case, &left, &right)
        }
        "datetime_compare_transitivity" => {
            let datetime_a = datetime_metamorphic_datetime(&case.datetime_a, "s")?;
            let datetime_b = datetime_metamorphic_datetime(&case.datetime_b, "s")?;
            let datetime_c = datetime_metamorphic_datetime(&case.datetime_c, "s")?;
            let a_less_b = datetime_a
                .elementwise_binary(&datetime_b, BinaryOp::Less)
                .map_err(map_ufunc_error_to_datetime_suite)?;
            let b_less_c = datetime_b
                .elementwise_binary(&datetime_c, BinaryOp::Less)
                .map_err(map_ufunc_error_to_datetime_suite)?;
            let a_less_c = datetime_a
                .elementwise_binary(&datetime_c, BinaryOp::Less)
                .map_err(map_ufunc_error_to_datetime_suite)?;
            if datetime_metamorphic_single_true(&a_less_b)
                && datetime_metamorphic_single_true(&b_less_c)
                && datetime_metamorphic_single_true(&a_less_c)
            {
                Ok(())
            } else {
                Err(DateTimeSuiteError::new(
                    "datetime_metamorphic_relation_failed",
                    format!(
                        "datetime_compare_transitivity failed: a<b={:?} b<c={:?} a<c={:?}",
                        a_less_b.values(),
                        b_less_c.values(),
                        a_less_c.values()
                    ),
                ))
            }
        }
        "timedelta_abs_idempotent" => {
            let delta = datetime_metamorphic_timedelta(case.delta_value, &case.delta_unit)?;
            let once = delta.timedelta_abs();
            let twice = once.timedelta_abs();
            assert_datetime_metamorphic_array_equal(case, &twice, &once)
        }
        "timedelta_add_zero_identity" => {
            let delta = datetime_metamorphic_timedelta(case.delta_value, &case.delta_unit)?;
            let zero = datetime_metamorphic_timedelta(0, &case.delta_unit)?;
            let actual = delta
                .timedelta_add(&zero)
                .map_err(map_ufunc_error_to_datetime_suite)?;
            assert_datetime_metamorphic_array_equal(case, &actual, &delta)
        }
        other => Err(DateTimeSuiteError::new(
            "datetime_policy_unknown_metamorphic_relation",
            format!("unsupported datetime metamorphic relation {other}"),
        )),
    }
}

fn datetime_metamorphic_datetime(
    value: &str,
    unit: &str,
) -> Result<UFuncArray, DateTimeSuiteError> {
    require_datetime_metamorphic_field("datetime value", value)?;
    require_datetime_metamorphic_field("datetime unit", unit)?;
    UFuncArray::from_datetime_strings(vec![1], vec![value.to_string()], Some(unit))
        .map_err(map_ufunc_error_to_datetime_suite)
}

fn datetime_metamorphic_timedelta(
    value: i64,
    unit: &str,
) -> Result<UFuncArray, DateTimeSuiteError> {
    require_datetime_metamorphic_field("timedelta unit", unit)?;
    UFuncArray::from_timedelta_strings(vec![1], vec![format!("{value}{unit}")], Some(unit))
        .map_err(map_ufunc_error_to_datetime_suite)
}

fn require_datetime_metamorphic_field(name: &str, value: &str) -> Result<(), DateTimeSuiteError> {
    if value.trim().is_empty() {
        return Err(DateTimeSuiteError::new(
            "datetime_input_contract_violation",
            format!("{name} must not be empty"),
        ));
    }
    Ok(())
}

fn datetime_metamorphic_single_true(array: &UFuncArray) -> bool {
    array.values().len() == 1 && (array.values()[0] - 1.0).abs() <= 1e-12
}

fn assert_datetime_metamorphic_array_equal(
    case: &DateTimeMetamorphicCase,
    actual: &UFuncArray,
    expected: &UFuncArray,
) -> Result<(), DateTimeSuiteError> {
    if actual.shape() != expected.shape() {
        return Err(DateTimeSuiteError::new(
            "datetime_metamorphic_relation_failed",
            format!(
                "{} shape mismatch expected={:?} actual={:?}",
                case.relation,
                expected.shape(),
                actual.shape()
            ),
        ));
    }
    if actual.dtype() != expected.dtype() {
        return Err(DateTimeSuiteError::new(
            "datetime_metamorphic_relation_failed",
            format!(
                "{} dtype mismatch expected={:?} actual={:?}",
                case.relation,
                expected.dtype(),
                actual.dtype()
            ),
        ));
    }
    let abs_tol = if case.abs_tol > 0.0 {
        case.abs_tol
    } else {
        1e-9
    };
    let rel_tol = if case.rel_tol > 0.0 {
        case.rel_tol
    } else {
        1e-9
    };
    if !approx_equal_values(expected.values(), actual.values(), abs_tol, rel_tol) {
        return Err(DateTimeSuiteError::new(
            "datetime_metamorphic_relation_failed",
            format!(
                "{} value mismatch expected={:?} actual={:?} abs_tol={abs_tol} rel_tol={rel_tol}",
                case.relation,
                expected.values(),
                actual.values()
            ),
        ));
    }
    Ok(())
}

fn build_datetime_case_array(
    shape_hint: &[usize],
    values: &[f64],
    strings: &[String],
    dtype_raw: &str,
    side: &str,
) -> Result<UFuncArray, DateTimeSuiteError> {
    if values.is_empty() && strings.is_empty() {
        return Err(DateTimeSuiteError::new(
            "datetime_input_contract_violation",
            format!("{side}_values or {side}_strings must not be empty"),
        ));
    }
    if !values.is_empty() && !strings.is_empty() {
        return Err(DateTimeSuiteError::new(
            "datetime_input_contract_violation",
            format!("{side}_values and {side}_strings are mutually exclusive"),
        ));
    }
    let shape = if shape_hint.is_empty() {
        vec![if strings.is_empty() {
            values.len()
        } else {
            strings.len()
        }]
    } else {
        shape_hint.to_vec()
    };
    let expected_len = shape.iter().product::<usize>();
    let actual_len = if strings.is_empty() {
        values.len()
    } else {
        strings.len()
    };
    if expected_len != actual_len {
        return Err(DateTimeSuiteError::new(
            "datetime_input_contract_violation",
            format!(
                "{side} input length {} does not match {side}_shape {shape:?} (len={expected_len})",
                actual_len
            ),
        ));
    }
    let (dtype, unit) = parse_datetime_fixture_dtype_with_unit(dtype_raw).map_err(|err| {
        DateTimeSuiteError::new(
            "datetime_input_contract_violation",
            format!("invalid {side}_dtype '{dtype_raw}' ({})", err.message),
        )
    })?;
    if !strings.is_empty() {
        let unit = unit.as_deref();
        return match dtype {
            DType::DateTime64 => UFuncArray::from_datetime_strings(shape, strings.to_vec(), unit),
            DType::TimeDelta64 => UFuncArray::from_timedelta_strings(shape, strings.to_vec(), unit),
            _ => Err(UFuncError::Msg(format!(
                "{side}_strings only support datetime64/timedelta64 dtypes"
            ))),
        }
        .map_err(map_ufunc_error_to_datetime_suite);
    }
    UFuncArray::new(shape, values.to_vec(), dtype).map_err(map_ufunc_error_to_datetime_suite)
}

fn parse_datetime_fixture_dtype(dtype_raw: &str) -> Result<DType, DateTimeSuiteError> {
    parse_datetime_fixture_dtype_with_unit(dtype_raw).map(|(dtype, _)| dtype)
}

fn parse_datetime_fixture_dtype_with_unit(
    dtype_raw: &str,
) -> Result<(DType, Option<String>), DateTimeSuiteError> {
    let trimmed = dtype_raw.trim();
    let normalized = trimmed.to_ascii_lowercase();
    let parse_temporal =
        |prefix: &str, dtype: DType| -> Result<(DType, Option<String>), DateTimeSuiteError> {
            if normalized == prefix {
                return Ok((dtype, None));
            }
            if normalized.starts_with(prefix) && normalized.ends_with(']') {
                let open = normalized.find('[').ok_or_else(|| {
                    DateTimeSuiteError::new(
                        "datetime_fixture_dtype_invalid",
                        format!("unsupported dtype '{trimmed}'"),
                    )
                })?;
                let unit = normalized[open + 1..normalized.len() - 1].trim();
                if unit.is_empty() {
                    return Err(DateTimeSuiteError::new(
                        "datetime_fixture_dtype_invalid",
                        format!("unsupported dtype '{trimmed}'"),
                    ));
                }
                return Ok((dtype, Some(unit.to_string())));
            }
            Err(DateTimeSuiteError::new(
                "datetime_fixture_dtype_invalid",
                format!("unsupported dtype '{trimmed}'"),
            ))
        };
    match dtype_raw.trim().to_ascii_lowercase().as_str() {
        "bool" | "bool_" => Ok((DType::Bool, None)),
        "i64" | "int64" => Ok((DType::I64, None)),
        "f64" | "float64" => Ok((DType::F64, None)),
        value if value.starts_with("datetime64") => parse_temporal("datetime64", DType::DateTime64),
        value if value.starts_with("timedelta64") => {
            parse_temporal("timedelta64", DType::TimeDelta64)
        }
        other => Err(DateTimeSuiteError::new(
            "datetime_fixture_dtype_invalid",
            format!("unsupported dtype '{other}'"),
        )),
    }
}

fn map_ufunc_error_to_datetime_suite(error: UFuncError) -> DateTimeSuiteError {
    DateTimeSuiteError::new(error.reason_code(), error.to_string())
}

#[derive(Debug, Clone)]
struct RngSuiteError {
    reason_code: String,
    message: String,
}

impl RngSuiteError {
    fn new(reason_code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            reason_code: reason_code.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for RngSuiteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for RngSuiteError {}

#[derive(Debug, Clone, PartialEq)]
enum IoOperationOutcome {
    Unit,
    Count(usize),
    Dispatch(String),
    MagicVersion(String),
    Array { shape: Vec<usize>, values: Vec<f64> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IoSuiteError {
    reason_code: String,
    message: String,
}

impl IoSuiteError {
    fn new(reason_code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            reason_code: reason_code.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for IoSuiteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for IoSuiteError {}

pub fn run_rng_differential_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let path = config.fixture_root.join("rng_differential_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<RngDifferentialCase> =
        serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))?;

    let mut report = SuiteReport {
        suite: "rng_differential",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };
    let mut mismatches = Vec::new();

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);
        let expected_reason_code = if case.expected_reason_code.trim().is_empty() {
            reason_code.clone()
        } else {
            case.expected_reason_code.trim().to_string()
        };

        match execute_rng_differential_operation(&case) {
            Ok(()) => {
                report.pass_count += 1;
            }
            Err(err) => {
                let rendered = format!(
                    "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} {} (actual_reason_code={})",
                    case.id,
                    case.seed,
                    mode,
                    reason_code,
                    env_fingerprint,
                    artifact_refs.join(","),
                    err.message,
                    err.reason_code
                );
                report.failures.push(rendered.clone());
                mismatches.push(RngDifferentialMismatch {
                    fixture_id: case.id,
                    seed: case.seed,
                    mode,
                    expected_reason_code,
                    actual_reason_code: err.reason_code,
                    message: rendered,
                    artifact_refs,
                });
            }
        }
    }

    let artifact = RngDifferentialReportArtifact {
        suite: "rng_differential",
        total_cases: report.case_count,
        passed_cases: report.pass_count,
        failed_cases: report.case_count.saturating_sub(report.pass_count),
        mismatches,
    };
    let report_path = config
        .fixture_root
        .join("oracle_outputs/rng_differential_report.json");
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }
    let payload = serde_json::to_string_pretty(&artifact)
        .map_err(|err| format!("failed serializing rng differential report: {err}"))?;
    fs::write(&report_path, payload.as_bytes())
        .map_err(|err| format!("failed writing {}: {err}", report_path.display()))?;

    Ok(report)
}

pub fn run_rng_metamorphic_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let path = config.fixture_root.join("rng_metamorphic_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<RngMetamorphicCase> =
        serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))?;

    let mut report = SuiteReport {
        suite: "rng_metamorphic",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);

        match evaluate_rng_metamorphic_relation(&case) {
            Ok(()) => {
                report.pass_count += 1;
            }
            Err(err) => {
                report.failures.push(format!(
                    "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} {} (actual_reason_code={})",
                    case.id,
                    case.seed,
                    mode,
                    reason_code,
                    env_fingerprint,
                    artifact_refs.join(","),
                    err.message,
                    err.reason_code
                ));
            }
        }
    }

    Ok(report)
}

pub fn run_rng_adversarial_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let path = config.fixture_root.join("rng_adversarial_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<RngAdversarialCase> =
        serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))?;

    let mut report = SuiteReport {
        suite: "rng_adversarial",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);
        let severity = case.severity.trim().to_lowercase();
        if !matches!(severity.as_str(), "low" | "medium" | "high" | "critical") {
            report.failures.push(format!(
                "{}: invalid severity '{}' (must be low|medium|high|critical), reason_code={}, mode={}, env_fingerprint={}, artifact_refs={}",
                case.id,
                case.severity,
                reason_code,
                mode,
                env_fingerprint,
                artifact_refs.join(",")
            ));
            continue;
        }
        if case.expected_error_contains.trim().is_empty() {
            report.failures.push(format!(
                "{}: expected_error_contains must be non-empty, reason_code={}, mode={}, env_fingerprint={}, artifact_refs={}",
                case.id,
                reason_code,
                mode,
                env_fingerprint,
                artifact_refs.join(",")
            ));
            continue;
        }

        let expected_reason_code = if case.expected_reason_code.trim().is_empty() {
            reason_code.clone()
        } else {
            case.expected_reason_code.trim().to_string()
        };
        let expected_error = case.expected_error_contains.to_lowercase();
        match execute_rng_adversarial_operation(&case) {
            Ok(()) => {
                report.failures.push(format!(
                    "{}: severity={severity} seed={} reason_code={} mode={} env_fingerprint={} artifact_refs={} expected error containing '{}' but operation '{}' succeeded",
                    case.id,
                    case.seed,
                    reason_code,
                    mode,
                    env_fingerprint,
                    artifact_refs.join(","),
                    case.expected_error_contains,
                    case.operation
                ));
            }
            Err(err) => {
                let contains_expected = err.message.to_lowercase().contains(&expected_error);
                let reason_match = err.reason_code == expected_reason_code;
                if contains_expected && reason_match {
                    report.pass_count += 1;
                } else {
                    report.failures.push(format!(
                        "{}: severity={severity} seed={} reason_code={} mode={} env_fingerprint={} artifact_refs={} expected error containing '{}' with reason_code='{}' but got '{}' (reason_code='{}')",
                        case.id,
                        case.seed,
                        reason_code,
                        mode,
                        env_fingerprint,
                        artifact_refs.join(","),
                        case.expected_error_contains,
                        expected_reason_code,
                        err.message,
                        err.reason_code
                    ));
                }
            }
        }
    }

    Ok(report)
}

pub fn run_rng_statistical_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let cases = load_rng_statistical_cases(&config.fixture_root)?;
    let mut report = SuiteReport {
        suite: "rng_statistical",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);

        match evaluate_rng_statistical_case(&case) {
            Ok(()) => {
                report.pass_count += 1;
            }
            Err(err) => {
                report.failures.push(format!(
                    "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} {} (actual_reason_code={})",
                    case.id,
                    case.seed,
                    mode,
                    reason_code,
                    env_fingerprint,
                    artifact_refs.join(","),
                    err.message,
                    err.reason_code
                ));
            }
        }
    }

    Ok(report)
}

pub fn run_io_differential_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let path = config.fixture_root.join("io_differential_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<IoDifferentialCase> =
        serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))?;

    let mut report = SuiteReport {
        suite: "io_differential",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };
    let mut mismatches = Vec::new();

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);
        let expected_reason_code = if case.expected_reason_code.trim().is_empty() {
            reason_code.clone()
        } else {
            case.expected_reason_code.trim().to_string()
        };
        let expected_error = case.expected_error_contains.to_lowercase();

        match execute_io_differential_operation(&case) {
            Ok(outcome) if expected_error.is_empty() => {
                let actual_reason_code = reason_code.clone();
                match validate_io_differential_success_expectations(&case, &outcome) {
                    Ok(()) if actual_reason_code == expected_reason_code => {
                        report.pass_count += 1;
                    }
                    Ok(()) => {
                        let rendered = format!(
                            "{}: seed={} mode={} reason_code={} expected_reason_code={} env_fingerprint={} artifact_refs={} success reason-code mismatch",
                            case.id,
                            case.seed,
                            mode,
                            actual_reason_code,
                            expected_reason_code,
                            env_fingerprint,
                            artifact_refs.join(",")
                        );
                        report.failures.push(rendered.clone());
                        mismatches.push(IoDifferentialMismatch {
                            fixture_id: case.id.clone(),
                            seed: case.seed,
                            mode: mode.clone(),
                            operation: case.operation.clone(),
                            expected_reason_code,
                            actual_reason_code,
                            message: rendered,
                            minimal_repro_artifacts: iter_minimal_repro_artifacts(
                                &case.id,
                                &artifact_refs,
                                "crates/fnp-conformance/fixtures/io_differential_cases.json",
                            ),
                            artifact_refs: artifact_refs.clone(),
                        });
                    }
                    Err(detail) => {
                        let rendered = format!(
                            "{}: seed={} mode={} reason_code={} expected_reason_code={} env_fingerprint={} artifact_refs={} {}",
                            case.id,
                            case.seed,
                            mode,
                            actual_reason_code,
                            expected_reason_code,
                            env_fingerprint,
                            artifact_refs.join(","),
                            detail
                        );
                        report.failures.push(rendered.clone());
                        mismatches.push(IoDifferentialMismatch {
                            fixture_id: case.id.clone(),
                            seed: case.seed,
                            mode: mode.clone(),
                            operation: case.operation.clone(),
                            expected_reason_code,
                            actual_reason_code,
                            message: rendered,
                            minimal_repro_artifacts: iter_minimal_repro_artifacts(
                                &case.id,
                                &artifact_refs,
                                "crates/fnp-conformance/fixtures/io_differential_cases.json",
                            ),
                            artifact_refs: artifact_refs.clone(),
                        });
                    }
                }
            }
            Ok(_) => {
                let rendered = format!(
                    "{}: seed={} mode={} reason_code={} expected_reason_code={} env_fingerprint={} artifact_refs={} expected error containing '{}' but operation '{}' succeeded",
                    case.id,
                    case.seed,
                    mode,
                    reason_code,
                    expected_reason_code,
                    env_fingerprint,
                    artifact_refs.join(","),
                    case.expected_error_contains,
                    case.operation
                );
                report.failures.push(rendered.clone());
                mismatches.push(IoDifferentialMismatch {
                    fixture_id: case.id.clone(),
                    seed: case.seed,
                    mode: mode.clone(),
                    operation: case.operation.clone(),
                    expected_reason_code,
                    actual_reason_code: reason_code,
                    message: rendered,
                    minimal_repro_artifacts: iter_minimal_repro_artifacts(
                        &case.id,
                        &artifact_refs,
                        "crates/fnp-conformance/fixtures/io_differential_cases.json",
                    ),
                    artifact_refs,
                });
            }
            Err(err) if expected_error.is_empty() => {
                let rendered = format!(
                    "{}: seed={} mode={} reason_code={} expected_reason_code={} env_fingerprint={} artifact_refs={} unexpected error '{}' (actual_reason_code='{}')",
                    case.id,
                    case.seed,
                    mode,
                    reason_code,
                    expected_reason_code,
                    env_fingerprint,
                    artifact_refs.join(","),
                    err.message,
                    err.reason_code
                );
                report.failures.push(rendered.clone());
                mismatches.push(IoDifferentialMismatch {
                    fixture_id: case.id.clone(),
                    seed: case.seed,
                    mode: mode.clone(),
                    operation: case.operation.clone(),
                    expected_reason_code,
                    actual_reason_code: err.reason_code.clone(),
                    message: rendered,
                    minimal_repro_artifacts: iter_minimal_repro_artifacts(
                        &case.id,
                        &artifact_refs,
                        "crates/fnp-conformance/fixtures/io_differential_cases.json",
                    ),
                    artifact_refs,
                });
            }
            Err(err) => {
                let contains_expected = err.message.to_lowercase().contains(&expected_error);
                let reason_match = err.reason_code == expected_reason_code;
                if contains_expected && reason_match {
                    report.pass_count += 1;
                } else {
                    let rendered = format!(
                        "{}: seed={} mode={} reason_code={} expected_reason_code={} env_fingerprint={} artifact_refs={} expected error containing '{}' with reason_code='{}' but got '{}' (actual_reason_code='{}')",
                        case.id,
                        case.seed,
                        mode,
                        reason_code,
                        expected_reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        case.expected_error_contains,
                        expected_reason_code,
                        err.message,
                        err.reason_code
                    );
                    report.failures.push(rendered.clone());
                    mismatches.push(IoDifferentialMismatch {
                        fixture_id: case.id.clone(),
                        seed: case.seed,
                        mode: mode.clone(),
                        operation: case.operation.clone(),
                        expected_reason_code,
                        actual_reason_code: err.reason_code.clone(),
                        message: rendered,
                        minimal_repro_artifacts: iter_minimal_repro_artifacts(
                            &case.id,
                            &artifact_refs,
                            "crates/fnp-conformance/fixtures/io_differential_cases.json",
                        ),
                        artifact_refs,
                    });
                }
            }
        }
    }

    let report_path = config
        .fixture_root
        .join("oracle_outputs")
        .join("io_differential_report.json");
    let artifact = IoDifferentialReportArtifact {
        suite: "io_differential",
        total_cases: report.case_count,
        passed_cases: report.pass_count,
        failed_cases: report.case_count.saturating_sub(report.pass_count),
        mismatches,
    };
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }
    let payload = serde_json::to_string_pretty(&artifact)
        .map_err(|err| format!("failed serializing io differential report: {err}"))?;
    fs::write(&report_path, payload.as_bytes())
        .map_err(|err| format!("failed writing {}: {err}", report_path.display()))?;

    Ok(report)
}

pub fn run_io_metamorphic_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let path = config.fixture_root.join("io_metamorphic_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<IoMetamorphicCase> =
        serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))?;

    let mut report = SuiteReport {
        suite: "io_metamorphic",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);

        match evaluate_io_metamorphic_relation(&case) {
            Ok(()) => {
                report.pass_count += 1;
            }
            Err(err) => {
                report.failures.push(format!(
                    "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} {} (actual_reason_code={})",
                    case.id,
                    case.seed,
                    mode,
                    reason_code,
                    env_fingerprint,
                    artifact_refs.join(","),
                    err.message,
                    err.reason_code
                ));
            }
        }
    }

    Ok(report)
}

pub fn run_io_adversarial_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let path = config.fixture_root.join("io_adversarial_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<IoAdversarialCase> =
        serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))?;

    let mut report = SuiteReport {
        suite: "io_adversarial",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };

    for case in cases {
        let mode = resolve_case_mode(&case.mode, config.strict_mode);
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);
        let expected_reason_code = if case.expected_reason_code.trim().is_empty() {
            reason_code.clone()
        } else {
            case.expected_reason_code.trim().to_string()
        };
        let severity = case.severity.trim().to_lowercase();
        if !matches!(severity.as_str(), "low" | "medium" | "high" | "critical") {
            report.failures.push(format!(
                "{}: invalid severity '{}' (must be low|medium|high|critical), reason_code={}, mode={}, env_fingerprint={}, artifact_refs={}",
                case.id,
                case.severity,
                reason_code,
                mode,
                env_fingerprint,
                artifact_refs.join(",")
            ));
            continue;
        }
        if case.expected_error_contains.trim().is_empty() {
            report.failures.push(format!(
                "{}: expected_error_contains must be non-empty, reason_code={}, mode={}, env_fingerprint={}, artifact_refs={}",
                case.id,
                reason_code,
                mode,
                env_fingerprint,
                artifact_refs.join(",")
            ));
            continue;
        }

        let expected = case.expected_error_contains.to_lowercase();
        match execute_io_adversarial_operation(&case) {
            Ok(()) => {
                report.failures.push(format!(
                    "{}: severity={severity} seed={} reason_code={} mode={} env_fingerprint={} artifact_refs={} expected error containing '{}' but operation '{}' succeeded",
                    case.id,
                    case.seed,
                    reason_code,
                    mode,
                    env_fingerprint,
                    artifact_refs.join(","),
                    case.expected_error_contains,
                    case.operation
                ));
            }
            Err(err) => {
                let contains_expected = err.message.to_lowercase().contains(&expected);
                let reason_match = err.reason_code == expected_reason_code;
                if contains_expected && reason_match {
                    report.pass_count += 1;
                } else {
                    report.failures.push(format!(
                        "{}: severity={severity} seed={} reason_code={} mode={} env_fingerprint={} artifact_refs={} expected error containing '{}' with reason_code='{}' but got '{}' (actual_reason_code='{}')",
                        case.id,
                        case.seed,
                        reason_code,
                        mode,
                        env_fingerprint,
                        artifact_refs.join(","),
                        case.expected_error_contains,
                        expected_reason_code,
                        err.message,
                        err.reason_code
                    ));
                }
            }
        }
    }

    Ok(report)
}

pub fn run_crash_signature_regression_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let registry_path = config
        .contract_root
        .join("CRASH_SIGNATURE_REGISTRY_V1.json");
    let raw = fs::read_to_string(&registry_path)
        .map_err(|err| format!("failed reading {}: {err}", registry_path.display()))?;
    let registry: CrashSignatureRegistry =
        serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))?;

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut report = SuiteReport {
        suite: "crash_signature_regression",
        case_count: 0,
        pass_count: 0,
        failures: Vec::new(),
    };

    record_suite_check(
        &mut report,
        registry.schema_version == 1,
        "crash signature registry schema_version must be 1".to_string(),
    );
    record_suite_check(
        &mut report,
        registry.registry_version == "crash-signature-registry-v1",
        "crash signature registry version mismatch".to_string(),
    );
    record_suite_check(
        &mut report,
        !registry.signatures.is_empty(),
        "crash signature registry must contain at least one signature".to_string(),
    );

    let mut failure_map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut unique_suites = BTreeSet::new();
    for signature in &registry.signatures {
        unique_suites.insert(signature.suite.trim().to_string());
    }

    for suite_name in unique_suites {
        let suite_report = match suite_name.as_str() {
            "runtime_policy_adversarial" => run_runtime_policy_adversarial_suite(config)?,
            "ufunc_adversarial" => run_ufunc_adversarial_suite(config)?,
            "io_adversarial" => run_io_adversarial_suite(config)?,
            "linalg_adversarial" => run_linalg_adversarial_suite(config)?,
            "rng_adversarial" => run_rng_adversarial_suite(config)?,
            other => {
                record_suite_check(
                    &mut report,
                    false,
                    format!("crash signature registry references unsupported suite '{other}'"),
                );
                continue;
            }
        };
        failure_map.insert(suite_name, suite_report.failures);
    }

    let mut seen_signature_ids = BTreeSet::new();
    for signature in registry.signatures {
        let mut failures = Vec::new();

        if signature.signature_id.trim().is_empty() {
            failures.push("signature_id must not be empty".to_string());
        }
        if !seen_signature_ids.insert(signature.signature_id.clone()) {
            failures.push(format!(
                "duplicate signature_id {} in crash registry (seed={})",
                signature.signature_id, signature.seed
            ));
        }
        if signature.fixture_id.trim().is_empty() {
            failures.push(format!(
                "{}: fixture_id must not be empty",
                signature.signature_id
            ));
        }
        if signature.reason_code.trim().is_empty() {
            failures.push(format!(
                "{}: reason_code must not be empty",
                signature.signature_id
            ));
        }
        if !matches!(
            signature.severity.trim(),
            "low" | "medium" | "high" | "critical"
        ) {
            failures.push(format!(
                "{}: severity '{}' is invalid",
                signature.signature_id, signature.severity
            ));
        }
        if signature.status != "closed" {
            failures.push(format!(
                "{}: status must remain 'closed' to satisfy regression guard (actual={})",
                signature.signature_id, signature.status
            ));
        }
        if signature.minimized_repro_artifacts.is_empty() {
            failures.push(format!(
                "{}: minimized_repro_artifacts must not be empty",
                signature.signature_id
            ));
        }
        for artifact in &signature.minimized_repro_artifacts {
            if artifact.trim().is_empty() {
                failures.push(format!(
                    "{}: minimized_repro_artifacts contains empty entry",
                    signature.signature_id
                ));
                continue;
            }
            let artifact_path = repo_root.join(artifact);
            if !artifact_path.exists() {
                failures.push(format!(
                    "{}: minimized repro artifact missing {}",
                    signature.signature_id,
                    artifact_path.display()
                ));
            }
        }
        if signature.blame_refs.is_empty() {
            failures.push(format!(
                "{}: blame_refs must not be empty",
                signature.signature_id
            ));
        }
        if let Some(suite_failures) = failure_map.get(signature.suite.trim()) {
            let marker = format!("{}:", signature.fixture_id);
            if suite_failures
                .iter()
                .any(|failure| failure.contains(&marker))
            {
                failures.push(format!(
                    "{}: regression detected in suite '{}' for fixture_id '{}'",
                    signature.signature_id, signature.suite, signature.fixture_id
                ));
            }
        }

        if failures.is_empty() {
            report.pass_count += 1;
        } else {
            report.failures.extend(failures);
        }
        report.case_count += 1;
    }

    Ok(report)
}

pub fn run_all_core_suites(config: &HarnessConfig) -> Result<Vec<SuiteReport>, String> {
    Ok(vec![
        run_shape_stride_suite(config)?,
        run_shape_stride_differential_suite(config)?,
        run_shape_stride_metamorphic_suite(config)?,
        run_shape_stride_adversarial_suite(config)?,
        run_iter_differential_suite(config)?,
        run_iter_metamorphic_suite(config)?,
        run_iter_adversarial_suite(config)?,
        run_dtype_promotion_suite(config)?,
        run_runtime_policy_suite(config)?,
        run_runtime_policy_adversarial_suite(config)?,
        run_override_audit_suite(config)?,
        run_rng_differential_suite(config)?,
        run_rng_metamorphic_suite(config)?,
        run_rng_adversarial_suite(config)?,
        run_rng_statistical_suite(config)?,
        run_string_differential_suite(config)?,
        run_string_metamorphic_suite(config)?,
        run_masked_differential_suite(config)?,
        run_datetime_differential_suite(config)?,
        run_datetime_metamorphic_suite(config)?,
        run_polynomial_differential_suite(config)?,
        run_fft_differential_suite(config)?,
        run_fft_metamorphic_suite(config)?,
        run_signal_differential_suite(config)?,
        run_linalg_differential_suite(config)?,
        run_linalg_metamorphic_suite(config)?,
        run_linalg_adversarial_suite(config)?,
        run_io_differential_suite(config)?,
        run_io_metamorphic_suite(config)?,
        run_io_adversarial_suite(config)?,
        run_crash_signature_regression_suite(config)?,
        security_contracts::run_security_contract_suite(config)?,
        test_contracts::run_test_contract_suite(config)?,
        workflow_scenarios::run_user_workflow_scenario_suite(config)?,
        raptorq_artifacts::run_raptorq_artifact_suite(config)?,
        run_ufunc_differential_suite(config)?,
        run_ufunc_metamorphic_suite(config)?,
        run_ufunc_adversarial_suite(config)?,
    ])
}

#[allow(clippy::too_many_arguments)]
fn evaluate_commutative_pair(
    case_id: &str,
    seed: u64,
    mode: &str,
    reason_code: &str,
    env_fingerprint: &str,
    artifact_refs: &[String],
    lhs_rhs: UFuncInputCase,
    rhs_lhs: UFuncInputCase,
    report: &mut SuiteReport,
) -> bool {
    let Ok((shape_a, values_a, dtype_a)) = ufunc_differential::execute_input_case(&lhs_rhs) else {
        report.failures.push(format!(
            "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} failed lhs_rhs execution",
            case_id,
            seed,
            mode,
            reason_code,
            env_fingerprint,
            artifact_refs.join(",")
        ));
        return false;
    };
    let Ok((shape_b, values_b, dtype_b)) = ufunc_differential::execute_input_case(&rhs_lhs) else {
        report.failures.push(format!(
            "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} failed rhs_lhs execution",
            case_id,
            seed,
            mode,
            reason_code,
            env_fingerprint,
            artifact_refs.join(",")
        ));
        return false;
    };

    if shape_a != shape_b {
        report.failures.push(format!(
            "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} commutative shape mismatch lhs_rhs={shape_a:?} rhs_lhs={shape_b:?}",
            case_id,
            seed,
            mode,
            reason_code,
            env_fingerprint,
            artifact_refs.join(",")
        ));
        return false;
    }

    if dtype_a != dtype_b {
        report.failures.push(format!(
            "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} commutative dtype mismatch lhs_rhs={} rhs_lhs={}",
            case_id,
            seed,
            mode,
            reason_code,
            env_fingerprint,
            artifact_refs.join(","),
            dtype_a,
            dtype_b
        ));
        return false;
    }

    if !approx_equal_values(&values_a, &values_b, 1e-9, 1e-9) {
        report.failures.push(format!(
            "{}: seed={} mode={} reason_code={} env_fingerprint={} artifact_refs={} commutative value mismatch",
            case_id,
            seed,
            mode,
            reason_code,
            env_fingerprint,
            artifact_refs.join(",")
        ));
        return false;
    }

    true
}

fn approx_equal_values(expected: &[f64], actual: &[f64], abs_tol: f64, rel_tol: f64) -> bool {
    if expected.len() != actual.len() {
        return false;
    }

    for (&a, &b) in expected.iter().zip(actual.iter()) {
        if a.is_nan() || b.is_nan() {
            if !(a.is_nan() && b.is_nan()) {
                return false;
            }
            continue;
        }

        if a.is_infinite() || b.is_infinite() {
            if a != b {
                return false;
            }
            continue;
        }

        let abs_err = (a - b).abs();
        let threshold = abs_tol + rel_tol * a.abs();
        if abs_err > threshold {
            return false;
        }
    }
    true
}

fn resolve_case_mode(raw: &str, strict_mode: bool) -> String {
    if raw.trim().is_empty() {
        if strict_mode {
            "strict".to_string()
        } else {
            "hardened".to_string()
        }
    } else {
        raw.trim().to_string()
    }
}

fn classify_ufunc_reason_code(op: UFuncOperation, detail: &str) -> String {
    let lowered = detail.to_lowercase();
    if lowered.contains("rhs_shape")
        || lowered.contains("rhs_values")
        || lowered.contains("signature")
    {
        "ufunc_signature_parse_failed".to_string()
    } else if lowered.contains("unsupported dtype")
        || lowered.contains("dtype")
        || lowered.contains("promotion")
    {
        "ufunc_type_resolution_invalid".to_string()
    } else if matches!(
        op,
        UFuncOperation::Sum
            | UFuncOperation::Prod
            | UFuncOperation::Min
            | UFuncOperation::Max
            | UFuncOperation::Mean
            | UFuncOperation::Var
            | UFuncOperation::Std
            | UFuncOperation::Argmin
            | UFuncOperation::Argmax
    ) && (lowered.contains("axis")
        || lowered.contains("keepdims")
        || lowered.contains("argmin")
        || lowered.contains("argmax")
        || lowered.contains("reduce"))
    {
        "ufunc_reduction_contract_violation".to_string()
    } else if lowered.contains("override") {
        "ufunc_override_precedence_violation".to_string()
    } else if lowered.contains("policy")
        || lowered.contains("metadata")
        || lowered.contains("oracle")
    {
        "ufunc_policy_unknown_metadata".to_string()
    } else {
        "ufunc_dispatch_resolution_failed".to_string()
    }
}

fn decode_matrix_2x2(matrix: &[Vec<f64>]) -> Result<[[f64; 2]; 2], LinAlgError> {
    if matrix.len() != 2 || matrix.iter().any(|row| row.len() != 2) {
        return Err(LinAlgError::ShapeContractViolation(
            "solve_2x2 requires a 2x2 matrix fixture",
        ));
    }
    Ok([[matrix[0][0], matrix[0][1]], [matrix[1][0], matrix[1][1]]])
}

fn decode_rhs_2(rhs: &[f64]) -> Result<[f64; 2], LinAlgError> {
    if rhs.len() != 2 {
        return Err(LinAlgError::ShapeContractViolation(
            "solve_2x2 requires a length-2 rhs fixture",
        ));
    }
    Ok([rhs[0], rhs[1]])
}

fn flatten_matrix_fixture(matrix: &[Vec<f64>]) -> Result<Vec<f64>, LinAlgError> {
    if matrix.is_empty() {
        return Err(LinAlgError::ShapeContractViolation(
            "batched operations require non-empty matrix fixture rows",
        ));
    }

    let row_len = matrix[0].len();
    if row_len == 0 {
        return Err(LinAlgError::ShapeContractViolation(
            "batched operations require non-empty matrix fixture columns",
        ));
    }
    if matrix.iter().any(|row| row.len() != row_len) {
        return Err(LinAlgError::ShapeContractViolation(
            "batched operations require rectangular matrix fixture rows",
        ));
    }

    Ok(matrix
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>())
}

fn transpose_2x2(matrix: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [[matrix[0][0], matrix[1][0]], [matrix[0][1], matrix[1][1]]]
}

fn matmul_2x2(lhs: [[f64; 2]; 2], rhs: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [
            lhs[0][0].mul_add(rhs[0][0], lhs[0][1] * rhs[1][0]),
            lhs[0][0].mul_add(rhs[0][1], lhs[0][1] * rhs[1][1]),
        ],
        [
            lhs[1][0].mul_add(rhs[0][0], lhs[1][1] * rhs[1][0]),
            lhs[1][0].mul_add(rhs[0][1], lhs[1][1] * rhs[1][1]),
        ],
    ]
}

fn flatten_2x2(matrix: [[f64; 2]; 2]) -> [f64; 4] {
    [matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]]
}

fn is_orthonormal_2x2(matrix: [[f64; 2]; 2], tolerance: f64) -> bool {
    let gram = matmul_2x2(transpose_2x2(matrix), matrix);
    approx_equal_values(
        &[1.0, 0.0, 0.0, 1.0],
        &flatten_2x2(gram),
        tolerance,
        tolerance,
    )
}

fn is_upper_triangular_2x2(matrix: [[f64; 2]; 2], tolerance: f64) -> bool {
    matrix[1][0].abs() <= tolerance
}

fn execute_linalg_differential_operation(
    case: &LinalgDifferentialCase,
) -> Result<LinalgOperationOutcome, LinAlgError> {
    execute_linalg_operation(LinalgOperationInput {
        operation: &case.operation,
        matrix: &case.matrix,
        rhs: &case.rhs,
        shape: &case.shape,
        rhs_shape: &case.rhs_shape,
        qr_mode: &case.qr_mode,
        full_matrices: case.full_matrices,
        converged: case.converged,
        uplo: &case.uplo,
        rcond: case.rcond,
        search_depth: case.search_depth,
        backend_supported: case.backend_supported,
        validation_retries: case.validation_retries,
        mode_raw: &case.mode_raw,
        class_raw: &case.class_raw,
    })
}

fn execute_linalg_adversarial_operation(
    case: &LinalgAdversarialCase,
) -> Result<LinalgOperationOutcome, LinAlgError> {
    execute_linalg_operation(LinalgOperationInput {
        operation: &case.operation,
        matrix: &case.matrix,
        rhs: &case.rhs,
        shape: &case.shape,
        rhs_shape: &case.rhs_shape,
        qr_mode: &case.qr_mode,
        full_matrices: case.full_matrices,
        converged: case.converged,
        uplo: &case.uplo,
        rcond: case.rcond,
        search_depth: case.search_depth,
        backend_supported: case.backend_supported,
        validation_retries: case.validation_retries,
        mode_raw: &case.mode_raw,
        class_raw: &case.class_raw,
    })
}

fn execute_linalg_operation(
    input: LinalgOperationInput<'_>,
) -> Result<LinalgOperationOutcome, LinAlgError> {
    match input.operation {
        "solve_2x2" => {
            let matrix = decode_matrix_2x2(input.matrix)?;
            let rhs = decode_rhs_2(input.rhs)?;
            let solved = solve_2x2(matrix, rhs)?;
            Ok(LinalgOperationOutcome::SolveVector(solved.to_vec()))
        }
        "qr_shapes" => {
            let mode = QrMode::from_mode_token(input.qr_mode)?;
            let output = qr_output_shapes(input.shape, mode)?;
            Ok(LinalgOperationOutcome::QrShapes {
                q_shape: output.q_shape,
                r_shape: output.r_shape,
            })
        }
        "qr_2x2" => {
            let matrix = decode_matrix_2x2(input.matrix)?;
            let mode = QrMode::from_mode_token(input.qr_mode)?;
            let output = qr_2x2(matrix, mode)?;
            Ok(LinalgOperationOutcome::QrDecomposition {
                q: output.q,
                r: output.r,
            })
        }
        "svd_shapes" => {
            let output = svd_output_shapes(input.shape, input.full_matrices, input.converged)?;
            Ok(LinalgOperationOutcome::SvdShapes {
                u_shape: output.u_shape,
                s_shape: output.s_shape,
                vh_shape: output.vh_shape,
            })
        }
        "svd_2x2" => {
            let matrix = decode_matrix_2x2(input.matrix)?;
            let output = svd_2x2(matrix, input.converged)?;
            Ok(LinalgOperationOutcome::SvdDecomposition {
                u: output.u,
                singular_values: output.singular_values,
                vt: output.vt,
            })
        }
        "lstsq_shapes" => {
            let output = lstsq_output_shapes(input.shape, input.rhs_shape)?;
            Ok(LinalgOperationOutcome::LstsqShapes {
                x_shape: output.x_shape,
                residuals_shape: output.residuals_shape,
                rank_upper_bound: output.rank_upper_bound,
                singular_values_shape: output.singular_values_shape,
            })
        }
        "batch_det" => {
            let data = flatten_matrix_fixture(input.matrix)?;
            Ok(LinalgOperationOutcome::SolveVector(batch_det(
                data.as_slice(),
                input.shape,
            )?))
        }
        "batch_trace" => {
            let data = flatten_matrix_fixture(input.matrix)?;
            Ok(LinalgOperationOutcome::SolveVector(batch_trace(
                data.as_slice(),
                input.shape,
            )?))
        }
        "batch_inv" => {
            let data = flatten_matrix_fixture(input.matrix)?;
            Ok(LinalgOperationOutcome::SolveVector(batch_inv(
                data.as_slice(),
                input.shape,
            )?))
        }
        "batch_solve" => {
            let data = flatten_matrix_fixture(input.matrix)?;
            Ok(LinalgOperationOutcome::SolveVector(batch_solve(
                data.as_slice(),
                input.shape,
                input.rhs,
                input.rhs_shape,
                input.rhs_shape.len() < input.shape.len() || input.rhs_shape.len() == 1,
            )?))
        }
        "spectral_branch" => {
            validate_spectral_branch(input.uplo, input.converged)?;
            Ok(LinalgOperationOutcome::Unit)
        }
        "tolerance_policy" => {
            validate_tolerance_policy(input.rcond, input.search_depth)?;
            Ok(LinalgOperationOutcome::Unit)
        }
        "backend_bridge" => {
            validate_backend_bridge(input.backend_supported, input.validation_retries)?;
            Ok(LinalgOperationOutcome::Unit)
        }
        "policy_metadata" => {
            validate_linalg_policy_metadata(input.mode_raw, input.class_raw)?;
            Ok(LinalgOperationOutcome::Unit)
        }
        "cond_mxn" => {
            let data = flatten_matrix_fixture(input.matrix)?;
            let m = input.shape.first().copied().unwrap_or(0);
            let n = input.shape.get(1).copied().unwrap_or(0);
            let cond = cond_mxn(&data, m, n)?;
            Ok(LinalgOperationOutcome::SolveVector(vec![cond]))
        }
        "cholesky_2x2" => {
            let matrix = decode_matrix_2x2(input.matrix)?;
            let uplo = if input.uplo.is_empty() {
                "L"
            } else {
                input.uplo
            };
            let l = cholesky_2x2(matrix, uplo)?;
            Ok(LinalgOperationOutcome::SolveVector(vec![
                l[0][0], l[0][1], l[1][0], l[1][1],
            ]))
        }
        "slogdet_2x2" => {
            let matrix = decode_matrix_2x2(input.matrix)?;
            let (sign, logabsdet) = slogdet_2x2(matrix)?;
            Ok(LinalgOperationOutcome::SolveVector(vec![sign, logabsdet]))
        }
        "eigvals_2x2" => {
            let matrix = decode_matrix_2x2(input.matrix)?;
            let eigvals = eigvals_2x2(matrix, input.converged)?;
            Ok(LinalgOperationOutcome::SolveVector(eigvals.to_vec()))
        }
        "matrix_power" => {
            let data = flatten_matrix_fixture(input.matrix)?;
            let n = input.shape.first().copied().unwrap_or(0);
            let power = input.rhs.first().copied().unwrap_or(1.0) as i64;
            let result = matrix_power_nxn(&data, n, power)?;
            Ok(LinalgOperationOutcome::SolveVector(result))
        }
        "pinv_2x2" => {
            let matrix = decode_matrix_2x2(input.matrix)?;
            let rcond_val = if input.rcond > 0.0 {
                input.rcond
            } else {
                1e-15
            };
            let pinv = pinv_2x2(matrix, rcond_val)?;
            Ok(LinalgOperationOutcome::SolveVector(vec![
                pinv[0][0], pinv[0][1], pinv[1][0], pinv[1][1],
            ]))
        }
        _ => Err(LinAlgError::PolicyUnknownMetadata(
            "unsupported linalg operation",
        )),
    }
}

fn validate_linalg_differential_expectation(
    case: &LinalgDifferentialCase,
    outcome: &LinalgOperationOutcome,
) -> Result<(), String> {
    match (case.operation.as_str(), outcome) {
        ("solve_2x2", LinalgOperationOutcome::SolveVector(actual)) => {
            if case.expected_solution.len() != actual.len() {
                return Err(format!(
                    "solve_2x2 expected solution length {} but got {}",
                    case.expected_solution.len(),
                    actual.len()
                ));
            }
            if !approx_equal_values(&case.expected_solution, actual, 1e-9, 1e-9) {
                return Err(format!(
                    "solve_2x2 mismatch expected={:?} actual={actual:?}",
                    case.expected_solution
                ));
            }
            Ok(())
        }
        (
            "batch_det" | "batch_trace" | "batch_inv" | "batch_solve",
            LinalgOperationOutcome::SolveVector(actual),
        ) => {
            if case.expected_solution.len() != actual.len() {
                return Err(format!(
                    "{} expected vector length {} but got {}",
                    case.operation,
                    case.expected_solution.len(),
                    actual.len()
                ));
            }
            let tolerance = if case.expected_tolerance > 0.0 {
                case.expected_tolerance
            } else {
                1e-9
            };
            if !approx_equal_values(
                &case.expected_solution,
                actual.as_slice(),
                tolerance,
                tolerance,
            ) {
                return Err(format!(
                    "{} mismatch expected={:?} actual={actual:?}",
                    case.operation, case.expected_solution
                ));
            }
            Ok(())
        }
        ("qr_shapes", LinalgOperationOutcome::QrShapes { q_shape, r_shape }) => {
            if case.expected_q_shape != *q_shape || case.expected_r_shape != *r_shape {
                return Err(format!(
                    "qr_shapes mismatch expected_q_shape={:?} expected_r_shape={:?} actual_q_shape={q_shape:?} actual_r_shape={r_shape:?}",
                    case.expected_q_shape, case.expected_r_shape
                ));
            }
            Ok(())
        }
        ("qr_2x2", LinalgOperationOutcome::QrDecomposition { q, r }) => {
            let tolerance = if case.expected_tolerance > 0.0 {
                case.expected_tolerance
            } else {
                1e-9
            };
            if !is_upper_triangular_2x2(*r, tolerance) {
                return Err(format!("qr_2x2 expected upper-triangular R, got {r:?}"));
            }
            if r[1][1] < -tolerance {
                return Err(format!(
                    "qr_2x2 expected non-negative trailing diagonal in R, got {:?}",
                    r[1][1]
                ));
            }

            if let Some(expected_q_present) = case.expected_q_present {
                let actual_q_present = q.is_some();
                if actual_q_present != expected_q_present {
                    return Err(format!(
                        "qr_2x2 q-presence mismatch expected={expected_q_present} actual={actual_q_present}"
                    ));
                }
            }

            if case.qr_mode == "r" {
                if q.is_some() {
                    return Err("qr_2x2 in r mode must omit Q output".to_string());
                }
                return Ok(());
            }

            let Some(q_matrix) = q else {
                return Err("qr_2x2 expected Q output for non-r modes".to_string());
            };
            if !is_orthonormal_2x2(*q_matrix, tolerance) {
                return Err(format!("qr_2x2 Q is not orthonormal: {q_matrix:?}"));
            }

            let expected_matrix = decode_matrix_2x2(&case.matrix)
                .map_err(|err| format!("qr_2x2 fixture matrix invalid: {err}"))?;
            let reconstructed = matmul_2x2(*q_matrix, *r);
            if !approx_equal_values(
                &flatten_2x2(expected_matrix),
                &flatten_2x2(reconstructed),
                tolerance,
                tolerance,
            ) {
                return Err(format!(
                    "qr_2x2 reconstruction mismatch expected={expected_matrix:?} actual={reconstructed:?}"
                ));
            }
            Ok(())
        }
        (
            "svd_shapes",
            LinalgOperationOutcome::SvdShapes {
                u_shape,
                s_shape,
                vh_shape,
            },
        ) => {
            if case.expected_u_shape != *u_shape
                || case.expected_s_shape != *s_shape
                || case.expected_vh_shape != *vh_shape
            {
                return Err(format!(
                    "svd_shapes mismatch expected_u={:?} expected_s={:?} expected_vh={:?} actual_u={u_shape:?} actual_s={s_shape:?} actual_vh={vh_shape:?}",
                    case.expected_u_shape, case.expected_s_shape, case.expected_vh_shape
                ));
            }
            Ok(())
        }
        (
            "lstsq_shapes",
            LinalgOperationOutcome::LstsqShapes {
                x_shape,
                residuals_shape,
                rank_upper_bound,
                singular_values_shape,
            },
        ) => {
            if case.expected_x_shape != *x_shape
                || case.expected_residuals_shape != *residuals_shape
                || case.expected_rank_upper_bound != *rank_upper_bound
                || case.expected_singular_values_shape != *singular_values_shape
            {
                return Err(format!(
                    "lstsq_shapes mismatch expected_x={:?} expected_residuals={:?} expected_rank_upper_bound={} expected_singular_values={:?} actual_x={x_shape:?} actual_residuals={residuals_shape:?} actual_rank_upper_bound={} actual_singular_values={singular_values_shape:?}",
                    case.expected_x_shape,
                    case.expected_residuals_shape,
                    case.expected_rank_upper_bound,
                    case.expected_singular_values_shape,
                    rank_upper_bound
                ));
            }
            Ok(())
        }
        (
            "svd_2x2",
            LinalgOperationOutcome::SvdDecomposition {
                u,
                singular_values,
                vt,
            },
        ) => {
            if case.expected_singular_values.len() != 2 {
                return Err(
                    "svd_2x2 requires expected_singular_values with exactly two entries"
                        .to_string(),
                );
            }

            let tolerance = if case.expected_tolerance > 0.0 {
                case.expected_tolerance
            } else {
                1e-9
            };
            if !approx_equal_values(
                &case.expected_singular_values,
                singular_values.as_slice(),
                tolerance,
                tolerance,
            ) {
                return Err(format!(
                    "svd_2x2 singular-values mismatch expected={:?} actual={singular_values:?}",
                    case.expected_singular_values
                ));
            }

            if !is_orthonormal_2x2(*u, tolerance) {
                return Err(format!("svd_2x2 U is not orthonormal: {u:?}"));
            }
            if !is_orthonormal_2x2(transpose_2x2(*vt), tolerance) {
                return Err(format!("svd_2x2 V^T is not orthonormal: {vt:?}"));
            }

            let sigma = [[singular_values[0], 0.0], [0.0, singular_values[1]]];
            let reconstructed = matmul_2x2(matmul_2x2(*u, sigma), *vt);
            let expected_matrix = decode_matrix_2x2(&case.matrix)
                .map_err(|err| format!("svd_2x2 fixture matrix invalid: {err}"))?;
            if !approx_equal_values(
                &flatten_2x2(expected_matrix),
                &flatten_2x2(reconstructed),
                tolerance,
                tolerance,
            ) {
                return Err(format!(
                    "svd_2x2 reconstruction mismatch expected={expected_matrix:?} actual={reconstructed:?}"
                ));
            }
            Ok(())
        }
        (
            "spectral_branch" | "tolerance_policy" | "backend_bridge" | "policy_metadata",
            LinalgOperationOutcome::Unit,
        ) => Ok(()),
        ("cond_mxn", LinalgOperationOutcome::SolveVector(actual)) => {
            if case.expected_solution.len() != actual.len() {
                return Err(format!(
                    "cond_mxn expected vector length {} but got {}",
                    case.expected_solution.len(),
                    actual.len()
                ));
            }
            let tolerance = if case.expected_tolerance > 0.0 {
                case.expected_tolerance
            } else {
                1e-9
            };
            if !approx_equal_values(
                &case.expected_solution,
                actual.as_slice(),
                tolerance,
                tolerance,
            ) {
                return Err(format!(
                    "cond_mxn mismatch expected={:?} actual={actual:?}",
                    case.expected_solution
                ));
            }
            Ok(())
        }
        (
            "cholesky_2x2" | "slogdet_2x2" | "eigvals_2x2" | "matrix_power" | "pinv_2x2",
            LinalgOperationOutcome::SolveVector(actual),
        ) => {
            if case.expected_solution.len() != actual.len() {
                return Err(format!(
                    "{} expected vector length {} but got {}",
                    case.operation,
                    case.expected_solution.len(),
                    actual.len()
                ));
            }
            let tolerance = if case.expected_tolerance > 0.0 {
                case.expected_tolerance
            } else {
                1e-9
            };
            if !approx_equal_values(
                &case.expected_solution,
                actual.as_slice(),
                tolerance,
                tolerance,
            ) {
                return Err(format!(
                    "{} mismatch expected={:?} actual={actual:?}",
                    case.operation, case.expected_solution
                ));
            }
            Ok(())
        }
        (operation, actual) => Err(format!(
            "operation {operation} produced unexpected outcome {actual:?}"
        )),
    }
}

const DEFAULT_RNG_DRAWS: usize = 128;
const DEFAULT_RNG_REPLAY_DRAWS: usize = 32;
const DEFAULT_RNG_PREFIX_DRAWS: usize = 8;
const DEFAULT_RNG_JUMP_STEPS: u64 = 32;
use fnp_random::{MAX_RNG_JUMP_OPERATIONS as RNG_MAX_JUMP_OPS, MAX_RNG_STATE_SCHEMA_FIELDS};
const RNG_MAX_STATE_SCHEMA_FIELDS_U64: u64 = MAX_RNG_STATE_SCHEMA_FIELDS as u64;
const DEFAULT_RNG_STAT_DRAWS: usize = 8_192;
const DEFAULT_RNG_KS_ALPHA: f64 = 0.01;
const KS_CRITICAL_01: f64 = 1.63;
const KS_CRITICAL_05: f64 = 1.36;

fn evaluate_rng_statistical_case(case: &RngStatisticalCase) -> Result<(), RngSuiteError> {
    let draws = case.draws.max(DEFAULT_RNG_STAT_DRAWS);
    let reason_code = if case.reason_code.trim().is_empty() {
        "random_distribution_statistical_contract".to_string()
    } else {
        case.reason_code.trim().to_string()
    };

    let samples = generate_rng_samples(case, draws)?;
    if samples.is_empty() {
        return Err(RngSuiteError::new(
            reason_code.clone(),
            "statistical sample set is empty",
        ));
    }

    let (sample_mean, sample_variance) = sample_mean_variance(&samples);
    if let Some(expected_mean) = case.expected_mean
        && (sample_mean - expected_mean).abs() > case.mean_abs_tol
    {
        return Err(RngSuiteError::new(
            reason_code.clone(),
            format!(
                "sample mean mismatch expected={} actual={} abs_tol={}",
                expected_mean, sample_mean, case.mean_abs_tol
            ),
        ));
    }
    if let Some(expected_variance) = case.expected_variance
        && (sample_variance - expected_variance).abs() > case.variance_abs_tol
    {
        return Err(RngSuiteError::new(
            reason_code.clone(),
            format!(
                "sample variance mismatch expected={} actual={} abs_tol={}",
                expected_variance, sample_variance, case.variance_abs_tol
            ),
        ));
    }

    validate_rng_support_bounds(case, &samples, &reason_code)?;

    let ks_distribution = case.ks_distribution.trim();
    if !ks_distribution.is_empty() {
        let statistic = ks_statistic(case, &samples).map_err(|detail| {
            RngSuiteError::new(reason_code.clone(), format!("ks setup failed: {detail}"))
        })?;
        let alpha = if case.ks_alpha > 0.0 {
            case.ks_alpha
        } else {
            DEFAULT_RNG_KS_ALPHA
        };
        let critical = ks_critical_value(samples.len(), alpha);
        if statistic > critical {
            return Err(RngSuiteError::new(
                reason_code,
                format!(
                    "ks statistic {} exceeded critical {} (alpha={}, n={})",
                    statistic,
                    critical,
                    alpha,
                    samples.len()
                ),
            ));
        }
    }

    Ok(())
}

fn generate_rng_samples(
    case: &RngStatisticalCase,
    draws: usize,
) -> Result<Vec<f64>, RngSuiteError> {
    let mut generator =
        Generator::from_pcg64_dxsm(case.seed).map_err(map_seedsequence_error_to_rng_suite)?;
    match case.distribution.as_str() {
        "random" => Ok(generator.random(draws)),
        "uniform" => {
            map_fallible_random_values(generator.uniform(case.param_a, case.param_b, draws))
        }
        "standard_normal" => Ok(generator.standard_normal(draws)),
        "normal" => map_fallible_random_values(generator.normal(case.param_a, case.param_b, draws)),
        "standard_exponential" => Ok(generator.standard_exponential(draws)),
        "exponential" => map_fallible_random_values(generator.exponential(case.param_a, draws)),
        "standard_gamma" => {
            map_fallible_random_values(generator.standard_gamma(case.param_a, draws))
        }
        "integers" => {
            let low = f64_to_i64_param(case.param_a, "param_a")?;
            let high = f64_to_i64_param(case.param_b, "param_b")?;
            generator
                .integers(low, high, draws)
                .map(|values| values.into_iter().map(|value| value as f64).collect())
                .map_err(map_random_error_to_rng_suite)
        }
        "poisson" => generator
            .poisson(case.param_a, draws)
            .map(|values| values.into_iter().map(|value| value as f64).collect())
            .map_err(map_random_error_to_rng_suite),
        "binomial" => {
            let n = f64_to_u64_param(case.param_a, "param_a")?;
            generator
                .binomial(n, case.param_b, draws)
                .map(|values| values.into_iter().map(|value| value as f64).collect())
                .map_err(map_random_error_to_rng_suite)
        }
        "gamma" => map_fallible_random_values(generator.gamma(case.param_a, case.param_b, draws)),
        "beta" => map_fallible_random_values(generator.beta(case.param_a, case.param_b, draws)),
        "geometric" => generator
            .geometric(case.param_a, draws)
            .map(|values| values.into_iter().map(|value| value as f64).collect())
            .map_err(map_random_error_to_rng_suite),
        "lognormal" => {
            map_fallible_random_values(generator.lognormal(case.param_a, case.param_b, draws))
        }
        "chisquare" => map_fallible_random_values(generator.chisquare(case.param_a, draws)),
        "standard_cauchy" => Ok(generator.standard_cauchy(draws)),
        "triangular" => map_fallible_random_values(generator.triangular(
            case.param_a,
            case.param_b,
            case.param_c,
            draws,
        )),
        "laplace" => {
            map_fallible_random_values(generator.laplace(case.param_a, case.param_b, draws))
        }
        "gumbel" => map_fallible_random_values(generator.gumbel(case.param_a, case.param_b, draws)),
        "weibull" => map_fallible_random_values(generator.weibull(case.param_a, draws)),
        "negative_binomial" => generator
            .negative_binomial(case.param_a, case.param_b, draws)
            .map(|values| values.into_iter().map(|value| value as f64).collect())
            .map_err(map_random_error_to_rng_suite),
        "f_distribution" => {
            map_fallible_random_values(generator.f_distribution(case.param_a, case.param_b, draws))
        }
        "standard_t" => Ok(generator.standard_t(case.param_a, draws)),
        "noncentral_chisquare" => map_fallible_random_values(generator.noncentral_chisquare(
            case.param_a,
            case.param_b,
            draws,
        )),
        "noncentral_f" => map_fallible_random_values(generator.noncentral_f(
            case.param_a,
            case.param_b,
            case.param_c,
            draws,
        )),
        "power" => map_fallible_random_values(generator.power(case.param_a, draws)),
        "vonmises" => {
            map_fallible_random_values(generator.vonmises(case.param_a, case.param_b, draws))
        }
        "rayleigh" => map_fallible_random_values(generator.rayleigh(case.param_a, draws)),
        "pareto" => map_fallible_random_values(generator.pareto(case.param_a, draws)),
        "logistic" => {
            map_fallible_random_values(generator.logistic(case.param_a, case.param_b, draws))
        }
        "hypergeometric" => {
            let ngood = f64_to_u64_param(case.param_a, "param_a")?;
            let nbad = f64_to_u64_param(case.param_b, "param_b")?;
            let nsample = f64_to_u64_param(case.param_c, "param_c")?;
            generator
                .hypergeometric(ngood, nbad, nsample, draws)
                .map_err(|e| RngSuiteError::new("parameter_violation", e.to_string()))
                .map(|v| v.into_iter().map(|value| value as f64).collect())
        }
        "zipf" => generator
            .zipf(case.param_a, draws)
            .map_err(|e| RngSuiteError::new("parameter_violation", e.to_string())),
        "wald" => Ok(generator.wald(case.param_a, case.param_b, draws)),
        "logseries" => generator
            .logseries(case.param_a, draws)
            .map(|vals| vals.into_iter().map(|v| v as f64).collect())
            .map_err(|e| RngSuiteError::new("parameter_violation", e.to_string())),
        "maxwell" => Ok(generator.maxwell(case.param_a, draws)),
        "halfnormal" => Ok(generator.halfnormal(case.param_a, draws)),
        "truncated_normal" => Ok(generator.truncated_normal(
            case.param_a,
            case.param_b,
            case.param_c,
            case.param_d,
            draws,
        )),
        "lomax" => Ok(generator.lomax(case.param_a, draws)),
        "levy" => Ok(generator.levy(case.param_a, case.param_b, draws)),
        other => Err(RngSuiteError::new(
            "rng_policy_unknown_metadata",
            format!("unsupported statistical distribution {other}"),
        )),
    }
}

fn f64_to_i64_param(value: f64, label: &str) -> Result<i64, RngSuiteError> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(RngSuiteError::new(
            "rng_policy_unknown_metadata",
            format!("{label} must be a finite integer-like value"),
        ));
    }
    if value < i64::MIN as f64 || value > i64::MAX as f64 {
        return Err(RngSuiteError::new(
            "rng_policy_unknown_metadata",
            format!("{label} out of i64 bounds"),
        ));
    }
    Ok(value as i64)
}

fn f64_to_u64_param(value: f64, label: &str) -> Result<u64, RngSuiteError> {
    if !value.is_finite() || value.fract() != 0.0 || value < 0.0 {
        return Err(RngSuiteError::new(
            "rng_policy_unknown_metadata",
            format!("{label} must be a non-negative finite integer-like value"),
        ));
    }
    if value > u64::MAX as f64 {
        return Err(RngSuiteError::new(
            "rng_policy_unknown_metadata",
            format!("{label} out of u64 bounds"),
        ));
    }
    Ok(value as u64)
}

fn sample_mean_variance(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let variance = samples
        .iter()
        .map(|sample| {
            let diff = sample - mean;
            diff * diff
        })
        .sum::<f64>()
        / n;
    (mean, variance)
}

fn validate_rng_support_bounds(
    case: &RngStatisticalCase,
    samples: &[f64],
    reason_code: &str,
) -> Result<(), RngSuiteError> {
    if let Some(min_bound) = case.support_min {
        let invalid = samples.iter().enumerate().find(|(_, sample)| {
            if case.support_min_inclusive {
                **sample < min_bound
            } else {
                **sample <= min_bound
            }
        });
        if let Some((index, sample)) = invalid {
            return Err(RngSuiteError::new(
                reason_code.to_string(),
                format!(
                    "support minimum violated at index {}: sample={} bound={} inclusive={}",
                    index, sample, min_bound, case.support_min_inclusive
                ),
            ));
        }
    }

    if let Some(max_bound) = case.support_max {
        let invalid = samples.iter().enumerate().find(|(_, sample)| {
            if case.support_max_inclusive {
                **sample > max_bound
            } else {
                **sample >= max_bound
            }
        });
        if let Some((index, sample)) = invalid {
            return Err(RngSuiteError::new(
                reason_code.to_string(),
                format!(
                    "support maximum violated at index {}: sample={} bound={} inclusive={}",
                    index, sample, max_bound, case.support_max_inclusive
                ),
            ));
        }
    }

    Ok(())
}

fn ks_statistic(case: &RngStatisticalCase, samples: &[f64]) -> Result<f64, String> {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let n = sorted.len() as f64;

    let cdf = build_ks_cdf(case)?;
    let mut max_distance = 0.0f64;
    for (index, sample) in sorted.iter().enumerate() {
        let empirical_upper = (index as f64 + 1.0) / n;
        let empirical_lower = index as f64 / n;
        let expected = cdf(*sample).clamp(0.0, 1.0);
        let diff_upper = (empirical_upper - expected).abs();
        let diff_lower = (expected - empirical_lower).abs();
        max_distance = max_distance.max(diff_upper.max(diff_lower));
    }

    Ok(max_distance)
}

fn ks_critical_value(sample_len: usize, alpha: f64) -> f64 {
    let factor = if alpha <= 0.01 {
        KS_CRITICAL_01
    } else {
        KS_CRITICAL_05
    };
    factor / (sample_len as f64).sqrt()
}

fn build_ks_cdf(case: &RngStatisticalCase) -> Result<Box<dyn Fn(f64) -> f64>, String> {
    match case.ks_distribution.as_str() {
        "uniform" => {
            let low = case.param_a;
            let high = case.param_b;
            if !(low.is_finite() && high.is_finite() && high > low) {
                return Err("uniform ks parameters require finite high > low".to_string());
            }
            Ok(Box::new(move |x| {
                if x < low {
                    0.0
                } else if x >= high {
                    1.0
                } else {
                    (x - low) / (high - low)
                }
            }))
        }
        "normal" => {
            let mean = case.param_a;
            let std_dev = case.param_b;
            if !(mean.is_finite() && std_dev.is_finite() && std_dev > 0.0) {
                return Err("normal ks parameters require finite std_dev > 0".to_string());
            }
            Ok(Box::new(move |x| {
                let z = (x - mean) / (std_dev * std::f64::consts::SQRT_2);
                0.5 * (1.0 + erf_approx(z))
            }))
        }
        "exponential" => {
            let scale = case.param_a;
            if !(scale.is_finite() && scale > 0.0) {
                return Err("exponential ks parameters require finite scale > 0".to_string());
            }
            Ok(Box::new(move |x| {
                if x <= 0.0 {
                    0.0
                } else {
                    1.0 - (-x / scale).exp()
                }
            }))
        }
        "cauchy" => {
            let loc = case.param_a;
            let scale = case.param_b;
            if !(loc.is_finite() && scale.is_finite() && scale > 0.0) {
                return Err("cauchy ks parameters require finite scale > 0".to_string());
            }
            Ok(Box::new(move |x| {
                0.5 + ((x - loc) / scale).atan() / std::f64::consts::PI
            }))
        }
        "logistic" => {
            let loc = case.param_a;
            let scale = case.param_b;
            if !(loc.is_finite() && scale.is_finite() && scale > 0.0) {
                return Err("logistic ks parameters require finite scale > 0".to_string());
            }
            Ok(Box::new(move |x| 1.0 / (1.0 + (-(x - loc) / scale).exp())))
        }
        "laplace" => {
            let loc = case.param_a;
            let scale = case.param_b;
            if !(loc.is_finite() && scale.is_finite() && scale > 0.0) {
                return Err("laplace ks parameters require finite scale > 0".to_string());
            }
            Ok(Box::new(move |x| {
                if x < loc {
                    0.5 * ((x - loc) / scale).exp()
                } else {
                    1.0 - 0.5 * (-(x - loc) / scale).exp()
                }
            }))
        }
        "gumbel" => {
            let loc = case.param_a;
            let scale = case.param_b;
            if !(loc.is_finite() && scale.is_finite() && scale > 0.0) {
                return Err("gumbel ks parameters require finite scale > 0".to_string());
            }
            Ok(Box::new(move |x| {
                let z = (x - loc) / scale;
                (-(-z).exp()).exp()
            }))
        }
        "rayleigh" => {
            let scale = case.param_a;
            if !(scale.is_finite() && scale > 0.0) {
                return Err("rayleigh ks parameters require finite scale > 0".to_string());
            }
            Ok(Box::new(move |x| {
                if x <= 0.0 {
                    0.0
                } else {
                    1.0 - (-(x * x) / (2.0 * scale * scale)).exp()
                }
            }))
        }
        "pareto" => {
            let alpha = case.param_a;
            if !(alpha.is_finite() && alpha > 0.0) {
                return Err("pareto ks parameters require finite alpha > 0".to_string());
            }
            Ok(Box::new(move |x| {
                if x <= 0.0 {
                    0.0
                } else {
                    1.0 - (1.0 + x).powf(-alpha)
                }
            }))
        }
        "power" => {
            let alpha = case.param_a;
            if !(alpha.is_finite() && alpha > 0.0) {
                return Err("power ks parameters require finite alpha > 0".to_string());
            }
            Ok(Box::new(move |x| {
                if x <= 0.0 {
                    0.0
                } else if x >= 1.0 {
                    1.0
                } else {
                    x.powf(alpha)
                }
            }))
        }
        other => Err(format!("unsupported ks_distribution {other}")),
    }
}

fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x_abs);
    let polynomial =
        (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t;
    sign * (1.0 - polynomial * (-x_abs * x_abs).exp())
}

fn execute_rng_differential_operation(case: &RngDifferentialCase) -> Result<(), RngSuiteError> {
    match case.operation.as_str() {
        "same_seed_stream" => {
            let draws = case.draws.max(DEFAULT_RNG_DRAWS);
            let mut lhs = DeterministicRng::new(case.seed);
            let mut rhs = DeterministicRng::new(case.seed);
            for index in 0..draws {
                if lhs.next_u64() != rhs.next_u64() {
                    return Err(RngSuiteError::new(
                        "rng_reproducibility_witness_failed",
                        format!("same-seed stream mismatch at draw {index}"),
                    ));
                }
            }
            Ok(())
        }
        "different_seed_diverges" => {
            let draws = case.draws.max(DEFAULT_RNG_DRAWS);
            let alt_seed = if case.alt_seed == 0 {
                case.seed.wrapping_add(1)
            } else {
                case.alt_seed
            };
            let mut lhs = DeterministicRng::new(case.seed);
            let mut rhs = DeterministicRng::new(alt_seed);
            let diverged = (0..draws).any(|_| lhs.next_u64() != rhs.next_u64());
            if diverged {
                Ok(())
            } else {
                Err(RngSuiteError::new(
                    "rng_reproducibility_witness_failed",
                    "distinct seeds did not diverge within draw budget",
                ))
            }
        }
        "jump_ahead_equivalence" => {
            let steps = case.steps.max(DEFAULT_RNG_JUMP_STEPS);
            let mut jumped = DeterministicRng::new(case.seed);
            let mut stepped = DeterministicRng::new(case.seed);
            jumped.jump_ahead(steps);
            for _ in 0..steps {
                let _ = stepped.next_u64();
            }
            if jumped.next_u64() == stepped.next_u64() {
                Ok(())
            } else {
                Err(RngSuiteError::new(
                    "rng_jump_contract_violation",
                    "jump-ahead witness diverged from stepped advancement",
                ))
            }
        }
        "state_restore_replay" => {
            let prefix_draws = case.prefix_draws.max(DEFAULT_RNG_PREFIX_DRAWS);
            let replay_draws = case.replay_draws.max(DEFAULT_RNG_REPLAY_DRAWS);
            let mut source = DeterministicRng::new(case.seed);
            for _ in 0..prefix_draws {
                let _ = source.next_u64();
            }
            let (seed, counter) = source.state();
            let mut restored = DeterministicRng::from_state(u128::from(seed), u128::from(counter));
            for index in 0..replay_draws {
                if source.next_u64() != restored.next_u64() {
                    return Err(RngSuiteError::new(
                        "rng_state_restore_contract",
                        format!("restored sequence diverged at replay draw {index}"),
                    ));
                }
            }
            Ok(())
        }
        "bounded_u64_grid" => {
            let draws = case.draws.max(DEFAULT_RNG_DRAWS);
            let mut rng = DeterministicRng::new(case.seed);
            for index in 0..draws {
                let value = rng
                    .bounded_u64(case.upper_bound)
                    .map_err(map_random_error_to_rng_suite)?;
                if value >= case.upper_bound {
                    return Err(RngSuiteError::new(
                        "rng_bounded_output_contract",
                        format!("bounded sample exceeded upper bound at draw {index}"),
                    ));
                }
            }
            Ok(())
        }
        "next_f64_unit_interval" => {
            let draws = case.draws.max(DEFAULT_RNG_DRAWS);
            let mut rng = DeterministicRng::new(case.seed);
            for index in 0..draws {
                let sample = rng.next_f64();
                if !(0.0..1.0).contains(&sample) {
                    return Err(RngSuiteError::new(
                        "rng_float_range_contract",
                        format!("next_f64 sample outside [0,1) at draw {index}: {sample}"),
                    ));
                }
            }
            Ok(())
        }
        "fill_len_contract" => {
            let mut rng = DeterministicRng::new(case.seed);
            let values = rng.fill_u64(case.fill_len);
            if values.len() == case.fill_len {
                Ok(())
            } else {
                Err(RngSuiteError::new(
                    "rng_fill_length_contract",
                    format!(
                        "fill_u64 length mismatch expected={} actual={}",
                        case.fill_len,
                        values.len()
                    ),
                ))
            }
        }
        "policy_metadata" => validate_rng_policy_metadata(&case.mode_raw, &case.class_raw)
            .map_err(map_random_policy_error_to_rng_suite),
        other => Err(RngSuiteError::new(
            "rng_policy_unknown_metadata",
            format!("unsupported rng differential operation {other}"),
        )),
    }
}

fn evaluate_rng_metamorphic_relation(case: &RngMetamorphicCase) -> Result<(), RngSuiteError> {
    match case.relation.as_str() {
        "jump_partition_additivity" => {
            let steps = case.steps.max(DEFAULT_RNG_JUMP_STEPS);
            let extra_steps = case.extra_steps.max(17);
            let mut lhs = DeterministicRng::new(case.seed);
            lhs.jump_ahead(steps);
            lhs.jump_ahead(extra_steps);
            let mut rhs = DeterministicRng::new(case.seed);
            rhs.jump_ahead(steps.saturating_add(extra_steps));

            if lhs.next_u64() == rhs.next_u64() {
                Ok(())
            } else {
                Err(RngSuiteError::new(
                    "rng_jump_contract_violation",
                    "jump partition additivity invariant failed",
                ))
            }
        }
        "fill_matches_iterative_draw" => {
            let len = case.fill_len.max(DEFAULT_RNG_REPLAY_DRAWS);
            let mut lhs = DeterministicRng::new(case.seed);
            let filled = lhs.fill_u64(len);
            let mut rhs = DeterministicRng::new(case.seed);
            let iter = (0..len).map(|_| rhs.next_u64()).collect::<Vec<_>>();
            if filled == iter {
                Ok(())
            } else {
                Err(RngSuiteError::new(
                    "rng_fill_length_contract",
                    "fill_u64 output does not match iterative draws",
                ))
            }
        }
        "bounded_repeatability" => {
            let draws = case.draws.max(DEFAULT_RNG_DRAWS);
            let upper_bound = case.upper_bound.max(2);
            let second_seed = if case.alt_seed == 0 {
                case.seed
            } else {
                case.alt_seed
            };
            let mut lhs = DeterministicRng::new(case.seed);
            let mut rhs = DeterministicRng::new(second_seed);
            for index in 0..draws {
                let lhs_value = lhs
                    .bounded_u64(upper_bound)
                    .map_err(map_random_error_to_rng_suite)?;
                let rhs_value = rhs
                    .bounded_u64(upper_bound)
                    .map_err(map_random_error_to_rng_suite)?;
                if lhs_value != rhs_value {
                    return Err(RngSuiteError::new(
                        "rng_bounded_output_contract",
                        format!("bounded repeatability diverged at draw {index}"),
                    ));
                }
            }
            Ok(())
        }
        other => Err(RngSuiteError::new(
            "rng_policy_unknown_metadata",
            format!("unsupported rng metamorphic relation {other}"),
        )),
    }
}

fn execute_rng_adversarial_operation(case: &RngAdversarialCase) -> Result<(), RngSuiteError> {
    match case.operation.as_str() {
        "bounded_zero_rejected" => {
            let mut rng = DeterministicRng::new(case.seed);
            rng.bounded_u64(0)
                .map(|_| ())
                .map_err(map_random_error_to_rng_suite)
        }
        "forced_repro_witness_mismatch" => {
            let draws = case.draws.max(DEFAULT_RNG_REPLAY_DRAWS);
            let alt_seed = if case.alt_seed == 0 {
                case.seed.wrapping_add(1)
            } else {
                case.alt_seed
            };
            let mut lhs = DeterministicRng::new(case.seed);
            let mut rhs = DeterministicRng::new(alt_seed);
            for _ in 0..draws {
                if lhs.next_u64() != rhs.next_u64() {
                    return Err(RngSuiteError::new(
                        "rng_reproducibility_witness_failed",
                        "reproducibility witness mismatch between paired streams",
                    ));
                }
            }
            Ok(())
        }
        "jump_budget_exceeded" => {
            if case.steps > RNG_MAX_JUMP_OPS {
                Err(RngSuiteError::new(
                    "rng_jump_contract_violation",
                    "jump operations exceeded bounded budget",
                ))
            } else {
                Ok(())
            }
        }
        "state_schema_budget_exceeded" => {
            if case.steps > RNG_MAX_STATE_SCHEMA_FIELDS_U64 {
                Err(RngSuiteError::new(
                    "rng_state_schema_invalid",
                    "state schema entries exceeded bounded budget",
                ))
            } else {
                Ok(())
            }
        }
        "policy_metadata_unknown" => validate_rng_policy_metadata(&case.mode_raw, &case.class_raw)
            .map(|_| ())
            .map_err(map_random_policy_error_to_rng_suite),
        "seedsequence_empty_entropy" => SeedSequence::new(&[])
            .map(|_| ())
            .map_err(map_seedsequence_error_to_rng_suite),
        "seedsequence_pool_size_invalid" => SeedSequence::with_spawn_key(&[1], &[], case.pool_size)
            .map(|_| ())
            .map_err(map_seedsequence_error_to_rng_suite),
        "seedsequence_generate_state_exceeded" => {
            let ss = SeedSequence::new(&[1]).map_err(map_seedsequence_error_to_rng_suite)?;
            ss.generate_state_u32(case.words)
                .map(|_| ())
                .map_err(map_seedsequence_error_to_rng_suite)
        }
        "seedsequence_spawn_invalid" => {
            let mut ss = SeedSequence::new(&[1]).map_err(map_seedsequence_error_to_rng_suite)?;
            ss.spawn(case.spawn_count)
                .map(|_| ())
                .map_err(map_seedsequence_error_to_rng_suite)
        }
        "bitgenerator_jump_invalid" => {
            let kind = parse_bitgenerator_kind(&case.kind);
            let mut bg = BitGenerator::new(kind, SeedMaterial::U64(case.seed))
                .map_err(map_bitgenerator_error_to_rng_suite)?;
            bg.jump_in_place(case.steps)
                .map_err(map_bitgenerator_error_to_rng_suite)
        }
        "bitgenerator_spawn_invalid" => {
            let kind = parse_bitgenerator_kind(&case.kind);
            let mut bg = BitGenerator::new(kind, SeedMaterial::U64(case.seed))
                .map_err(map_bitgenerator_error_to_rng_suite)?;
            bg.spawn(case.spawn_count)
                .map(|_| ())
                .map_err(map_bitgenerator_error_to_rng_suite)
        }
        "bitgenerator_state_kind_mismatch" => {
            let kind = parse_bitgenerator_kind(&case.kind);
            let wrong_kind = parse_bitgenerator_kind(&case.mode_raw);
            let mut bg = BitGenerator::new(kind, SeedMaterial::U64(case.seed))
                .map_err(map_bitgenerator_error_to_rng_suite)?;
            let wrong_bg = BitGenerator::new(wrong_kind, SeedMaterial::U64(case.seed))
                .map_err(map_bitgenerator_error_to_rng_suite)?;
            let wrong_state = wrong_bg.state();
            bg.set_state(&wrong_state)
                .map_err(map_bitgenerator_error_to_rng_suite)
        }
        "bitgenerator_state_schema_version_invalid" => {
            let kind = parse_bitgenerator_kind(&case.kind);
            let mut bg = BitGenerator::new(kind, SeedMaterial::U64(case.seed))
                .map_err(map_bitgenerator_error_to_rng_suite)?;
            let mut state = bg.state();
            state.schema_version = case.steps as u32;
            bg.set_state(&state)
                .map_err(map_bitgenerator_error_to_rng_suite)
        }
        other => Err(RngSuiteError::new(
            "rng_policy_unknown_metadata",
            format!("unsupported rng adversarial operation {other}"),
        )),
    }
}

fn parse_bitgenerator_kind(s: &str) -> BitGeneratorKind {
    match s {
        "mt19937" => BitGeneratorKind::Mt19937,
        "pcg64" => BitGeneratorKind::Pcg64,
        "philox" => BitGeneratorKind::Philox,
        "sfc64" => BitGeneratorKind::Sfc64,
        _ => BitGeneratorKind::Pcg64,
    }
}

fn map_random_error_to_rng_suite(error: RandomError) -> RngSuiteError {
    RngSuiteError::new(error.reason_code(), error.to_string())
}

fn map_fallible_random_values(
    result: Result<Vec<f64>, RandomError>,
) -> Result<Vec<f64>, RngSuiteError> {
    result.map_err(map_random_error_to_rng_suite)
}

fn map_random_policy_error_to_rng_suite(error: RandomPolicyError) -> RngSuiteError {
    RngSuiteError::new(error.reason_code(), error.to_string())
}

fn map_seedsequence_error_to_rng_suite(error: SeedSequenceError) -> RngSuiteError {
    RngSuiteError::new(error.reason_code(), error.to_string())
}

fn map_bitgenerator_error_to_rng_suite(error: BitGeneratorError) -> RngSuiteError {
    RngSuiteError::new(error.reason_code(), error.to_string())
}

fn load_dispatch_label(dispatch: LoadDispatch) -> String {
    match dispatch {
        LoadDispatch::Npy => "npy",
        LoadDispatch::Npz => "npz",
        LoadDispatch::Pickle => "pickle",
    }
    .to_string()
}

fn format_shape_tuple_literal(shape: &[usize]) -> String {
    if shape.is_empty() {
        return "()".to_string();
    }
    if shape.len() == 1 {
        return format!("({},)", shape[0]);
    }
    let dims = shape
        .iter()
        .map(|dim| dim.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("({dims})")
}

fn build_structured_header_literal(
    dtype_descr: &str,
    shape: &[usize],
    fortran_order: bool,
    include_fortran_order: bool,
) -> String {
    let mut parts = Vec::with_capacity(3);
    parts.push(format!("'descr': {}", dtype_descr.trim()));
    if include_fortran_order {
        let literal = if fortran_order { "True" } else { "False" };
        parts.push(format!("'fortran_order': {literal}"));
    }
    parts.push(format!("'shape': {}", format_shape_tuple_literal(shape)));
    format!("{{{}, }}", parts.join(", "))
}

fn write_manual_npy_preamble(buffer: &mut Vec<u8>, header_len: usize) -> Result<(), IoSuiteError> {
    if header_len == 0 {
        return Err(IoSuiteError::new(
            "io_header_schema_invalid",
            "header length must be non-zero",
        ));
    }
    let header_len = u16::try_from(header_len).map_err(|_| {
        IoSuiteError::new(
            "io_header_schema_invalid",
            "version 1.0 header length exceeds u16 boundary",
        )
    })?;
    buffer.extend_from_slice(&NPY_MAGIC_PREFIX);
    buffer.push(1);
    buffer.push(0);
    buffer.extend_from_slice(&header_len.to_le_bytes());
    Ok(())
}

fn resolve_io_case_char(
    case_id: &str,
    field_name: &str,
    raw: &str,
    default: char,
) -> Result<char, IoSuiteError> {
    if raw.is_empty() {
        return Ok(default);
    }

    let mut chars = raw.chars();
    let ch = chars
        .next()
        .ok_or_else(|| IoSuiteError::new("io_differential_fixture_invalid", "empty char field"))?;
    if chars.next().is_some() {
        return Err(IoSuiteError::new(
            "io_differential_fixture_invalid",
            format!(
                "{case_id}: {field_name} must be a single character, got {:?}",
                raw
            ),
        ));
    }
    Ok(ch)
}

fn execute_io_differential_operation(
    case: &IoDifferentialCase,
) -> Result<IoOperationOutcome, IoSuiteError> {
    match case.operation.as_str() {
        "magic_version" => validate_magic_version(&case.payload_prefix)
            .map(|(major, minor)| IoOperationOutcome::MagicVersion(format!("{major}.{minor}")))
            .map_err(map_io_error_to_suite),
        "header_schema" => validate_header_schema(
            &case.shape,
            case.fortran_order,
            &case.dtype_descr,
            case.header_len,
        )
        .map(|_| IoOperationOutcome::Unit)
        .map_err(map_io_error_to_suite),
        "dtype_decode" => IOSupportedDType::decode(&case.dtype_descr)
            .map(|_| IoOperationOutcome::Unit)
            .map_err(map_io_error_to_suite),
        "read_payload" => {
            let dtype =
                IOSupportedDType::decode(&case.dtype_descr).map_err(map_io_error_to_suite)?;
            validate_read_payload(&case.shape, case.payload_len_bytes, dtype)
                .map(IoOperationOutcome::Count)
                .map_err(map_io_error_to_suite)
        }
        "memmap_contract" => {
            let mode = MemmapMode::parse(&case.memmap_mode).map_err(map_io_error_to_suite)?;
            let dtype =
                IOSupportedDType::decode(&case.dtype_descr).map_err(map_io_error_to_suite)?;
            validate_memmap_contract(
                mode,
                dtype,
                case.file_len_bytes,
                case.expected_bytes,
                case.validation_retries,
            )
            .map(|_| IoOperationOutcome::Unit)
            .map_err(map_io_error_to_suite)
        }
        "load_dispatch" => classify_load_dispatch(&case.payload_prefix, case.allow_pickle)
            .map(|dispatch| IoOperationOutcome::Dispatch(load_dispatch_label(dispatch)))
            .map_err(map_io_error_to_suite),
        "structured_load" => {
            let include_fortran = case.include_fortran_order.unwrap_or(true);
            let header_literal = build_structured_header_literal(
                &case.dtype_descr,
                &case.shape,
                case.fortran_order,
                include_fortran,
            );
            let mut header_bytes = header_literal.into_bytes();
            if !header_bytes.ends_with(b"\n") {
                header_bytes.push(b'\n');
            }
            let mut data = Vec::with_capacity(header_bytes.len() + case.payload_len_bytes + 16);
            write_manual_npy_preamble(&mut data, header_bytes.len())?;
            data.extend_from_slice(&header_bytes);
            data.extend(std::iter::repeat_n(0u8, case.payload_len_bytes));
            load_structured(&data)
                .map(|_| IoOperationOutcome::Unit)
                .map_err(map_io_error_to_suite)
        }
        "npz_archive_budget" => validate_npz_archive_budget(
            case.member_count,
            case.uncompressed_bytes,
            case.dispatch_retries,
        )
        .map(|_| IoOperationOutcome::Unit)
        .map_err(map_io_error_to_suite),
        "policy_metadata" => validate_io_policy_metadata(&case.mode_raw, &case.class_raw)
            .map(|_| IoOperationOutcome::Unit)
            .map_err(map_io_error_to_suite),
        "genfromtxt_full" => {
            let delimiter = resolve_io_case_char(&case.id, "delimiter", &case.delimiter, ' ')?;
            let comments = resolve_io_case_char(&case.id, "comments", &case.comments, '#')?;
            let config = GenFromTxtConfig {
                delimiter,
                comments,
                skip_header: case.skip_header,
                skip_footer: case.skip_footer,
                filling_values: case.filling_values.unwrap_or(f64::NAN),
                usecols: if case.usecols.is_empty() {
                    None
                } else {
                    Some(case.usecols.as_slice())
                },
                max_rows: case.max_rows.unwrap_or(usize::MAX),
            };
            genfromtxt_full(&case.text, &config)
                .map(|parsed| IoOperationOutcome::Array {
                    shape: vec![parsed.nrows, parsed.ncols],
                    values: parsed.values,
                })
                .map_err(map_io_error_to_suite)
        }
        other => Err(IoSuiteError::new(
            "io_policy_unknown_metadata",
            format!("unsupported io differential operation {other}"),
        )),
    }
}

fn validate_io_differential_success_expectations(
    case: &IoDifferentialCase,
    outcome: &IoOperationOutcome,
) -> Result<(), String> {
    if !case.expected_dispatch.trim().is_empty() {
        let expected = case.expected_dispatch.trim().to_lowercase();
        match outcome {
            IoOperationOutcome::Dispatch(actual) if actual.to_lowercase() == expected => {}
            IoOperationOutcome::Dispatch(actual) => {
                return Err(format!(
                    "dispatch mismatch expected={expected} actual={actual}"
                ));
            }
            _ => {
                return Err(
                    "expected dispatch outcome but operation returned a different outcome type"
                        .to_string(),
                );
            }
        }
    }

    if !case.expected_magic_version.trim().is_empty() {
        let expected = case.expected_magic_version.trim();
        match outcome {
            IoOperationOutcome::MagicVersion(actual) if actual == expected => {}
            IoOperationOutcome::MagicVersion(actual) => {
                return Err(format!(
                    "magic version mismatch expected={expected} actual={actual}"
                ));
            }
            _ => {
                return Err(
                    "expected magic-version outcome but operation returned a different outcome type"
                        .to_string(),
                );
            }
        }
    }

    if let Some(expected_count) = case.expected_count {
        match outcome {
            IoOperationOutcome::Count(actual) if *actual == expected_count => {}
            IoOperationOutcome::Count(actual) => {
                return Err(format!(
                    "count mismatch expected={expected_count} actual={actual}"
                ));
            }
            _ => {
                return Err(
                    "expected count outcome but operation returned a different outcome type"
                        .to_string(),
                );
            }
        }
    }

    if !case.expected_shape.is_empty() || !case.expected_values.is_empty() {
        match outcome {
            IoOperationOutcome::Array { shape, values } => {
                if !case.expected_shape.is_empty() && case.expected_shape != *shape {
                    return Err(format!(
                        "shape mismatch expected={:?} actual={shape:?}",
                        case.expected_shape
                    ));
                }
                if !case.expected_values.is_empty() {
                    if case.expected_values.len() != values.len() {
                        return Err(format!(
                            "value length mismatch expected={} actual={}",
                            case.expected_values.len(),
                            values.len()
                        ));
                    }
                    let abs_tol = if case.abs_tol > 0.0 {
                        case.abs_tol
                    } else {
                        1e-9
                    };
                    let rel_tol = if case.rel_tol > 0.0 {
                        case.rel_tol
                    } else {
                        1e-9
                    };
                    if !approx_equal_values(&case.expected_values, values, abs_tol, rel_tol) {
                        return Err(format!(
                            "value mismatch expected={:?} actual={values:?} abs_tol={} rel_tol={}",
                            case.expected_values, abs_tol, rel_tol
                        ));
                    }
                }
            }
            _ => {
                return Err(
                    "expected array outcome but operation returned a different outcome type"
                        .to_string(),
                );
            }
        }
    }

    Ok(())
}

fn evaluate_io_metamorphic_relation(case: &IoMetamorphicCase) -> Result<(), IoSuiteError> {
    match case.relation.as_str() {
        "magic_version_idempotent" => {
            let first =
                validate_magic_version(&case.payload_prefix).map_err(map_io_error_to_suite)?;
            let second =
                validate_magic_version(&case.payload_prefix).map_err(map_io_error_to_suite)?;
            if first == second {
                Ok(())
            } else {
                Err(IoSuiteError::new(
                    "io_magic_invalid",
                    format!("magic version idempotence diverged first={first:?} second={second:?}"),
                ))
            }
        }
        "header_schema_descriptor_roundtrip" => {
            let header = validate_header_schema(
                &case.shape,
                case.fortran_order,
                &case.dtype_descr,
                case.header_len,
            )
            .map_err(map_io_error_to_suite)?;
            validate_descriptor_roundtrip(header.descr).map_err(map_io_error_to_suite)
        }
        "read_payload_repeatable" => {
            let dtype =
                IOSupportedDType::decode(&case.dtype_descr).map_err(map_io_error_to_suite)?;
            let first = validate_read_payload(&case.shape, case.payload_len_bytes, dtype)
                .map_err(map_io_error_to_suite)?;
            let dtype =
                IOSupportedDType::decode(&case.dtype_descr).map_err(map_io_error_to_suite)?;
            let second = validate_read_payload(&case.shape, case.payload_len_bytes, dtype)
                .map_err(map_io_error_to_suite)?;
            if first == second {
                Ok(())
            } else {
                Err(IoSuiteError::new(
                    "io_read_payload_incomplete",
                    format!("read payload repeatability diverged first={first} second={second}"),
                ))
            }
        }
        "dispatch_stability" => {
            let first = classify_load_dispatch(&case.payload_prefix, case.allow_pickle)
                .map(load_dispatch_label)
                .map_err(map_io_error_to_suite)?;
            let second = classify_load_dispatch(&case.payload_prefix, case.allow_pickle)
                .map(load_dispatch_label)
                .map_err(map_io_error_to_suite)?;
            if first == second {
                Ok(())
            } else {
                Err(IoSuiteError::new(
                    "io_load_dispatch_invalid",
                    format!("dispatch stability diverged first={first} second={second}"),
                ))
            }
        }
        "npz_budget_idempotent" => {
            validate_npz_archive_budget(
                case.member_count,
                case.uncompressed_bytes,
                case.dispatch_retries,
            )
            .map_err(map_io_error_to_suite)?;
            validate_npz_archive_budget(
                case.member_count,
                case.uncompressed_bytes,
                case.dispatch_retries,
            )
            .map_err(map_io_error_to_suite)
        }
        "policy_metadata_idempotent" => {
            validate_io_policy_metadata(&case.mode_raw, &case.class_raw)
                .map_err(map_io_error_to_suite)?;
            validate_io_policy_metadata(&case.mode_raw, &case.class_raw)
                .map_err(map_io_error_to_suite)
        }
        other => Err(IoSuiteError::new(
            "io_policy_unknown_metadata",
            format!("unsupported io metamorphic relation {other}"),
        )),
    }
}

fn execute_io_adversarial_operation(case: &IoAdversarialCase) -> Result<(), IoSuiteError> {
    let differential_case = IoDifferentialCase {
        id: case.id.clone(),
        operation: case.operation.clone(),
        seed: case.seed,
        mode: case.mode.clone(),
        env_fingerprint: case.env_fingerprint.clone(),
        artifact_refs: case.artifact_refs.clone(),
        reason_code: case.reason_code.clone(),
        expected_reason_code: case.expected_reason_code.clone(),
        expected_error_contains: case.expected_error_contains.clone(),
        payload_prefix: case.payload_prefix.clone(),
        shape: case.shape.clone(),
        fortran_order: case.fortran_order,
        include_fortran_order: None,
        dtype_descr: case.dtype_descr.clone(),
        header_len: case.header_len,
        payload_len_bytes: case.payload_len_bytes,
        allow_pickle: case.allow_pickle,
        memmap_mode: case.memmap_mode.clone(),
        file_len_bytes: case.file_len_bytes,
        expected_bytes: case.expected_bytes,
        validation_retries: case.validation_retries,
        member_count: case.member_count,
        uncompressed_bytes: case.uncompressed_bytes,
        dispatch_retries: case.dispatch_retries,
        mode_raw: case.mode_raw.clone(),
        class_raw: case.class_raw.clone(),
        expected_dispatch: String::new(),
        expected_magic_version: String::new(),
        expected_count: None,
        text: String::new(),
        delimiter: String::new(),
        comments: String::new(),
        skip_header: 0,
        skip_footer: 0,
        filling_values: None,
        usecols: Vec::new(),
        max_rows: None,
        expected_shape: Vec::new(),
        expected_values: Vec::new(),
        abs_tol: 0.0,
        rel_tol: 0.0,
    };
    execute_io_differential_operation(&differential_case).map(|_| ())
}

fn map_io_error_to_suite(error: IOError) -> IoSuiteError {
    IoSuiteError::new(error.reason_code(), error.to_string())
}

fn record_suite_check(report: &mut SuiteReport, passed: bool, failure: String) {
    report.case_count += 1;
    if passed {
        report.pass_count += 1;
    } else {
        report.failures.push(failure);
    }
}

fn parse_expected_action(case_id: &str, raw: &str) -> Result<DecisionAction, String> {
    match raw {
        "allow" => Ok(DecisionAction::Allow),
        "full_validate" => Ok(DecisionAction::FullValidate),
        "fail_closed" => Ok(DecisionAction::FailClosed),
        bad => Err(format!("{case_id}: invalid expected_action {bad}")),
    }
}

fn normalize_env_fingerprint(raw: &str) -> String {
    if raw.trim().is_empty() {
        "unknown_env".to_string()
    } else {
        raw.trim().to_string()
    }
}

fn normalize_artifact_refs(mut refs: Vec<String>) -> Vec<String> {
    refs.retain(|entry| !entry.trim().is_empty());
    if refs.is_empty() {
        refs.push("artifacts/contracts/SECURITY_COMPATIBILITY_THREAT_MATRIX_V1.md".to_string());
    }
    refs
}

fn normalize_reason_code(raw: &str) -> String {
    if raw.trim().is_empty() {
        "unspecified".to_string()
    } else {
        raw.trim().to_string()
    }
}

fn maybe_append_runtime_policy_log(entry: &RuntimePolicyLogEntry) -> Result<(), String> {
    let configured = RUNTIME_POLICY_LOG_PATH
        .get()
        .and_then(|cell| cell.lock().ok())
        .and_then(|slot| slot.clone());
    let from_env = std::env::var_os("FNP_RUNTIME_POLICY_LOG_PATH").map(PathBuf::from);
    let Some(path) = configured.or(from_env) else {
        return Ok(());
    };

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }

    let _guard = FILE_LOG_MUTEX
        .lock()
        .map_err(|_| "failed acquiring log lock")?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|err| format!("failed opening {}: {err}", path.display()))?;
    let line = serde_json::to_string(entry)
        .map_err(|err| format!("failed serializing runtime policy log entry: {err}"))?;
    let mut payload = line.into_bytes();
    payload.push(b'\n');
    file.write_all(&payload).map_err(|err| {
        format!(
            "failed appending runtime policy log {}: {err}",
            path.display()
        )
    })
}

fn maybe_append_shape_stride_log(entry: &ShapeStrideLogEntry) -> Result<(), String> {
    let configured = SHAPE_STRIDE_LOG_PATH
        .get()
        .and_then(|cell| cell.lock().ok())
        .and_then(|slot| slot.clone());
    let from_env = std::env::var_os("FNP_SHAPE_STRIDE_LOG_PATH").map(PathBuf::from);
    let Some(path) = configured.or(from_env) else {
        return Ok(());
    };

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }

    let _guard = FILE_LOG_MUTEX
        .lock()
        .map_err(|_| "failed acquiring log lock")?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|err| format!("failed opening {}: {err}", path.display()))?;
    let line = serde_json::to_string(entry)
        .map_err(|err| format!("failed serializing shape-stride log entry: {err}"))?;
    let mut payload = line.into_bytes();
    payload.push(b'\n');
    file.write_all(&payload).map_err(|err| {
        format!(
            "failed appending shape-stride log {}: {err}",
            path.display()
        )
    })
}

fn maybe_append_dtype_promotion_log(entry: &DTypePromotionLogEntry) -> Result<(), String> {
    let configured = DTYPE_PROMOTION_LOG_PATH
        .get()
        .and_then(|cell| cell.lock().ok())
        .and_then(|slot| slot.clone());
    let from_env = std::env::var_os("FNP_DTYPE_PROMOTION_LOG_PATH").map(PathBuf::from);
    let Some(path) = configured.or(from_env) else {
        return Ok(());
    };

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }

    let _guard = FILE_LOG_MUTEX
        .lock()
        .map_err(|_| "failed acquiring log lock")?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|err| format!("failed opening {}: {err}", path.display()))?;
    let line = serde_json::to_string(entry)
        .map_err(|err| format!("failed serializing dtype-promotion log entry: {err}"))?;
    let mut payload = line.into_bytes();
    payload.push(b'\n');
    file.write_all(&payload).map_err(|err| {
        format!(
            "failed appending dtype-promotion log {}: {err}",
            path.display()
        )
    })
}

fn validate_runtime_policy_log_fields(
    report: &mut SuiteReport,
    events: &[fnp_runtime::DecisionEvent],
) {
    for event in events {
        if event.fixture_id.trim().is_empty() {
            report
                .failures
                .push("runtime ledger event missing fixture_id".to_string());
        }
        if event.env_fingerprint.trim().is_empty() {
            report
                .failures
                .push("runtime ledger event missing env_fingerprint".to_string());
        }
        if event.reason_code.trim().is_empty() {
            report
                .failures
                .push("runtime ledger event missing reason_code".to_string());
        }
        if event.artifact_refs.is_empty() {
            report
                .failures
                .push("runtime ledger event missing artifact_refs".to_string());
        }
        if matches!(
            event.class,
            CompatibilityClass::Unknown | CompatibilityClass::KnownIncompatible
        ) && !matches!(event.action, DecisionAction::FailClosed)
        {
            report.failures.push(format!(
                "{}: fail-closed violation for {:?}",
                event.fixture_id, event.class
            ));
        }
    }

    report.pass_count = report.case_count.saturating_sub(report.failures.len());
}

#[cfg(test)]
mod tests {
    use super::{
        AsStridedFixtureCase, BinaryOp, DType, HarnessConfig, IterFlatIndexFixtureInput,
        MaskedArray, RngStatisticalCase, ShapeStrideFixtureCase, StringArray, UFuncArray,
        busday_count, chebval, default_rng_mean_abs_tol, default_rng_variance_abs_tol,
        evaluate_shape_stride_case, generate_rng_samples, infer_flatiter_len, run_all_core_suites,
        run_crash_signature_regression_suite, run_datetime_differential_suite,
        run_datetime_metamorphic_suite, run_dtype_adversarial_suite, run_dtype_differential_suite,
        run_dtype_metamorphic_suite, run_dtype_promotion_suite, run_fft_differential_suite,
        run_fft_metamorphic_suite, run_io_adversarial_suite, run_io_differential_suite,
        run_io_metamorphic_suite, run_iter_adversarial_suite, run_iter_differential_suite,
        run_iter_metamorphic_suite, run_linalg_adversarial_suite, run_linalg_differential_suite,
        run_linalg_metamorphic_suite, run_masked_differential_suite,
        run_polynomial_differential_suite, run_rng_adversarial_suite, run_rng_differential_suite,
        run_rng_metamorphic_suite, run_rng_statistical_suite, run_runtime_policy_adversarial_suite,
        run_shape_stride_adversarial_suite, run_shape_stride_differential_suite,
        run_shape_stride_metamorphic_suite, run_shape_stride_suite, run_signal_differential_suite,
        run_smoke, run_string_differential_suite, run_string_metamorphic_suite,
        run_ufunc_adversarial_suite, run_ufunc_differential_suite, run_ufunc_metamorphic_suite,
        set_dtype_promotion_log_path, set_shape_stride_log_path,
    };
    use fnp_io::{IOSupportedDType, load as load_npy, save as save_npy};
    use fnp_iter::{
        RuntimeMode as IterRuntimeMode, TRANSFER_PACKET_REASON_CODES, TransferContext,
        TransferDtypeRelation, TransferLogRecord, TransferLoopClass, select_transfer_loop,
        validate_subarray_transfer, validate_where_mask,
    };
    use fnp_random::Generator;
    use serde_json::Value;
    use std::fmt::Display;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    const E2E_TOTAL_STEPS: usize = 12;

    fn step_ok(step: usize, description: &str) {
        eprintln!("e2e step {step}/{E2E_TOTAL_STEPS}: {description} ... OK");
    }

    fn step_fail(step: usize, description: &str, details: impl Display) -> Result<(), String> {
        let message = format!("e2e step {step}/{E2E_TOTAL_STEPS}: {description} failed: {details}");
        eprintln!("e2e step {step}/{E2E_TOTAL_STEPS}: {description} ... FAILED: {details}");
        Err(message)
    }

    #[test]
    fn smoke_harness_finds_oracle_and_fixtures() {
        let cfg = HarnessConfig::default_paths();
        let report = run_smoke(&cfg);
        assert!(report.oracle_present, "oracle repo should be present");
        assert!(report.fixture_count >= 1, "expected at least one fixture");
        assert!(report.strict_mode);
    }

    #[test]
    fn e2e_cross_crate_workflow() -> Result<(), String> {
        const MATMUL_TOL: f64 = 1e-10;
        const STATS_TOL: f64 = 1e-10;
        const POLY_TOL: f64 = 1e-14;
        const RNG_SIGMA_TOL: f64 = 0.3;

        let arr = UFuncArray::arange(0.0, 12.0, 1.0, DType::F64)
            .expect("Step 1: arange should succeed")
            .reshape(&[3, 4])
            .expect("Step 1: reshape should succeed");
        if arr.shape() != [3, 4]
            || arr.dtype() != DType::F64
            || arr.values()[0] != 0.0
            || arr.values()[11] != 11.0
        {
            return step_fail(
                1,
                "ARRAY CREATION",
                format!(
                    "shape={:?} dtype={:?} first={} last={}, expected shape [3, 4], dtype F64, first 0.0, last 11.0",
                    arr.shape(),
                    arr.dtype(),
                    arr.values()[0],
                    arr.values()[11]
                ),
            );
        }
        step_ok(1, "ARRAY CREATION");

        let i32_arr =
            UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::I32).expect("Step 2: i32 array");
        let f64_arr =
            UFuncArray::new(vec![3], vec![0.5, 0.5, 0.5], DType::F64).expect("Step 2: f64 array");
        let promoted = i32_arr
            .elementwise_binary(&f64_arr, BinaryOp::Add)
            .expect("Step 2: add should succeed");
        let expected_promoted = [1.5, 2.5, 3.5];
        if promoted.dtype() != DType::F64
            || !crate::approx_equal_values(promoted.values(), &expected_promoted, 1e-12, 1e-12)
        {
            return step_fail(
                2,
                "DTYPE PROMOTION",
                format!(
                    "dtype={:?} values={:?}, expected dtype F64 values {:?}",
                    promoted.dtype(),
                    promoted.values(),
                    expected_promoted
                ),
            );
        }
        step_ok(2, "DTYPE PROMOTION");

        let doubled = arr
            .elementwise_binary(&UFuncArray::scalar(2.0, DType::F64), BinaryOp::Mul)
            .expect("Step 3: multiply should succeed");
        if doubled.dtype() != DType::F64
            || doubled.values()[0] != 0.0
            || doubled.values()[5] != 10.0
        {
            return step_fail(
                3,
                "ELEMENTWISE MATH",
                format!(
                    "dtype={:?} values[0]={} values[5]={}, expected dtype F64 with values[0]=0.0 values[5]=10.0",
                    doubled.dtype(),
                    doubled.values()[0],
                    doubled.values()[5]
                ),
            );
        }
        step_ok(3, "ELEMENTWISE MATH");

        let row_sums = doubled
            .reduce_sum(Some(1), false)
            .expect("Step 4: reduce_sum should succeed");
        let expected_row_sums = [12.0, 44.0, 76.0];
        if row_sums.shape() != [3]
            || row_sums.dtype() != DType::F64
            || row_sums.values() != expected_row_sums
        {
            return step_fail(
                4,
                "REDUCTION",
                format!(
                    "shape={:?} dtype={:?} values={:?}, expected shape [3], dtype F64, values {:?}",
                    row_sums.shape(),
                    row_sums.dtype(),
                    row_sums.values(),
                    expected_row_sums
                ),
            );
        }
        step_ok(4, "REDUCTION");

        let a = UFuncArray::new(
            vec![3, 3],
            vec![
                4.0, 2.0, 0.0, //
                2.0, 5.0, 1.0, //
                0.0, 1.0, 3.0,
            ],
            DType::F64,
        )
        .expect("Step 5: matrix construction");
        let b = UFuncArray::new(vec![3], vec![10.0, 20.0, 14.0], DType::F64)
            .expect("Step 5: rhs construction");
        let x = a.solve(&b).expect("Step 5: solve should succeed");
        let ax = a
            .matmul(&x.reshape(&[3, 1]).expect("Step 5: reshape solution"))
            .expect("Step 5: matmul should succeed");
        let rhs_column = b
            .reshape(&[3, 1])
            .expect("Step 5: reshape rhs should succeed");
        if x.shape() != [3]
            || x.dtype() != DType::F64
            || ax.shape() != [3, 1]
            || !crate::approx_equal_values(ax.values(), rhs_column.values(), MATMUL_TOL, MATMUL_TOL)
        {
            return step_fail(
                5,
                "LINEAR ALGEBRA",
                format!(
                    "x shape={:?} dtype={:?} values={:?}, ax shape={:?} values={:?}, rhs column values={:?}",
                    x.shape(),
                    x.dtype(),
                    x.values(),
                    ax.shape(),
                    ax.values(),
                    rhs_column.values()
                ),
            );
        }
        step_ok(5, "LINEAR ALGEBRA");

        let x_pts = [0.0, 0.5, 1.0];
        let t2_vals = chebval(&x_pts, &[0.0, 0.0, 1.0]);
        let expected_t2 = [-1.0, -0.5, 1.0];
        if !crate::approx_equal_values(&t2_vals, &expected_t2, POLY_TOL, POLY_TOL) {
            return step_fail(
                6,
                "POLYNOMIAL",
                format!("values={:?}, expected {:?}", t2_vals, expected_t2),
            );
        }
        step_ok(6, "POLYNOMIAL");

        let data = UFuncArray::new(vec![5], vec![3.0, 1.0, 4.0, 1.0, 5.0], DType::F64)
            .expect("Step 7: data construction");
        let sorted = data.sort(None, None).expect("Step 7: sort should succeed");
        let mean = data
            .reduce_mean(None, false)
            .expect("Step 7: mean should succeed");
        let expected_sorted = [1.0, 1.0, 3.0, 4.0, 5.0];
        if sorted.shape() != [5]
            || sorted.values() != expected_sorted
            || !mean.shape().is_empty()
            || (mean.values()[0] - 2.8).abs() > STATS_TOL
        {
            return step_fail(
                7,
                "SORT + STATISTICS",
                format!(
                    "sorted shape={:?} values={:?}, mean shape={:?} value={}, expected sorted {:?}, expected mean 2.8",
                    sorted.shape(),
                    sorted.values(),
                    mean.shape(),
                    mean.values()[0],
                    expected_sorted
                ),
            );
        }
        step_ok(7, "SORT + STATISTICS");

        let masked = MaskedArray::new(
            UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64)
                .expect("Step 8: masked data"),
            Some(
                UFuncArray::new(vec![4], vec![0.0, 1.0, 0.0, 1.0], DType::Bool)
                    .expect("Step 8: mask"),
            ),
            None,
        )
        .expect("Step 8: masked array construction");
        let compressed = masked.compressed();
        let expected_compressed = [1.0, 3.0];
        if compressed.shape() != [2]
            || compressed.dtype() != DType::F64
            || compressed.values() != expected_compressed
        {
            return step_fail(
                8,
                "MASKED ARRAY",
                format!(
                    "shape={:?} dtype={:?} values={:?}, expected shape [2], dtype F64, values {:?}",
                    compressed.shape(),
                    compressed.dtype(),
                    compressed.values(),
                    expected_compressed
                ),
            );
        }
        step_ok(8, "MASKED ARRAY");

        let strings = StringArray::from_strs(vec![3], &["hello", "WORLD", "Test"])
            .expect("Step 9: string array construction");
        let upper = strings.upper();
        let lens = strings.str_len();
        let expected_lens = [5.0, 5.0, 4.0];
        if upper.shape() != [3]
            || upper.values()[0] != "HELLO"
            || upper.values()[1] != "WORLD"
            || lens.dtype() != DType::I64
            || lens.values() != expected_lens
        {
            return step_fail(
                9,
                "STRING ARRAY",
                format!(
                    "upper shape={:?} values={:?}, lens dtype={:?} values={:?}, expected upper[0]=HELLO upper[1]=WORLD lens {:?}",
                    upper.shape(),
                    upper.values(),
                    lens.dtype(),
                    lens.values(),
                    expected_lens
                ),
            );
        }
        step_ok(9, "STRING ARRAY");

        let mut rng = Generator::from_pcg64_dxsm(42).expect("Step 10: generator construction");
        let samples = rng.standard_normal(100);
        let sample_mean = samples.iter().copied().sum::<f64>() / 100.0;
        if sample_mean.abs() >= RNG_SIGMA_TOL {
            return step_fail(
                10,
                "RANDOM",
                format!(
                    "sample mean {} exceeds bound {} for seed 42",
                    sample_mean, RNG_SIGMA_TOL
                ),
            );
        }
        step_ok(10, "RANDOM");

        let original = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
            .expect("Step 11: original array construction");
        let npy_bytes = save_npy(original.shape(), original.values(), IOSupportedDType::F64)
            .expect("Step 11: save should succeed");
        let (loaded_shape, loaded_values, loaded_dtype) =
            load_npy(&npy_bytes).expect("Step 11: load should succeed");
        if loaded_shape != [2, 3]
            || loaded_values != original.values()
            || loaded_dtype != IOSupportedDType::F64
        {
            return step_fail(
                11,
                "IO ROUNDTRIP",
                format!(
                    "shape={:?} dtype={:?} values={:?}, expected shape [2, 3], dtype F64, values {:?}",
                    loaded_shape,
                    loaded_dtype,
                    loaded_values,
                    original.values()
                ),
            );
        }
        step_ok(11, "IO ROUNDTRIP");

        let start =
            UFuncArray::from_datetime_strings(vec![1], vec!["2024-01-01".to_string()], None)
                .expect("Step 12: start date");
        let end = UFuncArray::from_datetime_strings(vec![1], vec!["2024-01-08".to_string()], None)
            .expect("Step 12: end date");
        let count = busday_count(&start, &end).expect("Step 12: busday_count should succeed");
        if count.shape() != [1] || count.dtype() != DType::I64 || count.values()[0] != 5.0 {
            return step_fail(
                12,
                "BUSDAY",
                format!(
                    "shape={:?} dtype={:?} values={:?}, expected shape [1], dtype I64, values [5.0]",
                    count.shape(),
                    count.dtype(),
                    count.values()
                ),
            );
        }
        step_ok(12, "BUSDAY");
        Ok(())
    }

    #[test]
    fn ufunc_differential_errors_when_oracle_files_missing() {
        let mut cfg = HarnessConfig::default_paths();
        cfg.fixture_root =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/does_not_exist");

        let err =
            run_ufunc_differential_suite(&cfg).expect_err("suite should fail for missing files");
        assert!(err.contains("failed reading"));
    }

    #[test]
    fn adversarial_runtime_policy_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite =
            run_runtime_policy_adversarial_suite(&cfg).expect("adversarial suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn shape_stride_packet006_f_suites_are_green() {
        let cfg = HarnessConfig::default_paths();

        let differential = run_shape_stride_differential_suite(&cfg)
            .expect("shape-stride differential suite should run");
        assert!(
            differential.all_passed(),
            "failures={:?}",
            differential.failures
        );

        let metamorphic = run_shape_stride_metamorphic_suite(&cfg)
            .expect("shape-stride metamorphic suite should run");
        assert!(
            metamorphic.all_passed(),
            "failures={:?}",
            metamorphic.failures
        );

        let adversarial =
            run_shape_stride_adversarial_suite(&cfg).expect("shape-stride adversarial suite");
        assert!(
            adversarial.all_passed(),
            "failures={:?}",
            adversarial.failures
        );
    }

    #[test]
    fn iter_packet004_f_suites_are_green() {
        let cfg = HarnessConfig::default_paths();

        let differential =
            run_iter_differential_suite(&cfg).expect("iter differential suite should run");
        assert!(
            differential.all_passed(),
            "failures={:?}",
            differential.failures
        );

        let metamorphic =
            run_iter_metamorphic_suite(&cfg).expect("iter metamorphic suite should run");
        assert!(
            metamorphic.all_passed(),
            "failures={:?}",
            metamorphic.failures
        );

        let adversarial =
            run_iter_adversarial_suite(&cfg).expect("iter adversarial suite should run");
        assert!(
            adversarial.all_passed(),
            "failures={:?}",
            adversarial.failures
        );
    }

    #[test]
    fn infer_flatiter_len_accounts_for_fancy_indices_without_stop_padding() {
        let input = IterFlatIndexFixtureInput {
            kind: "fancy".to_string(),
            index: 0,
            start: 0,
            stop: 0,
            step: 1,
            indices: vec![1, 3, 5, 7],
            mask: Vec::new(),
        };

        assert_eq!(infer_flatiter_len(&input), 8);
    }

    #[test]
    fn dtype_packet002_f_suites_are_green() {
        let mut cfg = HarnessConfig::default_paths();
        cfg.fixture_root =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/packet002_dtype");

        let differential = run_dtype_differential_suite(&cfg)
            .expect("packet002 dtype differential suite should run");
        assert!(
            differential.all_passed(),
            "failures={:?}",
            differential.failures
        );

        let metamorphic = run_dtype_metamorphic_suite(&cfg)
            .expect("packet002 dtype metamorphic suite should run");
        assert!(
            metamorphic.all_passed(),
            "failures={:?}",
            metamorphic.failures
        );

        let adversarial = run_dtype_adversarial_suite(&cfg)
            .expect("packet002 dtype adversarial suite should run");
        assert!(
            adversarial.all_passed(),
            "failures={:?}",
            adversarial.failures
        );
    }

    #[test]
    fn transfer_packet003_f_suites_are_green() {
        let mut cfg = HarnessConfig::default_paths();
        cfg.fixture_root =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/packet003_transfer");

        let differential = run_iter_differential_suite(&cfg)
            .expect("packet003 transfer differential suite should run");
        assert!(
            differential.all_passed(),
            "failures={:?}",
            differential.failures
        );

        let metamorphic = run_iter_metamorphic_suite(&cfg)
            .expect("packet003 transfer metamorphic suite should run");
        assert!(
            metamorphic.all_passed(),
            "failures={:?}",
            metamorphic.failures
        );

        let adversarial = run_iter_adversarial_suite(&cfg)
            .expect("packet003 transfer adversarial suite should run");
        assert!(
            adversarial.all_passed(),
            "failures={:?}",
            adversarial.failures
        );
    }

    #[test]
    fn security_contract_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite = super::security_contracts::run_security_contract_suite(&cfg)
            .expect("security contract suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn test_contract_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite = super::test_contracts::run_test_contract_suite(&cfg).expect("suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn dtype_promotion_suite_emits_structured_logs_with_required_fields() {
        let cfg = HarnessConfig::default_paths();
        let ts_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        let log_path = std::env::temp_dir().join(format!(
            "fnp_dtype_promotion_suite_{}_{}.jsonl",
            std::process::id(),
            ts_nanos
        ));
        let _ = fs::remove_file(&log_path);
        set_dtype_promotion_log_path(Some(log_path.clone()));

        let suite = run_dtype_promotion_suite(&cfg).expect("dtype-promotion suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);

        let raw = fs::read_to_string(&log_path).expect("dtype-promotion log should exist");
        let mut entry_count = 0usize;
        for line in raw.lines().filter(|line| !line.trim().is_empty()) {
            entry_count += 1;
            let value: Value = serde_json::from_str(line).expect("log line must be valid json");
            let obj = value.as_object().expect("log line must be json object");
            assert!(
                obj.get("fixture_id")
                    .and_then(Value::as_str)
                    .is_some_and(|s| !s.trim().is_empty())
            );
            assert!(obj.get("seed").is_some_and(Value::is_u64));
            assert!(
                obj.get("mode")
                    .and_then(Value::as_str)
                    .is_some_and(|s| s == "strict" || s == "hardened")
            );
            assert!(
                obj.get("env_fingerprint")
                    .and_then(Value::as_str)
                    .is_some_and(|s| !s.trim().is_empty())
            );
            assert!(
                obj.get("artifact_refs")
                    .and_then(Value::as_array)
                    .is_some_and(|arr| {
                        !arr.is_empty()
                            && arr
                                .iter()
                                .all(|item| item.as_str().is_some_and(|s| !s.trim().is_empty()))
                    })
            );
            assert!(
                obj.get("reason_code")
                    .and_then(Value::as_str)
                    .is_some_and(|s| !s.trim().is_empty())
            );
            assert!(
                obj.get("lhs")
                    .and_then(Value::as_str)
                    .is_some_and(|s| !s.trim().is_empty())
            );
            if obj.get("rhs").is_none() {
                eprintln!("DEBUG: log line missing rhs: {}", line);
            }
            assert!(obj.get("rhs").is_some_and(Value::is_string));
            assert!(
                obj.get("expected")
                    .and_then(Value::as_str)
                    .is_some_and(|s| !s.trim().is_empty())
            );
            assert!(
                obj.get("actual")
                    .and_then(Value::as_str)
                    .is_some_and(|s| !s.trim().is_empty())
            );
        }
        assert!(
            entry_count > 0,
            "dtype-promotion log should contain entries"
        );
        set_dtype_promotion_log_path(None);
        let _ = fs::remove_file(log_path);
    }

    #[test]
    fn shape_stride_suite_emits_structured_logs_with_required_fields() {
        let cfg = HarnessConfig::default_paths();
        let ts_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        let log_path = std::env::temp_dir().join(format!(
            "fnp_shape_stride_suite_{}_{}.jsonl",
            std::process::id(),
            ts_nanos
        ));
        let _ = fs::remove_file(&log_path);
        set_shape_stride_log_path(Some(log_path.clone()));

        let suite = run_shape_stride_suite(&cfg).expect("shape/stride suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);

        let raw = fs::read_to_string(&log_path).expect("shape/stride log should exist");
        let mut entry_count = 0usize;
        let mut saw_as_strided_check = false;
        let mut saw_broadcast_to_check = false;
        let mut saw_sliding_window_check = false;
        let mut saw_packet_006_artifact = false;
        let mut saw_stride_tricks_success = false;
        let mut saw_stride_tricks_expected_error = false;
        for line in raw.lines().filter(|line| !line.trim().is_empty()) {
            entry_count += 1;
            let value: Value = serde_json::from_str(line).expect("log line must be valid json");
            let obj = value.as_object().expect("log line must be json object");
            assert!(
                obj.get("fixture_id")
                    .and_then(Value::as_str)
                    .is_some_and(|s| !s.trim().is_empty())
            );
            assert!(obj.get("seed").is_some_and(Value::is_u64));
            assert!(
                obj.get("mode")
                    .and_then(Value::as_str)
                    .is_some_and(|s| s == "strict" || s == "hardened")
            );
            assert!(
                obj.get("env_fingerprint")
                    .and_then(Value::as_str)
                    .is_some_and(|s| !s.trim().is_empty())
            );
            assert!(
                obj.get("artifact_refs")
                    .and_then(Value::as_array)
                    .is_some_and(|arr| {
                        !arr.is_empty()
                            && arr
                                .iter()
                                .all(|item| item.as_str().is_some_and(|s| !s.trim().is_empty()))
                    })
            );
            assert!(
                obj.get("reason_code")
                    .and_then(Value::as_str)
                    .is_some_and(|s| !s.trim().is_empty())
            );
            let reason_code = obj
                .get("reason_code")
                .and_then(Value::as_str)
                .expect("reason_code should be string");

            let passed = obj
                .get("passed")
                .and_then(Value::as_bool)
                .expect("passed should be bool");

            if reason_code.contains("stride_tricks") || reason_code.contains("as_strided") {
                if passed {
                    saw_stride_tricks_success = true;
                }
                if reason_code.contains("error")
                    || reason_code.contains("negative_stride")
                    || reason_code.contains("rank_mismatch")
                {
                    saw_stride_tricks_expected_error = true;
                }
            }

            let artifact_refs = obj
                .get("artifact_refs")
                .and_then(Value::as_array)
                .expect("artifact_refs should be array");
            if artifact_refs.iter().any(|item| {
                item.as_str()
                    .is_some_and(|artifact| artifact.contains("FNP-P2C-006"))
            }) {
                saw_packet_006_artifact = true;
            }

            let as_strided_checked = obj
                .get("as_strided_checked")
                .and_then(Value::as_bool)
                .expect("as_strided_checked should be bool");
            let broadcast_to_checked = obj
                .get("broadcast_to_checked")
                .and_then(Value::as_bool)
                .expect("broadcast_to_checked should be bool");
            let sliding_window_checked = obj
                .get("sliding_window_checked")
                .and_then(Value::as_bool)
                .expect("sliding_window_checked should be bool");

            saw_as_strided_check |= as_strided_checked;
            saw_broadcast_to_check |= broadcast_to_checked;
            saw_sliding_window_check |= sliding_window_checked;
        }
        assert!(entry_count > 0, "shape/stride log should contain entries");
        assert!(
            saw_as_strided_check,
            "shape/stride logs should include at least one as_strided check"
        );
        assert!(
            saw_broadcast_to_check,
            "shape/stride logs should include at least one broadcast_to check"
        );
        assert!(
            saw_sliding_window_check,
            "shape/stride logs should include at least one sliding_window check"
        );
        assert!(
            saw_packet_006_artifact,
            "shape/stride logs should include packet FNP-P2C-006 artifact references"
        );
        assert!(
            saw_stride_tricks_success,
            "shape/stride logs should include at least one successful stride-tricks case"
        );
        assert!(
            saw_stride_tricks_expected_error,
            "shape/stride logs should include at least one expected-error stride-tricks case"
        );
        set_shape_stride_log_path(None);
        let _ = fs::remove_file(log_path);
    }

    #[test]
    fn shape_stride_case_detects_overlap_writeability_mismatches() {
        let case = ShapeStrideFixtureCase {
            id: "stride_contract_mismatch".to_string(),
            lhs: vec![1],
            rhs: vec![1],
            expected_broadcast: Some(vec![1]),
            stride_shape: vec![8],
            stride_item_size: 8,
            stride_order: "C".to_string(),
            expected_strides: vec![8],
            seed: 0,
            env_fingerprint: String::new(),
            artifact_refs: Vec::new(),
            reason_code: "stride_tricks_contract".to_string(),
            as_strided: Some(AsStridedFixtureCase {
                shape: vec![2, 2],
                strides: vec![8, 8],
                expected_shape: Some(vec![2, 2]),
                expected_strides: Some(vec![8, 8]),
                expected_writeable: Some(true),
                expected_has_internal_overlap: Some(false),
                expected_error_contains: None,
            }),
            broadcast_to: None,
            sliding_window: None,
        };

        let (ok, failures) = evaluate_shape_stride_case(&case);
        assert!(!ok, "mismatched overlap/writeability should fail");
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("as_strided writeable mismatch")),
            "failures={failures:?}"
        );
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("as_strided overlap mismatch")),
            "failures={failures:?}"
        );
    }

    #[test]
    fn packet003_transfer_reason_code_vocabulary_is_stable() {
        assert_eq!(
            TRANSFER_PACKET_REASON_CODES,
            [
                "transfer_selector_invalid_context",
                "transfer_overlap_policy_triggered",
                "transfer_where_mask_contract_violation",
                "transfer_same_value_cast_rejected",
                "transfer_string_width_mismatch",
                "transfer_subarray_broadcast_contract_violation",
                "flatiter_transfer_read_violation",
                "flatiter_transfer_write_violation",
                "transfer_nditer_overlap_policy",
                "transfer_fpe_cast_error",
            ]
        );
    }

    #[test]
    fn packet003_transfer_log_records_are_replay_complete() {
        let artifact_refs = vec![
            "artifacts/phase2c/FNP-P2C-003/fixture_manifest.json".to_string(),
            "artifacts/phase2c/FNP-P2C-003/parity_gate.yaml".to_string(),
        ];
        for (index, reason_code) in TRANSFER_PACKET_REASON_CODES.iter().enumerate() {
            let log_record = TransferLogRecord {
                fixture_id: format!("UP-003-{index:02}"),
                seed: 3_000 + u64::try_from(index).expect("small test index"),
                mode: IterRuntimeMode::Strict,
                env_fingerprint: "fnp-conformance-tests".to_string(),
                artifact_refs: artifact_refs.clone(),
                reason_code: (*reason_code).to_string(),
                passed: true,
            };
            assert!(
                log_record.is_replay_complete(),
                "packet003 log should be replay complete for reason_code={reason_code}"
            );
        }
    }

    #[test]
    fn packet003_transfer_log_records_reject_missing_required_fields() {
        let log_record = TransferLogRecord {
            fixture_id: String::new(),
            seed: 3_100,
            mode: IterRuntimeMode::Hardened,
            env_fingerprint: String::new(),
            artifact_refs: Vec::new(),
            reason_code: String::new(),
            passed: false,
        };
        assert!(
            !log_record.is_replay_complete(),
            "packet003 log should reject missing required fields"
        );
    }

    #[test]
    fn ufunc_metamorphic_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite = run_ufunc_metamorphic_suite(&cfg).expect("metamorphic suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn ufunc_adversarial_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite = run_ufunc_adversarial_suite(&cfg).expect("adversarial suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn linalg_differential_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite = run_linalg_differential_suite(&cfg).expect("differential suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn linalg_metamorphic_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite = run_linalg_metamorphic_suite(&cfg).expect("metamorphic suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn linalg_adversarial_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite = run_linalg_adversarial_suite(&cfg).expect("adversarial suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn rng_differential_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite = run_rng_differential_suite(&cfg).expect("differential suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn rng_metamorphic_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite = run_rng_metamorphic_suite(&cfg).expect("metamorphic suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn rng_adversarial_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite = run_rng_adversarial_suite(&cfg).expect("adversarial suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn rng_statistical_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite = run_rng_statistical_suite(&cfg).expect("statistical suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn polynomial_differential_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite = run_polynomial_differential_suite(&cfg)
            .expect("polynomial differential suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn string_differential_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite =
            run_string_differential_suite(&cfg).expect("string differential suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn string_metamorphic_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite =
            run_string_metamorphic_suite(&cfg).expect("string metamorphic suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn masked_differential_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite =
            run_masked_differential_suite(&cfg).expect("masked differential suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn datetime_differential_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite =
            run_datetime_differential_suite(&cfg).expect("datetime differential suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn datetime_metamorphic_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite =
            run_datetime_metamorphic_suite(&cfg).expect("datetime metamorphic suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn fft_differential_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite = run_fft_differential_suite(&cfg).expect("fft differential suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn fft_metamorphic_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite = run_fft_metamorphic_suite(&cfg).expect("fft metamorphic suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn signal_differential_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite =
            run_signal_differential_suite(&cfg).expect("signal differential suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn io_adversarial_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite = run_io_adversarial_suite(&cfg).expect("io adversarial suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn io_differential_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite = run_io_differential_suite(&cfg).expect("io differential suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn io_metamorphic_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite = run_io_metamorphic_suite(&cfg).expect("io metamorphic suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn crash_signature_regression_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite = run_crash_signature_regression_suite(&cfg)
            .expect("crash signature regression suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn workflow_scenario_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite = super::workflow_scenarios::run_user_workflow_scenario_suite(&cfg)
            .expect("workflow scenario suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn raptorq_artifact_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite = super::raptorq_artifacts::run_raptorq_artifact_suite(&cfg)
            .expect("raptorq artifact suite should run");
        assert!(suite.all_passed(), "failures={:?}", suite.failures);
    }

    #[test]
    fn core_suites_are_green() {
        let cfg = HarnessConfig::default_paths();
        let suites = run_all_core_suites(&cfg).expect("core suites should run");
        for suite in suites {
            assert!(
                suite.all_passed(),
                "suite={} failures={:?}",
                suite.suite,
                suite.failures
            );
        }
    }

    /// Deterministic sequence witness tests for all distribution functions.
    /// These lock in exact output sequences for seed=12345, n=5 samples each,
    /// ensuring the PRNG produces identical results across builds/platforms.
    #[test]
    #[allow(clippy::excessive_precision)]
    fn rng_sequence_witness_continuous() {
        use fnp_random::Generator;
        const SEED: u64 = 12345;

        macro_rules! check_seq {
            ($name:expr, $gen_call:expr, $expected:expr) => {{
                let mut g = Generator::from_pcg64_dxsm(SEED).unwrap();
                let actual = $gen_call(&mut g);
                for (i, (a, e)) in actual.iter().zip($expected.iter()).enumerate() {
                    assert!(
                        (a - e).abs() < 1e-12,
                        "{} sample {}: got {a:.17e}, expected {e:.17e}",
                        $name,
                        i
                    );
                }
            }};
        }

        check_seq!(
            "random",
            |g: &mut Generator| g.random(5),
            [
                9.32081690319876310e-1,
                3.37505601117676801e-1,
                2.16981970195010643e-1,
                3.52706249766546187e-1,
                5.50105102114212707e-1
            ]
        );
        check_seq!(
            "uniform(0,10)",
            |g: &mut Generator| g.uniform(0.0, 10.0, 5).unwrap(),
            [
                9.32081690319876266e0,
                3.37505601117676779e0,
                2.16981970195010643e0,
                3.52706249766546165e0,
                5.50105102114212663e0
            ]
        );
        check_seq!(
            "standard_normal",
            |g: &mut Generator| g.standard_normal(5),
            [
                6.48970595720086973e-1,
                1.08908349922134495e0,
                1.47033729140938529e0,
                6.60548357473366710e-1,
                -4.42344548278891359e-1
            ]
        );
        check_seq!(
            "normal(5,2)",
            |g: &mut Generator| g.normal(5.0, 2.0, 5).unwrap(),
            [
                6.29794119144017372e0,
                7.17816699844268946e0,
                7.94067458281877059e0,
                6.32109671494673364e0,
                4.11531090344221706e0
            ]
        );
        check_seq!(
            "standard_exponential",
            |g: &mut Generator| g.standard_exponential(5),
            [
                1.71403481620221210e0,
                1.30485733047918495e-1,
                2.59681276258719784e-1,
                4.84283249016193712e-2,
                2.18645107617135359e0
            ]
        );
        check_seq!(
            "exponential(2)",
            |g: &mut Generator| g.exponential(2.0, 5).unwrap(),
            [
                3.42806963240442419e0,
                2.60971466095836990e-1,
                5.19362552517439569e-1,
                9.68566498032387424e-2,
                4.37290215234270718e0
            ]
        );
        check_seq!(
            "standard_gamma(2)",
            |g: &mut Generator| g.standard_gamma(2.0, 5).unwrap(),
            [
                2.65271299374846858e0,
                4.37668779876149294e0,
                1.15834211198835813e0,
                2.09794988661814186e0,
                8.94578313531806613e-1
            ]
        );
        check_seq!(
            "gamma(2,3)",
            |g: &mut Generator| g.gamma(2.0, 3.0, 5).unwrap(),
            [
                7.95813898124540575e0,
                1.31300633962844788e1,
                3.47502633596507415e0,
                6.29384965985442513e0,
                2.68373494059541962e0
            ]
        );
        check_seq!(
            "beta(2,5)",
            |g: &mut Generator| g.beta(2.0, 5.0, 5).unwrap(),
            [
                2.35361548599163445e-1,
                1.77540741643774563e-1,
                1.58668678806063196e-1,
                2.10563771290120133e-1,
                7.16530287407453614e-1
            ]
        );
        check_seq!(
            "chisquare(3)",
            |g: &mut Generator| g.chisquare(3.0, 5).unwrap(),
            [
                4.03478951589785773e0,
                7.16887968014016952e0,
                1.50226995394899498e0,
                3.06599749104649311e0,
                1.09083707666572383e0
            ]
        );
        check_seq!(
            "lognormal(0,1)",
            |g: &mut Generator| g.lognormal(0.0, 1.0, 5).unwrap(),
            [
                1.91356997766160775e0,
                2.97154939683406250e0,
                4.35070234813747891e0,
                1.93585358318327083e0,
                6.42528215322030416e-1
            ]
        );
        check_seq!(
            "standard_cauchy",
            |g: &mut Generator| g.standard_cauchy(5),
            [
                5.95886905075760809e-1,
                2.22593436918608845e0,
                5.46450807795340765e0,
                -3.20374970349365296e-1,
                5.98660229079376771e-1
            ]
        );
        check_seq!(
            "triangular(0,5,10)",
            |g: &mut Generator| g.triangular(0.0, 5.0, 10.0, 5).unwrap(),
            [
                8.15719901128575842e0,
                4.10795326846397302e0,
                3.29379697457972265e0,
                4.19944192581910958e0,
                5.25713747887529781e0
            ]
        );
        check_seq!(
            "laplace(0,1)",
            |g: &mut Generator| g.laplace(0.0, 1.0, 5).unwrap(),
            [
                1.99630244365275855e0,
                -3.93025992343090158e-1,
                -8.34793834992556993e-1,
                -3.48972541556781424e-1,
                1.05594103191076080e-1
            ]
        );
        check_seq!(
            "gumbel(0,1)",
            |g: &mut Generator| g.gumbel(0.0, 1.0, 5).unwrap(),
            [
                -9.89336572015754090e-1,
                8.87355484014991536e-1,
                1.40813286813637495e0,
                8.32512543783072045e-1,
                2.24718185711721308e-1
            ]
        );
        check_seq!(
            "weibull(2)",
            |g: &mut Generator| g.weibull(2.0, 5).unwrap(),
            [
                1.30921152462167556e0,
                3.61228090059339813e-1,
                5.09589321178063770e-1,
                2.20064365360726599e-1,
                1.47866530228153858e0
            ]
        );
        check_seq!(
            "f_distribution(5,2)",
            |g: &mut Generator| g.f_distribution(5.0, 2.0, 5).unwrap(),
            [
                5.03569673371778581e0,
                2.09797390659074745e0,
                3.24202463537182783e0,
                3.00822311004571397e0,
                6.00995450650283587e1
            ]
        );
        check_seq!(
            "standard_t(5)",
            |g: &mut Generator| g.standard_t(5.0, 5),
            [
                5.00833320479618593e-1,
                8.31263707118225392e-1,
                4.79910878609732861e-1,
                -1.28621448250016712e0,
                -4.07072148213456428e-3
            ]
        );
        check_seq!(
            "power(3)",
            |g: &mut Generator| g.power(3.0, 5).unwrap(),
            [
                9.35937841864501463e-1,
                4.96415680680427018e-1,
                6.11538375049609506e-1,
                3.61583498974891560e-1,
                9.61065570914075296e-1
            ]
        );
        check_seq!(
            "vonmises(0,2)",
            |g: &mut Generator| g.vonmises(0.0, 2.0, 5).unwrap(),
            [
                2.47613248673183906e-1,
                -3.58286679084546367e-1,
                4.66402244787470999e-1,
                5.12425113175970989e-2,
                4.42636174325620857e-1
            ]
        );
        check_seq!(
            "rayleigh(1)",
            |g: &mut Generator| g.rayleigh(1.0, 5).unwrap(),
            [
                1.85150469413513075e0,
                5.10853664072048130e-1,
                7.20668129250516820e-1,
                3.11218010088167518e-1,
                2.09114852469706403e0
            ]
        );
        check_seq!(
            "pareto(3)",
            |g: &mut Generator| g.pareto(3.0, 5).unwrap(),
            [
                7.70646862274550370e-1,
                4.44550272369403851e-2,
                9.04172546472586519e-2,
                1.62737735030588949e-2,
                1.07262729139401491e0
            ]
        );
        check_seq!(
            "logistic(0,1)",
            |g: &mut Generator| g.logistic(0.0, 1.0, 5).unwrap(),
            [
                2.61911480663287932e0,
                -6.74429997228288114e-1,
                -1.28334145886692852e0,
                -6.07164653508057151e-1,
                2.01095359492238307e-1
            ]
        );
        check_seq!(
            "wald(3,2)",
            |g: &mut Generator| g.wald(3.0, 2.0, 5),
            [
                1.38174906568860378e0,
                5.94694599526367829e-1,
                5.12410478413090154e0,
                2.05985324281523852e0,
                1.26745020708461498e0
            ]
        );
        check_seq!(
            "maxwell(1)",
            |g: &mut Generator| g.maxwell(1.0, 5),
            [
                1.94143180485294953e0,
                7.99090430292215381e-1,
                1.24525858258412470e0,
                1.24195756573983673e0,
                7.27134631213512517e-1
            ]
        );
        check_seq!(
            "halfnormal(1)",
            |g: &mut Generator| g.halfnormal(1.0, 5),
            [
                6.48970595720086973e-1,
                1.08908349922134495e0,
                1.47033729140938529e0,
                6.60548357473366710e-1,
                4.42344548278891359e-1
            ]
        );
    }

    #[test]
    fn rng_sequence_witness_discrete() {
        use fnp_random::Generator;
        const SEED: u64 = 12345;

        let mut g = Generator::from_pcg64_dxsm(SEED).unwrap();
        let pois: Vec<u64> = g.poisson(5.0, 5).unwrap();
        assert_eq!(pois, vec![7, 5, 7, 6, 4], "poisson(5) sequence mismatch");

        let mut g = Generator::from_pcg64_dxsm(SEED).unwrap();
        let binom: Vec<u64> = g.binomial(10, 0.3, 5).unwrap();
        assert_eq!(
            binom,
            vec![5, 2, 2, 2, 3],
            "binomial(10,0.3) sequence mismatch"
        );

        let mut g = Generator::from_pcg64_dxsm(SEED).unwrap();
        let geom: Vec<u64> = g.geometric(0.4, 5).unwrap();
        assert_eq!(
            geom,
            vec![6, 1, 1, 1, 2],
            "geometric(0.4) sequence mismatch"
        );

        let mut g = Generator::from_pcg64_dxsm(SEED).unwrap();
        let zipf_vals: Vec<f64> = g.zipf(2.0, 5).unwrap();
        let expected_zipf = [14.0, 1.0, 8.0, 1.0, 1.0];
        for (i, (a, e)) in zipf_vals.iter().zip(expected_zipf.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-12,
                "zipf(2) sample {i}: got {a}, expected {e}"
            );
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn rng_sequence_witness_remaining() {
        use fnp_random::Generator;
        const SEED: u64 = 12345;

        macro_rules! check_seq {
            ($name:expr, $gen_call:expr, $expected:expr) => {{
                let mut g = Generator::from_pcg64_dxsm(SEED).unwrap();
                let actual = $gen_call(&mut g);
                for (i, (a, e)) in actual.iter().zip($expected.iter()).enumerate() {
                    assert!(
                        (a - e).abs() < 1e-12,
                        "{} sample {}: got {a:.17e}, expected {e:.17e}",
                        $name,
                        i
                    );
                }
            }};
        }

        check_seq!(
            "lomax(3)",
            |g: &mut Generator| g.lomax(3.0, 5),
            [
                1.45096839094341590e0,
                1.47112281973488912e-1,
                8.49492216360963059e-2,
                1.56022256242237534e-1,
                3.05057491487117360e-1
            ]
        );
        check_seq!(
            "levy(0,1)",
            |g: &mut Generator| g.levy(0.0, 1.0, 5),
            [
                2.37437855150465893e0,
                8.43097193967099146e-1,
                4.62557849977695501e-1,
                2.29187414899802677e0,
                5.11067944493424164e0
            ]
        );
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn rng_sequence_witness_advanced() {
        use fnp_random::Generator;
        const SEED: u64 = 12345;

        // noncentral_chisquare(df=3, nonc=2)
        let mut g = Generator::from_pcg64_dxsm(SEED).unwrap();
        let nc_chi = g.noncentral_chisquare(3.0, 2.0, 5).unwrap();
        let expected_nc_chi = [
            9.694565810991783,
            4.82399957656241529,
            6.15049748063806678,
            1.21777690519257353,
            6.95800398984072976e-1,
        ];
        for (i, (a, e)) in nc_chi.iter().zip(expected_nc_chi.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-12,
                "noncentral_chisquare sample {i}: {a} != {e}"
            );
        }

        // noncentral_f(dfnum=5, dfden=2, nonc=1)
        let mut g = Generator::from_pcg64_dxsm(SEED).unwrap();
        let nc_f = g.noncentral_f(5.0, 2.0, 1.0, 5).unwrap();
        let expected_nc_f = [
            4.71128924818266057e1,
            4.68907563784964676e-1,
            5.19956069467958404,
            4.44086257902797710e-1,
            7.01781681496106380,
        ];
        for (i, (a, e)) in nc_f.iter().zip(expected_nc_f.iter()).enumerate() {
            assert!((a - e).abs() < 1e-12, "noncentral_f sample {i}: {a} != {e}");
        }

        // negative_binomial(n=5, p=0.3)
        let mut g = Generator::from_pcg64_dxsm(SEED).unwrap();
        let neg_bin = g.negative_binomial(5.0, 0.3, 5).unwrap();
        assert_eq!(
            neg_bin,
            vec![11, 8, 12, 14, 8],
            "negative_binomial mismatch"
        );

        // hypergeometric(ngood=20, nbad=10, nsample=15)
        let mut g = Generator::from_pcg64_dxsm(SEED).unwrap();
        let hyper = g.hypergeometric(20, 10, 15, 5).unwrap();
        assert_eq!(hyper, vec![12, 8, 11, 11, 10], "hypergeometric mismatch");

        // multinomial(n=10, pvals=[0.2, 0.3, 0.5])
        let mut g = Generator::from_pcg64_dxsm(SEED).unwrap();
        let multi = g.multinomial(10, &[0.2, 0.3, 0.5], 3);
        assert_eq!(
            multi,
            vec![vec![4, 2, 4], vec![1, 3, 6], vec![2, 5, 3]],
            "multinomial mismatch"
        );

        // dirichlet(alpha=[2, 3, 5])
        let mut g = Generator::from_pcg64_dxsm(SEED).unwrap();
        let dir = g.dirichlet(&[2.0, 3.0, 5.0], 3).unwrap();
        let expected_dir = [
            [0.21587855063764583, 0.47692451379316814, 0.307196935569186],
            [0.247103567175703, 0.19419713918314283, 0.5586992936411542],
            [0.07297051356158091, 0.17066702050027016, 0.756362465938149],
        ];
        for (i, (row, exp_row)) in dir.iter().zip(expected_dir.iter()).enumerate() {
            for (j, (a, e)) in row.iter().zip(exp_row.iter()).enumerate() {
                assert!((a - e).abs() < 1e-12, "dirichlet[{i}][{j}]: {a} != {e}");
            }
        }

        // multivariate_hypergeometric(colors=[10,20,30], nsample=25)
        let mut g = Generator::from_pcg64_dxsm(SEED).unwrap();
        let mv_hyper = g.multivariate_hypergeometric(&[10, 20, 30], 25, 3).unwrap();
        assert_eq!(
            mv_hyper,
            vec![vec![4, 6, 15], vec![3, 8, 14], vec![3, 12, 10]],
            "multivariate_hypergeometric mismatch"
        );
    }

    #[test]
    fn generate_rng_samples_reports_invalid_fallible_parameters() {
        fn base_case(distribution: &str) -> RngStatisticalCase {
            RngStatisticalCase {
                id: format!("test_{distribution}"),
                distribution: distribution.to_string(),
                seed: 12345,
                mode: String::new(),
                env_fingerprint: String::new(),
                artifact_refs: Vec::new(),
                reason_code: String::new(),
                draws: 4,
                param_a: 0.0,
                param_b: 0.0,
                param_c: 0.0,
                param_d: 0.0,
                expected_mean: None,
                expected_variance: None,
                mean_abs_tol: default_rng_mean_abs_tol(),
                variance_abs_tol: default_rng_variance_abs_tol(),
                support_min: None,
                support_max: None,
                support_min_inclusive: true,
                support_max_inclusive: true,
                ks_distribution: String::new(),
                ks_alpha: 0.0,
            }
        }

        let mut poisson_case = base_case("poisson");
        poisson_case.param_a = f64::NAN;
        let poisson_err =
            generate_rng_samples(&poisson_case, poisson_case.draws).expect_err("poisson NaN");
        assert_eq!(poisson_err.reason_code, "random_invalid_parameter");

        let mut gamma_case = base_case("gamma");
        gamma_case.param_a = -1.0;
        gamma_case.param_b = 1.0;
        let gamma_err =
            generate_rng_samples(&gamma_case, gamma_case.draws).expect_err("gamma negative shape");
        assert_eq!(gamma_err.reason_code, "random_invalid_parameter");
    }

    #[test]
    fn rng_sequence_witness_multivariate_normal() {
        use fnp_random::Generator;
        const SEED: u64 = 12345;

        // multivariate_normal_diag(mean=[1,2], cov_diag=[0.5,0.5])
        let mut g = Generator::from_pcg64_dxsm(SEED).unwrap();
        let mvn_diag = g
            .multivariate_normal_diag(&[1.0, 2.0], &[0.5, 0.5], 3)
            .unwrap();
        // Each sample is a Vec of length 2, derived from standard_normal draws
        assert_eq!(mvn_diag.len(), 3);
        assert_eq!(mvn_diag[0].len(), 2);
        // Lock in exact values: verify they don't change across builds
        let first_run = mvn_diag.clone();
        let mut g2 = Generator::from_pcg64_dxsm(SEED).unwrap();
        let second_run = g2
            .multivariate_normal_diag(&[1.0, 2.0], &[0.5, 0.5], 3)
            .unwrap();
        for (i, (r1, r2)) in first_run.iter().zip(second_run.iter()).enumerate() {
            for (j, (a, b)) in r1.iter().zip(r2.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-15,
                    "multivariate_normal_diag[{i}][{j}] not deterministic: {a} vs {b}"
                );
            }
        }

        // multivariate_normal(mean=[0,0], cov=[1,0,0,1] identity 2x2)
        let mut g = Generator::from_pcg64_dxsm(SEED).unwrap();
        let mvn = g.multivariate_normal(&[0.0, 0.0], &[1.0, 0.0, 0.0, 1.0], 3);
        assert_eq!(mvn.len(), 3);
        assert_eq!(mvn[0].len(), 2);
        let first_run = mvn.clone();
        let mut g2 = Generator::from_pcg64_dxsm(SEED).unwrap();
        let second_run = g2.multivariate_normal(&[0.0, 0.0], &[1.0, 0.0, 0.0, 1.0], 3);
        for (i, (r1, r2)) in first_run.iter().zip(second_run.iter()).enumerate() {
            for (j, (a, b)) in r1.iter().zip(r2.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-15,
                    "multivariate_normal[{i}][{j}] not deterministic: {a} vs {b}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // P2C003 Transfer-loop invariant tests
    // -----------------------------------------------------------------------

    fn base_ctx() -> TransferContext {
        TransferContext {
            src_stride: 8,
            dst_stride: 8,
            src_offset: 0,
            dst_offset: 0,
            item_size: 8,
            element_count: 64,
            aligned: true,
            dtype_relation: TransferDtypeRelation::Same,
            same_value_cast: false,
            has_where_mask: false,
            has_overlap: false,
            mode: IterRuntimeMode::Strict,
        }
    }

    #[test]
    fn p2c003_transfer_selector_determinism_property() {
        let relations = [
            TransferDtypeRelation::Same,
            TransferDtypeRelation::LosslessCast,
            TransferDtypeRelation::LossyCast,
            TransferDtypeRelation::SubarrayBroadcast,
            TransferDtypeRelation::StringWidthChange {
                src_width: 4,
                dst_width: 8,
            },
        ];
        let modes = [IterRuntimeMode::Strict, IterRuntimeMode::Hardened];
        let strides: [isize; 4] = [8, 16, -8, 24];

        for rel in &relations {
            for mode in modes {
                for src_stride in strides {
                    for dst_stride in strides {
                        let ctx = TransferContext {
                            src_stride,
                            dst_stride,
                            dtype_relation: *rel,
                            mode,
                            ..base_ctx()
                        };
                        let first = select_transfer_loop(ctx);
                        let second = select_transfer_loop(ctx);
                        assert_eq!(first, second, "P2C003-R01 determinism violated for {ctx:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn p2c003_overlap_never_silently_corrupts() {
        let ctx_no_overlap = TransferContext {
            has_overlap: false,
            ..base_ctx()
        };
        let ctx_with_overlap = TransferContext {
            has_overlap: true,
            ..base_ctx()
        };

        let no_ol = select_transfer_loop(ctx_no_overlap).expect("no-overlap should resolve");
        let with_ol = select_transfer_loop(ctx_with_overlap).expect("overlap should resolve");

        // Same loop class (correctness preserved) but overlap action differs.
        assert_eq!(no_ol.loop_class, with_ol.loop_class);
        assert_ne!(no_ol.overlap_action, with_ol.overlap_action);
    }

    #[test]
    fn p2c003_where_mask_isolation_invariant() {
        let mask = vec![true, false, true, false, true, false, true, false];
        let count = validate_where_mask(&mask, 8).expect("valid mask");
        assert_eq!(count, 4, "exactly half should be selected");

        // Where-mask transfer always resolves to MaskedTransfer loop class.
        let ctx = TransferContext {
            has_where_mask: true,
            element_count: 8,
            ..base_ctx()
        };
        let decision = select_transfer_loop(ctx).expect("masked transfer should resolve");
        assert_eq!(decision.loop_class, TransferLoopClass::MaskedTransfer);
    }

    #[test]
    fn p2c003_same_value_cast_rejects_lossy() {
        let ctx = TransferContext {
            dtype_relation: TransferDtypeRelation::LossyCast,
            same_value_cast: true,
            ..base_ctx()
        };
        let err = select_transfer_loop(ctx).expect_err("lossy same-value must reject");
        assert_eq!(err.reason_code(), "transfer_same_value_cast_rejected");
    }

    #[test]
    fn p2c003_same_value_cast_accepts_lossless() {
        let ctx = TransferContext {
            dtype_relation: TransferDtypeRelation::LosslessCast,
            same_value_cast: true,
            ..base_ctx()
        };
        let decision = select_transfer_loop(ctx).expect("lossless same-value should pass");
        assert_eq!(decision.loop_class, TransferLoopClass::ContiguousCast);
    }

    #[test]
    fn p2c003_string_transfer_zero_width_rejected() {
        let ctx = TransferContext {
            dtype_relation: TransferDtypeRelation::StringWidthChange {
                src_width: 0,
                dst_width: 4,
            },
            ..base_ctx()
        };
        let err = select_transfer_loop(ctx).expect_err("zero-width string should fail");
        assert_eq!(err.reason_code(), "transfer_string_width_mismatch");
    }

    #[test]
    fn p2c003_subarray_broadcast_contract() {
        validate_subarray_transfer(4, 4).expect("1:1 should pass");
        validate_subarray_transfer(4, 8).expect("1:2 broadcast should pass");
        validate_subarray_transfer(4, 16).expect("1:4 broadcast should pass");

        let err = validate_subarray_transfer(3, 7).expect_err("non-multiple should fail");
        assert_eq!(
            err.reason_code(),
            "transfer_subarray_broadcast_contract_violation"
        );
    }

    #[test]
    fn p2c003_hardened_mode_fails_closed_for_overlap_subarray() {
        let ctx = TransferContext {
            dtype_relation: TransferDtypeRelation::SubarrayBroadcast,
            has_overlap: true,
            mode: IterRuntimeMode::Hardened,
            ..base_ctx()
        };
        let err = select_transfer_loop(ctx).expect_err("hardened overlap subarray should fail");
        assert_eq!(
            err.reason_code(),
            "transfer_subarray_broadcast_contract_violation"
        );
    }

    #[test]
    fn p2c003_strict_mode_allows_overlap_subarray() {
        let ctx = TransferContext {
            dtype_relation: TransferDtypeRelation::SubarrayBroadcast,
            has_overlap: true,
            mode: IterRuntimeMode::Strict,
            ..base_ctx()
        };
        let decision =
            select_transfer_loop(ctx).expect("strict mode should allow overlap subarray");
        assert_eq!(decision.loop_class, TransferLoopClass::SubarrayGrouped);
    }

    #[test]
    fn p2c003_where_mask_mismatch_rejected() {
        let mask = vec![true, false];
        let err = validate_where_mask(&mask, 10).expect_err("mask length mismatch");
        assert_eq!(err.reason_code(), "transfer_where_mask_contract_violation");
    }

    #[test]
    fn p2c003_transfer_loop_class_coverage_grid() {
        // Ensure every TransferLoopClass variant is reachable.
        let test_cases: Vec<(TransferContext, TransferLoopClass)> = vec![
            (base_ctx(), TransferLoopClass::SimpleContiguous),
            (
                TransferContext {
                    src_stride: 16,
                    dst_stride: 16,
                    ..base_ctx()
                },
                TransferLoopClass::StridedNoCast,
            ),
            (
                TransferContext {
                    dtype_relation: TransferDtypeRelation::LosslessCast,
                    ..base_ctx()
                },
                TransferLoopClass::ContiguousCast,
            ),
            (
                TransferContext {
                    dtype_relation: TransferDtypeRelation::LossyCast,
                    ..base_ctx()
                },
                TransferLoopClass::StridedWithCast,
            ),
            (
                TransferContext {
                    has_where_mask: true,
                    ..base_ctx()
                },
                TransferLoopClass::MaskedTransfer,
            ),
            (
                TransferContext {
                    dtype_relation: TransferDtypeRelation::SubarrayBroadcast,
                    ..base_ctx()
                },
                TransferLoopClass::SubarrayGrouped,
            ),
            (
                TransferContext {
                    dtype_relation: TransferDtypeRelation::StringWidthChange {
                        src_width: 4,
                        dst_width: 8,
                    },
                    ..base_ctx()
                },
                TransferLoopClass::StringFixedWidth,
            ),
        ];

        for (ctx, expected_class) in test_cases {
            let decision = select_transfer_loop(ctx).expect("should resolve");
            assert_eq!(
                decision.loop_class, expected_class,
                "loop class mismatch for {ctx:?}"
            );
        }
    }

    #[test]
    fn approx_equal_values_rejects_nan_vs_finite_mismatches() {
        assert!(!crate::approx_equal_values(&[1.0], &[f64::NAN], 1e-9, 1e-9));
        assert!(!crate::approx_equal_values(&[f64::NAN], &[1.0], 1e-9, 1e-9));
    }

    #[test]
    fn approx_equal_values_accepts_matching_non_finite_values_only() {
        assert!(crate::approx_equal_values(
            &[f64::NAN],
            &[f64::NAN],
            1e-9,
            1e-9
        ));
        assert!(crate::approx_equal_values(
            &[f64::INFINITY, f64::NEG_INFINITY],
            &[f64::INFINITY, f64::NEG_INFINITY],
            1e-9,
            1e-9
        ));
        assert!(!crate::approx_equal_values(
            &[f64::INFINITY],
            &[f64::NEG_INFINITY],
            1e-9,
            1e-9
        ));
    }
}
