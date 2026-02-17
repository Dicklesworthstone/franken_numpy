#![forbid(unsafe_code)]

pub mod benchmark;
pub mod contract_schema;
pub mod raptorq_artifacts;
pub mod security_contracts;
pub mod test_contracts;
pub mod ufunc_differential;
pub mod workflow_scenarios;

use crate::ufunc_differential::{UFuncInputCase, UFuncOperation};
use fnp_dtype::{DType, promote};
use fnp_io::{
    IOError, IOSupportedDType, LoadDispatch, MemmapMode, classify_load_dispatch,
    validate_descriptor_roundtrip, validate_header_schema, validate_io_policy_metadata,
    validate_magic_version, validate_memmap_contract, validate_npz_archive_budget,
    validate_read_payload,
};
use fnp_iter::{
    FlatIterIndex, NditerTransferFlags, OverlapAction, TransferClass, TransferError,
    TransferSelectorInput, overlap_copy_policy, select_transfer_class, validate_flatiter_read,
    validate_flatiter_write, validate_nditer_flags,
};
use fnp_linalg::{
    LinAlgError, QrMode, lstsq_output_shapes, qr_output_shapes, solve_2x2, svd_output_shapes,
    validate_backend_bridge, validate_policy_metadata as validate_linalg_policy_metadata,
    validate_spectral_branch, validate_tolerance_policy,
};
use fnp_ndarray::{MemoryOrder, NdLayout, broadcast_shape, contiguous_strides};
use fnp_random::{DeterministicRng, RandomError};
use fnp_runtime::{
    CompatibilityClass, DecisionAction, DecisionAuditContext, EvidenceLedger, RuntimeMode,
    decide_and_record_with_context, decide_compatibility_from_wire,
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
    axis: Option<usize>,
    keepdims: Option<bool>,
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
    len: Option<usize>,
    #[serde(default)]
    values_len: Option<usize>,
    #[serde(default)]
    expected_transfer_class: Option<String>,
    #[serde(default)]
    expected_overlap_action: Option<String>,
    #[serde(default)]
    expected_selected: Option<usize>,
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

static RUNTIME_POLICY_LOG_PATH: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
static SHAPE_STRIDE_LOG_PATH: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
static DTYPE_PROMOTION_LOG_PATH: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();

fn default_f64_dtype_name() -> String {
    "f64".to_string()
}

fn default_one_usize() -> usize {
    1
}

fn default_true() -> bool {
    true
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
            match base.sliding_window_view(sliding_window_case.window_shape.clone()) {
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
            let first = base.sliding_window_view(sliding_window_case.window_shape.clone());
            let second = base.sliding_window_view(sliding_window_case.window_shape.clone());
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
            match base.sliding_window_view(sliding_window_case.window_shape.clone()) {
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

fn execute_iter_operation(
    case_id: &str,
    operation: &str,
    selector: Option<&IterSelectorFixtureInput>,
    overlap: Option<&IterOverlapFixtureInput>,
    flags: Option<&IterFlagsFixtureInput>,
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
        _ => execute_iter_operation(case_id, operation, selector, overlap, flags, values_len),
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
                );
                let second = execute_iter_operation(
                    &case.id,
                    "select_transfer_class",
                    case.selector.as_ref(),
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
                );
                let second = execute_iter_operation(
                    &case.id,
                    "overlap_copy_policy",
                    None,
                    case.overlap.as_ref(),
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
                let len = flat_index
                    .mask
                    .len()
                    .max(flat_index.stop)
                    .max(flat_index.index.saturating_add(1));
                let Ok(IterOperationOutcome::SelectedCount(selected)) =
                    execute_iter_operation_with_len(
                        &case.id,
                        "validate_flatiter_read",
                        None,
                        None,
                        Some(flat_index),
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
                );
                let second = execute_iter_operation(
                    &case.id,
                    "validate_nditer_flags",
                    None,
                    None,
                    case.flags.as_ref(),
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
                    keepdims: None,
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
                    keepdims: None,
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
                    keepdims: None,
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
                    keepdims: None,
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
                    keepdims: Some(false),
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
                    keepdims: None,
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
                    keepdims: Some(false),
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
            keepdims: case.keepdims,
            seed: case.seed,
            mode: mode.clone(),
            env_fingerprint: env_fingerprint.clone(),
            artifact_refs: artifact_refs.clone(),
            reason_code: reason_code.clone(),
            expected_reason_code: expected_reason_code.clone(),
            expected_error_contains: case.expected_error_contains.clone(),
        };

        match ufunc_differential::execute_input_case(&input) {
            Ok((shape, _, _)) => {
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
            Err(err) => {
                let expected = case.expected_error_contains.to_lowercase();
                let actual = err.to_lowercase();
                let actual_reason_code = classify_ufunc_reason_code(case.op, &err);
                if actual.contains(&expected) && actual_reason_code == expected_reason_code {
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

#[derive(Debug)]
enum LinalgOperationOutcome {
    Unit,
    SolveVector(Vec<f64>),
    QrShapes {
        q_shape: Option<Vec<usize>>,
        r_shape: Vec<usize>,
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

#[derive(Debug, Clone, PartialEq, Eq)]
enum IoOperationOutcome {
    Unit,
    Count(usize),
    Dispatch(String),
    MagicVersion(String),
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
        run_rng_differential_suite(config)?,
        run_rng_metamorphic_suite(config)?,
        run_rng_adversarial_suite(config)?,
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
    } else if matches!(op, UFuncOperation::Sum)
        && (lowered.contains("axis") || lowered.contains("keepdims") || lowered.contains("reduce"))
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
        "svd_shapes" => {
            let output = svd_output_shapes(input.shape, input.full_matrices, input.converged)?;
            Ok(LinalgOperationOutcome::SvdShapes {
                u_shape: output.u_shape,
                s_shape: output.s_shape,
                vh_shape: output.vh_shape,
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
        ("qr_shapes", LinalgOperationOutcome::QrShapes { q_shape, r_shape }) => {
            if case.expected_q_shape != *q_shape || case.expected_r_shape != *r_shape {
                return Err(format!(
                    "qr_shapes mismatch expected_q_shape={:?} expected_r_shape={:?} actual_q_shape={q_shape:?} actual_r_shape={r_shape:?}",
                    case.expected_q_shape, case.expected_r_shape
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
            "spectral_branch" | "tolerance_policy" | "backend_bridge" | "policy_metadata",
            LinalgOperationOutcome::Unit,
        ) => Ok(()),
        (operation, actual) => Err(format!(
            "operation {operation} produced unexpected outcome {actual:?}"
        )),
    }
}

const DEFAULT_RNG_DRAWS: usize = 128;
const DEFAULT_RNG_REPLAY_DRAWS: usize = 32;
const DEFAULT_RNG_PREFIX_DRAWS: usize = 8;
const DEFAULT_RNG_JUMP_STEPS: u64 = 32;
const RNG_MAX_JUMP_OPS: u64 = 1024;
const RNG_MAX_STATE_SCHEMA_FIELDS: u64 = 4096;

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
            let mut restored = DeterministicRng::from_state(seed, counter);
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
            if case.steps > RNG_MAX_STATE_SCHEMA_FIELDS {
                Err(RngSuiteError::new(
                    "rng_state_schema_invalid",
                    "state schema entries exceeded bounded budget",
                ))
            } else {
                Ok(())
            }
        }
        other => Err(RngSuiteError::new(
            "rng_policy_unknown_metadata",
            format!("unsupported rng adversarial operation {other}"),
        )),
    }
}

fn map_random_error_to_rng_suite(error: RandomError) -> RngSuiteError {
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
        HarnessConfig, run_all_core_suites, run_crash_signature_regression_suite,
        run_dtype_promotion_suite, run_io_adversarial_suite, run_io_differential_suite,
        run_io_metamorphic_suite, run_iter_adversarial_suite, run_iter_differential_suite,
        run_iter_metamorphic_suite, run_linalg_adversarial_suite, run_linalg_differential_suite,
        run_linalg_metamorphic_suite, run_rng_adversarial_suite, run_rng_differential_suite,
        run_rng_metamorphic_suite, run_runtime_policy_adversarial_suite,
        run_shape_stride_adversarial_suite, run_shape_stride_differential_suite,
        run_shape_stride_metamorphic_suite, run_shape_stride_suite, run_smoke,
        run_ufunc_adversarial_suite, run_ufunc_differential_suite, run_ufunc_metamorphic_suite,
        set_dtype_promotion_log_path, set_shape_stride_log_path,
    };
    use fnp_iter::{
        RuntimeMode as IterRuntimeMode, TRANSFER_PACKET_REASON_CODES, TransferLogRecord,
    };
    use serde_json::Value;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn smoke_harness_finds_oracle_and_fixtures() {
        let cfg = HarnessConfig::default_paths();
        let report = run_smoke(&cfg);
        assert!(report.oracle_present, "oracle repo should be present");
        assert!(report.fixture_count >= 1, "expected at least one fixture");
        assert!(report.strict_mode);
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
            assert!(
                obj.get("rhs")
                    .and_then(Value::as_str)
                    .is_some_and(|s| !s.trim().is_empty())
            );
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
}
