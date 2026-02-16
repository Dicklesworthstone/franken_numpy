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
    IOSupportedDType, MemmapMode, classify_load_dispatch, validate_header_schema,
    validate_io_policy_metadata, validate_magic_version, validate_memmap_contract,
    validate_npz_archive_budget, validate_read_payload,
};
use fnp_linalg::{
    LinAlgError, QrMode, lstsq_output_shapes, qr_output_shapes, solve_2x2, svd_output_shapes,
    validate_backend_bridge, validate_policy_metadata as validate_linalg_policy_metadata,
    validate_spectral_branch, validate_tolerance_policy,
};
use fnp_ndarray::{MemoryOrder, NdLayout, broadcast_shape, contiguous_strides};
use fnp_runtime::{
    CompatibilityClass, DecisionAction, DecisionAuditContext, EvidenceLedger, RuntimeMode,
    decide_and_record_with_context, decide_compatibility_from_wire,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
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
    seed: u64,
}

#[derive(Debug, Deserialize)]
struct IoAdversarialCase {
    id: String,
    operation: String,
    expected_error_contains: String,
    #[serde(default)]
    severity: String,
    #[serde(default)]
    seed: u64,
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

static RUNTIME_POLICY_LOG_PATH: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
static SHAPE_STRIDE_LOG_PATH: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
static DTYPE_PROMOTION_LOG_PATH: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();

fn default_f64_dtype_name() -> String {
    "f64".to_string()
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

pub fn run_shape_stride_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let path = config.fixture_root.join("shape_stride_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<ShapeStrideFixtureCase> =
        serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))?;

    let mut report = SuiteReport {
        suite: "shape_stride",
        case_count: cases.len(),
        pass_count: 0,
        failures: Vec::new(),
    };

    for case in cases {
        let mut ok = true;
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

        match (
            &case.expected_broadcast,
            broadcast_shape(&case.lhs, &case.rhs),
        ) {
            (Some(expected), Ok(actual)) if expected == &actual => {}
            (None, Err(_)) => {}
            (Some(expected), Ok(actual)) => {
                ok = false;
                report.failures.push(format!(
                    "{}: broadcast mismatch expected={expected:?} actual={actual:?}",
                    case.id
                ));
            }
            (Some(expected), Err(err)) => {
                ok = false;
                report.failures.push(format!(
                    "{}: broadcast expected={expected:?} but failed: {err}",
                    case.id
                ));
            }
            (None, Ok(actual)) => {
                ok = false;
                report.failures.push(format!(
                    "{}: broadcast expected failure but got {actual:?}",
                    case.id
                ));
            }
        }

        let order = match case.stride_order.as_str() {
            "C" => MemoryOrder::C,
            "F" => MemoryOrder::F,
            bad => {
                ok = false;
                report
                    .failures
                    .push(format!("{}: invalid stride_order={bad}", case.id));
                MemoryOrder::C
            }
        };

        let computed_strides =
            match contiguous_strides(&case.stride_shape, case.stride_item_size, order) {
                Ok(strides) if strides == case.expected_strides => Some(strides),
                Ok(strides) => {
                    ok = false;
                    report.failures.push(format!(
                        "{}: stride mismatch expected={:?} actual={strides:?}",
                        case.id, case.expected_strides
                    ));
                    Some(strides)
                }
                Err(err) => {
                    ok = false;
                    report
                        .failures
                        .push(format!("{}: stride computation failed: {err}", case.id));
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
                            report.failures.push(format!(
                                "{}: as_strided expected error containing '{}' but succeeded",
                                case.id, needle
                            ));
                        } else {
                            if let Some(expected_shape) = &as_strided_case.expected_shape
                                && view.shape != *expected_shape
                            {
                                ok = false;
                                report.failures.push(format!(
                                    "{}: as_strided shape mismatch expected={expected_shape:?} actual={:?}",
                                    case.id, view.shape
                                ));
                            }

                            if let Some(expected_strides) = &as_strided_case.expected_strides
                                && view.strides != *expected_strides
                            {
                                ok = false;
                                report.failures.push(format!(
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
                                report.failures.push(format!(
                                    "{}: as_strided expected error containing '{}' but got '{}'",
                                    case.id, needle, err
                                ));
                            }
                        } else {
                            ok = false;
                            report.failures.push(format!(
                                "{}: as_strided unexpectedly failed: {}",
                                case.id, err
                            ));
                        }
                    }
                }
            } else {
                ok = false;
                report.failures.push(format!(
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
                            report.failures.push(format!(
                                "{}: broadcast_to expected error containing '{}' but succeeded",
                                case.id, needle
                            ));
                        } else {
                            if let Some(expected_shape) = &broadcast_to_case.expected_shape
                                && view.shape != *expected_shape
                            {
                                ok = false;
                                report.failures.push(format!(
                                    "{}: broadcast_to shape mismatch expected={expected_shape:?} actual={:?}",
                                    case.id, view.shape
                                ));
                            }

                            if let Some(expected_strides) = &broadcast_to_case.expected_strides
                                && view.strides != *expected_strides
                            {
                                ok = false;
                                report.failures.push(format!(
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
                                report.failures.push(format!(
                                    "{}: broadcast_to expected error containing '{}' but got '{}'",
                                    case.id, needle, err
                                ));
                            }
                        } else {
                            ok = false;
                            report.failures.push(format!(
                                "{}: broadcast_to unexpectedly failed: {}",
                                case.id, err
                            ));
                        }
                    }
                }
            } else {
                ok = false;
                report.failures.push(format!(
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
                            report.failures.push(format!(
                                "{}: sliding_window_view expected error containing '{}' but succeeded",
                                case.id, needle
                            ));
                        } else {
                            if let Some(expected_shape) = &sliding_window_case.expected_shape
                                && view.shape != *expected_shape
                            {
                                ok = false;
                                report.failures.push(format!(
                                    "{}: sliding_window_view shape mismatch expected={expected_shape:?} actual={:?}",
                                    case.id, view.shape
                                ));
                            }

                            if let Some(expected_strides) = &sliding_window_case.expected_strides
                                && view.strides != *expected_strides
                            {
                                ok = false;
                                report.failures.push(format!(
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
                                report.failures.push(format!(
                                    "{}: sliding_window_view expected error containing '{}' but got '{}'",
                                    case.id, needle, err
                                ));
                            }
                        } else {
                            ok = false;
                            report.failures.push(format!(
                                "{}: sliding_window_view unexpectedly failed: {}",
                                case.id, err
                            ));
                        }
                    }
                }
            } else {
                ok = false;
                report.failures.push(format!(
                    "{}: cannot validate sliding_window_view without valid base layout",
                    case.id
                ));
            }
        }

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

    let failures = report
        .failures
        .iter()
        .map(|failure| {
            format!(
                "{}: {}",
                failure.id,
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
        let pass = match case.relation.as_str() {
            "add_commutative" => {
                let Some(rhs_shape) = case.rhs_shape.clone() else {
                    report.failures.push(format!(
                        "{}: missing rhs_shape for add_commutative",
                        case.id
                    ));
                    continue;
                };
                let Some(rhs_values) = case.rhs_values.clone() else {
                    report.failures.push(format!(
                        "{}: missing rhs_values for add_commutative",
                        case.id
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
                };
                evaluate_commutative_pair(&case.id, case.seed, lhs_rhs, rhs_lhs, &mut report)
            }
            "mul_commutative" => {
                let Some(rhs_shape) = case.rhs_shape.clone() else {
                    report.failures.push(format!(
                        "{}: missing rhs_shape for mul_commutative",
                        case.id
                    ));
                    continue;
                };
                let Some(rhs_values) = case.rhs_values.clone() else {
                    report.failures.push(format!(
                        "{}: missing rhs_values for mul_commutative",
                        case.id
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
                };
                evaluate_commutative_pair(&case.id, case.seed, lhs_rhs, rhs_lhs, &mut report)
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
                };

                let Ok((_, base_values, _)) = ufunc_differential::execute_input_case(&sum_case)
                else {
                    report.failures.push(format!(
                        "{}: seed={} failed evaluating base sum",
                        case.id, case.seed
                    ));
                    continue;
                };
                let Ok((scaled_shape, scaled_values, _)) =
                    ufunc_differential::execute_input_case(&scale_case)
                else {
                    report.failures.push(format!(
                        "{}: seed={} failed evaluating scaled array",
                        case.id, case.seed
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
                };

                let Ok((_, sum_scaled_values, _)) =
                    ufunc_differential::execute_input_case(&scaled_sum_case)
                else {
                    report.failures.push(format!(
                        "{}: seed={} failed evaluating scaled sum",
                        case.id, case.seed
                    ));
                    continue;
                };

                let expected = base_values.first().copied().unwrap_or(0.0) * scalar;
                let actual = sum_scaled_values.first().copied().unwrap_or(0.0);
                let abs_err = (expected - actual).abs();
                let threshold = 1e-9 + 1e-9 * expected.abs();
                if abs_err > threshold {
                    report.failures.push(format!(
                        "{}: seed={} sum_linearity mismatch expected={} actual={} abs_err={} threshold={}",
                        case.id, case.seed, expected, actual, abs_err, threshold
                    ));
                    false
                } else {
                    true
                }
            }
            other => {
                report
                    .failures
                    .push(format!("{}: unsupported relation {}", case.id, other));
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
        };

        match ufunc_differential::execute_input_case(&input) {
            Ok((shape, _, _)) => {
                report.failures.push(format!(
                    "{}: seed={} expected error containing '{}' but execution succeeded with shape={shape:?}",
                    case.id, case.seed, case.expected_error_contains
                ));
            }
            Err(err) => {
                let expected = case.expected_error_contains.to_lowercase();
                let actual = err.to_lowercase();
                if actual.contains(&expected) {
                    report.pass_count += 1;
                } else {
                    report.failures.push(format!(
                        "{}: seed={} expected error containing '{}' but got '{}'",
                        case.id, case.seed, case.expected_error_contains, err
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
                    let contains_expected = err.to_string().to_lowercase().contains(&expected_error);
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
        let env_fingerprint = normalize_env_fingerprint(&case.env_fingerprint);
        let artifact_refs = normalize_artifact_refs(case.artifact_refs.clone());
        let reason_code = normalize_reason_code(&case.reason_code);
        let severity = case.severity.trim().to_lowercase();
        if !matches!(severity.as_str(), "low" | "medium" | "high" | "critical") {
            report.failures.push(format!(
                "{}: invalid severity '{}' (must be low|medium|high|critical), reason_code={}, env_fingerprint={}, artifact_refs={}",
                case.id,
                case.severity,
                reason_code,
                env_fingerprint,
                artifact_refs.join(",")
            ));
            continue;
        }
        if case.expected_error_contains.trim().is_empty() {
            report.failures.push(format!(
                "{}: expected_error_contains must be non-empty, reason_code={}, env_fingerprint={}, artifact_refs={}",
                case.id,
                reason_code,
                env_fingerprint,
                artifact_refs.join(",")
            ));
            continue;
        }

        let expected = case.expected_error_contains.to_lowercase();
        match execute_io_adversarial_operation(&case) {
            Ok(()) => {
                report.failures.push(format!(
                    "{}: severity={severity} seed={} reason_code={} env_fingerprint={} artifact_refs={} expected error containing '{}' but operation '{}' succeeded",
                    case.id,
                    case.seed,
                    reason_code,
                    env_fingerprint,
                    artifact_refs.join(","),
                    case.expected_error_contains,
                    case.operation
                ));
            }
            Err(actual_error) => {
                if actual_error.to_lowercase().contains(&expected) {
                    report.pass_count += 1;
                } else {
                    report.failures.push(format!(
                        "{}: severity={severity} seed={} reason_code={} env_fingerprint={} artifact_refs={} expected error containing '{}' but got '{}'",
                        case.id,
                        case.seed,
                        reason_code,
                        env_fingerprint,
                        artifact_refs.join(","),
                        case.expected_error_contains,
                        actual_error
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
        run_dtype_promotion_suite(config)?,
        run_runtime_policy_suite(config)?,
        run_runtime_policy_adversarial_suite(config)?,
        run_linalg_differential_suite(config)?,
        run_linalg_metamorphic_suite(config)?,
        run_linalg_adversarial_suite(config)?,
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

fn evaluate_commutative_pair(
    case_id: &str,
    seed: u64,
    lhs_rhs: UFuncInputCase,
    rhs_lhs: UFuncInputCase,
    report: &mut SuiteReport,
) -> bool {
    let Ok((shape_a, values_a, dtype_a)) = ufunc_differential::execute_input_case(&lhs_rhs) else {
        report.failures.push(format!(
            "{}: seed={} failed lhs_rhs execution",
            case_id, seed
        ));
        return false;
    };
    let Ok((shape_b, values_b, dtype_b)) = ufunc_differential::execute_input_case(&rhs_lhs) else {
        report.failures.push(format!(
            "{}: seed={} failed rhs_lhs execution",
            case_id, seed
        ));
        return false;
    };

    if shape_a != shape_b {
        report.failures.push(format!(
            "{}: seed={} commutative shape mismatch lhs_rhs={shape_a:?} rhs_lhs={shape_b:?}",
            case_id, seed
        ));
        return false;
    }

    if dtype_a != dtype_b {
        report.failures.push(format!(
            "{}: seed={} commutative dtype mismatch lhs_rhs={} rhs_lhs={}",
            case_id, seed, dtype_a, dtype_b
        ));
        return false;
    }

    if !approx_equal_values(&values_a, &values_b, 1e-9, 1e-9) {
        report.failures.push(format!(
            "{}: seed={} commutative value mismatch",
            case_id, seed
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
            if case.expected_q_shape.as_ref() != Some(q_shape)
                || case.expected_r_shape != *r_shape
            {
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

fn execute_io_adversarial_operation(case: &IoAdversarialCase) -> Result<(), String> {
    match case.operation.as_str() {
        "magic_version" => validate_magic_version(&case.payload_prefix)
            .map(|_| ())
            .map_err(|err| err.to_string()),
        "header_schema" => validate_header_schema(
            &case.shape,
            case.fortran_order,
            &case.dtype_descr,
            case.header_len,
        )
        .map(|_| ())
        .map_err(|err| err.to_string()),
        "dtype_decode" => IOSupportedDType::decode(&case.dtype_descr)
            .map(|_| ())
            .map_err(|err| err.to_string()),
        "read_payload" => {
            let dtype = IOSupportedDType::decode(&case.dtype_descr)
                .map_err(|err| format!("{}: dtype decode failed: {err}", case.id))?;
            validate_read_payload(&case.shape, case.payload_len_bytes, dtype)
                .map(|_| ())
                .map_err(|err| err.to_string())
        }
        "memmap_contract" => {
            let mode = MemmapMode::parse(&case.memmap_mode).map_err(|err| err.to_string())?;
            let dtype = IOSupportedDType::decode(&case.dtype_descr)
                .map_err(|err| format!("{}: dtype decode failed: {err}", case.id))?;
            validate_memmap_contract(
                mode,
                dtype,
                case.file_len_bytes,
                case.expected_bytes,
                case.validation_retries,
            )
            .map_err(|err| err.to_string())
        }
        "load_dispatch" => classify_load_dispatch(&case.payload_prefix, case.allow_pickle)
            .map(|_| ())
            .map_err(|err| err.to_string()),
        "npz_archive_budget" => validate_npz_archive_budget(
            case.member_count,
            case.uncompressed_bytes,
            case.dispatch_retries,
        )
        .map_err(|err| err.to_string()),
        "policy_metadata" => validate_io_policy_metadata(&case.mode_raw, &case.class_raw)
            .map_err(|err| err.to_string()),
        other => Err(format!("unsupported io_adversarial operation {other}")),
    }
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
        run_dtype_promotion_suite, run_io_adversarial_suite, run_runtime_policy_adversarial_suite,
        run_shape_stride_suite, run_smoke, run_ufunc_adversarial_suite,
        run_ufunc_differential_suite, run_ufunc_metamorphic_suite, set_dtype_promotion_log_path,
        set_shape_stride_log_path,
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
    fn io_adversarial_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite = run_io_adversarial_suite(&cfg).expect("io adversarial suite should run");
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
