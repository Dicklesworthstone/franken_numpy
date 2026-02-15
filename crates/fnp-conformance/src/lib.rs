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
use fnp_ndarray::{MemoryOrder, broadcast_shape, contiguous_strides};
use fnp_runtime::{
    CompatibilityClass, DecisionAction, DecisionAuditContext, EvidenceLedger, RuntimeMode,
    decide_and_record_with_context, decide_compatibility_from_wire,
};
use serde::{Deserialize, Serialize};
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

static RUNTIME_POLICY_LOG_PATH: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
static SHAPE_STRIDE_LOG_PATH: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
static DTYPE_PROMOTION_LOG_PATH: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();

fn default_f64_dtype_name() -> String {
    "f64".to_string()
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

        match contiguous_strides(&case.stride_shape, case.stride_item_size, order) {
            Ok(strides) if strides == case.expected_strides => {}
            Ok(strides) => {
                ok = false;
                report.failures.push(format!(
                    "{}: stride mismatch expected={:?} actual={strides:?}",
                    case.id, case.expected_strides
                ));
            }
            Err(err) => {
                ok = false;
                report
                    .failures
                    .push(format!("{}: stride computation failed: {err}", case.id));
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

pub fn run_all_core_suites(config: &HarnessConfig) -> Result<Vec<SuiteReport>, String> {
    Ok(vec![
        run_shape_stride_suite(config)?,
        run_dtype_promotion_suite(config)?,
        run_runtime_policy_suite(config)?,
        run_runtime_policy_adversarial_suite(config)?,
        security_contracts::run_security_contract_suite(config)?,
        test_contracts::run_test_contract_suite(config)?,
        workflow_scenarios::run_user_workflow_scenario_suite(config)?,
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
    file.write_all(line.as_bytes())
        .and_then(|_| file.write_all(b"\n"))
        .map_err(|err| {
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
    file.write_all(line.as_bytes())
        .and_then(|_| file.write_all(b"\n"))
        .map_err(|err| {
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
    file.write_all(line.as_bytes())
        .and_then(|_| file.write_all(b"\n"))
        .map_err(|err| {
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
        HarnessConfig, run_all_core_suites, run_dtype_promotion_suite,
        run_runtime_policy_adversarial_suite, run_shape_stride_suite, run_smoke,
        run_ufunc_adversarial_suite, run_ufunc_differential_suite, run_ufunc_metamorphic_suite,
        set_dtype_promotion_log_path, set_shape_stride_log_path,
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
        }
        assert!(entry_count > 0, "shape/stride log should contain entries");
        set_shape_stride_log_path(None);
        let _ = fs::remove_file(log_path);
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
    fn workflow_scenario_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let suite = super::workflow_scenarios::run_user_workflow_scenario_suite(&cfg)
            .expect("workflow scenario suite should run");
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
