#![forbid(unsafe_code)]

pub mod benchmark;
pub mod contract_schema;
pub mod raptorq_artifacts;
pub mod security_contracts;
pub mod ufunc_differential;

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
}

#[derive(Debug, Deserialize)]
struct PromotionFixtureCase {
    id: String,
    lhs: String,
    rhs: String,
    expected: String,
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

static RUNTIME_POLICY_LOG_PATH: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();

pub fn set_runtime_policy_log_path(path: Option<PathBuf>) {
    let cell = RUNTIME_POLICY_LOG_PATH.get_or_init(|| Mutex::new(None));
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
        if actual == expected {
            report.pass_count += 1;
        } else {
            report.failures.push(format!(
                "{}: promotion mismatch expected={} actual={}",
                case.id,
                expected.name(),
                actual.name()
            ));
        }
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

pub fn run_all_core_suites(config: &HarnessConfig) -> Result<Vec<SuiteReport>, String> {
    Ok(vec![
        run_shape_stride_suite(config)?,
        run_dtype_promotion_suite(config)?,
        run_runtime_policy_suite(config)?,
        run_runtime_policy_adversarial_suite(config)?,
        security_contracts::run_security_contract_suite(config)?,
        run_ufunc_differential_suite(config)?,
    ])
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
        HarnessConfig, run_all_core_suites, run_runtime_policy_adversarial_suite, run_smoke,
        run_ufunc_differential_suite,
    };
    use std::path::PathBuf;

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
