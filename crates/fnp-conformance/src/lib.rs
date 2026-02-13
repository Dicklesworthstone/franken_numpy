#![forbid(unsafe_code)]

pub mod benchmark;
pub mod contract_schema;
pub mod raptorq_artifacts;
pub mod ufunc_differential;

use fnp_dtype::{DType, promote};
use fnp_ndarray::{MemoryOrder, broadcast_shape, contiguous_strides};
use fnp_runtime::{
    CompatibilityClass, DecisionAction, EvidenceLedger, RuntimeMode, decide_and_record,
};
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct HarnessConfig {
    pub oracle_root: PathBuf,
    pub fixture_root: PathBuf,
    pub strict_mode: bool,
}

impl HarnessConfig {
    #[must_use]
    pub fn default_paths() -> Self {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        Self {
            oracle_root: repo_root.join("legacy_numpy_code/numpy"),
            fixture_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"),
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
        let mode = match case.mode.as_str() {
            "strict" => RuntimeMode::Strict,
            "hardened" => RuntimeMode::Hardened,
            bad => return Err(format!("{}: invalid mode {bad}", case.id)),
        };

        let class = match case.class.as_str() {
            "known_compatible" => CompatibilityClass::KnownCompatible,
            "known_incompatible" => CompatibilityClass::KnownIncompatible,
            "unknown" => CompatibilityClass::Unknown,
            bad => return Err(format!("{}: invalid class {bad}", case.id)),
        };

        let expected_action = match case.expected_action.as_str() {
            "allow" => DecisionAction::Allow,
            "full_validate" => DecisionAction::FullValidate,
            "fail_closed" => DecisionAction::FailClosed,
            bad => return Err(format!("{}: invalid expected_action {bad}", case.id)),
        };

        let actual = decide_and_record(
            &mut ledger,
            mode,
            class,
            case.risk_score,
            case.threshold,
            &case.id,
        );

        if actual == expected_action {
            report.pass_count += 1;
        } else {
            report.failures.push(format!(
                "{}: action mismatch expected={expected_action:?} actual={actual:?}",
                case.id
            ));
        }
    }

    if ledger.events().len() != report.case_count {
        report.failures.push(format!(
            "ledger size mismatch expected={} actual={}",
            report.case_count,
            ledger.events().len()
        ));
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
        run_ufunc_differential_suite(config)?,
    ])
}

#[cfg(test)]
mod tests {
    use super::{HarnessConfig, run_all_core_suites, run_smoke, run_ufunc_differential_suite};
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
