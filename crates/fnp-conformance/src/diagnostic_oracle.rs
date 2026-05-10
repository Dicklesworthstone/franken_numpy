#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

pub const DIAGNOSTIC_ORACLE_SCHEMA_VERSION: &str = "diagnostic-oracle-v1";

const PYTHON_DRIVER: &str = r#"
import json
import os
import platform
import sys
import warnings

request = json.load(sys.stdin)
require_numpy = bool(request.get("require_numpy", True))

numpy_available = False
numpy_version = None
numpy_import_error = None
try:
    import numpy as np
    numpy_available = True
    numpy_version = np.__version__
except Exception as exc:
    np = None
    numpy_import_error = f"{type(exc).__module__}.{type(exc).__name__}: {exc}"
    if require_numpy:
        print(json.dumps({
            "schema_version": request.get("schema_version", "diagnostic-oracle-v1"),
            "environment": {
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "numpy_version": None,
                "numpy_available": False,
                "numpy_import_error": numpy_import_error,
                "fnp_oracle_python": os.environ.get("FNP_ORACLE_PYTHON"),
            },
            "observations": [],
            "diagnostics": [{
                "reason_code": "numpy_unavailable",
                "message": numpy_import_error,
            }],
        }, sort_keys=True))
        raise SystemExit(0)

observations = []
for case in request.get("cases", []):
    namespace = {
        "__builtins__": __builtins__,
        "np": np,
        "warnings": warnings,
    }
    caught_warnings = []
    outcome = "success"
    exception = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            exec(case["python"], namespace, namespace)
        except BaseException as exc:
            outcome = "exception"
            exception = {
                "class_name": type(exc).__name__,
                "module": type(exc).__module__,
                "message": str(exc),
            }
        for warning in caught:
            caught_warnings.append({
                "category": type(warning.message).__name__,
                "module": type(warning.message).__module__,
                "message": str(warning.message),
            })
    observations.append({
        "case_id": case["id"],
        "outcome": outcome,
        "exception": exception,
        "warnings": caught_warnings,
    })

print(json.dumps({
    "schema_version": request.get("schema_version", "diagnostic-oracle-v1"),
    "environment": {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": numpy_version,
        "numpy_available": numpy_available,
        "numpy_import_error": numpy_import_error,
        "fnp_oracle_python": os.environ.get("FNP_ORACLE_PYTHON"),
    },
    "observations": observations,
    "diagnostics": [],
}, sort_keys=True))
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticMode {
    Strict,
    Hardened,
    OracleOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticRequirementLevel {
    Must,
    Should,
    May,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticOutcome {
    Success,
    Exception,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticCase {
    pub id: String,
    pub surface: String,
    pub requirement_level: DiagnosticRequirementLevel,
    pub mode: DiagnosticMode,
    pub python: String,
    pub expected: DiagnosticExpectation,
    #[serde(default)]
    pub version_guards: Vec<DiagnosticVersionGuard>,
    #[serde(default)]
    pub intentional_divergence: Option<String>,
    #[serde(default)]
    pub exploratory: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticExpectation {
    pub outcome: DiagnosticOutcome,
    #[serde(default)]
    pub exception_class: Option<String>,
    #[serde(default)]
    pub warning_categories: Vec<String>,
    #[serde(default)]
    pub message_fragments: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticVersionGuard {
    pub package: String,
    #[serde(default)]
    pub min_inclusive: Option<String>,
    #[serde(default)]
    pub max_exclusive: Option<String>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticOracleOptions {
    pub python: String,
    pub require_numpy: bool,
}

impl Default for DiagnosticOracleOptions {
    fn default() -> Self {
        Self {
            python: resolve_oracle_python(),
            require_numpy: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticOracleRequest {
    pub schema_version: String,
    pub require_numpy: bool,
    pub cases: Vec<DiagnosticCase>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticOracleReport {
    pub schema_version: String,
    pub environment: DiagnosticEnvironment,
    pub observations: Vec<DiagnosticObservation>,
    #[serde(default)]
    pub diagnostics: Vec<DiagnosticRunnerDiagnostic>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticEnvironment {
    pub python_executable: String,
    pub python_version: String,
    pub platform: String,
    pub numpy_version: Option<String>,
    pub numpy_available: bool,
    pub numpy_import_error: Option<String>,
    pub fnp_oracle_python: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticObservation {
    pub case_id: String,
    pub outcome: DiagnosticOutcome,
    pub exception: Option<DiagnosticException>,
    pub warnings: Vec<DiagnosticWarning>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticException {
    pub class_name: String,
    pub module: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticWarning {
    pub category: String,
    pub module: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticRunnerDiagnostic {
    pub reason_code: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticCaseVerdict {
    pub case_id: String,
    pub status: DiagnosticVerdictStatus,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticVerdictStatus {
    Pass,
    Fail,
    Skipped,
}

#[must_use]
pub fn resolve_oracle_python() -> String {
    std::env::var("FNP_ORACLE_PYTHON")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "python3".to_string())
}

pub fn validate_diagnostic_cases(cases: &[DiagnosticCase]) -> Result<(), String> {
    for case in cases {
        if case.id.trim().is_empty() {
            return Err("diagnostic case id must not be empty".to_string());
        }
        if case.surface.trim().is_empty() {
            return Err(format!("{}: surface must not be empty", case.id));
        }
        if case.python.trim().is_empty() {
            return Err(format!("{}: python snippet must not be empty", case.id));
        }
        if case.expected.outcome == DiagnosticOutcome::Unknown && !case.exploratory {
            return Err(format!(
                "{}: unknown expectation requires exploratory=true",
                case.id
            ));
        }
        if case.expected.outcome == DiagnosticOutcome::Exception
            && case.expected.exception_class.is_none()
            && !case.exploratory
        {
            return Err(format!(
                "{}: exception expectation requires exception_class or exploratory=true",
                case.id
            ));
        }
        for guard in &case.version_guards {
            if guard.package != "numpy" {
                return Err(format!(
                    "{}: unsupported version guard package {}",
                    case.id, guard.package
                ));
            }
            if guard.reason.trim().is_empty() {
                return Err(format!("{}: version guard reason is required", case.id));
            }
        }
    }
    Ok(())
}

pub fn run_diagnostic_oracle(
    cases: &[DiagnosticCase],
    options: &DiagnosticOracleOptions,
) -> Result<DiagnosticOracleReport, String> {
    validate_diagnostic_cases(cases)?;
    let request = DiagnosticOracleRequest {
        schema_version: DIAGNOSTIC_ORACLE_SCHEMA_VERSION.to_string(),
        require_numpy: options.require_numpy,
        cases: cases.to_vec(),
    };
    let raw_request =
        serde_json::to_vec(&request).map_err(|err| format!("serialize oracle request: {err}"))?;
    let mut child = Command::new(&options.python)
        .arg("-c")
        .arg(PYTHON_DRIVER)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| format!("spawn oracle python {}: {err}", options.python))?;
    child
        .stdin
        .as_mut()
        .ok_or_else(|| "oracle python stdin unavailable".to_string())?
        .write_all(&raw_request)
        .map_err(|err| format!("write oracle request: {err}"))?;
    let output = child
        .wait_with_output()
        .map_err(|err| format!("wait for oracle python: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "oracle python failed with status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    serde_json::from_slice(&output.stdout).map_err(|err| {
        format!(
            "parse oracle report: {err}; stdout={}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

#[must_use]
pub fn evaluate_diagnostic_report(
    cases: &[DiagnosticCase],
    report: &DiagnosticOracleReport,
) -> Vec<DiagnosticCaseVerdict> {
    let numpy_version = report.environment.numpy_version.as_deref();
    cases
        .iter()
        .map(|case| {
            if !case_applies(case, numpy_version) {
                return DiagnosticCaseVerdict {
                    case_id: case.id.clone(),
                    status: DiagnosticVerdictStatus::Skipped,
                    reasons: vec!["version_guard_not_satisfied".to_string()],
                };
            }
            let Some(observation) = report
                .observations
                .iter()
                .find(|observation| observation.case_id == case.id)
            else {
                return DiagnosticCaseVerdict {
                    case_id: case.id.clone(),
                    status: DiagnosticVerdictStatus::Fail,
                    reasons: vec!["missing_observation".to_string()],
                };
            };
            evaluate_observation(case, observation)
        })
        .collect()
}

#[must_use]
pub fn smoke_cases() -> Vec<DiagnosticCase> {
    vec![
        DiagnosticCase {
            id: "smoke_success".to_string(),
            surface: "diagnostic_oracle".to_string(),
            requirement_level: DiagnosticRequirementLevel::Must,
            mode: DiagnosticMode::OracleOnly,
            python: "value = np.array([1, 2, 3]).sum(); assert int(value) == 6".to_string(),
            expected: DiagnosticExpectation {
                outcome: DiagnosticOutcome::Success,
                exception_class: None,
                warning_categories: Vec::new(),
                message_fragments: Vec::new(),
            },
            version_guards: Vec::new(),
            intentional_divergence: None,
            exploratory: false,
        },
        DiagnosticCase {
            id: "smoke_exception".to_string(),
            surface: "diagnostic_oracle".to_string(),
            requirement_level: DiagnosticRequirementLevel::Must,
            mode: DiagnosticMode::OracleOnly,
            python: "np.array([1, 2, 3]).reshape(2, 2)".to_string(),
            expected: DiagnosticExpectation {
                outcome: DiagnosticOutcome::Exception,
                exception_class: Some("ValueError".to_string()),
                warning_categories: Vec::new(),
                message_fragments: vec!["reshape".to_string()],
            },
            version_guards: Vec::new(),
            intentional_divergence: None,
            exploratory: false,
        },
        DiagnosticCase {
            id: "smoke_warning".to_string(),
            surface: "diagnostic_oracle".to_string(),
            requirement_level: DiagnosticRequirementLevel::Must,
            mode: DiagnosticMode::OracleOnly,
            python: "warnings.warn('diagnostic warning one', UserWarning)".to_string(),
            expected: DiagnosticExpectation {
                outcome: DiagnosticOutcome::Success,
                exception_class: None,
                warning_categories: vec!["UserWarning".to_string()],
                message_fragments: vec!["diagnostic warning one".to_string()],
            },
            version_guards: Vec::new(),
            intentional_divergence: None,
            exploratory: false,
        },
        DiagnosticCase {
            id: "smoke_multi_warning".to_string(),
            surface: "diagnostic_oracle".to_string(),
            requirement_level: DiagnosticRequirementLevel::Must,
            mode: DiagnosticMode::OracleOnly,
            python:
                "warnings.warn('first diagnostic warning', UserWarning); warnings.warn('second diagnostic warning', RuntimeWarning)"
                    .to_string(),
            expected: DiagnosticExpectation {
                outcome: DiagnosticOutcome::Success,
                exception_class: None,
                warning_categories: vec!["UserWarning".to_string(), "RuntimeWarning".to_string()],
                message_fragments: vec![
                    "first diagnostic warning".to_string(),
                    "second diagnostic warning".to_string(),
                ],
            },
            version_guards: Vec::new(),
            intentional_divergence: None,
            exploratory: false,
        },
    ]
}

pub fn load_cases(path: &Path) -> Result<Vec<DiagnosticCase>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("read diagnostic cases {}: {err}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|err| format!("parse diagnostic cases {}: {err}", path.display()))
}

pub fn write_report(path: &Path, report: &DiagnosticOracleReport) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("create report dir {}: {err}", parent.display()))?;
    }
    let raw = serde_json::to_string_pretty(report)
        .map_err(|err| format!("serialize diagnostic report: {err}"))?;
    fs::write(path, raw).map_err(|err| format!("write report {}: {err}", path.display()))
}

pub fn write_jsonl(path: &Path, report: &DiagnosticOracleReport) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("create jsonl dir {}: {err}", parent.display()))?;
    }
    let mut lines = String::new();
    for observation in &report.observations {
        lines.push_str(
            &serde_json::to_string(observation)
                .map_err(|err| format!("serialize diagnostic observation: {err}"))?,
        );
        lines.push('\n');
    }
    fs::write(path, lines).map_err(|err| format!("write jsonl {}: {err}", path.display()))
}

fn evaluate_observation(
    case: &DiagnosticCase,
    observation: &DiagnosticObservation,
) -> DiagnosticCaseVerdict {
    let mut reasons = Vec::new();
    if observation.outcome != case.expected.outcome {
        reasons.push(format!(
            "outcome mismatch: expected {:?}, observed {:?}",
            case.expected.outcome, observation.outcome
        ));
    }
    if let Some(expected_class) = case.expected.exception_class.as_deref() {
        match observation.exception.as_ref() {
            Some(exception) if exception.class_name == expected_class => {}
            Some(exception) => reasons.push(format!(
                "exception class mismatch: expected {expected_class}, observed {}",
                exception.class_name
            )),
            None => reasons.push(format!(
                "missing exception: expected class {expected_class}"
            )),
        }
    }
    let observed_categories = observation
        .warnings
        .iter()
        .map(|warning| warning.category.as_str())
        .collect::<Vec<_>>();
    let expected_categories = case
        .expected
        .warning_categories
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    if observed_categories != expected_categories {
        reasons.push(format!(
            "warning categories mismatch: expected {:?}, observed {:?}",
            expected_categories, observed_categories
        ));
    }
    let diagnostic_text = observation_text(observation);
    for fragment in &case.expected.message_fragments {
        if !diagnostic_text.contains(fragment) {
            reasons.push(format!("missing message fragment {fragment:?}"));
        }
    }
    DiagnosticCaseVerdict {
        case_id: case.id.clone(),
        status: if reasons.is_empty() {
            DiagnosticVerdictStatus::Pass
        } else {
            DiagnosticVerdictStatus::Fail
        },
        reasons,
    }
}

fn observation_text(observation: &DiagnosticObservation) -> String {
    let mut text = String::new();
    if let Some(exception) = &observation.exception {
        text.push_str(&exception.message);
        text.push('\n');
    }
    for warning in &observation.warnings {
        text.push_str(&warning.message);
        text.push('\n');
    }
    text
}

fn case_applies(case: &DiagnosticCase, numpy_version: Option<&str>) -> bool {
    case.version_guards.iter().all(|guard| {
        if guard.package != "numpy" {
            return false;
        }
        let Some(version) = numpy_version else {
            return false;
        };
        guard
            .min_inclusive
            .as_deref()
            .is_none_or(|minimum| compare_versions(version, minimum).is_ge())
            && guard
                .max_exclusive
                .as_deref()
                .is_none_or(|maximum| compare_versions(version, maximum).is_lt())
    })
}

fn compare_versions(lhs: &str, rhs: &str) -> std::cmp::Ordering {
    let lhs_parts = version_parts(lhs);
    let rhs_parts = version_parts(rhs);
    for index in 0..lhs_parts.len().max(rhs_parts.len()) {
        let lhs_part = lhs_parts.get(index).copied().unwrap_or(0);
        let rhs_part = rhs_parts.get(index).copied().unwrap_or(0);
        match lhs_part.cmp(&rhs_part) {
            std::cmp::Ordering::Equal => {}
            ordering => return ordering,
        }
    }
    std::cmp::Ordering::Equal
}

fn version_parts(version: &str) -> Vec<u64> {
    version
        .split(|ch: char| !ch.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .take(3)
        .map(|part| part.parse::<u64>().unwrap_or(0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_oracle_rejects_unknown_non_exploratory_expectation() {
        let mut case = smoke_cases()[0].clone();
        case.expected.outcome = DiagnosticOutcome::Unknown;
        case.exploratory = false;

        let err = validate_diagnostic_cases(&[case]).expect_err("unknown should fail closed");
        assert!(err.contains("exploratory=true"));
    }

    #[test]
    fn diagnostic_oracle_allows_unknown_exploratory_expectation() {
        let mut case = smoke_cases()[0].clone();
        case.expected.outcome = DiagnosticOutcome::Unknown;
        case.exploratory = true;

        validate_diagnostic_cases(&[case]).expect("exploratory unknown is allowed");
    }

    #[test]
    fn diagnostic_oracle_version_guard_filters_by_numpy_version() {
        let mut case = smoke_cases()[0].clone();
        case.version_guards.push(DiagnosticVersionGuard {
            package: "numpy".to_string(),
            min_inclusive: Some("2.0.0".to_string()),
            max_exclusive: Some("3.0.0".to_string()),
            reason: "NumPy 2.x diagnostic shape".to_string(),
        });

        assert!(case_applies(&case, Some("2.1.3")));
        assert!(!case_applies(&case, Some("1.26.4")));
        assert!(!case_applies(&case, Some("3.0.0")));
    }

    #[test]
    fn diagnostic_oracle_evaluates_success_exception_and_warnings() {
        let cases = smoke_cases();
        let options = DiagnosticOracleOptions {
            python: resolve_oracle_python(),
            require_numpy: false,
        };
        let report = run_diagnostic_oracle(&cases, &options).expect("run diagnostic oracle");

        let verdicts = evaluate_diagnostic_report(&cases, &report);
        assert_eq!(verdicts.len(), cases.len());
        assert!(
            verdicts
                .iter()
                .all(|verdict| verdict.status == DiagnosticVerdictStatus::Pass),
            "{verdicts:?}"
        );
        assert!(
            report
                .observations
                .iter()
                .any(|observation| observation.warnings.len() == 2)
        );
        assert!(report.environment.numpy_version.is_some());
    }

    #[test]
    fn diagnostic_oracle_jsonl_is_deterministic() {
        let report = DiagnosticOracleReport {
            schema_version: DIAGNOSTIC_ORACLE_SCHEMA_VERSION.to_string(),
            environment: DiagnosticEnvironment {
                python_executable: "/usr/bin/python3".to_string(),
                python_version: "3.14.0".to_string(),
                platform: "test-platform".to_string(),
                numpy_version: Some("2.3.0".to_string()),
                numpy_available: true,
                numpy_import_error: None,
                fnp_oracle_python: None,
            },
            observations: vec![DiagnosticObservation {
                case_id: "case-a".to_string(),
                outcome: DiagnosticOutcome::Success,
                exception: None,
                warnings: vec![DiagnosticWarning {
                    category: "UserWarning".to_string(),
                    module: "builtins".to_string(),
                    message: "stable warning".to_string(),
                }],
            }],
            diagnostics: Vec::new(),
        };

        let first = serde_json::to_string(&report.observations[0]).expect("serialize");
        let second = serde_json::to_string(&report.observations[0]).expect("serialize");
        assert_eq!(first, second);
    }
}
