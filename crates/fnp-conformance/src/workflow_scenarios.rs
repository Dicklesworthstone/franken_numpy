#![forbid(unsafe_code)]

use crate::ufunc_differential::{UFuncInputCase, execute_input_case, load_input_cases};
use crate::{HarnessConfig, SuiteReport};
use fnp_runtime::{
    CompatibilityClass, DecisionAction, RuntimeMode, decide_compatibility,
    decide_compatibility_from_wire,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

const REQUIRED_SCENARIO_CATEGORIES: &[&str] = &["high_frequency", "high_risk", "adversarial"];
const EXPECTED_SCENARIO_STATUSES: &[&str] = &["pass", "fail_closed"];

#[derive(Debug, Clone, Deserialize)]
struct WorkflowScenarioCase {
    id: String,
    category: String,
    description: String,
    seed: u64,
    env_fingerprint: String,
    artifact_refs: Vec<String>,
    reason_code: String,
    strict: WorkflowModeExpectation,
    hardened: WorkflowModeExpectation,
    steps: Vec<WorkflowStep>,
    links: WorkflowLinks,
    gaps: Vec<WorkflowGap>,
}

#[derive(Debug, Clone, Deserialize)]
struct WorkflowModeExpectation {
    expected_status: String,
}

#[derive(Debug, Clone, Deserialize)]
struct WorkflowLinks {
    differential_fixture_ids: Vec<String>,
    e2e_script_paths: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct WorkflowGap {
    bead_id: String,
    owner: String,
    priority: String,
    description: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum WorkflowStep {
    UfuncInput {
        id: String,
        case_id: String,
        #[serde(default)]
        expect_error_contains: Option<String>,
    },
    ShapeStrideCase {
        id: String,
        case_id: String,
    },
    RuntimePolicy {
        id: String,
        case_id: String,
        expected_action_strict: String,
        expected_action_hardened: String,
    },
    RuntimePolicyWire {
        id: String,
        case_id: String,
        expected_action_strict: String,
        expected_action_hardened: String,
    },
    IterFixtureCase {
        id: String,
        case_id: String,
        fixture_set: String,
    },
    IoFixtureCase {
        id: String,
        case_id: String,
        fixture_set: String,
    },
    LinalgFixtureCase {
        id: String,
        case_id: String,
        fixture_set: String,
    },
}

#[derive(Debug, Clone, Deserialize)]
struct PolicyFixtureCase {
    id: String,
    class: String,
    risk_score: f64,
    threshold: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct PolicyWireFixtureCase {
    id: String,
    mode_raw: String,
    class_raw: String,
    risk_score: f64,
    threshold: f64,
}

#[derive(Debug, Clone, Serialize)]
struct WorkflowScenarioLogEntry {
    suite: &'static str,
    fixture_id: String,
    seed: u64,
    mode: String,
    env_fingerprint: String,
    artifact_refs: Vec<String>,
    reason_code: String,
    scenario_id: String,
    step_id: String,
    step_kind: String,
    expected: String,
    actual: String,
    passed: bool,
    detail: String,
}

#[derive(Debug)]
struct ModeExecution {
    actual_status: String,
    failures: Vec<String>,
}

#[derive(Debug)]
struct IterStepResult {
    expected: String,
    actual: String,
    passed: bool,
    detail: String,
}

static WORKFLOW_SCENARIO_LOG_PATH: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
static WORKFLOW_SCENARIO_LOG_REQUIRED: OnceLock<Mutex<bool>> = OnceLock::new();

pub fn set_workflow_scenario_log_path(path: Option<PathBuf>) {
    let cell = WORKFLOW_SCENARIO_LOG_PATH.get_or_init(|| Mutex::new(None));
    if let Ok(mut slot) = cell.lock() {
        *slot = path;
    }
}

pub fn set_workflow_scenario_log_required(required: bool) {
    let cell = WORKFLOW_SCENARIO_LOG_REQUIRED.get_or_init(|| Mutex::new(false));
    if let Ok(mut slot) = cell.lock() {
        *slot = required;
    }
}

fn workflow_scenario_log_required() -> bool {
    WORKFLOW_SCENARIO_LOG_REQUIRED
        .get()
        .and_then(|cell| cell.lock().ok().map(|slot| *slot))
        .unwrap_or(false)
}

pub fn run_user_workflow_scenario_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let scenario_path = config.fixture_root.join("workflow_scenario_corpus.json");
    let raw = fs::read_to_string(&scenario_path)
        .map_err(|err| format!("failed reading {}: {err}", scenario_path.display()))?;
    let scenarios: Vec<WorkflowScenarioCase> =
        serde_json::from_str(&raw).map_err(|err| format!("invalid json: {err}"))?;

    let ufunc_cases = load_ufunc_case_map(&config.fixture_root.join("ufunc_input_cases.json"))?;
    let shape_stride_cases = load_shape_stride_case_map(&config.fixture_root)?;
    let policy_cases =
        load_runtime_policy_case_map(&config.fixture_root.join("runtime_policy_cases.json"))?;
    let wire_cases = load_runtime_policy_wire_case_map(
        &config
            .fixture_root
            .join("runtime_policy_adversarial_cases.json"),
    )?;
    let iter_differential_cases = load_iter_differential_case_map(&config.fixture_root)?;
    let iter_adversarial_cases = load_iter_adversarial_case_map(&config.fixture_root)?;
    let io_differential_cases = load_io_differential_case_map(&config.fixture_root)?;
    let io_adversarial_cases = load_io_adversarial_case_map(&config.fixture_root)?;
    let linalg_differential_cases = load_linalg_differential_case_map(&config.fixture_root)?;
    let linalg_adversarial_cases = load_linalg_adversarial_case_map(&config.fixture_root)?;

    let repo_root = derive_repo_root(&config.fixture_root)?;

    let mut report = SuiteReport {
        suite: "workflow_scenarios",
        case_count: 0,
        pass_count: 0,
        failures: Vec::new(),
    };
    let mut categories = BTreeSet::new();
    let mut ids = BTreeSet::new();

    for scenario in &scenarios {
        record_check(
            &mut report,
            ids.insert(scenario.id.clone()),
            format!(
                "workflow_scenario_corpus duplicate scenario id {}",
                scenario.id
            ),
        );
        record_check(
            &mut report,
            !scenario.description.trim().is_empty(),
            format!("{}: description must not be empty", scenario.id),
        );
        record_check(
            &mut report,
            !scenario.env_fingerprint.trim().is_empty(),
            format!("{}: env_fingerprint must not be empty", scenario.id),
        );
        record_check(
            &mut report,
            !scenario.reason_code.trim().is_empty(),
            format!("{}: reason_code must not be empty", scenario.id),
        );
        record_check(
            &mut report,
            !scenario.artifact_refs.is_empty(),
            format!("{}: artifact_refs must not be empty", scenario.id),
        );
        record_check(
            &mut report,
            !scenario.steps.is_empty(),
            format!("{}: steps must not be empty", scenario.id),
        );
        record_check(
            &mut report,
            !scenario.links.differential_fixture_ids.is_empty(),
            format!(
                "{}: links.differential_fixture_ids must not be empty",
                scenario.id
            ),
        );
        record_check(
            &mut report,
            !scenario.links.e2e_script_paths.is_empty(),
            format!("{}: links.e2e_script_paths must not be empty", scenario.id),
        );
        record_check(
            &mut report,
            !scenario.gaps.is_empty(),
            format!("{}: gaps must not be empty", scenario.id),
        );

        for fixture_id in &scenario.links.differential_fixture_ids {
            record_check(
                &mut report,
                ufunc_cases.contains_key(fixture_id.as_str())
                    || shape_stride_cases.contains_key(fixture_id.as_str())
                    || iter_differential_cases.contains_key(fixture_id.as_str())
                    || iter_adversarial_cases.contains_key(fixture_id.as_str())
                    || io_differential_cases.contains_key(fixture_id.as_str())
                    || io_adversarial_cases.contains_key(fixture_id.as_str())
                    || linalg_differential_cases.contains_key(fixture_id.as_str())
                    || linalg_adversarial_cases.contains_key(fixture_id.as_str()),
                format!(
                    "{}: differential fixture id not found in ufunc_input_cases.json, shape_stride_cases.json, iter_differential_cases.json, iter_adversarial_cases.json, io_differential_cases.json, io_adversarial_cases.json, linalg_differential_cases.json, or linalg_adversarial_cases.json: {}",
                    scenario.id, fixture_id
                ),
            );
        }

        for script in &scenario.links.e2e_script_paths {
            let script_path = repo_root.join(script);
            record_check(
                &mut report,
                script_path.is_file(),
                format!(
                    "{}: linked e2e script does not exist: {}",
                    scenario.id,
                    script_path.display()
                ),
            );
        }

        for gap in &scenario.gaps {
            record_check(
                &mut report,
                !gap.bead_id.trim().is_empty(),
                format!("{}: gap bead_id must not be empty", scenario.id),
            );
            record_check(
                &mut report,
                !gap.owner.trim().is_empty(),
                format!("{}: gap owner must not be empty", scenario.id),
            );
            record_check(
                &mut report,
                !gap.priority.trim().is_empty(),
                format!("{}: gap priority must not be empty", scenario.id),
            );
            record_check(
                &mut report,
                !gap.description.trim().is_empty(),
                format!("{}: gap description must not be empty", scenario.id),
            );
        }

        let category = scenario.category.trim().to_string();
        record_check(
            &mut report,
            REQUIRED_SCENARIO_CATEGORIES.contains(&category.as_str()),
            format!(
                "{}: category '{}' must be one of {}",
                scenario.id,
                category,
                REQUIRED_SCENARIO_CATEGORIES.join(", ")
            ),
        );
        categories.insert(category);

        let strict = execute_mode(
            scenario,
            RuntimeMode::Strict,
            &ufunc_cases,
            &shape_stride_cases,
            &policy_cases,
            &wire_cases,
            &iter_differential_cases,
            &iter_adversarial_cases,
            &io_differential_cases,
            &io_adversarial_cases,
            &linalg_differential_cases,
            &linalg_adversarial_cases,
        )?;
        let hardened = execute_mode(
            scenario,
            RuntimeMode::Hardened,
            &ufunc_cases,
            &shape_stride_cases,
            &policy_cases,
            &wire_cases,
            &iter_differential_cases,
            &iter_adversarial_cases,
            &io_differential_cases,
            &io_adversarial_cases,
            &linalg_differential_cases,
            &linalg_adversarial_cases,
        )?;

        let strict_expected = scenario.strict.expected_status.trim().to_lowercase();
        record_check(
            &mut report,
            EXPECTED_SCENARIO_STATUSES.contains(&strict_expected.as_str()),
            format!(
                "{}: strict.expected_status '{}' must be one of {}",
                scenario.id,
                strict_expected,
                EXPECTED_SCENARIO_STATUSES.join(", ")
            ),
        );
        record_check(
            &mut report,
            strict.actual_status == strict_expected && strict.failures.is_empty(),
            format!(
                "{}: strict mode status mismatch expected={} actual={} step_failures={:?}",
                scenario.id, strict_expected, strict.actual_status, strict.failures
            ),
        );

        let hardened_expected = scenario.hardened.expected_status.trim().to_lowercase();
        record_check(
            &mut report,
            EXPECTED_SCENARIO_STATUSES.contains(&hardened_expected.as_str()),
            format!(
                "{}: hardened.expected_status '{}' must be one of {}",
                scenario.id,
                hardened_expected,
                EXPECTED_SCENARIO_STATUSES.join(", ")
            ),
        );
        record_check(
            &mut report,
            hardened.actual_status == hardened_expected && hardened.failures.is_empty(),
            format!(
                "{}: hardened mode status mismatch expected={} actual={} step_failures={:?}",
                scenario.id, hardened_expected, hardened.actual_status, hardened.failures
            ),
        );
    }

    for required in REQUIRED_SCENARIO_CATEGORIES {
        record_check(
            &mut report,
            categories.contains(*required),
            format!(
                "workflow_scenario_corpus missing required category {}",
                required
            ),
        );
    }

    Ok(report)
}

fn load_ufunc_case_map(path: &Path) -> Result<BTreeMap<String, UFuncInputCase>, String> {
    let cases = load_input_cases(path)?;
    let mut map = BTreeMap::new();
    for case in cases {
        let id = case.id.clone();
        if map.insert(id.clone(), case).is_some() {
            return Err(format!("duplicate ufunc input fixture id: {id}"));
        }
    }
    Ok(map)
}

fn load_runtime_policy_case_map(
    path: &Path,
) -> Result<BTreeMap<String, PolicyFixtureCase>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<PolicyFixtureCase> = serde_json::from_str(&raw)
        .map_err(|err| format!("invalid json {}: {err}", path.display()))?;
    let mut map = BTreeMap::new();
    for case in cases {
        if map.insert(case.id.clone(), case).is_some() {
            return Err(format!(
                "duplicate runtime policy fixture id in {}",
                path.display()
            ));
        }
    }
    Ok(map)
}

fn load_runtime_policy_wire_case_map(
    path: &Path,
) -> Result<BTreeMap<String, PolicyWireFixtureCase>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<PolicyWireFixtureCase> = serde_json::from_str(&raw)
        .map_err(|err| format!("invalid json {}: {err}", path.display()))?;
    let mut map = BTreeMap::new();
    for case in cases {
        if map.insert(case.id.clone(), case).is_some() {
            return Err(format!(
                "duplicate runtime policy adversarial fixture id in {}",
                path.display()
            ));
        }
    }
    Ok(map)
}

fn load_iter_differential_case_map(
    fixture_root: &Path,
) -> Result<BTreeMap<String, crate::IterDifferentialCase>, String> {
    let cases = crate::load_iter_differential_cases(fixture_root)?;
    let mut map = BTreeMap::new();
    for case in cases {
        if map.insert(case.id.clone(), case).is_some() {
            return Err(format!(
                "duplicate iter differential fixture id in {}",
                fixture_root.join("iter_differential_cases.json").display()
            ));
        }
    }
    Ok(map)
}

fn load_iter_adversarial_case_map(
    fixture_root: &Path,
) -> Result<BTreeMap<String, crate::IterAdversarialCase>, String> {
    let cases = crate::load_iter_adversarial_cases(fixture_root)?;
    let mut map = BTreeMap::new();
    for case in cases {
        if map.insert(case.id.clone(), case).is_some() {
            return Err(format!(
                "duplicate iter adversarial fixture id in {}",
                fixture_root.join("iter_adversarial_cases.json").display()
            ));
        }
    }
    Ok(map)
}

fn load_io_differential_case_map(
    fixture_root: &Path,
) -> Result<BTreeMap<String, crate::IoDifferentialCase>, String> {
    let path = fixture_root.join("io_differential_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<crate::IoDifferentialCase> = serde_json::from_str(&raw)
        .map_err(|err| format!("invalid json {}: {err}", path.display()))?;
    let mut map = BTreeMap::new();
    for case in cases {
        if map.insert(case.id.clone(), case).is_some() {
            return Err(format!(
                "duplicate io differential fixture id in {}",
                path.display()
            ));
        }
    }
    Ok(map)
}

fn load_io_adversarial_case_map(
    fixture_root: &Path,
) -> Result<BTreeMap<String, crate::IoAdversarialCase>, String> {
    let path = fixture_root.join("io_adversarial_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<crate::IoAdversarialCase> = serde_json::from_str(&raw)
        .map_err(|err| format!("invalid json {}: {err}", path.display()))?;
    let mut map = BTreeMap::new();
    for case in cases {
        if map.insert(case.id.clone(), case).is_some() {
            return Err(format!(
                "duplicate io adversarial fixture id in {}",
                path.display()
            ));
        }
    }
    Ok(map)
}

fn load_linalg_differential_case_map(
    fixture_root: &Path,
) -> Result<BTreeMap<String, crate::LinalgDifferentialCase>, String> {
    let path = fixture_root.join("linalg_differential_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<crate::LinalgDifferentialCase> = serde_json::from_str(&raw)
        .map_err(|err| format!("invalid json {}: {err}", path.display()))?;
    let mut map = BTreeMap::new();
    for case in cases {
        if map.insert(case.id.clone(), case).is_some() {
            return Err(format!(
                "duplicate linalg differential fixture id in {}",
                path.display()
            ));
        }
    }
    Ok(map)
}

fn load_linalg_adversarial_case_map(
    fixture_root: &Path,
) -> Result<BTreeMap<String, crate::LinalgAdversarialCase>, String> {
    let path = fixture_root.join("linalg_adversarial_cases.json");
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<crate::LinalgAdversarialCase> = serde_json::from_str(&raw)
        .map_err(|err| format!("invalid json {}: {err}", path.display()))?;
    let mut map = BTreeMap::new();
    for case in cases {
        if map.insert(case.id.clone(), case).is_some() {
            return Err(format!(
                "duplicate linalg adversarial fixture id in {}",
                path.display()
            ));
        }
    }
    Ok(map)
}

fn load_shape_stride_case_map(
    fixture_root: &Path,
) -> Result<BTreeMap<String, crate::ShapeStrideFixtureCase>, String> {
    let cases = crate::load_shape_stride_cases(fixture_root)?;
    let mut map = BTreeMap::new();
    for case in cases {
        if map.insert(case.id.clone(), case).is_some() {
            return Err(format!(
                "duplicate shape_stride fixture id in {}",
                fixture_root.join("shape_stride_cases.json").display()
            ));
        }
    }
    Ok(map)
}

fn derive_repo_root(fixture_root: &Path) -> Result<PathBuf, String> {
    fixture_root
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| {
            format!(
                "unable to derive repository root from fixture_root {}",
                fixture_root.display()
            )
        })
}

#[allow(clippy::too_many_arguments)]
fn execute_mode(
    scenario: &WorkflowScenarioCase,
    mode: RuntimeMode,
    ufunc_cases: &BTreeMap<String, UFuncInputCase>,
    shape_stride_cases: &BTreeMap<String, crate::ShapeStrideFixtureCase>,
    policy_cases: &BTreeMap<String, PolicyFixtureCase>,
    wire_cases: &BTreeMap<String, PolicyWireFixtureCase>,
    iter_differential_cases: &BTreeMap<String, crate::IterDifferentialCase>,
    iter_adversarial_cases: &BTreeMap<String, crate::IterAdversarialCase>,
    io_differential_cases: &BTreeMap<String, crate::IoDifferentialCase>,
    io_adversarial_cases: &BTreeMap<String, crate::IoAdversarialCase>,
    linalg_differential_cases: &BTreeMap<String, crate::LinalgDifferentialCase>,
    linalg_adversarial_cases: &BTreeMap<String, crate::LinalgAdversarialCase>,
) -> Result<ModeExecution, String> {
    let mut failures = Vec::new();
    let mut saw_fail_closed = false;
    let mode_name = mode.as_str().to_string();

    for step in &scenario.steps {
        match step {
            WorkflowStep::UfuncInput {
                id,
                case_id,
                expect_error_contains,
            } => {
                let mut passed = false;
                let expected = expect_error_contains.as_ref().map_or_else(
                    || "ok".to_string(),
                    |needle| format!("error_contains:{needle}"),
                );
                let mut actual = "missing_case".to_string();
                let mut detail = String::new();

                if let Some(case) = ufunc_cases.get(case_id.as_str()) {
                    match execute_input_case(case) {
                        Ok((shape, _values, dtype)) => {
                            actual = format!("ok shape={shape:?} dtype={dtype}");
                            if let Some(needle) = expect_error_contains {
                                detail = format!(
                                    "expected error containing '{}' but execution succeeded",
                                    needle
                                );
                            } else {
                                passed = true;
                            }
                        }
                        Err(err) => {
                            actual = format!("error:{err}");
                            if let Some(needle) = expect_error_contains {
                                if err.to_lowercase().contains(&needle.to_lowercase()) {
                                    passed = true;
                                } else {
                                    detail = format!(
                                        "expected error containing '{}' but got '{}'",
                                        needle, err
                                    );
                                }
                            } else {
                                detail = format!("unexpected execution error: {err}");
                            }
                        }
                    }
                } else {
                    detail = format!("ufunc case id '{}' not found", case_id);
                }

                let fixture_id = format!("{}::{}", scenario.id, id);
                let entry = WorkflowScenarioLogEntry {
                    suite: "workflow_scenarios",
                    fixture_id: fixture_id.clone(),
                    seed: scenario.seed,
                    mode: mode_name.clone(),
                    env_fingerprint: scenario.env_fingerprint.clone(),
                    artifact_refs: scenario.artifact_refs.clone(),
                    reason_code: scenario.reason_code.clone(),
                    scenario_id: scenario.id.clone(),
                    step_id: id.clone(),
                    step_kind: "ufunc_input".to_string(),
                    expected,
                    actual,
                    passed,
                    detail: detail.clone(),
                };
                maybe_append_workflow_log(&entry)?;

                if !passed {
                    failures.push(format!("{fixture_id}: {detail}"));
                }
            }
            WorkflowStep::ShapeStrideCase { id, case_id } => {
                let mut passed = false;
                let expected = "pass".to_string();
                let actual: String;
                let mut detail = String::new();

                if let Some(case) = shape_stride_cases.get(case_id.as_str()) {
                    let (case_passed, case_failures) = crate::evaluate_shape_stride_case(case);
                    passed = case_passed;
                    if passed {
                        actual = "pass".to_string();
                    } else if !case_failures.is_empty() {
                        detail = case_failures.join("; ");
                        actual = detail.clone();
                    } else {
                        detail = "shape_stride validation failed".to_string();
                        actual = detail.clone();
                    }
                } else {
                    detail = format!("shape/stride case id '{}' not found", case_id);
                    actual = detail.clone();
                }

                let fixture_id = format!("{}::{}", scenario.id, id);
                let entry = WorkflowScenarioLogEntry {
                    suite: "workflow_scenarios",
                    fixture_id: fixture_id.clone(),
                    seed: scenario.seed,
                    mode: mode_name.clone(),
                    env_fingerprint: scenario.env_fingerprint.clone(),
                    artifact_refs: scenario.artifact_refs.clone(),
                    reason_code: scenario.reason_code.clone(),
                    scenario_id: scenario.id.clone(),
                    step_id: id.clone(),
                    step_kind: "shape_stride_case".to_string(),
                    expected,
                    actual,
                    passed,
                    detail: detail.clone(),
                };
                maybe_append_workflow_log(&entry)?;

                if !passed {
                    let failure_detail = if detail.is_empty() {
                        "shape_stride validation failed".to_string()
                    } else {
                        detail.clone()
                    };
                    failures.push(format!("{fixture_id}: {failure_detail}"));
                }
            }
            WorkflowStep::RuntimePolicy {
                id,
                case_id,
                expected_action_strict,
                expected_action_hardened,
            } => {
                let expected_raw = match mode {
                    RuntimeMode::Strict => expected_action_strict,
                    RuntimeMode::Hardened => expected_action_hardened,
                };
                let expected_action = parse_action(expected_raw)?;
                let mut passed = false;
                let mut actual = "missing_case".to_string();
                let mut detail = String::new();

                if let Some(case) = policy_cases.get(case_id.as_str()) {
                    let class = CompatibilityClass::from_wire(&case.class);
                    let action = decide_compatibility(mode, class, case.risk_score, case.threshold);
                    actual = action.as_str().to_string();
                    passed = action == expected_action;
                    if matches!(action, DecisionAction::FailClosed) {
                        saw_fail_closed = true;
                    }
                    if !passed {
                        detail = format!(
                            "expected action={} actual={}",
                            expected_action.as_str(),
                            action.as_str()
                        );
                    }
                } else {
                    detail = format!("runtime policy case id '{}' not found", case_id);
                }

                let fixture_id = format!("{}::{}", scenario.id, id);
                let entry = WorkflowScenarioLogEntry {
                    suite: "workflow_scenarios",
                    fixture_id: fixture_id.clone(),
                    seed: scenario.seed,
                    mode: mode_name.clone(),
                    env_fingerprint: scenario.env_fingerprint.clone(),
                    artifact_refs: scenario.artifact_refs.clone(),
                    reason_code: scenario.reason_code.clone(),
                    scenario_id: scenario.id.clone(),
                    step_id: id.clone(),
                    step_kind: "runtime_policy".to_string(),
                    expected: expected_action.as_str().to_string(),
                    actual,
                    passed,
                    detail: detail.clone(),
                };
                maybe_append_workflow_log(&entry)?;

                if !passed {
                    failures.push(format!("{fixture_id}: {detail}"));
                }
            }
            WorkflowStep::RuntimePolicyWire {
                id,
                case_id,
                expected_action_strict,
                expected_action_hardened,
            } => {
                let expected_raw = match mode {
                    RuntimeMode::Strict => expected_action_strict,
                    RuntimeMode::Hardened => expected_action_hardened,
                };
                let expected_action = parse_action(expected_raw)?;
                let mut passed = false;
                let mut actual = "missing_case".to_string();
                let mut detail = String::new();

                if let Some(case) = wire_cases.get(case_id.as_str()) {
                    let action = decide_compatibility_from_wire(
                        &case.mode_raw,
                        &case.class_raw,
                        case.risk_score,
                        case.threshold,
                    );
                    actual = action.as_str().to_string();
                    passed = action == expected_action;
                    if matches!(action, DecisionAction::FailClosed) {
                        saw_fail_closed = true;
                    }
                    if !passed {
                        detail = format!(
                            "expected action={} actual={}",
                            expected_action.as_str(),
                            action.as_str()
                        );
                    }
                } else {
                    detail = format!("runtime policy wire case id '{}' not found", case_id);
                }

                let fixture_id = format!("{}::{}", scenario.id, id);
                let entry = WorkflowScenarioLogEntry {
                    suite: "workflow_scenarios",
                    fixture_id: fixture_id.clone(),
                    seed: scenario.seed,
                    mode: mode_name.clone(),
                    env_fingerprint: scenario.env_fingerprint.clone(),
                    artifact_refs: scenario.artifact_refs.clone(),
                    reason_code: scenario.reason_code.clone(),
                    scenario_id: scenario.id.clone(),
                    step_id: id.clone(),
                    step_kind: "runtime_policy_wire".to_string(),
                    expected: expected_action.as_str().to_string(),
                    actual,
                    passed,
                    detail: detail.clone(),
                };
                maybe_append_workflow_log(&entry)?;

                if !passed {
                    failures.push(format!("{fixture_id}: {detail}"));
                }
            }
            WorkflowStep::IterFixtureCase {
                id,
                case_id,
                fixture_set,
            } => {
                let fixture_id = format!("{}::{}", scenario.id, id);
                let (expected, actual, passed, detail) = match fixture_set.as_str() {
                    "differential" => {
                        if let Some(case) = iter_differential_cases.get(case_id.as_str()) {
                            let result = execute_iter_differential_step(case);
                            (result.expected, result.actual, result.passed, result.detail)
                        } else {
                            (
                                String::new(),
                                "missing_case".to_string(),
                                false,
                                format!("iter differential case id '{}' not found", case_id),
                            )
                        }
                    }
                    "adversarial" => {
                        if let Some(case) = iter_adversarial_cases.get(case_id.as_str()) {
                            let result = execute_iter_adversarial_step(case);
                            (result.expected, result.actual, result.passed, result.detail)
                        } else {
                            (
                                String::new(),
                                "missing_case".to_string(),
                                false,
                                format!("iter adversarial case id '{}' not found", case_id),
                            )
                        }
                    }
                    other => (
                        String::new(),
                        "unsupported_fixture_set".to_string(),
                        false,
                        format!(
                            "unsupported iter fixture_set '{}' (expected differential|adversarial)",
                            other
                        ),
                    ),
                };

                let entry = WorkflowScenarioLogEntry {
                    suite: "workflow_scenarios",
                    fixture_id: fixture_id.clone(),
                    seed: scenario.seed,
                    mode: mode_name.clone(),
                    env_fingerprint: scenario.env_fingerprint.clone(),
                    artifact_refs: scenario.artifact_refs.clone(),
                    reason_code: scenario.reason_code.clone(),
                    scenario_id: scenario.id.clone(),
                    step_id: id.clone(),
                    step_kind: "iter_fixture_case".to_string(),
                    expected,
                    actual,
                    passed,
                    detail: detail.clone(),
                };
                maybe_append_workflow_log(&entry)?;

                if !passed {
                    let failure_detail = if detail.is_empty() {
                        "iterator fixture step failed".to_string()
                    } else {
                        detail
                    };
                    failures.push(format!("{fixture_id}: {failure_detail}"));
                }
            }
            WorkflowStep::IoFixtureCase {
                id,
                case_id,
                fixture_set,
            } => {
                let fixture_id = format!("{}::{}", scenario.id, id);
                let (expected, actual, passed, detail) = match fixture_set.as_str() {
                    "differential" => {
                        if let Some(case) = io_differential_cases.get(case_id.as_str()) {
                            let result = execute_io_differential_step(case);
                            (result.expected, result.actual, result.passed, result.detail)
                        } else {
                            (
                                String::new(),
                                "missing_case".to_string(),
                                false,
                                format!("io differential case id '{}' not found", case_id),
                            )
                        }
                    }
                    "adversarial" => {
                        if let Some(case) = io_adversarial_cases.get(case_id.as_str()) {
                            let result = execute_io_adversarial_step(case);
                            (result.expected, result.actual, result.passed, result.detail)
                        } else {
                            (
                                String::new(),
                                "missing_case".to_string(),
                                false,
                                format!("io adversarial case id '{}' not found", case_id),
                            )
                        }
                    }
                    other => (
                        String::new(),
                        "unsupported_fixture_set".to_string(),
                        false,
                        format!(
                            "unsupported io fixture_set '{}' (expected differential|adversarial)",
                            other
                        ),
                    ),
                };

                let entry = WorkflowScenarioLogEntry {
                    suite: "workflow_scenarios",
                    fixture_id: fixture_id.clone(),
                    seed: scenario.seed,
                    mode: mode_name.clone(),
                    env_fingerprint: scenario.env_fingerprint.clone(),
                    artifact_refs: scenario.artifact_refs.clone(),
                    reason_code: scenario.reason_code.clone(),
                    scenario_id: scenario.id.clone(),
                    step_id: id.clone(),
                    step_kind: "io_fixture_case".to_string(),
                    expected,
                    actual,
                    passed,
                    detail: detail.clone(),
                };
                maybe_append_workflow_log(&entry)?;

                if !passed {
                    let failure_detail = if detail.is_empty() {
                        "io fixture step failed".to_string()
                    } else {
                        detail
                    };
                    failures.push(format!("{fixture_id}: {failure_detail}"));
                }
            }
            WorkflowStep::LinalgFixtureCase {
                id,
                case_id,
                fixture_set,
            } => {
                let fixture_id = format!("{}::{}", scenario.id, id);
                let (expected, actual, passed, detail) = match fixture_set.as_str() {
                    "differential" => {
                        if let Some(case) = linalg_differential_cases.get(case_id.as_str()) {
                            let result = execute_linalg_differential_step(case);
                            (result.expected, result.actual, result.passed, result.detail)
                        } else {
                            (
                                String::new(),
                                "missing_case".to_string(),
                                false,
                                format!("linalg differential case id '{}' not found", case_id),
                            )
                        }
                    }
                    "adversarial" => {
                        if let Some(case) = linalg_adversarial_cases.get(case_id.as_str()) {
                            let result = execute_linalg_adversarial_step(case);
                            (result.expected, result.actual, result.passed, result.detail)
                        } else {
                            (
                                String::new(),
                                "missing_case".to_string(),
                                false,
                                format!("linalg adversarial case id '{}' not found", case_id),
                            )
                        }
                    }
                    other => (
                        String::new(),
                        "unsupported_fixture_set".to_string(),
                        false,
                        format!(
                            "unsupported linalg fixture_set '{}' (expected differential|adversarial)",
                            other
                        ),
                    ),
                };

                let entry = WorkflowScenarioLogEntry {
                    suite: "workflow_scenarios",
                    fixture_id: fixture_id.clone(),
                    seed: scenario.seed,
                    mode: mode_name.clone(),
                    env_fingerprint: scenario.env_fingerprint.clone(),
                    artifact_refs: scenario.artifact_refs.clone(),
                    reason_code: scenario.reason_code.clone(),
                    scenario_id: scenario.id.clone(),
                    step_id: id.clone(),
                    step_kind: "linalg_fixture_case".to_string(),
                    expected,
                    actual,
                    passed,
                    detail: detail.clone(),
                };
                maybe_append_workflow_log(&entry)?;

                if !passed {
                    let failure_detail = if detail.is_empty() {
                        "linalg fixture step failed".to_string()
                    } else {
                        detail
                    };
                    failures.push(format!("{fixture_id}: {failure_detail}"));
                }
            }
        }
    }

    let actual_status = if failures.is_empty() {
        if saw_fail_closed {
            "fail_closed".to_string()
        } else {
            "pass".to_string()
        }
    } else {
        "fail".to_string()
    };

    Ok(ModeExecution {
        actual_status,
        failures,
    })
}

fn flat_index_len_hint(index: &crate::IterFlatIndexFixtureInput) -> usize {
    index
        .stop
        .max(index.mask.len())
        .max(index.index.saturating_add(1))
}

fn execute_iter_differential_step(case: &crate::IterDifferentialCase) -> IterStepResult {
    let reason_code = crate::normalize_reason_code(&case.reason_code);
    let expected_reason_code = if case.expected_reason_code.trim().is_empty() {
        reason_code.clone()
    } else {
        case.expected_reason_code.trim().to_string()
    };
    let expected_error = case.expected_error_contains.to_lowercase();
    let flat_len = case
        .len
        .unwrap_or_else(|| case.flat_index.as_ref().map_or(0, flat_index_len_hint));

    if expected_error.is_empty() {
        match crate::execute_iter_operation_with_len(
            &case.id,
            &case.operation,
            case.selector.as_ref(),
            case.overlap.as_ref(),
            case.flat_index.as_ref(),
            case.flags.as_ref(),
            flat_len,
            case.values_len,
        ) {
            Ok(outcome) => match crate::validate_iter_success_expectations(case, &outcome) {
                Ok(()) => IterStepResult {
                    expected: format!("ok reason_code={expected_reason_code}"),
                    actual: format!("ok reason_code={reason_code}"),
                    passed: reason_code == expected_reason_code,
                    detail: if reason_code == expected_reason_code {
                        String::new()
                    } else {
                        format!(
                            "success reason-code mismatch expected={} actual={}",
                            expected_reason_code, reason_code
                        )
                    },
                },
                Err(err) => IterStepResult {
                    expected: format!("ok reason_code={expected_reason_code}"),
                    actual: format!("error:{err}"),
                    passed: false,
                    detail: err,
                },
            },
            Err(err) => IterStepResult {
                expected: format!("ok reason_code={expected_reason_code}"),
                actual: format!("error:{} reason_code={}", err.message, err.reason_code),
                passed: false,
                detail: format!(
                    "expected successful execution but operation '{}' failed",
                    case.operation
                ),
            },
        }
    } else {
        match crate::execute_iter_operation_with_len(
            &case.id,
            &case.operation,
            case.selector.as_ref(),
            case.overlap.as_ref(),
            case.flat_index.as_ref(),
            case.flags.as_ref(),
            flat_len,
            case.values_len,
        ) {
            Ok(_) => IterStepResult {
                expected: format!(
                    "error_contains:{} reason_code={}",
                    case.expected_error_contains, expected_reason_code
                ),
                actual: format!("ok reason_code={reason_code}"),
                passed: false,
                detail: format!(
                    "expected error containing '{}' but operation '{}' succeeded",
                    case.expected_error_contains, case.operation
                ),
            },
            Err(err) => {
                let contains_expected = err.message.to_lowercase().contains(&expected_error);
                let reason_match = err.reason_code == expected_reason_code;
                let passed = contains_expected && reason_match;
                let detail = if passed {
                    String::new()
                } else {
                    format!(
                        "expected error containing '{}' with reason_code='{}' but got '{}' (actual_reason_code='{}')",
                        case.expected_error_contains,
                        expected_reason_code,
                        err.message,
                        err.reason_code
                    )
                };
                IterStepResult {
                    expected: format!(
                        "error_contains:{} reason_code={}",
                        case.expected_error_contains, expected_reason_code
                    ),
                    actual: format!("error:{} reason_code={}", err.message, err.reason_code),
                    passed,
                    detail,
                }
            }
        }
    }
}

fn execute_iter_adversarial_step(case: &crate::IterAdversarialCase) -> IterStepResult {
    let reason_code = crate::normalize_reason_code(&case.reason_code);
    let expected_reason_code = if case.expected_reason_code.trim().is_empty() {
        reason_code
    } else {
        case.expected_reason_code.trim().to_string()
    };
    let expected_error = case.expected_error_contains.to_lowercase();
    let flat_len = case
        .len
        .unwrap_or_else(|| case.flat_index.as_ref().map_or(0, flat_index_len_hint));

    match crate::execute_iter_operation_with_len(
        &case.id,
        &case.operation,
        case.selector.as_ref(),
        case.overlap.as_ref(),
        case.flat_index.as_ref(),
        case.flags.as_ref(),
        flat_len,
        case.values_len,
    ) {
        Ok(_) => IterStepResult {
            expected: format!(
                "error_contains:{} reason_code={}",
                case.expected_error_contains, expected_reason_code
            ),
            actual: "ok".to_string(),
            passed: false,
            detail: format!(
                "expected error containing '{}' but operation '{}' succeeded",
                case.expected_error_contains, case.operation
            ),
        },
        Err(err) => {
            let contains_expected = err.message.to_lowercase().contains(&expected_error);
            let reason_match = err.reason_code == expected_reason_code;
            let passed = contains_expected && reason_match;
            let detail = if passed {
                String::new()
            } else {
                format!(
                    "expected error containing '{}' with reason_code='{}' but got '{}' (actual_reason_code='{}')",
                    case.expected_error_contains,
                    expected_reason_code,
                    err.message,
                    err.reason_code
                )
            };
            IterStepResult {
                expected: format!(
                    "error_contains:{} reason_code={}",
                    case.expected_error_contains, expected_reason_code
                ),
                actual: format!("error:{} reason_code={}", err.message, err.reason_code),
                passed,
                detail,
            }
        }
    }
}

fn execute_io_differential_step(case: &crate::IoDifferentialCase) -> IterStepResult {
    let reason_code = crate::normalize_reason_code(&case.reason_code);
    let expected_reason_code = if case.expected_reason_code.trim().is_empty() {
        reason_code.clone()
    } else {
        case.expected_reason_code.trim().to_string()
    };
    let expected_error = case.expected_error_contains.to_lowercase();

    if expected_error.is_empty() {
        match crate::execute_io_differential_operation(case) {
            Ok(outcome) => {
                match crate::validate_io_differential_success_expectations(case, &outcome) {
                    Ok(()) => IterStepResult {
                        expected: format!("ok reason_code={expected_reason_code}"),
                        actual: format!("ok reason_code={reason_code}"),
                        passed: reason_code == expected_reason_code,
                        detail: if reason_code == expected_reason_code {
                            String::new()
                        } else {
                            format!(
                                "success reason-code mismatch expected={} actual={}",
                                expected_reason_code, reason_code
                            )
                        },
                    },
                    Err(err) => IterStepResult {
                        expected: format!("ok reason_code={expected_reason_code}"),
                        actual: format!("error:{err}"),
                        passed: false,
                        detail: err,
                    },
                }
            }
            Err(err) => IterStepResult {
                expected: format!("ok reason_code={expected_reason_code}"),
                actual: format!("error:{} reason_code={}", err.message, err.reason_code),
                passed: false,
                detail: format!(
                    "expected successful execution but operation '{}' failed",
                    case.operation
                ),
            },
        }
    } else {
        match crate::execute_io_differential_operation(case) {
            Ok(_) => IterStepResult {
                expected: format!(
                    "error_contains:{} reason_code={}",
                    case.expected_error_contains, expected_reason_code
                ),
                actual: format!("ok reason_code={reason_code}"),
                passed: false,
                detail: format!(
                    "expected error containing '{}' but operation '{}' succeeded",
                    case.expected_error_contains, case.operation
                ),
            },
            Err(err) => {
                let contains_expected = err.message.to_lowercase().contains(&expected_error);
                let reason_match = err.reason_code == expected_reason_code;
                let passed = contains_expected && reason_match;
                let detail = if passed {
                    String::new()
                } else {
                    format!(
                        "expected error containing '{}' with reason_code='{}' but got '{}' (actual_reason_code='{}')",
                        case.expected_error_contains,
                        expected_reason_code,
                        err.message,
                        err.reason_code
                    )
                };
                IterStepResult {
                    expected: format!(
                        "error_contains:{} reason_code={}",
                        case.expected_error_contains, expected_reason_code
                    ),
                    actual: format!("error:{} reason_code={}", err.message, err.reason_code),
                    passed,
                    detail,
                }
            }
        }
    }
}

fn execute_io_adversarial_step(case: &crate::IoAdversarialCase) -> IterStepResult {
    let reason_code = crate::normalize_reason_code(&case.reason_code);
    let expected_reason_code = if case.expected_reason_code.trim().is_empty() {
        reason_code
    } else {
        case.expected_reason_code.trim().to_string()
    };
    let expected_error = case.expected_error_contains.to_lowercase();

    match crate::execute_io_adversarial_operation(case) {
        Ok(()) => IterStepResult {
            expected: format!(
                "error_contains:{} reason_code={}",
                case.expected_error_contains, expected_reason_code
            ),
            actual: "ok".to_string(),
            passed: false,
            detail: format!(
                "expected error containing '{}' but operation '{}' succeeded",
                case.expected_error_contains, case.operation
            ),
        },
        Err(err) => {
            let contains_expected = err.message.to_lowercase().contains(&expected_error);
            let reason_match = err.reason_code == expected_reason_code;
            let passed = contains_expected && reason_match;
            let detail = if passed {
                String::new()
            } else {
                format!(
                    "expected error containing '{}' with reason_code='{}' but got '{}' (actual_reason_code='{}')",
                    case.expected_error_contains,
                    expected_reason_code,
                    err.message,
                    err.reason_code
                )
            };
            IterStepResult {
                expected: format!(
                    "error_contains:{} reason_code={}",
                    case.expected_error_contains, expected_reason_code
                ),
                actual: format!("error:{} reason_code={}", err.message, err.reason_code),
                passed,
                detail,
            }
        }
    }
}

fn execute_linalg_differential_step(case: &crate::LinalgDifferentialCase) -> IterStepResult {
    let reason_code = crate::normalize_reason_code(&case.reason_code);
    let expected_reason_code = if case.expected_reason_code.trim().is_empty() {
        reason_code.clone()
    } else {
        case.expected_reason_code.trim().to_string()
    };
    let expected_error = case.expected_error_contains.to_lowercase();

    if expected_error.is_empty() {
        match crate::execute_linalg_differential_operation(case) {
            Ok(outcome) => match crate::validate_linalg_differential_expectation(case, &outcome) {
                Ok(()) => IterStepResult {
                    expected: format!("ok reason_code={expected_reason_code}"),
                    actual: format!("ok reason_code={reason_code}"),
                    passed: reason_code == expected_reason_code,
                    detail: if reason_code == expected_reason_code {
                        String::new()
                    } else {
                        format!(
                            "success reason-code mismatch expected={} actual={}",
                            expected_reason_code, reason_code
                        )
                    },
                },
                Err(err) => IterStepResult {
                    expected: format!("ok reason_code={expected_reason_code}"),
                    actual: format!("error:{err}"),
                    passed: false,
                    detail: err,
                },
            },
            Err(err) => IterStepResult {
                expected: format!("ok reason_code={expected_reason_code}"),
                actual: format!("error:{err} reason_code={}", err.reason_code()),
                passed: false,
                detail: format!(
                    "expected successful execution but operation '{}' failed",
                    case.operation
                ),
            },
        }
    } else {
        match crate::execute_linalg_differential_operation(case) {
            Ok(_) => IterStepResult {
                expected: format!(
                    "error_contains:{} reason_code={}",
                    case.expected_error_contains, expected_reason_code
                ),
                actual: format!("ok reason_code={reason_code}"),
                passed: false,
                detail: format!(
                    "expected error containing '{}' but operation '{}' succeeded",
                    case.expected_error_contains, case.operation
                ),
            },
            Err(err) => {
                let actual_reason_code = err.reason_code().to_string();
                let contains_expected = err.to_string().to_lowercase().contains(&expected_error);
                let reason_match = actual_reason_code == expected_reason_code;
                let passed = contains_expected && reason_match;
                let detail = if passed {
                    String::new()
                } else {
                    format!(
                        "expected error containing '{}' with reason_code='{}' but got '{}' (actual_reason_code='{}')",
                        case.expected_error_contains, expected_reason_code, err, actual_reason_code
                    )
                };
                IterStepResult {
                    expected: format!(
                        "error_contains:{} reason_code={}",
                        case.expected_error_contains, expected_reason_code
                    ),
                    actual: format!("error:{err} reason_code={actual_reason_code}"),
                    passed,
                    detail,
                }
            }
        }
    }
}

fn execute_linalg_adversarial_step(case: &crate::LinalgAdversarialCase) -> IterStepResult {
    let reason_code = crate::normalize_reason_code(&case.reason_code);
    let expected_reason_code = if case.expected_reason_code.trim().is_empty() {
        reason_code
    } else {
        case.expected_reason_code.trim().to_string()
    };
    let expected_error = case.expected_error_contains.to_lowercase();

    match crate::execute_linalg_adversarial_operation(case) {
        Ok(_) => IterStepResult {
            expected: format!(
                "error_contains:{} reason_code={}",
                case.expected_error_contains, expected_reason_code
            ),
            actual: "ok".to_string(),
            passed: false,
            detail: format!(
                "expected error containing '{}' but operation '{}' succeeded",
                case.expected_error_contains, case.operation
            ),
        },
        Err(err) => {
            let actual_reason_code = err.reason_code().to_string();
            let contains_expected = err.to_string().to_lowercase().contains(&expected_error);
            let reason_match = actual_reason_code == expected_reason_code;
            let passed = contains_expected && reason_match;
            let detail = if passed {
                String::new()
            } else {
                format!(
                    "expected error containing '{}' with reason_code='{}' but got '{}' (actual_reason_code='{}')",
                    case.expected_error_contains, expected_reason_code, err, actual_reason_code
                )
            };
            IterStepResult {
                expected: format!(
                    "error_contains:{} reason_code={}",
                    case.expected_error_contains, expected_reason_code
                ),
                actual: format!("error:{err} reason_code={actual_reason_code}"),
                passed,
                detail,
            }
        }
    }
}

fn parse_action(raw: &str) -> Result<DecisionAction, String> {
    match raw.trim() {
        "allow" => Ok(DecisionAction::Allow),
        "full_validate" => Ok(DecisionAction::FullValidate),
        "fail_closed" => Ok(DecisionAction::FailClosed),
        other => Err(format!("invalid expected decision action: {other}")),
    }
}

fn maybe_append_workflow_log(entry: &WorkflowScenarioLogEntry) -> Result<(), String> {
    let configured = WORKFLOW_SCENARIO_LOG_PATH
        .get()
        .and_then(|cell| cell.lock().ok())
        .and_then(|slot| slot.clone());
    let from_env = std::env::var_os("FNP_WORKFLOW_SCENARIO_LOG_PATH").map(PathBuf::from);
    let Some(path) = configured.or(from_env) else {
        if workflow_scenario_log_required() {
            return Err(
                "workflow scenario log path is required but unset; configure --log-path or FNP_WORKFLOW_SCENARIO_LOG_PATH".to_string(),
            );
        }
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
        .map_err(|err| format!("failed serializing workflow scenario log entry: {err}"))?;
    let mut payload = line.into_bytes();
    payload.push(b'\n');
    file.write_all(&payload).map_err(|err| {
        format!(
            "failed appending workflow scenario log {}: {err}",
            path.display()
        )
    })
}

fn record_check(report: &mut SuiteReport, passed: bool, failure: String) {
    report.case_count += 1;
    if passed {
        report.pass_count += 1;
    } else {
        report.failures.push(failure);
    }
}

#[cfg(test)]
mod tests {
    use crate::HarnessConfig;

    #[test]
    fn workflow_scenario_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let report = super::run_user_workflow_scenario_suite(&cfg)
            .expect("workflow scenario suite should run");
        assert!(report.all_passed(), "failures={:?}", report.failures);
    }
}
