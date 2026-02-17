#![forbid(unsafe_code)]

use fnp_conformance::workflow_scenarios::{
    run_user_workflow_scenario_suite, set_workflow_scenario_log_path,
    set_workflow_scenario_log_required,
};
use fnp_conformance::{HarnessConfig, SuiteReport};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize)]
struct SuiteSummary {
    suite: String,
    case_count: usize,
    pass_count: usize,
    failures: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct AttemptSummary {
    attempt: usize,
    status: String,
    workflow_log: String,
    suite: SuiteSummary,
}

#[derive(Debug, Clone, Serialize)]
struct ReliabilityDiagnostic {
    subsystem: String,
    reason_code: String,
    message: String,
    evidence_refs: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ReliabilitySummary {
    retries: usize,
    attempts_run: usize,
    flaky_failures: usize,
    flake_budget: usize,
    coverage_floor: f64,
    coverage_ratio: f64,
    diagnostics: Vec<ReliabilityDiagnostic>,
}

#[derive(Debug, Serialize)]
struct GateSummary {
    status: &'static str,
    workflow_log: String,
    replay_command: String,
    attempts: Vec<AttemptSummary>,
    suites: Vec<SuiteSummary>,
    reliability: ReliabilitySummary,
    report_path: Option<String>,
    artifact_index_path: String,
    forensics_artifact_index: ForensicsArtifactIndex,
}

#[derive(Debug, Deserialize)]
struct WorkflowScenarioFixtureCase {
    id: String,
}

#[derive(Debug, Clone, Serialize)]
struct TriageReport {
    failure_class: String,
    first_bad_evidence: String,
    suggested_next_action: String,
}

#[derive(Debug, Clone, Serialize)]
struct FailureEnvelope {
    gate: String,
    suite: String,
    failure_class: String,
    reason_code: String,
    first_bad_evidence: String,
    fixture_lineage: Vec<String>,
    replay_command: String,
    evidence_refs: Vec<String>,
    suggested_next_action: String,
}

#[derive(Debug, Clone, Serialize)]
struct ArtifactIndexEntry {
    id: String,
    kind: String,
    path: String,
    description: String,
}

#[derive(Debug, Clone, Serialize)]
struct ForensicsArtifactIndex {
    schema_version: u8,
    gate: String,
    generated_at_unix_ms: u128,
    status: String,
    execution_context: String,
    triage: Option<TriageReport>,
    failure_envelopes: Vec<FailureEnvelope>,
    artifacts: Vec<ArtifactIndexEntry>,
}

#[derive(Debug)]
struct WorkflowFailureEvidence {
    fixture_id: String,
    scenario_id: String,
    step_id: String,
    reason_code: String,
    detail: String,
    artifact_refs: Vec<String>,
}

struct ForensicsBuildContext<'a> {
    status: &'a str,
    attempts: &'a [AttemptSummary],
    diagnostics: &'a [ReliabilityDiagnostic],
    workflow_log: &'a str,
    report_path: Option<&'a str>,
    flake_budget: usize,
    coverage_floor: f64,
}

struct FailureEnvelopeInput<'a> {
    reason_code: &'a str,
    failure_class: &'a str,
    first_bad_evidence: String,
    workflow_log: &'a str,
    report_path: Option<&'a str>,
    log_evidence: Option<&'a WorkflowFailureEvidence>,
    replay_command: &'a str,
    suggested_next_action: &'a str,
    diagnostic_evidence_refs: Vec<String>,
}

#[derive(Debug)]
struct GateOptions {
    log_path: PathBuf,
    artifact_index_path: PathBuf,
    retries: usize,
    flake_budget: usize,
    coverage_floor: f64,
    report_path: Option<PathBuf>,
}

struct WorkflowLogRequirementGuard;

impl Drop for WorkflowLogRequirementGuard {
    fn drop(&mut self) {
        set_workflow_scenario_log_required(false);
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("run_workflow_scenario_gate failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let options = parse_args()?;

    set_workflow_scenario_log_required(true);
    let _log_requirement_guard = WorkflowLogRequirementGuard;

    let cfg = HarnessConfig::default_paths();
    let mut attempts = Vec::new();

    for attempt in 0..=options.retries {
        let attempt_log_path = derive_attempt_log_path(&options.log_path, attempt);
        set_workflow_scenario_log_path(Some(attempt_log_path.clone()));

        let suite = run_user_workflow_scenario_suite(&cfg)?;
        let mut suite_summary = summarize_suite(suite);
        if let Err(err) = validate_workflow_log_coverage(&attempt_log_path, &cfg.fixture_root) {
            suite_summary
                .failures
                .push(format!("workflow_log_validation: {err}"));
        }

        let attempt_status = if suite_summary.pass_count == suite_summary.case_count
            && suite_summary.failures.is_empty()
        {
            "pass".to_string()
        } else {
            "fail".to_string()
        };

        attempts.push(AttemptSummary {
            attempt,
            status: attempt_status.clone(),
            workflow_log: attempt_log_path.display().to_string(),
            suite: suite_summary,
        });

        if attempt_status == "pass" {
            break;
        }
    }

    let pass_attempt_index = attempts.iter().position(|attempt| attempt.status == "pass");
    let final_attempt = attempts
        .last()
        .ok_or_else(|| "workflow gate produced no attempts".to_string())?;
    let workflow_log = final_attempt.workflow_log.clone();
    let final_suite = final_attempt.suite.clone();
    let coverage_ratio = coverage_ratio(&final_suite);
    let flaky_failures = pass_attempt_index.unwrap_or(0);

    let mut diagnostics = Vec::new();
    if pass_attempt_index.is_none() {
        diagnostics.push(ReliabilityDiagnostic {
            subsystem: "workflow_scenarios".to_string(),
            reason_code: "deterministic_failure".to_string(),
            message: "workflow scenario suite did not pass within retry budget".to_string(),
            evidence_refs: vec![workflow_log.clone()],
        });
    }
    if flaky_failures > options.flake_budget {
        diagnostics.push(ReliabilityDiagnostic {
            subsystem: "workflow_scenarios".to_string(),
            reason_code: "flake_budget_exceeded".to_string(),
            message: format!(
                "flaky failures {} exceeded flake budget {}",
                flaky_failures, options.flake_budget
            ),
            evidence_refs: attempts
                .iter()
                .map(|attempt| attempt.workflow_log.clone())
                .collect(),
        });
    }
    if coverage_ratio + f64::EPSILON < options.coverage_floor {
        diagnostics.push(ReliabilityDiagnostic {
            subsystem: "workflow_scenarios".to_string(),
            reason_code: "coverage_floor_breach".to_string(),
            message: format!(
                "coverage ratio {:.6} is below floor {:.6}",
                coverage_ratio, options.coverage_floor
            ),
            evidence_refs: vec![workflow_log.clone()],
        });
    }

    let status = if diagnostics.is_empty() {
        "pass"
    } else {
        "fail"
    };
    let attempts_run = attempts.len();
    let report_path = options
        .report_path
        .as_ref()
        .map(|path| path.display().to_string());
    let replay_command =
        replay_command(&workflow_log, options.flake_budget, options.coverage_floor);

    let forensics_artifact_index = build_forensics_artifact_index(ForensicsBuildContext {
        status,
        attempts: &attempts,
        diagnostics: &diagnostics,
        workflow_log: &workflow_log,
        report_path: report_path.as_deref(),
        flake_budget: options.flake_budget,
        coverage_floor: options.coverage_floor,
    })?;
    write_json_file(&options.artifact_index_path, &forensics_artifact_index)?;

    let summary = GateSummary {
        status,
        workflow_log: workflow_log.clone(),
        replay_command,
        attempts,
        suites: vec![final_suite],
        reliability: ReliabilitySummary {
            retries: options.retries,
            attempts_run,
            flaky_failures,
            flake_budget: options.flake_budget,
            coverage_floor: options.coverage_floor,
            coverage_ratio,
            diagnostics,
        },
        report_path,
        artifact_index_path: options.artifact_index_path.display().to_string(),
        forensics_artifact_index,
    };

    let summary_json = serde_json::to_string_pretty(&summary)
        .map_err(|err| format!("failed serializing summary: {err}"))?;

    if let Some(report_path) = options.report_path {
        if let Some(parent) = report_path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed creating report directory {}: {err}",
                    parent.display()
                )
            })?;
        }
        fs::write(&report_path, summary_json.as_bytes())
            .map_err(|err| format!("failed writing report {}: {err}", report_path.display()))?;
    }

    println!("{summary_json}");

    if status == "fail" {
        std::process::exit(2);
    }
    Ok(())
}

fn parse_args() -> Result<GateOptions, String> {
    let mut log_path: Option<PathBuf> = None;
    let mut artifact_index_path: Option<PathBuf> = None;
    let mut report_path: Option<PathBuf> = None;
    let mut retries = 0usize;
    let mut flake_budget = 0usize;
    let mut coverage_floor = 1.0f64;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--log-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--log-path requires a value".to_string())?;
                log_path = Some(PathBuf::from(value));
            }
            "--artifact-index-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--artifact-index-path requires a value".to_string())?;
                artifact_index_path = Some(PathBuf::from(value));
            }
            "--report-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--report-path requires a value".to_string())?;
                report_path = Some(PathBuf::from(value));
            }
            "--retries" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--retries requires a value".to_string())?;
                retries = value
                    .parse::<usize>()
                    .map_err(|err| format!("invalid --retries value '{value}': {err}"))?;
            }
            "--flake-budget" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--flake-budget requires a value".to_string())?;
                flake_budget = value
                    .parse::<usize>()
                    .map_err(|err| format!("invalid --flake-budget value '{value}': {err}"))?;
            }
            "--coverage-floor" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--coverage-floor requires a value".to_string())?;
                coverage_floor = value
                    .parse::<f64>()
                    .map_err(|err| format!("invalid --coverage-floor value '{value}': {err}"))?;
                if !(0.0..=1.0).contains(&coverage_floor) {
                    return Err(format!(
                        "--coverage-floor must be between 0.0 and 1.0, got {coverage_floor}"
                    ));
                }
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run -p fnp-conformance --bin run_workflow_scenario_gate -- [--log-path <path>] [--artifact-index-path <path>] [--report-path <path>] [--retries <n>] [--flake-budget <n>] [--coverage-floor <ratio>]"
                );
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }

    let ts_millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis());
    let log_path = log_path.unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../artifacts/logs")
            .join(format!("workflow_scenario_e2e_{ts_millis}.jsonl"))
    });
    let artifact_index_path = artifact_index_path.unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../artifacts/logs")
            .join(format!("workflow_scenario_artifact_index_{ts_millis}.json"))
    });

    Ok(GateOptions {
        log_path,
        artifact_index_path,
        retries,
        flake_budget,
        coverage_floor,
        report_path,
    })
}

fn summarize_suite(report: SuiteReport) -> SuiteSummary {
    SuiteSummary {
        suite: report.suite.to_string(),
        case_count: report.case_count,
        pass_count: report.pass_count,
        failures: report.failures,
    }
}

fn validate_workflow_log_coverage(log_path: &Path, fixture_root: &Path) -> Result<(), String> {
    let expected_ids =
        load_expected_scenario_ids(&fixture_root.join("workflow_scenario_corpus.json"))?;
    if expected_ids.is_empty() {
        return Err("workflow scenario corpus did not define any scenario ids".to_string());
    }

    let raw = fs::read_to_string(log_path)
        .map_err(|err| format!("failed reading workflow log {}: {err}", log_path.display()))?;
    if raw.lines().all(|line| line.trim().is_empty()) {
        return Err(format!("workflow log is empty: {}", log_path.display()));
    }

    let mut actual_ids = BTreeSet::new();
    for (idx, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line).map_err(|err| {
            format!(
                "workflow log parse error {} line {}: {err}",
                log_path.display(),
                idx + 1
            )
        })?;
        let scenario_id = value
            .get("scenario_id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                format!(
                    "workflow log missing scenario_id {} line {}",
                    log_path.display(),
                    idx + 1
                )
            })?;
        if scenario_id.trim().is_empty() {
            return Err(format!(
                "workflow log has empty scenario_id {} line {}",
                log_path.display(),
                idx + 1
            ));
        }
        actual_ids.insert(scenario_id.to_string());
    }

    if actual_ids.is_empty() {
        return Err(format!(
            "workflow log has no scenario entries: {}",
            log_path.display()
        ));
    }

    let missing: Vec<String> = expected_ids.difference(&actual_ids).cloned().collect();
    if !missing.is_empty() {
        return Err(format!(
            "workflow log missing scenario coverage for {} scenario(s): {}",
            missing.len(),
            missing.join(", ")
        ));
    }

    Ok(())
}

fn load_expected_scenario_ids(path: &Path) -> Result<BTreeSet<String>, String> {
    let raw = fs::read_to_string(path).map_err(|err| {
        format!(
            "failed reading workflow scenario corpus {}: {err}",
            path.display()
        )
    })?;
    let cases: Vec<WorkflowScenarioFixtureCase> = serde_json::from_str(&raw).map_err(|err| {
        format!(
            "invalid workflow scenario corpus json {}: {err}",
            path.display()
        )
    })?;

    Ok(cases.into_iter().map(|case| case.id).collect())
}

fn derive_attempt_log_path(base: &Path, attempt: usize) -> PathBuf {
    if attempt == 0 {
        return base.to_path_buf();
    }

    let stem = base
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("workflow_scenario_e2e");
    let ext = base.extension().and_then(|ext| ext.to_str());
    let file_name = if let Some(ext) = ext {
        format!("{stem}.attempt{attempt}.{ext}")
    } else {
        format!("{stem}.attempt{attempt}")
    };

    if let Some(parent) = base.parent() {
        parent.join(file_name)
    } else {
        PathBuf::from(file_name)
    }
}

fn coverage_ratio(summary: &SuiteSummary) -> f64 {
    if summary.case_count == 0 {
        0.0
    } else {
        summary.pass_count as f64 / summary.case_count as f64
    }
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn build_forensics_artifact_index(
    context: ForensicsBuildContext<'_>,
) -> Result<ForensicsArtifactIndex, String> {
    let first_failed_log = load_first_failed_workflow_evidence(Path::new(context.workflow_log))?;
    let first_suite_failure = context
        .attempts
        .iter()
        .flat_map(|attempt| attempt.suite.failures.iter())
        .find(|failure| !failure.trim().is_empty())
        .cloned();

    let reason_code = if let Some(diagnostic) = context.diagnostics.first() {
        diagnostic.reason_code.clone()
    } else if let Some(log) = &first_failed_log {
        log.reason_code.clone()
    } else {
        "scenario_assertion_failed".to_string()
    };
    let failure_class = failure_class_from_reason_code(&reason_code).to_string();

    let first_bad_evidence = if let Some(log) = &first_failed_log {
        format!(
            "{}::{}::{}: {}",
            log.scenario_id, log.step_id, log.fixture_id, log.detail
        )
    } else if let Some(failure) = &first_suite_failure {
        failure.clone()
    } else if let Some(diagnostic) = context.diagnostics.first() {
        diagnostic.message.clone()
    } else {
        "no failure evidence recorded".to_string()
    };

    let suggested_next_action = suggested_next_action(
        &reason_code,
        context.workflow_log,
        context.flake_budget,
        context.coverage_floor,
        first_failed_log.as_ref(),
    );

    let triage = if context.status == "fail" {
        Some(TriageReport {
            failure_class: failure_class.clone(),
            first_bad_evidence: first_bad_evidence.clone(),
            suggested_next_action: suggested_next_action.clone(),
        })
    } else {
        None
    };

    let replay_command = replay_command(
        context.workflow_log,
        context.flake_budget,
        context.coverage_floor,
    );
    let failure_envelopes = if context.status == "fail" {
        let mut envelopes = Vec::new();
        if context.diagnostics.is_empty() {
            envelopes.push(build_failure_envelope(FailureEnvelopeInput {
                reason_code: "scenario_assertion_failed",
                failure_class: &failure_class,
                first_bad_evidence: first_suite_failure
                    .unwrap_or_else(|| first_bad_evidence.clone()),
                workflow_log: context.workflow_log,
                report_path: context.report_path,
                log_evidence: first_failed_log.as_ref(),
                replay_command: &replay_command,
                suggested_next_action: &suggested_next_action,
                diagnostic_evidence_refs: Vec::new(),
            }));
        } else {
            for diagnostic in context.diagnostics {
                envelopes.push(build_failure_envelope(FailureEnvelopeInput {
                    reason_code: &diagnostic.reason_code,
                    failure_class: failure_class_from_reason_code(&diagnostic.reason_code),
                    first_bad_evidence: diagnostic.message.clone(),
                    workflow_log: context.workflow_log,
                    report_path: context.report_path,
                    log_evidence: first_failed_log.as_ref(),
                    replay_command: &replay_command,
                    suggested_next_action: &suggested_next_action,
                    diagnostic_evidence_refs: diagnostic.evidence_refs.clone(),
                }));
            }
        }
        envelopes
    } else {
        Vec::new()
    };

    let mut artifacts = vec![
        ArtifactIndexEntry {
            id: "workflow_log".to_string(),
            kind: "jsonl_log".to_string(),
            path: context.workflow_log.to_string(),
            description: "Step-level workflow replay log with pass/fail entries".to_string(),
        },
        ArtifactIndexEntry {
            id: "workflow_corpus".to_string(),
            kind: "fixture_collection".to_string(),
            path: "crates/fnp-conformance/fixtures/workflow_scenario_corpus.json".to_string(),
            description: "Scenario fixture corpus referenced by workflow gate".to_string(),
        },
        ArtifactIndexEntry {
            id: "ufunc_input_cases".to_string(),
            kind: "fixture_collection".to_string(),
            path: "crates/fnp-conformance/fixtures/ufunc_input_cases.json".to_string(),
            description: "Differential fixture cases linked from workflow scenarios".to_string(),
        },
        ArtifactIndexEntry {
            id: "shape_stride_cases".to_string(),
            kind: "fixture_collection".to_string(),
            path: "crates/fnp-conformance/fixtures/shape_stride_cases.json".to_string(),
            description: "Shape/stride fixture cases linked from workflow scenarios".to_string(),
        },
        ArtifactIndexEntry {
            id: "iter_differential_cases".to_string(),
            kind: "fixture_collection".to_string(),
            path: "crates/fnp-conformance/fixtures/iter_differential_cases.json".to_string(),
            description: "Iterator differential fixture cases linked from workflow scenarios"
                .to_string(),
        },
        ArtifactIndexEntry {
            id: "iter_adversarial_cases".to_string(),
            kind: "fixture_collection".to_string(),
            path: "crates/fnp-conformance/fixtures/iter_adversarial_cases.json".to_string(),
            description: "Iterator adversarial fixture cases linked from workflow scenarios"
                .to_string(),
        },
        ArtifactIndexEntry {
            id: "io_differential_cases".to_string(),
            kind: "fixture_collection".to_string(),
            path: "crates/fnp-conformance/fixtures/io_differential_cases.json".to_string(),
            description: "IO differential fixture cases linked from workflow scenarios".to_string(),
        },
        ArtifactIndexEntry {
            id: "io_adversarial_cases".to_string(),
            kind: "fixture_collection".to_string(),
            path: "crates/fnp-conformance/fixtures/io_adversarial_cases.json".to_string(),
            description: "IO adversarial fixture cases linked from workflow scenarios".to_string(),
        },
        ArtifactIndexEntry {
            id: "linalg_differential_cases".to_string(),
            kind: "fixture_collection".to_string(),
            path: "crates/fnp-conformance/fixtures/linalg_differential_cases.json".to_string(),
            description: "Linalg differential fixture cases linked from workflow scenarios"
                .to_string(),
        },
        ArtifactIndexEntry {
            id: "linalg_adversarial_cases".to_string(),
            kind: "fixture_collection".to_string(),
            path: "crates/fnp-conformance/fixtures/linalg_adversarial_cases.json".to_string(),
            description: "Linalg adversarial fixture cases linked from workflow scenarios"
                .to_string(),
        },
        ArtifactIndexEntry {
            id: "workflow_runner_script".to_string(),
            kind: "replay_script".to_string(),
            path: "scripts/e2e/run_workflow_scenario_gate.sh".to_string(),
            description: "Canonical script for local/CI replay".to_string(),
        },
        ArtifactIndexEntry {
            id: "linalg_packet_workflow_script".to_string(),
            kind: "replay_script".to_string(),
            path: "scripts/e2e/run_linalg_contract_journey.sh".to_string(),
            description: "Packet-008 workflow replay wrapper script".to_string(),
        },
        ArtifactIndexEntry {
            id: "io_packet_workflow_script".to_string(),
            kind: "replay_script".to_string(),
            path: "scripts/e2e/run_io_contract_journey.sh".to_string(),
            description: "Packet-009 workflow replay wrapper script".to_string(),
        },
    ];
    if let Some(path) = context.report_path {
        artifacts.push(ArtifactIndexEntry {
            id: "reliability_report".to_string(),
            kind: "gate_report".to_string(),
            path: path.to_string(),
            description: "Workflow reliability summary report".to_string(),
        });
    }

    Ok(ForensicsArtifactIndex {
        schema_version: 1,
        gate: "run_workflow_scenario_gate".to_string(),
        generated_at_unix_ms: now_unix_ms(),
        status: context.status.to_string(),
        execution_context: execution_context().to_string(),
        triage,
        failure_envelopes,
        artifacts,
    })
}

fn build_failure_envelope(input: FailureEnvelopeInput<'_>) -> FailureEnvelope {
    let mut evidence_refs = vec![input.workflow_log.to_string()];
    if let Some(path) = input.report_path {
        evidence_refs.push(path.to_string());
    }
    evidence_refs.extend(input.diagnostic_evidence_refs);
    if let Some(log) = input.log_evidence {
        evidence_refs.extend(log.artifact_refs.iter().cloned());
    }
    evidence_refs.sort();
    evidence_refs.dedup();

    let fixture_lineage = if let Some(log) = input.log_evidence {
        vec![
            format!("scenario_id={}", log.scenario_id),
            format!("step_id={}", log.step_id),
            format!("fixture_id={}", log.fixture_id),
        ]
    } else {
        Vec::new()
    };

    FailureEnvelope {
        gate: "run_workflow_scenario_gate".to_string(),
        suite: "workflow_scenarios".to_string(),
        failure_class: input.failure_class.to_string(),
        reason_code: input.reason_code.to_string(),
        first_bad_evidence: input.first_bad_evidence,
        fixture_lineage,
        replay_command: input.replay_command.to_string(),
        evidence_refs,
        suggested_next_action: input.suggested_next_action.to_string(),
    }
}

fn replay_command(workflow_log: &str, flake_budget: usize, coverage_floor: f64) -> String {
    format!(
        "cargo run -p fnp-conformance --bin run_workflow_scenario_gate -- --log-path '{}' --retries 0 --flake-budget {} --coverage-floor {}",
        workflow_log, flake_budget, coverage_floor
    )
}

fn execution_context() -> &'static str {
    if std::env::var_os("CI").is_some() {
        "ci"
    } else {
        "local"
    }
}

fn failure_class_from_reason_code(reason_code: &str) -> &'static str {
    match reason_code {
        "deterministic_failure" => "deterministic_regression",
        "flake_budget_exceeded" => "flake_budget",
        "coverage_floor_breach" => "coverage_floor",
        "workflow_log_validation_failure" => "artifact_log_validation",
        _ => "scenario_assertion",
    }
}

fn suggested_next_action(
    reason_code: &str,
    workflow_log: &str,
    flake_budget: usize,
    coverage_floor: f64,
    evidence: Option<&WorkflowFailureEvidence>,
) -> String {
    match reason_code {
        "flake_budget_exceeded" => format!(
            "Inspect per-attempt logs, confirm deterministic regressions first, then rerun with an explicit retry budget. Replay command: {}",
            replay_command(workflow_log, flake_budget, coverage_floor)
        ),
        "coverage_floor_breach" => format!(
            "Find missing scenario IDs in {} and add/update workflow corpus entries before rerunning.",
            workflow_log
        ),
        _ => {
            if let Some(log) = evidence {
                format!(
                    "Inspect scenario '{}' step '{}' (fixture '{}') in {} and replay with: {}",
                    log.scenario_id,
                    log.step_id,
                    log.fixture_id,
                    workflow_log,
                    replay_command(workflow_log, flake_budget, coverage_floor)
                )
            } else {
                format!(
                    "Inspect the first failure in {} and replay with: {}",
                    workflow_log,
                    replay_command(workflow_log, flake_budget, coverage_floor)
                )
            }
        }
    }
}

fn load_first_failed_workflow_evidence(
    path: &Path,
) -> Result<Option<WorkflowFailureEvidence>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed reading workflow log {}: {err}", path.display()))?;
    for (line_no, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line).map_err(|err| {
            format!(
                "failed parsing workflow log {} line {}: {err}",
                path.display(),
                line_no + 1
            )
        })?;
        if value.get("passed").and_then(Value::as_bool).unwrap_or(true) {
            continue;
        }

        let fixture_id = non_empty_field(&value, "fixture_id")
            .unwrap_or_else(|| format!("unknown_fixture_line_{}", line_no + 1));
        let scenario_id = non_empty_field(&value, "scenario_id")
            .unwrap_or_else(|| "unknown_scenario".to_string());
        let step_id =
            non_empty_field(&value, "step_id").unwrap_or_else(|| "unknown_step".to_string());
        let reason_code =
            non_empty_field(&value, "reason_code").unwrap_or_else(|| "unspecified".to_string());
        let detail = non_empty_field(&value, "detail").unwrap_or_else(|| "no detail".to_string());
        let artifact_refs = value
            .get("artifact_refs")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::trim)
                    .filter(|item| !item.is_empty())
                    .map(ToString::to_string)
                    .collect()
            })
            .unwrap_or_default();

        return Ok(Some(WorkflowFailureEvidence {
            fixture_id,
            scenario_id,
            step_id,
            reason_code,
            detail,
            artifact_refs,
        }));
    }
    Ok(None)
}

fn non_empty_field(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(ToString::to_string)
}

fn write_json_file<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }
    let payload = serde_json::to_string_pretty(value)
        .map_err(|err| format!("failed serializing json: {err}"))?;
    fs::write(path, payload).map_err(|err| format!("failed writing {}: {err}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::{
        derive_attempt_log_path, failure_class_from_reason_code,
        load_first_failed_workflow_evidence,
    };
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn derive_attempt_log_path_keeps_base_for_first_attempt() {
        let base = PathBuf::from("/tmp/workflow_scenario_e2e.jsonl");
        assert_eq!(derive_attempt_log_path(&base, 0), base);
    }

    #[test]
    fn derive_attempt_log_path_adds_attempt_suffix() {
        let base = PathBuf::from("/tmp/workflow_scenario_e2e.jsonl");
        let actual = derive_attempt_log_path(&base, 2);
        assert_eq!(
            actual,
            PathBuf::from("/tmp/workflow_scenario_e2e.attempt2.jsonl")
        );
    }

    #[test]
    fn load_first_failed_workflow_evidence_returns_first_failed_entry() {
        let ts_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        let log_path = std::env::temp_dir().join(format!(
            "workflow_forensics_test_{}_{}.jsonl",
            std::process::id(),
            ts_nanos
        ));
        let payload = concat!(
            "{\"scenario_id\":\"happy\",\"step_id\":\"ok\",\"fixture_id\":\"happy::ok\",\"passed\":true,\"reason_code\":\"rc\",\"detail\":\"\",\"artifact_refs\":[\"a\"]}\n",
            "{\"scenario_id\":\"bad\",\"step_id\":\"step_1\",\"fixture_id\":\"bad::step_1\",\"passed\":false,\"reason_code\":\"bad_reason\",\"detail\":\"boom\",\"artifact_refs\":[\"x\",\"y\"]}\n"
        );
        fs::write(&log_path, payload).expect("should write temp workflow log");

        let evidence = load_first_failed_workflow_evidence(&log_path)
            .expect("should parse workflow log")
            .expect("should find failed entry");
        assert_eq!(evidence.scenario_id, "bad");
        assert_eq!(evidence.step_id, "step_1");
        assert_eq!(evidence.fixture_id, "bad::step_1");
        assert_eq!(evidence.reason_code, "bad_reason");
        assert_eq!(evidence.detail, "boom");
        assert_eq!(evidence.artifact_refs, vec!["x", "y"]);
    }

    #[test]
    fn failure_class_mapping_covers_known_reasons() {
        assert_eq!(
            failure_class_from_reason_code("deterministic_failure"),
            "deterministic_regression"
        );
        assert_eq!(
            failure_class_from_reason_code("flake_budget_exceeded"),
            "flake_budget"
        );
        assert_eq!(
            failure_class_from_reason_code("coverage_floor_breach"),
            "coverage_floor"
        );
        assert_eq!(
            failure_class_from_reason_code("unknown_reason"),
            "scenario_assertion"
        );
    }
}
