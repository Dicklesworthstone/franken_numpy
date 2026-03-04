#![forbid(unsafe_code)]

use fnp_conformance::{
    HarnessConfig, SuiteReport, run_crash_signature_regression_suite, run_dtype_adversarial_suite,
    run_dtype_differential_suite, run_dtype_metamorphic_suite, run_io_adversarial_suite,
    run_io_differential_suite, run_io_metamorphic_suite, run_iter_adversarial_suite,
    run_iter_differential_suite, run_iter_metamorphic_suite, run_linalg_adversarial_suite,
    run_linalg_differential_suite, run_linalg_metamorphic_suite, run_rng_adversarial_suite,
    run_rng_differential_suite, run_rng_metamorphic_suite, run_rng_statistical_suite,
    run_runtime_policy_adversarial_suite, run_runtime_policy_suite,
    run_shape_stride_adversarial_suite, run_shape_stride_differential_suite,
    run_shape_stride_metamorphic_suite, run_ufunc_adversarial_suite, run_ufunc_differential_suite,
    run_ufunc_metamorphic_suite, set_runtime_policy_log_path, test_contracts,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const REQUIRED_LOG_FIELDS: &[&str] = &[
    "fixture_id",
    "seed",
    "mode",
    "env_fingerprint",
    "artifact_refs",
    "reason_code",
];

const CI_GATE_TOPOLOGY_CONTRACT_PATH: &str = "artifacts/contracts/ci_gate_topology_v1.json";
const CI_TOPOLOGY_RUNNER_SCRIPT: &str = "scripts/e2e/run_ci_gate_topology.sh";
const CI_TOPOLOGY_SCHEMA_VERSION: &str = "ci-gate-topology-v1";
const EXPECTED_GATE_IDS: &[&str] = &["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"];
const REQUIRED_BRANCH_CHECKS: &[&str] = &[
    "ci/g1-fmt-lint",
    "ci/g2-unit-property",
    "ci/g3-differential",
    "ci/g4-adversarial-security",
    "ci/g5-test-contract",
    "ci/g6-workflow-forensics",
    "ci/g7-performance-budget",
    "ci/g8-durability-decode",
];
const REQUIRED_MERGE_BLOCKERS: &[&str] =
    &["compatibility_drift", "missing_artifacts", "budget_breach"];
const REQUIRED_G7_COMMAND_FRAGMENT: &str = "scripts/e2e/run_performance_budget_gate.sh";
const REQUIRED_G8_COMMAND_FRAGMENT: &str = "scripts/e2e/run_raptorq_gate.sh";
const REQUIRED_G7_OUTPUTS: &[&str] = &["candidate_baseline", "performance_budget_report"];
const REQUIRED_G8_OUTPUTS: &[&str] = &[
    "sidecar",
    "scrub_report",
    "decode_proof",
    "raptorq_reliability_report",
];

#[derive(Debug, Deserialize)]
struct CiGateTopologyContract {
    schema_version: String,
    runner_script: String,
    gates: Vec<CiGateDefinition>,
    branch_protection: BranchProtectionContract,
}

#[derive(Debug, Deserialize)]
struct CiGateDefinition {
    id: String,
    required_command_fragment: String,
    required_outputs: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct BranchProtectionContract {
    required_checks: Vec<String>,
    merge_blockers: Vec<String>,
}

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
    runtime_policy_log: String,
    suites: Vec<SuiteSummary>,
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
    runtime_policy_log: String,
    attempts: Vec<AttemptSummary>,
    suites: Vec<SuiteSummary>,
    reliability: ReliabilitySummary,
    report_path: Option<String>,
}

#[derive(Debug)]
struct GateOptions {
    log_path: PathBuf,
    retries: usize,
    flake_budget: usize,
    coverage_floor: f64,
    report_path: Option<PathBuf>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("run_test_contract_gate failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let options = parse_args()?;
    let cfg = HarnessConfig::default_paths();
    let mut attempts = Vec::new();

    for attempt in 0..=options.retries {
        let attempt_log_path = derive_attempt_log_path(&options.log_path, attempt);
        set_runtime_policy_log_path(Some(attempt_log_path.clone()));

        let suites = run_gate_suites(&cfg, &attempt_log_path)?
            .into_iter()
            .map(summarize_suite)
            .collect::<Vec<_>>();
        let attempt_passed = suites
            .iter()
            .all(|suite| suite.case_count == suite.pass_count && suite.failures.is_empty());
        let attempt_status = if attempt_passed { "pass" } else { "fail" };

        attempts.push(AttemptSummary {
            attempt,
            status: attempt_status.to_string(),
            runtime_policy_log: attempt_log_path.display().to_string(),
            suites,
        });

        if attempt_passed {
            break;
        }
    }

    let pass_attempt_index = attempts.iter().position(|attempt| attempt.status == "pass");
    let final_attempt = attempts
        .last()
        .ok_or_else(|| "test contract gate produced no attempts".to_string())?;
    let runtime_policy_log = final_attempt.runtime_policy_log.clone();
    let final_suites = final_attempt.suites.clone();
    let coverage_ratio = coverage_ratio(&final_suites);
    let flaky_failures = pass_attempt_index.unwrap_or(0);

    let mut diagnostics = Vec::new();
    if pass_attempt_index.is_none() {
        diagnostics.push(ReliabilityDiagnostic {
            subsystem: "test_contracts".to_string(),
            reason_code: "deterministic_failure".to_string(),
            message: "test-contract gate did not pass within retry budget".to_string(),
            evidence_refs: attempts
                .iter()
                .map(|attempt| attempt.runtime_policy_log.clone())
                .collect(),
        });
    }
    if flaky_failures > options.flake_budget {
        diagnostics.push(ReliabilityDiagnostic {
            subsystem: "test_contracts".to_string(),
            reason_code: "flake_budget_exceeded".to_string(),
            message: format!(
                "flaky failures {} exceeded flake budget {}",
                flaky_failures, options.flake_budget
            ),
            evidence_refs: attempts
                .iter()
                .map(|attempt| attempt.runtime_policy_log.clone())
                .collect(),
        });
    }
    if coverage_ratio + f64::EPSILON < options.coverage_floor {
        diagnostics.push(ReliabilityDiagnostic {
            subsystem: "test_contracts".to_string(),
            reason_code: "coverage_floor_breach".to_string(),
            message: format!(
                "coverage ratio {:.6} is below floor {:.6}",
                coverage_ratio, options.coverage_floor
            ),
            evidence_refs: vec![runtime_policy_log.clone()],
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

    let summary = GateSummary {
        status,
        runtime_policy_log,
        attempts,
        suites: final_suites,
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
                    "Usage: cargo run -p fnp-conformance --bin run_test_contract_gate -- [--log-path <path>] [--report-path <path>] [--retries <n>] [--flake-budget <n>] [--coverage-floor <ratio>]"
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
            .join(format!("test_contract_e2e_{ts_millis}.jsonl"))
    });

    Ok(GateOptions {
        log_path,
        retries,
        flake_budget,
        coverage_floor,
        report_path,
    })
}

fn run_gate_suites(cfg: &HarnessConfig, log_path: &Path) -> Result<Vec<SuiteReport>, String> {
    let mut packet002_dtype_cfg = cfg.clone();
    packet002_dtype_cfg.fixture_root = cfg.fixture_root.join("packet002_dtype");

    Ok(vec![
        test_contracts::run_test_contract_suite(cfg)?,
        run_runtime_policy_suite(cfg)?,
        run_runtime_policy_adversarial_suite(cfg)?,
        run_shape_stride_differential_suite(cfg)?,
        run_shape_stride_metamorphic_suite(cfg)?,
        run_shape_stride_adversarial_suite(cfg)?,
        run_iter_differential_suite(cfg)?,
        run_iter_metamorphic_suite(cfg)?,
        run_iter_adversarial_suite(cfg)?,
        run_ufunc_differential_suite(cfg)?,
        run_ufunc_metamorphic_suite(cfg)?,
        run_ufunc_adversarial_suite(cfg)?,
        run_dtype_differential_suite(&packet002_dtype_cfg)?,
        run_dtype_metamorphic_suite(&packet002_dtype_cfg)?,
        run_dtype_adversarial_suite(&packet002_dtype_cfg)?,
        run_rng_differential_suite(cfg)?,
        run_rng_metamorphic_suite(cfg)?,
        run_rng_adversarial_suite(cfg)?,
        run_rng_statistical_suite(cfg)?,
        run_linalg_differential_suite(cfg)?,
        run_linalg_metamorphic_suite(cfg)?,
        run_linalg_adversarial_suite(cfg)?,
        run_io_differential_suite(cfg)?,
        run_io_metamorphic_suite(cfg)?,
        run_io_adversarial_suite(cfg)?,
        run_crash_signature_regression_suite(cfg)?,
        validate_runtime_policy_log(log_path)?,
        validate_ci_gate_topology_contract()?,
    ])
}

fn summarize_suite(report: SuiteReport) -> SuiteSummary {
    SuiteSummary {
        suite: report.suite.to_string(),
        case_count: report.case_count,
        pass_count: report.pass_count,
        failures: report.failures,
    }
}

fn coverage_ratio(suites: &[SuiteSummary]) -> f64 {
    let case_count = suites.iter().map(|suite| suite.case_count).sum::<usize>();
    let pass_count = suites.iter().map(|suite| suite.pass_count).sum::<usize>();
    if case_count == 0 {
        0.0
    } else {
        pass_count as f64 / case_count as f64
    }
}

fn derive_attempt_log_path(base: &Path, attempt: usize) -> PathBuf {
    if attempt == 0 {
        return base.to_path_buf();
    }

    let stem = base
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("test_contract_e2e");
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

fn validate_runtime_policy_log(path: &Path) -> Result<SuiteReport, String> {
    let raw = fs::read_to_string(path).map_err(|err| {
        format!(
            "failed reading runtime policy log {}: {err}",
            path.display()
        )
    })?;

    let mut report = SuiteReport {
        suite: "runtime_policy_log_contract",
        case_count: 0,
        pass_count: 0,
        failures: Vec::new(),
    };

    let mut entry_count = 0usize;
    for (line_no, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        entry_count += 1;

        let value: Value = match serde_json::from_str(line) {
            Ok(value) => value,
            Err(err) => {
                report.case_count += 1;
                report.failures.push(format!(
                    "line {} invalid json in {}: {err}",
                    line_no + 1,
                    path.display()
                ));
                continue;
            }
        };

        let Some(obj) = value.as_object() else {
            report.case_count += 1;
            report.failures.push(format!(
                "line {} must be a JSON object in {}",
                line_no + 1,
                path.display()
            ));
            continue;
        };

        for field in REQUIRED_LOG_FIELDS {
            report.case_count += 1;
            let pass = match *field {
                "artifact_refs" => obj
                    .get(*field)
                    .and_then(Value::as_array)
                    .is_some_and(|arr| {
                        !arr.is_empty()
                            && arr
                                .iter()
                                .all(|item| item.as_str().is_some_and(|s| !s.trim().is_empty()))
                    }),
                "seed" => obj.get(*field).is_some_and(Value::is_u64),
                _ => obj
                    .get(*field)
                    .and_then(Value::as_str)
                    .is_some_and(|s| !s.trim().is_empty()),
            };

            if pass {
                report.pass_count += 1;
            } else {
                report.failures.push(format!(
                    "line {} missing/invalid required field {}",
                    line_no + 1,
                    field
                ));
            }
        }
    }

    if entry_count == 0 {
        report.case_count += 1;
        report
            .failures
            .push("runtime policy log must contain at least one entry".to_string());
    } else {
        report.pass_count += 1;
        report.case_count += 1;
    }

    Ok(report)
}

fn validate_ci_gate_topology_contract() -> Result<SuiteReport, String> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let contract_path = repo_root.join(CI_GATE_TOPOLOGY_CONTRACT_PATH);
    let runner_path = repo_root.join(CI_TOPOLOGY_RUNNER_SCRIPT);

    let mut report = SuiteReport {
        suite: "ci_gate_topology_contract",
        case_count: 0,
        pass_count: 0,
        failures: Vec::new(),
    };

    let contract_raw = match fs::read_to_string(&contract_path) {
        Ok(raw) => {
            record_contract_check(
                &mut report,
                true,
                format!("failed reading {}", contract_path.display()),
            );
            raw
        }
        Err(err) => {
            record_contract_check(
                &mut report,
                false,
                format!("failed reading {}: {err}", contract_path.display()),
            );
            return Ok(report);
        }
    };

    let contract: CiGateTopologyContract = match serde_json::from_str(&contract_raw) {
        Ok(contract) => {
            record_contract_check(
                &mut report,
                true,
                format!("failed parsing {}", contract_path.display()),
            );
            contract
        }
        Err(err) => {
            record_contract_check(
                &mut report,
                false,
                format!("failed parsing {}: {err}", contract_path.display()),
            );
            return Ok(report);
        }
    };

    record_contract_check(
        &mut report,
        contract.schema_version == CI_TOPOLOGY_SCHEMA_VERSION,
        format!(
            "schema_version must be '{}' in {}",
            CI_TOPOLOGY_SCHEMA_VERSION,
            contract_path.display()
        ),
    );
    record_contract_check(
        &mut report,
        contract.runner_script == CI_TOPOLOGY_RUNNER_SCRIPT,
        format!(
            "runner_script must be '{}' in {}",
            CI_TOPOLOGY_RUNNER_SCRIPT,
            contract_path.display()
        ),
    );

    let gate_ids = contract
        .gates
        .iter()
        .map(|gate| gate.id.as_str())
        .collect::<Vec<_>>();
    record_contract_check(
        &mut report,
        gate_ids == EXPECTED_GATE_IDS,
        format!(
            "gate ids/order mismatch in {}; expected {}",
            contract_path.display(),
            EXPECTED_GATE_IDS.join(", ")
        ),
    );

    let unique_gate_ids = contract
        .gates
        .iter()
        .map(|gate| gate.id.as_str())
        .collect::<BTreeSet<_>>();
    record_contract_check(
        &mut report,
        unique_gate_ids.len() == contract.gates.len(),
        format!("duplicate gate ids found in {}", contract_path.display()),
    );

    record_contract_check(
        &mut report,
        !contract.branch_protection.required_checks.is_empty(),
        format!(
            "branch_protection.required_checks must not be empty in {}",
            contract_path.display()
        ),
    );
    for required_check in REQUIRED_BRANCH_CHECKS {
        record_contract_check(
            &mut report,
            contract
                .branch_protection
                .required_checks
                .iter()
                .any(|entry| entry == required_check),
            format!(
                "branch_protection.required_checks missing '{}' in {}",
                required_check,
                contract_path.display()
            ),
        );
    }

    for blocker in REQUIRED_MERGE_BLOCKERS {
        record_contract_check(
            &mut report,
            contract
                .branch_protection
                .merge_blockers
                .iter()
                .any(|entry| entry == blocker),
            format!(
                "branch_protection.merge_blockers missing '{}' in {}",
                blocker,
                contract_path.display()
            ),
        );
    }

    let runner_raw = match fs::read_to_string(&runner_path) {
        Ok(raw) => {
            record_contract_check(
                &mut report,
                true,
                format!("failed reading {}", runner_path.display()),
            );
            raw
        }
        Err(err) => {
            record_contract_check(
                &mut report,
                false,
                format!("failed reading {}: {err}", runner_path.display()),
            );
            return Ok(report);
        }
    };

    let mut prev_offset = 0usize;
    for gate in &contract.gates {
        record_contract_check(
            &mut report,
            !gate.required_command_fragment.trim().is_empty(),
            format!(
                "gate '{}' has empty required_command_fragment in {}",
                gate.id,
                contract_path.display()
            ),
        );
        record_contract_check(
            &mut report,
            !gate.required_outputs.is_empty(),
            format!(
                "gate '{}' must define required_outputs in {}",
                gate.id,
                contract_path.display()
            ),
        );

        if gate.required_command_fragment.trim().is_empty() {
            continue;
        }

        let maybe_index = runner_raw
            .get(prev_offset..)
            .and_then(|slice| slice.find(&gate.required_command_fragment))
            .map(|idx| prev_offset + idx);
        let found = maybe_index.is_some();
        record_contract_check(
            &mut report,
            found,
            format!(
                "runner script {} missing command fragment for gate '{}': {}",
                runner_path.display(),
                gate.id,
                gate.required_command_fragment
            ),
        );
        if let Some(index) = maybe_index {
            prev_offset = index + gate.required_command_fragment.len();
        }
    }

    if let Some(g6) = contract.gates.iter().find(|gate| gate.id == "G6") {
        for required_field in ["artifact_index_path", "replay_command"] {
            record_contract_check(
                &mut report,
                g6.required_outputs
                    .iter()
                    .any(|field| field == required_field),
                format!(
                    "G6 required_outputs missing '{}' in {}",
                    required_field,
                    contract_path.display()
                ),
            );
        }
    } else {
        record_contract_check(
            &mut report,
            false,
            format!("G6 gate missing in {}", contract_path.display()),
        );
    }

    if let Some(g7) = contract.gates.iter().find(|gate| gate.id == "G7") {
        record_contract_check(
            &mut report,
            g7.required_command_fragment == REQUIRED_G7_COMMAND_FRAGMENT,
            format!(
                "G7 required_command_fragment must be '{}' in {}",
                REQUIRED_G7_COMMAND_FRAGMENT,
                contract_path.display()
            ),
        );
        for required_field in REQUIRED_G7_OUTPUTS {
            record_contract_check(
                &mut report,
                g7.required_outputs
                    .iter()
                    .any(|field| field == required_field),
                format!(
                    "G7 required_outputs missing '{}' in {}",
                    required_field,
                    contract_path.display()
                ),
            );
        }
    } else {
        record_contract_check(
            &mut report,
            false,
            format!("G7 gate missing in {}", contract_path.display()),
        );
    }

    if let Some(g8) = contract.gates.iter().find(|gate| gate.id == "G8") {
        record_contract_check(
            &mut report,
            g8.required_command_fragment == REQUIRED_G8_COMMAND_FRAGMENT,
            format!(
                "G8 required_command_fragment must be '{}' in {}",
                REQUIRED_G8_COMMAND_FRAGMENT,
                contract_path.display()
            ),
        );
        for required_field in REQUIRED_G8_OUTPUTS {
            record_contract_check(
                &mut report,
                g8.required_outputs
                    .iter()
                    .any(|field| field == required_field),
                format!(
                    "G8 required_outputs missing '{}' in {}",
                    required_field,
                    contract_path.display()
                ),
            );
        }
    } else {
        record_contract_check(
            &mut report,
            false,
            format!("G8 gate missing in {}", contract_path.display()),
        );
    }

    Ok(report)
}

fn record_contract_check(report: &mut SuiteReport, pass: bool, failure_message: String) {
    report.case_count += 1;
    if pass {
        report.pass_count += 1;
    } else {
        report.failures.push(failure_message);
    }
}
