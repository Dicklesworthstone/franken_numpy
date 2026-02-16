#![forbid(unsafe_code)]

use fnp_conformance::{
    HarnessConfig, SuiteReport, run_runtime_policy_adversarial_suite, run_runtime_policy_suite,
    security_contracts, set_runtime_policy_log_path,
};
use serde::Serialize;
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
        eprintln!("run_security_gate failed: {err}");
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

        let suites = run_gate_suites(&cfg)?
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
        .ok_or_else(|| "security gate produced no attempts".to_string())?;
    let runtime_policy_log = final_attempt.runtime_policy_log.clone();
    let final_suites = final_attempt.suites.clone();
    let coverage_ratio = coverage_ratio(&final_suites);
    let flaky_failures = pass_attempt_index.unwrap_or(0);

    let mut diagnostics = Vec::new();
    if pass_attempt_index.is_none() {
        diagnostics.push(ReliabilityDiagnostic {
            subsystem: "runtime_policy".to_string(),
            reason_code: "deterministic_failure".to_string(),
            message: "security gate did not pass within retry budget".to_string(),
            evidence_refs: attempts
                .iter()
                .map(|attempt| attempt.runtime_policy_log.clone())
                .collect(),
        });
    }
    if flaky_failures > options.flake_budget {
        diagnostics.push(ReliabilityDiagnostic {
            subsystem: "runtime_policy".to_string(),
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
            subsystem: "runtime_policy".to_string(),
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
                    "Usage: cargo run -p fnp-conformance --bin run_security_gate -- [--log-path <path>] [--report-path <path>] [--retries <n>] [--flake-budget <n>] [--coverage-floor <ratio>]"
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
            .join(format!("runtime_policy_e2e_{ts_millis}.jsonl"))
    });

    Ok(GateOptions {
        log_path,
        retries,
        flake_budget,
        coverage_floor,
        report_path,
    })
}

fn run_gate_suites(cfg: &HarnessConfig) -> Result<Vec<SuiteReport>, String> {
    Ok(vec![
        run_runtime_policy_suite(cfg)?,
        run_runtime_policy_adversarial_suite(cfg)?,
        security_contracts::run_security_contract_suite(cfg)?,
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
        .unwrap_or("runtime_policy_e2e");
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
