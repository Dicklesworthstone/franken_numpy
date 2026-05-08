#![forbid(unsafe_code)]

use fnp_conformance::{
    HarnessConfig, SuiteReport,
    raptorq_artifacts::{
        self, RaptorQParallelismConfig, RaptorQStressGateConfig, RaptorQStressMode,
        RaptorQStressReport, validate_raptorq_stress_report,
    },
};
use serde::Serialize;
use std::fs;
use std::path::PathBuf;

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
    expected_parallelism: Option<RaptorQParallelismConfig>,
    coverage_ratio: f64,
    diagnostics: Vec<ReliabilityDiagnostic>,
}

#[derive(Debug, Serialize)]
struct GateSummary {
    status: &'static str,
    attempts: Vec<AttemptSummary>,
    suites: Vec<SuiteSummary>,
    stress: Option<RaptorQStressReport>,
    reliability: ReliabilitySummary,
    report_path: Option<String>,
}

#[derive(Debug)]
struct GateOptions {
    retries: usize,
    flake_budget: usize,
    coverage_floor: f64,
    parallelism: Option<RaptorQParallelismConfig>,
    stress_mode: RaptorQStressMode,
    stress_output_dir: Option<PathBuf>,
    stress_source_bytes: Option<usize>,
    stress_parallelism: Option<RaptorQParallelismConfig>,
    report_path: Option<PathBuf>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("run_raptorq_gate failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let options = parse_args()?;
    let cfg = HarnessConfig::default_paths();
    let manifest_repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let repo_root = fs::canonicalize(&manifest_repo_root).unwrap_or(manifest_repo_root);
    let artifact_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/raptorq");
    let artifact_ref = artifact_root.display().to_string();
    let mut attempts = Vec::new();

    for attempt in 0..=options.retries {
        let suite = raptorq_artifacts::run_raptorq_artifact_suite_with_parallelism(
            &cfg,
            options.parallelism,
        )?;
        let suite_summary = summarize_suite(suite);
        let attempt_passed = suite_summary.case_count == suite_summary.pass_count
            && suite_summary.failures.is_empty();
        let attempt_status = if attempt_passed { "pass" } else { "fail" };

        attempts.push(AttemptSummary {
            attempt,
            status: attempt_status.to_string(),
            suites: vec![suite_summary],
        });

        if attempt_passed {
            break;
        }
    }

    let pass_attempt_index = attempts.iter().position(|attempt| attempt.status == "pass");
    let final_attempt = attempts
        .last()
        .ok_or_else(|| "raptorq gate produced no attempts".to_string())?;
    let final_suites = final_attempt.suites.clone();
    let coverage_ratio = coverage_ratio(&final_suites);
    let flaky_failures = pass_attempt_index.unwrap_or(0);

    let mut diagnostics = Vec::new();
    if pass_attempt_index.is_none() {
        diagnostics.push(ReliabilityDiagnostic {
            subsystem: "raptorq_artifacts".to_string(),
            reason_code: "deterministic_failure".to_string(),
            message: "raptorq gate did not pass within retry budget".to_string(),
            evidence_refs: vec![artifact_ref.clone()],
        });
    }
    if flaky_failures > options.flake_budget {
        diagnostics.push(ReliabilityDiagnostic {
            subsystem: "raptorq_artifacts".to_string(),
            reason_code: "flake_budget_exceeded".to_string(),
            message: format!(
                "flaky failures {} exceeded flake budget {}",
                flaky_failures, options.flake_budget
            ),
            evidence_refs: vec![artifact_ref.clone()],
        });
    }
    if coverage_ratio + f64::EPSILON < options.coverage_floor {
        diagnostics.push(ReliabilityDiagnostic {
            subsystem: "raptorq_artifacts".to_string(),
            reason_code: "coverage_floor_breach".to_string(),
            message: format!(
                "coverage ratio {:.6} is below floor {:.6}",
                coverage_ratio, options.coverage_floor
            ),
            evidence_refs: vec![artifact_ref],
        });
    }

    let stress_parallelism = options
        .stress_parallelism
        .or(options.parallelism)
        .unwrap_or_else(RaptorQParallelismConfig::serial);
    let stress_output_dir = options.stress_output_dir.clone().unwrap_or_else(|| {
        repo_root
            .join("target/raptorq_stress_gate")
            .join(options.stress_mode.as_str())
    });
    let stress_source_bytes = options
        .stress_source_bytes
        .unwrap_or_else(|| options.stress_mode.default_source_bytes());
    let stress_replay_command = replay_command(
        &options,
        &stress_output_dir,
        stress_source_bytes,
        stress_parallelism,
    );
    let stress_config = RaptorQStressGateConfig {
        repo_root,
        output_dir: stress_output_dir.clone(),
        mode: options.stress_mode,
        parallelism: stress_parallelism,
        source_bytes: stress_source_bytes,
        replay_command: stress_replay_command,
    };
    let stress = match raptorq_artifacts::run_raptorq_stress_gate(&stress_config) {
        Ok(report) => match validate_raptorq_stress_report(&report) {
            Ok(()) => Some(report),
            Err(err) => {
                diagnostics.push(ReliabilityDiagnostic {
                    subsystem: "raptorq_artifacts".to_string(),
                    reason_code: "stress_report_schema_failure".to_string(),
                    message: err,
                    evidence_refs: vec![stress_output_dir.display().to_string()],
                });
                Some(report)
            }
        },
        Err(err) => {
            diagnostics.push(ReliabilityDiagnostic {
                subsystem: "raptorq_artifacts".to_string(),
                reason_code: "stress_gate_failure".to_string(),
                message: err,
                evidence_refs: vec![stress_output_dir.display().to_string()],
            });
            None
        }
    };

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
        attempts,
        suites: final_suites,
        stress,
        reliability: ReliabilitySummary {
            retries: options.retries,
            attempts_run,
            flaky_failures,
            flake_budget: options.flake_budget,
            coverage_floor: options.coverage_floor,
            expected_parallelism: options.parallelism,
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
    parse_args_from(std::env::args().skip(1))
}

fn parse_args_from<I>(args: I) -> Result<GateOptions, String>
where
    I: IntoIterator<Item = String>,
{
    let mut report_path: Option<PathBuf> = None;
    let mut retries = 0usize;
    let mut flake_budget = 0usize;
    let mut coverage_floor = 1.0f64;
    let mut parallelism: Option<RaptorQParallelismConfig> = None;
    let mut stress_mode = RaptorQStressMode::Smoke;
    let mut stress_output_dir: Option<PathBuf> = None;
    let mut stress_source_bytes: Option<usize> = None;
    let mut stress_parallelism: Option<RaptorQParallelismConfig> = None;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
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
            "--parallelism" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--parallelism requires a value".to_string())?;
                let worker_count = value
                    .parse::<usize>()
                    .map_err(|err| format!("invalid --parallelism value '{value}': {err}"))?;
                parallelism = Some(RaptorQParallelismConfig::from_worker_count(worker_count)?);
            }
            "--stress-mode" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--stress-mode requires a value".to_string())?;
                stress_mode = RaptorQStressMode::parse(&value)?;
            }
            "--stress-output-dir" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--stress-output-dir requires a value".to_string())?;
                stress_output_dir = Some(PathBuf::from(value));
            }
            "--stress-source-bytes" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--stress-source-bytes requires a value".to_string())?;
                let bytes = value.parse::<usize>().map_err(|err| {
                    format!("invalid --stress-source-bytes value '{value}': {err}")
                })?;
                if bytes == 0 {
                    return Err("--stress-source-bytes must be greater than zero".to_string());
                }
                stress_source_bytes = Some(bytes);
            }
            "--stress-parallelism" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--stress-parallelism requires a value".to_string())?;
                let worker_count = value.parse::<usize>().map_err(|err| {
                    format!("invalid --stress-parallelism value '{value}': {err}")
                })?;
                stress_parallelism =
                    Some(RaptorQParallelismConfig::from_worker_count(worker_count)?);
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run -p fnp-conformance --bin run_raptorq_gate -- [--report-path <path>] [--retries <n>] [--flake-budget <n>] [--coverage-floor <ratio>] [--parallelism <n>] [--stress-mode smoke|local] [--stress-output-dir <path>] [--stress-source-bytes <n>] [--stress-parallelism <n>]"
                );
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }

    Ok(GateOptions {
        retries,
        flake_budget,
        coverage_floor,
        parallelism,
        stress_mode,
        stress_output_dir,
        stress_source_bytes,
        stress_parallelism,
        report_path,
    })
}

fn replay_command(
    options: &GateOptions,
    stress_output_dir: &std::path::Path,
    stress_source_bytes: usize,
    stress_parallelism: RaptorQParallelismConfig,
) -> String {
    let report_arg = options
        .report_path
        .as_ref()
        .map(|path| format!(" --report-path {}", path.display()))
        .unwrap_or_default();
    format!(
        "cargo run -p fnp-conformance --bin run_raptorq_gate --{report_arg} --retries {} --flake-budget {} --coverage-floor {} --stress-mode {} --stress-output-dir {} --stress-source-bytes {} --stress-parallelism {}",
        options.retries,
        options.flake_budget,
        options.coverage_floor,
        options.stress_mode.as_str(),
        stress_output_dir.display(),
        stress_source_bytes,
        stress_parallelism.worker_count
    )
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

#[cfg(test)]
mod tests {
    use super::{
        AttemptSummary, GateSummary, ReliabilitySummary, SuiteSummary, parse_args_from,
        replay_command,
    };
    use fnp_conformance::raptorq_artifacts::{
        RAPTORQ_STRESS_REPORT_SCHEMA_VERSION, RaptorQParallelismConfig, RaptorQStressMode,
        RaptorQStressReport,
    };
    use std::path::PathBuf;

    fn args(raw: &[&str]) -> Vec<String> {
        raw.iter().map(|arg| (*arg).to_string()).collect()
    }

    #[test]
    fn run_raptorq_gate_parse_args_accepts_stress_mode_and_paths() {
        let parsed = parse_args_from(args(&[
            "--report-path",
            "target/report.json",
            "--stress-mode",
            "local",
            "--stress-output-dir",
            "target/stress",
            "--stress-source-bytes",
            "4096",
            "--stress-parallelism",
            "3",
        ]))
        .expect("parse args");

        assert_eq!(
            parsed.report_path,
            Some(PathBuf::from("target/report.json"))
        );
        assert_eq!(parsed.stress_mode, RaptorQStressMode::Local);
        assert_eq!(
            parsed.stress_output_dir,
            Some(PathBuf::from("target/stress"))
        );
        assert_eq!(parsed.stress_source_bytes, Some(4096));
        assert_eq!(
            parsed.stress_parallelism,
            Some(RaptorQParallelismConfig { worker_count: 3 })
        );
    }

    #[test]
    fn run_raptorq_gate_parse_args_rejects_invalid_stress_mode() {
        let err = parse_args_from(args(&["--stress-mode", "huge"]))
            .expect_err("invalid stress mode should fail");
        assert!(err.contains("invalid raptorq stress mode"));
    }

    #[test]
    fn run_raptorq_gate_report_schema_includes_stress_fields() {
        let stress = RaptorQStressReport {
            schema_version: RAPTORQ_STRESS_REPORT_SCHEMA_VERSION,
            bundle_id: "raptorq_stress_smoke".to_string(),
            mode: RaptorQStressMode::Smoke,
            status: "pass".to_string(),
            worker_count: 2,
            output_dir: "target/stress".to_string(),
            input_files: vec!["target/stress/input.bin".to_string()],
            sidecar_path: "target/stress/sidecar.json".to_string(),
            scrub_report_path: "target/stress/scrub.json".to_string(),
            decode_proof_path: "target/stress/proof.json".to_string(),
            input_hash: "a".repeat(64),
            recovered_hash: "a".repeat(64),
            elapsed_ms: 1,
            source_size: 4096,
            source_symbols: 16,
            repair_symbols: 4,
            total_symbols: 20,
            dropped_symbol_scenario: Some("sbn=0 esi=0 kind=source".to_string()),
            recovery_symbols_used: 19,
            replay_command: "cargo run -p fnp-conformance --bin run_raptorq_gate".to_string(),
            diagnostics: Vec::new(),
        };
        let summary = GateSummary {
            status: "pass",
            attempts: vec![AttemptSummary {
                attempt: 0,
                status: "pass".to_string(),
                suites: vec![SuiteSummary {
                    suite: "raptorq_artifacts".to_string(),
                    case_count: 1,
                    pass_count: 1,
                    failures: Vec::new(),
                }],
            }],
            suites: Vec::new(),
            stress: Some(stress),
            reliability: ReliabilitySummary {
                retries: 0,
                attempts_run: 1,
                flaky_failures: 0,
                flake_budget: 0,
                coverage_floor: 1.0,
                expected_parallelism: None,
                coverage_ratio: 1.0,
                diagnostics: Vec::new(),
            },
            report_path: None,
        };

        let value = serde_json::to_value(summary).expect("serialize summary");
        let stress = value
            .get("stress")
            .and_then(serde_json::Value::as_object)
            .expect("stress object");
        for field in [
            "worker_count",
            "input_hash",
            "recovered_hash",
            "elapsed_ms",
            "source_symbols",
            "repair_symbols",
            "total_symbols",
            "dropped_symbol_scenario",
            "replay_command",
        ] {
            assert!(stress.contains_key(field), "missing stress field {field}");
        }
    }

    #[test]
    fn run_raptorq_gate_replay_command_names_stress_mode() {
        let options = parse_args_from(args(&["--stress-mode", "smoke"])).expect("parse smoke args");
        let command = replay_command(
            &options,
            std::path::Path::new("target/raptorq_stress_gate/smoke"),
            1024,
            RaptorQParallelismConfig::serial(),
        );
        assert!(command.contains("--stress-mode smoke"));
        assert!(command.contains("--stress-source-bytes 1024"));
        assert!(command.contains("run_raptorq_gate"));
    }
}
