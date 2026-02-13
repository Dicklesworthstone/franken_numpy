#![forbid(unsafe_code)]

use fnp_conformance::{
    HarnessConfig, run_runtime_policy_adversarial_suite, run_runtime_policy_suite,
    security_contracts, set_runtime_policy_log_path,
};
use serde::Serialize;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Serialize)]
struct SuiteSummary {
    suite: &'static str,
    case_count: usize,
    pass_count: usize,
    failures: Vec<String>,
}

#[derive(Debug, Serialize)]
struct GateSummary {
    status: &'static str,
    runtime_policy_log: String,
    suites: Vec<SuiteSummary>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("run_security_gate failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut log_path: Option<PathBuf> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--log-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--log-path requires a value".to_string())?;
                log_path = Some(PathBuf::from(value));
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run -p fnp-conformance --bin run_security_gate -- [--log-path <path>]"
                );
                return Ok(());
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
    set_runtime_policy_log_path(Some(log_path.clone()));

    let cfg = HarnessConfig::default_paths();
    let runtime = run_runtime_policy_suite(&cfg)?;
    let adversarial = run_runtime_policy_adversarial_suite(&cfg)?;
    let contracts = security_contracts::run_security_contract_suite(&cfg)?;

    let suites = vec![runtime, adversarial, contracts];
    let status = if suites.iter().all(|suite| suite.all_passed()) {
        "pass"
    } else {
        "fail"
    };

    let summary = GateSummary {
        status,
        runtime_policy_log: log_path.display().to_string(),
        suites: suites
            .into_iter()
            .map(|suite| SuiteSummary {
                suite: suite.suite,
                case_count: suite.case_count,
                pass_count: suite.pass_count,
                failures: suite.failures,
            })
            .collect(),
    };

    let summary_json = serde_json::to_string_pretty(&summary)
        .map_err(|err| format!("failed serializing summary: {err}"))?;
    println!("{summary_json}");

    if status == "fail" {
        std::process::exit(2);
    }
    Ok(())
}
