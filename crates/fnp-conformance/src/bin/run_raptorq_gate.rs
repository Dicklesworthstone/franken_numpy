#![forbid(unsafe_code)]

use fnp_conformance::{HarnessConfig, SuiteReport, raptorq_artifacts};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct SuiteSummary {
    suite: String,
    case_count: usize,
    pass_count: usize,
    failures: Vec<String>,
}

#[derive(Debug, Serialize)]
struct GateSummary {
    status: &'static str,
    suites: Vec<SuiteSummary>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("run_raptorq_gate failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        if args.len() == 1 && matches!(args[0].as_str(), "--help" | "-h") {
            println!("Usage: cargo run -p fnp-conformance --bin run_raptorq_gate");
            return Ok(());
        }
        return Err(format!("unknown arguments: {}", args.join(" ")));
    }

    let cfg = HarnessConfig::default_paths();
    let suite = raptorq_artifacts::run_raptorq_artifact_suite(&cfg)?;
    let status = if suite.all_passed() { "pass" } else { "fail" };

    let summary = GateSummary {
        status,
        suites: vec![summarize_suite(suite)],
    };

    let summary_json = serde_json::to_string_pretty(&summary)
        .map_err(|err| format!("failed serializing summary: {err}"))?;
    println!("{summary_json}");

    if status == "fail" {
        std::process::exit(2);
    }
    Ok(())
}

fn summarize_suite(report: SuiteReport) -> SuiteSummary {
    SuiteSummary {
        suite: report.suite.to_string(),
        case_count: report.case_count,
        pass_count: report.pass_count,
        failures: report.failures,
    }
}
