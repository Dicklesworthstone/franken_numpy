#![forbid(unsafe_code)]

use fnp_conformance::diagnostic_oracle::{
    DiagnosticOracleOptions, evaluate_diagnostic_report, load_cases, run_diagnostic_oracle,
    smoke_cases, write_jsonl, write_report,
};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Debug)]
struct Options {
    case_json: Option<PathBuf>,
    report_path: Option<PathBuf>,
    jsonl_path: Option<PathBuf>,
    python: Option<String>,
    smoke: bool,
    require_numpy: bool,
}

#[derive(Debug, Serialize)]
struct RunnerSummary {
    status: &'static str,
    case_count: usize,
    pass_count: usize,
    fail_count: usize,
    skipped_count: usize,
    report_path: Option<String>,
    jsonl_path: Option<String>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("run_diagnostic_oracle failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let options = parse_args()?;
    let cases = match (options.smoke, options.case_json.as_ref()) {
        (true, _) | (_, None) => smoke_cases(),
        (false, Some(path)) => load_cases(path)?,
    };
    let oracle_options = DiagnosticOracleOptions {
        python: options
            .python
            .unwrap_or_else(fnp_conformance::diagnostic_oracle::resolve_oracle_python),
        require_numpy: options.require_numpy,
    };
    let report = run_diagnostic_oracle(&cases, &oracle_options)?;
    let verdicts = evaluate_diagnostic_report(&cases, &report);
    if let Some(path) = options.report_path.as_ref() {
        write_report(path, &report)?;
    }
    if let Some(path) = options.jsonl_path.as_ref() {
        write_jsonl(path, &report)?;
    }
    let fail_count = verdicts
        .iter()
        .filter(|verdict| {
            verdict.status == fnp_conformance::diagnostic_oracle::DiagnosticVerdictStatus::Fail
        })
        .count();
    let skipped_count = verdicts
        .iter()
        .filter(|verdict| {
            verdict.status == fnp_conformance::diagnostic_oracle::DiagnosticVerdictStatus::Skipped
        })
        .count();
    let pass_count = verdicts.len() - fail_count - skipped_count;
    let summary = RunnerSummary {
        status: if fail_count == 0 { "pass" } else { "fail" },
        case_count: verdicts.len(),
        pass_count,
        fail_count,
        skipped_count,
        report_path: options
            .report_path
            .as_ref()
            .map(|path| path.display().to_string()),
        jsonl_path: options
            .jsonl_path
            .as_ref()
            .map(|path| path.display().to_string()),
    };
    let raw = serde_json::to_string_pretty(&summary)
        .map_err(|err| format!("serialize runner summary: {err}"))?;
    println!("{raw}");
    if fail_count == 0 {
        Ok(())
    } else {
        Err(format!(
            "diagnostic oracle reported {fail_count} failure(s)"
        ))
    }
}

fn parse_args() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let mut options = Options {
        case_json: None,
        report_path: None,
        jsonl_path: None,
        python: None,
        smoke: false,
        require_numpy: true,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--case-json" => {
                options.case_json = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--case-json requires a path".to_string())?,
                ));
            }
            "--report-path" => {
                options.report_path =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        "--report-path requires a path".to_string()
                    })?));
            }
            "--jsonl-path" => {
                options.jsonl_path = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--jsonl-path requires a path".to_string())?,
                ));
            }
            "--python" => {
                options.python = Some(
                    args.next()
                        .ok_or_else(|| "--python requires a value".to_string())?,
                );
            }
            "--smoke" => options.smoke = true,
            "--allow-missing-numpy" => options.require_numpy = false,
            "--help" | "-h" => return Err(usage()),
            unknown => return Err(format!("unknown argument {unknown}\n{}", usage())),
        }
    }
    if options.case_json.is_none() {
        options.smoke = true;
    }
    Ok(options)
}

fn usage() -> String {
    "Usage: run_diagnostic_oracle [--smoke] [--case-json <path>] [--report-path <path>] [--jsonl-path <path>] [--python <path>] [--allow-missing-numpy]".to_string()
}
