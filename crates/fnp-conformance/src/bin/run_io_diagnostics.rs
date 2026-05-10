#![forbid(unsafe_code)]

#[path = "../io_diagnostics.rs"]
mod io_diagnostics;

use fnp_conformance::diagnostic_oracle::{
    DiagnosticOracleOptions, DiagnosticVerdictStatus, write_jsonl, write_report,
};
use io_diagnostics::{io_diagnostic_cases, run_io_diagnostics};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Debug)]
struct Options {
    report_path: Option<PathBuf>,
    jsonl_path: Option<PathBuf>,
    python: Option<String>,
    require_numpy: bool,
    list_cases: bool,
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
        eprintln!("run_io_diagnostics failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let options = parse_args(std::env::args().skip(1))?;
    if options.list_cases {
        for case in io_diagnostic_cases() {
            println!("{}\t{}", case.id, case.surface);
        }
        return Ok(());
    }

    let oracle_options = DiagnosticOracleOptions {
        python: options
            .python
            .unwrap_or_else(fnp_conformance::diagnostic_oracle::resolve_oracle_python),
        require_numpy: options.require_numpy,
    };
    let run = run_io_diagnostics(&oracle_options)?;
    if let Some(path) = options.report_path.as_ref() {
        write_report(path, &run.report)?;
    }
    if let Some(path) = options.jsonl_path.as_ref() {
        write_jsonl(path, &run.report)?;
    }

    let fail_count = run
        .verdicts
        .iter()
        .filter(|verdict| verdict.status == DiagnosticVerdictStatus::Fail)
        .count();
    let skipped_count = run
        .verdicts
        .iter()
        .filter(|verdict| verdict.status == DiagnosticVerdictStatus::Skipped)
        .count();
    let pass_count = run.verdicts.len() - fail_count - skipped_count;
    let summary = RunnerSummary {
        status: if fail_count == 0 { "pass" } else { "fail" },
        case_count: run.case_count,
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
        Err(format!("IO diagnostics reported {fail_count} failure(s)"))
    }
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Options, String> {
    let mut options = Options {
        report_path: None,
        jsonl_path: None,
        python: None,
        require_numpy: true,
        list_cases: false,
    };
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
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
            "--allow-missing-numpy" => options.require_numpy = false,
            "--list-cases" => options.list_cases = true,
            "--help" | "-h" => return Err(usage()),
            unknown => return Err(format!("unknown argument {unknown}\n{}", usage())),
        }
    }
    Ok(options)
}

fn usage() -> String {
    "Usage: run_io_diagnostics [--report-path <path>] [--jsonl-path <path>] [--python <path>] [--allow-missing-numpy] [--list-cases]".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_diagnostics_cli_accepts_report_paths() {
        let options = parse_args([
            "--report-path".to_string(),
            "target/io_diagnostics.json".to_string(),
            "--jsonl-path".to_string(),
            "target/io_diagnostics.jsonl".to_string(),
            "--python".to_string(),
            "/opt/numpy/bin/python".to_string(),
        ])
        .expect("parse options");

        assert_eq!(
            options.report_path,
            Some(PathBuf::from("target/io_diagnostics.json"))
        );
        assert_eq!(
            options.jsonl_path,
            Some(PathBuf::from("target/io_diagnostics.jsonl"))
        );
        assert_eq!(options.python, Some("/opt/numpy/bin/python".to_string()));
    }

    #[test]
    fn io_diagnostics_cli_can_list_cases_without_running_numpy() {
        let options = parse_args(["--list-cases".to_string()]).expect("parse list mode");

        assert!(options.list_cases);
        assert!(options.require_numpy);
    }
}
