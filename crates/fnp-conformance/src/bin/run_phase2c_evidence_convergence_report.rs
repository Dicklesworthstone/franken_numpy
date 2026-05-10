#![forbid(unsafe_code)]

use fnp_conformance::contract_schema::{
    build_phase2c_evidence_convergence_report, write_phase2c_evidence_convergence_report_json,
    write_phase2c_evidence_convergence_report_markdown,
};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Json,
    Markdown,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("run_phase2c_evidence_convergence_report failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut ledger: Option<PathBuf> = None;
    let mut phase2c_root: Option<PathBuf> = None;
    let mut report_path: Option<PathBuf> = None;
    let mut markdown_report_path: Option<PathBuf> = None;
    let mut format = OutputFormat::Json;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--ledger" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--ledger requires a value".to_string())?;
                ledger = Some(PathBuf::from(value));
            }
            "--phase2c-root" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--phase2c-root requires a value".to_string())?;
                phase2c_root = Some(PathBuf::from(value));
            }
            "--report-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--report-path requires a value".to_string())?;
                report_path = Some(PathBuf::from(value));
            }
            "--markdown-report-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--markdown-report-path requires a value".to_string())?;
                markdown_report_path = Some(PathBuf::from(value));
            }
            "--format" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--format requires a value".to_string())?;
                format = match value.as_str() {
                    "json" => OutputFormat::Json,
                    "markdown" | "md" => OutputFormat::Markdown,
                    _ => return Err(format!("unknown --format value: {value}")),
                };
            }
            "--help" | "-h" => {
                println!("{}", usage());
                return Ok(());
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let ledger = ledger.unwrap_or_else(|| {
        repo_root.join("artifacts/contracts/PORTING_ESSENCE_EXTRACTION_LEDGER_V1.md")
    });
    let phase2c_root = phase2c_root.unwrap_or_else(|| repo_root.join("artifacts/phase2c"));
    let report_path = report_path.unwrap_or_else(|| {
        repo_root.join("artifacts/logs/phase2c_evidence_convergence_report.json")
    });
    let markdown_report_path = markdown_report_path
        .unwrap_or_else(|| repo_root.join("artifacts/logs/phase2c_evidence_convergence_report.md"));

    let report = build_phase2c_evidence_convergence_report(&ledger, &phase2c_root)?;
    write_phase2c_evidence_convergence_report_json(&report_path, &report)?;
    write_phase2c_evidence_convergence_report_markdown(&markdown_report_path, &report)?;

    match format {
        OutputFormat::Json => {
            let raw = serde_json::to_string_pretty(&report)
                .map_err(|err| format!("failed serializing convergence report: {err}"))?;
            println!("{raw}");
        }
        OutputFormat::Markdown => print!("{}", report.to_markdown()),
    }

    eprintln!(
        "wrote json={} markdown={}",
        report_path.display(),
        markdown_report_path.display()
    );

    Ok(())
}

fn usage() -> &'static str {
    concat!(
        "Usage: cargo run -p fnp-conformance --bin run_phase2c_evidence_convergence_report -- \\\n",
        "  [--ledger <path>] [--phase2c-root <path>] [--report-path <path>] \\\n",
        "  [--markdown-report-path <path>] [--format json|markdown]"
    )
}
