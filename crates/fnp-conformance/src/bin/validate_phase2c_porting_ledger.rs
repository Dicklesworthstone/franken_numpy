#![forbid(unsafe_code)]

use fnp_conformance::contract_schema::{
    validate_phase2c_porting_ledger, write_porting_ledger_freshness_report,
};
use std::path::PathBuf;

fn main() {
    if let Err(err) = run() {
        eprintln!("validate_phase2c_porting_ledger failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut ledger: Option<PathBuf> = None;
    let mut phase2c_root: Option<PathBuf> = None;
    let mut report_out: Option<PathBuf> = None;

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
            "--report-out" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--report-out requires a value".to_string())?;
                report_out = Some(PathBuf::from(value));
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run -p fnp-conformance --bin validate_phase2c_porting_ledger -- \\\n  [--ledger <path>] [--phase2c-root <path>] [--report-out <path>]"
                );
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
    let report_out = report_out.unwrap_or_else(|| {
        repo_root.join("artifacts/logs/phase2c_porting_ledger_freshness_report.json")
    });

    let report = validate_phase2c_porting_ledger(&ledger, &phase2c_root)?;
    write_porting_ledger_freshness_report(&report_out, &report)?;

    println!(
        "ledger_status={} checked_packets={} ready_packets={} stale_rows={} diagnostics={}",
        report.status,
        report.checked_packet_count,
        report.ready_packet_count,
        report.stale_row_count,
        report.diagnostics.len()
    );
    println!("report={}", report_out.display());

    if !report.is_fresh() {
        std::process::exit(2);
    }
    Ok(())
}
