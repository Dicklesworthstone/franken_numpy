#![forbid(unsafe_code)]

use fnp_conformance::contract_schema::{validate_phase2c_packet, write_packet_readiness_report};
use std::path::PathBuf;

fn main() {
    if let Err(err) = run() {
        eprintln!("validate_phase2c_packet failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut packet_id: Option<String> = None;
    let mut phase2c_root: Option<PathBuf> = None;
    let mut report_out: Option<PathBuf> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--packet-id" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--packet-id requires a value".to_string())?;
                packet_id = Some(value);
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
                    "Usage: cargo run -p fnp-conformance --bin validate_phase2c_packet -- \\\n  --packet-id <FNP-P2C-XXX> [--phase2c-root <path>] [--report-out <path>]"
                );
                return Ok(());
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }

    let packet_id = packet_id.ok_or_else(|| "--packet-id is required".to_string())?;
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let phase2c_root = phase2c_root.unwrap_or_else(|| repo_root.join("artifacts/phase2c"));
    let packet_dir = phase2c_root.join(&packet_id);
    let report_out = report_out.unwrap_or_else(|| packet_dir.join("packet_readiness_report.json"));

    let report = validate_phase2c_packet(&packet_id, &packet_dir);
    write_packet_readiness_report(&report_out, &report)?;

    println!(
        "packet={} status={} missing_artifacts={} missing_fields={} parse_errors={}",
        packet_id,
        report.status,
        report.missing_artifacts.len(),
        report.missing_fields.len(),
        report.parse_errors.len()
    );
    println!("report={}", report_out.display());

    if !report.is_ready() {
        std::process::exit(2);
    }
    Ok(())
}
