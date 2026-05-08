#![forbid(unsafe_code)]

use fnp_conformance::fnp_python_conformance_shards::{
    FnpPythonConformanceManifest, FnpPythonConformanceShard, ManifestDiagnostic,
    SHARD_REPORT_SCHEMA_VERSION, build_manifest_from_suites, default_repo_root, discover_suites,
    select_shards, shard_ids, validate_manifest,
};
use serde::Serialize;
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug)]
struct Options {
    repo_root: PathBuf,
    report_path: PathBuf,
    shard_selector: String,
    dry_run: bool,
    allow_local: bool,
    list_shards: bool,
}

#[derive(Debug, Serialize)]
struct ManifestEvent<'a> {
    event: &'static str,
    schema_version: &'static str,
    dry_run: bool,
    shard_selector: &'a str,
    selected_shards: Vec<String>,
    diagnostics: &'a [ManifestDiagnostic],
    manifest: &'a FnpPythonConformanceManifest,
}

#[derive(Debug, Serialize)]
struct CommandEvent<'a> {
    event: &'static str,
    schema_version: &'static str,
    dry_run: bool,
    shard_id: &'a str,
    suite_name: &'a str,
    test_file: &'a str,
    command: &'a str,
    status: String,
    exit_code: Option<i32>,
    elapsed_ms: u128,
}

#[derive(Debug, Serialize)]
struct SummaryEvent {
    event: &'static str,
    schema_version: &'static str,
    status: String,
    dry_run: bool,
    shard_selector: String,
    selected_shards: Vec<String>,
    suite_count: usize,
    failed_count: usize,
    report_path: String,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("run_fnp_python_conformance_shards failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let options = parse_args()?;
    let discovered_suites = discover_suites(&options.repo_root)?;
    let discovered = discovered_suites
        .iter()
        .map(|suite| suite.suite_name.clone())
        .collect::<BTreeSet<_>>();
    let manifest = build_manifest_from_suites(discovered_suites);
    let diagnostics = validate_manifest(&manifest, &discovered);
    let selected_shards = select_shards(&manifest, &options.shard_selector)?;

    if options.list_shards {
        print_shards(&manifest)?;
        return Ok(());
    }
    if diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity == "error")
    {
        return Err(format!(
            "manifest validation failed with {} diagnostic(s)",
            diagnostics.len()
        ));
    }

    if !options.dry_run && !options.allow_local && !rch_available() {
        return Err("execution mode requires rch; pass --allow-local to override".to_string());
    }

    if let Some(parent) = options.report_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }
    let mut report = File::create(&options.report_path)
        .map_err(|err| format!("failed creating {}: {err}", options.report_path.display()))?;

    let selected_ids = selected_shards
        .iter()
        .map(|shard| shard.id.clone())
        .collect::<Vec<_>>();
    write_jsonl(
        &mut report,
        &ManifestEvent {
            event: "manifest",
            schema_version: SHARD_REPORT_SCHEMA_VERSION,
            dry_run: options.dry_run,
            shard_selector: &options.shard_selector,
            selected_shards: selected_ids.clone(),
            diagnostics: &diagnostics,
            manifest: &manifest,
        },
    )?;

    let mut failed_count = 0usize;
    let mut suite_count = 0usize;
    for shard in selected_shards {
        for suite in &shard.suites {
            suite_count += 1;
            let start = Instant::now();
            let (status, exit_code) = if options.dry_run {
                ("dry_run".to_string(), None)
            } else {
                let exit_code = run_suite(&options, &suite.suite_name)?;
                let status = if exit_code == 0 { "pass" } else { "fail" };
                if exit_code != 0 {
                    failed_count += 1;
                }
                (status.to_string(), Some(exit_code))
            };
            write_jsonl(
                &mut report,
                &CommandEvent {
                    event: "command",
                    schema_version: SHARD_REPORT_SCHEMA_VERSION,
                    dry_run: options.dry_run,
                    shard_id: &shard.id,
                    suite_name: &suite.suite_name,
                    test_file: &suite.test_file,
                    command: &suite.command,
                    status,
                    exit_code,
                    elapsed_ms: start.elapsed().as_millis(),
                },
            )?;
        }
    }

    let status = if failed_count == 0 { "pass" } else { "fail" };
    write_jsonl(
        &mut report,
        &SummaryEvent {
            event: "summary",
            schema_version: SHARD_REPORT_SCHEMA_VERSION,
            status: status.to_string(),
            dry_run: options.dry_run,
            shard_selector: options.shard_selector,
            selected_shards: selected_ids,
            suite_count,
            failed_count,
            report_path: options.report_path.display().to_string(),
        },
    )?;
    report
        .flush()
        .map_err(|err| format!("failed flushing {}: {err}", options.report_path.display()))?;

    if failed_count == 0 {
        println!(
            "wrote {} with {} suite(s)",
            options.report_path.display(),
            suite_count
        );
        Ok(())
    } else {
        Err(format!("{failed_count} conformance suite(s) failed"))
    }
}

fn parse_args() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let mut repo_root = default_repo_root();
    let mut report_path: Option<PathBuf> = None;
    let mut shard_selector = "fnp-python-smoke".to_string();
    let mut dry_run = true;
    let mut allow_local = false;
    let mut list_shards = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--repo-root" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--repo-root requires a value".to_string())?;
                repo_root = PathBuf::from(value);
            }
            "--report-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--report-path requires a value".to_string())?;
                report_path = Some(PathBuf::from(value));
            }
            "--shard" => {
                shard_selector = args
                    .next()
                    .ok_or_else(|| "--shard requires a value".to_string())?;
            }
            "--dry-run" => dry_run = true,
            "--execute" => dry_run = false,
            "--allow-local" => allow_local = true,
            "--list-shards" => list_shards = true,
            "--help" | "-h" => return Err(usage()),
            other => return Err(format!("unknown argument '{other}'\n{}", usage())),
        }
    }

    let report_path = report_path.unwrap_or_else(default_report_path);
    Ok(Options {
        repo_root,
        report_path,
        shard_selector,
        dry_run,
        allow_local,
        list_shards,
    })
}

fn default_report_path() -> PathBuf {
    default_repo_root().join("artifacts/logs").join(format!(
        "fnp_python_conformance_shards_{}.jsonl",
        unix_timestamp()
    ))
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

fn run_suite(options: &Options, suite_name: &str) -> Result<i32, String> {
    let mut process = if rch_available() {
        let mut process = Command::new("rch");
        process.args([
            "exec",
            "--",
            "cargo",
            "test",
            "-p",
            "fnp-python",
            "--test",
            suite_name,
            "--",
            "--nocapture",
        ]);
        process
    } else {
        let mut process = Command::new("cargo");
        process.args([
            "test",
            "-p",
            "fnp-python",
            "--test",
            suite_name,
            "--",
            "--nocapture",
        ]);
        process
    };
    process.current_dir(&options.repo_root);
    let status = process.status().map_err(|err| {
        format!(
            "failed running suite '{}' from {}: {err}",
            suite_name,
            options.repo_root.display()
        )
    })?;
    Ok(status.code().unwrap_or(1))
}

fn rch_available() -> bool {
    Command::new("rch")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
}

fn print_shards(manifest: &FnpPythonConformanceManifest) -> Result<(), String> {
    let summary = manifest
        .shards
        .iter()
        .map(shard_summary)
        .collect::<Vec<_>>();
    let raw = serde_json::to_string_pretty(&summary)
        .map_err(|err| format!("failed serializing shard summary: {err}"))?;
    println!("{raw}");
    println!("available shard ids: {}", shard_ids(manifest).join(", "));
    Ok(())
}

fn shard_summary(shard: &FnpPythonConformanceShard) -> serde_json::Value {
    serde_json::json!({
        "id": shard.id,
        "domain": shard.domain,
        "expected_cost": shard.expected_cost,
        "suite_count": shard.suite_count,
    })
}

fn write_jsonl<T: Serialize>(writer: &mut File, value: &T) -> Result<(), String> {
    let raw = serde_json::to_string(value)
        .map_err(|err| format!("failed serializing JSONL event: {err}"))?;
    writeln!(writer, "{raw}").map_err(|err| format!("failed writing JSONL event: {err}"))
}

fn usage() -> String {
    "Usage: cargo run -p fnp-conformance --bin run_fnp_python_conformance_shards -- [--repo-root <path>] [--report-path <path>] [--shard <id|all>] [--dry-run|--execute] [--allow-local] [--list-shards]".to_string()
}
