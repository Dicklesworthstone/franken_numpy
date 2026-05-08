#![forbid(unsafe_code)]

use fnp_conformance::fnp_python_api_coverage::{build_api_coverage_report, default_report_path};
use fnp_conformance::fnp_python_conformance_shards::default_repo_root;
use std::fs;
use std::path::PathBuf;

fn main() {
    if let Err(err) = run() {
        eprintln!("run_fnp_python_api_coverage failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let options = Options::parse()?;
    let report = build_api_coverage_report(&options.repo_root, options.fail_on_missing)?;
    if let Some(parent) = options.report_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("create report dir {}: {err}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(&report)
        .map_err(|err| format!("serialize coverage report: {err}"))?;
    fs::write(&options.report_path, format!("{json}\n"))
        .map_err(|err| format!("write {}: {err}", options.report_path.display()))?;

    println!(
        "wrote {}: exports={} covered={} missing={} excluded={} orphan_suites={}",
        options.report_path.display(),
        report.export_count,
        report.covered_count,
        report.missing_count,
        report.excluded_count,
        report.orphan_suite_count
    );

    if options.fail_on_missing && report.has_missing_exports() {
        return Err(format!(
            "{} public fnp-python export(s) lack conformance evidence or an explicit exclusion",
            report.missing_count
        ));
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct Options {
    repo_root: PathBuf,
    report_path: PathBuf,
    fail_on_missing: bool,
}

impl Options {
    fn parse() -> Result<Self, String> {
        let mut repo_root = default_repo_root();
        let mut report_path = default_report_path();
        let mut fail_on_missing = false;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--repo-root" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "--repo-root requires a value".to_string())?;
                    repo_root = PathBuf::from(value);
                }
                "--report-path" | "--output-path" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("{arg} requires a value"))?;
                    report_path = PathBuf::from(value);
                }
                "--fail-on-missing" => {
                    fail_on_missing = true;
                }
                "--help" | "-h" => {
                    return Err(usage());
                }
                other => return Err(format!("unknown argument '{other}'\n{}", usage())),
            }
        }

        Ok(Self {
            repo_root,
            report_path,
            fail_on_missing,
        })
    }
}

fn usage() -> String {
    "Usage: cargo run -p fnp-conformance --bin run_fnp_python_api_coverage -- [--repo-root <path>] [--report-path <path>] [--fail-on-missing]".to_string()
}
