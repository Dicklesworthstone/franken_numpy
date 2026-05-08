#![forbid(unsafe_code)]

use fnp_conformance::swarm_handoff::{
    build_swarm_handoff_report_from_paths, default_conformance_manifest_path, default_issues_path,
    default_output_path, read_bv_robot_triage, render_terminal_report, write_report_json,
};
use std::fs;
use std::path::PathBuf;

fn main() {
    if let Err(err) = run() {
        eprintln!("generate_swarm_handoff_report failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let options = Options::parse()?;
    let (bv_source, bv_json) = match &options.bv_json_path {
        Some(path) => {
            let raw = fs::read_to_string(path)
                .map_err(|err| format!("read --bv-json-path {}: {err}", path.display()))?;
            (path.display().to_string(), raw)
        }
        None => (
            "bv --robot-triage".to_string(),
            read_bv_robot_triage(&options.repo_root)?,
        ),
    };

    let report = build_swarm_handoff_report_from_paths(
        &options.repo_root,
        &options.issues_path,
        &bv_source,
        &bv_json,
        &options.conformance_manifest_path,
    )?;
    write_report_json(&report, &options.output_path)?;

    print!("{}", render_terminal_report(&report));
    println!("wrote {}", options.output_path.display());
    Ok(())
}

#[derive(Debug, Clone)]
struct Options {
    repo_root: PathBuf,
    issues_path: PathBuf,
    bv_json_path: Option<PathBuf>,
    conformance_manifest_path: PathBuf,
    output_path: PathBuf,
}

impl Options {
    fn parse() -> Result<Self, String> {
        let mut repo_root = default_repo_root();
        let mut issues_path = None;
        let mut bv_json_path = None;
        let mut conformance_manifest_path = None;
        let mut output_path = None;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--repo-root" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "--repo-root requires a value".to_string())?;
                    repo_root = PathBuf::from(value);
                }
                "--issues-path" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "--issues-path requires a value".to_string())?;
                    issues_path = Some(PathBuf::from(value));
                }
                "--bv-json-path" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "--bv-json-path requires a value".to_string())?;
                    bv_json_path = Some(PathBuf::from(value));
                }
                "--conformance-manifest-path" | "--manifest-path" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("{arg} requires a value"))?;
                    conformance_manifest_path = Some(PathBuf::from(value));
                }
                "--output" | "--output-path" | "--report-path" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("{arg} requires a value"))?;
                    output_path = Some(PathBuf::from(value));
                }
                "--help" | "-h" => {
                    println!("{}", usage());
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument '{other}'\n{}", usage())),
            }
        }

        let issues_path = issues_path.unwrap_or_else(|| default_issues_path(&repo_root));
        let conformance_manifest_path = conformance_manifest_path
            .unwrap_or_else(|| default_conformance_manifest_path(&repo_root));
        let output_path = output_path.unwrap_or_else(|| default_output_path(&repo_root));

        Ok(Self {
            repo_root,
            issues_path,
            bv_json_path,
            conformance_manifest_path,
            output_path,
        })
    }
}

fn default_repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn usage() -> String {
    "Usage: cargo run -p fnp-conformance --bin generate_swarm_handoff_report -- [--repo-root <path>] [--issues-path <path>] [--bv-json-path <path>] [--conformance-manifest-path <path>] [--output <path>]".to_string()
}
