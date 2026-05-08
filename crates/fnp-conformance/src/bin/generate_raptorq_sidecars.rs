#![forbid(unsafe_code)]

use fnp_conformance::raptorq_artifacts::{
    RaptorQParallelismConfig, generate_default_bundle_sidecars_and_reports,
};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy)]
struct GenerateOptions {
    emit_artifact_markers: bool,
    parallelism: RaptorQParallelismConfig,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("generate_raptorq_sidecars failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let options = parse_args()?;
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");

    let conformance_sidecar_path =
        repo_root.join("artifacts/raptorq/conformance_bundle_v1.sidecar.json");
    let conformance_scrub_path =
        repo_root.join("artifacts/raptorq/conformance_bundle_v1.scrub_report.json");
    let conformance_decode_proof_path =
        repo_root.join("artifacts/raptorq/conformance_bundle_v1.decode_proof.json");

    generate_default_bundle_sidecars_and_reports(&repo_root, options.parallelism)?;

    if options.emit_artifact_markers {
        emit_artifact_with_markers(&repo_root, &conformance_sidecar_path)?;
        emit_artifact_with_markers(&repo_root, &conformance_scrub_path)?;
        emit_artifact_with_markers(&repo_root, &conformance_decode_proof_path)?;
    }

    println!("generated RaptorQ sidecars and reports for conformance + benchmark bundles");
    Ok(())
}

fn parse_args() -> Result<GenerateOptions, String> {
    let mut emit_artifact_markers = false;
    let mut parallelism = RaptorQParallelismConfig::serial();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--emit-artifact-markers" => {
                emit_artifact_markers = true;
            }
            "--parallelism" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--parallelism requires a value".to_string())?;
                let worker_count = value
                    .parse::<usize>()
                    .map_err(|err| format!("invalid --parallelism value '{value}': {err}"))?;
                parallelism = RaptorQParallelismConfig::from_worker_count(worker_count)?;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run -p fnp-conformance --bin generate_raptorq_sidecars -- [--parallelism <n>] [--emit-artifact-markers]"
                );
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }
    Ok(GenerateOptions {
        emit_artifact_markers,
        parallelism,
    })
}

fn emit_artifact_with_markers(repo_root: &Path, path: &Path) -> Result<(), String> {
    let marker_path = path.strip_prefix(repo_root).unwrap_or(path);
    let raw = fs::read_to_string(path).map_err(|err| {
        format!(
            "failed reading {} for marker emission: {err}",
            path.display()
        )
    })?;
    println!("BEGIN_FILE:{}", marker_path.display());
    print!("{raw}");
    if !raw.ends_with('\n') {
        println!();
    }
    println!("END_FILE:{}", marker_path.display());
    Ok(())
}
