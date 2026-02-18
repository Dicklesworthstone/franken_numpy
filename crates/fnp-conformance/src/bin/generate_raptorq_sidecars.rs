#![forbid(unsafe_code)]

use fnp_conformance::raptorq_artifacts::generate_bundle_sidecar_and_reports;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    if let Err(err) = run() {
        eprintln!("generate_raptorq_sidecars failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let emit_artifact_markers = parse_args()?;
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture_root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fnp-conformance/fixtures");

    let conformance_sidecar_path =
        repo_root.join("artifacts/raptorq/conformance_bundle_v1.sidecar.json");
    let conformance_scrub_path =
        repo_root.join("artifacts/raptorq/conformance_bundle_v1.scrub_report.json");
    let conformance_decode_proof_path =
        repo_root.join("artifacts/raptorq/conformance_bundle_v1.decode_proof.json");

    let conformance_files = vec![
        fixture_root.join("ufunc_input_cases.json"),
        fixture_root.join("workflow_scenario_corpus.json"),
        fixture_root.join("oracle_outputs/ufunc_oracle_output.json"),
        fixture_root.join("oracle_outputs/ufunc_differential_report.json"),
    ];

    generate_bundle_sidecar_and_reports(
        "conformance_bundle_v1",
        &repo_root,
        &conformance_files,
        &conformance_sidecar_path,
        &conformance_scrub_path,
        &conformance_decode_proof_path,
        1001,
    )?;

    let benchmark_files = vec![repo_root.join("artifacts/baselines/ufunc_benchmark_baseline.json")];

    let benchmark_sidecar_path =
        repo_root.join("artifacts/raptorq/benchmark_bundle_v1.sidecar.json");
    let benchmark_scrub_path =
        repo_root.join("artifacts/raptorq/benchmark_bundle_v1.scrub_report.json");
    let benchmark_decode_proof_path =
        repo_root.join("artifacts/raptorq/benchmark_bundle_v1.decode_proof.json");

    generate_bundle_sidecar_and_reports(
        "benchmark_bundle_v1",
        &repo_root,
        &benchmark_files,
        &benchmark_sidecar_path,
        &benchmark_scrub_path,
        &benchmark_decode_proof_path,
        1002,
    )?;

    if emit_artifact_markers {
        emit_artifact_with_markers(&repo_root, &conformance_sidecar_path)?;
        emit_artifact_with_markers(&repo_root, &conformance_scrub_path)?;
        emit_artifact_with_markers(&repo_root, &conformance_decode_proof_path)?;
    }

    println!("generated RaptorQ sidecars and reports for conformance + benchmark bundles");
    Ok(())
}

fn parse_args() -> Result<bool, String> {
    let mut emit_artifact_markers = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--emit-artifact-markers" => {
                emit_artifact_markers = true;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run -p fnp-conformance --bin generate_raptorq_sidecars -- [--emit-artifact-markers]"
                );
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }
    Ok(emit_artifact_markers)
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
