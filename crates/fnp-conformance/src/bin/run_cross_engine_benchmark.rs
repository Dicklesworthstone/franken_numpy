#![forbid(unsafe_code)]

use fnp_conformance::cross_engine_benchmark::{
    DEFAULT_CROSS_ENGINE_BENCHMARK_MANIFEST, DEFAULT_CROSS_ENGINE_BENCHMARK_OUTPUT,
    run_cross_engine_benchmark,
};
use std::path::{Path, PathBuf};

fn main() {
    if let Err(err) = run() {
        eprintln!("run_cross_engine_benchmark failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let options = parse_args(&repo_root)?;
    let report = run_cross_engine_benchmark(
        &repo_root,
        &options.manifest_path,
        &options.output_path,
        options.quick,
        options.oracle_python.as_deref(),
    )?;
    println!(
        "generated cross-engine benchmark with {} workloads",
        report.workloads.len()
    );
    println!("wrote {}", options.output_path.display());
    Ok(())
}

struct RunOptions {
    manifest_path: PathBuf,
    output_path: PathBuf,
    quick: bool,
    oracle_python: Option<String>,
}

fn parse_args(repo_root: &Path) -> Result<RunOptions, String> {
    let mut manifest_path = repo_root.join(DEFAULT_CROSS_ENGINE_BENCHMARK_MANIFEST);
    let mut output_path = repo_root.join(DEFAULT_CROSS_ENGINE_BENCHMARK_OUTPUT);
    let mut quick = false;
    let mut oracle_python = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--manifest-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--manifest-path requires a value".to_string())?;
                manifest_path = PathBuf::from(value);
            }
            "--output-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--output-path requires a value".to_string())?;
                output_path = PathBuf::from(value);
            }
            "--oracle-python" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--oracle-python requires a value".to_string())?;
                oracle_python = Some(value);
            }
            "--quick" => {
                quick = true;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run -p fnp-conformance --bin run_cross_engine_benchmark -- [--manifest-path <path>] [--output-path <path>] [--oracle-python <path>] [--quick]"
                );
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }

    Ok(RunOptions {
        manifest_path,
        output_path,
        quick,
        oracle_python,
    })
}
