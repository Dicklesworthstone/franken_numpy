#![forbid(unsafe_code)]

use fnp_conformance::benchmark::generate_benchmark_baseline;
use std::path::PathBuf;

fn main() {
    if let Err(err) = run() {
        eprintln!("generate_benchmark_baseline failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let repo_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output_path = parse_output_path(&repo_root)?;

    let baseline = generate_benchmark_baseline(&repo_root, &output_path)?;
    println!(
        "generated benchmark baseline with {} workloads",
        baseline.workloads.len()
    );
    println!("wrote {}", output_path.display());
    Ok(())
}

fn parse_output_path(repo_root: &std::path::Path) -> Result<PathBuf, String> {
    let mut output_path: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--output-path requires a value".to_string())?;
                output_path = Some(PathBuf::from(value));
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run -p fnp-conformance --bin generate_benchmark_baseline -- [--output-path <path>]"
                );
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }

    Ok(output_path
        .unwrap_or_else(|| repo_root.join("artifacts/baselines/ufunc_benchmark_baseline.json")))
}
