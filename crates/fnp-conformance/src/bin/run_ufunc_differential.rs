#![forbid(unsafe_code)]

use fnp_conformance::HarnessConfig;
use fnp_conformance::ufunc_differential::{compare_against_oracle, write_differential_report};

fn main() {
    if let Err(err) = run() {
        eprintln!("run_ufunc_differential failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let cfg = HarnessConfig::default_paths();
    let input_path = cfg.fixture_root.join("ufunc_input_cases.json");
    let oracle_path = cfg
        .fixture_root
        .join("oracle_outputs/ufunc_oracle_output.json");
    let report_path = cfg
        .fixture_root
        .join("oracle_outputs/ufunc_differential_report.json");

    let report = compare_against_oracle(&input_path, &oracle_path, 1e-9, 1e-9)?;
    write_differential_report(&report_path, &report)?;

    println!(
        "ufunc differential: total={} passed={} failed={}",
        report.total_cases, report.passed_cases, report.failed_cases
    );
    println!("wrote {}", report_path.display());
    // A differential that cannot fail is not a gate. Until 2026-09-03 this binary exited
    // 0 whatever `failed_cases` said, so CI's G3 accepted 17 parity regressions that the
    // committed April report had at 381/381 (deadlock-audit-r5fy8). The report is still
    // written first so the failing cases are on disk for whoever reads the red run.
    if report.failed_cases > 0 {
        return Err(format!(
            "{} of {} ufunc differential cases diverge from the oracle; see {}",
            report.failed_cases,
            report.total_cases,
            report_path.display()
        ));
    }
    Ok(())
}
