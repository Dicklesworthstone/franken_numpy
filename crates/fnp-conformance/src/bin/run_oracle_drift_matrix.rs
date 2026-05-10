#[path = "../oracle_drift_matrix.rs"]
mod oracle_drift_matrix;

use oracle_drift_matrix::{
    DEFAULT_REPORT_PATH, LaneSpec, default_fixture_ids, default_lane_specs, parse_lane_spec,
    run_report,
};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug)]
struct Options {
    lanes: Vec<LaneSpec>,
    fixtures: Vec<String>,
    report_path: PathBuf,
}

fn main() -> ExitCode {
    match parse_args(std::env::args().skip(1)) {
        Ok(options) => {
            let report = run_report(&options.lanes, &options.fixtures);
            if let Err(error) = report.write_pretty(&options.report_path) {
                eprintln!("run_oracle_drift_matrix failed: {error}");
                return ExitCode::from(1);
            }
            println!(
                "oracle_drift_matrix report={} available_lanes={} stable={} divergent={} unavailable={}",
                options.report_path.display(),
                report.summary.available_lanes,
                report.summary.stable_fixtures,
                report.summary.divergent_fixtures,
                report.summary.unavailable_fixtures
            );
            if report.has_required_lane_failure() {
                eprintln!(
                    "required oracle lanes unavailable: {}",
                    report.summary.unavailable_required_lanes.join(",")
                );
                ExitCode::from(2)
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(error) => {
            eprintln!("{error}");
            eprintln!("{}", usage());
            ExitCode::from(2)
        }
    }
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Options, String> {
    let mut lanes = Vec::new();
    let mut fixtures = Vec::new();
    let mut report_path = PathBuf::from(DEFAULT_REPORT_PATH);
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--smoke" => {
                fixtures = default_fixture_ids();
            }
            "--fixture" => {
                fixtures.push(
                    args.next()
                        .ok_or_else(|| "--fixture requires a value".to_string())?,
                );
            }
            "--lane" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "--lane requires id=python".to_string())?;
                lanes.push(parse_lane_spec(&raw, false)?);
            }
            "--required-lane" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "--required-lane requires id=python".to_string())?;
                lanes.push(parse_lane_spec(&raw, true)?);
            }
            "--report-path" => {
                report_path = PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--report-path requires a value".to_string())?,
                );
            }
            "--help" | "-h" => return Err(usage()),
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    if lanes.is_empty() {
        lanes = default_lane_specs();
    }
    if fixtures.is_empty() {
        fixtures = default_fixture_ids();
    }
    Ok(Options {
        lanes,
        fixtures,
        report_path,
    })
}

fn usage() -> String {
    "Usage: cargo run -p fnp-conformance --bin run_oracle_drift_matrix -- [--smoke] [--report-path <path>] [--fixture <id>] [--lane <id=python>] [--required-lane <id=python>]".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_drift_matrix_cli_defaults_to_env_lanes_and_smoke_fixtures() {
        let options = parse_args(Vec::<String>::new()).unwrap();

        assert!(!options.lanes.is_empty());
        assert_eq!(options.fixtures, default_fixture_ids());
        assert_eq!(options.report_path, PathBuf::from(DEFAULT_REPORT_PATH));
    }

    #[test]
    fn oracle_drift_matrix_cli_accepts_required_and_optional_lanes() {
        let options = parse_args([
            "--required-lane".to_string(),
            "numpy2=/opt/numpy2/python".to_string(),
            "--lane".to_string(),
            "nightly=/opt/nightly/python".to_string(),
            "--fixture".to_string(),
            "dtype_promotion_int_float".to_string(),
            "--report-path".to_string(),
            "target/custom.json".to_string(),
        ])
        .unwrap();

        assert_eq!(options.lanes.len(), 2);
        assert!(options.lanes[0].required);
        assert!(!options.lanes[1].required);
        assert_eq!(options.fixtures, ["dtype_promotion_int_float"]);
        assert_eq!(options.report_path, PathBuf::from("target/custom.json"));
    }
}
