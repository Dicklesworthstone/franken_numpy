#![forbid(unsafe_code)]

use fnp_conformance::diagnostic_oracle::load_cases;
use fnp_conformance::divergence_ledger::{
    DEFAULT_DIVERGENCE_LEDGER_PATH, DivergenceExpectation, default_diagnostic_expectations,
    evaluate_divergence_ledger, expectations_from_diagnostic_cases, load_ledger,
};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Json,
    Markdown,
}

#[derive(Debug)]
struct Options {
    ledger_path: PathBuf,
    case_json: Option<PathBuf>,
    report_path: Option<PathBuf>,
    fail_on_missing: bool,
    format: OutputFormat,
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            eprintln!("run_divergence_ledger failed: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<bool, String> {
    let options = parse_args(std::env::args().skip(1))?;
    let entries = load_ledger(&options.ledger_path)?;
    let expectations = load_expectations(options.case_json.as_ref())?;
    let report = evaluate_divergence_ledger(&options.ledger_path, &entries, &expectations);
    let rendered = match options.format {
        OutputFormat::Json => serde_json::to_string_pretty(&report)
            .map_err(|error| format!("serialize divergence ledger report: {error}"))?,
        OutputFormat::Markdown => report.to_markdown(),
    };
    if let Some(path) = options.report_path.as_ref() {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)
                .map_err(|error| format!("create report dir {}: {error}", parent.display()))?;
        }
        fs::write(path, &rendered)
            .map_err(|error| format!("write report {}: {error}", path.display()))?;
    }
    println!("{rendered}");
    Ok(!options.fail_on_missing || !report.has_errors())
}

fn load_expectations(case_json: Option<&PathBuf>) -> Result<Vec<DivergenceExpectation>, String> {
    let mut expectations = default_diagnostic_expectations();
    if let Some(path) = case_json {
        let cases = load_cases(path)?;
        expectations.extend(expectations_from_diagnostic_cases(&cases));
    }
    Ok(expectations)
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Options, String> {
    let mut options = Options {
        ledger_path: PathBuf::from(DEFAULT_DIVERGENCE_LEDGER_PATH),
        case_json: None,
        report_path: None,
        fail_on_missing: false,
        format: OutputFormat::Json,
    };
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--ledger-path" => {
                options.ledger_path = PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--ledger-path requires a path".to_string())?,
                );
            }
            "--case-json" => {
                options.case_json = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--case-json requires a path".to_string())?,
                ));
            }
            "--report-path" => {
                options.report_path =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        "--report-path requires a path".to_string()
                    })?));
            }
            "--fail-on-missing" => options.fail_on_missing = true,
            "--format" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "--format requires json or markdown".to_string())?;
                options.format = match raw.as_str() {
                    "json" => OutputFormat::Json,
                    "markdown" => OutputFormat::Markdown,
                    _ => return Err(format!("unknown format {raw}\n{}", usage())),
                };
            }
            "--help" | "-h" => return Err(usage()),
            unknown => return Err(format!("unknown argument {unknown}\n{}", usage())),
        }
    }
    Ok(options)
}

fn usage() -> String {
    "Usage: cargo run -p fnp-conformance --bin run_divergence_ledger -- [--ledger-path <path>] [--case-json <path>] [--report-path <path>] [--fail-on-missing] [--format json|markdown]".to_string()
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_DIVERGENCE_LEDGER_PATH, OutputFormat, parse_args};
    use std::path::PathBuf;

    #[test]
    fn divergence_ledger_cli_defaults_to_repo_ledger() {
        let options = parse_args(Vec::<String>::new()).expect("parse defaults");

        assert_eq!(
            options.ledger_path,
            PathBuf::from(DEFAULT_DIVERGENCE_LEDGER_PATH)
        );
        assert_eq!(options.format, OutputFormat::Json);
        assert!(!options.fail_on_missing);
    }

    #[test]
    fn divergence_ledger_cli_accepts_markdown_report() {
        let options = parse_args([
            "--ledger-path".to_string(),
            "docs/DIVERGENCES.md".to_string(),
            "--report-path".to_string(),
            "target/divergence_ledger.md".to_string(),
            "--format".to_string(),
            "markdown".to_string(),
            "--fail-on-missing".to_string(),
        ])
        .expect("parse options");

        assert_eq!(options.ledger_path, PathBuf::from("docs/DIVERGENCES.md"));
        assert_eq!(
            options.report_path,
            Some(PathBuf::from("target/divergence_ledger.md"))
        );
        assert_eq!(options.format, OutputFormat::Markdown);
        assert!(options.fail_on_missing);
    }
}
