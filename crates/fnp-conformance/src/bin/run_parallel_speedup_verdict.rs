#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const SCHEMA_VERSION: u8 = 1;
const DEFAULT_REPORT_PATH: &str = "target/parallel_calibration_matrix.json";
const DEFAULT_VERDICT_PATH: &str = "target/parallel_speedup_verdict.json";
const DEFAULT_MIN_P50_SPEEDUP: f64 = 1.05;
const DEFAULT_MIN_P95_SPEEDUP: f64 = 1.0;
const DEFAULT_MIN_P95_NO_REGRESSION_RATIO: f64 = 1.0;
const DEFAULT_MIN_P99_NO_REGRESSION_RATIO: f64 = 0.98;
const DEFAULT_MAX_RSS_TO_PEAK_LIVE_RATIO: u64 = 64;
const DEFAULT_RSS_BASELINE_ALLOWANCE_BYTES: u64 = 128 * 1024 * 1024;

fn main() {
    match run() {
        Ok(exit_code) => std::process::exit(exit_code),
        Err(err) => {
            eprintln!("run_parallel_speedup_verdict failed: {err}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<i32, String> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let options = Options::parse(std::env::args().skip(1), &repo_root)?;
    let calibration = load_calibration_report(&options.report_path)?;
    let report = build_verdict_report(&options, &calibration);
    write_report(&options.verdict_path, &report)?;
    println!(
        "parallel_speedup_verdict status=ok overall_status={} entries={} candidate={} disabled={} unsafe={} verdict={}",
        report.overall_status.as_str(),
        report.entries.len(),
        report.candidate_count,
        report.disabled_count,
        report.unsafe_count,
        options.verdict_path.display()
    );
    if options.enforce && report.unsafe_count > 0 {
        return Ok(2);
    }
    Ok(0)
}

#[derive(Debug, Clone, PartialEq)]
struct Options {
    report_path: PathBuf,
    verdict_path: PathBuf,
    policy: VerdictPolicy,
    enforce: bool,
}

impl Options {
    fn parse<I>(args: I, repo_root: &Path) -> Result<Self, String>
    where
        I: IntoIterator<Item = String>,
    {
        let mut report_path = repo_root.join(DEFAULT_REPORT_PATH);
        let mut verdict_path = repo_root.join(DEFAULT_VERDICT_PATH);
        let mut policy = VerdictPolicy::default();
        let mut enforce = false;

        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--report-path" | "--calibration-path" => {
                    report_path = PathBuf::from(require_value(&mut args, &arg)?);
                }
                "--verdict-out" | "--output-path" => {
                    verdict_path = PathBuf::from(require_value(&mut args, &arg)?);
                }
                "--min-p50-speedup" => {
                    let value = require_value(&mut args, "--min-p50-speedup")?;
                    policy.min_p50_speedup = parse_nonnegative_f64("--min-p50-speedup", &value)?;
                }
                "--min-p95-speedup" => {
                    let value = require_value(&mut args, "--min-p95-speedup")?;
                    policy.min_p95_speedup = parse_nonnegative_f64("--min-p95-speedup", &value)?;
                }
                "--min-p95-no-regression-ratio" => {
                    let value = require_value(&mut args, "--min-p95-no-regression-ratio")?;
                    policy.min_p95_no_regression_ratio =
                        parse_nonnegative_f64("--min-p95-no-regression-ratio", &value)?;
                }
                "--min-p99-no-regression-ratio" => {
                    let value = require_value(&mut args, "--min-p99-no-regression-ratio")?;
                    policy.min_p99_no_regression_ratio =
                        parse_nonnegative_f64("--min-p99-no-regression-ratio", &value)?;
                }
                "--max-rss-to-peak-live-ratio" => {
                    let value = require_value(&mut args, "--max-rss-to-peak-live-ratio")?;
                    policy.max_rss_to_peak_live_ratio =
                        parse_nonzero_u64("--max-rss-to-peak-live-ratio", &value)?;
                }
                "--rss-baseline-allowance-bytes" => {
                    let value = require_value(&mut args, "--rss-baseline-allowance-bytes")?;
                    policy.rss_baseline_allowance_bytes =
                        parse_u64("--rss-baseline-allowance-bytes", &value)?;
                }
                "--enforce" => enforce = true,
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                unknown => return Err(format!("unknown argument: {unknown}")),
            }
        }

        Ok(Self {
            report_path,
            verdict_path,
            policy,
            enforce,
        })
    }
}

fn print_help() {
    println!(
        "Usage: cargo run -p fnp-conformance --bin run_parallel_speedup_verdict -- [--report-path <path>] [--verdict-out <path>] [--enforce]"
    );
}

fn require_value<I>(args: &mut I, flag: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_nonnegative_f64(flag: &str, value: &str) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|err| format!("{flag} must be a finite non-negative number: {err}"))?;
    if !parsed.is_finite() || parsed < 0.0 {
        return Err(format!("{flag} must be a finite non-negative number"));
    }
    Ok(parsed)
}

fn parse_u64(flag: &str, value: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|err| format!("{flag} must be an unsigned integer: {err}"))
}

fn parse_nonzero_u64(flag: &str, value: &str) -> Result<u64, String> {
    let parsed = parse_u64(flag, value)?;
    if parsed == 0 {
        return Err(format!("{flag} must be at least 1"));
    }
    Ok(parsed)
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
struct VerdictPolicy {
    min_p50_speedup: f64,
    min_p95_speedup: f64,
    min_p95_no_regression_ratio: f64,
    min_p99_no_regression_ratio: f64,
    max_rss_to_peak_live_ratio: u64,
    rss_baseline_allowance_bytes: u64,
}

impl Default for VerdictPolicy {
    fn default() -> Self {
        Self {
            min_p50_speedup: DEFAULT_MIN_P50_SPEEDUP,
            min_p95_speedup: DEFAULT_MIN_P95_SPEEDUP,
            min_p95_no_regression_ratio: DEFAULT_MIN_P95_NO_REGRESSION_RATIO,
            min_p99_no_regression_ratio: DEFAULT_MIN_P99_NO_REGRESSION_RATIO,
            max_rss_to_peak_live_ratio: DEFAULT_MAX_RSS_TO_PEAK_LIVE_RATIO,
            rss_baseline_allowance_bytes: DEFAULT_RSS_BASELINE_ALLOWANCE_BYTES,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct CalibrationReport {
    #[serde(default)]
    schema_version: Option<u8>,
    #[serde(default)]
    git_commit: String,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    sample_count: Option<usize>,
    #[serde(default)]
    entries: Vec<CalibrationEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct CalibrationEntry {
    #[serde(default)]
    workload_id: String,
    #[serde(default)]
    thread_count: Option<usize>,
    #[serde(default)]
    serial: Option<TimingSummary>,
    #[serde(default)]
    parallel: Option<TimingSummary>,
    #[serde(default)]
    speedup: Option<SpeedupSummary>,
    #[serde(default)]
    telemetry: Option<CalibrationTelemetry>,
}

#[derive(Debug, Clone, Deserialize)]
struct TimingSummary {
    #[serde(default)]
    samples_ms: Vec<f64>,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct SpeedupSummary {
    p50_ratio: f64,
    p95_ratio: f64,
    p99_ratio: f64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct CalibrationTelemetry {
    #[serde(default)]
    input_elements_per_run: Option<usize>,
    #[serde(default)]
    output_elements_per_run: Option<usize>,
    #[serde(default)]
    bytes_touched_estimate_per_run: Option<usize>,
    #[serde(default)]
    peak_live_bytes_per_run: Option<usize>,
    #[serde(default)]
    process_high_water_rss_bytes: Option<u64>,
    #[serde(default)]
    configured_thread_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct SpeedupVerdictReport {
    schema_version: u8,
    generated_at_unix_ms: u128,
    source_report_path: String,
    source_schema_version: Option<u8>,
    source_git_commit: String,
    source_mode: String,
    source_sample_count: Option<usize>,
    policy: VerdictPolicy,
    overall_status: VerdictStatus,
    candidate_count: usize,
    disabled_count: usize,
    unsafe_count: usize,
    entries: Vec<WorkloadVerdict>,
}

#[derive(Debug, Clone, Serialize)]
struct WorkloadVerdict {
    workload_id: String,
    thread_count: Option<usize>,
    configured_thread_count: Option<usize>,
    status: VerdictStatus,
    evidence_valid: bool,
    opt_in_allowed: bool,
    reasons: Vec<String>,
    observed: ObservedVerdictMetrics,
}

#[derive(Debug, Clone, Default, Serialize)]
struct ObservedVerdictMetrics {
    p50_speedup_ratio: Option<f64>,
    p95_speedup_ratio: Option<f64>,
    p99_speedup_ratio: Option<f64>,
    serial_sample_count: Option<usize>,
    parallel_sample_count: Option<usize>,
    input_elements_per_run: Option<usize>,
    output_elements_per_run: Option<usize>,
    bytes_touched_estimate_per_run: Option<usize>,
    peak_live_bytes_per_run: Option<usize>,
    process_high_water_rss_bytes: Option<u64>,
    rss_allowance_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum VerdictStatus {
    CandidateForOptIn,
    DisabledByDefault,
    UnsafeOrUnproven,
}

impl VerdictStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::CandidateForOptIn => "candidate_for_opt_in",
            Self::DisabledByDefault => "disabled_by_default",
            Self::UnsafeOrUnproven => "unsafe_or_unproven",
        }
    }
}

fn load_calibration_report(path: &Path) -> Result<CalibrationReport, String> {
    let raw = fs::read_to_string(path).map_err(|err| {
        format!(
            "failed reading calibration report {}: {err}",
            path.display()
        )
    })?;
    serde_json::from_str::<CalibrationReport>(&raw)
        .map_err(|err| format!("invalid calibration report {}: {err}", path.display()))
}

fn build_verdict_report(
    options: &Options,
    calibration: &CalibrationReport,
) -> SpeedupVerdictReport {
    let entries = calibration
        .entries
        .iter()
        .map(|entry| evaluate_entry(entry, calibration.sample_count, &options.policy))
        .collect::<Vec<_>>();
    let candidate_count = entries
        .iter()
        .filter(|entry| entry.status == VerdictStatus::CandidateForOptIn)
        .count();
    let disabled_count = entries
        .iter()
        .filter(|entry| entry.status == VerdictStatus::DisabledByDefault)
        .count();
    let unsafe_count = entries
        .iter()
        .filter(|entry| entry.status == VerdictStatus::UnsafeOrUnproven)
        .count();
    let overall_status = if unsafe_count > 0 || entries.is_empty() {
        VerdictStatus::UnsafeOrUnproven
    } else if disabled_count > 0 {
        VerdictStatus::DisabledByDefault
    } else {
        VerdictStatus::CandidateForOptIn
    };

    SpeedupVerdictReport {
        schema_version: SCHEMA_VERSION,
        generated_at_unix_ms: now_unix_ms(),
        source_report_path: options.report_path.display().to_string(),
        source_schema_version: calibration.schema_version,
        source_git_commit: calibration.git_commit.clone(),
        source_mode: calibration.mode.clone(),
        source_sample_count: calibration.sample_count,
        policy: options.policy,
        overall_status,
        candidate_count,
        disabled_count,
        unsafe_count,
        entries,
    }
}

fn evaluate_entry(
    entry: &CalibrationEntry,
    expected_sample_count: Option<usize>,
    policy: &VerdictPolicy,
) -> WorkloadVerdict {
    let mut unsafe_reasons = Vec::new();
    let mut disabled_reasons = Vec::new();
    let serial_samples = entry.serial.as_ref().map(|serial| serial.samples_ms.len());
    let parallel_samples = entry
        .parallel
        .as_ref()
        .map(|parallel| parallel.samples_ms.len());
    let telemetry = entry.telemetry;
    let speedup = entry.speedup;
    let rss_allowance_bytes =
        telemetry.and_then(|telemetry| rss_allowance_bytes(telemetry, policy));

    if entry.workload_id.is_empty() {
        unsafe_reasons.push("missing workload_id".to_string());
    }
    if entry.thread_count.is_none() {
        unsafe_reasons.push("missing explicit thread_count".to_string());
    }
    if entry.serial.is_none() {
        unsafe_reasons.push("missing serial timing telemetry".to_string());
    }
    if entry.parallel.is_none() {
        unsafe_reasons.push("missing parallel timing telemetry".to_string());
    }
    if speedup.is_none() {
        unsafe_reasons.push("missing speedup summary".to_string());
    }
    if telemetry.is_none() {
        unsafe_reasons.push("missing workload telemetry".to_string());
    }

    if let (Some(serial), Some(parallel)) = (&entry.serial, &entry.parallel) {
        validate_timing_summary("serial", serial, &mut unsafe_reasons);
        validate_timing_summary("parallel", parallel, &mut unsafe_reasons);
        if serial.samples_ms.is_empty() || parallel.samples_ms.is_empty() {
            unsafe_reasons.push("serial and parallel sample lists must be non-empty".to_string());
        } else if serial.samples_ms.len() != parallel.samples_ms.len() {
            unsafe_reasons.push(format!(
                "serial/parallel sample count mismatch serial={} parallel={}",
                serial.samples_ms.len(),
                parallel.samples_ms.len()
            ));
        }
        if let Some(expected) = expected_sample_count
            && (serial.samples_ms.len() != expected || parallel.samples_ms.len() != expected)
        {
            unsafe_reasons.push(format!(
                "sample_count mismatch expected={} serial={} parallel={}",
                expected,
                serial.samples_ms.len(),
                parallel.samples_ms.len()
            ));
        }
    }

    if let Some(speedup) = speedup {
        validate_speedup_summary(speedup, &mut unsafe_reasons);
        if speedup.p95_ratio < policy.min_p95_no_regression_ratio {
            unsafe_reasons.push(format!(
                "p95 regressed: ratio {:.6} below no-regression floor {:.6}",
                speedup.p95_ratio, policy.min_p95_no_regression_ratio
            ));
        }
        if speedup.p99_ratio < policy.min_p99_no_regression_ratio {
            unsafe_reasons.push(format!(
                "p99 regressed beyond tolerance: ratio {:.6} below floor {:.6}",
                speedup.p99_ratio, policy.min_p99_no_regression_ratio
            ));
        }
        if speedup.p50_ratio < policy.min_p50_speedup {
            disabled_reasons.push(format!(
                "p50 speedup {:.6} below opt-in threshold {:.6}",
                speedup.p50_ratio, policy.min_p50_speedup
            ));
        }
        if speedup.p95_ratio < policy.min_p95_speedup {
            disabled_reasons.push(format!(
                "p95 speedup {:.6} below opt-in threshold {:.6}",
                speedup.p95_ratio, policy.min_p95_speedup
            ));
        }
    }

    if let Some(telemetry) = telemetry {
        validate_telemetry(telemetry, policy, rss_allowance_bytes, &mut unsafe_reasons);
        if entry.thread_count != telemetry.configured_thread_count {
            unsafe_reasons.push(format!(
                "thread_count/configured_thread_count mismatch entry={:?} telemetry={:?}",
                entry.thread_count, telemetry.configured_thread_count
            ));
        }
    }

    let observed = ObservedVerdictMetrics {
        p50_speedup_ratio: speedup.map(|speedup| speedup.p50_ratio),
        p95_speedup_ratio: speedup.map(|speedup| speedup.p95_ratio),
        p99_speedup_ratio: speedup.map(|speedup| speedup.p99_ratio),
        serial_sample_count: serial_samples,
        parallel_sample_count: parallel_samples,
        input_elements_per_run: telemetry.and_then(|telemetry| telemetry.input_elements_per_run),
        output_elements_per_run: telemetry.and_then(|telemetry| telemetry.output_elements_per_run),
        bytes_touched_estimate_per_run: telemetry
            .and_then(|telemetry| telemetry.bytes_touched_estimate_per_run),
        peak_live_bytes_per_run: telemetry.and_then(|telemetry| telemetry.peak_live_bytes_per_run),
        process_high_water_rss_bytes: telemetry
            .and_then(|telemetry| telemetry.process_high_water_rss_bytes),
        rss_allowance_bytes,
    };

    let (status, reasons) = if !unsafe_reasons.is_empty() {
        (VerdictStatus::UnsafeOrUnproven, unsafe_reasons)
    } else if !disabled_reasons.is_empty() {
        (VerdictStatus::DisabledByDefault, disabled_reasons)
    } else {
        (
            VerdictStatus::CandidateForOptIn,
            vec![
                "parallel evidence satisfies conservative speedup, tail, and RSS policy"
                    .to_string(),
            ],
        )
    };

    WorkloadVerdict {
        workload_id: entry.workload_id.clone(),
        thread_count: entry.thread_count,
        configured_thread_count: telemetry.and_then(|telemetry| telemetry.configured_thread_count),
        status,
        evidence_valid: status != VerdictStatus::UnsafeOrUnproven,
        opt_in_allowed: status == VerdictStatus::CandidateForOptIn,
        reasons,
        observed,
    }
}

fn validate_timing_summary(label: &str, summary: &TimingSummary, reasons: &mut Vec<String>) {
    if !summary.p50_ms.is_finite() || !summary.p95_ms.is_finite() || !summary.p99_ms.is_finite() {
        reasons.push(format!(
            "{label} timing summary contains non-finite percentile"
        ));
    }
    if summary
        .samples_ms
        .iter()
        .any(|sample| !sample.is_finite() || *sample < 0.0)
    {
        reasons.push(format!("{label} timing samples contain invalid values"));
    }
}

fn validate_speedup_summary(speedup: SpeedupSummary, reasons: &mut Vec<String>) {
    if !speedup.p50_ratio.is_finite()
        || !speedup.p95_ratio.is_finite()
        || !speedup.p99_ratio.is_finite()
    {
        reasons.push("speedup summary contains non-finite ratio".to_string());
    }
}

fn validate_telemetry(
    telemetry: CalibrationTelemetry,
    policy: &VerdictPolicy,
    rss_allowance_bytes: Option<u64>,
    reasons: &mut Vec<String>,
) {
    if telemetry.configured_thread_count.is_none() {
        reasons.push("missing configured_thread_count telemetry".to_string());
    }
    if telemetry.input_elements_per_run.unwrap_or(0) == 0 {
        reasons.push("input_elements_per_run telemetry must be positive".to_string());
    }
    if telemetry.output_elements_per_run.unwrap_or(0) == 0 {
        reasons.push("output_elements_per_run telemetry must be positive".to_string());
    }
    if telemetry.bytes_touched_estimate_per_run.unwrap_or(0) == 0 {
        reasons.push("bytes_touched_estimate_per_run telemetry must be positive".to_string());
    }
    if telemetry.peak_live_bytes_per_run.unwrap_or(0) == 0 {
        reasons.push("peak_live_bytes_per_run telemetry must be positive".to_string());
    }
    let Some(rss_bytes) = telemetry.process_high_water_rss_bytes else {
        reasons.push("missing process_high_water_rss_bytes telemetry".to_string());
        return;
    };
    let Some(allowance) = rss_allowance_bytes else {
        reasons.push("could not compute RSS allowance".to_string());
        return;
    };
    if rss_bytes > allowance {
        reasons.push(format!(
            "RSS high-water {} bytes exceeds allowance {} bytes (baseline {} + peak_live * {})",
            rss_bytes,
            allowance,
            policy.rss_baseline_allowance_bytes,
            policy.max_rss_to_peak_live_ratio
        ));
    }
}

fn rss_allowance_bytes(telemetry: CalibrationTelemetry, policy: &VerdictPolicy) -> Option<u64> {
    let peak_live = telemetry.peak_live_bytes_per_run?;
    let scaled_peak = u64::try_from(peak_live)
        .ok()?
        .checked_mul(policy.max_rss_to_peak_live_ratio)?;
    policy.rss_baseline_allowance_bytes.checked_add(scaled_peak)
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn write_report(path: &Path, report: &SpeedupVerdictReport) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating verdict dir {}: {err}", parent.display()))?;
    }
    let payload = serde_json::to_string_pretty(report)
        .map_err(|err| format!("failed serializing verdict report: {err}"))?;
    fs::write(path, payload).map_err(|err| format!("failed writing {}: {err}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options_with_policy(policy: VerdictPolicy) -> Options {
        Options {
            report_path: PathBuf::from("target/parallel_calibration_matrix.json"),
            verdict_path: PathBuf::from("target/parallel_speedup_verdict.json"),
            policy,
            enforce: false,
        }
    }

    fn base_report(entry: CalibrationEntry) -> CalibrationReport {
        CalibrationReport {
            schema_version: Some(1),
            git_commit: "abc123".to_string(),
            mode: "quick".to_string(),
            sample_count: Some(2),
            entries: vec![entry],
        }
    }

    fn timing(p50: f64, p95: f64, p99: f64) -> TimingSummary {
        TimingSummary {
            samples_ms: vec![p50, p95],
            p50_ms: p50,
            p95_ms: p95,
            p99_ms: p99,
        }
    }

    fn telemetry(rss_bytes: u64) -> CalibrationTelemetry {
        CalibrationTelemetry {
            input_elements_per_run: Some(1024),
            output_elements_per_run: Some(1024),
            bytes_touched_estimate_per_run: Some(24_576),
            peak_live_bytes_per_run: Some(24_576),
            process_high_water_rss_bytes: Some(rss_bytes),
            configured_thread_count: Some(4),
        }
    }

    fn passing_entry() -> CalibrationEntry {
        CalibrationEntry {
            workload_id: "broadcast_add_32x32_by_32".to_string(),
            thread_count: Some(4),
            serial: Some(timing(10.0, 12.0, 13.0)),
            parallel: Some(timing(8.0, 10.0, 12.0)),
            speedup: Some(SpeedupSummary {
                p50_ratio: 1.25,
                p95_ratio: 1.2,
                p99_ratio: 1.08,
            }),
            telemetry: Some(telemetry(1_000_000)),
        }
    }

    #[test]
    fn parallel_speedup_verdict_clean_speedup_passes() {
        let options = options_with_policy(VerdictPolicy::default());
        let report = build_verdict_report(&options, &base_report(passing_entry()));
        assert_eq!(report.overall_status, VerdictStatus::CandidateForOptIn);
        assert_eq!(report.candidate_count, 1);
        assert!(report.entries[0].opt_in_allowed);
    }

    #[test]
    fn parallel_speedup_verdict_missing_serial_fails_closed() {
        let options = options_with_policy(VerdictPolicy::default());
        let mut entry = passing_entry();
        entry.serial = None;
        let report = build_verdict_report(&options, &base_report(entry));
        assert_eq!(report.overall_status, VerdictStatus::UnsafeOrUnproven);
        assert!(
            report.entries[0]
                .reasons
                .iter()
                .any(|reason| reason.contains("missing serial"))
        );
    }

    #[test]
    fn parallel_speedup_verdict_missing_parallel_fails_closed() {
        let options = options_with_policy(VerdictPolicy::default());
        let mut entry = passing_entry();
        entry.parallel = None;
        let report = build_verdict_report(&options, &base_report(entry));
        assert_eq!(report.overall_status, VerdictStatus::UnsafeOrUnproven);
        assert!(
            report.entries[0]
                .reasons
                .iter()
                .any(|reason| reason.contains("missing parallel"))
        );
    }

    #[test]
    fn parallel_speedup_verdict_tail_regression_fails_closed() {
        let options = options_with_policy(VerdictPolicy::default());
        let mut entry = passing_entry();
        entry.speedup = Some(SpeedupSummary {
            p50_ratio: 1.3,
            p95_ratio: 0.99,
            p99_ratio: 0.97,
        });
        let report = build_verdict_report(&options, &base_report(entry));
        assert_eq!(report.overall_status, VerdictStatus::UnsafeOrUnproven);
        assert!(
            report.entries[0]
                .reasons
                .iter()
                .any(|reason| reason.contains("p95 regressed"))
        );
    }

    #[test]
    fn parallel_speedup_verdict_rss_growth_fails_closed() {
        let policy = VerdictPolicy {
            rss_baseline_allowance_bytes: 0,
            max_rss_to_peak_live_ratio: 1,
            ..VerdictPolicy::default()
        };
        let options = options_with_policy(policy);
        let mut entry = passing_entry();
        entry.telemetry = Some(telemetry(1_000_000));
        let report = build_verdict_report(&options, &base_report(entry));
        assert_eq!(report.overall_status, VerdictStatus::UnsafeOrUnproven);
        assert!(
            report.entries[0]
                .reasons
                .iter()
                .any(|reason| reason.contains("RSS high-water"))
        );
    }

    #[test]
    fn parallel_speedup_verdict_insufficient_speedup_disables_without_failure() {
        let options = options_with_policy(VerdictPolicy::default());
        let mut entry = passing_entry();
        entry.speedup = Some(SpeedupSummary {
            p50_ratio: 1.01,
            p95_ratio: 1.0,
            p99_ratio: 1.0,
        });
        let report = build_verdict_report(&options, &base_report(entry));
        assert_eq!(report.overall_status, VerdictStatus::DisabledByDefault);
        assert_eq!(report.unsafe_count, 0);
        assert!(report.entries[0].evidence_valid);
        assert!(!report.entries[0].opt_in_allowed);
    }

    #[test]
    fn parallel_speedup_verdict_missing_thread_count_fails_closed() {
        let options = options_with_policy(VerdictPolicy::default());
        let mut entry = passing_entry();
        entry.thread_count = None;
        let report = build_verdict_report(&options, &base_report(entry));
        assert_eq!(report.overall_status, VerdictStatus::UnsafeOrUnproven);
        assert!(
            report.entries[0]
                .reasons
                .iter()
                .any(|reason| reason.contains("thread_count"))
        );
    }

    #[test]
    fn parallel_speedup_verdict_smoke_fixture_is_candidate() {
        let raw = include_str!("../../fixtures/parallel_calibration_matrix_smoke.json");
        let calibration: CalibrationReport = serde_json::from_str(raw).expect("fixture parses");
        let report =
            build_verdict_report(&options_with_policy(VerdictPolicy::default()), &calibration);
        assert_eq!(report.overall_status, VerdictStatus::CandidateForOptIn);
        assert_eq!(report.candidate_count, 1);
    }
}
