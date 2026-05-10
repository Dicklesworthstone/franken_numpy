#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const SCHEMA_VERSION: u8 = 1;
const DEFAULT_CALIBRATION_PATH: &str = "target/parallel_calibration_matrix.json";
const DEFAULT_RECOMMENDATION_PATH: &str = "target/parallel_thread_recommendations.json";
const DEFAULT_MIN_SAMPLES: usize = 2;
const DEFAULT_MIN_P50_SPEEDUP: f64 = 1.05;
const DEFAULT_MIN_P95_SPEEDUP: f64 = 1.0;
const DEFAULT_MIN_P99_SPEEDUP: f64 = 0.98;
const DEFAULT_PROOF_GATE_COMMAND: &str =
    "cargo run -p fnp-conformance --bin run_parallel_speedup_verdict";

fn main() {
    if let Err(err) = run() {
        eprintln!("run_parallel_thread_recommendations failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let options = Options::parse(std::env::args().skip(1), &repo_root)?;
    let calibration = load_calibration_report(&options.calibration_path)?;
    let report = build_recommendation_report(&options, &calibration);
    write_report(&options.recommendation_path, &report)?;
    println!(
        "parallel_thread_recommendations status=ok workloads={} parallel_candidates={} serial={} insufficient_evidence={} report={}",
        report.recommendation_count,
        report.parallel_candidate_count,
        report.serial_recommended_count,
        report.insufficient_evidence_count,
        options.recommendation_path.display()
    );
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
struct Options {
    calibration_path: PathBuf,
    recommendation_path: PathBuf,
    policy: RecommendationPolicy,
}

impl Options {
    fn parse<I>(args: I, repo_root: &Path) -> Result<Self, String>
    where
        I: IntoIterator<Item = String>,
    {
        let mut calibration_path = repo_root.join(DEFAULT_CALIBRATION_PATH);
        let mut recommendation_path = repo_root.join(DEFAULT_RECOMMENDATION_PATH);
        let mut policy = RecommendationPolicy::default();

        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--calibration-path" | "--report-path" => {
                    calibration_path = PathBuf::from(require_value(&mut args, &arg)?);
                }
                "--recommendation-out" | "--output-path" => {
                    recommendation_path = PathBuf::from(require_value(&mut args, &arg)?);
                }
                "--min-samples" => {
                    let value = require_value(&mut args, "--min-samples")?;
                    policy.min_samples = parse_nonzero_usize("--min-samples", &value)?;
                }
                "--min-p50-speedup" => {
                    let value = require_value(&mut args, "--min-p50-speedup")?;
                    policy.min_p50_speedup = parse_nonnegative_f64("--min-p50-speedup", &value)?;
                }
                "--min-p95-speedup" => {
                    let value = require_value(&mut args, "--min-p95-speedup")?;
                    policy.min_p95_speedup = parse_nonnegative_f64("--min-p95-speedup", &value)?;
                }
                "--min-p99-speedup" => {
                    let value = require_value(&mut args, "--min-p99-speedup")?;
                    policy.min_p99_speedup = parse_nonnegative_f64("--min-p99-speedup", &value)?;
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                unknown => return Err(format!("unknown argument: {unknown}")),
            }
        }

        Ok(Self {
            calibration_path,
            recommendation_path,
            policy,
        })
    }
}

fn print_help() {
    println!(
        "Usage: cargo run -p fnp-conformance --bin run_parallel_thread_recommendations -- [--calibration-path <path>] [--recommendation-out <path>] [--min-samples <n>]"
    );
}

fn require_value<I>(args: &mut I, flag: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_nonzero_usize(flag: &str, value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|err| format!("{flag} must be a positive integer: {err}"))?;
    if parsed == 0 {
        return Err(format!("{flag} must be at least 1"));
    }
    Ok(parsed)
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct RecommendationPolicy {
    min_samples: usize,
    min_p50_speedup: f64,
    min_p95_speedup: f64,
    min_p99_speedup: f64,
}

impl Default for RecommendationPolicy {
    fn default() -> Self {
        Self {
            min_samples: DEFAULT_MIN_SAMPLES,
            min_p50_speedup: DEFAULT_MIN_P50_SPEEDUP,
            min_p95_speedup: DEFAULT_MIN_P95_SPEEDUP,
            min_p99_speedup: DEFAULT_MIN_P99_SPEEDUP,
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
    operation: Option<String>,
    #[serde(default)]
    shape: Vec<usize>,
    #[serde(default)]
    rhs_shape: Option<Vec<usize>>,
    #[serde(default)]
    axis: Option<isize>,
    #[serde(default)]
    keepdims: bool,
    #[serde(default)]
    input_elements_per_run: Option<usize>,
    #[serde(default)]
    output_elements_per_run: Option<usize>,
    #[serde(default)]
    bytes_touched_estimate_per_run: Option<usize>,
    #[serde(default)]
    thread_count: Option<usize>,
    #[serde(default)]
    min_elements_per_chunk: Option<usize>,
    #[serde(default)]
    expected_chunk_count: Option<usize>,
    #[serde(default)]
    serial: Option<TimingSummary>,
    #[serde(default)]
    parallel: Option<TimingSummary>,
    #[serde(default)]
    speedup: Option<SpeedupSummary>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ThreadRecommendationReport {
    schema_version: u8,
    generated_at_unix_ms: u128,
    source_calibration_path: String,
    source_schema_version: Option<u8>,
    source_git_commit: String,
    source_mode: String,
    source_sample_count: Option<usize>,
    policy: RecommendationPolicy,
    proof_gate_required: bool,
    proof_gate_command: String,
    recommendation_count: usize,
    parallel_candidate_count: usize,
    serial_recommended_count: usize,
    insufficient_evidence_count: usize,
    recommendations: Vec<WorkloadRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkloadRecommendation {
    workload_id: String,
    workload_signature: WorkloadSignature,
    recommendation: RecommendationKind,
    selected_thread_count: Option<usize>,
    selected_min_elements_per_chunk: Option<usize>,
    selected_expected_chunk_count: Option<usize>,
    selected_speedup: Option<ObservedSpeedup>,
    reasons: Vec<String>,
    evaluated_candidates: Vec<CandidateEvaluation>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
struct WorkloadSignature {
    operation: Option<String>,
    shape: Vec<usize>,
    rhs_shape: Option<Vec<usize>>,
    axis: Option<isize>,
    keepdims: bool,
    input_elements_per_run: Option<usize>,
    output_elements_per_run: Option<usize>,
    bytes_touched_estimate_per_run: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RecommendationKind {
    ParallelCandidatePendingVerdict,
    SerialRecommended,
    InsufficientEvidence,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct ObservedSpeedup {
    p50_ratio: f64,
    p95_ratio: f64,
    p99_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CandidateEvaluation {
    thread_count: Option<usize>,
    min_elements_per_chunk: Option<usize>,
    expected_chunk_count: Option<usize>,
    serial_sample_count: Option<usize>,
    parallel_sample_count: Option<usize>,
    p50_speedup_ratio: Option<f64>,
    p95_speedup_ratio: Option<f64>,
    p99_speedup_ratio: Option<f64>,
    evidence_valid: bool,
    passes_thresholds: bool,
    reasons: Vec<String>,
}

impl CandidateEvaluation {
    fn sort_key(&self) -> CandidateSortKey {
        CandidateSortKey {
            thread_count: self.thread_count.unwrap_or(usize::MAX),
            min_elements_per_chunk: self.min_elements_per_chunk.unwrap_or(usize::MAX),
            expected_chunk_count: self.expected_chunk_count.unwrap_or(usize::MAX),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CandidateSortKey {
    thread_count: usize,
    min_elements_per_chunk: usize,
    expected_chunk_count: usize,
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

fn build_recommendation_report(
    options: &Options,
    calibration: &CalibrationReport,
) -> ThreadRecommendationReport {
    let mut groups = BTreeMap::<WorkloadGroupKey, Vec<&CalibrationEntry>>::new();
    for entry in &calibration.entries {
        groups
            .entry(WorkloadGroupKey::from(entry))
            .or_default()
            .push(entry);
    }

    let recommendations = groups
        .into_values()
        .filter_map(|entries| recommend_workload(&entries, &options.policy))
        .collect::<Vec<_>>();
    let parallel_candidate_count = recommendations
        .iter()
        .filter(|rec| rec.recommendation == RecommendationKind::ParallelCandidatePendingVerdict)
        .count();
    let serial_recommended_count = recommendations
        .iter()
        .filter(|rec| rec.recommendation == RecommendationKind::SerialRecommended)
        .count();
    let insufficient_evidence_count = recommendations
        .iter()
        .filter(|rec| rec.recommendation == RecommendationKind::InsufficientEvidence)
        .count();

    ThreadRecommendationReport {
        schema_version: SCHEMA_VERSION,
        generated_at_unix_ms: now_unix_ms(),
        source_calibration_path: options.calibration_path.display().to_string(),
        source_schema_version: calibration.schema_version,
        source_git_commit: calibration.git_commit.clone(),
        source_mode: calibration.mode.clone(),
        source_sample_count: calibration.sample_count,
        policy: options.policy,
        proof_gate_required: true,
        proof_gate_command: DEFAULT_PROOF_GATE_COMMAND.to_string(),
        recommendation_count: recommendations.len(),
        parallel_candidate_count,
        serial_recommended_count,
        insufficient_evidence_count,
        recommendations,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct WorkloadGroupKey {
    workload_id: String,
    signature: WorkloadSignature,
}

impl From<&CalibrationEntry> for WorkloadGroupKey {
    fn from(entry: &CalibrationEntry) -> Self {
        Self {
            workload_id: entry.workload_id.clone(),
            signature: WorkloadSignature::from(entry),
        }
    }
}

impl From<&CalibrationEntry> for WorkloadSignature {
    fn from(entry: &CalibrationEntry) -> Self {
        Self {
            operation: entry.operation.clone(),
            shape: entry.shape.clone(),
            rhs_shape: entry.rhs_shape.clone(),
            axis: entry.axis,
            keepdims: entry.keepdims,
            input_elements_per_run: entry.input_elements_per_run,
            output_elements_per_run: entry.output_elements_per_run,
            bytes_touched_estimate_per_run: entry.bytes_touched_estimate_per_run,
        }
    }
}

fn recommend_workload(
    entries: &[&CalibrationEntry],
    policy: &RecommendationPolicy,
) -> Option<WorkloadRecommendation> {
    let first = entries.first().copied()?;
    let mut evaluations = entries
        .iter()
        .map(|entry| evaluate_candidate(entry, policy))
        .collect::<Vec<_>>();
    evaluations.sort_by_key(CandidateEvaluation::sort_key);

    let selected = evaluations
        .iter()
        .find(|evaluation| evaluation.evidence_valid && evaluation.passes_thresholds);
    let has_valid_evidence = evaluations
        .iter()
        .any(|evaluation| evaluation.evidence_valid);
    let (recommendation, reasons) = if let Some(selected) = selected {
        (
            RecommendationKind::ParallelCandidatePendingVerdict,
            vec![format!(
                "selected lowest passing thread_count={} min_elements_per_chunk={} pending speedup verdict gate",
                selected.thread_count.unwrap_or(0),
                selected.min_elements_per_chunk.unwrap_or(0)
            )],
        )
    } else if has_valid_evidence {
        (
            RecommendationKind::SerialRecommended,
            vec![
                "all valid candidates failed configured speedup thresholds; keep serial execution"
                    .to_string(),
            ],
        )
    } else {
        (
            RecommendationKind::InsufficientEvidence,
            vec![
                "no candidate had enough comparable serial/parallel samples for recommendation"
                    .to_string(),
            ],
        )
    };

    Some(WorkloadRecommendation {
        workload_id: first.workload_id.clone(),
        workload_signature: WorkloadSignature::from(first),
        recommendation,
        selected_thread_count: selected.and_then(|selected| selected.thread_count),
        selected_min_elements_per_chunk: selected
            .and_then(|selected| selected.min_elements_per_chunk),
        selected_expected_chunk_count: selected.and_then(|selected| selected.expected_chunk_count),
        selected_speedup: selected.map(|selected| ObservedSpeedup {
            p50_ratio: selected.p50_speedup_ratio.unwrap_or(0.0),
            p95_ratio: selected.p95_speedup_ratio.unwrap_or(0.0),
            p99_ratio: selected.p99_speedup_ratio.unwrap_or(0.0),
        }),
        reasons,
        evaluated_candidates: evaluations,
    })
}

fn evaluate_candidate(
    entry: &CalibrationEntry,
    policy: &RecommendationPolicy,
) -> CandidateEvaluation {
    let mut reasons = Vec::new();
    let serial_sample_count = entry.serial.as_ref().map(|serial| serial.samples_ms.len());
    let parallel_sample_count = entry
        .parallel
        .as_ref()
        .map(|parallel| parallel.samples_ms.len());
    let speedup = entry.speedup.map(|speedup| ObservedSpeedup {
        p50_ratio: speedup.p50_ratio,
        p95_ratio: speedup.p95_ratio,
        p99_ratio: speedup.p99_ratio,
    });

    if entry.thread_count.is_none() {
        reasons.push("missing thread_count".to_string());
    }
    if entry.min_elements_per_chunk.is_none() {
        reasons.push("missing min_elements_per_chunk".to_string());
    }
    if entry.serial.is_none() {
        reasons.push("missing serial timing telemetry".to_string());
    }
    if entry.parallel.is_none() {
        reasons.push("missing parallel timing telemetry".to_string());
    }
    if speedup.is_none() {
        reasons.push("missing speedup summary".to_string());
    }

    if let (Some(serial), Some(parallel)) = (&entry.serial, &entry.parallel) {
        validate_timing_summary("serial", serial, &mut reasons);
        validate_timing_summary("parallel", parallel, &mut reasons);
        if serial.samples_ms.len() != parallel.samples_ms.len() {
            reasons.push(format!(
                "serial/parallel sample count mismatch serial={} parallel={}",
                serial.samples_ms.len(),
                parallel.samples_ms.len()
            ));
        }
        if serial.samples_ms.len() < policy.min_samples
            || parallel.samples_ms.len() < policy.min_samples
        {
            reasons.push(format!(
                "insufficient samples: required={} serial={} parallel={}",
                policy.min_samples,
                serial.samples_ms.len(),
                parallel.samples_ms.len()
            ));
        }
    }

    if let Some(speedup) = speedup {
        validate_speedup(speedup, &mut reasons);
    }

    let evidence_valid = reasons.is_empty();
    let passes_thresholds = evidence_valid
        && speedup.is_some_and(|speedup| {
            speedup.p50_ratio >= policy.min_p50_speedup
                && speedup.p95_ratio >= policy.min_p95_speedup
                && speedup.p99_ratio >= policy.min_p99_speedup
        });
    if evidence_valid && !passes_thresholds {
        reasons.push(format!(
            "below thresholds: p50={:.6} p95={:.6} p99={:.6} required p50>={:.6} p95>={:.6} p99>={:.6}",
            speedup.map_or(0.0, |speedup| speedup.p50_ratio),
            speedup.map_or(0.0, |speedup| speedup.p95_ratio),
            speedup.map_or(0.0, |speedup| speedup.p99_ratio),
            policy.min_p50_speedup,
            policy.min_p95_speedup,
            policy.min_p99_speedup
        ));
    }
    if passes_thresholds {
        reasons.push("passes configured speedup thresholds".to_string());
    }

    CandidateEvaluation {
        thread_count: entry.thread_count,
        min_elements_per_chunk: entry.min_elements_per_chunk,
        expected_chunk_count: entry.expected_chunk_count,
        serial_sample_count,
        parallel_sample_count,
        p50_speedup_ratio: speedup.map(|speedup| speedup.p50_ratio),
        p95_speedup_ratio: speedup.map(|speedup| speedup.p95_ratio),
        p99_speedup_ratio: speedup.map(|speedup| speedup.p99_ratio),
        evidence_valid,
        passes_thresholds,
        reasons,
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

fn validate_speedup(speedup: ObservedSpeedup, reasons: &mut Vec<String>) {
    if !speedup.p50_ratio.is_finite()
        || !speedup.p95_ratio.is_finite()
        || !speedup.p99_ratio.is_finite()
    {
        reasons.push("speedup summary contains non-finite ratio".to_string());
    }
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn write_report(path: &Path, report: &ThreadRecommendationReport) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed creating recommendation dir {}: {err}",
                parent.display()
            )
        })?;
    }
    let payload = serde_json::to_string_pretty(report)
        .map_err(|err| format!("failed serializing recommendation report: {err}"))?;
    fs::write(path, payload).map_err(|err| format!("failed writing {}: {err}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options_with_policy(policy: RecommendationPolicy) -> Options {
        Options {
            calibration_path: PathBuf::from("target/parallel_calibration_matrix.json"),
            recommendation_path: PathBuf::from("target/parallel_thread_recommendations.json"),
            policy,
        }
    }

    fn base_report(entries: Vec<CalibrationEntry>) -> CalibrationReport {
        CalibrationReport {
            schema_version: Some(1),
            git_commit: "abc123".to_string(),
            mode: "quick".to_string(),
            sample_count: Some(2),
            entries,
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

    fn entry(thread_count: usize, min_chunk: usize, speedup: SpeedupSummary) -> CalibrationEntry {
        CalibrationEntry {
            workload_id: "broadcast_add_32x32_by_32".to_string(),
            operation: Some("broadcast_add".to_string()),
            shape: vec![32, 32],
            rhs_shape: Some(vec![32]),
            axis: None,
            keepdims: false,
            input_elements_per_run: Some(1024),
            output_elements_per_run: Some(1024),
            bytes_touched_estimate_per_run: Some(24_576),
            thread_count: Some(thread_count),
            min_elements_per_chunk: Some(min_chunk),
            expected_chunk_count: Some(thread_count),
            serial: Some(timing(10.0, 12.0, 13.0)),
            parallel: Some(timing(8.0, 10.0, 12.0)),
            speedup: Some(speedup),
        }
    }

    fn passing_speedup() -> SpeedupSummary {
        SpeedupSummary {
            p50_ratio: 1.25,
            p95_ratio: 1.2,
            p99_ratio: 1.08,
        }
    }

    #[test]
    fn parallel_recommendation_picks_lowest_passing_thread_count() {
        let report = build_recommendation_report(
            &options_with_policy(RecommendationPolicy::default()),
            &base_report(vec![
                entry(8, 128, passing_speedup()),
                entry(2, 128, passing_speedup()),
                entry(4, 128, passing_speedup()),
            ]),
        );
        let rec = &report.recommendations[0];
        assert_eq!(
            rec.recommendation,
            RecommendationKind::ParallelCandidatePendingVerdict
        );
        assert_eq!(rec.selected_thread_count, Some(2));
        assert_eq!(rec.parallel_candidate_count(), 3);
    }

    #[test]
    fn parallel_recommendation_uses_min_chunk_tie_breaker() {
        let report = build_recommendation_report(
            &options_with_policy(RecommendationPolicy::default()),
            &base_report(vec![
                entry(4, 512, passing_speedup()),
                entry(4, 128, passing_speedup()),
            ]),
        );
        let rec = &report.recommendations[0];
        assert_eq!(rec.selected_thread_count, Some(4));
        assert_eq!(rec.selected_min_elements_per_chunk, Some(128));
    }

    #[test]
    fn parallel_recommendation_insufficient_samples_rejected() {
        let policy = RecommendationPolicy {
            min_samples: 3,
            ..RecommendationPolicy::default()
        };
        let report = build_recommendation_report(
            &options_with_policy(policy),
            &base_report(vec![entry(4, 128, passing_speedup())]),
        );
        let rec = &report.recommendations[0];
        assert_eq!(rec.recommendation, RecommendationKind::InsufficientEvidence);
        assert!(
            rec.evaluated_candidates[0]
                .reasons
                .iter()
                .any(|reason| reason.contains("insufficient samples"))
        );
    }

    #[test]
    fn parallel_recommendation_serial_when_speedup_absent() {
        let report = build_recommendation_report(
            &options_with_policy(RecommendationPolicy::default()),
            &base_report(vec![entry(
                4,
                128,
                SpeedupSummary {
                    p50_ratio: 1.01,
                    p95_ratio: 1.0,
                    p99_ratio: 1.0,
                },
            )]),
        );
        let rec = &report.recommendations[0];
        assert_eq!(rec.recommendation, RecommendationKind::SerialRecommended);
        assert_eq!(rec.selected_thread_count, None);
    }

    #[test]
    fn parallel_recommendation_report_serialization_round_trips() {
        let report = build_recommendation_report(
            &options_with_policy(RecommendationPolicy::default()),
            &base_report(vec![entry(4, 128, passing_speedup())]),
        );
        let raw = serde_json::to_string_pretty(&report).expect("recommendation serializes");
        let parsed: ThreadRecommendationReport =
            serde_json::from_str(&raw).expect("recommendation report parses");
        assert_eq!(parsed.schema_version, SCHEMA_VERSION);
        assert_eq!(parsed.recommendation_count, 1);
        assert!(parsed.proof_gate_required);
    }

    #[test]
    fn parallel_recommendation_smoke_fixture_is_candidate() {
        let raw = include_str!("../../fixtures/parallel_calibration_matrix_smoke.json");
        let calibration: CalibrationReport = serde_json::from_str(raw).expect("fixture parses");
        let report = build_recommendation_report(
            &options_with_policy(RecommendationPolicy::default()),
            &calibration,
        );
        assert_eq!(report.recommendation_count, 1);
        assert_eq!(
            report.recommendations[0].recommendation,
            RecommendationKind::ParallelCandidatePendingVerdict
        );
        assert_eq!(report.recommendations[0].selected_thread_count, Some(4));
    }

    impl WorkloadRecommendation {
        fn parallel_candidate_count(&self) -> usize {
            self.evaluated_candidates
                .iter()
                .filter(|candidate| candidate.passes_thresholds)
                .count()
        }
    }
}
