#![forbid(unsafe_code)]

use fnp_iter::{FlatIterIndex, TransferError, validate_flatiter_read};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const PACKET_ID: &str = "FNP-P2C-004";
const SUBTASK_ID: &str = "FNP-P2C-004-H";

#[derive(Debug, Clone, Serialize)]
struct PercentileSummary {
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ProfileSummary {
    implementation: String,
    runs: usize,
    elements_per_run: usize,
    percentiles: PercentileSummary,
    throughput_elements_per_sec_p50: f64,
    throughput_elements_per_sec_p95: f64,
}

#[derive(Debug, Clone, Serialize)]
struct DeltaSummary {
    p50_delta_percent: f64,
    p95_delta_percent: f64,
    p99_delta_percent: f64,
    throughput_gain_percent_p50: f64,
    throughput_gain_percent_p95: f64,
}

#[derive(Debug, Clone, Serialize)]
struct LeverSummary {
    id: String,
    description: String,
    rationale: String,
    risk_note: String,
    rollback_command: String,
}

#[derive(Debug, Clone, Serialize)]
struct EVSummary {
    impact: f64,
    confidence: f64,
    reuse: f64,
    effort: f64,
    adoption_friction: f64,
    score: f64,
    promoted: bool,
}

#[derive(Debug, Clone, Serialize)]
struct IsomorphismCheck {
    case_id: String,
    status: String,
    details: String,
}

#[derive(Debug, Serialize)]
struct Packet004OptimizationReport {
    schema_version: u8,
    packet_id: String,
    subtask_id: String,
    generated_at_unix_ms: u128,
    environment_fingerprint: String,
    reproducibility_command: String,
    lever: LeverSummary,
    baseline_profile: ProfileSummary,
    rebaseline_profile: ProfileSummary,
    delta: DeltaSummary,
    ev: EVSummary,
    isomorphism_checks: Vec<IsomorphismCheck>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("generate_packet004_optimization_report failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output_path = parse_output_path(&repo_root)?;

    let mask_len = 1_048_576usize;
    let mask = make_deterministic_mask(mask_len);
    let index = FlatIterIndex::BoolMask(mask.clone());

    let baseline_profile =
        profile_implementation("baseline_filter_count_path", 80, mask_len, || {
            legacy_validate_flatiter_read(mask_len, &index)
                .map_err(|err| format!("legacy run failed: {err}"))
        })?;
    let rebaseline_profile =
        profile_implementation("rebaseline_branchless_unrolled_path", 80, mask_len, || {
            validate_flatiter_read(mask_len, &index)
                .map_err(|err| format!("optimized run failed: {err}"))
        })?;

    let delta = DeltaSummary {
        p50_delta_percent: percent_delta(
            baseline_profile.percentiles.p50_ms,
            rebaseline_profile.percentiles.p50_ms,
        ),
        p95_delta_percent: percent_delta(
            baseline_profile.percentiles.p95_ms,
            rebaseline_profile.percentiles.p95_ms,
        ),
        p99_delta_percent: percent_delta(
            baseline_profile.percentiles.p99_ms,
            rebaseline_profile.percentiles.p99_ms,
        ),
        throughput_gain_percent_p50: percent_delta(
            baseline_profile.throughput_elements_per_sec_p50,
            rebaseline_profile.throughput_elements_per_sec_p50,
        ),
        throughput_gain_percent_p95: percent_delta(
            baseline_profile.throughput_elements_per_sec_p95,
            rebaseline_profile.throughput_elements_per_sec_p95,
        ),
    };

    let isomorphism_checks = run_isomorphism_checks();
    let all_checks_pass = isomorphism_checks
        .iter()
        .all(|check| check.status == "pass");

    let impact = if delta.throughput_gain_percent_p95 > 5.0 {
        4.0
    } else {
        2.0
    };
    let confidence = if all_checks_pass { 4.0 } else { 1.0 };
    let reuse = 3.0;
    let effort = 2.0;
    let adoption_friction = 1.0;
    let score = (impact * confidence * reuse) / (effort * adoption_friction);
    let ev = EVSummary {
        impact,
        confidence,
        reuse,
        effort,
        adoption_friction,
        score,
        promoted: score >= 2.0 && all_checks_pass,
    };

    let report = Packet004OptimizationReport {
        schema_version: 1,
        packet_id: PACKET_ID.to_string(),
        subtask_id: SUBTASK_ID.to_string(),
        generated_at_unix_ms: now_unix_ms(),
        environment_fingerprint: environment_fingerprint(),
        reproducibility_command:
            "cargo run -p fnp-conformance --bin generate_packet004_optimization_report"
                .to_string(),
        lever: LeverSummary {
            id: "P2C004-H-LEVER-001".to_string(),
            description:
                "Switch flatiter bool-mask lane counting to a branchless, unrolled accumulator"
                    .to_string(),
            rationale:
                "Bool-mask validation is a packet hotspot; replacing iterator/filter counting with a branchless unrolled path lowers per-lane overhead while preserving contract behavior."
                    .to_string(),
            risk_note:
                "Risk is low because only counting mechanics changed; bounds checks and all error-class branches remain unchanged."
                    .to_string(),
            rollback_command: "git restore --source <last-green-commit> -- crates/fnp-iter/src/lib.rs"
                .to_string(),
        },
        baseline_profile,
        rebaseline_profile,
        delta,
        ev,
        isomorphism_checks,
    };

    let raw = serde_json::to_string_pretty(&report)
        .map_err(|err| format!("failed serializing optimization report: {err}"))?;
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }
    fs::write(&output_path, raw)
        .map_err(|err| format!("failed writing {}: {err}", output_path.display()))?;

    println!("wrote {}", output_path.display());
    Ok(())
}

fn parse_output_path(repo_root: &Path) -> Result<PathBuf, String> {
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
                    "Usage: cargo run -p fnp-conformance --bin generate_packet004_optimization_report -- [--output-path <path>]"
                );
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }

    Ok(output_path.unwrap_or_else(|| {
        repo_root.join("artifacts/phase2c/FNP-P2C-004/optimization_profile_report.json")
    }))
}

fn profile_implementation<F>(
    implementation: &str,
    runs: usize,
    elements_per_run: usize,
    mut run_fn: F,
) -> Result<ProfileSummary, String>
where
    F: FnMut() -> Result<usize, String>,
{
    let mut samples_ms = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        let selected = run_fn()?;
        std::hint::black_box(selected);
        samples_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let percentiles = summarize_samples(&samples_ms);
    let throughput_elements_per_sec_p50 =
        compute_throughput(elements_per_run as f64, percentiles.p50_ms);
    let throughput_elements_per_sec_p95 =
        compute_throughput(elements_per_run as f64, percentiles.p95_ms);

    Ok(ProfileSummary {
        implementation: implementation.to_string(),
        runs,
        elements_per_run,
        percentiles,
        throughput_elements_per_sec_p50,
        throughput_elements_per_sec_p95,
    })
}

fn run_isomorphism_checks() -> Vec<IsomorphismCheck> {
    vec![
        compare_case(
            "case_boolmask_all_true",
            16,
            FlatIterIndex::BoolMask(vec![true; 16]),
        ),
        compare_case(
            "case_boolmask_sparse",
            15,
            FlatIterIndex::BoolMask((0..15).map(|idx| idx % 4 == 1).collect()),
        ),
        compare_case(
            "case_boolmask_length_mismatch",
            12,
            FlatIterIndex::BoolMask(vec![true, false, true]),
        ),
        compare_case(
            "case_fancy_out_of_bounds",
            8,
            FlatIterIndex::Fancy(vec![0, 2, 9]),
        ),
    ]
}

fn compare_case(case_id: &str, len: usize, index: FlatIterIndex) -> IsomorphismCheck {
    let baseline = legacy_validate_flatiter_read(len, &index);
    let optimized = validate_flatiter_read(len, &index);
    if baseline == optimized {
        IsomorphismCheck {
            case_id: case_id.to_string(),
            status: "pass".to_string(),
            details: "baseline and optimized outputs matched".to_string(),
        }
    } else {
        IsomorphismCheck {
            case_id: case_id.to_string(),
            status: "fail".to_string(),
            details: format!("baseline={baseline:?} optimized={optimized:?}"),
        }
    }
}

fn legacy_validate_flatiter_read(
    len: usize,
    index: &FlatIterIndex,
) -> Result<usize, TransferError> {
    legacy_count_selected_indices(len, index)
}

fn legacy_count_selected_indices(
    len: usize,
    index: &FlatIterIndex,
) -> Result<usize, TransferError> {
    match index {
        FlatIterIndex::Single(i) => {
            if *i >= len {
                Err(TransferError::FlatiterReadViolation(
                    "single index out of bounds",
                ))
            } else {
                Ok(1)
            }
        }
        FlatIterIndex::Slice { start, stop, step } => {
            if *step == 0 {
                return Err(TransferError::FlatiterReadViolation(
                    "slice step must be > 0",
                ));
            }
            if *start > *stop || *stop > len {
                return Err(TransferError::FlatiterReadViolation(
                    "slice bounds are invalid for flatiter",
                ));
            }
            let span = stop - start;
            Ok(span.div_ceil(*step))
        }
        FlatIterIndex::Fancy(indices) => {
            if indices.iter().any(|idx| *idx >= len) {
                Err(TransferError::FlatiterReadViolation(
                    "fancy index out of bounds",
                ))
            } else {
                Ok(indices.len())
            }
        }
        FlatIterIndex::BoolMask(mask) => {
            if mask.len() != len {
                return Err(TransferError::FlatiterReadViolation(
                    "bool mask length must equal flatiter length",
                ));
            }
            Ok(mask.iter().filter(|flag| **flag).count())
        }
    }
}

#[must_use]
fn make_deterministic_mask(len: usize) -> Vec<bool> {
    let mut mask = Vec::with_capacity(len);
    let mut state: u64 = 0x4d595df4d0f33173;
    for _ in 0..len {
        state ^= state << 7;
        state ^= state >> 9;
        state ^= state << 8;
        mask.push((state & 0b111) < 0b011);
    }
    mask
}

fn summarize_samples(samples_ms: &[f64]) -> PercentileSummary {
    let mut sorted = samples_ms.to_vec();
    sorted.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    PercentileSummary {
        p50_ms: percentile(&sorted, 0.50),
        p95_ms: percentile(&sorted, 0.95),
        p99_ms: percentile(&sorted, 0.99),
        min_ms: *sorted.first().unwrap_or(&0.0),
        max_ms: *sorted.last().unwrap_or(&0.0),
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let max_index = sorted.len() - 1;
    let rank = (p.clamp(0.0, 1.0) * max_index as f64).round() as usize;
    sorted[rank.min(max_index)]
}

#[must_use]
fn compute_throughput(elements: f64, latency_ms: f64) -> f64 {
    if latency_ms <= 0.0 {
        return 0.0;
    }
    elements / (latency_ms / 1000.0)
}

#[must_use]
fn percent_delta(baseline: f64, updated: f64) -> f64 {
    if baseline == 0.0 {
        return 0.0;
    }
    ((updated - baseline) / baseline) * 100.0
}

#[must_use]
fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

#[must_use]
fn environment_fingerprint() -> String {
    let rustc = Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "rustc-unknown".to_string());
    format!(
        "os={} arch={} cpus={} rustc={}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::thread::available_parallelism().map_or(0, usize::from),
        rustc
    )
}
