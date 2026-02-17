#![forbid(unsafe_code)]

use fnp_ndarray::{ShapeError, broadcast_shape};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const PACKET_ID: &str = "FNP-P2C-006";
const SUBTASK_ID: &str = "FNP-P2C-006-H";

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
struct Packet006OptimizationReport {
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
        eprintln!("generate_packet006_optimization_report failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output_path = parse_output_path(&repo_root)?;

    let workload = build_workload(4096);
    let ops_per_run = workload.len();

    let baseline_profile =
        profile_implementation("baseline_reverse_index_path", 90, ops_per_run, || {
            run_workload(&workload, legacy_broadcast_shape)
        })?;

    let rebaseline_profile =
        profile_implementation("rebaseline_same_rank_fast_path", 90, ops_per_run, || {
            run_workload(&workload, broadcast_shape)
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

    let report = Packet006OptimizationReport {
        schema_version: 1,
        packet_id: PACKET_ID.to_string(),
        subtask_id: SUBTASK_ID.to_string(),
        generated_at_unix_ms: now_unix_ms(),
        environment_fingerprint: environment_fingerprint(),
        reproducibility_command:
            "cargo run -p fnp-conformance --bin generate_packet006_optimization_report"
                .to_string(),
        lever: LeverSummary {
            id: "P2C006-H-LEVER-001".to_string(),
            description:
                "Add same-rank/identical-shape fast paths in broadcast_shape".to_string(),
            rationale:
                "Stride-tricks and broadcast planning repeatedly merge same-rank shapes; bypassing reverse-index lookups and reverse() cuts per-call overhead while preserving merge semantics."
                    .to_string(),
            risk_note:
                "Fast paths only restructure shape-merge control flow; incompatibility detection and error-class behavior are unchanged."
                    .to_string(),
            rollback_command:
                "git restore --source <last-green-commit> -- crates/fnp-ndarray/src/lib.rs"
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
                    "Usage: cargo run -p fnp-conformance --bin generate_packet006_optimization_report -- [--output-path <path>]"
                );
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }

    Ok(output_path.unwrap_or_else(|| {
        repo_root.join("artifacts/phase2c/FNP-P2C-006/optimization_profile_report.json")
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
        let checksum = run_fn()?;
        std::hint::black_box(checksum);
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

fn run_workload<F>(workload: &[(Vec<usize>, Vec<usize>)], mut merge_fn: F) -> Result<usize, String>
where
    F: FnMut(&[usize], &[usize]) -> Result<Vec<usize>, ShapeError>,
{
    let mut checksum = 0usize;
    for (lhs, rhs) in workload {
        let out = merge_fn(lhs, rhs)
            .map_err(|err| format!("broadcast merge failed lhs={lhs:?} rhs={rhs:?}: {err}"))?;
        checksum = checksum.wrapping_add(out.iter().copied().sum::<usize>());
    }
    Ok(checksum)
}

fn run_isomorphism_checks() -> Vec<IsomorphismCheck> {
    vec![
        compare_case("case_identical_shape", &[4, 8, 16], &[4, 8, 16]),
        compare_case("case_same_rank_axiswise", &[1, 4, 1, 3], &[2, 1, 5, 3]),
        compare_case("case_rank_mismatch", &[8, 1, 6, 1], &[7, 1, 5]),
        compare_case("case_incompatible_shape", &[4, 3], &[5, 3]),
    ]
}

fn compare_case(case_id: &str, lhs: &[usize], rhs: &[usize]) -> IsomorphismCheck {
    let baseline = legacy_broadcast_shape(lhs, rhs);
    let optimized = broadcast_shape(lhs, rhs);
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

fn legacy_broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, ShapeError> {
    let nd = lhs.len().max(rhs.len());
    let mut out = Vec::with_capacity(nd);

    for idx in 0..nd {
        let l = *lhs.get(lhs.len().wrapping_sub(1 + idx)).unwrap_or(&1);
        let r = *rhs.get(rhs.len().wrapping_sub(1 + idx)).unwrap_or(&1);

        let merged = if l == r {
            l
        } else if l == 1 {
            r
        } else if r == 1 {
            l
        } else {
            return Err(ShapeError::IncompatibleBroadcast {
                lhs: lhs.to_vec(),
                rhs: rhs.to_vec(),
            });
        };

        out.push(merged);
    }

    out.reverse();
    Ok(out)
}

#[must_use]
fn build_workload(case_repetitions: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    let seed_cases: [([usize; 4], [usize; 4]); 5] = [
        ([1, 256, 1, 64], [32, 1, 8, 64]),
        ([4, 8, 16, 32], [4, 8, 16, 32]),
        ([1, 64, 1, 64], [32, 64, 8, 1]),
        ([2, 3, 5, 7], [1, 3, 1, 7]),
        ([1, 512, 1, 2], [8, 1, 64, 2]),
    ];

    let mut out = Vec::with_capacity(case_repetitions * seed_cases.len());
    for _ in 0..case_repetitions {
        for (lhs, rhs) in seed_cases {
            out.push((lhs.to_vec(), rhs.to_vec()));
        }
    }
    out
}

fn summarize_samples(samples_ms: &[f64]) -> PercentileSummary {
    let mut sorted = samples_ms.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

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
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn compute_throughput(elements: f64, millis: f64) -> f64 {
    if millis <= f64::EPSILON {
        return 0.0;
    }
    elements / (millis / 1000.0)
}

fn percent_delta(baseline: f64, candidate: f64) -> f64 {
    if baseline.abs() <= f64::EPSILON {
        return 0.0;
    }
    ((candidate - baseline) / baseline) * 100.0
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn environment_fingerprint() -> String {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let cpus = std::thread::available_parallelism().map_or(1usize, std::num::NonZeroUsize::get);
    let rustc = Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map_or_else(
            || "rustc-unknown".to_string(),
            |value| value.trim().to_string(),
        );

    format!("os={os} arch={arch} cpus={cpus} rustc={rustc}")
}
