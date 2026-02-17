#![forbid(unsafe_code)]

use fnp_io::{IOError, synthesize_npz_member_names};
use serde::Serialize;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const PACKET_ID: &str = "FNP-P2C-009";
const SUBTASK_ID: &str = "FNP-P2C-009-H";

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
    members_per_run: usize,
    percentiles: PercentileSummary,
    throughput_members_per_sec_p50: f64,
    throughput_members_per_sec_p95: f64,
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
struct Packet009OptimizationReport {
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
        eprintln!("generate_packet009_optimization_report failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output_path = parse_output_path(&repo_root)?;

    let positional_count = 2048usize;
    let keyword_names = (0..1024)
        .map(|idx| format!("kw_{idx:04}"))
        .collect::<Vec<_>>();
    let keyword_refs = keyword_names.iter().map(String::as_str).collect::<Vec<_>>();
    let members_per_run = positional_count + keyword_refs.len();

    let baseline_profile =
        profile_implementation("baseline_btreeset", 60, members_per_run, || {
            legacy_synthesize_npz_member_names(positional_count, &keyword_refs)
        })?;
    let rebaseline_profile =
        profile_implementation("rebaseline_hashset", 60, members_per_run, || {
            synthesize_npz_member_names(positional_count, &keyword_refs)
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
            baseline_profile.throughput_members_per_sec_p50,
            rebaseline_profile.throughput_members_per_sec_p50,
        ),
        throughput_gain_percent_p95: percent_delta(
            baseline_profile.throughput_members_per_sec_p95,
            rebaseline_profile.throughput_members_per_sec_p95,
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

    let report = Packet009OptimizationReport {
        schema_version: 1,
        packet_id: PACKET_ID.to_string(),
        subtask_id: SUBTASK_ID.to_string(),
        generated_at_unix_ms: now_unix_ms(),
        environment_fingerprint: environment_fingerprint(),
        reproducibility_command:
            "cargo run -p fnp-conformance --bin generate_packet009_optimization_report"
                .to_string(),
        lever: LeverSummary {
            id: "P2C009-H-LEVER-001".to_string(),
            description:
                "Replace NPZ member uniqueness structure from BTreeSet to HashSet in synthesis path"
                    .to_string(),
            rationale:
                "Member-name uniqueness checks are membership-heavy and do not require sorted iteration; HashSet reduces lookup cost while preserving deterministic name emission order in output vector."
                    .to_string(),
            risk_note:
                "Behavioral risk is low because set ordering is not externally observed; output ordering still follows positional then keyword insertion sequence."
                    .to_string(),
            rollback_command: "git checkout -- crates/fnp-io/src/lib.rs".to_string(),
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
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed creating report directory {}: {err}",
                parent.display()
            )
        })?;
    }
    fs::write(&output_path, raw).map_err(|err| {
        format!(
            "failed writing optimization report {}: {err}",
            output_path.display()
        )
    })?;

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
                    "Usage: cargo run -p fnp-conformance --bin generate_packet009_optimization_report -- [--output-path <path>]"
                );
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }

    Ok(output_path.unwrap_or_else(|| {
        repo_root.join("artifacts/phase2c/FNP-P2C-009/optimization_profile_report.json")
    }))
}

fn profile_implementation<F>(
    implementation: &str,
    runs: usize,
    members_per_run: usize,
    mut run_fn: F,
) -> Result<ProfileSummary, String>
where
    F: FnMut() -> Result<Vec<String>, IOError>,
{
    let mut samples_ms = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        let names = run_fn().map_err(|err| format!("{implementation} run failed: {err}"))?;
        std::hint::black_box(names.len());
        samples_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let percentiles = summarize_samples(&samples_ms);
    let throughput_members_per_sec_p50 =
        compute_throughput_members_per_sec(members_per_run as f64, percentiles.p50_ms);
    let throughput_members_per_sec_p95 =
        compute_throughput_members_per_sec(members_per_run as f64, percentiles.p95_ms);

    Ok(ProfileSummary {
        implementation: implementation.to_string(),
        runs,
        members_per_run,
        percentiles,
        throughput_members_per_sec_p50,
        throughput_members_per_sec_p95,
    })
}

fn legacy_synthesize_npz_member_names(
    positional_count: usize,
    keyword_names: &[&str],
) -> Result<Vec<String>, IOError> {
    let member_count = positional_count.checked_add(keyword_names.len()).ok_or(
        IOError::NpzArchiveContractViolation("archive member count overflowed"),
    )?;
    if member_count == 0 || member_count > fnp_io::MAX_ARCHIVE_MEMBERS {
        return Err(IOError::NpzArchiveContractViolation(
            "archive member count is outside bounded limits",
        ));
    }

    let mut names = Vec::with_capacity(member_count);
    let mut seen = BTreeSet::new();

    for idx in 0..positional_count {
        let name = format!("arr_{idx}");
        let _ = seen.insert(name.clone());
        names.push(name);
    }

    for &name in keyword_names {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(IOError::NpzArchiveContractViolation(
                "keyword member name cannot be empty",
            ));
        }
        if !seen.insert(trimmed.to_string()) {
            return Err(IOError::NpzArchiveContractViolation(
                "archive member names must be unique",
            ));
        }
        names.push(trimmed.to_string());
    }

    Ok(names)
}

fn run_isomorphism_checks() -> Vec<IsomorphismCheck> {
    let cases = [
        (
            "case_valid_small",
            3usize,
            vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()],
        ),
        (
            "case_valid_large",
            64usize,
            (0..64)
                .map(|idx| format!("name_{idx:02}"))
                .collect::<Vec<_>>(),
        ),
        (
            "case_duplicate_keyword",
            1usize,
            vec!["dup".to_string(), "dup".to_string()],
        ),
        (
            "case_empty_keyword",
            1usize,
            vec!["ok".to_string(), " ".to_string()],
        ),
    ];

    cases
        .into_iter()
        .map(|(case_id, positional_count, keywords)| {
            let refs = keywords.iter().map(String::as_str).collect::<Vec<_>>();
            let baseline = legacy_synthesize_npz_member_names(positional_count, &refs);
            let optimized = synthesize_npz_member_names(positional_count, &refs);
            if canonicalize_result(&baseline) == canonicalize_result(&optimized) {
                IsomorphismCheck {
                    case_id: case_id.to_string(),
                    status: "pass".to_string(),
                    details: "baseline and optimized outputs matched".to_string(),
                }
            } else {
                IsomorphismCheck {
                    case_id: case_id.to_string(),
                    status: "fail".to_string(),
                    details: format!(
                        "baseline={} optimized={}",
                        canonicalize_result(&baseline),
                        canonicalize_result(&optimized)
                    ),
                }
            }
        })
        .collect()
}

fn canonicalize_result(result: &Result<Vec<String>, IOError>) -> String {
    match result {
        Ok(values) => format!("ok:{}", values.join("|")),
        Err(err) => format!("err:{}", err.reason_code()),
    }
}

fn summarize_samples(samples: &[f64]) -> PercentileSummary {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|lhs, rhs| lhs.total_cmp(rhs));

    let min_ms = sorted.first().copied().unwrap_or(0.0);
    let max_ms = sorted.last().copied().unwrap_or(0.0);
    let p50_ms = sorted
        .get(percentile_index(sorted.len(), 50))
        .copied()
        .unwrap_or(0.0);
    let p95_ms = sorted
        .get(percentile_index(sorted.len(), 95))
        .copied()
        .unwrap_or(0.0);
    let p99_ms = sorted
        .get(percentile_index(sorted.len(), 99))
        .copied()
        .unwrap_or(0.0);

    PercentileSummary {
        p50_ms,
        p95_ms,
        p99_ms,
        min_ms,
        max_ms,
    }
}

fn percentile_index(len: usize, percentile_num: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let last = len - 1;
    (last * percentile_num + 50) / 100
}

fn compute_throughput_members_per_sec(members_per_run: f64, sample_ms: f64) -> f64 {
    if sample_ms <= 0.0 {
        return 0.0;
    }
    members_per_run * 1000.0 / sample_ms
}

fn percent_delta(reference: f64, candidate: f64) -> f64 {
    if reference <= 0.0 {
        return 0.0;
    }
    ((candidate - reference) / reference) * 100.0
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn environment_fingerprint() -> String {
    let rustc = Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    format!(
        "os={} arch={} cpus={} rustc={}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::thread::available_parallelism().map_or(1, usize::from),
        rustc
    )
}
