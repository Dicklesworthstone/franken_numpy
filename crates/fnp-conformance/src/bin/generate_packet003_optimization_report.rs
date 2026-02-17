#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const PACKET_ID: &str = "FNP-P2C-003";
const SUBTASK_ID: &str = "FNP-P2C-003-H";

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
    lookups_per_run: usize,
    percentiles: PercentileSummary,
    throughput_lookups_per_sec_p50: f64,
    throughput_lookups_per_sec_p95: f64,
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
struct Packet003OptimizationReport {
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

#[derive(Debug, Deserialize)]
struct TransferDifferentialFixtureCase {
    id: String,
}

#[derive(Debug, Deserialize)]
struct TransferAdversarialFixtureCase {
    id: String,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("generate_packet003_optimization_report failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output_path = parse_output_path(&repo_root)?;

    let fixture_root = repo_root.join("crates/fnp-conformance/fixtures/packet003_transfer");
    let differential_ids =
        load_transfer_differential_case_ids(&fixture_root.join("iter_differential_cases.json"))?;
    let adversarial_ids =
        load_transfer_adversarial_case_ids(&fixture_root.join("iter_adversarial_cases.json"))?;

    let mut all_case_ids = differential_ids;
    all_case_ids.extend(adversarial_ids);
    if all_case_ids.is_empty() {
        return Err(format!(
            "packet003 transfer fixture IDs are empty in {}",
            fixture_root.display()
        ));
    }

    let scaled_case_ids = expand_case_ids(&all_case_ids, 192);
    let selected_case_ids = select_case_ids(&scaled_case_ids, 2048);
    let query_workload = build_query_workload(&selected_case_ids, 72);
    let lookups_per_run = query_workload.len();

    let case_index_map = build_case_index_map(&scaled_case_ids);

    let baseline_profile =
        profile_implementation("baseline_linear_id_scan", 80, lookups_per_run, || {
            run_linear_workload(&scaled_case_ids, &query_workload)
        });

    let rebaseline_profile =
        profile_implementation("rebaseline_hash_map_lookup", 80, lookups_per_run, || {
            run_map_workload(&case_index_map, &query_workload)
        });

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
            baseline_profile.throughput_lookups_per_sec_p50,
            rebaseline_profile.throughput_lookups_per_sec_p50,
        ),
        throughput_gain_percent_p95: percent_delta(
            baseline_profile.throughput_lookups_per_sec_p95,
            rebaseline_profile.throughput_lookups_per_sec_p95,
        ),
    };

    let isomorphism_checks =
        run_isomorphism_checks(&scaled_case_ids, &case_index_map, &selected_case_ids);
    let all_checks_pass = isomorphism_checks
        .iter()
        .all(|check| check.status == "pass");

    let impact = if delta.throughput_gain_percent_p95 >= 10.0 {
        4.0
    } else if delta.throughput_gain_percent_p95 > 0.0 {
        3.0
    } else {
        1.0
    };
    let confidence = if all_checks_pass { 4.0 } else { 1.0 };
    let reuse = 4.0;
    let effort = 2.0;
    let adoption_friction = 1.0;
    let score = (impact * confidence * reuse) / (effort * adoption_friction);
    let promoted = score >= 2.0
        && all_checks_pass
        && delta.throughput_gain_percent_p95 > 0.0
        && delta.p95_delta_percent < 0.0;

    let ev = EVSummary {
        impact,
        confidence,
        reuse,
        effort,
        adoption_friction,
        score,
        promoted,
    };

    let report = Packet003OptimizationReport {
        schema_version: 1,
        packet_id: PACKET_ID.to_string(),
        subtask_id: SUBTASK_ID.to_string(),
        generated_at_unix_ms: now_unix_ms(),
        environment_fingerprint: environment_fingerprint(),
        reproducibility_command:
            "cargo run -p fnp-conformance --bin generate_packet003_optimization_report"
                .to_string(),
        lever: LeverSummary {
            id: "P2C003-H-LEVER-001".to_string(),
            description:
                "Replace linear transfer-fixture ID scans with precomputed hash-index dispatch"
                    .to_string(),
            rationale:
                "Packet-003 workflow replay repeatedly resolves transfer fixture IDs for strict/hardened scenario steps; replacing linear scans with precomputed HashMap lookups removes O(n) per-step dispatch overhead while preserving deterministic fixture resolution semantics."
                    .to_string(),
            risk_note:
                "Optimization only changes fixture ID lookup strategy in workflow dispatch; fixture payloads, ordering, and reason-code outcomes remain unchanged."
                    .to_string(),
            rollback_command:
                "git restore --source <last-green-commit> -- crates/fnp-conformance/src/workflow_scenarios.rs"
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

    println!("BEGIN_PACKET003_OPTIMIZATION_REPORT_JSON");
    println!(
        "{}",
        fs::read_to_string(&output_path)
            .map_err(|err| format!("failed reading {}: {err}", output_path.display()))?
    );
    println!("END_PACKET003_OPTIMIZATION_REPORT_JSON");
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
                    "Usage: cargo run -p fnp-conformance --bin generate_packet003_optimization_report -- [--output-path <path>]"
                );
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }

    Ok(output_path.unwrap_or_else(|| {
        repo_root.join("artifacts/phase2c/FNP-P2C-003/optimization_profile_report.json")
    }))
}

fn load_transfer_differential_case_ids(path: &Path) -> Result<Vec<String>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<TransferDifferentialFixtureCase> = serde_json::from_str(&raw)
        .map_err(|err| format!("invalid JSON {}: {err}", path.display()))?;
    Ok(cases.into_iter().map(|case| case.id).collect())
}

fn load_transfer_adversarial_case_ids(path: &Path) -> Result<Vec<String>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    let cases: Vec<TransferAdversarialFixtureCase> = serde_json::from_str(&raw)
        .map_err(|err| format!("invalid JSON {}: {err}", path.display()))?;
    Ok(cases.into_iter().map(|case| case.id).collect())
}

fn select_case_ids(case_ids: &[String], max_count: usize) -> Vec<String> {
    let count = max_count.min(case_ids.len());
    case_ids.iter().take(count).cloned().collect()
}

fn expand_case_ids(case_ids: &[String], multiplier: usize) -> Vec<String> {
    let mut expanded = Vec::with_capacity(case_ids.len().saturating_mul(multiplier));
    for repeat in 0..multiplier {
        for case_id in case_ids {
            if repeat == 0 {
                expanded.push(case_id.clone());
            } else {
                expanded.push(format!("{case_id}::scaled::{repeat}"));
            }
        }
    }
    expanded
}

fn build_query_workload(case_ids: &[String], repeats: usize) -> Vec<String> {
    let window: Vec<String> = case_ids.iter().take(256).cloned().collect();
    if window.is_empty() {
        return Vec::new();
    }

    let mut workload = Vec::with_capacity(window.len().saturating_mul(repeats).saturating_mul(2));
    for repeat in 0..repeats {
        for (idx, case_id) in window.iter().enumerate() {
            workload.push(case_id.clone());
            if idx % 8 == 0 {
                workload.push(format!("unknown_transfer_case::{repeat}::{idx}"));
            }
        }
    }
    workload
}

fn build_case_index_map(case_ids: &[String]) -> HashMap<String, usize> {
    case_ids
        .iter()
        .enumerate()
        .map(|(idx, case_id)| (case_id.clone(), idx))
        .collect()
}

fn run_linear_workload(case_ids: &[String], query_workload: &[String]) -> Result<usize, String> {
    let mut hits = 0usize;
    for query in query_workload {
        if case_ids.iter().any(|candidate| candidate == query) {
            hits += 1;
        }
    }
    Ok(hits)
}

fn run_map_workload(
    case_index_map: &HashMap<String, usize>,
    query_workload: &[String],
) -> Result<usize, String> {
    let mut hits = 0usize;
    for query in query_workload {
        if case_index_map.contains_key(query) {
            hits += 1;
        }
    }
    Ok(hits)
}

fn profile_implementation<F>(
    implementation: &str,
    runs: usize,
    lookups_per_run: usize,
    mut run_fn: F,
) -> ProfileSummary
where
    F: FnMut() -> Result<usize, String>,
{
    let mut samples_ms = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        let hits = run_fn().unwrap_or(0);
        std::hint::black_box(hits);
        samples_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let percentiles = summarize_samples(&samples_ms);
    let throughput_lookups_per_sec_p50 =
        compute_throughput(lookups_per_run as f64, percentiles.p50_ms);
    let throughput_lookups_per_sec_p95 =
        compute_throughput(lookups_per_run as f64, percentiles.p95_ms);

    ProfileSummary {
        implementation: implementation.to_string(),
        runs,
        lookups_per_run,
        percentiles,
        throughput_lookups_per_sec_p50,
        throughput_lookups_per_sec_p95,
    }
}

fn run_isomorphism_checks(
    case_ids: &[String],
    case_index_map: &HashMap<String, usize>,
    selected_case_ids: &[String],
) -> Vec<IsomorphismCheck> {
    let known_queries: Vec<String> = selected_case_ids.iter().take(64).cloned().collect();
    let unknown_queries: Vec<String> = (0..32)
        .map(|idx| format!("definitely_missing_transfer_case::{idx}"))
        .collect();
    let mixed_queries = known_queries
        .iter()
        .take(16)
        .cloned()
        .chain(unknown_queries.iter().take(16).cloned())
        .collect::<Vec<_>>();
    let scaled_queries = case_ids
        .iter()
        .filter(|id| id.contains("::scaled::"))
        .take(32)
        .cloned()
        .collect::<Vec<_>>();

    vec![
        compare_query_set("case_known_hits", case_ids, case_index_map, &known_queries),
        compare_query_set(
            "case_unknown_misses",
            case_ids,
            case_index_map,
            &unknown_queries,
        ),
        compare_query_set(
            "case_mixed_hit_miss",
            case_ids,
            case_index_map,
            &mixed_queries,
        ),
        compare_query_set(
            "case_scaled_suffix_hits",
            case_ids,
            case_index_map,
            if scaled_queries.is_empty() {
                &known_queries
            } else {
                &scaled_queries
            },
        ),
    ]
}

fn compare_query_set(
    case_id: &str,
    case_ids: &[String],
    case_index_map: &HashMap<String, usize>,
    queries: &[String],
) -> IsomorphismCheck {
    let baseline = run_linear_workload(case_ids, queries).unwrap_or(0);
    let optimized = run_map_workload(case_index_map, queries).unwrap_or(0);

    if baseline == optimized {
        IsomorphismCheck {
            case_id: case_id.to_string(),
            status: "pass".to_string(),
            details: format!(
                "baseline_hits={baseline} optimized_hits={optimized} query_count={}",
                queries.len()
            ),
        }
    } else {
        IsomorphismCheck {
            case_id: case_id.to_string(),
            status: "fail".to_string(),
            details: format!(
                "lookup mismatch baseline_hits={baseline} optimized_hits={optimized} query_count={}",
                queries.len()
            ),
        }
    }
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

fn compute_throughput(lookups: f64, latency_ms: f64) -> f64 {
    if latency_ms <= 0.0 {
        return 0.0;
    }
    lookups / (latency_ms / 1000.0)
}

fn percent_delta(baseline: f64, updated: f64) -> f64 {
    if baseline == 0.0 {
        return 0.0;
    }
    ((updated - baseline) / baseline) * 100.0
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
