#![forbid(unsafe_code)]

use fnp_conformance::benchmark::{
    ALLOCATION_CHURN_SLO_PATH, ALLOCATOR_FRAGMENTATION_SLO_PATH, BenchmarkBaseline,
    BenchmarkWorkload, MEMORY_FOOTPRINT_SLO_PATH, REQUIRED_SLO_PATHS, generate_benchmark_baseline,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy)]
struct WorkloadBudget {
    name: &'static str,
    path_family: &'static str,
    p95_budget_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct WorkloadDeltaSummary {
    name: String,
    path_family: String,
    reference_p95_ms: Option<f64>,
    candidate_p95_ms: Option<f64>,
    reference_p99_ms: Option<f64>,
    candidate_p99_ms: Option<f64>,
    p95_delta_percent: Option<f64>,
    p99_delta_percent: Option<f64>,
    status: String,
    violations: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ReliabilityDiagnostic {
    subsystem: String,
    reason_code: String,
    message: String,
    evidence_refs: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    workload_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    path_family: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    expected_measurement_fields: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    remediation: Option<String>,
}

impl ReliabilityDiagnostic {
    fn generic(
        subsystem: impl Into<String>,
        reason_code: impl Into<String>,
        message: impl Into<String>,
        evidence_refs: Vec<String>,
    ) -> Self {
        Self {
            subsystem: subsystem.into(),
            reason_code: reason_code.into(),
            message: message.into(),
            evidence_refs,
            workload_name: None,
            path_family: None,
            expected_measurement_fields: Vec::new(),
            remediation: None,
        }
    }

    fn for_workload(
        budget: &WorkloadBudget,
        reason_code: impl Into<String>,
        message: impl Into<String>,
        evidence_refs: Vec<String>,
        expected_measurement_fields: Vec<String>,
        remediation: impl Into<String>,
    ) -> Self {
        Self {
            subsystem: budget.name.to_string(),
            reason_code: reason_code.into(),
            message: message.into(),
            evidence_refs,
            workload_name: Some(budget.name.to_string()),
            path_family: Some(budget.path_family.to_string()),
            expected_measurement_fields,
            remediation: Some(remediation.into()),
        }
    }
}

#[derive(Debug, Serialize)]
struct ReliabilitySummary {
    coverage_floor: f64,
    coverage_ratio: f64,
    max_p99_regression_ratio: f64,
    missing_instrumentation_policy: &'static str,
    diagnostics: Vec<ReliabilityDiagnostic>,
    warnings: Vec<ReliabilityDiagnostic>,
}

#[derive(Debug, Serialize)]
struct GateSummary {
    status: &'static str,
    reference_path: String,
    candidate_path: String,
    reference_git_commit: String,
    candidate_git_commit: String,
    reference_environment_fingerprint: String,
    candidate_environment_fingerprint: String,
    workloads: Vec<WorkloadDeltaSummary>,
    uninstrumented_budget_paths: Vec<String>,
    reliability: ReliabilitySummary,
    report_path: Option<String>,
}

#[derive(Debug)]
struct GateOptions {
    reference_path: PathBuf,
    candidate_path: PathBuf,
    report_path: Option<PathBuf>,
    max_p99_regression_ratio: f64,
    coverage_floor: f64,
    generate_candidate: bool,
    /// Same-job A/B mode (what CI's G7 runs): one `generate_benchmark_baseline` JSON per arm
    /// per round, both arms built and run on ONE host in alternating order.
    ab_reference_runs: Vec<PathBuf>,
    ab_candidate_runs: Vec<PathBuf>,
    max_median_regression_ratio: f64,
}

/// One budgeted workload in a same-job A/B run. The per-round statistic is the median of that
/// run's samples; the effect is the per-round candidate/reference ratio (the two arms of a round
/// ran back to back), and each arm's consecutive-round ratios are its A/A null.
#[derive(Debug, Clone, Serialize)]
struct AbWorkloadSummary {
    name: String,
    path_family: String,
    rounds: usize,
    reference_median_ms: Option<f64>,
    candidate_median_ms: Option<f64>,
    candidate_p95_ms: Option<f64>,
    effect_median_ratio: Option<f64>,
    effect_ci95: Option<[f64; 2]>,
    reference_null_ci95: Option<[f64; 2]>,
    candidate_null_ci95: Option<[f64; 2]>,
    null_half_width: Option<f64>,
    verdict: String,
    status: String,
    violations: Vec<String>,
}

#[derive(Debug, Serialize)]
struct AbReliabilitySummary {
    coverage_floor: f64,
    coverage_ratio: f64,
    max_median_regression_ratio: f64,
    missing_instrumentation_policy: &'static str,
    diagnostics: Vec<ReliabilityDiagnostic>,
}

#[derive(Debug, Serialize)]
struct AbGateSummary {
    status: &'static str,
    mode: &'static str,
    rounds: usize,
    bootstrap_resamples: usize,
    decision_rule: &'static str,
    reference_git_commits: Vec<String>,
    candidate_git_commits: Vec<String>,
    reference_environment_fingerprints: Vec<String>,
    candidate_environment_fingerprints: Vec<String>,
    reference_cargo_profiles: Vec<String>,
    candidate_cargo_profiles: Vec<String>,
    reference_runs: Vec<String>,
    candidate_runs: Vec<String>,
    verdict_counts: BTreeMap<String, usize>,
    workloads: Vec<AbWorkloadSummary>,
    uninstrumented_budget_paths: Vec<String>,
    reliability: AbReliabilitySummary,
    report_path: Option<String>,
}

const AB_MIN_ROUNDS: usize = 3;
const AB_BOOTSTRAP_RESAMPLES: usize = 4000;
const AB_DECISION_RULE: &str = "fail iff effect CI95 lower bound > 1, effect median - 1 > 2 x the larger A/A null half-width (measured from 1), and effect median - 1 > max_median_regression_ratio";
const VERDICT_REGRESSION: &str = "decidable_regression";
const VERDICT_WIN: &str = "decidable_win";
const VERDICT_UNDECIDED: &str = "undecided";
const VERDICT_NO_REFERENCE: &str = "no_reference";
const VERDICT_MISSING_CANDIDATE: &str = "missing_candidate";

const MISSING_INSTRUMENTATION_POLICY: &str = "fail_closed";

const WORKLOAD_BUDGETS: &[WorkloadBudget] = &[
    WorkloadBudget {
        name: "ufunc_add_broadcast_256x256_by_256",
        path_family: "broadcast add/mul",
        p95_budget_ms: 180.0,
    },
    WorkloadBudget {
        name: "ufunc_add_broadcast_1024x1024_by_1024",
        path_family: "broadcast add/mul",
        p95_budget_ms: 1300.0,
    },
    WorkloadBudget {
        name: "reduce_sum_axis1_keepdims_false_256x256",
        path_family: "reduction sum/mean",
        p95_budget_ms: 210.0,
    },
    WorkloadBudget {
        name: "reduce_sum_all_keepdims_false_256x256",
        path_family: "reduction sum/mean",
        p95_budget_ms: 210.0,
    },
    WorkloadBudget {
        name: "matmul_256x256_by_256x256",
        path_family: "matmul/dot",
        p95_budget_ms: 2400.0,
    },
    WorkloadBudget {
        name: "sort_quicksort_1m",
        path_family: "sorting/searching",
        p95_budget_ms: 1600.0,
    },
    WorkloadBudget {
        name: "fft_65536",
        path_family: "fft transforms",
        p95_budget_ms: 1200.0,
    },
    WorkloadBudget {
        name: "astype_f64_to_i32_1024x1024",
        path_family: "dtype conversion",
        p95_budget_ms: 950.0,
    },
    WorkloadBudget {
        name: "reshape_1024x1024_to_2048x512",
        path_family: "reshape/view operations",
        p95_budget_ms: 250.0,
    },
    WorkloadBudget {
        name: "io_npy_save_load_512x512_f64",
        path_family: "npy parse + load",
        p95_budget_ms: 1800.0,
    },
];

fn main() {
    if let Err(err) = run() {
        eprintln!("run_performance_budget_gate failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let options = parse_args()?;
    if !options.ab_reference_runs.is_empty() || !options.ab_candidate_runs.is_empty() {
        return run_ab(&options);
    }
    if options.generate_candidate {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        generate_benchmark_baseline(&repo_root, &options.candidate_path)?;
    }
    let reference = load_baseline(&options.reference_path)?;
    let candidate = load_baseline(&options.candidate_path)?;

    let summary = evaluate_gate(&options, reference, candidate)?;
    let status = summary.status;
    let summary_json = serde_json::to_string_pretty(&summary)
        .map_err(|err| format!("failed serializing summary: {err}"))?;

    if let Some(report_path) = options.report_path {
        if let Some(parent) = report_path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed creating report directory {}: {err}",
                    parent.display()
                )
            })?;
        }
        fs::write(&report_path, summary_json.as_bytes())
            .map_err(|err| format!("failed writing report {}: {err}", report_path.display()))?;
    }

    println!("{summary_json}");

    if status == "fail" {
        std::process::exit(2);
    }

    Ok(())
}

fn evaluate_gate(
    options: &GateOptions,
    reference: BenchmarkBaseline,
    candidate: BenchmarkBaseline,
) -> Result<GateSummary, String> {
    let reference_map: BTreeMap<&str, &BenchmarkWorkload> = reference
        .workloads
        .iter()
        .map(|workload| (workload.name.as_str(), workload))
        .collect();
    let candidate_map: BTreeMap<&str, &BenchmarkWorkload> = candidate
        .workloads
        .iter()
        .map(|workload| (workload.name.as_str(), workload))
        .collect();

    let mut diagnostics = Vec::new();
    let mut workload_summaries = Vec::with_capacity(WORKLOAD_BUDGETS.len());
    let mut covered_workloads = 0usize;

    for budget in WORKLOAD_BUDGETS {
        let reference_workload = reference_map.get(budget.name).copied();
        let candidate_workload = candidate_map.get(budget.name).copied();
        let (summary, mut workload_diagnostics, covered) = evaluate_budget(
            budget,
            reference_workload,
            candidate_workload,
            options.max_p99_regression_ratio,
            &options.reference_path,
            &options.candidate_path,
        );
        workload_summaries.push(summary);
        diagnostics.append(&mut workload_diagnostics);
        if covered {
            covered_workloads += 1;
        }
    }

    let coverage_ratio = if WORKLOAD_BUDGETS.is_empty() {
        0.0
    } else {
        covered_workloads as f64 / WORKLOAD_BUDGETS.len() as f64
    };

    if coverage_ratio + f64::EPSILON < options.coverage_floor {
        diagnostics.push(ReliabilityDiagnostic::generic(
            "performance_budget",
            "coverage_floor_breach",
            format!(
                "coverage ratio {:.6} is below floor {:.6}",
                coverage_ratio, options.coverage_floor
            ),
            vec![
                options.reference_path.display().to_string(),
                options.candidate_path.display().to_string(),
            ],
        ));
    }

    let uninstrumented_budget_paths = missing_slo_paths(&candidate);
    diagnostics.extend(missing_slo_path_diagnostics(
        &uninstrumented_budget_paths,
        &options.candidate_path,
    ));
    let warnings = Vec::new();

    let status = if diagnostics.is_empty() {
        "pass"
    } else {
        "fail"
    };
    let report_path = options
        .report_path
        .as_ref()
        .map(|path| path.display().to_string());

    Ok(GateSummary {
        status,
        reference_path: options.reference_path.display().to_string(),
        candidate_path: options.candidate_path.display().to_string(),
        reference_git_commit: reference.git_commit,
        candidate_git_commit: candidate.git_commit,
        reference_environment_fingerprint: reference.environment_fingerprint,
        candidate_environment_fingerprint: candidate.environment_fingerprint,
        workloads: workload_summaries,
        uninstrumented_budget_paths,
        reliability: ReliabilitySummary {
            coverage_floor: options.coverage_floor,
            coverage_ratio,
            max_p99_regression_ratio: options.max_p99_regression_ratio,
            missing_instrumentation_policy: MISSING_INSTRUMENTATION_POLICY,
            diagnostics,
            warnings,
        },
        report_path,
    })
}

fn run_ab(options: &GateOptions) -> Result<(), String> {
    let reference = options
        .ab_reference_runs
        .iter()
        .map(|path| load_baseline(path))
        .collect::<Result<Vec<_>, _>>()?;
    let candidate = options
        .ab_candidate_runs
        .iter()
        .map(|path| load_baseline(path))
        .collect::<Result<Vec<_>, _>>()?;
    let summary = evaluate_ab_gate(options, &reference, &candidate)?;
    let summary_json = serde_json::to_string_pretty(&summary)
        .map_err(|err| format!("failed serializing summary: {err}"))?;
    if let Some(report_path) = &options.report_path {
        if let Some(parent) = report_path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed creating report directory {}: {err}",
                    parent.display()
                )
            })?;
        }
        fs::write(report_path, summary_json.as_bytes())
            .map_err(|err| format!("failed writing report {}: {err}", report_path.display()))?;
    }
    println!("{summary_json}");
    if summary.status == "fail" {
        std::process::exit(2);
    }
    Ok(())
}

/// The same-job A/B gate. Both arms ran on ONE host in the same job, alternating which arm went
/// first each round, so a ratio of the two measures code rather than hardware - the defect of the
/// single-snapshot mode, which compared a baseline captured on another host months earlier with
/// the runner's own measurement (bead deadlock-audit-rc0923-epic-71qy3.28).
fn evaluate_ab_gate(
    options: &GateOptions,
    reference: &[BenchmarkBaseline],
    candidate: &[BenchmarkBaseline],
) -> Result<AbGateSummary, String> {
    if reference.len() != candidate.len() {
        return Err(format!(
            "A/B mode needs one reference run per candidate run: {} reference, {} candidate",
            reference.len(),
            candidate.len()
        ));
    }
    if reference.len() < AB_MIN_ROUNDS {
        return Err(format!(
            "A/B mode needs at least {AB_MIN_ROUNDS} rounds, got {}",
            reference.len()
        ));
    }

    let mut diagnostics = Vec::new();
    let mut workloads = Vec::with_capacity(WORKLOAD_BUDGETS.len());
    let mut covered = 0usize;
    for budget in WORKLOAD_BUDGETS {
        let (summary, mut workload_diagnostics) = evaluate_ab_workload(
            budget,
            reference,
            candidate,
            options.max_median_regression_ratio,
        )?;
        if summary.verdict != VERDICT_MISSING_CANDIDATE {
            covered += 1;
        }
        diagnostics.append(&mut workload_diagnostics);
        workloads.push(summary);
    }

    // Coverage is a property of the instrument under test, the CANDIDATE: a workload the
    // reference predates has nothing to regress against and is reported `no_reference`.
    let coverage_ratio = if WORKLOAD_BUDGETS.is_empty() {
        0.0
    } else {
        covered as f64 / WORKLOAD_BUDGETS.len() as f64
    };
    if coverage_ratio + f64::EPSILON < options.coverage_floor {
        diagnostics.push(ReliabilityDiagnostic::generic(
            "performance_budget",
            "coverage_floor_breach",
            format!(
                "coverage ratio {:.6} is below floor {:.6}",
                coverage_ratio, options.coverage_floor
            ),
            options
                .ab_candidate_runs
                .iter()
                .map(|path| path.display().to_string())
                .collect(),
        ));
    }

    let first_candidate_path = options
        .ab_candidate_runs
        .first()
        .cloned()
        .unwrap_or_default();
    let uninstrumented_budget_paths = missing_slo_paths(&candidate[0]);
    diagnostics.extend(missing_slo_path_diagnostics(
        &uninstrumented_budget_paths,
        &first_candidate_path,
    ));

    let mut verdict_counts = BTreeMap::new();
    for workload in &workloads {
        *verdict_counts.entry(workload.verdict.clone()).or_insert(0) += 1;
    }
    let distinct = |runs: &[BenchmarkBaseline], field: fn(&BenchmarkBaseline) -> String| {
        runs.iter()
            .map(field)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
    };
    let paths = |runs: &[PathBuf]| {
        runs.iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
    };

    Ok(AbGateSummary {
        status: if diagnostics.is_empty() {
            "pass"
        } else {
            "fail"
        },
        mode: "same_job_ab",
        rounds: reference.len(),
        bootstrap_resamples: AB_BOOTSTRAP_RESAMPLES,
        decision_rule: AB_DECISION_RULE,
        reference_git_commits: distinct(reference, |run| run.git_commit.clone()),
        candidate_git_commits: distinct(candidate, |run| run.git_commit.clone()),
        reference_environment_fingerprints: distinct(reference, |run| {
            run.environment_fingerprint.clone()
        }),
        candidate_environment_fingerprints: distinct(candidate, |run| {
            run.environment_fingerprint.clone()
        }),
        reference_cargo_profiles: distinct(reference, |run| {
            run.reproducibility.cargo_profile.clone()
        }),
        candidate_cargo_profiles: distinct(candidate, |run| {
            run.reproducibility.cargo_profile.clone()
        }),
        reference_runs: paths(&options.ab_reference_runs),
        candidate_runs: paths(&options.ab_candidate_runs),
        verdict_counts,
        workloads,
        uninstrumented_budget_paths,
        reliability: AbReliabilitySummary {
            coverage_floor: options.coverage_floor,
            coverage_ratio,
            max_median_regression_ratio: options.max_median_regression_ratio,
            missing_instrumentation_policy: MISSING_INSTRUMENTATION_POLICY,
            diagnostics,
        },
        report_path: options
            .report_path
            .as_ref()
            .map(|path| path.display().to_string()),
    })
}

/// The per-round medians of one workload in every run, or `None` when a run lacks it.
fn per_round_medians(runs: &[BenchmarkBaseline], name: &str) -> Option<Vec<f64>> {
    runs.iter()
        .map(|run| {
            run.workloads
                .iter()
                .find(|workload| workload.name == name)
                .filter(|workload| !workload.samples_ms.is_empty())
                .map(|workload| median(&workload.samples_ms))
        })
        .collect()
}

fn evaluate_ab_workload(
    budget: &WorkloadBudget,
    reference: &[BenchmarkBaseline],
    candidate: &[BenchmarkBaseline],
    max_median_regression_ratio: f64,
) -> Result<(AbWorkloadSummary, Vec<ReliabilityDiagnostic>), String> {
    let rounds = reference.len();
    let mut summary = AbWorkloadSummary {
        name: budget.name.to_string(),
        path_family: budget.path_family.to_string(),
        rounds,
        reference_median_ms: None,
        candidate_median_ms: None,
        candidate_p95_ms: None,
        effect_median_ratio: None,
        effect_ci95: None,
        reference_null_ci95: None,
        candidate_null_ci95: None,
        null_half_width: None,
        verdict: String::new(),
        status: "pass".to_string(),
        violations: Vec::new(),
    };
    let mut diagnostics = Vec::new();

    let Some(candidate_medians) = per_round_medians(candidate, budget.name) else {
        let violation = format!("workload '{}' missing from a candidate run", budget.name);
        diagnostics.push(ReliabilityDiagnostic::for_workload(
            budget,
            "missing_workload",
            violation.clone(),
            Vec::new(),
            vec![
                "workloads[].name".to_string(),
                "workloads[].samples_ms".to_string(),
            ],
            "keep every budgeted workload in generate_benchmark_baseline",
        ));
        summary.verdict = VERDICT_MISSING_CANDIDATE.to_string();
        summary.status = "fail".to_string();
        summary.violations.push(violation);
        return Ok((summary, diagnostics));
    };
    summary.candidate_median_ms = Some(median(&candidate_medians));

    // The absolute ceiling is an SLO on the candidate itself, over every sample it produced.
    let pooled: Vec<f64> = candidate
        .iter()
        .flat_map(|run| run.workloads.iter().filter(|w| w.name == budget.name))
        .flat_map(|workload| workload.samples_ms.iter().copied())
        .collect();
    let candidate_p95 = percentile(&pooled, 95);
    summary.candidate_p95_ms = Some(candidate_p95);
    if candidate_p95 > budget.p95_budget_ms {
        let violation = format!(
            "p95 {:.6}ms exceeded budget {:.6}ms",
            candidate_p95, budget.p95_budget_ms
        );
        diagnostics.push(ReliabilityDiagnostic::for_workload(
            budget,
            "p95_budget_exceeded",
            violation.clone(),
            Vec::new(),
            vec!["workloads[].samples_ms".to_string()],
            "profile the workload, optimize or update the explicit budget with evidence",
        ));
        summary.violations.push(violation);
    }

    let reference_medians = match per_round_medians(reference, budget.name) {
        Some(medians) => medians,
        None if reference
            .iter()
            .all(|run| run.workloads.iter().all(|w| w.name != budget.name)) =>
        {
            summary.verdict = VERDICT_NO_REFERENCE.to_string();
            if !summary.violations.is_empty() {
                summary.status = "fail".to_string();
            }
            return Ok((summary, diagnostics));
        }
        None => {
            return Err(format!(
                "workload '{}' is present in some reference runs and not others",
                budget.name
            ));
        }
    };
    summary.reference_median_ms = Some(median(&reference_medians));

    let effect: Vec<f64> = candidate_medians
        .iter()
        .zip(&reference_medians)
        .map(|(candidate, reference)| candidate / reference)
        .collect();
    let consecutive = |medians: &[f64]| {
        medians
            .windows(2)
            .map(|pair| pair[1] / pair[0])
            .collect::<Vec<_>>()
    };
    let seed = fnv1a(budget.name.as_bytes());
    let effect_median = median(&effect);
    let effect_ci = bootstrap_median_ci(&effect, seed);
    let reference_null = bootstrap_median_ci(&consecutive(&reference_medians), seed ^ 0x5eed_0001);
    let candidate_null = bootstrap_median_ci(&consecutive(&candidate_medians), seed ^ 0x5eed_0002);
    let half_width = |ci: [f64; 2]| (ci[0] - 1.0).abs().max((ci[1] - 1.0).abs());
    let null_half_width = half_width(reference_null).max(half_width(candidate_null));
    summary.effect_median_ratio = Some(effect_median);
    summary.effect_ci95 = Some(effect_ci);
    summary.reference_null_ci95 = Some(reference_null);
    summary.candidate_null_ci95 = Some(candidate_null);
    summary.null_half_width = Some(null_half_width);

    let verdict = if effect_ci[0] > 1.0 && effect_median - 1.0 > 2.0 * null_half_width {
        VERDICT_REGRESSION
    } else if effect_ci[1] < 1.0 && 1.0 - effect_median > 2.0 * null_half_width {
        VERDICT_WIN
    } else {
        VERDICT_UNDECIDED
    };
    summary.verdict = verdict.to_string();
    if verdict == VERDICT_REGRESSION && effect_median - 1.0 > max_median_regression_ratio {
        let violation = format!(
            "decidable median regression {:.6} (CI95 {:.6}..{:.6}, null half-width {:.6}) exceeded budget {:.6}",
            effect_median - 1.0,
            effect_ci[0],
            effect_ci[1],
            null_half_width,
            max_median_regression_ratio
        );
        diagnostics.push(ReliabilityDiagnostic::for_workload(
            budget,
            "median_regression_budget_exceeded",
            violation.clone(),
            Vec::new(),
            vec!["workloads[].samples_ms".to_string()],
            "profile the regression between the two commits and fix it, or change the budget with evidence",
        ));
        summary.violations.push(violation);
    }
    if !summary.violations.is_empty() {
        summary.status = "fail".to_string();
    }
    Ok((summary, diagnostics))
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.is_empty() {
        0.0
    } else if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    }
}

/// Nearest-rank percentile, as `benchmark::summarize_samples` computes it.
fn percentile(values: &[f64], percent: usize) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.is_empty() {
        return 0.0;
    }
    let last = sorted.len() - 1;
    sorted[(last * percent + 50) / 100]
}

/// Percentile-bootstrap 95% interval of the median, from a fixed seed so a report reproduces.
fn bootstrap_median_ci(values: &[f64], seed: u64) -> [f64; 2] {
    if values.is_empty() {
        return [f64::NAN, f64::NAN];
    }
    let mut state = seed;
    let mut resample = vec![0.0; values.len()];
    let mut medians = Vec::with_capacity(AB_BOOTSTRAP_RESAMPLES);
    for _ in 0..AB_BOOTSTRAP_RESAMPLES {
        for slot in &mut resample {
            *slot = values[(splitmix64(&mut state) % values.len() as u64) as usize];
        }
        medians.push(median(&resample));
    }
    medians.sort_by(f64::total_cmp);
    let at = |fraction: f64| medians[((medians.len() - 1) as f64 * fraction).round() as usize];
    [at(0.025), at(0.975)]
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn parse_args() -> Result<GateOptions, String> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let default_path = repo_root.join("artifacts/baselines/ufunc_benchmark_baseline.json");

    let mut reference_path: Option<PathBuf> = None;
    let mut candidate_path: Option<PathBuf> = None;
    let mut report_path: Option<PathBuf> = None;
    let mut max_p99_regression_ratio = 0.07f64;
    let mut coverage_floor = 1.0f64;
    let mut generate_candidate = false;
    let mut ab_reference_runs = Vec::new();
    let mut ab_candidate_runs = Vec::new();
    let mut max_median_regression_ratio = 0.07f64;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--ab-reference-run" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--ab-reference-run requires a value".to_string())?;
                ab_reference_runs.push(PathBuf::from(value));
            }
            "--ab-candidate-run" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--ab-candidate-run requires a value".to_string())?;
                ab_candidate_runs.push(PathBuf::from(value));
            }
            "--max-median-regression-ratio" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--max-median-regression-ratio requires a value".to_string())?;
                max_median_regression_ratio = value.parse::<f64>().map_err(|err| {
                    format!("invalid --max-median-regression-ratio value '{value}': {err}")
                })?;
                if max_median_regression_ratio < 0.0 {
                    return Err(format!(
                        "--max-median-regression-ratio must be >= 0.0, got {max_median_regression_ratio}"
                    ));
                }
            }
            "--reference-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--reference-path requires a value".to_string())?;
                reference_path = Some(PathBuf::from(value));
            }
            "--candidate-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--candidate-path requires a value".to_string())?;
                candidate_path = Some(PathBuf::from(value));
            }
            "--report-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--report-path requires a value".to_string())?;
                report_path = Some(PathBuf::from(value));
            }
            "--max-p99-regression-ratio" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--max-p99-regression-ratio requires a value".to_string())?;
                max_p99_regression_ratio = value.parse::<f64>().map_err(|err| {
                    format!("invalid --max-p99-regression-ratio value '{value}': {err}")
                })?;
                if max_p99_regression_ratio < 0.0 {
                    return Err(format!(
                        "--max-p99-regression-ratio must be >= 0.0, got {max_p99_regression_ratio}"
                    ));
                }
            }
            "--coverage-floor" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--coverage-floor requires a value".to_string())?;
                coverage_floor = value
                    .parse::<f64>()
                    .map_err(|err| format!("invalid --coverage-floor value '{value}': {err}"))?;
                if !(0.0..=1.0).contains(&coverage_floor) {
                    return Err(format!(
                        "--coverage-floor must be between 0.0 and 1.0, got {coverage_floor}"
                    ));
                }
            }
            "--generate-candidate" => {
                generate_candidate = true;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run -p fnp-conformance --bin run_performance_budget_gate -- [--reference-path <path>] [--candidate-path <path>] [--report-path <path>] [--max-p99-regression-ratio <ratio>] [--coverage-floor <ratio>] [--generate-candidate]\n\
                     Same-job A/B mode (CI's G7, scripts/e2e/run_performance_budget_gate.sh): --ab-reference-run <path> --ab-candidate-run <path> (one pair per round, >= {AB_MIN_ROUNDS} rounds) [--max-median-regression-ratio <ratio>] [--coverage-floor <ratio>] [--report-path <path>]"
                );
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }

    Ok(GateOptions {
        reference_path: reference_path.unwrap_or_else(|| default_path.clone()),
        candidate_path: candidate_path.unwrap_or(default_path),
        report_path,
        max_p99_regression_ratio,
        coverage_floor,
        generate_candidate,
        ab_reference_runs,
        ab_candidate_runs,
        max_median_regression_ratio,
    })
}

fn load_baseline(path: &Path) -> Result<BenchmarkBaseline, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed reading baseline {}: {err}", path.display()))?;
    serde_json::from_str::<BenchmarkBaseline>(&raw)
        .map_err(|err| format!("failed parsing baseline {}: {err}", path.display()))
}

fn evaluate_budget(
    budget: &WorkloadBudget,
    reference: Option<&BenchmarkWorkload>,
    candidate: Option<&BenchmarkWorkload>,
    max_p99_regression_ratio: f64,
    reference_path: &Path,
    candidate_path: &Path,
) -> (WorkloadDeltaSummary, Vec<ReliabilityDiagnostic>, bool) {
    let mut diagnostics = Vec::new();

    let (Some(reference_workload), Some(candidate_workload)) = (reference, candidate) else {
        let mut violations = Vec::new();
        if reference.is_none() {
            violations.push("missing_reference_workload".to_string());
        }
        if candidate.is_none() {
            violations.push("missing_candidate_workload".to_string());
        }

        diagnostics.push(ReliabilityDiagnostic::for_workload(
            budget,
            "missing_workload",
            format!(
                "workload '{}' missing in reference or candidate baseline",
                budget.name
            ),
            vec![
                reference_path.display().to_string(),
                candidate_path.display().to_string(),
            ],
            vec![
                "workloads[].name".to_string(),
                "workloads[].percentiles.p95_ms".to_string(),
                "workloads[].percentiles.p99_ms".to_string(),
                "workloads[].telemetry".to_string(),
            ],
            "regenerate the candidate baseline with generate_benchmark_baseline and keep every budgeted workload present",
        ));

        return (
            WorkloadDeltaSummary {
                name: budget.name.to_string(),
                path_family: budget.path_family.to_string(),
                reference_p95_ms: reference.map(|workload| workload.percentiles.p95_ms),
                candidate_p95_ms: candidate.map(|workload| workload.percentiles.p95_ms),
                reference_p99_ms: reference.map(|workload| workload.percentiles.p99_ms),
                candidate_p99_ms: candidate.map(|workload| workload.percentiles.p99_ms),
                p95_delta_percent: reference.zip(candidate).and_then(|(lhs, rhs)| {
                    percent_delta(lhs.percentiles.p95_ms, rhs.percentiles.p95_ms)
                }),
                p99_delta_percent: reference.zip(candidate).and_then(|(lhs, rhs)| {
                    percent_delta(lhs.percentiles.p99_ms, rhs.percentiles.p99_ms)
                }),
                status: "fail".to_string(),
                violations,
            },
            diagnostics,
            false,
        );
    };

    let reference_p95 = reference_workload.percentiles.p95_ms;
    let candidate_p95 = candidate_workload.percentiles.p95_ms;
    let reference_p99 = reference_workload.percentiles.p99_ms;
    let candidate_p99 = candidate_workload.percentiles.p99_ms;

    let mut violations = Vec::new();

    if candidate_p95 > budget.p95_budget_ms {
        let violation = format!(
            "p95 {:.6}ms exceeded budget {:.6}ms",
            candidate_p95, budget.p95_budget_ms
        );
        diagnostics.push(ReliabilityDiagnostic::for_workload(
            budget,
            "p95_budget_exceeded",
            violation.clone(),
            vec![candidate_path.display().to_string()],
            vec!["workloads[].percentiles.p95_ms".to_string()],
            "profile the workload, optimize or update the explicit budget with evidence",
        ));
        violations.push(violation);
    }

    match regression_ratio(reference_p99, candidate_p99) {
        Some(value) if value > max_p99_regression_ratio => {
            let violation = format!(
                "p99 regression ratio {:.6} exceeded budget {:.6}",
                value, max_p99_regression_ratio
            );
            diagnostics.push(ReliabilityDiagnostic::for_workload(
                budget,
                "p99_regression_budget_exceeded",
                violation.clone(),
                vec![
                    reference_path.display().to_string(),
                    candidate_path.display().to_string(),
                ],
                vec!["workloads[].percentiles.p99_ms".to_string()],
                "compare reference and candidate profiles, then fix the tail regression or provide an explicit budget change",
            ));
            violations.push(violation);
        }
        Some(_) => {}
        None => {
            let violation = "reference p99 must be > 0 to evaluate tail regression".to_string();
            diagnostics.push(ReliabilityDiagnostic::for_workload(
                budget,
                "invalid_reference_tail",
                violation.clone(),
                vec![reference_path.display().to_string()],
                vec!["workloads[].percentiles.p99_ms".to_string()],
                "regenerate the reference baseline so p99_ms is positive for every budgeted workload",
            ));
            violations.push(violation);
        }
    }

    let status = if violations.is_empty() {
        "pass"
    } else {
        "fail"
    };

    (
        WorkloadDeltaSummary {
            name: budget.name.to_string(),
            path_family: budget.path_family.to_string(),
            reference_p95_ms: Some(reference_p95),
            candidate_p95_ms: Some(candidate_p95),
            reference_p99_ms: Some(reference_p99),
            candidate_p99_ms: Some(candidate_p99),
            p95_delta_percent: percent_delta(reference_p95, candidate_p95),
            p99_delta_percent: percent_delta(reference_p99, candidate_p99),
            status: status.to_string(),
            violations,
        },
        diagnostics,
        true,
    )
}

fn missing_slo_paths(candidate: &BenchmarkBaseline) -> Vec<String> {
    let covered_paths: BTreeSet<&'static str> = candidate
        .workloads
        .iter()
        .flat_map(|workload| workload.telemetry.covered_slo_paths())
        .collect();
    REQUIRED_SLO_PATHS
        .iter()
        .filter(|path| !covered_paths.contains(**path))
        .map(|path| (*path).to_string())
        .collect()
}

fn missing_slo_path_diagnostics(
    paths: &[String],
    candidate_path: &Path,
) -> Vec<ReliabilityDiagnostic> {
    paths
        .iter()
        .map(|path| {
            let (expected_measurement_fields, remediation) = slo_path_instrumentation_policy(path);
            ReliabilityDiagnostic {
                subsystem: "performance_budget".to_string(),
                reason_code: "budget_path_uninstrumented".to_string(),
                message: format!(
                    "SLO path '{}' is not covered by generated benchmark workload telemetry; strict performance-budget gates fail closed for missing instrumentation",
                    path
                ),
                evidence_refs: vec![candidate_path.display().to_string()],
                workload_name: Some("candidate_baseline".to_string()),
                path_family: Some(path.clone()),
                expected_measurement_fields,
                remediation: Some(remediation),
            }
        })
        .collect()
}

fn slo_path_instrumentation_policy(path: &str) -> (Vec<String>, String) {
    match path {
        MEMORY_FOOTPRINT_SLO_PATH => (
            vec![
                "workloads[].telemetry.peak_live_bytes_per_run".to_string(),
                "workloads[].telemetry.process_high_water_rss_bytes".to_string(),
            ],
            "record positive peak live bytes for at least one generated candidate workload"
                .to_string(),
        ),
        ALLOCATION_CHURN_SLO_PATH => (
            vec!["workloads[].telemetry.heap_allocations_per_run".to_string()],
            "record positive heap allocation counts for at least one generated candidate workload"
                .to_string(),
        ),
        ALLOCATOR_FRAGMENTATION_SLO_PATH => (
            vec!["workloads[].telemetry.allocator_stress=adversarial".to_string()],
            "include an adversarial allocator-stress workload in the candidate baseline"
                .to_string(),
        ),
        _ => (
            vec!["workloads[].telemetry.covered_slo_paths()".to_string()],
            "extend WorkloadTelemetry::covered_slo_paths and generated baseline telemetry for this SLO path"
                .to_string(),
        ),
    }
}

fn regression_ratio(reference: f64, candidate: f64) -> Option<f64> {
    if reference <= 0.0 {
        return None;
    }
    Some((candidate - reference) / reference)
}

fn percent_delta(reference: f64, candidate: f64) -> Option<f64> {
    regression_ratio(reference, candidate).map(|value| value * 100.0)
}

#[cfg(test)]
mod tests {
    use super::{
        GateOptions, MISSING_INSTRUMENTATION_POLICY, WORKLOAD_BUDGETS, WorkloadBudget,
        evaluate_ab_gate, evaluate_budget, evaluate_gate, missing_slo_paths,
    };
    use fnp_conformance::benchmark::{
        ALLOCATION_CHURN_SLO_PATH, ALLOCATOR_FRAGMENTATION_SLO_PATH, AllocatorStressLevel,
        BenchmarkBaseline, BenchmarkWorkload, MEMORY_FOOTPRINT_SLO_PATH, PercentileSummary,
        ReproMetadata, WorkloadTelemetry,
    };
    use std::path::PathBuf;

    fn workload(name: &str, p95_ms: f64, p99_ms: f64) -> BenchmarkWorkload {
        BenchmarkWorkload {
            name: name.to_string(),
            runs: 5,
            samples_ms: vec![p95_ms, p99_ms],
            percentiles: PercentileSummary {
                p50_ms: p95_ms,
                p95_ms,
                p99_ms,
                min_ms: p95_ms,
                max_ms: p99_ms,
            },
            telemetry: WorkloadTelemetry::default(),
        }
    }

    fn baseline_with_telemetry(telemetry: WorkloadTelemetry) -> BenchmarkBaseline {
        BenchmarkBaseline {
            schema_version: 1,
            generated_at_unix_ms: 0,
            git_commit: "test".to_string(),
            workloads: vec![BenchmarkWorkload {
                name: "coverage".to_string(),
                runs: 1,
                samples_ms: vec![1.0],
                percentiles: PercentileSummary {
                    p50_ms: 1.0,
                    p95_ms: 1.0,
                    p99_ms: 1.0,
                    min_ms: 1.0,
                    max_ms: 1.0,
                },
                telemetry,
            }],
            environment_fingerprint: "test-env".to_string(),
            reproducibility: ReproMetadata::default(),
            evidence_log_refs: Vec::new(),
        }
    }

    fn budgeted_baseline_with_telemetry(telemetry: WorkloadTelemetry) -> BenchmarkBaseline {
        BenchmarkBaseline {
            schema_version: 1,
            generated_at_unix_ms: 0,
            git_commit: "test".to_string(),
            workloads: WORKLOAD_BUDGETS
                .iter()
                .map(|budget| BenchmarkWorkload {
                    name: budget.name.to_string(),
                    runs: 5,
                    samples_ms: vec![1.0, 1.1, 1.2],
                    percentiles: PercentileSummary {
                        p50_ms: 1.0,
                        p95_ms: budget.p95_budget_ms * 0.25,
                        p99_ms: 1.2,
                        min_ms: 1.0,
                        max_ms: 1.2,
                    },
                    telemetry: telemetry.clone(),
                })
                .collect(),
            environment_fingerprint: "test-env".to_string(),
            reproducibility: ReproMetadata::default(),
            evidence_log_refs: Vec::new(),
        }
    }

    fn fully_instrumented_telemetry() -> WorkloadTelemetry {
        WorkloadTelemetry {
            peak_live_bytes_per_run: 4096,
            process_high_water_rss_bytes: Some(8192),
            heap_allocations_per_run: 3,
            allocator_stress: AllocatorStressLevel::Adversarial,
            ..WorkloadTelemetry::default()
        }
    }

    fn gate_options() -> GateOptions {
        GateOptions {
            reference_path: PathBuf::from("reference.json"),
            candidate_path: PathBuf::from("candidate.json"),
            report_path: Some(PathBuf::from("report.json")),
            max_p99_regression_ratio: 0.07,
            coverage_floor: 1.0,
            generate_candidate: false,
            ab_reference_runs: Vec::new(),
            ab_candidate_runs: Vec::new(),
            max_median_regression_ratio: 0.07,
        }
    }

    /// One `generate_benchmark_baseline` run: every budgeted workload, five samples of
    /// `base x scale(round, name) x (1 +- jitter)`, base well under the absolute ceiling.
    fn ab_round(
        round: usize,
        scale: impl Fn(usize, &str) -> f64,
        jitter: f64,
    ) -> BenchmarkBaseline {
        let mut baseline = budgeted_baseline_with_telemetry(fully_instrumented_telemetry());
        for workload in &mut baseline.workloads {
            let budget = WORKLOAD_BUDGETS
                .iter()
                .find(|budget| budget.name == workload.name)
                .expect("budgeted workload");
            let base = budget.p95_budget_ms * 0.1 * scale(round, &workload.name);
            workload.samples_ms = (0..5)
                .map(|sample| {
                    // A fixed zig-zag, so every test reproduces.
                    let sign = if (round + sample).is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    };
                    base * (1.0 + sign * jitter * ((sample % 3) as f64) / 2.0)
                })
                .collect();
        }
        baseline
    }

    fn ab_rounds(
        rounds: usize,
        scale: impl Fn(usize, &str) -> f64 + Copy,
        jitter: f64,
    ) -> Vec<BenchmarkBaseline> {
        (0..rounds)
            .map(|round| ab_round(round, scale, jitter))
            .collect()
    }

    fn verdict_of<'a>(summary: &'a super::AbGateSummary, name: &str) -> &'a str {
        &summary
            .workloads
            .iter()
            .find(|workload| workload.name == name)
            .expect("workload in summary")
            .verdict
    }

    #[test]
    fn ab_gate_passes_an_a_a_run() {
        // Same code both arms, a few percent of round-to-round noise.
        let noise = |round: usize, _: &str| 1.0 + 0.03 * (((round * 7) % 5) as f64 - 2.0) / 2.0;
        let reference = ab_rounds(9, noise, 0.02);
        let candidate = ab_rounds(9, |round, name| noise(round + 3, name), 0.02);
        let summary = evaluate_ab_gate(&gate_options(), &reference, &candidate).expect("summary");
        assert_eq!(summary.status, "pass", "{summary:#?}");
        assert!(
            summary
                .workloads
                .iter()
                .all(|workload| workload.verdict != super::VERDICT_REGRESSION)
        );
    }

    #[test]
    fn ab_gate_fails_a_decidable_regression_beyond_budget() {
        let regressed = WORKLOAD_BUDGETS[2].name;
        let reference = ab_rounds(9, |_, _| 1.0, 0.01);
        let candidate = ab_rounds(9, |_, name| if name == regressed { 1.3 } else { 1.0 }, 0.01);
        let summary = evaluate_ab_gate(&gate_options(), &reference, &candidate).expect("summary");
        assert_eq!(summary.status, "fail");
        assert_eq!(verdict_of(&summary, regressed), super::VERDICT_REGRESSION);
        assert!(summary.reliability.diagnostics.iter().any(|diagnostic| {
            diagnostic.reason_code == "median_regression_budget_exceeded"
                && diagnostic.workload_name.as_deref() == Some(regressed)
        }));
    }

    #[test]
    fn ab_gate_passes_a_decidable_regression_within_budget() {
        let regressed = WORKLOAD_BUDGETS[0].name;
        let reference = ab_rounds(9, |_, _| 1.0, 0.001);
        let candidate = ab_rounds(
            9,
            |_, name| if name == regressed { 1.04 } else { 1.0 },
            0.001,
        );
        let summary = evaluate_ab_gate(&gate_options(), &reference, &candidate).expect("summary");
        assert_eq!(verdict_of(&summary, regressed), super::VERDICT_REGRESSION);
        assert_eq!(summary.status, "pass", "{summary:#?}");
    }

    #[test]
    fn ab_gate_does_not_fail_a_regression_its_nulls_cannot_resolve() {
        // A naive "median ratio above 1.07 fails" rule fails this; the arms' own round-to-round
        // spread (x1 / x2) is wider than the effect, so it is undecided.
        let regressed = WORKLOAD_BUDGETS[1].name;
        let swing = |round: usize| if round.is_multiple_of(2) { 1.0 } else { 2.0 };
        let reference = ab_rounds(9, |round, _| swing(round), 0.01);
        let candidate = ab_rounds(
            9,
            |round, name| swing(round + 1) * if name == regressed { 1.3 } else { 1.0 },
            0.01,
        );
        let summary = evaluate_ab_gate(&gate_options(), &reference, &candidate).expect("summary");
        assert_eq!(verdict_of(&summary, regressed), super::VERDICT_UNDECIDED);
        assert_eq!(summary.status, "pass", "{summary:#?}");
    }

    #[test]
    fn ab_gate_reports_a_decidable_win_without_failing() {
        let improved = WORKLOAD_BUDGETS[3].name;
        let reference = ab_rounds(9, |_, _| 1.0, 0.01);
        let candidate = ab_rounds(9, |_, name| if name == improved { 0.5 } else { 1.0 }, 0.01);
        let summary = evaluate_ab_gate(&gate_options(), &reference, &candidate).expect("summary");
        assert_eq!(verdict_of(&summary, improved), super::VERDICT_WIN);
        assert_eq!(summary.status, "pass");
    }

    #[test]
    fn ab_gate_fails_a_missing_candidate_workload_and_passes_a_new_one() {
        let dropped = WORKLOAD_BUDGETS[4].name;
        let added = WORKLOAD_BUDGETS[5].name;
        let mut reference = ab_rounds(3, |_, _| 1.0, 0.01);
        for run in &mut reference {
            run.workloads.retain(|workload| workload.name != added);
        }
        let mut candidate = ab_rounds(3, |_, _| 1.0, 0.01);
        let summary = evaluate_ab_gate(&gate_options(), &reference, &candidate).expect("summary");
        assert_eq!(verdict_of(&summary, added), super::VERDICT_NO_REFERENCE);
        assert_eq!(summary.status, "pass", "{summary:#?}");

        candidate[1]
            .workloads
            .retain(|workload| workload.name != dropped);
        let summary = evaluate_ab_gate(&gate_options(), &reference, &candidate).expect("summary");
        assert_eq!(
            verdict_of(&summary, dropped),
            super::VERDICT_MISSING_CANDIDATE
        );
        assert_eq!(summary.status, "fail");
    }

    #[test]
    fn ab_gate_refuses_unpaired_or_too_few_rounds() {
        let three = ab_rounds(3, |_, _| 1.0, 0.01);
        let two = ab_rounds(2, |_, _| 1.0, 0.01);
        assert!(evaluate_ab_gate(&gate_options(), &three, &two).is_err());
        assert!(evaluate_ab_gate(&gate_options(), &two, &two).is_err());
    }

    #[test]
    fn bootstrap_median_ci_is_reproducible_and_brackets_the_median() {
        let values = [0.98, 1.01, 1.02, 0.99, 1.03, 1.0, 1.05];
        let ci = super::bootstrap_median_ci(&values, 7);
        assert_eq!(ci, super::bootstrap_median_ci(&values, 7));
        let median = super::median(&values);
        assert!(ci[0] <= median && median <= ci[1], "{ci:?} vs {median}");
    }

    #[test]
    fn run_performance_budget_gate_passes_when_every_budget_path_is_instrumented() {
        let reference = budgeted_baseline_with_telemetry(fully_instrumented_telemetry());
        let candidate = budgeted_baseline_with_telemetry(fully_instrumented_telemetry());

        let summary = evaluate_gate(&gate_options(), reference, candidate).expect("gate summary");

        assert_eq!(summary.status, "pass");
        assert!(summary.uninstrumented_budget_paths.is_empty());
        assert_eq!(
            summary.reliability.missing_instrumentation_policy,
            MISSING_INSTRUMENTATION_POLICY
        );
        assert!(summary.reliability.diagnostics.is_empty());
        assert!(summary.reliability.warnings.is_empty());
    }

    #[test]
    fn run_performance_budget_gate_fails_when_candidate_omits_one_budgeted_workload() {
        let reference = budgeted_baseline_with_telemetry(fully_instrumented_telemetry());
        let mut candidate = budgeted_baseline_with_telemetry(fully_instrumented_telemetry());
        let omitted_workload = WORKLOAD_BUDGETS.first().expect("workload budgets").name;
        candidate
            .workloads
            .retain(|workload| workload.name != omitted_workload);

        let summary = evaluate_gate(&gate_options(), reference, candidate).expect("gate summary");

        assert_eq!(summary.status, "fail");
        let missing_summary = summary
            .workloads
            .iter()
            .find(|workload| workload.name == omitted_workload)
            .expect("missing workload summary");
        assert_eq!(missing_summary.status, "fail");
        assert!(
            missing_summary
                .violations
                .iter()
                .any(|violation| violation == "missing_candidate_workload")
        );
        let diagnostic = summary
            .reliability
            .diagnostics
            .iter()
            .find(|diagnostic| diagnostic.reason_code == "missing_workload")
            .expect("missing workload diagnostic");
        assert_eq!(
            diagnostic.workload_name.as_deref(),
            Some(WORKLOAD_BUDGETS[0].name)
        );
        assert_eq!(
            diagnostic.path_family.as_deref(),
            Some(WORKLOAD_BUDGETS[0].path_family)
        );
        assert!(
            diagnostic
                .expected_measurement_fields
                .iter()
                .any(|field| field == "workloads[].percentiles.p95_ms")
        );
        assert!(
            diagnostic
                .remediation
                .as_deref()
                .is_some_and(|message| message.contains("generate_benchmark_baseline"))
        );
    }

    #[test]
    fn run_performance_budget_gate_fails_closed_when_slo_path_is_uninstrumented() {
        let reference = budgeted_baseline_with_telemetry(fully_instrumented_telemetry());
        let candidate = budgeted_baseline_with_telemetry(WorkloadTelemetry {
            peak_live_bytes_per_run: 4096,
            heap_allocations_per_run: 3,
            allocator_stress: AllocatorStressLevel::Steady,
            ..WorkloadTelemetry::default()
        });

        let summary = evaluate_gate(&gate_options(), reference, candidate).expect("gate summary");

        assert_eq!(summary.status, "fail");
        assert_eq!(
            summary.uninstrumented_budget_paths,
            vec![ALLOCATOR_FRAGMENTATION_SLO_PATH.to_string()]
        );
        assert!(summary.reliability.warnings.is_empty());
        let diagnostic = summary
            .reliability
            .diagnostics
            .iter()
            .find(|diagnostic| diagnostic.reason_code == "budget_path_uninstrumented")
            .expect("missing instrumentation diagnostic");
        assert_eq!(
            diagnostic.path_family.as_deref(),
            Some(ALLOCATOR_FRAGMENTATION_SLO_PATH)
        );
        assert!(
            diagnostic
                .expected_measurement_fields
                .iter()
                .any(|field| field == "workloads[].telemetry.allocator_stress=adversarial")
        );
        assert!(
            diagnostic
                .remediation
                .as_deref()
                .is_some_and(|message| message.contains("adversarial allocator-stress"))
        );
    }

    #[test]
    fn workload_budget_passes_when_within_limits() {
        let budget = WorkloadBudget {
            name: "w",
            path_family: "broadcast",
            p95_budget_ms: 2.0,
        };
        let reference = workload("w", 1.0, 1.0);
        let candidate = workload("w", 1.5, 1.05);

        let (summary, diagnostics, covered) = evaluate_budget(
            &budget,
            Some(&reference),
            Some(&candidate),
            0.07,
            std::path::Path::new("reference.json"),
            std::path::Path::new("candidate.json"),
        );

        assert!(covered);
        assert_eq!(summary.status, "pass");
        assert!(summary.violations.is_empty());
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn workload_budget_fails_for_tail_regression() {
        let budget = WorkloadBudget {
            name: "w",
            path_family: "broadcast",
            p95_budget_ms: 2.0,
        };
        let reference = workload("w", 1.0, 1.0);
        let candidate = workload("w", 1.5, 1.2);

        let (summary, diagnostics, covered) = evaluate_budget(
            &budget,
            Some(&reference),
            Some(&candidate),
            0.07,
            std::path::Path::new("reference.json"),
            std::path::Path::new("candidate.json"),
        );

        assert!(covered);
        assert_eq!(summary.status, "fail");
        assert!(!summary.violations.is_empty());
        assert!(!diagnostics.is_empty());
    }

    #[test]
    fn workload_budget_flags_missing_workloads() {
        let budget = WorkloadBudget {
            name: "missing",
            path_family: "broadcast",
            p95_budget_ms: 2.0,
        };
        let reference = workload("w", 1.0, 1.0);

        let (summary, diagnostics, covered) = evaluate_budget(
            &budget,
            Some(&reference),
            None,
            0.07,
            std::path::Path::new("reference.json"),
            std::path::Path::new("candidate.json"),
        );

        assert!(!covered);
        assert_eq!(summary.status, "fail");
        assert!(summary.violations.iter().any(|v| v.contains("missing")));
        assert_eq!(diagnostics[0].reason_code, "missing_workload");
    }

    #[test]
    fn missing_slo_paths_are_derived_from_workload_telemetry() {
        let baseline = baseline_with_telemetry(WorkloadTelemetry {
            peak_live_bytes_per_run: 4096,
            heap_allocations_per_run: 3,
            allocator_stress: AllocatorStressLevel::Steady,
            ..WorkloadTelemetry::default()
        });

        let missing = missing_slo_paths(&baseline);

        assert_eq!(missing, vec![ALLOCATOR_FRAGMENTATION_SLO_PATH.to_string()]);
    }

    #[test]
    fn adversarial_allocator_workload_clears_all_slo_paths() {
        let baseline = baseline_with_telemetry(WorkloadTelemetry {
            peak_live_bytes_per_run: 4096,
            heap_allocations_per_run: 3,
            allocator_stress: AllocatorStressLevel::Adversarial,
            ..WorkloadTelemetry::default()
        });

        let missing = missing_slo_paths(&baseline);

        assert!(missing.is_empty());
    }

    #[test]
    fn build_baseline_types_for_bin_tests() {
        let _ = ReproMetadata::default();
        let _ = MEMORY_FOOTPRINT_SLO_PATH;
        let _ = ALLOCATION_CHURN_SLO_PATH;
    }
}
