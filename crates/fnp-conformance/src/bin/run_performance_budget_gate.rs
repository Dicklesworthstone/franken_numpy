#![forbid(unsafe_code)]

use fnp_conformance::benchmark::{
    ALLOCATION_CHURN_SLO_PATH, ALLOCATOR_FRAGMENTATION_SLO_PATH, BenchmarkBaseline,
    BenchmarkWorkload, MEMORY_FOOTPRINT_SLO_PATH, REQUIRED_SLO_PATHS,
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
}

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

fn parse_args() -> Result<GateOptions, String> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let default_path = repo_root.join("artifacts/baselines/ufunc_benchmark_baseline.json");

    let mut reference_path: Option<PathBuf> = None;
    let mut candidate_path: Option<PathBuf> = None;
    let mut report_path: Option<PathBuf> = None;
    let mut max_p99_regression_ratio = 0.07f64;
    let mut coverage_floor = 1.0f64;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
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
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run -p fnp-conformance --bin run_performance_budget_gate -- [--reference-path <path>] [--candidate-path <path>] [--report-path <path>] [--max-p99-regression-ratio <ratio>] [--coverage-floor <ratio>]"
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
        evaluate_budget, evaluate_gate, missing_slo_paths,
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
        }
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
