#![forbid(unsafe_code)]

use fnp_conformance::benchmark::{BenchmarkBaseline, BenchmarkWorkload};
use serde::Serialize;
use std::collections::BTreeMap;
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
}

#[derive(Debug, Serialize)]
struct ReliabilitySummary {
    coverage_floor: f64,
    coverage_ratio: f64,
    max_p99_regression_ratio: f64,
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

const WORKLOAD_BUDGETS: &[WorkloadBudget] = &[
    WorkloadBudget {
        name: "ufunc_add_broadcast_256x256_by_256",
        path_family: "broadcast add/mul",
        p95_budget_ms: 180.0,
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
];

const UNINSTRUMENTED_BUDGET_PATHS: &[&str] = &[
    "reshape/view operations",
    "dtype conversion",
    "npy parse + load",
    "memory footprint",
    "allocation churn",
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
        diagnostics.push(ReliabilityDiagnostic {
            subsystem: "performance_budget".to_string(),
            reason_code: "coverage_floor_breach".to_string(),
            message: format!(
                "coverage ratio {:.6} is below floor {:.6}",
                coverage_ratio, options.coverage_floor
            ),
            evidence_refs: vec![
                options.reference_path.display().to_string(),
                options.candidate_path.display().to_string(),
            ],
        });
    }

    let warnings = UNINSTRUMENTED_BUDGET_PATHS
        .iter()
        .map(|path| ReliabilityDiagnostic {
            subsystem: "performance_budget".to_string(),
            reason_code: "budget_path_uninstrumented".to_string(),
            message: format!(
                "SLO path '{}' is not yet covered by generated benchmark workloads",
                path
            ),
            evidence_refs: vec![options.candidate_path.display().to_string()],
        })
        .collect::<Vec<_>>();

    let status = if diagnostics.is_empty() {
        "pass"
    } else {
        "fail"
    };
    let report_path = options
        .report_path
        .as_ref()
        .map(|path| path.display().to_string());

    let summary = GateSummary {
        status,
        reference_path: options.reference_path.display().to_string(),
        candidate_path: options.candidate_path.display().to_string(),
        reference_git_commit: reference.git_commit,
        candidate_git_commit: candidate.git_commit,
        reference_environment_fingerprint: reference.environment_fingerprint,
        candidate_environment_fingerprint: candidate.environment_fingerprint,
        workloads: workload_summaries,
        uninstrumented_budget_paths: UNINSTRUMENTED_BUDGET_PATHS
            .iter()
            .map(|path| (*path).to_string())
            .collect(),
        reliability: ReliabilitySummary {
            coverage_floor: options.coverage_floor,
            coverage_ratio,
            max_p99_regression_ratio: options.max_p99_regression_ratio,
            diagnostics,
            warnings,
        },
        report_path,
    };

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

        diagnostics.push(ReliabilityDiagnostic {
            subsystem: "performance_budget".to_string(),
            reason_code: "missing_workload".to_string(),
            message: format!(
                "workload '{}' missing in reference or candidate baseline",
                budget.name
            ),
            evidence_refs: vec![
                reference_path.display().to_string(),
                candidate_path.display().to_string(),
            ],
        });

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
        diagnostics.push(ReliabilityDiagnostic {
            subsystem: budget.name.to_string(),
            reason_code: "p95_budget_exceeded".to_string(),
            message: violation.clone(),
            evidence_refs: vec![candidate_path.display().to_string()],
        });
        violations.push(violation);
    }

    match regression_ratio(reference_p99, candidate_p99) {
        Some(value) if value > max_p99_regression_ratio => {
            let violation = format!(
                "p99 regression ratio {:.6} exceeded budget {:.6}",
                value, max_p99_regression_ratio
            );
            diagnostics.push(ReliabilityDiagnostic {
                subsystem: budget.name.to_string(),
                reason_code: "p99_regression_budget_exceeded".to_string(),
                message: violation.clone(),
                evidence_refs: vec![
                    reference_path.display().to_string(),
                    candidate_path.display().to_string(),
                ],
            });
            violations.push(violation);
        }
        Some(_) => {}
        None => {
            let violation = "reference p99 must be > 0 to evaluate tail regression".to_string();
            diagnostics.push(ReliabilityDiagnostic {
                subsystem: budget.name.to_string(),
                reason_code: "invalid_reference_tail".to_string(),
                message: violation.clone(),
                evidence_refs: vec![reference_path.display().to_string()],
            });
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
    use super::{WorkloadBudget, evaluate_budget};
    use fnp_conformance::benchmark::{
        BenchmarkWorkload, PercentileSummary, ReproMetadata, WorkloadTelemetry,
    };

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
    fn build_baseline_types_for_bin_tests() {
        let _ = ReproMetadata::default();
    }
}
