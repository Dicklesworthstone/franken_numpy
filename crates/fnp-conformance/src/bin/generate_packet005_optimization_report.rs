#![forbid(unsafe_code)]

use fnp_dtype::DType;
use fnp_ndarray::{broadcast_shape, element_count};
use fnp_ufunc::{BinaryOp, UFuncArray, UFuncError};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const PACKET_ID: &str = "FNP-P2C-005";
const SUBTASK_ID: &str = "FNP-P2C-005-H";

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
struct Packet005OptimizationReport {
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
        eprintln!("generate_packet005_optimization_report failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output_path = parse_output_path(&repo_root)?;

    let shape = vec![256usize, 256usize];
    let count = element_count(&shape).map_err(|err| format!("shape invalid: {err}"))?;
    let lhs_values = (0..count).map(|idx| idx as f64 * 0.25).collect::<Vec<_>>();
    let rhs_values = (0..count)
        .map(|idx| (count - idx) as f64 * 0.125)
        .collect::<Vec<_>>();
    let lhs = UFuncArray::new(shape.clone(), lhs_values, DType::F64)
        .map_err(|err| format!("failed creating lhs: {err}"))?;
    let rhs = UFuncArray::new(shape.clone(), rhs_values, DType::F64)
        .map_err(|err| format!("failed creating rhs: {err}"))?;

    let baseline_profile = profile_implementation("baseline_odometer_path", 60, count, || {
        legacy_elementwise_binary(&lhs, &rhs, BinaryOp::Add)
            .map_err(|err| format!("legacy run failed: {err}"))
    })?;
    let rebaseline_profile =
        profile_implementation("rebaseline_same_shape_fast_path", 60, count, || {
            lhs.elementwise_binary(&rhs, BinaryOp::Add)
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

    let report = Packet005OptimizationReport {
        schema_version: 1,
        packet_id: PACKET_ID.to_string(),
        subtask_id: SUBTASK_ID.to_string(),
        generated_at_unix_ms: now_unix_ms(),
        environment_fingerprint: environment_fingerprint(),
        reproducibility_command:
            "cargo run -p fnp-conformance --bin generate_packet005_optimization_report"
                .to_string(),
        lever: LeverSummary {
            id: "P2C005-H-LEVER-001".to_string(),
            description: "Add same-shape fast path for ufunc elementwise_binary".to_string(),
            rationale:
                "When shapes already match, broadcasting odometer bookkeeping is unnecessary; direct zipped iteration lowers per-element overhead."
                    .to_string(),
            risk_note:
                "Behavioral risk is low because fast path only triggers under exact shape equality and uses identical op/dtype semantics."
                    .to_string(),
            rollback_command: "git restore --source <last-green-commit> -- crates/fnp-ufunc/src/lib.rs"
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
                    "Usage: cargo run -p fnp-conformance --bin generate_packet005_optimization_report -- [--output-path <path>]"
                );
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }

    Ok(output_path.unwrap_or_else(|| {
        repo_root.join("artifacts/phase2c/FNP-P2C-005/optimization_profile_report.json")
    }))
}

fn profile_implementation<F>(
    implementation: &str,
    runs: usize,
    elements_per_run: usize,
    mut run_fn: F,
) -> Result<ProfileSummary, String>
where
    F: FnMut() -> Result<UFuncArray, String>,
{
    let mut samples_ms = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        let out = run_fn()?;
        std::hint::black_box(out.values().len());
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

fn legacy_elementwise_binary(
    lhs: &UFuncArray,
    rhs: &UFuncArray,
    op: BinaryOp,
) -> Result<UFuncArray, UFuncError> {
    let out_shape = broadcast_shape(lhs.shape(), rhs.shape()).map_err(UFuncError::Shape)?;
    let out_count = element_count(&out_shape).map_err(UFuncError::Shape)?;

    let lhs_strides = contiguous_strides_elems(lhs.shape());
    let rhs_strides = contiguous_strides_elems(rhs.shape());
    let lhs_axis_steps = aligned_broadcast_axis_steps(out_shape.len(), lhs.shape(), &lhs_strides);
    let rhs_axis_steps = aligned_broadcast_axis_steps(out_shape.len(), rhs.shape(), &rhs_strides);

    let mut out_multi = vec![0usize; out_shape.len()];
    let mut lhs_flat = 0usize;
    let mut rhs_flat = 0usize;
    let mut out_values = Vec::with_capacity(out_count);

    for flat in 0..out_count {
        out_values.push(op.apply(lhs.values()[lhs_flat], rhs.values()[rhs_flat]));

        if flat + 1 == out_count || out_shape.is_empty() {
            continue;
        }

        for axis in (0..out_shape.len()).rev() {
            out_multi[axis] += 1;
            lhs_flat += lhs_axis_steps[axis];
            rhs_flat += rhs_axis_steps[axis];

            if out_multi[axis] < out_shape[axis] {
                break;
            }

            out_multi[axis] = 0;
            lhs_flat -= lhs_axis_steps[axis] * out_shape[axis];
            rhs_flat -= rhs_axis_steps[axis] * out_shape[axis];
        }
    }

    UFuncArray::new(
        out_shape,
        out_values,
        fnp_dtype::promote(lhs.dtype(), rhs.dtype()),
    )
}

#[must_use]
fn contiguous_strides_elems(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut strides = vec![0usize; shape.len()];
    let mut stride = 1usize;
    for (idx, &dim) in shape.iter().enumerate().rev() {
        strides[idx] = stride;
        stride = stride.saturating_mul(dim);
    }
    strides
}

#[must_use]
fn aligned_broadcast_axis_steps(
    out_ndim: usize,
    src_shape: &[usize],
    src_strides: &[usize],
) -> Vec<usize> {
    if out_ndim == 0 {
        return Vec::new();
    }

    let mut axis_steps = vec![0usize; out_ndim];
    let offset = out_ndim - src_shape.len();
    for (axis, (&dim, &stride)) in src_shape.iter().zip(src_strides).enumerate() {
        axis_steps[axis + offset] = if dim == 1 { 0 } else { stride };
    }
    axis_steps
}

fn run_isomorphism_checks() -> Vec<IsomorphismCheck> {
    let same_lhs = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)
        .expect("same_lhs");
    let same_rhs = UFuncArray::new(
        vec![2, 3],
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        DType::F64,
    )
    .expect("same_rhs");
    let row_rhs = UFuncArray::new(vec![3], vec![0.5, -1.0, 2.0], DType::F64).expect("row_rhs");
    let scalar_rhs = UFuncArray::scalar(3.5, DType::F64);
    let bad_rhs = UFuncArray::new(vec![3, 1], vec![1.0, 2.0, 3.0], DType::F64).expect("bad_rhs");

    let cases = vec![
        (
            "case_same_shape_add",
            legacy_elementwise_binary(&same_lhs, &same_rhs, BinaryOp::Add),
            same_lhs.elementwise_binary(&same_rhs, BinaryOp::Add),
        ),
        (
            "case_broadcast_row",
            legacy_elementwise_binary(&same_lhs, &row_rhs, BinaryOp::Mul),
            same_lhs.elementwise_binary(&row_rhs, BinaryOp::Mul),
        ),
        (
            "case_scalar_broadcast",
            legacy_elementwise_binary(&same_lhs, &scalar_rhs, BinaryOp::Sub),
            same_lhs.elementwise_binary(&scalar_rhs, BinaryOp::Sub),
        ),
        (
            "case_shape_error",
            legacy_elementwise_binary(&same_lhs, &bad_rhs, BinaryOp::Add),
            same_lhs.elementwise_binary(&bad_rhs, BinaryOp::Add),
        ),
    ];

    cases
        .into_iter()
        .map(|(case_id, baseline, optimized)| {
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

fn canonicalize_result(result: &Result<UFuncArray, UFuncError>) -> String {
    match result {
        Ok(array) => format!(
            "ok:shape={:?}:dtype={:?}:values={:?}",
            array.shape(),
            array.dtype(),
            array.values()
        ),
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

fn compute_throughput(elements_per_run: f64, sample_ms: f64) -> f64 {
    if sample_ms <= 0.0 {
        return 0.0;
    }
    elements_per_run * 1000.0 / sample_ms
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
