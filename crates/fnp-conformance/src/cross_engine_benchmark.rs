#![forbid(unsafe_code)]

use crate::raptorq_artifacts::generate_bundle_sidecar_and_reports;
use fnp_io::{IOSupportedDType, genfromtxt, load, save};
use fnp_random::Generator;
use fnp_ufunc::{BinaryOp, UFuncArray};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

pub const DEFAULT_CROSS_ENGINE_BENCHMARK_MANIFEST: &str =
    "artifacts/contracts/cross_engine_benchmark_workloads_v1.yaml";
pub const DEFAULT_CROSS_ENGINE_BENCHMARK_OUTPUT: &str =
    "artifacts/baselines/cross_engine_benchmark_v1.json";
pub const CROSS_ENGINE_RATIO_DIRECTION: &str = "fnp_p50 / numpy_p50 (>1 means FNP is slower)";
const QUICK_WARMUP_OVERRIDE: usize = 3;
const QUICK_SAMPLE_OVERRIDE: usize = 5;
const PYTHON_BENCHMARK_HARNESS: &str = r#"
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

def load_array(path_value):
    if not path_value:
        return None
    return np.load(path_value, allow_pickle=False)

def build_runner(request):
    spec = request["spec"]
    op = spec["operation"]
    lhs = load_array(request.get("lhs_path"))
    rhs = load_array(request.get("rhs_path"))
    axis = spec.get("axis")
    q = spec.get("percentile_q")
    shape = spec.get("shape", [])

    if op == "binary_add":
        return lambda: lhs + rhs
    if op == "binary_mul":
        return lambda: lhs * rhs
    if op == "binary_div":
        return lambda: lhs / rhs
    if op == "reduce_sum":
        return lambda: np.sum(lhs, axis=axis)
    if op == "reduce_mean":
        return lambda: np.mean(lhs, axis=axis)
    if op == "reduce_std":
        return lambda: np.std(lhs, axis=axis, ddof=0)
    if op == "reduce_argmax":
        return lambda: np.argmax(lhs, axis=axis)
    if op == "sort":
        return lambda: np.sort(lhs, axis=-1)
    if op == "percentile":
        return lambda: np.percentile(lhs, q=q, axis=axis)
    if op == "matmul":
        return lambda: lhs @ rhs
    if op == "solve":
        return lambda: np.linalg.solve(lhs, rhs)
    if op == "svd":
        return lambda: np.linalg.svd(lhs, full_matrices=True)
    if op == "eig":
        return lambda: np.linalg.eig(lhs)
    if op == "fft":
        return lambda: np.fft.fft(lhs)
    if op == "standard_normal":
        rng = np.random.Generator(np.random.PCG64DXSM(12345))
        return lambda: rng.standard_normal(shape[0])
    if op == "binomial":
        rng = np.random.Generator(np.random.PCG64DXSM(12345))
        return lambda: rng.binomial(spec["binomial_n"], spec["binomial_p"], shape[0])
    if op == "poisson":
        rng = np.random.Generator(np.random.PCG64DXSM(12345))
        return lambda: rng.poisson(spec["poisson_lambda"], shape[0])
    if op == "npy_roundtrip":
        def run_npy_roundtrip():
            buf = io.BytesIO()
            np.save(buf, lhs)
            buf.seek(0)
            return np.load(buf, allow_pickle=False)
        return run_npy_roundtrip
    if op == "gen_from_txt":
        text_path = request.get("text_path")
        if not text_path:
            raise ValueError("gen_from_txt requires text_path")
        return lambda: np.genfromtxt(text_path, delimiter=",")
    raise ValueError(f"unsupported operation: {op}")

def main():
    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    runner = build_runner(request)
    spec = request["spec"]
    warmup = spec["warmup"]
    samples = spec["samples"]

    for _ in range(warmup):
        runner()

    timings = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        runner()
        timings.append(time.perf_counter_ns() - start)

    print(json.dumps({"timings_ns": timings}))

if __name__ == "__main__":
    main()
"#;
const PYTHON_ENV_HARNESS: &str = r#"
import contextlib
import io
import json
import platform
import sys

import numpy as np

capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    np.__config__.show()
blas_backend = capture.getvalue().strip() or "unknown"
print(
    json.dumps(
        {
            "numpy_version": np.__version__,
            "blas_backend": blas_backend,
            "python_executable": sys.executable,
            "platform": platform.platform(),
        }
    )
)
"#;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossEngineBenchmarkManifest {
    pub version: String,
    pub workloads: Vec<WorkloadSpec>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkloadOperation {
    BinaryAdd,
    BinaryMul,
    BinaryDiv,
    ReduceSum,
    ReduceMean,
    ReduceStd,
    ReduceArgmax,
    Sort,
    Percentile,
    Matmul,
    Solve,
    Svd,
    Eig,
    Fft,
    StandardNormal,
    Binomial,
    Poisson,
    NpyRoundtrip,
    GenFromTxt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadSpec {
    pub family: String,
    pub name: String,
    pub size_tier: String,
    pub operation: WorkloadOperation,
    #[serde(default)]
    pub shape: Vec<usize>,
    #[serde(default)]
    pub rhs_shape: Vec<usize>,
    #[serde(default)]
    pub axis: Option<isize>,
    #[serde(default)]
    pub percentile_q: Option<f64>,
    pub warmup: usize,
    pub samples: usize,
    #[serde(default)]
    pub quick: bool,
    #[serde(default)]
    pub binomial_n: Option<u64>,
    #[serde(default)]
    pub binomial_p: Option<f64>,
    #[serde(default)]
    pub poisson_lambda: Option<f64>,
}

impl WorkloadSpec {
    fn effective_for_mode(&self, quick: bool) -> Self {
        let mut cloned = self.clone();
        if quick {
            cloned.warmup = QUICK_WARMUP_OVERRIDE;
            cloned.samples = QUICK_SAMPLE_OVERRIDE;
        }
        cloned
    }

    fn element_count(&self) -> Result<usize, String> {
        if self.shape.is_empty() {
            return Ok(0);
        }
        self.shape.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim)
                .ok_or_else(|| format!("{}: shape product overflow", self.name))
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingStats {
    pub p50_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
    pub mean_ns: f64,
    pub stddev_ns: f64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub sample_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentFingerprint {
    pub rust_version: String,
    pub numpy_version: String,
    pub blas_backend: String,
    pub cpu: String,
    pub os: String,
    pub python_executable: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossEngineWorkloadResult {
    pub family: String,
    pub name: String,
    pub size_tier: String,
    pub operation: WorkloadOperation,
    pub shape: Vec<usize>,
    pub rhs_shape: Vec<usize>,
    pub element_count: usize,
    pub warmup: usize,
    pub samples: usize,
    pub fnp: TimingStats,
    pub numpy: TimingStats,
    pub ratio: f64,
    pub ratio_direction: String,
    pub band: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossEngineBenchmarkReport {
    pub version: String,
    pub generated_at: String,
    pub git_sha: String,
    pub env_fingerprint: EnvironmentFingerprint,
    pub workloads: Vec<CrossEngineWorkloadResult>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PythonWorkloadRequest {
    spec: WorkloadSpec,
    lhs_path: Option<String>,
    rhs_path: Option<String>,
    text_path: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PythonTimingOutput {
    timings_ns: Vec<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PythonEnvironmentOutput {
    numpy_version: String,
    blas_backend: String,
    python_executable: String,
    platform: String,
}

#[derive(Debug, Clone, Default)]
struct BenchmarkFixtures {
    lhs_path: Option<PathBuf>,
    rhs_path: Option<PathBuf>,
    text_path: Option<PathBuf>,
}

pub fn parse_manifest(raw: &str) -> Result<CrossEngineBenchmarkManifest, String> {
    let manifest: CrossEngineBenchmarkManifest = serde_yaml_ng::from_str(raw)
        .map_err(|err| format!("invalid benchmark manifest yaml: {err}"))?;
    validate_manifest(&manifest)?;
    Ok(manifest)
}

pub fn load_manifest(path: &Path) -> Result<CrossEngineBenchmarkManifest, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed reading manifest {}: {err}", path.display()))?;
    parse_manifest(&raw)
}

pub fn run_cross_engine_benchmark(
    repo_root: &Path,
    manifest_path: &Path,
    output_path: &Path,
    quick: bool,
    oracle_python_override: Option<&str>,
) -> Result<CrossEngineBenchmarkReport, String> {
    let manifest = load_manifest(manifest_path)?;
    let oracle_python = resolve_oracle_python(repo_root, oracle_python_override)?;
    let env_fingerprint = collect_environment_fingerprint(&oracle_python)?;
    let selected_workloads = select_workloads(&manifest, quick);
    if selected_workloads.is_empty() {
        return Err("benchmark manifest selected zero workloads".to_string());
    }

    let run_stamp = sanitized_stamp(&iso8601_now());
    let fixture_root = repo_root
        .join("artifacts/logs/cross_engine_benchmark_inputs")
        .join(run_stamp);
    fs::create_dir_all(&fixture_root).map_err(|err| {
        format!(
            "failed creating benchmark fixture root {}: {err}",
            fixture_root.display()
        )
    })?;

    let mut results = Vec::with_capacity(selected_workloads.len());
    for spec in selected_workloads {
        let effective_spec = spec.effective_for_mode(quick);
        let workload_dir = fixture_root.join(sanitize_file_component(&effective_spec.name));
        let fixtures = generate_fixtures(&effective_spec, &workload_dir)?;
        let fnp_timings = run_fnp_workload(&effective_spec, &fixtures)?;
        let numpy_timings =
            run_numpy_workload(&oracle_python, &effective_spec, &fixtures, &workload_dir)?;
        let fnp_stats = compute_timing_stats(&fnp_timings)?;
        let numpy_stats = compute_timing_stats(&numpy_timings)?;
        let ratio = compute_ratio(fnp_stats.p50_ns, numpy_stats.p50_ns);
        results.push(CrossEngineWorkloadResult {
            family: effective_spec.family.clone(),
            name: effective_spec.name.clone(),
            size_tier: effective_spec.size_tier.clone(),
            operation: effective_spec.operation,
            shape: effective_spec.shape.clone(),
            rhs_shape: effective_spec.rhs_shape.clone(),
            element_count: effective_spec.element_count()?,
            warmup: effective_spec.warmup,
            samples: effective_spec.samples,
            fnp: fnp_stats,
            numpy: numpy_stats,
            ratio,
            ratio_direction: CROSS_ENGINE_RATIO_DIRECTION.to_string(),
            band: ratio_band(ratio).to_string(),
        });
    }

    let report = CrossEngineBenchmarkReport {
        version: manifest.version,
        generated_at: iso8601_now(),
        git_sha: git_sha(repo_root),
        env_fingerprint,
        workloads: results,
    };
    write_report_bundle(repo_root, manifest_path, output_path, &report)?;
    Ok(report)
}

pub fn generate_markdown_report(report: &CrossEngineBenchmarkReport) -> String {
    let mut lines = Vec::new();
    lines.push("# Cross-Engine Benchmark v1".to_string());
    lines.push(String::new());
    lines.push(format!("- Generated at: {}", report.generated_at));
    lines.push(format!("- Git SHA: {}", report.git_sha));
    lines.push(format!(
        "- NumPy: {} via {}",
        report.env_fingerprint.numpy_version, report.env_fingerprint.python_executable
    ));
    lines.push(format!(
        "- BLAS backend: {}",
        report.env_fingerprint.blas_backend
    ));
    lines.push(format!(
        "- Host: {} / {}",
        report.env_fingerprint.cpu, report.env_fingerprint.os
    ));
    lines.push(format!("- Total workloads: {}", report.workloads.len()));

    let ratios = report
        .workloads
        .iter()
        .map(|workload| workload.ratio)
        .collect::<Vec<_>>();
    let band_counts = count_bands(&report.workloads);
    lines.push(format!(
        "- Median ratio: {}",
        format_ratio(median_ratio(&ratios))
    ));
    lines.push(format!(
        "- Best ratio: {}",
        format_ratio(best_ratio(&ratios))
    ));
    lines.push(format!(
        "- Worst ratio: {}",
        format_ratio(worst_ratio(&ratios))
    ));
    lines.push(format!(
        "- Band counts: green={} yellow={} red={}",
        band_counts.0, band_counts.1, band_counts.2
    ));
    lines.push(String::new());

    let mut grouped = report.workloads.clone();
    grouped.sort_by(|lhs, rhs| {
        lhs.family
            .cmp(&rhs.family)
            .then(lhs.name.cmp(&rhs.name))
            .then(lhs.size_tier.cmp(&rhs.size_tier))
    });

    let mut current_family = String::new();
    for workload in grouped {
        if workload.family != current_family {
            current_family = workload.family.clone();
            lines.push(format!("## {}", current_family));
            lines.push(String::new());
            lines.push("| Workload | Size Tier | FNP p50 | NumPy p50 | Ratio | Band |".to_string());
            lines.push("| --- | --- | --- | --- | --- | --- |".to_string());
        }
        lines.push(format!(
            "| {} | {} | {} | {} | {} | {} |",
            workload.name,
            workload.size_tier,
            format_ns_as_ms(workload.fnp.p50_ns),
            format_ns_as_ms(workload.numpy.p50_ns),
            format_ratio(workload.ratio),
            workload.band
        ));
    }

    lines.push(String::new());
    lines.join("\n")
}

fn validate_manifest(manifest: &CrossEngineBenchmarkManifest) -> Result<(), String> {
    if manifest.version.trim().is_empty() {
        return Err("benchmark manifest version must not be empty".to_string());
    }
    if manifest.workloads.is_empty() {
        return Err("benchmark manifest must contain at least one workload".to_string());
    }
    for workload in &manifest.workloads {
        if workload.family.trim().is_empty() {
            return Err("benchmark workload family must not be empty".to_string());
        }
        if workload.name.trim().is_empty() {
            return Err("benchmark workload name must not be empty".to_string());
        }
        if workload.size_tier.trim().is_empty() {
            return Err(format!("{}: size_tier must not be empty", workload.name));
        }
        if workload.warmup == 0 || workload.samples == 0 {
            return Err(format!(
                "{}: warmup and samples must both be > 0",
                workload.name
            ));
        }
        validate_workload_shape_requirements(workload)?;
    }
    Ok(())
}

fn validate_workload_shape_requirements(workload: &WorkloadSpec) -> Result<(), String> {
    match workload.operation {
        WorkloadOperation::BinaryAdd
        | WorkloadOperation::BinaryMul
        | WorkloadOperation::BinaryDiv
        | WorkloadOperation::Matmul
        | WorkloadOperation::Solve => {
            if workload.shape.is_empty() || workload.rhs_shape.is_empty() {
                return Err(format!(
                    "{}: operation requires both shape and rhs_shape",
                    workload.name
                ));
            }
        }
        WorkloadOperation::ReduceSum
        | WorkloadOperation::ReduceMean
        | WorkloadOperation::ReduceStd
        | WorkloadOperation::ReduceArgmax
        | WorkloadOperation::Sort
        | WorkloadOperation::Percentile
        | WorkloadOperation::Svd
        | WorkloadOperation::Eig
        | WorkloadOperation::Fft
        | WorkloadOperation::StandardNormal
        | WorkloadOperation::Binomial
        | WorkloadOperation::Poisson
        | WorkloadOperation::NpyRoundtrip
        | WorkloadOperation::GenFromTxt => {
            if workload.shape.is_empty() {
                return Err(format!("{}: operation requires shape", workload.name));
            }
        }
    }
    if matches!(workload.operation, WorkloadOperation::Percentile)
        && workload.percentile_q.is_none()
    {
        return Err(format!(
            "{}: percentile workload requires percentile_q",
            workload.name
        ));
    }
    if matches!(workload.operation, WorkloadOperation::Binomial)
        && (workload.binomial_n.is_none() || workload.binomial_p.is_none())
    {
        return Err(format!(
            "{}: binomial workload requires binomial_n and binomial_p",
            workload.name
        ));
    }
    if matches!(workload.operation, WorkloadOperation::Poisson) && workload.poisson_lambda.is_none()
    {
        return Err(format!(
            "{}: poisson workload requires poisson_lambda",
            workload.name
        ));
    }
    Ok(())
}

fn select_workloads(manifest: &CrossEngineBenchmarkManifest, quick: bool) -> Vec<WorkloadSpec> {
    if !quick {
        return manifest.workloads.clone();
    }
    let quick_marked = manifest
        .workloads
        .iter()
        .filter(|workload| workload.quick)
        .cloned()
        .collect::<Vec<_>>();
    if quick_marked.is_empty() {
        manifest.workloads.iter().take(5).cloned().collect()
    } else {
        quick_marked
    }
}

fn run_fnp_workload(spec: &WorkloadSpec, fixtures: &BenchmarkFixtures) -> Result<Vec<u64>, String> {
    match spec.operation {
        WorkloadOperation::BinaryAdd => {
            let lhs = make_dense_array(&spec.shape)?;
            let rhs = make_dense_array(&spec.rhs_shape)?;
            time_operation(spec.warmup, spec.samples, || {
                black_box(
                    lhs.elementwise_binary(&rhs, BinaryOp::Add)
                        .map_err(to_string)?,
                );
                Ok(())
            })
        }
        WorkloadOperation::BinaryMul => {
            let lhs = make_dense_array(&spec.shape)?;
            let rhs = make_dense_array(&spec.rhs_shape)?;
            time_operation(spec.warmup, spec.samples, || {
                black_box(
                    lhs.elementwise_binary(&rhs, BinaryOp::Mul)
                        .map_err(to_string)?,
                );
                Ok(())
            })
        }
        WorkloadOperation::BinaryDiv => {
            let lhs = make_positive_array(&spec.shape)?;
            let rhs = make_positive_array(&spec.rhs_shape)?;
            time_operation(spec.warmup, spec.samples, || {
                black_box(
                    lhs.elementwise_binary(&rhs, BinaryOp::Div)
                        .map_err(to_string)?,
                );
                Ok(())
            })
        }
        WorkloadOperation::ReduceSum => {
            let lhs = make_dense_array(&spec.shape)?;
            time_operation(spec.warmup, spec.samples, || {
                black_box(lhs.reduce_sum(spec.axis, false).map_err(to_string)?);
                Ok(())
            })
        }
        WorkloadOperation::ReduceMean => {
            let lhs = make_dense_array(&spec.shape)?;
            time_operation(spec.warmup, spec.samples, || {
                black_box(lhs.reduce_mean(spec.axis, false).map_err(to_string)?);
                Ok(())
            })
        }
        WorkloadOperation::ReduceStd => {
            let lhs = make_dense_array(&spec.shape)?;
            time_operation(spec.warmup, spec.samples, || {
                black_box(lhs.reduce_std(spec.axis, false, 0).map_err(to_string)?);
                Ok(())
            })
        }
        WorkloadOperation::ReduceArgmax => {
            let lhs = make_dense_array(&spec.shape)?;
            time_operation(spec.warmup, spec.samples, || {
                black_box(lhs.reduce_argmax(spec.axis).map_err(to_string)?);
                Ok(())
            })
        }
        WorkloadOperation::Sort => {
            let lhs = make_dense_array(&spec.shape)?;
            time_operation(spec.warmup, spec.samples, || {
                black_box(lhs.sort(None, None).map_err(to_string)?);
                Ok(())
            })
        }
        WorkloadOperation::Percentile => {
            let lhs = make_dense_array(&spec.shape)?;
            let percentile_q = spec
                .percentile_q
                .ok_or_else(|| format!("{} missing percentile_q", spec.name))?;
            time_operation(spec.warmup, spec.samples, || {
                black_box(lhs.percentile(percentile_q, spec.axis).map_err(to_string)?);
                Ok(())
            })
        }
        WorkloadOperation::Matmul => {
            let lhs = make_dense_array(&spec.shape)?;
            let rhs = make_dense_array(&spec.rhs_shape)?;
            time_operation(spec.warmup, spec.samples, || {
                black_box(lhs.matmul(&rhs).map_err(to_string)?);
                Ok(())
            })
        }
        WorkloadOperation::Solve => {
            let lhs = make_solve_matrix_array(&spec.shape)?;
            let rhs = make_dense_array(&spec.rhs_shape)?;
            time_operation(spec.warmup, spec.samples, || {
                black_box(lhs.solve(&rhs).map_err(to_string)?);
                Ok(())
            })
        }
        WorkloadOperation::Svd => {
            let lhs = make_dense_array(&spec.shape)?;
            time_operation(spec.warmup, spec.samples, || {
                black_box(lhs.svd().map_err(to_string)?);
                Ok(())
            })
        }
        WorkloadOperation::Eig => {
            let lhs = make_symmetric_matrix_array(&spec.shape)?;
            time_operation(spec.warmup, spec.samples, || {
                black_box(lhs.eig().map_err(to_string)?);
                Ok(())
            })
        }
        WorkloadOperation::Fft => {
            let lhs = make_fft_array(&spec.shape)?;
            time_operation(spec.warmup, spec.samples, || {
                black_box(lhs.fft(None).map_err(to_string)?);
                Ok(())
            })
        }
        WorkloadOperation::StandardNormal => {
            let size = first_dim(spec)?;
            let mut rng = Generator::from_pcg64_dxsm(12345).map_err(to_string)?;
            time_operation(spec.warmup, spec.samples, || {
                black_box(rng.standard_normal(size));
                Ok(())
            })
        }
        WorkloadOperation::Binomial => {
            let size = first_dim(spec)?;
            let n = spec
                .binomial_n
                .ok_or_else(|| format!("{} missing binomial_n", spec.name))?;
            let p = spec
                .binomial_p
                .ok_or_else(|| format!("{} missing binomial_p", spec.name))?;
            let mut rng = Generator::from_pcg64_dxsm(12345).map_err(to_string)?;
            time_operation(spec.warmup, spec.samples, || {
                black_box(rng.binomial(n, p, size).map_err(to_string)?);
                Ok(())
            })
        }
        WorkloadOperation::Poisson => {
            let size = first_dim(spec)?;
            let lam = spec
                .poisson_lambda
                .ok_or_else(|| format!("{} missing poisson_lambda", spec.name))?;
            let mut rng = Generator::from_pcg64_dxsm(12345).map_err(to_string)?;
            time_operation(spec.warmup, spec.samples, || {
                black_box(rng.poisson(lam, size).map_err(to_string)?);
                Ok(())
            })
        }
        WorkloadOperation::NpyRoundtrip => {
            let values = make_dense_values(&spec.shape)?;
            time_operation(spec.warmup, spec.samples, || {
                let payload =
                    save(&spec.shape, &values, IOSupportedDType::F64).map_err(to_string)?;
                black_box(load(&payload).map_err(to_string)?);
                Ok(())
            })
        }
        WorkloadOperation::GenFromTxt => {
            let text_path = fixtures
                .text_path
                .as_ref()
                .ok_or_else(|| format!("{} missing text fixture", spec.name))?;
            time_operation(spec.warmup, spec.samples, || {
                let text = fs::read_to_string(text_path).map_err(to_string)?;
                black_box(genfromtxt(&text, ',', '#', 0, 0.0).map_err(to_string)?);
                Ok(())
            })
        }
    }
}

fn time_operation<F>(warmup: usize, samples: usize, mut op: F) -> Result<Vec<u64>, String>
where
    F: FnMut() -> Result<(), String>,
{
    for _ in 0..warmup {
        op()?;
    }

    let mut timings = Vec::with_capacity(samples);
    for _ in 0..samples {
        let started = Instant::now();
        op()?;
        let nanos = started.elapsed().as_nanos();
        let sample = u64::try_from(nanos).unwrap_or(u64::MAX);
        timings.push(sample);
    }
    Ok(timings)
}

fn run_numpy_workload(
    python: &str,
    spec: &WorkloadSpec,
    fixtures: &BenchmarkFixtures,
    workload_dir: &Path,
) -> Result<Vec<u64>, String> {
    let request_path = workload_dir.join("python_request.json");
    let request = PythonWorkloadRequest {
        spec: spec.clone(),
        lhs_path: fixtures
            .lhs_path
            .as_ref()
            .map(|path| path.display().to_string()),
        rhs_path: fixtures
            .rhs_path
            .as_ref()
            .map(|path| path.display().to_string()),
        text_path: fixtures
            .text_path
            .as_ref()
            .map(|path| path.display().to_string()),
    };
    let request_raw = serde_json::to_string_pretty(&request)
        .map_err(|err| format!("failed serializing python workload request: {err}"))?;
    fs::write(&request_path, request_raw).map_err(|err| {
        format!(
            "failed writing python workload request {}: {err}",
            request_path.display()
        )
    })?;

    let output = Command::new(python)
        .arg("-c")
        .arg(PYTHON_BENCHMARK_HARNESS)
        .arg(&request_path)
        .output()
        .map_err(|err| {
            format!(
                "failed invoking NumPy benchmark harness via {}: {err}",
                python
            )
        })?;
    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "NumPy benchmark harness failed for {} status={} stdout={} stderr={}",
            spec.name,
            output.status,
            stdout.trim(),
            stderr.trim()
        ));
    }

    let parsed: PythonTimingOutput = serde_json::from_slice(&output.stdout).map_err(|err| {
        format!(
            "invalid NumPy benchmark harness output for {}: {err}",
            spec.name
        )
    })?;
    Ok(parsed.timings_ns)
}

fn generate_fixtures(
    spec: &WorkloadSpec,
    workload_dir: &Path,
) -> Result<BenchmarkFixtures, String> {
    fs::create_dir_all(workload_dir).map_err(|err| {
        format!(
            "failed creating workload dir {}: {err}",
            workload_dir.display()
        )
    })?;
    let mut fixtures = BenchmarkFixtures::default();

    match spec.operation {
        WorkloadOperation::BinaryAdd
        | WorkloadOperation::BinaryMul
        | WorkloadOperation::BinaryDiv
        | WorkloadOperation::Matmul
        | WorkloadOperation::Solve => {
            let lhs_path = workload_dir.join("lhs.npy");
            let rhs_path = workload_dir.join("rhs.npy");
            let lhs_values = values_for_lhs(spec)?;
            let rhs_values = values_for_rhs(spec)?;
            write_npy_fixture(&lhs_path, &spec.shape, &lhs_values)?;
            write_npy_fixture(&rhs_path, &spec.rhs_shape, &rhs_values)?;
            fixtures.lhs_path = Some(lhs_path);
            fixtures.rhs_path = Some(rhs_path);
        }
        WorkloadOperation::ReduceSum
        | WorkloadOperation::ReduceMean
        | WorkloadOperation::ReduceStd
        | WorkloadOperation::ReduceArgmax
        | WorkloadOperation::Sort
        | WorkloadOperation::Percentile
        | WorkloadOperation::Svd
        | WorkloadOperation::Eig
        | WorkloadOperation::Fft
        | WorkloadOperation::NpyRoundtrip => {
            let lhs_path = workload_dir.join("lhs.npy");
            write_npy_fixture(&lhs_path, &spec.shape, &values_for_lhs(spec)?)?;
            fixtures.lhs_path = Some(lhs_path);
        }
        WorkloadOperation::GenFromTxt => {
            let text_path = workload_dir.join("input.csv");
            let text = make_csv_fixture(spec)?;
            fs::write(&text_path, text).map_err(|err| {
                format!("failed writing text fixture {}: {err}", text_path.display())
            })?;
            fixtures.text_path = Some(text_path);
        }
        WorkloadOperation::StandardNormal
        | WorkloadOperation::Binomial
        | WorkloadOperation::Poisson => {}
    }
    Ok(fixtures)
}

fn values_for_lhs(spec: &WorkloadSpec) -> Result<Vec<f64>, String> {
    match spec.operation {
        WorkloadOperation::BinaryDiv => make_positive_values(&spec.shape),
        WorkloadOperation::Solve => make_solve_matrix(&spec.shape),
        WorkloadOperation::Eig => make_symmetric_matrix(&spec.shape),
        WorkloadOperation::Fft => make_fft_values(&spec.shape),
        _ => make_dense_values(&spec.shape),
    }
}

fn values_for_rhs(spec: &WorkloadSpec) -> Result<Vec<f64>, String> {
    match spec.operation {
        WorkloadOperation::BinaryDiv => make_positive_values(&spec.rhs_shape),
        WorkloadOperation::Solve => make_dense_values(&spec.rhs_shape),
        _ => make_dense_values(&spec.rhs_shape),
    }
}

fn write_npy_fixture(path: &Path, shape: &[usize], values: &[f64]) -> Result<(), String> {
    let payload = save(shape, values, IOSupportedDType::F64).map_err(to_string)?;
    fs::write(path, payload)
        .map_err(|err| format!("failed writing npy fixture {}: {err}", path.display()))
}

fn collect_environment_fingerprint(python: &str) -> Result<EnvironmentFingerprint, String> {
    let output = Command::new(python)
        .arg("-c")
        .arg(PYTHON_ENV_HARNESS)
        .output()
        .map_err(|err| {
            format!(
                "failed invoking NumPy environment harness via {}: {err}",
                python
            )
        })?;
    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "NumPy environment harness failed status={} stdout={} stderr={}",
            output.status,
            stdout.trim(),
            stderr.trim()
        ));
    }
    let python_env: PythonEnvironmentOutput = serde_json::from_slice(&output.stdout)
        .map_err(|err| format!("invalid NumPy environment harness output: {err}"))?;
    Ok(EnvironmentFingerprint {
        rust_version: rustc_version(),
        numpy_version: python_env.numpy_version,
        blas_backend: python_env.blas_backend,
        cpu: cpu_fingerprint(),
        os: python_env.platform,
        python_executable: python_env.python_executable,
    })
}

fn write_report_bundle(
    repo_root: &Path,
    manifest_path: &Path,
    output_path: &Path,
    report: &CrossEngineBenchmarkReport,
) -> Result<(), String> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }
    let report_raw = serde_json::to_string_pretty(report)
        .map_err(|err| format!("failed serializing cross-engine benchmark report: {err}"))?;
    fs::write(output_path, report_raw)
        .map_err(|err| format!("failed writing {}: {err}", output_path.display()))?;

    let markdown_path = sibling_path(output_path, ".report.md")?;
    let markdown = generate_markdown_report(report);
    fs::write(&markdown_path, markdown)
        .map_err(|err| format!("failed writing {}: {err}", markdown_path.display()))?;

    let sidecar_path = sibling_path(output_path, ".raptorq.json")?;
    let scrub_path = sibling_path(output_path, ".scrub_report.json")?;
    let decode_path = sibling_path(output_path, ".decode_proof.json")?;
    generate_bundle_sidecar_and_reports(
        "cross_engine_benchmark_v1",
        repo_root,
        &[
            output_path.to_path_buf(),
            markdown_path,
            manifest_path.to_path_buf(),
        ],
        &sidecar_path,
        &scrub_path,
        &decode_path,
        0xC0DE_5E11_u64,
    )
}

fn sibling_path(path: &Path, suffix: &str) -> Result<PathBuf, String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", path.display()))?;
    let stem = path
        .file_stem()
        .ok_or_else(|| format!("{} has no file stem", path.display()))?
        .to_string_lossy();
    Ok(parent.join(format!("{stem}{suffix}")))
}

fn compute_timing_stats(samples: &[u64]) -> Result<TimingStats, String> {
    if samples.is_empty() {
        return Err("cannot compute timing statistics for empty sample set".to_string());
    }

    let mut sorted = samples.to_vec();
    sorted.sort_unstable();

    let sample_count = sorted.len();
    let sum = sorted.iter().map(|&value| value as f64).sum::<f64>();
    let mean_ns = sum / sample_count as f64;
    let variance = sorted
        .iter()
        .map(|&value| {
            let delta = value as f64 - mean_ns;
            delta * delta
        })
        .sum::<f64>()
        / sample_count as f64;

    Ok(TimingStats {
        p50_ns: sorted[percentile_index(sample_count, 50)],
        p95_ns: sorted[percentile_index(sample_count, 95)],
        p99_ns: sorted[percentile_index(sample_count, 99)],
        mean_ns,
        stddev_ns: variance.sqrt(),
        min_ns: *sorted.first().unwrap_or(&0),
        max_ns: *sorted.last().unwrap_or(&0),
        sample_count,
    })
}

fn percentile_index(len: usize, percentile: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    (((len * percentile) + 99) / 100)
        .saturating_sub(1)
        .min(len - 1)
}

fn compute_ratio(fnp_p50_ns: u64, numpy_p50_ns: u64) -> f64 {
    if numpy_p50_ns == 0 {
        return f64::INFINITY;
    }
    fnp_p50_ns as f64 / numpy_p50_ns as f64
}

fn ratio_band(ratio: f64) -> &'static str {
    if !ratio.is_finite() || ratio > 10.0 {
        "red"
    } else if ratio > 2.0 {
        "yellow"
    } else {
        "green"
    }
}

fn format_ratio(ratio: f64) -> String {
    if ratio.is_finite() {
        format!("{ratio:.2}x")
    } else {
        "inf".to_string()
    }
}

fn format_ns_as_ms(value_ns: u64) -> String {
    format!("{:.3} ms", value_ns as f64 / 1_000_000.0)
}

fn median_ratio(ratios: &[f64]) -> f64 {
    let mut sorted = finite_ratios(ratios);
    if sorted.is_empty() {
        return f64::INFINITY;
    }
    sorted.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal));
    sorted[sorted.len() / 2]
}

fn best_ratio(ratios: &[f64]) -> f64 {
    finite_ratios(ratios)
        .into_iter()
        .min_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal))
        .unwrap_or(f64::INFINITY)
}

fn worst_ratio(ratios: &[f64]) -> f64 {
    finite_ratios(ratios)
        .into_iter()
        .max_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal))
        .unwrap_or(f64::INFINITY)
}

fn finite_ratios(ratios: &[f64]) -> Vec<f64> {
    ratios
        .iter()
        .copied()
        .filter(|ratio| ratio.is_finite())
        .collect()
}

fn count_bands(workloads: &[CrossEngineWorkloadResult]) -> (usize, usize, usize) {
    let mut green = 0usize;
    let mut yellow = 0usize;
    let mut red = 0usize;
    for workload in workloads {
        match workload.band.as_str() {
            "green" => green += 1,
            "yellow" => yellow += 1,
            _ => red += 1,
        }
    }
    (green, yellow, red)
}

fn make_dense_array(shape: &[usize]) -> Result<UFuncArray, String> {
    UFuncArray::new(
        shape.to_vec(),
        make_dense_values(shape)?,
        fnp_dtype::DType::F64,
    )
    .map_err(to_string)
}

fn make_positive_array(shape: &[usize]) -> Result<UFuncArray, String> {
    UFuncArray::new(
        shape.to_vec(),
        make_positive_values(shape)?,
        fnp_dtype::DType::F64,
    )
    .map_err(to_string)
}

fn make_solve_matrix_array(shape: &[usize]) -> Result<UFuncArray, String> {
    UFuncArray::new(
        shape.to_vec(),
        make_solve_matrix(shape)?,
        fnp_dtype::DType::F64,
    )
    .map_err(to_string)
}

fn make_symmetric_matrix_array(shape: &[usize]) -> Result<UFuncArray, String> {
    UFuncArray::new(
        shape.to_vec(),
        make_symmetric_matrix(shape)?,
        fnp_dtype::DType::F64,
    )
    .map_err(to_string)
}

fn make_fft_array(shape: &[usize]) -> Result<UFuncArray, String> {
    UFuncArray::new(
        shape.to_vec(),
        make_fft_values(shape)?,
        fnp_dtype::DType::F64,
    )
    .map_err(to_string)
}

fn make_dense_values(shape: &[usize]) -> Result<Vec<f64>, String> {
    let len = element_count(shape)?;
    Ok((0..len)
        .map(|index| ((index % 257) as f64 * 0.25) + ((index / 257 % 17) as f64 * 0.5))
        .collect())
}

fn make_positive_values(shape: &[usize]) -> Result<Vec<f64>, String> {
    let len = element_count(shape)?;
    Ok((0..len)
        .map(|index| 1.0 + (index % 251) as f64 * 0.125)
        .collect())
}

fn make_solve_matrix(shape: &[usize]) -> Result<Vec<f64>, String> {
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err("solve workloads require a square matrix shape".to_string());
    }
    let n = shape[0];
    let mut values = vec![0.0; n * n];
    for row in 0..n {
        for col in 0..n {
            values[row * n + col] = if row == col {
                (n as f64) + 5.0 + row as f64 * 0.5
            } else {
                ((row + col + 1) % 11) as f64 * 0.05
            };
        }
    }
    Ok(values)
}

fn make_symmetric_matrix(shape: &[usize]) -> Result<Vec<f64>, String> {
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err("eig workloads require a square matrix shape".to_string());
    }
    let n = shape[0];
    let mut values = vec![0.0; n * n];
    for row in 0..n {
        for col in row..n {
            let base = if row == col {
                10.0 + row as f64 * 0.25
            } else {
                ((row + col + 3) % 19) as f64 * 0.1
            };
            values[row * n + col] = base;
            values[col * n + row] = base;
        }
    }
    Ok(values)
}

fn make_fft_values(shape: &[usize]) -> Result<Vec<f64>, String> {
    let len = first_shape_dim(shape)?;
    Ok((0..len)
        .map(|index| ((index as f64) * 0.013).sin() + ((index as f64) * 0.007).cos() * 0.5)
        .collect())
}

fn make_csv_fixture(spec: &WorkloadSpec) -> Result<String, String> {
    if spec.shape.len() != 2 {
        return Err(format!("{}: gen_from_txt requires a 2-D shape", spec.name));
    }
    let rows = spec.shape[0];
    let cols = spec.shape[1];
    let mut lines = Vec::with_capacity(rows + 1);
    lines.push(
        (0..cols)
            .map(|index| format!("c{index}"))
            .collect::<Vec<_>>()
            .join(","),
    );
    for row in 0..rows {
        let mut values = Vec::with_capacity(cols);
        for col in 0..cols {
            let value = (row * cols + col) as f64 * 0.125 + col as f64 * 0.5;
            values.push(format!("{value:.6}"));
        }
        lines.push(values.join(","));
    }
    Ok(lines.join("\n"))
}

fn element_count(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| "shape element_count overflow".to_string())
    })
}

fn first_shape_dim(shape: &[usize]) -> Result<usize, String> {
    shape
        .first()
        .copied()
        .ok_or_else(|| "shape must not be empty".to_string())
}

fn first_dim(spec: &WorkloadSpec) -> Result<usize, String> {
    first_shape_dim(&spec.shape).map_err(|_| format!("{}: shape must not be empty", spec.name))
}

fn git_sha(repo_root: &Path) -> String {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo_root)
        .output();
    match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        _ => "unknown".to_string(),
    }
}

fn rustc_version() -> String {
    let output = Command::new("rustc").arg("--version").output();
    match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        _ => "unknown".to_string(),
    }
}

fn cpu_fingerprint() -> String {
    if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo")
        && let Some(line) = cpuinfo
            .lines()
            .find(|line| line.starts_with("model name"))
            .and_then(|line| line.split(':').nth(1))
    {
        return line.trim().to_string();
    }
    std::env::consts::ARCH.to_string()
}

fn iso8601_now() -> String {
    let output = Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output();
    match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        _ => "1970-01-01T00:00:00Z".to_string(),
    }
}

fn sanitized_stamp(input: &str) -> String {
    input
        .chars()
        .map(|ch| match ch {
            '0'..='9' | 'A'..='Z' | 'a'..='z' => ch,
            _ => '_',
        })
        .collect()
}

fn sanitize_file_component(input: &str) -> String {
    let sanitized = input
        .chars()
        .map(|ch| match ch {
            '0'..='9' | 'A'..='Z' | 'a'..='z' | '-' | '_' => ch,
            _ => '_',
        })
        .collect::<String>();
    if sanitized.is_empty() {
        "workload".to_string()
    } else {
        sanitized
    }
}

fn require_real_numpy_oracle() -> bool {
    match std::env::var("FNP_REQUIRE_REAL_NUMPY_ORACLE") {
        Ok(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        }
        Err(_) => false,
    }
}

fn configured_oracle_python(override_python: Option<&str>) -> Option<String> {
    override_python
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            std::env::var("FNP_ORACLE_PYTHON")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
}

fn is_default_python_selector(value: &str) -> bool {
    matches!(value.trim(), "python3" | "python")
}

fn install_numpy_into_user_interpreter(python: &str) -> Result<String, String> {
    let install = Command::new(python)
        .arg("-m")
        .arg("pip")
        .arg("install")
        .arg("--user")
        .arg("--break-system-packages")
        .arg("numpy")
        .output()
        .map_err(|err| format!("failed to install numpy into user interpreter via pip: {err}"))?;
    if !install.status.success() {
        let stdout = String::from_utf8_lossy(&install.stdout);
        let stderr = String::from_utf8_lossy(&install.stderr);
        return Err(format!(
            "failed to install numpy into user interpreter via pip (stdout={} stderr={})",
            stdout.trim(),
            stderr.trim()
        ));
    }
    Ok(python.to_string())
}

fn bootstrap_repo_numpy_venv(python_path: &Path, bootstrap_python: &str) -> Result<String, String> {
    let venv_dir = python_path
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| format!("invalid oracle venv path {}", python_path.display()))?;

    if let Ok(uv_check) = Command::new("uv").arg("--version").output()
        && uv_check.status.success()
    {
        let create = Command::new("uv")
            .args(["venv", "--python", "3.14"])
            .arg(venv_dir)
            .output()
            .map_err(|err| format!("failed to bootstrap oracle venv via uv venv: {err}"))?;
        if !create.status.success() {
            let stdout = String::from_utf8_lossy(&create.stdout);
            let stderr = String::from_utf8_lossy(&create.stderr);
            return Err(format!(
                "failed to bootstrap oracle venv via uv venv (stdout={} stderr={})",
                stdout.trim(),
                stderr.trim()
            ));
        }

        let install = Command::new("uv")
            .arg("pip")
            .arg("install")
            .arg("--python")
            .arg(python_path)
            .arg("numpy")
            .output()
            .map_err(|err| format!("failed to install numpy into oracle venv via uv pip: {err}"))?;
        if !install.status.success() {
            let stdout = String::from_utf8_lossy(&install.stdout);
            let stderr = String::from_utf8_lossy(&install.stderr);
            return Err(format!(
                "failed to install numpy into oracle venv via uv pip (stdout={} stderr={})",
                stdout.trim(),
                stderr.trim()
            ));
        }

        return Ok(python_path.display().to_string());
    }

    let create = Command::new(bootstrap_python)
        .arg("-m")
        .arg("venv")
        .arg(venv_dir)
        .output()
        .map_err(|err| {
            format!("failed to bootstrap oracle venv via `{bootstrap_python} -m venv`: {err}")
        })?;
    if !create.status.success() {
        let stdout = String::from_utf8_lossy(&create.stdout);
        let stderr = String::from_utf8_lossy(&create.stderr);
        return install_numpy_into_user_interpreter(bootstrap_python).map_err(|pip_err| {
            format!(
                "failed to bootstrap oracle venv via `{bootstrap_python} -m venv` (stdout={} stderr={}); fallback user-site install also failed: {}",
                stdout.trim(),
                stderr.trim(),
                pip_err
            )
        });
    }

    let install = Command::new(python_path)
        .arg("-m")
        .arg("pip")
        .arg("install")
        .arg("numpy")
        .output()
        .map_err(|err| format!("failed to install numpy into oracle venv via pip: {err}"))?;
    if !install.status.success() {
        let stdout = String::from_utf8_lossy(&install.stdout);
        let stderr = String::from_utf8_lossy(&install.stderr);
        return install_numpy_into_user_interpreter(bootstrap_python).map_err(|pip_err| {
            format!(
                "failed to install numpy into oracle venv via pip (stdout={} stderr={}); fallback user-site install also failed: {}",
                stdout.trim(),
                stderr.trim(),
                pip_err
            )
        });
    }

    Ok(python_path.display().to_string())
}

fn resolve_oracle_python(
    repo_root: &Path,
    override_python: Option<&str>,
) -> Result<String, String> {
    let configured = configured_oracle_python(override_python);
    let require_real = require_real_numpy_oracle();
    let repo_python = repo_root.join(".venv-numpy314/bin/python3");

    if let Some(configured_value) = configured.as_deref()
        && !(require_real && is_default_python_selector(configured_value))
    {
        return Ok(configured_value.to_string());
    }

    if repo_python.is_file() {
        return Ok(repo_python.display().to_string());
    }

    if require_real {
        let bootstrap_python = configured.as_deref().unwrap_or("python3");
        return bootstrap_repo_numpy_venv(&repo_python, bootstrap_python);
    }

    Ok(configured.unwrap_or_else(|| "python3".to_string()))
}

fn to_string(err: impl std::fmt::Display) -> String {
    err.to_string()
}

#[cfg(test)]
mod tests {
    use super::{
        CROSS_ENGINE_RATIO_DIRECTION, CrossEngineBenchmarkReport, EnvironmentFingerprint,
        TimingStats, WorkloadOperation, compute_ratio, compute_timing_stats,
        generate_markdown_report, parse_manifest, ratio_band,
    };
    use crate::raptorq_artifacts::generate_bundle_sidecar_and_reports;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_test_dir(label: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        std::env::temp_dir().join(format!("franken_numpy_{label}_{stamp}"))
    }

    #[test]
    fn manifest_parsing_accepts_valid_yaml() {
        let manifest = parse_manifest(
            r#"
version: "1.0.0"
workloads:
  - family: ufunc-elementwise
    name: add_medium
    size_tier: medium
    operation: binary_add
    shape: [100, 100]
    rhs_shape: [100, 100]
    warmup: 5
    samples: 50
"#,
        )
        .expect("manifest should parse");
        assert_eq!(manifest.workloads.len(), 1);
        assert_eq!(
            manifest.workloads[0].operation,
            WorkloadOperation::BinaryAdd
        );
    }

    #[test]
    fn manifest_validation_rejects_missing_shape_requirements() {
        let err = parse_manifest(
            r#"
version: "1.0.0"
workloads:
  - family: ufunc-elementwise
    name: invalid_add
    size_tier: medium
    operation: binary_add
    warmup: 5
    samples: 50
"#,
        )
        .expect_err("manifest should reject missing shape");
        assert!(err.contains("requires both shape and rhs_shape"));
    }

    #[test]
    fn percentile_computation_matches_expected_positions() {
        let stats = compute_timing_stats(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).expect("stats");
        assert_eq!(stats.p50_ns, 5);
        assert_eq!(stats.p95_ns, 10);
        assert_eq!(stats.p99_ns, 10);
    }

    #[test]
    fn percentile_computation_rejects_empty_input() {
        let err = compute_timing_stats(&[]).expect_err("empty samples must fail");
        assert!(err.contains("empty sample set"));
    }

    #[test]
    fn ratio_and_band_assignment_follow_expected_direction() {
        assert_eq!(compute_ratio(150, 100), 1.5);
        assert_eq!(ratio_band(1.5), "green");
        assert_eq!(ratio_band(5.0), "yellow");
        assert_eq!(ratio_band(15.0), "red");
        assert!(!compute_ratio(100, 0).is_finite());
    }

    #[test]
    fn report_json_round_trip_preserves_fields() {
        let report = CrossEngineBenchmarkReport {
            version: "1.0.0".to_string(),
            generated_at: "2026-04-09T20:00:00Z".to_string(),
            git_sha: "deadbeef".to_string(),
            env_fingerprint: EnvironmentFingerprint {
                rust_version: "rustc 1.90.0-nightly".to_string(),
                numpy_version: "2.3.0".to_string(),
                blas_backend: "openblas".to_string(),
                cpu: "x86_64".to_string(),
                os: "Linux".to_string(),
                python_executable: "python3".to_string(),
            },
            workloads: vec![super::CrossEngineWorkloadResult {
                family: "ufunc-elementwise".to_string(),
                name: "add_medium".to_string(),
                size_tier: "medium".to_string(),
                operation: WorkloadOperation::BinaryAdd,
                shape: vec![100, 100],
                rhs_shape: vec![100, 100],
                element_count: 10_000,
                warmup: 5,
                samples: 50,
                fnp: TimingStats {
                    p50_ns: 10,
                    p95_ns: 20,
                    p99_ns: 25,
                    mean_ns: 12.0,
                    stddev_ns: 3.0,
                    min_ns: 9,
                    max_ns: 25,
                    sample_count: 50,
                },
                numpy: TimingStats {
                    p50_ns: 5,
                    p95_ns: 9,
                    p99_ns: 11,
                    mean_ns: 6.0,
                    stddev_ns: 1.0,
                    min_ns: 4,
                    max_ns: 11,
                    sample_count: 50,
                },
                ratio: 2.0,
                ratio_direction: CROSS_ENGINE_RATIO_DIRECTION.to_string(),
                band: "green".to_string(),
            }],
        };
        let raw = serde_json::to_string(&report).expect("serialize report");
        let round_trip: CrossEngineBenchmarkReport =
            serde_json::from_str(&raw).expect("deserialize report");
        assert_eq!(round_trip.workloads[0].name, "add_medium");
        assert_eq!(round_trip.env_fingerprint.numpy_version, "2.3.0");
    }

    #[test]
    fn markdown_report_contains_summary_and_workload_rows() {
        let report = CrossEngineBenchmarkReport {
            version: "1.0.0".to_string(),
            generated_at: "2026-04-09T20:00:00Z".to_string(),
            git_sha: "deadbeef".to_string(),
            env_fingerprint: EnvironmentFingerprint {
                rust_version: "rustc nightly".to_string(),
                numpy_version: "2.3.0".to_string(),
                blas_backend: "openblas".to_string(),
                cpu: "x86_64".to_string(),
                os: "Linux".to_string(),
                python_executable: "python3".to_string(),
            },
            workloads: vec![super::CrossEngineWorkloadResult {
                family: "ufunc-elementwise".to_string(),
                name: "add_medium".to_string(),
                size_tier: "medium".to_string(),
                operation: WorkloadOperation::BinaryAdd,
                shape: vec![100, 100],
                rhs_shape: vec![100, 100],
                element_count: 10_000,
                warmup: 5,
                samples: 50,
                fnp: TimingStats {
                    p50_ns: 10,
                    p95_ns: 20,
                    p99_ns: 30,
                    mean_ns: 15.0,
                    stddev_ns: 2.0,
                    min_ns: 9,
                    max_ns: 30,
                    sample_count: 50,
                },
                numpy: TimingStats {
                    p50_ns: 5,
                    p95_ns: 8,
                    p99_ns: 9,
                    mean_ns: 6.0,
                    stddev_ns: 1.0,
                    min_ns: 4,
                    max_ns: 9,
                    sample_count: 50,
                },
                ratio: 2.0,
                ratio_direction: CROSS_ENGINE_RATIO_DIRECTION.to_string(),
                band: "green".to_string(),
            }],
        };
        let markdown = generate_markdown_report(&report);
        assert!(markdown.contains("Total workloads: 1"));
        assert!(markdown.contains("add_medium"));
        assert!(markdown.contains("NumPy: 2.3.0"));
    }

    #[test]
    fn npy_roundtrip_preserves_shape_and_values() {
        let shape = vec![2, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let bytes = fnp_io::save(&shape, &values, fnp_io::IOSupportedDType::F64).expect("save");
        let (loaded_shape, loaded_values, dtype) = fnp_io::load(&bytes).expect("load");
        assert_eq!(loaded_shape, shape);
        assert_eq!(loaded_values, values);
        assert_eq!(dtype, fnp_io::IOSupportedDType::F64);
    }

    #[test]
    fn raptorq_artifacts_can_wrap_cross_engine_report_bundle() {
        let dir = unique_test_dir("cross_engine_raptorq");
        fs::create_dir_all(&dir).expect("create temp dir");

        let report_path = dir.join("cross_engine_benchmark_v1.json");
        let markdown_path = dir.join("cross_engine_benchmark_v1.report.md");
        let manifest_path = dir.join("cross_engine_benchmark_workloads_v1.yaml");
        fs::write(&report_path, "{\"ok\":true}").expect("write report");
        fs::write(&markdown_path, "# report").expect("write markdown");
        fs::write(&manifest_path, "version: 1.0.0\nworkloads: []\n").expect("write manifest");

        let sidecar_path = dir.join("cross_engine_benchmark_v1.raptorq.json");
        let scrub_path = dir.join("cross_engine_benchmark_v1.scrub_report.json");
        let decode_path = dir.join("cross_engine_benchmark_v1.decode_proof.json");
        generate_bundle_sidecar_and_reports(
            "cross_engine_benchmark_v1",
            &dir,
            &[report_path, markdown_path, manifest_path],
            &sidecar_path,
            &scrub_path,
            &decode_path,
            1234,
        )
        .expect("generate bundle artifacts");

        assert!(sidecar_path.is_file());
        assert!(scrub_path.is_file());
        assert!(decode_path.is_file());
    }
}
