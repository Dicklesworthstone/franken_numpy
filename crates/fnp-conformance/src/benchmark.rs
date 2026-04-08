#![forbid(unsafe_code)]

use fnp_dtype::DType;
use fnp_io::{IOSupportedDType, load, save};
use fnp_ufunc::{BinaryOp, UFuncArray};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const REPRO_COMMAND: &str = "cargo run -p fnp-conformance --bin generate_benchmark_baseline";
pub const MEMORY_FOOTPRINT_SLO_PATH: &str = "memory footprint";
pub const ALLOCATION_CHURN_SLO_PATH: &str = "allocation churn";
pub const ALLOCATOR_FRAGMENTATION_SLO_PATH: &str =
    "allocator fragmentation under adversarial workloads";
pub const REQUIRED_SLO_PATHS: &[&str] = &[
    MEMORY_FOOTPRINT_SLO_PATH,
    ALLOCATION_CHURN_SLO_PATH,
    ALLOCATOR_FRAGMENTATION_SLO_PATH,
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PercentileSummary {
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkWorkload {
    pub name: String,
    pub runs: usize,
    pub samples_ms: Vec<f64>,
    pub percentiles: PercentileSummary,
    #[serde(default)]
    pub telemetry: WorkloadTelemetry,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AllocatorStressLevel {
    #[default]
    None,
    Steady,
    Adversarial,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkloadTelemetry {
    pub elements_per_run: usize,
    pub bytes_processed_per_run: usize,
    pub throughput_elements_per_sec_p50: f64,
    pub throughput_elements_per_sec_p95: f64,
    pub bandwidth_mib_per_sec_p50: f64,
    pub bandwidth_mib_per_sec_p95: f64,
    #[serde(default)]
    pub peak_live_bytes_per_run: usize,
    #[serde(default)]
    pub heap_allocations_per_run: usize,
    #[serde(default)]
    pub allocator_stress: AllocatorStressLevel,
}

impl WorkloadTelemetry {
    pub fn covered_slo_paths(&self) -> Vec<&'static str> {
        let mut covered = Vec::with_capacity(3);
        if self.peak_live_bytes_per_run > 0 {
            covered.push(MEMORY_FOOTPRINT_SLO_PATH);
        }
        if self.heap_allocations_per_run > 0 {
            covered.push(ALLOCATION_CHURN_SLO_PATH);
        }
        if self.allocator_stress == AllocatorStressLevel::Adversarial {
            covered.push(ALLOCATOR_FRAGMENTATION_SLO_PATH);
        }
        covered
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReproMetadata {
    pub command: String,
    pub repo_root: String,
    pub output_path: String,
    pub rustc_version: String,
    pub cargo_profile: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkBaseline {
    pub schema_version: u8,
    pub generated_at_unix_ms: u128,
    pub git_commit: String,
    pub workloads: Vec<BenchmarkWorkload>,
    #[serde(default)]
    pub environment_fingerprint: String,
    #[serde(default)]
    pub reproducibility: ReproMetadata,
    #[serde(default)]
    pub evidence_log_refs: Vec<String>,
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn git_commit_short(repo_root: &Path) -> String {
    let output = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
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

fn compute_per_second(units_per_run: f64, sample_ms: f64) -> f64 {
    if sample_ms <= 0.0 {
        return 0.0;
    }
    units_per_run * 1000.0 / sample_ms
}

fn relative_path_or_display(repo_root: &Path, path: &Path) -> String {
    match path.strip_prefix(repo_root) {
        Ok(relative) => relative.display().to_string(),
        Err(_) => path.display().to_string(),
    }
}

fn newest_log_with_prefix(log_dir: &Path, prefix: &str) -> Option<String> {
    let mut candidates = Vec::new();
    let Ok(entries) = fs::read_dir(log_dir) else {
        return None;
    };

    for entry in entries.filter_map(Result::ok) {
        let name_os = entry.file_name();
        let name = name_os.to_string_lossy();
        if name.starts_with(prefix) {
            candidates.push(name.to_string());
        }
    }

    candidates.sort();
    candidates.pop()
}

fn discover_evidence_log_refs(repo_root: &Path) -> Vec<String> {
    let log_dir = repo_root.join("artifacts/logs");
    let mut refs = Vec::new();
    for prefix in [
        "runtime_policy_e2e_",
        "test_contract_e2e_",
        "workflow_scenario_e2e_",
    ] {
        if let Some(file_name) = newest_log_with_prefix(&log_dir, prefix) {
            refs.push(format!("artifacts/logs/{file_name}"));
        }
    }

    if refs.is_empty() {
        refs.push("artifacts/contracts/test_logging_contract_v1.json".to_string());
        refs.push("scripts/e2e/run_test_contract_gate.sh".to_string());
        refs.push("scripts/e2e/run_workflow_scenario_gate.sh".to_string());
    }

    refs
}

fn percentile_index(len: usize, percentile_num: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let last = len - 1;
    (last * percentile_num + 50) / 100
}

fn summarize_samples(samples: &[f64]) -> PercentileSummary {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

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

#[derive(Debug, Clone, Copy)]
struct WorkloadInstrumentation {
    elements_per_run: usize,
    bytes_processed_per_run: usize,
    peak_live_bytes_per_run: usize,
    heap_allocations_per_run: usize,
    allocator_stress: AllocatorStressLevel,
}

fn time_workload<F>(
    name: &str,
    runs: usize,
    instrumentation: WorkloadInstrumentation,
    mut run_fn: F,
) -> Result<BenchmarkWorkload, String>
where
    F: FnMut() -> Result<(), String>,
{
    let mut samples_ms = Vec::with_capacity(runs);

    for _ in 0..runs {
        let start = Instant::now();
        run_fn()?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        samples_ms.push(elapsed_ms);
    }

    let percentiles = summarize_samples(&samples_ms);
    let bytes_per_run_mib = instrumentation.bytes_processed_per_run as f64 / (1024.0 * 1024.0);
    let telemetry = WorkloadTelemetry {
        elements_per_run: instrumentation.elements_per_run,
        bytes_processed_per_run: instrumentation.bytes_processed_per_run,
        throughput_elements_per_sec_p50: compute_per_second(
            instrumentation.elements_per_run as f64,
            percentiles.p50_ms,
        ),
        throughput_elements_per_sec_p95: compute_per_second(
            instrumentation.elements_per_run as f64,
            percentiles.p95_ms,
        ),
        bandwidth_mib_per_sec_p50: compute_per_second(bytes_per_run_mib, percentiles.p50_ms),
        bandwidth_mib_per_sec_p95: compute_per_second(bytes_per_run_mib, percentiles.p95_ms),
        peak_live_bytes_per_run: instrumentation.peak_live_bytes_per_run,
        heap_allocations_per_run: instrumentation.heap_allocations_per_run,
        allocator_stress: instrumentation.allocator_stress,
    };

    Ok(BenchmarkWorkload {
        name: name.to_string(),
        runs,
        samples_ms,
        percentiles,
        telemetry,
    })
}

pub fn generate_benchmark_baseline(
    repo_root: &Path,
    output_path: &Path,
) -> Result<BenchmarkBaseline, String> {
    let item_size = DType::F64.item_size();
    let lhs_add = UFuncArray::new(
        vec![256, 256],
        (0..(256 * 256)).map(|i| f64::from(i as u32)).collect(),
        DType::F64,
    )
    .map_err(|err| format!("benchmark lhs_add init failed: {err}"))?;
    let rhs_add = UFuncArray::new(
        vec![256],
        (0..256).map(|i| f64::from((i * 2) as u32)).collect(),
        DType::F64,
    )
    .map_err(|err| format!("benchmark rhs_add init failed: {err}"))?;

    let reduce_in = UFuncArray::new(
        vec![256, 256],
        (0..(256 * 256))
            .map(|i| f64::from((i % 97) as u32))
            .collect(),
        DType::F64,
    )
    .map_err(|err| format!("benchmark reduce init failed: {err}"))?;

    let add_elements_per_run = 256 * 256;
    let add_bytes_per_run = add_elements_per_run * item_size * 3;
    let add_workload = time_workload(
        "ufunc_add_broadcast_256x256_by_256",
        20,
        WorkloadInstrumentation {
            elements_per_run: add_elements_per_run,
            bytes_processed_per_run: add_bytes_per_run,
            peak_live_bytes_per_run: add_bytes_per_run,
            heap_allocations_per_run: 1,
            allocator_stress: AllocatorStressLevel::Steady,
        },
        || {
            let out = lhs_add
                .elementwise_binary(&rhs_add, BinaryOp::Add)
                .map_err(|err| format!("broadcast add failed: {err}"))?;
            std::hint::black_box(out.values()[0]);
            Ok(())
        },
    )?;

    let reduce_axis1_elements_per_run = 256;
    let reduce_axis1_bytes_per_run = (256 * 256 + reduce_axis1_elements_per_run) * item_size;
    let reduce_axis1 = time_workload(
        "reduce_sum_axis1_keepdims_false_256x256",
        20,
        WorkloadInstrumentation {
            elements_per_run: reduce_axis1_elements_per_run,
            bytes_processed_per_run: reduce_axis1_bytes_per_run,
            peak_live_bytes_per_run: reduce_axis1_bytes_per_run,
            heap_allocations_per_run: 1,
            allocator_stress: AllocatorStressLevel::Steady,
        },
        || {
            let out = reduce_in
                .reduce_sum(Some(1), false)
                .map_err(|err| format!("axis reduction failed: {err}"))?;
            std::hint::black_box(out.values()[0]);
            Ok(())
        },
    )?;

    let reduce_all_elements_per_run = 1;
    let reduce_all_bytes_per_run = (256 * 256 + reduce_all_elements_per_run) * item_size;
    let reduce_all = time_workload(
        "reduce_sum_all_keepdims_false_256x256",
        20,
        WorkloadInstrumentation {
            elements_per_run: reduce_all_elements_per_run,
            bytes_processed_per_run: reduce_all_bytes_per_run,
            peak_live_bytes_per_run: reduce_all_bytes_per_run,
            heap_allocations_per_run: 1,
            allocator_stress: AllocatorStressLevel::Steady,
        },
        || {
            let out = reduce_in
                .reduce_sum(None, false)
                .map_err(|err| format!("global reduction failed: {err}"))?;
            std::hint::black_box(out.values()[0]);
            Ok(())
        },
    )?;

    let large_add_dim = 1024usize;
    let large_add_elements_per_run = large_add_dim * large_add_dim;
    let lhs_add_large = UFuncArray::new(
        vec![large_add_dim, large_add_dim],
        (0..large_add_elements_per_run)
            .map(|i| f64::from((i % 257) as u32))
            .collect(),
        DType::F64,
    )
    .map_err(|err| format!("benchmark lhs_add_large init failed: {err}"))?;
    let rhs_add_large = UFuncArray::new(
        vec![large_add_dim],
        (0..large_add_dim)
            .map(|i| f64::from((i % 31) as u32))
            .collect(),
        DType::F64,
    )
    .map_err(|err| format!("benchmark rhs_add_large init failed: {err}"))?;
    let add_large_bytes_per_run = large_add_elements_per_run * item_size * 3;
    let add_large_workload = time_workload(
        "ufunc_add_broadcast_1024x1024_by_1024",
        8,
        WorkloadInstrumentation {
            elements_per_run: large_add_elements_per_run,
            bytes_processed_per_run: add_large_bytes_per_run,
            peak_live_bytes_per_run: add_large_bytes_per_run,
            heap_allocations_per_run: 1,
            allocator_stress: AllocatorStressLevel::Steady,
        },
        || {
            let out = lhs_add_large
                .elementwise_binary(&rhs_add_large, BinaryOp::Add)
                .map_err(|err| format!("large broadcast add failed: {err}"))?;
            std::hint::black_box(out.values()[0]);
            Ok(())
        },
    )?;

    let matmul_dim = 256usize;
    let matmul_elements = matmul_dim * matmul_dim;
    let matmul_lhs = UFuncArray::new(
        vec![matmul_dim, matmul_dim],
        (0..matmul_elements)
            .map(|i| f64::from(((i * 13) % 997) as u32) / 17.0)
            .collect(),
        DType::F64,
    )
    .map_err(|err| format!("benchmark matmul_lhs init failed: {err}"))?;
    let matmul_rhs = UFuncArray::new(
        vec![matmul_dim, matmul_dim],
        (0..matmul_elements)
            .map(|i| f64::from(((i * 17) % 991) as u32) / 19.0)
            .collect(),
        DType::F64,
    )
    .map_err(|err| format!("benchmark matmul_rhs init failed: {err}"))?;
    let matmul_bytes_per_run = (matmul_elements * 3) * item_size;
    let matmul_workload = time_workload(
        "matmul_256x256_by_256x256",
        6,
        WorkloadInstrumentation {
            elements_per_run: matmul_elements,
            bytes_processed_per_run: matmul_bytes_per_run,
            peak_live_bytes_per_run: matmul_bytes_per_run,
            heap_allocations_per_run: 1,
            allocator_stress: AllocatorStressLevel::Steady,
        },
        || {
            let out = matmul_lhs
                .matmul(&matmul_rhs)
                .map_err(|err| format!("matmul workload failed: {err}"))?;
            std::hint::black_box(out.values()[0]);
            Ok(())
        },
    )?;

    let sort_len = 1_000_000usize;
    let sort_in = UFuncArray::new(
        vec![sort_len],
        (0..sort_len)
            .map(|i| f64::from(((i * 48_271) % sort_len) as u32))
            .collect(),
        DType::F64,
    )
    .map_err(|err| format!("benchmark sort input init failed: {err}"))?;
    let sort_bytes_per_run = sort_len * item_size * 2;
    let sort_workload = time_workload(
        "sort_quicksort_1m",
        8,
        WorkloadInstrumentation {
            elements_per_run: sort_len,
            bytes_processed_per_run: sort_bytes_per_run,
            peak_live_bytes_per_run: sort_bytes_per_run,
            heap_allocations_per_run: 1,
            allocator_stress: AllocatorStressLevel::Steady,
        },
        || {
            let out = sort_in
                .sort(None, Some("quicksort"))
                .map_err(|err| format!("sort workload failed: {err}"))?;
            std::hint::black_box(out.values()[0]);
            Ok(())
        },
    )?;

    let fft_len = 65_536usize;
    let fft_in = UFuncArray::new(
        vec![fft_len],
        (0..fft_len)
            .map(|i| {
                let t = i as f64 / fft_len as f64;
                (std::f64::consts::TAU * 5.0 * t).sin()
                    + 0.5 * (std::f64::consts::TAU * 13.0 * t).cos()
            })
            .collect(),
        DType::F64,
    )
    .map_err(|err| format!("benchmark fft input init failed: {err}"))?;
    let fft_bytes_per_run = fft_len * item_size * 3;
    let fft_workload = time_workload(
        "fft_65536",
        8,
        WorkloadInstrumentation {
            elements_per_run: fft_len,
            bytes_processed_per_run: fft_bytes_per_run,
            peak_live_bytes_per_run: fft_bytes_per_run,
            heap_allocations_per_run: 1,
            allocator_stress: AllocatorStressLevel::Steady,
        },
        || {
            let out = fft_in
                .fft(None)
                .map_err(|err| format!("fft workload failed: {err}"))?;
            std::hint::black_box(out.values()[0]);
            Ok(())
        },
    )?;

    let astype_in = UFuncArray::new(
        vec![large_add_dim, large_add_dim],
        (0..large_add_elements_per_run)
            .map(|i| f64::from(((i * 19) % 10_003) as u32))
            .collect(),
        DType::F64,
    )
    .map_err(|err| format!("benchmark astype input init failed: {err}"))?;
    let astype_bytes_per_run =
        large_add_elements_per_run * (DType::F64.item_size() + DType::I32.item_size());
    let astype_workload = time_workload(
        "astype_f64_to_i32_1024x1024",
        10,
        WorkloadInstrumentation {
            elements_per_run: large_add_elements_per_run,
            bytes_processed_per_run: astype_bytes_per_run,
            peak_live_bytes_per_run: astype_bytes_per_run,
            heap_allocations_per_run: 1,
            allocator_stress: AllocatorStressLevel::Steady,
        },
        || {
            let out = astype_in.astype(DType::I32);
            std::hint::black_box(out.values()[0]);
            Ok(())
        },
    )?;

    let reshape_workload = time_workload(
        "reshape_1024x1024_to_2048x512",
        20,
        WorkloadInstrumentation {
            elements_per_run: large_add_elements_per_run,
            bytes_processed_per_run: large_add_elements_per_run * item_size,
            peak_live_bytes_per_run: large_add_elements_per_run * item_size,
            heap_allocations_per_run: 0,
            allocator_stress: AllocatorStressLevel::None,
        },
        || {
            let out = astype_in
                .reshape(&[2048, 512])
                .map_err(|err| format!("reshape workload failed: {err}"))?;
            std::hint::black_box(out.shape()[0]);
            Ok(())
        },
    )?;

    let io_dim = 512usize;
    let io_elements = io_dim * io_dim;
    let io_values: Vec<f64> = (0..io_elements)
        .map(|i| f64::from(((i * 29) % 65_537) as u32) / 11.0)
        .collect();
    let fragmentation_lengths = [257usize, 8_192, 521, 16_384, 1_031, 32_768, 2_053, 4_096];
    let fragmentation_inputs: Vec<Vec<f64>> = fragmentation_lengths
        .iter()
        .map(|&len| {
            (0..len)
                .map(|i| f64::from(((i * 37 + len) % 65_537) as u32) / 13.0)
                .collect()
        })
        .collect();
    let io_npy_workload = time_workload(
        "io_npy_save_load_512x512_f64",
        8,
        WorkloadInstrumentation {
            elements_per_run: io_elements,
            bytes_processed_per_run: io_elements * item_size * 2,
            peak_live_bytes_per_run: io_elements * item_size * 3,
            heap_allocations_per_run: 3,
            allocator_stress: AllocatorStressLevel::Steady,
        },
        || {
            let payload = save(&[io_dim, io_dim], &io_values, IOSupportedDType::F64)
                .map_err(|err| format!("io save workload failed: {err}"))?;
            let (shape, values, dtype) =
                load(&payload).map_err(|err| format!("io load workload failed: {err}"))?;
            if shape != vec![io_dim, io_dim] {
                return Err(format!(
                    "io workload shape mismatch expected=[{io_dim},{io_dim}] actual={shape:?}"
                ));
            }
            if dtype != IOSupportedDType::F64 {
                return Err(format!(
                    "io workload dtype mismatch expected=F64 actual={dtype:?}"
                ));
            }
            std::hint::black_box(values[0]);
            Ok(())
        },
    )?;
    let fragmentation_elements_per_run = fragmentation_lengths.iter().sum::<usize>();
    let fragmentation_peak_len = fragmentation_lengths.iter().copied().max().unwrap_or(0);
    let fragmentation_workload = time_workload(
        "allocator_fragmentation_adversarial_save_load_mix",
        6,
        WorkloadInstrumentation {
            elements_per_run: fragmentation_elements_per_run,
            bytes_processed_per_run: fragmentation_elements_per_run * item_size * 2,
            peak_live_bytes_per_run: fragmentation_peak_len * item_size * 3,
            heap_allocations_per_run: fragmentation_lengths.len() * 3,
            allocator_stress: AllocatorStressLevel::Adversarial,
        },
        || {
            let mut checksum = 0.0f64;
            for (&len, values) in fragmentation_lengths
                .iter()
                .zip(fragmentation_inputs.iter())
            {
                let payload = save(&[len], values, IOSupportedDType::F64)
                    .map_err(|err| format!("fragmentation save workload failed: {err}"))?;
                let (shape, decoded, dtype) = load(&payload)
                    .map_err(|err| format!("fragmentation load workload failed: {err}"))?;
                if shape != vec![len] {
                    return Err(format!(
                        "fragmentation workload shape mismatch expected=[{len}] actual={shape:?}"
                    ));
                }
                if dtype != IOSupportedDType::F64 {
                    return Err(format!(
                        "fragmentation workload dtype mismatch expected=F64 actual={dtype:?}"
                    ));
                }
                checksum += decoded.first().copied().unwrap_or(0.0);
            }
            std::hint::black_box(checksum);
            Ok(())
        },
    )?;

    let rustc = rustc_version();
    let environment_fingerprint = format!(
        "os={} arch={} cpus={} rustc={}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::thread::available_parallelism().map_or(1, usize::from),
        rustc
    );
    let reproducibility = ReproMetadata {
        command: REPRO_COMMAND.to_string(),
        repo_root: repo_root.display().to_string(),
        output_path: relative_path_or_display(repo_root, output_path),
        rustc_version: rustc,
        cargo_profile: "release".to_string(),
    };

    let baseline = BenchmarkBaseline {
        schema_version: 1,
        generated_at_unix_ms: now_unix_ms(),
        git_commit: git_commit_short(repo_root),
        workloads: vec![
            add_workload,
            reduce_axis1,
            reduce_all,
            add_large_workload,
            matmul_workload,
            sort_workload,
            fft_workload,
            astype_workload,
            reshape_workload,
            io_npy_workload,
            fragmentation_workload,
        ],
        environment_fingerprint,
        reproducibility,
        evidence_log_refs: discover_evidence_log_refs(repo_root),
    };

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }

    let raw = serde_json::to_string_pretty(&baseline)
        .map_err(|err| format!("failed serializing baseline: {err}"))?;
    fs::write(output_path, raw)
        .map_err(|err| format!("failed writing {}: {err}", output_path.display()))?;

    Ok(baseline)
}

#[cfg(test)]
mod tests {
    use super::{BenchmarkBaseline, REQUIRED_SLO_PATHS, generate_benchmark_baseline};
    use std::collections::BTreeSet;
    use std::fs;

    fn temp_file(name: &str) -> std::path::PathBuf {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        std::env::temp_dir().join(format!("fnp_{name}_{ts}.json"))
    }

    #[test]
    fn baseline_generator_writes_json() {
        let output_path = temp_file("baseline");
        let repo_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");

        let baseline = generate_benchmark_baseline(&repo_root, &output_path)
            .expect("baseline generation should succeed");
        assert_eq!(baseline.schema_version, 1);
        assert!(!baseline.workloads.is_empty());
        assert!(!baseline.environment_fingerprint.trim().is_empty());
        assert!(
            baseline
                .reproducibility
                .command
                .contains("generate_benchmark_baseline")
        );
        assert!(!baseline.evidence_log_refs.is_empty());
        for workload in &baseline.workloads {
            assert!(workload.telemetry.bytes_processed_per_run > 0);
            assert!(workload.telemetry.bandwidth_mib_per_sec_p50 > 0.0);
            assert!(workload.telemetry.throughput_elements_per_sec_p50 > 0.0);
        }
        let observed_slo_paths: BTreeSet<&'static str> = baseline
            .workloads
            .iter()
            .flat_map(|workload| workload.telemetry.covered_slo_paths())
            .collect();
        let required_slo_paths: BTreeSet<&'static str> =
            REQUIRED_SLO_PATHS.iter().copied().collect();
        assert_eq!(observed_slo_paths, required_slo_paths);

        let raw = fs::read_to_string(&output_path).expect("baseline file readable");
        let parsed: BenchmarkBaseline = serde_json::from_str(&raw).expect("baseline json parse");
        assert_eq!(parsed.workloads.len(), baseline.workloads.len());
        assert_eq!(
            parsed.reproducibility.command,
            baseline.reproducibility.command
        );
        let parsed_slo_paths: BTreeSet<&'static str> = parsed
            .workloads
            .iter()
            .flat_map(|workload| workload.telemetry.covered_slo_paths())
            .collect();
        assert_eq!(parsed_slo_paths, required_slo_paths);

        let _ = fs::remove_file(output_path);
    }
}
