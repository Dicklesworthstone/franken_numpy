#![forbid(unsafe_code)]

use fnp_dtype::DType;
use fnp_ufunc::{BinaryOp, ParallelPartitionConfig, UFuncArray};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const SCHEMA_VERSION: u8 = 1;
const DEFAULT_REPORT_PATH: &str = "target/parallel_calibration_matrix.json";
const DEFAULT_SAMPLE_COUNT: usize = 3;
const DEFAULT_MIN_ELEMENTS_PER_CHUNK: usize = 8_192;
const REPRO_COMMAND: &str = "cargo run -p fnp-conformance --bin run_parallel_calibration_matrix";

fn main() {
    if let Err(err) = run() {
        eprintln!("run_parallel_calibration_matrix failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let options = Options::parse(std::env::args().skip(1), &repo_root)?;
    let workloads = calibration_workloads(options.mode);
    let report = build_report(&options, &workloads)?;
    write_report(&options.report_path, &report)?;
    println!(
        "parallel_calibration_matrix status=ok workloads={} entries={} report={}",
        report.workload_count,
        report.entries.len(),
        options.report_path.display()
    );
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Options {
    repo_root: PathBuf,
    mode: CalibrationMode,
    report_path: PathBuf,
    sample_count: usize,
    thread_counts: Vec<usize>,
    min_elements_per_chunk: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CalibrationMode {
    Quick,
    Standard,
    LargeHost,
}

impl CalibrationMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Quick => "quick",
            Self::Standard => "standard",
            Self::LargeHost => "large_host",
        }
    }
}

impl Options {
    fn parse<I>(args: I, repo_root: &Path) -> Result<Self, String>
    where
        I: IntoIterator<Item = String>,
    {
        let mut mode = CalibrationMode::Standard;
        let mut report_path = repo_root.join(DEFAULT_REPORT_PATH);
        let mut sample_count = DEFAULT_SAMPLE_COUNT;
        let mut thread_counts = vec![2, 4];
        let mut thread_counts_were_explicit = false;
        let mut min_elements_per_chunk = DEFAULT_MIN_ELEMENTS_PER_CHUNK;

        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--quick" => {
                    if mode == CalibrationMode::LargeHost {
                        return Err("--quick and --large-host cannot be combined".to_string());
                    }
                    mode = CalibrationMode::Quick;
                }
                "--large-host" => {
                    if mode == CalibrationMode::Quick {
                        return Err("--quick and --large-host cannot be combined".to_string());
                    }
                    mode = CalibrationMode::LargeHost;
                }
                "--report-path" | "--output-path" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("{arg} requires a value"))?;
                    report_path = PathBuf::from(value);
                }
                "--samples" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "--samples requires a value".to_string())?;
                    sample_count = parse_nonzero_usize("--samples", &value)?;
                }
                "--threads" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "--threads requires a value".to_string())?;
                    thread_counts = parse_thread_counts(&value)?;
                    thread_counts_were_explicit = true;
                }
                "--min-elements-per-chunk" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "--min-elements-per-chunk requires a value".to_string())?;
                    min_elements_per_chunk =
                        parse_nonzero_usize("--min-elements-per-chunk", &value)?;
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                unknown => return Err(format!("unknown argument: {unknown}")),
            }
        }

        if mode == CalibrationMode::LargeHost && !thread_counts_were_explicit {
            thread_counts = vec![2, 4, 8, 16, 32, 64];
        }

        Ok(Self {
            repo_root: repo_root.to_path_buf(),
            mode,
            report_path,
            sample_count,
            thread_counts,
            min_elements_per_chunk,
        })
    }
}

fn print_help() {
    println!(
        "Usage: {REPRO_COMMAND} -- [--quick|--large-host] [--report-path <path>] [--samples <n>] [--threads <csv>] [--min-elements-per-chunk <n>]"
    );
}

fn parse_nonzero_usize(flag: &str, value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|err| format!("{flag} must be a positive integer: {err}"))?;
    if parsed == 0 {
        return Err(format!("{flag} must be at least 1"));
    }
    Ok(parsed)
}

fn parse_thread_counts(raw: &str) -> Result<Vec<usize>, String> {
    let mut counts = raw
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| parse_nonzero_usize("--threads", value))
        .collect::<Result<Vec<_>, _>>()?;
    counts.sort_unstable();
    counts.dedup();
    if counts.is_empty() {
        return Err("--threads must include at least one positive integer".to_string());
    }
    Ok(counts)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum CalibrationOperation {
    BroadcastAdd,
    AxisSum,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WorkloadSpec {
    workload_id: String,
    operation: CalibrationOperation,
    shape: Vec<usize>,
    rhs_shape: Option<Vec<usize>>,
    axis: Option<isize>,
    keepdims: bool,
}

impl WorkloadSpec {
    fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    fn output_count(&self) -> Result<usize, String> {
        match (self.operation, self.axis) {
            (CalibrationOperation::AxisSum, Some(axis)) => {
                let normalized = normalize_axis(axis, self.shape.len())?;
                Ok(self
                    .shape
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, dim)| (idx != normalized).then_some(*dim))
                    .product::<usize>()
                    .max(1))
            }
            (CalibrationOperation::AxisSum, None) => Ok(1),
            (CalibrationOperation::BroadcastAdd, _) => Ok(self.element_count()),
        }
    }

    fn bytes_touched_estimate(&self) -> Result<usize, String> {
        let item_size = DType::F64.item_size();
        match self.operation {
            CalibrationOperation::BroadcastAdd => Ok(self.element_count() * item_size * 3),
            CalibrationOperation::AxisSum => {
                Ok((self.element_count() + self.output_count()?) * item_size)
            }
        }
    }
}

fn calibration_workloads(mode: CalibrationMode) -> Vec<WorkloadSpec> {
    match mode {
        CalibrationMode::Quick => vec![
            broadcast_add_workload(128, 128),
            axis_sum_workload(128, 128),
        ],
        CalibrationMode::Standard => vec![
            broadcast_add_workload(128, 128),
            broadcast_add_workload(512, 512),
            axis_sum_workload(128, 128),
            axis_sum_workload(512, 512),
        ],
        CalibrationMode::LargeHost => vec![
            broadcast_add_workload(512, 512),
            broadcast_add_workload(1024, 1024),
            broadcast_add_workload(2048, 2048),
            axis_sum_workload(512, 512),
            axis_sum_workload(1024, 1024),
            axis_sum_workload(2048, 2048),
        ],
    }
}

fn broadcast_add_workload(rows: usize, cols: usize) -> WorkloadSpec {
    WorkloadSpec {
        workload_id: format!("broadcast_add_{rows}x{cols}_by_{cols}"),
        operation: CalibrationOperation::BroadcastAdd,
        shape: vec![rows, cols],
        rhs_shape: Some(vec![cols]),
        axis: None,
        keepdims: false,
    }
}

fn axis_sum_workload(rows: usize, cols: usize) -> WorkloadSpec {
    WorkloadSpec {
        workload_id: format!("reduce_sum_axis1_{rows}x{cols}"),
        operation: CalibrationOperation::AxisSum,
        shape: vec![rows, cols],
        rhs_shape: None,
        axis: Some(1),
        keepdims: false,
    }
}

#[derive(Debug, Clone, Serialize)]
struct CalibrationReport {
    schema_version: u8,
    generated_at_unix_ms: u128,
    git_commit: String,
    mode: &'static str,
    workload_count: usize,
    sample_count: usize,
    repo_root: String,
    reproduction_command: String,
    host: HostMetadata,
    entries: Vec<CalibrationEntry>,
}

#[derive(Debug, Clone, Serialize)]
struct HostMetadata {
    available_parallelism: usize,
}

#[derive(Debug, Clone, Serialize)]
struct CalibrationEntry {
    workload_id: String,
    operation: CalibrationOperation,
    shape: Vec<usize>,
    rhs_shape: Option<Vec<usize>>,
    axis: Option<isize>,
    keepdims: bool,
    input_elements_per_run: usize,
    output_elements_per_run: usize,
    bytes_touched_estimate_per_run: usize,
    thread_count: usize,
    min_elements_per_chunk: usize,
    expected_chunk_count: usize,
    serial: TimingSummary,
    parallel: TimingSummary,
    speedup: SpeedupSummary,
    telemetry: CalibrationTelemetry,
}

#[derive(Debug, Clone, Serialize)]
struct TimingSummary {
    samples_ms: Vec<f64>,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct SpeedupSummary {
    p50_ratio: f64,
    p95_ratio: f64,
    p99_ratio: f64,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct CalibrationTelemetry {
    input_elements_per_run: usize,
    output_elements_per_run: usize,
    elements_per_run: usize,
    bytes_touched_estimate_per_run: usize,
    bytes_processed_per_run: usize,
    peak_live_bytes_per_run: usize,
    process_high_water_rss_bytes: Option<u64>,
    available_parallelism: usize,
    configured_thread_count: Option<usize>,
    serial_throughput_elements_per_sec_p50: f64,
    serial_throughput_elements_per_sec_p95: f64,
    parallel_throughput_elements_per_sec_p50: f64,
    parallel_throughput_elements_per_sec_p95: f64,
    serial_bandwidth_mib_per_sec_p50: f64,
    serial_bandwidth_mib_per_sec_p95: f64,
    parallel_bandwidth_mib_per_sec_p50: f64,
    parallel_bandwidth_mib_per_sec_p95: f64,
}

fn build_report(
    options: &Options,
    workloads: &[WorkloadSpec],
) -> Result<CalibrationReport, String> {
    let mut entries = Vec::with_capacity(workloads.len() * options.thread_counts.len());
    for workload in workloads {
        for &thread_count in &options.thread_counts {
            entries.push(run_entry(options, workload, thread_count)?);
        }
    }

    Ok(CalibrationReport {
        schema_version: SCHEMA_VERSION,
        generated_at_unix_ms: now_unix_ms(),
        git_commit: git_commit_short(&options.repo_root),
        mode: options.mode.as_str(),
        workload_count: workloads.len(),
        sample_count: options.sample_count,
        repo_root: options.repo_root.display().to_string(),
        reproduction_command: REPRO_COMMAND.to_string(),
        host: HostMetadata {
            available_parallelism: available_parallelism(),
        },
        entries,
    })
}

fn run_entry(
    options: &Options,
    workload: &WorkloadSpec,
    thread_count: usize,
) -> Result<CalibrationEntry, String> {
    let config = ParallelPartitionConfig::from_worker_count(thread_count)?
        .with_min_elements_per_chunk(options.min_elements_per_chunk)?;
    let arrays = WorkloadArrays::new(workload)?;
    let input_count = workload.element_count();
    let output_count = workload.output_count()?;
    let bytes_touched_estimate = workload.bytes_touched_estimate()?;
    let expected_chunk_count = expected_chunk_count(output_count, config);
    let mut serial_samples = Vec::with_capacity(options.sample_count);
    let mut parallel_samples = Vec::with_capacity(options.sample_count);

    for _ in 0..options.sample_count {
        let (serial_ms, parallel_ms) = arrays.measure(workload, config)?;
        serial_samples.push(serial_ms);
        parallel_samples.push(parallel_ms);
    }

    let serial = summarize_samples(serial_samples)?;
    let parallel = summarize_samples(parallel_samples)?;
    let speedup = SpeedupSummary {
        p50_ratio: ratio(serial.p50_ms, parallel.p50_ms),
        p95_ratio: ratio(serial.p95_ms, parallel.p95_ms),
        p99_ratio: ratio(serial.p99_ms, parallel.p99_ms),
    };
    let telemetry = CalibrationTelemetry {
        input_elements_per_run: input_count,
        output_elements_per_run: output_count,
        elements_per_run: input_count,
        bytes_touched_estimate_per_run: bytes_touched_estimate,
        bytes_processed_per_run: bytes_touched_estimate,
        peak_live_bytes_per_run: bytes_touched_estimate,
        process_high_water_rss_bytes: process_high_water_rss_bytes(),
        available_parallelism: available_parallelism(),
        configured_thread_count: Some(thread_count),
        serial_throughput_elements_per_sec_p50: compute_per_second(input_count, serial.p50_ms)?,
        serial_throughput_elements_per_sec_p95: compute_per_second(input_count, serial.p95_ms)?,
        parallel_throughput_elements_per_sec_p50: compute_per_second(input_count, parallel.p50_ms)?,
        parallel_throughput_elements_per_sec_p95: compute_per_second(input_count, parallel.p95_ms)?,
        serial_bandwidth_mib_per_sec_p50: compute_mib_per_second(
            bytes_touched_estimate,
            serial.p50_ms,
        )?,
        serial_bandwidth_mib_per_sec_p95: compute_mib_per_second(
            bytes_touched_estimate,
            serial.p95_ms,
        )?,
        parallel_bandwidth_mib_per_sec_p50: compute_mib_per_second(
            bytes_touched_estimate,
            parallel.p50_ms,
        )?,
        parallel_bandwidth_mib_per_sec_p95: compute_mib_per_second(
            bytes_touched_estimate,
            parallel.p95_ms,
        )?,
    };

    Ok(CalibrationEntry {
        workload_id: workload.workload_id.clone(),
        operation: workload.operation,
        shape: workload.shape.clone(),
        rhs_shape: workload.rhs_shape.clone(),
        axis: workload.axis,
        keepdims: workload.keepdims,
        input_elements_per_run: input_count,
        output_elements_per_run: output_count,
        bytes_touched_estimate_per_run: bytes_touched_estimate,
        thread_count,
        min_elements_per_chunk: options.min_elements_per_chunk,
        expected_chunk_count,
        serial,
        parallel,
        speedup,
        telemetry,
    })
}

struct WorkloadArrays {
    lhs: UFuncArray,
    rhs: Option<UFuncArray>,
}

impl WorkloadArrays {
    fn new(workload: &WorkloadSpec) -> Result<Self, String> {
        let lhs = make_array(&workload.shape)?;
        let rhs = workload.rhs_shape.as_deref().map(make_array).transpose()?;
        Ok(Self { lhs, rhs })
    }

    fn measure(
        &self,
        workload: &WorkloadSpec,
        config: ParallelPartitionConfig,
    ) -> Result<(f64, f64), String> {
        match workload.operation {
            CalibrationOperation::BroadcastAdd => {
                let rhs = self
                    .rhs
                    .as_ref()
                    .ok_or_else(|| "broadcast add workload missing rhs".to_string())?;
                measure_pair(
                    || {
                        self.lhs
                            .elementwise_binary(rhs, BinaryOp::Add)
                            .map_err(|err| format!("serial broadcast add failed: {err}"))
                    },
                    || {
                        self.lhs
                            .elementwise_binary_parallel(rhs, BinaryOp::Add, config)
                            .map_err(|err| format!("parallel broadcast add failed: {err}"))
                    },
                )
            }
            CalibrationOperation::AxisSum => measure_pair(
                || {
                    self.lhs
                        .reduce_sum(workload.axis, workload.keepdims)
                        .map_err(|err| format!("serial axis sum failed: {err}"))
                },
                || {
                    self.lhs
                        .reduce_sum_parallel(workload.axis, workload.keepdims, config)
                        .map_err(|err| format!("parallel axis sum failed: {err}"))
                },
            ),
        }
    }
}

fn make_array(shape: &[usize]) -> Result<UFuncArray, String> {
    let element_count = shape.iter().product::<usize>();
    let values = (0..element_count)
        .map(|idx| {
            let bounded = u32::try_from(idx % 257)
                .map_err(|err| format!("fixture value conversion failed: {err}"))?;
            Ok(f64::from(bounded))
        })
        .collect::<Result<Vec<_>, String>>()?;
    UFuncArray::new(shape.to_vec(), values, DType::F64)
        .map_err(|err| format!("failed creating workload array: {err}"))
}

fn measure_pair<S, P>(serial_fn: S, parallel_fn: P) -> Result<(f64, f64), String>
where
    S: Fn() -> Result<UFuncArray, String>,
    P: Fn() -> Result<UFuncArray, String>,
{
    let serial_start = Instant::now();
    let serial = serial_fn()?;
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1_000.0;
    std::hint::black_box(serial.values());

    let parallel_start = Instant::now();
    let parallel = parallel_fn()?;
    let parallel_ms = parallel_start.elapsed().as_secs_f64() * 1_000.0;
    std::hint::black_box(parallel.values());

    assert_same_output(&serial, &parallel)?;
    Ok((serial_ms, parallel_ms))
}

fn assert_same_output(serial: &UFuncArray, parallel: &UFuncArray) -> Result<(), String> {
    if serial.shape() != parallel.shape() {
        return Err(format!(
            "serial/parallel shape mismatch serial={:?} parallel={:?}",
            serial.shape(),
            parallel.shape()
        ));
    }
    if serial.values() != parallel.values() {
        return Err("serial/parallel value mismatch".to_string());
    }
    Ok(())
}

fn summarize_samples(mut samples_ms: Vec<f64>) -> Result<TimingSummary, String> {
    if samples_ms.is_empty() {
        return Err("cannot summarize zero timing samples".to_string());
    }
    samples_ms.sort_by(f64::total_cmp);
    let min_ms = samples_ms
        .first()
        .copied()
        .ok_or_else(|| "cannot read min from empty timing samples".to_string())?;
    let max_ms = samples_ms
        .last()
        .copied()
        .ok_or_else(|| "cannot read max from empty timing samples".to_string())?;
    Ok(TimingSummary {
        p50_ms: percentile(&samples_ms, 50, 100),
        p95_ms: percentile(&samples_ms, 95, 100),
        p99_ms: percentile(&samples_ms, 99, 100),
        min_ms,
        max_ms,
        samples_ms,
    })
}

fn percentile(sorted_samples: &[f64], numerator: usize, denominator: usize) -> f64 {
    let max_index = sorted_samples.len() - 1;
    let raw_index = max_index.saturating_mul(numerator).div_ceil(denominator);
    sorted_samples
        .get(raw_index.min(max_index))
        .copied()
        .unwrap_or(0.0)
}

fn ratio(serial_ms: f64, parallel_ms: f64) -> f64 {
    if parallel_ms <= 0.0 {
        return 0.0;
    }
    serial_ms / parallel_ms
}

fn compute_per_second(units_per_run: usize, sample_ms: f64) -> Result<f64, String> {
    if sample_ms <= 0.0 {
        return Ok(0.0);
    }
    Ok(usize_to_f64(units_per_run)? * 1_000.0 / sample_ms)
}

fn compute_mib_per_second(bytes_per_run: usize, sample_ms: f64) -> Result<f64, String> {
    let bytes_per_second = compute_per_second(bytes_per_run, sample_ms)?;
    Ok(bytes_per_second / 1_048_576.0)
}

fn usize_to_f64(value: usize) -> Result<f64, String> {
    let narrowed = u32::try_from(value).map_err(|err| {
        format!("calibration numeric value {value} exceeds f64-safe range: {err}")
    })?;
    Ok(f64::from(narrowed))
}

fn expected_chunk_count(total: usize, config: ParallelPartitionConfig) -> usize {
    if total == 0 {
        return 0;
    }
    let chunks_by_size = total.div_ceil(config.min_elements_per_chunk);
    config.worker_count.min(chunks_by_size).max(1)
}

fn normalize_axis(axis: isize, ndim: usize) -> Result<usize, String> {
    let ndim = isize::try_from(ndim).map_err(|_| "ndim does not fit isize".to_string())?;
    let normalized = if axis < 0 { ndim + axis } else { axis };
    if !(0..ndim).contains(&normalized) {
        return Err(format!("axis {axis} out of bounds for ndim {ndim}"));
    }
    usize::try_from(normalized).map_err(|_| "normalized axis does not fit usize".to_string())
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn available_parallelism() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

fn process_high_water_rss_bytes() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(raw) = line.strip_prefix("VmHWM:") {
            let kib = raw.split_whitespace().next()?.parse::<u64>().ok()?;
            return kib.checked_mul(1024);
        }
    }
    None
}

fn git_commit_short(repo_root: &Path) -> String {
    let output = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(repo_root)
        .output();
    match output {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).trim().to_string()
        }
        _ => "unknown".to_string(),
    }
}

fn write_report(path: &Path, report: &CalibrationReport) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating report dir {}: {err}", parent.display()))?;
    }
    let payload = serde_json::to_string_pretty(report)
        .map_err(|err| format!("failed serializing calibration report: {err}"))?;
    fs::write(path, payload).map_err(|err| format!("failed writing {}: {err}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_calibration_thread_counts_are_sorted_and_deduped() {
        assert_eq!(
            parse_thread_counts("4,2,4, 1").expect("parse"),
            vec![1, 2, 4]
        );
    }

    #[test]
    fn parallel_calibration_rejects_empty_thread_count_list() {
        let err = parse_thread_counts(" , ").expect_err("empty list should fail");
        assert!(err.contains("at least one"));
    }

    #[test]
    fn parallel_calibration_modes_select_deterministic_working_set_matrix() {
        let quick = calibration_workloads(CalibrationMode::Quick);
        let standard = calibration_workloads(CalibrationMode::Standard);
        let large = calibration_workloads(CalibrationMode::LargeHost);
        assert_eq!(quick.len(), 2);
        assert_eq!(standard.len(), 4);
        assert_eq!(large.len(), 6);
        assert!(
            standard
                .iter()
                .any(|workload| workload.workload_id == "broadcast_add_512x512_by_512")
        );
    }

    #[test]
    fn parallel_calibration_large_host_mode_expands_default_threads() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let options = Options::parse(["--large-host".to_string()], &repo_root).expect("options");
        assert_eq!(options.mode, CalibrationMode::LargeHost);
        assert_eq!(options.thread_counts, vec![2, 4, 8, 16, 32, 64]);
    }

    #[test]
    fn parallel_calibration_expected_chunk_count_respects_min_size() {
        let config = ParallelPartitionConfig::from_worker_count(8)
            .expect("worker count")
            .with_min_elements_per_chunk(10)
            .expect("chunk size");
        assert_eq!(expected_chunk_count(95, config), 8);
        assert_eq!(expected_chunk_count(9, config), 1);
        assert_eq!(expected_chunk_count(0, config), 0);
    }

    #[test]
    fn parallel_calibration_percentiles_use_sorted_ceiling_rank() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let summary = summarize_samples(samples).expect("summary");
        assert_eq!(summary.p50_ms, 3.0);
        assert_eq!(summary.p95_ms, 5.0);
        assert_eq!(summary.p99_ms, 5.0);
    }

    #[test]
    fn parallel_calibration_builds_report_for_tiny_matrix() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let options = Options {
            repo_root,
            mode: CalibrationMode::Quick,
            report_path: PathBuf::from("target/test_parallel_calibration.json"),
            sample_count: 1,
            thread_counts: vec![2],
            min_elements_per_chunk: 2,
        };
        let workloads = vec![
            WorkloadSpec {
                workload_id: "tiny_add".to_string(),
                operation: CalibrationOperation::BroadcastAdd,
                shape: vec![4, 4],
                rhs_shape: Some(vec![4]),
                axis: None,
                keepdims: false,
            },
            WorkloadSpec {
                workload_id: "tiny_reduce".to_string(),
                operation: CalibrationOperation::AxisSum,
                shape: vec![4, 4],
                rhs_shape: None,
                axis: Some(1),
                keepdims: false,
            },
        ];

        let report = build_report(&options, &workloads).expect("report");
        assert_eq!(report.schema_version, SCHEMA_VERSION);
        assert_eq!(report.workload_count, 2);
        assert_eq!(report.entries.len(), 2);
        assert_eq!(report.mode, "quick");
        assert!(
            report
                .entries
                .iter()
                .all(|entry| entry.thread_count == 2 && entry.serial.samples_ms.len() == 1)
        );
        assert!(
            report
                .entries
                .iter()
                .all(|entry| entry.telemetry.bytes_touched_estimate_per_run
                    == entry.bytes_touched_estimate_per_run)
        );
    }
}
