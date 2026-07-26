//! Shared helpers for the split per-domain criterion bench binaries.
//!
//! Each per-domain bench binary pulls this in with
//! `#[path = "common/mod.rs"] mod common;`. The helpers are extracted verbatim
//! from the former monolithic `criterion_python_surface.rs` so that each
//! per-domain binary compiles only its own bench functions instead of forcing
//! the whole 200-plus-function monolith to compile just to run one group.
//!
//! `benches/common/mod.rs` is not itself a bench target: Cargo auto-discovers
//! `benches/*.rs` and `benches/*/main.rs`, and this is neither.
//!
//! Each per-domain bench binary that `#[path]`-includes this module uses only a
//! subset of the shared helpers, so items unused by a given binary are expected;
//! `#![allow(dead_code)]` keeps those honest, cross-binary-unused helpers from
//! tripping the `-D warnings` gate in the binaries that do not call them.
#![allow(dead_code)]

use criterion::Criterion;
use pyo3::types::PyAnyMethods;
use pyo3::{Bound, Py, PyAny, PyResult, Python};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::fmt::Write as _;
use std::hint::black_box;
use std::time::{Duration, Instant};

pub const CONTRACT_ROUNDS: usize = 41;
const CONTRACT_MIN_OF: usize = 3;
const CONTRACT_BOOTSTRAP_RESAMPLES: usize = 4_096;

#[derive(Clone, Copy)]
pub struct ContractPairStats {
    pub ratio_median: f64,
    pub ratio_ci_low: f64,
    pub ratio_ci_high: f64,
    pub ratio_cv_pct: f64,
    pub ratio_mad: f64,
    pub arm_a_median_ns: f64,
    pub arm_b_median_ns: f64,
    pub checksum: u64,
}

#[derive(Clone, Copy)]
pub struct ContractObservation {
    pub elapsed: Duration,
    pub checksum: u64,
}

fn self_identity() -> String {
    let Ok(path) = std::env::current_exe() else {
        return "unavailable".to_string();
    };
    let Ok(bytes) = std::fs::read(&path) else {
        return "unavailable".to_string();
    };
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let digest = hasher.finalize();
    let mut hash = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut hash, "{byte:02x}").expect("writing to String cannot fail");
    }
    format!("{} ({} bytes) {}", hash, bytes.len(), path.display())
}

fn report_bench_identity() {
    println!("bench_elf_sha256={}", self_identity());
    std::io::Write::flush(&mut std::io::stdout()).expect("flushing stdout cannot fail");
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() & 1 == 0 {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}

fn bootstrap_median_ci(values: &[f64]) -> (f64, f64) {
    let mut state = 0x4d59_5df4_d0f3_3173u64 ^ values.len() as u64;
    let mut resample = vec![0.0; values.len()];
    let mut medians = Vec::with_capacity(CONTRACT_BOOTSTRAP_RESAMPLES);
    for _ in 0..CONTRACT_BOOTSTRAP_RESAMPLES {
        for slot in &mut resample {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *slot = values[(state as usize) % values.len()];
        }
        medians.push(median(&mut resample));
    }
    medians.sort_by(f64::total_cmp);
    let low = CONTRACT_BOOTSTRAP_RESAMPLES * 25 / 1_000;
    let high = (CONTRACT_BOOTSTRAP_RESAMPLES * 975 / 1_000).min(CONTRACT_BOOTSTRAP_RESAMPLES - 1);
    (medians[low], medians[high])
}

fn mix_checksum(state: u64, value: u64) -> u64 {
    state.rotate_left(11) ^ value.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0xa076_1d64_78bd_642f
}

fn min_observation<F>(operation: &mut F) -> ContractObservation
where
    F: FnMut() -> ContractObservation,
{
    let mut best = operation();
    let mut checksum = best.checksum;
    for _ in 1..CONTRACT_MIN_OF {
        let observation = operation();
        checksum = mix_checksum(checksum, observation.checksum);
        if observation.elapsed < best.elapsed {
            best.elapsed = observation.elapsed;
        }
    }
    ContractObservation {
        elapsed: best.elapsed,
        checksum,
    }
}

fn contract_pair_stats(arm_a: &[f64], arm_b: &[f64], checksum: u64) -> ContractPairStats {
    let ratios = arm_a
        .iter()
        .zip(arm_b)
        .map(|(a, b)| a / b)
        .collect::<Vec<_>>();
    let mut arm_a_sorted = arm_a.to_vec();
    let mut arm_b_sorted = arm_b.to_vec();
    let mut ratio_sorted = ratios.clone();
    let arm_a_median_ns = median(&mut arm_a_sorted);
    let arm_b_median_ns = median(&mut arm_b_sorted);
    let ratio_median = median(&mut ratio_sorted);
    let (ratio_ci_low, ratio_ci_high) = bootstrap_median_ci(&ratios);
    let ratio_mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
    let ratio_variance = ratios
        .iter()
        .map(|ratio| {
            let delta = ratio - ratio_mean;
            delta * delta
        })
        .sum::<f64>()
        / (ratios.len() - 1) as f64;
    let mut deviations = ratios
        .iter()
        .map(|ratio| (ratio - ratio_median).abs())
        .collect::<Vec<_>>();

    ContractPairStats {
        ratio_median,
        ratio_ci_low,
        ratio_ci_high,
        ratio_cv_pct: ratio_variance.sqrt() * 100.0 / ratio_mean,
        ratio_mad: median(&mut deviations),
        arm_a_median_ns,
        arm_b_median_ns,
        checksum,
    }
}

fn report_contract_pair(row: &str, stats: ContractPairStats) {
    println!(
        "PAIRED row={row} rounds={CONTRACT_ROUNDS} min_of={CONTRACT_MIN_OF} \
         arm_a_median_ms={:.6} arm_b_median_ms={:.6} ratio_median={:.6} \
         ratio_median_ci95=[{:.6},{:.6}] ratio_cv_pct={:.3} ratio_mad={:.6} checksum={:016x}",
        stats.arm_a_median_ns / 1_000_000.0,
        stats.arm_b_median_ns / 1_000_000.0,
        stats.ratio_median,
        stats.ratio_ci_low,
        stats.ratio_ci_high,
        stats.ratio_cv_pct,
        stats.ratio_mad,
        stats.checksum,
    );
}

fn report_contract_gate(row: &str, effect: ContractPairStats, null: ContractPairStats) {
    let null_half_width = (null.ratio_ci_low - 1.0)
        .abs()
        .max((null.ratio_ci_high - 1.0).abs());
    let required_delta = (2.0 * null_half_width).max(0.01);
    let effect_delta = effect.ratio_median - 1.0;
    let outside_null_ci =
        effect.ratio_median < null.ratio_ci_low || effect.ratio_median > null.ratio_ci_high;
    let verdict = if outside_null_ci && effect_delta >= required_delta {
        "DECIDABLE_WIN"
    } else if outside_null_ci && effect_delta <= -required_delta {
        "DECIDABLE_REGRESSION"
    } else {
        "UNDECIDED"
    };
    println!(
        "MEDIAN_CI_GATE row={row} verdict={verdict} effect_ratio={:.6} \
         null_ci95=[{:.6},{:.6}] null_half_width={:.6} required_2x_delta={required_delta:.6} \
         cv_is_provenance_only=true",
        effect.ratio_median, null.ratio_ci_low, null.ratio_ci_high, null_half_width,
    );
}

/// Run a base/base null first, then an interleaved base/candidate effect in one
/// process. Callers prove output parity before entering this timer.
pub fn run_median_ci_contract<A, B>(
    row: &str,
    mut former: A,
    mut candidate: B,
) -> (ContractPairStats, ContractPairStats)
where
    A: FnMut() -> ContractObservation,
    B: FnMut() -> ContractObservation,
{
    for _ in 0..4 {
        black_box(min_observation(&mut former));
        black_box(min_observation(&mut former));
    }

    // Contract order is deliberate: establish the base/base null before the
    // candidate can perturb caches, allocator state, or worker scheduling.
    let mut null_a_samples = Vec::with_capacity(CONTRACT_ROUNDS);
    let mut null_b_samples = Vec::with_capacity(CONTRACT_ROUNDS);
    let mut null_checksum = 0_u64;
    for round in 0..CONTRACT_ROUNDS {
        let (a, b) = if round & 1 == 0 {
            (min_observation(&mut former), min_observation(&mut former))
        } else {
            let b = min_observation(&mut former);
            (min_observation(&mut former), b)
        };
        assert_eq!(
            a.checksum, b.checksum,
            "base/base null arms produced different output checksums"
        );
        null_a_samples.push(a.elapsed.as_secs_f64() * 1.0e9);
        null_b_samples.push(b.elapsed.as_secs_f64() * 1.0e9);
        null_checksum = mix_checksum(null_checksum, a.checksum);
    }
    let null = contract_pair_stats(&null_a_samples, &null_b_samples, null_checksum);
    report_contract_pair("null_base_aa", null);

    for round in 0..4 {
        if round & 1 == 0 {
            black_box(min_observation(&mut former));
            black_box(min_observation(&mut candidate));
        } else {
            black_box(min_observation(&mut candidate));
            black_box(min_observation(&mut former));
        }
    }
    let mut former_samples = Vec::with_capacity(CONTRACT_ROUNDS);
    let mut candidate_samples = Vec::with_capacity(CONTRACT_ROUNDS);
    let mut effect_checksum = 0_u64;
    for round in 0..CONTRACT_ROUNDS {
        let (former_elapsed, candidate_elapsed) = if round & 1 == 0 {
            (
                min_observation(&mut former),
                min_observation(&mut candidate),
            )
        } else {
            let candidate_elapsed = min_observation(&mut candidate);
            (min_observation(&mut former), candidate_elapsed)
        };
        assert_eq!(
            former_elapsed.checksum, candidate_elapsed.checksum,
            "base/candidate arms produced different output checksums"
        );
        former_samples.push(former_elapsed.elapsed.as_secs_f64() * 1.0e9);
        candidate_samples.push(candidate_elapsed.elapsed.as_secs_f64() * 1.0e9);
        effect_checksum = mix_checksum(effect_checksum, former_elapsed.checksum);
    }
    let effect = contract_pair_stats(&former_samples, &candidate_samples, effect_checksum);
    report_contract_pair("effect_former_over_candidate", effect);
    report_contract_gate(row, effect, null);
    (effect, null)
}

/// Import numpy on the interpreter, mapping the module handle away; every bench
/// group calls this before allocating its inputs so a missing numpy fails loud.
pub fn ensure_numpy_available(py: Python<'_>) -> PyResult<()> {
    py.import("numpy").map(drop)
}

/// Mean and coefficient-of-variation (percent) over the last <=10 retained
/// paired samples. Panics below two samples, which Criterion always retains.
pub fn ledger_tail_stats(samples: &RefCell<Vec<f64>>) -> (usize, f64, f64) {
    let samples = samples.borrow();
    let count = samples.len().min(10);
    assert!(
        count >= 2,
        "Criterion must retain at least two paired samples"
    );
    let tail = &samples[samples.len() - count..];
    let mean = tail.iter().sum::<f64>() / count as f64;
    let variance = tail
        .iter()
        .map(|sample| {
            let delta = sample - mean;
            delta * delta
        })
        .sum::<f64>()
        / (count - 1) as f64;
    (count, mean, variance.sqrt() * 100.0 / mean)
}

/// Emit the paired `LEDGER_AUDIT` line the negative-evidence flow parses:
/// candidate/orig means (ms), CVs (%), and the orig/candidate ratio.
pub fn report_ledger_pair(
    row: &str,
    candidate_samples: &RefCell<Vec<f64>>,
    orig_samples: &RefCell<Vec<f64>>,
) {
    if candidate_samples.borrow().is_empty() && orig_samples.borrow().is_empty() {
        return;
    }
    let (candidate_n, candidate_ns, candidate_cv) = ledger_tail_stats(candidate_samples);
    let (orig_n, orig_ns, orig_cv) = ledger_tail_stats(orig_samples);
    assert_eq!(candidate_n, orig_n);
    println!(
        "LEDGER_AUDIT row={row} samples={candidate_n} candidate_mean_ms={:.6} \
         candidate_cv_pct={candidate_cv:.3} orig_mean_ms={:.6} orig_cv_pct={orig_cv:.3} \
         orig_over_candidate={:.4}",
        candidate_ns / 1_000_000.0,
        orig_ns / 1_000_000.0,
        orig_ns / candidate_ns,
    );
}

/// `FNP_BENCH_GROUPS`, when set to a comma-separated substring list, restricts a
/// run to the group functions whose names contain one of the tokens; unset
/// preserves run-everything behavior. Identical semantics to the former
/// monolith gate, so existing reproduction commands keep working.
pub fn group_enabled(group_fn_name: &str) -> bool {
    let Ok(spec) = std::env::var("FNP_BENCH_GROUPS") else {
        return true;
    };
    spec.split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .any(|token| group_fn_name.contains(token))
}

/// A named bench group: the group function's name (for `FNP_BENCH_GROUPS`
/// gating) paired with the function itself.
pub type BenchGroup = (&'static str, fn(&mut Criterion));

/// Drive the selected bench group functions under one `Criterion`, then emit the
/// final summary. Mirrors the former `gated_benches!` macro's `main`: each entry
/// is `(group_fn_name, group_fn)`, gated by [`group_enabled`].
pub fn gated_main(targets: &[BenchGroup]) {
    // Line one, before Criterion is constructed: Criterion may print its
    // backend notice during construction.
    report_bench_identity();
    let mut criterion = Criterion::default().configure_from_args();
    for (name, target) in targets {
        if group_enabled(name) {
            target(&mut criterion);
        }
    }
    Criterion::default().configure_from_args().final_summary();
}

// Ledger-integrity retries for three historical REJECT rows. These helpers live only in the
// benchmark binary: production dispatch is deliberately untouched. `inline(never)` gives perf
// an exact execution marker for each reconstructed candidate and each NumPy ORIG reference.
#[inline]
pub fn ledger_f64_sortable_key(value: f64) -> u64 {
    let bits = if value == 0.0 { 0 } else { value.to_bits() };
    bits ^ ((((bits as i64) >> 63) as u64) | 0x8000_0000_0000_0000)
}

#[inline]
pub fn ledger_f64_from_sortable_key(key: u64) -> f64 {
    let bits = if key & 0x8000_0000_0000_0000 != 0 {
        key ^ 0x8000_0000_0000_0000
    } else {
        !key
    };
    f64::from_bits(bits)
}

#[inline(never)]
pub fn ledger_radix_select_key(mut current: Vec<u64>, mut rank: usize, start_byte: i32) -> u64 {
    let mut byte = start_byte;
    loop {
        let len = current.len();
        if len <= 1 || byte < 0 {
            return current[rank];
        }
        let shift = (byte as u64) * 8;
        let histogram: [usize; 256] = if len > (1 << 16) {
            let chunk_size = (len / (rayon::current_num_threads() * 4).max(1)).max(1);
            current
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut local = [0usize; 256];
                    for &key in chunk {
                        local[((key >> shift) & 0xff) as usize] += 1;
                    }
                    local
                })
                .reduce(
                    || [0usize; 256],
                    |mut left, right| {
                        for digit in 0..256 {
                            left[digit] += right[digit];
                        }
                        left
                    },
                )
        } else {
            let mut local = [0usize; 256];
            for &key in &current {
                local[((key >> shift) & 0xff) as usize] += 1;
            }
            local
        };
        let mut prefix = 0usize;
        let mut selected = 255usize;
        for (digit, &count) in histogram.iter().enumerate() {
            if prefix + count > rank {
                selected = digit;
                break;
            }
            prefix += count;
        }
        current = if len > (1 << 16) {
            current
                .par_iter()
                .copied()
                .filter(|&key| ((key >> shift) & 0xff) as usize == selected)
                .collect()
        } else {
            current
                .iter()
                .copied()
                .filter(|&key| ((key >> shift) & 0xff) as usize == selected)
                .collect()
        };
        rank -= prefix;
        byte -= 1;
    }
}

#[inline(never)]
pub fn ledger_radix_median_f64(data: &[f64]) -> f64 {
    assert!(!data.par_iter().any(|value| value.is_nan()));
    let keys: Vec<u64> = data
        .par_iter()
        .map(|&value| ledger_f64_sortable_key(value))
        .collect();
    let n = keys.len();
    if n % 2 == 1 {
        ledger_f64_from_sortable_key(ledger_radix_select_key(keys, n / 2, 7))
    } else {
        let low = ledger_f64_from_sortable_key(ledger_radix_select_key(keys.clone(), n / 2 - 1, 7));
        let high = ledger_f64_from_sortable_key(ledger_radix_select_key(keys, n / 2, 7));
        (low + high) / 2.0
    }
}

#[inline(never)]
pub fn ledger_orig_median_reference(
    numpy_median: &Bound<'_, PyAny>,
    input: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    numpy_median.call1((input,))?.extract()
}

#[inline(never)]
pub fn ledger_try_native_f16_sort(
    numpy_sort: &Bound<'_, PyAny>,
    input: &Bound<'_, PyAny>,
    input_bits: &[u16],
) -> PyResult<Py<PyAny>> {
    let must_defer = input_bits
        .par_iter()
        .any(|&bits| bits == 0x8000 || ((bits & 0x7c00) == 0x7c00 && (bits & 0x03ff) != 0));
    assert!(
        !must_defer,
        "finite positive f16 audit input must stay on candidate route"
    );
    let widened = input.call_method1("astype", ("float32",))?;
    let sorted = numpy_sort.call1((&widened,))?;
    Ok(sorted.call_method1("astype", ("float16",))?.unbind())
}

#[inline(never)]
pub fn ledger_orig_f16_sort_reference(
    numpy_sort: &Bound<'_, PyAny>,
    input: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    Ok(numpy_sort.call1((input,))?.unbind())
}

#[inline(never)]
pub fn ledger_f32_tie_argsort_candidate(
    fnp_argsort: &Bound<'_, PyAny>,
    input: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    Ok(fnp_argsort.call1((input,))?.unbind())
}

#[inline(never)]
pub fn ledger_orig_f32_argsort_reference(
    numpy_argsort: &Bound<'_, PyAny>,
    input: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    Ok(numpy_argsort.call1((input,))?.unbind())
}

pub fn report_substrate_v2_pair(
    row: &str,
    candidate_samples: &RefCell<Vec<f64>>,
    orig_samples: &RefCell<Vec<f64>>,
) {
    if candidate_samples.borrow().is_empty() && orig_samples.borrow().is_empty() {
        return;
    }
    let (candidate_n, candidate_ns, candidate_cv) = ledger_tail_stats(candidate_samples);
    let (orig_n, orig_ns, orig_cv) = ledger_tail_stats(orig_samples);
    assert_eq!(candidate_n, orig_n);
    println!(
        "SUBSTRATE_V2 row={row} samples={candidate_n} candidate_mean_ms={:.6} \
         candidate_cv_pct={candidate_cv:.3} orig_mean_ms={:.6} orig_cv_pct={orig_cv:.3} \
         orig_over_candidate={:.4}",
        candidate_ns / 1_000_000.0,
        orig_ns / 1_000_000.0,
        orig_ns / candidate_ns,
    );
}

pub const MEDIAN_GATE_FINAL_BATCHES: usize = 10;
pub const MEDIAN_GATE_OBSERVATIONS_PER_BATCH: usize = 2;

#[derive(Clone, Copy)]
pub struct MedianGateDistribution {
    median: f64,
    p10: f64,
    p90: f64,
    low: f64,
    high: f64,
    cv_pct: f64,
    above_one: usize,
}

pub fn median_gate_quantile(sorted: &[f64], quantile: f64) -> f64 {
    assert!(!sorted.is_empty());
    let position = quantile * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = position - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

pub fn median_gate_distribution(samples: &[f64]) -> MedianGateDistribution {
    assert!(samples.len() >= 2);
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples
        .iter()
        .map(|sample| {
            let delta = sample - mean;
            delta * delta
        })
        .sum::<f64>()
        / (samples.len() - 1) as f64;
    MedianGateDistribution {
        median: median_gate_quantile(&sorted, 0.5),
        p10: median_gate_quantile(&sorted, 0.1),
        p90: median_gate_quantile(&sorted, 0.9),
        low: sorted[0],
        high: sorted[sorted.len() - 1],
        cv_pct: variance.sqrt() * 100.0 / mean,
        above_one: samples.iter().filter(|&&ratio| ratio > 1.0).count(),
    }
}

pub fn median_gate_tail(samples: &RefCell<Vec<f64>>) -> Vec<f64> {
    let samples = samples.borrow();
    let retained = MEDIAN_GATE_FINAL_BATCHES * MEDIAN_GATE_OBSERVATIONS_PER_BATCH;
    assert!(
        samples.len() >= retained,
        "Criterion must retain {retained} median-gate observations"
    );
    samples[samples.len() - retained..].to_vec()
}

pub fn report_median_gate_pair(
    row: &str,
    null_base_ns: &RefCell<Vec<f64>>,
    null_peer_ns: &RefCell<Vec<f64>>,
    null_ratios: &RefCell<Vec<f64>>,
    base_ns: &RefCell<Vec<f64>>,
    candidate_ns: &RefCell<Vec<f64>>,
    effect_ratios: &RefCell<Vec<f64>>,
) {
    if effect_ratios.borrow().is_empty() {
        return;
    }
    let null_base = median_gate_tail(null_base_ns);
    let null_peer = median_gate_tail(null_peer_ns);
    let null = median_gate_distribution(&median_gate_tail(null_ratios));
    let base = median_gate_distribution(&median_gate_tail(base_ns));
    let candidate = median_gate_distribution(&median_gate_tail(candidate_ns));
    let effect = median_gate_distribution(&median_gate_tail(effect_ratios));
    let null_brackets_one = null.p10 <= 1.0 && null.p90 >= 1.0;
    let verdict = if !null_brackets_one {
        "BIASED_NULL"
    } else if effect.median > null.p90 {
        "WIN"
    } else if effect.median < null.p10 {
        "PROFILE_REQUIRED"
    } else {
        "UNDECIDED"
    };
    let null_base_cv = median_gate_distribution(&null_base).cv_pct;
    let null_peer_cv = median_gate_distribution(&null_peer).cv_pct;
    println!(
        "NULL_MEDIAN_GATE row={row} observations={} base_median_ms={:.6} \
         candidate_median_ms={:.6} base_cv_pct={:.3} candidate_cv_pct={:.3} \
         effect_median={:.6} effect_p10={:.6} effect_p90={:.6} \
         effect_low={:.6} effect_high={:.6} effect_cv_pct={:.3} effect_above_one={} \
         null_median={:.6} null_p10={:.6} null_p90={:.6} null_low={:.6} \
         null_high={:.6} null_cv_pct={:.3} null_base_cv_pct={:.3} \
         null_peer_cv_pct={:.3} null_corrected_median={:.6} verdict={verdict}",
        effect_ratios
            .borrow()
            .len()
            .min(MEDIAN_GATE_FINAL_BATCHES * MEDIAN_GATE_OBSERVATIONS_PER_BATCH),
        base.median / 1_000_000.0,
        candidate.median / 1_000_000.0,
        base.cv_pct,
        candidate.cv_pct,
        effect.median,
        effect.p10,
        effect.p90,
        effect.low,
        effect.high,
        effect.cv_pct,
        effect.above_one,
        null.median,
        null.p10,
        null.p90,
        null.low,
        null.high,
        null.cv_pct,
        null_base_cv,
        null_peer_cv,
        effect.median / null.median,
    );
}

pub fn time_python_binary_call<'py>(
    function: &Bound<'py, PyAny>,
    lhs: &Bound<'py, PyAny>,
    rhs: &Bound<'py, PyAny>,
) -> Duration {
    let start = Instant::now();
    let function = black_box(function);
    let lhs = black_box(lhs);
    let rhs = black_box(rhs);
    let result = function
        .call1((lhs, rhs))
        .expect("median-gate binary Python call");
    drop(black_box(result));
    start.elapsed()
}

pub fn time_python_unary_call<'py>(
    function: &Bound<'py, PyAny>,
    input: &Bound<'py, PyAny>,
) -> Duration {
    let start = Instant::now();
    let function = black_box(function);
    let input = black_box(input);
    let result = function
        .call1((input,))
        .expect("median-gate unary Python call");
    drop(black_box(result));
    start.elapsed()
}
