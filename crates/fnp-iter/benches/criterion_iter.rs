//! Focused performance benchmarks for iterator planning and chunk emission.

use criterion::{Criterion, Throughput, criterion_group};
use fnp_iter::{Nditer, NditerOptions, NditerOrder, NditerPlan, NditerStep};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::hint::black_box;
use std::time::{Duration, Instant};

const CONTRACT_ROUNDS: usize = 41;
const CONTRACT_MIN_OF: usize = 3;
const CONTRACT_BOOTSTRAP_RESAMPLES: usize = 4_096;

#[derive(Clone, Copy)]
struct TimedStep {
    elapsed: Duration,
    checksum: u64,
}

#[derive(Clone, Copy)]
struct PairStats {
    arm_a_median_ns: f64,
    arm_b_median_ns: f64,
    ratio_median: f64,
    ratio_ci_low: f64,
    ratio_ci_high: f64,
    ratio_cv_pct: f64,
    ratio_mad: f64,
    checksum: u64,
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

fn mix_checksum(state: u64, value: u64) -> u64 {
    state.rotate_left(11) ^ value.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0xa076_1d64_78bd_642f
}

fn checksum_step(step: &NditerStep) -> u64 {
    let state = step
        .multi_index
        .iter()
        .fold(step.iterindex as u64, |state, &value| {
            mix_checksum(state, value as u64)
        });
    step.linear_indices
        .iter()
        .fold(state, |state, &value| mix_checksum(state, value as u64))
}

fn time_step<F>(operation: F) -> TimedStep
where
    F: FnOnce() -> NditerStep,
{
    let started = Instant::now();
    let output = black_box(operation());
    let elapsed = started.elapsed();
    TimedStep {
        elapsed,
        checksum: checksum_step(&output),
    }
}

fn min_of<F>(arm: &mut F) -> TimedStep
where
    F: FnMut() -> TimedStep,
{
    let mut best = arm();
    let mut checksum = best.checksum;
    for _ in 1..CONTRACT_MIN_OF {
        let observation = arm();
        checksum = mix_checksum(checksum, observation.checksum);
        if observation.elapsed < best.elapsed {
            best.elapsed = observation.elapsed;
        }
    }
    TimedStep {
        elapsed: best.elapsed,
        checksum,
    }
}

fn paired<A, B>(round: usize, arm_a: &mut A, arm_b: &mut B) -> (TimedStep, TimedStep)
where
    A: FnMut() -> TimedStep,
    B: FnMut() -> TimedStep,
{
    let (a, b) = if round & 1 == 0 {
        (min_of(arm_a), min_of(arm_b))
    } else {
        let b = min_of(arm_b);
        (min_of(arm_a), b)
    };
    assert_eq!(
        a.checksum, b.checksum,
        "paired benchmark arms produced different output checksums"
    );
    (a, b)
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

fn pair_stats(arm_a: &[f64], arm_b: &[f64], checksum: u64) -> PairStats {
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

    PairStats {
        arm_a_median_ns,
        arm_b_median_ns,
        ratio_median,
        ratio_ci_low,
        ratio_ci_high,
        ratio_cv_pct: ratio_variance.sqrt() * 100.0 / ratio_mean,
        ratio_mad: median(&mut deviations),
        checksum,
    }
}

fn report_pair(row: &str, stats: PairStats) {
    println!(
        "PAIRED row={row} rounds={CONTRACT_ROUNDS} min_of={CONTRACT_MIN_OF} \
         arm_a_median_ms={:.6} arm_b_median_ms={:.6} ratio_median={:.6} \
         ratio_median_ci95=[{:.6},{:.6}] ratio_cv_pct={:.3} ratio_mad={:.6} \
         checksum={:016x}",
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

fn report_median_ci_gate(row: &str, effect: PairStats, null: PairStats) {
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

fn former_f_order_external_chunk(plan: NditerPlan) -> NditerStep {
    let iterindex = 0usize;
    let end = iterindex
        .checked_add(plan.inner_loop_len())
        .expect("former chunk end must fit");
    let linear_indices = (iterindex..end)
        .map(|index| {
            let multi_index = plan
                .linear_index_to_multi_index(index)
                .expect("former multi-index conversion must succeed");
            multi_index
                .iter()
                .enumerate()
                .try_fold(0usize, |linear, (axis, &coordinate)| {
                    linear
                        .checked_mul(plan.shape()[axis])
                        .and_then(|value| value.checked_add(coordinate))
                })
                .expect("former operand index must fit")
        })
        .collect();

    NditerStep {
        iterindex,
        multi_index: plan
            .linear_index_to_multi_index(iterindex)
            .expect("former chunk start must resolve"),
        linear_indices,
    }
}

fn public_f_order_external_chunk(plan: NditerPlan) -> NditerStep {
    Nditer::from_plan(plan)
        .next()
        .expect("F-order external-loop chunk must exist")
}

/// Frozen replica of the pre-`6807f4df` C-order chunk path. It decodes every
/// logical index to a fresh multi-index and folds those digits back into the
/// same row-major operand index.
fn former_c_order_external_chunk(plan: NditerPlan) -> NditerStep {
    let iterindex = 0usize;
    let end = iterindex
        .checked_add(plan.inner_loop_len())
        .expect("former chunk end must fit");
    let linear_indices = (iterindex..end)
        .map(|index| {
            let multi_index = plan
                .linear_index_to_multi_index(index)
                .expect("former C-order multi-index conversion must succeed");
            multi_index
                .iter()
                .enumerate()
                .try_fold(0usize, |linear, (axis, &coordinate)| {
                    linear
                        .checked_mul(plan.shape()[axis])
                        .and_then(|value| value.checked_add(coordinate))
                })
                .expect("former C-order operand index must fit")
        })
        .collect();

    NditerStep {
        iterindex,
        multi_index: plan
            .linear_index_to_multi_index(iterindex)
            .expect("former C-order chunk start must resolve"),
        linear_indices,
    }
}

fn public_c_order_external_chunk(plan: NditerPlan) -> NditerStep {
    Nditer::from_plan(plan)
        .next()
        .expect("C-order external-loop chunk must exist")
}

fn run_c_order_contract() {
    let plan = NditerPlan::new(
        vec![8, 8, 16, 64],
        8,
        NditerOptions {
            order: NditerOrder::C,
            external_loop: true,
        },
    )
    .expect("C-order external-loop plan must build");
    assert_eq!(
        public_c_order_external_chunk(plan.clone()),
        former_c_order_external_chunk(plan.clone())
    );

    for _ in 0..4 {
        black_box(time_step(|| former_c_order_external_chunk(plan.clone())));
        black_box(time_step(|| former_c_order_external_chunk(plan.clone())));
    }

    let mut null_a_samples = Vec::with_capacity(CONTRACT_ROUNDS);
    let mut null_b_samples = Vec::with_capacity(CONTRACT_ROUNDS);
    let mut null_checksum = 0u64;
    for round in 0..CONTRACT_ROUNDS {
        let mut base_a = || time_step(|| former_c_order_external_chunk(plan.clone()));
        let mut base_b = || time_step(|| former_c_order_external_chunk(plan.clone()));
        let (base_a, base_b) = paired(round, &mut base_a, &mut base_b);
        null_a_samples.push(base_a.elapsed.as_secs_f64() * 1.0e9);
        null_b_samples.push(base_b.elapsed.as_secs_f64() * 1.0e9);
        null_checksum = mix_checksum(null_checksum, base_a.checksum);
    }
    let null = pair_stats(&null_a_samples, &null_b_samples, null_checksum);
    report_pair("null_base_aa", null);

    for round in 0..4 {
        let mut former = || time_step(|| former_c_order_external_chunk(plan.clone()));
        let mut candidate = || time_step(|| public_c_order_external_chunk(plan.clone()));
        let _ = paired(round, &mut former, &mut candidate);
    }
    let mut former_samples = Vec::with_capacity(CONTRACT_ROUNDS);
    let mut candidate_samples = Vec::with_capacity(CONTRACT_ROUNDS);
    let mut effect_checksum = 0u64;
    for round in 0..CONTRACT_ROUNDS {
        let mut former = || time_step(|| former_c_order_external_chunk(plan.clone()));
        let mut candidate = || time_step(|| public_c_order_external_chunk(plan.clone()));
        let (former, candidate) = paired(round, &mut former, &mut candidate);
        former_samples.push(former.elapsed.as_secs_f64() * 1.0e9);
        candidate_samples.push(candidate.elapsed.as_secs_f64() * 1.0e9);
        effect_checksum = mix_checksum(effect_checksum, former.checksum);
    }
    let effect = pair_stats(&former_samples, &candidate_samples, effect_checksum);
    report_pair("effect_former_over_direct_range", effect);
    report_median_ci_gate("nditer_c_external_chunk", effect, null);
}

fn bench_nditer_c_external_chunk(c: &mut Criterion) {
    let plan = NditerPlan::new(
        vec![8, 8, 16, 64],
        8,
        NditerOptions {
            order: NditerOrder::C,
            external_loop: true,
        },
    )
    .expect("C-order external-loop plan must build");

    let mut group = c.benchmark_group("nditer_c_external_chunk");
    group.throughput(Throughput::Elements(plan.inner_loop_len() as u64));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.bench_function("former_index_round_trip", |bench| {
        bench.iter(|| black_box(former_c_order_external_chunk(black_box(plan.clone()))))
    });
    group.bench_function("public_direct_range", |bench| {
        bench.iter(|| black_box(public_c_order_external_chunk(black_box(plan.clone()))))
    });
    group.finish();
}

fn bench_nditer_f_external_chunk(c: &mut Criterion) {
    let plan = NditerPlan::new(
        vec![65_536, 2, 2, 2],
        8,
        NditerOptions {
            order: NditerOrder::F,
            external_loop: true,
        },
    )
    .expect("F-order external-loop plan must build");
    assert_eq!(
        public_f_order_external_chunk(plan.clone()),
        former_f_order_external_chunk(plan.clone())
    );

    let mut group = c.benchmark_group("nditer_f_external_chunk");
    group.throughput(Throughput::Elements(plan.inner_loop_len() as u64));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.bench_function("former_index_round_trip", |bench| {
        bench.iter(|| black_box(former_f_order_external_chunk(black_box(plan.clone()))))
    });
    group.bench_function("public_chunk_path", |bench| {
        bench.iter(|| black_box(public_f_order_external_chunk(black_box(plan.clone()))))
    });
    group.finish();
}

/// Replica of the current non-external F-order per-element step cost via the
/// public plan API: the step's `multi_index` decode PLUS the chunk fallback's
/// second decode-and-fold of the same iterindex (two Vec allocations and two
/// per-axis divmod loops per element), frozen so the single-decode A/B
/// isolates its lever inside one binary.
fn former_f_order_element_steps(plan: &NditerPlan) -> Vec<NditerStep> {
    (0..plan.element_count())
        .map(|iterindex| {
            let multi_index = plan
                .linear_index_to_multi_index(iterindex)
                .expect("former step multi-index must resolve");
            let second_decode = plan
                .linear_index_to_multi_index(iterindex)
                .expect("former operand decode must resolve");
            let linear = second_decode
                .iter()
                .enumerate()
                .try_fold(0usize, |acc, (axis, &coordinate)| {
                    acc.checked_mul(plan.shape()[axis])
                        .and_then(|value| value.checked_add(coordinate))
                })
                .expect("former operand index must fit");
            NditerStep {
                iterindex,
                multi_index,
                linear_indices: vec![linear],
            }
        })
        .collect()
}

fn public_f_order_element_steps(plan: &NditerPlan) -> Vec<NditerStep> {
    Nditer::from_plan(plan.clone()).collect()
}

/// The public iterator plus exactly the work the single-decode lever removed:
/// one additional multi-index decode (and its Vec) per element, matching the
/// former chunk fallback's second decode of the same iterindex. Identical
/// iterator machinery in both arms, so the A/B isolates the removed work.
fn former_model_f_order_element_steps(plan: &NditerPlan) -> Vec<NditerStep> {
    Nditer::from_plan(plan.clone())
        .inspect(|step| {
            black_box(
                plan.linear_index_to_multi_index(step.iterindex)
                    .expect("former-model second decode must resolve"),
            );
        })
        .collect()
}

fn bench_nditer_f_element_steps(c: &mut Criterion) {
    let plan = NditerPlan::new(
        vec![64, 64, 64],
        8,
        NditerOptions {
            order: NditerOrder::F,
            external_loop: false,
        },
    )
    .expect("F-order element-step plan must build");
    // Complete step-stream equality before timing.
    assert_eq!(
        public_f_order_element_steps(&plan),
        former_f_order_element_steps(&plan)
    );

    let mut group = c.benchmark_group("nditer_f_element_steps");
    group.throughput(Throughput::Elements(plan.element_count() as u64));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.bench_function("former_model_extra_decode", |bench| {
        bench.iter(|| black_box(former_model_f_order_element_steps(black_box(&plan))))
    });
    group.bench_function("public_step_path", |bench| {
        bench.iter(|| black_box(public_f_order_element_steps(black_box(&plan))))
    });
    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default().configure_from_args()
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets =
        bench_nditer_c_external_chunk,
        bench_nditer_f_external_chunk,
        bench_nditer_f_element_steps
}

fn main() {
    report_bench_identity();
    run_c_order_contract();
    benches();
    Criterion::default().configure_from_args().final_summary();
}
