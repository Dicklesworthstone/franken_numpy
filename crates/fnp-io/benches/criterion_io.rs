//! Criterion benchmarks for fnp-io.
//!
//! Measures performance baselines for I/O operations:
//! - write_npy_bytes: serialize array to .npy format
//! - read_npy_bytes: deserialize array from .npy format
//! - write_npz_bytes: serialize multiple arrays to .npz archive
//! - read_npz_bytes: deserialize .npz archive
//!
//! These operations are critical for data persistence workflows.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group};
use fnp_io::{
    IOSupportedDType, NpyHeader, StructuredIODescriptor, StructuredIOField, StructuredNpyData,
    fromfile, fromfile_structured, fromfile_text, load, read_npy_bytes, read_npz_bytes,
    read_npz_bytes_linear_overlap_control, write_npy_bytes, write_npz_bytes,
};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::fmt::Write as _;
use std::hint::black_box;
use std::time::{Duration, Instant};

const CONTRACT_ROUNDS: usize = 41;
const CONTRACT_MIN_OF: usize = 3;
const CONTRACT_BOOTSTRAP_RESAMPLES: usize = 4_096;

#[derive(Clone, Copy)]
struct TimedValue {
    elapsed: Duration,
    checksum: u64,
}

#[derive(Clone, Copy)]
struct PairStats {
    rounds: usize,
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

fn checksum_text_array(output: &fnp_io::TextArrayData) -> u64 {
    output.values.iter().fold(
        mix_checksum(output.nrows as u64, output.ncols as u64),
        |state, value| mix_checksum(state, value.to_bits()),
    )
}

fn min_of<F>(arm: &mut F) -> TimedValue
where
    F: FnMut() -> TimedValue,
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
    TimedValue {
        elapsed: best.elapsed,
        checksum,
    }
}

fn paired<A, B>(round: usize, arm_a: &mut A, arm_b: &mut B) -> (TimedValue, TimedValue)
where
    A: FnMut() -> TimedValue,
    B: FnMut() -> TimedValue,
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

fn pair_stats(
    arm_a_samples: &RefCell<Vec<f64>>,
    arm_b_samples: &RefCell<Vec<f64>>,
    checksum: u64,
) -> Option<PairStats> {
    let arm_a_samples = arm_a_samples.borrow();
    let arm_b_samples = arm_b_samples.borrow();
    let count = arm_a_samples
        .len()
        .min(arm_b_samples.len())
        .min(CONTRACT_ROUNDS);
    if count < CONTRACT_ROUNDS {
        return None;
    }
    let arm_a = &arm_a_samples[arm_a_samples.len() - count..];
    let arm_b = &arm_b_samples[arm_b_samples.len() - count..];
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
    let ratio_mean = ratios.iter().sum::<f64>() / count as f64;
    let ratio_variance = ratios
        .iter()
        .map(|ratio| {
            let delta = ratio - ratio_mean;
            delta * delta
        })
        .sum::<f64>()
        / (count - 1) as f64;
    let mut deviations = ratios
        .iter()
        .map(|ratio| (ratio - ratio_median).abs())
        .collect::<Vec<_>>();

    Some(PairStats {
        rounds: count,
        arm_a_median_ns,
        arm_b_median_ns,
        ratio_median,
        ratio_ci_low,
        ratio_ci_high,
        ratio_cv_pct: ratio_variance.sqrt() * 100.0 / ratio_mean,
        ratio_mad: median(&mut deviations),
        checksum,
    })
}

fn report_pair(
    row: &str,
    arm_a_samples: &RefCell<Vec<f64>>,
    arm_b_samples: &RefCell<Vec<f64>>,
    checksum: u64,
) -> Option<PairStats> {
    let stats = pair_stats(arm_a_samples, arm_b_samples, checksum)?;
    println!(
        "PAIRED row={row} rounds={} min_of={CONTRACT_MIN_OF} arm_a_median_ms={:.6} \
         arm_b_median_ms={:.6} ratio_median={:.6} ratio_median_ci95=[{:.6},{:.6}] \
         ratio_cv_pct={:.3} ratio_mad={:.6} checksum={:016x}",
        stats.rounds,
        stats.arm_a_median_ns / 1_000_000.0,
        stats.arm_b_median_ns / 1_000_000.0,
        stats.ratio_median,
        stats.ratio_ci_low,
        stats.ratio_ci_high,
        stats.ratio_cv_pct,
        stats.ratio_mad,
        stats.checksum,
    );
    Some(stats)
}

fn null_half_width_of(null: PairStats) -> f64 {
    (null.ratio_ci_low - 1.0)
        .abs()
        .max((null.ratio_ci_high - 1.0).abs())
}

/// Does this A/A null's 95% CI actually contain 1.0? (`deadlock-audit-7xcq2`)
///
/// An A/A null runs the SAME arm against ITSELF, so its ratio must be 1.0. A CI that
/// EXCLUDES unity means the harness systematically favoured one position in the
/// schedule, and the effect measured in that same position inherits the bias.
///
/// THIS IS A VISIBILITY FIELD, NOT A VETO, and the distinction is worth stating because
/// the reasoning that first motivated it was wrong. The claim was that the gate is BLIND
/// to bias, on the theory that it scales its threshold by the null's half-width, so a
/// biased-but-TIGHT null would slip through on a small threshold. False:
/// `null_half_width_of` measures the distance from **1.0** to the furthest CI bound, not
/// the width of the interval, so a displaced null yields a LARGER half-width and a
/// STRICTER threshold. The gate already charges the bias.
///
/// What was actually missing is that nothing in the emitted line SAID a null was biased.
/// Both instances observed on 2026-08-16 were banked with hand-written prose caveats
/// instead of a machine-readable flag, and the ledger carries 116 prose `ci95=` mentions
/// against ZERO pasted null rows - so nobody can count how often this happens. These
/// fields make that population exist from here on.
fn null_straddles_unity(null: PairStats) -> bool {
    null.ratio_ci_low <= 1.0 && null.ratio_ci_high >= 1.0
}

/// How far the null's CENTRE sits from unity - the quantity the half-width misses.
fn null_bias(null: PairStats) -> f64 {
    (null.ratio_median - 1.0).abs()
}

/// The verdict `fnp-python`'s canonical `contract_gate_verdict` would return for this pair.
///
/// THE DIVERGENCE THIS EXISTS TO EXPOSE (`deadlock-audit-7xcq2`): the local rule in
/// `report_median_ci_gate` below never consults the EFFECT's own CI - only its median,
/// against the null envelope. The canonical rule in
/// `crates/fnp-python/benches/common/mod.rs` additionally requires
/// `effect.ratio_ci_low > 1.0` for a WIN (and `ratio_ci_high < 1.0` for a REGRESSION).
/// So a row whose effect CI STRADDLES unity - an effect not statistically distinguishable
/// from no-change - is stamped DECIDABLE_WIN by this file and UNDECIDED by `fnp-python`,
/// from identical numbers. `verify_median_ci_gate_semantics` pins a worked instance.
///
/// NOTE WHICH CI THIS IS. The bead that opened this line looked for the hole in the
/// NULL's CI and correctly found none - the null's bias is already charged into the
/// threshold. The hole is in the EFFECT's CI, and it is not in `fnp-python` at all; it is
/// in the three crates (fnp-io, fnp-iter, fnp-random) that carry this byte-identical weak
/// copy of the gate.
///
/// EMITTED ALONGSIDE the local verdict rather than replacing it. An unknown number of
/// banked rows rest on the weak rule - unknown precisely because gate lines are almost
/// never pasted into the ledger - so silently reversing them is worse than flagging them.
/// Harmonising the rule is a separate decision that needs its own registration and a
/// re-run, not a side effect of adding a field.
fn strict_gate_verdict(effect: PairStats, null: PairStats) -> &'static str {
    let required_delta = (2.0 * null_half_width_of(null)).max(0.01);
    let effect_delta = effect.ratio_median - 1.0;
    let effect_ci_above_one = effect.ratio_ci_low > 1.0;
    let effect_ci_below_one = effect.ratio_ci_high < 1.0;
    let above_null_envelope = effect.ratio_median > null.ratio_ci_high;
    let below_null_envelope = effect.ratio_median < null.ratio_ci_low;
    if effect_ci_above_one && above_null_envelope && effect_delta >= required_delta {
        "DECIDABLE_WIN"
    } else if effect_ci_below_one && below_null_envelope && effect_delta <= -required_delta {
        "DECIDABLE_REGRESSION"
    } else {
        "UNDECIDED"
    }
}

fn report_median_ci_gate(row: &str, effect: PairStats, null: PairStats) {
    let null_half_width = null_half_width_of(null);
    let required_delta = (2.0 * null_half_width).max(0.01);
    let effect_delta = effect.ratio_median - 1.0;
    let outside_null_ci =
        effect.ratio_median < null.ratio_ci_low || effect.ratio_median > null.ratio_ci_high;
    // HARMONISED 2026-08-17 (deadlock-audit-7xcq2): the verdict is now the SAME rule
    // fnp-python's contract_gate_verdict applies, so a row's verdict no longer depends on
    // which crate measured it. Decided on 19 real gate rows (10 live + 9 historical) scored
    // under BOTH rules with ZERO divergences, so harmonising rejects nothing that was ever
    // observed to pass, and the bead's stated fear - "a gate that starts rejecting
    // historical rows wholesale" - is empirically absent here. The old rule is still
    // computed and emitted as verdict_legacy_weak so banked rows stay comparable.
    let _ = (outside_null_ci, effect_delta);
    let verdict = strict_gate_verdict(effect, null);
    println!(
        "MEDIAN_CI_GATE row={row} verdict={verdict} effect_ratio={:.6} \
         effect_ci95=[{:.6},{:.6}] effect_ci_excludes_one={} \
         null_ci95=[{:.6},{:.6}] null_half_width={:.6} required_2x_delta={required_delta:.6} \
         null_straddles_unity={} null_bias={:.6} verdict_legacy_weak={} \
         gate_rule=canonical_matches_fnp_python \
         cv_is_provenance_only=true",
        effect.ratio_median,
        effect.ratio_ci_low,
        effect.ratio_ci_high,
        effect.ratio_ci_low > 1.0 || effect.ratio_ci_high < 1.0,
        null.ratio_ci_low,
        null.ratio_ci_high,
        null_half_width,
        null_straddles_unity(null),
        null_bias(null),
        legacy_weak_verdict(effect, null),
    );
}

/// The rule these files used BEFORE 2026-08-17, kept ONLY so banked rows stay comparable.
///
/// It never consults the EFFECT's CI - only its median, against the null envelope - so it
/// can stamp DECIDABLE_WIN on an effect that is not statistically distinguishable from
/// no-change. It is no longer the verdict; it is emitted as verdict_legacy_weak so a reader
/// holding an old row can tell whether that row would still pass under today's rule.
fn legacy_weak_verdict(effect: PairStats, null: PairStats) -> &'static str {
    let required_delta = (2.0 * null_half_width_of(null)).max(0.01);
    let effect_delta = effect.ratio_median - 1.0;
    let outside_null_ci =
        effect.ratio_median < null.ratio_ci_low || effect.ratio_median > null.ratio_ci_high;
    if outside_null_ci && effect_delta >= required_delta {
        "DECIDABLE_WIN"
    } else if outside_null_ci && effect_delta <= -required_delta {
        "DECIDABLE_REGRESSION"
    } else {
        "UNDECIDED"
    }
}

/// Builds a `PairStats` carrying only the four fields the gate reads, for the self-check.
fn gate_probe_stats(median: f64, ci_low: f64, ci_high: f64) -> PairStats {
    PairStats {
        rounds: 8,
        arm_a_median_ns: 1000.0,
        arm_b_median_ns: 1000.0,
        ratio_median: median,
        ratio_ci_low: ci_low,
        ratio_ci_high: ci_high,
        ratio_cv_pct: 0.0,
        ratio_mad: 0.0,
        checksum: 0x7c02,
    }
}

/// Pins the null-quality fields and the local-vs-canonical verdict divergence
/// (`deadlock-audit-7xcq2`).
///
/// Runs from `main`, not as a `#[test]`: this bench sets `harness = false`, so a `#[test]`
/// here would never execute on any real run and would prove nothing.
fn verify_median_ci_gate_semantics() {
    // A null displaced off unity but TIGHT, against one centred on unity but WIDER. Both
    // sets of bounds are the two instances observed on thinkstation1 on 2026-08-16.
    let biased_tight = gate_probe_stats(1.030928, 1.030405, 1.034364);
    let healthy_wide = gate_probe_stats(1.000000, 0.976336, 1.032376);
    assert!(
        !null_straddles_unity(biased_tight),
        "a null CI of [1.030405,1.034364] excludes unity and must be flagged"
    );
    assert!(
        null_straddles_unity(healthy_wide),
        "a null CI of [0.976336,1.032376] contains unity and must not be flagged"
    );
    assert!(
        null_straddles_unity(gate_probe_stats(1.0005, 1.000000, 1.001000)),
        "a CI whose bound is exactly 1.0 must count as straddling"
    );
    // PINS THE CORRECTED MECHANISM: half_width is the distance from 1.0 to the furthest
    // bound, NOT the interval width, so the displaced-but-tight null yields the LARGER
    // threshold. If a later edit redefines it as interval width, this fails and forces the
    // author to re-read why these fields exist - rather than silently opening the hole
    // that was once wrongly claimed to be here already.
    assert!(
        null_half_width_of(biased_tight) > null_half_width_of(healthy_wide),
        "displaced-but-tight null must yield the LARGER half-width: {} vs {}",
        null_half_width_of(biased_tight),
        null_half_width_of(healthy_wide)
    );
    assert!(
        null_bias(biased_tight) > null_bias(healthy_wide),
        "only null_bias distinguishes a displaced null from a merely noisy one"
    );

    // THE DIVERGENCE, worked. A clean tight null, and an effect whose median clears both
    // the null envelope and the 2x threshold while its OWN CI still contains 1.0.
    let null = gate_probe_stats(1.000000, 0.990000, 1.010000);
    let wide_effect = gate_probe_stats(1.100000, 0.950000, 1.250000);
    assert_eq!(
        strict_gate_verdict(wide_effect, null),
        "UNDECIDED",
        "an effect whose own CI contains 1.0 is not distinguishable from no-change"
    );
    // ...and the LEGACY rule called those same numbers a win. The divergence stays pinned in
    // executable form AFTER the harmonisation, because it is the reason the harmonisation
    // happened and because verdict_legacy_weak is still emitted: a reader comparing an old
    // banked row against a new one needs this pair to keep its meaning.
    assert_eq!(
        legacy_weak_verdict(wide_effect, null),
        "DECIDABLE_WIN",
        "the pre-2026-08-17 rule certified this pair on median alone; if that stops being \
         true then verdict_legacy_weak no longer reproduces the old rule and banked rows \
         can no longer be compared against it"
    );
    // And the SHIPPED verdict must now refuse it - the harmonisation itself, pinned.
    assert_eq!(
        strict_gate_verdict(wide_effect, null),
        "UNDECIDED",
        "after harmonisation the emitted verdict must be the canonical one"
    );
    // A properly separated effect must still WIN under the strict rule, or the strict
    // verdict is merely rejecting everything and carries no information.
    assert_eq!(
        strict_gate_verdict(gate_probe_stats(1.100000, 1.050000, 1.150000), null),
        "DECIDABLE_WIN",
        "an effect CI clear of unity and outside the null envelope must still win"
    );
    assert_eq!(
        strict_gate_verdict(gate_probe_stats(0.900000, 0.850000, 0.950000), null),
        "DECIDABLE_REGRESSION",
        "the mirrored regression case must survive the effect-CI requirement"
    );
}

fn run_median_ci_contract<A, B>(row: &str, mut base: A, mut candidate: B) -> (PairStats, PairStats)
where
    A: FnMut() -> TimedValue,
    B: FnMut() -> TimedValue,
{
    for _ in 0..4 {
        black_box(min_of(&mut base));
        black_box(min_of(&mut base));
    }

    let null_a = RefCell::new(Vec::with_capacity(CONTRACT_ROUNDS));
    let null_b = RefCell::new(Vec::with_capacity(CONTRACT_ROUNDS));
    let mut null_checksum = 0_u64;
    for round in 0..CONTRACT_ROUNDS {
        let (a, b) = if round & 1 == 0 {
            (min_of(&mut base), min_of(&mut base))
        } else {
            let b = min_of(&mut base);
            (min_of(&mut base), b)
        };
        assert_eq!(
            a.checksum, b.checksum,
            "base/base null arms produced different output checksums"
        );
        null_a.borrow_mut().push(a.elapsed.as_secs_f64() * 1.0e9);
        null_b.borrow_mut().push(b.elapsed.as_secs_f64() * 1.0e9);
        null_checksum = mix_checksum(null_checksum, a.checksum);
    }
    let null = report_pair("null_base_aa", &null_a, &null_b, null_checksum)
        .expect("the explicit contract always supplies 41 null rounds");

    for round in 0..4 {
        let _ = paired(round, &mut base, &mut candidate);
    }
    let base_samples = RefCell::new(Vec::with_capacity(CONTRACT_ROUNDS));
    let candidate_samples = RefCell::new(Vec::with_capacity(CONTRACT_ROUNDS));
    let mut effect_checksum = 0_u64;
    for round in 0..CONTRACT_ROUNDS {
        let (base_value, candidate_value) = paired(round, &mut base, &mut candidate);
        base_samples
            .borrow_mut()
            .push(base_value.elapsed.as_secs_f64() * 1.0e9);
        candidate_samples
            .borrow_mut()
            .push(candidate_value.elapsed.as_secs_f64() * 1.0e9);
        effect_checksum = mix_checksum(effect_checksum, base_value.checksum);
    }
    let effect = report_pair(
        "effect_base_over_candidate",
        &base_samples,
        &candidate_samples,
        effect_checksum,
    )
    .expect("the explicit contract always supplies 41 effect rounds");
    report_median_ci_gate(row, effect, null);
    (effect, null)
}

fn generate_f64_data(n: usize) -> Vec<u8> {
    let data: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    bytemuck::cast_slice(&data).to_vec()
}

fn make_npy_header(shape: &[usize]) -> NpyHeader {
    NpyHeader {
        descr: IOSupportedDType::F64,
        fortran_order: false,
        shape: shape.to_vec(),
    }
}

fn native_u64_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::U64
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::U64Be
    }
}

fn non_native_u64_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::U64Be
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::U64
    }
}

fn native_i64_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::I64
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::I64Be
    }
}

fn non_native_i64_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::I64Be
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::I64
    }
}

fn non_native_f64_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::F64Be
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::F64
    }
}

fn native_f32_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::F32
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::F32Be
    }
}

fn non_native_f32_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::F32Be
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::F32
    }
}

fn native_i32_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::I32
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::I32Be
    }
}

fn non_native_i32_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::I32Be
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::I32
    }
}

fn native_u32_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::U32
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::U32Be
    }
}

fn non_native_u32_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::U32Be
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::U32
    }
}

fn native_i16_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::I16
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::I16Be
    }
}

fn non_native_i16_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::I16Be
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::I16
    }
}

fn native_u16_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::U16
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::U16Be
    }
}

fn non_native_u16_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::U16Be
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::U16
    }
}

fn fromfile_u64_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<u64>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            u64::from_ne_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as f64
        })
        .collect()
}

fn fromfile_non_native_u64_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<u64>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            u64::from_ne_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
            .swap_bytes() as f64
        })
        .collect()
}

fn fromfile_i64_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<i64>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            i64::from_ne_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as f64
        })
        .collect()
}

fn fromfile_non_native_i64_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<i64>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            i64::from_ne_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
            .swap_bytes() as f64
        })
        .collect()
}

fn fromfile_non_native_f64_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<f64>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            f64::from_bits(
                u64::from_ne_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
                .swap_bytes(),
            )
        })
        .collect()
}

fn fromfile_f32_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<f32>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            f64::from(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        })
        .collect()
}

fn fromfile_non_native_f32_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<f32>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            f64::from(f32::from_bits(
                u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).swap_bytes(),
            ))
        })
        .collect()
}

fn fromfile_i32_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<i32>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            f64::from(i32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        })
        .collect()
}

fn fromfile_non_native_i32_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<i32>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            f64::from(i32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).swap_bytes())
        })
        .collect()
}

fn fromfile_u32_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<u32>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            f64::from(u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        })
        .collect()
}

fn fromfile_non_native_u32_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<u32>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            f64::from(u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).swap_bytes())
        })
        .collect()
}

fn fromfile_i16_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<i16>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            f64::from(i16::from_ne_bytes([chunk[0], chunk[1]]))
        })
        .collect()
}

fn fromfile_non_native_i16_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<i16>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            f64::from(i16::from_ne_bytes([chunk[0], chunk[1]]).swap_bytes())
        })
        .collect()
}

fn fromfile_u16_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<u16>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            f64::from(u16::from_ne_bytes([chunk[0], chunk[1]]))
        })
        .collect()
}

fn fromfile_non_native_u16_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<u16>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            f64::from(u16::from_ne_bytes([chunk[0], chunk[1]]).swap_bytes())
        })
        .collect()
}

#[inline(never)]
fn fromfile_text_bounded_prefix_former(text: &str, count: usize) -> Vec<f64> {
    let fields: Vec<&str> = text.split_whitespace().collect();
    fields
        .into_iter()
        .take(count)
        .map(|field| field.trim().parse::<f64>().unwrap())
        .collect()
}

fn bench_fromfile_text_bounded_prefix(c: &mut Criterion) {
    const TOKEN_COUNT: usize = 131_071;
    const PREFIX_COUNT: usize = 32;

    let text = "1.25 ".repeat(TOKEN_COUNT);
    let former = fromfile_text_bounded_prefix_former(&text, PREFIX_COUNT);
    let current = fromfile_text(&text, " ", Some(PREFIX_COUNT)).unwrap();
    assert!(
        current
            .iter()
            .zip(&former)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );
    assert_eq!(current.len(), PREFIX_COUNT);

    let mut group = c.benchmark_group("fromfile_text_bounded_prefix");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Elements(PREFIX_COUNT as u64));
    group.bench_function("former_eager_collect", |bench| {
        bench.iter(|| {
            black_box(fromfile_text_bounded_prefix_former(
                black_box(&text),
                PREFIX_COUNT,
            ))
        })
    });
    group.bench_function("public_bounded_count", |bench| {
        bench.iter(|| black_box(fromfile_text(black_box(&text), " ", Some(PREFIX_COUNT)).unwrap()))
    });
    group.finish();
}

/// Faithful replica of the CURRENT general path for a pure-literal separator
/// with a bounded count: eager whole-input `split(sep)` collect, then the
/// same trim/empty-field/parse loop with the count break. The candidate must
/// reproduce it bit-for-bit while only streaming the tokenization.
#[inline(never)]
fn fromfile_text_literal_bounded_former(text: &str, sep: &str, count: usize) -> Vec<f64> {
    let fields: Vec<&str> = text.split(sep).collect();
    let mut values = Vec::new();
    let mut iter = fields.into_iter().peekable();
    while let Some(field) = iter.next() {
        if values.len() >= count {
            break;
        }
        let field = field.trim();
        if field.is_empty() {
            if iter.peek().is_none() {
                continue;
            }
            panic!("unexpected empty field in benchmark input");
        }
        values.push(field.parse::<f64>().unwrap());
    }
    values
}

fn bench_fromfile_text_literal_bounded_prefix(c: &mut Criterion) {
    const TOKEN_COUNT: usize = 131_071;
    const PREFIX_COUNT: usize = 32;

    let text = vec!["1.25"; TOKEN_COUNT].join(",");
    let former = fromfile_text_literal_bounded_former(&text, ",", PREFIX_COUNT);
    let current = fromfile_text(&text, ",", Some(PREFIX_COUNT)).unwrap();
    assert_eq!(current.len(), PREFIX_COUNT);
    assert_eq!(former.len(), PREFIX_COUNT);
    assert!(
        current
            .iter()
            .zip(&former)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut group = c.benchmark_group("fromfile_text_literal_bounded_prefix");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Elements(PREFIX_COUNT as u64));
    group.bench_function("former_eager_collect", |bench| {
        bench.iter(|| {
            black_box(fromfile_text_literal_bounded_former(
                black_box(&text),
                black_box(","),
                PREFIX_COUNT,
            ))
        })
    });
    group.bench_function("public_bounded_count", |bench| {
        bench.iter(|| black_box(fromfile_text(black_box(&text), ",", Some(PREFIX_COUNT)).unwrap()))
    });
    group.finish();
}

/// Faithful replica of the CURRENT space-wildcard path with a bounded count:
/// eager whole-input scan collecting every field, then the same
/// trim/empty-field/parse loop with the count break. Frozen copy of the
/// production scanner (tokens, wildcard matcher, emission order) so the lazy
/// candidate must reproduce it bit-for-bit.
#[derive(Clone, Copy)]
enum FormerSepToken {
    SpaceWildcard,
    Literal(char),
}

fn former_match_space_wildcard_sep(
    text: &str,
    start: usize,
    tokens: &[FormerSepToken],
) -> Option<usize> {
    let mut offset = 0usize;
    let mut iter = text[start..].chars().peekable();
    for token in tokens {
        match token {
            FormerSepToken::SpaceWildcard => {
                while let Some(&ch) = iter.peek() {
                    if ch.is_whitespace() {
                        iter.next();
                        offset += ch.len_utf8();
                    } else {
                        break;
                    }
                }
            }
            FormerSepToken::Literal(expected) => match iter.next() {
                Some(ch) if ch == *expected => {
                    offset += ch.len_utf8();
                }
                _ => return None,
            },
        }
    }
    Some(start + offset)
}

#[inline(never)]
fn fromfile_text_wildcard_bounded_former(text: &str, sep: &str, count: usize) -> Vec<f64> {
    let tokens: Vec<FormerSepToken> = sep
        .chars()
        .map(|c| {
            if c.is_whitespace() {
                FormerSepToken::SpaceWildcard
            } else {
                FormerSepToken::Literal(c)
            }
        })
        .collect();
    let mut parts = Vec::new();
    let mut field_start = 0usize;
    let mut idx = 0usize;
    while idx <= text.len() {
        if let Some(end) = former_match_space_wildcard_sep(text, idx, &tokens) {
            parts.push(&text[field_start..idx]);
            field_start = end;
            idx = end;
            continue;
        }
        if idx == text.len() {
            break;
        }
        let ch = text[idx..].chars().next().expect("valid utf-8");
        idx += ch.len_utf8();
    }
    parts.push(&text[field_start..]);

    let mut values = Vec::new();
    let mut iter = parts.into_iter().peekable();
    while let Some(field) = iter.next() {
        if values.len() >= count {
            break;
        }
        let field = field.trim();
        if field.is_empty() {
            if iter.peek().is_none() {
                continue;
            }
            panic!("unexpected empty field in benchmark input");
        }
        values.push(field.parse::<f64>().unwrap());
    }
    values
}

fn bench_fromfile_text_wildcard_bounded_prefix(c: &mut Criterion) {
    const TOKEN_COUNT: usize = 131_071;
    const PREFIX_COUNT: usize = 32;

    let text = vec!["1.25"; TOKEN_COUNT].join(", ");
    let former = fromfile_text_wildcard_bounded_former(&text, ", ", PREFIX_COUNT);
    let current = fromfile_text(&text, ", ", Some(PREFIX_COUNT)).unwrap();
    assert_eq!(current.len(), PREFIX_COUNT);
    assert_eq!(former.len(), PREFIX_COUNT);
    assert!(
        current
            .iter()
            .zip(&former)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut group = c.benchmark_group("fromfile_text_wildcard_bounded_prefix");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Elements(PREFIX_COUNT as u64));
    group.bench_function("former_eager_scan", |bench| {
        bench.iter(|| {
            black_box(fromfile_text_wildcard_bounded_former(
                black_box(&text),
                black_box(", "),
                PREFIX_COUNT,
            ))
        })
    });
    group.bench_function("public_bounded_count", |bench| {
        bench.iter(|| black_box(fromfile_text(black_box(&text), ", ", Some(PREFIX_COUNT)).unwrap()))
    });
    group.finish();
}

/// Faithful replica of the FORMER `loadtxt_usecols` unquoted f64 path: the
/// row loop (comment strip, trims, ragged checks) with the column plan -
/// `BTreeMap<column, output positions>` plus its inner Vecs - rebuilt for
/// every accepted row, exactly as production did before the hoist.
#[inline(never)]
fn loadtxt_usecols_former(
    text: &str,
    delimiter: char,
    comments: char,
    cols: &[usize],
) -> (Vec<f64>, usize, usize) {
    use std::collections::BTreeMap;
    let mut values = Vec::new();
    let mut ncols: Option<usize> = None;
    let mut nrows = 0usize;
    for line in text.lines() {
        let trimmed = match line.find(comments) {
            Some(pos) => &line[..pos],
            None => line,
        }
        .trim();
        if trimmed.is_empty() || trimmed.starts_with(comments) {
            continue;
        }
        // Former per-row plan build.
        let mut positions: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut max_col = 0usize;
        for (pos, &col) in cols.iter().enumerate() {
            positions.entry(col).or_default().push(pos);
            if col > max_col {
                max_col = col;
            }
        }
        let mut selected = vec![0.0; cols.len()];
        let mut col_idx = 0usize;
        if delimiter == ' ' {
            for token in trimmed.split_whitespace() {
                if col_idx > max_col {
                    break;
                }
                if let Some(pos_list) = positions.get(&col_idx) {
                    let value = token.parse::<f64>().unwrap();
                    for &pos in pos_list {
                        selected[pos] = value;
                    }
                }
                col_idx += 1;
            }
        } else {
            for token in trimmed.split(delimiter) {
                if col_idx > max_col {
                    break;
                }
                if let Some(pos_list) = positions.get(&col_idx) {
                    let value = token.trim().parse::<f64>().unwrap();
                    for &pos in pos_list {
                        selected[pos] = value;
                    }
                }
                col_idx += 1;
            }
        }
        assert!(col_idx > max_col, "usecols index out of bounds");
        match ncols {
            None => ncols = Some(selected.len()),
            Some(expected) => assert_eq!(selected.len(), expected),
        }
        values.extend(selected);
        nrows += 1;
    }
    (values, nrows, ncols.unwrap_or(0))
}

/// Faithful replica of the CURRENT unselected `loadtxt` path: a fresh
/// `Vec<f64>` collected per accepted row (with parse short-circuit), then the
/// caller's ncols-before-budget checks, then a copy into the output - exactly
/// production's per-row allocation shape.
#[inline(never)]
fn loadtxt_plain_former(text: &str, delimiter: char, comments: char) -> (Vec<f64>, usize, usize) {
    let mut values = Vec::new();
    let mut ncols: Option<usize> = None;
    let mut nrows = 0usize;
    for line in text.lines() {
        let trimmed = match line.find(comments) {
            Some(pos) => &line[..pos],
            None => line,
        }
        .trim();
        if trimmed.is_empty() || trimmed.starts_with(comments) {
            continue;
        }
        let row_vals: Vec<f64> = if delimiter == ' ' {
            trimmed
                .split_whitespace()
                .map(|s| s.parse::<f64>().unwrap())
                .collect()
        } else {
            trimmed
                .split(delimiter)
                .map(|s| s.trim().parse::<f64>().unwrap())
                .collect()
        };
        match ncols {
            None => ncols = Some(row_vals.len()),
            Some(expected) => assert_eq!(row_vals.len(), expected),
        }
        values.extend(row_vals);
        nrows += 1;
    }
    (values, nrows, ncols.unwrap_or(0))
}

/// Faithful replica of the CURRENT unselected `genfromtxt_full` path: the
/// eager `all_lines` collect (kept by the lever, shared cost in both arms)
/// plus a fresh `Vec<f64>` per row copied into the output.
#[inline(never)]
fn genfromtxt_full_plain_former(
    text: &str,
    delimiter: char,
    comments: char,
    filling_values: f64,
) -> (Vec<f64>, usize, usize) {
    let all_lines: Vec<&str> = text
        .lines()
        .filter_map(|line| {
            let trimmed = match line.find(comments) {
                Some(pos) => &line[..pos],
                None => line,
            }
            .trim();
            if trimmed.is_empty() || trimmed.starts_with(comments) {
                None
            } else {
                Some(trimmed)
            }
        })
        .collect();

    let mut values = Vec::new();
    let mut ncols: Option<usize> = None;
    let mut nrows = 0usize;
    for &trimmed in all_lines.iter() {
        let row_vals: Vec<f64> = trimmed
            .split(delimiter)
            .map(|s| s.trim().parse::<f64>().unwrap_or(filling_values))
            .collect();
        match ncols {
            None => ncols = Some(row_vals.len()),
            Some(expected) => assert_eq!(row_vals.len(), expected),
        }
        values.extend(row_vals);
        nrows += 1;
    }
    (values, nrows, ncols.unwrap_or(0))
}

/// Faithful replica of the CURRENT `tofile_text`: every integral-valued
/// element routed through `write!` fmt machinery as an i64, floats through
/// `write!("{v}")`, no capacity hint - exactly production's shape.
#[inline(never)]
fn tofile_text_former(values: &[f64], sep: &str) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    for (idx, v) in values.iter().enumerate() {
        if idx > 0 {
            out.push_str(sep);
        }
        if v.fract() == 0.0
            && v.is_finite()
            && v.abs() < 1e15
            && !(*v == 0.0 && v.is_sign_negative())
        {
            let _ = write!(&mut out, "{}", *v as i64);
        } else {
            let _ = write!(&mut out, "{v}");
        }
    }
    out
}

fn push_i64_decimal_manual(out: &mut String, value: i64) {
    let negative = value.is_negative();
    let mut magnitude = value.unsigned_abs();
    let mut digits = [0_u8; 20];
    let mut cursor = digits.len();
    loop {
        cursor -= 1;
        digits[cursor] = b'0' + (magnitude % 10) as u8;
        magnitude /= 10;
        if magnitude == 0 {
            break;
        }
    }
    if negative {
        out.push('-');
    }
    out.push_str(
        std::str::from_utf8(&digits[cursor..]).expect("decimal digits are valid ASCII UTF-8"),
    );
}

/// Reconstructs the byte-identical stack-buffer candidate formerly rejected in
/// `franken_numpy-ixs5y.350`. The resurrection contract retains this independent
/// replica so future runs can compare the shipped production path with both the
/// former implementation and the exact candidate that received the KEEP.
#[inline(never)]
fn tofile_text_manual_int_candidate(values: &[f64], sep: &str) -> String {
    let mut out = String::new();
    for (idx, v) in values.iter().enumerate() {
        if idx > 0 {
            out.push_str(sep);
        }
        if v.fract() == 0.0
            && v.is_finite()
            && v.abs() < 1e15
            && !(*v == 0.0 && v.is_sign_negative())
        {
            push_i64_decimal_manual(&mut out, *v as i64);
        } else {
            write!(&mut out, "{v}").expect("writing to String cannot fail");
        }
    }
    out
}

fn tofile_text_integral_fixture() -> Vec<f64> {
    const ELEMENTS: usize = 131_072;
    (0..ELEMENTS)
        .map(|i| match i % 19 {
            17 => (i as f64) * 0.25 + 0.5,
            18 => -(i as f64) * 1.75,
            _ => ((i as i64 * 7919) % 2_000_003 - 1_000_001) as f64,
        })
        .collect()
}

fn checksum_string(output: &str) -> u64 {
    output
        .as_bytes()
        .iter()
        .fold(output.len() as u64, |state, byte| {
            mix_checksum(state, u64::from(*byte))
        })
}

fn time_tofile_text_integral(values: &[f64], candidate: bool) -> TimedValue {
    let started = Instant::now();
    let output = if candidate {
        fnp_io::tofile_text(values, ",")
    } else {
        tofile_text_former(values, ",")
    };
    let mut elapsed = started.elapsed();
    let checksum = checksum_string(&output);
    let drop_started = Instant::now();
    drop(black_box(output));
    elapsed += drop_started.elapsed();
    TimedValue { elapsed, checksum }
}

fn run_tofile_text_integral_contract() {
    for value in [
        i64::MIN,
        i64::MIN + 1,
        -1_000_000_000_000_000,
        -10,
        -1,
        0,
        1,
        10,
        1_000_000_000_000_000,
        i64::MAX - 1,
        i64::MAX,
    ] {
        let mut actual = String::new();
        push_i64_decimal_manual(&mut actual, value);
        assert_eq!(actual, value.to_string());
    }
    for value in -10_000_i64..=10_000 {
        let mut actual = String::new();
        push_i64_decimal_manual(&mut actual, value);
        assert_eq!(actual, value.to_string());
    }

    let values = tofile_text_integral_fixture();
    let former = tofile_text_former(&values, ",");
    let candidate = tofile_text_manual_int_candidate(&values, ",");
    let current = fnp_io::tofile_text(&values, ",");
    assert_eq!(candidate, former);
    assert_eq!(current, former);

    let _ = run_median_ci_contract(
        "tofile_text_manual_int_resurrection",
        || time_tofile_text_integral(&values, false),
        || time_tofile_text_integral(&values, true),
    );
}

fn bench_tofile_text_integral(c: &mut Criterion) {
    const ELEMENTS: usize = 131_072;
    let values = tofile_text_integral_fixture();

    let former = tofile_text_former(&values, ",");
    let candidate = tofile_text_manual_int_candidate(&values, ",");
    assert_eq!(candidate, former);

    // Variance protocol: 20 samples, 2 s window; floor predeclared in the
    // bead (disjoint AND >= 1.05x).
    let mut group = c.benchmark_group("tofile_text_integral");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(ELEMENTS as u64));
    group.bench_function("former_fmt_machinery", |bench| {
        bench.iter(|| black_box(tofile_text_former(black_box(&values), black_box(","))))
    });
    group.bench_function("candidate_manual_int", |bench| {
        bench.iter(|| black_box(fnp_io::tofile_text(black_box(&values), black_box(","))))
    });
    group.finish();
}

fn bench_genfromtxt_full_plain_rows(c: &mut Criterion) {
    const ROWS: usize = 8_192;
    const COLS: usize = 16;

    let mut text = String::new();
    for row in 0..ROWS {
        for col in 0..COLS {
            if col > 0 {
                text.push(',');
            }
            if (row + col) % 37 == 0 {
                text.push_str("n/a");
            } else {
                text.push_str(&format!("{}.{}", row % 977, col));
            }
        }
        text.push('\n');
    }

    let (former_values, former_rows, former_cols) =
        genfromtxt_full_plain_former(&text, ',', '#', -9.5);
    let config = fnp_io::GenFromTxtConfig {
        delimiter: ',',
        filling_values: -9.5,
        ..Default::default()
    };
    let current = fnp_io::genfromtxt_full(&text, &config).unwrap();
    assert_eq!(current.nrows, former_rows);
    assert_eq!(current.ncols, former_cols);
    assert!(
        current
            .values
            .iter()
            .zip(&former_values)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    // Variance protocol: 20 samples, 2 s window, quiet worker; floor
    // predeclared in the bead (disjoint AND >= 1.05x).
    let mut group = c.benchmark_group("genfromtxt_full_plain_rows");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements((ROWS * COLS) as u64));
    group.bench_function("former_per_row_vec", |bench| {
        bench.iter(|| {
            black_box(genfromtxt_full_plain_former(
                black_box(&text),
                ',',
                '#',
                -9.5,
            ))
        })
    });
    group.bench_function("candidate_direct_extend", |bench| {
        bench.iter(|| black_box(fnp_io::genfromtxt_full(black_box(&text), &config).unwrap()))
    });
    group.finish();
}

fn bench_loadtxt_plain_rows(c: &mut Criterion) {
    const ROWS: usize = 8_192;
    const COLS: usize = 16;

    let mut text = String::new();
    for row in 0..ROWS {
        for col in 0..COLS {
            if col > 0 {
                text.push(',');
            }
            text.push_str(&format!("{}.{}", row % 977, col));
        }
        text.push('\n');
    }

    let (former_values, former_rows, former_cols) = loadtxt_plain_former(&text, ',', '#');
    let current = fnp_io::loadtxt_usecols(&text, ',', '#', 0, usize::MAX, None).unwrap();
    assert_eq!(current.nrows, former_rows);
    assert_eq!(current.ncols, former_cols);
    assert!(
        current
            .values
            .iter()
            .zip(&former_values)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    // Variance protocol: 20 samples, 2 s window, quiet worker; floor
    // predeclared in the bead (disjoint AND >= 1.05x).
    let mut group = c.benchmark_group("loadtxt_plain_rows");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements((ROWS * COLS) as u64));
    group.bench_function("former_per_row_vec", |bench| {
        bench.iter(|| black_box(loadtxt_plain_former(black_box(&text), ',', '#')))
    });
    group.bench_function("candidate_direct_extend", |bench| {
        bench.iter(|| {
            black_box(
                fnp_io::loadtxt_usecols(black_box(&text), ',', '#', 0, usize::MAX, None).unwrap(),
            )
        })
    });
    group.finish();
}

/// Faithful replica of the CURRENT (post-.342) usecols path: the column plan
/// hoisted once per call, but each row still scattering into a fresh
/// `selected` Vec (`vec![0.0; n_out]`) that is then copied into the output.
/// The scatter-into candidate must reproduce it bit-for-bit while removing
/// only the per-row Vec and copy.
#[inline(never)]
fn loadtxt_usecols_hoisted_former(
    text: &str,
    delimiter: char,
    comments: char,
    cols: &[usize],
) -> (Vec<f64>, usize, usize) {
    use std::collections::BTreeMap;
    let mut positions: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut max_col = 0usize;
    for (pos, &col) in cols.iter().enumerate() {
        positions.entry(col).or_default().push(pos);
        if col > max_col {
            max_col = col;
        }
    }
    let mut values = Vec::new();
    let mut ncols: Option<usize> = None;
    let mut nrows = 0usize;
    for line in text.lines() {
        let trimmed = match line.find(comments) {
            Some(pos) => &line[..pos],
            None => line,
        }
        .trim();
        if trimmed.is_empty() || trimmed.starts_with(comments) {
            continue;
        }
        let mut selected = vec![0.0; cols.len()];
        let mut col_idx = 0usize;
        for token in trimmed.split(delimiter) {
            if col_idx > max_col {
                break;
            }
            if let Some(pos_list) = positions.get(&col_idx) {
                let value = token.trim().parse::<f64>().unwrap();
                for &pos in pos_list {
                    selected[pos] = value;
                }
            }
            col_idx += 1;
        }
        assert!(col_idx > max_col, "usecols index out of bounds");
        match ncols {
            None => ncols = Some(selected.len()),
            Some(expected) => assert_eq!(selected.len(), expected),
        }
        values.extend(selected);
        nrows += 1;
    }
    (values, nrows, ncols.unwrap_or(0))
}

#[inline(never)]
fn loadtxt_usecols_scatter_candidate(
    text: &str,
    delimiter: char,
    comments: char,
    cols: &[usize],
) -> (Vec<f64>, usize, usize) {
    use std::collections::BTreeMap;
    let mut positions: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut max_col = 0usize;
    for (pos, &col) in cols.iter().enumerate() {
        positions.entry(col).or_default().push(pos);
        max_col = max_col.max(col);
    }

    let mut values = Vec::new();
    let mut nrows = 0usize;
    for line in text.lines() {
        let trimmed = match line.find(comments) {
            Some(pos) => &line[..pos],
            None => line,
        }
        .trim();
        if trimmed.is_empty() || trimmed.starts_with(comments) {
            continue;
        }

        let row_start = values.len();
        values.resize(row_start + cols.len(), 0.0);
        let mut col_idx = 0usize;
        for token in trimmed.split(delimiter) {
            if col_idx > max_col {
                break;
            }
            if let Some(pos_list) = positions.get(&col_idx) {
                let value = token.trim().parse::<f64>().unwrap();
                for &pos in pos_list {
                    values[row_start + pos] = value;
                }
            }
            col_idx += 1;
        }
        assert!(col_idx > max_col, "usecols index out of bounds");
        nrows += 1;
    }
    (values, nrows, cols.len())
}

fn checksum_loadtxt_tuple(output: &(Vec<f64>, usize, usize)) -> u64 {
    output.0.iter().fold(
        mix_checksum(output.1 as u64, output.2 as u64),
        |state, value| mix_checksum(state, value.to_bits()),
    )
}

fn time_loadtxt_usecols(
    text: &str,
    delimiter: char,
    comments: char,
    cols: &[usize],
    candidate: bool,
) -> TimedValue {
    let started = Instant::now();
    let output = if candidate {
        loadtxt_usecols_scatter_candidate(text, delimiter, comments, cols)
    } else {
        loadtxt_usecols_hoisted_former(text, delimiter, comments, cols)
    };
    let mut elapsed = started.elapsed();
    let checksum = checksum_loadtxt_tuple(&output);
    let drop_started = Instant::now();
    drop(black_box(output));
    elapsed += drop_started.elapsed();
    TimedValue { elapsed, checksum }
}

fn bench_loadtxt_usecols_scatter(c: &mut Criterion) {
    const ROWS: usize = 8_192;
    const COLS: usize = 16;
    const USECOLS: [usize; 4] = [13, 1, 7, 13];

    let mut text = String::new();
    for row in 0..ROWS {
        for col in 0..COLS {
            if col > 0 {
                text.push(',');
            }
            text.push_str(&format!("{}.{}", row % 977, col));
        }
        text.push('\n');
    }

    let (former_values, former_rows, former_cols) =
        loadtxt_usecols_hoisted_former(&text, ',', '#', &USECOLS);
    let candidate = loadtxt_usecols_scatter_candidate(&text, ',', '#', &USECOLS);
    assert_eq!(candidate.1, former_rows);
    assert_eq!(candidate.2, former_cols);
    assert!(
        candidate
            .0
            .iter()
            .zip(&former_values)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );
    let current = fnp_io::loadtxt_usecols(&text, ',', '#', 0, usize::MAX, Some(&USECOLS)).unwrap();
    assert_eq!(current.nrows, former_rows);
    assert_eq!(current.ncols, former_cols);
    assert!(
        current
            .values
            .iter()
            .zip(&former_values)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let _ = run_median_ci_contract(
        "loadtxt_usecols_scatter_resurrection",
        || time_loadtxt_usecols(&text, ',', '#', &USECOLS, false),
        || time_loadtxt_usecols(&text, ',', '#', &USECOLS, true),
    );

    // Variance protocol: 20 samples, 2 s window, quiet worker; floor
    // predeclared in the bead (disjoint AND >= 1.05x).
    let mut group = c.benchmark_group("loadtxt_usecols_scatter");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements((ROWS * USECOLS.len()) as u64));
    group.bench_function("former_selected_vec", |bench| {
        bench.iter(|| {
            black_box(loadtxt_usecols_hoisted_former(
                black_box(&text),
                ',',
                '#',
                black_box(&USECOLS),
            ))
        })
    });
    group.bench_function("candidate_scatter_into", |bench| {
        bench.iter(|| {
            black_box(loadtxt_usecols_scatter_candidate(
                black_box(&text),
                ',',
                '#',
                black_box(&USECOLS),
            ))
        })
    });
    group.finish();
}

fn bench_loadtxt_usecols_plan(c: &mut Criterion) {
    const ROWS: usize = 8_192;
    const COLS: usize = 16;
    // Duplicate and out-of-order selections deliberately present (x6teb shape).
    const USECOLS: [usize; 4] = [13, 1, 7, 13];

    let mut text = String::new();
    for row in 0..ROWS {
        for col in 0..COLS {
            if col > 0 {
                text.push(' ');
            }
            text.push_str(&format!("{}.{}", row % 977, col));
        }
        text.push('\n');
    }

    let (former_values, former_rows, former_cols) =
        loadtxt_usecols_former(&text, ' ', '#', &USECOLS);
    let current = fnp_io::loadtxt_usecols(&text, ' ', '#', 0, usize::MAX, Some(&USECOLS)).unwrap();
    assert_eq!(current.nrows, former_rows);
    assert_eq!(current.ncols, former_cols);
    assert_eq!(current.values.len(), former_values.len());
    assert!(
        current
            .values
            .iter()
            .zip(&former_values)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    // x6teb retry protocol: 20 samples and a 2 s window on a warm pinned
    // worker, with the significance floor predeclared in the bead/ledger.
    let mut group = c.benchmark_group("loadtxt_usecols_plan");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements((ROWS * USECOLS.len()) as u64));
    group.bench_function("former_per_row_planner", |bench| {
        bench.iter(|| {
            black_box(loadtxt_usecols_former(
                black_box(&text),
                ' ',
                '#',
                black_box(&USECOLS),
            ))
        })
    });
    group.bench_function("hoisted_plan_candidate", |bench| {
        bench.iter(|| {
            black_box(
                fnp_io::loadtxt_usecols(
                    black_box(&text),
                    ' ',
                    '#',
                    0,
                    usize::MAX,
                    black_box(Some(&USECOLS)),
                )
                .unwrap(),
            )
        })
    });
    group.finish();
}

/// Former signed-usecols valid-input path, retained as the A/B comparator for
/// the exact nonnegative corpus measured by this benchmark.
#[inline(never)]
fn loadtxt_signed_nonnegative_former(text: &str, usecols: &[isize]) -> fnp_io::TextArrayData {
    let mut values = Vec::new();
    let mut ncols = None;
    let mut nrows = 0usize;
    for line in text.lines() {
        let trimmed = line
            .split_once('#')
            .map_or(line, |(prefix, _)| prefix)
            .trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields = trimmed.split(',').collect::<Vec<_>>();
        let mut row_values = Vec::with_capacity(usecols.len());
        for &column in usecols {
            let index = usize::try_from(column).expect("nonnegative former usecol");
            assert!(index < fields.len(), "former usecol in bounds");
            row_values.push(fields[index].trim().parse::<f64>().unwrap());
        }
        match ncols {
            None => ncols = Some(row_values.len()),
            Some(expected) => assert_eq!(row_values.len(), expected),
        }
        values.extend(row_values);
        nrows += 1;
    }
    fnp_io::TextArrayData {
        values,
        nrows,
        ncols: ncols.unwrap_or(0),
    }
}

#[inline(never)]
fn loadtxt_signed_nonnegative_staged(
    text: &str,
    delimiter: char,
    comments: char,
    usecols: &[isize],
) -> fnp_io::TextArrayData {
    let unsigned = usecols
        .iter()
        .map(|&col| usize::try_from(col).expect("nonnegative staged usecol"))
        .collect::<Vec<_>>();
    fnp_io::loadtxt_usecols(text, delimiter, comments, 0, usize::MAX, Some(&unsigned)).unwrap()
}

fn time_text_array<F>(mut operation: F) -> TimedValue
where
    F: FnMut() -> fnp_io::TextArrayData,
{
    const REPETITIONS: u32 = 8;
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    for _ in 0..REPETITIONS {
        let started = Instant::now();
        let output = operation();
        elapsed += started.elapsed();
        checksum = mix_checksum(checksum, checksum_text_array(&output));
        let drop_started = Instant::now();
        drop(black_box(output));
        elapsed += drop_started.elapsed();
    }
    TimedValue {
        elapsed: elapsed / REPETITIONS,
        checksum,
    }
}

fn time_loadtxt_signed(text: &str, usecols: &[isize], staged: bool) -> TimedValue {
    time_text_array(|| {
        if staged {
            fnp_io::loadtxt_usecols_signed(text, ',', '#', 0, usize::MAX, Some(usecols)).unwrap()
        } else {
            loadtxt_signed_nonnegative_former(text, usecols)
        }
    })
}

#[derive(Debug, PartialEq, Eq)]
struct SelectedBoolOutput {
    values: Vec<bool>,
    nrows: usize,
    ncols: usize,
}

/// Frozen replica of the selected-bool `loadtxt` path rejected in `.377`:
/// own every token, clone selected tokens into nested rows, then parse.
#[inline(never)]
fn selected_bool_former(
    text: &str,
    delimiter: char,
    comments: char,
    skiprows: usize,
    usecols: &[usize],
) -> Option<SelectedBoolOutput> {
    let mut rows = Vec::new();
    for (lineno, raw_line) in text.lines().enumerate() {
        if lineno < skiprows {
            continue;
        }
        let effective = match raw_line.split_once(comments) {
            Some((lhs, _)) => lhs,
            None => raw_line,
        };
        let trimmed = effective.trim();
        if trimmed.is_empty() {
            continue;
        }
        let tokens = trimmed
            .split(delimiter)
            .map(|token| token.trim().to_string())
            .collect::<Vec<_>>();
        let mut selected = Vec::with_capacity(usecols.len());
        for &column in usecols {
            selected.push(tokens.get(column)?.clone());
        }
        if !selected.is_empty() {
            rows.push(selected);
        }
    }
    let ncols = rows.first()?.len();
    if rows.iter().any(|row| row.len() != ncols) {
        return None;
    }
    let nrows = rows.len();
    let mut values = Vec::with_capacity(nrows * ncols);
    for row in rows {
        for token in row {
            values.push(token.parse::<i64>().ok()? != 0);
        }
    }
    Some(SelectedBoolOutput {
        values,
        nrows,
        ncols,
    })
}

/// The `.377` candidate: retain borrowed row tokens and parse only requested
/// positive columns directly into the final bool storage.
#[inline(never)]
fn selected_bool_candidate(
    text: &str,
    delimiter: char,
    comments: char,
    skiprows: usize,
    usecols: &[usize],
) -> Option<SelectedBoolOutput> {
    let mut values = Vec::new();
    let mut nrows = 0usize;
    for (lineno, raw_line) in text.lines().enumerate() {
        if lineno < skiprows {
            continue;
        }
        let effective = match raw_line.split_once(comments) {
            Some((lhs, _)) => lhs,
            None => raw_line,
        };
        let trimmed = effective.trim();
        if trimmed.is_empty() {
            continue;
        }
        let tokens = trimmed.split(delimiter).map(str::trim).collect::<Vec<_>>();
        for &column in usecols {
            values.push(tokens.get(column)?.parse::<i64>().ok()? != 0);
        }
        nrows += 1;
    }
    if nrows == 0 || usecols.is_empty() {
        return None;
    }
    Some(SelectedBoolOutput {
        values,
        nrows,
        ncols: usecols.len(),
    })
}

fn checksum_selected_bool(output: &SelectedBoolOutput) -> u64 {
    output.values.iter().fold(
        mix_checksum(output.nrows as u64, output.ncols as u64),
        |state, &value| mix_checksum(state, value as u64),
    )
}

fn time_selected_bool(text: &str, usecols: &[usize], candidate: bool) -> TimedValue {
    const REPETITIONS: u32 = 8;
    let mut elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    for _ in 0..REPETITIONS {
        let started = Instant::now();
        let output = if candidate {
            selected_bool_candidate(text, ',', '#', 1, usecols)
        } else {
            selected_bool_former(text, ',', '#', 1, usecols)
        }
        .expect("selected bool fixture must parse");
        elapsed += started.elapsed();
        checksum = mix_checksum(checksum, checksum_selected_bool(&output));
        let drop_started = Instant::now();
        drop(black_box(output));
        elapsed += drop_started.elapsed();
    }
    TimedValue {
        elapsed: elapsed / REPETITIONS,
        checksum,
    }
}

fn bench_loadtxt_selected_bool_direct_parse(c: &mut Criterion) {
    const ROWS: usize = 8_192;
    const COLS: usize = 16;
    const USECOLS: [usize; 4] = [13, 1, 7, 13];

    let mut text = String::from("ignored,header,row\n");
    for row in 0..ROWS {
        for col in 0..COLS {
            if col > 0 {
                text.push(',');
            }
            let value = ((row + col) % 5) as i64 - 2;
            text.push_str(&value.to_string());
        }
        text.push('\n');
    }

    let former = selected_bool_former(&text, ',', '#', 1, &USECOLS).expect("former bool fixture");
    let candidate =
        selected_bool_candidate(&text, ',', '#', 1, &USECOLS).expect("candidate bool fixture");
    assert_eq!(candidate, former);

    let invalid_unselected = "1,invalid,0 # comment\n0,also-invalid,1\n";
    assert_eq!(
        selected_bool_candidate(invalid_unselected, ',', '#', 0, &[2, 0]),
        selected_bool_former(invalid_unselected, ',', '#', 0, &[2, 0])
    );
    assert!(selected_bool_candidate("1,0\n", ',', '#', 0, &[2]).is_none());
    assert!(selected_bool_former("1,0\n", ',', '#', 0, &[2]).is_none());

    let mut group = c.benchmark_group("loadtxt_selected_bool_direct_parse");
    group.sample_size(CONTRACT_ROUNDS);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Elements((ROWS * USECOLS.len()) as u64));

    let _ = run_median_ci_contract(
        "loadtxt_selected_bool_direct_parse",
        || time_selected_bool(&text, &USECOLS, false),
        || time_selected_bool(&text, &USECOLS, true),
    );
    group.bench_function("former_owned_rows", |bench| {
        bench.iter(|| black_box(selected_bool_former(&text, ',', '#', 1, &USECOLS)))
    });
    group.bench_function("candidate_direct_borrowed", |bench| {
        bench.iter(|| black_box(selected_bool_candidate(&text, ',', '#', 1, &USECOLS)))
    });
    group.finish();
}

fn bench_loadtxt_signed_nonnegative_staging(c: &mut Criterion) {
    const ROWS: usize = 8_192;
    const COLS: usize = 16;
    const USECOLS: [isize; 4] = [13, 1, 7, 13];

    let mut text = String::new();
    for row in 0..ROWS {
        for col in 0..COLS {
            if col > 0 {
                text.push(',');
            }
            text.push_str(&format!("{}.{}", row % 977, col));
        }
        text.push('\n');
    }

    let current =
        fnp_io::loadtxt_usecols_signed(&text, ',', '#', 0, usize::MAX, Some(&USECOLS)).unwrap();
    let staged = loadtxt_signed_nonnegative_staged(&text, ',', '#', &USECOLS);
    assert_eq!(current.nrows, staged.nrows);
    assert_eq!(current.ncols, staged.ncols);
    assert_eq!(current.values.len(), staged.values.len());
    assert!(
        current
            .values
            .iter()
            .zip(&staged.values)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut group = c.benchmark_group("loadtxt_signed_nonnegative_staging");
    group.sample_size(CONTRACT_ROUNDS);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Elements((ROWS * USECOLS.len()) as u64));

    let _ = run_median_ci_contract(
        "loadtxt_signed_nonnegative_staging",
        || time_loadtxt_signed(&text, &USECOLS, false),
        || time_loadtxt_signed(&text, &USECOLS, true),
    );
    group.bench_function("base_current", |bench| {
        bench.iter(|| {
            black_box(
                fnp_io::loadtxt_usecols_signed(&text, ',', '#', 0, usize::MAX, Some(&USECOLS))
                    .expect("base signed loadtxt"),
            )
        })
    });
    group.bench_function("candidate_nonnegative_staging", |bench| {
        bench.iter(|| black_box(loadtxt_signed_nonnegative_staged(&text, ',', '#', &USECOLS)))
    });
    group.finish();
}

#[inline(never)]
fn loadtxt_signed_tail_former(
    text: &str,
    delimiter: char,
    comments: char,
    usecols: &[isize],
) -> fnp_io::TextArrayData {
    let mut values = Vec::new();
    let mut ncols = None;
    let mut nrows = 0usize;
    for line in text.lines() {
        let trimmed = line
            .split_once(comments)
            .map_or(line, |(prefix, _)| prefix)
            .trim();
        if trimmed.is_empty() || trimmed.starts_with(comments) {
            continue;
        }
        let fields = if delimiter == ' ' {
            trimmed.split_whitespace().collect::<Vec<_>>()
        } else {
            trimmed.split(delimiter).collect::<Vec<_>>()
        };
        let mut row_values = Vec::with_capacity(usecols.len());
        for &column in usecols {
            let offset = usize::try_from(
                column
                    .checked_neg()
                    .expect("former fixture uses negative columns"),
            )
            .expect("former fixture tail offset must fit");
            let index = fields
                .len()
                .checked_sub(offset)
                .expect("former fixture usecol in bounds");
            row_values.push(fields[index].trim().parse::<f64>().unwrap());
        }
        match ncols {
            None => ncols = Some(row_values.len()),
            Some(expected) => assert_eq!(row_values.len(), expected),
        }
        values.extend(row_values);
        nrows += 1;
    }
    fnp_io::TextArrayData {
        values,
        nrows,
        ncols: ncols.unwrap_or(0),
    }
}

#[inline(never)]
fn loadtxt_signed_tail_candidate(
    text: &str,
    delimiter: char,
    comments: char,
    usecols: &[isize],
) -> fnp_io::TextArrayData {
    let offsets = usecols
        .iter()
        .map(|&column| {
            assert!(column < 0, "tail candidate requires negative usecols");
            usize::try_from(column.checked_neg().expect("representable tail offset"))
                .expect("positive tail offset")
        })
        .collect::<Vec<_>>();
    let max_tail = offsets.iter().copied().max().expect("nonempty usecols");

    let mut values = Vec::new();
    let mut ncols = None;
    let mut nrows = 0usize;
    for line in text.lines() {
        let trimmed = line
            .split_once(comments)
            .map_or(line, |(prefix, _)| prefix)
            .trim();
        if trimmed.is_empty() || trimmed.starts_with(comments) {
            continue;
        }

        let mut tail = vec![None; max_tail];
        let mut width = 0usize;
        if delimiter == ' ' {
            for field in trimmed.split_whitespace() {
                tail[width % max_tail] = Some(field);
                width += 1;
            }
        } else {
            for field in trimmed.split(delimiter) {
                tail[width % max_tail] = Some(field);
                width += 1;
            }
        }

        let mut row_values = Vec::with_capacity(offsets.len());
        for &offset in &offsets {
            assert!(offset <= width, "tail candidate usecol in bounds");
            let field = tail[(width - offset) % max_tail].expect("retained tail field");
            row_values.push(field.trim().parse::<f64>().unwrap());
        }
        match ncols {
            None => ncols = Some(row_values.len()),
            Some(expected) => assert_eq!(row_values.len(), expected),
        }
        values.extend(row_values);
        nrows += 1;
    }

    fnp_io::TextArrayData {
        values,
        nrows,
        ncols: ncols.unwrap_or(0),
    }
}

fn time_loadtxt_signed_tail(text: &str, usecols: &[isize], candidate: bool) -> TimedValue {
    time_text_array(|| {
        if candidate {
            fnp_io::loadtxt_usecols_signed(text, ',', '#', 0, usize::MAX, Some(usecols)).unwrap()
        } else {
            loadtxt_signed_tail_former(text, ',', '#', usecols)
        }
    })
}

fn bench_loadtxt_signed_tail_staging(c: &mut Criterion) {
    const ROWS: usize = 8_192;
    const COLS: usize = 64;
    const USECOLS: [isize; 4] = [-1, -8, -32, -1];

    let mut text = String::new();
    for row in 0..ROWS {
        for col in 0..COLS {
            if col > 0 {
                text.push(',');
            }
            text.push_str(&format!("{}.{}", row % 977, col));
        }
        text.push('\n');
    }

    let output =
        fnp_io::loadtxt_usecols_signed(&text, ',', '#', 0, usize::MAX, Some(&USECOLS)).unwrap();
    assert_eq!(output.nrows, ROWS);
    assert_eq!(output.ncols, USECOLS.len());
    let candidate = loadtxt_signed_tail_candidate(&text, ',', '#', &USECOLS);
    let former = loadtxt_signed_tail_former(&text, ',', '#', &USECOLS);
    assert_eq!(output.nrows, candidate.nrows);
    assert_eq!(output.ncols, candidate.ncols);
    assert_eq!(output.values.len(), candidate.values.len());
    assert!(
        output
            .values
            .iter()
            .zip(&candidate.values)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );
    assert_eq!(output.nrows, former.nrows);
    assert_eq!(output.ncols, former.ncols);
    assert_eq!(output.values.len(), former.values.len());
    assert!(
        output
            .values
            .iter()
            .zip(&former.values)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut group = c.benchmark_group("loadtxt_signed_tail_staging");
    group.sample_size(CONTRACT_ROUNDS);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Elements((ROWS * COLS) as u64));
    group.bench_function("current_width_relative", |bench| {
        bench.iter(|| {
            black_box(
                fnp_io::loadtxt_usecols_signed(
                    black_box(&text),
                    ',',
                    '#',
                    0,
                    usize::MAX,
                    black_box(Some(&USECOLS)),
                )
                .unwrap(),
            )
        })
    });

    let _ = run_median_ci_contract(
        "loadtxt_signed_tail_staging",
        || time_loadtxt_signed_tail(&text, &USECOLS, false),
        || time_loadtxt_signed_tail(&text, &USECOLS, true),
    );
    group.bench_function("former_full_row_tokens", |bench| {
        bench.iter(|| black_box(loadtxt_signed_tail_former(&text, ',', '#', &USECOLS)))
    });
    group.bench_function("candidate_bounded_tail_ring", |bench| {
        bench.iter(|| {
            black_box(
                fnp_io::loadtxt_usecols_signed(&text, ',', '#', 0, usize::MAX, Some(&USECOLS))
                    .expect("candidate tail-ring loadtxt"),
            )
        })
    });
    group.finish();
}

/// Faithful replica of the CURRENT `genfromtxt` comma path: a fresh
/// `Vec<f64>` collected per accepted row, then copied into the output -
/// exactly production's per-row allocation shape.
#[inline(never)]
fn genfromtxt_former(
    text: &str,
    delimiter: char,
    comments: char,
    filling_values: f64,
) -> (Vec<f64>, usize, usize) {
    let mut values = Vec::new();
    let mut ncols: Option<usize> = None;
    let mut nrows = 0usize;
    for line in text.lines() {
        let trimmed = match line.find(comments) {
            Some(pos) => &line[..pos],
            None => line,
        }
        .trim();
        if trimmed.is_empty() || trimmed.starts_with(comments) {
            continue;
        }
        let row_vals: Vec<f64> = trimmed
            .split(delimiter)
            .map(|s| s.trim().parse::<f64>().unwrap_or(filling_values))
            .collect();
        let current_ncols = row_vals.len();
        match ncols {
            None => ncols = Some(current_ncols),
            Some(expected) => assert_eq!(current_ncols, expected),
        }
        values.extend(row_vals);
        nrows += 1;
    }
    (values, nrows, ncols.unwrap_or(0))
}

fn bench_genfromtxt_row_scratch(c: &mut Criterion) {
    const ROWS: usize = 8_192;
    const COLS: usize = 16;

    let mut text = String::new();
    for row in 0..ROWS {
        for col in 0..COLS {
            if col > 0 {
                text.push(',');
            }
            // Mix parseable floats with unparseable tokens to exercise
            // filling_values.
            if (row + col) % 37 == 0 {
                text.push_str("n/a");
            } else {
                text.push_str(&format!("{}.{}", row % 977, col));
            }
        }
        text.push('\n');
    }

    let (former_values, former_rows, former_cols) = genfromtxt_former(&text, ',', '#', -9.5);
    let current = fnp_io::genfromtxt(&text, ',', '#', 0, -9.5).unwrap();
    assert_eq!(current.nrows, former_rows);
    assert_eq!(current.ncols, former_cols);
    assert!(
        current
            .values
            .iter()
            .zip(&former_values)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    // Variance protocol: 20 samples, 2 s window, warm pinned worker; floor
    // predeclared in the bead (disjoint AND >= 1.05x).
    let mut group = c.benchmark_group("genfromtxt_row_scratch");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements((ROWS * COLS) as u64));
    group.bench_function("former_per_row_vec", |bench| {
        bench.iter(|| black_box(genfromtxt_former(black_box(&text), ',', '#', -9.5)))
    });
    group.bench_function("candidate_scratch_reuse", |bench| {
        bench.iter(|| black_box(fnp_io::genfromtxt(black_box(&text), ',', '#', 0, -9.5).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_native_u64(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<u64> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => 0,
            1 => u64::MAX,
            2 => 1,
            3 => (1_u64 << 53) - 1,
            4 => 1_u64 << 53,
            5 => (1_u64 << 53) + 1,
            6 => 1_u64 << 63,
            _ => (index as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407),
        })
        .collect();
    let bytes: &[u8] = bytemuck::cast_slice(&values);
    let former = fromfile_u64_former(bytes, None);
    let candidate = fromfile(bytes, native_u64_dtype(), None).unwrap();
    assert!(
        candidate
            .iter()
            .zip(&former)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<u64>()];
    let misaligned_offset = (0..core::mem::align_of::<u64>())
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<u64>())
        })
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_u64_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, native_u64_dtype(), Some(257)).unwrap();
    assert!(
        candidate_misaligned
            .iter()
            .zip(&former_misaligned)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut group = c.benchmark_group("fromfile_native_u64_typed_slice");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_u64_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), native_u64_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_non_native_u64(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<u64> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => 0,
            1 => u64::MAX,
            2 => 1,
            3 => (1_u64 << 53) - 1,
            4 => 1_u64 << 53,
            5 => (1_u64 << 53) + 1,
            6 => 1_u64 << 63,
            _ => (index as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407),
        })
        .collect();
    let stored: Vec<u64> = values.iter().map(|&value| value.swap_bytes()).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&stored);
    let former = fromfile_non_native_u64_former(bytes, None);
    let candidate = fromfile(bytes, non_native_u64_dtype(), None).unwrap();
    assert!(
        candidate
            .iter()
            .zip(&former)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<u64>()];
    let misaligned_offset = (0..core::mem::align_of::<u64>())
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<u64>())
        })
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_non_native_u64_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, non_native_u64_dtype(), Some(257)).unwrap();
    assert!(
        candidate_misaligned
            .iter()
            .zip(&former_misaligned)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut group = c.benchmark_group("fromfile_non_native_u64_typed_byteswap");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_non_native_u64_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_byteswap_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), non_native_u64_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_native_i64(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<i64> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => i64::MIN,
            1 => i64::MAX,
            2 => -1,
            3 => 0,
            4 => (1_i64 << 53) - 1,
            5 => 1_i64 << 53,
            6 => (1_i64 << 53) + 1,
            _ => (index as i64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407),
        })
        .collect();
    let bytes: &[u8] = bytemuck::cast_slice(&values);
    let former = fromfile_i64_former(bytes, None);
    let candidate = fromfile(bytes, native_i64_dtype(), None).unwrap();
    assert!(
        candidate
            .iter()
            .zip(&former)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<i64>()];
    let misaligned_offset = (0..core::mem::align_of::<i64>())
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<i64>())
        })
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_i64_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, native_i64_dtype(), Some(257)).unwrap();
    assert!(
        candidate_misaligned
            .iter()
            .zip(&former_misaligned)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut group = c.benchmark_group("fromfile_native_i64_typed_slice");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_i64_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), native_i64_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_non_native_i64(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<i64> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => i64::MIN,
            1 => i64::MAX,
            2 => -1,
            3 => 0,
            4 => (1_i64 << 53) - 1,
            5 => 1_i64 << 53,
            6 => (1_i64 << 53) + 1,
            _ => (index as i64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407),
        })
        .collect();
    let stored: Vec<i64> = values.iter().map(|&value| value.swap_bytes()).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&stored);
    let former = fromfile_non_native_i64_former(bytes, None);
    let candidate = fromfile(bytes, non_native_i64_dtype(), None).unwrap();
    assert!(
        candidate
            .iter()
            .zip(&former)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<i64>()];
    let misaligned_offset = (0..core::mem::align_of::<i64>())
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<i64>())
        })
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_non_native_i64_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, non_native_i64_dtype(), Some(257)).unwrap();
    assert!(
        candidate_misaligned
            .iter()
            .zip(&former_misaligned)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut group = c.benchmark_group("fromfile_non_native_i64_typed_byteswap");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_non_native_i64_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_byteswap_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), non_native_i64_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_non_native_f64(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let bits: Vec<u64> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => 0.0_f64.to_bits(),
            1 => (-0.0_f64).to_bits(),
            2 => f64::INFINITY.to_bits(),
            3 => f64::NEG_INFINITY.to_bits(),
            4 => 1,
            5 => 0x7fef_ffff_ffff_ffff,
            6 => 0x7ff8_0000_0000_0042,
            _ => (index as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407),
        })
        .collect();
    let stored: Vec<u64> = bits.iter().map(|&value| value.swap_bytes()).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&stored);
    let former = fromfile_non_native_f64_former(bytes, None);
    let candidate = fromfile(bytes, non_native_f64_dtype(), None).unwrap();
    assert!(
        candidate
            .iter()
            .zip(&former)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<u64>()];
    let misaligned_offset = (0..core::mem::align_of::<u64>())
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<u64>())
        })
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_non_native_f64_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, non_native_f64_dtype(), Some(257)).unwrap();
    assert!(
        candidate_misaligned
            .iter()
            .zip(&former_misaligned)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut group = c.benchmark_group("fromfile_non_native_f64_typed_byteswap");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_non_native_f64_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_byteswap_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), non_native_f64_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_native_i16(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<i16> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => i16::MIN,
            1 => i16::MAX,
            2 => -1,
            3 => 0,
            _ => (index as i16).wrapping_mul(257).wrapping_sub(17),
        })
        .collect();
    let bytes: &[u8] = bytemuck::cast_slice(&values);
    let former = fromfile_i16_former(bytes, None);
    let candidate = fromfile(bytes, native_i16_dtype(), None).unwrap();
    assert_eq!(candidate, former);

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<i16>()];
    let misaligned_offset = (0..core::mem::align_of::<i16>())
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<i16>())
        })
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_i16_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, native_i16_dtype(), Some(257)).unwrap();
    assert_eq!(candidate_misaligned, former_misaligned);

    let mut group = c.benchmark_group("fromfile_native_i16_typed_slice");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_i16_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), native_i16_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_non_native_i16(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<i16> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => i16::MIN,
            1 => i16::MAX,
            2 => -1,
            3 => 0,
            _ => (index as i16).wrapping_mul(257).wrapping_sub(17),
        })
        .collect();
    let stored: Vec<i16> = values.iter().map(|&value| value.swap_bytes()).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&stored);
    let former = fromfile_non_native_i16_former(bytes, None);
    let candidate = fromfile(bytes, non_native_i16_dtype(), None).unwrap();
    assert_eq!(candidate, former);

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<i16>()];
    let misaligned_offset = (0..core::mem::align_of::<i16>())
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<i16>())
        })
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_non_native_i16_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, non_native_i16_dtype(), Some(257)).unwrap();
    assert_eq!(candidate_misaligned, former_misaligned);

    let mut group = c.benchmark_group("fromfile_non_native_i16_typed_byteswap");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_non_native_i16_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_byteswap_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), non_native_i16_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_native_u16(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<u16> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => 0,
            1 => u16::MAX,
            2 => 1,
            3 => 0x8000,
            _ => (index as u16).wrapping_mul(257).wrapping_add(17),
        })
        .collect();
    let bytes: &[u8] = bytemuck::cast_slice(&values);
    let former = fromfile_u16_former(bytes, None);
    let candidate = fromfile(bytes, native_u16_dtype(), None).unwrap();
    assert_eq!(candidate, former);

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<u16>()];
    let misaligned_offset = (0..core::mem::align_of::<u16>())
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<u16>())
        })
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_u16_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, native_u16_dtype(), Some(257)).unwrap();
    assert_eq!(candidate_misaligned, former_misaligned);

    let mut group = c.benchmark_group("fromfile_native_u16_typed_slice");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_u16_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), native_u16_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_non_native_u16(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<u16> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => 0,
            1 => u16::MAX,
            2 => 1,
            3 => 0x8000,
            _ => (index as u16).wrapping_mul(257).wrapping_add(17),
        })
        .collect();
    let stored: Vec<u16> = values.iter().map(|&value| value.swap_bytes()).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&stored);
    let former = fromfile_non_native_u16_former(bytes, None);
    let candidate = fromfile(bytes, non_native_u16_dtype(), None).unwrap();
    assert_eq!(candidate, former);

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<u16>()];
    let misaligned_offset = (0..core::mem::align_of::<u16>())
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<u16>())
        })
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_non_native_u16_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, non_native_u16_dtype(), Some(257)).unwrap();
    assert_eq!(candidate_misaligned, former_misaligned);

    let mut group = c.benchmark_group("fromfile_non_native_u16_typed_byteswap");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_non_native_u16_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_byteswap_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), non_native_u16_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_native_u32(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<u32> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => 0,
            1 => u32::MAX,
            2 => 1,
            3 => 0x8000_0000,
            _ => (index as u32).wrapping_mul(2_654_435_761).wrapping_add(17),
        })
        .collect();
    let bytes: &[u8] = bytemuck::cast_slice(&values);
    let former = fromfile_u32_former(bytes, None);
    let candidate = fromfile(bytes, native_u32_dtype(), None).unwrap();
    assert_eq!(candidate, former);

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<u32>()];
    let misaligned_offset = (0..core::mem::align_of::<u32>())
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<u32>())
        })
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_u32_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, native_u32_dtype(), Some(257)).unwrap();
    assert_eq!(candidate_misaligned, former_misaligned);

    let mut group = c.benchmark_group("fromfile_native_u32_typed_slice");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_u32_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), native_u32_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_non_native_u32(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<u32> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => 0,
            1 => u32::MAX,
            2 => 1,
            3 => 0x8000_0000,
            _ => (index as u32).wrapping_mul(2_654_435_761).wrapping_add(17),
        })
        .collect();
    let stored: Vec<u32> = values.iter().map(|&value| value.swap_bytes()).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&stored);
    let former = fromfile_non_native_u32_former(bytes, None);
    let candidate = fromfile(bytes, non_native_u32_dtype(), None).unwrap();
    assert_eq!(candidate, former);

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<u32>()];
    let misaligned_offset = (0..core::mem::align_of::<u32>())
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<u32>())
        })
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_non_native_u32_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, non_native_u32_dtype(), Some(257)).unwrap();
    assert_eq!(candidate_misaligned, former_misaligned);

    let mut group = c.benchmark_group("fromfile_non_native_u32_typed_byteswap");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_non_native_u32_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_byteswap_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), non_native_u32_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_native_i32(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<i32> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => i32::MIN,
            1 => i32::MAX,
            2 => -1,
            3 => 0,
            _ => (index as i32).wrapping_mul(65_537).wrapping_sub(17),
        })
        .collect();
    let bytes: &[u8] = bytemuck::cast_slice(&values);
    let former = fromfile_i32_former(bytes, None);
    let candidate = fromfile(bytes, native_i32_dtype(), None).unwrap();
    assert_eq!(candidate, former);

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<i32>()];
    let misaligned_offset = (0..core::mem::align_of::<i32>())
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<i32>())
        })
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_i32_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, native_i32_dtype(), Some(257)).unwrap();
    assert_eq!(candidate_misaligned, former_misaligned);

    let mut group = c.benchmark_group("fromfile_native_i32_typed_slice");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_i32_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), native_i32_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_non_native_i32(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<i32> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => i32::MIN,
            1 => i32::MAX,
            2 => -1,
            3 => 0,
            _ => (index as i32).wrapping_mul(65_537).wrapping_sub(17),
        })
        .collect();
    let stored: Vec<i32> = values.iter().map(|&value| value.swap_bytes()).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&stored);
    let former = fromfile_non_native_i32_former(bytes, None);
    let candidate = fromfile(bytes, non_native_i32_dtype(), None).unwrap();
    assert_eq!(candidate, former);

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<i32>()];
    let misaligned_offset = (0..core::mem::align_of::<i32>())
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<i32>())
        })
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_non_native_i32_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, non_native_i32_dtype(), Some(257)).unwrap();
    assert_eq!(candidate_misaligned, former_misaligned);

    let mut group = c.benchmark_group("fromfile_non_native_i32_typed_byteswap");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_non_native_i32_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_byteswap_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), non_native_i32_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_native_f32(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<f32> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => -0.0,
            1 => f32::from_bits(0x7fc0_0042),
            2 => f32::INFINITY,
            3 => f32::NEG_INFINITY,
            _ => index as f32 * 0.25 - 17.0,
        })
        .collect();
    let bytes: &[u8] = bytemuck::cast_slice(&values);
    let former = fromfile_f32_former(bytes, None);
    let candidate = fromfile(bytes, native_f32_dtype(), None).unwrap();
    assert_eq!(
        candidate
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        former
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let mut misaligned = Vec::with_capacity(bytes.len() + 1);
    misaligned.push(0);
    misaligned.extend_from_slice(bytes);
    let former_misaligned = fromfile_f32_former(&misaligned[1..], Some(257));
    let candidate_misaligned = fromfile(&misaligned[1..], native_f32_dtype(), Some(257)).unwrap();
    assert_eq!(
        candidate_misaligned
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        former_misaligned
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let mut group = c.benchmark_group("fromfile_native_f32_typed_slice");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_f32_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), native_f32_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_non_native_f32(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let bits: Vec<u32> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => 0.0_f32.to_bits(),
            1 => (-0.0_f32).to_bits(),
            2 => f32::INFINITY.to_bits(),
            3 => f32::NEG_INFINITY.to_bits(),
            4 => 1,
            5 => 0x7f7f_ffff,
            6 => 0x7fc0_0042,
            _ => (index as u32)
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223),
        })
        .collect();
    let stored: Vec<u32> = bits.iter().map(|&value| value.swap_bytes()).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&stored);
    let former = fromfile_non_native_f32_former(bytes, None);
    let candidate = fromfile(bytes, non_native_f32_dtype(), None).unwrap();
    assert!(
        candidate
            .iter()
            .zip(&former)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<u32>()];
    let misaligned_offset = (0..core::mem::align_of::<u32>())
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<u32>())
        })
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_non_native_f32_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, non_native_f32_dtype(), Some(257)).unwrap();
    assert!(
        candidate_misaligned
            .iter()
            .zip(&former_misaligned)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut group = c.benchmark_group("fromfile_non_native_f32_typed_byteswap");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_non_native_f32_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_byteswap_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), non_native_f32_dtype(), None).unwrap()))
    });
    group.finish();
}

fn load_npy_owned_body_former(data: &[u8]) -> (Vec<usize>, Vec<f64>, IOSupportedDType) {
    let npy = read_npy_bytes(data, false).expect("read NPY");
    let dtype = npy.header.descr;
    let shape = npy.header.shape;
    let values = fromfile(npy.payload.as_ref(), dtype, None).expect("decode NPY body");
    (shape, values, dtype)
}

fn assert_loaded_f64_bits_eq(
    former: &(Vec<usize>, Vec<f64>, IOSupportedDType),
    current: &(Vec<usize>, Vec<f64>, IOSupportedDType),
) {
    assert_eq!(current.0, former.0);
    assert_eq!(current.2, former.2);
    assert_eq!(current.1.len(), former.1.len());
    for (current, former) in current.1.iter().zip(&former.1) {
        assert_eq!(current.to_bits(), former.to_bits());
    }
}

fn bench_load_npy_borrowed_body(c: &mut Criterion) {
    const ELEMENTS: usize = 1_000_000;

    let data = generate_f64_data(ELEMENTS);
    let header = make_npy_header(&[ELEMENTS]);
    let npy_bytes = write_npy_bytes(&header, &data, false).expect("write NPY");
    let owned_npy = read_npy_bytes(&npy_bytes, false).expect("read NPY profile fixture");
    let former = load_npy_owned_body_former(&npy_bytes);
    let current = load(&npy_bytes).expect("load NPY");
    assert_loaded_f64_bits_eq(&former, &current);

    let mut group = c.benchmark_group("load_npy_borrowed_body");
    group.throughput(Throughput::Bytes(npy_bytes.len() as u64));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.bench_function("profile_parse_and_owned_body_copy", |bench| {
        bench.iter(|| black_box(read_npy_bytes(black_box(&npy_bytes), false).expect("read NPY")))
    });
    group.bench_function("profile_decode_owned_body", |bench| {
        bench.iter(|| {
            black_box(
                fromfile(
                    black_box(owned_npy.payload.as_ref()),
                    owned_npy.header.descr,
                    None,
                )
                .expect("decode NPY body"),
            )
        })
    });
    group.bench_function("former_owned_body_copy", |bench| {
        bench.iter(|| black_box(load_npy_owned_body_former(black_box(&npy_bytes))))
    });
    group.bench_function("public_load", |bench| {
        bench.iter(|| black_box(load(black_box(&npy_bytes)).expect("load NPY")))
    });
    group.finish();
}

#[inline(never)]
fn fromfile_structured_single_field_former(
    data: &[u8],
    descriptor: &StructuredIODescriptor,
    count: Option<usize>,
) -> StructuredNpyData {
    let record_size = descriptor.record_size().expect("valid descriptor");
    let max_records = data.len() / record_size;
    let n = count.map_or(max_records, |requested| requested.min(max_records));
    let offsets = descriptor.field_offsets().expect("field offsets");
    let mut columns: Vec<Vec<u8>> = descriptor
        .fields
        .iter()
        .map(|field| {
            let size = field.dtype.item_size().expect("sized dtype");
            Vec::with_capacity(n * size)
        })
        .collect();

    for record_idx in 0..n {
        let record_start = record_idx * record_size;
        for (field_idx, field) in descriptor.fields.iter().enumerate() {
            let field_size = field.dtype.item_size().expect("sized dtype");
            let field_start = record_start + offsets[field_idx];
            let field_end = field_start + field_size;
            columns[field_idx].extend_from_slice(&data[field_start..field_end]);
        }
    }

    StructuredNpyData {
        shape: vec![n],
        fortran_order: false,
        descriptor: descriptor.clone(),
        columns,
    }
}

fn assert_structured_data_eq(lhs: &StructuredNpyData, rhs: &StructuredNpyData) {
    assert_eq!(lhs.shape, rhs.shape);
    assert_eq!(lhs.fortran_order, rhs.fortran_order);
    assert_eq!(lhs.descriptor, rhs.descriptor);
    assert_eq!(lhs.columns, rhs.columns);
}

fn bench_fromfile_structured_single_field(c: &mut Criterion) {
    const RECORDS: usize = 1_048_576;

    let descriptor = StructuredIODescriptor {
        fields: vec![StructuredIOField {
            name: "value".to_string(),
            dtype: IOSupportedDType::F64,
        }],
    };
    let values: Vec<u64> = (0..RECORDS)
        .map(|index| (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .collect();
    let data = bytemuck::cast_slice::<u64, u8>(&values);
    let current = fromfile_structured(data, &descriptor, None).expect("structured read");
    let former = fromfile_structured_single_field_former(data, &descriptor, None);
    assert_structured_data_eq(&current, &former);

    let mut group = c.benchmark_group("fromfile_structured_single_field");
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.bench_function("former_exact_record_loop", |bench| {
        bench.iter(|| {
            black_box(fromfile_structured_single_field_former(
                black_box(data),
                black_box(&descriptor),
                None,
            ))
        })
    });
    group.bench_function("public_single_prefix_bulk_copy", |bench| {
        bench.iter(|| {
            black_box(
                fromfile_structured(black_box(data), black_box(&descriptor), None)
                    .expect("structured read"),
            )
        })
    });
    group.finish();
}

fn bench_write_npy(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_npy_bytes");

    for n in [1_000, 10_000, 100_000, 1_000_000] {
        let data = generate_f64_data(n);
        let header = make_npy_header(&[n]);

        group.throughput(Throughput::Bytes((n * 8) as u64));
        group.bench_with_input(BenchmarkId::new("elements", n), &n, |bench, _| {
            bench.iter(|| {
                let result = write_npy_bytes(black_box(&header), black_box(&data), false);
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_read_npy(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_npy_bytes");

    for n in [1_000, 10_000, 100_000, 1_000_000] {
        let data = generate_f64_data(n);
        let header = make_npy_header(&[n]);
        let npy_bytes = write_npy_bytes(&header, &data, false).expect("write");

        group.throughput(Throughput::Bytes(npy_bytes.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("elements", n),
            &npy_bytes,
            |bench, payload| {
                bench.iter(|| {
                    let result = read_npy_bytes(black_box(payload), false);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn bench_write_npz(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_npz_bytes");

    for num_arrays in [1, 5, 10, 20] {
        let n = 10_000;
        let data = generate_f64_data(n);
        let header = make_npy_header(&[n]);

        let entries: Vec<(String, NpyHeader, Vec<u8>)> = (0..num_arrays)
            .map(|i| (format!("arr_{i}"), header.clone(), data.clone()))
            .collect();

        let entry_refs: Vec<(&str, &NpyHeader, &[u8])> = entries
            .iter()
            .map(|(name, h, d)| (name.as_str(), h, d.as_slice()))
            .collect();

        let total_bytes = (num_arrays * n * 8) as u64;
        group.throughput(Throughput::Bytes(total_bytes));
        group.bench_with_input(
            BenchmarkId::new("num_arrays", num_arrays),
            &entry_refs,
            |bench, refs| {
                bench.iter(|| {
                    let result = write_npz_bytes(black_box(refs));
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn bench_read_npz(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_npz_bytes");

    for num_arrays in [1, 5, 10, 20] {
        let n = 10_000;
        let data = generate_f64_data(n);
        let header = make_npy_header(&[n]);

        let entries: Vec<(String, NpyHeader, Vec<u8>)> = (0..num_arrays)
            .map(|i| (format!("arr_{i}"), header.clone(), data.clone()))
            .collect();

        let entry_refs: Vec<(&str, &NpyHeader, &[u8])> = entries
            .iter()
            .map(|(name, h, d)| (name.as_str(), h, d.as_slice()))
            .collect();

        let npz_bytes = write_npz_bytes(&entry_refs).expect("write npz");

        group.throughput(Throughput::Bytes(npz_bytes.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("num_arrays", num_arrays),
            &npz_bytes,
            |bench, payload| {
                bench.iter(|| {
                    let result = read_npz_bytes(black_box(payload), false);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn bench_read_npz_overlap_tracking(c: &mut Criterion) {
    // Maximum legal member count, but only one f64 per member: this isolates
    // ZIP metadata validation rather than payload bandwidth.
    let num_arrays = 4_096usize;
    let data = generate_f64_data(1);
    let header = make_npy_header(&[1]);
    let entries: Vec<(String, NpyHeader, Vec<u8>)> = (0..num_arrays)
        .map(|i| (format!("arr_{i}"), header.clone(), data.clone()))
        .collect();
    let entry_refs: Vec<(&str, &NpyHeader, &[u8])> = entries
        .iter()
        .map(|(name, h, d)| (name.as_str(), h, d.as_slice()))
        .collect();
    let npz_bytes = write_npz_bytes(&entry_refs).expect("write metadata-heavy npz");

    let linear =
        read_npz_bytes_linear_overlap_control(&npz_bytes, false).expect("linear overlap control");
    let ordered = read_npz_bytes(&npz_bytes, false).expect("ordered overlap candidate");
    assert_eq!(ordered, linear);

    let mut group = c.benchmark_group("read_npz_overlap_tracking");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_secs(1));
    group.bench_function("linear_control_4096", |bench| {
        bench.iter(|| {
            black_box(read_npz_bytes_linear_overlap_control(black_box(&npz_bytes), false).unwrap())
        })
    });
    group.bench_function("ordered_candidate_4096", |bench| {
        bench.iter(|| black_box(read_npz_bytes(black_box(&npz_bytes), false).unwrap()))
    });
    group.finish();
}

fn bench_npy_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("npy_roundtrip");

    for n in [10_000, 100_000] {
        let data = generate_f64_data(n);
        let header = make_npy_header(&[n]);

        group.throughput(Throughput::Bytes((n * 8) as u64));
        group.bench_with_input(BenchmarkId::new("elements", n), &n, |bench, _| {
            bench.iter(|| {
                let written = write_npy_bytes(black_box(&header), black_box(&data), false).unwrap();
                let read = read_npy_bytes(black_box(&written), false).unwrap();
                black_box(read)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_fromfile_text_bounded_prefix,
    bench_fromfile_text_literal_bounded_prefix,
    bench_fromfile_text_wildcard_bounded_prefix,
    bench_loadtxt_usecols_plan,
    bench_loadtxt_usecols_scatter,
    bench_loadtxt_selected_bool_direct_parse,
    bench_loadtxt_signed_nonnegative_staging,
    bench_loadtxt_signed_tail_staging,
    bench_loadtxt_plain_rows,
    bench_genfromtxt_full_plain_rows,
    bench_tofile_text_integral,
    bench_genfromtxt_row_scratch,
    bench_fromfile_native_u64,
    bench_fromfile_non_native_u64,
    bench_fromfile_native_i64,
    bench_fromfile_non_native_i64,
    bench_fromfile_non_native_f64,
    bench_fromfile_native_i16,
    bench_fromfile_non_native_i16,
    bench_fromfile_native_u16,
    bench_fromfile_non_native_u16,
    bench_fromfile_native_u32,
    bench_fromfile_non_native_u32,
    bench_fromfile_native_i32,
    bench_fromfile_non_native_i32,
    bench_fromfile_native_f32,
    bench_fromfile_non_native_f32,
    bench_load_npy_borrowed_body,
    bench_fromfile_structured_single_field,
    bench_write_npy,
    bench_read_npy,
    bench_write_npz,
    bench_read_npz,
    bench_read_npz_overlap_tracking,
    bench_npy_roundtrip,
);

fn main() {
    report_bench_identity();
    verify_median_ci_gate_semantics();
    if std::env::var_os("FNP_IO_TOFILE_RESURRECTION_ONLY").is_some() {
        run_tofile_text_integral_contract();
        return;
    }
    benches();
    Criterion::default().configure_from_args().final_summary();
}
