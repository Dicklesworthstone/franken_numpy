//! Standalone batched A/B for the `i64` small-sort kernel: `slice::sort_unstable`
//! against the `std::simd` bitonic network, both in ONE process on the same corpus.
//!
//! This exists to satisfy one specific, written retry predicate. `docs/NEGATIVE_EVIDENCE.md`
//! (2026-08-26, `deadlock-audit-call-shape-priced-25ns-lk8zb`) closed the kernel lever for the
//! `sort(int64, n=256)` cell on an AVX2 host and named the only door back in:
//!
//!   "A next attempt must (a) use `std::simd`, (b) beat `sort_unstable`'s 1062 ns at n=256 in the
//!    standalone batched harness BEFORE any route-level claim, and (c) be measured under
//!    `bench_flat_i64_sort_256_dual_null` with interleaved arms."
//!
//! This is (a) and (b). It makes NO vs-NumPy claim and cannot: there is no incumbent in this
//! process. If the network does not beat `sort_unstable` here, the lever is a measured LOSS and
//! (c) must never be run.
//!
//! Method, following the ledger's own note that a single `Instant` pair costs ~20 ns at 10 ns
//! granularity: time a BATCH of `BATCH` sorts per timer pair, and measure the refill loop that
//! restores the unsorted corpus as its own arm so it can be subtracted. Sorting an
//! already-sorted slice flatters pattern-defeating quicksort enormously, so every batch starts
//! from the same unsorted corpus. Arms are interleaved ABBA/BAAB and reported as medians of
//! round medians, with an A/A null between two identical `sort_unstable` arms.
//!
//! `RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- \
//!    cargo run --release -p fnp-ufunc --example h2h_sort_i64_kernel`

use std::hint::black_box;
use std::time::Instant;

use fnp_ufunc::sort_small::{sort_i64_network_forced, sort_i64_unstable_forced};

const BATCH: usize = 64;
const ROUNDS: usize = 21;
const REPS: usize = 40;

fn xorshift(state: &mut u64) -> i64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state as i64
}

/// Nanoseconds per sort for one arm, refill cost already netted out.
fn time_arm(corpus: &[i64], scratch: &mut [i64], n: usize, kernel: fn(&mut [i64])) -> f64 {
    let start = Instant::now();
    for _ in 0..REPS {
        scratch.copy_from_slice(corpus);
        for chunk in scratch.chunks_exact_mut(n) {
            kernel(black_box(chunk));
        }
        black_box(&scratch[0]);
    }
    let total = start.elapsed().as_nanos() as f64;
    total / (REPS * BATCH) as f64
}

/// The refill loop alone, at the same repetition count, so it can be subtracted.
fn time_refill(corpus: &[i64], scratch: &mut [i64]) -> f64 {
    let start = Instant::now();
    for _ in 0..REPS {
        scratch.copy_from_slice(corpus);
        black_box(&scratch[0]);
    }
    let total = start.elapsed().as_nanos() as f64;
    total / (REPS * BATCH) as f64
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

/// Interleave two kernels ABBA/BAAB across rounds and return their median ns/sort.
fn interleaved(
    corpus: &[i64],
    scratch: &mut [i64],
    n: usize,
    a: fn(&mut [i64]),
    b: fn(&mut [i64]),
) -> (f64, f64) {
    let mut ta = Vec::with_capacity(ROUNDS * 2);
    let mut tb = Vec::with_capacity(ROUNDS * 2);
    for r in 0..ROUNDS {
        let order = if r % 2 == 0 {
            [true, false, false, true]
        } else {
            [false, true, true, false]
        };
        for is_a in order {
            let t = if is_a {
                time_arm(corpus, scratch, n, a)
            } else {
                time_arm(corpus, scratch, n, b)
            };
            if is_a { &mut ta } else { &mut tb }.push(t);
        }
    }
    (median(&mut ta), median(&mut tb))
}

fn main() {
    println!(
        "host={} rounds={ROUNDS} reps={REPS} batch={BATCH} (medians of interleaved ABBA rounds)",
        std::fs::read_to_string("/proc/sys/kernel/hostname")
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| "unknown".into())
    );
    println!(
        "{:>6}{:>14}{:>14}{:>12}{:>10}{:>12}",
        "n", "unstable_ns", "bitonic_ns", "speedup", "AA_null", "refill_ns"
    );

    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut verified = 0usize;
    for n in [64usize, 96, 128, 192, 256] {
        let corpus: Vec<i64> = (0..n * BATCH).map(|_| xorshift(&mut state)).collect();
        let mut scratch = vec![0i64; n * BATCH];

        // PARITY BEFORE TIMING: a faster kernel that returns a different multiset is not a
        // result. Both arms are checked on this exact corpus before either is timed.
        for chunk_index in 0..BATCH {
            let src = &corpus[chunk_index * n..(chunk_index + 1) * n];
            let mut want = src.to_vec();
            want.sort_unstable();
            let mut got = src.to_vec();
            sort_i64_network_forced(&mut got);
            assert_eq!(got, want, "network diverged at n={n} chunk={chunk_index}");
            verified += 1;
        }

        let refill = time_refill(&corpus, &mut scratch);
        let (unstable, bitonic) = interleaved(
            &corpus,
            &mut scratch,
            n,
            sort_i64_unstable_forced,
            sort_i64_network_forced,
        );
        let (null_a, null_b) = interleaved(
            &corpus,
            &mut scratch,
            n,
            sort_i64_unstable_forced,
            sort_i64_unstable_forced,
        );
        let (u, b) = (unstable - refill, bitonic - refill);
        println!(
            "{n:>6}{:>14.1}{:>14.1}{:>11.3}x{:>10.3}{:>12.1}",
            u,
            b,
            u / b,
            null_b / null_a,
            refill
        );
    }
    println!("parity: {verified} slices, 0 divergences (checked before any timing)");
}
