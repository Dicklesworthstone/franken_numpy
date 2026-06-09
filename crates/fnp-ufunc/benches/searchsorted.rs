//! searchsorted kernel comparison — and the evidence for why the production
//! path stays a per-needle rayon-parallel branchy binary search.
//!
//! Three alternative kernels were measured against the current parallel branchy
//! search over a large sorted f64 haystack with many needles (bead 6gjpn):
//!
//!   * `eytzinger`  — BFS-layout branchless probe. LOSES (≈2.5x slower even
//!     single-threaded): the classic eytzinger win comes entirely from software
//!     prefetch hiding the deep-level cache misses, and `fnp-ufunc` is
//!     `#![forbid(unsafe_code)]` so no prefetch intrinsic is available. Without
//!     it the BFS layout is pure overhead (full-depth probes + O(n) build).
//!   * `merge`      — sort needles + two-pointer streaming merge. WINS ≈2-2.5x
//!     *single-threaded* (sequential access beats random probes) but...
//!   * `merge_par`  — the parallel chunked merge LOSES ≈4x to `branchy_par`:
//!     the O(n) sequential scan plus the O(q log q) needle sort cost more than
//!     the parallel branchy search, whose 64-way memory-level parallelism turns
//!     the latency-bound random gather into a bandwidth-bound one (~1.1ms for
//!     131072 queries over a 1M array).
//!
//! Conclusion: parallel branchy is the right algorithm for this hardware; the
//! eytzinger/merge levers are rejected. This bench is kept as the reproducible
//! evidence and a guard against naively swapping the kernel.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::hint::black_box;

#[inline]
fn membership_key(v: f64) -> u64 {
    if v.is_nan() {
        return u64::MAX;
    }
    let bits = if v == 0.0 { 0u64 } else { v.to_bits() };
    if bits & 0x8000_0000_0000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000_0000_0000
    }
}

fn membership_cmp(lhs: f64, rhs: f64) -> Ordering {
    match (lhs.is_nan(), rhs.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => {
            if lhs == 0.0 && rhs == 0.0 {
                Ordering::Equal
            } else {
                lhs.total_cmp(&rhs)
            }
        }
    }
}

/// Existing kernel: per-needle branchy binary search over the sorted slice.
fn branchy(data: &[f64], needles: &[f64]) -> Vec<i64> {
    let n = data.len();
    needles
        .iter()
        .map(|&needle| {
            let mut lo = 0usize;
            let mut hi = n;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if membership_cmp(data[mid], needle) == Ordering::Less {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo as i64
        })
        .collect()
}

fn eytzinger_fill(
    k: usize,
    n: usize,
    next: &mut usize,
    keys: &[u64],
    tree: &mut [u64],
    ranks: &mut [usize],
) {
    if k <= n {
        eytzinger_fill(2 * k, n, next, keys, tree, ranks);
        tree[k] = keys[*next];
        ranks[k] = *next;
        *next += 1;
        eytzinger_fill(2 * k + 1, n, next, keys, tree, ranks);
    }
}

/// New kernel: one-time eytzinger build + branchless probe per needle.
fn eytzinger(data: &[f64], needles: &[f64]) -> Vec<i64> {
    let n = data.len();
    let keys: Vec<u64> = data.iter().map(|&v| membership_key(v)).collect();
    let mut tree = vec![0u64; n + 1];
    let mut ranks = vec![0usize; n + 1];
    let mut next = 0usize;
    eytzinger_fill(1, n, &mut next, &keys, &mut tree, &mut ranks);
    needles
        .iter()
        .map(|&needle| {
            let nk = membership_key(needle);
            let mut k = 1usize;
            while k <= n {
                k = 2 * k + (tree[k] < nk) as usize;
            }
            let j = k >> ((!k).trailing_zeros() + 1);
            if j == 0 { n as i64 } else { ranks[j] as i64 }
        })
        .collect()
}

/// Candidate kernel: sort needles by key, two-pointer merge against the sorted
/// haystack (sequential streaming over both arrays), scatter ranks back.
fn merge(data: &[f64], needles: &[f64]) -> Vec<i64> {
    let n = data.len();
    let q = needles.len();
    let dkeys: Vec<u64> = data.iter().map(|&v| membership_key(v)).collect();
    let nkeys: Vec<u64> = needles.iter().map(|&v| membership_key(v)).collect();
    let mut order: Vec<u32> = (0..q as u32).collect();
    order.sort_unstable_by_key(|&i| nkeys[i as usize]);
    let mut out = vec![0i64; q];
    let mut p = 0usize;
    for &oi in &order {
        let nk = nkeys[oi as usize];
        while p < n && dkeys[p] < nk {
            p += 1;
        }
        out[oi as usize] = p as i64;
    }
    out
}

/// Current production baseline: per-needle branchy binary search, parallel
/// across needles via rayon (matches `UFuncArray::searchsorted`'s hot path).
fn branchy_parallel(data: &[f64], needles: &[f64]) -> Vec<i64> {
    let n = data.len();
    needles
        .par_iter()
        .map(|&needle| {
            let mut lo = 0usize;
            let mut hi = n;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if membership_cmp(data[mid], needle) == Ordering::Less {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo as i64
        })
        .collect()
}

/// Parallel two-pointer merge: parallel sort needles by key, split the sorted
/// needles into chunks, each chunk binary-searches its start pointer then
/// streams sequentially through the haystack. Disjoint output scatter.
fn merge_parallel(data: &[f64], needles: &[f64]) -> Vec<i64> {
    let n = data.len();
    let q = needles.len();
    let dkeys: Vec<u64> = data.iter().map(|&v| membership_key(v)).collect();
    let nkeys: Vec<u64> = needles.iter().map(|&v| membership_key(v)).collect();
    let mut order: Vec<u32> = (0..q as u32).collect();
    order.par_sort_unstable_by_key(|&i| nkeys[i as usize]);
    let mut out = vec![0i64; q];
    let chunks = (rayon::current_num_threads() * 4).max(1);
    let chunk_len = q.div_ceil(chunks);
    let out_slice = &mut out[..];
    // SAFETY-FREE scatter: each chunk writes disjoint original indices, but we
    // collect (idx, rank) pairs per chunk then apply, to keep it safe-Rust.
    let results: Vec<Vec<(u32, i64)>> = order
        .par_chunks(chunk_len)
        .map(|chunk| {
            let first = nkeys[chunk[0] as usize];
            // start pointer = lower_bound of first key in dkeys
            let mut p = dkeys.partition_point(|&d| d < first);
            let mut local = Vec::with_capacity(chunk.len());
            for &oi in chunk {
                let nk = nkeys[oi as usize];
                while p < n && dkeys[p] < nk {
                    p += 1;
                }
                local.push((oi, p as i64));
            }
            local
        })
        .collect();
    for local in &results {
        for &(oi, r) in local {
            out_slice[oi as usize] = r;
        }
    }
    out
}

fn bench_searchsorted(c: &mut Criterion) {
    // Target regime from the bead: large sorted array, many needles.
    for &(n, nq) in &[(1usize << 20, 1usize << 17), (1usize << 18, 1usize << 16)] {
        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 - 1e6).collect();
        let needles: Vec<f64> = (0..nq)
            .map(|i| {
                let r = ((i as u64).wrapping_mul(6364136223846793005).rotate_left(21)
                    % (n as u64 * 2)) as f64;
                r * 0.5 - 1.05e6
            })
            .collect();
        // Sanity: all kernels agree (cheap, runs once outside timing).
        let base = branchy(&data, &needles);
        assert_eq!(base, eytzinger(&data, &needles));
        assert_eq!(base, merge(&data, &needles));
        assert_eq!(base, branchy_parallel(&data, &needles));
        assert_eq!(base, merge_parallel(&data, &needles));

        let mut group = c.benchmark_group(format!("searchsorted_n{n}_q{nq}"));
        group.bench_with_input(BenchmarkId::new("branchy_par", nq), &nq, |b, _| {
            b.iter(|| black_box(branchy_parallel(black_box(&data), black_box(&needles))))
        });
        group.bench_with_input(BenchmarkId::new("merge_par", nq), &nq, |b, _| {
            b.iter(|| black_box(merge_parallel(black_box(&data), black_box(&needles))))
        });
        group.finish();
    }
}

criterion_group!(benches, bench_searchsorted);
criterion_main!(benches);
