//! Small fixed-width value-sort kernels.
//!
//! These are deliberately dtype-specific.  A value sort of `i64` has no
//! NaN/signed-zero ordering surface, and equal values have identical bytes, so
//! Rust's unstable integer order is byte-exact for every NumPy `kind`.

use std::simd::cmp::SimdOrd;
use std::simd::{Mask, Simd, simd_swizzle};

const LANES: usize = 4;
type V = Simd<i64, LANES>;
/// The mask type must be named explicitly: `Mask::from_array` alone leaves the
/// element type open, and the inherent `select` cannot then be resolved.
type M = Mask<i64, LANES>;

/// Largest length the branchless network handles; above it `sort_unstable` is
/// the kernel.  256 is the size of the campaign's measured cell and is the
/// point at which a bitonic network's `O(n log^2 n)` comparator count is still
/// paying for its branchlessness.
const BITONIC_MAX: usize = 256;

/// Below this the padded network cannot win: the scratch buffer's fill and the
/// write-back are fixed costs, while `sort_unstable`'s own insertion base case
/// is already near-optimal on a handful of elements.  The AVX2 attempt recorded
/// in the ledger lost at n=32 and n=64 precisely here.
const BITONIC_MIN: usize = 64;

const MAX_VECS: usize = BITONIC_MAX / LANES;

/// One intra-register bitonic stage: `j` is 2 or 1, i.e. the two stages whose
/// comparator partners live inside a single 4-lane register and therefore need
/// a lane shuffle rather than a second register.
///
/// `asc` is the direction of the whole register, which is well defined because
/// these stages only ever run with `k >= 4` — the sub-sequence being merged is
/// at least as wide as the register, so every lane in it shares a direction.
/// The `k == 2` stage is the sole exception and is peeled out below.
#[inline]
fn intra_stage(v: V, j: usize, asc: bool) -> V {
    let partner = if j == 2 {
        simd_swizzle!(v, [2, 3, 0, 1])
    } else {
        simd_swizzle!(v, [1, 0, 3, 2])
    };
    let mn = v.simd_min(partner);
    let mx = v.simd_max(partner);
    // Lanes selected from `mn` are the ones holding the LOW index of their
    // comparator pair when ascending, and the high index when descending.
    let take_min = match (j, asc) {
        (2, true) => M::from_array([true, true, false, false]),
        (2, false) => M::from_array([false, false, true, true]),
        (_, true) => M::from_array([true, false, true, false]),
        (_, false) => M::from_array([false, true, false, true]),
    };
    take_min.select(mn, mx)
}

/// Bitonic sort of `buf` in ascending order.  `buf.len()` must be a power of
/// two, so the element count `4 * buf.len()` is one too.
///
/// The standard iterative bitonic network, split by where a comparator's two
/// operands live.  With `j >= LANES` the partner of an element is in ANOTHER
/// register at the same lane, so the comparator is a plain elementwise
/// `min`/`max` of two registers and costs no shuffle at all; that is where a
/// vector bitonic sort earns its keep.  With `j < LANES` the partner is another
/// lane of the same register and needs `intra_stage`.
fn bitonic_sort_pow2(buf: &mut [V]) {
    let nv = buf.len();
    let n = nv * LANES;

    // k == 2: the only stage whose direction alternates WITHIN a register.
    // Pairs (lane0,lane1) ascend and (lane2,lane3) descend, identically in
    // every register, so the mask is a constant.
    for v in buf.iter_mut() {
        let partner = simd_swizzle!(*v, [1, 0, 3, 2]);
        let mn = v.simd_min(partner);
        let mx = v.simd_max(partner);
        *v = M::from_array([true, false, false, true]).select(mn, mx);
    }

    let mut k = 4;
    while k <= n {
        let mut j = k / 2;
        while j >= LANES {
            let block = j / LANES;
            for v in 0..nv {
                if v & block == 0 {
                    let p = v | block;
                    let asc = ((v * LANES) & k) == 0;
                    let a = buf[v];
                    let b = buf[p];
                    let mn = a.simd_min(b);
                    let mx = a.simd_max(b);
                    if asc {
                        buf[v] = mn;
                        buf[p] = mx;
                    } else {
                        buf[v] = mx;
                        buf[p] = mn;
                    }
                }
            }
            j /= 2;
        }
        for jj in [2usize, 1] {
            for v in 0..nv {
                let asc = ((v * LANES) & k) == 0;
                buf[v] = intra_stage(buf[v], jj, asc);
            }
        }
        k *= 2;
    }
}

/// Sort `values` (length `BITONIC_MIN..=BITONIC_MAX`) with the branchless
/// network.  Shorter or longer slices are not handled here.
///
/// The slice is padded up to a power of two with `i64::MAX`.  Padding with the
/// maximum keeps every real element ahead of every pad element, so truncating
/// the result back to `values.len()` yields exactly the sorted real multiset —
/// including when the real data itself contains `i64::MAX`, because equal
/// values are indistinguishable in a value sort.
fn sort_i64_network(values: &mut [i64]) {
    let n = values.len().next_power_of_two();
    let nv = n / LANES;
    let mut buf = [V::splat(i64::MAX); MAX_VECS];
    for (i, &x) in values.iter().enumerate() {
        buf[i / LANES][i % LANES] = x;
    }
    bitonic_sort_pow2(&mut buf[..nv]);
    for (i, slot) in values.iter_mut().enumerate() {
        *slot = buf[i / LANES][i % LANES];
    }
}

/// Sort a short `int64` value slice in ascending NumPy order.
///
/// `slice::sort_unstable` uses Rust's pattern-defeating quicksort with its
/// small-slice sorting network/insertion base cases.  It avoids Rayon setup on
/// the tiny flat Python route while retaining the proven large-array kernel.
/// In the `BITONIC_MIN..=BITONIC_MAX` window the branchless network above is
/// used instead when it has been measured to win; see `sort_i64_network`.
#[inline]
pub fn sort_i64(values: &mut [i64]) {
    if (BITONIC_MIN..=BITONIC_MAX).contains(&values.len()) && bitonic_enabled() {
        sort_i64_network(values);
    } else {
        values.sort_unstable();
    }
}

/// The network is OFF by default and must be switched on explicitly.
///
/// It exists so the standalone batched comparison the ledger's retry predicate
/// demands ("beat `sort_unstable`'s 1062 ns at n=256 BEFORE any route-level
/// claim") can run BOTH kernels in ONE process on the same corpus.  Shipping it
/// on by default before that gate passes would be exactly the unmeasured
/// route-level claim the predicate bars.
///
/// NOT an `env::var_os` per call.  `getenv` walks `environ` and allocates, and
/// this is read on the entry path of the very cell being measured — a route
/// whose WHOLE remaining deficit is 326 ns.  A switch that costs a hundred
/// nanoseconds to read would be paid by the shipped default arm and would
/// corrupt the number it exists to produce.  The environment is consulted once;
/// after that this is a relaxed atomic load.
#[inline]
fn bitonic_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("FNP_SORT_I64_BITONIC").is_some_and(|v| v == *"1"))
}

/// The comparison-sort arm, named so the standalone A/B harness can hold both
/// kernels as values of the same function type and interleave them.
pub fn sort_i64_unstable_forced(values: &mut [i64]) {
    values.sort_unstable();
}

/// Sort with the branchless network regardless of the env switch, for the
/// standalone A/B harness and the differential tests.  Falls back for lengths
/// the network does not cover.
pub fn sort_i64_network_forced(values: &mut [i64]) {
    if (BITONIC_MIN..=BITONIC_MAX).contains(&values.len()) {
        sort_i64_network(values);
    } else {
        values.sort_unstable();
    }
}

#[cfg(test)]
mod tests {
    use super::{BITONIC_MAX, BITONIC_MIN, sort_i64, sort_i64_network_forced};

    #[test]
    fn i64_small_sort_orders_extrema_and_duplicates() {
        let mut values = [i64::MAX, 0, -7, i64::MIN, -7, 1, 0, -1];
        sort_i64(&mut values);
        assert_eq!(values, [i64::MIN, -7, -7, -1, 0, 0, 1, i64::MAX]);
    }

    #[test]
    fn i64_small_sort_planted_negative_is_not_left_unsorted() {
        let mut values = [2_i64, -1, 1];
        sort_i64(&mut values);
        assert_ne!(values, [2, -1, 1], "the route must perform the sort");
        assert_eq!(values, [-1, 1, 2]);
    }

    /// A bitonic network is a FIXED comparator sequence, so a length that is
    /// not a power of two, a run of equal keys, or a value equal to the pad
    /// sentinel are the three ways it silently returns the wrong multiset.
    /// Every covered length is checked against `sort_unstable` on eight input
    /// shapes, which is the corpus the ledger's AVX2 attempt used.
    #[test]
    fn i64_network_matches_sort_unstable_over_every_covered_length() {
        let mut state = 0x243f_6a88_85a3_08d3_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as i64
        };
        for len in BITONIC_MIN..=BITONIC_MAX {
            for shape in 0..8 {
                let src: Vec<i64> = (0..len)
                    .map(|i| match shape {
                        0 => next(),
                        1 => next() % 4,
                        2 => 7,
                        3 => i as i64,
                        4 => (len - i) as i64,
                        5 => i64::MAX,
                        6 => i64::MIN,
                        _ => {
                            if i % 2 == 0 {
                                i64::MIN
                            } else {
                                i64::MAX
                            }
                        }
                    })
                    .collect();
                let mut want = src.clone();
                want.sort_unstable();
                let mut got = src.clone();
                sort_i64_network_forced(&mut got);
                assert_eq!(got, want, "len={len} shape={shape}");
            }
        }
    }

    /// Lengths outside the window must still sort — the dispatch is part of the
    /// kernel's contract, not an implementation detail.
    #[test]
    fn i64_sort_dispatch_covers_lengths_around_the_network_window() {
        for len in [0usize, 1, 2, 3, 63, 64, 255, 256, 257, 1000] {
            let src: Vec<i64> = (0..len).map(|i| ((len - i) as i64) * 3 - 5).collect();
            let mut want = src.clone();
            want.sort_unstable();
            let mut got = src.clone();
            sort_i64(&mut got);
            assert_eq!(got, want, "len={len}");
        }
    }
}
