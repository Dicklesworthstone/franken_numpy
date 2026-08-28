//! Exact small-array `searchsorted` helpers for f32 and f64.
//!
//! NumPy orders NaN after every finite value for searchsorted.  The sorted-needle
//! batch below keeps that order, including duplicate and signed-zero boundaries,
//! while avoiding the sparse-query gallop that costs more probes than a lower
//! bound for the common tiny array-needle call.
//!
//! GENERIC OVER THE FLOAT WIDTH WAS BUILT AND REJECTED. A `T: Copy + PartialOrd` version of these
//! four functions - one implementation instead of two - measured 1.374x-1.381x of NumPy's
//! instructions where the concrete f64 code measures 0.867x-0.889x, and `#[inline(always)]` on the
//! predicates did not recover it. In generic code `probe < key` goes through `PartialOrd::lt`,
//! whose default body materialises an `Option<Ordering>` from `partial_cmp`; on a concrete float
//! it is a single compare. The macro below keeps one source of truth without paying that.

macro_rules! searchsorted_needle_helpers {
    ($float:ty, $sorted:ident, $branchless:ident, $before:ident, $nondec:ident) => {
        #[inline]
        fn $before(probe: $float, key: $float, right: bool) -> bool {
            if right {
                !(key < probe || (probe.is_nan() && !key.is_nan()))
            } else {
                probe < key || (key.is_nan() && !probe.is_nan())
            }
        }

        #[inline]
        fn $nondec(left: $float, right: $float) -> bool {
            !(right < left || (left.is_nan() && !right.is_nan()))
        }

        /// Fill `out` for a nondecreasing needle batch and return whether that admission
        /// predicate held. Each lower/upper bound has fixed halving shape; LLVM lowers the
        /// update selects without a data-dependent branch on x86.
        pub(crate) fn $sorted(
            haystack: &[$float],
            needles: &[$float],
            right: bool,
            out: &mut [i64],
        ) -> bool {
            debug_assert_eq!(needles.len(), out.len());
            if needles.windows(2).any(|pair| !$nondec(pair[0], pair[1])) {
                return false;
            }
            $branchless(haystack, needles, right, out);
            true
        }

        /// The same fixed-shape bound, with NO sortedness precondition.
        ///
        /// The per-key bound does not depend on the ORDER of the needles - the nondecreasing test
        /// above is an admission predicate for the batch, not a requirement of this loop - so the
        /// identical branchless fill serves an unsorted batch, and that is where it is worth the
        /// most. The caller's alternative for unsorted needles was the guess-seeded GALLOP, which
        /// is the wrong shape twice over: consecutive random keys make the hint worthless, and
        /// every probe is a data-dependent branch that a random key stream mispredicts about half
        /// the time. Counted at a 2^20 haystack, OPENBLAS pinned, unsorted f64 queries ran 1.320x
        /// NumPy's instructions on the gallop against 0.867x on this loop.
        pub(crate) fn $branchless(
            haystack: &[$float],
            needles: &[$float],
            right: bool,
            out: &mut [i64],
        ) {
            debug_assert_eq!(needles.len(), out.len());
            for (slot, &key) in out.iter_mut().zip(needles) {
                let mut left = 0usize;
                let mut len = haystack.len();
                while len > 0 {
                    let half = len / 2;
                    let mid = left + half;
                    let advance = usize::from($before(haystack[mid], key, right));
                    let next_left = mid + 1;
                    let next_len = len - half - 1;
                    left = if advance == 1 { next_left } else { left };
                    len = if advance == 1 { next_len } else { half };
                }
                *slot = left as i64;
            }
        }
    };
}

searchsorted_needle_helpers!(
    f64,
    f64_sorted_needle_indices,
    f64_needle_indices_branchless,
    f64_sorts_before_insertion,
    f64_numpy_nondecreasing
);
searchsorted_needle_helpers!(
    f32,
    f32_sorted_needle_indices,
    f32_needle_indices_branchless,
    f32_sorts_before_insertion,
    f32_numpy_nondecreasing
);

#[cfg(test)]
mod tests {
    use super::{f32_sorted_needle_indices, f64_sorted_needle_indices};

    #[test]
    fn sorted_batch_is_nan_last_and_side_exact() {
        let haystack = [-f64::INFINITY, -0.0, 0.0, 1.0, 1.0, f64::INFINITY, f64::NAN];
        let needles = [-f64::INFINITY, -0.0, 0.0, 1.0, f64::INFINITY, f64::NAN];
        let mut left = [0_i64; 6];
        let mut right = [0_i64; 6];
        assert!(f64_sorted_needle_indices(
            &haystack, &needles, false, &mut left
        ));
        assert!(f64_sorted_needle_indices(
            &haystack, &needles, true, &mut right
        ));
        assert_eq!(left, [0, 1, 1, 3, 5, 6]);
        assert_eq!(right, [1, 3, 3, 5, 6, 7]);
    }

    #[test]
    fn f32_sorted_batch_matches_the_f64_shape() {
        let haystack = [-f32::INFINITY, -0.0, 0.0, 1.0, 1.0, f32::INFINITY, f32::NAN];
        let needles = [-f32::INFINITY, -0.0, 0.0, 1.0, f32::INFINITY, f32::NAN];
        let mut left = [0_i64; 6];
        let mut right = [0_i64; 6];
        assert!(f32_sorted_needle_indices(
            &haystack, &needles, false, &mut left
        ));
        assert!(f32_sorted_needle_indices(
            &haystack, &needles, true, &mut right
        ));
        assert_eq!(left, [0, 1, 1, 3, 5, 6]);
        assert_eq!(right, [1, 3, 3, 5, 6, 7]);
    }

    #[test]
    fn unsorted_needles_are_a_planted_negative() {
        let mut out = [0_i64; 2];
        assert!(!f64_sorted_needle_indices(
            &[0.0, 1.0],
            &[1.0, 0.0],
            false,
            &mut out
        ));
    }
}
