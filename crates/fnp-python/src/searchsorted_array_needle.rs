//! Exact small-array f64 `searchsorted` helpers.
//!
//! NumPy orders NaN after every finite value for searchsorted.  The sorted-needle
//! batch below keeps that order, including duplicate and signed-zero boundaries,
//! while avoiding the sparse-query gallop that costs more probes than a lower
//! bound for the common tiny array-needle call.

#[inline]
fn sorts_before_insertion(probe: f64, key: f64, right: bool) -> bool {
    if right {
        !(key < probe || (probe.is_nan() && !key.is_nan()))
    } else {
        probe < key || (key.is_nan() && !probe.is_nan())
    }
}

#[inline]
fn numpy_nondecreasing(left: f64, right: f64) -> bool {
    !(right < left || (left.is_nan() && !right.is_nan()))
}

/// Fill `out` for a nondecreasing f64 needle batch and return whether that
/// admission predicate held. Each lower/upper bound has fixed halving shape;
/// LLVM lowers the update selects without a data-dependent branch on x86.
pub(crate) fn f64_sorted_needle_indices(
    haystack: &[f64],
    needles: &[f64],
    right: bool,
    out: &mut [i64],
) -> bool {
    debug_assert_eq!(needles.len(), out.len());
    if needles
        .windows(2)
        .any(|pair| !numpy_nondecreasing(pair[0], pair[1]))
    {
        return false;
    }

    for (slot, &key) in out.iter_mut().zip(needles) {
        let mut left = 0usize;
        let mut len = haystack.len();
        while len > 0 {
            let half = len / 2;
            let mid = left + half;
            let advance = usize::from(sorts_before_insertion(haystack[mid], key, right));
            let next_left = mid + 1;
            let next_len = len - half - 1;
            left = if advance == 1 { next_left } else { left };
            len = if advance == 1 { next_len } else { half };
        }
        *slot = left as i64;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::f64_sorted_needle_indices;

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
