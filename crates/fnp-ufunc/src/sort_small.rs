//! Small fixed-width value-sort kernels.
//!
//! These are deliberately dtype-specific.  A value sort of `i64` has no
//! NaN/signed-zero ordering surface, and equal values have identical bytes, so
//! Rust's unstable integer order is byte-exact for every NumPy `kind`.

/// Sort a short `int64` value slice in ascending NumPy order.
///
/// `slice::sort_unstable` uses Rust's pattern-defeating quicksort with its
/// small-slice sorting network/insertion base cases.  It avoids Rayon setup on
/// the tiny flat Python route while retaining the proven large-array kernel.
#[inline]
pub fn sort_i64(values: &mut [i64]) {
    values.sort_unstable();
}

#[cfg(test)]
mod tests {
    use super::sort_i64;

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
}
