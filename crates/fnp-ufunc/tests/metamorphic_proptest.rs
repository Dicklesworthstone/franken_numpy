//! Property-based metamorphic tests using proptest.
//!
//! Metamorphic testing verifies relations that must hold regardless of specific
//! input values, without requiring an exact oracle.

use fnp_dtype::DType;
use fnp_ufunc::{BinaryOp, UFuncArray, UnaryOp};
use proptest::prelude::*;

const EPSILON: f64 = 1e-10;

fn approx_eq_f64(a: f64, b: f64, eps: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }
    (a - b).abs() < eps
}

fn arrays_approx_eq(a: &UFuncArray, b: &UFuncArray, eps: f64) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    let a_vals = a.values();
    let b_vals = b.values();
    a_vals
        .iter()
        .zip(b_vals.iter())
        .all(|(x, y)| approx_eq_f64(*x, *y, eps))
}

fn arrays_bitwise_eq(a: &UFuncArray, b: &UFuncArray) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    let a_vals = a.values();
    let b_vals = b.values();
    a_vals
        .iter()
        .zip(b_vals.iter())
        .all(|(x, y)| x.to_bits() == y.to_bits())
}

fn from_vec_f64(v: Vec<f64>) -> UFuncArray {
    UFuncArray::from_vec(v)
}

fn from_2d_f64(data: Vec<Vec<f64>>) -> Option<UFuncArray> {
    if data.is_empty() {
        return None;
    }
    let rows = data.len();
    let cols = data[0].len();
    if cols == 0 || data.iter().any(|row| row.len() != cols) {
        return None;
    }
    let flat: Vec<f64> = data.into_iter().flatten().collect();
    UFuncArray::new(vec![rows, cols], flat, DType::F64).ok()
}

// Strategy for bounded f64 values to avoid overflow in sums
fn bounded_f64() -> impl Strategy<Value = f64> {
    (-1e10f64..1e10f64).prop_filter("must be finite", |x| x.is_finite())
}

// Strategy for small finite f64 values to avoid overflow
fn small_f64() -> impl Strategy<Value = f64> {
    (-1e6f64..1e6f64).prop_filter("must be finite", |x| x.is_finite())
}

// Strategy for vectors of bounded f64 (for tests involving sums)
fn finite_vec(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(bounded_f64(), min_len..=max_len)
}

// Strategy for 2D arrays (rows x cols of finite f64)
fn finite_2d(
    min_rows: usize,
    max_rows: usize,
    min_cols: usize,
    max_cols: usize,
) -> impl Strategy<Value = Vec<Vec<f64>>> {
    (min_rows..=max_rows).prop_flat_map(move |rows| {
        (min_cols..=max_cols).prop_flat_map(move |cols| {
            prop::collection::vec(
                prop::collection::vec(bounded_f64(), cols..=cols),
                rows..=rows,
            )
        })
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    // =========================================================================
    // MR1: Transpose twice equals identity
    // transpose(transpose(A)) == A
    // =========================================================================
    #[test]
    fn mr_transpose_twice_is_identity(data in finite_2d(1, 20, 1, 20)) {
        if let Some(arr) = from_2d_f64(data) {
            let t1 = arr.transpose(None).expect("first transpose");
            let t2 = t1.transpose(None).expect("second transpose");
            prop_assert!(
                arrays_bitwise_eq(&arr, &t2),
                "transpose(transpose(A)) should equal A. Got shape {:?} vs {:?}",
                arr.shape(),
                t2.shape()
            );
        }
    }

    // =========================================================================
    // MR2: Sum is permutation-invariant (within floating-point tolerance)
    // sum(reverse(x)) ≈ sum(x)
    // Note: Floating-point addition is not truly associative, so we use
    // relative tolerance that accounts for summation error growth.
    // Use small values to avoid overflow and reduce accumulation errors.
    // =========================================================================
    #[test]
    fn mr_sum_permutation_invariant(data in prop::collection::vec(small_f64(), 1..1000)) {
        let arr1 = from_vec_f64(data.clone());
        let sum1 = arr1.reduce_sum(None, false).expect("sum1");

        // Create a reversed copy as a simple permutation
        let mut permuted = data;
        permuted.reverse();

        let arr2 = from_vec_f64(permuted);
        let sum2 = arr2.reduce_sum(None, false).expect("sum2");

        // Use relative tolerance: sqrt(n) * epsilon * max(|sum|)
        let n = arr1.values().len() as f64;
        let rel_tol = n.sqrt() * 1e-12 * sum1.values()[0].abs().max(1.0);
        prop_assert!(
            approx_eq_f64(sum1.values()[0], sum2.values()[0], rel_tol),
            "sum should be permutation-invariant: {} vs {} (tol={})",
            sum1.values()[0],
            sum2.values()[0],
            rel_tol
        );
    }

    // =========================================================================
    // MR3: Mean is permutation-invariant (within floating-point tolerance)
    // mean(reverse(x)) ≈ mean(x)
    // Use small values to avoid floating-point accumulation errors
    // =========================================================================
    #[test]
    fn mr_mean_permutation_invariant(data in prop::collection::vec(small_f64(), 1..1000)) {
        let arr1 = from_vec_f64(data.clone());
        let mean1 = arr1.reduce_mean(None, false).expect("mean1");

        // Create a reversed copy as a simple permutation
        let mut permuted = data;
        permuted.reverse();

        let arr2 = from_vec_f64(permuted);
        let mean2 = arr2.reduce_mean(None, false).expect("mean2");

        // Use relative tolerance
        let n = arr1.values().len() as f64;
        let rel_tol = n.sqrt() * 1e-12 * mean1.values()[0].abs().max(1.0);
        prop_assert!(
            approx_eq_f64(mean1.values()[0], mean2.values()[0], rel_tol),
            "mean should be permutation-invariant: {} vs {} (tol={})",
            mean1.values()[0],
            mean2.values()[0],
            rel_tol
        );
    }

    // =========================================================================
    // MR4: Sort then reduce equals reduce (for commutative ops)
    // sum(sort(x)) ≈ sum(x) (within floating-point tolerance)
    // Use small values to avoid floating-point accumulation errors
    // =========================================================================
    #[test]
    fn mr_sort_then_sum_equals_sum(data in prop::collection::vec(small_f64(), 1..500)) {
        let arr = from_vec_f64(data);
        let sum_original = arr.reduce_sum(None, false).expect("sum original");

        let sorted = arr.sort(Some(-1), None).expect("sort");
        let sum_sorted = sorted.reduce_sum(None, false).expect("sum sorted");

        // Tolerance must scale with the CONDITIONING of the summation, not its
        // result. Reordering f64 additions perturbs the sum by ~eps * Σ|x_i|
        // (the magnitude that bounds floating accumulation error), so a sum with
        // catastrophic cancellation (|Σx_i| << Σ|x_i|) can drift far more than its
        // own tiny value. Scaling by |Σx_i|.max(1) understated this and failed on
        // near-cancelling inputs; scale by Σ|x_i| instead.
        let n = arr.values().len() as f64;
        let sum_abs: f64 = arr.values().iter().map(|v| v.abs()).sum();
        let rel_tol = n.sqrt() * 1e-12 * sum_abs.max(1.0);
        prop_assert!(
            approx_eq_f64(
                sum_original.values()[0],
                sum_sorted.values()[0],
                rel_tol
            ),
            "sum(sort(x)) should equal sum(x): {} vs {} (tol={})",
            sum_original.values()[0],
            sum_sorted.values()[0],
            rel_tol
        );
    }

    // =========================================================================
    // MR5: Reverse twice equals identity
    // flip(flip(x)) == x
    // =========================================================================
    #[test]
    fn mr_reverse_twice_is_identity(data in finite_vec(1, 500)) {
        let arr = from_vec_f64(data);

        // Reverse using flip
        let rev1 = arr.flip(Some(0)).expect("first flip");
        let rev2 = rev1.flip(Some(0)).expect("second flip");

        prop_assert!(
            arrays_bitwise_eq(&arr, &rev2),
            "reverse(reverse(x)) should equal x"
        );
    }

    // =========================================================================
    // MR6: Concatenate then split equals identity
    // split_at(concat([a, b]), [len(a)]) == [a, b]
    // =========================================================================
    #[test]
    fn mr_concat_split_identity(
        data1 in finite_vec(1, 100),
        data2 in finite_vec(1, 100)
    ) {
        let arr1 = from_vec_f64(data1);
        let arr2 = from_vec_f64(data2);
        let len1 = arr1.values().len();

        // Concatenate using static method
        let concat = UFuncArray::concatenate(&[&arr1, &arr2], 0).expect("concatenate");

        // Split at the boundary using split_at
        let splits = concat.split_at(&[len1], 0).expect("split_at");

        prop_assert_eq!(splits.len(), 2, "should produce 2 splits");

        prop_assert!(
            arrays_bitwise_eq(&arr1, &splits[0]),
            "first split should equal first array"
        );
        prop_assert!(
            arrays_bitwise_eq(&arr2, &splits[1]),
            "second split should equal second array"
        );
    }

    // =========================================================================
    // MR7: Astype round-trip for lossless dtypes
    // astype(astype(x, float32), float64) preserves values (within f32 precision)
    // =========================================================================
    #[test]
    fn mr_astype_f64_to_f32_roundtrip(data in prop::collection::vec(small_f64(), 1..100)) {
        let arr = from_vec_f64(data);

        // Convert to f32
        let as_f32 = arr.astype(DType::F32);

        // Convert back to f64
        let back_f64 = as_f32.astype(DType::F64);

        // Values should match within f32 precision
        let orig_vals = arr.values();
        let back_vals = back_f64.values();

        for (i, (&orig, &back)) in orig_vals.iter().zip(back_vals.iter()).enumerate() {
            let expected = orig as f32 as f64;
            prop_assert!(
                approx_eq_f64(back, expected, 1e-6),
                "round-trip at index {}: {} -> {} (expected {})",
                i,
                orig,
                back,
                expected
            );
        }
    }

    // =========================================================================
    // MR8: Astype to same dtype is identity
    // astype(x, dtype(x)) == x
    // =========================================================================
    #[test]
    fn mr_astype_same_dtype_identity(data in finite_vec(1, 100)) {
        let arr = from_vec_f64(data);
        let same = arr.astype(DType::F64);

        prop_assert!(
            arrays_bitwise_eq(&arr, &same),
            "astype to same dtype should be identity"
        );
    }

    // =========================================================================
    // MR9: Broadcasting equivalence: scalar op vs elementwise
    // x + scalar == x + broadcast(scalar, shape(x))
    // =========================================================================
    #[test]
    fn mr_broadcast_scalar_equivalence(
        data in finite_vec(1, 100),
        scalar in bounded_f64()
    ) {
        let arr = from_vec_f64(data);
        let scalar_arr = UFuncArray::new(vec![], vec![scalar], DType::F64).expect("scalar");

        // Scalar broadcast add
        let result1 = arr.elementwise_binary(&scalar_arr, BinaryOp::Add).expect("scalar add");

        // Manual elementwise add
        let broadcast_scalar = UFuncArray::new(
            arr.shape().to_vec(),
            vec![scalar; arr.values().len()],
            DType::F64,
        )
        .expect("broadcast scalar");
        let result2 = arr.elementwise_binary(&broadcast_scalar, BinaryOp::Add).expect("elem add");

        prop_assert!(
            arrays_approx_eq(&result1, &result2, EPSILON),
            "scalar broadcast should equal explicit elementwise"
        );
    }

    // =========================================================================
    // MR10: Add commutativity
    // a + b == b + a
    // =========================================================================
    #[test]
    fn mr_add_commutative(
        data1 in finite_vec(1, 100),
        data2 in finite_vec(1, 100)
    ) {
        // Make same length
        let len = data1.len().min(data2.len());
        let arr1 = from_vec_f64(data1[..len].to_vec());
        let arr2 = from_vec_f64(data2[..len].to_vec());

        let r1 = arr1.elementwise_binary(&arr2, BinaryOp::Add).expect("a+b");
        let r2 = arr2.elementwise_binary(&arr1, BinaryOp::Add).expect("b+a");

        prop_assert!(
            arrays_bitwise_eq(&r1, &r2),
            "addition should be commutative"
        );
    }

    // =========================================================================
    // MR11: Multiply commutativity
    // a * b == b * a
    // =========================================================================
    #[test]
    fn mr_multiply_commutative(
        data1 in finite_vec(1, 100),
        data2 in finite_vec(1, 100)
    ) {
        let len = data1.len().min(data2.len());
        let arr1 = from_vec_f64(data1[..len].to_vec());
        let arr2 = from_vec_f64(data2[..len].to_vec());

        let r1 = arr1.elementwise_binary(&arr2, BinaryOp::Mul).expect("a*b");
        let r2 = arr2.elementwise_binary(&arr1, BinaryOp::Mul).expect("b*a");

        prop_assert!(
            arrays_bitwise_eq(&r1, &r2),
            "multiplication should be commutative"
        );
    }

    // =========================================================================
    // MR12: Add associativity
    // (a + b) + c == a + (b + c) (within floating point tolerance)
    // =========================================================================
    #[test]
    fn mr_add_associative(
        data1 in prop::collection::vec(small_f64(), 1..50),
        data2 in prop::collection::vec(small_f64(), 1..50),
        data3 in prop::collection::vec(small_f64(), 1..50)
    ) {
        let len = data1.len().min(data2.len()).min(data3.len());
        let arr1 = from_vec_f64(data1[..len].to_vec());
        let arr2 = from_vec_f64(data2[..len].to_vec());
        let arr3 = from_vec_f64(data3[..len].to_vec());

        // (a + b) + c
        let ab = arr1.elementwise_binary(&arr2, BinaryOp::Add).expect("a+b");
        let abc1 = ab.elementwise_binary(&arr3, BinaryOp::Add).expect("(a+b)+c");

        // a + (b + c)
        let bc = arr2.elementwise_binary(&arr3, BinaryOp::Add).expect("b+c");
        let abc2 = arr1.elementwise_binary(&bc, BinaryOp::Add).expect("a+(b+c)");

        prop_assert!(
            arrays_approx_eq(&abc1, &abc2, EPSILON * 10.0),
            "addition should be associative within tolerance"
        );
    }

    // =========================================================================
    // MR13: Negation twice is identity
    // negative(negative(x)) == x
    // =========================================================================
    #[test]
    fn mr_negate_twice_identity(data in finite_vec(1, 100)) {
        let arr = from_vec_f64(data);

        let neg1 = arr.elementwise_unary(UnaryOp::Negative);
        let neg2 = neg1.elementwise_unary(UnaryOp::Negative);

        prop_assert!(
            arrays_bitwise_eq(&arr, &neg2),
            "double negation should be identity"
        );
    }

    // =========================================================================
    // MR14: Absolute value is idempotent
    // abs(abs(x)) == abs(x)
    // =========================================================================
    #[test]
    fn mr_abs_idempotent(data in finite_vec(1, 100)) {
        let arr = from_vec_f64(data);

        let abs1 = arr.elementwise_unary(UnaryOp::Abs);
        let abs2 = abs1.elementwise_unary(UnaryOp::Abs);

        prop_assert!(
            arrays_bitwise_eq(&abs1, &abs2),
            "abs should be idempotent"
        );
    }

    // =========================================================================
    // MR15: Sort is idempotent
    // sort(sort(x)) == sort(x)
    // =========================================================================
    #[test]
    fn mr_sort_idempotent(data in finite_vec(1, 200)) {
        let arr = from_vec_f64(data);

        let sorted1 = arr.sort(Some(-1), None).expect("sort1");
        let sorted2 = sorted1.sort(Some(-1), None).expect("sort2");

        prop_assert!(
            arrays_bitwise_eq(&sorted1, &sorted2),
            "sort should be idempotent"
        );
    }

    // =========================================================================
    // MR16: Reshape preserves total elements and sum
    // sum(reshape(x, new_shape)) == sum(x) when prod(new_shape) == len(x)
    // =========================================================================
    #[test]
    fn mr_reshape_preserves_sum(data in finite_vec(12, 12)) {
        // Use exactly 12 elements for easy reshape testing (2x6, 3x4, 4x3, 6x2)
        let arr = from_vec_f64(data);
        let sum_orig = arr.reduce_sum(None, false).expect("sum orig");

        // Reshape to 3x4
        let reshaped = arr.reshape(&[3, 4]).expect("reshape to 3x4");
        let sum_reshaped = reshaped.reduce_sum(None, false).expect("sum reshaped");

        prop_assert!(
            approx_eq_f64(sum_orig.values()[0], sum_reshaped.values()[0], EPSILON * 12.0),
            "reshape should preserve sum: {} vs {}",
            sum_orig.values()[0],
            sum_reshaped.values()[0]
        );
    }

    // =========================================================================
    // MR17: Squeeze then expand_dims is identity (for squeezable dims)
    // =========================================================================
    #[test]
    fn mr_squeeze_expand_identity(data in finite_vec(1, 50)) {
        let arr = from_vec_f64(data);

        // Add a dimension
        let expanded = arr.expand_dims(0).expect("expand");
        prop_assert_eq!(expanded.shape()[0], 1, "expanded should have dim 1 at axis 0");

        // Squeeze it back
        let squeezed = expanded.squeeze(Some(0)).expect("squeeze");

        prop_assert!(
            arrays_bitwise_eq(&arr, &squeezed),
            "squeeze(expand_dims(x)) should be identity"
        );
    }

    // =========================================================================
    // MR18: Tile then check replication
    // tile(x, [n]) produces n copies of x
    // =========================================================================
    #[test]
    fn mr_tile_replicates_values(data in finite_vec(1, 20), n in 2usize..5) {
        let arr = from_vec_f64(data);
        let original_len = arr.values().len();

        // Tile n times
        let tiled = arr.tile(&[n]).expect("tile");
        prop_assert_eq!(
            tiled.values().len(),
            original_len * n,
            "tiled length should be n * original"
        );

        // The tiled array should contain n copies of the original
        let tiled_vals = tiled.values();
        for i in 0..original_len {
            for j in 0..n {
                prop_assert!(
                    approx_eq_f64(arr.values()[i], tiled_vals[j * original_len + i], EPSILON),
                    "tile should replicate values"
                );
            }
        }
    }

    // =========================================================================
    // MR19: Min/max bounds
    // min(x) <= mean(x) <= max(x) (for non-empty arrays)
    // =========================================================================
    #[test]
    fn mr_min_mean_max_ordering(data in finite_vec(1, 100)) {
        let arr = from_vec_f64(data);

        let min_val = arr.reduce_min(None, false).expect("min").values()[0];
        let max_val = arr.reduce_max(None, false).expect("max").values()[0];
        let mean_val = arr.reduce_mean(None, false).expect("mean").values()[0];

        prop_assert!(
            min_val <= mean_val + EPSILON,
            "min should be <= mean: {} vs {}",
            min_val,
            mean_val
        );
        prop_assert!(
            mean_val <= max_val + EPSILON,
            "mean should be <= max: {} vs {}",
            mean_val,
            max_val
        );
    }

    // =========================================================================
    // MR20: Variance is non-negative
    // var(x) >= 0
    // =========================================================================
    #[test]
    fn mr_variance_nonnegative(data in finite_vec(2, 100)) {
        let arr = from_vec_f64(data);
        let var = arr.reduce_var(None, false, 0).expect("var");
        let var_val = var.values()[0];

        prop_assert!(
            var_val >= -EPSILON,
            "variance should be non-negative: {}",
            var_val
        );
    }

    // =========================================================================
    // MR21: Std is non-negative
    // std(x) >= 0
    // =========================================================================
    #[test]
    fn mr_std_nonnegative(data in finite_vec(2, 100)) {
        let arr = from_vec_f64(data);
        let std = arr.reduce_std(None, false, 0).expect("std");
        let std_val = std.values()[0];

        prop_assert!(
            std_val >= -EPSILON,
            "std should be non-negative: {}",
            std_val
        );
    }

    // =========================================================================
    // MR22: Var = Std^2
    // var(x) == std(x)^2
    // =========================================================================
    #[test]
    fn mr_var_equals_std_squared(data in prop::collection::vec(small_f64(), 2..100)) {
        let arr = from_vec_f64(data);
        let var = arr.reduce_var(None, false, 0).expect("var");
        let std = arr.reduce_std(None, false, 0).expect("std");

        let var_val = var.values()[0];
        let std_val = std.values()[0];

        // Use relative tolerance for larger values
        let tol = (var_val.abs() * 1e-10).max(EPSILON * 100.0);
        prop_assert!(
            approx_eq_f64(var_val, std_val * std_val, tol),
            "var should equal std^2: {} vs {}",
            var_val,
            std_val * std_val
        );
    }
}
