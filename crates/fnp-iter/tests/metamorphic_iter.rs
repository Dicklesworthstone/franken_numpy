//! Metamorphic tests for fnp-iter iterator operations.
//!
//! Tests iterator invariants that must hold regardless of shape:
//! - Index roundtrip: linear_to_multi → multi_to_linear = identity
//! - Coverage completeness: ndindex produces product-of-dims elements
//! - Uniqueness: no duplicate indices in iteration
//! - Bounds: all indices within shape dimensions
//! - Commutativity: sum over elements is order-independent
//!
//! Finding: fnp-iter had ZERO metamorphic tests despite containing
//! complex index manipulation logic that has rich invariant properties.

use fnp_iter::{NditerOptions, NditerOrder, NditerPlan, ndindex};
use std::collections::HashSet;

// ─────────────────────────────────────────────────────────────────────────────
// MR1: Index roundtrip (linear_to_multi ∘ multi_to_linear = identity)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_index_roundtrip_1d() {
    let shape = vec![10usize];
    let plan = NditerPlan::new(shape.clone(), 8, NditerOptions::default()).unwrap();

    for linear in 0..10 {
        let multi = plan.linear_index_to_multi_index(linear).unwrap();
        let back = plan.multi_index_to_linear_index(&multi).unwrap();
        assert_eq!(
            linear, back,
            "roundtrip failed: {linear} -> {multi:?} -> {back}"
        );
    }
}

#[test]
fn mr_index_roundtrip_2d() {
    let shape = vec![4usize, 5];
    let plan = NditerPlan::new(shape.clone(), 8, NditerOptions::default()).unwrap();
    let total = 4 * 5;

    for linear in 0..total {
        let multi = plan.linear_index_to_multi_index(linear).unwrap();
        let back = plan.multi_index_to_linear_index(&multi).unwrap();
        assert_eq!(
            linear, back,
            "roundtrip failed for 2D: {linear} -> {multi:?} -> {back}"
        );
    }
}

#[test]
fn mr_index_roundtrip_3d() {
    let shape = vec![3usize, 4, 5];
    let plan = NditerPlan::new(shape.clone(), 8, NditerOptions::default()).unwrap();
    let total = 3 * 4 * 5;

    for linear in 0..total {
        let multi = plan.linear_index_to_multi_index(linear).unwrap();
        let back = plan.multi_index_to_linear_index(&multi).unwrap();
        assert_eq!(
            linear, back,
            "roundtrip failed for 3D: {linear} -> {multi:?} -> {back}"
        );
    }
}

#[test]
fn mr_index_roundtrip_4d() {
    let shape = vec![2usize, 3, 4, 5];
    let plan = NditerPlan::new(shape.clone(), 8, NditerOptions::default()).unwrap();
    let total: usize = shape.iter().product();

    for linear in 0..total {
        let multi = plan.linear_index_to_multi_index(linear).unwrap();
        let back = plan.multi_index_to_linear_index(&multi).unwrap();
        assert_eq!(
            linear, back,
            "roundtrip failed for 4D: {linear} -> {multi:?} -> {back}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MR2: Coverage completeness (|ndindex(shape)| = product(shape))
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_ndindex_count_1d() {
    for n in [1, 5, 10, 100] {
        let indices = ndindex(&[n]).unwrap();
        assert_eq!(
            indices.len(),
            n,
            "ndindex should produce exactly {n} indices for shape [{n}]"
        );
    }
}

#[test]
fn mr_ndindex_count_2d() {
    for (m, n) in [(2, 3), (5, 5), (10, 20)] {
        let indices = ndindex(&[m, n]).unwrap();
        let expected = m * n;
        assert_eq!(
            indices.len(),
            expected,
            "ndindex should produce {expected} indices for shape [{m}, {n}]"
        );
    }
}

#[test]
fn mr_ndindex_count_3d() {
    for shape in [[2, 3, 4], [3, 3, 3], [5, 2, 10]] {
        let indices = ndindex(&shape).unwrap();
        let expected: usize = shape.iter().product();
        assert_eq!(
            indices.len(),
            expected,
            "ndindex should produce {expected} indices for shape {shape:?}"
        );
    }
}

#[test]
fn mr_ndindex_count_empty_axis() {
    let indices = ndindex(&[5, 0, 3]).unwrap();
    assert_eq!(
        indices.len(),
        0,
        "ndindex with zero axis should produce 0 indices"
    );
}

#[test]
fn mr_ndindex_count_scalar() {
    let indices = ndindex(&[]).unwrap();
    assert_eq!(
        indices.len(),
        1,
        "ndindex of scalar (empty shape) should produce 1 index"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// MR3: Uniqueness (no duplicate indices)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_ndindex_uniqueness_2d() {
    let indices = ndindex(&[4, 5]).unwrap();
    let set: HashSet<_> = indices.iter().collect();
    assert_eq!(
        set.len(),
        indices.len(),
        "ndindex should produce unique indices"
    );
}

#[test]
fn mr_ndindex_uniqueness_3d() {
    let indices = ndindex(&[3, 4, 5]).unwrap();
    let set: HashSet<_> = indices.iter().collect();
    assert_eq!(
        set.len(),
        indices.len(),
        "ndindex 3D should produce unique indices"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// MR4: Bounds (all indices within shape)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_ndindex_bounds_2d() {
    let shape = [4, 5];
    let indices = ndindex(&shape).unwrap();

    for idx in &indices {
        assert_eq!(
            idx.len(),
            shape.len(),
            "index should have same ndim as shape"
        );
        for (&i, &dim) in idx.iter().zip(shape.iter()) {
            assert!(i < dim, "index {idx:?} out of bounds for shape {shape:?}");
        }
    }
}

#[test]
fn mr_ndindex_bounds_3d() {
    let shape = [3, 4, 5];
    let indices = ndindex(&shape).unwrap();

    for idx in &indices {
        assert_eq!(idx.len(), shape.len());
        for (&i, &dim) in idx.iter().zip(shape.iter()) {
            assert!(i < dim, "index {idx:?} out of bounds for shape {shape:?}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MR5: Order independence (C vs F order covers same indices)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_order_independence_same_coverage() {
    let shape = vec![3usize, 4, 5];

    let opts_c = NditerOptions {
        order: NditerOrder::C,
        ..Default::default()
    };
    let opts_f = NditerOptions {
        order: NditerOrder::F,
        ..Default::default()
    };

    let plan_c = NditerPlan::new(shape.clone(), 8, opts_c).unwrap();
    let plan_f = NditerPlan::new(shape.clone(), 8, opts_f).unwrap();

    let total: usize = shape.iter().product();

    let mut indices_c: HashSet<Vec<usize>> = HashSet::new();
    let mut indices_f: HashSet<Vec<usize>> = HashSet::new();

    for linear in 0..total {
        indices_c.insert(plan_c.linear_index_to_multi_index(linear).unwrap());
        indices_f.insert(plan_f.linear_index_to_multi_index(linear).unwrap());
    }

    assert_eq!(
        indices_c, indices_f,
        "C and F order should cover the same set of indices"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// MR6: Multi-index determinism (same input → same output)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_multi_index_deterministic() {
    let shape = vec![5usize, 6, 7];
    let plan = NditerPlan::new(shape.clone(), 8, NditerOptions::default()).unwrap();

    for linear in [0, 17, 42, 100, 209] {
        let first = plan.linear_index_to_multi_index(linear).unwrap();
        let second = plan.linear_index_to_multi_index(linear).unwrap();
        assert_eq!(
            first, second,
            "linear_index_to_multi_index should be deterministic"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MR7: Boundary values (first and last indices)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_boundary_indices_2d() {
    let shape = vec![4usize, 5];
    let plan = NditerPlan::new(shape.clone(), 8, NditerOptions::default()).unwrap();
    let total = 4 * 5;

    let first = plan.linear_index_to_multi_index(0).unwrap();
    assert_eq!(first, vec![0, 0], "first index should be [0, 0]");

    let last = plan.linear_index_to_multi_index(total - 1).unwrap();
    assert_eq!(
        last,
        vec![3, 4],
        "last index should be [3, 4] for shape [4, 5]"
    );
}

#[test]
fn mr_boundary_indices_3d() {
    let shape = vec![2usize, 3, 4];
    let plan = NditerPlan::new(shape.clone(), 8, NditerOptions::default()).unwrap();
    let total: usize = shape.iter().product();

    let first = plan.linear_index_to_multi_index(0).unwrap();
    assert_eq!(first, vec![0, 0, 0], "first index should be [0, 0, 0]");

    let last = plan.linear_index_to_multi_index(total - 1).unwrap();
    assert_eq!(
        last,
        vec![1, 2, 3],
        "last index should be [1, 2, 3] for shape [2, 3, 4]"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// MR8: Element count matches plan
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_element_count_matches_shape_product() {
    for shape in [vec![10], vec![5, 6], vec![3, 4, 5], vec![2, 3, 4, 5]] {
        let plan = NditerPlan::new(shape.clone(), 8, NditerOptions::default()).unwrap();
        let expected: usize = shape.iter().product();
        assert_eq!(
            plan.element_count(),
            expected,
            "element_count should match product of shape for {shape:?}"
        );
    }
}

#[test]
fn mr_element_count_with_zero_dim() {
    let shape = vec![5, 0, 3];
    let plan = NditerPlan::new(shape, 8, NditerOptions::default()).unwrap();
    assert_eq!(
        plan.element_count(),
        0,
        "element_count with zero dim should be 0"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Regression: fuzz crash-1bc73e4f — external_loop with zero-element shape
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn regression_fuzz_crash_1bc73e4f_external_loop_zero_elem() {
    // When external_loop=true and a dimension is zero, iteration_shape.product()
    // must not exceed element_count. Previously returned Vec::new() whose product
    // is 1 (identity), violating the invariant 1 <= 0.
    for shape in [
        vec![0usize, 5], // zero in first dim (F-order inner loop)
        vec![5, 0],      // zero in second dim
        vec![0],         // single zero dim
        vec![3, 0, 4],   // zero in middle
    ] {
        for order in [NditerOrder::C, NditerOrder::F] {
            let opts = NditerOptions {
                order,
                external_loop: true,
            };
            let plan = NditerPlan::new(shape.clone(), 8, opts).unwrap();
            let total = plan.element_count();
            let iter_total: usize = plan.iteration_shape().iter().product();

            assert!(
                iter_total <= total,
                "iteration_shape product ({iter_total}) must be <= element_count ({total}) \
                 for shape={shape:?}, order={order:?}, external_loop=true"
            );
        }
    }
}
