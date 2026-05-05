//! Metamorphic tests for fnp-ndarray shape operations.
//!
//! These tests verify mathematical properties that MUST hold for correct
//! implementations, without requiring an oracle for expected outputs.

use fnp_ndarray::{
    broadcast_shape, broadcast_shapes, contiguous_strides, element_count, fix_unknown_dimension,
    MemoryOrder,
};

// ─────────────────────────────────────────────────────────────────────────────
// MR1: Commutativity — broadcast_shape(a, b) == broadcast_shape(b, a)
// Category: Equivalence
// Fault sensitivity: 4/5 — catches order-dependent bugs in dimension merging
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_broadcast_commutativity_same_rank() {
    let cases: Vec<(&[usize], &[usize])> = vec![
        (&[3, 4], &[3, 1]),
        (&[1, 5], &[3, 5]),
        (&[2, 3, 4], &[1, 3, 4]),
        (&[5, 1, 7], &[5, 6, 1]),
    ];
    for (a, b) in cases {
        let ab = broadcast_shape(a, b);
        let ba = broadcast_shape(b, a);
        assert_eq!(ab, ba, "commutativity violated: broadcast({a:?}, {b:?}) != broadcast({b:?}, {a:?})");
    }
}

#[test]
fn mr_broadcast_commutativity_different_rank() {
    let cases: Vec<(&[usize], &[usize])> = vec![
        (&[3, 4], &[4]),
        (&[5], &[2, 3, 5]),
        (&[1], &[7, 8, 9]),
        (&[2, 1], &[3, 2, 5]),
    ];
    for (a, b) in cases {
        let ab = broadcast_shape(a, b);
        let ba = broadcast_shape(b, a);
        assert_eq!(ab, ba, "commutativity violated for different ranks");
    }
}

#[test]
fn mr_broadcast_commutativity_incompatible() {
    let cases: Vec<(&[usize], &[usize])> = vec![
        (&[3, 4], &[5, 4]),
        (&[2, 3], &[4, 3]),
    ];
    for (a, b) in cases {
        let ab = broadcast_shape(a, b);
        let ba = broadcast_shape(b, a);
        assert!(ab.is_err() && ba.is_err(), "commutativity: both should fail for {a:?}, {b:?}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MR2: Idempotency — broadcast_shape(a, a) == a
// Category: Equivalence
// Fault sensitivity: 3/5 — catches self-broadcast bugs
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_broadcast_idempotency() {
    let shapes: Vec<&[usize]> = vec![
        &[],
        &[1],
        &[5],
        &[3, 4],
        &[2, 3, 4],
        &[1, 1, 1, 1],
        &[7, 8, 9, 10],
    ];
    for shape in shapes {
        let result = broadcast_shape(shape, shape).expect("self-broadcast should succeed");
        assert_eq!(result.as_slice(), shape, "idempotency violated: broadcast({shape:?}, {shape:?}) != {shape:?}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MR3: Identity — broadcast_shape(a, [1]) preserves trailing dims
// Category: Additive (identity element)
// Fault sensitivity: 3/5 — catches scalar broadcast bugs
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_broadcast_scalar_identity() {
    let shapes: Vec<&[usize]> = vec![
        &[5],
        &[3, 4],
        &[2, 3, 4],
        &[1, 2, 3, 4],
    ];
    let scalar: &[usize] = &[1];
    for shape in shapes {
        let result = broadcast_shape(shape, scalar).expect("scalar broadcast should succeed");
        assert_eq!(result.as_slice(), shape, "scalar identity violated for {shape:?}");
    }
}

#[test]
fn mr_broadcast_empty_identity() {
    let shapes: Vec<&[usize]> = vec![
        &[],
        &[1],
        &[5],
        &[3, 4],
    ];
    let empty: &[usize] = &[];
    for shape in shapes {
        let result = broadcast_shape(shape, empty).expect("empty broadcast should succeed");
        assert_eq!(result.as_slice(), shape, "empty identity violated for {shape:?}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MR4: Associativity — broadcast_shapes order independence
// Category: Permutative
// Fault sensitivity: 5/5 — catches fold-order bugs
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_broadcast_shapes_associativity() {
    let a: &[usize] = &[3, 1];
    let b: &[usize] = &[1, 4];
    let c: &[usize] = &[3, 4];

    let abc = broadcast_shapes(&[a, b, c]).expect("abc");
    let acb = broadcast_shapes(&[a, c, b]).expect("acb");
    let bac = broadcast_shapes(&[b, a, c]).expect("bac");
    let bca = broadcast_shapes(&[b, c, a]).expect("bca");
    let cab = broadcast_shapes(&[c, a, b]).expect("cab");
    let cba = broadcast_shapes(&[c, b, a]).expect("cba");

    assert_eq!(abc, acb, "permutation abc != acb");
    assert_eq!(abc, bac, "permutation abc != bac");
    assert_eq!(abc, bca, "permutation abc != bca");
    assert_eq!(abc, cab, "permutation abc != cab");
    assert_eq!(abc, cba, "permutation abc != cba");
}

#[test]
fn mr_broadcast_shapes_pairwise_equals_bulk() {
    let a: &[usize] = &[2, 1, 4];
    let b: &[usize] = &[1, 3, 1];
    let c: &[usize] = &[2, 1, 1];

    let bulk = broadcast_shapes(&[a, b, c]).expect("bulk");
    let pairwise = {
        let ab = broadcast_shape(a, b).expect("ab");
        broadcast_shape(&ab, c).expect("ab_c")
    };
    assert_eq!(bulk, pairwise, "bulk broadcast != sequential pairwise");
}

// ─────────────────────────────────────────────────────────────────────────────
// MR5: Element count preservation — reshape preserves product
// Category: Invertive (round-trip)
// Fault sensitivity: 5/5 — catches element count miscalculation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_fix_unknown_preserves_element_count() {
    let cases: Vec<(usize, &[isize])> = vec![
        (24, &[-1, 4]),
        (24, &[2, -1, 4]),
        (24, &[2, 3, -1]),
        (60, &[-1, 3, 4]),
        (60, &[5, -1, 4]),
        (60, &[5, 3, -1]),
        (120, &[2, -1, 3, 4]),
    ];
    for (old_count, new_shape) in cases {
        let resolved = fix_unknown_dimension(new_shape, old_count)
            .expect(&format!("fix_unknown({new_shape:?}, {old_count}) should succeed"));
        let new_count = element_count(&resolved).expect("element_count");
        assert_eq!(
            old_count, new_count,
            "element count changed: {old_count} -> {new_count} for shape {new_shape:?}"
        );
    }
}

#[test]
fn mr_element_count_multiplicative() {
    let shapes: Vec<&[usize]> = vec![
        &[2, 3],
        &[4, 5, 6],
        &[1, 2, 3, 4],
        &[7, 8],
    ];
    for shape in shapes {
        let count = element_count(shape).expect("element_count");
        let manual: usize = shape.iter().product();
        assert_eq!(count, manual, "element_count({shape:?}) != manual product");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MR6: C/F stride transpose duality
// C_strides(shape) == reverse(F_strides(reverse(shape)))
// Category: Permutative
// Fault sensitivity: 4/5 — catches memory layout bugs
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_stride_cf_transpose_duality() {
    let shapes: Vec<&[usize]> = vec![
        &[3, 4],
        &[2, 3, 4],
        &[5, 6, 7, 8],
    ];
    let item_size = 8usize;

    for shape in shapes {
        let c_strides = contiguous_strides(shape, item_size, MemoryOrder::C).expect("C");

        let mut rev_shape: Vec<usize> = shape.to_vec();
        rev_shape.reverse();
        let f_strides_rev_shape = contiguous_strides(&rev_shape, item_size, MemoryOrder::F).expect("F");

        let mut f_rev = f_strides_rev_shape.clone();
        f_rev.reverse();

        assert_eq!(
            c_strides, f_rev,
            "C/F transpose duality violated for {shape:?}: C={c_strides:?}, F(rev)_rev={f_rev:?}"
        );
    }
}

#[test]
fn mr_stride_memory_span_consistency() {
    let shapes: Vec<&[usize]> = vec![
        &[3, 4],
        &[2, 3, 4],
        &[5, 6, 7],
    ];
    let item_size = 8usize;

    for shape in shapes {
        let elem_count = element_count(shape).expect("elem");
        let expected_span = elem_count * item_size;

        for order in [MemoryOrder::C, MemoryOrder::F] {
            let strides = contiguous_strides(shape, item_size, order).expect("strides");
            let max_offset: usize = shape
                .iter()
                .zip(strides.iter())
                .map(|(&dim, &stride)| {
                    if dim > 0 {
                        (dim - 1) * (stride as usize)
                    } else {
                        0
                    }
                })
                .sum();
            let actual_span = max_offset + item_size;
            assert_eq!(
                actual_span, expected_span,
                "memory span mismatch for {shape:?} {order:?}: {actual_span} != {expected_span}"
            );
        }
    }
}

#[test]
fn mr_stride_first_element_offset_zero() {
    let shapes: Vec<&[usize]> = vec![
        &[3, 4],
        &[2, 3, 4],
        &[1, 1, 1],
    ];
    for shape in shapes {
        for order in [MemoryOrder::C, MemoryOrder::F] {
            let strides = contiguous_strides(shape, 8, order).expect("strides");
            let offset: isize = shape
                .iter()
                .zip(strides.iter())
                .map(|(&dim, &stride)| if dim > 0 { 0 } else { 0 } * stride)
                .sum();
            assert_eq!(offset, 0, "first element offset != 0 for {shape:?} {order:?}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MR7: Broadcast rank monotonicity — result.len() >= max(a.len(), b.len())
// Category: Inclusive
// Fault sensitivity: 3/5 — catches dimension dropping bugs
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_broadcast_rank_monotonic() {
    let cases: Vec<(&[usize], &[usize])> = vec![
        (&[3], &[4, 3]),
        (&[1, 2], &[5, 1, 2]),
        (&[2, 3, 4], &[4]),
        (&[1], &[2, 3, 4, 5]),
    ];
    for (a, b) in cases {
        let result = broadcast_shape(a, b).expect("broadcast");
        let expected_rank = a.len().max(b.len());
        assert!(
            result.len() >= expected_rank,
            "rank monotonicity violated: broadcast({a:?}, {b:?}) = {result:?}, rank {} < max({})",
            result.len(),
            expected_rank
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MR8: Zero dimension propagation
// Category: Multiplicative (annihilator)
// Fault sensitivity: 4/5 — catches zero-size array edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_zero_dimension_element_count() {
    let shapes: Vec<&[usize]> = vec![
        &[0],
        &[0, 5],
        &[3, 0, 4],
        &[2, 3, 0, 4],
    ];
    for shape in shapes {
        let count = element_count(shape).expect("element_count");
        assert_eq!(count, 0, "zero dimension should yield zero elements: {shape:?}");
    }
}

#[test]
fn mr_zero_dimension_broadcast() {
    let a: &[usize] = &[0, 3];
    let b: &[usize] = &[1, 3];
    let result = broadcast_shape(a, b).expect("broadcast with zero");
    assert!(
        result.contains(&0),
        "broadcasting zero-dim should preserve zero: {a:?} + {b:?} = {result:?}"
    );
}

#[test]
fn mr_zero_strides_for_zero_shape() {
    let shape: &[usize] = &[0, 3, 4];
    let strides = contiguous_strides(shape, 8, MemoryOrder::C).expect("strides");
    assert!(
        strides.iter().all(|&s| s == 0),
        "zero-shape should have all-zero strides: {strides:?}"
    );
}
