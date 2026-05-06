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

// ─────────────────────────────────────────────────────────────────────────────
// MR9: Construction invariant — NdLayout::contiguous is always contiguous
// Category: Equivalence
// Fault sensitivity: 5/5 — catches broken contiguity detection
// ─────────────────────────────────────────────────────────────────────────────

use fnp_ndarray::NdLayout;

#[test]
fn mr_layout_contiguous_is_contiguous_c() {
    let shapes: Vec<Vec<usize>> = vec![
        vec![],
        vec![1],
        vec![5],
        vec![3, 4],
        vec![2, 3, 4],
        vec![1, 2, 3, 4],
        vec![10, 20, 30],
    ];
    for shape in shapes {
        let layout = NdLayout::contiguous(shape.clone(), 8, MemoryOrder::C)
            .expect("contiguous construction");
        assert!(
            layout.is_contiguous(),
            "C-contiguous layout should report is_contiguous: {shape:?}"
        );
    }
}

#[test]
fn mr_layout_contiguous_is_contiguous_f() {
    let shapes: Vec<Vec<usize>> = vec![
        vec![],
        vec![1],
        vec![5],
        vec![3, 4],
        vec![2, 3, 4],
        vec![1, 2, 3, 4],
    ];
    for shape in shapes {
        let layout = NdLayout::contiguous(shape.clone(), 8, MemoryOrder::F)
            .expect("contiguous construction");
        assert!(
            layout.is_fortran_contiguous(),
            "F-contiguous layout should report is_fortran_contiguous: {shape:?}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MR10: Contiguous layout has no internal overlap
// Category: Equivalence
// Fault sensitivity: 4/5 — catches overlap detection bugs
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_layout_contiguous_no_overlap() {
    let shapes: Vec<Vec<usize>> = vec![
        vec![3, 4],
        vec![2, 3, 4],
        vec![5, 6, 7],
    ];
    for shape in shapes {
        for order in [MemoryOrder::C, MemoryOrder::F] {
            let layout = NdLayout::contiguous(shape.clone(), 8, order)
                .expect("contiguous");
            assert!(
                !layout.has_internal_overlap(),
                "contiguous layout should have no internal overlap: {shape:?} {order:?}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MR11: broadcast_to stride zeroing — broadcasted dims get stride 0
// Category: Equivalence
// Fault sensitivity: 5/5 — catches broadcast stride bugs
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_layout_broadcast_stride_zeroing() {
    let layout = NdLayout::contiguous(vec![1, 4], 8, MemoryOrder::C).expect("base");
    let broadcasted = layout.broadcast_to(vec![3, 4]).expect("broadcast");

    assert_eq!(
        broadcasted.strides[0], 0,
        "broadcasted dimension (1 -> 3) should have stride 0"
    );
    assert_ne!(
        broadcasted.strides[1], 0,
        "non-broadcasted dimension should retain stride"
    );
}

#[test]
fn mr_layout_broadcast_readonly() {
    let layout = NdLayout::contiguous(vec![1, 4], 8, MemoryOrder::C).expect("base");
    let broadcasted = layout.broadcast_to(vec![3, 4]).expect("broadcast");

    assert!(
        !broadcasted.is_writeable(),
        "broadcast view should be readonly"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// MR12: sliding_window_view output shape formula
// output_shape = [dim_i - window_i + 1 for each axis] ++ window_shape
// Category: Additive
// Fault sensitivity: 5/5 — catches sliding window shape bugs
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_layout_sliding_window_shape() {
    let cases: Vec<(Vec<usize>, Vec<usize>)> = vec![
        (vec![10], vec![3]),
        (vec![5, 6], vec![2, 3]),
        (vec![8, 9, 10], vec![2, 3, 4]),
    ];

    for (base_shape, window_shape) in cases {
        let layout = NdLayout::contiguous(base_shape.clone(), 8, MemoryOrder::C)
            .expect("base");
        let view = layout.sliding_window_view(&window_shape).expect("window");

        let expected_ndim = base_shape.len() * 2;
        assert_eq!(
            view.ndim(),
            expected_ndim,
            "sliding_window_view ndim should double: base={base_shape:?} window={window_shape:?}"
        );

        for (i, (&base_dim, &win_dim)) in base_shape.iter().zip(&window_shape).enumerate() {
            let expected = base_dim - win_dim + 1;
            assert_eq!(
                view.shape[i], expected,
                "axis {i} output dim: expected {expected}, got {}", view.shape[i]
            );
        }

        for (i, &win_dim) in window_shape.iter().enumerate() {
            let idx = base_shape.len() + i;
            assert_eq!(
                view.shape[idx], win_dim,
                "window axis {i} should match window_shape"
            );
        }
    }
}

#[test]
fn mr_layout_sliding_window_readonly() {
    let layout = NdLayout::contiguous(vec![10, 10], 8, MemoryOrder::C).expect("base");
    let view = layout.sliding_window_view(&[3, 3]).expect("window");

    assert!(
        !view.is_writeable(),
        "sliding_window_view should be readonly"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// MR13: as_strided preserves item_size
// Category: Equivalence
// Fault sensitivity: 3/5 — catches item_size propagation bugs
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_layout_as_strided_preserves_item_size() {
    let item_sizes: Vec<usize> = vec![1, 2, 4, 8, 16];

    for item_size in item_sizes {
        let layout = NdLayout::contiguous(vec![10, 10], item_size, MemoryOrder::C)
            .expect("base");
        let view = layout
            .as_strided(vec![5, 5], vec![item_size as isize * 2, item_size as isize])
            .expect("strided");

        assert_eq!(
            view.item_size, item_size,
            "as_strided should preserve item_size"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MR14: 1D layout is both C and F contiguous
// Category: Equivalence
// Fault sensitivity: 4/5 — catches edge case in contiguity checks
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_layout_1d_both_contiguous() {
    let layout_c = NdLayout::contiguous(vec![10], 8, MemoryOrder::C).expect("C");
    let layout_f = NdLayout::contiguous(vec![10], 8, MemoryOrder::F).expect("F");

    assert!(layout_c.is_contiguous(), "1D C layout is_contiguous");
    assert!(layout_c.is_fortran_contiguous(), "1D C layout is_fortran_contiguous");
    assert!(layout_f.is_contiguous(), "1D F layout is_contiguous");
    assert!(layout_f.is_fortran_contiguous(), "1D F layout is_fortran_contiguous");
}

#[test]
fn mr_layout_scalar_both_contiguous() {
    let layout = NdLayout::contiguous(vec![], 8, MemoryOrder::C).expect("scalar");

    assert!(layout.is_contiguous(), "scalar is_contiguous");
    assert!(layout.is_fortran_contiguous(), "scalar is_fortran_contiguous");
}

// ─────────────────────────────────────────────────────────────────────────────
// MR15: broadcast_strides consistency with broadcast_shape
// If broadcast_shape(src, dst) succeeds, broadcast_strides should produce
// valid strides with zeros for broadcasted dimensions.
// Category: Equivalence
// Fault sensitivity: 5/5 — catches stride/shape desync
// ─────────────────────────────────────────────────────────────────────────────

use fnp_ndarray::broadcast_strides;

#[test]
fn mr_broadcast_strides_zeros_for_expanded_dims() {
    let src_shape = vec![1, 4];
    let src_strides = vec![32isize, 8isize];
    let dst_shape = vec![3, 4];

    let result = broadcast_strides(&src_shape, &src_strides, &dst_shape).expect("broadcast");

    assert_eq!(result[0], 0, "expanded dim 0 (1→3) must have stride 0");
    assert_eq!(result[1], 8, "non-expanded dim 1 keeps original stride");
}

#[test]
fn mr_broadcast_strides_rank_extension() {
    let src_shape = vec![4];
    let src_strides = vec![8isize];
    let dst_shape = vec![3, 4];

    let result = broadcast_strides(&src_shape, &src_strides, &dst_shape).expect("broadcast");

    assert_eq!(result.len(), 2, "output strides match dst rank");
    assert_eq!(result[0], 0, "prepended dim has stride 0");
    assert_eq!(result[1], 8, "original dim keeps stride");
}

// ─────────────────────────────────────────────────────────────────────────────
// MR16: as_strided out-of-bounds rejection
// A strided view that would read beyond the base storage must be rejected.
// Category: Inclusive (boundary)
// Fault sensitivity: 5/5 — catches buffer overread bugs
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_as_strided_rejects_oob() {
    let layout = NdLayout::contiguous(vec![10], 8, MemoryOrder::C).expect("base");

    let result = layout.as_strided(vec![11], vec![8]);
    assert!(
        result.is_err(),
        "as_strided should reject view requiring 11 elements from 10-element base"
    );
}

#[test]
fn mr_as_strided_accepts_exact_fit() {
    let layout = NdLayout::contiguous(vec![10], 8, MemoryOrder::C).expect("base");

    let result = layout.as_strided(vec![10], vec![8]);
    assert!(
        result.is_ok(),
        "as_strided should accept view that exactly fits"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// MR17: sliding_window rejects oversized window
// Category: Inclusive (boundary)
// Fault sensitivity: 4/5 — catches window validation bugs
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_sliding_window_rejects_oversized() {
    let layout = NdLayout::contiguous(vec![5, 5], 8, MemoryOrder::C).expect("base");

    let result = layout.sliding_window_view(&[6, 3]);
    assert!(
        result.is_err(),
        "sliding_window should reject window larger than base dim"
    );
}

#[test]
fn mr_sliding_window_accepts_exact_fit() {
    let layout = NdLayout::contiguous(vec![5, 5], 8, MemoryOrder::C).expect("base");

    let view = layout.sliding_window_view(&[5, 5]).expect("exact fit");
    assert_eq!(view.shape, vec![1, 1, 5, 5], "exact fit produces 1x1 output grid");
}

// ─────────────────────────────────────────────────────────────────────────────
// MR18: Negative strides preserve element accessibility
// If as_strided uses negative strides, it should still correctly compute
// required_nbytes and overlap detection.
// Category: Equivalence
// Fault sensitivity: 5/5 — catches signed arithmetic bugs
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_negative_stride_overlap_detection() {
    let layout = NdLayout::contiguous(vec![10], 8, MemoryOrder::C).expect("base");

    let view = layout.as_strided(vec![5], vec![-8]).expect("negative stride");

    assert!(
        !view.has_internal_overlap(),
        "negative stride with item_size gap should not overlap"
    );
}

#[test]
fn mr_negative_stride_small_gap_overlaps() {
    let layout = NdLayout::contiguous(vec![20], 8, MemoryOrder::C).expect("base");

    let view = layout.as_strided(vec![5], vec![-4]).expect("small negative stride");

    assert!(
        view.has_internal_overlap(),
        "stride magnitude < item_size should indicate overlap"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// MR19: Zero-stride dimension implies overlap when dim > 1
// Category: Equivalence
// Fault sensitivity: 5/5 — catches broadcast overlap detection
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_zero_stride_overlap() {
    let layout = NdLayout::contiguous(vec![10], 8, MemoryOrder::C).expect("base");

    let view = layout.as_strided(vec![5, 2], vec![8, 0]).expect("zero stride in dim 1");

    assert!(
        view.has_internal_overlap(),
        "zero stride with dim > 1 should indicate overlap"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// MR20: NumPy oracle conformance for broadcast_shape
// Verify fnp-ndarray matches numpy.broadcast_shapes exactly
// Category: Differential (oracle-based)
// Fault sensitivity: 5/5 — catches any divergence from NumPy
// ─────────────────────────────────────────────────────────────────────────────

fn numpy_broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    use std::process::Command;
    let script = format!(
        "import numpy as np; print(list(np.broadcast_shapes({:?}, {:?})))",
        a, b
    );
    let output = Command::new("python3")
        .args(["-c", &script])
        .output()
        .map_err(|e| format!("python3 failed: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("could not be broadcast") {
            return Err("incompatible".to_string());
        }
        return Err(format!("numpy error: {stderr}"));
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let parsed: Vec<usize> = stdout
        .trim_matches(|c| c == '[' || c == ']')
        .split(", ")
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    Ok(parsed)
}

#[test]
fn mr_numpy_conformance_broadcast_compatible() {
    let cases: Vec<(&[usize], &[usize])> = vec![
        (&[3, 4], &[4]),
        (&[1, 5], &[3, 1]),
        (&[2, 1, 4], &[3, 4]),
        (&[1], &[5, 6, 7]),
        (&[], &[3, 4]),
        (&[1, 1], &[2, 3]),
    ];

    for (a, b) in cases {
        let fnp = broadcast_shape(a, b);
        let np = numpy_broadcast(a, b);

        match (fnp, np) {
            (Ok(fnp_result), Ok(np_result)) => {
                assert_eq!(
                    fnp_result, np_result,
                    "broadcast divergence: fnp({a:?}, {b:?}) = {fnp_result:?} vs numpy = {np_result:?}"
                );
            }
            (Err(_), Err(_)) => {}
            (fnp_r, np_r) => {
                panic!("broadcast agreement mismatch: fnp={fnp_r:?} vs numpy={np_r:?} for ({a:?}, {b:?})");
            }
        }
    }
}

#[test]
fn mr_numpy_conformance_broadcast_incompatible() {
    let cases: Vec<(&[usize], &[usize])> = vec![
        (&[3, 4], &[5, 4]),
        (&[2, 3], &[4, 3]),
        (&[2, 3, 4], &[5, 3, 4]),
    ];

    for (a, b) in cases {
        let fnp = broadcast_shape(a, b);
        let np = numpy_broadcast(a, b);

        assert!(
            fnp.is_err() && np.is_err(),
            "both should reject: fnp({a:?}, {b:?}) = {fnp:?}, numpy = {np:?}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MR21: NumPy oracle conformance for fix_unknown_dimension (reshape -1)
// Category: Differential (oracle-based)
// Fault sensitivity: 5/5 — catches reshape edge case divergence
// ─────────────────────────────────────────────────────────────────────────────

fn numpy_reshape(count: usize, new_shape: &[isize]) -> Result<Vec<usize>, String> {
    use std::process::Command;
    let script = format!(
        "import numpy as np; a = np.zeros({}); print(list(a.reshape({:?}).shape))",
        count, new_shape
    );
    let output = Command::new("python3")
        .args(["-c", &script])
        .output()
        .map_err(|e| format!("python3 failed: {e}"))?;
    if !output.status.success() {
        return Err("reshape failed".to_string());
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let parsed: Vec<usize> = stdout
        .trim_matches(|c| c == '[' || c == ']')
        .split(", ")
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    Ok(parsed)
}

#[test]
fn mr_numpy_conformance_reshape_with_unknown() {
    let cases: Vec<(usize, &[isize])> = vec![
        (24, &[2, -1, 4]),
        (24, &[-1, 6]),
        (12, &[3, -1]),
        (1, &[-1]),
        (6, &[2, 3, -1]),
        (6, &[-1, 2, 3]),
    ];

    for (count, new_shape) in cases {
        let fnp = fix_unknown_dimension(new_shape, count);
        let np = numpy_reshape(count, new_shape);

        match (fnp, np) {
            (Ok(fnp_result), Ok(np_result)) => {
                assert_eq!(
                    fnp_result, np_result,
                    "reshape divergence: fnp({count}, {new_shape:?}) = {fnp_result:?} vs numpy = {np_result:?}"
                );
            }
            (Err(_), Err(_)) => {}
            (fnp_r, np_r) => {
                panic!("reshape agreement mismatch: fnp={fnp_r:?} vs numpy={np_r:?} for ({count}, {new_shape:?})");
            }
        }
    }
}

#[test]
fn mr_numpy_conformance_reshape_zero_size() {
    let fnp = fix_unknown_dimension(&[-1], 0);
    let np = numpy_reshape(0, &[-1]);

    assert!(
        fnp.is_ok() && np.is_ok(),
        "reshape(0, (-1,)) should succeed: fnp={fnp:?}, numpy={np:?}"
    );
    assert_eq!(
        fnp.unwrap(),
        np.unwrap(),
        "reshape(0, (-1,)) result mismatch"
    );
}

#[test]
fn mr_numpy_conformance_reshape_zero_ambiguous() {
    let fnp = fix_unknown_dimension(&[0, -1], 0);
    let np = numpy_reshape(0, &[0, -1]);

    assert!(
        fnp.is_err() && np.is_err(),
        "reshape(0, (0, -1)) should fail: fnp={fnp:?}, numpy={np:?}"
    );
}
