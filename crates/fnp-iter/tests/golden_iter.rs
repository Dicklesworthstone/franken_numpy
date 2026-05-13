//! Golden tests for fnp-iter deterministic functions.
//!
//! These tests verify that core iterator functions produce stable, expected outputs.

use fnp_iter::{
    FlatIterIndex, NditerTransferFlags, OverlapAction, TransferClass, TransferSelectorInput,
    ndenumerate, ndindex, overlap_copy_policy, resolve_flatiter_indices, select_transfer_class,
    validate_nditer_flags,
};

// ─────────────────────────────────────────────────────────────────────────────
// ndindex golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_ndindex_empty_shape() {
    let result = ndindex(&[]).unwrap();
    assert_eq!(result, vec![Vec::<usize>::new()]);
}

#[test]
fn golden_ndindex_scalar() {
    let result = ndindex(&[1]).unwrap();
    assert_eq!(result, vec![vec![0]]);
}

#[test]
fn golden_ndindex_1d() {
    let result = ndindex(&[4]).unwrap();
    assert_eq!(result, vec![vec![0], vec![1], vec![2], vec![3]]);
}

#[test]
fn golden_ndindex_2d() {
    let result = ndindex(&[2, 3]).unwrap();
    assert_eq!(
        result,
        vec![
            vec![0, 0],
            vec![0, 1],
            vec![0, 2],
            vec![1, 0],
            vec![1, 1],
            vec![1, 2],
        ]
    );
}

#[test]
fn golden_ndindex_3d() {
    let result = ndindex(&[2, 2, 2]).unwrap();
    assert_eq!(
        result,
        vec![
            vec![0, 0, 0],
            vec![0, 0, 1],
            vec![0, 1, 0],
            vec![0, 1, 1],
            vec![1, 0, 0],
            vec![1, 0, 1],
            vec![1, 1, 0],
            vec![1, 1, 1],
        ]
    );
}

#[test]
fn golden_ndindex_zero_dim() {
    let result = ndindex(&[0]).unwrap();
    assert!(result.is_empty());
}

#[test]
fn golden_ndindex_zero_inner() {
    let result = ndindex(&[2, 0, 3]).unwrap();
    assert!(result.is_empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// ndenumerate golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_ndenumerate_1d() {
    let result = ndenumerate(&[3], &[10.0, 20.0, 30.0]).unwrap();
    assert_eq!(
        result,
        vec![(vec![0], 10.0), (vec![1], 20.0), (vec![2], 30.0),]
    );
}

#[test]
fn golden_ndenumerate_2d() {
    let result = ndenumerate(&[2, 2], &[1, 2, 3, 4]).unwrap();
    assert_eq!(
        result,
        vec![
            (vec![0, 0], 1),
            (vec![0, 1], 2),
            (vec![1, 0], 3),
            (vec![1, 1], 4),
        ]
    );
}

#[test]
fn golden_ndenumerate_mismatch() {
    let result = ndenumerate(&[2, 2], &[1, 2, 3]);
    assert!(result.is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// select_transfer_class golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_transfer_class_contiguous() {
    let input = TransferSelectorInput {
        src_stride: 8,
        dst_stride: 8,
        item_size: 8,
        element_count: 100,
        aligned: true,
        cast_is_lossless: true,
        same_value_cast: false,
    };
    assert_eq!(
        select_transfer_class(input).unwrap(),
        TransferClass::Contiguous
    );
}

#[test]
fn golden_transfer_class_strided() {
    let input = TransferSelectorInput {
        src_stride: 16,
        dst_stride: 8,
        item_size: 8,
        element_count: 50,
        aligned: true,
        cast_is_lossless: true,
        same_value_cast: false,
    };
    assert_eq!(
        select_transfer_class(input).unwrap(),
        TransferClass::Strided
    );
}

#[test]
fn golden_transfer_class_strided_cast() {
    let input = TransferSelectorInput {
        src_stride: 8,
        dst_stride: 8,
        item_size: 8,
        element_count: 100,
        aligned: true,
        cast_is_lossless: false,
        same_value_cast: false,
    };
    assert_eq!(
        select_transfer_class(input).unwrap(),
        TransferClass::StridedCast
    );
}

#[test]
fn golden_transfer_class_zero_item() {
    let input = TransferSelectorInput {
        src_stride: 8,
        dst_stride: 8,
        item_size: 0,
        element_count: 100,
        aligned: true,
        cast_is_lossless: true,
        same_value_cast: false,
    };
    assert!(select_transfer_class(input).is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// overlap_copy_policy golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_overlap_disjoint_before() {
    assert_eq!(
        overlap_copy_policy(0, 100, 50).unwrap(),
        OverlapAction::NoCopy
    );
}

#[test]
fn golden_overlap_disjoint_after() {
    assert_eq!(
        overlap_copy_policy(100, 0, 50).unwrap(),
        OverlapAction::NoCopy
    );
}

#[test]
fn golden_overlap_forward() {
    assert_eq!(
        overlap_copy_policy(50, 25, 50).unwrap(),
        OverlapAction::ForwardCopy
    );
}

#[test]
fn golden_overlap_backward() {
    assert_eq!(
        overlap_copy_policy(25, 50, 50).unwrap(),
        OverlapAction::BackwardCopy
    );
}

#[test]
fn golden_overlap_exact_same() {
    assert_eq!(
        overlap_copy_policy(100, 100, 50).unwrap(),
        OverlapAction::ForwardCopy
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// validate_nditer_flags golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_flags_valid_defaults() {
    let flags = NditerTransferFlags {
        no_broadcast: false,
        observed_broadcast: false,
        copy_if_overlap: false,
        observed_overlap: false,
    };
    assert!(validate_nditer_flags(flags).is_ok());
}

#[test]
fn golden_flags_conflict() {
    let flags = NditerTransferFlags {
        no_broadcast: true,
        observed_broadcast: true,
        copy_if_overlap: false,
        observed_overlap: false,
    };
    assert!(validate_nditer_flags(flags).is_err());
}

#[test]
fn golden_flags_broadcast_ok() {
    let flags = NditerTransferFlags {
        no_broadcast: false,
        observed_broadcast: true,
        copy_if_overlap: true,
        observed_overlap: false,
    };
    assert!(validate_nditer_flags(flags).is_ok());
}

// ─────────────────────────────────────────────────────────────────────────────
// resolve_flatiter_indices golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_flatiter_single() {
    let result = resolve_flatiter_indices(10, &FlatIterIndex::Single(3)).unwrap();
    assert_eq!(result, vec![3]);
}

#[test]
fn golden_flatiter_slice() {
    let result = resolve_flatiter_indices(
        10,
        &FlatIterIndex::Slice {
            start: 2,
            stop: 6,
            step: 2,
        },
    )
    .unwrap();
    assert_eq!(result, vec![2, 4]);
}

#[test]
fn golden_flatiter_bool_mask() {
    let mask = vec![true, false, true, false, true];
    let result = resolve_flatiter_indices(5, &FlatIterIndex::BoolMask(mask)).unwrap();
    assert_eq!(result, vec![0, 2, 4]);
}

#[test]
fn golden_flatiter_fancy() {
    let indices = vec![1, 3, 5, 7];
    let result = resolve_flatiter_indices(10, &FlatIterIndex::Fancy(indices)).unwrap();
    assert_eq!(result, vec![1, 3, 5, 7]);
}

#[test]
fn golden_flatiter_out_of_bounds() {
    let result = resolve_flatiter_indices(5, &FlatIterIndex::Single(10));
    assert!(result.is_err());
}

#[test]
fn golden_flatiter_step_2() {
    let result = resolve_flatiter_indices(
        10,
        &FlatIterIndex::Slice {
            start: 0,
            stop: 8,
            step: 2,
        },
    )
    .unwrap();
    assert_eq!(result, vec![0, 2, 4, 6]);
}
