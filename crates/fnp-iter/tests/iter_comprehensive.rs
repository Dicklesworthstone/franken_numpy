//! Comprehensive tests for fnp-iter.
//!
//! This test suite covers:
//! - TransferSelectorInput and select_transfer_class()
//! - overlap_copy_policy() for memory overlap detection
//! - validate_nditer_flags()
//! - FlatIterIndex variants and resolve_flatiter_indices()
//! - validate_flatiter_read/write()
//! - TransferError reason codes
//!
//! Motivation: fnp-iter had 0 test files despite 3745 lines of source.
//! These tests port NumPy iterator semantics verification to Rust.

use fnp_iter::{
    FlatIterContractError, FlatIterIndex, NditerTransferFlags, OverlapAction,
    TRANSFER_PACKET_REASON_CODES, TransferClass, TransferError, TransferSelectorInput,
    overlap_copy_policy, resolve_flatiter_indices, select_transfer_class, validate_flatiter_read,
    validate_flatiter_write, validate_nditer_flags,
};

// ─────────────────────────────────────────────────────────────────────────────
// select_transfer_class() tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn transfer_class_contiguous_aligned_unit_stride() {
    let input = TransferSelectorInput {
        src_stride: 8,
        dst_stride: 8,
        item_size: 8,
        element_count: 100,
        aligned: true,
        cast_is_lossless: true,
        same_value_cast: false,
    };
    assert_eq!(select_transfer_class(input), Ok(TransferClass::Contiguous));
}

#[test]
fn transfer_class_strided_non_unit_stride() {
    let input = TransferSelectorInput {
        src_stride: 16,
        dst_stride: 8,
        item_size: 8,
        element_count: 50,
        aligned: true,
        cast_is_lossless: true,
        same_value_cast: false,
    };
    assert_eq!(select_transfer_class(input), Ok(TransferClass::Strided));
}

#[test]
fn transfer_class_strided_cast_lossy() {
    let input = TransferSelectorInput {
        src_stride: 8,
        dst_stride: 8,
        item_size: 8,
        element_count: 100,
        aligned: true,
        cast_is_lossless: false,
        same_value_cast: false,
    };
    assert_eq!(select_transfer_class(input), Ok(TransferClass::StridedCast));
}

#[test]
fn transfer_class_rejects_zero_item_size() {
    let input = TransferSelectorInput {
        src_stride: 8,
        dst_stride: 8,
        item_size: 0,
        element_count: 100,
        aligned: true,
        cast_is_lossless: true,
        same_value_cast: false,
    };
    assert!(matches!(
        select_transfer_class(input),
        Err(TransferError::SelectorInvalidContext(_))
    ));
}

#[test]
fn transfer_class_rejects_zero_element_count() {
    let input = TransferSelectorInput {
        src_stride: 8,
        dst_stride: 8,
        item_size: 8,
        element_count: 0,
        aligned: true,
        cast_is_lossless: true,
        same_value_cast: false,
    };
    assert!(matches!(
        select_transfer_class(input),
        Err(TransferError::SelectorInvalidContext(_))
    ));
}

#[test]
fn transfer_class_rejects_lossy_same_value_cast() {
    let input = TransferSelectorInput {
        src_stride: 8,
        dst_stride: 8,
        item_size: 8,
        element_count: 100,
        aligned: true,
        cast_is_lossless: false,
        same_value_cast: true,
    };
    assert!(matches!(
        select_transfer_class(input),
        Err(TransferError::SameValueCastRejected)
    ));
}

#[test]
fn transfer_class_rejects_non_multiple_stride() {
    let input = TransferSelectorInput {
        src_stride: 7,
        dst_stride: 8,
        item_size: 8,
        element_count: 100,
        aligned: true,
        cast_is_lossless: true,
        same_value_cast: false,
    };
    assert!(matches!(
        select_transfer_class(input),
        Err(TransferError::SelectorInvalidContext(_))
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// overlap_copy_policy() tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn overlap_disjoint_src_before_dst() {
    assert_eq!(overlap_copy_policy(0, 100, 50), Ok(OverlapAction::NoCopy));
}

#[test]
fn overlap_disjoint_dst_before_src() {
    assert_eq!(overlap_copy_policy(100, 0, 50), Ok(OverlapAction::NoCopy));
}

#[test]
fn overlap_forward_copy_needed() {
    assert_eq!(
        overlap_copy_policy(50, 25, 50),
        Ok(OverlapAction::ForwardCopy)
    );
}

#[test]
fn overlap_backward_copy_needed() {
    assert_eq!(
        overlap_copy_policy(25, 50, 50),
        Ok(OverlapAction::BackwardCopy)
    );
}

#[test]
fn overlap_exact_same_range() {
    assert_eq!(
        overlap_copy_policy(100, 100, 50),
        Ok(OverlapAction::ForwardCopy)
    );
}

#[test]
fn overlap_rejects_zero_byte_len() {
    assert!(matches!(
        overlap_copy_policy(0, 100, 0),
        Err(TransferError::OverlapPolicyTriggered(_))
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// validate_nditer_flags() tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nditer_flags_valid_no_constraints() {
    let flags = NditerTransferFlags {
        copy_if_overlap: true,
        no_broadcast: false,
        observed_overlap: false,
        observed_broadcast: false,
    };
    assert!(validate_nditer_flags(flags).is_ok());
}

#[test]
fn nditer_flags_rejects_broadcast_violation() {
    let flags = NditerTransferFlags {
        copy_if_overlap: true,
        no_broadcast: true,
        observed_overlap: false,
        observed_broadcast: true,
    };
    assert!(matches!(
        validate_nditer_flags(flags),
        Err(TransferError::NditerOverlapPolicy(_))
    ));
}

#[test]
fn nditer_flags_rejects_overlap_violation() {
    let flags = NditerTransferFlags {
        copy_if_overlap: false,
        no_broadcast: false,
        observed_overlap: true,
        observed_broadcast: false,
    };
    assert!(matches!(
        validate_nditer_flags(flags),
        Err(TransferError::NditerOverlapPolicy(_))
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// resolve_flatiter_indices() tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn flatiter_single_index_valid() {
    let indices = resolve_flatiter_indices(10, &FlatIterIndex::Single(5));
    assert_eq!(indices, Ok(vec![5]));
}

#[test]
fn flatiter_single_index_out_of_bounds() {
    let indices = resolve_flatiter_indices(10, &FlatIterIndex::Single(15));
    assert!(matches!(
        indices,
        Err(FlatIterContractError::IndexingViolation(_))
    ));
}

#[test]
fn flatiter_slice_basic() {
    let indices = resolve_flatiter_indices(
        10,
        &FlatIterIndex::Slice {
            start: 2,
            stop: 7,
            step: 1,
        },
    );
    assert_eq!(indices, Ok(vec![2, 3, 4, 5, 6]));
}

#[test]
fn flatiter_slice_with_step() {
    let indices = resolve_flatiter_indices(
        10,
        &FlatIterIndex::Slice {
            start: 0,
            stop: 10,
            step: 2,
        },
    );
    assert_eq!(indices, Ok(vec![0, 2, 4, 6, 8]));
}

#[test]
fn flatiter_slice_clamped_to_len() {
    let indices = resolve_flatiter_indices(
        5,
        &FlatIterIndex::Slice {
            start: 0,
            stop: 100,
            step: 1,
        },
    );
    assert_eq!(indices, Ok(vec![0, 1, 2, 3, 4]));
}

#[test]
fn flatiter_slice_empty_when_start_equals_stop() {
    let indices = resolve_flatiter_indices(
        10,
        &FlatIterIndex::Slice {
            start: 5,
            stop: 5,
            step: 1,
        },
    );
    assert_eq!(indices, Ok(vec![]));
}

#[test]
fn flatiter_slice_rejects_zero_step() {
    let indices = resolve_flatiter_indices(
        10,
        &FlatIterIndex::Slice {
            start: 0,
            stop: 5,
            step: 0,
        },
    );
    assert!(matches!(
        indices,
        Err(FlatIterContractError::IndexingViolation(_))
    ));
}

#[test]
fn flatiter_slice_rejects_inverted_bounds() {
    let indices = resolve_flatiter_indices(
        10,
        &FlatIterIndex::Slice {
            start: 7,
            stop: 3,
            step: 1,
        },
    );
    assert!(matches!(
        indices,
        Err(FlatIterContractError::IndexingViolation(_))
    ));
}

#[test]
fn flatiter_fancy_valid() {
    let indices = resolve_flatiter_indices(10, &FlatIterIndex::Fancy(vec![1, 5, 3, 9]));
    assert_eq!(indices, Ok(vec![1, 5, 3, 9]));
}

#[test]
fn flatiter_fancy_out_of_bounds() {
    let indices = resolve_flatiter_indices(10, &FlatIterIndex::Fancy(vec![1, 15, 3]));
    assert!(matches!(
        indices,
        Err(FlatIterContractError::IndexingViolation(_))
    ));
}

#[test]
fn flatiter_bool_mask_valid() {
    let mask = vec![true, false, true, false, true];
    let indices = resolve_flatiter_indices(5, &FlatIterIndex::BoolMask(mask));
    assert_eq!(indices, Ok(vec![0, 2, 4]));
}

#[test]
fn flatiter_bool_mask_all_false() {
    let mask = vec![false, false, false];
    let indices = resolve_flatiter_indices(3, &FlatIterIndex::BoolMask(mask));
    assert_eq!(indices, Ok(vec![]));
}

#[test]
fn flatiter_bool_mask_length_mismatch() {
    let mask = vec![true, false, true];
    let indices = resolve_flatiter_indices(10, &FlatIterIndex::BoolMask(mask));
    assert!(matches!(
        indices,
        Err(FlatIterContractError::IndexingViolation(_))
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// validate_flatiter_read/write tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn flatiter_read_valid() {
    let count = validate_flatiter_read(10, &FlatIterIndex::Single(5));
    assert_eq!(count, Ok(1));
}

#[test]
fn flatiter_read_slice_count() {
    let count = validate_flatiter_read(
        10,
        &FlatIterIndex::Slice {
            start: 0,
            stop: 5,
            step: 1,
        },
    );
    assert_eq!(count, Ok(5));
}

#[test]
fn flatiter_write_scalar_value() {
    let count = validate_flatiter_write(
        10,
        &FlatIterIndex::Slice {
            start: 0,
            stop: 5,
            step: 1,
        },
        1,
    );
    assert_eq!(count, Ok(5));
}

#[test]
fn flatiter_write_matching_values() {
    let count = validate_flatiter_write(
        10,
        &FlatIterIndex::Slice {
            start: 0,
            stop: 5,
            step: 1,
        },
        5,
    );
    assert_eq!(count, Ok(5));
}

#[test]
fn flatiter_write_mismatched_values() {
    let result = validate_flatiter_write(
        10,
        &FlatIterIndex::Slice {
            start: 0,
            stop: 5,
            step: 1,
        },
        3,
    );
    assert!(matches!(
        result,
        Err(TransferError::FlatiterWriteViolation(_))
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// TransferError reason_code() tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn transfer_error_reason_codes_match_constants() {
    let errors = [
        TransferError::SelectorInvalidContext(""),
        TransferError::OverlapPolicyTriggered(""),
        TransferError::WhereMaskContractViolation(""),
        TransferError::SameValueCastRejected,
        TransferError::StringWidthMismatch(""),
        TransferError::SubarrayBroadcastContractViolation(""),
        TransferError::FlatiterReadViolation(""),
        TransferError::FlatiterWriteViolation(""),
        TransferError::NditerOverlapPolicy(""),
        TransferError::FpeCastError(""),
    ];

    for err in &errors {
        let code = err.reason_code();
        assert!(
            TRANSFER_PACKET_REASON_CODES.contains(&code),
            "reason_code '{}' not in TRANSFER_PACKET_REASON_CODES",
            code
        );
    }
}

#[test]
fn transfer_error_display_not_empty() {
    let errors = [
        TransferError::SelectorInvalidContext("test message"),
        TransferError::SameValueCastRejected,
    ];
    for err in &errors {
        let msg = format!("{}", err);
        assert!(!msg.is_empty(), "TransferError Display should not be empty");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FlatIterContractError tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn flatiter_contract_error_reason_code() {
    let err = FlatIterContractError::IndexingViolation("test");
    assert_eq!(err.reason_code(), "flatiter_indexing_contract_violation");
}

#[test]
fn flatiter_contract_error_display() {
    let err = FlatIterContractError::IndexingViolation("out of bounds");
    let msg = format!("{}", err);
    assert!(msg.contains("out of bounds"));
}
