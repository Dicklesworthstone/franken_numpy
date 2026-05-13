//! Comprehensive strided view tests for fnp-ndarray.
//!
//! Tests NdLayout::as_strided with various stride patterns including
//! negative strides, zero strides (broadcasting), and edge cases.

use fnp_ndarray::{MemoryOrder, NdLayout};

// ─────────────────────────────────────────────────────────────────────────────
// Original regression test
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_strided_flaw_negative_stride() {
    let layout = NdLayout::contiguous(vec![5], 8, MemoryOrder::C).unwrap();
    let view = layout.as_strided(vec![5], vec![-8]);
    assert!(view.is_ok(), "negative stride view should be valid");
}

// ─────────────────────────────────────────────────────────────────────────────
// Positive stride tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_strided_contiguous_1d() {
    let layout = NdLayout::contiguous(vec![10], 8, MemoryOrder::C).unwrap();
    let view = layout.as_strided(vec![10], vec![8]);
    assert!(view.is_ok());
    let v = view.unwrap();
    assert_eq!(v.shape.as_slice(), &[10]);
    assert_eq!(v.strides.as_slice(), &[8]);
}

#[test]
fn test_strided_contiguous_2d_c_order() {
    let layout = NdLayout::contiguous(vec![3, 4], 8, MemoryOrder::C).unwrap();
    let view = layout.as_strided(vec![3, 4], vec![32, 8]);
    assert!(view.is_ok());
    let v = view.unwrap();
    assert_eq!(v.shape.as_slice(), &[3, 4]);
    assert_eq!(v.strides.as_slice(), &[32, 8]);
}

#[test]
fn test_strided_contiguous_2d_f_order() {
    let layout = NdLayout::contiguous(vec![3, 4], 8, MemoryOrder::F).unwrap();
    let view = layout.as_strided(vec![3, 4], vec![8, 24]);
    assert!(view.is_ok());
    let v = view.unwrap();
    assert_eq!(v.shape.as_slice(), &[3, 4]);
    assert_eq!(v.strides.as_slice(), &[8, 24]);
}

#[test]
fn test_strided_skip_elements() {
    let layout = NdLayout::contiguous(vec![10], 8, MemoryOrder::C).unwrap();
    let view = layout.as_strided(vec![5], vec![16]);
    assert!(view.is_ok());
    let v = view.unwrap();
    assert_eq!(v.shape.as_slice(), &[5]);
    assert_eq!(v.strides.as_slice(), &[16]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Negative stride tests (reverse iteration)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_strided_negative_1d() {
    let layout = NdLayout::contiguous(vec![5], 8, MemoryOrder::C).unwrap();
    let view = layout.as_strided(vec![5], vec![-8]);
    assert!(view.is_ok(), "negative stride should be valid");
}

#[test]
fn test_strided_negative_2d_axis0() {
    let layout = NdLayout::contiguous(vec![3, 4], 8, MemoryOrder::C).unwrap();
    let view = layout.as_strided(vec![3, 4], vec![-32, 8]);
    assert!(view.is_ok(), "negative stride on axis 0 should be valid");
}

#[test]
fn test_strided_negative_2d_axis1() {
    let layout = NdLayout::contiguous(vec![3, 4], 8, MemoryOrder::C).unwrap();
    let view = layout.as_strided(vec![3, 4], vec![32, -8]);
    assert!(view.is_ok(), "negative stride on axis 1 should be valid");
}

#[test]
fn test_strided_negative_both_axes() {
    let layout = NdLayout::contiguous(vec![3, 4], 8, MemoryOrder::C).unwrap();
    let view = layout.as_strided(vec![3, 4], vec![-32, -8]);
    assert!(
        view.is_ok(),
        "negative strides on both axes should be valid"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Zero stride tests (broadcasting)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_strided_zero_stride_broadcast_row() {
    let layout = NdLayout::contiguous(vec![4], 8, MemoryOrder::C).unwrap();
    let view = layout.as_strided(vec![3, 4], vec![0, 8]);
    assert!(view.is_ok(), "zero stride (broadcast) should be valid");
    let v = view.unwrap();
    assert_eq!(v.shape.as_slice(), &[3, 4]);
    assert_eq!(v.strides.as_slice(), &[0, 8]);
}

#[test]
fn test_strided_zero_stride_broadcast_col() {
    let layout = NdLayout::contiguous(vec![3], 8, MemoryOrder::C).unwrap();
    let view = layout.as_strided(vec![3, 4], vec![8, 0]);
    assert!(view.is_ok(), "zero stride (broadcast) should be valid");
    let v = view.unwrap();
    assert_eq!(v.shape.as_slice(), &[3, 4]);
    assert_eq!(v.strides.as_slice(), &[8, 0]);
}

#[test]
fn test_strided_scalar_broadcast() {
    let layout = NdLayout::contiguous(vec![1], 8, MemoryOrder::C).unwrap();
    let view = layout.as_strided(vec![5, 5], vec![0, 0]);
    assert!(view.is_ok(), "scalar broadcast should be valid");
    let v = view.unwrap();
    assert_eq!(v.shape.as_slice(), &[5, 5]);
    assert_eq!(v.strides.as_slice(), &[0, 0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_strided_empty_shape() {
    let layout = NdLayout::contiguous(vec![], 8, MemoryOrder::C).unwrap();
    let view = layout.as_strided(vec![], vec![]);
    assert!(view.is_ok(), "scalar (empty shape) should be valid");
    let v = view.unwrap();
    assert!(v.shape.as_slice().is_empty());
    assert!(v.strides.as_slice().is_empty());
}

#[test]
fn test_strided_zero_dim() {
    let layout = NdLayout::contiguous(vec![0, 5], 8, MemoryOrder::C).unwrap();
    let view = layout.as_strided(vec![0, 5], vec![40, 8]);
    assert!(view.is_ok(), "zero-size dimension should be valid");
    let v = view.unwrap();
    assert_eq!(v.shape.as_slice(), &[0, 5]);
}

#[test]
fn test_strided_single_element() {
    let layout = NdLayout::contiguous(vec![1], 8, MemoryOrder::C).unwrap();
    let view = layout.as_strided(vec![1], vec![8]);
    assert!(view.is_ok());
    let v = view.unwrap();
    assert_eq!(v.shape.as_slice(), &[1]);
}

#[test]
fn test_strided_3d() {
    let layout = NdLayout::contiguous(vec![2, 3, 4], 8, MemoryOrder::C).unwrap();
    let view = layout.as_strided(vec![2, 3, 4], vec![96, 32, 8]);
    assert!(view.is_ok());
    let v = view.unwrap();
    assert_eq!(v.shape.as_slice(), &[2, 3, 4]);
    assert_eq!(v.strides.as_slice(), &[96, 32, 8]);
}

#[test]
fn test_strided_reshape_via_stride() {
    let layout = NdLayout::contiguous(vec![12], 8, MemoryOrder::C).unwrap();
    let view = layout.as_strided(vec![3, 4], vec![32, 8]);
    assert!(view.is_ok(), "reshape via strides should be valid");
    let v = view.unwrap();
    assert_eq!(v.shape.as_slice(), &[3, 4]);
}

#[test]
fn test_strided_transpose_via_stride() {
    let layout = NdLayout::contiguous(vec![3, 4], 8, MemoryOrder::C).unwrap();
    let view = layout.as_strided(vec![4, 3], vec![8, 32]);
    assert!(view.is_ok(), "transpose via strides should be valid");
    let v = view.unwrap();
    assert_eq!(v.shape.as_slice(), &[4, 3]);
    assert_eq!(v.strides.as_slice(), &[8, 32]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Contiguity checks
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_contiguous_layout_is_contiguous() {
    let layout = NdLayout::contiguous(vec![3, 4], 8, MemoryOrder::C).unwrap();
    assert!(layout.is_contiguous());
}

#[test]
fn test_strided_view_may_not_be_contiguous() {
    let layout = NdLayout::contiguous(vec![10], 8, MemoryOrder::C).unwrap();
    let view = layout.as_strided(vec![5], vec![16]).unwrap();
    assert!(!view.is_contiguous(), "skip-stride view is not contiguous");
}

#[test]
fn test_broadcast_view_not_contiguous() {
    let layout = NdLayout::contiguous(vec![4], 8, MemoryOrder::C).unwrap();
    let view = layout.as_strided(vec![3, 4], vec![0, 8]).unwrap();
    assert!(!view.is_contiguous(), "broadcast view is not contiguous");
}
