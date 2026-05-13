//! Golden tests for fnp-ndarray layout computations.
//!
//! Tests broadcast_shape, broadcast_shapes, element_count, contiguous_strides,
//! broadcast_strides, and NdLayout operations against known-good outputs.
//!
//! Finding: fnp-ndarray had only 2 test files for 1531 LOC. This adds golden
//! tests for the core shape/stride computation functions.

use fnp_ndarray::{
    MemoryOrder, NdLayout, broadcast_shape, broadcast_shapes, broadcast_strides,
    contiguous_strides, element_count, fix_unknown_dimension,
};

// ─────────────────────────────────────────────────────────────────────────────
// element_count golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_element_count_scalar() {
    assert_eq!(element_count(&[]).unwrap(), 1);
}

#[test]
fn golden_element_count_1d() {
    assert_eq!(element_count(&[5]).unwrap(), 5);
    assert_eq!(element_count(&[100]).unwrap(), 100);
}

#[test]
fn golden_element_count_2d() {
    assert_eq!(element_count(&[3, 4]).unwrap(), 12);
    assert_eq!(element_count(&[10, 10]).unwrap(), 100);
}

#[test]
fn golden_element_count_3d() {
    assert_eq!(element_count(&[2, 3, 4]).unwrap(), 24);
    assert_eq!(element_count(&[5, 5, 5]).unwrap(), 125);
}

#[test]
fn golden_element_count_with_ones() {
    assert_eq!(element_count(&[1, 1, 1]).unwrap(), 1);
    assert_eq!(element_count(&[1, 5, 1]).unwrap(), 5);
}

#[test]
fn golden_element_count_with_zero() {
    assert_eq!(element_count(&[0]).unwrap(), 0);
    assert_eq!(element_count(&[3, 0, 4]).unwrap(), 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// contiguous_strides golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_contiguous_strides_1d_c() {
    let strides = contiguous_strides(&[5], 8, MemoryOrder::C).unwrap();
    assert_eq!(strides, vec![8isize]);
}

#[test]
fn golden_contiguous_strides_2d_c() {
    let strides = contiguous_strides(&[3, 4], 8, MemoryOrder::C).unwrap();
    assert_eq!(strides, vec![32isize, 8]); // row-major: inner stride is item_size
}

#[test]
fn golden_contiguous_strides_2d_f() {
    let strides = contiguous_strides(&[3, 4], 8, MemoryOrder::F).unwrap();
    assert_eq!(strides, vec![8isize, 24]); // column-major: outer stride is item_size
}

#[test]
fn golden_contiguous_strides_3d_c() {
    let strides = contiguous_strides(&[2, 3, 4], 8, MemoryOrder::C).unwrap();
    assert_eq!(strides, vec![96isize, 32, 8]); // 3*4*8, 4*8, 8
}

#[test]
fn golden_contiguous_strides_3d_f() {
    let strides = contiguous_strides(&[2, 3, 4], 8, MemoryOrder::F).unwrap();
    assert_eq!(strides, vec![8isize, 16, 48]); // 8, 2*8, 2*3*8
}

#[test]
fn golden_contiguous_strides_scalar() {
    let strides = contiguous_strides(&[], 8, MemoryOrder::C).unwrap();
    assert_eq!(strides, Vec::<isize>::new());
}

// ─────────────────────────────────────────────────────────────────────────────
// broadcast_shape golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_broadcast_shape_same() {
    let result = broadcast_shape(&[3, 4], &[3, 4]).unwrap();
    assert_eq!(result, vec![3, 4]);
}

#[test]
fn golden_broadcast_shape_scalar_to_array() {
    let result = broadcast_shape(&[], &[3, 4]).unwrap();
    assert_eq!(result, vec![3, 4]);
}

#[test]
fn golden_broadcast_shape_array_to_scalar() {
    let result = broadcast_shape(&[3, 4], &[]).unwrap();
    assert_eq!(result, vec![3, 4]);
}

#[test]
fn golden_broadcast_shape_1_to_n() {
    let result = broadcast_shape(&[1, 4], &[3, 4]).unwrap();
    assert_eq!(result, vec![3, 4]);
}

#[test]
fn golden_broadcast_shape_n_to_1() {
    let result = broadcast_shape(&[3, 4], &[3, 1]).unwrap();
    assert_eq!(result, vec![3, 4]);
}

#[test]
fn golden_broadcast_shape_extend_dims() {
    let result = broadcast_shape(&[4], &[3, 4]).unwrap();
    assert_eq!(result, vec![3, 4]);
}

#[test]
fn golden_broadcast_shape_complex() {
    let result = broadcast_shape(&[1, 5, 1], &[4, 1, 3]).unwrap();
    assert_eq!(result, vec![4, 5, 3]);
}

#[test]
fn golden_broadcast_shape_incompatible() {
    let result = broadcast_shape(&[3, 4], &[5, 4]);
    assert!(result.is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// broadcast_shapes (multiple) golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_broadcast_shapes_single() {
    let result = broadcast_shapes(&[&[3, 4][..]]).unwrap();
    assert_eq!(result, vec![3, 4]);
}

#[test]
fn golden_broadcast_shapes_two() {
    let result = broadcast_shapes(&[&[3, 1][..], &[1, 4][..]]).unwrap();
    assert_eq!(result, vec![3, 4]);
}

#[test]
fn golden_broadcast_shapes_three() {
    let result = broadcast_shapes(&[&[1, 5, 1][..], &[4, 1, 1][..], &[1, 1, 3][..]]).unwrap();
    assert_eq!(result, vec![4, 5, 3]);
}

#[test]
fn golden_broadcast_shapes_mixed_dims() {
    let result = broadcast_shapes(&[&[5][..], &[3, 5][..], &[2, 3, 5][..]]).unwrap();
    assert_eq!(result, vec![2, 3, 5]);
}

// ─────────────────────────────────────────────────────────────────────────────
// broadcast_strides golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_broadcast_strides_no_broadcast() {
    let strides = broadcast_strides(&[3, 4], &[32, 8], &[3, 4]).unwrap();
    assert_eq!(strides, vec![32isize, 8]);
}

#[test]
fn golden_broadcast_strides_extend_dims() {
    let strides = broadcast_strides(&[4], &[8], &[3, 4]).unwrap();
    assert_eq!(strides, vec![0isize, 8]); // new dim gets stride 0
}

#[test]
fn golden_broadcast_strides_1_to_n() {
    let strides = broadcast_strides(&[1, 4], &[32, 8], &[3, 4]).unwrap();
    assert_eq!(strides, vec![0isize, 8]); // broadcast dim gets stride 0
}

#[test]
fn golden_broadcast_strides_scalar() {
    let strides = broadcast_strides(&[], &[], &[3, 4]).unwrap();
    assert_eq!(strides, vec![0isize, 0]); // all strides 0 for scalar broadcast
}

// ─────────────────────────────────────────────────────────────────────────────
// fix_unknown_dimension golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_fix_unknown_dimension_no_unknown() {
    let result = fix_unknown_dimension(&[3, 4], 12).unwrap();
    assert_eq!(result, vec![3, 4]);
}

#[test]
fn golden_fix_unknown_dimension_infer_first() {
    let result = fix_unknown_dimension(&[-1, 4], 12).unwrap();
    assert_eq!(result, vec![3, 4]);
}

#[test]
fn golden_fix_unknown_dimension_infer_last() {
    let result = fix_unknown_dimension(&[3, -1], 12).unwrap();
    assert_eq!(result, vec![3, 4]);
}

#[test]
fn golden_fix_unknown_dimension_infer_middle() {
    let result = fix_unknown_dimension(&[2, -1, 3], 24).unwrap();
    assert_eq!(result, vec![2, 4, 3]);
}

// ─────────────────────────────────────────────────────────────────────────────
// NdLayout golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_ndlayout_contiguous_2d() {
    let layout = NdLayout::contiguous(vec![3, 4], 8, MemoryOrder::C).unwrap();
    assert_eq!(layout.shape, vec![3, 4]);
    assert_eq!(layout.strides, vec![32isize, 8]);
    assert!(layout.is_contiguous());
}

#[test]
fn golden_ndlayout_contiguous_fortran() {
    let layout = NdLayout::contiguous(vec![3, 4], 8, MemoryOrder::F).unwrap();
    assert_eq!(layout.shape, vec![3, 4]);
    assert_eq!(layout.strides, vec![8isize, 24]);
    assert!(layout.is_fortran_contiguous());
}

#[test]
fn golden_ndlayout_broadcast_to() {
    let layout = NdLayout::contiguous(vec![1, 4], 8, MemoryOrder::C).unwrap();
    let broadcast = layout.broadcast_to(vec![3, 4]).unwrap();
    assert_eq!(broadcast.shape, vec![3, 4]);
    assert_eq!(broadcast.strides, vec![0isize, 8]); // first dim is broadcast (stride 0)
}

#[test]
fn golden_ndlayout_nbytes() {
    let layout = NdLayout::contiguous(vec![3, 4], 8, MemoryOrder::C).unwrap();
    assert_eq!(layout.nbytes().unwrap(), 96); // 3 * 4 * 8
}

#[test]
fn golden_ndlayout_ndim() {
    let layout = NdLayout::contiguous(vec![2, 3, 4], 8, MemoryOrder::C).unwrap();
    assert_eq!(layout.ndim(), 3);
}

#[test]
fn golden_ndlayout_scalar() {
    let layout = NdLayout::contiguous(vec![], 8, MemoryOrder::C).unwrap();
    assert_eq!(layout.shape, Vec::<usize>::new());
    assert_eq!(layout.ndim(), 0);
    assert!(layout.is_contiguous());
}

#[test]
fn golden_ndlayout_as_strided() {
    let layout = NdLayout::contiguous(vec![6], 8, MemoryOrder::C).unwrap();
    let strided = layout.as_strided(vec![2, 3], vec![24, 8]).unwrap();
    assert_eq!(strided.shape, vec![2, 3]);
    assert_eq!(strided.strides, vec![24isize, 8]);
}

#[test]
fn golden_ndlayout_is_contiguous_false_for_strided() {
    let layout = NdLayout::contiguous(vec![6], 8, MemoryOrder::C).unwrap();
    let strided = layout.as_strided(vec![3], vec![16]).unwrap(); // stride != item_size
    assert!(!strided.is_contiguous());
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_element_count_large() {
    assert_eq!(element_count(&[1000, 1000, 1000]).unwrap(), 1_000_000_000);
}

#[test]
fn golden_broadcast_shape_4d() {
    let result = broadcast_shape(&[1, 2, 1, 4], &[5, 1, 3, 1]).unwrap();
    assert_eq!(result, vec![5, 2, 3, 4]);
}
