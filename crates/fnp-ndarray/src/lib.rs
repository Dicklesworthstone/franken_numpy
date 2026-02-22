#![forbid(unsafe_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrder {
    C,
    F,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    InvalidItemSize,
    InvalidDimension(isize),
    MultipleUnknownDimensions,
    Overflow,
    RankMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidWindowDimension {
        axis: usize,
        window: usize,
        dim: usize,
    },
    NegativeStride(isize),
    OutOfBoundsView {
        required_nbytes: usize,
        available_nbytes: usize,
    },
    IncompatibleBroadcast {
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },
    IncompatibleElementCount {
        old: usize,
        new: usize,
    },
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidItemSize => write!(f, "item size must be > 0"),
            Self::InvalidDimension(dim) => write!(f, "invalid dimension {dim}"),
            Self::MultipleUnknownDimensions => write!(f, "only one -1 dimension is allowed"),
            Self::Overflow => write!(f, "size arithmetic overflow"),
            Self::RankMismatch { expected, actual } => {
                write!(f, "rank mismatch expected={expected} actual={actual}")
            }
            Self::InvalidWindowDimension { axis, window, dim } => write!(
                f,
                "invalid sliding window on axis={axis} window={window} dim={dim}"
            ),
            Self::NegativeStride(stride) => {
                write!(f, "negative strides are not supported stride={stride}")
            }
            Self::OutOfBoundsView {
                required_nbytes,
                available_nbytes,
            } => write!(
                f,
                "view exceeds base storage required_nbytes={required_nbytes} available_nbytes={available_nbytes}"
            ),
            Self::IncompatibleBroadcast { lhs, rhs } => {
                write!(f, "cannot broadcast {:?} with {:?}", lhs, rhs)
            }
            Self::IncompatibleElementCount { old, new } => {
                write!(f, "element count mismatch old={old} new={new}")
            }
        }
    }
}

impl std::error::Error for ShapeError {}

#[must_use]
pub fn can_broadcast(lhs: &[usize], rhs: &[usize]) -> bool {
    broadcast_shape(lhs, rhs).is_ok()
}

pub fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, ShapeError> {
    if lhs == rhs {
        return Ok(lhs.to_vec());
    }

    if lhs.len() == rhs.len() {
        let mut out = Vec::with_capacity(lhs.len());
        for (&l, &r) in lhs.iter().zip(rhs.iter()) {
            let merged = if l == r {
                l
            } else if l == 1 {
                r
            } else if r == 1 {
                l
            } else {
                return Err(ShapeError::IncompatibleBroadcast {
                    lhs: lhs.to_vec(),
                    rhs: rhs.to_vec(),
                });
            };
            out.push(merged);
        }
        return Ok(out);
    }

    let nd = lhs.len().max(rhs.len());
    let mut out = Vec::with_capacity(nd);

    for idx in 0..nd {
        let l = *lhs.get(lhs.len().wrapping_sub(1 + idx)).unwrap_or(&1);
        let r = *rhs.get(rhs.len().wrapping_sub(1 + idx)).unwrap_or(&1);

        let merged = if l == r {
            l
        } else if l == 1 {
            r
        } else if r == 1 {
            l
        } else {
            return Err(ShapeError::IncompatibleBroadcast {
                lhs: lhs.to_vec(),
                rhs: rhs.to_vec(),
            });
        };

        out.push(merged);
    }

    out.reverse();
    Ok(out)
}

pub fn broadcast_shapes(shapes: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
    let mut acc = Vec::new();
    for shape in shapes {
        acc = broadcast_shape(&acc, shape)?;
    }
    Ok(acc)
}

pub fn element_count(shape: &[usize]) -> Result<usize, ShapeError> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or(ShapeError::Overflow)
    })
}

pub fn fix_unknown_dimension(
    new_shape: &[isize],
    old_element_count: usize,
) -> Result<Vec<usize>, ShapeError> {
    let mut known_product: usize = 1;
    let mut unknown_index: Option<usize> = None;
    let mut out = Vec::with_capacity(new_shape.len());

    for (idx, &dim) in new_shape.iter().enumerate() {
        match dim {
            -1 => {
                if unknown_index.replace(idx).is_some() {
                    return Err(ShapeError::MultipleUnknownDimensions);
                }
                out.push(0);
            }
            d if d < -1 => return Err(ShapeError::InvalidDimension(d)),
            d => {
                let d = usize::try_from(d).map_err(|_| ShapeError::InvalidDimension(d))?;
                known_product = known_product.checked_mul(d).ok_or(ShapeError::Overflow)?;
                out.push(d);
            }
        }
    }

    match unknown_index {
        Some(idx) => {
            if known_product == 0 {
                if old_element_count == 0 {
                    out[idx] = 0;
                } else {
                    return Err(ShapeError::IncompatibleElementCount {
                        old: old_element_count,
                        new: known_product,
                    });
                }
            } else if !old_element_count.is_multiple_of(known_product) {
                return Err(ShapeError::IncompatibleElementCount {
                    old: old_element_count,
                    new: known_product,
                });
            } else {
                out[idx] = old_element_count / known_product;
            }
        }
        None => {
            if known_product != old_element_count {
                return Err(ShapeError::IncompatibleElementCount {
                    old: old_element_count,
                    new: known_product,
                });
            }
        }
    }

    Ok(out)
}

pub fn contiguous_strides(
    shape: &[usize],
    item_size: usize,
    order: MemoryOrder,
) -> Result<Vec<isize>, ShapeError> {
    if item_size == 0 {
        return Err(ShapeError::InvalidItemSize);
    }

    let mut strides_bytes = vec![0usize; shape.len()];

    match order {
        MemoryOrder::C => {
            let mut stride = item_size;
            for (i, &dim) in shape.iter().enumerate().rev() {
                strides_bytes[i] = stride;
                stride = stride.checked_mul(dim).ok_or(ShapeError::Overflow)?;
            }
        }
        MemoryOrder::F => {
            let mut stride = item_size;
            for (i, &dim) in shape.iter().enumerate() {
                strides_bytes[i] = stride;
                stride = stride.checked_mul(dim).ok_or(ShapeError::Overflow)?;
            }
        }
    }

    strides_bytes
        .into_iter()
        .map(|s| isize::try_from(s).map_err(|_| ShapeError::Overflow))
        .collect()
}

pub fn broadcast_strides(
    src_shape: &[usize],
    src_strides: &[isize],
    dst_shape: &[usize],
) -> Result<Vec<isize>, ShapeError> {
    if src_shape.len() != src_strides.len() {
        return Err(ShapeError::RankMismatch {
            expected: src_shape.len(),
            actual: src_strides.len(),
        });
    }

    let merged = broadcast_shape(src_shape, dst_shape)?;
    if merged != dst_shape {
        return Err(ShapeError::IncompatibleBroadcast {
            lhs: src_shape.to_vec(),
            rhs: dst_shape.to_vec(),
        });
    }

    if dst_shape.is_empty() {
        return Ok(Vec::new());
    }

    let mut out = vec![0isize; dst_shape.len()];
    let offset = dst_shape.len().saturating_sub(src_shape.len());

    for (axis, (&src_dim, &src_stride)) in src_shape.iter().zip(src_strides).enumerate() {
        let out_axis = axis + offset;
        let dst_dim = dst_shape[out_axis];
        out[out_axis] = if src_dim == dst_dim {
            src_stride
        } else if src_dim == 1 {
            0
        } else {
            return Err(ShapeError::IncompatibleBroadcast {
                lhs: src_shape.to_vec(),
                rhs: dst_shape.to_vec(),
            });
        };
    }

    Ok(out)
}

fn required_view_nbytes(
    shape: &[usize],
    strides: &[isize],
    item_size: usize,
) -> Result<usize, ShapeError> {
    if item_size == 0 {
        return Err(ShapeError::InvalidItemSize);
    }
    if shape.len() != strides.len() {
        return Err(ShapeError::RankMismatch {
            expected: shape.len(),
            actual: strides.len(),
        });
    }
    if shape.contains(&0) {
        return Ok(0);
    }

    // Compute the total byte span from the element with the smallest offset
    // to the element with the largest offset, plus item_size.
    // For negative strides, the minimum offset contribution is negative.
    let mut max_offset: isize = 0;
    let mut min_offset: isize = 0;
    for (&dim, &stride) in shape.iter().zip(strides) {
        if dim <= 1 {
            continue;
        }
        let end = stride
            .checked_mul((dim - 1) as isize)
            .ok_or(ShapeError::Overflow)?;
        if end >= 0 {
            max_offset = max_offset.checked_add(end).ok_or(ShapeError::Overflow)?;
        } else {
            min_offset = min_offset.checked_add(end).ok_or(ShapeError::Overflow)?;
        }
    }

    let span = max_offset
        .checked_sub(min_offset)
        .ok_or(ShapeError::Overflow)?;
    let span_usize = usize::try_from(span).map_err(|_| ShapeError::Overflow)?;
    span_usize
        .checked_add(item_size)
        .ok_or(ShapeError::Overflow)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NdLayout {
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub item_size: usize,
}

impl NdLayout {
    pub fn contiguous(
        shape: Vec<usize>,
        item_size: usize,
        order: MemoryOrder,
    ) -> Result<Self, ShapeError> {
        let strides = contiguous_strides(&shape, item_size, order)?;
        Ok(Self {
            shape,
            strides,
            item_size,
        })
    }

    pub fn nbytes(&self) -> Result<usize, ShapeError> {
        element_count(&self.shape)?
            .checked_mul(self.item_size)
            .ok_or(ShapeError::Overflow)
    }

    pub fn as_strided(&self, shape: Vec<usize>, strides: Vec<isize>) -> Result<Self, ShapeError> {
        let required_nbytes = required_view_nbytes(&shape, &strides, self.item_size)?;
        let available_nbytes = required_view_nbytes(&self.shape, &self.strides, self.item_size)?;
        if required_nbytes > available_nbytes {
            return Err(ShapeError::OutOfBoundsView {
                required_nbytes,
                available_nbytes,
            });
        }
        Ok(Self {
            shape,
            strides,
            item_size: self.item_size,
        })
    }

    pub fn broadcast_to(&self, shape: Vec<usize>) -> Result<Self, ShapeError> {
        let strides = broadcast_strides(&self.shape, &self.strides, &shape)?;
        Ok(Self {
            shape,
            strides,
            item_size: self.item_size,
        })
    }

    /// Returns the number of dimensions (rank) of this layout.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns `true` if the layout represents a C-contiguous (row-major) array.
    ///
    /// A C-contiguous array has strides that match what `contiguous_strides(..., C)`
    /// would produce: the last axis has stride == item_size, and each prior axis
    /// has stride == (product of subsequent dimensions) * item_size.
    ///
    /// A zero-dimensional or scalar layout is always considered contiguous.
    /// A layout with any zero-length dimension is contiguous by convention.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        if self.item_size == 0 {
            return false;
        }
        if self.shape.is_empty() {
            return true;
        }
        if self.shape.contains(&0) {
            return true;
        }
        let mut expected_stride = self.item_size as isize;
        for (&dim, &stride) in self.shape.iter().zip(&self.strides).rev() {
            if dim == 1 {
                // Dimensions of size 1 don't constrain the stride
                continue;
            }
            if stride != expected_stride {
                return false;
            }
            expected_stride = stride * dim as isize;
        }
        true
    }

    /// Returns `true` if the layout represents a Fortran-contiguous (column-major) array.
    ///
    /// A Fortran-contiguous array has strides that match what `contiguous_strides(..., F)`
    /// would produce: the first axis has stride == item_size, and each subsequent axis
    /// has stride == (product of prior dimensions) * item_size.
    #[must_use]
    pub fn is_fortran_contiguous(&self) -> bool {
        if self.item_size == 0 {
            return false;
        }
        if self.shape.is_empty() {
            return true;
        }
        if self.shape.contains(&0) {
            return true;
        }
        let mut expected_stride = self.item_size as isize;
        for (&dim, &stride) in self.shape.iter().zip(&self.strides) {
            if dim == 1 {
                continue;
            }
            if stride != expected_stride {
                return false;
            }
            expected_stride = stride * dim as isize;
        }
        true
    }

    pub fn sliding_window_view(&self, window_shape: Vec<usize>) -> Result<Self, ShapeError> {
        if window_shape.len() != self.shape.len() {
            return Err(ShapeError::RankMismatch {
                expected: self.shape.len(),
                actual: window_shape.len(),
            });
        }

        let mut shape = Vec::with_capacity(self.shape.len() * 2);
        for (axis, (&dim, &window)) in self.shape.iter().zip(&window_shape).enumerate() {
            if window == 0 || window > dim {
                return Err(ShapeError::InvalidWindowDimension { axis, window, dim });
            }
            shape.push(dim - window + 1);
        }
        shape.extend(window_shape.iter().copied());

        let mut strides = Vec::with_capacity(self.strides.len() * 2);
        strides.extend(self.strides.iter().copied());
        strides.extend(self.strides.iter().copied());

        Ok(Self {
            shape,
            strides,
            item_size: self.item_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MemoryOrder, NdLayout, ShapeError, broadcast_shape, broadcast_shapes, broadcast_strides,
        can_broadcast, contiguous_strides, element_count, fix_unknown_dimension,
    };

    #[test]
    fn broadcast_shape_matches_numpy_style() {
        let out = broadcast_shape(&[8, 1, 6, 1], &[7, 1, 5]).expect("broadcast should succeed");
        assert_eq!(out, vec![8, 7, 6, 5]);
    }

    #[test]
    fn broadcast_shape_handles_same_rank_axiswise_merge() {
        let out = broadcast_shape(&[1, 4, 1, 3], &[2, 1, 5, 3]).expect("broadcast should succeed");
        assert_eq!(out, vec![2, 4, 5, 3]);
    }

    #[test]
    fn broadcast_shape_preserves_identical_shape() {
        let shape = vec![2, 3, 4];
        let out = broadcast_shape(&shape, &shape).expect("identical shapes should broadcast");
        assert_eq!(out, shape);
    }

    #[test]
    fn broadcast_shape_rejects_incompatible_shapes() {
        let err = broadcast_shape(&[4, 3], &[5, 3]).expect_err("should fail");
        assert!(matches!(err, ShapeError::IncompatibleBroadcast { .. }));
        assert!(!can_broadcast(&[4, 3], &[5, 3]));
    }

    #[test]
    fn broadcast_many_shapes() {
        let shapes: [&[usize]; 3] = [&[3, 1], &[1, 7], &[5, 3, 7]];
        let out = broadcast_shapes(&shapes).expect("broadcast should succeed");
        assert_eq!(out, vec![5, 3, 7]);
    }

    #[test]
    fn contiguous_strides_c_and_f_orders() {
        let c = contiguous_strides(&[2, 3, 4], 8, MemoryOrder::C).expect("c-order");
        let f = contiguous_strides(&[2, 3, 4], 8, MemoryOrder::F).expect("f-order");
        assert_eq!(c, vec![96, 32, 8]);
        assert_eq!(f, vec![8, 16, 48]);
    }

    #[test]
    fn unknown_dimension_resolution() {
        let resolved = fix_unknown_dimension(&[2, -1, 3], 24).expect("should infer -1 dimension");
        assert_eq!(resolved, vec![2, 4, 3]);
    }

    #[test]
    fn unknown_dimension_validation() {
        let err = fix_unknown_dimension(&[-1, -1], 8).expect_err("only one unknown allowed");
        assert!(matches!(err, ShapeError::MultipleUnknownDimensions));
    }

    #[test]
    fn layout_nbytes_is_consistent() {
        let layout = NdLayout::contiguous(vec![5, 6], 4, MemoryOrder::C).expect("layout");
        assert_eq!(layout.nbytes().expect("nbytes"), 120);
        assert_eq!(element_count(&layout.shape).expect("elements"), 30);
    }

    #[test]
    fn broadcast_shape_is_commutative_for_sample_shapes() {
        let shapes: &[&[usize]] = &[
            &[],
            &[1],
            &[3],
            &[1, 4],
            &[2, 1, 4],
            &[2, 3, 4],
            &[1, 1, 5, 1],
        ];

        for lhs in shapes {
            for rhs in shapes {
                let ab = broadcast_shape(lhs, rhs);
                let ba = broadcast_shape(rhs, lhs);
                match (ab, ba) {
                    (Ok(left), Ok(right)) => assert_eq!(left, right),
                    (Err(_), Err(_)) => {}
                    (left, right) => assert_eq!(
                        left.is_ok(),
                        right.is_ok(),
                        "broadcast commutativity mismatch: {left:?} vs {right:?}"
                    ),
                }
            }
        }
    }

    #[test]
    fn element_count_scalar_shape_is_one() {
        assert_eq!(element_count(&[]).expect("scalar count"), 1);
    }

    #[test]
    fn element_count_overflow_is_rejected() {
        let err = element_count(&[usize::MAX, 2]).expect_err("overflow should fail");
        assert!(matches!(err, ShapeError::Overflow));
    }

    #[test]
    fn contiguous_strides_rejects_zero_item_size() {
        let err = contiguous_strides(&[2, 3], 0, MemoryOrder::C).expect_err("must reject");
        assert!(matches!(err, ShapeError::InvalidItemSize));
    }

    #[test]
    fn invalid_negative_dimension_is_rejected() {
        let err = fix_unknown_dimension(&[2, -2], 2).expect_err("negative dimension should fail");
        assert!(matches!(err, ShapeError::InvalidDimension(-2)));
    }

    #[test]
    fn zero_known_product_with_unknown_dimension_resolves_to_zero() {
        // NumPy: np.zeros(0).reshape(0, -1) → shape (0, 0)
        // When known_product is 0 and element_count is 0, the unknown dim resolves to 0.
        let result = fix_unknown_dimension(&[0, -1], 0).expect("zero reshape is valid");
        assert_eq!(result, vec![0, 0]);
    }

    #[test]
    fn layout_nbytes_overflow_is_rejected() {
        let layout = NdLayout {
            shape: vec![usize::MAX],
            strides: vec![1],
            item_size: 2,
        };
        let err = layout.nbytes().expect_err("nbytes overflow should fail");
        assert!(matches!(err, ShapeError::Overflow));
    }

    #[test]
    fn broadcast_strides_zeroes_broadcast_axes() {
        let base = NdLayout::contiguous(vec![2, 1, 3], 8, MemoryOrder::C).expect("layout");
        let out = broadcast_strides(&base.shape, &base.strides, &[2, 4, 3]).expect("strides");
        assert_eq!(out, vec![24, 0, 8]);
    }

    #[test]
    fn broadcast_strides_rejects_rank_mismatch() {
        let err = broadcast_strides(&[2, 3], &[24], &[2, 3])
            .expect_err("shape/stride rank mismatch should fail");
        assert!(matches!(
            err,
            ShapeError::RankMismatch {
                expected: 2,
                actual: 1
            }
        ));
    }

    #[test]
    fn broadcast_strides_rejects_incompatible_target() {
        let err = broadcast_strides(&[2, 3], &[24, 8], &[2, 2])
            .expect_err("incompatible target shape should fail");
        assert!(matches!(err, ShapeError::IncompatibleBroadcast { .. }));
    }

    #[test]
    fn broadcast_to_produces_zero_stride_view() {
        let base = NdLayout::contiguous(vec![2, 1, 3], 8, MemoryOrder::C).expect("layout");
        let out = base.broadcast_to(vec![2, 4, 3]).expect("broadcast_to");
        assert_eq!(out.shape, vec![2, 4, 3]);
        assert_eq!(out.strides, vec![24, 0, 8]);
    }

    #[test]
    fn broadcast_to_rejects_incompatible_target() {
        let base = NdLayout::contiguous(vec![2, 3], 8, MemoryOrder::C).expect("layout");
        let err = base
            .broadcast_to(vec![2, 2])
            .expect_err("incompatible broadcast should fail");
        assert!(matches!(err, ShapeError::IncompatibleBroadcast { .. }));
    }

    #[test]
    fn as_strided_accepts_in_bounds_view() {
        let base = NdLayout::contiguous(vec![4, 4], 8, MemoryOrder::C).expect("layout");
        let view = base
            .as_strided(vec![2, 2], vec![16, 8])
            .expect("valid strided view");
        assert_eq!(view.shape, vec![2, 2]);
        assert_eq!(view.strides, vec![16, 8]);
    }

    #[test]
    fn as_strided_allows_zero_extent_view() {
        let base = NdLayout::contiguous(vec![4], 8, MemoryOrder::C).expect("layout");
        let view = base
            .as_strided(vec![0], vec![8])
            .expect("zero-extent views should be in-bounds");
        assert_eq!(view.shape, vec![0]);
        assert_eq!(view.strides, vec![8]);
    }

    #[test]
    fn as_strided_rejects_out_of_bounds_view() {
        let base = NdLayout::contiguous(vec![4, 4], 8, MemoryOrder::C).expect("layout");
        let err = base
            .as_strided(vec![4, 4], vec![64, 8])
            .expect_err("view must fit in base storage");
        assert!(matches!(err, ShapeError::OutOfBoundsView { .. }));
    }

    #[test]
    fn as_strided_negative_stride_reverse() {
        // A 4-element array with negative stride simulates arr[::-1]
        let base = NdLayout::contiguous(vec![4], 8, MemoryOrder::C).expect("layout");
        let view = base
            .as_strided(vec![4], vec![-8])
            .expect("negative strides should be supported for reverse slicing");
        assert_eq!(view.shape, vec![4]);
        assert_eq!(view.strides, vec![-8]);
    }

    #[test]
    fn as_strided_negative_stride_2d() {
        // 3x3 array, reverse rows only
        let base = NdLayout::contiguous(vec![3, 3], 8, MemoryOrder::C).expect("layout");
        let view = base
            .as_strided(vec![3, 3], vec![-24, 8])
            .expect("negative row stride should be supported");
        assert_eq!(view.strides, vec![-24, 8]);
    }

    #[test]
    fn as_strided_negative_stride_oob_rejected() {
        // Negative stride creating a span larger than base → should be rejected
        let base = NdLayout::contiguous(vec![2], 8, MemoryOrder::C).expect("layout");
        let err = base
            .as_strided(vec![4], vec![-8])
            .expect_err("span exceeds base storage");
        assert!(matches!(err, ShapeError::OutOfBoundsView { .. }));
    }

    #[test]
    fn as_strided_rejects_rank_mismatch() {
        let base = NdLayout::contiguous(vec![4], 8, MemoryOrder::C).expect("layout");
        let err = base
            .as_strided(vec![2, 2], vec![8])
            .expect_err("shape/stride rank mismatch should fail");
        assert!(matches!(
            err,
            ShapeError::RankMismatch {
                expected: 2,
                actual: 1
            }
        ));
    }

    #[test]
    fn sliding_window_view_builds_expected_shape_and_strides() {
        let base = NdLayout::contiguous(vec![4, 5], 8, MemoryOrder::C).expect("layout");
        let view = base
            .sliding_window_view(vec![2, 3])
            .expect("valid sliding window");
        assert_eq!(view.shape, vec![3, 3, 2, 3]);
        assert_eq!(view.strides, vec![40, 8, 40, 8]);
    }

    #[test]
    fn sliding_window_view_rejects_invalid_window_size() {
        let base = NdLayout::contiguous(vec![4, 5], 8, MemoryOrder::C).expect("layout");
        let err = base
            .sliding_window_view(vec![0, 3])
            .expect_err("window size zero should fail");
        assert!(matches!(
            err,
            ShapeError::InvalidWindowDimension {
                axis: 0,
                window: 0,
                dim: 4
            }
        ));
    }

    #[test]
    fn sliding_window_view_rejects_rank_mismatch() {
        let base = NdLayout::contiguous(vec![4, 5], 8, MemoryOrder::C).expect("layout");
        let err = base
            .sliding_window_view(vec![2])
            .expect_err("window rank must match input rank");
        assert!(matches!(
            err,
            ShapeError::RankMismatch {
                expected: 2,
                actual: 1
            }
        ));
    }

    #[test]
    fn broadcast_to_zero_stride_property_grid() {
        let cases: &[(&[usize], &[usize])] = &[
            (&[2, 1, 3], &[2, 4, 3]),
            (&[1, 5], &[7, 5]),
            (&[3, 1, 1], &[3, 2, 4]),
            (&[5], &[2, 5]),
        ];

        for &(src_shape, dst_shape) in cases {
            let base =
                NdLayout::contiguous(src_shape.to_vec(), 8, MemoryOrder::C).expect("base layout");
            let out = base
                .broadcast_to(dst_shape.to_vec())
                .expect("broadcast_to should succeed");
            assert_eq!(out.item_size, base.item_size);
            assert_eq!(out.shape, dst_shape);

            let offset = dst_shape.len() - src_shape.len();
            for axis in 0..dst_shape.len() {
                let expected_stride = if axis < offset {
                    0
                } else {
                    let src_axis = axis - offset;
                    let src_dim = src_shape[src_axis];
                    let src_stride = base.strides[src_axis];
                    if src_dim == dst_shape[axis] {
                        src_stride
                    } else {
                        0
                    }
                };

                assert_eq!(
                    out.strides[axis], expected_stride,
                    "stride mismatch src_shape={src_shape:?} dst_shape={dst_shape:?} axis={axis}"
                );
            }
        }
    }

    #[test]
    fn as_strided_zero_extent_property_grid() {
        let base = NdLayout::contiguous(vec![4, 4, 4], 8, MemoryOrder::C).expect("layout");
        let cases = [
            (vec![0], vec![8]),
            (vec![0, 3], vec![0, 8]),
            (vec![2, 0, 1], vec![64, 16, 8]),
        ];

        for (shape, strides) in cases {
            let view = base
                .as_strided(shape.clone(), strides.clone())
                .expect("zero-extent view should be accepted");
            assert_eq!(view.shape, shape);
            assert_eq!(view.strides, strides);
        }
    }

    #[test]
    fn sliding_window_shape_stride_property_grid() {
        let cases = [
            (vec![4, 5], vec![2, 3]),
            (vec![6, 3], vec![4, 1]),
            (vec![3, 3, 3], vec![1, 2, 3]),
        ];

        for (shape, window_shape) in cases {
            let base = NdLayout::contiguous(shape.clone(), 8, MemoryOrder::C).expect("layout");
            let view = base
                .sliding_window_view(window_shape.clone())
                .expect("window should be valid");

            let mut expected_shape: Vec<usize> = shape
                .iter()
                .zip(&window_shape)
                .map(|(&dim, &window)| dim - window + 1)
                .collect();
            expected_shape.extend(window_shape.iter().copied());
            assert_eq!(view.shape, expected_shape);

            let mut expected_strides = base.strides.clone();
            expected_strides.extend(base.strides.iter().copied());
            assert_eq!(view.strides, expected_strides);
        }
    }

    #[test]
    fn broadcast_shape_associativity_property_grid() {
        let shapes: &[&[usize]] = &[
            &[],
            &[1],
            &[3],
            &[1, 4],
            &[2, 1, 4],
            &[2, 3, 1],
            &[1, 1, 5, 1],
        ];

        for a in shapes {
            for b in shapes {
                for c in shapes {
                    let ab_c = broadcast_shape(a, b).and_then(|ab| broadcast_shape(&ab, c));
                    let a_bc = broadcast_shape(b, c).and_then(|bc| broadcast_shape(a, &bc));
                    match (ab_c, a_bc) {
                        (Ok(lhs), Ok(rhs)) => assert_eq!(lhs, rhs, "a={a:?} b={b:?} c={c:?}"),
                        (Err(_), Err(_)) => {}
                        (lhs, rhs) => assert_eq!(
                            lhs.is_ok(),
                            rhs.is_ok(),
                            "broadcast associativity mismatch a={a:?} b={b:?} c={c:?}: lhs={lhs:?} rhs={rhs:?}"
                        ),
                    }
                }
            }
        }
    }

    #[test]
    fn is_contiguous_c_order() {
        let layout = NdLayout::contiguous(vec![3, 4, 5], 8, MemoryOrder::C).expect("layout");
        assert!(layout.is_contiguous());
        assert!(!layout.is_fortran_contiguous());
    }

    #[test]
    fn is_contiguous_f_order() {
        let layout = NdLayout::contiguous(vec![3, 4, 5], 8, MemoryOrder::F).expect("layout");
        assert!(!layout.is_contiguous());
        assert!(layout.is_fortran_contiguous());
    }

    #[test]
    fn is_contiguous_scalar() {
        let layout = NdLayout::contiguous(vec![], 8, MemoryOrder::C).expect("layout");
        assert!(layout.is_contiguous());
        assert!(layout.is_fortran_contiguous());
    }

    #[test]
    fn is_contiguous_1d() {
        let layout = NdLayout::contiguous(vec![10], 8, MemoryOrder::C).expect("layout");
        // 1-D contiguous is both C and F contiguous
        assert!(layout.is_contiguous());
        assert!(layout.is_fortran_contiguous());
    }

    #[test]
    fn is_contiguous_zero_dim() {
        let layout = NdLayout::contiguous(vec![0, 5], 8, MemoryOrder::C).expect("layout");
        // Zero-length arrays are contiguous by convention
        assert!(layout.is_contiguous());
        assert!(layout.is_fortran_contiguous());
    }

    #[test]
    fn is_contiguous_broadcast_not_contiguous() {
        let base = NdLayout::contiguous(vec![1, 5], 8, MemoryOrder::C).expect("layout");
        let broadcast = base.broadcast_to(vec![3, 5]).expect("broadcast");
        // Broadcast arrays with 0-strides are not contiguous
        assert!(!broadcast.is_contiguous());
        assert!(!broadcast.is_fortran_contiguous());
    }

    #[test]
    fn is_contiguous_singleton_dims_ignored() {
        // Shape [1, 5, 1] with C strides should be both C and F contiguous
        // because dimensions of size 1 don't constrain strides
        let layout = NdLayout::contiguous(vec![1, 5, 1], 8, MemoryOrder::C).expect("layout");
        assert!(layout.is_contiguous());
        // Size-1 dimensions are unconstrained, so F order also holds
        assert!(layout.is_fortran_contiguous());
    }

    #[test]
    fn ndim_matches_shape_len() {
        let layout = NdLayout::contiguous(vec![3, 4, 5], 8, MemoryOrder::C).expect("layout");
        assert_eq!(layout.ndim(), 3);
        let scalar = NdLayout::contiguous(vec![], 8, MemoryOrder::C).expect("layout");
        assert_eq!(scalar.ndim(), 0);
    }

    #[test]
    fn as_strided_rank_mismatch_property_grid() {
        let base = NdLayout::contiguous(vec![4, 4], 8, MemoryOrder::C).expect("layout");
        let shapes = [vec![], vec![1], vec![1, 1], vec![1, 1, 1]];
        let strides = [vec![], vec![8], vec![8, 8], vec![8, 8, 8]];

        for shape in &shapes {
            for stride in &strides {
                if shape.len() == stride.len() {
                    continue;
                }
                let err = base
                    .as_strided(shape.clone(), stride.clone())
                    .expect_err("rank mismatch should fail");
                assert!(matches!(
                    err,
                    ShapeError::RankMismatch { expected, actual }
                        if expected == shape.len() && actual == stride.len()
                ));
            }
        }
    }
}
