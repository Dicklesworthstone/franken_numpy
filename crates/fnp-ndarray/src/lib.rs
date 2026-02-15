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
    IncompatibleBroadcast { lhs: Vec<usize>, rhs: Vec<usize> },
    IncompatibleElementCount { old: usize, new: usize },
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidItemSize => write!(f, "item size must be > 0"),
            Self::InvalidDimension(dim) => write!(f, "invalid dimension {dim}"),
            Self::MultipleUnknownDimensions => write!(f, "only one -1 dimension is allowed"),
            Self::Overflow => write!(f, "size arithmetic overflow"),
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
            if known_product == 0 || !old_element_count.is_multiple_of(known_product) {
                return Err(ShapeError::IncompatibleElementCount {
                    old: old_element_count,
                    new: known_product,
                });
            }
            out[idx] = old_element_count / known_product;
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
}

#[cfg(test)]
mod tests {
    use super::{
        MemoryOrder, NdLayout, ShapeError, broadcast_shape, broadcast_shapes, can_broadcast,
        contiguous_strides, element_count, fix_unknown_dimension,
    };

    #[test]
    fn broadcast_shape_matches_numpy_style() {
        let out = broadcast_shape(&[8, 1, 6, 1], &[7, 1, 5]).expect("broadcast should succeed");
        assert_eq!(out, vec![8, 7, 6, 5]);
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
    fn zero_known_product_with_unknown_dimension_is_rejected() {
        let err = fix_unknown_dimension(&[0, -1], 0).expect_err("zero known product is invalid");
        assert!(matches!(
            err,
            ShapeError::IncompatibleElementCount { old: 0, new: 0 }
        ));
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
}
