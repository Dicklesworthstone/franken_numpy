#![no_main]

use fnp_ndarray::{MemoryOrder, broadcast_shape, contiguous_strides, fix_unknown_dimension};
use libfuzzer_sys::fuzz_target;

const MAX_FUZZ_INPUT_BYTES: usize = 256;
const MAX_DIMS: usize = 8;
const MAX_DIM_SIZE: usize = 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_FUZZ_INPUT_BYTES || data.is_empty() {
        return;
    }

    let mut idx = 0;

    // Parse two shapes from fuzz input
    let shape1 = parse_shape(data, &mut idx);
    let shape2 = parse_shape(data, &mut idx);

    // Fuzz broadcast_shape
    let _ = broadcast_shape(&shape1, &shape2);

    // Also try reversed order (should be commutative)
    let _ = broadcast_shape(&shape2, &shape1);

    // Fuzz fix_unknown_dimension (reshape with -1)
    if let Some(element_count) = checked_element_count(&shape1)
        && element_count > 0
        && element_count < 1_000_000
    {
        // Try reshaping to same element count with -1 dimension
        let mut new_shape: Vec<isize> = shape1.iter().map(|&d| d as isize).collect();
        if !new_shape.is_empty() {
            new_shape[0] = -1; // Infer first dimension
            let _ = fix_unknown_dimension(&new_shape, element_count);
        }

        // Try various reshape patterns
        let _ = fix_unknown_dimension(&[-1], element_count);
        let _ = fix_unknown_dimension(&[element_count as isize], element_count);
        if element_count > 1 {
            let _ = fix_unknown_dimension(&[-1, 1], element_count);
            let _ = fix_unknown_dimension(&[1, -1], element_count);
        }
    }

    // Fuzz contiguous_strides
    if let Some(element_count) = checked_element_count(&shape1)
        && element_count > 0
        && element_count < 1_000_000
    {
        let _ = contiguous_strides(&shape1, 8, MemoryOrder::C); // f64 C-order
        let _ = contiguous_strides(&shape1, 8, MemoryOrder::F); // f64 F-order
        let _ = contiguous_strides(&shape1, 4, MemoryOrder::C); // f32 C-order
        let _ = contiguous_strides(&shape1, 1, MemoryOrder::C); // u8 C-order
    }
});

fn checked_element_count(shape: &[usize]) -> Option<usize> {
    if shape.is_empty() {
        return None;
    }
    shape
        .iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
}

fn parse_shape(data: &[u8], idx: &mut usize) -> Vec<usize> {
    if *idx >= data.len() {
        return vec![];
    }

    let ndim = (data[*idx] as usize % MAX_DIMS)
        .saturating_add(1)
        .min(MAX_DIMS);
    *idx += 1;

    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        if *idx >= data.len() {
            break;
        }
        // Allow dimension sizes 0, 1, or larger values
        let dim = match data[*idx] % 4 {
            0 => 0,
            1 => 1,
            2 => (data[*idx] as usize).min(MAX_DIM_SIZE),
            _ => {
                if *idx + 1 < data.len() {
                    let hi = data[*idx] as usize;
                    let lo = data[*idx + 1] as usize;
                    *idx += 1;
                    ((hi << 8) | lo).min(MAX_DIM_SIZE)
                } else {
                    data[*idx] as usize
                }
            }
        };
        *idx += 1;
        shape.push(dim);
    }
    shape
}
