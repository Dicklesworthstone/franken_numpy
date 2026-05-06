#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use fnp_ndarray::{NdLayout, MemoryOrder};

#[derive(Arbitrary, Debug)]
struct StridedInput {
    base_shape: Vec<u16>,
    new_shape: Vec<u16>,
    new_strides: Vec<i32>,
    item_size: u8,
    fortran_order: bool,
}

fuzz_target!(|input: StridedInput| {
    if input.base_shape.len() > 6 || input.new_shape.len() > 6 {
        return;
    }
    if input.new_shape.len() != input.new_strides.len() {
        return;
    }

    let item_size = (input.item_size as usize).max(1).min(16);
    let base_shape: Vec<usize> = input.base_shape.iter().map(|&d| d as usize).collect();
    let order = if input.fortran_order {
        MemoryOrder::F
    } else {
        MemoryOrder::C
    };

    let Ok(layout) = NdLayout::contiguous(base_shape, item_size, order) else {
        return;
    };

    let new_shape: Vec<usize> = input.new_shape.iter().map(|&d| d as usize).collect();
    let new_strides: Vec<isize> = input.new_strides.iter().map(|&s| s as isize).collect();

    let result = layout.as_strided(new_shape, new_strides);

    if let Ok(view) = result {
        let _ = view.is_contiguous();
        let _ = view.is_fortran_contiguous();
        let _ = view.has_internal_overlap();
        let _ = view.nbytes();
    }
});
