#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use fnp_ndarray::{NdLayout, MemoryOrder};

#[derive(Arbitrary, Debug)]
struct SlidingWindowInput {
    base_shape: Vec<u16>,
    window_shape: Vec<u16>,
    item_size: u8,
}

fuzz_target!(|input: SlidingWindowInput| {
    if input.base_shape.len() > 6 || input.window_shape.len() > 6 {
        return;
    }

    let item_size = (input.item_size as usize).max(1).min(16);
    let base_shape: Vec<usize> = input.base_shape.iter().map(|&d| d as usize).collect();
    let window_shape: Vec<usize> = input.window_shape.iter().map(|&d| d as usize).collect();

    let Ok(layout) = NdLayout::contiguous(base_shape, item_size, MemoryOrder::C) else {
        return;
    };

    let result = layout.sliding_window_view(&window_shape);

    if let Ok(view) = result {
        let _ = view.ndim();
        let _ = view.is_contiguous();
        let _ = view.nbytes();
    }
});
