#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ReshapeInput {
    shape: Vec<i32>,
    element_count: u32,
}

fuzz_target!(|input: ReshapeInput| {
    if input.shape.len() > 8 {
        return;
    }

    let shape: Vec<isize> = input.shape.iter().map(|&d| d as isize).collect();
    let count = input.element_count as usize;

    let _ = fnp_ndarray::fix_unknown_dimension(&shape, count);
});
