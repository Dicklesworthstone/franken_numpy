#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct BroadcastInput {
    shapes: Vec<Vec<u16>>,
}

fuzz_target!(|input: BroadcastInput| {
    if input.shapes.len() > 8 || input.shapes.iter().any(|s| s.len() > 8) {
        return;
    }

    let shapes: Vec<Vec<usize>> = input
        .shapes
        .iter()
        .map(|s| s.iter().map(|&d| d as usize).collect())
        .collect();

    if shapes.len() == 2 {
        let _ = fnp_ndarray::broadcast_shape(&shapes[0], &shapes[1]);
    }

    if !shapes.is_empty() {
        let shape_refs: Vec<&[usize]> = shapes.iter().map(|s| s.as_slice()).collect();
        let _ = fnp_ndarray::broadcast_shapes(&shape_refs);
    }

    for shape in &shapes {
        let _ = fnp_ndarray::element_count(shape);
    }
});
