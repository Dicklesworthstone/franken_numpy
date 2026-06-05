#![no_main]

use fnp_ufunc::UFuncArray;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &str| {
    // Test einsum subscript parsing with dummy operands
    // The parser should handle any string input without panicking
    let dummy = UFuncArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let dummy_2d = UFuncArray::from_2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

    // Try with various operand counts
    let _ = UFuncArray::einsum(data, &[&dummy]);
    let _ = UFuncArray::einsum(data, &[&dummy, &dummy]);
    let _ = UFuncArray::einsum(data, &[&dummy_2d]);
    let _ = UFuncArray::einsum(data, &[&dummy_2d, &dummy_2d]);
});
