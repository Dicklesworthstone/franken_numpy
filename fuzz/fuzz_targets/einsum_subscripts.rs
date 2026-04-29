#![no_main]

use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;
use libfuzzer_sys::fuzz_target;

const MAX_FUZZ_INPUT_BYTES: usize = 1 << 10;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_FUZZ_INPUT_BYTES {
        return;
    }

    if let Ok(subscripts) = std::str::from_utf8(data) {
        let Ok(a) = UFuncArray::new(vec![2, 3], vec![1.0; 6], DType::F64) else {
            return;
        };
        let Ok(b) = UFuncArray::new(vec![3, 4], vec![1.0; 12], DType::F64) else {
            return;
        };
        let Ok(c) = UFuncArray::new(vec![4, 2], vec![1.0; 8], DType::F64) else {
            return;
        };
        let Ok(batch_a) = UFuncArray::new(vec![2, 2, 3], vec![1.0; 12], DType::F64) else {
            return;
        };
        let Ok(batch_b) = UFuncArray::new(vec![2, 3, 4], vec![1.0; 24], DType::F64) else {
            return;
        };

        let _ = UFuncArray::einsum(subscripts, &[&a]);
        let _ = UFuncArray::einsum(subscripts, &[&a, &b]);
        let _ = UFuncArray::einsum(subscripts, &[&a, &b, &c]);
        let _ = UFuncArray::einsum(subscripts, &[&batch_a]);
        let _ = UFuncArray::einsum(subscripts, &[&batch_a, &batch_b]);
        let _ = UFuncArray::einsum_path(subscripts, &[&a, &b]);
        let _ = UFuncArray::einsum_path(subscripts, &[&a, &b, &c]);
        let _ = UFuncArray::einsum_path(subscripts, &[&batch_a, &batch_b]);
        let _ = UFuncArray::einsum_optimized(subscripts, &[&a, &b, &c], "greedy");
        let _ = UFuncArray::einsum_optimized(subscripts, &[&a, &b, &c], "optimal");
        let _ = UFuncArray::einsum_optimized(subscripts, &[&batch_a, &batch_b], "greedy");

        let scalar = UFuncArray::scalar(1.0, DType::F64);
        let _ = UFuncArray::einsum(subscripts, &[&scalar]);
        let _ = UFuncArray::einsum(subscripts, &[&scalar, &scalar]);
        let _ = UFuncArray::einsum_path(subscripts, &[&scalar, &scalar]);
    }
});
