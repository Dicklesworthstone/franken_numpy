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
        let a = UFuncArray::new(vec![2, 3], vec![1.0; 6], DType::F64).unwrap();
        let b = UFuncArray::new(vec![3, 4], vec![1.0; 12], DType::F64).unwrap();
        let c = UFuncArray::new(vec![4], vec![1.0; 4], DType::F64).unwrap();

        let _ = UFuncArray::einsum(subscripts, &[&a]);
        let _ = UFuncArray::einsum(subscripts, &[&a, &b]);
        let _ = UFuncArray::einsum(subscripts, &[&a, &b, &c]);

        let scalar = UFuncArray::scalar(1.0, DType::F64);
        let _ = UFuncArray::einsum(subscripts, &[&scalar]);
        let _ = UFuncArray::einsum(subscripts, &[&scalar, &scalar]);
    }
});
