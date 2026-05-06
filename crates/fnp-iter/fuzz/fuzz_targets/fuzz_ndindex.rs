#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct NdindexInput {
    shape: Vec<u8>,
}

fuzz_target!(|input: NdindexInput| {
    let shape: Vec<usize> = input.shape.iter().map(|&x| x as usize).collect();

    if shape.len() > 8 {
        return;
    }
    let total: usize = shape.iter().product();
    if total > 100_000 {
        return;
    }

    let result = fnp_iter::ndindex(&shape);

    if let Ok(indices) = result {
        if total > 0 {
            assert_eq!(indices.len(), total, "ndindex should produce shape.product() indices");
        }

        for idx in &indices {
            assert_eq!(idx.len(), shape.len(), "each index should have shape.len() dimensions");
            for (i, &val) in idx.iter().enumerate() {
                assert!(val < shape[i], "index component must be < shape dimension");
            }
        }

        if !indices.is_empty() {
            let first = &indices[0];
            assert!(first.iter().all(|&x| x == 0), "first index should be all zeros");
        }
    }
});
