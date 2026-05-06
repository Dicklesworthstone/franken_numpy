#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use fnp_iter::{NditerPlan, NditerOptions, NditerOrder};

#[derive(Debug, Arbitrary)]
struct NditerPlanInput {
    shape: Vec<u8>,
    item_size: u8,
    order_is_c: bool,
    external_loop: bool,
}

fuzz_target!(|input: NditerPlanInput| {
    let shape: Vec<usize> = input.shape.iter().map(|&x| x as usize).collect();
    let item_size = (input.item_size as usize).max(1);

    if shape.len() > 8 {
        return;
    }
    let total: usize = shape.iter().product();
    if total > 1_000_000 {
        return;
    }

    let options = NditerOptions {
        order: if input.order_is_c { NditerOrder::C } else { NditerOrder::F },
        external_loop: input.external_loop,
    };

    let result = NditerPlan::new(shape.clone(), item_size, options);

    if let Ok(plan) = result {
        assert_eq!(plan.shape(), &shape, "plan shape should match input shape");
        assert_eq!(plan.element_count(), total, "element_count should match product of shape");

        let iteration_shape = plan.iteration_shape();
        let iter_total: usize = iteration_shape.iter().product();

        if input.external_loop && !shape.is_empty() {
            assert!(
                iter_total <= total,
                "external_loop iteration shape product should be <= element_count"
            );
        } else {
            assert_eq!(
                iter_total, total,
                "non-external-loop iteration shape product should equal element_count"
            );
        }

        for i in 0..total.min(100) {
            if let Ok(multi_idx) = plan.linear_index_to_multi_index(i) {
                assert_eq!(multi_idx.len(), shape.len());
                for (dim, &idx) in multi_idx.iter().enumerate() {
                    assert!(idx < shape[dim], "multi_index component out of bounds");
                }

                if let Ok(linear) = plan.multi_index_to_linear_index(&multi_idx) {
                    assert_eq!(linear, i, "round-trip linear->multi->linear should be identity");
                }
            }
        }
    }
});
