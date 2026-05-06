#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use fnp_iter::{select_transfer_class, TransferSelectorInput, TransferClass};

#[derive(Debug, Arbitrary)]
struct TransferInput {
    src_stride: i16,
    dst_stride: i16,
    item_size: u16,
    element_count: u16,
    aligned: bool,
    cast_is_lossless: bool,
    same_value_cast: bool,
}

fuzz_target!(|input: TransferInput| {
    let selector = TransferSelectorInput {
        src_stride: input.src_stride as isize,
        dst_stride: input.dst_stride as isize,
        item_size: input.item_size as usize,
        element_count: input.element_count as usize,
        aligned: input.aligned,
        cast_is_lossless: input.cast_is_lossless,
        same_value_cast: input.same_value_cast,
    };

    let result = select_transfer_class(selector.clone());

    if let Ok(class) = result {
        if selector.item_size > 0
            && selector.src_stride == selector.item_size as isize
            && selector.dst_stride == selector.item_size as isize
            && selector.aligned
            && selector.cast_is_lossless
            && !selector.same_value_cast
        {
            assert_eq!(
                class, TransferClass::Contiguous,
                "contiguous conditions should yield Contiguous class"
            );
        }

        if selector.item_size == 0 {
            panic!("item_size=0 should have returned an error");
        }
    } else {
        if selector.item_size == 0 {
            return;
        }
    }
});
