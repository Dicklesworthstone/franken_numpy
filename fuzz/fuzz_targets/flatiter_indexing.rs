#![no_main]

use fnp_iter::{
    overlap_copy_policy, resolve_flatiter_indices, select_transfer_class,
    validate_flatiter_read, validate_flatiter_write, validate_nditer_flags,
    FlatIterIndex, NditerTransferFlags, TransferSelectorInput,
};
use libfuzzer_sys::fuzz_target;

const MAX_FUZZ_INPUT_BYTES: usize = 512;
const MAX_ARRAY_LEN: usize = 10_000;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_FUZZ_INPUT_BYTES || data.is_empty() {
        return;
    }

    let mut idx = 0;

    // Parse array length from input
    let array_len = parse_u16(data, &mut idx) as usize % MAX_ARRAY_LEN;

    // Fuzz FlatIterIndex::Single
    if idx < data.len() {
        let single_idx = parse_u16(data, &mut idx) as usize;
        let index = FlatIterIndex::Single(single_idx);
        let _ = resolve_flatiter_indices(array_len, &index);
        let _ = validate_flatiter_read(array_len, &index);
        let _ = validate_flatiter_write(array_len, &index, 1);
    }

    // Fuzz FlatIterIndex::Slice
    if idx + 3 < data.len() {
        let start = parse_u16(data, &mut idx) as usize % (array_len.saturating_add(1));
        let stop = parse_u16(data, &mut idx) as usize % (array_len.saturating_add(1));
        let step = (data[idx] as usize).max(1) % 100;
        idx += 1;

        let index = FlatIterIndex::Slice { start, stop, step };
        let _ = resolve_flatiter_indices(array_len, &index);
        let _ = validate_flatiter_read(array_len, &index);

        // Try various values_len for write validation
        let _ = validate_flatiter_write(array_len, &index, 1);
        if let Ok(count) = validate_flatiter_read(array_len, &index) {
            let _ = validate_flatiter_write(array_len, &index, count);
        }
    }

    // Fuzz FlatIterIndex::Fancy
    if idx < data.len() {
        let n_indices = (data[idx] as usize) % 32;
        idx += 1;
        let mut indices = Vec::with_capacity(n_indices);
        for _ in 0..n_indices {
            if idx >= data.len() {
                break;
            }
            indices.push(parse_u16(data, &mut idx) as usize);
        }
        let index = FlatIterIndex::Fancy(indices);
        let _ = resolve_flatiter_indices(array_len, &index);
        let _ = validate_flatiter_read(array_len, &index);
    }

    // Fuzz FlatIterIndex::BoolMask
    if idx < data.len() {
        let mask_len = (data[idx] as usize) % (array_len.saturating_add(1));
        idx += 1;
        let mut mask = Vec::with_capacity(mask_len);
        for i in 0..mask_len {
            if idx < data.len() {
                mask.push(data[idx] % 2 == 1);
                idx += 1;
            } else {
                mask.push(i % 2 == 0);
            }
        }
        let index = FlatIterIndex::BoolMask(mask);
        let _ = resolve_flatiter_indices(array_len, &index);
        let _ = validate_flatiter_read(array_len, &index);
    }

    // Fuzz overlap_copy_policy
    if idx + 6 <= data.len() {
        let src_offset = parse_u16(data, &mut idx) as usize;
        let dst_offset = parse_u16(data, &mut idx) as usize;
        let byte_len = parse_u16(data, &mut idx) as usize;
        let _ = overlap_copy_policy(src_offset, dst_offset, byte_len);
    }

    // Fuzz validate_nditer_flags
    if idx + 1 <= data.len() {
        let flags_byte = data[idx];
        idx += 1;
        let flags = NditerTransferFlags {
            copy_if_overlap: flags_byte & 0x01 != 0,
            no_broadcast: flags_byte & 0x02 != 0,
            observed_overlap: flags_byte & 0x04 != 0,
            observed_broadcast: flags_byte & 0x08 != 0,
        };
        let _ = validate_nditer_flags(flags);
    }

    // Fuzz select_transfer_class
    if idx + 8 <= data.len() {
        let src_stride = parse_i16(data, &mut idx) as isize;
        let dst_stride = parse_i16(data, &mut idx) as isize;
        let item_size = (parse_u16(data, &mut idx) as usize).saturating_add(1);
        let element_count = (parse_u16(data, &mut idx) as usize).saturating_add(1);

        let input = TransferSelectorInput {
            src_stride,
            dst_stride,
            item_size,
            element_count,
            aligned: data.get(idx).map(|b| b % 2 == 0).unwrap_or(true),
            cast_is_lossless: data.get(idx + 1).map(|b| b % 2 == 0).unwrap_or(true),
            same_value_cast: data.get(idx + 2).map(|b| b % 2 == 0).unwrap_or(false),
        };
        let _ = select_transfer_class(input);
    }
});

fn parse_u16(data: &[u8], idx: &mut usize) -> u16 {
    if *idx + 2 > data.len() {
        if *idx < data.len() {
            let val = data[*idx] as u16;
            *idx += 1;
            return val;
        }
        return 0;
    }
    let val = u16::from_le_bytes([data[*idx], data[*idx + 1]]);
    *idx += 2;
    val
}

fn parse_i16(data: &[u8], idx: &mut usize) -> i16 {
    parse_u16(data, idx) as i16
}
