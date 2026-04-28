#![no_main]

use fnp_io::{read_npy_bytes, read_npz_bytes};
use libfuzzer_sys::fuzz_target;

const MAX_FUZZ_INPUT_BYTES: usize = 1 << 20;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_FUZZ_INPUT_BYTES {
        return;
    }

    let _ = read_npy_bytes(data, false);
    let _ = read_npz_bytes(data, false);
});
