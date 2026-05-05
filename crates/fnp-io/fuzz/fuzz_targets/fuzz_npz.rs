#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() > 10_000_000 {
        return;
    }
    let _ = fnp_io::read_npz_bytes(data, false);
    let _ = fnp_io::load_npz(data, false);
});
