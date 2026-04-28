#![no_main]

use fnp_ufunc::{parse_fixed_signature_string, parse_gufunc_signature};
use libfuzzer_sys::fuzz_target;

const MAX_FUZZ_INPUT_BYTES: usize = 1 << 12;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_FUZZ_INPUT_BYTES {
        return;
    }

    if let Ok(text) = std::str::from_utf8(data) {
        let _ = parse_gufunc_signature(Some(text), None);
        let _ = parse_gufunc_signature(None, Some(text));
        let _ = parse_gufunc_signature(Some(text), Some(text));

        for nin in [1, 2, 3] {
            for nout in [1, 2] {
                let _ = parse_fixed_signature_string(text, nin, nout);
            }
        }
    }
});
