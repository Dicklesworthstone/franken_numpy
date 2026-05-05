#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() > 1000 {
        return;
    }
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = fnp_dtype::DType::parse(s);
    }
});
