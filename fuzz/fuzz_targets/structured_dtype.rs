#![no_main]

use fnp_io::{IOSupportedDType, parse_structured_descr};
use libfuzzer_sys::fuzz_target;

const MAX_FUZZ_INPUT_BYTES: usize = 1 << 12;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_FUZZ_INPUT_BYTES {
        return;
    }

    if let Ok(text) = std::str::from_utf8(data) {
        let _ = parse_structured_descr(text);

        let _ = IOSupportedDType::decode(text);

        for prefix in ["[", "[(", "[('x',"] {
            let combined = format!("{prefix}{text}");
            let _ = parse_structured_descr(&combined);
        }
    }
});
