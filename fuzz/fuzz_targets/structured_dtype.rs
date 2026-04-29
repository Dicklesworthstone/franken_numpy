#![no_main]

use fnp_io::{IOSupportedDType, parse_structured_descr};
use libfuzzer_sys::fuzz_target;

const MAX_FUZZ_INPUT_BYTES: usize = 1 << 12;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_FUZZ_INPUT_BYTES {
        return;
    }

    if let Ok(text) = std::str::from_utf8(data) {
        if let Ok(descriptor) = parse_structured_descr(text) {
            let _ = descriptor.record_size();
            let _ = descriptor.field_offsets();

            let serialized = descriptor.to_descr_string();
            let reparsed = parse_structured_descr(&serialized);
            assert!(
                reparsed.is_ok(),
                "structured dtype descriptor did not serialize back to a parseable descriptor: {serialized}"
            );
            if let Ok(reparsed) = reparsed {
                assert_eq!(
                    reparsed, descriptor,
                    "structured dtype descriptor changed after serialization: {serialized}"
                );
            }
        }

        let _ = IOSupportedDType::decode(text);

        for prefix in ["[", "[(", "[('x',"] {
            let combined = format!("{prefix}{text}");
            let _ = parse_structured_descr(&combined);
        }
    }
});
