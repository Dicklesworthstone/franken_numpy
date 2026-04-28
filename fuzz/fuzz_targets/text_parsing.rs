#![no_main]

use fnp_io::{IOSupportedDType, fromfile_text, fromstring};
use libfuzzer_sys::fuzz_target;

const MAX_FUZZ_INPUT_BYTES: usize = 1 << 16;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_FUZZ_INPUT_BYTES {
        return;
    }

    // Fuzz fromfile_text with various separators
    if let Ok(text) = std::str::from_utf8(data) {
        for sep in [" ", ",", "\t", ";", "\n", "  ", ", "] {
            let _ = fromfile_text(text, sep, None);
            let _ = fromfile_text(text, sep, Some(10));
            let _ = fromfile_text(text, sep, Some(100));
        }
    }

    // Fuzz fromstring with various dtypes and separators
    for dtype in [
        IOSupportedDType::F64,
        IOSupportedDType::F32,
        IOSupportedDType::I64,
        IOSupportedDType::I32,
        IOSupportedDType::U64,
        IOSupportedDType::U8,
    ] {
        for sep in ["", " ", ","] {
            let _ = fromstring(data, dtype, sep);
        }
    }
});
