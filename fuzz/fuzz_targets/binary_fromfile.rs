#![no_main]

use fnp_io::{fromfile, tofile, IOSupportedDType};
use libfuzzer_sys::fuzz_target;

const MAX_FUZZ_INPUT_BYTES: usize = 1 << 16;

const DTYPES: [IOSupportedDType; 16] = [
    IOSupportedDType::F64,
    IOSupportedDType::F64Be,
    IOSupportedDType::F32,
    IOSupportedDType::F32Be,
    IOSupportedDType::I64,
    IOSupportedDType::I64Be,
    IOSupportedDType::I32,
    IOSupportedDType::I32Be,
    IOSupportedDType::I16,
    IOSupportedDType::I16Be,
    IOSupportedDType::I8,
    IOSupportedDType::U64,
    IOSupportedDType::U64Be,
    IOSupportedDType::U32,
    IOSupportedDType::U32Be,
    IOSupportedDType::U8,
];

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_FUZZ_INPUT_BYTES {
        return;
    }

    for dtype in DTYPES {
        let _ = fromfile(data, dtype, None);
        let _ = fromfile(data, dtype, Some(1));
        let _ = fromfile(data, dtype, Some(10));
        let _ = fromfile(data, dtype, Some(100));

        if let Ok(values) = fromfile(data, dtype, None) {
            if !values.is_empty() && values.iter().all(|v| v.is_finite()) {
                if let Ok(encoded) = tofile(&values, dtype) {
                    let decoded = fromfile(&encoded, dtype, None);
                    assert!(
                        decoded.is_ok(),
                        "round-trip encode/decode failed for {:?}",
                        dtype
                    );
                }
            }
        }
    }

    if data.len() >= 2 {
        let dtype_idx = (data[0] as usize) % DTYPES.len();
        let dtype = DTYPES[dtype_idx];
        let count = data[1] as usize;
        let _ = fromfile(&data[2..], dtype, Some(count));
    }
});
