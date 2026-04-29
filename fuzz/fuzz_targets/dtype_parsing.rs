#![no_main]

use fnp_dtype::{DType, can_cast_lossless, common_type, finfo, iinfo, promote, result_type};
use libfuzzer_sys::fuzz_target;

const DTYPES: [DType; 18] = [
    DType::Bool,
    DType::I8,
    DType::I16,
    DType::I32,
    DType::I64,
    DType::U8,
    DType::U16,
    DType::U32,
    DType::U64,
    DType::F16,
    DType::F32,
    DType::F64,
    DType::Complex64,
    DType::Complex128,
    DType::Str,
    DType::DateTime64,
    DType::TimeDelta64,
    DType::Structured,
];

fn exercise_dtype(dtype: DType) {
    assert_eq!(DType::parse(dtype.name()), Some(dtype));
    assert!(dtype.item_size() > 0);
    let _ = iinfo(dtype);
    let _ = finfo(dtype);

    for other in DTYPES {
        assert_eq!(promote(dtype, other), promote(other, dtype));
        let _ = can_cast_lossless(dtype, other);
        let _ = result_type(&[dtype, other]);
        let _ = common_type(&[dtype, other]);
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() > 512 {
        return;
    }

    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };

    let mut parsed = Vec::new();
    for candidate in [text, text.trim(), text.trim_matches('\0')] {
        if let Some(dtype) = DType::parse(candidate) {
            exercise_dtype(dtype);
            parsed.push(dtype);
        }
    }

    for token in text.split(|c: char| c == ',' || c == ';' || c.is_whitespace()) {
        if let Some(dtype) = DType::parse(token) {
            exercise_dtype(dtype);
            parsed.push(dtype);
        }
    }

    if !parsed.is_empty() {
        let _ = result_type(&parsed);
        let _ = common_type(&parsed);
    }
});
