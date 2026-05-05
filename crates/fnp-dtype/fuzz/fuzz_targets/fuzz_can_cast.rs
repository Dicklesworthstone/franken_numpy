#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use fnp_dtype::DType;

#[derive(Debug, Arbitrary)]
struct DTypeIndex(u8);

impl DTypeIndex {
    fn to_dtype(&self) -> DType {
        match self.0 % 17 {
            0 => DType::Bool,
            1 => DType::I8,
            2 => DType::I16,
            3 => DType::I32,
            4 => DType::I64,
            5 => DType::U8,
            6 => DType::U16,
            7 => DType::U32,
            8 => DType::U64,
            9 => DType::F16,
            10 => DType::F32,
            11 => DType::F64,
            12 => DType::Complex64,
            13 => DType::Complex128,
            14 => DType::Str,
            15 => DType::DateTime64,
            _ => DType::TimeDelta64,
        }
    }
}

#[derive(Debug, Arbitrary)]
struct CanCastInput {
    from: DTypeIndex,
    to: DTypeIndex,
    casting_mode: u8,
}

fuzz_target!(|input: CanCastInput| {
    let from = input.from.to_dtype();
    let to = input.to.to_dtype();
    let casting = match input.casting_mode % 5 {
        0 => "no",
        1 => "equiv",
        2 => "safe",
        3 => "same_kind",
        _ => "unsafe",
    };
    let _ = fnp_dtype::can_cast(from, to, casting);
});
