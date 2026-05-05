#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use fnp_dtype::DType;

#[derive(Debug, Arbitrary)]
struct DTypeInput {
    index: u8,
}

impl DTypeInput {
    fn to_dtype(&self) -> DType {
        match self.index % 17 {
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
struct ResultTypeInput {
    dtypes: Vec<DTypeInput>,
}

fuzz_target!(|input: ResultTypeInput| {
    if input.dtypes.is_empty() || input.dtypes.len() > 10 {
        return;
    }
    let dtypes: Vec<DType> = input.dtypes.iter().map(|d| d.to_dtype()).collect();
    let _ = fnp_dtype::result_type(&dtypes);
});
