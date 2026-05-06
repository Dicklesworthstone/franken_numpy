#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use fnp_io::IOSupportedDType;

/// Structure-aware input for binary file parsing
#[derive(Arbitrary, Debug)]
struct FromfileInput<'a> {
    data: &'a [u8],
    dtype_idx: u8,
    count: Option<u16>,
}

impl FromfileInput<'_> {
    fn dtype(&self) -> IOSupportedDType {
        match self.dtype_idx % 10 {
            0 => IOSupportedDType::F64,
            1 => IOSupportedDType::F32,
            2 => IOSupportedDType::I64,
            3 => IOSupportedDType::I32,
            4 => IOSupportedDType::I16,
            5 => IOSupportedDType::I8,
            6 => IOSupportedDType::U64,
            7 => IOSupportedDType::U32,
            8 => IOSupportedDType::U16,
            9 => IOSupportedDType::U8,
            _ => IOSupportedDType::F64,
        }
    }
}

fuzz_target!(|input: FromfileInput| {
    if input.data.len() > 1_000_000 {
        return;
    }

    let dtype = input.dtype();
    let count = input.count.map(|c| c as usize);

    // Fuzz fromfile with various dtypes and counts
    let _ = fnp_io::fromfile(input.data, dtype, count);

    // Fuzz complex binary parsing
    let _ = fnp_io::fromfile_complex(input.data, IOSupportedDType::Complex128, count);
    let _ = fnp_io::fromfile_complex(input.data, IOSupportedDType::Complex64, count);

    // Test round-trip invariant: tobytes(fromfile(data)) should not panic
    if let Ok(values) = fnp_io::fromfile(input.data, dtype, count) {
        if !values.is_empty() {
            let _ = fnp_io::tobytes(&values, dtype);
            let _ = fnp_io::tostring(&values, dtype);
        }
    }
});
