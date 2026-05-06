#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use fnp_io::IOSupportedDType;

/// Structure-aware input for header validation fuzzing
#[derive(Arbitrary, Debug)]
struct HeaderInput<'a> {
    header_bytes: &'a [u8],
    dtype_idx: u8,
    fortran_order: bool,
    shape_dims: Vec<u16>,
    descr: &'a str,
    header_len: u16,
}

impl HeaderInput<'_> {
    fn dtype(&self) -> IOSupportedDType {
        match self.dtype_idx % 12 {
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
            10 => IOSupportedDType::Complex128,
            11 => IOSupportedDType::Complex64,
            _ => IOSupportedDType::F64,
        }
    }

    fn shape(&self) -> Vec<usize> {
        self.shape_dims
            .iter()
            .take(8) // Max 8 dimensions
            .map(|&d| (d as usize).saturating_add(1).min(1000))
            .collect()
    }
}

fuzz_target!(|input: HeaderInput| {
    if input.header_bytes.len() > 100_000 {
        return;
    }

    let dtype = input.dtype();
    let shape = input.shape();

    // Fuzz header schema validation: (shape, fortran_order, descr, header_len)
    let _ = fnp_io::validate_header_schema(
        &shape,
        input.fortran_order,
        input.descr,
        input.header_len as usize,
    );

    // Fuzz descriptor roundtrip
    let _ = fnp_io::validate_descriptor_roundtrip(dtype);

    // Fuzz magic/version validation
    let _ = fnp_io::validate_magic_version(input.header_bytes);

    // Fuzz load dispatch classification
    let _ = fnp_io::classify_load_dispatch(input.header_bytes, false);
    let _ = fnp_io::classify_load_dispatch(input.header_bytes, true);

    // Fuzz pickle policy enforcement
    let _ = fnp_io::enforce_pickle_policy(dtype, false);
    let _ = fnp_io::enforce_pickle_policy(dtype, true);

    // Fuzz IO policy metadata
    if let Ok(mode) = std::str::from_utf8(&input.header_bytes[..input.header_bytes.len().min(10)]) {
        let _ = fnp_io::validate_io_policy_metadata(mode, "array");
        let _ = fnp_io::validate_io_policy_metadata(mode, "file");
    }
});
