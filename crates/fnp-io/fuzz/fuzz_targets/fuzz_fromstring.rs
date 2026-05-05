#![no_main]

use libfuzzer_sys::fuzz_target;
use fnp_io::IOSupportedDType;

fuzz_target!(|data: &[u8]| {
    if data.len() > 1_000_000 {
        return;
    }
    let _ = fnp_io::fromstring(data, IOSupportedDType::F64, " ");
    let _ = fnp_io::fromstring(data, IOSupportedDType::F64, ",");
    let _ = fnp_io::fromstring(data, IOSupportedDType::F32, " ");
    let _ = fnp_io::fromstring(data, IOSupportedDType::I64, " ");
    let _ = fnp_io::fromstring(data, IOSupportedDType::I32, " ");
});
