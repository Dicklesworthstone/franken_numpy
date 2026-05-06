#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &str| {
    if data.len() > 1024 {
        return;
    }
    let _ = fnp_ufunc::parse_gufunc_signature(Some(data), None);
    let _ = fnp_ufunc::parse_gufunc_signature(None, Some(data));
});
