#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &str| {
    if data.len() > 64 {
        return;
    }
    let _ = fnp_ufunc::DateTimeUnit::parse(data);
});
