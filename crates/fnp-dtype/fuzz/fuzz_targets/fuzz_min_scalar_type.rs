#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: f64| {
    let _ = fnp_dtype::min_scalar_type(data);
});
