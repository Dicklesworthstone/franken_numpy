#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() > 10_000_000 {
        return;
    }
    let _ = fnp_io::load_auto(data, false);
    let _ = fnp_io::load(data);
    let _ = fnp_io::load_complex(data);
    let _ = fnp_io::load_structured(data);
    let _ = fnp_io::load_strings(data);
});
