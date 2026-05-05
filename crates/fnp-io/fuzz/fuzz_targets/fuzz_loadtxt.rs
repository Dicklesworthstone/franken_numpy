#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() > 1_000_000 {
        return;
    }
    if let Ok(text) = std::str::from_utf8(data) {
        let _ = fnp_io::loadtxt(text, ' ', '#', 0, usize::MAX);
        let _ = fnp_io::loadtxt(text, ',', '#', 0, usize::MAX);
        let _ = fnp_io::loadtxt(text, '\t', '#', 0, usize::MAX);
        let _ = fnp_io::genfromtxt(text, ' ', '#', 0, f64::NAN);
        if let Some(count) = data.len().checked_sub(10) {
            let _ = fnp_io::fromfile_text(text, " ", Some(count.min(1000)));
        }
    }
});
