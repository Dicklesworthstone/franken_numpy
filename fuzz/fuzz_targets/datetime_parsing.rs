#![no_main]

use fnp_ufunc::UFuncArray;
use libfuzzer_sys::fuzz_target;

const MAX_FUZZ_INPUT_BYTES: usize = 1 << 12;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_FUZZ_INPUT_BYTES {
        return;
    }

    if let Ok(text) = std::str::from_utf8(data) {
        for unit in [None, Some("s"), Some("ms"), Some("us"), Some("ns"), Some("D")] {
            let _ = UFuncArray::from_datetime_strings(vec![1], vec![text.to_string()], unit);
            let _ = UFuncArray::from_timedelta_strings(vec![1], vec![text.to_string()], unit);
        }

        if let Some((left, right)) = text.split_once('\n') {
            let values = vec![left.to_string(), right.to_string()];
            let _ = UFuncArray::from_datetime_strings(vec![2], values.clone(), None);
            let _ = UFuncArray::from_timedelta_strings(vec![2], values, None);
        }
    }
});
