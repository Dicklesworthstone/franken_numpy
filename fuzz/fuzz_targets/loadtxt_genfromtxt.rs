#![no_main]

use fnp_io::{genfromtxt, loadtxt, loadtxt_unpack, loadtxt_usecols};
use libfuzzer_sys::fuzz_target;

const MAX_FUZZ_INPUT_BYTES: usize = 1 << 16;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_FUZZ_INPUT_BYTES {
        return;
    }

    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };

    for delimiter in [',', ' ', '\t', ';', '|'] {
        for comments in ['#', '%', '/'] {
            for skiprows in [0, 1, 2] {
                for max_rows in [0, 10, 100] {
                    let _ = loadtxt(text, delimiter, comments, skiprows, max_rows);

                    for usecols in [None, Some(&[0usize][..]), Some(&[0, 1][..]), Some(&[1, 0][..])] {
                        let _ = loadtxt_usecols(
                            text, delimiter, comments, skiprows, max_rows, usecols,
                        );
                        for unpack in [false, true] {
                            let _ = loadtxt_unpack(
                                text, delimiter, comments, skiprows, max_rows, usecols, unpack,
                            );
                        }
                    }
                }
            }

            for skip_header in [0, 1, 2] {
                for filling_values in [0.0, f64::NAN, -999.0] {
                    let _ = genfromtxt(text, delimiter, comments, skip_header, filling_values);
                }
            }
        }
    }

    if data.len() > 2 {
        let custom_delimiter = data[0] as char;
        let custom_comment = data[1] as char;
        if custom_delimiter.is_ascii_graphic() && custom_comment.is_ascii_graphic() {
            let _ = loadtxt(&text[2..], custom_delimiter, custom_comment, 0, 0);
            let _ = genfromtxt(&text[2..], custom_delimiter, custom_comment, 0, 0.0);
        }
    }
});
