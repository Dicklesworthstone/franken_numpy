//! Differential conformance: `fnp_io::savetxt` per-format float rendering must
//! match C `printf` (== NumPy's `savetxt` fmt engine) bit-for-bit.
//!
//! The golden fixture `savetxt_printf_golden.tsv` was produced by CPython's
//! `%`-formatting of `numpy.float64` scalars (the exact path NumPy's `savetxt`
//! uses: `fmt % tuple(row)`), one row per `(fmt, value)`:
//!   `<fmt>\t<value-bits-hex>\t<expected-output>`
//! Values are encoded as their IEEE-754 little-endian bit pattern so `-0.0`,
//! subnormals, and exact doubles round-trip without decimal-parse drift.
//!
//! Coverage spans the conversion letters (`e E f g G s r`), precisions, and the
//! full printf flag/width matrix (`0`-pad, `+`, space, `-` left-align, `#`
//! alternate, field width) — including platform quirks such as zero-padding a
//! non-finite value (`%08.2f` of `inf` -> `"00000inf"`). When NumPy *rejects* a
//! `(fmt, value)` pair (e.g. an integer conversion that overflows), the golden
//! records `ERR:<PyType>` and the test only requires that `savetxt` also errors.

use fnp_io::{savetxt, SaveTxtConfig};

fn render(fmt: &str, v: f64) -> String {
    let config = SaveTxtConfig {
        delimiter: " ",
        newline: "\n",
        fmt,
        header: "",
        footer: "",
        comments: "# ",
    };
    match savetxt(&[v], 1, 1, &config) {
        Ok(s) => s.trim_end_matches('\n').to_string(),
        Err(e) => format!("ERR:{e:?}"),
    }
}

#[test]
fn savetxt_float_formats_match_c_printf() {
    let fixture = include_str!("savetxt_printf_golden.tsv");
    let mut mismatches: Vec<String> = Vec::new();
    let mut checked = 0usize;

    for line in fixture.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let mut cols = line.splitn(3, '\t');
        let fmt = cols.next().expect("fmt column");
        let bits_hex = cols.next().expect("bits column");
        let expected = cols.next().unwrap_or("");

        let bits = u64::from_str_radix(bits_hex.trim_start_matches("0x"), 16)
            .unwrap_or_else(|_| panic!("bad bits hex: {bits_hex}"));
        let v = f64::from_bits(bits);

        let actual = render(fmt, v);
        checked += 1;

        // NumPy rejecting a (fmt, value) pair only obligates fnp to also reject.
        if expected.starts_with("ERR") && actual.starts_with("ERR:") {
            continue;
        }
        if actual != expected {
            mismatches.push(format!(
                "fmt={fmt:<10} bits={bits_hex} numpy={expected:?} fnp={actual:?}"
            ));
        }
    }

    assert!(
        mismatches.is_empty(),
        "{} of {checked} savetxt printf cases diverged from NumPy/C printf:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}
