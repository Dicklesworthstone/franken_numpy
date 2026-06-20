# fnp-linalg column-norm SIMD lane accumulation

Date: 2026-06-20
Agent: BlackThrush / cod-a
Parent bead: franken_numpy-ixs5y
Target dir: CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a
Decision: KEEP

## Lever

`matrix_norm_nxn` order `1` and `-1` now route cache-linear column reductions
with `n >= 256` through a safe `std::simd::Simd<f64, 8>` helper. The scalar
cache-linear helper remains separate and is selected before the SIMD helper for
`n < 256`.

## Measured Rows

RCH Criterion baseline/candidate and direct NumPy comparator ran on
`vmi1227854`. NumPy used Python 3.13.7, NumPy 2.4.6, and single-thread env
`OMP/OPENBLAS/MKL/NUMEXPR=1`.

| Row | Old FNP | New FNP | NumPy p50 | New/Old | New/NumPy | Outcome |
|---|---:|---:|---:|---:|---:|---|
| `one/128` | 6,631 ns | 6,161 ns | 9,024 ns | 0.929x | 0.683x | guardrail win |
| `neg_one/128` | 6,816 ns | 7,134 ns | 9,224 ns | 1.047x | 0.774x | guardrail neutral/noisy |
| `one/256` | 34,821 ns | 6,496 ns | 24,116 ns | 0.187x | 0.269x | WIN |
| `neg_one/256` | 26,663 ns | 6,251 ns | 24,537 ns | 0.234x | 0.255x | WIN |
| `one/512` | 102,390 ns | 26,176 ns | 78,408 ns | 0.256x | 0.334x | WIN |
| `neg_one/512` | 163,924 ns | 25,195 ns | 77,666 ns | 0.154x | 0.324x | WIN |
| `one/1024` | 421,756 ns | 118,415 ns | 355,402 ns | 0.281x | 0.333x | WIN |
| `neg_one/1024` | 410,832 ns | 112,363 ns | 374,671 ns | 0.274x | 0.300x | WIN |

Target rows (`n >= 256`) vs NumPy: 6 wins / 0 losses / 0 neutral.
Full observed sweep vs NumPy: 8 wins / 0 losses / 0 neutral.
Old/new guardrail: 7 wins / 0 losses / 1 neutral/noisy.

## Validation

- PASS: `rch exec -- cargo test -p fnp-linalg matrix_norm_column_reduction_matches_strided_reference_bits -- --nocapture`
- PASS: `rch exec -- cargo check -p fnp-linalg --all-targets`
- PASS: `rch exec -- cargo clippy -p fnp-linalg --all-targets -- -D warnings`
- PASS: `rch exec -- cargo build -p fnp-linalg --release`
- KNOWN GAP: `cargo fmt -p fnp-linalg -- --check` reports broad pre-existing fmt drift outside this hunk.
- KNOWN GAP: `ubs crates/fnp-linalg/src/lib.rs` reports broad pre-existing inventory unrelated to the touched helper; its own fmt/clippy/check sub-gates were green.
