# 2026-06-20 fnp-linalg matrix norm row-block SIMD keep

Bead: `franken_numpy-ixs5y.274`

## Verdict

Keep the 4-row SIMD accumulator for large `ord=1/-1` matrix norm column
reductions.

- Same-worker `hz2` old FNP -> final FNP: 6 wins / 0 losses / 0 neutral.
- Same-worker `hz2` final FNP -> NumPy: 6 wins / 0 losses / 0 neutral.
- The fresh baseline already beat NumPy on this lane, but the row-blocked
  accumulator widened the lead and cut the remaining Rust time.
- No source hunk was reverted in this slice.

## `hz2` Column Norm Rows

Final FNP uses the repeat candidate run. NumPy uses the fresh same-host direct
Python comparator on `hz2`.

| Row | Baseline FNP ns | Final FNP ns | NumPy ns | Baseline/NumPy | Final/Baseline | Final/NumPy |
|---|---:|---:|---:|---:|---:|---:|
| `one/256` | 11441 | 5337 | 33589 | 0.341x | 0.466x | 0.159x |
| `neg_one/256` | 9268 | 5093 | 33875 | 0.274x | 0.550x | 0.150x |
| `one/512` | 37970 | 28409 | 96297 | 0.394x | 0.748x | 0.295x |
| `neg_one/512` | 37477 | 28023 | 92790 | 0.404x | 0.748x | 0.302x |
| `one/1024` | 151777 | 123032 | 342892 | 0.443x | 0.811x | 0.359x |
| `neg_one/1024` | 152666 | 123074 | 341621 | 0.447x | 0.806x | 0.360x |

## First Candidate Pass

| Row | Candidate FNP ns | Candidate/Baseline | Candidate/NumPy |
|---|---:|---:|---:|
| `one/256` | 5280 | 0.462x | 0.157x |
| `neg_one/256` | 5336 | 0.576x | 0.158x |
| `one/512` | 27744 | 0.731x | 0.288x |
| `neg_one/512` | 28399 | 0.758x | 0.306x |
| `one/1024` | 120512 | 0.794x | 0.352x |
| `neg_one/1024` | 121084 | 0.793x | 0.354x |

## Implementation Note

The old SIMD fill loaded and stored each `col_sums` vector once per matrix row.
The kept lever processes four adjacent rows at a time, adds `row0`, `row1`,
`row2`, and `row3` into the same column lane in that order, then stores the
lane once. Remainder rows keep the prior one-row SIMD path. Tail scalar columns
still check NaN before adding, and SIMD-lane NaNs are detected by the final
`col_sums` scan as before.

## Artifacts

- Baseline FNP on `hz2`: `baseline_matrix_norm_orders_hz2.txt`
- Fresh NumPy on `hz2`: `numpy_matrix_norm_hz2.txt`
- Focused bit guard: `cargo_test_column_rowblock.txt`
- First candidate on `hz2`: `candidate_matrix_norm_orders_hz2.txt`
- Repeat candidate on `hz2`: `candidate_matrix_norm_orders_repeat_hz2.txt`
- Compile/lint: `cargo_check_fnp_linalg.txt`,
  `cargo_build_release_fnp_linalg.txt`, `cargo_clippy_fnp_linalg.txt`
- Hygiene: `cargo_fmt_check_fnp_linalg.txt`, `git_diff_check.txt`,
  `git_diff_check_final.txt`, `ubs_changed_paths.txt`

## Validation

- `rch exec -- cargo test -p fnp-linalg matrix_norm_column_reduction_matches_strided_reference_bits -- --nocapture`: pass on `hz2`.
- `rch exec -- cargo check -p fnp-linalg --all-targets`: pass on `hz2`.
- `rch exec -- cargo build -p fnp-linalg --release`: pass on `hz2`.
- `rch exec -- cargo clippy -p fnp-linalg --all-targets -- -D warnings`: pass on `hz2`.
- `git diff --check`: pass.
- `cargo fmt -p fnp-linalg -- --check`: fails on broad pre-existing
  `fnp-linalg` formatting drift in benches/examples and unrelated `lib.rs`
  sections; the edited SIMD hunk was not reported in the rustfmt diff.
- `ubs crates/fnp-linalg/src/lib.rs docs/NEGATIVE_EVIDENCE.md
  docs/RELEASE_READINESS_SCORECARD.md
  tests/artifacts/perf/2026-06-20_linalg_column_norm_rowblock_cod_b/scorecard.md`:
  nonzero from broad existing `fnp-linalg` whole-file inventory; no finding was
  reported against the edited row-block SIMD hunk.

## Next Lever

This row family is now well ahead of NumPy for 256-1024 square matrices on
`hz2`. Do not spend the next no-gaps slice here unless a later benchmark shows a
fresh regression. The next credible linalg target should move back to a current
measured loss, likely deeper SVD/eig/solve kernels, with fresh same-host NumPy
capture before source work.
