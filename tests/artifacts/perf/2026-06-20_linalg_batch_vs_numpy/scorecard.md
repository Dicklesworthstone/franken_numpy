# 2026-06-20 fnp-linalg matrix norm column reductions

Bead: `franken_numpy-ixs5y.235`

## Verdict

Keep the stack scratch buffer for large `ord=1/-1` matrix norm column reductions.

- Same-worker `hz2` old FNP -> final FNP: 8 wins / 0 losses / 0 neutral.
- Same-worker `hz2` final FNP -> NumPy: 2 wins / 6 losses / 0 neutral.
- This is a gap-narrowing keep, not a full NumPy closeout.
- Frobenius unroll was reverted after same-worker batch regressions.

## `hz2` Column Norm Rows

| Row | Old FNP ns | Final FNP ns | NumPy ns | Final/Old | Final/NumPy |
|---|---:|---:|---:|---:|---:|
| `one/128` | 9603 | 7544 | 9553 | 0.786x | 0.790x |
| `neg_one/128` | 9375 | 7484 | 9574 | 0.798x | 0.782x |
| `one/256` | 38032 | 30444 | 27712 | 0.800x | 1.099x |
| `neg_one/256` | 37675 | 29924 | 28312 | 0.794x | 1.057x |
| `one/512` | 154304 | 116333 | 103667 | 0.754x | 1.122x |
| `neg_one/512` | 152028 | 116827 | 102987 | 0.768x | 1.134x |
| `one/1024` | 615716 | 458082 | 397192 | 0.744x | 1.153x |
| `neg_one/1024` | 603420 | 466084 | 393621 | 0.772x | 1.184x |

## Reverted Frobenius Probe

| Row | Old FNP ns | Candidate FNP ns | Candidate/Old | Verdict |
|---|---:|---:|---:|---|
| `matrix_norm_nxn_orders/fro/128` | 15559 | 15484 | 0.995x | neutral |
| `matrix_norm_nxn_orders/fro/256` | 62444 | 63010 | 1.009x | neutral/loss |
| `matrix_norm_nxn_orders/fro/512` | 256472 | 251532 | 0.981x | neutral |
| `matrix_norm_nxn_orders/fro/1024` | 1030224 | 998542 | 0.969x | small win |
| `batch_matrix_norm_fro/4096x8x8` | 76821 | 84812 | 1.104x | loss, reverted |
| `batch_matrix_norm_fro/1024x32x32` | 194123 | 221803 | 1.143x | loss, reverted |

## Artifacts

- Old `hz2` FNP and full routing baseline: `criterion_linalg_current.txt`
- Final gated candidate, selected worker `hz2`: `criterion_linalg_column_stack_gated_candidate_hz1.txt`
- NumPy comparator on `hz2`: `numpy_linalg_hz2.txt`
- Same-worker `vmi1149989` old/code rerun: `criterion_linalg_column_head_baseline_rerun.txt`, `criterion_linalg_column_stack_gated_candidate.txt`
- Focused conformance guard: `cargo_test_column_sum_stack.txt`
- Frobenius no-ship: `criterion_linalg_fro_head_baseline.txt`, `criterion_linalg_fro_unroll_candidate.txt`
- Invalid/non-counted RCH Python probe: `numpy_column_vmi_rch.txt`

## Validation

- `rch exec -- cargo test -p fnp-linalg matrix_norm_column_reduction_matches_strided_reference_bits -- --nocapture`: pass on `hz1`.
- `rch exec -- cargo check -p fnp-linalg --all-targets`: pass on `hz1`.
- `rch exec -- cargo clippy -p fnp-linalg --all-targets -- -D warnings`: pass on `hz2`.
- `git diff --check`: pass.
- `cargo fmt --check --package fnp-linalg`: fails on broad pre-existing crate formatting drift outside this slice; my hunk was manually adjusted to the rustfmt suggestion.
- `ubs crates/fnp-linalg/src/lib.rs docs/NEGATIVE_EVIDENCE.md tests/artifacts/perf/2026-06-20_linalg_batch_vs_numpy/scorecard.md`: nonzero from broad pre-existing whole-file `fnp-linalg` inventory; UBS internal fmt/clippy/check/test-build sections were clean.

## Next Lever

The remaining 256-1024 losses are accumulation throughput, not allocation overhead. A credible next attempt needs SIMD absolute-value extraction or column strip mining while preserving per-column scalar addition order and NaN behavior.
