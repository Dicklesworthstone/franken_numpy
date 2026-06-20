# 2026-06-20 fnp-linalg batched column-sum norm vs NumPy

Bead: `franken_numpy-ixs5y.240`
Agent: `BlackThrush` / `cod-b`
Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`

## Verdict

Keep and close the existing direct `batch_matrix_norm` column-sum lane fill for
`ord="1"` and `ord="-1"`.

- Same-worker `hz2` final FrankenNumPy vs NumPy: 4 wins / 0 losses / 0 neutral.
- No source hunk was added in this verification slice.
- The unpinned `vmi1153651` Rust sample and local Python comparator are recorded
  as invalid routing evidence, not keep/reject proof.

## Scored Rows

| Workload | FrankenNumPy ns | NumPy ns | FNP/NumPy | Verdict |
|---|---:|---:|---:|---|
| `batch_matrix_norm_column_sum/shape/1_4096x8x8` | 81,679 | 903,879 | 0.090x | Win |
| `batch_matrix_norm_column_sum/shape/1_1024x32x32` | 101,266 | 917,304 | 0.110x | Win |
| `batch_matrix_norm_column_sum/shape/-1_4096x8x8` | 78,648 | 989,052 | 0.080x | Win |
| `batch_matrix_norm_column_sum/shape/-1_1024x32x32` | 95,781 | 991,737 | 0.097x | Win |

## Commands

- `RCH_WORKER=hz2 RCH_REQUIRE_REMOTE=1 RCH_DAEMON_WAIT_RESPONSE_TIMEOUT_SECS=240 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg batch_matrix_norm_column_sum -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `ssh hz2 'cd /data/projects/franken_numpy && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python3 - <<PY ... PY'`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-linalg batch_matrix_norm_column_sum_direct_lane_fill_matches_per_lane_reference_bits -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo check -p fnp-linalg --all-targets`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo clippy -p fnp-linalg --all-targets -- -D warnings`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo build -p fnp-linalg --release`

## Validation

- Focused bit-preservation test: pass.
- `cargo check -p fnp-linalg --all-targets`: pass on retry after an `ovh-b`
  worker `SIGILL` in the `zerocopy` build script.
- `cargo clippy -p fnp-linalg --all-targets -- -D warnings`: pass.
- `cargo build -p fnp-linalg --release`: pass.
- `cargo fmt --package fnp-linalg -- --check`: known broad pre-existing drift
  outside this no-source verification slice.

## Negative Evidence

- Do not retry the whole-matrix NaN prefilter or stack-threshold-only variants
  already rejected for the column norm residual.
- Do not score cross-host Python comparators. `rch exec -- python3 - ...` is not
  a supported remote comparator path; it warned and ran locally.
- A credible next column primitive needs SIMD absolute-value extraction or
  strip-mined multi-column accumulation while preserving addition order and NaN
  semantics.
