# 2026-06-20 `fnp-linalg` batched Frobenius norm vs NumPy

Bead: `franken_numpy-ixs5y.238`

Worker: `hz1`

Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`

## Decision

Keep and close the existing direct batched Frobenius lane-fill path. No new
source edit was needed in this closeout; the code-first lever is now measured
head-to-head against NumPy on the same worker.

Win/loss/neutral vs NumPy: `2/0/0`.

| Workload | FrankenNumPy median | NumPy median | FNP/NumPy | NumPy/FNP | Verdict |
|---|---:|---:|---:|---:|---|
| `batch_matrix_norm_fro/shape/4096x8x8` | 76177 ns | 234973 ns | 0.324x | 3.08x | Win |
| `batch_matrix_norm_fro/shape/1024x32x32` | 218772 ns | 581466 ns | 0.376x | 2.66x | Win |

## Artifacts

- `fnp_batch_matrix_norm_fro_current.txt`: RCH Criterion run, selected worker
  `hz1`.
- `numpy_batch_matrix_norm_fro_hz1_success.txt`: direct Python comparator on
  `hz1`, Python 3.14.4, NumPy 2.3.5, same data formula and shapes.
- `fnp_linalg_batch_fro_focused_test.txt`: focused bit-preservation test.
- `fnp_linalg_release_build.txt`: per-crate release build gate through RCH.
- `numpy_batch_matrix_norm_fro_hz1.txt`: invalid local shell-quote attempt, not
  counted.

## Validation

`CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-linalg batch_matrix_norm_fro_direct_lane_fill_matches_per_lane_reference_bits -- --nocapture`

Result: pass, 1 test passed, 0 failed.

`CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo build -p fnp-linalg --release`

Result: pass, release profile finished.
