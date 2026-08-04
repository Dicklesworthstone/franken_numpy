# 2026-06-20 `fnp-linalg` batched trace vs NumPy

Bead: `franken_numpy-ixs5y.237`

Worker: `hz1`

Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`

## Decision

Keep and close the existing direct batched trace lane-fill path. No source edit
was needed in this closeout; the code-first lever is now measured head-to-head
against NumPy on the same worker.

Win/loss/neutral vs NumPy: `2/0/0`.

| Workload | FrankenNumPy median | NumPy median | FNP/NumPy | NumPy/FNP | Verdict |
|---|---:|---:|---:|---:|---|
| `batch_trace/shape/4096x8x8` | 47188 ns | 102977 ns | 0.458x | 2.18x | Win |
| `batch_trace/shape/1024x32x32` | 47184 ns | 61381 ns | 0.769x | 1.30x | Win |

## Artifacts

- `fnp_batch_trace_current_hz1.txt`: RCH Criterion run, pinned worker `hz1`.
- `numpy_batch_trace_hz1.txt`: direct Python comparator on `hz1`, Python
  3.14.4, NumPy 2.3.5, same data formula and shapes.
- `fnp_linalg_batch_trace_focused_test.txt`: focused bit-preservation test.
- `fnp_linalg_release_build.txt`: per-crate release build gate through RCH.

## Validation

`RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-linalg batch_trace_direct_lane_fill_matches_per_lane_reference_bits -- --nocapture`

Result: pass, 1 test passed, 0 failed.

`RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo build -p fnp-linalg --release`

Result: pass, release profile finished.
