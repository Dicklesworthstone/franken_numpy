# 2026-06-19 ufunc count_nonzero vs NumPy

Bead: `franken_numpy-ixs5y.246`

This directory contains the gauntlet evidence for the flat F64
`count_nonzero(axis=None)` optimization. The original code-first candidate tied
parallel activation and chunk size to `1 << 14`. Head-to-head measurement showed
that was too eager for 100k elements, so the final code keeps parallelism only
above `1 << 19` elements while preserving 4096-element worker chunks.

## Decision rows

Reference: NumPy 2.4.3 on Python 3.13.7, same host `thinkstation1`.

| Candidate | Workload | FrankenNumPy median | NumPy median | FNP/NumPy ratio | Verdict |
|---|---:|---:|---:|---:|---|
| Original `1 << 14` threshold | 100k | 138.89 us | 39.006 us | 3.56x | Rejected, too eager |
| Original `1 << 14` threshold | 1M | 92.072 us | 384.147 us | 0.240x | Keep large-row signal |
| Final `1 << 19` threshold, 4096 chunk | 100k | 8.3121 us | 39.006 us | 0.213x | Keep |
| Final `1 << 19` threshold, 4096 chunk | 1M | 110.42 us | 384.147 us | 0.287x | Keep, noisy CI |

## Artifacts

- `count_nonzero_guard.txt`: pre-edit targeted golden guard, passed remotely.
- `count_nonzero_guard_after_threshold.txt`: expected failure after raising the
  threshold because the threshold-crossing fixture digest changed.
- `count_nonzero_guard_after_threshold_golden.txt`: updated digest guard, passed.
- `count_nonzero_guard_final.txt`: final local targeted guard, passed.
- `criterion_count_nonzero_local.txt`: original candidate Criterion rows.
- `criterion_count_nonzero_local_after_threshold.txt`: threshold-only rows.
- `criterion_count_nonzero_local_final.txt`: final threshold plus separate chunk rows.
- `numpy_count_nonzero_local.txt`: NumPy timing rows for the same data formula.
- `cargo_check_fnp_ufunc.txt`: `cargo check -p fnp-ufunc`, passed.
- `cargo_clippy_fnp_ufunc.txt`: `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`, passed.
- `cargo_fmt_check.txt`: `cargo fmt --check`, failed due broad pre-existing
  workspace formatting drift outside this slice.
