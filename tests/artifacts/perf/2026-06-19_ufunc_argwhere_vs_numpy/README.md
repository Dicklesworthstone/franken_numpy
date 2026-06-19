# 2026-06-19 ufunc argwhere vs NumPy

Bead: `franken_numpy-ixs5y.248`

This directory contains the gauntlet evidence for the flat F64
`argwhere()` interleaved coordinate materialization optimization. The workload
is the `argwhere_f64_2d_sparse` Criterion row at 512x512 and 1024x1024 using the
same sparse/nonzero formula as the NumPy timing script.

## Decision rows

Reference: NumPy 2.4.3 on Python 3.13.7, same host `thinkstation1`.

| Workload | FrankenNumPy median | NumPy median | FNP/NumPy ratio | Verdict |
|---:|---:|---:|---:|---|
| 512x512 | 392.63 us | 1195.008 us | 0.329x | Keep, 3.04x faster |
| 1024x1024 | 1054.2 us | 5047.868 us | 0.209x | Keep, 4.79x faster |

NumPy timing CV was above 10% for both rows, but the NumPy minimum remained well
above the FrankenNumPy Criterion upper bound on both sizes, so the keep decision
does not depend on a noisy median edge.

## Artifacts

- `argwhere_guard.txt`: targeted golden guard, passed.
- `criterion_argwhere_local.txt`: Criterion rows for `argwhere_f64_2d_sparse`.
- `numpy_argwhere_local.txt`: NumPy timing rows for the same data formula.
- `cargo_check_fnp_ufunc.txt`: `cargo check -p fnp-ufunc`, passed.
- `cargo_clippy_fnp_ufunc.txt`: `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`, passed.
- `cargo_fmt_check.txt`: `cargo fmt --check`, failed due broad pre-existing
  workspace formatting drift outside this slice.
