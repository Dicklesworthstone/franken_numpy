# 2026-06-20 fnp-ufunc where_nonzero vs NumPy

Bead: `franken_numpy-ixs5y.247`
Agent: `BlackThrush` / `cod-b`
Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`

## Decision

Keep the existing guarded large-F64 `where_nonzero` parallel coordinate-gather
path. No production performance hunk was added in this verification slice.

## Head-to-head

FNP Criterion used RCH on `hz2`. NumPy comparator ran directly on `hz2` and
reported host `hetzner2`, Python 3.14.4, NumPy 2.3.5.

| Workload | FNP | NumPy median | FNP/NumPy | Verdict |
|---|---:|---:|---:|---|
| `where_nonzero_f64_2d_sparse/262144` | 290,959 ns | 1,162,745 ns | 0.250x, 4.00x faster | Win |
| `where_nonzero_f64_2d_sparse/1048576` | 677,198 ns | 4,658,292 ns | 0.145x, 6.88x faster | Win |

Win/loss/neutral: `2/0/0`.

## Validation

| Gate | Result | Artifact |
|---|---|---|
| Focused where_nonzero golden | PASS | `test_where_nonzero_golden_hz2.txt` |
| Full `fnp-ufunc` suite | PASS | `cargo_test_fnp_ufunc_after_polynomial_fix_hz1.txt` |
| `cargo check -p fnp-ufunc --all-targets` | PASS | `cargo_check_fnp_ufunc_hz2.txt` |
| `cargo clippy -p fnp-ufunc --all-targets -- -D warnings` | PASS | `cargo_clippy_fnp_ufunc_after_polynomial_fix.txt` |
| `cargo build -p fnp-ufunc --release` | PASS | `cargo_build_release_fnp_ufunc.txt` |
| `cargo fmt --package fnp-ufunc -- --check` | KNOWN GAP | `cargo_fmt_fnp_ufunc_check_after_polynomial_fix.txt` |

The first full-suite run exposed a rounded Legendre multiplication golden:
`1.033333` was too coarse for the test tolerance. The test row now uses full
NumPy f64 coefficients for `legmul([1,2,3,4], [0.5,-1,2])`; the rerun passed.

## Retry Predicate

Do not retest generic F64 `where_nonzero` chunk gather or threshold-only tuning
as standalone work. A credible next lever must be a distinct coordinate
primitive, such as division-free 2-D coordinate reconstruction or row-run
tables, while preserving C-order coordinates, sidecar fallback, NaN truth, and
signed-zero false behavior.
