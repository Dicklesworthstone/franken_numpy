# 2026-06-20 `fnp-linalg` cond / values-only SVD vs NumPy

Bead: `franken_numpy-ixs5y.234`

Worker: `hz1`

Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`

## Decision

Keep the exact-symmetric spectral `cond_nxn` fast path. The original
values-only SVD in-place singular sort was already present and its bit guard
passed, but the head-to-head `cond_nxn` baseline still lost badly to NumPy.

The kept lever is mathematical rather than a retread of the SVD sort:
for finite exact-symmetric real matrices, singular values are the absolute
eigenvalues, so `cond(A, 2)` and `cond(A, -2)` can use `eigvalsh_nxn` instead
of the values-only SVD. Non-symmetric, rectangular, NaN, Inf, and non-spectral
orders keep the prior paths.

Final win/loss/neutral vs NumPy: `3/1/0`.

## Scorecard

| Workload | Old FNP ns | Final FNP ns | NumPy ns | Final/Old | Final/NumPy | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `cond_nxn/size/64` | 51961635 | 215148 | 229157 | 0.004x | 0.939x | Win, 1.07x faster than NumPy |
| `cond_nxn/size/128` | 287303721 | 1746263 | 1388876 | 0.006x | 1.257x | Keep, residual NumPy loss |
| `cond_nxn/size/256` | 1715056173 | 10107470 | 15179317 | 0.006x | 0.666x | Win, 1.50x faster than NumPy |
| `cond_nxn/size/512` | timed out in full old run | 60907729 | 121812521 | n/a | 0.500x | Win, 2.00x faster than NumPy |

## Artifacts

- `fnp_cond_nxn_current_hz1.txt`: initial full current run; `512` did not finish and the run was interrupted.
- `fnp_cond_nxn_64_128_256_hz1.txt`: clean old/current bounded baseline for 64, 128, 256.
- `fnp_cond_nxn_symmetric_fast_path_hz1.txt`: post-change Criterion run, full `cond_nxn` group, pinned worker `hz1`.
- `numpy_cond_nxn_hz1.txt`: direct NumPy comparator on `hz1`, Python 3.14.4, NumPy 2.3.5.
- `numpy_cond_bench.py`: exact comparator source with the same matrix generator as Criterion.
- `fnp_linalg_cond_symmetric_tests_final.txt`: focused fast-path correctness tests.
- `fnp_linalg_values_sort_test.txt`: original values-only SVD sort bit-preservation guard.
- `fnp_linalg_release_build.txt`: per-crate release build gate through RCH.
- `fnp_linalg_clippy.txt`: per-crate clippy gate through RCH.
- `cargo_fmt_fnp_linalg_check.txt`: `cargo fmt -p fnp-linalg -- --check` output; fails on broad pre-existing crate formatting drift outside this change.

## Validation

`RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-linalg cond_p_spectral_symmetric -- --nocapture`

Result: pass, 2 tests passed, 0 failed.

`RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-linalg values_only_svd_in_place_sort_matches_former_index_schedule -- --nocapture`

Result: pass, 1 test passed, 0 failed.

`RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo build -p fnp-linalg --release`

Result: pass, release profile finished.

`RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo clippy -p fnp-linalg --all-targets -- -D warnings`

Result: pass.

`git diff --check`

Result: pass.

## Residual Gap

`cond_nxn/size/128` still loses to NumPy by 1.257x on `hz1`. Do not reopen this
bead for SVD sort retuning; the sort allocation was not the dominant gap.
The next credible route is an `eigvalsh_nxn` 128-size values-only improvement
or a broader symmetric spectral fast path only if a same-worker `eigvalsh`
profile shows the exact frame.
