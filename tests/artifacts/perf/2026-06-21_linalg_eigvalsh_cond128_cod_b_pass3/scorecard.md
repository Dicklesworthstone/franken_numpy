# 2026-06-21 linalg eigvalsh/cond 128 pass3 scorecard

Agent: YellowElk / cod-b
Parent bead: franken_numpy-ixs5y
Target dir: /data/projects/.rch-targets/franken_numpy-cod-b
Scope: fnp-linalg only

## Result

No production source kept. The values-only exact-128 unblocked tridiagonal
reducer route was slower than NumPy on both measured rows and was reverted.

| Probe | Worker | FNP ns | NumPy ns | FNP/NumPy | Verdict |
|---|---|---:|---:|---:|---|
| Current eigvalsh_nxn/128 | hz1 | 1,906,955 | 911,490 | 2.092x | current loss |
| Current cond_nxn/128 | hz1 | 1,787,593 | 1,372,420 | 1.303x | current loss |
| Current rerun eigvalsh_nxn/128 | ovh-a | 1,318,349 | 669,516 | 1.969x | current loss |
| Current rerun cond_nxn/128 | ovh-a | 1,226,881 | 1,009,183 | 1.216x | current loss |
| Candidate unblocked-128 eigvalsh_nxn/128 | vmi1153651 | 4,243,947 | 803,699 | 5.280x | no-ship |
| Candidate unblocked-128 cond_nxn/128 | vmi1153651 | 3,856,139 | 1,541,118 | 2.502x | no-ship |

## Files

- `current_eigvalsh_cond128_rch.txt`: earlier fixed `hz1` current Rust run.
- `numpy_eigvalsh_cond128_hz1.txt`: direct `hz1` NumPy comparator.
- `current_eigvalsh_cond128_rerun_hz1.txt`: current Rust rerun where RCH selected `ovh-a`.
- `numpy_eigvalsh_cond128_ovh_a_rerun.txt`: direct `ovh-a` NumPy comparator.
- `candidate_unblocked128_rch.txt`: candidate Rust run where RCH selected `vmi1153651`.
- `numpy_eigvalsh_cond128_vmi1153651_candidate_host.txt`: direct `vmi1153651` NumPy comparator.
- `final_test_tridiag_release.txt`: `cargo test -p fnp-linalg tridiag --release`
  passed 7 tests with 4 ignored probes through RCH.
- `final_check_linalg.txt`: `cargo check -p fnp-linalg --all-targets` passed
  through RCH.
- `final_clippy_linalg.txt`: `cargo clippy -p fnp-linalg --all-targets -- -D warnings`
  passed through RCH.
- `final_build_linalg_release.txt`: `cargo build -p fnp-linalg --release` passed
  through RCH.
- `final_git_diff_check.txt`: `git diff --check` passed.
- `final_fmt_linalg.txt`: `cargo fmt -p fnp-linalg --check` reported
  pre-existing rustfmt drift in linalg benches/examples/tests and unrelated
  source regions; no formatting rewrite was applied.

## Decision

The current 128-size spectral gap is real, but the unblocked values-only reducer
route is not a viable fix. Keep the negative evidence and route future work to a
shared-work tridiagonal eigensolver, true two-stage band reduction, or a genuinely
generated 128-specialized reducer with paired same-worker proof.
