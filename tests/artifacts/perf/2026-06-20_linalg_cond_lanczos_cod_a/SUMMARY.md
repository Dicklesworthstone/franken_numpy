# 2026-06-20 fnp-linalg symmetric spectral bold-verify

Agent: `YellowElk` / `cod-a`

Bead: `deadlock-audit-yy5qp`

Worker: `vmi1227854`

Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`

## Decision

No production source was kept. The current batch eigvalsh rows already beat
NumPy, while the remaining single-matrix spectral loss is in the deeper
Householder/tridiagonalization class. A fixed-iteration Lanczos/power extremal
shortcut for `cond` was rejected before implementation because the benchmark
SPD spectra are tightly clustered and the residual probe was not accurate
enough for NumPy-compatible semantics.

## Counted Ratios

| Row | FNP ns | NumPy ns | FNP/NumPy | Verdict |
|---|---:|---:|---:|---|
| `eigvalsh_nxn/size/128` | 1,172,682 | 380,630 | 3.081x | loss |
| `cond_nxn/size/64` | 165,033 | 117,136 | 1.409x | loss |
| `cond_nxn/size/128` | 919,355 | 1,070,705 | 0.859x | win |
| `cond_nxn/size/256` | 6,340,763 | 4,440,063 | 1.428x | loss |
| `cond_nxn/size/512` | 41,765,364 | 96,972,744 | 0.431x | win |
| `batch_eigvalsh/shape/64x128x128` | 10,513,359 | 18,205,409 | 0.577x | win |
| `batch_eigvalsh/shape/16x256x256` | 17,286,747 | 3,043,820,218 | 0.0057x | win |

## Artifacts

- `baseline_cond_eigvalsh_vmi1227854.txt`: counted Rust Criterion rows for
  `eigvalsh_nxn/128` and `cond_nxn/{64,128,256,512}`.
- `numpy_cond_eigvalsh_vmi1227854_remote.txt`: counted same-worker NumPy
  comparator for the same deterministic matrices.
- `numpy_cond_eigvalsh_vmi1227854.txt`: invalid retained comparator; `rch exec`
  did not offload the Python command and the run happened locally.
- `baseline_batch_eigvalsh_vmi1227854.txt`: counted Rust Criterion rows for
  `batch_eigvalsh`.
- `numpy_batch_eigvalsh_vmi1227854_remote.txt`: counted same-worker NumPy
  comparator for the exact batch matrix generator.
- `current_tridiag_qr_profile_vmi1227854.txt`: current values-only QR profile;
  the existing scaled-hypot path is still 1.24x-1.25x faster than the old
  libm-hypot path.

## Validation

- `rch exec -- cargo test -p fnp-linalg tridiag_eigvals_qr_perf_report --release -- --ignored --nocapture`
  passed on `vmi1227854`.
- `rch exec -- cargo test -p fnp-linalg --release` attempted RCH execution, then
  fell back locally because workers were under critical pressure. Result: 313
  unit tests, 37 conformance tests, 19 golden tests, 19 metamorphic tests, 4
  solve perf tests, and doctests passed.
- `rch exec -- cargo check -p fnp-linalg --all-targets` passed on `vmi1227854`.
- `rch exec -- cargo clippy -p fnp-linalg --all-targets -- -D warnings` passed
  on `vmi1227854`.
- `git diff --check` passed.
- `ubs` on the changed markdown evidence files exited 0 with no source-language
  findings.
- No `crates/fnp-linalg/src/lib.rs` production diff survived the gauntlet.

## Next Route

The next credible source attempt must attack the reduction/eigensolver class
itself: dsytrd-class blocked Householder, communication-avoiding or two-stage
tridiagonalization, or a fully convergent tridiagonal eigensolver with golden
proof. Do not retry sort, direct-extrema, fixed-iteration extremal, or threshold
microlevers for this gap.
