# BOLD-VERIFY Scorecard: eigvalsh 128 Sturm bisection no-ship

Bead/directive: `franken_numpy-ixs5y`
Agent: `YellowElk` / `cod-b`
Crate/API: `fnp-linalg::eigvalsh_nxn`
Decision: NO-SHIP, source reverted

## Baseline

Counted Rust baseline:
- Worker: `hz2`
- Command: `rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg eigvalsh_nxn/size/128 -- --sample-size 12 --warm-up-time 1 --measurement-time 2 --output-format bencher`
- Result: `1,545,094 ns/iter (+/- 109,248)`

Counted NumPy comparator:
- Worker: `hz2` via `ssh hz2`
- NumPy: `2.3.5`, Python `3.14.4`
- Environment: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`
- Result: `750,348 ns` median
- Baseline ratio: `2.059x` slower than NumPy

## Candidate

Candidate: route exact `n == 128` values-only symmetric-tridiagonal eigenvalues
through a Sturm-count bisection solver after the existing blocked Householder
reduction.

| Probe | FNP ns | NumPy ns | FNP/NumPy | Candidate/Baseline | Verdict |
|---|---:|---:|---:|---:|---|
| Current QR path | 1,545,094 | 750,348 | 2.059x | n/a | current loss |
| Candidate bisection | 5,133,686 | 750,348 | 6.842x | 3.322x | reject |

## Validation

- Candidate-only focused test passed:
  `cargo test -p fnp-linalg eigvalsh_128_bisection_matches_qr_reference --release`.
- The test compared the bisection values against the established QR values on
  the deterministic 128x128 SPD benchmark matrix with max-diff `< 1e-9`.
- After reverting the candidate, `cargo test -p fnp-linalg tridiag --release`
  passed 7/7 with 4 ignored timing reports.
- `cargo build -p fnp-linalg --release` passed.
- `git diff --check` passed.
- `cargo fmt --check -p fnp-linalg` still reports broad pre-existing linalg
  formatting drift; this evidence-only pass did not normalize unrelated files.

## Outcome

Correctness was acceptable, but the per-eigenvalue bisection approach destroyed
the performance target. The candidate source and temporary test were reverted.
Do not retry full-spectrum Sturm bisection for this residual; use a shared-work
tridiagonal eigensolver, true two-stage band reduction, or a generated
128-specific reducer instead.
