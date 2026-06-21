# BOLD-VERIFY Scorecard: eigvalsh 128 values-only reducer probe

Bead: `franken_numpy-ixs5y.277`
Agent: `YellowElk` / `cod-b`
Crate/API: `fnp-linalg::eigvalsh_nxn`
Decision: NO-SHIP, source reverted

## Baseline

Counted Rust baseline:
- Worker: `vmi1149989`
- Command: `rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg eigvalsh_nxn/size/128 -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- Result: `1,372,654 ns/iter (+/- 65,688)`

Counted NumPy comparator:
- Worker: `vmi1149989` via SSH alias
- NumPy: `2.2.4`, Python `3.13.7`
- Environment: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`
- Result: `708,451 ns` median
- Baseline ratio: `1.937x` slower than NumPy

## Candidate

Candidate: small blocked reducer matvec used tail-local `u`/`v` slices for
`n < 192`, leaving the existing gated row-dot and large-matrix paths unchanged.

| Probe | Baseline ns | Candidate ns | Candidate/Baseline | Candidate/NumPy | Verdict |
|---|---:|---:|---:|---:|---|
| First candidate vs RCH baseline | 1,372,654 | 1,295,452 | 0.944x | 1.829x | inconclusive |
| Paired direct repeat | 1,295,211 | 1,380,393 | 1.066x | 1.949x | reject |

## Validation

- `cargo test -p fnp-linalg tridiag --release`: pass on RCH-selected
  `vmi1152480`; 7 passed, 0 failed, 4 ignored. This includes the full-row-dot
  bit-equivalence test and the rank2k/eigvalsh golden.
- `cargo test -p fnp-linalg tridiag_eigvals_qr_perf_report --release -- --ignored --nocapture`:
  pass on `vmi1149989`; values-only QR remains 1.23x-1.24x faster than the old
  libm-hypot path at n=256/512/768.

## Outcome

The first run was below baseline but inside timing noise. The paired direct A/B
on the same worker reversed direction and regressed by `1.066x`, so the candidate
failed the gauntlet. The source hunk was reverted. Keep only the evidence.
