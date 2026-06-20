# Performance Release Readiness Scorecard

Scope: rolling gauntlet verification of measured FrankenNumPy performance slices
against original NumPy.

## 2026-06-20 Linalg Spectral Bold-Verify

| Area | Score | Verdict |
|---|---:|---|
| Current `batch_cholesky` 64/128/256 rows | 9/10 | Release-ready current win |
| `eigvalsh_nxn/128` | 2/10 | Not release-ready; 3.051x slower than NumPy |
| `cond_nxn/128` | 4/10 | Not release-ready; 1.583x slower than NumPy |
| Small-threshold / sort / cond-extrema probes | 0/10 | Rejected and reverted |

Evidence:
- Artifact: `tests/artifacts/perf/2026-06-20_linalg_batch_cholesky_cod_a/`.
- Current batch Cholesky FNP/NumPy ratios: `0.281x`, `0.226x`, `0.152x`.
- Current spectral FNP/NumPy ratios on `vmi1227854`: `eigvalsh_nxn/128 = 3.051x`,
  `cond_nxn/128 = 1.583x`.
- Rejected probes:
  `TRIDIAG_BLOCK_MIN=192` failed golden digest,
  `cond_nxn` direct extrema regressed paired A/B by `1.026x`,
  `eigvalsh_nxn sort_unstable_by` regressed paired A/B by `1.113x`.
- Production `crates/fnp-linalg/src/lib.rs` source returned to baseline after
  rejected probes.

Decision:
- Keep no new linalg source from this slice.
- Preserve the batch-Cholesky benchmark evidence as a current win.
- Route next spectral work to a deeper tridiagonal reduction/QR primitive; do
  not retry threshold, sort, or post-processing-only levers for this loss class.

---

## 2026-06-19 Random PCG Backlog

## Summary

| Area | Score | Verdict |
|---|---:|---|
| `franken_numpy-ixs5y.255` parallel PCG raw `fill_u64` | 9/10 | Release-ready keep |
| `franken_numpy-ixs5y.257` PCG bytes word-fill | 0/10 | Rejected and reverted |
| Current `Generator::bytes` parity/perf gap | 4/10 | Correctness guarded, performance not release-ready |
| `franken_numpy-ixs5y.250` parallel PCG gumbel inverse-CDF fill | 9/10 | Release-ready keep |
| `franken_numpy-ixs5y.253` parallel PCG laplace inverse-CDF fill | 9/10 | Release-ready keep |

## Gate Results

| Gate | Result | Evidence |
|---|---|---|
| Crate bench build | Pass | `rch exec -- cargo check -p fnp-random --benches` |
| Raw-fill conformance | Pass | `parallel_pcg_fill_u64_matches_serial_stream_state` |
| Bytes stream conformance | Pass | `bytes_large_calls_match_serial_uint32_stream_state` |
| Gumbel stream conformance | Pass | `parallel_pcg_gumbel_matches_serial_stream_state` |
| Laplace stream conformance | Pass | `parallel_pcg_laplace_matches_serial_stream_state` |
| Distribution live NumPy oracle | Pass | `gumbel_matches_live_numpy_oracle`, `laplace_matches_live_numpy_oracle` |
| Head-to-head Criterion vs NumPy | Pass with mixed verdicts | `tests/artifacts/perf/2026-06-19_random_vs_numpy_pcg/` |
| Head-to-head distribution Criterion vs NumPy | Pass | `tests/artifacts/perf/2026-06-19_random_vs_numpy_pcg_distributions/` |
| Negative-evidence ledger | Updated | `docs/NEGATIVE_EVIDENCE.md` |
| Required reverts | Done | Removed `.257` production word-fill path from `Generator::bytes` |

## Decision

Keep `.255`: final-code `fill_u64` is faster than NumPy by 3.72x at 100k u64 and 2.03x at 1M u64 on `hz1`.

Reject `.257`: pre-revert bytes word-fill was slower than NumPy by 1.64x at 100k bytes and 1.99x at 1M bytes on `ovh-a`. The production optimization was reverted; current serial bytes remains a measured open gap, not a kept optimization.

Keep `.250`: final-code PCG64 gumbel is faster than NumPy by 6.01x at 100k f64 and 7.15x at 1M f64 on `ovh-a`.

Keep `.253`: final-code PCG64 laplace is faster than NumPy by 6.76x at 100k f64 and 8.67x at 1M f64 on `ovh-a`.
