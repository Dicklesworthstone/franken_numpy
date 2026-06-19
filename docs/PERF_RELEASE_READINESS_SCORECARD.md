# Performance Release Readiness Scorecard

Scope: 2026-06-19 gauntlet verification of recent `fnp-random` PCG backlog against original NumPy.

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
