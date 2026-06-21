# Performance Release Readiness Scorecard

Scope: rolling gauntlet verification of measured FrankenNumPy performance slices
against original NumPy.

## 2026-06-21 fnp-python cholesky 2-D Delegate Code-Only Slice

| Area | Score | Verdict |
|---|---:|---|
| `fnp_python.linalg.cholesky` exact ndarray real 2-D square inputs | 3/10 | Code-only pending bench |
| Stacked / non-ndarray cholesky paths | 8/10 | Left unchanged |
| Focused conformance | 0/10 | Pending disk recovery |
| Fresh Criterion ratio-vs-NumPy | 0/10 | Pending disk recovery |

Evidence:
- Bead: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-a`.
- Existing disk-low probe recorded the native 2-D `cholesky` path losing to
  NumPy by `2.95x` at 200x200 and `6.28x` at 800x800.
- Code-only lever: delegate exact NumPy ndarray real 2-D square inputs to
  `numpy.linalg.cholesky` before extracting into Rust; preserve `upper` fallback
  semantics and keep stacked / non-ndarray paths unchanged.
- No new cargo build/bench/test/check was started after the 48G disk-low
  instruction. Agent Mail writes are blocked by the corrupt DB circuit breaker.

Decision:
- Keep as a pending-bench code-only commit, not a measured win.
- Next admissible turn must run focused `fnp-python` Criterion rows and cholesky
  conformance, then either score the ratio-vs-NumPy or revert on ~0 gain.

---

## 2026-06-21 Linalg Eigvalsh 128 Values-Only Reducer Probe

| Area | Score | Verdict |
|---|---:|---|
| `eigvalsh_nxn/128` current row | 2/10 | Not release-ready; 1.937x slower than NumPy on `vmi1149989` |
| Tail-local small-n reducer matvec | 0/10 | Rejected; paired direct A/B regressed 1.066x |
| Tridiagonal correctness gates | 9/10 | Focused release tests passed |
| Source/revert discipline | 9/10 | No production linalg diff kept |

Evidence:
- Artifact: `tests/artifacts/perf/2026-06-21_linalg_eigvalsh128_values_reducer_cod_b/`.
- Current baseline: `eigvalsh_nxn/size/128 = 1,372,654 ns`; same-worker NumPy
  median `708,451 ns`; FNP/NumPy `1.937x`.
- Candidate first run: `1,295,452 ns`, still `1.829x` NumPy and within baseline
  noise.
- Paired direct repeat on `vmi1149989`: baseline `1,295,211 ns`, candidate
  `1,380,393 ns`; candidate/baseline `1.066x` regression.
- `cargo test -p fnp-linalg tridiag --release` passed; QR profile stayed on the
  already-optimized scaled-hypot path.

Decision:
- Keep no source from this probe.
- Keep the negative evidence and route `eigvalsh_nxn/128` to a different
  reducer/eigensolver primitive.

---

## 2026-06-20 fnp-python Einsum Reduce-All Scalar Builder

| Area | Score | Verdict |
|---|---:|---|
| `fnp_einsum_reduce_all_f64_1000` | 8/10 | Release-ready current win for this worker |
| Existing f64 single-operand reduction conformance | 9/10 | Golden SHA and scalar parity green |
| Adjacent `reduce_rows_f64_1000` guard | 4/10 | Needs separate focused recheck; candidate run was 1.035x slower than NumPy |
| `fnp-python` all-targets/clippy/fmt hygiene | 3/10 | Blocked by pre-existing unrelated crate debt |

Evidence:
- Artifact: `tests/artifacts/perf/2026-06-20_python_einsum_reduce_all_scalar_cod_a/`.
- Same-worker `vmi1149989` target row:
  baseline `119,524 ns` vs NumPy `115,252 ns` (`1.037x` loss);
  candidate `100,778 ns` vs NumPy `104,427 ns` (`0.965x` win);
  candidate/old FNP `0.843x`.
- Guard rows from the same Criterion group:
  trace `0.754x`, diagonal `0.775x`, reduce rows `1.035x`, reduce cols `0.350x`
  candidate FNP/NumPy.
- `cargo test -p fnp-python --test conformance_einsum` passed after RCH failed
  open locally; `cargo build -p fnp-python --release` passed on `hz1`.
- `cargo check -p fnp-python --all-targets`, clippy, and fmt remain blocked by
  pre-existing unrelated `fnp-python` debt recorded in the artifact logs.
- Bounded UBS on the changed Rust file exited nonzero from broad existing
  `fnp-python` inventory; `git diff --check` passed.

Decision:
- Keep the exact-contiguous-f64 `einsum("ij->")` scalar builder fast path.
- Treat this single-operand reduce-all row as a current measured win rather than
  an active gap.
- Recheck row reductions separately before acting on the candidate-run
  row-guard loss; this patch does not alter that branch's source path.

---

## 2026-06-20 Linalg Symmetric Spectral / Batch Eigvalsh Bold-Verify

| Area | Score | Verdict |
|---|---:|---|
| `batch_eigvalsh` 64x128x128 and 16x256x256 rows | 9/10 | Release-ready current win |
| `cond_nxn` 128 and 512 exact-symmetric rows | 8/10 | Current win on this worker |
| `cond_nxn` 64 and 256 exact-symmetric rows | 3/10 | Not release-ready; still slower than NumPy |
| `eigvalsh_nxn/128` | 2/10 | Not release-ready; 3.081x slower than NumPy |
| Lanczos/power extremal-cond shortcut | 0/10 | Rejected before source edit; clustered spectra made residuals too loose |

Evidence:
- Artifact: `tests/artifacts/perf/2026-06-20_linalg_cond_lanczos_cod_a/`.
- Same-worker `vmi1227854` current FNP/NumPy ratios:
  `eigvalsh_nxn/128 = 3.081x`,
  `cond_nxn/64 = 1.409x`,
  `cond_nxn/128 = 0.859x`,
  `cond_nxn/256 = 1.428x`,
  `cond_nxn/512 = 0.431x`,
  `batch_eigvalsh/64x128x128 = 0.577x`,
  `batch_eigvalsh/16x256x256 = 0.0057x`.
- The initial Python comparator file is retained but invalid because `rch exec`
  ran it locally; counted NumPy rows come from direct SSH on `vmi1227854`.
- Fresh QR profile passed and reported the current scaled-hypot values-only QR
  path remains 1.24x-1.25x faster than the old libm-hypot path.
- Production `crates/fnp-linalg/src/lib.rs` source remains unchanged.

Decision:
- Keep no new linalg source from this slice.
- Treat batch eigvalsh as a current measured win, not a gap.
- Route the remaining spectral work to a deeper reduction/eigensolver primitive:
  dsytrd-class blocked Householder, two-stage tridiagonalization, or a fully
  convergent tridiagonal eigensolver. Do not repeat sort, threshold,
  direct-extrema, or fixed-iteration extremal shortcuts for this class.

---

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
