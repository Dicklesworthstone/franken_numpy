# Performance Release Readiness Scorecard

Scope: rolling gauntlet verification of measured FrankenNumPy performance slices
against original NumPy.

## 2026-06-21 cod-b fnp-python Compress Mask Count/Compaction Keep

| Area | Score | Verdict |
|---|---:|---|
| `compress_f64_axis_none` vs NumPy | 9/10 | 2 wins, 0 losses; candidate ratios 0.363x and 0.498x |
| Revert discipline | 8/10 | Failed first 16-lane attempt was fixed before keep; no regression hunk retained |
| Focused conformance | 8/10 | Filtered `compress` shard passed 13/13; full shard's lone failure is unrelated `choose` parity |
| Release build | 8/10 | `cargo build -p fnp-python --release` passed through `rch` |
| Hygiene gates | 6/10 | UBS/fmt report broad pre-existing `fnp-python` debt; no broad cleanup mixed into this perf commit |

Evidence:
- Bead/directive: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-b`.
- Source: `crates/fnp-python/src/lib.rs`, flat f64 `compress` fast path and
  generic typed mask compactor.
- Artifact directory:
  `tests/artifacts/perf/2026-06-21_fnp_python_compress_cod_b/`.
- Baseline (`hz1`): `compress_f64_axis_none_100000` FNP/NumPy `1.123x`;
  `compress_f64_axis_none_1000000` FNP/NumPy `1.077x`.
- Candidate (`vmi1149989`, same process FNP vs NumPy): 100K row
  `62,745 ns` vs `172,737 ns` (`0.363x`); 1M row `883,588 ns` vs
  `1,773,287 ns` (`0.498x`).
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`; no new
  `.scratch` worktree.

Decision:
- Release-ready for this exact flat f64 `compress(axis=None)` row.
- Cross-worker candidate-vs-baseline movement is not used as proof; the proof is
  the candidate head-to-head ratio against NumPy in the same Criterion process.
- Next target should be a current measured loss, not another pass over the
  already-fixed 8-lane branch.

---

## 2026-06-21 cod-a fnp-python 2-D Linalg Delegate Criterion Recheck

| Area | Score | Verdict |
|---|---:|---|
| 2-D `eigvalsh`/`eigh`/`cholesky` delegate rows | 8/10 | 2 wins, 0 losses, 4 neutral; old dense Python-surface loss remains closed |
| `matrix_power` boundary rows | 5/10 | `n=0` parity; `n=1` exposes a 2.407x micro-dispatch loss |
| Guard linalg boundary rows | 9/10 | Batch `slogdet`/`inv`/`solve` still dominate NumPy; batched Cholesky stays parity/win |
| Revert discipline | 9/10 | Kept benchmark rows only; no production edit while `fnp-python/src/lib.rs` is peer-dirty |
| Focused conformance | 9/10 | `conformance_linalg*` release shards passed 69/69 |

Evidence:
- Bead/directive: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-a`.
- Added exact 2-D delegate rows to
  `crates/fnp-python/benches/criterion_python_surface.rs`.
- Counted bench worker: `ovh-a`; command:
  `rch exec -- cargo bench -p fnp-python --bench criterion_python_surface --
  python_linalg_boundary --output-format bencher`.
- New delegate ratios: `eigvalsh` n=200 `1.002x`, `eigvalsh` n=800 `1.011x`,
  `eigh` n=200 `0.886x`, `eigh` n=800 `0.996x`, `cholesky` n=200 `0.906x`,
  `cholesky` n=800 `0.997x`.
- `matrix_power(A, 0)` n=800 is `1.015x`; `matrix_power(A, 1)` n=800 is
  `2.407x`, but only `1,401 ns` versus NumPy `582 ns`.
- Same run guard score across all linalg boundary pairs:
  **11 wins / 1 loss / 9 neutral**.
- Focused conformance used `rch` with no admissible worker and fell back locally,
  still using `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`:
  `conformance_linalg` 1/1, `conformance_linalg_advanced` 29/29,
  `conformance_linalg_decomp` 39/39.

Decision:
- Treat dense 2-D `eigvalsh`/`eigh`/`cholesky` as release-ready for the Python
  surface; do not reopen native-kernel work for those exact ndarray rows.
- Keep no production source change from this pass. The remaining measured loss
  is a narrow `matrix_power(A, 1)` wrapper dispatch floor and should wait until
  `crates/fnp-python/src/lib.rs` is free from peer-owned compress work.

---

## 2026-06-21 cod-a fnp-python Batch Inv/Solve Current Recheck

| Area | Score | Verdict |
|---|---:|---|
| Python `inv` batch rows vs NumPy | 9/10 | 3 wins, 0 losses, 0 neutral |
| Python `solve` guard rows vs NumPy | 9/10 | 2 wins, 0 losses, 0 neutral |
| Revert discipline | 9/10 | Rejected source-kernel edit; kept only benchmark rows |
| Focused conformance | 8/10 | `conformance_linalg` 1/1 passed on `ovh-a` |
| Same-process comparator freshness | 8/10 | FNP and NumPy ran inside the same Criterion bench process |

Evidence:
- Bead/directive: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-a`.
- Added `fnp_inv`/`numpy_inv` rows to
  `crates/fnp-python/benches/criterion_python_surface.rs`.
- Counted `inv` worker: `ovh-a`; command:
  `rch exec -- cargo bench -p fnp-python --bench criterion_python_surface --
  inv_f64 --output-format bencher`.
- `inv` FNP/NumPy ratios: batch8192 4x4 `0.155x`, batch64 128x128
  `0.067x`, batch16 256x256 `0.134x`.
- Counted `solve` guard worker: `vmi1149989`; command:
  `rch exec -- cargo bench -p fnp-python --bench criterion_python_surface --
  solve_f64_batch8192_4x4 --output-format bencher`.
- `solve` FNP/NumPy ratios: vector RHS `0.231x`, matrix RHS `0.252x`.

Decision:
- Mark the previous batch `inv` / `solve` light-lane loss routing as stale for
  the Python-boundary API rows measured here.
- No source kernel edit. A generated direct small-N inverse/solve path would be
  premature without a fresh same-process NumPy loss.
- Route future BOLD-VERIFY work to a current measured residual outside this
  closed slice (`eigvalsh_nxn/128`, architectural `sqrt` zero-init, or
  peer-owned Python wrapper lanes).

---

## 2026-06-21 cod-b fnp-linalg Matrix Norm Current Recheck

| Area | Score | Verdict |
|---|---:|---|
| Current `matrix_norm_nxn_orders` 1/-1 rows | 8/10 | Current head wins all six checked rows versus prior direct NumPy and local routing comparator |
| Revert discipline | 9/10 | No source hunk attempted; avoided the already-rejected scalar strip-mine family |
| Focused conformance/build | 8/10 | Focused column-reduction bit test and `fnp-linalg` release build passed through `rch` |
| Same-host NumPy comparator freshness | 4/10 | Fresh same-host Python was blocked by SSH auth; local comparator is routing-only |

Evidence:
- Bead: `franken_numpy-ixs5y.281`; agent `YellowElk` / `cod-b`.
- Current Rust bench worker: `vmi1152480`; command:
  `rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg
  'matrix_norm_nxn_orders/(one|neg_one)/(256|512|1024)'`.
- Current FNP medians: `one/256` 7,743 ns, `neg_one/256` 5,207 ns,
  `one/512` 26,211 ns, `neg_one/512` 25,737 ns, `one/1024` 99,936 ns,
  `neg_one/1024` 98,382 ns.
- Against the prior direct `hz2` NumPy rows, current FNP/NumPy ratios are
  `0.279x`, `0.184x`, `0.253x`, `0.250x`, `0.252x`, and `0.250x`.
- SSH to the selected worker was denied, and `rch exec -- python3` runs locally
  for non-compilation commands, so the fresh `thinkstation1` NumPy 2.4.3 ratios
  are recorded only as cross-host routing evidence.

Decision:
- Keep no source change. Mark the previous matrix-norm 1/-1 column-reduction
  gap as stale at current head.
- Route future work to a fresh measured loss, not another scalar strip-mine or
  allocation-only matrix-norm retune.

---

## 2026-06-21 cod-a fnp-linalg Spectral Cond No-Ship Recheck

| Area | Score | Verdict |
|---|---:|---|
| `cond_nxn/128` target gap | 2/10 | Still 1.115x slower than NumPy after candidate |
| `eigvalsh_nxn/128` adjacent gap | 2/10 | Still 1.820x slower than NumPy after candidate |
| Guard rows already winning vs NumPy | 7/10 | `cond_nxn` 64/256/512 stayed wins, but this does not close the target |
| Revert discipline | 9/10 | Scan/sort elision source was reverted after neutral target result |
| Focused conformance | 8/10 | `cond_p_spectral_symmetric` focused release test passed |

Evidence:
- Bead/directive: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-a`.
- Counted worker: `hz2`; candidate ran directly in existing warm RCH target
  `.rch-target-hz2-pool-f4ecbc5a8032ed7eb8c61438ab6b2cc8`.
- Current target baseline: `cond_nxn/128 = 1,242,314 ns`; NumPy `1,110,135 ns`;
  current FNP/NumPy `1.119x`.
- Candidate target: `cond_nxn/128 = 1,237,760 ns`; candidate/current `0.996x`;
  candidate/NumPy `1.115x`.
- Adjacent `eigvalsh_nxn/128`: candidate `1,359,806 ns` vs NumPy `747,108 ns`,
  `1.820x`.

Decision:
- No release-ready improvement from this slice.
- Keep no source. Route the next spectral attempt to a deeper reduction or
  eigensolver primitive, not scan elision or eigenvalue postprocessing.

---

## 2026-06-21 cod-a fnp-random PCG Current Recheck

| Area | Score | Verdict |
|---|---:|---|
| Current PCG head-to-head ratio-vs-NumPy | 9/10 | 10 wins, 0 losses, 0 neutral rows |
| `Generator::bytes` stale-gap closure | 9/10 | Both byte rows now win after `.265` direct final-buffer append/fill |
| Revert discipline | 9/10 | `.257` intermediate word-vector bytes path remains rejected and absent |
| Focused conformance/build | 8/10 | Per-crate `fnp-random` gates rerun for this docs recheck |

Evidence:
- Bead: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-a`.
- RCH worker for counted bench: `vmi1152480`; command:
  `rch exec -- cargo bench -p fnp-random --bench random_vs_numpy --
  --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format
  bencher`.
- Ratio table: raw `fill_u64` 0.390x and 0.460x, `Generator::bytes` 0.522x
  and 0.268x, `gumbel` 0.299x and 0.172x, `laplace` 0.314x and 0.139x,
  full-range `uint8` 0.932x and 0.293x.

Decision:
- Mark the previous "current `Generator::bytes` parity/perf gap" as stale.
- No source change and no revert in this pass.
- Route future BOLD-VERIFY work to active measured losses outside this closed PCG
  cluster.

---

## 2026-06-21 cod-a fnp-python Linalg Boundary Reverify

| Area | Score | Verdict |
|---|---:|---|
| Python linalg boundary ratio-vs-NumPy | 9/10 | 6 wins, 0 losses, 2 neutral rows |
| Delegate behavior boundary | 9/10 | Exact 2-D LAPACK-shaped ndarray calls delegated; batched/native winning paths preserved |
| Focused conformance | 8/10 | `conformance_linalg` 1/1 and `conformance_linalg_decomp` 39/39 pass; advanced shard 28/29 with only missing SciPy |
| Current dirty-worktree independence | 6/10 | Later filtered rerun blocked by unowned `fnp-ufunc` unsafe edit, not by linalg delegate behavior |

Evidence:
- Bead: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-a`.
- RCH worker for counted bench: `vmi1149989`; command:
  `rch exec -- cargo bench -p fnp-python --bench criterion_python_surface --
  python_linalg_boundary --sample-size 10 --measurement-time 2 --warm-up-time 1
  --output-format bencher`.
- Ratio table: `slogdet` 0.331x, `solve` vec 0.367x, `solve` mat2 0.469x,
  `cholesky` 4x4 1.010x, 8x8 0.870x, 16x16 0.853x, 32x32 0.919x,
  64x64 0.989x.
- Counted conformance: `conformance_linalg` 1/1 PASS; `conformance_linalg_decomp`
  39/39 PASS. `conformance_linalg_advanced` passed 28/29 and stopped only because
  `solve_triangular_complex` imports `scipy`, which was not installed on the
  worker.

Decision:
- Mark the previous code-only 2-D dense-linalg delegate rows as superseded by
  measured evidence for this focused boundary slice.
- No source change and no revert in this cod-a pass.
- Remaining target gaps are not this wrapper cliff; route future work to the
  measured kernel/batching losses (`batch_inv`, `batch_solve`, and native
  `eigvalsh_nxn/128`) with a different primitive.

---

## 2026-06-21 fnp-python matrix_power n=0/1 Boundary Delegate Code-Only Slice

| Area | Score | Verdict |
|---|---:|---|
| `matrix_power` exact ndarray `n=0` and `n=1` | 3/10 | Code-only pending bench |
| `matrix_power` powers `>=2` native path | 8/10 | Left unchanged |
| Focused conformance | 0/10 | Pending disk recovery |
| Fresh Criterion ratio-vs-NumPy | 0/10 | Pending disk recovery |

Evidence:
- Bead: `franken_numpy-ixs5y`; agent `YellowElk` / `cod-a`.
- Code-only lever: delegate exact NumPy ndarray boundary exponents `0` and `1`
  to `numpy.linalg.matrix_power` before extracting into Rust.
- Rationale: NumPy's boundary paths need only shape/dtype for `n=0` and return
  the asarray result for `n=1`; the previous wrapper paid an avoidable full
  matrix extract plus finite scan first.
- No new cargo build/bench/test/check was started under the 45G disk-low
  instruction.

Decision:
- Keep as a pending-bench code-only commit, not a measured win.
- Next admissible turn must run focused `fnp-python` Criterion rows and
  `matrix_power` conformance, then either score the ratio-vs-NumPy or revert on
  ~0 gain/regression.

---

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

## 2026-06-21 fnp-python 2-D Eigh Delegate Code-Only Candidate

| Area | Score | Verdict |
|---|---:|---|
| Existing 2-D `eigh` native path | 2/10 | Not release-ready; prior Python-surface native path loses 4.18x@200 and 4.05x@800 vs NumPy |
| Delegate candidate source | 5/10 | Code-only candidate is already on `main` via `76712a2b`; exact real 2-D square float `ndarray` routes to NumPy before extraction |
| Batched `batch_eigh` preservation | 6/10 | Source path untouched; guard benchmark still pending |
| Validation status | 1/10 | No direct cargo build/bench run and no focused conformance; targeted UBS failed on broad pre-existing inventory |

Evidence:
- Bead: `franken_numpy-ixs5y.278`.
- Existing measured native ratios: `np.linalg.eigh` real 2-D square float
  ndarray loses `4.18x` at n=200 and `4.05x` at n=800.
- Source change: remote `main` already applied the same metadata-only
  shape/dtype peek used by `eigvalsh` in `76712a2b`; matching 2-D inputs call
  `numpy.linalg.eigh(..., UPLO=UPLO)` before Rust extraction. The duplicate
  local hunk from bead `.278` was skipped during rebase.
- No after-ratio is recorded yet. Build, focused conformance, and head-to-head
  bench are pending the next disk-safe turn.
- Targeted UBS on the changed file set exited nonzero from existing
  `fnp-python` findings; it did not identify the new `eigh` hunk as the cause.
- Agent Mail reservation could not be recorded because the database corruption
  circuit breaker refused writes; coordination is via `docs/NEGATIVE_EVIDENCE.md`.

Decision:
- Keep the upstream code-only delegate candidate for the next validation slice.
- Do not mark the row release-ready until 2-D `eigh` after-ratios are measured
  and the batched native guard still routes correctly.
- If validation fails, revert the single wrapper hunk and keep the ledger entry
  as negative evidence.

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
| Current `Generator::bytes` direct final-buffer path | 9/10 | Fresh 2026-06-21 rerun: 2/0/0 wins vs NumPy |
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
| Head-to-head Criterion vs NumPy | Pass | 2026-06-21 current rerun is 10/0/0 vs NumPy |
| Head-to-head distribution Criterion vs NumPy | Pass | `tests/artifacts/perf/2026-06-19_random_vs_numpy_pcg_distributions/` |
| Negative-evidence ledger | Updated | `docs/NEGATIVE_EVIDENCE.md` |
| Required reverts | Done | Removed `.257` production word-fill path from `Generator::bytes` |

## Decision

Keep `.255`: final-code `fill_u64` is faster than NumPy by 3.72x at 100k u64 and 2.03x at 1M u64 on `hz1`.

Reject `.257`: pre-revert bytes word-fill was slower than NumPy by 1.64x at 100k bytes and 1.99x at 1M bytes on `ovh-a`. The production optimization was reverted. The later `.265` direct final-buffer append/fill path supersedes the old serial gap; the 2026-06-21 current rerun measured `Generator::bytes` as 0.522x NumPy at 100k and 0.268x NumPy at 1M.

Keep `.250`: final-code PCG64 gumbel is faster than NumPy by 6.01x at 100k f64 and 7.15x at 1M f64 on `ovh-a`.

Keep `.253`: final-code PCG64 laplace is faster than NumPy by 6.76x at 100k f64 and 8.67x at 1M f64 on `ovh-a`.
