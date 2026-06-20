# FrankenNumPy Release-Readiness Scorecard

This is a rolling gauntlet scorecard. It summarizes measured evidence for the
current verification slice and does not certify the whole project for release.

## 2026-06-20 - Linalg Kron Identity Verification Slice

Scope:
- Recent code-first pending backlog measured: `franken_numpy-ixs5y.236`.
- Crate: `fnp-linalg`.
- Reference: NumPy 2.3.5.
- Same-worker decision host: `hz2` / `hetzner2`.
- Evidence: `tests/artifacts/perf/2026-06-20_linalg_kron_identity_vs_numpy/`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | PASS | 2/2 measured rows faster than NumPy; speedups were 5.72x and 3.72x. |
| Noise discipline | PASS | FNP Criterion ran on `hz2`; NumPy comparator ran directly on `hz2` and reported host/version metadata. |
| Targeted correctness | PASS | `cargo test -p fnp-linalg kron_ -- --nocapture` passed 4 focused kron tests. |
| Crate compile health | PASS | Same no-source linalg tree passed `cargo check -p fnp-linalg --all-targets` in the preceding verification slice. |
| Clippy health | PASS | Same no-source linalg tree passed `cargo clippy -p fnp-linalg --all-targets -- -D warnings`. |
| Release build health | PASS | Same no-source linalg tree passed `cargo build -p fnp-linalg --release`. |
| Formatting health | KNOWN GAP | `cargo fmt --package fnp-linalg -- --check` reports broad pre-existing drift outside this no-source verification slice. |
| Evidence durability | PASS | Results recorded in `docs/NEGATIVE_EVIDENCE.md` and the per-run scorecard. |

Cluster score: **86 / 100**

Score rationale:
- +35 performance: both target rows beat NumPy decisively.
- +20 correctness: focused identity, fallback, scalar, and golden kron tests
  passed.
- +15 reproducibility: same-worker FNP and NumPy evidence.
- +15 ledger discipline: all rows, validation, retry predicate, and artifact
  path recorded.
- +3 no-source discipline: existing implementation verified without extra churn.
- -2 residual validation: format drift remains in untouched `fnp-linalg`
  regions and was not normalized in this slice.

Current release posture:
- `franken_numpy-ixs5y.236` is **measured keep**, not pending.
- Structured identity-RHS `kron_nxn` is ahead of NumPy for the measured block
  operator shapes; generic kron and other structured RHS classes remain separate
  future work.

## 2026-06-20 - Linalg Batched Column-Sum Norm Verification Slice

Scope:
- Recent code-first pending backlog measured: `franken_numpy-ixs5y.240`.
- Crate: `fnp-linalg`.
- Reference: NumPy 2.3.5.
- Same-worker decision host: `hz2` / `hetzner2`.
- Evidence: `tests/artifacts/perf/2026-06-20_linalg_batch_column_sum_vs_numpy/`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | PASS | 4/4 measured rows faster than NumPy; speedups ranged from 9.06x to 12.58x. |
| Noise discipline | PASS | FNP Criterion ran on `hz2`; NumPy comparator ran directly on `hz2` and reported host/version metadata. |
| Targeted correctness | PASS | `batch_matrix_norm_column_sum_direct_lane_fill_matches_per_lane_reference_bits` passed. |
| Crate compile health | PASS | `cargo check -p fnp-linalg --all-targets` passed through RCH after one worker-local `SIGILL` retry. |
| Clippy health | PASS | `cargo clippy -p fnp-linalg --all-targets -- -D warnings` passed through RCH. |
| Release build health | PASS | `cargo build -p fnp-linalg --release` passed through RCH. |
| Formatting health | KNOWN GAP | `cargo fmt --package fnp-linalg -- --check` reports broad pre-existing drift outside this no-source verification slice. |
| Evidence durability | PASS | Results recorded in `docs/NEGATIVE_EVIDENCE.md` and the per-run scorecard. |

Cluster score: **88 / 100**

Score rationale:
- +40 performance: all target rows beat NumPy decisively.
- +20 correctness: focused bit-preservation guard passed.
- +15 reproducibility: same-worker FNP and NumPy evidence, with cross-host and
  unsupported Python-offload attempts excluded from scoring.
- +15 ledger discipline: all rows, invalid probes, validation, retry predicate,
  and artifact path recorded.
- -2 residual validation: format drift remains in untouched `fnp-linalg`
  regions and was not normalized in this slice.

Current release posture:
- `franken_numpy-ixs5y.240` is **measured keep**, not pending.
- This slice further improves confidence in stacked matrix norm diagnostics; it
  is not project-wide release certification.

## 2026-06-20 - Linalg Batched Row-Sum Norm Verification Slice

Scope:
- Recent code-first pending backlog measured: `franken_numpy-ixs5y.239`.
- Crate: `fnp-linalg`.
- Reference: NumPy 2.3.5.
- Same-worker decision host: `hz1` / `hetzner1`.
- Evidence: `tests/artifacts/perf/2026-06-20_linalg_batch_row_sum_vs_numpy/`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | PASS | 4/4 measured rows faster than NumPy; speedups ranged from 7.29x to 12.40x. |
| Noise discipline | PASS | FNP Criterion ran on `hz1`; NumPy comparator ran directly on `hz1` and reported host/version metadata. |
| Targeted correctness | PASS | `batch_matrix_norm_row_sum_direct_lane_fill_matches_per_lane_reference_bits` passed. |
| Crate compile health | PASS | `cargo check -p fnp-linalg --all-targets` passed through RCH. |
| Clippy health | PASS | `cargo clippy -p fnp-linalg --all-targets -- -D warnings` passed through RCH. |
| Release build health | PASS | `cargo build -p fnp-linalg --release` passed through RCH. |
| Formatting health | KNOWN GAP | `cargo fmt -p fnp-linalg -- --check` reports broad pre-existing drift outside this no-source verification slice. |
| Evidence durability | PASS | Results recorded in `docs/NEGATIVE_EVIDENCE.md` and the per-run scorecard. |

Cluster score: **88 / 100**

Score rationale:
- +40 performance: all target rows beat NumPy decisively.
- +20 correctness: focused bit-preservation guard passed.
- +15 reproducibility: same-worker FNP and NumPy evidence, with the invalid
  `rch exec -- python3` comparator attempt excluded.
- +15 ledger discipline: all rows, validation, retry predicate, and artifact
  path recorded.
- -2 residual validation: format drift remains in untouched `fnp-linalg`
  regions and was not normalized in this slice.

Current release posture:
- `franken_numpy-ixs5y.239` is **measured keep**, not pending.
- This slice improves confidence in stacked matrix norm diagnostics; it is not
  project-wide release certification.

## 2026-06-19 - Ufunc Flatnonzero Rejection Slice

Scope:
- Recent code-first pending backlog measured: `franken_numpy-ixs5y.245` via
  cod-a verification bead `franken_numpy-ixs5y.260`.
- Crate: `fnp-ufunc`.
- Reference: NumPy 2.4.3.
- Same-host decision machine: `thinkstation1`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | MIXED, REJECT CANDIDATE | Candidate was 1.07x slower at 100k and 3.57x faster at 1M; the 100k regression failed the keep gate. |
| Final code performance | PASS | Post-revert serial sidecar path is 3.21x faster than NumPy at 100k and 3.18x faster at 1M. |
| Revert discipline | PASS | Removed the `.245` parallel flatnonzero production fast path after measurement. |
| Targeted correctness | PASS | Pre-revert and post-revert flatnonzero golden guards passed remotely through `rch`. |
| Crate compile health | PASS | `rch exec -- cargo check -p fnp-ufunc --all-targets` passed with `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`. |
| Clippy health | PASS | `rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings` passed. |
| Formatting health | KNOWN GAP | `cargo fmt --check` and `cargo fmt -p fnp-ufunc -- --check` still report broad pre-existing format drift outside this slice. |
| UBS health | KNOWN GAP | UBS reports the established broad `fnp-ufunc` inventory; no references to the touched flatnonzero lines were reported. |
| Evidence durability | PASS | Results recorded in `docs/NEGATIVE_EVIDENCE.md` and `tests/artifacts/perf/2026-06-19_ufunc_flatnonzero_vs_numpy/`. |

Cluster score: **78 / 100**

Score rationale:
- +20 correctness: candidate and post-revert golden guards passed.
- +18 performance: final code beats NumPy on both measured rows, but the
  measured candidate itself had a 100k regression.
- +15 reproducibility: same-host local FNP vs NumPy evidence is recorded, with
  remote correctness evidence kept separate.
- +15 ledger discipline: candidate loss/win, final-code rows, and retry
  predicate are recorded.
- +15 revert discipline: mixed/regressing production fast path removed.
- -5 residual validation: fmt and UBS still have broad pre-existing
  `fnp-ufunc` inventory outside this slice.

Current release posture:
- `franken_numpy-ixs5y.245` is **measured rejected for the parallel branch**, not
  pending.
- The serial exact int64 sidecar export path remains the accepted final code for
  this workload and is faster than NumPy in the measured rows.

## 2026-06-19 - Ufunc Extract Rejection Slice

Scope:
- Recent code-first pending backlog measured: `franken_numpy-ixs5y.244` via
  cod-a verification bead `franken_numpy-ixs5y.259`.
- Crate: `fnp-ufunc`.
- Reference: NumPy 2.4.3.
- Same-host decision machine: `thinkstation1`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | FAIL | Candidate was 2.18x slower at 100k and 1.22x slower at 1M. |
| Revert discipline | PASS | Removed the `.244` parallel extract production fast path after measurement. |
| Targeted correctness | PASS | Post-revert extract and boolean-index golden guards passed remotely through `rch`. |
| Crate compile health | PASS | `rch exec -- cargo check -p fnp-ufunc --all-targets` passed with `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`. |
| Clippy health | PASS | `rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings` passed. |
| Formatting health | KNOWN GAP | `cargo fmt --check` and `cargo fmt -p fnp-ufunc -- --check` still report broad pre-existing format drift in untouched regions. |
| Evidence durability | PASS | Results recorded in `docs/NEGATIVE_EVIDENCE.md` and `tests/artifacts/perf/2026-06-19_ufunc_extract_vs_numpy/`. |

Cluster score: **62 / 100**

Score rationale:
- +20 correctness: candidate and post-revert golden guards passed.
- +14 reproducibility: same-host local FNP vs NumPy evidence is recorded, with
  remote correctness/build evidence kept separate.
- +15 ledger discipline: reject, final-code gap, and retry predicate are recorded.
- +15 revert discipline: losing production fast path removed.
- -38 performance: the measured candidate lost to NumPy on both rows; the final
  serial path is still slower than NumPy at 1M.
- -4 formatting residual: package/workspace fmt drift remains outside this
  slice.

Current release posture:
- `franken_numpy-ixs5y.244` is **measured rejected**, not pending.
- `extract(condition, arr)` is acceptable only as the reverted serial path for
  this slice; the 1M sparse-mask row remains an open performance gap versus
  NumPy.

## 2026-06-19 - Random PCG Distribution Verification Slice

Scope:
- Recent code-first pending backlog measured: `franken_numpy-ixs5y.250` and
  `franken_numpy-ixs5y.253`.
- Crate: `fnp-random`.
- Reference: NumPy 2.4.3.
- Worker: `ovh-a` via `rch exec`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | PASS | 4/4 measured size rows faster than NumPy median; speedups ranged from 6.01x to 8.67x. |
| Noise discipline | PASS | Criterion `bencher` output recorded all rows with sample size 10 and the same worker for FNP and NumPy rows. |
| Targeted correctness | PASS | PCG gumbel/laplace stream-state guards and live NumPy oracle checks passed. |
| Crate bench compile health | PASS | `rch exec -- cargo check -p fnp-random --benches` passed with requested `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`. |
| Revert decision | PASS | No revert required; no measured row was neutral or regressed in this slice. |
| Evidence durability | PASS | Results recorded in `docs/NEGATIVE_EVIDENCE.md` with retry predicates and artifacts under `tests/artifacts/perf/2026-06-19_random_vs_numpy_pcg_distributions/`. |

Cluster score: **90 / 100**

Score rationale:
- +40 performance: all target rows beat NumPy by at least 6.01x.
- +20 correctness: stream-state guards and live NumPy oracle checks passed.
- +15 reproducibility: same worker and explicit target-dir rewrite recorded.
- +15 ledger discipline: every result and retry predicate recorded.
- -10 project-wide release gap: this is not a full workspace gauntlet, full
  conformance, or 10-round convergence run.

## 2026-06-19 - Ufunc Data-Movement Verification Slice

Scope:
- Recent code-first pending backlog measured: `franken_numpy-ixs5y.256` and
  `franken_numpy-ixs5y.258`.
- Crate: `fnp-ufunc`.
- Reference: NumPy 2.4.3.
- Worker: `thinkstation1` via `rch exec`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | PASS | 4/4 measured size rows faster than NumPy median; speedups ranged from 1.19x to 1.73x. |
| Noise discipline | PARTIAL PASS | 3/4 batched NumPy rows were at or near the 5% CV gate; 1M insert stayed noisy but still had NumPy minimum slower than FNP Criterion upper CI. |
| Targeted correctness | PASS | Both new golden SHA guards ran and passed with real test execution. |
| Crate compile health | PASS | `rch exec -- cargo check -p fnp-ufunc` passed with `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`. |
| Revert decision | PASS | No revert required; no measured row was neutral or regressed in this slice. |
| Evidence durability | PASS | Results recorded in `docs/NEGATIVE_EVIDENCE.md` with retry predicates. |

Cluster score: **82 / 100**

Score rationale:
- +35 performance: all target rows beat NumPy medians, with one weak-but-positive
  delete row and one noisy-but-positive insert row.
- +20 correctness: targeted golden guards passed.
- +15 reproducibility: same rch worker and explicit target dir recorded.
- +12 ledger discipline: every result, discarded noisy attempt, and retry
  predicate recorded.
- -18 project-wide release gap: this is not a full workspace gauntlet, full
  conformance, or 10-round convergence run.

Current release posture:
- `fnp-ufunc` data-movement cluster is **measured keep** for the verified rows.
- Project-wide release certification remains **not certified** from this slice
  alone; continue converting `code-first batch-test pending` beads into measured
  ledger entries before claiming global performance dominance.

## 2026-06-19 - Ufunc Compress Rejection Slice

Scope:
- Recent code-first pending backlog measured: `franken_numpy-ixs5y.249`.
- Crate: `fnp-ufunc`.
- Reference: NumPy 2.4.3.
- Same-host decision machine: `thinkstation1`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | FAIL | Candidate was 7.15x slower at 100k and 2.05x slower at 1M. |
| Revert discipline | PASS | Removed the `.249` parallel compress production fast path after measurement. |
| Targeted correctness | PASS AFTER GUARD FIX | Bitwise guard passed remotely after replacing a `NaN`-unsafe assertion; post-revert compress tests passed locally. |
| Crate compile health | PASS | `cargo check -p fnp-ufunc` passed with `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`. |
| Clippy health | PASS | `cargo clippy -p fnp-ufunc --all-targets -- -D warnings` passed after replacing an approximate `2/pi` literal with `std::f64::consts::FRAC_2_PI`. |
| Evidence durability | PASS | Results recorded in `docs/NEGATIVE_EVIDENCE.md` and `tests/artifacts/perf/2026-06-19_ufunc_selection_vs_numpy/`. |

Cluster score: **64 / 100**

Score rationale:
- +20 correctness: candidate guard passed after the test fix and post-revert tests passed.
- +14 reproducibility: same-host local FNP vs NumPy evidence is recorded, with remote routing evidence kept separate.
- +15 ledger discipline: reject, final-code gap, and retry predicate are recorded.
- +15 revert discipline: regressing production fast path removed.
- -36 performance: the measured candidate lost to NumPy and regressed local Criterion history.

Current release posture:
- `franken_numpy-ixs5y.249` is **measured rejected**, not pending.
- `compress(condition, axis=None)` remains an open performance gap versus NumPy.

## 2026-06-19 - Ufunc Count Nonzero Verification Slice

Scope:
- Recent code-first pending backlog measured: `franken_numpy-ixs5y.246`.
- Crate: `fnp-ufunc`.
- Reference: NumPy 2.4.3.
- Same-host decision machine: `thinkstation1`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | PASS AFTER NARROWING | Original 16k activation was 3.56x slower than NumPy at 100k; final threshold was 4.69x faster at 100k and 3.48x faster at 1M. |
| Revert discipline | PASS | The regressing 16k activation threshold was removed; final code raises activation to `1 << 19` and keeps 4096-element chunks for large rows. |
| Targeted correctness | PASS | `count_nonzero_f64_parallel_matches_serial_reference_and_golden_sha256` passed after the intentional threshold-fixture digest update. |
| Crate compile health | PASS | `cargo check -p fnp-ufunc` passed with `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`. |
| Clippy health | PASS | `cargo clippy -p fnp-ufunc --all-targets -- -D warnings` passed. |
| Formatting health | WARN | `cargo fmt --check` still reports broad pre-existing workspace formatting drift outside this slice. |
| Evidence durability | PASS | Results recorded in `docs/NEGATIVE_EVIDENCE.md` and `tests/artifacts/perf/2026-06-19_ufunc_count_nonzero_vs_numpy/`. |

Cluster score: **84 / 100**

Score rationale:
- +34 performance: the final code beats NumPy on both measured rows, but the
  first measurement found a real 100k regression that had to be narrowed away.
- +20 correctness: targeted golden guard passed after the intentional fixture
  digest update.
- +15 reproducibility: same-host FNP and NumPy timings and explicit target dir
  are recorded.
- +15 ledger discipline: every win, loss, weakened neutral-ish correction, and
  retry predicate is recorded.
- -16 project-wide release gap: this is one verified pending optimization, not a
  full workspace gauntlet or global release certification.

Current release posture:
- `franken_numpy-ixs5y.246` is **measured keep after narrowing**, not pending.
- Continue converting the remaining batch-test backlog into measured rows before
  claiming broader `fnp-ufunc` performance readiness.

## 2026-06-19 - Ufunc Argwhere Verification Slice

Scope:
- Recent code-first pending backlog measured: `franken_numpy-ixs5y.248`.
- Crate: `fnp-ufunc`.
- Reference: NumPy 2.4.3.
- Same-host decision machine: `thinkstation1`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | PASS | Final code was 3.04x faster at 512x512 and 4.79x faster at 1024x1024. |
| Noise discipline | PASS WITH CAVEAT | NumPy CV was above 10%, but NumPy minimum still exceeded the FNP Criterion upper bound on both rows. |
| Targeted correctness | PASS | `argwhere_f64_parallel_matches_serial_reference_and_golden_sha256` passed. |
| Crate compile health | PASS | `cargo check -p fnp-ufunc` passed with `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`. |
| Clippy health | PASS | `cargo clippy -p fnp-ufunc --all-targets -- -D warnings` passed. |
| Formatting health | WARN | `cargo fmt --check` still reports broad pre-existing workspace formatting drift outside this slice. |
| Evidence durability | PASS | Results recorded in `docs/NEGATIVE_EVIDENCE.md` and `tests/artifacts/perf/2026-06-19_ufunc_argwhere_vs_numpy/`. |

Cluster score: **86 / 100**

Score rationale:
- +38 performance: both measured rows clear NumPy by more than 3x.
- +18 correctness: targeted interleaved-coordinate golden guard passed.
- +15 reproducibility: same-host FNP and NumPy timings and explicit target dir
  are recorded.
- +15 ledger discipline: every row and retry predicate is recorded.
- -14 project-wide release gap: this is one verified pending optimization, not a
  full workspace gauntlet or global release certification.

Current release posture:
- `franken_numpy-ixs5y.248` is **measured keep**, not pending.
- Continue converting the remaining batch-test backlog into measured rows before
  claiming broader `fnp-ufunc` performance readiness.

## 2026-06-20 - Linalg Column Norm Reject Slice

Scope:
- Parent bead measured: `franken_numpy-ixs5y`.
- Crate: `fnp-linalg`.
- Worker: `hz2`.
- Artifact: `tests/artifacts/perf/2026-06-20_linalg_column_norm_prefilter_stack256/`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | FAIL | Existing final code still loses to NumPy on the 256-1024 `ord=1/-1` matrix norm rows; the rejected candidates were 1.047x-1.848x NumPy time. |
| Candidate vs current FNP | FAIL | Whole-matrix NaN prefilter regressed all target rows; stack256-only regressed `neg_one/512` by 1.056x and `one/1024` by 1.018x while giving only a small `neg_one/256` gain. |
| Targeted correctness | PASS | `matrix_norm_column_reduction_matches_strided_reference_bits` passed for both candidates. |
| Revert discipline | PASS | Both production source changes were removed after measurement. |
| Evidence durability | PASS | No-ship table is recorded in `docs/NEGATIVE_EVIDENCE.md` and the artifact scorecard. |

Cluster score: **54 / 100**

Score rationale:
- +20 correctness: focused column-reduction reference test passed.
- +14 reproducibility: same-worker `hz2` FNP and NumPy rows are recorded with crate-scoped RCH commands.
- +15 ledger discipline: both failed levers and retry predicates are recorded.
- +15 revert discipline: no regressing source remains in production.
- -10 performance: no kept win; the 256-1024 column-norm gap remains open.

Current release posture:
- The column-norm residual remains **open**.
- Do not retry whole-matrix NaN prefilter or stack256-only. A future keep needs SIMD or strip-mined multi-column accumulation with same-worker proof against NumPy and no target-row regressions.

## 2026-06-20 - Python Einsum Diagonal Keep Slice

Scope:
- Parent bead measured: `franken_numpy-ixs5y`.
- Crate: `fnp-python`.
- Worker proof: final remote bench on `vmi1227854`; local baseline/final in `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Artifact: `tests/artifacts/perf/2026-06-20_python_einsum_diag_cod_a/`.

| Gate | Result | Evidence |
|---|---|---|
| Head-to-head performance vs NumPy | PASS FOR DIAGONAL | Local final diagonal is 0.808x NumPy time; rch final diagonal is 0.905x NumPy time. |
| Candidate vs current FNP | PASS | Local final diagonal is 0.193x baseline FNP time; local trace is 0.830x baseline FNP time. |
| Negative evidence discipline | PASS | Two intermediate candidates and the rch trace residual are recorded in `docs/NEGATIVE_EVIDENCE.md`. |
| Targeted correctness | PASS | `rch exec -- cargo test -p fnp-python --test conformance_einsum` passed 28/28. |
| Crate compile health | PASS WITH WARNINGS | `cargo check -p fnp-python --lib --bench criterion_python_surface` passed locally and `rch exec -- cargo build -p fnp-python --release` passed on `vmi1149989`; pre-existing warnings remain. |
| Formatting health | WARN | `cargo fmt -p fnp-python -- --check` reports broad pre-existing formatting drift; no formatter was run because it would rewrite unrelated files. |
| Broader all-target health | WARN | `cargo check -p fnp-python --benches` reaches unrelated pre-existing lib-test call-site drift. |
| Evidence durability | PASS | Baseline, candidates, rch conformance, final rch bench, and artifact scorecard are stored under the artifact directory. |

Cluster score: **82 / 100**

Score rationale:
- +32 performance: the target diagonal row now beats NumPy locally and on rch; trace improves locally but remains slower than NumPy on the rch worker.
- +18 correctness: focused einsum conformance and writable-view golden tests passed.
- +15 reproducibility: local baseline/final and final rch paired FNP-vs-NumPy rows are recorded.
- +15 ledger discipline: every candidate and residual loss is recorded.
- +2 integration hygiene: compile passes for the non-test library and exact bench target, but pre-existing warnings and drift block broader clean claims.
- -18 project-wide release gap: this is one verified Python-boundary keep, not a full workspace release certification.

Current release posture:
- `fnp_einsum_diag_f64_4000` is **measured keep**, not pending.
- The rch trace residual is **not closed**; future work should only target trace if it removes remaining scalar construction or view/fallback overhead without weakening diagonal writability semantics.
