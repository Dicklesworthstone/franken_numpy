# FrankenNumPy Release-Readiness Scorecard

This is a rolling gauntlet scorecard. It summarizes measured evidence for the
current verification slice and does not certify the whole project for release.

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
