# FrankenNumPy Release-Readiness Scorecard

This is a rolling gauntlet scorecard. It summarizes measured evidence for the
current verification slice and does not certify the whole project for release.

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
