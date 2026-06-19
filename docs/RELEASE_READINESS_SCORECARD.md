# FrankenNumPy Release-Readiness Scorecard

This is a rolling gauntlet scorecard. It summarizes measured evidence for the
current verification slice and does not certify the whole project for release.

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
