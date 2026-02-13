# FEATURE_PARITY

## Status Legend

- `not_started`
- `in_progress`
- `parity_green`
- `parity_gap`

## Parity Matrix

| Feature Family | Status | Current Evidence | Next Gate |
|---|---|---|---|
| Shape/stride/view semantics | in_progress | fixture-driven shape/stride suites green in `fnp-conformance` | broaden corpus against extracted `shape.c` edge cases |
| Broadcasting legality | in_progress | deterministic broadcast cases green | broaden mixed-rank/multi-axis corpus |
| Dtype promotion/casting | in_progress | scoped promotion table + fixture suite green | extract and verify broader cast table parity |
| Contract schema + artifact topology lock | in_progress | `phase2c-contract-v1` locked in `artifacts/contracts/` + packet readiness validator in `fnp-conformance` | populate packet artifact directories and enforce validator in CI |
| Strict/hardened policy split | in_progress | fail-closed runtime policy fixtures green | wire policy enforcement into io/ufunc execution paths |
| Ufunc arithmetic/reduction | in_progress | broadcasted binary ops + reduction core implemented; differential suite green against captured oracle corpus | increase corpus breadth and run against full NumPy oracle environment |
| RNG deterministic streams | not_started | none yet | implement `FNP-P2C-007` deterministic stream harness |
| NPY/NPZ format parity | not_started | none yet | implement `FNP-P2C-009` parser/writer fixtures |
| Linalg first-wave | not_started | none yet | implement `FNP-P2C-008` scoped linalg bridge |
| RaptorQ artifact durability | in_progress | sidecar + scrub + decode proof artifacts generated for conformance and benchmark bundles | integrate generation/verification into CI and expand recovery matrix |

## Required Evidence Per Family

1. Differential fixture report.
2. Edge-case/adversarial test report.
3. Benchmark delta report (for perf-sensitive families).
4. Strict/hardened divergence report.
5. RaptorQ sidecar manifest + scrub/decode proof (or explicit defer note).

## Current Gaps

1. Oracle capture is now running against `system` NumPy (local `uv` Python 3.14 venv), but legacy-vendored NumPy parity runs are not yet established as a regular gate.
2. Differential corpus is still small and does not yet represent the full extraction packet surface.
3. Bench baseline exists but regression gate enforcement is not yet wired in CI.

## Near-Term Milestones

1. Expand `FNP-P2C-005` differential corpus to adversarial broadcast/reduction edges.
2. Add full NumPy oracle environment path in CI/container.
3. Promote sidecar/scrub/decode checks to blocking gate.
