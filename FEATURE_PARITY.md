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
| Strict/hardened policy split | in_progress | strict+hardened and adversarial runtime policy suites green, wire-unknown inputs fail-closed, override gate audited, JSONL evidence logs emitted | wire policy enforcement into io/ufunc execution paths |
| Security/compatibility threat matrix gating | in_progress | threat matrix + allowlist + executable control map machine-validated by `security_contracts` suite | enforce this suite as blocking CI gate and expand packet-specific threat controls |
| Ufunc arithmetic/reduction | in_progress | broadcasted binary ops + reduction core implemented; differential suite green against captured oracle corpus | increase corpus breadth and run against full NumPy oracle environment |
| RNG deterministic streams | in_progress | packet `FNP-P2C-007` differential/metamorphic/adversarial fixtures wired; `rng_differential_report.json` green (7/7); packet readiness `status=ready` | broaden generator/distribution matrix and enforce real NumPy oracle path in CI |
| NPY/NPZ format parity | in_progress | packet `FNP-P2C-009` differential/metamorphic/adversarial fixtures wired; `io_differential_report.json` green (8/8); packet readiness `status=ready` | expand parser/writer edge corpus and broaden hostile archive coverage |
| Linalg first-wave | in_progress | packet `FNP-P2C-008` differential/metamorphic/adversarial fixtures wired; `linalg_differential_report.json` green (8/8); packet readiness `status=ready` | expand backend/solver tolerance matrix and increase oracle-comparison breadth |
| RaptorQ artifact durability | in_progress | sidecar + scrub + decode proof artifacts generated for conformance and benchmark bundles | integrate generation/verification into CI and expand recovery matrix |

## Required Evidence Per Family

1. Differential fixture report.
2. Edge-case/adversarial test report.
3. Benchmark delta report (for perf-sensitive families).
4. Strict/hardened divergence report.
5. RaptorQ sidecar manifest + scrub/decode proof (or explicit defer note).

## Current Gaps

1. Oracle capture is now running against `system` NumPy (local `uv` Python 3.14 venv), but legacy-vendored NumPy parity runs are not yet established as a regular gate.
2. Differential corpora for packet families (including 007/008/009) are still scoped and do not yet represent full legacy NumPy surface area.
3. Bench baseline exists but regression gate enforcement is not yet wired in CI.

## Near-Term Milestones

1. Expand `FNP-P2C-005` differential corpus to adversarial broadcast/reduction edges.
2. Add full NumPy oracle environment path in CI/container.
3. Expand `FNP-P2C-007`, `FNP-P2C-008`, and `FNP-P2C-009` toward full legacy-matrix parity breadth (differential + metamorphic + adversarial).
4. Promote sidecar/scrub/decode checks to blocking gate.
