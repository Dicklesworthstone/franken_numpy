# FNP-P2C-008 Rust Implementation Plan

packet_id: `FNP-P2C-008`  
subsystem: `linalg bridge first wave`

## 1. Crate and Module Boundary Status

| Crate | Planned module boundary | Responsibility | Public surface contract |
|---|---|---|---|
| `crates/fnp-linalg` | error taxonomy/reason-code core (packet-D/E landed inline) | deterministic mapping for `LinAlgError` families and packet reason-code taxonomy | `LinAlgError`, `reason_code`, `LINALG_PACKET_REASON_CODES`, `LinAlgLogRecord` |
| `crates/fnp-linalg` | solver ops (packet-D/E landed inline) | `solve`, `inv`, `pinv` precondition validation and deterministic failure routing | `solve_2x2`, `inv_2x2`, `pinv_*`, `batch_inv` |
| `crates/fnp-linalg` | factorization ops (packet-D/E landed inline) | `cholesky`, `qr`, `svd` mode normalization and output-shape policy seams | `QrMode`, `qr_*`, `svd_*`, `cholesky_*`, output-shape validators |
| `crates/fnp-linalg` | spectral ops (packet-D/E landed inline) | `eig`, `eigvals`, `eigh`, `eigvalsh` branch and convergence policy seams | `eig_*`, `eigvals_*`, `eigh_*`, `eigvalsh_*`, batch spectral entrypoints |
| `crates/fnp-linalg` | least-squares ops (packet-D/E landed inline) | `lstsq` tuple-output class enforcement with tolerance policy gate | `LstsqOutputShapes`, `lstsq_svd`, `lstsq_2x2`, `lstsq_output_shapes` |
| `crates/fnp-linalg` | norm/det/rank/tolerance ops (packet-D/E landed inline) | `norm`, `det`, `slogdet`, `matrix_rank`, `pinv` legality/tolerance boundaries | `VectorNormOrder`, `MatrixNormOrder`, `det_*`, `slogdet_*`, `matrix_rank_*`, `validate_tolerance_policy` |
| `crates/fnp-linalg` | backend bridge/policy guard (packet-D/E landed inline) | backend seam, parameter validation, and deterministic backend error-hook mapping | `validate_policy_metadata`, bounded backend/tolerance constants, fail-closed reason codes |
| `crates/fnp-runtime` | policy/audit decision context (existing) | strict/hardened fail-closed mediation with reason-code and evidence logging for linalg packet decisions | `decide_and_record_with_context` integration from linalg suites |
| `crates/fnp-conformance` | linalg packet suite (packet-F landed) | fixture-driven differential/metamorphic/adversarial linalg coverage (singular, non-convergence, tolerance-edge, backend anomalies) | linalg differential/adversarial fixtures, oracle outputs, packet-F evidence |
| `crates/fnp-conformance` | workflow scenario integration (packet-G/H/I landed for current scope) | strict/hardened replay scenarios linking linalg operations to packet workflows | packet-008 workflow scenario artifacts and final evidence pack |

## 2. Implementation Sequence (D-Stage to I-Stage)

1. Keep landed packet-D/E `fnp-linalg` error taxonomy, solver, factorization, spectral, least-squares, norm/det/rank, backend-policy, reason-code, and structured-log boundaries green.
2. Maintain packet reason-code taxonomy alignment with contract rows `P2C008-R01`..`R10`.
3. Preserve deterministic shape/ndim/square-ness legality checks at linalg entrypoints before backend dispatch.
4. Preserve solver pathways (`solve`, `inv`, `pinv`) with deterministic singular/incompatible-system failure class handling.
5. Maintain factorization and spectral mode gates (`qr`/`svd`/`cholesky`/`eig*`) with deterministic branch and output-class boundaries.
6. Maintain least-squares tuple contract and tolerance-policy seam for `lstsq`.
7. Maintain norm/det/rank/tolerance boundaries with deterministic class handling.
8. Keep backend seam policy and unsupported backend states fail-closed.
9. Expand packet-F linalg fixture breadth for larger singular/non-convergence/tolerance/backend adversarial lanes.
10. Expand packet-G workflow scenarios where linalg fixtures should cover additional replay/e2e journeys.
11. Preserve packet linalg policy decisions in runtime audit context (`fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`).
12. Keep packet-H optimization work tied to baseline/profile/isomorphism evidence.
13. Keep packet-I parity summary + risk + durability sidecar/scrub/decode-proof artifacts ready.

## 3. Public Surface Contract Notes

- Packet-D additions remain clean-room and contract-driven (no compatibility shims).
- Unknown or incompatible linalg semantics remain fail-closed in strict and hardened modes.
- Solver/factorization/spectral/tolerance outcomes must remain deterministic for fixed inputs.
- No `unsafe` pathways are introduced by the packet-D planning boundary.

## 4. Instrumentation Insertion Points

| Lane | Insertion point | Evidence artifact target |
|---|---|---|
| Unit/property | packet-E tests in `fnp-linalg` solver/factorization/spectral/tolerance/backend modules | packet-E invariant logs + coverage artifacts |
| Differential/metamorphic/adversarial | packet-F linalg runner + fixture manifests in `crates/fnp-conformance` | packet-F parity/differential reports |
| E2E/replay | packet-G workflow scenario corpus + e2e scripts in `scripts/e2e/` | replay logs under `artifacts/logs/` |
| Runtime policy audit | linalg packet suites using runtime decision/audit integration | security gate + policy evidence ledger outputs |

## 5. Structured Logging Emission Points

- entrypoint shape/ndim legality decisions,
- solver singular/incompatible rejection and success branches,
- factorization/spectral mode-branch and convergence outcomes,
- least-squares tuple/tolerance policy decisions,
- norm/det/rank/pinv tolerance and axis/order validation branches,
- backend bridge/lapack adapter validation and error-hook mapping events,
- runtime policy mediation events for strict/hardened packet boundaries.

All emissions must include:

- `fixture_id`
- `seed`
- `mode`
- `env_fingerprint`
- `artifact_refs`
- `reason_code`

## 6. Artifact Boundary Plan

- Packet-local artifacts: `artifacts/phase2c/FNP-P2C-008/`
- Contract schemas and policy maps: `artifacts/contracts/`
- Replay/security logs: `artifacts/logs/`
- Durability artifacts: `artifacts/raptorq/` + packet-I packet-local outputs

## 7. Compile-Safe Skeleton Validation

- Planning-stage validation rules:
  - no behavior-changing linalg backend migration is shipped in this bead;
  - packet contract and reason-code taxonomy remain internally consistent;
  - packet validator remains `ready` for the landed E-I artifact set.
- Validation command (offloaded via `rch`):  
  `rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-008`

## 8. Alien Recommendation Contract Guardrails

- Graveyard mapping required for packet decisions: `alien_cs_graveyard.md` §0.4, §0.19, §6.12.
- FrankenSuite summary mapping required: `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19.
- EV gate for non-doc policy/optimization levers: ship only when `EV >= 2.0`; otherwise track explicit deferred debt.
- Hotspot evidence prerequisite for policy/optimization shifts: baseline/profile artifact pair (or documented profiler-unavailable fallback).
- Isomorphism proof requirement for behavior-affecting changes: ordering/tie-break note + golden checks + reproducible benchmark delta.

## 9. Rollback Handle

- Rollback command path:  
  `git restore --source <last-green-commit> -- artifacts/phase2c/FNP-P2C-008/implementation_plan.md`
- Baseline comparator to beat/restore:
  - last green `rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-008` report,
  - plus last green packet-linked security/workflow gate artifacts.
- If comparator fails, restore prior planning baseline and re-run packet validation before continuing.
