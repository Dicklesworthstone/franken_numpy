# FNP-P2C-008 Rust Implementation Plan

packet_id: `FNP-P2C-008`  
subsystem: `linalg bridge first wave`

## 1. Crate and Module Boundary Skeleton

| Crate | Planned module boundary | Responsibility | Public surface contract |
|---|---|---|---|
| `crates/fnp-linalg` | `linalg_error_taxonomy` (packet-D planned boundary; crate currently skeletal) | deterministic mapping for `LinAlgError` families and packet reason-code taxonomy | stable error-class surface for linalg entrypoints (`P2C008-R01`..`R10`) |
| `crates/fnp-linalg` | `solver_ops` (packet-D planned boundary) | `solve`, `inv`, `pinv` precondition validation and deterministic failure routing | solver contract boundary (`P2C008-R02`, `P2C008-R08`) |
| `crates/fnp-linalg` | `factorization_ops` (packet-D planned boundary) | `cholesky`, `qr`, `svd` mode normalization and output-shape policy seams | factorization contracts (`P2C008-R03`, `P2C008-R04`, `P2C008-R05`) |
| `crates/fnp-linalg` | `spectral_ops` (packet-D planned boundary) | `eig`, `eigvals`, `eigh`, `eigvalsh` branch and convergence policy seams | spectral contract (`P2C008-R06`) |
| `crates/fnp-linalg` | `least_squares_ops` (packet-D planned boundary) | `lstsq` tuple-output class enforcement with tolerance policy gate | least-squares contract (`P2C008-R07`) |
| `crates/fnp-linalg` | `norm_det_rank_ops` (packet-D planned boundary) | `norm`, `det`, `slogdet`, `matrix_rank`, `pinv` legality/tolerance boundaries | norm/det/rank contract (`P2C008-R08`) |
| `crates/fnp-linalg` | `backend_bridge` + `lapack_adapter` (packet-D planned boundary) | backend seam, parameter validation, and deterministic backend error-hook mapping | backend bridge contract (`P2C008-R09`) |
| `crates/fnp-runtime` | policy/audit decision context (existing) | strict/hardened fail-closed mediation with reason-code and evidence logging for linalg packet decisions | `decide_and_record_with_context` integration from linalg suites |
| `crates/fnp-conformance` | `linalg_packet_suite` (packet-F planned boundary) | fixture-driven differential/metamorphic/adversarial linalg coverage (singular, non-convergence, tolerance-edge, backend anomalies) | packet-F linalg runner + fixture manifests (planned) |
| `crates/fnp-conformance` | workflow scenario integration (existing + packet-G extension) | strict/hardened replay scenarios linking linalg operations to packet workflows | packet-G linalg scenario entries in workflow corpus (planned) |

## 2. Implementation Sequence (D-Stage to I-Stage)

1. Land packet-D `fnp-linalg` module skeletons (`linalg_error_taxonomy`, `solver_ops`, `factorization_ops`, `spectral_ops`, `least_squares_ops`, `norm_det_rank_ops`, `backend_bridge`, `lapack_adapter`) with explicit TODO gates for deferred parity debt.
2. Define packet reason-code taxonomy aligned with contract rows `P2C008-R01`..`R10`.
3. Implement deterministic shape/ndim/square-ness legality checks at linalg entrypoints before backend dispatch.
4. Implement solver pathways (`solve`, `inv`, `pinv`) with deterministic singular/incompatible-system failure class handling.
5. Implement factorization and spectral mode gates (`qr`/`svd`/`cholesky`/`eig*`) with deterministic branch and output-class boundaries.
6. Implement least-squares tuple contract and tolerance-policy seam for `lstsq`.
7. Implement norm/det/rank/tolerance boundaries with deterministic class handling.
8. Implement backend seam policy (`backend_bridge`, `lapack_adapter`, error-hook mapping) with fail-closed unsupported backend states.
9. Add packet-F linalg conformance harness placeholders and fixture schemas for singular/non-convergence/tolerance/backend adversarial lanes.
10. Add packet-G workflow scenario placeholders linking linalg fixture IDs to replay/e2e scripts.
11. Wire packet linalg policy decisions into runtime audit context (`fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`).
12. Gate packet-H optimization work behind baseline/profile/isomorphism evidence.
13. Close packet-I with parity summary + risk + durability sidecar/scrub/decode-proof artifacts.

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
  - packet validator may remain `not_ready` until downstream E-I artifacts land.
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
