# FNP-P2C-003 Rust Implementation Plan

packet_id: `FNP-P2C-003`  
subsystem: `strided transfer semantics`

## 1. Crate and Module Boundary Skeleton

| Crate | Planned module boundary | Responsibility | Public surface contract |
|---|---|---|---|
| `crates/fnp-iter` | `transfer_selector` (packet-D planned boundary; crate currently stub) | deterministic transfer-loop selection based on dtype/alignment/stride context | transfer selector API returning stable transfer-class decisions (planned) |
| `crates/fnp-iter` | `transfer_overlap_guard` (packet-D/E planned boundary) | overlap-aware direction/copy mediation and alias-policy checks | overlap policy interfaces + decision reason-code mapping (planned) |
| `crates/fnp-iter` | `flatiter_transfer` (packet-D/E planned boundary) | flatiter read/write transfer semantics for integer/slice/fancy/bool indexing classes | flatiter transfer APIs and error taxonomy hooks (planned) |
| `crates/fnp-ufunc` | `transfer_executor` (packet-D planned boundary) | migration seam from current broadcast-odometer traversal to reusable transfer-selector/guard stack | adapter seam for `elementwise_binary`/reduction transfer pathways (planned) |
| `crates/fnp-dtype` | `cast_policy_core` (existing in `src/lib.rs`) | cast compatibility policy primitives used by transfer gating | `promote`, `can_cast_lossless` |
| `crates/fnp-ndarray` | `shape_stride_core` (existing in `src/lib.rs`) | shape/broadcast/stride legality primitives used by transfer traversal planning | `broadcast_shape`, `broadcast_shapes`, `contiguous_strides`, `NdLayout` |
| `crates/fnp-conformance` | `transfer_packet_suite` (packet-F planned boundary) | packet-specific differential/metamorphic/adversarial transfer fixtures and parity checks | packet-F transfer runner entrypoints (planned) |
| `crates/fnp-conformance` | workflow scenario integration (existing + packet-G extension) | strict/hardened replay scenarios with transfer fixture linkage | packet-G transfer scenarios in workflow corpus (planned) |
| `crates/fnp-runtime` | policy/audit decision context (existing) | strict/hardened fail-closed mediation with reason-code and artifact linkage for transfer decisions | `decide_and_record_with_context` integration from transfer harnesses |

## 2. Implementation Sequence (D-Stage to I-Stage)

1. Land packet-D module skeleton for transfer selector/overlap guard/flatiter transfer boundaries in `fnp-iter`.
2. Define transfer reason-code taxonomy and strict/hardened decision boundaries from `P2C003-R01`..`R10`.
3. Wire transfer selector inputs to `fnp-dtype` cast policy and `fnp-ndarray` stride/broadcast primitives.
4. Introduce `fnp-ufunc` adapter seam so current traversal can incrementally migrate without behavior drift.
5. Add packet-F transfer fixture schemas and runner placeholders in `fnp-conformance`.
6. Add packet-G workflow scenario placeholders linking transfer fixture IDs + e2e scripts and artifact refs.
7. Connect transfer policy decisions to runtime audit context fields (`fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`).
8. Gate packet-H optimization work behind baseline/profile/isomorphism evidence artifacts.
9. Close packet-I with parity summary + risk + durability sidecar/scrub/decode-proof linkage.

## 3. Public Surface Contract Notes

- Existing authoritative primitives remain:
  - `promote`
  - `can_cast_lossless`
  - `broadcast_shape`
  - `broadcast_shapes`
  - `contiguous_strides`
  - `NdLayout`
- Packet-D transfer boundaries must remain clean-room and contract-driven (no compatibility shims).
- Unknown/incompatible transfer semantics remain fail-closed in both strict and hardened modes.
- No `unsafe` pathways are introduced by this packet-D planning boundary.

## 4. Instrumentation Insertion Points

| Lane | Insertion point | Evidence artifact target |
|---|---|---|
| Unit/property | packet-E additions in `fnp-iter`/`fnp-ufunc` transfer tests | packet-E transfer invariant logs + coverage artifacts |
| Differential/metamorphic/adversarial | packet-F transfer runner + fixture manifests in `crates/fnp-conformance` | packet-F parity/differential reports |
| E2E/replay | packet-G workflow scenario corpus + e2e scripts in `scripts/e2e/` | replay logs under `artifacts/logs/` |
| Runtime policy audit | transfer suites using runtime decision/audit integration | security gate + policy evidence ledger outputs |

## 5. Structured Logging Emission Points

- Transfer-loop selector decisions and rejects.
- Overlap mediation decisions (direction/copy/reject).
- Where-mask transfer branch outcomes.
- Same-value cast gate accept/reject paths.
- Flatiter transfer read/write validations.
- Runtime policy mediation events for strict/hardened transfer boundaries.

All emissions must include:

- `fixture_id`
- `seed`
- `mode`
- `env_fingerprint`
- `artifact_refs`
- `reason_code`

## 6. Artifact Boundary Plan

- Packet-local artifacts: `artifacts/phase2c/FNP-P2C-003/`
- Contract schemas and policy maps: `artifacts/contracts/`
- Replay/security logs: `artifacts/logs/`
- Durability artifacts: `artifacts/raptorq/` + packet-I packet-local outputs

## 7. Compile-Safe Skeleton Validation

- Planning-stage validation rules:
  - no behavior-changing transfer migration is shipped in this bead;
  - packet contract/reason-code taxonomy remains internally consistent;
  - packet validator continues reporting complete artifact fields for `FNP-P2C-003` (status may remain `not_ready` until downstream E-I artifacts land).
- Validation command (offloaded via `rch`):  
  `rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-003`

## 8. Alien Recommendation Contract Guardrails

- Graveyard mapping required for packet decisions: `alien_cs_graveyard.md` §0.4, §0.19, §6.12.
- FrankenSuite summary mapping required: `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19.
- EV gate for non-doc policy/optimization levers: ship only when `EV >= 2.0`, otherwise track as explicit deferred debt.
- Hotspot evidence prerequisite for policy/optimization shifts: baseline/profile artifact pair (or documented profiler-unavailable fallback).
- Isomorphism proof requirement for behavior-affecting changes: ordering/tie-break note + golden checks + reproducible benchmark delta.

## 9. Rollback Handle

- Rollback command path:  
  `git restore --source <last-green-commit> -- artifacts/phase2c/FNP-P2C-003/implementation_plan.md`
- Baseline comparator to beat/restore:
  - last green `rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-003` report,
  - plus last green packet-linked security/workflow gate artifacts.
- If comparator fails, restore prior planning baseline and re-run packet validation before continuing.
