# FNP-P2C-003 Rust Implementation Plan

packet_id: `FNP-P2C-003`  
subsystem: `strided transfer semantics`

## 1. Crate and Module Boundary Status

| Crate | Module boundary | Responsibility | Current surface contract |
|---|---|---|---|
| `crates/fnp-iter` | `transfer_selector` (packet-D core landed) | deterministic transfer-loop selection based on dtype/alignment/stride context | `TransferSelectorInput`, `select_transfer_class`, `TransferContext`, `select_transfer_loop` |
| `crates/fnp-iter` | `transfer_overlap_guard` (packet-D/E core landed) | overlap-aware direction/copy mediation and alias-policy checks | `overlap_copy_policy`, `NditerTransferFlags`, transfer decision reason-code mapping |
| `crates/fnp-iter` | `flatiter_transfer` (packet-D/E core landed) | flatiter read/write transfer semantics for integer/slice/fancy/bool indexing classes | `FlatIterIndex`, `resolve_flatiter_indices`, `validate_flatiter_read`, `validate_flatiter_write` |
| `crates/fnp-ufunc` | transfer traversal adapter seam (current) | current broadcast-odometer traversal remains behavior-preserving while broader migration to reusable transfer selectors stays residual integration debt | `elementwise_binary`/reduction traversal remains the comparator path for transfer-adjacent evidence |
| `crates/fnp-dtype` | `cast_policy_core` (existing in `src/lib.rs`) | cast compatibility policy primitives used by transfer gating | `promote`, `can_cast_lossless` |
| `crates/fnp-ndarray` | `shape_stride_core` (existing in `src/lib.rs`) | shape/broadcast/stride legality primitives used by transfer traversal planning | `broadcast_shape`, `broadcast_shapes`, `contiguous_strides`, `NdLayout` |
| `crates/fnp-conformance` | `transfer_packet_suite` (packet-E/F landed) | packet-specific unit/property, differential, metamorphic, and adversarial transfer fixtures and parity checks | `unit_property_evidence.json` and `differential_metamorphic_adversarial_evidence.json` |
| `crates/fnp-conformance` | workflow scenario integration (packet-G/H landed) | strict/hardened replay scenarios with transfer fixture linkage and optimization-isomorphism evidence | `workflow_scenario_packet003_*` artifacts and optimization profile evidence |
| `crates/fnp-runtime` | policy/audit decision context (existing) | strict/hardened fail-closed mediation with reason-code and artifact linkage for transfer decisions | `decide_and_record_with_context` integration from transfer harnesses |

## 2. Maintenance Sequence (Post-I Evidence Baseline)

1. Keep the landed packet-D transfer selector, overlap guard, and flatiter transfer boundaries in `fnp-iter` green.
2. Preserve the reason-code taxonomy and strict/hardened decision boundaries from `P2C003-R01`..`R10`.
3. Maintain transfer selector inputs against `fnp-dtype` cast policy and `fnp-ndarray` stride/broadcast primitives.
4. Treat broader `fnp-ufunc` migration to reusable transfer selectors as integration debt that requires behavior-isomorphism proof before landing.
5. Expand packet-E/F fixture breadth for grouped, subarray, and fixed-width string/unicode transfer families without weakening current gates.
6. Expand packet-G workflow scenario breadth for overlap, where-mask, and flatiter transfer journeys while preserving replay-complete structured fields.
7. Keep transfer policy decisions linked to runtime audit context fields (`fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`).
8. Gate future packet-H optimization changes behind fresh baseline/profile/isomorphism evidence artifacts.
9. Preserve packet-I parity summary, risk, durability sidecar, scrub report, and decode-proof linkage.

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
| Unit/property | packet-E additions in `fnp-iter`/`fnp-ufunc` transfer tests | `unit_property_evidence.json` |
| Differential/metamorphic/adversarial | packet-F transfer runner + fixture manifests in `crates/fnp-conformance` | `differential_metamorphic_adversarial_evidence.json` |
| E2E/replay | packet-G workflow scenario corpus + e2e scripts in `scripts/e2e/` | `e2e_replay_forensics_evidence.json` and `workflow_scenario_packet003_*` artifacts |
| Runtime policy audit | transfer suites using runtime decision/audit integration | security gate + policy evidence ledger outputs referenced by `final_evidence_pack.json` |

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

- Maintenance validation rules:
  - no behavior-changing transfer migration is shipped in this documentation bead;
  - packet contract/reason-code taxonomy remains internally consistent;
  - packet validator reports `FNP-P2C-003` as ready with zero missing artifacts, missing fields, or parse errors.
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
