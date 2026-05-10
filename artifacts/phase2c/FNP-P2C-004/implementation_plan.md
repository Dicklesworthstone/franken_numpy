# FNP-P2C-004 Rust Implementation Plan

packet_id: `FNP-P2C-004`  
subsystem: `NDIter traversal/index semantics`

## 1. Crate and Module Boundary State

| Crate | Boundary | Responsibility | Public surface contract |
|---|---|---|---|
| `crates/fnp-iter` | `NditerPlan` / `Nditer` (packet-D core landed) | iterator constructor/state machine for shape/order/external-loop planning, reset/seek/index-mode transitions, and deterministic step emission | `NditerPlan::new`, `linear_index_to_multi_index`, `multi_index_to_linear_index`, `Nditer::reset`, `set_iterindex`, `set_multi_index`, `iterindex`, `multi_index`, `Iterator<Item = NditerStep>` |
| `crates/fnp-iter` | Python oracle bridge (packet-F baseline landed) | NumPy-backed comparison lane for local iterator state and external-loop chunk behavior | `nditer_python`, `nditer_python_with_interpreter`, `PythonNditer::steps`, `steps_from_iterindex`, `steps_from_multi_index` |
| `crates/fnp-iter` | flatiter/index-stream surface (packet-D/E landed core, expansion ongoing) | flat iterator indexing/assignment compatibility envelope with stable error taxonomy | `FlatIter`, `FlatIndex`, `ndindex`, `ndenumerate`, transfer-plan helpers, overlap-policy helpers |
| `crates/fnp-ndarray` | `shape_stride_core` (existing in `src/lib.rs`) | canonical shape/broadcast/stride calculus used by iterator planning and legality checks | `broadcast_shape`, `broadcast_shapes`, `contiguous_strides`, `element_count`, `NdLayout` |
| `crates/fnp-ufunc` | iterator-adjacent traversal adapters (migration ongoing) | migration seam from local odometer traversal to shared iterator cursor without semantic drift | current ufunc traversal remains behavior-preserving while packet integration expands |
| `crates/fnp-conformance` | iterator smoke and packet evidence (packet-F baseline landed) | fixture-driven differential/metamorphic/adversarial coverage for iterator constructor/traversal/index/flatiter contracts | NumPy oracle smoke coverage for iterindex/multi-index progression, seek, and external-loop chunks; packet-local fixture breadth remains parity debt |
| `crates/fnp-conformance` | workflow scenario integration (existing + packet-G extension) | strict/hardened replay scenarios linking differential fixtures and e2e scripts | packet-G scenario entries in workflow corpus; iterator-specific breadth expansion remains parity debt |
| `crates/fnp-runtime` | policy/audit context (existing) | strict/hardened fail-closed mediation with reason-code and evidence logging for iterator policy decisions | `decide_and_record_with_context` integration from packet harnesses |

## 2. Implementation Sequence (Current D/F Baseline to I-Stage)

1. Keep the landed packet-D `fnp-iter` core green: `NditerPlan`, `Nditer`, reset/seek, multi-index conversion, external-loop chunking, flatiter/index-stream helpers, and stable `NditerError` reason codes.
2. Maintain constructor and state-transition error taxonomy alignment with contract rows `P2C004-R01`..`P2C004-R11`; unsupported option combinations must fail closed rather than silently degrade.
3. Continue wiring iterator planning APIs through `fnp-ndarray` shape/stride primitives for deterministic legality checks.
4. Expand `fnp-ufunc` adapter coverage so local odometer traversal can migrate to iterator cursor usage incrementally without behavior changes.
5. Broaden packet-F conformance from the landed NumPy smoke checks into packet-local differential/metamorphic/adversarial fixtures for `op_axes`, `itershape`, no-broadcast, overlap, and flatiter assignment cases.
6. Extend packet-G workflow scenarios linking iterator fixture IDs to replay/e2e scripts and structured logging paths.
7. Connect iterator policy decisions to runtime audit context (`fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`) for hostile and replay lanes.
8. Keep packet-H optimization artifacts gated by baseline/profile/isomorphism evidence.
9. Close packet-I with parity summary + risk + durability sidecar/scrub/decode-proof artifacts.

## 3. Public Surface Contract Notes

- Existing authoritative primitives remain:
  - `NditerPlan`
  - `Nditer`
  - `NditerStep`
  - `nditer_python_with_interpreter`
  - `broadcast_shape`
  - `broadcast_shapes`
  - `contiguous_strides`
  - `element_count`
  - `NdLayout`
- Packet-D additions remain contract-driven and clean-room (no compatibility shims).
- Unknown or incompatible iterator semantics remain fail-closed in both strict and hardened modes.
- No `unsafe` pathways are introduced by the packet-D iterator core.

## 4. Instrumentation Insertion Points

| Lane | Insertion point | Evidence artifact target |
|---|---|---|
| Unit/property | `crates/fnp-iter/src/lib.rs` + `crates/fnp-iter/tests/metamorphic_iter.rs` + packet-E test additions | packet-E test evidence + reason-code matrix |
| Differential/metamorphic/adversarial | NumPy iterator smoke checks in `crates/fnp-conformance/tests/smoke.rs` + packet-F runner additions under `crates/fnp-conformance` + packet fixtures | packet-F parity reports and fixture manifests |
| E2E/replay | packet-G workflow scenario corpus + e2e scripts in `scripts/e2e/` | replay logs under `artifacts/logs/` |
| Runtime policy audit | `fnp-runtime` decision/audit logging from packet suites | security gate and policy evidence ledger outputs |

## 5. Structured Logging Emission Points

- Iterator constructor validation/rejection paths.
- Seek/index/mode-transition operations (`multi_index`, `index`, `iterindex`, `iterrange`).
- No-broadcast and overlap-policy decision points.
- Flatiter indexing/assignment validation branches.
- Runtime policy mediation events for strict/hardened fail-closed boundaries.

All emissions must include:

- `fixture_id`
- `seed`
- `mode`
- `env_fingerprint`
- `artifact_refs`
- `reason_code`

## 6. Artifact Boundary Plan

- Packet-local artifacts: `artifacts/phase2c/FNP-P2C-004/`
- Contract schemas and policy maps: `artifacts/contracts/`
- Replay/security logs: `artifacts/logs/`
- Durability artifacts: `artifacts/raptorq/` + packet-I packet-local outputs

## 7. Evidence Refresh Validation

- Planning-stage validation rules:
  - no behavior-changing API migration is shipped by evidence refreshes;
  - packet boundaries and reason-code taxonomy remain internally consistent;
  - packet validator continues to report complete artifact fields for `FNP-P2C-004`.
- Validation command (offloaded via `rch`):  
  `rch cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-004`

## 8. Alien Recommendation Contract Guardrails

- Graveyard mapping required for packet decisions: `alien_cs_graveyard.md` §0.4, §0.19, §6.12.
- FrankenSuite summary mapping required: `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19.
- EV gate for non-doc policy/optimization levers: ship only when `EV >= 2.0`, otherwise track as explicit deferred debt.
- Hotspot evidence prerequisite for optimization/policy shifts: baseline/profile artifact pair (or documented profiler-unavailable fallback).
- Isomorphism proof requirement for behavior-affecting changes: ordering/tie-break note + golden checks + reproducible before/after benchmark delta.

## 9. Rollback Handle

- Rollback command path:  
  `git restore --source <last-green-commit> -- artifacts/phase2c/FNP-P2C-004/implementation_plan.md`
- Baseline comparator to beat/restore:
  - last green `rch cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-004` report,
  - plus last green packet-linked security/workflow gate artifacts.
- If comparator fails, restore the prior planning baseline and re-run packet validation before continuing.
