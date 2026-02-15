# FNP-P2C-004 Rust Implementation Plan

packet_id: `FNP-P2C-004`  
subsystem: `NDIter traversal/index semantics`

## 1. Crate and Module Boundary Skeleton

| Crate | Planned module boundary | Responsibility | Public surface contract |
|---|---|---|---|
| `crates/fnp-iter` | `nditer_state` (packet-D planned boundary; crate currently stub) | iterator constructor/state machine for flags/op_flags/op_axes/itershape, reset/seek/index-mode transitions | `NDIteratorBuilder`, `NDIterator`, `reset`, `goto_index`, `goto_multi_index`, `set_iterrange` (planned) |
| `crates/fnp-iter` | `nditer_traversal_cursor` (packet-D planned boundary) | deterministic traversal cursor exposing `multi_index`, `c_index`, `f_index`, and external-loop compatible stepping | cursor/introspection APIs used by packet-E/F fixtures (planned) |
| `crates/fnp-iter` | `flatiter_surface` (packet-D/E planned boundary) | flat iterator indexing/assignment compatibility envelope with stable error taxonomy | flat indexing/assignment contract entrypoints (planned) |
| `crates/fnp-ndarray` | `shape_stride_core` (existing in `src/lib.rs`) | canonical shape/broadcast/stride calculus used by iterator planning and legality checks | `broadcast_shape`, `broadcast_shapes`, `contiguous_strides`, `element_count`, `NdLayout` |
| `crates/fnp-ufunc` | `iterator_adapter` (packet-D planned boundary) | migration seam from local odometer traversal to shared iterator cursor without semantic drift | adapter surface for `elementwise_binary`/reduction traversal (planned) |
| `crates/fnp-conformance` | `nditer_packet_suite` (packet-F planned boundary) | fixture-driven differential/metamorphic/adversarial coverage for iterator constructor/traversal/index/flatiter contracts | packet-F runner and packet-local fixture manifests (planned) |
| `crates/fnp-conformance` | workflow scenario integration (existing + packet-G extension) | strict/hardened replay scenarios linking differential fixtures and e2e scripts | packet-G scenario entries in workflow corpus (planned) |
| `crates/fnp-runtime` | policy/audit context (existing) | strict/hardened fail-closed mediation with reason-code and evidence logging for iterator policy decisions | `decide_and_record_with_context` integration from packet harnesses |

## 2. Implementation Sequence (D-Stage to I-Stage)

1. Land packet-D module skeleton in `fnp-iter` (`nditer_state`, traversal cursor, flatiter boundary), with explicit TODO gates for deferred parity debt.
2. Define constructor and state-transition error taxonomy aligned with contract rows `P2C004-R01`..`P2C004-R11`.
3. Wire `fnp-iter` planning APIs to `fnp-ndarray` shape/stride primitives for deterministic legality checks.
4. Introduce `fnp-ufunc` adapter seam so current traversal logic can migrate to iterator cursor incrementally without behavior changes in this bead.
5. Add packet-F conformance harness placeholders and fixture schemas for iterator differential/metamorphic/adversarial suites.
6. Add packet-G workflow scenario placeholders linking iterator fixture IDs to replay/e2e scripts and structured logging paths.
7. Connect iterator policy decisions to runtime audit context (`fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`).
8. Promote packet-H optimization artifacts only after baseline/profile/isomorphism evidence is available.
9. Close packet-I with parity summary + risk + durability sidecar/scrub/decode-proof artifacts.

## 3. Public Surface Contract Notes

- Existing authoritative primitives remain:
  - `broadcast_shape`
  - `broadcast_shapes`
  - `contiguous_strides`
  - `element_count`
  - `NdLayout`
- Packet-D additions must remain contract-driven and clean-room (no compatibility shims).
- Unknown or incompatible iterator semantics remain fail-closed in both strict and hardened modes.
- No `unsafe` pathways are introduced by the packet-D boundary skeleton.

## 4. Instrumentation Insertion Points

| Lane | Insertion point | Evidence artifact target |
|---|---|---|
| Unit/property | `crates/fnp-iter/src/lib.rs` + packet-E test additions | packet-E test evidence + reason-code matrix |
| Differential/metamorphic/adversarial | packet-F runner additions under `crates/fnp-conformance` + packet fixtures | packet-F parity reports and fixture manifests |
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

## 7. Compile-Safe Skeleton Validation

- Planning-stage validation rules:
  - no behavior-changing API migration is shipped in this bead;
  - packet boundaries and reason-code taxonomy remain internally consistent;
  - packet validator continues to report complete artifact fields for `FNP-P2C-004`.
- Validation command (offloaded via `rch`):  
  `rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-004`

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
  - last green `rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-004` report,
  - plus last green packet-linked security/workflow gate artifacts.
- If comparator fails, restore the prior planning baseline and re-run packet validation before continuing.
