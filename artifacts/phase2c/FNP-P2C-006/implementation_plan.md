# FNP-P2C-006 Rust Implementation Plan

packet_id: `FNP-P2C-006`  
subsystem: `stride-tricks and broadcasting API`

## 1. Crate and Module Boundary Status

| Crate | Planned module boundary | Responsibility | Public surface contract |
|---|---|---|---|
| `crates/fnp-ndarray` | `shape_broadcast_core` (existing in `src/lib.rs`) | deterministic shape broadcast legality and merge rules | `broadcast_shape`, `broadcast_shapes`, `can_broadcast` |
| `crates/fnp-ndarray` | `layout_stride_core` (existing in `src/lib.rs`) | contiguous stride calculus + layout metadata ownership | `contiguous_strides`, `NdLayout` |
| `crates/fnp-ndarray` | `stride_tricks_api` (packet-D/E landed) | clean-room `broadcast_to`/`broadcast_arrays`/`as_strided` API-layer semantics | `NdLayout::broadcast_to`, `NdLayout::as_strided`, shape/broadcast validators |
| `crates/fnp-iter` | `iterator_axisdata` / transfer planning (packet-D core landed) | nditer-like construction, operand broadcast planning, no-broadcast handling, compatible stride introspection | `NditerPlan`, `Nditer`, operand broadcast plan structs, `TransferContext`, `select_transfer_loop` |
| `crates/fnp-conformance` | shape/stride suite + packet-F/G/H/I evidence (landed) | fixture-driven parity checks for shape/stride/stride-tricks behavior | shape-stride differential/metamorphic/adversarial fixtures, workflow scenario artifacts, final evidence pack |
| `crates/fnp-runtime` | policy audit integration (existing) | strict/hardened decision logging and fail-closed handling at packet boundaries | runtime decision/audit interfaces |

## 2. Implementation Sequence (D-Stage to I-Stage)

1. Keep the landed packet-D/E Rust boundaries for stride-tricks and iterator responsibilities green without introducing unsafe code.
2. Maintain `fnp-ndarray` `broadcast_to`/`as_strided` semantics aligned to packet contract rows.
3. Extend the landed `fnp-iter` axisdata/no-broadcast and compatible-stride planning core where additional stride-tricks integration fixtures expose gaps.
4. Expand packet-F conformance coverage for stride-tricks differential/metamorphic/adversarial breadth.
5. Maintain packet-G e2e stride-tricks + iterator replay scenarios with structured forensics logs.
6. Keep packet-I parity/risk/durability artifacts ready and linked to current final evidence.

## 3. Public Surface Contract Notes

- Current stable boundary:
  - `broadcast_shape`
  - `broadcast_shapes`
  - `contiguous_strides`
  - `NdLayout`
- Packet-D additions must be explicitly contract-driven (no compatibility shims).
- Writeability and overlap-risk semantics must stay policy-visible and reason-code tagged.

## 4. Instrumentation Insertion Points

| Lane | Insertion point | Evidence artifact target |
|---|---|---|
| Unit/property | `crates/fnp-ndarray/src/lib.rs` + packet-E additions | packet-E structured logs + property artifacts |
| Differential/metamorphic/adversarial | packet-F conformance hook expansion + stride-tricks fixtures | packet-F differential/parity reports |
| E2E/replay | packet-G workflow scenario scripts for stride/broadcast/iterator flows | replay logs under `artifacts/logs/` |
| Runtime policy audit | `fnp-runtime` decision + override logs | security gate + policy evidence ledger |

## 5. Structured Logging Emission Points

- Broadcast shape/reject paths (`fixture_id`, `seed`, `mode`, `reason_code`).
- Stride-tricks writeability/overlap-sensitive paths (`reason_code`, `artifact_refs`).
- Iterator no-broadcast and shape-introspection failures (`env_fingerprint`, `artifact_refs`).

## 6. Artifact Boundary Plan

- Packet artifacts under `artifacts/phase2c/FNP-P2C-006/`.
- Program controls under `artifacts/contracts/`.
- Replay/security logs under `artifacts/logs/`.
- Durability artifacts under `artifacts/raptorq/` and packet-I outputs.

## 7. Compile-Safe Skeleton Validation

- Planning-stage validation rules:
  - No API-breaking behavior changes in this bead.
  - Existing `fnp-ndarray`/conformance boundary remains compilable.
  - Packet validator maintains complete packet artifact field coverage as packet files are added.
- Validation command (offloaded via `rch`):  
  `rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-006`

## 8. Graveyard and FrankenSuite Mapping

- `alien_cs_graveyard.md` §0.19 (evidence ledger schema)
- `alien_cs_graveyard.md` §0.4 (decision-theoretic runtime contracts)
- `alien_cs_graveyard.md` §6.12 (property-based testing with shrinking)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19

## 9. Rollback Handle

If implementation boundaries prove invalid, revert `artifacts/phase2c/FNP-P2C-006/implementation_plan.md` and restore the previous packet-D planning baseline.
