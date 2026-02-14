# FNP-P2C-001 Rust Implementation Plan

packet_id: `FNP-P2C-001`  
subsystem: `shape/reshape legality`

## 1. Crate and Module Boundary Skeleton

| Crate | Planned module boundary | Responsibility | Public surface contract |
|---|---|---|---|
| `crates/fnp-ndarray` | `shape_contract` (existing core in `src/lib.rs`) | broadcast legality, `-1` inference, element-count and stride calculus | `broadcast_shape`, `broadcast_shapes`, `fix_unknown_dimension`, `contiguous_strides`, `NdLayout` |
| `crates/fnp-ndarray` | `reshape_api` (packet-D planned boundary) | canonical reshape legality entrypoint and wrapper parity path | `reshape_with_order(...) -> Result<NdLayout, ShapeError>` (planned) |
| `crates/fnp-ndarray` | `alias_guard` (packet-D/E planned boundary) | no-copy reshape legality + alias-sensitive transition checks | `can_reshape_without_copy(...) -> Result<bool, ShapeError>` (planned) |
| `crates/fnp-conformance` | `shape_stride_suite` (existing) | fixture-driven broadcast/stride checks against contract rows | `run_shape_stride_suite` |
| `crates/fnp-conformance` | `shape_packet_differential` (packet-F planned boundary) | strict/hardened differential/metamorphic/adversarial packet checks | packet-F runner entrypoint (planned) |
| `crates/fnp-runtime` | policy/log gate integration (existing) | strict/hardened decision policy + fail-closed envelope for hostile metadata | runtime decision/audit interfaces used by packet suites |

## 2. Implementation Sequence (D-Stage to I-Stage)

1. Introduce packet-D reshape API skeleton in `fnp-ndarray` around existing SCE primitives without changing established contracts.
2. Add alias-guard stubs and explicit TODO gates for no-copy parity closure (deferred to packet E/F evidence).
3. Wire packet-specific conformance fixture ingestion in `fnp-conformance` with deterministic reason-code taxonomy.
4. Add packet-G scenario hooks for replay/forensics logs using structured logging schema.
5. Promote packet-I parity/risk artifacts once E/F/G/H evidence is green.

## 3. Public Surface Contract Notes

- Existing public contract remains authoritative for current wave:
  - `broadcast_shape`, `broadcast_shapes`, `fix_unknown_dimension`, `contiguous_strides`, `NdLayout`.
- Planned packet-D additions must remain additive and backward-compatible within this clean-room project phase.
- No `unsafe` pathways are introduced by the packet-D boundary plan.

## 4. Instrumentation Insertion Points

| Lane | Insertion point | Evidence artifact target |
|---|---|---|
| Unit/property | `crates/fnp-ndarray/src/lib.rs` tests + packet-E additions | packet-E test evidence + reason codes |
| Differential/metamorphic/adversarial | `crates/fnp-conformance` packet-F shape runner (planned) | packet-F differential reports + fixture manifests |
| E2E/replay | `scripts/e2e/*` packet-G workflow hooks | packet-G replay logs under `artifacts/logs/` |
| Runtime policy audit | `fnp-runtime` decision + override logs | security gate JSONL + policy evidence ledger |

## 5. Structured Logging Emission Points

- Conformance fixture execution entrypoints (`fixture_id`, `seed`, `mode`).
- Runtime policy decisions for fail-closed/hardened mediation (`reason_code`, `artifact_refs`).
- E2E scenario runners (`env_fingerprint`, trace-linkable artifact handles).

## 6. Artifact Boundary Plan

- Packet-local artifacts remain under `artifacts/phase2c/FNP-P2C-001/`.
- Program-level controls remain under `artifacts/contracts/`.
- Replay/security logs remain under `artifacts/logs/`.
- RaptorQ durability artifacts remain under `artifacts/raptorq/` and packet-I outputs.

## 7. Compile-Safe Skeleton Validation

- Pseudo-compile validation rule for this planning stage:
  - No code-level API break is introduced in this bead.
  - Existing crate boundaries and public functions compile as-is.
  - Packet validator continues to parse current packet artifacts with `missing_fields=0`.
- Validation command used (offloaded via `rch`):  
  `rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-001`

## 8. Graveyard and FrankenSuite Mapping

- `alien_cs_graveyard.md` §0.19 (evidence ledger schema)
- `alien_cs_graveyard.md` §0.4 (decision-theoretic runtime contracts)
- `alien_cs_graveyard.md` §6.12 (property-based testing with shrinking)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19

## 9. Rollback Handle

If boundary planning introduces invalid assumptions, revert `artifacts/phase2c/FNP-P2C-001/implementation_plan.md` and restore the previous packet-D planning baseline tied to last green packet validation output.
