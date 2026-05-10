# FNP-P2C-001 Rust Implementation Plan and Current Evidence

packet_id: `FNP-P2C-001`  
subsystem: `shape/reshape legality`

## 1. Current Crate and Module Boundaries

| Crate | Current boundary | Responsibility | Public surface contract |
|---|---|---|---|
| `crates/fnp-ndarray` | `shape_contract` (existing core in `src/lib.rs`) | broadcast legality, `-1` inference, element-count and stride calculus | `broadcast_shape`, `broadcast_shapes`, `fix_unknown_dimension`, `contiguous_strides`, `NdLayout` |
| `crates/fnp-ndarray` | `NdLayout` view legality | alias-sensitive view construction, broadcasted zero-stride views, negative-stride bounds checks, and overlap detection | `NdLayout::as_strided`, `NdLayout::broadcast_to`, `NdLayout::has_internal_overlap` |
| `crates/fnp-conformance` | `shape_stride_suite` | fixture-driven broadcast/stride checks against contract rows with structured replay fields | `run_shape_stride_suite` |
| `crates/fnp-conformance` | packet-F shape/stride gates | strict/hardened differential, metamorphic, and adversarial packet checks | `shape_stride_packet006_f_suites_are_green`, test-contract gate, security gate |
| `crates/fnp-conformance` | packet-G workflow replay | strict/hardened replay and hostile guardrail scenarios for shape/reshape policy | `workflow_scenario_packet001_e2e.jsonl`, `run_workflow_scenario_gate` |
| `crates/fnp-runtime` | policy/log gate integration (existing) | strict/hardened decision policy + fail-closed envelope for hostile metadata | runtime decision/audit interfaces used by packet suites |

## 2. Implementation and Evidence State

1. Shape calculus primitives are landed in `fnp-ndarray`: element counts, right-aligned broadcast legality, `-1` reshape inference, contiguous stride derivation, and broadcast stride synthesis.
2. View legality is represented by `NdLayout` with `as_strided`, `broadcast_to`, negative-stride bounds checks, zero-stride broadcast handling, and internal-overlap/read-only policy.
3. Packet-E evidence is present in `unit_property_evidence.json`: 8 shape invariant unit tests, 49 broadcast pair property checks, 3 shape/stride fixture cases, and structured-log field coverage.
4. Packet-F evidence is present in `differential_metamorphic_adversarial_evidence.json`: 10/10 differential cases, 26/26 metamorphic checks, and 7/7 adversarial checks passing.
5. Packet-G evidence is present in `e2e_replay_forensics_evidence.json`: 400/400 workflow checks covering strict and hardened replay plus hostile guardrails.
6. Packet-I evidence is present in `final_evidence_pack.json`: strict and hardened parity signals are 1.0, no waivers are recorded, and packet validation is listed as a passing quality gate.
7. Remaining packet-local debt is breadth, not boundary scaffolding: expand exact NumPy no-copy reshape alias/view parity, workflow scenario breadth, and calibration thresholds as the full reshape/broadcast corpus grows.

## 3. Public Surface Contract Notes

- Existing public contract remains authoritative for current wave:
  - `broadcast_shape`, `broadcast_shapes`, `fix_unknown_dimension`, `contiguous_strides`, `NdLayout`.
- `NdLayout` view methods preserve fail-closed bounds behavior and mark overlapping views read-only.
- Future reshape-surface additions must preserve current SCE contracts and attach evidence through the packet gate artifacts.
- No `unsafe` pathways are introduced by the packet-D boundary plan.

## 4. Instrumentation Insertion Points

| Lane | Insertion point | Evidence artifact target |
|---|---|---|
| Unit/property | `crates/fnp-ndarray/src/lib.rs` tests + packet-E additions | `unit_property_evidence.json` + reason codes |
| Differential/metamorphic/adversarial | `crates/fnp-conformance` packet-F shape runner | `differential_metamorphic_adversarial_evidence.json` + fixture manifests |
| E2E/replay | `scripts/e2e/run_shape_contract_journey.sh` + workflow scenario gate | packet-G replay logs under `artifacts/phase2c/FNP-P2C-001/` |
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

## 7. Packet Validation

- Validation rule for this packet:
  - Artifact refreshes must not change code-level API.
  - Existing crate boundaries and public functions compile as-is.
  - Packet validator continues to parse current packet artifacts with `status=ready`, `missing_artifacts=0`, `missing_fields=0`, and `parse_errors=0`.
- Validation command (offload via `rch`):
  `rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-001`

## 8. Graveyard and FrankenSuite Mapping

- `alien_cs_graveyard.md` §0.19 (evidence ledger schema)
- `alien_cs_graveyard.md` §0.4 (decision-theoretic runtime contracts)
- `alien_cs_graveyard.md` §6.12 (property-based testing with shrinking)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19

## 9. Rollback Handle

If boundary planning introduces invalid assumptions, revert `artifacts/phase2c/FNP-P2C-001/implementation_plan.md` and restore the previous packet-D planning baseline tied to last green packet validation output.
