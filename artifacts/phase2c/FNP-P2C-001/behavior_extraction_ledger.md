# FNP-P2C-001 Behavior Extraction Ledger

Packet: `FNP-P2C-001-A`  
Subsystem: `Shape/reshape legality`

## 1. Observable Contract Ledger

| Contract ID | Observable behavior | Strict mode expectation | Hardened mode expectation | Legacy anchors |
|---|---|---|---|---|
| `P2C001-C01` | Reshape with one `-1` infers exactly one dimension from prior element count | inference succeeds only when `old_size % known_product == 0` | same outward rule; malformed payloads fail-closed with stable error class | `shape.c:246`, `shape.c:467` |
| `P2C001-C02` | Multiple unknown dimensions are rejected | deterministic failure when more than one unknown dimension appears | same failure semantics with bounded diagnostic addenda | `shape.c:485` |
| `P2C001-C03` | Reshape preserves element-count invariance | reject shape requests where product differs from source size | same external contract; overflow/malformed arithmetic must fail-closed | `shape.c:499`, `shape.c:505` |
| `P2C001-C04` | Broadcast legality uses right-aligned merge with singleton expansion | legal when dimensions are equal or one side is `1`; otherwise reject | same legality rule, same rejection class for incompatible shapes | `shape.c` reshape/broadcast family + `test_shape_base.py:86`, `test_shape_base.py:111` |
| `P2C001-C05` | C/F contiguous stride derivation is deterministic for valid shape + item size | computed byte strides are deterministic and order-consistent | same deterministic strides; reject invalid item size/overflow | `shape.c:271`, `_attempt_nocopy_reshape` stride handling + Rust SCE |
| `P2C001-C06` | Legacy `reshape` API delegates to canonical reshape path | `PyArray_Reshape` behavioral surface matches `PyArray_Newshape` contract | same API behavior under hardened validation | `shape.c:325`, `shape.c:333` |
| `P2C001-C07` | Large broadcasted shapes remain shape-correct (subject to platform limits) | preserve broadcasted shape outcomes in high-cardinality cases | preserve outcome; fail-closed when resource/security gates trigger | `test_shape_base.py:450` |
| `P2C001-C08` | No-copy reshape preserves the source traversal's byte addresses in requested C or F order | `NdLayout::reshape_view` returns a layout sharing the same starting byte and access flags, or `Ok(None)` when copying is necessary | same alias rules, with checked stride arithmetic and shape errors reported separately | `shape.c:_reshape_with_copy_arg`, `shape.c:_attempt_nocopy_reshape` |

## 2. Compatibility Invariants

1. Element conservation invariant: reshape never changes logical element count.
2. Unknown-dimension uniqueness invariant: exactly zero or one `-1` is admissible.
3. Broadcast determinism invariant: same input shapes always produce same merged shape or same incompatibility class.
4. Order-sensitive stride invariant: C and F contiguous stride derivations remain deterministic for identical inputs.
5. Error-class stability invariant: incompatible reshape/broadcast failures remain class-stable across strict and hardened modes.

## 3. Undefined or Under-Specified Edges (Tagged)

| Unknown ID | Description | Risk | Owner bead | Closure criteria |
|---|---|---|---|---|
| `P2C001-U01` | Rust layout support now models no-copy reshape eligibility in C and F order (`NdLayout::reshape_view` and `P2C001-C08`). The address-enumeration regression test checks the alias boundary. | high | `bd-23m.12.4` | Implement alias-sensitive reshape/view boundary and land matching contract table entries with explicit invariants. |
| `P2C001-U02` | closed: differential/metamorphic/adversarial packet-F lanes now verify reshape/broadcast mismatch class behavior and reason-code stability. | closed | `bd-23m.12.6` | closed by `run_shape_stride_{differential,metamorphic,adversarial}_suite` wiring plus `artifacts/phase2c/FNP-P2C-001/differential_metamorphic_adversarial_evidence.json`. |
| `P2C001-U03` | Hardened-mode budget policy for extreme broadcast/reshape requests (size/resource thresholds) is not yet codified for this packet. | high | `bd-23m.12.3` | Threat model + policy table define explicit bounded-mode behavior and reason-code logging. |
| `P2C001-U04` | closed: replay-forensics linkage from packet-specific shape failures to workflow scenarios is implemented for packet-G. | closed | `bd-23m.12.7` | closed by `shape_reshape_packet_replay` + `shape_reshape_packet_hostile_guardrails` scenarios, `scripts/e2e/run_shape_contract_journey.sh`, and packet-G evidence artifacts. |
| `P2C001-U05` | closed: packet-H optimization/profile lane for shape broadcast legality was implemented with behavior-isomorphism proof. | closed | `bd-23m.12.8` | closed by same-rank/identical-shape fast paths in `crates/fnp-ndarray/src/lib.rs`, `generate_packet001_optimization_report`, and packet-H optimization evidence artifacts. |

## 4. Verification Hooks

| Verification lane | Planned hook | Artifact target |
|---|---|---|
| Unit/property | Extend SCE tests for reshape inference, overflow boundaries, and broadcast edge matrices | `crates/fnp-ndarray/src/lib.rs` + packet-E additions |
| Differential/metamorphic/adversarial | packet-F fixture corpus and oracle-comparison suites are implemented for reshape/broadcast mismatch/adversarial classes | `crates/fnp-conformance/fixtures/shape_stride_cases.json`, `crates/fnp-conformance/fixtures/oracle_outputs/shape_stride_differential_report.json`, `artifacts/phase2c/FNP-P2C-001/differential_metamorphic_adversarial_evidence.json` |
| E2E | packet-scoped reshape/broadcast scenario chain is implemented in strict and hardened replay modes | `crates/fnp-conformance/fixtures/workflow_scenario_corpus.json`, `scripts/e2e/run_shape_contract_journey.sh`, `artifacts/phase2c/FNP-P2C-001/workflow_scenario_packet001_{e2e,reliability,artifact_index}.json`, `artifacts/phase2c/FNP-P2C-001/e2e_replay_forensics_evidence.json` |
| Optimization/isomorphism | packet-H profile-first single-lever optimization with parity-preserving verification | `crates/fnp-conformance/src/bin/generate_packet001_optimization_report.rs`, `artifacts/phase2c/FNP-P2C-001/optimization_profile_report.json`, `artifacts/phase2c/FNP-P2C-001/optimization_profile_isomorphism_evidence.json` |
| Structured logging | Enforce required fields across packet checks: `fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code` | `artifacts/contracts/test_logging_contract_v1.json`, `scripts/e2e/run_test_contract_gate.sh` |

## 5. Method-Stack Artifacts and EV Gate

- Alien decision contract: policy-mediated shape/broadcast decisions must include state/action/loss rationale when hardened guardrails alter flow.
- Optimization gate: reshape/broadcast performance work requires baseline/profile + one-lever + isomorphism proof artifacts.
- EV gate: optimization levers are promotable only when `EV >= 2.0`; otherwise remain explicit deferred debt.
- RaptorQ scope: packet-I must attach sidecar/scrub/decode-proof artifacts for parity bundle durability.
- Packet-F closure evidence: `artifacts/phase2c/FNP-P2C-001/differential_metamorphic_adversarial_evidence.json` captures differential/metamorphic/adversarial suite and gate pass evidence for this packet lane.
- Packet-G closure evidence: `artifacts/phase2c/FNP-P2C-001/e2e_replay_forensics_evidence.json` captures reshape/broadcast replay logs, reliability diagnostics, and forensics artifact indexing.
- Packet-H closure evidence: `artifacts/phase2c/FNP-P2C-001/optimization_profile_report.json` and `artifacts/phase2c/FNP-P2C-001/optimization_profile_isomorphism_evidence.json` show EV promotion (`24.0`) with no behavior drift.

## 6. Rollback Handle

If packet-local changes introduce compatibility drift, revert packet-local artifacts under `artifacts/phase2c/FNP-P2C-001/` and restore the last green `fnp-ndarray`/conformance shape-stride evidence baseline prior to the drift.
