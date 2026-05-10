# FNP-P2C-004 Risk Note

packet_id: `FNP-P2C-004`  
subsystem: `NDIter traversal/index semantics`

## compatibility_risks

| risk_id | threat_class | hostile input / abuse vector | strict mode handling | hardened mode handling | control anchors | verification hooks | reason_code family |
|---|---|---|---|---|---|---|---|
| `P2C004-RISK-01` | `malformed_iterator_config` | Conflicting `flags`/`op_flags`/`op_axes`/`itershape`, invalid casting combinations, malformed constructor metadata. | Deterministic reject with stable failure class. | Same failure class plus bounded diagnostics and audit context. | `security_control_checks_v1.yaml` -> `malformed_shape`; packet contract rows `P2C004-R01`. | packet-E constructor matrix + packet-F constructor differential corpus. | `nditer_constructor_invalid_configuration` |
| `P2C004-RISK-02` | `iterator_state_abuse` | Out-of-bounds or mode-invalid `multi_index`/`index`/`iterindex`/`iterrange` seeks; illegal state transitions to desynchronize cursor/index state. | Reject invalid seek/transition requests with stable error class. | Same externally visible rejection with deterministic state-validation audit fields. | `security_control_checks_v1.yaml` -> `unknown_metadata_version`; contract rows `P2C004-R03`/`R04`/`R05`/`R06`. | packet-E seek coherence property tests; packet-F seek oracle fixtures. | `nditer_seek_mode_mismatch`, `nditer_index_out_of_bounds`, `nditer_iterrange_invalid`, `nditer_transition_precondition_failed` |
| `P2C004-RISK-03` | `no_broadcast_bypass` | Attempted broadcast expansion of operands explicitly marked `no_broadcast`. | Fail with stable non-broadcastable class. | Same failure class and explicit reason-code audit record. | `security_control_checks_v1.yaml` -> `malformed_shape`; contract row `P2C004-R08`. | packet-E no-broadcast laws; packet-F adversarial no-broadcast fixtures. | `nditer_no_broadcast_violation` |
| `P2C004-RISK-04` | `overlap_policy_abuse` | Crafted overlapping read/write operands attempting to bypass `copy_if_overlap` safeguards or misuse `overlap_assume_elementwise`. | Preserve legacy outcome class; unsafe overlap paths are rejected/adjusted deterministically. | Same outward behavior with bounded overlap-risk policy logging and fail-closed on unknown overlap semantics. | `security_control_checks_v1.yaml` -> `malicious_stride_alias`; contract row `P2C004-R09`. | packet-E overlap property corpus; packet-F overlap differential oracle; packet-G replay lane. | `nditer_overlap_policy_triggered` |
| `P2C004-RISK-05` | `flatiter_indexing_abuse` | Malformed integer/slice/fancy/boolean indexing and assignment, including shape-mismatched RHS payloads and out-of-range addressing. | Stable success/failure class parity for valid/invalid forms. | Same behavior with deterministic fail-closed handling for malformed payload classes. | `security_control_checks_v1.yaml` -> `adversarial_fixture`; contract row `P2C004-R10`. | packet-E flatiter unit/property matrix; packet-F flatiter differential corpus. | `flatiter_indexing_contract_violation` |
| `P2C004-RISK-06` | `index_stream_shape_abuse` | Negative/non-integer/overflow-prone dimension inputs to `ndindex`/`ndenumerate`. | Deterministic validation failure for invalid dimensions. | Same failure class with bounded diagnostics. | `security_control_checks_v1.yaml` -> `malformed_shape`; contract row `P2C004-R11`. | packet-E ndindex/ndenumerate laws; packet-F index-stream oracle checks. | `ndindex_shape_validation_failed` |
| `P2C004-RISK-07` | `policy_override_abuse` | Attempts to force incompatible iterator semantics via runtime policy override channels. | No override for unknown/incompatible semantics; fail-closed. | Audited-only overrides where explicitly allowlisted; fail-closed otherwise. | `fnp_runtime::evaluate_policy_override`; security contract gate controls. | runtime-policy adversarial suite + security gate. | `override_*` |
| `P2C004-RISK-08` | `unknown_metadata_version` | Unknown wire mode/class metadata at packet policy boundaries. | Fail-closed. | Fail-closed with deterministic reason-code emission. | `security_control_checks_v1.yaml` -> `unknown_metadata_version`. | `run_runtime_policy_suite`, `run_runtime_policy_adversarial_suite`. | `wire_unknown_*` |
| `P2C004-RISK-09` | `adversarial_fixture` | Poisoned fixture metadata/payloads intended to hide iterator parity drift. | Reject malformed fixture bundles. | Reject + quarantine/audit references; no partial silent recovery. | `security_control_checks_v1.yaml` -> `adversarial_fixture`. | `run_test_contract_gate`, packet-F corpus schema validation. | `fixture_contract_violation` |
| `P2C004-RISK-10` | `corrupt_durable_artifact` | Tampered packet durability artifacts (sidecar/scrub/decode-proof mismatch). | Fail validation gate on integrity mismatch. | Deterministic bounded recovery only with successful decode/hash proof. | `security_control_checks_v1.yaml` -> `corrupt_durable_artifact`. | `validate_phase2c_packet --packet-id FNP-P2C-004`. | `artifact_integrity_*` |

## Threat Envelope and Hardened Recovery

- Unknown or incompatible iterator semantics are fail-closed in both strict and hardened modes.
- Hardened mode may add bounded validation and audit enrichment, but cannot alter the API-visible success/failure class for covered packet contracts.
- Recovery behavior is deterministic and explicit: `allow`, `full_validate`, or `fail_closed`; no silent repairs.
- Replay and forensic records for policy decisions must include `fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, and `reason_code`.

## Adversarial Fixture Set Definition

| lane | fixture family | objective | owner bead |
|---|---|---|---|
| Fuzz/property | landed iterator invariant corpus + packet-E expansion | discover malformed constructor/state/seek/overlap edge cases with shrinkable counterexamples | `bd-23m.15.5` |
| Differential/metamorphic | landed NumPy smoke checks for iterator state/external-loop chunks + packet-F corpus expansion | enforce strict parity for success/failure classes and traversal/index streams | `bd-23m.15.6` |
| E2E/replay | packet-G iterator workflow scenarios `nditer_packet_replay` + `nditer_packet_hostile_guardrails` | verify strict/hardened replay traceability and policy-forensics linkage | `bd-23m.15.7` |

## Residual Risks and Compensating Controls

| residual_id | residual risk | compensating controls | closure gate |
|---|---|---|---|
| `P2C004-RES-01` | The `fnp-iter` NDIter core is implemented, but full `op_axes`/`itershape`/buffered operand parity is not yet closure-complete. | Keep unsupported iterator semantics fail-closed and require packet-local differential fixtures before promotion. | packet-E/F baseline expansion |
| `P2C004-RES-02` | Full `op_axes`/`itershape` parity matrix remains partially unpinned in differential fixtures. | Expand contract rows + adversarial fixture taxonomy for complex remap/reduction shapes. | `bd-23m.15.6` |
| `P2C004-RES-03` | Overlap-policy edge behavior can regress as scenario breadth grows across parity debt. | Maintain packet-G overlap replay scenarios with deterministic reason-code linkage and add profile-backed regressions checks. | `bd-23m.15.8` |
| `P2C004-RES-04` | Error-text parity for all iterator edge cases is not yet closure-complete. | Maintain strict class/family parity gates and track message-level drift as explicit parity debt. | packet-I parity sign-off |
| `P2C004-RES-05` | Hardened budget/calibration thresholds are now profiled for packet-H hotspot behavior but are not yet tuned for full corpus scale. | Use packet-H profile/isomorphism evidence as calibration baseline and keep conservative fallback trigger active until packet-I sign-off. | packet-H optimization evidence + packet-I risk closure |

## Budgeted Mode and Decision-Theoretic Controls

### Explicit bounded caps (hardened policy path)

| control | cap | deterministic exhaustion behavior |
|---|---|---|
| constructor metadata normalization | `<= 256` operand descriptors per iterator request | `fail_closed` with `nditer_constructor_invalid_configuration` |
| seek/reset mutation operations | `<= 1_000_000` state mutations per fixture replay | `fail_closed` with `nditer_seek_mode_mismatch` |
| overlap remediations | `<= 128` copy/remap decisions per iterator request | stop remediation and `fail_closed` with `nditer_overlap_policy_triggered` |
| policy override evaluations | `<= 16` override checks per request | exhaust to conservative default (`fail_closed`) with `override_budget_exhausted` |
| packet-local audit buffer | `<= 64 MiB` structured event payload | truncate optional diagnostics, preserve required fields, keep decision path deterministic |

### Expected-loss model

| state | action set | primary loss if wrong |
|---|---|---|
| constructor ambiguity / mixed flags | `{reject, full_validate}` | silent semantic drift or acceptance of incompatible traversal policy |
| overlap-risk detected | `{copy_path, reject}` | data corruption/alias unsafety vs false-negative acceptance |
| index/seek transition request | `{accept_seek, reject_seek}` | cursor desynchronization leading to incorrect iteration output |
| unknown metadata/policy class | `{fail_closed}` | undefined behavior if admitted |

### Calibration and fallback trigger

- Trigger fallback when either condition is true:
  - strict vs hardened failure-class drift rate for packet corpus exceeds `0.1%`, or
  - unknown/uncategorized reason-code rate exceeds `0.01%`.
- Fallback action: force conservative deterministic path (`fail_closed` or `full_validate` only) until recalibration artifacts are produced and validated.

### Packet-H calibration artifact

- Profile baseline/rebaseline: `artifacts/phase2c/FNP-P2C-004/optimization_profile_report.json`.
- Isomorphism evidence: `artifacts/phase2c/FNP-P2C-004/optimization_profile_isomorphism_evidence.json`.
- Workflow replay evidence: `artifacts/phase2c/FNP-P2C-004/workflow_scenario_packet004_opt_e2e.jsonl`.
- Workflow forensics index: `artifacts/phase2c/FNP-P2C-004/workflow_scenario_packet004_opt_artifact_index.json`.
- Workflow reliability report: `artifacts/phase2c/FNP-P2C-004/workflow_scenario_packet004_opt_reliability.json`.
- Current packet-H calibration signal:
  - p95 latency delta `-52.972%`,
  - p95 throughput delta `+112.638%`,
  - failure-class drift `0` across packet-H isomorphism checks,
  - workflow gate replay coverage ratio `1.0` (298/298 scenarios passed).

## Alien Recommendation Contract Mapping

- Graveyard mappings: `alien_cs_graveyard.md` §0.4, §0.19, §6.12.
- FrankenSuite mappings: `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19.
- EV gate: policy/optimization levers ship only if `EV = (Impact * Confidence * Reuse) / (Effort * AdoptionFriction) >= 2.0`; otherwise they remain explicit deferred parity debt.
- Isomorphism proof artifacts required for policy/optimization changes:
  - ordering/tie-break note,
  - before/after golden parity checks,
  - reproducible benchmark delta artifact.
- Hotspot evidence requirement for non-doc follow-on work: attach baseline/profile artifacts before changing policy/optimization behavior (or include documented profiler-unavailable fallback note).

## oracle_tests

- `legacy_numpy_code/numpy/numpy/_core/tests/test_nditer.py`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_indexing.py`
- `legacy_numpy_code/numpy/numpy/lib/tests/test_index_tricks.py`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_multiarray.py`

## raptorq_artifacts

- `artifacts/phase2c/FNP-P2C-004/parity_report.raptorq.json`
- `artifacts/phase2c/FNP-P2C-004/parity_report.scrub_report.json`
- `artifacts/phase2c/FNP-P2C-004/parity_report.decode_proof.json`
- `artifacts/phase2c/FNP-P2C-004/packet_readiness_report.json`
- `artifacts/raptorq/conformance_bundle_v1.sidecar.json` (program-level baseline reference)
- `artifacts/raptorq/conformance_bundle_v1.scrub_report.json` (program-level baseline reference)
- `artifacts/raptorq/conformance_bundle_v1.decode_proof.json` (program-level baseline reference)

## residual_risk_monitoring

owner: `packet-004-maintainers`  
follow_up_gate: `bd-23m.11 readiness drill + packet-I residual risk review`

follow_up_actions:
- expand packet-004 workflow scenario breadth for `op_axes`/`itershape`, iterator seek, overlap-policy, and hostile flatiter workflows before readiness drill sign-off.
- recalibrate hardened iterator budget thresholds against the full adversarial iterator corpus and document drift trends.

## Rollback Handle

- Rollback command path: `git restore --source <last-green-commit> -- artifacts/phase2c/FNP-P2C-004/risk_note.md`
- Baseline comparator to beat/restore:
  - last green `rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-004` packet report,
  - plus last green security/test/workflow gate artifacts tied to packet `FNP-P2C-004`.
- If comparator is not met, restore risk-note baseline and re-run packet gates before reattempting policy changes.
