# FNP-P2C-006 Risk Note

packet_id: `FNP-P2C-006`  
subsystem: `stride-tricks and broadcasting API`

## compatibility_risks

| risk_id | threat_class | hostile input / abuse vector | strict mode handling | hardened mode handling | control anchors | verification hooks | reason_code family |
|---|---|---|---|---|---|---|---|
| `P2C006-RISK-01` | `malformed_shape` | negative dimensions, incompatible target shapes, scalar/non-scalar misuse in broadcast APIs | deterministic rejection with stable failure class | same rejection plus bounded audit context | `security_control_checks_v1.yaml` -> `malformed_shape` | `run_shape_stride_suite`, packet-F differential corpus | `broadcast_to_shape_invalid`, `broadcast_shapes_incompatible` |
| `P2C006-RISK-02` | `malicious_stride_alias` | crafted `as_strided` metadata creating overlapping/unsafe write patterns | preserve legacy class behavior with explicit failure paths where scoped | preserve observable behavior, log overlap-risk decisions, fail-closed on unknown incompatible semantics | `security_control_checks_v1.yaml` -> `malicious_stride_alias` | packet-E unit/property overlap cases + packet-G replay lane | `as_strided_contract_violation` |
| `P2C006-RISK-03` | `adversarial_fixture` | poisoned packet fixtures intended to mask broadcast/iterator regressions | reject malformed fixture payloads | reject + quarantine/audit entry | `security_control_checks_v1.yaml` -> `adversarial_fixture` | `run_runtime_policy_adversarial_suite`, packet-F fixture validation | `fixture_contract_violation` |
| `P2C006-RISK-04` | `policy_override_abuse` | attempts to bypass non-broadcastable or fail-closed paths through override channels | no override for incompatible semantics | audited-only overrides, otherwise fail-closed | `fnp_runtime::evaluate_policy_override` + override controls | runtime-policy adversarial suite + security gate | `override_*` |
| `P2C006-RISK-05` | `unknown_metadata_version` | unknown/invalid mode/class metadata at policy boundaries for packet replay and gates | fail-closed | fail-closed with deterministic reason codes and audit references | `security_control_checks_v1.yaml` -> `unknown_metadata_version` | `run_runtime_policy_suite`, `run_runtime_policy_adversarial_suite` | `wire_unknown_*` |
| `P2C006-RISK-06` | `corrupt_durable_artifact` | tampered packet evidence and replay bundles | fail gate on integrity mismatch | bounded recovery only with decode/hash proof | `security_control_checks_v1.yaml` -> `corrupt_durable_artifact` | `validate_phase2c_packet`, security contract suite | `artifact_integrity_*` |

## Threat Envelope and Hardened Recovery

- Unknown or incompatible stride/broadcast semantics are fail-closed in strict and hardened modes.
- Hardened mode preserves API-visible outputs while adding bounded validation and explicit audit records for overlap-risk and no-broadcast decision points.
- Policy-mediated outcomes must emit replayable fields: `fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`.
- Recovery behavior is deterministic: either fail-closed, full-validate, or audited bounded handling with no silent repairs.

## Adversarial Fixture Set Definition

| lane | fixture family | objective | owner bead |
|---|---|---|---|
| Fuzz/property | packet-E stride/broadcast invariant corpus (planned) | detect malformed shape, overlap-risk, and no-broadcast invariant violations with shrinkable counterexamples | `bd-23m.17.5` |
| Differential/metamorphic | packet-F oracle corpus for stride-tricks/broadcast API (planned) | enforce strict parity for success/failure classes and shape/readonly outcomes | `bd-23m.17.6` |
| E2E/replay | packet-G stride-tricks + iterator workflow scenarios (planned) | verify strict/hardened replay and forensics traceability in integration paths | `bd-23m.17.7` |

## Residual Risks and Compensating Controls

| residual_id | residual risk | compensating controls | closure gate |
|---|---|---|---|
| `P2C006-RES-01` | The `fnp-iter` NDIter core is implemented, but packet-006 still needs stride-tricks integration coverage for iterator-sensitive broadcast/view paths. | Keep unsupported iterator-sensitive semantics explicitly fail-closed; require contract-row mapping and packet-local differential fixtures before promotion. | packet-E/F baseline expansion |
| `P2C006-RES-02` | High-arity broadcast behavior (`>64` inputs) may drift without dedicated oracle corpora. | add targeted differential corpus and high-arity metamorphic invariants | `bd-23m.17.6` |
| `P2C006-RES-03` | Warning-level compatibility around `broadcast_arrays` writeability path remains subtle/version-sensitive. | lock class/family parity in contract rows and capture explicit reason codes in replay logs | `bd-23m.17.5` + `bd-23m.17.6` |
| `P2C006-RES-04` | Overlap-risk policy for dangerous stride views may regress without integration-level replay. | require packet-G e2e forensics scenarios and strict/hardened comparison logs | `bd-23m.17.7` |
| `P2C006-RES-05` | Hardened budget/calibration thresholds are now profiled for packet-H hotspot behavior but are not yet tuned for full adversarial corpus scale. | use packet-H profile/isomorphism evidence as calibration baseline and keep conservative fallback trigger active until packet-I sign-off | packet-I closure (`artifacts/phase2c/FNP-P2C-006/packet_readiness_report.json`) |

## Budgeted Mode and Decision-Theoretic Controls

### Explicit bounded caps (hardened policy path)

| control | cap | deterministic exhaustion behavior |
|---|---|---|
| broadcast rank merge depth | `<= 1024` dimensions per merge request | `fail_closed` with `broadcast_shapes_incompatible` |
| high-arity broadcast operands | `<= 4096` operands per request | `fail_closed` with `broadcast_shapes_incompatible` |
| overlap-risk validations | `<= 256` overlap decisions per request | `fail_closed` with `as_strided_contract_violation` |
| policy override evaluations | `<= 16` override checks per request | exhaust to conservative default (`fail_closed`) with `override_budget_exhausted` |
| packet-local audit payload | `<= 64 MiB` structured event payload | truncate optional diagnostics while preserving mandatory fields |

### Expected-loss model

| state | action set | primary loss if wrong |
|---|---|---|
| ambiguous broadcast merge semantics | `{merge, reject}` | invalid shape accepted and propagated through downstream iterator paths |
| overlap-risk stride view | `{allow_view, reject_view}` | alias-unsafe write/read behavior admitted |
| warning-level broadcast writeability compatibility | `{legacy_compat_path, reject}` | user-visible warning/error-class drift from legacy behavior |
| unknown metadata class | `{fail_closed}` | undefined policy semantics admitted into runtime path |

### Calibration and fallback trigger

- Trigger fallback when either condition is true:
  - strict vs hardened failure-class drift exceeds `0.1%`, or
  - unknown/uncategorized reason-code rate exceeds `0.01%`.
- Fallback action: force conservative deterministic behavior (`full_validate` or `fail_closed`) until recalibration artifacts are produced and validated.

### Packet-H calibration artifact

- Profile baseline/rebaseline: `artifacts/phase2c/FNP-P2C-006/optimization_profile_report.json`.
- Isomorphism evidence: `artifacts/phase2c/FNP-P2C-006/optimization_profile_isomorphism_evidence.json`.
- Workflow replay evidence: `artifacts/phase2c/FNP-P2C-006/workflow_scenario_packet006_opt_e2e.jsonl`.
- Workflow forensics index: `artifacts/phase2c/FNP-P2C-006/workflow_scenario_packet006_opt_artifact_index.json`.
- Workflow reliability report: `artifacts/phase2c/FNP-P2C-006/workflow_scenario_packet006_opt_reliability.json`.
- Current packet-H calibration signal:
  - p95 latency delta `-40.331%`,
  - p95 throughput delta `+67.590%`,
  - failure-class drift `0` across packet-H isomorphism checks,
  - workflow gate replay coverage ratio `1.0` (298/298 scenarios passed).

## oracle_tests

- `legacy_numpy_code/numpy/numpy/lib/tests/test_stride_tricks.py`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_nditer.py`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_shape_base.py` (broadcast-edge anchors)

## raptorq_artifacts

- `artifacts/phase2c/FNP-P2C-006/parity_report.raptorq.json`
- `artifacts/phase2c/FNP-P2C-006/parity_report.scrub_report.json`
- `artifacts/phase2c/FNP-P2C-006/parity_report.decode_proof.json`
- `artifacts/phase2c/FNP-P2C-006/packet_readiness_report.json`
- `artifacts/raptorq/conformance_bundle_v1.sidecar.json` (program-level baseline reference)
- `artifacts/raptorq/conformance_bundle_v1.scrub_report.json` (program-level baseline reference)
- `artifacts/raptorq/conformance_bundle_v1.decode_proof.json` (program-level baseline reference)

## residual_risk_monitoring

owner: `packet-006-maintainers`  
follow_up_gate: `bd-23m.11 readiness drill + packet-I residual risk review`

follow_up_actions:
- expand packet-006 workflow scenario breadth for broadcast/stride hostile journeys and iterator-sensitive view integration before readiness sign-off.
- recalibrate hardened broadcast/stride budget thresholds against the full adversarial corpus and document drift trends.

## Rollback Handle

- Rollback command path: `git restore --source <last-green-commit> -- artifacts/phase2c/FNP-P2C-006/risk_note.md`
- Baseline comparator to beat/restore:
  - last green `rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-006` packet report,
  - plus last green security/test/workflow gate artifacts tied to packet `FNP-P2C-006`.
- If comparator is not met, restore risk-note baseline and re-run packet gates before reattempting policy changes.
