# FNP-P2C-002 Risk Note

packet_id: `FNP-P2C-002`  
subsystem: `dtype descriptors and promotion`

## compatibility_risks

| risk_id | threat_class | hostile input / abuse vector | strict mode handling | hardened mode handling | control anchors | verification hooks | reason_code family |
|---|---|---|---|---|---|---|---|
| `P2C002-RISK-01` | `unsafe_cast_path` | crafted cast requests intended to bypass lossless/admissibility rules | deterministic cast rejection based on scoped matrix | same rejection plus bounded metadata validation | `security_control_checks_v1.yaml` -> `unsafe_cast_path` | `run_dtype_promotion_suite`, packet-F differential cases | `dtype_cast_*` |
| `P2C002-RISK-02` | `unknown_metadata_version` | unknown/invalid descriptor metadata forms and unsupported type objects | fail-closed on unsupported metadata | fail-closed with deterministic audit context | runtime policy controls + security control map | runtime policy suite + adversarial runtime-policy fixtures | `dtype_normalization_failed`, `wire_unknown_*` |
| `P2C002-RISK-03` | `adversarial_fixture` | malformed fixture payloads poisoning promotion/cast replay corpus | reject malformed fixtures | reject + quarantine/audit entry | `security_control_checks_v1.yaml` -> `adversarial_fixture` | adversarial suite + packet-G replay checks | `fixture_contract_violation` |
| `P2C002-RISK-04` | `policy_override_abuse` | attempts to force unsafe promotion/cast behavior through override channels | no override for incompatible semantics | audited-only overrides, otherwise fail-closed | `fnp_runtime::evaluate_policy_override` + override audit controls | runtime-policy adversarial suite + security gate | `override_*` |
| `P2C002-RISK-05` | `corrupt_durable_artifact` | tampered parity/cast evidence bundles | fail gate on integrity mismatch | bounded recovery only with decode/hash proof | RaptorQ control mapping in security checks | security contract suite + decode-proof checks | `artifact_integrity_*` |

## Threat Envelope and Hardened Recovery

- Unknown or incompatible descriptor/cast semantics are fail-closed in both strict and hardened modes.
- Hardened mode preserves API-level dtype outcomes while allowing only deterministic bounded validation behavior.
- All policy-mediated outcomes must emit replayable logs (`fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`).

## Adversarial Fixture Set Definition

| lane | fixture family | objective | owner bead |
|---|---|---|---|
| Fuzz/property | packet-E dtype property corpus | detect cast-table and normalization invariant violations with replayable structured logs | `unit_property_evidence.json` |
| Differential/metamorphic | packet-F dtype promotion/cast oracle corpus | enforce strict parity for promotion/cast outcomes and failure classes across the current fixture matrix | `differential_metamorphic_adversarial_evidence.json` |
| E2E/replay | packet-G mixed-dtype pipeline scenarios | verify strict/hardened replay and policy-forensics traceability through dtype fixture journeys | `e2e_replay_forensics_evidence.json` + `workflow_scenario_packet002_*` artifacts |

## Residual Risks and Compensating Controls

| residual_id | residual risk | compensating controls | closure gate |
|---|---|---|---|
| `P2C002-RES-01` | Full legacy cast-table parity remains incomplete beyond scoped subset | keep out-of-scope cast combinations fail-closed and explicitly logged | `bd-23m.13.4` + `bd-23m.13.5` |
| `P2C002-RES-02` | Structured dtype cast semantics have current packet coverage but not full legacy breadth | enforce strict contract rows + expand unit/property + differential coverage | packet-E/F evidence green plus expanded structured-dtype oracle matrix |
| `P2C002-RES-03` | Promotion edge behavior for exotic/weak scalar interactions may drift without full oracle matrix | maintain metamorphic permutation checks and deterministic drift gates | packet-F strict drift gate green plus expanded weak-scalar fixtures |

## oracle_tests

- `legacy_numpy_code/numpy/numpy/_core/tests/test_dtype.py`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_casting_floatingpoint_errors.py`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_ufunc.py` (dtype resolution anchors)

## raptorq_artifacts

- `artifacts/phase2c/FNP-P2C-002/parity_report.raptorq.json`
- `artifacts/phase2c/FNP-P2C-002/parity_report.scrub_report.json`
- `artifacts/phase2c/FNP-P2C-002/parity_report.decode_proof.json`
- `artifacts/phase2c/FNP-P2C-002/final_evidence_pack.json`
- `artifacts/raptorq/conformance_bundle_v1.sidecar.json` (program-level baseline reference)
- `artifacts/raptorq/conformance_bundle_v1.scrub_report.json` (program-level baseline reference)
- `artifacts/raptorq/conformance_bundle_v1.decode_proof.json` (program-level baseline reference)

## Rollback Handle

If packet threat controls regress compatibility guarantees, revert `artifacts/phase2c/FNP-P2C-002/risk_note.md` and restore the last green risk baseline tied to security gate evidence.
