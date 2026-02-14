# FNP-P2C-001 Risk Note

packet_id: `FNP-P2C-001`  
subsystem: `shape/reshape legality`

## compatibility_risks

| risk_id | threat_class | hostile input / abuse vector | strict mode handling | hardened mode handling | control anchors | verification hooks | reason_code family |
|---|---|---|---|---|---|---|---|
| `P2C001-RISK-01` | `malformed_shape` | negative dimensions below `-1`, multiple unknown dims, incompatible reshape products | deterministic reject with stable failure class | same rejection with bounded diagnostic context only | `security_control_checks_v1.yaml` -> `malformed_shape` | `run_shape_stride_suite`, packet-F differential fixtures | `reshape_*` |
| `P2C001-RISK-02` | `malicious_stride_alias` | crafted shape/order/item-size combinations intended to trigger illegal stride/alias transitions | reject unsafe transitions; preserve legality invariants | same reject path, fail-closed on overflow or unsafe transitions | `security_control_checks_v1.yaml` -> `malicious_stride_alias` | `run_shape_stride_suite`, packet-E unit/property alias checks | `stride_invalid_item_or_overflow`, `reshape_wrapper_canonical_mismatch` |
| `P2C001-RISK-03` | `adversarial_fixture` | malformed corpus payloads, inconsistent fixture metadata, replay poisoning | reject malformed fixtures | reject + quarantine/audit context | `security_control_checks_v1.yaml` -> `adversarial_fixture` | `run_runtime_policy_adversarial_suite`, packet-G replay checks | `fixture_contract_violation` |
| `P2C001-RISK-04` | `policy_override_abuse` | attempts to bypass packet boundary fail-closed behavior via policy overrides | no override permitted for unknown/incompatible semantics | audited-only overrides, deterministic logging, fail-closed otherwise | `fnp_runtime::evaluate_policy_override`, security gate controls | runtime-policy adversarial suite + security gate | `override_*` |
| `P2C001-RISK-05` | `unknown_metadata_version` | unknown mode/class metadata entering packet-adjacent policy gates | fail-closed | fail-closed | runtime policy contracts + security controls | `run_runtime_policy_suite`, `run_runtime_policy_adversarial_suite` | `wire_unknown_*` |

## Threat Envelope and Hardened Recovery

- Fail-closed is mandatory for unknown or incompatible semantics in both strict and hardened modes.
- Hardened-only deviations are restricted to allowlisted class `admission_guard_caps` for `FNP-P2C-001` and must preserve API-level shape/dtype contract.
- Any hardened recovery must be deterministic, bounded, and fully audit logged (`fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`).

## Adversarial Fixture Set Definition

| lane | fixture family | objective | owner bead |
|---|---|---|---|
| Fuzz/property | packet-E shape-stride property corpus (planned) | discover malformed/overflow boundary counterexamples with deterministic shrinking | `bd-23m.12.5` |
| Differential/metamorphic | packet-F reshape/broadcast oracle fixtures (planned) | enforce strict parity for failure classes and broadcast legality outcomes | `bd-23m.12.6` |
| E2E/replay | packet-G reshape workflow scenarios (planned) | ensure fail-closed and hardened traces remain reproducible with forensics links | `bd-23m.12.7` |

## Residual Risks and Compensating Controls

| residual_id | residual risk | compensating controls | closure gate |
|---|---|---|---|
| `P2C001-RES-01` | Full no-copy alias-preserving reshape parity not yet implemented | keep alias-sensitive transitions fail-closed until packet-D implementation lands | `bd-23m.12.4` complete + packet validator evidence update |
| `P2C001-RES-02` | Error-text parity across all reshape mismatch variants not yet fully differential-tested | enforce error-class parity gates and expand oracle fixture taxonomy | `bd-23m.12.6` strict drift gate green |
| `P2C001-RES-03` | High-cardinality workload policy caps may require calibration tuning | admission guard caps + audit review + deterministic fallback trigger | packet-H optimization + packet-I parity/risk sign-off |

## oracle_tests

- `legacy_numpy_code/numpy/numpy/lib/tests/test_shape_base.py`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_shape_base.py`

## raptorq_artifacts

- `artifacts/phase2c/FNP-P2C-001/parity_report.raptorq.json` (planned at packet-I)
- `artifacts/phase2c/FNP-P2C-001/parity_report.decode_proof.json` (planned at packet-I)
- `artifacts/raptorq/conformance_bundle_v1.sidecar.json` (program-level baseline reference)
- `artifacts/raptorq/conformance_bundle_v1.scrub_report.json` (program-level baseline reference)
- `artifacts/raptorq/conformance_bundle_v1.decode_proof.json` (program-level baseline reference)

## Rollback Handle

If threat-model or compatibility-envelope changes regress packet behavior, revert `artifacts/phase2c/FNP-P2C-001/risk_note.md` and restore the last green packet risk baseline tied to security gate evidence.
