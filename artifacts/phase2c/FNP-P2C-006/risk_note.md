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
| `P2C006-RES-01` | Full iterator parity (`fnp-iter`) is not implemented yet, increasing risk of silent traversal divergence once integrated. | keep iterator-sensitive semantics explicitly fail-closed where unsupported; require contract-row mapping before implementation | `bd-23m.17.4` + `bd-23m.17.5` |
| `P2C006-RES-02` | High-arity broadcast behavior (`>64` inputs) may drift without dedicated oracle corpora. | add targeted differential corpus and high-arity metamorphic invariants | `bd-23m.17.6` |
| `P2C006-RES-03` | Warning-level compatibility around `broadcast_arrays` writeability path remains subtle/version-sensitive. | lock class/family parity in contract rows and capture explicit reason codes in replay logs | `bd-23m.17.5` + `bd-23m.17.6` |
| `P2C006-RES-04` | Overlap-risk policy for dangerous stride views may regress without integration-level replay. | require packet-G e2e forensics scenarios and strict/hardened comparison logs | `bd-23m.17.7` |

## oracle_tests

- `legacy_numpy_code/numpy/numpy/lib/tests/test_stride_tricks.py`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_nditer.py`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_shape_base.py` (broadcast-edge anchors)

## raptorq_artifacts

- `artifacts/phase2c/FNP-P2C-006/parity_report.raptorq.json` (planned at packet-I)
- `artifacts/phase2c/FNP-P2C-006/parity_report.decode_proof.json` (planned at packet-I)
- `artifacts/raptorq/conformance_bundle_v1.sidecar.json` (program-level baseline reference)
- `artifacts/raptorq/conformance_bundle_v1.scrub_report.json` (program-level baseline reference)
- `artifacts/raptorq/conformance_bundle_v1.decode_proof.json` (program-level baseline reference)

## Rollback Handle

If packet threat controls regress compatibility guarantees, revert `artifacts/phase2c/FNP-P2C-006/risk_note.md` and restore the last green risk baseline tied to security gate evidence.
