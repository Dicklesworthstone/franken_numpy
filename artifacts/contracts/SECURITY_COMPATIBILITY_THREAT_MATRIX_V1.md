# Security Compatibility Threat Matrix V1

Schema lock id: `phase2c-security-compat-v1`  
Version: `1`

Doctrine mapping:

- `strict` mode: maximize NumPy-observable compatibility in scoped surface.
- `hardened` mode: preserve API contract with bounded defensive controls.
- Unknown incompatible feature/metadata: fail-closed in both modes.

## Threat Classes

| Threat Class | Ingress Boundary | Strict Mode Policy | Hardened Mode Policy | Required Evidence Artifact |
|---|---|---|---|---|
| malformed_shape | array shape/reshape/broadcast API | reject invalid shape transitions | reject + bounded diagnostic context | shape admission audit record |
| unsafe_cast_path | dtype cast/coercion boundary | deny unsupported cast | deny unsupported cast + allowlist checks | cast decision ledger |
| malicious_stride_alias | stride/view/index paths | deny illegal alias transitions | deny illegal alias transitions + overlap witness | alias witness report |
| malformed_npy_npz | parser/IO layer | fail-closed | fail-closed + recovery metadata | parser incident report |
| unknown_metadata_version | compatibility negotiation layer | fail-closed | fail-closed | compatibility drift report |
| adversarial_fixture | conformance harness ingest | reject malformed fixture | reject malformed fixture + quarantine entry | fixture integrity report |
| corrupt_durable_artifact | artifact decode/scrub pipeline | fail gate | RaptorQ recover then verify hash | scrub report + decode proof |
| policy_override_abuse | runtime policy control plane | explicit audited override only | explicit audited override only | override audit log |

## Hardened-Mode Deviation Policy

Hardened mode is allowed to diverge only for explicitly allowlisted classes in
`hardened_mode_allowlist_v1.yaml`, and only when all of the following hold:

1. API-level result shape/dtype contract remains preserved.
2. Divergence is deterministic and audit-logged.
3. Divergence class is mapped to a mitigation and bounded recovery path.
4. Unknown classes remain fail-closed.

## Packet Family Coverage

Threat coverage must be declared per packet (`FNP-P2C-001`..`FNP-P2C-009`) and
included in packet risk notes and parity gates before packet promotion.

## Executable Control Mapping

Executable mapping for each threat class is locked in:

- `security_control_checks_v1.yaml`

This map is machine-validated by the `security_contracts` suite and requires,
per threat class:

1. at least one executable check
2. at least one fixture hook
3. explicit compatibility drift gate
4. explicit override-audit requirement
5. required structured log evidence fields

## Structured Log Evidence Contract

All runtime-policy checks for this matrix must emit deterministic JSONL entries
with:

- `fixture_id`
- `seed`
- `mode`
- `env_fingerprint`
- `artifact_refs`
- `reason_code`

The end-to-end gate command is:

```bash
scripts/e2e/run_security_policy_gate.sh
```

The gate writes a timestamped log under `artifacts/logs/` and fails if any
threat-control mapping, fail-closed requirement, or log field contract is
violated.
