# PHASE2C Artifact Topology V1

Schema lock id: `phase2c-contract-v1`  
Version: `1`

Normative packet path:

- `artifacts/phase2c/<packet_id>/legacy_anchor_map.md`
- `artifacts/phase2c/<packet_id>/contract_table.md`
- `artifacts/phase2c/<packet_id>/fixture_manifest.json`
- `artifacts/phase2c/<packet_id>/parity_gate.yaml`
- `artifacts/phase2c/<packet_id>/risk_note.md`
- `artifacts/phase2c/<packet_id>/parity_report.json`
- `artifacts/phase2c/<packet_id>/parity_report.raptorq.json`
- `artifacts/phase2c/<packet_id>/parity_report.decode_proof.json`

Readiness rule:

- Any missing required file => packet status `not_ready`.
- Any missing mandatory field in required files => packet status `not_ready`.
- Strict drift budget in parity gate must remain `0.0`.
