# RAPTORQ_STATUS

Status: first implementation delivered.

Implemented artifacts:
- `artifacts/raptorq/conformance_bundle_v1.sidecar.json`
- `artifacts/raptorq/conformance_bundle_v1.scrub_report.json`
- `artifacts/raptorq/conformance_bundle_v1.decode_proof.json`
- `artifacts/raptorq/benchmark_bundle_v1.sidecar.json`
- `artifacts/raptorq/benchmark_bundle_v1.scrub_report.json`
- `artifacts/raptorq/benchmark_bundle_v1.decode_proof.json`

Current scope limitations:
1. Bundle schemas are project-local and not yet versioned as external stable contracts.
2. Multi-symbol recovery stress profiles now run through the RaptorQ stress gate and record a recovery matrix with one- and two-symbol source-loss scenarios.
3. Artifact signing/attestation is not implemented yet.

Next expansion:
- promote recovery-matrix evidence into long-lived release packets,
- add cryptographic attestation chain,
- integrate sidecar generation into CI gate pipeline.
