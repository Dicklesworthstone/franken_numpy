# ISOMORPHISM_PROOF_ROUND1

## Change: SCE + dtype + runtime + ufunc/reduction + differential/raptorq artifacts

- Ordering preserved: yes; ufunc and reduction iterate deterministic C-order mappings.
- Tie-breaking unchanged: yes; broadcasting uses right-aligned dimension merge and fixed source-index mapping.
- Floating-point: identical arithmetic operators for implemented ops (`add/sub/mul/div/sum`), with IEEE-754 behavior for division edge cases.
- RNG seeds: N/A for this historical slice; later packets now implement
  `fnp-random` seed/state/replay contracts and preserve RNG-specific evidence
  under `artifacts/phase2c/FNP-P2C-007/`.
- Golden outputs: fixture-driven oracle/differential artifacts generated and checked.

## Behavioral Witnesses

- `shape_stride_cases.json` validates broadcast legality and stride derivation.
- `dtype_promotion_cases.json` validates deterministic promotion results.
- `runtime_policy_cases.json` validates strict/hardened fail-closed policy decisions.
- `ufunc_input_cases.json` + `oracle_outputs/ufunc_oracle_output.json` + `oracle_outputs/ufunc_differential_report.json` validate ufunc/reduction parity for the first wave.

## Durability Witnesses

- `artifacts/raptorq/conformance_bundle_v1.sidecar.json`
- `artifacts/raptorq/conformance_bundle_v1.scrub_report.json`
- `artifacts/raptorq/conformance_bundle_v1.decode_proof.json`
- `artifacts/raptorq/benchmark_bundle_v1.sidecar.json`
- `artifacts/raptorq/benchmark_bundle_v1.scrub_report.json`
- `artifacts/raptorq/benchmark_bundle_v1.decode_proof.json`

## Regression Surface

- No destructive or behavior-altering repair paths introduced.
- Unknown/incompatible semantics still fail closed in runtime policy.
- Oracle capture records source mode (`legacy`, `system`, or `pure_python_fallback`) to keep provenance explicit.
