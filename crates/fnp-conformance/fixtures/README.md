# Conformance Fixtures

This folder stores normalized oracle-vs-target fixtures for `fnp-conformance`.

- `smoke_case.json`: minimal bootstrap fixture ensuring harness wiring works.
- `shape_stride_cases.json`: broadcast legality + stride derivation checks.
- `dtype_promotion_cases.json`: deterministic promotion-table checks.
- `runtime_policy_cases.json`: strict/hardened fail-closed policy checks with structured log context.
- `runtime_policy_adversarial_cases.json`: malformed/unknown wire-class inputs proving fail-closed behavior.
- `ufunc_input_cases.json`: differential ufunc/reduction input corpus.
- `ufunc_metamorphic_cases.json`: deterministic metamorphic checks (commutativity, linearity).
- `ufunc_adversarial_cases.json`: adversarial ufunc inputs expecting controlled errors.
- `oracle_outputs/ufunc_oracle_output.json`: captured NumPy oracle outputs.
- `oracle_outputs/ufunc_differential_report.json`: comparator report against current Rust implementation.
