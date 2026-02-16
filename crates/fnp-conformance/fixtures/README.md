# Conformance Fixtures

This folder stores normalized oracle-vs-target fixtures for `fnp-conformance`.

- `smoke_case.json`: minimal bootstrap fixture ensuring harness wiring works.
- `shape_stride_cases.json`: broadcast legality, stride derivation, and stride-tricks API checks (`as_strided`, `broadcast_to`, `sliding_window_view`).
- `dtype_promotion_cases.json`: deterministic promotion-table checks.
- `runtime_policy_cases.json`: strict/hardened fail-closed policy checks with structured log context.
- `runtime_policy_adversarial_cases.json`: malformed/unknown wire-class inputs proving fail-closed behavior.
- `io_adversarial_cases.json`: parser-boundary adversarial IO corpus with severity-classified failure expectations.
- `linalg_differential_cases.json`: packet `FNP-P2C-008` differential oracle corpus for solver/factorization/spectral/tolerance/backend/policy seams.
- `linalg_metamorphic_cases.json`: deterministic linalg metamorphic invariants (solve scaling, qr determinism, lstsq tuple consistency).
- `linalg_adversarial_cases.json`: hostile linalg inputs with expected fail-closed reason-code classes.
- `ufunc_input_cases.json`: differential ufunc/reduction input corpus.
- `ufunc_metamorphic_cases.json`: deterministic metamorphic checks (commutativity, linearity).
- `ufunc_adversarial_cases.json`: adversarial ufunc inputs expecting controlled errors.
- `workflow_scenario_corpus.json`: user workflow golden journeys linking differential fixtures, e2e scripts, and prioritized gap beads.
- `oracle_outputs/ufunc_oracle_output.json`: captured NumPy oracle outputs.
- `oracle_outputs/ufunc_differential_report.json`: comparator report against current Rust implementation.
- `oracle_outputs/linalg_differential_report.json`: machine-readable mismatch report for packet `FNP-P2C-008` differential checks.
