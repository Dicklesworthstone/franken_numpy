# FNP-P2C-005 Behavior Extraction Ledger

Packet: `FNP-P2C-005-A`  
Subsystem: `Ufunc dispatch + gufunc signature`

## 1. Observable Contract Ledger

| Contract ID | Observable behavior | Strict mode expectation | Hardened mode expectation | Legacy anchors |
|---|---|---|---|---|
| `P2C005-C01` | `sig` and `signature` are mutually exclusive | fail with stable type error on dual specification | same fail-closed behavior | `ufunc_object.c:3782`, `test_ufunc.py:51` |
| `P2C005-C02` | signature grammar errors must produce deterministic failure class | parse failure with stable diagnostic family | parse failure with bounded diagnostic addenda only | `ufunc_object.c:307`, `test_ufunc.py:426`, `test_ufunc.py:553` |
| `P2C005-C03` | override dispatch respects `__array_ufunc__` precedence | use override path matching legacy order | same order; reject malformed override payloads fail-closed | `override.c:206`, `test_umath.py:3447`, `test_umath.py:3973` |
| `P2C005-C04` | gufunc exception propagation preserves error visibility | propagate loop exceptions (no silent swallow) | same, plus audit reason code on policy-mediated fallback | `test_umath.py:5094` |
| `P2C005-C05` | dispatch loop/type selection is deterministic for fixed inputs/signature | deterministic loop selection and result dtype | same deterministic selection; unknown incompatibilities fail-closed | `dispatching.h:23`, `ufunc_object.c:2339`, `ufunc_object.c:4688` |
| `P2C005-C06` | reduction wrapper path honors axis/keepdims contract | stable reduction semantics and axis bounds checks | same semantics; add bounded guardrails for hostile axis payloads | `reduction.c:178`, `ufunc_object.c:2561` |

## 2. Compatibility Invariants

1. Signature normalization invariant: canonical input signature representation is unique for equivalent user inputs.
2. Dispatch determinism invariant: same operand dtypes + signature + method => same selected kernel path.
3. Override precedence invariant: no inversion in override vs built-in dispatch ordering.
4. Error-shape invariant: signature parse failures and conflict failures remain class-stable.

## 3. Resolved Edges and Residual Breadth (Tagged)

| Edge ID | Current status | Risk | Evidence owner | Residual criteria |
|---|---|---|---|---|
| `P2C005-U01` | Exact diagnostic string stability for signature parse errors may drift across legacy micro-revisions. | medium | `bd-23m.16.6` | Differential harness stores normalized error-class taxonomy and verifies containment rules instead of exact full-string equality. |
| `P2C005-U02` | Custom-loop registration semantics for user dtypes remain residual breadth beyond the current packet matrix. | high | packet-D/F evidence | add loop registry model and conformance fixtures for registration + dispatch resolution |
| `P2C005-U03` | Override dispatch interactions have packet-scoped logging and targeted coverage; mixed subclass hierarchy breadth remains residual. | medium | `unit_property_evidence.json` + `differential_metamorphic_adversarial_evidence.json` | expand subclass permutation matrix while preserving `reason_code=override_precedence` structured logs |
| `P2C005-U04` | Gufunc signature edge cases have current fixture coverage; optional-dimension (`?`) shrink breadth remains residual. | medium | `unit_property_evidence.json` + packet-F evidence | add deeper property-based signature corpus with deterministic shrink replay fields (`seed`, `artifact_refs`) |

## 4. Verification Hooks

| Verification lane | Hook | Artifact target |
|---|---|---|
| Unit/property | signature parse corpus, override precedence properties, deterministic replay-log checks, and reason-code registry lock | `unit_property_evidence.json` |
| Differential/metamorphic/adversarial | ufunc differential suite with signature/override/gufunc adversarial fixtures and targeted relation checks | `differential_metamorphic_adversarial_evidence.json` |
| E2E/workflow | strict/hardened workflow replay and optimization-isomorphism evidence for current ufunc dispatch paths | `workflow_scenario_packet005_opt_*` artifacts + `optimization_profile_isomorphism_evidence.json` |
| Structured logging | ensure packet tests emit `fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code` | `artifacts/contracts/test_logging_contract_v1.json`, test-contract gate outputs |

## 5. Method-Stack Artifacts and EV Gate

- Alien decision contract: dispatch/override policy decisions must log state, action, and expected-loss rationale when policy mediation occurs.
- Optimization gate: no dispatch optimization accepted without baseline/profile + single-lever + isomorphism proof artifact.
- EV gate: dispatch optimization levers promoted only when `EV >= 2.0`; otherwise tracked as deferred research.
- RaptorQ scope: durable packet evidence bundle for `FNP-P2C-005` includes sidecar/scrub/decode-proof links at packet-I closure.

### Packet-H Closure (`bd-23m.16.8`)

- Accepted lever: `P2C005-H-LEVER-001` adds a same-shape fast path in `UFuncArray::elementwise_binary`.
- Baseline/rebaseline profile artifact: `artifacts/phase2c/FNP-P2C-005/optimization_profile_report.json`.
- Isomorphism proof artifact: `artifacts/phase2c/FNP-P2C-005/optimization_profile_isomorphism_evidence.json`.
- Post-change regression evidence:
  - unit/property lane rerun: `rch exec -- cargo test -p fnp-ufunc` (17/17 pass).
  - e2e lane rerun: `rch exec -- cargo run -p fnp-conformance --bin run_workflow_scenario_gate -- --log-path artifacts/phase2c/FNP-P2C-005/workflow_scenario_packet005_opt_e2e.jsonl --artifact-index-path artifacts/phase2c/FNP-P2C-005/workflow_scenario_packet005_opt_artifact_index.json --report-path artifacts/phase2c/FNP-P2C-005/workflow_scenario_packet005_opt_reliability.json --retries 0 --flake-budget 0 --coverage-floor 1.0` (298/298 pass, coverage 1.0).
- Packet-H e2e artifacts:
  - `artifacts/phase2c/FNP-P2C-005/workflow_scenario_packet005_opt_e2e.jsonl`
  - `artifacts/phase2c/FNP-P2C-005/workflow_scenario_packet005_opt_artifact_index.json`
  - `artifacts/phase2c/FNP-P2C-005/workflow_scenario_packet005_opt_reliability.json`
- Measured deltas: `p50 -69.488%`, `p95 -72.205%`, `p99 -72.102%`, throughput gains `p50 +227.744%`, `p95 +259.779%`.
- EV outcome: `24.0` (`>= 2.0`), promoted.
- Isomorphism checks: same-shape, broadcast-row, scalar-broadcast, and shape-error paths all match baseline behavior.

### Packet-I Closure (`bd-23m.16.9`)

- Final evidence index: `artifacts/phase2c/FNP-P2C-005/final_evidence_pack.json`.
- Packet readiness gate report: `artifacts/phase2c/FNP-P2C-005/packet_readiness_report.json` with `status=ready`.
- Packet parity summary/gates:
  - `artifacts/phase2c/FNP-P2C-005/fixture_manifest.json`
  - `artifacts/phase2c/FNP-P2C-005/parity_gate.yaml`
  - `artifacts/phase2c/FNP-P2C-005/parity_report.json`
- Durability artifacts:
  - `artifacts/phase2c/FNP-P2C-005/parity_report.raptorq.json`
  - `artifacts/phase2c/FNP-P2C-005/parity_report.scrub_report.json`
  - `artifacts/phase2c/FNP-P2C-005/parity_report.decode_proof.json`

## 6. Rollback Handle

If packet-local changes cause compatibility drift, rollback to previous packet boundary baseline by reverting `artifacts/phase2c/FNP-P2C-005/*` and restoring prior `fnp-ufunc` dispatch behavior snapshot from the last green differential report.
