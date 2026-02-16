# FNP-P2C-008 Behavior Extraction Ledger

Packet: `FNP-P2C-008-A`  
Subsystem: `linalg bridge first wave`

## 1. Observable Contract Ledger

| Contract ID | Observable behavior | Strict mode expectation | Hardened mode expectation | Legacy anchors |
|---|---|---|---|---|
| `P2C008-C01` | linalg API shape/ndim preconditions and `LinAlgError` failure classes are deterministic. | invalid shape/ndim requests reject with stable error-class family. | same class with bounded diagnostics and audit linkage. | `_linalg.py:115`, `_linalg.py:229`, `_linalg.py:243`, `test_linalg.py:524` |
| `P2C008-C02` | `solve`/`inv` contract preserves deterministic solution/inversion classes and singular failure behavior. | singular/incompatible systems produce legacy-compatible failure classes. | same classes; no permissive fallback for incompatible solves. | `_linalg.py:363`, `_linalg.py:536`, `umath_linalg.cpp:1765`, `umath_linalg.cpp:1839`, `test_linalg.py:818` |
| `P2C008-C03` | decomposition families (`cholesky`, `qr`, `svd`) preserve mode-specific output and failure semantics. | decomposition outputs and non-convergence/invalid-input failures match legacy class behavior. | same behavior with bounded guardrails and deterministic reason-code emission. | `_linalg.py:778`, `_linalg.py:965`, `_linalg.py:1668`, `umath_linalg.cpp:2010`, `umath_linalg.cpp:2999`, `umath_linalg.cpp:3362` |
| `P2C008-C04` | spectral families (`eig`, `eigvals`, `eigh`, `eigvalsh`) preserve output-shape and convergence failure classes. | deterministic eigen output/error classes for fixed input class and `UPLO` branches. | same classes with fail-closed handling for unknown/incompatible metadata. | `_linalg.py:1170`, `_linalg.py:1362`, `_linalg.py:1515`, `umath_linalg.cpp:2494`, `umath_linalg.cpp:1601` |
| `P2C008-C05` | `lstsq` output tuple semantics (solution/residual/rank/singular values) remain class-stable across shapes. | deterministic output/failure classes for supported least-squares inputs. | same behavior with bounded validation and reason-code logging. | `_linalg.py:2418`, `umath_linalg.cpp:4125`, `test_regression.py:143` |
| `P2C008-C06` | `det`/`slogdet`/`norm`/`matrix_rank`/`pinv` preserve deterministic value/error class behavior for supported inputs. | deterministic output class and stable failures on invalid/singular/tolerance-edge cases. | same classes with bounded tolerance policy checks. | `_linalg.py:2272`, `_linalg.py:2356`, `_linalg.py:2599`, `_linalg.py:2035`, `_linalg.py:2154`, `umath_linalg.cpp:4284` |
| `P2C008-C07` | backend adapter and lapack-lite error hook behavior preserve deterministic failure signaling. | backend parameter and error signaling classes remain stable. | same class with fail-closed behavior for unsupported backend states. | `lapack_litemodule.c:170`, `lapack_litemodule.c:351`, `test_linalg.py:1991` |
| `P2C008-C08` | stacked/batched matrix pathways preserve deterministic broadcast/output-shape class behavior. | deterministic class behavior across batched linalg routes. | same behavior with bounded validation of hostile batch shapes. | `umath_linalg.cpp:4284`, `_linalg.py:965`, `_linalg.py:1668`, `test_linalg.py:1753` |

## 2. Compatibility Invariants

1. Shape-legality invariant: linalg entrypoints reject incompatible dimensionality with stable error classes.
2. Error-taxonomy invariant: singular/non-convergence/invalid-input classes remain deterministic and mode-consistent.
3. Decomposition-mode invariant: mode flags (QR/SVD/UPLO) produce deterministic output-shape families.
4. Backend-bridge invariant: backend adapter state cannot silently alter API-visible success/failure class.
5. Batched-output invariant: stacked matrix routes preserve deterministic batch-shape/value class behavior.

## 3. Undefined or Under-Specified Edges (Tagged)

| Unknown ID | Description | Risk | Owner bead | Closure criteria |
|---|---|---|---|---|
| `P2C008-U01` | Exact tolerance/conditioning boundaries for near-singular cases may differ across backend realizations. | high | `bd-23m.19.2` | contract table pins tolerance-class rules and differential fixtures enforce class stability. |
| `P2C008-U02` | Full backend abstraction for lapack-lite vs future Rust/native backend needs explicit compatibility policy. | high | `bd-23m.19.4` | implementation plan defines backend seam and fail-closed fallback behavior for unsupported backend classes. |
| `P2C008-U03` | Differential corpus for non-convergence and mixed-dtype linalg edge cases is now implemented through packet-F fixtures and gate wiring. | closed | `bd-23m.19.6` | closed by packet-F fixture lanes in `crates/fnp-conformance/fixtures/linalg_*_cases.json` and conformance suite integration in `crates/fnp-conformance/src/lib.rs`. |
| `P2C008-U04` | E2E replay and forensic linkage for linalg-first workflows is not yet packet-specific. | medium | `bd-23m.19.7` | packet-G scenarios encode step-level replay logs and artifact cross-links. |

## 4. Planned Verification Hooks

| Verification lane | Planned hook | Artifact target |
|---|---|---|
| Unit/property | solver/decomposition/spectral/tolerance law suites with shrinkable counterexamples | `crates/fnp-conformance/fixtures/linalg_property_cases.json` (planned), structured JSONL logs |
| Differential/metamorphic/adversarial | packet-F harness checks singular/non-convergence/tolerance-edge/backend-policy classes | `crates/fnp-conformance/src/lib.rs` (`run_linalg_differential_suite`, `run_linalg_metamorphic_suite`, `run_linalg_adversarial_suite`), `crates/fnp-conformance/fixtures/linalg_differential_cases.json`, `crates/fnp-conformance/fixtures/linalg_metamorphic_cases.json`, `crates/fnp-conformance/fixtures/linalg_adversarial_cases.json` |
| E2E | linalg workflow scenarios chained with upstream/downstream packet behaviors | `scripts/e2e/run_linalg_bridge_journey.sh` (planned) |
| Structured logging | enforce mandatory packet logging schema for all linalg suites and gates | `artifacts/contracts/test_logging_contract_v1.json`, `scripts/e2e/run_test_contract_gate.sh` |

## 5. Method-Stack Artifacts and EV Gate

- Alien decision contract: solver/decomposition policy mediation must log state, action, and expected-loss rationale.
- Optimization gate: no linalg optimization is accepted without baseline/profile + single-lever + isomorphism proof artifact.
- EV gate: linalg optimization levers are promoted only when `EV >= 2.0`; otherwise tracked as deferred research debt.
- RaptorQ scope: packet `FNP-P2C-008` durable evidence bundle must include sidecar/scrub/decode-proof links at packet-I closure.

## 6. Rollback Handle

If packet-local linalg extraction drifts from contract intent, roll back `artifacts/phase2c/FNP-P2C-008/*` to the last green packet baseline and restore prior differential/security evidence references before continuing.
