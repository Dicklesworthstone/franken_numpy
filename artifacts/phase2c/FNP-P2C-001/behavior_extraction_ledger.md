# FNP-P2C-001 Behavior Extraction Ledger

Packet: `FNP-P2C-001-A`  
Subsystem: `Shape/reshape legality`

## 1. Observable Contract Ledger

| Contract ID | Observable behavior | Strict mode expectation | Hardened mode expectation | Legacy anchors |
|---|---|---|---|---|
| `P2C001-C01` | Reshape with one `-1` infers exactly one dimension from prior element count | inference succeeds only when `old_size % known_product == 0` | same outward rule; malformed payloads fail-closed with stable error class | `shape.c:246`, `shape.c:467` |
| `P2C001-C02` | Multiple unknown dimensions are rejected | deterministic failure when more than one unknown dimension appears | same failure semantics with bounded diagnostic addenda | `shape.c:485` |
| `P2C001-C03` | Reshape preserves element-count invariance | reject shape requests where product differs from source size | same external contract; overflow/malformed arithmetic must fail-closed | `shape.c:499`, `shape.c:505` |
| `P2C001-C04` | Broadcast legality uses right-aligned merge with singleton expansion | legal when dimensions are equal or one side is `1`; otherwise reject | same legality rule, same rejection class for incompatible shapes | `shape.c` reshape/broadcast family + `test_shape_base.py:86`, `test_shape_base.py:111` |
| `P2C001-C05` | C/F contiguous stride derivation is deterministic for valid shape + item size | computed byte strides are deterministic and order-consistent | same deterministic strides; reject invalid item size/overflow | `shape.c:271`, `_attempt_nocopy_reshape` stride handling + Rust SCE |
| `P2C001-C06` | Legacy `reshape` API delegates to canonical reshape path | `PyArray_Reshape` behavioral surface matches `PyArray_Newshape` contract | same API behavior under hardened validation | `shape.c:325`, `shape.c:333` |
| `P2C001-C07` | Large broadcasted shapes remain shape-correct (subject to platform limits) | preserve broadcasted shape outcomes in high-cardinality cases | preserve outcome; fail-closed when resource/security gates trigger | `test_shape_base.py:450` |

## 2. Compatibility Invariants

1. Element conservation invariant: reshape never changes logical element count.
2. Unknown-dimension uniqueness invariant: exactly zero or one `-1` is admissible.
3. Broadcast determinism invariant: same input shapes always produce same merged shape or same incompatibility class.
4. Order-sensitive stride invariant: C and F contiguous stride derivations remain deterministic for identical inputs.
5. Error-class stability invariant: incompatible reshape/broadcast failures remain class-stable across strict and hardened modes.

## 3. Undefined or Under-Specified Edges (Tagged)

| Unknown ID | Description | Risk | Owner bead | Closure criteria |
|---|---|---|---|---|
| `P2C001-U01` | Full no-copy reshape alias-preservation parity (`_attempt_nocopy_reshape`) is not yet modeled end-to-end in Rust. | high | `bd-23m.12.4` | Implement alias-sensitive reshape/view boundary and land matching contract table entries with explicit invariants. |
| `P2C001-U02` | Exact legacy error-text equivalence for reshape mismatch cases is not yet differential-verified (class is covered, text parity is partial). | medium | `bd-23m.12.6` | Differential oracle fixtures include mismatch/error taxonomy with strict drift budget `0.0` for class/type behavior. |
| `P2C001-U03` | Hardened-mode budget policy for extreme broadcast/reshape requests (size/resource thresholds) is not yet codified for this packet. | high | `bd-23m.12.3` | Threat model + policy table define explicit bounded-mode behavior and reason-code logging. |
| `P2C001-U04` | Replay-forensics link from packet-specific shape failures to workflow scenarios is not yet packaged. | medium | `bd-23m.12.7` | E2E scenarios emit packet-scoped `fixture_id`, `mode`, `artifact_refs`, and `reason_code` for shape-law transitions. |

## 4. Planned Verification Hooks

| Verification lane | Planned hook | Artifact target |
|---|---|---|
| Unit/property | Extend SCE tests for reshape inference, overflow boundaries, and broadcast edge matrices | `crates/fnp-ndarray/src/lib.rs` + packet-E additions |
| Differential/metamorphic/adversarial | Expand shape-stride fixture corpus and add oracle-comparison reshaping suites (including hostile dimensions and mismatch classes) | `crates/fnp-conformance/fixtures/shape_stride_cases.json` + packet-F fixture expansions |
| E2E | Add packet-scoped scenario chain for reshape+broadcast workflows in strict and hardened replay modes | packet-G script/artifact set under `scripts/e2e/` + `artifacts/logs/` |
| Structured logging | Enforce required fields across packet checks: `fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code` | `artifacts/contracts/test_logging_contract_v1.json`, `scripts/e2e/run_test_contract_gate.sh` |

## 5. Method-Stack Artifacts and EV Gate

- Alien decision contract: policy-mediated shape/broadcast decisions must include state/action/loss rationale when hardened guardrails alter flow.
- Optimization gate: reshape/broadcast performance work requires baseline/profile + one-lever + isomorphism proof artifacts.
- EV gate: optimization levers are promotable only when `EV >= 2.0`; otherwise remain explicit deferred debt.
- RaptorQ scope: packet-I must attach sidecar/scrub/decode-proof artifacts for parity bundle durability.

## 6. Rollback Handle

If packet-local changes introduce compatibility drift, revert packet-local artifacts under `artifacts/phase2c/FNP-P2C-001/` and restore the last green `fnp-ndarray`/conformance shape-stride evidence baseline prior to the drift.
