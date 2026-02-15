# FNP-P2C-006 Behavior Extraction Ledger

Packet: `FNP-P2C-006-A`  
Subsystem: `Stride-tricks and broadcasting API`

## 1. Observable Contract Ledger

| Contract ID | Observable behavior | Strict mode expectation | Hardened mode expectation | Legacy anchors |
|---|---|---|---|---|
| `P2C006-C01` | Broadcast shape merge is deterministic and right-aligned across rank mismatches | identical merge/reject outcomes for equal inputs | same outcome; incompatible or unknown shapes fail-closed with normalized reason codes | `_stride_tricks_impl.py:446`, `_stride_tricks_impl.py:467`, `test_stride_tricks.py:304`, `test_stride_tricks.py:345` |
| `P2C006-C02` | `broadcast_to` validates requested shape (non-negative, scalar/non-scalar constraints) | raise deterministic `ValueError` class on invalid shape requests | same class with bounded diagnostics only | `_stride_tricks_impl.py:373`, `_stride_tricks_impl.py:377`, `_stride_tricks_impl.py:378`, `test_stride_tricks.py:268` |
| `P2C006-C03` | `broadcast_to` returns a readonly broadcast view | output is non-writeable and assignment fails | same readonly behavior; hostile write attempts remain fail-closed | `_stride_tricks_impl.py:401`, `_stride_tricks_impl.py:443`, `test_stride_tricks.py:580` |
| `P2C006-C04` | `broadcast_arrays` uses writable compatibility path for broadcasted outputs | preserve legacy compatibility (writeable path + warning behavior) | preserve observable output while logging policy reason code for potentially unsafe writes | `_stride_tricks_impl.py:515`, `_stride_tricks_impl.py:579`, `_stride_tricks_impl.py:580`, `test_stride_tricks.py:587`, `test_stride_tricks.py:629` |
| `P2C006-C05` | `as_strided` preserves dtype and supports subclass passthrough via `subok` | dtype/subclass semantics match legacy for supported inputs | same semantics; unsafe writeability patterns audited in ledger | `_stride_tricks_impl.py:38`, `_stride_tricks_impl.py:104`, `_stride_tricks_impl.py:106`, `test_stride_tricks.py:362`, `test_stride_tricks.py:543` |
| `P2C006-C06` | Iterator broadcast planning sets zero strides for broadcast dimensions and rejects explicit no-broadcast mismatches | zero-stride and rejection semantics match legacy iterator rules | same semantics; malformed axis remaps and unknown flags fail-closed | `nditer_constr.c:1463`, `nditer_constr.c:1597`, `nditer_constr.c:1605`, `nditer_constr.c:1830` |
| `P2C006-C07` | Broadcast mismatch diagnostics differentiate generic mismatch vs non-broadcastable output mismatch | error family and operand/broadcast shape relationship preserved | same family with normalized reason-code taxonomy | `nditer_constr.c:1714`, `nditer_constr.c:1830`, `test_stride_tricks.py:268` |
| `P2C006-C08` | Iterator shape/stride introspection must remain traversal-consistent | shape and compatible-stride derivation preserve iterator traversal law | same law; unsupported mode/class combinations fail-closed | `nditer_api.c:1001`, `nditer_api.c:1059`, `_core/tests/test_nditer.py:2586` |

## 2. Compatibility Invariants

1. Broadcast determinism invariant: broadcast result shape is unique for a fixed ordered input tuple.
2. Broadcast rejection invariant: incompatible shape tuples always fail with stable failure class.
3. Readonly invariant: `broadcast_to` result remains non-writeable in strict and hardened modes.
4. Stride-view identity invariant: `as_strided` does not alter dtype identity even for structured/custom dtypes.
5. No-broadcast invariant: operands marked non-broadcastable cannot be implicitly expanded.
6. Iterator traversal invariant: compatible strides and exposed shape remain consistent with iterator traversal order.

## 3. Undefined or Under-Specified Edges (Tagged)

| Unknown ID | Description | Risk | Owner bead | Closure criteria |
|---|---|---|---|---|
| `P2C006-U01` | Legacy warning behavior around `broadcast_arrays` writeability (`FutureWarning`/`DeprecationWarning`) is subtle and version-sensitive. | medium | `bd-23m.17.2` | Contract table formalizes warning-class handling and reason-code normalization policy for strict/hardened replay. |
| `P2C006-U02` | Full `op_axes` remap semantics and reduction-axis markers in iterator paths are not yet represented in Rust. | high | `bd-23m.17.4` | Rust module boundary skeleton encodes axis-remap, forced-reduction, and no-broadcast decision points with explicit tests. |
| `P2C006-U03` | High-arity broadcast behavior (`>64` args) needs explicit parity corpus and mismatch minimization strategy. | medium | `bd-23m.17.6` | Differential/adversarial fixtures include high-arity cases and reproducer artifacts with stable reduction to minimal failing tuples. |
| `P2C006-U04` | Exact policy for dangerous writeable overlapping stride views (`as_strided`) in hardened mode needs closure. | high | `bd-23m.17.3` | Threat model + allowlist contract defines fail-closed/full-validate boundaries for overlap-risk operations. |
| `P2C006-U05` | End-to-end replay narratives for stride-tricks + iterator interoperability are absent. | medium | `bd-23m.17.7` | Add e2e journey scripts with deterministic replay metadata and links to unit/differential artifacts. |

## 4. Planned Verification Hooks

| Verification lane | Planned hook | Artifact target |
|---|---|---|
| Unit/property | Add stride-tricks API law tests: readonly, no-broadcast mismatch classes, dtype-preservation, and high-arity broadcast-shapes | `crates/fnp-conformance/fixtures/shape_stride_cases.json` expansion + packet-E tests (`bd-23m.17.5`) |
| Differential/metamorphic/adversarial | Compare `broadcast_to`/`broadcast_arrays`/`broadcast_shapes` results and error families against oracle across adversarial shape corpora | `crates/fnp-conformance/src/ufunc_differential.rs` packet-F extension + dedicated fixture corpus (`bd-23m.17.6`) |
| E2E | Add packet journey spanning `as_strided` -> broadcast operations -> iterator-consuming operation with strict/hardened replay logs | `scripts/e2e/run_workflow_scenario_gate.sh` scenario additions (`bd-23m.17.7`) |
| Structured logging | Ensure tests and e2e artifacts emit `fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code` | `artifacts/contracts/test_logging_contract_v1.json`, `crates/fnp-conformance/src/lib.rs` runtime log plumbing |

## 5. Method-Stack Artifacts and EV Gate

- Alien decision contract: any hardened-mode mitigation on stride-tricks/broadcast boundaries must record state, action, expected-loss rationale, and replay metadata.
- Optimization gate: no stride/broadcast optimization accepted without baseline/profile + one lever + isomorphism proof.
- EV gate: optimization levers promoted only when `EV >= 2.0`; otherwise tracked as deferred parity debt.
- RaptorQ scope: packet closure (`bd-23m.17.9`) must include sidecar/scrub/decode-proof for final evidence bundle.

## 6. Rollback Handle

If packet-local behavior drift is detected, rollback to the previous packet baseline by reverting `artifacts/phase2c/FNP-P2C-006/*` and restoring the last green stride/broadcast conformance report set.
