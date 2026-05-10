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

## 3. Resolved Edges and Residual Breadth (Tagged)

| Edge ID | Current status | Risk | Evidence owner | Residual criteria |
|---|---|---|---|---|
| `P2C006-U01` | Legacy warning behavior around `broadcast_arrays` writeability (`FutureWarning`/`DeprecationWarning`) is subtle and version-sensitive. | medium | `bd-23m.17.2` | Contract table formalizes warning-class handling and reason-code normalization policy for strict/hardened replay. |
| `P2C006-U02` | `op_axes` remap and reduction-axis marker coverage exists for the current packet scope; full iterator-path parity breadth remains residual. | high | packet-D/F evidence | expand oracle fixtures for axis-remap, forced-reduction, and no-broadcast decision points |
| `P2C006-U03` | High-arity broadcast behavior has current differential/adversarial coverage, including high-rank broadcast fixture evidence. | medium | `differential_metamorphic_adversarial_evidence.json` | extend `>64` operand corpus and mismatch minimization artifacts |
| `P2C006-U04` | Dangerous writeable overlapping stride-view policy has threat controls and Packet-H replay/isomorphism evidence; broader overlap integrations remain monitored. | medium | risk note + workflow artifacts | keep fail-closed/full-validate boundaries stable while adding deeper overlap-risk replay coverage |
| `P2C006-U05` | End-to-end replay narratives for stride-tricks + iterator interoperability are present through packet workflow artifacts. | low | `workflow_scenario_packet006_*` artifacts | expand journey breadth for hostile broadcast/stride and iterator-sensitive view integrations |

## 4. Verification Hooks

| Verification lane | Hook | Artifact target |
|---|---|---|
| Unit/property | stride-tricks API law tests for readonly, no-broadcast mismatch classes, dtype preservation, high-arity broadcast shapes, and reason-code logging | packet-E evidence and `crates/fnp-conformance/fixtures/shape_stride_cases.json` |
| Differential/metamorphic/adversarial | `broadcast_to`/`broadcast_arrays`/`broadcast_shapes` result and error-family comparisons against oracle across adversarial shape corpora | `differential_metamorphic_adversarial_evidence.json` |
| E2E/workflow | packet journeys spanning `as_strided`, broadcast operations, and iterator-consuming paths with strict/hardened replay logs | `workflow_scenario_packet006_*` artifacts |
| Structured logging | ensure tests and e2e artifacts emit `fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code` | `artifacts/contracts/test_logging_contract_v1.json`, test-contract gate outputs |

## 5. Method-Stack Artifacts and EV Gate

- Alien decision contract: any hardened-mode mitigation on stride-tricks/broadcast boundaries must record state, action, expected-loss rationale, and replay metadata.
- Optimization gate: no stride/broadcast optimization accepted without baseline/profile + one lever + isomorphism proof.
- EV gate: optimization levers promoted only when `EV >= 2.0`; otherwise tracked as deferred parity debt.
- RaptorQ scope: packet closure (`bd-23m.17.9`) includes sidecar/scrub/decode-proof for final evidence bundle.

### Packet-H Closure (`bd-23m.17.8`)

- Accepted lever: `P2C006-H-LEVER-001` adds identical-shape and same-rank fast paths in `broadcast_shape`.
- Baseline/rebaseline profile artifact: `artifacts/phase2c/FNP-P2C-006/optimization_profile_report.json`.
- Isomorphism proof artifact: `artifacts/phase2c/FNP-P2C-006/optimization_profile_isomorphism_evidence.json`.
- Post-change regression evidence:
  - unit/property lane rerun: `rch exec -- cargo test -p fnp-ndarray` (34/34 pass).
  - e2e lane rerun: `rch exec -- cargo run -p fnp-conformance --bin run_workflow_scenario_gate -- --log-path artifacts/phase2c/FNP-P2C-006/workflow_scenario_packet006_opt_e2e.jsonl --artifact-index-path artifacts/phase2c/FNP-P2C-006/workflow_scenario_packet006_opt_artifact_index.json --report-path artifacts/phase2c/FNP-P2C-006/workflow_scenario_packet006_opt_reliability.json --retries 0 --flake-budget 0 --coverage-floor 1.0` (298/298 pass, coverage 1.0).
- Packet-H e2e artifacts:
  - `artifacts/phase2c/FNP-P2C-006/workflow_scenario_packet006_opt_e2e.jsonl`
  - `artifacts/phase2c/FNP-P2C-006/workflow_scenario_packet006_opt_artifact_index.json`
  - `artifacts/phase2c/FNP-P2C-006/workflow_scenario_packet006_opt_reliability.json`
- Measured deltas: `p50 -39.003%`, `p95 -40.331%`, `p99 -44.177%`, throughput gains `p50 +63.944%`, `p95 +67.590%`.
- EV outcome: `24.0` (`>= 2.0`), promoted.
- Isomorphism checks: identical-shape, same-rank axiswise merge, rank-mismatch merge, and incompatible-shape rejection paths match baseline behavior.

### Packet-I Closure (`bd-23m.17.9`)

- Final evidence index: `artifacts/phase2c/FNP-P2C-006/final_evidence_pack.json`.
- Packet readiness gate report: `artifacts/phase2c/FNP-P2C-006/packet_readiness_report.json` with `status=ready`.
- Packet parity summary/gates:
  - `artifacts/phase2c/FNP-P2C-006/fixture_manifest.json`
  - `artifacts/phase2c/FNP-P2C-006/parity_gate.yaml`
  - `artifacts/phase2c/FNP-P2C-006/parity_report.json`
- Durability artifacts:
  - `artifacts/phase2c/FNP-P2C-006/parity_report.raptorq.json`
  - `artifacts/phase2c/FNP-P2C-006/parity_report.scrub_report.json`
  - `artifacts/phase2c/FNP-P2C-006/parity_report.decode_proof.json`
- Final parity signals: `strict_parity=1.0`, `hardened_parity=1.0`, `compatibility_drift_hash=sha256:f2ac7bc69c9f6c16f4593c9dac0e38255d8f7c90918768ce46908845708e13ae`.

## 6. Rollback Handle

If packet-local behavior drift is detected, rollback to the previous packet baseline by reverting `artifacts/phase2c/FNP-P2C-006/*` and restoring the last green stride/broadcast conformance report set.
