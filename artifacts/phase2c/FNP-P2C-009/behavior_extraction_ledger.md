# FNP-P2C-009 Behavior Extraction Ledger

Packet: `FNP-P2C-009-A`  
Subsystem: `NPY/NPZ IO contract`

## 1. Observable Contract Ledger

| Contract ID | Observable behavior | Strict mode expectation | Hardened mode expectation | Legacy anchors |
|---|---|---|---|---|
| `P2C009-C01` | `.npy` magic/version parsing enforces prefix and supported version family. | malformed or unsupported magic/version rejects with stable failure class. | same class, no permissive fallback for unknown versions. | `_format_impl.py:206`, `_format_impl.py:230`, `test_format.py:815`, `test_format.py:836` |
| `P2C009-C02` | header schema parse requires exact keys (`shape`, `fortran_order`, `descr`) and bounded header size policy. | invalid/missing/extra keys, malformed shapes, and invalid descriptors reject deterministically. | same reject class with bounded diagnostics and audit linkage. | `_format_impl.py:621`, `test_format.py:865`, `test_format.py:878`, `test_format.py:900` |
| `P2C009-C03` | dtype descriptor serialization/deserialization preserves supported legacy dtype classes. | deterministic descriptor roundtrip for supported dtypes; unsupported classes reject with stable class. | same outward class; unknown/incompatible descriptor semantics fail closed. | `_format_impl.py:251`, `_format_impl.py:311` |
| `P2C009-C04` | `.npy` read/write roundtrip preserves shape/order/value class for C/F contiguous paths and deterministic non-contiguous handling. | roundtrip determinism for supported arrays; truncated payloads reject with stable class. | same contract with bounded read guards and deterministic reason-code emission. | `_format_impl.py:703`, `_format_impl.py:781` |
| `P2C009-C05` | object-array and user-defined-dtype pathways are gated by `allow_pickle` policy. | `allow_pickle=False` rejects object-array load/save classes deterministically. | same reject classes; no silent policy bypass. | `_format_impl.py:703`, `_format_impl.py:781`, `_npyio_impl.py:312`, `test_io.py:2722` |
| `P2C009-C06` | memmap surface enforces filename-only + non-object-dtype constraints with mode-legal behavior. | invalid memmap modes/dtypes reject with stable class; valid paths preserve shape/order contract. | same contract with bounded validation and fail-closed unknown metadata handling. | `_format_impl.py:893`, `test_io.py:2528` |
| `P2C009-C07` | `np.load` dispatch resolves `.npz` vs `.npy` vs pickle path by magic/prefix and `allow_pickle` gating. | deterministic dispatch and failure classes for unsupported/trusted pathways. | same dispatch with no widening of accepted unsafe classes. | `_npyio_impl.py:312` |
| `P2C009-C08` | `.npz` archive semantics preserve key naming (`arr_N` defaults), lazy loading, and compressed/uncompressed save/load parity. | deterministic key mapping + lazy member loading contract; duplicate key conflicts reject. | same behavior with bounded archive validation and deterministic audit fields. | `_npyio_impl.py:116`, `_npyio_impl.py:581`, `_npyio_impl.py:682`, `_npyio_impl.py:756`, `test_io.py:2549`, `test_io.py:2557` |

## 2. Compatibility Invariants

1. Header schema invariant: accepted headers must have exactly required keys and valid type classes.
2. Roundtrip invariant: supported `.npy`/`.npz` read-write cycles preserve array shape/order/value class deterministically.
3. Pickle policy invariant: object-array and pickle pathways are admitted only under explicit `allow_pickle` policy.
4. Dispatch invariant: `np.load` route selection (`npz`/`npy`/pickle) is deterministic for identical byte prefixes and policy inputs.
5. Archive key invariant: `.npz` key naming and lazy access behavior remain deterministic for fixed input argument order.

## 3. Resolved Edges and Residual Breadth (Tagged)

| Edge ID | Current status | Risk | Evidence owner | Residual criteria |
|---|---|---|---|---|
| `P2C009-U01` | Exact warning-message parity for Python-2 header fallback/filtering paths may vary by legacy revision. | medium | `bd-23m.20.6` | differential harness verifies failure/warning class containment instead of full-string equality. |
| `P2C009-U02` | Supported handle and memmap constraints are modeled for the packet scope; wider cross-platform pathlib, duck-typed handle, and OS-mode breadth remains residual. | high | packet-D/E evidence | expand file-like and memmap replay matrices when new platform-specific edge classes are admitted |
| `P2C009-U03` | `.npz` key naming, duplicate rejection, and member-order invariants have unit/property and optimization isomorphism evidence. | medium | packet-E/H evidence | deepen duplicate-name, archive-member ordering, compression, and hostile-size fixture breadth |
| `P2C009-U04` | Hostile header, malformed descriptor, pickle-policy, and archive-contract boundaries have packet-E/F coverage and fail-closed reason codes. | medium | risk note + adversarial fixtures | continue saturating large-header and malformed descriptor corpora as new IO envelopes are found |

## 4. Verification Hooks

| Verification lane | Hook | Artifact target |
|---|---|---|
| Unit/property | header parse, roundtrip, pickle policy, memmap, archive key, and reason-code law coverage | `artifacts/phase2c/FNP-P2C-009/unit_property_evidence.json`, `crates/fnp-io/src/lib.rs` tests |
| Differential/metamorphic/adversarial | fixture-driven IO differential/metamorphic/adversarial suite execution with reason-code mismatch artifacts | `crates/fnp-conformance/fixtures/io_differential_cases.json`, `crates/fnp-conformance/fixtures/io_metamorphic_cases.json`, `crates/fnp-conformance/fixtures/io_adversarial_cases.json`, oracle output artifacts |
| E2E/workflow | packet-009 workflow scenarios replaying golden and hostile IO lanes with step-level forensics logging | `scripts/e2e/run_io_contract_journey.sh`, `scripts/e2e/run_workflow_scenario_gate.sh`, `artifacts/phase2c/FNP-P2C-009/e2e_replay_forensics_evidence.json`, `workflow_scenario_packet009_*` artifacts |
| Structured logging | mandatory logging fields for packet IO tests and gates | `artifacts/contracts/test_logging_contract_v1.json`, packet-E evidence, test-contract gate outputs |

## 5. Method-Stack Artifacts and EV Gate

- Alien decision contract: parser and policy mediation decisions must log state, action, and expected-loss rationale where compatibility/security mediation occurs.
- Optimization gate: no IO parsing/perf optimization is accepted without baseline/profile + single-lever + isomorphism proof artifact.
- EV gate: IO optimization levers are promoted only when `EV >= 2.0`; otherwise tracked as deferred research debt.
- RaptorQ scope: packet `FNP-P2C-009` durable evidence bundle includes sidecar/scrub/decode-proof links from packet-I closure.

### Packet-H Closure (`bd-23m.20.8`)

- Accepted lever: `P2C009-H-LEVER-001` replaced `BTreeSet` with `HashSet` for NPZ member uniqueness tracking in `synthesize_npz_member_names`.
- Baseline/rebaseline profile artifact: `artifacts/phase2c/FNP-P2C-009/optimization_profile_report.json`.
- Isomorphism proof artifact: `artifacts/phase2c/FNP-P2C-009/optimization_profile_isomorphism_evidence.json`.
- Measured deltas: `p50 -48.925%`, `p95 -45.237%`, `p99 -38.036%`, throughput gains `p50 +95.792%`, `p95 +82.604%`.
- EV outcome: `24.0` (`>= 2.0`), promoted.
- Ordering/tie-break invariants preserved: positional `arr_N` emission then keyword insertion order; duplicate and empty-keyword error pathways unchanged.

### Packet-I Closure (`bd-23m.20.9`)

- Final evidence index: `artifacts/phase2c/FNP-P2C-009/final_evidence_pack.json`.
- Packet readiness gate report: `artifacts/phase2c/FNP-P2C-009/packet_readiness_report.json` with `status=ready`.
- Packet parity summary/gates:
  - `artifacts/phase2c/FNP-P2C-009/fixture_manifest.json`
  - `artifacts/phase2c/FNP-P2C-009/parity_gate.yaml`
  - `artifacts/phase2c/FNP-P2C-009/parity_report.json`
- Durability artifacts:
  - `artifacts/phase2c/FNP-P2C-009/parity_report.raptorq.json`
  - `artifacts/phase2c/FNP-P2C-009/parity_report.scrub_report.json`
  - `artifacts/phase2c/FNP-P2C-009/parity_report.decode_proof.json`

## 6. Rollback Handle

If packet-local IO extraction drifts from contract intent, roll back `artifacts/phase2c/FNP-P2C-009/*` to the last green packet baseline and restore prior differential/security evidence links before continuing.
