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

## 3. Undefined or Under-Specified Edges (Tagged)

| Unknown ID | Description | Risk | Owner bead | Closure criteria |
|---|---|---|---|---|
| `P2C009-U01` | Exact warning-message parity for Python-2 header fallback/filtering paths may vary by legacy revision. | medium | `bd-23m.20.6` | differential harness verifies failure/warning class containment instead of full-string equality. |
| `P2C009-U02` | Full cross-platform file-like/mmap edge behavior (pathlib/duck-typed handles/OS-specific modes) is not yet scoped in Rust boundaries. | high | `bd-23m.20.4` | packet-D boundary explicitly models supported handle classes and fail-closed behavior for unsupported cases. |
| `P2C009-U03` | Complete `.npz` duplicate-name and archive-member ordering corner cases are not yet represented in packet fixtures. | high | `bd-23m.20.5` | packet-E unit/property corpus adds ordering/duplicate-key matrix with deterministic replay fields. |
| `P2C009-U04` | Parser acceptance boundary for hostile large headers and malformed descriptor payloads needs adversarial fixture saturation. | high | `bd-23m.20.3` | threat model + packet-F adversarial fixtures demonstrate fail-closed coverage for hostile metadata envelopes. |

## 4. Planned Verification Hooks

| Verification lane | Planned hook | Artifact target |
|---|---|---|
| Unit/property | header parse corpus + roundtrip laws + pickle policy gates + archive key contracts | `crates/fnp-conformance/fixtures/io_property_cases.json` (planned), structured JSONL logs |
| Differential/metamorphic/adversarial | extend differential harness with malformed header, roundtrip, and archive adversarial fixtures | `crates/fnp-conformance/src/io_differential.rs` (planned), `crates/fnp-conformance/fixtures/io_adversarial_cases.json` (planned) |
| E2E | IO workflow scenarios chaining load/save/savez into downstream compute/replay paths | `scripts/e2e/run_io_contract_journey.sh` (planned) |
| Structured logging | enforce mandatory logging fields for all packet IO tests and gates | `artifacts/contracts/test_logging_contract_v1.json`, `scripts/e2e/run_test_contract_gate.sh` |

## 5. Method-Stack Artifacts and EV Gate

- Alien decision contract: parser and policy mediation decisions must log state, action, and expected-loss rationale where compatibility/security mediation occurs.
- Optimization gate: no IO parsing/perf optimization is accepted without baseline/profile + single-lever + isomorphism proof artifact.
- EV gate: IO optimization levers are promoted only when `EV >= 2.0`; otherwise tracked as deferred research debt.
- RaptorQ scope: packet `FNP-P2C-009` durable evidence bundle must include sidecar/scrub/decode-proof links at packet-I closure.

## 6. Rollback Handle

If packet-local IO extraction drifts from contract intent, roll back `artifacts/phase2c/FNP-P2C-009/*` to the last green packet baseline and restore prior differential/security evidence links before continuing.
