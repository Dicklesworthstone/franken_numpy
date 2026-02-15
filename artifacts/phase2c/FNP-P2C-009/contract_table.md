# FNP-P2C-009 Contract Table

packet_id: `FNP-P2C-009`  
subsystem: `NPY/NPZ IO contract`

## strict_mode_policy

Strict mode preserves legacy-observable `.npy`/`.npz` parse-write behavior classes for magic/version handling, header schema validation, dtype descriptor mapping, roundtrip semantics, pickle gating, memmap constraints, and archive dispatch.

## hardened_mode_policy

Hardened mode preserves the same public IO outcomes while adding bounded validation and deterministic audit semantics; unknown or incompatible semantics remain fail-closed.

## excluded_scope

- Full parity for every historical Python-2 warning string and deprecated compatibility message is deferred to packet-E/F closure.
- Exact warning/error text parity is deferred; class/family parity is required now.
- APIs outside packet `FNP-P2C-009` are excluded from this table.

## performance_sentinels

- header parsing overhead under adversarial large-header envelopes.
- non-contiguous `.npy` read/write throughput and reshape overhead.
- `.npz` lazy-member access and archive key lookup overhead.
- memmap open latency and mode transition cost under repeated workflows.

## Machine-Checkable Contract Rows

| contract_id | preconditions | shape_stride_contract | dtype_cast_contract | memory_alias_contract | strict_mode_policy | hardened_mode_policy | postconditions | error_contract | failure_reason_code | unit_property_ids | differential_ids | e2e_ids |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `P2C009-R01` | `.npy`/`.npz` ingest begins with magic/version bytes | magic/version parse occurs before header/data decode | dtype path not entered on invalid magic/version | no data-buffer mutation on magic failure path | malformed/unsupported magic rejects with stable class | same class; no permissive fallback for unknown version tuples | supported magic/version proceeds deterministically to header/archive dispatch | reject malformed magic prefix or unsupported version class | `io_magic_invalid` | `UP-009-magic-version` | `DF-009-magic-header-oracle` | `E2E-009-header-failure-replay` |
| `P2C009-R02` | `.npy` header bytes available after valid magic/version | header requires exact keys and valid `shape`/`fortran_order` classes | `descr` must decode to valid dtype descriptor class | invalid header path cannot mutate target buffers | malformed header schema rejects with stable class | same class with bounded diagnostics and audit linkage | accepted header yields deterministic `(shape, fortran_order, dtype)` triple | reject missing/extra keys, malformed shape, invalid descriptor, oversized header | `io_header_schema_invalid` | `UP-009-header-schema` | `DF-009-header-adversarial` | `E2E-009-header-failure-replay` |
| `P2C009-R03` | dtype descriptor serialization/deserialization requested | N/A (descriptor codec path) | descriptor encode/decode roundtrip is deterministic for supported classes | codec path cannot mutate array buffer alias state | supported descriptors roundtrip with stable class behavior | same behavior; unknown/incompatible descriptors fail closed | descriptor contract remains deterministic for fixed dtype input | reject invalid or unsupported descriptor classes | `io_dtype_descriptor_invalid` | `UP-009-descr-roundtrip` | `DF-009-descr-oracle` | `E2E-009-io-ingest-replay` |
| `P2C009-R04` | `.npy` write requested with ndarray-like payload | header shape/order metadata matches emitted data traversal order | write path preserves dtype class contract for supported dtypes | write path must not bypass object/policy gating | write success/failure classes match legacy for contiguous/non-contiguous payloads | same classes with bounded write guards | emitted payload is deterministically readable by same contract family | reject unsupported payload/policy combinations | `io_write_contract_violation` | `UP-009-write-contiguity` | `DF-009-npy-roundtrip-oracle` | `E2E-009-io-ufunc-reduce` |
| `P2C009-R05` | `.npy` read requested on valid header + payload stream | read count/reshape semantics must align with header `shape`/`fortran_order` | read dtype lane must match header descriptor class | read path cannot silently alias partial/truncated payload as valid | read success/failure classes match legacy for complete/truncated payloads | same classes with bounded read guards and deterministic reason codes | complete payload yields deterministic array class; truncated payload rejects | reject incomplete payload or shape/count mismatch | `io_read_payload_incomplete` | `UP-009-read-reshape-count` | `DF-009-read-truncation-oracle` | `E2E-009-truncated-input-replay` |
| `P2C009-R06` | object-array or pickle path is reached via load/save | header/shape legality unchanged by pickle policy gate | object-array/user-dtype pickle gate follows explicit `allow_pickle` contract | pickle-gated path cannot bypass policy through archive indirection | `allow_pickle=False` rejects object-array/pickle paths with stable class | same class; no silent policy widening in hardened mode | allowed paths preserve deterministic class behavior | reject disallowed pickle/object payload classes | `io_pickle_policy_violation` | `UP-009-pickle-policy` | `DF-009-pickle-policy-oracle` | `E2E-009-policy-replay` |
| `P2C009-R07` | memmap open/create path invoked | memmap shape/order semantics match header/create metadata | memmap dtype must satisfy non-object contract | memmap path enforces file-backed alias contract (no handle misuse) | invalid memmap mode/object-dtype classes reject deterministically | same class with bounded mode validation and fail-closed unknown metadata handling | valid memmap path yields deterministic view contract | reject object-dtype memmap and invalid mode/handle class combinations | `io_memmap_contract_violation` | `UP-009-memmap-safety` | `DF-009-memmap-oracle` | `E2E-009-memmap-replay` |
| `P2C009-R08` | `np.load` dispatch receives byte stream and policy flags | dispatch branch selection (`npz`/`npy`/pickle) is deterministic for fixed prefix/policy | selected branch preserves dtype/pickle contract class | dispatch must not bypass policy gates on branch transitions | dispatch class outcomes match legacy for supported input classes | same outcomes; unknown/unsupported class fails closed | deterministic dispatch branch + branch-specific result class | reject unsupported/trust-violating payload classes | `io_load_dispatch_invalid` | `UP-009-load-dispatch` | `DF-009-load-dispatch-oracle` | `E2E-009-io-ingest-replay` |
| `P2C009-R09` | `.npz` save/load path with positional/keyword arrays | archive member shape/order metadata remains deterministic per member | per-member dtype/write policy follows `.npy` contract | archive lazy-load and key lookup must not bypass policy/validation checks | key naming (`arr_N` defaults), duplicate-key rejection, and lazy-member behavior match legacy class | same behavior with bounded archive validation | archive roundtrip is deterministic for fixed member set/order | reject invalid key collisions/member contract violations | `io_npz_archive_contract_violation` | `UP-009-npz-archive-keys` | `DF-009-archive-roundtrip-oracle` | `E2E-009-npz-roundtrip-replay` |
| `P2C009-R10` | runtime policy mediation receives mode/class metadata for IO boundary | unknown header/metadata classes are non-admissible | unknown dtype/pickle policy classes are non-admissible | overrides cannot bypass incompatible-class policy gates | unknown/incompatible metadata fails closed | allowlisted compatible overrides may be audited in hardened mode only | decision action/reason-code remains deterministic and replay-complete | reject unknown metadata and incompatible policy classes | `io_policy_unknown_metadata` | `UP-009-policy-fail-closed` | `DF-009-policy-adversarial-oracle` | `E2E-009-policy-replay` |

## Logging and Failure Semantics

All packet IO validations must emit:

- `fixture_id`
- `seed`
- `mode`
- `env_fingerprint`
- `artifact_refs`
- `reason_code`

Reason-code vocabulary for this packet:

- `io_magic_invalid`
- `io_header_schema_invalid`
- `io_dtype_descriptor_invalid`
- `io_write_contract_violation`
- `io_read_payload_incomplete`
- `io_pickle_policy_violation`
- `io_memmap_contract_violation`
- `io_load_dispatch_invalid`
- `io_npz_archive_contract_violation`
- `io_policy_unknown_metadata`

## Budgeted Mode and Expected-Loss Notes

- Budgeted mode: hardened IO execution enforces explicit caps on header size parsing, archive member validation, and policy retries with deterministic exhaustion behavior.
- Expected-loss model: parser and dispatch policy decisions record state/action/loss rationale for replayable audits.
- Calibration trigger: if strict/hardened failure-class drift exceeds packet threshold, fallback to conservative deterministic behavior (`full_validate` or `fail_closed`) and emit audited reason codes.

## Graveyard and FrankenSuite Mapping

- `alien_cs_graveyard.md` §0.19 (evidence ledger schema)
- `alien_cs_graveyard.md` §0.4 (decision-theoretic runtime contracts)
- `alien_cs_graveyard.md` §6.12 (property-based testing with shrinking)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19

## Rollback Handle

If IO contract drift is detected, revert `artifacts/phase2c/FNP-P2C-009/contract_table.md` and restore the last green packet boundary contract baseline.
