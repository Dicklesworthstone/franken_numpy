# FNP-P2C-009 Rust Implementation Plan

packet_id: `FNP-P2C-009`  
subsystem: `NPY/NPZ IO contract`

## 1. Crate and Module Boundary State

| Crate | Landed module boundary | Responsibility | Public surface contract |
|---|---|---|---|
| `crates/fnp-io` | `NpyHeader`, header/version helpers, descriptor validation | encode/decode magic/version bytes; validate header schema (`shape`, `fortran_order`, `descr`) and bounded header sizes | magic/header/descriptor contracts (`P2C009-R01`, `P2C009-R02`, `P2C009-R03`) |
| `crates/fnp-io` | `read_npy_bytes`, `write_npy_bytes`, `write_npy_bytes_with_version`, `save`, `load` | `.npy` read/write pipeline including payload-count checks, header version selection, and C/F layout metadata | read/write contracts (`P2C009-R04`, `P2C009-R05`) |
| `crates/fnp-io` | `NpzEntry`, `NpzCompression`, `read_npz_bytes`, `write_npz_bytes`, `savez`, `savez_compressed`, `load_npz` | archive member loading, key mapping, compression mode handling, and duplicate/collision validation | archive read/write contract (`P2C009-R09`) |
| `crates/fnp-io` | `classify_load_dispatch`, `load_auto`, `enforce_pickle_policy` | deterministic branch selection for `.npz`/`.npy`/pickle and explicit `allow_pickle` policy gating | dispatch/policy contracts (`P2C009-R06`, `P2C009-R08`) |
| `crates/fnp-io` | `loadtxt`, `loadtxt_usecols`, `loadtxt_unpack`, `genfromtxt`, `genfromtxt_full` | text IO parsing, delimiter/comment handling, use-column selection, missing-value handling, and unpack semantics | text IO contract (`P2C009-R10`) |
| `crates/fnp-io` | `MemmapArray`, `memmap_npy`, `open_memmap` | memmap open/create boundary with mode, dtype, and object-payload restrictions | memmap contract (`P2C009-R07`) |
| `crates/fnp-runtime` | policy/audit decision context (existing) | strict/hardened fail-closed mediation with reason-code and evidence logging for IO packet decisions | `decide_and_record_with_context` integration from IO harnesses |
| `crates/fnp-conformance` | `io_diagnostics`, packet fixture artifacts, `npy_npz_diagnostic` tests | fixture-driven differential/metamorphic/adversarial IO coverage (header, roundtrip, archive, policy) | packet-F IO diagnostics and parity report artifacts |
| `crates/fnp-conformance` | workflow scenario corpus entries `io_packet_replay` and `io_packet_hostile_guardrails` | strict/hardened replay scenarios linking IO ingest/save flows to downstream workflows | packet-G IO scenario evidence |

## 2. Implementation Sequence (Current D-I Baseline to Breadth Expansion)

1. Preserve the landed packet-D `fnp-io` boundaries for header/version validation, descriptor handling, `.npy` roundtrip, `.npz` archive handling, load dispatch, pickle policy, text IO, and memmap entrypoints.
2. Keep packet reason-code taxonomy aligned with contract rows `P2C009-R01`..`R10`.
3. Maintain deterministic magic/version + header schema boundaries with fail-closed behavior for unknown/incompatible metadata.
4. Maintain descriptor codec and `.npy` read/write pathways with deterministic count/reshape and truncated-payload failure semantics.
5. Maintain dispatch/pickle policy boundaries for `.npz`/`.npy`/pickle route selection and explicit `allow_pickle` gating.
6. Maintain `.npz` archive boundaries for key naming, collision checks, member read paths, and compression mode handling.
7. Maintain memmap boundaries with deterministic mode and dtype restrictions.
8. Expand packet-F IO conformance breadth from the landed diagnostic fixtures toward broader malformed-header, roundtrip, archive, text, and policy lanes.
9. Extend packet-G workflow scenarios beyond `io_packet_replay` and `io_packet_hostile_guardrails` when new IO fixture IDs land.
10. Preserve packet IO policy decisions in runtime audit context (`fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`).
11. Gate packet-H optimization work behind baseline/profile/isomorphism evidence.
12. Preserve packet-I readiness with parity summary, risk notes, durability sidecars, scrub artifacts, and decode-proof artifacts.

## 3. Public Surface Contract Notes

- Landed packet-D boundaries remain clean-room and contract-driven (no compatibility shims).
- Unknown or incompatible IO semantics remain fail-closed in strict and hardened modes.
- Read/write/dispatch/archive outcomes must remain deterministic for fixed inputs.
- No `unsafe` pathways are introduced by the packet-D IO boundary.

## 4. Instrumentation Insertion Points

| Lane | Insertion point | Evidence artifact target |
|---|---|---|
| Unit/property | `crates/fnp-io/src/lib.rs`, `crates/fnp-io/tests/metamorphic_io.rs`, `crates/fnp-io/tests/npy_npz_diagnostic.rs` | packet-E invariant logs + coverage artifacts |
| Differential/metamorphic/adversarial | `crates/fnp-io/tests/npy_numpy_conformance.rs`, `crates/fnp-conformance/src/io_diagnostics.rs`, packet fixtures | packet-F parity/differential reports |
| E2E/replay | workflow scenarios `io_packet_replay` and `io_packet_hostile_guardrails` plus e2e scripts in `scripts/e2e/` | replay logs under `artifacts/logs/` |
| Runtime policy audit | IO packet suites using runtime decision/audit integration | security gate + policy evidence ledger outputs |

## 5. Structured Logging Emission Points

- magic/version parse and branch-dispatch outcomes,
- header schema validation and descriptor decode outcomes,
- read/write payload count and reshape validation outcomes,
- pickle policy gate decisions (`allow_pickle` pathways),
- archive key/collision/lazy-member validation events,
- memmap mode/dtype/handle validation branches,
- runtime policy mediation events for strict/hardened packet boundaries.

All emissions must include:

- `fixture_id`
- `seed`
- `mode`
- `env_fingerprint`
- `artifact_refs`
- `reason_code`

## 6. Artifact Boundary Plan

- Packet-local artifacts: `artifacts/phase2c/FNP-P2C-009/`
- Contract schemas and policy maps: `artifacts/contracts/`
- Replay/security logs: `artifacts/logs/`
- Durability artifacts: `artifacts/raptorq/` + packet-I packet-local outputs

## 7. Evidence Refresh Validation

- Evidence-refresh validation rules:
  - no behavior-changing IO parser/writer migration is shipped in this bead;
  - packet boundaries describe the current landed IO implementation;
  - packet contract and reason-code taxonomy remain internally consistent;
  - packet validator reports `ready` with no missing artifacts, fields, or parse errors.
- Validation command (offloaded via `rch`):  
  `rch exec -- env CARGO_TARGET_DIR=/data/projects/.cargo-target-franken_numpy-NavyPine-ns5i7 CARGO_INCREMENTAL=0 cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-009`

## 8. Alien Recommendation Contract Guardrails

- Graveyard mapping required for packet decisions: `alien_cs_graveyard.md` §0.4, §0.19, §6.12.
- FrankenSuite summary mapping required: `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19.
- EV gate for non-doc policy/optimization levers: ship only when `EV >= 2.0`; otherwise track explicit deferred debt.
- Hotspot evidence prerequisite for policy/optimization shifts: baseline/profile artifact pair (or documented profiler-unavailable fallback).
- Isomorphism proof requirement for behavior-affecting changes: ordering/tie-break note + golden checks + reproducible benchmark delta.

## 9. Rollback Handle

- Rollback command path:  
  `git restore --source <last-green-commit> -- artifacts/phase2c/FNP-P2C-009/implementation_plan.md`
- Baseline comparator to beat/restore:
  - last green `rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-009` report,
  - plus last green packet-linked security/workflow gate artifacts.
- If comparator fails, restore prior evidence-refresh baseline and re-run packet validation before continuing.
