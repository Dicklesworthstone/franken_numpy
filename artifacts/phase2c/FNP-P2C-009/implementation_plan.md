# FNP-P2C-009 Rust Implementation Plan

packet_id: `FNP-P2C-009`  
subsystem: `NPY/NPZ IO contract`

## 1. Crate and Module Boundary Skeleton

| Crate | Planned module boundary | Responsibility | Public surface contract |
|---|---|---|---|
| `crates/fnp-io` | `npy_magic_version` (packet-D planned boundary; crate currently stub) | encode/decode magic/version bytes and fail-closed version validation | magic/version parser contract (`P2C009-R01`) |
| `crates/fnp-io` | `npy_header_codec` (packet-D planned boundary) | header read/write schema validation (`shape`, `fortran_order`, `descr`) and bounded header-size checks | header parse/write contract (`P2C009-R02`) |
| `crates/fnp-io` | `dtype_descr_codec` (packet-D planned boundary) | deterministic descriptor encode/decode boundaries for supported dtype classes | descriptor roundtrip contract (`P2C009-R03`) |
| `crates/fnp-io` | `npy_reader` (packet-D planned boundary) | `.npy` ingest pipeline including payload-count checks and C/F reshape semantics | read contract (`P2C009-R05`) |
| `crates/fnp-io` | `npy_writer` (packet-D planned boundary) | `.npy` write pipeline including contiguous/non-contiguous handling and policy-gated object payloads | write contract (`P2C009-R04`) |
| `crates/fnp-io` | `npy_memmap_adapter` (packet-D planned boundary) | memmap open/create boundary with mode and object-dtype restrictions | memmap contract (`P2C009-R07`) |
| `crates/fnp-io` | `npz_archive_reader` (packet-D planned boundary) | lazy archive member loading and key-mapping semantics | archive read contract (`P2C009-R09`) |
| `crates/fnp-io` | `npz_archive_writer` (packet-D planned boundary) | archive write behavior for positional/keyword entries, compression mode, and collision checks | archive write contract (`P2C009-R09`) |
| `crates/fnp-io` | `io_load_dispatch` + `pickle_policy_gate` (packet-D planned boundary) | deterministic branch selection for `.npz`/`.npy`/pickle and explicit `allow_pickle` policy gating | dispatch/policy contracts (`P2C009-R06`, `P2C009-R08`) |
| `crates/fnp-runtime` | policy/audit decision context (existing) | strict/hardened fail-closed mediation with reason-code and evidence logging for IO packet decisions | `decide_and_record_with_context` integration from IO harnesses |
| `crates/fnp-conformance` | `io_packet_suite` (packet-F planned boundary) | fixture-driven differential/metamorphic/adversarial IO coverage (header, roundtrip, archive, policy) | packet-F IO runner + fixture manifests (planned) |
| `crates/fnp-conformance` | workflow scenario integration (existing + packet-G extension) | strict/hardened replay scenarios linking IO ingest/save flows to downstream workflows | packet-G IO scenario entries in workflow corpus (planned) |

## 2. Implementation Sequence (D-Stage to I-Stage)

1. Land packet-D `fnp-io` module skeletons (`npy_magic_version`, `npy_header_codec`, `dtype_descr_codec`, `npy_reader`, `npy_writer`, `npy_memmap_adapter`, `npz_archive_reader`, `npz_archive_writer`, `io_load_dispatch`, `pickle_policy_gate`) with explicit TODO gates for deferred parity debt.
2. Define packet reason-code taxonomy aligned with contract rows `P2C009-R01`..`R10`.
3. Implement deterministic magic/version + header schema boundaries with fail-closed behavior for unknown/incompatible metadata.
4. Implement descriptor codec and `.npy` read/write pathways with deterministic count/reshape and truncated-payload failure semantics.
5. Implement dispatch/pickle policy boundaries for `.npz`/`.npy`/pickle route selection and explicit `allow_pickle` gating.
6. Implement `.npz` archive boundaries (key naming, collision checks, lazy member read path, compression mode handling).
7. Implement memmap boundary with deterministic mode and dtype restrictions.
8. Add packet-F IO conformance harness placeholders and fixture schemas for malformed-header, roundtrip, archive, and policy lanes.
9. Add packet-G workflow scenario placeholders linking IO fixture IDs to replay/e2e scripts.
10. Wire packet IO policy decisions into runtime audit context (`fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`).
11. Gate packet-H optimization work behind baseline/profile/isomorphism evidence.
12. Close packet-I with parity summary + risk + durability sidecar/scrub/decode-proof artifacts.

## 3. Public Surface Contract Notes

- Packet-D additions remain clean-room and contract-driven (no compatibility shims).
- Unknown or incompatible IO semantics remain fail-closed in strict and hardened modes.
- Read/write/dispatch/archive outcomes must remain deterministic for fixed inputs.
- No `unsafe` pathways are introduced by the packet-D planning boundary.

## 4. Instrumentation Insertion Points

| Lane | Insertion point | Evidence artifact target |
|---|---|---|
| Unit/property | packet-E tests in `fnp-io` parser/reader/writer/archive/memmap modules | packet-E invariant logs + coverage artifacts |
| Differential/metamorphic/adversarial | packet-F IO runner + fixture manifests in `crates/fnp-conformance` | packet-F parity/differential reports |
| E2E/replay | packet-G workflow scenario corpus + e2e scripts in `scripts/e2e/` | replay logs under `artifacts/logs/` |
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

## 7. Compile-Safe Skeleton Validation

- Planning-stage validation rules:
  - no behavior-changing IO parser/writer migration is shipped in this bead;
  - packet contract and reason-code taxonomy remain internally consistent;
  - packet validator may remain `not_ready` until downstream E-I artifacts land.
- Validation command (offloaded via `rch`):  
  `rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-009`

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
- If comparator fails, restore prior planning baseline and re-run packet validation before continuing.
