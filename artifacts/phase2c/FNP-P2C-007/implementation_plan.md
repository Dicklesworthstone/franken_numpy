# FNP-P2C-007 Rust Implementation Plan

packet_id: `FNP-P2C-007`  
subsystem: `RNG core and constructor contract`

## 1. Crate and Module Boundary Skeleton

| Crate | Planned module boundary | Responsibility | Public surface contract |
|---|---|---|---|
| `crates/fnp-random` | RNG constructor and generator facade surface (implemented crate boundary) | normalize supported seed input classes and route to deterministic generator initialization paths | generator construction/state APIs with NumPy-aligned reason codes |
| `crates/fnp-random` | SeedSequence core (implemented crate boundary) | entropy/spawn-key state model and deterministic `generate_state`/`spawn` behavior | seed-sequence generation + lineage interfaces |
| `crates/fnp-random` | Bit-generator state core (implemented crate boundary) | bit-generator lifecycle, state schema contracts, spawn/jump hooks | bit-generator state/get-set/spawn/jump interfaces |
| `crates/fnp-random` | PCG64/PCG64DXSM and related adapter lanes (implemented/covered boundary) | scoped adapters for supported constructor, stream, jump, and state classes | deterministic algorithm behavior with unsupported classes kept fail-closed |
| `crates/fnp-conformance` | RNG packet suite (implemented packet-F/I evidence) | fixture-driven differential/metamorphic/adversarial RNG contract checks | RNG runner, fixture manifests, and packet readiness artifacts |
| `crates/fnp-conformance` | workflow scenario integration (implemented packet-G/H evidence) | replay scenarios for constructor/state/spawn/jump workflows | packet-G RNG scenario entries and packet-H optimization replay artifacts |
| `crates/fnp-runtime` | policy/audit decision context (existing) | strict/hardened fail-closed mediation with seed-aware reason-code logging | `decide_and_record_with_context` integration for RNG harness paths |

## 2. Implementation Sequence (D-Stage to I-Stage)

1. Land packet-D module boundary in `fnp-random` for constructor, seed-sequence, bit-generator state, and generator facade behavior.
2. Define constructor/state/spawn/jump reason-code taxonomy aligned to contract rows `P2C007-R01`..`R10`.
3. Encode deterministic seed-sequence lineage model and algorithm-neutral state schema boundaries.
4. Keep unsupported algorithm classes fail-closed while supported PCG64/PCG64DXSM lanes carry deterministic state and replay evidence.
5. Add packet-F RNG fixture schema and conformance runner coverage.
6. Add packet-G workflow scenario artifacts linking RNG fixture IDs to replay/e2e script paths.
7. Wire RNG policy decisions into runtime audit context fields (`fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`).
8. Gate packet-H optimization work behind baseline/profile/isomorphism evidence.
9. Close packet-I with parity summary + risk + durability sidecar/scrub/decode-proof artifacts.

## 3. Public Surface Contract Notes

- Packet-D additions must remain clean-room and contract-driven (no compatibility shims).
- Unknown or incompatible RNG semantics remain fail-closed in strict and hardened modes.
- Constructor/state/spawn/jump behavior classes must be deterministic for fixed inputs.
- No `unsafe` pathways are introduced by this packet-D planning boundary.

## 4. Instrumentation Insertion Points

| Lane | Insertion point | Evidence artifact target |
|---|---|---|
| Unit/property | packet-E tests in `fnp-random` constructor/seed/state/spawn/jump modules | packet-E RNG invariant logs + coverage artifacts |
| Differential/metamorphic/adversarial | packet-F RNG runner + fixture manifests in `crates/fnp-conformance` | packet-F parity/differential reports |
| E2E/replay | packet-G workflow scenario corpus + e2e scripts in `scripts/e2e/` | replay logs under `artifacts/logs/` |
| Runtime policy audit | RNG suites using runtime decision/audit integration | security gate + policy evidence ledger outputs |

## 5. Structured Logging Emission Points

- constructor seed normalization outcomes,
- SeedSequence `generate_state` and `spawn` lineage decisions,
- bit-generator state set/get validation and jump outcomes,
- generator serialization/restore lifecycle checks,
- runtime policy mediation events for strict/hardened RNG boundaries.

All emissions must include:

- `fixture_id`
- `seed`
- `mode`
- `env_fingerprint`
- `artifact_refs`
- `reason_code`

## 6. Artifact Boundary Plan

- Packet-local artifacts: `artifacts/phase2c/FNP-P2C-007/`
- Contract schemas and policy maps: `artifacts/contracts/`
- Replay/security logs: `artifacts/logs/`
- Durability artifacts: `artifacts/raptorq/` + packet-I packet-local outputs

## 7. Compile-Safe Skeleton Validation

- Validation rules:
  - doc-only refreshes must not change RNG behavior;
  - packet contract and reason-code taxonomy remain internally consistent;
  - packet validator continues reporting complete artifact fields for `FNP-P2C-007`.
- Validation command (offloaded via `rch`):  
  `rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-007`

## 8. Alien Recommendation Contract Guardrails

- Graveyard mapping required for packet decisions: `alien_cs_graveyard.md` §0.4, §0.19, §6.12.
- FrankenSuite summary mapping required: `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19.
- EV gate for non-doc policy/optimization levers: ship only when `EV >= 2.0`; otherwise track as explicit deferred debt.
- Hotspot evidence prerequisite for policy/optimization shifts: baseline/profile artifact pair (or documented profiler-unavailable fallback).
- Isomorphism proof requirement for behavior-affecting changes: ordering/tie-break note + golden checks + reproducible benchmark delta.

## 9. Rollback Handle

- Rollback command path:  
  `git restore --source <last-green-commit> -- artifacts/phase2c/FNP-P2C-007/implementation_plan.md`
- Baseline comparator to beat/restore:
  - last green `rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-007` report,
  - plus last green packet-linked security/workflow gate artifacts.
- If comparator fails, restore prior planning baseline and re-run packet validation before continuing.
