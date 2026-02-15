# FNP-P2C-005 Rust Implementation Plan

packet_id: `FNP-P2C-005`  
subsystem: `ufunc dispatch + gufunc signature`

## 1. Crate and Module Boundary Skeleton

| Crate | Planned module boundary | Responsibility | Public surface contract |
|---|---|---|---|
| `crates/fnp-ufunc` | `signature_parser` (packet-D planned boundary) | parse/normalize gufunc signatures, enforce `sig`/`signature` conflict rules, and produce deterministic signature IR | signature parse/normalize entrypoints used by ufunc call setup (planned) |
| `crates/fnp-ufunc` | `argument_normalizer` (packet-D planned boundary) | canonicalize positional/keyword ufunc call state before dispatch | normalized call context for dispatch planner (planned) |
| `crates/fnp-ufunc` | `dispatch_planner` (packet-D planned boundary) | deterministic method/loop selection using signature + dtype constraints | deterministic dispatch selection contract (`P2C005-R05`) |
| `crates/fnp-ufunc` | `override_bridge` (packet-D planned boundary) | evaluate `__array_ufunc__` precedence ordering and payload class validity | override ordering/validation boundary (`P2C005-R04`) |
| `crates/fnp-ufunc` | `reduction_adapter` (packet-D planned boundary) | reduction wrapper integration for `axis`/`keepdims`/`where` semantics | reduction contract surface (`P2C005-R08`) |
| `crates/fnp-ufunc` | `loop_registry` (packet-D/E planned boundary) | custom loop registration compatibility envelope and fail-closed unknown semantics | loop-registration contract surface (`P2C005-R09`) |
| `crates/fnp-dtype` + `crates/fnp-ufunc` | `type_resolution_bridge` (packet-D planned boundary) | deterministic promotion/default-type-resolution bridge for dispatch planner | type-resolution contract (`P2C005-R06`) |
| `crates/fnp-runtime` | policy/audit decision context (existing) | strict/hardened fail-closed mediation with reason-code and evidence logging | `decide_and_record_with_context` integration from ufunc packet suites |
| `crates/fnp-conformance` | `ufunc_packet_suite` (packet-F planned boundary) | fixture-driven differential/metamorphic/adversarial coverage for signature/override/dispatch/reduction contracts | packet-F runner + fixture manifests (planned) |
| `crates/fnp-conformance` | workflow scenario integration (existing + packet-G extension) | strict/hardened replay scenarios for signature/dispatch/reduction journeys | packet-G scenario entries in workflow corpus (planned) |

## 2. Implementation Sequence (D-Stage to I-Stage)

1. Land packet-D module skeletons in `fnp-ufunc` (`signature_parser`, `argument_normalizer`, `dispatch_planner`, `override_bridge`, `reduction_adapter`, `loop_registry`) with explicit TODO gates for deferred parity debt.
2. Define packet reason-code taxonomy aligned with contract rows `P2C005-R01`..`R10`.
3. Implement deterministic signature conflict and grammar normalization boundary (`sig`/`signature`/fixed signature forms).
4. Implement override precedence boundary with runtime-policy mediation hooks for strict/hardened behavior classes.
5. Implement deterministic dispatch planner + type-resolution bridge with `fnp-dtype` promotion contracts.
6. Add reduction wrapper contract seam for `axis`/`keepdims`/`where` legality and deterministic failure taxonomy.
7. Add loop-registry boundary stubs with explicit fail-closed handling for unsupported registration semantics.
8. Add packet-F conformance harness placeholders and fixture schemas for signature/override/dispatch/reduction differential lanes.
9. Add packet-G workflow scenario placeholders linking fixture IDs to replay/e2e script paths.
10. Wire packet policy decisions into runtime audit context fields (`fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`).
11. Gate packet-H optimization work behind baseline/profile/isomorphism evidence.
12. Close packet-I with parity summary + risk + durability sidecar/scrub/decode-proof artifacts.

## 3. Public Surface Contract Notes

- Packet-D additions remain clean-room and contract-driven (no compatibility shims).
- Unknown or incompatible ufunc/gufunc semantics remain fail-closed in strict and hardened modes.
- Dispatch/type-resolution/reduction outcomes must remain deterministic for fixed inputs.
- No `unsafe` pathways are introduced by the packet-D planning boundary.

## 4. Instrumentation Insertion Points

| Lane | Insertion point | Evidence artifact target |
|---|---|---|
| Unit/property | packet-E tests in `fnp-ufunc` signature/override/dispatch/reduction modules | packet-E invariant logs + coverage artifacts |
| Differential/metamorphic/adversarial | packet-F runner + fixture manifests in `crates/fnp-conformance` | packet-F parity/differential reports |
| E2E/replay | packet-G workflow scenario corpus + e2e scripts in `scripts/e2e/` | replay logs under `artifacts/logs/` |
| Runtime policy audit | ufunc packet suites using runtime decision/audit integration | security gate + policy evidence ledger outputs |

## 5. Structured Logging Emission Points

- signature keyword conflict and grammar normalization outcomes,
- fixed-signature resolution and canonicalization decisions,
- override precedence path selection and payload class validation,
- dispatch planner candidate resolution and type-resolution decisions,
- reduction wrapper validation and failure taxonomy branches,
- loop-registry acceptance/rejection paths,
- runtime policy mediation events for strict/hardened packet boundaries.

All emissions must include:

- `fixture_id`
- `seed`
- `mode`
- `env_fingerprint`
- `artifact_refs`
- `reason_code`

## 6. Artifact Boundary Plan

- Packet-local artifacts: `artifacts/phase2c/FNP-P2C-005/`
- Contract schemas and policy maps: `artifacts/contracts/`
- Replay/security logs: `artifacts/logs/`
- Durability artifacts: `artifacts/raptorq/` + packet-I packet-local outputs

## 7. Compile-Safe Skeleton Validation

- Planning-stage validation rules:
  - no behavior-changing dispatch migration is shipped in this bead;
  - packet contract and reason-code taxonomy remain internally consistent;
  - packet validator may remain `not_ready` until downstream E-I artifacts land.
- Validation command (offloaded via `rch`):  
  `rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-005`

## 8. Alien Recommendation Contract Guardrails

- Graveyard mapping required for packet decisions: `alien_cs_graveyard.md` §0.4, §0.19, §6.12.
- FrankenSuite summary mapping required: `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19.
- EV gate for non-doc policy/optimization levers: ship only when `EV >= 2.0`; otherwise track explicit deferred debt.
- Hotspot evidence prerequisite for policy/optimization shifts: baseline/profile artifact pair (or documented profiler-unavailable fallback).
- Isomorphism proof requirement for behavior-affecting changes: ordering/tie-break note + golden checks + reproducible benchmark delta.

## 9. Rollback Handle

- Rollback command path:  
  `git restore --source <last-green-commit> -- artifacts/phase2c/FNP-P2C-005/implementation_plan.md`
- Baseline comparator to beat/restore:
  - last green `rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-005` report,
  - plus last green packet-linked security/workflow gate artifacts.
- If comparator fails, restore prior planning baseline and re-run packet validation before continuing.
