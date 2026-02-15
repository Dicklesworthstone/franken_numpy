# FNP-P2C-007 Behavior Extraction Ledger

Packet: `FNP-P2C-007-A`  
Subsystem: `RNG core and constructor contract`

## 1. Observable Contract Ledger

| Contract ID | Observable behavior | Strict mode expectation | Hardened mode expectation | Legacy anchors |
|---|---|---|---|---|
| `P2C007-C01` | `default_rng` normalizes supported seed input classes into deterministic generator construction paths. | deterministic constructor acceptance/rejection for supported seed classes | same outward constructor classes; unknown/incompatible seed metadata fail-closed | `_generator.pyx:4991`, `_generator.pyx:142` |
| `P2C007-C02` | `Generator(bit_generator)` preserves generator/bit-generator identity contracts and serialization hooks. | deterministic bit-generator binding and constructor behavior | same behavior with bounded diagnostics for malformed state payloads | `_generator.pyx:196`, `_generator.pyx:217`, `_generator.pyx:224`, `_generator.pyx:230` |
| `P2C007-C03` | `SeedSequence.generate_state` is deterministic for fixed entropy/spawn-key/pool settings. | identical state words for identical seed-sequence inputs | same output class; malformed seed metadata fail-closed | `bit_generator.pyx:254`, `bit_generator.pyx:407`, `test_seed_sequence.py:6` |
| `P2C007-C04` | `SeedSequence.spawn` deterministically derives child sequences and advances spawn counters predictably. | child spawn-key lineage and counter progression are deterministic | same lineage behavior with bounded audit metadata | `bit_generator.pyx:455`, `bit_generator.pyx:485`, `test_seed_sequence.py:56` |
| `P2C007-C05` | BitGenerator constructors (`MT19937`, `PCG64`, `Philox`, `SFC64`) consume seed inputs via deterministic initialization pathways. | constructor outcomes are stable for fixed seed inputs | same outcomes; malformed seed/state payloads fail-closed | `_mt19937.pyx:130`, `_pcg64.pyx:122`, `_philox.pyx:166`, `_sfc64.pyx:89` |
| `P2C007-C06` | `jumped` APIs yield deterministic partitioned streams/states for fixed input state and jump count. | jumped state parity for each supported algorithm class | same jumped state class with deterministic rejection for malformed jumps | `_mt19937.pyx:213`, `_pcg64.pyx:160`, `_philox.pyx:264`, `test_generator_mt19937.py:2651` |
| `P2C007-C07` | BitGenerator `state` getter/setter roundtrips and invalid-state rejection classes are stable. | valid state roundtrip succeeds; invalid state class rejected predictably | same classes with bounded audit context and deterministic reason codes | `bit_generator.pyx:569`, `bit_generator.pyx:584`, `test_random.py:159` |
| `P2C007-C08` | Pickle/restore flow preserves seed sequence and generator state contracts. | restored generators retain expected seed/state fields | same state-preservation class with fail-closed malformed payload handling | `test_generator_mt19937.py:2790` |

## 2. Compatibility Invariants

1. Constructor determinism invariant: identical constructor inputs must produce equivalent generator class/state outcomes.
2. Seed-sequence determinism invariant: fixed entropy/spawn-key inputs produce identical generated state words.
3. Spawn lineage invariant: parent->child spawn key/counter derivation is deterministic and monotonic.
4. Jump partition invariant: jumped streams/states are deterministic for fixed source state and jump count.
5. State roundtrip invariant: valid state getter/setter roundtrip preserves generator contract; invalid states fail with stable class.
6. Strict/hardened parity invariant: hardened mode may add bounded validation/audit metadata but must preserve public success/failure class for covered contracts.

## 3. Undefined or Under-Specified Edges (Tagged)

| Unknown ID | Description | Risk | Owner bead | Closure criteria |
|---|---|---|---|---|
| `P2C007-U01` | `fnp-random` is currently a stub and has no generator/seed-sequence/bit-generator boundary implementation. | high | `bd-23m.18.4` | land module skeleton with constructor/state/spawn/jump interfaces and baseline tests |
| `P2C007-U02` | Deterministic-seed witness suites are not yet present in packet-scoped unit/property tests. | high | `bd-23m.18.5` | packet-E suites cover constructor/state/spawn/jump invariants with structured logs |
| `P2C007-U03` | Differential oracle corpus for RNG constructor/state behavior is not packet-scoped yet. | high | `bd-23m.18.6` | packet-F fixtures and differential runner additions cover RNG contract rows |
| `P2C007-U04` | Replay/e2e scenarios do not yet include packet-specific RNG state/seed lineage traces. | medium | `bd-23m.18.7` | packet-G scenario logs include seed lineage + artifact refs + reason codes |
| `P2C007-U05` | Constructor normalization across all supported seed input classes (including legacy adapters) remains unpinned in Rust tests. | medium | `bd-23m.18.2` + `bd-23m.18.5` | contract rows and packet-E tests lock accepted/rejected class behavior |
| `P2C007-U06` | Exact state schema compatibility for algorithm-specific state payloads is not yet represented in Rust models. | medium | `bd-23m.18.4` + `bd-23m.18.6` | algorithm-specific state schema contracts and differential fixtures are codified |

## 4. Planned Verification Hooks

| Verification lane | Planned hook | Artifact target |
|---|---|---|
| Unit/property | deterministic seed/state/spawn/jump invariants for RNG constructors and bit-generator state model | packet-E tests in `fnp-random` with structured logging |
| Differential/metamorphic/adversarial | RNG constructor/state/spawn/jump fixture corpus compared against legacy oracle captures | packet-F fixtures + conformance runner extensions |
| E2E | strict/hardened RNG workflow scenarios with seed lineage and replay metadata | packet-G scenario scripts + `artifacts/logs/` |
| Structured logging | enforce required fields (`fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`) for RNG evidence | `artifacts/contracts/TESTING_AND_LOGGING_CONVENTIONS_V1.md` + packet-E/F/G outputs |

## 5. Method-Stack Artifacts and EV Gate

- Alien decision contract: RNG policy mediation must record state/action/loss rationale with deterministic fallback behavior.
- Optimization gate: RNG performance changes require baseline/profile + one-lever + behavior-isomorphism proof.
- EV gate: policy/optimization levers ship only when `EV >= 2.0`; otherwise remain explicit deferred parity debt.
- RaptorQ scope: packet-I RNG evidence bundle requires sidecar/scrub/decode-proof linkage for durable artifacts.

## 6. Rollback Handle

If packet-local RNG behavior extraction drifts or is contradicted, revert `artifacts/phase2c/FNP-P2C-007/*` to the prior green baseline and re-run packet validation before reapplying changes.
