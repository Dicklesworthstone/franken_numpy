# FNP-P2C-007 Legacy Anchor Map

Packet: `FNP-P2C-007`  
Subsystem: `RNG core and constructor contract`

## Scope

This map captures concrete legacy NumPy RNG anchors for constructor normalization, seed sequence semantics, bit-generator lifecycle/state, spawn/jump behavior, and reproducibility guarantees. It binds those anchors to current and planned Rust boundaries for packet `FNP-P2C-007`.

## packet_id

`FNP-P2C-007`

## legacy_paths

- `legacy_numpy_code/numpy/numpy/random/_generator.pyx`
- `legacy_numpy_code/numpy/numpy/random/bit_generator.pyx`
- `legacy_numpy_code/numpy/numpy/random/_mt19937.pyx`
- `legacy_numpy_code/numpy/numpy/random/_pcg64.pyx`
- `legacy_numpy_code/numpy/numpy/random/_philox.pyx`
- `legacy_numpy_code/numpy/numpy/random/_sfc64.pyx`
- `legacy_numpy_code/numpy/numpy/random/tests/test_generator_mt19937.py`
- `legacy_numpy_code/numpy/numpy/random/tests/test_seed_sequence.py`
- `legacy_numpy_code/numpy/numpy/random/tests/test_random.py`
- `legacy_numpy_code/numpy/numpy/random/tests/test_generator_mt19937_regressions.py`

## legacy_symbols

- `cdef class Generator`
- `default_rng`
- `cdef class SeedSequence`
- `SeedSequence.generate_state`
- `SeedSequence.spawn`
- `cdef class BitGenerator`
- `BitGenerator.spawn`
- `MT19937.jumped`
- `PCG64.jumped`
- `Philox.jumped`

## Legacy-to-Rust Anchor Table

| Legacy path | Symbol/anchor | Role in observable behavior | Planned Rust boundary |
|---|---|---|---|
| `legacy_numpy_code/numpy/numpy/random/_generator.pyx:142` | `cdef class Generator` | high-level RNG facade lifecycle and method dispatch over bit generators | `crates/fnp-random` generator facade boundary (`bd-23m.18.4`) |
| `legacy_numpy_code/numpy/numpy/random/_generator.pyx:241` | `Generator.spawn` | child-generator derivation semantics from underlying bit generator/seed sequence | `fnp-random` spawn contract + deterministic child stream mapping |
| `legacy_numpy_code/numpy/numpy/random/_generator.pyx:4991` | `default_rng(seed=None)` | constructor normalization of seed input classes (`None`, int, array, SeedSequence, BitGenerator, Generator, RandomState) | packet-D constructor normalization API in `fnp-random` |
| `legacy_numpy_code/numpy/numpy/random/bit_generator.pyx:254` | `cdef class SeedSequence` | canonical entropy/spawn-key state model for reproducible stream initialization | `fnp-random` seed-sequence core model (`bd-23m.18.4`) |
| `legacy_numpy_code/numpy/numpy/random/bit_generator.pyx:407` | `SeedSequence.generate_state` | deterministic generation of uint32/uint64 initialization state words | packet-E/F deterministic-seed witness and differential hooks |
| `legacy_numpy_code/numpy/numpy/random/bit_generator.pyx:455` | `SeedSequence.spawn` | deterministic child-sequence derivation by extending spawn keys and counters | packet-D spawn interface + packet-G replay linkage |
| `legacy_numpy_code/numpy/numpy/random/bit_generator.pyx:499` | `cdef class BitGenerator` | base bit-generator state lifecycle and seed-sequence integration | `fnp-random` bit-generator trait/object boundary |
| `legacy_numpy_code/numpy/numpy/random/bit_generator.pyx:603` | `BitGenerator.spawn` | bit-generator child derivation contract and seed-sequence dependency | packet-D bit-generator spawn surface |
| `legacy_numpy_code/numpy/numpy/random/_mt19937.pyx:130` | `MT19937.__init__` | MT19937 constructor seeding and state initialization contract | packet-D algorithm adapter (MT19937 lane) |
| `legacy_numpy_code/numpy/numpy/random/_mt19937.pyx:213` | `MT19937.jumped` | deterministic jump-ahead stream partitioning semantics | packet-E/F jump witness contract |
| `legacy_numpy_code/numpy/numpy/random/_pcg64.pyx:122` | `PCG64.__init__` | PCG64 constructor seed normalization/state setup | packet-D algorithm adapter (PCG lane) |
| `legacy_numpy_code/numpy/numpy/random/_pcg64.pyx:160` | `PCG64.jumped` | deterministic jumped stream partitioning semantics | packet-E/F jump witness contract |
| `legacy_numpy_code/numpy/numpy/random/_philox.pyx:166` | `Philox.__init__` | Philox constructor/key/counter initialization semantics | packet-D algorithm adapter (Philox lane) |
| `legacy_numpy_code/numpy/numpy/random/_philox.pyx:264` | `Philox.jumped` | deterministic counter-based jump behavior | packet-E/F jump witness contract |

## Oracle Test Anchors

| Test path | Anchor | Behavior family |
|---|---|---|
| `legacy_numpy_code/numpy/numpy/random/tests/test_seed_sequence.py:6` | `test_reference_data` | SeedSequence reference-vector determinism (`generate_state`) |
| `legacy_numpy_code/numpy/numpy/random/tests/test_seed_sequence.py:56` | `test_zero_padding` | entropy/spawn-key zero-padding and compatibility invariants |
| `legacy_numpy_code/numpy/numpy/random/tests/test_generator_mt19937.py:69` | `test_seedsequence` | MT19937 constructor acceptance of SeedSequence |
| `legacy_numpy_code/numpy/numpy/random/tests/test_generator_mt19937.py:2651` | `test_jumped` | jumped-state deterministic witness |
| `legacy_numpy_code/numpy/numpy/random/tests/test_generator_mt19937.py:2790` | `test_pickle_preserves_seed_sequence` | pickled generator preserves seed-sequence state |
| `legacy_numpy_code/numpy/numpy/random/tests/test_generator_mt19937_regressions.py:58` | `test_call_within_randomstate` | regression guard for generator invocation behavior |
| `legacy_numpy_code/numpy/numpy/random/tests/test_random.py:159` | `test_set_invalid_state` | state setter rejection/error-class behavior |

## Current Rust Anchor Evidence

| Rust path | Anchor | Coverage note |
|---|---|---|
| `crates/fnp-random/src/lib.rs:1` | placeholder `add` function | RNG subsystem is currently a stub; no constructor/state/spawn/jump contract exists yet |
| `crates/fnp-conformance/src/lib.rs:103` | fixture/report seed fields | harness carries seed fields but has no RNG-specific differential suite yet |
| `crates/fnp-conformance/fixtures/runtime_policy_cases.json:9` | runtime policy fixture seed metadata | seed-aware logging schema exists at program level |
| `crates/fnp-runtime/src/lib.rs:107` | `DecisionAuditContext.seed` | runtime decision ledger already models deterministic seed context |

## Graveyard and FrankenSuite Mapping

- `alien_cs_graveyard.md` §0.19 (evidence ledger schema)
- `alien_cs_graveyard.md` §0.4 (decision-theoretic runtime contracts)
- `alien_cs_graveyard.md` §6.12 (property-based testing with shrinking)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19

## Notes for Follow-on Packet Steps

- Packet B (`bd-23m.18.2`) must codify strict/hardened invariants for constructor normalization, seed/state transitions, spawn semantics, and jump behavior.
- Packet C (`bd-23m.18.3`) must define threat controls for seed/state abuse, replay poisoning, and override misuse.
- Packet D (`bd-23m.18.4`) must introduce real RNG module boundaries in `fnp-random` (generator facade, seed sequence model, bit-generator interfaces).
- Packet E/F (`bd-23m.18.5`, `bd-23m.18.6`) must add deterministic-seed witness suites and oracle differential coverage for constructor/state/spawn/jump families.
- Packet G (`bd-23m.18.7`) must attach replay-forensics scenarios with structured seed/artifact linkage.
