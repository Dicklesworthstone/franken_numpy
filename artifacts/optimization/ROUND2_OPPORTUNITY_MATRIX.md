# ROUND2_OPPORTUNITY_MATRIX

## Scope

Round 2 applied one optimization lever to the measured hotspot in `fnp-ufunc::UFuncArray::elementwise_binary`.

## Baseline + Profile Evidence

- Baseline command: `hyperfine --warmup 3 --runs 10 '/data/tmp/cargo-target/release/generate_benchmark_baseline'`
- Pre-change artifact: `artifacts/optimization/hyperfine_generate_benchmark_baseline_before.json`
- Post-change artifact: `artifacts/optimization/hyperfine_generate_benchmark_baseline_after.json`
- Syscall profile artifact (fallback because `perf_event_paranoid=4` blocked flamegraphs):
  - `strace -c /data/tmp/cargo-target/release/generate_benchmark_baseline`

## Opportunity Matrix

| Hotspot | Impact (1-5) | Confidence (1-5) | Effort (1-5) | Score | Decision |
|---|---:|---:|---:|---:|---|
| `fnp-ufunc::elementwise_binary` per-element broadcast reindexing | 5 | 4 | 2 | 10.0 | Implemented (Round 2) |
| `fnp-ufunc::reduce_sum` ravel/unravel loop | 4 | 4 | 3 | 5.3 | Deferred |
| `fnp-runtime` posterior/loss serialization overhead | 2 | 3 | 3 | 2.0 | Deferred |

## One-Lever Change

- Replaced repeated per-element `unravel_index + broadcasted_source_index` with an incremental broadcast cursor.
- The cursor advances output indices odometer-style and updates source flat offsets in O(1) per output element.
- API, dtype, and operation semantics unchanged.

## Measured Delta

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| mean wall time (ms) | 26.077 | 25.814 | -0.263 ms |
| mean wall time (%) | 100.00 | 98.99 | -1.01% |
| stddev (ms) | 0.786 | 0.538 | improved |

## Graveyard Mapping

- `alien_cs_graveyard.md` §0.1: Mandatory optimization loop
- `alien_cs_graveyard.md` §0.2: Opportunity matrix gate
- `alien_cs_graveyard.md` §0.3: Isomorphism proof block
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.2 and §0.12: optimization loop + evidence ledger discipline
