# Core `UFuncArray` last-axis sum: coarse row-band keep

Agent: `GoldSummit`
Date: 2026-07-10
Bead: `franken_numpy-ixs5y.283`
Crate: `fnp-ufunc`
Benchmark: `core_ops/reduce_sum_axis1_1024x1024`

## Scope

This is the core `UFuncArray::reduce_sum` sibling of the 2026-06-24
`fnp-python` `try_zerocopy_f64_sum_lastaxis` keep. The Python-surface keep uses
NumPy-compatible pairwise lanes; this change only addresses the previously serial
contiguous last-axis branch in `reduce_sum_axis_contiguous` and continues to call the
existing `reduce_sum_values` for every row.

## Profile and lever

The strict-remote baseline profile used `perf record -F 399 -e cycles:u -g
--call-graph dwarf,16384` around the Criterion row. It captured about 3,000 samples
with zero lost; `UFuncArray::reduce_sum` held 95.45% self-time.

One lever was applied: once the contiguous last-axis input reaches `1 << 18`
elements, distribute independent output rows to coarse Rayon row bands. Each row is
still reduced serially by the unchanged `reduce_sum_values`, preserving its input
order and floating-point bit pattern. The first granular row-task version profiled
substantial scheduler and crossbeam-epoch overhead; grouping work into two row bands
per worker is the same ownership lever with lower scheduling frequency.

## Median gate

Both baseline and candidate used worker `vmi1149989`, `RAYON_NUM_THREADS=8`, the
same benchmark input, and this strict-remote command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1149989 RAYON_NUM_THREADS=8 env -u CARGO_TARGET_DIR rch exec -- cargo bench -p fnp-conformance --profile release-perf --bench criterion_core_ops -- reduce_sum_axis1_1024x1024 --sample-size 30 --warm-up-time 3 --measurement-time 8 --output-format bencher --noplot
```

| read | baseline median | candidate median | candidate / baseline | speedup | median reduction |
|---|---:|---:|---:|---:|---:|
| candidate 1 | 519,108.222 ns | 331,913.276 ns | 0.63939x | **1.56399x** | **36.0609%** |
| candidate 2 | 519,108.222 ns | 437,321.594 ns | 0.84245x | **1.18702x** | **15.7552%** |

The bencher summaries were `519,108 ns/iter (+/- 13,499)`, `331,913 ns/iter
(+/- 131,319)`, and `437,321 ns/iter (+/- 319,024)`; the table uses Criterion's
median point estimates. The second candidate read intentionally repeated the exact
strict-remote command on the same worker. Its spread is broad, but the requested
median direction replicated. The decision gate is the median, not the reported
spread.

## Bit proof and validation

`reduce_sum_last_axis_parallel_matches_serial_row_bits` compares raw `f64::to_bits()`
outputs with a serial per-row `reduce_sum_values` reference above the parallel gate.
Its rows cover cancellation-sensitive values, signed zero, NaN, positive and negative
infinity, and deterministic finite values. The strict-remote focused reduction suite
passed 41 tests with one ignored performance test.

Strict-remote `cargo clippy -p fnp-ufunc --all-targets -- -D warnings -A
dead-code` passed. The allowance is limited to the existing `nan_filtered` warning.
The required workspace check remains independently red in three pre-existing
`fnp-python` `where_py` test call sites; no `fnp-python` file is part of this change.
The repository-wide rustfmt check also has pre-existing drift elsewhere in the large
`fnp-ufunc` source file, while `git diff --check` passes for this patch.

UBS was run on every changed file. It exited nonzero on its existing whole-file
inventory for the roughly 70K-line `fnp-ufunc` source (for example, test panics and
generic float/security heuristics); its reported examples were outside this lever.
Because UBS has no clean changed-line baseline here, that infrastructure result is
surfaced rather than conflated with the clean targeted clippy and bit-proof gates.

## Decision

SHIP. Both same-worker candidate medians beat the baseline and the reduction order is
bit-identical. This is a core self-time improvement, not a new NumPy-relative claim.
