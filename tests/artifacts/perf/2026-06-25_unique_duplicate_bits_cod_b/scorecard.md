# f64 unique duplicate-bit HashSet no-ship

Agent: BlackThrush / cod-b
Date: 2026-06-25
Crate: fnp-python

## Lever

Tested a duplicate-heavy f64 `np.unique` fast path that sampled cardinality, then
collected raw `f64::to_bits()` keys into `HashSet<u64, FastIntBuildHasher>`,
sorted the resulting unique values, and emitted the right-sized NumPy output.
NaNs and mixed signed zeros deferred to the existing path. Source change was
reverted after measurement.

## Main routing baseline

Command:

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-python --profile release --bench criterion_python_surface unique_medium -- --sample-size 10 --warm-up-time 1 --measurement-time 3 --output-format bencher
```

Worker: `vmi1152480`

| Row | FNP ns | NumPy ns | FNP/NumPy |
|---|---:|---:|---:|
| repeated 50k | 575,764 | 607,129 | 0.948x |
| repeated 512k | 9,438,837 | 9,739,316 | 0.969x |
| repeated 1m gate | 19,620,280 | 12,225,741 | 1.605x |
| distinct 1m gate | 20,939,530 | 12,207,685 | 1.715x |

## Candidate

Command:

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-python --profile release --bench criterion_python_surface unique_medium -- --sample-size 10 --warm-up-time 1 --measurement-time 3 --output-format bencher
```

Worker: `hz2`

| Row | FNP ns | NumPy ns | FNP/NumPy |
|---|---:|---:|---:|
| repeated 50k | 2,000,509 | 1,989,030 | 1.006x |
| repeated 512k | 26,485,917 | 27,363,289 | 0.968x |
| repeated 1m gate | 4,760,984,153 | 61,839,986 | 76.996x |
| distinct 1m gate | 36,267,207 | 57,284,282 | 0.633x |

## Verdict

No-ship. The duplicate-heavy 1m target row regressed to a catastrophic `76.996x`
loss versus NumPy. Keep only the added byte-level conformance row for the
repeated modulo f64 input; do not retry this raw-bit HashSet strategy.
