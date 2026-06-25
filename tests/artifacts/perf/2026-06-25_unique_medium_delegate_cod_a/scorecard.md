# 2026-06-25 unique medium delegate verification

Agent: `BlackThrush` / `cod-a`

Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`

Bench command:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo bench -p fnp-python --profile release --bench criterion_python_surface -- unique_medium
```

## Final current-main rows

Implementation source was restored to the committed Vec-based `try_zerocopy_f64_unique_flat`
path before this run. `rch` fell back local because remote workers were excluded/busy.

| Row | FNP mean | NumPy mean | FNP/NumPy | Verdict |
|---|---:|---:|---:|---|
| `unique(float64 repeated)`, 50k | 428.43 us | 417.77 us | 1.026x | delegate parity/slight loss |
| `unique(float64 repeated)`, 512k | 5.7810 ms | 5.7695 ms | 1.002x | delegate parity |
| `unique(float64 repeated)`, 1,048,576 | 15.611 ms | 12.170 ms | 1.283x | native loss |
| `unique(float64 distinct)`, 1,048,576 | 12.467 ms | 12.627 ms | 0.987x | tiny existing native win |

## Rejected lever

Tested allocating `numpy.empty(n)`, copying the borrowed input into it, parallel-sorting in
the output buffer, compacting adjacent duplicates, and shrinking with `resize(refcheck=False)`.

Rejected evidence:

| Candidate | Worker | Row | FNP mean | NumPy mean | FNP/NumPy | Decision |
|---|---|---|---:|---:|---:|---|
| blanket output-buffer | `vmi1149989` | repeated 1,048,576 | 10.064 ms | 9.7204 ms | 1.035x | revert |
| high-cardinality gated | local fallback | distinct 1,048,576 | 13.231 ms | 12.700 ms | 1.042x | revert |

The source lever was reverted. Durable kept changes are the benchmark surface and the
above-gate conformance row.
