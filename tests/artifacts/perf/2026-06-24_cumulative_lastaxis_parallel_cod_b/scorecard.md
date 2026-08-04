# cumsum/cumprod/nancum* last-axis parallel-across-lanes keep

Agent: CreamEagle / cod-b
Date: 2026-06-24
Crate: `fnp-python`
Surface: `np.cumsum`/`np.cumprod`/`np.nancumsum`/`np.nancumprod` along the contiguous
LAST axis of a C-contiguous f64 array (the shared `try_zerocopy_f64_cumulative_axis`
`inner == 1` path).

## Change

The last-axis cumulative scan was already native+zero-copy but ran SERIALLY across the
`outer` lanes (`for o in 0..outer`). Each lane is an INDEPENDENT contiguous prefix scan,
so the per-lane sequential add/mul latency chain can be fanned across the rayon pool;
numpy runs the whole thing single-threaded. Parallelized via
`out.par_chunks_mut(axis_len).zip(in.par_chunks(axis_len))` above a `1<<18` element gate.

Cache-friendly (each thread owns a contiguous row range — unlike a strided column scan).
Bit-exact: each lane's register-carried scan is unchanged; only the order independent
lanes are processed differs. Benefits cumsum, cumprod, nancumsum, nancumprod (and is the
analog of the var/std-axis lane parallelism). inner>1 (non-last axis) slab path unchanged.

## Benchmark

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo bench -p fnp-python --profile release \
  --bench criterion_python_surface cumsum_lastaxis \
  -- --sample-size 15 --warm-up-time 2 --measurement-time 4 --output-format bencher
```

| Row | FNP ns | NumPy ns | FNP/NumPy | Speedup |
|---|---:|---:|---:|---:|
| `cumsum(axis=-1) 8192x1024` | 2,780,644 | 30,185,124 | 0.092x | 10.86x |
| `cumsum(axis=-1) 65536x256` | 5,431,249 | 59,230,179 | 0.109x | 10.90x |

Verdict: keep. ~10.9x — parallelizing the previously-serial per-row scan against
single-threaded numpy.

## Correctness

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cc \
  rch exec -- cargo test -p fnp-python --test conformance_cumulative --test conformance_cumsum \
    --test conformance_cumsum_zerocopy --test conformance_cumprod_zerocopy --test conformance_nan_funcs
```

- `conformance_cumulative` 18/18, `conformance_cumsum` 2/2, `conformance_cumsum_zerocopy` 1/1,
  `conformance_cumprod_zerocopy` 2/2, `conformance_nan_funcs` 37/37 — all pass.
The change is bit-exact (per-lane scan identical to the serial version), so the existing
cumulative conformance suites are the proof; no new test needed.
