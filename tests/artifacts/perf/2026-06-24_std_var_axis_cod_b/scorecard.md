# std/var last-axis native two-pass keep

Agent: CreamEagle / cod-b
Date: 2026-06-24
Crate: `fnp-python`
Surface: `np.var(a, axis=-1)` / `np.std(a, axis=-1)` for C-contiguous f64 arrays

## Change

Extend the existing flat `compute_f64_var_flat` idea to a last-axis helper:
`try_zerocopy_f64_var_axis`. The helper runs a no-allocation two-pass pairwise
fold per contiguous row, reusing the same pairwise sum and squared-deviation
kernel family as `nanvar`. It is gated to f64 C-contiguous ndarrays, single
last-axis reductions, native integer `ddof`, no `out`, no `dtype`, empty kwargs,
and falls back to NumPy for zero-length, `ddof >= len`, non-finite means, and
non-last axes.

## Benchmark

Command:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b \
  rch exec -- cargo bench -p fnp-python --profile release \
  --bench criterion_python_surface std_var_axis \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 3 --output-format bencher
```

RCH worker: `ovh-a`.

| Row | FNP ns | NumPy ns | FNP/NumPy | Speedup |
|---|---:|---:|---:|---:|
| `var(axis=-1) 4096x512` | 322,666 | 2,829,918 | 0.1140x | 8.77x |
| `std(axis=-1) 4096x512` | 326,623 | 2,786,256 | 0.1172x | 8.53x |
| `var(axis=-1) 8192x1024` | 1,476,796 | 18,021,325 | 0.0819x | 12.20x |
| `std(axis=-1) 8192x1024` | 1,543,123 | 18,401,976 | 0.0839x | 11.93x |

Verdict: keep. The new last-axis native path is a measured 8.5-12.2x faster
than NumPy on the targeted rows.

## Correctness

Command:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b \
  rch exec -- cargo test -p fnp-python --test conformance_std --test conformance_var
```

Result:

- `conformance_std`: 15 passed / 0 failed
- `conformance_var`: 15 passed / 0 failed

The behavior-preservation boundary is explicit: special values and unsupported
signature variants defer to NumPy, so the native helper only covers finite,
contiguous, last-axis f64 reductions where the pairwise fold is intended to
match NumPy's numeric result.

## Tool Notes

- `cargo check -p fnp-python --all-targets` passed through `rch exec`; existing
  dead-code warnings remain in `fnp-python` and `fnp-ufunc`.
- `cargo fmt -p fnp-python --check` reports broad pre-existing formatting drift
  across the crate. This perf commit did not apply a crate-wide rustfmt rewrite.
- Remote clippy on `vmi1153651` failed before code analysis because that worker
  lacks `cargo-clippy` for `nightly-2026-02-20`.
- Local `cargo clippy -p fnp-python --all-targets -- -D warnings` is blocked by
  existing `fnp-ufunc::nan_filtered` dead-code. Local `--no-deps` reaches
  `fnp-python` and reports the existing crate lint backlog; none of the reported
  line numbers are in the new `std`/`var` helper or benchmark group.
