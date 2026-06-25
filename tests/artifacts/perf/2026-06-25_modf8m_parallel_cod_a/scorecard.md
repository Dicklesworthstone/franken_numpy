# 2026-06-25 - f64 modf 8M parallel reverify

Agent: BlackThrush
Crate: `fnp-python`
Target dir: `/data/projects/.rch-targets/franken_numpy-cod-a`

## Subject

`try_zerocopy_f64_modf` was already landed in `4e87a144` with a large-buffer raw-slice Rayon map. This pass landed the durable 8M Python-boundary Criterion row and an above-gate byte-exact conformance test.

## Benchmark

Command:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo bench -p fnp-python --profile release --bench criterion_python_surface -- 'python_modf_boundary|python_clip_boundary' --sample-size 10 --warm-up-time 1 --measurement-time 3
```

RCH worker: `ovh-a`.

| Row | FNP p50 | NumPy p50 | FNP/NumPy | Verdict |
|---|---:|---:|---:|---|
| `clip_f32_8m` | 3.0458 ms | 4.9897 ms | 0.610x | existing f32 clip win reverified |
| `modf_f64_1m` | 269.35 us | 1.7925 ms | 0.150x | below-gate serial path still wins |
| `modf_f64_8m` | 15.936 ms | 83.305 ms | 0.191x | keep; 5.23x faster than NumPy |

## Validation

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-python --test conformance_nan_to_num_clip clip_f32_parallel_large_bit_exact_matches_numpy -- --nocapture
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-python --test conformance_frexp_modf modf_f64_parallel_large_bit_exact_matches_numpy -- --nocapture
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo check -p fnp-python --all-targets
rustfmt --edition 2024 --check crates/fnp-python/tests/conformance_nan_to_num_clip.rs crates/fnp-python/tests/conformance_frexp_modf.rs
git diff --check
```

Results:

- `clip_f32_parallel_large_bit_exact_matches_numpy`: pass. RCH fell back local because workers were full; correctness only, not timing.
- `modf_f64_parallel_large_bit_exact_matches_numpy`: pass. RCH fell back local because workers were full; correctness only, not timing.
- `cargo check -p fnp-python --all-targets`: pass on `vmi1264463`.
- `rustfmt --check` on the two touched test files: pass.
- `git diff --check`: pass.
- `cargo fmt -p fnp-python -- --check`: blocked by broad pre-existing `fnp-python` formatting drift.
- `cargo clippy -p fnp-python --all-targets -- -D warnings`: blocked before fnp-python changes by pre-existing `fnp-ufunc::nan_filtered` dead-code warning.

