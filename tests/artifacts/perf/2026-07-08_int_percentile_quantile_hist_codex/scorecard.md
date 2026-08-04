# 2026-07-08 integer percentile/quantile histogram rank scorecard

## Candidate

- Surface: `fnp-python` scalar `percentile` / `quantile`
- Lever: bounded integer histogram order-statistics for default linear scalar rank lookup
- Comparator / ORIG: current fallback path that delegates to NumPy
- Guardrails: axis None / flatten only, scalar finite q, default or `method="linear"` only, no out, no keepdims, no weights, no overwrite_input, integer buffer only, bounded span <= `1 << 22`, large n only

## Bench

Command:

```bash
AGENT_NAME=${AGENT_NAME:-Codex} CARGO_TARGET_DIR=/data/projects/.rch-targets/numpy-cod rch exec -- cargo bench -p fnp-python --profile release --bench criterion_python_surface -- 'python_int_percentile_quantile_histogram_boundary/(fnp_percentile_i64_dense_16m_p12_5|numpy_percentile_i64_dense_16m_p12_5|fnp_quantile_u16_dense_16m_q75|numpy_quantile_u16_dense_16m_q75)' --sample-size 10 --warm-up-time 1 --measurement-time 3 --output-format bencher --noplot
```

RCH had no admissible workers at run time and fell back locally. Full output is in `bench_release.txt`.

| Probe | fnp | ORIG numpy delegate | ORIG/fnp |
|---|---:|---:|---:|
| `percentile(i64[16M], q=12.5)` | 9,576,870 ns | 193,727,193 ns | 20.23x |
| `quantile(u16[16M], q=0.75)` | 14,613,098 ns | 100,890,597 ns | 6.90x |

## Correctness

Focused conformance:

```bash
AGENT_NAME=${AGENT_NAME:-Codex} CARGO_TARGET_DIR=/data/projects/.rch-targets/numpy-cod rch exec -- cargo test -p fnp-python --test conformance_percentile_median percentile_quantile_large_bounded_integer_scalar_match_numpy -- --nocapture
```

Result: PASS, 1 passed / 0 failed.

Focused compile:

```bash
AGENT_NAME=${AGENT_NAME:-Codex} CARGO_TARGET_DIR=/data/projects/.rch-targets/numpy-cod rch exec -- cargo check -p fnp-python --lib --bench criterion_python_surface --test conformance_percentile_median
```

Result: PASS. Existing warning debt remains in `fnp-python` and dependency crates.

Focused shard conformance:

```bash
AGENT_NAME=${AGENT_NAME:-Codex} CARGO_TARGET_DIR=/data/projects/.rch-targets/numpy-cod rch exec -- cargo test -p fnp-python --test conformance_percentile_median -- --nocapture
```

Result: PASS, 26 passed / 0 failed.

Other checks:

- `git diff --check`: PASS.
- `UBS_MODULE_TIMEOUT=900 ubs --only=rust crates/fnp-python/src/lib.rs crates/fnp-python/tests/conformance_percentile_median.rs crates/fnp-python/benches/criterion_python_surface.rs`: completed, but exited nonzero on broad pre-existing inventory in the scanned giant files (panic/unwrap surfaces, unsafe inventory, security heuristics); UBS internal build-health section reported formatting, clippy, cargo check, and test-build clean.
- `cargo clippy -p fnp-python --lib --test conformance_percentile_median -- -D warnings`: blocked by pre-existing dependency warning `fnp-ufunc::UFuncArray::nan_filtered` dead code before reaching this lane.
