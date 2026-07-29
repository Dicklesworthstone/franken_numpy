# 2026-07-09 complex stable argsort bounded-grid scorecard

## Profile Target

Profile command:

```bash
AGENT_NAME=${AGENT_NAME:-Codex} CARGO_TARGET_DIR=/data/projects/.rch-targets/numpy-cod rch exec -- cargo bench -p fnp-python --profile release --bench criterion_python_surface -- 'python_argsort_temporal_complex_stable_boundary/(fnp_argsort_c128_dense_stable_8m|numpy_argsort_c128_dense_stable_8m)|python_argsort_string_stable_boundary/(fnp_argsort_U6_stable_2m|numpy_argsort_U6_stable_2m)|python_searchsorted_struct_boundary/(fnp_searchsorted_struct_2xi8_2m_2m|numpy_searchsorted_struct_2xi8_2m_2m)|python_c128_setops_boundary/(fnp_union1d_c128_2m_2m|numpy_union1d_c128_2m_2m)|python_datetime_searchsorted_isin_boundary/(fnp_searchsorted_datetime_2m_2m|numpy_searchsorted_datetime_2m_2m)' --sample-size 10 --warm-up-time 1 --measurement-time 2 --output-format bencher --noplot
```

Profile artifact: `profile_candidates.txt`.

| Rank | Row | FNP time | ORIG NumPy time | Notes |
|---:|---|---:|---:|---|
| 1 | `argsort_c128_dense_stable_8m` | 605,388,340 ns | 2,093,880,676 ns | selected hot residual |
| 2 | `searchsorted_struct_2xi8_2m_2m` | 477,775,765 ns | 7,528,050,884 ns | already specialized |
| 3 | `argsort_U6_stable_2m` | 179,735,259 ns | 677,670,324 ns | already specialized |
| 4 | `union1d_c128_2m_2m` | 153,326,270 ns | 510,299,313 ns | already specialized |
| 5 | `searchsorted_datetime_2m_2m` | 45,701,053 ns | 1,073,581,496 ns | already specialized |

## Candidate

- Surface: `fnp.argsort(a, kind="stable"|"mergesort")` for flat `complex64` / `complex128`.
- Lever: bounded-grid 2-D histogram over integer-valued `(real, imag)` components.
- Comparator / ORIG: NumPy stable complex argsort.
- Fallback: non-flat, non-contiguous, small n, NaN/inf, non-integral components, or grid size above `1 << 18` uses the existing comparison path / NumPy passthrough.
- Graveyard lineage: radix hash join local histograms + prefix partitioning, applied to lexicographic ordering rather than joins.

## Bench

Command:

```bash
AGENT_NAME=${AGENT_NAME:-Codex} CARGO_TARGET_DIR=/data/projects/.rch-targets/numpy-cod rch exec -- cargo bench -p fnp-python --profile release --bench criterion_python_surface -- 'python_argsort_temporal_complex_stable_boundary/(fnp_argsort_c128_dense_stable_8m|numpy_argsort_c128_dense_stable_8m)' --sample-size 10 --warm-up-time 1 --measurement-time 3 --output-format bencher --noplot
```

Worker: `ovh-a`. Full output: `bench_complex_argsort_counting.txt`.

| Probe | fnp | ORIG NumPy | ORIG/fnp |
|---|---:|---:|---:|
| `argsort(complex128[8M], stable)` | 58,866,541 ns | 1,640,191,335 ns | 27.87x |

## Correctness

Focused compile:

```bash
AGENT_NAME=${AGENT_NAME:-Codex} CARGO_TARGET_DIR=/data/projects/.rch-targets/numpy-cod rch exec -- cargo check -p fnp-python --lib --test conformance_sort_search --bench criterion_python_surface
```

Result: PASS, with existing warning debt.

Focused conformance:

```bash
AGENT_NAME=${AGENT_NAME:-Codex} CARGO_TARGET_DIR=/data/projects/.rch-targets/numpy-cod rch exec -- cargo test -p fnp-python --test conformance_sort_search argsort_temporal_complex_stable_dense_matches_numpy -- --nocapture
```

Result: PASS, 1 passed / 0 failed.

## Caveats

- `cargo clippy -D warnings` remains blocked by pre-existing warning debt (`fnp-ufunc::nan_filtered` dead code was already observed in this checkout).
- Agent Mail registration/reservation was attempted but the server refused writes due a corrupted SQLite database circuit breaker.
