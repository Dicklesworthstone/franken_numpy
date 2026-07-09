# 2026-07-09 Codex deeper primitive scorecard

## Primitive

Target: `python_c128_setops_boundary/fnp_union1d_c128_2m_2m`.

Technique: dense-domain `complex128` union using a direct 2-D presence table over
finite integer-valued `(real, imag)` components. The path validates the bounded
grid, marks presence from both inputs, and emits buckets in NumPy complex
lexicographic order. It falls back for NaN, `-0.0`, non-integral components,
small inputs, and too-wide domains.

## Profile

Command:

```text
AGENT_NAME=${AGENT_NAME:-Codex} CARGO_TARGET_DIR=/data/projects/.rch-targets/numpy-cod rch exec -- cargo bench -p fnp-python --profile release --bench criterion_python_surface -- 'python_argsort_temporal_complex_stable_boundary/(fnp_argsort_c128_dense_stable_8m|numpy_argsort_c128_dense_stable_8m)|python_argsort_string_stable_boundary/(fnp_argsort_U6_stable_2m|numpy_argsort_U6_stable_2m)|python_searchsorted_struct_boundary/(fnp_searchsorted_struct_2xi8_2m_2m|numpy_searchsorted_struct_2xi8_2m_2m)|python_c128_setops_boundary/(fnp_union1d_c128_2m_2m|numpy_union1d_c128_2m_2m)|python_datetime_searchsorted_isin_boundary/(fnp_searchsorted_datetime_2m_2m|numpy_searchsorted_datetime_2m_2m)' --sample-size 10 --warm-up-time 1 --measurement-time 2 --output-format bencher --noplot
```

Worker: `hz2`.

Relevant rows:

| Row | Time |
|---|---:|
| `fnp_union1d_c128_2m_2m` | 222.020 ms |
| `numpy_union1d_c128_2m_2m` | 501.006 ms |

## Bench

Command:

```text
AGENT_NAME=${AGENT_NAME:-Codex} CARGO_TARGET_DIR=/data/projects/.rch-targets/numpy-cod rch exec -- cargo bench -p fnp-python --profile release --bench criterion_python_surface -- 'python_c128_setops_boundary/(fnp_union1d_c128_2m_2m|numpy_union1d_c128_2m_2m)' --sample-size 10 --warm-up-time 1 --measurement-time 3 --output-format bencher --noplot
```

Worker: `vmi1264463`.

| Row | Time | Ratio |
|---|---:|---:|
| `fnp_union1d_c128_2m_2m` | 189.638 ms | 1.00x |
| `numpy_union1d_c128_2m_2m` | 5869.978 ms | 30.95x slower |

## Conformance

`cargo check -p fnp-python --lib --test conformance_setops --bench criterion_python_surface` passed on `hz2`.

`cargo test -p fnp-python --test conformance_setops -- --nocapture` passed on `ovh-a`:

```text
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

Setops matrix:

```text
MUST 9/9
SHOULD 14/14
MAY 2/2
```
