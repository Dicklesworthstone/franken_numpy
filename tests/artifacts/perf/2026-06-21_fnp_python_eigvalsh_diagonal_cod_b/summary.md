## Python-surface diagonal eigvalsh fast path

Agent: `YellowElk` / `cod-b`
Parent bead: `franken_numpy-ixs5y`
Date: 2026-06-21

Lever: exact `float64` 2-D square ndarray with diagonal selected `UPLO`
triangle bypasses the broad NumPy dense-eigvalsh delegate and returns sorted
diagonal values directly.

RCH benchmark command:

```text
AGENT_NAME=YellowElk CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-python --bench criterion_python_surface eigvalsh_diagonal_f64_2d -- --sample-size 10 --warm-up-time 1 --measurement-time 2 --output-format bencher
```

Same-process RCH benchmark worker: `hz2`.

| Row | FNP ns | NumPy ns | FNP/NumPy |
|---|---:|---:|---:|
| `python_linalg_boundary/fnp_eigvalsh_diagonal_f64_2d_n200` | 17,559 | 1,655,615 | 0.0106x |
| `python_linalg_boundary/fnp_eigvalsh_diagonal_f64_2d_n800` | 203,248 | 87,648,968 | 0.0023x |

Validation:

- `cargo test -p fnp-python eigvalsh_matches_numpy_across_uplo_batched_and_complex -- --nocapture`: pass on RCH worker `ovh-a`.
- `cargo build -p fnp-python --release`: pass on RCH worker `vmi1153651`.
- `git diff --check`: pass.
- `cargo fmt --check -p fnp-python`: reports broad pre-existing package rustfmt drift outside this hunk.
- `ubs crates/fnp-python/src/lib.rs crates/fnp-python/benches/criterion_python_surface.rs docs/NEGATIVE_EVIDENCE.md`: broad existing inventory; internal cargo check/test/clippy clean.
