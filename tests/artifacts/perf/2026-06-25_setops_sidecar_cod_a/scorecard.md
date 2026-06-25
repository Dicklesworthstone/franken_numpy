# 2026-06-25 setops float32 setxor1d scorecard

Agent: BlackThrush

## Decision

KEEP: `setxor1d` now has a tightly gated native float32 eighth-step range bitmap.

The old non-main i64 sidecar worktree was not landed. Current main already wins at the Python boundary
for the measured i64 setops row: baseline `fnp_intersect1d_i64_smallrange_1m` 3.9149 ms vs NumPy
72.727 ms = 0.054x. The live loss was `fnp_setxor1d_f32_repeated_1m`: 19.163 ms vs NumPy 17.468 ms =
1.097x on the earlier vmi1152480 baseline.

## Candidate benchmark

Command:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo bench -p fnp-python --profile release --bench criterion_python_surface -- python_setops_boundary
```

Remote worker: `hz2`

Console rows:

| row | FNP | NumPy | FNP/NumPy |
| --- | ---: | ---: | ---: |
| setdiff1d_i32_smallrange_1m | 1.4725 ms | 6.3616 ms | 0.232x |
| intersect1d_i64_smallrange_1m | 2.5274 ms | 7.7152 ms | 0.328x |
| intersect1d_f64_repeated_1m | 121.14 ms | 125.35 ms | 0.966x |
| setxor1d_f32_repeated_1m | 11.103 ms | 70.786 ms | 0.157x |

Criterion JSON mean point estimates:

| row | FNP mean | NumPy mean | FNP/NumPy |
| --- | ---: | ---: | ---: |
| setdiff1d_i32_smallrange_1m | 1.489450 ms | 6.428611 ms | 0.232x |
| intersect1d_i64_smallrange_1m | 2.508593 ms | 7.634589 ms | 0.329x |
| intersect1d_f64_repeated_1m | 121.140602 ms | 125.350629 ms | 0.966x |
| setxor1d_f32_repeated_1m | 10.872153 ms | 72.355183 ms | 0.150x |

JSON sources under `/data/projects/.rch-targets/franken_numpy-cod-a/criterion/python_setops_boundary/*/new/estimates.json`.

## Validation

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-python setxor1d_f32_eighth_bitmap_matches_numpy_and_defers_edge_cases -- --nocapture
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-python --test conformance_setops -- --nocapture
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo check -p fnp-python --all-targets
```

Results:

- Targeted unit parity: passed. This run fell back local after RCH daemon recovery, so it is compile/parity feedback only.
- Remote `conformance_setops` on `ovh-a`: passed, 9/9 MUST, 13/13 SHOULD, 2/2 MAY.
- Remote `cargo check -p fnp-python --all-targets` on `hz2`: passed with pre-existing dead-code warnings.
- `cargo fmt -p fnp-python -- --check`: blocked by pre-existing package-wide formatting drift in untouched benches/tests; no package-wide rustfmt applied.
- `ubs crates/fnp-python/src/lib.rs docs/NEGATIVE_EVIDENCE.md tests/artifacts/perf/2026-06-25_setops_sidecar_cod_a/scorecard.md`: exit 1 from broad pre-existing `fnp-python/src/lib.rs` inventory (old unwrap/panic/security-heuristic/direct-index findings); its fmt/clippy/build sections were clean.
