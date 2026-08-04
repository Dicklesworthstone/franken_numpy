# UBS touched-subset summary

Command:

```sh
ubs crates/fnp-ufunc/src/lib.rs docs/NEGATIVE_EVIDENCE.md tests/artifacts/perf/2026-06-19_ufunc_extract_simd_cod_a/scorecard.md
```

Result: exit 1 after 222 seconds.

UBS scanned one Rust source file, `crates/fnp-ufunc/src/lib.rs`, and reported
the existing broad inventory in that 70k-line file:

- Critical issues: 488
- Warning issues: 14,615
- Info items: 3,244

Representative sampled findings were on pre-existing lines, including
`unwrap()`/`panic!` in old code and tests, broad direct-indexing inventory, and
security heuristics that classify numeric equality checks as token comparisons.

This perf change is still guarded by the focused golden tests, `cargo check -p
fnp-ufunc --all-targets`, `cargo clippy -p fnp-ufunc --all-targets -- -D
warnings`, and `git diff --check`.
