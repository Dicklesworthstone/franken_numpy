# BlackThrush Perf Proof: franken_numpy-61ges

## Target

`UFuncArray::sort(None, Some("quicksort"))` remained the top measured
`criterion_core_ops` hotspot after the earlier F64 construction and dense-sort
passes.

Fresh pass-7 baseline:

```text
worker: vmi1149989
command: rch exec -- cargo bench -p fnp-conformance --bench criterion_core_ops -- core_ops --sample-size 12 --measurement-time 5 --warm-up-time 2
core_ops/sort_quicksort_1m
time: [10.286 ms 11.574 ms 12.877 ms]
```

The benchmark input is a 1,000,000-element exact integer permutation, so it
exercises `exact_i64_counting_sort_output` through the dense unique quicksort
fast path.

## Lever

One lever only: when the validated integer range is dense
(`range_len == data.len()`), verify uniqueness with a compact `Vec<u64>` bitmap
before allocating the existing `Vec<usize>` count array.

If every bucket is seen once, the function emits the same sorted dense sequence
as the previous path. If a duplicate is found, it falls through to the existing
counting sort path, preserving multiset behavior.

Score gate: Impact 4 x Confidence 5 / Effort 2 = 10.

## Benchmark Evidence

Same worker as baseline: `vmi1149989`.

Before:

```text
core_ops/sort_quicksort_1m
time: [10.286 ms 11.574 ms 12.877 ms]
```

After:

```text
command: rch exec -- cargo bench -p fnp-conformance --bench criterion_core_ops -- sort_quicksort_1m --sample-size 12 --measurement-time 5 --warm-up-time 2
core_ops/sort_quicksort_1m
time: [3.2517 ms 3.3772 ms 3.5428 ms]
```

Center estimate speedup: `11.574 / 3.3772 = 3.43x`.

## Behavior Proof

Isomorphism checks:

- Ordering preserved: yes. Dense permutation output is still generated as
  `min_value..=max_value` in increasing order.
- Tie-breaking unchanged: yes. The dense unique path has no equal-value ties;
  duplicate inputs fall through to the previous counting path.
- Floating-point preserved: yes. The existing finite, exact-integer, and
  negative-zero rejection checks still gate the path. NaN, negative zero,
  non-integer, and overflow cases use the existing fallback behavior.
- RNG unchanged: N/A.
- Golden outputs unchanged:
  `crates/fnp-ufunc/tests/golden/public_api.golden`
  sha256 `1caa4743eb3f8f97cfc6bb08e373d14c17eafb2573583c2d8b9c2847d7bc0ad1`.

Focused validation:

```text
rch exec -- cargo test -p fnp-ufunc sort_kind_quicksort --lib -- --nocapture
result: 5 passed

rch exec -- cargo test -p fnp-ufunc --test public_api_golden public_api_output_matches_golden -- --nocapture
result: 1 passed

rch exec -- cargo check -p fnp-ufunc --all-targets
result: passed

rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings
result: passed

rustfmt --edition 2024 --check crates/fnp-ufunc/src/lib.rs
result: passed
```

UBS:

```text
ubs crates/fnp-ufunc/src/lib.rs
exit: 1
```

The UBS failure is the known broad monolith inventory for `fnp-ufunc/src/lib.rs`
(thousands of pre-existing unwrap/assert/index/float-equality/security-heuristic
findings). Its own fmt, clippy, cargo check, test build, cargo-audit, and
cargo-deny sections were clean, and it did not identify a new finding specific
to the bitmap path.
