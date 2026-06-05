# BlackThrush Perf Proof: franken_numpy-71n7p

## Target

`UFuncArray::from_storage_with_dtype` rebuilt matching `ArrayStorage::F64`
outputs through `cast_to(F64) -> to_f64_vec() -> Self::new()`, forcing a
per-element `get_f64` dispatch on every F64 elementwise result.

## Lever

One lever only: for compatible F64 storage, move the inner `Vec<f64>` directly
through `Self::new(shape, values, dtype)`.

## Benchmark Evidence

Command:

```text
rch exec -- env RUSTUP_TOOLCHAIN=nightly-2026-02-20 cargo bench -p fnp-conformance --bench criterion_core_ops -- ufunc_add_broadcast
```

Worker: `vmi1152480`.

Before:

```text
core_ops/ufunc_add_broadcast/1024x1024_by_1024
time: [13.915 ms 14.389 ms 14.898 ms]
```

After:

```text
core_ops/ufunc_add_broadcast/1024x1024_by_1024
time: [1.1141 ms 1.2843 ms 1.4555 ms]
```

Center estimate speedup: `14.389 / 1.2843 = 11.20x`.

## Behavior Proof

Golden artifact:

```text
artifacts/proofs/BLACKTHRUSH_PERF_71N7P_GOLDEN.json
sha256 4202cf4b2f9304d5edb2ffc44434f69bb5c7ae930e1774425b03dc82d1707977
```

Isomorphism checks:

- F64 fast path preserves raw bits for `+0.0`, `-0.0`, payload NaN, and `+inf`.
- Broadcast-add reference output remains `[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]`.
- No ordering, tie-breaking, RNG, or floating-point arithmetic changed; only result construction bypassed redundant storage casting for already-F64 output.

Focused validation:

```text
rch exec -- env RUSTUP_TOOLCHAIN=nightly-2026-02-20 cargo test -p fnp-ufunc from_storage_with_dtype -- --nocapture
rch exec -- env RUSTUP_TOOLCHAIN=nightly-2026-02-20 cargo test -p fnp-conformance --test numpy_reference_ops ufunc_ops_match_live_numpy_reference -- --nocapture
rch exec -- env RUSTUP_TOOLCHAIN=nightly-2026-02-20 cargo check -p fnp-ufunc --all-targets
rch exec -- env RUSTUP_TOOLCHAIN=nightly-2026-02-20 cargo clippy -p fnp-ufunc --all-targets -- -D warnings
```

Formatting:

```text
rustfmt --edition 2024 --check crates/fnp-ufunc/src/lib.rs
```

UBS:

```text
ubs crates/fnp-ufunc/src/lib.rs artifacts/proofs/BLACKTHRUSH_PERF_71N7P.md artifacts/proofs/BLACKTHRUSH_PERF_71N7P_GOLDEN.json .skill-loop-progress.md
```

UBS exit code was `1` due to the pre-existing monolithic `fnp-ufunc` warning set;
its own summary reported formatting, clippy, cargo check, test build,
cargo-audit, and cargo-deny clean for the scanned file.

Score gate: Impact `5` x Confidence `5` / Effort `1` = `25`.
