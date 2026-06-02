# franken_numpy-3any2 - FFT first butterfly proof

## Target

- Bead: `franken_numpy-3any2`
- Hotspot: `core_ops/fft_65536`
- Baseline source: rch Criterion `core_ops` profile on `vmi1149989` after commit `30a6f57c`
- Baseline: `4.5750 ms` center, CI `[4.2350 ms, 4.9701 ms]`

Sort remained higher wall time at `6.1723 ms`, but the next obvious sort lever
was rejected under the proof rules. This pass targeted the next profiler-evident
hotspot with a separate score.

## Lever

`fft_pow2` used the generic butterfly loop for `len=2`, where every butterfly has
twiddle `w = (1, 0)`. The change handles that first pass directly after
bit-reversal and then starts the unchanged generic loop at `len=4`.

The direct path preserves the old zero-twiddle behavior explicitly:

- `fft_mul(0, finite)` is represented by `0.0 * finite`, preserving signed zero.
- `fft_mul(0, nonfinite)` is represented by `0.0`, preserving the Inf/NaN guard.
- The remaining multiply-by-one terms are direct value uses.

## Measurements

Profile baseline command:

```bash
AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/tmp/rch_target_franken_numpy_blackthrush \
  rch exec -- cargo bench -p fnp-conformance --bench criterion_core_ops -- \
  core_ops --sample-size 12 --measurement-time 5 --warm-up-time 2
```

Baseline on `vmi1149989`: `fft_65536` center `4.5750 ms`, CI
`[4.2350 ms, 4.9701 ms]`.

Same-worker golden/timing harness command:

```bash
AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/tmp/rch_target_franken_numpy_blackthrush \
  rch exec -- cargo run --release -p fnp-ufunc --example perf_fft_golden
```

Result on `vmi1149989`:

- current FNV: `0x6e299d6aba6a9af4`
- reference FNV: `0x6e299d6aba6a9af4`
- reference median: `2.6408 ms`
- current median: `2.5827 ms`

Official after Criterion command:

```bash
AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/tmp/rch_target_franken_numpy_blackthrush \
  rch exec -- cargo bench -p fnp-conformance --bench criterion_core_ops -- \
  fft_65536 --sample-size 12 --measurement-time 5 --warm-up-time 2
```

Result on `vmi1227854`: `4.6574 ms` center, CI
`[4.3350 ms, 5.0250 ms]`. This is cross-worker supporting evidence only.

Score: Impact `1` x Confidence `3` / Effort `1` = `3.0`.

## Behavior proof

- Ordering and butterfly topology are unchanged after bit reversal.
- The specialized pass is algebraically the old `len=2` generic pass with
  `w_re = 1.0` and `w_im = 0.0`.
- Signed-zero, Inf, and NaN zero-twiddle behavior is preserved by matching
  `fft_mul(0, x)` for finite and non-finite inputs.
- The generic `len>=4` loop, inverse scaling, Bluestein path, and RNG state are
  unchanged.

## Validation

- `rch exec -- cargo run --release -p fnp-ufunc --example perf_fft_golden`: passed,
  FNV `0x6e299d6aba6a9af4`.
- `rch exec -- cargo test -p fnp-ufunc fft --lib -- --nocapture`: 38 passed.
- `rch exec -- cargo test -p fnp-ufunc --test public_api_golden public_api_output_matches_golden -- --nocapture`: passed.
- `rch exec -- cargo check -p fnp-ufunc --all-targets`: passed.
- `rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings`: passed.
- `rustfmt --edition 2024 --check crates/fnp-ufunc/src/lib.rs`: passed.
- Public golden sha256:
  `1caa4743eb3f8f97cfc6bb08e373d14c17eafb2573583c2d8b9c2847d7bc0ad1`.
- `ubs crates/fnp-ufunc/src/lib.rs`: exit 1 for pre-existing monolith inventory;
  UBS fmt, clippy, cargo check, test build, audit, and deny sections were clean.
