# FNP place vs NumPy scorecard

Run date: 2026-06-20
Agent: BlackThrush
Bead: `franken_numpy-ixs5y.252`
Commit under test before landing: working tree candidate
Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`

## Workload

`place_f64_masked_cycling` uses a flat F64 destination, bool mask with true lanes at `i % 19 in {3, 7, 11}`, and 257 cyclic F64 values.

## Decision Table

| Row | FNP current | FNP final | NumPy median | Current FNP/NumPy | Final FNP/NumPy | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `place_f64_masked_cycling/100000` | 87.094 us | 21.050 us | 66.616 us | 1.307x loss | 0.316x win | Keep |
| `place_f64_masked_cycling/1000000` | 441.318 us | 273.974 us | 815.936 us | 0.541x win | 0.336x win | Keep |

## Ratios

Current vs NumPy: win/loss/neutral = 1/1/0.

Final vs NumPy: win/loss/neutral = 2/0/0.

FNP current -> final speedup:
- 100k: 4.14x faster.
- 1M: 1.61x faster.

## Lever

F64/bool/F64 no-sidecar `UFuncArray::place` now bypasses generic integer-sidecar mutation plumbing. Rows below `1 << 20` use an 8-lane SIMD mask scan plus modulo-free cyclic value indexing. Rows at or above `1 << 20` keep the segmented-prefix Rayon path with fixed 4K chunks.

## Evidence Files

- Current rch baseline command/worker: `baseline_fnp_place_rch.txt`.
- Current baseline estimates: `baseline_estimates_extracted_before_candidate.txt`.
- Candidate rch bench: `candidate_fnp_place_rch.txt`.
- Candidate Criterion estimates: `criterion_estimates_base_new.txt`.
- NumPy comparator: `numpy_place_local.txt`.
- Focused golden: `test_place_golden_candidate.txt`, `test_place_golden_final.txt`.
- Validation: `cargo_check_fnp_ufunc.txt`, `cargo_clippy_fnp_ufunc.txt`, `cargo_clippy_fnp_ufunc_retry_hz1.txt`, `cargo_fmt_fnp_ufunc_check.txt`, `git_diff_check.txt`, `ubs_fnp_ufunc_lib.txt`.

## Validation Summary

- Focused golden `place_f64_parallel_matches_serial_reference_and_golden_sha256`: passed after digest update to `41ebf3fa471d4b7c9b29ddc1cde3e96b7b972072359d9ed98ac53ee806bf7add`.
- `rch exec -- cargo check -p fnp-ufunc --all-targets`: passed on `hz1`.
- Initial clippy hit an rch worker `SIGILL` in `zerocopy` build script on `ovh-b`; retry on `hz1` passed.
- `git diff --check`: clean.
- `cargo fmt -p fnp-ufunc -- --check`: broad pre-existing formatting drift remains; no formatter was run.
- UBS: timed out at 120s after starting Rust scan; not counted as pass.
