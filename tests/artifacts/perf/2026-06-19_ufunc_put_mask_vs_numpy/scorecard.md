# FNP put_mask vs NumPy scorecard - 2026-06-19

Bead: `franken_numpy-ixs5y.254`

Subject: `fnp-ufunc` `UFuncArray::put_mask` on flat F64 destination and bool mask, no integer sidecars.

Kept lever: use the dtype/layout proof to bypass integer-sidecar mutation dispatch for F64/no-sidecar arrays, scan the serial path with 8-lane SIMD mask bit extraction, cycle the values index without `%` in the hot true-lane loop, and reserve Rayon for rows at or above `1 << 20` elements.

Artifacts:
- FNP fresh current baseline: `baseline_fnp_put_mask_rch.txt` (`hz2`).
- NumPy refreshed reference: `numpy_put_mask_local.txt` (`thinkstation1`, NumPy 2.4.3).
- Rejected threshold/direct serial probes: `candidate_fnp_put_mask_rch.txt`, `candidate_fnp_put_mask_rch_confirm.txt`.
- Rejected too-low SIMD parallel cutoff probe: `candidate_fnp_put_mask_simd_rch.txt`.
- Final FNP bench: `final_fnp_put_mask_simd_threshold_rch.txt` (`ovh-a`).
- Correctness proof: `test_put_mask_golden_threshold_1m_final.txt`.

| Variant | Worker/host | 100k FNP | 1M FNP | 100k FNP/NumPy | 1M FNP/NumPy | Verdict |
|---|---|---:|---:|---:|---:|---|
| Current fresh baseline | `hz2` | 244.411 us | 483.383 us | 7.866x | 0.704x | 100k loss selected |
| Threshold/direct serial, no SIMD | `hz1` | 76.797 us | 445.529 us | 2.472x | 0.649x | Rejected, still loses 100k |
| SIMD serial with `1 << 19` cutoff | `hz2` | 19.037 us | 709.480 us | 0.613x | 1.034x | Rejected, 1M neutral/loss |
| Final SIMD serial with `1 << 20` cutoff | `ovh-a` | 15.858 us | 335.444 us | 0.510x | 0.489x | Keep |

NumPy refreshed local reference:
- `put_mask_f64_masked_cycling/100000`: 31.069 us median.
- `put_mask_f64_masked_cycling/1000000`: 686.361 us median.

Win/loss/neutral:
- Current fresh baseline vs refreshed NumPy: 1/1/0.
- Final remote candidate vs refreshed NumPy: 2/0/0.

Same-worker historical routing note:
- Earlier same-day current-code ovh-a artifact in `tests/artifacts/perf/2026-06-19_ufunc_remaining_masked_vs_numpy/criterion_remaining_masked_current.txt` had `put_mask_f64_masked_cycling` at 81.857 us / 388.001 us.
- Final ovh-a is 15.858 us / 335.444 us, a 5.16x 100k speedup and 1.16x 1M speedup against that same-worker current snapshot.

Correctness:
- `put_mask_f64_parallel_matches_serial_reference_and_golden_sha256` passed after updating the intentional threshold-crossing digest to `f8a49cce66312e0fb3fdfdbcc5e31b70662343eff3f8d49ae4f01ae828da3c0c`.
- The sidecar fallback fixture now uses `(1_i64 << 53) - 2`, which stays within this crate's exact temporary F64 bridge contract while still proving large integer sidecar preservation.

Validation:
- `cargo check -p fnp-ufunc --all-targets` passed on rch worker `vmi1149989`.
- `cargo clippy -p fnp-ufunc --all-targets -- -D warnings` passed on rch worker `hz2`.
- `git diff --check` passed.
- `cargo fmt --check` still reports broad pre-existing workspace formatting drift outside this slice; the put_mask hunk was manually adjusted to match the targeted rustfmt diff.
- `timeout 120s ubs --only=rust crates/fnp-ufunc/src/lib.rs` did not produce a completion summary before the wrapper returned, and zsh did not preserve the exit code in the artifact; keep `ubs_fnp_ufunc_lib.txt` as inconclusive, not a pass.
