# Batch Cholesky SIMD-Across-Lanes No-Ship

Run identity:
- Date: 2026-06-20.
- Agent: BlackThrush / cod-a.
- Branch/worktree: `cod-a-batch-cholesky-simd-20260620` at baseline `64ad3a25`.
- Target: `fnp_linalg::batch_cholesky` medium stacked SPD matrices.
- Artifact directory: `tests/artifacts/perf/2026-06-20_linalg_batch_cholesky_simd_cod_a/`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.

Candidate:
- Safe Rust `std::simd::Simd<f64, 4>` across four independent batch lanes for `16 <= n < 128`.
- Per matrix, the `k` loop accumulation order stayed scalar-order identical.
- Tail lanes used `cholesky_nxn_into_out`.

Correctness:
- `rch exec -- cargo test -p fnp-linalg batch_cholesky_ -- --nocapture`
- Result: 3 passed, 0 failed, 1 ignored.
- The candidate proof `batch_cholesky_simd4_matches_per_lane_cholesky_nxn_bits` passed for n=16, 32, 64.

Performance:
- Baseline RCH Criterion on `ovh-a`:
  - `batch_cholesky/shape/64x128x128`: 1.6258 ms center.
  - `batch_cholesky/shape/16x256x256`: 3.2794 ms center.
- Candidate RCH Criterion on same worker `ovh-a`:
  - `64x128x128`: 2.3148 ms center, +45.662% regression.
  - `16x256x256`: 3.8253 ms center, +16.109% regression.
- Decision: NO-SHIP. Source reverted before commit.

NumPy comparator:
- Baseline local ABI Python sweep against NumPy 2.4.3, BLAS threads pinned to 1:
  - B=4000 d=8: 3.12x slower.
  - B=2000 d=16: 19.65x slower.
  - B=1000 d=32: 4.67x slower.
  - B=500 d=64: 6.10x slower.
  - B=200 d=100: 1.49x slower.
  - B=64 d=200: 1.36x slower.
  - B=10000 d=4: 2.42x slower.
- Candidate was not rerun against NumPy because the same-worker Rust broad gate already failed.

Ledger verdict:
- Candidate broad gate: 0 wins / 2 losses / 0 neutral.
- Existing code vs NumPy in this baseline sweep: 0 wins / 7 losses / 0 neutral.
- Retry predicate: do not retry gather/scatter `f64x4` portable-SIMD across batch lanes as a standalone Cholesky lever. A credible retry needs a different memory layout or batched-panel algorithm that proves medium rows and broad n>=128 rows in the same run window.
