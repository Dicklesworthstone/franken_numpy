# Baseline — core_ops (BASELINE phase)

- **Run ID:** `2026-06-01_perf_baseline`
- **Tool:** criterion, **n = 30 samples** (CLI `--sample-size 30 --measurement-time 8 --warm-up-time 3`; overrides the bench's hardcoded `sample_size(12)`)
- **Host:** AMD Ryzen Threadripper PRO 5975WX (64T), Ubuntu 25.10, kernel 6.17, ext4. Governor=`powersave` (untuned — see caveat in `00_scenario.md`). Bench execution `taskset -c 56-63`; compilation offloaded via rch.
- **Build:** `[profile.bench]` `opt-level=3`, `debug=line-tables-only`, `strip=false`, `RUSTFLAGS=-C force-frame-pointers=yes`.
- **Raw logs:** `core_ops_criterion_n30.log` (+ sanity `core_ops_criterion.log`, n=12). Criterion JSON: `/data/tmp/cargo-target/criterion/core_ops/*/new/estimates.json`.

## Results (median ± MAD, 95% CI)

| Op | Fixture | Median | 95% CI | MAD | Throughput |
|----|---------|--------|--------|-----|-----------|
| ufunc_add_broadcast | 1024×1024 ⊕ 1024 (1M out) | **14.863 ms** | [14.780, 15.007] | 0.254 ms | 67.3 M elem/s |
| sort_quicksort | 1M f64 | 4.669 ms | [4.527, 4.757] | 0.229 ms | 214 M elem/s |
| matmul | 256×256 · 256×256 | 2.759 ms | [2.753, 2.762] | 0.011 ms | ~12.2 GFLOP/s |
| fft | 65 536 complex | 2.509 ms | [2.506, 2.517] | 0.006 ms | 26.1 M elem/s |
| astype f64→i32 | 1024×1024 (1M) | 0.828 ms | [0.817, 0.841] | 0.026 ms | 1.21 B elem/s |
| reduce_sum_axis1 | 1024×1024 (1M in) | 0.699 ms | [0.6993, 0.6995] | 0.0003 ms | 1.43 B elem/s |
| reshape | 1024×1024→2048×512 | 0.189 ms | [0.187, 0.191] | 0.006 ms | metadata path |
| io_npy save+load | 512×512 f64 | 0.123 ms | [0.1228, 0.1230] | 0.0001 ms | 2.1 B elem/s |

## Variance

All CIs are tight (MAD ≤ 1.7% of median for every op except sort at 4.9% — quicksort pivot variance). Within the skill's ≤10% envelope; numbers are publishable. The 2026-05-05 baseline (n=12, on rch worker ts2) reported add_broadcast 16.85 ms / reduce 766 µs / matmul 3.11 ms / sort 4.85 ms / fft 2.87 ms — same ranking, ~10% faster here (different host + warm cache).

## Headline signal

`ufunc_add_broadcast` processes the **same 1 M elements** as `reduce_sum` but is **21.3× slower** (14.86 ms vs 0.699 ms). The gap is the entry point to the hotspot analysis (`20_hotspot_table.md`): it is **not** the arithmetic — it is result-array construction. See `levelsplit_errstate.txt` for the stage-isolation proof.
