# Dense Row-Unique Occupancy/Rank Rejection

Agent: BlackThrush
Date: 2026-07-09 UTC / 2026-07-08 America/New_York
Command: `AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/numpy-cod rch exec -- cargo bench -p fnp-python --profile release --bench criterion_python_surface -- 'python_unique_rows_(boundary|full_boundary)/(fnp_unique_rows_500k4_axis0|numpy_unique_rows_500k4_axis0|fnp_unique_rows_full_500k4|numpy_unique_rows_full_500k4)' --sample-size 10 --warm-up-time 1 --measurement-time 2 --output-format bencher --noplot`

Primitive tried: mixed-radix row code -> dense occupancy/count/rank table -> code-order decode, replacing the existing packed-composite u64 sort/group path for small-domain integer `unique(axis=0)`.

Candidate bench (`bench_unique_rows_release.txt`, worker `hz1`):

| Probe | Candidate FNP | ORIG NumPy | ORIG/candidate |
|---|---:|---:|---:|
| `unique(axis=0, return_index/inverse/counts)` | 21.891 ms | 1319.755 ms | 60.3x |
| `unique(axis=0)` | 10.954 ms | 1363.510 ms | 124.5x |

Reverted/current bench (`bench_unique_rows_reverted_release.txt`, worker `vmi1149989`):

| Probe | Current FNP | ORIG NumPy | ORIG/current |
|---|---:|---:|---:|
| `unique(axis=0, return_index/inverse/counts)` | 14.803 ms | 1008.451 ms | 68.1x |
| `unique(axis=0)` | 10.428 ms | 719.586 ms | 69.0x |

Decision: NO-SHIP. The dense table primitive remains fast versus ORIG NumPy but does not beat the existing packed-composite row-unique implementation; source and focused test changes were reverted.
