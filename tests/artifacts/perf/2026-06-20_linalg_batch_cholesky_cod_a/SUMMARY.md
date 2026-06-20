# 2026-06-20 linalg bold-verify summary

Agent: `YellowElk` / `cod-a`

Parent bead: `franken_numpy-ixs5y`

Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`

## Current wins and losses

| Row | FNP | NumPy | FNP/NumPy | Worker |
|---|---:|---:|---:|---|
| `batch_cholesky/shape/500x64x64` | 4,879,321 ns | 17,379,269 ns | 0.281x | `vmi1149989` |
| `batch_cholesky/shape/64x128x128` | 3,808,040 ns | 16,877,623 ns | 0.226x | `vmi1149989` |
| `batch_cholesky/shape/16x256x256` | 4,152,904 ns | 27,276,163 ns | 0.152x | `vmi1149989` |
| `eigvalsh_nxn/size/128` | 1,330,011 ns | 435,883 ns | 3.051x | `vmi1227854` |
| `cond_nxn/size/128` | 1,146,114 ns | 724,139 ns | 1.583x | `vmi1227854` |

## Rejected probes

| Probe | Result |
|---|---|
| `TRIDIAG_BLOCK_MIN=192` | Failed `tridiag_rank2k_fused_update_preserves_spectra_and_golden_sha256`; no benchmark counted. |
| private `cond_nxn` direct extrema scan | Paired `vmi1227854` recheck regressed `cond_nxn/128`: 1,191,551 ns vs baseline 1,161,511 ns. |
| public `eigvalsh_nxn sort_unstable_by(total_cmp)` | Paired `hz1` recheck regressed `eigvalsh_nxn/128`: 2,101,688 ns vs baseline 1,888,909 ns. |

Production source was restored to baseline after the probes.

## Raw logs

- `baseline_batch_cholesky_hz1.txt`
- `numpy_batch_cholesky_exact_vmi1149989.txt`
- `baseline_cond_eigvalsh_vmi1149989.txt`
- `numpy_cond_eigvalsh_vmi1227854.txt`
- `candidate_cond_scan_extrema_vmi1227854.txt`
- `baseline_recheck_cond_scan_extrema_vmi1227854.txt`
- `candidate_recheck_cond_scan_extrema_vmi1227854.txt`
- `candidate_sort_unstable_eigvalsh_vmi1227854.txt` (rch selected `hz1`)
- `baseline_recheck_sort_unstable_hz1.txt`
- `numpy_cond_eigvalsh_hz1.txt`
