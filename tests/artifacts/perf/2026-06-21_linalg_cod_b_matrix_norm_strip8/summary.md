# 2026-06-21 fnp-linalg matrix norm current recheck

Bead: `franken_numpy-ixs5y.281`
Agent: `YellowElk` / `cod-b`

Current `fnp-linalg` medians from `rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg 'matrix_norm_nxn_orders/(one|neg_one)/(256|512|1024)'` on `vmi1152480`:

| Workload | FNP median ns |
|---|---:|
| `one/256` | 7,743 |
| `neg_one/256` | 5,207 |
| `one/512` | 26,211 |
| `neg_one/512` | 25,737 |
| `one/1024` | 99,936 |
| `neg_one/1024` | 98,382 |

Comparator status:
- `numpy_matrix_norm_vmi1152480.txt` is invalid evidence; SSH to the worker was denied.
- `numpy_matrix_norm_local_thinkstation1.txt` is local routing evidence only.
- The counted conclusion in `docs/NEGATIVE_EVIDENCE.md` uses current remote FNP plus prior direct `hz2` NumPy rows to mark the old matrix-norm gap stale, not a fresh same-host NumPy proof.

Validation:
- Focused bit-preservation test passed: `matrix_norm_column_reduction_matches_strided_reference_bits`.
- `cargo build -p fnp-linalg --release` passed through `rch`.
- No source change was kept or attempted for this slice.
