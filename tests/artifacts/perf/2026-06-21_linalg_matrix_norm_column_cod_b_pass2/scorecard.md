# Matrix Norm Column Current Recheck

Run: `2026-06-21_linalg_matrix_norm_column_cod_b_pass2`

Agent: `YellowElk` / `cod-b`

Parent bead: `franken_numpy-ixs5y`

This is a BOLD-VERIFY current-code recheck of the previous
`matrix_norm_nxn_orders/(one|neg_one)` column-sum residual. No source change was
made in this pass. Current `main` already contains the safe `std::simd`
cache-linear column accumulation path, so the proof is a head-to-head
FrankenNumPy-vs-NumPy measurement plus focused conformance.

Authoritative same-worker comparator: RCH selected `vmi1152480` for the Rust
bench in `current_matrix_norm_column_hz1_rerun.txt`; the NumPy script was then
run directly on `vmi1152480` and saved as `numpy_matrix_norm_column_vmi1152480.txt`.

| Workload | FNP ns | NumPy ns | FNP/NumPy | Verdict |
|---|---:|---:|---:|---|
| `one/128` | 7,684 | 9,615 | 0.799x | win |
| `neg_one/128` | 7,773 | 9,583 | 0.811x | win |
| `one/256` | 4,983 | 22,594 | 0.221x | win |
| `neg_one/256` | 5,129 | 27,742 | 0.185x | win |
| `one/512` | 25,621 | 97,495 | 0.263x | win |
| `neg_one/512` | 25,818 | 93,719 | 0.275x | win |
| `one/1024` | 129,460 | 478,653 | 0.270x | win |
| `neg_one/1024` | 122,906 | 461,018 | 0.267x | win |

Ratio scorecard: win/loss/neutral = **8/0/0**.

Validation:
- `cargo test -p fnp-linalg matrix_norm_column_reduction_matches_strided_reference_bits --release -- --nocapture` passed on RCH-selected `vmi1153651`.
- `cargo build -p fnp-linalg --release` passed on RCH-selected `vmi1152480`.
- No production source changes were made or kept.

Decision:
- The older matrix-norm column residual is stale on current `main`.
- Do not retry allocation-only stack-threshold or NaN-prefilter families for
  this lane.
- Reopen only if a same-worker rerun shows a current FNP/NumPy loss or if the
  column-sum kernel changes its scalar addition order, NaN behavior, or stride
  contract.
