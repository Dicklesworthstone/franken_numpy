# Python Einsum Reduce-All Current-Head Rerun

Run date: 2026-06-20
Agent: YellowElk / cod-b
Parent bead: `franken_numpy-ixs5y`
Crate: `fnp-python`
Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`

## Result

Current `main` was rerun before any source edit because the previous scorecard
called out `einsum_reduce_all_f64_1000` as the next visible residual. RCH
selected worker `vmi1293453`.

| Row | FNP | NumPy | FNP/NumPy | Verdict |
|---|---:|---:|---:|---|
| `einsum_trace_f64_4000` | 97,103 ns | 107,426 ns | 0.904x | win |
| `einsum_diag_f64_4000` | 2,244 ns | 2,483 ns | 0.904x | win |
| `einsum_reduce_all_f64_1000` | 438,624 ns | 600,537 ns | 0.730x | win |
| `einsum_reduce_rows_f64_1000` | 323,154 ns | 544,627 ns | 0.594x | win |
| `einsum_reduce_cols_f64_1000` | 624,904 ns | 732,167 ns | 0.854x | win |

Win/loss/neutral: 5/0/0. No source edit was made for this target.

## Decision

The former reduce-all near-loss is not a current actionable gap on this worker.
Do not reopen the scalar-builder or diagonal shortcut families for this row
without fresh losing evidence.
