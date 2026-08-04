# fnp-python einsum diagonal pre-policy shortcut no-ship scorecard

Run identity:
- Bead: `franken_numpy-ixs5y.269`.
- Agent: `BlackThrush` / `cod-a`.
- Subject API: Python-boundary `fnp.einsum` through `criterion_python_surface`.
- Oracle/reference: NumPy inside the same Criterion harness process.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Decision: no-ship; the source hunk was reverted.

Lever tested:
- Route f64 single-operand diagonal/trace einsum forms through the existing zero-copy diagonal fast path before wrapper dtype-policy work.
- Intended alien mapping: constants-kill-you specialization plus zero-copy/view-preserving layout reuse.
- Failure mode: the diagonal target still lost to NumPy after the shortcut, and the trace control row also lost on the candidate worker.

Commands:
- Baseline: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a RCH_WORKER=hz2 RCH_WORKERS=hz2 rch exec -- cargo bench -p fnp-python --bench criterion_python_surface -- python_einsum_boundary --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`
- Focused conformance: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-python einsum -- --nocapture`
- Candidate: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a RCH_WORKER=hz1 RCH_WORKERS=hz1 rch exec -- cargo bench -p fnp-python --bench criterion_python_surface -- python_einsum_boundary --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`

Baseline on current origin/main:

| Workload | Worker reported by RCH | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---:|---:|---|
| `fnp_einsum_trace_f64_4000` | `hz1` | 32,125 ns | 40,179 ns | 0.800x, 1.25x faster | Win |
| `fnp_einsum_diag_f64_4000` | `hz1` | 10,466 ns | 2,652 ns | 3.95x slower | Loss |
| `fnp_einsum_reduce_all_f64_1000` | `hz1` | 187,003 ns | 195,641 ns | 0.956x, 1.05x faster | Win |
| `fnp_einsum_reduce_rows_f64_1000` | `hz1` | 183,629 ns | 195,289 ns | 0.940x, 1.06x faster | Win |
| `fnp_einsum_reduce_cols_f64_1000` | `hz1` | 220,197 ns | 546,982 ns | 0.403x, 2.48x faster | Win |

Candidate with pre-policy shortcut:

| Workload | Worker reported by RCH | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---:|---:|---|
| `fnp_einsum_trace_f64_4000` | `hz2` | 15,981 ns | 5,163 ns | 3.10x slower | Loss |
| `fnp_einsum_diag_f64_4000` | `hz2` | 2,902 ns | 974 ns | 2.98x slower | Loss |
| `fnp_einsum_reduce_all_f64_1000` | `hz2` | 115,773 ns | 118,392 ns | 0.978x, 1.02x faster | Win |
| `fnp_einsum_reduce_rows_f64_1000` | `hz2` | 108,724 ns | 114,108 ns | 0.953x, 1.05x faster | Win |
| `fnp_einsum_reduce_cols_f64_1000` | `hz2` | 129,175 ns | 311,932 ns | 0.414x, 2.41x faster | Win |

Scorecard:
- Baseline vs NumPy: win/loss/neutral = 4/1/0.
- Candidate vs NumPy: win/loss/neutral = 3/2/0.
- Target row remained a NumPy loss: `fnp_einsum_diag_f64_4000` was 2.98x slower than NumPy on the candidate run.
- Cross-worker old-to-new movement is routing evidence only; RCH ignored the requested worker pin in both runs (`hz1` baseline, `hz2` candidate).
- Source decision: reverted. No Rust source from this candidate is kept.

Conformance:
- `cargo test -p fnp-python einsum -- --nocapture` passed on RCH with the candidate hunk before revert.
- Covered inline einsum tests, 28 `conformance_einsum` tests including diagonal/trace golden cases, and metamorphic einsum tests.

Retry predicate:
- Do not retry a wrapper-level pre-policy call into the existing diagonal helper by itself.
- A deeper retry must remove or avoid the remaining Python method dispatch / view construction overhead for `ii->i`, preserve NumPy writable-view semantics, and beat NumPy's roughly 1 us diagonal-view row in this same harness.
