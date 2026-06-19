# franken_numpy-ixs5y.261 PCG bytes scorecard

Decision worker: `vmi1293453`

Control: `origin/main` at `3da8ac35`, worktree `/data/projects/.scratch/franken_numpy-cod-a-baseline-20260619-0518`

Candidate: direct PCG final `Vec<u8>` fill in `/data/projects/.scratch/franken_numpy-cod-a-20260619-0505`

## Same-Worker A/B

| Workload | Old FNP | New FNP | New/old | Candidate NumPy | Candidate FNP/NumPy | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `vs_numpy_pcg64_bytes/franken_bytes/100000` | 186,767 ns | 184,751 ns | 0.989x | 425,145 ns | 0.435x | Neutral keep |
| `vs_numpy_pcg64_bytes/franken_bytes/1000000` | 1,683,609 ns | 1,029,616 ns | 0.611x | 3,420,805 ns | 0.301x | Keep |

Score: 1 win, 0 losses, 1 neutral versus old FNP; 2 wins, 0 losses versus NumPy.

Correctness gates:

- `cargo test -p fnp-random bytes_large_calls_match_serial_uint32_stream_state -- --nocapture`: passed.
- `cargo test -p fnp-random bytes_match_live_numpy_oracle_when_available -- --nocapture`: passed.

Retained raw transcripts:

- `control_origin_main_vmi1293453.txt`
- `candidate_threshold_direct_fill_vmi1149989.txt` (attempted pin in filename; transcript selected `vmi1293453`)
- `baseline_main_vs_numpy_pcg64_bytes.txt`
- `candidate_direct_fill_vs_numpy_pcg64_bytes.txt`
