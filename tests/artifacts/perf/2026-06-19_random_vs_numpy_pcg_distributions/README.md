# 2026-06-19 PCG Distribution Head-to-Head

Scope: gauntlet verification for `franken_numpy-ixs5y.250` and
`franken_numpy-ixs5y.253`.

Reference: `/usr/bin/python3`, NumPy 2.4.3, using
`np.random.Generator(np.random.PCG64(42))`.

Subject: direct Rust `fnp-random` Criterion rows using `Generator::gumbel` and
`Generator::laplace` from a PCG64 generator seeded with `SeedSequence([42])`.

Remote worker:
- Benchmarks: `ovh-a`
- Target dir requested:
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`
- RCH worker-scoped target observed:
  `/data/projects/franken_numpy/.rch-target-ovh-a-pool-f4ecbc5a8032ed7eb8c61438ab6b2cc8`

Commands:
- `rch exec -- cargo check -p fnp-random --benches`
- `rch exec -- cargo bench -p fnp-random --bench random_vs_numpy -- gumbel --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`
- `rch exec -- cargo bench -p fnp-random --bench random_vs_numpy -- laplace --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`
- `rch exec -- cargo test -p fnp-random parallel_pcg_gumbel_matches_serial_stream_state -- --nocapture`
- `rch exec -- cargo test -p fnp-random parallel_pcg_laplace_matches_serial_stream_state -- --nocapture`
- `rch exec -- cargo test -p fnp-random gumbel_matches_live_numpy_oracle -- --nocapture`
- `rch exec -- cargo test -p fnp-random laplace_matches_live_numpy_oracle -- --nocapture`

Results:

| Bead | Workload | Size | FNP | NumPy | FNP/NumPy ratio | Speedup | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| `franken_numpy-ixs5y.250` | PCG64 gumbel | 100,000 | 248,006 ns | 1,489,338 ns | 0.167x | 6.01x | Keep |
| `franken_numpy-ixs5y.250` | PCG64 gumbel | 1,000,000 | 2,105,737 ns | 15,047,299 ns | 0.140x | 7.15x | Keep |
| `franken_numpy-ixs5y.253` | PCG64 laplace | 100,000 | 204,760 ns | 1,384,891 ns | 0.148x | 6.76x | Keep |
| `franken_numpy-ixs5y.253` | PCG64 laplace | 1,000,000 | 1,599,666 ns | 13,871,270 ns | 0.115x | 8.67x | Keep |

No revert was required: every measured row beat NumPy by more than 5x and the
targeted stream-state and live-NumPy oracle checks passed.
