# fnp-random uint8 full-range byte-fill scorecard

Run identity:
- Date: 2026-06-19.
- Agent: YellowElk / cod-a.
- Bead: `franken_numpy-ixs5y.264`.
- Parent: `franken_numpy-ixs5y`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Subject API: `Generator::integers_u8_shaped(0, 256, size, endpoint=false)`.
- NumPy reference: `np.random.Generator(np.random.PCG64(42)).integers(0, 256, size=size, dtype=np.uint8)`.
- Coordination note: Agent Mail registration/reservations were unavailable because the local Agent Mail SQLite database reported `database disk image is malformed`; work was isolated in a clean detached scratch worktree.

Optimization hypothesis:
- The old full-range `uint8` path still routed through the narrow bounded-integer loop. For `rng == u8::MAX`, rejection sampling is unnecessary and NumPy's stream semantics reduce to raw byte consumption plus wrapping offset.
- Kept lever: a full-range byte stream fast path for `u8` and `i8`, with the large PCG-family case delegated to the existing direct final-buffer `bytes` fill and sub-threshold calls expanded four bytes per `next_uint32`.
- Graveyard mapping: Vectorized Execution / morsel-style final-buffer fill for the large path, plus the "constants kill you" rule for the small row where avoiding the generic bounded loop mattered more than parallelism.

Head-to-head rows:

| Stage | Worker | Workload | FrankenNumPy | NumPy | FNP/NumPy | Verdict |
|---|---|---:|---:|---:|---:|---|
| Baseline scalar bounded `uint8` | `hz2` | 100k | 329,987 ns | 105,292 ns | 3.13x slower | Loss |
| Baseline scalar bounded `uint8` | `hz2` | 1M | 3,241,197 ns | 788,860 ns | 4.11x slower | Loss |
| Candidate A: direct `bytes` reuse only | `hz2` | 100k | 106,506 ns | 101,368 ns | 1.05x slower | Neutral/loss, superseded |
| Candidate A: direct `bytes` reuse only | `hz2` | 1M | 286,011 ns | 757,753 ns | 0.377x, 2.65x faster | Win, superseded |
| Candidate B kept: manual sub-threshold + direct large byte fill | `vmi1149989` | 100k | 104,370 ns | 127,730 ns | 0.817x, 1.22x faster | Keep |
| Candidate B kept: manual sub-threshold + direct large byte fill | `vmi1149989` | 1M | 725,711 ns | 1,155,285 ns | 0.628x, 1.59x faster | Keep |
| Candidate B supplemental long run | `vmi1149989` | 100k | 88,959 ns | 118,758 ns | 0.749x, 1.34x faster | Confirmation |
| Candidate B supplemental long run | `vmi1149989` | 1M | 432,705 ns | 1,092,147 ns | 0.396x, 2.52x faster | Confirmation |

Scorecard:
- Baseline vs NumPy: win/loss/neutral = 0/2/0.
- Kept final vs NumPy, decision rows: win/loss/neutral = 2/0/0.
- Rejected/superseded candidate rows: win/loss/neutral = 1/1/0, with the 100k row not good enough to keep.
- Same-run final head-to-head: both decision rows beat NumPy on the same remote worker process running the Criterion benchmark and embedded Python timing.

Validation:
- `rch exec -- cargo bench -p fnp-random --bench random_vs_numpy -- vs_numpy_pcg64_uint8_full_range --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher` passed on `vmi1149989`.
- `rch exec -- cargo test -p fnp-random full_range_byte_integers_match_scalar_narrow_stream_and_state -- --nocapture` passed on `hz2`.
- `rch exec -- cargo test -p fnp-random narrow_width_integers_match_live_numpy_oracle_when_available -- --nocapture` passed on `hz1`; first attempt on `ovh-b` failed before repo code with `zerocopy` build-script `SIGILL`.
- `rch exec -- cargo check -p fnp-random --all-targets` passed on `vmi1149989`.
- `rch exec -- cargo clippy -p fnp-random --all-targets -- -D warnings` passed on `hz2`.
- `rch exec -- cargo build -p fnp-random --release` passed on `hz2`.
- `git diff --check` passed.
- `cargo fmt --check -p fnp-random` reports broad pre-existing rustfmt drift in `crates/fnp-random/src/lib.rs`; this perf commit did not absorb unrelated formatting churn.
- `ubs crates/fnp-random/src/lib.rs crates/fnp-random/benches/random_vs_numpy.rs docs/NEGATIVE_EVIDENCE.md tests/artifacts/perf/2026-06-19_random_uint8_full_range_byte_fill/scorecard.md` exited 1 after scanning the two Rust files, reporting the crate's broad existing inventory (66 critical, 2141 warnings, 659 info). Sampled findings were pre-existing `unwrap`/assert/direct-index/security-heuristic inventory outside the new full-range byte fast path.

Retry predicate:
- Do not retry generic bounded-loop tweaks for full-range byte integers.
- Only revisit this path if a same-run NumPy comparison regresses on both 100k and 1M rows, or if a future direct fill can prove better final-buffer throughput while preserving the scalar byte stream and post-call RNG state checks.
