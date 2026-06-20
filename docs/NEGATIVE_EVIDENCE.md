# FrankenNumPy Negative-Evidence Ledger

This ledger is append-only evidence for performance hypotheses. It records wins,
losses, neutral results, noisy discarded measurements, and retry predicates so
dead ends are not rediscovered as fresh ideas.

## 2026-06-20 - BOLD-VERIFY No-Ship: fnp-python pre-policy f64 einsum diagonal shortcut

Artifact directory: `tests/artifacts/perf/2026-06-20_python_einsum_diag_pre_policy/`

Run identity:
- Bead: `franken_numpy-ixs5y.269`.
- Agent: `BlackThrush` / `cod-a`.
- Subject API: Python-boundary `fnp.einsum` through `criterion_python_surface`.
- Oracle/reference: NumPy inside the same Criterion harness process.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Decision: no-ship; source hunk reverted.

Lever:
- Tried routing f64 single-operand diagonal/trace einsum forms through the existing zero-copy diagonal fast path before wrapper dtype-policy work.
- Alien-graveyard mapping: constants-kill-you specialization plus zero-copy/view-preserving layout reuse.
- Failure mode: the diagonal target still lost to NumPy after the shortcut, and the trace control row also lost on the candidate worker.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a RCH_WORKER=hz2 RCH_WORKERS=hz2 rch exec -- cargo bench -p fnp-python --bench criterion_python_surface -- python_einsum_boundary --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-python einsum -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a RCH_WORKER=hz1 RCH_WORKERS=hz1 rch exec -- cargo bench -p fnp-python --bench criterion_python_surface -- python_einsum_boundary --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`

| Run | Workload | Worker reported by RCH | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---|---:|---:|---:|---|
| Origin/main baseline | `fnp_einsum_trace_f64_4000` | `hz1` | 32,125 ns | 40,179 ns | 0.800x, 1.25x faster | Win |
| Origin/main baseline | `fnp_einsum_diag_f64_4000` | `hz1` | 10,466 ns | 2,652 ns | 3.95x slower | Open gap |
| Origin/main baseline | `fnp_einsum_reduce_all_f64_1000` | `hz1` | 187,003 ns | 195,641 ns | 0.956x, 1.05x faster | Win |
| Origin/main baseline | `fnp_einsum_reduce_rows_f64_1000` | `hz1` | 183,629 ns | 195,289 ns | 0.940x, 1.06x faster | Win |
| Origin/main baseline | `fnp_einsum_reduce_cols_f64_1000` | `hz1` | 220,197 ns | 546,982 ns | 0.403x, 2.48x faster | Win |
| Candidate shortcut | `fnp_einsum_trace_f64_4000` | `hz2` | 15,981 ns | 5,163 ns | 3.10x slower | No-ship loss |
| Candidate shortcut | `fnp_einsum_diag_f64_4000` | `hz2` | 2,902 ns | 974 ns | 2.98x slower | No-ship loss |
| Candidate shortcut | `fnp_einsum_reduce_all_f64_1000` | `hz2` | 115,773 ns | 118,392 ns | 0.978x, 1.02x faster | Win |
| Candidate shortcut | `fnp_einsum_reduce_rows_f64_1000` | `hz2` | 108,724 ns | 114,108 ns | 0.953x, 1.05x faster | Win |
| Candidate shortcut | `fnp_einsum_reduce_cols_f64_1000` | `hz2` | 129,175 ns | 311,932 ns | 0.414x, 2.41x faster | Win |

Scorecard:
- Baseline vs NumPy: win/loss/neutral = 4/1/0.
- Candidate vs NumPy: win/loss/neutral = 3/2/0.
- Candidate target row remained a NumPy loss: `fnp_einsum_diag_f64_4000` was 2.98x slower than NumPy.
- Cross-worker old-to-new movement is routing evidence only; RCH ignored the requested worker pin in both runs (`hz1` baseline, `hz2` candidate).
- Source decision: reverted. No Rust source from this candidate is kept.

Validation notes:
- Focused `cargo test -p fnp-python einsum -- --nocapture` passed with the candidate hunk before revert.
- Covered inline einsum tests, 28 `conformance_einsum` tests including diagonal/trace golden cases, and metamorphic einsum tests.
- Retry predicate: do not retry a wrapper-level pre-policy call into the existing diagonal helper by itself. A deeper retry must remove or avoid the remaining Python method dispatch / view construction overhead for `ii->i`, preserve NumPy writable-view semantics, and beat NumPy's roughly 1 us diagonal-view row in this same harness.

## 2026-06-20 - BOLD-VERIFY Keep: fnp-python sorted f32 histogram edge-pointer

Artifact directory: `tests/artifacts/perf/2026-06-20_python_histogram_f32_sorted_edges/`

Run identity:
- Bead: `franken_numpy-ixs5y.268`.
- Agent: `BlackThrush` / `cod-a`.
- Subject API: Python-boundary `fnp.histogram` through `criterion_python_surface`.
- Oracle/reference: NumPy inside the same Criterion harness process.
- Decision worker: `hz2`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.

Lever:
- Detect monotone nondecreasing `float32` histogram inputs during the existing finite min/max pass.
- Classify monotone data with a streaming pointer over the existing `float32` edge array, reducing the sorted case to `O(n + bins)` and removing per-element affine division.
- Preserve fallback, strict edge validation, float32 edge construction, and the unsorted scalar classifier.
- Alien-graveyard mapping: sorted-stream cursoring / merge-path style specialization plus constants-kill-you removal of hot scalar division when input order gives a stronger invariant.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo bench -p fnp-python --bench criterion_python_surface -- 'python_histogram_boundary|python_setops_boundary|python_statistics_boundary|python_einsum_boundary|python_linalg_boundary|python_char_ascii_boundary' --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a RCH_WORKER=hz2 RCH_WORKERS=hz2 rch exec -- cargo bench -p fnp-python --bench criterion_python_surface -- python_histogram_boundary --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a RCH_WORKER=hz2 RCH_WORKERS=hz2 rch exec -- cargo test -p fnp-python histogram_matches_numpy_across_bins_range_density_weights_and_empty -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo check -p fnp-python --all-targets`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo build --release -p fnp-python`
- `git diff --check`

| Bead | Lever | Workload | Worker | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.266` | Origin/main routing baseline | `histogram_i64_100k_50` | `hz2` | 599,955 ns | 861,321 ns | 0.697x, 1.44x faster | Baseline win |
| `franken_numpy-ixs5y.266` | Origin/main routing baseline | `histogram_f32_100k_50` | `hz2` | 668,179 ns | 613,046 ns | 1.09x slower | Open gap |
| `franken_numpy-ixs5y.266` | Raise parallel threshold | `histogram_i64_100k_50` | `hz2` | 845,866 ns | 867,590 ns | 0.975x, 1.03x faster | No-ship: i64 margin collapsed |
| `franken_numpy-ixs5y.266` | Raise parallel threshold | `histogram_f32_100k_50` | `hz2` | 673,497 ns | 590,563 ns | 1.14x slower | No-ship: f32 still lost |
| `franken_numpy-ixs5y.267` | Local count accumulator | `histogram_i64_100k_50` | `hz2` | 878,361 ns | 830,049 ns | 1.06x slower | No-ship |
| `franken_numpy-ixs5y.267` | Local count accumulator | `histogram_f32_100k_50` | `hz2` | 686,940 ns | 608,869 ns | 1.13x slower | No-ship |
| `franken_numpy-ixs5y.268` | Sorted edge-pointer count | `histogram_i64_100k_50` | `hz2` | 651,730 ns | 840,532 ns | 0.775x, 1.29x faster | Keep |
| `franken_numpy-ixs5y.268` | Sorted edge-pointer count | `histogram_f32_100k_50` | `hz2` | 449,882 ns | 584,574 ns | 0.770x, 1.30x faster | Keep |

Scorecard:
- Routing baseline vs NumPy on histogram rows: win/loss/neutral = 1/1/0.
- Threshold candidate vs NumPy: win/loss/neutral = 1/1/0, rejected because f32 still lost and i64 regressed vs origin baseline.
- Local-count candidate vs NumPy: win/loss/neutral = 0/2/0, rejected.
- Sorted edge-pointer candidate vs NumPy: win/loss/neutral = 2/0/0, kept.
- Primary targeted gap moved from 1.09x slower than NumPy to 0.770x of NumPy time; FrankenNumPy f32 histogram old-to-new improved 668,179 ns to 449,882 ns (0.673x, 1.49x faster).

Validation notes:
- Focused histogram parity test passed, including a new sorted `float32` 100k, 50-bin case.
- Supplemental `cargo test -p fnp-python` cleared 531 inline tests plus early conformance shards, then failed outside this path in `conformance_argwhere::argwhere_python_container_surfaces_match_numpy` because the NumPy oracle script emitted an `IndentationError`.
- `cargo check -p fnp-python --all-targets` passed on RCH with the crate's pre-existing three dead-code warnings.
- `cargo build --release -p fnp-python` passed on RCH worker `vmi1149989` with the same three warnings.
- `git diff --check` passed.
- `ubs crates/fnp-python/src/lib.rs` completed in 198s and exited 1 with broad file-wide inventory (`473` critical heuristic findings, `3661` warnings, `4554` info); no hunk-local histogram finding was identified.
- `cargo fmt -p fnp-python -- --check` remains blocked by broad pre-existing rustfmt drift in `fnp-python`, outside this perf hunk.
- `cargo clippy -p fnp-python --all-targets -- -D warnings` remains blocked by broad pre-existing fnp-python lint debt, outside this histogram path.
- Retry predicate: do not retry the parallel-threshold or local-count-only variants for this f32 histogram row. Deeper retries should exploit data order, edge layout, or a broader zero-copy histogram primitive and must beat 449,882 ns on the same head-to-head harness.

## 2026-06-20 - BOLD-VERIFY Keep: fnp-random small PCG bytes direct append fill

Artifact directory: `tests/artifacts/perf/2026-06-20_random_bytes_small_direct_append/`

Run identity:
- Bead: `franken_numpy-ixs5y.265`.
- Agent: `BlackThrush` / `cod-a`.
- Subject API: direct Rust `fnp-random` `Generator::bytes(length)`.
- Oracle/reference: NumPy `np.random.Generator(np.random.PCG64(42)).bytes(length)` inside the Criterion benchmark harness.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.

Lever:
- For sub-threshold PCG byte requests, append directly into the final byte vector from `next_u64` words.
- Preserve the exact `next_uint32` low/high half-buffer contract by consuming pending `u32_buf` first and buffering the high half only when the final direct append consumed a low half.
- This is not the rejected `.257` intermediate `Vec<u64>` transcode; no intermediate word vector is allocated.
- Alien-graveyard mapping: final-buffer/vectorized execution under a constants-kill-you threshold, with an artifact-level RNG state invariant as the proof obligation.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo bench -p fnp-random --bench random_vs_numpy -- vs_numpy_pcg64_bytes --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random bytes_large_calls_match_serial_uint32_stream_state -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random bytes_match_live_numpy_oracle_when_available -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo check -p fnp-random --all-targets`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo clippy -p fnp-random --all-targets -- -D warnings`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo build --release -p fnp-random`
- `git diff --check`

| Bead | Lever | Workload | Worker | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Old-to-new ratio | Verdict |
|---|---|---:|---|---|---:|---:|---:|---:|---|
| `franken_numpy-ixs5y.265` | Origin/main baseline | 100k bytes | `ovh-a` | `scorecard.md` | 87,044 ns | 94,154 ns | 0.925x, 1.08x faster | - | Baseline |
| `franken_numpy-ixs5y.265` | Direct final-Vec append | 100k bytes | `ovh-a` | `scorecard.md` | 32,920 ns | 47,212 ns | 0.697x, 1.43x faster | 0.378x, 2.64x faster | Keep |
| `franken_numpy-ixs5y.265` | Origin/main baseline | 1M bytes | `ovh-a` | `scorecard.md` | 154,618 ns | 429,977 ns | 0.360x, 2.78x faster | - | Baseline |
| `franken_numpy-ixs5y.265` | Direct final-Vec append | 1M bytes | `ovh-a` | `scorecard.md` | 122,926 ns | 427,988 ns | 0.287x, 3.48x faster | 0.795x, 1.26x faster | Keep |
| `franken_numpy-ixs5y.265` | Final source supplemental | 100k bytes | `vmi1153651` | `scorecard.md` | 85,242 ns | 465,151 ns | 0.183x, 5.46x faster | - | Noisy confirmation |
| `franken_numpy-ixs5y.265` | Final source supplemental | 1M bytes | `vmi1153651` | `scorecard.md` | 2,410,257 ns | 4,857,309 ns | 0.496x, 2.02x faster | - | Noisy confirmation |

Scorecard:
- Same-worker old-to-new: win/loss/neutral = 2/0/0.
- Candidate vs NumPy decisive rows: win/loss/neutral = 2/0/0.
- Candidate vs NumPy including supplemental rows: win/loss/neutral = 8/0/0.
- The `hz1` fresh-origin control reproduced the 100k loss class at 131,543 ns FNP vs 73,649 ns NumPy; the kept same-worker candidate moved the row to a win.

Validation notes:
- Focused stream-state and live NumPy oracle tests passed.
- Full `cargo test -p fnp-random` passed: 431 unit tests, 12 golden tests, 16 metamorphic tests.
- `cargo check -p fnp-random --all-targets`, `cargo clippy -p fnp-random --all-targets -- -D warnings`, `cargo build --release -p fnp-random`, and `git diff --check` passed.
- `cargo fmt --check` and `cargo fmt -p fnp-random --check` remain blocked by pre-existing broad rustfmt drift outside this perf hunk.
- Retry predicate: do not retry intermediate word-vector transcodes for PCG bytes. Retry only with a same-worker candidate that preserves the half-buffer invariant and beats this direct append path at both 100k and 1M.

## 2026-06-19 - BOLD-VERIFY Keep: fnp-random full-range uint8 integers byte stream

Artifact directory: `tests/artifacts/perf/2026-06-19_random_uint8_full_range_byte_fill/`

Run identity:
- Bead: `franken_numpy-ixs5y.264`.
- Agent: `YellowElk` / `cod-a`.
- Subject API: direct Rust `fnp-random` `Generator::integers_u8_shaped(0, 256, Some(&[size]), false)`.
- Oracle/reference: NumPy `np.random.Generator(np.random.PCG64(42)).integers(0, 256, size=size, dtype=np.uint8)` inside the Criterion benchmark harness.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Coordination note: Agent Mail registration/reservation failed before edits because the local Agent Mail SQLite DB reported `database disk image is malformed`; work was isolated in a clean detached scratch worktree and avoided cod-b-owned `fnp-ufunc` / `fnp-linalg` surfaces.

Lever:
- Old path sent full-range byte integers through the buffered bounded-Lemire helper even though `rng == u8::MAX` has no rejection. The kept path emits the raw byte stream directly and applies the wrapping offset, using the existing direct PCG final-buffer byte fill for large arrays and a four-byte-per-`next_uint32` scalar writer below the direct-fill threshold.
- Alien-graveyard mapping: Vectorized Execution / morsel-style final-buffer fill for the large PCG path, plus "constants kill you" for the 100k row where eliminating generic bounded-loop overhead mattered more than adding parallelism.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo bench -p fnp-random --bench random_vs_numpy -- vs_numpy_pcg64_uint8_full_range --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random full_range_byte_integers_match_scalar_narrow_stream_and_state -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random narrow_width_integers_match_live_numpy_oracle_when_available -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo check -p fnp-random --all-targets`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo clippy -p fnp-random --all-targets -- -D warnings`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo build -p fnp-random --release`
- `git diff --check`

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.264` | Baseline scalar bounded `uint8` loop | 100k, `hz2` | `scorecard.md` | 329,987 ns | 105,292 ns | 3.13x slower | Open gap confirmed |
| `franken_numpy-ixs5y.264` | Baseline scalar bounded `uint8` loop | 1M, `hz2` | `scorecard.md` | 3,241,197 ns | 788,860 ns | 4.11x slower | Open gap confirmed |
| `franken_numpy-ixs5y.264` | Candidate A, direct `bytes` reuse only | 100k, `hz2` | `scorecard.md` | 106,506 ns | 101,368 ns | 1.05x slower | Superseded |
| `franken_numpy-ixs5y.264` | Candidate A, direct `bytes` reuse only | 1M, `hz2` | `scorecard.md` | 286,011 ns | 757,753 ns | 0.377x, 2.65x faster | Superseded |
| `franken_numpy-ixs5y.264` | Candidate B kept, manual sub-threshold + direct large byte fill | 100k, `vmi1149989` | `scorecard.md` | 104,370 ns | 127,730 ns | 0.817x, 1.22x faster | Keep |
| `franken_numpy-ixs5y.264` | Candidate B kept, manual sub-threshold + direct large byte fill | 1M, `vmi1149989` | `scorecard.md` | 725,711 ns | 1,155,285 ns | 0.628x, 1.59x faster | Keep |
| `franken_numpy-ixs5y.264` | Candidate B supplemental long run | 100k, `vmi1149989` | `scorecard.md` | 88,959 ns | 118,758 ns | 0.749x, 1.34x faster | Confirmation |
| `franken_numpy-ixs5y.264` | Candidate B supplemental long run | 1M, `vmi1149989` | `scorecard.md` | 432,705 ns | 1,092,147 ns | 0.396x, 2.52x faster | Confirmation |

Scorecard:
- Baseline vs NumPy: win/loss/neutral = 0/2/0.
- Kept final vs NumPy decision rows: win/loss/neutral = 2/0/0.
- Rejected/superseded candidate rows: win/loss/neutral = 1/1/0.
- The final keep is based on same-run head-to-head Criterion rows where the Rust and embedded Python NumPy timings ran on the same remote worker process. Old-to-new absolute speedup is not used as a same-worker decision because RCH did not preserve a single worker across every exploratory run.

Validation notes:
- New scalar-stream/state guard passed for PCG64 and PCG64DXSM.
- Existing live NumPy narrow-integer oracle shard passed on rerun. The first attempt selected `ovh-b` and failed before repository code with `zerocopy` build-script `SIGILL`; this is recorded as worker infra noise, not repo evidence.
- `cargo check -p fnp-random --all-targets`, `cargo clippy -p fnp-random --all-targets -- -D warnings`, `cargo build -p fnp-random --release`, and `git diff --check` passed.
- `cargo fmt --check -p fnp-random` still reports broad pre-existing rustfmt drift in unrelated `crates/fnp-random/src/lib.rs` sections; this commit keeps that out of the perf proof.
- UBS on the changed-file set exited 1 after scanning the two Rust source files, with the crate's broad existing inventory (66 critical, 2141 warnings, 659 info); sampled findings were pre-existing `unwrap`/assert/direct-index/security-heuristic inventory outside the new full-range byte fast path.
- Retry predicate: do not revisit generic bounded-loop tweaks for full-range byte integers. Retry only with same-run NumPy rows that preserve the scalar byte stream and beat the kept path at both 100k and 1M.

## 2026-06-19 - BOLD-VERIFY Keep: FNP compress direct bool-mask decode vs NumPy

Artifact directory: `tests/artifacts/perf/2026-06-19_ufunc_compress_simd_cod_a/`

Run identity:
- Bead: `franken_numpy-ixs5y.263`.
- Parent gap: `.249` left serial `compress(condition, axis=None)` slower than
  NumPy after reverting the per-chunk `Vec<Vec<f64>>` parallel gather.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::compress` Criterion row
  `compress_f64_bool_flat_sparse`.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7 using the same value and
  bool-mask formula.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Worker notes: baseline and early candidates ran under `rch` on `hz1`; the
  final remote candidate ran on `hz2`; direct SSH NumPy timing on `hz2` failed
  with `Permission denied (publickey,password)`, so same-host keep/reject uses
  the local FNP Criterion confirmation plus the local NumPy probe.

Commands:
- `rch exec -- cargo bench -p fnp-ufunc --bench elementwise compress_f64_bool_flat_sparse -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `rch exec -- cargo test -p fnp-ufunc compress_f64_bool_flat_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `rch exec -- cargo test -p fnp-ufunc compress -- --nocapture`
- `rch exec -- cargo check -p fnp-ufunc --all-targets`
- `rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings`
- Local same-host confirmation: `cargo bench -p fnp-ufunc --bench elementwise compress_f64_bool_flat_sparse -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- Python NumPy timing probe in `baseline_numpy_compress_rch.txt`; `rch exec`
  warned that arbitrary `python3 -` is non-compilation and did not provide a
  worker pin.

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.263` | Baseline serial index materialization + `take` | 100k, `hz1` FNP vs local NumPy | `baseline_fnp_compress_rch.txt`, `baseline_numpy_compress_rch.txt` | 113.207 us | 52.056 us | 2.18x slower | Open gap confirmed |
| `franken_numpy-ixs5y.263` | Baseline serial index materialization + `take` | 1M, `hz1` FNP vs local NumPy | `baseline_fnp_compress_rch.txt`, `baseline_numpy_compress_rch.txt` | 1.232777 ms | 503.993 us | 2.45x slower | Open gap confirmed |
| `franken_numpy-ixs5y.263` | Two-pass bool mask count/decode, exact allocation | 100k, `hz1` FNP vs local NumPy | `candidate_fnp_compress_rch.txt`, `baseline_numpy_compress_rch.txt` | 85.032 us | 52.056 us | 1.63x slower | Superseded |
| `franken_numpy-ixs5y.263` | Two-pass bool mask count/decode, exact allocation | 1M, `hz1` FNP vs local NumPy | `candidate_fnp_compress_rch.txt`, `baseline_numpy_compress_rch.txt` | 520.739 us | 503.993 us | 1.03x slower | Neutral, superseded |
| `franken_numpy-ixs5y.263` | Single-pass bool bitmask decode, full input capacity | 100k, `hz1` FNP vs local NumPy | `candidate_fnp_compress_single_pass_rch.txt`, `baseline_numpy_compress_rch.txt` | 52.650 us | 52.056 us | 1.01x slower | Neutral, superseded |
| `franken_numpy-ixs5y.263` | Single-pass bool bitmask decode, full input capacity | 1M, `hz1` FNP vs local NumPy | `candidate_fnp_compress_single_pass_rch.txt`, `baseline_numpy_compress_rch.txt` | 508.565 us | 503.993 us | 1.01x slower | Neutral, superseded |
| `franken_numpy-ixs5y.263` | Single-pass bool bitmask decode, quarter-capacity output | 100k same-host local | `candidate_fnp_compress_capacity_local.txt`, `baseline_numpy_compress_rch.txt` | 44.374 us | 52.056 us | 0.852x, 1.17x faster | Keep |
| `franken_numpy-ixs5y.263` | Single-pass bool bitmask decode, quarter-capacity output | 1M same-host local | `candidate_fnp_compress_capacity_local.txt`, `baseline_numpy_compress_rch.txt` | 410.823 us | 503.993 us | 0.815x, 1.23x faster | Keep |
| `franken_numpy-ixs5y.263` | Single-pass bool bitmask decode, quarter-capacity output | 100k remote routing, `hz2` | `candidate_fnp_compress_capacity_rch.txt`, `baseline_numpy_compress_rch.txt` | 33.082 us | 52.056 us | 0.635x, 1.57x faster | Routing confirmation |
| `franken_numpy-ixs5y.263` | Single-pass bool bitmask decode, quarter-capacity output | 1M remote routing, `hz2` | `candidate_fnp_compress_capacity_rch.txt`, `baseline_numpy_compress_rch.txt` | 339.188 us | 503.993 us | 0.673x, 1.49x faster | Routing confirmation |

Scorecard:
- Final same-host vs NumPy: win/loss/neutral = 2/0/0.
- Rejected/superseded candidates vs NumPy: win/loss/neutral = 0/3/1.
- Old-to-final local comparison against the prior `.249` post-revert local
  serial rows: 100k improved 90.800 us -> 44.374 us (2.05x faster); 1M
  improved 1.1369 ms -> 410.823 us (2.77x faster).

Notes:
- Kept path is not the rejected `.249` per-chunk parallel gather. It avoids
  `Vec<Vec<f64>>`, avoids building an index vector, preserves the existing
  `take` fallback for true bits beyond the array end, and is limited to
  sidecar-free F64 `axis=None`.
- The final lever maps to Vectorized Execution-style selection bitmasks over a
  cache-local flat buffer plus the "constants kill you" rule: the exact-count
  two-pass and full-capacity one-pass versions were neutral/losses, so only the
  measured quarter-capacity one-pass decoder was kept.
- Golden guard digest:
  `81276111fdbfe090ecd3c825cf1ecc3fb5c6601e318fbd5683b9dfe6877d550f`.
- Focused validation passed for `cargo test -p fnp-ufunc compress`,
  `cargo check -p fnp-ufunc --all-targets`, `cargo clippy -p fnp-ufunc
  --all-targets -- -D warnings`, and `git diff --check`.
- `cargo fmt --check -p fnp-ufunc` still reports pre-existing rustfmt drift in
  unrelated fnp-ufunc sections and bench rows; it is recorded in
  `cargo_fmt_check_fnp_ufunc.txt` and kept out of this commit.
- UBS on the changed-file set exited nonzero with broad pre-existing
  `fnp-ufunc` inventory (489 critical, 14639 warnings); sampled findings are
  outside the new compress helper/path and the full output is recorded in
  `ubs_changed_files.txt`.

## 2026-06-19 - Gauntlet Verify: FNP flatnonzero gather vs NumPy

Artifact directory: `tests/artifacts/perf/2026-06-19_ufunc_flatnonzero_vs_numpy/`

Run identity:
- Verification bead: `franken_numpy-ixs5y.260`.
- Original optimization bead: `franken_numpy-ixs5y.245`.
- Subject commit before revert: `68a5d002`.
- Final code: `.245` parallel F64 `flatnonzero` index-gather fast path removed; serial exact int64 sidecar export retained.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::flatnonzero` Criterion row.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7.
- Same-host decision machine: `thinkstation1` for both local FNP Criterion and local NumPy timing.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.

Commands:
- `rch exec -- cargo test -p fnp-ufunc flatnonzero_f64_parallel_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `cargo bench -p fnp-ufunc --bench elementwise flatnonzero_f64_sparse -- --sample-size 20 --warm-up-time 1 --measurement-time 3`
- Python NumPy timing script in `numpy_flatnonzero_local.txt` using the same sparse F64 formula, warmups, GC disabled, and per-sample inner loops.
- `rch exec -- cargo test -p fnp-ufunc flatnonzero_f64_matches_serial_reference_and_golden_sha256 -- --nocapture`

Decision ratios use the same-host local NumPy timing in
`numpy_flatnonzero_local.txt`. The remote Criterion run on `vmi1227854` is
retained as routing evidence only because `rch exec` does not offload arbitrary
Python commands, and direct SSH to the worker was denied.

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.245` | Parallel F64 `flatnonzero` per-chunk index gather candidate | 100k local candidate | `criterion_flatnonzero_local_candidate.txt`, `numpy_flatnonzero_local.txt` | 255.53 us | 239.794 us | 1.07x slower | Reverted |
| `franken_numpy-ixs5y.245` | Parallel F64 `flatnonzero` per-chunk index gather candidate | 1M local candidate | `criterion_flatnonzero_local_candidate.txt`, `numpy_flatnonzero_local.txt` | 703.16 us | 2512.132 us | 0.280x, 3.57x faster | Reverted despite win |
| `franken_numpy-ixs5y.245` | Final serial `flatnonzero` with exact int64 sidecar export | 100k post-revert | `criterion_flatnonzero_local_post_revert.txt`, `numpy_flatnonzero_local.txt` | 74.662 us | 239.794 us | 0.311x, 3.21x faster | Final code |
| `franken_numpy-ixs5y.245` | Final serial `flatnonzero` with exact int64 sidecar export | 1M post-revert | `criterion_flatnonzero_local_post_revert.txt`, `numpy_flatnonzero_local.txt` | 789.35 us | 2512.132 us | 0.314x, 3.18x faster | Final code |

Notes:
- The candidate passed the golden guard, but correctness was not enough: it lost
  to NumPy at 100k, regressed local Criterion history at 100k, and was only a
  modest 1.12x faster than the final serial path at 1M.
- The production parallel branch was removed. The final serial sidecar-export
  path beats NumPy on both measured rows and avoids the 100k regression.
- Final focused validation passed for the post-revert golden guard,
  `rch exec -- cargo check -p fnp-ufunc --all-targets`, `rch exec -- cargo
  clippy -p fnp-ufunc --all-targets -- -D warnings`, and `git diff --check`.
- `cargo fmt --check`, `cargo fmt -p fnp-ufunc -- --check`, and UBS still
  report broad pre-existing `fnp-ufunc` drift/inventory outside the touched
  flatnonzero lines.
- Retry condition: retry `flatnonzero` parallel index gather only with a design
  that avoids per-chunk `Vec<Vec<i64>>` allocation, proves same-host speed over
  NumPy and over the serial sidecar path at both 100k and 1M sparse rows, and
  keeps NumPy timing CV below 10% on the decision rows.

## 2026-06-19 - fnp-random PCG raw fill and bytes cluster

Artifact directory: `tests/artifacts/perf/2026-06-19_random_vs_numpy_pcg/`

Run identity:
- Random subject commit before measured commit: `e32d58ea`.
- Integration base before this commit: `70bae5da`; intervening changes were ufunc evidence/docs and did not touch `fnp-random`.
- Subject API: direct Rust `fnp-random` Criterion rows.
- Oracle/reference: NumPy `np.random.Generator(np.random.PCG64(42))`; local preflight observed NumPy 2.4.3 on `/usr/bin/python3`.
- Workers: `ovh-a` for pre-revert candidate run, `hz1` for final-code run.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo bench -p fnp-random --bench random_vs_numpy -- --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo check -p fnp-random --benches`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random parallel_pcg_fill_u64_matches_serial_stream_state -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random bytes_large_calls_match_serial_uint32_stream_state -- --nocapture`

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | Ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.255` | Parallel PCG64 `fill_u64` jump-ahead | 100k u64 final code, `hz1` | `criterion_random_vs_numpy_post_revert.txt` | 144,560 ns | 538,164 ns | 0.269x | Keep |
| `franken_numpy-ixs5y.255` | Parallel PCG64 `fill_u64` jump-ahead | 1M u64 final code, `hz1` | `criterion_random_vs_numpy_post_revert.txt` | 2,194,406 ns | 4,447,414 ns | 0.493x | Keep |
| `franken_numpy-ixs5y.257` | PCG bytes via u64 word-fill transcode | 100k bytes pre-revert, `ovh-a` | `criterion_random_vs_numpy_prerevert.txt` | 80,103 ns | 48,911 ns | 1.638x | Reverted |
| `franken_numpy-ixs5y.257` | PCG bytes via u64 word-fill transcode | 1M bytes pre-revert, `ovh-a` | `criterion_random_vs_numpy_prerevert.txt` | 850,688 ns | 428,300 ns | 1.986x | Reverted |
| `franken_numpy-ixs5y.257` | Current serial `Generator::bytes` after revert | 100k bytes final code, `hz1` | `criterion_random_vs_numpy_post_revert.txt` | 119,901 ns | 74,988 ns | 1.599x | Open gap |
| `franken_numpy-ixs5y.257` | Current serial `Generator::bytes` after revert | 1M bytes final code, `hz1` | `criterion_random_vs_numpy_post_revert.txt` | 1,214,954 ns | 982,093 ns | 1.237x | Open gap |

Notes:
- `.255` is kept because the final code remains faster than NumPy on both large raw-buffer workloads while preserving stream state.
- `.257` is rejected and the production word-fill path was removed. The u64-word transcode approach allocated/interpreted an intermediate word buffer and lost to NumPy bytes on both measured rows.
- Retry condition for `.257`: only revisit `Generator::bytes` if the candidate fills the final `Vec<u8>` directly from PCG state without an intermediate `Vec<u64>`, preserves the exact `next_uint32` half-buffer contract, and is remeasured head-to-head against NumPy on the same worker. Do not retry the removed `fill_u64(...).to_le_bytes()` transcode family.

## 2026-06-19 - fnp-random PCG bytes direct final-buffer fill

Artifact directory: `tests/artifacts/perf/2026-06-19_random_bytes_direct_fill/`

Run identity:
- Verification bead: `franken_numpy-ixs5y.261`.
- Subject API: direct Rust `fnp-random` `Generator::bytes` Criterion rows.
- Oracle/reference: NumPy `np.random.Generator(np.random.PCG64(42))` from the benchmark harness.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Same-worker A/B decision machine: `vmi1293453`.
- Control worktree: `/data/projects/.scratch/franken_numpy-cod-a-baseline-20260619-0518` at `origin/main` (`3da8ac35`).
- Candidate worktree: `/data/projects/.scratch/franken_numpy-cod-a-20260619-0505`.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random bytes_large_calls_match_serial_uint32_stream_state -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random bytes_match_live_numpy_oracle_when_available -- --nocapture`
- `RCH_WORKER=vmi1293453 RCH_WORKERS=vmi1293453 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo bench -p fnp-random --bench random_vs_numpy -- vs_numpy_pcg64_bytes --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`

Decision rows:

| Bead | Lever | Workload | Artifact | Old FNP | New FNP | FNP new/old | New NumPy | New FNP/NumPy | Verdict |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `franken_numpy-ixs5y.261` | Direct PCG final `Vec<u8>` fill; small rows keep old append loop | 100k bytes, `vmi1293453` | `control_origin_main_vmi1293453.txt`, `candidate_threshold_direct_fill_vmi1149989.txt` | 186,767 ns | 184,751 ns | 0.989x | 425,145 ns | 0.435x | Neutral keep |
| `franken_numpy-ixs5y.261` | Direct PCG final `Vec<u8>` fill via jump-ahead chunks | 1M bytes, `vmi1293453` | `control_origin_main_vmi1293453.txt`, `candidate_threshold_direct_fill_vmi1149989.txt` | 1,683,609 ns | 1,029,616 ns | 0.611x, 1.64x faster | 3,420,805 ns | 0.301x, 3.32x faster | Keep |

Scorecard:
- Win/loss/neutral versus old FNP: 1 win, 0 losses, 1 neutral.
- Win/loss versus NumPy for the kept candidate rows: 2 wins, 0 losses.
- The earlier `.257` post-revert open gap did not reproduce on today’s workers:
  `vmi1153651` baseline was already faster than NumPy at 100k and 1M, and the
  `vmi1293453` control was also faster than NumPy. Treat the old gap as
  worker-sensitive routing evidence, not a current production loss.

Notes:
- `candidate_threshold_direct_fill_vmi1149989.txt` is the same-worker
  candidate artifact used above; its filename records the attempted pin, while
  the RCH transcript inside shows the actual selected worker was `vmi1293453`.
- Kept code fills the final byte vector directly from PCG64/PCG64DXSM `u64`
  words and never materializes the rejected intermediate `Vec<u64>` transcode.
- The old serial append loop is still used below `PCG_BYTES_DIRECT_MIN_LEN` and
  for non-PCG bit generators. A first unthresholded candidate regressed the 100k
  row because safe Rust zero-initialized the final buffer; that artifact is
  retained in `candidate_direct_fill_vs_numpy_pcg64_bytes.txt`.
- Correctness gates passed for the large serial-stream state guard and live
  NumPy byte oracle. The large guard exercises PCG64 and PCG64DXSM, prebuffered
  and unbuffered `next_uint32` state, back-to-back byte calls, and post-call
  `next_uint32` continuation.
- Retry condition: do not retry the old word-fill transcode. Revisit bytes only
  if a future same-worker head-to-head shows the direct-fill row losing to NumPy,
  or if a broader random conformance gate finds a `next_uint32` half-buffer state
  mismatch.

## 2026-06-19 - fnp-random PCG gumbel/laplace distribution cluster

Artifact directory: `tests/artifacts/perf/2026-06-19_random_vs_numpy_pcg_distributions/`

Run identity:
- Subject commit before measured commit: `0442da80`.
- Subject API: direct Rust `fnp-random` Criterion rows.
- Oracle/reference: NumPy `np.random.Generator(np.random.PCG64(42))`; local preflight observed NumPy 2.4.3 on `/usr/bin/python3`.
- Worker: `ovh-a` for both benchmark filters and all targeted correctness tests.
- Target dir requested: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- RCH worker-scoped target observed: `/data/projects/franken_numpy/.rch-target-ovh-a-pool-f4ecbc5a8032ed7eb8c61438ab6b2cc8`.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo check -p fnp-random --benches`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo bench -p fnp-random --bench random_vs_numpy -- gumbel --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo bench -p fnp-random --bench random_vs_numpy -- laplace --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random parallel_pcg_gumbel_matches_serial_stream_state -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random parallel_pcg_laplace_matches_serial_stream_state -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random gumbel_matches_live_numpy_oracle -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random laplace_matches_live_numpy_oracle -- --nocapture`

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.250` | Parallel PCG64 gumbel inverse-CDF fill | 100k f64, `ovh-a` | `criterion_gumbel_vs_numpy.txt` | 248,006 ns | 1,489,338 ns | 0.167x | Keep |
| `franken_numpy-ixs5y.250` | Parallel PCG64 gumbel inverse-CDF fill | 1M f64, `ovh-a` | `criterion_gumbel_vs_numpy.txt` | 2,105,737 ns | 15,047,299 ns | 0.140x | Keep |
| `franken_numpy-ixs5y.253` | Parallel PCG64 laplace inverse-CDF fill | 100k f64, `ovh-a` | `criterion_laplace_vs_numpy.txt` | 204,760 ns | 1,384,891 ns | 0.148x | Keep |
| `franken_numpy-ixs5y.253` | Parallel PCG64 laplace inverse-CDF fill | 1M f64, `ovh-a` | `criterion_laplace_vs_numpy.txt` | 1,599,666 ns | 13,871,270 ns | 0.115x | Keep |

Notes:
- `.250` is kept because both gumbel rows beat NumPy by 6.01x and 7.15x while `parallel_pcg_gumbel_matches_serial_stream_state` and `gumbel_matches_live_numpy_oracle` passed.
- `.253` is kept because both laplace rows beat NumPy by 6.76x and 8.67x while `parallel_pcg_laplace_matches_serial_stream_state` and `laplace_matches_live_numpy_oracle` passed.
- No optimization was reverted in this distribution slice.
- Retry condition for `.250`: revisit only if a same-worker rerun shows the PCG64 gumbel median at or above NumPy's median, if a broader distribution gate exposes a stream-state mismatch, or if NumPy changes PCG64 gumbel semantics in a way that invalidates fixed one-uniform jump-ahead.
- Retry condition for `.253`: revisit only if a same-worker rerun shows the PCG64 laplace median at or above NumPy's median, if a broader distribution gate exposes a stream-state mismatch, or if NumPy changes PCG64 laplace semantics in a way that invalidates fixed one-uniform jump-ahead.

## Carried No-Retry Families

These remain excluded unless a new profile identifies a different primitive and the retry condition is explicit:

| Family | Status | Retry condition |
|---|---|---|
| SVD row/panel/finalization micro-levers | Rejected in prior gauntlet runs | Only retry through a deeper bidiagonal/full-to-band primitive with a fresh `svd_mxn_full/512` proof. |
| Inverse/TRSM broad `batch_solve` routing | Rejected in prior gauntlet runs | Only retry with a different algorithmic route and same-worker evidence beating NumPy. |
| Packed-GEMM tile-width retunes | Rejected in prior gauntlet runs | Only retry as shared packed-panel/RHS redesign, not `PACKED_NR` width tuning. |
| Variable-consumption random distributions | Rejected for jump-ahead parallelization | Only retry if a live NumPy oracle proves fixed-consumption semantics for that distribution. |

## 2026-06-19 - Gauntlet Verify: FNP ufunc data movement vs NumPy

Run identity:
- Subject commit: `e32d58ea` (`main`, mirrored to `master` before this verify pass).
- Subject API: direct Rust `fnp-ufunc` `UFuncArray` Criterion rows.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7.
- Worker: `thinkstation1` via `rch exec`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.
- Commands:
  - `cargo bench -p fnp-ufunc --bench elementwise delete_flat_f64_sparse_indices -- --sample-size 20 --warm-up-time 1 --measurement-time 3`
  - `cargo bench -p fnp-ufunc --bench elementwise insert_flat_f64_midpoint_many -- --sample-size 20 --warm-up-time 1 --measurement-time 3`
  - Python batched NumPy timing script with 41 samples, warmups, and per-sample inner loops.

Measurement caveat: these rows compare the optimized Rust `fnp-ufunc` API to
NumPy's Python API on equivalent flat array workloads. They are valid for the
`fnp-ufunc` data-movement cluster, not a full `fnp-python` boundary claim.

| Bead | Workload | Size | FNP Criterion median | NumPy batched median | FNP speed vs NumPy | NumPy CV | Verdict | Retry predicate |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `franken_numpy-ixs5y.256` | `delete_flat_f64_sparse_indices` | 100,000 | 48.657 us | 74.719 us | 1.54x faster | 4.33% | KEEP | Reopen only if same-worker Criterion median regresses above 72 us or a batched NumPy rerun beats the FNP upper CI bound. |
| `franken_numpy-ixs5y.256` | `delete_flat_f64_sparse_indices` | 1,000,000 | 659.58 us | 787.97 us | 1.19x faster | 5.08% | KEEP, borderline CV | Reopen if a low-CV same-worker rerun shows NumPy median <= FNP median, or if broader delete workloads show the sort/dedup span path below 1.05x. |
| `franken_numpy-ixs5y.258` | `insert_flat_f64_midpoint_many` | 100,000 | 17.695 us | 24.896 us | 1.41x faster | 3.60% | KEEP | Reopen only if same-worker Criterion median regresses above 24 us or NumPy batched median drops below FNP upper CI bound. |
| `franken_numpy-ixs5y.258` | `insert_flat_f64_midpoint_many` | 1,000,000 | 256.97 us | 445.04 us | 1.73x faster | 17.92% | KEEP, noisy NumPy allocation row | Reopen with allocator-isolated batching if future evidence puts NumPy p50 below 272 us; current NumPy minimum was still 333.31 us. |

Discarded / non-decision evidence:
- Raw unbatched Python timings were discarded for keep-gate decisions because CV
  was 15-71% on microsecond-scale operations. Those numbers may be useful only
  as a smoke check that the workload shape was valid, not as a pass/fail gate.
- An exact-filter test invocation ran zero inline tests because Cargo's exact
  test name did not match the module-qualified path. The subsequent substring
  filter ran and passed both targeted golden guards.

Conformance / correctness guard:
- `cargo test -p fnp-ufunc delete_flat_f64_span_copy_matches_hashset_reference_and_golden_sha256 -- --nocapture` passed: 1 test run.
- `cargo test -p fnp-ufunc insert_flat_f64_splice_matches_repeated_insert_and_golden_sha256 -- --nocapture` passed: 1 test run.
- `cargo check -p fnp-ufunc` passed after the test-only type inference fix.

Action:
- No optimization reverted. Both measured rows clear the head-to-head NumPy median
  gate and have targeted golden guards green.
- Do not retry the prior per-element `HashSet` scan for large flat F64 delete
  or repeated `Vec::insert` shifting for large flat F64 insert unless the retry
  predicates above fire.

## 2026-06-19 - Gauntlet Verify: FNP compress bool-mask vs NumPy

Artifact directory: `tests/artifacts/perf/2026-06-19_ufunc_selection_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.249`.
- Subject commit before revert: `0442da80` plus the local candidate guard fix.
- Final code: `.249` parallel compress fast path removed; serial `compress` path retained.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::compress` Criterion row.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7.
- Same-host decision worker: `thinkstation1` for both local FNP Criterion and local NumPy timing.
- Remote routing evidence: `vmi1149989` Criterion candidate run, not used as the keep/reject gate because the NumPy command could not be pinned to that worker.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.

Commands:
- `cargo test -p fnp-ufunc compress_f64_parallel_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `cargo bench -p fnp-ufunc --bench elementwise compress_f64_bool_flat_sparse -- --sample-size 20 --warm-up-time 1 --measurement-time 3`
- Python NumPy timing script in `numpy_compress_local.txt` using the same value and bool-mask formulas.
- `cargo test -p fnp-ufunc compress -- --nocapture`
- `cargo check -p fnp-ufunc`
- `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | Ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.249` | Parallel F64 `compress` bool-mask gather candidate | 100k local candidate | `criterion_compress_local.txt`, `numpy_compress_local.txt` | 472.85 us | 66.147 us | 7.15x slower | Reverted |
| `franken_numpy-ixs5y.249` | Parallel F64 `compress` bool-mask gather candidate | 1M local candidate | `criterion_compress_local.txt`, `numpy_compress_local.txt` | 1.0645 ms | 518.349 us | 2.05x slower | Reverted |
| `franken_numpy-ixs5y.249` | Final serial `compress` after revert | 100k post-revert | `criterion_compress_local_post_revert.txt`, `numpy_compress_local.txt` | 90.800 us | 66.147 us | 1.37x slower | Open gap |
| `franken_numpy-ixs5y.249` | Final serial `compress` after revert | 1M post-revert | `criterion_compress_local_post_revert.txt`, `numpy_compress_local.txt` | 1.1369 ms | 518.349 us | 2.19x slower | Open gap |

Notes:
- The first guard attempt failed because the test used `assert_eq!` on a selected slice containing `NaN`; after switching that edge assertion to bitwise comparison, the candidate guard passed.
- Passing correctness was not enough: same-host local Criterion showed the parallel candidate regressed the local Criterion history by +339.69% at 100k and +66.84% at 1M, and it lost badly to NumPy on both measured sizes.
- The production parallel fast path was removed. The remaining serial path is still slower than NumPy, but it is less bad at 100k and avoids keeping a regressing optimization.
- Final focused validation passed for `cargo test -p fnp-ufunc compress`, `cargo check -p fnp-ufunc`, and `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`; `cargo fmt --check` still reports broad workspace format drift outside this slice.
- Retry condition: retry `compress(condition, axis=None)` only if a new design avoids per-chunk `Vec<Vec<f64>>` allocation and proves same-host speed over NumPy on both 100k and 1M bool-mask rows with CV below 10%; do not retry this per-chunk parallel gather shape as a standalone patch.

## 2026-06-19 - Gauntlet Verify: FNP extract masked gather vs NumPy

Artifact directory: `tests/artifacts/perf/2026-06-19_ufunc_extract_vs_numpy/`

Run identity:
- Verification bead: `franken_numpy-ixs5y.259`.
- Original optimization bead: `franken_numpy-ixs5y.244`.
- Subject commit before revert: `298f05dd`.
- Final code: `.244` parallel F64 `extract` masked-gather fast path removed; serial `extract` path retained.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::extract` Criterion row.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7.
- Same-host decision machine: `thinkstation1` for both local FNP Criterion and local NumPy timing.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.

Commands:
- `rch exec -- cargo test -p fnp-ufunc extract_f64_parallel_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `cargo bench -p fnp-ufunc --bench elementwise extract_f64_masked -- --sample-size 20 --warm-up-time 1 --measurement-time 3`
- Python NumPy timing scripts in `numpy_extract_local.txt` and `numpy_extract_local_rerun.txt` using the same value and bool-mask formulas.
- `rch exec -- cargo test -p fnp-ufunc extract_f64_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `rch exec -- cargo test -p fnp-ufunc boolean_index_f64_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `rch exec -- cargo check -p fnp-ufunc --all-targets`
- `rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings`
- `cargo fmt --check`

Decision ratios use the longer `numpy_extract_local_rerun.txt` reference with
GC disabled.

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | Ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.244` | Parallel F64 `extract` per-chunk gather candidate | 100k local candidate | `criterion_extract_local.txt`, `numpy_extract_local_rerun.txt` | 275.46 us | 126.540 us | 2.18x slower | Reverted |
| `franken_numpy-ixs5y.244` | Parallel F64 `extract` per-chunk gather candidate | 1M local candidate | `criterion_extract_local.txt`, `numpy_extract_local_rerun.txt` | 668.54 us | 547.298 us | 1.22x slower | Reverted |
| `franken_numpy-ixs5y.244` | Final serial `extract` after revert | 100k post-revert | `criterion_extract_local_post_revert.txt`, `numpy_extract_local_rerun.txt` | 79.896 us | 126.540 us | 1.58x faster | Final code |
| `franken_numpy-ixs5y.244` | Final serial `extract` after revert | 1M post-revert | `criterion_extract_local_post_revert.txt`, `numpy_extract_local_rerun.txt` | 951.42 us | 547.298 us | 1.74x slower | Open gap |

Notes:
- The candidate passed the golden guard, but correctness was not enough: it lost
  to NumPy on both measured rows and was 3.45x slower than the final serial path
  at 100k.
- The candidate was faster than serial at 1M, but still 1.22x slower than NumPy
  and therefore did not clear the gauntlet's neutral/regression rule.
- Removing the parallel `extract` branch also removes the implicit parallel
  acceleration that `boolean_index` reached through `extract`; the boolean-index
  golden guard was rerun post-revert and passed. `boolean_index` remains a
  separate open benchmark target rather than an unmeasured keep.
- Final focused validation passed for the two golden guards, `cargo check -p
  fnp-ufunc --all-targets`, and `cargo clippy -p fnp-ufunc --all-targets -- -D
  warnings`.
- `cargo fmt --check` and package-scoped `cargo fmt -p fnp-ufunc -- --check`
  still report broad pre-existing format drift in untouched regions; the
  extract revert itself is compiled and clippy-clean.
- Retry condition: retry `extract(condition, arr)` only with a design that avoids
  per-chunk `Vec<Vec<f64>>` allocation, proves same-host speed over NumPy at both
  100k and 1M sparse-mask rows, keeps NumPy timing CV below 10%, and separately
  remeasures the `boolean_index_f64_masked_sparse` dependent workload.

## 2026-06-19 - BOLD-VERIFY: FNP extract SIMD mask decode keep

Artifact directories:
- `tests/artifacts/perf/2026-06-19_ufunc_extract_values_only_vs_numpy/`
- `tests/artifacts/perf/2026-06-19_ufunc_boolean_index_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.244`.
- Subject base before local edit: `f4cfc942`.
- Final code: F64/no-integer-sidecar `extract` path counts and decodes the f64
  bool mask with safe portable SIMD bitmasks, then pushes selected values in
  lane order. The integer-sidecar and non-F64 paths keep the existing
  source-index implementation.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::extract`; dependent
  `boolean_index` path reuses `extract`.
- Same-worker FNP decision worker: `hz2`.
- Same-host confirmation machine: `thinkstation1` for both local Criterion and
  NumPy reference.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.

Commands:
- `rch exec -- cargo test -p fnp-ufunc extract_f64_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `rch exec -- cargo test -p fnp-ufunc boolean_index_f64_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `rch exec -- cargo bench -p fnp-ufunc --bench elementwise extract_f64_masked -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `rch exec -- cargo bench -p fnp-ufunc --bench elementwise boolean_index_f64_masked_sparse -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `cargo bench -p fnp-ufunc --bench elementwise extract_f64_masked -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `cargo bench -p fnp-ufunc --bench elementwise boolean_index_f64_masked_sparse -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `rch exec -- cargo check -p fnp-ufunc --all-targets`
- `rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings`
- `cargo fmt -p fnp-ufunc -- --check`

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.244` | One-pass F64 no-sidecar sparse-capacity candidate | 100k extract, `ovh-a` routing | terminal transcript | 100.348 us | 54.443 us | 1.843x | Reverted |
| `franken_numpy-ixs5y.244` | One-pass F64 no-sidecar sparse-capacity candidate | 1M extract, `ovh-a` routing | terminal transcript | 1,171.660 us | 613.212 us | 1.911x | Reverted |
| `franken_numpy-ixs5y.244` | SIMD f64-mask count/decode | 100k extract, `hz2` | `criterion_extract_simd_keep.txt`, `numpy_extract_local_baseline.txt` | 44.666 us | 54.443 us | 0.820x | Keep |
| `franken_numpy-ixs5y.244` | SIMD f64-mask count/decode | 1M extract, `hz2` | `criterion_extract_simd_keep.txt`, `numpy_extract_local_baseline.txt` | 496.772 us | 613.212 us | 0.810x | Keep |
| `franken_numpy-ixs5y.244` | SIMD f64-mask count/decode | 100k extract, local confirmation | `criterion_extract_simd_keep.txt`, `numpy_extract_local_baseline.txt` | 42.327 us | 54.443 us | 0.777x | Keep |
| `franken_numpy-ixs5y.244` | SIMD f64-mask count/decode | 1M extract, local confirmation | `criterion_extract_simd_keep.txt`, `numpy_extract_local_baseline.txt` | 610.601 us | 613.212 us | 0.996x | Keep, borderline |
| `franken_numpy-ixs5y.244` | Dependent `boolean_index` through SIMD extract | 100k boolean index, `hz2` | `criterion_boolean_index_simd_keep.txt`, `numpy_boolean_index_local_baseline.txt` | 45.643 us | 87.115 us | 0.524x | Keep |
| `franken_numpy-ixs5y.244` | Dependent `boolean_index` through SIMD extract | 1M boolean index, `hz2` | `criterion_boolean_index_simd_keep.txt`, `numpy_boolean_index_local_baseline.txt` | 466.771 us | 976.900 us | 0.478x | Keep |
| `franken_numpy-ixs5y.244` | Dependent `boolean_index` through SIMD extract | 100k boolean index, local confirmation | `criterion_boolean_index_simd_keep.txt`, `numpy_boolean_index_local_baseline.txt` | 46.343 us | 87.115 us | 0.532x | Keep |
| `franken_numpy-ixs5y.244` | Dependent `boolean_index` through SIMD extract | 1M boolean index, local confirmation | `criterion_boolean_index_simd_keep.txt`, `numpy_boolean_index_local_baseline.txt` | 568.270 us | 976.900 us | 0.582x | Keep |

Notes:
- The scalar no-sidecar count/copy candidate improved over the original
  source-index path on one remote routing run but still lost to NumPy; it was
  not kept as the final lever.
- The one-pass sparse-capacity candidate removed the count pass but regressed
  badly, likely from realloc/capacity behavior plus scalar branch pressure; it
  was reverted before final validation.
- The final SIMD path preserves NumPy truthiness for this representation:
  `NaN != 0.0` is selected, `-0.0 == 0.0` is false, and bitmask lanes are pushed
  in ascending order to preserve output order.
- `cargo fmt -p fnp-ufunc -- --check` still reports broad pre-existing format
  drift in untouched benches, tests, imports, and polynomial/cross-product
  regions. The edited block was manually adjusted to the rustfmt import/order
  style shown for that block.
- A fresh local NumPy rerun after the keep was slower than the earlier NumPy
  reference, so the table uses the stricter earlier NumPy medians. The earlier
  NumPy extract 1M CV was high, making the local 1M extract confirmation a
  borderline keep despite clearing the median.
- Final focused validation passed for both golden guards, package check, and
  package clippy with `-D warnings`.
- Retry condition: reopen `.244` if a low-CV same-host NumPy rerun shows NumPy
  median at or below the local FNP median for 1M extract, if a broader extract
  density matrix shows dense-mask regressions from the SIMD path, or if the
  representation bridge grows a true compact bool mask storage path that can
  remove the current f64-mask memory traffic entirely.
- Follow-up bead `franken_numpy-ixs5y.262` adds independent cod-a same-worker
  verification of the same SIMD source body; after rebase it keeps the upstream
  source unchanged and records the extra proof bundle.

## 2026-06-19 - BOLD-VERIFY Keep: FNP extract SIMD mask decode vs NumPy

Artifact directory:
`tests/artifacts/perf/2026-06-19_ufunc_extract_simd_cod_a/`

Run identity:
- Verification bead: `franken_numpy-ixs5y.262`.
- Parent gap: the first `franken_numpy-ixs5y.244` verification left the serial
  1M `extract` row 1.74x slower than NumPy after reverting its per-chunk
  `Vec<Vec<f64>>` candidate.
- Baseline commit for the measured old FNP rows: `39bb1e78`.
- Rebased parent carrying the same SIMD source body: `0d3be5d0`.
- Final code after rebase: the upstream sidecar-free F64 `extract` SIMD
  mask-count and bitmask decode fast path is retained unchanged; `.262` adds
  independent cod-a verification artifacts and ledger evidence rather than an
  extra source delta.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::extract` Criterion row.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7.
- Same-worker decision machine: `hz2` for the FNP Criterion rows and the NumPy
  timing probes.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.

Commands:
- `rch exec -- cargo bench -p fnp-ufunc --bench elementwise extract_f64_masked -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `rch exec -- python3 -` using the bool-mask NumPy extract timing probe in
  `baseline_numpy_extract_bool_hz2.txt`.
- `rch exec -- cargo bench -p fnp-ufunc --bench elementwise boolean_index_f64_masked_sparse -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `rch exec -- python3 -` using the bool-mask NumPy boolean-index timing probe in
  `baseline_numpy_boolean_index_hz2.txt`.
- `rch exec -- cargo test -p fnp-ufunc extract_f64_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `rch exec -- cargo test -p fnp-ufunc boolean_index_f64_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `rch exec -- cargo check -p fnp-ufunc --all-targets`
- `rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings`
- `cargo fmt -p fnp-ufunc -- --check`
- `git diff --check`

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.262` | SIMD F64 `extract` mask count/decode | 100k candidate | `candidate_fnp_extract_hz2.txt`, `baseline_numpy_extract_bool_hz2.txt` | 46.853 us | 52.078 us | 0.900x | Keep |
| `franken_numpy-ixs5y.262` | SIMD F64 `extract` mask count/decode | 1M candidate | `candidate_fnp_extract_hz2.txt`, `baseline_numpy_extract_bool_hz2.txt` | 485.711 us | 506.924 us | 0.958x | Borderline keep |
| `franken_numpy-ixs5y.262` | Same `extract` fast path via `boolean_index` | 100k candidate | `candidate_fnp_boolean_index_hz2.txt`, `baseline_numpy_boolean_index_hz2.txt` | 43.993 us | 93.464 us | 0.471x | Keep |
| `franken_numpy-ixs5y.262` | Same `extract` fast path via `boolean_index` | 1M candidate | `candidate_fnp_boolean_index_hz2.txt`, `baseline_numpy_boolean_index_hz2.txt` | 479.160 us | 896.004 us | 0.535x | Keep |

Old-vs-new FNP extract delta on `hz2`:
- 100k: 74.721 us -> 46.853 us, candidate/baseline 0.627x.
- 1M: 793.914 us -> 485.711 us, candidate/baseline 0.612x.

Win/loss/neutral score:
- Head-to-head kept rows: 4 win / 0 loss / 0 neutral.
- Old-vs-new direct extract rows: 2 win / 0 loss / 0 neutral.

Notes:
- The verified SIMD source body deliberately avoids the rejected `.244`
  allocation shape: there are no per-chunk output vectors and no Rayon gather
  merge. It uses one exact-capacity output vector after a SIMD count pass.
- The retained `baseline_numpy_extract_hz2.txt` float-condition probe is not the
  decision comparator. The fair comparator is the bool-mask NumPy row in
  `baseline_numpy_extract_bool_hz2.txt`, matching the FNP `DType::Bool` bench
  intent.
- The 1M direct extract NumPy row has `cv_pct=12.21`, above the preferred 10%
  noise bound. The candidate still beats the NumPy median and is below the NumPy
  minimum (`485.711 us` vs `490.076 us`), so this is accepted as a borderline
  keep rather than a neutral.
- Focused validation passed for the two golden guards, `cargo check -p
  fnp-ufunc --all-targets`, `cargo clippy -p fnp-ufunc --all-targets -- -D
  warnings`, and `git diff --check`.
- After rebasing onto parent `0d3be5d0`, the extract golden guard,
  boolean-index golden guard, `cargo check -p fnp-ufunc --all-targets`, and
  `cargo clippy -p fnp-ufunc --all-targets -- -D warnings` all passed again
  through `rch`.
- `cargo fmt -p fnp-ufunc -- --check` still reports broad pre-existing format
  drift in `fnp-ufunc` benches and untouched `lib.rs` regions; no formatter was
  run for this perf commit.
- `ubs` completed on the touched subset with exit 1 against the existing broad
  `fnp-ufunc/src/lib.rs` inventory; see `ubs_touched_subset_summary.md` for the
  sampled pre-existing finding classes.
- Retry condition: reopen only if a low-CV same-worker rerun shows NumPy median
  or minimum at or below the candidate 1M direct extract row, if sidecar golden
  behavior changes, or if a broader dtype-general path can prove wins without
  regressing the sidecar-preserving scalar path.

## 2026-06-19 - Gauntlet Verify: FNP count_nonzero vs NumPy

Artifact directory: `tests/artifacts/perf/2026-06-19_ufunc_count_nonzero_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.246`.
- Subject before measured correction: code-first flat F64 `count_nonzero(axis=None)` parallel candidate with `1 << 14` activation threshold.
- Final code: parallel activation threshold raised to `1 << 19`, parallel chunk size kept at 4096 elements.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::count_nonzero` Criterion row.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7.
- Same-host decision machine: `thinkstation1`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.

Commands:
- `cargo test -p fnp-ufunc count_nonzero_f64_parallel_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `cargo bench -p fnp-ufunc --bench elementwise count_nonzero_flat_f64_sparse -- --sample-size 20 --warm-up-time 1 --measurement-time 3`
- Python NumPy timing script in `numpy_count_nonzero_local.txt` using the same data formula.
- `cargo check -p fnp-ufunc`
- `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`
- `cargo fmt --check`

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.246` | Original `1 << 14` parallel activation | 100k local candidate | `criterion_count_nonzero_local.txt`, `numpy_count_nonzero_local.txt` | 138.89 us | 39.006 us | 3.56x | Rejected, too eager |
| `franken_numpy-ixs5y.246` | Original `1 << 14` parallel activation | 1M local candidate | `criterion_count_nonzero_local.txt`, `numpy_count_nonzero_local.txt` | 92.072 us | 384.147 us | 0.240x | Keep large-row signal |
| `franken_numpy-ixs5y.246` | Raised threshold only, chunk coupled to threshold | 100k local correction | `criterion_count_nonzero_local_after_threshold.txt`, `numpy_count_nonzero_local.txt` | 8.4582 us | 39.006 us | 0.217x | Keep serial gate |
| `franken_numpy-ixs5y.246` | Raised threshold only, chunk coupled to threshold | 1M local correction | `criterion_count_nonzero_local_after_threshold.txt`, `numpy_count_nonzero_local.txt` | 173.68 us | 384.147 us | 0.452x | Weakened keep |
| `franken_numpy-ixs5y.246` | Final `1 << 19` activation with 4096-element chunks | 100k final | `criterion_count_nonzero_local_final.txt`, `numpy_count_nonzero_local.txt` | 8.3121 us | 39.006 us | 0.213x | Keep |
| `franken_numpy-ixs5y.246` | Final `1 << 19` activation with 4096-element chunks | 1M final | `criterion_count_nonzero_local_final.txt`, `numpy_count_nonzero_local.txt` | 110.42 us | 384.147 us | 0.287x | Keep, noisy CI |

Notes:
- The original optimization was partially rejected: the 16k activation threshold sent 100k arrays to Rayon and lost to NumPy by 3.56x.
- Raising the activation threshold fixed the 100k row by taking the existing serial path, but coupling chunk size to the threshold weakened the 1M parallel row. Splitting `COUNT_NONZERO_PARALLEL_CHUNK_ELEMS` restored small chunks for large arrays.
- The threshold-crossing golden fixture digest changed intentionally after raising the threshold; the updated digest guard passed after the fixture moved from the parallel path to the serial path.
- Final focused validation passed for `cargo test -p fnp-ufunc count_nonzero_f64_parallel_matches_serial_reference_and_golden_sha256 -- --nocapture`, `cargo check -p fnp-ufunc`, and `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`.
- `cargo fmt --check` still reports broad workspace formatting drift outside this slice; no workspace formatter was run.
- Retry condition: do not restore `COUNT_NONZERO_PARALLEL_MIN_ELEMS = 1 << 14` unless same-host 100k evidence beats NumPy and stays inside the prior FNP CI. Reopen the final 1M path only if a same-host rerun shows the final median at or above NumPy's median, or if the golden guard changes again without an intentional threshold fixture update.

## 2026-06-19 - Gauntlet Verify: FNP argwhere vs NumPy

Artifact directory: `tests/artifacts/perf/2026-06-19_ufunc_argwhere_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.248`.
- Subject code: code-first flat F64 `argwhere()` parallel interleaved coordinate gather candidate.
- Final code: unchanged; no revert required.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::argwhere` Criterion row.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7.
- Same-host decision machine: `thinkstation1`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.

Commands:
- `cargo test -p fnp-ufunc argwhere_f64_parallel_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `cargo bench -p fnp-ufunc --bench elementwise argwhere_f64_2d_sparse -- --sample-size 20 --warm-up-time 1 --measurement-time 3`
- Python NumPy timing script in `numpy_argwhere_local.txt` using the same data formula.
- `cargo check -p fnp-ufunc`
- `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`
- `cargo fmt --check`

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.248` | Parallel F64 `argwhere` interleaved coordinate gather | 512x512 final | `criterion_argwhere_local.txt`, `numpy_argwhere_local.txt` | 392.63 us | 1195.008 us | 0.329x | Keep |
| `franken_numpy-ixs5y.248` | Parallel F64 `argwhere` interleaved coordinate gather | 1024x1024 final | `criterion_argwhere_local.txt`, `numpy_argwhere_local.txt` | 1054.2 us | 5047.868 us | 0.209x | Keep |

Notes:
- The 512x512 row is 3.04x faster than NumPy; the 1024x1024 row is 4.79x faster.
- NumPy CV was noisy at 12.06% and 12.21%, but the NumPy minima were still above the FNP Criterion upper bounds on both sizes, so the keep decision does not rest on a noisy median edge.
- Final focused validation passed for `argwhere_f64_parallel_matches_serial_reference_and_golden_sha256`, `cargo check -p fnp-ufunc`, and `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`.
- `cargo fmt --check` still reports broad workspace formatting drift outside this slice; no workspace formatter was run.
- Retry condition: reopen only if a same-host rerun shows NumPy minimum below the FNP Criterion upper CI bound on either measured size, or if the interleaved C-order coordinate golden guard changes. Do not retry this as a standalone patch solely for lower NumPy-CV reruns.

## 2026-06-19 - Gauntlet Verify: FNP masked copyto vs NumPy

Artifact directory: `tests/artifacts/perf/2026-06-19_ufunc_remaining_masked_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.242`.
- Subject before measured correction: equal-shape F64 masked `copyto` paid a full `broadcast_to` clone before the equal-shape fast path, then activated Rayon at `1 << 14` elements.
- Final code: equal-shape F64/no-sidecar masked fast path is selected before source broadcasting; arrays below `1 << 20` elements use a direct serial fused mask/copy loop, with Rayon reserved for larger arrays.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::copyto` Criterion row.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7.
- Same-host decision machine: `thinkstation1`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.

Commands:
- `cargo bench -p fnp-ufunc --bench elementwise 'where_nonzero_f64_2d_sparse|copyto_equal_shape_masked|putmask_f64_masked|place_f64_masked_cycling|put_mask_f64_masked_cycling' -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `cargo bench -p fnp-ufunc --bench elementwise copyto_equal_shape_masked -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- Python NumPy timing script in `numpy_remaining_masked_local.txt` using the same data formulas.
- `cargo test -p fnp-ufunc copyto_masked_equal_shape_parallel_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `cargo check -p fnp-ufunc --all-targets`
- `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`
- `cargo fmt -p fnp-ufunc -- --check`
- `git diff --check`
- `timeout 120s ubs --only=rust crates/fnp-ufunc/src/lib.rs`

Triage scorecard:
- Current masked-family routing run vs local NumPy median: win/loss/neutral = 5/5/0 across 10 rows. The 2/2 copyto losses were selected because they were structural and fully inside this crate.
- Final focused copyto run vs local NumPy median: win/loss/neutral = 2/0/0 across the two same-host decision rows.

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.242` | Current code, routing evidence | 100k current | `criterion_remaining_masked_current.txt`, `numpy_remaining_masked_local.txt` | 358.730 us | 198.643 us | 1.806x | Loss, selected |
| `franken_numpy-ixs5y.242` | Current code, routing evidence | 1M current | `criterion_remaining_masked_current.txt`, `numpy_remaining_masked_local.txt` | 3438.882 us | 2253.369 us | 1.526x | Loss, selected |
| `franken_numpy-ixs5y.242` | Defer source broadcast only | 100k local candidate | `criterion_copyto_after_defer_broadcast_local.txt`, `numpy_remaining_masked_local.txt` | 1174.983 us | 198.643 us | 5.915x | Rejected |
| `franken_numpy-ixs5y.242` | Defer source broadcast only | 1M local candidate | `criterion_copyto_after_defer_broadcast_local.txt`, `numpy_remaining_masked_local.txt` | 2295.453 us | 2253.369 us | 1.019x | Rejected, neutral/loss |
| `franken_numpy-ixs5y.242` | Defer source broadcast only | 100k remote probe | `criterion_copyto_after_defer_broadcast.txt`, `numpy_remaining_masked_local.txt` | 204.972 us | 198.643 us | 1.032x | Rejected, noisy neutral |
| `franken_numpy-ixs5y.242` | Defer source broadcast only | 1M remote probe | `criterion_copyto_after_defer_broadcast.txt`, `numpy_remaining_masked_local.txt` | 2644.748 us | 2253.369 us | 1.174x | Rejected |
| `franken_numpy-ixs5y.242` | Final serial gate below `1 << 20` | 100k final local | `criterion_copyto_after_serial_threshold_local.txt`, `numpy_remaining_masked_local.txt` | 42.961 us | 198.643 us | 0.216x | Keep |
| `franken_numpy-ixs5y.242` | Final serial gate below `1 << 20` | 1M final local | `criterion_copyto_after_serial_threshold_local.txt`, `numpy_remaining_masked_local.txt` | 1316.171 us | 2253.369 us | 0.584x | Keep |
| `franken_numpy-ixs5y.242` | Final serial gate below `1 << 20` | 100k remote confirmation | `criterion_copyto_after_serial_threshold_rch.txt`, `numpy_remaining_masked_local.txt` | 24.632 us | 198.643 us | 0.124x | Confirming signal |
| `franken_numpy-ixs5y.242` | Final serial gate below `1 << 20` | 1M remote confirmation | `criterion_copyto_after_serial_threshold_rch.txt`, `numpy_remaining_masked_local.txt` | 908.107 us | 2253.369 us | 0.403x | Confirming signal |

Notes:
- The first exotic lever was only half-right: moving the broadcast out of the equal-shape path removed clone work, but it exposed that the `1 << 14` parallel threshold was still a bad morsel size for the 100k and 1M copyto rows.
- The kept lever is the cache/simplex version of the graveyard lesson: keep the common equal-shape dense loop fused and serial until the loop body has enough work to amortize Rayon scheduling, and avoid materializing a broadcast array that the SCE shape equality already proves unnecessary.
- The golden fixture digest changed intentionally because the threshold-crossing fixture moved below the new parallel activation point; elementwise reference comparison passed before the digest assertion, and the updated digest guard then passed.
- Final focused validation passed for `copyto_masked_equal_shape_parallel_matches_serial_reference_and_golden_sha256`, `cargo check -p fnp-ufunc --all-targets`, `cargo clippy -p fnp-ufunc --all-targets -- -D warnings` on `ovh-a`, and `git diff --check`.
- The first clippy attempt hit an rch worker missing `cargo-clippy` for `nightly-2026-02-20`; that environment failure is recorded in `cargo_clippy_fnp_ufunc.txt`, and the successful retry is recorded in `cargo_clippy_fnp_ufunc_retry_ovh_a.txt`.
- `cargo fmt -p fnp-ufunc -- --check` still reports broad pre-existing formatting drift outside this slice; no workspace formatter was run.
- `timeout 120s ubs --only=rust crates/fnp-ufunc/src/lib.rs` did not emit a completion summary before the cap; keep the incomplete `ubs_fnp_ufunc_lib.txt` artifact as a tooling caveat, not a pass.
- Retry condition: reopen only if a same-host NumPy rerun beats the final local FNP median on either row, if compact bool-mask storage replaces the current f64 mask representation, or if a larger-copy workload shows the raised Rayon threshold losing above `1 << 20`.

## 2026-06-19 - Gauntlet Verify: FNP putmask vs NumPy

Artifact directory: `tests/artifacts/perf/2026-06-19_ufunc_putmask_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.243`.
- Subject before measured correction: F64/no-sidecar `putmask` activated Rayon at `1 << 14` elements, so the 100k masked cycling-fill row paid scheduler overhead and the serial fallback still routed every write through integer-mutation sidecar plumbing.
- Final code: F64/no-sidecar `putmask` uses direct dense serial loops below `1 << 20`; above that threshold it keeps the existing Rayon path, including position-index cycling with `values[i % values.len()]`.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::putmask` Criterion row.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7.
- Same-worker FNP confirmation: rch worker `vmi1227854`.
- Same-host decision machine: `thinkstation1`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-ufunc --bench elementwise putmask_f64_masked -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `RCH_WORKER=vmi1227854 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-ufunc --bench elementwise putmask_f64_masked -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- Python NumPy timing script in `numpy_putmask_local.txt` using the same data formula.
- `cargo bench -p fnp-ufunc --bench elementwise putmask_f64_masked -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `RCH_WORKER=vmi1227854 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-ufunc putmask_f64_parallel_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo check -p fnp-ufunc --all-targets`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings`
- `cargo fmt -p fnp-ufunc -- --check`
- `timeout 120s ubs --only=rust crates/fnp-ufunc/src/lib.rs`

Triage scorecard:
- Current focused `putmask` run vs local NumPy median: win/loss/neutral = 1/1/0 across the two decision rows. The 100k row lost by 1.362x and was selected.
- Final rch same-worker `putmask` run vs local NumPy median: win/loss/neutral = 2/0/0.
- Final local same-host `putmask` run vs local NumPy median: win/loss/neutral = 2/0/0.

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.243` | Current code, routing evidence | 100k current rch | `criterion_putmask_current_rch.txt`, `numpy_putmask_local.txt` | 128.047 us | 93.991 us | 1.362x | Loss, selected |
| `franken_numpy-ixs5y.243` | Current code, routing evidence | 1M current rch | `criterion_putmask_current_rch.txt`, `numpy_putmask_local.txt` | 632.938 us | 1209.271 us | 0.523x | Existing win |
| `franken_numpy-ixs5y.243` | Final direct serial gate below `1 << 20` | 100k final rch | `criterion_putmask_after_serial_threshold_rch.txt`, `numpy_putmask_local.txt` | 65.564 us | 93.991 us | 0.698x | Keep |
| `franken_numpy-ixs5y.243` | Final direct serial gate below `1 << 20` | 1M final rch | `criterion_putmask_after_serial_threshold_rch.txt`, `numpy_putmask_local.txt` | 554.537 us | 1209.271 us | 0.459x | Keep |
| `franken_numpy-ixs5y.243` | Final direct serial gate below `1 << 20` | 100k final local | `criterion_putmask_after_serial_threshold_local.txt`, `numpy_putmask_local.txt` | 53.237 us | 93.991 us | 0.566x | Keep, same-host |
| `franken_numpy-ixs5y.243` | Final direct serial gate below `1 << 20` | 1M final local | `criterion_putmask_after_serial_threshold_local.txt`, `numpy_putmask_local.txt` | 559.970 us | 1209.271 us | 0.463x | Keep, same-host |

Notes:
- Same-worker FNP delta on `vmi1227854`: 100k improved from 128.047 us to 65.564 us, a 1.95x speedup; 1M improved from 632.938 us to 554.537 us, a 1.14x speedup.
- The kept lever is the graveyard "constants kill you" correction: avoid tiny Rayon morsels and integer-sidecar mutation machinery when SCE and dtype checks prove a dense F64/no-sidecar loop. The exotic idea was deliberately small but architectural: use the layout proof to select the flat cache-local loop before generic mutation dispatch.
- NumPy 100k timing was noisy at 11.47% CV, but the final local FNP median of 53.237 us is still below the NumPy minimum of 90.348 us, so the keep decision does not rest on a noisy median edge.
- The first focused golden run failed only at the SHA-256 digest after the elementwise serial-reference comparison had already passed. The updated digest `4fffe2fd2c9e96fa07d22719917ae99810b0f84a3ee5fb1d7c5128f910da2b75` reflects the intentional threshold-fixture path change, and the final golden test passed.
- Final focused validation passed for `putmask_f64_parallel_matches_serial_reference_and_golden_sha256`, `cargo check -p fnp-ufunc --all-targets`, and `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`.
- `cargo fmt -p fnp-ufunc -- --check` still reports broad pre-existing formatting drift in `crates/fnp-ufunc/benches/elementwise.rs` and unrelated `crates/fnp-ufunc/src/lib.rs` regions; no workspace formatter was run.
- `timeout 120s ubs --only=rust crates/fnp-ufunc/src/lib.rs` timed out with `UBS_EXIT:124`; keep `ubs_fnp_ufunc_lib.txt` as a tooling caveat, not a pass.
- Retry condition: reopen only if a same-host NumPy rerun beats the final local FNP median on either row, if `putmask` semantics change away from position-index cycling, if compact bool-mask storage changes the measured loop body, or if larger rows above `1 << 20` show the raised Rayon threshold losing.

## 2026-06-19 - Gauntlet Verify: FNP put_mask vs NumPy

Artifact directory: `tests/artifacts/perf/2026-06-19_ufunc_put_mask_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.254`.
- Subject before measured correction: F64/no-sidecar `put_mask` activated Rayon at `1 << 14` elements, so the 100k true-rank cycling-fill row paid segmented-prefix scheduler overhead and the serial fallback still routed every write through integer-mutation sidecar plumbing.
- Final code: F64/no-sidecar `put_mask` uses an 8-lane SIMD mask scan and modulo-free value cycling below `1 << 20`; above that threshold it keeps the segmented-prefix Rayon path with a fixed 4K chunk size.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::put_mask` Criterion row.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7, timed with the same true-rank cycling formula.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-ufunc --bench elementwise put_mask_f64_masked_cycling -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- Python NumPy timing script in `numpy_put_mask_local.txt` using the same data formula.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-ufunc put_mask_f64_parallel_matches_serial_reference_and_golden_sha256 -- --nocapture`

Triage scorecard:
- Current fresh focused run vs refreshed NumPy median: win/loss/neutral = 1/1/0 across the two decision rows. The 100k row lost by 7.866x and was selected.
- Final remote run vs refreshed NumPy median: win/loss/neutral = 2/0/0.
- Earlier same-day same-worker ovh-a current snapshot to final ovh-a delta: 100k improved 81.857 us -> 15.858 us (5.16x); 1M improved 388.001 us -> 335.444 us (1.16x).

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.254` | Current code, fresh routing evidence | 100k current rch `hz2` | `baseline_fnp_put_mask_rch.txt`, `numpy_put_mask_local.txt` | 244.411 us | 31.069 us | 7.866x | Loss, selected |
| `franken_numpy-ixs5y.254` | Current code, fresh routing evidence | 1M current rch `hz2` | `baseline_fnp_put_mask_rch.txt`, `numpy_put_mask_local.txt` | 483.383 us | 686.361 us | 0.704x | Existing win |
| `franken_numpy-ixs5y.254` | Threshold/direct serial, no SIMD | 100k candidate rch `hz1` | `candidate_fnp_put_mask_rch_confirm.txt`, `numpy_put_mask_local.txt` | 76.797 us | 31.069 us | 2.472x | Rejected, still loses 100k |
| `franken_numpy-ixs5y.254` | Threshold/direct serial, no SIMD | 1M candidate rch `hz1` | `candidate_fnp_put_mask_rch_confirm.txt`, `numpy_put_mask_local.txt` | 445.529 us | 686.361 us | 0.649x | Win but not enough |
| `franken_numpy-ixs5y.254` | SIMD serial with `1 << 19` cutoff | 100k candidate rch `hz2` | `candidate_fnp_put_mask_simd_rch.txt`, `numpy_put_mask_local.txt` | 19.037 us | 31.069 us | 0.613x | Win |
| `franken_numpy-ixs5y.254` | SIMD serial with `1 << 19` cutoff | 1M candidate rch `hz2` | `candidate_fnp_put_mask_simd_rch.txt`, `numpy_put_mask_local.txt` | 709.480 us | 686.361 us | 1.034x | Rejected, neutral/loss |
| `franken_numpy-ixs5y.254` | Final SIMD serial with `1 << 20` cutoff | 100k final rch `ovh-a` | `final_fnp_put_mask_simd_threshold_rch.txt`, `numpy_put_mask_local.txt` | 15.858 us | 31.069 us | 0.510x | Keep |
| `franken_numpy-ixs5y.254` | Final SIMD serial with `1 << 20` cutoff | 1M final rch `ovh-a` | `final_fnp_put_mask_simd_threshold_rch.txt`, `numpy_put_mask_local.txt` | 335.444 us | 686.361 us | 0.489x | Keep |

Notes:
- The threshold-only lever was insufficient: it improved 1M but still lost the 100k row by 2.472x versus the refreshed NumPy reference.
- The first SIMD lever was too aggressive about re-entering Rayon at `1 << 19`: it won 100k but regressed the 1M row to a neutral/loss against NumPy. Raising the cutoff to `1 << 20` keeps both measured rows on the cache-local SIMD serial path.
- The kept lever is the graveyard "constants kill you" correction plus vectorized mask extraction: use SCE/dtype proof to skip generic mutation machinery, use SIMD only to build sparse lane masks, and cycle values with an increment/reset index instead of `%` in each true lane.
- The golden fixture digest changed intentionally as the threshold-crossing fixture moved to `PUT_MASK_PARALLEL_MIN_ELEMS + 421`; elementwise serial-reference comparison passed before the SHA assertion, and the updated digest `f8a49cce66312e0fb3fdfdbcc5e31b70662343eff3f8d49ae4f01ae828da3c0c` passed.
- The sidecar fallback test fixture was corrected from `(1_i64 << 53) + 2` to `(1_i64 << 53) - 2`, staying within the crate's exact temporary F64 bridge contract while preserving the large-integer sidecar proof.
- Final focused validation passed for `put_mask_f64_parallel_matches_serial_reference_and_golden_sha256`, `cargo check -p fnp-ufunc --all-targets`, `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`, and `git diff --check`.
- `cargo fmt --check` still reports broad pre-existing workspace formatting drift outside this slice; the put_mask hunk was manually adjusted to match the targeted rustfmt diff and no unrelated formatting was applied.
- `timeout 120s ubs --only=rust crates/fnp-ufunc/src/lib.rs` did not produce a completion summary before the wrapper returned, and zsh did not preserve the exit code in the artifact; keep `ubs_fnp_ufunc_lib.txt` as inconclusive, not a pass.
- Retry condition: reopen only if a same-host NumPy rerun beats the final FNP median on either row, if compact bool-mask storage changes the loop body, or if rows above `1 << 20` show the retained segmented-prefix parallel path losing to the SIMD serial path.

## 2026-06-20 - Gauntlet Verify: FNP place vs NumPy

Artifact directory: `tests/artifacts/perf/2026-06-20_ufunc_place_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.252`.
- Subject before measured correction: F64/no-sidecar `place` activated Rayon at `1 << 14` elements, so the 100k true-rank cycling-fill row paid segmented-prefix scheduler overhead and the serial fallback still routed every write through integer-mutation sidecar plumbing.
- Final code: F64/no-sidecar `place` uses an 8-lane SIMD mask scan and modulo-free value cycling below `1 << 20`; above that threshold it keeps the segmented-prefix Rayon path with a fixed 4K chunk size.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::place` Criterion row.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7, timed with the same mask and cyclic value formula.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-ufunc --bench elementwise place_f64_masked_cycling -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- Python NumPy timing script in `numpy_place_local.txt` using the same data formula.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-ufunc place_f64_parallel_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo check -p fnp-ufunc --all-targets`
- `RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings`
- `cargo fmt -p fnp-ufunc -- --check`
- `git diff --check`
- `timeout 120s ubs --only=rust crates/fnp-ufunc/src/lib.rs`

Triage scorecard:
- Current focused run vs refreshed NumPy median: win/loss/neutral = 1/1/0 across the two decision rows. The 100k row lost by 1.307x and was selected.
- Final remote run vs refreshed NumPy median: win/loss/neutral = 2/0/0.

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.252` | Current code, fresh routing evidence | 100k current rch `hz1` | `baseline_estimates_extracted_before_candidate.txt`, `numpy_place_local.txt` | 87.094 us | 66.616 us | 1.307x | Loss, selected |
| `franken_numpy-ixs5y.252` | Current code, fresh routing evidence | 1M current rch `hz1` | `baseline_estimates_extracted_before_candidate.txt`, `numpy_place_local.txt` | 441.318 us | 815.936 us | 0.541x | Existing win |
| `franken_numpy-ixs5y.252` | Final SIMD serial with `1 << 20` cutoff | 100k final rch `hz2` | `candidate_fnp_place_rch.txt`, `numpy_place_local.txt` | 21.050 us | 66.616 us | 0.316x | Keep |
| `franken_numpy-ixs5y.252` | Final SIMD serial with `1 << 20` cutoff | 1M final rch `hz2` | `candidate_fnp_place_rch.txt`, `numpy_place_local.txt` | 273.974 us | 815.936 us | 0.336x | Keep |

Notes:
- This is the same successful "constants kill you" correction as `copyto`, `putmask`, and `put_mask`, but applied to `place`'s true-rank cyclic semantics: avoid tiny Rayon morsels and integer-sidecar mutation machinery when dtype and sidecar checks prove a flat F64/no-sidecar loop.
- The SIMD serial path scans F64 mask chunks as 8-lane vectors, emits a bitmask of nonzero lanes, and advances the cyclic value index with increment/reset rather than `%` on every write.
- The golden fixture digest changed intentionally after the threshold-crossing fixture moved below the new parallel activation point; elementwise serial-reference comparison passed before the SHA assertion, and the updated digest `41ebf3fa471d4b7c9b29ddc1cde3e96b7b972072359d9ed98ac53ee806bf7add` passed.
- Final focused validation passed for `place_f64_parallel_matches_serial_reference_and_golden_sha256`, `cargo check -p fnp-ufunc --all-targets`, `cargo clippy -p fnp-ufunc --all-targets -- -D warnings` on retry worker `hz1`, and `git diff --check`.
- Initial clippy on `ovh-b` failed in a dependency build script with `SIGILL`; the retry on `hz1` is the passing clippy gate.
- `cargo fmt -p fnp-ufunc -- --check` still reports broad pre-existing formatting drift outside this slice; no workspace formatter was run.
- `timeout 120s ubs --only=rust crates/fnp-ufunc/src/lib.rs` timed out after starting the Rust scan; keep `ubs_fnp_ufunc_lib.txt` as inconclusive, not a pass.
- Retry condition: reopen only if a same-host NumPy rerun beats the final FNP median on either row, if compact bool-mask storage changes the loop body, or if rows above `1 << 20` show the retained segmented-prefix parallel path losing to the SIMD serial path.

## 2026-06-20 - Gauntlet Verify: `fnp-linalg` matrix norm column reductions

Artifact directory: `tests/artifacts/perf/2026-06-20_linalg_batch_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.235`.
- Subject before measured correction: `matrix_norm_nxn_orders` already scanned row-major for large `ord=1/-1` matrices, but allocated the column-sum scratch buffer on the heap for every call.
- Kept lever: stack-resident scratch for 512 through 1024 columns, with the existing heap path retained outside that measured window.
- No-ship lever: an unrolled Frobenius accumulator was tested and reverted after batch Frobenius rows regressed.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg 'batch_trace|batch_matrix_norm|matrix_norm_nxn_orders|kron_nxn' -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg 'matrix_norm_nxn_orders/(one|neg_one)' -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-linalg matrix_norm_column_reduction_matches_strided_reference_bits -- --nocapture`
- Direct Python NumPy comparator on `hz2` in `numpy_linalg_hz2.txt`.

Triage scorecard:
- Current `hz2` FNP vs NumPy: win/loss/neutral = 1/7/0 across the eight column norm rows. The `one/128` row was effectively neutral/loss at 1.005x, and the 256 through 1024 rows were clear losses.
- Final `hz2` FNP vs old `hz2` FNP: win/loss/neutral = 8/0/0.
- Final `hz2` FNP vs NumPy: win/loss/neutral = 2/6/0. This is a kept gap-narrowing lever, not a full NumPy domination closeout.

| Bead | Lever | Workload | Artifact | Old FNP ns | Final FNP ns | NumPy ns | Final/Old | Final/NumPy | Verdict |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `franken_numpy-ixs5y.235` | Stack scratch 512-1024 cols | `one/128` on `hz2` | `criterion_linalg_current.txt`, `criterion_linalg_column_stack_gated_candidate_hz1.txt`, `numpy_linalg_hz2.txt` | 9603 | 7544 | 9553 | 0.786x | 0.790x | Keep/supporting win |
| `franken_numpy-ixs5y.235` | Stack scratch 512-1024 cols | `neg_one/128` on `hz2` | same | 9375 | 7484 | 9574 | 0.798x | 0.782x | Keep/supporting win |
| `franken_numpy-ixs5y.235` | Stack scratch 512-1024 cols | `one/256` on `hz2` | same | 38032 | 30444 | 27712 | 0.800x | 1.099x | Keep, still loses |
| `franken_numpy-ixs5y.235` | Stack scratch 512-1024 cols | `neg_one/256` on `hz2` | same | 37675 | 29924 | 28312 | 0.794x | 1.057x | Keep, still loses |
| `franken_numpy-ixs5y.235` | Stack scratch 512-1024 cols | `one/512` on `hz2` | same | 154304 | 116333 | 103667 | 0.754x | 1.122x | Keep, still loses |
| `franken_numpy-ixs5y.235` | Stack scratch 512-1024 cols | `neg_one/512` on `hz2` | same | 152028 | 116827 | 102987 | 0.768x | 1.134x | Keep, still loses |
| `franken_numpy-ixs5y.235` | Stack scratch 512-1024 cols | `one/1024` on `hz2` | same | 615716 | 458082 | 397192 | 0.744x | 1.153x | Keep, still loses |
| `franken_numpy-ixs5y.235` | Stack scratch 512-1024 cols | `neg_one/1024` on `hz2` | same | 603420 | 466084 | 393621 | 0.772x | 1.184x | Keep, still loses |
| `franken_numpy-ixs5y.235` | Frobenius unroll | `batch_matrix_norm_fro/4096x8x8` on `hz1` | `criterion_linalg_fro_head_baseline.txt`, `criterion_linalg_fro_unroll_candidate.txt` | 76821 | 84812 | n/a | 1.104x | n/a | Reverted |
| `franken_numpy-ixs5y.235` | Frobenius unroll | `batch_matrix_norm_fro/1024x32x32` on `hz1` | same | 194123 | 221803 | n/a | 1.143x | n/a | Reverted |

Notes:
- The stack path preserves the old scalar addition order per column; the focused test compares both a heap-cache case and a stack-cache case against the former strided reference bits, including NaN propagation through `1`, `-1`, `inf`, and `-inf`.
- The first stack candidate was not kept as-is because direct `hz1` evidence showed a small `one/256` regression. The final code gates stack scratch to the measured 512-1024 column range and keeps the heap path elsewhere.
- `numpy_column_vmi_rch.txt` is an invalid probe artifact only: RCH warned that the command was non-compilation and the Python quoting failed before timing. It is not counted in any ratio.
- Remaining gap: NumPy is still faster for 256 through 1024 column reductions on `hz2`. Next deeper lever should be vectorized absolute-value accumulation or multiple-column strip mining that preserves per-column scalar addition order, not another allocation-only retune.

## 2026-06-20 - Gauntlet Verify: `fnp-linalg` batched Frobenius norm lane fill

Artifact directory: `tests/artifacts/perf/2026-06-20_linalg_batch_fro_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.238`.
- Subject before measured closeout: `batch_matrix_norm(..., ord="fro")` already had the direct lane-fill path from the code-first child, but it had not been put through a same-worker NumPy ratio gate.
- Kept lever: direct batched Frobenius lane fill after one shape/data validation; each lane uses the same row-major `v * v` accumulation and final `sqrt` as the per-lane `matrix_norm_nxn(..., "fro")` reference.
- No new source edit was made in this closeout. The measured decision is keep/close, not another speculative tweak.
- Worker: `hz1`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg 'batch_matrix_norm_fro' -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- Direct Python NumPy comparator on `hz1` in `numpy_batch_matrix_norm_fro_hz1_success.txt`.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-linalg batch_matrix_norm_fro_direct_lane_fill_matches_per_lane_reference_bits -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo build -p fnp-linalg --release`

Triage scorecard:
- Final same-worker `hz1` FNP vs NumPy: win/loss/neutral = 2/0/0.

| Bead | Lever | Workload | Artifact | FrankenNumPy ns | NumPy ns | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.238` | Direct batched Frobenius lane fill | `4096x8x8` on `hz1` | `fnp_batch_matrix_norm_fro_current.txt`, `numpy_batch_matrix_norm_fro_hz1_success.txt` | 76177 | 234973 | 0.324x | Keep, 3.08x faster |
| `franken_numpy-ixs5y.238` | Direct batched Frobenius lane fill | `1024x32x32` on `hz1` | same | 218772 | 581466 | 0.376x | Keep, 2.66x faster |

Notes:
- The focused bit-preservation guard passed for `batch_matrix_norm_fro_direct_lane_fill_matches_per_lane_reference_bits`, covering serial and threshold-crossing batch shapes plus NaN, Inf, and signed-zero inputs against the old per-lane reference.
- `numpy_batch_matrix_norm_fro_hz1.txt` is an invalid shell-quote attempt and is not counted in the ratio table. The counted comparator is `numpy_batch_matrix_norm_fro_hz1_success.txt` on Python 3.14.4 / NumPy 2.3.5 on `hz1`.
- This bead should not be reopened for another Frobenius micro-retune unless a same-worker NumPy rerun beats either final Rust median, or a future change alters the accumulation order, batch parallel threshold, or matrix-norm dispatch path.
