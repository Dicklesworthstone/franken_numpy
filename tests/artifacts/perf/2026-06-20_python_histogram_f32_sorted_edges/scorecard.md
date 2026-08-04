# fnp-python histogram f32 sorted-edge scorecard

Run identity:
- Bead: `franken_numpy-ixs5y.268`.
- Agent: `BlackThrush` / `cod-a`.
- Subject API: Python-boundary `fnp.histogram` through `crates/fnp-python/benches/criterion_python_surface.rs`.
- Oracle/reference: NumPy inside the same Criterion harness process.
- Worker for decision rows: `hz2`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Starting commit: `93d2fa6a` (`perf(fnp-python): generalize nanprod axis fast path to non-last axis (2.77x loss -> 15x win)`).

Lever kept:
- During the existing finite min/max scan for `float32` histogram inputs, detect monotone nondecreasing data.
- For monotone data, classify bins with one streaming edge pointer over the existing `float32` edges, replacing per-element affine division plus correction with `O(n + bins)` edge advancement.
- Preserve the original fallback, strict edge validation, float32 edge construction, and unsorted scalar classifier.

Alien and profiling mapping:
- Alien-graveyard: distribution-sort/merge-path style monotone cursoring over sorted streams, with "constants kill you" avoiding per-element division when the input order gives a stronger primitive.
- Alien-artifact-coding: specialize a data-order invariant detected for free during an existing scan rather than adding a new public contract.
- Extreme optimization: one lever, same-worker head-to-head benchmark, conformance guard, rejected candidates recorded.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo bench -p fnp-python --bench criterion_python_surface -- 'python_histogram_boundary|python_setops_boundary|python_statistics_boundary|python_einsum_boundary|python_linalg_boundary|python_char_ascii_boundary' --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a RCH_WORKER=hz2 RCH_WORKERS=hz2 rch exec -- cargo bench -p fnp-python --bench criterion_python_surface -- python_histogram_boundary --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a RCH_WORKER=hz2 RCH_WORKERS=hz2 rch exec -- cargo test -p fnp-python histogram_matches_numpy_across_bins_range_density_weights_and_empty -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo check -p fnp-python --all-targets`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo build --release -p fnp-python`
- `cargo fmt -p fnp-python -- --check`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo clippy -p fnp-python --all-targets -- -D warnings`
- `git diff --check`

Decision table:

| Bead | Lever | Workload | Worker | FrankenNumPy | NumPy | FNP/NumPy ratio | Old-to-new ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---:|---|
| `franken_numpy-ixs5y.266` | Origin/main routing baseline | `histogram_i64_100k_50` | `hz2` | 599,955 ns | 861,321 ns | 0.697x, 1.44x faster | - | Baseline win |
| `franken_numpy-ixs5y.266` | Origin/main routing baseline | `histogram_f32_100k_50` | `hz2` | 668,179 ns | 613,046 ns | 1.09x slower | - | Open gap |
| `franken_numpy-ixs5y.266` | Raise parallel threshold | `histogram_i64_100k_50` | `hz2` | 845,866 ns | 867,590 ns | 0.975x, 1.03x faster | 1.41x slower vs baseline FNP | No-ship |
| `franken_numpy-ixs5y.266` | Raise parallel threshold | `histogram_f32_100k_50` | `hz2` | 673,497 ns | 590,563 ns | 1.14x slower | 1.01x slower vs baseline FNP | No-ship |
| `franken_numpy-ixs5y.267` | Local count accumulator | `histogram_i64_100k_50` | `hz2` | 878,361 ns | 830,049 ns | 1.06x slower | control row noisy | No-ship |
| `franken_numpy-ixs5y.267` | Local count accumulator | `histogram_f32_100k_50` | `hz2` | 686,940 ns | 608,869 ns | 1.13x slower | 1.03x slower vs baseline FNP | No-ship |
| `franken_numpy-ixs5y.268` | Sorted edge-pointer count | `histogram_i64_100k_50` | `hz2` | 651,730 ns | 840,532 ns | 0.775x, 1.29x faster | 1.09x slower vs origin baseline FNP | Keep, i64 still beats NumPy |
| `franken_numpy-ixs5y.268` | Sorted edge-pointer count | `histogram_f32_100k_50` | `hz2` | 449,882 ns | 584,574 ns | 0.770x, 1.30x faster | 0.673x, 1.49x faster | Keep |

Scorecard:
- Routing baseline vs NumPy on histogram rows: win/loss/neutral = 1/1/0.
- Threshold candidate vs NumPy: win/loss/neutral = 1/1/0, rejected because it did not close f32 and erased most of the i64 margin.
- Local-count candidate vs NumPy: win/loss/neutral = 0/2/0, rejected.
- Sorted edge-pointer candidate vs NumPy: win/loss/neutral = 2/0/0, kept.
- Primary targeted gap moved from 1.09x slower than NumPy to 0.770x of NumPy time.

Conformance and validation:
- Focused histogram parity test passed, including the new `float32` sorted 100k, 50-bin case.
- Supplemental `cargo test -p fnp-python` cleared 531 inline tests plus early conformance shards, then failed outside this path in `conformance_argwhere::argwhere_python_container_surfaces_match_numpy` because the NumPy oracle script emitted an `IndentationError`.
- `cargo check -p fnp-python --all-targets` passed on RCH with the crate's pre-existing three dead-code warnings.
- `cargo build --release -p fnp-python` passed on RCH worker `vmi1149989` with the same three warnings.
- `git diff --check` passed.
- `ubs crates/fnp-python/src/lib.rs` completed in 198s and exited 1 with broad file-wide inventory (`473` critical heuristic findings, `3661` warnings, `4554` info); no hunk-local histogram finding was identified.
- `cargo fmt -p fnp-python -- --check` is blocked by broad pre-existing rustfmt drift across `fnp-python`, outside this hunk.
- `cargo clippy -p fnp-python --all-targets -- -D warnings` is blocked by broad pre-existing fnp-python lint debt, not this histogram path.

Isomorphism notes:
- The sorted fast path activates only after all inputs pass the existing finite scan and remain monotone nondecreasing.
- It uses the already-created `float32` edge array, so first/last endpoint rounding and bin boundaries remain tied to NumPy's float32 histogram behavior.
- The loop preserves `[edge_i, edge_{i+1})` with the final bin inclusive through `while idx != nbins - 1 && x >= es[idx + 1]`.
- Values below the first edge are skipped, values above the final edge terminate only because sorted order proves no later value can re-enter the range.
- Unsorted inputs retain the previous affine index plus edge-correction classifier.

Coordination notes:
- Work was isolated in `/data/projects/.scratch/franken_numpy-cod-a-20260620-0125`.
- Agent Mail read and reservations worked, but write/send operations reported a malformed SQLite write circuit; no mail start/completion message was persisted.
- Rejected probes `.266` and `.267` are deliberately recorded here so this f32 histogram lane does not repeat threshold or local-count-only variants.
