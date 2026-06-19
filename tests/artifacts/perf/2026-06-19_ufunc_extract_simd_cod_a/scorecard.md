# 2026-06-19 fnp-ufunc extract SIMD mask-decode scorecard

Agent: `cod-a` / `YellowElk`
Bead: `franken_numpy-ixs5y.262`
Baseline commit for the measured old FNP rows: `39bb1e78`
Rebased parent carrying the same SIMD source body: `0d3be5d0`
Worktree: `/data/projects/.scratch/franken_numpy-cod-a-20260619-0535`
Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`

## Lever

`UFuncArray::extract` uses a sidecar-free F64 fast path:
portable SIMD counts selected lanes, allocates one exact-capacity output vector,
then decodes each lane bitmask with `trailing_zeros()`.

The path is intentionally dtype/storage narrow:
- `arr.dtype == DType::F64`
- `arr.integer_sidecar.is_none()`

Sidecar arrays and other dtypes keep the previous source-index path. During
rebase, `origin/main` already contained the same SIMD source body from
`0d3be5d0`, so this commit retains that source unchanged and adds independent
cod-a verification artifacts and ledger evidence.

## Decision Rows

All FNP benches were run through `rch exec -- cargo bench -p fnp-ufunc --bench
elementwise ... -- --sample-size 20 --warm-up-time 1 --measurement-time 3
--output-format bencher`.

| Workload | Baseline FNP | Candidate FNP | Candidate/Baseline | NumPy 2.4.3 median | Candidate/NumPy | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `extract_f64_masked/100000` | 74,721 ns | 46,853 ns | 0.627x | 52,078 ns | 0.900x | Win |
| `extract_f64_masked/1000000` | 793,914 ns | 485,711 ns | 0.612x | 506,924 ns | 0.958x | Narrow win |
| `boolean_index_f64_masked_sparse/100000` | n/a | 43,993 ns | n/a | 93,464 ns | 0.471x | Win |
| `boolean_index_f64_masked_sparse/1000000` | n/a | 479,160 ns | n/a | 896,004 ns | 0.535x | Win |

Win/loss/neutral score:
- Direct measured rows kept: 4 win / 0 loss / 0 neutral.
- Direct extract old-vs-new rows: 2 win / 0 loss / 0 neutral.

## Artifact Map

- Baseline FNP extract: `baseline_fnp_extract_hz2.txt`
- Candidate FNP extract: `candidate_fnp_extract_hz2.txt`
- Fair NumPy extract bool-mask comparator: `baseline_numpy_extract_bool_hz2.txt`
- Retained non-decision NumPy float-condition probe: `baseline_numpy_extract_hz2.txt`
- Candidate dependent boolean-index row: `candidate_fnp_boolean_index_hz2.txt`
- NumPy dependent boolean-index comparator: `baseline_numpy_boolean_index_hz2.txt`
- Extract golden guard: `candidate_test_extract_golden.txt`
- Boolean-index golden guard: `candidate_test_boolean_index_golden.txt`
- Cargo check: `candidate_cargo_check_fnp_ufunc.txt`
- Cargo clippy: `candidate_cargo_clippy_fnp_ufunc.txt`
- Post-rebase extract golden guard: `post_rebase_test_extract_golden.txt`
- Post-rebase boolean-index golden guard:
  `post_rebase_test_boolean_index_golden.txt`
- Post-rebase cargo check: `post_rebase_cargo_check_fnp_ufunc.txt`
- Post-rebase cargo clippy: `post_rebase_cargo_clippy_fnp_ufunc.txt`
- Fmt check output: `cargo_fmt_fnp_ufunc_check.txt`
- Whitespace check: `git_diff_check.txt`
- UBS touched-subset summary: `ubs_touched_subset_summary.md`

## Verification

Passed:
- `rch exec -- cargo test -p fnp-ufunc extract_f64_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `rch exec -- cargo test -p fnp-ufunc boolean_index_f64_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `rch exec -- cargo check -p fnp-ufunc --all-targets`
- `rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings`
- `git diff --check`

Post-rebase against parent `0d3be5d0`, also passed:
- `rch exec -- cargo test -p fnp-ufunc extract_f64_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `rch exec -- cargo test -p fnp-ufunc boolean_index_f64_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `rch exec -- cargo check -p fnp-ufunc --all-targets`
- `rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings`

Known unrelated gate state:
- `cargo fmt -p fnp-ufunc -- --check` still reports broad pre-existing
  formatting drift in `fnp-ufunc` benches and untouched `lib.rs` regions. No
  formatter was run for this perf commit.
- `ubs crates/fnp-ufunc/src/lib.rs docs/NEGATIVE_EVIDENCE.md
  tests/artifacts/perf/2026-06-19_ufunc_extract_simd_cod_a/scorecard.md`
  completed with exit 1 after 222 seconds on the existing broad inventory in
  the 70k-line `fnp-ufunc/src/lib.rs` file; see `ubs_touched_subset_summary.md`.

## Caveat And Retry Condition

The fair NumPy 1M extract row has `cv_pct=12.21`, above the preferred 10%
noise bound. The candidate still beats the NumPy median and is slightly below
the NumPy minimum (`485,711 ns` vs `490,076 ns`), so this is accepted as a
borderline keep. Reopen if a low-CV same-worker rerun shows NumPy median or
minimum at or below the candidate extract row.
