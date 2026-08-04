# Batch Cholesky Direct-Write n=16/32 Scorecard

Bead: `franken_numpy-ixs5y.273`
Crate: `fnp-linalg`
Agent: `YellowElk` / `cod-b`
Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`

## Verdict

Kept. The source change is limited to the `n <= 32` `batch_cholesky` direct-write branch. Same-worker `vmi1227854` affected rows improved and beat same-host NumPy:

| Workload | Baseline FNP ns | Candidate FNP ns | Candidate/Baseline | NumPy median ns | Candidate/NumPy |
|---|---:|---:|---:|---:|---:|
| `2000x16x16` | 572680 | 450154 | 0.786x | 2454268 | 0.183x |
| `1000x32x32` | 1357341 | 971594 | 0.716x | 4061998 | 0.239x |

Guard rows were recorded but not attributed to this branch because the candidate condition is `n <= 32`:

| Workload | Baseline FNP ns | Candidate FNP ns | Candidate/Baseline | Candidate/NumPy | Note |
|---|---:|---:|---:|---:|---|
| `500x64x64` | 3140923 | 4005072 | 1.275x | 0.657x | measured guard loss |
| `64x128x128` | 1887548 | 2179264 | 1.155x | 0.214x | measured guard loss |
| `16x256x256` | 2672825 | 3306358 | 1.237x | 0.219x | measured guard loss |

## Raw Artifacts

- `baseline_batch_cholesky_vmi1227854_pinned_after_revert.txt`: decisive paired baseline on `vmi1227854`.
- `candidate_batch_cholesky_vmi1153651_pinned.txt`: decisive paired candidate file; despite the filename, selected worker was `vmi1227854`.
- `candidate_batch_cholesky_rch_j1_after_reapply.txt`: repeat candidate routing evidence on `vmi1227854`.
- `numpy_batch_cholesky_vmi1227854.txt`: same-host NumPy 2.4.6 comparator.
- `candidate_source.patch`: kept source patch.
- `final_cholesky_bit_test_rch.txt`: focused bit-identity test, pass.
- `final_check_fnp_linalg_all_targets.txt`: per-crate check, pass.
- `final_clippy_fnp_linalg_all_targets.txt`: per-crate clippy with `-D warnings`, pass.
- `final_fmt_check_fnp_linalg.txt`: format check, fail on pre-existing broad drift.
- `ubs_changed_files.txt`: UBS scan, nonzero on pre-existing broad `fnp-linalg/src/lib.rs` inventory.

## Validation

- `rch exec -- cargo test -j 1 -p fnp-linalg batch_cholesky_scratch_matches_per_lane_cholesky_nxn_bits -- --nocapture`: pass on `vmi1149989`.
- `rch exec -- cargo check -j 1 -p fnp-linalg --all-targets`: pass on `vmi1149989`.
- `rch exec -- cargo clippy -j 1 -p fnp-linalg --all-targets -- -D warnings`: pass on `vmi1149989`.
- `cargo fmt -p fnp-linalg -- --check`: fail due pre-existing rustfmt drift; no formatter was run.
- `git diff --check`: pass.
- `ubs crates/fnp-linalg/src/lib.rs docs/NEGATIVE_EVIDENCE.md docs/RELEASE_READINESS_SCORECARD.md tests/artifacts/perf/2026-06-20_linalg_batch_cholesky_direct_write_cod_b/scorecard.md`: nonzero due pre-existing inventory findings in `fnp-linalg/src/lib.rs`.
