# 2026-06-20 batch Cholesky ordered-dot reject

Bead: `franken_numpy-ixs5y.270`
Agent: `YellowElk` / `cod-b`
Crate: `fnp-linalg`
Target dir: `/data/projects/.rch-targets/franken_numpy-cod-b`

## Candidate

Extend the ordered 4-wide scalar dot helper from the current small-N unblocked Cholesky path into blocked Cholesky diagonal and panel loops. The candidate preserved scalar operation order but added wider helper calls in the medium batch path.

## Commands

```bash
RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg 'batch_cholesky' -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-linalg cholesky_ -- --nocapture
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg 'batch_cholesky' -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo check -p fnp-linalg --all-targets
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo clippy -p fnp-linalg --all-targets -- -D warnings
cargo fmt -p fnp-linalg -- --check
```

RCH selected `vmi1153651` for the counted baseline and candidate Criterion runs.

## Results

| Workload | Baseline FNP ns | Candidate FNP ns | Candidate/Baseline | Verdict |
|---|---:|---:|---:|---|
| `batch_cholesky/shape/64x128x128` | 14844832 | 13567919 | 0.914x | mixed win |
| `batch_cholesky/shape/16x256x256` | 20811194 | 22141744 | 1.064x | loss |

## NumPy comparator

Direct Python capture on `vmi1153651` failed with SSH authentication denial. `rch exec -- python3` ran locally on `thinkstation1` with Python 3.13.7, so no same-host NumPy ratio is counted for this attempt.

## Correctness

`cargo test -p fnp-linalg cholesky_ -- --nocapture` passed on `vmi1153651`; 21 unit tests passed, 2 ignored, 303 filtered, and the filtered Cholesky golden/metamorphic integration tests passed.

## Validation

`cargo check -p fnp-linalg --all-targets` passed on RCH worker `vmi1149989`.

`cargo clippy -p fnp-linalg --all-targets -- -D warnings` passed on RCH worker `vmi1227854`.

`cargo fmt -p fnp-linalg -- --check` reported existing formatting drift across `fnp-linalg` benches/examples and `src/lib.rs`; no formatter was run because that would rewrite unrelated files.

## Decision

Rejected and reverted. The smaller batch row improved, but the larger row regressed and the same-host NumPy comparator was unavailable. The production source was restored to HEAD after the experiment.

Retry only with a deeper kernel change: generated size-specialized microkernel, real safe SIMD dot primitive with preserved bit contracts, or a blocked/batched panel kernel that has same-host NumPy capture and zero target-row regressions.
