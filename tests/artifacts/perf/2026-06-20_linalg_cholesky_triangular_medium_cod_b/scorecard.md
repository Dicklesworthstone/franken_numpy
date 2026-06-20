# Medium Cholesky Lower-Triangular Threshold Probe

Bead: `franken_numpy-ixs5y.271`
Agent: `YellowElk` / `cod-b`
Crate: `fnp-linalg`
Worker: `vmi1264463`
Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`

## Decision

NO-SHIP. Lowering `SYRK_MID_TRIANGULAR_MIN_TRAIL` from 384 to 64 preserved
the pinned Cholesky mid-panel outputs, but regressed both same-worker
`batch_cholesky` target rows. The source hunk was reverted before commit.

## Performance

| Row | Baseline FNP | Candidate FNP | Candidate/Baseline | NumPy ratio | Verdict |
|---|---:|---:|---:|---:|---|
| `batch_cholesky/shape/64x128x128` | 31,931,504 ns | 66,366,114 ns | 2.078x | not counted; SSH auth blocked same-host Python | loss |
| `batch_cholesky/shape/16x256x256` | 114,361,825 ns | 174,294,182 ns | 1.524x | not counted; SSH auth blocked same-host Python | loss |

Candidate old/new tally: 0 wins / 2 losses / 0 neutral.

## Validation

- Candidate `cholesky_mid_panel` golden tests: 2 passed, 0 failed.
- Invalid first test-filter attempt: ran 0 tests and is not counted.
- Post-revert `batch_cholesky` focused tests: 2 passed, 0 failed, 1 ignored.
- Direct NumPy comparator attempts on `root@38.242.209.154` and
  `ubuntu@38.242.209.154` failed with SSH authentication denial.
- Post-revert `crates/fnp-linalg/src/lib.rs` diff: empty.

## Artifacts

- `baseline_batch_cholesky_hz2.txt`
- `candidate_batch_cholesky_vmi1264463.txt`
- `candidate_cholesky_mid_panel_tests_vmi1264463.txt`
- `candidate_cholesky_tests_vmi1264463.txt`
- `numpy_batch_cholesky_vmi1264463.txt`
- `numpy_batch_cholesky_vmi1264463_ubuntu.txt`
- `post_revert_batch_cholesky_tests_vmi1264463.txt`
