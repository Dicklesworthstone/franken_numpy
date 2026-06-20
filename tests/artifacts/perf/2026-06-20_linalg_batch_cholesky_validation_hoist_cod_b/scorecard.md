# Batch Cholesky Validation-Hoist No-Ship

Run date: 2026-06-20
Agent: YellowElk / cod-b
Parent bead: `franken_numpy-ixs5y`
Crate: `fnp-linalg`
Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`

## Candidate

Hoist `batch_cholesky` finite validation to one full-batch scan. If the whole
input batch is finite, call internal finite-unchecked Cholesky helpers per lane;
if any non-finite value exists, fall back to the original checked per-lane path
to preserve error ordering. The candidate preserved scalar arithmetic order and
did not change public `cholesky_nxn` validation.

## Same-Worker Gate

Worker: `vmi1153651`

| Row | Baseline | Candidate | Candidate/Baseline | Verdict |
|---|---:|---:|---:|---|
| `batch_cholesky/shape/64x128x128` | 18,102,653 ns | 20,809,809 ns | 1.150x | loss |
| `batch_cholesky/shape/16x256x256` | 12,748,878 ns | 44,004,085 ns | 3.451x | loss |

Win/loss/neutral: 0/2/0. Candidate source was reverted.

## Validation

- Candidate compile check passed: `candidate_cargo_check_fnp_linalg_lib.txt`.
- Post-revert focused tests passed: `post_revert_test_batch_cholesky.txt`.
- Post-revert release build passed: `post_revert_build_fnp_linalg_release.txt`.

## Decision

No-ship. Validation hoisting did not address the scalar per-lane Cholesky wall
identified by prior NumPy evidence, and it regressed the same-worker Rust broad
gate before earning another NumPy comparator run.

Retry only with a structurally different Cholesky kernel: blocked/batched panels
or a dot-product kernel that preserves the project Cholesky bit contracts and
proves medium rows plus n>=128 rows in the same run window.
