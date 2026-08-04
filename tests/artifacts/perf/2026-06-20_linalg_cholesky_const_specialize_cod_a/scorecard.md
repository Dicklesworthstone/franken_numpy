# fnp-linalg Cholesky const-specialization probe

Date: 2026-06-20
Agent: BlackThrush / cod-a
Parent bead: franken_numpy-ixs5y
Target dir: CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a
Decision: NO-SHIP, source reverted

## Lever

The rejected candidate added `cholesky_unblocked_const<const N>` for
N=16/32/64/100 and routed `cholesky_nxn` through it for those fixed sizes.

## Same-Worker Rust Criterion

Worker: `vmi1149989`.

| Row | Baseline | Candidate | Candidate/Baseline | NumPy ratio | Outcome |
|---|---:|---:|---:|---:|---|
| `cholesky_nxn/size/16` | 1,152 ns | 1,084 ns | 0.941x | not rerun | neutral/small win |
| `cholesky_nxn/size/32` | 5,597 ns | 5,142 ns | 0.919x | not rerun | neutral/small win |
| `cholesky_nxn/size/64` | 32,431 ns | 30,845 ns | 0.951x | not rerun | neutral/small win |
| `cholesky_nxn/size/128` | 226,889 ns | 119,611 ns | 0.527x | not rerun | noisy non-owned row |
| `cholesky_nxn/size/256` | 1,228,708 ns | 695,743 ns | 0.566x | not rerun | noisy non-owned row |
| `cholesky_nxn/size/512` | 8,866,316 ns | 5,587,315 ns | 0.630x | not rerun | noisy non-owned row |
| `cholesky_nxn/size/768` | 20,093,048 ns | 11,838,452 ns | 0.589x | not rerun | noisy non-owned row |
| `batch_cholesky/64x128x128` | 4,237,881 ns | 2,920,691 ns | 0.689x | not rerun | noisy non-owned row |
| `batch_cholesky/16x256x256` | 5,548,820 ns | 4,049,209 ns | 0.730x | not rerun | noisy non-owned row |

Owned target rows vs old FNP: 3 small wins / 0 losses / 0 neutral.
Owned target rows vs NumPy: 0 wins / 0 losses / 3 not measured.
Broad rows were treated as neutral/noisy because the candidate did not own their route.

## Validation

- PASS while candidate existed: `rch exec -- cargo test -p fnp-linalg cholesky_const_specializations_match_dynamic_scalar_reference_bits -- --nocapture`
- PASS while candidate existed: `rch exec -- cargo check -p fnp-linalg --all-targets`
- REVERTED: no Cholesky production hunk remains in the final commit.
