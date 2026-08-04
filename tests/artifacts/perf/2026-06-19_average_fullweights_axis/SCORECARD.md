# np.average(axis, weights=<full-shape>) — parallel per-lane weighted average (BlackThrush, 2026-06-19)

## Gap (we LOST)
np.average(a, axis=1, weights=w) with w.shape == a.shape (element-wise weights) was
2.81x SLOWER than numpy on 4096x4096 f64. try_zerocopy_f64_average_axis only accepted
1-D weights (constant denominator), so full-shape weights fell to the cold extract path.

## Lever
For full-shape weights the denominator is PER-LANE: numpy computes
sum_j(a*w)/sum_j(w) per lane. Added a contiguous-last-axis branch that folds each
lane's numerator + denominator with the same 8-accumulator unroll as the 1-D path,
fanned across the rayon pool (read-only &[f64] views, zero copy). `returned=True`
emits the per-lane weight sums. A lane whose weights sum to zero defers to numpy
(ZeroDivisionError). 1-D weights / non-last axis / inner>1 unchanged.

## MEASURED (4096x4096 f64, 64 cores)
| case               | NumPy us | fnp us | fnp/np |
|--------------------|----------|--------|--------|
| returned=False     | ~90500   | ~4900  | **0.054** (18x) |
| returned=True      | ~90500   | ~5000  | **0.055** (18x) |

Before: 2.81x SLOWER. After: 18x FASTER.

## Parity
21/21 differential cases (allclose) across shapes {2-D,3-D} x last-axis x returned
{T,F} + 1-D weights preserved + zero-sum-weights defer. conformance_statistics
average suite (6 tests incl 2 golden-SHA256 + zerodivision) green.
