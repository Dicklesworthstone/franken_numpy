# np.nanstd / np.nanvar along last axis — zero-copy per-lane pairwise + parallel (BlackThrush, 2026-06-19)

## Gap targeted (we LOST)
np.nanstd(d, axis=1) / np.nanvar(d, axis=1) on 4096x4096 f64 was 1.40x SLOWER than
NumPy (~208ms vs ~149ms). Only axis=None had a fast path; an explicit axis fell to
the COLD extract → native nanstd path (128MB copy of the whole array, then native).

## Lever
Added try_zerocopy_f64_nanvar_axis for the contiguous LAST axis: each lane is an
independent 1-D reduction — exactly what numpy's per-lane nanvar does — so the SAME
bit-exact pairwise nansum/count + pairwise sum-of-squared-deviations helpers used by
the axis=None flat fast path (compute_f64_nanvar_flat) reproduce numpy's result per
lane. Parallelized via par_chunks_exact over lanes (the lane's &[f64] is viewed back
as the ReadOnlyCell slice the pairwise helpers take — a single-threaded local temp).
Any lane with count <= ddof defers the whole call to numpy (DoF<=0 warning + NaN).
take_sqrt selects nanstd vs nanvar; wired into both. Non-last axis / non-contiguous
/ non-f64 / keepdims defer.

## MEASURED (4096x4096 f64, 64 cores)
| case             | NumPy us | fnp us | fnp/np |
|------------------|----------|--------|--------|
| nanstd_ax1       | 152889   | 2780   | **0.018** (55x) |
| nanvar_ax1       | 148543   | 3432   | **0.023** (43x) |
| nanstd_ax1_nonan | 150297   | 2856   | **0.019** (53x) |

Before: 1.40x SLOWER. After: 43-55x FASTER. (numpy builds a full |x-mean|^2 128MB
temp + nanmean; the per-lane pairwise fold has no temps and parallelizes.)
nanstd_ax0 (non-last, native path) already 0.60x — left as is.

## Parity
84/84 differential cases BIT-EXACT (np.array_equal equal_nan) across 7 shapes x
last-axis x ddof {0,1,2} x {plain, partial-NaN} x {nanstd, nanvar}, plus error
parity. conformance_nan_funcs nanstd/nanvar tests green.
