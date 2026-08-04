# np.ptp along an axis — parallel + branchless (BlackThrush, 2026-06-19)

## Gap targeted (we LOST)
np.ptp(d, axis=0) on a 4096x4096 f64 array was 3.72x SLOWER than NumPy (44.7ms vs
12.0ms) — the largest single loss found this session. The strided (non-last axis)
branch of try_zerocopy_f64_ptp_axis ran a SERIAL row walk with data-dependent
`if v > *m { *m = v }` / `if v < *n` updates that don't vectorize.

## Lever
ptp tracks NaN-presence SEPARATELY (any_nan -> emit NaN), so the running max/min can
use branchless NaN-skipping f64::max/min (autovectorizes) with no behavior change:
when a lane has no NaN it is the true max/min; when it has a NaN the any_nan override
forces NaN regardless of the skipped extreme; and a ±0 extreme can't change a
difference `a - 0`. Then parallelize the read-only &[f64] (sound under the GIL):
- last axis (inner==1): independent lanes -> par_chunks_exact;
- non-last axis: privatized inner-wide (max,min,nan) plane fold -- >=2 outer groups
  fan across groups, a single group (2-D axis=0) privatizes across row-blocks and
  merges planes elementwise.

## MEASURED (4096x4096 f64, 64 cores)
| case    | NumPy us | fnp us | fnp/np |
|---------|----------|--------|--------|
| ptp_ax0 | 12187    | 4901   | **0.40** (2.5x) |
| ptp_ax1 | 18024    | 2827   | **0.16** (6.4x) |

Before: ptp_ax0 3.72x SLOWER. After: 2.5x FASTER. ptp_ax1 0.62x -> 0.16x.

## Parity
168/168 differential cases bit-exact (np.array_equal equal_nan) across 10 shapes x
all axes x {plain, signed-zero, partial-NaN, all-NaN}, plus error-parity for
0-d/empty axes. conformance_diagnostics + conformance_reductions green.
