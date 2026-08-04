# f64 parallel reductions — size-gate to REVERT small-array regressions (BlackThrush, 2026-06-20)

## Regression found (self-audit, REVERT mandate)
The cross-product audit exposed that MY earlier f64 parallel reductions regressed
small/medium arrays (parallel dispatch overhead > work, or serial scalar column fold
< numpy SIMD):
- nanmax/nanmin flat: 256K elements = 3.98x SLOWER (threshold 1<<18 too low).
- nanmax/nanmin axis: 64-256K = 1.4-2.4x (threshold 1<<16 too low).
- f64 ptp axis: 256K-1M = 1.7-5.9x (serial column fold loses to numpy SIMD; parallel
  has plane overhead).
(nanstd/nanvar/nanmean/nansum/nanprod axis WIN at all sizes -- cheap per-lane pairwise;
the earlier 'regression' was an all-NaN-row test artifact triggering the correct bail.)

## Fix
- NANEXTREME_PARALLEL_MIN 1<<18 -> 1<<20 (flat nanmax/nanmin win past ~1M).
- NANEXTREME_AXIS_PARALLEL_MIN 1<<16 -> 1<<20.
- PTP_AXIS_PARALLEL_MIN 1<<16 -> 1<<21; plus a wrapper size-gate delegating f64
  per-axis ptp below 4M to numpy (neither serial nor parallel beats SIMD there).

## MEASURED
nanmax flat: 256K 3.98x->1.08x (parity), 1M 0.57x, 4M 0.08x (win). nanmax_ax1: all
win 0.14-0.58x. ptp_ax0/ax1: <4M parity (1.0-1.1x), >=4M 0.05-0.33x (win). 0 losses.

## Parity
0 fails (nanmax/nanmin/ptp across sizes/axes); conformance green.
