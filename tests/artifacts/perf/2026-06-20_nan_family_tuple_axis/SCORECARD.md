# nan-family multi-axis (tuple) reductions — delegate to numpy (BlackThrush, 2026-06-20)

## Gap (we LOST)
nansum/nanmean/nanmax/nanmin/nanprod/nanvar/nanstd with a TUPLE axis (e.g. axis=(0,1))
on a (512,512,16) array was 1.4-2.9x SLOWER than numpy (nanmax/nanmin worst). The
single-int/None fast paths didn't cover tuple axes, so they extracted the WHOLE array
through the f64 bridge, THEN the axis re-parse fell back to numpy — paying the cold
extract for nothing.

## Lever
The functions already fall back to numpy on multi-axis (axes.len()!=1), just after a
wasteful extract. Delegate any tuple-axis nan reduction to numpy UP FRONT (before the
extract). Single-int + None zero-copy fast paths (up to 12x faster) untouched.

## MEASURED (512x512x16 f64, axis=(0,1))
nansum 1.74x->0.99x; nanmean 1.52x->~1x; nanmax 2.90x->1.00x; nanmin 2.93x->~1x;
nanprod 1.78x->1.01x; nanvar 1.42x->~1x; nanstd 1.41x->~1x. Single-axis nansum 0.08x
(unchanged, 12x faster). LOSS -> parity.

## Parity
2 shapes x 6 axes (tuple/single/None) x keepdims x 7 functions 0 fails (allclose
equal_nan); conformance_nan_funcs green.
