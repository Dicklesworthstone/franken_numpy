# np.count_nonzero(axis=tuple) — delegate multi-axis to numpy (BlackThrush, 2026-06-20)

## Gap (we LOST)
count_nonzero(b, axis=(0,1)) on a bool (512,512,16) array was 8.56x SLOWER than numpy.
The single-int/None fast path didn't cover tuple (multi-axis) reductions, so they fell
to extract_numeric_array which bridges bool->f64 (8x memory blowup) then counts.

## Lever
numpy's multi-axis count_nonzero is optimized; the bool->f64 bridge can't beat it.
Delegate any tuple-axis count_nonzero to numpy (fallback) before the extract. The
single-int + None zero-copy fast paths (0.5x faster) are untouched.

## MEASURED (bool 512x512x16)
count_nonzero(axis=(0,1)) 8.56x->1.00x; axis=(1,2) 0.95x; axis=1 0.52x (unchanged);
axis=None 0.58x (unchanged). LOSS -> parity.

## Parity
3 shapes x {bool,f64,i32} x 5 tuple/single axes x keepdims 0 fails (array_equal);
conformance_count_nonzero_zerocopy green.
