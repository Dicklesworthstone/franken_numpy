# Non-contiguous array_equal/array_equiv/allclose/nextafter/roll — delegate (BlackThrush, 2026-06-20)

## Gap (we LOST)
More of the non-contiguous class: array_equal(d.T) 45x, nextafter(d.T) 5.36x, roll(d.T)
3.45x, allclose(d.T) 2.88x SLOWER than numpy. The comparison-reduction early-exit folds,
the binary ufunc fast path, and roll's block-copy paths all gate c_contiguous and bail
transposed/strided ndarrays into the cold transpose-copy extract.

## Lever
Delegated non-c-contiguous operands to numpy (shared noncontiguous_ndarray helper) at:
array_equal, array_equiv, allclose (after the f64/f32 fast paths), nextafter (before
the binary fast path), and roll (up front, before the block-copy fast paths). Contiguous
fast paths untouched.

## MEASURED (4096x4096 f64)
array_equal(d.T) 45x->1.10x; allclose(d.T) 2.88x->1.01x; nextafter(d.T) 5.36x->0.99x;
roll(d.T) 3.45x->0.99x. Before: 2.9-45x SLOWER. After: parity.

## Parity
same/differ/slice/transposed x {array_equal,array_equiv,allclose,nextafter,roll
(axis None/0/1)} 0 fails. conformance: array_equal 20, allclose 15, roll
(array_transform) 26, nextafter (fp_introspect) 12 — all green.
