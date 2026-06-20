# int min/max/ptp — size-gated narrow-or-small delegate (BlackThrush, 2026-06-20)

## Refinement (REVERT a small-array regression)
The prior int min/max/ptp parallel native fold (054790ff/fb3fd75c) parallelized for any
array >= 64K elements. But numpy's SIMD beats a scalar parallel fold for WIDE ints
below ~8M elements (the 64-core bandwidth edge hasn't overtaken SIMD's lanes-per-
instruction yet) AND the non-last privatized inner-wide plane carries per-thread Vec +
merge overhead. Result: small/medium wide-int reductions were 1.5-11x SLOWER.

## Lever
Caller-level SIZE + dtype gate: narrow ints (any size) AND wide ints below 1<<23 (~8.4M)
elements delegate to numpy's SIMD (parity). Only large wide ints (>=8.4M) fall through
to the parallel native fold (where it wins 3-5x).

## MEASURED (int32)
n=1024 (1M): max/ptp all axes ~1.0-1.08x (parity, delegated). n=2048 (4M): ~1.0x
(parity). n=4096 (16M): max_ax0 0.40x, max_ax1 0.25x, max_flat 0.20x, ptp_ax0 0.23x
(3-5x WIN). 0 losses across all sizes (was 1.5-11x for small/medium).

## Parity
6 int dtypes x 6 shapes x all axes x {max,min,ptp} 0 fails; conformance green.
