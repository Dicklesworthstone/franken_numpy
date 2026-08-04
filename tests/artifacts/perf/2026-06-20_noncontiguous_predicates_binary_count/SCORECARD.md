# Non-contiguous predicates/rounding/sign/count_nonzero/binary-transcendentals — delegate (BlackThrush, 2026-06-20)

## Gap (we LOST) — deeper layer of the non-contiguous class
Predicates (isfinite/isinf/signbit), floor/ceil/trunc, sign, count_nonzero, and the
binary transcendentals (hypot/arctan2/logaddexp/mod) on transposed/strided ndarrays
bailed the contiguous-only zero-copy paths into the cold extract -> rebuild
(transpose-copy): count_nonzero(d.T) 342x, isfinite/isinf(d.T) ~105x, signbit(d.T) 55x,
count_nonzero(d.T,axis=1) 36x, ceil/trunc(d.T) ~6.2x, sign(d.T) 5.6x, hypot(d.T) 4.8x,
arctan2 3.4x, logaddexp 3.0x, mod 2.4x SLOWER than numpy's strided ufunc.

## Lever
Added a shared `noncontiguous_ndarray(numpy, x)` helper and delegated non-c-contiguous
operands to numpy at each chokepoint: native_rounding_unary (floor/ceil/trunc),
isfinite/isinf/signbit native predicates, sign, count_nonzero, hypot, logaddexp,
native_binary_arctan2_or_passthrough, native_binary_remainder_or_passthrough.
Contiguous fast paths untouched.

## MEASURED (4096x4096 f64, 64 cores) — all before→after
count_nonzero(d.T) 342x->1.07x; isfinite(d.T) 105x->1.00x; isinf 106x->~1x;
signbit 55x->1.04x; count_nonzero(d.T,axis=1) 36x->1.00x; ceil/trunc 6.2x->1.00x;
sign 5.6x->0.99x; hypot 4.8x->1.00x; arctan2 3.4x->1.00x; logaddexp 3.0x->~1x;
mod 2.4x->1.00x. Before: 2.4-342x SLOWER. After: parity. Contiguous unchanged.

## Parity
transposed/sliced/with-inf/nan x {isfinite,isinf,signbit,sign,floor,ceil,trunc,
count_nonzero,count_nonzero-axis,hypot,logaddexp,arctan2,mod} 0 fails (array_equal
equal_nan). conformance_special_math + conformance_count_nonzero_zerocopy green.
