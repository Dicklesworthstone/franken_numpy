# f32 + complex128 ufunc sweep vs NumPy — all dominated (cod-b / BlackThrush, 2026-06-21)

Hunted for an UNCONTENDED winnable perf lever after roll/compress (fnp-python) were
found contended (a peer is applying my roll memcpy recipe `d8ffffb9` live, uncommitted)
and the f64 ufunc surface was already swept (only sqrt loses = forbid(unsafe) no-ship).
Tested two unchecked classes where a scalar/widening impl would lose to NumPy's native
SIMD. Result: NO loss — fnp dominates both. (min-of-9, load~20, OMP=1.)

## f32 ufuncs (NumPy uses 8-wide AVX2 for f32) — all parity/win, dtype correct
exp 1.02 · log 1.00 · sin 1.00 · cos 1.03 · tanh 1.00 · **sqrt 0.95** · expm1 1.03 ·
cbrt 1.04 · arctan 1.00 · add 0.99 · multiply 0.91 · divide 1.00 · hypot 0.98 ·
arctan2 1.00 · maximum 1.06 · power 1.02. All return float32 (no f64 widening).
Note: f32 sqrt is parity (0.95x) unlike f64 sqrt (1.5x) — f32's lighter memory makes
the forbid(unsafe) zero-init tax proportionally smaller.

## complex128 ufuncs — all parity/win
abs 1.01 · conj 0.94 · exp 1.00 · log 1.00 · sqrt 1.00 · square 1.11 · sin 1.01 ·
angle 0.99 · add 1.04 · multiply 1.10 · divide 1.01.
FALSE ALARM: real/imag "2.3-2.4x" are O(1) view ops (~0.5us) — dispatch noise, not a
loss (same as flip).

## Conclusion
The full ufunc surface (f64 [prior] + f32 + complex128) is DOMINATED. Rules out the
f32-widening and complex-scalar loss hypotheses — don't re-hunt. The only open ufunc
loss is f64 sqrt (forbid(unsafe) zero-init no-ship). Remaining vs-numpy losses are
roll/compress (fnp-python, contended — roll being applied via my recipe) and the
batched inv/solve + spectral kernel no-ships. No uncontended winnable lever this turn.
