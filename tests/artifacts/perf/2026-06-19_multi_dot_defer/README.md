# 2026-06-19 multi_dot: defer f64 chain to numpy — SHIP (win)

Agent: CobaltForge (claude-opus-4-8). Host: thinkstation1 (Threadripper PRO 5975WX),
load ~6 during measurement. NumPy 2.4.3 scipy-openblas64. `.probe` cdylib differential
+ rch conformance.

## Gap
`np.linalg.multi_dot` (optimally-ordered chain of matmuls) measured **1.7–7.6x slower**
than numpy across every chain shape. fnp's native `UFuncArray::multi_dot` DOES compute
the optimal DP parenthesization (verified — same order as numpy), but evaluates each
product through the register-tiled native GEMM, which is the known native-GEMM-vs-
OpenBLAS gap. So the loss is purely the per-product BLAS gap multiplied across the
chain. Integer/complex chains already defer to numpy; only the f64 path routed native.

## Measured (native vs numpy, before)
| chain | tiny3 | small3 | mixed4 | tall_chain | large3 |
|---|---|---|---|---|---|
| native/numpy | 4.53x | 4.11x | 7.61x | 6.64x | 1.69x |

Native loses at ALL sizes (no winning regime — unlike matrix_rank's tiny-matrix
window). All results allclose to numpy.

## Fix
multi_dot's only conformance contract is allclose (multi_dot_chain) and integer
array_equal (multi_dot_basic, already deferred) — NO bit-exact golden pins the f64
output stream. So defer the f64 path to numpy too (one-line: the wrapper now always
`fallback()`s after the out check). numpy computes the same optimal order and runs each
product through threaded BLAS. The native `UFuncArray::multi_dot` stays for direct Rust
callers + its own fnp-ufunc tests.

## Result (after, vs numpy)
| chain | tiny3 | mixed4 | large3 |
|---|---|---|---|
| fnp/numpy | 1.23x | 0.93x | 0.90x |

mixed4 7.6x→0.93x, large3 1.7x→0.90x (parity). tiny3 1.23x is residual pyfunction
dispatch overhead (~1µs absolute, was 4.5x). Results match numpy (allclose) incl
integer chains and 1-D vector endpoints. Conformance: multi_dot_basic + multi_dot_chain
green via rch. Hunk rustfmt-clean (file has pre-existing fmt drift, none in my region).

## Ledger row
- WIN: multi_dot f64 defer — 1.7–7.6x loss → parity, exact-order numpy parity. Shipped.
- Reusable: native-linalg-loses-where-sibling-defers (allclose/integer output, no
  bit-golden) — now applied to matrix_rank (integer) and multi_dot (allclose float).
