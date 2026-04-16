# FEATURE_PARITY

## Status Legend

- `not_started` — No implementation exists
- `in_progress` — Partial implementation, actively being worked on
- `parity_green` — Feature is implemented, has passing tests in the Rust test suite, and core behavior matches NumPy semantics. Note: this means "implemented and tested" — it does NOT guarantee bit-for-bit oracle verification against NumPy for every edge case. Features with oracle-verified differential coverage are noted in the Evidence column.
- `parity_gap` — Known behavioral divergence from NumPy that needs to be closed

## Parity Matrix

| Feature Family | Status | Current Evidence | Next Gate |
|---|---|---|---|
| Shape/stride/view semantics | parity_green | fixture-driven shape/stride suites green; reshape, transpose, flatten, broadcast, squeeze, expand_dims, swapaxes all implemented | — |
| Broadcasting legality | parity_green | deterministic broadcast cases green; mixed-rank/multi-axis/scalar broadcasting verified | — |
| Dtype promotion/casting | parity_green | scoped promotion table + fixture suite green; copyto casting implemented | — |
| Core math (ufunc) | parity_green | Extensive ufunc coverage green; frexp, modf, gcd, lcm, divmod, isposinf, isneginf, bitwise_count, sort_complex, and related edge-case parity are implemented | — |
| Reductions | parity_green | sum, prod, min, max, mean, var, std, argmin, argmax, cumsum, cumprod, count_nonzero(axis), nansum/nanprod/nanmin/nanmax/nanmean | — |
| Sorting/searching | parity_green | sort, argsort, searchsorted(side,sorter), partition, argpartition, unique, unique_all/counts/inverse/values, where_nonzero, isin(invert) | — |
| Set operations | parity_green | union1d, intersect1d, setdiff1d, setxor1d, in1d | — |
| Indexing | parity_green | take, put, choose, compress, diagonal, triu/tril, indices, nonzero, flatnonzero, unravel_index, ravel_multi_index | — |
| Polynomial: power series | parity_green | polyval, polyder, polyint, polyfit, polymul, polyadd, polysub, polydiv, polyroots | — |
| Polynomial: Chebyshev | parity_green | chebval, chebadd, chebsub, chebmul, chebdiv, chebder, chebint, chebroots, chebfromroots, chebfit, cheb2poly, poly2cheb | — |
| Polynomial: Legendre | parity_green | legval, legadd, legsub, legmul, legdiv, legder, legint, legroots, legfromroots, legfit, leg2poly, poly2leg | — |
| Polynomial: Hermite | parity_green | hermval, hermadd, hermsub, hermmul, hermdiv, hermder, hermint, hermroots, hermfromroots, hermfit, herm2poly, poly2herm (physicist); hermeval, hermeadd, hermesub, hermemul, hermediv, hermeroots, hermefromroots, hermefit, herme2poly, poly2herme (probabilist) | — |
| Polynomial: Laguerre | parity_green | lagval, lagadd, lagsub, lagmul, lagdiv, lagder, lagint, lagroots, lagfromroots, lagfit, lag2poly, poly2lag | — |
| Pad modes | parity_green | constant, edge, reflect, symmetric, wrap, linear_ramp, maximum, minimum, mean, median, empty | — |
| Financial | parity_green | fv, pv, pmt, ppmt, ipmt, nper, rate, npv, irr, mirr | — |
| Statistics | parity_green | histogram, histogram_bin_edges, bincount, digitize, percentile, quantile, median, average, corrcoef, cov | — |
| FFT | parity_green | fft, ifft, fft2, ifft2, fftn, ifftn, rfft, irfft, fftfreq, rfftfreq, fftshift, ifftshift | — |
| Gradient/diff | parity_green | gradient, diff, ediff1d, cross, trapz | — |
| Interpolation | parity_green | interp (1-D linear) | — |
| Windowing | parity_green | bartlett, blackman, hamming, hanning, kaiser | — |
| String arrays | parity_green | add, multiply, upper, lower, capitalize, title, center, ljust, rjust, zfill, strip, lstrip, rstrip, replace, find, rfind, count, startswith, endswith, isnumeric, isalpha, isdigit, isdecimal, str_len, encode, decode, translate, maketrans, partition, rpartition, split, rsplit, join, expandtabs, swapcase | — |
| Masked arrays | parity_green | MaskedArray with reshape, transpose, concatenate, comparison ops, filled, compressed, shrink_mask, anom, fix_invalid, is_masked, make_mask, mask_or | — |
| Datetime/timedelta | parity_green | DatetimeArray, TimedeltaArray with arithmetic, comparison, busday_count, busday_offset, is_busday | — |
| Stride tricks | parity_green | as_strided, sliding_window_view | — |
| numpy.lib.scimath | parity_green | scimath_sqrt, scimath_log, scimath_log2, scimath_log10, scimath_logn, scimath_power, scimath_arccos, scimath_arcsin, scimath_arctanh | — |
| NumPy 2.0+ API | parity_green | unique_all, unique_counts, unique_inverse, unique_values, permuted, matrix_transpose, cumulative_sum, cumulative_prod, trapezoid, unstack, vecdot | — |
| Parameter completeness | parity_green | count_nonzero(axis,keepdims), isin(invert), searchsorted(side,sorter), where(1-arg), sum/prod(initial), copyto(casting), partition/argpartition(axis), packbits/unpackbits(axis) | — |
| Higher-order callable wrappers | parity_green | `frompyfunc` now supports Rust-closure-backed numeric and object-value ufunc construction plus source-evaluated and imported Python callables via `frompyfunc_python*`, and live callable-object/package exposure via `fnp-python::frompyfunc`, with broadcasting, nested object outputs, multi-output, and NumPy parity coverage; existing `vectorize*` helpers remain green | — |
| Linalg | parity_green | solve, det, inv, eig, svd, qr, cholesky, lstsq, norm, matrix_rank, matrix_power, multi_dot, tensorsolve, tensorinv, pinv, cond, slogdet, and funm are implemented with oracle and regression coverage green | — |
| Random (numpy.random) | parity_green | PCG64DXSM generator with oracle-verified distributions; Lemire bounded integers + buffered uint32; BTPE binomial + inversion; HRUA hypergeometric + direct; PTRS Poisson + multiplicative; NumPy-exact gamma; zipf with Umin clamping; oracle and reproducibility coverage green | — |
| I/O (npy/npz) | parity_green | load, save, savez, savez_compressed, loadtxt, savetxt, genfromtxt, fromfile, tofile, and array2string implemented; DEFLATE compression and oracle format coverage green | — |
| Conformance harness | parity_green | Differential corpus, metamorphic suite, adversarial fuzzing, oracle validation, and P2C evidence packets are all green | — |
| Contract schema + artifact topology | parity_green | `phase2c-contract-v1` locked; packet readiness validator green and enforced in CI across `FNP-P2C-001`..`FNP-P2C-009` | — |
| RaptorQ artifact durability | parity_green | sidecar + scrub + decode proof artifacts generated and enforced by the G8 CI gate | — |

## Test Coverage Summary

| Crate | Tests | Description |
|---|---|---|
| fnp-ufunc | 1,794 | Core array operations, math, sorting, polynomials (Chebyshev div/roots/fromroots, Legendre/Hermite/Laguerre div/roots/fromroots edge cases), reductions, oracle tests, linalg bridge, FFT (hfft/ihfft), hermfit/lagfit, masked cov/corrcoef, datetime parsing, gufunc validation, parameter parity (equal_nan, bitorder, mode, endpoint, trim, period, axes, prepend/append, left/right), r_/c_ concat helpers, GridSpec mgrid/ogrid, linspace_retstep, concatenate_flat, einsum coverage, NaN/Inf/signed-zero edge cases (maximum, heaviside, logaddexp, floor_divide, remainder, clip, divmod, cummin/cummax), NaN set-op parity, behavioral edge cases |
| fnp-ndarray | 87 | Shape legality, stride calculus, broadcast contracts, overlap detection, multi-axis negative strides, broadcast/reshape/stride edge cases, F-order, required_view_nbytes |
| fnp-linalg | 235 | Linear algebra decompositions, solvers, norms, batch ops (det/inv/solve/trace), 16 NumPy oracle tests, extreme-scale regression, non-finite parity (cond_p, cross_product 2D, NaN/Inf propagation), rectangular MxN norm/cond support, cond_mxn NaN handling |
| fnp-random | 196 | RNG distributions with statistical conformance coverage, permuted (1D/2D/axis/deterministic), seeding, reproducibility, large-n binomial/multinomial |
| fnp-iter | 110 | Transfer-loop selector, NDIter traversal/broadcast/overlap contracts, stateful `Nditer` wrapper (`iterindex`/`multi_index`/reset/seek/external-loop chunks), flatiter indexing/assignment, ndindex/ndenumerate iterators |
| fnp-io | 176 | NPY/NPZ read/write, text formats, compression, 7 format oracle tests, genfromtxt_full, fromfile_text/tofile_text |
| fnp-conformance | 144 | Differential parity, metamorphic identities, adversarial fuzzing, witness stability, matmul conformance |
| fnp-dtype | 124 | Dtype taxonomy, promotion table (all 324 pairs explicit), cast policy primitives, NumPy byte-width parsing |
| fnp-python | 9 | PyO3 package surface coverage for `frompyfunc`, `vectorize`, and `digitize`, including live callable parity, large-uint64 bridge coverage, and module export wiring |
| fnp-runtime | 54 | Mode split, fail-closed decoding, override-audit gate, risk-aware decision engine, evidence ledger |
| **Total** | **2,923** | |

## Remaining Gaps (Python-specific, low priority)

1. Expanded Python package surface beyond `nditer`, `frompyfunc`, `vectorize`, and `digitize` — `fnp-python` now exposes `PyNditer`, `frompyfunc`, `vectorize`, and `digitize`, but broader Python-facing packaging and FFI coverage for additional NumPy APIs is still incomplete
2. Expanded CI matrix for alternate oracle environments and longer-horizon benchmark trend regression

## Intentional Design Decisions

1. **`register_custom_loop()` fail-closed stub** (`fnp-ufunc`): Always returns `Err` directing callers to `UFuncLoopRegistry::register()`. This is deliberate — the registry-based API is the canonical entry point; the standalone function exists only for discovery and always rejects registration.
2. **`empty()` / `empty_like()` delegate to `zeros()`**: In safe Rust (`#![forbid(unsafe_code)]`), there is no uninitialized memory. These functions correctly zero-fill, matching NumPy's observed behavior for freshly allocated arrays.
3. **Dtype promotion exhaustive match**: All 324 DType pairs are explicitly handled in `promote()` with no catch-all fallback. The compiler enforces exhaustiveness — adding a new DType variant will cause a compile error until promotion rules are added for it.
4. **Oracle pure-Python fallback**: When real NumPy is unavailable, `fnp-conformance` falls back to a simplified Python reimplementation. Set `FNP_REQUIRE_REAL_NUMPY_ORACLE=1` to enforce real NumPy. The `oracle_source` field in capture results records which oracle was used.

## API Surface Inventory

### Implemented (non-exhaustive highlights)

**Array creation**: zeros, ones, empty, full, arange, linspace, logspace, geomspace, eye, identity, diag, meshgrid, mgrid, ogrid, fromfunction, frombuffer, fromfile, copy, asarray, array

**Shape manipulation**: reshape, ravel, flatten, transpose, swapaxes, expand_dims, squeeze, broadcast_to, broadcast_arrays, broadcast_shapes, concatenate, stack, vstack, hstack, dstack, split, array_split, tile, repeat, pad, append, delete, insert, roll, flip, fliplr, flipud, rot90, moveaxis, rollaxis, resize, trim_zeros, column_stack, r_, c_

**Math (unary)**: abs, negative, positive, sign, sqrt, square, cbrt, exp, exp2, expm1, log, log2, log10, log1p, sin, cos, tan, arcsin, arccos, arctan, sinh, cosh, tanh, arcsinh, arccosh, arctanh, degrees, radians, floor, ceil, rint, trunc, round, reciprocal, spacing, fabs, signbit, isnan, isinf, isfinite, logical_not, bitwise_not

**Math (binary)**: add, subtract, multiply, divide, floor_divide, remainder, power, float_power, fmod, arctan2, copysign, heaviside, nextafter, fmax, fmin, logaddexp, logaddexp2, ldexp, hypot, gcd, lcm, bitwise_and, bitwise_or, bitwise_xor, logical_and, logical_or, logical_xor, equal, not_equal, less, less_equal, greater, greater_equal

**Math (special)**: frexp, modf, divmod, isposinf, isneginf, bitwise_count, sort_complex, clip, where, copyto, sinc, unwrap, conj, real, imag, real_if_close, angle

**Reductions**: sum, prod, min, max, mean, var, std, argmin, argmax, cumsum, cumprod, all, any, count_nonzero, nansum, nanprod, nanmin, nanmax, nanmean, nanstd, nanvar, nanargmin, nanargmax, nancumsum, nancumprod, nanpercentile, nanquantile, ptp

**Sorting/searching**: sort, argsort, searchsorted, partition, argpartition, unique, unique_all, unique_counts, unique_inverse, unique_values, nonzero, flatnonzero, where_nonzero, argwhere, isin, extract, place, select, piecewise

**Set operations**: union1d, intersect1d, setdiff1d, setxor1d, in1d

**Polynomials**: polyval, polyfit, polyder, polyint, polymul, polyadd, polysub, polydiv, polyroots, chebval, chebadd, chebsub, chebmul, chebdiv, chebder, chebint, chebroots, chebfromroots, chebfit, cheb2poly, poly2cheb, legval, legder, legint, legfit, hermval, hermeval, hermder, hermint, hermfit, hermefit, lagval, lagder, lagint, lagfit

**Financial**: fv, pv, pmt, ppmt, ipmt, nper, rate, npv, irr, mirr

**Signal processing**: convolve, correlate, convolve2d, correlate2d

**Statistics**: histogram, histogram2d, histogramdd, histogram_bin_edges, bincount, digitize, percentile, quantile, median, average, corrcoef, cov

**String ops**: 33 numpy.char functions

**I/O**: load, save, savez, savez_compressed, loadtxt, savetxt, genfromtxt, fromfile, tofile, array2string

**Linear algebra**: dot, vdot, cross, matmul, inner, outer, tensordot, einsum, einsum_path, kron, solve, det, inv, eig, eigh, eigvals, eigvalsh, svd, qr, cholesky, lstsq, norm, matrix_rank, matrix_power, multi_dot, tensorsolve, tensorinv, pinv, cond, slogdet, funm, solve_triangular, block

**Indexing helpers**: ix_, mgrid, ogrid, indices, unravel_index, ravel_multi_index, take, put, choose, compress, diagonal, triu, tril

**Random**: PCG64DXSM-backed distribution coverage plus permutation and state helpers

**Scimath**: sqrt, log, log2, log10, logn, power, arccos, arcsin, arctanh (complex-aware)
