# FEATURE_PARITY

## Status Legend

- `not_started`
- `in_progress`
- `parity_green`
- `parity_gap`

## Parity Matrix

| Feature Family | Status | Current Evidence | Next Gate |
|---|---|---|---|
| Shape/stride/view semantics | parity_green | fixture-driven shape/stride suites green; reshape, transpose, flatten, broadcast, squeeze, expand_dims, swapaxes all implemented | — |
| Broadcasting legality | parity_green | deterministic broadcast cases green; mixed-rank/multi-axis/scalar broadcasting verified | — |
| Dtype promotion/casting | parity_green | scoped promotion table + fixture suite green; copyto casting implemented | — |
| Core math (ufunc) | parity_green | 1,249 tests passing; frexp, modf, gcd, lcm, divmod, isposinf, isneginf, bitwise_count, sort_complex all implemented | — |
| Reductions | parity_green | sum, prod, min, max, mean, var, std, argmin, argmax, cumsum, cumprod, count_nonzero(axis), nansum/nanprod/nanmin/nanmax/nanmean | — |
| Sorting/searching | parity_green | sort, argsort, searchsorted(side,sorter), partition, argpartition, unique, unique_all/counts/inverse/values, where_nonzero, isin(invert) | — |
| Set operations | parity_green | union1d, intersect1d, setdiff1d, setxor1d, in1d | — |
| Indexing | parity_green | take, put, choose, compress, diagonal, triu/tril, indices, nonzero, flatnonzero, unravel_index, ravel_multi_index | — |
| Polynomial: power series | parity_green | polyval, polyder, polyint, polyfit, polymul, polyadd, polysub, polydiv, polyroots | — |
| Polynomial: Chebyshev | parity_green | chebval, chebadd, chebsub, chebmul, chebdiv, chebder, chebint, chebroots, chebfromroots, chebfit, cheb2poly, poly2cheb | — |
| Polynomial: Legendre | parity_green | legval, legder, legint, legfit | — |
| Polynomial: Hermite | parity_green | hermval (physicist), hermeval (probabilist), hermder, hermint | — |
| Polynomial: Laguerre | parity_green | lagval, lagder, lagint | — |
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
| numpy.lib.scimath | parity_green | scimath_sqrt, scimath_log, scimath_log2, scimath_log10, scimath_power, scimath_arccos, scimath_arcsin, scimath_arctanh | — |
| NumPy 2.0+ API | parity_green | unique_all, unique_counts, unique_inverse, unique_values, permuted, matrix_transpose, cumulative_sum, cumulative_prod, trapezoid, unstack, vecdot | — |
| Parameter completeness | parity_green | count_nonzero(axis,keepdims), isin(invert), searchsorted(side,sorter), where(1-arg), sum/prod(initial), copyto(casting), partition/argpartition(axis), packbits/unpackbits(axis) | — |
| Linalg | parity_green | solve, det, inv, eig, svd, qr, cholesky, lstsq, norm, matrix_rank, matrix_power, multi_dot, tensorsolve, tensorinv, pinv, cond, slogdet, funm; 199 tests | — |
| Random (numpy.random) | parity_green | PCG64DXSM generator with 40 oracle-verified distributions; Lemire bounded integers + buffered uint32; BTPE binomial + inversion; HRUA hypergeometric + direct; PTRS Poisson + multiplicative; NumPy-exact gamma (shape<1 rejection, shape=1 exponential, Marsaglia-Tsang); zipf with Umin clamping; 40/40 oracle tests passing; 182 tests | — |
| I/O (npy/npz) | parity_green | load, save, savez, savez_compressed, loadtxt, savetxt, genfromtxt, fromfile, tofile, array2string; DEFLATE compression; 143 tests | — |
| Conformance harness | parity_green | 122 tests: differential corpus, metamorphic suite (13 algebraic identities), adversarial fuzzing, oracle validation, P2C evidence packets | — |
| Contract schema + artifact topology | parity_green | `phase2c-contract-v1` locked; packet readiness validator green and enforced in CI across `FNP-P2C-001`..`FNP-P2C-009` | — |
| RaptorQ artifact durability | parity_green | sidecar + scrub + decode proof artifacts generated and enforced by the G8 CI gate | — |

## Test Coverage Summary

| Crate | Tests | Description |
|---|---|---|
| fnp-ufunc | 1,612 | Core array operations, math, sorting, polynomials, reductions, oracle tests, linalg bridge, FFT (hfft/ihfft), hermfit/lagfit, masked cov/corrcoef, datetime parsing, gufunc validation, parameter parity (equal_nan, bitorder, mode, endpoint, trim, period, axes, prepend/append, left/right), r_/c_ concat helpers, GridSpec mgrid/ogrid, linspace_retstep, concatenate_flat, einsum coverage |
| fnp-ndarray | 55 | Shape legality, stride calculus, broadcast contracts, overlap detection, multi-axis negative strides |
| fnp-linalg | 217 | Linear algebra decompositions, solvers, norms, 16 NumPy oracle tests, extreme-scale regression, non-finite parity (cond_p, cross_product 2D, NaN/Inf propagation) |
| fnp-random | 191 | RNG distributions (40 oracle-verified), seeding, reproducibility, large-n binomial/multinomial |
| fnp-iter | 103 | Transfer-loop selector, NDIter traversal/broadcast/overlap contracts, flatiter indexing/assignment, ndindex/ndenumerate iterators |
| fnp-io | 174 | NPY/NPZ read/write, text formats, compression, 7 format oracle tests, genfromtxt_full, fromfile_text/tofile_text |
| fnp-conformance | 117 | Differential parity, metamorphic identities, adversarial fuzzing, witness stability, matmul conformance |
| fnp-dtype | 117 | Dtype taxonomy, promotion table, cast policy primitives, NumPy byte-width parsing |
| fnp-runtime | 54 | Mode split, fail-closed decoding, override-audit gate, Bayesian decision engine, evidence ledger |
| **Total** | **2,669** | |

## Remaining Gaps (Python-specific, low priority)

1. `frompyfunc` — requires Python callable protocol (N/A for Rust)
2. `nditer` — Python-level iterator protocol (N/A for Rust)
3. Expanded CI matrix for alternate oracle environments and longer-horizon benchmark trend regression

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

**Polynomials**: polyval, polyfit, polyder, polyint, polymul, polyadd, polysub, polydiv, polyroots, chebval, chebadd, chebsub, chebmul, chebdiv, chebder, chebint, chebroots, chebfromroots, chebfit, cheb2poly, poly2cheb, legval, legder, legint, legfit, hermval, hermeval, hermder, hermint, lagval, lagder, lagint

**Financial**: fv, pv, pmt, ppmt, ipmt, nper, rate, npv, irr, mirr

**Signal processing**: convolve, correlate, convolve2d, correlate2d

**Statistics**: histogram, histogram2d, histogramdd, histogram_bin_edges, bincount, digitize, percentile, quantile, median, average, corrcoef, cov

**String ops**: 33 numpy.char functions

**I/O**: load, save, savez, savez_compressed, loadtxt, savetxt, genfromtxt, fromfile, tofile, array2string

**Linear algebra**: dot, vdot, cross, matmul, inner, outer, tensordot, einsum, einsum_path, kron, solve, det, inv, eig, eigh, eigvals, eigvalsh, svd, qr, cholesky, lstsq, norm, matrix_rank, matrix_power, multi_dot, tensorsolve, tensorinv, pinv, cond, slogdet, funm, solve_triangular, block

**Indexing helpers**: ix_, mgrid, ogrid, indices, unravel_index, ravel_multi_index, take, put, choose, compress, diagonal, triu, tril

**Random**: 39 statistical distributions via PCG64 plus permutation/state helpers

**Scimath**: sqrt, log, log2, log10, power, arccos, arcsin, arctanh (complex-aware)
