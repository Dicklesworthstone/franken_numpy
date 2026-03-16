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
| Core math (ufunc) | parity_green | 999 tests passing; frexp, modf, gcd, lcm, divmod, isposinf, isneginf, bitwise_count, sort_complex all implemented | — |
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
| NumPy 2.0 API | parity_green | unique_all, unique_counts, unique_inverse, unique_values, permuted | — |
| Parameter completeness | parity_green | count_nonzero(axis,keepdims), isin(invert), searchsorted(side,sorter), where(1-arg), sum/prod(initial), copyto(casting), partition/argpartition(axis), packbits/unpackbits(axis) | — |
| Linalg | parity_green | solve, det, inv, eig, svd, qr, cholesky, lstsq, norm, matrix_rank, matrix_power, multi_dot, tensorsolve, tensorinv, pinv, cond, slogdet, funm; 147 tests | — |
| Random (numpy.random) | parity_green | PCG64DXSM generator with 39 oracle-verified distributions; Lemire bounded integers + buffered uint32; BTPE binomial + inversion; HRUA hypergeometric + direct; PTRS Poisson + multiplicative; NumPy-exact gamma (shape<1 rejection, shape=1 exponential, Marsaglia-Tsang); zipf with Umin clamping; 39/39 oracle tests passing; 178 tests | — |
| I/O (npy/npz) | parity_green | load, save, savez, savez_compressed, loadtxt, savetxt, genfromtxt, fromfile, tofile, array2string; DEFLATE compression; 76 tests | — |
| Conformance harness | parity_green | 83 tests: differential corpus (40 unary, 20 binary, 30 reduction combos), metamorphic suite (13 algebraic identities), adversarial fuzzing (12 edge cases), oracle validation (5 tests) | — |
| Contract schema + artifact topology | in_progress | `phase2c-contract-v1` locked; packet readiness validator green | populate packet artifact directories in CI |
| RaptorQ artifact durability | in_progress | sidecar + scrub + decode proof artifacts generated | integrate into CI |

## Test Coverage Summary

| Crate | Tests | Description |
|---|---|---|
| fnp-ufunc | 1,121 | Core array operations, math, sorting, polynomials, e2e pipelines |
| fnp-ndarray | 136 | Shape legality, stride calculus, broadcast contracts |
| fnp-linalg | 181 | Linear algebra decompositions, solvers, norms |
| fnp-random | 178 | RNG distributions (39 oracle-verified), seeding, reproducibility |
| fnp-iter | 13 | Transfer semantics, overlap detection, flatiter/nditer |
| fnp-io | 44 | NPY/NPZ read/write, text formats, compression |
| fnp-conformance | 94 | Differential parity, metamorphic identities, adversarial fuzzing, witness stability |
| fnp-dtype | 114 | Dtype taxonomy, promotion table, cast policy primitives |
| fnp-runtime | 12 | Mode split, fail-closed decoding, override-audit gate |
| **Total** | **1,893** | |

## Remaining Gaps (Python-specific, low priority)

1. `frompyfunc` — requires Python callable protocol (N/A for Rust)
2. `nditer` — Python-level iterator protocol (N/A for Rust)
3. Full CI pipeline wiring for oracle environment, RaptorQ gates, benchmark regression

## API Surface Inventory

### Implemented (non-exhaustive highlights)

**Array creation**: zeros, ones, empty, full, arange, linspace, logspace, geomspace, eye, identity, diag, meshgrid, fromfunction, frombuffer, fromfile, copy, asarray, array

**Shape manipulation**: reshape, ravel, flatten, transpose, swapaxes, expand_dims, squeeze, broadcast_to, broadcast_arrays, concatenate, stack, vstack, hstack, dstack, split, array_split, tile, repeat, pad

**Math (unary)**: abs, negative, positive, sign, sqrt, square, cbrt, exp, exp2, expm1, log, log2, log10, log1p, sin, cos, tan, arcsin, arccos, arctan, sinh, cosh, tanh, arcsinh, arccosh, arctanh, degrees, radians, floor, ceil, rint, trunc, round, reciprocal, spacing, fabs, signbit, isnan, isinf, isfinite, logical_not, bitwise_not

**Math (binary)**: add, subtract, multiply, divide, floor_divide, remainder, power, float_power, fmod, arctan2, copysign, heaviside, nextafter, fmax, fmin, logaddexp, logaddexp2, ldexp, hypot, gcd, lcm, bitwise_and, bitwise_or, bitwise_xor, logical_and, logical_or, logical_xor, equal, not_equal, less, less_equal, greater, greater_equal

**Math (special)**: frexp, modf, divmod, isposinf, isneginf, bitwise_count, sort_complex, clip, where, copyto

**Reductions**: sum, prod, min, max, mean, var, std, argmin, argmax, cumsum, cumprod, all, any, count_nonzero, nansum, nanprod, nanmin, nanmax, nanmean, nanstd, nanvar, ptp

**Sorting/searching**: sort, argsort, searchsorted, partition, argpartition, unique, unique_all, unique_counts, unique_inverse, unique_values, nonzero, flatnonzero, where_nonzero, argwhere, isin

**Set operations**: union1d, intersect1d, setdiff1d, setxor1d, in1d

**Polynomials**: polyval, polyfit, polyder, polyint, polymul, polyadd, polysub, polydiv, polyroots, chebval, chebadd, chebsub, chebmul, chebdiv, chebder, chebint, chebroots, chebfromroots, chebfit, cheb2poly, poly2cheb, legval, legder, legint, legfit, hermval, hermeval, hermder, hermint, lagval, lagder, lagint

**Financial**: fv, pv, pmt, ppmt, ipmt, nper, rate, npv, irr, mirr

**Statistics**: histogram, histogram_bin_edges, bincount, digitize, percentile, quantile, median, average, corrcoef, cov

**String ops**: 33 numpy.char functions

**I/O**: load, save, savez, savez_compressed, loadtxt, savetxt, genfromtxt, fromfile, tofile, array2string

**Linalg**: solve, det, inv, eig, svd, qr, cholesky, lstsq, norm, matrix_rank, matrix_power, multi_dot, tensorsolve, tensorinv, pinv, cond, slogdet, funm

**Random**: 39 distributions via PCG64

**Scimath**: sqrt, log, log2, log10, power, arccos, arcsin, arctanh (complex-aware)
