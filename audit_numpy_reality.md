# fnp-* reality-check — numpy.__all__ coverage — 2026-04-22

Comparing the Python-facing surface exposed by `fnp_python` against `numpy.__all__` (NumPy 2.x). Produced by CC agent via the `reality-check-for-project` skill. **This is a gap list, not a plan — bridge plan + bead generation is the next tick.**

## Vision (from README.md)

> "Absolute behavioral compatibility with legacy NumPy. Not a subset, not 'inspired by.' The full API, edge cases and all."

The measuring stick is `numpy.__all__` — the 499 names NumPy publishes as its public Python surface.

## Headline numbers

- `numpy.__all__` names: **499**
- Names exposed in `fnp_python` (via `wrap_pyfunction!` + the one `#[pyo3(name="broadcast_shapes")]` rename): **352**
- `numpy.__all__` ∩ `fnp_python`: **216** (43.3% of numpy's top-level surface is reachable as `fnp_python.<name>`)
- `numpy.__all__` \ `fnp_python` (gap): **283** (56.7%)
- Names in `fnp_python` that are NOT in `numpy.__all__`: **136** (flat-namespace versions of `numpy.linalg.*`, `numpy.ma.*`, `numpy.testing.*`, `numpy.lib.recfunctions.*`, `numpy.fft.*`, `numpy.polynomial.*`, plus a small number of internal helpers)

**43% direct coverage of `numpy.__all__` falls short of the README's "full API, edge cases and all" claim.** The shortfall is concentrated in the core numeric surface (ufuncs + fundamental reductions) that every numpy user touches; the *tail* of less-common helpers is actually well covered.

## Important nuance: engine coverage vs Python surface coverage

The gap is **surface-level, not engine-level**:

- `fnp-ufunc/src/lib.rs` exposes **804 `pub fn`** including `add`, `subtract`, `multiply`, `divide`, `sum`, `prod`, `cumsum`, `cumprod`, `mean`, `median`, `std`, `var`, `max`, `min`, `argmax`, `argmin`, `sin`/`cos`/`tan`/`log`/`exp` through the broader ufunc dispatcher, etc.
- The Rust engine implements the numpy semantics for nearly everything on the missing list below.
- The `fnp_python` crate is a parity-oracle surface — by design it *passes through* to real numpy for differential testing while the pure-Rust engine is verified. That wave has shipped 350+ wrappers but targets the less-common tail because common ufuncs are invoked through the ndarray dispatcher path, not named top-level wrappers.

**If the vision is "users can `import fnp_python` and call everything numpy exposes," we are at 43%.**

## Gap breakdown (283 missing names)

### Category 1 — Core ufuncs NOT surfaced as `fnp_python.<name>` (70)

These are the biggest user-visible gap. Each is implemented in `fnp-ufunc` but has no named Python wrapper.

```
abs, absolute, acos, acosh, add, arccos, arccosh, arcsin, arcsinh, arctan, arctan2,
arctanh, asin, asinh, atan, atan2, atanh, bitwise_and, bitwise_count, bitwise_invert,
bitwise_left_shift, bitwise_not, bitwise_or, bitwise_right_shift, bitwise_xor, conj,
cos, cosh, divide, divmod, equal, exp, exp2, float_power, fmax, fmin, gcd, greater,
greater_equal, isnat, lcm, left_shift, less, less_equal, log, log10, log2, logical_and,
logical_not, logical_or, logical_xor, matmul, maximum, minimum, mod, multiply,
not_equal, power, remainder, right_shift, sin, sinh, sqrt, subtract, tan, tanh,
true_divide (subset duplicated), vecmat, matvec, matrix_multiply
```

(aliases folded together)

- **Priority:** P0 — these are the numpy API baseline. Every tutorial uses `np.add`, `np.sin`, `np.log`.
- **Implementation cost:** LOW per wrapper (one-line passthrough to numpy like the existing 350+). The pattern is established.
- **Blockers:** None.

### Category 2 — Core array functions NOT surfaced (106)

High-impact named functions that do NOT yet have `fnp_python` wrappers:

```
all, amax, amin, any, apply_along_axis, apply_over_axes, argmax, argmin, around,
array, array2string, array_repr, array_str, asmatrix, astype, atleast_1d, atleast_2d,
atleast_3d, base_repr, binary_repr, block, bmat, busday_count, busday_offset,
can_cast, concat, convolve, correlate, cumprod, cumsum, cumulative_prod,
cumulative_sum, datetime_as_string, datetime_data, dot, ediff1d, einsum, empty,
format_float_positional, format_float_scientific, get_array_wrap, get_include,
histogram, histogram2d, histogramdd, hsplit, interp, is_busday, isclose, isdtype,
isfortran, isnan, isneginf, issubdtype, lookfor, matmul (dup), matrix_transpose,
max, maximum_sctype, mean, median, meshgrid (dup — check), min, moveaxis, nanargmax,
nanargmin, nancumprod, nancumsum, nanmax, nanmin, nanpercentile, nanquantile, nanstd,
nansum, nanvar, ones, piecewise, place, printoptions, prod, promote_types,
put_along_axis, putmask, ravel, repeat, result_type, rollaxis, round, searchsorted,
setbufsize, set_printoptions, shape, shares_memory, sort, source, squeeze,
std, sum, swapaxes, take, take_along_axis, tensordot, trace, transpose, trapezoid,
trapz, tri, tril_indices, trim_zeros, triu_indices, unique, unique_all, unique_counts,
unique_inverse, unique_values, unravel_index, var, vdot, where, zeros
```

Many of these have partial coverage (e.g. `sum` as `size_count`, `argmax` as `ma_argmax`, `tri` exists but `tril_indices`/`triu_indices` do not; `median` exists; `transpose` exists) — so the true net-new count is lower than 106 but still substantial. A precise set-diff per name against `crates/fnp-python/src/lib.rs` would resolve false positives before bead creation.

- **Priority:** P0 for `sum`/`mean`/`max`/`min`/`std`/`var`/`array`/`empty`/`ones`/`zeros`/`arange`/`argmax`/`argmin`/`where`/`astype`/`dot`/`einsum`/`sort`/`take`/`unique`; P1 for the rest.
- **Implementation cost:** LOW–MEDIUM per wrapper.
- **Blockers:** None.

### Category 3 — Classes (21)

```
broadcast, busdaycalendar, dtype, errstate, finfo, flatiter, iinfo, int_, long,
matrix, memmap, ndarray, ndenumerate, ndindex, nditer, poly1d, recarray, record,
ufunc, uint, ulong
```

- **Key ones:** `ndarray`, `dtype`, `nditer`, `errstate`, `finfo`/`iinfo`, `memmap`.
- `fnp-io` implements memmap support internally but `fnp_python.memmap` is not a class.
- `ndarray` and `dtype` would require real pyclass definitions, substantially more work than a passthrough wrapper.
- **Priority:** P1 for `errstate`/`finfo`/`iinfo` (passthrough); P2–P3 for `ndarray`/`dtype`/`nditer` (proper class wrappers are a design discussion).

### Category 4 — Dtype scalar types (50)

```
bool, bool_, byte, bytes_, cdouble, character, clongdouble, complex128, complex256,
complex64, complexfloating, csingle, datetime64, double, flexible, float128, float16,
float32, float64, floating, generic, half, inexact, int16, int32, int64, int8, intc,
integer, intp, longdouble, longlong, number, object_, short, signedinteger, single,
str_, timedelta64, ubyte, uint, uint16, uint32, uint64, uint8, uintc, uintp, ulong,
ulonglong, ushort, unsignedinteger, void
```

- **Priority:** P2. These are scalar *types*, typically used as dtype arguments (`dtype=numpy.int8`). Users can currently get them via `numpy.int8` and pass through fnp_python's wrappers, since fnp_python accepts `Py<PyAny>` for dtype. A pure re-export would be nice but is not blocking.

### Category 5 — Submodules (16)

```
char, core, ctypeslib, dtypes, emath, exceptions, f2py, lib, linalg, ma, polynomial,
random, rec, strings, testing, typing
```

- `fnp_python` exposes `linalg`, `ma`, `testing`, `lib.recfunctions`, `fft`, `polynomial`, `random` **flat** (`linalg_*`, `ma_*`, `testing_assert_*`, `recfunctions_*`, `chebadd`, etc.).
- The vision claim of "full API" is hurt by the flat namespace: a user doing `import fnp_python as np; np.linalg.svd(...)` fails — must write `np.svd(...)` or `np.linalg_svd(...)`.
- **Priority:** P1 — decision needed on whether to add genuine submodule structure. A pyo3 `PyModule::new` nested-module construction would let `fnp_python.linalg.svd` work.

### Category 6 — Constants (6)

```
e, euler_gamma, inf, little_endian, nan, pi
```

- **Priority:** P1 — trivial to add via `m.add("pi", std::f64::consts::PI)` at module init. Skipping these is a user-visible drawback.
- **Implementation cost:** LOW.

### Category 7 — Dunders / version (2)

```
__version__, __array_namespace_info__
```

- **Priority:** P1 for `__version__` (trivial, should match workspace version).
- `__array_namespace_info__` is Array-API related; P2.

### Category 8 — Indexing objects (12 "other")

```
False_, True_, ScalarType, c_, index_exp, mgrid, newaxis, ogrid, r_, s_, sctypeDict, typecodes
```

- `fnp_python` already has `mgrid`, `ogrid`, `r_`, `c_` (confirmed by getattr smoke test).
- The wrap_pyfunction extraction missed them because they're registered via `m.add(...)` / different pyo3 machinery. This is a false-positive in the 283 count; the real gap for this category is likely **5-7**.
- `newaxis` (just `None` alias), `s_`, `index_exp`, `True_`, `False_` — trivial constants.
- **Priority:** P2 (cosmetic).

## Extrapolated true coverage

After netting out false positives from Categories 4, 5, 8:

- Dtype scalar re-exports (50) — mostly cosmetic, users work around with `numpy.int8`.
- Submodule coverage (16) — flat-namespace already exposes linalg/ma/testing/recfunctions/fft/polynomial surface, so the module itself isn't "missing" functionality, just the namespace.
- Indexing objects (12) — at least 5–7 are already exposed.

**Conservative estimate: ~200 net user-facing gaps** (70 core ufuncs + ~100 core functions + 20 constants/types/classes). The **~70 core ufuncs** are the most important of these.

## Ambition-round seed (for the next tick)

If we go into ambition rounds with the above findings, the right next question is:

> "Given our vision says `full API, edge cases and all`, does it matter that 57% of `numpy.__all__` is not callable as `fnp_python.<name>`? Or do we treat fnp_python as a parity-oracle surface and accept that the *real* API lives in fnp-ufunc + fnp-ndarray + fnp-linalg Rust APIs?"

Two valid strategic paths:

1. **Close the Python surface:** Ship ~200 more passthrough wrappers using the established pattern (each ~30-80 lines including the parity test). Would take several more sessions at current cadence (11 wrappers/hour). Output: `fnp_python` that fully re-exports numpy.
2. **Pivot the claim:** Update README to clarify that `fnp_python` is the parity-test bridge, and the real API lives in `fnp-ufunc`/etc. Ship a thin `fnp_python.__all__` proclaiming the subset that IS exposed.

Both are honest. The README today implies (1).

## Draft beads (to file when DB contention clears)

1. `[REALITY] fnp_python covers 43% of numpy.__all__ vs README claim of "full API"` — `type=docs`, `priority=1`. Points to this file.
2. `[REALITY-EPIC] Close fnp_python passthrough gap for numpy.__all__ (~200 wrappers)` — `type=epic`, `priority=1`. Parent of per-category child beads.
3. Per-category task beads (P0/P1/P2 as labeled above), each listing the 10–70 names in scope.
4. `[REALITY] Decide submodule namespace strategy for fnp_python` — `type=question`, `priority=1`. Resolves whether `fnp_python.linalg.svd` should work.
5. `[REALITY] Expose numpy constants (pi, e, nan, inf, euler_gamma, little_endian)` — `type=task`, `priority=2`. 10 lines of Rust.
6. `[REALITY] Expose numpy.__version__ from fnp_python` — `type=task`, `priority=2`. 1 line of Rust.

## Reproduction

```bash
# Dump the oracle surface
python3 -c "import numpy; print('\n'.join(sorted(numpy.__all__)))" > /tmp/numpy_all.txt

# Dump fnp_python registrations
rg -oN "wrap_pyfunction!\(([a-z_][a-zA-Z_0-9]*)" crates/fnp-python/src/lib.rs -r '$1' \
  | sed 's/^py_broadcast_shapes$/broadcast_shapes/' | sort -u > /tmp/fnp_exposed.txt

# Gap list
comm -23 /tmp/numpy_all.txt /tmp/fnp_exposed.txt > /tmp/gap.txt
wc -l /tmp/gap.txt   # 283

# Categorize via numpy introspection
python3 <<'PY'
import numpy as np, inspect
for n in open('/tmp/gap.txt').read().split():
    if n.startswith('_'): k = 'dunder'
    else:
        try: o = getattr(np, n)
        except: o = None
        if isinstance(o, np.ufunc): k = 'ufunc'
        elif inspect.isclass(o): k = 'class'
        elif inspect.ismodule(o): k = 'module'
        elif isinstance(o, (int,float,complex,str,bytes)): k = 'constant'
        else: k = 'function'
    print(f'{k}\t{n}')
PY
```

## Notes

- This audit is code-level against the current `main` branch. Beads filed via earlier sessions may reference specific numpy functions; a follow-up cross-check (`grep -f /tmp/gap.txt .beads/issues.jsonl`) would identify which of the 283 already have open beads vs truly untracked.
- Audit scope was fnp_python's Python surface vs `numpy.__all__`. Engine-level feature parity (fnp-ufunc vs numpy ufuncs, fnp-linalg vs numpy.linalg internals) is tracked separately via the `fnp-conformance` differential oracle.
