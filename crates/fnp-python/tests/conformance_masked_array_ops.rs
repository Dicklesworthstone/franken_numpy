//! Conformance tests for numpy.ma masked array operations against NumPy oracle.
//!
//! Tests: compress_rows, compress_cols, clump_masked, clump_unmasked,
//! flatnotmasked_edges, flatnotmasked_contiguous

use std::process::Command;

fn numpy_oracle(script: &str) -> Result<String, String> {
    let output = Command::new("python3")
        .args(["-c", script])
        .output()
        .map_err(|error| format!("python3 should be available: {error}\nScript: {script}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("NumPy oracle failed: {stderr}\nScript: {script}"));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

mod support;

fn fnp_script(body: String) -> String {
    support::fnp_script_with("import numpy.ma as ma\n", false, body)
}

// ─────────────────────────────────────────────────────────────────────────────
// compress_rows
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compress_rows_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([[1, 2], [3, 4], [5, 6]], mask=[[0, 0], [1, 0], [0, 0]])
fnp_result = fnp.compress_rows(x)
np_result = ma.compress_rows(x)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "compress_rows basic mismatch");
    Ok(())
}

#[test]
fn compress_rows_no_masked() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([[1, 2], [3, 4], [5, 6]])
fnp_result = fnp.compress_rows(x)
np_result = ma.compress_rows(x)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "compress_rows no masked mismatch");
    Ok(())
}

#[test]
fn compress_rows_all_masked() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([[1, 2], [3, 4]], mask=[[1, 1], [1, 1]])
fnp_result = fnp.compress_rows(x)
np_result = ma.compress_rows(x)
print(fnp_result.shape == np_result.shape)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "compress_rows all masked mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// compress_cols
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compress_cols_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([[1, 2, 3], [4, 5, 6]], mask=[[0, 1, 0], [0, 0, 0]])
fnp_result = fnp.compress_cols(x)
np_result = ma.compress_cols(x)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "compress_cols basic mismatch");
    Ok(())
}

#[test]
fn compress_cols_no_masked() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([[1, 2, 3], [4, 5, 6]])
fnp_result = fnp.compress_cols(x)
np_result = ma.compress_cols(x)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "compress_cols no masked mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// clump_masked
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn clump_masked_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3, 4, 5], mask=[0, 1, 1, 0, 1])
fnp_result = fnp.clump_masked(a)
np_result = ma.clump_masked(a)
match = len(fnp_result) == len(np_result)
if match:
    for f, n in zip(fnp_result, np_result):
        if f != n:
            match = False
            break
print(match)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "clump_masked basic mismatch");
    Ok(())
}

#[test]
fn clump_masked_no_masked() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3, 4, 5])
fnp_result = fnp.clump_masked(a)
np_result = ma.clump_masked(a)
print(len(fnp_result) == len(np_result) == 0)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "clump_masked no masked mismatch");
    Ok(())
}

#[test]
fn clump_masked_all_masked() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3], mask=[1, 1, 1])
fnp_result = fnp.clump_masked(a)
np_result = ma.clump_masked(a)
match = len(fnp_result) == len(np_result) == 1
if match:
    match = fnp_result[0] == np_result[0]
print(match)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "clump_masked all masked mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// clump_unmasked
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn clump_unmasked_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3, 4, 5], mask=[1, 0, 0, 1, 0])
fnp_result = fnp.clump_unmasked(a)
np_result = ma.clump_unmasked(a)
match = len(fnp_result) == len(np_result)
if match:
    for f, n in zip(fnp_result, np_result):
        if f != n:
            match = False
            break
print(match)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "clump_unmasked basic mismatch");
    Ok(())
}

#[test]
fn clump_unmasked_all_masked() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3], mask=[1, 1, 1])
fnp_result = fnp.clump_unmasked(a)
np_result = ma.clump_unmasked(a)
print(len(fnp_result) == len(np_result) == 0)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "clump_unmasked all masked mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// flatnotmasked_edges
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn flatnotmasked_edges_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3, 4, 5], mask=[1, 0, 0, 0, 1])
fnp_result = fnp.flatnotmasked_edges(a)
np_result = ma.flatnotmasked_edges(a)
# flatnotmasked_edges returns a tuple or None
if np_result is None:
    print(fnp_result is None)
else:
    print(fnp_result[0] == np_result[0] and fnp_result[1] == np_result[1])
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "flatnotmasked_edges basic mismatch");
    Ok(())
}

#[test]
fn flatnotmasked_edges_all_masked() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3], mask=[1, 1, 1])
fnp_result = fnp.flatnotmasked_edges(a)
np_result = ma.flatnotmasked_edges(a)
print(fnp_result is None and np_result is None)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "flatnotmasked_edges all masked mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// flatnotmasked_contiguous
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn flatnotmasked_contiguous_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3, 4, 5], mask=[0, 1, 1, 0, 0])
fnp_result = fnp.flatnotmasked_contiguous(a)
np_result = ma.flatnotmasked_contiguous(a)
match = len(fnp_result) == len(np_result)
if match:
    for f, n in zip(fnp_result, np_result):
        if f != n:
            match = False
            break
print(match)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "flatnotmasked_contiguous basic mismatch");
    Ok(())
}

#[test]
fn flatnotmasked_contiguous_no_masked() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3, 4, 5])
fnp_result = fnp.flatnotmasked_contiguous(a)
np_result = ma.flatnotmasked_contiguous(a)
match = len(fnp_result) == len(np_result) == 1
if match:
    match = fnp_result[0] == np_result[0]
print(match)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(
        output, "True",
        "flatnotmasked_contiguous no masked mismatch"
    );
    Ok(())
}

// numpy.ma re-export lock.
//
// Several numpy.ma names collide with an fnp TOP-LEVEL wrapper while being a
// different, masked-aware function: ma.cov, ma.corrcoef, ma.allclose, ma.ptp,
// ma.choose, ma.put. fnp registers them by pulling the objects straight out of
// numpy.ma (`np_ma.getattr(name)`), which is correct - but nothing asserted it,
// and swapping in the same-named top-level wrapper is exactly the mistake that
// was live in the linalg namespace until
// deadlock-audit-linalg-aliases-toplevel-wrappers-mi3k6. There it returned the
// wrong array for ndim > 2; here it would silently drop the MASK.
//
// Each case is compared against numpy.ma, and the test additionally asserts the
// PRECONDITIONS that make the comparison meaningful: ma.cov and np.cov must
// actually disagree on masked input, and the ma-only keywords must be rejected
// by the top-level functions. Without those, an alias regression could leave
// every case still passing.
#[test]
fn ma_namespace_is_numpy_ma_not_the_toplevel_wrappers() -> Result<(), String> {
    let script = fnp_script(
        r#"
import platform
import numpy.ma as npma

def masked():
    return npma.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], mask=[[0, 0, 1], [0, 0, 0]])

def described(value):
    array = np.asarray(value)
    return (str(array.dtype), tuple(array.shape), np.round(array, 9).tolist())

def ma_cov(module):
    return described(module.ma.cov(masked()))

def ma_cov_allow_masked(module):
    return described(module.ma.cov(masked(), allow_masked=True))

def ma_corrcoef(module):
    return described(module.ma.corrcoef(masked()))

def ma_allclose_masked_equal(module):
    return bool(module.ma.allclose(masked(), masked(), masked_equal=True))

def ma_ptp(module):
    return described(module.ma.ptp(masked()))

def ma_count(module):
    return described(module.ma.count(masked()))

def ma_average_returned(module):
    value, weight = module.ma.average(masked(), axis=0, returned=True)
    return (described(value), described(weight))

def ma_masked_is_singleton(module):
    return module.ma.masked is npma.masked

def ma_maskedarray_is_type(module):
    return module.ma.MaskedArray is npma.MaskedArray

cases = [
    ("ma.cov on masked input", ma_cov),
    ("ma.cov allow_masked=", ma_cov_allow_masked),
    ("ma.corrcoef on masked input", ma_corrcoef),
    ("ma.allclose masked_equal=", ma_allclose_masked_equal),
    ("ma.ptp on masked input", ma_ptp),
    ("ma.count on masked input", ma_count),
    ("ma.average returned=", ma_average_returned),
    ("ma.masked singleton identity", ma_masked_is_singleton),
    ("ma.MaskedArray type identity", ma_maskedarray_is_type),
]

def outcome(module, call):
    try:
        return ("ok", call(module))
    except Exception as exc:
        return ("err", type(exc).__name__)

ok = True
for label, call in cases:
    actual = outcome(fnp, call)
    expected = outcome(np, call)
    if actual != expected:
        print(label)
        print(actual)
        print(expected)
        ok = False

# Preconditions. If these ever stop holding, the cases above can no longer
# distinguish numpy.ma's function from the same-named top-level one, and this
# test has quietly stopped testing what it exists for.
x = masked()
if np.array_equal(np.asarray(npma.cov(x)), np.asarray(np.cov(x))):
    print("PRECONDITION LOST: ma.cov and np.cov agree on masked input")
    ok = False
for name, kwargs in [("cov", {"allow_masked": True}), ("allclose", {"masked_equal": True})]:
    try:
        if name == "cov":
            getattr(np, name)(x, **kwargs)
        else:
            getattr(np, name)(x, x, **kwargs)
        print(f"PRECONDITION LOST: top-level np.{name} accepted {list(kwargs)[0]}")
        ok = False
    except TypeError:
        pass

print(ok)
print("oracle", platform.node(), np.__version__)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.trim().lines().rev();
    let provenance = lines.next().unwrap_or("").trim();
    let verdict = lines.next().unwrap_or("").trim();
    assert_eq!(
        verdict, "True",
        "ma namespace should be numpy.ma's own objects ({provenance}): {result}"
    );
    Ok(())
}

/// Every function fnp.ma implements itself (53 names) against numpy.ma over masked arrays with
/// no / partial / full / row masks, float64 and int32 data with NaN and inf, a float32 array
/// with a user fill_value, bool, complex, plain ndarrays, lists, 0-d and empty inputs, and each
/// axis: result type, data, mask (and nomask-ness), fill_value, dtype and shape must match
/// (1,110 cases). Before the fixes (bead .8, 86 mismatches): the masked_<cmp>/inside/outside/
/// where/values/invalid wrappers rebuilt MaskedArray inputs with the dtype-default fill_value
/// (a float32 array with fill_value=-5 came back with 1e20); weighted average and ediff1d
/// reported the input's integer fill (999999) on float results and ediff1d computed different
/// data under the mask; allequal returned a Python bool and answered False for broadcastable
/// shapes; masked_invalid shrank an all-valid mask to nomask (numpy keeps it, gh-22842); filled
/// returned a scalar for a 0-d array and a copy of a plain ndarray.
#[test]
fn fnp_ma_functions_match_numpy_ma_types_masks_and_fill_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
import numpy.ma as npma
warnings.simplefilter("ignore")
rng = np.random.default_rng(5)
d = rng.standard_normal((6, 5)) * 10
d[1, 2] = np.nan; d[4, 0] = np.inf
A = {}
for tag, mask in (("somemask", rng.random((6, 5)) < 0.3), ("nomask", npma.nomask),
                  ("allmask", np.ones((6, 5), bool)), ("rowmask", np.repeat(rng.random((6, 1)) < 0.5, 5, 1))):
    A[f"f8 {tag}"] = npma.array(d, mask=mask)
    A[f"i4 {tag}"] = npma.array(np.nan_to_num(d * 3).astype(np.int32), mask=mask)
A["f4 fill"] = npma.array(rng.standard_normal(12).astype(np.float32), mask=rng.random(12) < 0.4, fill_value=-5)
A["bool"] = npma.array(rng.random(10) < 0.5, mask=rng.random(10) < 0.3)
A["c16"] = npma.array(rng.standard_normal(8) + 1j, mask=[0, 1] * 4)
A["plain"] = rng.standard_normal((3, 4))
A["list"] = [1.0, 2.0, np.nan, 4.0]
A["0-d"] = npma.array(3.0, mask=True)
A["empty"] = npma.array(np.zeros((0, 3)), mask=np.zeros((0, 3), bool))
cases = []
def add(name, fn):
    cases.append((name, fn))
for tag, a in A.items():
    for f in ("getmask", "getmaskarray", "is_masked", "count_masked", "compressed", "filled",
              "masked_invalid", "fix_invalid", "default_fill_value", "maximum_fill_value", "minimum_fill_value"):
        add(f"{f} {tag}", lambda m, f=f, a=a: getattr(m, f)(a))
    for f in ("clump_masked", "clump_unmasked", "flatnotmasked_contiguous", "flatnotmasked_edges"):
        add(f"{f} {tag}", lambda m, f=f, a=a: getattr(m, f)(npma.ravel(npma.asarray(a))))
    add(f"flatten_mask {tag}", lambda m, a=a: m.flatten_mask(npma.getmaskarray(a)))
    add(f"make_mask_none {tag}", lambda m, a=a: m.make_mask_none(np.shape(a)))
    add(f"filled value {tag}", lambda m, a=a: m.filled(a, -99))
    for axis in (None, 0, 1, -1):
        for f in ("count", "argmax", "argmin", "average", "notmasked_contiguous", "notmasked_edges"):
            add(f"{f} axis={axis} {tag}", lambda m, f=f, a=a, ax=axis: getattr(m, f)(a, axis=ax))
        add(f"compress_nd axis={axis} {tag}", lambda m, a=a, ax=axis: m.compress_nd(a, axis=ax))
        if axis is not None:
            add(f"apply_along_axis {axis} {tag}", lambda m, a=a, ax=axis: m.apply_along_axis(np.sum, ax, a))
    add(f"average weights {tag}", lambda m, a=a: m.average(a, axis=0, weights=np.arange(1.0, 1.0 + np.shape(a)[0]) if np.ndim(a) else None))
    add(f"average returned {tag}", lambda m, a=a: m.average(a, returned=True))
    for f in ("compress_rows", "compress_cols", "compress_rowcols", "mask_rows", "mask_cols", "mask_rowcols"):
        add(f"{f} {tag}", lambda m, f=f, a=a: getattr(m, f)(a))
    add(f"ediff1d {tag}", lambda m, a=a: m.ediff1d(a, to_begin=[-1.0]))
    add(f"apply_over_axes {tag}", lambda m, a=a: m.apply_over_axes(np.sum, a, [0]))
    for f, args in (("masked_equal", (1,)), ("masked_not_equal", (1,)), ("masked_greater", (0.5,)),
                    ("masked_greater_equal", (0.5,)), ("masked_less", (0.5,)), ("masked_less_equal", (0.5,)),
                    ("masked_inside", (-1, 1)), ("masked_outside", (-1, 1)), ("masked_values", (2.0,))):
        add(f"{f} {tag}", lambda m, f=f, a=a, args=args: getattr(m, f)(a, *args))
    add(f"masked_where {tag}", lambda m, a=a: m.masked_where(np.asarray(npma.getdata(a)) > 0, a))
    add(f"masked_where copy=False {tag}", lambda m, a=a: m.masked_where(np.asarray(npma.getdata(a)) > 0, a, copy=False))
    add(f"allequal {tag}", lambda m, a=a: m.allequal(a, a))
    add(f"allequal fill=False {tag}", lambda m, a=a: m.allequal(a, npma.array(a) * 1, fill_value=False))
    add(f"set_fill_value {tag}", lambda m, a=a: (lambda b: (m.set_fill_value(b, 7), npma.array(b).fill_value)[1])(npma.array(a, copy=True)))
add("allequal broadcast", lambda m: m.allequal(np.array([1, 2]), np.array([[1, 2], [1, 2]])))
m1 = np.array([1, 0, 1, 0], bool); m2 = np.array([0, 0, 1, 1], bool)
add("mask_or", lambda m: m.mask_or(m1, m2))
add("mask_or nomask", lambda m: m.mask_or(npma.nomask, m2))
add("make_mask", lambda m: m.make_mask([1, 0, 2]))
add("make_mask shrink=False", lambda m: m.make_mask([0, 0, 0], shrink=False))
add("make_mask_descr", lambda m: m.make_mask_descr(np.dtype([("a", "f8"), ("b", "i4")])))
meta = lambda r: (type(r).__name__, str(r.dtype), r.shape, np.asarray(r.mask).tobytes(), repr(r.fill_value))
add("masked_all", lambda m: meta(m.masked_all((2, 3), dtype=np.int16)))
add("masked_all_like", lambda m: meta(m.masked_all_like(np.zeros((2, 2), np.float32))))
add("masked_object", lambda m: m.masked_object(np.array([1, None, 3], dtype=object), None))
add("common_fill_value", lambda m: m.common_fill_value(npma.array([1], fill_value=3), npma.array([2], fill_value=3)))
add("flatten_structured_array", lambda m: m.flatten_structured_array(npma.array([(1, 2.0)], dtype=[("a", "i4"), ("b", "f8")])))
add("default_fill_value dtypes", lambda m: [m.default_fill_value(np.dtype(t)) for t in ("f8", "i4", "u1", "?", "U3", "c8", "M8[s]", "O")])
add("filled identity", lambda m: (lambda x: m.filled(x) is x)(np.arange(3.0)))

class Raised:
    def __init__(self, ex): self.name = type(ex).__name__

def norm(v):
    if v is npma.masked:
        return ("masked",)
    if isinstance(v, npma.MaskedArray):
        return ("MA", str(v.dtype), v.shape, np.asarray(v.data).tobytes(),
                np.asarray(npma.getmaskarray(v)).tobytes(), repr(v.fill_value), v.mask is npma.nomask)
    if isinstance(v, (list, tuple)):
        return (type(v).__name__,) + tuple(norm(x) for x in v)
    if isinstance(v, slice):
        return ("slice", v.start, v.stop, v.step)
    if isinstance(v, np.ndarray):
        return ("nd", str(v.dtype), v.shape, v.tobytes())
    if isinstance(v, np.generic):
        return (type(v).__name__, np.asarray(v).tobytes())
    return (type(v).__name__, repr(v))

bad = []
for name, fn in cases:
    try:
        s = fn(npma)
    except Exception as ex:
        s = Raised(ex)
    try:
        r = fn(fnp.ma)
    except Exception as ex:
        r = Raised(ex)
    if isinstance(s, Raised) or isinstance(r, Raised):
        if not (isinstance(s, Raised) and isinstance(r, Raised) and s.name == r.name):
            bad.append(f"{name}: fnp={getattr(r, 'name', 'ok')} numpy={getattr(s, 'name', 'ok')}")
    elif norm(r) != norm(s):
        bad.append(name)
print(len(cases), bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (cases, bad) = result.trim().split_once(' ').unwrap_or(("0", &result));
    assert!(
        cases.parse::<usize>().unwrap_or(0) >= 1000,
        "case table drifted: {result}"
    );
    assert_eq!(bad, "[]", "fnp.ma must match numpy.ma: {result}");
    Ok(())
}
