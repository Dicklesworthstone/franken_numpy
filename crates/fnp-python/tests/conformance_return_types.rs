//! Cross-cutting result-parity sweeps: what comes back from a call.
//!
//! Three sweeps live here, one per attribute of the result - its TYPE (numpy
//! scalar vs 0-d ndarray), its DTYPE, and, when the call fails, the EXCEPTION
//! CLASS. Each compares fnp against numpy in the same interpreter, and each
//! carries preconditions so it cannot go quietly slack.
//!
//! ## Return type
//!
//! numpy returns a numpy SCALAR (np.float64, np.int64, np.bool_) for a full
//! reduction, not a 0-d ndarray. Same dtype, same shape, different type - and
//! any caller doing `type(x)`, `isinstance`, or `x is np.True_` can see it.
//! fnp picks between `build_numpy_scalar_or_array` (which collapses the 0-d
//! case) and `build_numpy_array_from_ufunc` (which does not) by hand, once per
//! wrapper, ~200 times. The `choose` wrapper's own comment records getting that
//! wrong (deadlock-audit-41nl1).
//!
//! Roughly 40 shards assert the return type of their own function. None of them
//! sweeps, so which wrappers LACK the check was unknown - which is what this
//! file is for. Every case compares fnp against numpy in the same interpreter;
//! nothing is pinned to a build.

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

fn fnp_script(body: String) -> String {
    let library_name = format!(
        "{}fnp_python{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    );
    let module_path = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.join(&library_name)))
        .unwrap_or_else(|| library_name.into());
    let module_literal = format!("{module_path:?}");
    format!(
        "import importlib.util\n\
         import numpy as np\n\
         spec = importlib.util.spec_from_file_location('fnp_python', {module_literal})\n\
         fnp = importlib.util.module_from_spec(spec)\n\
         spec.loader.exec_module(fnp)\n\
         {body}"
    )
}

#[test]
fn return_type_parity_across_reductions_and_elementwise() -> Result<(), String> {
    let script = fnp_script(
        r#"
import platform
import warnings

warnings.simplefilter("ignore")

f64 = np.array([1.0, 2.0, 3.0, 4.0])
i64 = np.array([1, 2, 3, 4], dtype=np.int64)
b = np.array([True, False, True])
grid = np.array([[1.0, 2.0], [3.0, 4.0]])

def shape_of(value):
    # type name is the point; ndim and ndarray-ness make the failure readable.
    return (type(value).__name__, isinstance(value, np.ndarray), getattr(value, "ndim", None))

# (label, callable) — each returns the object whose TYPE is compared.
full_reductions = [
    ("sum f64", lambda m: m.sum(f64)),
    ("sum i64", lambda m: m.sum(i64)),
    ("prod f64", lambda m: m.prod(f64)),
    ("mean f64", lambda m: m.mean(f64)),
    ("std f64", lambda m: m.std(f64)),
    ("var f64", lambda m: m.var(f64)),
    ("min f64", lambda m: m.min(f64)),
    ("max f64", lambda m: m.max(f64)),
    ("ptp f64", lambda m: m.ptp(f64)),
    ("median f64", lambda m: m.median(f64)),
    ("average f64", lambda m: m.average(f64)),
    ("any bool", lambda m: m.any(b)),
    ("all bool", lambda m: m.all(b)),
    ("argmin f64", lambda m: m.argmin(f64)),
    ("argmax f64", lambda m: m.argmax(f64)),
    ("count_nonzero f64", lambda m: m.count_nonzero(f64)),
    ("trace 2-D", lambda m: m.trace(grid)),
    ("nansum f64", lambda m: m.nansum(f64)),
    ("nanmean f64", lambda m: m.nanmean(f64)),
    ("nanstd f64", lambda m: m.nanstd(f64)),
    ("nanvar f64", lambda m: m.nanvar(f64)),
    ("nanmin f64", lambda m: m.nanmin(f64)),
    ("nanmax f64", lambda m: m.nanmax(f64)),
    ("nanprod f64", lambda m: m.nanprod(f64)),
    ("nanmedian f64", lambda m: m.nanmedian(f64)),
    ("nanargmin f64", lambda m: m.nanargmin(f64)),
    ("nanargmax f64", lambda m: m.nanargmax(f64)),
    ("percentile scalar q", lambda m: m.percentile(f64, 50)),
    ("quantile scalar q", lambda m: m.quantile(f64, 0.5)),
    ("dot 1-D", lambda m: m.dot(f64, f64)),
    ("vdot 1-D", lambda m: m.vdot(f64, f64)),
    ("inner 1-D", lambda m: m.inner(f64, f64)),
    ("linalg.det", lambda m: m.linalg.det(grid)),
    ("linalg.norm", lambda m: m.linalg.norm(f64)),
]

# CONTROL GROUP: these must stay ndarray. Without them the sweep could pass by
# asserting "everything collapses to a scalar", which is the opposite error.
must_stay_array = [
    ("sum axis=0", lambda m: m.sum(grid, axis=0)),
    ("mean axis=1", lambda m: m.mean(grid, axis=1)),
    ("argmin axis=0", lambda m: m.argmin(grid, axis=0)),
    ("any axis=0", lambda m: m.any(grid, axis=0)),
    ("sum keepdims", lambda m: m.sum(f64, keepdims=True)),
    ("median keepdims", lambda m: m.median(f64, keepdims=True)),
    ("percentile list q", lambda m: m.percentile(f64, [25, 50])),
    ("sqrt on array", lambda m: m.sqrt(f64)),
    ("cumsum", lambda m: m.cumsum(f64)),
]

# Elementwise on a SCALAR input: numpy returns a numpy scalar, not a 0-d array.
scalar_inputs = [
    ("sqrt python float", lambda m: m.sqrt(4.0)),
    ("sqrt np.float64", lambda m: m.sqrt(np.float64(4.0))),
    ("abs python int", lambda m: m.abs(-3)),
    ("negative np.int64", lambda m: m.negative(np.int64(3))),
    ("exp python float", lambda m: m.exp(0.0)),
    ("isnan np.float64", lambda m: m.isnan(np.float64(1.0))),
    ("sign np.float64", lambda m: m.sign(np.float64(-2.0))),
    ("sqrt 0-d array", lambda m: m.sqrt(np.array(4.0))),
]

def outcome(module, call):
    try:
        return ("ok", shape_of(call(module)))
    except Exception as exc:
        return ("err", type(exc).__name__)

ok = True
for group, cases in (("reduction", full_reductions),
                     ("stays-array", must_stay_array),
                     ("scalar-input", scalar_inputs)):
    for label, call in cases:
        actual = outcome(fnp, call)
        expected = outcome(np, call)
        if actual != expected:
            print(f"{group}: {label}")
            print(f"  fnp   {actual}")
            print(f"  numpy {expected}")
            ok = False

# Preconditions. If numpy ever stopped returning scalars for full reductions,
# or stopped returning arrays for axis reductions, the groups above would no
# longer be testing opposite things and this sweep would quietly go slack.
if isinstance(np.sum(f64), np.ndarray):
    print("PRECONDITION LOST: numpy's full reduction returned an ndarray")
    ok = False
if not isinstance(np.sum(grid, axis=0), np.ndarray):
    print("PRECONDITION LOST: numpy's axis reduction did not return an ndarray")
    ok = False

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
        "return types should match numpy ({provenance}): {result}"
    );
    Ok(())
}

/// Output DTYPE across the dtype grid - the same question as the sweep above
/// (what comes back from a call), one attribute over.
///
/// fnp's generic extraction path canonicalises to an f64 Vec, and the source
/// says so wherever someone hit it: "extract_numeric_array canonicalizes to an
/// f64 Vec element-wise (~49x slower than numpy for int32)" (signbit), "the
/// cold extract residual below still widens narrow widths, so it is guarded to
/// bool/8-byte only" (take), "the native path extracted the whole array to a
/// widened-f64 UFuncArray" (isposinf). Each of those was a path that would have
/// returned float64 where numpy returns int8/int32/float32, and each was fixed
/// by adding a guard or a zero-copy route BY HAND. A missed guard is a silently
/// wrong output dtype - a memory blow-up, a broken astype round trip, or an
/// integer that stops wrapping.
///
/// bool and float16 are in the grid deliberately: they are what the f64
/// canonicalisation mangles most visibly.
#[test]
fn output_dtype_parity_across_the_dtype_grid() -> Result<(), String> {
    let script = fnp_script(
        r#"
import platform
import warnings

warnings.simplefilter("ignore")

dtypes = ["int8", "int16", "int32", "int64",
          "uint8", "uint16", "uint32", "uint64",
          "float16", "float32", "float64",
          "complex64", "complex128", "bool"]

def sample(name):
    if name == "bool":
        return np.array([True, False, True, True])
    if name.startswith("complex"):
        return np.array([1 + 2j, 3 - 1j, 2 + 0j, 0 + 4j], dtype=name)
    if name.startswith("uint"):
        return np.array([1, 2, 3, 4], dtype=name)
    return np.array([1, -2, 3, -4], dtype=name)

# Structural ops must preserve the input dtype EXACTLY - no promotion rules
# apply, so any divergence here is unambiguous.
structural = [
    ("reshape", lambda m, x: m.reshape(x, (2, 2))),
    ("transpose", lambda m, x: m.transpose(m.reshape(x, (2, 2)))),
    ("sort", lambda m, x: m.sort(x)),
    ("unique", lambda m, x: m.unique(x)),
    ("concatenate", lambda m, x: m.concatenate([x, x])),
    ("repeat", lambda m, x: m.repeat(x, 2)),
    ("tile", lambda m, x: m.tile(x, 2)),
    ("flip", lambda m, x: m.flip(x)),
    ("roll", lambda m, x: m.roll(x, 1)),
    ("ravel", lambda m, x: m.ravel(m.reshape(x, (2, 2)))),
    ("take", lambda m, x: m.take(x, [0, 2])),
    ("compress", lambda m, x: m.compress([True, False, True, False], x)),
    ("where", lambda m, x: m.where(np.array([True, False, True, False]), x, x)),
    ("append", lambda m, x: m.append(x, x)),
]

# Elementwise and reductions follow numpy's own promotion/accumulator rules;
# the point is that fnp reproduces THOSE, not that the dtype is preserved.
elementwise = [
    ("abs", lambda m, x: m.abs(x)),
    ("negative", lambda m, x: m.negative(x)),
    ("square", lambda m, x: m.square(x)),
    ("sign", lambda m, x: m.sign(x)),
    ("add python int", lambda m, x: m.add(x, 1)),
    ("add python float", lambda m, x: m.add(x, 1.0)),
    ("maximum self", lambda m, x: m.maximum(x, x)),
]

reductions = [
    ("sum", lambda m, x: m.sum(x)),
    ("prod", lambda m, x: m.prod(x)),
    ("mean", lambda m, x: m.mean(x)),
    ("min", lambda m, x: m.min(x)),
    ("max", lambda m, x: m.max(x)),
    ("cumsum", lambda m, x: m.cumsum(x)),
]

def outcome(module, call, x):
    try:
        return ("ok", str(np.asarray(call(module, x)).dtype))
    except Exception as exc:
        return ("err", type(exc).__name__)

ok = True
compared = 0
for group, cases in (("structural", structural),
                     ("elementwise", elementwise),
                     ("reduction", reductions)):
    for name in dtypes:
        for label, call in cases:
            x = sample(name)
            actual = outcome(fnp, call, x)
            expected = outcome(np, call, x)
            compared += 1
            if actual != expected:
                print(f"{group}/{name}: {label}")
                print(f"  fnp   {actual}")
                print(f"  numpy {expected}")
                ok = False

# Preconditions. The grid is only meaningful if numpy itself is NOT returning
# float64 everywhere - otherwise "matches numpy" would be satisfied by the very
# f64 canonicalisation this sweep exists to catch.
narrow = str(np.asarray(np.sort(sample("int8"))).dtype)
if narrow != "int8":
    print(f"PRECONDITION LOST: numpy's sort(int8) returned {narrow}")
    ok = False
half = str(np.asarray(np.abs(sample("float16"))).dtype)
if half != "float16":
    print(f"PRECONDITION LOST: numpy's abs(float16) returned {half}")
    ok = False
if compared < 300:
    print(f"PRECONDITION LOST: only {compared} comparisons ran")
    ok = False

print(ok)
print("oracle", platform.node(), np.__version__, compared)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.trim().lines().rev();
    let provenance = lines.next().unwrap_or("").trim();
    let verdict = lines.next().unwrap_or("").trim();
    assert_eq!(
        verdict, "True",
        "output dtypes should match numpy ({provenance}): {result}"
    );
    Ok(())
}

/// EXCEPTION CLASS for a bad input - what comes back when the call fails.
///
/// fnp's wrappers reach numpy's error surface three different ways, chosen per
/// site: delegate the whole call so numpy raises its own error; let a native
/// path raise a house-style error (the `choose` wrapper's comment records
/// exactly that - "the native path raises its own house-style strings ...
/// `choose: unsupported mode 'bad'` where numpy says `clipmode must be one of
/// ...`" - and resolves it by handing failing cases back); or map a UFuncError
/// through `map_ufunc_error`, which picks the Python class.
///
/// Getting the CLASS wrong is worse than getting the message wrong: an
/// `except IndexError` silently stops catching. numpy is specific here -
/// out-of-bounds indexing is IndexError, a bad axis is
/// numpy.exceptions.AxisError (which subclasses BOTH ValueError and
/// IndexError), a singular matrix is numpy.linalg.LinAlgError.
///
/// The exception's defining MODULE is asserted alongside its name: a
/// hand-rolled AxisError and numpy's would both print "AxisError", and only the
/// module tells them apart.
#[test]
fn exception_class_parity_for_bad_inputs() -> Result<(), String> {
    let script = fnp_script(
        r#"
import platform
import warnings

warnings.simplefilter("ignore")

flat = np.array([1.0, 2.0, 3.0])
singular = np.array([[1.0, 2.0], [2.0, 4.0]])
square = np.array([[1.0, 2.0], [3.0, 4.0]])

bad_calls = [
    ("take out of range", lambda m: m.take(flat, [9])),
    ("delete out of range", lambda m: m.delete(flat, 9)),
    ("insert out of range", lambda m: m.insert(flat, 9, 1.0)),
    ("sum bad axis", lambda m: m.sum(flat, axis=5)),
    ("unique bad axis", lambda m: m.unique(flat, axis=3)),
    ("expand_dims bad axis", lambda m: m.expand_dims(flat, axis=7)),
    ("squeeze bad axis", lambda m: m.squeeze(flat, axis=1)),
    ("reshape incompatible", lambda m: m.reshape(flat, (2, 2))),
    ("concatenate of nothing", lambda m: m.concatenate([])),
    ("percentile q out of range", lambda m: m.percentile(flat, 150)),
    ("searchsorted bad side", lambda m: m.searchsorted(flat, 1, side="middle")),
    ("choose bad mode", lambda m: m.choose([0], [flat], mode="bad")),
    ("linalg.inv singular", lambda m: m.linalg.inv(singular)),
    ("linalg.solve non-square", lambda m: m.linalg.solve(np.ones((2, 3)), np.ones(2))),
    ("sqrt on string array", lambda m: m.sqrt(np.array(["a", "b"]))),
    ("astype nonsense dtype", lambda m: m.asarray(flat).astype("notadtype")),
    ("matmul shape mismatch", lambda m: m.matmul(square, np.ones((3, 3)))),
    ("dot shape mismatch", lambda m: m.dot(flat, np.ones(5))),
    ("diag on 3-D", lambda m: m.diag(np.ones((2, 2, 2)))),
]

# MUST NOT RAISE. Without this group the sweep could pass with every call
# raising the same class - or with fnp raising where numpy succeeds, which is
# the failure mode liz1c actually found (take(complex64) raised TypeError while
# numpy returned an array).
must_succeed = [
    ("take in range", lambda m: m.take(flat, [0, 2])),
    ("take complex64 list index", lambda m: m.take(np.array([1 + 2j, 3 - 1j], dtype=np.complex64), [0])),
    ("sum axis 0", lambda m: m.sum(flat, axis=0)),
    ("reshape compatible", lambda m: m.reshape(flat, (3, 1))),
    ("percentile q in range", lambda m: m.percentile(flat, 50)),
    ("linalg.inv non-singular", lambda m: m.linalg.inv(square)),
    ("searchsorted valid side", lambda m: m.searchsorted(flat, 1, side="right")),
]

def raised(module, call):
    try:
        call(module)
        return ("no raise",)
    except Exception as exc:
        # The MODULE matters: a hand-rolled AxisError and numpy's would both
        # print "AxisError".
        return ("raised", f"{type(exc).__module__}.{type(exc).__name__}")

ok = True
for label, call in bad_calls + must_succeed:
    actual = raised(fnp, call)
    expected = raised(np, call)
    if actual != expected:
        print(label)
        print(f"  fnp   {actual}")
        print(f"  numpy {expected}")
        ok = False

# Preconditions: numpy must really raise for the bad group and really succeed
# for the control group, and the classes must not all collapse to one - if numpy
# ever raised plain ValueError everywhere, matching it would prove nothing about
# AxisError or LinAlgError.
numpy_classes = {raised(np, call)[-1] for _, call in bad_calls if raised(np, call)[0] == "raised"}
if len(numpy_classes) < 4:
    print(f"PRECONDITION LOST: numpy raised only {len(numpy_classes)} distinct classes: {sorted(numpy_classes)}")
    ok = False
if any(raised(np, call)[0] == "no raise" for _, call in bad_calls):
    print("PRECONDITION LOST: a bad-input case no longer raises on numpy")
    ok = False
if any(raised(np, call)[0] == "raised" for _, call in must_succeed):
    print("PRECONDITION LOST: a control case now raises on numpy")
    ok = False

print(ok)
print("oracle", platform.node(), np.__version__, sorted(numpy_classes))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.trim().lines().rev();
    let provenance = lines.next().unwrap_or("").trim();
    let verdict = lines.next().unwrap_or("").trim();
    assert_eq!(
        verdict, "True",
        "exception classes should match numpy ({provenance}): {result}"
    );
    Ok(())
}

/// keepdims x axis SHAPE parity across the reduction surface.
///
/// fnp's native reduction paths do not get keepdims from numpy - they
/// reconstruct the output shape themselves, via `keepdims_expand_axis` or by
/// reshaping to `vec![1usize; ndim]`, once per site, over axis=None, a single
/// positive axis, a NEGATIVE axis, and a TUPLE of axes. That arithmetic has
/// been wrong before: nanpercentile's comment names the class outright,
/// "(keepdims-on-axis class, BlackThrush 2026-06-22)".
///
/// A wrong keepdims shape is silent enough to matter - values and dtype are
/// right, and it only surfaces when the result is broadcast back against the
/// input, which is the entire reason keepdims exists. So the sweep also asserts
/// that broadcast: `a / result` must succeed and match numpy's.
#[test]
fn keepdims_and_axis_shapes_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import platform
import warnings

warnings.simplefilter("ignore")

cube = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
cube[0, 0, 0] = np.nan

plain = ["sum", "prod", "mean", "std", "var", "min", "max", "median",
         "any", "all", "argmin", "argmax"]
nanned = ["nansum", "nanprod", "nanmean", "nanstd", "nanvar", "nanmin",
          "nanmax", "nanmedian"]
axes = [None, 0, 1, 2, -1, -2, (0, 1), (0, -1), (1, 2)]

def describe(value):
    array = np.asarray(value)
    return (tuple(array.shape), str(array.dtype),
            np.round(np.nan_to_num(array, nan=-12345.0), 9).tolist())

def call(module, name, axis, keepdims):
    fn = getattr(module, name)
    source = np.nan_to_num(cube, nan=0.5) if name in ("median",) else cube
    if axis is None:
        return fn(source, keepdims=keepdims)
    return fn(source, axis=axis, keepdims=keepdims)

ok = True
compared = 0
for name in plain + nanned:
    for axis in axes:
        for keepdims in (False, True):
            # argmin/argmax take a single axis or None, never a tuple.
            if name.endswith(("argmin", "argmax")) and isinstance(axis, tuple):
                continue
            try:
                actual = ("ok",) + describe(call(fnp, name, axis, keepdims))
            except Exception as exc:
                actual = ("err", type(exc).__name__)
            try:
                expected = ("ok",) + describe(call(np, name, axis, keepdims))
            except Exception as exc:
                expected = ("err", type(exc).__name__)
            compared += 1
            if actual != expected:
                print(f"{name} axis={axis} keepdims={keepdims}")
                print(f"  fnp   {actual[:3]}")
                print(f"  numpy {expected[:3]}")
                ok = False

# The property keepdims exists FOR: a keepdims=True result must broadcast back
# against the input. A shape that is merely "some tuple with ones in it" can
# still pass a shape comparison if both sides are wrong in the same way; this
# cannot.
for name in ("sum", "mean", "max", "nansum", "nanmean"):
    for axis in (0, 1, 2, -1, (0, 1)):
        try:
            fnp_ratio = describe(np.nan_to_num(cube, nan=1.0) / call(fnp, name, axis, True))
        except Exception as exc:
            fnp_ratio = ("err", type(exc).__name__)
        try:
            np_ratio = describe(np.nan_to_num(cube, nan=1.0) / call(np, name, axis, True))
        except Exception as exc:
            np_ratio = ("err", type(exc).__name__)
        compared += 1
        if fnp_ratio != np_ratio:
            print(f"broadcast-back {name} axis={axis}")
            print(f"  fnp   {str(fnp_ratio)[:120]}")
            print(f"  numpy {str(np_ratio)[:120]}")
            ok = False

# Preconditions: numpy must actually DIFFER between keepdims False and True
# (otherwise the grid proves nothing), and the sweep must have run.
if np.sum(cube, axis=0, keepdims=True).shape == np.sum(cube, axis=0).shape:
    print("PRECONDITION LOST: numpy's keepdims no longer changes the shape")
    ok = False
if compared < 300:
    print(f"PRECONDITION LOST: only {compared} comparisons ran")
    ok = False

print(ok)
print("oracle", platform.node(), np.__version__, compared)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.trim().lines().rev();
    let provenance = lines.next().unwrap_or("").trim();
    let verdict = lines.next().unwrap_or("").trim();
    assert_eq!(
        verdict, "True",
        "keepdims/axis shapes should match numpy ({provenance}): {result}"
    );
    Ok(())
}

/// NON-CONTIGUOUS INPUT parity - the first sweep here that varies the INPUT
/// rather than the result attribute.
///
/// Nearly every fnp fast path gates on C-contiguity and bails when it fails:
/// "Non-contiguous (transposed/strided) source ndarrays bail into the cold
/// extract; delegate" (take); "non-contiguous bails the fast path into the
/// transpose-copy extract (~55x slower). Delegate both to numpy" (signbit);
/// "A size-changing uintN view requires C-contiguous data; otherwise defer"
/// (concatenate). So for an awkward layout the answer comes from a DIFFERENT
/// path than the one every contiguous test exercises - numpy itself, or a cold
/// extract that rebuilds through a Vec - and the contiguous tests cannot see a
/// defect in either.
///
/// The nastiest layout is a BROADCAST view: 0-stride and read-only. A buffer
/// consumer that assumes `stride == itemsize` reads one element repeatedly or
/// walks off the end. liz1c showed fnp reaching an extract path for a dtype its
/// guard did not intend; a 0-stride input is the layout equivalent.
#[test]
fn non_contiguous_input_layouts_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import platform
import warnings

warnings.simplefilter("ignore")

base = np.arange(24, dtype=np.float64).reshape(4, 6)

layouts = [
    ("c_contiguous", lambda: base.copy()),                 # control
    ("transposed", lambda: base.copy().T),
    ("fortran_order", lambda: np.asfortranarray(base)),
    ("strided_rows", lambda: base.copy()[::2]),
    ("negative_stride", lambda: base.copy()[::-1]),
    ("strided_2d_slice", lambda: base.copy()[1:, ::2]),
    ("broadcast_view", lambda: np.broadcast_to(np.arange(6, dtype=np.float64), (4, 6))),
]

ops = [
    ("sum", lambda m, x: m.sum(x)),
    ("sum axis0", lambda m, x: m.sum(x, axis=0)),
    ("sum axis1", lambda m, x: m.sum(x, axis=1)),
    ("mean", lambda m, x: m.mean(x)),
    ("min", lambda m, x: m.min(x)),
    ("max", lambda m, x: m.max(x)),
    ("argmin", lambda m, x: m.argmin(x)),
    ("argmax", lambda m, x: m.argmax(x)),
    ("cumsum", lambda m, x: m.cumsum(x)),
    ("sort", lambda m, x: m.sort(x)),
    ("sort axis0", lambda m, x: m.sort(x, axis=0)),
    ("unique", lambda m, x: m.unique(x)),
    ("take", lambda m, x: m.take(x, [0, 2])),
    ("take axis1", lambda m, x: m.take(x, [0, 1], axis=1)),
    ("compress", lambda m, x: m.compress([True, False, True, False], x, axis=0)),
    ("count_nonzero", lambda m, x: m.count_nonzero(x)),
    ("nonzero", lambda m, x: m.nonzero(x)),
    ("clip", lambda m, x: m.clip(x, 5.0, 15.0)),
    ("abs", lambda m, x: m.abs(x)),
    ("sqrt", lambda m, x: m.sqrt(x)),
    ("negative", lambda m, x: m.negative(x)),
    ("signbit", lambda m, x: m.signbit(x)),
    ("flip", lambda m, x: m.flip(x)),
    ("ravel", lambda m, x: m.ravel(x)),
    ("reshape", lambda m, x: m.reshape(x, (-1,))),
    ("repeat", lambda m, x: m.repeat(x, 2)),
    ("diff", lambda m, x: m.diff(x)),
    ("concatenate self", lambda m, x: m.concatenate([x, x])),
    ("where", lambda m, x: m.where(x > 10.0, x, 0.0)),
    ("transpose", lambda m, x: m.transpose(x)),
]

def describe(value):
    if isinstance(value, tuple):          # nonzero returns a tuple of arrays
        return ("tuple",) + tuple(describe(item) for item in value)
    array = np.asarray(value)
    return (str(array.dtype), tuple(array.shape),
            np.round(array, 9).tolist())

def outcome(module, call, make):
    try:
        return ("ok", describe(call(module, make())))
    except Exception as exc:
        return ("err", type(exc).__name__)

ok = True
compared = 0
for layout_name, make in layouts:
    for op_name, call in ops:
        actual = outcome(fnp, call, make)
        expected = outcome(np, call, make)
        compared += 1
        if actual != expected:
            print(f"{layout_name}: {op_name}")
            print(f"  fnp   {str(actual)[:150]}")
            print(f"  numpy {str(expected)[:150]}")
            ok = False

# Preconditions: the layouts must ACTUALLY be non-contiguous, or this sweep is
# six spellings of the same contiguous array.
checks = {
    "transposed": lambda a: not a.flags.c_contiguous,
    "fortran_order": lambda a: a.flags.f_contiguous and not a.flags.c_contiguous,
    "strided_rows": lambda a: not a.flags.c_contiguous,
    "negative_stride": lambda a: not a.flags.c_contiguous,
    "strided_2d_slice": lambda a: not a.flags.c_contiguous,
    "broadcast_view": lambda a: 0 in a.strides,
}
for layout_name, make in layouts:
    check = checks.get(layout_name)
    if check is not None and not check(make()):
        print(f"PRECONDITION LOST: layout {layout_name} is not what it claims to be")
        ok = False
if compared < 200:
    print(f"PRECONDITION LOST: only {compared} comparisons ran")
    ok = False

print(ok)
print("oracle", platform.node(), np.__version__, compared)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.trim().lines().rev();
    let provenance = lines.next().unwrap_or("").trim();
    let verdict = lines.next().unwrap_or("").trim();
    assert_eq!(
        verdict, "True",
        "non-contiguous input layouts should match numpy ({provenance}): {result}"
    );
    Ok(())
}

/// EXOTIC dtype families - datetime64, timedelta64, structured/void, unicode,
/// bytes, object - against numpy.
///
/// The mechanical prior is `deadlock-audit-output-dtype-parity-sweep-liz1c`
/// exactly: take's residual guard read `itemsize == 8 || kind == "b"`, and
/// complex64's itemsize is 8 - the same width as f64/i64/u64 - so it satisfied
/// a guard written for the numeric widths, reached `extract_numeric_array`,
/// which cannot represent complex, and raised TypeError where numpy returned an
/// array. That was fixed at one site by adding `kind != "c"`.
///
/// datetime64 and timedelta64 are ALSO itemsize 8, with kinds 'M' and 'm'. Any
/// guard spelled as a WIDTH test rather than a KIND test has the same hole for
/// them; a structured `[('a','i4')]` is itemsize 4 and '<U2' is itemsize 8, so
/// the collision is not confined to 8 bytes.
///
/// These are the families the numeric extract path cannot represent at all, so
/// every wrapper must delegate them - which makes the expected behaviour easy
/// to state and any divergence unambiguous. Exception class is compared
/// alongside values, because for several of these pairs numpy itself raises and
/// reproducing the raise IS the contract.
#[test]
fn exotic_dtype_families_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import platform
import warnings

warnings.simplefilter("ignore")

structured = np.array([(1, 1.5), (2, 2.5), (3, 3.5), (4, 4.5)],
                      dtype=[("a", "i4"), ("b", "f8")])

samples = [
    ("float64_control", np.array([3.0, 1.0, 4.0, 1.0])),
    ("datetime64_D", np.array(["2026-01-01", "2026-01-03", "2026-01-02", "2026-01-02"],
                              dtype="datetime64[D]")),
    ("datetime64_ns", np.array([1, 3, 2, 2], dtype="datetime64[ns]")),
    ("timedelta64_D", np.array([1, 3, 2, 2], dtype="timedelta64[D]")),
    ("structured", structured),
    ("unicode_U3", np.array(["ab", "cd", "aa", "cd"], dtype="<U3")),
    ("bytes_S3", np.array([b"ab", b"cd", b"aa", b"cd"], dtype="S3")),
    ("object", np.array([3, 1, 4, 1], dtype=object)),
]

ops = [
    ("take list index", lambda m, x: m.take(x, [0, 2])),
    ("take array index", lambda m, x: m.take(x, np.array([0, 2]))),
    ("compress", lambda m, x: m.compress([True, False, True, False], x)),
    ("concatenate", lambda m, x: m.concatenate([x, x])),
    ("repeat", lambda m, x: m.repeat(x, 2)),
    ("tile", lambda m, x: m.tile(x, 2)),
    ("sort", lambda m, x: m.sort(x)),
    ("argsort", lambda m, x: m.argsort(x)),
    ("unique", lambda m, x: m.unique(x)),
    ("flip", lambda m, x: m.flip(x)),
    ("ravel", lambda m, x: m.ravel(x)),
    ("reshape", lambda m, x: m.reshape(x, (2, 2))),
    ("where self", lambda m, x: m.where(np.array([True, False, True, False]), x, x)),
    ("count_nonzero", lambda m, x: m.count_nonzero(x)),
    ("nonzero", lambda m, x: m.nonzero(x)),
    ("min", lambda m, x: m.min(x)),
    ("max", lambda m, x: m.max(x)),
    ("sum", lambda m, x: m.sum(x)),
    ("equal self", lambda m, x: m.equal(x, x)),
    ("isnan", lambda m, x: m.isnan(x)),
    ("signbit", lambda m, x: m.signbit(x)),
    ("astype str", lambda m, x: m.asarray(x).astype("<U24")),
]

def describe(value):
    if isinstance(value, tuple):
        return ("tuple",) + tuple(describe(item) for item in value)
    array = np.asarray(value)
    # repr of the values keeps datetimes/structured comparable without
    # depending on tolist()'s type mapping.
    return (str(array.dtype), tuple(array.shape), repr(array.tolist())[:400])

def outcome(module, call, x):
    try:
        return ("ok", describe(call(module, x)))
    except Exception as exc:
        return ("err", type(exc).__name__)

ok = True
compared = 0
for dtype_label, sample in samples:
    for op_label, call in ops:
        actual = outcome(fnp, call, sample)
        expected = outcome(np, call, sample)
        compared += 1
        if actual != expected:
            print(f"{dtype_label}: {op_label}")
            print(f"  fnp   {str(actual)[:170]}")
            print(f"  numpy {str(expected)[:170]}")
            ok = False

# Preconditions: the exotic dtypes must really be exotic, and must really
# collide on width with the numeric families - that collision is the whole
# reason this sweep exists.
if np.dtype("datetime64[D]").itemsize != 8:
    print("PRECONDITION LOST: datetime64[D] is no longer itemsize 8")
    ok = False
if np.dtype("timedelta64[D]").itemsize != 8:
    print("PRECONDITION LOST: timedelta64[D] is no longer itemsize 8")
    ok = False
if structured.dtype.kind != "V":
    print(f"PRECONDITION LOST: structured dtype kind is {structured.dtype.kind}, not V")
    ok = False
if compared < 150:
    print(f"PRECONDITION LOST: only {compared} comparisons ran")
    ok = False

print(ok)
print("oracle", platform.node(), np.__version__, compared)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.trim().lines().rev();
    let provenance = lines.next().unwrap_or("").trim();
    let verdict = lines.next().unwrap_or("").trim();
    assert_eq!(
        verdict, "True",
        "exotic dtype families should match numpy ({provenance}): {result}"
    );
    Ok(())
}

/// Keyword INTERACTIONS - the first sweep here that crosses parameters instead
/// of varying one attribute.
///
/// Every keyword touched this session is handled per-site by hand: `out=`
/// forces delegation in one wrapper and is forwarded in another; `keepdims` is
/// reconstructed by `keepdims_expand_axis` at some sites and by reshaping to
/// `vec![1usize; ndim]` at others; `where=` gates the native path in five
/// reductions and is forwarded in the rest; `dtype=` forces an accumulator
/// type. Each was verified ALONE.
///
/// The combination is where that hand-rolled arithmetic has to agree, because
/// **`out=`'s required shape is a function of `axis` AND `keepdims`**. A
/// wrapper that computes its own return shape correctly but derives the `out=`
/// check or the write from a different expression is wrong only in the crossed
/// case - which no single-keyword test can reach.
#[test]
fn keyword_interactions_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import platform
import warnings

warnings.simplefilter("ignore")

cube = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
mask3 = (cube % 3.0) != 0.0

reductions = ["sum", "prod", "mean", "min", "max", "any", "all"]
nan_reductions = ["nansum", "nanprod", "nanmean", "nanmin", "nanmax"]
axes = [None, 0, 1, -1, (0, 1)]

def described(value):
    # NaN must be substituted before comparison: nan != nan, so a plain tuple
    # compare reports a divergence whenever BOTH sides legitimately produce NaN.
    # where= on a fully-masked position does exactly that for mean, and the
    # first run of this sweep "failed" four such cases while printing identical
    # values on both sides. That is the probe lying, not fnp - the keepdims
    # sweep above already carries the same nan_to_num for the same reason.
    array = np.asarray(value)
    finite = np.nan_to_num(np.asarray(array, dtype=np.float64), nan=-987654.0,
                           posinf=1e300, neginf=-1e300) if array.dtype.kind == "f" else array
    return (str(array.dtype), tuple(array.shape), np.round(finite, 9).tolist())

def run(module, body):
    try:
        return ("ok",) + tuple(body(module))
    except Exception as exc:
        return ("err", type(exc).__name__)

ok = True
compared = 0

# ---- out= x axis x keepdims -------------------------------------------------
# The out buffer is allocated at the shape NUMPY says the result has for that
# (axis, keepdims) pair, so a wrapper deriving the shape differently fails here
# and only here.
for name in reductions + nan_reductions:
    for axis in axes:
        for keepdims in (False, True):
            kwargs = {"keepdims": keepdims}
            if axis is not None:
                kwargs["axis"] = axis
            try:
                reference = getattr(np, name)(cube, **kwargs)
            except Exception:
                continue
            reference = np.asarray(reference)

            def with_out(module, name=name, kwargs=kwargs, reference=reference):
                out = np.zeros(reference.shape, dtype=reference.dtype)
                result = getattr(module, name)(cube, out=out, **kwargs)
                return (result is out, described(out))

            actual, expected = run(fnp, with_out), run(np, with_out)
            compared += 1
            if actual != expected:
                print(f"out x axis x keepdims: {name} axis={axis} keepdims={keepdims}")
                print(f"  fnp   {str(actual)[:150]}")
                print(f"  numpy {str(expected)[:150]}")
                ok = False

            # And an out buffer that is WRONG for this (axis, keepdims) pair:
            # both sides must reject it the same way.
            def with_bad_out(module, name=name, kwargs=kwargs):
                out = np.zeros((7, 5), dtype=np.float64)
                result = getattr(module, name)(cube, out=out, **kwargs)
                return (result is out, described(out))

            actual, expected = run(fnp, with_bad_out), run(np, with_bad_out)
            compared += 1
            if actual != expected:
                print(f"bad-out x axis x keepdims: {name} axis={axis} keepdims={keepdims}")
                print(f"  fnp   {str(actual)[:150]}")
                print(f"  numpy {str(expected)[:150]}")
                ok = False

# ---- where= x axis x keepdims, and where= x dtype= --------------------------
for name in ["sum", "prod", "mean", "min", "max", "any", "all"]:
    for axis in axes:
        for keepdims in (False, True):
            def with_where(module, name=name, axis=axis, keepdims=keepdims):
                kwargs = {"where": mask3, "keepdims": keepdims}
                if axis is not None:
                    kwargs["axis"] = axis
                if name in ("min", "max"):
                    kwargs["initial"] = 0.0 if name == "max" else 1e9
                return (described(getattr(module, name)(cube, **kwargs)),)

            actual, expected = run(fnp, with_where), run(np, with_where)
            compared += 1
            if actual != expected:
                print(f"where x axis x keepdims: {name} axis={axis} keepdims={keepdims}")
                print(f"  fnp   {str(actual)[:150]}")
                print(f"  numpy {str(expected)[:150]}")
                ok = False

for name in ["sum", "prod", "mean"]:
    for dtype in ["float32", "float64"]:
        for axis in (None, 0, 1):
            def with_where_dtype(module, name=name, dtype=dtype, axis=axis):
                kwargs = {"where": mask3, "dtype": dtype}
                if axis is not None:
                    kwargs["axis"] = axis
                return (described(getattr(module, name)(cube, **kwargs)),)

            actual, expected = run(fnp, with_where_dtype), run(np, with_where_dtype)
            compared += 1
            if actual != expected:
                print(f"where x dtype x axis: {name} dtype={dtype} axis={axis}")
                print(f"  fnp   {str(actual)[:150]}")
                print(f"  numpy {str(expected)[:150]}")
                ok = False

# ---- dtype= x axis x keepdims x out= ---------------------------------------
for name in ["sum", "prod", "mean", "nansum", "nanmean"]:
    for axis in (None, 0, -1):
        for keepdims in (False, True):
            def with_dtype_out(module, name=name, axis=axis, keepdims=keepdims):
                kwargs = {"dtype": "float32", "keepdims": keepdims}
                if axis is not None:
                    kwargs["axis"] = axis
                reference = np.asarray(getattr(np, name)(cube, **kwargs))
                out = np.zeros(reference.shape, dtype=np.float32)
                result = getattr(module, name)(cube, out=out, **kwargs)
                return (result is out, described(out))

            actual, expected = run(fnp, with_dtype_out), run(np, with_dtype_out)
            compared += 1
            if actual != expected:
                print(f"dtype x axis x keepdims x out: {name} axis={axis} keepdims={keepdims}")
                print(f"  fnp   {str(actual)[:150]}")
                print(f"  numpy {str(expected)[:150]}")
                ok = False

# Preconditions: the cross must actually cross. If numpy's required out shape
# were the same with and without keepdims, every out= case above would collapse
# into the single-keyword case the earlier sweeps already cover.
if np.sum(cube, axis=0, keepdims=True).shape == np.sum(cube, axis=0).shape:
    print("PRECONDITION LOST: keepdims no longer changes the required out shape")
    ok = False
if np.sum(cube, axis=0, where=mask3).tolist() == np.sum(cube, axis=0).tolist():
    print("PRECONDITION LOST: the where= mask no longer changes the result")
    ok = False
if compared < 200:
    print(f"PRECONDITION LOST: only {compared} comparisons ran")
    ok = False

print(ok)
print("oracle", platform.node(), np.__version__, compared)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.trim().lines().rev();
    let provenance = lines.next().unwrap_or("").trim();
    let verdict = lines.next().unwrap_or("").trim();
    assert_eq!(
        verdict, "True",
        "keyword interactions should match numpy ({provenance}): {result}"
    );
    Ok(())
}

/// SEEDED RANDOMISED differential - the only non-curated test in this file.
///
/// Every other sweep here is a hand-written grid, so its blind spot is exactly
/// its author's blind spot. Randomised inputs do not share that: they reach
/// zero-size shapes, 0-d arrays, all-NaN and mixed NaN/inf inputs, denormals,
/// extreme magnitudes and high-rank shapes that no case list contains. Zero-size
/// input especially has never been swept here, and numpy's behaviour on it is
/// branch-heavy - sum of empty is 0.0, max of empty RAISES, mean of empty warns
/// and returns nan.
///
/// The seed is FIXED. A test whose inputs change per run is not a gate: it fails
/// for one agent, passes for the next, and the failure cannot be reproduced.
/// This is a differential regression test over a large arbitrary CONSTANT
/// corpus - genuine fuzzing lives in the `fuzz/` crates (docs/FUZZING.md).
/// Divergences print the iteration index and the full case so they replay.
#[test]
fn seeded_random_differential_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import platform
import warnings

warnings.simplefilter("ignore")

SEED = 20260809
rng = np.random.default_rng(SEED)

dtypes = ["float64", "float32", "int64", "int32", "int16", "uint8", "bool",
          "complex128", "float16"]
profiles = ["ordinary", "with_nan", "with_inf", "denormal", "extreme", "zeros"]

def make_array(dtype, shape, profile):
    size = int(np.prod(shape)) if shape else 1
    if dtype == "bool":
        data = rng.integers(0, 2, size=size).astype(bool)
    elif dtype.startswith(("int", "uint")):
        info = np.iinfo(dtype)
        if profile == "extreme":
            data = rng.choice([info.min, info.max, 0, 1, -1 if info.min < 0 else 1],
                              size=size).astype(dtype)
        elif profile == "zeros":
            data = np.zeros(size, dtype=dtype)
        else:
            lo = max(info.min, -1000)
            hi = min(info.max, 1000)
            data = rng.integers(lo, hi, size=size, endpoint=True).astype(dtype)
    else:
        base = rng.standard_normal(size)
        if profile == "with_nan" and size:
            base[rng.integers(0, size, size=max(1, size // 3))] = np.nan
        elif profile == "with_inf" and size:
            base[rng.integers(0, size, size=max(1, size // 3))] = np.inf
            if size > 1:
                base[0] = -np.inf
        elif profile == "denormal":
            base = base * 5e-324
        elif profile == "extreme":
            base = base * 1e308
        elif profile == "zeros":
            base = np.zeros(size)
            if size:
                base[0] = -0.0
        data = base.astype(dtype)
    return data.reshape(shape) if shape else np.asarray(data[0], dtype=dtype)

unary = ["sum", "prod", "mean", "min", "max", "any", "all", "argmin", "argmax",
         "cumsum", "sort", "unique", "ravel", "flip", "abs", "negative",
         "sign", "count_nonzero", "nonzero", "isnan", "isfinite", "signbit",
         "trim_zeros", "diff", "square", "conj"]

def described(value):
    if isinstance(value, tuple):
        return ("tuple",) + tuple(described(item) for item in value)
    array = np.asarray(value)
    if array.dtype.kind in "fc":
        finite = np.nan_to_num(array.astype("complex128" if array.dtype.kind == "c"
                                            else "float64"),
                               nan=-987654.0, posinf=1e300, neginf=-1e300)
        # Round hard: fnp and numpy may associate a reduction differently, and
        # this test is about SHAPE/DTYPE/branch parity, not last-ULP accuracy -
        # which the byte-exact goldens elsewhere already own.
        return (str(array.dtype), tuple(array.shape),
                repr(np.round(finite, 6).tolist())[:300])
    return (str(array.dtype), tuple(array.shape), repr(array.tolist())[:300])

cases = []
for i in range(400):
    dtype = str(rng.choice(dtypes))
    profile = str(rng.choice(profiles))
    ndim = int(rng.integers(0, 4))
    if ndim == 0:
        shape = ()
    else:
        shape = tuple(int(rng.integers(0, 5)) for _ in range(ndim))
    op = str(rng.choice(unary))
    axis = None
    if shape and rng.integers(0, 2):
        axis = int(rng.integers(-len(shape), len(shape)))
    cases.append((i, dtype, profile, shape, op, axis))

def invoke(module, x, op, axis):
    # The array is built ONCE per case and handed to both sides. Building it
    # inside invoke() drew fresh values from the shared rng for each module, so
    # fnp and numpy were compared on DIFFERENT inputs - the first run of this
    # test reported 131 of 400 "failures" that way, none of them real.
    fn = getattr(module, op)
    if axis is not None and op in ("sum", "prod", "mean", "min", "max", "any",
                                   "all", "argmin", "argmax", "cumsum", "sort",
                                   "flip", "diff"):
        return fn(x, axis=axis)
    return fn(x)

ok = True
failures = 0
zero_size = zero_dim = nan_bearing = 0
for index, dtype, profile, shape, op, axis in cases:
    if shape and 0 in shape:
        zero_size += 1
    if shape == ():
        zero_dim += 1
    if profile == "with_nan":
        nan_bearing += 1
    sample = make_array(dtype, shape, profile)
    try:
        actual = ("ok", described(invoke(fnp, sample.copy(), op, axis)))
    except Exception as exc:
        actual = ("err", type(exc).__name__)
    try:
        expected = ("ok", described(invoke(np, sample.copy(), op, axis)))
    except Exception as exc:
        expected = ("err", type(exc).__name__)
    if actual != expected:
        failures += 1
        if failures <= 8:
            print(f"case {index}: {op} dtype={dtype} profile={profile} "
                  f"shape={shape} axis={axis}")
            print(f"  fnp   {str(actual)[:190]}")
            print(f"  numpy {str(expected)[:190]}")
        ok = False

# Preconditions: the generated corpus must actually contain the interesting
# shapes. A generator that silently degenerated into 3-element float64 vectors
# would pass every case above while testing none of what this bead is for.
if zero_size < 5:
    print(f"PRECONDITION LOST: only {zero_size} zero-size cases generated")
    ok = False
if zero_dim < 5:
    print(f"PRECONDITION LOST: only {zero_dim} 0-d cases generated")
    ok = False
if nan_bearing < 5:
    print(f"PRECONDITION LOST: only {nan_bearing} NaN-bearing cases generated")
    ok = False

print(ok)
print("oracle", platform.node(), np.__version__,
      f"cases={len(cases)} zero_size={zero_size} zero_dim={zero_dim} "
      f"nan={nan_bearing} failures={failures} seed={SEED}")
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.trim().lines().rev();
    let provenance = lines.next().unwrap_or("").trim();
    let verdict = lines.next().unwrap_or("").trim();
    assert_eq!(
        verdict, "True",
        "seeded random differential should match numpy ({provenance}): {result}"
    );
    Ok(())
}
