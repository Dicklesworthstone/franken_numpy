//! Cross-cutting RETURN-TYPE parity sweep.
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
