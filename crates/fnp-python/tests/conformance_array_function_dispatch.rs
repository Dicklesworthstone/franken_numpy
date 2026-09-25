//! NEP 18 `__array_function__` dispatch parity.
//!
//! NumPy routes every public array function through `_ArrayFunctionDispatcher`, so an argument
//! whose type overrides `__array_function__` (dask, xarray, pint, cupy, sparse, astropy's
//! `Quantity`) receives the call. fnp's native functions ignored the protocol and coerced such
//! an argument through `__array__`, answering with an ndarray - measured on 87 of the 306
//! dispatcher names fnp implements natively. These suites pin the protocol for EVERY such
//! name, not a sample, so a newly added native function cannot regress it unnoticed.

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
use support::fnp_script;

/// Every NumPy dispatcher fnp implements natively (top level and fnp submodules) hands a
/// duck-array argument to its `__array_function__`, with NumPy's own function object as
/// `func` - the key downstream libraries' handler tables use. The arity for each name is
/// whichever NumPy itself dispatches on first, so no name is skipped for want of a
/// hand-written call. Also asserts the sweep covered a substantial population, so an empty
/// or broken enumeration cannot pass.
#[test]
fn every_native_dispatcher_honours_array_function_overrides() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.simplefilter("ignore")
class Duck:
    def __array_function__(self, func, types, args, kwargs):
        return ("DISPATCHED", func)
    def __array__(self, dtype=None, copy=None):
        return np.arange(1.0, 4.0)
    def __len__(self):
        return 3
    def __getitem__(self, i):
        return np.arange(1.0, 4.0)[i]
    def __iter__(self):
        return iter(np.arange(1.0, 4.0))
d = Duck()
def dispatched(r):
    return type(r) is tuple and len(r) == 2 and isinstance(r[0], str) and r[0] == "DISPATCHED"
def module_pairs():
    yield "", np, fnp
    for sub in ("linalg", "fft", "lib", "lib.stride_tricks", "lib.npyio", "ma", "polynomial"):
        a, b = np, fnp
        try:
            for part in sub.split("."):
                a, b = getattr(a, part), getattr(b, part)
        except AttributeError:
            continue
        yield sub + ".", a, b
total, bad = 0, []
for prefix, npm, fm in module_pairs():
    for name in sorted(dir(npm)):
        theirs = getattr(npm, name, None)
        ours = getattr(fm, name, None)
        if ours is None or ours is theirs or not hasattr(theirs, "_implementation"):
            continue
        for args in [(d,), (d, d), (d, d, d), ([d, d],), (d, 1), (d, 0), (1, d)]:
            try:
                r = theirs(*args)
            except Exception:
                continue
            if dispatched(r):
                break
        else:
            continue
        total += 1
        try:
            r = ours(*args)
        except Exception as exc:
            r = f"raised {type(exc).__name__}"
        if not (dispatched(r) and r[1] is theirs):
            bad.append(prefix + name)
print(total, bad if bad else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let last = result.lines().last().unwrap_or("");
    let (total, verdict) = last.split_once(' ').unwrap_or(("0", last));
    let total: usize = total.parse().unwrap_or(0);
    assert!(
        total >= 250,
        "the sweep must reach the dispatcher population (saw {total}): {result}"
    );
    assert_eq!(
        verdict.trim(),
        "True",
        "native functions must dispatch __array_function__ like numpy: {result}"
    );
    Ok(())
}

/// The dispatcher sends a SMALL first operand to numpy's own function only for the dtypes where
/// numpy answers it faster. `unique` on an integer or bool array stays native (5-50x faster than
/// numpy's there) and so does an integer `sort` (the small-integer flat sorts); float `unique`,
/// float/bool `sort` and every small `argsort` go to numpy. Checked with a spy on the live numpy
/// function, and every answer must be numpy's bytes either way. With the float64-only thresholds
/// this replaced, the int16/int64/bool `unique` and int64 `sort` rows were sent to numpy.
#[test]
fn small_operand_gate_routes_by_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(8)
def operand(dt, n):
    return (rng.random(n) > 0.5) if dt == "?" else rng.integers(0, 50, n).astype(dt)
EXPECT_NUMPY = {
    ("unique", "f8"): True, ("unique", "f4"): True, ("unique", "i8"): False,
    ("unique", "i2"): False, ("unique", "?"): False,
    ("sort", "f8"): True, ("sort", "?"): True, ("sort", "i8"): False, ("sort", "i4"): False,
    ("argsort", "f8"): True, ("argsort", "i8"): True,
}
bad = []
for (name, dt), expect in EXPECT_NUMPY.items():
    for n in (16, 256):
        a = operand(dt, n)
        original = getattr(np, name)
        calls = []
        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)
        setattr(np, name, spy)
        try:
            ours = getattr(fnp, name)(a)
        finally:
            setattr(np, name, original)
        theirs = original(a)
        if bool(calls) != expect:
            bad.append((name, dt, n, "numpy" if calls else "native"))
        if ours.dtype != theirs.dtype or ours.tobytes() != theirs.tobytes():
            bad.append((name, dt, n, "bytes"))
print(bad if bad else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "True",
        "the small-operand gate routed a dtype the wrong way: {result}"
    );
    Ok(())
}

/// The protocol is reached through sequences too (`concatenate`/`stack` operands, `block`'s
/// nested lists) and keyword arguments (`out=`), and an ndarray SUBCLASS that overrides the
/// hook is foreign while one that keeps ndarray's (MaskedArray, matrix) is not. Plain ndarray
/// and data-list calls are the controls: they must keep answering with ordinary arrays.
#[test]
fn array_function_dispatch_reaches_sequences_keywords_and_subclasses() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.simplefilter("ignore")
class Duck:
    def __array_function__(self, func, types, args, kwargs):
        return ("DISPATCHED", func.__name__)
    def __array__(self, dtype=None, copy=None):
        return np.arange(1.0, 4.0)
class Overriding(np.ndarray):
    def __array_function__(self, func, types, args, kwargs):
        return ("DISPATCHED", func.__name__)
d = Duck()
o = np.arange(3.0).view(Overriding)
a = np.arange(3.0)
def outcome(fn):
    try:
        r = fn()
    except Exception as exc:
        return ("raised", type(exc).__name__)
    if isinstance(r, tuple) and r and r[0] == "DISPATCHED":
        return r
    r = np.asarray(r) if not isinstance(r, np.ndarray) else r
    return (type(r).__name__, r.dtype.str, r.tolist())
cases = [
    lambda m: m.concatenate([a, d]),
    lambda m: m.lexsort((a, d)),
    lambda m: m.choose([0, 1, 0], [a, d]),
    lambda m: m.select([a > 0], [d]),
    lambda m: m.einsum("i,i", a, d),
    lambda m: m.stack((a, a, d)),
    lambda m: m.block([[a, a], [a, d]]),
    lambda m: m.concatenate([a, a], out=d),
    lambda m: m.flip(o),
    lambda m: m.isclose(o, o),
    lambda m: m.outer(o, o),
    lambda m: m.cross(o, o),
    lambda m: m.mean(o),
    lambda m: m.linalg.norm(d),
    lambda m: m.fft.fft(d),
    lambda m: m.concatenate([a, a]),
    lambda m: m.cumsum(a),
    lambda m: m.sum([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
    lambda m: m.mean(np.ma.masked_array([1.0, 2.0, 3.0], mask=[0, 1, 0])),
    lambda m: m.unique(np.array([3, 1, 3])),
]
bad = [i for i, c in enumerate(cases) if outcome(lambda: c(fnp)) != outcome(lambda: c(np))]
print(bad if bad else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "True",
        "dispatch through sequences/keywords/subclasses must match numpy: {result}"
    );
    Ok(())
}

/// A dispatcher reports NumPy's signature and docs, pickles by reference to itself (including
/// submodule functions - the plain native `fft.fft` could not be pickled at all), and keeps
/// its name. numpy's own TestTextSignatures::test_c_func_dispatcher_signature cases are
/// included.
#[test]
fn dispatchers_report_numpys_signature_and_pickle_by_reference() -> Result<(), String> {
    let script = fnp_script(
        r#"
import inspect, pickle, sys
sys.modules[fnp.__name__] = fnp
names = ["inner", "where", "lexsort", "can_cast", "min_scalar_type", "result_type", "dot",
         "vdot", "bincount", "ravel_multi_index", "unravel_index", "copyto", "putmask",
         "unpackbits", "shares_memory", "may_share_memory", "is_busday", "busday_offset",
         "busday_count", "datetime_as_string", "concatenate", "empty_like", "mean", "clip"]
def signature_or_error(fn):
    # numpy < 2.4's C dispatchers carry no signature at all (inspect raises ValueError).
    try:
        return str(inspect.signature(fn))
    except (ValueError, TypeError) as exc:
        return type(exc).__name__
bad = []
for name in names:
    ours, theirs = getattr(fnp, name), getattr(np, name)
    if ours is theirs:
        continue
    if signature_or_error(ours) != signature_or_error(theirs):
        bad.append(("signature", name))
    if ours.__name__ != theirs.__name__ or ours.__doc__ != theirs.__doc__:
        bad.append(("name/doc", name))
for obj in (fnp.concatenate, fnp.mean, fnp.linalg.norm, fnp.fft.fft, fnp.lib.stride_tricks.sliding_window_view):
    try:
        if pickle.loads(pickle.dumps(obj)) is not obj:
            bad.append(("pickle-identity", obj.__name__))
    except Exception as exc:
        bad.append(("pickle", obj.__name__, type(exc).__name__))
print(bad if bad else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "True",
        "dispatcher metadata/pickling must match numpy: {result}"
    );
    Ok(())
}
