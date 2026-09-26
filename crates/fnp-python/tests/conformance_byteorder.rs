//! Non-native byte order parity.
//!
//! Every zero-copy route in `fnp_python` reads operand bytes through a typed buffer and
//! interprets them in host byte order. pyo3 0.28.3 accepts a leading `>` in the buffer
//! format on little-endian hosts, so before the crate-local `PyBuffer` guard landed a
//! big-endian array (`dtype='>f8'`) passed the typed-buffer check and its bytes were read
//! raw: measured 2026-09-02, 15 of 65 ops returned wrong values for `>f8` input and
//! `sort` on `>i8` returned garbage, with no error raised.
//!
//! This suite is the named probe for that defect class (bead deadlock-audit-2kqw3). It
//! sweeps a matrix of entry points over every big-endian dtype numpy can build and requires,
//! per cell, that fnp and numpy either both raise the same exception type or both return the
//! same bytes, dtype (byte order included) and shape. A native route that silently computes
//! on swapped bytes fails on values; a route that "declines by raising" fails on raise parity.

use std::ffi::CString;

use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyModule};

const SWEEP: &str = r#"
import warnings
import numpy as np
warnings.simplefilter("ignore")

rng = np.random.default_rng(2026)
base_f = rng.standard_normal(64)
base_f[3] = np.nan
base_f[7] = np.inf
base_f[11] = -np.inf
base_f[13] = -0.0
base_i = rng.integers(-1000, 1000, 64)
base_u = rng.integers(0, 2000, 64)
base_c = rng.standard_normal(64) + 1j * rng.standard_normal(64)
base_c[5] = complex(np.nan, 1.0)

def parity(ours, theirs):
    if isinstance(ours, tuple) and isinstance(theirs, tuple):
        return len(ours) == len(theirs) and all(parity(a, b) for a, b in zip(ours, theirs))
    if isinstance(ours, list) and isinstance(theirs, list):
        return len(ours) == len(theirs) and all(parity(a, b) for a, b in zip(ours, theirs))
    if isinstance(theirs, np.ndarray) or isinstance(ours, np.ndarray):
        if not (isinstance(ours, np.ndarray) and isinstance(theirs, np.ndarray)):
            return False
        if ours.dtype != theirs.dtype or ours.shape != theirs.shape:
            return False
        return ours.tobytes() == theirs.tobytes()
    if type(ours) is not type(theirs):
        return False
    try:
        if ours != ours and theirs != theirs:
            return True
    except Exception:
        pass
    return bool(ours == theirs)

def common(ns, x):
    return {
        "sum": lambda: ns.sum(x),
        "mean": lambda: ns.mean(x),
        "std": lambda: ns.std(x),
        "var": lambda: ns.var(x),
        "max": lambda: ns.max(x),
        "min": lambda: ns.min(x),
        "argmax": lambda: ns.argmax(x),
        "argmin": lambda: ns.argmin(x),
        "cumsum": lambda: ns.cumsum(x),
        "cumprod": lambda: ns.cumprod(x[:8]),
        "prod": lambda: ns.prod(x[:8]),
        "any": lambda: ns.any(x),
        "all": lambda: ns.all(x),
        "diff": lambda: ns.diff(x),
        "count_nonzero": lambda: ns.count_nonzero(x),
        "sort": lambda: ns.sort(x),
        "argsort": lambda: ns.argsort(x),
        "unique": lambda: ns.unique(x),
        "abs": lambda: ns.abs(x),
        "negative": lambda: ns.negative(x),
        "square": lambda: ns.square(x),
        "sign": lambda: ns.sign(x),
        "ptp": lambda: ns.ptp(x),
        "searchsorted": lambda: ns.searchsorted(np.sort(x), x),
        "isin": lambda: ns.isin(x, x[:8]),
        "tile": lambda: ns.tile(x, 2),
        "concatenate": lambda: ns.concatenate([x, x]),
        "where": lambda: ns.where(x > 0, x, x),
        "take": lambda: ns.take(x, [0, 5, 9]),
        "repeat": lambda: ns.repeat(x, 2),
        "flip": lambda: ns.flip(x),
        "roll": lambda: ns.roll(x, 3),
        "add": lambda: ns.add(x, x),
        "multiply": lambda: ns.multiply(x, x),
        "maximum": lambda: ns.maximum(x, x[::-1]),
        "equal": lambda: ns.equal(x, x),
        "histogram": lambda: ns.histogram(x),
    }

def floats(ns, x):
    return {
        "isnan": lambda: ns.isnan(x),
        "isfinite": lambda: ns.isfinite(x),
        "isinf": lambda: ns.isinf(x),
        "signbit": lambda: ns.signbit(x),
        "floor": lambda: ns.floor(x),
        "ceil": lambda: ns.ceil(x),
        "rint": lambda: ns.rint(x),
        "round": lambda: ns.round(x, 2),
        "nan_to_num": lambda: ns.nan_to_num(x),
        "nanmax": lambda: ns.nanmax(x),
        "nanmin": lambda: ns.nanmin(x),
        "nansum": lambda: ns.nansum(x),
        "nanmean": lambda: ns.nanmean(x),
        "nanargmax": lambda: ns.nanargmax(x),
        "sqrt": lambda: ns.sqrt(np.abs(x)),
        "exp": lambda: ns.exp(x),
        "log": lambda: ns.log(np.abs(x) + 1),
        "median": lambda: ns.median(x),
        "nanmedian": lambda: ns.nanmedian(x),
        "percentile": lambda: ns.percentile(x, 50),
        "clip": lambda: ns.clip(x, -1.0, 1.0),
        "divide": lambda: ns.divide(x, 2.0),
        "interp": lambda: ns.interp(x, np.linspace(-3, 3, 16), np.linspace(0, 1, 16)),
    }

def ints(ns, x):
    return {
        "left_shift": lambda: ns.left_shift(x, 2),
        "bitwise_and": lambda: ns.bitwise_and(x, 7),
        "floor_divide": lambda: ns.floor_divide(x, 3),
        "remainder": lambda: ns.remainder(x, 7),
        "clip": lambda: ns.clip(x, 1, 10),
        "bincount": lambda: ns.bincount(np.abs(x)),
    }

def complexes(ns, x):
    return {
        "real": lambda: ns.real(x),
        "imag": lambda: ns.imag(x),
        "conj": lambda: ns.conj(x),
        "nan_to_num": lambda: ns.nan_to_num(x),
        "round": lambda: ns.round(x, 2),
    }

cases = []
for code in (">f8", ">f4", ">f2"):
    cases.append((code, np.asarray(base_f, dtype=code), (common, floats)))
for code in (">i8", ">i4", ">i2"):
    cases.append((code, np.asarray(base_i, dtype=code), (common, ints)))
for code in (">u8", ">u4", ">u2"):
    cases.append((code, np.asarray(base_u, dtype=code), (common, ints)))
cases.append((">c16", np.asarray(base_c, dtype=">c16"), (common, complexes)))
cases.append((">c8", np.asarray(base_c, dtype=">c8"), (common, complexes)))

failures = []
cells = 0
for code, x, families in cases:
    assert x.dtype.byteorder == ">", (code, x.dtype.byteorder)
    for family in families:
        theirs_ops = family(np, x)
        ours_ops = family(fnp, x)
        for name, run_theirs in theirs_ops.items():
            cells += 1
            try:
                theirs = run_theirs()
                theirs_exc = None
            except Exception as exc:
                theirs, theirs_exc = None, type(exc).__name__
            try:
                ours = ours_ops[name]()
                ours_exc = None
            except Exception as exc:
                ours, ours_exc = None, type(exc).__name__
            if theirs_exc or ours_exc:
                if theirs_exc != ours_exc:
                    failures.append(f"{code} {name}: numpy {theirs_exc or 'returned'} vs fnp {ours_exc or 'returned'}")
                continue
            if not parity(ours, theirs):
                failures.append(f"{code} {name}: values differ (numpy {theirs!r} vs fnp {ours!r})")
"#;

#[test]
fn big_endian_inputs_match_numpy_values_and_raises() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "fnp_python_byteorder_test").expect("test module");
        fnp_python(&module).expect("initialize fnp_python test module");
        let globals = PyDict::new(py);
        globals
            .set_item("fnp", &module)
            .expect("bind fnp into the sweep globals");
        let script = CString::new(SWEEP).expect("sweep script is valid C string");
        py.run(&script, Some(&globals), None)
            .expect("byte-order sweep executes");
        let cells: usize = globals
            .get_item("cells")
            .expect("cells lookup")
            .expect("cells present")
            .extract()
            .expect("cells is an integer");
        let failures: Vec<String> = globals
            .get_item("failures")
            .expect("failures lookup")
            .expect("failures present")
            .extract()
            .expect("failures is a list of strings");
        assert!(
            cells >= 500,
            "sweep covered only {cells} cells; the matrix was expected to be at least 500 cells"
        );
        assert!(
            failures.is_empty(),
            "{} of {cells} big-endian cells diverge from numpy:\n  {}",
            failures.len(),
            failures.join("\n  ")
        );
    });
}

/// The sweep above uses 64-element arrays, below every size gate, so the parallel native routes
/// never ran on a big-endian operand. At 2^20 elements (finite, moderate values: a NaN makes many
/// routes decline, which hid the defect) the complex routes read '>c16'/'>c8' operands through
/// `.view(float64)`/`.view(float32)`, whose dtype IS native, so the swapped bytes were taken as
/// native floats: exp/sin/cos/sinh/cosh/sign/multiply/sort/sort_complex/intersect1d answered
/// garbage (`exp` of `>c16` [1+2j] was 1+3e-322j) with no error. The same routes viewed a
/// non-contiguous complex operand (strided, reversed, broadcast) before checking contiguity and
/// raised ValueError where numpy answers; and concatenate built '>f4' / mixed uint32+int32 inputs
/// through a kind-widening extraction, answering float64 / float64 where numpy answers float32 /
/// int64. The np.strings routes read a '>U' array's code points byte-swapped the same way
/// (find/rfind/count/zfill/center/ljust/rjust). 101 of the first 472 cells failed on 9fca9306 (numpy
/// 2.4.3 and 2.3.5); the 12 temporal / trim_zeros cells added after them failed 4 on 94eb055a.
const LARGE_SWEEP: &str = r#"
import warnings
import numpy as np
warnings.simplefilter("ignore")
rng = np.random.default_rng(9)
N = 1 << 20

def make(dt, n):
    d = np.dtype(dt)
    if d.kind in "iu":
        v = rng.integers(0 if d.kind == "u" else -100, 100, n).astype(d)
    elif d.kind == "f":
        v = (rng.integers(-40, 40, n) / 4).astype(d)
    else:
        v = (rng.integers(-40, 40, n) / 4 + 1j * (rng.integers(-40, 40, n) / 8)).astype(d)
    return v

def outcome(call):
    try:
        value = call()
    except Exception as ex:
        return (type(ex).__name__, str(ex)[:80])
    if isinstance(value, tuple):
        return tuple(outcome(lambda v=v: v) for v in value)
    value = np.asarray(value)
    return ("ok", value.dtype.str, value.shape, value.tobytes())

unary = ["exp", "sin", "cos", "sinh", "cosh", "tanh", "sign", "sqrt", "abs", "negative", "square",
         "angle", "nan_to_num", "conjugate", "isfinite", "sort", "sort_complex", "argsort", "unique",
         "cumsum", "sum", "mean", "prod"]
binary = ["multiply", "divide", "add", "subtract", "equal", "power"]
setops = ["union1d", "intersect1d", "setdiff1d", "setxor1d", "isin"]

cells = 0
failures = []
def check(label, call):
    global cells
    cells += 1
    ours, theirs = outcome(lambda: call(fnp)), outcome(lambda: call(np))
    if ours != theirs:
        failures.append(f"{label}: fnp={str(ours)[:120]} numpy={str(theirs)[:120]}")

for dt in ("c16", "c8"):
    native = make(dt, N)
    swapped = native.astype(native.dtype.newbyteorder(">"))
    wide = make(dt, 2 * N)
    operands = {"big-endian": swapped, "strided": wide[::2], "reversed": native[::-1],
                "broadcast": np.broadcast_to(native[7], (N,)), "offset": wide[1:N + 1]}
    for lname, a in operands.items():
        for name in unary:
            check(f"{name} {lname} {dt}", lambda m, name=name, a=a: getattr(m, name)(a))
        for name in binary:
            check(f"{name} {lname} {dt}", lambda m, name=name, a=a: getattr(m, name)(a, a[::-1]))
            check(f"{name} {lname}+native {dt}", lambda m, name=name, a=a: getattr(m, name)(a, native))
        for name in setops:
            check(f"{name} {lname} {dt}", lambda m, name=name, a=a: getattr(m, name)(a, a[::3]))
        check(f"searchsorted {lname} {dt}", lambda m, a=a: m.searchsorted(np.sort(a), a[:5000]))
for x, y in [("u4", "i4"), ("i4", ">i4"), ("f4", ">f4"), (">f4", ">f4"), (">i4", ">u4"), ("i2", "i4"),
             ("f4", "f2"), ("u1", "i2"), ("?", "i2"), ("u2", "u4"), (">c16", "c16"), ("f8", ">f8")]:
    a, b = np.arange(N).astype(x), np.arange(N).astype(y)
    check(f"concatenate {x}+{y}", lambda m, a=a, b=b: m.concatenate([a, b]))
# '>m8'/'>M8': the timedelta add/subtract route read both operands through `.view(int64)` and
# added the swapped words as native integers (786,432 of 2^20 sums wrong once a carry crossed a
# byte); trim_zeros answered a native copy where numpy returns a view of the '>f8' input.
td = rng.integers(-10 ** 6, 10 ** 6, N).astype("m8[s]")
step = rng.integers(1, 10 ** 4, N).astype("m8[s]")
for label, t in (("native", lambda x: x), ("big-endian", lambda x: x.astype(x.dtype.newbyteorder(">")))):
    for name in ("add", "subtract", "floor_divide", "remainder"):
        check(f"timedelta {name} {label}", lambda m, name=name, t=t: getattr(m, name)(t(td), t(step)))
    check(f"datetime + timedelta {label}", lambda m, t=t: m.add(t(td.astype("M8[s]")), t(step)))
    padded = np.concatenate([np.zeros(10), rng.integers(-40, 40, 5000) / 4, np.zeros(9)])
    check(f"trim_zeros {label}", lambda m, t=t, p=padded: m.trim_zeros(t(p)))
# '>U' strings: the native np.strings routes read code points through `.view(uint32)`, and a
# big-endian array's came out byte-swapped (find/count/zfill/center answered wrongly).
words = np.array([f"s{v:05d}" for v in rng.integers(0, 50000, 1 << 18)])
for label, s in (("native", words), ("big-endian", words.astype(">U6"))):
    for name, call in {
        "find": lambda m, s: m.strings.find(s, "1"), "rfind": lambda m, s: m.strings.rfind(s, "1"),
        "count": lambda m, s: m.strings.count(s, "1"), "zfill": lambda m, s: m.strings.zfill(s, 10),
        "center": lambda m, s: m.strings.center(s, 12), "ljust": lambda m, s: m.strings.ljust(s, 9),
        "rjust": lambda m, s: m.strings.rjust(s, 9), "upper": lambda m, s: m.strings.upper(s),
        "lower": lambda m, s: m.strings.lower(s), "swapcase": lambda m, s: m.strings.swapcase(s),
        "capitalize": lambda m, s: m.strings.capitalize(s), "str_len": lambda m, s: m.strings.str_len(s),
        "startswith": lambda m, s: m.strings.startswith(s, "s0"), "replace": lambda m, s: m.strings.replace(s, "1", "xy"),
        "strip": lambda m, s: m.strings.strip(s, "s"), "translate": lambda m, s: m.strings.translate(s, {49: 50}),
        "isdigit": lambda m, s: m.strings.isdigit(s), "isalpha": lambda m, s: m.strings.isalpha(s),
        "multiply": lambda m, s: m.strings.multiply(s, 3), "expandtabs": lambda m, s: m.strings.expandtabs(s, 4),
        "partition": lambda m, s: m.strings.partition(s, "1"), "rpartition": lambda m, s: m.strings.rpartition(s, "1"),
        "sort": lambda m, s: m.sort(s), "unique": lambda m, s: m.unique(s), "isin": lambda m, s: m.isin(s, s[:500]),
    }.items():
        check(f"strings.{name} {label}", lambda m, call=call, s=s: call(m, s))
"#;

#[test]
fn large_big_endian_and_non_contiguous_complex_operands_match_numpy() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "fnp_python_byteorder_large_test").expect("test module");
        fnp_python(&module).expect("initialize fnp_python test module");
        let globals = PyDict::new(py);
        globals
            .set_item("fnp", &module)
            .expect("bind fnp into the sweep globals");
        let script = CString::new(LARGE_SWEEP).expect("sweep script is valid C string");
        py.run(&script, Some(&globals), None)
            .expect("large byte-order sweep executes");
        let cells: usize = globals
            .get_item("cells")
            .expect("cells lookup")
            .expect("cells present")
            .extract()
            .expect("cells is an integer");
        let failures: Vec<String> = globals
            .get_item("failures")
            .expect("failures lookup")
            .expect("failures present")
            .extract()
            .expect("failures is a list of strings");
        assert!(
            cells >= 400,
            "large sweep covered only {cells} cells; expected at least 400"
        );
        assert!(
            failures.is_empty(),
            "{} of {cells} large big-endian / non-contiguous cells diverge from numpy:\n  {}",
            failures.len(),
            failures.join("\n  ")
        );
    });
}

/// The same defect class on the REQUEST side: a non-native `dtype=` argument. The shared dtype
/// parser read `np.dtype(dtype).name`, which drops the byte order ('>i4' and '<i4' are both
/// "int32"). As a result eye, identity, indices, fromstring, loadtxt, genfromtxt, masked_all
/// and linspace answered a big-endian request with a native array; fromfile read the file's
/// bytes as native, i.e. wrong VALUES; Generator.integers accepted a dtype numpy refuses; and
/// SeedSequence.generate_state returned native uint32 for '>u4' (bead rc0923 .8). 50 of these
/// cells failed on the pre-fix build. masked_all is compared on dtype, shape, mask and fill
/// value, because its data is np.empty's.
const REQUEST_SWEEP: &str = r#"
import io
import os
import tempfile
import warnings
import numpy as np
warnings.simplefilter("ignore")

def outcome(call):
    try:
        value = call()
    except Exception as ex:
        return (type(ex).__name__, str(ex))
    if isinstance(value, np.ma.MaskedArray):
        # masked_all's data is np.empty's: only dtype, shape, mask and fill value are defined.
        return ("masked", value.dtype.str, value.shape, value.mask.tolist(), repr(value.fill_value))
    value = np.asarray(value)
    return ("ok", value.dtype.str, value.shape, value.tobytes())

handle = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
handle.write(np.arange(4, dtype="<i4").tobytes())
handle.close()
matrix = np.arange(6, dtype=np.float64).reshape(2, 3)
cells = 0
failures = []
for dt in (">i4", ">f8", ">u4", ">i8", ">i2", ">f4", "<i4", "=f8", "i4"):
    calls = {
        "eye": lambda m: m.eye(3, dtype=dt),
        "identity": lambda m: m.identity(3, dtype=dt),
        "indices": lambda m: m.indices((2, 3), dtype=dt),
        "fromstring": lambda m: m.fromstring("1 2 3", sep=" ", dtype=dt),
        "fromfile": lambda m: m.fromfile(handle.name, dtype=dt),
        "loadtxt": lambda m: m.loadtxt(io.StringIO("1 2\n3 4\n"), dtype=dt),
        "genfromtxt": lambda m: m.genfromtxt(io.StringIO("1 2\n3 4\n"), dtype=dt),
        "masked_all": lambda m: m.ma.masked_all((3,), dtype=dt),
        "linspace": lambda m: m.linspace(0, 4, 5, dtype=dt),
        "zeros": lambda m: m.zeros(3, dtype=dt),
        "full": lambda m: m.full(3, 2, dtype=dt),
        "arange": lambda m: m.arange(5, dtype=dt),
        "array": lambda m: m.array([1, 2, 3], dtype=dt),
        "ascontiguousarray": lambda m: m.ascontiguousarray(matrix.T, dtype=dt),
        "asfortranarray": lambda m: m.asfortranarray(matrix, dtype=dt),
        "full_like": lambda m: m.full_like(matrix, 3, dtype=dt),
        "sum": lambda m: m.sum(matrix, axis=0, dtype=dt),
        "cumsum": lambda m: m.cumsum(matrix, dtype=dt),
        "tri": lambda m: m.tri(3, dtype=dt),
        "integers": lambda m: m.random.default_rng(1).integers(0, 9, size=3, dtype=dt),
        "randint": lambda m: m.random.RandomState(1).randint(0, 9, size=3, dtype=dt),
        "generate_state": lambda m: m.random.SeedSequence(5).generate_state(3, dtype=dt),
    }
    for name, call in calls.items():
        cells += 1
        ours, theirs = outcome(lambda: call(fnp)), outcome(lambda: call(np))
        if ours != theirs:
            failures.append(f"{name}(dtype={dt!r}): fnp={str(ours)[:150]} numpy={str(theirs)[:150]}")
os.unlink(handle.name)
"#;

#[test]
fn non_native_dtype_requests_match_numpy_dtype_bytes_and_raises() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "fnp_python_byteorder_request_test").expect("test module");
        fnp_python(&module).expect("initialize fnp_python test module");
        let globals = PyDict::new(py);
        globals
            .set_item("fnp", &module)
            .expect("bind fnp into the sweep globals");
        let script = CString::new(REQUEST_SWEEP).expect("sweep script is valid C string");
        py.run(&script, Some(&globals), None)
            .expect("byte-order request sweep executes");
        let cells: usize = globals
            .get_item("cells")
            .expect("cells lookup")
            .expect("cells present")
            .extract()
            .expect("cells is an integer");
        let failures: Vec<String> = globals
            .get_item("failures")
            .expect("failures lookup")
            .expect("failures present")
            .extract()
            .expect("failures is a list of strings");
        assert_eq!(cells, 198, "request matrix drifted");
        assert!(
            failures.is_empty(),
            "{} of {cells} dtype= requests diverge from numpy:\n  {}",
            failures.len(),
            failures.join("\n  ")
        );
    });
}
