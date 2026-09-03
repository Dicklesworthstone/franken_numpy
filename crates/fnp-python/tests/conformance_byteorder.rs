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
