//! Live NumPy-vs-FNP head-to-head for the `isnan(float64, 2^20)` survey cell.
//!
//! `rch exec -- cargo run --release -p fnp-python --example h2h_isnan` embeds the
//! extension module in this executable, so the benchmarked FNP arm and the live
//! NumPy incumbent run in one interpreter. Output intentionally goes to stderr:
//! remote RCH preserves stderr from the worker.

use pyo3::prelude::*;
use std::ffi::CString;

use fnp_python::fnp_python;

const HARNESS: &str = r#"
import hashlib, os, statistics, sys, timeit

def out(*items):
    print(*items, file=sys.stderr, flush=True)

import numpy as np
import fnp_python as fnp

out("python", sys.version.split()[0], "| numpy", np.__version__)
out("in-process ELF sha256", hashlib.sha256(open(EXE_PATH, "rb").read()).hexdigest())
out("host", os.uname().nodename, "| loadavg", [round(value, 2) for value in os.getloadavg()])

def interleave(numpy_stmt, fnp_stmt, globals_, calls, rounds=6):
    numpy_times, fnp_times = [], []
    for round_ in range(rounds):
        order = (("numpy", "fnp", "fnp", "numpy") if round_ % 2 == 0
                 else ("fnp", "numpy", "numpy", "fnp"))
        for arm in order:
            elapsed = timeit.timeit(
                numpy_stmt if arm == "numpy" else fnp_stmt,
                globals=globals_,
                number=calls,
            ) / calls * 1e9
            (numpy_times if arm == "numpy" else fnp_times).append(elapsed)
    return statistics.median(numpy_times), statistics.median(fnp_times)

def bytes_match(array):
    expected = np.isnan(array)
    actual = fnp.isnan(array)
    return (type(actual) is type(expected) and actual.dtype == expected.dtype
            and actual.shape == expected.shape and actual.tobytes() == expected.tobytes())

# Include empty, dimensionality, signed NaNs and a bit-pattern-created payload NaN;
# `isnan` must only classify, so output bytes are the complete contract.
parity_cases = [
    np.array([], dtype=np.float64),
    np.array([0.0], dtype=np.float64),
    np.array([0.0, -0.0, np.inf, -np.inf, np.nan, -np.nan], dtype=np.float64),
    np.array([0x7ff8000000000123, 0xfff8000000000456], dtype=np.uint64).view(np.float64),
    np.arange(105, dtype=np.float64).reshape(3, 5, 7),
]
parity_cases[-1].reshape(-1)[::13] = np.nan
bad = sum(not bytes_match(case) for case in parity_cases)
out("parity:", len(parity_cases), "cases,", bad, "divergences")
if bad:
    raise SystemExit("ABORTING: a timing ratio for a wrong result is meaningless")

N = 1 << 20
rng = np.random.default_rng(SEED)
finite = rng.standard_normal(N)
sparse_nan = finite.copy()
sparse_nan[::97] = np.nan

out("")
out("%-16s%14s%14s%9s%9s%9s" % ("corpus", "numpy_ns", "fnp_ns", "ratio", "A/A np", "A/A fnp"))
for label, array in (("all-finite", finite), ("sparse-nan", sparse_nan)):
    globals_ = {"np": np, "fnp": fnp, "array": array}
    calls = 24
    numpy_ns, fnp_ns = interleave("np.isnan(array)", "fnp.isnan(array)", globals_, calls)
    np_first, np_second = interleave("np.isnan(array)", "np.isnan(array)", globals_, calls)
    fnp_first, fnp_second = interleave("fnp.isnan(array)", "fnp.isnan(array)", globals_, calls)
    np_null, fnp_null = np_second / np_first, fnp_second / fnp_first
    valid = 0.97 <= np_null <= 1.03 and 0.97 <= fnp_null <= 1.03
    out("%-16s%14.1f%14.1f%8.3fx%9.3f%9.3f%s" % (
        label, numpy_ns, fnp_ns, fnp_ns / numpy_ns, np_null, fnp_null,
        "" if valid else "  VOID"))

# A fixed unrelated operation makes host movement visible without being used for
# an `isnan` claim.
control_globals = {"np": np, "fnp": fnp, "array": finite}
control_np, control_fnp = interleave("np.sum(array)", "fnp.sum(array)", control_globals, 30)
out("out-of-family np.sum f64 2^20: numpy %.1f fnp %.1f -> %.3fx" % (
    control_np, control_fnp, control_fnp / control_np))
"#;

fn main() -> PyResult<()> {
    pyo3::append_to_inittab!(fnp_python);
    Python::initialize();
    let exe = std::env::current_exe()
        .map_err(|error| PyErr::new::<pyo3::exceptions::PyOSError, _>(error.to_string()))?;
    let seed: i64 = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(556);
    Python::attach(|py| {
        let globals = pyo3::types::PyDict::new(py);
        globals.set_item("EXE_PATH", exe.to_string_lossy().as_ref())?;
        globals.set_item("SEED", seed)?;
        py.run(&CString::new(HARNESS).unwrap(), Some(&globals), None)
    })
}
