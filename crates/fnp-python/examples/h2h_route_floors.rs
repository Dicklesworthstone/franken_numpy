//! DOES EACH NATIVE ROUTE HAVE A CROSSOVER, AND DOES ANYTHING MARK IT? (`deadlock-audit-66w2d`)
//!
//! `np.concatenate` turned out to engage its native routes at EVERY size while losing to NumPy
//! below 8 MiB of output - by 4.6x to 7.7x on ten of eleven dtypes. Nothing marked the crossover;
//! adding the floor was the largest single lever of this lane. That defect is not concatenate-
//! shaped. Any route that engages unconditionally has it whenever its native entry costs more
//! than NumPy's whole call, which is exactly what the small-n board keeps reporting:
//!
//!     isnan n=64  1.858x (+678 ns)    sqrt n=64  1.916x (+745)    abs n=64 1.627x (+504)
//!     max n=64    1.150x (+439)       mean n=64  1.082x (+432)    divide n=64 1.434x (+374)
//!     multiply    1.337x (+302)       add n=64   1.314x (+287)    sum n=64 1.110x (+335)
//!
//! So this sweeps each of those routes across seven sizes and asks ONE question per route: is
//! there a size at which it starts winning, and where? A route that never wins should not have a
//! native path at all; a route that wins only above some n needs a floor there. Both answers are
//! actionable and neither can be read off a single-size board - which is the whole reason the
//! campaign's 24-cell board reported "no actionable loss" while these were live.

use pyo3::prelude::*;
use std::ffi::CString;

use fnp_python::fnp_python;

const HARNESS: &str = r#"
import hashlib, os, statistics, sys, timeit
def out(*a):
    print(*a, file=sys.stderr, flush=True)
import numpy as np
import fnp_python as fnp

out("python", sys.version.split()[0], "| numpy", np.__version__)
out("in-process ELF sha256", hashlib.sha256(open(EXE_PATH, "rb").read()).hexdigest())
out("host", os.uname().nodename, "| loadavg", [round(x, 2) for x in os.getloadavg()])

def inter(sa, sb, g, k, rounds=4):
    ta, tb = [], []
    for r in range(rounds):
        for w in (("a","b","b","a") if r % 2 == 0 else ("b","a","a","b")):
            t = timeit.timeit(sa if w == "a" else sb, globals=g, number=k) / k * 1e9
            (ta if w == "a" else tb).append(t)
    return statistics.median(ta), statistics.median(tb), ta, tb

def spread(s):
    return (max(s) - min(s)) / statistics.median(s)

def K(n):
    return int(max(3, min(6000, 3e7 // max(n, 1))))

SIZES = (3, 8, 12, 16, 18, 20, 22)
# One op per cell. `M` is substituted for np / fnp so both arms evaluate the identical expression.
OPS = [
    ("isnan",    "M.isnan(a)"),
    ("sqrt",     "M.sqrt(a)"),
    ("abs",      "M.abs(a)"),
    ("clip",     "M.clip(a, 0.2, 0.8)"),
    ("sum",      "M.sum(a)"),
    ("mean",     "M.mean(a)"),
    ("max",      "M.max(a)"),
    ("argmax",   "M.argmax(a)"),
    ("add",      "M.add(a, b)"),
    ("multiply", "M.multiply(a, b)"),
    ("divide",   "M.divide(a, b)"),
    ("asarray",  "M.asarray(a)"),
]

out("")
out("%-10s%-6s%12s%13s%13s%9s%13s%8s%9s%9s"
    % ("op", "n", "MiB", "numpy_ns", "fnp_ns", "ratio", "excess_ns", "nullNP", "nullFNP", "incSprd"))
rng = np.random.default_rng(SEED)
for name, expr in OPS:
    for lg in SIZES:
        n = 1 << lg
        # strictly positive and finite: sqrt never warns, divide never hits zero
        a = np.abs(rng.standard_normal(n)) + 0.25
        b = np.abs(rng.standard_normal(n)) + 0.25
        g = {"np": np, "fnp": fnp, "a": a, "b": b}
        sa = expr.replace("M.", "np.")
        sb = expr.replace("M.", "fnp.")
        try:
            wv, gv = np.asarray(eval(sa, g)), np.asarray(eval(sb, g))
            agree = wv.shape == gv.shape and wv.dtype == gv.dtype and wv.tobytes() == gv.tobytes()
        except Exception as e:
            out("%-10s%-6s  SKIPPED (%s)" % (name, "2^%d" % lg, type(e).__name__)); continue
        k = K(n)
        tn, tf, sn, _ = inter(sa, sb, g, k)
        n1, n2, na, nb = inter(sa, sa, g, k)
        c1, c2, _, _ = inter(sb, sb, g, k)
        nn, nf = n2 / n1, c2 / c1
        ok = abs(nn - 1) <= 0.02 and abs(nf - 1) <= 0.02
        isp = max(spread(sn), spread(na), spread(nb))
        flag = "" if ok else "  VOID"
        if not agree:
            flag += "  BYTES DIFFER"
        if ok and abs(tf / tn - 1.0) <= isp:
            flag += "  NOISE>EFFECT"
        out("%-10s%-6s%12.2f%13.1f%13.1f%8.3fx%13.1f%8.3f%9.3f%8.1f%%%s"
            % (name, "2^%d" % lg, n * 8 / (1 << 20), tn, tf, tf / tn, tf - tn, nn, nf,
               100 * isp, flag))
"#;

fn main() -> PyResult<()> {
    pyo3::append_to_inittab!(fnp_python);
    Python::initialize();
    let exe = std::env::current_exe()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyOSError, _>(e.to_string()))?;
    let seed: i64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(555);
    Python::attach(|py| {
        let globals = pyo3::types::PyDict::new(py);
        globals.set_item("EXE_PATH", exe.to_string_lossy().as_ref())?;
        globals.set_item("SEED", seed)?;
        py.run(&CString::new(HARNESS).unwrap(), Some(&globals), None)
    })
}
