//! WHERE IS THE CROSSOVER FOR THE PREDICATE AND UNARY FAMILIES? (`deadlock-audit-zsn2y`)
//!
//! The route-floor sweep put `isnan` at 1.90x / +725 ns at n=8 with a crossover near 2^15, `sqrt`
//! at 2.02x / +818 ns with a crossover near 2^10, and `abs` at 1.71x / +582 ns with one near 2^14.
//! Each of those is a route with no floor, exactly like `np.concatenate` was.
//!
//! BUT A FLOOR CANNOT BE FITTED ON ONE OP, because these ops do not have their own helpers - they
//! SHARE them. `zerocopy_f64_predicate_flat` serves isnan/isinf/isfinite/signbit, and
//! `zerocopy_f64_unary_flat` serves the whole f64 unary family from `abs` to the transcendentals.
//! A gate placed in either is paid by every op that routes through it, so every op has to be
//! measured first - this repo has already destroyed one 395x win by generalising a dtype-shaped
//! fix from too few cells, and the same trap is open here in the op dimension.
//!
//! Sizes are dense between 2^8 and 2^15 because that is where the sweep put the crossings.

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

SIZES = (3, 8, 11, 13, 15, 18)
# family tag, name, expression. The tag is the SHARED HELPER a gate would sit in.
OPS = [
    ("pred", "isnan",      "M.isnan(a)"),
    ("pred", "isinf",      "M.isinf(a)"),
    ("pred", "isfinite",   "M.isfinite(a)"),
    ("pred", "signbit",    "M.signbit(a)"),
    ("uny",  "abs",        "M.abs(a)"),
    ("uny",  "negative",   "M.negative(a)"),
    ("uny",  "square",     "M.square(a)"),
    ("uny",  "floor",      "M.floor(a)"),
    ("uny",  "sign",       "M.sign(a)"),
    ("uny",  "reciprocal", "M.reciprocal(a)"),
    ("uny",  "sqrt",       "M.sqrt(a)"),
    ("uny",  "cbrt",       "M.cbrt(a)"),
    ("uny",  "exp",        "M.exp(a)"),
    ("uny",  "log",        "M.log(a)"),
    ("uny",  "sin",        "M.sin(a)"),
    ("uny",  "tanh",       "M.tanh(a)"),
]

out("")
out("%-5s%-11s%-6s%13s%13s%9s%13s%8s%9s%9s"
    % ("fam", "op", "n", "numpy_ns", "fnp_ns", "ratio", "excess_ns", "nullNP", "nullFNP", "incSprd"))
rng = np.random.default_rng(SEED)
for fam, name, expr in OPS:
    for lg in SIZES:
        n = 1 << lg
        # positive and finite so sqrt/log never warn and exp never overflows
        a = np.abs(rng.standard_normal(n)) + 0.25
        g = {"np": np, "fnp": fnp, "a": a}
        sa = expr.replace("M.", "np.")
        sb = expr.replace("M.", "fnp.")
        try:
            wv, gv = np.asarray(eval(sa, g)), np.asarray(eval(sb, g))
            agree = wv.shape == gv.shape and wv.dtype == gv.dtype and wv.tobytes() == gv.tobytes()
        except Exception as e:
            out("%-5s%-11s%-6s  SKIPPED (%s)" % (fam, name, "2^%d" % lg, type(e).__name__))
            continue
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
        out("%-5s%-11s%-6s%13.1f%13.1f%8.3fx%13.1f%8.3f%9.3f%8.1f%%%s"
            % (fam, name, "2^%d" % lg, tn, tf, tf / tn, tf - tn, nn, nf, 100 * isp, flag))
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
