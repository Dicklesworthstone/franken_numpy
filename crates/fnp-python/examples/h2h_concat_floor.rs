//! WHERE DOES THE NATIVE f64 CONCATENATE ACTUALLY START WINNING? (`deadlock-audit-66w2d`)
//!
//! After five entry-path levers, `np.concatenate([a, b])` on f64 is still a LOSS at both sizes the
//! decomposition probe measures: +783 ns / 1.611x at n=64 and +1144 ns / 1.129x at n=16384. But the
//! wide board has it as a decisive WIN at 2^20 (0.362x on the narrow board's own cell). So the
//! native route has a crossover and no gate marks it - the route engages at EVERY size and loses
//! below that point.
//!
//! Fixing that needs the crossover measured, not guessed: this sweeps two-array f64 concatenate
//! from 2^3 to 2^21 and prints the ratio and the absolute excess at each size, under the same
//! interleaved-with-nulls contract as every other board here. The floor that comes out of it is a
//! FITTED constant, so it also needs the size at which the sign is stable across seeds - hence two
//! seeds in one invocation rather than one.

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
    return int(max(3, min(8000, 4e7 // max(n, 1))))

out("")
out("%-6s%-8s%13s%13s%9s%13s%8s%9s%9s"
    % ("seed", "n", "numpy_ns", "fnp_ns", "ratio", "excess_ns", "nullNP", "nullFNP", "incSprd"))
for seed in (SEED, SEED + 222):
    rng = np.random.default_rng(seed)
    for lg in range(3, 22):
        n = 1 << lg
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        g = {"np": np, "fnp": fnp, "a": a, "b": b}
        sa, sb = "np.concatenate([a, b])", "fnp.concatenate([a, b])"
        wv, gv = np.asarray(eval(sa, g)), np.asarray(eval(sb, g))
        agree = wv.shape == gv.shape and wv.dtype == gv.dtype and wv.tobytes() == gv.tobytes()
        k = K(2 * n)
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
        out("%-6d%-8s%13.1f%13.1f%8.3fx%13.1f%8.3f%9.3f%8.1f%%%s"
            % (seed, "2^%d" % lg, tn, tf, tf / tn, tf - tn, nn, nf, 100 * isp, flag))
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
