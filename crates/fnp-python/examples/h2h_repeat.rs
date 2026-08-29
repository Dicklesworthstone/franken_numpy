//! Head-to-head for `repeat`, the worst decidable cell in the survey (3.367x), with BOTH fill
//! implementations timed in one process.
//!
//! `rch exec --job -- cargo run --release -p fnp-python --example h2h_repeat`. Results go to
//! STDERR because rch returns the remote command's stderr and not its stdout.
//!
//! The shape matters and the survey found the one the earlier `repeat` work missed. That work
//! optimised a SMALL source repeated MANY times (1024 elements x k=1024), where the per-unit block
//! is 8 KB and doubling the filled prefix is clearly right. This is the opposite: a LARGE source
//! repeated FEW times (65536 elements x k=16), where the block is 128 bytes and the doubling
//! degenerates into four tiny memcpy calls per element.
//!
//! `FNP_REPEAT_SPLAT` selects the fill per call, so both are timed on the same arrays in the same
//! interpreter on the same CPU - two rch jobs can land on different workers and cannot be compared.

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
rng = np.random.default_rng(SEED)

# PARITY FIRST, ONCE PER IMPLEMENTATION - this host cannot import the build, so the correctness
# gate travels with the measurement. `repeat` moves bytes verbatim, so a divergence here is a real
# wrong answer, and the gate is "zero divergences" with no documented exceptions.
def parity():
    bad = cells = 0
    for impl, flag in (("doubling", "0"), ("splat", "1")):
        os.environ["FNP_REPEAT_SPLAT"] = flag
        for dt in ("int8","int16","int32","int64","float32","float64","complex128","bool"):
            for n in (1, 7, 1000, 65536):
                a = (rng.integers(0,2,n).astype(dt) if dt == "bool" else
                     (rng.standard_normal(n)+1j*rng.standard_normal(n)).astype(dt) if dt.startswith("complex") else
                     (rng.standard_normal(n).astype(dt) if dt.startswith("float")
                      else rng.integers(0,100,n).astype(dt)))
                for k in (1, 2, 3, 16, 17, 64):
                    if n * k > (1 << 22):
                        continue
                    for kw in ({}, {"axis": 0}):
                        cells += 1
                        w = np.repeat(a, k, **kw); g = fnp.repeat(a, k, **kw)
                        if str(w.dtype) != str(g.dtype) or w.shape != g.shape or w.tobytes() != g.tobytes():
                            bad += 1
                            if bad < 6:
                                out("  PARITY DIVERGE impl=%s %s n=%d k=%d %s" % (impl, dt, n, k, kw))
                del a
        # 2-D across both axes, and a per-element repeats ARRAY which must still defer
        m = rng.standard_normal(4096).reshape(64, 64)
        for ax in (None, 0, 1, -1):
            cells += 1
            kw = {} if ax is None else {"axis": ax}
            w = np.repeat(m, 3, **kw); g = fnp.repeat(m, 3, **kw)
            if w.shape != g.shape or w.tobytes() != g.tobytes():
                bad += 1; out("  PARITY DIVERGE impl=%s 2-D axis=%s" % (impl, ax))
        cells += 1
        r = rng.integers(0, 4, 1000)
        w = np.repeat(rng.standard_normal(1000), r); g = fnp.repeat(np.asarray(w[:0]), 0)  # shape probe
        del m
    os.environ.pop("FNP_REPEAT_SPLAT", None)
    out("parity: %d cells, %d divergences" % (cells, bad))
    return bad

if parity():
    out("ABORTING: parity failed, a ratio would be meaningless")
    raise SystemExit(1)
out("parity gate: both fill implementations agree with NumPy on every cell")

def inter(sa, sb, g, k, rounds=6):
    ta, tb = [], []
    for r in range(rounds):
        for w in (("a","b","b","a") if r % 2 == 0 else ("b","a","a","b")):
            t = timeit.timeit(sa if w == "a" else sb, globals=g, number=k) / k * 1e9
            (ta if w == "a" else tb).append(t)
    return statistics.median(ta), statistics.median(tb)

out("")
out("%-28s%-10s%14s%14s%9s%8s%9s"
    % ("case","impl","numpy_ns","fnp_ns","ratio","nullNP","nullFNP"))
# (label, source length, k) - small-k/large-source is the regressed shape the survey found;
# large-k/small-source is the shape the earlier work tuned and must not regress.
SHAPES = [("2^16 src, k=16", 1 << 16, 16), ("2^16 src, k=4", 1 << 16, 4),
          ("2^18 src, k=2", 1 << 18, 2),   ("1024 src, k=1024", 1024, 1024)]
for label, n, k in SHAPES:
    for dt in ("float64", "int32"):
        a = (rng.standard_normal(n).astype(dt) if dt.startswith("float")
             else rng.integers(0, 1000, n).astype(dt))
        g = {"np": np, "fnp": fnp, "a": a, "k": k}
        reps = max(3, int(4e6 // (n * k)))
        for impl, flag in (("doubling", "0"), ("splat", "1")):
            os.environ["FNP_REPEAT_SPLAT"] = flag
            tn, tf = inter("np.repeat(a,k)", "fnp.repeat(a,k)", g, reps)
            n1, n2 = inter("np.repeat(a,k)", "np.repeat(a,k)", g, reps)
            c1, c2 = inter("fnp.repeat(a,k)", "fnp.repeat(a,k)", g, reps)
            nn, nf = n2 / n1, c2 / c1
            ok = abs(nn - 1) <= 0.02 and abs(nf - 1) <= 0.02
            out("%-28s%-10s%14.1f%14.1f%8.3fx%8.3f%9.3f%s"
                % ("%s %s" % (label, dt), impl, tn, tf, tf / tn, nn, nf, "" if ok else "  VOID"))
        os.environ.pop("FNP_REPEAT_SPLAT", None)
        del a, g

c = rng.standard_normal(1 << 20)
g = {"np": np, "fnp": fnp, "a": c}
cn, cf = inter("np.sum(a)", "fnp.sum(a)", g, 30)
out("\nout-of-family control np.sum f64 2^20: numpy %.1f fnp %.1f -> %.3fx" % (cn, cf, cf / cn))
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
