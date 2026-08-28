//! Head-to-head: `fnp_python.lexsort` against the LIVE `numpy.lexsort`, both arms in ONE process.
//!
//! Why an example and not a local build: this workspace's artifacts are produced by remote `rch`
//! workers whose glibc (2.43) and CPython (3.14) are both ahead of the developer host, so a
//! retrieved `.so` cannot be imported locally at all. Running the whole comparison *on the worker*
//! removes the transfer entirely - `rch exec --job -- cargo run --release -p fnp-python --example
//! h2h_lexsort` builds and runs there, and the ratio comes back in the command's STDERR. rch
//! returns the remote command's stderr (that is how cargo's own build log reaches you) and does
//! NOT forward its stdout, so the harness prints results to stderr deliberately.
//!
//! No `.so` is involved even on the worker: `append_to_inittab!` registers the extension module
//! into the embedded interpreter, so `import fnp_python` resolves to the code linked into THIS
//! binary. That is what makes the two arms genuinely same-process and same-CPU, and it is why the
//! binary can self-report the SHA-256 of itself as the artifact identity.
//!
//! Subject: `lexsort` on low-cardinality float keys, the campaign's worst standing measured cell.
//! A comparison sort degrades as ties dominate while NumPy's per-key pass does not.

use pyo3::prelude::*;
use std::ffi::CString;

// The `#[pymodule]` macro generates its `__PYO3_NAME` / `__pyo3_init` items alongside the
// function, so `append_to_inittab!` needs THAT item in scope. Naming it bare resolves to the
// CRATE `fnp_python` instead - which is the same identifier - and fails with
// "cannot find value `__PYO3_NAME` in crate `fnp_python`".
use fnp_python::fnp_python;

const HARNESS: &str = r#"
import hashlib, os, statistics, sys, timeit
# RESULTS GO TO STDERR: rch returns the remote command's stderr (that is how cargo's own
# build log surfaces) but does NOT forward its stdout, so a print() here would be lost.
def out(*a):
    print(*a, file=sys.stderr, flush=True)
import numpy as np
import fnp_python as fnp

out("python", sys.version.split()[0], "| numpy", np.__version__)
out("in-process ELF sha256", hashlib.sha256(open(EXE_PATH, "rb").read()).hexdigest())
out("host", os.uname().nodename, "| loadavg", [round(x, 2) for x in os.getloadavg()])
out("fnp module ->", fnp.__name__, "(linked into this binary, no .so)")

def inter(sa, sb, g, k, rounds=6):
    """ABBAABBA so a monotone drift in the host cancels between the arms."""
    ta, tb = [], []
    for r in range(rounds):
        for w in (("a","b","b","a") if r % 2 == 0 else ("b","a","a","b")):
            t = timeit.timeit(sa if w == "a" else sb, globals=g, number=k) / k * 1e9
            (ta if w == "a" else tb).append(t)
    return statistics.median(ta), statistics.median(tb)

N = 1 << 16
REPS = 2
rng = np.random.default_rng(SEED)
out()
out("%-20s%4s%14s%14s%9s%8s%9s" % ("case","rep","numpy_ns","fnp_ns","ratio","nullNP","nullFNP"))
summary = {}
for card in (2, 4, 8, 32):
    good = []
    for rep in range(REPS):
        # FRESH DRAW PER REP: one draw of one corpus has read 1.001x and 3.897x for this shape,
        # so the sign across draws is the evidence, not any single magnitude.
        k1 = rng.integers(0, card, N).astype(np.float64)
        k2 = rng.integers(0, card, N).astype(np.float64)
        g = {"np": np, "fnp": fnp, "k": (k1, k2)}
        tn, tf = inter("np.lexsort(k)", "fnp.lexsort(k)", g, 45)
        n1, n2 = inter("np.lexsort(k)", "np.lexsort(k)", g, 45)
        c1, c2 = inter("fnp.lexsort(k)", "fnp.lexsort(k)", g, 45)
        nn, nf = n2 / n1, c2 / c1
        ok = abs(nn - 1) <= 0.02 and abs(nf - 1) <= 0.02
        if ok:
            good.append(tf / tn)
        out("%-20s%4d%14.1f%14.1f%8.3fx%8.3f%9.3f%s"
              % ("2 x f64 card=%d" % card, rep, tn, tf, tf / tn, nn, nf, "" if ok else "  VOID"))
        del k1, k2, g
    summary[card] = good

# An out-of-family control: if this is off, the whole run is suspect regardless of the nulls.
a = rng.standard_normal(1 << 20)
g = {"np": np, "fnp": fnp, "a": a}
cn, cf = inter("np.sum(a)", "fnp.sum(a)", g, 30)
out("\nout-of-family control np.sum f64 2^20: numpy %.1f fnp %.1f -> %.3fx" % (cn, cf, cf / cn))

out("\nDECIDABLE ratios per cardinality (both A/A nulls within 2%):")
for card, rs in summary.items():
    if rs:
        out("  card=%-5d n=%d  min %.3fx  median %.3fx  max %.3fx"
              % (card, len(rs), min(rs), statistics.median(rs), max(rs)))
    else:
        out("  card=%-5d no decidable cells" % card)
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
