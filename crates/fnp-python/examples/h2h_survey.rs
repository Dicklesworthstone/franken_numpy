//! Head-to-head SURVEY: many ops against the LIVE numpy, both arms in one process on one worker.
//!
//! Why this exists: every loss this campaign is chasing was ranked with a local harness, and the
//! developer host can no longer import a build at all (worker glibc 2.43 > local 2.42, worker
//! CPython 3.14 vs a numpy installed only for local 3.13). The rankings that survive that change
//! of method have to be re-established, not assumed.
//!
//! Run with `rch exec --job -- cargo run --release -p fnp-python --example h2h_survey`. Results go
//! to STDERR because rch returns the remote command's stderr and not its stdout.
//!
//! Same contract as `h2h_lexsort`: `append_to_inittab!` puts the module INTO this binary so no
//! `.so` is involved, the incumbent is the live numpy in the same interpreter, each cell is
//! interleaved ABBAABBA with a dual A/A null, and the binary hashes itself so the artifact
//! identity cannot drift from what was timed.

use pyo3::prelude::*;
use std::ffi::CString;

// `append_to_inittab!` needs the item the `#[pymodule]` macro generates, which shares its name
// with the crate; without this `use` the bare identifier resolves to the crate and fails.
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

def inter(sa, sb, g, k, rounds=4):
    ta, tb = [], []
    for r in range(rounds):
        for w in (("a","b","b","a") if r % 2 == 0 else ("b","a","a","b")):
            t = timeit.timeit(sa if w == "a" else sb, globals=g, number=k) / k * 1e9
            (ta if w == "a" else tb).append(t)
    return statistics.median(ta), statistics.median(tb)

N  = 1 << 20
NS = 1 << 16          # sorts and set ops: smaller, they are O(n log n)
f  = rng.standard_normal(N)
f2 = rng.standard_normal(N)
i64 = rng.integers(0, 1 << 20, N)
fs = rng.standard_normal(NS)
isr = np.sort(rng.integers(0, 1 << 20, NS))
low = rng.integers(0, 8, NS)
b  = rng.integers(0, 2, N).astype(bool)

# (label, expression, globals) - one expression, evaluated identically against np and fnp.
CASES = [
    ("sum f64 2^20",         "M.sum(f)",                    {"f": f}),
    ("mean f64 2^20",        "M.mean(f)",                   {"f": f}),
    ("std f64 2^20",         "M.std(f)",                    {"f": f}),
    ("min f64 2^20",         "M.min(f)",                    {"f": f}),
    ("argmin f64 2^20",      "M.argmin(f)",                 {"f": f}),
    ("cumsum f64 2^20",      "M.cumsum(f)",                 {"f": f}),
    ("add f64 2^20",         "M.add(f, f2)",                {"f": f, "f2": f2}),
    ("multiply f64 2^20",    "M.multiply(f, f2)",           {"f": f, "f2": f2}),
    ("divide f64 2^20",      "M.divide(f, f2)",             {"f": f, "f2": f2}),
    ("sqrt f64 2^20",        "M.sqrt(M.abs(f))",            {"f": f}),
    ("isnan f64 2^20",       "M.isnan(f)",                  {"f": f}),
    ("where f64 2^20",       "M.where(b, f, f2)",           {"b": b, "f": f, "f2": f2}),
    ("count_nonzero 2^20",   "M.count_nonzero(b)",          {"b": b}),
    ("dot f64 2^20",         "M.dot(f, f2)",                {"f": f, "f2": f2}),
    ("sort f64 2^16",        "M.sort(fs)",                  {"fs": fs}),
    ("argsort f64 2^16",     "M.argsort(fs)",               {"fs": fs}),
    ("unique i64 2^16",      "M.unique(low)",               {"low": low}),
    ("searchsorted 2^16",    "M.searchsorted(isr, isr)",    {"isr": isr}),
    ("take f64 2^20",        "M.take(f, i)",                {"f": f, "i": i64}),
    ("repeat f64 2^16",      "M.repeat(fs, 16)",            {"fs": fs}),
    ("concatenate f64 2^20", "M.concatenate([f, f2])",      {"f": f, "f2": f2}),
    ("cumprod f64 2^16",     "M.cumprod(fs)",               {"fs": fs}),
    ("diff f64 2^20",        "M.diff(f)",                   {"f": f}),
    ("clip f64 2^20",        "M.clip(f, -1.0, 1.0)",        {"f": f}),
]

out("")
out("%-24s%14s%14s%9s%8s%9s" % ("case","numpy_ns","fnp_ns","ratio","nullNP","nullFNP"))
rows = []
for label, expr, base in CASES:
    gn = dict(base); gn["M"] = np
    gf = dict(base); gf["M"] = fnp
    g  = dict(base); g["np"] = np; g["fnp"] = fnp
    sa = expr.replace("M.", "np.")
    sb = expr.replace("M.", "fnp.")
    try:
        # correctness first: a ratio for a wrong answer is worthless
        wv, gv = np.asarray(eval(sa, gn | {"np": np})), np.asarray(eval(sb, gf | {"fnp": fnp}))
        agree = wv.shape == gv.shape and np.allclose(wv, gv, rtol=1e-12, atol=0, equal_nan=True)
    except Exception as e:
        out("%-24s  SKIPPED (%s)" % (label, type(e).__name__)); continue
    k = max(3, int(2e7 // N))
    tn, tf = inter(sa, sb, g, k)
    n1, n2 = inter(sa, sa, g, k)
    c1, c2 = inter(sb, sb, g, k)
    nn, nf = n2 / n1, c2 / c1
    ok = abs(nn - 1) <= 0.02 and abs(nf - 1) <= 0.02
    flag = "" if ok else "  VOID"
    if not agree:
        flag += "  VALUES DIFFER"
    if ok and agree:
        rows.append((tf / tn, label, tn, tf))
    out("%-24s%14.1f%14.1f%8.3fx%8.3f%9.3f%s" % (label, tn, tf, tf / tn, nn, nf, flag))

out("")
out("DECIDABLE cells ranked by ratio (worst first):")
for r, label, tn, tf in sorted(rows, reverse=True):
    marker = "  <-- LOSS" if r > 1.10 else ("  win" if r < 0.90 else "")
    out("  %8.3fx  %-24s numpy %12.1f  fnp %12.1f%s" % (r, label, tn, tf, marker))
out("%d of %d cells decidable" % (len(rows), len(CASES)))
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
