//! Head-to-head for INTEGER `searchsorted`, the worst decidable cell on the re-priced board
//! (2.473x), with both query strategies timed in one process.
//!
//! `rch exec --job -- cargo run --release -p fnp-python --example h2h_searchsorted`. Results go to
//! STDERR because rch returns the remote command's stderr and not its stdout.
//!
//! The float paths learned in `8f6cc811` that the right loop depends on the QUERY ORDER; the
//! integer path never got it and still ran a branchy bisection per query through `Cell` reads.
//! `FNP_SEARCHSORTED_MERGE` selects the loop per call so all three strategies are timed on the
//! same arrays in the same interpreter - two rch jobs can land on different workers and cannot be
//! compared. `0` = bisection, unset/`1` = the shipped span-gated merge, `force` = the merge with
//! the span gate bypassed, which is what makes the gate's LOSING side a measurement rather than
//! an assertion.

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

# PARITY FIRST, ONCE PER STRATEGY. This host cannot import the build, so the gate travels with the
# measurement. Integer ordering is total, so there are no documented exceptions here: the gate is
# ZERO divergences. The corpora are the places a merge can go wrong - duplicates in the haystack
# and in the queries (side left/right must differ on ties), queries below/above the whole haystack,
# an empty haystack, and needles that are sorted, reverse-sorted, or sorted-except-one-swap, which
# is the exact boundary of the nondecreasing admission test.
def parity():
    bad = cells = 0
    for impl, flag in (("bisect", "0"), ("merge", "1"), ("force", "force")):
        os.environ["FNP_SEARCHSORTED_MERGE"] = flag
        for dt in ("int8","int16","int32","int64","uint8","uint32","uint64"):
            hi = 100 if dt.endswith("8") else 100000
            for n in (0, 1, 2, 17, 1000, 65536):
                hay = np.sort(rng.integers(0, hi, n).astype(dt))
                for m in (0, 1, 17, 1000):
                    q = rng.integers(0, hi, m).astype(dt)
                    for lbl, qq in (("rand", q), ("sorted", np.sort(q)),
                                    ("rev", np.sort(q)[::-1].copy()),
                                    ("exact", hay[rng.integers(0, max(n,1), m)] if n else q)):
                        for side in ("left", "right"):
                            cells += 1
                            w = np.searchsorted(hay, qq, side=side)
                            g = fnp.searchsorted(hay, qq, side=side)
                            if not np.array_equal(np.asarray(w), np.asarray(g)):
                                bad += 1
                                if bad < 6:
                                    out("  PARITY DIVERGE impl=%s %s n=%d m=%d %s %s"
                                        % (impl, dt, n, m, lbl, side))
                    # sorted EXCEPT one swap: the admission predicate's exact boundary
                    if m > 2:
                        nq = np.sort(q); nq[m//2], nq[m//2-1] = nq[m//2-1], nq[m//2]
                        for side in ("left", "right"):
                            cells += 1
                            if not np.array_equal(np.asarray(np.searchsorted(hay, nq, side=side)),
                                                  np.asarray(fnp.searchsorted(hay, nq, side=side))):
                                bad += 1; out("  PARITY DIVERGE impl=%s almost-sorted n=%d" % (impl, n))
                    del q
                # heavy duplicates on both sides, and out-of-range queries
                if n:
                    d = np.sort(rng.integers(0, 3, n).astype(dt))
                    for side in ("left", "right"):
                        cells += 1
                        qq = np.array([0, 1, 2, 3], dtype=dt)
                        if not np.array_equal(np.asarray(np.searchsorted(d, qq, side=side)),
                                              np.asarray(fnp.searchsorted(d, qq, side=side))):
                            bad += 1; out("  PARITY DIVERGE impl=%s dups n=%d %s" % (impl, n, side))
                    del d
                del hay
    os.environ.pop("FNP_SEARCHSORTED_MERGE", None)
    out("parity: %d cells, %d divergences" % (cells, bad))
    return bad

if parity():
    out("ABORTING: parity failed, a ratio would be meaningless")
    raise SystemExit(1)
out("parity gate: both query strategies agree with NumPy on every cell")

def inter(sa, sb, g, k, rounds=6):
    ta, tb = [], []
    for r in range(rounds):
        for w in (("a","b","b","a") if r % 2 == 0 else ("b","a","a","b")):
            t = timeit.timeit(sa if w == "a" else sb, globals=g, number=k) / k * 1e9
            (ta if w == "a" else tb).append(t)
    return statistics.median(ta), statistics.median(tb)

out("")
out("%-30s%-9s%14s%14s%9s%8s%9s"
    % ("case","impl","numpy_ns","fnp_ns","ratio","nullNP","nullFNP"))
N = 1 << 16
hay = np.sort(rng.integers(0, 1 << 20, N))
CASES = [
    ("self-search (sorted q)", hay),                                   # the survey's cell
    ("sorted q, independent", np.sort(rng.integers(0, 1 << 20, N))),
    ("RANDOM q", rng.integers(0, 1 << 20, N)),                         # merge must decline
    ("reverse-sorted q", np.sort(rng.integers(0, 1 << 20, N))[::-1].copy()),
]
def cell(label, hay_, q, k):
    g = {"np": np, "fnp": fnp, "h": hay_, "q": q}
    for impl, flag in (("bisect", "0"), ("merge", "1"), ("force", "force")):
        os.environ["FNP_SEARCHSORTED_MERGE"] = flag
        tn, tf = inter("np.searchsorted(h,q)", "fnp.searchsorted(h,q)", g, k)
        n1, n2 = inter("np.searchsorted(h,q)", "np.searchsorted(h,q)", g, k)
        c1, c2 = inter("fnp.searchsorted(h,q)", "fnp.searchsorted(h,q)", g, k)
        nn, nf = n2 / n1, c2 / c1
        ok = abs(nn - 1) <= 0.02 and abs(nf - 1) <= 0.02
        out("%-30s%-9s%14.1f%14.1f%8.3fx%8.3f%9.3f%s"
            % (label, impl, tn, tf, tf / tn, nn, nf, "" if ok else "  VOID"))
    os.environ.pop("FNP_SEARCHSORTED_MERGE", None)
    del g

for label, q in CASES:
    cell(label, hay, q, 20)

# THE GATE'S LOSING SIDE. A nondecreasing needle batch is admissible on the QUERY test alone, but
# the walk's cost is span in the HAYSTACK: 64 sorted queries spread over 2^22 elements make the
# pointer cross the whole array where the bisection pays 64*23 probes. `merge` must decline here
# and land on `bisect`; `force` bypasses the span gate and prices what declining is worth.
out("")
big = np.sort(rng.integers(0, 1 << 30, 1 << 22))
for label, q in (("2^22 hay, 64 spread q", np.sort(rng.integers(0, 1 << 30, 64))),
                 ("2^22 hay, 64 clustered q", np.sort(rng.integers(0, 1 << 12, 64)))):
    cell(label, big, q, 50)
del big

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
