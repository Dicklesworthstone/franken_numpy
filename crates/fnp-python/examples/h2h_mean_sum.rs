//! CLOSURE probe for the last standing loss on the survey board: `mean f64 2^20` at 1.461x.
//!
//! This deliberately attempts NO lever. It answers the questions that decide which lever is even
//! admissible, because the board contains an asymmetry that no guess should be built on:
//!
//!   mean f64 2^20   numpy  233507 ns   fnp  341187 ns   1.461x   LOSS
//!   sum  f64 2^20   numpy  236798 ns   fnp  332042 ns   1.402x   LOSS
//!   std  f64 2^20   numpy 1764468 ns   fnp  726650 ns   0.412x   WIN
//!
//! Our `std` beats NumPy 2.4x while computing a mean internally, yet our `mean` loses. So the loss
//! is very unlikely to be "mean" at all.
//!
//! FOUR QUESTIONS, in the order that makes later ones matter:
//!   1. ENGAGEMENT - do fnp.mean / fnp.sum delegate to NumPy? If they do, the ratio is dispatch
//!      overhead and no kernel lever exists. Spied, not assumed.
//!   2. DECOMPOSITION - is fnp.mean just fnp.sum plus a divide? If mean ~= sum, "mean" is the
//!      wrong name for the defect and every mean-specific lever is misdirected.
//!   3. BIT-EXACTNESS - is our sum bit-identical to NumPy's? NumPy uses pairwise summation; any
//!      reordering (including parallel blocking) changes the result. This is the constraint that
//!      decides whether a parallel or re-blocked reduction is admissible AT ALL, and this repo has
//!      a standing rule that blocking is never bit-exact. Establish it BEFORE proposing one.
//!   4. HEADROOM - what does a plain serial pass over the same buffer cost? That bounds what any
//!      kernel lever could win, and separates "our kernel is slow" from "we are at the bandwidth".
//!
//! `RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- \
//!    cargo run -j2 --release -p fnp-python --example h2h_mean_sum`

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
out("host", os.uname().nodename, "| loadavg", [round(x, 2) for x in os.getloadavg()])
out("nproc", os.cpu_count())
out("bench_elf_sha256=%s" % hashlib.sha256(open(EXE_PATH, "rb").read()).hexdigest())
NP_SO = np._core._multiarray_umath.__file__
out("incumbent artifact_sha256=%s" % hashlib.sha256(open(NP_SO, "rb").read()).hexdigest())
out("invocation_id=%s-%d-%d" % (os.uname().nodename, os.getpid(), int(__import__("time").time())))
assert np.__name__ == "numpy"
rng = np.random.default_rng(SEED)

def inter(sa, sb, g, k, rounds=4):
    ta, tb = [], []
    for r in range(rounds):
        for w in ("a","b","b","a","a","b","b","a"):
            t = timeit.timeit(sa if w == "a" else sb, globals=g, number=k) / k * 1e9
            (ta if w == "a" else tb).append(t)
    return statistics.median(ta), statistics.median(tb)

# ---- 1. ENGAGEMENT -------------------------------------------------------------------------
# A delegating route makes every later number a numpy-vs-numpy comparison. Spy on the incumbent
# entry points and count calls made while our function runs.
def spy(name, call):
    real = getattr(np, name)
    hits = []
    def probe(*a, **k):
        hits.append(1)
        return real(*a, **k)
    setattr(np, name, probe)
    try:
        call()
    finally:
        setattr(np, name, real)
    return len(hits)

f = rng.standard_normal(1 << 20)
out("")
for name, call in (("mean", lambda: fnp.mean(f)), ("sum", lambda: fnp.sum(f))):
    n_same = spy(name, call)
    n_add  = spy("add", call)
    out("engagement fnp.%-5s -> numpy.%-5s calls=%d   numpy.add calls=%d" % (name, name, n_same, n_add))

# ---- 2. DECOMPOSITION ----------------------------------------------------------------------
# If mean is just sum + a divide, the defect is the SUM and "mean" is a misnomer for it.
g = {"np": np, "fnp": fnp, "f": f}
out("")
out("%-34s%14s%14s%9s%8s%9s" % ("cell","numpy_ns","fnp_ns","ratio","nullNP","nullFNP"))
def cell(label, expr, k=30):
    tn, tf = inter(expr.replace("M.", "np."), expr.replace("M.", "fnp."), g, k)
    n1, n2 = inter(expr.replace("M.", "np."), expr.replace("M.", "np."), g, k)
    c1, c2 = inter(expr.replace("M.", "fnp."), expr.replace("M.", "fnp."), g, k)
    nn, nf = n2/n1, c2/c1
    ok = abs(nn-1) <= 0.02 and abs(nf-1) <= 0.02
    out("%-34s%14.1f%14.1f%8.3fx%8.3f%9.3f%s"
        % (label, tn, tf, tf/tn, nn, nf, "" if ok else "  VOID"))
    return tn, tf

cell("sum f64 2^20",  "M.sum(f)")
cell("mean f64 2^20", "M.mean(f)")
cell("std f64 2^20",  "M.std(f)")

# ---- 3. BIT-EXACTNESS, which gates every reordering lever ----------------------------------
# NumPy sums pairwise. If our result already differs, we are NOT bit-exact today and a reordering
# lever is not barred by exactness. If it matches, any re-blocked or parallel sum must reproduce
# NumPy's exact tree or it ships a silent numeric change.
out("")
mism = 0
for n in (1 << 10, 1 << 16, 1 << 20, (1 << 20) + 7):
    x = rng.standard_normal(n)
    for label, ours, theirs in (("sum", fnp.sum(x), np.sum(x)),
                                ("mean", fnp.mean(x), np.mean(x))):
        same = np.asarray(ours).tobytes() == np.asarray(theirs).tobytes()
        if not same:
            mism += 1
            out("  BITDIFF %-5s n=%-9d ours=%.20g theirs=%.20g ulp_delta=%d"
                % (label, n, ours, theirs,
                   abs(np.asarray(ours).view("int64") - np.asarray(theirs).view("int64"))))
out("bit-exactness vs numpy: %d of 8 cells differ (0 = we are bit-exact today)" % mism)

# ---- 4. HEADROOM ---------------------------------------------------------------------------
# What a single pass over the same bytes costs, measured through numpy primitives that do exactly
# one pass. This bounds any kernel lever: a sum cannot beat the cost of reading the data once.
out("")
gh = {"np": np, "f": f, "e": np.empty_like(f)}
one_pass = statistics.median([timeit.timeit("np.copyto(e, f)", globals=gh, number=30)/30*1e9
                              for _ in range(5)])
out("one full pass over 8 MiB (np.copyto, read+write) = %.1f ns" % one_pass)
sn, sf = cell("sum f64 2^20 (repeat for headroom)", "M.sum(f)")
out("numpy.sum = %.2f x one copy;  fnp.sum = %.2f x one copy;  fnp excess = %.1f ns"
    % (sn / one_pass, sf / one_pass, sf - sn))
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
