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

# ---- 5. THE SIGN TEST, which is what makes this cell DECIDABLE ------------------------------
# `deadlock-audit-qpylx` measured this cell four times and got 1.094x-1.456x, every one VOID:
# the A/A null controls WITHIN-run drift and is blind to run-to-run spread. A sign test is not.
# Each round times both arms ADJACENTLY and keeps only which was faster, so a slow drift that
# moves both arms together cancels instead of adding noise, and the statistic is exact rather
# than an assumption about the distribution: under "no difference" the wins are Binomial(n, 1/2).
#
# The A/A control matters as much as the test. If numpy-vs-numpy also comes out decisive, the
# instrument is measuring order and not the routes, and the result is discarded.
import math

def sign_test(statement_a, statement_b, g, calls, rounds):
    wins, deltas, a_times = 0, [], []
    for round_index in range(rounds):
        # Alternate which arm runs first so a per-round warm-up cost cannot masquerade as a win.
        order = ("a", "b") if round_index % 2 == 0 else ("b", "a")
        timings = {}
        for arm in order:
            statement = statement_a if arm == "a" else statement_b
            timings[arm] = timeit.timeit(statement, globals=g, number=calls) / calls * 1e9
        deltas.append(timings["b"] - timings["a"])
        a_times.append(timings["a"])
        wins += timings["b"] > timings["a"]
    return wins, deltas, statistics.median(a_times)

def two_sided_p(wins, rounds):
    extreme = max(wins, rounds - wins)
    tail = sum(math.comb(rounds, k) for k in range(extreme, rounds + 1))
    return min(1.0, 2.0 * tail / (2 ** rounds))

ROUNDS = 21
out("")
out("%-22s%9s%12s%14s%11s%s" % ("sign test", "b>a", "p", "median_dns", "median_%", "verdict"))
for label, expression in (("sum f64 2^20", "M.sum(f)"),
                          ("mean f64 2^20", "M.mean(f)"),
                          ("std f64 2^20", "M.std(f)")):
    numpy_statement = expression.replace("M.", "np.")
    fnp_statement = expression.replace("M.", "fnp.")
    wins, deltas, numpy_median = sign_test(numpy_statement, fnp_statement, g, 30, ROUNDS)
    control_wins, _, _ = sign_test(numpy_statement, numpy_statement, g, 30, ROUNDS)
    p_value, control_p = two_sided_p(wins, ROUNDS), two_sided_p(control_wins, ROUNDS)
    median_delta = statistics.median(deltas)
    decided = p_value < 0.01 and control_p >= 0.01
    out("%-22s%6d/%-3d%12.5f%13.1f%10.2f%%   %s"
        % (label, wins, ROUNDS, p_value, median_delta,
           100.0 * median_delta / numpy_median,
           ("DECIDED: fnp SLOWER" if wins > ROUNDS / 2 else "DECIDED: fnp FASTER")
           if decided else "not decided"))
    out("    A/A control: %d/%d wins, p=%.5f%s"
        % (control_wins, ROUNDS, control_p,
           "  <-- CONTROL IS DECISIVE, DISCARD THE ROW" if control_p < 0.01 else ""))

# ---- 6. THE SIZE LADDER ACROSS THE PARALLEL FLOOR -------------------------------------------
# `try_zerocopy_float_sum_flat` admits the exact-tree parallel route at
# F64_SUM_PARALLEL_MIN_ELEMENTS = 1_000_000, i.e. 8_000_000 bytes. 2^20 elements is 8_388_608
# bytes, so the route is ADMITTED at the very cell that measures slower. This ladder decides
# whether the loss belongs to that route WITHOUT touching the source: below the floor the
# native path declines and the call falls through, above it the native path runs. If the sign
# is a loss only ABOVE the floor, the route the floor admits is the thing that is losing.
out("")
out("%-30s%9s%12s%13s   %s" % ("size ladder (sum f64)", "b>a", "p", "median_dns", "verdict"))
LADDER = (1 << 18, 1 << 19, 1 << 20, 1_100_000, 1_250_000, 1_500_000, 1_750_000,
          2_000_000, 1 << 21, 1 << 22)
for count in LADDER:
    ladder_globals = {"np": np, "fnp": fnp, "f": rng.standard_normal(count)}
    calls = max(4, min(30, (1 << 22) // count))
    wins, deltas, numpy_median = sign_test("np.sum(f)", "fnp.sum(f)", ladder_globals,
                                           calls, ROUNDS)
    control_wins, _, _ = sign_test("np.sum(f)", "np.sum(f)", ladder_globals, calls, ROUNDS)
    p_value, control_p = two_sided_p(wins, ROUNDS), two_sided_p(control_wins, ROUNDS)
    verdict = "not decided"
    if p_value < 0.01 and control_p >= 0.01:
        verdict = "DECIDED: fnp SLOWER" if wins > ROUNDS / 2 else "DECIDED: fnp FASTER"
    out("%9d elem %10d B %6d/%-3d%12.5f%13.1f   %s%s"
        % (count, count * 8, wins, ROUNDS, p_value, statistics.median(deltas), verdict,
           "" if control_p >= 0.01 else "  (control decisive, discard)"))
out("the parallel exact-tree route is ADMITTED from 1000000 elements / 8000000 bytes upward")
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
