//! Head-to-head for INTEGER `searchsorted`, the worst decidable cell on the re-priced board
//! (2.473x), with both query strategies timed in one process.
//!
//! `rch exec --job -- cargo run --release -p fnp-python --example h2h_searchsorted`. Results go to
//! STDERR because rch returns the remote command's stderr and not its stdout.
//!
//! The float paths learned in `8f6cc811` that the right loop depends on the QUERY ORDER; the
//! integer path never got it and still ran a branchy bisection per query through `Cell` reads.
//! `FNP_SEARCHSORTED_MERGE` selects the loop per call so all four strategies are timed on the
//! same arrays in the same interpreter - two rch jobs can land on different workers and cannot be
//! compared. `0` = bisection, unset/`1` = the shipped span-gated merge with previous-hit gallop
//! fallback, `gallop` = force that fallback, and `force` = the merge with the span gate bypassed,
//! which is what makes the gate's LOSING side a measurement rather than an assertion.

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

# THE incumbent-win CONTRACT'S OWN FIELDS, emitted by the process that does the measuring.
# `bench_elf_sha256` is spelled the way the ledger gate greps for it. The incumbent artifact is
# NumPy's compiled extension - the thing that actually performs np.searchsorted - and NOT this
# ELF; the gate rejects the row if the two hashes are equal, because that is provenance
# substitution. `invocation_id` ties both arms to this one process.
NP_SO = np._core._multiarray_umath.__file__
NP_SHA = hashlib.sha256(open(NP_SO, "rb").read()).hexdigest()
INVOCATION = "%s-%d-%d" % (os.uname().nodename, os.getpid(), int(__import__("time").time()))
out("bench_elf_sha256=%s" % hashlib.sha256(open(EXE_PATH, "rb").read()).hexdigest())
out("incumbent artifact %s" % NP_SO)
out("incumbent artifact_sha256=%s" % NP_SHA)
out("invocation_id=%s" % INVOCATION)

# DISPATCH ASSERT, at runtime, inside the measured process: the incumbent arm must be genuine
# NumPy and must not be one of ours. This is trap 1 of the six, and it has produced a false win
# in this fleet before.
assert np.__name__ == "numpy", np.__name__
assert "fnp" not in type(np.searchsorted).__module__.lower()
assert np.searchsorted is not fnp.searchsorted
out("dispatch_assert=passed incumbent=numpy.searchsorted candidate=fnp.searchsorted")

# PARITY FIRST, ONCE PER STRATEGY. This host cannot import the build, so the gate travels with the
# measurement. Integer ordering is total, so there are no documented exceptions here: the gate is
# ZERO divergences. The corpora are the places a merge can go wrong - duplicates in the haystack
# and in the queries (side left/right must differ on ties), queries below/above the whole haystack,
# an empty haystack, and needles that are sorted, reverse-sorted, or sorted-except-one-swap, which
# is the exact boundary of the nondecreasing admission test.
def parity():
    bad = cells = 0
    for impl, flag in (("perquery", "perquery"), ("bisect", "0"), ("merge", "1"),
                       ("gallop", "gallop"), ("force", "force"),
                       ("ser", "ser"), ("par", "par")):
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

def parity_f32():
    # The f32 gate changes only chunking, but its raw helper owns NaN-last and signed-zero
    # ordering. Exercise both forced sides on ordinary, monotone, and NaN-containing arrays.
    bad = cells = 0
    for impl, flag in (("ser", "ser"), ("par", "par")):
        os.environ["FNP_SEARCHSORTED_MERGE"] = flag
        for n in (0, 1, 17, 65536):
            hay = np.sort(rng.standard_normal(n).astype(np.float32))
            for m in (0, 1, 17, 2048):
                q = rng.standard_normal(m).astype(np.float32)
                variants = (("random", q), ("sorted", np.sort(q)))
                if m:
                    variants += (("nan", np.concatenate((
                        q[:-1], np.array([np.nan], dtype=np.float32)))),)
                for label, qq in variants:
                    for side in ("left", "right"):
                        cells += 1
                        want = np.searchsorted(hay, qq, side=side)
                        got = fnp.searchsorted(hay, qq, side=side)
                        if not np.array_equal(want, got):
                            bad += 1
                            if bad < 6:
                                out("  F32 PARITY DIVERGE impl=%s n=%d m=%d %s %s"
                                    % (impl, n, m, label, side))
                del q
            del hay
    os.environ.pop("FNP_SEARCHSORTED_MERGE", None)
    out("f32 parity: %d cells, %d divergences" % (cells, bad))
    return bad

if parity() or parity_f32():
    out("ABORTING: parity failed, a ratio would be meaningless")
    raise SystemExit(1)
out("parity gate: both query strategies agree with NumPy on every cell")

def inter(sa, sb, g, k, rounds=4):
    ta, tb = [], []
    for r in range(rounds):
        # Keep an explicit ABBAABBA schedule.  It has both warm/cold directions,
        # is stable under a rerun, and makes the two A/A controls directly
        # comparable with the NumPy-vs-fnp row.
        for w in ("a", "b", "b", "a", "a", "b", "b", "a"):
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
    # `perquery` is the FORMER fallback loop - one full bisection per key - kept selectable so the
    # batched search can be A/B'd against it in THIS process rather than across two runs.
    # `perquery` is the FORMER fallback loop - one full bisection per key - kept selectable so the
    # batched search can be A/B'd against it in THIS process rather than across two runs.
    # `ser`/`par` force the two sides of the rayon threshold for the same reason.
    for impl, flag in (("perquery", "perquery"), ("bisect", "0"), ("merge", "1"),
                       ("gallop", "gallop"), ("force", "force"),
                       ("ser", "ser"), ("par", "par")):
        os.environ["FNP_SEARCHSORTED_MERGE"] = flag
        tn, tf = inter("np.searchsorted(h,q)", "fnp.searchsorted(h,q)", g, k)
        n1, n2 = inter("np.searchsorted(h,q)", "np.searchsorted(h,q)", g, k)
        c1, c2 = inter("fnp.searchsorted(h,q)", "fnp.searchsorted(h,q)", g, k)
        nn, nf = n2 / n1, c2 / c1
        ok = abs(nn - 1) <= 0.02 and abs(nf - 1) <= 0.02
        out("%-30s%-9s%14.1f%14.1f%8.3fx%8.3f%9.3f%s"
            % (label, impl, tn, tf, tf / tn, nn, nf, "" if ok else "  VOID"))
        RESULTS[(label, impl)] = (tn, tf, tf / tn, nn, nf)
    os.environ.pop("FNP_SEARCHSORTED_MERGE", None)
    del g

RESULTS = {}
for label, q in CASES:
    cell(label, hay, q, 20)

# THE SHARED TIMED COMPONENT, QUANTIFIED RATHER THAN DENIED. Our arm allocates its output with
# `numpy.empty`, so a piece of the INCUMBENT's own code is inside the candidate's timing. That is
# never to be declared as `none`. Time it directly and report it as a share of the candidate arm,
# so the ledger row can disclose the number instead of asserting isolation it does not have.
_g = {"np": np, "N": N}
_e1, _e2 = inter("np.empty(N, 'intp')", "np.empty(N, 'intp')", _g, 20)
_empty_ns = (_e1 + _e2) / 2
for _impl in ("merge", "gallop"):
    _tn, _tf, _r, _, _ = RESULTS[("self-search (sorted q)", _impl)]
    out("shared_timed_component numpy.empty=%.1f ns = %.2f%% of candidate %s arm (%.1f ns)"
        % (_empty_ns, 100.0 * _empty_ns / _tf, _impl, _tf))
del _g

# Native f16 is intentionally a separate cell: it has a finite-order cumulative
# table rather than the integer merge, and its admission gate needs enough
# needles to amortise the 65,536-key table.  Include signed zeros, infinities,
# duplicates, and exact haystack values so both `side` conventions are exercised
# before the timing row below.
f16_hay = np.sort(np.concatenate((
    np.linspace(-500.0, 500.0, N - 8, dtype=np.float16),
    np.array([-np.inf, -0.0, 0.0, 0.0, 1.0, 1.0, np.inf, np.inf], dtype=np.float16),
)))
f16_q = np.sort(rng.choice(f16_hay, size=N, replace=True)).astype(np.float16)
for side in ("left", "right"):
    want = np.searchsorted(f16_hay, f16_q, side=side)
    got = fnp.searchsorted(f16_hay, f16_q, side=side)
    if not np.array_equal(want, got):
        out("ABORTING: native f16 parity divergence", side)
        raise SystemExit(1)
g = {"np": np, "fnp": fnp, "h": f16_hay, "q": f16_q}
tn, tf = inter("np.searchsorted(h,q)", "fnp.searchsorted(h,q)", g, 20)
n1, n2 = inter("np.searchsorted(h,q)", "np.searchsorted(h,q)", g, 20)
c1, c2 = inter("fnp.searchsorted(h,q)", "fnp.searchsorted(h,q)", g, 20)
out("%-30s%-9s%14.1f%14.1f%8.3fx%8.3f%9.3f%s"
    % ("f16 finite table", "native", tn, tf, tf / tn, n2 / n1, c2 / c1,
       "" if abs(n2/n1-1) <= .02 and abs(c2/c1-1) <= .02 else "  VOID"))
del g, f16_hay, f16_q

# WHERE DOES THE PARALLEL THRESHOLD BELONG? Timing one size tells you that size is above the
# crossover, not where the crossover is. Sweep the query count with the two sides forced, on random
# needles (the only shape the parallel arm now accepts), and read the gate off the data.
out("")
out("%-30s%-9s%14s%14s%9s%8s%9s"
    % ("parallel-threshold sweep","impl","numpy_ns","fnp_ns","ratio","nullNP","nullFNP"))
sweep_hay = np.sort(rng.integers(0, 1 << 20, 1 << 16))
for mq in (1 << 8, 1 << 10, 1 << 11, 1 << 12, 1 << 14, 1 << 16):
    cell("random m=2^%d" % (mq.bit_length() - 1), sweep_hay,
         rng.integers(0, 1 << 20, mq), max(2, 2000 // max(1, mq >> 8)))
del sweep_hay

# The SAME sweep on the f64 route, whose threshold is a separate constant still at its inherited
# 1<<21. Same element width and a near-identical search cost, so the integer crossover is a
# plausible prior - which is exactly why it has to be measured rather than assumed.
f64_hay = np.sort(rng.standard_normal(1 << 16))
for mq in (1 << 10, 1 << 11, 1 << 12, 1 << 14, 1 << 16):
    cell("f64 random m=2^%d" % (mq.bit_length() - 1), f64_hay,
         rng.standard_normal(mq), max(2, 2000 // max(1, mq >> 8)))
# f64 SORTED needles: the band below the f64 merge's own m>=1<<19 gate, where a widened parallel
# arm must not cost anything relative to the serial guess-based search it replaces.
cell("f64 sorted m=2^16", f64_hay, np.sort(rng.standard_normal(1 << 16)), 20)
del f64_hay

# The f32 route, whose threshold is a THIRD separate constant still at the inherited 1<<21. f32
# halves the element width, so twice as much haystack fits in each cache level and the serial arm
# is less latency-bound - the very quantity the crossover depends on. The f64 result is a prior
# here, not an answer.
f32_hay = np.sort(rng.standard_normal(1 << 16).astype(np.float32))
for mq in (1 << 10, 1 << 11, 1 << 12, 1 << 14, 1 << 16):
    cell("f32 random m=2^%d" % (mq.bit_length() - 1), f32_hay,
         rng.standard_normal(mq).astype(np.float32), max(2, 2000 // max(1, mq >> 8)))
cell("f32 sorted m=2^16", f32_hay,
     np.sort(rng.standard_normal(1 << 16).astype(np.float32)), 20)
del f32_hay

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
