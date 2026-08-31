//! Head-to-head for `np.take`'s flat gather, fitting its parallel threshold.
//!
//! `TAKE_PARALLEL_MIN` is `1 << 21`, set by the 2026-06-25 parallel-take win which demonstrated
//! 5.2x at 8M indices and never swept the threshold. That is the identical provenance, author and
//! day as the three `searchsorted` gates - two of which turned out to be three decades too high
//! and were worth 12-15x once fitted (`deadlock-audit-sfgg3`). This harness asks whether `take`
//! has the same defect, and answers it the way that campaign settled its own: force BOTH sides of
//! the threshold per call with `FNP_TAKE_PARALLEL=ser|par`, sweep the index count, and ship a gate
//! only between a measured LOSING point and a measured WINNING one.
//!
//! It does NOT assume the searchsorted answer transfers. `take` moves whole elements rather than
//! probing, so its serial arm is bandwidth-bound where searchsorted's is latency-bound, and the
//! crossover has no reason to land in the same place.
//!
//! `RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- \
//!    cargo run -j2 --release -p fnp-python --example h2h_take`
//! Results go to STDERR because rch returns the remote command's stderr, not its stdout.

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
rng = np.random.default_rng(SEED)

# The incumbent-win contract's own fields, emitted by the process doing the measuring. The
# incumbent artifact is NumPy's compiled extension, NOT this ELF; the ledger gate rejects the row
# if the two hashes match, because that is provenance substitution.
NP_SO = np._core._multiarray_umath.__file__
out("bench_elf_sha256=%s" % hashlib.sha256(open(EXE_PATH, "rb").read()).hexdigest())
out("incumbent artifact_sha256=%s" % hashlib.sha256(open(NP_SO, "rb").read()).hexdigest())
out("invocation_id=%s-%d-%d" % (os.uname().nodename, os.getpid(), int(__import__("time").time())))

assert np.__name__ == "numpy", np.__name__
assert np.take is not fnp.take
out("dispatch_assert=passed incumbent=numpy.take candidate=fnp.take")

# PARITY FIRST, ONCE PER FORCED SIDE. Chunking cannot change a gather's result, but the parallel
# arm also owns the out-of-range bail (a shared flag checked AFTER the pass), so the corpora below
# include negative indices, exact-boundary indices, and an OOB index that must raise the same
# IndexError numpy raises - the one behaviour a parallel arm can get wrong.
def parity():
    bad = cells = 0
    for impl, flag in (("ser", "ser"), ("par", "par")):
        os.environ["FNP_TAKE_PARALLEL"] = flag
        for dt in ("float64", "int64", "int32", "float32"):
            for n in (1, 17, 4096):
                src = rng.standard_normal(n).astype(dt) if dt.startswith("float") \
                      else rng.integers(-1000, 1000, n).astype(dt)
                for m in (0, 1, 17, 4096):
                    for label, idx in (("random", rng.integers(0, n, m)),
                                       ("negative", rng.integers(-n, 0, m)),
                                       ("boundary", np.array([0, n - 1] * (m // 2), dtype=np.int64)),
                                       ("repeat", np.zeros(m, dtype=np.int64))):
                        cells += 1
                        want = np.take(src, idx)
                        got = fnp.take(src, idx)
                        if not (np.array_equal(want, got) and want.dtype == got.dtype
                                and want.shape == got.shape):
                            bad += 1
                            if bad < 6:
                                out("  PARITY DIVERGE impl=%s %s n=%d m=%d %s" % (impl, dt, n, m, label))
                # OOB must raise exactly as numpy does, from BOTH arms
                for bad_idx in (n, -n - 1):
                    cells += 1
                    try:
                        np.take(src, np.array([bad_idx])); w = "no-raise"
                    except IndexError: w = "IndexError"
                    try:
                        fnp.take(src, np.array([bad_idx])); g = "no-raise"
                    except IndexError: g = "IndexError"
                    if w != g:
                        bad += 1; out("  OOB DIVERGE impl=%s n=%d idx=%d numpy=%s fnp=%s"
                                      % (impl, n, bad_idx, w, g))
    os.environ.pop("FNP_TAKE_PARALLEL", None)
    out("take parity: %d cells, %d divergences" % (cells, bad))
    return bad

# The allocation switch resolves its PRESENCE once, on the first fnp.take call - which happens
# inside parity() - so it must be set BEFORE that or the kwargs arm can never engage. This is the
# same first-call-wins hazard the searchsorted switches have; setting a non-"kwargs" value here
# arms the switch while leaving the SHIPPED positional spelling in force.
os.environ["FNP_TAKE_ALLOC"] = "positional"

if parity():
    out("ABORTING: parity failed, a ratio would be meaningless")
    raise SystemExit(1)

# ENGAGEMENT PROBE, AND IT IS NOT OPTIONAL. A first run of this harness produced ser and par
# timings identical to three decimals at every size - which is not "the gate does not matter", it
# is the signature of a route that never engages, so neither forced side reaches the gate at all.
# Spy on numpy.take: if fnp.take calls it, we are timing a DELEGATION and no threshold of ours is
# on the path. This is the "green run that measured nothing" trap wearing a new hat.
def engagement(src, idx):
    real = np.take
    calls = []
    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)
    np.take = spy
    try:
        fnp.take(src, idx)
    finally:
        np.take = real
    return len(calls)

_src = rng.standard_normal(1 << 22)
_idx = rng.integers(0, 1 << 22, 1 << 16)
os.environ["FNP_TAKE_DEBUG"] = "1"
_n = engagement(_src, _idx)
os.environ.pop("FNP_TAKE_DEBUG", None)
out("engagement: fnp.take made %d numpy.take call(s) on the timed shape (0 = native route)" % _n)
if _n:
    out("ABORTING: fnp.take DELEGATES on this shape, so the parallel gate is not on the measured")
    out("path and any ser/par comparison would be timing numpy against numpy.")
    raise SystemExit(1)
del _src, _idx

def inter(sa, sb, g, k, rounds=4):
    ta, tb = [], []
    for r in range(rounds):
        for w in ("a", "b", "b", "a", "a", "b", "b", "a"):
            t = timeit.timeit(sa if w == "a" else sb, globals=g, number=k) / k * 1e9
            (ta if w == "a" else tb).append(t)
    return statistics.median(ta), statistics.median(tb)

out("")
out("%-28s%-7s%14s%14s%9s%8s%9s" % ("case","impl","numpy_ns","fnp_ns","ratio","nullNP","nullFNP"))

def cell(label, src, idx, k):
    g = {"np": np, "fnp": fnp, "a": src, "i": idx}
    # `kwargs` restores the FORMER output-allocation spelling (a per-call PyDict) against the
    # shipped positional one; `ser`/`par` force the two sides of the parallel threshold. All are
    # selected per call so every comparison is inside THIS process.
    for impl, flag in (("ser", "ser"), ("par", "par"), ("kwargs", "ser")):
        if impl == "kwargs":
            os.environ["FNP_TAKE_ALLOC"] = "kwargs"
        os.environ["FNP_TAKE_PARALLEL"] = flag
        tn, tf = inter("np.take(a,i)", "fnp.take(a,i)", g, k)
        n1, n2 = inter("np.take(a,i)", "np.take(a,i)", g, k)
        c1, c2 = inter("fnp.take(a,i)", "fnp.take(a,i)", g, k)
        nn, nf = n2 / n1, c2 / c1
        ok = abs(nn - 1) <= 0.02 and abs(nf - 1) <= 0.02
        out("%-28s%-7s%14.1f%14.1f%8.3fx%8.3f%9.3f%s"
            % (label, impl, tn, tf, tf / tn, nn, nf, "" if ok else "  VOID"))
        os.environ["FNP_TAKE_ALLOC"] = "positional"
    os.environ.pop("FNP_TAKE_PARALLEL", None)
    del g

# A 2^22 f64 source = 32 MB, far past any cache, so the gather is genuinely DRAM-latency-bound -
# the regime the parallel arm exists for. Sweep the index count across the gate.
src = rng.standard_normal(1 << 22)
for mq in (1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20):
    cell("f64 gather m=2^%d" % (mq.bit_length() - 1), src,
         rng.integers(0, 1 << 22, mq), max(2, 4000 // max(1, mq >> 8)))
del src

# A SMALL source is the opposite regime: it stays cache-resident, so the serial gather is not
# latency-bound and parallelism has much less to recover. A gate fitted only on the large source
# would be wrong here, which is why both are measured.
small = rng.standard_normal(1 << 12)
for mq in (1 << 14, 1 << 18):
    cell("f64 small-src m=2^%d" % (mq.bit_length() - 1), small,
         rng.integers(0, 1 << 12, mq), max(2, 4000 // max(1, mq >> 8)))
del small

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
