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

# PARITY FIRST, IN THIS SAME PROCESS. The developer host cannot import this build at all, so the
# correctness gate has to travel with the measurement instead of being skipped. A ratio for a
# route that answers wrongly is worth nothing, so a failure here aborts before any timing.
def parity():
    rp = np.random.default_rng(4242)
    bad = 0
    cells = 0
    seen = set()
    def chk(lbl, keys):
        nonlocal bad, cells
        cells += 1
        w = np.asarray(np.lexsort(keys)); g = np.asarray(fnp.lexsort(keys))
        if str(w.dtype) != str(g.dtype) or w.shape != g.shape or not np.array_equal(w, g):
            bad += 1
            seen.add(lbl)
    # straddle the counting path's own gates: n > 4096 to admit it, and a distinct-record count
    # either side of its 1024 cap (c*c records for two keys of cardinality c).
    for n in (4095, 4096, 4097, 8192, 65536):
        for card in (1, 2, 3, 8, 100):
            for nk in (2, 3):
                chk("n=%d card=%d nk=%d" % (n, card, nk),
                    tuple(rp.integers(0, card, n).astype(np.float64) for _ in range(nk)))
        for c in (31, 32, 33):
            chk("cap-straddle n=%d c=%d (%d records)" % (n, c, c * c),
                (rp.integers(0, c, n).astype(np.float64), rp.integers(0, c, n).astype(np.float64)))
        # record widths across the 16-byte pack limit: 12 bytes, 8 bytes, 24 bytes (must fall back)
        chk("i4+f8 n=%d" % n, (rp.integers(0, 4, n).astype(np.int32),
                               rp.integers(0, 4, n).astype(np.float64)))
        chk("f4+f4 n=%d" % n, (rp.integers(0, 4, n).astype(np.float32),
                               rp.integers(0, 4, n).astype(np.float32)))
        chk("3xf8 n=%d" % n, tuple(rp.integers(0, 4, n).astype(np.float64) for _ in range(3)))
        # ties that must stay STABLE: signed zero compares equal, and an all-equal batch must come
        # back as the identity permutation - a direct stability assertion, not a numpy comparison.
        z = np.where(rp.integers(0, 2, n) == 0, 0.0, -0.0)
        chk("signed-zero n=%d" % n, (z, rp.integers(0, 3, n).astype(np.float64)))
        ident = (np.zeros(n), np.zeros(n))
        chk("all-equal n=%d" % n, ident)
        cells += 1
        if not np.array_equal(np.asarray(fnp.lexsort(ident)), np.arange(n)):
            bad += 1
            out("  STABILITY FAIL all-equal n=%d: not the identity permutation" % n)
        # NaN and infinities inside a low-cardinality batch
        v = rp.choice(np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1.0]), n)
        chk("specials n=%d" % n, (v, rp.integers(0, 3, n).astype(np.float64)))
    # THE GATE PINS THE KNOWN-BROKEN SET EXACTLY - it is a regression gate, not a relaxed one.
    # `lexsort` has a DOCUMENTED pre-existing defect on multi-key float: -NaN sorts first instead
    # of last, and a 0.0/-0.0 tie comes back in the opposite order from the input, so the sort is
    # not stable across a signed-zero tie. Those two corpora therefore diverge at every size, and
    # they diverge identically at HEAD - verified by stashing the change under test and re-running
    # this harness, which produced the same 10 labels and the same first-differing indices.
    #
    # Asserting SET EQUALITY rather than "no divergences" keeps the gate strict in BOTH directions:
    # a new divergence adds a label and fails, and silently fixing one removes a label and also
    # fails, so neither can pass unnoticed. It cannot hide anything the way a count could.
    expected = set()
    for n in (4095, 4096, 4097, 8192, 65536):
        expected.add("signed-zero n=%d" % n)
        expected.add("specials n=%d" % n)
    out("parity: %d cells, %d divergences" % (cells, bad))
    unexpected = sorted(seen - expected)
    repaired = sorted(expected - seen)
    if unexpected:
        out("  NEW divergences (not in the documented pre-existing set):", unexpected)
    if repaired:
        out("  expected-to-diverge cells that now AGREE (gate needs updating):", repaired)
    return unexpected, repaired

# RUN THE GATE ONCE PER IMPLEMENTATION. The route reads the flags per call, so a single parity
# pass would certify only whichever path the ambient environment selected. Every selectable route
# must produce the identical documented divergence set.
for _impl, _counting, _lsd_radix in (
        ("cmp-sort", "0", "0"),
        ("counting", "1", "0"),
        ("lsd-radix", "0", "1")):
    os.environ["FNP_LEXSORT_COUNTING"] = _counting
    os.environ["FNP_LEXSORT_LSD_RADIX"] = _lsd_radix
    out("parity for impl=%s" % _impl)
    NEW_BAD, REPAIRED = parity()
    if NEW_BAD or REPAIRED:
        out("ABORTING: parity set changed for impl=%s, a ratio would be meaningless" % _impl)
        raise SystemExit(1)
os.environ.pop("FNP_LEXSORT_COUNTING", None)
os.environ.pop("FNP_LEXSORT_LSD_RADIX", None)
out("parity gate: ALL implementations match the documented pre-existing defect set exactly")

def inter(sa, sb, g, k, rounds=1):
    """ABBAABBA so a monotone drift in the host cancels between the arms."""
    ta, tb = [], []
    for r in range(rounds):
        for w in ("a", "b", "b", "a", "a", "b", "b", "a"):
            t = timeit.timeit(sa if w == "a" else sb, globals=g, number=k) / k * 1e9
            (ta if w == "a" else tb).append(t)
    return statistics.median(ta), statistics.median(tb)

N = 1 << 16
REPS = 2
rng = np.random.default_rng(SEED)
out()
out("%-12s%-9s%4s%14s%14s%9s%8s%9s" % ("case","impl","rep","numpy_ns","fnp_ns","ratio","nullNP","nullFNP"))
summary = {}
for card in (2, 4, 8, 32):
    good = []
    for rep in range(REPS):
        # FRESH DRAW PER REP: one draw of one corpus has read 1.001x and 3.897x for this shape,
        # so the sign across draws is the evidence, not any single magnitude.
        k1 = rng.integers(0, card, N).astype(np.float64)
        k2 = rng.integers(0, card, N).astype(np.float64)
        g = {"np": np, "fnp": fnp, "k": (k1, k2)}
        # ALL fnp IMPLEMENTATIONS AND numpy, same arrays, same process, same CPU. The env vars are
        # read per call inside the route, so flipping them here selects the comparison sort, counting
        # pass, or per-key LSD radix pass without rebuilding - which is the only way to compare them soundly, since
        # two rch jobs can land on different workers.
        for impl_name, counting, lsd_radix in (
                ("cmp-sort", "0", "0"),
                ("counting", "1", "0"),
                ("lsd-radix", "0", "1")):
            os.environ["FNP_LEXSORT_COUNTING"] = counting
            os.environ["FNP_LEXSORT_LSD_RADIX"] = lsd_radix
            tn, tf = inter("np.lexsort(k)", "fnp.lexsort(k)", g, 45)
            n1, n2 = inter("np.lexsort(k)", "np.lexsort(k)", g, 45)
            c1, c2 = inter("fnp.lexsort(k)", "fnp.lexsort(k)", g, 45)
            nn, nf = n2 / n1, c2 / c1
            ok = abs(nn - 1) <= 0.02 and abs(nf - 1) <= 0.02
            if ok:
                good.append((impl_name, tf / tn))
            out("%-12s%-9s%4d%14.1f%14.1f%8.3fx%8.3f%9.3f%s"
                  % ("card=%d" % card, impl_name, rep, tn, tf, tf / tn, nn, nf,
                     "" if ok else "  VOID"))
        os.environ.pop("FNP_LEXSORT_COUNTING", None)
        os.environ.pop("FNP_LEXSORT_LSD_RADIX", None)
        del k1, k2, g
    summary[card] = good

# WHERE DOES THE TIME GO? The counting path materialises an n x total_width byte record array
# before it sorts anything. If that traffic dominates, fnp's own time should scale with the RECORD
# WIDTH; if the sort dominates, it should not. Two f64 keys give 16-byte records, two f32 keys give
# 8-byte records, and both are admitted by the counting path - so this separates the two without
# touching the library. numpy times are printed alongside only as a sanity check; the discriminator
# is fnp's OWN time, which removes the incumbent's variance.
out("\n-- phase discriminator: fnp's own time vs RECORD WIDTH, card=2, n=2^16 --")
out("%-26s%6s%16s%16s" % ("keys", "bytes", "numpy_ns", "fnp_ns"))
for label, width, mk in (
        ("2 x f64", 16, lambda m: rng.integers(0, 2, m).astype(np.float64)),
        ("2 x f32",  8, lambda m: rng.integers(0, 2, m).astype(np.float32)),
        ("2 x i32",  8, lambda m: rng.integers(0, 2, m).astype(np.int32)),
        ("2 x i16",  4, lambda m: rng.integers(0, 2, m).astype(np.int16)),
):
    kk = (mk(N), mk(N))
    gg = {"np": np, "fnp": fnp, "k": kk}
    os.environ["FNP_LEXSORT_COUNTING"] = "1"
    tn, tf = inter("np.lexsort(k)", "fnp.lexsort(k)", gg, 45)
    out("%-26s%6d%16.1f%16.1f" % (label, width, tn, tf))
    del kk, gg
os.environ.pop("FNP_LEXSORT_COUNTING", None)

# An out-of-family control: if this is off, the whole run is suspect regardless of the nulls.
a = rng.standard_normal(1 << 20)
g = {"np": np, "fnp": fnp, "a": a}
cn, cf = inter("np.sum(a)", "fnp.sum(a)", g, 30)
out("\nout-of-family control np.sum f64 2^20: numpy %.1f fnp %.1f -> %.3fx" % (cn, cf, cf / cn))

out("\nDECIDABLE ratios per cardinality (both A/A nulls within 2%):")
for card, rs in summary.items():
    for impl_name in ("cmp-sort", "counting", "lsd-radix"):
        vals = [v for (nm, v) in rs if nm == impl_name]
        if vals:
            out("  card=%-4d %-9s n=%d  min %.3fx  median %.3fx  max %.3fx"
                  % (card, impl_name, len(vals), min(vals), statistics.median(vals), max(vals)))
        else:
            out("  card=%-4d %-9s no decidable cells" % (card, impl_name))
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
