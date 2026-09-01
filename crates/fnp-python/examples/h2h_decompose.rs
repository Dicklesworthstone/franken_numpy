//! DECOMPOSITION probe for the two veins the wide board (`deadlock-audit-2nudi`) opened.
//!
//! The wide board reported two things the narrow 24-cell board could not see:
//!   1. `sqrt f64 [::2]` at 1.721x against a 2.0% incumbent spread - 802 US of excess on a single
//!      call, the largest absolute headroom anywhere on either board.
//!   2. a `tiny` family where THIRTEEN small-n cells lose 1.07-2.41x with 1.4-7.5% spreads.
//!
//! Neither is directly actionable, because both were measured on COMPOSED expressions
//! (`sqrt(abs(x))` is two calls) and the tiny cells report a RATIO where the thing that matters is
//! a flat per-call NS EXCESS. This probe decomposes both: one op per cell, and the excess in
//! nanoseconds printed next to the ratio.
//!
//! Same method as the boards: append_to_inittab, live numpy in the same invocation, interleaved
//! ABBAABBA, dual A/A null, incumbent-spread criterion.

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
out("cpus", os.cpu_count(), "| seed", SEED)
rng = np.random.default_rng(SEED)

def inter(sa, sb, g, k, rounds=4):
    ta, tb = [], []
    for r in range(rounds):
        for w in (("a","b","b","a") if r % 2 == 0 else ("b","a","a","b")):
            t = timeit.timeit(sa if w == "a" else sb, globals=g, number=k) / k * 1e9
            (ta if w == "a" else tb).append(t)
    return statistics.median(ta), statistics.median(tb), ta, tb

def spread(samples):
    lo, hi = min(samples), max(samples)
    return (hi - lo) / statistics.median(samples)

def K(n):
    return int(max(3, min(8000, 4e7 // max(n, 1))))

N19 = 1 << 19
src = np.abs(rng.standard_normal(1 << 20)) + 0.5      # strictly positive: sqrt never warns
c19 = src[:N19].copy()                                 # contiguous 2^19 f64
s19 = src[::2]                                         # STRIDED 2^19 f64, same element count
c19f = c19.astype(np.float32)
s19f = src.astype(np.float32)[::2]
c20 = src.copy()                                       # contiguous 2^20 f64

t8   = np.abs(rng.standard_normal(8)) + 0.5
t8b  = np.abs(rng.standard_normal(8)) + 0.5
t64  = np.abs(rng.standard_normal(64)) + 0.5
t64b = np.abs(rng.standard_normal(64)) + 0.5
t1k  = np.abs(rng.standard_normal(1024)) + 0.5
t1kb = np.abs(rng.standard_normal(1024)) + 0.5
t16k = np.abs(rng.standard_normal(16384)) + 0.5
t16kb= np.abs(rng.standard_normal(16384)) + 0.5

CASES = [
  # ---- VEIN 1: decompose sqrt f64 [::2]. ONE op per cell; the board cell was sqrt(abs(x)). ----
  ("v1", "abs  f64 contig 2^19",  "M.abs(a)",         {"a": c19}, N19),
  ("v1", "abs  f64 [::2] 2^19",   "M.abs(a)",         {"a": s19}, N19),
  ("v1", "sqrt f64 contig 2^19",  "M.sqrt(a)",        {"a": c19}, N19),
  ("v1", "sqrt f64 [::2] 2^19",   "M.sqrt(a)",        {"a": s19}, N19),
  ("v1", "sqrt f64 contig 2^20",  "M.sqrt(a)",        {"a": c20}, 1 << 20),
  ("v1", "sqrt(abs) contig 2^19", "M.sqrt(M.abs(a))", {"a": c19}, N19),
  ("v1", "sqrt(abs) [::2] 2^19",  "M.sqrt(M.abs(a))", {"a": s19}, N19),
  ("v1", "sqrt f32 contig 2^19",  "M.sqrt(a)",        {"a": c19f}, N19),
  ("v1", "sqrt f32 [::2] 2^19",   "M.sqrt(a)",        {"a": s19f}, N19),
  # cheap unaries on the same operands: is this sqrt, or the whole f64 unary route?
  ("v1", "square f64 contig 2^19","M.square(a)",      {"a": c19}, N19),
  ("v1", "square f64 [::2] 2^19", "M.square(a)",      {"a": s19}, N19),
  ("v1", "negative f64 [::2]",    "M.negative(a)",    {"a": s19}, N19),
  ("v1", "floor f64 [::2]",       "M.floor(a)",       {"a": s19}, N19),
  ("v1", "cbrt f64 contig 2^19",  "M.cbrt(a)",        {"a": c19}, N19),

  # ---- VEIN 2: entry cost. ONE op per cell; excess must be FLAT in n if it is a gate tax. ----
  ("v2", "sum  n=8",       "M.sum(a)",         {"a": t8}, 8),
  ("v2", "sum  n=64",      "M.sum(a)",         {"a": t64}, 64),
  ("v2", "sum  n=1024",    "M.sum(a)",         {"a": t1k}, 1024),
  ("v2", "sum  n=16384",   "M.sum(a)",         {"a": t16k}, 16384),
  ("v2", "add  n=8",       "M.add(a, c)",      {"a": t8,  "c": t8b}, 8),
  ("v2", "add  n=64",      "M.add(a, c)",      {"a": t64, "c": t64b}, 64),
  ("v2", "add  n=1024",    "M.add(a, c)",      {"a": t1k, "c": t1kb}, 1024),
  ("v2", "add  n=16384",   "M.add(a, c)",      {"a": t16k,"c": t16kb}, 16384),
  ("v2", "sqrt n=8",       "M.sqrt(a)",        {"a": t8}, 8),
  ("v2", "sqrt n=64",      "M.sqrt(a)",        {"a": t64}, 64),
  ("v2", "sqrt n=1024",    "M.sqrt(a)",        {"a": t1k}, 1024),
  ("v2", "sqrt n=16384",   "M.sqrt(a)",        {"a": t16k}, 16384),
  ("v2", "abs  n=64",      "M.abs(a)",         {"a": t64}, 64),
  ("v2", "isnan n=64",     "M.isnan(a)",       {"a": t64}, 64),
  ("v2", "max  n=64",      "M.max(a)",         {"a": t64}, 64),
  ("v2", "argmax n=64",    "M.argmax(a)",      {"a": t64}, 64),
  ("v2", "sort n=64",      "M.sort(a)",        {"a": t64}, 64),
  ("v2", "sort n=1024",    "M.sort(a)",        {"a": t1k}, 1024),
  ("v2", "argsort n=64",   "M.argsort(a)",     {"a": t64}, 64),
  ("v2", "concat n=64",    "M.concatenate([a, c])", {"a": t64, "c": t64b}, 64),
  ("v2", "concat n=16384", "M.concatenate([a, c])", {"a": t16k,"c": t16kb}, 16384),
  ("v2", "asarray n=64",   "M.asarray(a)",     {"a": t64}, 64),
  ("v2", "multiply n=64",  "M.multiply(a, c)", {"a": t64, "c": t64b}, 64),
  ("v2", "divide n=64",    "M.divide(a, c)",   {"a": t64, "c": t64b}, 64),
  ("v2", "mean n=64",      "M.mean(a)",        {"a": t64}, 64),
  ("v2", "clip n=64",      "M.clip(a, -1.0, 1.0)", {"a": t64}, 64),
]

out("")
out("%-4s%-24s%13s%13s%9s%13s%8s%9s%9s"
    % ("vein","case","numpy_ns","fnp_ns","ratio","excess_ns","nullNP","nullFNP","incSprd"))
for vein, label, expr, base, nelem in CASES:
    gn = dict(base); gn["M"] = np
    gf = dict(base); gf["M"] = fnp
    g  = dict(base); g["np"] = np; g["fnp"] = fnp
    sa = expr.replace("M.", "np.")
    sb = expr.replace("M.", "fnp.")
    try:
        wv, gv = np.asarray(eval(sa, gn | {"np": np})), np.asarray(eval(sb, gf | {"fnp": fnp}))
        agree = wv.shape == gv.shape and np.allclose(wv, gv, rtol=1e-9, atol=0, equal_nan=True)
    except Exception as e:
        out("%-4s%-24s  SKIPPED (%s: %s)" % (vein, label, type(e).__name__, str(e)[:60])); continue
    k = K(nelem)
    tn, tf, sn, sf = inter(sa, sb, g, k)
    n1, n2, na, nb = inter(sa, sa, g, k)
    c1, c2, _, _ = inter(sb, sb, g, k)
    nn, nf = n2 / n1, c2 / c1
    ok = abs(nn - 1) <= 0.02 and abs(nf - 1) <= 0.02
    inc_spread = max(spread(sn), spread(na), spread(nb))
    flag = "" if ok else "  VOID"
    if not agree:
        flag += "  VALUES DIFFER"
    if ok and agree and abs(tf / tn - 1.0) <= inc_spread:
        flag += "  NOISE>EFFECT"
    out("%-4s%-24s%13.1f%13.1f%8.3fx%13.1f%8.3f%9.3f%8.1f%%%s"
        % (vein, label, tn, tf, tf / tn, tf - tn, nn, nf, 100 * inc_spread, flag))
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
