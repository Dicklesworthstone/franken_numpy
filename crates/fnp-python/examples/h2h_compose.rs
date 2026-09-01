//! WHY DOES `fnp.sqrt(fnp.abs(a))` LOSE 2.03x WHEN NEITHER CALL LOSES ALONE?
//!
//! `h2h_decompose` (deadlock-audit-2nudi) measured, on the same contiguous 2^19 f64 operand in one
//! invocation:
//!     abs  alone   0.983x   (fnp 134.3 us vs numpy 136.6 us)
//!     sqrt alone   0.403x   (fnp 317.9 us vs numpy 789.2 us)  - a 2.5x WIN
//!     sqrt(abs)    2.027x   (fnp 1858.9 us vs numpy 917.0 us) - a 2.0x LOSS
//! numpy's composition is the sum of its parts (136.6 + 789.2 = 925.8 ~ 917.0). Ours is 1858.9
//! against a 452.2 sum - 1407 us that exists ONLY when the two calls are composed.
//!
//! A ratio cannot say which call gets slow, so this is a 2x2: each arm's sqrt is fed each arm's
//! abs output. Plus three controls that separate the three candidate explanations -
//!   (a) OUR OUTPUT IS A BAD INPUT: a persistent property of the array fnp.abs returns.
//!       -> cell 4 precomputes the intermediate ONCE, outside the timed statement. If the penalty
//!          is a property of the buffer it survives; if it is churn it vanishes.
//!   (b) TWO LIVE 4 MiB ALLOCATIONS PER ITERATION: allocator/page-fault churn, not composition.
//!       -> cell 6 allocates twice per iteration with NO composition at all.
//!   (c) SOMETHING SPECIFIC TO sqrt: -> cell 7 composes two cheap unaries instead.
//! and a size sweep, because a churn explanation must have a threshold and a code explanation
//! need not.

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
base = np.abs(rng.standard_normal(1 << 21)) + 0.5
a19 = base[:N19].copy()

# METADATA FIRST. If the two abs outputs differ in any flag, stride, alignment or base object,
# that is the answer and no timing is needed.
def describe(name, arr):
    ap = arr.__array_interface__["data"][0]
    out("  %-14s type=%-9s dtype=%-8s shape=%-10s strides=%-10s C=%-5s F=%-5s OWN=%-5s ALIGN=%-5s "
        "W=%-5s base=%-6s ptr%%4096=%d"
        % (name, type(arr).__name__, arr.dtype, arr.shape, arr.strides,
           arr.flags.c_contiguous, arr.flags.f_contiguous, arr.flags.owndata,
           arr.flags.aligned, arr.flags.writeable,
           type(arr.base).__name__ if arr.base is not None else "None", ap % 4096))
out("")
out("INTERMEDIATE METADATA (the array each abs hands to sqrt):")
describe("np.abs", np.abs(a19))
describe("fnp.abs", fnp.abs(a19))
na, fa = np.abs(a19), fnp.abs(a19)
out("  bytes_equal=%s" % (na.tobytes() == fa.tobytes()))

pre_np  = np.abs(a19)     # precomputed ONCE, reused every iteration
pre_fnp = fnp.abs(a19)
g = {"np": np, "fnp": fnp, "a": a19, "pre_np": pre_np, "pre_fnp": pre_fnp}

# (label, numpy-arm statement, fnp-arm statement, n)
CELLS = [
  ("1 sqrt(abs) BOTH",      "np.sqrt(np.abs(a))",     "fnp.sqrt(fnp.abs(a))",   N19),
  ("2 OUR sqrt, np abs",    "np.sqrt(np.abs(a))",     "fnp.sqrt(np.abs(a))",    N19),
  ("3 np sqrt, OUR abs",    "np.sqrt(np.abs(a))",     "np.sqrt(fnp.abs(a))",    N19),
  ("4 sqrt(PRE fnp.abs)",   "np.sqrt(pre_fnp)",       "fnp.sqrt(pre_fnp)",      N19),
  ("5 sqrt(PRE np.abs)",    "np.sqrt(pre_np)",        "fnp.sqrt(pre_np)",       N19),
  ("6 abs TWICE, no compose","np.abs(a); np.abs(a)",  "fnp.abs(a); fnp.abs(a)", N19),
  ("7 negative(abs)",       "np.negative(np.abs(a))", "fnp.negative(fnp.abs(a))", N19),
  ("8 square(abs)",         "np.square(np.abs(a))",   "fnp.square(fnp.abs(a))", N19),
  ("9 abs(abs)",            "np.abs(np.abs(a))",      "fnp.abs(fnp.abs(a))",    N19),
]
# size sweep on cell 1: a churn explanation needs a threshold, a code explanation does not.
for lg in (12, 14, 16, 18, 19, 20, 21):
    n = 1 << lg
    g["a%d" % lg] = base[:n].copy()
    CELLS.append(("S sqrt(abs) 2^%-2d" % lg,
                  "np.sqrt(np.abs(a%d))" % lg, "fnp.sqrt(fnp.abs(a%d))" % lg, n))

out("")
out("%-24s%13s%13s%9s%13s%8s%9s%9s"
    % ("cell","numpy_ns","fnp_ns","ratio","excess_ns","nullNP","nullFNP","incSprd"))
for label, sa, sb, n in CELLS:
    try:
        wv, gv = np.asarray(eval(sa.split(";")[-1], g)), np.asarray(eval(sb.split(";")[-1], g))
        agree = wv.shape == gv.shape and np.allclose(wv, gv, rtol=1e-9, atol=0, equal_nan=True)
    except Exception as e:
        out("%-24s  SKIPPED (%s: %s)" % (label, type(e).__name__, str(e)[:60])); continue
    k = K(n)
    tn, tf, sn, sf = inter(sa, sb, g, k)
    n1, n2, na_, nb_ = inter(sa, sa, g, k)
    c1, c2, _, _ = inter(sb, sb, g, k)
    nn, nf = n2 / n1, c2 / c1
    ok = abs(nn - 1) <= 0.02 and abs(nf - 1) <= 0.02
    isp = max(spread(sn), spread(na_), spread(nb_))
    flag = "" if ok else "  VOID"
    if not agree:
        flag += "  VALUES DIFFER"
    if ok and agree and abs(tf / tn - 1.0) <= isp:
        flag += "  NOISE>EFFECT"
    out("%-24s%13.1f%13.1f%8.3fx%13.1f%8.3f%9.3f%8.1f%%%s"
        % (label, tn, tf, tf / tn, tf - tn, nn, nf, 100 * isp, flag))
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
