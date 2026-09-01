//! Head-to-head WIDE SURVEY: the axes `h2h_survey` does not cover.
//!
//! `deadlock-audit-qpylx` closed the 24-cell board that harness measures: once the
//! incumbent-spread criterion landed, 8 of 24 cells were ACTIONABLE and SEVEN of those eight were
//! WINS. Its retry predicate says the next real perf work needs a WIDER board rather than another
//! pass over the same 24 cells, and names the axes: f32, complex, 2-D axis reductions, strided
//! inputs, and small-n entry costs.
//!
//! This binary is that board. The METHOD is copied verbatim from `h2h_survey` and is not up for
//! redesign: `append_to_inittab!` puts the module INTO this binary so no `.so` is involved, the
//! incumbent is the live numpy in the same interpreter, every cell is interleaved ABBAABBA with a
//! dual A/A null, the effect must exceed the incumbent's own within-run spread, and the binary
//! hashes itself so the artifact identity cannot drift from what was timed.
//!
//! Two deliberate exclusions. `dot`/`matmul` are absent: the dense-linalg lane is closed (packed
//! panel GEMM already shipped) and a BLAS cell would only re-open the thread-pinning argument.
//! And every case carries its OWN repeat count derived from its OWN element count - the narrow
//! board used one global `k`, which is wrong the moment a cell has 8 elements instead of 2^20.
//!
//! Run with `rch exec --job -- cargo run --release -p fnp-python --example h2h_survey_wide`.
//! Results go to STDERR because rch returns the remote command's stderr and not its stdout.

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
out("cpus", os.cpu_count(), "| seed", SEED)
rng = np.random.default_rng(SEED)

def inter(sa, sb, g, k, rounds=4):
    ta, tb = [], []
    for r in range(rounds):
        for w in (("a","b","b","a") if r % 2 == 0 else ("b","a","a","b")):
            t = timeit.timeit(sa if w == "a" else sb, globals=g, number=k) / k * 1e9
            (ta if w == "a" else tb).append(t)
    # Return the per-round samples too: the median alone cannot tell you whether an effect is
    # bigger than the noise the INCUMBENT arm carries within this very run.
    return statistics.median(ta), statistics.median(tb), ta, tb

def spread(samples):
    """Within-run spread of one arm, as a fraction of its own median."""
    lo, hi = min(samples), max(samples)
    return (hi - lo) / statistics.median(samples)

def K(n):
    """Per-case repeat count from the case's OWN element count, not a global constant."""
    return int(max(3, min(8000, 4e7 // max(n, 1))))

N   = 1 << 20
NS  = 1 << 16

f64a = rng.standard_normal(N)
f64b = rng.standard_normal(N)
f32a = f64a.astype(np.float32)
f32b = f64b.astype(np.float32)
c128a = f64a + 1j * f64b
c128b = f64b + 1j * f64a
c64a = c128a.astype(np.complex64)
c64b = c128b.astype(np.complex64)

fs64 = f64a[:NS].copy()
fs32 = f32a[:NS].copy()
cs128 = c128a[:NS].copy()
low32 = rng.integers(0, 8, NS).astype(np.float32)
isr32 = np.sort(rng.integers(0, 1 << 20, NS)).astype(np.float32)

# 2-D: 1024x1024 f64 is exactly the 2^20 elements the 1-D cells use, so an axis reduction is
# comparable in TRAFFIC to its flat sibling and any gap is the axis machinery, not the size.
m2 = rng.standard_normal((1024, 1024))
m2f = m2.astype(np.float32)
m2s = rng.standard_normal((512, 512))

# STRIDED views. Nothing is copied: these are the exact non-contiguous operands the narrow board
# never fed either arm.
s1a = f64a[::2]
s1b = f64b[::2]
s2c = m2[:, ::2]
s2r = m2[::2, :]

b   = rng.integers(0, 2, N).astype(bool)
i64 = rng.integers(0, 1 << 20, N)

t8    = rng.standard_normal(8)
t8b   = rng.standard_normal(8)
t64   = rng.standard_normal(64)
t64b  = rng.standard_normal(64)
t1k   = rng.standard_normal(1024)
t1kb  = rng.standard_normal(1024)

# (family, label, expression, globals, element-count-for-k)
CASES = [
  # ---- f32: the narrow board is f64 only ----------------------------------------------------
  ("f32", "sum f32 2^20",        "M.sum(a)",              {"a": f32a}, N),
  ("f32", "mean f32 2^20",       "M.mean(a)",             {"a": f32a}, N),
  ("f32", "std f32 2^20",        "M.std(a)",              {"a": f32a}, N),
  ("f32", "min f32 2^20",        "M.min(a)",              {"a": f32a}, N),
  ("f32", "argmin f32 2^20",     "M.argmin(a)",           {"a": f32a}, N),
  ("f32", "cumsum f32 2^20",     "M.cumsum(a)",           {"a": f32a}, N),
  ("f32", "add f32 2^20",        "M.add(a, c)",           {"a": f32a, "c": f32b}, N),
  ("f32", "multiply f32 2^20",   "M.multiply(a, c)",      {"a": f32a, "c": f32b}, N),
  ("f32", "divide f32 2^20",     "M.divide(a, c)",        {"a": f32a, "c": f32b}, N),
  ("f32", "sqrt f32 2^20",       "M.sqrt(M.abs(a))",      {"a": f32a}, N),
  ("f32", "isnan f32 2^20",      "M.isnan(a)",            {"a": f32a}, N),
  ("f32", "clip f32 2^20",       "M.clip(a, -1.0, 1.0)",  {"a": f32a}, N),
  ("f32", "diff f32 2^20",       "M.diff(a)",             {"a": f32a}, N),
  ("f32", "concatenate f32 2^20","M.concatenate([a, c])", {"a": f32a, "c": f32b}, N),
  ("f32", "sort f32 2^16",       "M.sort(a)",             {"a": fs32}, NS),
  ("f32", "argsort f32 2^16",    "M.argsort(a)",          {"a": fs32}, NS),
  ("f32", "unique f32 2^16",     "M.unique(a)",           {"a": low32}, NS),
  ("f32", "searchsorted f32 2^16","M.searchsorted(a, a)", {"a": isr32}, NS),

  # ---- complex ------------------------------------------------------------------------------
  ("cplx", "sum c128 2^20",      "M.sum(a)",              {"a": c128a}, N),
  ("cplx", "mean c128 2^20",     "M.mean(a)",             {"a": c128a}, N),
  ("cplx", "add c128 2^20",      "M.add(a, c)",           {"a": c128a, "c": c128b}, N),
  ("cplx", "multiply c128 2^20", "M.multiply(a, c)",      {"a": c128a, "c": c128b}, N),
  ("cplx", "abs c128 2^20",      "M.abs(a)",              {"a": c128a}, N),
  ("cplx", "conj c128 2^20",     "M.conj(a)",             {"a": c128a}, N),
  ("cplx", "add c64 2^20",       "M.add(a, c)",           {"a": c64a, "c": c64b}, N),
  ("cplx", "multiply c64 2^20",  "M.multiply(a, c)",      {"a": c64a, "c": c64b}, N),
  ("cplx", "abs c64 2^20",       "M.abs(a)",              {"a": c64a}, N),
  ("cplx", "sort c128 2^16",     "M.sort(a)",             {"a": cs128}, NS),

  # ---- 2-D axis reductions: same 2^20 elements, so a gap is the AXIS machinery ---------------
  ("axis", "sum f64 ax0 1024^2", "M.sum(a, axis=0)",      {"a": m2}, N),
  ("axis", "sum f64 ax1 1024^2", "M.sum(a, axis=1)",      {"a": m2}, N),
  ("axis", "mean f64 ax0",       "M.mean(a, axis=0)",     {"a": m2}, N),
  ("axis", "mean f64 ax1",       "M.mean(a, axis=1)",     {"a": m2}, N),
  ("axis", "max f64 ax0",        "M.max(a, axis=0)",      {"a": m2}, N),
  ("axis", "max f64 ax1",        "M.max(a, axis=1)",      {"a": m2}, N),
  ("axis", "argmax f64 ax0",     "M.argmax(a, axis=0)",   {"a": m2}, N),
  ("axis", "argmax f64 ax1",     "M.argmax(a, axis=1)",   {"a": m2}, N),
  ("axis", "std f64 ax0",        "M.std(a, axis=0)",      {"a": m2}, N),
  ("axis", "std f64 ax1",        "M.std(a, axis=1)",      {"a": m2}, N),
  ("axis", "cumsum f64 ax0",     "M.cumsum(a, axis=0)",   {"a": m2}, N),
  ("axis", "cumsum f64 ax1",     "M.cumsum(a, axis=1)",   {"a": m2}, N),
  ("axis", "sum f32 ax0",        "M.sum(a, axis=0)",      {"a": m2f}, N),
  ("axis", "sum f32 ax1",        "M.sum(a, axis=1)",      {"a": m2f}, N),
  ("axis", "sort f64 ax0 512^2", "M.sort(a, axis=0)",     {"a": m2s}, 1 << 18),
  ("axis", "sort f64 ax1 512^2", "M.sort(a, axis=1)",     {"a": m2s}, 1 << 18),

  # ---- STRIDED operands: never fed to either arm before -------------------------------------
  ("strd", "sum f64 [::2] 2^19", "M.sum(a)",              {"a": s1a}, 1 << 19),
  ("strd", "mean f64 [::2]",     "M.mean(a)",             {"a": s1a}, 1 << 19),
  ("strd", "max f64 [::2]",      "M.max(a)",              {"a": s1a}, 1 << 19),
  ("strd", "add f64 [::2]",      "M.add(a, c)",           {"a": s1a, "c": s1b}, 1 << 19),
  ("strd", "multiply f64 [::2]", "M.multiply(a, c)",      {"a": s1a, "c": s1b}, 1 << 19),
  ("strd", "sqrt f64 [::2]",     "M.sqrt(M.abs(a))",      {"a": s1a}, 1 << 19),
  ("strd", "isnan f64 [::2]",    "M.isnan(a)",            {"a": s1a}, 1 << 19),
  ("strd", "cumsum f64 [::2]",   "M.cumsum(a)",           {"a": s1a}, 1 << 19),
  ("strd", "clip f64 [::2]",     "M.clip(a, -1.0, 1.0)",  {"a": s1a}, 1 << 19),
  ("strd", "sum 2-D [:, ::2]",   "M.sum(a)",              {"a": s2c}, 1 << 19),
  ("strd", "sum 2-D [:, ::2] ax0","M.sum(a, axis=0)",     {"a": s2c}, 1 << 19),
  ("strd", "sum 2-D [::2, :] ax1","M.sum(a, axis=1)",     {"a": s2r}, 1 << 19),
  ("strd", "asarray f64 [::2]",  "M.asarray(a)",          {"a": s1a}, 1 << 19),

  # ---- small-n ENTRY COSTS: where a per-call gate tax is the whole measurement ---------------
  ("tiny", "sum f64 n=8",        "M.sum(a)",              {"a": t8}, 8),
  ("tiny", "sum f64 n=64",       "M.sum(a)",              {"a": t64}, 64),
  ("tiny", "sum f64 n=1024",     "M.sum(a)",              {"a": t1k}, 1024),
  ("tiny", "mean f64 n=64",      "M.mean(a)",             {"a": t64}, 64),
  ("tiny", "std f64 n=64",       "M.std(a)",              {"a": t64}, 64),
  ("tiny", "max f64 n=64",       "M.max(a)",              {"a": t64}, 64),
  ("tiny", "argmax f64 n=64",    "M.argmax(a)",           {"a": t64}, 64),
  ("tiny", "add f64 n=8",        "M.add(a, c)",           {"a": t8, "c": t8b}, 8),
  ("tiny", "add f64 n=64",       "M.add(a, c)",           {"a": t64, "c": t64b}, 64),
  ("tiny", "add f64 n=1024",     "M.add(a, c)",           {"a": t1k, "c": t1kb}, 1024),
  ("tiny", "multiply f64 n=64",  "M.multiply(a, c)",      {"a": t64, "c": t64b}, 64),
  ("tiny", "divide f64 n=64",    "M.divide(a, c)",        {"a": t64, "c": t64b}, 64),
  ("tiny", "sqrt f64 n=64",      "M.sqrt(M.abs(a))",      {"a": t64}, 64),
  ("tiny", "isnan f64 n=64",     "M.isnan(a)",            {"a": t64}, 64),
  ("tiny", "clip f64 n=64",      "M.clip(a, -1.0, 1.0)",  {"a": t64}, 64),
  ("tiny", "sort f64 n=64",      "M.sort(a)",             {"a": t64}, 64),
  ("tiny", "argsort f64 n=64",   "M.argsort(a)",          {"a": t64}, 64),
  ("tiny", "cumsum f64 n=1024",  "M.cumsum(a)",           {"a": t1k}, 1024),
  ("tiny", "asarray f64 n=64",   "M.asarray(a)",          {"a": t64}, 64),
  ("tiny", "concatenate n=64",   "M.concatenate([a, c])", {"a": t64, "c": t64b}, 64),
]

out("")
out("%-5s%-26s%13s%13s%9s%8s%9s%9s"
    % ("fam","case","numpy_ns","fnp_ns","ratio","nullNP","nullFNP","incSprd"))
rows, voids, mism, skips = [], 0, 0, 0
for fam, label, expr, base, nelem in CASES:
    gn = dict(base); gn["M"] = np
    gf = dict(base); gf["M"] = fnp
    g  = dict(base); g["np"] = np; g["fnp"] = fnp
    sa = expr.replace("M.", "np.")
    sb = expr.replace("M.", "fnp.")
    try:
        # correctness first: a ratio for a wrong answer is worthless
        wv, gv = np.asarray(eval(sa, gn | {"np": np})), np.asarray(eval(sb, gf | {"fnp": fnp}))
        agree = wv.shape == gv.shape and np.allclose(wv, gv, rtol=1e-9, atol=0, equal_nan=True)
    except Exception as e:
        out("%-5s%-26s  SKIPPED (%s: %s)" % (fam, label, type(e).__name__, str(e)[:60]))
        skips += 1
        continue
    k = K(nelem)
    tn, tf, sn, sf = inter(sa, sb, g, k)
    n1, n2, na, nb = inter(sa, sa, g, k)
    c1, c2, _, _ = inter(sb, sb, g, k)
    nn, nf = n2 / n1, c2 / c1
    ok = abs(nn - 1) <= 0.02 and abs(nf - 1) <= 0.02
    # LEDGER-293 CRITERION, same as the narrow board: an A/A null says the two arms did not DRIFT
    # apart; it says nothing about how much the incumbent bounces WITHIN the run. Take the largest
    # spread the numpy arm showed across its own samples in this cell - from the effect
    # measurement AND from both of its own A/A arms - and require the effect to exceed it.
    inc_spread = max(spread(sn), spread(na), spread(nb))
    effect = abs(tf / tn - 1.0)
    actionable = ok and agree and effect > inc_spread
    flag = "" if ok else "  VOID"
    if not ok:
        voids += 1
    if not agree:
        flag += "  VALUES DIFFER"
        mism += 1
    if ok and agree and not actionable:
        flag += "  NOISE>EFFECT"
    if actionable:
        rows.append((tf / tn, fam, label, tn, tf, inc_spread))
    out("%-5s%-26s%13.1f%13.1f%8.3fx%8.3f%9.3f%8.1f%%%s"
        % (fam, label, tn, tf, tf / tn, nn, nf, 100 * inc_spread, flag))

out("")
out("ACTIONABLE cells - A/A nulls clean AND |effect| exceeds the incumbent's within-run spread:")
for r, fam, label, tn, tf, isp in sorted(rows, reverse=True):
    marker = "  <-- LOSS" if r > 1.10 else ("  win" if r < 0.90 else "")
    out("  %8.3fx  %-5s%-26s numpy %11.1f  fnp %11.1f  incSprd %5.1f%%%s"
        % (r, fam, label, tn, tf, 100 * isp, marker))
losses = [x for x in rows if x[0] > 1.10]
out("")
out("%d of %d cells ACTIONABLE; %d VOID; %d VALUES DIFFER; %d SKIPPED; %d ACTIONABLE LOSSES"
    % (len(rows), len(CASES), voids, mism, skips, len(losses)))
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
