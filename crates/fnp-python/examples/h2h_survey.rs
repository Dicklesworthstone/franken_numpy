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
import hashlib, os, random, statistics, sys, timeit
def out(*a):
    print(*a, file=sys.stderr, flush=True)
import numpy as np
import fnp_python as fnp

out("python", sys.version.split()[0], "| numpy", np.__version__)
out("in-process ELF sha256", hashlib.sha256(open(EXE_PATH, "rb").read()).hexdigest())
out("host", os.uname().nodename, "| loadavg", [round(x, 2) for x in os.getloadavg()])
rng = np.random.default_rng(SEED)
boot = random.Random(SEED)
ROUNDS = 21

def pair_rounds(sa, sb, g, k):
    """ROUNDS interleaved rounds (ABBA / BAAB); per round, the mean of B's two samples over A's."""
    ratios, ta, tb = [], [], []
    for r in range(ROUNDS):
        t = {"a": [], "b": []}
        for w in (("a","b","b","a") if r % 2 == 0 else ("b","a","a","b")):
            t[w].append(timeit.timeit(sa if w == "a" else sb, globals=g, number=k) / k * 1e9)
        a, b = sum(t["a"]) / 2, sum(t["b"]) / 2
        ta.append(a); tb.append(b); ratios.append(b / a)
    return ratios, ta, tb

def median_ci(values, draws=2000):
    meds = sorted(statistics.median(boot.choices(values, k=len(values))) for _ in range(draws))
    return statistics.median(values), meds[int(0.025 * draws)], meds[int(0.975 * draws) - 1]

def half_width(ci):
    return max(abs(ci[1] - 1.0), abs(ci[2] - 1.0))

def spread(samples):
    """Within-run spread of one arm, as a fraction of its own median (the pre-2026-09-26 criterion)."""
    lo, hi = min(samples), max(samples)
    return (hi - lo) / statistics.median(samples)

N  = 1 << 20
NS = 1 << 16          # sorts and set ops: smaller, they are O(n log n)
f  = rng.standard_normal(N)
f2 = rng.standard_normal(N)
i64 = rng.integers(0, 1 << 20, N)
fs = rng.standard_normal(NS)
isr = np.sort(rng.integers(0, 1 << 20, NS))
low = rng.integers(0, 8, NS)
b  = rng.integers(0, 2, N).astype(bool)
s16 = rng.standard_normal(16)
s16b = rng.standard_normal(16)
wide = rng.integers(0, 1 << 40, N)

# (label, expression, globals) - one expression, evaluated identically against np and fnp.
CASES = [
    # Small-n and argsort cells named by deadlock-audit-rc0923-epic-71qy3.19: stable losses the
    # incumbent-spread criterion hid on 2026-09-23 (max n=16 2.33x, argsort i64 2^20 3.87x).
    ("max f64 n=16",         "M.max(s)",                    {"s": s16}),
    ("exp f64 n=16",         "M.exp(s)",                    {"s": s16}),
    ("sqrt f64 n=16",        "M.sqrt(M.abs(s))",            {"s": s16}),
    ("unique f64 n=16",      "M.unique(s)",                 {"s": s16}),
    ("add f64 n=16",         "M.add(s, t)",                 {"s": s16, "t": s16b}),
    ("multiply f64 n=16",    "M.multiply(s, t)",            {"s": s16, "t": s16b}),
    ("divide f64 n=16",      "M.divide(s, t)",              {"s": s16, "t": s16b}),
    ("argsort f64 2^20",     "M.argsort(f)",                {"f": f}),
    ("argsort i64<2^40 2^20", "M.argsort(w)",               {"w": wide}),
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

# THE CRITERION (deadlock-audit-rc0923-epic-71qy3.19) is the repo's live dual-null contract
# (`report_dual_null_contract_gate` in benches/common/mod.rs), not the incumbent's min-to-max spread
# this board used from 9a71376a: that took the WORST excursion as the noise estimate, which on a
# loaded host is 30-200% and hid every stable loss below it (max n=16 at 2.33x, argsort 2^20 at
# 2-3.9x). A cell is a LOSS when the effect's median-CI lies above 1, its median lies above both
# A/A null CIs, and it exceeds twice the larger null half-width (measured from 1.0, floor 1%); a WIN
# mirrors that below 1; anything else is UNDECIDED. The old verdict is printed beside it.
out("")
out("%-24s%12s%12s%9s%19s%17s%17s%9s  %-10s %s"
    % ("case","numpy_ns","fnp_ns","ratio","effect_ci95","npnull_ci95","fnpnull_ci95","req_2x","verdict","old"))
losses, wins = [], []
for label, expr, base in CASES:
    gn = dict(base); gn["M"] = np
    gf = dict(base); gf["M"] = fnp
    g  = dict(base); g["np"] = np; g["fnp"] = fnp
    sa = expr.replace("M.", "np.")
    sb = expr.replace("M.", "fnp.")
    try:
        # correctness first: a ratio for a wrong answer is worthless
        wv, gv = np.asarray(eval(sa, gn | {"np": np})), np.asarray(eval(sb, gf | {"fnp": fnp}))
        if "argsort" in expr:
            # The default sort is not stable: equal keys may come back in either order, so an
            # argsort agrees when it orders the VALUES identically.
            keys = next(iter(base.values()))
            agree = wv.shape == gv.shape and np.array_equal(keys[wv], keys[gv])
        else:
            agree = wv.shape == gv.shape and np.allclose(wv, gv, rtol=1e-12, atol=0, equal_nan=True)
    except Exception as e:
        out("%-24s  SKIPPED (%s)" % (label, type(e).__name__)); continue
    # About 4 ms per timed call, whatever the op costs.
    k = max(1, int(0.004 / max(timeit.timeit(sb, globals=g, number=1), 1e-7)))
    effect, tn, tf = pair_rounds(sa, sb, g, k)
    np_null, na, nb = pair_rounds(sa, sa, g, k)
    fnp_null, _, _ = pair_rounds(sb, sb, g, k)
    e, n1, n2 = median_ci(effect), median_ci(np_null), median_ci(fnp_null)
    required = max(2 * max(half_width(n1), half_width(n2)), 0.01)
    envelope_lo, envelope_hi = min(n1[1], n2[1]), max(n1[2], n2[2])
    if not agree:
        verdict = "WRONG"
    elif e[1] > 1.0 and e[0] > envelope_hi and e[0] - 1.0 >= required:
        verdict = "LOSS"
        losses.append((e[0], label, e))
    elif e[2] < 1.0 and e[0] < envelope_lo and 1.0 - e[0] >= required:
        verdict = "WIN"
        wins.append((e[0], label, e))
    else:
        verdict = "UNDECIDED"
    # The pre-2026-09-26 verdict, for the record: nulls within 2% and |effect| above the incumbent's
    # within-run min-to-max spread.
    old_ok = abs(n1[0] - 1) <= 0.02 and abs(n2[0] - 1) <= 0.02
    old = "actionable" if old_ok and agree and abs(e[0] - 1) > max(spread(tn), spread(na), spread(nb)) else "hidden"
    out("%-24s%12.1f%12.1f%8.3fx  [%6.3f,%6.3f]  [%6.3f,%6.3f]  [%6.3f,%6.3f]%9.3f  %-10s %s"
        % (label, statistics.median(tn), statistics.median(tf), e[0], e[1], e[2], n1[1], n1[2],
           n2[1], n2[2], required, verdict, old))

out("")
out("RANKED LOSSES (effect median-CI above 1, outside both nulls, beyond 2x the null half-width):")
for r, label, e in sorted(losses, reverse=True):
    out("  %8.3fx  %-24s ci95=[%.3f,%.3f]" % (r, label, e[1], e[2]))
out("WINS:")
for r, label, e in sorted(wins):
    out("  %8.3fx  %-24s ci95=[%.3f,%.3f]" % (r, label, e[1], e[2]))
out("%d LOSS, %d WIN, %d other of %d cells" % (len(losses), len(wins), len(CASES) - len(losses) - len(wins), len(CASES)))
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
