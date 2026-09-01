//! MECHANISM probe for the composition loss (`deadlock-audit-2nudi`, vein 1).
//!
//! `h2h_compose` established, on one worker in one invocation at 2^19 f64:
//!     np.sqrt(fnp.abs(a))   0.980x   - our abs is NOT the problem
//!     fnp.sqrt(np.abs(a))   2.536x   - our SQRT is, and only here
//!     fnp.sqrt(pre_np_abs)  0.430x   - the SAME sqrt on a buffer allocated ONCE is a 2.3x WIN
//! and a size sweep that brackets it: 2^12/2^14/2^16 win 0.83/0.65/0.60 (SERIAL - the gate is
//! `SQRT_PARALLEL_MIN = 1 << 17`), 2^18 5.396x, 2^19 3.012x, 2^20 1.117x, 2^21 0.414x win.
//! So the same code, same n, same thread count is up to 6x slower when its input array was
//! allocated INSIDE the timed statement. That is a REGIME, not a constant, and the 2^18/2^19
//! rows VOID their nulls (nullFNP 0.691 and 1.030) - the fnp arm itself drifts within the run,
//! which is what allocator warm-up looks like.
//!
//! Two hypotheses, and this probe separates them without rebuilding the kernel:
//!   (H1) PAGE-FAULT STORM. Two live 4 MiB blocks per iteration make the allocator hand back a
//!        FRESH mapping each time, and 64 rayon threads first-touch it concurrently. Test: count
//!        MINOR FAULTS per call (ru_minflt) for each arm. If ours does not exceed numpy's, H1 is
//!        dead - no rebuild needed to kill it.
//!   (H2) MMAP CHURN specifically, i.e. glibc munmapping and re-mmapping the block. Test: raise
//!        M_MMAP_THRESHOLD and M_TRIM_THRESHOLD through mallopt so 4 MiB comes from the heap and
//!        is never returned to the kernel. If the loss vanishes, H2 is the mechanism.
//! Both are read-only experiments on the SHIPPED binary.

use pyo3::prelude::*;
use std::ffi::CString;

use fnp_python::fnp_python;

const HARNESS: &str = r#"
import ctypes, hashlib, os, resource, statistics, sys, timeit
def out(*a):
    print(*a, file=sys.stderr, flush=True)
import numpy as np
import fnp_python as fnp

out("python", sys.version.split()[0], "| numpy", np.__version__)
out("in-process ELF sha256", hashlib.sha256(open(EXE_PATH, "rb").read()).hexdigest())
out("host", os.uname().nodename, "| loadavg", [round(x, 2) for x in os.getloadavg()])
out("cpus", os.cpu_count(), "| rayon", os.environ.get("RAYON_NUM_THREADS", "(default)"))
rng = np.random.default_rng(SEED)

def inter(sa, sb, g, k, rounds=4):
    ta, tb = [], []
    for r in range(rounds):
        for w in (("a","b","b","a") if r % 2 == 0 else ("b","a","a","b")):
            t = timeit.timeit(sa if w == "a" else sb, globals=g, number=k) / k * 1e9
            (ta if w == "a" else tb).append(t)
    return statistics.median(ta), statistics.median(tb), ta, tb

def spread(s):
    return (max(s) - min(s)) / statistics.median(s)

def faults_per_call(stmt, g, k):
    """Minor faults attributable to one call, measured on the SAME statement timeit runs."""
    timeit.timeit(stmt, globals=g, number=k)          # warm, so setup faults are not counted
    before = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
    timeit.timeit(stmt, globals=g, number=k)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
    return (after - before) / k

SIZES = [1 << 16, 1 << 18, 1 << 19, 1 << 20, 1 << 21]
base = np.abs(rng.standard_normal(1 << 21)) + 0.5

out("")
out("H1  MINOR FAULTS PER CALL - if fnp does not exceed numpy, the page-fault story is dead.")
out("    (%d pages would be touched by one %s-element f64 output)" % ((1 << 19) * 8 // 4096, "2^19"))
out("%-8s%14s%14s%14s%14s%14s"
    % ("n", "np_compose", "fnp_compose", "np_pre", "fnp_pre", "pages_out"))
for n in SIZES:
    a = base[:n].copy()
    pre = np.abs(a)
    g = {"np": np, "fnp": fnp, "a": a, "pre": pre}
    k = int(max(3, min(2000, 4e7 // n)))
    out("%-8s%14.1f%14.1f%14.1f%14.1f%14d"
        % ("2^%d" % (n.bit_length() - 1),
           faults_per_call("np.sqrt(np.abs(a))", g, k),
           faults_per_call("fnp.sqrt(np.abs(a))", g, k),
           faults_per_call("np.sqrt(pre)", g, k),
           faults_per_call("fnp.sqrt(pre)", g, k),
           n * 8 // 4096))

def sweep(tag):
    out("")
    out("%-6s%-22s%13s%13s%9s%13s%8s%9s%9s"
        % ("phase", "cell", "numpy_ns", "fnp_ns", "ratio", "excess_ns", "nullNP", "nullFNP", "incSprd"))
    for n in SIZES:
        a = base[:n].copy()
        pre = np.abs(a)
        g = {"np": np, "fnp": fnp, "a": a, "pre": pre}
        k = int(max(3, min(2000, 4e7 // n)))
        for label, sa, sb in (
            ("compose 2^%d" % (n.bit_length() - 1), "np.sqrt(np.abs(a))", "fnp.sqrt(np.abs(a))"),
            ("settled 2^%d" % (n.bit_length() - 1), "np.sqrt(pre)",       "fnp.sqrt(pre)"),
        ):
            tn, tf, sn, _ = inter(sa, sb, g, k, rounds=6)
            n1, n2, na, nb = inter(sa, sa, g, k, rounds=6)
            c1, c2, _, _ = inter(sb, sb, g, k, rounds=6)
            nn, nf = n2 / n1, c2 / c1
            ok = abs(nn - 1) <= 0.02 and abs(nf - 1) <= 0.02
            isp = max(spread(sn), spread(na), spread(nb))
            flag = "" if ok else "  VOID"
            if ok and abs(tf / tn - 1.0) <= isp:
                flag += "  NOISE>EFFECT"
            out("%-6s%-22s%13.1f%13.1f%8.3fx%13.1f%8.3f%9.3f%8.1f%%%s"
                % (tag, label, tn, tf, tf / tn, tf - tn, nn, nf, 100 * isp, flag))

out("")
out("H2a  BASELINE malloc settings")
sweep("base")

# M_MMAP_THRESHOLD = -3, M_TRIM_THRESHOLD = -1 (glibc malloc.h). Raising both makes a 4 MiB
# block come from the heap and never be returned to the kernel, so no mapping is ever fresh.
libc = ctypes.CDLL("libc.so.6", use_errno=True)
r1 = libc.mallopt(-3, 512 * 1024 * 1024)
r2 = libc.mallopt(-1, 512 * 1024 * 1024)
out("")
out("mallopt(M_MMAP_THRESHOLD=512MiB) -> %d ; mallopt(M_TRIM_THRESHOLD=512MiB) -> %d "
    "(1 = applied)" % (r1, r2))
out("H2b  AFTER raising the mmap/trim thresholds - if the loss vanishes, mmap churn IS the "
    "mechanism")
sweep("nommap")
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
