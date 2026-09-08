//! Head-to-head re-verification for the 4 strided-vein / scalar-take rows:
//! - Row 3 (deadlock-audit-yphwc, L57373): `fnp.take(a, scalar)` at n=2^20
//! - Row 4 (deadlock-audit-0iwez, L58666): `isposinf`, `isneginf`, `logical_not` strided vs contiguous
//! - Row 5 (deadlock-audit-0iwez, L58731): `all`, `any`, `ediff1d`, `frexp`, `modf` strided vs contiguous
//! - Row 6 (deadlock-audit-0iwez, L58810): `sinc`, `append` strided vs contiguous
//!
//! Run with:
//! `PYO3_PYTHON=/data/projects/franken_numpy/.venv-numpy314/bin/python RCH_CARGO_WRAPPER_BYPASS=1 CARGO_TARGET_DIR=/data/projects/franken_numpy/target cargo run --release -p fnp-python --example h2h_strided_vein`

use pyo3::prelude::*;
use std::ffi::CString;

use fnp_python::fnp_python;

const HARNESS: &str = r#"
import hashlib, os, statistics, sys, timeit
def out(*a):
    print(*a, file=sys.stderr, flush=True)
import numpy as np
import fnp_python as fnp

NP_SO = np._core._multiarray_umath.__file__
NP_SHA = hashlib.sha256(open(NP_SO, "rb").read()).hexdigest()
EXE_SHA = hashlib.sha256(open(EXE_PATH, "rb").read()).hexdigest()
INVOCATION = "%s-%d-%d" % (os.uname().nodename, os.getpid(), int(__import__("time").time()))

out("python", sys.version.split()[0], "| numpy", np.__version__)
out("bench_elf_sha256=%s" % EXE_SHA)
out("incumbent artifact_sha256=%s" % NP_SHA)
out("invocation_id=%s" % INVOCATION)
out("host", os.uname().nodename, "| loadavg", [round(x, 2) for x in os.getloadavg()])

assert np.__name__ == "numpy", np.__name__
assert np.take is not fnp.take
out("dispatch_assert=passed")

def inter(sa, sb, g, k, rounds=4):
    ta, tb = [], []
    for r in range(rounds):
        for w in (("a","b","b","a") if r % 2 == 0 else ("b","a","a","b")):
            t = timeit.timeit(sa if w == "a" else sb, globals=g, number=k) / k * 1e9
            (ta if w == "a" else tb).append(t)
    return statistics.median(ta), statistics.median(tb)

def measure(label, sa, sb, g, k):
    # Verify parity
    va = eval(sa, g)
    vb = eval(sb, g)
    if isinstance(va, tuple):
        for x, y in zip(va, vb):
            assert np.allclose(x, y, equal_nan=True), f"Mismatch in {label}"
    else:
        assert np.allclose(va, vb, equal_nan=True), f"Mismatch in {label}"

    tn, tf = inter(sa, sb, g, k)
    n1, n2 = inter(sa, sa, g, k)
    c1, c2 = inter(sb, sb, g, k)
    nn = n2 / n1
    nf = c2 / c1
    ratio = tf / tn
    speedup = tn / tf if tf > 0 else float('inf')
    out("%-36s numpy=%10.1f ns  fnp=%10.1f ns  ratio=%7.3fx  speedup=%7.3fx  nullNP=%6.3f  nullFNP=%6.3f" % (
        label, tn, tf, ratio, speedup, nn, nf))
    return tn, tf, ratio, speedup, nn, nf

rng = np.random.default_rng(SEED)

out("\n--- GROUP 1: Row 3 (yphwc) - take with scalar index at n=2^20 ---")
a_huge = rng.standard_normal(1 << 20)
g_take = {"np": np, "fnp": fnp, "a": a_huge}
measure("take(a, 2) n=2^20", "np.take(a, 2)", "fnp.take(a, 2)", g_take, 1000)
del a_huge, g_take

out("\n--- GROUP 2: Row 4 (0iwez) - isposinf, isneginf, logical_not ---")
f64_base = rng.standard_normal(1 << 16)
# inject inf, -inf
f64_base[0] = float('inf')
f64_base[1] = float('-inf')
s4 = f64_base[::2]
c4 = np.ascontiguousarray(s4)
g4_strd = {"np": np, "fnp": fnp, "a": s4}
g4_cont = {"np": np, "fnp": fnp, "a": c4}

measure("isneginf strided", "np.isneginf(a)", "fnp.isneginf(a)", g4_strd, 100)
measure("isneginf contiguous", "np.isneginf(a)", "fnp.isneginf(a)", g4_cont, 100)
measure("isposinf strided", "np.isposinf(a)", "fnp.isposinf(a)", g4_strd, 100)
measure("isposinf contiguous", "np.isposinf(a)", "fnp.isposinf(a)", g4_cont, 100)
measure("logical_not strided", "np.logical_not(a)", "fnp.logical_not(a)", g4_strd, 100)
measure("logical_not contiguous", "np.logical_not(a)", "fnp.logical_not(a)", g4_cont, 100)

out("\n--- GROUP 3: Row 5 (0iwez) - all, any, ediff1d, frexp, modf ---")
b_base = rng.integers(0, 2, 1 << 16).astype(bool)
b_strd = b_base[::2]
b_cont = np.ascontiguousarray(b_strd)
gb_strd = {"np": np, "fnp": fnp, "a": b_strd}
gb_cont = {"np": np, "fnp": fnp, "a": b_cont}

measure("all strided", "np.all(a)", "fnp.all(a)", gb_strd, 200)
measure("all contiguous", "np.all(a)", "fnp.all(a)", gb_cont, 200)
measure("any strided", "np.any(a)", "fnp.any(a)", gb_strd, 200)
measure("any contiguous", "np.any(a)", "fnp.any(a)", gb_cont, 200)
measure("ediff1d strided", "np.ediff1d(a)", "fnp.ediff1d(a)", g4_strd, 100)
measure("ediff1d contiguous", "np.ediff1d(a)", "fnp.ediff1d(a)", g4_cont, 100)
measure("frexp strided", "np.frexp(a)", "fnp.frexp(a)", g4_strd, 100)
measure("frexp contiguous", "np.frexp(a)", "fnp.frexp(a)", g4_cont, 100)
measure("modf strided", "np.modf(a)", "fnp.modf(a)", g4_strd, 100)
measure("modf contiguous", "np.modf(a)", "fnp.modf(a)", g4_cont, 100)

out("\n--- GROUP 4: Row 6 (0iwez) - sinc, append ---")
measure("sinc strided", "np.sinc(a)", "fnp.sinc(a)", g4_strd, 100)
measure("sinc contiguous", "np.sinc(a)", "fnp.sinc(a)", g4_cont, 100)
measure("append strided", "np.append(a, a)", "fnp.append(a, a)", g4_strd, 100)
measure("append contiguous", "np.append(a, a)", "fnp.append(a, a)", g4_cont, 100)
out("\nAll strided-vein re-measurements completed successfully.")
"#;

fn main() -> PyResult<()> {
    pyo3::append_to_inittab!(fnp_python);
    Python::initialize();
    let exe = std::env::current_exe()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyOSError, _>(e.to_string()))?;
    let seed: i64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    Python::attach(|py| {
        let globals = pyo3::types::PyDict::new(py);
        globals.set_item("EXE_PATH", exe.to_string_lossy().as_ref())?;
        globals.set_item("SEED", seed)?;
        py.run(&CString::new(HARNESS).unwrap(), Some(&globals), None)
    })
}
