//! WHERE DOES THE NATIVE f64 CONCATENATE ACTUALLY START WINNING? (`deadlock-audit-66w2d`)
//!
//! After five entry-path levers, `np.concatenate([a, b])` on f64 is still a LOSS at both sizes the
//! decomposition probe measures: +783 ns / 1.611x at n=64 and +1144 ns / 1.129x at n=16384. But the
//! wide board has it as a decisive WIN at 2^20 (0.362x on the narrow board's own cell). So the
//! native route has a crossover and no gate marks it - the route engages at EVERY size and loses
//! below that point.
//!
//! Fixing that needs the crossover measured, not guessed: this sweeps two-array f64 concatenate
//! from 2^3 to 2^21 and prints the ratio and the absolute excess at each size, under the same
//! interleaved-with-nulls contract as every other board here. The floor that comes out of it is a
//! FITTED constant, so it also needs the size at which the sign is stable across seeds - hence two
//! seeds in one invocation rather than one.

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

def inter(sa, sb, g, k, rounds=4):
    ta, tb = [], []
    for r in range(rounds):
        for w in (("a","b","b","a") if r % 2 == 0 else ("b","a","a","b")):
            t = timeit.timeit(sa if w == "a" else sb, globals=g, number=k) / k * 1e9
            (ta if w == "a" else tb).append(t)
    return statistics.median(ta), statistics.median(tb), ta, tb

def spread(s):
    return (max(s) - min(s)) / statistics.median(s)

def K(n):
    return int(max(3, min(8000, 4e7 // max(n, 1))))

# THE FLOOR HAS TO SIT ABOVE BOTH HELPERS, SO IT CANNOT BE FITTED ON f64 ALONE.
# try_zerocopy_f64_concatenate is not the only native route: try_zerocopy_bytes_concatenate
# accepts kind 'f' at itemsize 8 too, so a floor placed inside the f64 helper would merely
# REROUTE a small f64 concat into the bytes helper rather than delegate it. The floor belongs in
# `concatenate` itself, ahead of both - and a gate placed there is paid by EVERY dtype the bytes
# helper serves (kinds b/i/u/f/c at itemsize 1/2/4/8), so every one of them has to be measured.
# This repo's own rule: a dtype-shaped defect does not license a dtype-shaped fix, and dtype
# behaviour here has diverged by 460x before.
#
# THE NARROW DTYPES ARE THE ONES A BYTES FLOOR COULD ROB. A 1-byte dtype needs n = 2^22 per input
# to reach an 8 MiB output, so a floor stated in BYTES delegates every realistic bool/int8 concat.
# If any of them WINS below that, the floor destroys a live win - and this repo has banked
# narrow-int and bool wins elsewhere. So they are measured out to 2^22 rather than assumed.
WIDE = (6, 12, 16, 17, 18, 19, 20)
NARROW = (6, 16, 20, 22)
MAKERS = [
    ("f64",  WIDE,   lambda rng, n: rng.standard_normal(n)),
    ("f32",  WIDE,   lambda rng, n: rng.standard_normal(n).astype(np.float32)),
    ("i64",  WIDE,   lambda rng, n: rng.integers(0, 1 << 40, n)),
    ("c128", WIDE,   lambda rng, n: rng.standard_normal(n) + 1j * rng.standard_normal(n)),
    ("c64",  NARROW, lambda rng, n: (rng.standard_normal(n)
                                     + 1j * rng.standard_normal(n)).astype(np.complex64)),
    ("f16",  NARROW, lambda rng, n: rng.standard_normal(n).astype(np.float16)),
    ("i32",  NARROW, lambda rng, n: rng.integers(0, 1 << 20, n).astype(np.int32)),
    ("i16",  NARROW, lambda rng, n: rng.integers(0, 1 << 10, n).astype(np.int16)),
    ("i8",   NARROW, lambda rng, n: rng.integers(0, 100, n).astype(np.int8)),
    ("u8",   NARROW, lambda rng, n: rng.integers(0, 250, n).astype(np.uint8)),
    ("bool", NARROW, lambda rng, n: rng.integers(0, 2, n).astype(bool)),
]
out("")
out("%-6s%-6s%-8s%10s%13s%13s%9s%13s%8s%9s%9s"
    % ("seed", "dtype", "n", "out_MiB", "numpy_ns", "fnp_ns", "ratio", "excess_ns", "nullNP",
       "nullFNP", "incSprd"))
for seed in (SEED, SEED + 222):
  rng = np.random.default_rng(seed)
  for dname, sizes, make in MAKERS:
    for lg in sizes:
        n = 1 << lg
        a = make(rng, n)
        b = make(rng, n)
        out_mib = 2 * n * a.dtype.itemsize / (1 << 20)
        g = {"np": np, "fnp": fnp, "a": a, "b": b}
        sa, sb = "np.concatenate([a, b])", "fnp.concatenate([a, b])"
        wv, gv = np.asarray(eval(sa, g)), np.asarray(eval(sb, g))
        agree = wv.shape == gv.shape and wv.dtype == gv.dtype and wv.tobytes() == gv.tobytes()
        k = K(2 * n)
        tn, tf, sn, _ = inter(sa, sb, g, k)
        n1, n2, na, nb = inter(sa, sa, g, k)
        c1, c2, _, _ = inter(sb, sb, g, k)
        nn, nf = n2 / n1, c2 / c1
        ok = abs(nn - 1) <= 0.02 and abs(nf - 1) <= 0.02
        isp = max(spread(sn), spread(na), spread(nb))
        flag = "" if ok else "  VOID"
        if not agree:
            flag += "  BYTES DIFFER"
        if ok and abs(tf / tn - 1.0) <= isp:
            flag += "  NOISE>EFFECT"
        out("%-6d%-6s%-8s%10.2f%13.1f%13.1f%8.3fx%13.1f%8.3f%9.3f%8.1f%%%s"
            % (seed, dname, "2^%d" % lg, out_mib, tn, tf, tf / tn, tf - tn, nn, nf,
               100 * isp, flag))
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
