//! THREAD-COUNT probe for the sqrt composition loss (`deadlock-audit-iwt3o`).
//!
//! That bead CHARACTERISED a regime and REFUTED both of its pre-registered mechanisms:
//!     compose 2^21  `fnp.sqrt(np.abs(a))`   1.399x / 1.388x   a LOSS, clean nulls, replicated
//!     settled 2^20  `fnp.sqrt(pre_np_abs)`  0.436x / 0.404x   a 2.3-2.5x WIN, same kernel
//! H1 (page-fault storm) died on ru_minflt = 0.0-0.4 per call for EVERY arm; H2 (glibc mmap
//! churn) died on mallopt(M_MMAP_THRESHOLD/M_TRIM_THRESHOLD = 512 MiB) moving 1.399x to 1.388x.
//! Its retry predicate names this experiment:
//!
//!     "a probe that varies rayon thread count while holding n fixed - if the loss shrinks as
//!      threads fall while the settled win holds, chunk-count is the lever and a minimum-chunk-
//!      size gate is the fix."
//!
//! WHY THREAD COUNT IS THE RIGHT DIAL. The kernel is
//!     let chunk = n.div_ceil(rayon::current_num_threads());
//!     out.par_chunks_mut(chunk).zip(in.par_chunks(chunk)).for_each(run)
//! so CHUNK COUNT IS EXACTLY THREAD COUNT, and `current_num_threads()` reports the pool we are
//! INSTALLED IN. Building explicit `rayon::ThreadPool`s and running the timed statements inside
//! `pool.install(..)` therefore sweeps thread count AND chunk width together, in ONE process,
//! against ONE numpy, with no environment variable anywhere - `RAYON_NUM_THREADS` would need a
//! fresh invocation per point and a cross-invocation ratio comparison is not admissible here.
//!
//! THE REMAINING HYPOTHESIS this decides (producer-consumer cache locality): np.abs writes the
//! 16 MiB input from ONE core, leaving those lines singly-owned and dirty; our 64-way sqrt then
//! drags them across the coherence fabric, while numpy's sqrt reads them from the core that
//! wrote them. If that is the mechanism, the loss must SHRINK MONOTONICALLY as T falls, and at
//! T=1 - where our kernel takes the serial branch and reads from the same core that just ran
//! np.abs - it must be GONE. If the loss survives T=1, parallelism is exonerated outright and
//! the coherence story is dead too, which is worth as much as confirming it.
//!
//! THIRD CELL, THE DISCRIMINATOR. `fnp.abs` is itself parallel at n >= UNARY_PARALLEL_MIN =
//! 2^21, so `fnp.sqrt(fnp.abs(a))` hands the sqrt an input written by the SAME 64 threads that
//! are about to read it - each thread reading back roughly the lines it wrote. Under the
//! coherence hypothesis that composition must NOT lose, at the same n, same T and same fresh
//! allocation as the losing cell. Nothing but producer LOCALITY differs between cell 1 and
//! cell 3, so they separate "the input was freshly allocated" from "the input was written by
//! the wrong core" - two explanations wall-clock alone has never told apart here.
//!
//! PROTOCOL. Every cell is interleaved AB/BA with a dual A/A null (both arms must land in
//! 0.97-1.03) and an incumbent-spread gate; absolute per-arm ns are carried so a contaminated
//! incumbent is visible (a dual null alone is blind to it). The whole T sweep runs TWICE in
//! OPPOSITE ORDERS inside the one invocation, so a monotone host drift cannot masquerade as a
//! monotone thread-count trend: only a sign that replicates across both passes is reported.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::ffi::CString;

use fnp_python::fnp_python;

const SETUP: &str = r#"
import hashlib, os, statistics, sys, timeit
def out(*a):
    print(*a, file=sys.stderr, flush=True)
import numpy as np
import fnp_python as fnp

out("python", sys.version.split()[0], "| numpy", np.__version__)
out("in-process ELF sha256", hashlib.sha256(open(EXE_PATH, "rb").read()).hexdigest())
out("host", os.uname().nodename, "| loadavg", [round(x, 2) for x in os.getloadavg()])
out("cpus", os.cpu_count(), "| RAYON_NUM_THREADS env", os.environ.get("RAYON_NUM_THREADS", "(unset)"))
out("NOTE thread count is set by an installed rayon ThreadPool, not by the environment.")
rng = np.random.default_rng(SEED)
BASE = np.abs(rng.standard_normal(1 << 22)) + 0.5

def inter(sa, sb, g, k, rounds=4):
    ta, tb = [], []
    for r in range(rounds):
        for w in (("a","b","b","a") if r % 2 == 0 else ("b","a","a","b")):
            t = timeit.timeit(sa if w == "a" else sb, globals=g, number=k) / k * 1e9
            (ta if w == "a" else tb).append(t)
    return statistics.median(ta), statistics.median(tb), ta, tb

def spread(s):
    return (max(s) - min(s)) / statistics.median(s)

CELLS = (
    # label            numpy arm                  fnp arm                   what the input is
    ("np-written",  "np.sqrt(np.abs(a))",  "fnp.sqrt(np.abs(a))"),
    ("settled",     "np.sqrt(pre)",        "fnp.sqrt(pre)"),
    ("fnp-written", "np.sqrt(fnp.abs(a))", "fnp.sqrt(fnp.abs(a))"),
)

out("")
out("%-5s%-6s%-13s%-7s%10s%12s%12s%9s%12s%8s%9s%9s%s"
    % ("pass", "T", "cell", "n", "chunkKiB", "numpy_ns", "fnp_ns", "ratio", "excess_ns",
       "nullNP", "nullFNP", "incSprd", "  flag"))
"#;

const MEASURE: &str = r#"
sizes = [1 << 21] + ([1 << 22] if T in WIDE_T else [])
for n in sizes:
    a = BASE[:n].copy()
    pre = np.abs(a)
    g = {"np": np, "fnp": fnp, "a": a, "pre": pre}
    k = int(max(3, min(2000, 4e7 // n)))
    for label, sa, sb in CELLS:
        tn, tf, sn, _ = inter(sa, sb, g, k)
        n1, n2, na, nb = inter(sa, sa, g, k)
        c1, c2, _, _ = inter(sb, sb, g, k)
        nn, nf = n2 / n1, c2 / c1
        ok = abs(nn - 1) <= 0.02 and abs(nf - 1) <= 0.02
        isp = max(spread(sn), spread(na), spread(nb))
        flag = "" if ok else "  VOID"
        if ok and abs(tf / tn - 1.0) <= isp:
            flag += "  NOISE>EFFECT"
        # `chunk = n.div_ceil(current_num_threads())` in the kernel, so this column is the
        # actual per-thread working set - the quantity a minimum-chunk-size gate would read.
        out("%-5d%-6d%-13s%-7s%10.0f%12.1f%12.1f%8.3fx%12.1f%8.3f%9.3f%8.1f%%%s"
            % (PASS, T, label, "2^%d" % (n.bit_length() - 1), -(-n // T) * 8 / 1024,
               tn, tf, tf / tn, tf - tn, nn, nf, 100 * isp, flag))
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

    let cores = std::thread::available_parallelism()
        .map(|c| c.get())
        .unwrap_or(8);
    // Geometric ladder down from the machine's own width to the serial branch. T=1 is not a
    // decoration: `n >= SQRT_PARALLEL_MIN && current_num_threads() >= 2` is false there, so it
    // measures the SERIAL kernel reading a buffer the previous call just wrote from this very
    // core - the one configuration in which the coherence hypothesis predicts no loss at all.
    let mut threads: Vec<usize> = Vec::new();
    let mut t = cores;
    while t > 1 {
        threads.push(t);
        t /= 2;
    }
    threads.push(1);
    // The 2^22 rows cost as much as the 2^21 rows, so the second size runs only at the ends and
    // the middle of the ladder - enough to replicate a trend without doubling the whole sweep.
    let wide: Vec<usize> = vec![cores, 4, 1];

    let globals: Py<PyDict> = Python::attach(|py| -> PyResult<Py<PyDict>> {
        let g = PyDict::new(py);
        g.set_item("EXE_PATH", exe.to_string_lossy().as_ref())?;
        g.set_item("SEED", seed)?;
        g.set_item("WIDE_T", wide.clone())?;
        py.run(&CString::new(SETUP).unwrap(), Some(&g), None)?;
        Ok(g.unbind())
    })?;

    let pools: Vec<(usize, rayon::ThreadPool)> = threads
        .iter()
        .map(|&t| {
            let p = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .expect("rayon pool");
            (t, p)
        })
        .collect();

    let measure = CString::new(MEASURE).unwrap();
    for pass in 0..2 {
        // Pass 1 walks the ladder DOWN, pass 2 walks it UP. A host drifting monotonically in
        // time produces OPPOSITE apparent trends in the two passes; a real thread-count effect
        // produces the same one twice.
        let order: Vec<&(usize, rayon::ThreadPool)> = if pass == 0 {
            pools.iter().collect()
        } else {
            pools.iter().rev().collect()
        };
        for (t, pool) in order {
            let globals = &globals;
            let measure = &measure;
            pool.install(|| -> PyResult<()> {
                Python::attach(|py| {
                    let g = globals.bind(py);
                    g.set_item("T", *t)?;
                    g.set_item("PASS", pass + 1)?;
                    py.run(measure, Some(g), None)
                })
            })?;
        }
    }
    Ok(())
}
