//! Counted attribution for `np.take`'s remaining serial loss, by INSTRUCTION DIFF.
//!
//! `deadlock-audit-ddoeq` left one question: below the `1<<18` parallel gate, `fnp.take` is
//! 1.15-1.48x slower than NumPy and nothing measured so far explains it. Removing the duplicate
//! per-element bounds check moved the ratio by at most 4.8% — a sub-5% effect that this repo's own
//! rule says a single dual-null run cannot decide at all. So this deliberately does NOT time
//! anything.
//!
//! Instructions retired do not move with host load, so a counter diff stays valid on a shared
//! worker where a wall-clock contract could not run. This is the `counted-attribution-by-
//! instruction-diff` instrument, which resolved `add.accumulate`'s residual where CIs could not.
//!
//! THREE ARMS, differing ONLY in which callable the loop invokes. The `none` arm does interpreter
//! startup, the numpy import, the corpus build and the engagement probe but makes NO take calls,
//! so subtracting it removes everything the arms share. Run as a driver with no argument and it
//! spawns `perf stat -e instructions` on itself once per arm per rep and reports the netted
//! per-call counts; run with `--arm <none|numpy|fnp>` and it IS one arm.
//!
//! `RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- \
//!    cargo run -j2 --release -p fnp-python --example count_take_insns`

use pyo3::prelude::*;
use std::ffi::CString;
use std::process::Command;

use fnp_python::fnp_python;

const ARM: &str = r#"
import os, sys
import numpy as np
import fnp_python as fnp

# Everything below the call loop is SHARED by all three arms, so it nets out of the diff.
rng = np.random.default_rng(20260831)
src = rng.standard_normal(1 << 22)
idx = rng.integers(0, 1 << 22, N_IDX)

# Engagement probe, in every arm so its cost nets out too: a delegating route would make this a
# numpy-vs-numpy comparison and the instruction diff would read ~0 for the wrong reason.
_real = np.take
_calls = []
def _spy(*a, **k):
    _calls.append(1)
    return _real(*a, **k)
np.take = _spy
fnp.take(src, idx)
np.take = _real
if _calls:
    print("ENGAGEMENT_FAIL fnp.take delegated %d time(s)" % len(_calls), file=sys.stderr)
    raise SystemExit(2)

if ARM == "numpy":
    for _ in range(N_CALLS):
        np.take(src, idx)
elif ARM == "fnp":
    for _ in range(N_CALLS):
        fnp.take(src, idx)
# ARM == "none": no calls at all.
"#;

fn run_arm(arm: &str, n_calls: usize, n_idx: usize) -> PyResult<()> {
    pyo3::append_to_inittab!(fnp_python);
    Python::initialize();
    Python::attach(|py| {
        let globals = pyo3::types::PyDict::new(py);
        globals.set_item("ARM", arm)?;
        globals.set_item("N_CALLS", n_calls)?;
        globals.set_item("N_IDX", n_idx)?;
        py.run(&CString::new(ARM).unwrap(), Some(&globals), None)
    })
}

/// One `perf stat -e instructions` run of `self --arm <arm>`; returns instructions retired.
fn count(exe: &str, arm: &str, n_calls: usize, n_idx: usize) -> Option<u64> {
    let out = Command::new("perf")
        .args(["stat", "-x,", "-e", "instructions", exe, "--arm", arm])
        .arg("--calls")
        .arg(n_calls.to_string())
        .arg("--idx")
        .arg(n_idx.to_string())
        .output()
        .ok()?;
    // `-x,` puts the count first on the instructions line of stderr.
    String::from_utf8_lossy(&out.stderr)
        .lines()
        .find_map(|line| {
            line.contains("instructions")
                .then(|| line.split(',').next()?.trim().replace('.', "").parse().ok())
                .flatten()
        })
}

fn main() -> PyResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str, default: usize| -> usize {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let n_calls = flag("--calls", 2000);
    let n_idx = flag("--idx", 1 << 14);

    if let Some(i) = args.iter().position(|a| a == "--arm") {
        let arm = args.get(i + 1).map(String::as_str).unwrap_or("none");
        return run_arm(arm, n_calls, n_idx);
    }

    // Driver.
    let exe = std::env::current_exe()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyOSError, _>(e.to_string()))?;
    let exe = exe.to_string_lossy().to_string();
    eprintln!("counted attribution: np.take, m={n_idx}, {n_calls} calls/arm, 3 reps");
    eprintln!(
        "host {}",
        std::fs::read_to_string("/proc/sys/kernel/hostname")
            .map(|s| s.trim().to_string())
            .unwrap_or_default()
    );
    if count(&exe, "none", 1, 1024).is_none() {
        eprintln!("PERF UNAVAILABLE on this worker - `perf stat -e instructions` produced no");
        eprintln!("countable line. This harness measures nothing without it; not falling back to");
        eprintln!("wall clock, because a sub-5% effect is exactly what wall clock cannot decide.");
        return Ok(());
    }
    let mut totals = [[0u64; 3]; 3]; // [arm][rep]
    for (a, arm) in ["none", "numpy", "fnp"].iter().enumerate() {
        for rep in 0..3 {
            match count(&exe, arm, n_calls, n_idx) {
                Some(v) => totals[a][rep] = v,
                None => {
                    eprintln!("arm {arm} rep {rep} produced no count; aborting");
                    return Ok(());
                }
            }
        }
    }
    let mean = |v: [u64; 3]| v.iter().sum::<u64>() as f64 / 3.0;
    let spread = |v: [u64; 3]| {
        let (lo, hi) = (
            *v.iter().min().unwrap() as f64,
            *v.iter().max().unwrap() as f64,
        );
        (hi - lo) / hi * 100.0
    };
    let (none, np, fnp) = (mean(totals[0]), mean(totals[1]), mean(totals[2]));
    eprintln!("  none   {none:>16.0}  spread {:.2}%", spread(totals[0]));
    eprintln!("  numpy  {np:>16.0}  spread {:.2}%", spread(totals[1]));
    eprintln!("  fnp    {fnp:>16.0}  spread {:.2}%", spread(totals[2]));
    let per = |t: f64| (t - none) / n_calls as f64;
    let (np_per, fnp_per) = (per(np), per(fnp));
    eprintln!("  netted per call:  numpy {np_per:.1}  fnp {fnp_per:.1}");
    eprintln!(
        "  EXCESS {:+.1} insns/call = {:.3}x   ({:.2} insns per gathered element)",
        fnp_per - np_per,
        fnp_per / np_per,
        (fnp_per - np_per) / n_idx as f64
    );
    Ok(())
}
