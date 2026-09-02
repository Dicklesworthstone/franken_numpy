//! Counted price for the `(*args, **kwargs)` ENTRY SHAPE
//! (`deadlock-audit-defaulted-argument-three-state-parse`).
//!
//! That bead blocks ~40 argument-parity cells behind one question: what does it cost to stop
//! declaring typed parameters and read the call verbatim instead? A typed signature cannot
//! tell an omitted argument from an explicitly passed `None`, and numpy treats those
//! differently, so the fix is to take the call as it was written - but the bead rightly
//! refuses to apply that to forty entry points on an unmeasured assumption.
//!
//! IT MUST BE COUNTED, NOT TIMED. The effect is a handful of nanoseconds on a call with a
//! ~2000 ns floor, and this fleet could not decide a 20-50% effect on a 227 us call across
//! five runs - the sign test flipped ([[sign-test-decides-what-the-ratio-cannot]]).
//! Instructions retired do not move with host load, so a counter diff stays valid on a shared
//! worker where no wall-clock contract could run.
//!
//! HOW THE COMPARISON IS MADE: this harness does NOT compare two functions. It compares the
//! SAME function across two source revisions, because `convolve`'s body was left byte-for-byte
//! identical when its signature changed - only the entry shape differs. Run it at HEAD, then
//! run it again against the commit before the conversion with this example overlaid:
//!
//!     rch exec --job -- cargo run -j2 --release -p fnp-python --example count_entry_shape_insns
//!     rch exec --base <pre-conversion sha> --clean-overlay \
//!         --overlay-path crates/fnp-python/examples -- \
//!         cargo run -j2 --release -p fnp-python --example count_entry_shape_insns
//!
//! The difference between the two netted per-call counts IS the entry-shape price.
//!
//! OPERANDS ARE DELIBERATELY TINY. A 4x3 convolve spends almost nothing in the kernel, so the
//! per-call count is dominated by argument handling and dispatch - which is the thing being
//! priced. A large operand would bury it.

use pyo3::prelude::*;
use std::ffi::CString;
use std::process::Command;

use fnp_python::fnp_python;

const ARM: &str = r#"
import sys
import numpy as np
import fnp_python as fnp

# Shared by every arm, so it nets out of the diff.
a = np.array([1.0, 2.0, 3.0, 4.0])
v = np.array([0.5, 0.25, 0.125])

if ARM == "numpy":
    for _ in range(N_CALLS):
        np.convolve(a, v)
elif ARM == "fnp":
    for _ in range(N_CALLS):
        fnp.convolve(a, v)
elif ARM == "fnp_mode":
    # The three-argument form, which is what the typed signature used to parse and the
    # verbatim reader now handles.
    for _ in range(N_CALLS):
        fnp.convolve(a, v, "full")
# ARM == "none": no calls at all.
"#;

fn run_arm(arm: &str, n_calls: usize) -> PyResult<()> {
    pyo3::append_to_inittab!(fnp_python);
    Python::initialize();
    Python::attach(|py| {
        let globals = pyo3::types::PyDict::new(py);
        globals.set_item("ARM", arm)?;
        globals.set_item("N_CALLS", n_calls)?;
        py.run(&CString::new(ARM).unwrap(), Some(&globals), None)
    })
}

/// One `perf stat -e instructions` run of `self --arm <arm>`; returns instructions retired.
fn count(exe: &str, arm: &str, n_calls: usize) -> Option<u64> {
    let out = Command::new("perf")
        .args(["stat", "-x,", "-e", "instructions", exe, "--arm", arm])
        .arg("--calls")
        .arg(n_calls.to_string())
        .output()
        .ok()?;
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
    let n_calls = args
        .iter()
        .position(|a| a == "--calls")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(20000);

    if let Some(i) = args.iter().position(|a| a == "--arm") {
        let arm = args.get(i + 1).map(String::as_str).unwrap_or("none");
        return run_arm(arm, n_calls);
    }

    let exe = std::env::current_exe()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyOSError, _>(e.to_string()))?;
    let exe = exe.to_string_lossy().to_string();
    eprintln!("entry-shape price: convolve(4x3 f64), {n_calls} calls/arm, 3 reps");
    eprintln!(
        "host {}",
        std::fs::read_to_string("/proc/sys/kernel/hostname")
            .map(|s| s.trim().to_string())
            .unwrap_or_default()
    );
    if count(&exe, "none", 1).is_none() {
        eprintln!("PERF UNAVAILABLE on this worker - `perf stat -e instructions` produced no");
        eprintln!("countable line. This harness measures nothing without it, and it does NOT");
        eprintln!("fall back to wall clock: a few ns/call is exactly what wall clock cannot see.");
        return Ok(());
    }

    let arms = ["none", "numpy", "fnp", "fnp_mode"];
    let mut totals = [[0u64; 3]; 4];
    for (index, arm) in arms.iter().enumerate() {
        for (rep, slot) in totals[index].iter_mut().enumerate() {
            match count(&exe, arm, n_calls) {
                Some(value) => *slot = value,
                None => {
                    eprintln!("arm {arm} rep {rep} produced no count; aborting");
                    return Ok(());
                }
            }
        }
    }
    let mean = |values: [u64; 3]| values.iter().sum::<u64>() as f64 / 3.0;
    let spread = |values: [u64; 3]| {
        let (low, high) = (
            *values.iter().min().unwrap() as f64,
            *values.iter().max().unwrap() as f64,
        );
        (high - low) / mean(values) * 100.0
    };
    let baseline = mean(totals[0]);
    eprintln!();
    eprintln!(
        "{:<10}{:>18}{:>16}{:>10}",
        "arm", "insns", "per call", "spread"
    );
    for (index, arm) in arms.iter().enumerate() {
        let netted = mean(totals[index]) - baseline;
        eprintln!(
            "{:<10}{:>18.0}{:>16.1}{:>9.2}%",
            arm,
            mean(totals[index]),
            netted / n_calls as f64,
            spread(totals[index])
        );
    }
    eprintln!();
    eprintln!("The `fnp` per-call figure is the number to carry across revisions: run this at");
    eprintln!("HEAD and at the pre-conversion commit, and the DIFFERENCE is the entry-shape");
    eprintln!("price. A spread above a few percent means the counter itself is unstable here");
    eprintln!("and the run should be repeated before the number is quoted.");
    Ok(())
}
