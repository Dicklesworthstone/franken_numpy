//! Shared helpers for the split per-domain criterion bench binaries.
//!
//! Each per-domain bench binary pulls this in with
//! `#[path = "common/mod.rs"] mod common;`. The helpers are extracted verbatim
//! from the former monolithic `criterion_python_surface.rs` so that each
//! per-domain binary compiles only its own bench functions instead of forcing
//! the whole 200-plus-function monolith to compile just to run one group.
//!
//! `benches/common/mod.rs` is not itself a bench target: Cargo auto-discovers
//! `benches/*.rs` and `benches/*/main.rs`, and this is neither.

use criterion::Criterion;
use pyo3::{PyResult, Python};
use std::cell::RefCell;

/// Import numpy on the interpreter, mapping the module handle away; every bench
/// group calls this before allocating its inputs so a missing numpy fails loud.
pub fn ensure_numpy_available(py: Python<'_>) -> PyResult<()> {
    py.import("numpy").map(drop)
}

/// Mean and coefficient-of-variation (percent) over the last <=10 retained
/// paired samples. Panics below two samples, which Criterion always retains.
pub fn ledger_tail_stats(samples: &RefCell<Vec<f64>>) -> (usize, f64, f64) {
    let samples = samples.borrow();
    let count = samples.len().min(10);
    assert!(
        count >= 2,
        "Criterion must retain at least two paired samples"
    );
    let tail = &samples[samples.len() - count..];
    let mean = tail.iter().sum::<f64>() / count as f64;
    let variance = tail
        .iter()
        .map(|sample| {
            let delta = sample - mean;
            delta * delta
        })
        .sum::<f64>()
        / (count - 1) as f64;
    (count, mean, variance.sqrt() * 100.0 / mean)
}

/// Emit the paired `LEDGER_AUDIT` line the negative-evidence flow parses:
/// candidate/orig means (ms), CVs (%), and the orig/candidate ratio.
pub fn report_ledger_pair(
    row: &str,
    candidate_samples: &RefCell<Vec<f64>>,
    orig_samples: &RefCell<Vec<f64>>,
) {
    if candidate_samples.borrow().is_empty() && orig_samples.borrow().is_empty() {
        return;
    }
    let (candidate_n, candidate_ns, candidate_cv) = ledger_tail_stats(candidate_samples);
    let (orig_n, orig_ns, orig_cv) = ledger_tail_stats(orig_samples);
    assert_eq!(candidate_n, orig_n);
    println!(
        "LEDGER_AUDIT row={row} samples={candidate_n} candidate_mean_ms={:.6} \
         candidate_cv_pct={candidate_cv:.3} orig_mean_ms={:.6} orig_cv_pct={orig_cv:.3} \
         orig_over_candidate={:.4}",
        candidate_ns / 1_000_000.0,
        orig_ns / 1_000_000.0,
        orig_ns / candidate_ns,
    );
}

/// `FNP_BENCH_GROUPS`, when set to a comma-separated substring list, restricts a
/// run to the group functions whose names contain one of the tokens; unset
/// preserves run-everything behavior. Identical semantics to the former
/// monolith gate, so existing reproduction commands keep working.
pub fn group_enabled(group_fn_name: &str) -> bool {
    let Ok(spec) = std::env::var("FNP_BENCH_GROUPS") else {
        return true;
    };
    spec.split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .any(|token| group_fn_name.contains(token))
}

/// A named bench group: the group function's name (for `FNP_BENCH_GROUPS`
/// gating) paired with the function itself.
pub type BenchGroup = (&'static str, fn(&mut Criterion));

/// Drive the selected bench group functions under one `Criterion`, then emit the
/// final summary. Mirrors the former `gated_benches!` macro's `main`: each entry
/// is `(group_fn_name, group_fn)`, gated by [`group_enabled`].
pub fn gated_main(targets: &[BenchGroup]) {
    let mut criterion = Criterion::default().configure_from_args();
    for (name, target) in targets {
        if group_enabled(name) {
            target(&mut criterion);
        }
    }
    Criterion::default().configure_from_args().final_summary();
}
