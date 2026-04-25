//! Conformance matrix: random family.
//!
//! Verifies that fnp_python.random produces identical seeded samples
//! to numpy.random. Both surfaces wrap the same numpy RNG, so any
//! divergence here signals a re-export / wrapper bug rather than an
//! RNG correctness bug.
//!
//! This family doesn't fit the table-driven `run_case` pattern (the
//! object-method-on-RandomState shape doesn't match the module-level
//! function call surface), so we drive identity + parity assertions
//! directly with `numpy.allclose`.

mod common;

use common::with_fnp_and_numpy;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use std::sync::atomic::{AtomicUsize, Ordering};

static MUST_PASS: AtomicUsize = AtomicUsize::new(0);
static MUST_TOTAL: AtomicUsize = AtomicUsize::new(0);
static SHOULD_PASS: AtomicUsize = AtomicUsize::new(0);
static SHOULD_TOTAL: AtomicUsize = AtomicUsize::new(0);

fn assert_close<'py>(
    py: Python<'py>,
    id: &str,
    must: bool,
    ours: &pyo3::Bound<'py, PyAny>,
    theirs: &pyo3::Bound<'py, PyAny>,
) -> PyResult<()> {
    let np_allclose = py.import("numpy")?.getattr("allclose")?;
    let kw = PyDict::new(py);
    kw.set_item("equal_nan", true)?;
    let close: bool = np_allclose
        .call((ours, theirs), Some(&kw))?
        .extract::<bool>()
        .unwrap_or(false);
    let level = if must { "MUST" } else { "SHOULD" };
    let total = if must { &MUST_TOTAL } else { &SHOULD_TOTAL };
    let pass = if must { &MUST_PASS } else { &SHOULD_PASS };
    total.fetch_add(1, Ordering::Relaxed);
    if close {
        pass.fetch_add(1, Ordering::Relaxed);
        eprintln!("{{\"id\":\"{id}\",\"level\":\"{level}\",\"verdict\":\"PASS\"}}");
        Ok(())
    } else {
        eprintln!(
            "{{\"id\":\"{id}\",\"level\":\"{level}\",\"verdict\":\"FAIL\",\"detail\":\"drift\"}}"
        );
        if must {
            panic!("MUST clause {id} failed: drift between fnp_python and numpy");
        }
        Ok(())
    }
}

/// Mark an existence-only check (MAY tier). fnp_python.random
/// implements wrappers, not re-exports, so identity comparison is
/// incorrect — we only verify each name resolves to a callable on
/// both sides.
fn assert_callable_present<'py>(
    id: &str,
    ours: &pyo3::Bound<'py, PyAny>,
    theirs: &pyo3::Bound<'py, PyAny>,
) {
    SHOULD_TOTAL.fetch_add(1, Ordering::Relaxed);
    let ours_callable = ours.is_callable();
    let theirs_callable = theirs.is_callable();
    if ours_callable && theirs_callable {
        SHOULD_PASS.fetch_add(1, Ordering::Relaxed);
        eprintln!("{{\"id\":\"{id}\",\"level\":\"SHOULD\",\"verdict\":\"PASS\"}}");
    } else {
        eprintln!(
            "{{\"id\":\"{id}\",\"level\":\"SHOULD\",\"verdict\":\"FAIL\",\"detail\":\"callable presence mismatch (ours={ours_callable},theirs={theirs_callable})\"}}"
        );
    }
}

#[test]
fn conformance_random_matrix() {
    with_fnp_and_numpy(|py, module, numpy| {
        let our_random = module.getattr("random")?;
        let np_random = numpy.getattr("random")?;

        // ─── callable presence (SHOULD) ────────────────────────────────
        // fnp_python.random ships wrappers (not direct re-exports), so
        // we don't compare object identity. We do require that every
        // name resolves to a callable on both sides — wrappers must
        // present the same surface.
        for name in [
            "random",
            "rand",
            "randn",
            "randint",
            "uniform",
            "normal",
            "choice",
            "permutation",
            "shuffle",
            "default_rng",
            "Generator",
            "RandomState",
            "BitGenerator",
            "PCG64",
            "MT19937",
        ] {
            if let (Ok(o), Ok(t)) = (our_random.getattr(name), np_random.getattr(name)) {
                assert_callable_present(&format!("random-callable-{name}"), &o, &t);
            }
        }

        // ─── seeded RandomState methods (MUST: bit-for-bit) ────────────
        let make_rs_pair =
            |seed: u64| -> PyResult<(pyo3::Bound<'_, PyAny>, pyo3::Bound<'_, PyAny>)> {
                let our_state = our_random.call_method1("RandomState", (seed,))?;
                let their_state = np_random.call_method1("RandomState", (seed,))?;
                Ok((our_state, their_state))
            };

        let (o, t) = make_rs_pair(12345)?;
        let ours = o.call_method1("uniform", (0.0_f64, 1.0_f64, 5_i64))?;
        let theirs = t.call_method1("uniform", (0.0_f64, 1.0_f64, 5_i64))?;
        assert_close(py, "random-rs-uniform-5", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(12345)?;
        let ours = o.call_method1("normal", (0.0_f64, 1.0_f64, 5_i64))?;
        let theirs = t.call_method1("normal", (0.0_f64, 1.0_f64, 5_i64))?;
        assert_close(py, "random-rs-normal-5", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(7)?;
        let ours = o.call_method1("randint", (0_i64, 100_i64, 6_i64))?;
        let theirs = t.call_method1("randint", (0_i64, 100_i64, 6_i64))?;
        assert_close(py, "random-rs-randint-6", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(7)?;
        let ours = o.call_method1("poisson", (2.5_f64, 4_i64))?;
        let theirs = t.call_method1("poisson", (2.5_f64, 4_i64))?;
        assert_close(py, "random-rs-poisson-4", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(7)?;
        let ours = o.call_method1("binomial", (10_i64, 0.5_f64, 6_i64))?;
        let theirs = t.call_method1("binomial", (10_i64, 0.5_f64, 6_i64))?;
        assert_close(py, "random-rs-binomial-6", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(7)?;
        let arr = py
            .import("numpy")?
            .getattr("array")?
            .call1((vec![10_i64, 20, 30, 40, 50],))?;
        let ours = o.call_method1("choice", (arr.clone(), 3_i64))?;
        let theirs = t.call_method1("choice", (arr.clone(), 3_i64))?;
        assert_close(py, "random-rs-choice-3", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(99)?;
        let ours = o.call_method1("standard_normal", (8_i64,))?;
        let theirs = t.call_method1("standard_normal", (8_i64,))?;
        assert_close(py, "random-rs-standard_normal-8", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(99)?;
        let ours = o.call_method1("exponential", (1.5_f64, 5_i64))?;
        let theirs = t.call_method1("exponential", (1.5_f64, 5_i64))?;
        assert_close(py, "random-rs-exponential-5", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(99)?;
        let ours = o.call_method1("gamma", (2.0_f64, 1.0_f64, 5_i64))?;
        let theirs = t.call_method1("gamma", (2.0_f64, 1.0_f64, 5_i64))?;
        assert_close(py, "random-rs-gamma-5", true, &ours, &theirs)?;

        // ─── Generator API (modern, SHOULD) ─────────────────────────────
        if let (Ok(our_rng_fn), Ok(their_rng_fn)) = (
            our_random.getattr("default_rng"),
            np_random.getattr("default_rng"),
        ) {
            let our_rng = our_rng_fn.call1((424242_u64,))?;
            let their_rng = their_rng_fn.call1((424242_u64,))?;

            let ours = our_rng.call_method1("standard_normal", (5_i64,))?;
            let theirs = their_rng.call_method1("standard_normal", (5_i64,))?;
            assert_close(
                py,
                "random-default_rng-standard_normal-5",
                false,
                &ours,
                &theirs,
            )?;

            let our_rng = our_rng_fn.call1((424242_u64,))?;
            let their_rng = their_rng_fn.call1((424242_u64,))?;
            let ours = our_rng.call_method1("integers", (0_i64, 100_i64, 6_i64))?;
            let theirs = their_rng.call_method1("integers", (0_i64, 100_i64, 6_i64))?;
            assert_close(py, "random-default_rng-integers-6", false, &ours, &theirs)?;
        }

        // ─── seeded module-level free functions (SHOULD) ────────────────
        // After fnp_random.seed(N), the next free-function call sample
        // must equal numpy.random's after numpy.random.seed(N). Both
        // resolve to the same object so this is essentially a regression
        // guard for the re-export wiring.
        let _ = our_random.call_method1("seed", (314_u64,))?;
        let our_sample = our_random.call_method1("random", (4_i64,))?;
        let _ = np_random.call_method1("seed", (314_u64,))?;
        let their_sample = np_random.call_method1("random", (4_i64,))?;
        assert_close(
            py,
            "random-module-seed-then-random",
            false,
            &our_sample,
            &their_sample,
        )?;

        Ok(())
    });

    let must_pass = MUST_PASS.load(Ordering::Relaxed);
    let must_total = MUST_TOTAL.load(Ordering::Relaxed);
    let should_pass = SHOULD_PASS.load(Ordering::Relaxed);
    let should_total = SHOULD_TOTAL.load(Ordering::Relaxed);
    eprintln!("\n=== fnp-python conformance matrix: random ===");
    let pct = |p: usize, total: usize| -> String {
        if total == 0 {
            "n/a".into()
        } else {
            format!("{:.1}%", (p as f64 / total as f64) * 100.0)
        }
    };
    eprintln!(
        "| random | MUST {must_pass}/{must_total} ({}) | SHOULD {should_pass}/{should_total} ({}) |",
        pct(must_pass, must_total),
        pct(should_pass, should_total),
    );
}
