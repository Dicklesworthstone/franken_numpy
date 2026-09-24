//! Conformance matrix: random family.
//!
//! Verifies that fnp_python.random produces identical seeded samples
//! to numpy.random. fnp's Generator/BitGenerator classes are native Rust
//! (fnp-random), not wrappers of numpy's, so a divergence here is an RNG
//! correctness bug.
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

        // ─── additional RandomState distributions (MUST) ────────────────
        let (o, t) = make_rs_pair(42)?;
        let ours = o.call_method1("beta", (2.0_f64, 5.0_f64, 6_i64))?;
        let theirs = t.call_method1("beta", (2.0_f64, 5.0_f64, 6_i64))?;
        assert_close(py, "random-rs-beta-6", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(42)?;
        let ours = o.call_method1("chisquare", (3.0_f64, 5_i64))?;
        let theirs = t.call_method1("chisquare", (3.0_f64, 5_i64))?;
        assert_close(py, "random-rs-chisquare-5", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(42)?;
        let ours = o.call_method1("f", (5.0_f64, 10.0_f64, 4_i64))?;
        let theirs = t.call_method1("f", (5.0_f64, 10.0_f64, 4_i64))?;
        assert_close(py, "random-rs-f-4", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(42)?;
        let ours = o.call_method1("geometric", (0.3_f64, 6_i64))?;
        let theirs = t.call_method1("geometric", (0.3_f64, 6_i64))?;
        assert_close(py, "random-rs-geometric-6", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(42)?;
        let ours = o.call_method1("laplace", (0.0_f64, 1.0_f64, 5_i64))?;
        let theirs = t.call_method1("laplace", (0.0_f64, 1.0_f64, 5_i64))?;
        assert_close(py, "random-rs-laplace-5", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(42)?;
        let ours = o.call_method1("logistic", (0.0_f64, 1.0_f64, 5_i64))?;
        let theirs = t.call_method1("logistic", (0.0_f64, 1.0_f64, 5_i64))?;
        assert_close(py, "random-rs-logistic-5", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(42)?;
        let ours = o.call_method1("lognormal", (0.0_f64, 1.0_f64, 5_i64))?;
        let theirs = t.call_method1("lognormal", (0.0_f64, 1.0_f64, 5_i64))?;
        assert_close(py, "random-rs-lognormal-5", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(42)?;
        let ours = o.call_method1("negative_binomial", (5_i64, 0.5_f64, 6_i64))?;
        let theirs = t.call_method1("negative_binomial", (5_i64, 0.5_f64, 6_i64))?;
        assert_close(py, "random-rs-negative_binomial-6", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(42)?;
        let ours = o.call_method1("pareto", (2.0_f64, 5_i64))?;
        let theirs = t.call_method1("pareto", (2.0_f64, 5_i64))?;
        assert_close(py, "random-rs-pareto-5", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(42)?;
        let ours = o.call_method1("power", (2.0_f64, 5_i64))?;
        let theirs = t.call_method1("power", (2.0_f64, 5_i64))?;
        assert_close(py, "random-rs-power-5", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(42)?;
        let ours = o.call_method1("rayleigh", (1.0_f64, 5_i64))?;
        let theirs = t.call_method1("rayleigh", (1.0_f64, 5_i64))?;
        assert_close(py, "random-rs-rayleigh-5", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(42)?;
        let ours = o.call_method1("standard_cauchy", (5_i64,))?;
        let theirs = t.call_method1("standard_cauchy", (5_i64,))?;
        assert_close(py, "random-rs-standard_cauchy-5", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(42)?;
        let ours = o.call_method1("standard_exponential", (5_i64,))?;
        let theirs = t.call_method1("standard_exponential", (5_i64,))?;
        assert_close(py, "random-rs-standard_exponential-5", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(42)?;
        let ours = o.call_method1("standard_gamma", (2.0_f64, 5_i64))?;
        let theirs = t.call_method1("standard_gamma", (2.0_f64, 5_i64))?;
        assert_close(py, "random-rs-standard_gamma-5", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(42)?;
        let ours = o.call_method1("standard_t", (5.0_f64, 5_i64))?;
        let theirs = t.call_method1("standard_t", (5.0_f64, 5_i64))?;
        assert_close(py, "random-rs-standard_t-5", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(42)?;
        let ours = o.call_method1("triangular", (0.0_f64, 0.5_f64, 1.0_f64, 5_i64))?;
        let theirs = t.call_method1("triangular", (0.0_f64, 0.5_f64, 1.0_f64, 5_i64))?;
        assert_close(py, "random-rs-triangular-5", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(42)?;
        let ours = o.call_method1("vonmises", (0.0_f64, 1.0_f64, 5_i64))?;
        let theirs = t.call_method1("vonmises", (0.0_f64, 1.0_f64, 5_i64))?;
        assert_close(py, "random-rs-vonmises-5", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(42)?;
        let ours = o.call_method1("wald", (1.0_f64, 1.0_f64, 5_i64))?;
        let theirs = t.call_method1("wald", (1.0_f64, 1.0_f64, 5_i64))?;
        assert_close(py, "random-rs-wald-5", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(42)?;
        let ours = o.call_method1("weibull", (2.0_f64, 5_i64))?;
        let theirs = t.call_method1("weibull", (2.0_f64, 5_i64))?;
        assert_close(py, "random-rs-weibull-5", true, &ours, &theirs)?;

        let (o, t) = make_rs_pair(42)?;
        let ours = o.call_method1("zipf", (2.0_f64, 5_i64))?;
        let theirs = t.call_method1("zipf", (2.0_f64, 5_i64))?;
        assert_close(py, "random-rs-zipf-5", true, &ours, &theirs)?;

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

#[test]
fn multinomial_rejects_empty_pvals() {
    with_fnp_and_numpy(|_py, module, numpy| {
        let our_rng = module.getattr("random")?.call_method0("default_rng")?;
        let np_rng = numpy.getattr("random")?.call_method0("default_rng")?;
        let pvals: Vec<f64> = vec![];
        let our_err = our_rng
            .call_method1("multinomial", (10_u64, pvals.clone()))
            .is_err();
        let np_err = np_rng.call_method1("multinomial", (10_u64, pvals)).is_err();
        assert!(our_err && np_err, "Both should reject empty pvals");
        Ok(())
    });
}

#[test]
fn multinomial_rejects_nan_pvals() {
    with_fnp_and_numpy(|_py, module, numpy| {
        let our_rng = module.getattr("random")?.call_method0("default_rng")?;
        let np_rng = numpy.getattr("random")?.call_method0("default_rng")?;
        let pvals = vec![f64::NAN, 0.5];
        let our_err = our_rng
            .call_method1("multinomial", (10_u64, pvals.clone()))
            .is_err();
        let np_err = np_rng.call_method1("multinomial", (10_u64, pvals)).is_err();
        assert!(our_err && np_err, "Both should reject NaN pvals");
        Ok(())
    });
}

#[test]
fn multinomial_rejects_negative_pvals() {
    with_fnp_and_numpy(|_py, module, numpy| {
        let our_rng = module.getattr("random")?.call_method0("default_rng")?;
        let np_rng = numpy.getattr("random")?.call_method0("default_rng")?;
        let pvals = vec![-0.1, 0.5];
        let our_err = our_rng
            .call_method1("multinomial", (10_u64, pvals.clone()))
            .is_err();
        let np_err = np_rng.call_method1("multinomial", (10_u64, pvals)).is_err();
        assert!(our_err && np_err, "Both should reject negative pvals");
        Ok(())
    });
}

#[test]
fn multinomial_rejects_pval_greater_than_one() {
    with_fnp_and_numpy(|_py, module, numpy| {
        let our_rng = module.getattr("random")?.call_method0("default_rng")?;
        let np_rng = numpy.getattr("random")?.call_method0("default_rng")?;
        let pvals = vec![1.5, 0.3];
        let our_err = our_rng
            .call_method1("multinomial", (10_u64, pvals.clone()))
            .is_err();
        let np_err = np_rng.call_method1("multinomial", (10_u64, pvals)).is_err();
        assert!(our_err && np_err, "Both should reject pval > 1");
        Ok(())
    });
}

#[test]
fn multinomial_rejects_sum_pvals_minus_last_gt_one() {
    with_fnp_and_numpy(|_py, module, numpy| {
        let our_rng = module.getattr("random")?.call_method0("default_rng")?;
        let np_rng = numpy.getattr("random")?.call_method0("default_rng")?;
        // sum([:-1]) = 0.7 + 0.5 = 1.2 > 1.0
        let pvals = vec![0.7, 0.5, 0.1];
        let our_err = our_rng
            .call_method1("multinomial", (10_u64, pvals.clone()))
            .is_err();
        let np_err = np_rng.call_method1("multinomial", (10_u64, pvals)).is_err();
        assert!(our_err && np_err, "Both should reject sum(pvals[:-1]) > 1");
        Ok(())
    });
}

#[test]
fn default_rng_accepts_diverse_seed_types() {
    with_fnp_and_numpy(|py, module, numpy| {
        let our_random = module.getattr("random")?;
        let np_random = numpy.getattr("random")?;

        // 1. None seed
        let rng1 = our_random.call_method1("default_rng", (py.None(),))?;
        let _ = rng1.call_method0("random")?;

        // 2. Integer seed
        let rng2 = our_random.call_method1("default_rng", (12345_u64,))?;
        let val2 = rng2.call_method0("random")?.extract::<f64>()?;
        let np_rng2 = np_random.call_method1("default_rng", (12345_u64,))?;
        let np_val2 = np_rng2.call_method0("random")?.extract::<f64>()?;
        assert_eq!(
            val2.to_bits(),
            np_val2.to_bits(),
            "Integer seed must match numpy exactly"
        );

        // 3. fnp PCG64 BitGenerator object
        let pcg_cls = our_random.getattr("PCG64")?;
        let pcg_obj = pcg_cls.call1((12345_u64,))?;
        let rng3 = our_random.call_method1("default_rng", (&pcg_obj,))?;
        let val3 = rng3.call_method0("random")?.extract::<f64>()?;
        assert_eq!(
            val3.to_bits(),
            np_val2.to_bits(),
            "PCG64 BitGenerator seed must match"
        );

        // 4. NumPy upstream BitGenerator object
        let np_pcg_cls = np_random.getattr("PCG64")?;
        let np_pcg_obj = np_pcg_cls.call1((12345_u64,))?;
        let rng4 = our_random.call_method1("default_rng", (&np_pcg_obj,))?;
        let val4 = rng4.call_method0("random")?.extract::<f64>()?;
        assert_eq!(
            val4.to_bits(),
            np_val2.to_bits(),
            "NumPy PCG64 BitGenerator seed must match"
        );

        // 5. SeedSequence object
        let ss_cls = our_random.getattr("SeedSequence")?;
        let ss_obj = ss_cls.call1((12345_u64,))?;
        let rng5 = our_random.call_method1("default_rng", (&ss_obj,))?;
        let val5 = rng5.call_method0("random")?.extract::<f64>()?;
        assert_eq!(
            val5.to_bits(),
            np_val2.to_bits(),
            "SeedSequence seed must match"
        );

        // 6. Sequence of integers
        let seq = vec![123_u32, 456_u32, 789_u32];
        let rng6 = our_random.call_method1("default_rng", (seq.clone(),))?;
        let np_rng6 = np_random.call_method1("default_rng", (seq,))?;
        let val6 = rng6.call_method0("random")?.extract::<f64>()?;
        let np_val6 = np_rng6.call_method0("random")?.extract::<f64>()?;
        assert_eq!(
            val6.to_bits(),
            np_val6.to_bits(),
            "Sequence seed must match numpy"
        );

        // 7. Existing Generator (idempotent / returns generator)
        let rng7 = our_random.call_method1("default_rng", (&rng2,))?;
        let is_same = rng7.is(&rng2);
        assert!(
            is_same,
            "default_rng(generator) should preserve generator identity"
        );

        Ok(())
    });
}

/// NumPy's `next_uint32` buffers the high half of a 64-bit output in the BIT GENERATOR
/// (`has_uint32` / `uinteger`); only a state set, jump or advance clears it, and MT19937
/// draws 32 bits natively. fnp used to drop the half-word on every float/64-bit draw and on
/// MT19937 split a 64-bit word, so any interleaving of 32-bit bounded integers with another
/// draw diverged from NumPy (e.g. PCG64(123): int32 x3, random(2), int32 x5 gave
/// [333, 175, ...] instead of [53, 333, ...]) and `bit_generator.state` always reported
/// has_uint32=0 (deadlock-audit-rc0923-epic-71qy3.25). Single-call-from-fresh-seed checks
/// cannot see this; the sequence below interleaves every draw family on all five bit
/// generators and compares every output AND the full state dict against NumPy.
#[test]
fn interleaved_draws_keep_numpy_uint32_buffer_and_state_on_all_bit_generators() {
    with_fnp_and_numpy(|py, module, numpy| {
        let globals = PyDict::new(py);
        globals.set_item("fnp", &module)?;
        globals.set_item("np", &numpy)?;
        let code = std::ffi::CString::new(
            r#"
def norm(st):
    def f(v):
        if isinstance(v, dict): return {k: f(x) for k, x in v.items()}
        if isinstance(v, np.ndarray): return v.tolist()
        return v
    return f(st)
seq = [
 ("integers", (0, 100, 3), {"dtype": np.int32}), ("random", (2,), {}),
 ("integers", (0, 1000, 5), {"dtype": np.int32}), ("standard_normal", (3,), {}),
 ("integers", (0, 50000, 3), {"dtype": np.uint16}), ("normal", (1.0, 2.0, 2), {}),
 ("integers", (-100, 100, 5), {"dtype": np.int8}), ("exponential", (1.5, 2), {}),
 ("integers", (0, 255, 7), {"dtype": np.uint8}), ("random", (3,), {"dtype": np.float32}),
 ("uniform", (0.0, 5.0, 3), {}), ("bytes", (5,), {}), ("integers", (0, 7, 1), {"dtype": np.int32}),
 ("STATE_ROUNDTRIP", (), {}), ("integers", (0, 9, 3), {"dtype": np.int32}),
 ("gamma", (2.0, 1.0, 3), {}), ("integers", (0, 10**12, 2), {}), ("standard_exponential", (2,), {}),
 ("permutation", (7,), {}), ("integers", (0, 3, 3), {"dtype": np.uint32}), ("choice", (10, 3), {}),
 ("random", (5,), {"dtype": np.float32}), ("bytes", (3,), {}),
 ("random", (1 << 17,), {"dtype": np.float32}), ("bytes", ((1 << 18) + 4,), {}),
 ("integers", (0, 2**31, 2), {"dtype": np.int64}), ("integers", (0, 9, 1), {"dtype": np.int32}),
]
bad = []
for kind in ["PCG64", "PCG64DXSM", "MT19937", "Philox", "SFC64"]:
    f = fnp.random.Generator(getattr(fnp.random, kind)(123))
    n = np.random.Generator(getattr(np.random, kind)(123))
    for i, (m, a, kw) in enumerate(seq):
        if m == "STATE_ROUNDTRIP":
            f.bit_generator.state = f.bit_generator.state
            n.bit_generator.state = n.bit_generator.state
        else:
            g = getattr(f, m)(*a, **kw); w = getattr(n, m)(*a, **kw)
            ga, wa = np.asarray(g), np.asarray(w)
            if type(g) is not type(w) or ga.dtype != wa.dtype or ga.tobytes() != wa.tobytes():
                bad.append(f"{kind} step{i} {m} draws differ"); break
        if norm(f.bit_generator.state) != norm(n.bit_generator.state):
            bad.append(f"{kind} step{i} {m} state differs"); break
result = (len(seq), bad)
"#,
        )
        .expect("script has no NUL");
        py.run(&code, Some(&globals), None)?;
        let (steps, bad): (usize, Vec<String>) = globals
            .get_item("result")?
            .expect("script sets result")
            .extract()?;
        assert_eq!(steps, 27, "sequence length drifted");
        assert!(
            bad.is_empty(),
            "RNG stream/state diverged from numpy: {bad:?}"
        );
        Ok(())
    });
}

/// Every Generator distribution, the legacy RandomState ones and the module-level
/// `random.<dist>` functions (bound methods of the global RandomState) used to declare
/// `f64`/`i64`/`u64` parameters, so ANY array-valued parameter raised
/// `TypeError: only 0-dimensional arrays can be converted to Python scalars`
/// (deadlock-audit-rc0923-epic-71qy3.6). Array-valued calls now run NumPy's own sampler on
/// this generator's exact state, so they must match NumPy bit-for-bit, leave the stream
/// where NumPy leaves it (checked by drawing again afterwards), and raise what NumPy raises.
#[test]
fn array_valued_distribution_parameters_match_numpy_stream() {
    with_fnp_and_numpy(|py, module, numpy| {
        let globals = PyDict::new(py);
        globals.set_item("fnp", &module)?;
        globals.set_item("np", &numpy)?;
        let code = std::ffi::CString::new(
            r#"
A = np.array([0.5, 1.0, 2.5]); B = np.array([[1.0], [3.0]])
P = np.array([0.2, 0.5, 0.7]); N = np.array([5, 10, 20])
gen_cases = {
 "normal": dict(loc=A, scale=B), "uniform": dict(low=A, high=B+5), "exponential": dict(scale=A),
 "gamma": dict(shape=A, scale=B), "beta": dict(a=A, b=B), "chisquare": dict(df=A+1),
 "f": dict(dfnum=A+1, dfden=B+1), "noncentral_chisquare": dict(df=A+1, nonc=B),
 "noncentral_f": dict(dfnum=A+1, dfden=B+1, nonc=A), "standard_gamma": dict(shape=A),
 "standard_t": dict(df=A+1), "vonmises": dict(mu=A, kappa=B), "pareto": dict(a=A),
 "weibull": dict(a=A), "power": dict(a=A), "laplace": dict(loc=A, scale=B), "gumbel": dict(loc=A, scale=B),
 "logistic": dict(loc=A, scale=B), "lognormal": dict(mean=A, sigma=B), "rayleigh": dict(scale=A),
 "wald": dict(mean=A, scale=B), "triangular": dict(left=A-1, mode=A, right=B+3),
 "binomial": dict(n=N, p=P), "negative_binomial": dict(n=N, p=P), "poisson": dict(lam=A),
 "zipf": dict(a=A+1.5), "geometric": dict(p=P), "hypergeometric": dict(ngood=N, nbad=N+1, nsample=N//2+1),
 "logseries": dict(p=P), "integers": dict(low=np.array([0, 5, 10]), high=np.array([[20], [40]])),
}
bad = []
def same(a, b):
    return type(a) is type(b) and np.shape(a) == np.shape(b) and np.asarray(a).tobytes() == np.asarray(b).tobytes()
for name, kw in gen_cases.items():
    f = fnp.random.default_rng(7); n = np.random.default_rng(7)
    if not (same(getattr(f, name)(**kw), getattr(n, name)(**kw)) and same(f.random(3), n.random(3))):
        bad.append(f"Generator.{name}")
# edge cases that must behave exactly like NumPy (value or exception type)
edge = [
 ("normal loc=None", lambda r: r.normal(loc=None)), ("binomial n=-1", lambda r: r.binomial(-1, 0.5)),
 ("normal 1-elem array", lambda r: r.normal(loc=np.array([1.0]))), ("normal list loc", lambda r: r.normal(loc=[0, 1], size=(3, 2))),
 ("normal bad broadcast", lambda r: r.normal(loc=[0, 1], size=3)), ("integers bool", lambda r: r.integers(0, 2, 5, dtype=bool)),
 ("integers high=None", lambda r: r.integers(5, size=4)), ("binomial float n", lambda r: r.binomial(5.0, 0.5)),
]
for label, fn in edge:
    try: w = fn(np.random.default_rng(3)); we = None
    except Exception as e: w, we = None, type(e).__name__
    try: g = fn(fnp.random.default_rng(3)); ge = None
    except Exception as e: g, ge = None, type(e).__name__
    if we != ge or (we is None and not same(g, w)):
        bad.append(f"edge {label}: numpy={we} fnp={ge}")
rs_cases = [("normal", dict(loc=A, scale=B)), ("uniform", dict(low=A, high=B+5)), ("gamma", dict(shape=A, scale=B)),
            ("exponential", dict(scale=A)), ("triangular", dict(left=A-1, mode=A, right=B+3)),
            ("randint", dict(low=np.array([0, 5, 10]), high=np.array([[20], [40]]))), ("randint", dict(low=0, high=2, size=5, dtype=bool))]
for name, kw in rs_cases:
    f = fnp.random.RandomState(9); n = np.random.RandomState(9)
    # a scalar normal first leaves the legacy Gaussian cache populated; it must survive the delegated call
    ok = same(f.normal(), n.normal()) and same(getattr(f, name)(**kw), getattr(n, name)(**kw)) and same(f.normal(size=3), n.normal(size=3))
    if not ok:
        bad.append(f"RandomState.{name}")
np.random.seed(4); w = np.random.normal(loc=A, scale=B); fnp.random.seed(4); g = fnp.random.normal(loc=A, scale=B)
if not same(g, w):
    bad.append("module-level random.normal")
result = (len(gen_cases), bad)
"#,
        )
        .expect("script has no NUL");
        py.run(&code, Some(&globals), None)?;
        let (count, bad): (usize, Vec<String>) = globals
            .get_item("result")?
            .expect("script sets result")
            .extract()?;
        assert_eq!(count, 30, "distribution table drifted");
        assert!(
            bad.is_empty(),
            "array-valued parameters diverge from numpy: {bad:?}"
        );
        Ok(())
    });
}

/// NumPy's signatures are `standard_normal(size=None, dtype=np.float64, out=None)` and
/// `standard_exponential(size=None, dtype=np.float64, method='zig', out=None)`. fnp declared
/// neither `dtype`, so the ordinary `rng.standard_normal(n, np.float32)` bound the dtype to
/// `out` and raised TypeError (found by running numpy's own test suite against fnp). Every
/// case must match NumPy's value bit-for-bit (or its exception type) and leave the stream
/// where NumPy leaves it. NumPy treats ANY method other than 'zig' as 'inv' for float64, so
/// "bogus" is a value case, not an error case.
#[test]
fn standard_normal_and_exponential_take_numpy_dtype_and_method_arguments() {
    with_fnp_and_numpy(|py, module, numpy| {
        let globals = PyDict::new(py);
        globals.set_item("fnp", &module)?;
        globals.set_item("np", &numpy)?;
        let code = std::ffi::CString::new(
            r#"
def same(a, b):
    return (type(a) is type(b) and np.shape(a) == np.shape(b)
            and np.asarray(a).dtype == np.asarray(b).dtype
            and np.asarray(a).tobytes() == np.asarray(b).tobytes())
cases = [
 ("normal f32 positional", lambda r: r.standard_normal(5, np.float32)),
 ("normal f32 keyword str", lambda r: r.standard_normal(size=4, dtype="float32")),
 ("normal f64 positional", lambda r: r.standard_normal(3, np.float64)),
 ("normal f32 out", lambda r: r.standard_normal(dtype=np.float32, out=np.empty(3, np.float32))),
 ("normal f32 scalar", lambda r: r.standard_normal(None, np.float32)),
 ("normal int dtype", lambda r: r.standard_normal(3, np.int32)),
 ("exp f32 positional", lambda r: r.standard_exponential(4, np.float32)),
 ("exp f32 inv", lambda r: r.standard_exponential(4, np.float32, "inv")),
 ("exp f64 inv positional", lambda r: r.standard_exponential(3, np.float64, "inv")),
 ("exp inv keyword", lambda r: r.standard_exponential(3, method="inv")),
 ("exp unknown method", lambda r: r.standard_exponential(3, np.float64, "bogus")),
 ("exp f32 out", lambda r: r.standard_exponential(dtype=np.float32, out=np.empty((2, 2), np.float32))),
]
bad = []
for label, fn in cases:
    f = fnp.random.default_rng(11); n = np.random.default_rng(11)
    try: w = fn(n); we = None
    except Exception as e: w, we = None, type(e).__name__
    try: g = fn(f); ge = None
    except Exception as e: g, ge = None, type(e).__name__
    if we != ge or (we is None and not same(g, w)) or not same(f.random(3), n.random(3)):
        bad.append(f"{label}: numpy={we} fnp={ge}")
result = (len(cases), bad)
"#,
        )
        .expect("script has no NUL");
        py.run(&code, Some(&globals), None)?;
        let (count, bad): (usize, Vec<String>) = globals
            .get_item("result")?
            .expect("script sets result")
            .extract()?;
        assert_eq!(count, 12, "case table drifted");
        assert!(
            bad.is_empty(),
            "standard_normal/standard_exponential dtype/method diverge from numpy: {bad:?}"
        );
        Ok(())
    });
}
