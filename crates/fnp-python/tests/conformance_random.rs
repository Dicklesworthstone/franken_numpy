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

/// `jumped` is NumPy's parallel-streams recipe, so it must land exactly where NumPy lands.
/// fnp used a small per-kind stride: MT19937 did not move at all (a jumped generator REPLAYED
/// its parent's stream), and PCG64/PCG64DXSM/Philox jumped to states NumPy never produces.
/// `random_raw` on MT19937 spliced two 32-bit draws into each u64 where NumPy returns one
/// draw per element. Compares full state dicts and the raw stream after jumping, from fresh
/// and mid-stream states; SFC64 has no `jumped` in NumPy and must raise AttributeError.
#[test]
fn jumped_and_random_raw_match_numpy_for_every_bit_generator() {
    with_fnp_and_numpy(|py, module, numpy| {
        let globals = PyDict::new(py);
        globals.set_item("fnp", &module)?;
        globals.set_item("np", &numpy)?;
        let code = std::ffi::CString::new(
            r#"
def same_state(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    return np.array_equal(np.asarray(a), np.asarray(b))
bad = []
for kind in ("MT19937", "PCG64", "PCG64DXSM", "Philox"):
    for seed, warmup, jumps in ((1, 0, 1), (7, 3, 1), (12345, 5, 2), (99, 1, 3)):
        f, n = getattr(fnp.random, kind)(seed), getattr(np.random, kind)(seed)
        if not np.array_equal(f.random_raw(warmup + 1), n.random_raw(warmup + 1)):
            bad.append(f"{kind}({seed}).random_raw({warmup + 1})")
        fj, nj = f.jumped(jumps), n.jumped(jumps)
        if not same_state(fj.state, nj.state):
            bad.append(f"{kind}({seed}) after {warmup + 1} draws .jumped({jumps}).state")
        if not np.array_equal(fj.random_raw(5), nj.random_raw(5)):
            bad.append(f"{kind}({seed}).jumped({jumps}).random_raw(5)")
        if same_state(fj.state, f.state):
            bad.append(f"{kind}({seed}).jumped({jumps}) did not move")
    g, h = fnp.random.Generator(getattr(fnp.random, kind)(3).jumped()), np.random.Generator(getattr(np.random, kind)(3).jumped())
    if not np.array_equal(g.random(4), h.random(4)):
        bad.append(f"Generator({kind}(3).jumped()).random")
try:
    fnp.random.SFC64(1).jumped()
    bad.append("SFC64.jumped did not raise")
except AttributeError:
    pass
result = bad
"#,
        )
        .expect("script has no NUL");
        py.run(&code, Some(&globals), None)?;
        let bad: Vec<String> = globals
            .get_item("result")?
            .expect("script sets result")
            .extract()?;
        assert!(
            bad.is_empty(),
            "jumped/random_raw diverge from numpy: {bad:?}"
        );
        Ok(())
    });
}

/// RNG objects must survive `pickle` and `copy.deepcopy` mid-stream (multiprocessing,
/// joblib and checkpointing all pickle generators). The classes reported module `builtins`,
/// so `pickle.dumps` failed for Generator, every bit generator and SeedSequence, and
/// RandomState had no reduce at all. Each object is advanced first so the hidden state
/// matters - a Generator with a buffered uint32, a RandomState with a cached Gaussian, a
/// SeedSequence that has spawned - and each clone must continue the ORIGINAL's stream.
#[test]
fn rng_objects_pickle_and_deepcopy_mid_stream() {
    with_fnp_and_numpy(|py, module, numpy| {
        let globals = PyDict::new(py);
        globals.set_item("fnp", &module)?;
        globals.set_item("np", &numpy)?;
        let code = std::ffi::CString::new(
            r#"
import copy, pickle, sys
# pickle finds a class through its module: a real install imports `fnp_python`, but this
# harness builds the module in-process, so register it the way an import would.
sys.modules[fnp.__name__] = fnp
bad = []
def check(label, obj, draw):
    try:
        snap = pickle.dumps(obj)
        deep = copy.deepcopy(obj)
    except Exception as exc:
        bad.append(f"{label}: {type(exc).__name__}: {exc}")
        return
    expected = draw(obj)
    for how, clone in (("pickle", pickle.loads(snap)), ("deepcopy", deep)):
        if type(clone) is not type(obj):
            bad.append(f"{label} {how}: type {type(clone).__name__}")
        elif not np.array_equal(draw(clone), expected):
            bad.append(f"{label} {how}: stream differs")
for kind in ("MT19937", "PCG64", "PCG64DXSM", "Philox", "SFC64"):
    bg = getattr(fnp.random, kind)(11)
    bg.random_raw(3)
    check(kind, bg, lambda b: b.random_raw(4))
    g = fnp.random.Generator(getattr(fnp.random, kind)(5))
    g.integers(0, 10, size=3, dtype=np.int32)
    check(f"Generator({kind})", g, lambda r: np.concatenate([r.integers(0, 10, size=3, dtype=np.int32), r.random(3)]))
rs = fnp.random.RandomState(9)
rs.normal()
check("RandomState", rs, lambda r: r.normal(size=3))
ss = fnp.random.SeedSequence(5)
ss.spawn(2)
check("SeedSequence", ss, lambda s: np.concatenate([s.generate_state(4), [len(s.spawn(1)[0].spawn_key)]]))
result = bad
"#,
        )
        .expect("script has no NUL");
        py.run(&code, Some(&globals), None)?;
        let bad: Vec<String> = globals
            .get_item("result")?
            .expect("script sets result")
            .extract()?;
        assert!(
            bad.is_empty(),
            "RNG pickle/deepcopy round trips diverge: {bad:?}"
        );
        Ok(())
    });
}

/// numpy's Generator.multinomial broadcasts an ARRAY `n` against `pvals`, accepts N-D
/// `pvals`, and answers a negative `n` with `ValueError("n < 0")`. fnp declared `n: u64`
/// (TypeError / OverflowError) and flattened `pvals`. Values must match numpy bit-for-bit,
/// errors by type and message, and the stream must continue where numpy's does.
#[test]
fn multinomial_array_n_nd_pvals_and_negative_n_match_numpy() {
    with_fnp_and_numpy(|py, module, numpy| {
        let globals = PyDict::new(py);
        globals.set_item("fnp", &module)?;
        globals.set_item("np", &numpy)?;
        let code = std::ffi::CString::new(
            r#"
bad = []
cases = [
    ("array n", lambda r: r.multinomial([3, 4], [0.2, 0.8])),
    ("array n + size", lambda r: r.multinomial(np.array([5, 10]), [0.3, 0.7], size=(3, 2))),
    ("2-D pvals", lambda r: r.multinomial(5, [[0.2, 0.8], [0.5, 0.5]])),
    ("negative n", lambda r: r.multinomial(-1, [0.2, 0.8])),
    ("scalar n", lambda r: r.multinomial(7, [0.1, 0.2, 0.7], size=4)),
]
for label, fn in cases:
    f, n = fnp.random.default_rng(21), np.random.default_rng(21)
    try: w = fn(n); we = None
    except Exception as e: w, we = None, (type(e).__name__, str(e))
    try: g = fn(f); ge = None
    except Exception as e: g, ge = None, (type(e).__name__, str(e))
    if we != ge or (we is None and (g.dtype != w.dtype or g.shape != w.shape or not np.array_equal(g, w))):
        bad.append(f"{label}: numpy={we or w.shape} fnp={ge or g.shape}")
    elif not np.array_equal(f.random(3), n.random(3)):
        bad.append(f"{label}: stream diverged after the call")
result = bad
"#,
        )
        .expect("script has no NUL");
        py.run(&code, Some(&globals), None)?;
        let bad: Vec<String> = globals
            .get_item("result")?
            .expect("script sets result")
            .extract()?;
        assert!(
            bad.is_empty(),
            "multinomial broadcast/error surface diverges from numpy: {bad:?}"
        );
        Ok(())
    });
}

/// numpy's Generator.dirichlet requires a 1-D `alpha` with no negative entry and switches to
/// a beta-variate stick-breaking sampler when `alpha.max() < 0.1`; negative_binomial rejects
/// `n <= 0`, `p` outside (0, 1] and a Poisson-overflowing `(1-p)/p * (n + 10 sqrt(n))`.
/// fnp flattened a 2-D `alpha` into a wrongly shaped result, drew different small-alpha and
/// NaN-alpha values, and returned a value for `negative_binomial(2**62, 0.1)` (numpy's own
/// test_dirichlet_bad_alpha / test_dirichlet_small_alpha /
/// test_negative_binomial_invalid_p_n_combination). Values bit-for-bit, errors by type and
/// message, stream continuity after each call.
#[test]
fn dirichlet_and_negative_binomial_validation_and_small_alpha_match_numpy() {
    with_fnp_and_numpy(|py, module, numpy| {
        let globals = PyDict::new(py);
        globals.set_item("fnp", &module)?;
        globals.set_item("np", &numpy)?;
        let code = std::ffi::CString::new(
            r#"
bad = []
cases = [
    ("dirichlet 2-D", lambda r: r.dirichlet([[5, 1]])),
    ("dirichlet 2-D array", lambda r: r.dirichlet(np.array([[5, 1], [1, 5]]))),
    ("dirichlet negative", lambda r: r.dirichlet(np.array([5.4e-01, -1.0e-16]))),
    ("dirichlet NaN", lambda r: r.dirichlet([1.0, np.nan])),
    ("dirichlet small alpha", lambda r: r.dirichlet([0.05, 0.02, 0.01], size=3)),
    ("dirichlet mixed alpha", lambda r: r.dirichlet([0.05, 0.5], size=2)),
    ("dirichlet plain", lambda r: r.dirichlet([1.0, 2.0, 3.0], size=2)),
    ("negative_binomial overflow", lambda r: r.negative_binomial(2**62, 0.1)),
    ("negative_binomial n <= 0", lambda r: r.negative_binomial(0, 0.5)),
    ("negative_binomial p NaN", lambda r: r.negative_binomial(5, np.nan)),
    ("negative_binomial plain", lambda r: r.negative_binomial(5, 0.3, size=4)),
]
for label, fn in cases:
    f, n = fnp.random.default_rng(5), np.random.default_rng(5)
    try: w = fn(n); we = None
    except Exception as e: w, we = None, (type(e).__name__, str(e))
    try: g = fn(f); ge = None
    except Exception as e: g, ge = None, (type(e).__name__, str(e))
    if we != ge or (we is None and (np.shape(g) != np.shape(w) or np.asarray(g).tobytes() != np.asarray(w).tobytes())):
        bad.append(f"{label}: numpy={we or np.shape(w)} fnp={ge or np.shape(g)}")
    elif not np.array_equal(f.random(3), n.random(3)):
        bad.append(f"{label}: stream diverged after the call")
result = bad
"#,
        )
        .expect("script has no NUL");
        py.run(&code, Some(&globals), None)?;
        let bad: Vec<String> = globals
            .get_item("result")?
            .expect("script sets result")
            .extract()?;
        assert!(
            bad.is_empty(),
            "dirichlet/negative_binomial diverge from numpy: {bad:?}"
        );
        Ok(())
    });
}

/// numpy's legacy RandomState takes an int in [0, 2**32) (`init_genrand`) or a 1-d integer
/// array/sequence (`init_by_array`), and raises ValueError/TypeError with its own messages for
/// negative, oversized, empty, 2-d and float seeds. fnp's constructor was `seed: u64`, so
/// `RandomState([1, 2, 3])` or `RandomState(range(4))` was a TypeError and `RandomState(-1)` an
/// OverflowError (numpy's own TestSeed). Both the constructor and `seed()` must match numpy's
/// stream bit-for-bit, and its exception type and message.
#[test]
fn random_state_accepts_numpys_seed_forms() {
    with_fnp_and_numpy(|py, module, numpy| {
        let globals = PyDict::new(py);
        globals.set_item("fnp", &module)?;
        globals.set_item("np", &numpy)?;
        let code = std::ffi::CString::new(
            r#"
bad = []
seeds = [
    ("int", 12345), ("array", np.array([1, 2, 3])), ("list", [1, 2, 3, 4]), ("range", range(4)),
    ("uint32 array", np.arange(10, dtype=np.uint32)), ("0-d array", np.array(7)),
    ("MT19937", "mt"), ("None", None),
    ("negative", -1), ("too large", 2**32), ("2-d", [[1, 2], [3, 4]]), ("empty", []),
    ("float", 1.5), ("negative in array", [1, -2]),
]
def make(mod, seed, via):
    if isinstance(seed, str):
        seed = mod.random.MT19937(99)
    if via == "ctor":
        return mod.random.RandomState(seed)
    rs = mod.random.RandomState(0)
    rs.seed(seed)
    return rs
for label, seed in seeds:
    for via in ("ctor", "seed()"):
        if label == "MT19937" and via == "seed()":
            continue
        try: w = make(np, seed, via); we = None
        except Exception as e: w, we = None, (type(e).__name__, str(e))
        try: g = make(fnp, seed, via); ge = None
        except Exception as e: g, ge = None, (type(e).__name__, str(e))
        if we != ge:
            bad.append(f"{label} via {via}: numpy={we} fnp={ge}")
        elif we is None and label != "None":
            if not (np.array_equal(g.randint(0, 2**31, 5), w.randint(0, 2**31, 5))
                    and np.array_equal(g.normal(size=3), w.normal(size=3))):
                bad.append(f"{label} via {via}: stream differs")
result = bad
"#,
        )
        .expect("script has no NUL");
        py.run(&code, Some(&globals), None)?;
        let bad: Vec<String> = globals
            .get_item("result")?
            .expect("script sets result")
            .extract()?;
        assert!(
            bad.is_empty(),
            "RandomState seed forms diverge from numpy: {bad:?}"
        );
        Ok(())
    });
}

/// numpy's legacy `RandomState.randint` for every integer dtype and range, including a
/// near-full-width int64 range and the 8/16-bit dtypes: fnp raised "integer sample exceeds
/// int64" for `randint(iinfo(int64).min, iinfo(int64).max - 1, dtype=int64)` and drew DIFFERENT
/// values than numpy for int8/int16/uint8/uint16 (numpy buffers 32-bit draws there), found under
/// numpy's own test_multiarray::test_sort_int. Values bit-for-bit, then the stream.
#[test]
fn random_state_randint_matches_numpy_across_dtypes_and_full_ranges() {
    with_fnp_and_numpy(|py, module, numpy| {
        let globals = PyDict::new(py);
        globals.set_item("fnp", &module)?;
        globals.set_item("np", &numpy)?;
        let code = std::ffi::CString::new(
            r#"
bad = []
for dt in ("b", "B", "h", "H", "i", "I", "l", "L", "q", "Q"):
    ii = np.iinfo(dt)
    for label, lo, hi in (("full-1", ii.min, ii.max - 1), ("full", ii.min, ii.max), ("small", 0, 10)):
        f, n = fnp.random.RandomState(5), np.random.RandomState(5)
        try: w = n.randint(lo, hi, size=7, dtype=dt); we = None
        except Exception as e: w, we = None, type(e).__name__
        try: g = f.randint(lo, hi, size=7, dtype=dt); ge = None
        except Exception as e: g, ge = None, type(e).__name__
        if we != ge or (we is None and (g.dtype != w.dtype or not np.array_equal(g, w))):
            bad.append(f"{dt} {label}: numpy={we or w.tolist()} fnp={ge or (g.tolist() if g is not None else None)}")
        elif not np.array_equal(f.randint(0, 1000, 3), n.randint(0, 1000, 3)):
            bad.append(f"{dt} {label}: stream diverged after the call")
np.random.seed(11); w = np.random.randint(np.iinfo("l").min, np.iinfo("l").max - 1, size=5, dtype="l")
fnp.random.seed(11); g = fnp.random.randint(np.iinfo("l").min, np.iinfo("l").max - 1, size=5, dtype="l")
if not np.array_equal(g, w):
    bad.append("module-level randint full int64")
result = bad
"#,
        )
        .expect("script has no NUL");
        py.run(&code, Some(&globals), None)?;
        let bad: Vec<String> = globals
            .get_item("result")?
            .expect("script sets result")
            .extract()?;
        assert!(
            bad.is_empty(),
            "RandomState.randint diverges from numpy: {bad:?}"
        );
        Ok(())
    });
}

/// Bead rc0923 .6 acceptance: every Generator distribution in the method table, at 3 seeds, with
/// (a) scalar parameters, (b) a 1-D array parameter, (c) a 2-D (2, 1) parameter broadcast against
/// `size=(2, 3)`, and (d) an incompatible shape - a (3,) parameter with `size=(2,)` - must give
/// numpy's value byte-for-byte (type, dtype, shape, bytes) or numpy's exception type, and leave
/// the stream where numpy leaves it (checked by one more draw). The (2, 1)-against-size case is
/// the negative control for a sampler that draws per parameter column instead of in C order.
/// Every divergence is reported as `method form seed`.
#[test]
fn distribution_method_table_matches_numpy_across_param_shapes_and_seeds() {
    with_fnp_and_numpy(|py, module, numpy| {
        let globals = PyDict::new(py);
        globals.set_item("fnp", &module)?;
        globals.set_item("np", &numpy)?;
        let code = std::ffi::CString::new(
            r#"
# (method, first parameter, scalar value, 1-D values, other scalar kwargs)
table = [
 ("normal", "loc", 0.5, [0.5, 1.0, 2.5], {"scale": 2.0}),
 ("uniform", "low", 0.5, [0.0, 0.5, 1.0], {"high": 3.0}),
 ("exponential", "scale", 1.5, [0.5, 1.0, 2.5], {}),
 ("gamma", "shape", 1.5, [0.5, 1.0, 2.5], {"scale": 2.0}),
 ("beta", "a", 1.5, [0.5, 1.0, 2.5], {"b": 2.0}),
 ("chisquare", "df", 2.5, [1.5, 2.0, 3.5], {}),
 ("f", "dfnum", 2.5, [1.5, 2.0, 3.5], {"dfden": 4.0}),
 ("noncentral_chisquare", "df", 2.5, [1.5, 2.0, 3.5], {"nonc": 1.0}),
 ("noncentral_f", "dfnum", 2.5, [1.5, 2.0, 3.5], {"dfden": 4.0, "nonc": 1.0}),
 ("standard_gamma", "shape", 1.5, [0.5, 1.0, 2.5], {}),
 ("standard_t", "df", 2.5, [1.5, 2.0, 3.5], {}),
 ("vonmises", "mu", 0.5, [0.0, 0.5, 1.0], {"kappa": 1.5}),
 ("pareto", "a", 1.5, [0.5, 1.0, 2.5], {}),
 ("weibull", "a", 1.5, [0.5, 1.0, 2.5], {}),
 ("power", "a", 1.5, [0.5, 1.0, 2.5], {}),
 ("laplace", "loc", 0.5, [0.5, 1.0, 2.5], {"scale": 2.0}),
 ("gumbel", "loc", 0.5, [0.5, 1.0, 2.5], {"scale": 2.0}),
 ("logistic", "loc", 0.5, [0.5, 1.0, 2.5], {"scale": 2.0}),
 ("lognormal", "mean", 0.5, [0.5, 1.0, 2.5], {"sigma": 0.5}),
 ("rayleigh", "scale", 1.5, [0.5, 1.0, 2.5], {}),
 ("wald", "mean", 1.5, [0.5, 1.0, 2.5], {"scale": 2.0}),
 ("triangular", "left", -0.5, [-1.0, -0.5, 0.0], {"mode": 0.5, "right": 2.0}),
 ("binomial", "n", 10, [5, 10, 20], {"p": 0.4}),
 ("negative_binomial", "n", 10, [5, 10, 20], {"p": 0.4}),
 ("poisson", "lam", 3.5, [0.5, 3.0, 25.0], {}),
 ("zipf", "a", 2.5, [2.0, 2.5, 3.5], {}),
 ("geometric", "p", 0.4, [0.2, 0.5, 0.7], {}),
 ("hypergeometric", "ngood", 10, [5, 10, 20], {"nbad": 8, "nsample": 4}),
 ("logseries", "p", 0.4, [0.2, 0.5, 0.7], {}),
 ("integers", "low", 2, [0, 3, 7], {"high": 40}),
]
def same(a, b):
    return (type(a) is type(b) and np.shape(a) == np.shape(b)
            and np.asarray(a).dtype == np.asarray(b).dtype
            and np.asarray(a).tobytes() == np.asarray(b).tobytes())
def outcome(rng, name, kw):
    try:
        value = getattr(rng, name)(**kw)
        err = None
    except Exception as exc:
        value, err = None, type(exc).__name__
    return value, err, rng.random(2)
bad = []
count = 0
for name, first, scalar, one_d, others in table:
    column = np.array(one_d[:2]).reshape(2, 1)
    forms = {
        "scalar": dict(others, **{first: scalar}),
        "1-D": dict(others, **{first: np.array(one_d)}),
        "2-D(2,1) x size(2,3)": dict(others, size=(2, 3), **{first: column}),
        "incompatible (3,) x size(2,)": dict(others, size=(2,), **{first: np.array(one_d)}),
    }
    for form, kw in forms.items():
        for seed in (0, 7, 123):
            count += 1
            gv, ge, gnext = outcome(fnp.random.default_rng(seed), name, kw)
            wv, we, wnext = outcome(np.random.default_rng(seed), name, kw)
            if ge != we or (we is None and not same(gv, wv)) or not same(gnext, wnext):
                bad.append(f"{name} {form} seed={seed}: numpy_err={we} fnp_err={ge}")
result = (count, bad)
"#,
        )
        .expect("script has no NUL");
        py.run(&code, Some(&globals), None)?;
        let (count, bad): (usize, Vec<String>) = globals
            .get_item("result")?
            .expect("script sets result")
            .extract()?;
        assert_eq!(count, 30 * 4 * 3, "method table or form set drifted");
        assert!(
            bad.is_empty(),
            "distribution cells diverge from numpy: {bad:#?}"
        );
        Ok(())
    });
}

/// Former DISCREPANCIES.md DISC-004 (multivariate_normal "Cholesky, not SVD") and DISC-005
/// (multivariate_hypergeometric "sequential draws") claimed seeded streams that differ from
/// numpy. At the Python surface both are seed-exact: every draw and the stream position after
/// it must match numpy. docs/DIVERGENCES.md cites this test as the evidence for retiring them.
#[test]
fn multivariate_distributions_are_seed_exact_with_numpy() {
    with_fnp_and_numpy(|py, module, numpy| {
        let globals = PyDict::new(py);
        globals.set_item("fnp", &module)?;
        globals.set_item("np", &numpy)?;
        let code = std::ffi::CString::new(
            r#"
cov = [[2.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 0.5]]
singular = [[1.0, 1.0], [1.0, 1.0]]
cells = []
for seed in (0, 11, 2024):
    for method in ("marginals", "count"):
        for colors, nsample, size in (([5, 10, 15], 12, 6), ([0, 3, 40, 2], 20, (2, 3)), ([7], 7, None)):
            kw = dict(size=size, method=method)
            cells.append((seed, "multivariate_hypergeometric", (colors, nsample), kw))
    cells.append((seed, "multivariate_normal", ([0.0, 1.0, -2.0], cov), dict(size=4)))
    cells.append((seed, "multivariate_normal", ([0.5, -0.5], singular), dict(size=(2, 2), method="svd")))
    cells.append((seed, "multivariate_normal", ([0.0, 1.0, -2.0], cov), dict(size=3, method="cholesky")))
def run(rng, name, args, kw):
    try:
        value, err = getattr(rng, name)(*args, **kw), None
    except Exception as exc:
        value, err = None, type(exc).__name__
    return value, err, rng.random(2)
bad = []
distinct = set()
for seed, name, args, kw in cells:
    gv, ge, gnext = run(fnp.random.default_rng(seed), name, args, kw)
    wv, we, wnext = run(np.random.default_rng(seed), name, args, kw)
    if we is None:
        distinct.add(np.asarray(wv).tobytes())
    same = (ge == we and np.asarray(gnext).tobytes() == wnext.tobytes()
            and (we is not None or (np.asarray(gv).dtype == wv.dtype
                                    and np.shape(gv) == wv.shape
                                    and np.asarray(gv).tobytes() == wv.tobytes())))
    if not same:
        bad.append(f"{name}{args} {kw} seed={seed}: numpy_err={we} fnp_err={ge}")
result = (len(cells), len(distinct), bad)
"#,
        )
        .expect("script has no NUL");
        py.run(&code, Some(&globals), None)?;
        let (count, distinct, bad): (usize, usize, Vec<String>) = globals
            .get_item("result")?
            .expect("script sets result")
            .extract()?;
        assert_eq!(count, 3 * (2 * 3 + 3), "cell table drifted");
        // Negative control: the seeds and parameters must actually move the draws, or
        // byte equality above would be satisfied by a constant stream. The six
        // single-colour cells (colors=[7], nsample=7) are deterministic by construction and
        // collapse to one value; every other cell must be distinct.
        assert!(
            distinct >= count - 5,
            "only {distinct} distinct numpy draws across {count} cells"
        );
        assert!(
            bad.is_empty(),
            "multivariate draws diverge from numpy: {bad:#?}"
        );
        Ok(())
    });
}

/// Seed-exactness across the Generator surface (47 distribution calls incl. both branches of
/// gamma/binomial/poisson/hypergeometric/vonmises, plus choice/permutation/permuted/shuffle/
/// multinomial/dirichlet/bytes/spawn), RandomState's legacy methods, and every bit generator's
/// raw stream, jumped stream and SeedSequence state - byte-compared with numpy, with the stream
/// position after each Generator draw compared too. The rest of this file compares with
/// allclose, which is why two one-ulp divergences went unseen (bead .8): dirichlet divided by
/// the gamma sum where numpy multiplies by its reciprocal, and vonmises wrapped the angle with
/// rem_euclid where numpy folds |angle| with fmod (and bounds the kappa > 1e6 wrapped normal
/// with one conditional shift). vonmises(0.5, 4.0, size=1000) differed in 4 draws per seed.
#[test]
fn generator_randomstate_and_bit_generator_streams_are_seed_exact_with_numpy() {
    with_fnp_and_numpy(|py, module, numpy| {
        let globals = PyDict::new(py);
        globals.set_item("fnp", &module)?;
        globals.set_item("np", &numpy)?;
        let code = std::ffi::CString::new(
            r#"
GEN = [
    ("random", (), {}), ("random", (), {"dtype": np.float32}),
    ("integers", (0, 10), {}), ("integers", (-5, 2**40), {}), ("integers", (0, 256), {"dtype": np.uint8}),
    ("integers", (0, 7), {"endpoint": True, "dtype": np.int16}),
    ("standard_normal", (), {}), ("standard_normal", (), {"dtype": np.float32}),
    ("normal", (3.0, 2.5), {}), ("uniform", (-2.0, 5.0), {}), ("standard_exponential", (), {}),
    ("standard_exponential", (), {"method": "inv"}), ("exponential", (2.0,), {}),
    ("standard_gamma", (0.5,), {}), ("standard_gamma", (3.0,), {}), ("gamma", (2.0, 1.5), {}),
    ("beta", (0.3, 0.7), {}), ("beta", (2.0, 5.0), {}), ("chisquare", (3.0,), {}), ("f", (5.0, 7.0), {}),
    ("noncentral_chisquare", (3.0, 1.5), {}), ("noncentral_f", (5.0, 7.0, 0.5), {}),
    ("standard_t", (4.0,), {}), ("standard_cauchy", (), {}), ("laplace", (1.0, 2.0), {}),
    ("logistic", (0.5, 1.5), {}), ("lognormal", (0.2, 0.9), {}), ("gumbel", (0.0, 1.2), {}),
    ("weibull", (1.7,), {}), ("pareto", (3.0,), {}), ("power", (2.5,), {}), ("rayleigh", (1.5,), {}),
    ("wald", (1.0, 2.0), {}), ("vonmises", (0.5, 4.0), {}), ("vonmises", (-2.9, 50.0), {}),
    ("vonmises", (0.5, 2e6), {}), ("vonmises", (3.1, 1e-9), {}),
    ("triangular", (-1.0, 0.5, 2.0), {}), ("binomial", (10, 0.3), {}), ("binomial", (1000, 0.6), {}),
    ("negative_binomial", (5, 0.4), {}), ("poisson", (3.5,), {}), ("poisson", (150.0,), {}),
    ("geometric", (0.2,), {}), ("hypergeometric", (20, 30, 15), {}), ("hypergeometric", (2000, 3000, 500), {}),
    ("logseries", (0.7,), {}), ("zipf", (2.5,), {}),
]
EXTRA = [
    lambda g, n: g.choice(100, n), lambda g, n: g.choice(1000, min(n, 1000), replace=False),
    lambda g, n: g.choice(4, n, p=[0.1, 0.2, 0.3, 0.4]), lambda g, n: g.permutation(n),
    lambda g, n: g.permuted(np.arange(n)), lambda g, n: (lambda a: (g.shuffle(a), a)[1])(np.arange(n)),
    lambda g, n: g.multinomial(20, [0.1, 0.4, 0.5], n), lambda g, n: g.dirichlet([0.5, 1.0, 2.0], n),
    lambda g, n: g.dirichlet([5.0] * 6, n), lambda g, n: g.bytes(n),
]
LEGACY = [
    lambda r, n: r.rand(n), lambda r, n: r.randn(n), lambda r, n: r.randint(0, 100, n),
    lambda r, n: r.normal(1, 2, n), lambda r, n: r.standard_gamma(2.5, n), lambda r, n: r.beta(0.5, 0.5, n),
    lambda r, n: r.binomial(20, 0.3, n), lambda r, n: r.poisson(4.0, n), lambda r, n: r.choice(50, n),
    lambda r, n: r.permutation(n), lambda r, n: r.multinomial(10, [0.2, 0.3, 0.5], n),
    lambda r, n: r.hypergeometric(10, 20, 7, n), lambda r, n: r.zipf(3.0, n),
]
def same(a, b):
    if isinstance(b, bytes):
        return a == b
    a, b2 = np.asarray(a), np.asarray(b)
    return type(a) is type(b2) and a.dtype == b2.dtype and a.shape == b2.shape and a.tobytes() == b2.tobytes()
bad = []
cells = 0
distinct = set()
for seed in (0, 12345, 2**40 + 7):
    for size in (None, 1, 7, 1000):
        for name, args, kw in GEN:
            g1, g2 = fnp.random.default_rng(seed), np.random.default_rng(seed)
            r = getattr(g1, name)(*args, size=size, **kw)
            s = getattr(g2, name)(*args, size=size, **kw)
            cells += 1
            distinct.add(np.asarray(s).tobytes())
            if type(r) is not type(s) or not same(r, s) or not same(g1.random(2), g2.random(2)):
                bad.append(f"Generator.{name}{args}{kw} size={size} seed={seed}")
    for n in (1, 7, 500):
        for i, fn in enumerate(EXTRA):
            cells += 1
            if not same(fn(fnp.random.default_rng(seed), n), fn(np.random.default_rng(seed), n)):
                bad.append(f"Generator extra #{i} n={n} seed={seed}")
        for i, fn in enumerate(LEGACY):
            cells += 1
            if not same(fn(fnp.random.RandomState(seed % 2**32), n), fn(np.random.RandomState(seed % 2**32), n)):
                bad.append(f"RandomState #{i} n={n} seed={seed}")
    for bg in ("PCG64", "PCG64DXSM", "MT19937", "Philox", "SFC64"):
        b1, b2 = getattr(fnp.random, bg)(seed), getattr(np.random, bg)(seed)
        cells += 1
        if not same(b1.random_raw(64), b2.random_raw(64)):
            bad.append(f"{bg}.random_raw seed={seed}")
        if hasattr(np.random, bg) and bg != "SFC64":
            cells += 1
            if not same(b1.jumped(3).random_raw(8), b2.jumped(3).random_raw(8)):
                bad.append(f"{bg}.jumped seed={seed}")
    cells += 1
    if not same(fnp.random.SeedSequence(seed).generate_state(8), np.random.SeedSequence(seed).generate_state(8)):
        bad.append(f"SeedSequence seed={seed}")
result = (cells, len(distinct), bad)
"#,
        )
        .expect("script has no NUL");
        py.run(&code, Some(&globals), None)?;
        let (cells, distinct, bad): (usize, usize, Vec<String>) = globals
            .get_item("result")?
            .expect("script sets result")
            .extract()?;
        assert_eq!(cells, 813, "cell table drifted");
        // Negative control: seeds, sizes and parameters must move the draws, or byte equality
        // would be satisfied by a constant stream (421 distinct under numpy 2.4.3; draws that
        // coincide are the size=None / size=1 pairs and degenerate parameters).
        assert!(
            distinct >= 400,
            "only {distinct} distinct numpy Generator draws across the table"
        );
        assert!(
            bad.is_empty(),
            "random streams diverge from numpy: {bad:#?}"
        );
        Ok(())
    });
}

/// numpy's `advance(delta)` on PCG64, PCG64DXSM and Philox (bead .30): for seeds {0, 12345,
/// 2**40 + 7} and deltas {0, 1, 12345, 2**64 + 3, 2**127, -1} (and 2**200 on Philox's 256-bit
/// counter), `advance(d).random_raw(8)` must equal numpy's word for word, `advance` must return
/// the SAME object, and a uint32 buffered by `Generator.integers(..., dtype=uint32)` before an
/// advance must not leak after it. `delta` goes through numpy's own `delta & mask`, so a NumPy
/// integer is numpy's OverflowError and a float its TypeError. And the method SURFACE must be
/// numpy's: `hasattr(SFC64(0), "jumped")` and `hasattr(MT19937(0), "advance")` are False there
/// (fnp's shared pyclass used to give every generator `jumped`, and none of them `advance`).
#[test]
fn bit_generator_advance_matches_numpy_streams_and_surface() {
    with_fnp_and_numpy(|py, module, numpy| {
        let globals = PyDict::new(py);
        globals.set_item("fnp", &module)?;
        globals.set_item("np", &numpy)?;
        let code = std::ffi::CString::new(
            r#"
bad = []
cells = 0
for name in ("PCG64", "PCG64DXSM", "Philox"):
    deltas = [0, 1, 12345, 2**64 + 3, 2**127, -1] + ([2**200] if name == "Philox" else [])
    for seed in (0, 12345, 2**40 + 7):
        for d in deltas:
            ours, theirs = getattr(fnp.random, name)(seed), getattr(np.random, name)(seed)
            if ours.advance(d) is not ours:
                bad.append(f"{name}({seed}).advance({d}) did not return self")
            cells += 1
            if ours.random_raw(8).tolist() != theirs.advance(d).random_raw(8).tolist():
                bad.append(f"{name}({seed}).advance({d})")
        ours, theirs = getattr(fnp.random, name)(seed), getattr(np.random, name)(seed)
        og, tg = fnp.random.Generator(ours), np.random.Generator(theirs)
        first = (og.integers(0, 2**32, dtype=np.uint32), tg.integers(0, 2**32, dtype=np.uint32))
        ours.advance(5)
        theirs.advance(5)
        after = (og.integers(0, 2**32, dtype=np.uint32), tg.integers(0, 2**32, dtype=np.uint32))
        cells += 1
        if int(first[0]) != int(first[1]) or int(after[0]) != int(after[1]):
            bad.append(f"{name}({seed}) buffered uint32 across advance: {first} {after}")
    for d in (np.int64(5), 1.5, "5"):
        def outcome(module):
            try:
                getattr(module.random, name)(1).advance(d)
                return "ok"
            except Exception as ex:
                return type(ex).__name__
        cells += 1
        if outcome(fnp) != outcome(np):
            bad.append(f"{name}.advance({d!r}): fnp={outcome(fnp)} numpy={outcome(np)}")
for name in ("MT19937", "PCG64", "PCG64DXSM", "Philox", "SFC64"):
    for attr in ("jumped", "advance"):
        cells += 1
        mine = hasattr(getattr(fnp.random, name)(0), attr)
        if mine != hasattr(getattr(np.random, name)(0), attr):
            bad.append(f"hasattr({name}(0), {attr!r}) = {mine}")
result = (cells, bad)
"#,
        )
        .expect("script has no NUL");
        py.run(&code, Some(&globals), None)?;
        let (cells, bad): (usize, Vec<String>) = globals
            .get_item("result")?
            .expect("script sets result")
            .extract()?;
        assert_eq!(cells, 85, "cell table drifted");
        assert!(bad.is_empty(), "advance diverges from numpy: {bad:#?}");
        Ok(())
    });
}

/// Divergences numpy's own random test suites found through the drop-in harness (bead
/// rc0923 .8). Every cell compares fnp's full outcome with numpy's: value bytes, or the
/// exception type and message, plus the warnings raised. On the pre-fix build 58 of the 67
/// cells failed. Examples:
/// - `shuffle` of a list raised AttributeError (numpy shuffles any mutable sequence in place).
/// - `choice(n, p=float32_softmax)` raised "upper_bound must be > 0". numpy widens its sum
///   tolerance to the p dtype's sqrt(eps), and searches a normalized cdf.
/// - `integers(dtype='>i4')` and `random(dtype='>f8')` were accepted as native order.
/// - `default_rng(np.array([1, 2, 3]))` raised TypeError.
/// - `uniform(-1e308, 1e308)` drew infinities.
/// - `standard_gamma(dtype=float32)` was refused.
///
/// The `p32` cells are chosen so that draws land between the raw and the normalized cdf
/// boundary, which a raw-cumsum search gets wrong.
#[test]
fn generator_shuffle_choice_dtype_uniform_mvhg_and_seed_surfaces_match_numpy() {
    with_fnp_and_numpy(|py, module, numpy| {
        let globals = PyDict::new(py);
        globals.set_item("fnp", &module)?;
        globals.set_item("np", &numpy)?;
        let code = std::ffi::CString::new(
            r#"
import warnings

def outcome(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = call()
            if isinstance(value, np.ndarray):
                got = ("ok", value.dtype.str, value.shape, value.tobytes())
            elif isinstance(value, np.generic):
                got = ("ok", value.dtype.str, repr(value))
            else:
                got = ("ok", repr(value))
        except Exception as ex:
            got = (type(ex).__name__, str(ex))
    return got + (sorted((w.category.__name__, str(w.message)) for w in caught),)

def shuffled(m, x, **kw):
    m.random.default_rng(7).shuffle(x, **kw)
    return x

logits = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
softmax32 = np.exp(logits) / np.exp(logits).sum()
p32 = np.array([0.5, 0.4998], dtype=np.float32)
cases = {
    # shuffle: numpy's untyped path for any mutable sequence, same draws as the array path.
    "shuffle list": lambda m: shuffled(m, list(range(10))),
    "shuffle nested list": lambda m: shuffled(m, [[1, 2], [3, 4], [5, 6]]),
    "shuffle list then draw": lambda m: (lambda g: (g.shuffle(list(range(50))), g.integers(0, 1000, 5).tolist()))(m.random.default_rng(11)),
    "shuffle bytearray": lambda m: bytes(shuffled(m, bytearray(b"abcdef"))),
    "shuffle tuple": lambda m: shuffled(m, (1, 2, 3)),
    "shuffle list axis=1": lambda m: shuffled(m, [[1, 2], [3, 4]], axis=1),
    "shuffle dict warns": lambda m: shuffled(m, {0: "a", 1: "b", 2: "c"}),
    "shuffle int": lambda m: shuffled(m, 5),
    # choice: numpy's p checks, messages and float32/float16 tolerance; normalized cdf.
    "choice softmax32": lambda m: m.random.default_rng(1).choice(1000, size=50, p=softmax32),
    "choice softmax32 no replace": lambda m: m.random.default_rng(1).choice(1000, size=50, replace=False, p=softmax32),
    "choice p32 window 1e5": lambda m: m.random.default_rng(2).choice(2, size=100000, p=p32),
    "choice p32 size-1": lambda m: [int(m.random.default_rng(s).choice(2, p=p32)) for s in range(300)],
    "choice p16 array a": lambda m: m.random.default_rng(1).choice([7, 8, 9], size=20, p=np.array([0.33, 0.33, 0.33], dtype=np.float16)),
    "choice p nan": lambda m: m.random.default_rng(1).choice(3, p=[0.5, np.nan, 0.5]),
    "choice p inf -inf": lambda m: m.random.default_rng(1).choice(2, p=[np.inf, -np.inf]),
    "choice p negative": lambda m: m.random.default_rng(1).choice(3, p=[0.5, -0.1, 0.6]),
    "choice p sum": lambda m: m.random.default_rng(1).choice(3, p=[0.5, 0.1, 0.1]),
    "choice p 2-D": lambda m: m.random.default_rng(1).choice(4, p=[[0.25, 0.25], [0.25, 0.25]]),
    "choice p scalar": lambda m: m.random.default_rng(1).choice(3, p=1.0),
    "choice too big": lambda m: m.random.default_rng(1).choice(3, 4, replace=False),
    "choice array too big": lambda m: m.random.default_rng(1).choice([1, 2, 3], 4, replace=False),
    "choice fewer nonzero": lambda m: m.random.default_rng(1).choice(3, 3, replace=False, p=[0.5, 0.5, 0.0]),
    "choice empty a": lambda m: m.random.default_rng(1).choice([], 2),
    "choice float a": lambda m: m.random.default_rng(3).choice(5.0),
    "choice p ok f64": lambda m: m.random.default_rng(3).choice(5, 8, p=[0.1, 0.2, 0.3, 0.2, 0.2]),
    # dtypes: numpy's float32 standard_gamma; non-native byte orders refused like numpy.
    "standard_gamma f32": lambda m: m.random.default_rng(3).standard_gamma(1.0, size=4, dtype=np.float32),
    "standard_gamma f32 then draw": lambda m: (lambda g: (g.standard_gamma(0.5, 3, dtype=np.float32), g.random(2))[1])(m.random.default_rng(3)),
    "standard_gamma i4": lambda m: m.random.default_rng(3).standard_gamma(1.0, dtype=np.int32),
    "integers >i4": lambda m: m.random.default_rng(3).integers(0, 10, size=3, dtype=">i4"),
    "integers f8": lambda m: m.random.default_rng(3).integers(0, 5, dtype=np.float64),
    "randint f8": lambda m: m.random.RandomState(3).randint(0, 5, dtype=np.float64),
    "random >f8": lambda m: m.random.default_rng(3).random(3, dtype=">f8"),
    "random i4": lambda m: m.random.default_rng(3).random(3, dtype=np.int32),
    "standard_normal >f8": lambda m: m.random.default_rng(3).standard_normal(3, dtype=">f8"),
    "generate_state >u4": lambda m: m.random.SeedSequence(5).generate_state(2, dtype=">u4"),
    "generate_state None": lambda m: m.random.SeedSequence(5).generate_state(2, dtype=None),
    "generate_state u8": lambda m: m.random.SeedSequence(5).generate_state(2, dtype="u8"),
    # uniform: a non-finite range is OverflowError.
    "uniform huge range": lambda m: m.random.default_rng(3).uniform(-1e308, 1e308),
    "uniform inf": lambda m: m.random.default_rng(3).uniform(0, np.inf),
    "uniform nan": lambda m: m.random.default_rng(3).uniform(0, np.nan),
    "uniform negative range": lambda m: m.random.default_rng(3).uniform(1, 0),
    "legacy uniform inf": lambda m: m.random.RandomState(3).uniform(-np.inf, np.inf),
    "legacy uniform negative range": lambda m: m.random.RandomState(3).uniform(1, 0, 3),
    # multivariate_hypergeometric: numpy's colors / nsample validation.
    "mvhg negative color": lambda m: m.random.default_rng(3).multivariate_hypergeometric([3, -1], 2),
    "mvhg 2-D colors": lambda m: m.random.default_rng(3).multivariate_hypergeometric([[3, 4]], 2),
    "mvhg float colors": lambda m: m.random.default_rng(3).multivariate_hypergeometric([3.0, 4.0], 2),
    "mvhg empty colors": lambda m: m.random.default_rng(3).multivariate_hypergeometric([], 0, size=3),
    "mvhg nsample float": lambda m: m.random.default_rng(3).multivariate_hypergeometric([3, 4], 2.5),
    "mvhg nsample negative": lambda m: m.random.default_rng(3).multivariate_hypergeometric([3, 4], -1),
    "mvhg nsample > total": lambda m: m.random.default_rng(3).multivariate_hypergeometric([3, 4], 8),
    "mvhg marginals 1e9": lambda m: m.random.default_rng(3).multivariate_hypergeometric([10**9, 1], 1),
    "mvhg count": lambda m: m.random.default_rng(3).multivariate_hypergeometric([3, 4, 5], 6, size=(2, 2), method="count"),
    # seeds: numpy's _coerce_to_uint32_array and SeedSequence's entropy gate.
    "seed int64 array": lambda m: m.random.default_rng(np.array([1, 2, 3])).integers(0, 2**62, 4),
    "seed 2-D array": lambda m: m.random.default_rng(np.array([[1, 2], [3, 4]])).integers(0, 2**62, 4),
    "seed empty array": lambda m: m.random.default_rng(np.array([], dtype=np.int64)).integers(0, 2**62, 4),
    "seed uint64 array": lambda m: m.random.default_rng(np.array([2**63 + 5], dtype=np.uint64)).integers(0, 2**62, 4),
    "seed 0-d array": lambda m: m.random.default_rng(np.array(5)).integers(0, 2**62, 4),
    "seed negative array": lambda m: m.random.default_rng(np.array([1, -2])).integers(0, 2**62, 4),
    "seed float array": lambda m: m.random.default_rng(np.array([1.0, 2.0])).integers(0, 2**62, 4),
    "seed bool array": lambda m: m.random.default_rng(np.array([True, False])).integers(0, 2**62, 4),
    "MT19937 int8 array": lambda m: m.random.Generator(m.random.MT19937(np.array([1, 2], dtype=np.int8))).integers(0, 2**62, 4),
    "SeedSequence float": lambda m: m.random.SeedSequence(1.5).generate_state(2),
    "SeedSequence str": lambda m: m.random.SeedSequence("5").generate_state(2),
    "SeedSequence str list": lambda m: m.random.SeedSequence(["010", "0x10", "9"]).generate_state(2),
    "SeedSequence bad str": lambda m: m.random.SeedSequence(["x1"]).generate_state(2),
    "SeedSequence uint32 2-D": lambda m: m.random.SeedSequence(np.array([[1, 2], [3, 4]], dtype=np.uint32)).generate_state(2),
    "SeedSequence >u4": lambda m: m.random.SeedSequence(np.array([1, 2], dtype=">u4")).generate_state(2),
}
bad = []
for name, case in cases.items():
    ours, theirs = outcome(lambda: case(fnp)), outcome(lambda: case(np))
    if ours != theirs:
        bad.append(f"{name}: fnp={str(ours)[:160]} numpy={str(theirs)[:160]}")
result = (len(cases), bad)
"#,
        )
        .expect("script has no NUL");
        py.run(&code, Some(&globals), None)?;
        let (cells, bad): (usize, Vec<String>) = globals
            .get_item("result")?
            .expect("script sets result")
            .extract()?;
        assert_eq!(cells, 67, "cell table drifted");
        assert!(
            bad.is_empty(),
            "random surfaces diverge from numpy: {bad:#?}"
        );
        Ok(())
    });
}

/// Second round of numpy's own random suites through the drop-in harness (bead rc0923 .8).
/// Every cell compares fnp's full outcome with numpy's (MaskedArray data and mask; object
/// arrays by value). 37 of these 46 cells failed on 3d00b659. The failures included:
/// - `choice([None], size=())` returned the bare element instead of a 0-d array;
/// - `shuffle` of a MaskedArray left its mask behind;
/// - `randint(10)` returned np.int64 where numpy returns a Python int;
/// - `multivariate_hypergeometric(method='marginals')` left numpy's stream after the first
///   variate;
/// - the shuffle/permutation axis errors were plain ValueErrors instead of AxisError;
/// - `set_state(())` raised ValueError instead of IndexError.
#[test]
fn generator_and_random_state_round_two_surfaces_match_numpy() {
    with_fnp_and_numpy(|py, module, numpy| {
        let globals = PyDict::new(py);
        globals.set_item("fnp", &module)?;
        globals.set_item("np", &numpy)?;
        let code = std::ffi::CString::new(
            r#"
import warnings

def outcome(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = call()
            if isinstance(value, np.ma.MaskedArray):
                got = ("masked", value.dtype.str, value.data.tobytes(), np.ma.getmaskarray(value).tobytes())
            elif isinstance(value, np.ndarray) and value.dtype == object:
                got = ("ok", "O", value.shape, repr(value.tolist()))
            elif isinstance(value, np.ndarray):
                got = ("ok", value.dtype.str, value.shape, value.tobytes())
            elif isinstance(value, np.generic):
                got = ("ok", value.dtype.str, repr(value))
            else:
                got = ("ok", type(value).__name__, repr(value))
        except Exception as ex:
            got = (type(ex).__name__, str(ex))
    return got + (sorted((w.category.__name__, str(w.message)) for w in caught),)

def address_form(text):
    # `<name>(<bitgen>) at 0x<HEX>`: compare everything but the address digits themselves.
    head, _, address = text.partition(" at ")
    return (head, address[:2], address[2:] == address[2:].upper() and len(address) > 2)

def shuffled(m, x, **kw):
    m.random.default_rng(7).shuffle(x, **kw)
    return x

def masked(m):
    a = np.ma.masked_values(np.reshape(range(20), (5, 4)) % 3 - 1, -1)
    m.random.default_rng(4).shuffle(a)
    return a

def read_only():
    a = np.arange(5)
    a.flags.writeable = False
    return a

class ThrowingInteger(np.ndarray):
    def __int__(self):
        raise TypeError("no int")

obj_array = np.empty(1, dtype=object)
obj_array[0] = np.array([1, 2])
x32 = np.array([9.9e-01, 9.9e-01] + [1.0e-09] * 8, dtype=np.float32)
cases = {
    # choice: numpy's size=None / size=() tails.
    "choice int size=()": lambda m: m.random.default_rng(3).choice(5, ()),
    "choice array size=()": lambda m: m.random.default_rng(3).choice(np.arange(5) * 10, ()),
    "choice object size=()": lambda m: m.random.default_rng(3).choice([None], ()),
    "choice object scalar": lambda m: m.random.default_rng(3).choice([None]),
    "choice object array element": lambda m: m.random.default_rng(3).choice(obj_array),
    "choice p no-replace scalar": lambda m: m.random.default_rng(3).choice(2, replace=False, p=[0.1, 0.9]),
    "choice 2-D size=()": lambda m: m.random.default_rng(3).choice(np.arange(6).reshape(3, 2), ()),
    # shuffle / permutation: subclass writes, AxisError, read-only, numpy's int rule.
    "shuffle masked": lambda m: masked(m),
    "shuffle axis error": lambda m: shuffled(m, np.arange(10), axis=1),
    "shuffle read-only": lambda m: shuffled(m, read_only()),
    "shuffle 0-d": lambda m: shuffled(m, np.array(3)),
    "permutation str": lambda m: m.random.default_rng(3).permutation("abcd"),
    "permutation negative int": lambda m: m.random.default_rng(3).permutation(-3),
    "permutation 0-d array": lambda m: m.random.default_rng(3).permutation(np.array(5)),
    "permutation axis error": lambda m: m.random.default_rng(3).permutation(np.arange(9).reshape(3, 3), 3),
    "permutation np.int8": lambda m: m.random.default_rng(3).permutation(np.int8(6)),
    # integers / randint: zero size first, numpy's bounds messages, Python ints for dtype=int.
    "randint default scalar type": lambda m: m.random.RandomState(1).randint(10),
    "randint dtype=int": lambda m: m.random.RandomState(1).randint(0, 10, dtype=int),
    "randint dtype=None": lambda m: m.random.RandomState(1).randint(0, 10, dtype=None),
    "randint zero size": lambda m: m.random.RandomState(1).randint(0, 0, size=(3, 0, 4)),
    "randint low >= high": lambda m: m.random.RandomState(1).randint(5, 3),
    "randint high <= 0": lambda m: m.random.RandomState(1).randint(0),
    "integers high <= 0": lambda m: m.random.default_rng(1).integers(-2),
    "integers high < 0": lambda m: m.random.default_rng(1).integers(-2, endpoint=True),
    "integers low >= high": lambda m: m.random.default_rng(1).integers(5, 3),
    "integers low > high": lambda m: m.random.default_rng(1).integers(5, 3, endpoint=True),
    "integers zero size bad bounds": lambda m: m.random.default_rng(1).integers(5, 3, size=0),
    "integers dtype=int": lambda m: m.random.default_rng(1).integers(0, 10, dtype=int),
    "integers dtype=int endpoint": lambda m: m.random.default_rng(1).integers(-2**63, 2**63 - 1, dtype=int, endpoint=True),
    # multivariate_hypergeometric marginals: numpy's loop (remainder to the last color, complement).
    "mvhg marginals": lambda m: m.random.Generator(m.random.MT19937(8675309)).multivariate_hypergeometric([20, 30, 50], 50, size=5, method="marginals"),
    "mvhg marginals > half": lambda m: m.random.Generator(m.random.MT19937(8675309)).multivariate_hypergeometric([20, 30, 50], 60, size=3, method="marginals"),
    # multinomial: Kahan sum and numpy's float32 message.
    "multinomial float32 pvals": lambda m: m.random.Generator(m.random.MT19937(1432985819)).multinomial(1, x32 / x32.sum()),
    # constructors, attributes, reprs, state.
    "Generator(class)": lambda m: m.random.Generator(m.random.MT19937),
    "Generator(int)": lambda m: m.random.Generator(5),
    "Generator _poisson_lam_max": lambda m: m.random.default_rng(1)._poisson_lam_max,
    "RandomState _poisson_lam_max": lambda m: m.random.RandomState(1)._poisson_lam_max,
    "Generator repr": lambda m: address_form(repr(m.random.Generator(m.random.PCG64(1)))),
    "Generator str": lambda m: str(m.random.Generator(m.random.PCG64(1))),
    "RandomState repr": lambda m: address_form(repr(m.random.RandomState(1))),
    "RandomState str": lambda m: str(m.random.RandomState(1)),
    "set_state ()": lambda m: m.random.RandomState(1).set_state(()),
    "set_state short": lambda m: m.random.RandomState(1).set_state(("MT19937",)),
    "set_state list": lambda m: (lambda r: (r.set_state(list(np.random.RandomState(2).get_state())), r.random_sample(3))[1])(m.random.RandomState(1)),
    "set_state int": lambda m: m.random.RandomState(1).set_state(5),
    "set_state bad name": lambda m: m.random.RandomState(1).set_state(("PCG64", np.zeros(624, dtype=np.uint32), 0)),
    "hypergeometric throwing int": lambda m: m.random.default_rng(1).hypergeometric(np.array(1).view(ThrowingInteger), 1, 1),
}
bad = []
for name, case in cases.items():
    ours, theirs = outcome(lambda: case(fnp)), outcome(lambda: case(np))
    if ours != theirs:
        bad.append(f"{name}: fnp={str(ours)[:160]} numpy={str(theirs)[:160]}")
result = (len(cases), bad)
"#,
        )
        .expect("script has no NUL");
        py.run(&code, Some(&globals), None)?;
        let (cells, bad): (usize, Vec<String>) = globals
            .get_item("result")?
            .expect("script sets result")
            .extract()?;
        assert_eq!(cells, 46, "cell table drifted");
        assert!(
            bad.is_empty(),
            "random surfaces diverge from numpy: {bad:#?}"
        );
        Ok(())
    });
}

/// The RNG objects under threads (bead rc0923 .8), as numpy's TestThread and ordinary
/// thread-pool code use them.
/// - Generator, bit generators and SeedSequence were `unsendable`, so a Generator built on
///   one thread and used on another raised PanicException.
/// - A RandomState or Generator SHARED by threads raised "RuntimeError: Already borrowed"
///   whenever one call released the GIL mid-method. That was 287 of 320 module-level
///   `np.random.*` calls from 8 threads.
/// - A `frompyfunc` object used from another thread panicked.
///
/// numpy serializes each object on its lock, so the shared RandomState must produce exactly
/// the serial stream. The re-entrant case has no numpy oracle: numpy's Lock deadlocks there,
/// and fnp must raise instead of hanging. The pre-fix build failed 9 of these checks.
#[test]
fn rng_objects_are_usable_and_serialized_across_threads() {
    with_fnp_and_numpy(|py, module, numpy| {
        let globals = PyDict::new(py);
        globals.set_item("fnp", &module)?;
        globals.set_item("np", &numpy)?;
        let code = std::ffi::CString::new(
            r#"
import threading
from concurrent.futures import ThreadPoolExecutor

bad = []

def on_threads(count, work):
    """Run work(i) on `count` threads; collect every exception, BaseException included."""
    errors = []
    def run(i):
        try:
            work(i)
        except BaseException as ex:
            errors.append(f"{type(ex).__name__}: {str(ex)[:80]}")
    threads = [threading.Thread(target=run, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors

# 1. numpy's TestThread shape: one Generator per thread, all built on this thread.
for name, draw in (
    ("normal", lambda g: g.normal(size=20000)),
    ("exponential", lambda g: g.exponential(scale=np.ones((50, 400)))),
    ("multinomial", lambda g: g.multinomial(10, [1 / 6.0] * 6, size=5000)),
):
    outputs = {}
    for label, m in (("fnp", fnp), ("numpy", np)):
        gens = [m.random.Generator(m.random.MT19937(seed)) for seed in range(4)]
        out = [None] * 4
        def work(i, gens=gens, out=out):
            out[i] = draw(gens[i])
        errors = on_threads(4, work)
        if errors:
            bad.append(f"per-thread {name} ({label}): {errors[:2]}")
        outputs[label] = out
    if not all(o is not None and t is not None and np.array_equal(o, t)
               for o, t in zip(outputs["fnp"], outputs["numpy"])):
        bad.append(f"per-thread {name}: values differ from numpy")

# 2. ONE RandomState shared by 8 threads. numpy serializes every call on its lock, so none
#    fails, and with fixed-consumption draws the final state and the multiset of values are
#    exactly those of the same calls made serially.
calls, per_call = 8 * 25, 1000
serial = np.random.RandomState(2024)
expected = np.sort(np.concatenate([serial.random_sample(per_call) for _ in range(calls)]))
shared = fnp.random.RandomState(2024)
chunks = []
chunks_lock = threading.Lock()
def work(i):
    for _ in range(25):
        chunk = shared.random_sample(per_call)
        with chunks_lock:
            chunks.append(chunk)
errors = on_threads(8, work)
if errors:
    bad.append(f"shared RandomState: {len(errors)} failures, e.g. {errors[:2]}")
elif not np.array_equal(np.sort(np.concatenate(chunks)), expected):
    bad.append("shared RandomState: the draws are not the serial stream")
else:
    ours, theirs = shared.get_state(), serial.get_state()
    if not (np.array_equal(ours[1], theirs[1]) and ours[2] == theirs[2]):
        bad.append("shared RandomState: final state differs from the serial one")

# 3. The module-level legacy functions (all bound to one RandomState) and a shared Generator,
#    mixing native and numpy-delegated methods: no call may fail.
rng = fnp.random.default_rng(7)
mixed = [
    lambda: fnp.random.rand(20000),
    lambda: fnp.random.normal(size=20000),
    lambda: fnp.random.randint(0, 100, 20000),
    lambda: fnp.random.gamma(2.0, size=20000),
    lambda: fnp.random.shuffle(np.arange(50000)),
    lambda: rng.normal(size=20000),
    lambda: rng.integers(0, 100, 20000),
    lambda: rng.standard_gamma(2.0, 20000, dtype=np.float32),
    lambda: rng.choice(1000, 2000, p=np.full(1000, 1e-3)),
    lambda: rng.permutation(50000),
]
def work(i):
    for j in range(20):
        mixed[(i + j) % len(mixed)]()
errors = on_threads(8, work)
if errors:
    bad.append(f"module functions / shared Generator: {len(errors)} failures, e.g. {errors[:2]}")

# 4. A frompyfunc object built here and called from other threads.
add_one = fnp.frompyfunc(lambda x: x + 1, 1, 1)
out = [None] * 4
def work(i):
    out[i] = add_one(np.arange(100) + i)
errors = on_threads(4, work)
if errors or any(o is None or o.tolist() != list(range(1 + i, 101 + i)) for i, o in enumerate(out)):
    bad.append(f"frompyfunc across threads: {errors[:2]}")

# 5. A call back into the RandomState from inside one of its own methods raises instead of
#    hanging (numpy's non-reentrant Lock would deadlock here, so there is no numpy oracle).
class Hostile(list):
    def __setitem__(self, index, value):
        reentrant.random_sample()
        super().__setitem__(index, value)
reentrant = fnp.random.RandomState(1)
try:
    reentrant.shuffle(Hostile(range(5)))
    bad.append("re-entrant call did not raise")
except RuntimeError as ex:
    if "already in use by the calling thread" not in str(ex):
        bad.append(f"re-entrant call raised the wrong RuntimeError: {ex}")
reentrant.random_sample()  # the lock was released
result = bad
"#,
        )
        .expect("script has no NUL");
        py.run(&code, Some(&globals), None)?;
        let bad: Vec<String> = globals
            .get_item("result")?
            .expect("script sets result")
            .extract()?;
        assert!(
            bad.is_empty(),
            "RNG objects misbehave across threads: {bad:#?}"
        );
        Ok(())
    });
}
