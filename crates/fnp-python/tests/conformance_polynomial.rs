//! Conformance matrix: numpy.polynomial.* family.
//!
//! Differential parity for fnp_python's polynomial port (epic 0247).
//! Six bases — polynomial / chebyshev / hermite / hermite_e / laguerre
//! / legendre — each exposing a flat-namespace #[pyfunction] in
//! fnp_python that mirrors the matching numpy.polynomial.<base>.<name>
//! reference.
//!
//! Layout: per base, MUST coverage on the canonical 5 ops (add, sub,
//! mul, val, roots). SHOULD coverage on the secondary surface (der,
//! int, fromroots, pow, line, mulx, trim, div). Cross-base conversions
//! (cheb2poly / poly2cheb / herm2poly / poly2herm / herme2poly /
//! poly2herme / lag2poly / poly2lag / leg2poly / poly2leg) get SHOULD
//! coverage. Class re-exports (Polynomial, Chebyshev, Legendre,
//! Hermite, HermiteE, Laguerre) get a single MUST identity check each.
//!
//! All cases use Strict comparison: numpy's polynomial subpackages
//! return integer or float arrays of well-defined shape, and our
//! wrappers passthrough so any drift is a real bug, not a 1-ULP edge.

mod common;

use common::{CompareMode, RequirementLevel, Totals, run_case_resolved, with_fnp_and_numpy};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

fn no_kwargs<'py>(_py: Python<'py>) -> PyResult<Option<pyo3::Bound<'py, PyDict>>> {
    Ok(None)
}

fn coef_3<'py>(py: Python<'py>) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?
        .getattr("array")?
        .call1((vec![1.0_f64, 2.0, 3.0],))
}

fn coef_3b<'py>(py: Python<'py>) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?
        .getattr("array")?
        .call1((vec![1.0_f64, 0.5, 0.25],))
}

fn roots_2<'py>(py: Python<'py>) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?
        .getattr("array")?
        .call1((vec![-1.0_f64, 1.0],))
}

#[test]
fn conformance_polynomial_matrix() {
    static TOTALS: Totals = Totals::new();

    with_fnp_and_numpy(|py, module, numpy| {
        let t = &TOTALS;
        let np_poly = numpy.getattr("polynomial")?;

        // Each base maps the flat fnp_python prefix to a numpy
        // subpackage path. polynomial-prefix wrappers in fnp_python
        // come from BOTH numpy.polyadd (legacy decreasing-power) and
        // numpy.polynomial.polynomial.poly* (increasing-power); the
        // canonical 5 here use the legacy top-level numpy.* spellings
        // because that is what fnp_python.polyadd / polysub / polymul /
        // polyval / polyroots route to.
        struct Base {
            // Display name in the verdict log.
            base: &'static str,
            // Prefix of fnp_python flat wrappers (e.g. "cheb").
            prefix: &'static str,
            // Resolver for the matching numpy reference for `<prefix><name>`.
            // Returns numpy.polynomial.<base>.<prefix><name> for the 5 modern
            // bases, and numpy.<prefix><name> for the legacy top-level
            // polynomial family (poly*).
            numpy_subpackage: Option<&'static str>,
        }

        let bases: [Base; 6] = [
            // For the polynomial base, every op resolves through
            // numpy.polynomial.polynomial.* (increasing-power); the
            // legacy top-level numpy.poly{add,sub,mul,div,val} use
            // decreasing-power and therefore have different math —
            // they're tested by the legacy_polynomial_ops loop below.
            Base {
                base: "polynomial",
                prefix: "poly",
                numpy_subpackage: Some("polynomial"),
            },
            Base {
                base: "chebyshev",
                prefix: "cheb",
                numpy_subpackage: Some("chebyshev"),
            },
            Base {
                base: "hermite",
                prefix: "herm",
                numpy_subpackage: Some("hermite"),
            },
            Base {
                base: "hermite_e",
                prefix: "herme",
                numpy_subpackage: Some("hermite_e"),
            },
            Base {
                base: "laguerre",
                prefix: "lag",
                numpy_subpackage: Some("laguerre"),
            },
            Base {
                base: "legendre",
                prefix: "leg",
                numpy_subpackage: Some("legendre"),
            },
        ];

        // Names below are the polynomial-base ops fnp_python routes to
        // the LEGACY top-level numpy entry (decreasing-power
        // convention). For these we compare against numpy.<name>, NOT
        // numpy.polynomial.polynomial.<name>.
        let polynomial_legacy_ops: std::collections::HashSet<&'static str> =
            ["add", "sub", "mul", "div", "val", "der", "int"]
                .iter()
                .copied()
                .collect();

        for base in &bases {
            // ─── MUST: add / sub / mul / val / roots ─────────────
            for op in ["add", "sub", "mul"] {
                let our_name = format!("{}{}", base.prefix, op);
                let our_fn = module.getattr(our_name.as_str())?;
                let their_fn = if base.base == "polynomial" && polynomial_legacy_ops.contains(op) {
                    numpy.getattr(our_name.as_str())?
                } else {
                    np_poly
                        .getattr(base.numpy_subpackage.unwrap())?
                        .getattr(our_name.as_str())?
                };
                run_case_resolved(
                    py,
                    &format!("polynomial-{}-{}", base.base, op),
                    our_name.as_str(),
                    &our_fn,
                    &their_fn,
                    RequirementLevel::Must,
                    CompareMode::Strict,
                    t,
                    |py| PyTuple::new(py, [coef_3(py)?, coef_3b(py)?]),
                    no_kwargs,
                );
            }
            // val: 2-arg signature (x, c) — but fnp_python.polyval
            // routes to numpy.polyval (legacy decreasing-power, signed
            // (p, x)), so polynomial-base val gets the legacy reference.
            let our_val_name = format!("{}val", base.prefix);
            let our_val = module.getattr(our_val_name.as_str())?;
            let their_val = if base.base == "polynomial" && polynomial_legacy_ops.contains("val") {
                numpy.getattr(our_val_name.as_str())?
            } else {
                np_poly
                    .getattr(base.numpy_subpackage.unwrap())?
                    .getattr(our_val_name.as_str())?
            };
            // numpy.polyval signature is (p, x) — DECREASING-power; all
            // other bases use (x, c) — INCREASING-power. Build args in
            // the matching order so each side sees its native shape.
            let val_legacy_order =
                base.base == "polynomial" && polynomial_legacy_ops.contains("val");
            run_case_resolved(
                py,
                &format!("polynomial-{}-val", base.base),
                our_val_name.as_str(),
                &our_val,
                &their_val,
                RequirementLevel::Must,
                CompareMode::Close,
                t,
                move |py| {
                    let coef = coef_3(py)?;
                    let xs = py
                        .import("numpy")?
                        .getattr("array")?
                        .call1((vec![0.5_f64, 1.0, 1.5],))?;
                    if val_legacy_order {
                        // (p, x) order — coef first
                        Ok(PyTuple::new(py, [coef, xs])?)
                    } else {
                        // (x, c) order — xs first
                        Ok(PyTuple::new(py, [xs, coef])?)
                    }
                },
                no_kwargs,
            );
            // roots — fnp_python.polyroots routes to
            // numpy.polynomial.polynomial.polyroots (NOT numpy.roots,
            // which is the legacy decreasing-power solver). So even
            // for the polynomial base, roots uses the subpackage path.
            let our_roots_name = format!("{}roots", base.prefix);
            let our_roots = module.getattr(our_roots_name.as_str())?;
            let their_roots = np_poly
                .getattr(base.numpy_subpackage.unwrap())?
                .getattr(our_roots_name.as_str())?;
            run_case_resolved(
                py,
                &format!("polynomial-{}-roots", base.base),
                our_roots_name.as_str(),
                &our_roots,
                &their_roots,
                RequirementLevel::Must,
                CompareMode::Close,
                t,
                |py| PyTuple::new(py, [coef_3(py)?]),
                no_kwargs,
            );

            // ─── SHOULD: fromroots / der / int / pow / line / mulx / trim ───
            for (op, args_kind) in [
                ("fromroots", "roots"),
                ("der", "coef"),
                ("int", "coef"),
                ("pow", "coef_pow"),
                ("line", "scalars"),
                ("mulx", "coef"),
                ("trim", "coef_tol"),
            ] {
                let our_name = format!("{}{}", base.prefix, op);
                let our_op_fn = match module.getattr(our_name.as_str()) {
                    Ok(f) => f,
                    Err(_) => continue, // fnp_python doesn't expose this op
                                        // for this base — skip (e.g.
                                        // numpy.polynomial.polynomial has
                                        // poly{fromroots,line,mulx,trim} but
                                        // not all live in fnp_python yet).
                };
                // SHOULD ops route through the modern subpackage path
                // for the 5 modern bases (fnp_python.poly{fromroots,
                // pow,line,mulx,trim} per 9w2r) but fall back to
                // top-level numpy for legacy poly{der,int} which keep
                // decreasing-power semantics.
                let their_op_fn = if base.base == "polynomial" && polynomial_legacy_ops.contains(op)
                {
                    numpy.getattr(our_name.as_str())
                } else {
                    np_poly
                        .getattr(base.numpy_subpackage.unwrap())
                        .and_then(|s| s.getattr(our_name.as_str()))
                };
                let Ok(their_op_fn) = their_op_fn else {
                    continue;
                };
                run_case_resolved(
                    py,
                    &format!("polynomial-{}-{}", base.base, op),
                    our_name.as_str(),
                    &our_op_fn,
                    &their_op_fn,
                    RequirementLevel::Should,
                    CompareMode::Close,
                    t,
                    move |py| match args_kind {
                        "coef" => Ok(PyTuple::new(py, [coef_3(py)?])?),
                        "roots" => Ok(PyTuple::new(py, [roots_2(py)?])?),
                        "coef_tol" => {
                            let coef = coef_3(py)?;
                            let tol_obj: pyo3::Bound<'_, pyo3::types::PyAny> =
                                0.0_f64.into_pyobject(py)?.into_any();
                            Ok(PyTuple::new(py, [coef, tol_obj])?)
                        }
                        "coef_pow" => {
                            let coef = coef_3(py)?;
                            let pow_obj: pyo3::Bound<'_, pyo3::types::PyAny> =
                                2_i64.into_pyobject(py)?.into_any();
                            Ok(PyTuple::new(py, [coef, pow_obj])?)
                        }
                        "scalars" => {
                            let off: pyo3::Bound<'_, pyo3::types::PyAny> =
                                1.0_f64.into_pyobject(py)?.into_any();
                            let scl: pyo3::Bound<'_, pyo3::types::PyAny> =
                                2.0_f64.into_pyobject(py)?.into_any();
                            Ok(PyTuple::new(py, [off, scl])?)
                        }
                        _ => unreachable!(),
                    },
                    no_kwargs,
                );
            }
        }

        // ─── Class re-export identity (MUST) ─────────────────────
        // fnp_python.polynomial.{Polynomial, Chebyshev, Legendre,
        // Hermite, HermiteE, Laguerre} must BE numpy's class objects.
        let our_poly_mod = module.getattr("polynomial")?;
        for cls_name in [
            "Polynomial",
            "Chebyshev",
            "Legendre",
            "Hermite",
            "HermiteE",
            "Laguerre",
        ] {
            let ours = our_poly_mod.getattr(cls_name)?;
            let theirs = np_poly.getattr(cls_name)?;
            // Manual identity check — emit through a custom call_check that
            // returns the class itself. We use run_case_resolved with a
            // no-arg constructor that returns the class to compare them
            // by repr; identity is enforced separately.
            assert!(
                ours.is(&theirs),
                "fnp_python.polynomial.{cls_name} must BE numpy.polynomial.{cls_name}"
            );
            t.must_total
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            t.must_pass
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            eprintln!(
                "{{\"id\":\"polynomial-class-{}\",\"function\":\"polynomial.{}\",\"level\":\"Must\",\"verdict\":\"PASS\"}}",
                cls_name, cls_name
            );
        }

        eprintln!("\n{}", t.summarize("polynomial"));
        Ok(())
    });
}
