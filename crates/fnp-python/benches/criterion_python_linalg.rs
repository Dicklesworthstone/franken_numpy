//! linalg-domain criterion benches. The TSQR least-squares A/B harness:
//! `fnp.lstsq` vs `numpy.linalg.lstsq` (LAPACK gelsd) on full-rank tall-skinny
//! float64 systems. The copy-based native TSQR wiring was REJECTED for
//! franken_numpy-ixs5y.i546h (no measured win — numpy's gelsd is competitive,
//! and the per-call buffer copy is a fixed overhead; see docs/NEGATIVE_EVIDENCE),
//! so `fnp.lstsq` is presently a numpy passthrough and this measures ~1x. The
//! harness is kept ready for the retry: zero-copy PyBuffer extraction + a
//! clean/unloaded worker. Split as its own per-domain bench binary so a re-measure
//! compiles without the whole monolith (bead deadlock-audit-x7nnf).

#[path = "common/mod.rs"]
mod common;

use common::ensure_numpy_available;
use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyDict, PyModule};
use std::hint::black_box;
use std::time::Duration;

fn bench_lstsq_tsqr_tall_skinny(c: &mut Criterion) {
    // numpy.linalg.lstsq runs LAPACK gelsd (a full divide-and-conquer SVD) on the
    // tall-skinny matrix; fnp routes full-rank tall-skinny float64 with a 1-D rhs
    // to native TSQR — one streaming pass to a tiny n×n R, back substitution for
    // x, and an n×n SVD of R for rank/singular values (σ(A)=σ(R)). Solutions are
    // allclose (every returned element is invariant to TSQR's Q/R sign choice).
    let mut group = c.benchmark_group("python_lstsq_tsqr_tall_skinny");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let setup = "import numpy as np\n\
rng = np.random.default_rng(12345)\n\
a1 = rng.standard_normal((1_000_000, 8))\n\
b1 = rng.standard_normal(1_000_000)\n\
a2 = rng.standard_normal((1_000_000, 16))\n\
b2 = rng.standard_normal(1_000_000)\n\
a3 = rng.standard_normal((2_000_000, 8))\n\
b3 = rng.standard_normal(2_000_000)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("lstsq bench setup");
        let fnp_lstsq = module.getattr("lstsq").expect("fnp lstsq");
        let numpy_lstsq = numpy
            .getattr("linalg")
            .expect("numpy.linalg")
            .getattr("lstsq")
            .expect("numpy lstsq");
        let allclose = numpy.getattr("allclose").expect("np.allclose");

        for (a_key, b_key, label) in [
            ("a1", "b1", "1000000x8"),
            ("a2", "b2", "1000000x16"),
            ("a3", "b3", "2000000x8"),
        ] {
            let a = ns.get_item(a_key).expect("operand a");
            let b = ns.get_item(b_key).expect("operand b");
            // Correctness gate on the bench corpus: the solutions must agree.
            let fnp_out = fnp_lstsq.call1((&a, &b)).expect("fnp lstsq");
            let np_out = numpy_lstsq.call1((&a, &b)).expect("numpy lstsq");
            let close: bool = allclose
                .call1((
                    fnp_out.get_item(0).expect("fnp solution"),
                    np_out.get_item(0).expect("numpy solution"),
                ))
                .expect("allclose")
                .extract()
                .expect("bool");
            assert!(close, "lstsq {label} solution diverged on bench corpus");

            group.bench_function(format!("fnp_lstsq_{label}"), |bench| {
                bench.iter(|| black_box(fnp_lstsq.call1((&a, &b)).expect("fnp lstsq")));
            });
            group.bench_function(format!("numpy_lstsq_{label}"), |bench| {
                bench.iter(|| black_box(numpy_lstsq.call1((&a, &b)).expect("numpy lstsq")));
            });
        }
    });
    group.finish();
}

fn main() {
    common::gated_main(&[("bench_lstsq_tsqr_tall_skinny", bench_lstsq_tsqr_tall_skinny)]);
}
