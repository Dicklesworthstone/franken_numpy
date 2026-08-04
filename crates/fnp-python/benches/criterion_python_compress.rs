//! Axis-aware compress boundary benchmark split from the surface binary.

use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyDict, PyModule};
use std::hint::black_box;
use std::time::Duration;

#[path = "common/mod.rs"]
mod common;
use common::*;

// The inner==1 f64 path delegates to NumPy; this guard keeps that parity explicit.
fn bench_compress_lastaxis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_compress_lastaxis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
x = rng.standard_normal((2048, 2048))\ncond = rng.random(2048) < 0.5\n",
            )
            .expect("compress setup CString")
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("compress lastaxis setup");
        let x = ns.get_item("x").expect("x");
        let cond = ns.get_item("cond").expect("cond");
        let fnp_compress = module.getattr("compress").expect("fnp compress");
        let numpy_compress = numpy.getattr("compress").expect("numpy compress");
        let array_equal = numpy.getattr("array_equal").expect("numpy array_equal");
        let kw = PyDict::new(py);
        kw.set_item("axis", 1_i64).expect("compress axis");
        let actual = fnp_compress
            .call((&cond, &x), Some(&kw))
            .expect("fnp compress parity");
        let expected = numpy_compress
            .call((&cond, &x), Some(&kw))
            .expect("numpy compress parity");
        let equal = array_equal
            .call1((&actual, &expected))
            .expect("compress array_equal")
            .extract::<bool>()
            .expect("compress equality boolean");
        assert!(equal, "fnp compress axis=1 must match NumPy before timing");
        group.bench_function("fnp_compress_2d_axis1", |b| {
            b.iter(|| {
                black_box(
                    fnp_compress
                        .call((&cond, &x), Some(&kw))
                        .expect("fnp compress"),
                )
            });
        });
        group.bench_function("numpy_compress_2d_axis1", |b| {
            b.iter(|| {
                black_box(
                    numpy_compress
                        .call((&cond, &x), Some(&kw))
                        .expect("numpy compress"),
                )
            });
        });
    });

    group.finish();
}

fn main() {
    common::gated_main(&[(
        "bench_compress_lastaxis_boundary",
        bench_compress_lastaxis_boundary,
    )]);
}
