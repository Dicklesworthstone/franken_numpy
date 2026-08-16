//! Float64 convolve and correlate boundary benchmarks split from the surface binary.

use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyDict, PyModule};
use std::hint::black_box;
use std::time::Duration;

#[path = "common/mod.rs"]
mod common;
use common::*;

fn bench_f64_convolve_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_f64_convolve_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let setup = "import numpy as np\n\
rng = np.random.default_rng(17)\n\
a = rng.standard_normal(1 << 20).astype(np.float64)\n\
v = rng.standard_normal(256).astype(np.float64)\n";
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup)
                .expect("convolve setup source")
                .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("f64 convolve setup");
        let a = namespace.get_item("a").expect("a");
        let v = namespace.get_item("v").expect("v");
        let fnp_conv = module.getattr("convolve").expect("fnp convolve");
        let numpy_conv = numpy.getattr("convolve").expect("numpy convolve");
        let fnp_corr = module.getattr("correlate").expect("fnp correlate");
        let numpy_corr = numpy.getattr("correlate").expect("numpy correlate");

        group.bench_function("fnp_convolve_f64_1m_256_same", |b| {
            b.iter(|| black_box(fnp_conv.call1((&a, &v, "same")).expect("fnp convolve")));
        });
        group.bench_function("numpy_convolve_f64_1m_256_same", |b| {
            b.iter(|| black_box(numpy_conv.call1((&a, &v, "same")).expect("numpy convolve")));
        });
        group.bench_function("fnp_correlate_f64_1m_256_valid", |b| {
            b.iter(|| black_box(fnp_corr.call1((&a, &v, "valid")).expect("fnp correlate")));
        });
        group.bench_function("numpy_correlate_f64_1m_256_valid", |b| {
            b.iter(|| {
                black_box(
                    numpy_corr
                        .call1((&a, &v, "valid"))
                        .expect("numpy correlate"),
                )
            });
        });
    });
    group.finish();
}

fn main() {
    common::gated_main_with_source(
        include_str!("criterion_python_f64_convolve.rs"),
        &[("bench_f64_convolve_boundary", bench_f64_convolve_boundary)],
    );
}
