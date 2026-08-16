//! Integer convolution boundary benchmark split from the Python surface target.

use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyDict, PyModule};
use std::hint::black_box;
use std::time::Duration;

#[path = "common/mod.rs"]
mod common;
use common::*;

fn bench_int_convolve_boundary(c: &mut Criterion) {
    // NumPy has no integer convolution fast path; retain the original direct
    // FNP-versus-NumPy boundary measurement unchanged.
    let mut group = c.benchmark_group("python_int_convolve_boundary");
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
rng = np.random.default_rng(12)\n\
a = rng.integers(-100, 100, 200_000).astype(np.int64)\n\
v = rng.integers(-100, 100, 256).astype(np.int64)\n";
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup)
                .expect("integer convolve setup source")
                .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("integer convolve setup");
        let a = namespace.get_item("a").expect("a");
        let v = namespace.get_item("v").expect("v");
        let fnp_conv = module.getattr("convolve").expect("fnp convolve");
        let numpy_conv = numpy.getattr("convolve").expect("numpy convolve");

        group.bench_function("fnp_convolve_i64_200k_256", |b| {
            b.iter(|| black_box(fnp_conv.call1((&a, &v, "full")).expect("fnp int convolve")));
        });
        group.bench_function("numpy_convolve_i64_200k_256", |b| {
            b.iter(|| {
                black_box(
                    numpy_conv
                        .call1((&a, &v, "full"))
                        .expect("numpy int convolve"),
                )
            });
        });
    });

    group.finish();
}

fn main() {
    common::gated_main_with_source(
        include_str!("criterion_python_int_convolve.rs"),
        &[("bench_int_convolve_boundary", bench_int_convolve_boundary)],
    );
}
