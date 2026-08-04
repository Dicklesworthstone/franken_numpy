//! Float isin boundary benchmark split from the surface binary.

use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyDict, PyModule};
use std::hint::black_box;
use std::time::Duration;

#[path = "common/mod.rs"]
mod common;
use common::*;

// NumPy falls back to a serial float sort; the native zero-copy hashed-set path is O(n+m).
fn bench_float_isin_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_float_isin_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_isin = module.getattr("isin").expect("fnp isin");
        let numpy_isin = numpy.getattr("isin").expect("numpy isin");
        let array_equal = numpy.getattr("array_equal").expect("numpy array_equal");
        let setup = "import numpy as np\n\
rng = np.random.default_rng(0)\n\
A64 = rng.standard_normal(8_000_000)\n\
B64 = rng.standard_normal(65_536)\n\
A32 = A64.astype(np.float32)\n\
B32 = B64.astype(np.float32)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup)
                .expect("isin setup CString")
                .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("isin setup");
        for (a_key, b_key, label) in [("A64", "B64", "f64"), ("A32", "B32", "f32")] {
            let a = ns.get_item(a_key).expect("a");
            let b = ns.get_item(b_key).expect("b");
            let actual = fnp_isin.call1((&a, &b)).expect("fnp isin parity");
            let expected = numpy_isin.call1((&a, &b)).expect("numpy isin parity");
            let equal = array_equal
                .call1((&actual, &expected))
                .expect("isin array_equal")
                .extract::<bool>()
                .expect("isin equality boolean");
            assert!(equal, "fnp isin {label} must match NumPy before timing");
            group.bench_function(format!("fnp_isin_{label}_8m"), |bench| {
                bench.iter(|| black_box(fnp_isin.call1((&a, &b)).expect("fnp isin")));
            });
            group.bench_function(format!("numpy_isin_{label}_8m"), |bench| {
                bench.iter(|| black_box(numpy_isin.call1((&a, &b)).expect("numpy isin")));
            });
        }
    });

    group.finish();
}

fn main() {
    common::gated_main(&[("bench_float_isin_boundary", bench_float_isin_boundary)]);
}
