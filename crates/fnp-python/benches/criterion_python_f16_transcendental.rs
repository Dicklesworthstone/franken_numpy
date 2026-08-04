//! Float16 binary-transcendental boundary benchmarks split from the surface binary.

use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyDict, PyModule};
use std::hint::black_box;
use std::time::Duration;

#[path = "common/mod.rs"]
mod common;
use common::*;

// f16 arctan2/hypot/logaddexp/logaddexp2: NumPy widens to f32, evaluates serially, and narrows.
// The native parallel widen-op-narrow is bit-exact for the finite fast-path domains.
fn bench_f16_binary_transcendental_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_f16_binary_transcendental_boundary");
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
rng = np.random.default_rng(0)\n\
x = rng.standard_normal(16_000_000).astype(np.float16)\n\
y = rng.standard_normal(16_000_000).astype(np.float16)\n\
pbase = (np.abs(rng.standard_normal(16_000_000)) + 0.5).astype(np.float16)\n\
pexp = (rng.standard_normal(16_000_000) * 0.5).astype(np.float16)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup)
                .expect("f16 binary setup CString")
                .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("f16 binary setup");
        let x = ns.get_item("x").expect("x");
        let y = ns.get_item("y").expect("y");
        let array_equal = numpy.getattr("array_equal").expect("numpy array_equal");
        for name in ["arctan2", "hypot", "logaddexp", "logaddexp2"] {
            let fnp_fn = module.getattr(name).expect("fnp fn");
            let numpy_fn = numpy.getattr(name).expect("numpy fn");
            let actual = fnp_fn.call1((&x, &y)).expect("fnp f16 parity");
            let expected = numpy_fn.call1((&x, &y)).expect("numpy f16 parity");
            let equal = array_equal
                .call1((&actual, &expected))
                .expect("f16 array_equal")
                .extract::<bool>()
                .expect("f16 equality boolean");
            assert!(equal, "fnp {name} f16 must match NumPy before timing");
            group.bench_function(format!("fnp_{name}_f16_16m"), |b| {
                b.iter(|| black_box(fnp_fn.call1((&x, &y)).expect("fnp f16 binary")));
            });
            group.bench_function(format!("numpy_{name}_f16_16m"), |b| {
                b.iter(|| black_box(numpy_fn.call1((&x, &y)).expect("numpy f16 binary")));
            });
        }

        let pbase = ns.get_item("pbase").expect("pbase");
        let pexp = ns.get_item("pexp").expect("pexp");
        let fnp_pow = module.getattr("power").expect("fnp power");
        let numpy_pow = numpy.getattr("power").expect("numpy power");
        let actual = fnp_pow
            .call1((&pbase, &pexp))
            .expect("fnp f16 power parity");
        let expected = numpy_pow
            .call1((&pbase, &pexp))
            .expect("numpy f16 power parity");
        let equal = array_equal
            .call1((&actual, &expected))
            .expect("f16 power array_equal")
            .extract::<bool>()
            .expect("f16 power equality boolean");
        assert!(equal, "fnp power f16 must match NumPy before timing");
        group.bench_function("fnp_power_f16_16m", |b| {
            b.iter(|| black_box(fnp_pow.call1((&pbase, &pexp)).expect("fnp f16 power")));
        });
        group.bench_function("numpy_power_f16_16m", |b| {
            b.iter(|| black_box(numpy_pow.call1((&pbase, &pexp)).expect("numpy f16 power")));
        });
    });

    group.finish();
}

fn main() {
    common::gated_main(&[(
        "bench_f16_binary_transcendental_boundary",
        bench_f16_binary_transcendental_boundary,
    )]);
}
