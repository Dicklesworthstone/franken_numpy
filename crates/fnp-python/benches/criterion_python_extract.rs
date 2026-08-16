//! Python-boundary buffer extraction benchmarks split from the former surface binary.

use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyModule};
use std::hint::black_box;
use std::time::Duration;

#[path = "common/mod.rs"]
mod common;
use common::*;

fn bench_sqrt_input_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_buffer_extract");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let input = numpy
            .call_method1("linspace", (0.0_f64, 1_000_000.0_f64, 1_000_000_usize))
            .expect("1M f64 input");
        let sqrt = module.getattr("sqrt").expect("fnp_python.sqrt");
        let numpy_sqrt = numpy.getattr("sqrt").expect("numpy.sqrt");
        let array_equal = numpy.getattr("array_equal").expect("numpy.array_equal");
        let exact: bool = array_equal
            .call1((
                sqrt.call1((&input,)).expect("fnp sqrt correctness call"),
                numpy_sqrt
                    .call1((&input,))
                    .expect("numpy sqrt correctness call"),
            ))
            .expect("sqrt equality check")
            .extract()
            .expect("sqrt equality result");
        assert!(exact, "native sqrt extraction benchmark must match NumPy");

        group.bench_function("sqrt_f64_1m", |bench| {
            bench.iter(|| {
                let result = sqrt.call1((&input,)).expect("sqrt benchmark call");
                black_box(result);
            });
        });
    });

    group.finish();
}

fn main() {
    common::gated_main_with_source(
        include_str!("criterion_python_extract.rs"),
        &[("bench_sqrt_input_extraction", bench_sqrt_input_extraction)],
    );
}
