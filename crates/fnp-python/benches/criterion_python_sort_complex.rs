//! Complex-sort boundary benchmarks split from the former surface binary.

use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyModule};
use std::hint::black_box;
use std::time::Duration;

#[path = "common/mod.rs"]
mod common;
use common::*;

fn bench_sort_complex_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_sort_complex_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_sort_complex = module.getattr("sort_complex").expect("fnp sort_complex");
        let numpy_sort_complex = numpy.getattr("sort_complex").expect("numpy sort_complex");
        let array_equal = numpy.getattr("array_equal").expect("numpy.array_equal");

        for size in [200_000_i64, 1_000_000_i64] {
            let values = numpy
                .call_method1("arange", (size,))
                .expect("sort_complex arange")
                .call_method1("__mul__", (1_103_515_245_i64,))
                .expect("sort_complex mix")
                .call_method1("__mod__", (size,))
                .expect("sort_complex modulo")
                .call_method1("astype", ("float64",))
                .expect("sort_complex f64 input");

            if size == 200_000 {
                let exact: bool = array_equal
                    .call1((
                        fnp_sort_complex
                            .call1((&values,))
                            .expect("fnp sort_complex correctness call"),
                        numpy_sort_complex
                            .call1((&values,))
                            .expect("numpy sort_complex correctness call"),
                    ))
                    .expect("sort_complex equality check")
                    .extract()
                    .expect("sort_complex equality result");
                assert!(exact, "native sort_complex benchmark must match NumPy");
            }

            group.bench_function(format!("fnp_sort_complex_real_f64_{size}"), |bench| {
                bench.iter(|| {
                    let result = fnp_sort_complex
                        .call1((&values,))
                        .expect("fnp sort_complex benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("numpy_sort_complex_real_f64_{size}"), |bench| {
                bench.iter(|| {
                    let result = numpy_sort_complex
                        .call1((&values,))
                        .expect("numpy sort_complex benchmark call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

fn main() {
    common::gated_main_with_source(
        include_str!("criterion_python_sort_complex.rs"),
        &[("bench_sort_complex_boundary", bench_sort_complex_boundary)],
    );
}
