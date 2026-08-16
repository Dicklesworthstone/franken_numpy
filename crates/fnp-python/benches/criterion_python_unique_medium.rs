//! Medium-size Python `unique` boundary benchmarks split from the surface binary.

use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyModule};
use std::hint::black_box;
use std::time::Duration;

#[path = "common/mod.rs"]
mod common;
use common::*;

fn bench_unique_medium_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_unique_medium_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_unique = module.getattr("unique").expect("fnp unique");
        let numpy_unique = numpy.getattr("unique").expect("numpy unique");
        let array_equal = numpy.getattr("array_equal").expect("numpy.array_equal");

        let make_repeated_f64 = |size: i64| {
            numpy
                .call_method1("arange", (size,))
                .expect("unique arange")
                .call_method1("__mul__", (37_i64,))
                .expect("unique mix")
                .call_method1("__mod__", (65_536_i64,))
                .expect("unique modulo")
                .call_method1("__truediv__", (16.0_f64,))
                .expect("unique scale")
                .call_method1("astype", ("float64",))
                .expect("unique f64 input")
        };

        let correctness_input = make_repeated_f64(50_000);
        let exact: bool = array_equal
            .call1((
                fnp_unique
                    .call1((&correctness_input,))
                    .expect("fnp unique correctness call"),
                numpy_unique
                    .call1((&correctness_input,))
                    .expect("numpy unique correctness call"),
            ))
            .expect("unique equality check")
            .extract()
            .expect("unique equality result");
        assert!(exact, "native unique benchmark must match NumPy");

        for (label, input) in [
            ("50k", correctness_input),
            ("512k", make_repeated_f64(512_000)),
            ("1m_gate", make_repeated_f64(1_048_576)),
        ] {
            group.bench_function(format!("fnp_unique_f64_repeated_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_unique.call1((&input,)).expect("fnp unique f64 call");
                    black_box(result);
                });
            });

            group.bench_function(format!("numpy_unique_f64_repeated_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_unique
                        .call1((&input,))
                        .expect("numpy unique f64 call");
                    black_box(result);
                });
            });
        }

        let distinct_1m = numpy
            .call_method1("arange", (1_048_576_i64,))
            .expect("unique distinct arange")
            .call_method1("__mul__", (1_103_515_245_i64,))
            .expect("unique distinct mix")
            .call_method1("__mod__", (2_147_483_647_i64,))
            .expect("unique distinct modulo")
            .call_method1("astype", ("float64",))
            .expect("unique distinct f64 input");

        group.bench_function("fnp_unique_f64_distinct_1m_gate", |bench| {
            bench.iter(|| {
                let result = fnp_unique
                    .call1((&distinct_1m,))
                    .expect("fnp unique distinct f64 call");
                black_box(result);
            });
        });

        group.bench_function("numpy_unique_f64_distinct_1m_gate", |bench| {
            bench.iter(|| {
                let result = numpy_unique
                    .call1((&distinct_1m,))
                    .expect("numpy unique distinct f64 call");
                black_box(result);
            });
        });
    });

    group.finish();
}

fn main() {
    common::gated_main_with_source(
        include_str!("criterion_python_unique_medium.rs"),
        &[("bench_unique_medium_boundary", bench_unique_medium_boundary)],
    );
}
