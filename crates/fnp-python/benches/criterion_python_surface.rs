//! Criterion benchmarks for the PyO3 `fnp_python` surface.
//!
//! These target Python-boundary costs that the Rust engine benches do not see.

use criterion::{Criterion, criterion_group, criterion_main};
use fnp_python::fnp_python;
use pyo3::types::{PyAnyMethods, PyModule, PyTuple};
use pyo3::{PyResult, Python};
use std::hint::black_box;
use std::time::Duration;

fn ensure_numpy_available(py: Python<'_>) -> PyResult<()> {
    py.import("numpy").map(drop)
}

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

        group.bench_function("sqrt_f64_1m", |bench| {
            bench.iter(|| {
                let result = sqrt.call1((&input,)).expect("sqrt benchmark call");
                black_box(result);
            });
        });
    });

    group.finish();
}

fn bench_ediff1d_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_ediff1d_boundary");
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
            .call_method1(
                "linspace",
                (-1_000_000.0_f64, 1_000_000.0_f64, 2_000_000_usize),
            )
            .expect("2M f64 input");
        let ediff1d = module.getattr("ediff1d").expect("fnp_python.ediff1d");

        group.bench_function("ediff1d_f64_2m", |bench| {
            bench.iter(|| {
                let result = ediff1d.call1((&input,)).expect("ediff1d benchmark call");
                black_box(result);
            });
        });
    });

    group.finish();
}

fn bench_select_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_select_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let base = numpy
            .call_method1("linspace", (-1.0_f64, 1.0_f64, 2_000_000_usize))
            .expect("2M f64 input");
        let cond_low = numpy
            .getattr("less")
            .expect("numpy.less")
            .call1((&base, -0.25_f64))
            .expect("low condition");
        let cond_high = numpy
            .getattr("greater")
            .expect("numpy.greater")
            .call1((&base, 0.25_f64))
            .expect("high condition");
        let choice_low = numpy
            .getattr("multiply")
            .expect("numpy.multiply")
            .call1((&base, -3.0_f64))
            .expect("low choice");
        let choice_high = numpy
            .getattr("add")
            .expect("numpy.add")
            .call1((&base, 7.0_f64))
            .expect("high choice");
        let condlist = PyTuple::new(py, [&cond_low, &cond_high]).expect("condlist");
        let choicelist = PyTuple::new(py, [&choice_low, &choice_high]).expect("choicelist");
        let select = module.getattr("select").expect("fnp_python.select");

        group.bench_function("select_2conds_f64_2m", |bench| {
            bench.iter(|| {
                let result = select
                    .call1((&condlist, &choicelist))
                    .expect("select benchmark call");
                black_box(result);
            });
        });
    });

    group.finish();
}

fn bench_ldexp_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_ldexp_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let x1 = numpy
            .call_method1(
                "linspace",
                (-1_000_000.0_f64, 1_000_000.0_f64, 2_000_000_usize),
            )
            .expect("2M f64 input");
        let x2 = numpy
            .call_method("full", ((2_000_000_usize,), 3_i32), None)
            .expect("2M int32 exponent")
            .call_method1("astype", ("int32",))
            .expect("int32 exponent dtype");
        let ldexp = module.getattr("ldexp").expect("fnp_python.ldexp");

        group.bench_function("ldexp_f64_i32_2m", |bench| {
            bench.iter(|| {
                let result = ldexp
                    .call1((&x1, &x2))
                    .expect("ldexp f64/int32 benchmark call");
                black_box(result);
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sqrt_input_extraction,
    bench_ediff1d_boundary,
    bench_select_boundary,
    bench_ldexp_boundary
);
criterion_main!(benches);
