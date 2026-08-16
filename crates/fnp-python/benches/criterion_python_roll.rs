//! Roll boundary benchmarks split from the surface binary.

use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyModule};
use std::hint::black_box;
use std::time::Duration;

#[path = "common/mod.rs"]
mod common;
use common::*;

fn bench_roll_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_roll_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_roll = module.getattr("roll").expect("fnp_python.roll");
        let numpy_roll = numpy.getattr("roll").expect("numpy.roll");

        let size = 4_000_000_i64;
        let shift = 1000_i64;
        let input = numpy
            .call_method1("arange", (size,))
            .expect("roll index")
            .call_method1("astype", ("float64",))
            .expect("roll f64 input");

        group.bench_function("fnp_roll_f64_axis_none_4m_shift1000", |bench| {
            bench.iter(|| {
                let result = fnp_roll
                    .call1((&input, shift))
                    .expect("fnp roll benchmark call");
                black_box(result);
            });
        });
        group.bench_function("numpy_roll_f64_axis_none_4m_shift1000", |bench| {
            bench.iter(|| {
                let result = numpy_roll
                    .call1((&input, shift))
                    .expect("numpy roll benchmark call");
                black_box(result);
            });
        });

        let input2d = numpy
            .call_method1("arange", (2000_i64 * 2000_i64,))
            .expect("roll 2d index")
            .call_method1("astype", ("float64",))
            .expect("roll 2d f64")
            .call_method1("reshape", ((2000_i64, 2000_i64),))
            .expect("roll 2d reshape");
        for (label, axis) in [("axis0", 0_i64), ("axis1", 1_i64)] {
            group.bench_function(format!("fnp_roll_f64_2d_{label}_shift7"), |bench| {
                bench.iter(|| {
                    black_box(
                        fnp_roll
                            .call1((&input2d, 7_i64, axis))
                            .expect("fnp roll 2d call"),
                    );
                });
            });
            group.bench_function(format!("numpy_roll_f64_2d_{label}_shift7"), |bench| {
                bench.iter(|| {
                    black_box(
                        numpy_roll
                            .call1((&input2d, 7_i64, axis))
                            .expect("numpy roll 2d call"),
                    );
                });
            });
        }
    });

    group.finish();
}

fn main() {
    common::gated_main_with_source(
        include_str!("criterion_python_roll.rs"),
        &[("bench_roll_boundary", bench_roll_boundary)],
    );
}
