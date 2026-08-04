//! Float16 matmul boundary benchmarks split from the surface binary.

use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyModule};
use std::hint::black_box;
use std::time::Duration;

#[path = "common/mod.rs"]
mod common;
use common::*;

fn bench_f16_matmul_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_f16_matmul_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_matmul = module.getattr("matmul").expect("fnp matmul");
        let numpy_matmul = numpy.getattr("matmul").expect("numpy matmul");

        let default_rng = numpy
            .getattr("random")
            .expect("numpy.random")
            .getattr("default_rng")
            .expect("default_rng");
        let make = |sz: usize| {
            let rng = default_rng.call1((sz as i64,)).expect("rng");
            let a = rng
                .call_method1("standard_normal", ((sz, sz),))
                .expect("a")
                .call_method1("__mul__", (0.3_f64,))
                .expect("scale a")
                .call_method1("astype", ("float16",))
                .expect("a f16");
            let b = rng
                .call_method1("standard_normal", ((sz, sz),))
                .expect("b")
                .call_method1("__mul__", (0.3_f64,))
                .expect("scale b")
                .call_method1("astype", ("float16",))
                .expect("b f16");
            (a, b)
        };

        let (parity_a, parity_b) = make(8);
        let parity: bool = numpy
            .getattr("array_equal")
            .expect("numpy array_equal")
            .call1((
                fnp_matmul
                    .call1((&parity_a, &parity_b))
                    .expect("fnp f16 parity matmul"),
                numpy_matmul
                    .call1((&parity_a, &parity_b))
                    .expect("numpy f16 parity matmul"),
            ))
            .expect("f16 parity comparison")
            .extract()
            .expect("f16 parity boolean");
        assert!(parity, "f16 matmul must match NumPy before timing");

        for sz in [256_usize, 512, 1024] {
            let (a, b) = make(sz);
            group.bench_function(format!("fnp_matmul_f16_{sz}"), |bch| {
                bch.iter(|| black_box(fnp_matmul.call1((&a, &b)).expect("fnp f16 matmul")));
            });
            group.bench_function(format!("numpy_matmul_f16_{sz}"), |bch| {
                bch.iter(|| black_box(numpy_matmul.call1((&a, &b)).expect("np f16 matmul")));
            });
        }

        let make3 = |batch: usize, sz: usize| {
            let rng = default_rng.call1(((batch + sz) as i64,)).expect("rng3");
            let a = rng
                .call_method1("standard_normal", ((batch, sz, sz),))
                .expect("a3")
                .call_method1("__mul__", (0.3_f64,))
                .expect("scale a3")
                .call_method1("astype", ("float16",))
                .expect("a3 f16");
            let b = rng
                .call_method1("standard_normal", ((batch, sz, sz),))
                .expect("b3")
                .call_method1("__mul__", (0.3_f64,))
                .expect("scale b3")
                .call_method1("astype", ("float16",))
                .expect("b3 f16");
            (a, b)
        };
        let (a3, b3) = make3(64, 128);
        group.bench_function("fnp_matmul_f16_batched_64x128", |bch| {
            bch.iter(|| black_box(fnp_matmul.call1((&a3, &b3)).expect("fnp f16 batched")));
        });
        group.bench_function("numpy_matmul_f16_batched_64x128", |bch| {
            bch.iter(|| black_box(numpy_matmul.call1((&a3, &b3)).expect("np f16 batched")));
        });

        let (ab, _) = make3(64, 128);
        let (bb2d, _) = make(128);
        group.bench_function("fnp_matmul_f16_bcast_64x128", |bch| {
            bch.iter(|| black_box(fnp_matmul.call1((&ab, &bb2d)).expect("fnp f16 bcast")));
        });
        group.bench_function("numpy_matmul_f16_bcast_64x128", |bch| {
            bch.iter(|| black_box(numpy_matmul.call1((&ab, &bb2d)).expect("np f16 bcast")));
        });

        let (at, bt) = make(512);
        let fnp_td = module.getattr("tensordot").expect("fnp tensordot");
        let numpy_td = numpy.getattr("tensordot").expect("numpy tensordot");
        let fnp_inner = module.getattr("inner").expect("fnp inner");
        let numpy_inner = numpy.getattr("inner").expect("numpy inner");
        let one = 1_i64;
        group.bench_function("fnp_tensordot_f16_512", |bch| {
            bch.iter(|| black_box(fnp_td.call1((&at, &bt, one)).expect("fnp f16 tensordot")));
        });
        group.bench_function("numpy_tensordot_f16_512", |bch| {
            bch.iter(|| black_box(numpy_td.call1((&at, &bt, one)).expect("np f16 tensordot")));
        });
        group.bench_function("fnp_inner_f16_512", |bch| {
            bch.iter(|| black_box(fnp_inner.call1((&at, &bt)).expect("fnp f16 inner")));
        });
        group.bench_function("numpy_inner_f16_512", |bch| {
            bch.iter(|| black_box(numpy_inner.call1((&at, &bt)).expect("np f16 inner")));
        });
    });

    group.finish();
}

fn main() {
    common::gated_main(&[("bench_f16_matmul_boundary", bench_f16_matmul_boundary)]);
}
