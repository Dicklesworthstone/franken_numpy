//! Structural boundary benchmarks split from the former surface binary.

use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyDict, PyModule};
use std::hint::black_box;
use std::time::Duration;

#[path = "common/mod.rs"]
mod common;
use common::*;

// np.insert(1-D, scalar idx, values block): numpy runs a serial page-fault-bound copy (~44ms@8M).
// The native parallel three-run byte copy (arr[:idx] | values | arr[idx:]) wins ~3x.
fn bench_insert_block_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_insert_block_boundary");
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
x = rng.standard_normal(8_000_000)\n\
block = rng.standard_normal(1000)\n\
mid = 4_000_000\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup)
                .expect("insert setup CString")
                .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("insert setup");
        let x = ns.get_item("x").expect("x");
        let block = ns.get_item("block").expect("block");
        let mid = ns.get_item("mid").expect("mid");
        let fnp_insert = module.getattr("insert").expect("fnp insert");
        let numpy_insert = numpy.getattr("insert").expect("numpy insert");
        let array_equal = numpy.getattr("array_equal").expect("numpy array_equal");
        let actual = fnp_insert
            .call1((&x, &mid, &block))
            .expect("fnp insert parity");
        let expected = numpy_insert
            .call1((&x, &mid, &block))
            .expect("numpy insert parity");
        let equal = array_equal
            .call1((&actual, &expected))
            .expect("insert array_equal")
            .extract::<bool>()
            .expect("insert equality boolean");
        assert!(equal, "fnp insert must match NumPy before timing");
        group.bench_function("fnp_insert_block_f64_8m", |b| {
            b.iter(|| black_box(fnp_insert.call1((&x, &mid, &block)).expect("fnp insert")));
        });
        group.bench_function("numpy_insert_block_f64_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_insert
                        .call1((&x, &mid, &block))
                        .expect("numpy insert"),
                )
            });
        });
    });

    group.finish();
}

// np.delete(1-D, bool mask / int index array): numpy builds a keep-mask then runs its serial
// compress (~50ms@8M). Routing the keep-mask through fnp's parallel compress wins (bool-mask
// ~1.9x; int-index ~1.3x, dragged by numpy's fancy-assign mask build).
fn bench_delete_mask_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_delete_mask_boundary");
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
x = rng.standard_normal(8_000_000)\n\
mask = rng.random(8_000_000) < 0.5\n\
idx = np.sort(rng.choice(8_000_000, size=2_000_000, replace=False))\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup)
                .expect("delete setup CString")
                .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("delete setup");
        let x = ns.get_item("x").expect("x");
        let mask = ns.get_item("mask").expect("mask");
        let idx = ns.get_item("idx").expect("idx");
        let fnp_delete = module.getattr("delete").expect("fnp delete");
        let numpy_delete = numpy.getattr("delete").expect("numpy delete");
        let array_equal = numpy.getattr("array_equal").expect("numpy array_equal");
        for (label, obj) in [("boolmask", &mask), ("intidx", &idx)] {
            let actual = fnp_delete.call1((&x, obj)).expect("fnp delete parity");
            let expected = numpy_delete.call1((&x, obj)).expect("numpy delete parity");
            let equal = array_equal
                .call1((&actual, &expected))
                .expect("delete array_equal")
                .extract::<bool>()
                .expect("delete equality boolean");
            assert!(equal, "fnp delete {label} must match NumPy before timing");
            group.bench_function(format!("fnp_delete_{label}_8m"), |b| {
                b.iter(|| black_box(fnp_delete.call1((&x, obj)).expect("fnp delete")));
            });
            group.bench_function(format!("numpy_delete_{label}_8m"), |b| {
                b.iter(|| black_box(numpy_delete.call1((&x, obj)).expect("numpy delete")));
            });
        }
    });

    group.finish();
}

fn main() {
    common::gated_main(&[
        ("bench_insert_block_boundary", bench_insert_block_boundary),
        ("bench_delete_mask_boundary", bench_delete_mask_boundary),
    ]);
}
