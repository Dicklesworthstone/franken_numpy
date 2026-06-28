//! Criterion benchmarks for the PyO3 `fnp_python` surface.
//!
//! These target Python-boundary costs that the Rust engine benches do not see.

use criterion::{Criterion, criterion_group, criterion_main};
use fnp_python::fnp_python;
use pyo3::types::{PyAnyMethods, PyDict, PyModule, PyTuple};
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

fn bench_int32_unary_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_int32_unary_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let input = numpy
            .call_method1("arange", (2_000_000_i64,))
            .expect("2M int32 input")
            .call_method1("astype", ("int32",))
            .expect("int32 input dtype")
            .call_method1("__sub__", (1_000_000_i64,))
            .expect("centered int32 range");
        let fnp_square = module.getattr("square").expect("fnp_python.square");
        let fnp_negative = module.getattr("negative").expect("fnp_python.negative");
        let numpy_square = numpy.getattr("square").expect("numpy.square");

        group.bench_function("fnp_square_i32_2m", |bench| {
            bench.iter(|| {
                let result = fnp_square
                    .call1((&input,))
                    .expect("fnp square int32 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("fnp_negative_i32_2m", |bench| {
            bench.iter(|| {
                let result = fnp_negative
                    .call1((&input,))
                    .expect("fnp negative int32 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_square_i32_2m", |bench| {
            bench.iter(|| {
                let result = numpy_square
                    .call1((&input,))
                    .expect("numpy square int32 benchmark call");
                black_box(result);
            });
        });
    });

    group.finish();
}

fn bench_narrow_int_unary_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_narrow_int_unary_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let base = numpy
            .call_method1("arange", (2_000_000_i64,))
            .expect("2M integer input");
        let input_i16 = base
            .call_method1("astype", ("int16",))
            .expect("int16 input");
        let input_u8 = base
            .call_method1("astype", ("uint8",))
            .expect("uint8 input");
        let input_u64 = base
            .call_method1("astype", ("uint64",))
            .expect("uint64 input");
        let fnp_square = module.getattr("square").expect("fnp_python.square");
        let fnp_negative = module.getattr("negative").expect("fnp_python.negative");
        let numpy_square = numpy.getattr("square").expect("numpy.square");
        let numpy_negative = numpy.getattr("negative").expect("numpy.negative");

        group.bench_function("fnp_square_i16_2m", |bench| {
            bench.iter(|| {
                let result = fnp_square
                    .call1((&input_i16,))
                    .expect("fnp square int16 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_square_i16_2m", |bench| {
            bench.iter(|| {
                let result = numpy_square
                    .call1((&input_i16,))
                    .expect("numpy square int16 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("fnp_negative_u8_2m", |bench| {
            bench.iter(|| {
                let result = fnp_negative
                    .call1((&input_u8,))
                    .expect("fnp negative uint8 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_negative_u8_2m", |bench| {
            bench.iter(|| {
                let result = numpy_negative
                    .call1((&input_u8,))
                    .expect("numpy negative uint8 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("fnp_square_u64_2m", |bench| {
            bench.iter(|| {
                let result = fnp_square
                    .call1((&input_u64,))
                    .expect("fnp square uint64 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_square_u64_2m", |bench| {
            bench.iter(|| {
                let result = numpy_square
                    .call1((&input_u64,))
                    .expect("numpy square uint64 benchmark call");
                black_box(result);
            });
        });
    });

    group.finish();
}

fn bench_remainder_mod_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_remainder_mod_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_mod = module.getattr("mod").expect("fnp_python.mod");
        let fnp_remainder = module.getattr("remainder").expect("fnp_python.remainder");
        let numpy_remainder = numpy.getattr("remainder").expect("numpy.remainder");

        for (label, len) in [("1m", 1_000_000_usize), ("8m", 8_000_000_usize)] {
            let x1 = numpy
                .call_method1("linspace", (-1_000_000.0_f64, 1_000_000.0_f64, len))
                .expect("f64 remainder dividend input");
            let x2 = numpy
                .call_method1("full", ((len,), 7.25_f64))
                .expect("f64 remainder divisor input");

            group.bench_function(format!("fnp_mod_f64_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_mod
                        .call1((&x1, &x2))
                        .expect("fnp mod f64 benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("fnp_remainder_f64_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_remainder
                        .call1((&x1, &x2))
                        .expect("fnp remainder f64 benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("numpy_remainder_f64_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_remainder
                        .call1((&x1, &x2))
                        .expect("numpy remainder f64 benchmark call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

fn bench_max_min_reduction_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_max_min_reduction_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let input = numpy
            .call_method1("linspace", (-1.0_f64, 1.0_f64, 2048_usize * 2048_usize))
            .expect("4M f64 input")
            .call_method1("reshape", ((2048_usize, 2048_usize),))
            .expect("2048x2048 f64 input");
        let fnp_max = module.getattr("max").expect("fnp_python.max");
        let fnp_min = module.getattr("min").expect("fnp_python.min");
        let numpy_max = numpy.getattr("max").expect("numpy.max");
        let numpy_min = numpy.getattr("min").expect("numpy.min");

        group.bench_function("fnp_max_axis1_f64_2048x2048", |bench| {
            bench.iter(|| {
                let result = fnp_max
                    .call1((&input, 1_i64))
                    .expect("fnp max axis=1 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_max_axis1_f64_2048x2048", |bench| {
            bench.iter(|| {
                let result = numpy_max
                    .call1((&input, 1_i64))
                    .expect("numpy max axis=1 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("fnp_max_flat_f64_4m", |bench| {
            bench.iter(|| {
                let result = fnp_max
                    .call1((&input,))
                    .expect("fnp max flat benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_max_flat_f64_4m", |bench| {
            bench.iter(|| {
                let result = numpy_max
                    .call1((&input,))
                    .expect("numpy max flat benchmark call");
                black_box(result);
            });
        });

        group.bench_function("fnp_min_flat_f64_4m", |bench| {
            bench.iter(|| {
                let result = fnp_min
                    .call1((&input,))
                    .expect("fnp min flat benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_min_flat_f64_4m", |bench| {
            bench.iter(|| {
                let result = numpy_min
                    .call1((&input,))
                    .expect("numpy min flat benchmark call");
                black_box(result);
            });
        });

        // axis=0 (non-last, strided for numpy): the parallel native fold's biggest win.
        group.bench_function("fnp_max_axis0_f64_2048x2048", |bench| {
            bench.iter(|| {
                let result = fnp_max
                    .call1((&input, 0_i64))
                    .expect("fnp max axis=0 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_max_axis0_f64_2048x2048", |bench| {
            bench.iter(|| {
                let result = numpy_max
                    .call1((&input, 0_i64))
                    .expect("numpy max axis=0 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("fnp_min_axis0_f64_2048x2048", |bench| {
            bench.iter(|| {
                let result = fnp_min
                    .call1((&input, 0_i64))
                    .expect("fnp min axis=0 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_min_axis0_f64_2048x2048", |bench| {
            bench.iter(|| {
                let result = numpy_min
                    .call1((&input, 0_i64))
                    .expect("numpy min axis=0 benchmark call");
                black_box(result);
            });
        });

        // 3-D middle axis (axis=1): block-parallel non-last path.
        let input3d = numpy
            .call_method1("linspace", (-1.0_f64, 1.0_f64, 256_usize * 256_usize * 64_usize))
            .expect("4M f64 3d source")
            .call_method1("reshape", ((256_usize, 256_usize, 64_usize),))
            .expect("256x256x64 reshape");
        group.bench_function("fnp_max_axis1_f64_256x256x64", |bench| {
            bench.iter(|| {
                let result = fnp_max
                    .call1((&input3d, 1_i64))
                    .expect("fnp max 3d axis=1 call");
                black_box(result);
            });
        });

        group.bench_function("numpy_max_axis1_f64_256x256x64", |bench| {
            bench.iter(|| {
                let result = numpy_max
                    .call1((&input3d, 1_i64))
                    .expect("numpy max 3d axis=1 call");
                black_box(result);
            });
        });

        group.bench_function("fnp_min_axis1_f64_256x256x64", |bench| {
            bench.iter(|| {
                let result = fnp_min
                    .call1((&input3d, 1_i64))
                    .expect("fnp min 3d axis=1 call");
                black_box(result);
            });
        });

        group.bench_function("numpy_min_axis1_f64_256x256x64", |bench| {
            bench.iter(|| {
                let result = numpy_min
                    .call1((&input3d, 1_i64))
                    .expect("numpy min 3d axis=1 call");
                black_box(result);
            });
        });
    });

    group.finish();
}

// ptp (max-min) over a reduction axis. numpy computes ptp as two strided passes
// (max then min) plus a subtract temp; fnp fuses min+max into one streaming pass.
// The axis=0 single-outer-group case used to fold over rows allocating an
// inner-wide plane per fold segment (~2.6x slower than numpy at large inner); the
// column-block parallel rewrite makes it a single pass. axis=1 (middle) is the
// already-fast block-parallel non-last path, kept here as a regression guard.
fn bench_ptp_axis0_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_ptp_axis0_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_ptp = module.getattr("ptp").expect("fnp_python.ptp");
        let numpy_ptp = numpy.getattr("ptp").expect("numpy.ptp");

        // 3-D axis=0: outer==1, inner=128*256=32768 — the fixed single-group path.
        let input3d = numpy
            .call_method1("linspace", (-1.0_f64, 1.0_f64, 256_usize * 128_usize * 256_usize))
            .expect("8M f64 3d source")
            .call_method1("reshape", ((256_usize, 128_usize, 256_usize),))
            .expect("256x128x256 reshape");
        group.bench_function("fnp_ptp_axis0_f64_256x128x256", |bench| {
            bench.iter(|| {
                let result = fnp_ptp
                    .call1((&input3d, 0_i64))
                    .expect("fnp ptp 3d axis=0 call");
                black_box(result);
            });
        });
        group.bench_function("numpy_ptp_axis0_f64_256x128x256", |bench| {
            bench.iter(|| {
                let result = numpy_ptp
                    .call1((&input3d, 0_i64))
                    .expect("numpy ptp 3d axis=0 call");
                black_box(result);
            });
        });

        // 2-D axis=0 (2048x2048): outer==1, inner=2048.
        let input2d = numpy
            .call_method1("linspace", (-1.0_f64, 1.0_f64, 2048_usize * 2048_usize))
            .expect("4M f64 source")
            .call_method1("reshape", ((2048_usize, 2048_usize),))
            .expect("2048x2048 reshape");
        group.bench_function("fnp_ptp_axis0_f64_2048x2048", |bench| {
            bench.iter(|| {
                let result = fnp_ptp
                    .call1((&input2d, 0_i64))
                    .expect("fnp ptp 2d axis=0 call");
                black_box(result);
            });
        });
        group.bench_function("numpy_ptp_axis0_f64_2048x2048", |bench| {
            bench.iter(|| {
                let result = numpy_ptp
                    .call1((&input2d, 0_i64))
                    .expect("numpy ptp 2d axis=0 call");
                black_box(result);
            });
        });

        // axis=1 (middle) regression guard: already-fast block-parallel non-last path.
        group.bench_function("fnp_ptp_axis1_f64_256x128x256", |bench| {
            bench.iter(|| {
                let result = fnp_ptp
                    .call1((&input3d, 1_i64))
                    .expect("fnp ptp 3d axis=1 call");
                black_box(result);
            });
        });
        group.bench_function("numpy_ptp_axis1_f64_256x128x256", |bench| {
            bench.iter(|| {
                let result = numpy_ptp
                    .call1((&input3d, 1_i64))
                    .expect("numpy ptp 3d axis=1 call");
                black_box(result);
            });
        });
    });

    group.finish();
}

fn bench_bool_minmax_reduction_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_bool_minmax_reduction_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let input = numpy
            .call_method1("arange", (2048_usize * 2048_usize,))
            .expect("4M bool source")
            .call_method1("__mod__", (3_i64,))
            .expect("periodic bool source")
            .call_method1("__eq__", (0_i64,))
            .expect("periodic bool input")
            .call_method1("reshape", ((2048_usize, 2048_usize),))
            .expect("2048x2048 bool input");
        let fnp_max = module.getattr("max").expect("fnp_python.max");
        let fnp_min = module.getattr("min").expect("fnp_python.min");
        let numpy_max = numpy.getattr("max").expect("numpy.max");
        let numpy_min = numpy.getattr("min").expect("numpy.min");

        group.bench_function("fnp_max_flat_bool_4m", |bench| {
            bench.iter(|| {
                let result = fnp_max.call1((&input,)).expect("fnp max bool flat");
                black_box(result);
            });
        });

        group.bench_function("numpy_max_flat_bool_4m", |bench| {
            bench.iter(|| {
                let result = numpy_max.call1((&input,)).expect("numpy max bool flat");
                black_box(result);
            });
        });

        group.bench_function("fnp_min_flat_bool_4m", |bench| {
            bench.iter(|| {
                let result = fnp_min.call1((&input,)).expect("fnp min bool flat");
                black_box(result);
            });
        });

        group.bench_function("numpy_min_flat_bool_4m", |bench| {
            bench.iter(|| {
                let result = numpy_min.call1((&input,)).expect("numpy min bool flat");
                black_box(result);
            });
        });

        group.bench_function("fnp_max_axis1_bool_2048x2048", |bench| {
            bench.iter(|| {
                let result = fnp_max.call1((&input, 1_i64)).expect("fnp max bool axis=1");
                black_box(result);
            });
        });

        group.bench_function("numpy_max_axis1_bool_2048x2048", |bench| {
            bench.iter(|| {
                let result = numpy_max
                    .call1((&input, 1_i64))
                    .expect("numpy max bool axis=1");
                black_box(result);
            });
        });
    });

    group.finish();
}

fn bench_prod_reduction_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_prod_reduction_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", "int64").expect("dtype kwarg");
        let input_i64 = numpy
            .call_method("full", ((2_000_000_usize,), 3_i64), Some(&kwargs))
            .expect("2M int64 input");
        let fnp_prod = module.getattr("prod").expect("fnp_python.prod");
        let numpy_prod = numpy.getattr("prod").expect("numpy.prod");

        group.bench_function("fnp_prod_i64_2m", |bench| {
            bench.iter(|| {
                let result = fnp_prod
                    .call1((&input_i64,))
                    .expect("fnp prod int64 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_prod_i64_2m", |bench| {
            bench.iter(|| {
                let result = numpy_prod
                    .call1((&input_i64,))
                    .expect("numpy prod int64 benchmark call");
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

        // 8M fnp-vs-numpy comparison (parallel consecutive-diff vs single-threaded numpy).
        let big = numpy
            .call_method1(
                "linspace",
                (-1_000_000.0_f64, 1_000_000.0_f64, 8_000_000_usize),
            )
            .expect("8M f64 input");
        let numpy_ediff1d = numpy.getattr("ediff1d").expect("numpy ediff1d");
        group.bench_function("fnp_ediff1d_f64_8m", |bench| {
            bench.iter(|| black_box(ediff1d.call1((&big,)).expect("fnp ediff1d 8m")));
        });
        group.bench_function("numpy_ediff1d_f64_8m", |bench| {
            bench.iter(|| black_box(numpy_ediff1d.call1((&big,)).expect("numpy ediff1d 8m")));
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

fn bench_float_power_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_float_power_boundary");
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
            .call_method1("linspace", (0.5_f64, 4.5_f64, 2_000_000_usize))
            .expect("2M f64 base input");
        let x2 = numpy
            .call_method1("linspace", (0.25_f64, 2.25_f64, 2_000_000_usize))
            .expect("2M f64 exponent input");
        let fnp_float_power = module
            .getattr("float_power")
            .expect("fnp_python.float_power");
        let numpy_float_power = numpy.getattr("float_power").expect("numpy.float_power");

        group.bench_function("fnp_float_power_f64_2m", |bench| {
            bench.iter(|| {
                let result = fnp_float_power
                    .call1((&x1, &x2))
                    .expect("fnp float_power f64/f64 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_float_power_f64_2m", |bench| {
            bench.iter(|| {
                let result = numpy_float_power
                    .call1((&x1, &x2))
                    .expect("numpy float_power f64/f64 benchmark call");
                black_box(result);
            });
        });
    });

    group.finish();
}

fn bench_frexp_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_frexp_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let input = numpy
            .call_method1(
                "linspace",
                (-1_000_000.0_f64, 1_000_000.0_f64, 1_000_000_usize),
            )
            .expect("1M f64 input");
        let fnp_frexp = module.getattr("frexp").expect("fnp_python.frexp");
        let numpy_frexp = numpy.getattr("frexp").expect("numpy.frexp");

        group.bench_function("frexp_f64_1m", |bench| {
            bench.iter(|| {
                let result = fnp_frexp.call1((&input,)).expect("frexp benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_frexp_f64_1m", |bench| {
            bench.iter(|| {
                let result = numpy_frexp
                    .call1((&input,))
                    .expect("numpy frexp benchmark call");
                black_box(result);
            });
        });
    });

    group.finish();
}

fn bench_modf_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_modf_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let input = numpy
            .call_method1(
                "linspace",
                (-1_000_000.75_f64, 1_000_000.75_f64, 1_000_000_usize),
            )
            .expect("1M f64 input");
        let fnp_modf = module.getattr("modf").expect("fnp_python.modf");
        let numpy_modf = numpy.getattr("modf").expect("numpy.modf");

        group.bench_function("fnp_modf_f64_1m", |bench| {
            bench.iter(|| {
                let result = fnp_modf.call1((&input,)).expect("fnp modf benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_modf_f64_1m", |bench| {
            bench.iter(|| {
                let result = numpy_modf
                    .call1((&input,))
                    .expect("numpy modf benchmark call");
                black_box(result);
            });
        });

        // 8M case: above the 1<<21 parallel gate (the 1M case stays serial).
        let input8 = numpy
            .call_method1(
                "linspace",
                (-1_000_000.75_f64, 1_000_000.75_f64, 8_000_000_usize),
            )
            .expect("8M f64 input");
        group.bench_function("fnp_modf_f64_8m", |bench| {
            bench.iter(|| black_box(fnp_modf.call1((&input8,)).expect("fnp modf 8m")));
        });
        group.bench_function("numpy_modf_f64_8m", |bench| {
            bench.iter(|| black_box(numpy_modf.call1((&input8,)).expect("numpy modf 8m")));
        });
    });

    group.finish();
}

fn bench_putmask_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_putmask_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let n = 1_000_000_i64;
        let index = numpy.call_method1("arange", (n,)).expect("1M index");
        let mask = index
            .call_method1("__mod__", (3_i64,))
            .expect("periodic mask index")
            .call_method1("__eq__", (0_i64,))
            .expect("periodic bool mask");
        let base_u8 = index
            .call_method1("astype", ("uint8",))
            .expect("uint8 putmask base");
        let base_i32 = index
            .call_method1("astype", ("int32",))
            .expect("int32 putmask base");
        let base_f32 = numpy
            .call_method1("linspace", (-1.0_f64, 1.0_f64, n as usize))
            .expect("f32 putmask linspace")
            .call_method1("astype", ("float32",))
            .expect("float32 putmask base");
        let vals_u8 = numpy
            .call_method1("array", (vec![7_i64, 255_i64, 1_i64, 128_i64],))
            .expect("uint8 values")
            .call_method1("astype", ("uint8",))
            .expect("uint8 values dtype");
        let vals_i32 = numpy
            .call_method1(
                "array",
                (vec![-2_000_000_000_i64, 0_i64, 1_234_567_i64, 99_i64],),
            )
            .expect("int32 values")
            .call_method1("astype", ("int32",))
            .expect("int32 values dtype");
        let vals_f32 = numpy
            .call_method1("array", (vec![-0.0_f32, 0.0_f32, f32::INFINITY, f32::NAN],))
            .expect("float32 values")
            .call_method1("astype", ("float32",))
            .expect("float32 values dtype");
        let fnp_putmask = module.getattr("putmask").expect("fnp_python.putmask");
        let numpy_putmask = numpy.getattr("putmask").expect("numpy.putmask");

        group.bench_function("fnp_putmask_u8_1m", |bench| {
            bench.iter(|| {
                let a = base_u8.call_method0("copy").expect("copy uint8 base");
                fnp_putmask
                    .call1((&a, &mask, &vals_u8))
                    .expect("fnp uint8 putmask benchmark call");
                black_box(a);
            });
        });

        group.bench_function("numpy_putmask_u8_1m", |bench| {
            bench.iter(|| {
                let a = base_u8.call_method0("copy").expect("copy uint8 base");
                numpy_putmask
                    .call1((&a, &mask, &vals_u8))
                    .expect("numpy uint8 putmask benchmark call");
                black_box(a);
            });
        });

        group.bench_function("fnp_putmask_i32_1m", |bench| {
            bench.iter(|| {
                let a = base_i32.call_method0("copy").expect("copy int32 base");
                fnp_putmask
                    .call1((&a, &mask, &vals_i32))
                    .expect("fnp int32 putmask benchmark call");
                black_box(a);
            });
        });

        group.bench_function("fnp_putmask_f32_1m", |bench| {
            bench.iter(|| {
                let a = base_f32.call_method0("copy").expect("copy float32 base");
                fnp_putmask
                    .call1((&a, &mask, &vals_f32))
                    .expect("fnp float32 putmask benchmark call");
                black_box(a);
            });
        });
    });

    group.finish();
}

fn bench_shift_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_shift_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let a = numpy
            .call_method1("arange", (1_000_000_i64,))
            .expect("1M int64 input")
            .call_method1("astype", ("int64",))
            .expect("int64 input dtype");
        let shifts = numpy
            .call_method1("arange", (1_000_000_i64,))
            .expect("1M int64 shifts")
            .call_method1("astype", ("int64",))
            .expect("int64 shift dtype")
            .call_method1("__mod__", (70_i64,))
            .expect("bounded shifts")
            .call_method1("__sub__", (3_i64,))
            .expect("signed shifts");
        let fnp_left_shift = module.getattr("left_shift").expect("fnp_python.left_shift");
        let fnp_right_shift = module
            .getattr("right_shift")
            .expect("fnp_python.right_shift");
        let numpy_left_shift = numpy.getattr("left_shift").expect("numpy.left_shift");

        group.bench_function("left_shift_i64_scalar_1m", |bench| {
            bench.iter(|| {
                let result = fnp_left_shift
                    .call1((&a, 7_i64))
                    .expect("left_shift scalar benchmark call");
                black_box(result);
            });
        });

        group.bench_function("right_shift_i64_array_1m", |bench| {
            bench.iter(|| {
                let result = fnp_right_shift
                    .call1((&a, &shifts))
                    .expect("right_shift array benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_left_shift_i64_scalar_1m", |bench| {
            bench.iter(|| {
                let result = numpy_left_shift
                    .call1((&a, 7_i64))
                    .expect("numpy left_shift scalar benchmark call");
                black_box(result);
            });
        });
    });

    group.finish();
}

fn bench_concat_hstack_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_concat_hstack_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let left = numpy
            .call_method1("linspace", (-1.0_f64, 1.0_f64, 1024_usize * 512_usize))
            .expect("left f64 input")
            .call_method1("reshape", ((1024_usize, 512_usize),))
            .expect("left 2-D input");
        let right = numpy
            .call_method1("linspace", (2.0_f64, 3.0_f64, 1024_usize * 256_usize))
            .expect("right f64 input")
            .call_method1("reshape", ((1024_usize, 256_usize),))
            .expect("right 2-D input");
        let arrays = PyTuple::new(py, [&left, &right]).expect("array tuple");
        let concatenate = module
            .getattr("concatenate")
            .expect("fnp_python.concatenate");
        let hstack = module.getattr("hstack").expect("fnp_python.hstack");

        group.bench_function("concatenate_axis1_f64_1024x512_256", |bench| {
            bench.iter(|| {
                let result = concatenate
                    .call1((&arrays, 1_i64))
                    .expect("concatenate axis=1 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("hstack_2d_f64_1024x512_256", |bench| {
            bench.iter(|| {
                let result = hstack.call1((&arrays,)).expect("hstack benchmark call");
                black_box(result);
            });
        });
    });

    group.finish();
}

fn bench_indices_construction_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_indices_construction_boundary");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(2));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_diag_indices = module
            .getattr("diag_indices")
            .expect("fnp_python.diag_indices");
        let numpy_diag_indices = numpy.getattr("diag_indices").expect("numpy.diag_indices");

        for n in [64_i64, 4096_i64] {
            group.bench_function(format!("fnp_diag_indices_{n}_2d"), |bench| {
                bench.iter(|| {
                    let result = fnp_diag_indices
                        .call1((n,))
                        .expect("fnp diag_indices benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("numpy_diag_indices_{n}_2d"), |bench| {
                bench.iter(|| {
                    let result = numpy_diag_indices
                        .call1((n,))
                        .expect("numpy diag_indices benchmark call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

fn bench_char_ascii_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_char_ascii_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", "<U20").expect("dtype kwarg");
        let input = numpy
            .call_method("full", ((1_000_000_usize,), "azByCxD0123_"), Some(&kwargs))
            .expect("1M U20 ASCII input");
        let fnp_char = module.getattr("char").expect("fnp_python.char");
        let numpy_char = numpy.getattr("char").expect("numpy.char");
        let fnp_upper = fnp_char.getattr("upper").expect("fnp.char.upper");
        let fnp_lower = fnp_char.getattr("lower").expect("fnp.char.lower");
        let numpy_upper = numpy_char.getattr("upper").expect("numpy.char.upper");
        let numpy_lower = numpy_char.getattr("lower").expect("numpy.char.lower");

        group.bench_function("fnp_char_upper_u20_ascii_1m", |bench| {
            bench.iter(|| {
                let result = fnp_upper
                    .call1((&input,))
                    .expect("fnp char.upper benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_char_upper_u20_ascii_1m", |bench| {
            bench.iter(|| {
                let result = numpy_upper
                    .call1((&input,))
                    .expect("numpy char.upper benchmark call");
                black_box(result);
            });
        });

        group.bench_function("fnp_char_lower_u20_ascii_1m", |bench| {
            bench.iter(|| {
                let result = fnp_lower
                    .call1((&input,))
                    .expect("fnp char.lower benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_char_lower_u20_ascii_1m", |bench| {
            bench.iter(|| {
                let result = numpy_lower
                    .call1((&input,))
                    .expect("numpy char.lower benchmark call");
                black_box(result);
            });
        });
    });

    group.finish();
}

fn bench_average_nansum_axis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_average_nansum_axis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let rows = 2048_usize;
        let cols = 512_usize;
        let total = rows * cols;
        let input = numpy
            .call_method1("linspace", (-1.0_f64, 1.0_f64, total))
            .expect("f64 input")
            .call_method1("reshape", ((rows, cols),))
            .expect("2-D f64 input");
        let weights = numpy
            .call_method1("linspace", (0.5_f64, 1.5_f64, cols))
            .expect("axis weights");
        let flat_index = numpy
            .call_method1("arange", (total,))
            .expect("flat index")
            .call_method1("reshape", ((rows, cols),))
            .expect("2-D index");
        let nan_mask = numpy
            .getattr("equal")
            .expect("numpy.equal")
            .call1((
                flat_index
                    .call_method1("__mod__", (17_i64,))
                    .expect("mod index"),
                0_i64,
            ))
            .expect("periodic nan mask");
        let nan_value = numpy.getattr("nan").expect("numpy.nan");
        let nan_input = numpy
            .getattr("where")
            .expect("numpy.where")
            .call1((&nan_mask, &nan_value, &input))
            .expect("input with periodic NaNs");

        let average_unweighted_kwargs = PyDict::new(py);
        average_unweighted_kwargs
            .set_item("axis", 1_i64)
            .expect("axis kwarg");
        let average_weighted_kwargs = PyDict::new(py);
        average_weighted_kwargs
            .set_item("axis", 1_i64)
            .expect("axis kwarg");
        average_weighted_kwargs
            .set_item("weights", &weights)
            .expect("weights kwarg");
        let nansum_kwargs = PyDict::new(py);
        nansum_kwargs.set_item("axis", 1_i64).expect("axis kwarg");

        let fnp_average = module.getattr("average").expect("fnp_python.average");
        let numpy_average = numpy.getattr("average").expect("numpy.average");
        let fnp_nansum = module.getattr("nansum").expect("fnp_python.nansum");
        let numpy_nansum = numpy.getattr("nansum").expect("numpy.nansum");

        group.bench_function("fnp_average_axis1_unweighted_f64_2048x512", |bench| {
            bench.iter(|| {
                let result = fnp_average
                    .call((&input,), Some(&average_unweighted_kwargs))
                    .expect("fnp average unweighted axis=1 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_average_axis1_unweighted_f64_2048x512", |bench| {
            bench.iter(|| {
                let result = numpy_average
                    .call((&input,), Some(&average_unweighted_kwargs))
                    .expect("numpy average unweighted axis=1 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("fnp_average_axis1_weighted_f64_2048x512", |bench| {
            bench.iter(|| {
                let result = fnp_average
                    .call((&input,), Some(&average_weighted_kwargs))
                    .expect("fnp average axis=1 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_average_axis1_weighted_f64_2048x512", |bench| {
            bench.iter(|| {
                let result = numpy_average
                    .call((&input,), Some(&average_weighted_kwargs))
                    .expect("numpy average axis=1 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("fnp_nansum_axis1_f64_2048x512", |bench| {
            bench.iter(|| {
                let result = fnp_nansum
                    .call((&nan_input,), Some(&nansum_kwargs))
                    .expect("fnp nansum axis=1 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_nansum_axis1_f64_2048x512", |bench| {
            bench.iter(|| {
                let result = numpy_nansum
                    .call((&nan_input,), Some(&nansum_kwargs))
                    .expect("numpy nansum axis=1 benchmark call");
                black_box(result);
            });
        });
    });

    group.finish();
}

fn bench_histogram_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_histogram_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let hist = module.getattr("histogram").expect("fnp_python.histogram");
        let numpy_hist = numpy.getattr("histogram").expect("numpy.histogram");
        let kwargs = PyDict::new(py);
        kwargs.set_item("bins", 50_i64).expect("bins kwarg");
        let int_input = numpy
            .call_method1("arange", (100_000_i64,))
            .expect("100k int input")
            .call_method1("__mod__", (5000_i64,))
            .expect("bounded int range")
            .call_method1("astype", ("int64",))
            .expect("int64 input");
        let float32_input = numpy
            .call_method1("linspace", (-1000.0_f64, 1000.0_f64, 100_000_usize))
            .expect("100k f32 input")
            .call_method1("astype", ("float32",))
            .expect("float32 input");

        group.bench_function("fnp_histogram_i64_100k_50", |bench| {
            bench.iter(|| {
                let result = hist
                    .call((&int_input,), Some(&kwargs))
                    .expect("fnp int histogram benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_histogram_i64_100k_50", |bench| {
            bench.iter(|| {
                let result = numpy_hist
                    .call((&int_input,), Some(&kwargs))
                    .expect("numpy int histogram benchmark call");
                black_box(result);
            });
        });

        group.bench_function("fnp_histogram_f32_100k_50", |bench| {
            bench.iter(|| {
                let result = hist
                    .call((&float32_input,), Some(&kwargs))
                    .expect("fnp f32 histogram benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_histogram_f32_100k_50", |bench| {
            bench.iter(|| {
                let result = numpy_hist
                    .call((&float32_input,), Some(&kwargs))
                    .expect("numpy f32 histogram benchmark call");
                black_box(result);
            });
        });

        // Large f64 inputs (256 bins): >= the 2M parallel gate, where the privatized
        // par_chunks tally (fold-trap fixed) beats numpy's single-threaded reduce 4-8x.
        let big_kwargs = PyDict::new(py);
        big_kwargs.set_item("bins", 256_i64).expect("bins kwarg");
        let setup = "import numpy as np\n\
rng = np.random.default_rng(0)\n\
x4 = rng.standard_normal(4_000_000)\n\
x8 = rng.standard_normal(8_000_000)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("histogram big setup");
        let x4 = ns.get_item("x4").expect("x4");
        let x8 = ns.get_item("x8").expect("x8");
        for (label, x) in [("f64_4m_256", &x4), ("f64_8m_256", &x8)] {
            group.bench_function(format!("fnp_histogram_{label}"), |bench| {
                bench.iter(|| black_box(hist.call((x,), Some(&big_kwargs)).expect("fnp hist big")));
            });
            group.bench_function(format!("numpy_histogram_{label}"), |bench| {
                bench.iter(|| {
                    black_box(numpy_hist.call((x,), Some(&big_kwargs)).expect("numpy hist big"))
                });
            });
        }
    });

    group.finish();
}

fn bench_setops_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_setops_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");

        let n = 1_000_000_i64;
        let raw = numpy.call_method1("arange", (n,)).expect("raw arange");
        let left_i32 = raw
            .call_method1("__mod__", (4096_i64,))
            .expect("left i32 modulo")
            .call_method1("__sub__", (2048_i64,))
            .expect("left i32 center")
            .call_method1("astype", ("int32",))
            .expect("left int32");
        let right_i32 = raw
            .call_method1("__mul__", (3_i64,))
            .expect("right i32 mul")
            .call_method1("__mod__", (4096_i64,))
            .expect("right i32 modulo")
            .call_method1("__sub__", (1024_i64,))
            .expect("right i32 center")
            .call_method1("astype", ("int32",))
            .expect("right int32");
        let left_i64 = raw
            .call_method1("__mod__", (8192_i64,))
            .expect("left i64 modulo")
            .call_method1("__sub__", (4096_i64,))
            .expect("left i64 center")
            .call_method1("astype", ("int64",))
            .expect("left int64");
        let right_i64 = raw
            .call_method1("__mul__", (5_i64,))
            .expect("right i64 mul")
            .call_method1("__mod__", (8192_i64,))
            .expect("right i64 modulo")
            .call_method1("__sub__", (2048_i64,))
            .expect("right i64 center")
            .call_method1("astype", ("int64",))
            .expect("right int64");
        let left_f64 = raw
            .call_method1("__mod__", (65536_i64,))
            .expect("left f64 modulo")
            .call_method1("__truediv__", (16.0_f64,))
            .expect("left f64 scale")
            .call_method1("astype", ("float64",))
            .expect("left float64");
        let right_f64 = raw
            .call_method1("__mul__", (7_i64,))
            .expect("right f64 mul")
            .call_method1("__mod__", (65536_i64,))
            .expect("right f64 modulo")
            .call_method1("__truediv__", (16.0_f64,))
            .expect("right f64 scale")
            .call_method1("astype", ("float64",))
            .expect("right float64");
        let left_f32 = raw
            .call_method1("__mod__", (32768_i64,))
            .expect("left f32 modulo")
            .call_method1("__truediv__", (8.0_f64,))
            .expect("left f32 scale")
            .call_method1("astype", ("float32",))
            .expect("left float32");
        let right_f32 = raw
            .call_method1("__mul__", (11_i64,))
            .expect("right f32 mul")
            .call_method1("__mod__", (32768_i64,))
            .expect("right f32 modulo")
            .call_method1("__truediv__", (8.0_f64,))
            .expect("right f32 scale")
            .call_method1("astype", ("float32",))
            .expect("right float32");

        let fnp_setdiff1d = module.getattr("setdiff1d").expect("fnp setdiff1d");
        let numpy_setdiff1d = numpy.getattr("setdiff1d").expect("numpy setdiff1d");
        let fnp_intersect1d = module.getattr("intersect1d").expect("fnp intersect1d");
        let numpy_intersect1d = numpy.getattr("intersect1d").expect("numpy intersect1d");

        group.bench_function("fnp_setdiff1d_i32_smallrange_1m", |bench| {
            bench.iter(|| {
                let result = fnp_setdiff1d
                    .call1((&left_i32, &right_i32))
                    .expect("fnp setdiff1d i32 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_setdiff1d_i32_smallrange_1m", |bench| {
            bench.iter(|| {
                let result = numpy_setdiff1d
                    .call1((&left_i32, &right_i32))
                    .expect("numpy setdiff1d i32 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("fnp_intersect1d_i64_smallrange_1m", |bench| {
            bench.iter(|| {
                let result = fnp_intersect1d
                    .call1((&left_i64, &right_i64))
                    .expect("fnp intersect1d i64 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_intersect1d_i64_smallrange_1m", |bench| {
            bench.iter(|| {
                let result = numpy_intersect1d
                    .call1((&left_i64, &right_i64))
                    .expect("numpy intersect1d i64 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("fnp_intersect1d_f64_repeated_1m", |bench| {
            bench.iter(|| {
                let result = fnp_intersect1d
                    .call1((&left_f64, &right_f64))
                    .expect("fnp intersect1d f64 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_intersect1d_f64_repeated_1m", |bench| {
            bench.iter(|| {
                let result = numpy_intersect1d
                    .call1((&left_f64, &right_f64))
                    .expect("numpy intersect1d f64 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("fnp_setxor1d_f32_repeated_1m", |bench| {
            bench.iter(|| {
                let result = module
                    .getattr("setxor1d")
                    .expect("fnp setxor1d")
                    .call1((&left_f32, &right_f32))
                    .expect("fnp setxor1d f32 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_setxor1d_f32_repeated_1m", |bench| {
            bench.iter(|| {
                let result = numpy
                    .getattr("setxor1d")
                    .expect("numpy setxor1d")
                    .call1((&left_f32, &right_f32))
                    .expect("numpy setxor1d f32 benchmark call");
                black_box(result);
            });
        });
    });

    group.finish();
}

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

        for (label, input) in [
            ("50k", make_repeated_f64(50_000)),
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

fn bench_statistics_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_statistics_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_cov = module.getattr("cov").expect("fnp_python.cov");
        let numpy_cov = numpy.getattr("cov").expect("numpy.cov");
        let fnp_corrcoef = module.getattr("corrcoef").expect("fnp_python.corrcoef");
        let numpy_corrcoef = numpy.getattr("corrcoef").expect("numpy.corrcoef");

        let make_input = |rows: usize, cols: usize| {
            let total = rows * cols;
            numpy
                .call_method1("linspace", (-2.0_f64, 3.0_f64, total))
                .expect("cov f64 input")
                .call_method1("reshape", ((rows, cols),))
                .expect("2-D cov input")
        };
        let inputs = [
            ("50x1000", make_input(50, 1000)),
            ("200x500", make_input(200, 500)),
            ("500x500", make_input(500, 500)),
            ("50x10000", make_input(50, 10_000)),
        ];

        for (shape, input) in inputs {
            group.bench_function(format!("fnp_cov_rowvar_f64_{shape}"), |bench| {
                bench.iter(|| {
                    let result = fnp_cov.call1((&input,)).expect("fnp cov benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("numpy_cov_rowvar_f64_{shape}"), |bench| {
                bench.iter(|| {
                    let result = numpy_cov
                        .call1((&input,))
                        .expect("numpy cov benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("fnp_corrcoef_rowvar_f64_{shape}"), |bench| {
                bench.iter(|| {
                    let result = fnp_corrcoef
                        .call1((&input,))
                        .expect("fnp corrcoef benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("numpy_corrcoef_rowvar_f64_{shape}"), |bench| {
                bench.iter(|| {
                    let result = numpy_corrcoef
                        .call1((&input,))
                        .expect("numpy corrcoef benchmark call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

fn bench_std_var_axis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_std_var_axis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_std = module.getattr("std").expect("fnp_python.std");
        let numpy_std = numpy.getattr("std").expect("numpy.std");
        let fnp_var = module.getattr("var").expect("fnp_python.var");
        let numpy_var = numpy.getattr("var").expect("numpy.var");

        for (label, rows, cols) in [
            ("4096x512", 4096_i64, 512_i64),
            ("8192x1024", 8192_i64, 1024_i64),
        ] {
            let size = rows * cols;
            let input = numpy
                .call_method1("linspace", (-4.0_f64, 6.0_f64, size))
                .expect("std/var axis f64 input")
                .call_method1("reshape", ((rows, cols),))
                .expect("std/var axis 2-D shape");

            group.bench_function(format!("fnp_var_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_var
                        .call1((&input, -1_i64))
                        .expect("fnp var axis benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("numpy_var_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_var
                        .call1((&input, -1_i64))
                        .expect("numpy var axis benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("fnp_std_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_std
                        .call1((&input, -1_i64))
                        .expect("fnp std axis benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("numpy_std_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_std
                        .call1((&input, -1_i64))
                        .expect("numpy std axis benchmark call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

fn bench_var_multiaxis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_var_multiaxis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_var = module.getattr("var").expect("fnp_python.var");
        let numpy_var = numpy.getattr("var").expect("numpy.var");
        let fnp_std = module.getattr("std").expect("fnp_python.std");
        let numpy_std = numpy.getattr("std").expect("numpy.std");
        let fnp_nanvar = module.getattr("nanvar").expect("fnp_python.nanvar");
        let numpy_nanvar = numpy.getattr("nanvar").expect("numpy.nanvar");
        let fnp_nanstd = module.getattr("nanstd").expect("fnp_python.nanstd");
        let numpy_nanstd = numpy.getattr("nanstd").expect("numpy.nanstd");

        for (label, b, m, n) in [
            ("4096x16x16", 4096_i64, 16_i64, 16_i64),
            ("2048x32x32", 2048_i64, 32_i64, 32_i64),
        ] {
            let size = b * m * n;
            let input = numpy
                .call_method1("linspace", (-4.0_f64, 6.0_f64, size))
                .expect("var multiaxis f64 input")
                .call_method1("reshape", ((b, m, n),))
                .expect("var multiaxis 3-D shape");

            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs
                .set_item("axis", (-2_i64, -1_i64))
                .expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs
                .set_item("axis", (-2_i64, -1_i64))
                .expect("numpy axis kwarg");

            group.bench_function(format!("fnp_var_f64_axis_m2m1_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_var
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp var multiaxis call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_var_f64_axis_m2m1_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_var
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy var multiaxis call");
                    black_box(result);
                });
            });
            group.bench_function(format!("fnp_std_f64_axis_m2m1_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_std
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp std multiaxis call");
                    black_box(result);
                });
            });
            group.bench_function(format!("fnp_nanvar_f64_axis_m2m1_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_nanvar
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp nanvar multiaxis call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_nanvar_f64_axis_m2m1_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_nanvar
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy nanvar multiaxis call");
                    black_box(result);
                });
            });
            group.bench_function(format!("fnp_nanstd_f64_axis_m2m1_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_nanstd
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp nanstd multiaxis call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_nanstd_f64_axis_m2m1_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_nanstd
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy nanstd multiaxis call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_std_f64_axis_m2m1_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_std
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy std multiaxis call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

// var/std along a MIDDLE axis (0 < ax < ndim-1) of a 3-D f64 stack. numpy reduces a
// non-last axis with a strided, two-temp-materializing pass; the native block-parallel
// streaming two-pass (try_zerocopy_f64_var_nonlast_axis) is bit-exact and much faster.
fn bench_var_midaxis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_var_midaxis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_var = module.getattr("var").expect("fnp_python.var");
        let numpy_var = numpy.getattr("var").expect("numpy.var");
        let fnp_std = module.getattr("std").expect("fnp_python.std");
        let numpy_std = numpy.getattr("std").expect("numpy.std");

        for (label, d0, d1, d2) in [
            ("256x256x64", 256_usize, 256_usize, 64_usize),
            ("128x512x64", 128_usize, 512_usize, 64_usize),
        ] {
            let size = (d0 * d1 * d2) as i64;
            let input = numpy
                .call_method1("linspace", (-4.0_f64, 6.0_f64, size))
                .expect("var midaxis f64 input")
                .call_method1("reshape", ((d0, d1, d2),))
                .expect("var midaxis 3-D shape");
            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("axis", 1_i64).expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs.set_item("axis", 1_i64).expect("numpy axis kwarg");

            group.bench_function(format!("fnp_var_f64_axis1_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_var
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp var axis1 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_var_f64_axis1_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_var
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy var axis1 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("fnp_std_f64_axis1_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_std
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp std axis1 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_std_f64_axis1_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_std
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy std axis1 call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

// FLOAT32 var/std along a non-last axis (middle ax=1 + axis 0). numpy keeps the float32
// accumulator and on a non-last axis reduces SEQUENTIALLY while materializing the (a-mean)
// and (a-mean)^2 whole-array f32 temps (~28ms@8M middle); try_zerocopy_f32_var_nonlast_axis
// runs a per-block sequential f32 two-pass (block-parallel for a middle axis, serial for
// axis 0) with no temp -> bit-identical and 3-33x faster. The f32 sibling of the f64
// midaxis/axis0 paths above.
fn bench_var_f32_axis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_var_f32_axis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let f32_dtype = numpy.getattr("float32").expect("numpy.float32");
        let fnp_var = module.getattr("var").expect("fnp_python.var");
        let numpy_var = numpy.getattr("var").expect("numpy.var");
        let fnp_std = module.getattr("std").expect("fnp_python.std");
        let numpy_std = numpy.getattr("std").expect("numpy.std");

        // (label, shape, reduce-axis): a middle axis (block-parallel) and axis 0 (serial).
        let mid = numpy
            .call_method1("linspace", (-4.0_f64, 6.0_f64, 256_i64 * 128 * 256))
            .expect("f32 mid source")
            .call_method1("reshape", ((256_usize, 128_usize, 256_usize),))
            .expect("256x128x256 reshape")
            .call_method1("astype", (&f32_dtype,))
            .expect("astype f32");
        let ax0 = numpy
            .call_method1("linspace", (-4.0_f64, 6.0_f64, 4000_i64 * 2000))
            .expect("f32 ax0 source")
            .call_method1("reshape", ((4000_usize, 2000_usize),))
            .expect("4000x2000 reshape")
            .call_method1("astype", (&f32_dtype,))
            .expect("astype f32");

        for (label, input, axis) in [("mid_256x128x256", &mid, 1_i64), ("axis0_4000x2000", &ax0, 0_i64)] {
            let fkw = PyDict::new(py);
            fkw.set_item("axis", axis).expect("axis");
            let nkw = PyDict::new(py);
            nkw.set_item("axis", axis).expect("axis");
            group.bench_function(format!("fnp_var_f32_{label}"), |bench| {
                bench.iter(|| black_box(fnp_var.call((input,), Some(&fkw)).expect("fnp var f32")));
            });
            group.bench_function(format!("numpy_var_f32_{label}"), |bench| {
                bench.iter(|| black_box(numpy_var.call((input,), Some(&nkw)).expect("numpy var f32")));
            });
            group.bench_function(format!("fnp_std_f32_{label}"), |bench| {
                bench.iter(|| black_box(fnp_std.call((input,), Some(&fkw)).expect("fnp std f32")));
            });
            group.bench_function(format!("numpy_std_f32_{label}"), |bench| {
                bench.iter(|| black_box(numpy_std.call((input,), Some(&nkw)).expect("numpy std f32")));
            });
        }
    });

    group.finish();
}

// FLOAT32 nanvar/nanstd along a non-last axis (middle ax=1 + axis 0) of a 3-D/2-D stack
// with ~10% NaN. numpy.nanvar on float32 keeps the f32 accumulator and materializes a
// NaN->0 copy, an isnan mask, a count, and the (a-mean)/squared f32 temps before two
// sequential strided reduces (~70-77ms@8M middle); try_zerocopy_f32_nanvar_nonlast_axis
// runs a per-block sequential f32 NaN-skip two-pass (block-parallel middle / serial axis0)
// with no temp -> bit-identical and 5-35x faster. f32 sibling of the f64 nanvar paths.
fn bench_nanvar_f32_axis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_nanvar_f32_axis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let f32_dtype = numpy.getattr("float32").expect("numpy.float32");
        let nan = numpy.getattr("nan").expect("np.nan");
        let fnp_nanvar = module.getattr("nanvar").expect("fnp_python.nanvar");
        let numpy_nanvar = numpy.getattr("nanvar").expect("numpy.nanvar");
        let fnp_nanstd = module.getattr("nanstd").expect("fnp_python.nanstd");
        let numpy_nanstd = numpy.getattr("nanstd").expect("numpy.nanstd");
        let fnp_nanmean = module.getattr("nanmean").expect("fnp_python.nanmean");
        let numpy_nanmean = numpy.getattr("nanmean").expect("numpy.nanmean");

        // Build an f32 array with ~10% NaN (deterministic stride), reshape to target.
        let build = |dims: &[usize], total: i64| {
            let arr = numpy
                .call_method1("linspace", (-4.0_f64, 6.0_f64, total))
                .expect("f32 nan source")
                .call_method1("astype", (&f32_dtype,))
                .expect("astype f32");
            let idx = numpy
                .call_method1("arange", (0_i64, total, 10_i64))
                .expect("nan stride");
            arr.call_method1("__setitem__", (idx, &nan)).expect("inject NaN");
            arr.call_method1("reshape", (PyTuple::new(py, dims.iter().copied()).unwrap(),))
                .expect("reshape")
        };
        let mid = build(&[256, 128, 256], 256 * 128 * 256);
        let ax0 = build(&[4000, 2000], 4000 * 2000);

        for (label, input, axis) in [("mid_256x128x256", &mid, 1_i64), ("axis0_4000x2000", &ax0, 0_i64)] {
            let fkw = PyDict::new(py);
            fkw.set_item("axis", axis).expect("axis");
            let nkw = PyDict::new(py);
            nkw.set_item("axis", axis).expect("axis");
            group.bench_function(format!("fnp_nanvar_f32_{label}"), |b| {
                b.iter(|| black_box(fnp_nanvar.call((input,), Some(&fkw)).expect("fnp nanvar f32")));
            });
            group.bench_function(format!("numpy_nanvar_f32_{label}"), |b| {
                b.iter(|| black_box(numpy_nanvar.call((input,), Some(&nkw)).expect("numpy nanvar f32")));
            });
            group.bench_function(format!("fnp_nanstd_f32_{label}"), |b| {
                b.iter(|| black_box(fnp_nanstd.call((input,), Some(&fkw)).expect("fnp nanstd f32")));
            });
            group.bench_function(format!("numpy_nanstd_f32_{label}"), |b| {
                b.iter(|| black_box(numpy_nanstd.call((input,), Some(&nkw)).expect("numpy nanstd f32")));
            });
            group.bench_function(format!("fnp_nanmean_f32_{label}"), |b| {
                b.iter(|| black_box(fnp_nanmean.call((input,), Some(&fkw)).expect("fnp nanmean f32")));
            });
            group.bench_function(format!("numpy_nanmean_f32_{label}"), |b| {
                b.iter(|| black_box(numpy_nanmean.call((input,), Some(&nkw)).expect("numpy nanmean f32")));
            });
        }
    });

    group.finish();
}

// nanvar/nanstd/nanmean along the CONTIGUOUS LAST axis (and a trailing tuple) of an f32
// stack with ~10% NaN. numpy.nanmean/nanvar on float32 materializes a NaN->0 copy + isnan
// mask then PAIRWISE-reduces the last axis (~3-7ms/2M unloaded, far worse loaded); the
// native per-lane bit-exact f32 pairwise paths (try_zerocopy_f32_nanmean_last_axis /
// try_zerocopy_f32_nanvar_last_axis) parallelize across the independent lanes.
fn bench_nanvar_f32_last_axis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_nanvar_f32_last_axis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let f32_dtype = numpy.getattr("float32").expect("numpy.float32");
        let nan = numpy.getattr("nan").expect("np.nan");
        let fnp_nanvar = module.getattr("nanvar").expect("fnp_python.nanvar");
        let numpy_nanvar = numpy.getattr("nanvar").expect("numpy.nanvar");
        let fnp_nanstd = module.getattr("nanstd").expect("fnp_python.nanstd");
        let numpy_nanstd = numpy.getattr("nanstd").expect("numpy.nanstd");
        let fnp_nanmean = module.getattr("nanmean").expect("fnp_python.nanmean");
        let numpy_nanmean = numpy.getattr("nanmean").expect("numpy.nanmean");

        let build = |dims: &[usize], total: i64| {
            let arr = numpy
                .call_method1("linspace", (-4.0_f64, 6.0_f64, total))
                .expect("f32 nan source")
                .call_method1("astype", (&f32_dtype,))
                .expect("astype f32");
            let idx = numpy
                .call_method1("arange", (0_i64, total, 10_i64))
                .expect("nan stride");
            arr.call_method1("__setitem__", (idx, &nan)).expect("inject NaN");
            arr.call_method1("reshape", (PyTuple::new(py, dims.iter().copied()).unwrap(),))
                .expect("reshape")
        };
        let last2d = build(&[1000, 2048], 1000 * 2048);
        let trail3d = build(&[512, 64, 64], 512 * 64 * 64);

        // Per-case kwargs dicts (axis=-1 for the single last axis; axis=(-2,-1) for the
        // contiguous trailing tuple). Built up front so the case loop is homogeneous.
        let fkw_last = PyDict::new(py);
        fkw_last.set_item("axis", -1_i64).expect("axis");
        let fkw_trail = PyDict::new(py);
        fkw_trail.set_item("axis", (-2_i64, -1_i64)).expect("axis");
        let cases = [
            ("last_1000x2048", &last2d, &fkw_last),
            ("trail_512x64x64", &trail3d, &fkw_trail),
        ];
        for (label, input, kw) in cases {
            let fkw = kw;
            let nkw = kw;
            group.bench_function(format!("fnp_nanvar_f32_{label}"), |b| {
                b.iter(|| black_box(fnp_nanvar.call((input,), Some(&fkw)).expect("fnp nanvar f32")));
            });
            group.bench_function(format!("numpy_nanvar_f32_{label}"), |b| {
                b.iter(|| black_box(numpy_nanvar.call((input,), Some(&nkw)).expect("numpy nanvar f32")));
            });
            group.bench_function(format!("fnp_nanstd_f32_{label}"), |b| {
                b.iter(|| black_box(fnp_nanstd.call((input,), Some(&fkw)).expect("fnp nanstd f32")));
            });
            group.bench_function(format!("numpy_nanstd_f32_{label}"), |b| {
                b.iter(|| black_box(numpy_nanstd.call((input,), Some(&nkw)).expect("numpy nanstd f32")));
            });
            group.bench_function(format!("fnp_nanmean_f32_{label}"), |b| {
                b.iter(|| black_box(fnp_nanmean.call((input,), Some(&fkw)).expect("fnp nanmean f32")));
            });
            group.bench_function(format!("numpy_nanmean_f32_{label}"), |b| {
                b.iter(|| black_box(numpy_nanmean.call((input,), Some(&nkw)).expect("numpy nanmean f32")));
            });
        }
    });

    group.finish();
}

// nanvar/nanstd along a MIDDLE axis (0 < ax < ndim-1) of a 3-D f64 stack with scattered
// NaN. numpy.nanvar on a non-last axis materializes a NaN->0 copy, an isnan mask, a count,
// and the (a-mean)/squared temps then strided-reduces; the native block-parallel NaN-skip
// two-pass (try_zerocopy_f64_nanvar_nonlast_axis) is bit-exact and far faster.
fn bench_nanvar_midaxis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_nanvar_midaxis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_nanvar = module.getattr("nanvar").expect("fnp_python.nanvar");
        let numpy_nanvar = numpy.getattr("nanvar").expect("numpy.nanvar");
        let fnp_nanstd = module.getattr("nanstd").expect("fnp_python.nanstd");
        let numpy_nanstd = numpy.getattr("nanstd").expect("numpy.nanstd");
        let fnp_nanmean = module.getattr("nanmean").expect("fnp_python.nanmean");
        let numpy_nanmean = numpy.getattr("nanmean").expect("numpy.nanmean");

        for (label, d0, d1, d2) in [
            ("256x256x64", 256_usize, 256_usize, 64_usize),
            ("128x512x64", 128_usize, 512_usize, 64_usize),
        ] {
            let size = (d0 * d1 * d2) as i64;
            // Build a 3-D f64 array, then poke ~10% NaN into it (deterministic stride).
            let input = numpy
                .call_method1("linspace", (-4.0_f64, 6.0_f64, size))
                .expect("nanvar midaxis f64 input")
                .call_method1("reshape", ((d0, d1, d2),))
                .expect("nanvar midaxis 3-D shape");
            let flat = input.call_method1("reshape", ((size,),)).expect("flat view");
            let idx = numpy
                .call_method1("arange", (0_i64, size, 10_i64))
                .expect("nan index stride");
            let nan = numpy.getattr("nan").expect("np.nan");
            flat.call_method1("__setitem__", (idx, nan))
                .expect("inject NaN");

            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("axis", 1_i64).expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs.set_item("axis", 1_i64).expect("numpy axis kwarg");

            group.bench_function(format!("fnp_nanvar_f64_axis1_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_nanvar
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp nanvar axis1 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_nanvar_f64_axis1_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_nanvar
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy nanvar axis1 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("fnp_nanstd_f64_axis1_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_nanstd
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp nanstd axis1 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_nanstd_f64_axis1_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_nanstd
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy nanstd axis1 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("fnp_nanmean_f64_axis1_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_nanmean
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp nanmean axis1 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_nanmean_f64_axis1_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_nanmean
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy nanmean axis1 call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

fn bench_var_axis0_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_var_axis0_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_var = module.getattr("var").expect("fnp_python.var");
        let numpy_var = numpy.getattr("var").expect("numpy.var");
        let fnp_std = module.getattr("std").expect("fnp_python.std");
        let numpy_std = numpy.getattr("std").expect("numpy.std");
        let fnp_nanvar = module.getattr("nanvar").expect("fnp_python.nanvar");
        let numpy_nanvar = numpy.getattr("nanvar").expect("numpy.nanvar");
        let fnp_nanstd = module.getattr("nanstd").expect("fnp_python.nanstd");
        let numpy_nanstd = numpy.getattr("nanstd").expect("numpy.nanstd");
        let fnp_nanmean = module.getattr("nanmean").expect("fnp_python.nanmean");
        let numpy_nanmean = numpy.getattr("nanmean").expect("numpy.nanmean");

        for (label, rows, cols) in [
            ("4096x512", 4096_i64, 512_i64),
            ("50000x64", 50000_i64, 64_i64),
        ] {
            let size = rows * cols;
            let input = numpy
                .call_method1("linspace", (-4.0_f64, 6.0_f64, size))
                .expect("var axis0 f64 input")
                .call_method1("reshape", ((rows, cols),))
                .expect("var axis0 2-D shape");

            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("axis", 0_i64).expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs.set_item("axis", 0_i64).expect("numpy axis kwarg");

            group.bench_function(format!("fnp_var_f64_axis0_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_var
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp var axis0 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_var_f64_axis0_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_var
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy var axis0 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("fnp_std_f64_axis0_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_std
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp std axis0 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_std_f64_axis0_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_std
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy std axis0 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("fnp_nanvar_f64_axis0_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_nanvar
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp nanvar axis0 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_nanvar_f64_axis0_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_nanvar
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy nanvar axis0 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("fnp_nanstd_f64_axis0_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_nanstd
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp nanstd axis0 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_nanstd_f64_axis0_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_nanstd
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy nanstd axis0 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("fnp_nanmean_f64_axis0_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_nanmean
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp nanmean axis0 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_nanmean_f64_axis0_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_nanmean
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy nanmean axis0 call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

fn bench_sum_lastaxis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_sum_lastaxis_boundary");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_sum = module.getattr("sum").expect("fnp_python.sum");
        let numpy_sum = numpy.getattr("sum").expect("numpy.sum");

        for (label, rows, cols) in [
            ("8192x1024", 8192_i64, 1024_i64),
            ("65536x256", 65536_i64, 256_i64),
        ] {
            let size = rows * cols;
            let input = numpy
                .call_method1("linspace", (-2.0_f64, 3.0_f64, size))
                .expect("sum input")
                .call_method1("reshape", ((rows, cols),))
                .expect("sum 2-D shape");
            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("axis", -1_i64).expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs
                .set_item("axis", -1_i64)
                .expect("numpy axis kwarg");

            group.bench_function(format!("fnp_sum_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_sum
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp sum call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_sum_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_sum
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy sum call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

fn bench_prod_lastaxis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_prod_lastaxis_boundary");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_prod = module.getattr("prod").expect("fnp_python.prod");
        let numpy_prod = numpy.getattr("prod").expect("numpy.prod");

        for (label, rows, cols) in [("8192x1024", 8192_i64, 1024_i64), ("65536x256", 65536_i64, 256_i64)]
        {
            let size = rows * cols;
            // values near 1.0 so the product stays finite across the axis.
            let input = numpy
                .call_method1("linspace", (0.9999_f64, 1.0001_f64, size))
                .expect("prod input")
                .call_method1("reshape", ((rows, cols),))
                .expect("prod 2-D shape");
            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("axis", -1_i64).expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs.set_item("axis", -1_i64).expect("numpy axis kwarg");

            group.bench_function(format!("fnp_prod_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_prod
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp prod call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_prod_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_prod
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy prod call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

fn bench_cumsum_lastaxis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_cumsum_lastaxis_boundary");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_cumsum = module.getattr("cumsum").expect("fnp_python.cumsum");
        let numpy_cumsum = numpy.getattr("cumsum").expect("numpy.cumsum");

        for (label, rows, cols) in [("8192x1024", 8192_i64, 1024_i64), ("65536x256", 65536_i64, 256_i64)]
        {
            let size = rows * cols;
            let input = numpy
                .call_method1("linspace", (-1.0_f64, 1.0_f64, size))
                .expect("cumsum input")
                .call_method1("reshape", ((rows, cols),))
                .expect("cumsum 2-D shape");
            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("axis", -1_i64).expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs.set_item("axis", -1_i64).expect("numpy axis kwarg");

            group.bench_function(format!("fnp_cumsum_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_cumsum
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp cumsum call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_cumsum_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_cumsum
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy cumsum call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

// cumsum/cumprod along a MIDDLE axis (0 < ax < ndim-1) of a 3-D f64 stack. numpy runs a
// non-last cumulative STRIDED + single-threaded; the native block-parallel slab-by-slab
// scan (try_zerocopy_f64_cumulative_axis inner>1 path) fans independent contiguous outer
// blocks across the pool. RAYON_NUM_THREADS=1 vs default isolates the parallelism gain.
fn bench_cum_midaxis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_cum_midaxis_boundary");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_cumsum = module.getattr("cumsum").expect("fnp_python.cumsum");
        let numpy_cumsum = numpy.getattr("cumsum").expect("numpy.cumsum");
        let fnp_cumprod = module.getattr("cumprod").expect("fnp_python.cumprod");
        let numpy_cumprod = numpy.getattr("cumprod").expect("numpy.cumprod");

        for (label, d0, d1, d2) in [
            ("256x256x64", 256_usize, 256_usize, 64_usize),
            ("128x512x64", 128_usize, 512_usize, 64_usize),
        ] {
            let size = (d0 * d1 * d2) as i64;
            let input = numpy
                .call_method1("linspace", (-1.0_f64, 1.0_f64, size))
                .expect("cum midaxis input")
                .call_method1("reshape", ((d0, d1, d2),))
                .expect("cum midaxis 3-D shape");
            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("axis", 1_i64).expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs.set_item("axis", 1_i64).expect("numpy axis kwarg");

            group.bench_function(format!("fnp_cumsum_f64_axis1_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_cumsum
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp cumsum axis1 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_cumsum_f64_axis1_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_cumsum
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy cumsum axis1 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("fnp_cumprod_f64_axis1_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_cumprod
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp cumprod axis1 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_cumprod_f64_axis1_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_cumprod
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy cumprod axis1 call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

// int64 cumsum along the LAST axis and a MIDDLE axis. numpy runs int cumsum
// single-threaded (strided on a non-last axis); the native cumsum_axis_typed path now
// fans independent contiguous lanes (last) / outer blocks (non-last) across the pool.
// RAYON_NUM_THREADS=1 vs default isolates the parallelism gain.
fn bench_int_cum_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_int_cum_boundary");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_cumsum = module.getattr("cumsum").expect("fnp_python.cumsum");
        let numpy_cumsum = numpy.getattr("cumsum").expect("numpy.cumsum");

        // last axis: 2-D (8192, 1024)
        let last2d = numpy
            .call_method1("arange", (8192_i64 * 1024_i64,))
            .expect("int last input")
            .call_method1("reshape", ((8192_usize, 1024_usize),))
            .expect("int last reshape");
        // middle axis: 3-D (256, 256, 64)
        let mid3d = numpy
            .call_method1("arange", (256_i64 * 256_i64 * 64_i64,))
            .expect("int mid input")
            .call_method1("reshape", ((256_usize, 256_usize, 64_usize),))
            .expect("int mid reshape");
        for (label, arr, ax) in [
            ("last_8192x1024", &last2d, -1_i64),
            ("mid_256x256x64", &mid3d, 1_i64),
        ] {
            let fk = PyDict::new(py);
            fk.set_item("axis", ax).expect("fnp axis");
            let nk = PyDict::new(py);
            nk.set_item("axis", ax).expect("np axis");
            group.bench_function(format!("fnp_cumsum_i64_{label}"), |bench| {
                bench.iter(|| {
                    let r = fnp_cumsum.call((arr,), Some(&fk)).expect("fnp int cumsum");
                    black_box(r);
                });
            });
            group.bench_function(format!("numpy_cumsum_i64_{label}"), |bench| {
                bench.iter(|| {
                    let r = numpy_cumsum.call((arr,), Some(&nk)).expect("numpy int cumsum");
                    black_box(r);
                });
            });
        }
    });

    group.finish();
}

fn bench_vander_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_vander_boundary");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_vander = module.getattr("vander").expect("fnp_python.vander");
        let numpy_vander = numpy.getattr("vander").expect("numpy.vander");

        for (label, n, cols) in [("200k_x8", 200_000_i64, 8_i64), ("500k_x12", 500_000_i64, 12_i64)]
        {
            let x = numpy
                .call_method1("linspace", (-1.5_f64, 1.5_f64, n))
                .expect("vander x input");
            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("N", cols).expect("fnp N kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs.set_item("N", cols).expect("numpy N kwarg");

            group.bench_function(format!("fnp_vander_f64_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_vander
                        .call((&x,), Some(&fnp_kwargs))
                        .expect("fnp vander call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_vander_f64_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_vander
                        .call((&x,), Some(&numpy_kwargs))
                        .expect("numpy vander call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

fn bench_polyval_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_polyval_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_polyval = module.getattr("polyval").expect("fnp_python.polyval");
        let numpy_polyval = numpy.getattr("polyval").expect("numpy.polyval");

        for (label, n, deg) in [("1M_deg5", 1_000_000_i64, 5_i64), ("4M_deg8", 4_000_000_i64, 8_i64)]
        {
            let x = numpy
                .call_method1("linspace", (-3.0_f64, 3.0_f64, n))
                .expect("polyval x input");
            let p = numpy
                .call_method1("linspace", (0.5_f64, 2.0_f64, deg + 1))
                .expect("polyval coeffs");

            group.bench_function(format!("fnp_polyval_f64_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_polyval
                        .call1((&p, &x))
                        .expect("fnp polyval call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_polyval_f64_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_polyval
                        .call1((&p, &x))
                        .expect("numpy polyval call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

fn bench_gradient_axis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_gradient_axis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_grad = module.getattr("gradient").expect("fnp_python.gradient");
        let numpy_grad = numpy.getattr("gradient").expect("numpy.gradient");

        for (label, rows, cols) in [
            ("4096x1024", 4096_i64, 1024_i64),
            ("1024x4096", 1024_i64, 4096_i64),
        ] {
            let size = rows * cols;
            let input = numpy
                .call_method1("linspace", (-4.0_f64, 6.0_f64, size))
                .expect("gradient f64 input")
                .call_method1("reshape", ((rows, cols),))
                .expect("gradient 2-D shape");
            // axis=0 is the strided (non-last) path.
            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("axis", 0_i64).expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs.set_item("axis", 0_i64).expect("numpy axis kwarg");

            group.bench_function(format!("fnp_gradient_f64_axis0_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_grad
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp gradient axis0 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_gradient_f64_axis0_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_grad
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy gradient axis0 call");
                    black_box(result);
                });
            });
            // No-axis full gradient: returns a tuple of per-axis gradients.
            group.bench_function(format!("fnp_gradient_f64_full_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_grad.call1((&input,)).expect("fnp gradient full call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_gradient_f64_full_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_grad.call1((&input,)).expect("numpy gradient full call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

fn bench_norm_axis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_norm_axis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_norm = module.getattr("norm").expect("fnp_python.norm");
        let numpy_norm = numpy
            .getattr("linalg")
            .expect("numpy.linalg")
            .getattr("norm")
            .expect("numpy.linalg.norm");

        for (label, rows, cols) in [
            ("4096x512", 4096_i64, 512_i64),
            ("8192x1024", 8192_i64, 1024_i64),
        ] {
            let size = rows * cols;
            let input = numpy
                .call_method1("linspace", (-4.0_f64, 6.0_f64, size))
                .expect("norm axis f64 input")
                .call_method1("reshape", ((rows, cols),))
                .expect("norm axis 2-D shape");

            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("axis", -1_i64).expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs
                .set_item("axis", -1_i64)
                .expect("numpy axis kwarg");

            group.bench_function(format!("fnp_norm_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_norm
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp norm axis benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("numpy_norm_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_norm
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy norm axis benchmark call");
                    black_box(result);
                });
            });

            let fnp_l1_kwargs = PyDict::new(py);
            fnp_l1_kwargs.set_item("ord", 1_i64).expect("fnp l1 ord kwarg");
            fnp_l1_kwargs
                .set_item("axis", -1_i64)
                .expect("fnp l1 axis kwarg");
            let numpy_l1_kwargs = PyDict::new(py);
            numpy_l1_kwargs
                .set_item("ord", 1_i64)
                .expect("numpy l1 ord kwarg");
            numpy_l1_kwargs
                .set_item("axis", -1_i64)
                .expect("numpy l1 axis kwarg");

            group.bench_function(format!("fnp_norm_l1_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_norm
                        .call((&input,), Some(&fnp_l1_kwargs))
                        .expect("fnp norm l1 axis benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("numpy_norm_l1_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_norm
                        .call((&input,), Some(&numpy_l1_kwargs))
                        .expect("numpy norm l1 axis benchmark call");
                    black_box(result);
                });
            });

            let inf = f64::INFINITY;
            let fnp_inf_kwargs = PyDict::new(py);
            fnp_inf_kwargs.set_item("ord", inf).expect("fnp inf ord kwarg");
            fnp_inf_kwargs
                .set_item("axis", -1_i64)
                .expect("fnp inf axis kwarg");
            let numpy_inf_kwargs = PyDict::new(py);
            numpy_inf_kwargs
                .set_item("ord", inf)
                .expect("numpy inf ord kwarg");
            numpy_inf_kwargs
                .set_item("axis", -1_i64)
                .expect("numpy inf axis kwarg");

            group.bench_function(format!("fnp_norm_inf_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_norm
                        .call((&input,), Some(&fnp_inf_kwargs))
                        .expect("fnp norm inf axis benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("numpy_norm_inf_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_norm
                        .call((&input,), Some(&numpy_inf_kwargs))
                        .expect("numpy norm inf axis benchmark call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

// Vector norm along a NON-LAST axis for the order-independent ords (ord in {+inf,
// -inf, 0}). numpy runs a serial materialize-then-reduce; the native block-parallel /
// band-privatized column reduction (try_zerocopy_f64_vector_norm_axis non-last branch)
// is bit-exact for these order-free reductions. L2/L1 are NOT here (they delegate -
// numpy's strided summation order is not reproducible bit-for-bit in parallel).
fn bench_norm_nonlast_axis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_norm_nonlast_axis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_norm = module.getattr("norm").expect("fnp_python.norm");
        let numpy_norm = numpy
            .getattr("linalg")
            .expect("numpy.linalg")
            .getattr("norm")
            .expect("numpy.linalg.norm");

        let inf = f64::INFINITY;
        // (label, shape, axis)
        let cases: [(&str, Vec<i64>, i64); 4] = [
            ("4096x2048_ax0", vec![4096, 2048], 0),
            ("8192x1024_ax0", vec![8192, 1024], 0),
            ("256x256x64_ax1", vec![256, 256, 64], 1),
            ("256x256x64_ax0", vec![256, 256, 64], 0),
        ];
        for (label, shape, axis) in cases {
            let size: i64 = shape.iter().product();
            let shape_tuple = PyTuple::new(py, shape.iter().copied()).expect("shape tuple");
            let input = numpy
                .call_method1("linspace", (-4.0_f64, 6.0_f64, size))
                .expect("norm nonlast f64 input")
                .call_method1("reshape", (shape_tuple,))
                .expect("norm nonlast reshape");

            for (ord_label, ord_val) in [("inf", inf), ("ninf", -inf), ("zero", 0.0_f64)] {
                let fnp_kwargs = PyDict::new(py);
                fnp_kwargs.set_item("ord", ord_val).expect("fnp ord kwarg");
                fnp_kwargs.set_item("axis", axis).expect("fnp axis kwarg");
                let numpy_kwargs = PyDict::new(py);
                numpy_kwargs.set_item("ord", ord_val).expect("numpy ord kwarg");
                numpy_kwargs.set_item("axis", axis).expect("numpy axis kwarg");

                group.bench_function(format!("fnp_norm_{ord_label}_{label}"), |bench| {
                    bench.iter(|| {
                        let result = fnp_norm
                            .call((&input,), Some(&fnp_kwargs))
                            .expect("fnp norm nonlast call");
                        black_box(result);
                    });
                });
                group.bench_function(format!("numpy_norm_{ord_label}_{label}"), |bench| {
                    bench.iter(|| {
                        let result = numpy_norm
                            .call((&input,), Some(&numpy_kwargs))
                            .expect("numpy norm nonlast call");
                        black_box(result);
                    });
                });
            }
        }
    });

    group.finish();
}

fn bench_norm_frobenius_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_norm_frobenius_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_norm = module.getattr("norm").expect("fnp_python.norm");
        let numpy_norm = numpy
            .getattr("linalg")
            .expect("numpy.linalg")
            .getattr("norm")
            .expect("numpy.linalg.norm");

        for (label, b, m, n) in [
            ("4096x16x16", 4096_i64, 16_i64, 16_i64),
            ("2048x32x32", 2048_i64, 32_i64, 32_i64),
        ] {
            let size = b * m * n;
            let input = numpy
                .call_method1("linspace", (-4.0_f64, 6.0_f64, size))
                .expect("frobenius f64 input")
                .call_method1("reshape", ((b, m, n),))
                .expect("frobenius 3-D shape");

            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs
                .set_item("axis", (-2_i64, -1_i64))
                .expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs
                .set_item("axis", (-2_i64, -1_i64))
                .expect("numpy axis kwarg");

            group.bench_function(format!("fnp_norm_fro_f64_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_norm
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp frobenius benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("numpy_norm_fro_f64_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_norm
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy frobenius benchmark call");
                    black_box(result);
                });
            });

            // Induced matrix inf-norm (max abs row sum).
            let fnp_inf = PyDict::new(py);
            fnp_inf.set_item("ord", f64::INFINITY).expect("fnp inf ord");
            fnp_inf
                .set_item("axis", (-2_i64, -1_i64))
                .expect("fnp inf axis");
            let np_inf = PyDict::new(py);
            np_inf.set_item("ord", f64::INFINITY).expect("np inf ord");
            np_inf
                .set_item("axis", (-2_i64, -1_i64))
                .expect("np inf axis");
            group.bench_function(format!("fnp_norm_inf_f64_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_norm
                        .call((&input,), Some(&fnp_inf))
                        .expect("fnp matrix inf-norm call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_norm_inf_f64_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_norm
                        .call((&input,), Some(&np_inf))
                        .expect("numpy matrix inf-norm call");
                    black_box(result);
                });
            });

            // Induced matrix 1-norm (max abs col sum).
            let fnp_l1 = PyDict::new(py);
            fnp_l1.set_item("ord", 1_i64).expect("fnp l1 ord");
            fnp_l1
                .set_item("axis", (-2_i64, -1_i64))
                .expect("fnp l1 axis");
            let np_l1 = PyDict::new(py);
            np_l1.set_item("ord", 1_i64).expect("np l1 ord");
            np_l1
                .set_item("axis", (-2_i64, -1_i64))
                .expect("np l1 axis");
            group.bench_function(format!("fnp_norm_l1_f64_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_norm
                        .call((&input,), Some(&fnp_l1))
                        .expect("fnp matrix 1-norm call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_norm_l1_f64_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_norm
                        .call((&input,), Some(&np_l1))
                        .expect("numpy matrix 1-norm call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

fn bench_compress_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_compress_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_compress = module.getattr("compress").expect("fnp_python.compress");
        let numpy_compress = numpy.getattr("compress").expect("numpy.compress");
        let logical_or = numpy.getattr("logical_or").expect("numpy.logical_or");

        for size in [100_000_i64, 1_000_000_i64] {
            let index = numpy
                .call_method1("arange", (size,))
                .expect("compress index");
            let arr = index
                .call_method1("astype", ("float64",))
                .expect("compress f64 input")
                .call_method1("__sub__", (size as f64 / 2.0,))
                .expect("centered compress input");
            let every_181 = index
                .call_method1("__mod__", (181_i64,))
                .expect("compress mod 181")
                .call_method1("__eq__", (0_i64,))
                .expect("compress mod 181 mask");
            let residue = index
                .call_method1("__mul__", (41_i64,))
                .expect("compress mask multiply")
                .call_method1("__add__", (17_i64,))
                .expect("compress mask add")
                .call_method1("__mod__", (23_i64,))
                .expect("compress mod 23");
            let residue_mask = numpy
                .getattr("isin")
                .expect("numpy.isin")
                .call1((&residue, vec![0_i64, 3, 8, 13, 21]))
                .expect("compress residue mask");
            let condition = logical_or
                .call1((&every_181, &residue_mask))
                .expect("compress bool mask");

            group.bench_function(format!("fnp_compress_f64_axis_none_{size}"), |bench| {
                bench.iter(|| {
                    let result = fnp_compress
                        .call1((&condition, &arr))
                        .expect("fnp compress benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("numpy_compress_f64_axis_none_{size}"), |bench| {
                bench.iter(|| {
                    let result = numpy_compress
                        .call1((&condition, &arr))
                        .expect("numpy compress benchmark call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

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

        // 2-D per-axis roll (scalar shift + explicit axis). axis=0 (outer) was a
        // per-element Cell loop that numpy's block roll beat ~1.7x; copy_from_slice
        // memmove fixes it. 2000x2000 f64 (4M).
        let input2d = numpy
            .call_method1("arange", (2000_i64 * 2000_i64,))
            .expect("roll 2d index")
            .call_method1("astype", ("float64",))
            .expect("roll 2d f64")
            .call_method1("reshape", ((2000_i64, 2000_i64),))
            .expect("roll 2d reshape");
        for (label, ax) in [("axis0", 0_i64), ("axis1", 1_i64)] {
            group.bench_function(format!("fnp_roll_f64_2d_{label}_shift7"), |bench| {
                bench.iter(|| {
                    black_box(
                        fnp_roll
                            .call1((&input2d, 7_i64, ax))
                            .expect("fnp roll 2d call"),
                    );
                });
            });
            group.bench_function(format!("numpy_roll_f64_2d_{label}_shift7"), |bench| {
                bench.iter(|| {
                    black_box(
                        numpy_roll
                            .call1((&input2d, 7_i64, ax))
                            .expect("numpy roll 2d call"),
                    );
                });
            });
        }
    });

    group.finish();
}

fn bench_einsum_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_einsum_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let n = 4000_usize;
        let input = numpy
            .call_method1("arange", (n * n,))
            .expect("einsum raw input")
            .call_method1("astype", ("float64",))
            .expect("einsum f64 input")
            .call_method1("reshape", ((n, n),))
            .expect("einsum square input");
        let reduce_n = 1000_usize;
        let reduce_input = numpy
            .call_method1("arange", (reduce_n * reduce_n,))
            .expect("einsum reduction raw input")
            .call_method1("astype", ("float64",))
            .expect("einsum reduction f64 input")
            .call_method1("reshape", ((reduce_n, reduce_n),))
            .expect("einsum reduction square input");
        let make_matmul_pair = |n: usize| {
            let left = numpy
                .call_method1("arange", (n * n,))
                .expect("einsum matmul lhs raw input")
                .call_method1("astype", ("float64",))
                .expect("einsum matmul lhs f64 input")
                .call_method1("reshape", ((n, n),))
                .expect("einsum matmul lhs shape")
                .call_method1("__mul__", (0.0001_f64,))
                .expect("scale einsum matmul lhs");
            let right = numpy
                .call_method1("arange", (n * n,))
                .expect("einsum matmul rhs raw input")
                .call_method1("astype", ("float64",))
                .expect("einsum matmul rhs f64 input")
                .call_method1("reshape", ((n, n),))
                .expect("einsum matmul rhs shape")
                .call_method1("__mul__", (0.0002_f64,))
                .expect("scale einsum matmul rhs");
            (left, right)
        };
        let fnp_einsum = module.getattr("einsum").expect("fnp_python.einsum");
        let numpy_einsum = numpy.getattr("einsum").expect("numpy.einsum");

        group.bench_function("fnp_einsum_trace_f64_4000", |bench| {
            bench.iter(|| {
                let result = fnp_einsum
                    .call1(("ii", &input))
                    .expect("fnp einsum trace benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_einsum_trace_f64_4000", |bench| {
            bench.iter(|| {
                let result = numpy_einsum
                    .call1(("ii", &input))
                    .expect("numpy einsum trace benchmark call");
                black_box(result);
            });
        });

        group.bench_function("fnp_einsum_diag_f64_4000", |bench| {
            bench.iter(|| {
                let result = fnp_einsum
                    .call1(("ii->i", &input))
                    .expect("fnp einsum diag benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_einsum_diag_f64_4000", |bench| {
            bench.iter(|| {
                let result = numpy_einsum
                    .call1(("ii->i", &input))
                    .expect("numpy einsum diag benchmark call");
                black_box(result);
            });
        });

        group.bench_function("fnp_einsum_reduce_all_f64_1000", |bench| {
            bench.iter(|| {
                let result = fnp_einsum
                    .call1(("ij->", &reduce_input))
                    .expect("fnp einsum reduce-all benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_einsum_reduce_all_f64_1000", |bench| {
            bench.iter(|| {
                let result = numpy_einsum
                    .call1(("ij->", &reduce_input))
                    .expect("numpy einsum reduce-all benchmark call");
                black_box(result);
            });
        });

        group.bench_function("fnp_einsum_reduce_rows_f64_1000", |bench| {
            bench.iter(|| {
                let result = fnp_einsum
                    .call1(("ij->i", &reduce_input))
                    .expect("fnp einsum reduce-rows benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_einsum_reduce_rows_f64_1000", |bench| {
            bench.iter(|| {
                let result = numpy_einsum
                    .call1(("ij->i", &reduce_input))
                    .expect("numpy einsum reduce-rows benchmark call");
                black_box(result);
            });
        });

        group.bench_function("fnp_einsum_reduce_cols_f64_1000", |bench| {
            bench.iter(|| {
                let result = fnp_einsum
                    .call1(("ij->j", &reduce_input))
                    .expect("fnp einsum reduce-cols benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_einsum_reduce_cols_f64_1000", |bench| {
            bench.iter(|| {
                let result = numpy_einsum
                    .call1(("ij->j", &reduce_input))
                    .expect("numpy einsum reduce-cols benchmark call");
                black_box(result);
            });
        });

        for n in [100_usize, 200, 400] {
            let (left, right) = make_matmul_pair(n);

            group.bench_function(format!("fnp_einsum_matmul_f64_n{n}"), |bench| {
                bench.iter(|| {
                    let result = fnp_einsum
                        .call1(("ij,jk->ik", &left, &right))
                        .expect("fnp einsum matmul benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("numpy_einsum_matmul_f64_n{n}"), |bench| {
                bench.iter(|| {
                    let result = numpy_einsum
                        .call1(("ij,jk->ik", &left, &right))
                        .expect("numpy einsum matmul benchmark call");
                    black_box(result);
                });
            });
        }

        // No-contraction ELEMENTWISE einsum ("ij,ij->ij"): every operand shares the
        // output subscripts -> a plain elementwise product. numpy's generic einsum runs
        // it 3-4x slower than the multiply ufunc; fnp routes 2-operand elementwise to
        // np.multiply (bit-identical, wins at small/medium, parity at large). Guards
        // that fast path against regression.
        let (ew_l, ew_r) = make_matmul_pair(1024);
        group.bench_function("fnp_einsum_elementwise_f64_1024", |bench| {
            bench.iter(|| {
                black_box(
                    fnp_einsum
                        .call1(("ij,ij->ij", &ew_l, &ew_r))
                        .expect("fnp einsum elementwise call"),
                );
            });
        });
        group.bench_function("numpy_einsum_elementwise_f64_1024", |bench| {
            bench.iter(|| {
                black_box(
                    numpy_einsum
                        .call1(("ij,ij->ij", &ew_l, &ew_r))
                        .expect("numpy einsum elementwise call"),
                );
            });
        });

        // No-contraction BROADCAST einsum ("ij,j->ij" = a * b[None,:]): a matrix scaled by a
        // per-column vector. The generic native kernel ran this 10-15x slower than numpy;
        // fnp aligns the operands (reshape to output rank) and multiplies. Guards the
        // broadcast arm of the no-contraction fast path.
        let bc_mat = make_matmul_pair(1024).0;
        let bc_vec = numpy
            .call_method1("arange", (1024_usize,))
            .expect("bc vec raw")
            .call_method1("astype", ("float64",))
            .expect("bc vec f64")
            .call_method1("__mul__", (0.0001_f64,))
            .expect("scale bc vec");
        group.bench_function("fnp_einsum_broadcast_ij_j_f64_1024", |bench| {
            bench.iter(|| {
                black_box(
                    fnp_einsum
                        .call1(("ij,j->ij", &bc_mat, &bc_vec))
                        .expect("fnp einsum broadcast call"),
                );
            });
        });
        group.bench_function("numpy_einsum_broadcast_ij_j_f64_1024", |bench| {
            bench.iter(|| {
                black_box(
                    numpy_einsum
                        .call1(("ij,j->ij", &bc_mat, &bc_vec))
                        .expect("numpy einsum broadcast call"),
                );
            });
        });

        // All-shared NON-PREFIX contraction ("ijk,ijk->k" = reduce a Hadamard product over the
        // leading axes): the native generic kernel ran this 13-18x slower than numpy (strided
        // reduction); fnp now delegates the non-prefix forms to numpy.einsum. Guards that the
        // delegation holds at parity (and the prefix-kept "ijk,ijk->i" stays native/winning).
        let nc = numpy
            .call_method1("arange", (256_usize * 128 * 128,))
            .expect("nc raw")
            .call_method1("astype", ("float64",))
            .expect("nc f64")
            .call_method1("reshape", ((256_usize, 128, 128),))
            .expect("nc shape");
        group.bench_function("fnp_einsum_allshared_ijk_k_f64", |bench| {
            bench.iter(|| {
                black_box(
                    fnp_einsum
                        .call1(("ijk,ijk->k", &nc, &nc))
                        .expect("fnp einsum allshared call"),
                );
            });
        });
        group.bench_function("numpy_einsum_allshared_ijk_k_f64", |bench| {
            bench.iter(|| {
                black_box(
                    numpy_einsum
                        .call1(("ijk,ijk->k", &nc, &nc))
                        .expect("numpy einsum allshared call"),
                );
            });
        });

        // Op2-subset MULTI-axis contraction ("ijk,ij->k": op2's labels are a strict subset of
        // op1's and all contracted, >=2 axes summed): a strided multi-axis reduction the native
        // kernel ran 36x slower than numpy; fnp now delegates it. Guards that parity.
        let sc_mat = numpy
            .call_method1("arange", (256_usize * 128,))
            .expect("sc mat raw")
            .call_method1("astype", ("float64",))
            .expect("sc mat f64")
            .call_method1("reshape", ((256_usize, 128),))
            .expect("sc mat shape");
        group.bench_function("fnp_einsum_subcontract_ijk_ij_k_f64", |bench| {
            bench.iter(|| {
                black_box(
                    fnp_einsum
                        .call1(("ijk,ij->k", &nc, &sc_mat))
                        .expect("fnp einsum subcontract call"),
                );
            });
        });
        group.bench_function("numpy_einsum_subcontract_ijk_ij_k_f64", |bench| {
            bench.iter(|| {
                black_box(
                    numpy_einsum
                        .call1(("ijk,ij->k", &nc, &sc_mat))
                        .expect("numpy einsum subcontract call"),
                );
            });
        });

        // TRANSPOSED full contraction ("ijk,jik->": same label set, different order, scalar out):
        // a strided elementwise-then-sum over a transposed operand the native kernel ran 28x
        // slower than numpy; fnp now delegates it (same-order "ijk,ijk->" stays native/winning).
        let tc = numpy
            .call_method1("arange", (160_usize * 160 * 160,))
            .expect("tc raw")
            .call_method1("astype", ("float64",))
            .expect("tc f64")
            .call_method1("reshape", ((160_usize, 160, 160),))
            .expect("tc shape");
        let tc2 = tc
            .call_method1("transpose", ((1_usize, 0, 2),))
            .expect("tc transpose")
            .call_method0("copy")
            .expect("tc2 contig");
        group.bench_function("fnp_einsum_transpose_full_ijk_jik_f64", |bench| {
            bench.iter(|| {
                black_box(
                    fnp_einsum
                        .call1(("ijk,jik->", &tc, &tc2))
                        .expect("fnp einsum transpose-full call"),
                );
            });
        });
        group.bench_function("numpy_einsum_transpose_full_ijk_jik_f64", |bench| {
            bench.iter(|| {
                black_box(
                    numpy_einsum
                        .call1(("ijk,jik->", &tc, &tc2))
                        .expect("numpy einsum transpose-full call"),
                );
            });
        });

        // HUB contraction ("i,ij,j->": both vectors select axes of a central matrix and the
        // output is scalar): numpy's einsum fuses this into one pass over the hub, while the
        // native generic kernel was a 10-20x scalar loss. Guards the hub delegate detector.
        let hub_n = 2048_usize;
        let hub_left = numpy
            .call_method1("arange", (hub_n,))
            .expect("hub left raw")
            .call_method1("astype", ("float64",))
            .expect("hub left f64")
            .call_method1("__mul__", (0.0001_f64,))
            .expect("scale hub left");
        let hub_matrix = numpy
            .call_method1("arange", (hub_n * hub_n,))
            .expect("hub matrix raw")
            .call_method1("astype", ("float64",))
            .expect("hub matrix f64")
            .call_method1("reshape", ((hub_n, hub_n),))
            .expect("hub matrix shape")
            .call_method1("__mul__", (0.0000001_f64,))
            .expect("scale hub matrix");
        let hub_right = numpy
            .call_method1("arange", (hub_n,))
            .expect("hub right raw")
            .call_method1("astype", ("float64",))
            .expect("hub right f64")
            .call_method1("__mul__", (0.0002_f64,))
            .expect("scale hub right");
        group.bench_function("fnp_einsum_hub_i_ij_j_scalar_f64", |bench| {
            bench.iter(|| {
                black_box(
                    fnp_einsum
                        .call1(("i,ij,j->", &hub_left, &hub_matrix, &hub_right))
                        .expect("fnp einsum hub call"),
                );
            });
        });
        group.bench_function("numpy_einsum_hub_i_ij_j_scalar_f64", |bench| {
            bench.iter(|| {
                black_box(
                    numpy_einsum
                        .call1(("i,ij,j->", &hub_left, &hub_matrix, &hub_right))
                        .expect("numpy einsum hub call"),
                );
            });
        });
    });

    group.finish();
}

fn bench_linalg_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_linalg_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let numpy_linalg = numpy.getattr("linalg").expect("numpy.linalg");

        let batch = 8192_usize;
        let n = 4_usize;
        let matrix_count = batch * n * n;
        let raw = numpy
            .call_method1("arange", (matrix_count,))
            .expect("batched linalg raw values")
            .call_method1("astype", ("float64",))
            .expect("f64 matrix values")
            .call_method1("reshape", ((batch, n, n),))
            .expect("batched matrix shape");
        let scaled = raw
            .call_method1("__mul__", (0.001_f64,))
            .expect("scale matrix values");
        let eye = numpy
            .call_method1("eye", (n,))
            .expect("identity matrix")
            .call_method1("__mul__", (3.0_f64,))
            .expect("scaled identity");
        let matrices = scaled
            .call_method1("__add__", (&eye,))
            .expect("well-conditioned batched matrices");
        let rhs_vec = numpy
            .call_method1("arange", (n,))
            .expect("vector rhs raw values")
            .call_method1("astype", ("float64",))
            .expect("f64 rhs values")
            .call_method1("__mul__", (0.01_f64,))
            .expect("scaled vector rhs");
        let rhs_matrix = numpy
            .call_method1("arange", (batch * n * 2,))
            .expect("matrix rhs raw values")
            .call_method1("astype", ("float64",))
            .expect("f64 matrix rhs values")
            .call_method1("reshape", ((batch, n, 2_usize),))
            .expect("batched matrix rhs")
            .call_method1("__mul__", (0.01_f64,))
            .expect("scaled matrix rhs");
        let shared_matrix_raw = numpy
            .call_method1("arange", (n * n,))
            .expect("shared solve matrix raw values")
            .call_method1("astype", ("float64",))
            .expect("shared solve matrix f64 values")
            .call_method1("reshape", ((n, n),))
            .expect("shared solve matrix shape")
            .call_method1("__mul__", (0.001_f64,))
            .expect("scale shared solve matrix");
        let shared_matrix = shared_matrix_raw
            .call_method1("__add__", (&eye,))
            .expect("well-conditioned shared solve matrix");
        let repeated_matrices = numpy
            .call_method1("broadcast_to", (&shared_matrix, (batch, n, n)))
            .expect("broadcast shared solve matrix")
            .call_method0("copy")
            .expect("materialized repeated solve matrix stack");
        let make_spd_stack = |batch: usize, dim: usize| {
            let rng = numpy
                .getattr("random")
                .expect("numpy.random")
                .call_method1("default_rng", (0xC401_u64 + dim as u64,))
                .expect("cholesky rng");
            let raw = rng
                .call_method1("standard_normal", ((batch, dim, dim),))
                .expect("stacked normal matrix")
                .call_method1("astype", ("float64",))
                .expect("stacked f64 matrix");
            let transposed = raw
                .call_method1("swapaxes", (-1_i64, -2_i64))
                .expect("stacked transpose");
            let gram = numpy
                .getattr("matmul")
                .expect("numpy.matmul")
                .call1((&raw, &transposed))
                .expect("stacked gram matrix");
            let eye = numpy
                .call_method1("eye", (dim,))
                .expect("cholesky identity")
                .call_method1("__mul__", (dim as f64 + 1.0_f64,))
                .expect("scaled cholesky identity");
            gram.call_method1("__add__", (&eye,))
                .expect("stacked SPD matrix")
        };
        let inv_stack_128 = make_spd_stack(64, 128);
        let inv_stack_256 = make_spd_stack(16, 256);
        let make_spd_2d = |dim: usize| {
            let rng = numpy
                .getattr("random")
                .expect("numpy.random")
                .call_method1("default_rng", (0xD361_u64 + dim as u64,))
                .expect("2-D linalg rng");
            let raw = rng
                .call_method1("standard_normal", ((dim, dim),))
                .expect("2-D normal matrix")
                .call_method1("astype", ("float64",))
                .expect("2-D f64 matrix");
            let transposed = raw.getattr("T").expect("2-D transpose view");
            let gram = numpy
                .getattr("matmul")
                .expect("numpy.matmul")
                .call1((&raw, &transposed))
                .expect("2-D gram matrix");
            let eye = numpy
                .call_method1("eye", (dim,))
                .expect("2-D identity")
                .call_method1("__mul__", (dim as f64 + 1.0_f64,))
                .expect("scaled 2-D identity");
            gram.call_method1("__add__", (&eye,))
                .expect("2-D SPD matrix")
        };
        let make_dense_2d = |dim: usize| {
            let total = dim * dim;
            let raw = numpy
                .call_method1("arange", (total,))
                .expect("dense 2-D raw values")
                .call_method1("astype", ("float64",))
                .expect("dense 2-D f64 values")
                .call_method1("reshape", ((dim, dim),))
                .expect("dense 2-D shape")
                .call_method1("__mul__", (0.0001_f64,))
                .expect("scaled dense 2-D matrix");
            let eye = numpy
                .call_method1("eye", (dim,))
                .expect("dense 2-D identity")
                .call_method1("__mul__", (2.0_f64,))
                .expect("scaled dense 2-D identity");
            raw.call_method1("__add__", (&eye,))
                .expect("well-conditioned dense 2-D matrix")
        };
        let make_diagonal_2d = |dim: usize| {
            let values = numpy
                .call_method1("arange", (dim,))
                .expect("diagonal raw values")
                .call_method1("astype", ("float64",))
                .expect("diagonal f64 values")
                .call_method1("__add__", (0.25_f64,))
                .expect("shifted diagonal values")
                .call_method1("__mul__", (-1.0_f64,))
                .expect("descending diagonal values");
            numpy
                .getattr("diag")
                .expect("numpy.diag")
                .call1((values,))
                .expect("diagonal 2-D matrix")
        };

        let fnp_slogdet = module.getattr("slogdet").expect("fnp_python.slogdet");
        let numpy_slogdet = numpy_linalg
            .getattr("slogdet")
            .expect("numpy.linalg.slogdet");
        let fnp_inv = module.getattr("inv").expect("fnp_python.inv");
        let numpy_inv = numpy_linalg.getattr("inv").expect("numpy.linalg.inv");
        let fnp_solve = module.getattr("solve").expect("fnp_python.solve");
        let numpy_solve = numpy_linalg.getattr("solve").expect("numpy.linalg.solve");
        let fnp_eigvalsh = module.getattr("eigvalsh").expect("fnp_python.eigvalsh");
        let numpy_eigvalsh = numpy_linalg
            .getattr("eigvalsh")
            .expect("numpy.linalg.eigvalsh");
        let fnp_eigh = module.getattr("eigh").expect("fnp_python.eigh");
        let numpy_eigh = numpy_linalg.getattr("eigh").expect("numpy.linalg.eigh");
        let fnp_cholesky = module.getattr("cholesky").expect("fnp_python.cholesky");
        let numpy_cholesky = numpy_linalg
            .getattr("cholesky")
            .expect("numpy.linalg.cholesky");
        let fnp_matrix_power = module
            .getattr("matrix_power")
            .expect("fnp_python.matrix_power");
        let numpy_matrix_power = numpy_linalg
            .getattr("matrix_power")
            .expect("numpy.linalg.matrix_power");

        group.bench_function("fnp_slogdet_f64_batch8192_4x4", |bench| {
            bench.iter(|| {
                let result = fnp_slogdet
                    .call1((&matrices,))
                    .expect("fnp slogdet benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_slogdet_f64_batch8192_4x4", |bench| {
            bench.iter(|| {
                let result = numpy_slogdet
                    .call1((&matrices,))
                    .expect("numpy slogdet benchmark call");
                black_box(result);
            });
        });

        group.bench_function("fnp_inv_f64_batch8192_4x4", |bench| {
            bench.iter(|| {
                let result = fnp_inv.call1((&matrices,)).expect("fnp inv benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_inv_f64_batch8192_4x4", |bench| {
            bench.iter(|| {
                let result = numpy_inv
                    .call1((&matrices,))
                    .expect("numpy inv benchmark call");
                black_box(result);
            });
        });

        for (label, input) in [
            ("batch64_128x128", inv_stack_128),
            ("batch16_256x256", inv_stack_256),
        ] {
            group.bench_function(format!("fnp_inv_f64_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_inv.call1((&input,)).expect("fnp inv benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("numpy_inv_f64_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_inv
                        .call1((&input,))
                        .expect("numpy inv benchmark call");
                    black_box(result);
                });
            });
        }

        group.bench_function("fnp_solve_f64_batch8192_4x4_vec", |bench| {
            bench.iter(|| {
                let result = fnp_solve
                    .call1((&matrices, &rhs_vec))
                    .expect("fnp solve benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_solve_f64_batch8192_4x4_vec", |bench| {
            bench.iter(|| {
                let result = numpy_solve
                    .call1((&matrices, &rhs_vec))
                    .expect("numpy solve benchmark call");
                black_box(result);
            });
        });

        group.bench_function("fnp_solve_repeated_a_f64_batch8192_4x4_vec", |bench| {
            bench.iter(|| {
                let result = fnp_solve
                    .call1((&repeated_matrices, &rhs_vec))
                    .expect("fnp solve repeated-A vector benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_solve_repeated_a_f64_batch8192_4x4_vec", |bench| {
            bench.iter(|| {
                let result = numpy_solve
                    .call1((&repeated_matrices, &rhs_vec))
                    .expect("numpy solve repeated-A vector benchmark call");
                black_box(result);
            });
        });

        group.bench_function("fnp_solve_repeated_a_f64_batch8192_4x4_mat2", |bench| {
            bench.iter(|| {
                let result = fnp_solve
                    .call1((&repeated_matrices, &rhs_matrix))
                    .expect("fnp solve repeated-A matrix benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_solve_repeated_a_f64_batch8192_4x4_mat2", |bench| {
            bench.iter(|| {
                let result = numpy_solve
                    .call1((&repeated_matrices, &rhs_matrix))
                    .expect("numpy solve repeated-A matrix benchmark call");
                black_box(result);
            });
        });

        group.bench_function("fnp_solve_f64_batch8192_4x4_mat2", |bench| {
            bench.iter(|| {
                let result = fnp_solve
                    .call1((&matrices, &rhs_matrix))
                    .expect("fnp solve benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_solve_f64_batch8192_4x4_mat2", |bench| {
            bench.iter(|| {
                let result = numpy_solve
                    .call1((&matrices, &rhs_matrix))
                    .expect("numpy solve benchmark call");
                black_box(result);
            });
        });

        for (label, input) in [("n200", make_spd_2d(200)), ("n800", make_spd_2d(800))] {
            group.bench_function(format!("fnp_eigvalsh_delegate_f64_2d_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_eigvalsh
                        .call1((&input,))
                        .expect("fnp eigvalsh delegate benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("numpy_eigvalsh_delegate_f64_2d_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_eigvalsh
                        .call1((&input,))
                        .expect("numpy eigvalsh benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("fnp_eigh_delegate_f64_2d_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_eigh
                        .call1((&input,))
                        .expect("fnp eigh delegate benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("numpy_eigh_delegate_f64_2d_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_eigh
                        .call1((&input,))
                        .expect("numpy eigh benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("fnp_cholesky_delegate_f64_2d_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_cholesky
                        .call1((&input,))
                        .expect("fnp cholesky delegate benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("numpy_cholesky_delegate_f64_2d_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_cholesky
                        .call1((&input,))
                        .expect("numpy cholesky benchmark call");
                    black_box(result);
                });
            });
        }

        for (label, input) in [
            ("n200", make_diagonal_2d(200)),
            ("n800", make_diagonal_2d(800)),
        ] {
            group.bench_function(format!("fnp_eigvalsh_diagonal_f64_2d_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_eigvalsh
                        .call1((&input,))
                        .expect("fnp eigvalsh diagonal benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("numpy_eigvalsh_diagonal_f64_2d_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_eigvalsh
                        .call1((&input,))
                        .expect("numpy eigvalsh diagonal benchmark call");
                    black_box(result);
                });
            });
        }

        let matrix_power_800 = make_dense_2d(800);
        for (label, power) in [("n0", 0_i64), ("n1", 1_i64)] {
            group.bench_function(
                format!("fnp_matrix_power_delegate_f64_2d_800_{label}"),
                |bench| {
                    bench.iter(|| {
                        let result = fnp_matrix_power
                            .call1((&matrix_power_800, power))
                            .expect("fnp matrix_power delegate benchmark call");
                        black_box(result);
                    });
                },
            );

            group.bench_function(
                format!("numpy_matrix_power_delegate_f64_2d_800_{label}"),
                |bench| {
                    bench.iter(|| {
                        let result = numpy_matrix_power
                            .call1((&matrix_power_800, power))
                            .expect("numpy matrix_power benchmark call");
                        black_box(result);
                    });
                },
            );
        }

        for (label, input) in [
            ("batch10000_4x4", make_spd_stack(10_000, 4)),
            ("batch4000_8x8", make_spd_stack(4_000, 8)),
            ("batch2000_16x16", make_spd_stack(2_000, 16)),
            ("batch1000_32x32", make_spd_stack(1_000, 32)),
            ("batch500_64x64", make_spd_stack(500, 64)),
        ] {
            group.bench_function(format!("fnp_cholesky_f64_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_cholesky
                        .call1((&input,))
                        .expect("fnp cholesky benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("numpy_cholesky_f64_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_cholesky
                        .call1((&input,))
                        .expect("numpy cholesky benchmark call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

fn bench_unary_parallel_boundary(c: &mut Criterion) {
    // f32 / i64 / i32 elementwise unary maps (square, abs) at 8M — above the 1<<21
    // parallel gate. The serial Cell loop lost to numpy's vectorized ufunc (square/f32
    // ~2x, square/i64 ~1.5x); the parallel raw-slice map should win. Bit-exact.
    let mut group = c.benchmark_group("python_unary_parallel_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");

        let base = numpy
            .call_method1("arange", (8_000_000_i64,))
            .expect("8M base")
            .call_method1("__sub__", (4_000_000_i64,))
            .expect("centered base");
        let f32_in = base.call_method1("astype", ("float32",)).expect("f32 input");
        let i64_in = base.call_method1("astype", ("int64",)).expect("i64 input");
        let i32_in = base.call_method1("astype", ("int32",)).expect("i32 input");
        let u64_in = base.call_method1("astype", ("uint64",)).expect("u64 input");
        let u32_in = base.call_method1("astype", ("uint32",)).expect("u32 input");
        let u16_in = base.call_method1("astype", ("uint16",)).expect("u16 input");
        let u8_in = base.call_method1("astype", ("uint8",)).expect("u8 input");

        let fnp_square = module.getattr("square").expect("fnp square");
        let fnp_abs = module.getattr("abs").expect("fnp abs");
        let numpy_square = numpy.getattr("square").expect("numpy square");
        let numpy_abs = numpy.getattr("abs").expect("numpy abs");

        macro_rules! pair {
            ($label:literal, $fnpf:expr, $npf:expr, $arg:expr) => {{
                group.bench_function(concat!("fnp_", $label), |b| {
                    b.iter(|| black_box($fnpf.call1(($arg,)).expect("fnp call")));
                });
                group.bench_function(concat!("numpy_", $label), |b| {
                    b.iter(|| black_box($npf.call1(($arg,)).expect("numpy call")));
                });
            }};
        }
        pair!("square_f32_8m", fnp_square, numpy_square, &f32_in);
        pair!("abs_f32_8m", fnp_abs, numpy_abs, &f32_in);
        pair!("square_i64_8m", fnp_square, numpy_square, &i64_in);
        pair!("square_i32_8m", fnp_square, numpy_square, &i32_in);
        pair!("square_u64_8m", fnp_square, numpy_square, &u64_in);
        pair!("square_u32_8m", fnp_square, numpy_square, &u32_in);
        pair!("square_u16_8m", fnp_square, numpy_square, &u16_in);
        pair!("square_u8_8m", fnp_square, numpy_square, &u8_in);
    });

    group.finish();
}

fn bench_clip_boundary(c: &mut Criterion) {
    // f64 np.clip at 8M — above the 1<<21 parallel gate. Serial Cell clamp was at numpy
    // parity (memory-bound single-thread); the parallel raw-slice clamp aggregates
    // bandwidth and should win ~2x. Bit-exact (same if-form, NaN-propagating).
    let mut group = c.benchmark_group("python_clip_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let input = numpy
            .call_method1("arange", (8_000_000_i64,))
            .expect("8M base")
            .call_method1("__sub__", (4_000_000_i64,))
            .expect("centered")
            .call_method1("astype", ("float64",))
            .expect("f64 input");
        let fnp_clip = module.getattr("clip").expect("fnp clip");
        let numpy_clip = numpy.getattr("clip").expect("numpy clip");
        group.bench_function("fnp_clip_f64_8m", |b| {
            b.iter(|| black_box(fnp_clip.call1((&input, -1000.0_f64, 1000.0_f64)).expect("fnp clip")));
        });
        group.bench_function("numpy_clip_f64_8m", |b| {
            b.iter(|| black_box(numpy_clip.call1((&input, -1000.0_f64, 1000.0_f64)).expect("numpy clip")));
        });

        let input_f32 = input
            .call_method1("astype", ("float32",))
            .expect("f32 input");
        group.bench_function("fnp_clip_f32_8m", |b| {
            b.iter(|| {
                black_box(
                    fnp_clip
                        .call1((&input_f32, -1000.0_f32, 1000.0_f32))
                        .expect("fnp f32 clip"),
                )
            });
        });
        group.bench_function("numpy_clip_f32_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_clip
                        .call1((&input_f32, -1000.0_f32, 1000.0_f32))
                        .expect("numpy f32 clip"),
                )
            });
        });

        let ibase = numpy
            .call_method1("arange", (8_000_000_i64,))
            .expect("8M ibase")
            .call_method1("__sub__", (4_000_000_i64,))
            .expect("centered ibase");
        let i64_in = ibase.call_method1("astype", ("int64",)).expect("i64");
        let i32_in = ibase.call_method1("astype", ("int32",)).expect("i32");
        macro_rules! iclip {
            ($label:literal, $arr:expr, $lo:expr, $hi:expr) => {{
                group.bench_function(concat!("fnp_", $label), |b| {
                    b.iter(|| black_box(fnp_clip.call1(($arr, $lo, $hi)).expect("fnp iclip")));
                });
                group.bench_function(concat!("numpy_", $label), |b| {
                    b.iter(|| black_box(numpy_clip.call1(($arr, $lo, $hi)).expect("numpy iclip")));
                });
            }};
        }
        iclip!("clip_i64_8m", &i64_in, -1000_i64, 1000_i64);
        iclip!("clip_i32_8m", &i32_in, -1000_i32, 1000_i32);
    });

    group.finish();
}

fn bench_where_boundary(c: &mut Criterion) {
    // f64 np.where(mask, a, b) arr/arr at 8M — above the 1<<21 gate. Serial Cell select
    // was numpy-parity; parallel raw-slice select aggregates bandwidth and should win.
    let mut group = c.benchmark_group("python_where_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let base = numpy
            .call_method1("arange", (8_000_000_i64,))
            .expect("8M base")
            .call_method1("astype", ("float64",))
            .expect("f64");
        let a = base.call_method1("__mul__", (2.0_f64,)).expect("a");
        let b = base.call_method1("__add__", (1.0_f64,)).expect("b");
        let mask = base
            .call_method1("__mod__", (2.0_f64,))
            .expect("mod")
            .call_method1("__gt__", (0.5_f64,))
            .expect("mask bool");
        let fnp_where = module.getattr("where").expect("fnp where");
        let numpy_where = numpy.getattr("where").expect("numpy where");
        group.bench_function("fnp_where_f64_8m", |bn| {
            bn.iter(|| black_box(fnp_where.call1((&mask, &a, &b)).expect("fnp where")));
        });
        group.bench_function("numpy_where_f64_8m", |bn| {
            bn.iter(|| black_box(numpy_where.call1((&mask, &a, &b)).expect("numpy where")));
        });

        let base_f32 = base.call_method1("astype", ("float32",)).expect("f32");
        let a32 = base_f32.call_method1("__mul__", (2.0_f32,)).expect("a32");
        let b32 = base_f32.call_method1("__add__", (1.0_f32,)).expect("b32");
        group.bench_function("fnp_where_f32_8m", |bn| {
            bn.iter(|| black_box(fnp_where.call1((&mask, &a32, &b32)).expect("fnp where f32")));
        });
        group.bench_function("numpy_where_f32_8m", |bn| {
            bn.iter(|| {
                black_box(
                    numpy_where
                        .call1((&mask, &a32, &b32))
                        .expect("numpy where f32"),
                )
            });
        });

        let ibase = numpy
            .call_method1("arange", (8_000_000_i64,))
            .expect("8M ibase");
        let ia = ibase.call_method1("__mul__", (2_i64,)).expect("ia");
        let ib = ibase.call_method1("__add__", (1_i64,)).expect("ib");
        let imask = ibase
            .call_method1("__mod__", (2_i64,))
            .expect("imod")
            .call_method1("__eq__", (1_i64,))
            .expect("imask bool");
        group.bench_function("fnp_where_i64_8m", |bn| {
            bn.iter(|| black_box(fnp_where.call1((&imask, &ia, &ib)).expect("fnp where i64")));
        });
        group.bench_function("numpy_where_i64_8m", |bn| {
            bn.iter(|| {
                black_box(
                    numpy_where
                        .call1((&imask, &ia, &ib))
                        .expect("numpy where i64"),
                )
            });
        });
    });

    group.finish();
}

fn bench_around_boundary(c: &mut Criterion) {
    // f64 np.around(a, 3) at 8M — serial Cell map (mul+round+divide) vs parallel raw-slice.
    let mut group = c.benchmark_group("python_around_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let input = numpy
            .call_method1("arange", (8_000_000_i64,))
            .expect("8M")
            .call_method1("astype", ("float64",))
            .expect("f64")
            .call_method1("__mul__", (0.12345_f64,))
            .expect("scaled");
        let fnp_around = module.getattr("around").expect("fnp around");
        let numpy_around = numpy.getattr("around").expect("numpy around");
        group.bench_function("fnp_around_f64_8m", |b| {
            b.iter(|| black_box(fnp_around.call1((&input, 3_i64)).expect("fnp around")));
        });
        group.bench_function("numpy_around_f64_8m", |b| {
            b.iter(|| black_box(numpy_around.call1((&input, 3_i64)).expect("numpy around")));
        });

        // complex128 sibling — numpy rounds complex via a slow multi-pass; fnp views the
        // 4M complex as 8M f64 components and reuses the parallel f64 around path.
        let input_c = input
            .call_method1("view", ("complex128",))
            .expect("c128 view");
        group.bench_function("fnp_around_c128_4m", |b| {
            b.iter(|| black_box(fnp_around.call1((&input_c, 3_i64)).expect("fnp around c128")));
        });
        group.bench_function("numpy_around_c128_4m", |b| {
            b.iter(|| black_box(numpy_around.call1((&input_c, 3_i64)).expect("numpy around c128")));
        });

        // f32 sibling — compute-heavy (round-ties-even + mul/div) so wins at 4-byte.
        let input32 = input.call_method1("astype", ("float32",)).expect("f32 input");
        group.bench_function("fnp_around_f32_8m", |b| {
            b.iter(|| black_box(fnp_around.call1((&input32, 3_i64)).expect("fnp around f32")));
        });
        group.bench_function("numpy_around_f32_8m", |b| {
            b.iter(|| black_box(numpy_around.call1((&input32, 3_i64)).expect("numpy around f32")));
        });
    });

    group.finish();
}

fn bench_cross_boundary(c: &mut Criterion) {
    // np.cross on stacked 3-vectors at 4M lanes (12M floats/operand) — above the 1<<21
    // parallel gate. The serial Cell loop reached numpy parity (single-thread, memory-bound);
    // the per-lane parallel map aggregates bandwidth + ALU (6 mul + 3 sub/lane) and should
    // win. Each output 3-vec depends only on its matching input 3-vecs => bit-exact.
    let mut group = c.benchmark_group("python_cross_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let n: i64 = 4_000_000;
        let total = n * 3;
        let a = numpy
            .call_method1("arange", (total,))
            .expect("a base")
            .call_method1("astype", ("float64",))
            .expect("a f64")
            .call_method1("reshape", ((n, 3_i64),))
            .expect("a reshape");
        let b = numpy
            .call_method1("arange", (total,))
            .expect("b base")
            .call_method1("__mul__", (2_i64,))
            .expect("b scaled")
            .call_method1("astype", ("float64",))
            .expect("b f64")
            .call_method1("reshape", ((n, 3_i64),))
            .expect("b reshape");
        let fnp_cross = module.getattr("cross").expect("fnp cross");
        let numpy_cross = numpy.getattr("cross").expect("numpy cross");
        group.bench_function("fnp_cross_f64_4m", |bch| {
            bch.iter(|| black_box(fnp_cross.call1((&a, &b)).expect("fnp cross")));
        });
        group.bench_function("numpy_cross_f64_4m", |bch| {
            bch.iter(|| black_box(numpy_cross.call1((&a, &b)).expect("numpy cross")));
        });

        let a32 = a.call_method1("astype", ("float32",)).expect("a f32");
        let b32 = b.call_method1("astype", ("float32",)).expect("b f32");
        group.bench_function("fnp_cross_f32_4m", |bch| {
            bch.iter(|| black_box(fnp_cross.call1((&a32, &b32)).expect("fnp cross f32")));
        });
        group.bench_function("numpy_cross_f32_4m", |bch| {
            bch.iter(|| black_box(numpy_cross.call1((&a32, &b32)).expect("numpy cross f32")));
        });
    });
    group.finish();
}

fn bench_nan_to_num_boundary(c: &mut Criterion) {
    // np.nan_to_num at 8M — numpy runs several single-threaded masked passes
    // (isnan/isposinf/isneginf + copyto); fnp does one fused parallel per-element pass.
    // ~1/8 of the elements are nan/inf so the branch is exercised. Bit-exact.
    let mut group = c.benchmark_group("python_nan_to_num_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        // base = standard_normal(8M); sprinkle nan/+inf/-inf every few elements.
        let setup = "import numpy as np\n\
rng = np.random.default_rng(0)\n\
a = rng.standard_normal(8_000_000)\n\
a[::8] = np.nan\n\
a[1::13] = np.inf\n\
a[2::17] = -np.inf\n\
a32 = a.astype(np.float32)\n\
ac = a.view(np.complex128)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            None,
        )
        .expect("nan_to_num setup");
        let a = ns.get_item("a").expect("a");
        let a32 = ns.get_item("a32").expect("a32");
        let ac = ns.get_item("ac").expect("ac");
        let fnp_n2n = module.getattr("nan_to_num").expect("fnp nan_to_num");
        let numpy_n2n = numpy.getattr("nan_to_num").expect("numpy nan_to_num");
        group.bench_function("fnp_nan_to_num_f64_8m", |bch| {
            bch.iter(|| black_box(fnp_n2n.call1((&a,)).expect("fnp nan_to_num")));
        });
        group.bench_function("numpy_nan_to_num_f64_8m", |bch| {
            bch.iter(|| black_box(numpy_n2n.call1((&a,)).expect("numpy nan_to_num")));
        });
        group.bench_function("fnp_nan_to_num_c128_4m", |bch| {
            bch.iter(|| black_box(fnp_n2n.call1((&ac,)).expect("fnp nan_to_num c128")));
        });
        group.bench_function("numpy_nan_to_num_c128_4m", |bch| {
            bch.iter(|| black_box(numpy_n2n.call1((&ac,)).expect("numpy nan_to_num c128")));
        });
        group.bench_function("fnp_nan_to_num_f32_8m", |bch| {
            bch.iter(|| black_box(fnp_n2n.call1((&a32,)).expect("fnp nan_to_num f32")));
        });
        group.bench_function("numpy_nan_to_num_f32_8m", |bch| {
            bch.iter(|| black_box(numpy_n2n.call1((&a32,)).expect("numpy nan_to_num f32")));
        });
    });
    group.finish();
}

fn bench_kron_boundary(c: &mut Criterion) {
    // f64 np.kron of two 2-D arrays with a ~4M-element output (above the 1<<21 gate).
    // numpy.kron is single-threaded (one broadcast-multiply); the row-parallel fill should
    // aggregate bandwidth + the per-element multiply. out[(i*bm+k),(j*bn+l)] = a[i,j]*b[k,l].
    let mut group = c.benchmark_group("python_kron_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        // A = (50,50), B = (40,40) -> output (2000,2000) = 4M elements.
        let mk = |rows: i64, cols: i64, scale: f64| {
            numpy
                .call_method1("arange", (rows * cols,))
                .expect("arange")
                .call_method1("astype", ("float64",))
                .expect("f64")
                .call_method1("__mul__", (scale,))
                .expect("scaled")
                .call_method1("reshape", ((rows, cols),))
                .expect("reshape")
        };
        let a = mk(50, 50, 0.5_f64);
        let b = mk(40, 40, 0.25_f64);
        let fnp_kron = module.getattr("kron").expect("fnp kron");
        let numpy_kron = numpy.getattr("kron").expect("numpy kron");
        group.bench_function("fnp_kron_f64_4m", |bn| {
            bn.iter(|| black_box(fnp_kron.call1((&a, &b)).expect("fnp kron")));
        });
        group.bench_function("numpy_kron_f64_4m", |bn| {
            bn.iter(|| black_box(numpy_kron.call1((&a, &b)).expect("numpy kron")));
        });

        // f32 + i64 2-D (kron2d_typed path).
        let a32 = a.call_method1("astype", ("float32",)).expect("a f32");
        let b32 = b.call_method1("astype", ("float32",)).expect("b f32");
        group.bench_function("fnp_kron_f32_4m", |bn| {
            bn.iter(|| black_box(fnp_kron.call1((&a32, &b32)).expect("fnp kron f32")));
        });
        group.bench_function("numpy_kron_f32_4m", |bn| {
            bn.iter(|| black_box(numpy_kron.call1((&a32, &b32)).expect("numpy kron f32")));
        });
        let ai = a.call_method1("astype", ("int64",)).expect("a i64");
        let bi = b.call_method1("astype", ("int64",)).expect("b i64");
        group.bench_function("fnp_kron_i64_4m", |bn| {
            bn.iter(|| black_box(fnp_kron.call1((&ai, &bi)).expect("fnp kron i64")));
        });
        group.bench_function("numpy_kron_i64_4m", |bn| {
            bn.iter(|| black_box(numpy_kron.call1((&ai, &bi)).expect("numpy kron i64")));
        });

        // 1-D kron (kron1d path): two 2000-vectors -> 4M output.
        let a1 = numpy
            .call_method1("arange", (2000_i64,))
            .expect("a1")
            .call_method1("astype", ("float64",))
            .expect("a1 f64");
        let b1 = numpy
            .call_method1("arange", (2000_i64,))
            .expect("b1")
            .call_method1("astype", ("float64",))
            .expect("b1 f64")
            .call_method1("__add__", (1.0_f64,))
            .expect("b1 shifted");
        group.bench_function("fnp_kron_1d_f64_4m", |bn| {
            bn.iter(|| black_box(fnp_kron.call1((&a1, &b1)).expect("fnp kron 1d")));
        });
        group.bench_function("numpy_kron_1d_f64_4m", |bn| {
            bn.iter(|| black_box(numpy_kron.call1((&a1, &b1)).expect("numpy kron 1d")));
        });
    });
    group.finish();
}

fn bench_tile_boundary(c: &mut Criterion) {
    // np.tile of a 1-D array (scalar reps) -> ~4M output. numpy.tile is a single-threaded
    // python helper (reshape + C repeat); fnp does a parallel block memcpy. Bit-exact.
    let mut group = c.benchmark_group("python_tile_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        // base 1-D of 4096 elements tiled 1024x -> ~4.19M output.
        let base = numpy
            .call_method1("arange", (4096_i64,))
            .expect("base")
            .call_method1("astype", ("float64",))
            .expect("f64");
        let fnp_tile = module.getattr("tile").expect("fnp tile");
        let numpy_tile = numpy.getattr("tile").expect("numpy tile");
        group.bench_function("fnp_tile_f64_4m", |bn| {
            bn.iter(|| black_box(fnp_tile.call1((&base, 1024_i64)).expect("fnp tile f64")));
        });
        group.bench_function("numpy_tile_f64_4m", |bn| {
            bn.iter(|| black_box(numpy_tile.call1((&base, 1024_i64)).expect("numpy tile f64")));
        });
    });
    group.finish();
}

fn bench_digitize_boundary(c: &mut Criterion) {
    // np.digitize(4M f64, 50 bins) — serial per-element binary search vs parallel raw-slice.
    let mut group = c.benchmark_group("python_digitize_boundary");
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
x = rng.standard_normal(4_000_000)\n\
bins = np.linspace(-4.0, 4.0, 50)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("digitize setup");
        let x = ns.get_item("x").expect("x");
        let bins = ns.get_item("bins").expect("bins");
        let fnp_dig = module.getattr("digitize").expect("fnp digitize");
        let numpy_dig = numpy.getattr("digitize").expect("numpy digitize");
        group.bench_function("fnp_digitize_f64_4m", |b| {
            b.iter(|| black_box(fnp_dig.call1((&x, &bins)).expect("fnp digitize")));
        });
        group.bench_function("numpy_digitize_f64_4m", |b| {
            b.iter(|| black_box(numpy_dig.call1((&x, &bins)).expect("numpy digitize")));
        });
    });

    group.finish();
}

// np.bincount(int64) — a tight serial scatter `ans[x[i]]++` in NumPy. The native zero-copy
// path drops the per-element bounds check (the max-scan proves every index is in range) so the
// serial tally matches/beats NumPy across the common range; the privatized parallel tally only
// engages for huge inputs (>=67M) where aggregate bandwidth beats the contention overhead on a
// loaded many-core box. Bins span N=2M (mid) and N=64M (huge/parallel).
fn bench_bincount_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_bincount_boundary");
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
x_mid = rng.integers(0, 256, 2_000_000)\n\
x_big = rng.integers(0, 512, 64_000_000)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("bincount setup");
        let x_mid = ns.get_item("x_mid").expect("x_mid");
        let x_big = ns.get_item("x_big").expect("x_big");
        let fnp_bc = module.getattr("bincount").expect("fnp bincount");
        let numpy_bc = numpy.getattr("bincount").expect("numpy bincount");
        for (label, x) in [("mid_2m_k256", &x_mid), ("big_64m_k512", &x_big)] {
            group.bench_function(format!("fnp_bincount_i64_{label}"), |b| {
                b.iter(|| black_box(fnp_bc.call1((x,)).expect("fnp bincount")));
            });
            group.bench_function(format!("numpy_bincount_i64_{label}"), |b| {
                b.iter(|| black_box(numpy_bc.call1((x,)).expect("numpy bincount")));
            });
        }
    });

    group.finish();
}

fn bench_searchsorted_boundary(c: &mut Criterion) {
    // np.searchsorted(1M sorted haystack, 4M queries) — serial per-query binary search vs parallel.
    let mut group = c.benchmark_group("python_searchsorted_boundary");
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
a = np.sort(rng.standard_normal(1_000_000))\n\
v = rng.standard_normal(4_000_000)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("searchsorted setup");
        let a = ns.get_item("a").expect("a");
        let v = ns.get_item("v").expect("v");
        let fnp_ss = module.getattr("searchsorted").expect("fnp searchsorted");
        let numpy_ss = numpy.getattr("searchsorted").expect("numpy searchsorted");
        group.bench_function("fnp_searchsorted_f64_4m", |b| {
            b.iter(|| black_box(fnp_ss.call1((&a, &v)).expect("fnp searchsorted")));
        });
        group.bench_function("numpy_searchsorted_f64_4m", |b| {
            b.iter(|| black_box(numpy_ss.call1((&a, &v)).expect("numpy searchsorted")));
        });
    });

    group.finish();
}

fn bench_take_boundary(c: &mut Criterion) {
    // np.take(16M f64 source, 8M random indices) — serial gather vs parallel raw-slice gather.
    let mut group = c.benchmark_group("python_take_boundary");
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
a = rng.standard_normal(16_000_000)\n\
idx = rng.integers(0, 16_000_000, 8_000_000).astype(np.int64)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("take setup");
        let a = ns.get_item("a").expect("a");
        let idx = ns.get_item("idx").expect("idx");
        let fnp_take = module.getattr("take").expect("fnp take");
        let numpy_take = numpy.getattr("take").expect("numpy take");
        group.bench_function("fnp_take_f64_8m", |b| {
            b.iter(|| black_box(fnp_take.call1((&a, &idx)).expect("fnp take")));
        });
        group.bench_function("numpy_take_f64_8m", |b| {
            b.iter(|| black_box(numpy_take.call1((&a, &idx)).expect("numpy take")));
        });
    });

    group.finish();
}

fn bench_take_along_axis_boundary(c: &mut Criterion) {
    // np.take_along_axis(4096x4096 f64, 4096x2048 idx, axis=1) — serial gather vs parallel.
    let mut group = c.benchmark_group("python_take_along_axis_boundary");
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
a = rng.standard_normal((4096, 4096))\n\
idx = rng.integers(0, 4096, (4096, 2048)).astype(np.int64)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("take_along_axis setup");
        let a = ns.get_item("a").expect("a");
        let idx = ns.get_item("idx").expect("idx");
        let fnp_t = module.getattr("take_along_axis").expect("fnp take_along_axis");
        let numpy_t = numpy.getattr("take_along_axis").expect("numpy take_along_axis");
        let axis = 1_i64;
        group.bench_function("fnp_take_along_axis_f64_8m", |b| {
            b.iter(|| black_box(fnp_t.call1((&a, &idx, axis)).expect("fnp take_along_axis")));
        });
        group.bench_function("numpy_take_along_axis_f64_8m", |b| {
            b.iter(|| black_box(numpy_t.call1((&a, &idx, axis)).expect("numpy take_along_axis")));
        });
    });

    group.finish();
}

fn bench_take_axis_boundary(c: &mut Criterion) {
    // np.take(4096x4096 f64, 2048 idx, axis=1) — serial per-axis gather vs parallel raw-slice.
    let mut group = c.benchmark_group("python_take_axis_boundary");
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
a = rng.standard_normal((4096, 4096))\n\
idx = rng.integers(0, 4096, 2048).astype(np.int64)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("take_axis setup");
        let a = ns.get_item("a").expect("a");
        let idx = ns.get_item("idx").expect("idx");
        let fnp_take = module.getattr("take").expect("fnp take");
        let numpy_take = numpy.getattr("take").expect("numpy take");
        let kwargs = PyDict::new(py);
        kwargs.set_item("axis", 1_i64).expect("axis");
        group.bench_function("fnp_take_axis1_f64_8m", |b| {
            b.iter(|| black_box(fnp_take.call((&a, &idx), Some(&kwargs)).expect("fnp take axis")));
        });
        group.bench_function("numpy_take_axis1_f64_8m", |b| {
            b.iter(|| black_box(numpy_take.call((&a, &idx), Some(&kwargs)).expect("numpy take axis")));
        });
    });

    group.finish();
}

fn bench_parallel_binary_boundary(c: &mut Criterion) {
    // float_power / remainder / nextafter / power / fmod / heaviside / maximum / minimum /
    // copysign at 8M — routed through the zero-copy parallel binary kernel (numpy runs these
    // single-threaded).
    let mut group = c.benchmark_group("python_parallel_binary_boundary");
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
a = np.abs(rng.standard_normal(8_000_000)) + 0.1\n\
b = rng.standard_normal(8_000_000) * 5.0\n\
bnz = np.where(b == 0.0, 1.0, b)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("parallel binary setup");
        let a = ns.get_item("a").expect("a");
        let b = ns.get_item("b").expect("b");
        let bnz = ns.get_item("bnz").expect("bnz");
        for (op, x, y) in [
            ("float_power", &a, &b),
            ("nextafter", &a, &b),
            ("remainder", &a, &bnz),
            ("power", &a, &b),
            ("fmod", &a, &bnz),
            ("heaviside", &b, &a),
            ("maximum", &a, &b),
            ("minimum", &a, &b),
            ("copysign", &a, &b),
            ("divide", &a, &b),
        ] {
            let fnp_fn = module.getattr(op).expect("fnp op");
            let numpy_fn = numpy.getattr(op).expect("numpy op");
            group.bench_function(format!("fnp_{op}_f64_8m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((x, y)).expect("fnp call")));
            });
            group.bench_function(format!("numpy_{op}_f64_8m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((x, y)).expect("numpy call")));
            });
        }
    });

    group.finish();
}

fn bench_sort_axis_boundary(c: &mut Criterion) {
    // np.sort / np.argsort along the LAST (contiguous) axis of a 2-D f64 array — newly routed
    // through the per-lane parallel sort (numpy sorts each lane single-threaded, sequentially).
    let mut group = c.benchmark_group("python_sort_axis_boundary");
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
m = rng.standard_normal((2048, 2048))\n\
mshort = rng.standard_normal((65536, 64))\n\
m3 = rng.standard_normal((4096, 32, 32))\n\
m3b = rng.standard_normal((256, 256, 64))\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("sort axis setup");
        let m = ns.get_item("m").expect("m");
        let mshort = ns.get_item("mshort").expect("mshort");
        let m3 = ns.get_item("m3").expect("m3");
        let m3b = ns.get_item("m3b").expect("m3b");
        for op in ["sort", "argsort"] {
            let fnp_fn = module.getattr(op).expect("fnp op");
            let numpy_fn = numpy.getattr(op).expect("numpy op");
            group.bench_function(format!("fnp_{op}_lastaxis_2048x2048"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&m,)).expect("fnp call")));
            });
            group.bench_function(format!("numpy_{op}_lastaxis_2048x2048"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&m,)).expect("numpy call")));
            });
            // SHORT-LANE last axis (65536 x 64, cols=64 < SORT_LANE_PARALLEL_MIN=256): tens of
            // thousands of tiny lanes. The eager-parallel path lost 1.8-3.6x here; gated to
            // DELEGATE to numpy -> parity. Guards the 106th-win regression fix.
            group.bench_function(format!("fnp_{op}_lastaxis_short_65536x64"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&mshort,)).expect("fnp call")));
            });
            group.bench_function(format!("numpy_{op}_lastaxis_short_65536x64"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&mshort,)).expect("numpy call")));
            });
            // axis=0 (lane sort): numpy's strided per-lane sort is ~2x its last-axis sort.
            let kw0 = PyDict::new(py);
            kw0.set_item("axis", 0).expect("axis kwarg");
            group.bench_function(format!("fnp_{op}_axis0_2048x2048"), |bch| {
                bch.iter(|| black_box(fnp_fn.call((&m,), Some(&kw0)).expect("fnp call")));
            });
            group.bench_function(format!("numpy_{op}_axis0_2048x2048"), |bch| {
                bch.iter(|| black_box(numpy_fn.call((&m,), Some(&kw0)).expect("numpy call")));
            });
            // ndim>=2 axis=0 on a 3-D batched shape (cols = prod(shape[1:]) lanes).
            group.bench_function(format!("fnp_{op}_axis0_4096x32x32"), |bch| {
                bch.iter(|| black_box(fnp_fn.call((&m3,), Some(&kw0)).expect("fnp call")));
            });
            group.bench_function(format!("numpy_{op}_axis0_4096x32x32"), |bch| {
                bch.iter(|| black_box(numpy_fn.call((&m3,), Some(&kw0)).expect("numpy call")));
            });
            // ndim>=3 MIDDLE axis (axis=1): gather-strided-lane -> sort -> scatter. numpy's
            // strided middle-axis sort is ~1.2-1.8x slower than its last-axis sort, single-threaded.
            let kw1 = PyDict::new(py);
            kw1.set_item("axis", 1).expect("axis kwarg");
            group.bench_function(format!("fnp_{op}_midaxis_4096x32x32"), |bch| {
                bch.iter(|| black_box(fnp_fn.call((&m3,), Some(&kw1)).expect("fnp call")));
            });
            group.bench_function(format!("numpy_{op}_midaxis_4096x32x32"), |bch| {
                bch.iter(|| black_box(numpy_fn.call((&m3,), Some(&kw1)).expect("numpy call")));
            });
            group.bench_function(format!("fnp_{op}_midaxis_256x256x64"), |bch| {
                bch.iter(|| black_box(fnp_fn.call((&m3b,), Some(&kw1)).expect("fnp call")));
            });
            group.bench_function(format!("numpy_{op}_midaxis_256x256x64"), |bch| {
                bch.iter(|| black_box(numpy_fn.call((&m3b,), Some(&kw1)).expect("numpy call")));
            });
        }
    });

    group.finish();
}

fn bench_sort_kind_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_sort_kind_boundary");
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
rng = np.random.default_rng(1)\n\
m = rng.standard_normal((2048, 2048))\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("sort kind setup");
        let m = ns.get_item("m").expect("m");
        let kwargs = PyDict::new(py);
        kwargs.set_item("kind", "stable").expect("kind kwarg");
        for op in ["sort", "argsort"] {
            let fnp_fn = module.getattr(op).expect("fnp op");
            let numpy_fn = numpy.getattr(op).expect("numpy op");
            group.bench_function(format!("fnp_{op}_stable_lastaxis_2048x2048"), |bch| {
                bch.iter(|| black_box(fnp_fn.call((&m,), Some(&kwargs)).expect("fnp call")));
            });
            group.bench_function(format!("numpy_{op}_stable_lastaxis_2048x2048"), |bch| {
                bch.iter(|| black_box(numpy_fn.call((&m,), Some(&kwargs)).expect("numpy call")));
            });
        }
    });

    group.finish();
}

// np.matmul / np.dot on 2-D f64 squares spanning the native GEMM gate window
// ([320..1024]). The native pure-Rust GEMM was profiled as winning here, but a
// later numpy/OpenBLAS speedup made the gate stale (now a 1.5-6x loss vs BLAS).
fn bench_matmul_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_matmul_boundary");
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
a512 = rng.standard_normal((512, 512))\n\
b512 = rng.standard_normal((512, 512))\n\
a1024 = rng.standard_normal((1024, 1024))\n\
b1024 = rng.standard_normal((1024, 1024))\n\
a1536 = rng.standard_normal((1536, 1536))\n\
b1536 = rng.standard_normal((1536, 1536))\n\
a2048 = rng.standard_normal((2048, 2048))\n\
b2048 = rng.standard_normal((2048, 2048))\n\
a3d = rng.standard_normal((64, 256, 256))\n\
b3d = rng.standard_normal((64, 256, 256))\n\
a3db = rng.standard_normal((256, 128, 128))\n\
b3db = rng.standard_normal((256, 128, 128))\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("matmul setup");
        let m_fn = module.getattr("matmul").expect("fnp matmul");
        let np_m = numpy.getattr("matmul").expect("np matmul");
        for op in ["matmul", "dot"] {
            let fnp_fn = module.getattr(op).expect("fnp op");
            let numpy_fn = numpy.getattr(op).expect("numpy op");
            for sz in ["512", "1024", "1536", "2048"] {
                let a = ns.get_item(format!("a{sz}")).expect("a");
                let b = ns.get_item(format!("b{sz}")).expect("b");
                group.bench_function(format!("fnp_{op}_{sz}x{sz}"), |bch| {
                    bch.iter(|| black_box(fnp_fn.call1((&a, &b)).expect("fnp call")));
                });
                group.bench_function(format!("numpy_{op}_{sz}x{sz}"), |bch| {
                    bch.iter(|| black_box(numpy_fn.call1((&a, &b)).expect("numpy call")));
                });
            }
        }
        let fnp_tensordot = module.getattr("tensordot").expect("fnp tensordot");
        let np_tensordot = numpy.getattr("tensordot").expect("np tensordot");
        for sz in ["1024", "1536"] {
            let a = ns.get_item(format!("a{sz}")).expect("a");
            let b = ns.get_item(format!("b{sz}")).expect("b");
            group.bench_function(format!("fnp_tensordot_axes1_{sz}x{sz}"), |bch| {
                bch.iter(|| {
                    black_box(
                        fnp_tensordot
                            .call1((&a, &b, 1_i64))
                            .expect("fnp call"),
                    )
                });
            });
            group.bench_function(format!("numpy_tensordot_axes1_{sz}x{sz}"), |bch| {
                bch.iter(|| {
                    black_box(
                        np_tensordot
                            .call1((&a, &b, 1_i64))
                            .expect("numpy call"),
                    )
                });
            });
        }
        // Batched (3-D) matmul: native parallel-across-batch packed GEMM vs numpy slow BLAS.
        for (tag, ak, bk) in [("64x256x256", "a3d", "b3d"), ("256x128x128", "a3db", "b3db")] {
            let a = ns.get_item(ak).expect("a3d");
            let b = ns.get_item(bk).expect("b3d");
            group.bench_function(format!("fnp_matmul_batched_{tag}"), |bch| {
                bch.iter(|| black_box(m_fn.call1((&a, &b)).expect("fnp call")));
            });
            group.bench_function(format!("numpy_matmul_batched_{tag}"), |bch| {
                bch.iter(|| black_box(np_m.call1((&a, &b)).expect("numpy call")));
            });
        }
        // Matrix-BROADCAST batched matmul: one 2-D operand applied across the other's batch.
        let bcast_setup = "import numpy as np\n\
rng = np.random.default_rng(2)\n\
ab = rng.standard_normal((64, 256, 256))\n\
wb = rng.standard_normal((256, 256))\n\
aw = rng.standard_normal((256, 256))\n\
bb = rng.standard_normal((64, 256, 256))\n";
        py.run(
            std::ffi::CString::new(bcast_setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("bcast setup");
        let ab = ns.get_item("ab").expect("ab");
        let wb = ns.get_item("wb").expect("wb");
        let aw = ns.get_item("aw").expect("aw");
        let bb = ns.get_item("bb").expect("bb");
        group.bench_function("fnp_matmul_bcast_3dA_2dB_64x256x256", |bch| {
            bch.iter(|| black_box(m_fn.call1((&ab, &wb)).expect("fnp call")));
        });
        group.bench_function("numpy_matmul_bcast_3dA_2dB_64x256x256", |bch| {
            bch.iter(|| black_box(np_m.call1((&ab, &wb)).expect("numpy call")));
        });
        group.bench_function("fnp_matmul_bcast_2dA_3dB_64x256x256", |bch| {
            bch.iter(|| black_box(m_fn.call1((&aw, &bb)).expect("fnp call")));
        });
        group.bench_function("numpy_matmul_bcast_2dA_3dB_64x256x256", |bch| {
            bch.iter(|| black_box(np_m.call1((&aw, &bb)).expect("numpy call")));
        });
        // matrix_power: 2-D square repeated-squaring through the native packed GEMM vs numpy.
        let fnp_mp = module.getattr("matrix_power").expect("fnp matrix_power");
        let np_mp = numpy
            .getattr("linalg")
            .expect("linalg")
            .getattr("matrix_power")
            .expect("np matrix_power");
        for (sz, p) in [("512", 8_i64), ("1024", 6_i64)] {
            let a = ns.get_item(format!("a{sz}")).expect("a");
            group.bench_function(format!("fnp_matrix_power_{sz}x{sz}_p{p}"), |bch| {
                bch.iter(|| black_box(fnp_mp.call1((&a, p)).expect("fnp call")));
            });
            group.bench_function(format!("numpy_matrix_power_{sz}x{sz}_p{p}"), |bch| {
                bch.iter(|| black_box(np_mp.call1((&a, p)).expect("numpy call")));
            });
        }
    });

    group.finish();
}

// np.char.add / np.strings.add (string concatenation) — numpy runs a slow per-element Python
// loop; fnp has a native parallel UCS4 codepoint-copy path.
fn bench_char_add_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_char_add_boundary");
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
a = np.array(['Hello World '+str(i%1000) for i in range(300000)], dtype='<U16')\n\
b = np.array(['suffix'+str(i%50) for i in range(300000)], dtype='<U10')\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("char add setup");
        let a = ns.get_item("a").expect("a");
        let b = ns.get_item("b").expect("b");
        let fnp_add = module
            .getattr("char")
            .expect("char")
            .getattr("add")
            .expect("fnp char.add");
        let np_add = numpy
            .getattr("char")
            .expect("char")
            .getattr("add")
            .expect("np char.add");
        group.bench_function("fnp_char_add_300k", |bch| {
            bch.iter(|| black_box(fnp_add.call1((&a, &b)).expect("fnp call")));
        });
        group.bench_function("numpy_char_add_300k", |bch| {
            bch.iter(|| black_box(np_add.call1((&a, &b)).expect("numpy call")));
        });
    });

    group.finish();
}

fn bench_asarray_dtype_boundary(c: &mut Criterion) {
    // np.asarray(ndarray, dtype=<convert>) — a dtype CONVERSION delegates the cast
    // to numpy, but the native pre-check used to copy the whole input into a
    // UFuncArray before discarding it (a wasted full-array copy on top of numpy's
    // own cast = a 2-3x regression). The fix delegates BEFORE that extract; this
    // guards the conversion path stays at parity with numpy.
    let mut group = c.benchmark_group("python_asarray_dtype_boundary");
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
f64 = rng.standard_normal(4_000_000)\n\
i32 = rng.integers(0, 1000, 4_000_000).astype(np.int32)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("asarray setup");
        let f64 = ns.get_item("f64").expect("f64");
        let i32 = ns.get_item("i32").expect("i32");
        let fnp_asarray = module.getattr("asarray").expect("fnp asarray");
        let np_asarray = numpy.getattr("asarray").expect("np asarray");
        for (name, arr, to) in [
            ("f64_to_f32", &f64, "float32"),
            ("i32_to_f64", &i32, "float64"),
        ] {
            let kw = PyDict::new(py);
            kw.set_item("dtype", to).expect("dtype kwarg");
            group.bench_function(format!("fnp_asarray_{name}_4m"), |bch| {
                bch.iter(|| black_box(fnp_asarray.call((arr,), Some(&kw)).expect("fnp call")));
            });
            group.bench_function(format!("numpy_asarray_{name}_4m"), |bch| {
                bch.iter(|| black_box(np_asarray.call((arr,), Some(&kw)).expect("numpy call")));
            });
        }
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_asarray_dtype_boundary,
    bench_char_add_boundary,
    bench_matmul_boundary,
    bench_sort_kind_boundary,
    bench_sort_axis_boundary,
    bench_parallel_binary_boundary,
    bench_take_axis_boundary,
    bench_take_along_axis_boundary,
    bench_take_boundary,
    bench_searchsorted_boundary,
    bench_digitize_boundary,
    bench_bincount_boundary,
    bench_tile_boundary,
    bench_kron_boundary,
    bench_nan_to_num_boundary,
    bench_cross_boundary,
    bench_sqrt_input_extraction,
    bench_around_boundary,
    bench_where_boundary,
    bench_clip_boundary,
    bench_unary_parallel_boundary,
    bench_int32_unary_boundary,
    bench_narrow_int_unary_boundary,
    bench_remainder_mod_boundary,
    bench_max_min_reduction_boundary,
    bench_ptp_axis0_boundary,
    bench_bool_minmax_reduction_boundary,
    bench_prod_reduction_boundary,
    bench_ediff1d_boundary,
    bench_select_boundary,
    bench_ldexp_boundary,
    bench_float_power_boundary,
    bench_frexp_boundary,
    bench_modf_boundary,
    bench_putmask_boundary,
    bench_shift_boundary,
    bench_concat_hstack_boundary,
    bench_indices_construction_boundary,
    bench_char_ascii_boundary,
    bench_average_nansum_axis_boundary,
    bench_histogram_boundary,
    bench_setops_boundary,
    bench_unique_medium_boundary,
    bench_sort_complex_boundary,
    bench_statistics_boundary,
    bench_std_var_axis_boundary,
    bench_var_multiaxis_boundary,
    bench_var_midaxis_boundary,
    bench_var_f32_axis_boundary,
    bench_nanvar_f32_axis_boundary,
    bench_nanvar_f32_last_axis_boundary,
    bench_nanvar_midaxis_boundary,
    bench_var_axis0_boundary,
    bench_sum_lastaxis_boundary,
    bench_prod_lastaxis_boundary,
    bench_cumsum_lastaxis_boundary,
    bench_cum_midaxis_boundary,
    bench_int_cum_boundary,
    bench_vander_boundary,
    bench_polyval_boundary,
    bench_gradient_axis_boundary,
    bench_norm_axis_boundary,
    bench_norm_nonlast_axis_boundary,
    bench_norm_frobenius_boundary,
    bench_compress_boundary,
    bench_roll_boundary,
    bench_einsum_boundary,
    bench_linalg_boundary
);
criterion_main!(benches);
