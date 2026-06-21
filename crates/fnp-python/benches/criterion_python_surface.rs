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

criterion_group!(
    benches,
    bench_sqrt_input_extraction,
    bench_int32_unary_boundary,
    bench_narrow_int_unary_boundary,
    bench_max_min_reduction_boundary,
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
    bench_sort_complex_boundary,
    bench_statistics_boundary,
    bench_compress_boundary,
    bench_roll_boundary,
    bench_einsum_boundary,
    bench_linalg_boundary
);
criterion_main!(benches);
