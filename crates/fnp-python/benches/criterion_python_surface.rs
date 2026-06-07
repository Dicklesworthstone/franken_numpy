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

        let average_kwargs = PyDict::new(py);
        average_kwargs.set_item("axis", 1_i64).expect("axis kwarg");
        average_kwargs
            .set_item("weights", &weights)
            .expect("weights kwarg");
        let nansum_kwargs = PyDict::new(py);
        nansum_kwargs.set_item("axis", 1_i64).expect("axis kwarg");

        let fnp_average = module.getattr("average").expect("fnp_python.average");
        let numpy_average = numpy.getattr("average").expect("numpy.average");
        let fnp_nansum = module.getattr("nansum").expect("fnp_python.nansum");
        let numpy_nansum = numpy.getattr("nansum").expect("numpy.nansum");

        group.bench_function("fnp_average_axis1_weighted_f64_2048x512", |bench| {
            bench.iter(|| {
                let result = fnp_average
                    .call((&input,), Some(&average_kwargs))
                    .expect("fnp average axis=1 benchmark call");
                black_box(result);
            });
        });

        group.bench_function("numpy_average_axis1_weighted_f64_2048x512", |bench| {
            bench.iter(|| {
                let result = numpy_average
                    .call((&input,), Some(&average_kwargs))
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

criterion_group!(
    benches,
    bench_sqrt_input_extraction,
    bench_int32_unary_boundary,
    bench_narrow_int_unary_boundary,
    bench_max_min_reduction_boundary,
    bench_prod_reduction_boundary,
    bench_ediff1d_boundary,
    bench_select_boundary,
    bench_ldexp_boundary,
    bench_float_power_boundary,
    bench_frexp_boundary,
    bench_shift_boundary,
    bench_concat_hstack_boundary,
    bench_char_ascii_boundary,
    bench_average_nansum_axis_boundary,
    bench_histogram_boundary
);
criterion_main!(benches);
