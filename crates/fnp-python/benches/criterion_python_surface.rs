//! Criterion benchmarks for the PyO3 `fnp_python` surface.
//!
//! These target Python-boundary costs that the Rust engine benches do not see.

use criterion::{Criterion, criterion_group, criterion_main};
use fnp_python::fnp_python;
use pyo3::Bound;
use pyo3::types::{PyAnyMethods, PyDict, PyModule, PyTuple};
use pyo3::{Py, PyAny, PyResult, Python};
use rayon::prelude::*;
use std::cell::{Cell, RefCell};
use std::hint::black_box;
use std::time::{Duration, Instant};

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

// datetime64 unit conversion (ns->us downcast = floor-div; s->ns upcast = mul): numpy runs it
// single-threaded per-element with NaT checks (~98ms@16M downcast). The native parallel path
// is bit-exact and wins. RAYON_NUM_THREADS=1 vs default isolates the parallel gain.
fn bench_temporal_astype_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_temporal_astype_boundary");
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
dt = rng.integers(0, 10**18, 16_000_000).astype(np.int64).view('datetime64[ns]')\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("astype setup");
        let dt = ns.get_item("dt").expect("dt");
        let fnp_astype = module.getattr("astype").expect("fnp astype");
        group.bench_function("fnp_datetime_ns_to_us_16m", |b| {
            b.iter(|| {
                black_box(
                    fnp_astype
                        .call1((&dt, "datetime64[us]"))
                        .expect("fnp astype"),
                )
            });
        });
        group.bench_function("numpy_datetime_ns_to_us_16m", |b| {
            b.iter(|| black_box(dt.call_method1("astype", ("datetime64[us]",)).expect("np astype")));
        });
    });

    group.finish();
}

// timedelta64 add/subtract (td +/- td -> td): numpy runs int64 add/sub with per-element NaT
// checks single-threaded (~90ms@16M). The native parallel wrapping op with inline NaT is
// bit-exact and wins ~4x. RAYON_NUM_THREADS=1 vs default isolates the parallel gain.
fn bench_timedelta_addsub_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_timedelta_addsub_boundary");
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
a = rng.integers(0, 10**9, 16_000_000).astype(np.int64).view('timedelta64[ns]')\n\
b = rng.integers(1, 10**6, 16_000_000).astype(np.int64).view('timedelta64[ns]')\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("td setup");
        let a = ns.get_item("a").expect("a");
        let b = ns.get_item("b").expect("b");
        for name in ["add", "subtract"] {
            let fnp_fn = module.getattr(name).expect("fnp fn");
            let numpy_fn = numpy.getattr(name).expect("numpy fn");
            group.bench_function(format!("fnp_{name}_timedelta_16m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&a, &b)).expect("fnp td addsub")));
            });
            group.bench_function(format!("numpy_{name}_timedelta_16m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&a, &b)).expect("np td addsub")));
            });
        }
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
// np.ptp(f32, axis): f32 had no ptp-axis kernel (int uses Ord-based, f64 its own), so f32
// delegated to numpy's two-pass amax/amin (~parity non-last, a LOSS on axis=0). The f32 twin
// (fused NaN-propagating max/min, parallel) wins ~4x (mid) and turns the axis=0 loss into a win.
fn bench_ptp_f32_axis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_ptp_f32_axis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
mid = rng.standard_normal((512, 512, 32)).astype(np.float32)\n\
tall = rng.standard_normal((524288, 32)).astype(np.float32)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("ptp f32 setup");
        let fnp_ptp = module.getattr("ptp").expect("fnp ptp");
        let numpy_ptp = numpy.getattr("ptp").expect("numpy ptp");
        for (label, var, ax) in [("mid", "mid", 1_i64), ("axis0", "tall", 0_i64)] {
            let arr = ns.get_item(var).expect("arr");
            let kw = PyDict::new(py);
            kw.set_item("axis", ax).unwrap();
            let kw2 = kw.clone();
            group.bench_function(format!("fnp_ptp_f32_{label}"), |b| {
                b.iter(|| black_box(fnp_ptp.call((&arr,), Some(&kw)).expect("fnp ptp")));
            });
            group.bench_function(format!("numpy_ptp_f32_{label}"), |b| {
                b.iter(|| black_box(numpy_ptp.call((&arr,), Some(&kw2)).expect("np ptp")));
            });
        }
    });

    group.finish();
}

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

// np.diff(1-D f64, n): the 1-D first-difference kernel was serial (parity/slight loss vs numpy's
// SIMD subtract); parallelizing it (mirroring ediff1d) wins ~3.2x, and n>1 iterates the fast diff.
fn bench_diff_1d_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_diff_1d_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let big = numpy
            .call_method1("linspace", (-1e6_f64, 1e6_f64, 8_000_000_usize))
            .expect("8M f64 input");
        let fnp_diff = module.getattr("diff").expect("fnp diff");
        let numpy_diff = numpy.getattr("diff").expect("numpy diff");
        for n in [1_i64, 2] {
            let kw = PyDict::new(py);
            kw.set_item("n", n).unwrap();
            let kw2 = kw.clone();
            group.bench_function(format!("fnp_diff_n{n}_f64_8m"), |b| {
                b.iter(|| black_box(fnp_diff.call((&big,), Some(&kw)).expect("fnp diff")));
            });
            group.bench_function(format!("numpy_diff_n{n}_f64_8m"), |b| {
                b.iter(|| black_box(numpy_diff.call((&big,), Some(&kw2)).expect("np diff")));
            });
        }
        // int64 1-D (diff_typed) and 2-D last-axis f64 (diff_axis) — both parallelized.
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
xi = rng.integers(-1000, 1000, 8_000_000).astype(np.int64)\n\
a2 = rng.standard_normal((4096, 2048))\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("diff setup");
        let xi = ns.get_item("xi").expect("xi");
        let a2 = ns.get_item("a2").expect("a2");
        group.bench_function("fnp_diff_int64_1d_8m", |b| {
            b.iter(|| black_box(fnp_diff.call1((&xi,)).expect("fnp diff int")));
        });
        group.bench_function("numpy_diff_int64_1d_8m", |b| {
            b.iter(|| black_box(numpy_diff.call1((&xi,)).expect("np diff int")));
        });
        let axkw = PyDict::new(py);
        axkw.set_item("axis", 1_i64).unwrap();
        let axkw2 = axkw.clone();
        group.bench_function("fnp_diff_f64_2d_axis1", |b| {
            b.iter(|| black_box(fnp_diff.call((&a2,), Some(&axkw)).expect("fnp diff ax1")));
        });
        group.bench_function("numpy_diff_f64_2d_axis1", |b| {
            b.iter(|| black_box(numpy_diff.call((&a2,), Some(&axkw2)).expect("np diff ax1")));
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

// np.logaddexp2(f64 array, f64 SCALAR): the scalar case fell to the single-threaded generic ufunc
// and LOST 0.37x to numpy; now broadcasts the scalar (np.full) into the fast parallel array/array
// kernel -> ~2.9x. (logaddexp2 is a slow per-element log2/exp2 op, so numpy is beatable.)
fn bench_logaddexp2_scalar_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_logaddexp2_scalar_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\nx = rng.standard_normal(1 << 22)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("logaddexp2 setup");
        let x = ns.get_item("x").expect("x");
        let fnp_le2 = module.getattr("logaddexp2").expect("fnp logaddexp2");
        let numpy_le2 = numpy.getattr("logaddexp2").expect("numpy logaddexp2");
        group.bench_function("fnp_logaddexp2_scalar", |b| {
            b.iter(|| black_box(fnp_le2.call1((&x, 2.0_f64)).expect("fnp logaddexp2")));
        });
        group.bench_function("numpy_logaddexp2_scalar", |b| {
            b.iter(|| black_box(numpy_le2.call1((&x, 2.0_f64)).expect("np logaddexp2")));
        });
    });

    group.finish();
}

// np.heaviside(f64 array, f64 SCALAR): the array/scalar case delegated to numpy's slow multi-pass
// scalar-broadcast (~7x below bandwidth); a fused single-pass parallel map wins ~4.5-7x.
fn bench_heaviside_scalar_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_heaviside_scalar_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\nx = rng.standard_normal(1 << 22)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("heaviside setup");
        let x = ns.get_item("x").expect("x");
        let fnp_hv = module.getattr("heaviside").expect("fnp heaviside");
        let numpy_hv = numpy.getattr("heaviside").expect("numpy heaviside");
        group.bench_function("fnp_heaviside_scalar", |b| {
            b.iter(|| black_box(fnp_hv.call1((&x, 0.5_f64)).expect("fnp heaviside")));
        });
        group.bench_function("numpy_heaviside_scalar", |b| {
            b.iter(|| black_box(numpy_hv.call1((&x, 0.5_f64)).expect("np heaviside")));
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

// np.column_stack / np.stack(axis=1) / np.dstack of 1-D arrays == column interleave to (N, K):
// numpy runs a serial page-fault-bound strided copy (~87ms@2x8M). The native parallel row-block
// interleave wins ~4x. All fixed-width dtypes via uint8-view.
fn bench_column_interleave_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_column_interleave_boundary");
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
arrs = [rng.standard_normal(8_000_000) for _ in range(3)]\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("colstack setup");
        let arrs = ns.get_item("arrs").expect("arrs");
        for name in ["column_stack", "dstack"] {
            let fnp_fn = module.getattr(name).expect("fnp fn");
            let numpy_fn = numpy.getattr(name).expect("numpy fn");
            group.bench_function(format!("fnp_{name}_3x8m"), |b| {
                b.iter(|| black_box(fnp_fn.call1((&arrs,)).expect("fnp interleave")));
            });
            group.bench_function(format!("numpy_{name}_3x8m"), |b| {
                b.iter(|| black_box(numpy_fn.call1((&arrs,)).expect("np interleave")));
            });
        }
    });

    group.finish();
}

// np.vstack / np.stack of 1-D equal-length arrays == concatenate(axis=0).reshape(K,N): numpy
// runs a serial page-fault-bound copy (~85ms@4x4M). Routing to fnp's fast concatenate wins ~4x.
fn bench_vstack_1d_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_vstack_1d_boundary");
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
arrs = [rng.standard_normal(4_000_000) for _ in range(4)]\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("vstack setup");
        let arrs = ns.get_item("arrs").expect("arrs");
        for name in ["vstack", "stack"] {
            let fnp_fn = module.getattr(name).expect("fnp fn");
            let numpy_fn = numpy.getattr(name).expect("numpy fn");
            group.bench_function(format!("fnp_{name}_1d_4x4m"), |b| {
                b.iter(|| black_box(fnp_fn.call1((&arrs,)).expect("fnp stack")));
            });
            group.bench_function(format!("numpy_{name}_1d_4x4m"), |b| {
                b.iter(|| black_box(numpy_fn.call1((&arrs,)).expect("np stack")));
            });
        }
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
        let Ok(fnp_char) = module.getattr("char") else {
            return;
        };
        let Ok(numpy_char) = numpy.getattr("char") else {
            return;
        };
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

        // capitalize / title (per-slot ASCII map, parallelized across whole-string slots)
        for op in ["capitalize", "title"] {
            let fnp_op = fnp_char.getattr(op).expect("fnp char op");
            let numpy_op = numpy_char.getattr(op).expect("numpy char op");
            group.bench_function(format!("fnp_char_{op}_u20_ascii_1m"), |bench| {
                bench.iter(|| black_box(fnp_op.call1((&input,)).expect("fnp char call")));
            });
            group.bench_function(format!("numpy_char_{op}_u20_ascii_1m"), |bench| {
                bench.iter(|| black_box(numpy_op.call1((&input,)).expect("numpy char call")));
            });
        }
        // translate (1:1 ASCII codepoint lookup remap, parallelized)
        let builtins = py.import("builtins").expect("builtins");
        let tbl = builtins
            .getattr("str")
            .expect("str")
            .call_method1("maketrans", ("abcdXYZ9", "ABCDxyz0"))
            .expect("maketrans");
        let fnp_tr = fnp_char.getattr("translate").expect("fnp char.translate");
        let numpy_tr = numpy_char.getattr("translate").expect("numpy char.translate");
        group.bench_function("fnp_char_translate_u20_ascii_1m", |bench| {
            bench.iter(|| black_box(fnp_tr.call1((&input, &tbl)).expect("fnp translate")));
        });
        group.bench_function("numpy_char_translate_u20_ascii_1m", |bench| {
            bench.iter(|| black_box(numpy_tr.call1((&input, &tbl)).expect("numpy translate")));
        });
        // char.add: element-wise concat (fixed output width), same-shape arrays
        let kw2 = PyDict::new(py);
        kw2.set_item("dtype", "<U12").expect("dtype kw2");
        let input_b = numpy
            .call_method("full", ((1_000_000_usize,), "_suffix9"), Some(&kw2))
            .expect("1M U12 second operand");
        let fnp_add = fnp_char.getattr("add").expect("fnp char.add");
        let numpy_add = numpy_char.getattr("add").expect("numpy char.add");
        group.bench_function("fnp_char_add_u20u12_ascii_1m", |bench| {
            bench.iter(|| black_box(fnp_add.call1((&input, &input_b)).expect("fnp add")));
        });
        group.bench_function("numpy_char_add_u20u12_ascii_1m", |bench| {
            bench.iter(|| black_box(numpy_add.call1((&input, &input_b)).expect("numpy add")));
        });
        // strip (whitespace, fixed width): input with leading/trailing spaces
        let kw3 = PyDict::new(py);
        kw3.set_item("dtype", "<U20").expect("dtype kw3");
        let input_ws = numpy
            .call_method("full", ((1_000_000_usize,), "   azByCxD0123   "), Some(&kw3))
            .expect("1M U20 ws-padded input");
        let fnp_strip = fnp_char.getattr("strip").expect("fnp char.strip");
        let numpy_strip = numpy_char.getattr("strip").expect("numpy char.strip");
        group.bench_function("fnp_char_strip_u20_ascii_1m", |bench| {
            bench.iter(|| black_box(fnp_strip.call1((&input_ws,)).expect("fnp strip")));
        });
        group.bench_function("numpy_char_strip_u20_ascii_1m", |bench| {
            bench.iter(|| black_box(numpy_strip.call1((&input_ws,)).expect("numpy strip")));
        });
        // replace (per-element non-overlapping, two-pass variable width)
        let fnp_rep = fnp_char.getattr("replace").expect("fnp char.replace");
        let numpy_rep = numpy_char.getattr("replace").expect("numpy char.replace");
        group.bench_function("fnp_char_replace_u20_ascii_1m", |bench| {
            bench.iter(|| black_box(fnp_rep.call1((&input, "C", "QR")).expect("fnp replace")));
        });
        group.bench_function("numpy_char_replace_u20_ascii_1m", |bench| {
            bench.iter(|| black_box(numpy_rep.call1((&input, "C", "QR")).expect("numpy replace")));
        });
        // multiply (repeat n times, two-pass variable width)
        let fnp_mul = fnp_char.getattr("multiply").expect("fnp char.multiply");
        let numpy_mul = numpy_char.getattr("multiply").expect("numpy char.multiply");
        let three = 3_i64;
        group.bench_function("fnp_char_multiply_u20_ascii_1m", |bench| {
            bench.iter(|| black_box(fnp_mul.call1((&input, three)).expect("fnp multiply")));
        });
        group.bench_function("numpy_char_multiply_u20_ascii_1m", |bench| {
            bench.iter(|| black_box(numpy_mul.call1((&input, three)).expect("numpy multiply")));
        });
        // is* bool predicates (fixed bool output, single pass)
        for op in ["isalpha", "isalnum"] {
            let fnp_op = fnp_char.getattr(op).expect("fnp char is-op");
            let numpy_op = numpy_char.getattr(op).expect("numpy char is-op");
            group.bench_function(format!("fnp_char_{op}_u20_ascii_1m"), |bench| {
                bench.iter(|| black_box(fnp_op.call1((&input,)).expect("fnp is-op")));
            });
            group.bench_function(format!("numpy_char_{op}_u20_ascii_1m"), |bench| {
                bench.iter(|| black_box(numpy_op.call1((&input,)).expect("numpy is-op")));
            });
        }
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

// complex128 exp/sin/cos/sinh/cosh/sign: numpy computes these per-element single-threaded
// (~180-420ms@8M). The native parallel real-libm composition is bit-exact (system libm ==
// numpy's npy_c*) and wins on core count. RAYON_NUM_THREADS=1 vs default isolates the
// parallel gain (the serial kernels are ~parity — the win is entirely parallelism).
fn bench_complex_exp_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_complex_exp_boundary");
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
z = (rng.standard_normal(8_000_000) + 1j*rng.standard_normal(8_000_000)).astype(np.complex128)\n\
z64 = z.astype(np.complex64)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("cexp setup");
        let z = ns.get_item("z").expect("z");
        let z64 = ns.get_item("z64").expect("z64");
        for name in ["exp", "sin", "cos", "sinh", "cosh", "sign"] {
            let fnp_fn = module.getattr(name).expect("fnp fn");
            let numpy_fn = numpy.getattr(name).expect("numpy fn");
            group.bench_function(format!("fnp_{name}_complex128_8m"), |b| {
                b.iter(|| black_box(fnp_fn.call1((&z,)).expect("fnp c128 unary")));
            });
            group.bench_function(format!("numpy_{name}_complex128_8m"), |b| {
                b.iter(|| black_box(numpy_fn.call1((&z,)).expect("np c128 unary")));
            });
            group.bench_function(format!("fnp_{name}_complex64_8m"), |b| {
                b.iter(|| black_box(fnp_fn.call1((&z64,)).expect("fnp c64 unary")));
            });
            group.bench_function(format!("numpy_{name}_complex64_8m"), |b| {
                b.iter(|| black_box(numpy_fn.call1((&z64,)).expect("np c64 unary")));
            });
        }
    });

    group.finish();
}

fn bench_complex_binary_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_complex_binary_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");

        // Deterministic non-zero complex operands (divisor never (0,0), so divide takes the
        // native Smith path instead of the zero-divisor defer).
        let make = |n: usize, dt: &str, lo: f64, hi: f64| {
            let re = numpy
                .call_method1("linspace", (lo, hi, n))
                .expect("re linspace");
            let im = numpy
                .call_method1("linspace", (hi, lo + 5.0, n))
                .expect("im linspace");
            let kw = PyDict::new(py);
            kw.set_item("dtype", dt).expect("dtype kw");
            let z = numpy.call_method("empty", (n,), Some(&kw)).expect("empty");
            z.setattr("real", &re).expect("set real");
            z.setattr("imag", &im).expect("set imag");
            z
        };

        let cases: [(&str, &[usize]); 2] = [
            ("complex128", &[262_144, 1_048_576, 16_000_000]),
            ("complex64", &[1_048_576, 16_000_000]),
        ];
        for (dt, sizes) in cases {
            for &n in sizes {
                let a = make(n, dt, -2.0, 3.0);
                let b = make(n, dt, 1.0, 4.0);
                let fmul = module.getattr("multiply").expect("fnp multiply");
                let nmul = numpy.getattr("multiply").expect("numpy multiply");
                let fdiv = module.getattr("divide").expect("fnp divide");
                let ndiv = numpy.getattr("divide").expect("numpy divide");
                let fsq = module.getattr("square").expect("fnp square");
                let nsq = numpy.getattr("square").expect("numpy square");
                group.bench_function(format!("fnp_mul_{dt}_{n}"), |bch| {
                    bch.iter(|| black_box(fmul.call1((&a, &b)).expect("fnp mul")));
                });
                group.bench_function(format!("numpy_mul_{dt}_{n}"), |bch| {
                    bch.iter(|| black_box(nmul.call1((&a, &b)).expect("np mul")));
                });
                group.bench_function(format!("fnp_div_{dt}_{n}"), |bch| {
                    bch.iter(|| black_box(fdiv.call1((&a, &b)).expect("fnp div")));
                });
                group.bench_function(format!("numpy_div_{dt}_{n}"), |bch| {
                    bch.iter(|| black_box(ndiv.call1((&a, &b)).expect("np div")));
                });
                group.bench_function(format!("fnp_sq_{dt}_{n}"), |bch| {
                    bch.iter(|| black_box(fsq.call1((&a,)).expect("fnp sq")));
                });
                group.bench_function(format!("numpy_sq_{dt}_{n}"), |bch| {
                    bch.iter(|| black_box(nsq.call1((&a,)).expect("np sq")));
                });
            }
        }
    });

    group.finish();
}

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
        for sz in [256_usize, 512, 1024] {
            let (a, b) = make(sz);
            group.bench_function(format!("fnp_matmul_f16_{sz}"), |bch| {
                bch.iter(|| black_box(fnp_matmul.call1((&a, &b)).expect("fnp f16 matmul")));
            });
            group.bench_function(format!("numpy_matmul_f16_{sz}"), |bch| {
                bch.iter(|| black_box(numpy_matmul.call1((&a, &b)).expect("np f16 matmul")));
            });
        }
        // batched (3-D) f16 matmul
        let make3 = |b: usize, sz: usize| {
            let rng = default_rng.call1(((b + sz) as i64,)).expect("rng3");
            let a = rng
                .call_method1("standard_normal", ((b, sz, sz),))
                .expect("a3")
                .call_method1("__mul__", (0.3_f64,))
                .expect("scale a3")
                .call_method1("astype", ("float16",))
                .expect("a3 f16");
            let bb = rng
                .call_method1("standard_normal", ((b, sz, sz),))
                .expect("b3")
                .call_method1("__mul__", (0.3_f64,))
                .expect("scale b3")
                .call_method1("astype", ("float16",))
                .expect("b3 f16");
            (a, bb)
        };
        let (a3, b3) = make3(64, 128);
        group.bench_function("fnp_matmul_f16_batched_64x128", |bch| {
            bch.iter(|| black_box(fnp_matmul.call1((&a3, &b3)).expect("fnp f16 batched")));
        });
        group.bench_function("numpy_matmul_f16_batched_64x128", |bch| {
            bch.iter(|| black_box(numpy_matmul.call1((&a3, &b3)).expect("np f16 batched")));
        });
        // broadcast batched: (B,m,k) @ (k,n) with b shared across the batch
        let (ab, _) = make3(64, 128);
        let (bb2d, _) = make(128);
        group.bench_function("fnp_matmul_f16_bcast_64x128", |bch| {
            bch.iter(|| black_box(fnp_matmul.call1((&ab, &bb2d)).expect("fnp f16 bcast")));
        });
        group.bench_function("numpy_matmul_f16_bcast_64x128", |bch| {
            bch.iter(|| black_box(numpy_matmul.call1((&ab, &bb2d)).expect("np f16 bcast")));
        });
        // f16 tensordot(axes=1) + inner at 512 (route to the f16 GEMM kernel)
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

fn bench_flat_sort_dtype_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_flat_sort_dtype_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_sort = module.getattr("sort").expect("fnp sort");
        let numpy_sort = numpy.getattr("sort").expect("numpy sort");
        let default_rng = numpy
            .getattr("random")
            .expect("numpy.random")
            .getattr("default_rng")
            .expect("default_rng");
        let rng = default_rng.call1((7_i64,)).expect("rng");
        let n = 16_000_000_usize;
        // int64, int32, float32 flat-sort inputs
        let i64a = rng
            .call_method1("integers", (i64::MIN, i64::MAX, n))
            .expect("int64 input");
        let i32a = rng
            .call_method1("integers", (-2_000_000_000_i64, 2_000_000_000_i64, n))
            .expect("int32 raw")
            .call_method1("astype", ("int32",))
            .expect("int32 input");
        let f32a = rng
            .call_method1("standard_normal", (n,))
            .expect("f32 raw")
            .call_method1("astype", ("float32",))
            .expect("f32 input");
        for (label, arr) in [("int64", &i64a), ("int32", &i32a), ("float32", &f32a)] {
            group.bench_function(format!("fnp_sort_{label}_16m"), |bch| {
                bch.iter(|| black_box(fnp_sort.call1((arr,)).expect("fnp sort call")));
            });
            group.bench_function(format!("numpy_sort_{label}_16m"), |bch| {
                bch.iter(|| black_box(numpy_sort.call1((arr,)).expect("numpy sort call")));
            });
        }
        // int64 2-D last-axis sort (many wide lanes): 16384 x 1024
        let m2 = rng
            .call_method1("integers", (i64::MIN, i64::MAX, (16384_usize, 1024_usize)))
            .expect("int64 2-D input");
        group.bench_function("fnp_sort_int64_lastaxis_16Mx", |bch| {
            bch.iter(|| black_box(fnp_sort.call1((&m2,)).expect("fnp lastaxis sort")));
        });
        group.bench_function("numpy_sort_int64_lastaxis_16Mx", |bch| {
            bch.iter(|| black_box(numpy_sort.call1((&m2,)).expect("numpy lastaxis sort")));
        });
        // int64 2-D AXIS-0 (column) sort: 1024 x 16384 (axis passed as kwarg so fnp's native
        // single-positional-arg fast path engages).
        let c2 = rng
            .call_method1("integers", (i64::MIN, i64::MAX, (1024_usize, 16384_usize)))
            .expect("int64 axis0 input");
        let axis0_kw = PyDict::new(py);
        axis0_kw.set_item("axis", 0_i64).expect("axis kw");
        group.bench_function("fnp_sort_int64_axis0_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    fnp_sort
                        .call((&c2,), Some(&axis0_kw))
                        .expect("fnp axis0 sort"),
                )
            });
        });
        group.bench_function("numpy_sort_int64_axis0_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    numpy_sort
                        .call((&c2,), Some(&axis0_kw))
                        .expect("numpy axis0 sort"),
                )
            });
        });
        // int64 flat argsort on DISTINCT data (shuffled permutation) -> native path
        let perm = rng
            .call_method1("permutation", (16_000_000_i64,))
            .expect("perm 16M");
        let fnp_argsort = module.getattr("argsort").expect("fnp argsort");
        let numpy_argsort = numpy.getattr("argsort").expect("numpy argsort");
        group.bench_function("fnp_argsort_int64_16m", |bch| {
            bch.iter(|| black_box(fnp_argsort.call1((&perm,)).expect("fnp argsort")));
        });
        group.bench_function("numpy_argsort_int64_16m", |bch| {
            bch.iter(|| black_box(numpy_argsort.call1((&perm,)).expect("numpy argsort")));
        });
        // f32 flat argsort on DISTINCT data (permutation 0..16M-1 < 2^24 = exact f32, no ties)
        let permf32 = perm.call_method1("astype", ("float32",)).expect("perm f32");
        group.bench_function("fnp_argsort_f32_16m", |bch| {
            bch.iter(|| black_box(fnp_argsort.call1((&permf32,)).expect("fnp argsort f32")));
        });
        group.bench_function("numpy_argsort_f32_16m", |bch| {
            bch.iter(|| black_box(numpy_argsort.call1((&permf32,)).expect("numpy argsort f32")));
        });
        // datetime64 flat argsort on DISTINCT ticks (int64-backed; numpy non-simd introsort)
        let permdt = perm.call_method1("astype", ("datetime64[s]",)).expect("perm datetime64");
        group.bench_function("fnp_argsort_datetime64_16m", |bch| {
            bch.iter(|| black_box(fnp_argsort.call1((&permdt,)).expect("fnp argsort dt64")));
        });
        group.bench_function("numpy_argsort_datetime64_16m", |bch| {
            bch.iter(|| black_box(numpy_argsort.call1((&permdt,)).expect("numpy argsort dt64")));
        });
        group.bench_function("fnp_sort_datetime64_16m", |bch| {
            bch.iter(|| black_box(fnp_sort.call1((&permdt,)).expect("fnp sort dt64")));
        });
        group.bench_function("numpy_sort_datetime64_16m", |bch| {
            bch.iter(|| black_box(numpy_sort.call1((&permdt,)).expect("numpy sort dt64")));
        });
        // complex128 flat argsort on DISTINCT real parts (permutation) -> tie-free lexicographic
        let cim = rng
            .call_method1("standard_normal", (16_000_000_usize,))
            .expect("c imag");
        let permc = perm
            .call_method1("__add__", (cim.call_method1("__mul__", (pyo3::types::PyComplex::from_doubles(py, 0.0, 1.0),)).expect("1j*im"),))
            .expect("re+1j*im")
            .call_method1("astype", ("complex128",))
            .expect("perm c128");
        group.bench_function("fnp_argsort_c128_16m", |bch| {
            bch.iter(|| black_box(fnp_argsort.call1((&permc,)).expect("fnp argsort c128")));
        });
        group.bench_function("numpy_argsort_c128_16m", |bch| {
            bch.iter(|| black_box(numpy_argsort.call1((&permc,)).expect("numpy argsort c128")));
        });
        // int64 last-axis argsort, 2-D distinct-per-lane: 16384 x 1024 (each lane a shuffled range)
        let la_randn = rng
            .call_method1("standard_normal", ((16384_usize, 1024_usize),))
            .expect("la randn");
        let la = numpy
            .call_method1("argsort", (la_randn,))
            .expect("la base")
            .call_method1("astype", ("int64",))
            .expect("la int64");
        group.bench_function("fnp_argsort_int64_lastaxis_16Mx", |bch| {
            bch.iter(|| black_box(fnp_argsort.call1((&la,)).expect("fnp argsort la")));
        });
        group.bench_function("numpy_argsort_int64_lastaxis_16Mx", |bch| {
            bch.iter(|| black_box(numpy_argsort.call1((&la,)).expect("numpy argsort la")));
        });
        // int64 AXIS-0 argsort, 2-D distinct-per-column: 1024 x 16384 (each column a shuffled range)
        let a0_randn = rng
            .call_method1("standard_normal", ((1024_usize, 16384_usize),))
            .expect("a0 randn");
        let axis0_kwargs = PyDict::new(py);
        axis0_kwargs.set_item("axis", 0_i64).expect("axis kw");
        let a0 = numpy
            .call_method("argsort", (a0_randn,), Some(&axis0_kwargs))
            .expect("a0 base")
            .call_method1("astype", ("int64",))
            .expect("a0 int64");
        group.bench_function("fnp_argsort_int64_axis0_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    fnp_argsort
                        .call((&a0,), Some(&axis0_kwargs))
                        .expect("fnp argsort a0"),
                )
            });
        });
        group.bench_function("numpy_argsort_int64_axis0_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    numpy_argsort
                        .call((&a0,), Some(&axis0_kwargs))
                        .expect("numpy argsort a0"),
                )
            });
        });
        // int64 MIDDLE-axis argsort, 3-D distinct-per-lane: (256, 256, 256) along axis=1
        let am_randn = rng
            .call_method1("standard_normal", ((256_usize, 256_usize, 256_usize),))
            .expect("am randn");
        let axis1_kwargs = PyDict::new(py);
        axis1_kwargs.set_item("axis", 1_i64).expect("axis1 kw");
        let am = numpy
            .call_method("argsort", (am_randn,), Some(&axis1_kwargs))
            .expect("am base")
            .call_method1("astype", ("int64",))
            .expect("am int64");
        group.bench_function("fnp_argsort_int64_midaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    fnp_argsort
                        .call((&am,), Some(&axis1_kwargs))
                        .expect("fnp argsort am"),
                )
            });
        });
        group.bench_function("numpy_argsort_int64_midaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    numpy_argsort
                        .call((&am,), Some(&axis1_kwargs))
                        .expect("numpy argsort am"),
                )
            });
        });
        // FLOAT32 axis argsort: reuse the distinct int arrays cast to f32 (values < 2^24 = exact, no ties)
        let la_f32 = la.call_method1("astype", ("float32",)).expect("la f32");
        group.bench_function("fnp_argsort_f32_lastaxis_16Mx", |bch| {
            bch.iter(|| black_box(fnp_argsort.call1((&la_f32,)).expect("fnp argsort la f32")));
        });
        group.bench_function("numpy_argsort_f32_lastaxis_16Mx", |bch| {
            bch.iter(|| black_box(numpy_argsort.call1((&la_f32,)).expect("numpy argsort la f32")));
        });
        let a0_f32 = a0.call_method1("astype", ("float32",)).expect("a0 f32");
        group.bench_function("fnp_argsort_f32_axis0_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    fnp_argsort
                        .call((&a0_f32,), Some(&axis0_kwargs))
                        .expect("fnp argsort a0 f32"),
                )
            });
        });
        group.bench_function("numpy_argsort_f32_axis0_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    numpy_argsort
                        .call((&a0_f32,), Some(&axis0_kwargs))
                        .expect("numpy argsort a0 f32"),
                )
            });
        });
        let am_f32 = am.call_method1("astype", ("float32",)).expect("am f32");
        group.bench_function("fnp_argsort_f32_midaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    fnp_argsort
                        .call((&am_f32,), Some(&axis1_kwargs))
                        .expect("fnp argsort am f32"),
                )
            });
        });
        group.bench_function("numpy_argsort_f32_midaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    numpy_argsort
                        .call((&am_f32,), Some(&axis1_kwargs))
                        .expect("numpy argsort am f32"),
                )
            });
        });
        // COMPLEX128 axis argsort: distinct-real lane arrays (la/a0/am) + 1j*randn -> tie-free lexicographic
        let onej = pyo3::types::PyComplex::from_doubles(py, 0.0, 1.0);
        let la_c = la
            .call_method1(
                "__add__",
                (rng
                    .call_method1("standard_normal", (la.getattr("shape").expect("la shape"),))
                    .expect("la imag")
                    .call_method1("__mul__", (&onej,))
                    .expect("1j*la_im"),),
            )
            .expect("la re+im")
            .call_method1("astype", ("complex128",))
            .expect("la c128");
        group.bench_function("fnp_argsort_c128_lastaxis_16Mx", |bch| {
            bch.iter(|| black_box(fnp_argsort.call1((&la_c,)).expect("fnp argsort la c128")));
        });
        group.bench_function("numpy_argsort_c128_lastaxis_16Mx", |bch| {
            bch.iter(|| black_box(numpy_argsort.call1((&la_c,)).expect("numpy argsort la c128")));
        });
        let a0_c = a0
            .call_method1(
                "__add__",
                (rng
                    .call_method1("standard_normal", (a0.getattr("shape").expect("a0 shape"),))
                    .expect("a0 imag")
                    .call_method1("__mul__", (&onej,))
                    .expect("1j*a0_im"),),
            )
            .expect("a0 re+im")
            .call_method1("astype", ("complex128",))
            .expect("a0 c128");
        group.bench_function("fnp_argsort_c128_axis0_16Mx", |bch| {
            bch.iter(|| {
                black_box(fnp_argsort.call((&a0_c,), Some(&axis0_kwargs)).expect("fnp argsort a0 c128"))
            });
        });
        group.bench_function("numpy_argsort_c128_axis0_16Mx", |bch| {
            bch.iter(|| {
                black_box(numpy_argsort.call((&a0_c,), Some(&axis0_kwargs)).expect("numpy argsort a0 c128"))
            });
        });
        let am_c = am
            .call_method1(
                "__add__",
                (rng
                    .call_method1("standard_normal", (am.getattr("shape").expect("am shape"),))
                    .expect("am imag")
                    .call_method1("__mul__", (&onej,))
                    .expect("1j*am_im"),),
            )
            .expect("am re+im")
            .call_method1("astype", ("complex128",))
            .expect("am c128");
        group.bench_function("fnp_argsort_c128_midaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(fnp_argsort.call((&am_c,), Some(&axis1_kwargs)).expect("fnp argsort am c128"))
            });
        });
        group.bench_function("numpy_argsort_c128_midaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(numpy_argsort.call((&am_c,), Some(&axis1_kwargs)).expect("numpy argsort am c128"))
            });
        });
        // COMPLEX128 VALUE sort (np.sort): flat (permc, 16M distinct-real) + last-axis (la_c, distinct-per-lane)
        group.bench_function("fnp_sort_c128_16m", |bch| {
            bch.iter(|| black_box(fnp_sort.call1((&permc,)).expect("fnp sort c128")));
        });
        group.bench_function("numpy_sort_c128_16m", |bch| {
            bch.iter(|| black_box(numpy_sort.call1((&permc,)).expect("numpy sort c128")));
        });
        group.bench_function("fnp_sort_c128_lastaxis_16Mx", |bch| {
            bch.iter(|| black_box(fnp_sort.call1((&la_c,)).expect("fnp sort la c128")));
        });
        group.bench_function("numpy_sort_c128_lastaxis_16Mx", |bch| {
            bch.iter(|| black_box(numpy_sort.call1((&la_c,)).expect("numpy sort la c128")));
        });
        // COMPLEX128 VALUE sort AXIS0 + MIDAXIS: reuse a0_c (distinct-per-column) + am_c (distinct-per-lane)
        group.bench_function("fnp_sort_c128_axis0_16Mx", |bch| {
            bch.iter(|| {
                black_box(fnp_sort.call((&a0_c,), Some(&axis0_kwargs)).expect("fnp sort a0 c128"))
            });
        });
        group.bench_function("numpy_sort_c128_axis0_16Mx", |bch| {
            bch.iter(|| {
                black_box(numpy_sort.call((&a0_c,), Some(&axis0_kwargs)).expect("numpy sort a0 c128"))
            });
        });
        group.bench_function("fnp_sort_c128_midaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(fnp_sort.call((&am_c,), Some(&axis1_kwargs)).expect("fnp sort am c128"))
            });
        });
        group.bench_function("numpy_sort_c128_midaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(numpy_sort.call((&am_c,), Some(&axis1_kwargs)).expect("numpy sort am c128"))
            });
        });
        // COMPLEX64 VALUE sort (np.sort): permc/la_c cast to complex64 (distinct-real -> tie-free)
        let permc64 = permc.call_method1("astype", ("complex64",)).expect("permc64");
        group.bench_function("fnp_sort_c64_16m", |bch| {
            bch.iter(|| black_box(fnp_sort.call1((&permc64,)).expect("fnp sort c64")));
        });
        group.bench_function("numpy_sort_c64_16m", |bch| {
            bch.iter(|| black_box(numpy_sort.call1((&permc64,)).expect("numpy sort c64")));
        });
        let la_c64 = la_c.call_method1("astype", ("complex64",)).expect("la_c64");
        group.bench_function("fnp_sort_c64_lastaxis_16Mx", |bch| {
            bch.iter(|| black_box(fnp_sort.call1((&la_c64,)).expect("fnp sort la c64")));
        });
        group.bench_function("numpy_sort_c64_lastaxis_16Mx", |bch| {
            bch.iter(|| black_box(numpy_sort.call1((&la_c64,)).expect("numpy sort la c64")));
        });
        // COMPLEX64 argsort: reuse permc64 (flat distinct-real) + la_c64 (last-axis distinct-per-lane)
        group.bench_function("fnp_argsort_c64_16m", |bch| {
            bch.iter(|| black_box(fnp_argsort.call1((&permc64,)).expect("fnp argsort c64")));
        });
        group.bench_function("numpy_argsort_c64_16m", |bch| {
            bch.iter(|| black_box(numpy_argsort.call1((&permc64,)).expect("numpy argsort c64")));
        });
        group.bench_function("fnp_argsort_c64_lastaxis_16Mx", |bch| {
            bch.iter(|| black_box(fnp_argsort.call1((&la_c64,)).expect("fnp argsort la c64")));
        });
        group.bench_function("numpy_argsort_c64_lastaxis_16Mx", |bch| {
            bch.iter(|| black_box(numpy_argsort.call1((&la_c64,)).expect("numpy argsort la c64")));
        });
        // COMPLEX64 argsort AXIS0 + MIDAXIS: reuse a0_c/am_c (distinct-real) cast to complex64
        let a0_c64 = a0_c.call_method1("astype", ("complex64",)).expect("a0_c64");
        group.bench_function("fnp_argsort_c64_axis0_16Mx", |bch| {
            bch.iter(|| {
                black_box(fnp_argsort.call((&a0_c64,), Some(&axis0_kwargs)).expect("fnp argsort a0 c64"))
            });
        });
        group.bench_function("numpy_argsort_c64_axis0_16Mx", |bch| {
            bch.iter(|| {
                black_box(numpy_argsort.call((&a0_c64,), Some(&axis0_kwargs)).expect("numpy argsort a0 c64"))
            });
        });
        let am_c64 = am_c.call_method1("astype", ("complex64",)).expect("am_c64");
        group.bench_function("fnp_argsort_c64_midaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(fnp_argsort.call((&am_c64,), Some(&axis1_kwargs)).expect("fnp argsort am c64"))
            });
        });
        group.bench_function("numpy_argsort_c64_midaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(numpy_argsort.call((&am_c64,), Some(&axis1_kwargs)).expect("numpy argsort am c64"))
            });
        });
        // COMPLEX64 VALUE sort AXIS0 + MIDAXIS: reuse a0_c64 (distinct-per-column) + am_c64 (distinct-per-lane)
        group.bench_function("fnp_sort_c64_axis0_16Mx", |bch| {
            bch.iter(|| {
                black_box(fnp_sort.call((&a0_c64,), Some(&axis0_kwargs)).expect("fnp sort a0 c64"))
            });
        });
        group.bench_function("numpy_sort_c64_axis0_16Mx", |bch| {
            bch.iter(|| {
                black_box(numpy_sort.call((&a0_c64,), Some(&axis0_kwargs)).expect("numpy sort a0 c64"))
            });
        });
        group.bench_function("fnp_sort_c64_midaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(fnp_sort.call((&am_c64,), Some(&axis1_kwargs)).expect("fnp sort am c64"))
            });
        });
        group.bench_function("numpy_sort_c64_midaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(numpy_sort.call((&am_c64,), Some(&axis1_kwargs)).expect("numpy sort am c64"))
            });
        });
        // datetime64 last-axis argsort: la (16384x1024 distinct-per-lane int64) cast to datetime64[s]
        let la_dt = la.call_method1("astype", ("datetime64[s]",)).expect("la dt64");
        group.bench_function("fnp_argsort_datetime64_lastaxis_16Mx", |bch| {
            bch.iter(|| black_box(fnp_argsort.call1((&la_dt,)).expect("fnp argsort la dt64")));
        });
        group.bench_function("numpy_argsort_datetime64_lastaxis_16Mx", |bch| {
            bch.iter(|| black_box(numpy_argsort.call1((&la_dt,)).expect("numpy argsort la dt64")));
        });
        // datetime64 last-axis VALUE sort (np.sort) on the same 16384x1024 distinct-per-lane dt64.
        group.bench_function("fnp_sort_datetime64_lastaxis_16Mx", |bch| {
            bch.iter(|| black_box(fnp_sort.call1((&la_dt,)).expect("fnp sort la dt64")));
        });
        group.bench_function("numpy_sort_datetime64_lastaxis_16Mx", |bch| {
            bch.iter(|| black_box(numpy_sort.call1((&la_dt,)).expect("numpy sort la dt64")));
        });
        // MIDDLE-axis sort: 3-D 64x4096x64 (=16M), int64 distinct-per-lane (argsort perm) + dt64 cast.
        let m3_randn = rng
            .call_method1("standard_normal", ((64_usize, 4096_usize, 64_usize),))
            .expect("m3 randn");
        let m3_kwargs = PyDict::new(py);
        m3_kwargs.set_item("axis", 1_i64).expect("axis kw");
        let m3 = numpy
            .call_method("argsort", (m3_randn,), Some(&m3_kwargs))
            .expect("m3 base")
            .call_method1("astype", ("int64",))
            .expect("m3 int64");
        group.bench_function("fnp_sort_int64_midaxis_16Mx", |bch| {
            bch.iter(|| black_box(fnp_sort.call((&m3,), Some(&m3_kwargs)).expect("fnp sort m3")));
        });
        group.bench_function("numpy_sort_int64_midaxis_16Mx", |bch| {
            bch.iter(|| black_box(numpy_sort.call((&m3,), Some(&m3_kwargs)).expect("numpy sort m3")));
        });
        let m3_dt = m3.call_method1("astype", ("datetime64[s]",)).expect("m3 dt64");
        group.bench_function("fnp_sort_datetime64_midaxis_16Mx", |bch| {
            bch.iter(|| black_box(fnp_sort.call((&m3_dt,), Some(&m3_kwargs)).expect("fnp sort m3 dt")));
        });
        group.bench_function("numpy_sort_datetime64_midaxis_16Mx", |bch| {
            bch.iter(|| black_box(numpy_sort.call((&m3_dt,), Some(&m3_kwargs)).expect("numpy sort m3 dt")));
        });
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

// Large-n_vars / large-n_obs cov+corrcoef: the Gram-path shapes where the register
// tile and its output stages dominate. bench_statistics_boundary tops out at 500x500,
// which never leaves the small-shape gates, so CI was blind both to Gram-kernel
// regressions and to the fault-storm allocation mode documented in
// docs/NEGATIVE_EVIDENCE.md 2026-07-10 (30.5 MiB result buffers refaulted per call in
// unlucky builds -- these rows make that mode visible as an fnp-vs-numpy ratio shift).
// Conformance is embedded: the group panics if fnp and numpy diverge beyond 1e-12.
fn bench_cov_large_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_cov_large_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
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
        let np_allclose = numpy.getattr("allclose").expect("np.allclose");

        let rng = numpy
            .getattr("random")
            .expect("np.random")
            .call_method1("default_rng", (0_u64,))
            .expect("default_rng");
        let make_input = |rows: usize, cols: usize| {
            rng.call_method1("standard_normal", ((rows, cols),))
                .expect("standard_normal input")
        };
        let inputs = [
            ("2000x500", make_input(2000, 500)),
            ("1000x1000", make_input(1000, 1000)),
            ("500x5000", make_input(500, 5000)),
        ];

        let tol = PyDict::new(py);
        tol.set_item("rtol", 1e-12_f64).expect("rtol");
        tol.set_item("atol", 1e-14_f64).expect("atol");

        for (shape, input) in inputs {
            for (opname, fnp_op, numpy_op) in [
                ("cov", &fnp_cov, &numpy_cov),
                ("corrcoef", &fnp_corrcoef, &numpy_corrcoef),
            ] {
                let ours = fnp_op.call1((&input,)).expect("fnp result");
                let oracle = numpy_op.call1((&input,)).expect("numpy result");
                let close: bool = np_allclose
                    .call((&ours, &oracle), Some(&tol))
                    .expect("allclose call")
                    .extract()
                    .expect("allclose bool");
                assert!(close, "fnp.{opname} diverges from numpy at {shape}");

                group.bench_function(format!("fnp_{opname}_rowvar_f64_{shape}"), |bench| {
                    bench.iter(|| {
                        black_box(fnp_op.call1((&input,)).expect("fnp benchmark call"));
                    });
                });
                group.bench_function(format!("numpy_{opname}_rowvar_f64_{shape}"), |bench| {
                    bench.iter(|| {
                        black_box(numpy_op.call1((&input,)).expect("numpy benchmark call"));
                    });
                });
            }
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
// np.nanmax/nanmin(f32, axis): f32 had no nanextreme-axis kernel (only f64+f16), so with NaN
// present it delegated to numpy which materializes a temp (~80ms@16M). The f32 twin (scalar
// f32::max/min skip-NaN fold, parallel) wins ~18x.
fn bench_nanextreme_f32_axis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_nanextreme_f32_axis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
a = rng.standard_normal((4096, 512, 8)).astype(np.float32)\n\
a[a > 2.0] = np.nan\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("nanextreme f32 setup");
        let a = ns.get_item("a").expect("a");
        for name in ["nanmax", "nanmin"] {
            let fnp_fn = module.getattr(name).expect("fnp fn");
            let numpy_fn = numpy.getattr(name).expect("numpy fn");
            let kw = PyDict::new(py);
            kw.set_item("axis", 1_i64).unwrap();
            let kw2 = kw.clone();
            group.bench_function(format!("fnp_{name}_f32_mid"), |b| {
                b.iter(|| black_box(fnp_fn.call((&a,), Some(&kw)).expect("fnp nanext")));
            });
            group.bench_function(format!("numpy_{name}_f32_mid"), |b| {
                b.iter(|| black_box(numpy_fn.call((&a,), Some(&kw2)).expect("np nanext")));
            });
        }
    });

    group.finish();
}

// np.nansum/nanprod(f32, non-last axis): f32 delegated to numpy's temp-materializing nansum
// (copy + isnan + reduce, ~34ms@8M) while f64 had a kernel. The f32 twin (sequential per-block,
// parallel over outer blocks) avoids the temp AND parallelizes -> ~53x.
fn bench_nansum_f32_axis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_nansum_f32_axis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
a = rng.standard_normal((512, 512, 32)).astype(np.float32)\n\
a[a > 2.0] = np.nan\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("nansum f32 setup");
        let a = ns.get_item("a").expect("a");
        let a64 = a.call_method1("astype", ("float64",)).expect("a64");
        for name in ["nansum", "nanprod"] {
            let fnp_fn = module.getattr(name).expect("fnp fn");
            let numpy_fn = numpy.getattr(name).expect("numpy fn");
            let kw = PyDict::new(py);
            kw.set_item("axis", 1_i64).unwrap();
            let kw2 = kw.clone();
            group.bench_function(format!("fnp_{name}_f32_mid"), |b| {
                b.iter(|| black_box(fnp_fn.call((&a,), Some(&kw)).expect("fnp nan")));
            });
            group.bench_function(format!("numpy_{name}_f32_mid"), |b| {
                b.iter(|| black_box(numpy_fn.call((&a,), Some(&kw2)).expect("np nan")));
            });
        }
        // f64 nansum non-last: the SERIAL branch was parallelized (its sibling nanprod was already
        // parallel) -> ~12x (temp-avoidance) becomes ~40x.
        let fnp_nansum = module.getattr("nansum").expect("fnp nansum");
        let numpy_nansum = numpy.getattr("nansum").expect("numpy nansum");
        let kw = PyDict::new(py);
        kw.set_item("axis", 1_i64).unwrap();
        let kw2 = kw.clone();
        group.bench_function("fnp_nansum_f64_mid", |b| {
            b.iter(|| black_box(fnp_nansum.call((&a64,), Some(&kw)).expect("fnp nansum f64")));
        });
        group.bench_function("numpy_nansum_f64_mid", |b| {
            b.iter(|| black_box(numpy_nansum.call((&a64,), Some(&kw2)).expect("np nansum f64")));
        });
        // f64 nansum LAST axis: per-lane pairwise (now bit-exact, was sequential) + parallel.
        let lax = PyDict::new(py);
        lax.set_item("axis", 2_i64).unwrap();
        let lax2 = lax.clone();
        group.bench_function("fnp_nansum_f64_last", |b| {
            b.iter(|| black_box(fnp_nansum.call((&a64,), Some(&lax)).expect("fnp nansum f64 last")));
        });
        group.bench_function("numpy_nansum_f64_last", |b| {
            b.iter(|| black_box(numpy_nansum.call((&a64,), Some(&lax2)).expect("np nansum f64 last")));
        });
    });

    group.finish();
}

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

fn bench_complex_cumprod_lastaxis_boundary(c: &mut Criterion) {
    // complex128/complex64 last-axis cumprod (multiply.accumulate). numpy runs this as a single-
    // threaded sequential dependency chain (no SIMD escape); the native per-lane parallel scan
    // carries one complex accumulator via the naive cmul, bit-exact, fanned across cores.
    let mut group = c.benchmark_group("python_complex_cumprod_lastaxis_boundary");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_cumprod = module.getattr("cumprod").expect("fnp_python.cumprod");
        let numpy_cumprod = numpy.getattr("cumprod").expect("numpy.cumprod");

        // 16M complex elements (4000x4000). Build unit-magnitude complex (cos + i*sin of a linspace)
        // so the product stays finite — measures the multiply chain, not inf propagation.
        let rows = 4000_i64;
        let cols = 4000_i64;
        let size = rows * cols;
        let theta = numpy
            .call_method1("linspace", (0.0_f64, 6.0_f64, size))
            .expect("theta");
        let re = theta.call_method1("__add__", (0.0_f64,)).expect("re tmp");
        let cosv = numpy.call_method1("cos", (&re,)).expect("cos");
        let sinv = numpy.call_method1("sin", (&theta,)).expect("sin");
        let j = numpy
            .getattr("complex128")
            .expect("complex128")
            .call1((0.0_f64, 1.0_f64))
            .expect("1j");
        let base = cosv
            .call_method1("__add__", (sinv.call_method1("__mul__", (&j,)).expect("i*sin"),))
            .expect("complex base")
            .call_method1("reshape", ((rows, cols),))
            .expect("complex 2-D shape");

        for (label, cname) in [("complex128", "complex128"), ("complex64", "complex64")] {
            let input = base.call_method1("astype", (cname,)).expect("astype complex");
            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("axis", -1_i64).expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs.set_item("axis", -1_i64).expect("numpy axis kwarg");

            group.bench_function(format!("fnp_cumprod_{label}_axis_last_16M"), |bench| {
                bench.iter(|| {
                    let result = fnp_cumprod
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp cumprod call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_cumprod_{label}_axis_last_16M"), |bench| {
                bench.iter(|| {
                    let result = numpy_cumprod
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy cumprod call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

fn bench_complex_nancumprod_lastaxis_boundary(c: &mut Criterion) {
    // complex128/complex64 last-axis nancumprod. numpy replaces NaN-complex with 1+0j then runs the
    // single-threaded multiply.accumulate chain (per-element isnan check + serial scan); the native
    // per-lane parallel nan-scan fans across cores. Bit-exact.
    let mut group = c.benchmark_group("python_complex_nancumprod_lastaxis_boundary");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_nancumprod = module.getattr("nancumprod").expect("fnp_python.nancumprod");
        let numpy_nancumprod = numpy.getattr("nancumprod").expect("numpy.nancumprod");

        let rows = 4000_i64;
        let cols = 4000_i64;
        let size = rows * cols;
        let theta = numpy
            .call_method1("linspace", (0.0_f64, 6.0_f64, size))
            .expect("theta");
        let cosv = numpy.call_method1("cos", (&theta,)).expect("cos");
        let sinv = numpy.call_method1("sin", (&theta,)).expect("sin");
        let j = numpy
            .getattr("complex128")
            .expect("complex128")
            .call1((0.0_f64, 1.0_f64))
            .expect("1j");
        let base = cosv
            .call_method1("__add__", (sinv.call_method1("__mul__", (&j,)).expect("i*sin"),))
            .expect("complex base")
            .call_method1("reshape", ((rows, cols),))
            .expect("complex 2-D shape");

        for (label, cname) in [("complex128", "complex128"), ("complex64", "complex64")] {
            let input = base.call_method1("astype", (cname,)).expect("astype complex");
            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("axis", -1_i64).expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs.set_item("axis", -1_i64).expect("numpy axis kwarg");

            group.bench_function(format!("fnp_nancumprod_{label}_axis_last_16M"), |bench| {
                bench.iter(|| {
                    let result = fnp_nancumprod
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp nancumprod call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_nancumprod_{label}_axis_last_16M"), |bench| {
                bench.iter(|| {
                    let result = numpy_nancumprod
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy nancumprod call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

fn bench_complex_cumulative_midaxis_boundary(c: &mut Criterion) {
    // complex128/complex64 MIDDLE-axis cumprod (3-D, axis=1). numpy runs a non-last complex
    // multiply.accumulate strided + single-threaded; the native per-outer-block parallel slab scan
    // (naive cmul) fans the >=2 independent blocks across cores. Bit-exact.
    let mut group = c.benchmark_group("python_complex_cumulative_midaxis_boundary");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_cumprod = module.getattr("cumprod").expect("fnp_python.cumprod");
        let numpy_cumprod = numpy.getattr("cumprod").expect("numpy.cumprod");
        let fnp_nancumprod = module.getattr("nancumprod").expect("fnp_python.nancumprod");
        let numpy_nancumprod = numpy.getattr("nancumprod").expect("numpy.nancumprod");

        // 256x256x256 = 16M complex, axis=1 (middle -> outer=256 blocks). Unit-magnitude complex so
        // the product stays finite.
        let n = 256_i64;
        let size = n * n * n;
        let theta = numpy
            .call_method1("linspace", (0.0_f64, 6.0_f64, size))
            .expect("theta");
        let cosv = numpy.call_method1("cos", (&theta,)).expect("cos");
        let sinv = numpy.call_method1("sin", (&theta,)).expect("sin");
        let j = numpy
            .getattr("complex128")
            .expect("complex128")
            .call1((0.0_f64, 1.0_f64))
            .expect("1j");
        let base = cosv
            .call_method1("__add__", (sinv.call_method1("__mul__", (&j,)).expect("i*sin"),))
            .expect("complex base")
            .call_method1("reshape", ((n, n, n),))
            .expect("complex 3-D shape");

        for (label, cname) in [("complex128", "complex128"), ("complex64", "complex64")] {
            let input = base.call_method1("astype", (cname,)).expect("astype complex");
            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("axis", 1_i64).expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs.set_item("axis", 1_i64).expect("numpy axis kwarg");

            group.bench_function(format!("fnp_cumprod_{label}_axis_mid_16M"), |bench| {
                bench.iter(|| {
                    let result = fnp_cumprod
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp cumprod mid call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_cumprod_{label}_axis_mid_16M"), |bench| {
                bench.iter(|| {
                    let result = numpy_cumprod
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy cumprod mid call");
                    black_box(result);
                });
            });
            group.bench_function(format!("fnp_nancumprod_{label}_axis_mid_16M"), |bench| {
                bench.iter(|| {
                    let result = fnp_nancumprod
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp nancumprod mid call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_nancumprod_{label}_axis_mid_16M"), |bench| {
                bench.iter(|| {
                    let result = numpy_nancumprod
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy nancumprod mid call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

fn bench_complex_cumulative_axis0_boundary(c: &mut Criterion) {
    // complex128/complex64 AXIS-0 cumprod (2-D). numpy scans DOWN columns = strided + single-threaded
    // (cache-hostile worst case); the native path gathers each column into a contiguous lane, scans in
    // parallel across lanes, and scatters back. Bit-exact.
    let mut group = c.benchmark_group("python_complex_cumulative_axis0_boundary");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_cumprod = module.getattr("cumprod").expect("fnp_python.cumprod");
        let numpy_cumprod = numpy.getattr("cumprod").expect("numpy.cumprod");

        // Sweep sizes to find the axis-0 crossover (gather/scan/scatter has ~2x numpy's traffic, so the
        // win shrinks at small sizes). Unit-magnitude complex (cos + i*sin) per square shape.
        let j = numpy
            .getattr("complex128")
            .expect("complex128")
            .call1((0.0_f64, 1.0_f64))
            .expect("1j");
        for (label, n) in [("1M", 1024_i64), ("4M", 2048_i64), ("16M", 4000_i64)] {
            let size = n * n;
            let theta = numpy
                .call_method1("linspace", (0.0_f64, 6.0_f64, size))
                .expect("theta");
            let cosv = numpy.call_method1("cos", (&theta,)).expect("cos");
            let sinv = numpy.call_method1("sin", (&theta,)).expect("sin");
            let base = cosv
                .call_method1("__add__", (sinv.call_method1("__mul__", (&j,)).expect("i*sin"),))
                .expect("complex base")
                .call_method1("reshape", ((n, n),))
                .expect("complex 2-D shape");
            for (dlabel, cname) in [("complex128", "complex128"), ("complex64", "complex64")] {
                let input = base.call_method1("astype", (cname,)).expect("astype complex");
                let fnp_kwargs = PyDict::new(py);
                fnp_kwargs.set_item("axis", 0_i64).expect("fnp axis kwarg");
                let numpy_kwargs = PyDict::new(py);
                numpy_kwargs.set_item("axis", 0_i64).expect("numpy axis kwarg");

                group.bench_function(format!("fnp_cumprod_{dlabel}_axis0_{label}"), |bench| {
                    bench.iter(|| {
                        let result = fnp_cumprod
                            .call((&input,), Some(&fnp_kwargs))
                            .expect("fnp cumprod axis0 call");
                        black_box(result);
                    });
                });
                group.bench_function(format!("numpy_cumprod_{dlabel}_axis0_{label}"), |bench| {
                    bench.iter(|| {
                        let result = numpy_cumprod
                            .call((&input,), Some(&numpy_kwargs))
                            .expect("numpy cumprod axis0 call");
                        black_box(result);
                    });
                });
            }
        }
    });

    group.finish();
}

fn bench_cumsum_flat_boundary(c: &mut Criterion) {
    // FLAT 1-D integer np.cumsum(8M) — a single-lane prefix sum. numpy's 1-D cumsum is
    // a serial dependency chain; the native two-pass block scan breaks it across cores
    // (bit-exact for wrapping integer add). Was serial (parity with numpy).
    let mut group = c.benchmark_group("python_cumsum_flat_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_cumsum = module.getattr("cumsum").expect("fnp cumsum");
        let numpy_cumsum = numpy.getattr("cumsum").expect("numpy cumsum");
        let i64_in = numpy
            .call_method1("arange", (8_000_000_i64,))
            .expect("8M i64");
        let i32_in = i64_in.call_method1("astype", ("int32",)).expect("i32");
        group.bench_function("fnp_cumsum_i64_flat_8m", |b| {
            b.iter(|| black_box(fnp_cumsum.call1((&i64_in,)).expect("fnp cumsum i64")));
        });
        group.bench_function("numpy_cumsum_i64_flat_8m", |b| {
            b.iter(|| black_box(numpy_cumsum.call1((&i64_in,)).expect("numpy cumsum i64")));
        });
        group.bench_function("fnp_cumsum_i32_flat_8m", |b| {
            b.iter(|| black_box(fnp_cumsum.call1((&i32_in,)).expect("fnp cumsum i32")));
        });
        group.bench_function("numpy_cumsum_i32_flat_8m", |b| {
            b.iter(|| black_box(numpy_cumsum.call1((&i32_in,)).expect("numpy cumsum i32")));
        });
    });

    group.finish();
}

fn bench_accumulate_extremum_boundary(c: &mut Criterion) {
    // FLAT 1-D f64 np.maximum.accumulate(8M) — running max. numpy delegates to a serial
    // prefix scan (dependency chain); the native two-pass parallel prefix breaks it.
    let mut group = c.benchmark_group("python_accumulate_extremum_boundary");
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
x = rng.standard_normal(8_000_000)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("accumulate setup");
        let x = ns.get_item("x").expect("x");
        let fnp_max = module.getattr("maximum").expect("fnp maximum");
        let numpy_max = numpy.getattr("maximum").expect("numpy maximum");
        group.bench_function("fnp_maximum_accumulate_f64_8m", |b| {
            b.iter(|| black_box(fnp_max.call_method1("accumulate", (&x,)).expect("fnp max.accum")));
        });
        group.bench_function("numpy_maximum_accumulate_f64_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_max
                        .call_method1("accumulate", (&x,))
                        .expect("numpy max.accum"),
                )
            });
        });

        // f32 + i64 running max share the generic two-pass (bit-exact: max/min
        // associative for float, no NaN/promotion for int).
        let x32 = x.call_method1("astype", ("float32",)).expect("x32");
        let xi = numpy
            .call_method1("arange", (8_000_000_i64,))
            .expect("8M i64 base")
            .call_method1("__mod__", (1_000_003_i64,))
            .expect("xi");
        group.bench_function("fnp_maximum_accumulate_f32_8m", |b| {
            b.iter(|| black_box(fnp_max.call_method1("accumulate", (&x32,)).expect("fnp max.accum f32")));
        });
        group.bench_function("numpy_maximum_accumulate_f32_8m", |b| {
            b.iter(|| black_box(numpy_max.call_method1("accumulate", (&x32,)).expect("np max.accum f32")));
        });
        group.bench_function("fnp_maximum_accumulate_i64_8m", |b| {
            b.iter(|| black_box(fnp_max.call_method1("accumulate", (&xi,)).expect("fnp max.accum i64")));
        });
        group.bench_function("numpy_maximum_accumulate_i64_8m", |b| {
            b.iter(|| black_box(numpy_max.call_method1("accumulate", (&xi,)).expect("np max.accum i64")));
        });

        // add.accumulate(int) routes to the parallel cumsum path (== np.cumsum); a win
        // here proves the routing engages (vs the prior full delegation to numpy serial).
        let fnp_add = module.getattr("add").expect("fnp add");
        let numpy_add = numpy.getattr("add").expect("numpy add");
        let xa = numpy
            .call_method1("arange", (8_000_000_i64,))
            .expect("8M i64 arange");
        group.bench_function("fnp_add_accumulate_i64_8m", |b| {
            b.iter(|| black_box(fnp_add.call_method1("accumulate", (&xa,)).expect("fnp add.accum i64")));
        });
        group.bench_function("numpy_add_accumulate_i64_8m", |b| {
            b.iter(|| black_box(numpy_add.call_method1("accumulate", (&xa,)).expect("np add.accum i64")));
        });

        // bitwise_or.accumulate(int) native two-pass prefix vs numpy serial.
        let fnp_or = module.getattr("bitwise_or").expect("fnp bitwise_or");
        let numpy_or = numpy.getattr("bitwise_or").expect("numpy bitwise_or");
        group.bench_function("fnp_bitwise_or_accumulate_i64_8m", |b| {
            b.iter(|| black_box(fnp_or.call_method1("accumulate", (&xi,)).expect("fnp or.accum i64")));
        });
        group.bench_function("numpy_bitwise_or_accumulate_i64_8m", |b| {
            b.iter(|| black_box(numpy_or.call_method1("accumulate", (&xi,)).expect("np or.accum i64")));
        });

        // logical_and/or/xor.accumulate(bool): numpy runs a serial dependency-chain scan
        // (~40ms/16M). Bool logical == bitwise (0/1 values), routed to the proven two-pass
        // bitwise prefix. Mask is ~86% True (realistic, avoids AND collapsing to all-False).
        let xb = numpy
            .call_method1("arange", (8_000_000_i64,))
            .expect("8M i64 arange for bool")
            .call_method1("__mod__", (7_i64,))
            .expect("mod 7")
            .call_method1("__ne__", (0_i64,))
            .expect("bool mask");
        let fnp_land = module.getattr("logical_and").expect("fnp logical_and");
        let numpy_land = numpy.getattr("logical_and").expect("numpy logical_and");
        group.bench_function("fnp_logical_and_accumulate_bool_8m", |b| {
            b.iter(|| black_box(fnp_land.call_method1("accumulate", (&xb,)).expect("fnp land.accum bool")));
        });
        group.bench_function("numpy_logical_and_accumulate_bool_8m", |b| {
            b.iter(|| black_box(numpy_land.call_method1("accumulate", (&xb,)).expect("np land.accum bool")));
        });
        let fnp_lor = module.getattr("logical_or").expect("fnp logical_or");
        let numpy_lor = numpy.getattr("logical_or").expect("numpy logical_or");
        group.bench_function("fnp_logical_or_accumulate_bool_8m", |b| {
            b.iter(|| black_box(fnp_lor.call_method1("accumulate", (&xb,)).expect("fnp lor.accum bool")));
        });
        group.bench_function("numpy_logical_or_accumulate_bool_8m", |b| {
            b.iter(|| black_box(numpy_lor.call_method1("accumulate", (&xb,)).expect("np lor.accum bool")));
        });
        let fnp_lxor = module.getattr("logical_xor").expect("fnp logical_xor");
        let numpy_lxor = numpy.getattr("logical_xor").expect("numpy logical_xor");
        group.bench_function("fnp_logical_xor_accumulate_bool_8m", |b| {
            b.iter(|| black_box(fnp_lxor.call_method1("accumulate", (&xb,)).expect("fnp lxor.accum bool")));
        });
        group.bench_function("numpy_logical_xor_accumulate_bool_8m", |b| {
            b.iter(|| black_box(numpy_lxor.call_method1("accumulate", (&xb,)).expect("np lxor.accum bool")));
        });
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

// np.gradient(f32, last axis / 1-D): f32 previously delegated (only f64 had a kernel); the f32
// twin (edge_order=1, bit-identical) wins ~6-8x over numpy's slow pure-Python slice gradient.
// np.gradient(2-D field, cy, cx, edge_order=1): numpy runs each axis through its slow multi-pass
// Python stencil (~215ms @4M); the fused per-axis parallel stencils win ~13x, returning [g0, g1].
fn bench_gradient_2d_coords_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_gradient_2d_coords_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
D = rng.standard_normal((2000, 2000))\ncy = np.sort(rng.standard_normal(2000))\ncx = np.sort(rng.standard_normal(2000))\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("gradient 2d coords setup");
        let d = ns.get_item("D").expect("D");
        let cy = ns.get_item("cy").expect("cy");
        let cx = ns.get_item("cx").expect("cx");
        let fnp_g = module.getattr("gradient").expect("fnp gradient");
        let numpy_g = numpy.getattr("gradient").expect("numpy gradient");
        group.bench_function("fnp_gradient_2d_coords", |b| {
            b.iter(|| black_box(fnp_g.call1((&d, &cy, &cx)).expect("fnp gradient")));
        });
        group.bench_function("numpy_gradient_2d_coords", |b| {
            b.iter(|| black_box(numpy_g.call1((&d, &cy, &cx)).expect("np gradient")));
        });
    });

    group.finish();
}

// np.gradient(f64 1-D, COORDINATE array, edge_order=1): numpy's non-uniform gradient is a multi-pass
// Python-level stencil (~245ms @4M, ~30x below bandwidth); a fused single-pass parallel stencil wins.
fn bench_gradient_coords_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_gradient_coords_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
fd = rng.standard_normal(1 << 22)\nxd = np.sort(rng.standard_normal(1 << 22))\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("gradient coords setup");
        let fd = ns.get_item("fd").expect("fd");
        let xd = ns.get_item("xd").expect("xd");
        let fnp_g = module.getattr("gradient").expect("fnp gradient");
        let numpy_g = numpy.getattr("gradient").expect("numpy gradient");
        group.bench_function("fnp_gradient_coords", |b| {
            b.iter(|| black_box(fnp_g.call1((&fd, &xd)).expect("fnp gradient")));
        });
        group.bench_function("numpy_gradient_coords", |b| {
            b.iter(|| black_box(numpy_g.call1((&fd, &xd)).expect("np gradient")));
        });
    });

    group.finish();
}

// np.gradient(f64 N-D, COORDINATE array, axis=k): a single coord array along one explicit axis of an
// N-D array returns a single array; numpy runs it through the same slow multi-pass Python stencil.
// The fused strided per-plane kernel (outer*la planes, each an inner slab combine) wins. axis=0 of a
// 3-D field exercises the strided (non-contiguous) axis path.
fn bench_gradient_nd_coords_axis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_gradient_nd_coords_axis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
D = rng.standard_normal((256, 256, 64))\ncz = np.sort(rng.standard_normal(256))\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("gradient nd coords setup");
        let d = ns.get_item("D").expect("D");
        let cz = ns.get_item("cz").expect("cz");
        let kwargs = PyDict::new(py);
        kwargs.set_item("axis", 0).expect("axis kwarg");
        let fnp_g = module.getattr("gradient").expect("fnp gradient");
        let numpy_g = numpy.getattr("gradient").expect("numpy gradient");
        group.bench_function("fnp_gradient_nd_coords_axis0", |b| {
            b.iter(|| black_box(fnp_g.call((&d, &cz), Some(&kwargs)).expect("fnp gradient")));
        });
        group.bench_function("numpy_gradient_nd_coords_axis0", |b| {
            b.iter(|| black_box(numpy_g.call((&d, &cz), Some(&kwargs)).expect("np gradient")));
        });
    });

    group.finish();
}

fn bench_gradient_f32_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_gradient_f32_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
x = rng.standard_normal(8_000_000).astype(np.float32)\n\
a2 = rng.standard_normal((4096, 2048)).astype(np.float32)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("gradient f32 setup");
        let x = ns.get_item("x").expect("x");
        let a2 = ns.get_item("a2").expect("a2");
        let fnp_grad = module.getattr("gradient").expect("fnp gradient");
        let numpy_grad = numpy.getattr("gradient").expect("numpy gradient");
        group.bench_function("fnp_gradient_f32_1d_8m", |b| {
            b.iter(|| black_box(fnp_grad.call1((&x,)).expect("fnp grad 1d")));
        });
        group.bench_function("numpy_gradient_f32_1d_8m", |b| {
            b.iter(|| black_box(numpy_grad.call1((&x,)).expect("np grad 1d")));
        });
        let kw = PyDict::new(py);
        kw.set_item("axis", 1_i64).unwrap();
        let kw2 = kw.clone();
        group.bench_function("fnp_gradient_f32_2d_axis1", |b| {
            b.iter(|| black_box(fnp_grad.call((&a2,), Some(&kw)).expect("fnp grad ax1")));
        });
        group.bench_function("numpy_gradient_f32_2d_axis1", |b| {
            b.iter(|| black_box(numpy_grad.call((&a2,), Some(&kw2)).expect("np grad ax1")));
        });
        // axis=0 is the strided (non-last) f32 twin; no-axis returns the per-axis tuple.
        let ax0 = PyDict::new(py);
        ax0.set_item("axis", 0_i64).unwrap();
        let ax0b = ax0.clone();
        group.bench_function("fnp_gradient_f32_2d_axis0", |b| {
            b.iter(|| black_box(fnp_grad.call((&a2,), Some(&ax0)).expect("fnp grad ax0")));
        });
        group.bench_function("numpy_gradient_f32_2d_axis0", |b| {
            b.iter(|| black_box(numpy_grad.call((&a2,), Some(&ax0b)).expect("np grad ax0")));
        });
        group.bench_function("fnp_gradient_f32_2d_noaxis", |b| {
            b.iter(|| black_box(fnp_grad.call1((&a2,)).expect("fnp grad noaxis")));
        });
        group.bench_function("numpy_gradient_f32_2d_noaxis", |b| {
            b.iter(|| black_box(numpy_grad.call1((&a2,)).expect("np grad noaxis")));
        });
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
// np.linalg.norm(f32, ord=+-inf/0, non-last axis): f32 had no norm-axis kernel, so numpy
// materialized abs(x) then a per-axis max/min/count reduce (~90ms@16M). The f32 order-free twin
// (fused max/min|x| fold, parallel) wins ~50x.
fn bench_norm_f32_orderfree_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_norm_f32_orderfree_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
a = rng.standard_normal((4096, 512, 8)).astype(np.float32)\ninf = np.inf\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("norm f32 setup");
        let a = ns.get_item("a").expect("a");
        let inf = ns.get_item("inf").expect("inf");
        let fnp_norm = module.getattr("linalg").unwrap().getattr("norm").expect("fnp norm");
        let numpy_norm = numpy.getattr("linalg").unwrap().getattr("norm").expect("np norm");
        for (label, ordv) in [("maxabs", inf.clone())] {
            let kw = PyDict::new(py);
            kw.set_item("ord", &ordv).unwrap();
            kw.set_item("axis", 1_i64).unwrap();
            let kw2 = kw.clone();
            group.bench_function(format!("fnp_norm_f32_{label}_mid"), |b| {
                b.iter(|| black_box(fnp_norm.call((&a,), Some(&kw)).expect("fnp norm")));
            });
            group.bench_function(format!("numpy_norm_f32_{label}_mid"), |b| {
                b.iter(|| black_box(numpy_norm.call((&a,), Some(&kw2)).expect("np norm")));
            });
        }
    });

    group.finish();
}

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

// f16 arctan2/hypot/logaddexp/logaddexp2: numpy has no f16 ALU, so it widens f16->f32, applies
// the f32 transcendental single-threaded, and narrows (~290/~170/~350/~276ms @16M). The
// native parallel widen-op-narrow is bit-exact for the finite fast-path domains and wins big.
// RAYON_NUM_THREADS=1 vs default isolates the parallel gain.
fn bench_f16_binary_transcendental_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_f16_binary_transcendental_boundary");
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
x = rng.standard_normal(16_000_000).astype(np.float16)\n\
y = rng.standard_normal(16_000_000).astype(np.float16)\n\
pbase = (np.abs(rng.standard_normal(16_000_000)) + 0.5).astype(np.float16)\n\
pexp = (rng.standard_normal(16_000_000) * 0.5).astype(np.float16)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("f16 binary setup");
        let x = ns.get_item("x").expect("x");
        let y = ns.get_item("y").expect("y");
        for name in ["arctan2", "hypot", "logaddexp", "logaddexp2"] {
            let fnp_fn = module.getattr(name).expect("fnp fn");
            let numpy_fn = numpy.getattr(name).expect("numpy fn");
            group.bench_function(format!("fnp_{name}_f16_16m"), |b| {
                b.iter(|| black_box(fnp_fn.call1((&x, &y)).expect("fnp f16 binary")));
            });
            group.bench_function(format!("numpy_{name}_f16_16m"), |b| {
                b.iter(|| black_box(numpy_fn.call1((&x, &y)).expect("np f16 binary")));
            });
        }
        // power uses positive bases + bounded exponents so it engages the native path
        // (negative base / overflow cases defer to numpy by design).
        let pbase = ns.get_item("pbase").expect("pbase");
        let pexp = ns.get_item("pexp").expect("pexp");
        let fnp_pow = module.getattr("power").expect("fnp power");
        let numpy_pow = numpy.getattr("power").expect("numpy power");
        group.bench_function("fnp_power_f16_16m", |b| {
            b.iter(|| black_box(fnp_pow.call1((&pbase, &pexp)).expect("fnp f16 power")));
        });
        group.bench_function("numpy_power_f16_16m", |b| {
            b.iter(|| black_box(numpy_pow.call1((&pbase, &pexp)).expect("np f16 power")));
        });
    });

    group.finish();
}

// np.isin over matched real-float dtypes (f64/f32). numpy can't use its fast
// integer 'table' method for floats, so it falls back to a serial sort of
// |element|+|test| (~3 s for 16M f64). The native zero-copy parallel hashed-set
// path is O(n+m). RAYON_NUM_THREADS=1 vs default isolates the parallel gain (the
// serial hash already crushes numpy's sort ~45x; parallel adds ~10x more).
fn bench_float_isin_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_float_isin_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_isin = module.getattr("isin").expect("fnp isin");
        let numpy_isin = numpy.getattr("isin").expect("numpy isin");
        let setup = "import numpy as np\n\
rng = np.random.default_rng(0)\n\
A64 = rng.standard_normal(8_000_000)\n\
B64 = rng.standard_normal(65_536)\n\
A32 = A64.astype(np.float32)\n\
B32 = B64.astype(np.float32)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("isin setup");
        for (a_key, b_key, label) in [("A64", "B64", "f64"), ("A32", "B32", "f32")] {
            let a = ns.get_item(a_key).expect("a");
            let b = ns.get_item(b_key).expect("b");
            group.bench_function(format!("fnp_isin_{label}_8m"), |bench| {
                bench.iter(|| black_box(fnp_isin.call1((&a, &b)).expect("fnp isin")));
            });
            group.bench_function(format!("numpy_isin_{label}_8m"), |bench| {
                bench.iter(|| black_box(numpy_isin.call1((&a, &b)).expect("np isin")));
            });
        }
    });

    group.finish();
}

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
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("insert setup");
        let x = ns.get_item("x").expect("x");
        let block = ns.get_item("block").expect("block");
        let mid = ns.get_item("mid").expect("mid");
        let fnp_insert = module.getattr("insert").expect("fnp insert");
        let numpy_insert = numpy.getattr("insert").expect("numpy insert");
        group.bench_function("fnp_insert_block_f64_8m", |b| {
            b.iter(|| black_box(fnp_insert.call1((&x, &mid, &block)).expect("fnp insert")));
        });
        group.bench_function("numpy_insert_block_f64_8m", |b| {
            b.iter(|| black_box(numpy_insert.call1((&x, &mid, &block)).expect("np insert")));
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
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("delete setup");
        let x = ns.get_item("x").expect("x");
        let mask = ns.get_item("mask").expect("mask");
        let idx = ns.get_item("idx").expect("idx");
        let fnp_delete = module.getattr("delete").expect("fnp delete");
        let numpy_delete = numpy.getattr("delete").expect("numpy delete");
        for (label, obj) in [("boolmask", &mask), ("intidx", &idx)] {
            group.bench_function(format!("fnp_delete_{label}_8m"), |b| {
                b.iter(|| black_box(fnp_delete.call1((&x, obj)).expect("fnp delete")));
            });
            group.bench_function(format!("numpy_delete_{label}_8m"), |b| {
                b.iter(|| black_box(numpy_delete.call1((&x, obj)).expect("np delete")));
            });
        }
    });

    group.finish();
}

// np.compress(cond, 2-D, axis=1): the native f64 compress-axis path did a scalar per-element column
// gather for the last axis (inner==1) and LOST 0.4-0.8x to numpy's SIMD strided gather. Now delegates
// the inner==1 case -> parity (regression guard: this should track numpy, not the old native loss).
fn bench_compress_lastaxis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_compress_lastaxis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
x = rng.standard_normal((2048, 2048))\ncond = rng.random(2048) < 0.5\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("compress lastaxis setup");
        let x = ns.get_item("x").expect("x");
        let cond = ns.get_item("cond").expect("cond");
        let fnp_compress = module.getattr("compress").expect("fnp compress");
        let numpy_compress = numpy.getattr("compress").expect("numpy compress");
        let kw = PyDict::new(py);
        kw.set_item("axis", 1_i64).unwrap();
        let kw2 = kw.clone();
        group.bench_function("fnp_compress_2d_axis1", |b| {
            b.iter(|| black_box(fnp_compress.call((&cond, &x), Some(&kw)).expect("fnp compress")));
        });
        group.bench_function("numpy_compress_2d_axis1", |b| {
            b.iter(|| black_box(numpy_compress.call((&cond, &x), Some(&kw2)).expect("np compress")));
        });
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

// np.roll(2-D, tuple shifts, tuple axes) for NON-f64 dtypes: numpy does successive full-copy
// concatenations; the f64 fused-parallel path won 3.6x but non-f64 delegated. Generalized to a
// uint8-view byte roll -> int64 3.0x / float32 2.6x / complex128 3.1x.
fn bench_roll_2d_multi_dtype_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_roll_2d_multi_dtype_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
x = rng.integers(-1000, 1000, (4096, 4096)).astype(np.int64)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("roll 2d multi setup");
        let x = ns.get_item("x").expect("x");
        let fnp_roll = module.getattr("roll").expect("fnp roll");
        let numpy_roll = numpy.getattr("roll").expect("numpy roll");
        let shifts = (3_i64, 5_i64);
        let axes = (0_i64, 1_i64);
        group.bench_function("fnp_roll_2d_multi_int64", |b| {
            b.iter(|| {
                black_box(
                    fnp_roll
                        .call1((&x, shifts, axes))
                        .expect("fnp roll"),
                )
            });
        });
        group.bench_function("numpy_roll_2d_multi_int64", |b| {
            b.iter(|| {
                black_box(
                    numpy_roll
                        .call1((&x, shifts, axes))
                        .expect("np roll"),
                )
            });
        });
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

        // INTEGER matrix_power: numpy has no BLAS (repeated naive int matmul). Native
        // binary-exp parallel GEMM should crush it.
        let imp_setup = "import numpy as np\n\
imp = np.random.default_rng(9).integers(-3, 3, (256, 256)).astype(np.int64)\n";
        let imp_ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(imp_setup).unwrap().as_c_str(),
            Some(&imp_ns),
            Some(&imp_ns),
        )
        .expect("int matpow setup");
        let imp = imp_ns.get_item("imp").expect("imp");
        group.bench_function("fnp_matrix_power_i64_256_n5", |bench| {
            bench.iter(|| black_box(fnp_matrix_power.call1((&imp, 5_i64)).expect("fnp int matpow")));
        });
        group.bench_function("numpy_matrix_power_i64_256_n5", |bench| {
            bench.iter(|| black_box(numpy_matrix_power.call1((&imp, 5_i64)).expect("np int matpow")));
        });

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

fn bench_int_convolve_boundary(c: &mut Criterion) {
    // Integer 1-D convolve/correlate: numpy is a direct serial loop (no int fast path).
    let mut group = c.benchmark_group("python_int_convolve_boundary");
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
rng = np.random.default_rng(12)\n\
a = rng.integers(-100, 100, 200_000).astype(np.int64)\n\
v = rng.integers(-100, 100, 256).astype(np.int64)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("int convolve setup");
        let a = ns.get_item("a").expect("a");
        let v = ns.get_item("v").expect("v");
        let fnp_conv = module.getattr("convolve").expect("fnp convolve");
        let numpy_conv = numpy.getattr("convolve").expect("numpy convolve");
        group.bench_function("fnp_convolve_i64_200k_256", |b| {
            b.iter(|| black_box(fnp_conv.call1((&a, &v, "full")).expect("fnp int convolve")));
        });
        group.bench_function("numpy_convolve_i64_200k_256", |b| {
            b.iter(|| black_box(numpy_conv.call1((&a, &v, "full")).expect("numpy int convolve")));
        });
    });

    group.finish();
}

fn bench_f64_convolve_boundary(c: &mut Criterion) {
    // Float64 1-D convolve/correlate with k=256: this is the stale
    // deadlock-audit-1nzxt band. Current main should use the parallel direct
    // gather path for large outputs and beat NumPy's serial direct loop.
    let mut group = c.benchmark_group("python_f64_convolve_boundary");
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
rng = np.random.default_rng(17)\n\
a = rng.standard_normal(1 << 20).astype(np.float64)\n\
v = rng.standard_normal(256).astype(np.float64)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("f64 convolve setup");
        let a = ns.get_item("a").expect("a");
        let v = ns.get_item("v").expect("v");
        let fnp_conv = module.getattr("convolve").expect("fnp convolve");
        let numpy_conv = numpy.getattr("convolve").expect("numpy convolve");
        let fnp_corr = module.getattr("correlate").expect("fnp correlate");
        let numpy_corr = numpy.getattr("correlate").expect("numpy correlate");

        group.bench_function("fnp_convolve_f64_1m_256_same", |b| {
            b.iter(|| black_box(fnp_conv.call1((&a, &v, "same")).expect("fnp convolve")));
        });
        group.bench_function("numpy_convolve_f64_1m_256_same", |b| {
            b.iter(|| {
                black_box(
                    numpy_conv
                        .call1((&a, &v, "same"))
                        .expect("numpy convolve"),
                )
            });
        });
        group.bench_function("fnp_correlate_f64_1m_256_valid", |b| {
            b.iter(|| black_box(fnp_corr.call1((&a, &v, "valid")).expect("fnp correlate")));
        });
        group.bench_function("numpy_correlate_f64_1m_256_valid", |b| {
            b.iter(|| {
                black_box(
                    numpy_corr
                        .call1((&a, &v, "valid"))
                        .expect("numpy correlate"),
                )
            });
        });
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

        // 4-byte int select: like f32, ~13 B/elem traffic crosses the bandwidth floor,
        // so the parallel raw-slice blend should beat numpy's single-threaded where
        // (i64 above is the 8-byte serial control that stays at parity).
        let ia32 = ia.call_method1("astype", ("int32",)).expect("ia32");
        let ib32 = ib.call_method1("astype", ("int32",)).expect("ib32");
        group.bench_function("fnp_where_i32_8m", |bn| {
            bn.iter(|| black_box(fnp_where.call1((&imask, &ia32, &ib32)).expect("fnp where i32")));
        });
        group.bench_function("numpy_where_i32_8m", |bn| {
            bn.iter(|| {
                black_box(
                    numpy_where
                        .call1((&imask, &ia32, &ib32))
                        .expect("numpy where i32"),
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

fn bench_pad_edge_boundary(c: &mut Criterion) {
    // np.pad(1-D, mode="edge"): numpy runs a slow (~0.8 GB/s) single-threaded python path
    // (~77ms @8M f64). fnp splats the first/last element bytes into the edge runs and
    // parallel-memcpys the interior (value-agnostic byte copy) — bit-exact. Covers f64 +
    // the byte path (int32 here). Correctness asserted vs numpy before timing.
    let mut group = c.benchmark_group("python_pad_edge_boundary");
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
xi = rng.integers(-1000, 1000, 8_000_000).astype(np.int32)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("pad edge setup");
        let x = ns.get_item("x").expect("x");
        let xi = ns.get_item("xi").expect("xi");
        let fnp_pad = module.getattr("pad").expect("fnp pad");
        let numpy_pad = numpy.getattr("pad").expect("numpy pad");
        let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
        // Correctness gate: fnp.pad(edge) == numpy.pad(edge) for f64 and int32, scalar
        // width and asymmetric tuple width; panics on any mismatch.
        for (arr, label) in [(&x, "f64"), (&xi, "i32")] {
            let scalar = (
                fnp_pad.call1((arr, 4000_i64, "edge")).expect("fnp pad edge scalar"),
                numpy_pad.call1((arr, 4000_i64, "edge")).expect("numpy pad edge scalar"),
            );
            let tuple = (
                fnp_pad.call1((arr, (3_i64, 7_i64), "edge")).expect("fnp pad edge tuple"),
                numpy_pad.call1((arr, (3_i64, 7_i64), "edge")).expect("numpy pad edge tuple"),
            );
            for (f, n) in [scalar, tuple] {
                let eq: bool = np_array_equal
                    .call1((&f, &n))
                    .expect("array_equal")
                    .extract()
                    .expect("bool");
                assert!(eq, "pad edge correctness mismatch: dtype={label}");
            }
        }
        group.bench_function("fnp_pad_edge_f64_8m", |b| {
            b.iter(|| black_box(fnp_pad.call1((&x, 4000_i64, "edge")).expect("fnp pad edge f64")));
        });
        group.bench_function("numpy_pad_edge_f64_8m", |b| {
            b.iter(|| black_box(numpy_pad.call1((&x, 4000_i64, "edge")).expect("numpy pad edge f64")));
        });
        group.bench_function("fnp_pad_edge_i32_8m", |b| {
            b.iter(|| black_box(fnp_pad.call1((&xi, 4000_i64, "edge")).expect("fnp pad edge i32")));
        });
        group.bench_function("numpy_pad_edge_i32_8m", |b| {
            b.iter(|| black_box(numpy_pad.call1((&xi, 4000_i64, "edge")).expect("numpy pad edge i32")));
        });
    });
    group.finish();
}

fn bench_pad_wrap_boundary(c: &mut Criterion) {
    // np.pad(1-D, mode="wrap") (before<=n & after<=n): numpy runs a slow single-threaded
    // python path; fnp copies the last-`before`/first-`after` contiguous slices into the edge
    // runs and parallel-memcpys the interior (value-agnostic byte copy) — bit-exact. Covers
    // f64 + the byte path (int32). Correctness asserted vs numpy before timing.
    let mut group = c.benchmark_group("python_pad_wrap_boundary");
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
	xi = rng.integers(-1000, 1000, 8_000_000).astype(np.int32)\n\
	xm = rng.integers(-1000, 1000, 4096).astype(np.int32)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("pad wrap setup");
        let x = ns.get_item("x").expect("x");
        let xi = ns.get_item("xi").expect("xi");
        let xm = ns.get_item("xm").expect("xm");
        let fnp_pad = module.getattr("pad").expect("fnp pad");
        let numpy_pad = numpy.getattr("pad").expect("numpy pad");
        let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
        // Correctness gate: fnp.pad(wrap) == numpy.pad(wrap) for f64 and int32, scalar
        // width and asymmetric tuple width; panics on any mismatch.
        for (arr, label) in [(&x, "f64"), (&xi, "i32")] {
            let scalar = (
                fnp_pad.call1((arr, 4000_i64, "wrap")).expect("fnp pad wrap scalar"),
                numpy_pad.call1((arr, 4000_i64, "wrap")).expect("numpy pad wrap scalar"),
            );
            let tuple = (
                fnp_pad.call1((arr, (3_i64, 7_i64), "wrap")).expect("fnp pad wrap tuple"),
                numpy_pad.call1((arr, (3_i64, 7_i64), "wrap")).expect("numpy pad wrap tuple"),
            );
            for (f, n) in [scalar, tuple] {
                let eq: bool = np_array_equal
                    .call1((&f, &n))
                    .expect("array_equal")
                    .extract()
                    .expect("bool");
                assert!(eq, "pad wrap correctness mismatch: dtype={label}");
            }
        }
        let multi_width = (4_000_000_i64, 4_003_000_i64);
        let f_multi = fnp_pad
            .call1((&xm, multi_width, "wrap"))
            .expect("fnp pad wrap multi-tile");
        let n_multi = numpy_pad
            .call1((&xm, multi_width, "wrap"))
            .expect("numpy pad wrap multi-tile");
        let eq: bool = np_array_equal
            .call1((&f_multi, &n_multi))
            .expect("array_equal")
            .extract()
            .expect("bool");
        assert!(eq, "pad wrap multi-tile correctness mismatch");
        group.bench_function("fnp_pad_wrap_f64_8m", |b| {
            b.iter(|| black_box(fnp_pad.call1((&x, 4000_i64, "wrap")).expect("fnp pad wrap f64")));
        });
        group.bench_function("numpy_pad_wrap_f64_8m", |b| {
            b.iter(|| black_box(numpy_pad.call1((&x, 4000_i64, "wrap")).expect("numpy pad wrap f64")));
        });
        group.bench_function("fnp_pad_wrap_i32_8m", |b| {
            b.iter(|| black_box(fnp_pad.call1((&xi, 4000_i64, "wrap")).expect("fnp pad wrap i32")));
        });
        group.bench_function("numpy_pad_wrap_i32_8m", |b| {
            b.iter(|| black_box(numpy_pad.call1((&xi, 4000_i64, "wrap")).expect("numpy pad wrap i32")));
        });
        group.bench_function("fnp_pad_wrap_i32_multitile_8m", |b| {
            b.iter(|| {
                black_box(
                    fnp_pad
                        .call1((&xm, multi_width, "wrap"))
                        .expect("fnp pad wrap multi-tile i32"),
                )
            });
        });
        group.bench_function("numpy_pad_wrap_i32_multitile_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_pad
                        .call1((&xm, multi_width, "wrap"))
                        .expect("numpy pad wrap multi-tile i32"),
                )
            });
        });
    });
    group.finish();
}

fn bench_pad_reflect_boundary(c: &mut Criterion) {
    // np.pad(1-D, mode in {"reflect","symmetric"}): numpy runs a slow single-threaded python
    // path; fnp mirrors the (small) edge runs and parallel-memcpys the (bulk) interior —
    // bit-exact. Covers f64 + the byte path (int32), both modes. Correctness asserted vs numpy.
    let mut group = c.benchmark_group("python_pad_reflect_boundary");
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
xi = rng.integers(-1000, 1000, 8_000_000).astype(np.int32)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("pad reflect setup");
        let x = ns.get_item("x").expect("x");
        let xi = ns.get_item("xi").expect("xi");
        let fnp_pad = module.getattr("pad").expect("fnp pad");
        let numpy_pad = numpy.getattr("pad").expect("numpy pad");
        let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
        // Correctness gate: fnp.pad == numpy.pad for reflect+symmetric, f64+int32, scalar
        // and asymmetric tuple width; panics on any mismatch.
        for md in ["reflect", "symmetric"] {
            for (arr, label) in [(&x, "f64"), (&xi, "i32")] {
                let scalar = (
                    fnp_pad.call1((arr, 4000_i64, md)).expect("fnp pad scalar"),
                    numpy_pad.call1((arr, 4000_i64, md)).expect("numpy pad scalar"),
                );
                let tuple = (
                    fnp_pad.call1((arr, (3_i64, 7_i64), md)).expect("fnp pad tuple"),
                    numpy_pad.call1((arr, (3_i64, 7_i64), md)).expect("numpy pad tuple"),
                );
                for (f, n) in [scalar, tuple] {
                    let eq: bool = np_array_equal
                        .call1((&f, &n))
                        .expect("array_equal")
                        .extract()
                        .expect("bool");
                    assert!(eq, "pad {md} correctness mismatch: dtype={label}");
                }
            }
        }
        group.bench_function("fnp_pad_reflect_f64_8m", |b| {
            b.iter(|| black_box(fnp_pad.call1((&x, 4000_i64, "reflect")).expect("fnp reflect f64")));
        });
        group.bench_function("numpy_pad_reflect_f64_8m", |b| {
            b.iter(|| black_box(numpy_pad.call1((&x, 4000_i64, "reflect")).expect("numpy reflect f64")));
        });
        group.bench_function("fnp_pad_symmetric_i32_8m", |b| {
            b.iter(|| black_box(fnp_pad.call1((&xi, 4000_i64, "symmetric")).expect("fnp symmetric i32")));
        });
        group.bench_function("numpy_pad_symmetric_i32_8m", |b| {
            b.iter(|| black_box(numpy_pad.call1((&xi, 4000_i64, "symmetric")).expect("numpy symmetric i32")));
        });
    });
    group.finish();
}

fn bench_string_sort_boundary(c: &mut Criterion) {
    // np.sort on a 1-D fixed-width unicode ('U') array. numpy sorts with a slow single-threaded
    // per-record codepoint comparator (~280ms @2M U8). For Latin-1 strings (all codepoints <0x100)
    // fnp sorts record indices by memcmp (== codepoint order) in parallel and gathers — bit-exact.
    let mut group = c.benchmark_group("python_string_sort_boundary");
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
x8 = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
x4 = rng.integers(97, 123, (2_000_000, 4), dtype=np.uint32).reshape(-1).view('U4')\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("string sort setup");
        let x8 = ns.get_item("x8").expect("x8");
        let x4 = ns.get_item("x4").expect("x4");
        let fnp_sort = module.getattr("sort").expect("fnp sort");
        let numpy_sort = numpy.getattr("sort").expect("numpy sort");
        let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
        // Correctness gate: fnp.sort == numpy.sort for U8 and U4; panics on mismatch.
        for (arr, label) in [(&x8, "U8"), (&x4, "U4")] {
            let f = fnp_sort.call1((arr,)).expect("fnp sort");
            let n = numpy_sort.call1((arr,)).expect("numpy sort");
            let eq: bool = np_array_equal
                .call1((&f, &n))
                .expect("array_equal")
                .extract()
                .expect("bool");
            assert!(eq, "string sort correctness mismatch: dtype={label}");
        }
        group.bench_function("fnp_sort_U8_2m", |b| {
            b.iter(|| black_box(fnp_sort.call1((&x8,)).expect("fnp sort U8")));
        });
        group.bench_function("numpy_sort_U8_2m", |b| {
            b.iter(|| black_box(numpy_sort.call1((&x8,)).expect("numpy sort U8")));
        });
        group.bench_function("fnp_sort_U4_2m", |b| {
            b.iter(|| black_box(fnp_sort.call1((&x4,)).expect("fnp sort U4")));
        });
        group.bench_function("numpy_sort_U4_2m", |b| {
            b.iter(|| black_box(numpy_sort.call1((&x4,)).expect("numpy sort U4")));
        });
    });
    group.finish();
}

fn bench_string_unique_boundary(c: &mut Criterion) {
    // np.unique on a 1-D fixed-width unicode ('U') array. numpy sorts with the slow per-record
    // codepoint comparator then dedups. For Latin-1 strings fnp sorts packed word keys, dedups
    // adjacent-equal records, and gathers the distinct records — bit-exact. The U16 case covers
    // the two-word wide-key route; U8 mostly-unique and U4 heavy-dedup guard the existing u64 path.
    let mut group = c.benchmark_group("python_string_unique_boundary");
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
x8 = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
xd = rng.integers(97, 101, (2_000_000, 4), dtype=np.uint32).reshape(-1).view('U4')\n\
x16 = rng.integers(97, 123, (1_000_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("string unique setup");
        let x8 = ns.get_item("x8").expect("x8");
        let xd = ns.get_item("xd").expect("xd");
        let x16 = ns.get_item("x16").expect("x16");
        let fnp_unique = module.getattr("unique").expect("fnp unique");
        let numpy_unique = numpy.getattr("unique").expect("numpy unique");
        let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
        // Correctness gate: fnp.unique == numpy.unique for both cases; panics on mismatch.
        for (arr, label) in [
            (&x8, "U8-mostly-unique"),
            (&xd, "U4-heavy-dedup"),
            (&x16, "U16-mostly-unique"),
        ] {
            let f = fnp_unique.call1((arr,)).expect("fnp unique");
            let n = numpy_unique.call1((arr,)).expect("numpy unique");
            let eq: bool = np_array_equal
                .call1((&f, &n))
                .expect("array_equal")
                .extract()
                .expect("bool");
            assert!(eq, "string unique correctness mismatch: case={label}");
        }
        group.bench_function("fnp_unique_U8_2m", |b| {
            b.iter(|| black_box(fnp_unique.call1((&x8,)).expect("fnp unique U8")));
        });
        group.bench_function("numpy_unique_U8_2m", |b| {
            b.iter(|| black_box(numpy_unique.call1((&x8,)).expect("numpy unique U8")));
        });
        group.bench_function("fnp_unique_U4_dedup_2m", |b| {
            b.iter(|| black_box(fnp_unique.call1((&xd,)).expect("fnp unique U4")));
        });
        group.bench_function("numpy_unique_U4_dedup_2m", |b| {
            b.iter(|| black_box(numpy_unique.call1((&xd,)).expect("numpy unique U4")));
        });
        group.bench_function("fnp_unique_U16_1m", |b| {
            b.iter(|| black_box(fnp_unique.call1((&x16,)).expect("fnp unique U16")));
        });
        group.bench_function("numpy_unique_U16_1m", |b| {
            b.iter(|| black_box(numpy_unique.call1((&x16,)).expect("numpy unique U16")));
        });
    });
    group.finish();
}

fn bench_string_unique_full_boundary(c: &mut Criterion) {
    // np.unique on fixed-width bytes with return_index/return_inverse/return_counts. numpy pays the
    // slow record sort plus factorization outputs; fnp carries original indices through the existing
    // byte-record sort and derives group boundaries in one pass.
    let mut group = c.benchmark_group("python_string_unique_full_boundary");
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
a = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint8).view('S8').reshape(-1)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("string unique full setup");
        let a = ns.get_item("a").expect("a");
        let fnp_unique = module.getattr("unique").expect("fnp unique");
        let numpy_unique = numpy.getattr("unique").expect("numpy unique");
        let kw = PyDict::new(py);
        kw.set_item("return_index", true).expect("ri");
        kw.set_item("return_inverse", true).expect("rinv");
        kw.set_item("return_counts", true).expect("rc");
        {
            let got = fnp_unique.call((&a,), Some(&kw)).expect("fnp unique full S8");
            let exp = numpy_unique.call((&a,), Some(&kw)).expect("numpy unique full S8");
            let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
            for t in 0..4usize {
                let g = got.get_item(t).expect("got item");
                let e = exp.get_item(t).expect("exp item");
                let eq: bool = np_array_equal
                    .call1((&g, &e))
                    .expect("array_equal")
                    .extract()
                    .expect("bool");
                assert!(eq, "string unique return_* correctness mismatch at tuple index {t}");
            }
        }
        group.bench_function("fnp_unique_S8_full_2m", |b| {
            b.iter(|| black_box(fnp_unique.call((&a,), Some(&kw)).expect("fnp unique full S8")));
        });
        group.bench_function("numpy_unique_S8_full_2m", |b| {
            b.iter(|| black_box(numpy_unique.call((&a,), Some(&kw)).expect("numpy unique full S8")));
        });
    });
    group.finish();
}

fn bench_string_searchsorted_boundary(c: &mut Criterion) {
    // np.searchsorted into a SORTED 1-D unicode ('U') haystack with a 'U' query array. numpy's
    // per-record codepoint binary search is single-threaded and pathologically slow (~2s @2M+2M).
    // For Latin-1 fnp packs fixed-width records into u64 keys, sorts queries once, and monotonic-merges
    // insertion indices byte-exactly.
    let mut group = c.benchmark_group("python_string_searchsorted_boundary");
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
h8 = np.sort(rng.integers(97, 123, (2_000_000, 8), dtype=np.uint32).reshape(-1).view('U8'))\n\
q8 = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("string searchsorted setup");
        let h8 = ns.get_item("h8").expect("h8");
        let q8 = ns.get_item("q8").expect("q8");
        let fnp_ss = module.getattr("searchsorted").expect("fnp searchsorted");
        let numpy_ss = numpy.getattr("searchsorted").expect("numpy searchsorted");
        let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
        // Correctness gate: fnp.searchsorted == numpy.searchsorted for left and right sides.
        for side in ["left", "right"] {
            let kw = PyDict::new(py);
            kw.set_item("side", side).unwrap();
            let f = fnp_ss.call((&h8, &q8), Some(&kw)).expect("fnp searchsorted");
            let n = numpy_ss.call((&h8, &q8), Some(&kw)).expect("numpy searchsorted");
            let eq: bool = np_array_equal
                .call1((&f, &n))
                .expect("array_equal")
                .extract()
                .expect("bool");
            assert!(eq, "string searchsorted correctness mismatch: side={side}");
        }
        group.bench_function("fnp_searchsorted_U8_left_2m", |b| {
            b.iter(|| black_box(fnp_ss.call1((&h8, &q8)).expect("fnp ss left")));
        });
        group.bench_function("numpy_searchsorted_U8_left_2m", |b| {
            b.iter(|| black_box(numpy_ss.call1((&h8, &q8)).expect("numpy ss left")));
        });
    });
    group.finish();
}

fn bench_string_isin_boundary(c: &mut Criterion) {
    // np.isin on a 1-D unicode ('U') element array against a 'U' test array. numpy sorts
    // |element|+|test| (~730ms .. 2.4s @2M); fnp builds a hashed record-byte set and does a
    // parallel membership scan — bit-exact for ALL codepoints (equality = byte equality).
    let mut group = c.benchmark_group("python_string_isin_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        // test = 100k members drawn from `a` + 100k random non-members, so both bool branches fire.
        let setup = "import numpy as np\n\
rng = np.random.default_rng(0)\n\
a = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
trand = rng.integers(97, 123, (100_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
test = np.concatenate([a[:100_000], trand])\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("string isin setup");
        let a = ns.get_item("a").expect("a");
        let test = ns.get_item("test").expect("test");
        let fnp_isin = module.getattr("isin").expect("fnp isin");
        let numpy_isin = numpy.getattr("isin").expect("numpy isin");
        let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
        // Correctness gate: fnp.isin == numpy.isin for default and invert=True.
        for inv in [false, true] {
            let kw = PyDict::new(py);
            kw.set_item("invert", inv).unwrap();
            let f = fnp_isin.call((&a, &test), Some(&kw)).expect("fnp isin");
            let n = numpy_isin.call((&a, &test), Some(&kw)).expect("numpy isin");
            let eq: bool = np_array_equal
                .call1((&f, &n))
                .expect("array_equal")
                .extract()
                .expect("bool");
            assert!(eq, "string isin correctness mismatch: invert={inv}");
        }
        group.bench_function("fnp_isin_U8_2m", |b| {
            b.iter(|| black_box(fnp_isin.call1((&a, &test)).expect("fnp isin")));
        });
        group.bench_function("numpy_isin_U8_2m", |b| {
            b.iter(|| black_box(numpy_isin.call1((&a, &test)).expect("numpy isin")));
        });
    });
    group.finish();
}

fn bench_string_union1d_boundary(c: &mut Criterion) {
    // np.union1d of two unicode ('U') arrays = sorted-unique of the concatenation. numpy does 2-3
    // slow per-record sorts (~3.2s @2M+2M); fnp concatenates + routes through the native 'U' unique
    // (memcmp sort + dedup) — bit-exact.
    let mut group = c.benchmark_group("python_string_union1d_boundary");
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
a = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
b = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("string union1d setup");
        let a = ns.get_item("a").expect("a");
        let b = ns.get_item("b").expect("b");
        let fnp_u = module.getattr("union1d").expect("fnp union1d");
        let numpy_u = numpy.getattr("union1d").expect("numpy union1d");
        let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
        // Correctness gate.
        let f = fnp_u.call1((&a, &b)).expect("fnp union1d");
        let n = numpy_u.call1((&a, &b)).expect("numpy union1d");
        let eq: bool = np_array_equal
            .call1((&f, &n))
            .expect("array_equal")
            .extract()
            .expect("bool");
        assert!(eq, "string union1d correctness mismatch");
        group.bench_function("fnp_union1d_U8_2m", |bn| {
            bn.iter(|| black_box(fnp_u.call1((&a, &b)).expect("fnp union1d")));
        });
        group.bench_function("numpy_union1d_U8_2m", |bn| {
            bn.iter(|| black_box(numpy_u.call1((&a, &b)).expect("numpy union1d")));
        });

        // Latin-1 U16: the wide two-word-key branch (dedicated per-operand pack, no concat).
        let setup16 = "import numpy as np\n\
rng = np.random.default_rng(2)\n\
a16 = rng.integers(97, 123, (1_000_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
b16 = rng.integers(97, 123, (1_000_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n";
        let ns16 = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup16).unwrap().as_c_str(),
            Some(&ns16),
            Some(&ns16),
        )
        .expect("string union1d U16 setup");
        let a16 = ns16.get_item("a16").expect("a16");
        let b16 = ns16.get_item("b16").expect("b16");
        let f = fnp_u.call1((&a16, &b16)).expect("fnp union1d U16");
        let n = numpy_u.call1((&a16, &b16)).expect("numpy union1d U16");
        let eq: bool = np_array_equal
            .call1((&f, &n))
            .expect("array_equal")
            .extract()
            .expect("bool");
        assert!(eq, "string union1d U16 correctness mismatch");
        group.bench_function("fnp_union1d_U16_1m", |bn| {
            bn.iter(|| black_box(fnp_u.call1((&a16, &b16)).expect("fnp union1d U16")));
        });
        group.bench_function("numpy_union1d_U16_1m", |bn| {
            bn.iter(|| black_box(numpy_u.call1((&a16, &b16)).expect("numpy union1d U16")));
        });
    });
    group.finish();
}

fn bench_string_setops_boundary(c: &mut Criterion) {
    // np.intersect1d / np.setdiff1d of two unicode ('U') arrays. numpy does 2-3 slow per-record
    // sorts (~3.1-3.7s @2M+2M); fnp = memcmp-sort unique(a) + hashed-set membership filter over b.
    let mut group = c.benchmark_group("python_string_setops_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        // b shares 1M values with a and adds 1M fresh, so intersect and setdiff are both non-trivial.
        let setup = "import numpy as np\n\
rng = np.random.default_rng(0)\n\
a = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
brand = rng.integers(97, 123, (1_000_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
b = np.concatenate([a[:1_000_000], brand])\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("string setops setup");
        let a = ns.get_item("a").expect("a");
        let b = ns.get_item("b").expect("b");
        let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
        // Correctness gate for both ops.
        for op in ["intersect1d", "setdiff1d"] {
            let f = module.getattr(op).unwrap().call1((&a, &b)).expect("fnp setop");
            let n = numpy.getattr(op).unwrap().call1((&a, &b)).expect("numpy setop");
            let eq: bool = np_array_equal
                .call1((&f, &n))
                .expect("array_equal")
                .extract()
                .expect("bool");
            assert!(eq, "string {op} correctness mismatch");
        }
        let fnp_i = module.getattr("intersect1d").unwrap();
        let numpy_i = numpy.getattr("intersect1d").unwrap();
        let fnp_d = module.getattr("setdiff1d").unwrap();
        let numpy_d = numpy.getattr("setdiff1d").unwrap();
        group.bench_function("fnp_intersect1d_U8_2m", |bn| {
            bn.iter(|| black_box(fnp_i.call1((&a, &b)).expect("fnp intersect")));
        });
        group.bench_function("numpy_intersect1d_U8_2m", |bn| {
            bn.iter(|| black_box(numpy_i.call1((&a, &b)).expect("numpy intersect")));
        });
        group.bench_function("fnp_setdiff1d_U8_2m", |bn| {
            bn.iter(|| black_box(fnp_d.call1((&a, &b)).expect("fnp setdiff")));
        });
        group.bench_function("numpy_setdiff1d_U8_2m", |bn| {
            bn.iter(|| black_box(numpy_d.call1((&a, &b)).expect("numpy setdiff")));
        });

        // Latin-1 U16: exercises the two-word packed-key branch (U9..U16), which replaces the
        // memcmp record-sort + hashed-membership fallback these widths previously used.
        let setup16 = "import numpy as np\n\
rng = np.random.default_rng(1)\n\
a16 = rng.integers(97, 123, (1_000_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
brand16 = rng.integers(97, 123, (500_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
b16 = np.concatenate([a16[:500_000], brand16])\n";
        let ns16 = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup16).unwrap().as_c_str(),
            Some(&ns16),
            Some(&ns16),
        )
        .expect("string setops U16 setup");
        let a16 = ns16.get_item("a16").expect("a16");
        let b16 = ns16.get_item("b16").expect("b16");
        for op in ["intersect1d", "setdiff1d"] {
            let f = module.getattr(op).unwrap().call1((&a16, &b16)).expect("fnp setop U16");
            let n = numpy.getattr(op).unwrap().call1((&a16, &b16)).expect("numpy setop U16");
            let eq: bool = np_array_equal
                .call1((&f, &n))
                .expect("array_equal")
                .extract()
                .expect("bool");
            assert!(eq, "string {op} U16 correctness mismatch");
        }
        group.bench_function("fnp_intersect1d_U16_1m", |bn| {
            bn.iter(|| black_box(fnp_i.call1((&a16, &b16)).expect("fnp intersect U16")));
        });
        group.bench_function("numpy_intersect1d_U16_1m", |bn| {
            bn.iter(|| black_box(numpy_i.call1((&a16, &b16)).expect("numpy intersect U16")));
        });
        group.bench_function("fnp_setdiff1d_U16_1m", |bn| {
            bn.iter(|| black_box(fnp_d.call1((&a16, &b16)).expect("fnp setdiff U16")));
        });
        group.bench_function("numpy_setdiff1d_U16_1m", |bn| {
            bn.iter(|| black_box(numpy_d.call1((&a16, &b16)).expect("numpy setdiff U16")));
        });
    });
    group.finish();
}

fn bench_string_setxor_boundary(c: &mut Criterion) {
    // np.setxor1d of two unicode ('U') arrays = sorted-unique symmetric difference. numpy does 2-3
    // slow per-record sorts (~3.8s @2M+2M); fnp source-tags + memcmp-sorts the concat, then keeps
    // runs present in exactly one array (no hashing) — bit-exact.
    let mut group = c.benchmark_group("python_string_setxor_boundary");
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
a = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
brand = rng.integers(97, 123, (1_000_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
b = np.concatenate([a[:1_000_000], brand])\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("string setxor setup");
        let a = ns.get_item("a").expect("a");
        let b = ns.get_item("b").expect("b");
        let fnp_x = module.getattr("setxor1d").expect("fnp setxor1d");
        let numpy_x = numpy.getattr("setxor1d").expect("numpy setxor1d");
        let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
        let f = fnp_x.call1((&a, &b)).expect("fnp setxor1d");
        let n = numpy_x.call1((&a, &b)).expect("numpy setxor1d");
        let eq: bool = np_array_equal
            .call1((&f, &n))
            .expect("array_equal")
            .extract()
            .expect("bool");
        assert!(eq, "string setxor1d correctness mismatch");
        group.bench_function("fnp_setxor1d_U8_2m", |bn| {
            bn.iter(|| black_box(fnp_x.call1((&a, &b)).expect("fnp setxor1d")));
        });
        group.bench_function("numpy_setxor1d_U8_2m", |bn| {
            bn.iter(|| black_box(numpy_x.call1((&a, &b)).expect("numpy setxor1d")));
        });

        // Latin-1 U16: the wide two-word-key source-tagged run-composition branch.
        let setup16 = "import numpy as np\n\
rng = np.random.default_rng(3)\n\
a16 = rng.integers(97, 123, (1_000_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
brand16 = rng.integers(97, 123, (500_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
b16 = np.concatenate([a16[:500_000], brand16])\n";
        let ns16 = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup16).unwrap().as_c_str(),
            Some(&ns16),
            Some(&ns16),
        )
        .expect("string setxor U16 setup");
        let a16 = ns16.get_item("a16").expect("a16");
        let b16 = ns16.get_item("b16").expect("b16");
        let f = fnp_x.call1((&a16, &b16)).expect("fnp setxor1d U16");
        let n = numpy_x.call1((&a16, &b16)).expect("numpy setxor1d U16");
        let eq: bool = np_array_equal
            .call1((&f, &n))
            .expect("array_equal")
            .extract()
            .expect("bool");
        assert!(eq, "string setxor1d U16 correctness mismatch");
        group.bench_function("fnp_setxor1d_U16_1m", |bn| {
            bn.iter(|| black_box(fnp_x.call1((&a16, &b16)).expect("fnp setxor1d U16")));
        });
        group.bench_function("numpy_setxor1d_U16_1m", |bn| {
            bn.iter(|| black_box(numpy_x.call1((&a16, &b16)).expect("numpy setxor1d U16")));
        });
    });
    group.finish();
}

fn bench_string_bytes_boundary(c: &mut Criterion) {
    // 'S' (bytes) dtype twins of sort/unique/union1d. Bytes are already in byte order == numpy order
    // (no codepoint subtlety, no Latin-1 gate), so the same byte helpers are bit-exact. numpy is
    // equally slow on 'S' (sort ~238ms, unique ~1245ms, union1d ~3162ms @2M).
    let mut group = c.benchmark_group("python_string_bytes_boundary");
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
a = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint8).view('S8').reshape(-1)\n\
b = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint8).view('S8').reshape(-1)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("string bytes setup");
        let a = ns.get_item("a").expect("a");
        let b = ns.get_item("b").expect("b");
        let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
        // Correctness gates for sort / unique / union1d on 'S8'.
        for op in ["sort", "unique"] {
            let f = module.getattr(op).unwrap().call1((&a,)).expect("fnp op");
            let n = numpy.getattr(op).unwrap().call1((&a,)).expect("numpy op");
            let eq: bool = np_array_equal.call1((&f, &n)).unwrap().extract().unwrap();
            assert!(eq, "bytes {op} correctness mismatch");
        }
        {
            let f = module.getattr("union1d").unwrap().call1((&a, &b)).expect("fnp union1d");
            let n = numpy.getattr("union1d").unwrap().call1((&a, &b)).expect("numpy union1d");
            let eq: bool = np_array_equal.call1((&f, &n)).unwrap().extract().unwrap();
            assert!(eq, "bytes union1d correctness mismatch");
        }
        let fnp_sort = module.getattr("sort").unwrap();
        let numpy_sort = numpy.getattr("sort").unwrap();
        let fnp_uniq = module.getattr("unique").unwrap();
        let numpy_uniq = numpy.getattr("unique").unwrap();
        let fnp_uni = module.getattr("union1d").unwrap();
        let numpy_uni = numpy.getattr("union1d").unwrap();
        group.bench_function("fnp_sort_S8_2m", |bn| bn.iter(|| black_box(fnp_sort.call1((&a,)).unwrap())));
        group.bench_function("numpy_sort_S8_2m", |bn| bn.iter(|| black_box(numpy_sort.call1((&a,)).unwrap())));
        group.bench_function("fnp_unique_S8_2m", |bn| bn.iter(|| black_box(fnp_uniq.call1((&a,)).unwrap())));
        group.bench_function("numpy_unique_S8_2m", |bn| bn.iter(|| black_box(numpy_uniq.call1((&a,)).unwrap())));
        group.bench_function("fnp_union1d_S8_2m", |bn| bn.iter(|| black_box(fnp_uni.call1((&a, &b)).unwrap())));
        group.bench_function("numpy_union1d_S8_2m", |bn| bn.iter(|| black_box(numpy_uni.call1((&a, &b)).unwrap())));
    });
    group.finish();
}

fn bench_string_bytes_ops2_boundary(c: &mut Criterion) {
    // 'S' (bytes) twins of searchsorted/isin/intersect1d/setdiff1d/setxor1d. Same byte helpers,
    // bit-exact for 'S' (no codepoint layer). numpy is slow on all (searchsorted ~766ms, isin ~440ms,
    // set-ops ~1-4s @2M).
    let mut group = c.benchmark_group("python_string_bytes_ops2_boundary");
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
a = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint8).view('S8').reshape(-1)\n\
h = np.sort(a)\n\
q = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint8).view('S8').reshape(-1)\n\
brand = rng.integers(97, 123, (1_000_000, 8), dtype=np.uint8).view('S8').reshape(-1)\n\
b = np.concatenate([a[:1_000_000], brand])\n\
trand = rng.integers(97, 123, (100_000, 8), dtype=np.uint8).view('S8').reshape(-1)\n\
test = np.concatenate([a[:100_000], trand])\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").unwrap();
        let h = ns.get_item("h").unwrap();
        let q = ns.get_item("q").unwrap();
        let b = ns.get_item("b").unwrap();
        let test = ns.get_item("test").unwrap();
        let eqf = numpy.getattr("array_equal").unwrap();
        // Correctness for all 5 ops.
        let ss_f = module.getattr("searchsorted").unwrap().call1((&h, &q)).unwrap();
        let ss_n = numpy.getattr("searchsorted").unwrap().call1((&h, &q)).unwrap();
        assert!(eqf.call1((&ss_f, &ss_n)).unwrap().extract::<bool>().unwrap(), "S searchsorted mismatch");
        let is_f = module.getattr("isin").unwrap().call1((&a, &test)).unwrap();
        let is_n = numpy.getattr("isin").unwrap().call1((&a, &test)).unwrap();
        assert!(eqf.call1((&is_f, &is_n)).unwrap().extract::<bool>().unwrap(), "S isin mismatch");
        for op in ["intersect1d", "setdiff1d", "setxor1d"] {
            let f = module.getattr(op).unwrap().call1((&a, &b)).unwrap();
            let n = numpy.getattr(op).unwrap().call1((&a, &b)).unwrap();
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "S {op} mismatch");
        }
        let ss_ff = module.getattr("searchsorted").unwrap();
        let ss_nn = numpy.getattr("searchsorted").unwrap();
        let is_ff = module.getattr("isin").unwrap();
        let is_nn = numpy.getattr("isin").unwrap();
        let xr_ff = module.getattr("setxor1d").unwrap();
        let xr_nn = numpy.getattr("setxor1d").unwrap();
        group.bench_function("fnp_searchsorted_S8_2m", |bn| bn.iter(|| black_box(ss_ff.call1((&h, &q)).unwrap())));
        group.bench_function("numpy_searchsorted_S8_2m", |bn| bn.iter(|| black_box(ss_nn.call1((&h, &q)).unwrap())));
        group.bench_function("fnp_isin_S8_2m", |bn| bn.iter(|| black_box(is_ff.call1((&a, &test)).unwrap())));
        group.bench_function("numpy_isin_S8_2m", |bn| bn.iter(|| black_box(is_nn.call1((&a, &test)).unwrap())));
        group.bench_function("fnp_setxor1d_S8_2m", |bn| bn.iter(|| black_box(xr_ff.call1((&a, &b)).unwrap())));
        group.bench_function("numpy_setxor1d_S8_2m", |bn| bn.iter(|| black_box(xr_nn.call1((&a, &b)).unwrap())));

        // S16: the wide two-word-key byte pack (S9..16 previously fell to the memcmp/FNV routes).
        // Full byte range incl. embedded nulls (raw padded memcmp == numpy 'S' order).
        let setup16 = "import numpy as np\n\
rng = np.random.default_rng(4)\n\
a16 = rng.integers(0, 256, (1_000_000, 16), dtype=np.uint8).view('S16').reshape(-1)\n\
brand16 = rng.integers(0, 256, (500_000, 16), dtype=np.uint8).view('S16').reshape(-1)\n\
b16 = np.concatenate([a16[:500_000], brand16])\n";
        let ns16 = PyDict::new(py);
        py.run(std::ffi::CString::new(setup16).unwrap().as_c_str(), Some(&ns16), Some(&ns16))
            .expect("S16 setup");
        let a16 = ns16.get_item("a16").expect("a16");
        let b16 = ns16.get_item("b16").expect("b16");
        let ix_ff = module.getattr("intersect1d").unwrap();
        let ix_nn = numpy.getattr("intersect1d").unwrap();
        for op in ["intersect1d", "setxor1d"] {
            let f = module.getattr(op).unwrap().call1((&a16, &b16)).unwrap();
            let n = numpy.getattr(op).unwrap().call1((&a16, &b16)).unwrap();
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "S16 {op} mismatch");
        }
        group.bench_function("fnp_intersect1d_S16_1m", |bn| bn.iter(|| black_box(ix_ff.call1((&a16, &b16)).unwrap())));
        group.bench_function("numpy_intersect1d_S16_1m", |bn| bn.iter(|| black_box(ix_nn.call1((&a16, &b16)).unwrap())));
        group.bench_function("fnp_setxor1d_S16_1m", |bn| bn.iter(|| black_box(xr_ff.call1((&a16, &b16)).unwrap())));
        group.bench_function("numpy_setxor1d_S16_1m", |bn| bn.iter(|| black_box(xr_nn.call1((&a16, &b16)).unwrap())));
    });
    group.finish();
}

fn bench_complex_unique_boundary(c: &mut Criterion) {
    // np.unique on a flat complex128 array. numpy sorts lexicographically (re, im) with a generic
    // introsort then dedups — ~2.8s @2M. fnp views as f64 pairs, parallel-sorts, dedups — bit-exact.
    let mut group = c.benchmark_group("python_complex_unique_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        // re,im drawn from 0..1000 => ~1M distinct values over 2M elements (heavy dedup).
        let setup = "import numpy as np\n\
rng = np.random.default_rng(0)\n\
re = rng.integers(0, 1000, 2_000_000).astype(np.float64)\n\
im = rng.integers(0, 1000, 2_000_000).astype(np.float64)\n\
c = re + 1j * im\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let cc = ns.get_item("c").expect("c");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let f = fnp_u.call1((&cc,)).expect("fnp unique");
        let n = numpy_u.call1((&cc,)).expect("numpy unique");
        assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "complex unique mismatch");
        group.bench_function("fnp_unique_c128_2m", |bn| bn.iter(|| black_box(fnp_u.call1((&cc,)).unwrap())));
        group.bench_function("numpy_unique_c128_2m", |bn| bn.iter(|| black_box(numpy_u.call1((&cc,)).unwrap())));
    });
    group.finish();
}

fn bench_complex64_unique_boundary(c: &mut Criterion) {
    // np.unique on a flat complex64 array (f32-pair twin of the c128 path). numpy lexicographic
    // introsort + dedup single-threaded; fnp parallel-sorts f32 pairs + dedups — bit-exact.
    let mut group = c.benchmark_group("python_complex64_unique_boundary");
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
re = rng.integers(0, 1000, 2_000_000).astype(np.float32)\n\
im = rng.integers(0, 1000, 2_000_000).astype(np.float32)\n\
c = (re + 1j * im).astype(np.complex64)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let cc = ns.get_item("c").expect("c");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let f = fnp_u.call1((&cc,)).expect("fnp unique");
        let n = numpy_u.call1((&cc,)).expect("numpy unique");
        assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "complex64 unique mismatch");
        group.bench_function("fnp_unique_c64_2m", |bn| bn.iter(|| black_box(fnp_u.call1((&cc,)).unwrap())));
        group.bench_function("numpy_unique_c64_2m", |bn| bn.iter(|| black_box(numpy_u.call1((&cc,)).unwrap())));
    });
    group.finish();
}

fn bench_complex_searchsorted_boundary(c: &mut Criterion) {
    // np.searchsorted into a sorted complex128 haystack with complex128 queries. numpy's per-element
    // lexicographic (re,im) binary search is single-threaded (~1.07s @2M+2M); fnp does it in parallel.
    let mut group = c.benchmark_group("python_complex_searchsorted_boundary");
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
h = np.sort(rng.integers(0, 1000, 2_000_000).astype(np.float64) + 1j * rng.integers(0, 1000, 2_000_000).astype(np.float64))\n\
q = rng.integers(0, 1000, 2_000_000).astype(np.float64) + 1j * rng.integers(0, 1000, 2_000_000).astype(np.float64)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let h = ns.get_item("h").expect("h");
        let q = ns.get_item("q").expect("q");
        let fnp_ss = module.getattr("searchsorted").expect("fnp searchsorted");
        let numpy_ss = numpy.getattr("searchsorted").expect("numpy searchsorted");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for side in ["left", "right"] {
            let kw = PyDict::new(py);
            kw.set_item("side", side).unwrap();
            let f = fnp_ss.call((&h, &q), Some(&kw)).expect("fnp ss");
            let n = numpy_ss.call((&h, &q), Some(&kw)).expect("numpy ss");
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "complex searchsorted mismatch side={side}");
        }
        group.bench_function("fnp_searchsorted_c128_2m", |bn| bn.iter(|| black_box(fnp_ss.call1((&h, &q)).unwrap())));
        group.bench_function("numpy_searchsorted_c128_2m", |bn| bn.iter(|| black_box(numpy_ss.call1((&h, &q)).unwrap())));
    });
    group.finish();
}

fn bench_complex_isin_boundary(c: &mut Criterion) {
    // np.isin on complex128 arrays via a hashed 16-byte-pattern set. numpy sorts |elem|+|test|
    // (~529ms @2M); fnp does a parallel membership scan — bit-exact for finite non-(-0.0) values.
    let mut group = c.benchmark_group("python_complex_isin_boundary");
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
a = rng.integers(0, 1000, 2_000_000).astype(np.float64) + 1j * rng.integers(0, 1000, 2_000_000).astype(np.float64)\n\
trand = rng.integers(0, 1000, 100_000).astype(np.float64) + 1j * rng.integers(0, 1000, 100_000).astype(np.float64)\n\
test = np.concatenate([a[:100_000], trand])\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let test = ns.get_item("test").expect("test");
        let fnp_isin = module.getattr("isin").expect("fnp isin");
        let numpy_isin = numpy.getattr("isin").expect("numpy isin");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for inv in [false, true] {
            let kw = PyDict::new(py);
            kw.set_item("invert", inv).unwrap();
            let f = fnp_isin.call((&a, &test), Some(&kw)).expect("fnp isin");
            let n = numpy_isin.call((&a, &test), Some(&kw)).expect("numpy isin");
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "complex isin mismatch invert={inv}");
        }
        group.bench_function("fnp_isin_c128_2m", |bn| bn.iter(|| black_box(fnp_isin.call1((&a, &test)).unwrap())));
        group.bench_function("numpy_isin_c128_2m", |bn| bn.iter(|| black_box(numpy_isin.call1((&a, &test)).unwrap())));
    });
    group.finish();
}

fn bench_complex64_ops_boundary(c: &mut Criterion) {
    // complex64 searchsorted + isin (f32 twins of the c128 paths). numpy ~551ms / ~401ms @2M.
    let mut group = c.benchmark_group("python_complex64_ops_boundary");
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
h = np.sort((rng.integers(0,1000,2_000_000)+1j*rng.integers(0,1000,2_000_000)).astype(np.complex64))\n\
q = (rng.integers(0,1000,2_000_000)+1j*rng.integers(0,1000,2_000_000)).astype(np.complex64)\n\
a = (rng.integers(0,1000,2_000_000)+1j*rng.integers(0,1000,2_000_000)).astype(np.complex64)\n\
trand = (rng.integers(0,1000,100_000)+1j*rng.integers(0,1000,100_000)).astype(np.complex64)\n\
test = np.concatenate([a[:100_000], trand])\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let h = ns.get_item("h").unwrap();
        let q = ns.get_item("q").unwrap();
        let a = ns.get_item("a").unwrap();
        let test = ns.get_item("test").unwrap();
        let eqf = numpy.getattr("array_equal").unwrap();
        // searchsorted correctness (left+right) + isin (default+invert).
        for side in ["left", "right"] {
            let kw = PyDict::new(py); kw.set_item("side", side).unwrap();
            let f = module.getattr("searchsorted").unwrap().call((&h, &q), Some(&kw)).unwrap();
            let n = numpy.getattr("searchsorted").unwrap().call((&h, &q), Some(&kw)).unwrap();
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "c64 searchsorted mismatch side={side}");
        }
        for inv in [false, true] {
            let kw = PyDict::new(py); kw.set_item("invert", inv).unwrap();
            let f = module.getattr("isin").unwrap().call((&a, &test), Some(&kw)).unwrap();
            let n = numpy.getattr("isin").unwrap().call((&a, &test), Some(&kw)).unwrap();
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "c64 isin mismatch invert={inv}");
        }
        let ss_f = module.getattr("searchsorted").unwrap();
        let ss_n = numpy.getattr("searchsorted").unwrap();
        let is_f = module.getattr("isin").unwrap();
        let is_n = numpy.getattr("isin").unwrap();
        group.bench_function("fnp_searchsorted_c64_2m", |bn| bn.iter(|| black_box(ss_f.call1((&h, &q)).unwrap())));
        group.bench_function("numpy_searchsorted_c64_2m", |bn| bn.iter(|| black_box(ss_n.call1((&h, &q)).unwrap())));
        group.bench_function("fnp_isin_c64_2m", |bn| bn.iter(|| black_box(is_f.call1((&a, &test)).unwrap())));
        group.bench_function("numpy_isin_c64_2m", |bn| bn.iter(|| black_box(is_n.call1((&a, &test)).unwrap())));
    });
    group.finish();
}

fn bench_datetime_unique_boundary(c: &mut Criterion) {
    // np.unique on datetime64/timedelta64. numpy sorts via a generic comparison introsort (~665ms @2M);
    // fnp routes to an int64 sort+dedup of the ticks viewed back as the same M8/m8[unit] — bit-exact.
    let mut group = c.benchmark_group("python_datetime_unique_boundary");
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
dt = rng.integers(0, 100000, 2_000_000).astype('datetime64[s]')\n\
td = rng.integers(0, 100000, 2_000_000).astype('timedelta64[s]')\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let dt = ns.get_item("dt").expect("dt");
        let td = ns.get_item("td").expect("td");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for (arr, label) in [(&dt, "datetime64"), (&td, "timedelta64")] {
            let f = fnp_u.call1((arr,)).expect("fnp unique");
            let n = numpy_u.call1((arr,)).expect("numpy unique");
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "{label} unique mismatch");
        }
        group.bench_function("fnp_unique_datetime64_2m", |bn| bn.iter(|| black_box(fnp_u.call1((&dt,)).unwrap())));
        group.bench_function("numpy_unique_datetime64_2m", |bn| bn.iter(|| black_box(numpy_u.call1((&dt,)).unwrap())));
    });
    group.finish();
}

fn bench_unique_struct_int_boundary(c: &mut Criterion) {
    // np.unique(1-D structured all-int64) — record dedup. numpy sorts records via its slow field value-lex
    // comparator (~776ms @1M x 2 fields); fnp views as (n,nfields) int64 and routes to the int row-unique.
    let mut group = c.benchmark_group("python_unique_struct_int_boundary");
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
n = 1_000_000\n\
a = np.zeros(n, dtype=[('a','<i8'),('b','<i8'),('c','<i8')])\n\
a['a'] = rng.integers(-50, 50, n)\n\
a['b'] = rng.integers(-50, 50, n)\n\
a['c'] = rng.integers(-50, 50, n)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let f = fnp_u.call1((&a,)).expect("fnp unique");
        let n = numpy_u.call1((&a,)).expect("numpy unique");
        assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "unique struct mismatch");
        group.bench_function("fnp_unique_struct_3xi8_1m", |bn| bn.iter(|| black_box(fnp_u.call1((&a,)).unwrap())));
        group.bench_function("numpy_unique_struct_3xi8_1m", |bn| bn.iter(|| black_box(numpy_u.call1((&a,)).unwrap())));
    });
    group.finish();
}

fn bench_unique_struct_mixed_boundary(c: &mut Criterion) {
    // np.unique on a 1-D MIXED int+float structured (record) array. numpy sorts records via its slow void
    // comparator (~778ms @1M i8+f8); fnp value-lex sorts a memcmp-comparable byte-transform of the records.
    let mut group = c.benchmark_group("python_unique_struct_mixed_boundary");
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
dt = [('id','<i8'),('val','<f8'),('flag','<i4')]\n\
n = 1_000_000\n\
a = np.zeros(n, dtype=dt)\n\
a['id'] = rng.integers(-100000, 100000, n)\n\
a['val'] = rng.integers(-100000, 100000, n).astype(np.float64)\n\
a['flag'] = rng.integers(0, 100, n)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let f = fnp_u.call1((&a,)).expect("fnp unique");
        let n = numpy_u.call1((&a,)).expect("numpy unique");
        assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "unique struct mixed mismatch");
        group.bench_function("fnp_unique_struct_i8f8i4_1m", |bn| bn.iter(|| black_box(fnp_u.call1((&a,)).unwrap())));
        group.bench_function("numpy_unique_struct_i8f8i4_1m", |bn| bn.iter(|| black_box(numpy_u.call1((&a,)).unwrap())));
    });
    group.finish();
}

fn bench_unique_rows_datetime_boundary(c: &mut Criterion) {
    // np.unique(2-D datetime64, axis=0) plain + factorize. numpy sorts records via its slow void comparator;
    // fnp views int64 and routes to the int row-unique, viewing the result back as datetime64[unit].
    let mut group = c.benchmark_group("python_unique_rows_datetime_boundary");
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
base = rng.integers(0, 100000, (250_000, 3)).astype('datetime64[s]')\n\
a = np.concatenate([base, base])\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        // plain
        let kw = PyDict::new(py); kw.set_item("axis", 0_i64).unwrap();
        let f = fnp_u.call((&a,), Some(&kw)).expect("fnp unique");
        let n = numpy_u.call((&a,), Some(&kw)).expect("numpy unique");
        assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "datetime rows plain mismatch");
        // factorize
        let kwf = PyDict::new(py);
        kwf.set_item("axis", 0_i64).unwrap();
        kwf.set_item("return_index", true).unwrap();
        kwf.set_item("return_inverse", true).unwrap();
        kwf.set_item("return_counts", true).unwrap();
        let ft = fnp_u.call((&a,), Some(&kwf)).expect("fnp full").downcast_into::<pyo3::types::PyTuple>().unwrap();
        let nt = numpy_u.call((&a,), Some(&kwf)).expect("numpy full").downcast_into::<pyo3::types::PyTuple>().unwrap();
        for i in 0..4 {
            let eq: bool = eqf.call1((ft.get_item(i).unwrap(), nt.get_item(i).unwrap())).unwrap().extract().unwrap();
            assert!(eq, "datetime rows factorize element {i} mismatch");
        }
        group.bench_function("fnp_unique_rows_datetime_500kx3", |bn| {
            let kw = PyDict::new(py); kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_datetime_500kx3", |bn| {
            let kw = PyDict::new(py); kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_rows_f16_boundary(c: &mut Criterion) {
    // np.unique(2-D float16, axis=0) plain + factorize. numpy f16 rows (void comparator, no SIMD) ~507ms;
    // fnp widens exact to f32, routes to the f32 row-unique, narrows the unique rows back to f16.
    let mut group = c.benchmark_group("python_unique_rows_f16_boundary");
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
base = rng.integers(0, 50, (250_000, 4)).astype(np.float16)\n\
a = np.concatenate([base, base])\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let kw = PyDict::new(py); kw.set_item("axis", 0_i64).unwrap();
        let f = fnp_u.call((&a,), Some(&kw)).expect("fnp unique");
        let n = numpy_u.call((&a,), Some(&kw)).expect("numpy unique");
        assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "f16 rows plain mismatch");
        let kwf = PyDict::new(py);
        kwf.set_item("axis", 0_i64).unwrap();
        kwf.set_item("return_index", true).unwrap();
        kwf.set_item("return_inverse", true).unwrap();
        kwf.set_item("return_counts", true).unwrap();
        let ft = fnp_u.call((&a,), Some(&kwf)).expect("fnp full").downcast_into::<pyo3::types::PyTuple>().unwrap();
        let nt = numpy_u.call((&a,), Some(&kwf)).expect("numpy full").downcast_into::<pyo3::types::PyTuple>().unwrap();
        for i in 0..4 {
            let eq: bool = eqf.call1((ft.get_item(i).unwrap(), nt.get_item(i).unwrap())).unwrap().extract().unwrap();
            assert!(eq, "f16 rows factorize element {i} mismatch");
        }
        group.bench_function("fnp_unique_rows_f16_500kx4", |bn| {
            let kw = PyDict::new(py); kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_f16_500kx4", |bn| {
            let kw = PyDict::new(py); kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_struct_int_factorize_boundary(c: &mut Criterion) {
    // np.unique(1-D structured all-int64, return_index+inverse+counts) — record factorize.
    let mut group = c.benchmark_group("python_unique_struct_int_factorize_boundary");
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
n = 1_000_000\n\
a = np.zeros(n, dtype=[('a','<i8'),('b','<i8'),('c','<i8')])\n\
a['a'] = rng.integers(-30, 30, n)\n\
a['b'] = rng.integers(-30, 30, n)\n\
a['c'] = rng.integers(-30, 30, n)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let kwfull = PyDict::new(py);
        kwfull.set_item("return_index", true).unwrap();
        kwfull.set_item("return_inverse", true).unwrap();
        kwfull.set_item("return_counts", true).unwrap();
        let ft = fnp_u.call((&a,), Some(&kwfull)).expect("fnp full").downcast_into::<pyo3::types::PyTuple>().unwrap();
        let nt = numpy_u.call((&a,), Some(&kwfull)).expect("numpy full").downcast_into::<pyo3::types::PyTuple>().unwrap();
        for i in 0..4 {
            let eq: bool = eqf.call1((ft.get_item(i).unwrap(), nt.get_item(i).unwrap())).unwrap().extract().unwrap();
            assert!(eq, "struct factorize element {i} mismatch");
        }
        group.bench_function("fnp_unique_struct_factorize_3xi8_1m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_struct_factorize_3xi8_1m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_struct_mixed_factorize_boundary(c: &mut Criterion) {
    // np.unique(1-D MIXED int+float structured, return_index+inverse+counts) — record factorize. numpy void
    // comparator + inverse build (~1.24s @1M i8+f8); fnp byte-transform stable sort + group scatter.
    let mut group = c.benchmark_group("python_unique_struct_mixed_factorize_boundary");
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
dt = [('id','<i8'),('val','<f8')]\n\
a = np.zeros(1_000_000, dtype=dt); a['id'] = rng.integers(0, 10000, 1_000_000); a['val'] = rng.integers(0, 10000, 1_000_000).astype(np.float64)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let kwf = PyDict::new(py);
        kwf.set_item("return_index", true).unwrap();
        kwf.set_item("return_inverse", true).unwrap();
        kwf.set_item("return_counts", true).unwrap();
        let ft = fnp_u.call((&a,), Some(&kwf)).expect("fnp full").downcast_into::<pyo3::types::PyTuple>().unwrap();
        let nt = numpy_u.call((&a,), Some(&kwf)).expect("numpy full").downcast_into::<pyo3::types::PyTuple>().unwrap();
        for i in 0..4 {
            let eq: bool = eqf.call1((ft.get_item(i).unwrap(), nt.get_item(i).unwrap())).unwrap().extract().unwrap();
            assert!(eq, "mixed struct factorize element {i} mismatch");
        }
        group.bench_function("fnp_unique_struct_mixed_factorize_1m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_struct_mixed_factorize_1m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_lexsort_float_boundary(c: &mut Criterion) {
    // np.lexsort with multiple non-integral float64 keys. numpy runs a K-pass comparison sort (~880ms 2 keys /
    // ~1.5s 3 keys @2M); fnp byte-transforms the keys into one memcmp-comparable record and sorts once — bit-exact.
    let mut group = c.benchmark_group("python_lexsort_float_boundary");
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
k0 = rng.standard_normal(2_000_000)\n\
k1 = rng.standard_normal(2_000_000)\n\
k2 = rng.standard_normal(2_000_000)\n\
keys2 = (k0, k1)\n\
keys3 = (k0, k1, k2)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let keys2 = ns.get_item("keys2").expect("keys2");
        let keys3 = ns.get_item("keys3").expect("keys3");
        let fnp_lx = module.getattr("lexsort").expect("fnp lexsort");
        let numpy_lx = numpy.getattr("lexsort").expect("numpy lexsort");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for (keys, label) in [(&keys2, "2key"), (&keys3, "3key")] {
            let f = fnp_lx.call1((keys,)).expect("fnp lexsort");
            let n = numpy_lx.call1((keys,)).expect("numpy lexsort");
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "lexsort float {label} mismatch");
        }
        group.bench_function("fnp_lexsort_f64_2key_2m", |bn| bn.iter(|| black_box(fnp_lx.call1((&keys2,)).unwrap())));
        group.bench_function("numpy_lexsort_f64_2key_2m", |bn| bn.iter(|| black_box(numpy_lx.call1((&keys2,)).unwrap())));
        group.bench_function("fnp_lexsort_f64_3key_2m", |bn| bn.iter(|| black_box(fnp_lx.call1((&keys3,)).unwrap())));
        group.bench_function("numpy_lexsort_f64_3key_2m", |bn| bn.iter(|| black_box(numpy_lx.call1((&keys3,)).unwrap())));
    });
    group.finish();
}

fn bench_sort_struct_mixed_boundary(c: &mut Criterion) {
    // np.sort on a 1-D MIXED int+float structured (record) array. numpy sorts records by field value-lex; the
    // existing fnp struct-sort path routes through numpy.lexsort (slow K-pass for float ~627ms @1M); fnp
    // byte-transforms the records and sorts once (no dedup) — bit-exact. The argsort sibling returns the same
    // transformed-key permutation when records are distinct, and defers on ties to preserve numpy's unstable order.
    let mut group = c.benchmark_group("python_sort_struct_mixed_boundary");
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
dt = [('id','<i8'),('val','<f8')]\n\
a = np.zeros(1_000_000, dtype=dt); a['id'] = rng.integers(0, 10000, 1_000_000); a['val'] = rng.integers(0, 10000, 1_000_000).astype(np.float64)\n\
a_argsort = np.zeros(1_000_000, dtype=dt); a_argsort['id'] = rng.permutation(1_000_000); a_argsort['val'] = rng.standard_normal(1_000_000)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let a_argsort = ns.get_item("a_argsort").expect("a_argsort");
        let fnp_s = module.getattr("sort").expect("fnp sort");
        let numpy_s = numpy.getattr("sort").expect("numpy sort");
        let fnp_as = module.getattr("argsort").expect("fnp argsort");
        let numpy_as = numpy.getattr("argsort").expect("numpy argsort");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let f = fnp_s.call1((&a,)).expect("fnp sort");
        let n = numpy_s.call1((&a,)).expect("numpy sort");
        assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "sort mixed struct mismatch");
        let f_idx = fnp_as.call1((&a_argsort,)).expect("fnp argsort");
        let n_idx = numpy_as.call1((&a_argsort,)).expect("numpy argsort");
        assert!(
            eqf.call1((&f_idx, &n_idx)).unwrap().extract::<bool>().unwrap(),
            "argsort mixed struct mismatch"
        );
        group.bench_function("fnp_sort_struct_i8f8_1m", |bn| bn.iter(|| black_box(fnp_s.call1((&a,)).unwrap())));
        group.bench_function("numpy_sort_struct_i8f8_1m", |bn| bn.iter(|| black_box(numpy_s.call1((&a,)).unwrap())));
        group.bench_function("fnp_argsort_struct_i8f8_distinct_1m", |bn| {
            bn.iter(|| black_box(fnp_as.call1((&a_argsort,)).unwrap()))
        });
        group.bench_function("numpy_argsort_struct_i8f8_distinct_1m", |bn| {
            bn.iter(|| black_box(numpy_as.call1((&a_argsort,)).unwrap()))
        });
    });
    group.finish();
}

fn bench_argsort_numeric_stable_boundary(c: &mut Criterion) {
    // np.argsort(1-D int/float, kind='stable') on DENSE data (heavy repeats — the tied case the default-kind
    // paths defer on). numpy stable value sort (~0.9-1.2s @8M); fnp (value, orig-index) parallel sort — bit-exact.
    let mut group = c.benchmark_group("python_argsort_numeric_stable_boundary");
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
di = rng.integers(0, 1000, 8_000_000).astype(np.int64)\n\
df = rng.integers(0, 1000, 8_000_000).astype(np.float64)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let di = ns.get_item("di").expect("di");
        let df = ns.get_item("df").expect("df");
        let fnp_as = module.getattr("argsort").expect("fnp argsort");
        let numpy_as = numpy.getattr("argsort").expect("numpy argsort");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for (arr, label) in [(&di, "i64"), (&df, "f64")] {
            let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
            let f = fnp_as.call((arr,), Some(&kw)).expect("fnp argsort");
            let n = numpy_as.call((arr,), Some(&kw)).expect("numpy argsort");
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "argsort {label} dense stable mismatch");
        }
        group.bench_function("fnp_argsort_i64_dense_stable_8m", |bn| {
            let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&di,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_i64_dense_stable_8m", |bn| {
            let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(numpy_as.call((&di,), Some(&kw)).unwrap()));
        });
        group.bench_function("fnp_argsort_f64_dense_stable_8m", |bn| {
            let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&df,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_f64_dense_stable_8m", |bn| {
            let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(numpy_as.call((&df,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_argsort_lastaxis_stable_boundary(c: &mut Criterion) {
    // np.argsort(2-D int/float, axis=-1, kind='stable') on DENSE rows (heavy repeats — the per-lane tied case
    // the default-kind last-axis path defers on). numpy per-lane stable sort (~0.3-0.4s @8M); fnp per-lane
    // (value, in-lane index) parallel sort across lanes — bit-exact.
    let mut group = c.benchmark_group("python_argsort_lastaxis_stable_boundary");
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
mi = np.ascontiguousarray(rng.integers(0, 100, (4000, 2000)).astype(np.int64))\n\
mf = np.ascontiguousarray(rng.integers(0, 100, (4000, 2000)).astype(np.float64))\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let mi = ns.get_item("mi").expect("mi");
        let mf = ns.get_item("mf").expect("mf");
        let fnp_as = module.getattr("argsort").expect("fnp argsort");
        let numpy_as = numpy.getattr("argsort").expect("numpy argsort");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for (arr, label) in [(&mi, "i64"), (&mf, "f64")] {
            let kw = PyDict::new(py); kw.set_item("axis", -1).unwrap(); kw.set_item("kind", "stable").unwrap();
            let f = fnp_as.call((arr,), Some(&kw)).expect("fnp argsort");
            let n = numpy_as.call((arr,), Some(&kw)).expect("numpy argsort");
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "argsort 2D {label} axis=-1 stable mismatch");
        }
        group.bench_function("fnp_argsort_i64_2d_lastaxis_stable_8m", |bn| {
            let kw = PyDict::new(py); kw.set_item("axis", -1).unwrap(); kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&mi,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_i64_2d_lastaxis_stable_8m", |bn| {
            let kw = PyDict::new(py); kw.set_item("axis", -1).unwrap(); kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(numpy_as.call((&mi,), Some(&kw)).unwrap()));
        });
        group.bench_function("fnp_argsort_f64_2d_lastaxis_stable_8m", |bn| {
            let kw = PyDict::new(py); kw.set_item("axis", -1).unwrap(); kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&mf,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_f64_2d_lastaxis_stable_8m", |bn| {
            let kw = PyDict::new(py); kw.set_item("axis", -1).unwrap(); kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(numpy_as.call((&mf,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_argsort_radix_stable_boundary(c: &mut Criterion) {
    // np.argsort(1-D WIDE-range int, kind='stable') — the range my counting sort defers on. numpy comparison
    // sort (~2s @16M i64); fnp PARALLEL LSD RADIX of (key,index) pairs (gather-free multi-pass). Ranges chosen
    // > 1<<20 (forces radix) but with ties (exercises stability). Bit-exact vs numpy stable.
    let mut group = c.benchmark_group("python_argsort_radix_stable_boundary");
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
wi = rng.integers(0, 2**30, 16_000_000).astype(np.int64)\n\
wu = rng.integers(0, 2**52, 16_000_000).astype(np.uint64)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let wi = ns.get_item("wi").expect("wi");
        let wu = ns.get_item("wu").expect("wu");
        let fnp_as = module.getattr("argsort").expect("fnp argsort");
        let numpy_as = numpy.getattr("argsort").expect("numpy argsort");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for (arr, label) in [(&wi, "i64_2p30"), (&wu, "u64_2p52")] {
            let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
            let f = fnp_as.call((arr,), Some(&kw)).expect("fnp argsort");
            let n = numpy_as.call((arr,), Some(&kw)).expect("numpy argsort");
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "argsort wide {label} stable mismatch");
        }
        group.bench_function("fnp_argsort_i64_wide_stable_16m", |bn| {
            let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&wi,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_i64_wide_stable_16m", |bn| {
            let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(numpy_as.call((&wi,), Some(&kw)).unwrap()));
        });
        group.bench_function("fnp_argsort_u64_wide_stable_16m", |bn| {
            let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&wu,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_u64_wide_stable_16m", |bn| {
            let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(numpy_as.call((&wu,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_argsort_radix_float_boundary(c: &mut Criterion) {
    // np.argsort(1-D FLOAT, kind='stable') — currently the gather-bound comparison sort (~2.7s @16M f64). fnp
    // LINEARIZES IEEE floats into radix-sortable u64 keys (monotonic bit-transform) + parallel LSD radix.
    // standard_normal has many exact-tie duplicates in float32 -> exercises stability. Bit-exact vs numpy stable.
    let mut group = c.benchmark_group("python_argsort_radix_float_boundary");
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
f64 = rng.standard_normal(16_000_000)\n\
f64[rng.integers(0, 16_000_000, 1000)] *= -1.0\n\
f32 = rng.integers(-100000, 100000, 16_000_000).astype(np.float32) / 7.0\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let f64a = ns.get_item("f64").expect("f64");
        let f32a = ns.get_item("f32").expect("f32");
        let fnp_as = module.getattr("argsort").expect("fnp argsort");
        let numpy_as = numpy.getattr("argsort").expect("numpy argsort");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for (arr, label) in [(&f64a, "f64"), (&f32a, "f32")] {
            let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
            let f = fnp_as.call((arr,), Some(&kw)).expect("fnp argsort");
            let n = numpy_as.call((arr,), Some(&kw)).expect("numpy argsort");
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "argsort float {label} stable mismatch");
        }
        group.bench_function("fnp_argsort_f64_stable_16m", |bn| {
            let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&f64a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_f64_stable_16m", |bn| {
            let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(numpy_as.call((&f64a,), Some(&kw)).unwrap()));
        });
        group.bench_function("fnp_argsort_f32_stable_16m", |bn| {
            let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&f32a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_f32_stable_16m", |bn| {
            let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(numpy_as.call((&f32a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_argsort_default_int_radix_boundary(c: &mut Criterion) {
    // np.argsort(1-D int) with the DEFAULT kind (unstable introsort) — the most common argsort call. For DISTINCT
    // data the perm is unique so the gather-free parallel radix reproduces numpy's default order (~1.3s @16M);
    // the fnp comparison path it replaces is gather-bound. Also asserts a TIED array still matches (radix defers
    // -> existing path). Bit-exact vs numpy default argsort.
    let mut group = c.benchmark_group("python_argsort_default_int_radix_boundary");
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
di = rng.permutation(16_000_000).astype(np.int64)\n\
du = rng.permutation(16_000_000).astype(np.uint64)\n\
tied = rng.integers(0, 1000, 16_000_000).astype(np.int64)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let di = ns.get_item("di").expect("di");
        let du = ns.get_item("du").expect("du");
        let tied = ns.get_item("tied").expect("tied");
        let fnp_as = module.getattr("argsort").expect("fnp argsort");
        let numpy_as = numpy.getattr("argsort").expect("numpy argsort");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for (arr, label) in [(&di, "i64_distinct"), (&du, "u64_distinct"), (&tied, "i64_tied")] {
            let f = fnp_as.call1((arr,)).expect("fnp argsort");
            let n = numpy_as.call1((arr,)).expect("numpy argsort");
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "default argsort {label} mismatch");
        }
        group.bench_function("fnp_argsort_i64_distinct_default_16m", |bn| bn.iter(|| black_box(fnp_as.call1((&di,)).unwrap())));
        group.bench_function("numpy_argsort_i64_distinct_default_16m", |bn| bn.iter(|| black_box(numpy_as.call1((&di,)).unwrap())));
        group.bench_function("fnp_argsort_u64_distinct_default_16m", |bn| bn.iter(|| black_box(fnp_as.call1((&du,)).unwrap())));
        group.bench_function("numpy_argsort_u64_distinct_default_16m", |bn| bn.iter(|| black_box(numpy_as.call1((&du,)).unwrap())));
    });
    group.finish();
}

fn bench_argsort_default_float_radix_boundary(c: &mut Criterion) {
    // np.argsort(1-D float) with the DEFAULT kind (unstable introsort) — the common float argsort call. For
    // DISTINCT data the perm is unique so the gather-free radix on IEEE-linearized keys reproduces numpy's order
    // (~1.3s @16M); the fnp comparison path it replaces is gather-bound. Also asserts a TIED array still matches.
    let mut group = c.benchmark_group("python_argsort_default_float_radix_boundary");
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
f64d = rng.standard_normal(16_000_000)\n\
f32d = rng.permutation(16_000_000).astype(np.float32)\n\
tied = np.round(rng.standard_normal(16_000_000), 2)\n";  // f32d: ints < 2**24 exact -> distinct; tied: dense dups -> defers
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let f64d = ns.get_item("f64d").expect("f64d");
        let f32d = ns.get_item("f32d").expect("f32d");
        let tied = ns.get_item("tied").expect("tied");
        let fnp_as = module.getattr("argsort").expect("fnp argsort");
        let numpy_as = numpy.getattr("argsort").expect("numpy argsort");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for (arr, label) in [(&f64d, "f64_distinct"), (&f32d, "f32_distinct"), (&tied, "f64_tied")] {
            let f = fnp_as.call1((arr,)).expect("fnp argsort");
            let n = numpy_as.call1((arr,)).expect("numpy argsort");
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "default float argsort {label} mismatch");
        }
        group.bench_function("fnp_argsort_f64_distinct_default_16m", |bn| bn.iter(|| black_box(fnp_as.call1((&f64d,)).unwrap())));
        group.bench_function("numpy_argsort_f64_distinct_default_16m", |bn| bn.iter(|| black_box(numpy_as.call1((&f64d,)).unwrap())));
        group.bench_function("fnp_argsort_f32_distinct_default_16m", |bn| bn.iter(|| black_box(fnp_as.call1((&f32d,)).unwrap())));
        group.bench_function("numpy_argsort_f32_distinct_default_16m", |bn| bn.iter(|| black_box(numpy_as.call1((&f32d,)).unwrap())));
    });
    group.finish();
}

fn bench_argsort_datetime_radix_boundary(c: &mut Criterion) {
    // np.argsort(1-D datetime64) default + stable — int64-backed, so route the .view('int64') to the gather-free
    // int radix/counting instead of the gather-bound comparison sort (numpy ~2s @16M). NaT defers. Bit-exact.
    let mut group = c.benchmark_group("python_argsort_datetime_radix_boundary");
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
dt = rng.permutation(16_000_000).astype('datetime64[s]')\n\
dt_tied = rng.integers(0, 1000, 16_000_000).astype('datetime64[s]')\n";  // distinct + dense-tied
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let dt = ns.get_item("dt").expect("dt");
        let dt_tied = ns.get_item("dt_tied").expect("dt_tied");
        let fnp_as = module.getattr("argsort").expect("fnp argsort");
        let numpy_as = numpy.getattr("argsort").expect("numpy argsort");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        // default kind: distinct + tied (tied defers -> comparison path, still bit-exact)
        for (arr, label) in [(&dt, "distinct_default"), (&dt_tied, "tied_default")] {
            let f = fnp_as.call1((arr,)).expect("fnp argsort");
            let n = numpy_as.call1((arr,)).expect("numpy argsort");
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "datetime argsort {label} mismatch");
        }
        // stable kind on tied datetime (dense) -> counting/radix, bit-exact
        let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
        let fs = fnp_as.call((&dt_tied,), Some(&kw)).expect("fnp stable");
        let ns_ = numpy_as.call((&dt_tied,), Some(&kw)).expect("numpy stable");
        assert!(eqf.call1((&fs, &ns_)).unwrap().extract::<bool>().unwrap(), "datetime argsort stable tied mismatch");
        group.bench_function("fnp_argsort_datetime_distinct_default_16m", |bn| bn.iter(|| black_box(fnp_as.call1((&dt,)).unwrap())));
        group.bench_function("numpy_argsort_datetime_distinct_default_16m", |bn| bn.iter(|| black_box(numpy_as.call1((&dt,)).unwrap())));
        group.bench_function("fnp_argsort_datetime_tied_stable_16m", |bn| {
            let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&dt_tied,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_datetime_tied_stable_16m", |bn| {
            let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(numpy_as.call((&dt_tied,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_median_int_histogram_boundary(c: &mut Criterion) {
    // np.median of a bounded-range INTEGER array. numpy partitions (introselect + int->f64 copy); fnp's own int
    // path delegates (widen-to-f64 "never beats numpy"). fnp histogram order-statistics = parallel histogram +
    // prefix-sum + rank binary-search — returns the same value with NO sort/partition. Bit-exact (odd + even n).
    let mut group = c.benchmark_group("python_median_int_histogram_boundary");
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
even_i64 = rng.integers(0, 1000, 16_000_000).astype(np.int64)\n\
odd_i64  = rng.integers(-500, 500, 16_000_001).astype(np.int64)\n\
i16 = rng.integers(0, 30000, 16_000_000).astype(np.int16)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let fnp_m = module.getattr("median").expect("fnp median");
        let numpy_m = numpy.getattr("median").expect("numpy median");
        for name in ["even_i64", "odd_i64", "i16"] {
            let arr = ns.get_item(name).expect("arr");
            let f = fnp_m.call1((&arr,)).expect("fnp median");
            let n = numpy_m.call1((&arr,)).expect("numpy median");
            let eq: bool = numpy.getattr("equal").unwrap().call1((&f, &n)).unwrap().extract().unwrap();
            assert!(eq, "median {name} mismatch: fnp {:?} numpy {:?}", f, n);
        }
        let ev = ns.get_item("even_i64").expect("ev");
        let i16a = ns.get_item("i16").expect("i16a");
        group.bench_function("fnp_median_i64_dense_16m", |bn| bn.iter(|| black_box(fnp_m.call1((&ev,)).unwrap())));
        group.bench_function("numpy_median_i64_dense_16m", |bn| bn.iter(|| black_box(numpy_m.call1((&ev,)).unwrap())));
        group.bench_function("fnp_median_i16_dense_16m", |bn| bn.iter(|| black_box(fnp_m.call1((&i16a,)).unwrap())));
        group.bench_function("numpy_median_i16_dense_16m", |bn| bn.iter(|| black_box(numpy_m.call1((&i16a,)).unwrap())));
    });
    group.finish();
}

fn bench_int_percentile_quantile_histogram_boundary(c: &mut Criterion) {
    // np.percentile/quantile of bounded-range INTEGER arrays, scalar default-linear q. Same primitive as the
    // histogram median win: one parallel histogram, rank lookup for the two straddling order statistics, then
    // f64 interpolation. The benchmark asserts byte-exact scalar outputs against numpy before timing.
    let mut group = c.benchmark_group("python_int_percentile_quantile_histogram_boundary");
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
rng = np.random.default_rng(8)\n\
i64 = rng.integers(-500, 500, 16_000_000).astype(np.int64)\n\
u16 = rng.integers(0, 30000, 16_000_000).astype(np.uint16)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let fnp_percentile = module.getattr("percentile").expect("fnp percentile");
        let numpy_percentile = numpy.getattr("percentile").expect("numpy percentile");
        let fnp_quantile = module.getattr("quantile").expect("fnp quantile");
        let numpy_quantile = numpy.getattr("quantile").expect("numpy quantile");
        let i64_arr = ns.get_item("i64").expect("i64");
        let u16_arr = ns.get_item("u16").expect("u16");
        for (name, arr, p, q) in [("i64", &i64_arr, 12.5_f64, 0.125_f64), ("u16", &u16_arr, 75.0, 0.75)] {
            let fp = fnp_percentile.call1((arr, p)).expect("fnp percentile");
            let np = numpy_percentile.call1((arr, p)).expect("numpy percentile");
            let fq = fnp_quantile.call1((arr, q)).expect("fnp quantile");
            let nq = numpy_quantile.call1((arr, q)).expect("numpy quantile");
            let eq_p: bool = numpy.getattr("array_equal").unwrap().call1((&fp, &np)).unwrap().extract().unwrap();
            let eq_q: bool = numpy.getattr("array_equal").unwrap().call1((&fq, &nq)).unwrap().extract().unwrap();
            assert!(eq_p, "percentile {name} mismatch: fnp {:?} numpy {:?}", fp, np);
            assert!(eq_q, "quantile {name} mismatch: fnp {:?} numpy {:?}", fq, nq);
        }
        group.bench_function("fnp_percentile_i64_dense_16m_p12_5", |bn| {
            bn.iter(|| black_box(fnp_percentile.call1((&i64_arr, 12.5_f64)).unwrap()));
        });
        group.bench_function("numpy_percentile_i64_dense_16m_p12_5", |bn| {
            bn.iter(|| black_box(numpy_percentile.call1((&i64_arr, 12.5_f64)).unwrap()));
        });
        group.bench_function("fnp_quantile_u16_dense_16m_q75", |bn| {
            bn.iter(|| black_box(fnp_quantile.call1((&u16_arr, 0.75_f64)).unwrap()));
        });
        group.bench_function("numpy_quantile_u16_dense_16m_q75", |bn| {
            bn.iter(|| black_box(numpy_quantile.call1((&u16_arr, 0.75_f64)).unwrap()));
        });
    });
    group.finish();
}

fn bench_argsort_temporal_complex_stable_boundary(c: &mut Criterion) {
    // np.argsort(1-D datetime/complex, kind='stable') on DENSE data. The tie-stable order is
    // reproducible as (value, original-index), unlike default-kind argsort.
    let mut group = c.benchmark_group("python_argsort_temporal_complex_stable_boundary");
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
dt = rng.integers(0, 1000, 8_000_000).astype('datetime64[s]')\n\
cz = (rng.integers(0, 100, 8_000_000) + 1j*rng.integers(0, 100, 8_000_000)).astype(np.complex128)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns))
            .expect("setup");
        let dt = ns.get_item("dt").expect("dt");
        let cz = ns.get_item("cz").expect("cz");
        let fnp_as = module.getattr("argsort").expect("fnp argsort");
        let numpy_as = numpy.getattr("argsort").expect("numpy argsort");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for (arr, label) in [(&dt, "datetime"), (&cz, "c128")] {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            let f = fnp_as.call((arr,), Some(&kw)).expect("fnp argsort");
            let n = numpy_as.call((arr,), Some(&kw)).expect("numpy argsort");
            assert!(
                eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
                "argsort {label} dense stable mismatch"
            );
        }
        group.bench_function("fnp_argsort_datetime_dense_stable_8m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&dt,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_datetime_dense_stable_8m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(numpy_as.call((&dt,), Some(&kw)).unwrap()));
        });
        group.bench_function("fnp_argsort_c128_dense_stable_8m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&cz,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_c128_dense_stable_8m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(numpy_as.call((&cz,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_argsort_string_stable_boundary(c: &mut Criterion) {
    // np.argsort(1-D 'U'/'S', kind='stable'). numpy stable-sorts strings via its per-record codepoint
    // comparator (~1.3s @2M U6); fnp memcmp stable index-sort returns the permutation directly — bit-exact.
    let mut group = c.benchmark_group("python_argsort_string_stable_boundary");
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
u = rng.integers(97, 123, (2_000_000, 6), dtype=np.uint32).reshape(-1).view('U6')\n\
s = rng.integers(97, 123, (2_000_000, 6), dtype=np.uint8).reshape(-1).view('S6')\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let u = ns.get_item("u").expect("u");
        let s = ns.get_item("s").expect("s");
        let fnp_as = module.getattr("argsort").expect("fnp argsort");
        let numpy_as = numpy.getattr("argsort").expect("numpy argsort");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for (arr, label) in [(&u, "U6"), (&s, "S6")] {
            let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
            let f = fnp_as.call((arr,), Some(&kw)).expect("fnp argsort");
            let n = numpy_as.call((arr,), Some(&kw)).expect("numpy argsort");
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "argsort {label} stable mismatch");
        }
        group.bench_function("fnp_argsort_U6_stable_2m", |bn| {
            let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&u,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_U6_stable_2m", |bn| {
            let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(numpy_as.call((&u,), Some(&kw)).unwrap()));
        });
        group.bench_function("fnp_argsort_S6_stable_2m", |bn| {
            let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&s,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_S6_stable_2m", |bn| {
            let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(numpy_as.call((&s,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_argsort_struct_stable_boundary(c: &mut Criterion) {
    // np.argsort(1-D structured, kind='stable'). numpy stable-sorts records by field value-lex via its void
    // comparator (~3.4s @2M i8+f8); fnp byte-transforms records + stable index sort — bit-exact.
    let mut group = c.benchmark_group("python_argsort_struct_stable_boundary");
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
dt = [('id','<i8'),('val','<f8')]\n\
a = np.zeros(2_000_000, dtype=dt); a['id'] = rng.integers(0, 100000, 2_000_000); a['val'] = rng.integers(0, 100000, 2_000_000).astype(np.float64)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_as = module.getattr("argsort").expect("fnp argsort");
        let numpy_as = numpy.getattr("argsort").expect("numpy argsort");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
        let f = fnp_as.call((&a,), Some(&kw)).expect("fnp argsort");
        let n = numpy_as.call((&a,), Some(&kw)).expect("numpy argsort");
        assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "argsort struct stable mismatch");
        group.bench_function("fnp_argsort_struct_i8f8_2m", |bn| {
            let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_struct_i8f8_2m", |bn| {
            let kw = PyDict::new(py); kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(numpy_as.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_arrayapi_boundary(c: &mut Criterion) {
    // np.unique_counts / unique_all (numpy 2.x array-API). numpy delegates to its generic unique (~411ms
    // unique_counts / ~742ms unique_all @2M U8); fnp routes through its fast string unique — bit-exact.
    let mut group = c.benchmark_group("python_unique_arrayapi_boundary");
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
s = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let s = ns.get_item("s").expect("s");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        // correctness: compare each namedtuple field of unique_counts and unique_all
        for op in ["unique_counts", "unique_all"] {
            let fnp_fn = module.getattr(op).expect("fnp op");
            let np_fn = numpy.getattr(op).expect("numpy op");
            let f = fnp_fn.call1((&s,)).expect("fnp").downcast_into::<pyo3::types::PyTuple>().unwrap();
            let n = np_fn.call1((&s,)).expect("numpy").downcast_into::<pyo3::types::PyTuple>().unwrap();
            let nfields = if op == "unique_counts" { 2 } else { 4 };
            for i in 0..nfields {
                let eq: bool = eqf.call1((f.get_item(i).unwrap(), n.get_item(i).unwrap())).unwrap().extract().unwrap();
                assert!(eq, "{op} field {i} mismatch");
            }
        }
        let fnp_uc = module.getattr("unique_counts").unwrap();
        let np_uc = numpy.getattr("unique_counts").unwrap();
        let fnp_ua = module.getattr("unique_all").unwrap();
        let np_ua = numpy.getattr("unique_all").unwrap();
        group.bench_function("fnp_unique_counts_U8_2m", |bn| bn.iter(|| black_box(fnp_uc.call1((&s,)).unwrap())));
        group.bench_function("numpy_unique_counts_U8_2m", |bn| bn.iter(|| black_box(np_uc.call1((&s,)).unwrap())));
        group.bench_function("fnp_unique_all_U8_2m", |bn| bn.iter(|| black_box(fnp_ua.call1((&s,)).unwrap())));
        group.bench_function("numpy_unique_all_U8_2m", |bn| bn.iter(|| black_box(np_ua.call1((&s,)).unwrap())));
    });
    group.finish();
}

fn bench_isin_struct_boundary(c: &mut Criterion) {
    // np.isin(1-D structured, structured) record membership. numpy falls back to a serial sort of
    // |element|+|test| (~4s @1M+500k); fnp hashes the test record bytes + parallel lookup — bit-exact.
    let mut group = c.benchmark_group("python_isin_struct_boundary");
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
dt = [('a','<i8'),('b','<i8')]\n\
a = np.zeros(1_000_000, dtype=dt); a['a'] = rng.integers(0, 1000, 1_000_000); a['b'] = rng.integers(0, 1000, 1_000_000)\n\
b = np.zeros(500_000, dtype=dt); b['a'] = rng.integers(0, 1000, 500_000); b['b'] = rng.integers(0, 1000, 500_000)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let b = ns.get_item("b").expect("b");
        let fnp_isin = module.getattr("isin").expect("fnp isin");
        let numpy_isin = numpy.getattr("isin").expect("numpy isin");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let f = fnp_isin.call1((&a, &b)).expect("fnp isin");
        let n = numpy_isin.call1((&a, &b)).expect("numpy isin");
        assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "isin struct mismatch");
        group.bench_function("fnp_isin_struct_2xi8_1m_500k", |bn| bn.iter(|| black_box(fnp_isin.call1((&a, &b)).unwrap())));
        group.bench_function("numpy_isin_struct_2xi8_1m_500k", |bn| bn.iter(|| black_box(numpy_isin.call1((&a, &b)).unwrap())));
    });
    group.finish();
}

fn bench_isin_struct_float_boundary(c: &mut Criterion) {
    // np.isin on a MIXED int+float structured (record) array. numpy delegates to a serial sort (~1.3s
    // @1M+500k); fnp hashes the record bytes (finite float fields, -0.0/NaN defer) — bit-exact.
    let mut group = c.benchmark_group("python_isin_struct_float_boundary");
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
dt = [('id','<i8'),('val','<f8')]\n\
a = np.zeros(1_000_000, dtype=dt); a['id'] = rng.integers(0, 1000, 1_000_000); a['val'] = rng.integers(0, 1000, 1_000_000).astype(np.float64)\n\
b = np.zeros(500_000, dtype=dt); b['id'] = rng.integers(0, 1000, 500_000); b['val'] = rng.integers(0, 1000, 500_000).astype(np.float64)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let b = ns.get_item("b").expect("b");
        let fnp_isin = module.getattr("isin").expect("fnp isin");
        let numpy_isin = numpy.getattr("isin").expect("numpy isin");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let f = fnp_isin.call1((&a, &b)).expect("fnp isin");
        let n = numpy_isin.call1((&a, &b)).expect("numpy isin");
        assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "isin struct int+f8 mismatch");
        group.bench_function("fnp_isin_struct_i8f8_1m_500k", |bn| bn.iter(|| black_box(fnp_isin.call1((&a, &b)).unwrap())));
        group.bench_function("numpy_isin_struct_i8f8_1m_500k", |bn| bn.iter(|| black_box(numpy_isin.call1((&a, &b)).unwrap())));
    });
    group.finish();
}

fn bench_searchsorted_struct_boundary(c: &mut Criterion) {
    // np.searchsorted(sorted structured, structured queries) — record binary search. numpy uses its slow
    // per-record void comparator (~5.5s @2M+2M); fnp value-lex binary search via the int64 view — bit-exact.
    let mut group = c.benchmark_group("python_searchsorted_struct_boundary");
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
dt = [('a','<i8'),('b','<i8')]\n\
hay = np.zeros(2_000_000, dtype=dt); hay['a'] = rng.integers(0, 100000, 2_000_000); hay['b'] = rng.integers(0, 100000, 2_000_000); hay = np.sort(hay)\n\
q = np.zeros(2_000_000, dtype=dt); q['a'] = rng.integers(0, 100000, 2_000_000); q['b'] = rng.integers(0, 100000, 2_000_000)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let hay = ns.get_item("hay").expect("hay");
        let q = ns.get_item("q").expect("q");
        let fnp_ss = module.getattr("searchsorted").expect("fnp searchsorted");
        let numpy_ss = numpy.getattr("searchsorted").expect("numpy searchsorted");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for side in ["left", "right"] {
            let kw = PyDict::new(py); kw.set_item("side", side).unwrap();
            let f = fnp_ss.call((&hay, &q), Some(&kw)).expect("fnp searchsorted");
            let n = numpy_ss.call((&hay, &q), Some(&kw)).expect("numpy searchsorted");
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "searchsorted struct side={side} mismatch");
        }
        group.bench_function("fnp_searchsorted_struct_2xi8_2m_2m", |bn| bn.iter(|| black_box(fnp_ss.call1((&hay, &q)).unwrap())));
        group.bench_function("numpy_searchsorted_struct_2xi8_2m_2m", |bn| bn.iter(|| black_box(numpy_ss.call1((&hay, &q)).unwrap())));

        let setup_u64 = "import numpy as np\n\
rng = np.random.default_rng(1)\n\
dt = [('a','<u8'),('b','<u8')]\n\
n = 1_000_000\n\
base = np.arange(n, dtype=np.uint64)\n\
hay_u = np.zeros(n, dtype=dt)\n\
hay_u['a'] = base // np.uint64(1000)\n\
hay_u['b'] = base % np.uint64(1000)\n\
q_u = np.zeros(n, dtype=dt)\n\
qa = rng.integers(0, n, n, dtype=np.uint64)\n\
q_u['a'] = qa // np.uint64(1000)\n\
q_u['b'] = qa % np.uint64(1000)\n";
        py.run(std::ffi::CString::new(setup_u64).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("u64 setup");
        let hay_u = ns.get_item("hay_u").expect("hay_u");
        let q_u = ns.get_item("q_u").expect("q_u");
        for side in ["left", "right"] {
            let kw = PyDict::new(py); kw.set_item("side", side).unwrap();
            let f = fnp_ss.call((&hay_u, &q_u), Some(&kw)).expect("fnp u64 searchsorted");
            let n = numpy_ss.call((&hay_u, &q_u), Some(&kw)).expect("numpy u64 searchsorted");
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "u64 searchsorted struct side={side} mismatch");
        }
        group.bench_function("fnp_searchsorted_struct_2xu8_1m_1m", |bn| bn.iter(|| black_box(fnp_ss.call1((&hay_u, &q_u)).unwrap())));
        group.bench_function("numpy_searchsorted_struct_2xu8_1m_1m", |bn| bn.iter(|| black_box(numpy_ss.call1((&hay_u, &q_u)).unwrap())));
    });
    group.finish();
}

fn bench_searchsorted_struct_mixed_boundary(c: &mut Criterion) {
    // np.searchsorted(sorted MIXED int+float structured, queries). numpy void-comparator binary search (~10s
    // @2M+2M i8+f8); fnp = byte-transform record keys + parallel memcmp binary search — bit-exact both sides.
    let mut group = c.benchmark_group("python_searchsorted_struct_mixed_boundary");
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
dt = [('id','<i8'),('val','<f8')]\n\
hay = np.zeros(2_000_000, dtype=dt); hay['id'] = rng.integers(0, 100000, 2_000_000); hay['val'] = rng.integers(0, 100000, 2_000_000).astype(np.float64); hay = np.sort(hay)\n\
q = np.zeros(2_000_000, dtype=dt); q['id'] = rng.integers(0, 100000, 2_000_000); q['val'] = rng.integers(0, 100000, 2_000_000).astype(np.float64)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let hay = ns.get_item("hay").expect("hay");
        let q = ns.get_item("q").expect("q");
        let fnp_ss = module.getattr("searchsorted").expect("fnp searchsorted");
        let numpy_ss = numpy.getattr("searchsorted").expect("numpy searchsorted");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for side in ["left", "right"] {
            let kw = PyDict::new(py); kw.set_item("side", side).unwrap();
            let f = fnp_ss.call((&hay, &q), Some(&kw)).expect("fnp searchsorted");
            let n = numpy_ss.call((&hay, &q), Some(&kw)).expect("numpy searchsorted");
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "searchsorted mixed struct side={side} mismatch");
        }
        group.bench_function("fnp_searchsorted_struct_i8f8_2m_2m", |bn| bn.iter(|| black_box(fnp_ss.call1((&hay, &q)).unwrap())));
        group.bench_function("numpy_searchsorted_struct_i8f8_2m_2m", |bn| bn.iter(|| black_box(numpy_ss.call1((&hay, &q)).unwrap())));
    });
    group.finish();
}

fn bench_struct_setops_boundary(c: &mut Criterion) {
    // np.union1d / intersect1d / setdiff1d / setxor1d on 1-D structured records. numpy does 2-3 serial
    // per-record void sorts (~2.5-8s @1M+1M); fnp reuses struct-unique + a hashed record-set filter.
    let mut group = c.benchmark_group("python_struct_setops_boundary");
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
dt = [('a','<i8'),('b','<i8')]\n\
a = np.zeros(1_000_000, dtype=dt); a['a'] = rng.integers(0, 3000, 1_000_000); a['b'] = rng.integers(0, 3000, 1_000_000)\n\
b = np.zeros(1_000_000, dtype=dt); b['a'] = rng.integers(0, 3000, 1_000_000); b['b'] = rng.integers(0, 3000, 1_000_000)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let b = ns.get_item("b").expect("b");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for op in ["union1d", "intersect1d", "setdiff1d", "setxor1d"] {
            let fnp_fn = module.getattr(op).expect("fnp op");
            let np_fn = numpy.getattr(op).expect("numpy op");
            let f = fnp_fn.call1((&a, &b)).expect("fnp setop");
            let n = np_fn.call1((&a, &b)).expect("numpy setop");
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "{op} struct mismatch");
            group.bench_function(format!("fnp_{op}_struct_1m_1m"), |bn| bn.iter(|| black_box(fnp_fn.call1((&a, &b)).unwrap())));
            group.bench_function(format!("numpy_{op}_struct_1m_1m"), |bn| bn.iter(|| black_box(np_fn.call1((&a, &b)).unwrap())));
        }
    });
    group.finish();
}

fn bench_struct_mixed_setops_boundary(c: &mut Criterion) {
    // np.union1d/intersect1d/setdiff1d/setxor1d on MIXED int+float structured records. numpy void-comparator
    // sorts (~2-7s @1M+1M); fnp = byte-transform value-lex unique base + hashed record filter (float finite).
    let mut group = c.benchmark_group("python_struct_mixed_setops_boundary");
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
dt = [('id','<i8'),('val','<f8')]\n\
a = np.zeros(1_000_000, dtype=dt); a['id'] = rng.integers(0, 3000, 1_000_000); a['val'] = rng.integers(0, 3000, 1_000_000).astype(np.float64)\n\
b = np.zeros(1_000_000, dtype=dt); b['id'] = rng.integers(0, 3000, 1_000_000); b['val'] = rng.integers(0, 3000, 1_000_000).astype(np.float64)\n\
dt32 = [('id','<i4'),('val','<f4')]\n\
a32 = np.zeros(1_000_000, dtype=dt32); a32['id'] = rng.integers(0, 3000, 1_000_000, dtype=np.int32); a32['val'] = rng.integers(0, 3000, 1_000_000).astype(np.float32)\n\
b32 = np.zeros(1_000_000, dtype=dt32); b32['id'] = rng.integers(0, 3000, 1_000_000, dtype=np.int32); b32['val'] = rng.integers(0, 3000, 1_000_000).astype(np.float32)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let b = ns.get_item("b").expect("b");
        let a32 = ns.get_item("a32").expect("a32");
        let b32 = ns.get_item("b32").expect("b32");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for op in ["union1d", "intersect1d", "setdiff1d", "setxor1d"] {
            let fnp_fn = module.getattr(op).expect("fnp op");
            let np_fn = numpy.getattr(op).expect("numpy op");
            let f = fnp_fn.call1((&a, &b)).expect("fnp setop");
            let n = np_fn.call1((&a, &b)).expect("numpy setop");
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "{op} mixed struct mismatch");
            group.bench_function(format!("fnp_{op}_struct_i8f8_1m_1m"), |bn| bn.iter(|| black_box(fnp_fn.call1((&a, &b)).unwrap())));
            group.bench_function(format!("numpy_{op}_struct_i8f8_1m_1m"), |bn| bn.iter(|| black_box(np_fn.call1((&a, &b)).unwrap())));
            let f32 = fnp_fn.call1((&a32, &b32)).expect("fnp i4f4 setop");
            let n32 = np_fn.call1((&a32, &b32)).expect("numpy i4f4 setop");
            assert!(
                eqf.call1((&f32, &n32)).unwrap().extract::<bool>().unwrap(),
                "{op} mixed i4f4 struct mismatch"
            );
            group.bench_function(format!("fnp_{op}_struct_i4f4_1m_1m"), |bn| {
                bn.iter(|| black_box(fnp_fn.call1((&a32, &b32)).unwrap()))
            });
            group.bench_function(format!("numpy_{op}_struct_i4f4_1m_1m"), |bn| {
                bn.iter(|| black_box(np_fn.call1((&a32, &b32)).unwrap()))
            });
        }
    });
    group.finish();
}

fn bench_c128_setops_boundary(c: &mut Criterion) {
    // np.union1d / intersect1d / setdiff1d / setxor1d on complex128. numpy delegates to a serial sort of
    // |a|+|b| (~2.7-3.1s @2M+2M); fnp reuses c128-unique + a hashed 16-byte-record filter (finite only).
    let mut group = c.benchmark_group("python_c128_setops_boundary");
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
a = (rng.integers(0, 3000, 2_000_000) + 1j * rng.integers(0, 3000, 2_000_000)).astype(np.complex128)\n\
b = (rng.integers(0, 3000, 2_000_000) + 1j * rng.integers(0, 3000, 2_000_000)).astype(np.complex128)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let b = ns.get_item("b").expect("b");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        // setxor1d re-included: the old hash route was ~parity (1.22x); the dense-domain grid wins.
        for op in ["union1d", "intersect1d", "setdiff1d", "setxor1d"] {
            let fnp_fn = module.getattr(op).expect("fnp op");
            let np_fn = numpy.getattr(op).expect("numpy op");
            let f = fnp_fn.call1((&a, &b)).expect("fnp setop");
            let n = np_fn.call1((&a, &b)).expect("numpy setop");
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "{op} c128 mismatch");
            group.bench_function(format!("fnp_{op}_c128_2m_2m"), |bn| bn.iter(|| black_box(fnp_fn.call1((&a, &b)).unwrap())));
            group.bench_function(format!("numpy_{op}_c128_2m_2m"), |bn| bn.iter(|| black_box(np_fn.call1((&a, &b)).unwrap())));
        }
    });
    group.finish();
}

fn bench_datetime_setops_boundary(c: &mut Criterion) {
    // np.union1d / intersect1d / setdiff1d / setxor1d on datetime64. numpy delegates to a serial sort
    // (~1.2-2.0s @2M+2M); fnp views int64 and routes to the fast int64 set-op, viewing back — bit-exact.
    let mut group = c.benchmark_group("python_datetime_setops_boundary");
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
a = rng.integers(0, 3000, 2_000_000).astype('datetime64[s]')\n\
b = rng.integers(0, 3000, 2_000_000).astype('datetime64[s]')\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let b = ns.get_item("b").expect("b");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for op in ["union1d", "intersect1d", "setdiff1d", "setxor1d"] {
            let fnp_fn = module.getattr(op).expect("fnp op");
            let np_fn = numpy.getattr(op).expect("numpy op");
            let f = fnp_fn.call1((&a, &b)).expect("fnp setop");
            let n = np_fn.call1((&a, &b)).expect("numpy setop");
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "{op} datetime mismatch");
            group.bench_function(format!("fnp_{op}_datetime_2m_2m"), |bn| bn.iter(|| black_box(fnp_fn.call1((&a, &b)).unwrap())));
            group.bench_function(format!("numpy_{op}_datetime_2m_2m"), |bn| bn.iter(|| black_box(np_fn.call1((&a, &b)).unwrap())));
        }
    });
    group.finish();
}

fn bench_datetime_searchsorted_isin_boundary(c: &mut Criterion) {
    // np.searchsorted(sorted datetime, datetime q) + np.isin(datetime, datetime). numpy delegates both
    // (~867ms / ~341ms); fnp routes the int64 view to the fast int searchsorted / int isin — bit-exact.
    let mut group = c.benchmark_group("python_datetime_searchsorted_isin_boundary");
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
hay = np.sort(rng.integers(0, 10**9, 2_000_000).astype('datetime64[s]'))\n\
q = rng.integers(0, 10**9, 2_000_000).astype('datetime64[s]')\n\
ia = rng.integers(0, 100000, 2_000_000).astype('datetime64[s]')\n\
ib = rng.integers(0, 100000, 1_000_000).astype('datetime64[s]')\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let (hay, q, ia, ib) = (ns.get_item("hay").unwrap(), ns.get_item("q").unwrap(), ns.get_item("ia").unwrap(), ns.get_item("ib").unwrap());
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let fnp_ss = module.getattr("searchsorted").unwrap();
        let numpy_ss = numpy.getattr("searchsorted").unwrap();
        let fnp_isin = module.getattr("isin").unwrap();
        let numpy_isin = numpy.getattr("isin").unwrap();
        for side in ["left", "right"] {
            let kw = PyDict::new(py); kw.set_item("side", side).unwrap();
            let f = fnp_ss.call((&hay, &q), Some(&kw)).unwrap();
            let n = numpy_ss.call((&hay, &q), Some(&kw)).unwrap();
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "datetime searchsorted {side} mismatch");
        }
        let fi = fnp_isin.call1((&ia, &ib)).unwrap();
        let ni = numpy_isin.call1((&ia, &ib)).unwrap();
        assert!(eqf.call1((&fi, &ni)).unwrap().extract::<bool>().unwrap(), "datetime isin mismatch");
        group.bench_function("fnp_searchsorted_datetime_2m_2m", |bn| bn.iter(|| black_box(fnp_ss.call1((&hay, &q)).unwrap())));
        group.bench_function("numpy_searchsorted_datetime_2m_2m", |bn| bn.iter(|| black_box(numpy_ss.call1((&hay, &q)).unwrap())));
        group.bench_function("fnp_isin_datetime_2m_1m", |bn| bn.iter(|| black_box(fnp_isin.call1((&ia, &ib)).unwrap())));
        group.bench_function("numpy_isin_datetime_2m_1m", |bn| bn.iter(|| black_box(numpy_isin.call1((&ia, &ib)).unwrap())));
    });
    group.finish();
}

fn bench_f16_ops_boundary(c: &mut Criterion) {
    // np.unique / isin / searchsorted on float16. numpy has no f16 SIMD (converts per-element -> ~170-330ms);
    // fnp widens exact to f32 and routes to the fast f32 kernels — bit-exact.
    let mut group = c.benchmark_group("python_f16_ops_boundary");
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
u = (rng.integers(0, 2000, 4_000_000) / 7).astype(np.float16)\n\
hs = np.sort((rng.integers(0, 2000, 2_000_000) / 7).astype(np.float16))\n\
hq = (rng.integers(0, 2000, 2_000_000) / 7).astype(np.float16)\n\
ia = (rng.integers(0, 2000, 2_000_000) / 7).astype(np.float16)\n\
ib = (rng.integers(0, 2000, 1_000_000) / 7).astype(np.float16)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let (u, hs, hq, ia, ib) = (ns.get_item("u").unwrap(), ns.get_item("hs").unwrap(), ns.get_item("hq").unwrap(), ns.get_item("ia").unwrap(), ns.get_item("ib").unwrap());
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let (fu, nu_) = (module.getattr("unique").unwrap(), numpy.getattr("unique").unwrap());
        let (fss, nss) = (module.getattr("searchsorted").unwrap(), numpy.getattr("searchsorted").unwrap());
        let (fi, ni) = (module.getattr("isin").unwrap(), numpy.getattr("isin").unwrap());
        assert!(eqf.call1((fu.call1((&u,)).unwrap(), nu_.call1((&u,)).unwrap())).unwrap().extract::<bool>().unwrap(), "f16 unique mismatch");
        assert!(eqf.call1((fss.call1((&hs, &hq)).unwrap(), nss.call1((&hs, &hq)).unwrap())).unwrap().extract::<bool>().unwrap(), "f16 searchsorted mismatch");
        assert!(eqf.call1((fi.call1((&ia, &ib)).unwrap(), ni.call1((&ia, &ib)).unwrap())).unwrap().extract::<bool>().unwrap(), "f16 isin mismatch");
        group.bench_function("fnp_unique_f16_4m", |bn| bn.iter(|| black_box(fu.call1((&u,)).unwrap())));
        group.bench_function("numpy_unique_f16_4m", |bn| bn.iter(|| black_box(nu_.call1((&u,)).unwrap())));
        group.bench_function("fnp_searchsorted_f16_2m_2m", |bn| bn.iter(|| black_box(fss.call1((&hs, &hq)).unwrap())));
        group.bench_function("numpy_searchsorted_f16_2m_2m", |bn| bn.iter(|| black_box(nss.call1((&hs, &hq)).unwrap())));
        group.bench_function("fnp_isin_f16_2m_1m", |bn| bn.iter(|| black_box(fi.call1((&ia, &ib)).unwrap())));
        group.bench_function("numpy_isin_f16_2m_1m", |bn| bn.iter(|| black_box(ni.call1((&ia, &ib)).unwrap())));
    });
    group.finish();
}

fn bench_f16_setops_boundary(c: &mut Criterion) {
    // np.union1d/intersect1d/setdiff1d/setxor1d on float16. numpy f16 set-ops ~172ms (no SIMD); fnp widens
    // exact to f32 -> f32 set-op -> narrows back to f16. bit-exact.
    let mut group = c.benchmark_group("python_f16_setops_boundary");
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
a = (rng.integers(0, 3000, 2_000_000) / 7).astype(np.float16)\n\
b = (rng.integers(0, 3000, 2_000_000) / 7).astype(np.float16)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let b = ns.get_item("b").expect("b");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for op in ["union1d", "intersect1d", "setdiff1d", "setxor1d"] {
            let fnp_fn = module.getattr(op).expect("fnp op");
            let np_fn = numpy.getattr(op).expect("numpy op");
            let f = fnp_fn.call1((&a, &b)).expect("fnp setop");
            let n = np_fn.call1((&a, &b)).expect("numpy setop");
            assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "{op} f16 mismatch");
            group.bench_function(format!("fnp_{op}_f16_2m_2m"), |bn| bn.iter(|| black_box(fnp_fn.call1((&a, &b)).unwrap())));
            group.bench_function(format!("numpy_{op}_f16_2m_2m"), |bn| bn.iter(|| black_box(np_fn.call1((&a, &b)).unwrap())));
        }
    });
    group.finish();
}

fn bench_unique_rows_lexsort_boundary(c: &mut Criterion) {
    // np.unique(2-D large-range int64, axis=0). numpy sorts rows with its slow void comparator
    // (~570ms @500kx3); the packed-composite path can't pack this range, so fnp does a parallel
    // value-lex row sort+dedup — bit-exact.
    let mut group = c.benchmark_group("python_unique_rows_lexsort_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        // range 1e15 per column over 3 cols => composite overflows u64 -> lexsort path; base tiled => dups.
        let setup = "import numpy as np\n\
rng = np.random.default_rng(0)\n\
base = rng.integers(-10**15, 10**15, (250_000, 3)).astype(np.int64)\n\
a = np.concatenate([base, base])\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let kw = PyDict::new(py); kw.set_item("axis", 0_i64).unwrap();
        let f = fnp_u.call((&a,), Some(&kw)).expect("fnp unique");
        let n = numpy_u.call((&a,), Some(&kw)).expect("numpy unique");
        assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "unique rows lexsort mismatch");
        group.bench_function("fnp_unique_rows_i64_500kx3", |bn| {
            let kw = PyDict::new(py); kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_i64_500kx3", |bn| {
            let kw = PyDict::new(py); kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_rows_narrow_int_boundary(c: &mut Criterion) {
    // np.unique(2-D int32, axis=0). The packed-composite path cannot pack this large range, but
    // exact int64 widening lets fnp reuse the value-lex row-unique and cast back to int32.
    let mut group = c.benchmark_group("python_unique_rows_narrow_int_boundary");
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
base = rng.integers(-(1 << 30), 1 << 30, (250_000, 4), dtype=np.int32)\n\
a = np.concatenate([base, base])\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let kw = PyDict::new(py);
        kw.set_item("axis", 0_i64).unwrap();
        let f = fnp_u.call((&a,), Some(&kw)).expect("fnp unique");
        let n = numpy_u.call((&a,), Some(&kw)).expect("numpy unique");
        assert!(
            f.getattr("dtype")
                .unwrap()
                .eq(n.getattr("dtype").unwrap())
                .unwrap(),
            "unique rows narrow-int dtype mismatch"
        );
        assert!(
            eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
            "unique rows narrow-int value mismatch"
        );
        let kwfull = PyDict::new(py);
        kwfull.set_item("axis", 0_i64).unwrap();
        kwfull.set_item("return_index", true).unwrap();
        kwfull.set_item("return_inverse", true).unwrap();
        kwfull.set_item("return_counts", true).unwrap();
        let ft = fnp_u
            .call((&a,), Some(&kwfull))
            .expect("fnp unique narrow full")
            .cast_into::<pyo3::types::PyTuple>()
            .unwrap();
        let nt = numpy_u
            .call((&a,), Some(&kwfull))
            .expect("numpy unique narrow full")
            .cast_into::<pyo3::types::PyTuple>()
            .unwrap();
        assert!(
            ft.get_item(0)
                .unwrap()
                .getattr("dtype")
                .unwrap()
                .eq(nt.get_item(0).unwrap().getattr("dtype").unwrap())
                .unwrap(),
            "unique rows narrow-int factorize dtype mismatch"
        );
        for i in 0..4 {
            let eq: bool = eqf
                .call1((ft.get_item(i).unwrap(), nt.get_item(i).unwrap()))
                .unwrap()
                .extract()
                .unwrap();
            assert!(eq, "unique rows narrow-int factorize element {i} mismatch");
        }
        group.bench_function("fnp_unique_rows_i32_500kx4", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_i32_500kx4", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("fnp_unique_rows_i32_factorize_500kx4", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_i32_factorize_500kx4", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_rows_factorize_boundary(c: &mut Criterion) {
    // np.unique(2-D large-range int64, axis=0, return_index+inverse+counts) — the row factorize/group-by.
    // numpy sorts rows via its slow void comparator AND builds the inverse; fnp value-lex sorts + scatters.
    let mut group = c.benchmark_group("python_unique_rows_factorize_boundary");
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
base = rng.integers(-10**15, 10**15, (250_000, 3)).astype(np.int64)\n\
a = np.concatenate([base, base])\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let kwfull = PyDict::new(py);
        kwfull.set_item("axis", 0_i64).unwrap();
        kwfull.set_item("return_index", true).unwrap();
        kwfull.set_item("return_inverse", true).unwrap();
        kwfull.set_item("return_counts", true).unwrap();
        // Correctness: compare each of the 4 tuple elements.
        let ft = fnp_u.call((&a,), Some(&kwfull)).expect("fnp unique full").downcast_into::<pyo3::types::PyTuple>().unwrap();
        let nt = numpy_u.call((&a,), Some(&kwfull)).expect("numpy unique full").downcast_into::<pyo3::types::PyTuple>().unwrap();
        for i in 0..4 {
            let eq: bool = eqf.call1((ft.get_item(i).unwrap(), nt.get_item(i).unwrap())).unwrap().extract().unwrap();
            assert!(eq, "unique rows factorize element {i} mismatch");
        }
        group.bench_function("fnp_unique_rows_factorize_500kx3", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_factorize_500kx3", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_rows_f64_boundary(c: &mut Criterion) {
    // np.unique(2-D float64, axis=0). numpy sorts rows value-lexicographically via its slow void
    // comparator (~771ms @500kx4); fnp value-lex sorts+dedups (finite, no -0.0/NaN) — bit-exact.
    let mut group = c.benchmark_group("python_unique_rows_f64_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        // finite f64 rows (int-valued, no -0.0/NaN), base tiled => dups.
        let setup = "import numpy as np\n\
rng = np.random.default_rng(0)\n\
base = rng.integers(-100, 100, (250_000, 4)).astype(np.float64)\n\
a = np.concatenate([base, base])\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let kw = PyDict::new(py); kw.set_item("axis", 0_i64).unwrap();
        let f = fnp_u.call((&a,), Some(&kw)).expect("fnp unique");
        let n = numpy_u.call((&a,), Some(&kw)).expect("numpy unique");
        assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "unique rows f64 mismatch");
        group.bench_function("fnp_unique_rows_f64_500kx4", |bn| {
            let kw = PyDict::new(py); kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_f64_500kx4", |bn| {
            let kw = PyDict::new(py); kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_rows_f32_boundary(c: &mut Criterion) {
    // np.unique(2-D float32, axis=0). numpy value-lex void comparator (~729ms @500kx4); fnp f32 row-unique.
    let mut group = c.benchmark_group("python_unique_rows_f32_boundary");
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
base = rng.integers(-100, 100, (250_000, 4)).astype(np.float32)\n\
a = np.concatenate([base, base])\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let kw = PyDict::new(py); kw.set_item("axis", 0_i64).unwrap();
        let f = fnp_u.call((&a,), Some(&kw)).expect("fnp unique");
        let n = numpy_u.call((&a,), Some(&kw)).expect("numpy unique");
        assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "unique rows f32 mismatch");
        group.bench_function("fnp_unique_rows_f32_500kx4", |bn| {
            let kw = PyDict::new(py); kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_f32_500kx4", |bn| {
            let kw = PyDict::new(py); kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_rows_f64_factorize_boundary(c: &mut Criterion) {
    // np.unique(2-D f64, axis=0, return_index+inverse+counts) — f64 row factorize. numpy ~1393ms @500kx4.
    let mut group = c.benchmark_group("python_unique_rows_f64_factorize_boundary");
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
base = rng.integers(-100, 100, (250_000, 4)).astype(np.float64)\n\
a = np.concatenate([base, base])\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let kwfull = PyDict::new(py);
        kwfull.set_item("axis", 0_i64).unwrap();
        kwfull.set_item("return_index", true).unwrap();
        kwfull.set_item("return_inverse", true).unwrap();
        kwfull.set_item("return_counts", true).unwrap();
        let ft = fnp_u.call((&a,), Some(&kwfull)).expect("fnp full").downcast_into::<pyo3::types::PyTuple>().unwrap();
        let nt = numpy_u.call((&a,), Some(&kwfull)).expect("numpy full").downcast_into::<pyo3::types::PyTuple>().unwrap();
        for i in 0..4 {
            let eq: bool = eqf.call1((ft.get_item(i).unwrap(), nt.get_item(i).unwrap())).unwrap().extract().unwrap();
            assert!(eq, "f64 factorize element {i} mismatch");
        }
        group.bench_function("fnp_unique_rows_f64_factorize_500kx4", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_f64_factorize_500kx4", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_rows_f32_factorize_boundary(c: &mut Criterion) {
    // np.unique(2-D f32, axis=0, return_index+inverse+counts) — f32 row factorize.
    let mut group = c.benchmark_group("python_unique_rows_f32_factorize_boundary");
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
base = rng.integers(-100, 100, (250_000, 4)).astype(np.float32)\n\
a = np.concatenate([base, base])\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let kwfull = PyDict::new(py);
        kwfull.set_item("axis", 0_i64).unwrap();
        kwfull.set_item("return_index", true).unwrap();
        kwfull.set_item("return_inverse", true).unwrap();
        kwfull.set_item("return_counts", true).unwrap();
        let ft = fnp_u.call((&a,), Some(&kwfull)).expect("fnp full").downcast_into::<pyo3::types::PyTuple>().unwrap();
        let nt = numpy_u.call((&a,), Some(&kwfull)).expect("numpy full").downcast_into::<pyo3::types::PyTuple>().unwrap();
        for i in 0..4 {
            let eq: bool = eqf.call1((ft.get_item(i).unwrap(), nt.get_item(i).unwrap())).unwrap().extract().unwrap();
            assert!(eq, "f32 factorize element {i} mismatch");
        }
        group.bench_function("fnp_unique_rows_f32_factorize_500kx4", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_f32_factorize_500kx4", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_rows_c128_boundary(c: &mut Criterion) {
    // np.unique(2-D complex128, axis=0) — routed to the f64 row-unique via a .view. numpy ~336ms @500kx3.
    let mut group = c.benchmark_group("python_unique_rows_c128_boundary");
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
base = (rng.integers(0, 50, (250_000, 3)) + 1j * rng.integers(0, 50, (250_000, 3))).astype(np.complex128)\n\
a = np.concatenate([base, base])\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let kw = PyDict::new(py); kw.set_item("axis", 0_i64).unwrap();
        let f = fnp_u.call((&a,), Some(&kw)).expect("fnp unique");
        let n = numpy_u.call((&a,), Some(&kw)).expect("numpy unique");
        assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "unique rows c128 mismatch");
        group.bench_function("fnp_unique_rows_c128_500kx3", |bn| {
            let kw = PyDict::new(py); kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_c128_500kx3", |bn| {
            let kw = PyDict::new(py); kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_rows_c64_boundary(c: &mut Criterion) {
    // np.unique(2-D complex64, axis=0) plain + factorize. numpy c64 rows (void comparator) ~424ms; fnp views
    // as f32 (n, 2*ncols), routes to the f32 row-unique, views back c64.
    let mut group = c.benchmark_group("python_unique_rows_c64_boundary");
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
base = (rng.integers(0, 50, (250_000, 3)) + 1j * rng.integers(0, 50, (250_000, 3))).astype(np.complex64)\n\
a = np.concatenate([base, base])\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let kw = PyDict::new(py); kw.set_item("axis", 0_i64).unwrap();
        let f = fnp_u.call((&a,), Some(&kw)).expect("fnp unique");
        let n = numpy_u.call((&a,), Some(&kw)).expect("numpy unique");
        assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "c64 rows plain mismatch");
        let kwf = PyDict::new(py);
        kwf.set_item("axis", 0_i64).unwrap();
        kwf.set_item("return_index", true).unwrap();
        kwf.set_item("return_inverse", true).unwrap();
        kwf.set_item("return_counts", true).unwrap();
        let ft = fnp_u.call((&a,), Some(&kwf)).expect("fnp full").downcast_into::<pyo3::types::PyTuple>().unwrap();
        let nt = numpy_u.call((&a,), Some(&kwf)).expect("numpy full").downcast_into::<pyo3::types::PyTuple>().unwrap();
        for i in 0..4 {
            let eq: bool = eqf.call1((ft.get_item(i).unwrap(), nt.get_item(i).unwrap())).unwrap().extract().unwrap();
            assert!(eq, "c64 rows factorize element {i} mismatch");
        }
        group.bench_function("fnp_unique_rows_c64_500kx3", |bn| {
            let kw = PyDict::new(py); kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_c64_500kx3", |bn| {
            let kw = PyDict::new(py); kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_rows_c128_factorize_boundary(c: &mut Criterion) {
    // np.unique(2-D complex128, axis=0, return_index+inverse+counts) — routed to the f64 _full via .view.
    let mut group = c.benchmark_group("python_unique_rows_c128_factorize_boundary");
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
base = (rng.integers(0, 50, (250_000, 3)) + 1j * rng.integers(0, 50, (250_000, 3))).astype(np.complex128)\n\
a = np.concatenate([base, base])\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let kwfull = PyDict::new(py);
        kwfull.set_item("axis", 0_i64).unwrap();
        kwfull.set_item("return_index", true).unwrap();
        kwfull.set_item("return_inverse", true).unwrap();
        kwfull.set_item("return_counts", true).unwrap();
        let ft = fnp_u.call((&a,), Some(&kwfull)).expect("fnp full").downcast_into::<pyo3::types::PyTuple>().unwrap();
        let nt = numpy_u.call((&a,), Some(&kwfull)).expect("numpy full").downcast_into::<pyo3::types::PyTuple>().unwrap();
        for i in 0..4 {
            let eq: bool = eqf.call1((ft.get_item(i).unwrap(), nt.get_item(i).unwrap())).unwrap().extract().unwrap();
            assert!(eq, "c128 factorize element {i} mismatch");
        }
        group.bench_function("fnp_unique_rows_c128_factorize_500kx3", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_c128_factorize_500kx3", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_cols_axis1_boundary(c: &mut Criterion) {
    // np.unique(2-D wide i64, axis=1) — unique columns. numpy sorts columns via its slow void comparator
    // (~515ms @3x500k); fnp transposes to C-contig, reuses the row-unique, transposes back — bit-exact.
    let mut group = c.benchmark_group("python_unique_cols_axis1_boundary");
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
base = rng.integers(-10**15, 10**15, (3, 250_000)).astype(np.int64)\n\
a = np.concatenate([base, base], axis=1)\n";
        let ns = PyDict::new(py);
        py.run(std::ffi::CString::new(setup).unwrap().as_c_str(), Some(&ns), Some(&ns)).expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let kw = PyDict::new(py); kw.set_item("axis", 1_i64).unwrap();
        let f = fnp_u.call((&a,), Some(&kw)).expect("fnp unique");
        let n = numpy_u.call((&a,), Some(&kw)).expect("numpy unique");
        assert!(eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(), "unique cols axis=1 mismatch");
        group.bench_function("fnp_unique_cols_i64_3x500k", |bn| {
            let kw = PyDict::new(py); kw.set_item("axis", 1_i64).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_cols_i64_3x500k", |bn| {
            let kw = PyDict::new(py); kw.set_item("axis", 1_i64).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
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
x_k1000 = rng.integers(0, 1000, 4_000_000)\n\
x_u8 = rng.integers(0, 256, 4_000_000).astype(np.uint8)\n\
x_big = rng.integers(0, 512, 64_000_000)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("bincount setup");
        let x_mid = ns.get_item("x_mid").expect("x_mid");
        let x_k1000 = ns.get_item("x_k1000").expect("x_k1000");
        let x_u8 = ns.get_item("x_u8").expect("x_u8");
        let x_big = ns.get_item("x_big").expect("x_big");
        let fnp_bc = module.getattr("bincount").expect("fnp bincount");
        let numpy_bc = numpy.getattr("bincount").expect("numpy bincount");
        // k1000 = the i64-path case the vectorized max-scan fixed (0.5x -> ~1.1-1.6x); u8_4m = the
        // narrow-int path (numpy's narrow bincount is very slow, ~15x after the same max-scan fix).
        for (label, x) in [
            ("mid_2m_k256", &x_mid),
            ("mid_4m_k1000", &x_k1000),
            ("narrow_u8_4m", &x_u8),
            ("big_64m_k512", &x_big),
        ] {
            group.bench_function(format!("fnp_bincount_i64_{label}"), |b| {
                b.iter(|| black_box(fnp_bc.call1((x,)).expect("fnp bincount")));
            });
            group.bench_function(format!("numpy_bincount_i64_{label}"), |b| {
                b.iter(|| black_box(numpy_bc.call1((x,)).expect("numpy bincount")));
            });
        }
        // Weighted bincount (float64 accumulate): the vectorized max-scan moved it from parity to
        // ~1.13-1.22x (the sequential float tally still dominates, so the win is modest).
        py.run(
            std::ffi::CString::new("w_k1000 = rng.standard_normal(4_000_000)")
                .unwrap()
                .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("weighted setup");
        let w_k1000 = ns.get_item("w_k1000").expect("w_k1000");
        let kw = PyDict::new(py);
        kw.set_item("weights", &w_k1000).unwrap();
        let kw2 = kw.clone();
        group.bench_function("fnp_bincount_weighted_4m_k1000", |b| {
            b.iter(|| black_box(fnp_bc.call((&x_k1000,), Some(&kw)).expect("fnp wbincount")));
        });
        group.bench_function("numpy_bincount_weighted_4m_k1000", |b| {
            b.iter(|| black_box(numpy_bc.call((&x_k1000,), Some(&kw2)).expect("np wbincount")));
        });
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

        // Larger sorted haystack (4M, well past cache) so the sort-merge path is squarely in its
        // cache-miss win regime, plus a correctness gate: the merge output must be byte-identical to
        // numpy for both sides. Panics abort the bench, so a remote `cargo bench` doubles as the
        // conformance check on a glibc-matched worker.
        let big_setup = "import numpy as np\n\
rng2 = np.random.default_rng(1)\n\
a_big = np.sort(rng2.standard_normal(4_000_000))\n\
v_big = rng2.standard_normal(4_000_000)\n";
        let ns2 = PyDict::new(py);
        py.run(
            std::ffi::CString::new(big_setup).unwrap().as_c_str(),
            Some(&ns2),
            Some(&ns2),
        )
        .expect("searchsorted big setup");
        let a_big = ns2.get_item("a_big").expect("a_big");
        let v_big = ns2.get_item("v_big").expect("v_big");
        {
            let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
            for side in ["left", "right"] {
                let kw = PyDict::new(py);
                kw.set_item("side", side).expect("side");
                let got = fnp_ss.call((&a_big, &v_big), Some(&kw)).expect("fnp ss");
                let exp = numpy_ss.call((&a_big, &v_big), Some(&kw)).expect("np ss");
                let eq: bool = np_array_equal
                    .call1((&got, &exp))
                    .expect("array_equal")
                    .extract()
                    .expect("bool");
                assert!(eq, "searchsorted merge correctness mismatch: side={side}");
            }
        }

        group.bench_function("fnp_searchsorted_f64_4m", |b| {
            b.iter(|| black_box(fnp_ss.call1((&a, &v)).expect("fnp searchsorted")));
        });
        group.bench_function("numpy_searchsorted_f64_4m", |b| {
            b.iter(|| black_box(numpy_ss.call1((&a, &v)).expect("numpy searchsorted")));
        });
        group.bench_function("fnp_searchsorted_f64_4m_haystack4m", |b| {
            b.iter(|| black_box(fnp_ss.call1((&a_big, &v_big)).expect("fnp searchsorted big")));
        });
        group.bench_function("numpy_searchsorted_f64_4m_haystack4m", |b| {
            b.iter(|| black_box(numpy_ss.call1((&a_big, &v_big)).expect("numpy searchsorted big")));
        });

        // f32 twin: 4M sorted f32 haystack + 4M f32 queries. Correctness gate (both sides byte-identical
        // to numpy) + timing; the merge path engages for these large sizes.
        let f32_setup = "import numpy as np\n\
rng3 = np.random.default_rng(2)\n\
a_f32 = np.sort(rng3.standard_normal(4_000_000).astype(np.float32))\n\
v_f32 = rng3.standard_normal(4_000_000).astype(np.float32)\n";
        let ns3 = PyDict::new(py);
        py.run(
            std::ffi::CString::new(f32_setup).unwrap().as_c_str(),
            Some(&ns3),
            Some(&ns3),
        )
        .expect("searchsorted f32 setup");
        let a_f32 = ns3.get_item("a_f32").expect("a_f32");
        let v_f32 = ns3.get_item("v_f32").expect("v_f32");
        {
            let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
            for side in ["left", "right"] {
                let kw = PyDict::new(py);
                kw.set_item("side", side).expect("side");
                let got = fnp_ss.call((&a_f32, &v_f32), Some(&kw)).expect("fnp ss f32");
                let exp = numpy_ss.call((&a_f32, &v_f32), Some(&kw)).expect("np ss f32");
                let eq: bool = np_array_equal
                    .call1((&got, &exp))
                    .expect("array_equal")
                    .extract()
                    .expect("bool");
                assert!(eq, "searchsorted f32 merge correctness mismatch: side={side}");
            }
        }
        group.bench_function("fnp_searchsorted_f32_4m_haystack4m", |b| {
            b.iter(|| black_box(fnp_ss.call1((&a_f32, &v_f32)).expect("fnp searchsorted f32")));
        });
        group.bench_function("numpy_searchsorted_f32_4m_haystack4m", |b| {
            b.iter(|| black_box(numpy_ss.call1((&a_f32, &v_f32)).expect("numpy searchsorted f32")));
        });

        // i64 twin: integer ordering is total, so the same query-sort + monotonic merge should
        // be byte-identical while avoiding one random binary-search walk per query.
        let i64_setup = "import numpy as np\n\
rng4 = np.random.default_rng(3)\n\
a_i64 = np.sort(rng4.integers(-2_000_000_000, 2_000_000_000, 4_000_000, dtype=np.int64))\n\
v_i64 = rng4.integers(-2_500_000_000, 2_500_000_000, 4_000_000, dtype=np.int64)\n";
        let ns4 = PyDict::new(py);
        py.run(
            std::ffi::CString::new(i64_setup).unwrap().as_c_str(),
            Some(&ns4),
            Some(&ns4),
        )
        .expect("searchsorted i64 setup");
        let a_i64 = ns4.get_item("a_i64").expect("a_i64");
        let v_i64 = ns4.get_item("v_i64").expect("v_i64");
        {
            let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
            for side in ["left", "right"] {
                let kw = PyDict::new(py);
                kw.set_item("side", side).expect("side");
                let got = fnp_ss.call((&a_i64, &v_i64), Some(&kw)).expect("fnp ss i64");
                let exp = numpy_ss.call((&a_i64, &v_i64), Some(&kw)).expect("np ss i64");
                let eq: bool = np_array_equal
                    .call1((&got, &exp))
                    .expect("array_equal")
                    .extract()
                    .expect("bool");
                assert!(eq, "searchsorted i64 merge correctness mismatch: side={side}");
            }
        }
        group.bench_function("fnp_searchsorted_i64_4m_haystack4m", |b| {
            b.iter(|| black_box(fnp_ss.call1((&a_i64, &v_i64)).expect("fnp searchsorted i64")));
        });
        group.bench_function("numpy_searchsorted_i64_4m_haystack4m", |b| {
            b.iter(|| black_box(numpy_ss.call1((&a_i64, &v_i64)).expect("numpy searchsorted i64")));
        });
    });

    group.finish();
}

// np.repeat(2-D, scalar count, axis=1): the scalar-repeat native path used to gate to axis 0/None;
// generalized to ANY axis (leading units of `inner` trailing elems, each copied k times). ~2.4x.
fn bench_repeat_axis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_repeat_axis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
m = rng.standard_normal((512, 4096))\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("repeat axis setup");
        let m = ns.get_item("m").expect("m");
        let fnp_repeat = module.getattr("repeat").expect("fnp repeat");
        let numpy_repeat = numpy.getattr("repeat").expect("numpy repeat");
        let kw = PyDict::new(py);
        kw.set_item("axis", 1_i64).unwrap();
        let kw2 = kw.clone();
        group.bench_function("fnp_repeat_2d_axis1_c3", |b| {
            b.iter(|| black_box(fnp_repeat.call((&m, 3_i64), Some(&kw)).expect("fnp repeat")));
        });
        group.bench_function("numpy_repeat_2d_axis1_c3", |b| {
            b.iter(|| black_box(numpy_repeat.call((&m, 3_i64), Some(&kw2)).expect("np repeat")));
        });
    });

    group.finish();
}

// np.repeat(a, counts_array) with a per-element int64 count array: numpy expands it
// single-threaded (~104ms@4M in). The native prefix-sum + disjoint parallel scatter wins.
fn bench_repeat_array_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_repeat_array_boundary");
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
counts = rng.integers(1, 8, 4_000_000).astype(np.int64)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("repeat setup");
        let x = ns.get_item("x").expect("x");
        let counts = ns.get_item("counts").expect("counts");
        let fnp_repeat = module.getattr("repeat").expect("fnp repeat");
        let numpy_repeat = numpy.getattr("repeat").expect("numpy repeat");
        group.bench_function("fnp_repeat_array_f64_4m", |b| {
            b.iter(|| black_box(fnp_repeat.call1((&x, &counts)).expect("fnp repeat")));
        });
        group.bench_function("numpy_repeat_array_f64_4m", |b| {
            b.iter(|| black_box(numpy_repeat.call1((&x, &counts)).expect("np repeat")));
        });
    });

    group.finish();
}

// np.take(float32, ...): the native gather was gated to bool/8-byte, so f32/int32/f16/complex64 etc.
// delegated. Generalized to a value-agnostic uint8/16/32/64-view byte gather -> all 1/2/4/8-byte
// dtypes win (flat f32 ~12x, take-axis f32/int32/complex64 ~2.5-4.3x).
fn bench_take_dtype_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_take_dtype_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
xf = rng.standard_normal(1 << 22).astype(np.float32)\n\
idxf = rng.integers(0, 1 << 22, 1 << 22)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("take dtype setup");
        let xf = ns.get_item("xf").expect("xf");
        let idxf = ns.get_item("idxf").expect("idxf");
        let fnp_take = module.getattr("take").expect("fnp take");
        let numpy_take = numpy.getattr("take").expect("numpy take");
        group.bench_function("fnp_take_flat_f32", |b| {
            b.iter(|| black_box(fnp_take.call1((&xf, &idxf)).expect("fnp take")));
        });
        group.bench_function("numpy_take_flat_f32", |b| {
            b.iter(|| black_box(numpy_take.call1((&xf, &idxf)).expect("np take")));
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

// np.take_along_axis(complex64, ...): take_along_axis / put_along_axis deferred ALL complex; complex64
// (8-byte) now gathers via the same uint64 bit-view as every 8-byte dtype (complex128 still delegates).
fn bench_take_along_axis_c64_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_take_along_axis_c64_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
x = (rng.standard_normal((2048, 2048)) + 1j*rng.standard_normal((2048, 2048))).astype(np.complex64)\n\
idx = rng.integers(0, 2048, (2048, 2048))\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("take_along c64 setup");
        let x = ns.get_item("x").expect("x");
        let idx = ns.get_item("idx").expect("idx");
        let fnp_ta = module.getattr("take_along_axis").expect("fnp take_along_axis");
        let numpy_ta = numpy.getattr("take_along_axis").expect("numpy take_along_axis");
        group.bench_function("fnp_take_along_c64", |b| {
            b.iter(|| black_box(fnp_ta.call1((&x, &idx, 1_i64)).expect("fnp ta")));
        });
        group.bench_function("numpy_take_along_c64", |b| {
            b.iter(|| black_box(numpy_ta.call1((&x, &idx, 1_i64)).expect("np ta")));
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
        // f16 add/multiply/subtract: numpy widens to f32 (compute-bound, ~2.2x slower
        // than f32); native parallel widen->op->narrow wins.
        let h_setup = "import numpy as np\n\
rng = np.random.default_rng(3)\n\
ha = rng.standard_normal(16_000_000).astype(np.float16)\n\
hb = rng.standard_normal(16_000_000).astype(np.float16)\n";
        py.run(
            std::ffi::CString::new(h_setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("f16 binary setup");
        let ha = ns.get_item("ha").expect("ha");
        let hb = ns.get_item("hb").expect("hb");
        for op in ["add", "multiply", "maximum", "minimum", "greater", "less"] {
            let fnp_fn = module.getattr(op).expect("fnp op");
            let numpy_fn = numpy.getattr(op).expect("numpy op");
            group.bench_function(format!("fnp_{op}_f16_16m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&ha, &hb)).expect("fnp f16 call")));
            });
            group.bench_function(format!("numpy_{op}_f16_16m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&ha, &hb)).expect("numpy f16 call")));
            });
        }
        // f16 fmod/remainder: numpy widens f16->f32 (~214ms / ~317ms @16M, slowest f16 binary).
        // hb has near-zero entries; replace them so divisors are non-zero (kernel engages).
        py.run(
            std::ffi::CString::new("hbnz = np.where(np.abs(hb) < np.float16(0.05), np.float16(1.5), hb)")
                .unwrap()
                .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("f16 nonzero divisor setup");
        let hbnz = ns.get_item("hbnz").expect("hbnz");
        // f16 divide/floor_divide: numpy widens f16->f32->op->narrow single-threaded (~222/~375ms@16M).
        for op in ["fmod", "remainder", "divide", "floor_divide"] {
            let fnp_fn = module.getattr(op).expect("fnp op");
            let numpy_fn = numpy.getattr(op).expect("numpy op");
            group.bench_function(format!("fnp_{op}_f16_16m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&ha, &hbnz)).expect("fnp f16 call")));
            });
            group.bench_function(format!("numpy_{op}_f16_16m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&ha, &hbnz)).expect("numpy f16 call")));
            });
        }
        // f16 copysign/heaviside/nextafter: numpy widens f16->f32 (~22 / ~100 / ~57ms @16M).
        for op in ["copysign", "heaviside", "nextafter"] {
            let fnp_fn = module.getattr(op).expect("fnp op");
            let numpy_fn = numpy.getattr(op).expect("numpy op");
            group.bench_function(format!("fnp_{op}_f16_16m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&ha, &hb)).expect("fnp f16 call")));
            });
            group.bench_function(format!("numpy_{op}_f16_16m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&ha, &hb)).expect("numpy f16 call")));
            });
        }
        // f16 unary rounding ops floor/ceil/trunc/rint: numpy has no native f16 ALU and emulates
        // via widen->f32->op->narrow (compute-bound, ~77-126ms at 16M); native parallel wins ~15-30x.
        for op in ["floor", "ceil", "trunc", "rint", "isnan", "isfinite", "signbit"] {
            let fnp_fn = module.getattr(op).expect("fnp unary op");
            let numpy_fn = numpy.getattr(op).expect("numpy unary op");
            group.bench_function(format!("fnp_{op}_f16_16m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&ha,)).expect("fnp f16 unary call")));
            });
            group.bench_function(format!("numpy_{op}_f16_16m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&ha,)).expect("numpy f16 unary call")));
            });
        }
        // f16 sqrt/square: numpy widens (compute-bound). Native parallel widen wins for the
        // warning-free common case (sqrt of non-negatives, square of |x|<256).
        let hsq_setup = "import numpy as np\n\
rng = np.random.default_rng(8)\n\
hsq = (np.abs(rng.standard_normal(16_000_000)) * 10.0).astype(np.float16)\n";
        py.run(
            std::ffi::CString::new(hsq_setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("f16 sqrt/square setup");
        let hsq = ns.get_item("hsq").expect("hsq");
        for op in ["sqrt", "square"] {
            let fnp_fn = module.getattr(op).expect("fnp f16 unary op");
            let numpy_fn = numpy.getattr(op).expect("numpy f16 unary op");
            group.bench_function(format!("fnp_{op}_f16_16m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&hsq,)).expect("fnp f16 unary call")));
            });
            group.bench_function(format!("numpy_{op}_f16_16m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&hsq,)).expect("numpy f16 unary call")));
            });
        }
        // f16 reciprocal: numpy widens f16->f32, 1/x (IEEE divide -> bit-exact), narrows (~62ms@16M).
        // Native parallel widen wins; values >= 0.5 so no 1/x overflow (overflow inputs would defer).
        py.run(
            std::ffi::CString::new(
                "hrecip = (np.abs(rng.standard_normal(16_000_000)) * 5.0 + 0.5).astype(np.float16)",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("f16 reciprocal setup");
        let hrecip = ns.get_item("hrecip").expect("hrecip");
        {
            let fnp_recip = module.getattr("reciprocal").expect("fnp reciprocal");
            let numpy_recip = numpy.getattr("reciprocal").expect("numpy reciprocal");
            group.bench_function("fnp_reciprocal_f16_16m", |bch| {
                bch.iter(|| black_box(fnp_recip.call1((&hrecip,)).expect("fnp f16 reciprocal")));
            });
            group.bench_function("numpy_reciprocal_f16_16m", |bch| {
                bch.iter(|| black_box(numpy_recip.call1((&hrecip,)).expect("numpy f16 reciprocal")));
            });
        }
        // f16 modf: numpy widens f16->f32, splits into (trunc, frac), narrows both — single-threaded
        // scalar loop (~158ms@16M = ~0.6 GB/s, deeply compute-bound). Native parallel split wins.
        {
            let fnp_modf = module.getattr("modf").expect("fnp modf");
            let numpy_modf = numpy.getattr("modf").expect("numpy modf");
            group.bench_function("fnp_modf_f16_16m", |bch| {
                bch.iter(|| black_box(fnp_modf.call1((&hsq,)).expect("fnp f16 modf")));
            });
            group.bench_function("numpy_modf_f16_16m", |bch| {
                bch.iter(|| black_box(numpy_modf.call1((&hsq,)).expect("numpy f16 modf")));
            });
        }
        // f16 frexp: numpy widens f16->f32, decomposes (mantissa, exponent), narrows — single-threaded
        // scalar loop (~123ms@16M = ~0.8 GB/s, compute-bound). Native parallel bit-split wins.
        {
            let fnp_frexp = module.getattr("frexp").expect("fnp frexp");
            let numpy_frexp = numpy.getattr("frexp").expect("numpy frexp");
            group.bench_function("fnp_frexp_f16_16m", |bch| {
                bch.iter(|| black_box(fnp_frexp.call1((&hsq,)).expect("fnp f16 frexp")));
            });
            group.bench_function("numpy_frexp_f16_16m", |bch| {
                bch.iter(|| black_box(numpy_frexp.call1((&hsq,)).expect("numpy f16 frexp")));
            });
        }
        // f16 ldexp: numpy widens f16->f32, scalbnf, narrows — single-threaded (~108ms@16M = ~1.2
        // GB/s, compute-bound). Native parallel exact-pow2-scale wins. (i32 exponent in [-5,5).)
        py.run(
            std::ffi::CString::new(
                "lde = rng.integers(-5, 5, 16_000_000, dtype=np.int32)",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("f16 ldexp setup");
        let lde = ns.get_item("lde").expect("lde");
        {
            let fnp_ldexp = module.getattr("ldexp").expect("fnp ldexp");
            let numpy_ldexp = numpy.getattr("ldexp").expect("numpy ldexp");
            group.bench_function("fnp_ldexp_f16_16m", |bch| {
                bch.iter(|| black_box(fnp_ldexp.call1((&hsq, &lde)).expect("fnp f16 ldexp")));
            });
            group.bench_function("numpy_ldexp_f16_16m", |bch| {
                bch.iter(|| black_box(numpy_ldexp.call1((&hsq, &lde)).expect("numpy f16 ldexp")));
            });
        }
        // complex128/complex64 last-axis cumsum (4000x4000=16M): numpy's complex cumsum is a single-
        // threaded sequential dependency chain (~177ms@16M c128); native per-lane parallel scan wins.
        py.run(
            std::ffi::CString::new(
                "cc128 = (rng.standard_normal((4000,4000))+1j*rng.standard_normal((4000,4000))).astype(np.complex128); cc64 = cc128.astype(np.complex64)",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("complex cumsum setup");
        let cc128 = ns.get_item("cc128").expect("cc128");
        let cc64 = ns.get_item("cc64").expect("cc64");
        let kw_ax1 = PyDict::new(py);
        kw_ax1.set_item("axis", 1i64).expect("axis kw");
        let fnp_cumsum = module.getattr("cumsum").expect("fnp cumsum");
        let numpy_cumsum = numpy.getattr("cumsum").expect("numpy cumsum");
        for (arr, tag) in [(&cc128, "c128"), (&cc64, "c64")] {
            group.bench_function(format!("fnp_cumsum_lastaxis_{tag}_16m"), |bch| {
                bch.iter(|| {
                    black_box(fnp_cumsum.call((arr,), Some(&kw_ax1)).expect("fnp complex cumsum"))
                });
            });
            group.bench_function(format!("numpy_cumsum_lastaxis_{tag}_16m"), |bch| {
                bch.iter(|| {
                    black_box(numpy_cumsum.call((arr,), Some(&kw_ax1)).expect("numpy complex cumsum"))
                });
            });
        }
        // f16 clip: numpy widens f16->f32 to clamp (~149ms@16M, biggest f16 elementwise gap).
        {
            let fnp_clip = module.getattr("clip").expect("fnp clip");
            let numpy_clip = numpy.getattr("clip").expect("numpy clip");
            group.bench_function("fnp_clip_f16_16m", |bch| {
                bch.iter(|| black_box(fnp_clip.call1((&ha, -0.5f64, 0.5f64)).expect("fnp f16 clip")));
            });
            group.bench_function("numpy_clip_f16_16m", |bch| {
                bch.iter(|| {
                    black_box(numpy_clip.call1((&ha, -0.5f64, 0.5f64)).expect("numpy f16 clip"))
                });
            });
        }
        // f16 round (decimals=0 == rint): numpy widens f16->f32 (~120ms@16M).
        {
            let fnp_round = module.getattr("round").expect("fnp round");
            let numpy_round = numpy.getattr("round").expect("numpy round");
            group.bench_function("fnp_round_f16_16m", |bch| {
                bch.iter(|| black_box(fnp_round.call1((&ha,)).expect("fnp f16 round")));
            });
            group.bench_function("numpy_round_f16_16m", |bch| {
                bch.iter(|| black_box(numpy_round.call1((&ha,)).expect("numpy f16 round")));
            });
        }
        // f16 nan_to_num: numpy widens f16->f32 (~112ms@16M); native uint16 bit-replacement wins.
        {
            let fnp_n2n = module.getattr("nan_to_num").expect("fnp nan_to_num");
            let numpy_n2n = numpy.getattr("nan_to_num").expect("numpy nan_to_num");
            group.bench_function("fnp_nantonum_f16_16m", |bch| {
                bch.iter(|| black_box(fnp_n2n.call1((&ha,)).expect("fnp f16 nan_to_num")));
            });
            group.bench_function("numpy_nantonum_f16_16m", |bch| {
                bch.iter(|| black_box(numpy_n2n.call1((&ha,)).expect("numpy f16 nan_to_num")));
            });
        }
        // f16 flat min/max reduction: numpy widens f16->f32 to reduce (~80ms@16M); native
        // parallel f32-fold reduce wins (bit-exact, defers NaN / zero-extremum). hsq is all
        // non-negative with a non-zero max -> exercises the kernel.
        for op in ["max", "min", "ptp", "argmax", "argmin"] {
            let fnp_fn = module.getattr(op).expect("fnp reduce op");
            let numpy_fn = numpy.getattr(op).expect("numpy reduce op");
            group.bench_function(format!("fnp_{op}reduce_f16_16m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&hsq,)).expect("fnp f16 reduce call")));
            });
            group.bench_function(format!("numpy_{op}reduce_f16_16m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&hsq,)).expect("numpy f16 reduce call")));
            });
        }
        // f16 last-axis argmax/argmin: numpy widens f16->f32 per lane; native per-lane scan wins.
        py.run(
            std::ffi::CString::new("hsq2 = hsq.reshape(4000, 4000)").unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("f16 2-D reshape");
        let hsq2 = ns.get_item("hsq2").expect("hsq2");
        let kw_axis = PyDict::new(py);
        kw_axis.set_item("axis", -1i64).expect("axis kwarg");
        // f16 min/max along last-axis + axis-0 (4000x4000): numpy widens f16->f32 per lane
        // (~76ms@16M); native per-lane parallel f32-fold reduce wins (bit-exact, NaN/zero defer).
        let kw_axis0 = PyDict::new(py);
        kw_axis0.set_item("axis", 0i64).expect("axis0 kwarg");
        for op in ["max", "min"] {
            let fnp_fn = module.getattr(op).expect("fnp f16 minmax op");
            let numpy_fn = numpy.getattr(op).expect("numpy f16 minmax op");
            group.bench_function(format!("fnp_{op}_lastaxis_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(fnp_fn.call((&hsq2,), Some(&kw_axis)).expect("fnp f16 lastaxis minmax"))
                });
            });
            group.bench_function(format!("numpy_{op}_lastaxis_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        numpy_fn.call((&hsq2,), Some(&kw_axis)).expect("numpy f16 lastaxis minmax"),
                    )
                });
            });
            group.bench_function(format!("fnp_{op}_axis0_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(fnp_fn.call((&hsq2,), Some(&kw_axis0)).expect("fnp f16 axis0 minmax"))
                });
            });
            group.bench_function(format!("numpy_{op}_axis0_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        numpy_fn.call((&hsq2,), Some(&kw_axis0)).expect("numpy f16 axis0 minmax"),
                    )
                });
            });
        }
        // f16 ptp along last-axis + axis-0 (4000x4000): numpy widens f16->f32 for BOTH passes then
        // subtracts (the slowest f16 reduction); native per-lane max-min wins (bit-exact, NaN defer).
        let fnp_ptp = module.getattr("ptp").expect("fnp ptp");
        let numpy_ptp = numpy.getattr("ptp").expect("numpy ptp");
        group.bench_function("fnp_ptp_lastaxis_f16_16m", |bch| {
            bch.iter(|| {
                black_box(fnp_ptp.call((&hsq2,), Some(&kw_axis)).expect("fnp f16 lastaxis ptp"))
            });
        });
        group.bench_function("numpy_ptp_lastaxis_f16_16m", |bch| {
            bch.iter(|| {
                black_box(numpy_ptp.call((&hsq2,), Some(&kw_axis)).expect("numpy f16 lastaxis ptp"))
            });
        });
        group.bench_function("fnp_ptp_axis0_f16_16m", |bch| {
            bch.iter(|| {
                black_box(fnp_ptp.call((&hsq2,), Some(&kw_axis0)).expect("fnp f16 axis0 ptp"))
            });
        });
        group.bench_function("numpy_ptp_axis0_f16_16m", |bch| {
            bch.iter(|| {
                black_box(numpy_ptp.call((&hsq2,), Some(&kw_axis0)).expect("numpy f16 axis0 ptp"))
            });
        });
        // f16 nanmin/nanmax flat + last-axis + axis-0: numpy widens f16->f32 skip-NaN (~32ms@16M);
        // native uint16-view skip-NaN reduce wins. Sparse NaN at stride 997 (coprime to 4000) so no
        // lane is all-NaN and the kernel engages (all-NaN / zero-extremum lanes would defer).
        py.run(
            std::ffi::CString::new(
                "hsqn = hsq.copy(); hsqn[::997] = np.float16(np.nan); hsqn2 = hsqn.reshape(4000, 4000)",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("f16 nan array");
        let hsqn = ns.get_item("hsqn").expect("hsqn");
        let hsqn2 = ns.get_item("hsqn2").expect("hsqn2");
        for op in ["nanmin", "nanmax"] {
            let fnp_fn = module.getattr(op).expect("fnp f16 nan op");
            let numpy_fn = numpy.getattr(op).expect("numpy f16 nan op");
            group.bench_function(format!("fnp_{op}_flat_f16_16m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&hsqn,)).expect("fnp f16 flat nan")));
            });
            group.bench_function(format!("numpy_{op}_flat_f16_16m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&hsqn,)).expect("numpy f16 flat nan")));
            });
            group.bench_function(format!("fnp_{op}_lastaxis_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(fnp_fn.call((&hsqn2,), Some(&kw_axis)).expect("fnp f16 lastaxis nan"))
                });
            });
            group.bench_function(format!("numpy_{op}_lastaxis_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        numpy_fn.call((&hsqn2,), Some(&kw_axis)).expect("numpy f16 lastaxis nan"),
                    )
                });
            });
            group.bench_function(format!("fnp_{op}_axis0_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(fnp_fn.call((&hsqn2,), Some(&kw_axis0)).expect("fnp f16 axis0 nan"))
                });
            });
            group.bench_function(format!("numpy_{op}_axis0_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        numpy_fn.call((&hsqn2,), Some(&kw_axis0)).expect("numpy f16 axis0 nan"),
                    )
                });
            });
        }
        // f16 cumsum/cumprod along last-axis + axis-0: numpy widens f16->f32 per element + narrows
        // each step, all lanes single-threaded (~138/106ms@16M); native per-lane parallel scan wins.
        for op in ["cumsum", "cumprod"] {
            let fnp_fn = module.getattr(op).expect("fnp f16 cum op");
            let numpy_fn = numpy.getattr(op).expect("numpy f16 cum op");
            group.bench_function(format!("fnp_{op}_lastaxis_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(fnp_fn.call((&hsq2,), Some(&kw_axis)).expect("fnp f16 lastaxis cum"))
                });
            });
            group.bench_function(format!("numpy_{op}_lastaxis_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        numpy_fn.call((&hsq2,), Some(&kw_axis)).expect("numpy f16 lastaxis cum"),
                    )
                });
            });
            group.bench_function(format!("fnp_{op}_axis0_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(fnp_fn.call((&hsq2,), Some(&kw_axis0)).expect("fnp f16 axis0 cum"))
                });
            });
            group.bench_function(format!("numpy_{op}_axis0_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        numpy_fn.call((&hsq2,), Some(&kw_axis0)).expect("numpy f16 axis0 cum"),
                    )
                });
            });
        }
        // f16 nancumsum/nancumprod last-axis + axis-0 (4000x4000, sparse NaN -> identity): numpy
        // widens f16->f32 + nan-mask + narrows each step (~202/171ms); native per-lane scan wins.
        for op in ["nancumsum", "nancumprod"] {
            let fnp_fn = module.getattr(op).expect("fnp f16 nancum op");
            let numpy_fn = numpy.getattr(op).expect("numpy f16 nancum op");
            group.bench_function(format!("fnp_{op}_lastaxis_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        fnp_fn.call((&hsqn2,), Some(&kw_axis)).expect("fnp f16 lastaxis nancum"),
                    )
                });
            });
            group.bench_function(format!("numpy_{op}_lastaxis_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        numpy_fn.call((&hsqn2,), Some(&kw_axis)).expect("numpy f16 lastaxis nancum"),
                    )
                });
            });
            group.bench_function(format!("fnp_{op}_axis0_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        fnp_fn.call((&hsqn2,), Some(&kw_axis0)).expect("fnp f16 axis0 nancum"),
                    )
                });
            });
            group.bench_function(format!("numpy_{op}_axis0_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        numpy_fn.call((&hsqn2,), Some(&kw_axis0)).expect("numpy f16 axis0 nancum"),
                    )
                });
            });
        }
        // f64/int64 cumsum AXIS-0 (4000x4000=16M): numpy runs cumsum single-threaded for every dtype
        // (~166/163ms); the transpose column-parallel axis-0 path parallelizes the previously-serial
        // axis-0 scan (last-axis was already parallel).
        py.run(
            std::ffi::CString::new(
                "c64 = rng.standard_normal((4000, 4000)); ci64 = rng.integers(-1000, 1000, (4000, 4000)).astype(np.int64)",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("cumsum axis0 arrays");
        let c64 = ns.get_item("c64").expect("c64");
        let ci64 = ns.get_item("ci64").expect("ci64");
        let fnp_cumsum = module.getattr("cumsum").expect("fnp cumsum");
        let numpy_cumsum = numpy.getattr("cumsum").expect("numpy cumsum");
        for (arr, tag) in [(&c64, "f64"), (&ci64, "i64")] {
            group.bench_function(format!("fnp_cumsum_axis0_{tag}_16m"), |bch| {
                bch.iter(|| {
                    black_box(fnp_cumsum.call((arr,), Some(&kw_axis0)).expect("fnp cumsum axis0"))
                });
            });
            group.bench_function(format!("numpy_cumsum_axis0_{tag}_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        numpy_cumsum.call((arr,), Some(&kw_axis0)).expect("numpy cumsum axis0"),
                    )
                });
            });
        }
        for op in ["argmax", "argmin"] {
            let fnp_fn = module.getattr(op).expect("fnp arg op");
            let numpy_fn = numpy.getattr(op).expect("numpy arg op");
            group.bench_function(format!("fnp_{op}_lastaxis_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        fnp_fn
                            .call((&hsq2,), Some(&kw_axis))
                            .expect("fnp f16 lastaxis arg"),
                    )
                });
            });
            group.bench_function(format!("numpy_{op}_lastaxis_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        numpy_fn
                            .call((&hsq2,), Some(&kw_axis))
                            .expect("numpy f16 lastaxis arg"),
                    )
                });
            });
        }
        // f16 NON-last-axis (axis=0) argmax/argmin: numpy widens f16->f32 per column.
        let kw_axis0 = PyDict::new(py);
        kw_axis0.set_item("axis", 0i64).expect("axis0 kwarg");
        for op in ["argmax", "argmin"] {
            let fnp_fn = module.getattr(op).expect("fnp arg op");
            let numpy_fn = numpy.getattr(op).expect("numpy arg op");
            group.bench_function(format!("fnp_{op}_axis0_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(fnp_fn.call((&hsq2,), Some(&kw_axis0)).expect("fnp f16 axis0 arg"))
                });
            });
            group.bench_function(format!("numpy_{op}_axis0_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        numpy_fn.call((&hsq2,), Some(&kw_axis0)).expect("numpy f16 axis0 arg"),
                    )
                });
            });
        }
        // f32 fmod/copysign: numpy runs f32 binary ufuncs single-threaded (fmod ~138ms @16M);
        // there was no f32 binary zero-copy path. Native parallel f32 kernel wins (bit-exact).
        let f32_setup = "import numpy as np\n\
rng = np.random.default_rng(5)\n\
af = (rng.standard_normal(16_000_000) * 1e3).astype(np.float32)\n\
bf = (rng.standard_normal(16_000_000) * 7.0).astype(np.float32)\n\
bf[np.abs(bf) < 1e-3] = np.float32(1.5)\n";
        py.run(
            std::ffi::CString::new(f32_setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("f32 binary setup");
        let af = ns.get_item("af").expect("af");
        let bf = ns.get_item("bf").expect("bf");
        for op in ["fmod", "copysign", "remainder", "nextafter"] {
            let fnp_fn = module.getattr(op).expect("fnp f32 op");
            let numpy_fn = numpy.getattr(op).expect("numpy f32 op");
            group.bench_function(format!("fnp_{op}_f32_16m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&af, &bf)).expect("fnp f32 call")));
            });
            group.bench_function(format!("numpy_{op}_f32_16m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&af, &bf)).expect("numpy f32 call")));
            });
        }
        // integer gcd: numpy np.gcd is a single-threaded Euclid element loop (16M int64 ~995ms);
        // native parallel Euclid kernel wins big (bit-exact).
        let gcd_setup = "import numpy as np\n\
rng = np.random.default_rng(6)\n\
ag = rng.integers(1, 10**9, 16_000_000).astype(np.int64)\n\
cg = rng.integers(1, 10**9, 16_000_000).astype(np.int64)\n\
apw = rng.integers(-1000, 1000, 16_000_000).astype(np.int64)\n\
epw = rng.integers(0, 12, 16_000_000).astype(np.int64)\n";
        py.run(
            std::ffi::CString::new(gcd_setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("gcd setup");
        let ag = ns.get_item("ag").expect("ag");
        let cg = ns.get_item("cg").expect("cg");
        for op in ["gcd", "lcm"] {
            let fnp_fn = module.getattr(op).expect("fnp int op");
            let numpy_fn = numpy.getattr(op).expect("numpy int op");
            group.bench_function(format!("fnp_{op}_i64_16m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&ag, &cg)).expect("fnp int call")));
            });
            group.bench_function(format!("numpy_{op}_i64_16m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&ag, &cg)).expect("numpy int call")));
            });
        }
        // integer power: numpy a**b single-threaded element loop (16M int64 ~340ms); native
        // parallel wrapping repeated-squaring wins (bit-exact).
        let apw = ns.get_item("apw").expect("apw");
        let epw = ns.get_item("epw").expect("epw");
        let fnp_pow = module.getattr("power").expect("fnp power");
        let numpy_pow = numpy.getattr("power").expect("numpy power");
        group.bench_function("fnp_power_i64_16m", |bch| {
            bch.iter(|| black_box(fnp_pow.call1((&apw, &epw)).expect("fnp power call")));
        });
        group.bench_function("numpy_power_i64_16m", |bch| {
            bch.iter(|| black_box(numpy_pow.call1((&apw, &epw)).expect("numpy power call")));
        });
        // integer floor_divide / remainder / divmod: numpy single-threaded element loops
        // (16M int64 ~98ms / ~93ms / ~163ms).
        for op in ["floor_divide", "remainder", "divmod"] {
            let fnp_fn = module.getattr(op).expect("fnp int op");
            let numpy_fn = numpy.getattr(op).expect("numpy int op");
            group.bench_function(format!("fnp_{op}_i64_16m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&ag, &cg)).expect("fnp int call")));
            });
            group.bench_function(format!("numpy_{op}_i64_16m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&ag, &cg)).expect("numpy int call")));
            });
        }
        // timedelta64 // timedelta64 -> int64: numpy single-threaded w/ per-element NaT (~212ms@16M).
        py.run(
            std::ffi::CString::new(
                "atd = (np.abs(ag).astype('timedelta64[s]')); ctd = (np.where(cg==0,1,np.abs(cg)).astype('timedelta64[s]'))",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("timedelta setup");
        let atd = ns.get_item("atd").expect("atd");
        let ctd = ns.get_item("ctd").expect("ctd");
        for op in ["floor_divide", "remainder"] {
            let fnp_fn = module.getattr(op).expect("fnp td op");
            let numpy_fn = numpy.getattr(op).expect("numpy td op");
            group.bench_function(format!("fnp_{op}_td64_16m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&atd, &ctd)).expect("fnp td op")));
            });
            group.bench_function(format!("numpy_{op}_td64_16m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&atd, &ctd)).expect("numpy td op")));
            });
        }
        // f32 searchsorted: numpy single-threaded cold-cache binary search per query (~1.6s for
        // 8M queries into a 1M sorted f32). Native parallel per-query search wins big.
        py.run(
            std::ffi::CString::new(
                "ssa = np.sort(np.random.default_rng(9).standard_normal(1_000_000).astype(np.float32)); ssv = np.random.default_rng(10).standard_normal(8_000_000).astype(np.float32)",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("searchsorted setup");
        let ssa = ns.get_item("ssa").expect("ssa");
        let ssv = ns.get_item("ssv").expect("ssv");
        let fnp_ss = module.getattr("searchsorted").expect("fnp searchsorted");
        let numpy_ss = numpy.getattr("searchsorted").expect("numpy searchsorted");
        group.bench_function("fnp_searchsorted_f32_8m", |bch| {
            bch.iter(|| black_box(fnp_ss.call1((&ssa, &ssv)).expect("fnp searchsorted")));
        });
        group.bench_function("numpy_searchsorted_f32_8m", |bch| {
            bch.iter(|| black_box(numpy_ss.call1((&ssa, &ssv)).expect("numpy searchsorted")));
        });
        // f32 polyval (deg-11 Horner): numpy single-threaded (~570ms@16M).
        py.run(
            std::ffi::CString::new(
                "pvp = np.random.default_rng(11).standard_normal(12).astype(np.float32); pvx = np.random.default_rng(12).standard_normal(16_000_000).astype(np.float32)",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("polyval setup");
        let pvp = ns.get_item("pvp").expect("pvp");
        let pvx = ns.get_item("pvx").expect("pvx");
        let fnp_pv = module.getattr("polyval").expect("fnp polyval");
        let numpy_pv = numpy.getattr("polyval").expect("numpy polyval");
        group.bench_function("fnp_polyval_f32_16m", |bch| {
            bch.iter(|| black_box(fnp_pv.call1((&pvp, &pvx)).expect("fnp polyval")));
        });
        group.bench_function("numpy_polyval_f32_16m", |bch| {
            bch.iter(|| black_box(numpy_pv.call1((&pvp, &pvx)).expect("numpy polyval")));
        });
        // f32 ldexp: numpy scalbnf single-threaded (~86ms@16M).
        py.run(
            std::ffi::CString::new(
                "lxx = np.random.default_rng(13).standard_normal(16_000_000).astype(np.float32); lxe = np.random.default_rng(14).integers(-40,40,16_000_000).astype(np.int32)",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("ldexp setup");
        let lxx = ns.get_item("lxx").expect("lxx");
        let lxe = ns.get_item("lxe").expect("lxe");
        let fnp_le = module.getattr("ldexp").expect("fnp ldexp");
        let numpy_le = numpy.getattr("ldexp").expect("numpy ldexp");
        group.bench_function("fnp_ldexp_f32_16m", |bch| {
            bch.iter(|| black_box(fnp_le.call1((&lxx, &lxe)).expect("fnp ldexp")));
        });
        group.bench_function("numpy_ldexp_f32_16m", |bch| {
            bch.iter(|| black_box(numpy_le.call1((&lxx, &lxe)).expect("numpy ldexp")));
        });
        // f32 spacing: numpy single-threaded ULP (~64ms@16M).
        let fnp_sp = module.getattr("spacing").expect("fnp spacing");
        let numpy_sp = numpy.getattr("spacing").expect("numpy spacing");
        group.bench_function("fnp_spacing_f32_16m", |bch| {
            bch.iter(|| black_box(fnp_sp.call1((&lxx,)).expect("fnp spacing")));
        });
        group.bench_function("numpy_spacing_f32_16m", |bch| {
            bch.iter(|| black_box(numpy_sp.call1((&lxx,)).expect("numpy spacing")));
        });
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
        // INTEGER matmul: numpy has no BLAS for ints (naive loop, 89x slower than float
        // at 512). Native parallel ikj GEMM (bit-exact wrapping) should crush it.
        let int_setup = "import numpy as np\n\
rng = np.random.default_rng(7)\n\
ai512 = rng.integers(-100, 100, (512, 512)).astype(np.int64)\n\
bi512 = rng.integers(-100, 100, (512, 512)).astype(np.int64)\n\
ai1024 = rng.integers(-100, 100, (1024, 1024)).astype(np.int64)\n\
bi1024 = rng.integers(-100, 100, (1024, 1024)).astype(np.int64)\n";
        py.run(
            std::ffi::CString::new(int_setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("int matmul setup");
        let fnp_mm = module.getattr("matmul").expect("fnp matmul");
        let np_mm = numpy.getattr("matmul").expect("np matmul");
        for sz in ["512", "1024"] {
            let a = ns.get_item(format!("ai{sz}")).expect("ai");
            let b = ns.get_item(format!("bi{sz}")).expect("bi");
            group.bench_function(format!("fnp_matmul_i64_{sz}x{sz}"), |bch| {
                bch.iter(|| black_box(fnp_mm.call1((&a, &b)).expect("fnp int matmul")));
            });
            group.bench_function(format!("numpy_matmul_i64_{sz}x{sz}"), |bch| {
                bch.iter(|| black_box(np_mm.call1((&a, &b)).expect("numpy int matmul")));
            });
        }
        // BATCHED integer matmul (3-D): numpy int has no BLAS (naive per-slice serial).
        let bint_setup = "import numpy as np\n\
rng = np.random.default_rng(8)\n\
abi = rng.integers(-100, 100, (64, 128, 128)).astype(np.int64)\n\
bbi = rng.integers(-100, 100, (64, 128, 128)).astype(np.int64)\n";
        py.run(
            std::ffi::CString::new(bint_setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("batched int setup");
        {
            let a = ns.get_item("abi").expect("abi");
            let b = ns.get_item("bbi").expect("bbi");
            group.bench_function("fnp_matmul_i64_batched_64x128x128", |bch| {
                bch.iter(|| black_box(fnp_mm.call1((&a, &b)).expect("fnp int batched matmul")));
            });
            group.bench_function("numpy_matmul_i64_batched_64x128x128", |bch| {
                bch.iter(|| black_box(np_mm.call1((&a, &b)).expect("numpy int batched matmul")));
            });
        }
        // integer np.dot(2d,2d) routes to the same native GEMM (== matmul).
        let fnp_dot = module.getattr("dot").expect("fnp dot");
        let np_dot = numpy.getattr("dot").expect("np dot");
        {
            let a = ns.get_item("ai512").expect("ai512");
            let b = ns.get_item("bi512").expect("bi512");
            group.bench_function("fnp_dot_i64_512x512", |bch| {
                bch.iter(|| black_box(fnp_dot.call1((&a, &b)).expect("fnp int dot")));
            });
            group.bench_function("numpy_dot_i64_512x512", |bch| {
                bch.iter(|| black_box(np_dot.call1((&a, &b)).expect("numpy int dot")));
            });
            // integer np.inner(2d,2d) = a @ b^T routes to the native int GEMM.
            let fnp_inner = module.getattr("inner").expect("fnp inner");
            let np_inner = numpy.getattr("inner").expect("np inner");
            group.bench_function("fnp_inner_i64_512x512", |bch| {
                bch.iter(|| black_box(fnp_inner.call1((&a, &b)).expect("fnp int inner")));
            });
            group.bench_function("numpy_inner_i64_512x512", |bch| {
                bch.iter(|| black_box(np_inner.call1((&a, &b)).expect("numpy int inner")));
            });
        }
        // integer multi_dot chain (5 x 256x256): numpy no-BLAS chain vs native GEMM chain.
        let mdi_setup = "import numpy as np\n\
rng = np.random.default_rng(14)\n\
mdi = [rng.integers(-50, 50, (256, 256)).astype(np.int64) for _ in range(5)]\n";
        py.run(
            std::ffi::CString::new(mdi_setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("int multi_dot setup");
        {
            let mdi = ns.get_item("mdi").expect("mdi");
            let fnp_md = module.getattr("multi_dot").expect("fnp multi_dot");
            let np_md = numpy
                .getattr("linalg")
                .expect("linalg")
                .getattr("multi_dot")
                .expect("np multi_dot");
            group.bench_function("fnp_multi_dot_i64_5x256", |bch| {
                bch.iter(|| black_box(fnp_md.call1((&mdi,)).expect("fnp int multi_dot")));
            });
            group.bench_function("numpy_multi_dot_i64_5x256", |bch| {
                bch.iter(|| black_box(np_md.call1((&mdi,)).expect("numpy int multi_dot")));
            });
        }
        // INTEGER tensordot(axes=1) (64,64,64): numpy no-BLAS slow; routes to native int GEMM.
        let tdi_setup = "import numpy as np\n\
rng = np.random.default_rng(10)\n\
ati = rng.integers(-100, 100, (64, 64, 64)).astype(np.int64)\n\
bti = rng.integers(-100, 100, (64, 64, 64)).astype(np.int64)\n";
        py.run(
            std::ffi::CString::new(tdi_setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("int tensordot setup");
        {
            let fnp_td = module.getattr("tensordot").expect("fnp tensordot");
            let np_td = numpy.getattr("tensordot").expect("np tensordot");
            let a = ns.get_item("ati").expect("ati");
            let b = ns.get_item("bti").expect("bti");
            group.bench_function("fnp_tensordot_i64_axes1_64x64x64", |bch| {
                bch.iter(|| black_box(fnp_td.call1((&a, &b, 1_i64)).expect("fnp int tensordot")));
            });
            group.bench_function("numpy_tensordot_i64_axes1_64x64x64", |bch| {
                bch.iter(|| black_box(np_td.call1((&a, &b, 1_i64)).expect("numpy int tensordot")));
            });
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
        let Ok(fnp_char) = module.getattr("char") else {
            return;
        };
        let fnp_add = fnp_char.getattr("add").expect("fnp char.add");
        let Ok(numpy_char) = numpy.getattr("char") else {
            return;
        };
        let np_add = numpy_char.getattr("add").expect("np char.add");
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

// np.cumsum(timedelta64, axis): int64-backed; integer prefix sum is order-preserving (bit-exact),
// so the int64-view routes to the native int cumsum (~2.3x, result viewed back as timedelta64[unit]).
fn bench_timedelta_cumsum_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_timedelta_cumsum_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
td = rng.integers(-50000, 50000, (4096, 512, 8)).astype('timedelta64[D]')\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("timedelta cumsum setup");
        let td = ns.get_item("td").expect("td");
        let fnp_cumsum = module.getattr("cumsum").expect("fnp cumsum");
        let numpy_cumsum = numpy.getattr("cumsum").expect("numpy cumsum");
        let kw = PyDict::new(py);
        kw.set_item("axis", 1_i64).unwrap();
        let kw2 = kw.clone();
        group.bench_function("fnp_cumsum_td_mid", |b| {
            b.iter(|| black_box(fnp_cumsum.call((&td,), Some(&kw)).expect("fnp cumsum")));
        });
        group.bench_function("numpy_cumsum_td_mid", |b| {
            b.iter(|| black_box(numpy_cumsum.call((&td,), Some(&kw2)).expect("np cumsum")));
        });
    });

    group.finish();
}

// np.max/min(datetime64/timedelta64, axis): int64-backed; the int64-view routes to the native int
// min/max (~5-8x, result viewed back as the SAME temporal dtype). NaT pre-scan + defer.
fn bench_datetime_minmax_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_datetime_minmax_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
dt = rng.integers(0, 100000, (4096, 512, 8)).astype('datetime64[D]')\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("datetime minmax setup");
        let dt = ns.get_item("dt").expect("dt");
        for name in ["max", "min"] {
            let fnp_fn = module.getattr(name).expect("fnp fn");
            let numpy_fn = numpy.getattr(name).expect("numpy fn");
            let kw = PyDict::new(py);
            kw.set_item("axis", 1_i64).unwrap();
            let kw2 = kw.clone();
            group.bench_function(format!("fnp_{name}_dt_mid"), |b| {
                b.iter(|| black_box(fnp_fn.call((&dt,), Some(&kw)).expect("fnp mm")));
            });
            group.bench_function(format!("numpy_{name}_dt_mid"), |b| {
                b.iter(|| black_box(numpy_fn.call((&dt,), Some(&kw2)).expect("np mm")));
            });
        }
    });

    group.finish();
}

// np.ptp(datetime64/timedelta64, axis): int64-backed; numpy's temporal ptp is slow while the
// int64-view routes to the native int ptp (~6x, result viewed back as timedelta64[unit]).
fn bench_datetime_ptp_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_datetime_ptp_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
dt = rng.integers(0, 100000, (4096, 512, 8)).astype('datetime64[D]')\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("datetime ptp setup");
        let dt = ns.get_item("dt").expect("dt");
        let fnp_ptp = module.getattr("ptp").expect("fnp ptp");
        let numpy_ptp = numpy.getattr("ptp").expect("numpy ptp");
        let kw = PyDict::new(py);
        kw.set_item("axis", 1_i64).unwrap();
        let kw2 = kw.clone();
        group.bench_function("fnp_ptp_dt_mid", |b| {
            b.iter(|| black_box(fnp_ptp.call((&dt,), Some(&kw)).expect("fnp ptp")));
        });
        group.bench_function("numpy_ptp_dt_mid", |b| {
            b.iter(|| black_box(numpy_ptp.call((&dt,), Some(&kw2)).expect("np ptp")));
        });
    });

    group.finish();
}

// np.argmin/argmax(datetime64/timedelta64, axis): temporal reductions are int64-backed; numpy runs
// a slow temporal reduce while the int64-view routes to the fast native int argextreme (~6x after
// the NaT pre-scan). Bit-exact indices (int64 ordering == temporal ordering); NaT defers.
fn bench_datetime_argextreme_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_datetime_argextreme_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
dt = rng.integers(0, 100000, (4096, 512, 8)).astype('datetime64[D]')\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("datetime arg setup");
        let dt = ns.get_item("dt").expect("dt");
        for name in ["argmin", "argmax"] {
            let fnp_fn = module.getattr(name).expect("fnp fn");
            let numpy_fn = numpy.getattr(name).expect("numpy fn");
            let kw = PyDict::new(py);
            kw.set_item("axis", 1_i64).unwrap();
            let kw2 = kw.clone();
            group.bench_function(format!("fnp_{name}_dt_mid"), |b| {
                b.iter(|| black_box(fnp_fn.call((&dt,), Some(&kw)).expect("fnp arg")));
            });
            group.bench_function(format!("numpy_{name}_dt_mid"), |b| {
                b.iter(|| black_box(numpy_fn.call((&dt,), Some(&kw2)).expect("np arg")));
            });
        }
    });

    group.finish();
}

// np.argmin/argmax(f32, non-last axis): f32 had no arg-axis kernel (only f64+f16), so it delegated
// to numpy's slow strided reduce (~parity). The parallel per-block f32 scan wins ~8x.
fn bench_argextreme_f32_axis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_argextreme_f32_axis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
a = rng.standard_normal((512, 512, 32)).astype(np.float32)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("argextreme f32 setup");
        let a = ns.get_item("a").expect("a");
        // f64 + int64 of the same shape: their non-last kernels were parallelized (were serial).
        let a64 = a.call_method1("astype", ("float64",)).expect("a64");
        let ai = a.call_method1("astype", ("int64",)).expect("ai");
        for name in ["argmin", "argmax"] {
            let fnp_fn = module.getattr(name).expect("fnp fn");
            let numpy_fn = numpy.getattr(name).expect("numpy fn");
            let kw = PyDict::new(py);
            kw.set_item("axis", 1_i64).unwrap();
            for (tag, arr) in [("f32", &a), ("f64", &a64), ("i64", &ai)] {
                let kwc = kw.clone();
                group.bench_function(format!("fnp_{name}_{tag}_mid"), |b| {
                    b.iter(|| black_box(fnp_fn.call((arr,), Some(&kw)).expect("fnp arg")));
                });
                group.bench_function(format!("numpy_{name}_{tag}_mid"), |b| {
                    b.iter(|| black_box(numpy_fn.call((arr,), Some(&kwc)).expect("np arg")));
                });
            }
        }
    });

    group.finish();
}

// np.nanargmin/nanargmax(f32/f64, NON-last axis): no native non-last nanarg kernel existed, so it
// fell to the extract path (f64 ~2.3x, f32 a 0.77x LOSS from the f32->f64 widen). numpy's nanarg
// copies NaN->-+inf then argmins (whole-array temp). The native per-column nan-skip arg wins ~40-53x.
fn bench_nanarg_nonlast_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_nanarg_nonlast_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
a = rng.standard_normal((4096, 512, 8)).astype(np.float32)\n\
a[a > 2.0] = np.nan\na[:, 0, :] = 0.5\n\
a64 = a.astype(np.float64)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("nanarg nonlast setup");
        let a = ns.get_item("a").expect("a");
        let a64 = ns.get_item("a64").expect("a64");
        for name in ["nanargmin", "nanargmax"] {
            let fnp_fn = module.getattr(name).expect("fnp fn");
            let numpy_fn = numpy.getattr(name).expect("numpy fn");
            let kw = PyDict::new(py);
            kw.set_item("axis", 1_i64).unwrap();
            for (tag, arr) in [("f32", &a), ("f64", &a64)] {
                let kwc = kw.clone();
                group.bench_function(format!("fnp_{name}_{tag}_mid"), |b| {
                    b.iter(|| black_box(fnp_fn.call((arr,), Some(&kw)).expect("fnp nanarg")));
                });
                group.bench_function(format!("numpy_{name}_{tag}_mid"), |b| {
                    b.iter(|| black_box(numpy_fn.call((arr,), Some(&kwc)).expect("np nanarg")));
                });
            }
        }
    });

    group.finish();
}

// np.nanargmin/nanargmax(f64, axis=-1): numpy copies the array replacing NaN with +-inf then
// argmins (~107-144ms@16M); the native fused single-pass per-lane nan-skip scan wins 10-46x.
// f64 previously had no last-axis path (only f32); this closes the gap.
fn bench_nanarg_lastaxis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_nanarg_lastaxis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
w = rng.standard_normal((8, 2_000_000))\nw[w > 2.0] = np.nan\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("nanarg setup");
        let w = ns.get_item("w").expect("w");
        for name in ["nanargmin", "nanargmax"] {
            let fnp_fn = module.getattr(name).expect("fnp fn");
            let numpy_fn = numpy.getattr(name).expect("numpy fn");
            let kw = PyDict::new(py);
            kw.set_item("axis", 1_i64).unwrap();
            let kw2 = kw.clone();
            group.bench_function(format!("fnp_{name}_f64_8x2m"), |b| {
                b.iter(|| black_box(fnp_fn.call((&w,), Some(&kw)).expect("fnp nanarg")));
            });
            group.bench_function(format!("numpy_{name}_f64_8x2m"), |b| {
                b.iter(|| black_box(numpy_fn.call((&w,), Some(&kw2)).expect("np nanarg")));
            });
        }
    });

    group.finish();
}

// np.lexsort(3 small-range int keys, 2M): numpy runs K sequential radix sorts; the packed-composite
// path does one parallel sort. Correctness gate (byte-identical to numpy) + timing.
fn bench_lexsort_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_lexsort_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
k0 = rng.integers(0, 100, 2_000_000).astype(np.int64)\n\
k1 = rng.integers(0, 100, 2_000_000).astype(np.int32)\n\
k2 = rng.integers(0, 100, 2_000_000).astype(np.int16)\n\
keys = (k0, k1, k2)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("lexsort setup");
        let keys = ns.get_item("keys").expect("keys");
        py.run(
            std::ffi::CString::new(
                "keys_f64 = (k0.astype(np.float64), k1.astype(np.float64), k2.astype(np.float64))",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("lexsort float setup");
        let keys_f64 = ns.get_item("keys_f64").expect("keys_f64");
        let fnp_lexsort = module.getattr("lexsort").expect("fnp lexsort");
        let numpy_lexsort = numpy.getattr("lexsort").expect("numpy lexsort");
        {
            let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
            let got = fnp_lexsort.call1((&keys,)).expect("fnp lexsort");
            let exp = numpy_lexsort.call1((&keys,)).expect("np lexsort");
            let eq: bool = np_array_equal
                .call1((&got, &exp))
                .expect("array_equal")
                .extract()
                .expect("bool");
            assert!(eq, "lexsort composite correctness mismatch");
            let got_f64 = fnp_lexsort.call1((&keys_f64,)).expect("fnp lexsort f64");
            let exp_f64 = numpy_lexsort.call1((&keys_f64,)).expect("np lexsort f64");
            let eq_f64: bool = np_array_equal
                .call1((&got_f64, &exp_f64))
                .expect("array_equal f64")
                .extract()
                .expect("bool");
            assert!(
                eq_f64,
                "lexsort integral-f64 composite correctness mismatch"
            );
        }
        group.bench_function("fnp_lexsort_3int_2m", |b| {
            b.iter(|| black_box(fnp_lexsort.call1((&keys,)).expect("fnp lexsort")));
        });
        group.bench_function("numpy_lexsort_3int_2m", |b| {
            b.iter(|| black_box(numpy_lexsort.call1((&keys,)).expect("numpy lexsort")));
        });
        group.bench_function("fnp_lexsort_3f64_intvalued_2m", |b| {
            b.iter(|| black_box(fnp_lexsort.call1((&keys_f64,)).expect("fnp lexsort f64")));
        });
        group.bench_function("numpy_lexsort_3f64_intvalued_2m", |b| {
            b.iter(|| {
                black_box(
                    numpy_lexsort
                        .call1((&keys_f64,))
                        .expect("numpy lexsort f64"),
                )
            });
        });
    });

    group.finish();
}

// np.unique(500k x 4 small-range int, axis=0): numpy sorts rows via a slow void comparator; the
// composite-pack path does one u64 sort+dedup+decode. Correctness gate (byte-identical) + timing.
fn bench_unique_rows_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_unique_rows_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
A = rng.integers(0, 20, (500_000, 4)).astype(np.int64)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("unique rows setup");
        let a = ns.get_item("A").expect("A");
        let fnp_unique = module.getattr("unique").expect("fnp unique");
        let numpy_unique = numpy.getattr("unique").expect("numpy unique");
        {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0).expect("axis");
            let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
            let got = fnp_unique.call((&a,), Some(&kw)).expect("fnp unique");
            let exp = numpy_unique.call((&a,), Some(&kw)).expect("np unique");
            let eq: bool = np_array_equal
                .call1((&got, &exp))
                .expect("array_equal")
                .extract()
                .expect("bool");
            assert!(eq, "unique(axis=0) composite correctness mismatch");
        }
        group.bench_function("fnp_unique_rows_500k4_axis0", |b| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0).expect("axis");
            b.iter(|| black_box(fnp_unique.call((&a,), Some(&kw)).expect("fnp unique")));
        });
        group.bench_function("numpy_unique_rows_500k4_axis0", |b| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0).expect("axis");
            b.iter(|| black_box(numpy_unique.call((&a,), Some(&kw)).expect("numpy unique")));
        });
    });

    group.finish();
}

// np.unique(4 x 500k small-range int, axis=1): column-record sibling of the unique-rows composite
// pack. Correctness gate (byte-identical) + timing.
fn bench_unique_cols_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_unique_cols_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
A = rng.integers(0, 20, (4, 500_000)).astype(np.int64)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("unique cols setup");
        let a = ns.get_item("A").expect("A");
        let fnp_unique = module.getattr("unique").expect("fnp unique");
        let numpy_unique = numpy.getattr("unique").expect("numpy unique");
        {
            let kw = PyDict::new(py);
            kw.set_item("axis", 1).expect("axis");
            let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
            let got = fnp_unique.call((&a,), Some(&kw)).expect("fnp unique");
            let exp = numpy_unique.call((&a,), Some(&kw)).expect("np unique");
            let eq: bool = np_array_equal
                .call1((&got, &exp))
                .expect("array_equal")
                .extract()
                .expect("bool");
            assert!(eq, "unique(axis=1) composite correctness mismatch");
        }
        group.bench_function("fnp_unique_cols_4x500k_axis1", |b| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 1).expect("axis");
            b.iter(|| black_box(fnp_unique.call((&a,), Some(&kw)).expect("fnp unique")));
        });
        group.bench_function("numpy_unique_cols_4x500k_axis1", |b| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 1).expect("axis");
            b.iter(|| black_box(numpy_unique.call((&a,), Some(&kw)).expect("numpy unique")));
        });
    });

    group.finish();
}

// np.unique(500k x 4 int, axis=0, return_index/inverse/counts): the group-by/factorize primitive; numpy
// does the slow void-row sort plus the extra outputs. Correctness gate (all outputs byte-identical) + timing.
fn bench_unique_rows_full_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_unique_rows_full_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
A = rng.integers(0, 20, (500_000, 4)).astype(np.int64)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("unique rows full setup");
        let a = ns.get_item("A").expect("A");
        let fnp_unique = module.getattr("unique").expect("fnp unique");
        let numpy_unique = numpy.getattr("unique").expect("numpy unique");
        let kw = PyDict::new(py);
        kw.set_item("axis", 0).expect("axis");
        kw.set_item("return_index", true).expect("ri");
        kw.set_item("return_inverse", true).expect("rinv");
        kw.set_item("return_counts", true).expect("rc");
        {
            let got = fnp_unique.call((&a,), Some(&kw)).expect("fnp unique full");
            let exp = numpy_unique.call((&a,), Some(&kw)).expect("np unique full");
            let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
            // Compare each element of the returned tuple (unique, index, inverse, counts).
            for t in 0..4usize {
                let g = got.get_item(t).expect("got item");
                let e = exp.get_item(t).expect("exp item");
                let eq: bool = np_array_equal
                    .call1((&g, &e))
                    .expect("array_equal")
                    .extract()
                    .expect("bool");
                assert!(eq, "unique(axis=0) return_* correctness mismatch at tuple index {t}");
            }
        }
        group.bench_function("fnp_unique_rows_full_500k4", |b| {
            b.iter(|| black_box(fnp_unique.call((&a,), Some(&kw)).expect("fnp unique full")));
        });
        group.bench_function("numpy_unique_rows_full_500k4", |b| {
            b.iter(|| black_box(numpy_unique.call((&a,), Some(&kw)).expect("numpy unique full")));
        });
    });

    group.finish();
}

// Ledger-integrity retries for three historical REJECT rows. These helpers live only in the
// benchmark binary: production dispatch is deliberately untouched. `inline(never)` gives perf
// an exact execution marker for each reconstructed candidate and each NumPy ORIG reference.
#[inline]
fn ledger_f64_sortable_key(value: f64) -> u64 {
    let bits = if value == 0.0 { 0 } else { value.to_bits() };
    bits ^ ((((bits as i64) >> 63) as u64) | 0x8000_0000_0000_0000)
}

#[inline]
fn ledger_f64_from_sortable_key(key: u64) -> f64 {
    let bits = if key & 0x8000_0000_0000_0000 != 0 {
        key ^ 0x8000_0000_0000_0000
    } else {
        !key
    };
    f64::from_bits(bits)
}

#[inline(never)]
fn ledger_radix_select_key(mut current: Vec<u64>, mut rank: usize, start_byte: i32) -> u64 {
    let mut byte = start_byte;
    loop {
        let len = current.len();
        if len <= 1 || byte < 0 {
            return current[rank];
        }
        let shift = (byte as u64) * 8;
        let histogram: [usize; 256] = if len > (1 << 16) {
            let chunk_size = (len / (rayon::current_num_threads() * 4).max(1)).max(1);
            current
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut local = [0usize; 256];
                    for &key in chunk {
                        local[((key >> shift) & 0xff) as usize] += 1;
                    }
                    local
                })
                .reduce(
                    || [0usize; 256],
                    |mut left, right| {
                        for digit in 0..256 {
                            left[digit] += right[digit];
                        }
                        left
                    },
                )
        } else {
            let mut local = [0usize; 256];
            for &key in &current {
                local[((key >> shift) & 0xff) as usize] += 1;
            }
            local
        };
        let mut prefix = 0usize;
        let mut selected = 255usize;
        for (digit, &count) in histogram.iter().enumerate() {
            if prefix + count > rank {
                selected = digit;
                break;
            }
            prefix += count;
        }
        current = if len > (1 << 16) {
            current
                .par_iter()
                .copied()
                .filter(|&key| ((key >> shift) & 0xff) as usize == selected)
                .collect()
        } else {
            current
                .iter()
                .copied()
                .filter(|&key| ((key >> shift) & 0xff) as usize == selected)
                .collect()
        };
        rank -= prefix;
        byte -= 1;
    }
}

#[inline(never)]
fn ledger_radix_median_f64(data: &[f64]) -> f64 {
    assert!(!data.par_iter().any(|value| value.is_nan()));
    let keys: Vec<u64> = data
        .par_iter()
        .map(|&value| ledger_f64_sortable_key(value))
        .collect();
    let n = keys.len();
    if n % 2 == 1 {
        ledger_f64_from_sortable_key(ledger_radix_select_key(keys, n / 2, 7))
    } else {
        let low = ledger_f64_from_sortable_key(ledger_radix_select_key(keys.clone(), n / 2 - 1, 7));
        let high = ledger_f64_from_sortable_key(ledger_radix_select_key(keys, n / 2, 7));
        (low + high) / 2.0
    }
}

#[inline(never)]
fn ledger_orig_median_reference(
    numpy_median: &Bound<'_, PyAny>,
    input: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    numpy_median.call1((input,))?.extract()
}

#[inline(never)]
fn ledger_try_native_f16_sort(
    numpy_sort: &Bound<'_, PyAny>,
    input: &Bound<'_, PyAny>,
    input_bits: &[u16],
) -> PyResult<Py<PyAny>> {
    let must_defer = input_bits
        .par_iter()
        .any(|&bits| bits == 0x8000 || ((bits & 0x7c00) == 0x7c00 && (bits & 0x03ff) != 0));
    assert!(
        !must_defer,
        "finite positive f16 audit input must stay on candidate route"
    );
    let widened = input.call_method1("astype", ("float32",))?;
    let sorted = numpy_sort.call1((&widened,))?;
    Ok(sorted.call_method1("astype", ("float16",))?.unbind())
}

#[inline(never)]
fn ledger_orig_f16_sort_reference(
    numpy_sort: &Bound<'_, PyAny>,
    input: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    Ok(numpy_sort.call1((input,))?.unbind())
}

#[inline(never)]
fn ledger_f32_tie_argsort_candidate(
    fnp_argsort: &Bound<'_, PyAny>,
    input: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    Ok(fnp_argsort.call1((input,))?.unbind())
}

#[inline(never)]
fn ledger_orig_f32_argsort_reference(
    numpy_argsort: &Bound<'_, PyAny>,
    input: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    Ok(numpy_argsort.call1((input,))?.unbind())
}

fn ledger_tail_stats(samples: &RefCell<Vec<f64>>) -> (usize, f64, f64) {
    let samples = samples.borrow();
    let count = samples.len().min(10);
    assert!(
        count >= 2,
        "Criterion must retain at least two paired samples"
    );
    let tail = &samples[samples.len() - count..];
    let mean = tail.iter().sum::<f64>() / count as f64;
    let variance = tail
        .iter()
        .map(|sample| {
            let delta = sample - mean;
            delta * delta
        })
        .sum::<f64>()
        / (count - 1) as f64;
    (count, mean, variance.sqrt() * 100.0 / mean)
}

fn report_ledger_pair(
    row: &str,
    candidate_samples: &RefCell<Vec<f64>>,
    orig_samples: &RefCell<Vec<f64>>,
) {
    if candidate_samples.borrow().is_empty() && orig_samples.borrow().is_empty() {
        return;
    }
    let (candidate_n, candidate_ns, candidate_cv) = ledger_tail_stats(candidate_samples);
    let (orig_n, orig_ns, orig_cv) = ledger_tail_stats(orig_samples);
    assert_eq!(candidate_n, orig_n);
    println!(
        "LEDGER_AUDIT row={row} samples={candidate_n} candidate_mean_ms={:.6} \
         candidate_cv_pct={candidate_cv:.3} orig_mean_ms={:.6} orig_cv_pct={orig_cv:.3} \
         orig_over_candidate={:.4}",
        candidate_ns / 1_000_000.0,
        orig_ns / 1_000_000.0,
        orig_ns / candidate_ns,
    );
}

fn report_substrate_v2_pair(
    row: &str,
    candidate_samples: &RefCell<Vec<f64>>,
    orig_samples: &RefCell<Vec<f64>>,
) {
    if candidate_samples.borrow().is_empty() && orig_samples.borrow().is_empty() {
        return;
    }
    let (candidate_n, candidate_ns, candidate_cv) = ledger_tail_stats(candidate_samples);
    let (orig_n, orig_ns, orig_cv) = ledger_tail_stats(orig_samples);
    assert_eq!(candidate_n, orig_n);
    println!(
        "SUBSTRATE_V2 row={row} samples={candidate_n} candidate_mean_ms={:.6} \
         candidate_cv_pct={candidate_cv:.3} orig_mean_ms={:.6} orig_cv_pct={orig_cv:.3} \
         orig_over_candidate={:.4}",
        candidate_ns / 1_000_000.0,
        orig_ns / 1_000_000.0,
        orig_ns / candidate_ns,
    );
}

const MEDIAN_GATE_FINAL_BATCHES: usize = 10;
const MEDIAN_GATE_OBSERVATIONS_PER_BATCH: usize = 2;

#[derive(Clone, Copy)]
struct MedianGateDistribution {
    median: f64,
    p10: f64,
    p90: f64,
    low: f64,
    high: f64,
    cv_pct: f64,
    above_one: usize,
}

fn median_gate_quantile(sorted: &[f64], quantile: f64) -> f64 {
    assert!(!sorted.is_empty());
    let position = quantile * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = position - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

fn median_gate_distribution(samples: &[f64]) -> MedianGateDistribution {
    assert!(samples.len() >= 2);
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples
        .iter()
        .map(|sample| {
            let delta = sample - mean;
            delta * delta
        })
        .sum::<f64>()
        / (samples.len() - 1) as f64;
    MedianGateDistribution {
        median: median_gate_quantile(&sorted, 0.5),
        p10: median_gate_quantile(&sorted, 0.1),
        p90: median_gate_quantile(&sorted, 0.9),
        low: sorted[0],
        high: sorted[sorted.len() - 1],
        cv_pct: variance.sqrt() * 100.0 / mean,
        above_one: samples.iter().filter(|&&ratio| ratio > 1.0).count(),
    }
}

fn median_gate_tail(samples: &RefCell<Vec<f64>>) -> Vec<f64> {
    let samples = samples.borrow();
    let retained = MEDIAN_GATE_FINAL_BATCHES * MEDIAN_GATE_OBSERVATIONS_PER_BATCH;
    assert!(
        samples.len() >= retained,
        "Criterion must retain {retained} median-gate observations"
    );
    samples[samples.len() - retained..].to_vec()
}

fn report_median_gate_pair(
    row: &str,
    null_base_ns: &RefCell<Vec<f64>>,
    null_peer_ns: &RefCell<Vec<f64>>,
    null_ratios: &RefCell<Vec<f64>>,
    base_ns: &RefCell<Vec<f64>>,
    candidate_ns: &RefCell<Vec<f64>>,
    effect_ratios: &RefCell<Vec<f64>>,
) {
    if effect_ratios.borrow().is_empty() {
        return;
    }
    let null_base = median_gate_tail(null_base_ns);
    let null_peer = median_gate_tail(null_peer_ns);
    let null = median_gate_distribution(&median_gate_tail(null_ratios));
    let base = median_gate_distribution(&median_gate_tail(base_ns));
    let candidate = median_gate_distribution(&median_gate_tail(candidate_ns));
    let effect = median_gate_distribution(&median_gate_tail(effect_ratios));
    let null_brackets_one = null.p10 <= 1.0 && null.p90 >= 1.0;
    let verdict = if !null_brackets_one {
        "BIASED_NULL"
    } else if effect.median > null.p90 {
        "WIN"
    } else if effect.median < null.p10 {
        "PROFILE_REQUIRED"
    } else {
        "UNDECIDED"
    };
    let null_base_cv = median_gate_distribution(&null_base).cv_pct;
    let null_peer_cv = median_gate_distribution(&null_peer).cv_pct;
    println!(
        "NULL_MEDIAN_GATE row={row} observations={} base_median_ms={:.6} \
         candidate_median_ms={:.6} base_cv_pct={:.3} candidate_cv_pct={:.3} \
         effect_median={:.6} effect_p10={:.6} effect_p90={:.6} \
         effect_low={:.6} effect_high={:.6} effect_cv_pct={:.3} effect_above_one={} \
         null_median={:.6} null_p10={:.6} null_p90={:.6} null_low={:.6} \
         null_high={:.6} null_cv_pct={:.3} null_base_cv_pct={:.3} \
         null_peer_cv_pct={:.3} null_corrected_median={:.6} verdict={verdict}",
        effect_ratios
            .borrow()
            .len()
            .min(MEDIAN_GATE_FINAL_BATCHES * MEDIAN_GATE_OBSERVATIONS_PER_BATCH),
        base.median / 1_000_000.0,
        candidate.median / 1_000_000.0,
        base.cv_pct,
        candidate.cv_pct,
        effect.median,
        effect.p10,
        effect.p90,
        effect.low,
        effect.high,
        effect.cv_pct,
        effect.above_one,
        null.median,
        null.p10,
        null.p90,
        null.low,
        null.high,
        null.cv_pct,
        null_base_cv,
        null_peer_cv,
        effect.median / null.median,
    );
}

fn time_python_binary_call<'py>(
    function: &Bound<'py, PyAny>,
    lhs: &Bound<'py, PyAny>,
    rhs: &Bound<'py, PyAny>,
) -> Duration {
    let start = Instant::now();
    let function = black_box(function);
    let lhs = black_box(lhs);
    let rhs = black_box(rhs);
    let result = function
        .call1((lhs, rhs))
        .expect("median-gate binary Python call");
    drop(black_box(result));
    start.elapsed()
}

fn time_python_unary_call<'py>(
    function: &Bound<'py, PyAny>,
    input: &Bound<'py, PyAny>,
) -> Duration {
    let start = Instant::now();
    let function = black_box(function);
    let input = black_box(input);
    let result = function
        .call1((input,))
        .expect("median-gate unary Python call");
    drop(black_box(result));
    start.elapsed()
}

fn bench_median_gate_python_binary<'py>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    bench_name: &'static str,
    row: &'static str,
    base: &Bound<'py, PyAny>,
    candidate: &Bound<'py, PyAny>,
    lhs: &Bound<'py, PyAny>,
    rhs: &Bound<'py, PyAny>,
) {
    let null_base_ns = RefCell::new(Vec::new());
    let null_peer_ns = RefCell::new(Vec::new());
    let null_ratios = RefCell::new(Vec::new());
    let base_ns = RefCell::new(Vec::new());
    let candidate_ns = RefCell::new(Vec::new());
    let effect_ratios = RefCell::new(Vec::new());
    group.bench_function(bench_name, |bench| {
        bench.iter_custom(|iterations| {
            let mut combined = Duration::ZERO;
            for _ in 0..iterations {
                // The exact-function A/A null is always measured first. The two observations
                // are ABBA then BAAB, balancing call position before the effect is observed.
                for observation in 0..MEDIAN_GATE_OBSERVATIONS_PER_BATCH {
                    let outer_base = observation & 1 == 0;
                    let (base_total, peer_total) = if outer_base {
                        let a1 = time_python_binary_call(base, lhs, rhs);
                        let b1 = time_python_binary_call(base, lhs, rhs);
                        let b2 = time_python_binary_call(base, lhs, rhs);
                        let a2 = time_python_binary_call(base, lhs, rhs);
                        (a1 + a2, b1 + b2)
                    } else {
                        let b1 = time_python_binary_call(base, lhs, rhs);
                        let a1 = time_python_binary_call(base, lhs, rhs);
                        let a2 = time_python_binary_call(base, lhs, rhs);
                        let b2 = time_python_binary_call(base, lhs, rhs);
                        (a1 + a2, b1 + b2)
                    };
                    let base_average = base_total.as_secs_f64() * 0.5e9;
                    let peer_average = peer_total.as_secs_f64() * 0.5e9;
                    null_base_ns.borrow_mut().push(base_average);
                    null_peer_ns.borrow_mut().push(peer_average);
                    null_ratios.borrow_mut().push(base_average / peer_average);
                    combined += base_total + peer_total;
                }
                for observation in 0..MEDIAN_GATE_OBSERVATIONS_PER_BATCH {
                    let outer_base = observation & 1 == 0;
                    let (base_total, candidate_total) = if outer_base {
                        let a1 = time_python_binary_call(base, lhs, rhs);
                        let b1 = time_python_binary_call(candidate, lhs, rhs);
                        let b2 = time_python_binary_call(candidate, lhs, rhs);
                        let a2 = time_python_binary_call(base, lhs, rhs);
                        (a1 + a2, b1 + b2)
                    } else {
                        let b1 = time_python_binary_call(candidate, lhs, rhs);
                        let a1 = time_python_binary_call(base, lhs, rhs);
                        let a2 = time_python_binary_call(base, lhs, rhs);
                        let b2 = time_python_binary_call(candidate, lhs, rhs);
                        (a1 + a2, b1 + b2)
                    };
                    let base_average = base_total.as_secs_f64() * 0.5e9;
                    let candidate_average = candidate_total.as_secs_f64() * 0.5e9;
                    base_ns.borrow_mut().push(base_average);
                    candidate_ns.borrow_mut().push(candidate_average);
                    effect_ratios
                        .borrow_mut()
                        .push(base_average / candidate_average);
                    combined += base_total + candidate_total;
                }
            }
            combined
        });
    });
    report_median_gate_pair(
        row,
        &null_base_ns,
        &null_peer_ns,
        &null_ratios,
        &base_ns,
        &candidate_ns,
        &effect_ratios,
    );
}

fn bench_median_gate_python_unary<'py>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    bench_name: &'static str,
    row: &'static str,
    base: &Bound<'py, PyAny>,
    candidate: &Bound<'py, PyAny>,
    input: &Bound<'py, PyAny>,
) {
    let null_base_ns = RefCell::new(Vec::new());
    let null_peer_ns = RefCell::new(Vec::new());
    let null_ratios = RefCell::new(Vec::new());
    let base_ns = RefCell::new(Vec::new());
    let candidate_ns = RefCell::new(Vec::new());
    let effect_ratios = RefCell::new(Vec::new());
    group.bench_function(bench_name, |bench| {
        bench.iter_custom(|iterations| {
            let mut combined = Duration::ZERO;
            for _ in 0..iterations {
                for observation in 0..MEDIAN_GATE_OBSERVATIONS_PER_BATCH {
                    let outer_base = observation & 1 == 0;
                    let (base_total, peer_total) = if outer_base {
                        let a1 = time_python_unary_call(base, input);
                        let b1 = time_python_unary_call(base, input);
                        let b2 = time_python_unary_call(base, input);
                        let a2 = time_python_unary_call(base, input);
                        (a1 + a2, b1 + b2)
                    } else {
                        let b1 = time_python_unary_call(base, input);
                        let a1 = time_python_unary_call(base, input);
                        let a2 = time_python_unary_call(base, input);
                        let b2 = time_python_unary_call(base, input);
                        (a1 + a2, b1 + b2)
                    };
                    let base_average = base_total.as_secs_f64() * 0.5e9;
                    let peer_average = peer_total.as_secs_f64() * 0.5e9;
                    null_base_ns.borrow_mut().push(base_average);
                    null_peer_ns.borrow_mut().push(peer_average);
                    null_ratios.borrow_mut().push(base_average / peer_average);
                    combined += base_total + peer_total;
                }
                for observation in 0..MEDIAN_GATE_OBSERVATIONS_PER_BATCH {
                    let outer_base = observation & 1 == 0;
                    let (base_total, candidate_total) = if outer_base {
                        let a1 = time_python_unary_call(base, input);
                        let b1 = time_python_unary_call(candidate, input);
                        let b2 = time_python_unary_call(candidate, input);
                        let a2 = time_python_unary_call(base, input);
                        (a1 + a2, b1 + b2)
                    } else {
                        let b1 = time_python_unary_call(candidate, input);
                        let a1 = time_python_unary_call(base, input);
                        let a2 = time_python_unary_call(base, input);
                        let b2 = time_python_unary_call(candidate, input);
                        (a1 + a2, b1 + b2)
                    };
                    let base_average = base_total.as_secs_f64() * 0.5e9;
                    let candidate_average = candidate_total.as_secs_f64() * 0.5e9;
                    base_ns.borrow_mut().push(base_average);
                    candidate_ns.borrow_mut().push(candidate_average);
                    effect_ratios
                        .borrow_mut()
                        .push(base_average / candidate_average);
                    combined += base_total + candidate_total;
                }
            }
            combined
        });
    });
    report_median_gate_pair(
        row,
        &null_base_ns,
        &null_peer_ns,
        &null_ratios,
        &base_ns,
        &candidate_ns,
        &effect_ratios,
    );
}

fn bench_substrate_v2_python_binary_pair<'py>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    bench_name: &'static str,
    row: &'static str,
    candidate: &Bound<'py, PyAny>,
    orig: &Bound<'py, PyAny>,
    lhs: &Bound<'py, PyAny>,
    rhs: &Bound<'py, PyAny>,
) {
    let candidate_samples = RefCell::new(Vec::new());
    let orig_samples = RefCell::new(Vec::new());
    let order = Cell::new(0_u64);
    group.bench_function(bench_name, |bench| {
        bench.iter_custom(|iterations| {
            // Slow Python surface rows otherwise collapse to one A/B pair per
            // Criterion sample.  Keep each sample order-balanced and average
            // enough interleaved pairs to make worker jitter visible instead
            // of letting a single interruption decide the row.
            let measured_iterations = iterations.max(4);
            let measured_iterations = measured_iterations + (measured_iterations & 1);
            let mut candidate_total = Duration::ZERO;
            let mut orig_total = Duration::ZERO;
            for _ in 0..measured_iterations {
                let orig_first = order.get() & 1 == 1;
                order.set(order.get().wrapping_add(1));
                let time_call = |function: &Bound<'py, PyAny>| {
                    let start = Instant::now();
                    let lhs = black_box(lhs);
                    let rhs = black_box(rhs);
                    let result = function
                        .call1((lhs, rhs))
                        .expect("paired binary Python call");
                    black_box(result);
                    start.elapsed()
                };
                if orig_first {
                    orig_total += time_call(orig);
                    candidate_total += time_call(candidate);
                } else {
                    candidate_total += time_call(candidate);
                    orig_total += time_call(orig);
                }
            }
            candidate_samples
                .borrow_mut()
                .push(candidate_total.as_secs_f64() * 1e9 / measured_iterations as f64);
            orig_samples
                .borrow_mut()
                .push(orig_total.as_secs_f64() * 1e9 / measured_iterations as f64);
            (candidate_total + orig_total)
                .mul_f64(iterations as f64 / measured_iterations as f64)
        });
    });
    report_substrate_v2_pair(row, &candidate_samples, &orig_samples);
}

fn bench_substrate_v2_python_unary_pair<'py>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    bench_name: &'static str,
    row: &'static str,
    candidate: &Bound<'py, PyAny>,
    orig: &Bound<'py, PyAny>,
    input: &Bound<'py, PyAny>,
) {
    let candidate_samples = RefCell::new(Vec::new());
    let orig_samples = RefCell::new(Vec::new());
    let order = Cell::new(0_u64);
    group.bench_function(bench_name, |bench| {
        bench.iter_custom(|iterations| {
            let measured_iterations = iterations.max(4);
            let measured_iterations = measured_iterations + (measured_iterations & 1);
            let mut candidate_total = Duration::ZERO;
            let mut orig_total = Duration::ZERO;
            for _ in 0..measured_iterations {
                let orig_first = order.get() & 1 == 1;
                order.set(order.get().wrapping_add(1));
                let time_call = |function: &Bound<'py, PyAny>| {
                    let start = Instant::now();
                    let input = black_box(input);
                    let result = function.call1((input,)).expect("paired unary Python call");
                    black_box(result);
                    start.elapsed()
                };
                if orig_first {
                    orig_total += time_call(orig);
                    candidate_total += time_call(candidate);
                } else {
                    candidate_total += time_call(candidate);
                    orig_total += time_call(orig);
                }
            }
            candidate_samples
                .borrow_mut()
                .push(candidate_total.as_secs_f64() * 1e9 / measured_iterations as f64);
            orig_samples
                .borrow_mut()
                .push(orig_total.as_secs_f64() * 1e9 / measured_iterations as f64);
            (candidate_total + orig_total)
                .mul_f64(iterations as f64 / measured_iterations as f64)
        });
    });
    report_substrate_v2_pair(row, &candidate_samples, &orig_samples);
}

fn bench_completion_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_completion_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        println!(
            "ISA_PROVENANCE target_arch={} avx2={} sse2={}",
            std::env::consts::ARCH,
            cfg!(target_feature = "avx2"),
            cfg!(target_feature = "sse2"),
        );
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_completion_median_gate")
            .expect("completion bench module");
        fnp_python(&module).expect("initialize fnp_python completion bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy version")
            .extract::<String>()
            .expect("numpy version string");
        let numpy_simd = numpy
            .getattr("__config__")
            .expect("numpy config")
            .getattr("CONFIG")
            .expect("numpy CONFIG")
            .get_item("SIMD Extensions")
            .expect("numpy SIMD Extensions")
            .str()
            .expect("numpy SIMD str")
            .extract::<String>()
            .expect("numpy SIMD string value");
        let numpy_cpu_features = numpy
            .getattr("_core")
            .expect("numpy core")
            .getattr("_multiarray_umath")
            .expect("numpy multiarray umath")
            .getattr("__cpu_features__")
            .expect("numpy runtime CPU features")
            .str()
            .expect("numpy runtime CPU feature str")
            .extract::<String>()
            .expect("numpy runtime CPU feature string value");
        println!(
            "NUMPY_PROVENANCE version={numpy_version} build_simd={numpy_simd} \
             runtime_cpu_features={numpy_cpu_features}"
        );
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 powers = np.power(np.uint64(26), np.arange(5, dtype=np.uint64))\n\
                 u_a_ids = np.arange(0, 1_000_000, dtype=np.uint64)\n\
                 u_a_words = np.zeros((1_000_000, 16), dtype=np.uint32)\n\
                 u_a_words[:, :5] = (97 + (u_a_ids[:, None] // powers) % 26).astype(np.uint32)\n\
                 u_a = u_a_words.reshape(-1).view('U16')\n\
                 u_fresh_ids = np.arange(1_000_000, 1_500_000, dtype=np.uint64)\n\
                 u_fresh_words = np.zeros((500_000, 16), dtype=np.uint32)\n\
                 u_fresh_words[:, :5] = (97 + (u_fresh_ids[:, None] // powers) % 26).astype(np.uint32)\n\
                 u_fresh = u_fresh_words.reshape(-1).view('U16')\n\
                 u_b = np.concatenate([u_a[:500_000], u_fresh])\n\
                 u_union_ids = np.arange(2_000_000, 3_000_000, dtype=np.uint64)\n\
                 u_union_words = np.zeros((1_000_000, 16), dtype=np.uint32)\n\
                 u_union_words[:, :5] = (97 + (u_union_ids[:, None] // powers) % 26).astype(np.uint32)\n\
                 u_union_b = u_union_words.reshape(-1).view('U16')\n",
            )
            .expect("completion setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("completion setup");
        let u_a = namespace.get_item("u_a").expect("u_a present");
        let u_b = namespace.get_item("u_b").expect("u_b present");
        let u_union_b = namespace.get_item("u_union_b").expect("u_union_b present");
        let array_equal = numpy.getattr("array_equal").expect("numpy.array_equal");

        let fnp_unique = module.getattr("unique").expect("fnp unique");
        let np_unique = numpy.getattr("unique").expect("numpy unique");
        let fnp_union = module.getattr("union1d").expect("fnp union1d");
        let np_union = numpy.getattr("union1d").expect("numpy union1d");
        let fnp_setxor = module.getattr("setxor1d").expect("fnp setxor1d");
        let np_setxor = numpy.getattr("setxor1d").expect("numpy setxor1d");

        for (label, candidate, base) in [
            (
                "U16 unique",
                fnp_unique.call1((&u_a,)).expect("fnp unique parity"),
                np_unique.call1((&u_a,)).expect("numpy unique parity"),
            ),
            (
                "U16 disjoint union",
                fnp_union
                    .call1((&u_a, &u_union_b))
                    .expect("fnp union parity"),
                np_union
                    .call1((&u_a, &u_union_b))
                    .expect("numpy union parity"),
            ),
            (
                "U16 50% overlap setxor",
                fnp_setxor.call1((&u_a, &u_b)).expect("fnp setxor parity"),
                np_setxor.call1((&u_a, &u_b)).expect("numpy setxor parity"),
            ),
        ] {
            let candidate_dtype = candidate.getattr("dtype").expect("candidate dtype");
            let base_dtype = base.getattr("dtype").expect("base dtype");
            assert_eq!(
                candidate_dtype
                    .getattr("str")
                    .expect("candidate dtype str")
                    .extract::<String>()
                    .expect("candidate dtype str value"),
                base_dtype
                    .getattr("str")
                    .expect("base dtype str")
                    .extract::<String>()
                    .expect("base dtype str value"),
                "{label} dtype string parity",
            );
            assert!(
                candidate_dtype
                    .getattr("metadata")
                    .expect("candidate dtype metadata")
                    .eq(base_dtype.getattr("metadata").expect("base dtype metadata"))
                    .expect("dtype metadata equality"),
                "{label} dtype metadata parity",
            );
            assert!(
                array_equal
                    .call1((&candidate, &base))
                    .expect("completion array_equal")
                    .extract::<bool>()
                    .expect("completion array_equal bool"),
                "{label} value parity",
            );
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "{label} byte parity",
            );
        }

        bench_median_gate_python_unary(
            &mut group,
            "u16_unique_1m_null_then_effect",
            "u16_unique_1m",
            &np_unique,
            &fnp_unique,
            &u_a,
        );
        bench_median_gate_python_binary(
            &mut group,
            "u16_union_disjoint_1m_null_then_effect",
            "u16_union_disjoint_1m",
            &np_union,
            &fnp_union,
            &u_a,
            &u_union_b,
        );
        bench_median_gate_python_binary(
            &mut group,
            "u16_setxor_1m_null_then_effect",
            "u16_setxor_1m",
            &np_setxor,
            &fnp_setxor,
            &u_a,
            &u_b,
        );
    });

    group.finish();
}

fn bench_f64_transcendental_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_f64_transcendental_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        println!(
            "ISA_PROVENANCE target_arch={} avx2={} sse2={}",
            std::env::consts::ARCH,
            cfg!(target_feature = "avx2"),
            cfg!(target_feature = "sse2"),
        );
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_f64_transcendental_median_gate")
            .expect("transcendental bench module");
        fnp_python(&module).expect("initialize fnp_python transcendental bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy version")
            .extract::<String>()
            .expect("numpy version string");
        let numpy_cpu_features = numpy
            .getattr("_core")
            .expect("numpy core")
            .getattr("_multiarray_umath")
            .expect("numpy multiarray umath")
            .getattr("__cpu_features__")
            .expect("numpy runtime CPU features")
            .str()
            .expect("numpy runtime CPU feature str")
            .extract::<String>()
            .expect("numpy runtime CPU feature string value");
        println!(
            "NUMPY_PROVENANCE version={numpy_version} runtime_cpu_features={numpy_cpu_features}"
        );
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260710)\n\
                 t_262k = rng.standard_normal(262_144)\n\
                 t_1m = rng.standard_normal(1_048_576)\n\
                 t_4m = rng.standard_normal(4_194_304)\n",
            )
            .expect("transcendental setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("transcendental setup");
        let t_262k = namespace.get_item("t_262k").expect("t_262k present");
        let t_1m = namespace.get_item("t_1m").expect("t_1m present");
        let t_4m = namespace.get_item("t_4m").expect("t_4m present");

        // Diagnostic parity probe (print, not assert): fnp's native f64 route is
        // scalar system libm; numpy may dispatch a SIMD kernel on some workers.
        // Byte-level agreement per worker is itself evidence for the transcendental
        // lane (see the 2026-07-10 ISA addendum), so record it instead of dying.
        for name in ["sin", "cos", "tan", "tanh", "expm1"] {
            let fnp_fn = module.getattr(name).expect("fnp transcendental fn");
            let np_fn = numpy.getattr(name).expect("numpy transcendental fn");
            for (label, input) in [("262k", &t_262k), ("1m", &t_1m), ("4m", &t_4m)] {
                let candidate = fnp_fn.call1((input,)).expect("fnp parity call");
                let base = np_fn.call1((input,)).expect("numpy parity call");
                let candidate_bytes: Vec<u8> = candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract()
                    .expect("candidate byte Vec");
                let base_bytes: Vec<u8> = base
                    .call_method0("tobytes")
                    .expect("base bytes")
                    .extract()
                    .expect("base byte Vec");
                let first_diff = candidate_bytes
                    .chunks_exact(8)
                    .zip(base_bytes.chunks_exact(8))
                    .position(|(a, b)| a != b);
                let diff_count = candidate_bytes
                    .chunks_exact(8)
                    .zip(base_bytes.chunks_exact(8))
                    .filter(|(a, b)| a != b)
                    .count();
                println!(
                    "TRANSCENDENTAL_PARITY op={name} n={label} byte_equal={} \
                     diff_elems={diff_count} first_diff_elem={:?}",
                    candidate_bytes == base_bytes,
                    first_diff,
                );
            }
        }

        let fnp_sin = module.getattr("sin").expect("fnp sin");
        let np_sin = numpy.getattr("sin").expect("numpy sin");
        let fnp_cos = module.getattr("cos").expect("fnp cos");
        let np_cos = numpy.getattr("cos").expect("numpy cos");
        let fnp_tan = module.getattr("tan").expect("fnp tan");
        let np_tan = numpy.getattr("tan").expect("numpy tan");
        let fnp_tanh = module.getattr("tanh").expect("fnp tanh");
        let np_tanh = numpy.getattr("tanh").expect("numpy tanh");
        let fnp_expm1 = module.getattr("expm1").expect("fnp expm1");
        let np_expm1 = numpy.getattr("expm1").expect("numpy expm1");

        bench_median_gate_python_unary(
            &mut group,
            "f64_sin_262k_null_then_effect",
            "f64_sin_262k",
            &np_sin,
            &fnp_sin,
            &t_262k,
        );
        bench_median_gate_python_unary(
            &mut group,
            "f64_sin_1m_null_then_effect",
            "f64_sin_1m",
            &np_sin,
            &fnp_sin,
            &t_1m,
        );
        bench_median_gate_python_unary(
            &mut group,
            "f64_sin_4m_null_then_effect",
            "f64_sin_4m",
            &np_sin,
            &fnp_sin,
            &t_4m,
        );
        bench_median_gate_python_unary(
            &mut group,
            "f64_cos_1m_null_then_effect",
            "f64_cos_1m",
            &np_cos,
            &fnp_cos,
            &t_1m,
        );
        bench_median_gate_python_unary(
            &mut group,
            "f64_tan_1m_null_then_effect",
            "f64_tan_1m",
            &np_tan,
            &fnp_tan,
            &t_1m,
        );
        bench_median_gate_python_unary(
            &mut group,
            "f64_tanh_1m_null_then_effect",
            "f64_tanh_1m",
            &np_tanh,
            &fnp_tanh,
            &t_1m,
        );
        bench_median_gate_python_unary(
            &mut group,
            "f64_expm1_262k_null_then_effect",
            "f64_expm1_262k",
            &np_expm1,
            &fnp_expm1,
            &t_262k,
        );
        bench_median_gate_python_unary(
            &mut group,
            "f64_expm1_1m_null_then_effect",
            "f64_expm1_1m",
            &np_expm1,
            &fnp_expm1,
            &t_1m,
        );
    });

    group.finish();
}

fn bench_f64_exp_log_probe(c: &mut Criterion) {
    // Probe for bead deadlock-audit-gkznn (reopen of the stale 2026-06-09
    // exp/log passthrough decision): (1) BYTE PROBE — does numpy's f64
    // exp/log/log2/log10 output match Rust scalar system-libm bit-for-bit on
    // this worker? (2) TIMING — does a rayon parallel scalar-libm map (with a
    // deliberate vec![0.0; n] zero-init handicap the real zero-copy path would
    // not pay) beat numpy's kernel? Both must hold before any production
    // rewiring; the probe writes evidence only.
    let mut group = c.benchmark_group("python_f64_exp_log_probe");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let numpy = py.import("numpy").expect("numpy oracle");
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy version")
            .extract::<String>()
            .expect("numpy version string");
        println!("EXP_LOG_PROBE_NUMPY version={numpy_version}");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 e_1m = rng.standard_normal(1_048_576)\n\
                 e_4m = rng.standard_normal(4_194_304)\n\
                 l_1m = np.abs(rng.standard_normal(1_048_576)) + 0.5\n\
                 l_4m = np.abs(rng.standard_normal(4_194_304)) + 0.5\n",
            )
            .expect("probe setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("probe setup");
        let e_1m = namespace.get_item("e_1m").expect("e_1m present");
        let e_4m = namespace.get_item("e_4m").expect("e_4m present");
        let l_1m = namespace.get_item("l_1m").expect("l_1m present");
        let l_4m = namespace.get_item("l_4m").expect("l_4m present");

        let to_vec = |arr: &Bound<'_, PyAny>| -> Vec<f64> {
            let raw: Vec<u8> = arr
                .call_method0("tobytes")
                .expect("probe input bytes")
                .extract()
                .expect("probe input byte Vec");
            raw.chunks_exact(8)
                .map(|chunk| f64::from_ne_bytes(chunk.try_into().expect("one native f64")))
                .collect()
        };

        // (1) BYTE PROBE: numpy output vs Rust scalar libm, element-exact.
        for (name, rust_fn, input) in [
            ("exp", f64::exp as fn(f64) -> f64, &e_1m),
            ("log", f64::ln as fn(f64) -> f64, &l_1m),
            ("log2", f64::log2 as fn(f64) -> f64, &l_1m),
            ("log10", f64::log10 as fn(f64) -> f64, &l_1m),
        ] {
            let data = to_vec(input);
            let np_bytes: Vec<u8> = numpy
                .getattr(name)
                .expect("numpy probe fn")
                .call1((input,))
                .expect("numpy probe call")
                .call_method0("tobytes")
                .expect("numpy probe bytes")
                .extract()
                .expect("numpy probe byte Vec");
            let mut diff_elems = 0usize;
            let mut first_diff = None;
            let mut max_bitdiff: u64 = 0;
            for (index, (np_chunk, &value)) in
                np_bytes.chunks_exact(8).zip(data.iter()).enumerate()
            {
                let np_bits = u64::from_ne_bytes(np_chunk.try_into().expect("np f64 chunk"));
                let mine_bits = rust_fn(value).to_bits();
                if np_bits != mine_bits {
                    diff_elems += 1;
                    if first_diff.is_none() {
                        first_diff = Some(index);
                    }
                    max_bitdiff = max_bitdiff.max(np_bits.abs_diff(mine_bits));
                }
            }
            println!(
                "EXP_LOG_PROBE op={name} n=1m byte_equal={} diff_elems={diff_elems} \
                 first_diff_elem={first_diff:?} max_bitdiff={max_bitdiff}",
                diff_elems == 0,
            );
        }

        // (2) TIMING: ledger-pair ABBA — candidate = parallel scalar-libm map
        // (zero-init handicap), orig = the numpy call. Plus numpy A/A nulls.
        for (row, name, rust_fn, input) in [
            ("exp_log_probe_exp_1m", "exp", f64::exp as fn(f64) -> f64, &e_1m),
            ("exp_log_probe_exp_4m", "exp", f64::exp as fn(f64) -> f64, &e_4m),
            ("exp_log_probe_log_1m", "log", f64::ln as fn(f64) -> f64, &l_1m),
            ("exp_log_probe_log_4m", "log", f64::ln as fn(f64) -> f64, &l_4m),
        ] {
            let data = to_vec(input);
            let np_fn = numpy.getattr(name).expect("numpy timing fn");
            let run_candidate = || {
                let n = data.len();
                let mut out = vec![0.0f64; n];
                let chunk = n.div_ceil(rayon::current_num_threads().max(1));
                out.par_chunks_mut(chunk)
                    .zip(data.par_chunks(chunk))
                    .for_each(|(o, i)| {
                        for (slot, &value) in o.iter_mut().zip(i.iter()) {
                            *slot = rust_fn(value);
                        }
                    });
                out
            };
            let candidate_samples = RefCell::new(Vec::new());
            let orig_samples = RefCell::new(Vec::new());
            let order = Cell::new(0u64);
            group.bench_function(format!("{row}_paired"), |bench| {
                bench.iter_custom(|iterations| {
                    let mut candidate_total = Duration::ZERO;
                    let mut orig_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let orig_first = order.get() & 1 == 1;
                        order.set(order.get().wrapping_add(1));
                        if orig_first {
                            let start = Instant::now();
                            black_box(np_fn.call1((input,)).expect("np timing call"));
                            orig_total += start.elapsed();
                            let start = Instant::now();
                            black_box(run_candidate());
                            candidate_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(run_candidate());
                            candidate_total += start.elapsed();
                            let start = Instant::now();
                            black_box(np_fn.call1((input,)).expect("np timing call"));
                            orig_total += start.elapsed();
                        }
                    }
                    candidate_samples
                        .borrow_mut()
                        .push(candidate_total.as_secs_f64() * 1e9 / iterations as f64);
                    orig_samples
                        .borrow_mut()
                        .push(orig_total.as_secs_f64() * 1e9 / iterations as f64);
                    candidate_total + orig_total
                });
            });
            report_ledger_pair(row, &candidate_samples, &orig_samples);

            let null_a = RefCell::new(Vec::new());
            let null_b = RefCell::new(Vec::new());
            let null_order = Cell::new(0u64);
            group.bench_function(format!("{row}_null_aa"), |bench| {
                bench.iter_custom(|iterations| {
                    let mut a_total = Duration::ZERO;
                    let mut b_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let b_first = null_order.get() & 1 == 1;
                        null_order.set(null_order.get().wrapping_add(1));
                        if b_first {
                            let start = Instant::now();
                            black_box(np_fn.call1((input,)).expect("np null call"));
                            b_total += start.elapsed();
                            let start = Instant::now();
                            black_box(np_fn.call1((input,)).expect("np null call"));
                            a_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(np_fn.call1((input,)).expect("np null call"));
                            a_total += start.elapsed();
                            let start = Instant::now();
                            black_box(np_fn.call1((input,)).expect("np null call"));
                            b_total += start.elapsed();
                        }
                    }
                    null_a
                        .borrow_mut()
                        .push(a_total.as_secs_f64() * 1e9 / iterations as f64);
                    null_b
                        .borrow_mut()
                        .push(b_total.as_secs_f64() * 1e9 / iterations as f64);
                    a_total + b_total
                });
            });
            report_ledger_pair(&format!("{row}_null"), &null_a, &null_b);
        }
    });

    group.finish();
}

fn bench_f64_exp_log_median_gate(c: &mut Criterion) {
    // SHIP rows for bead deadlock-audit-gkznn: the ACTUAL wired route
    // (fnp.exp/log/log2/log10 -> try_zerocopy_f64_unary parallel scalar-libm
    // map on non-AVX-512 hosts) vs numpy, with pre-timing byte parity asserts.
    // On an avx512f worker the ISA gate routes these to the numpy passthrough
    // and the rows read ~1.0x by construction; the probe group's
    // EXP_LOG_PROBE byte rows identify the worker class in the same run.
    let mut group = c.benchmark_group("python_f64_exp_log_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module =
            PyModule::new(py, "fnp_python_exp_log_median_gate").expect("exp/log bench module");
        fnp_python(&module).expect("initialize fnp_python exp/log bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        // Worker-class provenance: the rows below are only expected to beat
        // numpy where the ISA gate enables the native route (x86-64 with
        // avx512f=false); elsewhere they measure passthrough-vs-numpy ~1.0x.
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy version")
            .extract::<String>()
            .expect("numpy version string");
        #[cfg(target_arch = "x86_64")]
        let native_route = !std::arch::is_x86_feature_detected!("avx512f");
        #[cfg(not(target_arch = "x86_64"))]
        let native_route = false;
        println!("EXP_LOG_GATE_WORKER numpy={numpy_version} native_route={native_route}");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 e_1m = rng.standard_normal(1_048_576)\n\
                 e_4m = rng.standard_normal(4_194_304)\n\
                 l_1m = np.abs(rng.standard_normal(1_048_576)) + 0.5\n\
                 l_4m = np.abs(rng.standard_normal(4_194_304)) + 0.5\n",
            )
            .expect("exp/log gate setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("exp/log gate setup");
        let e_1m = namespace.get_item("e_1m").expect("e_1m present");
        let e_4m = namespace.get_item("e_4m").expect("e_4m present");
        let l_1m = namespace.get_item("l_1m").expect("l_1m present");
        let l_4m = namespace.get_item("l_4m").expect("l_4m present");

        let rows = [
            (
                "explog_exp_1m_null_then_effect",
                "explog_exp_1m",
                "exp",
                &e_1m,
            ),
            (
                "explog_exp_4m_null_then_effect",
                "explog_exp_4m",
                "exp",
                &e_4m,
            ),
            (
                "explog_exp2_4m_null_then_effect",
                "explog_exp2_4m",
                "exp2",
                &e_4m,
            ),
            (
                "explog_log_1m_null_then_effect",
                "explog_log_1m",
                "log",
                &l_1m,
            ),
            (
                "explog_log_4m_null_then_effect",
                "explog_log_4m",
                "log",
                &l_4m,
            ),
            (
                "explog_log2_4m_null_then_effect",
                "explog_log2_4m",
                "log2",
                &l_4m,
            ),
            (
                "explog_log10_4m_null_then_effect",
                "explog_log10_4m",
                "log10",
                &l_4m,
            ),
        ];
        for (bench_name, row, op, input) in rows {
            let fnp_fn = module.getattr(op).expect("fnp exp/log fn");
            let np_fn = numpy.getattr(op).expect("numpy exp/log fn");
            let candidate = fnp_fn.call1((input,)).expect("fnp exp/log parity call");
            let base = np_fn.call1((input,)).expect("numpy exp/log parity call");
            assert_eq!(
                candidate
                    .getattr("dtype")
                    .expect("candidate dtype")
                    .str()
                    .expect("candidate dtype str")
                    .to_string(),
                base.getattr("dtype")
                    .expect("base dtype")
                    .str()
                    .expect("base dtype str")
                    .to_string(),
                "exp/log {row} dtype parity",
            );
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "exp/log {row} byte parity",
            );
            bench_median_gate_python_unary(&mut group, bench_name, row, &np_fn, &fnp_fn, input);
        }
    });

    group.finish();
}

fn bench_bool_sort_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_bool_sort_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module =
            PyModule::new(py, "fnp_python_bool_sort_median_gate").expect("bool sort bench module");
        fnp_python(&module).expect("initialize fnp_python bool sort bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 b_8m = rng.integers(0, 2, 8_000_000).astype(bool)\n\
                 b_2m = rng.integers(0, 2, 2_000_000).astype(bool)\n",
            )
            .expect("bool sort setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("bool sort setup");
        let b_8m = namespace.get_item("b_8m").expect("b_8m present");
        let b_2m = namespace.get_item("b_2m").expect("b_2m present");

        let fnp_sort = module.getattr("sort").expect("fnp sort");
        let np_sort = numpy.getattr("sort").expect("numpy sort");
        for (label, input) in [("8m", &b_8m), ("2m", &b_2m)] {
            let candidate = fnp_sort.call1((input,)).expect("fnp bool sort parity");
            let base = np_sort.call1((input,)).expect("numpy bool sort parity");
            assert_eq!(
                candidate
                    .getattr("dtype")
                    .expect("candidate dtype")
                    .str()
                    .expect("candidate dtype str")
                    .to_string(),
                base.getattr("dtype")
                    .expect("base dtype")
                    .str()
                    .expect("base dtype str")
                    .to_string(),
                "bool sort {label} dtype parity",
            );
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "bool sort {label} byte parity",
            );
        }

        bench_median_gate_python_unary(
            &mut group,
            "bool_sort_8m_null_then_effect",
            "bool_sort_8m",
            &np_sort,
            &fnp_sort,
            &b_8m,
        );
        bench_median_gate_python_unary(
            &mut group,
            "bool_sort_2m_null_then_effect",
            "bool_sort_2m",
            &np_sort,
            &fnp_sort,
            &b_2m,
        );
    });

    group.finish();
}

fn bench_wide_string_sort_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_wide_string_sort_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_wide_string_sort_median_gate")
            .expect("wide string sort bench module");
        fnp_python(&module).expect("initialize fnp_python wide string sort bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 u9 = rng.integers(97, 123, (1_000_000, 9), dtype=np.uint32).reshape(-1).view('U9')\n\
                 u16 = rng.integers(97, 123, (1_000_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
                 s9 = rng.integers(97, 123, (1_000_000, 9), dtype=np.uint8).reshape(-1).view('S9')\n\
                 s16 = rng.integers(97, 123, (1_000_000, 16), dtype=np.uint8).reshape(-1).view('S16')\n",
            )
            .expect("wide string sort setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("wide string sort setup");
        let u9_input = namespace.get_item("u9").expect("u9 present");
        let u16_input = namespace.get_item("u16").expect("u16 present");
        let s9_input = namespace.get_item("s9").expect("s9 present");
        let s16_input = namespace.get_item("s16").expect("s16 present");
        let fnp_sort = module.getattr("sort").expect("fnp sort");
        let numpy_sort = numpy.getattr("sort").expect("numpy sort");

        for (label, input) in [
            ("U9", &u9_input),
            ("U16", &u16_input),
            ("S9", &s9_input),
            ("S16", &s16_input),
        ] {
            let candidate = fnp_sort.call1((input,)).expect("fnp wide string sort parity");
            let base = numpy_sort
                .call1((input,))
                .expect("numpy wide string sort parity");
            assert_eq!(
                candidate
                    .getattr("dtype")
                    .expect("candidate dtype")
                    .str()
                    .expect("candidate dtype str")
                    .to_string(),
                base.getattr("dtype")
                    .expect("base dtype")
                    .str()
                    .expect("base dtype str")
                    .to_string(),
                "wide string sort {label} dtype parity",
            );
            assert_eq!(
                candidate
                    .getattr("shape")
                    .expect("candidate shape")
                    .extract::<Vec<usize>>()
                    .expect("candidate shape Vec"),
                base.getattr("shape")
                    .expect("base shape")
                    .extract::<Vec<usize>>()
                    .expect("base shape Vec"),
                "wide string sort {label} shape parity",
            );
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "wide string sort {label} byte parity",
            );
            assert_eq!(
                candidate
                    .getattr("flags")
                    .expect("candidate flags")
                    .getattr("owndata")
                    .expect("candidate owndata")
                    .extract::<bool>()
                    .expect("candidate owndata bool"),
                base.getattr("flags")
                    .expect("base flags")
                    .getattr("owndata")
                    .expect("base owndata")
                    .extract::<bool>()
                    .expect("base owndata bool"),
                "wide string sort {label} ownership parity",
            );
        }

        group.bench_function("wide_string_sort_u16_1m_fnp_profile", |bench| {
            bench.iter(|| {
                black_box(
                    fnp_sort
                        .call1((black_box(&u16_input),))
                        .expect("profile fnp U16 sort"),
                )
            });
        });
        bench_median_gate_python_unary(
            &mut group,
            "wide_string_sort_u9_1m_null_then_effect",
            "wide_string_sort_u9_1m",
            &numpy_sort,
            &fnp_sort,
            &u9_input,
        );
        bench_median_gate_python_unary(
            &mut group,
            "wide_string_sort_u16_1m_null_then_effect",
            "wide_string_sort_u16_1m",
            &numpy_sort,
            &fnp_sort,
            &u16_input,
        );
        bench_median_gate_python_unary(
            &mut group,
            "wide_string_sort_s9_1m_null_then_effect",
            "wide_string_sort_s9_1m",
            &numpy_sort,
            &fnp_sort,
            &s9_input,
        );
        bench_median_gate_python_unary(
            &mut group,
            "wide_string_sort_s16_1m_null_then_effect",
            "wide_string_sort_s16_1m",
            &numpy_sort,
            &fnp_sort,
            &s16_input,
        );
    });

    group.finish();
    if std::env::var_os("FNP_WIDE_STRING_SORT_BENCH_ONLY").is_some() {
        use std::io::Write;
        let _ = std::io::stdout().flush();
        std::process::exit(0);
    }
}

fn bench_accumulate_extremum_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_accumulate_extremum_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_accumulate_extremum_median_gate")
            .expect("accumulate extremum bench module");
        fnp_python(&module).expect("initialize fnp_python accumulate extremum bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 x = rng.standard_normal(8_000_000).astype(np.float64)\n",
            )
            .expect("accumulate extremum setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("accumulate extremum setup");
        let input = namespace.get_item("x").expect("x present");
        let fnp_accumulate = module
            .getattr("maximum")
            .expect("fnp maximum")
            .getattr("accumulate")
            .expect("fnp maximum.accumulate");
        let numpy_accumulate = numpy
            .getattr("maximum")
            .expect("numpy maximum")
            .getattr("accumulate")
            .expect("numpy maximum.accumulate");

        let candidate = fnp_accumulate
            .call1((&input,))
            .expect("fnp maximum.accumulate parity");
        let base = numpy_accumulate
            .call1((&input,))
            .expect("numpy maximum.accumulate parity");
        assert_eq!(
            candidate
                .getattr("dtype")
                .expect("candidate dtype")
                .str()
                .expect("candidate dtype str")
                .to_string(),
            base.getattr("dtype")
                .expect("base dtype")
                .str()
                .expect("base dtype str")
                .to_string(),
            "maximum.accumulate dtype parity",
        );
        assert_eq!(
            candidate
                .getattr("shape")
                .expect("candidate shape")
                .extract::<Vec<usize>>()
                .expect("candidate shape Vec"),
            base.getattr("shape")
                .expect("base shape")
                .extract::<Vec<usize>>()
                .expect("base shape Vec"),
            "maximum.accumulate shape parity",
        );
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "maximum.accumulate byte parity",
        );

        group.bench_function("maximum_accumulate_f64_8m_fnp_profile", |bench| {
            bench.iter(|| {
                black_box(
                    fnp_accumulate
                        .call1((black_box(&input),))
                        .expect("profile fnp maximum.accumulate"),
                )
            });
        });
        bench_median_gate_python_unary(
            &mut group,
            "maximum_accumulate_f64_8m_null_then_effect",
            "maximum_accumulate_f64_8m",
            &numpy_accumulate,
            &fnp_accumulate,
            &input,
        );
    });

    group.finish();
    if std::env::var_os("FNP_ACCUMULATE_EXTREMUM_BENCH_ONLY").is_some() {
        use std::io::Write;
        let _ = std::io::stdout().flush();
        std::process::exit(0);
    }
}

fn bench_int_convolve_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_int_convolve_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_int_convolve_median_gate")
            .expect("int convolve bench module");
        fnp_python(&module).expect("initialize fnp_python int convolve bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 a = rng.integers(-2**31, 2**31, 200_000).astype(np.int64)\n\
                 v = rng.integers(-2**31, 2**31, 256).astype(np.int64)\n",
            )
            .expect("int convolve setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("int convolve setup");
        let a = namespace.get_item("a").expect("a present");
        let v = namespace.get_item("v").expect("v present");
        let fnp_convolve = module.getattr("convolve").expect("fnp convolve");
        let numpy_convolve = numpy.getattr("convolve").expect("numpy convolve");

        let candidate = fnp_convolve
            .call1((&a, &v))
            .expect("fnp int convolve parity");
        let base = numpy_convolve
            .call1((&a, &v))
            .expect("numpy int convolve parity");
        assert_eq!(
            candidate
                .getattr("dtype")
                .expect("candidate dtype")
                .str()
                .expect("candidate dtype str")
                .to_string(),
            base.getattr("dtype")
                .expect("base dtype")
                .str()
                .expect("base dtype str")
                .to_string(),
            "int convolve dtype parity",
        );
        assert_eq!(
            candidate
                .getattr("shape")
                .expect("candidate shape")
                .extract::<Vec<usize>>()
                .expect("candidate shape Vec"),
            base.getattr("shape")
                .expect("base shape")
                .extract::<Vec<usize>>()
                .expect("base shape Vec"),
            "int convolve shape parity",
        );
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "int convolve byte parity",
        );

        group.bench_function("int_convolve_i64_200k_256_fnp_profile", |bench| {
            bench.iter(|| {
                black_box(
                    fnp_convolve
                        .call1((black_box(&a), black_box(&v)))
                        .expect("profile fnp int convolve"),
                )
            });
        });
        bench_median_gate_python_binary(
            &mut group,
            "int_convolve_i64_200k_256_null_then_effect",
            "int_convolve_i64_200k_256",
            &numpy_convolve,
            &fnp_convolve,
            &a,
            &v,
        );
    });

    group.finish();
    if std::env::var_os("FNP_INT_CONVOLVE_BENCH_ONLY").is_some() {
        use std::io::Write;
        let _ = std::io::stdout().flush();
        std::process::exit(0);
    }
}

fn bench_int_matmul_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_int_matmul_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_int_matmul_median_gate")
            .expect("int matmul bench module");
        fnp_python(&module).expect("initialize fnp_python int matmul bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 a64 = rng.integers(-2**31, 2**31, (512, 512)).astype(np.int64)\n\
                 b64 = rng.integers(-2**31, 2**31, (512, 512)).astype(np.int64)\n\
                 a32 = rng.integers(-2**15, 2**15, (512, 512)).astype(np.int32)\n\
                 b32 = rng.integers(-2**15, 2**15, (512, 512)).astype(np.int32)\n\
                 ab64 = rng.integers(-2**31, 2**31, (64, 128, 128)).astype(np.int64)\n\
                 bb64 = rng.integers(-2**31, 2**31, (64, 128, 128)).astype(np.int64)\n\
                 mp64 = rng.integers(-2**31, 2**31, (256, 256)).astype(np.int64)\n\
                 p5 = 5\n",
            )
            .expect("int matmul setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("int matmul setup");
        let a64 = namespace.get_item("a64").expect("a64 present");
        let b64 = namespace.get_item("b64").expect("b64 present");
        let a32 = namespace.get_item("a32").expect("a32 present");
        let b32 = namespace.get_item("b32").expect("b32 present");

        let fnp_matmul = module.getattr("matmul").expect("fnp matmul");
        let np_matmul = numpy.getattr("matmul").expect("numpy matmul");
        for (label, x, y) in [
            ("i64_512", &a64, &b64),
            ("i32_512", &a32, &b32),
        ] {
            let candidate = fnp_matmul.call1((x, y)).expect("fnp int matmul parity");
            let base = np_matmul.call1((x, y)).expect("numpy int matmul parity");
            assert_eq!(
                candidate
                    .getattr("dtype")
                    .expect("candidate dtype")
                    .str()
                    .expect("candidate dtype str")
                    .to_string(),
                base.getattr("dtype")
                    .expect("base dtype")
                    .str()
                    .expect("base dtype str")
                    .to_string(),
                "int matmul {label} dtype parity",
            );
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "int matmul {label} byte parity",
            );
        }

        bench_median_gate_python_binary(
            &mut group,
            "int_matmul_i64_512_null_then_effect",
            "int_matmul_i64_512",
            &np_matmul,
            &fnp_matmul,
            &a64,
            &b64,
        );
        bench_median_gate_python_binary(
            &mut group,
            "int_matmul_i32_512_null_then_effect",
            "int_matmul_i32_512",
            &np_matmul,
            &fnp_matmul,
            &a32,
            &b32,
        );

        let ab64 = namespace.get_item("ab64").expect("ab64 present");
        let bb64 = namespace.get_item("bb64").expect("bb64 present");
        let mp64 = namespace.get_item("mp64").expect("mp64 present");
        let p5 = namespace.get_item("p5").expect("p5 present");
        let fnp_matrix_power = module
            .getattr("linalg")
            .expect("fnp linalg")
            .getattr("matrix_power")
            .expect("fnp matrix_power");
        let np_matrix_power = numpy
            .getattr("linalg")
            .expect("numpy linalg")
            .getattr("matrix_power")
            .expect("numpy matrix_power");
        for (label, f_c, f_b, x, y) in [
            ("i64_batched", &fnp_matmul, &np_matmul, &ab64, &bb64),
            ("i64_matpow5", &fnp_matrix_power, &np_matrix_power, &mp64, &p5),
        ] {
            let candidate = f_c.call1((x, y)).expect("fnp candidate parity");
            let base = f_b.call1((x, y)).expect("numpy base parity");
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "int {label} byte parity",
            );
        }
        bench_median_gate_python_binary(
            &mut group,
            "int_matmul_i64_batched_null_then_effect",
            "int_matmul_i64_batched",
            &np_matmul,
            &fnp_matmul,
            &ab64,
            &bb64,
        );
        bench_median_gate_python_binary(
            &mut group,
            "int_matrix_power_i64_256_p5_null_then_effect",
            "int_matrix_power_i64_256_p5",
            &np_matrix_power,
            &fnp_matrix_power,
            &mp64,
            &p5,
        );
    });

    group.finish();
}

fn bench_f16_matmul_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_f16_matmul_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_f16_matmul_median_gate")
            .expect("f16 matmul bench module");
        fnp_python(&module).expect("initialize fnp_python f16 matmul bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 h_a = rng.standard_normal((512, 512)).astype(np.float16)\n\
                 h_b = rng.standard_normal((512, 512)).astype(np.float16)\n\
                 hb_a = rng.standard_normal((8, 256, 256)).astype(np.float16)\n\
                 hb_b = rng.standard_normal((8, 256, 256)).astype(np.float16)\n\
                 hbc_a = rng.standard_normal((32, 128, 128)).astype(np.float16)\n\
                 hbc_b = rng.standard_normal((128, 96)).astype(np.float16)\n",
            )
            .expect("f16 matmul setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("f16 matmul setup");
        let h_a = namespace.get_item("h_a").expect("h_a present");
        let h_b = namespace.get_item("h_b").expect("h_b present");

        let fnp_matmul = module.getattr("matmul").expect("fnp matmul");
        let np_matmul = numpy.getattr("matmul").expect("numpy matmul");
        let candidate = fnp_matmul.call1((&h_a, &h_b)).expect("fnp f16 matmul parity");
        let base = np_matmul.call1((&h_a, &h_b)).expect("numpy f16 matmul parity");
        assert_eq!(
            candidate
                .getattr("dtype")
                .expect("candidate dtype")
                .str()
                .expect("candidate dtype str")
                .to_string(),
            base.getattr("dtype")
                .expect("base dtype")
                .str()
                .expect("base dtype str")
                .to_string(),
            "f16 matmul dtype parity",
        );
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 matmul byte parity",
        );

        bench_median_gate_python_binary(
            &mut group,
            "f16_matmul_512_null_then_effect",
            "f16_matmul_512",
            &np_matmul,
            &fnp_matmul,
            &h_a,
            &h_b,
        );

        let hb_a = namespace.get_item("hb_a").expect("hb_a present");
        let hb_b = namespace.get_item("hb_b").expect("hb_b present");
        let candidate = fnp_matmul
            .call1((&hb_a, &hb_b))
            .expect("fnp f16 batched matmul parity");
        let base = np_matmul
            .call1((&hb_a, &hb_b))
            .expect("numpy f16 batched matmul parity");
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 batched matmul byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_matmul_batched_8x256_null_then_effect",
            "f16_matmul_batched_8x256",
            &np_matmul,
            &fnp_matmul,
            &hb_a,
            &hb_b,
        );

        let hbc_a = namespace.get_item("hbc_a").expect("hbc_a present");
        let hbc_b = namespace.get_item("hbc_b").expect("hbc_b present");
        let candidate = fnp_matmul
            .call1((&hbc_a, &hbc_b))
            .expect("fnp f16 broadcast matmul parity");
        let base = np_matmul
            .call1((&hbc_a, &hbc_b))
            .expect("numpy f16 broadcast matmul parity");
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 broadcast matmul byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_matmul_broadcast_32x128_null_then_effect",
            "f16_matmul_broadcast_32x128",
            &np_matmul,
            &fnp_matmul,
            &hbc_a,
            &hbc_b,
        );
    });

    group.finish();
}

fn bench_multidot_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_multidot_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module =
            PyModule::new(py, "fnp_python_multidot_median_gate").expect("multidot bench module");
        fnp_python(&module).expect("initialize fnp_python multidot bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 md_args = [rng.standard_normal((512, 512)) for _ in range(3)]\n",
            )
            .expect("multidot setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("multidot setup");
        let md_args = namespace.get_item("md_args").expect("md_args present");

        let fnp_multidot = module
            .getattr("linalg")
            .expect("fnp linalg")
            .getattr("multi_dot")
            .expect("fnp multi_dot");
        let np_multidot = numpy
            .getattr("linalg")
            .expect("numpy linalg")
            .getattr("multi_dot")
            .expect("numpy multi_dot");
        let candidate = fnp_multidot
            .call1((&md_args,))
            .expect("fnp multi_dot parity");
        let base = np_multidot
            .call1((&md_args,))
            .expect("numpy multi_dot parity");
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "multi_dot byte parity",
        );

        bench_median_gate_python_unary(
            &mut group,
            "multidot_3x512_null_then_effect",
            "multidot_3x512",
            &np_multidot,
            &fnp_multidot,
            &md_args,
        );
    });

    group.finish();
}

fn bench_f16_einsum_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_f16_einsum_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_f16_einsum_median_gate")
            .expect("f16 einsum bench module");
        fnp_python(&module).expect("initialize fnp_python f16 einsum bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        namespace
            .set_item("fnp_mod", &module)
            .expect("expose fnp module");
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 es_a = (rng.standard_normal((512, 512)) * 0.3).astype(np.float16)\n\
                 es_b = (rng.standard_normal((512, 512)) * 0.3).astype(np.float16)\n\
                 fnp_es = lambda a, b: fnp_mod.einsum('ij,jk->ik', a, b)\n\
                 np_es = lambda a, b: np.einsum('ij,jk->ik', a, b)\n\
                 fnp_es_t = lambda a, b: fnp_mod.einsum('ij,lj->il', a, b)\n\
                 np_es_t = lambda a, b: np.einsum('ij,lj->il', a, b)\n\
                 fnp_es_g = lambda a, b: fnp_mod.einsum('ji,jl->il', a, b)\n\
                 np_es_g = lambda a, b: np.einsum('ji,jl->il', a, b)\n\
                 fnp_es_ts = lambda a, b: fnp_mod.einsum('ij,lj->li', a, b)\n\
                 np_es_ts = lambda a, b: np.einsum('ij,lj->li', a, b)\n\
                 fnp_es_gs = lambda a, b: fnp_mod.einsum('ji,jl->li', a, b)\n\
                 np_es_gs = lambda a, b: np.einsum('ji,jl->li', a, b)\n\
                 dot_a = (rng.standard_normal(8_388_608) * 0.3).astype(np.float16)\n\
                 dot_b = (rng.standard_normal(8_388_608) * 0.3).astype(np.float16)\n\
                 fnp_es_d = lambda a, b: fnp_mod.einsum('j,j->', a, b)\n\
                 np_es_d = lambda a, b: np.einsum('j,j->', a, b)\n\
                 fc_a = (rng.standard_normal((2896, 2896)) * 0.3).astype(np.float16)\n\
                 fc_b = (rng.standard_normal((2896, 2896)) * 0.3).astype(np.float16)\n\
                 fnp_es_fc = lambda a, b: fnp_mod.einsum('ij,ij->', a, b)\n\
                 np_es_fc = lambda a, b: np.einsum('ij,ij->', a, b)\n\
                 fnp_es_ew = lambda a, b: fnp_mod.einsum('j,j->j', a, b)\n\
                 np_es_ew = lambda a, b: np.einsum('j,j->j', a, b)\n\
                 ew64_a = rng.standard_normal(8_388_608)\n\
                 ew64_b = rng.standard_normal(8_388_608)\n\
                 bat_a = (rng.standard_normal((8, 256, 256)) * 0.3).astype(np.float16)\n\
                 bat_b = (rng.standard_normal((8, 256, 256)) * 0.3).astype(np.float16)\n\
                 fnp_es_b = lambda a, b: fnp_mod.einsum('bij,bjk->bik', a, b)\n\
                 np_es_b = lambda a, b: np.einsum('bij,bjk->bik', a, b)\n\
                 fnp_es_bt = lambda a, b: fnp_mod.einsum('bij,blj->bil', a, b)\n\
                 np_es_bt = lambda a, b: np.einsum('bij,blj->bil', a, b)\n\
                 fnp_es_bg = lambda a, b: fnp_mod.einsum('bji,bjl->bil', a, b)\n\
                 np_es_bg = lambda a, b: np.einsum('bji,bjl->bil', a, b)\n\
                 fnp_es_bts = lambda a, b: fnp_mod.einsum('bij,blj->bli', a, b)\n\
                 np_es_bts = lambda a, b: np.einsum('bij,blj->bli', a, b)\n\
                 fnp_es_bgs = lambda a, b: fnp_mod.einsum('bji,bjl->bli', a, b)\n\
                 np_es_bgs = lambda a, b: np.einsum('bji,bjl->bli', a, b)\n",
            )
            .expect("f16 einsum setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("f16 einsum setup");
        let es_a = namespace.get_item("es_a").expect("es_a present");
        let es_b = namespace.get_item("es_b").expect("es_b present");
        let fnp_es = namespace.get_item("fnp_es").expect("fnp_es present");
        let np_es = namespace.get_item("np_es").expect("np_es present");

        let candidate = fnp_es.call1((&es_a, &es_b)).expect("fnp f16 einsum parity");
        let base = np_es.call1((&es_a, &es_b)).expect("numpy f16 einsum parity");
        assert_eq!(
            candidate
                .getattr("dtype")
                .expect("candidate dtype")
                .str()
                .expect("candidate dtype str")
                .to_string(),
            base.getattr("dtype")
                .expect("base dtype")
                .str()
                .expect("base dtype str")
                .to_string(),
            "f16 einsum dtype parity",
        );
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum byte parity",
        );

        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_matmul_512_null_then_effect",
            "f16_einsum_matmul_512",
            &np_es,
            &fnp_es,
            &es_a,
            &es_b,
        );

        // Transposed spec ('ij,lj->il', the a@b.T idiom): a different numpy
        // contract class (wide-accumulate-once blocked-4) with its own kernel.
        let fnp_es_t = namespace.get_item("fnp_es_t").expect("fnp_es_t present");
        let np_es_t = namespace.get_item("np_es_t").expect("np_es_t present");
        let candidate_t = fnp_es_t
            .call1((&es_a, &es_b))
            .expect("fnp f16 einsum transposed parity");
        let base_t = np_es_t
            .call1((&es_a, &es_b))
            .expect("numpy f16 einsum transposed parity");
        assert_eq!(
            candidate_t
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_t
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum transposed byte parity",
        );

        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_transposed_512_null_then_effect",
            "f16_einsum_transposed_512",
            &np_es_t,
            &fnp_es_t,
            &es_a,
            &es_b,
        );

        // Gram spec ('ji,jl->il', the a.T@b idiom): the third numpy contract
        // class (per-step-narrow muladd rows, stride0_contig_outcontig) with
        // its own kernel. Same 512^2 operands (k = leading axis).
        let fnp_es_g = namespace.get_item("fnp_es_g").expect("fnp_es_g present");
        let np_es_g = namespace.get_item("np_es_g").expect("np_es_g present");
        let candidate_g = fnp_es_g
            .call1((&es_a, &es_b))
            .expect("fnp f16 einsum gram parity");
        let base_g = np_es_g
            .call1((&es_a, &es_b))
            .expect("numpy f16 einsum gram parity");
        assert_eq!(
            candidate_g
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_g
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum gram byte parity",
        );

        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_gram_512_null_then_effect",
            "f16_einsum_gram_512",
            &np_es_g,
            &fnp_es_g,
            &es_a,
            &es_b,
        );

        // Output-transposed variants: operand-swap arms of the transposed and
        // gram kernels ('ij,lj->li' / 'ji,jl->li'). Rows prove the swapped
        // dispatch engages the native route (effect >> 1, not numpy ~1.0x).
        for (bench_name, row, fnp_key, np_key) in [
            (
                "f16_einsum_transposed_swapped_512_null_then_effect",
                "f16_einsum_transposed_swapped_512",
                "fnp_es_ts",
                "np_es_ts",
            ),
            (
                "f16_einsum_gram_swapped_512_null_then_effect",
                "f16_einsum_gram_swapped_512",
                "fnp_es_gs",
                "np_es_gs",
            ),
        ] {
            let fnp_fn = namespace.get_item(fnp_key).expect("fnp swapped fn");
            let np_fn = namespace.get_item(np_key).expect("np swapped fn");
            let candidate = fnp_fn
                .call1((&es_a, &es_b))
                .expect("fnp swapped einsum parity");
            let base = np_fn
                .call1((&es_a, &es_b))
                .expect("numpy swapped einsum parity");
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "f16 einsum swapped-output byte parity ({row})",
            );
            bench_median_gate_python_binary(
                &mut group, bench_name, row, &np_fn, &fnp_fn, &es_a, &es_b,
            );
        }

        // 1-D dot ('j,j->') at 8M: per-8192-buffer trees in parallel, serial
        // f16 fold. Scalar output - parity assert via float16 byte equality.
        let dot_a = namespace.get_item("dot_a").expect("dot_a present");
        let dot_b = namespace.get_item("dot_b").expect("dot_b present");
        let fnp_es_d = namespace.get_item("fnp_es_d").expect("fnp_es_d present");
        let np_es_d = namespace.get_item("np_es_d").expect("np_es_d present");
        let candidate_d = fnp_es_d
            .call1((&dot_a, &dot_b))
            .expect("fnp f16 einsum dot parity");
        let base_d = np_es_d
            .call1((&dot_a, &dot_b))
            .expect("numpy f16 einsum dot parity");
        assert_eq!(
            candidate_d
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_d
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum 1-D dot byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_dot1d_8m_null_then_effect",
            "f16_einsum_dot1d_8m",
            &np_es_d,
            &fnp_es_d,
            &dot_a,
            &dot_b,
        );

        // 2-D full contraction ('ij,ij->') at 2896^2 ~ 8.4M: the coalesced
        // chunk-fold route through the generalized full-contraction parser.
        let fc_a = namespace.get_item("fc_a").expect("fc_a present");
        let fc_b = namespace.get_item("fc_b").expect("fc_b present");
        let fnp_es_fc = namespace.get_item("fnp_es_fc").expect("fnp_es_fc present");
        let np_es_fc = namespace.get_item("np_es_fc").expect("np_es_fc present");
        let candidate_fc = fnp_es_fc
            .call1((&fc_a, &fc_b))
            .expect("fnp f16 einsum full-contraction parity");
        let base_fc = np_es_fc
            .call1((&fc_a, &fc_b))
            .expect("numpy f16 einsum full-contraction parity");
        assert_eq!(
            candidate_fc
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_fc
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum full-contraction byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_fullc_2d_8m_null_then_effect",
            "f16_einsum_fullc_2d_8m",
            &np_es_fc,
            &fnp_es_fc,
            &fc_a,
            &fc_b,
        );

        // Elementwise product ('j,j->j') at 8M: zero-seeded parallel flat map.
        let fnp_es_ew = namespace.get_item("fnp_es_ew").expect("fnp_es_ew present");
        let np_es_ew = namespace.get_item("np_es_ew").expect("np_es_ew present");
        let candidate_ew = fnp_es_ew
            .call1((&dot_a, &dot_b))
            .expect("fnp f16 einsum elementwise parity");
        let base_ew = np_es_ew
            .call1((&dot_a, &dot_b))
            .expect("numpy f16 einsum elementwise parity");
        assert_eq!(
            candidate_ew
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_ew
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum elementwise byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_elemwise_8m_null_then_effect",
            "f16_einsum_elemwise_8m",
            &np_es_ew,
            &fnp_es_ew,
            &dot_a,
            &dot_b,
        );

        // f64 elementwise ('j,j->j') at 8M: the f64/f32 zero-seeded kernel.
        let ew64_a = namespace.get_item("ew64_a").expect("ew64_a present");
        let ew64_b = namespace.get_item("ew64_b").expect("ew64_b present");
        let candidate_e64 = fnp_es_ew
            .call1((&ew64_a, &ew64_b))
            .expect("fnp f64 einsum elementwise parity");
        let base_e64 = np_es_ew
            .call1((&ew64_a, &ew64_b))
            .expect("numpy f64 einsum elementwise parity");
        assert_eq!(
            candidate_e64
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_e64
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f64 einsum elementwise byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f64_einsum_elemwise_8m_null_then_effect",
            "f64_einsum_elemwise_8m",
            &np_es_ew,
            &fnp_es_ew,
            &ew64_a,
            &ew64_b,
        );

        // Batched matmul spec ('bij,bjk->bik') at (8,256,256)@(8,256,256):
        // the plain per-step chain per batch, parallel across batches.
        let bat_a = namespace.get_item("bat_a").expect("bat_a present");
        let bat_b = namespace.get_item("bat_b").expect("bat_b present");
        let fnp_es_b = namespace.get_item("fnp_es_b").expect("fnp_es_b present");
        let np_es_b = namespace.get_item("np_es_b").expect("np_es_b present");
        let candidate_b = fnp_es_b
            .call1((&bat_a, &bat_b))
            .expect("fnp f16 einsum batched parity");
        let base_b = np_es_b
            .call1((&bat_a, &bat_b))
            .expect("numpy f16 einsum batched parity");
        assert_eq!(
            candidate_b
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_b
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum batched byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_batched_8x256_null_then_effect",
            "f16_einsum_batched_8x256",
            &np_es_b,
            &fnp_es_b,
            &bat_a,
            &bat_b,
        );

        // Batched transposed spec ('bij,blj->bil'): buffered chunk-fold wide
        // trees per element, parallel across batches + row blocks.
        let fnp_es_bt = namespace.get_item("fnp_es_bt").expect("fnp_es_bt present");
        let np_es_bt = namespace.get_item("np_es_bt").expect("np_es_bt present");
        let candidate_bt = fnp_es_bt
            .call1((&bat_a, &bat_b))
            .expect("fnp f16 einsum batched-t parity");
        let base_bt = np_es_bt
            .call1((&bat_a, &bat_b))
            .expect("numpy f16 einsum batched-t parity");
        assert_eq!(
            candidate_bt
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_bt
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum batched transposed byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_batched_t_8x256_null_then_effect",
            "f16_einsum_batched_t_8x256",
            &np_es_bt,
            &fnp_es_bt,
            &bat_a,
            &bat_b,
        );

        // Batched gram spec ('bji,bjl->bil'): per-step muladd rows per batch
        // (chunk-immune per-step class). Same (8,256,256) operands.
        let fnp_es_bg = namespace.get_item("fnp_es_bg").expect("fnp_es_bg present");
        let np_es_bg = namespace.get_item("np_es_bg").expect("np_es_bg present");
        let candidate_bg = fnp_es_bg
            .call1((&bat_a, &bat_b))
            .expect("fnp f16 einsum batched-gram parity");
        let base_bg = np_es_bg
            .call1((&bat_a, &bat_b))
            .expect("numpy f16 einsum batched-gram parity");
        assert_eq!(
            candidate_bg
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_bg
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum batched gram byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_batched_g_8x256_null_then_effect",
            "f16_einsum_batched_g_8x256",
            &np_es_bg,
            &fnp_es_bg,
            &bat_a,
            &bat_b,
        );

        // Output-swapped batched forms: operand-swap arms of the batched
        // transposed/gram kernels. Rows prove the swapped dispatch engages.
        for (bench_name, row, fnp_key, np_key) in [
            (
                "f16_einsum_batched_ts_8x256_null_then_effect",
                "f16_einsum_batched_ts_8x256",
                "fnp_es_bts",
                "np_es_bts",
            ),
            (
                "f16_einsum_batched_gs_8x256_null_then_effect",
                "f16_einsum_batched_gs_8x256",
                "fnp_es_bgs",
                "np_es_bgs",
            ),
        ] {
            let fnp_fn = namespace.get_item(fnp_key).expect("fnp batched-swap fn");
            let np_fn = namespace.get_item(np_key).expect("np batched-swap fn");
            let candidate = fnp_fn
                .call1((&bat_a, &bat_b))
                .expect("fnp batched-swap parity");
            let base = np_fn
                .call1((&bat_a, &bat_b))
                .expect("numpy batched-swap parity");
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "f16 einsum batched swapped byte parity ({row})",
            );
            bench_median_gate_python_binary(
                &mut group, bench_name, row, &np_fn, &fnp_fn, &bat_a, &bat_b,
            );
        }
    });

    group.finish();
}

fn bench_wide_string_substrate_v2(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_wide_string_substrate_v2");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_wide_string_v2").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(611)\n\
                 u_a = rng.integers(97, 123, (1_000_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
                 u_fresh = rng.integers(97, 123, (500_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
                 u_b = np.concatenate([u_a[:500_000], u_fresh])\n\
                 u_union_b = rng.integers(97, 123, (1_000_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
                 s_a = rng.integers(0, 256, (1_000_000, 16), dtype=np.uint8).view('S16').reshape(-1)\n\
                 s_fresh = rng.integers(0, 256, (500_000, 16), dtype=np.uint8).view('S16').reshape(-1)\n\
                 s_b = np.concatenate([s_a[:500_000], s_fresh])\n",
            )
            .expect("wide string setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("wide string setup");
        let u_a = namespace.get_item("u_a").expect("u_a present");
        let u_b = namespace.get_item("u_b").expect("u_b present");
        let u_union_b = namespace
            .get_item("u_union_b")
            .expect("u_union_b present");
        let s_a = namespace.get_item("s_a").expect("s_a present");
        let s_b = namespace.get_item("s_b").expect("s_b present");
        let array_equal = numpy.getattr("array_equal").expect("numpy.array_equal");

        for (lhs, rhs) in [(&u_a, &u_b), (&s_a, &s_b)] {
            for op in ["unique", "union1d", "intersect1d", "setxor1d"] {
                let candidate_fn = module.getattr(op).expect("fnp op");
                let orig_fn = numpy.getattr(op).expect("numpy op");
                let candidate = if op == "unique" {
                    candidate_fn.call1((lhs,)).expect("fnp parity call")
                } else {
                    candidate_fn.call1((lhs, rhs)).expect("fnp parity call")
                };
                let orig = if op == "unique" {
                    orig_fn.call1((lhs,)).expect("numpy parity call")
                } else {
                    orig_fn.call1((lhs, rhs)).expect("numpy parity call")
                };
                assert!(
                    array_equal
                        .call1((&candidate, &orig))
                        .expect("array_equal")
                        .extract::<bool>()
                        .expect("array_equal bool"),
                    "wide string {op} parity",
                );
            }
        }
        let fnp_union_parity = module
            .getattr("union1d")
            .expect("fnp union1d parity function")
            .call1((&u_a, &u_union_b))
            .expect("fnp union1d parity call");
        let numpy_union_parity = numpy
            .getattr("union1d")
            .expect("numpy union1d parity function")
            .call1((&u_a, &u_union_b))
            .expect("numpy union1d parity call");
        assert!(
            array_equal
                .call1((&fnp_union_parity, &numpy_union_parity))
                .expect("union array_equal")
                .extract::<bool>()
                .expect("union array_equal bool"),
            "wide string disjoint union parity",
        );

        let fnp_unique = module.getattr("unique").expect("fnp unique");
        let np_unique = numpy.getattr("unique").expect("numpy unique");
        let fnp_union = module.getattr("union1d").expect("fnp union1d");
        let np_union = numpy.getattr("union1d").expect("numpy union1d");
        let fnp_intersect = module.getattr("intersect1d").expect("fnp intersect1d");
        let np_intersect = numpy.getattr("intersect1d").expect("numpy intersect1d");
        let fnp_setxor = module.getattr("setxor1d").expect("fnp setxor1d");
        let np_setxor = numpy.getattr("setxor1d").expect("numpy setxor1d");

        bench_substrate_v2_python_unary_pair(
            &mut group,
            "u16_unique_1m_paired",
            "u16_unique_1m",
            &fnp_unique,
            &np_unique,
            &u_a,
        );
        bench_substrate_v2_python_binary_pair(
            &mut group,
            "u16_union_disjoint_1m_paired",
            "u16_union_disjoint_1m",
            &fnp_union,
            &np_union,
            &u_a,
            &u_union_b,
        );
        bench_substrate_v2_python_binary_pair(
            &mut group,
            "u16_setxor_1m_paired",
            "u16_setxor_1m",
            &fnp_setxor,
            &np_setxor,
            &u_a,
            &u_b,
        );
        bench_substrate_v2_python_unary_pair(
            &mut group,
            "s16_unique_1m_paired",
            "s16_unique_1m",
            &fnp_unique,
            &np_unique,
            &s_a,
        );
        bench_substrate_v2_python_binary_pair(
            &mut group,
            "s16_intersect_1m_paired",
            "s16_intersect_1m",
            &fnp_intersect,
            &np_intersect,
            &s_a,
            &s_b,
        );
        bench_substrate_v2_python_binary_pair(
            &mut group,
            "s16_setxor_1m_paired",
            "s16_setxor_1m",
            &fnp_setxor,
            &np_setxor,
            &s_a,
            &s_b,
        );
    });

    group.finish();
}

fn bench_ledger_integrity_rejects(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_ledger_integrity_rejects");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(6));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_ledger_audit").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");

        {
            let namespace = PyDict::new(py);
            py.run(
                std::ffi::CString::new(
                    "import numpy as np\n\
                     rng = np.random.default_rng(0)\n\
                     median_input = rng.standard_normal(16_000_000).astype(np.float64)\n",
                )
                .expect("median setup CString")
                .as_c_str(),
                Some(&namespace),
                Some(&namespace),
            )
            .expect("median setup");
            let input = namespace
                .get_item("median_input")
                .expect("median input present");
            let raw: Vec<u8> = input
                .call_method0("tobytes")
                .expect("median bytes")
                .extract()
                .expect("extract median bytes");
            let data: Vec<f64> = raw
                .chunks_exact(8)
                .map(|chunk| {
                    f64::from_ne_bytes(chunk.try_into().expect("one native f64 per chunk"))
                })
                .collect();
            assert_eq!(data.len(), 16_000_000);
            let numpy_median = numpy.getattr("median").expect("numpy.median");
            let candidate = ledger_radix_median_f64(&data);
            let orig = ledger_orig_median_reference(&numpy_median, &input)
                .expect("NumPy median parity reference");
            assert_eq!(candidate.to_bits(), orig.to_bits(), "radix median parity");

            let candidate_samples = RefCell::new(Vec::new());
            let orig_samples = RefCell::new(Vec::new());
            let order = Cell::new(0u64);
            group.bench_function("radix_median_f64_normal_16m_paired", |bench| {
                bench.iter_custom(|iterations| {
                    let mut candidate_total = Duration::ZERO;
                    let mut orig_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let orig_first = order.get() & 1 == 1;
                        order.set(order.get().wrapping_add(1));
                        if orig_first {
                            let start = Instant::now();
                            black_box(
                                ledger_orig_median_reference(&numpy_median, &input)
                                    .expect("NumPy median audit call"),
                            );
                            orig_total += start.elapsed();
                            let start = Instant::now();
                            black_box(ledger_radix_median_f64(&data));
                            candidate_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(ledger_radix_median_f64(&data));
                            candidate_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                ledger_orig_median_reference(&numpy_median, &input)
                                    .expect("NumPy median audit call"),
                            );
                            orig_total += start.elapsed();
                        }
                    }
                    candidate_samples
                        .borrow_mut()
                        .push(candidate_total.as_secs_f64() * 1e9 / iterations as f64);
                    orig_samples
                        .borrow_mut()
                        .push(orig_total.as_secs_f64() * 1e9 / iterations as f64);
                    candidate_total + orig_total
                });
            });
            report_ledger_pair(
                "radix_median_f64_normal_16m",
                &candidate_samples,
                &orig_samples,
            );
        }

        {
            let namespace = PyDict::new(py);
            py.run(
                std::ffi::CString::new(
                    "import numpy as np\n\
                     rng = np.random.default_rng(0)\n\
                     f16_input = (rng.integers(1, 4000, 4_000_000) / 7).astype(np.float16)\n",
                )
                .expect("f16 setup CString")
                .as_c_str(),
                Some(&namespace),
                Some(&namespace),
            )
            .expect("f16 setup");
            let input = namespace.get_item("f16_input").expect("f16 input present");
            let bit_bytes: Vec<u8> = input
                .call_method1("view", ("uint16",))
                .expect("f16 uint16 view")
                .call_method0("tobytes")
                .expect("f16 bit bytes")
                .extract()
                .expect("extract f16 bit bytes");
            let input_bits: Vec<u16> = bit_bytes
                .chunks_exact(2)
                .map(|chunk| {
                    u16::from_ne_bytes(chunk.try_into().expect("one native u16 per chunk"))
                })
                .collect();
            assert_eq!(input_bits.len(), 4_000_000);
            let numpy_sort = numpy.getattr("sort").expect("numpy.sort");
            let equal = numpy.getattr("array_equal").expect("numpy.array_equal");
            let candidate = ledger_try_native_f16_sort(&numpy_sort, &input, &input_bits)
                .expect("f16 widening candidate parity call");
            let orig =
                ledger_orig_f16_sort_reference(&numpy_sort, &input).expect("f16 ORIG parity call");
            assert!(
                equal
                    .call1((candidate.bind(py), orig.bind(py)))
                    .expect("f16 array_equal")
                    .extract::<bool>()
                    .expect("f16 equality bool"),
                "f16 widening-sort parity",
            );

            let candidate_samples = RefCell::new(Vec::new());
            let orig_samples = RefCell::new(Vec::new());
            let order = Cell::new(0u64);
            group.bench_function("f16_sort_via_f32_widening_4m_paired", |bench| {
                bench.iter_custom(|iterations| {
                    let mut candidate_total = Duration::ZERO;
                    let mut orig_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let orig_first = order.get() & 1 == 1;
                        order.set(order.get().wrapping_add(1));
                        if orig_first {
                            let start = Instant::now();
                            black_box(
                                ledger_orig_f16_sort_reference(&numpy_sort, &input)
                                    .expect("f16 ORIG audit call"),
                            );
                            orig_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                ledger_try_native_f16_sort(&numpy_sort, &input, &input_bits)
                                    .expect("f16 widening audit call"),
                            );
                            candidate_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(
                                ledger_try_native_f16_sort(&numpy_sort, &input, &input_bits)
                                    .expect("f16 widening audit call"),
                            );
                            candidate_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                ledger_orig_f16_sort_reference(&numpy_sort, &input)
                                    .expect("f16 ORIG audit call"),
                            );
                            orig_total += start.elapsed();
                        }
                    }
                    candidate_samples
                        .borrow_mut()
                        .push(candidate_total.as_secs_f64() * 1e9 / iterations as f64);
                    orig_samples
                        .borrow_mut()
                        .push(orig_total.as_secs_f64() * 1e9 / iterations as f64);
                    candidate_total + orig_total
                });
            });
            report_ledger_pair(
                "f16_sort_via_f32_widening_4m",
                &candidate_samples,
                &orig_samples,
            );

            // PRODUCTION arm (bead deadlock-audit-98chw): fnp.sort(f16) now routes through
            // try_native_f16_sort_flat for this input; paired vs numpy.sort in the same
            // interleaved routine, plus an A/A null-control row (per-function noise floor).
            let fnp_sort = module.getattr("sort").expect("fnp sort");
            let prod = fnp_sort.call1((&input,)).expect("fnp f16 sort parity call");
            let prod_bytes: Vec<u8> = prod
                .call_method0("tobytes")
                .expect("prod bytes")
                .extract()
                .expect("extract prod bytes");
            let orig_bytes: Vec<u8> = orig
                .bind(py)
                .call_method0("tobytes")
                .expect("orig bytes")
                .extract()
                .expect("extract orig bytes");
            assert_eq!(prod_bytes, orig_bytes, "production f16 sort parity (tobytes)");

            let prod_samples = RefCell::new(Vec::new());
            let prod_orig_samples = RefCell::new(Vec::new());
            let prod_order = Cell::new(0u64);
            group.bench_function("f16_sort_production_4m_paired", |bench| {
                bench.iter_custom(|iterations| {
                    let mut cand_total = Duration::ZERO;
                    let mut orig_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let orig_first = prod_order.get() & 1 == 1;
                        prod_order.set(prod_order.get().wrapping_add(1));
                        if orig_first {
                            let start = Instant::now();
                            black_box(numpy_sort.call1((&input,)).expect("numpy f16 sort"));
                            orig_total += start.elapsed();
                            let start = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("fnp f16 sort"));
                            cand_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("fnp f16 sort"));
                            cand_total += start.elapsed();
                            let start = Instant::now();
                            black_box(numpy_sort.call1((&input,)).expect("numpy f16 sort"));
                            orig_total += start.elapsed();
                        }
                    }
                    prod_samples
                        .borrow_mut()
                        .push(cand_total.as_secs_f64() * 1e9 / iterations as f64);
                    prod_orig_samples
                        .borrow_mut()
                        .push(orig_total.as_secs_f64() * 1e9 / iterations as f64);
                    cand_total + orig_total
                });
            });
            report_ledger_pair(
                "f16_sort_production_4m",
                &prod_samples,
                &prod_orig_samples,
            );

            let null_a = RefCell::new(Vec::new());
            let null_b = RefCell::new(Vec::new());
            let null_order = Cell::new(0u64);
            group.bench_function("f16_sort_production_4m_null_control", |bench| {
                bench.iter_custom(|iterations| {
                    let mut a_total = Duration::ZERO;
                    let mut b_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let b_first = null_order.get() & 1 == 1;
                        null_order.set(null_order.get().wrapping_add(1));
                        if b_first {
                            let start = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("null b"));
                            b_total += start.elapsed();
                            let start = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("null a"));
                            a_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("null a"));
                            a_total += start.elapsed();
                            let start = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("null b"));
                            b_total += start.elapsed();
                        }
                    }
                    null_a
                        .borrow_mut()
                        .push(a_total.as_secs_f64() * 1e9 / iterations as f64);
                    null_b
                        .borrow_mut()
                        .push(b_total.as_secs_f64() * 1e9 / iterations as f64);
                    a_total + b_total
                });
            });
            report_ledger_pair("f16_sort_production_null_AA", &null_a, &null_b);

            // f16 STABLE ARGSORT production arm (widened stable radix; sibling lever): the
            // same 4M f16 input is tie-dense by construction (63k distinct values), the case
            // where stability is load-bearing. Paired vs numpy + A/A null control.
            let fnp_argsort = module.getattr("argsort").expect("fnp argsort");
            let numpy_argsort = numpy.getattr("argsort").expect("numpy argsort");
            let stable_kw = pyo3::types::PyDict::new(py);
            stable_kw.set_item("kind", "stable").expect("kind kwarg");
            let ag_prod = fnp_argsort
                .call((&input,), Some(&stable_kw))
                .expect("fnp f16 stable argsort parity call");
            let ag_orig = numpy_argsort
                .call((&input,), Some(&stable_kw))
                .expect("numpy f16 stable argsort parity call");
            assert!(
                equal
                    .call1((&ag_prod, &ag_orig))
                    .expect("f16 argsort array_equal")
                    .extract::<bool>()
                    .expect("f16 argsort equality bool"),
                "production f16 stable argsort parity",
            );

            let ag_samples = RefCell::new(Vec::new());
            let ag_orig_samples = RefCell::new(Vec::new());
            let ag_order = Cell::new(0u64);
            group.bench_function("f16_argsort_stable_production_4m_paired", |bench| {
                bench.iter_custom(|iterations| {
                    let mut cand_total = Duration::ZERO;
                    let mut orig_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let orig_first = ag_order.get() & 1 == 1;
                        ag_order.set(ag_order.get().wrapping_add(1));
                        if orig_first {
                            let start = Instant::now();
                            black_box(
                                numpy_argsort
                                    .call((&input,), Some(&stable_kw))
                                    .expect("numpy f16 argsort"),
                            );
                            orig_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                fnp_argsort
                                    .call((&input,), Some(&stable_kw))
                                    .expect("fnp f16 argsort"),
                            );
                            cand_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(
                                fnp_argsort
                                    .call((&input,), Some(&stable_kw))
                                    .expect("fnp f16 argsort"),
                            );
                            cand_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                numpy_argsort
                                    .call((&input,), Some(&stable_kw))
                                    .expect("numpy f16 argsort"),
                            );
                            orig_total += start.elapsed();
                        }
                    }
                    ag_samples
                        .borrow_mut()
                        .push(cand_total.as_secs_f64() * 1e9 / iterations as f64);
                    ag_orig_samples
                        .borrow_mut()
                        .push(orig_total.as_secs_f64() * 1e9 / iterations as f64);
                    cand_total + orig_total
                });
            });
            report_ledger_pair(
                "f16_argsort_stable_production_4m",
                &ag_samples,
                &ag_orig_samples,
            );

            let ag_null_a = RefCell::new(Vec::new());
            let ag_null_b = RefCell::new(Vec::new());
            let ag_null_order = Cell::new(0u64);
            group.bench_function("f16_argsort_stable_production_4m_null_control", |bench| {
                bench.iter_custom(|iterations| {
                    let mut a_total = Duration::ZERO;
                    let mut b_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let b_first = ag_null_order.get() & 1 == 1;
                        ag_null_order.set(ag_null_order.get().wrapping_add(1));
                        if b_first {
                            let start = Instant::now();
                            black_box(
                                fnp_argsort
                                    .call((&input,), Some(&stable_kw))
                                    .expect("null b"),
                            );
                            b_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                fnp_argsort
                                    .call((&input,), Some(&stable_kw))
                                    .expect("null a"),
                            );
                            a_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(
                                fnp_argsort
                                    .call((&input,), Some(&stable_kw))
                                    .expect("null a"),
                            );
                            a_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                fnp_argsort
                                    .call((&input,), Some(&stable_kw))
                                    .expect("null b"),
                            );
                            b_total += start.elapsed();
                        }
                    }
                    ag_null_a
                        .borrow_mut()
                        .push(a_total.as_secs_f64() * 1e9 / iterations as f64);
                    ag_null_b
                        .borrow_mut()
                        .push(b_total.as_secs_f64() * 1e9 / iterations as f64);
                    a_total + b_total
                });
            });
            report_ledger_pair("f16_argsort_stable_null_AA", &ag_null_a, &ag_null_b);

            // LAST-AXIS siblings (2000x2000 view of the same 4M input): per-lane widened
            // value sort + per-lane widened stable argsort, paired with A/A null controls.
            let input2d = input
                .call_method1("reshape", ((2000, 2000),))
                .expect("reshape 2000x2000");
            let axis_kw = pyo3::types::PyDict::new(py);
            axis_kw.set_item("axis", -1).expect("axis kwarg");
            let fnp_sort2 = module.getattr("sort").expect("fnp sort");
            let numpy_sort2 = numpy.getattr("sort").expect("numpy sort");
            let s2_f = fnp_sort2
                .call((&input2d,), Some(&axis_kw))
                .expect("fnp f16 lastaxis sort parity");
            let s2_n = numpy_sort2
                .call((&input2d,), Some(&axis_kw))
                .expect("numpy f16 lastaxis sort parity");
            let s2_fb: Vec<u8> = s2_f
                .call_method0("tobytes")
                .expect("bytes")
                .extract()
                .expect("extract");
            let s2_nb: Vec<u8> = s2_n
                .call_method0("tobytes")
                .expect("bytes")
                .extract()
                .expect("extract");
            assert_eq!(s2_fb, s2_nb, "f16 lastaxis sort parity (tobytes)");
            let stable_axis_kw = pyo3::types::PyDict::new(py);
            stable_axis_kw.set_item("axis", -1).expect("axis kwarg");
            stable_axis_kw.set_item("kind", "stable").expect("kind kwarg");
            let a2_f = fnp_argsort
                .call((&input2d,), Some(&stable_axis_kw))
                .expect("fnp f16 lastaxis argsort parity");
            let a2_n = numpy_argsort
                .call((&input2d,), Some(&stable_axis_kw))
                .expect("numpy f16 lastaxis argsort parity");
            assert!(
                equal
                    .call1((&a2_f, &a2_n))
                    .expect("array_equal")
                    .extract::<bool>()
                    .expect("bool"),
                "f16 lastaxis stable argsort parity",
            );

            for (label, fnp_fn, numpy_fn, kw) in [
                (
                    "f16_sort_lastaxis_2000x2000",
                    &fnp_sort2,
                    &numpy_sort2,
                    &axis_kw,
                ),
                (
                    "f16_argsort_stable_lastaxis_2000x2000",
                    &fnp_argsort,
                    &numpy_argsort,
                    &stable_axis_kw,
                ),
            ] {
                let cand = RefCell::new(Vec::new());
                let orig = RefCell::new(Vec::new());
                let ord = Cell::new(0u64);
                group.bench_function(format!("{label}_paired"), |bench| {
                    bench.iter_custom(|iterations| {
                        let mut ct = Duration::ZERO;
                        let mut ot = Duration::ZERO;
                        for _ in 0..iterations {
                            let of = ord.get() & 1 == 1;
                            ord.set(ord.get().wrapping_add(1));
                            if of {
                                let s = Instant::now();
                                black_box(numpy_fn.call((&input2d,), Some(kw)).expect("orig"));
                                ot += s.elapsed();
                                let s = Instant::now();
                                black_box(fnp_fn.call((&input2d,), Some(kw)).expect("cand"));
                                ct += s.elapsed();
                            } else {
                                let s = Instant::now();
                                black_box(fnp_fn.call((&input2d,), Some(kw)).expect("cand"));
                                ct += s.elapsed();
                                let s = Instant::now();
                                black_box(numpy_fn.call((&input2d,), Some(kw)).expect("orig"));
                                ot += s.elapsed();
                            }
                        }
                        cand.borrow_mut()
                            .push(ct.as_secs_f64() * 1e9 / iterations as f64);
                        orig.borrow_mut()
                            .push(ot.as_secs_f64() * 1e9 / iterations as f64);
                        ct + ot
                    });
                });
                report_ledger_pair(label, &cand, &orig);

                let na = RefCell::new(Vec::new());
                let nb = RefCell::new(Vec::new());
                let nord = Cell::new(0u64);
                group.bench_function(format!("{label}_null_control"), |bench| {
                    bench.iter_custom(|iterations| {
                        let mut at = Duration::ZERO;
                        let mut bt = Duration::ZERO;
                        for _ in 0..iterations {
                            let bf = nord.get() & 1 == 1;
                            nord.set(nord.get().wrapping_add(1));
                            if bf {
                                let s = Instant::now();
                                black_box(fnp_fn.call((&input2d,), Some(kw)).expect("nb"));
                                bt += s.elapsed();
                                let s = Instant::now();
                                black_box(fnp_fn.call((&input2d,), Some(kw)).expect("na"));
                                at += s.elapsed();
                            } else {
                                let s = Instant::now();
                                black_box(fnp_fn.call((&input2d,), Some(kw)).expect("na"));
                                at += s.elapsed();
                                let s = Instant::now();
                                black_box(fnp_fn.call((&input2d,), Some(kw)).expect("nb"));
                                bt += s.elapsed();
                            }
                        }
                        na.borrow_mut()
                            .push(at.as_secs_f64() * 1e9 / iterations as f64);
                        nb.borrow_mut()
                            .push(bt.as_secs_f64() * 1e9 / iterations as f64);
                        at + bt
                    });
                });
                report_ledger_pair(&format!("{label}_null_AA"), &na, &nb);
            }
        }

        {
            // Narrow-int counting sort: i16 8M full-range (the 2-byte case is where numpy's
            // serial radix is slowest). Paired vs numpy + A/A null control.
            let namespace = PyDict::new(py);
            py.run(
                std::ffi::CString::new(
                    "import numpy as np\n\
                     rng = np.random.default_rng(5)\n\
                     i16_input = rng.integers(-32768, 32768, 8_000_000, dtype=np.int16)\n",
                )
                .expect("i16 setup CString")
                .as_c_str(),
                Some(&namespace),
                Some(&namespace),
            )
            .expect("i16 setup");
            let input = namespace.get_item("i16_input").expect("i16 input");
            let fnp_sort = module.getattr("sort").expect("fnp sort");
            let numpy_sort = numpy.getattr("sort").expect("numpy sort");
            let equal = numpy.getattr("array_equal").expect("array_equal");
            let f = fnp_sort.call1((&input,)).expect("fnp i16 sort parity");
            let nres = numpy_sort.call1((&input,)).expect("numpy i16 sort parity");
            assert!(
                equal
                    .call1((&f, &nres))
                    .expect("array_equal")
                    .extract::<bool>()
                    .expect("bool"),
                "narrow-int i16 sort parity",
            );

            let cand = RefCell::new(Vec::new());
            let orig = RefCell::new(Vec::new());
            let ord = Cell::new(0u64);
            group.bench_function("narrow_int_i16_sort_8m_paired", |bench| {
                bench.iter_custom(|iterations| {
                    let mut ct = Duration::ZERO;
                    let mut ot = Duration::ZERO;
                    for _ in 0..iterations {
                        let of = ord.get() & 1 == 1;
                        ord.set(ord.get().wrapping_add(1));
                        if of {
                            let s = Instant::now();
                            black_box(numpy_sort.call1((&input,)).expect("orig"));
                            ot += s.elapsed();
                            let s = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("cand"));
                            ct += s.elapsed();
                        } else {
                            let s = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("cand"));
                            ct += s.elapsed();
                            let s = Instant::now();
                            black_box(numpy_sort.call1((&input,)).expect("orig"));
                            ot += s.elapsed();
                        }
                    }
                    cand.borrow_mut()
                        .push(ct.as_secs_f64() * 1e9 / iterations as f64);
                    orig.borrow_mut()
                        .push(ot.as_secs_f64() * 1e9 / iterations as f64);
                    ct + ot
                });
            });
            report_ledger_pair("narrow_int_i16_sort_8m", &cand, &orig);

            let na = RefCell::new(Vec::new());
            let nb = RefCell::new(Vec::new());
            let nord = Cell::new(0u64);
            group.bench_function("narrow_int_i16_sort_8m_null_control", |bench| {
                bench.iter_custom(|iterations| {
                    let mut at = Duration::ZERO;
                    let mut bt = Duration::ZERO;
                    for _ in 0..iterations {
                        let bf = nord.get() & 1 == 1;
                        nord.set(nord.get().wrapping_add(1));
                        if bf {
                            let s = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("nb"));
                            bt += s.elapsed();
                            let s = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("na"));
                            at += s.elapsed();
                        } else {
                            let s = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("na"));
                            at += s.elapsed();
                            let s = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("nb"));
                            bt += s.elapsed();
                        }
                    }
                    na.borrow_mut()
                        .push(at.as_secs_f64() * 1e9 / iterations as f64);
                    nb.borrow_mut()
                        .push(bt.as_secs_f64() * 1e9 / iterations as f64);
                    at + bt
                });
            });
            report_ledger_pair("narrow_int_i16_sort_null_AA", &na, &nb);

            // Stable ARGSORT sibling on the same 8M i16 input (dense ties by construction;
            // routes to the parallel counting-prefix stable argsort). Paired + A/A null.
            let fnp_argsort_n = module.getattr("argsort").expect("fnp argsort");
            let numpy_argsort_n = numpy.getattr("argsort").expect("numpy argsort");
            let skw = pyo3::types::PyDict::new(py);
            skw.set_item("kind", "stable").expect("kind kwarg");
            let af = fnp_argsort_n
                .call((&input,), Some(&skw))
                .expect("fnp i16 stable argsort parity");
            let an = numpy_argsort_n
                .call((&input,), Some(&skw))
                .expect("numpy i16 stable argsort parity");
            assert!(
                equal
                    .call1((&af, &an))
                    .expect("array_equal")
                    .extract::<bool>()
                    .expect("bool"),
                "narrow-int i16 stable argsort parity",
            );
            let cand2 = RefCell::new(Vec::new());
            let orig2 = RefCell::new(Vec::new());
            let ord2 = Cell::new(0u64);
            group.bench_function("narrow_int_i16_argsort_stable_8m_paired", |bench| {
                bench.iter_custom(|iterations| {
                    let mut ct = Duration::ZERO;
                    let mut ot = Duration::ZERO;
                    for _ in 0..iterations {
                        let of = ord2.get() & 1 == 1;
                        ord2.set(ord2.get().wrapping_add(1));
                        if of {
                            let s = Instant::now();
                            black_box(numpy_argsort_n.call((&input,), Some(&skw)).expect("orig"));
                            ot += s.elapsed();
                            let s = Instant::now();
                            black_box(fnp_argsort_n.call((&input,), Some(&skw)).expect("cand"));
                            ct += s.elapsed();
                        } else {
                            let s = Instant::now();
                            black_box(fnp_argsort_n.call((&input,), Some(&skw)).expect("cand"));
                            ct += s.elapsed();
                            let s = Instant::now();
                            black_box(numpy_argsort_n.call((&input,), Some(&skw)).expect("orig"));
                            ot += s.elapsed();
                        }
                    }
                    cand2
                        .borrow_mut()
                        .push(ct.as_secs_f64() * 1e9 / iterations as f64);
                    orig2
                        .borrow_mut()
                        .push(ot.as_secs_f64() * 1e9 / iterations as f64);
                    ct + ot
                });
            });
            report_ledger_pair("narrow_int_i16_argsort_stable_8m", &cand2, &orig2);

            let na2 = RefCell::new(Vec::new());
            let nb2 = RefCell::new(Vec::new());
            let nord2 = Cell::new(0u64);
            group.bench_function("narrow_int_i16_argsort_stable_8m_null_control", |bench| {
                bench.iter_custom(|iterations| {
                    let mut at = Duration::ZERO;
                    let mut bt = Duration::ZERO;
                    for _ in 0..iterations {
                        let bf = nord2.get() & 1 == 1;
                        nord2.set(nord2.get().wrapping_add(1));
                        if bf {
                            let s = Instant::now();
                            black_box(fnp_argsort_n.call((&input,), Some(&skw)).expect("nb"));
                            bt += s.elapsed();
                            let s = Instant::now();
                            black_box(fnp_argsort_n.call((&input,), Some(&skw)).expect("na"));
                            at += s.elapsed();
                        } else {
                            let s = Instant::now();
                            black_box(fnp_argsort_n.call((&input,), Some(&skw)).expect("na"));
                            at += s.elapsed();
                            let s = Instant::now();
                            black_box(fnp_argsort_n.call((&input,), Some(&skw)).expect("nb"));
                            bt += s.elapsed();
                        }
                    }
                    na2.borrow_mut()
                        .push(at.as_secs_f64() * 1e9 / iterations as f64);
                    nb2.borrow_mut()
                        .push(bt.as_secs_f64() * 1e9 / iterations as f64);
                    at + bt
                });
            });
            report_ledger_pair("narrow_int_i16_argsort_stable_null_AA", &na2, &nb2);

            // LAST-AXIS siblings on a 4000x2000 view of the same 8M i16 input: per-lane sort
            // + per-lane stable argsort, paired with A/A null controls.
            let input2d = input
                .call_method1("reshape", ((4000, 2000),))
                .expect("reshape 4000x2000");
            let axkw = pyo3::types::PyDict::new(py);
            axkw.set_item("axis", -1).expect("axis kwarg");
            let stax_kw = pyo3::types::PyDict::new(py);
            stax_kw.set_item("axis", -1).expect("axis kwarg");
            stax_kw.set_item("kind", "stable").expect("kind kwarg");
            let sf = fnp_sort.call((&input2d,), Some(&axkw)).expect("fnp lastaxis parity");
            let sn = numpy_sort
                .call((&input2d,), Some(&axkw))
                .expect("numpy lastaxis parity");
            let sfb: Vec<u8> = sf.call_method0("tobytes").expect("b").extract().expect("e");
            let snb: Vec<u8> = sn.call_method0("tobytes").expect("b").extract().expect("e");
            assert_eq!(sfb, snb, "narrow-int i16 lastaxis sort parity");
            let gf = fnp_argsort_n
                .call((&input2d,), Some(&stax_kw))
                .expect("fnp lastaxis argsort parity");
            let gn = numpy_argsort_n
                .call((&input2d,), Some(&stax_kw))
                .expect("numpy lastaxis argsort parity");
            assert!(
                equal
                    .call1((&gf, &gn))
                    .expect("array_equal")
                    .extract::<bool>()
                    .expect("bool"),
                "narrow-int i16 lastaxis stable argsort parity",
            );
            for (label, ffn, nfn, kw) in [
                (
                    "narrow_int_i16_sort_lastaxis_4000x2000",
                    &fnp_sort,
                    &numpy_sort,
                    &axkw,
                ),
                (
                    "narrow_int_i16_argsort_stable_lastaxis_4000x2000",
                    &fnp_argsort_n,
                    &numpy_argsort_n,
                    &stax_kw,
                ),
            ] {
                let cand = RefCell::new(Vec::new());
                let orig = RefCell::new(Vec::new());
                let ord = Cell::new(0u64);
                group.bench_function(format!("{label}_paired"), |bench| {
                    bench.iter_custom(|iterations| {
                        let mut ct = Duration::ZERO;
                        let mut ot = Duration::ZERO;
                        for _ in 0..iterations {
                            let of = ord.get() & 1 == 1;
                            ord.set(ord.get().wrapping_add(1));
                            if of {
                                let s = Instant::now();
                                black_box(nfn.call((&input2d,), Some(kw)).expect("orig"));
                                ot += s.elapsed();
                                let s = Instant::now();
                                black_box(ffn.call((&input2d,), Some(kw)).expect("cand"));
                                ct += s.elapsed();
                            } else {
                                let s = Instant::now();
                                black_box(ffn.call((&input2d,), Some(kw)).expect("cand"));
                                ct += s.elapsed();
                                let s = Instant::now();
                                black_box(nfn.call((&input2d,), Some(kw)).expect("orig"));
                                ot += s.elapsed();
                            }
                        }
                        cand.borrow_mut()
                            .push(ct.as_secs_f64() * 1e9 / iterations as f64);
                        orig.borrow_mut()
                            .push(ot.as_secs_f64() * 1e9 / iterations as f64);
                        ct + ot
                    });
                });
                report_ledger_pair(label, &cand, &orig);
                let na = RefCell::new(Vec::new());
                let nb = RefCell::new(Vec::new());
                let nord = Cell::new(0u64);
                group.bench_function(format!("{label}_null_control"), |bench| {
                    bench.iter_custom(|iterations| {
                        let mut at = Duration::ZERO;
                        let mut bt = Duration::ZERO;
                        for _ in 0..iterations {
                            let bf = nord.get() & 1 == 1;
                            nord.set(nord.get().wrapping_add(1));
                            if bf {
                                let s = Instant::now();
                                black_box(ffn.call((&input2d,), Some(kw)).expect("nb"));
                                bt += s.elapsed();
                                let s = Instant::now();
                                black_box(ffn.call((&input2d,), Some(kw)).expect("na"));
                                at += s.elapsed();
                            } else {
                                let s = Instant::now();
                                black_box(ffn.call((&input2d,), Some(kw)).expect("na"));
                                at += s.elapsed();
                                let s = Instant::now();
                                black_box(ffn.call((&input2d,), Some(kw)).expect("nb"));
                                bt += s.elapsed();
                            }
                        }
                        na.borrow_mut()
                            .push(at.as_secs_f64() * 1e9 / iterations as f64);
                        nb.borrow_mut()
                            .push(bt.as_secs_f64() * 1e9 / iterations as f64);
                        at + bt
                    });
                });
                report_ledger_pair(&format!("{label}_null_AA"), &na, &nb);
            }
        }

        {
            let namespace = PyDict::new(py);
            py.run(
                std::ffi::CString::new(
                    "import numpy as np\n\
                     rng = np.random.default_rng(0)\n\
                     f32_ties = np.round(rng.standard_normal(2_000_000), 2).astype(np.float32)\n",
                )
                .expect("f32 argsort setup CString")
                .as_c_str(),
                Some(&namespace),
                Some(&namespace),
            )
            .expect("f32 argsort setup");
            let input = namespace
                .get_item("f32_ties")
                .expect("f32 tie input present");
            let fnp_argsort = module.getattr("argsort").expect("fnp argsort");
            let numpy_argsort = numpy.getattr("argsort").expect("numpy.argsort");
            let equal = numpy.getattr("array_equal").expect("numpy.array_equal");
            let candidate = ledger_f32_tie_argsort_candidate(&fnp_argsort, &input)
                .expect("f32 tied candidate parity call");
            let orig = ledger_orig_f32_argsort_reference(&numpy_argsort, &input)
                .expect("f32 tied ORIG parity call");
            assert!(
                equal
                    .call1((candidate.bind(py), orig.bind(py)))
                    .expect("f32 argsort array_equal")
                    .extract::<bool>()
                    .expect("f32 argsort equality bool"),
                "tied f32 argsort parity",
            );

            let candidate_samples = RefCell::new(Vec::new());
            let orig_samples = RefCell::new(Vec::new());
            let order = Cell::new(0u64);
            group.bench_function("f32_argsort_rounded_ties_2m_paired", |bench| {
                bench.iter_custom(|iterations| {
                    let mut candidate_total = Duration::ZERO;
                    let mut orig_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let orig_first = order.get() & 1 == 1;
                        order.set(order.get().wrapping_add(1));
                        if orig_first {
                            let start = Instant::now();
                            black_box(
                                ledger_orig_f32_argsort_reference(&numpy_argsort, &input)
                                    .expect("f32 argsort ORIG audit call"),
                            );
                            orig_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                ledger_f32_tie_argsort_candidate(&fnp_argsort, &input)
                                    .expect("f32 argsort candidate audit call"),
                            );
                            candidate_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(
                                ledger_f32_tie_argsort_candidate(&fnp_argsort, &input)
                                    .expect("f32 argsort candidate audit call"),
                            );
                            candidate_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                ledger_orig_f32_argsort_reference(&numpy_argsort, &input)
                                    .expect("f32 argsort ORIG audit call"),
                            );
                            orig_total += start.elapsed();
                        }
                    }
                    candidate_samples
                        .borrow_mut()
                        .push(candidate_total.as_secs_f64() * 1e9 / iterations as f64);
                    orig_samples
                        .borrow_mut()
                        .push(orig_total.as_secs_f64() * 1e9 / iterations as f64);
                    candidate_total + orig_total
                });
            });
            report_ledger_pair(
                "f32_argsort_rounded_ties_2m",
                &candidate_samples,
                &orig_samples,
            );

            // NULL CONTROL (A/A): the candidate arm registered twice in the same interleaved
            // routine. Its ratio and cv are the harness noise floor - any lever effect below
            // this floor is undecidable on this harness (franken_whisper null-control rule).
            let null_a = RefCell::new(Vec::new());
            let null_b = RefCell::new(Vec::new());
            let null_order = Cell::new(0u64);
            group.bench_function("f32_argsort_rounded_ties_2m_null_control", |bench| {
                bench.iter_custom(|iterations| {
                    let mut a_total = Duration::ZERO;
                    let mut b_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let b_first = null_order.get() & 1 == 1;
                        null_order.set(null_order.get().wrapping_add(1));
                        if b_first {
                            let start = Instant::now();
                            black_box(
                                ledger_f32_tie_argsort_candidate(&fnp_argsort, &input)
                                    .expect("null-control arm b"),
                            );
                            b_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                ledger_f32_tie_argsort_candidate(&fnp_argsort, &input)
                                    .expect("null-control arm a"),
                            );
                            a_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(
                                ledger_f32_tie_argsort_candidate(&fnp_argsort, &input)
                                    .expect("null-control arm a"),
                            );
                            a_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                ledger_f32_tie_argsort_candidate(&fnp_argsort, &input)
                                    .expect("null-control arm b"),
                            );
                            b_total += start.elapsed();
                        }
                    }
                    null_a
                        .borrow_mut()
                        .push(a_total.as_secs_f64() * 1e9 / iterations as f64);
                    null_b
                        .borrow_mut()
                        .push(b_total.as_secs_f64() * 1e9 / iterations as f64);
                    a_total + b_total
                });
            });
            report_ledger_pair("f32_argsort_null_control_AA", &null_a, &null_b);

            // Self-time of the pre-check unit the dispatch dedupe removes: ONE full parallel
            // NaN scan + ONE 65,536-sample strided tie oracle over the same 2M f32 buffer
            // (bench-local reconstruction of the dispatch's NaN scan + argsort_sample_has_tie;
            // before the fix, dense-tie input paid this unit TWICE - radix candidate then
            // comparison candidate - before delegation).
            let raw: Vec<u8> = input
                .call_method0("tobytes")
                .expect("f32 tie bytes")
                .extract()
                .expect("extract f32 tie bytes");
            let data: Vec<f32> = raw
                .chunks_exact(4)
                .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("one native f32")))
                .collect();
            group.bench_function("f32_argsort_tie_precheck_selftime_2m", |bench| {
                bench.iter(|| {
                    use rayon::prelude::*;
                    let d = black_box(&data);
                    let nan = d.par_iter().any(|v| v.is_nan());
                    const TIE_SAMPLE: usize = 1 << 16;
                    let n = d.len();
                    let k = n.min(TIE_SAMPLE);
                    let stride = (n / k).max(1);
                    let mut sample: Vec<f32> = (0..k).map(|i| d[i * stride]).collect();
                    sample.sort_unstable_by(|a, b| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let tie = (1..sample.len()).any(|i| sample[i] == sample[i - 1]);
                    black_box((nan, tie))
                });
            });
        }
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_wide_string_sort_median_gate,
    bench_accumulate_extremum_median_gate,
    bench_int_convolve_median_gate,
    bench_completion_median_gate,
    bench_f64_transcendental_median_gate,
    bench_f64_exp_log_probe,
    bench_f64_exp_log_median_gate,
    bench_bool_sort_median_gate,
    bench_int_matmul_median_gate,
    bench_f16_matmul_median_gate,
    bench_multidot_median_gate,
    bench_f16_einsum_median_gate,
    bench_wide_string_substrate_v2,
    bench_ledger_integrity_rejects,
    bench_unique_rows_full_boundary,
    bench_unique_cols_boundary,
    bench_unique_rows_boundary,
    bench_lexsort_boundary,
    bench_unique_rows_narrow_int_boundary,
    bench_nanarg_lastaxis_boundary,
    bench_nanarg_nonlast_boundary,
    bench_argextreme_f32_axis_boundary,
    bench_datetime_argextreme_boundary,
    bench_datetime_ptp_boundary,
    bench_datetime_minmax_boundary,
    bench_timedelta_cumsum_boundary,
    bench_asarray_dtype_boundary,
    bench_char_add_boundary,
    bench_matmul_boundary,
    bench_sort_kind_boundary,
    bench_string_sort_boundary,
    bench_string_unique_full_boundary,
    bench_string_unique_boundary,
    bench_string_searchsorted_boundary,
    bench_string_isin_boundary,
    bench_string_union1d_boundary,
    bench_string_setops_boundary,
    bench_string_setxor_boundary,
    bench_string_bytes_boundary,
    bench_string_bytes_ops2_boundary,
    bench_complex_unique_boundary,
    bench_complex64_unique_boundary,
    bench_complex_searchsorted_boundary,
    bench_complex_isin_boundary,
    bench_complex64_ops_boundary,
    bench_datetime_unique_boundary,
    bench_unique_struct_int_boundary,
    bench_unique_struct_mixed_boundary,
    bench_unique_rows_datetime_boundary,
    bench_unique_rows_f16_boundary,
    bench_unique_struct_int_factorize_boundary,
    bench_unique_struct_mixed_factorize_boundary,
    bench_lexsort_float_boundary,
    bench_sort_struct_mixed_boundary,
    bench_argsort_struct_stable_boundary,
    bench_argsort_string_stable_boundary,
    bench_argsort_temporal_complex_stable_boundary,
    bench_argsort_numeric_stable_boundary,
    bench_argsort_radix_stable_boundary,
    bench_argsort_radix_float_boundary,
    bench_argsort_default_int_radix_boundary,
    bench_argsort_default_float_radix_boundary,
    bench_argsort_datetime_radix_boundary,
    bench_median_int_histogram_boundary,
    bench_int_percentile_quantile_histogram_boundary,
    bench_argsort_lastaxis_stable_boundary,
    bench_unique_arrayapi_boundary,
    bench_isin_struct_boundary,
    bench_isin_struct_float_boundary,
    bench_searchsorted_struct_boundary,
    bench_searchsorted_struct_mixed_boundary,
    bench_struct_setops_boundary,
    bench_struct_mixed_setops_boundary,
    bench_c128_setops_boundary,
    bench_datetime_setops_boundary,
    bench_datetime_searchsorted_isin_boundary,
    bench_f16_ops_boundary,
    bench_f16_setops_boundary,
    bench_unique_rows_lexsort_boundary,
    bench_unique_rows_factorize_boundary,
    bench_unique_rows_f64_boundary,
    bench_unique_rows_f32_boundary,
    bench_unique_rows_f64_factorize_boundary,
    bench_unique_rows_f32_factorize_boundary,
    bench_unique_rows_c128_boundary,
    bench_unique_rows_c64_boundary,
    bench_unique_rows_c128_factorize_boundary,
    bench_unique_cols_axis1_boundary,
    bench_sort_axis_boundary,
    bench_parallel_binary_boundary,
    bench_take_axis_boundary,
    bench_take_along_axis_c64_boundary,
    bench_take_along_axis_boundary,
    bench_take_dtype_boundary,
    bench_take_boundary,
    bench_repeat_array_boundary,
    bench_repeat_axis_boundary,
    bench_searchsorted_boundary,
    bench_digitize_boundary,
    bench_bincount_boundary,
    bench_tile_boundary,
    bench_pad_edge_boundary,
    bench_pad_wrap_boundary,
    bench_pad_reflect_boundary,
    bench_kron_boundary,
    bench_nan_to_num_boundary,
    bench_cross_boundary,
    bench_sqrt_input_extraction,
    bench_around_boundary,
    bench_where_boundary,
    bench_f64_convolve_boundary,
    bench_int_convolve_boundary,
    bench_clip_boundary,
    bench_unary_parallel_boundary,
    bench_int32_unary_boundary,
    bench_narrow_int_unary_boundary,
    bench_remainder_mod_boundary,
    bench_timedelta_addsub_boundary,
    bench_temporal_astype_boundary,
    bench_max_min_reduction_boundary,
    bench_ptp_axis0_boundary,
    bench_ptp_f32_axis_boundary,
    bench_bool_minmax_reduction_boundary,
    bench_prod_reduction_boundary,
    bench_ediff1d_boundary,
    bench_diff_1d_boundary,
    bench_select_boundary,
    bench_ldexp_boundary,
    bench_float_power_boundary,
    bench_logaddexp2_scalar_boundary,
    bench_heaviside_scalar_boundary,
    bench_frexp_boundary,
    bench_modf_boundary,
    bench_putmask_boundary,
    bench_shift_boundary,
    bench_concat_hstack_boundary,
    bench_vstack_1d_boundary,
    bench_column_interleave_boundary,
    bench_indices_construction_boundary,
    bench_char_ascii_boundary,
    bench_average_nansum_axis_boundary,
    bench_histogram_boundary,
    bench_setops_boundary,
    bench_float_isin_boundary,
    bench_f16_binary_transcendental_boundary,
    bench_unique_medium_boundary,
    bench_sort_complex_boundary,
    bench_complex_binary_boundary,
    bench_complex_exp_boundary,
    bench_f16_matmul_boundary,
    bench_flat_sort_dtype_boundary,
    bench_statistics_boundary,
    bench_cov_large_boundary,
    bench_std_var_axis_boundary,
    bench_var_multiaxis_boundary,
    bench_var_midaxis_boundary,
    bench_var_f32_axis_boundary,
    bench_nanvar_f32_axis_boundary,
    bench_nansum_f32_axis_boundary,
    bench_nanextreme_f32_axis_boundary,
    bench_nanvar_f32_last_axis_boundary,
    bench_nanvar_midaxis_boundary,
    bench_var_axis0_boundary,
    bench_sum_lastaxis_boundary,
    bench_prod_lastaxis_boundary,
    bench_cumsum_lastaxis_boundary,
    bench_complex_cumprod_lastaxis_boundary,
    bench_complex_nancumprod_lastaxis_boundary,
    bench_complex_cumulative_midaxis_boundary,
    bench_complex_cumulative_axis0_boundary,
    bench_cumsum_flat_boundary,
    bench_accumulate_extremum_boundary,
    bench_cum_midaxis_boundary,
    bench_int_cum_boundary,
    bench_vander_boundary,
    bench_polyval_boundary,
    bench_gradient_axis_boundary,
    bench_gradient_2d_coords_boundary,
    bench_gradient_coords_boundary,
    bench_gradient_nd_coords_axis_boundary,
    bench_gradient_f32_boundary,
    bench_norm_axis_boundary,
    bench_norm_f32_orderfree_boundary,
    bench_norm_nonlast_axis_boundary,
    bench_norm_frobenius_boundary,
    bench_compress_boundary,
    bench_compress_lastaxis_boundary,
    bench_delete_mask_boundary,
    bench_insert_block_boundary,
    bench_roll_2d_multi_dtype_boundary,
    bench_roll_boundary,
    bench_einsum_boundary,
    bench_linalg_boundary
);
criterion_main!(benches);
