//! integer/narrow-int unary, temporal astype, timedelta add/sub, remainder, and
//! reduction (max/min, ptp, bool minmax, prod, diff, ediff1d) criterion benches
//! split out of the monolithic `criterion_python_surface.rs` into their own
//! per-domain bench binary. See bead deadlock-audit-x7nnf.

#[path = "common/mod.rs"]
mod common;

use common::ensure_numpy_available;
use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyDict, PyModule};
use std::hint::black_box;
use std::time::Duration;

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
        let _numpy = py.import("numpy").expect("numpy oracle");
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
            b.iter(|| {
                black_box(
                    dt.call_method1("astype", ("datetime64[us]",))
                        .expect("np astype"),
                )
            });
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
            .call_method1(
                "linspace",
                (-1.0_f64, 1.0_f64, 256_usize * 256_usize * 64_usize),
            )
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
            .call_method1(
                "linspace",
                (-1.0_f64, 1.0_f64, 256_usize * 128_usize * 256_usize),
            )
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

fn main() {
    common::gated_main(&[
        ("bench_int32_unary_boundary", bench_int32_unary_boundary),
        ("bench_narrow_int_unary_boundary", bench_narrow_int_unary_boundary),
        ("bench_temporal_astype_boundary", bench_temporal_astype_boundary),
        ("bench_timedelta_addsub_boundary", bench_timedelta_addsub_boundary),
        ("bench_remainder_mod_boundary", bench_remainder_mod_boundary),
        ("bench_max_min_reduction_boundary", bench_max_min_reduction_boundary),
        ("bench_ptp_f32_axis_boundary", bench_ptp_f32_axis_boundary),
        ("bench_ptp_axis0_boundary", bench_ptp_axis0_boundary),
        ("bench_bool_minmax_reduction_boundary", bench_bool_minmax_reduction_boundary),
        ("bench_prod_reduction_boundary", bench_prod_reduction_boundary),
        ("bench_diff_1d_boundary", bench_diff_1d_boundary),
        ("bench_ediff1d_boundary", bench_ediff1d_boundary),
    ]);
}
