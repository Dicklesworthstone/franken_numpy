//! elementwise / stacking / select-family criterion benches — select, ldexp,
//! float_power, logaddexp2 and heaviside (scalar), frexp, modf, putmask, shift,
//! column interleave, vstack/hstack concat, indices construction, char ascii,
//! average nansum, histogram, and setops — split out of the monolithic
//! `criterion_python_surface.rs` into their own per-domain bench binary. See bead
//! deadlock-audit-x7nnf.

#[path = "common/mod.rs"]
mod common;

use common::ensure_numpy_available;
use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyDict, PyModule, PyTuple};
use rayon::prelude::*;
use std::hint::black_box;
use std::time::{Duration, Instant};

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
        let numpy_tr = numpy_char
            .getattr("translate")
            .expect("numpy char.translate");
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
            .call_method(
                "full",
                ((1_000_000_usize,), "   azByCxD0123   "),
                Some(&kw3),
            )
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
                    black_box(
                        numpy_hist
                            .call((x,), Some(&big_kwargs))
                            .expect("numpy hist big"),
                    )
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

fn main() {
    common::gated_main(&[
        ("bench_select_boundary", bench_select_boundary),
        ("bench_ldexp_boundary", bench_ldexp_boundary),
        ("bench_float_power_boundary", bench_float_power_boundary),
        (
            "bench_logaddexp2_scalar_boundary",
            bench_logaddexp2_scalar_boundary,
        ),
        (
            "bench_heaviside_scalar_boundary",
            bench_heaviside_scalar_boundary,
        ),
        ("bench_frexp_boundary", bench_frexp_boundary),
        ("bench_modf_boundary", bench_modf_boundary),
        ("bench_putmask_boundary", bench_putmask_boundary),
        ("bench_shift_boundary", bench_shift_boundary),
        (
            "bench_column_interleave_boundary",
            bench_column_interleave_boundary,
        ),
        ("bench_vstack_1d_boundary", bench_vstack_1d_boundary),
        ("bench_concat_hstack_boundary", bench_concat_hstack_boundary),
        (
            "bench_indices_construction_boundary",
            bench_indices_construction_boundary,
        ),
        ("bench_char_ascii_boundary", bench_char_ascii_boundary),
        (
            "bench_average_nansum_axis_boundary",
            bench_average_nansum_axis_boundary,
        ),
        ("bench_histogram_boundary", bench_histogram_boundary),
        ("bench_setops_boundary", bench_setops_boundary),
        (
            "bench_divide_fe_hazard_serial",
            bench_divide_fe_hazard_serial,
        ),
        (
            "bench_divide_fe_hazard_parallel",
            bench_divide_fe_hazard_parallel,
        ),
    ]);
}

// ─────────────────────────────────────────────────────────────────────────────
// f64 divide FE-hazard branch — deadlock-audit-jw7vk
// ─────────────────────────────────────────────────────────────────────────────
//
// deadlock-audit-2nmd1 made the f64 divide fast path defer to numpy on any
// element that would raise an IEEE FP exception, because bit-identical VALUES
// are not parity when numpy also emits a RuntimeWarning a pure-Rust kernel
// cannot raise into numpy's error state. The repair writes all quotients, scans
// the result buffer for non-normal values, and only then runs the precise
// operand-aware classifier on that exceptional subset.
//
// RESULT CLASS IS FIXED IN ADVANCE: the base arm is OUR OWN FORMER CODE, so any
// number here is `maintenance-self-speedup` — it says nothing about NumPy and
// must never be quoted as a competitive claim. The expected sign is a small
// REGRESSION, and a measured regression reported plainly is the success
// condition. The deferral is a correctness requirement with a named probe
// (`cargo test -p fnp-python --test conformance_diagnostics`) and is not on the
// table whatever these arms say.
//
// WHAT ACTUALLY DIFFERS, from reading commit 47b8ad17 rather than its summary:
// the `Div` arm is a SEPARATE loop from the generic `op.apply` loop, not the old
// loop plus a branch. So the former arm here reproduces the old loop's shape —
// `*s = op.apply(x, y)` with `apply` monomorphised to a divide — and the
// repaired arm reproduces the new two-pass shape. The predicate itself is a
// REPLICA — a bench crate cannot reach a crate-root item in fnp-python — so it
// is pinned to the shipped body by DIVIDE_HAZARD_TRUTH_TABLE below, which is
// asserted before any timing runs. lib.rs carries the same table in its own
// unit test, and the comment at the shipped predicate says both must move
// together.
//
// THIS IS A KERNEL-ONLY A/B, and this repo has already retracted one kernel-only
// row that overstated a ratio by omitting route overhead. So each arm also
// measures the END-TO-END `fnp.divide(a, b)` call at the same size and prints
// the kernel's share of it. A kernel ratio without that share is a FRAME %
// masquerading as a REMOVABLE %; read them together.

/// Byte-for-byte replica of `f64_divide_raises_fp_error` in
/// `crates/fnp-python/src/lib.rs`. A bench crate cannot reach that crate-root
/// item, so the body is duplicated and pinned by `DIVIDE_HAZARD_TRUTH_TABLE`,
/// which is asserted before any timing runs. If this drifts from the shipped
/// predicate, the measurement is of something we do not ship.
#[inline]
fn bench_divide_raises_fp_error(a: f64, b: f64, q: f64) -> bool {
    if q.is_normal() {
        return false;
    }
    if a.is_nan() || b.is_nan() {
        return false;
    }
    if b == 0.0 {
        return a.is_finite();
    }
    if a.is_infinite() {
        return b.is_infinite();
    }
    if b.is_infinite() {
        return false;
    }
    q.is_infinite() || (a != 0.0 && q.abs() < f64::MIN_POSITIVE)
}

/// (a, b, expected_hazard). Every arm of the predicate, both directions. The same
/// table lives beside the shipped predicate in lib.rs; they must move together.
const DIVIDE_HAZARD_TRUTH_TABLE: &[(f64, f64, bool)] = &[
    (6.0, 3.0, false),                    // normal quotient: fast accept
    (f64::NAN, 1.0, false),               // NaN operand propagates quietly
    (1.0, f64::NAN, false),               // ditto, other side
    (1.0, 0.0, true),                     // FE_DIVBYZERO
    (-1.0, 0.0, true),                    // FE_DIVBYZERO, signed
    (0.0, 0.0, true),                     // FE_INVALID
    (f64::INFINITY, 0.0, false),          // exact +inf, raises nothing
    (f64::INFINITY, f64::INFINITY, true), // FE_INVALID
    (f64::INFINITY, 2.0, false),          // exact inf
    (1.0, f64::INFINITY, false),          // exact zero
    (f64::MAX, f64::MIN_POSITIVE, true),  // FE_OVERFLOW
    (f64::MIN_POSITIVE, f64::MAX, true),  // FE_UNDERFLOW (conservative)
    (0.0, 2.0, false),                    // exact zero, not underflow
];

fn assert_divide_hazard_replica_matches_contract() {
    for &(a, b, expected) in DIVIDE_HAZARD_TRUTH_TABLE {
        let got = bench_divide_raises_fp_error(a, b, a / b);
        assert_eq!(
            got, expected,
            "bench replica of f64_divide_raises_fp_error disagrees with the pinned \
             contract at a={a:?} b={b:?}: got {got}, expected {expected}. The measured \
             branch is not the shipped one — fix the replica before trusting any ratio."
        );
    }
}

const DIVIDE_SERIAL_N: usize = 1 << 20; // below the kernel's 1<<21 rayon threshold
const DIVIDE_PARALLEL_N: usize = 1 << 22; // above it

/// Hazard-free operands: every quotient lands in (0.5, 2), i.e. normal, so the
/// repaired arm scans every result and never enters precise classification.
fn divide_hazard_free_operands(n: usize) -> (Vec<f64>, Vec<f64>) {
    let a = (0..n)
        .map(|i| 1.0 + ((i % 1000) as f64) / 1000.0)
        .collect::<Vec<f64>>();
    let b = (0..n)
        .map(|i| 1.25 + ((i % 997) as f64) / 997.0)
        .collect::<Vec<f64>>();
    (a, b)
}

fn divide_checksum(out: &[f64]) -> u64 {
    out[0].to_bits()
        ^ out[out.len() / 2].to_bits().rotate_left(17)
        ^ out[out.len() - 1].to_bits().rotate_left(37)
}

/// The pre-2nmd1 serial loop: a plain divide, no hazard test.
#[inline(never)]
fn divide_former_serial(a: &[f64], b: &[f64], out: &mut [f64]) {
    for ((s, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *s = x / y;
    }
}

/// The shipped serial loop: a vectorizable quotient pass, a result-buffer scan,
/// then precise classification only when a non-normal quotient exists.
#[inline(never)]
fn divide_repaired_serial(a: &[f64], b: &[f64], out: &mut [f64]) -> bool {
    for ((s, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *s = x / y;
    }
    out.iter().any(|q| !q.is_normal())
        && a.iter()
            .zip(b.iter())
            .zip(out.iter())
            .any(|((&x, &y), &q)| !q.is_normal() && bench_divide_raises_fp_error(x, y, q))
}

/// Same chunking the kernel uses: `n.div_ceil(rayon::current_num_threads())`.
#[inline(never)]
fn divide_former_parallel(a: &[f64], b: &[f64], out: &mut [f64]) {
    let chunk = out.len().div_ceil(rayon::current_num_threads());
    out.par_chunks_mut(chunk)
        .zip(a.par_chunks(chunk))
        .zip(b.par_chunks(chunk))
        .for_each(|((o, l), r)| {
            for ((s, &x), &y) in o.iter_mut().zip(l.iter()).zip(r.iter()) {
                *s = x / y;
            }
        });
}

#[inline(never)]
fn divide_repaired_parallel(a: &[f64], b: &[f64], out: &mut [f64]) -> bool {
    let chunk = out.len().div_ceil(rayon::current_num_threads());
    out.par_chunks_mut(chunk)
        .zip(a.par_chunks(chunk))
        .zip(b.par_chunks(chunk))
        .for_each(|((o, l), r)| {
            for ((s, &x), &y) in o.iter_mut().zip(l.iter()).zip(r.iter()) {
                *s = x / y;
            }
        });
    out.par_iter().any(|q| !q.is_normal())
        && a.par_iter()
            .zip(b.par_iter())
            .zip(out.par_iter())
            .any(|((&x, &y), &q)| !q.is_normal() && bench_divide_raises_fp_error(x, y, q))
}

/// Times the real `fnp.divide(a, b)` end to end and prints the kernel's share of
/// it, so the kernel ratio above is read against the call it actually sits in.
fn report_divide_route_share(n: usize, label: &str, kernel_ns: f64) {
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let locals = PyDict::new(py);
        locals.set_item("np", &numpy).expect("bind numpy");
        locals.set_item("n", n).expect("bind n");
        py.run(
            std::ffi::CString::new(
                "i = np.arange(n)\na = 1.0 + (i % 1000) / 1000.0\nb = 1.25 + (i % 997) / 997.0\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&locals),
            Some(&locals),
        )
        .expect("build operands");
        let a = locals.get_item("a").expect("a operand");
        let b = locals.get_item("b").expect("b operand");
        let divide = module.getattr("divide").expect("fnp.divide");
        // TRAP 1 (dispatch): this arm times OUR route, so assert it is ours and
        // not numpy's re-export before quoting a share.
        let numpy_divide = numpy.getattr("divide").expect("numpy.divide");
        assert!(
            !divide.is(&numpy_divide),
            "fnp.divide is numpy's object — the route under test is not ours"
        );
        let args = PyTuple::new(py, [&a, &b]).expect("args");
        for _ in 0..8 {
            black_box(divide.call1(&args).expect("warm divide"));
        }
        const ROUNDS: usize = 24;
        let mut best = f64::INFINITY;
        for _ in 0..ROUNDS {
            let started = std::time::Instant::now();
            black_box(divide.call1(&args).expect("divide"));
            best = best.min(started.elapsed().as_secs_f64() * 1.0e9);
        }
        println!(
            "DIVIDE_ROUTE_SHARE label={label} n={n} end_to_end_ns={best:.1} \
             kernel_ns={kernel_ns:.1} kernel_share={:.3}",
            kernel_ns / best
        );
    });
}

fn bench_divide_fe_hazard_serial(_c: &mut Criterion) {
    assert_divide_hazard_replica_matches_contract();
    let n = DIVIDE_SERIAL_N;
    let (a, b) = divide_hazard_free_operands(n);
    let mut former_out = vec![0.0_f64; n];
    let mut repaired_out = vec![0.0_f64; n];

    let time_former = || {
        let started = Instant::now();
        divide_former_serial(&a, &b, &mut former_out);
        let elapsed = started.elapsed();
        let checksum = divide_checksum(&former_out);
        black_box(&former_out);
        common::ContractObservation { elapsed, checksum }
    };
    let time_repaired = || {
        let started = Instant::now();
        let hazard = divide_repaired_serial(&a, &b, &mut repaired_out);
        let elapsed = started.elapsed();
        assert!(!hazard, "operands must be hazard-free for this measurement");
        let checksum = divide_checksum(&repaired_out);
        black_box(&repaired_out);
        common::ContractObservation { elapsed, checksum }
    };
    let (effect, _null) = common::run_median_ci_contract(
        "divide_fe_hazard_branch_serial_1m",
        time_former,
        time_repaired,
    );
    report_divide_route_share(n, "serial", effect.arm_b_median_ns);
}

fn bench_divide_fe_hazard_parallel(_c: &mut Criterion) {
    assert_divide_hazard_replica_matches_contract();
    let n = DIVIDE_PARALLEL_N;
    let (a, b) = divide_hazard_free_operands(n);
    let mut former_out = vec![0.0_f64; n];
    let mut repaired_out = vec![0.0_f64; n];

    let time_former = || {
        let started = Instant::now();
        divide_former_parallel(&a, &b, &mut former_out);
        let elapsed = started.elapsed();
        let checksum = divide_checksum(&former_out);
        black_box(&former_out);
        common::ContractObservation { elapsed, checksum }
    };
    let time_repaired = || {
        let started = Instant::now();
        let hazard = divide_repaired_parallel(&a, &b, &mut repaired_out);
        let elapsed = started.elapsed();
        assert!(!hazard, "operands must be hazard-free for this measurement");
        let checksum = divide_checksum(&repaired_out);
        black_box(&repaired_out);
        common::ContractObservation { elapsed, checksum }
    };
    let (effect, _null) = common::run_median_ci_contract(
        "divide_fe_hazard_branch_parallel_4m",
        time_former,
        time_repaired,
    );
    report_divide_route_share(n, "parallel", effect.arm_b_median_ns);
}
