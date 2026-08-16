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
        (
            "bench_divide_fe_hazard_fused_serial",
            bench_divide_fe_hazard_fused_serial,
        ),
        (
            "bench_divide_fe_hazard_fused_parallel",
            bench_divide_fe_hazard_fused_parallel,
        ),
        (
            "bench_divide_vs_numpy_incumbent",
            bench_divide_vs_numpy_incumbent,
        ),
        (
            "bench_divide_vs_numpy_incumbent_parallel",
            bench_divide_vs_numpy_incumbent_parallel,
        ),
        (
            "bench_percall_floor_stage_attribution",
            bench_percall_floor_stage_attribution,
        ),
        (
            "bench_binary_route_overhead_vs_numpy",
            bench_binary_route_overhead_vs_numpy,
        ),
        (
            "bench_route_floor_size_sweep_vs_numpy",
            bench_route_floor_size_sweep_vs_numpy,
        ),
        (
            "bench_percall_floor_across_ops_vs_numpy",
            bench_percall_floor_across_ops_vs_numpy,
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
// cannot raise into numpy's error state. The repair observes non-normal
// quotients while writing the result buffer, and only then runs the precise
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
// repaired arm reproduces the shape that writes the quotients and then STREAMS
// THEM AGAIN to look for a non-normal one.
//
// THREE ARMS, and which pair a row compares is what its class means:
//   `divide_former_*`   the pre-2nmd1 loop, no hazard observation at all
//   `divide_repaired_*` quotients written, then a second full pass to find a
//                       non-normal one  (what shipped 2nmd1..vqxoa)
//   `divide_fused_*`    the non-normal flag accumulated beside the divide, so
//                       the quotients are never re-read  (shipped by vqxoa)
// former-vs-repaired is the standing `deadlock-audit-jw7vk` row (what the
// correctness fix cost); repaired-vs-fused is `deadlock-audit-vqxoa` (how much
// of that the fusion gives back). Both are self-speedups. The predicate itself is a
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
fn bench_divide_quotient_is_normal(q: f64) -> bool {
    const EXPONENT_MASK: u64 = 0x7ff0_0000_0000_0000;
    let exponent = q.to_bits() & EXPONENT_MASK;
    exponent.wrapping_sub(1) < EXPONENT_MASK - 1
}

#[inline]
fn bench_divide_raises_fp_error(a: f64, b: f64, q: f64) -> bool {
    if bench_divide_fast_accepts_without_fp_error(a, b, q) {
        return false;
    }
    bench_divide_non_fast_raises_fp_error(a, b, q)
}

#[inline]
fn bench_divide_non_fast_raises_fp_error(a: f64, b: f64, q: f64) -> bool {
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

/// Replica of the shipped f64 divide fast-accept predicate.
#[inline]
fn bench_divide_fast_accepts_without_fp_error(a: f64, b: f64, q: f64) -> bool {
    bench_divide_quotient_is_normal(q) || (q == 0.0 && a == 0.0 && b.is_finite() && b != 0.0)
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
    assert_route_share_uses_matched_statistics();
    for &(a, b, expected) in DIVIDE_HAZARD_TRUTH_TABLE {
        let got = bench_divide_raises_fp_error(a, b, a / b);
        assert_eq!(
            got, expected,
            "bench replica of f64_divide_raises_fp_error disagrees with the pinned \
             contract at a={a:?} b={b:?}: got {got}, expected {expected}. The measured \
             branch is not the shipped one — fix the replica before trusting any ratio."
        );
    }
    for &(a, b) in &[(6.0, 3.0), (0.0, 2.0), (-0.0, -2.0)] {
        assert!(
            bench_divide_fast_accepts_without_fp_error(a, b, a / b),
            "bench fast accept must keep quiet {a} / {b} outside precise classification"
        );
    }

    // Test the complete serial replica as well as its two predicates. The
    // quiet lanes are deliberately non-normal, so a replica that checks only
    // normal quotients would incorrectly report a hazard; the final lane is a
    // genuine divide-by-zero negative case that must still report one.
    // Lane 2 carries the -0.0 quotient. The divisor must be POSITIVE to get one:
    // IEEE takes the quotient's sign from the XOR of the operand signs, so
    // -0.0 / -2.0 is +0.0, not -0.0. Landed as -2.0 by 34dc92fc and never run,
    // which made every group in this file panic before its first timed round.
    let quiet_lhs = [8.0, 0.0, -0.0, f64::NAN, 4.0];
    let quiet_rhs = [2.0, 2.0, 2.0, 2.0, f64::NAN];
    let mut quiet_out = [0.0; 5];
    assert!(
        !divide_repaired_serial(&quiet_lhs, &quiet_rhs, &mut quiet_out),
        "quiet signed-zero and NaN lanes must remain in the measured native arm"
    );
    assert_eq!(quiet_out[0].to_bits(), 4.0_f64.to_bits());
    assert_eq!(quiet_out[1].to_bits(), 0.0_f64.to_bits());
    assert_eq!(quiet_out[2].to_bits(), (-0.0_f64).to_bits());
    assert!(quiet_out[3].is_nan() && quiet_out[4].is_nan());

    let mut hazard_out = [0.0; 2];
    assert!(
        divide_repaired_serial(&[8.0, 1.0], &[2.0, 0.0], &mut hazard_out),
        "a real divide-by-zero must keep the benchmark's fallback branch armed"
    );
    assert_eq!(hazard_out[0], 4.0);
    assert!(hazard_out[1].is_infinite());

    // The fused arm must agree with the scanning arm on BOTH the verdict and
    // the bits, for quiet and hazardous input alike. The negative case that
    // matters is a hazard hiding behind normal neighbours: an accumulator that
    // is reset per iteration, or one that only records the LAST quotient,
    // returns false here and would silently stop deferring to NumPy.
    let mixed_lhs = [8.0, 6.0, 1.0, 3.0];
    let mixed_rhs = [2.0, 3.0, 0.0, 1.5];
    let mut scanned_mixed = [0.0; 4];
    let mut fused_mixed = [0.0; 4];
    assert!(
        divide_repaired_serial(&mixed_lhs, &mixed_rhs, &mut scanned_mixed),
        "the scanning replica must report the divide-by-zero in lane 2"
    );
    assert!(
        divide_fused_serial(&mixed_lhs, &mixed_rhs, &mut fused_mixed),
        "the fused replica must report a hazard surrounded by normal quotients"
    );
    assert_eq!(
        scanned_mixed.map(f64::to_bits),
        fused_mixed.map(f64::to_bits),
        "fusing the normality flag must not change a single quotient bit"
    );

    let mut scanned_quiet = [0.0; 5];
    let mut fused_quiet = [0.0; 5];
    assert!(!divide_repaired_serial(
        &quiet_lhs,
        &quiet_rhs,
        &mut scanned_quiet
    ));
    assert!(
        !divide_fused_serial(&quiet_lhs, &quiet_rhs, &mut fused_quiet),
        "the fused replica must keep the quiet signed-zero and NaN lanes native"
    );
    assert_eq!(
        scanned_quiet.map(f64::to_bits),
        fused_quiet.map(f64::to_bits)
    );

    // The parallel replica carries the same obligation, and additionally must
    // OR its per-chunk flags: a reduce that drops any chunk's verdict passes
    // every single-chunk test and fails here only once the hazard lands off
    // the first chunk.
    let parallel_len = 4_096;
    let parallel_lhs = (0..parallel_len)
        .map(|i| if i == parallel_len - 1 { 1.0 } else { 8.0 })
        .collect::<Vec<f64>>();
    let parallel_rhs = (0..parallel_len)
        .map(|i| if i == parallel_len - 1 { 0.0 } else { 2.0 })
        .collect::<Vec<f64>>();
    let mut scanned_parallel = vec![0.0; parallel_len];
    let mut fused_parallel = vec![0.0; parallel_len];
    assert!(divide_repaired_parallel(
        &parallel_lhs,
        &parallel_rhs,
        &mut scanned_parallel
    ));
    assert!(
        divide_fused_parallel(&parallel_lhs, &parallel_rhs, &mut fused_parallel),
        "the fused parallel replica must report a hazard in the final chunk"
    );
    assert_eq!(
        scanned_parallel
            .iter()
            .map(|q| q.to_bits())
            .collect::<Vec<u64>>(),
        fused_parallel
            .iter()
            .map(|q| q.to_bits())
            .collect::<Vec<u64>>(),
    );
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

/// The shipped serial loop keeps quotient production classifier-free, then
/// performs the rare exact scan only after a non-normal result is observed.
#[inline(never)]
fn divide_repaired_serial(a: &[f64], b: &[f64], out: &mut [f64]) -> bool {
    for ((slot, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *slot = x / y;
    }
    out.iter().any(|q| !bench_divide_quotient_is_normal(*q))
        && a.iter()
            .zip(b.iter())
            .zip(out.iter())
            .any(|((&x, &y), &q)| bench_divide_raises_fp_error(x, y, q))
}

/// The fused serial loop: the normality flag is accumulated beside the divide,
/// so the quotients are never streamed a second time to discover it. Replica of
/// the shipped `zerocopy_f64_binary_flat` serial `Div` arm.
#[inline(never)]
fn divide_fused_serial(a: &[f64], b: &[f64], out: &mut [f64]) -> bool {
    let mut saw_non_normal = false;
    for ((slot, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        let quotient = x / y;
        *slot = quotient;
        saw_non_normal |= !bench_divide_quotient_is_normal(quotient);
    }
    saw_non_normal
        && a.iter()
            .zip(b.iter())
            .zip(out.iter())
            .any(|((&x, &y), &q)| bench_divide_raises_fp_error(x, y, q))
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
    out.par_iter().any(|q| !bench_divide_quotient_is_normal(*q))
        && a.par_iter()
            .zip(b.par_iter())
            .zip(out.par_iter())
            .any(|((&x, &y), &q)| bench_divide_raises_fp_error(x, y, q))
}

/// Replica of the shipped parallel `Div` arm after fusion: each chunk returns
/// its own flag and the flags are OR-reduced, so no chunk's verdict is lost and
/// the output is never re-read.
#[inline(never)]
fn divide_fused_parallel(a: &[f64], b: &[f64], out: &mut [f64]) -> bool {
    let chunk = out.len().div_ceil(rayon::current_num_threads());
    let saw_non_normal = out
        .par_chunks_mut(chunk)
        .zip(a.par_chunks(chunk))
        .zip(b.par_chunks(chunk))
        .map(|((o, l), r)| {
            let mut chunk_saw_non_normal = false;
            for ((slot, &x), &y) in o.iter_mut().zip(l.iter()).zip(r.iter()) {
                let quotient = x / y;
                *slot = quotient;
                chunk_saw_non_normal |= !bench_divide_quotient_is_normal(quotient);
            }
            chunk_saw_non_normal
        })
        .reduce(|| false, |left, right| left | right);
    saw_non_normal
        && a.par_iter()
            .zip(b.par_iter())
            .zip(out.par_iter())
            .any(|((&x, &y), &q)| bench_divide_raises_fp_error(x, y, q))
}

/// Median of the end-to-end samples, sorted in place. The share's numerator is
/// the contract's arm MEDIAN, so its denominator has to be a median as well —
/// see `assert_route_share_uses_matched_statistics` for the direction of the
/// error a best-of denominator produces.
fn route_share_median_ns(samples: &mut [f64]) -> f64 {
    assert!(
        !samples.is_empty(),
        "a route share needs at least one end-to-end sample"
    );
    samples.sort_by(f64::total_cmp);
    let middle = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[middle - 1] + samples[middle]) / 2.0
    } else {
        samples[middle]
    }
}

/// Pins the statistic the route share divides by. Timing noise is right-skewed,
/// so a best-of denominator is strictly smaller than the median and the share
/// comes out too LARGE — the flattering direction, since the share is what stops
/// a kernel-only ratio being read as an end-to-end one. This ran as
/// `kernel_share=1.473`, a kernel apparently costing 147% of the call containing
/// it (`deadlock-audit-3i0uo`).
fn assert_route_share_uses_matched_statistics() {
    // Right-skewed like real timing samples: one slow outlier, three tight.
    let mut samples = [120.0, 100.0, 400.0, 110.0];
    let best = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let median = route_share_median_ns(&mut samples);
    assert_eq!(
        median, 115.0,
        "an even-length median is the mean of the two middle samples, not the smallest"
    );
    assert_eq!(best, 100.0, "the best-of is the minimum sample");
    // THE NEGATIVE CASE: an implementation that kept the best-of denominator
    // returns 100.0 here, passes every "is it a number" check, and reports a
    // share 15% larger than the truth on this sample set alone.
    let kernel_ns = 200.0;
    assert!(
        kernel_ns / best > kernel_ns / median,
        "a best-of denominator must be shown to inflate the share, not merely differ"
    );

    let mut odd = [300.0, 100.0, 200.0];
    assert_eq!(
        route_share_median_ns(&mut odd),
        200.0,
        "an odd-length median is the middle sample after sorting"
    );
    let mut single = [42.0];
    assert_eq!(route_share_median_ns(&mut single), 42.0);
}

/// The same three-lane checksum `divide_checksum` computes, read off the route's
/// NumPy result. Called after the timer stops, so the extraction never lands
/// inside the measurement. Both arms of the share contract must agree on it or
/// the round fails, which is what makes the share a ratio of two timings of ONE
/// computation rather than of two different ones.
fn numpy_divide_checksum(result: &pyo3::Bound<'_, pyo3::PyAny>, n: usize) -> u64 {
    let lane = |index: usize| -> u64 {
        result
            .get_item(index)
            .expect("route result index")
            .extract::<f64>()
            .expect("route result lane is f64")
            .to_bits()
    };
    lane(0) ^ lane(n / 2).rotate_left(17) ^ lane(n - 1).rotate_left(37)
}

/// Times the kernel replica and the real `fnp.divide(a, b)` call INTERLEAVED
/// inside one balanced-square contract, so the kernel's share of the route is a
/// ratio of two quantities measured under the same conditions.
///
/// The previous form ran the contract, then timed the route in a separate loop
/// afterwards, and divided one by the other. Those phases see different host
/// load, and the resulting "share" swung 0.804-1.911 for the same arm on the
/// same worker - it inherited the phase-to-phase load difference as if it were
/// signal (`deadlock-audit-c9rn8`). Interleaving is the same law the A/B arms
/// already obey; the share was the one number in this file exempt from it.
///
/// Ratio is kernel/route, so it reads directly as the share, and the contract's
/// A/A null says whether the host was quiet enough for it to mean anything.
/// Worker this process is executing on, read the same way `common`'s
/// HOST_BASELINE line reads it. `rch` cannot be pinned to a worker, so a row that
/// does not name the machine that produced it cannot be compared to any other
/// row: the fleet measured a 13.6x swing for one cell across two workers with
/// both A/A nulls passing, so a passing null does not license a cross-worker
/// comparison. Read from `/etc/hostname` rather than `$HOSTNAME`, which is not
/// exported over a non-interactive ssh.
fn measurement_worker() -> String {
    std::fs::read_to_string("/etc/hostname")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .chars()
                .map(|character| {
                    if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_') {
                        character
                    } else {
                        '_'
                    }
                })
                .collect()
        })
        .unwrap_or_else(|| "unavailable".to_owned())
}

fn report_divide_route_share<K>(n: usize, label: &str, mut kernel: K)
where
    K: FnMut() -> common::ContractObservation,
{
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
        let route = || {
            let started = Instant::now();
            let result = divide.call1(&args).expect("divide");
            let elapsed = started.elapsed();
            // Checksum AFTER the timer: it is a parity check between the arms,
            // not part of the route's cost.
            let checksum = numpy_divide_checksum(&result, n);
            common::ContractObservation { elapsed, checksum }
        };

        // PREFLIGHT the cross-arm parity the contract will assert. The kernel's
        // operands are built in Rust and the route's in NumPy; if those ever
        // stop agreeing bit for bit, the contract's checksum assertion would
        // panic and take the A/B rows down with it — the same way an unexecuted
        // assertion took every group in this file down for a day (34dc92fc).
        // The share is optional telemetry and must never cost a measured row,
        // so a mismatch is reported and skipped rather than raised.
        let kernel_probe = kernel();
        let route_probe = route();
        if kernel_probe.checksum != route_probe.checksum {
            println!(
                "DIVIDE_ROUTE_SHARE_SKIPPED label={label} n={n} \
                 worker={} harness=common::run_median_ci_contract \
                 verdict=kernel_and_route_operands_disagree \
                 kernel_checksum={:016x} route_checksum={:016x} \
                 share_not_reported=true",
                measurement_worker(),
                kernel_probe.checksum,
                route_probe.checksum,
            );
            return;
        }

        // Arm A is the kernel, arm B is the route, so the effect ratio IS the
        // share. The contract interleaves them ABBAABBA and runs a kernel/kernel
        // A/A null first; it also asserts both arms' checksums match every
        // round, so a share can only be reported for two timings of the same
        // computation.
        let (effect, null) =
            common::run_median_ci_contract(&format!("divide_route_share_{label}"), kernel, route);
        println!(
            "DIVIDE_ROUTE_SHARE label={label} n={n} interleaved=true \
             worker={} harness=common::run_median_ci_contract rounds={} \
             kernel_median_ns={:.1} route_median_ns={:.1} kernel_share={:.3} \
             kernel_share_ci95=[{:.3},{:.3}] null_ratio_median={:.6} \
             null_ci95=[{:.6},{:.6}] share_exceeds_one={} checksum={:016x}",
            measurement_worker(),
            common::CONTRACT_ROUNDS,
            effect.arm_a_median_ns,
            effect.arm_b_median_ns,
            effect.ratio_median,
            effect.ratio_ci_low,
            effect.ratio_ci_high,
            null.ratio_median,
            null.ratio_ci_low,
            null.ratio_ci_high,
            effect.ratio_median > 1.0,
            effect.checksum,
        );
        // A kernel timed strictly inside the end-to-end call cannot outlast it.
        // Now that both arms are interleaved this is a real signal rather than a
        // phase artifact: it means the replica is not modelling the shipped
        // route. Say so on the row instead of letting the number be quoted.
        if effect.ratio_ci_low > 1.0 {
            println!(
                "DIVIDE_ROUTE_SHARE_WARNING label={label} worker={} \
                 harness=common::run_median_ci_contract \
                 verdict=replica_slower_than_the_route_it_models \
                 do_not_quote_this_share=true",
                measurement_worker(),
            );
        }
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
    let (_effect, _null) = common::run_median_ci_contract(
        "divide_fe_hazard_branch_serial_1m",
        time_former,
        time_repaired,
    );
    let mut share_out = vec![0.0_f64; n];
    report_divide_route_share(n, "serial", || {
        let started = Instant::now();
        let hazard = divide_repaired_serial(&a, &b, &mut share_out);
        let elapsed = started.elapsed();
        assert!(!hazard, "operands must be hazard-free for this measurement");
        let checksum = divide_checksum(&share_out);
        common::ContractObservation { elapsed, checksum }
    });
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
    let (_effect, _null) = common::run_median_ci_contract(
        "divide_fe_hazard_branch_parallel_4m",
        time_former,
        time_repaired,
    );
    let mut share_out = vec![0.0_f64; n];
    report_divide_route_share(n, "parallel", || {
        let started = Instant::now();
        let hazard = divide_repaired_parallel(&a, &b, &mut share_out);
        let elapsed = started.elapsed();
        assert!(!hazard, "operands must be hazard-free for this measurement");
        let checksum = divide_checksum(&share_out);
        common::ContractObservation { elapsed, checksum }
    });
}

/// Isolates the fusion lever: base arm is the shipped scanning form, candidate
/// is the fused one. Ratio is base/candidate, so ABOVE 1.0 means fusion is
/// faster. Both arms are the same binary on the same operands, so this is a
/// self-speedup that says how much of the hazard scan's cost the fusion
/// recovers — it is NOT a NumPy comparison and must never be quoted as one.
fn bench_divide_fe_hazard_fused_serial(_c: &mut Criterion) {
    assert_divide_hazard_replica_matches_contract();
    let n = DIVIDE_SERIAL_N;
    let (a, b) = divide_hazard_free_operands(n);
    let mut scanned_out = vec![0.0_f64; n];
    let mut fused_out = vec![0.0_f64; n];

    let time_scanned = || {
        let started = Instant::now();
        let hazard = divide_repaired_serial(&a, &b, &mut scanned_out);
        let elapsed = started.elapsed();
        assert!(!hazard, "operands must be hazard-free for this measurement");
        let checksum = divide_checksum(&scanned_out);
        black_box(&scanned_out);
        common::ContractObservation { elapsed, checksum }
    };
    let time_fused = || {
        let started = Instant::now();
        let hazard = divide_fused_serial(&a, &b, &mut fused_out);
        let elapsed = started.elapsed();
        assert!(!hazard, "operands must be hazard-free for this measurement");
        let checksum = divide_checksum(&fused_out);
        black_box(&fused_out);
        common::ContractObservation { elapsed, checksum }
    };
    // The contract asserts both arms' checksums match every round, so a fusion
    // that changed a quotient bit fails the measurement rather than winning it.
    let (_effect, _null) = common::run_median_ci_contract(
        "divide_fe_hazard_fused_serial_1m",
        time_scanned,
        time_fused,
    );
    let mut share_out = vec![0.0_f64; n];
    report_divide_route_share(n, "fused_serial", || {
        let started = Instant::now();
        let hazard = divide_fused_serial(&a, &b, &mut share_out);
        let elapsed = started.elapsed();
        assert!(!hazard, "operands must be hazard-free for this measurement");
        let checksum = divide_checksum(&share_out);
        common::ContractObservation { elapsed, checksum }
    });
}

fn bench_divide_fe_hazard_fused_parallel(_c: &mut Criterion) {
    assert_divide_hazard_replica_matches_contract();
    let n = DIVIDE_PARALLEL_N;
    let (a, b) = divide_hazard_free_operands(n);
    let mut scanned_out = vec![0.0_f64; n];
    let mut fused_out = vec![0.0_f64; n];

    let time_scanned = || {
        let started = Instant::now();
        let hazard = divide_repaired_parallel(&a, &b, &mut scanned_out);
        let elapsed = started.elapsed();
        assert!(!hazard, "operands must be hazard-free for this measurement");
        let checksum = divide_checksum(&scanned_out);
        black_box(&scanned_out);
        common::ContractObservation { elapsed, checksum }
    };
    let time_fused = || {
        let started = Instant::now();
        let hazard = divide_fused_parallel(&a, &b, &mut fused_out);
        let elapsed = started.elapsed();
        assert!(!hazard, "operands must be hazard-free for this measurement");
        let checksum = divide_checksum(&fused_out);
        black_box(&fused_out);
        common::ContractObservation { elapsed, checksum }
    };
    let (_effect, _null) = common::run_median_ci_contract(
        "divide_fe_hazard_fused_parallel_4m",
        time_scanned,
        time_fused,
    );
    let mut share_out = vec![0.0_f64; n];
    report_divide_route_share(n, "fused_parallel", || {
        let started = Instant::now();
        let hazard = divide_fused_parallel(&a, &b, &mut share_out);
        let elapsed = started.elapsed();
        assert!(!hazard, "operands must be hazard-free for this measurement");
        let checksum = divide_checksum(&share_out);
        common::ContractObservation { elapsed, checksum }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// fnp.divide vs numpy.divide — deadlock-audit-su0i6
// ─────────────────────────────────────────────────────────────────────────────
//
// Every other divide row in this file is maintenance-self-speedup: our own code
// before vs after. This is the only arm that asks whether the route we have been
// optimising is ahead of or behind the incumbent, so it is the only one whose
// number may ever be quoted against NumPy — and only if it clears the
// incumbent-win contract, which is why the dispatch and parity traps below are
// checked at runtime rather than assumed.
fn bench_divide_vs_numpy_incumbent(_c: &mut Criterion) {
    let n = DIVIDE_SERIAL_N;
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
        let args = PyTuple::new(py, [&a, &b]).expect("args");

        let ours = module.getattr("divide").expect("fnp.divide");
        let theirs = numpy.getattr("divide").expect("numpy.divide");
        // TRAP 1 (dispatch). franken_networkx published a 2.6x whose baseline was
        // already their own code. Assert identity at runtime, inside the measured
        // binary, before any timing.
        assert!(
            !ours.is(&theirs),
            "fnp.divide IS numpy's object — there is no candidate arm here"
        );
        assert!(
            numpy
                .getattr("__name__")
                .expect("numpy.__name__")
                .extract::<String>()
                .expect("numpy name is a string")
                == "numpy",
            "the incumbent module must be genuine numpy"
        );
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy.__version__")
            .extract::<String>()
            .expect("numpy version is a string");

        // TRAP 2 (parity). Both arms must compute the same thing, or the ratio is
        // between two different jobs. Checked once up front on the real results,
        // and again every round by the contract's cross-arm checksum assertion.
        let ours_probe = ours.call1(&args).expect("fnp.divide probe");
        let theirs_probe = theirs.call1(&args).expect("numpy.divide probe");
        let ours_checksum = numpy_divide_checksum(&ours_probe, n);
        let theirs_checksum = numpy_divide_checksum(&theirs_probe, n);
        assert_eq!(
            ours_checksum, theirs_checksum,
            "fnp.divide and numpy.divide disagree on these operands — fix parity before timing"
        );

        let incumbent = || {
            let started = Instant::now();
            let result = theirs.call1(&args).expect("numpy.divide");
            let elapsed = started.elapsed();
            let checksum = numpy_divide_checksum(&result, n);
            common::ContractObservation { elapsed, checksum }
        };
        let candidate = || {
            let started = Instant::now();
            let result = ours.call1(&args).expect("fnp.divide");
            let elapsed = started.elapsed();
            let checksum = numpy_divide_checksum(&result, n);
            common::ContractObservation { elapsed, checksum }
        };

        // Ratio is incumbent/candidate, so ABOVE 1.0 means we are faster. The dual
        // null runs a numpy/numpy A/A and an fnp/fnp A/A, so a verdict has to clear
        // BOTH envelopes rather than one.
        let (effect, incumbent_null, candidate_null) = common::run_dual_null_median_ci_contract(
            "divide_f64_1m_vs_numpy",
            incumbent,
            candidate,
        );
        println!(
            "DIVIDE_VS_NUMPY n={n} numpy_version={numpy_version} worker={} \
             harness=common::run_dual_null_median_ci_contract rounds={} \
             incumbent_median_ns={:.1} candidate_median_ns={:.1} ratio_median={:.6} \
             ratio_ci95=[{:.6},{:.6}] incumbent_null_median={:.6} candidate_null_median={:.6} \
             faster_than_numpy={} checksum={:016x}",
            measurement_worker(),
            common::CONTRACT_ROUNDS,
            effect.arm_a_median_ns,
            effect.arm_b_median_ns,
            effect.ratio_median,
            effect.ratio_ci_low,
            effect.ratio_ci_high,
            incumbent_null.ratio_median,
            candidate_null.ratio_median,
            effect.ratio_ci_low > 1.0,
            effect.checksum,
        );
    });
}

/// Times one fnp binary ufunc against its NumPy counterpart under the dual-null
/// contract and returns (ratio_median, ci_low, ci_high, incumbent_ns,
/// candidate_ns). Ratio is incumbent/candidate, so BELOW 1.0 means we are
/// slower.
fn measure_binary_ufunc_vs_numpy(
    py: Python<'_>,
    module: &pyo3::Bound<'_, PyModule>,
    numpy: &pyo3::Bound<'_, PyModule>,
    name: &str,
    args: &pyo3::Bound<'_, PyTuple>,
    n: usize,
) -> (f64, f64, f64, f64, f64) {
    let ours = module.getattr(name).expect("fnp ufunc");
    let theirs = numpy.getattr(name).expect("numpy ufunc");
    assert!(
        !ours.is(&theirs),
        "fnp.{name} IS numpy's object — there is no candidate arm"
    );
    let ours_probe = ours.call1(args).expect("fnp probe");
    let theirs_probe = theirs.call1(args).expect("numpy probe");
    assert_eq!(
        numpy_divide_checksum(&ours_probe, n),
        numpy_divide_checksum(&theirs_probe, n),
        "fnp.{name} and numpy.{name} disagree on these operands"
    );
    let _ = py;

    let incumbent = || {
        let started = Instant::now();
        let result = theirs.call1(args).expect("numpy ufunc call");
        let elapsed = started.elapsed();
        let checksum = numpy_divide_checksum(&result, n);
        common::ContractObservation { elapsed, checksum }
    };
    let candidate = || {
        let started = Instant::now();
        let result = ours.call1(args).expect("fnp ufunc call");
        let elapsed = started.elapsed();
        let checksum = numpy_divide_checksum(&result, n);
        common::ContractObservation { elapsed, checksum }
    };
    // The row name carries the ACTUAL element count. It used to hardcode "1m",
    // which was true only while every caller passed DIVIDE_SERIAL_N; the parallel
    // group measures 1<<22 and the row would otherwise have published
    // `divide_f64_1m_vs_numpy_route` for a 4,194,304-element array. A row that
    // misstates its own size cannot be compared to anything, and this family's
    // ratio is REGIME-DEPENDENT on exactly that size.
    let (effect, _incumbent_null, _candidate_null) = common::run_dual_null_median_ci_contract(
        &format!("{name}_f64_n{n}_vs_numpy_route"),
        incumbent,
        candidate,
    );
    (
        effect.ratio_median,
        effect.ratio_ci_low,
        effect.ratio_ci_high,
        effect.arm_a_median_ns,
        effect.arm_b_median_ns,
    )
}

// Separates the two candidate causes of the divide loss measured by
// deadlock-audit-su0i6. `multiply` takes the SAME zerocopy_f64_binary_flat
// route, the same dtype/contiguity sniffing, the same numpy.empty output
// allocation and the same PyO3 dispatch as `divide` — and computes NO hazard
// scan. Measuring both in ONE invocation on ONE worker is what makes the
// comparison mean anything: multiply near unity while divide loses puts the cost
// in the scan, and both losing by a similar margin puts it in the route, which
// no divide-kernel lever can reach (deadlock-audit-1pt96).
/// The PARALLEL-REGIME twin of `bench_divide_vs_numpy_incumbent`.
///
/// `deadlock-audit-su0i6` banked a 1.16-1.22x loss against `numpy.divide`, and that
/// row is real but REGIME-SCOPED: it measures `DIVIDE_SERIAL_N` = 1<<20, which is
/// deliberately BELOW the kernel's `1 << 21` rayon gate for `BinaryOp::Div`. So it
/// compares our single-threaded arm against NumPy's single-threaded loop and says
/// nothing about the regime where we actually spend the cores. Quoting it as
/// "fnp.divide is slower than numpy.divide" without the size would overstate it.
///
/// This group measures the other side of the gate at `DIVIDE_PARALLEL_N` = 1<<22
/// through the SAME dual-null contract, same operands, same dispatch and parity
/// traps. NumPy's elementwise loops are single-threaded, so if the parallel arm does
/// not win here the deficit is structural rather than a threading gap - which is a
/// more useful thing to know than another serial number.
fn bench_divide_vs_numpy_incumbent_parallel(_c: &mut Criterion) {
    let n = DIVIDE_PARALLEL_N;
    assert!(
        n >= 1 << 21,
        "this group exists to exercise the rayon arm; n must clear the Div parallel gate"
    );
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
        let args = PyTuple::new(py, [&a, &b]).expect("args");
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy.__version__")
            .extract::<String>()
            .expect("numpy version is a string");

        // Reuses the shared helper, so the dispatch trap (fnp.divide is not NumPy's
        // object) and the cross-arm parity probe are the identical checks the serial
        // group runs - not a second, weaker copy of them.
        let (ratio, lo, hi, incumbent_ns, candidate_ns) =
            measure_binary_ufunc_vs_numpy(py, &module, &numpy, "divide", &args, n);

        println!(
            "DIVIDE_VS_NUMPY_PARALLEL n={n} regime=rayon_arm parallel_min=2097152 \
             numpy_version={numpy_version} worker={} \
             harness=common::run_dual_null_median_ci_contract rounds={} \
             rayon_threads={} incumbent_median_ns={incumbent_ns:.1} \
             candidate_median_ns={candidate_ns:.1} ratio_median={ratio:.6} \
             ratio_ci95=[{lo:.6},{hi:.6}] faster_than_numpy={}",
            measurement_worker(),
            common::CONTRACT_ROUNDS,
            rayon::current_num_threads(),
            lo > 1.0,
        );
    });
}

fn bench_binary_route_overhead_vs_numpy(_c: &mut Criterion) {
    let n = DIVIDE_SERIAL_N;
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
        let args = PyTuple::new(py, [&a, &b]).expect("args");
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy.__version__")
            .extract::<String>()
            .expect("numpy version is a string");

        let (mul_ratio, mul_lo, mul_hi, mul_inc, mul_cand) =
            measure_binary_ufunc_vs_numpy(py, &module, &numpy, "multiply", &args, n);
        let (div_ratio, div_lo, div_hi, div_inc, div_cand) =
            measure_binary_ufunc_vs_numpy(py, &module, &numpy, "divide", &args, n);

        // The verdict is stated from the intervals, not the point estimates: if
        // multiply's CI reaches unity while divide's does not, the scan is
        // implicated; if neither reaches unity, the route is.
        let multiply_is_at_parity = mul_hi >= 1.0;
        let divide_is_at_parity = div_hi >= 1.0;
        // "Both lose" is NOT the same finding as "both lose equally", and the
        // first draft of this label conflated them. Separation of the two
        // intervals is what says the divide carries a cost the shared route does
        // not explain; without it, one number could be doing all the work.
        let divide_loses_more = div_hi < mul_lo;
        let verdict = match (
            multiply_is_at_parity,
            divide_is_at_parity,
            divide_loses_more,
        ) {
            (true, false, _) => "scan_implicated_route_is_at_parity",
            (false, false, true) => "route_floor_PLUS_larger_divide_specific_cost",
            (false, false, false) => "route_overhead_only_both_arms_lose_alike",
            (true, true, _) => "no_loss_reproduced_in_this_invocation",
            (false, true, _) => "inconsistent_multiply_loses_while_divide_does_not",
        };
        // Attribute the excess in nanoseconds as well as in ratio: a ratio alone
        // cannot say how much of the divide's loss the route already explains,
        // because numpy's own divide is intrinsically dearer than its multiply.
        let multiply_excess_ns = mul_cand - mul_inc;
        let divide_excess_ns = div_cand - div_inc;
        let divide_specific_excess_ns = divide_excess_ns - multiply_excess_ns;
        println!(
            "BINARY_ROUTE_OVERHEAD n={n} numpy_version={numpy_version} \
             harness=common::run_dual_null_median_ci_contract \
             multiply_ratio={mul_ratio:.6} multiply_ci95=[{mul_lo:.6},{mul_hi:.6}] \
             multiply_numpy_ns={mul_inc:.1} multiply_fnp_ns={mul_cand:.1} \
             divide_ratio={div_ratio:.6} divide_ci95=[{div_lo:.6},{div_hi:.6}] \
             divide_numpy_ns={div_inc:.1} divide_fnp_ns={div_cand:.1} \
             multiply_excess_ns={multiply_excess_ns:.1} divide_excess_ns={divide_excess_ns:.1} \
             divide_specific_excess_ns={divide_specific_excess_ns:.1} \
             divide_specific_share_of_divide_excess={:.3} \
             intervals_disjoint={divide_loses_more} same_invocation=true verdict={verdict}",
            divide_specific_excess_ns / divide_excess_ns,
        );
    });
}

// Separates a FIXED per-call route cost from a PROPORTIONAL kernel cost, which
// need opposite levers (deadlock-audit-m7tti). `multiply` is the probe because it
// has no hazard scan, so what remains is the route: PyO3 dispatch, dtype and
// contiguity sniffing, and the `numpy.empty` output allocation with its
// first-touch faults.
//
// The discriminator is the EXCESS IN NANOSECONDS across sizes spanning 256x, not
// the ratio. A fixed cost holds excess_ns roughly constant while the ratio walks
// toward 1.0 as n grows; a proportional cost grows excess_ns with n while the
// ratio stays flat. Reading the ratio alone cannot tell those apart, which is why
// the single-size 1pt96 row could not answer it.
/// Attribute the ~6.6-7 us per-call floor to the STAGES that produce it
/// (`deadlock-audit-cydda`, following `deadlock-audit-isnd2`).
///
/// `isnd2` measured the floor and `cydda` eliminated the first suspect: the output
/// allocation and its dtype string are worth ~110 ns, i.e. 1.7% of the floor. The
/// entry path is now traced rather than assumed - `fnp.multiply` resolves to a
/// `PyUFunc` OBJECT (`m.add("multiply", PyUFunc { .. })`), not to the same-named
/// `#[pyfunction]`, which is unregistered dead code (`deadlock-audit-uxkqi`). So the
/// stages below are the ones `PyUFunc::__call__` actually walks before any element
/// is divided or multiplied.
///
/// METHOD AND ITS LIMIT, stated because it bounds what this proves: each stage is a
/// REPLICA timed standalone in this process, not an instrumentation probe inside the
/// live route. A standalone `getattr` can be cheaper than the same call inside the
/// route (branch and cache state differ), so treat the per-stage numbers as a LOWER
/// BOUND on each stage and the sum as a lower bound on the accounted fraction. What
/// it can prove is that a stage is EXPENSIVE - a lower bound of several microseconds
/// is still several microseconds - and that is what aims the next lever.
fn bench_percall_floor_stage_attribution(_c: &mut Criterion) {
    const N: usize = 256;
    const TRIALS: usize = 2001;

    // min-of-TRIALS per stage: these are sub-microsecond operations where the
    // minimum is the least contaminated estimator on a shared host, and the whole
    // point is to compare stages against each other inside ONE process.
    fn min_ns(trials: usize, mut op: impl FnMut()) -> f64 {
        let mut best = u128::MAX;
        for _ in 0..trials {
            let started = Instant::now();
            op();
            best = best.min(started.elapsed().as_nanos());
        }
        best as f64
    }

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy.__version__")
            .extract::<String>()
            .expect("numpy version is a string");

        let locals = PyDict::new(py);
        locals.set_item("np", &numpy).expect("bind numpy");
        locals.set_item("n", N).expect("bind n");
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
        let args = PyTuple::new(py, [&a, &b]).expect("args");

        let ours = module.getattr("multiply").expect("fnp.multiply");
        let theirs = numpy.getattr("multiply").expect("numpy.multiply");
        assert!(
            !ours.is(&theirs),
            "fnp.multiply IS numpy's object — there is no candidate arm"
        );

        // Stage 1: the route re-imports numpy on EVERY call.
        let import_ns = min_ns(TRIALS, || {
            black_box(py.import("numpy").expect("numpy import"));
        });
        // Stage 2: the f64 dtype guard on both operands.
        let dtype_ns = min_ns(TRIALS, || {
            black_box(a.getattr("dtype").expect("a dtype"));
            black_box(b.getattr("dtype").expect("b dtype"));
        });
        // Stage 3: the preamble's ndarray lookup.
        let ndarray_ns = min_ns(TRIALS, || {
            black_box(numpy.getattr("ndarray").expect("ndarray"));
        });
        // Stage 4: the output allocation, already known cheap - kept so the
        // accounted sum is complete rather than selectively reported.
        let empty_ns = min_ns(TRIALS, || {
            black_box(numpy.call_method1("empty", (N,)).expect("numpy.empty"));
        });
        // Stage 5 and the reference: whole calls, same operands.
        let ours_ns = min_ns(TRIALS, || {
            black_box(ours.call1(&args).expect("fnp.multiply call"));
        });
        let numpy_ns = min_ns(TRIALS, || {
            black_box(theirs.call1(&args).expect("numpy.multiply call"));
        });

        let accounted = import_ns + dtype_ns + ndarray_ns + empty_ns;
        println!(
            "PERCALL_FLOOR_STAGES n={N} numpy_version={numpy_version} worker={} \
             harness=stage_replica_min_of_{TRIALS} trials={TRIALS} \
             stages_are_standalone_replicas=true stage_numbers_are_lower_bounds=true \
             import_numpy_ns={import_ns:.1} dtype_guard_both_operands_ns={dtype_ns:.1} \
             getattr_ndarray_ns={ndarray_ns:.1} numpy_empty_ns={empty_ns:.1} \
             accounted_ns={accounted:.1} fnp_multiply_ns={ours_ns:.1} \
             numpy_multiply_ns={numpy_ns:.1} accounted_fraction={:.3} \
             fnp_over_numpy={:.3}",
            measurement_worker(),
            accounted / ours_ns,
            ours_ns / numpy_ns,
        );
    });
}

fn bench_route_floor_size_sweep_vs_numpy(_c: &mut Criterion) {
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy.__version__")
            .extract::<String>()
            .expect("numpy version is a string");

        // Down to 2^8 (2 KiB, L1-resident) where the kernel has essentially
        // nothing to do, so excess_ns there is very nearly the per-call cost
        // itself rather than an extrapolation (deadlock-audit-isnd2).
        for exponent in [8u32, 12, 16, 20, 24] {
            let n = 1usize << exponent;
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
            let args = PyTuple::new(py, [&a, &b]).expect("args");

            let (ratio, lo, hi, numpy_ns, fnp_ns) =
                measure_binary_ufunc_vs_numpy(py, &module, &numpy, "multiply", &args, n);
            let excess_ns = fnp_ns - numpy_ns;
            println!(
                "ROUTE_FLOOR_SWEEP op=multiply n={n} log2n={exponent} \
                 numpy_version={numpy_version} \
                 harness=common::run_dual_null_median_ci_contract \
                 ratio={ratio:.6} ratio_ci95=[{lo:.6},{hi:.6}] \
                 numpy_ns={numpy_ns:.1} fnp_ns={fnp_ns:.1} excess_ns={excess_ns:.1} \
                 excess_ns_per_element={:.6} at_parity={}",
                excess_ns / n as f64,
                hi >= 1.0,
            );
        }
    });
}

// Is the ~6.6 us per-call floor a property of PyUFunc::__call__ itself, or of
// whatever each op's body does? Reading the dispatch says f64 `add`, `subtract`
// and `multiply` all map to `None` in the f64 binop match and DELEGATE to NumPy,
// while `divide` is in that set and takes the native zerocopy route
// (deadlock-audit-cydda). So at a size where there is almost no data to touch,
// a floor that is uniform across all four is a property of the wrapper, and one
// that tracks which route the op takes is a property of the bodies.
//
// n=2^8 deliberately: 2 KiB operands are L1-resident, so the excess is nearly the
// per-call cost itself rather than an extrapolation. Uses only ops already on the
// public surface, so nothing is added to the module just to measure it.
fn bench_percall_floor_across_ops_vs_numpy(_c: &mut Criterion) {
    let n = 1usize << 8;
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy.__version__")
            .extract::<String>()
            .expect("numpy version is a string");
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
        let args = PyTuple::new(py, [&a, &b]).expect("args");

        // divide is the one of these four that takes the native f64 route.
        for (name, routes_natively) in [
            ("add", false),
            ("subtract", false),
            ("multiply", false),
            ("divide", true),
        ] {
            let (ratio, lo, hi, numpy_ns, fnp_ns) =
                measure_binary_ufunc_vs_numpy(py, &module, &numpy, name, &args, n);
            println!(
                "PERCALL_FLOOR op={name} n={n} routes_natively={routes_natively} \
                 numpy_version={numpy_version} \
                 harness=common::run_dual_null_median_ci_contract \
                 ratio={ratio:.6} ratio_ci95=[{lo:.6},{hi:.6}] \
                 numpy_ns={numpy_ns:.1} fnp_ns={fnp_ns:.1} excess_ns={:.1}",
                fnp_ns - numpy_ns,
            );
        }
    });
}
