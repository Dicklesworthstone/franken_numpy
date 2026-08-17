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
use pyo3::types::{PyAnyMethods, PyDict, PyModule, PyModuleMethods, PyTuple, PyTupleMethods};
use pyo3::{Bound, Py, PyAny, PyResult, Python, pyclass, pymethods};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::fmt::Write as _;
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
    common::gated_main_with_source(
        include_str!("criterion_python_elementwise.rs"),
        &[
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
                "bench_percall_floor_partition",
                bench_percall_floor_partition,
            ),
            (
                "bench_divide_accumulate_isolation_vs_numpy",
                bench_divide_accumulate_isolation_vs_numpy,
            ),
            (
                "bench_divide_classifier_accumulator_form",
                bench_divide_classifier_accumulator_form,
            ),
            (
                "bench_divide_kernel_on_numpy_buffers",
                bench_divide_kernel_on_numpy_buffers,
            ),
            (
                "bench_divide_allocator_provenance",
                bench_divide_allocator_provenance,
            ),
            (
                "bench_divide_provenance_counter_rust",
                bench_divide_provenance_counter_rust,
            ),
            (
                "bench_divide_provenance_counter_numpy",
                bench_divide_provenance_counter_numpy,
            ),
            (
                "bench_dtype_probe_fanout_ceiling",
                bench_dtype_probe_fanout_ceiling,
            ),
            (
                "bench_delegation_kwargs_shape",
                bench_delegation_kwargs_shape,
            ),
            (
                "bench_pyo3_signature_binding_cost",
                bench_pyo3_signature_binding_cost,
            ),
            (
                "bench_pyo3_signature_parameter_scaling",
                bench_pyo3_signature_parameter_scaling,
            ),
            (
                "bench_wrapper_remainder_stages",
                bench_wrapper_remainder_stages,
            ),
            ("bench_noop_reshape_cost", bench_noop_reshape_cost),
            (
                "bench_native_binary_family_vs_numpy",
                bench_native_binary_family_vs_numpy,
            ),
            (
                "bench_output_construction_decomposition",
                bench_output_construction_decomposition,
            ),
            (
                "bench_python_lookup_hoisting_ceiling",
                bench_python_lookup_hoisting_ceiling,
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
                "bench_add_route_floor_size_sweep_vs_numpy",
                bench_add_route_floor_size_sweep_vs_numpy,
            ),
            (
                "bench_add_tiny_n_floor_vs_numpy",
                bench_add_tiny_n_floor_vs_numpy,
            ),
            ("bench_out_kwarg_vs_numpy", bench_out_kwarg_vs_numpy),
            ("bench_accumulate_counter_fnp", bench_accumulate_counter_fnp),
            ("bench_binary_counter_add_fnp", bench_binary_counter_add_fnp),
            (
                "bench_binary_counter_add_numpy",
                bench_binary_counter_add_numpy,
            ),
            (
                "bench_binary_counter_divide_fnp",
                bench_binary_counter_divide_fnp,
            ),
            (
                "bench_binary_counter_divide_numpy",
                bench_binary_counter_divide_numpy,
            ),
            ("bench_reduce_counter_fnp", bench_reduce_counter_fnp),
            ("bench_reduce_counter_numpy", bench_reduce_counter_numpy),
            ("bench_outer_counter_fnp", bench_outer_counter_fnp),
            ("bench_outer_counter_numpy", bench_outer_counter_numpy),
            ("bench_reduceat_counter_fnp", bench_reduceat_counter_fnp),
            ("bench_reduceat_counter_numpy", bench_reduceat_counter_numpy),
            (
                "bench_accumulate_counter_numpy",
                bench_accumulate_counter_numpy,
            ),
            (
                "bench_ufunc_at_percall_floor_vs_numpy",
                bench_ufunc_at_percall_floor_vs_numpy,
            ),
            (
                "bench_predecline_levers_vs_numpy",
                bench_predecline_levers_vs_numpy,
            ),
            (
                "bench_axis_default_wrappers_vs_numpy",
                bench_axis_default_wrappers_vs_numpy,
            ),
            (
                "bench_reduceat_percall_floor_vs_numpy",
                bench_reduceat_percall_floor_vs_numpy,
            ),
            (
                "bench_accumulate_size_crossover_vs_numpy",
                bench_accumulate_size_crossover_vs_numpy,
            ),
            (
                "bench_ufunc_method_percall_floor_vs_numpy",
                bench_ufunc_method_percall_floor_vs_numpy,
            ),
            (
                "bench_percall_floor_across_ops_vs_numpy",
                bench_percall_floor_across_ops_vs_numpy,
            ),
            (
                "bench_percall_floor_across_sizes_vs_numpy",
                bench_percall_floor_across_sizes_vs_numpy,
            ),
            (
                "bench_signature_keyword_binding_cost",
                bench_signature_keyword_binding_cost,
            ),
            (
                "bench_signature_shape_pyclass_control",
                bench_signature_shape_pyclass_control,
            ),
            (
                "bench_divide_size_gate_vs_numpy",
                bench_divide_size_gate_vs_numpy,
            ),
            (
                "bench_remainder_vs_numpy_incumbent",
                bench_remainder_vs_numpy_incumbent,
            ),
            (
                "bench_delegating_probe_chain_cost",
                bench_delegating_probe_chain_cost,
            ),
            ("bench_probe_decline_ordering", bench_probe_decline_ordering),
            ("bench_maximum_arms_vs_numpy", bench_maximum_arms_vs_numpy),
            (
                "bench_incumbent_interference_from_candidate",
                bench_incumbent_interference_from_candidate,
            ),
            (
                "bench_incumbent_interference_remainder_route",
                bench_incumbent_interference_remainder_route,
            ),
            (
                "bench_incumbent_interference_shadow_held_constant",
                bench_incumbent_interference_shadow_held_constant,
            ),
            (
                "bench_interference_vs_incumbent_duration",
                bench_interference_vs_incumbent_duration,
            ),
        ],
    );
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

/// Emits the shipped fused serial divide loop under a caller-chosen name
/// (`deadlock-audit-ascyl`).
///
/// WHY A MACRO AND NOT ONE FUNCTION CALLED TWICE. The question is whether buffer
/// PROVENANCE — Rust `Vec` versus numpy-allocated — changes the loop's cost. That
/// needs the two arms separable by `perf` symbol, and one function called with two
/// different slices is ONE symbol. It equally needs the two bodies to be identical,
/// or the comparison measures codegen instead of memory. Generating both from one
/// macro makes the source textually identical by construction; the group then
/// checks the claim rather than asserting it, by disassembling both and comparing
/// their main-loop instruction census (see the group's comment).
macro_rules! emit_divide_fused_serial {
    ($name:ident) => {
        #[inline(never)]
        fn $name(a: &[f64], b: &[f64], out: &mut [f64]) -> bool {
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
    };
}

emit_divide_fused_serial!(divide_fused_on_rust_vec);
emit_divide_fused_serial!(divide_fused_on_numpy_buffer);

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

/// The same classifier, re-expressed so the reduction is one `OR` chain over
/// exponent evidence instead of a boolean OR over per-lane comparisons
/// (`deadlock-audit-6y5wp`).
///
/// WHY THIS SHAPE. Disassembled from a local release ELF, the fused loop above
/// costs SEVEN vector ops per 4 doubles to carry the flag — `vandpd`, `vpaddq`,
/// `vpxor`, `vpcmpgtq`, `vextracti128`, `vpackssdw`, `vpor` — of which the last
/// three exist only to narrow a 4-lane compare mask down to a `bool`, and
/// `vextracti128` is cross-lane. Accumulating a `u64` keeps the reduction inside
/// one lane-local `vpor` chain, so the narrowing disappears.
///
/// ISOMORPHISM PROOF. Let `e = bits & EXPONENT_MASK`, so `e` is exactly one of
/// `{0, 0x0010…, 0x0020…, …, 0x7ff0…}`.
///   * `e.wrapping_sub(1)` has bit 63 set  <=>  `e == 0`.
///     For `e >= 0x0010…` we get `e - 1 <= 0x7fef_ffff_ffff_ffff < 2^63`;
///     for `e == 0` we get `0xffff_ffff_ffff_ffff`.
///   * `e.wrapping_add(EXPONENT_STEP)` has bit 63 set  <=>  `e == 0x7ff0…`.
///     For `e <= 0x7fe0…` we get at most `0x7ff0… < 2^63`;
///     for `e == 0x7ff0…` we get exactly `0x8000_0000_0000_0000`.
///
/// `bench_divide_quotient_is_normal` is false exactly when `e` is `0` or
/// `0x7ff0…`, so OR-ing both terms over every element and testing bit 63 once at
/// the end decides the identical predicate as OR-ing the per-element booleans.
/// The rare exact second pass is unchanged, so a hazard verdict is unchanged too.
///
/// THE NEGATIVE CASE. Dropping the `wrapping_add` term still catches `±0` and
/// subnormal quotients, so an implementation missing it looks correct on the
/// obvious inputs and silently stops deferring on `inf`/`nan` — the exact
/// divergence class `deadlock-audit-2nmd1` closed.
/// `assert_bitmask_classifier_matches_boolean_classifier` pins both halves.
#[inline(never)]
fn divide_bitmask_fused_serial(a: &[f64], b: &[f64], out: &mut [f64]) -> bool {
    const EXPONENT_MASK: u64 = 0x7ff0_0000_0000_0000;
    const EXPONENT_STEP: u64 = 0x0010_0000_0000_0000;
    let mut evidence: u64 = 0;
    for ((slot, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        let quotient = x / y;
        *slot = quotient;
        let exponent = quotient.to_bits() & EXPONENT_MASK;
        evidence |= exponent.wrapping_sub(1) | exponent.wrapping_add(EXPONENT_STEP);
    }
    evidence >> 63 != 0
        && a.iter()
            .zip(b.iter())
            .zip(out.iter())
            .any(|((&x, &y), &q)| bench_divide_raises_fp_error(x, y, q))
}

/// Pins the bitmask accumulator against the boolean one it replaces, on operand
/// pairs chosen so that each non-normal exponent class is the ONLY thing that
/// separates a case from an all-normal control (`deadlock-audit-6y5wp`).
///
/// The `inf`/`nan` cases are the load-bearing ones: an accumulator built from
/// `e.wrapping_sub(1)` alone passes every zero/subnormal case and fails only
/// these, which is why they are here and why this assertion runs before timing.
fn assert_bitmask_classifier_matches_boolean_classifier() {
    // (numerator, divisor, what the quotient's exponent field exercises)
    let cases: &[(f64, f64, &str)] = &[
        (6.0, 3.0, "normal control"),
        (1.0, 0.0, "+inf quotient: exponent all ones"),
        (-1.0, 0.0, "-inf quotient: exponent all ones"),
        (0.0, 0.0, "nan quotient: exponent all ones"),
        (f64::INFINITY, f64::INFINITY, "nan from inf/inf"),
        (0.0, 4.0, "+0 quotient: exponent zero"),
        (-0.0, 4.0, "-0 quotient: exponent zero"),
        (f64::MIN_POSITIVE, 4.0, "subnormal quotient: exponent zero"),
        (5e-324, 2.0, "subnormal flushed toward zero"),
        (f64::MAX, 0.5, "+inf by overflow"),
        (f64::MIN_POSITIVE, 1.0, "min normal stays normal"),
        (f64::MAX, 1.0, "max normal stays normal"),
    ];
    for (numerator, divisor, what) in cases {
        // One exceptional pair embedded in an otherwise all-normal run, so the
        // accumulators must survive being OR-ed with many normal quotients.
        let mut a = vec![6.0_f64; 37];
        let mut b = vec![3.0_f64; 37];
        a[17] = *numerator;
        b[17] = *divisor;
        let mut boolean_out = vec![0.0_f64; a.len()];
        let mut bitmask_out = vec![0.0_f64; a.len()];
        let boolean_verdict = divide_fused_serial(&a, &b, &mut boolean_out);
        let bitmask_verdict = divide_bitmask_fused_serial(&a, &b, &mut bitmask_out);
        assert_eq!(
            boolean_verdict, bitmask_verdict,
            "bitmask accumulator disagrees with the boolean classifier on {what}"
        );
        assert!(
            boolean_out
                .iter()
                .zip(bitmask_out.iter())
                .all(|(l, r)| l.to_bits() == r.to_bits()),
            "bitmask accumulator changed the quotients on {what}"
        );
    }
    // A run with NO exceptional element must report no hazard from either form,
    // or the assertion above would be satisfied by two arms that both say `true`.
    let a = vec![6.0_f64; 37];
    let b = vec![3.0_f64; 37];
    let mut out = vec![0.0_f64; a.len()];
    assert!(
        !divide_fused_serial(&a, &b, &mut out),
        "the all-normal control must not report a hazard"
    );
    assert!(
        !divide_bitmask_fused_serial(&a, &b, &mut out),
        "the all-normal control must not report a hazard under the bitmask form"
    );
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
        // `import_numpy_ns` IS NO LONGER A LIVE ROUTE COST, and counting it as
        // accounted overstates how well this wrapper is understood — in the
        // FLATTERING direction, which is the direction that stops people hunting.
        // `e4cdd808`, `f8475a76` and `5c7735da` replaced every per-call
        // `py.import("numpy")` on the binary ufunc route — the one at the top of
        // `PyUFunc::__call__` and the one each probe was making before it declined
        // — with the cached `cached_numpy` module handle. The route never executes
        // this stage now, so its replica cannot be part of the live total.
        //
        // THE REPLICA IS KEPT AND STILL PRINTED, deliberately. Deleting it would
        // silently break comparability with every pre-fix row whose sums include
        // it (`deadlock-audit-cydda`, and the invalidated `thinkstation1` row) —
        // those rows would then be reading a differently-defined `accounted_ns`
        // under the same name, which is worse than an overstatement because it is
        // invisible. So both totals are emitted, each labelled with what it
        // contains, and `accounted_fraction` keeps its historical meaning.
        // `deadlock-audit-kido6`.
        let live_accounted = dtype_ns + ndarray_ns + empty_ns;
        let live_unattributed = ours_ns - live_accounted;
        let worker = measurement_worker();
        let accounted_fraction = accounted / ours_ns;
        let live_unattributed_fraction = live_unattributed / ours_ns;
        let fnp_over_numpy = ours_ns / numpy_ns;
        println!(
            "PERCALL_FLOOR_STAGES n={N} numpy_version={numpy_version} worker={worker} \
             harness=stage_replica_min_of_{TRIALS} trials={TRIALS} \
             stages_are_standalone_replicas=true stage_numbers_are_lower_bounds=true \
             import_numpy_ns={import_ns:.1} import_is_live_route_cost=false \
             dtype_guard_both_operands_ns={dtype_ns:.1} \
             getattr_ndarray_ns={ndarray_ns:.1} numpy_empty_ns={empty_ns:.1} \
             accounted_ns={accounted:.1} accounted_includes_dead_import=true \
             accounted_fraction={accounted_fraction:.3} \
             live_accounted_ns={live_accounted:.1} \
             live_unattributed_ns={live_unattributed:.1} \
             live_unattributed_fraction={live_unattributed_fraction:.3} \
             fnp_multiply_ns={ours_ns:.1} numpy_multiply_ns={numpy_ns:.1} \
             fnp_over_numpy={fnp_over_numpy:.3}"
        );
        // The live partition must be exact by construction: every nanosecond of the
        // call is either in a named live stage or in the unattributed remainder.
        // A future edit that adds a stage to one total and forgets the other shows
        // up here rather than as a quietly wrong fraction in a banked row.
        assert!(
            (live_accounted + live_unattributed - ours_ns).abs() < 1e-6,
            "live stages must partition the whole call: {live_accounted:.1} + \
             {live_unattributed:.1} != {ours_ns:.1}"
        );
    });
}

// PARTITION attribution of the per-call floor — the successor method to the stage
// replicas above (`deadlock-audit-ei9jz`).
//
// WHY A NEW METHOD. `bench_percall_floor_stage_attribution` prices stages as
// STANDALONE REPLICAS and reached accounted_fraction=0.397 on thinkstation1: 950 ns
// of a 2394 ns call. Seven stages have now been priced that way and the floor has
// not fallen to any of them. A replica can only price a stage someone already
// suspected, so the method structurally cannot find the remaining 60%.
//
// WHAT THIS DOES INSTEAD. It PARTITIONS the real call into three pieces that are
// individually measured, mutually exclusive, and exhaustive by construction:
//
//   numpy_multiply_ns    the incumbent's own work, which we also pay when we delegate
//   probe_chain_ns       what we spend DECIDING not to take a fast path
//   wrapper_residual_ns  PyO3 entry + kwarg binding + the delegation call itself
//
// The probe chain is separated using the control that already exists in shipped
// code and needs no instrumentation: the entire probe block in `PyUFunc::__call__`
// sits behind a guard requiring `casting == "same_kind"`, so passing any other
// casting value skips EVERY probe (`deadlock-audit-wsd7h`).
//
// THAT SUBTRACTION IS CONTAMINATED and the correction below cancels the keyword
// tail rather than avoiding it (`deadlock-audit-uj3r3`, found by RedLynx when this
// group's own guard panicked at -40 ns; the mechanism and the cancellation are
// documented at the correction site, not repeated here).
//
// WHY NO ARM SHAPE AVOIDS IT — the reason there is no "probes on, keywords on" call
// to compare against. `deadlock-audit-s2fkk`'s delegation fast path is guarded by
// the IDENTICAL boolean expression as the probe block:
//
//     out.is_none() && where.is_none() && dtype.is_none() && signature.is_none()
//       && casting == "same_kind" && order == "K" && subok
//
// and it tests VALUES, not keyword presence. The two costs are therefore perfectly
// coupled: every input that disables the probes also forces the keyword tail, and
// every input that keeps the fast tail also keeps the probes. Passing `order="K"`
// explicitly does not split them either, because "K" is the value the guard tests
// for. So giving both arms the same non-default kwarg — the obvious fix — would
// disable the probes in BOTH arms and leave nothing to measure.
//
// HONESTY ABOUT WHAT THE FRACTION MEANS. Under a partition accounted_fraction is
// 1.0 BY CONSTRUCTION and is therefore NOT evidence of anything. The informative
// output is the SPLIT — how 2394 ns divides across the three pieces — and the fact
// that the pieces are measured rather than inferred. A reader who quotes
// "accounted_fraction=1.000" as progress has misread this group. `multiply` is used
// because it is NOT in the fast-path binop set (Remainder/Power/Maximum/Minimum/
// Divide), so it exercises the DELEGATING route, which is the route the floor
// belongs to.
//
// NEGATIVE CASE this asserts, which a naive implementation would fail: a partition
// built by summing overlapping in-situ measurements double-counts. Every piece is
// asserted non-negative and the three are asserted to sum to the whole call within
// tolerance, so an implementation that measures overlapping regions and adds them
// panics here rather than publishing a flattering fraction.
fn bench_percall_floor_partition(_c: &mut Criterion) {
    const N: usize = 256;
    const TRIALS: usize = 2001;

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

        // The probe-skipping arm must still produce numpy's answer, or the two arms
        // are not the same computation and the subtraction is meaningless.
        let probed_probe = ours.call1(&args).expect("fnp.multiply probe");
        let unsafe_kwargs = PyDict::new(py);
        unsafe_kwargs
            .set_item("casting", "unsafe")
            .expect("bind casting");
        // Both arms pass the SAME prebuilt `args` tuple. Constructing a fresh tuple
        // in one arm and reusing a prebuilt one in the other would charge that
        // construction to the probe-skipped arm alone, shrinking `probe_chain_ns`
        // by the difference and, on a cheap probe chain, driving it negative.
        let skipped_probe = ours
            .call(&args, Some(&unsafe_kwargs))
            .expect("fnp.multiply casting=unsafe probe");
        assert_eq!(
            numpy_divide_checksum(&probed_probe, N),
            numpy_divide_checksum(&skipped_probe, N),
            "the probed and probe-skipped arms disagree — they are not the same call"
        );

        // The three keywords our SLOW TAIL always sets when any kwarg is non-default
        // (`casting`, `order`, `subok`). Handed to NumPy's own ufunc below to price the
        // parse half of `kwargs_overhead_ns` without involving our probe chain.
        let tail_kwargs = PyDict::new(py);
        tail_kwargs
            .set_item("casting", "unsafe")
            .expect("bind tail casting");
        tail_kwargs.set_item("order", "K").expect("bind tail order");
        tail_kwargs
            .set_item("subok", true)
            .expect("bind tail subok");

        // Whole call, full probe chain, every kwarg at its default.
        let whole_ns = min_ns(TRIALS, || {
            black_box(ours.call1(&args).expect("fnp.multiply call"));
        });
        // Same call with every probe skipped by the shipped casting guard.
        let skipped_ns = min_ns(TRIALS, || {
            black_box(
                ours.call(&args, Some(&unsafe_kwargs))
                    .expect("fnp.multiply casting=unsafe call"),
            );
        });
        // The incumbent's own work, which the delegation tail also pays.
        let numpy_ns = min_ns(TRIALS, || {
            black_box(theirs.call1(&args).expect("numpy.multiply call"));
        });

        // THE TWO ARMS DIFFER BY MORE THAN THE PROBE CHAIN, and correcting for that is
        // what makes this partition valid again (`deadlock-audit-uj3r3`).
        //
        // The probe block is gated on ALL SEVEN of out/where/dtype/signature being absent
        // and casting/order/subok being at their defaults, and `deadlock-audit-s2fkk`'s
        // fast path keys off the SAME predicate. So "probes engaged" and "no keyword tail"
        // are not two knobs, they are one: the all-defaults arm sends NO keywords, while
        // the `casting="unsafe"` arm falls to the tail that allocates a PyDict, sets
        // casting/order/subok, and makes NumPy parse three keywords — a shape s2fkk priced
        // at ~727 ns. The raw subtraction is therefore `probe_chain MINUS kwargs`, and
        // once the per-call imports were removed from the probes (f8475a76, 5c7735da) the
        // kwargs term dominated and drove it NEGATIVE (-40.0 ns), tripping the guard below.
        //
        // There is no spelling of "probes on, keywords on" to compare against, so the term
        // is CANCELLED rather than eliminated. Both halves are measurable:
        //   (a) NumPy's own parse of the same three keywords our tail always sets, taken on
        //       NumPy's ufunc so no probe chain is involved;
        //   (b) our PyDict construction, as a standalone replica.
        // Replica-based, so a LOWER BOUND, which is the same caveat the stage decomposition
        // already carries — and it is checked below against s2fkk's independent figure.
        let numpy_kwargs_ns = min_ns(TRIALS, || {
            black_box(
                theirs
                    .call(&args, Some(&tail_kwargs))
                    .expect("numpy.multiply with the tail's three keywords"),
            );
        });
        let pydict_build_ns = min_ns(TRIALS, || {
            let kwargs = PyDict::new(py);
            kwargs.set_item("casting", "unsafe").expect("set casting");
            kwargs.set_item("order", "K").expect("set order");
            kwargs.set_item("subok", true).expect("set subok");
            black_box(kwargs);
        });
        let kwargs_overhead_ns = (numpy_kwargs_ns - numpy_ns) + pydict_build_ns;
        // What the probe-skipping arm WOULD have cost had it taken the fast tail, i.e. the
        // arm the probed one should have been compared against all along.
        let skipped_fast_tail_ns = skipped_ns - kwargs_overhead_ns;
        let probe_chain_ns = whole_ns - skipped_fast_tail_ns;
        let wrapper_residual_ns = skipped_fast_tail_ns - numpy_ns;

        // The correction must be POSITIVE: the keyword tail does strictly more work than
        // the fast tail — it builds a PyDict and hands NumPy three keywords to parse. A
        // non-positive value means the replica is not capturing that work at all, and a
        // correction that is not measuring anything must not be silently applied.
        assert!(
            kwargs_overhead_ns > 0.0,
            "kwargs overhead measured NON-POSITIVE ({kwargs_overhead_ns:.1} ns): the keyword \
             tail cannot be cheaper than the fast tail, so this correction is invalid"
        );
        // Independent sanity check on the correction, emitted rather than asserted:
        // `deadlock-audit-s2fkk` priced this same call shape at ~727 ns by a different
        // method. A ratio far from 1.0 means the correction is mis-scaled and the
        // probe-chain figure below should not be trusted — but the threshold is a
        // judgement for the reader, not a hard gate, since the two methods differ and
        // 727 ns was measured on another host.
        let kwargs_overhead_vs_s2fkk = kwargs_overhead_ns / 727.0;

        // A partition cannot have negative parts. If the probe-skipping arm is not
        // cheaper than the probed one, or delegation is not dearer than NumPy's own
        // call, the subtraction has been contaminated and the row must not publish.
        assert!(
            probe_chain_ns >= 0.0,
            "probe chain measured NEGATIVE ({probe_chain_ns:.1} ns) even after cancelling \
             the {kwargs_overhead_ns:.1} ns keyword tail: the corrected probe-skipping arm \
             was still not cheaper than the probed arm, so this partition is invalid"
        );
        assert!(
            wrapper_residual_ns >= 0.0,
            "wrapper residual measured NEGATIVE ({wrapper_residual_ns:.1} ns): our delegating \
             call was faster than numpy's own call, so this partition is invalid"
        );
        // NOT asserted here: that the three pieces sum to the whole call. That check
        // would be TAUTOLOGICAL — the partition telescopes
        // (numpy + (whole - skipped) + (skipped - numpy) == whole identically), so it
        // can never fail and would be testing this function's arithmetic rather than
        // the measurement. The two non-negativity assertions above are the real
        // gate: each compares two INDEPENDENTLY timed arms and fails when the
        // subtraction is contaminated.
        //
        // The premise check that CAN fail: we are supposed to be slower than the
        // incumbent on this route, and the probe-skipped arm still delegates to it.
        // If either stops holding, the partition is describing a different route
        // than the one this row claims to measure.
        assert!(
            whole_ns > numpy_ns,
            "fnp.multiply ({whole_ns:.1} ns) was not slower than numpy.multiply \
             ({numpy_ns:.1} ns) — this row's premise, that the delegating route carries \
             a per-call floor, does not hold on this host and the split is meaningless"
        );
        assert!(
            skipped_ns >= numpy_ns,
            "the probe-skipped arm ({skipped_ns:.1} ns) was faster than the numpy call \
             it delegates to ({numpy_ns:.1} ns) — impossible if it really delegates, so \
             the casting=unsafe arm is not taking the route this row assumes"
        );

        println!(
            "PERCALL_FLOOR_PARTITION n={N} numpy_version={numpy_version} worker={} \
             harness=partition_min_of_{TRIALS} trials={TRIALS} op=multiply \
             route=delegating partition_is_exhaustive_by_construction=true \
             accounted_fraction_is_tautological=true \
             probe_separation=shipped_casting_guard_deadlock-audit-wsd7h \
             fnp_multiply_ns={whole_ns:.1} probes_skipped_raw_ns={skipped_ns:.1} \
             numpy_kwargs_ns={numpy_kwargs_ns:.1} pydict_build_ns={pydict_build_ns:.1} \
             kwargs_overhead_ns={kwargs_overhead_ns:.1} \
             kwargs_overhead_vs_s2fkk_727ns={kwargs_overhead_vs_s2fkk:.3} \
             correction=deadlock-audit-uj3r3_keyword_tail_cancelled \
             probes_skipped_ns={skipped_fast_tail_ns:.1} \
             numpy_multiply_ns={numpy_ns:.1} probe_chain_ns={probe_chain_ns:.1} \
             wrapper_residual_ns={wrapper_residual_ns:.1} \
             probe_chain_share={:.3} wrapper_residual_share={:.3} numpy_share={:.3} \
             fnp_over_numpy={:.3}",
            measurement_worker(),
            probe_chain_ns / whole_ns,
            wrapper_residual_ns / whole_ns,
            numpy_ns / whole_ns,
            whole_ns / numpy_ns,
        );
    });
}

// Does the f64 divide deficit come from the NORMALITY ACCUMULATE or from our divide
// CODEGEN? (`deadlock-audit-0ppym`)
//
// WHAT IS ALREADY DECIDED, and why this group is the next question rather than a
// repeat. On thinkstation1 at n=2^20, `multiply` — which shares every route stage
// with `divide` by construction and computes NO hazard scan — reads 0.965590 against
// NumPy while `divide` reads 0.845461. Divide's excess over NumPy is 9.52x
// multiply's, and the route sweep shows `excess_ns` roughly constant from 2^8 to
// 2^24, reaching parity by 2^24. So the wrapper route is exonerated and ~84 us of
// divide's ~93 us deficit at 2^20 is divide-specific.
//
// WHAT IS NOT DECIDED. That subtraction is across two DIFFERENT ops, so it cannot
// separate the accumulate from our divide codegen. `bench_divide_fe_hazard_serial`
// and `bench_divide_fe_hazard_fused_serial` compare our-before against our-after —
// self-speedup — and structurally cannot settle it either. This group puts BOTH
// replica shapes against the INCUMBENT in one invocation:
//
//   `divide_former_serial` writes quotients and accumulates nothing.
//   `divide_fused_serial`  is the replica of the shipped arm, accumulating the
//                          normality flag beside the divide.
//
// If the accumulate-free arm lands near multiply's 0.9656 the accumulate is the
// cost and `deadlock-audit-vqxoa`'s "close to free" is refuted against the
// incumbent. If it still reads ~0.845 the accumulate is innocent and the cost is
// our divide codegen — a different lever.
//
// NEITHER ARM ALLOCATES, which is the whole reason this is a fair kernel
// comparison. The incumbent is given a preallocated `out=` array so NumPy does not
// allocate either, and both replicas write into a preallocated Vec. Comparing an
// in-place Rust loop against an allocating `numpy.divide(a, b)` would have flattered
// us by the cost of one 8 MB buffer, which at this size is not small.
//
// THIS ARM MUST NEVER SHIP. `divide_former_serial` has no FE-hazard deferral;
// routing it in production would reintroduce the six divergence rows
// `deadlock-audit-2nmd1` closed, because NumPy also raises a RuntimeWarning we
// cannot, and bit-identical values are not parity. It exists here to attribute a
// cost, not to be adopted.
fn bench_divide_accumulate_isolation_vs_numpy(_c: &mut Criterion) {
    assert_divide_hazard_replica_matches_contract();
    let n = DIVIDE_SERIAL_N;
    let (a_vec, b_vec) = divide_hazard_free_operands(n);

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let numpy = py.import("numpy").expect("numpy oracle");
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy.__version__")
            .extract::<String>()
            .expect("numpy version is a string");

        let locals = PyDict::new(py);
        locals.set_item("np", &numpy).expect("bind numpy");
        locals.set_item("n", n).expect("bind n");
        // The SAME generator the Rust replicas use, so both sides divide identical
        // operands and the checksums are comparable rather than merely similar.
        py.run(
            std::ffi::CString::new(
                "i = np.arange(n)\na = 1.0 + (i % 1000) / 1000.0\nb = 1.25 + (i % 997) / 997.0\nout = np.empty(n)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&locals),
            Some(&locals),
        )
        .expect("build operands");
        let a_py = locals.get_item("a").expect("a operand");
        let b_py = locals.get_item("b").expect("b operand");
        let out_py = locals.get_item("out").expect("out buffer");
        let args = PyTuple::new(py, [&a_py, &b_py]).expect("args");
        let out_kwargs = PyDict::new(py);
        out_kwargs.set_item("out", &out_py).expect("bind out");
        let numpy_divide = numpy.getattr("divide").expect("numpy.divide");

        // Parity gate before any timing: if a replica does not reproduce NumPy's
        // quotients bit for bit, the ratio it produces is meaningless.
        let numpy_result = numpy_divide
            .call(&args, Some(&out_kwargs))
            .expect("numpy.divide probe");
        let numpy_sum = numpy_divide_checksum(&numpy_result, n);
        let mut probe_out = vec![0.0_f64; n];
        divide_former_serial(&a_vec, &b_vec, &mut probe_out);
        assert_eq!(
            divide_checksum(&probe_out),
            numpy_sum,
            "the accumulate-free replica does not reproduce numpy.divide bit for bit"
        );
        let fused_hazard = divide_fused_serial(&a_vec, &b_vec, &mut probe_out);
        assert!(
            !fused_hazard,
            "these operands must be hazard-free or the fused arm takes its rare \
             second pass and the comparison measures a different branch"
        );
        assert_eq!(
            divide_checksum(&probe_out),
            numpy_sum,
            "the fused replica does not reproduce numpy.divide bit for bit"
        );

        let mut former_out = vec![0.0_f64; n];
        let mut fused_out = vec![0.0_f64; n];

        let incumbent_former = || {
            let started = Instant::now();
            let result = numpy_divide
                .call(&args, Some(&out_kwargs))
                .expect("numpy.divide call");
            let elapsed = started.elapsed();
            let checksum = numpy_divide_checksum(&result, n);
            common::ContractObservation { elapsed, checksum }
        };
        let candidate_former = || {
            let started = Instant::now();
            divide_former_serial(&a_vec, &b_vec, &mut former_out);
            let elapsed = started.elapsed();
            let checksum = divide_checksum(&former_out);
            black_box(&former_out);
            common::ContractObservation { elapsed, checksum }
        };
        let (former_effect, _in_null, _cand_null) = common::run_dual_null_median_ci_contract(
            "divide_f64_1m_accumulate_free_vs_numpy",
            incumbent_former,
            candidate_former,
        );

        let incumbent_fused = || {
            let started = Instant::now();
            let result = numpy_divide
                .call(&args, Some(&out_kwargs))
                .expect("numpy.divide call");
            let elapsed = started.elapsed();
            let checksum = numpy_divide_checksum(&result, n);
            common::ContractObservation { elapsed, checksum }
        };
        let candidate_fused = || {
            let started = Instant::now();
            let hazard = divide_fused_serial(&a_vec, &b_vec, &mut fused_out);
            let elapsed = started.elapsed();
            assert!(!hazard, "operands must stay hazard-free during timing");
            let checksum = divide_checksum(&fused_out);
            black_box(&fused_out);
            common::ContractObservation { elapsed, checksum }
        };
        let (fused_effect, _in_null2, _cand_null2) = common::run_dual_null_median_ci_contract(
            "divide_f64_1m_fused_accumulate_vs_numpy",
            incumbent_fused,
            candidate_fused,
        );

        // The attribution: how much of the divide deficit the accumulate carries.
        // Reported as a difference of two ratios measured against the SAME
        // incumbent in the SAME invocation, which is the only way these two are
        // comparable — quoting either against a remembered number would not be.
        let accumulate_cost_ns = fused_effect.arm_b_median_ns - former_effect.arm_b_median_ns;
        println!(
            "DIVIDE_ACCUMULATE_ISOLATION n={n} numpy_version={numpy_version} worker={} \
             harness=common::run_dual_null_median_ci_contract \
             arms_are_preallocated_no_alloc_either_side=true \
             accumulate_free_ratio={:.6} accumulate_free_ci95=[{:.6},{:.6}] \
             fused_ratio={:.6} fused_ci95=[{:.6},{:.6}] \
             numpy_ns={:.1} accumulate_free_ns={:.1} fused_ns={:.1} \
             accumulate_cost_ns={accumulate_cost_ns:.1} \
             multiply_same_route_reference_ratio=0.965590 \
             shipped_route_divide_reference_ratio=0.845461",
            measurement_worker(),
            former_effect.ratio_median,
            former_effect.ratio_ci_low,
            former_effect.ratio_ci_high,
            fused_effect.ratio_median,
            fused_effect.ratio_ci_low,
            fused_effect.ratio_ci_high,
            former_effect.arm_a_median_ns,
            former_effect.arm_b_median_ns,
            fused_effect.arm_b_median_ns,
        );
    });
}

// Is the shipped divide classifier expensive because of what it COMPUTES, or
// because of the SHAPE its reduction is lowered to? (`deadlock-audit-6y5wp`)
//
// WHY THIS GROUP EXISTS, and why it is not another guess. `deadlock-audit-6y5wp`
// listed three candidate levers for the 94.0 us kernel gap — restore packing,
// split the accumulate out of the hot loop, wider unroll. A static census of the
// three loops involved, taken from a local release ELF and from the numpy the
// same host imports, closes two of them:
//
//   loop                                     insns/iter  doubles/iter  unroll
//   numpy DOUBLE_divide_X86_V3 @0x523e00         10           8         2x ymm
//   ours divide_former_serial  @0x700e20          6           4         1x ymm
//   ours divide_fused_serial   @0x6feda0         43          16         4x ymm
//
// Every one of the three is packed `vdivpd` on ymm, so "restore packing" is
// refuted (RedLynx reached the same conclusion from the shipped route). And the
// 4x-unrolled arm is the SLOWEST of the three while the 2x arm is the fastest,
// so "wider unroll" is refuted too: LLVM already unrolls our fused arm twice as
// wide as numpy's and it does not help.
//
// What the census DOES leave standing is the third candidate, sharpened. Of the
// fused arm's 43 instructions per 16 doubles, 4 are divides (each taking its
// divisor straight from memory), 8 are the numerator loads and quotient stores
// every arm pays, 3 are loop control — and the remaining 28 are the classifier,
// seven vector ops per 4 doubles: `vandpd`, `vpaddq`, `vpxor`, `vpcmpgtq`,
// `vextracti128`, `vpackssdw`, `vpor`. The last three are not arithmetic at all;
// they narrow a 4-lane compare mask into the `bool` that `saw_non_normal |= …`
// asks for, and `vextracti128` is cross-lane. `divide_bitmask_fused_serial`
// computes the identical predicate with an OR over exponent evidence, which needs
// no narrowing (proof on that function).
//
// WHAT WOULD MAKE THIS LEVER DEAD, registered BEFORE the run: at n=2^20 this loop
// issues 262144 `vdivpd`, and if the 256-bit divider is the binding constraint
// then the classifier's ops are already hidden underneath it, removing eight of
// them per 16 doubles buys nothing, and the head-to-head ratio moves by less than
// the null spread.
//
// THAT IS ESSENTIALLY WHAT HAPPENED, and the group is kept to say so. The bitmask
// arm does emit the codegen it was designed to emit — 35 instructions per 16
// doubles against the boolean arm's 43, classifier ops 28 -> 20, and no cross-lane
// `vextracti128` left inside the loop — and it buys almost nothing. `perf record
// -e cycles:u,instructions:u` attributed by symbol over a full run of this group
// is the counted form: retired instructions fall 18.4% (matching the static census
// 35/43 = 0.814 to within 0.3%) and CYCLES FALL 0.76%. The classifier is in the
// divider's shadow.
//
// READ THE WALL CLOCK OVER RUNS, NOT WITHIN ONE. Five runs of one ELF put the
// head-to-head at 1.010865 / 1.049398 / 1.026644 / 1.036853 / 1.014776 — median
// 1.027, every one above unity, but R1's and R2's 95% intervals are DISJOINT and
// the between-run stdev (0.0159) exceeds the mean within-run half-width (0.0138).
// One contract cannot decide a sub-5% effect here. So the lever is a real ~2.7%
// self-speedup, not zero and not the 18.4% its instruction count suggests, and the
// shipped `zerocopy_f64_binary_flat` arm was NOT changed on the strength of it —
// see `deadlock-audit-6y5wp` in docs/NEGATIVE_EVIDENCE.md.
//
// PIN `OPENBLAS_NUM_THREADS=1` WHEN RUNNING THIS. The same perf record found
// `blas_thread_server` burning 18.4% of the process's user cycles at IPC 0.408,
// spinning on the very cores the arms are pinned to.
//
// The vs-numpy rows are why the head-to-head is here at all. On the first run the
// two candidates' ratios looked cleanly separated (0.750216 vs 0.820084, CIs
// disjoint) — but the two contracts are different schedules, numpy itself drifted
// 383021 -> 398134 ns between them, and on the re-run with the head-to-head added
// the same two ratios came back OVERLAPPING (0.799745 vs 0.811567). The apparent
// separation was the host. Only the same-schedule head-to-head decides this.
//
// BOTH ARMS ARE REPLICAS, not the shipped route, and neither allocates: numpy is
// handed a preallocated `out=` and both replicas write into preallocated Vecs.
// The two candidate ratios are measured against the SAME incumbent in the SAME
// invocation, which is the only thing that makes their difference readable.
fn bench_divide_classifier_accumulator_form(_c: &mut Criterion) {
    assert_divide_hazard_replica_matches_contract();
    assert_bitmask_classifier_matches_boolean_classifier();
    let n = DIVIDE_SERIAL_N;
    let (a_vec, b_vec) = divide_hazard_free_operands(n);

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
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
                "i = np.arange(n)\na = 1.0 + (i % 1000) / 1000.0\nb = 1.25 + (i % 997) / 997.0\nout = np.empty(n)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&locals),
            Some(&locals),
        )
        .expect("build operands");
        let a_py = locals.get_item("a").expect("a operand");
        let b_py = locals.get_item("b").expect("b operand");
        let out_py = locals.get_item("out").expect("out buffer");
        let args = PyTuple::new(py, [&a_py, &b_py]).expect("args");
        let out_kwargs = PyDict::new(py);
        out_kwargs.set_item("out", &out_py).expect("bind out");
        let numpy_divide = numpy.getattr("divide").expect("numpy.divide");

        // Parity gate before any timing, on the real operands rather than the
        // crafted ones: a replica whose quotients are not numpy's bit for bit
        // produces a meaningless ratio.
        let numpy_result = numpy_divide
            .call(&args, Some(&out_kwargs))
            .expect("numpy.divide probe");
        let numpy_sum = numpy_divide_checksum(&numpy_result, n);
        let mut probe_out = vec![0.0_f64; n];
        let fused_hazard = divide_fused_serial(&a_vec, &b_vec, &mut probe_out);
        assert!(
            !fused_hazard,
            "these operands must be hazard-free or an arm takes its rare second \
             pass and the comparison measures a different branch"
        );
        assert_eq!(
            divide_checksum(&probe_out),
            numpy_sum,
            "the fused replica does not reproduce numpy.divide bit for bit"
        );
        let bitmask_hazard = divide_bitmask_fused_serial(&a_vec, &b_vec, &mut probe_out);
        assert!(
            !bitmask_hazard,
            "the bitmask arm must agree with the fused arm that these operands are \
             hazard-free"
        );
        assert_eq!(
            divide_checksum(&probe_out),
            numpy_sum,
            "the bitmask replica does not reproduce numpy.divide bit for bit"
        );

        let mut fused_out = vec![0.0_f64; n];
        let mut bitmask_out = vec![0.0_f64; n];

        let incumbent_fused = || {
            let started = Instant::now();
            let result = numpy_divide
                .call(&args, Some(&out_kwargs))
                .expect("numpy.divide call");
            let elapsed = started.elapsed();
            let checksum = numpy_divide_checksum(&result, n);
            common::ContractObservation { elapsed, checksum }
        };
        let candidate_fused = || {
            let started = Instant::now();
            let hazard = divide_fused_serial(&a_vec, &b_vec, &mut fused_out);
            let elapsed = started.elapsed();
            assert!(!hazard, "operands must stay hazard-free during timing");
            let checksum = divide_checksum(&fused_out);
            black_box(&fused_out);
            common::ContractObservation { elapsed, checksum }
        };
        let (fused_effect, _in_null, _cand_null) = common::run_dual_null_median_ci_contract(
            "divide_f64_1m_boolean_classifier_vs_numpy",
            incumbent_fused,
            candidate_fused,
        );

        let incumbent_bitmask = || {
            let started = Instant::now();
            let result = numpy_divide
                .call(&args, Some(&out_kwargs))
                .expect("numpy.divide call");
            let elapsed = started.elapsed();
            let checksum = numpy_divide_checksum(&result, n);
            common::ContractObservation { elapsed, checksum }
        };
        let candidate_bitmask = || {
            let started = Instant::now();
            let hazard = divide_bitmask_fused_serial(&a_vec, &b_vec, &mut bitmask_out);
            let elapsed = started.elapsed();
            assert!(!hazard, "operands must stay hazard-free during timing");
            let checksum = divide_checksum(&bitmask_out);
            black_box(&bitmask_out);
            common::ContractObservation { elapsed, checksum }
        };
        let (bitmask_effect, _in_null2, _cand_null2) = common::run_dual_null_median_ci_contract(
            "divide_f64_1m_bitmask_classifier_vs_numpy",
            incumbent_bitmask,
            candidate_bitmask,
        );

        // THE HEAD-TO-HEAD, and it is here because the two ratios above are NOT
        // enough to size the lever. Each is internally valid — its two arms are
        // interleaved on one core in one schedule — but the two contracts are
        // different schedules, so subtracting one candidate median from the other
        // straddles whatever the host did in between. On the first run that
        // mattered: numpy itself read 383021 ns in the boolean contract and
        // 398134 ns in the bitmask contract, a 3.9% drift that flatters the
        // second candidate's ratio without either arm changing. Putting the two
        // shapes in ONE schedule against each other is the only way to say how
        // much of the ratio movement is the lever.
        //
        // This arm is a self-speedup and is labelled as one: it attributes the
        // change, it does not license a vs-incumbent claim. The vs-numpy rows
        // above remain the only competitive statement this group makes.
        let mut boolean_head = vec![0.0_f64; n];
        let mut bitmask_head = vec![0.0_f64; n];
        let boolean_arm = || {
            let started = Instant::now();
            let hazard = divide_fused_serial(&a_vec, &b_vec, &mut boolean_head);
            let elapsed = started.elapsed();
            assert!(!hazard, "operands must stay hazard-free during timing");
            let checksum = divide_checksum(&boolean_head);
            black_box(&boolean_head);
            common::ContractObservation { elapsed, checksum }
        };
        let bitmask_arm = || {
            let started = Instant::now();
            let hazard = divide_bitmask_fused_serial(&a_vec, &b_vec, &mut bitmask_head);
            let elapsed = started.elapsed();
            assert!(!hazard, "operands must stay hazard-free during timing");
            let checksum = divide_checksum(&bitmask_head);
            black_box(&bitmask_head);
            common::ContractObservation { elapsed, checksum }
        };
        let (head_to_head, _in_null3, _cand_null3) = common::run_dual_null_median_ci_contract(
            "divide_f64_1m_bitmask_over_boolean_classifier",
            boolean_arm,
            bitmask_arm,
        );

        let classifier_shape_saving_ns =
            head_to_head.arm_a_median_ns - head_to_head.arm_b_median_ns;
        println!(
            "DIVIDE_CLASSIFIER_SHAPE n={n} numpy_version={numpy_version} worker={} \
             harness=common::run_dual_null_median_ci_contract \
             arms_are_replicas_not_the_shipped_route=true \
             arms_are_preallocated_no_alloc_either_side=true \
             static_census_numpy_insns_per_16_doubles=20 \
             static_census_boolean_insns_per_16_doubles=43 \
             static_census_bitmask_insns_per_16_doubles=35 \
             boolean_ratio={:.6} boolean_ci95=[{:.6},{:.6}] \
             bitmask_ratio={:.6} bitmask_ci95=[{:.6},{:.6}] \
             numpy_ns_in_boolean_contract={:.1} numpy_ns_in_bitmask_contract={:.1} \
             boolean_ns={:.1} bitmask_ns={:.1} \
             head_to_head_is_a_self_speedup_not_a_win=true \
             head_to_head_ratio={:.6} head_to_head_ci95=[{:.6},{:.6}] \
             classifier_shape_saving_ns={classifier_shape_saving_ns:.1}",
            measurement_worker(),
            fused_effect.ratio_median,
            fused_effect.ratio_ci_low,
            fused_effect.ratio_ci_high,
            bitmask_effect.ratio_median,
            bitmask_effect.ratio_ci_low,
            bitmask_effect.ratio_ci_high,
            fused_effect.arm_a_median_ns,
            bitmask_effect.arm_a_median_ns,
            head_to_head.arm_a_median_ns,
            head_to_head.arm_b_median_ns,
            head_to_head.ratio_median,
            head_to_head.ratio_ci_low,
            head_to_head.ratio_ci_high,
        );
    });
}

// Does buffer PROVENANCE — Rust `Vec` versus numpy-allocated — change what the
// identical divide loop costs? (`deadlock-audit-ascyl`)
//
// THE OBSERVATION THAT FORCED THIS GROUP. `perf record -e dTLB-load-misses:u`
// over `bench_divide_accumulate_isolation_vs_numpy`, normalised per unit of work,
// found our replica arms taking 10.5x numpy's dTLB LOAD MISSES while taking the
// SAME L1 data-cache misses (1.033x). Identical cache-line traffic, ten times the
// address translations. Both Rust arms showed the same 10.5x, so it tracks the
// BUFFERS and not either loop body.
//
// WHY IT MATTERS MORE THAN A COUNTER CURIOSITY. The replicas allocate Rust
// `Vec<f64>`. The SHIPPED route does not: `zerocopy_f64_binary_flat` reads two
// numpy arrays and writes into a `numpy.empty` output. If the dTLB gap belongs to
// the allocator, then every replica-based divide kernel number — including
// `deadlock-audit-0ppym`'s 1.2652x and 1.5372x — is measuring our ALLOCATOR
// against numpy's and crediting it to our loop.
//
// TWO MECHANISMS ARE ALREADY REFUTED and are not re-tested here: transparent huge
// pages (numpy reports `AnonHugePages: 0 kB` with its own `_set_madvise_hugepage`
// switch ON and OFF; THP on this host is madvise-mode and nothing is collapsed),
// and buffer count (the smaller control group shows the effect at least as
// strongly as the larger one). This group tests the third and does not assume it.
//
// EXACTLY ONE VARIABLE CHANGES, and the linker proved it rather than me. Both
// arms are emitted from the same `emit_divide_fused_serial!` macro, and LLVM then
// folded them: `nm` reports `divide_fused_on_rust_vec`,
// `divide_fused_on_numpy_buffer` AND the original `divide_fused_serial` all at
// address 0x7178f0 in the built ELF. The two arms are not merely similar code,
// they are the SAME code at the SAME address, so no codegen difference can exist
// between them and the only remaining variable is which memory the pointers point
// at. The macro was originally written to get two `perf`-separable symbols; the
// fold makes that impossible, which is a fair trade for an airtight control.
//
// CONSEQUENCE FOR ATTRIBUTION: `perf` cannot split these two arms by symbol, so
// this group decides the question on the WALL CLOCK head-to-head instead. That is
// sufficient for the decision rule below — if provenance does not cost time, then
// whatever the dTLB counter says, it is not what makes us 1.25x slower.
//
// THE NEGATIVE CASE, and it is the one that would silently void the whole group:
// `PyBuffer` will hand back a COPY rather than a view if the array is not
// C-contiguous float64, and a copy would be freshly allocated by *us*, which is
// precisely the thing under test. So the group asserts the buffer pointer EQUALS
// the numpy array's own `ctypes.data`, and that both arms reproduce
// `numpy.divide` bit for bit. A copy, a stride, or a dtype surprise panics
// instead of quietly measuring Rust memory twice and reporting "no difference".
//
// DECISION RULE, registered before the run: if the numpy-buffer arm is faster by
// more than the null spread, the allocator is implicated and `0ppym`'s kernel
// numbers must be re-taken on numpy memory. If the two arms TIE, the allocator is
// EXONERATED — the replicas' Rust `Vec`s are not what makes them slow, the 10.5x
// dTLB gap is not costing time, and the third mechanism joins the two already
// refuted. Bank either way; a tie is the more useful outcome because it closes a
// door that would otherwise invalidate a whole lane of banked numbers.
fn bench_divide_allocator_provenance(_c: &mut Criterion) {
    let n = DIVIDE_SERIAL_N;
    let (a_vec, b_vec) = divide_hazard_free_operands(n);
    let mut rust_out = vec![0.0_f64; n];

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let numpy = py.import("numpy").expect("numpy oracle");
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy.__version__")
            .extract::<String>()
            .expect("numpy version is a string");

        let locals = PyDict::new(py);
        locals.set_item("np", &numpy).expect("bind numpy");
        locals.set_item("n", n).expect("bind n");
        // The SAME generator `divide_hazard_free_operands` uses, so the two
        // provenances hold bit-identical values and the checksums are comparable.
        py.run(
            std::ffi::CString::new(
                "i = np.arange(n)\na = 1.0 + (i % 1000) / 1000.0\nb = 1.25 + (i % 997) / 997.0\nout = np.empty(n)\nnpout = np.empty(n)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&locals),
            Some(&locals),
        )
        .expect("build operands");
        let a_py = locals.get_item("a").expect("a operand");
        let b_py = locals.get_item("b").expect("b operand");
        let out_py = locals.get_item("out").expect("out buffer");
        let np_out_py = locals.get_item("npout").expect("candidate out buffer");
        let args = PyTuple::new(py, [&a_py, &b_py]).expect("args");
        let out_kwargs = PyDict::new(py);
        out_kwargs.set_item("out", &out_py).expect("bind out");
        let numpy_divide = numpy.getattr("divide").expect("numpy.divide");

        // Zero-copy views of numpy's own memory, the same acquisition the shipped
        // route performs.
        let a_buffer = pyo3::buffer::PyBuffer::<f64>::get(&a_py).expect("a buffer");
        let b_buffer = pyo3::buffer::PyBuffer::<f64>::get(&b_py).expect("b buffer");
        let out_buffer = pyo3::buffer::PyBuffer::<f64>::get(&np_out_py).expect("out buffer");

        // THE NO-COPY GATE. If any of these is a copy, the "numpy-allocated" arm is
        // running on memory we allocated and the group is measuring nothing.
        for (label, buffer, object) in [
            ("a", &a_buffer, &a_py),
            ("b", &b_buffer, &b_py),
            ("npout", &out_buffer, &np_out_py),
        ] {
            let owner_ptr = object
                .getattr("ctypes")
                .and_then(|c| c.getattr("data"))
                .and_then(|d| d.extract::<usize>())
                .expect("numpy array exposes ctypes.data");
            assert_eq!(
                buffer.buf_ptr() as usize,
                owner_ptr,
                "PyBuffer handed back a COPY for `{label}`, not a view of numpy's \
                 allocation - this group would then compare Rust memory to Rust memory"
            );
            assert!(
                buffer.is_c_contiguous(),
                "`{label}` is not C-contiguous, so the loop would not be walking \
                 numpy's buffer the way the shipped route does"
            );
            assert_eq!(
                buffer.item_count(),
                n,
                "`{label}` has an unexpected element count"
            );
        }

        // `ReadOnlyCell<f64>`/`Cell<f64>` are `repr(transparent)` over `f64`, the
        // operands are read-only under the GIL, and `npout` is a distinct fresh
        // array that neither operand aliases - the same argument the shipped
        // `zerocopy_f64_binary_flat` makes for the same conversion.
        let a_np: &[f64] =
            unsafe { std::slice::from_raw_parts(a_buffer.buf_ptr().cast::<f64>().cast_const(), n) };
        let b_np: &[f64] =
            unsafe { std::slice::from_raw_parts(b_buffer.buf_ptr().cast::<f64>().cast_const(), n) };
        let numpy_out: &mut [f64] =
            unsafe { std::slice::from_raw_parts_mut(out_buffer.buf_ptr().cast::<f64>(), n) };

        // The two provenances must hold identical values, or the arms are dividing
        // different numbers and any ratio between them is meaningless.
        assert!(
            a_np.iter()
                .zip(a_vec.iter())
                .all(|(l, r)| l.to_bits() == r.to_bits())
                && b_np
                    .iter()
                    .zip(b_vec.iter())
                    .all(|(l, r)| l.to_bits() == r.to_bits()),
            "the numpy operands and the Rust operands are not bit-identical"
        );

        // Parity gate against the incumbent, on both provenances.
        let numpy_result = numpy_divide
            .call(&args, Some(&out_kwargs))
            .expect("numpy.divide probe");
        let numpy_sum = numpy_divide_checksum(&numpy_result, n);
        assert!(
            !divide_fused_on_rust_vec(&a_vec, &b_vec, &mut rust_out),
            "operands must be hazard-free"
        );
        assert_eq!(
            divide_checksum(&rust_out),
            numpy_sum,
            "the Vec-backed arm does not reproduce numpy.divide bit for bit"
        );
        assert!(
            !divide_fused_on_numpy_buffer(a_np, b_np, numpy_out),
            "operands must be hazard-free"
        );
        assert_eq!(
            divide_checksum(numpy_out),
            numpy_sum,
            "the numpy-buffer arm does not reproduce numpy.divide bit for bit"
        );

        let rust_arm = || {
            let started = Instant::now();
            let hazard = divide_fused_on_rust_vec(&a_vec, &b_vec, &mut rust_out);
            let elapsed = started.elapsed();
            assert!(!hazard, "operands must stay hazard-free during timing");
            let checksum = divide_checksum(&rust_out);
            black_box(&rust_out);
            common::ContractObservation { elapsed, checksum }
        };
        let numpy_buffer_arm = || {
            let started = Instant::now();
            let hazard = divide_fused_on_numpy_buffer(a_np, b_np, numpy_out);
            let elapsed = started.elapsed();
            assert!(!hazard, "operands must stay hazard-free during timing");
            let checksum = divide_checksum(numpy_out);
            black_box(&numpy_out);
            common::ContractObservation { elapsed, checksum }
        };
        let (provenance, _in_null, _cand_null) = common::run_dual_null_median_ci_contract(
            "divide_f64_1m_numpy_buffer_over_rust_vec",
            rust_arm,
            numpy_buffer_arm,
        );

        println!(
            "DIVIDE_ALLOCATOR_PROVENANCE n={n} numpy_version={numpy_version} worker={} \
             harness=common::run_dual_null_median_ci_contract \
             loop_body_emitted_from_one_macro=true \
             arms_folded_by_llvm_to_one_code_address=true \
             buffers_are_views_not_copies=true \
             both_arms_match_numpy_bitwise=true \
             this_is_a_self_speedup_not_a_win=true \
             ratio_numpy_buffer_over_rust_vec={:.6} ci95=[{:.6},{:.6}] \
             rust_vec_ns={:.1} numpy_buffer_ns={:.1}",
            measurement_worker(),
            provenance.ratio_median,
            provenance.ratio_ci_low,
            provenance.ratio_ci_high,
            provenance.arm_a_median_ns,
            provenance.arm_b_median_ns,
        );
    });
}

/// `deadlock-audit-0ppym`'s two banked kernel ratios, kept as constants so the
/// re-take below prints what it supersedes on the same line rather than asking a
/// reader to go and find them. They were measured with Rust `Vec` buffers.
const OPPYM_VEC_ACCUMULATE_FREE_RATIO: f64 = 0.790499;
const OPPYM_VEC_FUSED_RATIO: f64 = 0.764919;

// RE-TAKING `deadlock-audit-0ppym`'s divide kernel rows on the memory the SHIPPED
// route actually uses (`deadlock-audit-ascyl`).
//
// WHY THESE ROWS NEED RE-TAKING RATHER THAN ADJUSTING. `bench_divide_allocator_
// provenance` certified that the identical divide loop runs a MEDIAN 1.072x faster
// (range 1.011-1.157 over five runs) on numpy-allocated buffers than on Rust
// `Vec`s, with a counted mechanism: 5.96x the dTLB load misses at identical L1D
// misses and identical instruction counts, paid at 55 cycles per excess miss.
// `0ppym` timed Rust-`Vec` replicas against an `out=`-fed `numpy.divide`, so its
// 1.2652x (accumulate-free) and 1.5372x (fused) each carry a provenance tax that
// `zerocopy_f64_binary_flat` never pays — it reads two numpy arrays and writes a
// `numpy.empty` output. Dividing the old ratios by 1.072 would be arithmetic on
// two numbers from different windows; this group MEASURES the corrected pair
// instead, both arms against the same incumbent in the same invocation.
//
// WHAT IS HELD FIXED. The loop bodies are the byte-identical
// `divide_former_serial` and `divide_fused_serial` that `0ppym` timed — not
// re-implementations — so the only difference from that row is where the bytes
// live. Neither side allocates during timing: NumPy is handed a preallocated
// `out=` and both replicas write into preallocated NUMPY arrays.
//
// THE NEGATIVE CASE, and it is the one that would quietly void the whole group:
// `PyBuffer` returns a COPY for a non-contiguous or wrong-dtype array, and a copy
// is memory WE allocate — which is exactly the provenance being corrected for, so
// the group would silently re-measure `0ppym` and "confirm" it. Every buffer is
// therefore asserted to be a view (`buf_ptr()` equals the array's own
// `ctypes.data`), C-contiguous, and of the expected length, and both arms are
// asserted to reproduce `numpy.divide` bit for bit before any timing.
//
// THIS IS STILL A REPLICA COMPARISON, not the shipped route: it prices the KERNEL
// under the shipped memory configuration. It does not license a claim about
// `fnp.divide` as called from Python, which carries the wrapper floor on top.
fn bench_divide_kernel_on_numpy_buffers(_c: &mut Criterion) {
    assert_divide_hazard_replica_matches_contract();
    let n = DIVIDE_SERIAL_N;
    let (a_vec, b_vec) = divide_hazard_free_operands(n);

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let numpy = py.import("numpy").expect("numpy oracle");
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy.__version__")
            .extract::<String>()
            .expect("numpy version is a string");

        let locals = PyDict::new(py);
        locals.set_item("np", &numpy).expect("bind numpy");
        locals.set_item("n", n).expect("bind n");
        // The same generator `divide_hazard_free_operands` uses, so the operands
        // are bit-identical to the ones `0ppym` divided.
        py.run(
            std::ffi::CString::new(
                "i = np.arange(n)\na = 1.0 + (i % 1000) / 1000.0\nb = 1.25 + (i % 997) / 997.0\nout = np.empty(n)\nformer_out = np.empty(n)\nfused_out = np.empty(n)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&locals),
            Some(&locals),
        )
        .expect("build operands");
        let a_py = locals.get_item("a").expect("a operand");
        let b_py = locals.get_item("b").expect("b operand");
        let out_py = locals.get_item("out").expect("incumbent out buffer");
        let former_py = locals.get_item("former_out").expect("former out buffer");
        let fused_py = locals.get_item("fused_out").expect("fused out buffer");
        let args = PyTuple::new(py, [&a_py, &b_py]).expect("args");
        let out_kwargs = PyDict::new(py);
        out_kwargs.set_item("out", &out_py).expect("bind out");
        let numpy_divide = numpy.getattr("divide").expect("numpy.divide");

        let a_buffer = pyo3::buffer::PyBuffer::<f64>::get(&a_py).expect("a buffer");
        let b_buffer = pyo3::buffer::PyBuffer::<f64>::get(&b_py).expect("b buffer");
        let former_buffer =
            pyo3::buffer::PyBuffer::<f64>::get(&former_py).expect("former out buffer");
        let fused_buffer = pyo3::buffer::PyBuffer::<f64>::get(&fused_py).expect("fused out buffer");

        for (label, buffer, object) in [
            ("a", &a_buffer, &a_py),
            ("b", &b_buffer, &b_py),
            ("former_out", &former_buffer, &former_py),
            ("fused_out", &fused_buffer, &fused_py),
        ] {
            let owner_ptr = object
                .getattr("ctypes")
                .and_then(|c| c.getattr("data"))
                .and_then(|d| d.extract::<usize>())
                .expect("numpy array exposes ctypes.data");
            assert_eq!(
                buffer.buf_ptr() as usize,
                owner_ptr,
                "PyBuffer handed back a COPY for `{label}` - the arms would then run \
                 on memory we allocated, which is the very provenance this group \
                 exists to correct for"
            );
            assert!(
                buffer.is_c_contiguous(),
                "`{label}` is not C-contiguous, so the loop would not walk numpy's \
                 buffer the way the shipped route does"
            );
            assert_eq!(buffer.item_count(), n, "`{label}` has the wrong length");
        }

        // Same `repr(transparent)` argument the shipped `zerocopy_f64_binary_flat`
        // makes: operands are read-only under the GIL and the two outputs are
        // distinct fresh arrays that neither operand aliases.
        let a_np: &[f64] =
            unsafe { std::slice::from_raw_parts(a_buffer.buf_ptr().cast::<f64>().cast_const(), n) };
        let b_np: &[f64] =
            unsafe { std::slice::from_raw_parts(b_buffer.buf_ptr().cast::<f64>().cast_const(), n) };
        let former_out: &mut [f64] =
            unsafe { std::slice::from_raw_parts_mut(former_buffer.buf_ptr().cast::<f64>(), n) };
        let fused_out: &mut [f64] =
            unsafe { std::slice::from_raw_parts_mut(fused_buffer.buf_ptr().cast::<f64>(), n) };

        assert!(
            a_np.iter()
                .zip(a_vec.iter())
                .all(|(l, r)| l.to_bits() == r.to_bits())
                && b_np
                    .iter()
                    .zip(b_vec.iter())
                    .all(|(l, r)| l.to_bits() == r.to_bits()),
            "the numpy operands are not bit-identical to the ones `0ppym` divided"
        );

        let numpy_result = numpy_divide
            .call(&args, Some(&out_kwargs))
            .expect("numpy.divide probe");
        let numpy_sum = numpy_divide_checksum(&numpy_result, n);
        divide_former_serial(a_np, b_np, former_out);
        assert_eq!(
            divide_checksum(former_out),
            numpy_sum,
            "the accumulate-free arm does not reproduce numpy.divide bit for bit"
        );
        assert!(
            !divide_fused_serial(a_np, b_np, fused_out),
            "these operands must be hazard-free or the fused arm takes its rare \
             second pass and the comparison measures a different branch"
        );
        assert_eq!(
            divide_checksum(fused_out),
            numpy_sum,
            "the fused arm does not reproduce numpy.divide bit for bit"
        );

        let incumbent_former = || {
            let started = Instant::now();
            let result = numpy_divide
                .call(&args, Some(&out_kwargs))
                .expect("numpy.divide call");
            let elapsed = started.elapsed();
            let checksum = numpy_divide_checksum(&result, n);
            common::ContractObservation { elapsed, checksum }
        };
        let candidate_former = || {
            let started = Instant::now();
            divide_former_serial(a_np, b_np, former_out);
            let elapsed = started.elapsed();
            let checksum = divide_checksum(former_out);
            black_box(&former_out);
            common::ContractObservation { elapsed, checksum }
        };
        let (former_effect, _in_null, _cand_null) = common::run_dual_null_median_ci_contract(
            "divide_f64_1m_accumulate_free_on_numpy_buffers_vs_numpy",
            incumbent_former,
            candidate_former,
        );

        let incumbent_fused = || {
            let started = Instant::now();
            let result = numpy_divide
                .call(&args, Some(&out_kwargs))
                .expect("numpy.divide call");
            let elapsed = started.elapsed();
            let checksum = numpy_divide_checksum(&result, n);
            common::ContractObservation { elapsed, checksum }
        };
        let candidate_fused = || {
            let started = Instant::now();
            let hazard = divide_fused_serial(a_np, b_np, fused_out);
            let elapsed = started.elapsed();
            assert!(!hazard, "operands must stay hazard-free during timing");
            let checksum = divide_checksum(fused_out);
            black_box(&fused_out);
            common::ContractObservation { elapsed, checksum }
        };
        let (fused_effect, _in_null2, _cand_null2) = common::run_dual_null_median_ci_contract(
            "divide_f64_1m_fused_on_numpy_buffers_vs_numpy",
            incumbent_fused,
            candidate_fused,
        );

        println!(
            "DIVIDE_KERNEL_ON_NUMPY_BUFFERS n={n} numpy_version={numpy_version} worker={} \
             harness=common::run_dual_null_median_ci_contract \
             arms_are_replicas_not_the_shipped_route=true \
             arms_are_preallocated_no_alloc_either_side=true \
             all_buffers_are_numpy_allocated_views_not_copies=true \
             supersedes=deadlock-audit-0ppym_vec_backed_rows \
             accumulate_free_ratio={:.6} accumulate_free_ci95=[{:.6},{:.6}] \
             fused_ratio={:.6} fused_ci95=[{:.6},{:.6}] \
             numpy_ns={:.1} accumulate_free_ns={:.1} fused_ns={:.1} \
             vec_backed_accumulate_free_ratio_for_reference={OPPYM_VEC_ACCUMULATE_FREE_RATIO:.6} \
             vec_backed_fused_ratio_for_reference={OPPYM_VEC_FUSED_RATIO:.6}",
            measurement_worker(),
            former_effect.ratio_median,
            former_effect.ratio_ci_low,
            former_effect.ratio_ci_high,
            fused_effect.ratio_median,
            fused_effect.ratio_ci_low,
            fused_effect.ratio_ci_high,
            former_effect.arm_a_median_ns,
            former_effect.arm_b_median_ns,
            fused_effect.arm_b_median_ns,
        );
    });
}

/// How many calls each counter probe makes. Fixed, and identical across the two
/// probes, so `perf stat` totals from two separate processes are comparable
/// without any normalisation (`deadlock-audit-ascyl`).
const PROVENANCE_COUNTER_CALLS: usize = 400;

// COUNTING the provenance effect instead of just timing it (`deadlock-audit-ascyl`).
//
// `bench_divide_allocator_provenance` shows the numpy-allocated arm beating the
// Vec-backed arm on the wall clock, but it cannot say WHY: LLVM folds the two arms
// to one code address, so `perf` cannot attribute a counter to one arm or the
// other inside a single process. The suspected mechanism is the 10.5x dTLB
// load-miss gap, and a suspicion is not a mechanism.
//
// These two probes make it countable. Each runs the SAME loop the SAME number of
// times over the SAME values, differing only in whether the buffers came from
// Rust's allocator or numpy's, and each is meant to be run in its OWN process
// under `perf stat`. Process setup, interpreter start and operand construction are
// identical between them, so the DIFFERENCE in counted events is the provenance
// effect and nothing else.
//
// Run them as:
//   perf stat -e dTLB-load-misses:u,L1-dcache-load-misses:u,cycles:u -- \
//     <elf> --bench fnp-group=bench_divide_provenance_counter_rust
//   ... same with ..._numpy
//
// NEGATIVE CASE: if the two probes report the same dTLB misses, the 10.5x gap seen
// across symbols was NOT a property of the buffers and the wall-clock difference
// has some other cause — which must then be found before anyone credits the
// allocator. A probe that silently made a copy would also show no difference, so
// both probes re-assert the no-copy and bit-parity gates rather than trusting the
// sibling group to have checked them.
fn bench_divide_provenance_counter_rust(_c: &mut Criterion) {
    let n = DIVIDE_SERIAL_N;
    let (a_vec, b_vec) = divide_hazard_free_operands(n);
    let mut out = vec![0.0_f64; n];

    // THE SETUP IS PART OF THE CONTROL, and the first version of this probe got it
    // wrong. `perf stat` counts a whole PROCESS, so the two probes are only
    // comparable if they do the same work outside the loop. The first version
    // skipped Python entirely while its sibling started the interpreter, imported
    // numpy and built the operands — worth 426 M instructions and 228 M cycles of
    // pure difference, which swamped the effect being measured. So this probe pays
    // the identical setup and then simply does not divide into it.
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let numpy = py.import("numpy").expect("numpy oracle");
        let locals = PyDict::new(py);
        locals.set_item("np", &numpy).expect("bind numpy");
        locals.set_item("n", n).expect("bind n");
        py.run(
            std::ffi::CString::new(
                "i = np.arange(n)\na = 1.0 + (i % 1000) / 1000.0\nb = 1.25 + (i % 997) / 997.0\nnpout = np.empty(n)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&locals),
            Some(&locals),
        )
        .expect("build operands");
        let a_py = locals.get_item("a").expect("a operand");
        let b_py = locals.get_item("b").expect("b operand");
        let np_out_py = locals.get_item("npout").expect("out buffer");
        // Acquire the buffers too, so even the PyBuffer cost matches. They are then
        // deliberately unused: this arm divides out of Rust `Vec`s.
        let a_buffer = pyo3::buffer::PyBuffer::<f64>::get(&a_py).expect("a buffer");
        let b_buffer = pyo3::buffer::PyBuffer::<f64>::get(&b_py).expect("b buffer");
        let out_buffer = pyo3::buffer::PyBuffer::<f64>::get(&np_out_py).expect("out buffer");
        black_box((
            a_buffer.item_count(),
            b_buffer.item_count(),
            out_buffer.item_count(),
        ));

        let mut hazards = 0usize;
        for _ in 0..PROVENANCE_COUNTER_CALLS {
            if divide_fused_on_rust_vec(&a_vec, &b_vec, &mut out) {
                hazards += 1;
            }
            black_box(&out);
        }
        assert_eq!(hazards, 0, "operands must be hazard-free");
        println!(
            "PROVENANCE_COUNTER_PROBE provenance=rust_vec n={n} \
             calls={PROVENANCE_COUNTER_CALLS} checksum={:016x} \
             setup_matches_sibling_probe=true run_this_under_perf_stat=true",
            divide_checksum(&out)
        );
    });
}

fn bench_divide_provenance_counter_numpy(_c: &mut Criterion) {
    let n = DIVIDE_SERIAL_N;
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let numpy = py.import("numpy").expect("numpy oracle");
        let locals = PyDict::new(py);
        locals.set_item("np", &numpy).expect("bind numpy");
        locals.set_item("n", n).expect("bind n");
        py.run(
            std::ffi::CString::new(
                "i = np.arange(n)\na = 1.0 + (i % 1000) / 1000.0\nb = 1.25 + (i % 997) / 997.0\nnpout = np.empty(n)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&locals),
            Some(&locals),
        )
        .expect("build operands");
        let a_py = locals.get_item("a").expect("a operand");
        let b_py = locals.get_item("b").expect("b operand");
        let np_out_py = locals.get_item("npout").expect("out buffer");

        let a_buffer = pyo3::buffer::PyBuffer::<f64>::get(&a_py).expect("a buffer");
        let b_buffer = pyo3::buffer::PyBuffer::<f64>::get(&b_py).expect("b buffer");
        let out_buffer = pyo3::buffer::PyBuffer::<f64>::get(&np_out_py).expect("out buffer");
        // Same no-copy gate as the timed group: a copy would be OUR memory and the
        // probe would report "no difference" for the wrong reason.
        for (label, buffer, object) in [
            ("a", &a_buffer, &a_py),
            ("b", &b_buffer, &b_py),
            ("npout", &out_buffer, &np_out_py),
        ] {
            let owner_ptr = object
                .getattr("ctypes")
                .and_then(|c| c.getattr("data"))
                .and_then(|d| d.extract::<usize>())
                .expect("numpy array exposes ctypes.data");
            assert_eq!(
                buffer.buf_ptr() as usize,
                owner_ptr,
                "PyBuffer handed back a COPY for `{label}`, so this probe would be \
                 counting Rust memory while claiming to count numpy's"
            );
            assert!(buffer.is_c_contiguous(), "`{label}` is not C-contiguous");
        }
        let a_np: &[f64] =
            unsafe { std::slice::from_raw_parts(a_buffer.buf_ptr().cast::<f64>().cast_const(), n) };
        let b_np: &[f64] =
            unsafe { std::slice::from_raw_parts(b_buffer.buf_ptr().cast::<f64>().cast_const(), n) };
        let out: &mut [f64] =
            unsafe { std::slice::from_raw_parts_mut(out_buffer.buf_ptr().cast::<f64>(), n) };

        let mut hazards = 0usize;
        for _ in 0..PROVENANCE_COUNTER_CALLS {
            if divide_fused_on_numpy_buffer(a_np, b_np, out) {
                hazards += 1;
            }
            black_box(&out);
        }
        assert_eq!(hazards, 0, "operands must be hazard-free");
        println!(
            "PROVENANCE_COUNTER_PROBE provenance=numpy_allocated n={n} \
             calls={PROVENANCE_COUNTER_CALLS} checksum={:016x} \
             run_this_under_perf_stat=true",
            divide_checksum(out)
        );
    });
}

// How much is there to win by fetching `dtype` ONCE and threading it through the
// probe chain, instead of every probe fetching it again? (`deadlock-audit-v46rn`)
//
// WHY THIS IS THE NEXT LEVER. The multiply route sweep measured `excess_ns` roughly
// CONSTANT at ~2.0-9.8 us from n=2^8 to n=2^24 — a fixed per-call floor. At n=256
// that floor is 2044 ns against a NumPy call that takes 410 ns in total, i.e. we
// spend 5x NumPy's entire runtime before delegating to it. Separately, the stage
// attribution priced ONE `dtype` fetch of both operands at 140 ns, and a static read
// of `PyUFunc::__call__` (lines 317-582) finds ELEVEN predicate/probe call sites on
// the delegating path — `numpy_dtype_is_f64` x2, `numpy_dtype_is_f32` x2,
// `try_zerocopy_f16_binary_widen` x2, plus the f64/f32/complex/floor_divide/f16_compare
// probes — each of which fetches `dtype` for itself. Not all run for every op, but
// nothing shares the fetch.
//
// THIS IS A CEILING, exactly like the lookup-hoisting control: the hoisted arm holds
// the dtype in a local, which is the best any threading-through could achieve and
// which a real refactor must additionally pay for in plumbing. Measure the prize
// before touching `PyUFunc::__call__`.
//
// THE NEGATIVE CASE, and it is the whole reason this group can be trusted: CPython
// may make a repeated `getattr("dtype")` nearly free — numpy caches the descriptor,
// and attribute lookup is a dict hit. If so the "prize" collapses to ~0 for a reason
// that has nothing to do with our route, and a naive implementation would report a
// tiny saving and conclude the lever is dead. So this group measures the repeated arm
// at THREE fanouts (1, 2, and PROBE_FANOUT) and asserts the cost actually SCALES with
// the number of fetches. If it does not scale, the fetch is being cached or elided and
// the row says the measurement is void rather than reporting a small number as if it
// were a finding.
fn bench_dtype_probe_fanout_ceiling(_c: &mut Criterion) {
    // The count of dtype-fetching predicate/probe sites on the delegating path in
    // `PyUFunc::__call__`. Read from source, not guessed; see the header comment.
    const PROBE_FANOUT: usize = 6;
    const N: usize = 256;
    const TRIALS: usize = 2001;

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

        // One fetch of BOTH operands, repeated `fanout` times — what the probe chain
        // does today, each probe fetching for itself.
        let repeated = |fanout: usize| -> f64 {
            min_ns(TRIALS, || {
                for _ in 0..fanout {
                    black_box(a.getattr("dtype").expect("a dtype"));
                    black_box(b.getattr("dtype").expect("b dtype"));
                }
            })
        };
        // Fetched once, then reused `fanout` times — the ceiling on any threading.
        let hoisted = |fanout: usize| -> f64 {
            min_ns(TRIALS, || {
                let da = a.getattr("dtype").expect("a dtype");
                let db = b.getattr("dtype").expect("b dtype");
                for _ in 0..fanout {
                    black_box(&da);
                    black_box(&db);
                }
            })
        };

        let repeated_1 = repeated(1);
        let repeated_2 = repeated(2);
        let repeated_n = repeated(PROBE_FANOUT);
        let hoisted_n = hoisted(PROBE_FANOUT);

        // PARITY: hoisting must not change the answer. The dtype read once must equal
        // the dtype read per-probe, or the "optimisation" is a different computation.
        let once = a.getattr("dtype").expect("a dtype");
        let again = a.getattr("dtype").expect("a dtype");
        assert!(
            once.eq(&again).expect("dtype equality"),
            "a hoisted dtype must equal a per-probe dtype, or hoisting changes behaviour"
        );

        // THE NEGATIVE CASE. If repeated fetches are cached or elided, cost does not
        // grow with fanout and any "saving" this group reports is an artefact. Require
        // the 6-fetch arm to cost meaningfully more than the 1-fetch arm — at least
        // 2x, which is far below the 6x that genuinely-uncached fetches would give and
        // so tolerates measurement noise without tolerating a cached lookup.
        let scaling = repeated_n / repeated_1.max(1.0);
        assert!(
            scaling >= 2.0,
            "repeated dtype fetches did NOT scale with fanout ({repeated_1:.1} ns at 1 \
             vs {repeated_n:.1} ns at {PROBE_FANOUT}, ratio {scaling:.2}): the fetch is \
             being cached or optimised away, so this group cannot measure the probe \
             chain's dtype cost and its numbers must not be banked"
        );

        let saved_ns = repeated_n - hoisted_n;
        let per_fetch_ns = (repeated_2 - repeated_1).max(0.0);
        println!(
            "DTYPE_PROBE_FANOUT_CEILING n={N} numpy_version={numpy_version} worker={} \
             harness=replica_min_of_{TRIALS} trials={TRIALS} probe_fanout={PROBE_FANOUT} \
             fanout_from_static_read_of_pyufunc_call=true \
             repeated_1_ns={repeated_1:.1} repeated_2_ns={repeated_2:.1} \
             repeated_n_ns={repeated_n:.1} hoisted_n_ns={hoisted_n:.1} \
             per_fetch_pair_ns={per_fetch_ns:.1} saved_ns={saved_ns:.1} \
             scaling_ratio={scaling:.3} \
             route_floor_reference_excess_ns_at_n256=2044.0 \
             numpy_whole_call_reference_ns_at_n256=410.0",
            measurement_worker(),
        );
    });
}

// The SIZE AXIS for the campaign's actual worst vs-incumbent cell (`deadlock-audit-ei9jz`).
//
// `bench_percall_floor_across_ops_vs_numpy` established, over TWO invocations with all
// eight A/A nulls clean, that `add` — not `multiply` — is the worst cell: 3.633x then
// 3.808x slower than NumPy at n=256, with the ordering identical both times. But that
// group measures ONE size, so everything known about the worst op is a single point.
//
// `bench_route_floor_size_sweep_vs_numpy` has the size axis but hardcodes `multiply`,
// where the answer is already known: excess_ns is roughly constant from 2^8 to 2^20 and
// parity arrives at 2^24. Whether `add` behaves the same way is UNMEASURED, and it does
// not follow from multiply's shape: the two sit in different cost clusters — {add,
// subtract} carry 1076-1372 ns of excess against {multiply, divide} at 877-947 ns, the one
// part of the op-spread finding that replicated across both runs.
//
// So this asks the one question the worst cell has never been asked: does `add` reach
// parity at all, and if so where? A per-call floor that amortises by 2^24 is a small-n
// problem; a floor that does not is a different and worse thing.
//
// Deliberately a SEPARATE group rather than a parameter on the multiply sweep: that group
// is cited by banked rows, and changing its shape would silently change what those rows
// refer to.
// How bad does the campaign's worst ratio get as n -> 1? (`deadlock-audit-ei9jz`)
//
// EVERY size sweep in this ledger bottoms out at n=256, which is the SMALLEST n ever
// measured for any binary op. That is a gap, not a floor: the per-call cost is FIXED
// (excess_ns sits at ~1100-1700 ns from 2^8 to 2^16 while the work grows 256x), so the
// ratio must keep worsening below 256 and approach an asymptote of
// numpy_per_call / fnp_per_call. The campaign quotes "~3.5-3.8x slower at n=256" as its
// worst number; that is the worst MEASURED number, not the worst number.
//
// This measures 1, 4, 16, 64 and anchors on 256. The anchor is the point: 256 is measured
// in the SAME invocation as the new cells, so the comparison is within-invocation rather
// than against a remembered figure from another run - the cross-run subtraction that has
// already produced three retracted numbers today.
//
// WHAT WOULD MAKE THIS ROW MEANINGLESS, stated so the reader can check it: if the ratio
// FLATTENS below 256 the asymptote is already reached and n=256 was a fair worst case. If
// it keeps falling, every "worst ratio" figure in this ledger understates the small-array
// case. Either answer is worth having; the group does not presuppose one.
// The paired `out=` arm every recent retry predicate asks for, and which did not exist
// (`deadlock-audit-ei9jz`).
//
// Three rows now end with "certify with a paired out= arm" and none could be run, because
// no group exercises `out=` at all. This is that group.
//
// WHY IT MATTERS MORE THAN IT LOOKS. The corrected maximum arms measured NumPy's own cost
// collapsing when it stops allocating - 6441077 ns allocating against 3814213 ns with
// `out=` - while ours fell further, 7075751 to 2250726. So the SAME op reads 0.907848
// (a loss) when both sides allocate and 1.702079 (a win) when neither does. `out=` is not
// a small saving on this route; it is the parameter that decides the sign of the result.
// That prediction is exactly what this group exists to test.
//
// BOTH ARMS ARE SYMMETRIC BY CONSTRUCTION, which is the defect `deadlock-audit-48by6`
// caught in the old maximum arms: each side gets its OWN preallocated `out` of the same
// dtype and shape, so neither is handed a buffer the other has to build. Separate buffers
// rather than one shared, so no arm can be accused of warming the other's cache line.
//
// CELLS: `divide` at 2^20 (native serial regime, where the per-call floor dominates) and
// `maximum` at 2^22 (above the parallel threshold, where the 1.702079 kernel win should
// appear if `out=` really is the deciding parameter).
fn bench_out_kwarg_vs_numpy(_c: &mut Criterion) {
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

        // All THREE of the ~0.91 memory-bound cells, plus divide's serial regime.
        //
        // The certified `maximum` cell flipped 0.907848 (loss, both sides allocating) to
        // 1.501804 (WIN, neither allocating), which means `out=` decides the SIGN there.
        // `minimum` 0.913424 and `divide` parallel 0.920786 are the other two cells of
        // that class and have NOT been tested this way. Restating them as wins on
        // `maximum` alone would be the cross-cell transfer `deadlock-audit-48by6`
        // withdrew one row after making, so they get their own cells here.
        //
        // `divide` appears at BOTH 2^20 and 2^22 deliberately: 2^20 is its serial regime,
        // where `out=` did NOT flip it (0.824614), and 2^22 is the parallel regime where
        // the ~0.91 figure was measured. If divide behaves differently across its own
        // parallel_min, that is a property of the gate rather than of `out=`.
        for (op_name, exponent) in [
            ("divide", 20u32),
            ("maximum", 22u32),
            ("minimum", 22u32),
            ("divide", 22u32),
        ] {
            let n = 1usize << exponent;
            let locals = PyDict::new(py);
            locals.set_item("np", &numpy).expect("bind numpy");
            locals.set_item("n", n).expect("bind n");
            py.run(
                std::ffi::CString::new(
                    "i = np.arange(n)\na = 1.0 + (i % 1000) / 1000.0\nb = 1.25 + (i % 997) / 997.0\nout_np = np.empty(n)\nout_fnp = np.empty(n)\n",
                )
                .unwrap()
                .as_c_str(),
                Some(&locals),
                Some(&locals),
            )
            .expect("build operands");
            let a = locals.get_item("a").expect("a operand");
            let b = locals.get_item("b").expect("b operand");
            let out_np = locals.get_item("out_np").expect("numpy out buffer");
            let out_fnp = locals.get_item("out_fnp").expect("fnp out buffer");
            let args = PyTuple::new(py, [&a, &b]).expect("args");

            let ours = module.getattr(op_name).expect("fnp op");
            let theirs = numpy.getattr(op_name).expect("numpy op");
            assert!(
                !ours.is(&theirs),
                "fnp.{op_name} IS numpy's object - there is no candidate arm"
            );
            let np_kwargs = PyDict::new(py);
            np_kwargs.set_item("out", &out_np).expect("bind numpy out");
            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("out", &out_fnp).expect("bind fnp out");

            // Parity before timing: the two arms must agree, and each must have actually
            // written its own buffer rather than returning a fresh array.
            let numpy_probe = theirs
                .call(&args, Some(&np_kwargs))
                .expect("numpy out= probe");
            let fnp_probe = ours.call(&args, Some(&fnp_kwargs)).expect("fnp out= probe");
            assert!(
                numpy_probe.is(&out_np),
                "numpy.{op_name} must return the out buffer it was given"
            );
            assert!(
                fnp_probe.is(&out_fnp),
                "fnp.{op_name} must return the out buffer it was given, as numpy does"
            );
            assert_eq!(
                numpy_divide_checksum(&numpy_probe, n),
                numpy_divide_checksum(&fnp_probe, n),
                "fnp.{op_name} and numpy.{op_name} disagree under out="
            );

            let incumbent = || {
                let started = Instant::now();
                let result = theirs
                    .call(&args, Some(&np_kwargs))
                    .expect("numpy out= call");
                let elapsed = started.elapsed();
                let checksum = numpy_divide_checksum(&result, n);
                common::ContractObservation { elapsed, checksum }
            };
            let candidate = || {
                let started = Instant::now();
                let result = ours.call(&args, Some(&fnp_kwargs)).expect("fnp out= call");
                let elapsed = started.elapsed();
                let checksum = numpy_divide_checksum(&result, n);
                common::ContractObservation { elapsed, checksum }
            };
            let (effect, _incumbent_null, _candidate_null) =
                common::run_dual_null_median_ci_contract(
                    &format!("{op_name}_f64_n{n}_out_kwarg_vs_numpy"),
                    incumbent,
                    candidate,
                );
            println!(
                "OUT_KWARG_VS_NUMPY op={op_name} n={n} log2n={exponent} \
                 numpy_version={numpy_version} worker={} \
                 harness=common::run_dual_null_median_ci_contract \
                 both_arms_preallocate_their_own_out=true \
                 ratio={:.6} ratio_ci95=[{:.6},{:.6}] numpy_ns={:.1} fnp_ns={:.1} \
                 faster_than_numpy={}",
                measurement_worker(),
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                effect.arm_a_median_ns,
                effect.arm_b_median_ns,
                effect.ratio_ci_low > 1.0,
            );
        }
    });
}

fn bench_add_tiny_n_floor_vs_numpy(_c: &mut Criterion) {
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

        for n in [1usize, 4, 16, 64, 256] {
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
                measure_binary_ufunc_vs_numpy(py, &module, &numpy, "add", &args, n);
            let excess_ns = fnp_ns - numpy_ns;
            println!(
                "TINY_N_FLOOR op=add n={n} numpy_version={numpy_version} \
                 harness=common::run_dual_null_median_ci_contract \
                 ratio={ratio:.6} ratio_ci95=[{lo:.6},{hi:.6}] \
                 numpy_ns={numpy_ns:.1} fnp_ns={fnp_ns:.1} excess_ns={excess_ns:.1} \
                 is_anchor_cell={} anchored_in_same_invocation=true",
                n == 256,
            );
        }
    });
}

fn bench_add_route_floor_size_sweep_vs_numpy(_c: &mut Criterion) {
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

        // Same ladder as the multiply sweep, so the two are directly comparable cell for
        // cell. Same operands too — a and b are built by the identical generator.
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
                measure_binary_ufunc_vs_numpy(py, &module, &numpy, "add", &args, n);
            let excess_ns = fnp_ns - numpy_ns;
            println!(
                "ROUTE_FLOOR_SWEEP op=add n={n} log2n={exponent} \
                 numpy_version={numpy_version} \
                 harness=common::run_dual_null_median_ci_contract \
                 ratio={ratio:.6} ratio_ci95=[{lo:.6},{hi:.6}] \
                 numpy_ns={numpy_ns:.1} fnp_ns={fnp_ns:.1} excess_ns={excess_ns:.1} \
                 excess_ns_per_element={:.6} at_parity={} \
                 worst_cell_of_the_campaign=true",
                excess_ns / n as f64,
                hi >= 1.0,
            );
        }
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
/// Isolate the cost of the DELEGATION CALL SHAPE, which is the lever in
/// `deadlock-audit-s2fkk`, rather than the route that contains it.
///
/// `PyUFunc::__call__`'s delegation tail used to allocate a `PyDict` and set
/// `casting`, `order` and `subok` on EVERY call, even when all three already held
/// NumPy's own defaults (`same_kind`, `K`, `True`). Both arms below call NumPy's
/// own `multiply` on identical operands; they differ ONLY in whether those three
/// default-valued keywords are passed. Arm A is the old shape, arm B the new one.
///
/// This is the cleanest available control for this lever: one binary, one
/// invocation, one worker, interleaved ABBAABBA with an A/A null, and NO fnp code
/// in either arm. It cannot be confounded by worker heterogeneity, by build
/// profile, or by a peer's uncommitted change to the route — the three things that
/// make a before/after across two builds unsound here. It measures exactly what
/// was removed, and deliberately NOT the whole route, which is reported separately
/// and must not be attributed to this lever.
/// Two `#[pyfunction]`s with IDENTICAL bodies that differ only in their PyO3
/// binding signature, so the delta between them is the binding cost and nothing
/// else (`deadlock-audit-7ocfa`).
///
/// `PyUFunc::__call__` declares nine parameters, six optional with defaults, and
/// PyO3 binds all of them on every call before any element is touched. `probe_nine`
/// reproduces that exact signature; `probe_varargs` takes `(*args, **kwargs)` and
/// parses nothing. Both immediately return their first argument, so nothing but
/// argument handling is inside the timer.
///
/// Written as a control BEFORE changing the real signature: if binding turns out to
/// be cheap, the change should not be made at all, and that is a result worth
/// banking. Same-binary and same-invocation for the same reasons as
/// `bench_delegation_kwargs_shape` - the fleet is heterogeneous, cannot be pinned,
/// and the shared tree carries peers' uncommitted work, so a two-build before/after
/// would not be attributable to the signature.
#[pyo3::pyfunction]
#[pyo3(signature = (x1, _x2, /, _out=None, *, _where=None, _casting="same_kind", _order="K", _dtype=None, _subok=true, _signature=None))]
#[allow(clippy::too_many_arguments)]
fn probe_nine(
    x1: pyo3::Py<pyo3::PyAny>,
    _x2: pyo3::Py<pyo3::PyAny>,
    _out: Option<pyo3::Py<pyo3::PyAny>>,
    _where: Option<pyo3::Py<pyo3::PyAny>>,
    _casting: &str,
    _order: &str,
    _dtype: Option<pyo3::Py<pyo3::PyAny>>,
    _subok: bool,
    _signature: Option<pyo3::Py<pyo3::PyAny>>,
) -> pyo3::Py<pyo3::PyAny> {
    x1
}

/// The PARAMETER-COUNT SCALING probe `deadlock-audit-7ocfa` made the condition of its
/// own reopening (`deadlock-audit-t4lri`).
///
/// 7ocfa measured nine-parameter binding against `(*args, **kwargs)` at 0.0 ns and
/// REJECTED the signature as a wrapper-floor suspect, but wrote an explicit retry
/// predicate: "Reopen ONLY if a probe shows a per-call cost that scales with the NUMBER
/// of declared parameters - e.g. the same two probes at 3, 9 and 20 parameters." That
/// predicate has never been discharged, and `deadlock-audit-t4lri` proposes rewriting
/// `PyUFunc::__call__`'s signature without discharging it.
///
/// WHY IT MATTERS NOW. With the probe chain 83% removed, `wrapper_residual_ns` = 370 ns
/// is 43% of our `multiply` call and the largest remaining component. The nine-parameter
/// signature sits inside it. If binding cost is FLAT in parameter count, that half of
/// t4lri is closed without touching production code - and the rewrite it proposes is the
/// risky kind, since hand-parsing the keyword surface is what previously drifted NumPy's
/// `where=` to `where_arg`.
///
/// THE CONTROL: three `#[pyfunction]`s with IDENTICAL bodies - each returns its first
/// argument - differing ONLY in how many parameters they declare. Same binary, same
/// invocation, same operands.
#[pyo3::pyfunction]
#[pyo3(signature = (x1, _x2, /, _out=None))]
fn probe_three(
    x1: pyo3::Py<pyo3::PyAny>,
    _x2: pyo3::Py<pyo3::PyAny>,
    _out: Option<pyo3::Py<pyo3::PyAny>>,
) -> pyo3::Py<pyo3::PyAny> {
    x1
}

#[pyo3::pyfunction]
#[pyo3(signature = (
    x1, _x2, /, _out=None, *, _p1=None, _p2=None, _p3=None, _p4=None, _p5=None, _p6=None,
    _p7=None, _p8=None, _p9=None, _p10=None, _p11=None, _p12=None, _p13=None, _p14=None,
    _p15=None, _p16=None, _p17=None
))]
#[allow(clippy::too_many_arguments)]
fn probe_twenty(
    x1: pyo3::Py<pyo3::PyAny>,
    _x2: pyo3::Py<pyo3::PyAny>,
    _out: Option<pyo3::Py<pyo3::PyAny>>,
    _p1: Option<pyo3::Py<pyo3::PyAny>>,
    _p2: Option<pyo3::Py<pyo3::PyAny>>,
    _p3: Option<pyo3::Py<pyo3::PyAny>>,
    _p4: Option<pyo3::Py<pyo3::PyAny>>,
    _p5: Option<pyo3::Py<pyo3::PyAny>>,
    _p6: Option<pyo3::Py<pyo3::PyAny>>,
    _p7: Option<pyo3::Py<pyo3::PyAny>>,
    _p8: Option<pyo3::Py<pyo3::PyAny>>,
    _p9: Option<pyo3::Py<pyo3::PyAny>>,
    _p10: Option<pyo3::Py<pyo3::PyAny>>,
    _p11: Option<pyo3::Py<pyo3::PyAny>>,
    _p12: Option<pyo3::Py<pyo3::PyAny>>,
    _p13: Option<pyo3::Py<pyo3::PyAny>>,
    _p14: Option<pyo3::Py<pyo3::PyAny>>,
    _p15: Option<pyo3::Py<pyo3::PyAny>>,
    _p16: Option<pyo3::Py<pyo3::PyAny>>,
    _p17: Option<pyo3::Py<pyo3::PyAny>>,
) -> pyo3::Py<pyo3::PyAny> {
    x1
}

#[pyo3::pyfunction]
#[pyo3(signature = (*args, **_kwargs))]
fn probe_varargs(
    args: &pyo3::Bound<'_, PyTuple>,
    _kwargs: Option<&pyo3::Bound<'_, PyDict>>,
) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
    Ok(args.get_item(0)?.unbind())
}

/// Attribute or bound the LAST two unexplored stages of the `PyUFunc::__call__`
/// wrapper floor: `PyBuffer` acquisition and output construction
/// (`deadlock-audit-tmmud`).
///
/// Everything else named has been measured and eliminated: four stages at 700 ns
/// combined (`deadlock-audit-cydda`), the delegation default-kwargs dict at 727 ns
/// (`deadlock-audit-s2fkk`, removed), and PyO3 argument binding at 0.0 ns
/// (`deadlock-audit-7ocfa`, refuted). Against a 6968-8375 ns floor that leaves
/// buffer acquisition and result construction.
///
/// BATCHED ON PURPOSE. `deadlock-audit-7ocfa`'s probe bottomed out: at a 60 ns call
/// two IDENTICAL arms read 7% apart from timer quantisation, so a stage cheaper than
/// ~10 ns is invisible to a single-call arm. Each observation here runs `BATCH`
/// repetitions inside one timer, so a 10 ns stage shows up as a 2.5 us
/// observation. An empty-loop arm is measured and SUBTRACTED so the reported
/// per-call figures are the stage, not the loop.
/// Hold the NO-OP RESHAPE up on its own (`deadlock-audit-6twge`).
///
/// `try_zerocopy_f64_binary` used to call `reshape((n,))` on an output that already
/// had shape `(n,)`. Arm A is that no-op; arm B skips it. Both arms allocate the
/// same array first, so the delta is the reshape and nothing else. One binary, one
/// invocation, one worker, ABBAABBA, A/A null first — the shape that has now
/// isolated four levers in a row, and the only shape that is sound while the rch
/// fleet is heterogeneous, unpinnable, and shared with peers' uncommitted work.
/// End-to-end vs-NumPy for the four native f64 binary routes that have never had an
/// op-level row: `power`, `maximum`, `minimum`, `floor_divide`
/// (`deadlock-audit-4j5ba`). `remainder` and `divide` are deliberately absent - they
/// are being worked elsewhere and duplicating them would just collide.
///
/// SIZE IS LOAD-BEARING. n = 1<<22 clears every gate involved:
/// `FLOAT_POWER_PARALLEL_MIN_LEN` is 16_384 and the Maximum/Minimum/Div
/// `parallel_min` is 1<<21, so all four take their PARALLEL native arm. Measuring
/// below a gate characterises the serial arm only, which is the correction
/// `deadlock-audit-su0i6`'s row needed.
///
/// Operands are the file's standard pair - `a` in [1,2), `b` in [1.25,2.25) - which
/// are safe for all four without special-casing: `power` cannot overflow or produce
/// NaN there, `floor_divide` has no zero divisor.
///
/// Each op goes through `measure_binary_ufunc_vs_numpy`, so each carries the runtime
/// dispatch trap (fnp's callable is asserted not to be NumPy's object) and the
/// cross-arm parity probe, under the dual-null contract with both A/A nulls. Ratio
/// is incumbent/candidate = numpy/fnp, so BELOW 1.0 means we are slower.
/// Decompose OUTPUT CONSTRUCTION, the largest unexplained piece of the PyUFunc
/// per-call floor, inside ONE invocation (`deadlock-audit-tmmud` follow-up).
///
/// The numbers that motivate this do not add up, and they cannot be made to add up
/// because they come from different workers with different harnesses:
///   `numpy.empty` alone            220 ns   (`deadlock-audit-cydda`, vmi1149989, min-of-2001)
///   `numpy.empty` + `reshape`     1310 ns   (`deadlock-audit-tmmud`, vmi1152480, batched min-of-401)
///   the reshape in isolation       200 ns   (`deadlock-audit-6twge`, vmi1293453, contract A/B)
/// 220 + 200 is 420, not 1310, so roughly 890 ns is unaccounted — but subtracting
/// across those rows is precisely the cross-worker arithmetic that already
/// overstated one lever by 5.5x, so the gap is NOT evidence of anything yet. This
/// group measures every piece side by side on ONE worker in ONE process so the
/// arithmetic is finally legitimate.
///
/// BATCHED, because `deadlock-audit-7ocfa` showed a single-call arm bottoms out at
/// the timer floor: two IDENTICAL arms read 7% apart at a 60 ns call. Each stage runs
/// `BATCH` repetitions inside one timer and an empty loop is measured and subtracted.
///
/// The `empty_at_final_shape` stage is the one with a lever behind it: if allocating
/// directly at the target shape costs about the same as allocating flat, then the
/// higher-rank sites can skip their reshape too, not just the 1-D ones.
/// Ceiling for the cached-callable lever (`deadlock-audit-v8nx6`): what would we save
/// if the per-call Python lookups were free?
///
/// The route pays two lookups on every invocation - `py.import("numpy")` at the top of
/// `PyUFunc::__call__`, and the attribute half of `numpy.call_method("empty", ..)`.
/// They were measured separately at 310 ns (`deadlock-audit-cydda`) and 110.09 ns
/// (`deadlock-audit-k4yus`), on different workers with different harnesses, so they
/// have never been compared like for like or against a hoisted alternative.
///
/// Each pair below is IDENTICAL work differing only in whether the lookup is repeated
/// or hoisted, in one binary, one invocation, ABBAABBA, A/A null first. The hoisted arm
/// is the ceiling: a real cache cannot beat holding the handle in a local, and it must
/// additionally survive GIL reacquisition and interpreter finalisation, which is the
/// risk this measurement exists to justify or refuse.
fn bench_python_lookup_hoisting_ceiling(_c: &mut Criterion) {
    const N: usize = 4096;
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let numpy = py.import("numpy").expect("numpy oracle");
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy.__version__")
            .extract::<String>()
            .expect("numpy version is a string");
        let hoisted_empty = numpy.getattr("empty").expect("hoisted empty");

        // Pair 1: the attribute lookup on numpy.empty, repeated versus hoisted.
        let lookup_each_call = || {
            let started = Instant::now();
            let out = numpy
                .call_method1("empty", (N,))
                .expect("empty via call_method1");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: u64::from(out.is_none()),
            }
        };
        let lookup_hoisted = || {
            let started = Instant::now();
            let out = hoisted_empty
                .call1((N,))
                .expect("empty via hoisted callable");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: u64::from(out.is_none()),
            }
        };
        let (empty_effect, empty_null) = common::run_median_ci_contract(
            "python_lookup_hoisting_empty",
            lookup_each_call,
            lookup_hoisted,
        );

        // Pair 2: py.import("numpy") repeated versus a held module handle. Both arms do
        // the same trivial attribute read afterwards so the import is what differs.
        let import_each_call = || {
            let started = Instant::now();
            let m = py.import("numpy").expect("import numpy");
            let out = m.getattr("pi").expect("pi");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: u64::from(out.is_none()),
            }
        };
        let import_hoisted = || {
            let started = Instant::now();
            let out = numpy.getattr("pi").expect("pi");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: u64::from(out.is_none()),
            }
        };
        let (import_effect, import_null) = common::run_median_ci_contract(
            "python_lookup_hoisting_import",
            import_each_call,
            import_hoisted,
        );

        println!(
            "PYTHON_LOOKUP_HOISTING n={N} numpy_version={numpy_version} worker={} \
             harness=common::run_median_ci_contract rounds={} \
             arms=identical_work_differing_only_in_repeated_vs_hoisted_lookup \
             empty_repeated_ns={:.1} empty_hoisted_ns={:.1} empty_ratio={:.6} \
             empty_ci95=[{:.6},{:.6}] empty_null={:.6} empty_saves_ns={:.1} \
             import_repeated_ns={:.1} import_hoisted_ns={:.1} import_ratio={:.6} \
             import_ci95=[{:.6},{:.6}] import_null={:.6} import_saves_ns={:.1} \
             combined_ceiling_ns={:.1}",
            measurement_worker(),
            common::CONTRACT_ROUNDS,
            empty_effect.arm_a_median_ns,
            empty_effect.arm_b_median_ns,
            empty_effect.ratio_median,
            empty_effect.ratio_ci_low,
            empty_effect.ratio_ci_high,
            empty_null.ratio_median,
            empty_effect.arm_a_median_ns - empty_effect.arm_b_median_ns,
            import_effect.arm_a_median_ns,
            import_effect.arm_b_median_ns,
            import_effect.ratio_median,
            import_effect.ratio_ci_low,
            import_effect.ratio_ci_high,
            import_null.ratio_median,
            import_effect.arm_a_median_ns - import_effect.arm_b_median_ns,
            (empty_effect.arm_a_median_ns - empty_effect.arm_b_median_ns)
                + (import_effect.arm_a_median_ns - import_effect.arm_b_median_ns),
        );
    });
}

fn bench_output_construction_decomposition(_c: &mut Criterion) {
    const N: usize = 4096;
    const BATCH: usize = 256;
    const TRIALS: usize = 401;

    fn min_batch_ns(trials: usize, batch: usize, mut op: impl FnMut()) -> f64 {
        let mut best = u128::MAX;
        for _ in 0..trials {
            let started = Instant::now();
            for _ in 0..batch {
                op();
            }
            best = best.min(started.elapsed().as_nanos());
        }
        best as f64 / batch as f64
    }

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let numpy = py.import("numpy").expect("numpy oracle");
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy.__version__")
            .extract::<String>()
            .expect("numpy version is a string");
        let flat_shape = PyTuple::new(py, [N]).expect("flat shape");
        let two_d = PyTuple::new(py, [64usize, 64usize]).expect("2-D shape");

        let empty_loop = min_batch_ns(TRIALS, BATCH, || {
            black_box(());
        });
        // The attribute lookup alone, so it can be separated from the call.
        let getattr_empty = min_batch_ns(TRIALS, BATCH, || {
            black_box(numpy.getattr("empty").expect("getattr empty"));
        });
        // Allocation only.
        let empty_flat = min_batch_ns(TRIALS, BATCH, || {
            black_box(numpy.call_method1("empty", (N,)).expect("empty"));
        });
        // Allocation + the no-op reshape to the shape it already has.
        let empty_plus_noop_reshape = min_batch_ns(TRIALS, BATCH, || {
            let f = numpy.call_method1("empty", (N,)).expect("empty");
            black_box(f.call_method1("reshape", (&flat_shape,)).expect("reshape"));
        });
        // Allocation + a REAL reshape to a different rank.
        let empty_plus_real_reshape = min_batch_ns(TRIALS, BATCH, || {
            let f = numpy.call_method1("empty", (N,)).expect("empty");
            black_box(f.call_method1("reshape", (&two_d,)).expect("reshape"));
        });
        // Allocation directly at the final 2-D shape - the candidate lever.
        let empty_at_final_shape = min_batch_ns(TRIALS, BATCH, || {
            black_box(numpy.call_method1("empty", (&two_d,)).expect("empty 2-D"));
        });

        let noop_reshape = empty_plus_noop_reshape - empty_flat;
        let real_reshape = empty_plus_real_reshape - empty_flat;
        let direct_saves = empty_plus_real_reshape - empty_at_final_shape;
        println!(
            "OUTPUT_CONSTRUCTION_DECOMPOSITION n={N} batch={BATCH} trials={TRIALS} \
             numpy_version={numpy_version} worker={} \
             harness=batched_min_of_{TRIALS}_minus_empty_loop one_invocation=true \
             empty_loop_ns={empty_loop:.2} getattr_empty_ns={:.2} \
             empty_flat_ns={:.2} empty_plus_noop_reshape_ns={:.2} \
             empty_plus_real_reshape_ns={:.2} empty_at_final_shape_ns={:.2} \
             noop_reshape_ns={noop_reshape:.2} real_reshape_ns={real_reshape:.2} \
             allocating_at_final_shape_saves_ns={direct_saves:.2}",
            measurement_worker(),
            getattr_empty - empty_loop,
            empty_flat - empty_loop,
            empty_plus_noop_reshape - empty_loop,
            empty_plus_real_reshape - empty_loop,
            empty_at_final_shape - empty_loop,
        );
    });
}

fn bench_native_binary_family_vs_numpy(_c: &mut Criterion) {
    let n = 1usize << 22;
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

        for op in ["power", "maximum", "minimum", "floor_divide"] {
            let (ratio, lo, hi, incumbent_ns, candidate_ns) =
                measure_binary_ufunc_vs_numpy(py, &module, &numpy, op, &args, n);
            println!(
                "NATIVE_BINARY_VS_NUMPY op={op} n={n} regime=parallel_arm \
                 numpy_version={numpy_version} worker={} \
                 harness=common::run_dual_null_median_ci_contract rounds={} \
                 rayon_threads={} numpy_ns={incumbent_ns:.1} fnp_ns={candidate_ns:.1} \
                 ratio_median={ratio:.6} ratio_ci95=[{lo:.6},{hi:.6}] \
                 faster_than_numpy={} slower_than_numpy={}",
                measurement_worker(),
                common::CONTRACT_ROUNDS,
                rayon::current_num_threads(),
                lo > 1.0,
                hi < 1.0,
            );
        }
    });
}

fn bench_noop_reshape_cost(_c: &mut Criterion) {
    const N: usize = 256;
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let numpy = py.import("numpy").expect("numpy oracle");
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy.__version__")
            .extract::<String>()
            .expect("numpy version is a string");
        let shape = PyTuple::new(py, [N]).expect("shape tuple");

        let with_reshape = || {
            let started = Instant::now();
            let flat = numpy.call_method1("empty", (N,)).expect("numpy.empty");
            let out = flat.call_method1("reshape", (&shape,)).expect("reshape");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: u64::from(out.is_none()),
            }
        };
        let without_reshape = || {
            let started = Instant::now();
            let out = numpy.call_method1("empty", (N,)).expect("numpy.empty");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: u64::from(out.is_none()),
            }
        };

        let (effect, null) =
            common::run_median_ci_contract("noop_reshape_cost_n256", with_reshape, without_reshape);
        println!(
            "NOOP_RESHAPE_COST n={N} numpy_version={numpy_version} worker={} \
             harness=common::run_median_ci_contract rounds={} \
             arms=numpy_empty_only_no_fnp_code \
             with_reshape_ns={:.1} without_reshape_ns={:.1} ratio_median={:.6} \
             ratio_ci95=[{:.6},{:.6}] null_ratio_median={:.6} null_ci95=[{:.6},{:.6}] \
             reshape_cost_ns={:.1} skipping_is_faster={}",
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
            effect.arm_a_median_ns - effect.arm_b_median_ns,
            effect.ratio_ci_low > 1.0,
        );
    });
}

fn bench_wrapper_remainder_stages(_c: &mut Criterion) {
    const N: usize = 256;
    const BATCH: usize = 256;
    const TRIALS: usize = 401;

    fn min_batch_ns(trials: usize, batch: usize, mut op: impl FnMut()) -> f64 {
        let mut best = u128::MAX;
        for _ in 0..trials {
            let started = Instant::now();
            for _ in 0..batch {
                op();
            }
            best = best.min(started.elapsed().as_nanos());
        }
        best as f64 / batch as f64
    }

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
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
        let np_add = numpy.getattr("add").expect("numpy.add");

        // Loop overhead, subtracted from every stage below.
        let empty_ns = min_batch_ns(TRIALS, BATCH, || {
            black_box(());
        });
        // Stage: acquire a zero-copy f64 view of BOTH operands, which is what the
        // native route does before it can touch an element.
        let buffers_ns = min_batch_ns(TRIALS, BATCH, || {
            let ba = pyo3::buffer::PyBuffer::<f64>::get(&a).expect("buffer a");
            let bb = pyo3::buffer::PyBuffer::<f64>::get(&b).expect("buffer b");
            black_box((ba.item_count(), bb.item_count()));
        });
        // Stage: construct the output the way the route does - allocate flat, then
        // reshape back to the operand shape.
        let output_ns = min_batch_ns(TRIALS, BATCH, || {
            let flat = numpy.call_method1("empty", (N,)).expect("numpy.empty");
            let shaped = flat.call_method1("reshape", ((N,),)).expect("reshape");
            black_box(shaped);
        });
        // Denominator: a whole delegating call through NumPy itself.
        let numpy_add_ns = min_batch_ns(TRIALS, BATCH, || {
            black_box(np_add.call1(&args).expect("numpy.add"));
        });

        let buffers = buffers_ns - empty_ns;
        let output = output_ns - empty_ns;
        println!(
            "WRAPPER_REMAINDER_STAGES n={N} batch={BATCH} trials={TRIALS} \
             numpy_version={numpy_version} worker={} harness=batched_min_of_{TRIALS}_minus_empty_loop \
             stages_are_standalone_replicas=true \
             empty_loop_ns={empty_ns:.2} pybuffer_get_both_ns={buffers:.2} \
             output_empty_plus_reshape_ns={output:.2} numpy_add_whole_call_ns={numpy_add_ns:.2} \
             two_stages_sum_ns={:.2}",
            measurement_worker(),
            buffers + output,
        );
    });
}

// Discharges `deadlock-audit-7ocfa`'s retry predicate, which `deadlock-audit-t4lri`
// currently proposes acting against without having discharged it.
//
// 7ocfa rejected the nine-parameter signature as a wrapper-floor suspect (binding measured
// at 0.0 ns against `(*args, **kwargs)`) but named exactly one condition for reopening: a
// probe showing per-call cost that SCALES with the number of declared parameters, at 3, 9
// and 20. This is that probe.
//
// WHY IT IS WORTH RUNNING NOW rather than earlier: with the probe chain 83% removed
// (521 -> 91 ns), `wrapper_residual_ns` = 370 ns is 43% of our `multiply` call and the
// largest remaining term. The signature sits inside it, so this decides whether t4lri's
// rewrite is justified at all — and that rewrite is the risky kind, since hand-parsing the
// keyword surface previously drifted NumPy's `where=` to `where_arg`.
//
// READ THE RESULT THIS WAY. Three identical bodies differing only in declared parameter
// count. If the three medians sit within the harness's own resolution, cost is FLAT in
// parameter count, 7ocfa's predicate is NOT met, and the signature half of t4lri closes
// with no production change. If they scale, the predicate IS met and t4lri reopens with a
// measured slope rather than an assumption.
//
// The row deliberately reports the raw per-probe medians and the two deltas rather than a
// verdict: this ledger has three retractions today from differencing nearly-equal measured
// quantities, and a 3-to-20 delta on sub-microsecond calls is exactly that shape.
fn bench_pyo3_signature_parameter_scaling(_c: &mut Criterion) {
    const TRIALS: usize = 2001;

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
        let numpy = py.import("numpy").expect("numpy oracle");
        let probe_mod = PyModule::new(py, "fnp_param_scaling_probe").expect("probe module");
        probe_mod
            .add_function(pyo3::wrap_pyfunction!(probe_three, &probe_mod).expect("wrap three"))
            .expect("register three");
        probe_mod
            .add_function(pyo3::wrap_pyfunction!(probe_nine, &probe_mod).expect("wrap nine"))
            .expect("register nine");
        probe_mod
            .add_function(pyo3::wrap_pyfunction!(probe_twenty, &probe_mod).expect("wrap twenty"))
            .expect("register twenty");
        let three = probe_mod.getattr("probe_three").expect("probe_three");
        let nine = probe_mod.getattr("probe_nine").expect("probe_nine");
        let twenty = probe_mod.getattr("probe_twenty").expect("probe_twenty");

        let locals = PyDict::new(py);
        locals.set_item("np", &numpy).expect("bind numpy");
        py.run(
            std::ffi::CString::new("a = np.arange(256.0)\nb = np.arange(256.0) + 1\n")
                .unwrap()
                .as_c_str(),
            Some(&locals),
            Some(&locals),
        )
        .expect("build operands");
        let a = locals.get_item("a").expect("a");
        let b = locals.get_item("b").expect("b");
        let args = PyTuple::new(py, [&a, &b]).expect("args");

        // Every probe must return its first argument, or the bodies are not identical and
        // the comparison measures something other than binding.
        for (name, probe) in [("three", &three), ("nine", &nine), ("twenty", &twenty)] {
            let out = probe.call1(&args).expect("probe call");
            assert!(
                out.is(&a),
                "probe_{name} must return its FIRST argument unchanged, or the three \
                 bodies are not identical and this comparison is meaningless"
            );
        }

        let three_ns = min_ns(TRIALS, || {
            black_box(three.call1(&args).expect("probe_three call"));
        });
        let nine_ns = min_ns(TRIALS, || {
            black_box(nine.call1(&args).expect("probe_nine call"));
        });
        let twenty_ns = min_ns(TRIALS, || {
            black_box(twenty.call1(&args).expect("probe_twenty call"));
        });

        let per_param_3_to_9 = (nine_ns - three_ns) / 6.0;
        let per_param_9_to_20 = (twenty_ns - nine_ns) / 11.0;
        println!(
            "PYO3_PARAM_SCALING worker={} harness=replica_min_of_{TRIALS} trials={TRIALS} \
             arms=identical_bodies_differing_only_in_declared_parameter_count \
             params_3_ns={three_ns:.1} params_9_ns={nine_ns:.1} params_20_ns={twenty_ns:.1} \
             delta_3_to_9_ns={:.1} delta_9_to_20_ns={:.1} \
             per_param_3_to_9_ns={per_param_3_to_9:.2} per_param_9_to_20_ns={per_param_9_to_20:.2} \
             discharges=deadlock-audit-7ocfa_retry_predicate \
             wrapper_residual_reference_ns=370.0",
            measurement_worker(),
            nine_ns - three_ns,
            twenty_ns - nine_ns,
        );
    });
}

fn bench_pyo3_signature_binding_cost(_c: &mut Criterion) {
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let numpy = py.import("numpy").expect("numpy oracle");
        let probe_mod = PyModule::new(py, "fnp_binding_probe").expect("probe module");
        probe_mod
            .add_function(pyo3::wrap_pyfunction!(probe_nine, &probe_mod).expect("wrap nine"))
            .expect("add nine");
        probe_mod
            .add_function(pyo3::wrap_pyfunction!(probe_varargs, &probe_mod).expect("wrap varargs"))
            .expect("add varargs");
        let nine = probe_mod.getattr("probe_nine").expect("probe_nine");
        let varargs = probe_mod.getattr("probe_varargs").expect("probe_varargs");

        // The bodies ignore the operands, but binding an ndarray is what the real
        // signature does, so hand them the same kind of object.
        let locals = PyDict::new(py);
        locals.set_item("np", &numpy).expect("bind numpy");
        py.run(
            std::ffi::CString::new("a = np.arange(256, dtype=np.float64)\nb = a + 1.0\n")
                .unwrap()
                .as_c_str(),
            Some(&locals),
            Some(&locals),
        )
        .expect("build operands");
        let a = locals.get_item("a").expect("a");
        let b = locals.get_item("b").expect("b");
        let args = PyTuple::new(py, [&a, &b]).expect("args");

        let call_nine = || {
            let started = Instant::now();
            let out = nine.call1(&args).expect("probe_nine call");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: u64::from(out.is_none()),
            }
        };
        let call_varargs = || {
            let started = Instant::now();
            let out = varargs.call1(&args).expect("probe_varargs call");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: u64::from(out.is_none()),
            }
        };

        let (effect, null) =
            common::run_median_ci_contract("pyo3_signature_binding_cost", call_nine, call_varargs);
        println!(
            "PYO3_BINDING_COST worker={} harness=common::run_median_ci_contract rounds={} \
             arms=identical_bodies_differing_only_in_signature \
             nine_param_ns={:.1} varargs_ns={:.1} ratio_median={:.6} \
             ratio_ci95=[{:.6},{:.6}] null_ratio_median={:.6} null_ci95=[{:.6},{:.6}] \
             binding_cost_ns={:.1} varargs_is_faster={}",
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
            effect.arm_a_median_ns - effect.arm_b_median_ns,
            effect.ratio_ci_low > 1.0,
        );
    });
}

fn bench_delegation_kwargs_shape(_c: &mut Criterion) {
    const N: usize = 256;
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
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
        let np_multiply = numpy.getattr("multiply").expect("numpy.multiply");

        // Arm A: the shape the delegation tail used to emit.
        let with_default_kwargs = || {
            let started = Instant::now();
            let kwargs = PyDict::new(py);
            kwargs.set_item("casting", "same_kind").expect("casting");
            kwargs.set_item("order", "K").expect("order");
            kwargs.set_item("subok", true).expect("subok");
            let result = np_multiply
                .call(&args, Some(&kwargs))
                .expect("numpy multiply with default kwargs");
            let elapsed = started.elapsed();
            let checksum = numpy_divide_checksum(&result, N);
            common::ContractObservation { elapsed, checksum }
        };
        // Arm B: the shape it emits now.
        let without_kwargs = || {
            let started = Instant::now();
            let result = np_multiply.call1(&args).expect("numpy multiply bare");
            let elapsed = started.elapsed();
            let checksum = numpy_divide_checksum(&result, N);
            common::ContractObservation { elapsed, checksum }
        };

        let (effect, null) = common::run_median_ci_contract(
            "delegation_kwargs_shape_n256",
            with_default_kwargs,
            without_kwargs,
        );
        println!(
            "DELEGATION_KWARGS_SHAPE n={N} numpy_version={numpy_version} worker={} \
             harness=common::run_median_ci_contract rounds={} \
             arms=numpy_multiply_only_no_fnp_code \
             with_default_kwargs_ns={:.1} without_kwargs_ns={:.1} \
             ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
             null_ratio_median={:.6} null_ci95=[{:.6},{:.6}] \
             saved_ns={:.1} bare_call_is_faster={}",
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
            effect.arm_a_median_ns - effect.arm_b_median_ns,
            effect.ratio_ci_low > 1.0,
        );
    });
}

/// The DELEGATING CONTROL the divide-gate decision needs (`deadlock-audit-q00ev`).
///
/// `deadlock-audit-qapyb` set `F64_DIV_NATIVE_MIN_LEN = 1 << 14`. Three sizes measured
/// post-fix show the native divide is at its WORST just above that gate — 28.4% overhead
/// at 2^16 against 5.4% at 2^20 — which raises the question of whether the gate admits a
/// band where DELEGATING would be cheaper than our own kernel.
///
/// THAT QUESTION CANNOT BE ANSWERED FROM THOSE THREE ROWS, and the reason is the whole
/// point of this group. Answering it needs the WRAPPER cost at each size, and the wrapper
/// is NOT constant in n: `multiply` delegates at every size here and its excess grew 6.2x
/// between n=256 and n=2^20. Carrying the small-n wrapper figure upward is exactly the
/// cross-size subtraction that overstated `deadlock-audit-6twge` by 5.5x.
///
/// So both ops are measured at EVERY size in ONE invocation. `multiply` delegates
/// throughout, so its excess IS the wrapper cost at that size; `divide` goes native above
/// the gate. Every subtraction below is therefore WITHIN a size and WITHIN an invocation.
///
/// THE METHOD CARRIES ITS OWN NULL. At 2^12 — below the gate — BOTH ops delegate, so they
/// pay the same wrapper and `divide_specific_excess_ns` must come out near zero. If it does
/// not, the subtraction is contaminated and no row from this group may be believed. That
/// cell exists to be checked, not to be reported as a finding.
fn bench_percall_floor_across_sizes_vs_numpy(_c: &mut Criterion) {
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

        let mut previous_incumbent: Option<(u32, f64, f64)> = None;
        let mut incumbent_scaling_violations: Vec<String> = Vec::new();
        let mut null_cell_violations: Vec<String> = Vec::new();

        // 2^12 is BELOW the 1<<14 gate and is the method null; 2^14 sits at it; the rest
        // are above it. Four cells above the gate locate the worst band instead of
        // inferring it from the two the existing size-gate group happens to carry.
        for exponent in [12u32, 14, 16, 18, 20] {
            let n = 1usize << exponent;
            let below_gate = n < (1usize << 19);
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

            // `multiply` first and `divide` second, both in this invocation, both against
            // their own NumPy arm. The order is fixed across sizes so any drift is common.
            let (mul_ratio, mul_lo, mul_hi, mul_numpy_ns, mul_fnp_ns) =
                measure_binary_ufunc_vs_numpy(py, &module, &numpy, "multiply", &args, n);
            let (div_ratio, div_lo, div_hi, div_numpy_ns, div_fnp_ns) =
                measure_binary_ufunc_vs_numpy(py, &module, &numpy, "divide", &args, n);

            // STATISTIC CONSISTENCY (`deadlock-audit-q00ev`, defect found on this group's
            // own first runs). `ratio` from the harness is a MEDIAN OF PAIRED RATIOS;
            // the `*_ns` are ARM MEDIANS. They are different statistics and diverge when
            // the paired ratios are skewed — at n=2^20 one run reported
            // multiply_ratio=0.983313 while its own arm medians said 305925/300104 =
            // 1.019. Deriving the gate decision from the ns DIFFERENCE mixed the two and
            // produced wrapper_ns=-5821 (a negative wrapper cost) and
            // projected_delegating_ratio=1.018389, which asserts that delegating to NumPy
            // is faster than NumPy's own call. Both are impossible, not merely imprecise.
            //
            // THE DECISION IS NOW DERIVED FROM THE RATIO ALONE. If `divide` delegated it
            // would pay exactly what `multiply` pays — same wrapper, same delegation tail
            // — so its ratio WOULD BE multiply's ratio. That is the projection. It uses
            // one statistic end to end and cannot produce an impossible value, because
            // multiply_ratio is itself a measured ratio against NumPy. It also explains
            // why the old projection landed within ~1% of multiply_ratio at four of five
            // sizes: that agreement was the model being right, not the arithmetic.
            // CORRECTED (`deadlock-audit-q00ev`). The previous form set
            // `projected_delegating_ratio = mul_ratio`, which is a WRONG MODEL: a
            // delegating op costs `numpy_op + wrapper`, so its ratio is
            // `numpy_op / (numpy_op + wrapper)`. The wrapper is shared between the two
            // ops but `numpy_divide` is DEARER than `numpy_multiply`, so the same wrapper
            // is a smaller fraction of a dearer call and a delegating divide necessarily
            // scores BETTER than a delegating multiply. Equating the two ratios discarded
            // the denominator that makes them differ, and the 2^12 null cell caught it by
            // reporting `delegating_looks_better=false` where both ops delegate.
            //
            // The original model was right; only its arithmetic mixed statistics. So keep
            // the model and estimate the wrapper from the RATIO rather than from an
            // arm-median difference — one statistic, and no impossible values.
            // POSITIVITY PRECONDITION (`deadlock-audit-q00ev`, the fourth and last known gap
            // in this group). The wrapper is estimated from `multiply_ratio`, so when
            // multiply has reached PARITY the estimate is meaningless: at n=2^20 one run
            // measured multiply_ratio=1.003433 with a CI of [0.983145,1.034903] straddling
            // unity, and the formula returned wrapper = -1473.9 ns and a projected ratio of
            // 1.002870 — asserting that delegating to NumPy beats NumPy's own call. The
            // model is fine; it is being asked a question the data cannot answer, because
            // the wrapper at that size is smaller than the measurement floor. The
            // arm-median estimator fails there too (-120.0 ns), so this is a property of
            // the data rather than of either estimator.
            //
            // So the projection is declared UNAVAILABLE whenever multiply's CI contains
            // unity, rather than computed into an impossible number that a reader has to
            // catch by hand.
            let projection_available = !(mul_lo <= 1.0 && 1.0 <= mul_hi);
            let wrapper_from_ratio_ns = mul_numpy_ns * (1.0 / mul_ratio - 1.0);
            let projected_delegating_fnp_ns = div_numpy_ns + wrapper_from_ratio_ns;
            let projected_delegating_ratio = div_numpy_ns / projected_delegating_fnp_ns;
            let delegating_looks_better = projected_delegating_ratio > div_ratio;

            // NULL-CELL SELF-CHECK. Below the gate BOTH ops delegate, so the projection
            // must land slightly ABOVE the measured divide ratio — above, not equal,
            // because a delegating divide still pays the small divide-specific residual
            // (the f64 binop-block size and dtype guards that multiply skips). If it
            // inverts, the model is wrong again and no cell in this group may be read.
            if below_gate && projection_available && projected_delegating_ratio <= div_ratio {
                null_cell_violations.push(format!(
                    "2^{exponent}: projected {projected_delegating_ratio:.6} did not exceed \
                     measured {div_ratio:.6} at a cell where both ops delegate"
                ));
            }

            // Absolute nanoseconds below are ARM MEDIANS, emitted for provenance only.
            // They must NOT be combined with the ratios above, which is the mistake this
            // block exists to prevent; the field names say so.
            let wrapper_arm_median_ns = mul_fnp_ns - mul_numpy_ns;
            let divide_excess_arm_median_ns = div_fnp_ns - div_numpy_ns;
            let divide_specific_excess_arm_median_ns =
                divide_excess_arm_median_ns - wrapper_arm_median_ns;

            // INCUMBENT SCALING GUARD. NumPy's own cost must grow with n. A cell where it
            // does not has an unrepresentative incumbent arm — exactly the shape that
            // produced a spurious "our divide is 1.23x FASTER than NumPy" at n=2^14 in
            // one run, where our arm was stable to 0.5% while NumPy's moved 2.4x and
            // broke its own scaling trend. A clean A/A null CANNOT detect this: the null
            // compares an arm against ITSELF within one invocation, so a uniformly slow
            // incumbent leaves the null on unity while the effect is wrong by a factor.
            // This cross-cell check is what catches it, and it fails the run rather than
            // relying on someone reading the column.
            let incumbent_scaling_ok = match previous_incumbent {
                Some((prev_exponent, prev_mul_ns, prev_div_ns)) => {
                    let ok = mul_numpy_ns > prev_mul_ns && div_numpy_ns > prev_div_ns;
                    if !ok {
                        incumbent_scaling_violations.push(format!(
                            "2^{exponent}: numpy multiply {mul_numpy_ns:.1} vs 2^{prev_exponent} \
                             {prev_mul_ns:.1}; numpy divide {div_numpy_ns:.1} vs {prev_div_ns:.1}"
                        ));
                    }
                    ok
                }
                None => true,
            };
            previous_incumbent = Some((exponent, mul_numpy_ns, div_numpy_ns));

            println!(
                "PERCALL_FLOOR_SIZES n={n} log2n={exponent} below_gate={below_gate} \
                 is_method_null_cell={below_gate} numpy_version={numpy_version} \
                 harness=common::run_dual_null_median_ci_contract \
                 multiply_ratio={mul_ratio:.6} multiply_ci95=[{mul_lo:.6},{mul_hi:.6}] \
                 multiply_numpy_ns={mul_numpy_ns:.1} multiply_fnp_ns={mul_fnp_ns:.1} \
                 divide_ratio={div_ratio:.6} divide_ci95=[{div_lo:.6},{div_hi:.6}] \
                 divide_numpy_ns={div_numpy_ns:.1} divide_fnp_ns={div_fnp_ns:.1} \
                 wrapper_arm_median_ns={wrapper_arm_median_ns:.1} \
                 divide_excess_arm_median_ns={divide_excess_arm_median_ns:.1} \
                 divide_specific_excess_arm_median_ns={divide_specific_excess_arm_median_ns:.1} \
                 ns_fields_are_arm_medians_do_not_mix_with_ratios=true \
                 projected_delegating_fnp_ns={projected_delegating_fnp_ns:.1} \
                 projected_delegating_ratio={projected_delegating_ratio:.6} \
                 wrapper_from_ratio_ns={wrapper_from_ratio_ns:.1} \
                 projection_available={projection_available} \
                 projection_derived_from=ratio_median_only \
                 incumbent_scaling_ok={incumbent_scaling_ok} \
                 delegating_looks_better={delegating_looks_better} same_invocation=true"
            );
        }

        // Printed first, failed second: the cells stay visible for diagnosis, but a run
        // whose incumbent did not scale must not be banked.
        assert!(
            incumbent_scaling_violations.is_empty(),
            "INCUMBENT ARM DID NOT SCALE WITH n — the cells above are printed for diagnosis \
             but MUST NOT be banked: {}",
            incumbent_scaling_violations.join(" | ")
        );
        assert!(
            null_cell_violations.is_empty(),
            "THE BELOW-GATE NULL CELL INVERTED — the projection model is wrong and no cell \
             in this group may be read: {}",
            null_cell_violations.join(" | ")
        );
    });
}

/// The CLEAN experiment for `deadlock-audit-t4lri`: a nine-parameter `__call__` against a
/// `(*args, **kwargs)` one, doing identical work.
///
/// `bench_signature_keyword_binding_cost` could only price binding seven PRESENT keywords
/// (276 ns) and said so in its own output — it is an upper bound on the keyword-handling
/// component, not the lever's value, because the hot path passes NO keywords and pays
/// PyO3 filling defaults instead. Pricing that needs two call surfaces that differ ONLY
/// in their signature, which needs two types. That was deferred while the build freeze
/// held, since no bench in this crate had ever declared a `#[pyclass]` and unverifiable
/// macro surface in a shared binary is a bad trade. Builds are permitted now.
///
/// Both types are called the SAME WAY — two positionals, no keywords, exactly the hot
/// path's shape — and both return their first argument. The nine-parameter arm pays PyO3
/// binding two positionals and filling seven defaults; the varargs arm pays a tuple
/// extract. That difference IS the lever.
#[pyclass]
struct SignatureNineParam;

#[pymethods]
impl SignatureNineParam {
    #[pyo3(signature = (x1, x2, /, out=None, *, r#where=None, casting="same_kind", order="K", dtype=None, subok=true, signature=None))]
    #[allow(clippy::too_many_arguments)]
    fn __call__(
        &self,
        x1: Py<PyAny>,
        x2: Py<PyAny>,
        out: Option<Py<PyAny>>,
        r#where: Option<Py<PyAny>>,
        casting: &str,
        order: &str,
        dtype: Option<Py<PyAny>>,
        subok: bool,
        signature: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        // Touch every bound parameter so none can be optimised out of the signature.
        let _ = (
            &x2, &out, &r#where, casting, order, &dtype, subok, &signature,
        );
        Ok(x1)
    }
}

#[pyclass]
struct SignatureVarargs;

#[pymethods]
impl SignatureVarargs {
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        &self,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        // A real lazy implementation would parse kwargs only when present; the hot path
        // passes none, so this is exactly the work it would do.
        let _ = kwargs;
        Ok(args.get_item(0)?.unbind())
    }
}

/// Price the nine-parameter signature directly (`deadlock-audit-t4lri`).
fn bench_signature_shape_pyclass_control(_c: &mut Criterion) {
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let numpy = py.import("numpy").expect("numpy oracle");
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy.__version__")
            .extract::<String>()
            .expect("numpy version is a string");
        let a = numpy
            .call_method1("arange", (256_usize,))
            .expect("operand a");
        let b = numpy
            .call_method1("arange", (256_usize,))
            .expect("operand b");
        let args = PyTuple::new(py, [&a, &b]).expect("args");

        let nine = Py::new(py, SignatureNineParam).expect("nine-parameter callable");
        let varargs = Py::new(py, SignatureVarargs).expect("varargs callable");
        let nine = nine.bind(py);
        let varargs = varargs.bind(py);

        // PARITY BEFORE TIMING: both must hand back the same object, or they are not
        // doing the same work and the difference is not a signature cost.
        let nine_probe = nine.call1(&args).expect("nine-parameter probe");
        let varargs_probe = varargs.call1(&args).expect("varargs probe");
        assert!(
            nine_probe.is(&a) && varargs_probe.is(&a),
            "both signature arms must return their first argument unchanged"
        );

        let nine_arm = || {
            let started = Instant::now();
            let out = nine.call1(&args).expect("nine-parameter call");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: u64::from(out.is_none()),
            }
        };
        let varargs_arm = || {
            let started = Instant::now();
            let out = varargs.call1(&args).expect("varargs call");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: u64::from(out.is_none()),
            }
        };
        let (effect, null) =
            common::run_median_ci_contract("signature_shape_pyclass", nine_arm, varargs_arm);

        let signature_cost_ns = effect.arm_a_median_ns - effect.arm_b_median_ns;
        println!(
            "SIGNATURE_SHAPE_PYCLASS n=256 numpy_version={numpy_version} worker={} \
             harness=common::run_median_ci_contract \
             arms=nine_parameter_pyclass_vs_varargs_pyclass_same_work \
             called_with_two_positionals_no_keywords=true \
             nine_param_ns={:.1} varargs_ns={:.1} signature_cost_ns={signature_cost_ns:.1} \
             ratio={:.6} ratio_ci95=[{:.6},{:.6}] null={:.6} \
             this_IS_eager_vs_lazy_on_the_hot_path_shape=true",
            measurement_worker(),
            effect.arm_a_median_ns,
            effect.arm_b_median_ns,
            effect.ratio_median,
            effect.ratio_ci_low,
            effect.ratio_ci_high,
            null.ratio_median,
        );
    });
}

/// TRIAGE for the nine-parameter signature (`deadlock-audit-t4lri`), using the REAL
/// route and adding no new types.
///
/// `PyUFunc::__call__` is declared with nine parameters, seven of them optional with
/// defaults, all bound by PyO3 before any work happens. `deadlock-audit-ei9jz` leaves
/// ~702 ns of a 1232 ns call unattributed and names this as the largest never-priced
/// candidate. The clean experiment is a nine-parameter signature against a
/// `(*args, **kwargs)` one, but that needs two new `#[pyclass]`es with `__call__`, and
/// no bench in this crate has ever declared a pyclass — new macro surface that cannot
/// be compiled while the build freeze holds.
///
/// SO THIS PRICES THE HALF THAT NEEDS NO NEW SURFACE. Both arms call the shipped ufunc
/// and are SEMANTICALLY all-defaults, because the route's gate compares keyword VALUES,
/// not presence: passing `casting="same_kind"`, `order="K"`, `subok=True` and
/// `out/where/dtype/signature=None` leaves every one of the seven gate conditions true,
/// so the probe block still runs and the fast delegation tail is still chosen. The only
/// difference is whether PyO3 binds seven keywords OUT OF A DICT or fills them from its
/// own defaults.
///
/// WHAT IT DECIDES. If binding seven present keywords costs little, then filling seven
/// defaults costs less still, PyO3's keyword machinery is not where the 702 ns lives,
/// and the whole `(*args, **kwargs)` proposal can be dropped without writing it — which
/// is the cheaper outcome. If it costs a lot, the pyclass experiment is worth building
/// when the freeze lifts.
///
/// WHAT IT DOES NOT DECIDE, stated so the row is not over-read: it is NOT a measurement
/// of eager-versus-lazy binding on the hot path. The hot path passes no keywords at all,
/// and this control cannot isolate what PyO3 spends filling defaults there. It bounds
/// the family; it does not price the lever.
fn bench_signature_keyword_binding_cost(_c: &mut Criterion) {
    const N: usize = 256;
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

        // Every keyword at the value PyO3 would have defaulted it to. This keeps
        // `out.is_none() && where.is_none() && dtype.is_none() && signature.is_none()
        // && casting == "same_kind" && order == "K" && subok` all true, so the route
        // taken is byte-for-byte the one `call1` takes.
        let default_kwargs = PyDict::new(py);
        default_kwargs.set_item("out", py.None()).expect("bind out");
        default_kwargs
            .set_item("where", py.None())
            .expect("bind where");
        default_kwargs
            .set_item("casting", "same_kind")
            .expect("bind casting");
        default_kwargs.set_item("order", "K").expect("bind order");
        default_kwargs
            .set_item("dtype", py.None())
            .expect("bind dtype");
        default_kwargs.set_item("subok", true).expect("bind subok");
        default_kwargs
            .set_item("signature", py.None())
            .expect("bind signature");

        // PARITY BEFORE TIMING: if the two call shapes do not agree bit for bit they are
        // not the same computation and the difference is not a signature cost.
        let bare_probe = ours.call1(&args).expect("bare call probe");
        let kwargs_probe = ours
            .call(&args, Some(&default_kwargs))
            .expect("explicit-defaults call probe");
        assert_eq!(
            numpy_divide_checksum(&bare_probe, N),
            numpy_divide_checksum(&kwargs_probe, N),
            "passing every keyword at its default changed the result — the two arms are \
             not the same computation and this control is invalid"
        );

        let bare = || {
            let started = Instant::now();
            let out = ours.call1(&args).expect("bare call");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: numpy_divide_checksum(&out, N),
            }
        };
        let explicit = || {
            let started = Instant::now();
            let out = ours
                .call(&args, Some(&default_kwargs))
                .expect("explicit-defaults call");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: numpy_divide_checksum(&out, N),
            }
        };
        let (effect, null) =
            common::run_median_ci_contract("signature_keyword_binding", explicit, bare);

        let keyword_binding_ns = effect.arm_a_median_ns - effect.arm_b_median_ns;
        println!(
            "SIGNATURE_KEYWORD_BINDING n={N} numpy_version={numpy_version} worker={} \
             harness=common::run_median_ci_contract \
             arms=same_route_same_result_differing_only_in_keyword_presence \
             seven_keywords_passed_at_their_defaults=true \
             explicit_kwargs_ns={:.1} bare_call_ns={:.1} \
             keyword_binding_ns={keyword_binding_ns:.1} \
             ratio={:.6} ratio_ci95=[{:.6},{:.6}] null={:.6} \
             prices_binding_present_keywords_NOT_eager_vs_lazy=true",
            measurement_worker(),
            effect.arm_a_median_ns,
            effect.arm_b_median_ns,
            effect.ratio_median,
            effect.ratio_ci_low,
            effect.ratio_ci_high,
            null.ratio_median,
        );
    });
}

/// Calls a ufunc METHOD with the arity that method takes. A closure cannot express
/// this: it would tie the returned `Bound` to the borrow of `target` rather than to
/// the interpreter lifetime, which is what the elided form gets wrong.
fn call_ufunc_method<'py>(
    target: &pyo3::Bound<'py, pyo3::PyAny>,
    method: &str,
    operand: &pyo3::Bound<'py, pyo3::PyAny>,
) -> pyo3::Bound<'py, pyo3::PyAny> {
    if method == "outer" {
        target
            .call_method1(method, (operand, operand))
            .expect("ufunc method call")
    } else {
        target
            .call_method1(method, (operand,))
            .expect("ufunc method call")
    }
}

/// Restores a mutable target from a pristine copy. Must be called BEFORE `Instant::now()`
/// so the restore is outside the timed region - that is the whole reason this group can
/// exist (`deadlock-audit-v46rn`).
fn restore_target<'py>(
    numpy: &pyo3::Bound<'py, PyModule>,
    target: &pyo3::Bound<'py, pyo3::PyAny>,
    pristine: &pyo3::Bound<'py, pyo3::PyAny>,
) {
    numpy
        .call_method1("copyto", (target, pristine))
        .expect("restore target");
}

// `ufunc.at` is the last method entry point with no row (`deadlock-audit-v46rn`). Row 41
// deliberately left it out because it MUTATES its target and returns None: under an
// interleaved ABBA schedule each arm would accumulate a different number of applications,
// the arms' checksums would diverge legitimately, and the contract would be comparing two
// STATES rather than two implementations. A green row measuring the wrong thing is worse
// than no row, and this ledger has enough of those.
//
// The fix is the restore-outside-the-timer harness row 41 asked for: each arm copies a
// pristine buffer into its own target BEFORE starting the clock, times only the `at` call,
// and checksums the mutated target afterwards. Both arms therefore start from an identical
// state every round, so their checksums MATCH by construction - which is what makes the
// contract's checksum agreement meaningful here rather than accidental.
//
// Two cells, because `at` has two regimes and one row cannot speak for both:
//   f64 n=2^8      no native path applies - measures the wrapper floor, comparable with
//                  the reduce/outer/accumulate/reduceat rows at the same size
//   int64 n=2^20   the regime `try_parallel_int_scatter_at` was written for (i64 target,
//                  order-free op, large target), so this is where a native route can engage
//
// DUPLICATE INDICES ON PURPOSE: `at` exists to apply unbuffered accumulation, so with
// repeated indices its result differs from plain fancy indexing. Indices without duplicates
// would measure a path that is not what anyone uses `at` for.
//
// THE FIRST FORMULA HERE HAD NO DUPLICATES AND THE ASSERT CAUGHT IT. `(rng * 7) % (n / 2)`
// looks like it should collide and does not: 7 is coprime with the modulus and the range
// never wraps, so all 64 indices came out distinct. The run aborted rather than publishing
// a green row for the buffered path. The strided form below takes `m = idx_n / 4` distinct
// slots and spreads them across the whole target, giving exactly 4 repetitions per slot
// while still covering 94-100% of the buffer - duplicates AND a realistic scatter.
fn bench_ufunc_at_percall_floor_vs_numpy(_c: &mut Criterion) {
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

        for (dtype, exponent, idx_exponent) in [("float64", 8u32, 6u32), ("int64", 20, 16)] {
            let n = 1usize << exponent;
            let idx_n = 1usize << idx_exponent;
            let locals = PyDict::new(py);
            locals.set_item("np", &numpy).expect("bind numpy");
            locals.set_item("n", n).expect("bind n");
            locals.set_item("idx_n", idx_n).expect("bind idx_n");
            locals.set_item("dt", dtype).expect("bind dtype");
            py.run(
                std::ffi::CString::new(
                    "pristine = np.arange(n).astype(dt)\n\
                     ours_target = pristine.copy()\n\
                     theirs_target = pristine.copy()\n\
                     rng = np.arange(idx_n)\n\
                     m = idx_n // 4\n\
                     idx = ((rng % m) * (n // m) % n).astype(np.intp)\n\
                     vals = np.arange(idx_n).astype(dt)\n",
                )
                .unwrap()
                .as_c_str(),
                Some(&locals),
                Some(&locals),
            )
            .expect("build operands");
            let pristine = locals.get_item("pristine").expect("pristine");
            let ours_target = locals.get_item("ours_target").expect("ours target");
            let theirs_target = locals.get_item("theirs_target").expect("theirs target");
            let idx = locals.get_item("idx").expect("idx");
            let vals = locals.get_item("vals").expect("vals");

            // The duplicates are the point; assert they exist rather than assuming the
            // index expression produced them.
            let unique_len = numpy
                .call_method1("unique", (&idx,))
                .expect("unique")
                .len()
                .expect("unique len");
            assert!(
                unique_len < idx_n,
                "index set has no duplicates at dtype={dtype}, so this would not measure \
                 the unbuffered path `at` exists for"
            );

            let ours = module.getattr("add").expect("fnp add");
            let theirs = numpy.getattr("add").expect("numpy add");
            assert!(
                !ours.is(&theirs),
                "fnp.add IS numpy's object - there is no candidate arm"
            );

            let checksum_of = |target: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
                target
                    .call_method0("sum")
                    .expect("target sums")
                    .extract::<f64>()
                    .expect("sum extracts as f64")
                    .to_bits()
            };

            // Parity first: same pristine state, same operation, must land identically.
            restore_target(&numpy, &ours_target, &pristine);
            restore_target(&numpy, &theirs_target, &pristine);
            ours.call_method1("at", (&ours_target, &idx, &vals))
                .expect("fnp at");
            theirs
                .call_method1("at", (&theirs_target, &idx, &vals))
                .expect("numpy at");
            assert_eq!(
                checksum_of(&ours_target),
                checksum_of(&theirs_target),
                "fnp.add.at and numpy.add.at disagree at dtype={dtype} n={n}"
            );

            let incumbent = || {
                restore_target(&numpy, &theirs_target, &pristine);
                let started = Instant::now();
                theirs
                    .call_method1("at", (&theirs_target, &idx, &vals))
                    .expect("numpy at");
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&theirs_target),
                }
            };
            let candidate = || {
                restore_target(&numpy, &ours_target, &pristine);
                let started = Instant::now();
                ours.call_method1("at", (&ours_target, &idx, &vals))
                    .expect("fnp at");
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&ours_target),
                }
            };

            let row = format!("add_at_{dtype}_n{n}_vs_numpy_method_route");
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract(&row, incumbent, candidate);
            println!(
                "UFUNC_AT_FLOOR dtype={dtype} n={n} indices={idx_n} \
                 numpy_version={numpy_version} worker={} \
                 harness=common::run_dual_null_median_ci_contract \
                 restore_is_outside_the_timed_region=true duplicate_indices=true \
                 ratio={:.6} ratio_ci95=[{:.6},{:.6}] \
                 numpy_ns={:.1} fnp_ns={:.1} excess_ns={:.1} \
                 incumbent_aa_null={:.6} candidate_aa_null={:.6}",
                measurement_worker(),
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                effect.arm_a_median_ns,
                effect.arm_b_median_ns,
                effect.arm_b_median_ns - effect.arm_a_median_ns,
                incumbent_null.ratio_median,
                candidate_null.ratio_median,
            );
        }
    });
}

/// Explicit lifetimes for the same reason `call_ufunc_method` needs them: a closure ties
/// the returned `Bound` to the borrow of `target` instead of to the interpreter.
fn call_reduceat<'py>(
    target: &pyo3::Bound<'py, pyo3::PyAny>,
    a: &pyo3::Bound<'py, pyo3::PyAny>,
    idx: &pyo3::Bound<'py, pyo3::PyAny>,
) -> pyo3::Bound<'py, pyo3::PyAny> {
    target
        .call_method1("reduceat", (a, idx))
        .expect("reduceat call")
}

// CORRECTED (`deadlock-audit-v46rn`): this group first printed
// `delegates_unconditionally=false` for reduceat, from a hardcoded literal. Reading the
// method shows it has NO native route at all - it is pure delegation, exactly like reduce
// and outer. Same defect class as the `routes_natively` literal corrected earlier in this
// file: a constant that describes runtime behaviour is a CLAIM and needs a source, not a
// guess. The banked row read the false label and called reduceat a routing method; the
// correction is that its 771 ns excess is call SHAPE, which is what made the axis-default
// lever findable.
//
// The two remaining exception-as-control-flow levers, each paired with its OWN control in
// the same group (`deadlock-audit-v46rn`). Their commits registered sharp predictions and
// neither had a cell to check them against.
//
//   clip          `np.clip(a, 0, None)` raised on every call, because float(None) raises.
//                 A TWO-SIDED scalar clip never raised, so it must NOT move. If both move,
//                 the lever is not what is being measured.
//   searchsorted  an ARRAY needle ran the whole scalar probe - ndim, dtype, kind, itemsize -
//                 and then raised. A SCALAR needle took the native path and must not move.
//
// A lever whose control moves with it is measuring the window, not the change. That is why
// each pair is here rather than in two separate groups: same binary, same run, same window.
fn bench_predecline_levers_vs_numpy(_c: &mut Criterion) {
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
        py.run(
            std::ffi::CString::new(
                "a = np.arange(4096.0)\n\
                 hay = np.arange(0.0, 8192.0, 2.0)\n\
                 needle_arr = np.array([1.0, 777.0, 8191.0])\n\
                 CASES = {\n\
                   'clip_one_sided': ('clip', (a, 0.0, None)),\n\
                   'clip_two_sided': ('clip', (a, 0.0, 3000.0)),\n\
                   'ss_array_needle': ('searchsorted', (hay, needle_arr)),\n\
                   'ss_scalar_needle': ('searchsorted', (hay, 777.0)),\n\
                 }\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&locals),
            Some(&locals),
        )
        .expect("build operands");
        let cases = locals.get_item("CASES").expect("CASES");

        for (label, takes_lever) in [
            ("clip_one_sided", true),
            ("clip_two_sided", false),
            ("ss_array_needle", true),
            ("ss_scalar_needle", false),
        ] {
            let case = cases.get_item(label).expect("case");
            let fn_name = case
                .get_item(0)
                .expect("fn name")
                .extract::<String>()
                .expect("fn name is a string");
            let args = case
                .get_item(1)
                .expect("args")
                .cast_into::<PyTuple>()
                .expect("args tuple");

            let ours = module.getattr(fn_name.as_str()).expect("fnp fn");
            let theirs = numpy.getattr(fn_name.as_str()).expect("numpy fn");
            assert!(
                !ours.is(&theirs),
                "fnp.{fn_name} IS numpy's object - there is no candidate arm"
            );

            let checksum_of = |r: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
                r.call_method0("sum")
                    .expect("sum")
                    .extract::<f64>()
                    .expect("f64 sum")
                    .to_bits()
            };
            assert_eq!(
                checksum_of(&ours.call1(&args).expect("fnp probe")),
                checksum_of(&theirs.call1(&args).expect("numpy probe")),
                "{label}: fnp and numpy disagree"
            );

            let incumbent = || {
                let started = Instant::now();
                let r = theirs.call1(&args).expect("numpy call");
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&r),
                }
            };
            let candidate = || {
                let started = Instant::now();
                let r = ours.call1(&args).expect("fnp call");
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&r),
                }
            };

            let row = format!("{label}_predecline_vs_numpy");
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract(&row, incumbent, candidate);
            println!(
                "PREDECLINE_LEVER case={label} takes_lever={takes_lever} \
                 numpy_version={numpy_version} worker={} \
                 harness=common::run_dual_null_median_ci_contract \
                 ratio={:.6} ratio_ci95=[{:.6},{:.6}] \
                 numpy_ns={:.1} fnp_ns={:.1} excess_ns={:.1} \
                 incumbent_aa_null={:.6} candidate_aa_null={:.6}",
                measurement_worker(),
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                effect.arm_a_median_ns,
                effect.arm_b_median_ns,
                effect.arm_b_median_ns - effect.arm_a_median_ns,
                incumbent_null.ratio_median,
                candidate_null.ratio_median,
            );
        }
    });
}

// The axis-default tranche has never been measured, and its own ledger row says it must be
// before anyone extrapolates to it (`deadlock-audit-v46rn`). 24 wrappers stopped forwarding
// an `axis` NumPy already defaults to; the measured instance of that lever was 2003
// instructions/call on `accumulate`/`reduceat`, which are per-op entry points, while these
// are cold wrappers whose own work may dwarf a dict entry.
//
// Two representatives, chosen because they bracket the tranche's cost range: `linspace`
// does a trivial amount of real work, so a per-call dict entry is the largest fraction of
// it available anywhere in the tranche; `fft` does a genuine transform, so it is the case
// where the lever should be invisible. If the lever is worth anything on cold wrappers it
// has to show on `linspace`; if it shows on neither, the lane is closed.
//
// n is small for the same reason the method rows are: it maximises the share a fixed
// per-call cost can occupy, which is the regime that makes a small lever detectable at all.
fn bench_axis_default_wrappers_vs_numpy(_c: &mut Criterion) {
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
        py.run(
            std::ffi::CString::new(
                "start = np.zeros(4)\n\
                 stop = np.ones(4)\n\
                 sig = np.arange(64.0)\n\
                 coef = np.arange(1.0, 9.0)\n\
                 one = np.ones(4)\n\
                 eight = np.full(4, 8.0)\n\
                 NPFN = {'linspace': np.linspace, 'fft': np.fft.fft,\n\
                         'chebder': np.polynomial.chebyshev.chebder,\n\
                         'geomspace': np.geomspace}\n\
                 ARGS = {'linspace': (start, stop, 8), 'fft': (sig,),\n\
                         'chebder': (coef,), 'geomspace': (one, eight, 4)}\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&locals),
            Some(&locals),
        )
        .expect("build operands");
        let npfn = locals.get_item("NPFN").expect("NPFN");
        let argtbl = locals.get_item("ARGS").expect("ARGS");

        // chebder and geomspace are the two cells row 52 asked for - swept wrappers that
        // are NOT linspace or fft, to test whether the ~656 ns figure transfers within the
        // family or was specific to the two cells that produced it.
        for name in ["linspace", "fft", "chebder", "geomspace"] {
            let ours = module.getattr(name).expect("fnp wrapper");
            let theirs = npfn.get_item(name).expect("numpy callable");
            assert!(
                !ours.is(&theirs),
                "fnp.{name} IS numpy's object - there is no candidate arm"
            );
            let args = argtbl
                .get_item(name)
                .expect("args")
                .cast_into::<PyTuple>()
                .expect("args tuple");

            // Both arms must agree before either is timed; `fft` returns complex, so the
            // checksum goes through `abs().sum()` rather than assuming a real result.
            let checksum_of = |r: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
                r.call_method0("__abs__")
                    .expect("abs")
                    .call_method0("sum")
                    .expect("sum")
                    .extract::<f64>()
                    .expect("f64 sum")
                    .to_bits()
            };
            assert_eq!(
                checksum_of(&ours.call1(&args).expect("fnp probe")),
                checksum_of(&theirs.call1(&args).expect("numpy probe")),
                "fnp.{name} and numpy disagree on these operands"
            );

            let incumbent = || {
                let started = Instant::now();
                let r = theirs.call1(&args).expect("numpy call");
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&r),
                }
            };
            let candidate = || {
                let started = Instant::now();
                let r = ours.call1(&args).expect("fnp call");
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&r),
                }
            };

            let row = format!("{name}_axis_default_wrapper_vs_numpy");
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract(&row, incumbent, candidate);
            println!(
                "AXIS_DEFAULT_WRAPPER fn={name} numpy_version={numpy_version} worker={} \
                 harness=common::run_dual_null_median_ci_contract \
                 ratio={:.6} ratio_ci95=[{:.6},{:.6}] \
                 numpy_ns={:.1} fnp_ns={:.1} excess_ns={:.1} \
                 incumbent_aa_null={:.6} candidate_aa_null={:.6}",
                measurement_worker(),
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                effect.arm_a_median_ns,
                effect.arm_b_median_ns,
                effect.arm_b_median_ns - effect.arm_a_median_ns,
                incumbent_null.ratio_median,
                candidate_null.ratio_median,
            );
        }
    });
}

// `reduceat` is the last ufunc METHOD entry point with a clean measurable shape and no
// row of its own (`deadlock-audit-v46rn`). The one before it, `accumulate`, turned out to
// be 4.76x SLOWER than NumPy while nobody was looking, because every per-call row in this
// campaign goes through `PyUFunc::__call__` and the methods have their own prologue and
// their own routing. Absence of a row is not evidence of health, so this adds the row.
//
// n is small on purpose, as with the other method rows: NumPy's own reduceat over 256
// elements is fast enough that a wrapper prologue is a large fraction of it, which is the
// regime where the last two defects were found. A large n would hide exactly what this is
// looking for.
//
// `at` is deliberately NOT measured here. It mutates its target in place and returns None,
// so under an interleaved ABBA schedule each arm would accumulate a different number of
// applications and the two arms' checksums would legitimately diverge - the contract would
// be comparing different states, not different implementations. It needs a harness that
// restores the target outside the timed region, and that is a separate piece of work
// rather than something to bolt on here and get quietly wrong.
fn bench_reduceat_percall_floor_vs_numpy(_c: &mut Criterion) {
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
                "i = np.arange(n)\n\
                 a = 1.0 + (i % 1000) / 1000.0\n\
                 idx = np.arange(0, n, 16, dtype=np.intp)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&locals),
            Some(&locals),
        )
        .expect("build operands");
        let a = locals.get_item("a").expect("a operand");
        let idx = locals.get_item("idx").expect("idx operand");

        let ours = module.getattr("add").expect("fnp add");
        let theirs = numpy.getattr("add").expect("numpy add");
        assert!(
            !ours.is(&theirs),
            "fnp.add IS numpy's object - there is no candidate arm"
        );

        let checksum_of = |result: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
            result
                .call_method0("sum")
                .expect("reduceat result sums")
                .extract::<f64>()
                .expect("sum is f64")
                .to_bits()
        };

        assert_eq!(
            checksum_of(&call_reduceat(&ours, &a, &idx)),
            checksum_of(&call_reduceat(&theirs, &a, &idx)),
            "fnp.add.reduceat and numpy.add.reduceat disagree on these operands"
        );

        let incumbent = || {
            let started = Instant::now();
            let result = call_reduceat(&theirs, &a, &idx);
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: checksum_of(&result),
            }
        };
        let candidate = || {
            let started = Instant::now();
            let result = call_reduceat(&ours, &a, &idx);
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: checksum_of(&result),
            }
        };

        let row = format!("add_reduceat_n{n}_vs_numpy_method_route");
        let (effect, incumbent_null, candidate_null) =
            common::run_dual_null_median_ci_contract(&row, incumbent, candidate);
        println!(
            "UFUNC_METHOD_FLOOR method=reduceat n={n} numpy_version={numpy_version} \
             worker={} harness=common::run_dual_null_median_ci_contract \
             delegates_unconditionally=true \
             ratio={:.6} ratio_ci95=[{:.6},{:.6}] \
             numpy_ns={:.1} fnp_ns={:.1} excess_ns={:.1} \
             incumbent_aa_null={:.6} candidate_aa_null={:.6}",
            measurement_worker(),
            effect.ratio_median,
            effect.ratio_ci_low,
            effect.ratio_ci_high,
            effect.arm_a_median_ns,
            effect.arm_b_median_ns,
            effect.arm_b_median_ns - effect.arm_a_median_ns,
            incumbent_null.ratio_median,
            candidate_null.ratio_median,
        );
    });
}

// `add.accumulate` on f64 routes into fnp's native cumsum dispatch with NO SIZE GATE,
// and at n=256 that is a 4.61x LOSS (numpy 1357 ns against our 6262 ns, excess 4905 ns,
// `deadlock-audit-v46rn`). Divide already carries `F64_DIV_NATIVE_MIN_LEN` for exactly
// this shape of problem - a native route that wins large and loses small - so the gate
// is the obvious remedy and this group exists to find where to put it.
//
// A crossover has to be MEASURED, not guessed: gate too low and the loss stays, gate too
// high and a real win is thrown away. Each size is its own dual-null contract so the
// crossing point is read off decidable cells rather than off a trend line.
/// Calls per accumulate counter probe. Identical across the pair so `perf stat`
/// totals from two processes compare without normalisation, and large enough that
/// the loop dominates the shared setup (`deadlock-audit-v46rn`).
const ACCUMULATE_COUNTER_CALLS: usize = 400_000;

// COUNTING what `add.accumulate` on 256 f64 still spends over NumPy
// (`deadlock-audit-v46rn`).
//
// WHY COUNT AND NOT JUST TIME. The size gate (`95ed2802`) and the probe-to-decline
// cut (`760a03e9`) were both attributed by READING the path — accurate as far as
// it went, but a read cannot say how much is left, and the residual is what
// licenses or blocks the 75-site `dtype.kind`-to-`String` sweep that `760a03e9`
// says is "not licensed until this one is measured". A wall-clock ratio at n=256
// is a few hundred nanoseconds riding on a host that moves 10% between runs;
// retired instructions per call do not move with the host at all.
//
// THE PAIR. Both probes start Python, import numpy, build the fnp module and
// construct the same operand, then call ONE callable in a fixed loop. Everything
// outside the loop is identical, so the DIFFERENCE in counted events divided by
// `ACCUMULATE_COUNTER_CALLS` is the per-call excess and nothing else. That setup
// symmetry is not decoration: the first provenance probe pair in this file
// differed by 426 M instructions purely because one of them skipped interpreter
// startup, and `perf stat` counts a PROCESS.
//
// NEGATIVE CASE: a probe that silently accumulated a DIFFERENT array, or that
// returned early, would report a flatteringly small instruction count. Both probes
// therefore checksum their last result and assert it against NumPy's answer for
// the same operand before printing, so a route that returned the wrong thing —
// or nothing — fails instead of counting fast.
fn accumulate_counter_probe(use_fnp: bool) {
    const N: usize = 256;
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let locals = PyDict::new(py);
        locals.set_item("np", &numpy).expect("bind numpy");
        locals.set_item("n", N).expect("bind n");
        py.run(
            std::ffi::CString::new("i = np.arange(n)\na = 1.0 + (i % 1000) / 1000.0\n")
                .unwrap()
                .as_c_str(),
            Some(&locals),
            Some(&locals),
        )
        .expect("build operand");
        let a = locals.get_item("a").expect("a operand");

        // Both handles are resolved in both probes so the getattr work matches.
        let ours = module.getattr("add").expect("fnp add");
        let theirs = numpy.getattr("add").expect("numpy add");
        let target = if use_fnp { &ours } else { &theirs };

        let oracle = call_ufunc_method(&theirs, "accumulate", &a)
            .call_method0("sum")
            .expect("oracle sums")
            .extract::<f64>()
            .expect("sum is f64")
            .to_bits();

        // Checksum OUTSIDE the loop — see the note on `method_counter_probe`. A
        // per-iteration `.sum()` only cancels out of the excess if it costs the
        // same on both arms, which is an assumption about the result objects and
        // not a fact about them.
        let mut last_result = None;
        for _ in 0..ACCUMULATE_COUNTER_CALLS {
            let result = call_ufunc_method(target, "accumulate", &a);
            black_box(&result);
            last_result = Some(result);
        }
        let last = last_result
            .expect("the loop ran at least once")
            .call_method0("sum")
            .expect("result sums")
            .extract::<f64>()
            .expect("sum is f64")
            .to_bits();
        assert_eq!(
            last, oracle,
            "the counted probe did not reproduce NumPy's accumulate, so its \
             instruction count describes the wrong computation"
        );
        println!(
            "ACCUMULATE_COUNTER_PROBE arm={} n={N} calls={ACCUMULATE_COUNTER_CALLS} \
             checksum={last:016x} setup_matches_sibling_probe=true \
             run_this_under_perf_stat=true",
            if use_fnp { "fnp" } else { "numpy" }
        );
    });
}

fn bench_accumulate_counter_fnp(_c: &mut Criterion) {
    accumulate_counter_probe(true);
}

fn bench_accumulate_counter_numpy(_c: &mut Criterion) {
    accumulate_counter_probe(false);
}

// TURNING THE ROUTING-PROLOGUE CORRELATION INTO A COUNTED MECHANISM
// (`deadlock-audit-v46rn`).
//
// The method family at n=256 now reads:
//     reduce      1.2800x  excess 291 ns   delegates unconditionally
//     outer       1.1804x  excess 321 ns   delegates unconditionally
//     accumulate  1.5464x  excess 767 ns   routes natively above 2^12
//     reduceat    1.6821x  excess 771 ns   routes
// The two ROUTING methods carry ~770 ns and the two pure-DELEGATING ones ~300 ns.
// `RedLynx` banked that ~450 ns gap explicitly as a CORRELATION across two cells
// and refused to call it a mechanism — "it says where to look, not what is
// there". These probes are what converts it, and they do it in retired
// INSTRUCTIONS, which do not move with host load the way a 450 ns wall-clock
// difference does.
//
// WHY THE COMPARISON IS LEGITIMATE ACROSS METHODS. Each probe pair yields
// `excess(M) = fnp_instructions(M) - numpy_instructions(M)`. Both arms of a pair
// run the identical checksum step for that method, so its cost cancels inside
// each excess even though it differs BETWEEN methods (a `reduce` returns a scalar,
// an `accumulate` an array). Comparing `excess(accumulate)` against
// `excess(reduce)` therefore compares two WRAPPER costs and not two workloads,
// which is exactly the quantity the ns table above is made of.
//
// `reduce` is the delegating control and `reduceat` the worst cell; `accumulate`
// already has its pair above. If the routing prologue is real, the two routing
// methods should carry a common block of instructions that `reduce` does not, and
// the per-symbol diff should name it.
//
// NEGATIVE CASE: a probe that returned early, or accumulated a degenerate operand,
// would report a flatteringly small instruction count. Every probe checksums its
// LAST result and asserts it against NumPy's answer for the same operand before
// printing, so a wrong or absent route fails instead of counting fast.
fn method_counter_probe(method: &str, use_fnp: bool) {
    const N: usize = 256;
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let locals = PyDict::new(py);
        locals.set_item("np", &numpy).expect("bind numpy");
        locals.set_item("n", N).expect("bind n");
        py.run(
            std::ffi::CString::new(
                "i = np.arange(n)\na = 1.0 + (i % 1000) / 1000.0\nidx = np.arange(0, n, 16, dtype=np.intp)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&locals),
            Some(&locals),
        )
        .expect("build operands");
        let a = locals.get_item("a").expect("a operand");
        let idx = locals.get_item("idx").expect("idx operand");

        // Both handles resolved in both arms so the getattr work matches.
        let ours = module.getattr("add").expect("fnp add");
        let theirs = numpy.getattr("add").expect("numpy add");
        let target = if use_fnp { &ours } else { &theirs };

        // `reduce` yields a scalar, `accumulate`/`reduceat` an array. Extract
        // covers the first, `sum()` the second; whichever runs is the same on both
        // arms of a pair, so it cancels out of the excess.
        let checksum_of = |result: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
            if let Ok(scalar) = result.extract::<f64>() {
                return scalar.to_bits();
            }
            result
                .call_method0("sum")
                .expect("result sums")
                .extract::<f64>()
                .expect("sum is f64")
                .to_bits()
        };
        // A named fn rather than a closure: the closure form cannot express that
        // the returned `Bound` borrows the same `'py` as its inputs.
        fn invoke<'py>(
            method: &str,
            t: &pyo3::Bound<'py, pyo3::PyAny>,
            a: &pyo3::Bound<'py, pyo3::PyAny>,
            idx: &pyo3::Bound<'py, pyo3::PyAny>,
        ) -> pyo3::Bound<'py, pyo3::PyAny> {
            if method == "reduceat" {
                call_reduceat(t, a, idx)
            } else {
                call_ufunc_method(t, method, a)
            }
        }

        // `outer` ON THE SAME OPERAND AS THE WALL-CLOCK ROW, and this is a defect fix
        // rather than a preference (`deadlock-audit-v46rn`). `outer` on n=256 builds a
        // 256x256 result, so a probe that passes the full operand spends ~382 000
        // instructions per call on O(n^2) arithmetic and buries the wrapper it is
        // supposed to be counting - measured at a 0.7% excess, against 30-40% for the
        // other three methods. `bench_ufunc_method_percall_floor_vs_numpy` already
        // shortens `outer`'s operand to 16 elements for exactly this reason. Matching it
        // keeps the counter row and the wall-clock row comparable; without this the two
        // describe different regimes while both saying `n=256`.
        let a = if method == "outer" {
            a.get_item(pyo3::types::PySlice::new(py, 0, 16, 1))
                .expect("short operand for outer")
        } else {
            a.clone()
        };
        let operand_len = a.len().unwrap_or(N);

        let oracle = checksum_of(&invoke(method, &theirs, &a, &idx));
        // THE CHECKSUM IS OUTSIDE THE LOOP, and that placement is load-bearing.
        // Checksumming per iteration assumed `.sum()` costs the same on both arms
        // so it would cancel out of the excess. It does not have to: the two arms
        // may hand back differently-constructed result objects even when the
        // VALUES are bit-identical, and then the excess includes a difference in
        // the checksum rather than in the call under test. The first version of
        // this probe did exactly that and reported a `reduceat` excess 2.2x below
        // the banked wall-clock row. Only the call is in the loop now.
        let mut last_result = None;
        for _ in 0..ACCUMULATE_COUNTER_CALLS {
            let result = invoke(method, target, &a, &idx);
            black_box(&result);
            last_result = Some(result);
        }
        let last = checksum_of(&last_result.expect("the loop ran at least once"));
        assert_eq!(
            last, oracle,
            "the counted probe for `{method}` did not reproduce NumPy's answer, so \
             its instruction count describes the wrong computation"
        );
        println!(
            "METHOD_COUNTER_PROBE method={method} arm={} n={N} \
             operand_len={operand_len} \
             calls={ACCUMULATE_COUNTER_CALLS} checksum={last:016x} \
             setup_matches_sibling_probe=true run_this_under_perf_stat=true",
            if use_fnp { "fnp" } else { "numpy" }
        );
    });
}

// COUNTING THE BINARY ROUTE'S SHARED FLOOR (`deadlock-audit-ei9jz`,
// `deadlock-audit-6y5wp`).
//
// The binary route at n=256 is the worst lane measured: multiply 1.4567x, add
// 1.4745x, subtract 1.4951x, divide 1.5578x, against a method family that has
// converged to 1.0989-1.2159x. Of divide's 245 ns excess, 197 ns is the floor every
// binary op pays and 48 ns is divide alone entering the f64 block only to decline.
// Both figures are wall-clock, and a static sweep found no wrapper lever left in the
// floor - so the open question is what that 197 ns IS, which is `ei9jz`'s bead.
//
// These probes count it. Retired instructions do not move with host load, which is
// what makes them usable on a machine that has swung between loadavg 8 and 525
// today, and the per-symbol diff names where the instructions go rather than
// leaving them unattributed.
//
// `add` and `divide` are both here on purpose. `add` reports
// `enters_f64_binary_block=false` and `divide` reports TRUE at this size, so the
// DIFFERENCE between their excesses is the block-entry cost measured a second way,
// in a load-independent currency, against the 48 ns the wall clock gave.
//
// NEGATIVE CASE: a probe that returned early, or that compared against the wrong
// oracle, would report a flatteringly small count. Each probe checksums its LAST
// result - outside the timed loop, because a per-iteration checksum is only free if
// it costs the same on both arms, which is an assumption about the result objects
// rather than a fact - and asserts it against NumPy's answer for the same operands.
fn binary_counter_probe(op: &str, use_fnp: bool) {
    const N: usize = 256;
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
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

        // Both handles resolved in both arms so the getattr work matches.
        let ours = module.getattr(op).expect("fnp ufunc");
        let theirs = numpy.getattr(op).expect("numpy ufunc");
        let target = if use_fnp { &ours } else { &theirs };

        let checksum_of = |r: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
            r.call_method0("sum")
                .expect("result sums")
                .extract::<f64>()
                .expect("sum is f64")
                .to_bits()
        };
        let oracle = checksum_of(&theirs.call1((&a, &b)).expect("oracle call"));

        let mut last = None;
        for _ in 0..ACCUMULATE_COUNTER_CALLS {
            let r = target.call1((&a, &b)).expect("binary call");
            black_box(&r);
            last = Some(r);
        }
        let last = checksum_of(&last.expect("the loop ran at least once"));
        assert_eq!(
            last, oracle,
            "the binary counter probe for `{op}` did not reproduce NumPy's answer, so \
             its instruction count describes the wrong computation"
        );
        println!(
            "BINARY_COUNTER_PROBE op={op} arm={} n={N} calls={ACCUMULATE_COUNTER_CALLS} \
             checksum={last:016x} setup_matches_sibling_probe=true \
             run_this_under_perf_stat=true",
            if use_fnp { "fnp" } else { "numpy" }
        );
    });
}

fn bench_binary_counter_add_fnp(_c: &mut Criterion) {
    binary_counter_probe("add", true);
}

fn bench_binary_counter_add_numpy(_c: &mut Criterion) {
    binary_counter_probe("add", false);
}

fn bench_binary_counter_divide_fnp(_c: &mut Criterion) {
    binary_counter_probe("divide", true);
}

fn bench_binary_counter_divide_numpy(_c: &mut Criterion) {
    binary_counter_probe("divide", false);
}

fn bench_reduce_counter_fnp(_c: &mut Criterion) {
    method_counter_probe("reduce", true);
}

fn bench_reduce_counter_numpy(_c: &mut Criterion) {
    method_counter_probe("reduce", false);
}

fn bench_outer_counter_fnp(_c: &mut Criterion) {
    method_counter_probe("outer", true);
}

fn bench_outer_counter_numpy(_c: &mut Criterion) {
    method_counter_probe("outer", false);
}

fn bench_reduceat_counter_fnp(_c: &mut Criterion) {
    method_counter_probe("reduceat", true);
}

fn bench_reduceat_counter_numpy(_c: &mut Criterion) {
    method_counter_probe("reduceat", false);
}

fn bench_accumulate_size_crossover_vs_numpy(_c: &mut Criterion) {
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
        let ours = module.getattr("add").expect("fnp add");
        let theirs = numpy.getattr("add").expect("numpy add");

        for exponent in [8u32, 10, 12, 14, 16, 18, 20] {
            let n = 1usize << exponent;
            let locals = PyDict::new(py);
            locals.set_item("np", &numpy).expect("bind numpy");
            locals.set_item("n", n).expect("bind n");
            py.run(
                std::ffi::CString::new("i = np.arange(n)\na = 1.0 + (i % 1000) / 1000.0\n")
                    .unwrap()
                    .as_c_str(),
                Some(&locals),
                Some(&locals),
            )
            .expect("build operand");
            let a = locals.get_item("a").expect("a operand");

            let checksum_of = |result: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
                result
                    .call_method0("sum")
                    .expect("accumulate result sums")
                    .extract::<f64>()
                    .expect("sum is f64")
                    .to_bits()
            };
            assert_eq!(
                checksum_of(&call_ufunc_method(&ours, "accumulate", &a)),
                checksum_of(&call_ufunc_method(&theirs, "accumulate", &a)),
                "fnp and numpy add.accumulate disagree at n={n}"
            );

            let incumbent = || {
                let started = Instant::now();
                let result = call_ufunc_method(&theirs, "accumulate", &a);
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let candidate = || {
                let started = Instant::now();
                let result = call_ufunc_method(&ours, "accumulate", &a);
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };

            let row = format!("add_accumulate_n{n}_vs_numpy_crossover");
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract(&row, incumbent, candidate);
            println!(
                "ACCUMULATE_CROSSOVER n={n} exponent={exponent} numpy_version={numpy_version} \
                 worker={} harness=common::run_dual_null_median_ci_contract \
                 ratio={:.6} ratio_ci95=[{:.6},{:.6}] \
                 numpy_ns={:.1} fnp_ns={:.1} excess_ns={:.1} \
                 incumbent_aa_null={:.6} candidate_aa_null={:.6}",
                measurement_worker(),
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                effect.arm_a_median_ns,
                effect.arm_b_median_ns,
                effect.arm_b_median_ns - effect.arm_a_median_ns,
                incumbent_null.ratio_median,
                candidate_null.ratio_median,
            );
        }
    });
}

// The per-call floor of the ufunc METHOD path, which `bench_percall_floor_across_ops_vs_numpy`
// does NOT reach: that group calls `fnp.add(a, b)`, i.e. `PyUFunc::__call__`, while
// `reduce`/`accumulate`/`outer`/`reduceat`/`at` are separate `#[pymethods]` with their own
// prologue. `__call__` was converted to the cached numpy module handle when the import was
// priced at 310 ns per call; the methods were not converted until `deadlock-audit-v46rn`, so
// this group exists to measure that conversion on the path that actually changed.
//
// `reduce` is the right method to measure it on: it delegates UNCONDITIONALLY (no native fast
// path can absorb the call), so what is timed is the wrapper prologue plus NumPy's own work,
// which is exactly the quantity the lever moves. n is deliberately small - at 256 elements
// NumPy's reduce is a few hundred ns, so a 310 ns prologue is a large fraction and visible;
// at 2^20 it would be swamped by the reduction itself and the row would say nothing.
fn bench_ufunc_method_percall_floor_vs_numpy(_c: &mut Criterion) {
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
            std::ffi::CString::new("i = np.arange(n)\na = 1.0 + (i % 1000) / 1000.0\n")
                .unwrap()
                .as_c_str(),
            Some(&locals),
            Some(&locals),
        )
        .expect("build operand");
        let a = locals.get_item("a").expect("a operand");

        // `accumulate` took the same lever in the same commit and row38 flagged it as
        // UNMEASURED: it has a native fast path that may absorb the call, so it needs its
        // own arm rather than inheriting reduce's figure. Same call shape as reduce.
        for method in ["reduce", "accumulate", "outer"] {
            let ours = module.getattr("add").expect("fnp add");
            let theirs = numpy.getattr("add").expect("numpy add");
            assert!(
                !ours.is(&theirs),
                "fnp.add IS numpy's object - there is no candidate arm"
            );

            // `outer` on n=256 builds a 256x256 result, so it is timed on a SHORTER
            // operand to keep the call-shape cost visible rather than the O(n^2) work.
            let operand = if method == "outer" {
                a.get_item(pyo3::types::PySlice::new(py, 0, 16, 1))
                    .expect("short operand")
            } else {
                a.clone()
            };

            // A scalar (reduce) and an array (outer) need different checksums; both must
            // read the RESULT, or the timer could be measuring a call that returned nothing.
            let checksum_of = |result: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
                match result.extract::<f64>() {
                    Ok(scalar) => scalar.to_bits(),
                    Err(_) => result
                        .call_method0("sum")
                        .expect("array result sums")
                        .extract::<f64>()
                        .expect("sum is f64")
                        .to_bits(),
                }
            };

            assert_eq!(
                checksum_of(&call_ufunc_method(&ours, method, &operand)),
                checksum_of(&call_ufunc_method(&theirs, method, &operand)),
                "fnp.add.{method} and numpy.add.{method} disagree"
            );

            let incumbent = || {
                let started = Instant::now();
                let result = call_ufunc_method(&theirs, method, &operand);
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let candidate = || {
                let started = Instant::now();
                let result = call_ufunc_method(&ours, method, &operand);
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };

            let row = format!("add_{method}_n{n}_vs_numpy_method_route");
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract(&row, incumbent, candidate);
            let numpy_ns = effect.arm_a_median_ns;
            let fnp_ns = effect.arm_b_median_ns;
            println!(
                "UFUNC_METHOD_FLOOR method={method} n={n} numpy_version={numpy_version} \
                 worker={} harness=common::run_dual_null_median_ci_contract \
                 delegates_unconditionally={} \
                 ratio={:.6} ratio_ci95=[{:.6},{:.6}] \
                 numpy_ns={numpy_ns:.1} fnp_ns={fnp_ns:.1} excess_ns={:.1} \
                 incumbent_aa_null={:.6} candidate_aa_null={:.6}",
                measurement_worker(),
                method == "reduce" || method == "outer",
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                fnp_ns - numpy_ns,
                incumbent_null.ratio_median,
                candidate_null.ratio_median,
            );
        }
    });
}

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

        // CORRECTED (`deadlock-audit-ei9jz`): this loop used to hardcode
        // `routes_natively=true` for divide, and that label was FALSE at this size.
        // The native f64 route needs n >= F64_DIV_NATIVE_MIN_LEN (1<<14 = 16384) and
        // this group runs at n = 1<<8 = 256, so ALL FOUR ops delegate to NumPy here.
        // A row that read the old label concluded divide gained least from the
        // interning levers "because it takes the native route"; the true reason is
        // below and has the opposite shape.
        //
        // What actually separates divide from the other three is which ops ENTER the
        // f64 binary block at all: `PyUFunc::__call__` maps only Remainder, Power,
        // Maximum, Minimum and Divide to `Some(BinaryOp)`; add/subtract/multiply hit
        // the `_ => None` arm and skip the block entirely. So at n=256 divide alone
        // pays the block's dtype probes and THEN declines at the size gate - which is
        // why its excess was double multiply's while every op delegated.
        //
        // MIRRORED CONSTANT: F64_DIV_NATIVE_MIN_LEN is private to lib.rs (defined at
        // its `const F64_DIV_NATIVE_MIN_LEN: usize = 1 << 14`). If it changes without
        // this mirror changing, the LABEL goes stale, not the measurement - the lib
        // test `f64_div_native_min_len_is_mirrored_in_the_percall_floor_bench` fails
        // and names this line.
        const NATIVE_MIN_LEN_MIRROR: usize = 1 << 19;
        for (name, enters_f64_block) in [
            ("add", false),
            ("subtract", false),
            ("multiply", false),
            ("divide", true),
        ] {
            let routes_natively = enters_f64_block && n >= NATIVE_MIN_LEN_MIRROR;
            let (ratio, lo, hi, numpy_ns, fnp_ns) =
                measure_binary_ufunc_vs_numpy(py, &module, &numpy, name, &args, n);
            println!(
                "PERCALL_FLOOR op={name} n={n} routes_natively={routes_natively} \
                 enters_f64_binary_block={enters_f64_block} \
                 native_route_min_len={NATIVE_MIN_LEN_MIRROR} \
                 numpy_version={numpy_version} \
                 harness=common::run_dual_null_median_ci_contract \
                 ratio={ratio:.6} ratio_ci95=[{lo:.6},{hi:.6}] \
                 numpy_ns={numpy_ns:.1} fnp_ns={fnp_ns:.1} excess_ns={:.1}",
                fnp_ns - numpy_ns,
            );
        }
    });
}

// Straddles F64_DIV_NATIVE_MIN_LEN (1<<14) so ONE invocation shows both sides of
// the gate: 2^8 and 2^12 now DELEGATE to NumPy, 2^16 and 2^20 still take the
// native route (deadlock-audit-qapyb). A gate that helps below the threshold and
// hurts above it is a REJECT, and only measuring both sides can say which it is.
fn bench_divide_size_gate_vs_numpy(_c: &mut Criterion) {
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

        // Three sizes spanning the 1<<14 gate, each with its delegating control:
        // 2^8 delegates, 2^16 and 2^20 route natively. The earlier note here said a
        // four-cell sweep did not fit rch's 1800 s SSH ceiling - that constraint is on
        // REMOTE runs; these are built and measured locally, where six contracts fit.
        //
        // LOCATING THE CROSSOVER (`deadlock-audit-q00ev`, `deadlock-audit-6y5wp`).
        //
        // Already settled: at 2^16 the native route costs several times its delegating
        // control's excess, and at 2^20 it BEATS that control by ~10.7 us. So the route
        // is worth keeping and `F64_DIV_NATIVE_MIN_LEN = 1<<14` is set below the
        // crossover - but the crossover itself is only bracketed as (2^16, 2^20].
        //
        // 2^17, 2^18 and 2^19 are the unmeasured interior. Measuring them gives the gate
        // a MEASURED value instead of a bracket, which is the whole reason the current
        // 1<<14 became questionable: it was chosen without a control at the sizes it
        // admits.
        //
        // 2^8 stays as the FIXTURE GUARD, not for its own sake: there both ops delegate,
        // so their excesses must agree to within the ~48 ns block entry divide alone
        // pays. If that guard fails the pair is not comparable and no size above it can
        // be read. 2^16 and 2^20 are omitted here only because they are already banked.
        for exponent in [8u32, 17, 18, 19] {
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

            // MULTIPLY IS THE DELEGATING CONTROL, AND IT IS THE WHOLE POINT OF THIS
            // ADDITION (`deadlock-audit-q00ev`, `deadlock-audit-6y5wp`).
            //
            // `q00ev` found the native divide path at its worst just above its own gate
            // - 1.286x at 2^16 against 1.091x at 2^20 - and asked whether the gate is
            // simply wrong, i.e. whether DELEGATING at 2^16 would beat routing. It could
            // not answer that, and said so precisely: the arithmetic needs the wrapper's
            // cost AT 2^16, and subtracting the 938 ns measured at 2^8 would repeat a
            // cross-size subtraction error that once overstated a row by 5.5x. The
            // wrapper cost is not constant in n - `multiply`'s excess grew 6.2x between
            // 2^8 and 2^20.
            //
            // So the control has to be MEASURED at the same size, not extrapolated to it.
            // `multiply` delegates at every size (it hits the `_ => None` arm and never
            // enters the f64 block), so at 2^16 it is exactly "what divide would cost if
            // it delegated", up to the block-entry cost measured separately at ~48 ns.
            // Running both in ONE invocation is what makes the difference readable: the
            // two arms see the same host, the same window and the same operands.
            //
            // THE FIXTURE GUARD: at 2^8 BOTH ops delegate, so their excesses must agree
            // to within roughly that block-entry cost. If they do not, the pair is not
            // comparable and the 2^16 contrast means nothing - so the 2^8 row is not
            // decoration, it is the control's control.
            for op in ["divide", "multiply"] {
                let (ratio, lo, hi, numpy_ns, fnp_ns) =
                    measure_binary_ufunc_vs_numpy(py, &module, &numpy, op, &args, n);
                let routes_natively = op == "divide" && n >= (1usize << 19);
                println!(
                    "DIVIDE_SIZE_GATE op={op} n={n} log2n={exponent} \
                     routes_natively={routes_natively} \
                     delegates_under_gate={} is_delegating_control={} \
                     numpy_version={numpy_version} \
                     harness=common::run_dual_null_median_ci_contract \
                     ratio={ratio:.6} ratio_ci95=[{lo:.6},{hi:.6}] \
                     numpy_ns={numpy_ns:.1} fnp_ns={fnp_ns:.1} excess_ns={:.1}",
                    n < (1usize << 19),
                    op == "multiply",
                    fnp_ns - numpy_ns,
                );
            }
        }
    });
}

// `remainder` above its parallel threshold — the op the f64 fast-path set exists
// for. NumPy runs floored-mod single-threaded and it is compute-heavy (fmod +
// floor + correction per element); our route parallelizes it.
//
// n=1<<21 is load-bearing, not arbitrary: `parallel_min` for Remainder is exactly
// 1<<21, so at any smaller size the native path runs SERIAL and the parallel win
// cannot appear at all. Every earlier elementwise row in this ledger measured at
// 2^20 or below, i.e. strictly under the threshold (deadlock-audit-322j4).
//
// Unlike divide, remainder carries no per-element FE-hazard scan here: it defers
// wholesale on a zero divisor, and these operands never produce one
// (b = 1.25 + (i%997)/997).
fn bench_remainder_vs_numpy_incumbent(_c: &mut Criterion) {
    const N: usize = 1 << 21;
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

        // The incumbent's own artifact, hashed from inside the measuring process.
        // The ledger's incumbent-win contract requires a sha256 that identifies
        // NumPy and is distinct from our ELF, so a candidate binary cannot stand
        // in for the incumbent it claims to have beaten.
        let numpy_artifact = numpy
            .getattr("core")
            .and_then(|core| core.getattr("_multiarray_umath"))
            .and_then(|module| module.getattr("__file__"))
            .and_then(|path| path.extract::<String>())
            .ok()
            .and_then(|path| std::fs::read(&path).ok())
            .map(|bytes| {
                let mut hasher = Sha256::new();
                hasher.update(&bytes);
                let digest = hasher.finalize();
                let mut hex = String::with_capacity(digest.len() * 2);
                for byte in digest {
                    write!(&mut hex, "{byte:02x}").expect("writing to a String cannot fail");
                }
                hex
            })
            .unwrap_or_else(|| "unavailable".to_string());

        let (ratio, lo, hi, numpy_ns, fnp_ns) =
            measure_binary_ufunc_vs_numpy(py, &module, &numpy, "remainder", &args, N);
        println!(
            "REMAINDER_INCUMBENT_ARM name=NumPy version={numpy_version} \
             artifact_sha256={numpy_artifact} measured_ratio={ratio:.6}x"
        );
        println!(
            "REMAINDER_VS_NUMPY n={N} log2n=21 above_parallel_threshold=true \
             numpy_version={numpy_version} \
             harness=common::run_dual_null_median_ci_contract \
             ratio={ratio:.6} ratio_ci95=[{lo:.6},{hi:.6}] \
             numpy_ns={numpy_ns:.1} fnp_ns={fnp_ns:.1} \
             faster_than_numpy={} worst_bound={:.6}",
            lo > 1.0,
            lo,
        );
    });
}

// Isolates the probe chain a DELEGATING binary ufunc runs before it gives up and
// hands the call to NumPy (deadlock-audit-wsd7h).
//
// The control already exists in the shipped code: the entire probe block in
// `PyUFunc::__call__` sits behind a guard requiring out/where/dtype/signature to
// be None, `casting == "same_kind"`, `order == "K"` and `subok`. Passing any
// other casting value skips EVERY probe and falls straight to the delegation
// tail. So:
//
//   arm A  fnp.multiply(a, b)                     full probe chain, then delegate
//   arm B  fnp.multiply(a, b, casting="unsafe")   no probes, delegate directly
//
// Both enter the same PyO3 `__call__` with the same nine parameters and leave
// through the same delegation tail; the difference is the probe chain and nothing
// else. Deliberately CONSERVATIVE: arm B additionally pays for a non-empty kwargs
// dict on the delegation, measured elsewhere at 727 ns, so the comparison is
// biased AGAINST finding probe cost.
//
// Ratio is A/B, so ABOVE 1.0 means the probes cost real time. Results are
// identical because casting only governs coercion and both operands are already
// float64, so the contract's cross-arm checksum assertion holds throughout.
fn bench_delegating_probe_chain_cost(_c: &mut Criterion) {
    const N: usize = 1 << 8;
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
            "fnp.multiply IS numpy's object — there is no candidate arm here"
        );
        let skip_probes = PyDict::new(py);
        skip_probes
            .set_item("casting", "unsafe")
            .expect("bind casting");

        // Both arms must agree bit for bit before either is timed, or the ratio
        // is between two different computations.
        let probed_probe = ours.call1(&args).expect("probed probe");
        let skipped_probe = ours.call(&args, Some(&skip_probes)).expect("skipped probe");
        assert_eq!(
            numpy_divide_checksum(&probed_probe, N),
            numpy_divide_checksum(&skipped_probe, N),
            "the casting override changed the result — it must only change the ROUTE"
        );

        let with_probes = || {
            let started = Instant::now();
            let result = ours.call1(&args).expect("fnp.multiply");
            let elapsed = started.elapsed();
            let checksum = numpy_divide_checksum(&result, N);
            common::ContractObservation { elapsed, checksum }
        };
        let without_probes = || {
            let started = Instant::now();
            let result = ours
                .call(&args, Some(&skip_probes))
                .expect("fnp.multiply casting=unsafe");
            let elapsed = started.elapsed();
            let checksum = numpy_divide_checksum(&result, N);
            common::ContractObservation { elapsed, checksum }
        };

        let (effect, probed_null, skipped_null) = common::run_dual_null_median_ci_contract(
            "delegating_probe_chain_n256",
            with_probes,
            without_probes,
        );
        println!(
            "DELEGATING_PROBE_CHAIN n={N} numpy_version={numpy_version} \
             harness=common::run_dual_null_median_ci_contract \
             arms=fnp_multiply_probed_over_fnp_multiply_casting_unsafe \
             ratio={:.6} ratio_ci95=[{:.6},{:.6}] \
             probed_ns={:.1} skipped_ns={:.1} probe_chain_ns={:.1} \
             probed_null={:.6} skipped_null={:.6} probes_cost_real_time={} \
             worst_bound={:.6}",
            effect.ratio_median,
            effect.ratio_ci_low,
            effect.ratio_ci_high,
            effect.arm_a_median_ns,
            effect.arm_b_median_ns,
            effect.arm_a_median_ns - effect.arm_b_median_ns,
            probed_null.ratio_median,
            skipped_null.ratio_median,
            effect.ratio_ci_low > 1.0,
            effect.ratio_ci_low,
        );
    });
}

// Replica of the complex probe's DECLINE path in its LEGACY order: prove the
// operand is an exact ndarray first, then look at the dtype. Returns the decision
// so the caller can fold it into a checksum.
#[inline(never)]
fn probe_decline_legacy_order(
    numpy: &pyo3::Bound<'_, PyModule>,
    a: &pyo3::Bound<'_, pyo3::PyAny>,
    b: &pyo3::Bound<'_, pyo3::PyAny>,
) -> bool {
    let Ok(ndarray_type) = numpy.getattr("ndarray") else {
        return false;
    };
    if !a.is_exact_instance(&ndarray_type) || !b.is_exact_instance(&ndarray_type) {
        return false;
    }
    let (Ok(dta), Ok(dtb)) = (a.getattr("dtype"), b.getattr("dtype")) else {
        return false;
    };
    dta.getattr("kind")
        .and_then(|kind| kind.extract::<char>())
        .is_ok_and(|kind| kind == 'c')
        && dtb
            .getattr("kind")
            .and_then(|kind| kind.extract::<char>())
            .is_ok_and(|kind| kind == 'c')
}

// The same decision in the SHIPPED order: decline on the discriminating dtype
// first, and only reach the ndarray type check for an operand that is actually
// complex.
#[inline(never)]
fn probe_decline_dtype_first(
    numpy: &pyo3::Bound<'_, PyModule>,
    a: &pyo3::Bound<'_, pyo3::PyAny>,
    b: &pyo3::Bound<'_, pyo3::PyAny>,
) -> bool {
    let (Ok(dta), Ok(dtb)) = (a.getattr("dtype"), b.getattr("dtype")) else {
        return false;
    };
    let complex_kinds = dta
        .getattr("kind")
        .and_then(|kind| kind.extract::<char>())
        .is_ok_and(|kind| kind == 'c')
        && dtb
            .getattr("kind")
            .and_then(|kind| kind.extract::<char>())
            .is_ok_and(|kind| kind == 'c');
    if !complex_kinds {
        return false;
    }
    let Ok(ndarray_type) = numpy.getattr("ndarray") else {
        return false;
    };
    a.is_exact_instance(&ndarray_type) && b.is_exact_instance(&ndarray_type)
}

// Both orderings of the complex probe's decline path, interleaved in ONE binary
// under the dual-null contract (deadlock-audit-bpxn6).
//
// This exists because measuring the same question as a cross-build pair produced
// a false positive that looked entirely healthy: on vmi1227854 the reorder read
// 1.403108 -> 1.333774 with disjoint CIs and an invariant control arm, and the
// replication on vmi1152480 REVERSED the sign (1.396597 -> 1.428706, also
// disjoint). Both cannot be true of the code, so the effect is smaller than the
// harness's cross-worker spread and only a same-binary A/B can see it
// (deadlock-audit-80uph).
//
// SCOPE: these are replicas of the decline path, not the shipped probe. They
// measure what the sequence of Python operations costs — the same limitation
// jw7vk recorded for the divide kernel replicas. That is the question here.
//
// Ratio is legacy/dtype_first, so ABOVE 1.0 means declining on the dtype first is
// cheaper. The DECISION is folded into the checksum, so an ordering that decided
// differently would fail the contract rather than win it.
fn bench_probe_decline_ordering(_c: &mut Criterion) {
    const N: usize = 1 << 8;
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
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

        // Both must reach the SAME decision on these f64 operands (decline), or the
        // two orderings are not the same predicate and the ratio is meaningless.
        assert!(
            !probe_decline_legacy_order(&numpy, &a, &b),
            "f64 operands must decline the complex probe in the legacy order"
        );
        assert!(
            !probe_decline_dtype_first(&numpy, &a, &b),
            "f64 operands must decline the complex probe in the dtype-first order"
        );

        let legacy = || {
            let started = Instant::now();
            let admitted = probe_decline_legacy_order(&numpy, &a, &b);
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: u64::from(admitted),
            }
        };
        let dtype_first = || {
            let started = Instant::now();
            let admitted = probe_decline_dtype_first(&numpy, &a, &b);
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: u64::from(admitted),
            }
        };

        let (effect, legacy_null, dtype_first_null) = common::run_dual_null_median_ci_contract(
            "probe_decline_ordering_n256",
            legacy,
            dtype_first,
        );
        println!(
            "PROBE_DECLINE_ORDERING n={N} numpy_version={numpy_version} \
             harness=common::run_dual_null_median_ci_contract \
             arms=legacy_ndarray_first_over_dtype_kind_first same_binary=true \
             ratio={:.6} ratio_ci95=[{:.6},{:.6}] \
             legacy_ns={:.1} dtype_first_ns={:.1} saved_ns={:.1} \
             legacy_null={:.6} dtype_first_null={:.6} \
             dtype_first_is_cheaper={} worst_bound={:.6}",
            effect.ratio_median,
            effect.ratio_ci_low,
            effect.ratio_ci_high,
            effect.arm_a_median_ns,
            effect.arm_b_median_ns,
            effect.arm_a_median_ns - effect.arm_b_median_ns,
            legacy_null.ratio_median,
            dtype_first_null.ratio_median,
            effect.ratio_ci_low > 1.0,
            effect.ratio_ci_low,
        );
    });
}

/// Exact replica of `BinaryOp::Maximum` (crates/fnp-ufunc/src/lib.rs:854): NaN on
/// either side propagates, equal operands return the RIGHT one, otherwise `max`.
/// The equal-operands arm is not cosmetic — it is what makes -0.0/+0.0 agree with
/// NumPy, and a replica using bare `f64::max` would diverge there.
#[inline(always)]
fn maximum_elem(lhs: f64, rhs: f64) -> f64 {
    if lhs.is_nan() || rhs.is_nan() {
        f64::NAN
    } else if lhs == rhs {
        rhs
    } else {
        lhs.max(rhs)
    }
}

#[inline(never)]
fn maximum_serial(a: &[f64], b: &[f64], out: &mut [f64]) {
    for ((slot, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *slot = maximum_elem(x, y);
    }
}

/// Same chunking the shipped parallel arm uses.
#[inline(never)]
fn maximum_parallel(a: &[f64], b: &[f64], out: &mut [f64]) {
    let chunk = out.len().div_ceil(rayon::current_num_threads());
    out.par_chunks_mut(chunk)
        .zip(a.par_chunks(chunk))
        .zip(b.par_chunks(chunk))
        .for_each(|((o, l), r)| {
            for ((slot, &x), &y) in o.iter_mut().zip(l.iter()).zip(r.iter()) {
                *slot = maximum_elem(x, y);
            }
        });
}

// deadlock-audit-hzl1w asks whether the native parallel arm should decline for
// maximum/minimum. Its evidence compares the PARALLEL arm to NumPy and shows it
// losing, which argues for delegating — but it never measured the SERIAL native
// arm, so "decline the parallel arm" (keep native, drop rayon) and "delegate to
// NumPy" are two different changes and only one of them has data.
//
// This measures BOTH against NumPy in ONE invocation, so the three-way comparison
// is licensed: parallel-native vs NumPy and serial-native vs NumPy, same binary,
// same operands, same host, each under the dual-null contract. Ratio is
// numpy/candidate, so ABOVE 1.0 means our arm is faster.
//
// n=1<<22 sits above the 1<<21 parallel_min, the regime the losing measurements
// came from — below it the shipped route runs serially anyway.
/// Which logical CPU is this thread on right now? (`deadlock-audit-48by6`)
///
/// Two projects found broken arm placement on this box — one had BOTH arms pinned to a
/// single physical core, another voided rows over contention — so placement is checked
/// here rather than assumed. `/proc/self/stat` field 39 is the last CPU the task ran on;
/// the `comm` field can contain spaces and parentheses, so parsing starts after the LAST
/// `)`.
///
/// SMT is active on this host: `thread_siblings_list` for cpu0 is `0,32`, so logical CPUs
/// N and N+32 are siblings on 32 physical cores, and two arms on sibling CPUs share
/// execution resources while reporting different CPU ids.
fn current_cpu() -> i32 {
    match std::fs::read_to_string("/proc/self/stat") {
        Ok(text) => match text.rfind(')') {
            Some(close) => text[close + 2..]
                .split_whitespace()
                .nth(36)
                .and_then(|field| field.parse().ok())
                .unwrap_or(-1),
            None => -1,
        },
        Err(_) => -1,
    }
}

/// Physical core backing a logical CPU, so SMT siblings can be told apart from distinct cores.
fn core_id_of(cpu: i32) -> i32 {
    if cpu < 0 {
        return -1;
    }
    std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/topology/core_id"))
        .ok()
        .and_then(|text| text.trim().parse().ok())
        .unwrap_or(-1)
}

/// Most frequent CPU in a sample, and how many distinct CPUs the arm touched. A high
/// distinct count means the thread MIGRATED during the arm, which on a box with a 2.8x
/// cross-core spread is a measurement hazard in itself.
fn modal_cpu(samples: &[i32]) -> (i32, usize) {
    if samples.is_empty() {
        return (-1, 0);
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let distinct = {
        let mut d = sorted.clone();
        d.dedup();
        d.len()
    };
    let mut best = sorted[0];
    let mut best_run = 0usize;
    let mut cur = sorted[0];
    let mut run = 0usize;
    for &c in &sorted {
        if c == cur {
            run += 1;
        } else {
            if run > best_run {
                best_run = run;
                best = cur;
            }
            cur = c;
            run = 1;
        }
    }
    if run > best_run {
        best = cur;
    }
    (best, distinct)
}

/// Sample every core's current frequency, in MHz (`deadlock-audit-48by6`).
///
/// This host runs `amd-pstate-epp` under the `powersave` governor and shows a LIVE
/// CROSS-CORE SPREAD: 2565-4092 MHz measured simultaneously across 64 cores, a 1.595x
/// range, at a moment when the load average was 42. A fleet report put the spread as high
/// as 2.879x. Two arms of a paired benchmark can therefore run at materially different
/// clocks without either the load average or the A/A null showing anything — the null
/// compares an arm against ITSELF and cancels a difference that is common to both of its
/// halves.
///
/// Returns `(max, mean)` across all cores. Sampled OUTSIDE the timed region: 64 sysfs
/// reads cost tens of microseconds and would otherwise be charged to the arm.
fn sample_cpu_mhz() -> (f64, f64) {
    let mut seen = Vec::with_capacity(64);
    for cpu in 0..1024u32 {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq");
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                if let Ok(khz) = text.trim().parse::<f64>() {
                    seen.push(khz / 1000.0);
                }
            }
            // cpufreq absent for this index: the enumeration is done.
            Err(_) => break,
        }
    }
    if seen.is_empty() {
        return (0.0, 0.0);
    }
    let max = seen.iter().copied().fold(f64::MIN, f64::max);
    let mean = seen.iter().sum::<f64>() / seen.len() as f64;
    (max, mean)
}

/// Median of a sample, for the per-arm MHz columns.
fn median_of(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in MHz samples"));
    values[values.len() / 2]
}

/// Does the interference scale as a FIXED cost divided by the incumbent's duration?
/// (`deadlock-audit-48by6`)
///
/// The previous row found the ABSOLUTE disturbance nearly equal across two incumbents whose
/// durations differ sevenfold — 124700 ns against 97431 ns — and proposed that the discount
/// is simply `fixed_cost / incumbent_duration`. That replaced two earlier accounts of mine
/// that both appealed to a property of the ARMS and both failed.
///
/// This tests the replacement directly, and the design matters: **the SHADOW is pinned at
/// 2^22 for every cell** so the disturbance it inflicts is constant by construction, while
/// the INCUMBENT's size sweeps 2^20 / 2^22 / 2^24. A naive sweep that grew both together
/// would confound the disturbance with the duration and could not decide anything.
///
/// PRE-REGISTERED PREDICTION: `interference_ns` stays roughly constant across the three
/// cells while `ratio` falls monotonically as the incumbent gets slower. If instead
/// `interference_ns` grows with the incumbent's size, the cost is proportional rather than
/// fixed and the 1/duration model is wrong too.
fn bench_interference_vs_incumbent_duration(_c: &mut Criterion) {
    const SHADOW_N: usize = 1 << 22;
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let numpy = py.import("numpy").expect("numpy oracle");
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy.__version__")
            .extract::<String>()
            .expect("numpy version is a string");
        let np_maximum = numpy.getattr("maximum").expect("numpy.maximum");

        // The shadow, pinned at 2^22 for every cell below.
        let shadow_locals = PyDict::new(py);
        shadow_locals.set_item("np", &numpy).expect("bind numpy");
        shadow_locals.set_item("n", SHADOW_N).expect("bind n");
        py.run(
            std::ffi::CString::new(
                "i = np.arange(n)\nsa = 1.0 + (i % 1000) / 1000.0\nsb = 1.25 + (i % 997) / 997.0\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&shadow_locals),
            Some(&shadow_locals),
        )
        .expect("build shadow operands");
        let sa: Vec<f64> = shadow_locals
            .get_item("sa")
            .expect("sa")
            .call_method0("tolist")
            .expect("sa list")
            .extract()
            .expect("sa f64s");
        let sb: Vec<f64> = shadow_locals
            .get_item("sb")
            .expect("sb")
            .call_method0("tolist")
            .expect("sb list")
            .extract()
            .expect("sb f64s");
        let mut shadow_out = vec![0.0_f64; SHADOW_N];

        for exponent in [20u32, 22, 24] {
            let n = 1usize << exponent;
            let locals = PyDict::new(py);
            locals.set_item("np", &numpy).expect("bind numpy");
            locals.set_item("n", n).expect("bind n");
            py.run(
                std::ffi::CString::new("i = np.arange(n)\na = 1.0 + (i % 1000) / 1000.0\nb = 1.25 + (i % 997) / 997.0\n")
                    .unwrap()
                    .as_c_str(),
                Some(&locals),
                Some(&locals),
            )
            .expect("build incumbent operands");
            let a_obj = locals.get_item("a").expect("a operand");
            let b_obj = locals.get_item("b").expect("b operand");
            let args = PyTuple::new(py, [&a_obj, &b_obj]).expect("args");
            let np_out = numpy
                .call_method1("empty_like", (&a_obj,))
                .expect("preallocated numpy output");
            let out_kwargs = PyDict::new(py);
            out_kwargs.set_item("out", &np_out).expect("bind out=");

            let isolated = || {
                let started = Instant::now();
                let result = np_maximum
                    .call(&args, Some(&out_kwargs))
                    .expect("numpy.maximum isolated");
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: numpy_divide_checksum(&result, n),
                }
            };
            let shadowed = || {
                maximum_parallel(&sa, &sb, &mut shadow_out);
                let started = Instant::now();
                let result = np_maximum
                    .call(&args, Some(&out_kwargs))
                    .expect("numpy.maximum shadowed");
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: numpy_divide_checksum(&result, n),
                }
            };
            let (effect, null) = common::run_median_ci_contract(
                &format!("interference_duration_2p{exponent}"),
                shadowed,
                isolated,
            );
            let interference_ns = effect.arm_a_median_ns - effect.arm_b_median_ns;
            println!(
                "INTERFERENCE_VS_DURATION incumbent=maximum n={n} log2n={exponent} \
                 shadow_n={SHADOW_N} shadow_size_PINNED=true numpy_version={numpy_version} \
                 worker={} harness=common::run_median_ci_contract \
                 shadowed_ns={:.1} isolated_ns={:.1} interference_ns={interference_ns:.1} \
                 ratio={:.6} ratio_ci95=[{:.6},{:.6}] null={:.6} \
                 fixed_cost_model_predicts_interference_ns_constant=true",
                measurement_worker(),
                effect.arm_a_median_ns,
                effect.arm_b_median_ns,
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                null.ratio_median,
            );
        }
    });
}

/// THE CONFOUND-FREE interference experiment: hold the SHADOW constant, vary only the
/// incumbent (`deadlock-audit-48by6`).
///
/// Two rows ago I concluded that interference is governed by whether the INCUMBENT is
/// bandwidth-bound, from two cells: `numpy.maximum` shadowed by `maximum_parallel` showed
/// 1.067x, and `numpy.remainder` shadowed by `fnp.remainder` showed none. **Those two
/// cells differ in the SHADOW as well as the incumbent, and in size.** The conclusion is
/// therefore not licensed by that data — a different shadow could explain the difference
/// just as well as a different incumbent, and I did not notice while writing it.
///
/// This group removes the confound. ONE shadow (`maximum_parallel`), ONE size (2^22), two
/// incumbents that differ only in their regime:
///   `numpy.maximum`   — one trivial op per element, memory-bound
///   `numpy.remainder` — an fmod per element, compute-bound
///
/// PRE-REGISTERED PREDICTION: if the incumbent's regime is the driver, `maximum` shows
/// interference near the 1.067x already replicated and `remainder` shows none. If BOTH
/// show interference, the driver was the shadow all along and the compute-bound story is
/// wrong. If NEITHER does, the earlier maximum result was specific to something else again.
fn bench_incumbent_interference_shadow_held_constant(_c: &mut Criterion) {
    const N: usize = 1 << 22;
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
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
        let a_obj = locals.get_item("a").expect("a operand");
        let b_obj = locals.get_item("b").expect("b operand");
        let args = PyTuple::new(py, [&a_obj, &b_obj]).expect("args");
        let np_out = numpy
            .call_method1("empty_like", (&a_obj,))
            .expect("preallocated numpy output");
        let out_kwargs = PyDict::new(py);
        out_kwargs.set_item("out", &np_out).expect("bind out=");

        let a_vec: Vec<f64> = a_obj
            .call_method0("tolist")
            .expect("a list")
            .extract()
            .expect("a f64s");
        let b_vec: Vec<f64> = b_obj
            .call_method0("tolist")
            .expect("b list")
            .extract()
            .expect("b f64s");
        let mut shadow_out = vec![0.0_f64; N];

        for op in ["maximum", "remainder"] {
            let np_op = numpy.getattr(op).expect("numpy op");
            let isolated = || {
                let started = Instant::now();
                let result = np_op
                    .call(&args, Some(&out_kwargs))
                    .expect("numpy op isolated");
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: numpy_divide_checksum(&result, N),
                }
            };
            let shadowed = || {
                // The SAME shadow for both incumbents — this is the whole point.
                maximum_parallel(&a_vec, &b_vec, &mut shadow_out);
                let started = Instant::now();
                let result = np_op
                    .call(&args, Some(&out_kwargs))
                    .expect("numpy op shadowed");
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: numpy_divide_checksum(&result, N),
                }
            };
            let (effect, null) = common::run_median_ci_contract(
                &format!("interference_fixed_shadow_{op}"),
                shadowed,
                isolated,
            );
            let interference_ns = effect.arm_a_median_ns - effect.arm_b_median_ns;
            println!(
                "INTERFERENCE_FIXED_SHADOW incumbent={op} n={N} numpy_version={numpy_version} \
                 worker={} harness=common::run_median_ci_contract \
                 shadow=maximum_parallel_HELD_CONSTANT size_held_constant=true \
                 shadowed_ns={:.1} isolated_ns={:.1} interference_ns={interference_ns:.1} \
                 ratio={:.6} ratio_ci95=[{:.6},{:.6}] null={:.6}",
                measurement_worker(),
                effect.arm_a_median_ns,
                effect.arm_b_median_ns,
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                null.ratio_median,
            );
        }
    });
}

/// The same interference test, but for the REMAINDER cell and using the SHIPPED ROUTE as
/// the shadow (`deadlock-audit-48by6`).
///
/// The measured 6.26% interference factor was obtained with `maximum_parallel` shadowing
/// `numpy.maximum`. I then applied it to discount the remainder row from 6.967606x to
/// ~6.557x — which is precisely the cross-candidate application that row's own scope
/// paragraph warned against. This gives remainder its own factor instead of borrowing
/// one, and it shadows with `fnp.remainder` itself rather than a replica, so the number
/// describes the cell as actually measured.
///
/// Arm A: `numpy.remainder(a, b, out=)` timed alone.
/// Arm B: `fnp.remainder(a, b)` runs first, UNTIMED, then the same NumPy call is timed.
fn bench_incumbent_interference_remainder_route(_c: &mut Criterion) {
    const N: usize = 1 << 21;
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
        let a_obj = locals.get_item("a").expect("a operand");
        let b_obj = locals.get_item("b").expect("b operand");
        let args = PyTuple::new(py, [&a_obj, &b_obj]).expect("args");
        let np_remainder = numpy.getattr("remainder").expect("numpy.remainder");
        let fnp_remainder = module.getattr("remainder").expect("fnp.remainder");
        assert!(
            !fnp_remainder.is(&np_remainder),
            "fnp.remainder IS numpy's object — there is no candidate to shadow with"
        );
        let np_out = numpy
            .call_method1("empty_like", (&a_obj,))
            .expect("preallocated numpy output");
        let out_kwargs = PyDict::new(py);
        out_kwargs.set_item("out", &np_out).expect("bind out=");

        let isolated = || {
            let started = Instant::now();
            let result = np_remainder
                .call(&args, Some(&out_kwargs))
                .expect("numpy.remainder isolated");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: numpy_divide_checksum(&result, N),
            }
        };
        let shadowed = || {
            // The shipped route, deliberately OUTSIDE the timer.
            black_box(
                fnp_remainder
                    .call1(&args)
                    .expect("fnp.remainder shadow call"),
            );
            let started = Instant::now();
            let result = np_remainder
                .call(&args, Some(&out_kwargs))
                .expect("numpy.remainder shadowed");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: numpy_divide_checksum(&result, N),
            }
        };
        let (effect, null) =
            common::run_median_ci_contract("incumbent_interference_remainder", shadowed, isolated);

        let interference_ns = effect.arm_a_median_ns - effect.arm_b_median_ns;
        println!(
            "INCUMBENT_INTERFERENCE_REMAINDER n={N} numpy_version={numpy_version} worker={} \
             harness=common::run_median_ci_contract shadow=fnp_remainder_SHIPPED_ROUTE \
             candidate_work_is_outside_the_timer=true \
             shadowed_ns={:.1} isolated_ns={:.1} interference_ns={interference_ns:.1} \
             ratio={:.6} ratio_ci95=[{:.6},{:.6}] null={:.6} \
             discount_for_the_remainder_row_only=true",
            measurement_worker(),
            effect.arm_a_median_ns,
            effect.arm_b_median_ns,
            effect.ratio_median,
            effect.ratio_ci_low,
            effect.ratio_ci_high,
            null.ratio_median,
        );
    });
}

/// Does OUR arm slow down the incumbent it is measured against? (`deadlock-audit-48by6`)
///
/// On a quiet host the maximum-arms row showed NumPy's own arm getting SLOWER — 4232670
/// -> 4947948 ns (+17%) and 4111546 -> 4610579 ns (+12%) — while everything else moved
/// the expected way. The lead recorded there: inside an interleaved ABBAABBA schedule our
/// candidate runs immediately adjacent to NumPy's sample, and at 2^22 f64 both stream
/// 32 MB buffers. When our arm is FAST it saturates memory bandwidth harder per unit
/// time, so the NumPy sample that follows may start with colder caches and a busier
/// memory subsystem. If that is real, **every ratio in this ledger taken with a
/// bandwidth-heavy candidate is mildly self-flattering**, because we degrade the
/// incumbent we are divided by.
///
/// WHY THIS NEEDS ITS OWN GROUP RATHER THAN A READ-OFF. The harness already measures the
/// incumbent alone, in its own A/A null phase, and interleaved, in the effect phase — so
/// the comparison looks free. It is not: those phases run at DIFFERENT TIMES, the effect
/// phase runs last, and any load drift during the invocation lands entirely on that
/// difference. On a rising host the read-off would manufacture exactly the interference
/// signal it claims to detect. This group instead ALTERNATES the two conditions inside
/// one ABBAABBA schedule, so drift cancels.
///
/// Arm A: `numpy.maximum(a, b, out=)` timed alone.
/// Arm B: our parallel replica runs first, UNTIMED, then the same NumPy call is timed.
/// Both arms time exactly the same NumPy work; they differ only in what ran immediately
/// before. A ratio above 1.0 means NumPy is slower when shadowed, i.e. interference is
/// real and every affected ratio is optimistic by that factor.
fn bench_incumbent_interference_from_candidate(_c: &mut Criterion) {
    const N: usize = 1 << 22;
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
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
        let a_obj = locals.get_item("a").expect("a operand");
        let b_obj = locals.get_item("b").expect("b operand");
        let args = PyTuple::new(py, [&a_obj, &b_obj]).expect("args");
        let np_maximum = numpy.getattr("maximum").expect("numpy.maximum");
        let np_out = numpy
            .call_method1("empty_like", (&a_obj,))
            .expect("preallocated numpy output");
        let out_kwargs = PyDict::new(py);
        out_kwargs.set_item("out", &np_out).expect("bind out=");

        let a_vec: Vec<f64> = a_obj
            .call_method0("tolist")
            .expect("a list")
            .extract()
            .expect("a f64s");
        let b_vec: Vec<f64> = b_obj
            .call_method0("tolist")
            .expect("b list")
            .extract()
            .expect("b f64s");
        let mut shadow_out = vec![0.0_f64; N];

        // Arm A: the incumbent alone. Nothing of ours runs adjacent to it.
        // Per-arm clock sampling (`deadlock-audit-48by6`). If a 64-thread shadow pushes the
        // package into a lower turbo bin, the single-threaded NumPy call that follows runs
        // slower for a reason that has nothing to do with caches — and that would be a
        // FOURTH mechanism for this effect after three I have already withdrawn. PREDICTION,
        // registered here before the run: if frequency is the cause, the shadowed arm
        // samples LOWER MHz than the isolated arm.
        let isolated_mhz = RefCell::new(Vec::new());
        let shadowed_mhz = RefCell::new(Vec::new());
        let isolated_cpu: RefCell<Vec<i32>> = RefCell::new(Vec::new());
        let shadowed_cpu: RefCell<Vec<i32>> = RefCell::new(Vec::new());

        let isolated = || {
            let started = Instant::now();
            let result = np_maximum
                .call(&args, Some(&out_kwargs))
                .expect("numpy.maximum isolated");
            let elapsed = started.elapsed();
            // Sampled AFTER the timer stops, so the sysfs reads are not charged to the arm.
            let (max_mhz, _mean_mhz) = sample_cpu_mhz();
            isolated_mhz.borrow_mut().push(max_mhz);
            isolated_cpu.borrow_mut().push(current_cpu());
            common::ContractObservation {
                elapsed,
                checksum: numpy_divide_checksum(&result, N),
            }
        };
        // Arm B: the SAME incumbent call, but our bandwidth-heavy replica runs first and
        // is deliberately OUTSIDE the timer. Only NumPy's work is timed in both arms.
        let shadowed = || {
            maximum_parallel(&a_vec, &b_vec, &mut shadow_out);
            let started = Instant::now();
            let result = np_maximum
                .call(&args, Some(&out_kwargs))
                .expect("numpy.maximum shadowed");
            let elapsed = started.elapsed();
            let (max_mhz, _mean_mhz) = sample_cpu_mhz();
            shadowed_mhz.borrow_mut().push(max_mhz);
            shadowed_cpu.borrow_mut().push(current_cpu());
            common::ContractObservation {
                elapsed,
                checksum: numpy_divide_checksum(&result, N),
            }
        };
        let (effect, null) =
            common::run_median_ci_contract("incumbent_interference", shadowed, isolated);

        let interference_ns = effect.arm_a_median_ns - effect.arm_b_median_ns;
        let shadowed_mhz_median = median_of(&mut shadowed_mhz.borrow_mut());
        let isolated_mhz_median = median_of(&mut isolated_mhz.borrow_mut());
        let (iso_cpu, iso_distinct) = modal_cpu(&isolated_cpu.borrow());
        let (sh_cpu, sh_distinct) = modal_cpu(&shadowed_cpu.borrow());
        let iso_core = core_id_of(iso_cpu);
        let sh_core = core_id_of(sh_cpu);
        let arms_same_cpu = iso_cpu == sh_cpu;
        let arms_same_physical_core = iso_core >= 0 && iso_core == sh_core;
        let arms_are_smt_siblings = arms_same_physical_core && !arms_same_cpu;
        let mhz_ratio = if isolated_mhz_median > 0.0 {
            shadowed_mhz_median / isolated_mhz_median
        } else {
            0.0
        };
        println!(
            "INCUMBENT_INTERFERENCE n={N} numpy_version={numpy_version} worker={} \
             harness=common::run_median_ci_contract \
             arms=same_numpy_call_timed_differing_only_in_what_ran_immediately_before \
             candidate_work_is_outside_the_timer=true \
             shadowed_ns={:.1} isolated_ns={:.1} interference_ns={interference_ns:.1} \
             ratio={:.6} ratio_ci95=[{:.6},{:.6}] null={:.6} \
             shadowed_max_mhz={shadowed_mhz_median:.0} isolated_max_mhz={isolated_mhz_median:.0} \
             mhz_ratio_shadowed_over_isolated={mhz_ratio:.4} \
             isolated_cpu={iso_cpu} isolated_core_id={iso_core} isolated_distinct_cpus={iso_distinct} \
             shadowed_cpu={sh_cpu} shadowed_core_id={sh_core} shadowed_distinct_cpus={sh_distinct} \
             arms_same_cpu={arms_same_cpu} arms_same_physical_core={arms_same_physical_core} \
             arms_are_smt_siblings={arms_are_smt_siblings} \
             above_one_means_our_ratios_are_optimistic_by_this_factor=true",
            measurement_worker(),
            effect.arm_a_median_ns,
            effect.arm_b_median_ns,
            effect.ratio_median,
            effect.ratio_ci_low,
            effect.ratio_ci_high,
            null.ratio_median,
        );
    });
}

fn bench_maximum_arms_vs_numpy(_c: &mut Criterion) {
    const N: usize = 1 << 22;
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let numpy = py.import("numpy").expect("numpy oracle");
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy.__version__")
            .extract::<String>()
            .expect("numpy version is a string");
        let locals = PyDict::new(py);
        locals.set_item("np", &numpy).expect("bind numpy");
        locals.set_item("n", N).expect("bind n");
        // Mixed signs so neither operand dominates, plus NaN and signed-zero lanes
        // so the replica's semantics are exercised rather than assumed.
        py.run(
            std::ffi::CString::new(
                "i = np.arange(n)\na = (1.0 + (i % 1000) / 1000.0) * np.where(i % 3 == 0, -1.0, 1.0)\nb = (1.25 + (i % 997) / 997.0) * np.where(i % 5 == 0, -1.0, 1.0)\na[3] = np.nan\nb[7] = np.nan\na[11] = 0.0\nb[11] = -0.0\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&locals),
            Some(&locals),
        )
        .expect("build operands");
        let a_obj = locals.get_item("a").expect("a operand");
        let b_obj = locals.get_item("b").expect("b operand");
        let args = PyTuple::new(py, [&a_obj, &b_obj]).expect("args");
        let np_maximum = numpy.getattr("maximum").expect("numpy.maximum");

        // ALLOCATION SYMMETRY (`deadlock-audit-48by6`). This group used to time
        // `np_maximum.call1(&args)` — positional only — against Rust replicas writing
        // into `Vec`s allocated ONCE outside the timing loop. That handed the candidate,
        // for free, the single most expensive thing the incumbent did: a fresh 32 MB
        // output allocation and its first-touch page faults on every iteration. It made
        // this group read 2.430654x for `maximum` where the shipped route reads 0.907848x
        // at the same n on the same host — a 2.7x disagreement that was measuring a
        // buffer, not a kernel, and that a REJECT of `deadlock-audit-hzl1w` was resting on.
        //
        // The fix is to make NEITHER side allocate, which is the shape
        // `bench_divide_accumulate_isolation_vs_numpy` already uses: NumPy writes into a
        // preallocated `out=` exactly as the replicas write into preallocated `Vec`s.
        //
        // NOT fixed by making the CANDIDATE allocate per iteration instead: `numpy.empty`
        // and a Rust `Vec` do not have the same first-touch behaviour, so that swaps one
        // asymmetry for another and merely moves the bias. Both sides preallocated is the
        // only arrangement here where the difference is the kernel.
        //
        // A/A NULLS CANNOT SEE THIS. Both arms were internally reproducible and both nulls
        // sat on unity throughout; a null proves an arm is stable, never that two arms are
        // comparable. That blind spot is the reason this survived so long.
        let np_out = numpy
            .call_method1("empty_like", (&a_obj,))
            .expect("preallocated numpy output");
        let out_kwargs = PyDict::new(py);
        out_kwargs.set_item("out", &np_out).expect("bind out=");

        let a_vec: Vec<f64> = a_obj
            .call_method0("tolist")
            .expect("a list")
            .extract()
            .expect("a f64s");
        let b_vec: Vec<f64> = b_obj
            .call_method0("tolist")
            .expect("b list")
            .extract()
            .expect("b f64s");
        let mut serial_out = vec![0.0_f64; N];
        let mut parallel_out = vec![0.0_f64; N];

        // PARITY BEFORE TIMING: both replicas must reproduce NumPy bit for bit on
        // these operands, NaN and signed-zero lanes included, or the ratio is
        // between two different computations.
        maximum_serial(&a_vec, &b_vec, &mut serial_out);
        maximum_parallel(&a_vec, &b_vec, &mut parallel_out);
        let numpy_probe = np_maximum
            .call(&args, Some(&out_kwargs))
            .expect("numpy.maximum probe into out=");
        let numpy_checksum = numpy_divide_checksum(&numpy_probe, N);
        assert_eq!(
            divide_checksum(&serial_out),
            numpy_checksum,
            "serial maximum replica diverges from numpy.maximum"
        );
        assert_eq!(
            divide_checksum(&parallel_out),
            numpy_checksum,
            "parallel maximum replica diverges from numpy.maximum"
        );

        {
            let incumbent = || {
                let started = Instant::now();
                let result = np_maximum
                    .call(&args, Some(&out_kwargs))
                    .expect("numpy.maximum into out=");
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: numpy_divide_checksum(&result, N),
                }
            };
            let parallel_arm = || {
                let started = Instant::now();
                maximum_parallel(&a_vec, &b_vec, &mut parallel_out);
                let elapsed = started.elapsed();
                let checksum = divide_checksum(&parallel_out);
                common::ContractObservation { elapsed, checksum }
            };
            let (effect, _, _) = common::run_dual_null_median_ci_contract(
                "maximum_f64_parallel_vs_numpy",
                incumbent,
                parallel_arm,
            );
            println!(
                "MAXIMUM_ARM arm=parallel_native n={N} numpy_version={numpy_version} \
                 harness=common::run_dual_null_median_ci_contract \
                 allocation=neither_arm_allocates \
                 incumbent_writes_into_preallocated_numpy_out=true \
                 candidate_writes_into_preallocated_vec=true \
                 arms_are_replicas_not_the_shipped_route=true \
                 correction=deadlock-audit-48by6 \
                 ratio={:.6} ratio_ci95=[{:.6},{:.6}] numpy_ns={:.1} fnp_ns={:.1} \
                 faster_than_numpy={} worst_bound={:.6}",
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                effect.arm_a_median_ns,
                effect.arm_b_median_ns,
                effect.ratio_ci_low > 1.0,
                effect.ratio_ci_low,
            );
        }

        {
            let incumbent = || {
                let started = Instant::now();
                let result = np_maximum
                    .call(&args, Some(&out_kwargs))
                    .expect("numpy.maximum into out=");
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: numpy_divide_checksum(&result, N),
                }
            };
            let serial_arm = || {
                let started = Instant::now();
                maximum_serial(&a_vec, &b_vec, &mut serial_out);
                let elapsed = started.elapsed();
                let checksum = divide_checksum(&serial_out);
                common::ContractObservation { elapsed, checksum }
            };
            let (effect, _, _) = common::run_dual_null_median_ci_contract(
                "maximum_f64_serial_vs_numpy",
                incumbent,
                serial_arm,
            );
            println!(
                "MAXIMUM_ARM arm=serial_native n={N} numpy_version={numpy_version} \
                 harness=common::run_dual_null_median_ci_contract \
                 allocation=neither_arm_allocates \
                 incumbent_writes_into_preallocated_numpy_out=true \
                 candidate_writes_into_preallocated_vec=true \
                 arms_are_replicas_not_the_shipped_route=true \
                 correction=deadlock-audit-48by6 \
                 ratio={:.6} ratio_ci95=[{:.6},{:.6}] numpy_ns={:.1} fnp_ns={:.1} \
                 faster_than_numpy={} worst_bound={:.6} same_invocation_as_parallel_arm=true",
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                effect.arm_a_median_ns,
                effect.arm_b_median_ns,
                effect.ratio_ci_low > 1.0,
                effect.ratio_ci_low,
            );
        }
    });
}
