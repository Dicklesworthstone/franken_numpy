//! statistics / variance / nan-reduction / cumulative criterion benches —
//! statistics, cov, std and var across axes, nanvar/nansum/nanextreme (f32),
//! var axis-0, sum/prod/cumsum last-axis, cumsum-flat, and accumulate-extremum —
//! split out of the monolithic `criterion_python_surface.rs` into their own
//! per-domain bench binary. See bead deadlock-audit-x7nnf.

#[path = "common/mod.rs"]
mod common;

use common::ensure_numpy_available;
use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyDict, PyModule, PyTuple};
use std::hint::black_box;
use std::time::Duration;

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
            numpy_kwargs
                .set_item("axis", 1_i64)
                .expect("numpy axis kwarg");

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

        for (label, input, axis) in [
            ("mid_256x128x256", &mid, 1_i64),
            ("axis0_4000x2000", &ax0, 0_i64),
        ] {
            let fkw = PyDict::new(py);
            fkw.set_item("axis", axis).expect("axis");
            let nkw = PyDict::new(py);
            nkw.set_item("axis", axis).expect("axis");
            group.bench_function(format!("fnp_var_f32_{label}"), |bench| {
                bench.iter(|| black_box(fnp_var.call((input,), Some(&fkw)).expect("fnp var f32")));
            });
            group.bench_function(format!("numpy_var_f32_{label}"), |bench| {
                bench.iter(|| {
                    black_box(numpy_var.call((input,), Some(&nkw)).expect("numpy var f32"))
                });
            });
            group.bench_function(format!("fnp_std_f32_{label}"), |bench| {
                bench.iter(|| black_box(fnp_std.call((input,), Some(&fkw)).expect("fnp std f32")));
            });
            group.bench_function(format!("numpy_std_f32_{label}"), |bench| {
                bench.iter(|| {
                    black_box(numpy_std.call((input,), Some(&nkw)).expect("numpy std f32"))
                });
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
            b.iter(|| {
                black_box(
                    numpy_nansum
                        .call((&a64,), Some(&kw2))
                        .expect("np nansum f64"),
                )
            });
        });
        // f64 nansum LAST axis: per-lane pairwise (now bit-exact, was sequential) + parallel.
        let lax = PyDict::new(py);
        lax.set_item("axis", 2_i64).unwrap();
        let lax2 = lax.clone();
        group.bench_function("fnp_nansum_f64_last", |b| {
            b.iter(|| {
                black_box(
                    fnp_nansum
                        .call((&a64,), Some(&lax))
                        .expect("fnp nansum f64 last"),
                )
            });
        });
        group.bench_function("numpy_nansum_f64_last", |b| {
            b.iter(|| {
                black_box(
                    numpy_nansum
                        .call((&a64,), Some(&lax2))
                        .expect("np nansum f64 last"),
                )
            });
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
            arr.call_method1("__setitem__", (idx, &nan))
                .expect("inject NaN");
            arr.call_method1(
                "reshape",
                (PyTuple::new(py, dims.iter().copied()).unwrap(),),
            )
            .expect("reshape")
        };
        let mid = build(&[256, 128, 256], 256 * 128 * 256);
        let ax0 = build(&[4000, 2000], 4000 * 2000);

        for (label, input, axis) in [
            ("mid_256x128x256", &mid, 1_i64),
            ("axis0_4000x2000", &ax0, 0_i64),
        ] {
            let fkw = PyDict::new(py);
            fkw.set_item("axis", axis).expect("axis");
            let nkw = PyDict::new(py);
            nkw.set_item("axis", axis).expect("axis");
            group.bench_function(format!("fnp_nanvar_f32_{label}"), |b| {
                b.iter(|| {
                    black_box(
                        fnp_nanvar
                            .call((input,), Some(&fkw))
                            .expect("fnp nanvar f32"),
                    )
                });
            });
            group.bench_function(format!("numpy_nanvar_f32_{label}"), |b| {
                b.iter(|| {
                    black_box(
                        numpy_nanvar
                            .call((input,), Some(&nkw))
                            .expect("numpy nanvar f32"),
                    )
                });
            });
            group.bench_function(format!("fnp_nanstd_f32_{label}"), |b| {
                b.iter(|| {
                    black_box(
                        fnp_nanstd
                            .call((input,), Some(&fkw))
                            .expect("fnp nanstd f32"),
                    )
                });
            });
            group.bench_function(format!("numpy_nanstd_f32_{label}"), |b| {
                b.iter(|| {
                    black_box(
                        numpy_nanstd
                            .call((input,), Some(&nkw))
                            .expect("numpy nanstd f32"),
                    )
                });
            });
            group.bench_function(format!("fnp_nanmean_f32_{label}"), |b| {
                b.iter(|| {
                    black_box(
                        fnp_nanmean
                            .call((input,), Some(&fkw))
                            .expect("fnp nanmean f32"),
                    )
                });
            });
            group.bench_function(format!("numpy_nanmean_f32_{label}"), |b| {
                b.iter(|| {
                    black_box(
                        numpy_nanmean
                            .call((input,), Some(&nkw))
                            .expect("numpy nanmean f32"),
                    )
                });
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
            arr.call_method1("__setitem__", (idx, &nan))
                .expect("inject NaN");
            arr.call_method1(
                "reshape",
                (PyTuple::new(py, dims.iter().copied()).unwrap(),),
            )
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
            // Owned handles so the `Some(&fkw)` call sites below are a single
            // borrow (kw is already `&Bound`); clone is a refcount bump in
            // setup, outside every timed `b.iter` closure.
            let fkw = kw.clone();
            let nkw = kw.clone();
            group.bench_function(format!("fnp_nanvar_f32_{label}"), |b| {
                b.iter(|| {
                    black_box(
                        fnp_nanvar
                            .call((input,), Some(&fkw))
                            .expect("fnp nanvar f32"),
                    )
                });
            });
            group.bench_function(format!("numpy_nanvar_f32_{label}"), |b| {
                b.iter(|| {
                    black_box(
                        numpy_nanvar
                            .call((input,), Some(&nkw))
                            .expect("numpy nanvar f32"),
                    )
                });
            });
            group.bench_function(format!("fnp_nanstd_f32_{label}"), |b| {
                b.iter(|| {
                    black_box(
                        fnp_nanstd
                            .call((input,), Some(&fkw))
                            .expect("fnp nanstd f32"),
                    )
                });
            });
            group.bench_function(format!("numpy_nanstd_f32_{label}"), |b| {
                b.iter(|| {
                    black_box(
                        numpy_nanstd
                            .call((input,), Some(&nkw))
                            .expect("numpy nanstd f32"),
                    )
                });
            });
            group.bench_function(format!("fnp_nanmean_f32_{label}"), |b| {
                b.iter(|| {
                    black_box(
                        fnp_nanmean
                            .call((input,), Some(&fkw))
                            .expect("fnp nanmean f32"),
                    )
                });
            });
            group.bench_function(format!("numpy_nanmean_f32_{label}"), |b| {
                b.iter(|| {
                    black_box(
                        numpy_nanmean
                            .call((input,), Some(&nkw))
                            .expect("numpy nanmean f32"),
                    )
                });
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
            let flat = input
                .call_method1("reshape", ((size,),))
                .expect("flat view");
            let idx = numpy
                .call_method1("arange", (0_i64, size, 10_i64))
                .expect("nan index stride");
            let nan = numpy.getattr("nan").expect("np.nan");
            flat.call_method1("__setitem__", (idx, nan))
                .expect("inject NaN");

            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("axis", 1_i64).expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs
                .set_item("axis", 1_i64)
                .expect("numpy axis kwarg");

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
            numpy_kwargs
                .set_item("axis", 0_i64)
                .expect("numpy axis kwarg");

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

        for (label, rows, cols) in [
            ("8192x1024", 8192_i64, 1024_i64),
            ("65536x256", 65536_i64, 256_i64),
        ] {
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
            numpy_kwargs
                .set_item("axis", -1_i64)
                .expect("numpy axis kwarg");

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

        for (label, rows, cols) in [
            ("8192x1024", 8192_i64, 1024_i64),
            ("65536x256", 65536_i64, 256_i64),
        ] {
            let size = rows * cols;
            let input = numpy
                .call_method1("linspace", (-1.0_f64, 1.0_f64, size))
                .expect("cumsum input")
                .call_method1("reshape", ((rows, cols),))
                .expect("cumsum 2-D shape");
            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("axis", -1_i64).expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs
                .set_item("axis", -1_i64)
                .expect("numpy axis kwarg");

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
            b.iter(|| {
                black_box(
                    fnp_max
                        .call_method1("accumulate", (&x,))
                        .expect("fnp max.accum"),
                )
            });
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
            b.iter(|| {
                black_box(
                    fnp_max
                        .call_method1("accumulate", (&x32,))
                        .expect("fnp max.accum f32"),
                )
            });
        });
        group.bench_function("numpy_maximum_accumulate_f32_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_max
                        .call_method1("accumulate", (&x32,))
                        .expect("np max.accum f32"),
                )
            });
        });
        group.bench_function("fnp_maximum_accumulate_i64_8m", |b| {
            b.iter(|| {
                black_box(
                    fnp_max
                        .call_method1("accumulate", (&xi,))
                        .expect("fnp max.accum i64"),
                )
            });
        });
        group.bench_function("numpy_maximum_accumulate_i64_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_max
                        .call_method1("accumulate", (&xi,))
                        .expect("np max.accum i64"),
                )
            });
        });

        // add.accumulate(int) routes to the parallel cumsum path (== np.cumsum); a win
        // here proves the routing engages (vs the prior full delegation to numpy serial).
        let fnp_add = module.getattr("add").expect("fnp add");
        let numpy_add = numpy.getattr("add").expect("numpy add");
        let xa = numpy
            .call_method1("arange", (8_000_000_i64,))
            .expect("8M i64 arange");
        group.bench_function("fnp_add_accumulate_i64_8m", |b| {
            b.iter(|| {
                black_box(
                    fnp_add
                        .call_method1("accumulate", (&xa,))
                        .expect("fnp add.accum i64"),
                )
            });
        });
        group.bench_function("numpy_add_accumulate_i64_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_add
                        .call_method1("accumulate", (&xa,))
                        .expect("np add.accum i64"),
                )
            });
        });

        // bitwise_or.accumulate(int) native two-pass prefix vs numpy serial.
        let fnp_or = module.getattr("bitwise_or").expect("fnp bitwise_or");
        let numpy_or = numpy.getattr("bitwise_or").expect("numpy bitwise_or");
        group.bench_function("fnp_bitwise_or_accumulate_i64_8m", |b| {
            b.iter(|| {
                black_box(
                    fnp_or
                        .call_method1("accumulate", (&xi,))
                        .expect("fnp or.accum i64"),
                )
            });
        });
        group.bench_function("numpy_bitwise_or_accumulate_i64_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_or
                        .call_method1("accumulate", (&xi,))
                        .expect("np or.accum i64"),
                )
            });
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
            b.iter(|| {
                black_box(
                    fnp_land
                        .call_method1("accumulate", (&xb,))
                        .expect("fnp land.accum bool"),
                )
            });
        });
        group.bench_function("numpy_logical_and_accumulate_bool_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_land
                        .call_method1("accumulate", (&xb,))
                        .expect("np land.accum bool"),
                )
            });
        });
        let fnp_lor = module.getattr("logical_or").expect("fnp logical_or");
        let numpy_lor = numpy.getattr("logical_or").expect("numpy logical_or");
        group.bench_function("fnp_logical_or_accumulate_bool_8m", |b| {
            b.iter(|| {
                black_box(
                    fnp_lor
                        .call_method1("accumulate", (&xb,))
                        .expect("fnp lor.accum bool"),
                )
            });
        });
        group.bench_function("numpy_logical_or_accumulate_bool_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_lor
                        .call_method1("accumulate", (&xb,))
                        .expect("np lor.accum bool"),
                )
            });
        });
        let fnp_lxor = module.getattr("logical_xor").expect("fnp logical_xor");
        let numpy_lxor = numpy.getattr("logical_xor").expect("numpy logical_xor");
        group.bench_function("fnp_logical_xor_accumulate_bool_8m", |b| {
            b.iter(|| {
                black_box(
                    fnp_lxor
                        .call_method1("accumulate", (&xb,))
                        .expect("fnp lxor.accum bool"),
                )
            });
        });
        group.bench_function("numpy_logical_xor_accumulate_bool_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_lxor
                        .call_method1("accumulate", (&xb,))
                        .expect("np lxor.accum bool"),
                )
            });
        });
    });

    group.finish();
}

fn main() {
    common::gated_main(&[
        ("bench_statistics_boundary", bench_statistics_boundary),
        ("bench_cov_large_boundary", bench_cov_large_boundary),
        ("bench_std_var_axis_boundary", bench_std_var_axis_boundary),
        ("bench_var_multiaxis_boundary", bench_var_multiaxis_boundary),
        ("bench_var_midaxis_boundary", bench_var_midaxis_boundary),
        ("bench_var_f32_axis_boundary", bench_var_f32_axis_boundary),
        (
            "bench_nanextreme_f32_axis_boundary",
            bench_nanextreme_f32_axis_boundary,
        ),
        (
            "bench_nansum_f32_axis_boundary",
            bench_nansum_f32_axis_boundary,
        ),
        (
            "bench_nanvar_f32_axis_boundary",
            bench_nanvar_f32_axis_boundary,
        ),
        (
            "bench_nanvar_f32_last_axis_boundary",
            bench_nanvar_f32_last_axis_boundary,
        ),
        (
            "bench_nanvar_midaxis_boundary",
            bench_nanvar_midaxis_boundary,
        ),
        ("bench_var_axis0_boundary", bench_var_axis0_boundary),
        ("bench_sum_lastaxis_boundary", bench_sum_lastaxis_boundary),
        ("bench_prod_lastaxis_boundary", bench_prod_lastaxis_boundary),
        (
            "bench_cumsum_lastaxis_boundary",
            bench_cumsum_lastaxis_boundary,
        ),
        ("bench_cumsum_flat_boundary", bench_cumsum_flat_boundary),
        (
            "bench_accumulate_extremum_boundary",
            bench_accumulate_extremum_boundary,
        ),
    ]);
}
