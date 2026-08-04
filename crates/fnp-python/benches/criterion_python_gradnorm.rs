//! gradient / norm / poly criterion benches — cumulative on a middle axis, int
//! cumulative, Vandermonde, polyval, np.gradient (2-D/N-D coords, f32, axis), and
//! matrix/vector norms (axis, f32 order-free, non-last axis, Frobenius) — split
//! out of the monolithic `criterion_python_surface.rs` into their own per-domain
//! bench binary. See bead deadlock-audit-x7nnf.

#[path = "common/mod.rs"]
mod common;

use common::ensure_numpy_available;
use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyDict, PyModule, PyTuple};
use std::hint::black_box;
use std::time::Duration;

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
            numpy_kwargs
                .set_item("axis", 1_i64)
                .expect("numpy axis kwarg");

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
                    let r = numpy_cumsum
                        .call((arr,), Some(&nk))
                        .expect("numpy int cumsum");
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

        for (label, n, cols) in [
            ("200k_x8", 200_000_i64, 8_i64),
            ("500k_x12", 500_000_i64, 12_i64),
        ] {
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

        for (label, n, deg) in [
            ("1M_deg5", 1_000_000_i64, 5_i64),
            ("4M_deg8", 4_000_000_i64, 8_i64),
        ] {
            let x = numpy
                .call_method1("linspace", (-3.0_f64, 3.0_f64, n))
                .expect("polyval x input");
            let p = numpy
                .call_method1("linspace", (0.5_f64, 2.0_f64, deg + 1))
                .expect("polyval coeffs");

            group.bench_function(format!("fnp_polyval_f64_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_polyval.call1((&p, &x)).expect("fnp polyval call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_polyval_f64_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_polyval.call1((&p, &x)).expect("numpy polyval call");
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
            numpy_kwargs
                .set_item("axis", 0_i64)
                .expect("numpy axis kwarg");

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
                    let result = numpy_grad
                        .call1((&input,))
                        .expect("numpy gradient full call");
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
            fnp_l1_kwargs
                .set_item("ord", 1_i64)
                .expect("fnp l1 ord kwarg");
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
            fnp_inf_kwargs
                .set_item("ord", inf)
                .expect("fnp inf ord kwarg");
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
        let fnp_norm = module
            .getattr("linalg")
            .unwrap()
            .getattr("norm")
            .expect("fnp norm");
        let numpy_norm = numpy
            .getattr("linalg")
            .unwrap()
            .getattr("norm")
            .expect("np norm");
        {
            let (label, ordv) = ("maxabs", inf.clone());
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
                numpy_kwargs
                    .set_item("ord", ord_val)
                    .expect("numpy ord kwarg");
                numpy_kwargs
                    .set_item("axis", axis)
                    .expect("numpy axis kwarg");

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

fn main() {
    common::gated_main(&[
        ("bench_cum_midaxis_boundary", bench_cum_midaxis_boundary),
        ("bench_int_cum_boundary", bench_int_cum_boundary),
        ("bench_vander_boundary", bench_vander_boundary),
        ("bench_polyval_boundary", bench_polyval_boundary),
        (
            "bench_gradient_2d_coords_boundary",
            bench_gradient_2d_coords_boundary,
        ),
        (
            "bench_gradient_coords_boundary",
            bench_gradient_coords_boundary,
        ),
        (
            "bench_gradient_nd_coords_axis_boundary",
            bench_gradient_nd_coords_axis_boundary,
        ),
        ("bench_gradient_f32_boundary", bench_gradient_f32_boundary),
        ("bench_gradient_axis_boundary", bench_gradient_axis_boundary),
        ("bench_norm_axis_boundary", bench_norm_axis_boundary),
        (
            "bench_norm_f32_orderfree_boundary",
            bench_norm_f32_orderfree_boundary,
        ),
        (
            "bench_norm_nonlast_axis_boundary",
            bench_norm_nonlast_axis_boundary,
        ),
        (
            "bench_norm_frobenius_boundary",
            bench_norm_frobenius_boundary,
        ),
    ]);
}
