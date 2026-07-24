//! complex-dtype domain criterion benches — complex/complex64 arithmetic
//! (exp, binary ufuncs), cumulative reductions (cumprod / nancumprod /
//! cumulative on mid and axis-0), and set operations (unique / searchsorted /
//! isin / ops) — split out of the monolithic `criterion_python_surface.rs` into
//! their own per-domain bench binary, so a per-domain run compiles only these
//! groups. See bead deadlock-audit-x7nnf.

#[path = "common/mod.rs"]
mod common;

use common::ensure_numpy_available;
use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyDict, PyModule};
use std::hint::black_box;
use std::time::Duration;

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
            .call_method1(
                "__add__",
                (sinv.call_method1("__mul__", (&j,)).expect("i*sin"),),
            )
            .expect("complex base")
            .call_method1("reshape", ((rows, cols),))
            .expect("complex 2-D shape");

        for (label, cname) in [("complex128", "complex128"), ("complex64", "complex64")] {
            let input = base
                .call_method1("astype", (cname,))
                .expect("astype complex");
            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("axis", -1_i64).expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs
                .set_item("axis", -1_i64)
                .expect("numpy axis kwarg");

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
            .call_method1(
                "__add__",
                (sinv.call_method1("__mul__", (&j,)).expect("i*sin"),),
            )
            .expect("complex base")
            .call_method1("reshape", ((rows, cols),))
            .expect("complex 2-D shape");

        for (label, cname) in [("complex128", "complex128"), ("complex64", "complex64")] {
            let input = base
                .call_method1("astype", (cname,))
                .expect("astype complex");
            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("axis", -1_i64).expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs
                .set_item("axis", -1_i64)
                .expect("numpy axis kwarg");

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
            .call_method1(
                "__add__",
                (sinv.call_method1("__mul__", (&j,)).expect("i*sin"),),
            )
            .expect("complex base")
            .call_method1("reshape", ((n, n, n),))
            .expect("complex 3-D shape");

        for (label, cname) in [("complex128", "complex128"), ("complex64", "complex64")] {
            let input = base
                .call_method1("astype", (cname,))
                .expect("astype complex");
            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("axis", 1_i64).expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs
                .set_item("axis", 1_i64)
                .expect("numpy axis kwarg");

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
                .call_method1(
                    "__add__",
                    (sinv.call_method1("__mul__", (&j,)).expect("i*sin"),),
                )
                .expect("complex base")
                .call_method1("reshape", ((n, n),))
                .expect("complex 2-D shape");
            for (dlabel, cname) in [("complex128", "complex128"), ("complex64", "complex64")] {
                let input = base
                    .call_method1("astype", (cname,))
                    .expect("astype complex");
                let fnp_kwargs = PyDict::new(py);
                fnp_kwargs.set_item("axis", 0_i64).expect("fnp axis kwarg");
                let numpy_kwargs = PyDict::new(py);
                numpy_kwargs
                    .set_item("axis", 0_i64)
                    .expect("numpy axis kwarg");

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
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let cc = ns.get_item("c").expect("c");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let f = fnp_u.call1((&cc,)).expect("fnp unique");
        let n = numpy_u.call1((&cc,)).expect("numpy unique");
        assert!(
            eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
            "complex unique mismatch"
        );
        group.bench_function("fnp_unique_c128_2m", |bn| {
            bn.iter(|| black_box(fnp_u.call1((&cc,)).unwrap()))
        });
        group.bench_function("numpy_unique_c128_2m", |bn| {
            bn.iter(|| black_box(numpy_u.call1((&cc,)).unwrap()))
        });
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
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let cc = ns.get_item("c").expect("c");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let f = fnp_u.call1((&cc,)).expect("fnp unique");
        let n = numpy_u.call1((&cc,)).expect("numpy unique");
        assert!(
            eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
            "complex64 unique mismatch"
        );
        group.bench_function("fnp_unique_c64_2m", |bn| {
            bn.iter(|| black_box(fnp_u.call1((&cc,)).unwrap()))
        });
        group.bench_function("numpy_unique_c64_2m", |bn| {
            bn.iter(|| black_box(numpy_u.call1((&cc,)).unwrap()))
        });
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
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
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
            assert!(
                eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
                "complex searchsorted mismatch side={side}"
            );
        }
        group.bench_function("fnp_searchsorted_c128_2m", |bn| {
            bn.iter(|| black_box(fnp_ss.call1((&h, &q)).unwrap()))
        });
        group.bench_function("numpy_searchsorted_c128_2m", |bn| {
            bn.iter(|| black_box(numpy_ss.call1((&h, &q)).unwrap()))
        });
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
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
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
            assert!(
                eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
                "complex isin mismatch invert={inv}"
            );
        }
        group.bench_function("fnp_isin_c128_2m", |bn| {
            bn.iter(|| black_box(fnp_isin.call1((&a, &test)).unwrap()))
        });
        group.bench_function("numpy_isin_c128_2m", |bn| {
            bn.iter(|| black_box(numpy_isin.call1((&a, &test)).unwrap()))
        });
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
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let h = ns.get_item("h").unwrap();
        let q = ns.get_item("q").unwrap();
        let a = ns.get_item("a").unwrap();
        let test = ns.get_item("test").unwrap();
        let eqf = numpy.getattr("array_equal").unwrap();
        // searchsorted correctness (left+right) + isin (default+invert).
        for side in ["left", "right"] {
            let kw = PyDict::new(py);
            kw.set_item("side", side).unwrap();
            let f = module
                .getattr("searchsorted")
                .unwrap()
                .call((&h, &q), Some(&kw))
                .unwrap();
            let n = numpy
                .getattr("searchsorted")
                .unwrap()
                .call((&h, &q), Some(&kw))
                .unwrap();
            assert!(
                eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
                "c64 searchsorted mismatch side={side}"
            );
        }
        for inv in [false, true] {
            let kw = PyDict::new(py);
            kw.set_item("invert", inv).unwrap();
            let f = module
                .getattr("isin")
                .unwrap()
                .call((&a, &test), Some(&kw))
                .unwrap();
            let n = numpy
                .getattr("isin")
                .unwrap()
                .call((&a, &test), Some(&kw))
                .unwrap();
            assert!(
                eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
                "c64 isin mismatch invert={inv}"
            );
        }
        let ss_f = module.getattr("searchsorted").unwrap();
        let ss_n = numpy.getattr("searchsorted").unwrap();
        let is_f = module.getattr("isin").unwrap();
        let is_n = numpy.getattr("isin").unwrap();
        group.bench_function("fnp_searchsorted_c64_2m", |bn| {
            bn.iter(|| black_box(ss_f.call1((&h, &q)).unwrap()))
        });
        group.bench_function("numpy_searchsorted_c64_2m", |bn| {
            bn.iter(|| black_box(ss_n.call1((&h, &q)).unwrap()))
        });
        group.bench_function("fnp_isin_c64_2m", |bn| {
            bn.iter(|| black_box(is_f.call1((&a, &test)).unwrap()))
        });
        group.bench_function("numpy_isin_c64_2m", |bn| {
            bn.iter(|| black_box(is_n.call1((&a, &test)).unwrap()))
        });
    });
    group.finish();
}

fn main() {
    common::gated_main(&[
        ("bench_complex_exp_boundary", bench_complex_exp_boundary),
        ("bench_complex_binary_boundary", bench_complex_binary_boundary),
        ("bench_complex_cumprod_lastaxis_boundary", bench_complex_cumprod_lastaxis_boundary),
        ("bench_complex_nancumprod_lastaxis_boundary", bench_complex_nancumprod_lastaxis_boundary),
        ("bench_complex_cumulative_midaxis_boundary", bench_complex_cumulative_midaxis_boundary),
        ("bench_complex_cumulative_axis0_boundary", bench_complex_cumulative_axis0_boundary),
        ("bench_complex_unique_boundary", bench_complex_unique_boundary),
        ("bench_complex64_unique_boundary", bench_complex64_unique_boundary),
        ("bench_complex_searchsorted_boundary", bench_complex_searchsorted_boundary),
        ("bench_complex_isin_boundary", bench_complex_isin_boundary),
        ("bench_complex64_ops_boundary", bench_complex64_ops_boundary),
    ]);
}
