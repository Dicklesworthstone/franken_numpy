//! Criterion benchmarks for the PyO3 `fnp_python` surface.
//!
//! These target Python-boundary costs that the Rust engine benches do not see.

use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyDict, PyModule};
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Duration;

#[path = "common/mod.rs"]
mod common;
use common::*;

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
        let permdt = perm
            .call_method1("astype", ("datetime64[s]",))
            .expect("perm datetime64");
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
            .call_method1(
                "__add__",
                (cim.call_method1(
                    "__mul__",
                    (pyo3::types::PyComplex::from_doubles(py, 0.0, 1.0),),
                )
                .expect("1j*im"),),
            )
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
            bch.iter(|| {
                black_box(
                    numpy_argsort
                        .call1((&la_f32,))
                        .expect("numpy argsort la f32"),
                )
            });
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
                (
                    rng.call_method1("standard_normal", (la.getattr("shape").expect("la shape"),))
                        .expect("la imag")
                        .call_method1("__mul__", (&onej,))
                        .expect("1j*la_im"),
                ),
            )
            .expect("la re+im")
            .call_method1("astype", ("complex128",))
            .expect("la c128");
        group.bench_function("fnp_argsort_c128_lastaxis_16Mx", |bch| {
            bch.iter(|| black_box(fnp_argsort.call1((&la_c,)).expect("fnp argsort la c128")));
        });
        group.bench_function("numpy_argsort_c128_lastaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    numpy_argsort
                        .call1((&la_c,))
                        .expect("numpy argsort la c128"),
                )
            });
        });
        let a0_c = a0
            .call_method1(
                "__add__",
                (
                    rng.call_method1("standard_normal", (a0.getattr("shape").expect("a0 shape"),))
                        .expect("a0 imag")
                        .call_method1("__mul__", (&onej,))
                        .expect("1j*a0_im"),
                ),
            )
            .expect("a0 re+im")
            .call_method1("astype", ("complex128",))
            .expect("a0 c128");
        group.bench_function("fnp_argsort_c128_axis0_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    fnp_argsort
                        .call((&a0_c,), Some(&axis0_kwargs))
                        .expect("fnp argsort a0 c128"),
                )
            });
        });
        group.bench_function("numpy_argsort_c128_axis0_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    numpy_argsort
                        .call((&a0_c,), Some(&axis0_kwargs))
                        .expect("numpy argsort a0 c128"),
                )
            });
        });
        let am_c = am
            .call_method1(
                "__add__",
                (
                    rng.call_method1("standard_normal", (am.getattr("shape").expect("am shape"),))
                        .expect("am imag")
                        .call_method1("__mul__", (&onej,))
                        .expect("1j*am_im"),
                ),
            )
            .expect("am re+im")
            .call_method1("astype", ("complex128",))
            .expect("am c128");
        group.bench_function("fnp_argsort_c128_midaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    fnp_argsort
                        .call((&am_c,), Some(&axis1_kwargs))
                        .expect("fnp argsort am c128"),
                )
            });
        });
        group.bench_function("numpy_argsort_c128_midaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    numpy_argsort
                        .call((&am_c,), Some(&axis1_kwargs))
                        .expect("numpy argsort am c128"),
                )
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
                black_box(
                    fnp_sort
                        .call((&a0_c,), Some(&axis0_kwargs))
                        .expect("fnp sort a0 c128"),
                )
            });
        });
        group.bench_function("numpy_sort_c128_axis0_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    numpy_sort
                        .call((&a0_c,), Some(&axis0_kwargs))
                        .expect("numpy sort a0 c128"),
                )
            });
        });
        group.bench_function("fnp_sort_c128_midaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    fnp_sort
                        .call((&am_c,), Some(&axis1_kwargs))
                        .expect("fnp sort am c128"),
                )
            });
        });
        group.bench_function("numpy_sort_c128_midaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    numpy_sort
                        .call((&am_c,), Some(&axis1_kwargs))
                        .expect("numpy sort am c128"),
                )
            });
        });
        // COMPLEX64 VALUE sort (np.sort): permc/la_c cast to complex64 (distinct-real -> tie-free)
        let permc64 = permc
            .call_method1("astype", ("complex64",))
            .expect("permc64");
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
            bch.iter(|| {
                black_box(
                    numpy_argsort
                        .call1((&la_c64,))
                        .expect("numpy argsort la c64"),
                )
            });
        });
        // COMPLEX64 argsort AXIS0 + MIDAXIS: reuse a0_c/am_c (distinct-real) cast to complex64
        let a0_c64 = a0_c.call_method1("astype", ("complex64",)).expect("a0_c64");
        group.bench_function("fnp_argsort_c64_axis0_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    fnp_argsort
                        .call((&a0_c64,), Some(&axis0_kwargs))
                        .expect("fnp argsort a0 c64"),
                )
            });
        });
        group.bench_function("numpy_argsort_c64_axis0_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    numpy_argsort
                        .call((&a0_c64,), Some(&axis0_kwargs))
                        .expect("numpy argsort a0 c64"),
                )
            });
        });
        let am_c64 = am_c.call_method1("astype", ("complex64",)).expect("am_c64");
        group.bench_function("fnp_argsort_c64_midaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    fnp_argsort
                        .call((&am_c64,), Some(&axis1_kwargs))
                        .expect("fnp argsort am c64"),
                )
            });
        });
        group.bench_function("numpy_argsort_c64_midaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    numpy_argsort
                        .call((&am_c64,), Some(&axis1_kwargs))
                        .expect("numpy argsort am c64"),
                )
            });
        });
        // COMPLEX64 VALUE sort AXIS0 + MIDAXIS: reuse a0_c64 (distinct-per-column) + am_c64 (distinct-per-lane)
        group.bench_function("fnp_sort_c64_axis0_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    fnp_sort
                        .call((&a0_c64,), Some(&axis0_kwargs))
                        .expect("fnp sort a0 c64"),
                )
            });
        });
        group.bench_function("numpy_sort_c64_axis0_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    numpy_sort
                        .call((&a0_c64,), Some(&axis0_kwargs))
                        .expect("numpy sort a0 c64"),
                )
            });
        });
        group.bench_function("fnp_sort_c64_midaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    fnp_sort
                        .call((&am_c64,), Some(&axis1_kwargs))
                        .expect("fnp sort am c64"),
                )
            });
        });
        group.bench_function("numpy_sort_c64_midaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    numpy_sort
                        .call((&am_c64,), Some(&axis1_kwargs))
                        .expect("numpy sort am c64"),
                )
            });
        });
        // datetime64 last-axis argsort: la (16384x1024 distinct-per-lane int64) cast to datetime64[s]
        let la_dt = la
            .call_method1("astype", ("datetime64[s]",))
            .expect("la dt64");
        group.bench_function("fnp_argsort_datetime64_lastaxis_16Mx", |bch| {
            bch.iter(|| black_box(fnp_argsort.call1((&la_dt,)).expect("fnp argsort la dt64")));
        });
        group.bench_function("numpy_argsort_datetime64_lastaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    numpy_argsort
                        .call1((&la_dt,))
                        .expect("numpy argsort la dt64"),
                )
            });
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
            bch.iter(|| {
                black_box(
                    fnp_sort
                        .call((&m3,), Some(&m3_kwargs))
                        .expect("fnp sort m3"),
                )
            });
        });
        group.bench_function("numpy_sort_int64_midaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    numpy_sort
                        .call((&m3,), Some(&m3_kwargs))
                        .expect("numpy sort m3"),
                )
            });
        });
        let m3_dt = m3
            .call_method1("astype", ("datetime64[s]",))
            .expect("m3 dt64");
        group.bench_function("fnp_sort_datetime64_midaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    fnp_sort
                        .call((&m3_dt,), Some(&m3_kwargs))
                        .expect("fnp sort m3 dt"),
                )
            });
        });
        group.bench_function("numpy_sort_datetime64_midaxis_16Mx", |bch| {
            bch.iter(|| {
                black_box(
                    numpy_sort
                        .call((&m3_dt,), Some(&m3_kwargs))
                        .expect("numpy sort m3 dt"),
                )
            });
        });
    });

    group.finish();
}

fn bench_int32_flat_sort_small_pool_regate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_int32_flat_sort_small_pool_regate");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.warm_up_time(Duration::from_millis(250));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let threads = rayon::current_num_threads();
        assert_eq!(threads, 8, "proof row requires RAYON_NUM_THREADS=8");
        #[cfg(target_arch = "x86_64")]
        let avx2 = std::arch::is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let avx2 = false;
        assert!(avx2, "proof row requires NumPy's AVX2 int32 qsort basis");
        eprintln!(
            "INT32_SORT_REGATE host={} rayon_threads={threads} avx2={}",
            std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_owned()),
            avx2
        );

        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_sort = module.getattr("sort").expect("fnp sort");
        let numpy_sort = numpy.getattr("sort").expect("numpy sort");
        let rng = numpy
            .getattr("random")
            .expect("numpy.random")
            .getattr("default_rng")
            .expect("default_rng")
            .call1((7_i64,))
            .expect("rng");
        let n = 8_000_000_usize;
        let input = rng
            .call_method1("integers", (-2_000_000_000_i64, 2_000_000_000_i64, n))
            .expect("int32 raw")
            .call_method1("astype", ("int32",))
            .expect("int32 input");

        // Exact same-data reconstruction of the former production primitive:
        // allocate/copy, then Rayon comparison-sort. It is intentionally a
        // favorable control (Vec clone rather than numpy.empty export), so a
        // loss here is decisive evidence for delegating the small-pool basis.
        let input_bytes: Vec<u8> = input
            .call_method0("tobytes")
            .expect("input bytes")
            .extract()
            .expect("extract input bytes");
        let native_input: Vec<i32> = input_bytes
            .chunks_exact(std::mem::size_of::<i32>())
            .map(|bytes| i32::from_ne_bytes(bytes.try_into().expect("i32 bytes")))
            .collect();
        assert_eq!(native_input.len(), n);

        // Correctness is outside the timed loop: candidate, NumPy, and the old
        // native primitive must produce identical value-sort bytes.
        let fnp_sorted = fnp_sort.call1((&input,)).expect("fnp sort parity");
        let numpy_sorted = numpy_sort.call1((&input,)).expect("numpy sort parity");
        let fnp_bytes: Vec<u8> = fnp_sorted
            .call_method0("tobytes")
            .expect("fnp bytes")
            .extract()
            .expect("extract fnp bytes");
        let numpy_bytes: Vec<u8> = numpy_sorted
            .call_method0("tobytes")
            .expect("numpy bytes")
            .extract()
            .expect("extract numpy bytes");
        assert_eq!(fnp_bytes, numpy_bytes, "regated int32 sort byte mismatch");
        let mut native_sorted = native_input.clone();
        native_sorted.par_sort_unstable();
        assert!(
            native_sorted
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .eq(numpy_bytes.iter().copied()),
            "native int32 control byte mismatch"
        );

        group.bench_function("control_native_int32_8m", |bench| {
            bench.iter(|| {
                let mut output = native_input.clone();
                output.par_sort_unstable();
                black_box(output)
            });
        });
        group.bench_function("fnp_regated_int32_8m", |bench| {
            bench.iter(|| black_box(fnp_sort.call1((&input,)).expect("fnp sort")));
        });
        group.bench_function("numpy_int32_8m", |bench| {
            bench.iter(|| black_box(numpy_sort.call1((&input,)).expect("numpy sort")));
        });
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
            b.iter(|| {
                black_box(
                    fnp_compress
                        .call((&cond, &x), Some(&kw))
                        .expect("fnp compress"),
                )
            });
        });
        group.bench_function("numpy_compress_2d_axis1", |b| {
            b.iter(|| {
                black_box(
                    numpy_compress
                        .call((&cond, &x), Some(&kw2))
                        .expect("np compress"),
                )
            });
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
            b.iter(|| black_box(fnp_roll.call1((&x, shifts, axes)).expect("fnp roll")));
        });
        group.bench_function("numpy_roll_2d_multi_int64", |b| {
            b.iter(|| black_box(numpy_roll.call1((&x, shifts, axes)).expect("np roll")));
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
            bench.iter(|| {
                black_box(
                    fnp_matrix_power
                        .call1((&imp, 5_i64))
                        .expect("fnp int matpow"),
                )
            });
        });
        group.bench_function("numpy_matrix_power_i64_256_n5", |bench| {
            bench.iter(|| {
                black_box(
                    numpy_matrix_power
                        .call1((&imp, 5_i64))
                        .expect("np int matpow"),
                )
            });
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
        let f32_in = base
            .call_method1("astype", ("float32",))
            .expect("f32 input");
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
            b.iter(|| {
                black_box(
                    fnp_clip
                        .call1((&input, -1000.0_f64, 1000.0_f64))
                        .expect("fnp clip"),
                )
            });
        });
        group.bench_function("numpy_clip_f64_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_clip
                        .call1((&input, -1000.0_f64, 1000.0_f64))
                        .expect("numpy clip"),
                )
            });
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
            b.iter(|| {
                black_box(
                    numpy_conv
                        .call1((&a, &v, "full"))
                        .expect("numpy int convolve"),
                )
            });
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
            b.iter(|| black_box(numpy_conv.call1((&a, &v, "same")).expect("numpy convolve")));
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
            bn.iter(|| {
                black_box(
                    fnp_where
                        .call1((&imask, &ia32, &ib32))
                        .expect("fnp where i32"),
                )
            });
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
            b.iter(|| {
                black_box(
                    fnp_around
                        .call1((&input_c, 3_i64))
                        .expect("fnp around c128"),
                )
            });
        });
        group.bench_function("numpy_around_c128_4m", |b| {
            b.iter(|| {
                black_box(
                    numpy_around
                        .call1((&input_c, 3_i64))
                        .expect("numpy around c128"),
                )
            });
        });

        // f32 sibling — compute-heavy (round-ties-even + mul/div) so wins at 4-byte.
        let input32 = input
            .call_method1("astype", ("float32",))
            .expect("f32 input");
        group.bench_function("fnp_around_f32_8m", |b| {
            b.iter(|| black_box(fnp_around.call1((&input32, 3_i64)).expect("fnp around f32")));
        });
        group.bench_function("numpy_around_f32_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_around
                        .call1((&input32, 3_i64))
                        .expect("numpy around f32"),
                )
            });
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
                fnp_pad
                    .call1((arr, 4000_i64, "edge"))
                    .expect("fnp pad edge scalar"),
                numpy_pad
                    .call1((arr, 4000_i64, "edge"))
                    .expect("numpy pad edge scalar"),
            );
            let tuple = (
                fnp_pad
                    .call1((arr, (3_i64, 7_i64), "edge"))
                    .expect("fnp pad edge tuple"),
                numpy_pad
                    .call1((arr, (3_i64, 7_i64), "edge"))
                    .expect("numpy pad edge tuple"),
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
            b.iter(|| {
                black_box(
                    fnp_pad
                        .call1((&x, 4000_i64, "edge"))
                        .expect("fnp pad edge f64"),
                )
            });
        });
        group.bench_function("numpy_pad_edge_f64_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_pad
                        .call1((&x, 4000_i64, "edge"))
                        .expect("numpy pad edge f64"),
                )
            });
        });
        group.bench_function("fnp_pad_edge_i32_8m", |b| {
            b.iter(|| {
                black_box(
                    fnp_pad
                        .call1((&xi, 4000_i64, "edge"))
                        .expect("fnp pad edge i32"),
                )
            });
        });
        group.bench_function("numpy_pad_edge_i32_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_pad
                        .call1((&xi, 4000_i64, "edge"))
                        .expect("numpy pad edge i32"),
                )
            });
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
                fnp_pad
                    .call1((arr, 4000_i64, "wrap"))
                    .expect("fnp pad wrap scalar"),
                numpy_pad
                    .call1((arr, 4000_i64, "wrap"))
                    .expect("numpy pad wrap scalar"),
            );
            let tuple = (
                fnp_pad
                    .call1((arr, (3_i64, 7_i64), "wrap"))
                    .expect("fnp pad wrap tuple"),
                numpy_pad
                    .call1((arr, (3_i64, 7_i64), "wrap"))
                    .expect("numpy pad wrap tuple"),
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
            b.iter(|| {
                black_box(
                    fnp_pad
                        .call1((&x, 4000_i64, "wrap"))
                        .expect("fnp pad wrap f64"),
                )
            });
        });
        group.bench_function("numpy_pad_wrap_f64_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_pad
                        .call1((&x, 4000_i64, "wrap"))
                        .expect("numpy pad wrap f64"),
                )
            });
        });
        group.bench_function("fnp_pad_wrap_i32_8m", |b| {
            b.iter(|| {
                black_box(
                    fnp_pad
                        .call1((&xi, 4000_i64, "wrap"))
                        .expect("fnp pad wrap i32"),
                )
            });
        });
        group.bench_function("numpy_pad_wrap_i32_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_pad
                        .call1((&xi, 4000_i64, "wrap"))
                        .expect("numpy pad wrap i32"),
                )
            });
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
                    numpy_pad
                        .call1((arr, 4000_i64, md))
                        .expect("numpy pad scalar"),
                );
                let tuple = (
                    fnp_pad
                        .call1((arr, (3_i64, 7_i64), md))
                        .expect("fnp pad tuple"),
                    numpy_pad
                        .call1((arr, (3_i64, 7_i64), md))
                        .expect("numpy pad tuple"),
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
            b.iter(|| {
                black_box(
                    fnp_pad
                        .call1((&x, 4000_i64, "reflect"))
                        .expect("fnp reflect f64"),
                )
            });
        });
        group.bench_function("numpy_pad_reflect_f64_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_pad
                        .call1((&x, 4000_i64, "reflect"))
                        .expect("numpy reflect f64"),
                )
            });
        });
        group.bench_function("fnp_pad_symmetric_i32_8m", |b| {
            b.iter(|| {
                black_box(
                    fnp_pad
                        .call1((&xi, 4000_i64, "symmetric"))
                        .expect("fnp symmetric i32"),
                )
            });
        });
        group.bench_function("numpy_pad_symmetric_i32_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_pad
                        .call1((&xi, 4000_i64, "symmetric"))
                        .expect("numpy symmetric i32"),
                )
            });
        });
    });
    group.finish();
}

/// Drives the selected bench group functions via [`common::gated_main`], which
/// gates each on [`common::group_enabled`] (`FNP_BENCH_GROUPS`) and emits the
/// final summary. Replaces the former `gated_benches!` macro; the target list is
/// the same set of group functions in the same order.
fn main() {
    common::gated_main(&[
        ("bench_pad_edge_boundary", bench_pad_edge_boundary),
        ("bench_pad_wrap_boundary", bench_pad_wrap_boundary),
        ("bench_pad_reflect_boundary", bench_pad_reflect_boundary),
        ("bench_kron_boundary", bench_kron_boundary),
        ("bench_nan_to_num_boundary", bench_nan_to_num_boundary),
        ("bench_cross_boundary", bench_cross_boundary),
        ("bench_around_boundary", bench_around_boundary),
        ("bench_where_boundary", bench_where_boundary),
        ("bench_f64_convolve_boundary", bench_f64_convolve_boundary),
        ("bench_int_convolve_boundary", bench_int_convolve_boundary),
        ("bench_clip_boundary", bench_clip_boundary),
        (
            "bench_unary_parallel_boundary",
            bench_unary_parallel_boundary,
        ),
        ("bench_float_isin_boundary", bench_float_isin_boundary),
        (
            "bench_f16_binary_transcendental_boundary",
            bench_f16_binary_transcendental_boundary,
        ),
        ("bench_unique_medium_boundary", bench_unique_medium_boundary),
        ("bench_sort_complex_boundary", bench_sort_complex_boundary),
        ("bench_f16_matmul_boundary", bench_f16_matmul_boundary),
        (
            "bench_flat_sort_dtype_boundary",
            bench_flat_sort_dtype_boundary,
        ),
        (
            "bench_int32_flat_sort_small_pool_regate",
            bench_int32_flat_sort_small_pool_regate,
        ),
        ("bench_compress_boundary", bench_compress_boundary),
        (
            "bench_compress_lastaxis_boundary",
            bench_compress_lastaxis_boundary,
        ),
        ("bench_delete_mask_boundary", bench_delete_mask_boundary),
        ("bench_insert_block_boundary", bench_insert_block_boundary),
        (
            "bench_roll_2d_multi_dtype_boundary",
            bench_roll_2d_multi_dtype_boundary,
        ),
        ("bench_roll_boundary", bench_roll_boundary),
        ("bench_einsum_boundary", bench_einsum_boundary),
        ("bench_linalg_boundary", bench_linalg_boundary),
    ]);
}
