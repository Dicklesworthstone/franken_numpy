//! indexing / gather / binning misc domain criterion benches — f16 ops and
//! setops, tile, digitize, bincount, searchsorted, repeat (axis / array), and the
//! take family (take, take_dtype, take_axis, take_along_axis, c64) — split out of
//! the monolithic `criterion_python_surface.rs` into their own per-domain bench
//! binary. See bead deadlock-audit-x7nnf.

#[path = "common/mod.rs"]
mod common;

use common::ensure_numpy_available;
use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAny, PyAnyMethods, PyDict, PyModule, PyStringMethods};
use std::hint::black_box;
use std::time::{Duration, Instant};

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
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let (u, hs, hq, ia, ib) = (
            ns.get_item("u").unwrap(),
            ns.get_item("hs").unwrap(),
            ns.get_item("hq").unwrap(),
            ns.get_item("ia").unwrap(),
            ns.get_item("ib").unwrap(),
        );
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let (fu, nu_) = (
            module.getattr("unique").unwrap(),
            numpy.getattr("unique").unwrap(),
        );
        let (fss, nss) = (
            module.getattr("searchsorted").unwrap(),
            numpy.getattr("searchsorted").unwrap(),
        );
        let (fi, ni) = (
            module.getattr("isin").unwrap(),
            numpy.getattr("isin").unwrap(),
        );
        assert!(
            eqf.call1((fu.call1((&u,)).unwrap(), nu_.call1((&u,)).unwrap()))
                .unwrap()
                .extract::<bool>()
                .unwrap(),
            "f16 unique mismatch"
        );
        assert!(
            eqf.call1((
                fss.call1((&hs, &hq)).unwrap(),
                nss.call1((&hs, &hq)).unwrap()
            ))
            .unwrap()
            .extract::<bool>()
            .unwrap(),
            "f16 searchsorted mismatch"
        );
        assert!(
            eqf.call1((fi.call1((&ia, &ib)).unwrap(), ni.call1((&ia, &ib)).unwrap()))
                .unwrap()
                .extract::<bool>()
                .unwrap(),
            "f16 isin mismatch"
        );
        group.bench_function("fnp_unique_f16_4m", |bn| {
            bn.iter(|| black_box(fu.call1((&u,)).unwrap()))
        });
        group.bench_function("numpy_unique_f16_4m", |bn| {
            bn.iter(|| black_box(nu_.call1((&u,)).unwrap()))
        });
        group.bench_function("fnp_searchsorted_f16_2m_2m", |bn| {
            bn.iter(|| black_box(fss.call1((&hs, &hq)).unwrap()))
        });
        group.bench_function("numpy_searchsorted_f16_2m_2m", |bn| {
            bn.iter(|| black_box(nss.call1((&hs, &hq)).unwrap()))
        });
        group.bench_function("fnp_isin_f16_2m_1m", |bn| {
            bn.iter(|| black_box(fi.call1((&ia, &ib)).unwrap()))
        });
        group.bench_function("numpy_isin_f16_2m_1m", |bn| {
            bn.iter(|| black_box(ni.call1((&ia, &ib)).unwrap()))
        });
    });
    group.finish();
}

fn bench_f16_searchsorted_table_dual_null(c: &mut Criterion) {
    // Keep the measurement in the shipped Python route: the candidate must build
    // its cumulative table from the same native-endian, sorted float16 haystack a
    // user passes to `fnp.searchsorted`. Both arms allocate the intp result.
    let _ = c;
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module =
            PyModule::new(py, "fnp_python_f16_searchsorted_table_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\\
rng = np.random.default_rng(20260824)\n\\
haystack = np.sort((rng.integers(-3000, 3001, 1_000_000) / 7).astype(np.float16))\n\\
queries = (rng.integers(-3000, 3001, 1_000_000) / 7).astype(np.float16)\n",
            )
            .expect("f16 searchsorted table setup CString")
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("f16 searchsorted table setup");
        let haystack = ns.get_item("haystack").expect("f16 haystack");
        let queries = ns.get_item("queries").expect("f16 queries");
        assert!(
            haystack
                .getattr("flags")
                .expect("haystack flags")
                .getattr("c_contiguous")
                .expect("haystack C-contiguous flag")
                .extract::<bool>()
                .expect("haystack C-contiguous value"),
            "the table row requires a C-contiguous haystack"
        );
        assert_eq!(
            haystack
                .getattr("dtype")
                .expect("haystack dtype")
                .str()
                .expect("haystack dtype string")
                .to_str()
                .expect("UTF-8 dtype"),
            "float16",
            "the table row must use float16"
        );
        let fnp_searchsorted = module.getattr("searchsorted").expect("fnp.searchsorted");
        let numpy_searchsorted = numpy.getattr("searchsorted").expect("numpy.searchsorted");
        assert!(
            !fnp_searchsorted.is(&numpy_searchsorted),
            "dispatch trap: fnp.searchsorted resolved to NumPy"
        );
        common::report_numpy_incumbent_identity(py, "searchsorted", &numpy_searchsorted);

        for query_elements in [4_096_usize, 8_192, 16_384, 32_768, 65_536, 1_000_000] {
            let query = queries
                .call_method1(
                    "__getitem__",
                    (pyo3::types::PySlice::new(py, 0, query_elements as isize, 1),),
                )
                .expect("query prefix");
            for side in ["left", "right"] {
                let kwargs = PyDict::new(py);
                kwargs.set_item("side", side).expect("searchsorted side");
                let run_incumbent = || {
                    numpy_searchsorted
                        .call((&haystack, &query), Some(&kwargs))
                        .expect("NumPy float16 searchsorted arm")
                };
                let run_candidate = || {
                    fnp_searchsorted
                        .call((&haystack, &query), Some(&kwargs))
                        .expect("FrankenNumPy float16 searchsorted arm")
                };

                // Full output equality proves the candidate is the user-visible
                // table route before either arm enters the timing schedule.
                let expected = run_incumbent();
                let actual = run_candidate();
                assert_eq!(
                    actual
                        .getattr("dtype")
                        .expect("candidate dtype")
                        .str()
                        .expect("candidate dtype string")
                        .to_str()
                        .expect("candidate dtype UTF-8"),
                    expected
                        .getattr("dtype")
                        .expect("incumbent dtype")
                        .str()
                        .expect("incumbent dtype string")
                        .to_str()
                        .expect("incumbent dtype UTF-8"),
                    "f16 searchsorted {side} dtype differs from NumPy"
                );
                assert_eq!(
                    actual
                        .call_method0("tobytes")
                        .expect("candidate bytes")
                        .extract::<Vec<u8>>()
                        .expect("candidate byte vector"),
                    expected
                        .call_method0("tobytes")
                        .expect("incumbent bytes")
                        .extract::<Vec<u8>>()
                        .expect("incumbent byte vector"),
                    "f16 searchsorted {side} differs from NumPy"
                );
                let checksum = |result: &pyo3::Bound<'_, PyAny>| {
                    result
                        .call_method0("tobytes")
                        .expect("searchsorted result bytes")
                        .extract::<Vec<u8>>()
                        .expect("searchsorted result byte vector")
                        .into_iter()
                        .fold(0xcbf2_9ce4_8422_u64, |state, byte| {
                            (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                        })
                };
                let row =
                    format!("python_f16_searchsorted_table_1m_{query_elements}_{side}_vs_numpy");
                println!(
                    "PARITY row={row} exact_bytes=passed haystack_elements=1000000 \\
                     query_elements={query_elements} candidate_route=f16_cumulative_table \\
                     output_allocation=both_arms_intp"
                );
                let mut observe_incumbent = || {
                    let started = Instant::now();
                    let result = run_incumbent();
                    let elapsed = started.elapsed();
                    common::ContractObservation {
                        elapsed,
                        checksum: checksum(&result),
                    }
                };
                let mut observe_candidate = || {
                    let started = Instant::now();
                    let result = run_candidate();
                    let elapsed = started.elapsed();
                    common::ContractObservation {
                        elapsed,
                        checksum: checksum(&result),
                    }
                };
                let (effect, incumbent_null, candidate_null) =
                    common::run_dual_null_median_ci_contract_with_sampling(
                        &row,
                        &mut observe_incumbent,
                        &mut observe_candidate,
                        9,
                        1,
                    );
                let verdict =
                    common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
                println!(
                    "F16_SEARCHSORTED_TABLE_RESULT row={row} side={side} verdict={verdict} \\
                 incumbent_median_ms={:.6} candidate_median_ms={:.6} ratio_median={:.6} \\
                 ratio_ci95=[{:.6},{:.6}] incumbent_null_ratio={:.6} \\
                 incumbent_null_ci95=[{:.6},{:.6}] candidate_null_ratio={:.6} \\
                 candidate_null_ci95=[{:.6},{:.6}] incumbent=numpy_live_same_invocation \\
                 dual_nulls=required",
                    effect.arm_a_median_ns / 1_000_000.0,
                    effect.arm_b_median_ns / 1_000_000.0,
                    effect.ratio_median,
                    effect.ratio_ci_low,
                    effect.ratio_ci_high,
                    incumbent_null.ratio_median,
                    incumbent_null.ratio_ci_low,
                    incumbent_null.ratio_ci_high,
                    candidate_null.ratio_median,
                    candidate_null.ratio_ci_low,
                    candidate_null.ratio_ci_high,
                );
            }
        }
    });
}

fn bench_f64_reduction_dot_dual_null(c: &mut Criterion) {
    // Refresh the old cross-engine snapshot in the public Python surface.  The
    // candidate is fnp_python's actual exported callable; the incumbent is the
    // live NumPy callable imported into this very interpreter.  The scalar bit
    // pattern is checked before the timing contract, so a fast but different
    // summation tree cannot produce a benchmark row.
    let _ = c;
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_f64_reduction_dot_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\\
sum_mean_values = np.linspace(-3.0, 7.0, 1_000_000, dtype=np.float64)\n\\
dot_lhs = np.linspace(-1.0, 2.0, 10_000, dtype=np.float64)\n\\
dot_rhs = np.linspace(3.0, -2.0, 10_000, dtype=np.float64)\n",
            )
            .expect("f64 reduction/dot setup CString")
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("f64 reduction/dot setup");
        let values = ns.get_item("sum_mean_values").expect("f64 sum/mean values");
        let dot_lhs = ns.get_item("dot_lhs").expect("f64 dot lhs");
        let dot_rhs = ns.get_item("dot_rhs").expect("f64 dot rhs");

        for (operation, incumbent_name, candidate, incumbent, args) in [
            (
                "sum_1m",
                "sum",
                module.getattr("sum").expect("fnp.sum"),
                numpy.getattr("sum").expect("numpy.sum"),
                pyo3::types::PyTuple::new(py, [&values]).expect("sum arguments"),
            ),
            (
                "mean_1m",
                "mean",
                module.getattr("mean").expect("fnp.mean"),
                numpy.getattr("mean").expect("numpy.mean"),
                pyo3::types::PyTuple::new(py, [&values]).expect("mean arguments"),
            ),
            (
                "dot_10k",
                "dot",
                module.getattr("dot").expect("fnp.dot"),
                numpy.getattr("dot").expect("numpy.dot"),
                pyo3::types::PyTuple::new(py, [&dot_lhs, &dot_rhs]).expect("dot arguments"),
            ),
        ] {
            assert!(
                !candidate.is(&incumbent),
                "dispatch trap: fnp.{operation} resolved to NumPy"
            );
            common::report_numpy_incumbent_identity(py, incumbent_name, &incumbent);
            let expected = incumbent.call1(&args).expect("NumPy result");
            let actual = candidate.call1(&args).expect("FrankenNumPy result");
            let expected_bits = expected
                .extract::<f64>()
                .expect("NumPy f64 scalar")
                .to_bits();
            let actual_bits = actual
                .extract::<f64>()
                .expect("FrankenNumPy f64 scalar")
                .to_bits();
            assert_eq!(
                actual_bits, expected_bits,
                "f64 {operation} must match NumPy's scalar bit pattern"
            );
            let row = format!("python_f64_{operation}_vs_numpy");
            println!(
                "PARITY row={row} exact_scalar_bits=passed candidate_public_surface=fnp_python \\
                 incumbent=numpy_live_same_invocation"
            );
            let mut observe_incumbent = || {
                let started = Instant::now();
                let result = incumbent.call1(&args).expect("NumPy arm");
                common::ContractObservation {
                    elapsed: started.elapsed(),
                    checksum: result.extract::<f64>().expect("NumPy scalar").to_bits(),
                }
            };
            let mut observe_candidate = || {
                let started = Instant::now();
                let result = candidate.call1(&args).expect("FrankenNumPy arm");
                common::ContractObservation {
                    elapsed: started.elapsed(),
                    checksum: result
                        .extract::<f64>()
                        .expect("FrankenNumPy scalar")
                        .to_bits(),
                }
            };
            let (effect, incumbent_null, candidate_null) = common::run_dual_null_median_ci_contract(
                &row,
                &mut observe_incumbent,
                &mut observe_candidate,
            );
            let verdict =
                common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
            println!(
                "F64_REDUCTION_DOT_RESULT row={row} verdict={verdict} \\
                 incumbent_median_ns={:.1} candidate_median_ns={:.1} ratio_median={:.6} \\
                 ratio_ci95=[{:.6},{:.6}] incumbent_null_ratio={:.6} \\
                 incumbent_null_ci95=[{:.6},{:.6}] candidate_null_ratio={:.6} \\
                 candidate_null_ci95=[{:.6},{:.6}] dual_nulls=required",
                effect.arm_a_median_ns,
                effect.arm_b_median_ns,
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                incumbent_null.ratio_median,
                incumbent_null.ratio_ci_low,
                incumbent_null.ratio_ci_high,
                candidate_null.ratio_median,
                candidate_null.ratio_ci_low,
                candidate_null.ratio_ci_high,
            );
        }
    });
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
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let a = ns.get_item("a").expect("a");
        let b = ns.get_item("b").expect("b");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for op in ["union1d", "intersect1d", "setdiff1d", "setxor1d"] {
            let fnp_fn = module.getattr(op).expect("fnp op");
            let np_fn = numpy.getattr(op).expect("numpy op");
            let f = fnp_fn.call1((&a, &b)).expect("fnp setop");
            let n = np_fn.call1((&a, &b)).expect("numpy setop");
            assert!(
                eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
                "{op} f16 mismatch"
            );
            group.bench_function(format!("fnp_{op}_f16_2m_2m"), |bn| {
                bn.iter(|| black_box(fnp_fn.call1((&a, &b)).unwrap()))
            });
            group.bench_function(format!("numpy_{op}_f16_2m_2m"), |bn| {
                bn.iter(|| black_box(np_fn.call1((&a, &b)).unwrap()))
            });
        }
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
            b.iter(|| {
                black_box(
                    numpy_bc
                        .call((&x_k1000,), Some(&kw2))
                        .expect("np wbincount"),
                )
            });
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
            b.iter(|| {
                black_box(
                    fnp_ss
                        .call1((&a_big, &v_big))
                        .expect("fnp searchsorted big"),
                )
            });
        });
        group.bench_function("numpy_searchsorted_f64_4m_haystack4m", |b| {
            b.iter(|| {
                black_box(
                    numpy_ss
                        .call1((&a_big, &v_big))
                        .expect("numpy searchsorted big"),
                )
            });
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
                let got = fnp_ss
                    .call((&a_f32, &v_f32), Some(&kw))
                    .expect("fnp ss f32");
                let exp = numpy_ss
                    .call((&a_f32, &v_f32), Some(&kw))
                    .expect("np ss f32");
                let eq: bool = np_array_equal
                    .call1((&got, &exp))
                    .expect("array_equal")
                    .extract()
                    .expect("bool");
                assert!(
                    eq,
                    "searchsorted f32 merge correctness mismatch: side={side}"
                );
            }
        }
        group.bench_function("fnp_searchsorted_f32_4m_haystack4m", |b| {
            b.iter(|| {
                black_box(
                    fnp_ss
                        .call1((&a_f32, &v_f32))
                        .expect("fnp searchsorted f32"),
                )
            });
        });
        group.bench_function("numpy_searchsorted_f32_4m_haystack4m", |b| {
            b.iter(|| {
                black_box(
                    numpy_ss
                        .call1((&a_f32, &v_f32))
                        .expect("numpy searchsorted f32"),
                )
            });
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
                let got = fnp_ss
                    .call((&a_i64, &v_i64), Some(&kw))
                    .expect("fnp ss i64");
                let exp = numpy_ss
                    .call((&a_i64, &v_i64), Some(&kw))
                    .expect("np ss i64");
                let eq: bool = np_array_equal
                    .call1((&got, &exp))
                    .expect("array_equal")
                    .extract()
                    .expect("bool");
                assert!(
                    eq,
                    "searchsorted i64 merge correctness mismatch: side={side}"
                );
            }
        }
        group.bench_function("fnp_searchsorted_i64_4m_haystack4m", |b| {
            b.iter(|| {
                black_box(
                    fnp_ss
                        .call1((&a_i64, &v_i64))
                        .expect("fnp searchsorted i64"),
                )
            });
        });
        group.bench_function("numpy_searchsorted_i64_4m_haystack4m", |b| {
            b.iter(|| {
                black_box(
                    numpy_ss
                        .call1((&a_i64, &v_i64))
                        .expect("numpy searchsorted i64"),
                )
            });
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
            b.iter(|| {
                black_box(
                    numpy_repeat
                        .call((&m, 3_i64), Some(&kw2))
                        .expect("np repeat"),
                )
            });
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
        let fnp_ta = module
            .getattr("take_along_axis")
            .expect("fnp take_along_axis");
        let numpy_ta = numpy
            .getattr("take_along_axis")
            .expect("numpy take_along_axis");
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
        let fnp_t = module
            .getattr("take_along_axis")
            .expect("fnp take_along_axis");
        let numpy_t = numpy
            .getattr("take_along_axis")
            .expect("numpy take_along_axis");
        let axis = 1_i64;
        group.bench_function("fnp_take_along_axis_f64_8m", |b| {
            b.iter(|| black_box(fnp_t.call1((&a, &idx, axis)).expect("fnp take_along_axis")));
        });
        group.bench_function("numpy_take_along_axis_f64_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_t
                        .call1((&a, &idx, axis))
                        .expect("numpy take_along_axis"),
                )
            });
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
            b.iter(|| {
                black_box(
                    fnp_take
                        .call((&a, &idx), Some(&kwargs))
                        .expect("fnp take axis"),
                )
            });
        });
        group.bench_function("numpy_take_axis1_f64_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_take
                        .call((&a, &idx), Some(&kwargs))
                        .expect("numpy take axis"),
                )
            });
        });
    });

    group.finish();
}

fn main() {
    common::gated_main_with_source(
        include_str!("criterion_python_misc1.rs"),
        &[
            ("bench_f16_ops_boundary", bench_f16_ops_boundary),
            (
                "bench_f16_searchsorted_table_dual_null",
                bench_f16_searchsorted_table_dual_null,
            ),
            (
                "bench_f64_reduction_dot_dual_null",
                bench_f64_reduction_dot_dual_null,
            ),
            ("bench_f16_setops_boundary", bench_f16_setops_boundary),
            ("bench_tile_boundary", bench_tile_boundary),
            ("bench_digitize_boundary", bench_digitize_boundary),
            ("bench_bincount_boundary", bench_bincount_boundary),
            ("bench_searchsorted_boundary", bench_searchsorted_boundary),
            ("bench_repeat_axis_boundary", bench_repeat_axis_boundary),
            ("bench_repeat_array_boundary", bench_repeat_array_boundary),
            ("bench_take_dtype_boundary", bench_take_dtype_boundary),
            ("bench_take_boundary", bench_take_boundary),
            (
                "bench_take_along_axis_c64_boundary",
                bench_take_along_axis_c64_boundary,
            ),
            (
                "bench_take_along_axis_boundary",
                bench_take_along_axis_boundary,
            ),
            ("bench_take_axis_boundary", bench_take_axis_boundary),
        ],
    );
}
