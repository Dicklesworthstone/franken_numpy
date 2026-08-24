//! sort/histogram misc domain criterion benches — integer median/percentile
//! histograms, stable argsort for temporal-complex / string / struct keys, and
//! array-API unique (the interspersed leftovers between the extracted domains) —
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
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let fnp_m = module.getattr("median").expect("fnp median");
        let numpy_m = numpy.getattr("median").expect("numpy median");
        for name in ["even_i64", "odd_i64", "i16"] {
            let arr = ns.get_item(name).expect("arr");
            let f = fnp_m.call1((&arr,)).expect("fnp median");
            let n = numpy_m.call1((&arr,)).expect("numpy median");
            let eq: bool = numpy
                .getattr("equal")
                .unwrap()
                .call1((&f, &n))
                .unwrap()
                .extract()
                .unwrap();
            assert!(eq, "median {name} mismatch: fnp {:?} numpy {:?}", f, n);
        }
        let ev = ns.get_item("even_i64").expect("ev");
        let i16a = ns.get_item("i16").expect("i16a");
        group.bench_function("fnp_median_i64_dense_16m", |bn| {
            bn.iter(|| black_box(fnp_m.call1((&ev,)).unwrap()))
        });
        group.bench_function("numpy_median_i64_dense_16m", |bn| {
            bn.iter(|| black_box(numpy_m.call1((&ev,)).unwrap()))
        });
        group.bench_function("fnp_median_i16_dense_16m", |bn| {
            bn.iter(|| black_box(fnp_m.call1((&i16a,)).unwrap()))
        });
        group.bench_function("numpy_median_i16_dense_16m", |bn| {
            bn.iter(|| black_box(numpy_m.call1((&i16a,)).unwrap()))
        });
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
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let fnp_percentile = module.getattr("percentile").expect("fnp percentile");
        let numpy_percentile = numpy.getattr("percentile").expect("numpy percentile");
        let fnp_quantile = module.getattr("quantile").expect("fnp quantile");
        let numpy_quantile = numpy.getattr("quantile").expect("numpy quantile");
        let i64_arr = ns.get_item("i64").expect("i64");
        let u16_arr = ns.get_item("u16").expect("u16");
        for (name, arr, p, q) in [
            ("i64", &i64_arr, 12.5_f64, 0.125_f64),
            ("u16", &u16_arr, 75.0, 0.75),
        ] {
            let fp = fnp_percentile.call1((arr, p)).expect("fnp percentile");
            let np = numpy_percentile.call1((arr, p)).expect("numpy percentile");
            let fq = fnp_quantile.call1((arr, q)).expect("fnp quantile");
            let nq = numpy_quantile.call1((arr, q)).expect("numpy quantile");
            let eq_p: bool = numpy
                .getattr("array_equal")
                .unwrap()
                .call1((&fp, &np))
                .unwrap()
                .extract()
                .unwrap();
            let eq_q: bool = numpy
                .getattr("array_equal")
                .unwrap()
                .call1((&fq, &nq))
                .unwrap()
                .extract()
                .unwrap();
            assert!(
                eq_p,
                "percentile {name} mismatch: fnp {:?} numpy {:?}",
                fp, np
            );
            assert!(
                eq_q,
                "quantile {name} mismatch: fnp {:?} numpy {:?}",
                fq, nq
            );
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
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
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
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let u = ns.get_item("u").expect("u");
        let s = ns.get_item("s").expect("s");
        let fnp_as = module.getattr("argsort").expect("fnp argsort");
        let numpy_as = numpy.getattr("argsort").expect("numpy argsort");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for (arr, label) in [(&u, "U6"), (&s, "S6")] {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            let f = fnp_as.call((arr,), Some(&kw)).expect("fnp argsort");
            let n = numpy_as.call((arr,), Some(&kw)).expect("numpy argsort");
            assert!(
                eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
                "argsort {label} stable mismatch"
            );
        }
        group.bench_function("fnp_argsort_U6_stable_2m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&u,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_U6_stable_2m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(numpy_as.call((&u,), Some(&kw)).unwrap()));
        });
        group.bench_function("fnp_argsort_S6_stable_2m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&s,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_S6_stable_2m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
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
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_as = module.getattr("argsort").expect("fnp argsort");
        let numpy_as = numpy.getattr("argsort").expect("numpy argsort");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let kw = PyDict::new(py);
        kw.set_item("kind", "stable").unwrap();
        let f = fnp_as.call((&a,), Some(&kw)).expect("fnp argsort");
        let n = numpy_as.call((&a,), Some(&kw)).expect("numpy argsort");
        assert!(
            eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
            "argsort struct stable mismatch"
        );
        group.bench_function("fnp_argsort_struct_i8f8_2m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_struct_i8f8_2m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
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
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let s = ns.get_item("s").expect("s");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        // correctness: compare each namedtuple field of unique_counts and unique_all
        for op in ["unique_counts", "unique_all"] {
            let fnp_fn = module.getattr(op).expect("fnp op");
            let np_fn = numpy.getattr(op).expect("numpy op");
            let f = fnp_fn
                .call1((&s,))
                .expect("fnp")
                .cast_into::<pyo3::types::PyTuple>()
                .unwrap();
            let n = np_fn
                .call1((&s,))
                .expect("numpy")
                .cast_into::<pyo3::types::PyTuple>()
                .unwrap();
            let nfields = if op == "unique_counts" { 2 } else { 4 };
            for i in 0..nfields {
                let eq: bool = eqf
                    .call1((f.get_item(i).unwrap(), n.get_item(i).unwrap()))
                    .unwrap()
                    .extract()
                    .unwrap();
                assert!(eq, "{op} field {i} mismatch");
            }
        }
        let fnp_uc = module.getattr("unique_counts").unwrap();
        let np_uc = numpy.getattr("unique_counts").unwrap();
        let fnp_ua = module.getattr("unique_all").unwrap();
        let np_ua = numpy.getattr("unique_all").unwrap();
        group.bench_function("fnp_unique_counts_U8_2m", |bn| {
            bn.iter(|| black_box(fnp_uc.call1((&s,)).unwrap()))
        });
        group.bench_function("numpy_unique_counts_U8_2m", |bn| {
            bn.iter(|| black_box(np_uc.call1((&s,)).unwrap()))
        });
        group.bench_function("fnp_unique_all_U8_2m", |bn| {
            bn.iter(|| black_box(fnp_ua.call1((&s,)).unwrap()))
        });
        group.bench_function("numpy_unique_all_U8_2m", |bn| {
            bn.iter(|| black_box(np_ua.call1((&s,)).unwrap()))
        });
    });
    group.finish();
}

fn main() {
    common::gated_main_with_source(
        include_str!("criterion_python_sortmisc.rs"),
        &[
            (
                "bench_median_int_histogram_boundary",
                bench_median_int_histogram_boundary,
            ),
            (
                "bench_int_percentile_quantile_histogram_boundary",
                bench_int_percentile_quantile_histogram_boundary,
            ),
            (
                "bench_argsort_temporal_complex_stable_boundary",
                bench_argsort_temporal_complex_stable_boundary,
            ),
            (
                "bench_argsort_string_stable_boundary",
                bench_argsort_string_stable_boundary,
            ),
            (
                "bench_argsort_struct_stable_boundary",
                bench_argsort_struct_stable_boundary,
            ),
            (
                "bench_unique_arrayapi_boundary",
                bench_unique_arrayapi_boundary,
            ),
            (
                "bench_flat_f64_sort_median_gate",
                bench_flat_f64_sort_median_gate,
            ),
            (
                "bench_flat_f64_unique_median_gate",
                bench_flat_f64_unique_median_gate,
            ),
            (
                "bench_flat_i64_sort_256_dual_null",
                bench_flat_i64_sort_256_dual_null,
            ),
            (
                "bench_flat_i64_sort_256_allocation_control",
                bench_flat_i64_sort_256_allocation_control,
            ),
        ],
    );
}

/// Full `fnp.sort` dispatch versus the live `numpy.sort` incumbent for the
/// small integer cell that the retired counter harness could not certify.
///
/// The preflight spy is outside timing and records whether this build still
/// delegates to `numpy.sort`; the timed arms are always the public callables,
/// so both the baseline and the cutoff build charge the shipped route.
fn bench_flat_i64_sort_256_dual_null(_c: &mut criterion::Criterion) {
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_flat_i64_sort").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python flat-i64-sort module");
        let numpy = py.import("numpy").expect("numpy incumbent");
        let numpy_version: String = numpy
            .getattr("__version__")
            .expect("numpy version")
            .extract()
            .expect("numpy version string");
        assert_eq!(
            numpy_version, "2.4.3",
            "this benchmark is pinned to the live NumPy 2.4.3 incumbent"
        );
        let build_route =
            std::env::var("FNP_BENCH_BUILD_ROUTE").unwrap_or_else(|_| "unreported".to_owned());
        let np_sort = numpy.getattr("sort").expect("numpy.sort");
        let fnp_sort = module.getattr("sort").expect("fnp.sort");
        assert!(
            !fnp_sort.is(&np_sort),
            "dispatch trap: fnp.sort resolved to the NumPy callable"
        );
        common::report_numpy_incumbent_identity(py, "sort", &np_sort);
        common::report_incumbent_topology_with_shared_component(
            "fnp.sort",
            "numpy.sort",
            "per_call_output_allocation",
        );

        let setup = "import numpy as np\n\\
rng = np.random.default_rng(20260824)\n\\
a = rng.integers(-(1 << 62), 1 << 62, 256, dtype=np.int64)\n\\
a[:16] = np.array([np.iinfo(np.int64).min, np.iinfo(np.int64).max, -1, 0, 1, -1, 0, 1, -7, 7, -7, 7, 13, 13, -13, -13], dtype=np.int64)\n\\
rng.shuffle(a)\n";
        let ns = PyDict::new(py);
        ns.set_item("fnp", &module).expect("bind fnp module");
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("int64 corpus setup");
        let input = ns.get_item("a").expect("int64 corpus present");
        assert_eq!(
            input.getattr("size").unwrap().extract::<usize>().unwrap(),
            256,
            "the contract cell is exactly n=256"
        );
        assert_eq!(
            input.getattr("dtype").unwrap().str().unwrap().to_string(),
            "int64",
            "the contract cell is exactly int64"
        );
        assert!(
            input
                .getattr("flags")
                .unwrap()
                .getattr("c_contiguous")
                .unwrap()
                .extract::<bool>()
                .unwrap(),
            "the contract cell is C contiguous"
        );

        // Detect engagement without charging the spy to either timed arm.
        py.run(
            std::ffi::CString::new(
                "original_sort = np.sort\n\\
fnp_sort_calls = []\n\\
def sort_spy(*args, **kwargs):\n\\
    fnp_sort_calls.append(1)\n\\
    return original_sort(*args, **kwargs)\n\\
np.sort = sort_spy\n\\
try:\n\\
    route_probe = fnp.sort(a)\n\\
finally:\n\\
    np.sort = original_sort\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("route engagement probe");
        let candidate_numpy_sort_calls = ns
            .get_item("fnp_sort_calls")
            .expect("route probe calls")
            .len()
            .expect("route probe call count");

        let run_incumbent = || np_sort.call1((black_box(&input),)).expect("NumPy sort arm");
        let run_candidate = || {
            fnp_sort
                .call1((black_box(&input),))
                .expect("FrankenNumPy sort arm")
        };
        let ours = run_candidate();
        let theirs = run_incumbent();
        assert!(
            ours.get_type().is(theirs.get_type()),
            "int64 n=256 sorted result type differs from NumPy"
        );
        assert_eq!(
            ours.getattr("dtype").unwrap().str().unwrap().to_string(),
            theirs.getattr("dtype").unwrap().str().unwrap().to_string(),
            "int64 n=256 sorted dtype differs from NumPy"
        );
        assert_eq!(
            ours.call_method0("tobytes")
                .unwrap()
                .extract::<Vec<u8>>()
                .unwrap(),
            theirs
                .call_method0("tobytes")
                .unwrap()
                .extract::<Vec<u8>>()
                .unwrap(),
            "int64 n=256 sort is not byte-exact versus NumPy"
        );
        let checksum_of = |result: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
            result
                .call_method0("tobytes")
                .unwrap()
                .extract::<Vec<u8>>()
                .unwrap()
                .iter()
                .fold(0xcbf2_9ce4_8422_2325_u64, |state, &byte| {
                    (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                })
        };
        let candidate_route = if candidate_numpy_sort_calls == 0 {
            "native_int64_small_sort"
        } else {
            "numpy_sort_passthrough"
        };
        let row = "python_flat_i64_sort_n256_vs_numpy";
        println!(
            "PARITY row={row} exact_bytes=passed exact_dtype=passed numpy_version={numpy_version} \\
             input_elements=256 input_bytes=2048 checksum={:016x}",
            checksum_of(&theirs)
        );
        println!(
            "ROUTE_ENGAGEMENT row={row} candidate_route={candidate_route} \\
             candidate_numpy_sort_calls_preflight={candidate_numpy_sort_calls} \\
             axis=default dtype=int64 c_contiguous=true nan_placement=not_applicable"
        );
        println!(
            "ALLOCATION_PARITY row={row} incumbent_output_allocation=per_call \\
             candidate_output_allocation=per_call timed_path=public_callable"
        );

        let mut observe_incumbent = || {
            let started = std::time::Instant::now();
            let result = run_incumbent();
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: checksum_of(&result),
            }
        };
        let mut observe_candidate = || {
            let started = std::time::Instant::now();
            let result = run_candidate();
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: checksum_of(&result),
            }
        };
        let (effect, incumbent_null, candidate_null) = common::run_dual_null_median_ci_contract(
            row,
            &mut observe_incumbent,
            &mut observe_candidate,
        );
        let verdict = common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
        println!(
            "FLAT_I64_SORT_RESULT row={row} verdict={verdict} \\
             incumbent_median_ns={:.3} candidate_median_ns={:.3} \\
             ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \\
             incumbent_null_ratio={:.6} incumbent_null_ci95=[{:.6},{:.6}] \\
             candidate_null_ratio={:.6} candidate_null_ci95=[{:.6},{:.6}] \\
             incumbent=numpy_live_same_invocation build_route={build_route}",
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
    });
}

/// Measure the reached allocation call changed by the small-int64 route.
///
/// This is a same-binary maintenance control, not an incumbent comparison:
/// both arms allocate the identical fresh C-contiguous `int64` output and
/// differ only in whether dtype travels through a kwargs dict or NumPy's second
/// positional parameter.  It prices the allocation slice without pretending
/// the kernel or Python entry path executed in this control.
fn bench_flat_i64_sort_256_allocation_control(_c: &mut criterion::Criterion) {
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let numpy = py.import("numpy").expect("numpy incumbent");
        let n = 256usize;
        let kwargs = || {
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("dtype", "int64")
                .expect("set int64 dtype keyword");
            numpy
                .call_method("empty", (n,), Some(&kwargs))
                .expect("kwargs int64 allocation")
        };
        let positional = || {
            numpy
                .call_method1("empty", (n, "int64"))
                .expect("positional int64 allocation")
        };
        let kw_probe = kwargs();
        let positional_probe = positional();
        for result in [&kw_probe, &positional_probe] {
            assert_eq!(
                result.getattr("dtype").unwrap().str().unwrap().to_string(),
                "int64",
                "allocation control changed dtype"
            );
            assert_eq!(
                result
                    .getattr("shape")
                    .unwrap()
                    .extract::<Vec<usize>>()
                    .unwrap(),
                vec![n],
                "allocation control changed shape"
            );
            assert!(
                result
                    .getattr("flags")
                    .unwrap()
                    .getattr("c_contiguous")
                    .unwrap()
                    .extract::<bool>()
                    .unwrap(),
                "allocation control changed layout"
            );
        }
        let mut observe_kwargs = || {
            let started = std::time::Instant::now();
            let _result = kwargs();
            common::ContractObservation {
                elapsed: started.elapsed(),
                checksum: n as u64,
            }
        };
        let mut observe_positional = || {
            let started = std::time::Instant::now();
            let _result = positional();
            common::ContractObservation {
                elapsed: started.elapsed(),
                checksum: n as u64,
            }
        };
        let (effect, kwargs_null, positional_null) = common::run_dual_null_median_ci_contract(
            "python_flat_i64_sort_n256_allocation_positional_over_kwargs",
            &mut observe_positional,
            &mut observe_kwargs,
        );
        let verdict = common::dual_null_contract_verdict(effect, kwargs_null, positional_null);
        println!(
            "FLAT_I64_SORT_ALLOCATION_CONTROL n={n} verdict={verdict} \\
             positional_median_ns={:.3} kwargs_median_ns={:.3} \\
             positional_over_kwargs={:.6} ci95=[{:.6},{:.6}] \\
             kwargs_null={:.6} kwargs_null_ci95=[{:.6},{:.6}] \\
             positional_null={:.6} positional_null_ci95=[{:.6},{:.6}] \\
             same_binary=true same_result_dtype_shape_layout=true \\
             scope=reached_result_allocation_only_not_full_sort",
            effect.arm_a_median_ns,
            effect.arm_b_median_ns,
            effect.ratio_median,
            effect.ratio_ci_low,
            effect.ratio_ci_high,
            kwargs_null.ratio_median,
            kwargs_null.ratio_ci_low,
            kwargs_null.ratio_ci_high,
            positional_null.ratio_median,
            positional_null.ratio_ci_low,
            positional_null.ratio_ci_high,
        );
    });
}

/// Flat `float64` `np.unique` against live NumPy, same invocation, swept across
/// TIE DENSITY.
///
/// `try_zerocopy_f64_unique_flat` carries the same blanket AVX2 surrender the
/// flat-sort route carried before its regate, and its recorded basis is
/// `0.995x on distinct data and 0.547x on dense ties`. Those two numbers point
/// at different mechanisms, so this group sweeps the axis the loss is claimed
/// to vary with rather than inheriting the sort verdict
/// (deadlock-audit-f64-unique-flat-avx2-surrender-hey9a).
///
/// Incumbent topology measured on this host before any candidate existed
/// (numpy 2.4.3, host 9.5-11.4% busy): `numpy.unique` is single-threaded at
/// EVERY size and tie density sampled — cpu/wall 0.997-1.000 at n = 4M/16M/64M
/// across 4M/1M/64/2 distinct values — and the underlying `np.sort` is 73% of
/// the job on distinct input and 87% on dense ties. So dense ties do not buy
/// NumPy a threading layer; they only make its sort cheaper in absolute terms.
///
/// Route-qualified corpora only (no NaN, not both signed zeros, C-contiguous
/// f64, n above `1<<20`); the deferral regimes are covered by conformance.
fn bench_flat_f64_unique_median_gate(_c: &mut criterion::Criterion) {
    const REQUIRED_BUILD_PROFILE: &str = "release-perf";
    const CONTRACT_ROUNDS: usize = 21;
    const CONTRACT_MIN_OF: usize = 1;
    const THREAD_ACTIVITY_REPETITIONS: usize = 3;

    assert_eq!(
        std::env::var("FNP_BENCH_PROFILE").as_deref(),
        Ok(REQUIRED_BUILD_PROFILE),
        "ship-grade flat-unique evidence requires FNP_BENCH_PROFILE=release-perf"
    );
    let build_worker =
        std::env::var("FNP_BUILD_WORKER").expect("FNP_BUILD_WORKER records the build origin");
    assert!(
        !build_worker.trim().is_empty(),
        "FNP_BUILD_WORKER must be set"
    );
    let threads = std::env::var("RAYON_NUM_THREADS")
        .expect("RAYON_NUM_THREADS must be explicitly pinned before flat-unique timing");
    for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok("1"),
            "{variable} must be one: neither unique arm calls BLAS"
        );
    }
    let threads: usize = threads.parse().expect("thread count is numeric");
    assert_eq!(
        rayon::current_num_threads(),
        threads,
        "Rayon pool width does not match the pinned flat-unique configuration"
    );
    // Deliberately only 2, unlike the flat-sort sibling's 8: this group exists to
    // FIND the worker floor for the unique route, so the low-thread rows are part
    // of the measurement, not a misconfiguration. A row whose route defers reports
    // ratio ~1.0 with tight nulls and can never be read as a win.
    assert!(
        threads >= 2,
        "the flat-f64 unique arm requires at least 2 workers; below that the route \
         defers by construction and the group would measure NumPy against itself"
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module =
            PyModule::new(py, "fnp_python_flat_f64_unique").expect("flat-unique bench module");
        fnp_python(&module).expect("initialize fnp_python flat-unique module");
        let numpy = py.import("numpy").expect("numpy incumbent");

        // The candidate allocates its output through numpy.empty inside the timed
        // region — NumPy code inside the candidate arm, disclosed rather than
        // implied absent. The incumbent allocates its own result too, so the
        // disclosure is conservative.
        common::report_incumbent_topology_with_shared_component(
            "fnp.unique",
            "numpy.unique",
            "numpy.empty_output_allocation",
        );
        println!("NUMPY_BUILD_CONFIG_BEGIN workload=flat_f64_unique");
        numpy
            .getattr("show_config")
            .expect("numpy.show_config")
            .call0()
            .expect("report NumPy build configuration");
        println!("NUMPY_BUILD_CONFIG_END workload=flat_f64_unique");
        println!(
            "BLAS_RELEVANCE workload=flat_f64_unique numpy_unique_uses_blas=false \
             candidate_uses_blas=false blas_threads_pinned=1 reason=sort_dedup_no_gemm"
        );

        let np_unique = numpy.getattr("unique").expect("numpy.unique");
        let fnp_unique = module.getattr("unique").expect("fnp.unique");
        assert!(
            !fnp_unique.is(&np_unique),
            "dispatch trap: fnp.unique resolved to the NumPy callable"
        );
        common::report_numpy_incumbent_identity(py, "unique", &np_unique);

        // Order-sensitive digest over a strided sample: full byte-exactness is
        // asserted once per row before timing, so inside the contract this only
        // has to detect drift.
        let checksum_of = |result: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
            let n = result
                .getattr("size")
                .expect("result size")
                .extract::<usize>()
                .expect("result size value");
            let stride = (n / 4096).max(1);
            let sampled = result
                .call_method1(
                    "__getitem__",
                    (pyo3::types::PySlice::new(
                        result.py(),
                        0,
                        n as isize,
                        stride as isize,
                    ),),
                )
                .expect("strided digest slice")
                .call_method0("tobytes")
                .expect("strided digest tobytes")
                .extract::<Vec<u8>>()
                .expect("strided digest bytes");
            sampled
                .iter()
                .chain(n.to_le_bytes().iter())
                .fold(0xcbf2_9ce4_8422_2325_u64, |state, &byte| {
                    (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                })
        };

        // `drawn_from` is the number of distinct values the n elements are drawn
        // from; ties_per_value = n / drawn_from. Spanning 6 orders of magnitude of
        // tie density at fixed n is what makes the dense-ties claim decidable
        // rather than inherited.
        //
        // ROUTE TRAP, and why the tie pools are random doubles rather than the
        // obvious `rng.integers(...).astype(np.float64)`: `unique` tries
        // `try_zerocopy_f64_unique_binary_grid` BEFORE the sort route, and that
        // path claims any finite f64 corpus whose values are exact multiples of
        // 1/16 within a range under `max(4n, 1<<16)`. Integer-valued ties are
        // exactly that, so an integer-drawn tie corpus would have measured the
        // O(n+range) bucket path while the row claimed
        // `candidate_route=try_zerocopy_f64_unique_flat`. Uniform doubles are off
        // the 1/16 grid, so the grid scan bails at its first element and the sort
        // route is the one under test. Asserted below, not assumed.
        let setup = "import numpy as np\n\
rng = np.random.default_rng(20260804)\n\
uniform_4m = rng.random(4_000_000)\n\
uniform_16m = rng.random(16_000_000)\n\
uniform_64m = rng.random(64_000_000)\n\
pool_1m = rng.random(1_000_000)\n\
pool_64 = rng.random(64)\n\
pool_2 = rng.random(2)\n\
ties_1m_16m = pool_1m[rng.integers(0, 1_000_000, 16_000_000)]\n\
ties_64_16m = pool_64[rng.integers(0, 64, 16_000_000)]\n\
ties_2_16m = pool_2[rng.integers(0, 2, 16_000_000)]\n\
ties_64_64m = pool_64[rng.integers(0, 64, 64_000_000)]\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("flat-unique corpus setup");

        for (name, corpus, drawn_from) in [
            // CHEAPEST FIRST, ALTERNATING REGIME — not grouped by regime. Each
            // row is self-contained (its own live incumbent arm and its own two
            // A/A nulls, all inside one invocation), so the loop order is free
            // to be chosen, and on a shared box it has to be: the harness
            // re-checks host quiescence before EVERY contract, so a run can
            // clear the process preflight and still be killed partway through
            // when a peer's build wakes up.
            //
            // Grouping by regime is the mistake to avoid in both directions.
            // Distinct-first spent every survivable window on the compute-bound
            // rows; ties-first then starved the distinct rows, which are the
            // ones a worker floor below full width actually rests on (NumPy's
            // unique is single-threaded at 1.00x cpu/wall in every regime, so
            // the candidate's margin grows with workers in the compute-bound
            // distinct case, while the dense-tie case is bandwidth-bound and
            // came out FLAT at 1.12-1.22x across 16 and 32 workers). Ordering
            // by cost interleaves them, so a short-lived run answers both.
            ("uniform_4m", "distinct_uniform", 0_usize),
            ("ties_2_16m", "ties_2_distinct", 2),
            ("ties_64_16m", "ties_64_distinct", 64),
            ("ties_1m_16m", "ties_1m_distinct", 1_000_000),
            ("uniform_16m", "distinct_uniform", 0),
            ("ties_64_64m", "ties_64_distinct", 64),
            ("uniform_64m", "distinct_uniform", 0),
        ] {
            let input = ns.get_item(name).expect("corpus present");
            let elements = input
                .getattr("size")
                .expect("corpus size")
                .extract::<usize>()
                .expect("corpus size value");
            let input_bytes = input
                .getattr("nbytes")
                .expect("corpus nbytes")
                .extract::<usize>()
                .expect("corpus nbytes value");
            assert!(
                input
                    .getattr("flags")
                    .expect("corpus flags")
                    .getattr("c_contiguous")
                    .expect("corpus C-contiguous flag")
                    .extract::<bool>()
                    .expect("corpus C-contiguous value")
            );
            // Prove the earlier `try_zerocopy_f64_unique_binary_grid` route cannot
            // claim this corpus: it requires EVERY value to be an exact multiple of
            // 1/16. One off-grid element is enough to make it defer, and it scans
            // from index 0, so check the first few. Without this the row could
            // silently time the bucket path under the sort route's name.
            {
                let first_16: Vec<f64> = input
                    .call_method1("__getitem__", (pyo3::types::PySlice::new(py, 0, 16, 1),))
                    .expect("corpus head slice")
                    .call_method0("tolist")
                    .expect("corpus head tolist")
                    .extract()
                    .expect("corpus head values");
                assert!(
                    first_16.iter().any(|v| (v * 16.0).fract() != 0.0),
                    "{name}: corpus is on the 1/16 binary grid, so \
                     try_zerocopy_f64_unique_binary_grid would claim it before the \
                     sort route this group claims to measure"
                );
            }

            let run_incumbent = || {
                np_unique
                    .call1((black_box(&input),))
                    .expect("NumPy unique arm")
            };
            let run_candidate = || {
                fnp_unique
                    .call1((black_box(&input),))
                    .expect("FrankenNumPy unique arm")
            };

            // Full byte-exactness, once, before any timing.
            let ours = run_candidate();
            let theirs = run_incumbent();
            assert!(
                ours.get_type().is(theirs.get_type()),
                "{name}: unique result type differs from NumPy"
            );
            assert_eq!(
                ours.getattr("dtype").unwrap().str().unwrap().to_string(),
                theirs.getattr("dtype").unwrap().str().unwrap().to_string(),
                "{name}: unique dtype differs from NumPy"
            );
            assert_eq!(
                ours.call_method0("tobytes")
                    .expect("candidate tobytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate bytes"),
                theirs
                    .call_method0("tobytes")
                    .expect("incumbent tobytes")
                    .extract::<Vec<u8>>()
                    .expect("incumbent bytes"),
                "{name}: flat f64 unique is not byte-exact vs NumPy"
            );
            let distinct_out = theirs
                .getattr("size")
                .expect("distinct count")
                .extract::<usize>()
                .expect("distinct count value");

            let row = format!("python_flat_f64_unique_{name}_vs_numpy");
            println!(
                "PARITY row={row} exact_bytes=passed exact_dtype=passed \
                 corpus={corpus} input_elements={elements} input_bytes={input_bytes} \
                 drawn_from_distinct={drawn_from} distinct_out={distinct_out} \
                 checksum={:016x}",
                checksum_of(&theirs)
            );
            println!(
                "ROUTE_PRECONDITIONS row={row} axis=none dtype=float64 exact_ndarray=true \
                 c_contiguous=true input_elements={elements} \
                 parallel_min_elements=1048576 any_nan=false both_signed_zeros=false \
                 host_avx2={} host_avx512f={} pinned_threads={threads} \
                 candidate_route=try_zerocopy_f64_unique_flat",
                std::arch::is_x86_feature_detected!("avx2"),
                std::arch::is_x86_feature_detected!("avx512f"),
            );
            println!(
                "COUNTED_MECHANISM row={row} class=parallel_sort_then_dedup \
                 incumbent_algorithm=numpy_unique_sort_then_flag_compress_single_threaded \
                 candidate_algorithm=rayon_par_sort_unstable_then_vec_dedup \
                 incumbent_expected_threads=1 candidate_pinned_threads={threads} \
                 candidate_extra_full_passes=1_defer_scan_plus_1_input_copy \
                 shared_input=true"
            );

            common::report_observed_thread_activity(
                &row,
                "numpy",
                THREAD_ACTIVITY_REPETITIONS,
                || {
                    black_box(checksum_of(&run_incumbent()));
                },
            );
            common::report_observed_thread_activity(
                &row,
                "fnp",
                THREAD_ACTIVITY_REPETITIONS,
                || {
                    black_box(checksum_of(&run_candidate()));
                },
            );

            let mut observe_incumbent = || {
                let started = std::time::Instant::now();
                let result = run_incumbent();
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let mut observe_candidate = || {
                let started = std::time::Instant::now();
                let result = run_candidate();
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract_with_sampling(
                    &row,
                    &mut observe_incumbent,
                    &mut observe_candidate,
                    CONTRACT_ROUNDS,
                    CONTRACT_MIN_OF,
                );
            let verdict =
                common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
            println!(
                "FLAT_UNIQUE_RESULT row={row} corpus={corpus} elements={elements} \
                 drawn_from_distinct={drawn_from} threads={threads} \
                 verdict={verdict} incumbent_median_ms={:.6} candidate_median_ms={:.6} \
                 ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
                 incumbent_null_ratio={:.6} incumbent_null_ci95=[{:.6},{:.6}] \
                 candidate_null_ratio={:.6} candidate_null_ci95=[{:.6},{:.6}] \
                 corrected_dual_null_gate=true incumbent=numpy_live_same_invocation",
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
            let decision = if verdict == "DECIDABLE_WIN" {
                "choose_fnp"
            } else {
                "choose_numpy"
            };
            println!(
                "CHOOSER_STATEMENT workload=flat_f64_unique_{name} decision={decision} \
                 verdict={verdict} incumbent=numpy_live_same_invocation \
                 measured_scope={elements}_c_contiguous_float64_elements_drawn_from_{drawn_from}_distinct_at_{threads}_pinned_threads \
                 outside_scope=run_same_contract_before_choosing"
            );
        }
    });
}

/// Flat `float64` `np.sort` against live NumPy, same invocation.
///
/// NumPy's flat-f64 basis is x86-simd-sort's vectorized qsort — efficient per
/// element, but SINGLE-THREADED: NumPy has no threading layer for sort, measured
/// at cpu/wall 0.99x. The candidate copies into a fresh `numpy.empty` and runs a
/// Rayon `par_sort_unstable` across the pinned pool.
///
/// This arm was unreachable on every AVX2 host until the ISA gate was made
/// thread-count-aware, so a decidable ratio here is ALSO the engagement proof: a
/// deferred route returns NumPy's own answer and measures 1.0 by construction.
///
/// Route-qualified corpora only (no NaN, not both signed zeros, 1-D C-contiguous,
/// n above `1<<20`); the deferral regimes are covered by conformance, not here.
fn bench_flat_f64_sort_median_gate(_c: &mut criterion::Criterion) {
    const REQUIRED_BUILD_PROFILE: &str = "release-perf";
    const CONTRACT_ROUNDS: usize = 21;
    const CONTRACT_MIN_OF: usize = 1;
    const THREAD_ACTIVITY_REPETITIONS: usize = 3;

    assert_eq!(
        std::env::var("FNP_BENCH_PROFILE").as_deref(),
        Ok(REQUIRED_BUILD_PROFILE),
        "ship-grade flat-sort evidence requires FNP_BENCH_PROFILE=release-perf"
    );
    let build_worker =
        std::env::var("FNP_BUILD_WORKER").expect("FNP_BUILD_WORKER records the build origin");
    assert!(
        !build_worker.trim().is_empty(),
        "FNP_BUILD_WORKER must be set"
    );
    let threads = std::env::var("RAYON_NUM_THREADS")
        .expect("RAYON_NUM_THREADS must be explicitly pinned before flat-sort timing");
    for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok("1"),
            "{variable} must be one: neither sort arm calls BLAS"
        );
    }
    let threads: usize = threads.parse().expect("thread count is numeric");
    assert_eq!(
        rayon::current_num_threads(),
        threads,
        "Rayon pool width does not match the pinned flat-sort configuration"
    );
    assert!(
        threads >= 8,
        "the flat-f64 sort arm requires at least 8 workers on an AVX2 host \
         (F64_FLAT_SORT_SIMD_MIN_THREADS); below that it defers by design and \
         this group would measure the NumPy passthrough against itself"
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_flat_f64_sort").expect("flat-sort bench module");
        fnp_python(&module).expect("initialize fnp_python flat-sort module");
        let numpy = py.import("numpy").expect("numpy incumbent");

        // The candidate allocates its output through numpy.empty inside the timed
        // region. That is NumPy code running in the candidate arm, so it is
        // disclosed rather than implied absent — it is also paid by the incumbent,
        // which allocates its own result, so the disclosure is conservative.
        common::report_incumbent_topology_with_shared_component(
            "fnp.sort",
            "numpy.sort",
            "numpy.empty_output_allocation",
        );
        println!("NUMPY_BUILD_CONFIG_BEGIN workload=flat_f64_sort");
        numpy
            .getattr("show_config")
            .expect("numpy.show_config")
            .call0()
            .expect("report NumPy build configuration");
        println!("NUMPY_BUILD_CONFIG_END workload=flat_f64_sort");
        println!(
            "BLAS_RELEVANCE workload=flat_f64_sort numpy_sort_uses_blas=false \
             candidate_uses_blas=false blas_threads_pinned=1 reason=comparison_sort_no_gemm"
        );

        let np_sort = numpy.getattr("sort").expect("numpy.sort");
        let fnp_sort = module.getattr("sort").expect("fnp.sort");
        assert!(
            !fnp_sort.is(&np_sort),
            "dispatch trap: fnp.sort resolved to the NumPy callable"
        );
        common::report_numpy_incumbent_identity(py, "sort", &np_sort);

        // Strided digest: the full byte-exactness assertion runs once per row
        // below; inside the contract a cheap order-sensitive digest only has to
        // detect drift, not re-prove parity on 512 MiB per observation.
        let checksum_of = |result: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
            let n = result
                .getattr("size")
                .expect("result size")
                .extract::<usize>()
                .expect("result size value");
            let stride = (n / 4096).max(1);
            let sampled = result
                .call_method1(
                    "__getitem__",
                    (pyo3::types::PySlice::new(
                        result.py(),
                        0,
                        n as isize,
                        stride as isize,
                    ),),
                )
                .expect("strided digest slice")
                .call_method0("tobytes")
                .expect("strided digest tobytes")
                .extract::<Vec<u8>>()
                .expect("strided digest bytes");
            sampled
                .iter()
                .chain(n.to_le_bytes().iter())
                .fold(0xcbf2_9ce4_8422_2325_u64, |state, &byte| {
                    (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                })
        };

        let setup = "import numpy as np\n\
rng = np.random.default_rng(20260801)\n\
uniform_4m = rng.random(4_000_000)\n\
uniform_16m = rng.random(16_000_000)\n\
uniform_64m = rng.random(64_000_000)\n\
ties_16m = rng.integers(0, 64, 16_000_000).astype(np.float64)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("flat-sort corpus setup");

        for (name, corpus) in [
            ("uniform_4m", "distinct_uniform"),
            ("uniform_16m", "distinct_uniform"),
            ("uniform_64m", "distinct_uniform"),
            ("ties_16m", "dense_ties_64_distinct"),
        ] {
            let input = ns.get_item(name).expect("corpus present");
            let elements = input
                .getattr("size")
                .expect("corpus size")
                .extract::<usize>()
                .expect("corpus size value");
            let input_bytes = input
                .getattr("nbytes")
                .expect("corpus nbytes")
                .extract::<usize>()
                .expect("corpus nbytes value");
            assert!(
                input
                    .getattr("flags")
                    .expect("corpus flags")
                    .getattr("c_contiguous")
                    .expect("corpus C-contiguous flag")
                    .extract::<bool>()
                    .expect("corpus C-contiguous value")
            );

            let run_incumbent = || np_sort.call1((black_box(&input),)).expect("NumPy sort arm");
            let run_candidate = || {
                fnp_sort
                    .call1((black_box(&input),))
                    .expect("FrankenNumPy sort arm")
            };

            // Full byte-exactness, once, before any timing.
            let ours = run_candidate();
            let theirs = run_incumbent();
            assert!(
                ours.get_type().is(theirs.get_type()),
                "{name}: sorted result type differs from NumPy"
            );
            assert_eq!(
                ours.getattr("dtype").unwrap().str().unwrap().to_string(),
                theirs.getattr("dtype").unwrap().str().unwrap().to_string(),
                "{name}: sorted dtype differs from NumPy"
            );
            assert_eq!(
                ours.call_method0("tobytes")
                    .expect("candidate tobytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate bytes"),
                theirs
                    .call_method0("tobytes")
                    .expect("incumbent tobytes")
                    .extract::<Vec<u8>>()
                    .expect("incumbent bytes"),
                "{name}: flat f64 sort is not byte-exact vs NumPy"
            );

            let row = format!("python_flat_f64_sort_{name}_vs_numpy");
            println!(
                "PARITY row={row} exact_bytes=passed exact_dtype=passed \
                 corpus={corpus} input_elements={elements} input_bytes={input_bytes} \
                 checksum={:016x}",
                checksum_of(&theirs)
            );
            println!(
                "ROUTE_PRECONDITIONS row={row} axis=none dtype=float64 exact_ndarray=true \
                 c_contiguous=true ndim=1 input_elements={elements} \
                 parallel_min_elements=1048576 any_nan=false both_signed_zeros=false \
                 host_avx2=true host_avx512f=false pinned_threads={threads} \
                 min_threads_gate=8 candidate_route=try_zerocopy_f64_sort_flat"
            );
            println!(
                "COUNTED_MECHANISM row={row} class=parallel_comparison_value_sort \
                 incumbent_algorithm=x86_simd_sort_qsort_single_threaded \
                 candidate_algorithm=rayon_par_sort_unstable \
                 incumbent_expected_threads=1 candidate_pinned_threads={threads} \
                 candidate_extra_full_passes=1_defer_scan_plus_1_copy \
                 shared_input=true"
            );

            common::report_observed_thread_activity(
                &row,
                "numpy",
                THREAD_ACTIVITY_REPETITIONS,
                || {
                    black_box(checksum_of(&run_incumbent()));
                },
            );
            common::report_observed_thread_activity(
                &row,
                "fnp",
                THREAD_ACTIVITY_REPETITIONS,
                || {
                    black_box(checksum_of(&run_candidate()));
                },
            );

            let mut observe_incumbent = || {
                let started = std::time::Instant::now();
                let result = run_incumbent();
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let mut observe_candidate = || {
                let started = std::time::Instant::now();
                let result = run_candidate();
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract_with_sampling(
                    &row,
                    &mut observe_incumbent,
                    &mut observe_candidate,
                    CONTRACT_ROUNDS,
                    CONTRACT_MIN_OF,
                );
            let verdict =
                common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
            println!(
                "FLAT_SORT_RESULT row={row} corpus={corpus} elements={elements} \
                 verdict={verdict} incumbent_median_ms={:.6} candidate_median_ms={:.6} \
                 ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
                 incumbent_null_ratio={:.6} incumbent_null_ci95=[{:.6},{:.6}] \
                 candidate_null_ratio={:.6} candidate_null_ci95=[{:.6},{:.6}] \
                 corrected_dual_null_gate=true incumbent=numpy_live_same_invocation",
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
            let decision = if verdict == "DECIDABLE_WIN" {
                "choose_fnp"
            } else {
                "choose_numpy"
            };
            println!(
                "CHOOSER_STATEMENT workload=flat_f64_sort_{name} decision={decision} \
                 verdict={verdict} incumbent=numpy_live_same_invocation \
                 measured_scope={elements}_c_contiguous_float64_elements_at_{threads}_pinned_threads_avx2_no_avx512 \
                 outside_scope=run_same_contract_before_choosing"
            );
        }
    });
}
