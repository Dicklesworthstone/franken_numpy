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
    common::gated_main(&[
        ("bench_median_int_histogram_boundary", bench_median_int_histogram_boundary),
        (
            "bench_int_percentile_quantile_histogram_boundary",
            bench_int_percentile_quantile_histogram_boundary,
        ),
        (
            "bench_argsort_temporal_complex_stable_boundary",
            bench_argsort_temporal_complex_stable_boundary,
        ),
        ("bench_argsort_string_stable_boundary", bench_argsort_string_stable_boundary),
        ("bench_argsort_struct_stable_boundary", bench_argsort_struct_stable_boundary),
        ("bench_unique_arrayapi_boundary", bench_unique_arrayapi_boundary),
    ]);
}
