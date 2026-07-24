//! structured-array / unique / sort domain criterion benches — datetime unique,
//! structured-record unique (+ factorize), unique-rows for datetime and f16,
//! float lexsort, and structured-record sort — split out of the monolithic
//! `criterion_python_surface.rs` into their own per-domain bench binary, so a
//! per-domain run compiles only these groups. See bead deadlock-audit-x7nnf.

#[path = "common/mod.rs"]
mod common;

use common::ensure_numpy_available;
use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyDict, PyModule};
use std::hint::black_box;
use std::time::Duration;

fn bench_datetime_unique_boundary(c: &mut Criterion) {
    // np.unique on datetime64/timedelta64. numpy sorts via a generic comparison introsort (~665ms @2M);
    // fnp routes to an int64 sort+dedup of the ticks viewed back as the same M8/m8[unit] — bit-exact.
    let mut group = c.benchmark_group("python_datetime_unique_boundary");
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
dt = rng.integers(0, 100000, 2_000_000).astype('datetime64[s]')\n\
td = rng.integers(0, 100000, 2_000_000).astype('timedelta64[s]')\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let dt = ns.get_item("dt").expect("dt");
        let td = ns.get_item("td").expect("td");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for (arr, label) in [(&dt, "datetime64"), (&td, "timedelta64")] {
            let f = fnp_u.call1((arr,)).expect("fnp unique");
            let n = numpy_u.call1((arr,)).expect("numpy unique");
            assert!(
                eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
                "{label} unique mismatch"
            );
        }
        group.bench_function("fnp_unique_datetime64_2m", |bn| {
            bn.iter(|| black_box(fnp_u.call1((&dt,)).unwrap()))
        });
        group.bench_function("numpy_unique_datetime64_2m", |bn| {
            bn.iter(|| black_box(numpy_u.call1((&dt,)).unwrap()))
        });
    });
    group.finish();
}

fn bench_unique_struct_int_boundary(c: &mut Criterion) {
    // np.unique(1-D structured all-int64) — record dedup. numpy sorts records via its slow field value-lex
    // comparator (~776ms @1M x 2 fields); fnp views as (n,nfields) int64 and routes to the int row-unique.
    let mut group = c.benchmark_group("python_unique_struct_int_boundary");
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
n = 1_000_000\n\
a = np.zeros(n, dtype=[('a','<i8'),('b','<i8'),('c','<i8')])\n\
a['a'] = rng.integers(-50, 50, n)\n\
a['b'] = rng.integers(-50, 50, n)\n\
a['c'] = rng.integers(-50, 50, n)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let f = fnp_u.call1((&a,)).expect("fnp unique");
        let n = numpy_u.call1((&a,)).expect("numpy unique");
        assert!(
            eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
            "unique struct mismatch"
        );
        group.bench_function("fnp_unique_struct_3xi8_1m", |bn| {
            bn.iter(|| black_box(fnp_u.call1((&a,)).unwrap()))
        });
        group.bench_function("numpy_unique_struct_3xi8_1m", |bn| {
            bn.iter(|| black_box(numpy_u.call1((&a,)).unwrap()))
        });
    });
    group.finish();
}

fn bench_unique_struct_mixed_boundary(c: &mut Criterion) {
    // np.unique on a 1-D MIXED int+float structured (record) array. numpy sorts records via its slow void
    // comparator (~778ms @1M i8+f8); fnp value-lex sorts a memcmp-comparable byte-transform of the records.
    let mut group = c.benchmark_group("python_unique_struct_mixed_boundary");
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
dt = [('id','<i8'),('val','<f8'),('flag','<i4')]\n\
n = 1_000_000\n\
a = np.zeros(n, dtype=dt)\n\
a['id'] = rng.integers(-100000, 100000, n)\n\
a['val'] = rng.integers(-100000, 100000, n).astype(np.float64)\n\
a['flag'] = rng.integers(0, 100, n)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let f = fnp_u.call1((&a,)).expect("fnp unique");
        let n = numpy_u.call1((&a,)).expect("numpy unique");
        assert!(
            eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
            "unique struct mixed mismatch"
        );
        group.bench_function("fnp_unique_struct_i8f8i4_1m", |bn| {
            bn.iter(|| black_box(fnp_u.call1((&a,)).unwrap()))
        });
        group.bench_function("numpy_unique_struct_i8f8i4_1m", |bn| {
            bn.iter(|| black_box(numpy_u.call1((&a,)).unwrap()))
        });
    });
    group.finish();
}

fn bench_unique_rows_datetime_boundary(c: &mut Criterion) {
    // np.unique(2-D datetime64, axis=0) plain + factorize. numpy sorts records via its slow void comparator;
    // fnp views int64 and routes to the int row-unique, viewing the result back as datetime64[unit].
    let mut group = c.benchmark_group("python_unique_rows_datetime_boundary");
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
base = rng.integers(0, 100000, (250_000, 3)).astype('datetime64[s]')\n\
a = np.concatenate([base, base])\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        // plain
        let kw = PyDict::new(py);
        kw.set_item("axis", 0_i64).unwrap();
        let f = fnp_u.call((&a,), Some(&kw)).expect("fnp unique");
        let n = numpy_u.call((&a,), Some(&kw)).expect("numpy unique");
        assert!(
            eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
            "datetime rows plain mismatch"
        );
        // factorize
        let kwf = PyDict::new(py);
        kwf.set_item("axis", 0_i64).unwrap();
        kwf.set_item("return_index", true).unwrap();
        kwf.set_item("return_inverse", true).unwrap();
        kwf.set_item("return_counts", true).unwrap();
        let ft = fnp_u
            .call((&a,), Some(&kwf))
            .expect("fnp full")
            .cast_into::<pyo3::types::PyTuple>()
            .unwrap();
        let nt = numpy_u
            .call((&a,), Some(&kwf))
            .expect("numpy full")
            .cast_into::<pyo3::types::PyTuple>()
            .unwrap();
        for i in 0..4 {
            let eq: bool = eqf
                .call1((ft.get_item(i).unwrap(), nt.get_item(i).unwrap()))
                .unwrap()
                .extract()
                .unwrap();
            assert!(eq, "datetime rows factorize element {i} mismatch");
        }
        group.bench_function("fnp_unique_rows_datetime_500kx3", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_datetime_500kx3", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_rows_f16_boundary(c: &mut Criterion) {
    // np.unique(2-D float16, axis=0) plain + factorize. numpy f16 rows (void comparator, no SIMD) ~507ms;
    // fnp widens exact to f32, routes to the f32 row-unique, narrows the unique rows back to f16.
    let mut group = c.benchmark_group("python_unique_rows_f16_boundary");
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
base = rng.integers(0, 50, (250_000, 4)).astype(np.float16)\n\
a = np.concatenate([base, base])\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let kw = PyDict::new(py);
        kw.set_item("axis", 0_i64).unwrap();
        let f = fnp_u.call((&a,), Some(&kw)).expect("fnp unique");
        let n = numpy_u.call((&a,), Some(&kw)).expect("numpy unique");
        assert!(
            eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
            "f16 rows plain mismatch"
        );
        let kwf = PyDict::new(py);
        kwf.set_item("axis", 0_i64).unwrap();
        kwf.set_item("return_index", true).unwrap();
        kwf.set_item("return_inverse", true).unwrap();
        kwf.set_item("return_counts", true).unwrap();
        let ft = fnp_u
            .call((&a,), Some(&kwf))
            .expect("fnp full")
            .cast_into::<pyo3::types::PyTuple>()
            .unwrap();
        let nt = numpy_u
            .call((&a,), Some(&kwf))
            .expect("numpy full")
            .cast_into::<pyo3::types::PyTuple>()
            .unwrap();
        for i in 0..4 {
            let eq: bool = eqf
                .call1((ft.get_item(i).unwrap(), nt.get_item(i).unwrap()))
                .unwrap()
                .extract()
                .unwrap();
            assert!(eq, "f16 rows factorize element {i} mismatch");
        }
        group.bench_function("fnp_unique_rows_f16_500kx4", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_f16_500kx4", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_struct_int_factorize_boundary(c: &mut Criterion) {
    // np.unique(1-D structured all-int64, return_index+inverse+counts) — record factorize.
    let mut group = c.benchmark_group("python_unique_struct_int_factorize_boundary");
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
n = 1_000_000\n\
a = np.zeros(n, dtype=[('a','<i8'),('b','<i8'),('c','<i8')])\n\
a['a'] = rng.integers(-30, 30, n)\n\
a['b'] = rng.integers(-30, 30, n)\n\
a['c'] = rng.integers(-30, 30, n)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let kwfull = PyDict::new(py);
        kwfull.set_item("return_index", true).unwrap();
        kwfull.set_item("return_inverse", true).unwrap();
        kwfull.set_item("return_counts", true).unwrap();
        let ft = fnp_u
            .call((&a,), Some(&kwfull))
            .expect("fnp full")
            .cast_into::<pyo3::types::PyTuple>()
            .unwrap();
        let nt = numpy_u
            .call((&a,), Some(&kwfull))
            .expect("numpy full")
            .cast_into::<pyo3::types::PyTuple>()
            .unwrap();
        for i in 0..4 {
            let eq: bool = eqf
                .call1((ft.get_item(i).unwrap(), nt.get_item(i).unwrap()))
                .unwrap()
                .extract()
                .unwrap();
            assert!(eq, "struct factorize element {i} mismatch");
        }
        group.bench_function("fnp_unique_struct_factorize_3xi8_1m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_struct_factorize_3xi8_1m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_struct_mixed_factorize_boundary(c: &mut Criterion) {
    // np.unique(1-D MIXED int+float structured, return_index+inverse+counts) — record factorize. numpy void
    // comparator + inverse build (~1.24s @1M i8+f8); fnp byte-transform stable sort + group scatter.
    let mut group = c.benchmark_group("python_unique_struct_mixed_factorize_boundary");
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
a = np.zeros(1_000_000, dtype=dt); a['id'] = rng.integers(0, 10000, 1_000_000); a['val'] = rng.integers(0, 10000, 1_000_000).astype(np.float64)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let a = ns.get_item("a").expect("a");
        let fnp_u = module.getattr("unique").expect("fnp unique");
        let numpy_u = numpy.getattr("unique").expect("numpy unique");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let kwf = PyDict::new(py);
        kwf.set_item("return_index", true).unwrap();
        kwf.set_item("return_inverse", true).unwrap();
        kwf.set_item("return_counts", true).unwrap();
        let ft = fnp_u
            .call((&a,), Some(&kwf))
            .expect("fnp full")
            .cast_into::<pyo3::types::PyTuple>()
            .unwrap();
        let nt = numpy_u
            .call((&a,), Some(&kwf))
            .expect("numpy full")
            .cast_into::<pyo3::types::PyTuple>()
            .unwrap();
        for i in 0..4 {
            let eq: bool = eqf
                .call1((ft.get_item(i).unwrap(), nt.get_item(i).unwrap()))
                .unwrap()
                .extract()
                .unwrap();
            assert!(eq, "mixed struct factorize element {i} mismatch");
        }
        group.bench_function("fnp_unique_struct_mixed_factorize_1m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_struct_mixed_factorize_1m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_lexsort_float_boundary(c: &mut Criterion) {
    // np.lexsort with multiple non-integral float64 keys. numpy runs a K-pass comparison sort (~880ms 2 keys /
    // ~1.5s 3 keys @2M); fnp byte-transforms the keys into one memcmp-comparable record and sorts once — bit-exact.
    let mut group = c.benchmark_group("python_lexsort_float_boundary");
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
k0 = rng.standard_normal(2_000_000)\n\
k1 = rng.standard_normal(2_000_000)\n\
k2 = rng.standard_normal(2_000_000)\n\
keys2 = (k0, k1)\n\
keys3 = (k0, k1, k2)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let keys2 = ns.get_item("keys2").expect("keys2");
        let keys3 = ns.get_item("keys3").expect("keys3");
        let fnp_lx = module.getattr("lexsort").expect("fnp lexsort");
        let numpy_lx = numpy.getattr("lexsort").expect("numpy lexsort");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for (keys, label) in [(&keys2, "2key"), (&keys3, "3key")] {
            let f = fnp_lx.call1((keys,)).expect("fnp lexsort");
            let n = numpy_lx.call1((keys,)).expect("numpy lexsort");
            assert!(
                eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
                "lexsort float {label} mismatch"
            );
        }
        group.bench_function("fnp_lexsort_f64_2key_2m", |bn| {
            bn.iter(|| black_box(fnp_lx.call1((&keys2,)).unwrap()))
        });
        group.bench_function("numpy_lexsort_f64_2key_2m", |bn| {
            bn.iter(|| black_box(numpy_lx.call1((&keys2,)).unwrap()))
        });
        group.bench_function("fnp_lexsort_f64_3key_2m", |bn| {
            bn.iter(|| black_box(fnp_lx.call1((&keys3,)).unwrap()))
        });
        group.bench_function("numpy_lexsort_f64_3key_2m", |bn| {
            bn.iter(|| black_box(numpy_lx.call1((&keys3,)).unwrap()))
        });
    });
    group.finish();
}

fn bench_sort_struct_mixed_boundary(c: &mut Criterion) {
    // np.sort on a 1-D MIXED int+float structured (record) array. numpy sorts records by field value-lex; the
    // existing fnp struct-sort path routes through numpy.lexsort (slow K-pass for float ~627ms @1M); fnp
    // byte-transforms the records and sorts once (no dedup) — bit-exact. The argsort sibling returns the same
    // transformed-key permutation when records are distinct, and defers on ties to preserve numpy's unstable order.
    let mut group = c.benchmark_group("python_sort_struct_mixed_boundary");
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
a = np.zeros(1_000_000, dtype=dt); a['id'] = rng.integers(0, 10000, 1_000_000); a['val'] = rng.integers(0, 10000, 1_000_000).astype(np.float64)\n\
a_argsort = np.zeros(1_000_000, dtype=dt); a_argsort['id'] = rng.permutation(1_000_000); a_argsort['val'] = rng.standard_normal(1_000_000)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let a = ns.get_item("a").expect("a");
        let a_argsort = ns.get_item("a_argsort").expect("a_argsort");
        let fnp_s = module.getattr("sort").expect("fnp sort");
        let numpy_s = numpy.getattr("sort").expect("numpy sort");
        let fnp_as = module.getattr("argsort").expect("fnp argsort");
        let numpy_as = numpy.getattr("argsort").expect("numpy argsort");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let f = fnp_s.call1((&a,)).expect("fnp sort");
        let n = numpy_s.call1((&a,)).expect("numpy sort");
        assert!(
            eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
            "sort mixed struct mismatch"
        );
        let f_idx = fnp_as.call1((&a_argsort,)).expect("fnp argsort");
        let n_idx = numpy_as.call1((&a_argsort,)).expect("numpy argsort");
        assert!(
            eqf.call1((&f_idx, &n_idx))
                .unwrap()
                .extract::<bool>()
                .unwrap(),
            "argsort mixed struct mismatch"
        );
        group.bench_function("fnp_sort_struct_i8f8_1m", |bn| {
            bn.iter(|| black_box(fnp_s.call1((&a,)).unwrap()))
        });
        group.bench_function("numpy_sort_struct_i8f8_1m", |bn| {
            bn.iter(|| black_box(numpy_s.call1((&a,)).unwrap()))
        });
        group.bench_function("fnp_argsort_struct_i8f8_distinct_1m", |bn| {
            bn.iter(|| black_box(fnp_as.call1((&a_argsort,)).unwrap()))
        });
        group.bench_function("numpy_argsort_struct_i8f8_distinct_1m", |bn| {
            bn.iter(|| black_box(numpy_as.call1((&a_argsort,)).unwrap()))
        });
    });
    group.finish();
}

fn main() {
    common::gated_main(&[
        ("bench_datetime_unique_boundary", bench_datetime_unique_boundary),
        ("bench_unique_struct_int_boundary", bench_unique_struct_int_boundary),
        ("bench_unique_struct_mixed_boundary", bench_unique_struct_mixed_boundary),
        ("bench_unique_rows_datetime_boundary", bench_unique_rows_datetime_boundary),
        ("bench_unique_rows_f16_boundary", bench_unique_rows_f16_boundary),
        ("bench_unique_struct_int_factorize_boundary", bench_unique_struct_int_factorize_boundary),
        (
            "bench_unique_struct_mixed_factorize_boundary",
            bench_unique_struct_mixed_factorize_boundary,
        ),
        ("bench_lexsort_float_boundary", bench_lexsort_float_boundary),
        ("bench_sort_struct_mixed_boundary", bench_sort_struct_mixed_boundary),
    ]);
}
