//! set-operation domain criterion benches (isin / searchsorted / setops across
//! structured, complex128, and datetime dtypes) split out of the monolithic
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

fn bench_isin_struct_boundary(c: &mut Criterion) {
    // np.isin(1-D structured, structured) record membership. numpy falls back to a serial sort of
    // |element|+|test| (~4s @1M+500k); fnp hashes the test record bytes + parallel lookup — bit-exact.
    let mut group = c.benchmark_group("python_isin_struct_boundary");
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
dt = [('a','<i8'),('b','<i8')]\n\
a = np.zeros(1_000_000, dtype=dt); a['a'] = rng.integers(0, 1000, 1_000_000); a['b'] = rng.integers(0, 1000, 1_000_000)\n\
b = np.zeros(500_000, dtype=dt); b['a'] = rng.integers(0, 1000, 500_000); b['b'] = rng.integers(0, 1000, 500_000)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let a = ns.get_item("a").expect("a");
        let b = ns.get_item("b").expect("b");
        let fnp_isin = module.getattr("isin").expect("fnp isin");
        let numpy_isin = numpy.getattr("isin").expect("numpy isin");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let f = fnp_isin.call1((&a, &b)).expect("fnp isin");
        let n = numpy_isin.call1((&a, &b)).expect("numpy isin");
        assert!(
            eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
            "isin struct mismatch"
        );
        group.bench_function("fnp_isin_struct_2xi8_1m_500k", |bn| {
            bn.iter(|| black_box(fnp_isin.call1((&a, &b)).unwrap()))
        });
        group.bench_function("numpy_isin_struct_2xi8_1m_500k", |bn| {
            bn.iter(|| black_box(numpy_isin.call1((&a, &b)).unwrap()))
        });
    });
    group.finish();
}

fn bench_isin_struct_float_boundary(c: &mut Criterion) {
    // np.isin on a MIXED int+float structured (record) array. numpy delegates to a serial sort (~1.3s
    // @1M+500k); fnp hashes the record bytes (finite float fields, -0.0/NaN defer) — bit-exact.
    let mut group = c.benchmark_group("python_isin_struct_float_boundary");
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
a = np.zeros(1_000_000, dtype=dt); a['id'] = rng.integers(0, 1000, 1_000_000); a['val'] = rng.integers(0, 1000, 1_000_000).astype(np.float64)\n\
b = np.zeros(500_000, dtype=dt); b['id'] = rng.integers(0, 1000, 500_000); b['val'] = rng.integers(0, 1000, 500_000).astype(np.float64)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let a = ns.get_item("a").expect("a");
        let b = ns.get_item("b").expect("b");
        let fnp_isin = module.getattr("isin").expect("fnp isin");
        let numpy_isin = numpy.getattr("isin").expect("numpy isin");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let f = fnp_isin.call1((&a, &b)).expect("fnp isin");
        let n = numpy_isin.call1((&a, &b)).expect("numpy isin");
        assert!(
            eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
            "isin struct int+f8 mismatch"
        );
        group.bench_function("fnp_isin_struct_i8f8_1m_500k", |bn| {
            bn.iter(|| black_box(fnp_isin.call1((&a, &b)).unwrap()))
        });
        group.bench_function("numpy_isin_struct_i8f8_1m_500k", |bn| {
            bn.iter(|| black_box(numpy_isin.call1((&a, &b)).unwrap()))
        });
    });
    group.finish();
}

fn bench_searchsorted_struct_boundary(c: &mut Criterion) {
    // np.searchsorted(sorted structured, structured queries) — record binary search. numpy uses its slow
    // per-record void comparator (~5.5s @2M+2M); fnp value-lex binary search via the int64 view — bit-exact.
    let mut group = c.benchmark_group("python_searchsorted_struct_boundary");
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
dt = [('a','<i8'),('b','<i8')]\n\
hay = np.zeros(2_000_000, dtype=dt); hay['a'] = rng.integers(0, 100000, 2_000_000); hay['b'] = rng.integers(0, 100000, 2_000_000); hay = np.sort(hay)\n\
q = np.zeros(2_000_000, dtype=dt); q['a'] = rng.integers(0, 100000, 2_000_000); q['b'] = rng.integers(0, 100000, 2_000_000)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let hay = ns.get_item("hay").expect("hay");
        let q = ns.get_item("q").expect("q");
        let fnp_ss = module.getattr("searchsorted").expect("fnp searchsorted");
        let numpy_ss = numpy.getattr("searchsorted").expect("numpy searchsorted");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for side in ["left", "right"] {
            let kw = PyDict::new(py);
            kw.set_item("side", side).unwrap();
            let f = fnp_ss
                .call((&hay, &q), Some(&kw))
                .expect("fnp searchsorted");
            let n = numpy_ss
                .call((&hay, &q), Some(&kw))
                .expect("numpy searchsorted");
            assert!(
                eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
                "searchsorted struct side={side} mismatch"
            );
        }
        group.bench_function("fnp_searchsorted_struct_2xi8_2m_2m", |bn| {
            bn.iter(|| black_box(fnp_ss.call1((&hay, &q)).unwrap()))
        });
        group.bench_function("numpy_searchsorted_struct_2xi8_2m_2m", |bn| {
            bn.iter(|| black_box(numpy_ss.call1((&hay, &q)).unwrap()))
        });

        let setup_u64 = "import numpy as np\n\
rng = np.random.default_rng(1)\n\
dt = [('a','<u8'),('b','<u8')]\n\
n = 1_000_000\n\
base = np.arange(n, dtype=np.uint64)\n\
hay_u = np.zeros(n, dtype=dt)\n\
hay_u['a'] = base // np.uint64(1000)\n\
hay_u['b'] = base % np.uint64(1000)\n\
q_u = np.zeros(n, dtype=dt)\n\
qa = rng.integers(0, n, n, dtype=np.uint64)\n\
q_u['a'] = qa // np.uint64(1000)\n\
q_u['b'] = qa % np.uint64(1000)\n";
        py.run(
            std::ffi::CString::new(setup_u64).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("u64 setup");
        let hay_u = ns.get_item("hay_u").expect("hay_u");
        let q_u = ns.get_item("q_u").expect("q_u");
        for side in ["left", "right"] {
            let kw = PyDict::new(py);
            kw.set_item("side", side).unwrap();
            let f = fnp_ss
                .call((&hay_u, &q_u), Some(&kw))
                .expect("fnp u64 searchsorted");
            let n = numpy_ss
                .call((&hay_u, &q_u), Some(&kw))
                .expect("numpy u64 searchsorted");
            assert!(
                eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
                "u64 searchsorted struct side={side} mismatch"
            );
        }
        group.bench_function("fnp_searchsorted_struct_2xu8_1m_1m", |bn| {
            bn.iter(|| black_box(fnp_ss.call1((&hay_u, &q_u)).unwrap()))
        });
        group.bench_function("numpy_searchsorted_struct_2xu8_1m_1m", |bn| {
            bn.iter(|| black_box(numpy_ss.call1((&hay_u, &q_u)).unwrap()))
        });
    });
    group.finish();
}

fn bench_searchsorted_struct_mixed_boundary(c: &mut Criterion) {
    // np.searchsorted(sorted MIXED int+float structured, queries). numpy void-comparator binary search (~10s
    // @2M+2M i8+f8); fnp = byte-transform record keys + parallel memcmp binary search — bit-exact both sides.
    let mut group = c.benchmark_group("python_searchsorted_struct_mixed_boundary");
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
hay = np.zeros(2_000_000, dtype=dt); hay['id'] = rng.integers(0, 100000, 2_000_000); hay['val'] = rng.integers(0, 100000, 2_000_000).astype(np.float64); hay = np.sort(hay)\n\
q = np.zeros(2_000_000, dtype=dt); q['id'] = rng.integers(0, 100000, 2_000_000); q['val'] = rng.integers(0, 100000, 2_000_000).astype(np.float64)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let hay = ns.get_item("hay").expect("hay");
        let q = ns.get_item("q").expect("q");
        let fnp_ss = module.getattr("searchsorted").expect("fnp searchsorted");
        let numpy_ss = numpy.getattr("searchsorted").expect("numpy searchsorted");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for side in ["left", "right"] {
            let kw = PyDict::new(py);
            kw.set_item("side", side).unwrap();
            let f = fnp_ss
                .call((&hay, &q), Some(&kw))
                .expect("fnp searchsorted");
            let n = numpy_ss
                .call((&hay, &q), Some(&kw))
                .expect("numpy searchsorted");
            assert!(
                eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
                "searchsorted mixed struct side={side} mismatch"
            );
        }
        group.bench_function("fnp_searchsorted_struct_i8f8_2m_2m", |bn| {
            bn.iter(|| black_box(fnp_ss.call1((&hay, &q)).unwrap()))
        });
        group.bench_function("numpy_searchsorted_struct_i8f8_2m_2m", |bn| {
            bn.iter(|| black_box(numpy_ss.call1((&hay, &q)).unwrap()))
        });
    });
    group.finish();
}

fn bench_struct_setops_boundary(c: &mut Criterion) {
    // np.union1d / intersect1d / setdiff1d / setxor1d on 1-D structured records. numpy does 2-3 serial
    // per-record void sorts (~2.5-8s @1M+1M); fnp reuses struct-unique + a hashed record-set filter.
    let mut group = c.benchmark_group("python_struct_setops_boundary");
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
dt = [('a','<i8'),('b','<i8')]\n\
a = np.zeros(1_000_000, dtype=dt); a['a'] = rng.integers(0, 3000, 1_000_000); a['b'] = rng.integers(0, 3000, 1_000_000)\n\
b = np.zeros(1_000_000, dtype=dt); b['a'] = rng.integers(0, 3000, 1_000_000); b['b'] = rng.integers(0, 3000, 1_000_000)\n";
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
                "{op} struct mismatch"
            );
            group.bench_function(format!("fnp_{op}_struct_1m_1m"), |bn| {
                bn.iter(|| black_box(fnp_fn.call1((&a, &b)).unwrap()))
            });
            group.bench_function(format!("numpy_{op}_struct_1m_1m"), |bn| {
                bn.iter(|| black_box(np_fn.call1((&a, &b)).unwrap()))
            });
        }
    });
    group.finish();
}

fn bench_struct_mixed_setops_boundary(c: &mut Criterion) {
    // np.union1d/intersect1d/setdiff1d/setxor1d on MIXED int+float structured records. numpy void-comparator
    // sorts (~2-7s @1M+1M); fnp = byte-transform value-lex unique base + hashed record filter (float finite).
    let mut group = c.benchmark_group("python_struct_mixed_setops_boundary");
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
a = np.zeros(1_000_000, dtype=dt); a['id'] = rng.integers(0, 3000, 1_000_000); a['val'] = rng.integers(0, 3000, 1_000_000).astype(np.float64)\n\
b = np.zeros(1_000_000, dtype=dt); b['id'] = rng.integers(0, 3000, 1_000_000); b['val'] = rng.integers(0, 3000, 1_000_000).astype(np.float64)\n\
dt32 = [('id','<i4'),('val','<f4')]\n\
a32 = np.zeros(1_000_000, dtype=dt32); a32['id'] = rng.integers(0, 3000, 1_000_000, dtype=np.int32); a32['val'] = rng.integers(0, 3000, 1_000_000).astype(np.float32)\n\
b32 = np.zeros(1_000_000, dtype=dt32); b32['id'] = rng.integers(0, 3000, 1_000_000, dtype=np.int32); b32['val'] = rng.integers(0, 3000, 1_000_000).astype(np.float32)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let a = ns.get_item("a").expect("a");
        let b = ns.get_item("b").expect("b");
        let a32 = ns.get_item("a32").expect("a32");
        let b32 = ns.get_item("b32").expect("b32");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for op in ["union1d", "intersect1d", "setdiff1d", "setxor1d"] {
            let fnp_fn = module.getattr(op).expect("fnp op");
            let np_fn = numpy.getattr(op).expect("numpy op");
            let f = fnp_fn.call1((&a, &b)).expect("fnp setop");
            let n = np_fn.call1((&a, &b)).expect("numpy setop");
            assert!(
                eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
                "{op} mixed struct mismatch"
            );
            group.bench_function(format!("fnp_{op}_struct_i8f8_1m_1m"), |bn| {
                bn.iter(|| black_box(fnp_fn.call1((&a, &b)).unwrap()))
            });
            group.bench_function(format!("numpy_{op}_struct_i8f8_1m_1m"), |bn| {
                bn.iter(|| black_box(np_fn.call1((&a, &b)).unwrap()))
            });
            let f32 = fnp_fn.call1((&a32, &b32)).expect("fnp i4f4 setop");
            let n32 = np_fn.call1((&a32, &b32)).expect("numpy i4f4 setop");
            assert!(
                eqf.call1((&f32, &n32)).unwrap().extract::<bool>().unwrap(),
                "{op} mixed i4f4 struct mismatch"
            );
            group.bench_function(format!("fnp_{op}_struct_i4f4_1m_1m"), |bn| {
                bn.iter(|| black_box(fnp_fn.call1((&a32, &b32)).unwrap()))
            });
            group.bench_function(format!("numpy_{op}_struct_i4f4_1m_1m"), |bn| {
                bn.iter(|| black_box(np_fn.call1((&a32, &b32)).unwrap()))
            });
        }
    });
    group.finish();
}

fn bench_c128_setops_boundary(c: &mut Criterion) {
    // np.union1d / intersect1d / setdiff1d / setxor1d on complex128. numpy delegates to a serial sort of
    // |a|+|b| (~2.7-3.1s @2M+2M); fnp reuses c128-unique + a hashed 16-byte-record filter (finite only).
    let mut group = c.benchmark_group("python_c128_setops_boundary");
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
a = (rng.integers(0, 3000, 2_000_000) + 1j * rng.integers(0, 3000, 2_000_000)).astype(np.complex128)\n\
b = (rng.integers(0, 3000, 2_000_000) + 1j * rng.integers(0, 3000, 2_000_000)).astype(np.complex128)\n";
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
        // setxor1d re-included: the old hash route was ~parity (1.22x); the dense-domain grid wins.
        for op in ["union1d", "intersect1d", "setdiff1d", "setxor1d"] {
            let fnp_fn = module.getattr(op).expect("fnp op");
            let np_fn = numpy.getattr(op).expect("numpy op");
            let f = fnp_fn.call1((&a, &b)).expect("fnp setop");
            let n = np_fn.call1((&a, &b)).expect("numpy setop");
            assert!(
                eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
                "{op} c128 mismatch"
            );
            group.bench_function(format!("fnp_{op}_c128_2m_2m"), |bn| {
                bn.iter(|| black_box(fnp_fn.call1((&a, &b)).unwrap()))
            });
            group.bench_function(format!("numpy_{op}_c128_2m_2m"), |bn| {
                bn.iter(|| black_box(np_fn.call1((&a, &b)).unwrap()))
            });
        }
    });
    group.finish();
}

fn bench_datetime_setops_boundary(c: &mut Criterion) {
    // np.union1d / intersect1d / setdiff1d / setxor1d on datetime64. numpy delegates to a serial sort
    // (~1.2-2.0s @2M+2M); fnp views int64 and routes to the fast int64 set-op, viewing back — bit-exact.
    let mut group = c.benchmark_group("python_datetime_setops_boundary");
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
a = rng.integers(0, 3000, 2_000_000).astype('datetime64[s]')\n\
b = rng.integers(0, 3000, 2_000_000).astype('datetime64[s]')\n";
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
                "{op} datetime mismatch"
            );
            group.bench_function(format!("fnp_{op}_datetime_2m_2m"), |bn| {
                bn.iter(|| black_box(fnp_fn.call1((&a, &b)).unwrap()))
            });
            group.bench_function(format!("numpy_{op}_datetime_2m_2m"), |bn| {
                bn.iter(|| black_box(np_fn.call1((&a, &b)).unwrap()))
            });
        }
    });
    group.finish();
}

fn bench_datetime_searchsorted_isin_boundary(c: &mut Criterion) {
    // np.searchsorted(sorted datetime, datetime q) + np.isin(datetime, datetime). numpy delegates both
    // (~867ms / ~341ms); fnp routes the int64 view to the fast int searchsorted / int isin — bit-exact.
    let mut group = c.benchmark_group("python_datetime_searchsorted_isin_boundary");
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
hay = np.sort(rng.integers(0, 10**9, 2_000_000).astype('datetime64[s]'))\n\
q = rng.integers(0, 10**9, 2_000_000).astype('datetime64[s]')\n\
ia = rng.integers(0, 100000, 2_000_000).astype('datetime64[s]')\n\
ib = rng.integers(0, 100000, 1_000_000).astype('datetime64[s]')\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let (hay, q, ia, ib) = (
            ns.get_item("hay").unwrap(),
            ns.get_item("q").unwrap(),
            ns.get_item("ia").unwrap(),
            ns.get_item("ib").unwrap(),
        );
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        let fnp_ss = module.getattr("searchsorted").unwrap();
        let numpy_ss = numpy.getattr("searchsorted").unwrap();
        let fnp_isin = module.getattr("isin").unwrap();
        let numpy_isin = numpy.getattr("isin").unwrap();
        for side in ["left", "right"] {
            let kw = PyDict::new(py);
            kw.set_item("side", side).unwrap();
            let f = fnp_ss.call((&hay, &q), Some(&kw)).unwrap();
            let n = numpy_ss.call((&hay, &q), Some(&kw)).unwrap();
            assert!(
                eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
                "datetime searchsorted {side} mismatch"
            );
        }
        let fi = fnp_isin.call1((&ia, &ib)).unwrap();
        let ni = numpy_isin.call1((&ia, &ib)).unwrap();
        assert!(
            eqf.call1((&fi, &ni)).unwrap().extract::<bool>().unwrap(),
            "datetime isin mismatch"
        );
        group.bench_function("fnp_searchsorted_datetime_2m_2m", |bn| {
            bn.iter(|| black_box(fnp_ss.call1((&hay, &q)).unwrap()))
        });
        group.bench_function("numpy_searchsorted_datetime_2m_2m", |bn| {
            bn.iter(|| black_box(numpy_ss.call1((&hay, &q)).unwrap()))
        });
        group.bench_function("fnp_isin_datetime_2m_1m", |bn| {
            bn.iter(|| black_box(fnp_isin.call1((&ia, &ib)).unwrap()))
        });
        group.bench_function("numpy_isin_datetime_2m_1m", |bn| {
            bn.iter(|| black_box(numpy_isin.call1((&ia, &ib)).unwrap()))
        });
    });
    group.finish();
}

fn main() {
    common::gated_main(&[
        ("bench_isin_struct_boundary", bench_isin_struct_boundary),
        ("bench_isin_struct_float_boundary", bench_isin_struct_float_boundary),
        ("bench_searchsorted_struct_boundary", bench_searchsorted_struct_boundary),
        ("bench_searchsorted_struct_mixed_boundary", bench_searchsorted_struct_mixed_boundary),
        ("bench_struct_setops_boundary", bench_struct_setops_boundary),
        ("bench_struct_mixed_setops_boundary", bench_struct_mixed_setops_boundary),
        ("bench_c128_setops_boundary", bench_c128_setops_boundary),
        ("bench_datetime_setops_boundary", bench_datetime_setops_boundary),
        ("bench_datetime_searchsorted_isin_boundary", bench_datetime_searchsorted_isin_boundary),
    ]);
}
