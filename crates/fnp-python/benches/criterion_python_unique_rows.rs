//! unique_rows / unique_cols domain criterion benches (row/column dedup across
//! lexsort / factorize paths for int, f64, f32, c128, c64) split out of the
//! monolithic `criterion_python_surface.rs` into their own per-domain bench
//! binary, so a per-domain run compiles only these groups instead of all of
//! the monolith benches. See bead deadlock-audit-x7nnf.

#[path = "common/mod.rs"]
mod common;

use common::ensure_numpy_available;
use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyDict, PyModule};
use std::hint::black_box;
use std::time::Duration;

fn bench_unique_rows_lexsort_boundary(c: &mut Criterion) {
    // np.unique(2-D large-range int64, axis=0). numpy sorts rows with its slow void comparator
    // (~570ms @500kx3); the packed-composite path can't pack this range, so fnp does a parallel
    // value-lex row sort+dedup — bit-exact.
    let mut group = c.benchmark_group("python_unique_rows_lexsort_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        // range 1e15 per column over 3 cols => composite overflows u64 -> lexsort path; base tiled => dups.
        let setup = "import numpy as np\n\
rng = np.random.default_rng(0)\n\
base = rng.integers(-10**15, 10**15, (250_000, 3)).astype(np.int64)\n\
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
            "unique rows lexsort mismatch"
        );
        group.bench_function("fnp_unique_rows_i64_500kx3", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_i64_500kx3", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_rows_narrow_int_boundary(c: &mut Criterion) {
    // np.unique(2-D int32, axis=0). The packed-composite path cannot pack this large range, but
    // exact int64 widening lets fnp reuse the value-lex row-unique and cast back to int32.
    let mut group = c.benchmark_group("python_unique_rows_narrow_int_boundary");
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
base = rng.integers(-(1 << 30), 1 << 30, (250_000, 4), dtype=np.int32)\n\
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
            f.getattr("dtype")
                .unwrap()
                .eq(n.getattr("dtype").unwrap())
                .unwrap(),
            "unique rows narrow-int dtype mismatch"
        );
        assert!(
            eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
            "unique rows narrow-int value mismatch"
        );
        let kwfull = PyDict::new(py);
        kwfull.set_item("axis", 0_i64).unwrap();
        kwfull.set_item("return_index", true).unwrap();
        kwfull.set_item("return_inverse", true).unwrap();
        kwfull.set_item("return_counts", true).unwrap();
        let ft = fnp_u
            .call((&a,), Some(&kwfull))
            .expect("fnp unique narrow full")
            .cast_into::<pyo3::types::PyTuple>()
            .unwrap();
        let nt = numpy_u
            .call((&a,), Some(&kwfull))
            .expect("numpy unique narrow full")
            .cast_into::<pyo3::types::PyTuple>()
            .unwrap();
        assert!(
            ft.get_item(0)
                .unwrap()
                .getattr("dtype")
                .unwrap()
                .eq(nt.get_item(0).unwrap().getattr("dtype").unwrap())
                .unwrap(),
            "unique rows narrow-int factorize dtype mismatch"
        );
        for i in 0..4 {
            let eq: bool = eqf
                .call1((ft.get_item(i).unwrap(), nt.get_item(i).unwrap()))
                .unwrap()
                .extract()
                .unwrap();
            assert!(eq, "unique rows narrow-int factorize element {i} mismatch");
        }
        group.bench_function("fnp_unique_rows_i32_500kx4", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_i32_500kx4", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("fnp_unique_rows_i32_factorize_500kx4", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_i32_factorize_500kx4", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_rows_factorize_boundary(c: &mut Criterion) {
    // np.unique(2-D large-range int64, axis=0, return_index+inverse+counts) — the row factorize/group-by.
    // numpy sorts rows via its slow void comparator AND builds the inverse; fnp value-lex sorts + scatters.
    let mut group = c.benchmark_group("python_unique_rows_factorize_boundary");
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
base = rng.integers(-10**15, 10**15, (250_000, 3)).astype(np.int64)\n\
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
        let kwfull = PyDict::new(py);
        kwfull.set_item("axis", 0_i64).unwrap();
        kwfull.set_item("return_index", true).unwrap();
        kwfull.set_item("return_inverse", true).unwrap();
        kwfull.set_item("return_counts", true).unwrap();
        // Correctness: compare each of the 4 tuple elements.
        let ft = fnp_u
            .call((&a,), Some(&kwfull))
            .expect("fnp unique full")
            .cast_into::<pyo3::types::PyTuple>()
            .unwrap();
        let nt = numpy_u
            .call((&a,), Some(&kwfull))
            .expect("numpy unique full")
            .cast_into::<pyo3::types::PyTuple>()
            .unwrap();
        for i in 0..4 {
            let eq: bool = eqf
                .call1((ft.get_item(i).unwrap(), nt.get_item(i).unwrap()))
                .unwrap()
                .extract()
                .unwrap();
            assert!(eq, "unique rows factorize element {i} mismatch");
        }
        group.bench_function("fnp_unique_rows_factorize_500kx3", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_factorize_500kx3", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_rows_f64_boundary(c: &mut Criterion) {
    // np.unique(2-D float64, axis=0). numpy sorts rows value-lexicographically via its slow void
    // comparator (~771ms @500kx4); fnp value-lex sorts+dedups (finite, no -0.0/NaN) — bit-exact.
    let mut group = c.benchmark_group("python_unique_rows_f64_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        // finite f64 rows (int-valued, no -0.0/NaN), base tiled => dups.
        let setup = "import numpy as np\n\
rng = np.random.default_rng(0)\n\
base = rng.integers(-100, 100, (250_000, 4)).astype(np.float64)\n\
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
            "unique rows f64 mismatch"
        );
        group.bench_function("fnp_unique_rows_f64_500kx4", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_f64_500kx4", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_rows_f32_boundary(c: &mut Criterion) {
    // np.unique(2-D float32, axis=0). numpy value-lex void comparator (~729ms @500kx4); fnp f32 row-unique.
    let mut group = c.benchmark_group("python_unique_rows_f32_boundary");
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
base = rng.integers(-100, 100, (250_000, 4)).astype(np.float32)\n\
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
            "unique rows f32 mismatch"
        );
        group.bench_function("fnp_unique_rows_f32_500kx4", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_f32_500kx4", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_rows_f64_factorize_boundary(c: &mut Criterion) {
    // np.unique(2-D f64, axis=0, return_index+inverse+counts) — f64 row factorize. numpy ~1393ms @500kx4.
    let mut group = c.benchmark_group("python_unique_rows_f64_factorize_boundary");
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
base = rng.integers(-100, 100, (250_000, 4)).astype(np.float64)\n\
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
        let kwfull = PyDict::new(py);
        kwfull.set_item("axis", 0_i64).unwrap();
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
            assert!(eq, "f64 factorize element {i} mismatch");
        }
        group.bench_function("fnp_unique_rows_f64_factorize_500kx4", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_f64_factorize_500kx4", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_rows_f32_factorize_boundary(c: &mut Criterion) {
    // np.unique(2-D f32, axis=0, return_index+inverse+counts) — f32 row factorize.
    let mut group = c.benchmark_group("python_unique_rows_f32_factorize_boundary");
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
base = rng.integers(-100, 100, (250_000, 4)).astype(np.float32)\n\
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
        let kwfull = PyDict::new(py);
        kwfull.set_item("axis", 0_i64).unwrap();
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
            assert!(eq, "f32 factorize element {i} mismatch");
        }
        group.bench_function("fnp_unique_rows_f32_factorize_500kx4", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_f32_factorize_500kx4", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_rows_c128_boundary(c: &mut Criterion) {
    // np.unique(2-D complex128, axis=0) — routed to the f64 row-unique via a .view. numpy ~336ms @500kx3.
    let mut group = c.benchmark_group("python_unique_rows_c128_boundary");
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
base = (rng.integers(0, 50, (250_000, 3)) + 1j * rng.integers(0, 50, (250_000, 3))).astype(np.complex128)\n\
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
            "unique rows c128 mismatch"
        );
        group.bench_function("fnp_unique_rows_c128_500kx3", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_c128_500kx3", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_rows_c64_boundary(c: &mut Criterion) {
    // np.unique(2-D complex64, axis=0) plain + factorize. numpy c64 rows (void comparator) ~424ms; fnp views
    // as f32 (n, 2*ncols), routes to the f32 row-unique, views back c64.
    let mut group = c.benchmark_group("python_unique_rows_c64_boundary");
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
base = (rng.integers(0, 50, (250_000, 3)) + 1j * rng.integers(0, 50, (250_000, 3))).astype(np.complex64)\n\
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
            "c64 rows plain mismatch"
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
            assert!(eq, "c64 rows factorize element {i} mismatch");
        }
        group.bench_function("fnp_unique_rows_c64_500kx3", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_c64_500kx3", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_rows_c128_factorize_boundary(c: &mut Criterion) {
    // np.unique(2-D complex128, axis=0, return_index+inverse+counts) — routed to the f64 _full via .view.
    let mut group = c.benchmark_group("python_unique_rows_c128_factorize_boundary");
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
base = (rng.integers(0, 50, (250_000, 3)) + 1j * rng.integers(0, 50, (250_000, 3))).astype(np.complex128)\n\
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
        let kwfull = PyDict::new(py);
        kwfull.set_item("axis", 0_i64).unwrap();
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
            assert!(eq, "c128 factorize element {i} mismatch");
        }
        group.bench_function("fnp_unique_rows_c128_factorize_500kx3", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_rows_c128_factorize_500kx3", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0_i64).unwrap();
            kw.set_item("return_index", true).unwrap();
            kw.set_item("return_inverse", true).unwrap();
            kw.set_item("return_counts", true).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_unique_cols_axis1_boundary(c: &mut Criterion) {
    // np.unique(2-D wide i64, axis=1) — unique columns. numpy sorts columns via its slow void comparator
    // (~515ms @3x500k); fnp transposes to C-contig, reuses the row-unique, transposes back — bit-exact.
    let mut group = c.benchmark_group("python_unique_cols_axis1_boundary");
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
base = rng.integers(-10**15, 10**15, (3, 250_000)).astype(np.int64)\n\
a = np.concatenate([base, base], axis=1)\n";
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
        kw.set_item("axis", 1_i64).unwrap();
        let f = fnp_u.call((&a,), Some(&kw)).expect("fnp unique");
        let n = numpy_u.call((&a,), Some(&kw)).expect("numpy unique");
        assert!(
            eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
            "unique cols axis=1 mismatch"
        );
        group.bench_function("fnp_unique_cols_i64_3x500k", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 1_i64).unwrap();
            bn.iter(|| black_box(fnp_u.call((&a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_unique_cols_i64_3x500k", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 1_i64).unwrap();
            bn.iter(|| black_box(numpy_u.call((&a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn main() {
    common::gated_main(&[
        ("bench_unique_rows_lexsort_boundary", bench_unique_rows_lexsort_boundary),
        ("bench_unique_rows_narrow_int_boundary", bench_unique_rows_narrow_int_boundary),
        ("bench_unique_rows_factorize_boundary", bench_unique_rows_factorize_boundary),
        ("bench_unique_rows_f64_boundary", bench_unique_rows_f64_boundary),
        ("bench_unique_rows_f32_boundary", bench_unique_rows_f32_boundary),
        ("bench_unique_rows_f64_factorize_boundary", bench_unique_rows_f64_factorize_boundary),
        ("bench_unique_rows_f32_factorize_boundary", bench_unique_rows_f32_factorize_boundary),
        ("bench_unique_rows_c128_boundary", bench_unique_rows_c128_boundary),
        ("bench_unique_rows_c64_boundary", bench_unique_rows_c64_boundary),
        ("bench_unique_rows_c128_factorize_boundary", bench_unique_rows_c128_factorize_boundary),
        ("bench_unique_cols_axis1_boundary", bench_unique_cols_axis1_boundary),
    ]);
}
