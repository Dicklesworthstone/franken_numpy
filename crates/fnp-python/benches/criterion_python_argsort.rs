//! argsort-domain criterion benches (stable / radix / last-axis / default-kind
//! argsort across numeric, float, and datetime dtypes) split out of the
//! monolithic `criterion_python_surface.rs` into their own per-domain bench
//! binary, so a per-domain run compiles only these groups instead of all of the
//! monolith benches. See bead deadlock-audit-x7nnf.

#[path = "common/mod.rs"]
mod common;

use common::ensure_numpy_available;
use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyDict, PyModule};
use std::hint::black_box;
use std::time::Duration;

fn bench_argsort_numeric_stable_boundary(c: &mut Criterion) {
    // np.argsort(1-D int/float, kind='stable') on DENSE data (heavy repeats — the tied case the default-kind
    // paths defer on). numpy stable value sort (~0.9-1.2s @8M); fnp (value, orig-index) parallel sort — bit-exact.
    let mut group = c.benchmark_group("python_argsort_numeric_stable_boundary");
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
di = rng.integers(0, 1000, 8_000_000).astype(np.int64)\n\
df = rng.integers(0, 1000, 8_000_000).astype(np.float64)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let di = ns.get_item("di").expect("di");
        let df = ns.get_item("df").expect("df");
        let fnp_as = module.getattr("argsort").expect("fnp argsort");
        let numpy_as = numpy.getattr("argsort").expect("numpy argsort");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for (arr, label) in [(&di, "i64"), (&df, "f64")] {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            let f = fnp_as.call((arr,), Some(&kw)).expect("fnp argsort");
            let n = numpy_as.call((arr,), Some(&kw)).expect("numpy argsort");
            assert!(
                eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
                "argsort {label} dense stable mismatch"
            );
        }
        group.bench_function("fnp_argsort_i64_dense_stable_8m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&di,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_i64_dense_stable_8m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(numpy_as.call((&di,), Some(&kw)).unwrap()));
        });
        group.bench_function("fnp_argsort_f64_dense_stable_8m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&df,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_f64_dense_stable_8m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(numpy_as.call((&df,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_argsort_lastaxis_stable_boundary(c: &mut Criterion) {
    // np.argsort(2-D int/float, axis=-1, kind='stable') on DENSE rows (heavy repeats — the per-lane tied case
    // the default-kind last-axis path defers on). numpy per-lane stable sort (~0.3-0.4s @8M); fnp per-lane
    // (value, in-lane index) parallel sort across lanes — bit-exact.
    let mut group = c.benchmark_group("python_argsort_lastaxis_stable_boundary");
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
mi = np.ascontiguousarray(rng.integers(0, 100, (4000, 2000)).astype(np.int64))\n\
mf = np.ascontiguousarray(rng.integers(0, 100, (4000, 2000)).astype(np.float64))\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let mi = ns.get_item("mi").expect("mi");
        let mf = ns.get_item("mf").expect("mf");
        let fnp_as = module.getattr("argsort").expect("fnp argsort");
        let numpy_as = numpy.getattr("argsort").expect("numpy argsort");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for (arr, label) in [(&mi, "i64"), (&mf, "f64")] {
            let kw = PyDict::new(py);
            kw.set_item("axis", -1).unwrap();
            kw.set_item("kind", "stable").unwrap();
            let f = fnp_as.call((arr,), Some(&kw)).expect("fnp argsort");
            let n = numpy_as.call((arr,), Some(&kw)).expect("numpy argsort");
            assert!(
                eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
                "argsort 2D {label} axis=-1 stable mismatch"
            );
        }
        group.bench_function("fnp_argsort_i64_2d_lastaxis_stable_8m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", -1).unwrap();
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&mi,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_i64_2d_lastaxis_stable_8m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", -1).unwrap();
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(numpy_as.call((&mi,), Some(&kw)).unwrap()));
        });
        group.bench_function("fnp_argsort_f64_2d_lastaxis_stable_8m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", -1).unwrap();
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&mf,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_f64_2d_lastaxis_stable_8m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("axis", -1).unwrap();
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(numpy_as.call((&mf,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_argsort_radix_stable_boundary(c: &mut Criterion) {
    // np.argsort(1-D WIDE-range int, kind='stable') — the range my counting sort defers on. numpy comparison
    // sort (~2s @16M i64); fnp PARALLEL LSD RADIX of (key,index) pairs (gather-free multi-pass). Ranges chosen
    // > 1<<20 (forces radix) but with ties (exercises stability). Bit-exact vs numpy stable.
    let mut group = c.benchmark_group("python_argsort_radix_stable_boundary");
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
wi = rng.integers(0, 2**30, 16_000_000).astype(np.int64)\n\
wu = rng.integers(0, 2**52, 16_000_000).astype(np.uint64)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let wi = ns.get_item("wi").expect("wi");
        let wu = ns.get_item("wu").expect("wu");
        let fnp_as = module.getattr("argsort").expect("fnp argsort");
        let numpy_as = numpy.getattr("argsort").expect("numpy argsort");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for (arr, label) in [(&wi, "i64_2p30"), (&wu, "u64_2p52")] {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            let f = fnp_as.call((arr,), Some(&kw)).expect("fnp argsort");
            let n = numpy_as.call((arr,), Some(&kw)).expect("numpy argsort");
            assert!(
                eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
                "argsort wide {label} stable mismatch"
            );
        }
        group.bench_function("fnp_argsort_i64_wide_stable_16m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&wi,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_i64_wide_stable_16m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(numpy_as.call((&wi,), Some(&kw)).unwrap()));
        });
        group.bench_function("fnp_argsort_u64_wide_stable_16m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&wu,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_u64_wide_stable_16m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(numpy_as.call((&wu,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_argsort_radix_float_boundary(c: &mut Criterion) {
    // np.argsort(1-D FLOAT, kind='stable') — currently the gather-bound comparison sort (~2.7s @16M f64). fnp
    // LINEARIZES IEEE floats into radix-sortable u64 keys (monotonic bit-transform) + parallel LSD radix.
    // standard_normal has many exact-tie duplicates in float32 -> exercises stability. Bit-exact vs numpy stable.
    let mut group = c.benchmark_group("python_argsort_radix_float_boundary");
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
f64 = rng.standard_normal(16_000_000)\n\
f64[rng.integers(0, 16_000_000, 1000)] *= -1.0\n\
f32 = rng.integers(-100000, 100000, 16_000_000).astype(np.float32) / 7.0\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let f64a = ns.get_item("f64").expect("f64");
        let f32a = ns.get_item("f32").expect("f32");
        let fnp_as = module.getattr("argsort").expect("fnp argsort");
        let numpy_as = numpy.getattr("argsort").expect("numpy argsort");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for (arr, label) in [(&f64a, "f64"), (&f32a, "f32")] {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            let f = fnp_as.call((arr,), Some(&kw)).expect("fnp argsort");
            let n = numpy_as.call((arr,), Some(&kw)).expect("numpy argsort");
            assert!(
                eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
                "argsort float {label} stable mismatch"
            );
        }
        group.bench_function("fnp_argsort_f64_stable_16m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&f64a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_f64_stable_16m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(numpy_as.call((&f64a,), Some(&kw)).unwrap()));
        });
        group.bench_function("fnp_argsort_f32_stable_16m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&f32a,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_f32_stable_16m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(numpy_as.call((&f32a,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

fn bench_argsort_default_int_radix_boundary(c: &mut Criterion) {
    // np.argsort(1-D int) with the DEFAULT kind (unstable introsort) — the most common argsort call. For DISTINCT
    // data the perm is unique so the gather-free parallel radix reproduces numpy's default order (~1.3s @16M);
    // the fnp comparison path it replaces is gather-bound. Also asserts a TIED array still matches (radix defers
    // -> existing path). Bit-exact vs numpy default argsort.
    let mut group = c.benchmark_group("python_argsort_default_int_radix_boundary");
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
di = rng.permutation(16_000_000).astype(np.int64)\n\
du = rng.permutation(16_000_000).astype(np.uint64)\n\
tied = rng.integers(0, 1000, 16_000_000).astype(np.int64)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let di = ns.get_item("di").expect("di");
        let du = ns.get_item("du").expect("du");
        let tied = ns.get_item("tied").expect("tied");
        let fnp_as = module.getattr("argsort").expect("fnp argsort");
        let numpy_as = numpy.getattr("argsort").expect("numpy argsort");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for (arr, label) in [
            (&di, "i64_distinct"),
            (&du, "u64_distinct"),
            (&tied, "i64_tied"),
        ] {
            let f = fnp_as.call1((arr,)).expect("fnp argsort");
            let n = numpy_as.call1((arr,)).expect("numpy argsort");
            assert!(
                eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
                "default argsort {label} mismatch"
            );
        }
        group.bench_function("fnp_argsort_i64_distinct_default_16m", |bn| {
            bn.iter(|| black_box(fnp_as.call1((&di,)).unwrap()))
        });
        group.bench_function("numpy_argsort_i64_distinct_default_16m", |bn| {
            bn.iter(|| black_box(numpy_as.call1((&di,)).unwrap()))
        });
        group.bench_function("fnp_argsort_u64_distinct_default_16m", |bn| {
            bn.iter(|| black_box(fnp_as.call1((&du,)).unwrap()))
        });
        group.bench_function("numpy_argsort_u64_distinct_default_16m", |bn| {
            bn.iter(|| black_box(numpy_as.call1((&du,)).unwrap()))
        });
    });
    group.finish();
}

fn bench_argsort_default_float_radix_boundary(c: &mut Criterion) {
    // np.argsort(1-D float) with the DEFAULT kind (unstable introsort) — the common float argsort call. For
    // DISTINCT data the perm is unique so the gather-free radix on IEEE-linearized keys reproduces numpy's order
    // (~1.3s @16M); the fnp comparison path it replaces is gather-bound. Also asserts a TIED array still matches.
    let mut group = c.benchmark_group("python_argsort_default_float_radix_boundary");
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
f64d = rng.standard_normal(16_000_000)\n\
f32d = rng.permutation(16_000_000).astype(np.float32)\n\
tied = np.round(rng.standard_normal(16_000_000), 2)\n"; // f32d: ints < 2**24 exact -> distinct; tied: dense dups -> defers
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let f64d = ns.get_item("f64d").expect("f64d");
        let f32d = ns.get_item("f32d").expect("f32d");
        let tied = ns.get_item("tied").expect("tied");
        let fnp_as = module.getattr("argsort").expect("fnp argsort");
        let numpy_as = numpy.getattr("argsort").expect("numpy argsort");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        for (arr, label) in [
            (&f64d, "f64_distinct"),
            (&f32d, "f32_distinct"),
            (&tied, "f64_tied"),
        ] {
            let f = fnp_as.call1((arr,)).expect("fnp argsort");
            let n = numpy_as.call1((arr,)).expect("numpy argsort");
            assert!(
                eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
                "default float argsort {label} mismatch"
            );
        }
        group.bench_function("fnp_argsort_f64_distinct_default_16m", |bn| {
            bn.iter(|| black_box(fnp_as.call1((&f64d,)).unwrap()))
        });
        group.bench_function("numpy_argsort_f64_distinct_default_16m", |bn| {
            bn.iter(|| black_box(numpy_as.call1((&f64d,)).unwrap()))
        });
        group.bench_function("fnp_argsort_f32_distinct_default_16m", |bn| {
            bn.iter(|| black_box(fnp_as.call1((&f32d,)).unwrap()))
        });
        group.bench_function("numpy_argsort_f32_distinct_default_16m", |bn| {
            bn.iter(|| black_box(numpy_as.call1((&f32d,)).unwrap()))
        });
    });
    group.finish();
}

fn bench_argsort_datetime_radix_boundary(c: &mut Criterion) {
    // np.argsort(1-D datetime64) default + stable — int64-backed, so route the .view('int64') to the gather-free
    // int radix/counting instead of the gather-bound comparison sort (numpy ~2s @16M). NaT defers. Bit-exact.
    let mut group = c.benchmark_group("python_argsort_datetime_radix_boundary");
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
dt = rng.permutation(16_000_000).astype('datetime64[s]')\n\
dt_tied = rng.integers(0, 1000, 16_000_000).astype('datetime64[s]')\n"; // distinct + dense-tied
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let dt = ns.get_item("dt").expect("dt");
        let dt_tied = ns.get_item("dt_tied").expect("dt_tied");
        let fnp_as = module.getattr("argsort").expect("fnp argsort");
        let numpy_as = numpy.getattr("argsort").expect("numpy argsort");
        let eqf = numpy.getattr("array_equal").expect("np.array_equal");
        // default kind: distinct + tied (tied defers -> comparison path, still bit-exact)
        for (arr, label) in [(&dt, "distinct_default"), (&dt_tied, "tied_default")] {
            let f = fnp_as.call1((arr,)).expect("fnp argsort");
            let n = numpy_as.call1((arr,)).expect("numpy argsort");
            assert!(
                eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
                "datetime argsort {label} mismatch"
            );
        }
        // stable kind on tied datetime (dense) -> counting/radix, bit-exact
        let kw = PyDict::new(py);
        kw.set_item("kind", "stable").unwrap();
        let fs = fnp_as.call((&dt_tied,), Some(&kw)).expect("fnp stable");
        let ns_ = numpy_as.call((&dt_tied,), Some(&kw)).expect("numpy stable");
        assert!(
            eqf.call1((&fs, &ns_)).unwrap().extract::<bool>().unwrap(),
            "datetime argsort stable tied mismatch"
        );
        group.bench_function("fnp_argsort_datetime_distinct_default_16m", |bn| {
            bn.iter(|| black_box(fnp_as.call1((&dt,)).unwrap()))
        });
        group.bench_function("numpy_argsort_datetime_distinct_default_16m", |bn| {
            bn.iter(|| black_box(numpy_as.call1((&dt,)).unwrap()))
        });
        group.bench_function("fnp_argsort_datetime_tied_stable_16m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(fnp_as.call((&dt_tied,), Some(&kw)).unwrap()));
        });
        group.bench_function("numpy_argsort_datetime_tied_stable_16m", |bn| {
            let kw = PyDict::new(py);
            kw.set_item("kind", "stable").unwrap();
            bn.iter(|| black_box(numpy_as.call((&dt_tied,), Some(&kw)).unwrap()));
        });
    });
    group.finish();
}

/// Bead rc0923 .23: default-kind `argsort` at n = 2^20 against the live NumPy incumbent, in the
/// same process, under the dual-null median-CI contract. The grid covers the triage loss cells
/// (f64 normal 2.010x, i64 uniform [0, 2^40) 3.868x, f64 sorted 1.186x on thinkstation1) and their
/// neighbours (value range 2^32 / full width, i32, f32). Both arms run the public call end to end
/// and allocate their own int64 index output, so allocation is symmetric. A cell whose output is
/// not byte-identical to NumPy's is reported as PARITY=fail and not timed.
fn bench_argsort_default_grid_vs_numpy(c: &mut Criterion) {
    let _ = c;
    // Contract-grade by default (41 rounds). FNP_ARGSORT_GRID_ROUNDS overrides it for a triage run
    // without a rebuild: 11 rounds left every cell UNDECIDED on vmi1152480 (null half-widths up to
    // ~0.12 against effects of 0.14-0.39).
    let rounds: usize = std::env::var("FNP_ARGSORT_GRID_ROUNDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(common::CONTRACT_ROUNDS);
    const MIN_OF: usize = 1;
    let checksum = |out: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
        let bytes: Vec<u8> = out
            .call_method0("tobytes")
            .expect("argsort output tobytes")
            .extract()
            .expect("argsort output bytes");
        bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    };
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let version: String = numpy
            .getattr("__version__")
            .and_then(|v| v.extract())
            .expect("numpy version");
        let setup = "import numpy as np\n\
rng = np.random.default_rng(23)\n\
n = 1 << 20\n\
cells = {\n\
 'f64_normal': rng.standard_normal(n),\n\
 'f64_sorted': np.sort(rng.standard_normal(n)),\n\
 'i64_range2p40': rng.integers(0, 1 << 40, n, dtype=np.int64),\n\
 'i64_range2p32': rng.integers(0, 1 << 32, n, dtype=np.int64),\n\
 'i64_fullwidth': rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, n, dtype=np.int64),\n\
 'i32_fullwidth': rng.integers(np.iinfo(np.int32).min, np.iinfo(np.int32).max, n, dtype=np.int32),\n\
 'f32_normal': rng.standard_normal(n).astype(np.float32),\n\
}\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("argsort grid setup");
        let cells = ns.get_item("cells").expect("cells dict");
        let fnp_argsort = module.getattr("argsort").expect("fnp argsort");
        let numpy_argsort = numpy.getattr("argsort").expect("numpy argsort");
        assert!(
            !fnp_argsort.is(&numpy_argsort),
            "incumbent identity: fnp.argsort must not be numpy's own callable"
        );
        for label in [
            "f64_normal",
            "f64_sorted",
            "i64_range2p40",
            "i64_range2p32",
            "i64_fullwidth",
            "i32_fullwidth",
            "f32_normal",
        ] {
            let row = format!("argsort_default_{label}_n1048576");
            let data = cells.get_item(label).expect("grid cell");
            let ours = fnp_argsort.call1((&data,)).expect("fnp argsort parity");
            let theirs = numpy_argsort.call1((&data,)).expect("numpy argsort parity");
            let (ours_sum, theirs_sum) = (checksum(&ours), checksum(&theirs));
            if ours_sum != theirs_sum {
                println!("ARGSORT_GRID row={row} PARITY=fail numpy={version} not_timed=true");
                continue;
            }
            let mut observe_incumbent = || {
                let started = std::time::Instant::now();
                let output = numpy_argsort
                    .call1((black_box(&data),))
                    .expect("numpy argsort arm");
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum(&output),
                }
            };
            let mut observe_candidate = || {
                let started = std::time::Instant::now();
                let output = fnp_argsort
                    .call1((black_box(&data),))
                    .expect("fnp argsort arm");
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum(&output),
                }
            };
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract_with_sampling(
                    &row,
                    &mut observe_incumbent,
                    &mut observe_candidate,
                    rounds,
                    MIN_OF,
                );
            let verdict =
                common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
            println!(
                "ARGSORT_GRID row={row} PARITY=pass checksum={ours_sum:016x} numpy={version} \
                 verdict={verdict} numpy_median_ms={:.4} fnp_median_ms={:.4} \
                 numpy_over_fnp_median={:.4} numpy_over_fnp_ci95=[{:.4},{:.4}] \
                 numpy_null_ratio={:.4} fnp_null_ratio={:.4} rounds={rounds} min_of={MIN_OF} \
                 profile=release",
                effect.arm_a_median_ns / 1_000_000.0,
                effect.arm_b_median_ns / 1_000_000.0,
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                incumbent_null.ratio_median,
                candidate_null.ratio_median,
            );
        }
    });
}

fn main() {
    common::gated_main_with_source(
        include_str!("criterion_python_argsort.rs"),
        &[
            (
                "bench_argsort_default_grid_vs_numpy",
                bench_argsort_default_grid_vs_numpy,
            ),
            (
                "bench_argsort_numeric_stable_boundary",
                bench_argsort_numeric_stable_boundary,
            ),
            (
                "bench_argsort_lastaxis_stable_boundary",
                bench_argsort_lastaxis_stable_boundary,
            ),
            (
                "bench_argsort_radix_stable_boundary",
                bench_argsort_radix_stable_boundary,
            ),
            (
                "bench_argsort_radix_float_boundary",
                bench_argsort_radix_float_boundary,
            ),
            (
                "bench_argsort_default_int_radix_boundary",
                bench_argsort_default_int_radix_boundary,
            ),
            (
                "bench_argsort_default_float_radix_boundary",
                bench_argsort_default_float_radix_boundary,
            ),
            (
                "bench_argsort_datetime_radix_boundary",
                bench_argsort_datetime_radix_boundary,
            ),
        ],
    );
}
