// Fixture construction in a benchmark; retained capacity is not meaningful here.
#![allow(clippy::repeat_vec_with_capacity)]

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
use pyo3::types::{PyAnyMethods, PyDict, PyModule, PyModuleMethods};
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
                "bench_flat_i64_sort_256_stage_profile",
                bench_flat_i64_sort_256_stage_profile,
            ),
            (
                "bench_flat_i64_sort_256_dual_null",
                bench_flat_i64_sort_256_dual_null,
            ),
            (
                "bench_flat_i64_sort_256_allocation_control",
                bench_flat_i64_sort_256_allocation_control,
            ),
            (
                "bench_flat_i64_sort_256_median_decomposition",
                bench_flat_i64_sort_256_median_decomposition,
            ),
            (
                "bench_flat_i64_sort_256_entry_decomposition",
                bench_flat_i64_sort_256_entry_decomposition,
            ),
            (
                "bench_pyfunction_call_shape_price",
                bench_pyfunction_call_shape_price,
            ),
            (
                "bench_flat_i64_sort_256_single_arm",
                bench_flat_i64_sort_256_single_arm,
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
/// WHERE DO THE i64 n=256 SORT NANOSECONDS ACTUALLY GO
/// (`franken_numpy-ixs5y.409`)?
///
/// The cell is a live-NumPy regression - NumPy 1754 ns against our 4158 ns - and the
/// remaining cost was characterised as "Python-entry / copy + PDQ". Those are three very
/// different levers and only one of them is the comparison path, so this decomposes the
/// route before anyone optimises it:
///
///   1. `numpy.empty(n, "int64")` - the output allocation the route must do, because
///      `np.sort` returns a fresh owning array.
///   2. `PyBuffer` acquisition plus `copy_from_slice` of the 2 KiB input.
///   3. `sort_unstable` - the actual comparison/branch path.
///
/// Each stage is timed on its own, min-of-many so a scheduler blip cannot inflate a
/// stage, and the three are printed next to the whole-route and NumPy figures. If the
/// sort is a minority of the total then a faster comparison kernel cannot close this
/// cell, and that is worth knowing BEFORE writing a sorting network.
///
/// This is a decomposition of OUR arm, not a vs-NumPy claim: NumPy's own total is
/// reported alongside purely as the target to beat, and the stage numbers are self-timed
/// rather than contract-gated.
fn bench_flat_i64_sort_256_stage_profile(_c: &mut criterion::Criterion) {
    use std::time::Instant;
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let numpy = py.import("numpy").expect("numpy incumbent");
        let ns = PyDict::new(py);
        ns.set_item("np", &numpy).expect("bind numpy");
        py.run(
            std::ffi::CString::new(
                "rng = np.random.default_rng(20260824)\na = rng.integers(-(1 << 62), 1 << 62, 256, dtype=np.int64)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("int64 corpus setup");
        let input = ns.get_item("a").expect("corpus");
        let np_sort = numpy.getattr("sort").expect("numpy.sort");
        let n = 256usize;
        const REPS: usize = 2000;

        // Stage 1: the output allocation, positional dtype (the form the route uses).
        let mut alloc_min = u128::MAX;
        for _ in 0..REPS {
            let started = Instant::now();
            let out = numpy
                .call_method1(pyo3::intern!(py, "empty"), (n, "int64"))
                .expect("numpy.empty");
            let elapsed = started.elapsed().as_nanos();
            black_box(&out);
            alloc_min = alloc_min.min(elapsed);
        }

        // Stage 2: buffer acquisition + the 2 KiB copy, on a buffer allocated ONCE so
        // this stage does not re-charge stage 1.
        let scratch = numpy
            .call_method1(pyo3::intern!(py, "empty"), (n, "int64"))
            .expect("scratch");
        let src_buffer = pyo3::buffer::PyBuffer::<i64>::get(&input).expect("src buffer");
        let src_cells = src_buffer.as_slice(py).expect("src slice");
        let src: &[i64] =
            unsafe { std::slice::from_raw_parts(src_cells.as_ptr().cast::<i64>(), n) };
        let mut copy_min = u128::MAX;
        for _ in 0..REPS {
            let started = Instant::now();
            let out_buffer = pyo3::buffer::PyBuffer::<i64>::get(&scratch).expect("out buffer");
            let out_cells = out_buffer.as_mut_slice(py).expect("out slice");
            let dst: &mut [i64] =
                unsafe { std::slice::from_raw_parts_mut(out_cells.as_ptr() as *mut i64, n) };
            dst.copy_from_slice(src);
            let elapsed = started.elapsed().as_nanos();
            black_box(dst.as_ptr());
            copy_min = copy_min.min(elapsed);
        }

        // Stage 3: the comparison/branch path alone, on a Rust buffer refilled from the
        // SAME unsorted corpus each rep - sorting an already-sorted slice would flatter
        // PDQ enormously and is the obvious way to get this stage wrong.
        let mut work = vec![0i64; n];
        let mut sort_min = u128::MAX;
        for _ in 0..REPS {
            work.copy_from_slice(src);
            let started = Instant::now();
            fnp_ufunc::sort_small::sort_i64(&mut work);
            let elapsed = started.elapsed().as_nanos();
            black_box(work.as_ptr());
            sort_min = sort_min.min(elapsed);
        }

        // NumPy's whole call, as the target.
        let mut numpy_min = u128::MAX;
        for _ in 0..REPS {
            let started = Instant::now();
            let out = np_sort.call1((&input,)).expect("numpy.sort");
            let elapsed = started.elapsed().as_nanos();
            black_box(&out);
            numpy_min = numpy_min.min(elapsed);
        }

        // OUR WHOLE ROUTE, timed HERE rather than compared against a median banked in
        // another run. The residual below is only meaningful if the route and the stages
        // are the same statistic, on the same host, in the same moment - a min-of-2000
        // stage sum subtracted from someone else's median would invent overhead that is
        // really just min-vs-median.
        let module = PyModule::new(py, "fnp_python_sort_profile").expect("bench module");
        fnp_python(&module).expect("initialize fnp module");
        let fnp_sort = module.getattr("sort").expect("fnp.sort");
        assert!(
            !fnp_sort.is(&np_sort),
            "dispatch trap: fnp.sort resolved to the NumPy callable"
        );
        let mut route_min = u128::MAX;
        for _ in 0..REPS {
            let started = Instant::now();
            let out = fnp_sort.call1((&input,)).expect("fnp.sort");
            let elapsed = started.elapsed().as_nanos();
            black_box(&out);
            route_min = route_min.min(elapsed);
        }

        let stages = alloc_min + copy_min + sort_min;
        let residual = route_min.saturating_sub(stages);
        println!(
            "I64_SORT_STAGE_PROFILE n={n} reps={REPS} statistic=min_ns \
             numpy_whole_call_ns={numpy_min} fnp_whole_route_ns={route_min} \
             alloc_ns={alloc_min} buffer_plus_copy_ns={copy_min} sort_unstable_ns={sort_min} \
             stage_sum_ns={stages} entry_residual_ns={residual} \
             sort_share_of_route={:.4} residual_share_of_route={:.4} \
             route_floor_if_sort_were_free_ns={} \
             is_decomposition_of_our_arm_not_a_vs_numpy_claim=true",
            sort_min as f64 / route_min as f64,
            residual as f64 / route_min as f64,
            route_min - sort_min,
        );
    });
}

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

/// MEDIAN-DOMAIN decomposition of the `int64` n=256 sort route.
///
/// `bench_flat_i64_sort_256_stage_profile` reports min-of-N, and min-of-N said the
/// route was only 1.10x behind NumPy while the dual-null median contract on the same
/// host said 1.37x.  A min and a median are not the same statistic and the gap between
/// them is exactly the thing that has to be attributed, so this group re-decomposes the
/// route with the median that the contract actually reports, sampling every stage
/// ROUND-ROBIN in one loop so that machine drift lands on all stages alike instead of
/// on whichever one happened to be timed last.
fn bench_flat_i64_sort_256_median_decomposition(_c: &mut criterion::Criterion) {
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_sort_median_decomp").expect("bench module");
        fnp_python(&module).expect("initialize fnp module");
        let numpy = py.import("numpy").expect("numpy incumbent");
        let ns = PyDict::new(py);
        ns.set_item("np", &numpy).expect("bind numpy");
        py.run(
            std::ffi::CString::new(
                "rng = np.random.default_rng(20260824)\n\
a = rng.integers(-(1 << 62), 1 << 62, 256, dtype=np.int64)\n\
a[:16] = np.array([np.iinfo(np.int64).min, np.iinfo(np.int64).max, -1, 0, 1, -1, 0, 1, -7, 7, -7, 7, 13, 13, -13, -13], dtype=np.int64)\n\
rng.shuffle(a)\n\
dt = np.dtype('int64')\n\
empty = np.empty\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("int64 corpus setup");
        let input = ns.get_item("a").expect("corpus");
        let cached_dtype = ns.get_item("dt").expect("cached dtype object");
        let cached_empty = ns.get_item("empty").expect("cached numpy.empty callable");
        let np_sort = numpy.getattr("sort").expect("numpy.sort");
        let fnp_sort = module.getattr("sort").expect("fnp.sort");
        assert!(
            !fnp_sort.is(&np_sort),
            "dispatch trap: fnp.sort resolved to the NumPy callable"
        );
        let n = 256usize;

        // A numpy-allocated scratch buffer AND a Rust Vec holding the same bytes, so the
        // kernel is timed on both provenances.  A Rust `Vec` arm is known to carry a
        // buffer-provenance tax against numpy-allocated memory, and the shipped route
        // sorts a numpy buffer, so the Rust-Vec number alone would misprice the kernel.
        let src_buffer = pyo3::buffer::PyBuffer::<i64>::get(&input).expect("src buffer");
        let src_cells = src_buffer.as_slice(py).expect("src slice");
        let src: &[i64] =
            unsafe { std::slice::from_raw_parts(src_cells.as_ptr().cast::<i64>(), n) };
        let scratch = numpy
            .call_method1(pyo3::intern!(py, "empty"), (n, "int64"))
            .expect("scratch");
        let scratch_buffer = pyo3::buffer::PyBuffer::<i64>::get(&scratch).expect("scratch buffer");
        let scratch_cells = scratch_buffer.as_mut_slice(py).expect("scratch slice");
        let np_work: &mut [i64] =
            unsafe { std::slice::from_raw_parts_mut(scratch_cells.as_ptr() as *mut i64, n) };
        let mut rust_work = vec![0i64; n];

        const REPS: usize = 6000;
        const STAGES: usize = 8;
        let labels = [
            "numpy_sort_whole_call",
            "fnp_sort_whole_route",
            "alloc_empty_n_str_int64",
            "alloc_cached_empty_cached_dtype",
            "sort_kernel_on_numpy_buffer",
            "sort_kernel_on_rust_vec",
            "buffer_get_plus_2kib_copy",
            "empty_loop_timer_overhead",
        ];
        let mut samples: Vec<Vec<u64>> = vec![Vec::with_capacity(REPS); STAGES];

        for _ in 0..REPS {
            // 0: the incumbent, whole public call.
            let started = std::time::Instant::now();
            let out = np_sort.call1((black_box(&input),)).expect("numpy.sort");
            samples[0].push(started.elapsed().as_nanos() as u64);
            drop(black_box(out));

            // 1: our whole public route.
            let started = std::time::Instant::now();
            let out = fnp_sort.call1((black_box(&input),)).expect("fnp.sort");
            samples[1].push(started.elapsed().as_nanos() as u64);
            drop(black_box(out));

            // 2: the allocation the route performs today.
            let started = std::time::Instant::now();
            let out = numpy
                .call_method1(pyo3::intern!(py, "empty"), (n, "int64"))
                .expect("numpy.empty positional str dtype");
            samples[2].push(started.elapsed().as_nanos() as u64);
            drop(black_box(out));

            // 3: the same allocation off a cached callable and a cached dtype OBJECT,
            //    which is the candidate replacement for stage 2.
            let started = std::time::Instant::now();
            let out = cached_empty
                .call1((n, &cached_dtype))
                .expect("cached numpy.empty");
            samples[3].push(started.elapsed().as_nanos() as u64);
            drop(black_box(out));

            // 4: the comparison kernel alone, on the numpy-allocated buffer the route
            //    actually sorts, refilled from the unsorted corpus each rep.
            np_work.copy_from_slice(src);
            let started = std::time::Instant::now();
            fnp_ufunc::sort_small::sort_i64(np_work);
            samples[4].push(started.elapsed().as_nanos() as u64);
            black_box(np_work.as_ptr());

            // 5: the same kernel on Rust-owned memory.
            rust_work.copy_from_slice(src);
            let started = std::time::Instant::now();
            fnp_ufunc::sort_small::sort_i64(&mut rust_work);
            samples[5].push(started.elapsed().as_nanos() as u64);
            black_box(rust_work.as_ptr());

            // 6: buffer acquisition + the 2 KiB copy, on an already-allocated output.
            let started = std::time::Instant::now();
            let out_buffer = pyo3::buffer::PyBuffer::<i64>::get(&scratch).expect("out buffer");
            let out_cells = out_buffer.as_mut_slice(py).expect("out slice");
            let dst: &mut [i64] =
                unsafe { std::slice::from_raw_parts_mut(out_cells.as_ptr() as *mut i64, n) };
            dst.copy_from_slice(src);
            samples[6].push(started.elapsed().as_nanos() as u64);
            black_box(dst.as_ptr());

            // 7: the timing harness measuring nothing, so every number above can be read
            //    net of the `Instant` pair itself.
            let started = std::time::Instant::now();
            samples[7].push(started.elapsed().as_nanos() as u64);
        }

        let stat = |v: &mut Vec<u64>, q: f64| -> u64 {
            v.sort_unstable();
            v[((v.len() - 1) as f64 * q) as usize]
        };
        for (index, label) in labels.iter().enumerate() {
            let min = *samples[index].iter().min().expect("samples");
            let p25 = stat(&mut samples[index], 0.25);
            let median = stat(&mut samples[index], 0.50);
            let p75 = stat(&mut samples[index], 0.75);
            println!(
                "I64_SORT_MEDIAN_DECOMP stage={label} reps={REPS} \\
                 min_ns={min} p25_ns={p25} median_ns={median} p75_ns={p75} \\
                 median_over_min={:.4}",
                median as f64 / min.max(1) as f64,
            );
        }
    });
}

/// MEDIAN-DOMAIN decomposition of the `int64` n=256 route's ENTRY, and a price
/// list for the candidates that could replace it.
///
/// The stage decomposition put 591 ns of the route's 2174 ns median outside the
/// allocation, the copy and the comparison kernel.  That residual is the Python
/// entry, and this group takes it apart into the individual reached operations
/// so a lever can be chosen against measured numbers instead of a guess about
/// which attribute lookup is expensive.  Sampled ROUND-ROBIN in one loop, same
/// as the stage decomposition, so drift lands on every stage alike.
fn bench_flat_i64_sort_256_entry_decomposition(_c: &mut criterion::Criterion) {
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_sort_entry_decomp").expect("bench module");
        fnp_python(&module).expect("initialize fnp module");
        let numpy = py.import("numpy").expect("numpy incumbent");
        let ns = PyDict::new(py);
        ns.set_item("np", &numpy).expect("bind numpy");
        py.run(
            std::ffi::CString::new(
                "rng = np.random.default_rng(20260824)\n\
a = rng.integers(-(1 << 62), 1 << 62, 256, dtype=np.int64)\n\
dt = np.dtype('int64')\n\
empty = np.empty\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("int64 corpus setup");
        let input = ns.get_item("a").expect("corpus");
        let cached_dtype = ns.get_item("dt").expect("cached dtype object");
        let cached_empty = ns.get_item("empty").expect("cached numpy.empty callable");
        let np_sort = numpy.getattr("sort").expect("numpy.sort");
        let fnp_sort = module.getattr("sort").expect("fnp.sort");
        assert!(
            !fnp_sort.is(&np_sort),
            "dispatch trap: fnp.sort resolved to the NumPy callable"
        );
        let n = 256usize;

        // Parity check for the `a.copy()` candidate BEFORE it is timed: it has
        // to reproduce numpy.sort's result lifecycle, i.e. a fresh, writable,
        // C-contiguous int64 array of the same shape - and, unlike the
        // `numpy.empty` spelling it would replace, it also has to carry the
        // operand's own bytes so the kernel can sort in place.
        {
            let copied = input.call_method0("copy").expect("ndarray.copy");
            let allocated = numpy
                .call_method1(pyo3::intern!(py, "empty"), (n, "int64"))
                .expect("numpy.empty");
            assert!(copied.get_type().is(allocated.get_type()), "copy type");
            assert_eq!(
                copied.getattr("dtype").unwrap().str().unwrap().to_string(),
                "int64",
                "copy dtype"
            );
            assert_eq!(
                copied
                    .getattr("shape")
                    .unwrap()
                    .extract::<Vec<usize>>()
                    .unwrap(),
                vec![n],
                "copy shape"
            );
            assert!(
                copied
                    .getattr("flags")
                    .unwrap()
                    .getattr("c_contiguous")
                    .unwrap()
                    .extract::<bool>()
                    .unwrap(),
                "copy layout"
            );
            assert_eq!(
                copied
                    .call_method0("tobytes")
                    .unwrap()
                    .extract::<Vec<u8>>()
                    .unwrap(),
                input
                    .call_method0("tobytes")
                    .unwrap()
                    .extract::<Vec<u8>>()
                    .unwrap(),
                "copy must carry the operand's bytes"
            );
        }

        const REPS: usize = 6000;
        const STAGES: usize = 10;
        let labels = [
            "numpy_sort_whole_call",
            "fnp_sort_whole_route",
            "alloc_empty_n_str_int64",
            "alloc_cached_empty_cached_dtype",
            "input_copy_method",
            "pybuffer_get_input_plus_as_slice",
            "dtype_char_getattr_chain",
            "python_len",
            "dtype_names_probe",
            "empty_loop_timer_overhead",
        ];
        let mut samples: Vec<Vec<u64>> = vec![Vec::with_capacity(REPS); STAGES];

        for _ in 0..REPS {
            let started = std::time::Instant::now();
            let out = np_sort.call1((black_box(&input),)).expect("numpy.sort");
            samples[0].push(started.elapsed().as_nanos() as u64);
            drop(black_box(out));

            let started = std::time::Instant::now();
            let out = fnp_sort.call1((black_box(&input),)).expect("fnp.sort");
            samples[1].push(started.elapsed().as_nanos() as u64);
            drop(black_box(out));

            let started = std::time::Instant::now();
            let out = numpy
                .call_method1(pyo3::intern!(py, "empty"), (n, "int64"))
                .expect("numpy.empty positional str dtype");
            samples[2].push(started.elapsed().as_nanos() as u64);
            drop(black_box(out));

            let started = std::time::Instant::now();
            let out = cached_empty
                .call1((n, &cached_dtype))
                .expect("cached numpy.empty");
            samples[3].push(started.elapsed().as_nanos() as u64);
            drop(black_box(out));

            // THE STRUCTURAL CANDIDATE: one call that allocates AND fills,
            // replacing `numpy.empty` + a `PyBuffer` on the input + a 2 KiB
            // copy loop, and removing one of the route's two buffer
            // acquisitions with it.
            let started = std::time::Instant::now();
            let out = input
                .call_method0(pyo3::intern!(py, "copy"))
                .expect("ndarray.copy");
            samples[4].push(started.elapsed().as_nanos() as u64);
            drop(black_box(out));

            let started = std::time::Instant::now();
            let buffer = pyo3::buffer::PyBuffer::<i64>::get(&input).expect("input buffer");
            let cells = buffer.as_slice(py).expect("input slice");
            samples[5].push(started.elapsed().as_nanos() as u64);
            black_box(cells.as_ptr());
            drop(buffer);

            let started = std::time::Instant::now();
            let typechar = input
                .getattr(pyo3::intern!(py, "dtype"))
                .expect("dtype")
                .getattr(pyo3::intern!(py, "char"))
                .expect("char")
                .extract::<char>()
                .expect("char extract");
            samples[6].push(started.elapsed().as_nanos() as u64);
            black_box(typechar);

            let started = std::time::Instant::now();
            let len = input.len().expect("len");
            samples[7].push(started.elapsed().as_nanos() as u64);
            black_box(len);

            // What the two structured-sort probes reach before declining.
            let started = std::time::Instant::now();
            let names = input
                .getattr(pyo3::intern!(py, "dtype"))
                .expect("dtype")
                .getattr(pyo3::intern!(py, "names"))
                .expect("names");
            samples[8].push(started.elapsed().as_nanos() as u64);
            black_box(names.is_none());

            let started = std::time::Instant::now();
            samples[9].push(started.elapsed().as_nanos() as u64);
        }

        let stat = |v: &mut Vec<u64>, q: f64| -> u64 {
            v.sort_unstable();
            v[((v.len() - 1) as f64 * q) as usize]
        };
        for (index, label) in labels.iter().enumerate() {
            let min = *samples[index].iter().min().expect("samples");
            let p25 = stat(&mut samples[index], 0.25);
            let median = stat(&mut samples[index], 0.50);
            let p75 = stat(&mut samples[index], 0.75);
            println!(
                "I64_SORT_ENTRY_DECOMP stage={label} reps={REPS} \\
                 min_ns={min} p25_ns={p25} median_ns={median} p75_ns={p75}"
            );
        }
    });
}

/// What a `(*args, **kwargs)` signature costs per call, against a typed one.
///
/// Two `#[pyfunction]`s with IDENTICAL bodies - both return their first argument
/// untouched - differing only in the declared signature.  PyO3 gives a typed
/// signature `METH_FASTCALL | METH_KEYWORDS`, where CPython hands the callee a
/// C array of borrowed argument pointers; it gives `(*args, **kwargs)` plain
/// `METH_VARARGS | METH_KEYWORDS`, where CPython must BUILD a tuple (and, when
/// keywords are present, a dict) on every call before the callee sees anything.
/// NumPy's own `np.sort` is on the fast path.
///
/// This is the shared wrapper floor the ufunc-method family converged onto and
/// the same residual the `int64` n=256 sort route is left holding, so it is
/// worth ONE number rather than another guess.  The bodies do no work, so the
/// difference between the two arms is the calling convention and nothing else.
#[pyo3::pyfunction]
#[pyo3(signature = (*args, **kwargs))]
fn call_shape_varargs<'py>(
    args: &pyo3::Bound<'py, pyo3::types::PyTuple>,
    kwargs: Option<&pyo3::Bound<'py, pyo3::types::PyDict>>,
) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    let _ = kwargs;
    args.get_item(0)
}

/// The same function with `numpy.sort`'s own parameter list spelled out.
#[pyo3::pyfunction]
#[pyo3(signature = (a, axis=None, kind=None, order=None, stable=None))]
fn call_shape_typed<'py>(
    a: pyo3::Bound<'py, pyo3::types::PyAny>,
    axis: Option<pyo3::Bound<'py, pyo3::types::PyAny>>,
    kind: Option<pyo3::Bound<'py, pyo3::types::PyAny>>,
    order: Option<pyo3::Bound<'py, pyo3::types::PyAny>>,
    stable: Option<pyo3::Bound<'py, pyo3::types::PyAny>>,
) -> pyo3::Bound<'py, pyo3::types::PyAny> {
    let _ = (axis, kind, order, stable);
    a
}

fn bench_pyfunction_call_shape_price(_c: &mut criterion::Criterion) {
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let numpy = py.import("numpy").expect("numpy");
        let module = PyModule::new(py, "fnp_python_call_shape").expect("bench module");
        module
            .add_function(pyo3::wrap_pyfunction!(call_shape_varargs, &module).unwrap())
            .expect("bind varargs arm");
        module
            .add_function(pyo3::wrap_pyfunction!(call_shape_typed, &module).unwrap())
            .expect("bind typed arm");
        let varargs = module.getattr("call_shape_varargs").expect("varargs");
        let typed = module.getattr("call_shape_typed").expect("typed");

        let ns = PyDict::new(py);
        ns.set_item("np", &numpy).expect("bind numpy");
        py.run(
            std::ffi::CString::new(
                "rng = np.random.default_rng(20260824)\n\
a = rng.integers(-(1 << 62), 1 << 62, 256, dtype=np.int64)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("corpus");
        let input = ns.get_item("a").expect("corpus");

        // Both arms must genuinely return the operand, or one of them is being
        // timed doing less work than the other.
        assert!(
            varargs.call1((&input,)).unwrap().is(&input),
            "varargs arm did not return its operand"
        );
        assert!(
            typed.call1((&input,)).unwrap().is(&input),
            "typed arm did not return its operand"
        );

        let mut observe_typed = || {
            let started = std::time::Instant::now();
            let result = typed.call1((black_box(&input),)).expect("typed arm");
            common::ContractObservation {
                elapsed: started.elapsed(),
                checksum: result.is_none() as u64,
            }
        };
        let mut observe_varargs = || {
            let started = std::time::Instant::now();
            let result = varargs.call1((black_box(&input),)).expect("varargs arm");
            common::ContractObservation {
                elapsed: started.elapsed(),
                checksum: result.is_none() as u64,
            }
        };
        let (effect, typed_null, varargs_null) = common::run_dual_null_median_ci_contract(
            "pyfunction_call_shape_typed_over_varargs",
            &mut observe_typed,
            &mut observe_varargs,
        );
        let verdict = common::dual_null_contract_verdict(effect, typed_null, varargs_null);
        println!(
            "CALL_SHAPE_PRICE verdict={verdict} \\
             typed_median_ns={:.3} varargs_median_ns={:.3} \\
             varargs_minus_typed_ns={:.3} \\
             typed_over_varargs={:.6} ci95=[{:.6},{:.6}] \\
             typed_null={:.6} typed_null_ci95=[{:.6},{:.6}] \\
             varargs_null={:.6} varargs_null_ci95=[{:.6},{:.6}] \\
             bodies_identical=true same_binary=true \\
             scope=calling_convention_only_no_work_in_either_body",
            effect.arm_a_median_ns,
            effect.arm_b_median_ns,
            effect.arm_b_median_ns - effect.arm_a_median_ns,
            effect.ratio_median,
            effect.ratio_ci_low,
            effect.ratio_ci_high,
            typed_null.ratio_median,
            typed_null.ratio_ci_low,
            typed_null.ratio_ci_high,
            varargs_null.ratio_median,
            varargs_null.ratio_ci_low,
            varargs_null.ratio_ci_high,
        );
    });
}

/// ONE arm, N calls, no timing - a target for `perf stat`.
///
/// The `int64` n=256 sort cell's remaining deficit is ~326 ns spread over many
/// pieces of Python entry, each of them 1-4% of the route. That is below what
/// this harness can decide from wall clock in any reasonable number of runs, and
/// it is exactly the situation counted attribution is for: instructions retired
/// do not move with machine load, so a counter diff stays valid on a host that a
/// wall-clock contract could not be run on at all.
///
/// `FNP_SORT_ARM` selects `fnp`, `numpy`, or `none`. The `none` arm does
/// everything except the calls, so subtracting it removes interpreter startup,
/// the numpy import and the corpus build - all of which are large and identical
/// across arms - and leaves N times the per-call cost.
fn bench_flat_i64_sort_256_single_arm(_c: &mut criterion::Criterion) {
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_sort_single_arm").expect("bench module");
        fnp_python(&module).expect("initialize fnp module");
        let numpy = py.import("numpy").expect("numpy incumbent");
        let ns = PyDict::new(py);
        ns.set_item("np", &numpy).expect("bind numpy");
        py.run(
            std::ffi::CString::new(
                "rng = np.random.default_rng(20260824)\n\
a = rng.integers(-(1 << 62), 1 << 62, 256, dtype=np.int64)\n\
a[:16] = np.array([np.iinfo(np.int64).min, np.iinfo(np.int64).max, -1, 0, 1, -1, 0, 1, -7, 7, -7, 7, 13, 13, -13, -13], dtype=np.int64)\n\
rng.shuffle(a)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("int64 corpus setup");
        let input = ns.get_item("a").expect("corpus");
        let arm = std::env::var("FNP_SORT_ARM").unwrap_or_else(|_| "none".to_owned());
        let calls: usize = std::env::var("FNP_SORT_CALLS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(400_000);
        let np_sort = numpy.getattr("sort").expect("numpy.sort");
        let fnp_sort = module.getattr("sort").expect("fnp.sort");
        assert!(
            !fnp_sort.is(&np_sort),
            "dispatch trap: fnp.sort resolved to the NumPy callable"
        );
        // Prove the native route is engaged BEFORE the counted loop, so a counter
        // diff can never be a delegation in disguise.
        //
        // RAW STRING, DELIBERATELY: a `\n\` continuation inside a normal Rust
        // string literal STRIPS the leading whitespace of the next line, which
        // turns an indented Python block into an `IndentationError`. That is not
        // hypothetical - it is how this probe first shipped, and because the
        // panic happened before the counted loop, `perf` dutifully measured six
        // runs of a process that never made a single call.
        ns.set_item("fnp", &module).expect("bind fnp module");
        py.run(
            std::ffi::CString::new(
                r#"
original_sort = np.sort
calls_seen = []
def spy(*args, **kwargs):
    calls_seen.append(1)
    return original_sort(*args, **kwargs)
np.sort = spy
try:
    probe = fnp.sort(a)
finally:
    np.sort = original_sort
"#,
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("engagement probe");
        let numpy_sort_calls = ns
            .get_item("calls_seen")
            .expect("probe calls")
            .len()
            .expect("probe call count");
        assert_eq!(
            numpy_sort_calls, 0,
            "fnp.sort delegated to numpy.sort; a counter diff would be meaningless"
        );
        match arm.as_str() {
            "fnp" => {
                for _ in 0..calls {
                    black_box(fnp_sort.call1((black_box(&input),)).expect("fnp.sort"));
                }
            }
            "numpy" => {
                for _ in 0..calls {
                    black_box(np_sort.call1((black_box(&input),)).expect("numpy.sort"));
                }
            }
            _ => {}
        }
        println!(
            "SINGLE_ARM arm={arm} calls={calls} n=256 dtype=int64 \
             candidate_numpy_sort_calls_preflight={numpy_sort_calls} \
             statistic=none_timing_is_external"
        );
    });
}
