//! String-domain criterion benches (sort / unique / searchsorted / isin /
//! union1d / setops / setxor / bytes) split out of the monolithic
//! `criterion_python_surface.rs` into their own per-domain bench binary, so a
//! per-domain run compiles only these groups instead of all 200 monolith
//! benches. This is what unblocks the cold-`rch` time cap for a scoped run.
//! See bead deadlock-audit-x7nnf.

#[path = "common/mod.rs"]
mod common;

use common::ensure_numpy_available;
use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyDict, PyModule};
use std::hint::black_box;
use std::time::Duration;

fn bench_string_sort_boundary(c: &mut Criterion) {
    // np.sort on a 1-D fixed-width unicode ('U') array. numpy sorts with a slow single-threaded
    // per-record codepoint comparator (~280ms @2M U8). For Latin-1 strings (all codepoints <0x100)
    // fnp sorts record indices by memcmp (== codepoint order) in parallel and gathers — bit-exact.
    let mut group = c.benchmark_group("python_string_sort_boundary");
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
x8 = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
x4 = rng.integers(97, 123, (2_000_000, 4), dtype=np.uint32).reshape(-1).view('U4')\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("string sort setup");
        let x8 = ns.get_item("x8").expect("x8");
        let x4 = ns.get_item("x4").expect("x4");
        let fnp_sort = module.getattr("sort").expect("fnp sort");
        let numpy_sort = numpy.getattr("sort").expect("numpy sort");
        let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
        // Correctness gate: fnp.sort == numpy.sort for U8 and U4; panics on mismatch.
        for (arr, label) in [(&x8, "U8"), (&x4, "U4")] {
            let f = fnp_sort.call1((arr,)).expect("fnp sort");
            let n = numpy_sort.call1((arr,)).expect("numpy sort");
            let eq: bool = np_array_equal
                .call1((&f, &n))
                .expect("array_equal")
                .extract()
                .expect("bool");
            assert!(eq, "string sort correctness mismatch: dtype={label}");
        }
        group.bench_function("fnp_sort_U8_2m", |b| {
            b.iter(|| black_box(fnp_sort.call1((&x8,)).expect("fnp sort U8")));
        });
        group.bench_function("numpy_sort_U8_2m", |b| {
            b.iter(|| black_box(numpy_sort.call1((&x8,)).expect("numpy sort U8")));
        });
        group.bench_function("fnp_sort_U4_2m", |b| {
            b.iter(|| black_box(fnp_sort.call1((&x4,)).expect("fnp sort U4")));
        });
        group.bench_function("numpy_sort_U4_2m", |b| {
            b.iter(|| black_box(numpy_sort.call1((&x4,)).expect("numpy sort U4")));
        });
    });
    group.finish();
}

fn bench_string_unique_boundary(c: &mut Criterion) {
    // np.unique on a 1-D fixed-width unicode ('U') array. numpy sorts with the slow per-record
    // codepoint comparator then dedups. For Latin-1 strings fnp sorts packed word keys, dedups
    // adjacent-equal records, and gathers the distinct records — bit-exact. The U16 case covers
    // the two-word wide-key route; U8 mostly-unique and U4 heavy-dedup guard the existing u64 path.
    let mut group = c.benchmark_group("python_string_unique_boundary");
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
x8 = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
xd = rng.integers(97, 101, (2_000_000, 4), dtype=np.uint32).reshape(-1).view('U4')\n\
x16 = rng.integers(97, 123, (1_000_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("string unique setup");
        let x8 = ns.get_item("x8").expect("x8");
        let xd = ns.get_item("xd").expect("xd");
        let x16 = ns.get_item("x16").expect("x16");
        let fnp_unique = module.getattr("unique").expect("fnp unique");
        let numpy_unique = numpy.getattr("unique").expect("numpy unique");
        let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
        // Correctness gate: fnp.unique == numpy.unique for both cases; panics on mismatch.
        for (arr, label) in [
            (&x8, "U8-mostly-unique"),
            (&xd, "U4-heavy-dedup"),
            (&x16, "U16-mostly-unique"),
        ] {
            let f = fnp_unique.call1((arr,)).expect("fnp unique");
            let n = numpy_unique.call1((arr,)).expect("numpy unique");
            let eq: bool = np_array_equal
                .call1((&f, &n))
                .expect("array_equal")
                .extract()
                .expect("bool");
            assert!(eq, "string unique correctness mismatch: case={label}");
        }
        group.bench_function("fnp_unique_U8_2m", |b| {
            b.iter(|| black_box(fnp_unique.call1((&x8,)).expect("fnp unique U8")));
        });
        group.bench_function("numpy_unique_U8_2m", |b| {
            b.iter(|| black_box(numpy_unique.call1((&x8,)).expect("numpy unique U8")));
        });
        group.bench_function("fnp_unique_U4_dedup_2m", |b| {
            b.iter(|| black_box(fnp_unique.call1((&xd,)).expect("fnp unique U4")));
        });
        group.bench_function("numpy_unique_U4_dedup_2m", |b| {
            b.iter(|| black_box(numpy_unique.call1((&xd,)).expect("numpy unique U4")));
        });
        group.bench_function("fnp_unique_U16_1m", |b| {
            b.iter(|| black_box(fnp_unique.call1((&x16,)).expect("fnp unique U16")));
        });
        group.bench_function("numpy_unique_U16_1m", |b| {
            b.iter(|| black_box(numpy_unique.call1((&x16,)).expect("numpy unique U16")));
        });
    });
    group.finish();
}

fn bench_string_unique_full_boundary(c: &mut Criterion) {
    // np.unique on fixed-width bytes with return_index/return_inverse/return_counts. numpy pays the
    // slow record sort plus factorization outputs; fnp carries original indices through the existing
    // byte-record sort and derives group boundaries in one pass.
    let mut group = c.benchmark_group("python_string_unique_full_boundary");
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
rng = np.random.default_rng(1)\n\
a = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint8).view('S8').reshape(-1)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("string unique full setup");
        let a = ns.get_item("a").expect("a");
        let fnp_unique = module.getattr("unique").expect("fnp unique");
        let numpy_unique = numpy.getattr("unique").expect("numpy unique");
        let kw = PyDict::new(py);
        kw.set_item("return_index", true).expect("ri");
        kw.set_item("return_inverse", true).expect("rinv");
        kw.set_item("return_counts", true).expect("rc");
        {
            let got = fnp_unique
                .call((&a,), Some(&kw))
                .expect("fnp unique full S8");
            let exp = numpy_unique
                .call((&a,), Some(&kw))
                .expect("numpy unique full S8");
            let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
            for t in 0..4usize {
                let g = got.get_item(t).expect("got item");
                let e = exp.get_item(t).expect("exp item");
                let eq: bool = np_array_equal
                    .call1((&g, &e))
                    .expect("array_equal")
                    .extract()
                    .expect("bool");
                assert!(
                    eq,
                    "string unique return_* correctness mismatch at tuple index {t}"
                );
            }
        }
        group.bench_function("fnp_unique_S8_full_2m", |b| {
            b.iter(|| {
                black_box(
                    fnp_unique
                        .call((&a,), Some(&kw))
                        .expect("fnp unique full S8"),
                )
            });
        });
        group.bench_function("numpy_unique_S8_full_2m", |b| {
            b.iter(|| {
                black_box(
                    numpy_unique
                        .call((&a,), Some(&kw))
                        .expect("numpy unique full S8"),
                )
            });
        });
    });
    group.finish();
}

fn bench_string_searchsorted_boundary(c: &mut Criterion) {
    // np.searchsorted into a SORTED 1-D unicode ('U') haystack with a 'U' query array. numpy's
    // per-record codepoint binary search is single-threaded and pathologically slow (~2s @2M+2M).
    // For Latin-1 fnp packs fixed-width records into u64 keys, sorts queries once, and monotonic-merges
    // insertion indices byte-exactly.
    let mut group = c.benchmark_group("python_string_searchsorted_boundary");
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
h8 = np.sort(rng.integers(97, 123, (2_000_000, 8), dtype=np.uint32).reshape(-1).view('U8'))\n\
q8 = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("string searchsorted setup");
        let h8 = ns.get_item("h8").expect("h8");
        let q8 = ns.get_item("q8").expect("q8");
        let fnp_ss = module.getattr("searchsorted").expect("fnp searchsorted");
        let numpy_ss = numpy.getattr("searchsorted").expect("numpy searchsorted");
        let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
        // Correctness gate: fnp.searchsorted == numpy.searchsorted for left and right sides.
        for side in ["left", "right"] {
            let kw = PyDict::new(py);
            kw.set_item("side", side).unwrap();
            let f = fnp_ss
                .call((&h8, &q8), Some(&kw))
                .expect("fnp searchsorted");
            let n = numpy_ss
                .call((&h8, &q8), Some(&kw))
                .expect("numpy searchsorted");
            let eq: bool = np_array_equal
                .call1((&f, &n))
                .expect("array_equal")
                .extract()
                .expect("bool");
            assert!(eq, "string searchsorted correctness mismatch: side={side}");
        }
        group.bench_function("fnp_searchsorted_U8_left_2m", |b| {
            b.iter(|| black_box(fnp_ss.call1((&h8, &q8)).expect("fnp ss left")));
        });
        group.bench_function("numpy_searchsorted_U8_left_2m", |b| {
            b.iter(|| black_box(numpy_ss.call1((&h8, &q8)).expect("numpy ss left")));
        });
    });
    group.finish();
}

fn bench_string_isin_boundary(c: &mut Criterion) {
    // np.isin on a 1-D unicode ('U') element array against a 'U' test array. numpy sorts
    // |element|+|test| (~730ms .. 2.4s @2M); fnp builds a hashed record-byte set and does a
    // parallel membership scan — bit-exact for ALL codepoints (equality = byte equality).
    let mut group = c.benchmark_group("python_string_isin_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        // test = 100k members drawn from `a` + 100k random non-members, so both bool branches fire.
        let setup = "import numpy as np\n\
rng = np.random.default_rng(0)\n\
a = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
trand = rng.integers(97, 123, (100_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
test = np.concatenate([a[:100_000], trand])\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("string isin setup");
        let a = ns.get_item("a").expect("a");
        let test = ns.get_item("test").expect("test");
        let fnp_isin = module.getattr("isin").expect("fnp isin");
        let numpy_isin = numpy.getattr("isin").expect("numpy isin");
        let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
        // Correctness gate: fnp.isin == numpy.isin for default and invert=True.
        for inv in [false, true] {
            let kw = PyDict::new(py);
            kw.set_item("invert", inv).unwrap();
            let f = fnp_isin.call((&a, &test), Some(&kw)).expect("fnp isin");
            let n = numpy_isin.call((&a, &test), Some(&kw)).expect("numpy isin");
            let eq: bool = np_array_equal
                .call1((&f, &n))
                .expect("array_equal")
                .extract()
                .expect("bool");
            assert!(eq, "string isin correctness mismatch: invert={inv}");
        }
        group.bench_function("fnp_isin_U8_2m", |b| {
            b.iter(|| black_box(fnp_isin.call1((&a, &test)).expect("fnp isin")));
        });
        group.bench_function("numpy_isin_U8_2m", |b| {
            b.iter(|| black_box(numpy_isin.call1((&a, &test)).expect("numpy isin")));
        });
    });
    group.finish();
}

fn bench_string_union1d_boundary(c: &mut Criterion) {
    // np.union1d of two unicode ('U') arrays = sorted-unique of the concatenation. numpy does 2-3
    // slow per-record sorts (~3.2s @2M+2M); fnp concatenates + routes through the native 'U' unique
    // (memcmp sort + dedup) — bit-exact.
    let mut group = c.benchmark_group("python_string_union1d_boundary");
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
a = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
b = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("string union1d setup");
        let a = ns.get_item("a").expect("a");
        let b = ns.get_item("b").expect("b");
        let fnp_u = module.getattr("union1d").expect("fnp union1d");
        let numpy_u = numpy.getattr("union1d").expect("numpy union1d");
        let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
        // Correctness gate.
        let f = fnp_u.call1((&a, &b)).expect("fnp union1d");
        let n = numpy_u.call1((&a, &b)).expect("numpy union1d");
        let eq: bool = np_array_equal
            .call1((&f, &n))
            .expect("array_equal")
            .extract()
            .expect("bool");
        assert!(eq, "string union1d correctness mismatch");
        group.bench_function("fnp_union1d_U8_2m", |bn| {
            bn.iter(|| black_box(fnp_u.call1((&a, &b)).expect("fnp union1d")));
        });
        group.bench_function("numpy_union1d_U8_2m", |bn| {
            bn.iter(|| black_box(numpy_u.call1((&a, &b)).expect("numpy union1d")));
        });

        // Latin-1 U16: the wide two-word-key branch (dedicated per-operand pack, no concat).
        let setup16 = "import numpy as np\n\
rng = np.random.default_rng(2)\n\
a16 = rng.integers(97, 123, (1_000_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
b16 = rng.integers(97, 123, (1_000_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n";
        let ns16 = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup16).unwrap().as_c_str(),
            Some(&ns16),
            Some(&ns16),
        )
        .expect("string union1d U16 setup");
        let a16 = ns16.get_item("a16").expect("a16");
        let b16 = ns16.get_item("b16").expect("b16");
        let f = fnp_u.call1((&a16, &b16)).expect("fnp union1d U16");
        let n = numpy_u.call1((&a16, &b16)).expect("numpy union1d U16");
        let eq: bool = np_array_equal
            .call1((&f, &n))
            .expect("array_equal")
            .extract()
            .expect("bool");
        assert!(eq, "string union1d U16 correctness mismatch");
        group.bench_function("fnp_union1d_U16_1m", |bn| {
            bn.iter(|| black_box(fnp_u.call1((&a16, &b16)).expect("fnp union1d U16")));
        });
        group.bench_function("numpy_union1d_U16_1m", |bn| {
            bn.iter(|| black_box(numpy_u.call1((&a16, &b16)).expect("numpy union1d U16")));
        });
    });
    group.finish();
}

fn bench_string_setops_boundary(c: &mut Criterion) {
    // np.intersect1d / np.setdiff1d of two unicode ('U') arrays. numpy does 2-3 slow per-record
    // sorts (~3.1-3.7s @2M+2M); fnp = memcmp-sort unique(a) + hashed-set membership filter over b.
    let mut group = c.benchmark_group("python_string_setops_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        // b shares 1M values with a and adds 1M fresh, so intersect and setdiff are both non-trivial.
        let setup = "import numpy as np\n\
rng = np.random.default_rng(0)\n\
a = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
brand = rng.integers(97, 123, (1_000_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
b = np.concatenate([a[:1_000_000], brand])\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("string setops setup");
        let a = ns.get_item("a").expect("a");
        let b = ns.get_item("b").expect("b");
        let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
        // Correctness gate for both ops.
        for op in ["intersect1d", "setdiff1d"] {
            let f = module
                .getattr(op)
                .unwrap()
                .call1((&a, &b))
                .expect("fnp setop");
            let n = numpy
                .getattr(op)
                .unwrap()
                .call1((&a, &b))
                .expect("numpy setop");
            let eq: bool = np_array_equal
                .call1((&f, &n))
                .expect("array_equal")
                .extract()
                .expect("bool");
            assert!(eq, "string {op} correctness mismatch");
        }
        let fnp_i = module.getattr("intersect1d").unwrap();
        let numpy_i = numpy.getattr("intersect1d").unwrap();
        let fnp_d = module.getattr("setdiff1d").unwrap();
        let numpy_d = numpy.getattr("setdiff1d").unwrap();
        group.bench_function("fnp_intersect1d_U8_2m", |bn| {
            bn.iter(|| black_box(fnp_i.call1((&a, &b)).expect("fnp intersect")));
        });
        group.bench_function("numpy_intersect1d_U8_2m", |bn| {
            bn.iter(|| black_box(numpy_i.call1((&a, &b)).expect("numpy intersect")));
        });
        group.bench_function("fnp_setdiff1d_U8_2m", |bn| {
            bn.iter(|| black_box(fnp_d.call1((&a, &b)).expect("fnp setdiff")));
        });
        group.bench_function("numpy_setdiff1d_U8_2m", |bn| {
            bn.iter(|| black_box(numpy_d.call1((&a, &b)).expect("numpy setdiff")));
        });

        // Latin-1 U16: exercises the two-word packed-key branch (U9..U16), which replaces the
        // memcmp record-sort + hashed-membership fallback these widths previously used.
        let setup16 = "import numpy as np\n\
rng = np.random.default_rng(1)\n\
a16 = rng.integers(97, 123, (1_000_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
brand16 = rng.integers(97, 123, (500_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
b16 = np.concatenate([a16[:500_000], brand16])\n";
        let ns16 = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup16).unwrap().as_c_str(),
            Some(&ns16),
            Some(&ns16),
        )
        .expect("string setops U16 setup");
        let a16 = ns16.get_item("a16").expect("a16");
        let b16 = ns16.get_item("b16").expect("b16");
        for op in ["intersect1d", "setdiff1d"] {
            let f = module
                .getattr(op)
                .unwrap()
                .call1((&a16, &b16))
                .expect("fnp setop U16");
            let n = numpy
                .getattr(op)
                .unwrap()
                .call1((&a16, &b16))
                .expect("numpy setop U16");
            let eq: bool = np_array_equal
                .call1((&f, &n))
                .expect("array_equal")
                .extract()
                .expect("bool");
            assert!(eq, "string {op} U16 correctness mismatch");
        }
        group.bench_function("fnp_intersect1d_U16_1m", |bn| {
            bn.iter(|| black_box(fnp_i.call1((&a16, &b16)).expect("fnp intersect U16")));
        });
        group.bench_function("numpy_intersect1d_U16_1m", |bn| {
            bn.iter(|| black_box(numpy_i.call1((&a16, &b16)).expect("numpy intersect U16")));
        });
        group.bench_function("fnp_setdiff1d_U16_1m", |bn| {
            bn.iter(|| black_box(fnp_d.call1((&a16, &b16)).expect("fnp setdiff U16")));
        });
        group.bench_function("numpy_setdiff1d_U16_1m", |bn| {
            bn.iter(|| black_box(numpy_d.call1((&a16, &b16)).expect("numpy setdiff U16")));
        });
    });
    group.finish();
}

fn bench_string_setxor_boundary(c: &mut Criterion) {
    // np.setxor1d of two unicode ('U') arrays = sorted-unique symmetric difference. numpy does 2-3
    // slow per-record sorts (~3.8s @2M+2M); fnp source-tags + memcmp-sorts the concat, then keeps
    // runs present in exactly one array (no hashing) — bit-exact.
    let mut group = c.benchmark_group("python_string_setxor_boundary");
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
a = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
brand = rng.integers(97, 123, (1_000_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
b = np.concatenate([a[:1_000_000], brand])\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("string setxor setup");
        let a = ns.get_item("a").expect("a");
        let b = ns.get_item("b").expect("b");
        let fnp_x = module.getattr("setxor1d").expect("fnp setxor1d");
        let numpy_x = numpy.getattr("setxor1d").expect("numpy setxor1d");
        let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
        let f = fnp_x.call1((&a, &b)).expect("fnp setxor1d");
        let n = numpy_x.call1((&a, &b)).expect("numpy setxor1d");
        let eq: bool = np_array_equal
            .call1((&f, &n))
            .expect("array_equal")
            .extract()
            .expect("bool");
        assert!(eq, "string setxor1d correctness mismatch");
        group.bench_function("fnp_setxor1d_U8_2m", |bn| {
            bn.iter(|| black_box(fnp_x.call1((&a, &b)).expect("fnp setxor1d")));
        });
        group.bench_function("numpy_setxor1d_U8_2m", |bn| {
            bn.iter(|| black_box(numpy_x.call1((&a, &b)).expect("numpy setxor1d")));
        });

        // Latin-1 U16: the wide two-word-key source-tagged run-composition branch.
        let setup16 = "import numpy as np\n\
rng = np.random.default_rng(3)\n\
a16 = rng.integers(97, 123, (1_000_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
brand16 = rng.integers(97, 123, (500_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
b16 = np.concatenate([a16[:500_000], brand16])\n";
        let ns16 = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup16).unwrap().as_c_str(),
            Some(&ns16),
            Some(&ns16),
        )
        .expect("string setxor U16 setup");
        let a16 = ns16.get_item("a16").expect("a16");
        let b16 = ns16.get_item("b16").expect("b16");
        let f = fnp_x.call1((&a16, &b16)).expect("fnp setxor1d U16");
        let n = numpy_x.call1((&a16, &b16)).expect("numpy setxor1d U16");
        let eq: bool = np_array_equal
            .call1((&f, &n))
            .expect("array_equal")
            .extract()
            .expect("bool");
        assert!(eq, "string setxor1d U16 correctness mismatch");
        group.bench_function("fnp_setxor1d_U16_1m", |bn| {
            bn.iter(|| black_box(fnp_x.call1((&a16, &b16)).expect("fnp setxor1d U16")));
        });
        group.bench_function("numpy_setxor1d_U16_1m", |bn| {
            bn.iter(|| black_box(numpy_x.call1((&a16, &b16)).expect("numpy setxor1d U16")));
        });
    });
    group.finish();
}

fn bench_string_bytes_boundary(c: &mut Criterion) {
    // 'S' (bytes) dtype twins of sort/unique/union1d. Bytes are already in byte order == numpy order
    // (no codepoint subtlety, no Latin-1 gate), so the same byte helpers are bit-exact. numpy is
    // equally slow on 'S' (sort ~238ms, unique ~1245ms, union1d ~3162ms @2M).
    let mut group = c.benchmark_group("python_string_bytes_boundary");
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
a = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint8).view('S8').reshape(-1)\n\
b = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint8).view('S8').reshape(-1)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("string bytes setup");
        let a = ns.get_item("a").expect("a");
        let b = ns.get_item("b").expect("b");
        let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
        // Correctness gates for sort / unique / union1d on 'S8'.
        for op in ["sort", "unique"] {
            let f = module.getattr(op).unwrap().call1((&a,)).expect("fnp op");
            let n = numpy.getattr(op).unwrap().call1((&a,)).expect("numpy op");
            let eq: bool = np_array_equal.call1((&f, &n)).unwrap().extract().unwrap();
            assert!(eq, "bytes {op} correctness mismatch");
        }
        {
            let f = module
                .getattr("union1d")
                .unwrap()
                .call1((&a, &b))
                .expect("fnp union1d");
            let n = numpy
                .getattr("union1d")
                .unwrap()
                .call1((&a, &b))
                .expect("numpy union1d");
            let eq: bool = np_array_equal.call1((&f, &n)).unwrap().extract().unwrap();
            assert!(eq, "bytes union1d correctness mismatch");
        }
        let fnp_sort = module.getattr("sort").unwrap();
        let numpy_sort = numpy.getattr("sort").unwrap();
        let fnp_uniq = module.getattr("unique").unwrap();
        let numpy_uniq = numpy.getattr("unique").unwrap();
        let fnp_uni = module.getattr("union1d").unwrap();
        let numpy_uni = numpy.getattr("union1d").unwrap();
        group.bench_function("fnp_sort_S8_2m", |bn| {
            bn.iter(|| black_box(fnp_sort.call1((&a,)).unwrap()))
        });
        group.bench_function("numpy_sort_S8_2m", |bn| {
            bn.iter(|| black_box(numpy_sort.call1((&a,)).unwrap()))
        });
        group.bench_function("fnp_unique_S8_2m", |bn| {
            bn.iter(|| black_box(fnp_uniq.call1((&a,)).unwrap()))
        });
        group.bench_function("numpy_unique_S8_2m", |bn| {
            bn.iter(|| black_box(numpy_uniq.call1((&a,)).unwrap()))
        });
        group.bench_function("fnp_union1d_S8_2m", |bn| {
            bn.iter(|| black_box(fnp_uni.call1((&a, &b)).unwrap()))
        });
        group.bench_function("numpy_union1d_S8_2m", |bn| {
            bn.iter(|| black_box(numpy_uni.call1((&a, &b)).unwrap()))
        });
    });
    group.finish();
}

fn bench_string_bytes_ops2_boundary(c: &mut Criterion) {
    // 'S' (bytes) twins of searchsorted/isin/intersect1d/setdiff1d/setxor1d. Same byte helpers,
    // bit-exact for 'S' (no codepoint layer). numpy is slow on all (searchsorted ~766ms, isin ~440ms,
    // set-ops ~1-4s @2M).
    let mut group = c.benchmark_group("python_string_bytes_ops2_boundary");
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
a = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint8).view('S8').reshape(-1)\n\
h = np.sort(a)\n\
q = rng.integers(97, 123, (2_000_000, 8), dtype=np.uint8).view('S8').reshape(-1)\n\
brand = rng.integers(97, 123, (1_000_000, 8), dtype=np.uint8).view('S8').reshape(-1)\n\
b = np.concatenate([a[:1_000_000], brand])\n\
trand = rng.integers(97, 123, (100_000, 8), dtype=np.uint8).view('S8').reshape(-1)\n\
test = np.concatenate([a[:100_000], trand])\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("setup");
        let a = ns.get_item("a").unwrap();
        let h = ns.get_item("h").unwrap();
        let q = ns.get_item("q").unwrap();
        let b = ns.get_item("b").unwrap();
        let test = ns.get_item("test").unwrap();
        let eqf = numpy.getattr("array_equal").unwrap();
        // Correctness for all 5 ops.
        let ss_f = module
            .getattr("searchsorted")
            .unwrap()
            .call1((&h, &q))
            .unwrap();
        let ss_n = numpy
            .getattr("searchsorted")
            .unwrap()
            .call1((&h, &q))
            .unwrap();
        assert!(
            eqf.call1((&ss_f, &ss_n))
                .unwrap()
                .extract::<bool>()
                .unwrap(),
            "S searchsorted mismatch"
        );
        let is_f = module.getattr("isin").unwrap().call1((&a, &test)).unwrap();
        let is_n = numpy.getattr("isin").unwrap().call1((&a, &test)).unwrap();
        assert!(
            eqf.call1((&is_f, &is_n))
                .unwrap()
                .extract::<bool>()
                .unwrap(),
            "S isin mismatch"
        );
        for op in ["intersect1d", "setdiff1d", "setxor1d"] {
            let f = module.getattr(op).unwrap().call1((&a, &b)).unwrap();
            let n = numpy.getattr(op).unwrap().call1((&a, &b)).unwrap();
            assert!(
                eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
                "S {op} mismatch"
            );
        }
        let ss_ff = module.getattr("searchsorted").unwrap();
        let ss_nn = numpy.getattr("searchsorted").unwrap();
        let is_ff = module.getattr("isin").unwrap();
        let is_nn = numpy.getattr("isin").unwrap();
        let xr_ff = module.getattr("setxor1d").unwrap();
        let xr_nn = numpy.getattr("setxor1d").unwrap();
        group.bench_function("fnp_searchsorted_S8_2m", |bn| {
            bn.iter(|| black_box(ss_ff.call1((&h, &q)).unwrap()))
        });
        group.bench_function("numpy_searchsorted_S8_2m", |bn| {
            bn.iter(|| black_box(ss_nn.call1((&h, &q)).unwrap()))
        });
        group.bench_function("fnp_isin_S8_2m", |bn| {
            bn.iter(|| black_box(is_ff.call1((&a, &test)).unwrap()))
        });
        group.bench_function("numpy_isin_S8_2m", |bn| {
            bn.iter(|| black_box(is_nn.call1((&a, &test)).unwrap()))
        });
        group.bench_function("fnp_setxor1d_S8_2m", |bn| {
            bn.iter(|| black_box(xr_ff.call1((&a, &b)).unwrap()))
        });
        group.bench_function("numpy_setxor1d_S8_2m", |bn| {
            bn.iter(|| black_box(xr_nn.call1((&a, &b)).unwrap()))
        });

        // S16: the wide two-word-key byte pack (S9..16 previously fell to the memcmp/FNV routes).
        // Full byte range incl. embedded nulls (raw padded memcmp == numpy 'S' order).
        let setup16 = "import numpy as np\n\
rng = np.random.default_rng(4)\n\
a16 = rng.integers(0, 256, (1_000_000, 16), dtype=np.uint8).view('S16').reshape(-1)\n\
brand16 = rng.integers(0, 256, (500_000, 16), dtype=np.uint8).view('S16').reshape(-1)\n\
b16 = np.concatenate([a16[:500_000], brand16])\n";
        let ns16 = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup16).unwrap().as_c_str(),
            Some(&ns16),
            Some(&ns16),
        )
        .expect("S16 setup");
        let a16 = ns16.get_item("a16").expect("a16");
        let b16 = ns16.get_item("b16").expect("b16");
        let ix_ff = module.getattr("intersect1d").unwrap();
        let ix_nn = numpy.getattr("intersect1d").unwrap();
        for op in ["intersect1d", "setxor1d"] {
            let f = module.getattr(op).unwrap().call1((&a16, &b16)).unwrap();
            let n = numpy.getattr(op).unwrap().call1((&a16, &b16)).unwrap();
            assert!(
                eqf.call1((&f, &n)).unwrap().extract::<bool>().unwrap(),
                "S16 {op} mismatch"
            );
        }
        group.bench_function("fnp_intersect1d_S16_1m", |bn| {
            bn.iter(|| black_box(ix_ff.call1((&a16, &b16)).unwrap()))
        });
        group.bench_function("numpy_intersect1d_S16_1m", |bn| {
            bn.iter(|| black_box(ix_nn.call1((&a16, &b16)).unwrap()))
        });
        group.bench_function("fnp_setxor1d_S16_1m", |bn| {
            bn.iter(|| black_box(xr_ff.call1((&a16, &b16)).unwrap()))
        });
        group.bench_function("numpy_setxor1d_S16_1m", |bn| {
            bn.iter(|| black_box(xr_nn.call1((&a16, &b16)).unwrap()))
        });
    });
    group.finish();
}

fn main() {
    common::gated_main(&[
        ("bench_string_sort_boundary", bench_string_sort_boundary),
        ("bench_string_unique_boundary", bench_string_unique_boundary),
        (
            "bench_string_unique_full_boundary",
            bench_string_unique_full_boundary,
        ),
        (
            "bench_string_searchsorted_boundary",
            bench_string_searchsorted_boundary,
        ),
        ("bench_string_isin_boundary", bench_string_isin_boundary),
        ("bench_string_union1d_boundary", bench_string_union1d_boundary),
        ("bench_string_setops_boundary", bench_string_setops_boundary),
        ("bench_string_setxor_boundary", bench_string_setxor_boundary),
        ("bench_string_bytes_boundary", bench_string_bytes_boundary),
        (
            "bench_string_bytes_ops2_boundary",
            bench_string_bytes_ops2_boundary,
        ),
    ]);
}
