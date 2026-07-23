//! io/text (loadtxt/genfromtxt) criterion benches split out of the monolithic
//! `criterion_python_surface.rs` into their own per-domain bench binary, so a
//! per-domain run compiles only these groups. See bead deadlock-audit-x7nnf.

#[path = "common/mod.rs"]
mod common;

use common::{ensure_numpy_available, report_ledger_pair};
use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyModule};
use std::cell::{Cell, RefCell};
use std::hint::black_box;
use std::time::{Duration, Instant};

fn bench_loadtxt_text_boundary(c: &mut Criterion) {
    // BASELINE (parser-fork discovery, .364): fnp-python's loadtxt has its
    // OWN inline tokenizer (Vec<Vec<String>> - a String per token, a Vec per
    // row, all materialized before a second parse pass); the fnp-io text
    // wins do not flow here. This group pins the current fnp-vs-numpy
    // surface ratio on a plain f64 comma corpus before any lever lands.
    let mut group = c.benchmark_group("python_loadtxt_text_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    let mut text = String::new();
    for row in 0..8192usize {
        for col in 0..16usize {
            if col > 0 {
                text.push(',');
            }
            text.push_str(&format!("{}.{}", row % 977, col));
        }
        text.push('\n');
    }
    let path = std::env::temp_dir().join("fnp_bench_loadtxt_boundary.csv");
    std::fs::write(&path, &text).expect("write bench corpus");
    let path_str = path.to_str().expect("utf8 temp path").to_string();

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_loadtxt = module.getattr("loadtxt").expect("fnp loadtxt");
        let numpy_loadtxt = numpy.getattr("loadtxt").expect("numpy loadtxt");
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("delimiter", ",").expect("kwargs");

        // Pre-flight parity: same shape and bytes from both.
        let fnp_out = fnp_loadtxt
            .call((path_str.as_str(),), Some(&kwargs))
            .expect("fnp loadtxt");
        let numpy_out = numpy_loadtxt
            .call((path_str.as_str(),), Some(&kwargs))
            .expect("numpy loadtxt");
        let equal: bool = numpy
            .call_method1("array_equal", (&fnp_out, &numpy_out))
            .expect("array_equal")
            .extract()
            .expect("bool");
        assert!(
            equal,
            "fnp/numpy loadtxt outputs differ on the bench corpus"
        );

        group.bench_function("fnp_loadtxt_f64_8192x16", |b| {
            b.iter(|| {
                black_box(
                    fnp_loadtxt
                        .call((path_str.as_str(),), Some(&kwargs))
                        .expect("fnp loadtxt"),
                )
            });
        });
        group.bench_function("numpy_loadtxt_f64_8192x16", |b| {
            b.iter(|| {
                black_box(
                    numpy_loadtxt
                        .call((path_str.as_str(),), Some(&kwargs))
                        .expect("numpy loadtxt"),
                )
            });
        });
    });

    group.finish();
}

fn bench_loadtxt_integer_text_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_loadtxt_integer_text_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_millis(750));
    group.warm_up_time(Duration::from_millis(250));

    let mut text = String::new();
    for row in 0..8192usize {
        for col in 0..16usize {
            if col > 0 {
                text.push(',');
            }
            text.push_str(&((row % 977) * 16 + col).to_string());
        }
        text.push('\n');
    }
    let path = std::env::temp_dir().join("fnp_bench_loadtxt_integer_boundary.csv");
    std::fs::write(&path, &text).expect("write integer bench corpus");
    let path_str = path.to_str().expect("utf8 temp path").to_string();

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_integer_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_loadtxt = module.getattr("loadtxt").expect("fnp loadtxt");
        let numpy_loadtxt = numpy.getattr("loadtxt").expect("numpy loadtxt");
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("delimiter", ",").expect("delimiter kwarg");
        kwargs
            .set_item("dtype", numpy.getattr("int64").expect("numpy int64"))
            .expect("dtype kwarg");
        let former_kwargs = pyo3::types::PyDict::new(py);
        former_kwargs
            .set_item("delimiter", ",")
            .expect("former delimiter kwarg");
        former_kwargs
            .set_item("dtype", numpy.getattr("int64").expect("numpy int64"))
            .expect("former dtype kwarg");
        // The direct path deliberately excludes unpack. `unpack=true` therefore
        // exercises the exact retained former tokenizer/parser; transposing its
        // output back restores the candidate's observable shape. The two NumPy
        // view constructions are included in the former timing and are reported
        // as a small control bias rather than hidden.
        former_kwargs
            .set_item("unpack", true)
            .expect("former unpack kwarg");

        let fnp_out = fnp_loadtxt
            .call((path_str.as_str(),), Some(&kwargs))
            .expect("fnp integer loadtxt");
        let numpy_out = numpy_loadtxt
            .call((path_str.as_str(),), Some(&kwargs))
            .expect("numpy integer loadtxt");
        let former_out = fnp_loadtxt
            .call((path_str.as_str(),), Some(&former_kwargs))
            .expect("former integer loadtxt")
            .getattr("T")
            .expect("restore former shape");
        let equal: bool = numpy
            .call_method1("array_equal", (&fnp_out, &numpy_out))
            .expect("array_equal")
            .extract()
            .expect("bool");
        assert!(equal, "fnp/numpy integer loadtxt outputs differ");
        let former_equal: bool = numpy
            .call_method1("array_equal", (&fnp_out, &former_out))
            .expect("former array_equal")
            .extract()
            .expect("bool");
        assert!(former_equal, "direct/former integer loadtxt outputs differ");

        let candidate_samples = RefCell::new(Vec::new());
        let former_samples = RefCell::new(Vec::new());
        let effect_order = Cell::new(0u64);
        group.bench_function("paired_former_candidate", |bench| {
            bench.iter_custom(|iterations| {
                let mut candidate_total = Duration::ZERO;
                let mut former_total = Duration::ZERO;
                let time_candidate = || {
                    let started = Instant::now();
                    black_box(
                        fnp_loadtxt
                            .call((path_str.as_str(),), Some(&kwargs))
                            .expect("candidate integer loadtxt"),
                    );
                    started.elapsed()
                };
                let time_former = || {
                    let started = Instant::now();
                    let unpacked = fnp_loadtxt
                        .call((path_str.as_str(),), Some(&former_kwargs))
                        .expect("former integer loadtxt");
                    black_box(unpacked.getattr("T").expect("restore former shape"));
                    started.elapsed()
                };
                for _ in 0..iterations {
                    if effect_order.get() & 1 == 0 {
                        former_total += time_former();
                        candidate_total += time_candidate();
                        candidate_total += time_candidate();
                        former_total += time_former();
                    } else {
                        candidate_total += time_candidate();
                        former_total += time_former();
                        former_total += time_former();
                        candidate_total += time_candidate();
                    }
                    effect_order.set(effect_order.get().wrapping_add(1));
                }
                candidate_samples
                    .borrow_mut()
                    .push(candidate_total.as_secs_f64() * 0.5e9 / iterations as f64);
                former_samples
                    .borrow_mut()
                    .push(former_total.as_secs_f64() * 0.5e9 / iterations as f64);
                candidate_total + former_total
            });
        });
        report_ledger_pair(
            "loadtxt_i64_direct_parse_effect",
            &candidate_samples,
            &former_samples,
        );

        let null_lhs_samples = RefCell::new(Vec::new());
        let null_rhs_samples = RefCell::new(Vec::new());
        let null_order = Cell::new(0u64);
        group.bench_function("null_candidate_aa", |bench| {
            bench.iter_custom(|iterations| {
                let mut lhs_total = Duration::ZERO;
                let mut rhs_total = Duration::ZERO;
                let time_candidate = || {
                    let started = Instant::now();
                    black_box(
                        fnp_loadtxt
                            .call((path_str.as_str(),), Some(&kwargs))
                            .expect("candidate integer loadtxt"),
                    );
                    started.elapsed()
                };
                for _ in 0..iterations {
                    if null_order.get() & 1 == 0 {
                        lhs_total += time_candidate();
                        rhs_total += time_candidate();
                        rhs_total += time_candidate();
                        lhs_total += time_candidate();
                    } else {
                        rhs_total += time_candidate();
                        lhs_total += time_candidate();
                        lhs_total += time_candidate();
                        rhs_total += time_candidate();
                    }
                    null_order.set(null_order.get().wrapping_add(1));
                }
                null_lhs_samples
                    .borrow_mut()
                    .push(lhs_total.as_secs_f64() * 0.5e9 / iterations as f64);
                null_rhs_samples
                    .borrow_mut()
                    .push(rhs_total.as_secs_f64() * 0.5e9 / iterations as f64);
                lhs_total + rhs_total
            });
        });
        report_ledger_pair(
            "loadtxt_i64_direct_parse_null",
            &null_lhs_samples,
            &null_rhs_samples,
        );

        let plain_samples = RefCell::new(Vec::new());
        let viewed_samples = RefCell::new(Vec::new());
        let view_order = Cell::new(0u64);
        group.bench_function("former_view_adapter_bias", |bench| {
            bench.iter_custom(|iterations| {
                let mut plain_total = Duration::ZERO;
                let mut viewed_total = Duration::ZERO;
                let time_plain = || {
                    let started = Instant::now();
                    black_box(
                        fnp_loadtxt
                            .call((path_str.as_str(),), Some(&kwargs))
                            .expect("plain integer loadtxt"),
                    );
                    started.elapsed()
                };
                let time_viewed = || {
                    let started = Instant::now();
                    let direct = fnp_loadtxt
                        .call((path_str.as_str(),), Some(&kwargs))
                        .expect("viewed integer loadtxt");
                    black_box(
                        direct
                            .getattr("T")
                            .expect("first adapter view")
                            .getattr("T")
                            .expect("second adapter view"),
                    );
                    started.elapsed()
                };
                for _ in 0..iterations {
                    if view_order.get() & 1 == 0 {
                        viewed_total += time_viewed();
                        plain_total += time_plain();
                        plain_total += time_plain();
                        viewed_total += time_viewed();
                    } else {
                        plain_total += time_plain();
                        viewed_total += time_viewed();
                        viewed_total += time_viewed();
                        plain_total += time_plain();
                    }
                    view_order.set(view_order.get().wrapping_add(1));
                }
                plain_samples
                    .borrow_mut()
                    .push(plain_total.as_secs_f64() * 0.5e9 / iterations as f64);
                viewed_samples
                    .borrow_mut()
                    .push(viewed_total.as_secs_f64() * 0.5e9 / iterations as f64);
                plain_total + viewed_total
            });
        });
        report_ledger_pair(
            "loadtxt_i64_view_adapter_bias",
            &plain_samples,
            &viewed_samples,
        );

        group.bench_function("fnp_loadtxt_i64_8192x16", |bench| {
            bench.iter(|| {
                black_box(
                    fnp_loadtxt
                        .call((path_str.as_str(),), Some(&kwargs))
                        .expect("fnp integer loadtxt"),
                )
            });
        });
        group.bench_function("numpy_loadtxt_i64_8192x16", |bench| {
            bench.iter(|| {
                black_box(
                    numpy_loadtxt
                        .call((path_str.as_str(),), Some(&kwargs))
                        .expect("numpy integer loadtxt"),
                )
            });
        });
    });

    group.finish();
}

fn bench_loadtxt_bool_text_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_loadtxt_bool_text_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_millis(750));
    group.warm_up_time(Duration::from_millis(250));

    let mut text = String::new();
    for row in 0..8192usize {
        for col in 0..16usize {
            if col > 0 {
                text.push(',');
            }
            let value = ((row + col) % 5) as i64 - 2;
            text.push_str(&value.to_string());
        }
        text.push('\n');
    }
    let path = std::env::temp_dir().join("fnp_bench_loadtxt_bool_boundary.csv");
    std::fs::write(&path, &text).expect("write bool bench corpus");
    let path_str = path.to_str().expect("utf8 temp path").to_string();

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bool_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_loadtxt = module.getattr("loadtxt").expect("fnp loadtxt");
        let numpy_loadtxt = numpy.getattr("loadtxt").expect("numpy loadtxt");
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("delimiter", ",").expect("delimiter kwarg");
        kwargs
            .set_item("dtype", numpy.getattr("bool_").expect("numpy bool dtype"))
            .expect("dtype kwarg");
        let former_kwargs = pyo3::types::PyDict::new(py);
        former_kwargs
            .set_item("delimiter", ",")
            .expect("former delimiter kwarg");
        former_kwargs
            .set_item("dtype", numpy.getattr("bool_").expect("numpy bool dtype"))
            .expect("former dtype kwarg");
        // The direct bool path deliberately excludes unpack, exactly like the
        // integer arm. `unpack=true` therefore exercises the retained former
        // tokenizer/parser; transposing its output restores the candidate's
        // observable shape, and that O(1) view bias is measured in its own
        // arm below rather than hidden.
        former_kwargs
            .set_item("unpack", true)
            .expect("former unpack kwarg");

        let fnp_out = fnp_loadtxt
            .call((path_str.as_str(),), Some(&kwargs))
            .expect("fnp bool loadtxt");
        let numpy_out = numpy_loadtxt
            .call((path_str.as_str(),), Some(&kwargs))
            .expect("numpy bool loadtxt");
        let former_out = fnp_loadtxt
            .call((path_str.as_str(),), Some(&former_kwargs))
            .expect("former bool loadtxt")
            .getattr("T")
            .expect("restore former shape");
        let equal: bool = numpy
            .call_method1("array_equal", (&fnp_out, &numpy_out))
            .expect("array_equal")
            .extract()
            .expect("bool");
        assert!(equal, "fnp/numpy bool loadtxt outputs differ");
        let former_equal: bool = numpy
            .call_method1("array_equal", (&fnp_out, &former_out))
            .expect("former array_equal")
            .extract()
            .expect("bool");
        assert!(former_equal, "direct/former bool loadtxt outputs differ");

        let candidate_samples = RefCell::new(Vec::new());
        let former_samples = RefCell::new(Vec::new());
        let effect_order = Cell::new(0u64);
        group.bench_function("paired_former_candidate", |bench| {
            bench.iter_custom(|iterations| {
                let mut candidate_total = Duration::ZERO;
                let mut former_total = Duration::ZERO;
                let time_candidate = || {
                    let started = Instant::now();
                    black_box(
                        fnp_loadtxt
                            .call((path_str.as_str(),), Some(&kwargs))
                            .expect("candidate bool loadtxt"),
                    );
                    started.elapsed()
                };
                let time_former = || {
                    let started = Instant::now();
                    let unpacked = fnp_loadtxt
                        .call((path_str.as_str(),), Some(&former_kwargs))
                        .expect("former bool loadtxt");
                    black_box(unpacked.getattr("T").expect("restore former shape"));
                    started.elapsed()
                };
                for _ in 0..iterations {
                    if effect_order.get() & 1 == 0 {
                        former_total += time_former();
                        candidate_total += time_candidate();
                        candidate_total += time_candidate();
                        former_total += time_former();
                    } else {
                        candidate_total += time_candidate();
                        former_total += time_former();
                        former_total += time_former();
                        candidate_total += time_candidate();
                    }
                    effect_order.set(effect_order.get().wrapping_add(1));
                }
                candidate_samples
                    .borrow_mut()
                    .push(candidate_total.as_secs_f64() * 0.5e9 / iterations as f64);
                former_samples
                    .borrow_mut()
                    .push(former_total.as_secs_f64() * 0.5e9 / iterations as f64);
                candidate_total + former_total
            });
        });
        report_ledger_pair(
            "loadtxt_bool_direct_parse_effect",
            &candidate_samples,
            &former_samples,
        );

        let null_lhs_samples = RefCell::new(Vec::new());
        let null_rhs_samples = RefCell::new(Vec::new());
        let null_order = Cell::new(0u64);
        group.bench_function("null_candidate_aa", |bench| {
            bench.iter_custom(|iterations| {
                let mut lhs_total = Duration::ZERO;
                let mut rhs_total = Duration::ZERO;
                let time_candidate = || {
                    let started = Instant::now();
                    black_box(
                        fnp_loadtxt
                            .call((path_str.as_str(),), Some(&kwargs))
                            .expect("candidate bool loadtxt"),
                    );
                    started.elapsed()
                };
                for _ in 0..iterations {
                    if null_order.get() & 1 == 0 {
                        lhs_total += time_candidate();
                        rhs_total += time_candidate();
                        rhs_total += time_candidate();
                        lhs_total += time_candidate();
                    } else {
                        rhs_total += time_candidate();
                        lhs_total += time_candidate();
                        lhs_total += time_candidate();
                        rhs_total += time_candidate();
                    }
                    null_order.set(null_order.get().wrapping_add(1));
                }
                null_lhs_samples
                    .borrow_mut()
                    .push(lhs_total.as_secs_f64() * 0.5e9 / iterations as f64);
                null_rhs_samples
                    .borrow_mut()
                    .push(rhs_total.as_secs_f64() * 0.5e9 / iterations as f64);
                lhs_total + rhs_total
            });
        });
        report_ledger_pair(
            "loadtxt_bool_direct_parse_null",
            &null_lhs_samples,
            &null_rhs_samples,
        );

        let plain_samples = RefCell::new(Vec::new());
        let viewed_samples = RefCell::new(Vec::new());
        let view_order = Cell::new(0u64);
        group.bench_function("former_view_adapter_bias", |bench| {
            bench.iter_custom(|iterations| {
                let mut plain_total = Duration::ZERO;
                let mut viewed_total = Duration::ZERO;
                let time_plain = || {
                    let started = Instant::now();
                    black_box(
                        fnp_loadtxt
                            .call((path_str.as_str(),), Some(&kwargs))
                            .expect("plain bool loadtxt"),
                    );
                    started.elapsed()
                };
                let time_viewed = || {
                    let started = Instant::now();
                    let direct = fnp_loadtxt
                        .call((path_str.as_str(),), Some(&kwargs))
                        .expect("viewed bool loadtxt");
                    black_box(
                        direct
                            .getattr("T")
                            .expect("first adapter view")
                            .getattr("T")
                            .expect("second adapter view"),
                    );
                    started.elapsed()
                };
                for _ in 0..iterations {
                    if view_order.get() & 1 == 0 {
                        viewed_total += time_viewed();
                        plain_total += time_plain();
                        plain_total += time_plain();
                        viewed_total += time_viewed();
                    } else {
                        plain_total += time_plain();
                        viewed_total += time_viewed();
                        viewed_total += time_viewed();
                        plain_total += time_plain();
                    }
                    view_order.set(view_order.get().wrapping_add(1));
                }
                plain_samples
                    .borrow_mut()
                    .push(plain_total.as_secs_f64() * 0.5e9 / iterations as f64);
                viewed_samples
                    .borrow_mut()
                    .push(viewed_total.as_secs_f64() * 0.5e9 / iterations as f64);
                plain_total + viewed_total
            });
        });
        report_ledger_pair(
            "loadtxt_bool_view_adapter_bias",
            &plain_samples,
            &viewed_samples,
        );

        group.bench_function("fnp_loadtxt_bool_8192x16", |bench| {
            bench.iter(|| {
                black_box(
                    fnp_loadtxt
                        .call((path_str.as_str(),), Some(&kwargs))
                        .expect("fnp bool loadtxt"),
                )
            });
        });
        group.bench_function("numpy_loadtxt_bool_8192x16", |bench| {
            bench.iter(|| {
                black_box(
                    numpy_loadtxt
                        .call((path_str.as_str(),), Some(&kwargs))
                        .expect("numpy bool loadtxt"),
                )
            });
        });
    });

    group.finish();
}

fn bench_genfromtxt_text_boundary(c: &mut Criterion) {
    // SIBLING of the loadtxt fork (.364/.367): fnp-python's genfromtxt native
    // path carries the SAME inline Vec<Vec<String>> tokenizer. Its native gate
    // requires an explicit numeric dtype (default-None defers to numpy), so
    // this arm calls `dtype=float64`. numpy's genfromtxt is its SLOW
    // rich/Python-level parser, so this ratio decides whether a real vs-numpy
    // gap exists here at all before any delegation lever.
    let mut group = c.benchmark_group("python_genfromtxt_text_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    let mut text = String::new();
    for row in 0..8192usize {
        for col in 0..16usize {
            if col > 0 {
                text.push(',');
            }
            text.push_str(&format!("{}.{}", row % 977, col));
        }
        text.push('\n');
    }
    let path = std::env::temp_dir().join("fnp_bench_genfromtxt_boundary.csv");
    std::fs::write(&path, &text).expect("write bench corpus");
    let path_str = path.to_str().expect("utf8 temp path").to_string();

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_genfromtxt = module.getattr("genfromtxt").expect("fnp genfromtxt");
        let numpy_genfromtxt = numpy.getattr("genfromtxt").expect("numpy genfromtxt");
        let float64 = numpy.getattr("float64").expect("numpy.float64");
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("delimiter", ",").expect("kwargs");
        kwargs.set_item("dtype", &float64).expect("kwargs");

        // Pre-flight parity: same shape and bytes from both.
        let fnp_out = fnp_genfromtxt
            .call((path_str.as_str(),), Some(&kwargs))
            .expect("fnp genfromtxt");
        let numpy_out = numpy_genfromtxt
            .call((path_str.as_str(),), Some(&kwargs))
            .expect("numpy genfromtxt");
        let equal: bool = numpy
            .call_method1("array_equal", (&fnp_out, &numpy_out))
            .expect("array_equal")
            .extract()
            .expect("bool");
        assert!(
            equal,
            "fnp/numpy genfromtxt outputs differ on the bench corpus"
        );

        group.bench_function("fnp_genfromtxt_f64_8192x16", |b| {
            b.iter(|| {
                black_box(
                    fnp_genfromtxt
                        .call((path_str.as_str(),), Some(&kwargs))
                        .expect("fnp genfromtxt"),
                )
            });
        });
        group.bench_function("numpy_genfromtxt_f64_8192x16", |b| {
            b.iter(|| {
                black_box(
                    numpy_genfromtxt
                        .call((path_str.as_str(),), Some(&kwargs))
                        .expect("numpy genfromtxt"),
                )
            });
        });
    });

    group.finish();
}

fn main() {
    common::gated_main(&[
        ("bench_loadtxt_text_boundary", bench_loadtxt_text_boundary),
        (
            "bench_loadtxt_integer_text_boundary",
            bench_loadtxt_integer_text_boundary,
        ),
        (
            "bench_loadtxt_bool_text_boundary",
            bench_loadtxt_bool_text_boundary,
        ),
        ("bench_genfromtxt_text_boundary", bench_genfromtxt_text_boundary),
    ]);
}
