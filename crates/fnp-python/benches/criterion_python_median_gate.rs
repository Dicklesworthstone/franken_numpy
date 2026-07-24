//! median-gate / substrate-v2 variance-protocol criterion benches — the paired
//! `iter_custom` A/B harness with median-gate reporting (interleaved AB/BA,
//! null control, tail stats) plus its local timing helpers. Split out of the
//! monolithic `criterion_python_surface.rs` into their own per-domain bench
//! binary; this is the single largest domain (~3900 lines / 21 fns), so pulling
//! it out is the biggest compile-volume reduction of the split. See bead
//! deadlock-audit-x7nnf.

#[path = "common/mod.rs"]
mod common;

use common::*;
use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::types::{PyAnyMethods, PyDict, PyModule};
use pyo3::{Bound, PyAny, Python};
use rayon::prelude::*;
use std::cell::{Cell, RefCell};
use std::hint::black_box;
use std::time::{Duration, Instant};

fn bench_median_gate_python_binary<'py>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    bench_name: &'static str,
    row: &'static str,
    base: &Bound<'py, PyAny>,
    candidate: &Bound<'py, PyAny>,
    lhs: &Bound<'py, PyAny>,
    rhs: &Bound<'py, PyAny>,
) {
    let null_base_ns = RefCell::new(Vec::new());
    let null_peer_ns = RefCell::new(Vec::new());
    let null_ratios = RefCell::new(Vec::new());
    let base_ns = RefCell::new(Vec::new());
    let candidate_ns = RefCell::new(Vec::new());
    let effect_ratios = RefCell::new(Vec::new());
    group.bench_function(bench_name, |bench| {
        bench.iter_custom(|iterations| {
            let mut combined = Duration::ZERO;
            for _ in 0..iterations {
                // The exact-function A/A null is always measured first. The two observations
                // are ABBA then BAAB, balancing call position before the effect is observed.
                for observation in 0..MEDIAN_GATE_OBSERVATIONS_PER_BATCH {
                    let outer_base = observation & 1 == 0;
                    let (base_total, peer_total) = if outer_base {
                        let a1 = time_python_binary_call(base, lhs, rhs);
                        let b1 = time_python_binary_call(base, lhs, rhs);
                        let b2 = time_python_binary_call(base, lhs, rhs);
                        let a2 = time_python_binary_call(base, lhs, rhs);
                        (a1 + a2, b1 + b2)
                    } else {
                        let b1 = time_python_binary_call(base, lhs, rhs);
                        let a1 = time_python_binary_call(base, lhs, rhs);
                        let a2 = time_python_binary_call(base, lhs, rhs);
                        let b2 = time_python_binary_call(base, lhs, rhs);
                        (a1 + a2, b1 + b2)
                    };
                    let base_average = base_total.as_secs_f64() * 0.5e9;
                    let peer_average = peer_total.as_secs_f64() * 0.5e9;
                    null_base_ns.borrow_mut().push(base_average);
                    null_peer_ns.borrow_mut().push(peer_average);
                    null_ratios.borrow_mut().push(base_average / peer_average);
                    combined += base_total + peer_total;
                }
                for observation in 0..MEDIAN_GATE_OBSERVATIONS_PER_BATCH {
                    let outer_base = observation & 1 == 0;
                    let (base_total, candidate_total) = if outer_base {
                        let a1 = time_python_binary_call(base, lhs, rhs);
                        let b1 = time_python_binary_call(candidate, lhs, rhs);
                        let b2 = time_python_binary_call(candidate, lhs, rhs);
                        let a2 = time_python_binary_call(base, lhs, rhs);
                        (a1 + a2, b1 + b2)
                    } else {
                        let b1 = time_python_binary_call(candidate, lhs, rhs);
                        let a1 = time_python_binary_call(base, lhs, rhs);
                        let a2 = time_python_binary_call(base, lhs, rhs);
                        let b2 = time_python_binary_call(candidate, lhs, rhs);
                        (a1 + a2, b1 + b2)
                    };
                    let base_average = base_total.as_secs_f64() * 0.5e9;
                    let candidate_average = candidate_total.as_secs_f64() * 0.5e9;
                    base_ns.borrow_mut().push(base_average);
                    candidate_ns.borrow_mut().push(candidate_average);
                    effect_ratios
                        .borrow_mut()
                        .push(base_average / candidate_average);
                    combined += base_total + candidate_total;
                }
            }
            combined
        });
    });
    report_median_gate_pair(
        row,
        &null_base_ns,
        &null_peer_ns,
        &null_ratios,
        &base_ns,
        &candidate_ns,
        &effect_ratios,
    );
}

fn bench_median_gate_python_unary<'py>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    bench_name: &'static str,
    row: &'static str,
    base: &Bound<'py, PyAny>,
    candidate: &Bound<'py, PyAny>,
    input: &Bound<'py, PyAny>,
) {
    let null_base_ns = RefCell::new(Vec::new());
    let null_peer_ns = RefCell::new(Vec::new());
    let null_ratios = RefCell::new(Vec::new());
    let base_ns = RefCell::new(Vec::new());
    let candidate_ns = RefCell::new(Vec::new());
    let effect_ratios = RefCell::new(Vec::new());
    group.bench_function(bench_name, |bench| {
        bench.iter_custom(|iterations| {
            let mut combined = Duration::ZERO;
            for _ in 0..iterations {
                for observation in 0..MEDIAN_GATE_OBSERVATIONS_PER_BATCH {
                    let outer_base = observation & 1 == 0;
                    let (base_total, peer_total) = if outer_base {
                        let a1 = time_python_unary_call(base, input);
                        let b1 = time_python_unary_call(base, input);
                        let b2 = time_python_unary_call(base, input);
                        let a2 = time_python_unary_call(base, input);
                        (a1 + a2, b1 + b2)
                    } else {
                        let b1 = time_python_unary_call(base, input);
                        let a1 = time_python_unary_call(base, input);
                        let a2 = time_python_unary_call(base, input);
                        let b2 = time_python_unary_call(base, input);
                        (a1 + a2, b1 + b2)
                    };
                    let base_average = base_total.as_secs_f64() * 0.5e9;
                    let peer_average = peer_total.as_secs_f64() * 0.5e9;
                    null_base_ns.borrow_mut().push(base_average);
                    null_peer_ns.borrow_mut().push(peer_average);
                    null_ratios.borrow_mut().push(base_average / peer_average);
                    combined += base_total + peer_total;
                }
                for observation in 0..MEDIAN_GATE_OBSERVATIONS_PER_BATCH {
                    let outer_base = observation & 1 == 0;
                    let (base_total, candidate_total) = if outer_base {
                        let a1 = time_python_unary_call(base, input);
                        let b1 = time_python_unary_call(candidate, input);
                        let b2 = time_python_unary_call(candidate, input);
                        let a2 = time_python_unary_call(base, input);
                        (a1 + a2, b1 + b2)
                    } else {
                        let b1 = time_python_unary_call(candidate, input);
                        let a1 = time_python_unary_call(base, input);
                        let a2 = time_python_unary_call(base, input);
                        let b2 = time_python_unary_call(candidate, input);
                        (a1 + a2, b1 + b2)
                    };
                    let base_average = base_total.as_secs_f64() * 0.5e9;
                    let candidate_average = candidate_total.as_secs_f64() * 0.5e9;
                    base_ns.borrow_mut().push(base_average);
                    candidate_ns.borrow_mut().push(candidate_average);
                    effect_ratios
                        .borrow_mut()
                        .push(base_average / candidate_average);
                    combined += base_total + candidate_total;
                }
            }
            combined
        });
    });
    report_median_gate_pair(
        row,
        &null_base_ns,
        &null_peer_ns,
        &null_ratios,
        &base_ns,
        &candidate_ns,
        &effect_ratios,
    );
}

fn bench_substrate_v2_python_binary_pair<'py>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    bench_name: &'static str,
    row: &'static str,
    candidate: &Bound<'py, PyAny>,
    orig: &Bound<'py, PyAny>,
    lhs: &Bound<'py, PyAny>,
    rhs: &Bound<'py, PyAny>,
) {
    let candidate_samples = RefCell::new(Vec::new());
    let orig_samples = RefCell::new(Vec::new());
    let order = Cell::new(0_u64);
    group.bench_function(bench_name, |bench| {
        bench.iter_custom(|iterations| {
            // Slow Python surface rows otherwise collapse to one A/B pair per
            // Criterion sample.  Keep each sample order-balanced and average
            // enough interleaved pairs to make worker jitter visible instead
            // of letting a single interruption decide the row.
            let measured_iterations = iterations.max(4);
            let measured_iterations = measured_iterations + (measured_iterations & 1);
            let mut candidate_total = Duration::ZERO;
            let mut orig_total = Duration::ZERO;
            for _ in 0..measured_iterations {
                let orig_first = order.get() & 1 == 1;
                order.set(order.get().wrapping_add(1));
                let time_call = |function: &Bound<'py, PyAny>| {
                    let start = Instant::now();
                    let lhs = black_box(lhs);
                    let rhs = black_box(rhs);
                    let result = function
                        .call1((lhs, rhs))
                        .expect("paired binary Python call");
                    black_box(result);
                    start.elapsed()
                };
                if orig_first {
                    orig_total += time_call(orig);
                    candidate_total += time_call(candidate);
                } else {
                    candidate_total += time_call(candidate);
                    orig_total += time_call(orig);
                }
            }
            candidate_samples
                .borrow_mut()
                .push(candidate_total.as_secs_f64() * 1e9 / measured_iterations as f64);
            orig_samples
                .borrow_mut()
                .push(orig_total.as_secs_f64() * 1e9 / measured_iterations as f64);
            (candidate_total + orig_total).mul_f64(iterations as f64 / measured_iterations as f64)
        });
    });
    report_substrate_v2_pair(row, &candidate_samples, &orig_samples);
}

fn bench_substrate_v2_python_unary_pair<'py>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    bench_name: &'static str,
    row: &'static str,
    candidate: &Bound<'py, PyAny>,
    orig: &Bound<'py, PyAny>,
    input: &Bound<'py, PyAny>,
) {
    let candidate_samples = RefCell::new(Vec::new());
    let orig_samples = RefCell::new(Vec::new());
    let order = Cell::new(0_u64);
    group.bench_function(bench_name, |bench| {
        bench.iter_custom(|iterations| {
            let measured_iterations = iterations.max(4);
            let measured_iterations = measured_iterations + (measured_iterations & 1);
            let mut candidate_total = Duration::ZERO;
            let mut orig_total = Duration::ZERO;
            for _ in 0..measured_iterations {
                let orig_first = order.get() & 1 == 1;
                order.set(order.get().wrapping_add(1));
                let time_call = |function: &Bound<'py, PyAny>| {
                    let start = Instant::now();
                    let input = black_box(input);
                    let result = function.call1((input,)).expect("paired unary Python call");
                    black_box(result);
                    start.elapsed()
                };
                if orig_first {
                    orig_total += time_call(orig);
                    candidate_total += time_call(candidate);
                } else {
                    candidate_total += time_call(candidate);
                    orig_total += time_call(orig);
                }
            }
            candidate_samples
                .borrow_mut()
                .push(candidate_total.as_secs_f64() * 1e9 / measured_iterations as f64);
            orig_samples
                .borrow_mut()
                .push(orig_total.as_secs_f64() * 1e9 / measured_iterations as f64);
            (candidate_total + orig_total).mul_f64(iterations as f64 / measured_iterations as f64)
        });
    });
    report_substrate_v2_pair(row, &candidate_samples, &orig_samples);
}

fn bench_completion_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_completion_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        println!(
            "ISA_PROVENANCE target_arch={} avx2={} sse2={}",
            std::env::consts::ARCH,
            cfg!(target_feature = "avx2"),
            cfg!(target_feature = "sse2"),
        );
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_completion_median_gate")
            .expect("completion bench module");
        fnp_python(&module).expect("initialize fnp_python completion bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy version")
            .extract::<String>()
            .expect("numpy version string");
        let numpy_simd = numpy
            .getattr("__config__")
            .expect("numpy config")
            .getattr("CONFIG")
            .expect("numpy CONFIG")
            .get_item("SIMD Extensions")
            .expect("numpy SIMD Extensions")
            .str()
            .expect("numpy SIMD str")
            .extract::<String>()
            .expect("numpy SIMD string value");
        let numpy_cpu_features = numpy
            .getattr("_core")
            .expect("numpy core")
            .getattr("_multiarray_umath")
            .expect("numpy multiarray umath")
            .getattr("__cpu_features__")
            .expect("numpy runtime CPU features")
            .str()
            .expect("numpy runtime CPU feature str")
            .extract::<String>()
            .expect("numpy runtime CPU feature string value");
        println!(
            "NUMPY_PROVENANCE version={numpy_version} build_simd={numpy_simd} \
             runtime_cpu_features={numpy_cpu_features}"
        );
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 powers = np.power(np.uint64(26), np.arange(5, dtype=np.uint64))\n\
                 u_a_ids = np.arange(0, 1_000_000, dtype=np.uint64)\n\
                 u_a_words = np.zeros((1_000_000, 16), dtype=np.uint32)\n\
                 u_a_words[:, :5] = (97 + (u_a_ids[:, None] // powers) % 26).astype(np.uint32)\n\
                 u_a = u_a_words.reshape(-1).view('U16')\n\
                 u_fresh_ids = np.arange(1_000_000, 1_500_000, dtype=np.uint64)\n\
                 u_fresh_words = np.zeros((500_000, 16), dtype=np.uint32)\n\
                 u_fresh_words[:, :5] = (97 + (u_fresh_ids[:, None] // powers) % 26).astype(np.uint32)\n\
                 u_fresh = u_fresh_words.reshape(-1).view('U16')\n\
                 u_b = np.concatenate([u_a[:500_000], u_fresh])\n\
                 u_union_ids = np.arange(2_000_000, 3_000_000, dtype=np.uint64)\n\
                 u_union_words = np.zeros((1_000_000, 16), dtype=np.uint32)\n\
                 u_union_words[:, :5] = (97 + (u_union_ids[:, None] // powers) % 26).astype(np.uint32)\n\
                 u_union_b = u_union_words.reshape(-1).view('U16')\n",
            )
            .expect("completion setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("completion setup");
        let u_a = namespace.get_item("u_a").expect("u_a present");
        let u_b = namespace.get_item("u_b").expect("u_b present");
        let u_union_b = namespace.get_item("u_union_b").expect("u_union_b present");
        let array_equal = numpy.getattr("array_equal").expect("numpy.array_equal");

        let fnp_unique = module.getattr("unique").expect("fnp unique");
        let np_unique = numpy.getattr("unique").expect("numpy unique");
        let fnp_union = module.getattr("union1d").expect("fnp union1d");
        let np_union = numpy.getattr("union1d").expect("numpy union1d");
        let fnp_setxor = module.getattr("setxor1d").expect("fnp setxor1d");
        let np_setxor = numpy.getattr("setxor1d").expect("numpy setxor1d");

        for (label, candidate, base) in [
            (
                "U16 unique",
                fnp_unique.call1((&u_a,)).expect("fnp unique parity"),
                np_unique.call1((&u_a,)).expect("numpy unique parity"),
            ),
            (
                "U16 disjoint union",
                fnp_union
                    .call1((&u_a, &u_union_b))
                    .expect("fnp union parity"),
                np_union
                    .call1((&u_a, &u_union_b))
                    .expect("numpy union parity"),
            ),
            (
                "U16 50% overlap setxor",
                fnp_setxor.call1((&u_a, &u_b)).expect("fnp setxor parity"),
                np_setxor.call1((&u_a, &u_b)).expect("numpy setxor parity"),
            ),
        ] {
            let candidate_dtype = candidate.getattr("dtype").expect("candidate dtype");
            let base_dtype = base.getattr("dtype").expect("base dtype");
            assert_eq!(
                candidate_dtype
                    .getattr("str")
                    .expect("candidate dtype str")
                    .extract::<String>()
                    .expect("candidate dtype str value"),
                base_dtype
                    .getattr("str")
                    .expect("base dtype str")
                    .extract::<String>()
                    .expect("base dtype str value"),
                "{label} dtype string parity",
            );
            assert!(
                candidate_dtype
                    .getattr("metadata")
                    .expect("candidate dtype metadata")
                    .eq(base_dtype.getattr("metadata").expect("base dtype metadata"))
                    .expect("dtype metadata equality"),
                "{label} dtype metadata parity",
            );
            assert!(
                array_equal
                    .call1((&candidate, &base))
                    .expect("completion array_equal")
                    .extract::<bool>()
                    .expect("completion array_equal bool"),
                "{label} value parity",
            );
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "{label} byte parity",
            );
        }

        bench_median_gate_python_unary(
            &mut group,
            "u16_unique_1m_null_then_effect",
            "u16_unique_1m",
            &np_unique,
            &fnp_unique,
            &u_a,
        );
        bench_median_gate_python_binary(
            &mut group,
            "u16_union_disjoint_1m_null_then_effect",
            "u16_union_disjoint_1m",
            &np_union,
            &fnp_union,
            &u_a,
            &u_union_b,
        );
        bench_median_gate_python_binary(
            &mut group,
            "u16_setxor_1m_null_then_effect",
            "u16_setxor_1m",
            &np_setxor,
            &fnp_setxor,
            &u_a,
            &u_b,
        );
    });

    group.finish();
}

fn bench_f64_transcendental_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_f64_transcendental_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        println!(
            "ISA_PROVENANCE target_arch={} avx2={} sse2={}",
            std::env::consts::ARCH,
            cfg!(target_feature = "avx2"),
            cfg!(target_feature = "sse2"),
        );
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_f64_transcendental_median_gate")
            .expect("transcendental bench module");
        fnp_python(&module).expect("initialize fnp_python transcendental bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy version")
            .extract::<String>()
            .expect("numpy version string");
        let numpy_cpu_features = numpy
            .getattr("_core")
            .expect("numpy core")
            .getattr("_multiarray_umath")
            .expect("numpy multiarray umath")
            .getattr("__cpu_features__")
            .expect("numpy runtime CPU features")
            .str()
            .expect("numpy runtime CPU feature str")
            .extract::<String>()
            .expect("numpy runtime CPU feature string value");
        println!(
            "NUMPY_PROVENANCE version={numpy_version} runtime_cpu_features={numpy_cpu_features}"
        );
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260710)\n\
                 t_262k = rng.standard_normal(262_144)\n\
                 t_1m = rng.standard_normal(1_048_576)\n\
                 t_4m = rng.standard_normal(4_194_304)\n",
            )
            .expect("transcendental setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("transcendental setup");
        let t_262k = namespace.get_item("t_262k").expect("t_262k present");
        let t_1m = namespace.get_item("t_1m").expect("t_1m present");
        let t_4m = namespace.get_item("t_4m").expect("t_4m present");

        // Diagnostic parity probe (print, not assert): fnp's native f64 route is
        // scalar system libm; numpy may dispatch a SIMD kernel on some workers.
        // Byte-level agreement per worker is itself evidence for the transcendental
        // lane (see the 2026-07-10 ISA addendum), so record it instead of dying.
        for name in ["sin", "cos", "tan", "tanh", "expm1"] {
            let fnp_fn = module.getattr(name).expect("fnp transcendental fn");
            let np_fn = numpy.getattr(name).expect("numpy transcendental fn");
            for (label, input) in [("262k", &t_262k), ("1m", &t_1m), ("4m", &t_4m)] {
                let candidate = fnp_fn.call1((input,)).expect("fnp parity call");
                let base = np_fn.call1((input,)).expect("numpy parity call");
                let candidate_bytes: Vec<u8> = candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract()
                    .expect("candidate byte Vec");
                let base_bytes: Vec<u8> = base
                    .call_method0("tobytes")
                    .expect("base bytes")
                    .extract()
                    .expect("base byte Vec");
                let first_diff = candidate_bytes
                    .chunks_exact(8)
                    .zip(base_bytes.chunks_exact(8))
                    .position(|(a, b)| a != b);
                let diff_count = candidate_bytes
                    .chunks_exact(8)
                    .zip(base_bytes.chunks_exact(8))
                    .filter(|(a, b)| a != b)
                    .count();
                println!(
                    "TRANSCENDENTAL_PARITY op={name} n={label} byte_equal={} \
                     diff_elems={diff_count} first_diff_elem={:?}",
                    candidate_bytes == base_bytes,
                    first_diff,
                );
            }
        }

        let fnp_sin = module.getattr("sin").expect("fnp sin");
        let np_sin = numpy.getattr("sin").expect("numpy sin");
        let fnp_cos = module.getattr("cos").expect("fnp cos");
        let np_cos = numpy.getattr("cos").expect("numpy cos");
        let fnp_tan = module.getattr("tan").expect("fnp tan");
        let np_tan = numpy.getattr("tan").expect("numpy tan");
        let fnp_tanh = module.getattr("tanh").expect("fnp tanh");
        let np_tanh = numpy.getattr("tanh").expect("numpy tanh");
        let fnp_expm1 = module.getattr("expm1").expect("fnp expm1");
        let np_expm1 = numpy.getattr("expm1").expect("numpy expm1");

        bench_median_gate_python_unary(
            &mut group,
            "f64_sin_262k_null_then_effect",
            "f64_sin_262k",
            &np_sin,
            &fnp_sin,
            &t_262k,
        );
        bench_median_gate_python_unary(
            &mut group,
            "f64_sin_1m_null_then_effect",
            "f64_sin_1m",
            &np_sin,
            &fnp_sin,
            &t_1m,
        );
        bench_median_gate_python_unary(
            &mut group,
            "f64_sin_4m_null_then_effect",
            "f64_sin_4m",
            &np_sin,
            &fnp_sin,
            &t_4m,
        );
        bench_median_gate_python_unary(
            &mut group,
            "f64_cos_1m_null_then_effect",
            "f64_cos_1m",
            &np_cos,
            &fnp_cos,
            &t_1m,
        );
        bench_median_gate_python_unary(
            &mut group,
            "f64_tan_1m_null_then_effect",
            "f64_tan_1m",
            &np_tan,
            &fnp_tan,
            &t_1m,
        );
        bench_median_gate_python_unary(
            &mut group,
            "f64_tanh_1m_null_then_effect",
            "f64_tanh_1m",
            &np_tanh,
            &fnp_tanh,
            &t_1m,
        );
        bench_median_gate_python_unary(
            &mut group,
            "f64_expm1_262k_null_then_effect",
            "f64_expm1_262k",
            &np_expm1,
            &fnp_expm1,
            &t_262k,
        );
        bench_median_gate_python_unary(
            &mut group,
            "f64_expm1_1m_null_then_effect",
            "f64_expm1_1m",
            &np_expm1,
            &fnp_expm1,
            &t_1m,
        );
    });

    group.finish();
}

fn bench_f64_exp_log_probe(c: &mut Criterion) {
    // Probe for bead deadlock-audit-gkznn (reopen of the stale 2026-06-09
    // exp/log passthrough decision): (1) BYTE PROBE — does numpy's f64
    // exp/log/log2/log10 output match Rust scalar system-libm bit-for-bit on
    // this worker? (2) TIMING — does a rayon parallel scalar-libm map (with a
    // deliberate vec![0.0; n] zero-init handicap the real zero-copy path would
    // not pay) beat numpy's kernel? Both must hold before any production
    // rewiring; the probe writes evidence only.
    let mut group = c.benchmark_group("python_f64_exp_log_probe");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let numpy = py.import("numpy").expect("numpy oracle");
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy version")
            .extract::<String>()
            .expect("numpy version string");
        println!("EXP_LOG_PROBE_NUMPY version={numpy_version}");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 e_1m = rng.standard_normal(1_048_576)\n\
                 e_4m = rng.standard_normal(4_194_304)\n\
                 l_1m = np.abs(rng.standard_normal(1_048_576)) + 0.5\n\
                 l_4m = np.abs(rng.standard_normal(4_194_304)) + 0.5\n",
            )
            .expect("probe setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("probe setup");
        let e_1m = namespace.get_item("e_1m").expect("e_1m present");
        let e_4m = namespace.get_item("e_4m").expect("e_4m present");
        let l_1m = namespace.get_item("l_1m").expect("l_1m present");
        let l_4m = namespace.get_item("l_4m").expect("l_4m present");

        let to_vec = |arr: &Bound<'_, PyAny>| -> Vec<f64> {
            let raw: Vec<u8> = arr
                .call_method0("tobytes")
                .expect("probe input bytes")
                .extract()
                .expect("probe input byte Vec");
            raw.chunks_exact(8)
                .map(|chunk| f64::from_ne_bytes(chunk.try_into().expect("one native f64")))
                .collect()
        };

        // (1) BYTE PROBE: numpy output vs Rust scalar libm, element-exact.
        for (name, rust_fn, input) in [
            ("exp", f64::exp as fn(f64) -> f64, &e_1m),
            ("log", f64::ln as fn(f64) -> f64, &l_1m),
            ("log2", f64::log2 as fn(f64) -> f64, &l_1m),
            ("log10", f64::log10 as fn(f64) -> f64, &l_1m),
        ] {
            let data = to_vec(input);
            let np_bytes: Vec<u8> = numpy
                .getattr(name)
                .expect("numpy probe fn")
                .call1((input,))
                .expect("numpy probe call")
                .call_method0("tobytes")
                .expect("numpy probe bytes")
                .extract()
                .expect("numpy probe byte Vec");
            let mut diff_elems = 0usize;
            let mut first_diff = None;
            let mut max_bitdiff: u64 = 0;
            for (index, (np_chunk, &value)) in np_bytes.chunks_exact(8).zip(data.iter()).enumerate()
            {
                let np_bits = u64::from_ne_bytes(np_chunk.try_into().expect("np f64 chunk"));
                let mine_bits = rust_fn(value).to_bits();
                if np_bits != mine_bits {
                    diff_elems += 1;
                    if first_diff.is_none() {
                        first_diff = Some(index);
                    }
                    max_bitdiff = max_bitdiff.max(np_bits.abs_diff(mine_bits));
                }
            }
            println!(
                "EXP_LOG_PROBE op={name} n=1m byte_equal={} diff_elems={diff_elems} \
                 first_diff_elem={first_diff:?} max_bitdiff={max_bitdiff}",
                diff_elems == 0,
            );
        }

        // (2) TIMING: ledger-pair ABBA — candidate = parallel scalar-libm map
        // (zero-init handicap), orig = the numpy call. Plus numpy A/A nulls.
        for (row, name, rust_fn, input) in [
            (
                "exp_log_probe_exp_1m",
                "exp",
                f64::exp as fn(f64) -> f64,
                &e_1m,
            ),
            (
                "exp_log_probe_exp_4m",
                "exp",
                f64::exp as fn(f64) -> f64,
                &e_4m,
            ),
            (
                "exp_log_probe_log_1m",
                "log",
                f64::ln as fn(f64) -> f64,
                &l_1m,
            ),
            (
                "exp_log_probe_log_4m",
                "log",
                f64::ln as fn(f64) -> f64,
                &l_4m,
            ),
        ] {
            let data = to_vec(input);
            let np_fn = numpy.getattr(name).expect("numpy timing fn");
            let run_candidate = || {
                let n = data.len();
                let mut out = vec![0.0f64; n];
                let chunk = n.div_ceil(rayon::current_num_threads().max(1));
                out.par_chunks_mut(chunk)
                    .zip(data.par_chunks(chunk))
                    .for_each(|(o, i)| {
                        for (slot, &value) in o.iter_mut().zip(i.iter()) {
                            *slot = rust_fn(value);
                        }
                    });
                out
            };
            let candidate_samples = RefCell::new(Vec::new());
            let orig_samples = RefCell::new(Vec::new());
            let order = Cell::new(0u64);
            group.bench_function(format!("{row}_paired"), |bench| {
                bench.iter_custom(|iterations| {
                    let mut candidate_total = Duration::ZERO;
                    let mut orig_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let orig_first = order.get() & 1 == 1;
                        order.set(order.get().wrapping_add(1));
                        if orig_first {
                            let start = Instant::now();
                            black_box(np_fn.call1((input,)).expect("np timing call"));
                            orig_total += start.elapsed();
                            let start = Instant::now();
                            black_box(run_candidate());
                            candidate_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(run_candidate());
                            candidate_total += start.elapsed();
                            let start = Instant::now();
                            black_box(np_fn.call1((input,)).expect("np timing call"));
                            orig_total += start.elapsed();
                        }
                    }
                    candidate_samples
                        .borrow_mut()
                        .push(candidate_total.as_secs_f64() * 1e9 / iterations as f64);
                    orig_samples
                        .borrow_mut()
                        .push(orig_total.as_secs_f64() * 1e9 / iterations as f64);
                    candidate_total + orig_total
                });
            });
            report_ledger_pair(row, &candidate_samples, &orig_samples);

            let null_a = RefCell::new(Vec::new());
            let null_b = RefCell::new(Vec::new());
            let null_order = Cell::new(0u64);
            group.bench_function(format!("{row}_null_aa"), |bench| {
                bench.iter_custom(|iterations| {
                    let mut a_total = Duration::ZERO;
                    let mut b_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let b_first = null_order.get() & 1 == 1;
                        null_order.set(null_order.get().wrapping_add(1));
                        if b_first {
                            let start = Instant::now();
                            black_box(np_fn.call1((input,)).expect("np null call"));
                            b_total += start.elapsed();
                            let start = Instant::now();
                            black_box(np_fn.call1((input,)).expect("np null call"));
                            a_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(np_fn.call1((input,)).expect("np null call"));
                            a_total += start.elapsed();
                            let start = Instant::now();
                            black_box(np_fn.call1((input,)).expect("np null call"));
                            b_total += start.elapsed();
                        }
                    }
                    null_a
                        .borrow_mut()
                        .push(a_total.as_secs_f64() * 1e9 / iterations as f64);
                    null_b
                        .borrow_mut()
                        .push(b_total.as_secs_f64() * 1e9 / iterations as f64);
                    a_total + b_total
                });
            });
            report_ledger_pair(&format!("{row}_null"), &null_a, &null_b);
        }
    });

    group.finish();
}

fn bench_f64_exp_log_median_gate(c: &mut Criterion) {
    // SHIP rows for bead deadlock-audit-gkznn: the ACTUAL wired route
    // (fnp.exp/log/log2/log10 -> try_zerocopy_f64_unary parallel scalar-libm
    // map on non-AVX-512 hosts) vs numpy, with pre-timing byte parity asserts.
    // On an avx512f worker the ISA gate routes these to the numpy passthrough
    // and the rows read ~1.0x by construction; the probe group's
    // EXP_LOG_PROBE byte rows identify the worker class in the same run.
    let mut group = c.benchmark_group("python_f64_exp_log_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module =
            PyModule::new(py, "fnp_python_exp_log_median_gate").expect("exp/log bench module");
        fnp_python(&module).expect("initialize fnp_python exp/log bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        // Worker-class provenance: the rows below are only expected to beat
        // numpy where the ISA gate enables the native route (x86-64 with
        // avx512f=false); elsewhere they measure passthrough-vs-numpy ~1.0x.
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy version")
            .extract::<String>()
            .expect("numpy version string");
        #[cfg(target_arch = "x86_64")]
        let native_route = !std::arch::is_x86_feature_detected!("avx512f");
        #[cfg(not(target_arch = "x86_64"))]
        let native_route = false;
        println!("EXP_LOG_GATE_WORKER numpy={numpy_version} native_route={native_route}");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 e_1m = rng.standard_normal(1_048_576)\n\
                 e_4m = rng.standard_normal(4_194_304)\n\
                 l_1m = np.abs(rng.standard_normal(1_048_576)) + 0.5\n\
                 l_4m = np.abs(rng.standard_normal(4_194_304)) + 0.5\n",
            )
            .expect("exp/log gate setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("exp/log gate setup");
        let e_1m = namespace.get_item("e_1m").expect("e_1m present");
        let e_4m = namespace.get_item("e_4m").expect("e_4m present");
        let l_1m = namespace.get_item("l_1m").expect("l_1m present");
        let l_4m = namespace.get_item("l_4m").expect("l_4m present");

        let rows = [
            (
                "explog_exp_1m_null_then_effect",
                "explog_exp_1m",
                "exp",
                &e_1m,
            ),
            (
                "explog_exp_4m_null_then_effect",
                "explog_exp_4m",
                "exp",
                &e_4m,
            ),
            (
                "explog_exp2_4m_null_then_effect",
                "explog_exp2_4m",
                "exp2",
                &e_4m,
            ),
            (
                "explog_log_1m_null_then_effect",
                "explog_log_1m",
                "log",
                &l_1m,
            ),
            (
                "explog_log_4m_null_then_effect",
                "explog_log_4m",
                "log",
                &l_4m,
            ),
            (
                "explog_log2_4m_null_then_effect",
                "explog_log2_4m",
                "log2",
                &l_4m,
            ),
            (
                "explog_log10_4m_null_then_effect",
                "explog_log10_4m",
                "log10",
                &l_4m,
            ),
        ];
        for (bench_name, row, op, input) in rows {
            let fnp_fn = module.getattr(op).expect("fnp exp/log fn");
            let np_fn = numpy.getattr(op).expect("numpy exp/log fn");
            let candidate = fnp_fn.call1((input,)).expect("fnp exp/log parity call");
            let base = np_fn.call1((input,)).expect("numpy exp/log parity call");
            assert_eq!(
                candidate
                    .getattr("dtype")
                    .expect("candidate dtype")
                    .str()
                    .expect("candidate dtype str")
                    .to_string(),
                base.getattr("dtype")
                    .expect("base dtype")
                    .str()
                    .expect("base dtype str")
                    .to_string(),
                "exp/log {row} dtype parity",
            );
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "exp/log {row} byte parity",
            );
            bench_median_gate_python_unary(&mut group, bench_name, row, &np_fn, &fnp_fn, input);
        }
    });

    group.finish();
}

fn bench_bool_sort_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_bool_sort_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module =
            PyModule::new(py, "fnp_python_bool_sort_median_gate").expect("bool sort bench module");
        fnp_python(&module).expect("initialize fnp_python bool sort bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 b_8m = rng.integers(0, 2, 8_000_000).astype(bool)\n\
                 b_2m = rng.integers(0, 2, 2_000_000).astype(bool)\n",
            )
            .expect("bool sort setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("bool sort setup");
        let b_8m = namespace.get_item("b_8m").expect("b_8m present");
        let b_2m = namespace.get_item("b_2m").expect("b_2m present");

        let fnp_sort = module.getattr("sort").expect("fnp sort");
        let np_sort = numpy.getattr("sort").expect("numpy sort");
        for (label, input) in [("8m", &b_8m), ("2m", &b_2m)] {
            let candidate = fnp_sort.call1((input,)).expect("fnp bool sort parity");
            let base = np_sort.call1((input,)).expect("numpy bool sort parity");
            assert_eq!(
                candidate
                    .getattr("dtype")
                    .expect("candidate dtype")
                    .str()
                    .expect("candidate dtype str")
                    .to_string(),
                base.getattr("dtype")
                    .expect("base dtype")
                    .str()
                    .expect("base dtype str")
                    .to_string(),
                "bool sort {label} dtype parity",
            );
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "bool sort {label} byte parity",
            );
        }

        bench_median_gate_python_unary(
            &mut group,
            "bool_sort_8m_null_then_effect",
            "bool_sort_8m",
            &np_sort,
            &fnp_sort,
            &b_8m,
        );
        bench_median_gate_python_unary(
            &mut group,
            "bool_sort_2m_null_then_effect",
            "bool_sort_2m",
            &np_sort,
            &fnp_sort,
            &b_2m,
        );
    });

    group.finish();
}

fn bench_wide_string_sort_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_wide_string_sort_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_wide_string_sort_median_gate")
            .expect("wide string sort bench module");
        fnp_python(&module).expect("initialize fnp_python wide string sort bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 u9 = rng.integers(97, 123, (1_000_000, 9), dtype=np.uint32).reshape(-1).view('U9')\n\
                 u16 = rng.integers(97, 123, (1_000_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
                 s9 = rng.integers(97, 123, (1_000_000, 9), dtype=np.uint8).reshape(-1).view('S9')\n\
                 s16 = rng.integers(97, 123, (1_000_000, 16), dtype=np.uint8).reshape(-1).view('S16')\n",
            )
            .expect("wide string sort setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("wide string sort setup");
        let u9_input = namespace.get_item("u9").expect("u9 present");
        let u16_input = namespace.get_item("u16").expect("u16 present");
        let s9_input = namespace.get_item("s9").expect("s9 present");
        let s16_input = namespace.get_item("s16").expect("s16 present");
        let fnp_sort = module.getattr("sort").expect("fnp sort");
        let numpy_sort = numpy.getattr("sort").expect("numpy sort");

        for (label, input) in [
            ("U9", &u9_input),
            ("U16", &u16_input),
            ("S9", &s9_input),
            ("S16", &s16_input),
        ] {
            let candidate = fnp_sort
                .call1((input,))
                .expect("fnp wide string sort parity");
            let base = numpy_sort
                .call1((input,))
                .expect("numpy wide string sort parity");
            assert_eq!(
                candidate
                    .getattr("dtype")
                    .expect("candidate dtype")
                    .str()
                    .expect("candidate dtype str")
                    .to_string(),
                base.getattr("dtype")
                    .expect("base dtype")
                    .str()
                    .expect("base dtype str")
                    .to_string(),
                "wide string sort {label} dtype parity",
            );
            assert_eq!(
                candidate
                    .getattr("shape")
                    .expect("candidate shape")
                    .extract::<Vec<usize>>()
                    .expect("candidate shape Vec"),
                base.getattr("shape")
                    .expect("base shape")
                    .extract::<Vec<usize>>()
                    .expect("base shape Vec"),
                "wide string sort {label} shape parity",
            );
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "wide string sort {label} byte parity",
            );
            assert_eq!(
                candidate
                    .getattr("flags")
                    .expect("candidate flags")
                    .getattr("owndata")
                    .expect("candidate owndata")
                    .extract::<bool>()
                    .expect("candidate owndata bool"),
                base.getattr("flags")
                    .expect("base flags")
                    .getattr("owndata")
                    .expect("base owndata")
                    .extract::<bool>()
                    .expect("base owndata bool"),
                "wide string sort {label} ownership parity",
            );
        }

        group.bench_function("wide_string_sort_u16_1m_fnp_profile", |bench| {
            bench.iter(|| {
                black_box(
                    fnp_sort
                        .call1((black_box(&u16_input),))
                        .expect("profile fnp U16 sort"),
                )
            });
        });
        bench_median_gate_python_unary(
            &mut group,
            "wide_string_sort_u9_1m_null_then_effect",
            "wide_string_sort_u9_1m",
            &numpy_sort,
            &fnp_sort,
            &u9_input,
        );
        bench_median_gate_python_unary(
            &mut group,
            "wide_string_sort_u16_1m_null_then_effect",
            "wide_string_sort_u16_1m",
            &numpy_sort,
            &fnp_sort,
            &u16_input,
        );
        bench_median_gate_python_unary(
            &mut group,
            "wide_string_sort_s9_1m_null_then_effect",
            "wide_string_sort_s9_1m",
            &numpy_sort,
            &fnp_sort,
            &s9_input,
        );
        bench_median_gate_python_unary(
            &mut group,
            "wide_string_sort_s16_1m_null_then_effect",
            "wide_string_sort_s16_1m",
            &numpy_sort,
            &fnp_sort,
            &s16_input,
        );
    });

    group.finish();
    if std::env::var_os("FNP_WIDE_STRING_SORT_BENCH_ONLY").is_some() {
        use std::io::Write;
        let _ = std::io::stdout().flush();
        std::process::exit(0);
    }
}

fn bench_accumulate_extremum_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_accumulate_extremum_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_accumulate_extremum_median_gate")
            .expect("accumulate extremum bench module");
        fnp_python(&module).expect("initialize fnp_python accumulate extremum bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 x = rng.standard_normal(8_000_000).astype(np.float64)\n",
            )
            .expect("accumulate extremum setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("accumulate extremum setup");
        let input = namespace.get_item("x").expect("x present");
        let fnp_accumulate = module
            .getattr("maximum")
            .expect("fnp maximum")
            .getattr("accumulate")
            .expect("fnp maximum.accumulate");
        let numpy_accumulate = numpy
            .getattr("maximum")
            .expect("numpy maximum")
            .getattr("accumulate")
            .expect("numpy maximum.accumulate");

        let candidate = fnp_accumulate
            .call1((&input,))
            .expect("fnp maximum.accumulate parity");
        let base = numpy_accumulate
            .call1((&input,))
            .expect("numpy maximum.accumulate parity");
        assert_eq!(
            candidate
                .getattr("dtype")
                .expect("candidate dtype")
                .str()
                .expect("candidate dtype str")
                .to_string(),
            base.getattr("dtype")
                .expect("base dtype")
                .str()
                .expect("base dtype str")
                .to_string(),
            "maximum.accumulate dtype parity",
        );
        assert_eq!(
            candidate
                .getattr("shape")
                .expect("candidate shape")
                .extract::<Vec<usize>>()
                .expect("candidate shape Vec"),
            base.getattr("shape")
                .expect("base shape")
                .extract::<Vec<usize>>()
                .expect("base shape Vec"),
            "maximum.accumulate shape parity",
        );
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "maximum.accumulate byte parity",
        );

        group.bench_function("maximum_accumulate_f64_8m_fnp_profile", |bench| {
            bench.iter(|| {
                black_box(
                    fnp_accumulate
                        .call1((black_box(&input),))
                        .expect("profile fnp maximum.accumulate"),
                )
            });
        });
        bench_median_gate_python_unary(
            &mut group,
            "maximum_accumulate_f64_8m_null_then_effect",
            "maximum_accumulate_f64_8m",
            &numpy_accumulate,
            &fnp_accumulate,
            &input,
        );
    });

    group.finish();
    if std::env::var_os("FNP_ACCUMULATE_EXTREMUM_BENCH_ONLY").is_some() {
        use std::io::Write;
        let _ = std::io::stdout().flush();
        std::process::exit(0);
    }
}

fn bench_int_convolve_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_int_convolve_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_int_convolve_median_gate")
            .expect("int convolve bench module");
        fnp_python(&module).expect("initialize fnp_python int convolve bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 a = rng.integers(-2**31, 2**31, 200_000).astype(np.int64)\n\
                 v = rng.integers(-2**31, 2**31, 256).astype(np.int64)\n",
            )
            .expect("int convolve setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("int convolve setup");
        let a = namespace.get_item("a").expect("a present");
        let v = namespace.get_item("v").expect("v present");
        let fnp_convolve = module.getattr("convolve").expect("fnp convolve");
        let numpy_convolve = numpy.getattr("convolve").expect("numpy convolve");

        let candidate = fnp_convolve
            .call1((&a, &v))
            .expect("fnp int convolve parity");
        let base = numpy_convolve
            .call1((&a, &v))
            .expect("numpy int convolve parity");
        assert_eq!(
            candidate
                .getattr("dtype")
                .expect("candidate dtype")
                .str()
                .expect("candidate dtype str")
                .to_string(),
            base.getattr("dtype")
                .expect("base dtype")
                .str()
                .expect("base dtype str")
                .to_string(),
            "int convolve dtype parity",
        );
        assert_eq!(
            candidate
                .getattr("shape")
                .expect("candidate shape")
                .extract::<Vec<usize>>()
                .expect("candidate shape Vec"),
            base.getattr("shape")
                .expect("base shape")
                .extract::<Vec<usize>>()
                .expect("base shape Vec"),
            "int convolve shape parity",
        );
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "int convolve byte parity",
        );

        group.bench_function("int_convolve_i64_200k_256_fnp_profile", |bench| {
            bench.iter(|| {
                black_box(
                    fnp_convolve
                        .call1((black_box(&a), black_box(&v)))
                        .expect("profile fnp int convolve"),
                )
            });
        });
        bench_median_gate_python_binary(
            &mut group,
            "int_convolve_i64_200k_256_null_then_effect",
            "int_convolve_i64_200k_256",
            &numpy_convolve,
            &fnp_convolve,
            &a,
            &v,
        );
    });

    group.finish();
    if std::env::var_os("FNP_INT_CONVOLVE_BENCH_ONLY").is_some() {
        use std::io::Write;
        let _ = std::io::stdout().flush();
        std::process::exit(0);
    }
}

fn bench_int_matmul_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_int_matmul_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_int_matmul_median_gate")
            .expect("int matmul bench module");
        fnp_python(&module).expect("initialize fnp_python int matmul bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 a64 = rng.integers(-2**31, 2**31, (512, 512)).astype(np.int64)\n\
                 b64 = rng.integers(-2**31, 2**31, (512, 512)).astype(np.int64)\n\
                 a32 = rng.integers(-2**15, 2**15, (512, 512)).astype(np.int32)\n\
                 b32 = rng.integers(-2**15, 2**15, (512, 512)).astype(np.int32)\n\
                 ab64 = rng.integers(-2**31, 2**31, (64, 128, 128)).astype(np.int64)\n\
                 bb64 = rng.integers(-2**31, 2**31, (64, 128, 128)).astype(np.int64)\n\
                 mp64 = rng.integers(-2**31, 2**31, (256, 256)).astype(np.int64)\n\
                 p5 = 5\n",
            )
            .expect("int matmul setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("int matmul setup");
        let a64 = namespace.get_item("a64").expect("a64 present");
        let b64 = namespace.get_item("b64").expect("b64 present");
        let a32 = namespace.get_item("a32").expect("a32 present");
        let b32 = namespace.get_item("b32").expect("b32 present");

        let fnp_matmul = module.getattr("matmul").expect("fnp matmul");
        let np_matmul = numpy.getattr("matmul").expect("numpy matmul");
        for (label, x, y) in [("i64_512", &a64, &b64), ("i32_512", &a32, &b32)] {
            let candidate = fnp_matmul.call1((x, y)).expect("fnp int matmul parity");
            let base = np_matmul.call1((x, y)).expect("numpy int matmul parity");
            assert_eq!(
                candidate
                    .getattr("dtype")
                    .expect("candidate dtype")
                    .str()
                    .expect("candidate dtype str")
                    .to_string(),
                base.getattr("dtype")
                    .expect("base dtype")
                    .str()
                    .expect("base dtype str")
                    .to_string(),
                "int matmul {label} dtype parity",
            );
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "int matmul {label} byte parity",
            );
        }

        bench_median_gate_python_binary(
            &mut group,
            "int_matmul_i64_512_null_then_effect",
            "int_matmul_i64_512",
            &np_matmul,
            &fnp_matmul,
            &a64,
            &b64,
        );
        bench_median_gate_python_binary(
            &mut group,
            "int_matmul_i32_512_null_then_effect",
            "int_matmul_i32_512",
            &np_matmul,
            &fnp_matmul,
            &a32,
            &b32,
        );

        let ab64 = namespace.get_item("ab64").expect("ab64 present");
        let bb64 = namespace.get_item("bb64").expect("bb64 present");
        let mp64 = namespace.get_item("mp64").expect("mp64 present");
        let p5 = namespace.get_item("p5").expect("p5 present");
        let fnp_matrix_power = module
            .getattr("linalg")
            .expect("fnp linalg")
            .getattr("matrix_power")
            .expect("fnp matrix_power");
        let np_matrix_power = numpy
            .getattr("linalg")
            .expect("numpy linalg")
            .getattr("matrix_power")
            .expect("numpy matrix_power");
        for (label, f_c, f_b, x, y) in [
            ("i64_batched", &fnp_matmul, &np_matmul, &ab64, &bb64),
            (
                "i64_matpow5",
                &fnp_matrix_power,
                &np_matrix_power,
                &mp64,
                &p5,
            ),
        ] {
            let candidate = f_c.call1((x, y)).expect("fnp candidate parity");
            let base = f_b.call1((x, y)).expect("numpy base parity");
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "int {label} byte parity",
            );
        }
        bench_median_gate_python_binary(
            &mut group,
            "int_matmul_i64_batched_null_then_effect",
            "int_matmul_i64_batched",
            &np_matmul,
            &fnp_matmul,
            &ab64,
            &bb64,
        );
        bench_median_gate_python_binary(
            &mut group,
            "int_matrix_power_i64_256_p5_null_then_effect",
            "int_matrix_power_i64_256_p5",
            &np_matrix_power,
            &fnp_matrix_power,
            &mp64,
            &p5,
        );
    });

    group.finish();
}

fn bench_f16_matmul_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_f16_matmul_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_f16_matmul_median_gate")
            .expect("f16 matmul bench module");
        fnp_python(&module).expect("initialize fnp_python f16 matmul bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 h_a = rng.standard_normal((512, 512)).astype(np.float16)\n\
                 h_b = rng.standard_normal((512, 512)).astype(np.float16)\n\
                 hb_a = rng.standard_normal((8, 256, 256)).astype(np.float16)\n\
                 hb_b = rng.standard_normal((8, 256, 256)).astype(np.float16)\n\
                 hbc_a = rng.standard_normal((32, 128, 128)).astype(np.float16)\n\
                 hbc_b = rng.standard_normal((128, 96)).astype(np.float16)\n",
            )
            .expect("f16 matmul setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("f16 matmul setup");
        let h_a = namespace.get_item("h_a").expect("h_a present");
        let h_b = namespace.get_item("h_b").expect("h_b present");

        let fnp_matmul = module.getattr("matmul").expect("fnp matmul");
        let np_matmul = numpy.getattr("matmul").expect("numpy matmul");
        let candidate = fnp_matmul
            .call1((&h_a, &h_b))
            .expect("fnp f16 matmul parity");
        let base = np_matmul
            .call1((&h_a, &h_b))
            .expect("numpy f16 matmul parity");
        assert_eq!(
            candidate
                .getattr("dtype")
                .expect("candidate dtype")
                .str()
                .expect("candidate dtype str")
                .to_string(),
            base.getattr("dtype")
                .expect("base dtype")
                .str()
                .expect("base dtype str")
                .to_string(),
            "f16 matmul dtype parity",
        );
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 matmul byte parity",
        );

        bench_median_gate_python_binary(
            &mut group,
            "f16_matmul_512_null_then_effect",
            "f16_matmul_512",
            &np_matmul,
            &fnp_matmul,
            &h_a,
            &h_b,
        );

        let hb_a = namespace.get_item("hb_a").expect("hb_a present");
        let hb_b = namespace.get_item("hb_b").expect("hb_b present");
        let candidate = fnp_matmul
            .call1((&hb_a, &hb_b))
            .expect("fnp f16 batched matmul parity");
        let base = np_matmul
            .call1((&hb_a, &hb_b))
            .expect("numpy f16 batched matmul parity");
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 batched matmul byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_matmul_batched_8x256_null_then_effect",
            "f16_matmul_batched_8x256",
            &np_matmul,
            &fnp_matmul,
            &hb_a,
            &hb_b,
        );

        let hbc_a = namespace.get_item("hbc_a").expect("hbc_a present");
        let hbc_b = namespace.get_item("hbc_b").expect("hbc_b present");
        let candidate = fnp_matmul
            .call1((&hbc_a, &hbc_b))
            .expect("fnp f16 broadcast matmul parity");
        let base = np_matmul
            .call1((&hbc_a, &hbc_b))
            .expect("numpy f16 broadcast matmul parity");
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 broadcast matmul byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_matmul_broadcast_32x128_null_then_effect",
            "f16_matmul_broadcast_32x128",
            &np_matmul,
            &fnp_matmul,
            &hbc_a,
            &hbc_b,
        );
    });

    group.finish();
}

fn bench_f16_unique_median_gate(c: &mut Criterion) {
    // f16 unique at 8M: presence-table walk vs numpy's ~600ms-class sort.
    let mut group = c.benchmark_group("python_f16_unique_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module =
            PyModule::new(py, "fnp_python_f16_unique_median_gate").expect("unique bench module");
        fnp_python(&module).expect("initialize fnp_python unique bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260716)\n\
                 uq16 = (rng.standard_normal(8_000_000) * 2).astype(np.float16)\n",
            )
            .expect("unique setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("unique setup");
        let uq16 = namespace.get_item("uq16").expect("uq16 present");
        let fnp_unique = module.getattr("unique").expect("fnp unique");
        let np_unique = numpy.getattr("unique").expect("numpy unique");
        let candidate = fnp_unique.call1((&uq16,)).expect("fnp unique parity");
        let base = np_unique.call1((&uq16,)).expect("numpy unique parity");
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 unique byte parity",
        );
        bench_median_gate_python_unary(
            &mut group,
            "f16_unique_8m_null_then_effect",
            "f16_unique_8m",
            &np_unique,
            &fnp_unique,
            &uq16,
        );

        // f16 isin at 8M/1k: presence-bitmap membership vs numpy's ~1.2s sort path.
        py.run(
            std::ffi::CString::new("iq16 = (rng.standard_normal(1000) * 2).astype(np.float16)\n")
                .expect("isin setup CString")
                .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("isin setup");
        let iq16 = namespace.get_item("iq16").expect("iq16 present");
        let fnp_isin = module.getattr("isin").expect("fnp isin");
        let np_isin = numpy.getattr("isin").expect("numpy isin");
        let candidate_i = fnp_isin.call1((&uq16, &iq16)).expect("fnp isin parity");
        let base_i = np_isin.call1((&uq16, &iq16)).expect("numpy isin parity");
        assert_eq!(
            candidate_i
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_i
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 isin byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_isin_8m_null_then_effect",
            "f16_isin_8m",
            &np_isin,
            &fnp_isin,
            &uq16,
            &iq16,
        );
    });

    group.finish();
}

fn bench_f16_around_median_gate(c: &mut Criterion) {
    // f16 round(a, 2) at 8M: the per-step-narrow chain kernel vs numpy's serial
    // half multiply->rint->divide loops (~90ms class on hz1).
    let mut group = c.benchmark_group("python_f16_around_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module =
            PyModule::new(py, "fnp_python_f16_around_median_gate").expect("around bench module");
        fnp_python(&module).expect("initialize fnp_python around bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260718)\n\
                 ra16 = (rng.standard_normal(8_000_000) * 2).astype(np.float16)\n\
                 dec2 = 2\n",
            )
            .expect("around setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("around setup");
        let ra16 = namespace.get_item("ra16").expect("ra16 present");
        let dec2 = namespace.get_item("dec2").expect("dec2 present");
        let fnp_round = module.getattr("round").expect("fnp round");
        let np_round = numpy.getattr("round").expect("numpy round");
        let candidate = fnp_round.call1((&ra16, &dec2)).expect("fnp round parity");
        let base = np_round.call1((&ra16, &dec2)).expect("numpy round parity");
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 around byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_around_8m_null_then_effect",
            "f16_around_8m",
            &np_round,
            &fnp_round,
            &ra16,
            &dec2,
        );
    });

    group.finish();
}

fn bench_isclose_median_gate(c: &mut Criterion) {
    // f64 array-array isclose at 8M: the parallelized zero-copy predicate vs
    // numpy's temp-heavy ufunc chain (~180ms class on hz1).
    let mut group = c.benchmark_group("python_isclose_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module =
            PyModule::new(py, "fnp_python_isclose_median_gate").expect("isclose bench module");
        fnp_python(&module).expect("initialize fnp_python isclose bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260716)\n\
                 ic_a = rng.standard_normal(8_000_000)\n\
                 ic_b = ic_a + rng.standard_normal(8_000_000) * 1e-7\n",
            )
            .expect("isclose setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("isclose setup");
        let ic_a = namespace.get_item("ic_a").expect("ic_a present");
        let ic_b = namespace.get_item("ic_b").expect("ic_b present");
        let fnp_isclose = module.getattr("isclose").expect("fnp isclose");
        let np_isclose = numpy.getattr("isclose").expect("numpy isclose");
        let candidate = fnp_isclose
            .call1((&ic_a, &ic_b))
            .expect("fnp isclose parity");
        let base = np_isclose
            .call1((&ic_a, &ic_b))
            .expect("numpy isclose parity");
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "isclose byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "isclose_f64_8m_null_then_effect",
            "isclose_f64_8m",
            &np_isclose,
            &fnp_isclose,
            &ic_a,
            &ic_b,
        );
    });

    group.finish();
}

fn bench_multidot_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_multidot_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module =
            PyModule::new(py, "fnp_python_multidot_median_gate").expect("multidot bench module");
        fnp_python(&module).expect("initialize fnp_python multidot bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 md_args = [rng.standard_normal((512, 512)) for _ in range(3)]\n",
            )
            .expect("multidot setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("multidot setup");
        let md_args = namespace.get_item("md_args").expect("md_args present");

        let fnp_multidot = module
            .getattr("linalg")
            .expect("fnp linalg")
            .getattr("multi_dot")
            .expect("fnp multi_dot");
        let np_multidot = numpy
            .getattr("linalg")
            .expect("numpy linalg")
            .getattr("multi_dot")
            .expect("numpy multi_dot");
        let candidate = fnp_multidot
            .call1((&md_args,))
            .expect("fnp multi_dot parity");
        let base = np_multidot
            .call1((&md_args,))
            .expect("numpy multi_dot parity");
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "multi_dot byte parity",
        );

        bench_median_gate_python_unary(
            &mut group,
            "multidot_3x512_null_then_effect",
            "multidot_3x512",
            &np_multidot,
            &fnp_multidot,
            &md_args,
        );

        // f16 3-chain: numpy's pairs are the naive ~245x f16 loops; fnp
        // routes both pairs through the shipped byte-matched f16 matmul
        // kernel per the replicated _multi_dot_three order rule.
        py.run(
            std::ffi::CString::new(
                "md16_args = [(rng.standard_normal((256, 256)) * 0.3).astype(np.float16) for _ in range(3)]\n",
            )
            .expect("multidot f16 setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("multidot f16 setup");
        let md16_args = namespace.get_item("md16_args").expect("md16_args present");
        let candidate16 = fnp_multidot
            .call1((&md16_args,))
            .expect("fnp f16 multi_dot parity");
        let base16 = np_multidot
            .call1((&md16_args,))
            .expect("numpy f16 multi_dot parity");
        assert_eq!(
            candidate16
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base16
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 multi_dot byte parity",
        );
        bench_median_gate_python_unary(
            &mut group,
            "multidot_f16_3x256_null_then_effect",
            "multidot_f16_3x256",
            &np_multidot,
            &fnp_multidot,
            &md16_args,
        );
    });

    group.finish();
}

fn bench_f16_einsum_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_f16_einsum_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_f16_einsum_median_gate")
            .expect("f16 einsum bench module");
        fnp_python(&module).expect("initialize fnp_python f16 einsum bench module");
        let _numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        namespace
            .set_item("fnp_mod", &module)
            .expect("expose fnp module");
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 es_a = (rng.standard_normal((512, 512)) * 0.3).astype(np.float16)\n\
                 es_b = (rng.standard_normal((512, 512)) * 0.3).astype(np.float16)\n\
                 fnp_es = lambda a, b: fnp_mod.einsum('ij,jk->ik', a, b)\n\
                 np_es = lambda a, b: np.einsum('ij,jk->ik', a, b)\n\
                 fnp_es_t = lambda a, b: fnp_mod.einsum('ij,lj->il', a, b)\n\
                 np_es_t = lambda a, b: np.einsum('ij,lj->il', a, b)\n\
                 fnp_es_g = lambda a, b: fnp_mod.einsum('ji,jl->il', a, b)\n\
                 np_es_g = lambda a, b: np.einsum('ji,jl->il', a, b)\n\
                 fnp_es_ts = lambda a, b: fnp_mod.einsum('ij,lj->li', a, b)\n\
                 np_es_ts = lambda a, b: np.einsum('ij,lj->li', a, b)\n\
                 fnp_es_gs = lambda a, b: fnp_mod.einsum('ji,jl->li', a, b)\n\
                 np_es_gs = lambda a, b: np.einsum('ji,jl->li', a, b)\n\
                 dot_a = (rng.standard_normal(8_388_608) * 0.3).astype(np.float16)\n\
                 dot_b = (rng.standard_normal(8_388_608) * 0.3).astype(np.float16)\n\
                 fnp_es_d = lambda a, b: fnp_mod.einsum('j,j->', a, b)\n\
                 np_es_d = lambda a, b: np.einsum('j,j->', a, b)\n\
                 fc_a = (rng.standard_normal((2896, 2896)) * 0.3).astype(np.float16)\n\
                 fc_b = (rng.standard_normal((2896, 2896)) * 0.3).astype(np.float16)\n\
                 fnp_es_fc = lambda a, b: fnp_mod.einsum('ij,ij->', a, b)\n\
                 np_es_fc = lambda a, b: np.einsum('ij,ij->', a, b)\n\
                 fnp_es_ew = lambda a, b: fnp_mod.einsum('j,j->j', a, b)\n\
                 np_es_ew = lambda a, b: np.einsum('j,j->j', a, b)\n\
                 ew64_a = rng.standard_normal(8_388_608)\n\
                 ew64_b = rng.standard_normal(8_388_608)\n\
                 bc64_full = rng.standard_normal((2896, 2896))\n\
                 bc64_vec = rng.standard_normal(2896)\n\
                 red16 = (rng.standard_normal((2896, 2896)) * 0.3).astype(np.float16)\n\
                 red64 = rng.standard_normal((2896, 2896))\n\
                 red32 = rng.standard_normal((2896, 2896)).astype(np.float32)\n\
                 ch_a = (rng.standard_normal((512, 512)) * 0.3).astype(np.float16)\n\
                 ch_b = (rng.standard_normal((512, 512)) * 0.3).astype(np.float16)\n\
                 ch_c = (rng.standard_normal((512, 512)) * 0.3).astype(np.float16)\n\
                 fnp_es_ch = lambda a, b: fnp_mod.einsum('ij,jk,kl->il', a, b, ch_c, optimize=True)\n\
                 np_es_ch = lambda a, b: np.einsum('ij,jk,kl->il', a, b, ch_c, optimize=True)\n\
                 fnp_es_rj = lambda a: fnp_mod.einsum('ij->j', a)\n\
                 np_es_rj = lambda a: np.einsum('ij->j', a)\n\
                 fnp_es_ri = lambda a: fnp_mod.einsum('ij->i', a)\n\
                 np_es_ri = lambda a: np.einsum('ij->i', a)\n\
                 fnp_es_bc = lambda a, b: fnp_mod.einsum('ij,j->ij', a, b)\n\
                 np_es_bc = lambda a, b: np.einsum('ij,j->ij', a, b)\n\
                 bat_a = (rng.standard_normal((8, 256, 256)) * 0.3).astype(np.float16)\n\
                 bat_b = (rng.standard_normal((8, 256, 256)) * 0.3).astype(np.float16)\n\
                 fnp_es_b = lambda a, b: fnp_mod.einsum('bij,bjk->bik', a, b)\n\
                 np_es_b = lambda a, b: np.einsum('bij,bjk->bik', a, b)\n\
                 fnp_es_bt = lambda a, b: fnp_mod.einsum('bij,blj->bil', a, b)\n\
                 np_es_bt = lambda a, b: np.einsum('bij,blj->bil', a, b)\n\
                 fnp_es_bg = lambda a, b: fnp_mod.einsum('bji,bjl->bil', a, b)\n\
                 np_es_bg = lambda a, b: np.einsum('bji,bjl->bil', a, b)\n\
                 fnp_es_bts = lambda a, b: fnp_mod.einsum('bij,blj->bli', a, b)\n\
                 np_es_bts = lambda a, b: np.einsum('bij,blj->bli', a, b)\n\
                 fnp_es_bgs = lambda a, b: fnp_mod.einsum('bji,bjl->bli', a, b)\n\
                 np_es_bgs = lambda a, b: np.einsum('bji,bjl->bli', a, b)\n",
            )
            .expect("f16 einsum setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("f16 einsum setup");
        let es_a = namespace.get_item("es_a").expect("es_a present");
        let es_b = namespace.get_item("es_b").expect("es_b present");
        let fnp_es = namespace.get_item("fnp_es").expect("fnp_es present");
        let np_es = namespace.get_item("np_es").expect("np_es present");

        let candidate = fnp_es.call1((&es_a, &es_b)).expect("fnp f16 einsum parity");
        let base = np_es
            .call1((&es_a, &es_b))
            .expect("numpy f16 einsum parity");
        assert_eq!(
            candidate
                .getattr("dtype")
                .expect("candidate dtype")
                .str()
                .expect("candidate dtype str")
                .to_string(),
            base.getattr("dtype")
                .expect("base dtype")
                .str()
                .expect("base dtype str")
                .to_string(),
            "f16 einsum dtype parity",
        );
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum byte parity",
        );

        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_matmul_512_null_then_effect",
            "f16_einsum_matmul_512",
            &np_es,
            &fnp_es,
            &es_a,
            &es_b,
        );

        // Transposed spec ('ij,lj->il', the a@b.T idiom): a different numpy
        // contract class (wide-accumulate-once blocked-4) with its own kernel.
        let fnp_es_t = namespace.get_item("fnp_es_t").expect("fnp_es_t present");
        let np_es_t = namespace.get_item("np_es_t").expect("np_es_t present");
        let candidate_t = fnp_es_t
            .call1((&es_a, &es_b))
            .expect("fnp f16 einsum transposed parity");
        let base_t = np_es_t
            .call1((&es_a, &es_b))
            .expect("numpy f16 einsum transposed parity");
        assert_eq!(
            candidate_t
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_t
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum transposed byte parity",
        );

        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_transposed_512_null_then_effect",
            "f16_einsum_transposed_512",
            &np_es_t,
            &fnp_es_t,
            &es_a,
            &es_b,
        );

        // Gram spec ('ji,jl->il', the a.T@b idiom): the third numpy contract
        // class (per-step-narrow muladd rows, stride0_contig_outcontig) with
        // its own kernel. Same 512^2 operands (k = leading axis).
        let fnp_es_g = namespace.get_item("fnp_es_g").expect("fnp_es_g present");
        let np_es_g = namespace.get_item("np_es_g").expect("np_es_g present");
        let candidate_g = fnp_es_g
            .call1((&es_a, &es_b))
            .expect("fnp f16 einsum gram parity");
        let base_g = np_es_g
            .call1((&es_a, &es_b))
            .expect("numpy f16 einsum gram parity");
        assert_eq!(
            candidate_g
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_g
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum gram byte parity",
        );

        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_gram_512_null_then_effect",
            "f16_einsum_gram_512",
            &np_es_g,
            &fnp_es_g,
            &es_a,
            &es_b,
        );

        // Output-transposed variants: operand-swap arms of the transposed and
        // gram kernels ('ij,lj->li' / 'ji,jl->li'). Rows prove the swapped
        // dispatch engages the native route (effect >> 1, not numpy ~1.0x).
        for (bench_name, row, fnp_key, np_key) in [
            (
                "f16_einsum_transposed_swapped_512_null_then_effect",
                "f16_einsum_transposed_swapped_512",
                "fnp_es_ts",
                "np_es_ts",
            ),
            (
                "f16_einsum_gram_swapped_512_null_then_effect",
                "f16_einsum_gram_swapped_512",
                "fnp_es_gs",
                "np_es_gs",
            ),
        ] {
            let fnp_fn = namespace.get_item(fnp_key).expect("fnp swapped fn");
            let np_fn = namespace.get_item(np_key).expect("np swapped fn");
            let candidate = fnp_fn
                .call1((&es_a, &es_b))
                .expect("fnp swapped einsum parity");
            let base = np_fn
                .call1((&es_a, &es_b))
                .expect("numpy swapped einsum parity");
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "f16 einsum swapped-output byte parity ({row})",
            );
            bench_median_gate_python_binary(
                &mut group, bench_name, row, &np_fn, &fnp_fn, &es_a, &es_b,
            );
        }

        // 1-D dot ('j,j->') at 8M: per-8192-buffer trees in parallel, serial
        // f16 fold. Scalar output - parity assert via float16 byte equality.
        let dot_a = namespace.get_item("dot_a").expect("dot_a present");
        let dot_b = namespace.get_item("dot_b").expect("dot_b present");
        let fnp_es_d = namespace.get_item("fnp_es_d").expect("fnp_es_d present");
        let np_es_d = namespace.get_item("np_es_d").expect("np_es_d present");
        let candidate_d = fnp_es_d
            .call1((&dot_a, &dot_b))
            .expect("fnp f16 einsum dot parity");
        let base_d = np_es_d
            .call1((&dot_a, &dot_b))
            .expect("numpy f16 einsum dot parity");
        assert_eq!(
            candidate_d
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_d
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum 1-D dot byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_dot1d_8m_null_then_effect",
            "f16_einsum_dot1d_8m",
            &np_es_d,
            &fnp_es_d,
            &dot_a,
            &dot_b,
        );

        // 2-D full contraction ('ij,ij->') at 2896^2 ~ 8.4M: the coalesced
        // chunk-fold route through the generalized full-contraction parser.
        let fc_a = namespace.get_item("fc_a").expect("fc_a present");
        let fc_b = namespace.get_item("fc_b").expect("fc_b present");
        let fnp_es_fc = namespace.get_item("fnp_es_fc").expect("fnp_es_fc present");
        let np_es_fc = namespace.get_item("np_es_fc").expect("np_es_fc present");
        let candidate_fc = fnp_es_fc
            .call1((&fc_a, &fc_b))
            .expect("fnp f16 einsum full-contraction parity");
        let base_fc = np_es_fc
            .call1((&fc_a, &fc_b))
            .expect("numpy f16 einsum full-contraction parity");
        assert_eq!(
            candidate_fc
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_fc
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum full-contraction byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_fullc_2d_8m_null_then_effect",
            "f16_einsum_fullc_2d_8m",
            &np_es_fc,
            &fnp_es_fc,
            &fc_a,
            &fc_b,
        );

        // Elementwise product ('j,j->j') at 8M: zero-seeded parallel flat map.
        let fnp_es_ew = namespace.get_item("fnp_es_ew").expect("fnp_es_ew present");
        let np_es_ew = namespace.get_item("np_es_ew").expect("np_es_ew present");
        let candidate_ew = fnp_es_ew
            .call1((&dot_a, &dot_b))
            .expect("fnp f16 einsum elementwise parity");
        let base_ew = np_es_ew
            .call1((&dot_a, &dot_b))
            .expect("numpy f16 einsum elementwise parity");
        assert_eq!(
            candidate_ew
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_ew
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum elementwise byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_elemwise_8m_null_then_effect",
            "f16_einsum_elemwise_8m",
            &np_es_ew,
            &fnp_es_ew,
            &dot_a,
            &dot_b,
        );

        // f64 elementwise ('j,j->j') at 8M: the f64/f32 zero-seeded kernel.
        let ew64_a = namespace.get_item("ew64_a").expect("ew64_a present");
        let ew64_b = namespace.get_item("ew64_b").expect("ew64_b present");
        let candidate_e64 = fnp_es_ew
            .call1((&ew64_a, &ew64_b))
            .expect("fnp f64 einsum elementwise parity");
        let base_e64 = np_es_ew
            .call1((&ew64_a, &ew64_b))
            .expect("numpy f64 einsum elementwise parity");
        assert_eq!(
            candidate_e64
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_e64
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f64 einsum elementwise byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f64_einsum_elemwise_8m_null_then_effect",
            "f64_einsum_elemwise_8m",
            &np_es_ew,
            &fnp_es_ew,
            &ew64_a,
            &ew64_b,
        );

        // f64 broadcast form ('ij,j->ij') at 2896^2: the broadcast kernel.
        let bc64_full = namespace.get_item("bc64_full").expect("bc64_full present");
        let bc64_vec = namespace.get_item("bc64_vec").expect("bc64_vec present");
        let fnp_es_bc = namespace.get_item("fnp_es_bc").expect("fnp_es_bc present");
        let np_es_bc = namespace.get_item("np_es_bc").expect("np_es_bc present");
        let candidate_bc = fnp_es_bc
            .call1((&bc64_full, &bc64_vec))
            .expect("fnp f64 einsum broadcast parity");
        let base_bc = np_es_bc
            .call1((&bc64_full, &bc64_vec))
            .expect("numpy f64 einsum broadcast parity");
        assert_eq!(
            candidate_bc
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_bc
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f64 einsum broadcast byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f64_einsum_bcast_8m_null_then_effect",
            "f64_einsum_bcast_8m",
            &np_es_bc,
            &fnp_es_bc,
            &bc64_full,
            &bc64_vec,
        );

        // f16 3-op chain 512^3 with optimize=True: plan + shipped matmul
        // kernel per pair (c operand captured in the lambda closures).
        let ch_a = namespace.get_item("ch_a").expect("ch_a present");
        let ch_b = namespace.get_item("ch_b").expect("ch_b present");
        let fnp_es_ch = namespace.get_item("fnp_es_ch").expect("fnp_es_ch present");
        let np_es_ch = namespace.get_item("np_es_ch").expect("np_es_ch present");
        let candidate_ch = fnp_es_ch
            .call1((&ch_a, &ch_b))
            .expect("fnp f16 chain parity");
        let base_ch = np_es_ch
            .call1((&ch_a, &ch_b))
            .expect("numpy f16 chain parity");
        assert_eq!(
            candidate_ch
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_ch
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum chain3 byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_chain3_512_null_then_effect",
            "f16_einsum_chain3_512",
            &np_es_ch,
            &fnp_es_ch,
            &ch_a,
            &ch_b,
        );

        // f16 reduction specs at 2896^2: col-sum (the 27.9ms strided rank)
        // and row-sum.
        let red16 = namespace.get_item("red16").expect("red16 present");
        let red64 = namespace.get_item("red64").expect("red64 present");
        let red32 = namespace.get_item("red32").expect("red32 present");
        for (bench_name, row, fnp_key, np_key, input) in [
            (
                "f16_einsum_colsum_8m_null_then_effect",
                "f16_einsum_colsum_8m",
                "fnp_es_rj",
                "np_es_rj",
                &red16,
            ),
            (
                "f16_einsum_rowsum_8m_null_then_effect",
                "f16_einsum_rowsum_8m",
                "fnp_es_ri",
                "np_es_ri",
                &red16,
            ),
            (
                "f64_einsum_colsum_8m_null_then_effect",
                "f64_einsum_colsum_8m",
                "fnp_es_rj",
                "np_es_rj",
                &red64,
            ),
            (
                "f64_einsum_rowsum_8m_null_then_effect",
                "f64_einsum_rowsum_8m",
                "fnp_es_ri",
                "np_es_ri",
                &red64,
            ),
            (
                "f32_einsum_colsum_8m_null_then_effect",
                "f32_einsum_colsum_8m",
                "fnp_es_rj",
                "np_es_rj",
                &red32,
            ),
            (
                "f32_einsum_rowsum_8m_null_then_effect",
                "f32_einsum_rowsum_8m",
                "fnp_es_ri",
                "np_es_ri",
                &red32,
            ),
        ] {
            let fnp_fn = namespace.get_item(fnp_key).expect("fnp reduce fn");
            let np_fn = namespace.get_item(np_key).expect("np reduce fn");
            let candidate = fnp_fn.call1((input,)).expect("fnp reduce parity");
            let base = np_fn.call1((input,)).expect("numpy reduce parity");
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "f16 einsum reduction byte parity ({row})",
            );
            bench_median_gate_python_unary(&mut group, bench_name, row, &np_fn, &fnp_fn, input);
        }

        // Batched matmul spec ('bij,bjk->bik') at (8,256,256)@(8,256,256):
        // the plain per-step chain per batch, parallel across batches.
        let bat_a = namespace.get_item("bat_a").expect("bat_a present");
        let bat_b = namespace.get_item("bat_b").expect("bat_b present");
        let fnp_es_b = namespace.get_item("fnp_es_b").expect("fnp_es_b present");
        let np_es_b = namespace.get_item("np_es_b").expect("np_es_b present");
        let candidate_b = fnp_es_b
            .call1((&bat_a, &bat_b))
            .expect("fnp f16 einsum batched parity");
        let base_b = np_es_b
            .call1((&bat_a, &bat_b))
            .expect("numpy f16 einsum batched parity");
        assert_eq!(
            candidate_b
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_b
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum batched byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_batched_8x256_null_then_effect",
            "f16_einsum_batched_8x256",
            &np_es_b,
            &fnp_es_b,
            &bat_a,
            &bat_b,
        );

        // Batched transposed spec ('bij,blj->bil'): buffered chunk-fold wide
        // trees per element, parallel across batches + row blocks.
        let fnp_es_bt = namespace.get_item("fnp_es_bt").expect("fnp_es_bt present");
        let np_es_bt = namespace.get_item("np_es_bt").expect("np_es_bt present");
        let candidate_bt = fnp_es_bt
            .call1((&bat_a, &bat_b))
            .expect("fnp f16 einsum batched-t parity");
        let base_bt = np_es_bt
            .call1((&bat_a, &bat_b))
            .expect("numpy f16 einsum batched-t parity");
        assert_eq!(
            candidate_bt
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_bt
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum batched transposed byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_batched_t_8x256_null_then_effect",
            "f16_einsum_batched_t_8x256",
            &np_es_bt,
            &fnp_es_bt,
            &bat_a,
            &bat_b,
        );

        // Batched gram spec ('bji,bjl->bil'): per-step muladd rows per batch
        // (chunk-immune per-step class). Same (8,256,256) operands.
        let fnp_es_bg = namespace.get_item("fnp_es_bg").expect("fnp_es_bg present");
        let np_es_bg = namespace.get_item("np_es_bg").expect("np_es_bg present");
        let candidate_bg = fnp_es_bg
            .call1((&bat_a, &bat_b))
            .expect("fnp f16 einsum batched-gram parity");
        let base_bg = np_es_bg
            .call1((&bat_a, &bat_b))
            .expect("numpy f16 einsum batched-gram parity");
        assert_eq!(
            candidate_bg
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_bg
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum batched gram byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_batched_g_8x256_null_then_effect",
            "f16_einsum_batched_g_8x256",
            &np_es_bg,
            &fnp_es_bg,
            &bat_a,
            &bat_b,
        );

        // Output-swapped batched forms: operand-swap arms of the batched
        // transposed/gram kernels. Rows prove the swapped dispatch engages.
        for (bench_name, row, fnp_key, np_key) in [
            (
                "f16_einsum_batched_ts_8x256_null_then_effect",
                "f16_einsum_batched_ts_8x256",
                "fnp_es_bts",
                "np_es_bts",
            ),
            (
                "f16_einsum_batched_gs_8x256_null_then_effect",
                "f16_einsum_batched_gs_8x256",
                "fnp_es_bgs",
                "np_es_bgs",
            ),
        ] {
            let fnp_fn = namespace.get_item(fnp_key).expect("fnp batched-swap fn");
            let np_fn = namespace.get_item(np_key).expect("np batched-swap fn");
            let candidate = fnp_fn
                .call1((&bat_a, &bat_b))
                .expect("fnp batched-swap parity");
            let base = np_fn
                .call1((&bat_a, &bat_b))
                .expect("numpy batched-swap parity");
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "f16 einsum batched swapped byte parity ({row})",
            );
            bench_median_gate_python_binary(
                &mut group, bench_name, row, &np_fn, &fnp_fn, &bat_a, &bat_b,
            );
        }
    });

    group.finish();
}

fn bench_wide_string_substrate_v2(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_wide_string_substrate_v2");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_wide_string_v2").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(611)\n\
                 u_a = rng.integers(97, 123, (1_000_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
                 u_fresh = rng.integers(97, 123, (500_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
                 u_b = np.concatenate([u_a[:500_000], u_fresh])\n\
                 u_union_b = rng.integers(97, 123, (1_000_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
                 s_a = rng.integers(0, 256, (1_000_000, 16), dtype=np.uint8).view('S16').reshape(-1)\n\
                 s_fresh = rng.integers(0, 256, (500_000, 16), dtype=np.uint8).view('S16').reshape(-1)\n\
                 s_b = np.concatenate([s_a[:500_000], s_fresh])\n",
            )
            .expect("wide string setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("wide string setup");
        let u_a = namespace.get_item("u_a").expect("u_a present");
        let u_b = namespace.get_item("u_b").expect("u_b present");
        let u_union_b = namespace.get_item("u_union_b").expect("u_union_b present");
        let s_a = namespace.get_item("s_a").expect("s_a present");
        let s_b = namespace.get_item("s_b").expect("s_b present");
        let array_equal = numpy.getattr("array_equal").expect("numpy.array_equal");

        for (lhs, rhs) in [(&u_a, &u_b), (&s_a, &s_b)] {
            for op in ["unique", "union1d", "intersect1d", "setxor1d"] {
                let candidate_fn = module.getattr(op).expect("fnp op");
                let orig_fn = numpy.getattr(op).expect("numpy op");
                let candidate = if op == "unique" {
                    candidate_fn.call1((lhs,)).expect("fnp parity call")
                } else {
                    candidate_fn.call1((lhs, rhs)).expect("fnp parity call")
                };
                let orig = if op == "unique" {
                    orig_fn.call1((lhs,)).expect("numpy parity call")
                } else {
                    orig_fn.call1((lhs, rhs)).expect("numpy parity call")
                };
                assert!(
                    array_equal
                        .call1((&candidate, &orig))
                        .expect("array_equal")
                        .extract::<bool>()
                        .expect("array_equal bool"),
                    "wide string {op} parity",
                );
            }
        }
        let fnp_union_parity = module
            .getattr("union1d")
            .expect("fnp union1d parity function")
            .call1((&u_a, &u_union_b))
            .expect("fnp union1d parity call");
        let numpy_union_parity = numpy
            .getattr("union1d")
            .expect("numpy union1d parity function")
            .call1((&u_a, &u_union_b))
            .expect("numpy union1d parity call");
        assert!(
            array_equal
                .call1((&fnp_union_parity, &numpy_union_parity))
                .expect("union array_equal")
                .extract::<bool>()
                .expect("union array_equal bool"),
            "wide string disjoint union parity",
        );

        let fnp_unique = module.getattr("unique").expect("fnp unique");
        let np_unique = numpy.getattr("unique").expect("numpy unique");
        let fnp_union = module.getattr("union1d").expect("fnp union1d");
        let np_union = numpy.getattr("union1d").expect("numpy union1d");
        let fnp_intersect = module.getattr("intersect1d").expect("fnp intersect1d");
        let np_intersect = numpy.getattr("intersect1d").expect("numpy intersect1d");
        let fnp_setxor = module.getattr("setxor1d").expect("fnp setxor1d");
        let np_setxor = numpy.getattr("setxor1d").expect("numpy setxor1d");

        bench_substrate_v2_python_unary_pair(
            &mut group,
            "u16_unique_1m_paired",
            "u16_unique_1m",
            &fnp_unique,
            &np_unique,
            &u_a,
        );
        bench_substrate_v2_python_binary_pair(
            &mut group,
            "u16_union_disjoint_1m_paired",
            "u16_union_disjoint_1m",
            &fnp_union,
            &np_union,
            &u_a,
            &u_union_b,
        );
        bench_substrate_v2_python_binary_pair(
            &mut group,
            "u16_setxor_1m_paired",
            "u16_setxor_1m",
            &fnp_setxor,
            &np_setxor,
            &u_a,
            &u_b,
        );
        bench_substrate_v2_python_unary_pair(
            &mut group,
            "s16_unique_1m_paired",
            "s16_unique_1m",
            &fnp_unique,
            &np_unique,
            &s_a,
        );
        bench_substrate_v2_python_binary_pair(
            &mut group,
            "s16_intersect_1m_paired",
            "s16_intersect_1m",
            &fnp_intersect,
            &np_intersect,
            &s_a,
            &s_b,
        );
        bench_substrate_v2_python_binary_pair(
            &mut group,
            "s16_setxor_1m_paired",
            "s16_setxor_1m",
            &fnp_setxor,
            &np_setxor,
            &s_a,
            &s_b,
        );
    });

    group.finish();
}

fn bench_ledger_integrity_rejects(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_ledger_integrity_rejects");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(6));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_ledger_audit").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");

        {
            let namespace = PyDict::new(py);
            py.run(
                std::ffi::CString::new(
                    "import numpy as np\n\
                     rng = np.random.default_rng(0)\n\
                     median_input = rng.standard_normal(16_000_000).astype(np.float64)\n",
                )
                .expect("median setup CString")
                .as_c_str(),
                Some(&namespace),
                Some(&namespace),
            )
            .expect("median setup");
            let input = namespace
                .get_item("median_input")
                .expect("median input present");
            let raw: Vec<u8> = input
                .call_method0("tobytes")
                .expect("median bytes")
                .extract()
                .expect("extract median bytes");
            let data: Vec<f64> = raw
                .chunks_exact(8)
                .map(|chunk| {
                    f64::from_ne_bytes(chunk.try_into().expect("one native f64 per chunk"))
                })
                .collect();
            assert_eq!(data.len(), 16_000_000);
            let numpy_median = numpy.getattr("median").expect("numpy.median");
            let candidate = ledger_radix_median_f64(&data);
            let orig = ledger_orig_median_reference(&numpy_median, &input)
                .expect("NumPy median parity reference");
            assert_eq!(candidate.to_bits(), orig.to_bits(), "radix median parity");

            let candidate_samples = RefCell::new(Vec::new());
            let orig_samples = RefCell::new(Vec::new());
            let order = Cell::new(0u64);
            group.bench_function("radix_median_f64_normal_16m_paired", |bench| {
                bench.iter_custom(|iterations| {
                    let mut candidate_total = Duration::ZERO;
                    let mut orig_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let orig_first = order.get() & 1 == 1;
                        order.set(order.get().wrapping_add(1));
                        if orig_first {
                            let start = Instant::now();
                            black_box(
                                ledger_orig_median_reference(&numpy_median, &input)
                                    .expect("NumPy median audit call"),
                            );
                            orig_total += start.elapsed();
                            let start = Instant::now();
                            black_box(ledger_radix_median_f64(&data));
                            candidate_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(ledger_radix_median_f64(&data));
                            candidate_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                ledger_orig_median_reference(&numpy_median, &input)
                                    .expect("NumPy median audit call"),
                            );
                            orig_total += start.elapsed();
                        }
                    }
                    candidate_samples
                        .borrow_mut()
                        .push(candidate_total.as_secs_f64() * 1e9 / iterations as f64);
                    orig_samples
                        .borrow_mut()
                        .push(orig_total.as_secs_f64() * 1e9 / iterations as f64);
                    candidate_total + orig_total
                });
            });
            report_ledger_pair(
                "radix_median_f64_normal_16m",
                &candidate_samples,
                &orig_samples,
            );
        }

        {
            let namespace = PyDict::new(py);
            py.run(
                std::ffi::CString::new(
                    "import numpy as np\n\
                     rng = np.random.default_rng(0)\n\
                     f16_input = (rng.integers(1, 4000, 4_000_000) / 7).astype(np.float16)\n",
                )
                .expect("f16 setup CString")
                .as_c_str(),
                Some(&namespace),
                Some(&namespace),
            )
            .expect("f16 setup");
            let input = namespace.get_item("f16_input").expect("f16 input present");
            let bit_bytes: Vec<u8> = input
                .call_method1("view", ("uint16",))
                .expect("f16 uint16 view")
                .call_method0("tobytes")
                .expect("f16 bit bytes")
                .extract()
                .expect("extract f16 bit bytes");
            let input_bits: Vec<u16> = bit_bytes
                .chunks_exact(2)
                .map(|chunk| {
                    u16::from_ne_bytes(chunk.try_into().expect("one native u16 per chunk"))
                })
                .collect();
            assert_eq!(input_bits.len(), 4_000_000);
            let numpy_sort = numpy.getattr("sort").expect("numpy.sort");
            let equal = numpy.getattr("array_equal").expect("numpy.array_equal");
            let candidate = ledger_try_native_f16_sort(&numpy_sort, &input, &input_bits)
                .expect("f16 widening candidate parity call");
            let orig =
                ledger_orig_f16_sort_reference(&numpy_sort, &input).expect("f16 ORIG parity call");
            assert!(
                equal
                    .call1((candidate.bind(py), orig.bind(py)))
                    .expect("f16 array_equal")
                    .extract::<bool>()
                    .expect("f16 equality bool"),
                "f16 widening-sort parity",
            );

            let candidate_samples = RefCell::new(Vec::new());
            let orig_samples = RefCell::new(Vec::new());
            let order = Cell::new(0u64);
            group.bench_function("f16_sort_via_f32_widening_4m_paired", |bench| {
                bench.iter_custom(|iterations| {
                    let mut candidate_total = Duration::ZERO;
                    let mut orig_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let orig_first = order.get() & 1 == 1;
                        order.set(order.get().wrapping_add(1));
                        if orig_first {
                            let start = Instant::now();
                            black_box(
                                ledger_orig_f16_sort_reference(&numpy_sort, &input)
                                    .expect("f16 ORIG audit call"),
                            );
                            orig_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                ledger_try_native_f16_sort(&numpy_sort, &input, &input_bits)
                                    .expect("f16 widening audit call"),
                            );
                            candidate_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(
                                ledger_try_native_f16_sort(&numpy_sort, &input, &input_bits)
                                    .expect("f16 widening audit call"),
                            );
                            candidate_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                ledger_orig_f16_sort_reference(&numpy_sort, &input)
                                    .expect("f16 ORIG audit call"),
                            );
                            orig_total += start.elapsed();
                        }
                    }
                    candidate_samples
                        .borrow_mut()
                        .push(candidate_total.as_secs_f64() * 1e9 / iterations as f64);
                    orig_samples
                        .borrow_mut()
                        .push(orig_total.as_secs_f64() * 1e9 / iterations as f64);
                    candidate_total + orig_total
                });
            });
            report_ledger_pair(
                "f16_sort_via_f32_widening_4m",
                &candidate_samples,
                &orig_samples,
            );

            // PRODUCTION arm (bead deadlock-audit-98chw): fnp.sort(f16) now routes through
            // try_native_f16_sort_flat for this input; paired vs numpy.sort in the same
            // interleaved routine, plus an A/A null-control row (per-function noise floor).
            let fnp_sort = module.getattr("sort").expect("fnp sort");
            let prod = fnp_sort.call1((&input,)).expect("fnp f16 sort parity call");
            let prod_bytes: Vec<u8> = prod
                .call_method0("tobytes")
                .expect("prod bytes")
                .extract()
                .expect("extract prod bytes");
            let orig_bytes: Vec<u8> = orig
                .bind(py)
                .call_method0("tobytes")
                .expect("orig bytes")
                .extract()
                .expect("extract orig bytes");
            assert_eq!(
                prod_bytes, orig_bytes,
                "production f16 sort parity (tobytes)"
            );

            let prod_samples = RefCell::new(Vec::new());
            let prod_orig_samples = RefCell::new(Vec::new());
            let prod_order = Cell::new(0u64);
            group.bench_function("f16_sort_production_4m_paired", |bench| {
                bench.iter_custom(|iterations| {
                    let mut cand_total = Duration::ZERO;
                    let mut orig_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let orig_first = prod_order.get() & 1 == 1;
                        prod_order.set(prod_order.get().wrapping_add(1));
                        if orig_first {
                            let start = Instant::now();
                            black_box(numpy_sort.call1((&input,)).expect("numpy f16 sort"));
                            orig_total += start.elapsed();
                            let start = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("fnp f16 sort"));
                            cand_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("fnp f16 sort"));
                            cand_total += start.elapsed();
                            let start = Instant::now();
                            black_box(numpy_sort.call1((&input,)).expect("numpy f16 sort"));
                            orig_total += start.elapsed();
                        }
                    }
                    prod_samples
                        .borrow_mut()
                        .push(cand_total.as_secs_f64() * 1e9 / iterations as f64);
                    prod_orig_samples
                        .borrow_mut()
                        .push(orig_total.as_secs_f64() * 1e9 / iterations as f64);
                    cand_total + orig_total
                });
            });
            report_ledger_pair("f16_sort_production_4m", &prod_samples, &prod_orig_samples);

            let null_a = RefCell::new(Vec::new());
            let null_b = RefCell::new(Vec::new());
            let null_order = Cell::new(0u64);
            group.bench_function("f16_sort_production_4m_null_control", |bench| {
                bench.iter_custom(|iterations| {
                    let mut a_total = Duration::ZERO;
                    let mut b_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let b_first = null_order.get() & 1 == 1;
                        null_order.set(null_order.get().wrapping_add(1));
                        if b_first {
                            let start = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("null b"));
                            b_total += start.elapsed();
                            let start = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("null a"));
                            a_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("null a"));
                            a_total += start.elapsed();
                            let start = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("null b"));
                            b_total += start.elapsed();
                        }
                    }
                    null_a
                        .borrow_mut()
                        .push(a_total.as_secs_f64() * 1e9 / iterations as f64);
                    null_b
                        .borrow_mut()
                        .push(b_total.as_secs_f64() * 1e9 / iterations as f64);
                    a_total + b_total
                });
            });
            report_ledger_pair("f16_sort_production_null_AA", &null_a, &null_b);

            // f16 STABLE ARGSORT production arm (widened stable radix; sibling lever): the
            // same 4M f16 input is tie-dense by construction (63k distinct values), the case
            // where stability is load-bearing. Paired vs numpy + A/A null control.
            let fnp_argsort = module.getattr("argsort").expect("fnp argsort");
            let numpy_argsort = numpy.getattr("argsort").expect("numpy argsort");
            let stable_kw = pyo3::types::PyDict::new(py);
            stable_kw.set_item("kind", "stable").expect("kind kwarg");
            let ag_prod = fnp_argsort
                .call((&input,), Some(&stable_kw))
                .expect("fnp f16 stable argsort parity call");
            let ag_orig = numpy_argsort
                .call((&input,), Some(&stable_kw))
                .expect("numpy f16 stable argsort parity call");
            assert!(
                equal
                    .call1((&ag_prod, &ag_orig))
                    .expect("f16 argsort array_equal")
                    .extract::<bool>()
                    .expect("f16 argsort equality bool"),
                "production f16 stable argsort parity",
            );

            let ag_samples = RefCell::new(Vec::new());
            let ag_orig_samples = RefCell::new(Vec::new());
            let ag_order = Cell::new(0u64);
            group.bench_function("f16_argsort_stable_production_4m_paired", |bench| {
                bench.iter_custom(|iterations| {
                    let mut cand_total = Duration::ZERO;
                    let mut orig_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let orig_first = ag_order.get() & 1 == 1;
                        ag_order.set(ag_order.get().wrapping_add(1));
                        if orig_first {
                            let start = Instant::now();
                            black_box(
                                numpy_argsort
                                    .call((&input,), Some(&stable_kw))
                                    .expect("numpy f16 argsort"),
                            );
                            orig_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                fnp_argsort
                                    .call((&input,), Some(&stable_kw))
                                    .expect("fnp f16 argsort"),
                            );
                            cand_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(
                                fnp_argsort
                                    .call((&input,), Some(&stable_kw))
                                    .expect("fnp f16 argsort"),
                            );
                            cand_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                numpy_argsort
                                    .call((&input,), Some(&stable_kw))
                                    .expect("numpy f16 argsort"),
                            );
                            orig_total += start.elapsed();
                        }
                    }
                    ag_samples
                        .borrow_mut()
                        .push(cand_total.as_secs_f64() * 1e9 / iterations as f64);
                    ag_orig_samples
                        .borrow_mut()
                        .push(orig_total.as_secs_f64() * 1e9 / iterations as f64);
                    cand_total + orig_total
                });
            });
            report_ledger_pair(
                "f16_argsort_stable_production_4m",
                &ag_samples,
                &ag_orig_samples,
            );

            let ag_null_a = RefCell::new(Vec::new());
            let ag_null_b = RefCell::new(Vec::new());
            let ag_null_order = Cell::new(0u64);
            group.bench_function("f16_argsort_stable_production_4m_null_control", |bench| {
                bench.iter_custom(|iterations| {
                    let mut a_total = Duration::ZERO;
                    let mut b_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let b_first = ag_null_order.get() & 1 == 1;
                        ag_null_order.set(ag_null_order.get().wrapping_add(1));
                        if b_first {
                            let start = Instant::now();
                            black_box(
                                fnp_argsort
                                    .call((&input,), Some(&stable_kw))
                                    .expect("null b"),
                            );
                            b_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                fnp_argsort
                                    .call((&input,), Some(&stable_kw))
                                    .expect("null a"),
                            );
                            a_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(
                                fnp_argsort
                                    .call((&input,), Some(&stable_kw))
                                    .expect("null a"),
                            );
                            a_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                fnp_argsort
                                    .call((&input,), Some(&stable_kw))
                                    .expect("null b"),
                            );
                            b_total += start.elapsed();
                        }
                    }
                    ag_null_a
                        .borrow_mut()
                        .push(a_total.as_secs_f64() * 1e9 / iterations as f64);
                    ag_null_b
                        .borrow_mut()
                        .push(b_total.as_secs_f64() * 1e9 / iterations as f64);
                    a_total + b_total
                });
            });
            report_ledger_pair("f16_argsort_stable_null_AA", &ag_null_a, &ag_null_b);

            // LAST-AXIS siblings (2000x2000 view of the same 4M input): per-lane widened
            // value sort + per-lane widened stable argsort, paired with A/A null controls.
            let input2d = input
                .call_method1("reshape", ((2000, 2000),))
                .expect("reshape 2000x2000");
            let axis_kw = pyo3::types::PyDict::new(py);
            axis_kw.set_item("axis", -1).expect("axis kwarg");
            let fnp_sort2 = module.getattr("sort").expect("fnp sort");
            let numpy_sort2 = numpy.getattr("sort").expect("numpy sort");
            let s2_f = fnp_sort2
                .call((&input2d,), Some(&axis_kw))
                .expect("fnp f16 lastaxis sort parity");
            let s2_n = numpy_sort2
                .call((&input2d,), Some(&axis_kw))
                .expect("numpy f16 lastaxis sort parity");
            let s2_fb: Vec<u8> = s2_f
                .call_method0("tobytes")
                .expect("bytes")
                .extract()
                .expect("extract");
            let s2_nb: Vec<u8> = s2_n
                .call_method0("tobytes")
                .expect("bytes")
                .extract()
                .expect("extract");
            assert_eq!(s2_fb, s2_nb, "f16 lastaxis sort parity (tobytes)");
            let stable_axis_kw = pyo3::types::PyDict::new(py);
            stable_axis_kw.set_item("axis", -1).expect("axis kwarg");
            stable_axis_kw
                .set_item("kind", "stable")
                .expect("kind kwarg");
            let a2_f = fnp_argsort
                .call((&input2d,), Some(&stable_axis_kw))
                .expect("fnp f16 lastaxis argsort parity");
            let a2_n = numpy_argsort
                .call((&input2d,), Some(&stable_axis_kw))
                .expect("numpy f16 lastaxis argsort parity");
            assert!(
                equal
                    .call1((&a2_f, &a2_n))
                    .expect("array_equal")
                    .extract::<bool>()
                    .expect("bool"),
                "f16 lastaxis stable argsort parity",
            );

            for (label, fnp_fn, numpy_fn, kw) in [
                (
                    "f16_sort_lastaxis_2000x2000",
                    &fnp_sort2,
                    &numpy_sort2,
                    &axis_kw,
                ),
                (
                    "f16_argsort_stable_lastaxis_2000x2000",
                    &fnp_argsort,
                    &numpy_argsort,
                    &stable_axis_kw,
                ),
            ] {
                let cand = RefCell::new(Vec::new());
                let orig = RefCell::new(Vec::new());
                let ord = Cell::new(0u64);
                group.bench_function(format!("{label}_paired"), |bench| {
                    bench.iter_custom(|iterations| {
                        let mut ct = Duration::ZERO;
                        let mut ot = Duration::ZERO;
                        for _ in 0..iterations {
                            let of = ord.get() & 1 == 1;
                            ord.set(ord.get().wrapping_add(1));
                            if of {
                                let s = Instant::now();
                                black_box(numpy_fn.call((&input2d,), Some(kw)).expect("orig"));
                                ot += s.elapsed();
                                let s = Instant::now();
                                black_box(fnp_fn.call((&input2d,), Some(kw)).expect("cand"));
                                ct += s.elapsed();
                            } else {
                                let s = Instant::now();
                                black_box(fnp_fn.call((&input2d,), Some(kw)).expect("cand"));
                                ct += s.elapsed();
                                let s = Instant::now();
                                black_box(numpy_fn.call((&input2d,), Some(kw)).expect("orig"));
                                ot += s.elapsed();
                            }
                        }
                        cand.borrow_mut()
                            .push(ct.as_secs_f64() * 1e9 / iterations as f64);
                        orig.borrow_mut()
                            .push(ot.as_secs_f64() * 1e9 / iterations as f64);
                        ct + ot
                    });
                });
                report_ledger_pair(label, &cand, &orig);

                let na = RefCell::new(Vec::new());
                let nb = RefCell::new(Vec::new());
                let nord = Cell::new(0u64);
                group.bench_function(format!("{label}_null_control"), |bench| {
                    bench.iter_custom(|iterations| {
                        let mut at = Duration::ZERO;
                        let mut bt = Duration::ZERO;
                        for _ in 0..iterations {
                            let bf = nord.get() & 1 == 1;
                            nord.set(nord.get().wrapping_add(1));
                            if bf {
                                let s = Instant::now();
                                black_box(fnp_fn.call((&input2d,), Some(kw)).expect("nb"));
                                bt += s.elapsed();
                                let s = Instant::now();
                                black_box(fnp_fn.call((&input2d,), Some(kw)).expect("na"));
                                at += s.elapsed();
                            } else {
                                let s = Instant::now();
                                black_box(fnp_fn.call((&input2d,), Some(kw)).expect("na"));
                                at += s.elapsed();
                                let s = Instant::now();
                                black_box(fnp_fn.call((&input2d,), Some(kw)).expect("nb"));
                                bt += s.elapsed();
                            }
                        }
                        na.borrow_mut()
                            .push(at.as_secs_f64() * 1e9 / iterations as f64);
                        nb.borrow_mut()
                            .push(bt.as_secs_f64() * 1e9 / iterations as f64);
                        at + bt
                    });
                });
                report_ledger_pair(&format!("{label}_null_AA"), &na, &nb);
            }
        }

        {
            // Narrow-int counting sort: i16 8M full-range (the 2-byte case is where numpy's
            // serial radix is slowest). Paired vs numpy + A/A null control.
            let namespace = PyDict::new(py);
            py.run(
                std::ffi::CString::new(
                    "import numpy as np\n\
                     rng = np.random.default_rng(5)\n\
                     i16_input = rng.integers(-32768, 32768, 8_000_000, dtype=np.int16)\n",
                )
                .expect("i16 setup CString")
                .as_c_str(),
                Some(&namespace),
                Some(&namespace),
            )
            .expect("i16 setup");
            let input = namespace.get_item("i16_input").expect("i16 input");
            let fnp_sort = module.getattr("sort").expect("fnp sort");
            let numpy_sort = numpy.getattr("sort").expect("numpy sort");
            let equal = numpy.getattr("array_equal").expect("array_equal");
            let f = fnp_sort.call1((&input,)).expect("fnp i16 sort parity");
            let nres = numpy_sort.call1((&input,)).expect("numpy i16 sort parity");
            assert!(
                equal
                    .call1((&f, &nres))
                    .expect("array_equal")
                    .extract::<bool>()
                    .expect("bool"),
                "narrow-int i16 sort parity",
            );

            let cand = RefCell::new(Vec::new());
            let orig = RefCell::new(Vec::new());
            let ord = Cell::new(0u64);
            group.bench_function("narrow_int_i16_sort_8m_paired", |bench| {
                bench.iter_custom(|iterations| {
                    let mut ct = Duration::ZERO;
                    let mut ot = Duration::ZERO;
                    for _ in 0..iterations {
                        let of = ord.get() & 1 == 1;
                        ord.set(ord.get().wrapping_add(1));
                        if of {
                            let s = Instant::now();
                            black_box(numpy_sort.call1((&input,)).expect("orig"));
                            ot += s.elapsed();
                            let s = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("cand"));
                            ct += s.elapsed();
                        } else {
                            let s = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("cand"));
                            ct += s.elapsed();
                            let s = Instant::now();
                            black_box(numpy_sort.call1((&input,)).expect("orig"));
                            ot += s.elapsed();
                        }
                    }
                    cand.borrow_mut()
                        .push(ct.as_secs_f64() * 1e9 / iterations as f64);
                    orig.borrow_mut()
                        .push(ot.as_secs_f64() * 1e9 / iterations as f64);
                    ct + ot
                });
            });
            report_ledger_pair("narrow_int_i16_sort_8m", &cand, &orig);

            let na = RefCell::new(Vec::new());
            let nb = RefCell::new(Vec::new());
            let nord = Cell::new(0u64);
            group.bench_function("narrow_int_i16_sort_8m_null_control", |bench| {
                bench.iter_custom(|iterations| {
                    let mut at = Duration::ZERO;
                    let mut bt = Duration::ZERO;
                    for _ in 0..iterations {
                        let bf = nord.get() & 1 == 1;
                        nord.set(nord.get().wrapping_add(1));
                        if bf {
                            let s = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("nb"));
                            bt += s.elapsed();
                            let s = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("na"));
                            at += s.elapsed();
                        } else {
                            let s = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("na"));
                            at += s.elapsed();
                            let s = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("nb"));
                            bt += s.elapsed();
                        }
                    }
                    na.borrow_mut()
                        .push(at.as_secs_f64() * 1e9 / iterations as f64);
                    nb.borrow_mut()
                        .push(bt.as_secs_f64() * 1e9 / iterations as f64);
                    at + bt
                });
            });
            report_ledger_pair("narrow_int_i16_sort_null_AA", &na, &nb);

            // Stable ARGSORT sibling on the same 8M i16 input (dense ties by construction;
            // routes to the parallel counting-prefix stable argsort). Paired + A/A null.
            let fnp_argsort_n = module.getattr("argsort").expect("fnp argsort");
            let numpy_argsort_n = numpy.getattr("argsort").expect("numpy argsort");
            let skw = pyo3::types::PyDict::new(py);
            skw.set_item("kind", "stable").expect("kind kwarg");
            let af = fnp_argsort_n
                .call((&input,), Some(&skw))
                .expect("fnp i16 stable argsort parity");
            let an = numpy_argsort_n
                .call((&input,), Some(&skw))
                .expect("numpy i16 stable argsort parity");
            assert!(
                equal
                    .call1((&af, &an))
                    .expect("array_equal")
                    .extract::<bool>()
                    .expect("bool"),
                "narrow-int i16 stable argsort parity",
            );
            let cand2 = RefCell::new(Vec::new());
            let orig2 = RefCell::new(Vec::new());
            let ord2 = Cell::new(0u64);
            group.bench_function("narrow_int_i16_argsort_stable_8m_paired", |bench| {
                bench.iter_custom(|iterations| {
                    let mut ct = Duration::ZERO;
                    let mut ot = Duration::ZERO;
                    for _ in 0..iterations {
                        let of = ord2.get() & 1 == 1;
                        ord2.set(ord2.get().wrapping_add(1));
                        if of {
                            let s = Instant::now();
                            black_box(numpy_argsort_n.call((&input,), Some(&skw)).expect("orig"));
                            ot += s.elapsed();
                            let s = Instant::now();
                            black_box(fnp_argsort_n.call((&input,), Some(&skw)).expect("cand"));
                            ct += s.elapsed();
                        } else {
                            let s = Instant::now();
                            black_box(fnp_argsort_n.call((&input,), Some(&skw)).expect("cand"));
                            ct += s.elapsed();
                            let s = Instant::now();
                            black_box(numpy_argsort_n.call((&input,), Some(&skw)).expect("orig"));
                            ot += s.elapsed();
                        }
                    }
                    cand2
                        .borrow_mut()
                        .push(ct.as_secs_f64() * 1e9 / iterations as f64);
                    orig2
                        .borrow_mut()
                        .push(ot.as_secs_f64() * 1e9 / iterations as f64);
                    ct + ot
                });
            });
            report_ledger_pair("narrow_int_i16_argsort_stable_8m", &cand2, &orig2);

            let na2 = RefCell::new(Vec::new());
            let nb2 = RefCell::new(Vec::new());
            let nord2 = Cell::new(0u64);
            group.bench_function("narrow_int_i16_argsort_stable_8m_null_control", |bench| {
                bench.iter_custom(|iterations| {
                    let mut at = Duration::ZERO;
                    let mut bt = Duration::ZERO;
                    for _ in 0..iterations {
                        let bf = nord2.get() & 1 == 1;
                        nord2.set(nord2.get().wrapping_add(1));
                        if bf {
                            let s = Instant::now();
                            black_box(fnp_argsort_n.call((&input,), Some(&skw)).expect("nb"));
                            bt += s.elapsed();
                            let s = Instant::now();
                            black_box(fnp_argsort_n.call((&input,), Some(&skw)).expect("na"));
                            at += s.elapsed();
                        } else {
                            let s = Instant::now();
                            black_box(fnp_argsort_n.call((&input,), Some(&skw)).expect("na"));
                            at += s.elapsed();
                            let s = Instant::now();
                            black_box(fnp_argsort_n.call((&input,), Some(&skw)).expect("nb"));
                            bt += s.elapsed();
                        }
                    }
                    na2.borrow_mut()
                        .push(at.as_secs_f64() * 1e9 / iterations as f64);
                    nb2.borrow_mut()
                        .push(bt.as_secs_f64() * 1e9 / iterations as f64);
                    at + bt
                });
            });
            report_ledger_pair("narrow_int_i16_argsort_stable_null_AA", &na2, &nb2);

            // LAST-AXIS siblings on a 4000x2000 view of the same 8M i16 input: per-lane sort
            // + per-lane stable argsort, paired with A/A null controls.
            let input2d = input
                .call_method1("reshape", ((4000, 2000),))
                .expect("reshape 4000x2000");
            let axkw = pyo3::types::PyDict::new(py);
            axkw.set_item("axis", -1).expect("axis kwarg");
            let stax_kw = pyo3::types::PyDict::new(py);
            stax_kw.set_item("axis", -1).expect("axis kwarg");
            stax_kw.set_item("kind", "stable").expect("kind kwarg");
            let sf = fnp_sort
                .call((&input2d,), Some(&axkw))
                .expect("fnp lastaxis parity");
            let sn = numpy_sort
                .call((&input2d,), Some(&axkw))
                .expect("numpy lastaxis parity");
            let sfb: Vec<u8> = sf.call_method0("tobytes").expect("b").extract().expect("e");
            let snb: Vec<u8> = sn.call_method0("tobytes").expect("b").extract().expect("e");
            assert_eq!(sfb, snb, "narrow-int i16 lastaxis sort parity");
            let gf = fnp_argsort_n
                .call((&input2d,), Some(&stax_kw))
                .expect("fnp lastaxis argsort parity");
            let gn = numpy_argsort_n
                .call((&input2d,), Some(&stax_kw))
                .expect("numpy lastaxis argsort parity");
            assert!(
                equal
                    .call1((&gf, &gn))
                    .expect("array_equal")
                    .extract::<bool>()
                    .expect("bool"),
                "narrow-int i16 lastaxis stable argsort parity",
            );
            for (label, ffn, nfn, kw) in [
                (
                    "narrow_int_i16_sort_lastaxis_4000x2000",
                    &fnp_sort,
                    &numpy_sort,
                    &axkw,
                ),
                (
                    "narrow_int_i16_argsort_stable_lastaxis_4000x2000",
                    &fnp_argsort_n,
                    &numpy_argsort_n,
                    &stax_kw,
                ),
            ] {
                let cand = RefCell::new(Vec::new());
                let orig = RefCell::new(Vec::new());
                let ord = Cell::new(0u64);
                group.bench_function(format!("{label}_paired"), |bench| {
                    bench.iter_custom(|iterations| {
                        let mut ct = Duration::ZERO;
                        let mut ot = Duration::ZERO;
                        for _ in 0..iterations {
                            let of = ord.get() & 1 == 1;
                            ord.set(ord.get().wrapping_add(1));
                            if of {
                                let s = Instant::now();
                                black_box(nfn.call((&input2d,), Some(kw)).expect("orig"));
                                ot += s.elapsed();
                                let s = Instant::now();
                                black_box(ffn.call((&input2d,), Some(kw)).expect("cand"));
                                ct += s.elapsed();
                            } else {
                                let s = Instant::now();
                                black_box(ffn.call((&input2d,), Some(kw)).expect("cand"));
                                ct += s.elapsed();
                                let s = Instant::now();
                                black_box(nfn.call((&input2d,), Some(kw)).expect("orig"));
                                ot += s.elapsed();
                            }
                        }
                        cand.borrow_mut()
                            .push(ct.as_secs_f64() * 1e9 / iterations as f64);
                        orig.borrow_mut()
                            .push(ot.as_secs_f64() * 1e9 / iterations as f64);
                        ct + ot
                    });
                });
                report_ledger_pair(label, &cand, &orig);
                let na = RefCell::new(Vec::new());
                let nb = RefCell::new(Vec::new());
                let nord = Cell::new(0u64);
                group.bench_function(format!("{label}_null_control"), |bench| {
                    bench.iter_custom(|iterations| {
                        let mut at = Duration::ZERO;
                        let mut bt = Duration::ZERO;
                        for _ in 0..iterations {
                            let bf = nord.get() & 1 == 1;
                            nord.set(nord.get().wrapping_add(1));
                            if bf {
                                let s = Instant::now();
                                black_box(ffn.call((&input2d,), Some(kw)).expect("nb"));
                                bt += s.elapsed();
                                let s = Instant::now();
                                black_box(ffn.call((&input2d,), Some(kw)).expect("na"));
                                at += s.elapsed();
                            } else {
                                let s = Instant::now();
                                black_box(ffn.call((&input2d,), Some(kw)).expect("na"));
                                at += s.elapsed();
                                let s = Instant::now();
                                black_box(ffn.call((&input2d,), Some(kw)).expect("nb"));
                                bt += s.elapsed();
                            }
                        }
                        na.borrow_mut()
                            .push(at.as_secs_f64() * 1e9 / iterations as f64);
                        nb.borrow_mut()
                            .push(bt.as_secs_f64() * 1e9 / iterations as f64);
                        at + bt
                    });
                });
                report_ledger_pair(&format!("{label}_null_AA"), &na, &nb);
            }
        }

        {
            let namespace = PyDict::new(py);
            py.run(
                std::ffi::CString::new(
                    "import numpy as np\n\
                     rng = np.random.default_rng(0)\n\
                     f32_ties = np.round(rng.standard_normal(2_000_000), 2).astype(np.float32)\n",
                )
                .expect("f32 argsort setup CString")
                .as_c_str(),
                Some(&namespace),
                Some(&namespace),
            )
            .expect("f32 argsort setup");
            let input = namespace
                .get_item("f32_ties")
                .expect("f32 tie input present");
            let fnp_argsort = module.getattr("argsort").expect("fnp argsort");
            let numpy_argsort = numpy.getattr("argsort").expect("numpy.argsort");
            let equal = numpy.getattr("array_equal").expect("numpy.array_equal");
            let candidate = ledger_f32_tie_argsort_candidate(&fnp_argsort, &input)
                .expect("f32 tied candidate parity call");
            let orig = ledger_orig_f32_argsort_reference(&numpy_argsort, &input)
                .expect("f32 tied ORIG parity call");
            assert!(
                equal
                    .call1((candidate.bind(py), orig.bind(py)))
                    .expect("f32 argsort array_equal")
                    .extract::<bool>()
                    .expect("f32 argsort equality bool"),
                "tied f32 argsort parity",
            );

            let candidate_samples = RefCell::new(Vec::new());
            let orig_samples = RefCell::new(Vec::new());
            let order = Cell::new(0u64);
            group.bench_function("f32_argsort_rounded_ties_2m_paired", |bench| {
                bench.iter_custom(|iterations| {
                    let mut candidate_total = Duration::ZERO;
                    let mut orig_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let orig_first = order.get() & 1 == 1;
                        order.set(order.get().wrapping_add(1));
                        if orig_first {
                            let start = Instant::now();
                            black_box(
                                ledger_orig_f32_argsort_reference(&numpy_argsort, &input)
                                    .expect("f32 argsort ORIG audit call"),
                            );
                            orig_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                ledger_f32_tie_argsort_candidate(&fnp_argsort, &input)
                                    .expect("f32 argsort candidate audit call"),
                            );
                            candidate_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(
                                ledger_f32_tie_argsort_candidate(&fnp_argsort, &input)
                                    .expect("f32 argsort candidate audit call"),
                            );
                            candidate_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                ledger_orig_f32_argsort_reference(&numpy_argsort, &input)
                                    .expect("f32 argsort ORIG audit call"),
                            );
                            orig_total += start.elapsed();
                        }
                    }
                    candidate_samples
                        .borrow_mut()
                        .push(candidate_total.as_secs_f64() * 1e9 / iterations as f64);
                    orig_samples
                        .borrow_mut()
                        .push(orig_total.as_secs_f64() * 1e9 / iterations as f64);
                    candidate_total + orig_total
                });
            });
            report_ledger_pair(
                "f32_argsort_rounded_ties_2m",
                &candidate_samples,
                &orig_samples,
            );

            // NULL CONTROL (A/A): the candidate arm registered twice in the same interleaved
            // routine. Its ratio and cv are the harness noise floor - any lever effect below
            // this floor is undecidable on this harness (franken_whisper null-control rule).
            let null_a = RefCell::new(Vec::new());
            let null_b = RefCell::new(Vec::new());
            let null_order = Cell::new(0u64);
            group.bench_function("f32_argsort_rounded_ties_2m_null_control", |bench| {
                bench.iter_custom(|iterations| {
                    let mut a_total = Duration::ZERO;
                    let mut b_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let b_first = null_order.get() & 1 == 1;
                        null_order.set(null_order.get().wrapping_add(1));
                        if b_first {
                            let start = Instant::now();
                            black_box(
                                ledger_f32_tie_argsort_candidate(&fnp_argsort, &input)
                                    .expect("null-control arm b"),
                            );
                            b_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                ledger_f32_tie_argsort_candidate(&fnp_argsort, &input)
                                    .expect("null-control arm a"),
                            );
                            a_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(
                                ledger_f32_tie_argsort_candidate(&fnp_argsort, &input)
                                    .expect("null-control arm a"),
                            );
                            a_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                ledger_f32_tie_argsort_candidate(&fnp_argsort, &input)
                                    .expect("null-control arm b"),
                            );
                            b_total += start.elapsed();
                        }
                    }
                    null_a
                        .borrow_mut()
                        .push(a_total.as_secs_f64() * 1e9 / iterations as f64);
                    null_b
                        .borrow_mut()
                        .push(b_total.as_secs_f64() * 1e9 / iterations as f64);
                    a_total + b_total
                });
            });
            report_ledger_pair("f32_argsort_null_control_AA", &null_a, &null_b);

            // Self-time of the pre-check unit the dispatch dedupe removes: ONE full parallel
            // NaN scan + ONE 65,536-sample strided tie oracle over the same 2M f32 buffer
            // (bench-local reconstruction of the dispatch's NaN scan + argsort_sample_has_tie;
            // before the fix, dense-tie input paid this unit TWICE - radix candidate then
            // comparison candidate - before delegation).
            let raw: Vec<u8> = input
                .call_method0("tobytes")
                .expect("f32 tie bytes")
                .extract()
                .expect("extract f32 tie bytes");
            let data: Vec<f32> = raw
                .chunks_exact(4)
                .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("one native f32")))
                .collect();
            group.bench_function("f32_argsort_tie_precheck_selftime_2m", |bench| {
                bench.iter(|| {
                    use rayon::prelude::*;
                    let d = black_box(&data);
                    let nan = d.par_iter().any(|v| v.is_nan());
                    const TIE_SAMPLE: usize = 1 << 16;
                    let n = d.len();
                    let k = n.min(TIE_SAMPLE);
                    let stride = (n / k).max(1);
                    let mut sample: Vec<f32> = (0..k).map(|i| d[i * stride]).collect();
                    sample.sort_unstable_by(|a, b| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let tie = (1..sample.len()).any(|i| sample[i] == sample[i - 1]);
                    black_box((nan, tie))
                });
            });
        }
    });

    group.finish();
}

fn main() {
    common::gated_main(&[
        ("bench_wide_string_sort_median_gate", bench_wide_string_sort_median_gate),
        ("bench_accumulate_extremum_median_gate", bench_accumulate_extremum_median_gate),
        ("bench_int_convolve_median_gate", bench_int_convolve_median_gate),
        ("bench_completion_median_gate", bench_completion_median_gate),
        ("bench_f64_transcendental_median_gate", bench_f64_transcendental_median_gate),
        ("bench_f64_exp_log_probe", bench_f64_exp_log_probe),
        ("bench_f64_exp_log_median_gate", bench_f64_exp_log_median_gate),
        ("bench_bool_sort_median_gate", bench_bool_sort_median_gate),
        ("bench_int_matmul_median_gate", bench_int_matmul_median_gate),
        ("bench_f16_matmul_median_gate", bench_f16_matmul_median_gate),
        ("bench_multidot_median_gate", bench_multidot_median_gate),
        ("bench_isclose_median_gate", bench_isclose_median_gate),
        ("bench_f16_unique_median_gate", bench_f16_unique_median_gate),
        ("bench_f16_around_median_gate", bench_f16_around_median_gate),
        ("bench_f16_einsum_median_gate", bench_f16_einsum_median_gate),
        ("bench_wide_string_substrate_v2", bench_wide_string_substrate_v2),
        ("bench_ledger_integrity_rejects", bench_ledger_integrity_rejects),
    ]);
}
