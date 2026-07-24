//! back-half misc domain criterion benches — parallel binary ufuncs, axis/kind
//! sort, matmul, char add, asarray-dtype, timedelta cumsum, datetime
//! minmax/ptp/argextreme, argextreme-f32, nanarg (nonlast/lastaxis), lexsort, and
//! unique rows/cols — the remaining contiguous tail of
//! `criterion_python_surface.rs`, split into its own per-domain bench binary. See
//! bead deadlock-audit-x7nnf.

#[path = "common/mod.rs"]
mod common;

use common::ensure_numpy_available;
use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyDict, PyModule};
use std::hint::black_box;
use std::time::Duration;

fn bench_parallel_binary_boundary(c: &mut Criterion) {
    // float_power / remainder / nextafter / power / fmod / heaviside / maximum / minimum /
    // copysign at 8M — routed through the zero-copy parallel binary kernel (numpy runs these
    // single-threaded).
    let mut group = c.benchmark_group("python_parallel_binary_boundary");
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
a = np.abs(rng.standard_normal(8_000_000)) + 0.1\n\
b = rng.standard_normal(8_000_000) * 5.0\n\
bnz = np.where(b == 0.0, 1.0, b)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("parallel binary setup");
        let a = ns.get_item("a").expect("a");
        let b = ns.get_item("b").expect("b");
        let bnz = ns.get_item("bnz").expect("bnz");
        for (op, x, y) in [
            ("float_power", &a, &b),
            ("nextafter", &a, &b),
            ("remainder", &a, &bnz),
            ("power", &a, &b),
            ("fmod", &a, &bnz),
            ("heaviside", &b, &a),
            ("maximum", &a, &b),
            ("minimum", &a, &b),
            ("copysign", &a, &b),
            ("divide", &a, &b),
        ] {
            let fnp_fn = module.getattr(op).expect("fnp op");
            let numpy_fn = numpy.getattr(op).expect("numpy op");
            group.bench_function(format!("fnp_{op}_f64_8m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((x, y)).expect("fnp call")));
            });
            group.bench_function(format!("numpy_{op}_f64_8m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((x, y)).expect("numpy call")));
            });
        }
        // f16 add/multiply/subtract: numpy widens to f32 (compute-bound, ~2.2x slower
        // than f32); native parallel widen->op->narrow wins.
        let h_setup = "import numpy as np\n\
rng = np.random.default_rng(3)\n\
ha = rng.standard_normal(16_000_000).astype(np.float16)\n\
hb = rng.standard_normal(16_000_000).astype(np.float16)\n";
        py.run(
            std::ffi::CString::new(h_setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("f16 binary setup");
        let ha = ns.get_item("ha").expect("ha");
        let hb = ns.get_item("hb").expect("hb");
        for op in ["add", "multiply", "maximum", "minimum", "greater", "less"] {
            let fnp_fn = module.getattr(op).expect("fnp op");
            let numpy_fn = numpy.getattr(op).expect("numpy op");
            group.bench_function(format!("fnp_{op}_f16_16m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&ha, &hb)).expect("fnp f16 call")));
            });
            group.bench_function(format!("numpy_{op}_f16_16m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&ha, &hb)).expect("numpy f16 call")));
            });
        }
        // f16 fmod/remainder: numpy widens f16->f32 (~214ms / ~317ms @16M, slowest f16 binary).
        // hb has near-zero entries; replace them so divisors are non-zero (kernel engages).
        py.run(
            std::ffi::CString::new(
                "hbnz = np.where(np.abs(hb) < np.float16(0.05), np.float16(1.5), hb)",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("f16 nonzero divisor setup");
        let hbnz = ns.get_item("hbnz").expect("hbnz");
        // f16 divide/floor_divide: numpy widens f16->f32->op->narrow single-threaded (~222/~375ms@16M).
        for op in ["fmod", "remainder", "divide", "floor_divide"] {
            let fnp_fn = module.getattr(op).expect("fnp op");
            let numpy_fn = numpy.getattr(op).expect("numpy op");
            group.bench_function(format!("fnp_{op}_f16_16m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&ha, &hbnz)).expect("fnp f16 call")));
            });
            group.bench_function(format!("numpy_{op}_f16_16m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&ha, &hbnz)).expect("numpy f16 call")));
            });
        }
        // f16 copysign/heaviside/nextafter: numpy widens f16->f32 (~22 / ~100 / ~57ms @16M).
        for op in ["copysign", "heaviside", "nextafter"] {
            let fnp_fn = module.getattr(op).expect("fnp op");
            let numpy_fn = numpy.getattr(op).expect("numpy op");
            group.bench_function(format!("fnp_{op}_f16_16m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&ha, &hb)).expect("fnp f16 call")));
            });
            group.bench_function(format!("numpy_{op}_f16_16m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&ha, &hb)).expect("numpy f16 call")));
            });
        }
        // f16 unary rounding ops floor/ceil/trunc/rint: numpy has no native f16 ALU and emulates
        // via widen->f32->op->narrow (compute-bound, ~77-126ms at 16M); native parallel wins ~15-30x.
        for op in [
            "floor", "ceil", "trunc", "rint", "isnan", "isfinite", "signbit",
        ] {
            let fnp_fn = module.getattr(op).expect("fnp unary op");
            let numpy_fn = numpy.getattr(op).expect("numpy unary op");
            group.bench_function(format!("fnp_{op}_f16_16m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&ha,)).expect("fnp f16 unary call")));
            });
            group.bench_function(format!("numpy_{op}_f16_16m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&ha,)).expect("numpy f16 unary call")));
            });
        }
        // f16 sqrt/square: numpy widens (compute-bound). Native parallel widen wins for the
        // warning-free common case (sqrt of non-negatives, square of |x|<256).
        let hsq_setup = "import numpy as np\n\
rng = np.random.default_rng(8)\n\
hsq = (np.abs(rng.standard_normal(16_000_000)) * 10.0).astype(np.float16)\n";
        py.run(
            std::ffi::CString::new(hsq_setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("f16 sqrt/square setup");
        let hsq = ns.get_item("hsq").expect("hsq");
        for op in ["sqrt", "square"] {
            let fnp_fn = module.getattr(op).expect("fnp f16 unary op");
            let numpy_fn = numpy.getattr(op).expect("numpy f16 unary op");
            group.bench_function(format!("fnp_{op}_f16_16m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&hsq,)).expect("fnp f16 unary call")));
            });
            group.bench_function(format!("numpy_{op}_f16_16m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&hsq,)).expect("numpy f16 unary call")));
            });
        }
        // f16 reciprocal: numpy widens f16->f32, 1/x (IEEE divide -> bit-exact), narrows (~62ms@16M).
        // Native parallel widen wins; values >= 0.5 so no 1/x overflow (overflow inputs would defer).
        py.run(
            std::ffi::CString::new(
                "hrecip = (np.abs(rng.standard_normal(16_000_000)) * 5.0 + 0.5).astype(np.float16)",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("f16 reciprocal setup");
        let hrecip = ns.get_item("hrecip").expect("hrecip");
        {
            let fnp_recip = module.getattr("reciprocal").expect("fnp reciprocal");
            let numpy_recip = numpy.getattr("reciprocal").expect("numpy reciprocal");
            group.bench_function("fnp_reciprocal_f16_16m", |bch| {
                bch.iter(|| black_box(fnp_recip.call1((&hrecip,)).expect("fnp f16 reciprocal")));
            });
            group.bench_function("numpy_reciprocal_f16_16m", |bch| {
                bch.iter(|| {
                    black_box(numpy_recip.call1((&hrecip,)).expect("numpy f16 reciprocal"))
                });
            });
        }
        // f16 modf: numpy widens f16->f32, splits into (trunc, frac), narrows both — single-threaded
        // scalar loop (~158ms@16M = ~0.6 GB/s, deeply compute-bound). Native parallel split wins.
        {
            let fnp_modf = module.getattr("modf").expect("fnp modf");
            let numpy_modf = numpy.getattr("modf").expect("numpy modf");
            group.bench_function("fnp_modf_f16_16m", |bch| {
                bch.iter(|| black_box(fnp_modf.call1((&hsq,)).expect("fnp f16 modf")));
            });
            group.bench_function("numpy_modf_f16_16m", |bch| {
                bch.iter(|| black_box(numpy_modf.call1((&hsq,)).expect("numpy f16 modf")));
            });
        }
        // f16 frexp: numpy widens f16->f32, decomposes (mantissa, exponent), narrows — single-threaded
        // scalar loop (~123ms@16M = ~0.8 GB/s, compute-bound). Native parallel bit-split wins.
        {
            let fnp_frexp = module.getattr("frexp").expect("fnp frexp");
            let numpy_frexp = numpy.getattr("frexp").expect("numpy frexp");
            group.bench_function("fnp_frexp_f16_16m", |bch| {
                bch.iter(|| black_box(fnp_frexp.call1((&hsq,)).expect("fnp f16 frexp")));
            });
            group.bench_function("numpy_frexp_f16_16m", |bch| {
                bch.iter(|| black_box(numpy_frexp.call1((&hsq,)).expect("numpy f16 frexp")));
            });
        }
        // f16 ldexp: numpy widens f16->f32, scalbnf, narrows — single-threaded (~108ms@16M = ~1.2
        // GB/s, compute-bound). Native parallel exact-pow2-scale wins. (i32 exponent in [-5,5).)
        py.run(
            std::ffi::CString::new("lde = rng.integers(-5, 5, 16_000_000, dtype=np.int32)")
                .unwrap()
                .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("f16 ldexp setup");
        let lde = ns.get_item("lde").expect("lde");
        {
            let fnp_ldexp = module.getattr("ldexp").expect("fnp ldexp");
            let numpy_ldexp = numpy.getattr("ldexp").expect("numpy ldexp");
            group.bench_function("fnp_ldexp_f16_16m", |bch| {
                bch.iter(|| black_box(fnp_ldexp.call1((&hsq, &lde)).expect("fnp f16 ldexp")));
            });
            group.bench_function("numpy_ldexp_f16_16m", |bch| {
                bch.iter(|| black_box(numpy_ldexp.call1((&hsq, &lde)).expect("numpy f16 ldexp")));
            });
        }
        // complex128/complex64 last-axis cumsum (4000x4000=16M): numpy's complex cumsum is a single-
        // threaded sequential dependency chain (~177ms@16M c128); native per-lane parallel scan wins.
        py.run(
            std::ffi::CString::new(
                "cc128 = (rng.standard_normal((4000,4000))+1j*rng.standard_normal((4000,4000))).astype(np.complex128); cc64 = cc128.astype(np.complex64)",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("complex cumsum setup");
        let cc128 = ns.get_item("cc128").expect("cc128");
        let cc64 = ns.get_item("cc64").expect("cc64");
        let kw_ax1 = PyDict::new(py);
        kw_ax1.set_item("axis", 1i64).expect("axis kw");
        let fnp_cumsum = module.getattr("cumsum").expect("fnp cumsum");
        let numpy_cumsum = numpy.getattr("cumsum").expect("numpy cumsum");
        for (arr, tag) in [(&cc128, "c128"), (&cc64, "c64")] {
            group.bench_function(format!("fnp_cumsum_lastaxis_{tag}_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        fnp_cumsum
                            .call((arr,), Some(&kw_ax1))
                            .expect("fnp complex cumsum"),
                    )
                });
            });
            group.bench_function(format!("numpy_cumsum_lastaxis_{tag}_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        numpy_cumsum
                            .call((arr,), Some(&kw_ax1))
                            .expect("numpy complex cumsum"),
                    )
                });
            });
        }
        // f16 clip: numpy widens f16->f32 to clamp (~149ms@16M, biggest f16 elementwise gap).
        {
            let fnp_clip = module.getattr("clip").expect("fnp clip");
            let numpy_clip = numpy.getattr("clip").expect("numpy clip");
            group.bench_function("fnp_clip_f16_16m", |bch| {
                bch.iter(|| {
                    black_box(
                        fnp_clip
                            .call1((&ha, -0.5f64, 0.5f64))
                            .expect("fnp f16 clip"),
                    )
                });
            });
            group.bench_function("numpy_clip_f16_16m", |bch| {
                bch.iter(|| {
                    black_box(
                        numpy_clip
                            .call1((&ha, -0.5f64, 0.5f64))
                            .expect("numpy f16 clip"),
                    )
                });
            });
        }
        // f16 round (decimals=0 == rint): numpy widens f16->f32 (~120ms@16M).
        {
            let fnp_round = module.getattr("round").expect("fnp round");
            let numpy_round = numpy.getattr("round").expect("numpy round");
            group.bench_function("fnp_round_f16_16m", |bch| {
                bch.iter(|| black_box(fnp_round.call1((&ha,)).expect("fnp f16 round")));
            });
            group.bench_function("numpy_round_f16_16m", |bch| {
                bch.iter(|| black_box(numpy_round.call1((&ha,)).expect("numpy f16 round")));
            });
        }
        // f16 nan_to_num: numpy widens f16->f32 (~112ms@16M); native uint16 bit-replacement wins.
        {
            let fnp_n2n = module.getattr("nan_to_num").expect("fnp nan_to_num");
            let numpy_n2n = numpy.getattr("nan_to_num").expect("numpy nan_to_num");
            group.bench_function("fnp_nantonum_f16_16m", |bch| {
                bch.iter(|| black_box(fnp_n2n.call1((&ha,)).expect("fnp f16 nan_to_num")));
            });
            group.bench_function("numpy_nantonum_f16_16m", |bch| {
                bch.iter(|| black_box(numpy_n2n.call1((&ha,)).expect("numpy f16 nan_to_num")));
            });
        }
        // f16 flat min/max reduction: numpy widens f16->f32 to reduce (~80ms@16M); native
        // parallel f32-fold reduce wins (bit-exact, defers NaN / zero-extremum). hsq is all
        // non-negative with a non-zero max -> exercises the kernel.
        for op in ["max", "min", "ptp", "argmax", "argmin"] {
            let fnp_fn = module.getattr(op).expect("fnp reduce op");
            let numpy_fn = numpy.getattr(op).expect("numpy reduce op");
            group.bench_function(format!("fnp_{op}reduce_f16_16m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&hsq,)).expect("fnp f16 reduce call")));
            });
            group.bench_function(format!("numpy_{op}reduce_f16_16m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&hsq,)).expect("numpy f16 reduce call")));
            });
        }
        // f16 last-axis argmax/argmin: numpy widens f16->f32 per lane; native per-lane scan wins.
        py.run(
            std::ffi::CString::new("hsq2 = hsq.reshape(4000, 4000)")
                .unwrap()
                .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("f16 2-D reshape");
        let hsq2 = ns.get_item("hsq2").expect("hsq2");
        let kw_axis = PyDict::new(py);
        kw_axis.set_item("axis", -1i64).expect("axis kwarg");
        // f16 min/max along last-axis + axis-0 (4000x4000): numpy widens f16->f32 per lane
        // (~76ms@16M); native per-lane parallel f32-fold reduce wins (bit-exact, NaN/zero defer).
        let kw_axis0 = PyDict::new(py);
        kw_axis0.set_item("axis", 0i64).expect("axis0 kwarg");
        for op in ["max", "min"] {
            let fnp_fn = module.getattr(op).expect("fnp f16 minmax op");
            let numpy_fn = numpy.getattr(op).expect("numpy f16 minmax op");
            group.bench_function(format!("fnp_{op}_lastaxis_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        fnp_fn
                            .call((&hsq2,), Some(&kw_axis))
                            .expect("fnp f16 lastaxis minmax"),
                    )
                });
            });
            group.bench_function(format!("numpy_{op}_lastaxis_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        numpy_fn
                            .call((&hsq2,), Some(&kw_axis))
                            .expect("numpy f16 lastaxis minmax"),
                    )
                });
            });
            group.bench_function(format!("fnp_{op}_axis0_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        fnp_fn
                            .call((&hsq2,), Some(&kw_axis0))
                            .expect("fnp f16 axis0 minmax"),
                    )
                });
            });
            group.bench_function(format!("numpy_{op}_axis0_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        numpy_fn
                            .call((&hsq2,), Some(&kw_axis0))
                            .expect("numpy f16 axis0 minmax"),
                    )
                });
            });
        }
        // f16 ptp along last-axis + axis-0 (4000x4000): numpy widens f16->f32 for BOTH passes then
        // subtracts (the slowest f16 reduction); native per-lane max-min wins (bit-exact, NaN defer).
        let fnp_ptp = module.getattr("ptp").expect("fnp ptp");
        let numpy_ptp = numpy.getattr("ptp").expect("numpy ptp");
        group.bench_function("fnp_ptp_lastaxis_f16_16m", |bch| {
            bch.iter(|| {
                black_box(
                    fnp_ptp
                        .call((&hsq2,), Some(&kw_axis))
                        .expect("fnp f16 lastaxis ptp"),
                )
            });
        });
        group.bench_function("numpy_ptp_lastaxis_f16_16m", |bch| {
            bch.iter(|| {
                black_box(
                    numpy_ptp
                        .call((&hsq2,), Some(&kw_axis))
                        .expect("numpy f16 lastaxis ptp"),
                )
            });
        });
        group.bench_function("fnp_ptp_axis0_f16_16m", |bch| {
            bch.iter(|| {
                black_box(
                    fnp_ptp
                        .call((&hsq2,), Some(&kw_axis0))
                        .expect("fnp f16 axis0 ptp"),
                )
            });
        });
        group.bench_function("numpy_ptp_axis0_f16_16m", |bch| {
            bch.iter(|| {
                black_box(
                    numpy_ptp
                        .call((&hsq2,), Some(&kw_axis0))
                        .expect("numpy f16 axis0 ptp"),
                )
            });
        });
        // f16 nanmin/nanmax flat + last-axis + axis-0: numpy widens f16->f32 skip-NaN (~32ms@16M);
        // native uint16-view skip-NaN reduce wins. Sparse NaN at stride 997 (coprime to 4000) so no
        // lane is all-NaN and the kernel engages (all-NaN / zero-extremum lanes would defer).
        py.run(
            std::ffi::CString::new(
                "hsqn = hsq.copy(); hsqn[::997] = np.float16(np.nan); hsqn2 = hsqn.reshape(4000, 4000)",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("f16 nan array");
        let hsqn = ns.get_item("hsqn").expect("hsqn");
        let hsqn2 = ns.get_item("hsqn2").expect("hsqn2");
        for op in ["nanmin", "nanmax"] {
            let fnp_fn = module.getattr(op).expect("fnp f16 nan op");
            let numpy_fn = numpy.getattr(op).expect("numpy f16 nan op");
            group.bench_function(format!("fnp_{op}_flat_f16_16m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&hsqn,)).expect("fnp f16 flat nan")));
            });
            group.bench_function(format!("numpy_{op}_flat_f16_16m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&hsqn,)).expect("numpy f16 flat nan")));
            });
            group.bench_function(format!("fnp_{op}_lastaxis_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        fnp_fn
                            .call((&hsqn2,), Some(&kw_axis))
                            .expect("fnp f16 lastaxis nan"),
                    )
                });
            });
            group.bench_function(format!("numpy_{op}_lastaxis_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        numpy_fn
                            .call((&hsqn2,), Some(&kw_axis))
                            .expect("numpy f16 lastaxis nan"),
                    )
                });
            });
            group.bench_function(format!("fnp_{op}_axis0_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        fnp_fn
                            .call((&hsqn2,), Some(&kw_axis0))
                            .expect("fnp f16 axis0 nan"),
                    )
                });
            });
            group.bench_function(format!("numpy_{op}_axis0_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        numpy_fn
                            .call((&hsqn2,), Some(&kw_axis0))
                            .expect("numpy f16 axis0 nan"),
                    )
                });
            });
        }
        // f16 cumsum/cumprod along last-axis + axis-0: numpy widens f16->f32 per element + narrows
        // each step, all lanes single-threaded (~138/106ms@16M); native per-lane parallel scan wins.
        for op in ["cumsum", "cumprod"] {
            let fnp_fn = module.getattr(op).expect("fnp f16 cum op");
            let numpy_fn = numpy.getattr(op).expect("numpy f16 cum op");
            group.bench_function(format!("fnp_{op}_lastaxis_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        fnp_fn
                            .call((&hsq2,), Some(&kw_axis))
                            .expect("fnp f16 lastaxis cum"),
                    )
                });
            });
            group.bench_function(format!("numpy_{op}_lastaxis_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        numpy_fn
                            .call((&hsq2,), Some(&kw_axis))
                            .expect("numpy f16 lastaxis cum"),
                    )
                });
            });
            group.bench_function(format!("fnp_{op}_axis0_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        fnp_fn
                            .call((&hsq2,), Some(&kw_axis0))
                            .expect("fnp f16 axis0 cum"),
                    )
                });
            });
            group.bench_function(format!("numpy_{op}_axis0_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        numpy_fn
                            .call((&hsq2,), Some(&kw_axis0))
                            .expect("numpy f16 axis0 cum"),
                    )
                });
            });
        }
        // f16 nancumsum/nancumprod last-axis + axis-0 (4000x4000, sparse NaN -> identity): numpy
        // widens f16->f32 + nan-mask + narrows each step (~202/171ms); native per-lane scan wins.
        for op in ["nancumsum", "nancumprod"] {
            let fnp_fn = module.getattr(op).expect("fnp f16 nancum op");
            let numpy_fn = numpy.getattr(op).expect("numpy f16 nancum op");
            group.bench_function(format!("fnp_{op}_lastaxis_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        fnp_fn
                            .call((&hsqn2,), Some(&kw_axis))
                            .expect("fnp f16 lastaxis nancum"),
                    )
                });
            });
            group.bench_function(format!("numpy_{op}_lastaxis_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        numpy_fn
                            .call((&hsqn2,), Some(&kw_axis))
                            .expect("numpy f16 lastaxis nancum"),
                    )
                });
            });
            group.bench_function(format!("fnp_{op}_axis0_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        fnp_fn
                            .call((&hsqn2,), Some(&kw_axis0))
                            .expect("fnp f16 axis0 nancum"),
                    )
                });
            });
            group.bench_function(format!("numpy_{op}_axis0_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        numpy_fn
                            .call((&hsqn2,), Some(&kw_axis0))
                            .expect("numpy f16 axis0 nancum"),
                    )
                });
            });
        }
        // f64/int64 cumsum AXIS-0 (4000x4000=16M): numpy runs cumsum single-threaded for every dtype
        // (~166/163ms); the transpose column-parallel axis-0 path parallelizes the previously-serial
        // axis-0 scan (last-axis was already parallel).
        py.run(
            std::ffi::CString::new(
                "c64 = rng.standard_normal((4000, 4000)); ci64 = rng.integers(-1000, 1000, (4000, 4000)).astype(np.int64)",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("cumsum axis0 arrays");
        let c64 = ns.get_item("c64").expect("c64");
        let ci64 = ns.get_item("ci64").expect("ci64");
        let fnp_cumsum = module.getattr("cumsum").expect("fnp cumsum");
        let numpy_cumsum = numpy.getattr("cumsum").expect("numpy cumsum");
        for (arr, tag) in [(&c64, "f64"), (&ci64, "i64")] {
            group.bench_function(format!("fnp_cumsum_axis0_{tag}_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        fnp_cumsum
                            .call((arr,), Some(&kw_axis0))
                            .expect("fnp cumsum axis0"),
                    )
                });
            });
            group.bench_function(format!("numpy_cumsum_axis0_{tag}_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        numpy_cumsum
                            .call((arr,), Some(&kw_axis0))
                            .expect("numpy cumsum axis0"),
                    )
                });
            });
        }
        for op in ["argmax", "argmin"] {
            let fnp_fn = module.getattr(op).expect("fnp arg op");
            let numpy_fn = numpy.getattr(op).expect("numpy arg op");
            group.bench_function(format!("fnp_{op}_lastaxis_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        fnp_fn
                            .call((&hsq2,), Some(&kw_axis))
                            .expect("fnp f16 lastaxis arg"),
                    )
                });
            });
            group.bench_function(format!("numpy_{op}_lastaxis_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        numpy_fn
                            .call((&hsq2,), Some(&kw_axis))
                            .expect("numpy f16 lastaxis arg"),
                    )
                });
            });
        }
        // f16 NON-last-axis (axis=0) argmax/argmin: numpy widens f16->f32 per column.
        let kw_axis0 = PyDict::new(py);
        kw_axis0.set_item("axis", 0i64).expect("axis0 kwarg");
        for op in ["argmax", "argmin"] {
            let fnp_fn = module.getattr(op).expect("fnp arg op");
            let numpy_fn = numpy.getattr(op).expect("numpy arg op");
            group.bench_function(format!("fnp_{op}_axis0_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        fnp_fn
                            .call((&hsq2,), Some(&kw_axis0))
                            .expect("fnp f16 axis0 arg"),
                    )
                });
            });
            group.bench_function(format!("numpy_{op}_axis0_f16_16m"), |bch| {
                bch.iter(|| {
                    black_box(
                        numpy_fn
                            .call((&hsq2,), Some(&kw_axis0))
                            .expect("numpy f16 axis0 arg"),
                    )
                });
            });
        }
        // f32 fmod/copysign: numpy runs f32 binary ufuncs single-threaded (fmod ~138ms @16M);
        // there was no f32 binary zero-copy path. Native parallel f32 kernel wins (bit-exact).
        let f32_setup = "import numpy as np\n\
rng = np.random.default_rng(5)\n\
af = (rng.standard_normal(16_000_000) * 1e3).astype(np.float32)\n\
bf = (rng.standard_normal(16_000_000) * 7.0).astype(np.float32)\n\
bf[np.abs(bf) < 1e-3] = np.float32(1.5)\n";
        py.run(
            std::ffi::CString::new(f32_setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("f32 binary setup");
        let af = ns.get_item("af").expect("af");
        let bf = ns.get_item("bf").expect("bf");
        for op in ["fmod", "copysign", "remainder", "nextafter"] {
            let fnp_fn = module.getattr(op).expect("fnp f32 op");
            let numpy_fn = numpy.getattr(op).expect("numpy f32 op");
            group.bench_function(format!("fnp_{op}_f32_16m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&af, &bf)).expect("fnp f32 call")));
            });
            group.bench_function(format!("numpy_{op}_f32_16m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&af, &bf)).expect("numpy f32 call")));
            });
        }
        // integer gcd: numpy np.gcd is a single-threaded Euclid element loop (16M int64 ~995ms);
        // native parallel Euclid kernel wins big (bit-exact).
        let gcd_setup = "import numpy as np\n\
rng = np.random.default_rng(6)\n\
ag = rng.integers(1, 10**9, 16_000_000).astype(np.int64)\n\
cg = rng.integers(1, 10**9, 16_000_000).astype(np.int64)\n\
apw = rng.integers(-1000, 1000, 16_000_000).astype(np.int64)\n\
epw = rng.integers(0, 12, 16_000_000).astype(np.int64)\n";
        py.run(
            std::ffi::CString::new(gcd_setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("gcd setup");
        let ag = ns.get_item("ag").expect("ag");
        let cg = ns.get_item("cg").expect("cg");
        for op in ["gcd", "lcm"] {
            let fnp_fn = module.getattr(op).expect("fnp int op");
            let numpy_fn = numpy.getattr(op).expect("numpy int op");
            group.bench_function(format!("fnp_{op}_i64_16m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&ag, &cg)).expect("fnp int call")));
            });
            group.bench_function(format!("numpy_{op}_i64_16m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&ag, &cg)).expect("numpy int call")));
            });
        }
        // integer power: numpy a**b single-threaded element loop (16M int64 ~340ms); native
        // parallel wrapping repeated-squaring wins (bit-exact).
        let apw = ns.get_item("apw").expect("apw");
        let epw = ns.get_item("epw").expect("epw");
        let fnp_pow = module.getattr("power").expect("fnp power");
        let numpy_pow = numpy.getattr("power").expect("numpy power");
        group.bench_function("fnp_power_i64_16m", |bch| {
            bch.iter(|| black_box(fnp_pow.call1((&apw, &epw)).expect("fnp power call")));
        });
        group.bench_function("numpy_power_i64_16m", |bch| {
            bch.iter(|| black_box(numpy_pow.call1((&apw, &epw)).expect("numpy power call")));
        });
        // integer floor_divide / remainder / divmod: numpy single-threaded element loops
        // (16M int64 ~98ms / ~93ms / ~163ms).
        for op in ["floor_divide", "remainder", "divmod"] {
            let fnp_fn = module.getattr(op).expect("fnp int op");
            let numpy_fn = numpy.getattr(op).expect("numpy int op");
            group.bench_function(format!("fnp_{op}_i64_16m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&ag, &cg)).expect("fnp int call")));
            });
            group.bench_function(format!("numpy_{op}_i64_16m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&ag, &cg)).expect("numpy int call")));
            });
        }
        // timedelta64 // timedelta64 -> int64: numpy single-threaded w/ per-element NaT (~212ms@16M).
        py.run(
            std::ffi::CString::new(
                "atd = (np.abs(ag).astype('timedelta64[s]')); ctd = (np.where(cg==0,1,np.abs(cg)).astype('timedelta64[s]'))",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("timedelta setup");
        let atd = ns.get_item("atd").expect("atd");
        let ctd = ns.get_item("ctd").expect("ctd");
        for op in ["floor_divide", "remainder"] {
            let fnp_fn = module.getattr(op).expect("fnp td op");
            let numpy_fn = numpy.getattr(op).expect("numpy td op");
            group.bench_function(format!("fnp_{op}_td64_16m"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&atd, &ctd)).expect("fnp td op")));
            });
            group.bench_function(format!("numpy_{op}_td64_16m"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&atd, &ctd)).expect("numpy td op")));
            });
        }
        // f32 searchsorted: numpy single-threaded cold-cache binary search per query (~1.6s for
        // 8M queries into a 1M sorted f32). Native parallel per-query search wins big.
        py.run(
            std::ffi::CString::new(
                "ssa = np.sort(np.random.default_rng(9).standard_normal(1_000_000).astype(np.float32)); ssv = np.random.default_rng(10).standard_normal(8_000_000).astype(np.float32)",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("searchsorted setup");
        let ssa = ns.get_item("ssa").expect("ssa");
        let ssv = ns.get_item("ssv").expect("ssv");
        let fnp_ss = module.getattr("searchsorted").expect("fnp searchsorted");
        let numpy_ss = numpy.getattr("searchsorted").expect("numpy searchsorted");
        group.bench_function("fnp_searchsorted_f32_8m", |bch| {
            bch.iter(|| black_box(fnp_ss.call1((&ssa, &ssv)).expect("fnp searchsorted")));
        });
        group.bench_function("numpy_searchsorted_f32_8m", |bch| {
            bch.iter(|| black_box(numpy_ss.call1((&ssa, &ssv)).expect("numpy searchsorted")));
        });
        // f32 polyval (deg-11 Horner): numpy single-threaded (~570ms@16M).
        py.run(
            std::ffi::CString::new(
                "pvp = np.random.default_rng(11).standard_normal(12).astype(np.float32); pvx = np.random.default_rng(12).standard_normal(16_000_000).astype(np.float32)",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("polyval setup");
        let pvp = ns.get_item("pvp").expect("pvp");
        let pvx = ns.get_item("pvx").expect("pvx");
        let fnp_pv = module.getattr("polyval").expect("fnp polyval");
        let numpy_pv = numpy.getattr("polyval").expect("numpy polyval");
        group.bench_function("fnp_polyval_f32_16m", |bch| {
            bch.iter(|| black_box(fnp_pv.call1((&pvp, &pvx)).expect("fnp polyval")));
        });
        group.bench_function("numpy_polyval_f32_16m", |bch| {
            bch.iter(|| black_box(numpy_pv.call1((&pvp, &pvx)).expect("numpy polyval")));
        });
        // f32 ldexp: numpy scalbnf single-threaded (~86ms@16M).
        py.run(
            std::ffi::CString::new(
                "lxx = np.random.default_rng(13).standard_normal(16_000_000).astype(np.float32); lxe = np.random.default_rng(14).integers(-40,40,16_000_000).astype(np.int32)",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("ldexp setup");
        let lxx = ns.get_item("lxx").expect("lxx");
        let lxe = ns.get_item("lxe").expect("lxe");
        let fnp_le = module.getattr("ldexp").expect("fnp ldexp");
        let numpy_le = numpy.getattr("ldexp").expect("numpy ldexp");
        group.bench_function("fnp_ldexp_f32_16m", |bch| {
            bch.iter(|| black_box(fnp_le.call1((&lxx, &lxe)).expect("fnp ldexp")));
        });
        group.bench_function("numpy_ldexp_f32_16m", |bch| {
            bch.iter(|| black_box(numpy_le.call1((&lxx, &lxe)).expect("numpy ldexp")));
        });
        // f32 spacing: numpy single-threaded ULP (~64ms@16M).
        let fnp_sp = module.getattr("spacing").expect("fnp spacing");
        let numpy_sp = numpy.getattr("spacing").expect("numpy spacing");
        group.bench_function("fnp_spacing_f32_16m", |bch| {
            bch.iter(|| black_box(fnp_sp.call1((&lxx,)).expect("fnp spacing")));
        });
        group.bench_function("numpy_spacing_f32_16m", |bch| {
            bch.iter(|| black_box(numpy_sp.call1((&lxx,)).expect("numpy spacing")));
        });
    });

    group.finish();
}

fn bench_sort_axis_boundary(c: &mut Criterion) {
    // np.sort / np.argsort along the LAST (contiguous) axis of a 2-D f64 array — newly routed
    // through the per-lane parallel sort (numpy sorts each lane single-threaded, sequentially).
    let mut group = c.benchmark_group("python_sort_axis_boundary");
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
m = rng.standard_normal((2048, 2048))\n\
mshort = rng.standard_normal((65536, 64))\n\
m3 = rng.standard_normal((4096, 32, 32))\n\
m3b = rng.standard_normal((256, 256, 64))\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("sort axis setup");
        let m = ns.get_item("m").expect("m");
        let mshort = ns.get_item("mshort").expect("mshort");
        let m3 = ns.get_item("m3").expect("m3");
        let m3b = ns.get_item("m3b").expect("m3b");
        // This group is also the foreground proof vehicle for the cores-aware
        // f64 value-sort gate. Keep the parity assertion outside the timed loop.
        let fnp_sorted = module
            .getattr("sort")
            .expect("fnp sort")
            .call1((&m,))
            .expect("fnp sort parity");
        let numpy_sorted = numpy
            .getattr("sort")
            .expect("numpy sort")
            .call1((&m,))
            .expect("numpy sort parity");
        let fnp_bytes: Vec<u8> = fnp_sorted
            .call_method0("tobytes")
            .expect("fnp sort bytes")
            .extract()
            .expect("extract fnp sort bytes");
        let numpy_bytes: Vec<u8> = numpy_sorted
            .call_method0("tobytes")
            .expect("numpy sort bytes")
            .extract()
            .expect("extract numpy sort bytes");
        assert_eq!(fnp_bytes, numpy_bytes, "f64 last-axis sort byte mismatch");
        for op in ["sort", "argsort"] {
            let fnp_fn = module.getattr(op).expect("fnp op");
            let numpy_fn = numpy.getattr(op).expect("numpy op");
            group.bench_function(format!("fnp_{op}_lastaxis_2048x2048"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&m,)).expect("fnp call")));
            });
            group.bench_function(format!("numpy_{op}_lastaxis_2048x2048"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&m,)).expect("numpy call")));
            });
            // SHORT-LANE last axis (65536 x 64, cols=64 < SORT_LANE_PARALLEL_MIN=256): tens of
            // thousands of tiny lanes. The eager-parallel path lost 1.8-3.6x here; gated to
            // DELEGATE to numpy -> parity. Guards the 106th-win regression fix.
            group.bench_function(format!("fnp_{op}_lastaxis_short_65536x64"), |bch| {
                bch.iter(|| black_box(fnp_fn.call1((&mshort,)).expect("fnp call")));
            });
            group.bench_function(format!("numpy_{op}_lastaxis_short_65536x64"), |bch| {
                bch.iter(|| black_box(numpy_fn.call1((&mshort,)).expect("numpy call")));
            });
            // axis=0 (lane sort): numpy's strided per-lane sort is ~2x its last-axis sort.
            let kw0 = PyDict::new(py);
            kw0.set_item("axis", 0).expect("axis kwarg");
            group.bench_function(format!("fnp_{op}_axis0_2048x2048"), |bch| {
                bch.iter(|| black_box(fnp_fn.call((&m,), Some(&kw0)).expect("fnp call")));
            });
            group.bench_function(format!("numpy_{op}_axis0_2048x2048"), |bch| {
                bch.iter(|| black_box(numpy_fn.call((&m,), Some(&kw0)).expect("numpy call")));
            });
            // ndim>=2 axis=0 on a 3-D batched shape (cols = prod(shape[1:]) lanes).
            group.bench_function(format!("fnp_{op}_axis0_4096x32x32"), |bch| {
                bch.iter(|| black_box(fnp_fn.call((&m3,), Some(&kw0)).expect("fnp call")));
            });
            group.bench_function(format!("numpy_{op}_axis0_4096x32x32"), |bch| {
                bch.iter(|| black_box(numpy_fn.call((&m3,), Some(&kw0)).expect("numpy call")));
            });
            // ndim>=3 MIDDLE axis (axis=1): gather-strided-lane -> sort -> scatter. numpy's
            // strided middle-axis sort is ~1.2-1.8x slower than its last-axis sort, single-threaded.
            let kw1 = PyDict::new(py);
            kw1.set_item("axis", 1).expect("axis kwarg");
            group.bench_function(format!("fnp_{op}_midaxis_4096x32x32"), |bch| {
                bch.iter(|| black_box(fnp_fn.call((&m3,), Some(&kw1)).expect("fnp call")));
            });
            group.bench_function(format!("numpy_{op}_midaxis_4096x32x32"), |bch| {
                bch.iter(|| black_box(numpy_fn.call((&m3,), Some(&kw1)).expect("numpy call")));
            });
            group.bench_function(format!("fnp_{op}_midaxis_256x256x64"), |bch| {
                bch.iter(|| black_box(fnp_fn.call((&m3b,), Some(&kw1)).expect("fnp call")));
            });
            group.bench_function(format!("numpy_{op}_midaxis_256x256x64"), |bch| {
                bch.iter(|| black_box(numpy_fn.call((&m3b,), Some(&kw1)).expect("numpy call")));
            });
        }
    });

    group.finish();
}

fn bench_sort_kind_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_sort_kind_boundary");
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
m = rng.standard_normal((2048, 2048))\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("sort kind setup");
        let m = ns.get_item("m").expect("m");
        let kwargs = PyDict::new(py);
        kwargs.set_item("kind", "stable").expect("kind kwarg");
        for op in ["sort", "argsort"] {
            let fnp_fn = module.getattr(op).expect("fnp op");
            let numpy_fn = numpy.getattr(op).expect("numpy op");
            group.bench_function(format!("fnp_{op}_stable_lastaxis_2048x2048"), |bch| {
                bch.iter(|| black_box(fnp_fn.call((&m,), Some(&kwargs)).expect("fnp call")));
            });
            group.bench_function(format!("numpy_{op}_stable_lastaxis_2048x2048"), |bch| {
                bch.iter(|| black_box(numpy_fn.call((&m,), Some(&kwargs)).expect("numpy call")));
            });
        }
    });

    group.finish();
}

// np.matmul / np.dot on 2-D f64 squares spanning the native GEMM gate window
// ([320..1024]). The native pure-Rust GEMM was profiled as winning here, but a
// later numpy/OpenBLAS speedup made the gate stale (now a 1.5-6x loss vs BLAS).
fn bench_matmul_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_matmul_boundary");
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
a512 = rng.standard_normal((512, 512))\n\
b512 = rng.standard_normal((512, 512))\n\
a1024 = rng.standard_normal((1024, 1024))\n\
b1024 = rng.standard_normal((1024, 1024))\n\
a1536 = rng.standard_normal((1536, 1536))\n\
b1536 = rng.standard_normal((1536, 1536))\n\
a2048 = rng.standard_normal((2048, 2048))\n\
b2048 = rng.standard_normal((2048, 2048))\n\
a3d = rng.standard_normal((64, 256, 256))\n\
b3d = rng.standard_normal((64, 256, 256))\n\
a3db = rng.standard_normal((256, 128, 128))\n\
b3db = rng.standard_normal((256, 128, 128))\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("matmul setup");
        let m_fn = module.getattr("matmul").expect("fnp matmul");
        let np_m = numpy.getattr("matmul").expect("np matmul");
        for op in ["matmul", "dot"] {
            let fnp_fn = module.getattr(op).expect("fnp op");
            let numpy_fn = numpy.getattr(op).expect("numpy op");
            for sz in ["512", "1024", "1536", "2048"] {
                let a = ns.get_item(format!("a{sz}")).expect("a");
                let b = ns.get_item(format!("b{sz}")).expect("b");
                group.bench_function(format!("fnp_{op}_{sz}x{sz}"), |bch| {
                    bch.iter(|| black_box(fnp_fn.call1((&a, &b)).expect("fnp call")));
                });
                group.bench_function(format!("numpy_{op}_{sz}x{sz}"), |bch| {
                    bch.iter(|| black_box(numpy_fn.call1((&a, &b)).expect("numpy call")));
                });
            }
        }
        // INTEGER matmul: numpy has no BLAS for ints (naive loop, 89x slower than float
        // at 512). Native parallel ikj GEMM (bit-exact wrapping) should crush it.
        let int_setup = "import numpy as np\n\
rng = np.random.default_rng(7)\n\
ai512 = rng.integers(-100, 100, (512, 512)).astype(np.int64)\n\
bi512 = rng.integers(-100, 100, (512, 512)).astype(np.int64)\n\
ai1024 = rng.integers(-100, 100, (1024, 1024)).astype(np.int64)\n\
bi1024 = rng.integers(-100, 100, (1024, 1024)).astype(np.int64)\n";
        py.run(
            std::ffi::CString::new(int_setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("int matmul setup");
        let fnp_mm = module.getattr("matmul").expect("fnp matmul");
        let np_mm = numpy.getattr("matmul").expect("np matmul");
        for sz in ["512", "1024"] {
            let a = ns.get_item(format!("ai{sz}")).expect("ai");
            let b = ns.get_item(format!("bi{sz}")).expect("bi");
            group.bench_function(format!("fnp_matmul_i64_{sz}x{sz}"), |bch| {
                bch.iter(|| black_box(fnp_mm.call1((&a, &b)).expect("fnp int matmul")));
            });
            group.bench_function(format!("numpy_matmul_i64_{sz}x{sz}"), |bch| {
                bch.iter(|| black_box(np_mm.call1((&a, &b)).expect("numpy int matmul")));
            });
        }
        // BATCHED integer matmul (3-D): numpy int has no BLAS (naive per-slice serial).
        let bint_setup = "import numpy as np\n\
rng = np.random.default_rng(8)\n\
abi = rng.integers(-100, 100, (64, 128, 128)).astype(np.int64)\n\
bbi = rng.integers(-100, 100, (64, 128, 128)).astype(np.int64)\n";
        py.run(
            std::ffi::CString::new(bint_setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("batched int setup");
        {
            let a = ns.get_item("abi").expect("abi");
            let b = ns.get_item("bbi").expect("bbi");
            group.bench_function("fnp_matmul_i64_batched_64x128x128", |bch| {
                bch.iter(|| black_box(fnp_mm.call1((&a, &b)).expect("fnp int batched matmul")));
            });
            group.bench_function("numpy_matmul_i64_batched_64x128x128", |bch| {
                bch.iter(|| black_box(np_mm.call1((&a, &b)).expect("numpy int batched matmul")));
            });
        }
        // integer np.dot(2d,2d) routes to the same native GEMM (== matmul).
        let fnp_dot = module.getattr("dot").expect("fnp dot");
        let np_dot = numpy.getattr("dot").expect("np dot");
        {
            let a = ns.get_item("ai512").expect("ai512");
            let b = ns.get_item("bi512").expect("bi512");
            group.bench_function("fnp_dot_i64_512x512", |bch| {
                bch.iter(|| black_box(fnp_dot.call1((&a, &b)).expect("fnp int dot")));
            });
            group.bench_function("numpy_dot_i64_512x512", |bch| {
                bch.iter(|| black_box(np_dot.call1((&a, &b)).expect("numpy int dot")));
            });
            // integer np.inner(2d,2d) = a @ b^T routes to the native int GEMM.
            let fnp_inner = module.getattr("inner").expect("fnp inner");
            let np_inner = numpy.getattr("inner").expect("np inner");
            group.bench_function("fnp_inner_i64_512x512", |bch| {
                bch.iter(|| black_box(fnp_inner.call1((&a, &b)).expect("fnp int inner")));
            });
            group.bench_function("numpy_inner_i64_512x512", |bch| {
                bch.iter(|| black_box(np_inner.call1((&a, &b)).expect("numpy int inner")));
            });
        }
        // integer multi_dot chain (5 x 256x256): numpy no-BLAS chain vs native GEMM chain.
        let mdi_setup = "import numpy as np\n\
rng = np.random.default_rng(14)\n\
mdi = [rng.integers(-50, 50, (256, 256)).astype(np.int64) for _ in range(5)]\n";
        py.run(
            std::ffi::CString::new(mdi_setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("int multi_dot setup");
        {
            let mdi = ns.get_item("mdi").expect("mdi");
            let fnp_md = module.getattr("multi_dot").expect("fnp multi_dot");
            let np_md = numpy
                .getattr("linalg")
                .expect("linalg")
                .getattr("multi_dot")
                .expect("np multi_dot");
            group.bench_function("fnp_multi_dot_i64_5x256", |bch| {
                bch.iter(|| black_box(fnp_md.call1((&mdi,)).expect("fnp int multi_dot")));
            });
            group.bench_function("numpy_multi_dot_i64_5x256", |bch| {
                bch.iter(|| black_box(np_md.call1((&mdi,)).expect("numpy int multi_dot")));
            });
        }
        // INTEGER tensordot(axes=1) (64,64,64): numpy no-BLAS slow; routes to native int GEMM.
        let tdi_setup = "import numpy as np\n\
rng = np.random.default_rng(10)\n\
ati = rng.integers(-100, 100, (64, 64, 64)).astype(np.int64)\n\
bti = rng.integers(-100, 100, (64, 64, 64)).astype(np.int64)\n";
        py.run(
            std::ffi::CString::new(tdi_setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("int tensordot setup");
        {
            let fnp_td = module.getattr("tensordot").expect("fnp tensordot");
            let np_td = numpy.getattr("tensordot").expect("np tensordot");
            let a = ns.get_item("ati").expect("ati");
            let b = ns.get_item("bti").expect("bti");
            group.bench_function("fnp_tensordot_i64_axes1_64x64x64", |bch| {
                bch.iter(|| black_box(fnp_td.call1((&a, &b, 1_i64)).expect("fnp int tensordot")));
            });
            group.bench_function("numpy_tensordot_i64_axes1_64x64x64", |bch| {
                bch.iter(|| black_box(np_td.call1((&a, &b, 1_i64)).expect("numpy int tensordot")));
            });
        }
        let fnp_tensordot = module.getattr("tensordot").expect("fnp tensordot");
        let np_tensordot = numpy.getattr("tensordot").expect("np tensordot");
        for sz in ["1024", "1536"] {
            let a = ns.get_item(format!("a{sz}")).expect("a");
            let b = ns.get_item(format!("b{sz}")).expect("b");
            group.bench_function(format!("fnp_tensordot_axes1_{sz}x{sz}"), |bch| {
                bch.iter(|| black_box(fnp_tensordot.call1((&a, &b, 1_i64)).expect("fnp call")));
            });
            group.bench_function(format!("numpy_tensordot_axes1_{sz}x{sz}"), |bch| {
                bch.iter(|| black_box(np_tensordot.call1((&a, &b, 1_i64)).expect("numpy call")));
            });
        }
        // Batched (3-D) matmul: native parallel-across-batch packed GEMM vs numpy slow BLAS.
        for (tag, ak, bk) in [
            ("64x256x256", "a3d", "b3d"),
            ("256x128x128", "a3db", "b3db"),
        ] {
            let a = ns.get_item(ak).expect("a3d");
            let b = ns.get_item(bk).expect("b3d");
            group.bench_function(format!("fnp_matmul_batched_{tag}"), |bch| {
                bch.iter(|| black_box(m_fn.call1((&a, &b)).expect("fnp call")));
            });
            group.bench_function(format!("numpy_matmul_batched_{tag}"), |bch| {
                bch.iter(|| black_box(np_m.call1((&a, &b)).expect("numpy call")));
            });
        }
        // Matrix-BROADCAST batched matmul: one 2-D operand applied across the other's batch.
        let bcast_setup = "import numpy as np\n\
rng = np.random.default_rng(2)\n\
ab = rng.standard_normal((64, 256, 256))\n\
wb = rng.standard_normal((256, 256))\n\
aw = rng.standard_normal((256, 256))\n\
bb = rng.standard_normal((64, 256, 256))\n";
        py.run(
            std::ffi::CString::new(bcast_setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("bcast setup");
        let ab = ns.get_item("ab").expect("ab");
        let wb = ns.get_item("wb").expect("wb");
        let aw = ns.get_item("aw").expect("aw");
        let bb = ns.get_item("bb").expect("bb");
        group.bench_function("fnp_matmul_bcast_3dA_2dB_64x256x256", |bch| {
            bch.iter(|| black_box(m_fn.call1((&ab, &wb)).expect("fnp call")));
        });
        group.bench_function("numpy_matmul_bcast_3dA_2dB_64x256x256", |bch| {
            bch.iter(|| black_box(np_m.call1((&ab, &wb)).expect("numpy call")));
        });
        group.bench_function("fnp_matmul_bcast_2dA_3dB_64x256x256", |bch| {
            bch.iter(|| black_box(m_fn.call1((&aw, &bb)).expect("fnp call")));
        });
        group.bench_function("numpy_matmul_bcast_2dA_3dB_64x256x256", |bch| {
            bch.iter(|| black_box(np_m.call1((&aw, &bb)).expect("numpy call")));
        });
        // matrix_power: 2-D square repeated-squaring through the native packed GEMM vs numpy.
        let fnp_mp = module.getattr("matrix_power").expect("fnp matrix_power");
        let np_mp = numpy
            .getattr("linalg")
            .expect("linalg")
            .getattr("matrix_power")
            .expect("np matrix_power");
        for (sz, p) in [("512", 8_i64), ("1024", 6_i64)] {
            let a = ns.get_item(format!("a{sz}")).expect("a");
            group.bench_function(format!("fnp_matrix_power_{sz}x{sz}_p{p}"), |bch| {
                bch.iter(|| black_box(fnp_mp.call1((&a, p)).expect("fnp call")));
            });
            group.bench_function(format!("numpy_matrix_power_{sz}x{sz}_p{p}"), |bch| {
                bch.iter(|| black_box(np_mp.call1((&a, p)).expect("numpy call")));
            });
        }
    });

    group.finish();
}

// np.char.add / np.strings.add (string concatenation) — numpy runs a slow per-element Python
// loop; fnp has a native parallel UCS4 codepoint-copy path.
fn bench_char_add_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_char_add_boundary");
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
a = np.array(['Hello World '+str(i%1000) for i in range(300000)], dtype='<U16')\n\
b = np.array(['suffix'+str(i%50) for i in range(300000)], dtype='<U10')\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("char add setup");
        let a = ns.get_item("a").expect("a");
        let b = ns.get_item("b").expect("b");
        let Ok(fnp_char) = module.getattr("char") else {
            return;
        };
        let fnp_add = fnp_char.getattr("add").expect("fnp char.add");
        let Ok(numpy_char) = numpy.getattr("char") else {
            return;
        };
        let np_add = numpy_char.getattr("add").expect("np char.add");
        group.bench_function("fnp_char_add_300k", |bch| {
            bch.iter(|| black_box(fnp_add.call1((&a, &b)).expect("fnp call")));
        });
        group.bench_function("numpy_char_add_300k", |bch| {
            bch.iter(|| black_box(np_add.call1((&a, &b)).expect("numpy call")));
        });
    });

    group.finish();
}

fn bench_asarray_dtype_boundary(c: &mut Criterion) {
    // np.asarray(ndarray, dtype=<convert>) — a dtype CONVERSION delegates the cast
    // to numpy, but the native pre-check used to copy the whole input into a
    // UFuncArray before discarding it (a wasted full-array copy on top of numpy's
    // own cast = a 2-3x regression). The fix delegates BEFORE that extract; this
    // guards the conversion path stays at parity with numpy.
    let mut group = c.benchmark_group("python_asarray_dtype_boundary");
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
f64 = rng.standard_normal(4_000_000)\n\
i32 = rng.integers(0, 1000, 4_000_000).astype(np.int32)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("asarray setup");
        let f64 = ns.get_item("f64").expect("f64");
        let i32 = ns.get_item("i32").expect("i32");
        let fnp_asarray = module.getattr("asarray").expect("fnp asarray");
        let np_asarray = numpy.getattr("asarray").expect("np asarray");
        for (name, arr, to) in [
            ("f64_to_f32", &f64, "float32"),
            ("i32_to_f64", &i32, "float64"),
        ] {
            let kw = PyDict::new(py);
            kw.set_item("dtype", to).expect("dtype kwarg");
            group.bench_function(format!("fnp_asarray_{name}_4m"), |bch| {
                bch.iter(|| black_box(fnp_asarray.call((arr,), Some(&kw)).expect("fnp call")));
            });
            group.bench_function(format!("numpy_asarray_{name}_4m"), |bch| {
                bch.iter(|| black_box(np_asarray.call((arr,), Some(&kw)).expect("numpy call")));
            });
        }
    });

    group.finish();
}

// np.cumsum(timedelta64, axis): int64-backed; integer prefix sum is order-preserving (bit-exact),
// so the int64-view routes to the native int cumsum (~2.3x, result viewed back as timedelta64[unit]).
fn bench_timedelta_cumsum_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_timedelta_cumsum_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
td = rng.integers(-50000, 50000, (4096, 512, 8)).astype('timedelta64[D]')\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("timedelta cumsum setup");
        let td = ns.get_item("td").expect("td");
        let fnp_cumsum = module.getattr("cumsum").expect("fnp cumsum");
        let numpy_cumsum = numpy.getattr("cumsum").expect("numpy cumsum");
        let kw = PyDict::new(py);
        kw.set_item("axis", 1_i64).unwrap();
        let kw2 = kw.clone();
        group.bench_function("fnp_cumsum_td_mid", |b| {
            b.iter(|| black_box(fnp_cumsum.call((&td,), Some(&kw)).expect("fnp cumsum")));
        });
        group.bench_function("numpy_cumsum_td_mid", |b| {
            b.iter(|| black_box(numpy_cumsum.call((&td,), Some(&kw2)).expect("np cumsum")));
        });
    });

    group.finish();
}

// np.max/min(datetime64/timedelta64, axis): int64-backed; the int64-view routes to the native int
// min/max (~5-8x, result viewed back as the SAME temporal dtype). NaT pre-scan + defer.
fn bench_datetime_minmax_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_datetime_minmax_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
dt = rng.integers(0, 100000, (4096, 512, 8)).astype('datetime64[D]')\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("datetime minmax setup");
        let dt = ns.get_item("dt").expect("dt");
        for name in ["max", "min"] {
            let fnp_fn = module.getattr(name).expect("fnp fn");
            let numpy_fn = numpy.getattr(name).expect("numpy fn");
            let kw = PyDict::new(py);
            kw.set_item("axis", 1_i64).unwrap();
            let kw2 = kw.clone();
            group.bench_function(format!("fnp_{name}_dt_mid"), |b| {
                b.iter(|| black_box(fnp_fn.call((&dt,), Some(&kw)).expect("fnp mm")));
            });
            group.bench_function(format!("numpy_{name}_dt_mid"), |b| {
                b.iter(|| black_box(numpy_fn.call((&dt,), Some(&kw2)).expect("np mm")));
            });
        }
    });

    group.finish();
}

// np.ptp(datetime64/timedelta64, axis): int64-backed; numpy's temporal ptp is slow while the
// int64-view routes to the native int ptp (~6x, result viewed back as timedelta64[unit]).
fn bench_datetime_ptp_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_datetime_ptp_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
dt = rng.integers(0, 100000, (4096, 512, 8)).astype('datetime64[D]')\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("datetime ptp setup");
        let dt = ns.get_item("dt").expect("dt");
        let fnp_ptp = module.getattr("ptp").expect("fnp ptp");
        let numpy_ptp = numpy.getattr("ptp").expect("numpy ptp");
        let kw = PyDict::new(py);
        kw.set_item("axis", 1_i64).unwrap();
        let kw2 = kw.clone();
        group.bench_function("fnp_ptp_dt_mid", |b| {
            b.iter(|| black_box(fnp_ptp.call((&dt,), Some(&kw)).expect("fnp ptp")));
        });
        group.bench_function("numpy_ptp_dt_mid", |b| {
            b.iter(|| black_box(numpy_ptp.call((&dt,), Some(&kw2)).expect("np ptp")));
        });
    });

    group.finish();
}

// np.argmin/argmax(datetime64/timedelta64, axis): temporal reductions are int64-backed; numpy runs
// a slow temporal reduce while the int64-view routes to the fast native int argextreme (~6x after
// the NaT pre-scan). Bit-exact indices (int64 ordering == temporal ordering); NaT defers.
fn bench_datetime_argextreme_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_datetime_argextreme_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
dt = rng.integers(0, 100000, (4096, 512, 8)).astype('datetime64[D]')\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("datetime arg setup");
        let dt = ns.get_item("dt").expect("dt");
        for name in ["argmin", "argmax"] {
            let fnp_fn = module.getattr(name).expect("fnp fn");
            let numpy_fn = numpy.getattr(name).expect("numpy fn");
            let kw = PyDict::new(py);
            kw.set_item("axis", 1_i64).unwrap();
            let kw2 = kw.clone();
            group.bench_function(format!("fnp_{name}_dt_mid"), |b| {
                b.iter(|| black_box(fnp_fn.call((&dt,), Some(&kw)).expect("fnp arg")));
            });
            group.bench_function(format!("numpy_{name}_dt_mid"), |b| {
                b.iter(|| black_box(numpy_fn.call((&dt,), Some(&kw2)).expect("np arg")));
            });
        }
    });

    group.finish();
}

// np.argmin/argmax(f32, non-last axis): f32 had no arg-axis kernel (only f64+f16), so it delegated
// to numpy's slow strided reduce (~parity). The parallel per-block f32 scan wins ~8x.
fn bench_argextreme_f32_axis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_argextreme_f32_axis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
a = rng.standard_normal((512, 512, 32)).astype(np.float32)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("argextreme f32 setup");
        let a = ns.get_item("a").expect("a");
        // f64 + int64 of the same shape: their non-last kernels were parallelized (were serial).
        let a64 = a.call_method1("astype", ("float64",)).expect("a64");
        let ai = a.call_method1("astype", ("int64",)).expect("ai");
        for name in ["argmin", "argmax"] {
            let fnp_fn = module.getattr(name).expect("fnp fn");
            let numpy_fn = numpy.getattr(name).expect("numpy fn");
            let kw = PyDict::new(py);
            kw.set_item("axis", 1_i64).unwrap();
            for (tag, arr) in [("f32", &a), ("f64", &a64), ("i64", &ai)] {
                let kwc = kw.clone();
                group.bench_function(format!("fnp_{name}_{tag}_mid"), |b| {
                    b.iter(|| black_box(fnp_fn.call((arr,), Some(&kw)).expect("fnp arg")));
                });
                group.bench_function(format!("numpy_{name}_{tag}_mid"), |b| {
                    b.iter(|| black_box(numpy_fn.call((arr,), Some(&kwc)).expect("np arg")));
                });
            }
        }
    });

    group.finish();
}

// np.nanargmin/nanargmax(f32/f64, NON-last axis): no native non-last nanarg kernel existed, so it
// fell to the extract path (f64 ~2.3x, f32 a 0.77x LOSS from the f32->f64 widen). numpy's nanarg
// copies NaN->-+inf then argmins (whole-array temp). The native per-column nan-skip arg wins ~40-53x.
fn bench_nanarg_nonlast_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_nanarg_nonlast_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
a = rng.standard_normal((4096, 512, 8)).astype(np.float32)\n\
a[a > 2.0] = np.nan\na[:, 0, :] = 0.5\n\
a64 = a.astype(np.float64)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("nanarg nonlast setup");
        let a = ns.get_item("a").expect("a");
        let a64 = ns.get_item("a64").expect("a64");
        for name in ["nanargmin", "nanargmax"] {
            let fnp_fn = module.getattr(name).expect("fnp fn");
            let numpy_fn = numpy.getattr(name).expect("numpy fn");
            let kw = PyDict::new(py);
            kw.set_item("axis", 1_i64).unwrap();
            for (tag, arr) in [("f32", &a), ("f64", &a64)] {
                let kwc = kw.clone();
                group.bench_function(format!("fnp_{name}_{tag}_mid"), |b| {
                    b.iter(|| black_box(fnp_fn.call((arr,), Some(&kw)).expect("fnp nanarg")));
                });
                group.bench_function(format!("numpy_{name}_{tag}_mid"), |b| {
                    b.iter(|| black_box(numpy_fn.call((arr,), Some(&kwc)).expect("np nanarg")));
                });
            }
        }
    });

    group.finish();
}

// np.nanargmin/nanargmax(f64, axis=-1): numpy copies the array replacing NaN with +-inf then
// argmins (~107-144ms@16M); the native fused single-pass per-lane nan-skip scan wins 10-46x.
// f64 previously had no last-axis path (only f32); this closes the gap.
fn bench_nanarg_lastaxis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_nanarg_lastaxis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
w = rng.standard_normal((8, 2_000_000))\nw[w > 2.0] = np.nan\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("nanarg setup");
        let w = ns.get_item("w").expect("w");
        for name in ["nanargmin", "nanargmax"] {
            let fnp_fn = module.getattr(name).expect("fnp fn");
            let numpy_fn = numpy.getattr(name).expect("numpy fn");
            let kw = PyDict::new(py);
            kw.set_item("axis", 1_i64).unwrap();
            let kw2 = kw.clone();
            group.bench_function(format!("fnp_{name}_f64_8x2m"), |b| {
                b.iter(|| black_box(fnp_fn.call((&w,), Some(&kw)).expect("fnp nanarg")));
            });
            group.bench_function(format!("numpy_{name}_f64_8x2m"), |b| {
                b.iter(|| black_box(numpy_fn.call((&w,), Some(&kw2)).expect("np nanarg")));
            });
        }
    });

    group.finish();
}

// np.lexsort(3 small-range int keys, 2M): numpy runs K sequential radix sorts; the packed-composite
// path does one parallel sort. Correctness gate (byte-identical to numpy) + timing.
fn bench_lexsort_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_lexsort_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
k0 = rng.integers(0, 100, 2_000_000).astype(np.int64)\n\
k1 = rng.integers(0, 100, 2_000_000).astype(np.int32)\n\
k2 = rng.integers(0, 100, 2_000_000).astype(np.int16)\n\
keys = (k0, k1, k2)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("lexsort setup");
        let keys = ns.get_item("keys").expect("keys");
        py.run(
            std::ffi::CString::new(
                "keys_f64 = (k0.astype(np.float64), k1.astype(np.float64), k2.astype(np.float64))",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("lexsort float setup");
        let keys_f64 = ns.get_item("keys_f64").expect("keys_f64");
        let fnp_lexsort = module.getattr("lexsort").expect("fnp lexsort");
        let numpy_lexsort = numpy.getattr("lexsort").expect("numpy lexsort");
        {
            let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
            let got = fnp_lexsort.call1((&keys,)).expect("fnp lexsort");
            let exp = numpy_lexsort.call1((&keys,)).expect("np lexsort");
            let eq: bool = np_array_equal
                .call1((&got, &exp))
                .expect("array_equal")
                .extract()
                .expect("bool");
            assert!(eq, "lexsort composite correctness mismatch");
            let got_f64 = fnp_lexsort.call1((&keys_f64,)).expect("fnp lexsort f64");
            let exp_f64 = numpy_lexsort.call1((&keys_f64,)).expect("np lexsort f64");
            let eq_f64: bool = np_array_equal
                .call1((&got_f64, &exp_f64))
                .expect("array_equal f64")
                .extract()
                .expect("bool");
            assert!(
                eq_f64,
                "lexsort integral-f64 composite correctness mismatch"
            );
        }
        group.bench_function("fnp_lexsort_3int_2m", |b| {
            b.iter(|| black_box(fnp_lexsort.call1((&keys,)).expect("fnp lexsort")));
        });
        group.bench_function("numpy_lexsort_3int_2m", |b| {
            b.iter(|| black_box(numpy_lexsort.call1((&keys,)).expect("numpy lexsort")));
        });
        group.bench_function("fnp_lexsort_3f64_intvalued_2m", |b| {
            b.iter(|| black_box(fnp_lexsort.call1((&keys_f64,)).expect("fnp lexsort f64")));
        });
        group.bench_function("numpy_lexsort_3f64_intvalued_2m", |b| {
            b.iter(|| {
                black_box(
                    numpy_lexsort
                        .call1((&keys_f64,))
                        .expect("numpy lexsort f64"),
                )
            });
        });
    });

    group.finish();
}

// np.unique(500k x 4 small-range int, axis=0): numpy sorts rows via a slow void comparator; the
// composite-pack path does one u64 sort+dedup+decode. Correctness gate (byte-identical) + timing.
fn bench_unique_rows_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_unique_rows_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
A = rng.integers(0, 20, (500_000, 4)).astype(np.int64)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("unique rows setup");
        let a = ns.get_item("A").expect("A");
        let fnp_unique = module.getattr("unique").expect("fnp unique");
        let numpy_unique = numpy.getattr("unique").expect("numpy unique");
        {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0).expect("axis");
            let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
            let got = fnp_unique.call((&a,), Some(&kw)).expect("fnp unique");
            let exp = numpy_unique.call((&a,), Some(&kw)).expect("np unique");
            let eq: bool = np_array_equal
                .call1((&got, &exp))
                .expect("array_equal")
                .extract()
                .expect("bool");
            assert!(eq, "unique(axis=0) composite correctness mismatch");
        }
        group.bench_function("fnp_unique_rows_500k4_axis0", |b| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0).expect("axis");
            b.iter(|| black_box(fnp_unique.call((&a,), Some(&kw)).expect("fnp unique")));
        });
        group.bench_function("numpy_unique_rows_500k4_axis0", |b| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 0).expect("axis");
            b.iter(|| black_box(numpy_unique.call((&a,), Some(&kw)).expect("numpy unique")));
        });
    });

    group.finish();
}

// np.unique(4 x 500k small-range int, axis=1): column-record sibling of the unique-rows composite
// pack. Correctness gate (byte-identical) + timing.
fn bench_unique_cols_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_unique_cols_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
A = rng.integers(0, 20, (4, 500_000)).astype(np.int64)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("unique cols setup");
        let a = ns.get_item("A").expect("A");
        let fnp_unique = module.getattr("unique").expect("fnp unique");
        let numpy_unique = numpy.getattr("unique").expect("numpy unique");
        {
            let kw = PyDict::new(py);
            kw.set_item("axis", 1).expect("axis");
            let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
            let got = fnp_unique.call((&a,), Some(&kw)).expect("fnp unique");
            let exp = numpy_unique.call((&a,), Some(&kw)).expect("np unique");
            let eq: bool = np_array_equal
                .call1((&got, &exp))
                .expect("array_equal")
                .extract()
                .expect("bool");
            assert!(eq, "unique(axis=1) composite correctness mismatch");
        }
        group.bench_function("fnp_unique_cols_4x500k_axis1", |b| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 1).expect("axis");
            b.iter(|| black_box(fnp_unique.call((&a,), Some(&kw)).expect("fnp unique")));
        });
        group.bench_function("numpy_unique_cols_4x500k_axis1", |b| {
            let kw = PyDict::new(py);
            kw.set_item("axis", 1).expect("axis");
            b.iter(|| black_box(numpy_unique.call((&a,), Some(&kw)).expect("numpy unique")));
        });
    });

    group.finish();
}

// np.unique(500k x 4 int, axis=0, return_index/inverse/counts): the group-by/factorize primitive; numpy
// does the slow void-row sort plus the extra outputs. Correctness gate (all outputs byte-identical) + timing.
fn bench_unique_rows_full_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_unique_rows_full_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\nrng = np.random.default_rng(0)\n\
A = rng.integers(0, 20, (500_000, 4)).astype(np.int64)\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("unique rows full setup");
        let a = ns.get_item("A").expect("A");
        let fnp_unique = module.getattr("unique").expect("fnp unique");
        let numpy_unique = numpy.getattr("unique").expect("numpy unique");
        let kw = PyDict::new(py);
        kw.set_item("axis", 0).expect("axis");
        kw.set_item("return_index", true).expect("ri");
        kw.set_item("return_inverse", true).expect("rinv");
        kw.set_item("return_counts", true).expect("rc");
        {
            let got = fnp_unique.call((&a,), Some(&kw)).expect("fnp unique full");
            let exp = numpy_unique.call((&a,), Some(&kw)).expect("np unique full");
            let np_array_equal = numpy.getattr("array_equal").expect("np.array_equal");
            // Compare each element of the returned tuple (unique, index, inverse, counts).
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
                    "unique(axis=0) return_* correctness mismatch at tuple index {t}"
                );
            }
        }
        group.bench_function("fnp_unique_rows_full_500k4", |b| {
            b.iter(|| black_box(fnp_unique.call((&a,), Some(&kw)).expect("fnp unique full")));
        });
        group.bench_function("numpy_unique_rows_full_500k4", |b| {
            b.iter(|| {
                black_box(
                    numpy_unique
                        .call((&a,), Some(&kw))
                        .expect("numpy unique full"),
                )
            });
        });
    });

    group.finish();
}

fn main() {
    common::gated_main(&[
        ("bench_parallel_binary_boundary", bench_parallel_binary_boundary),
        ("bench_sort_axis_boundary", bench_sort_axis_boundary),
        ("bench_sort_kind_boundary", bench_sort_kind_boundary),
        ("bench_matmul_boundary", bench_matmul_boundary),
        ("bench_char_add_boundary", bench_char_add_boundary),
        ("bench_asarray_dtype_boundary", bench_asarray_dtype_boundary),
        ("bench_timedelta_cumsum_boundary", bench_timedelta_cumsum_boundary),
        ("bench_datetime_minmax_boundary", bench_datetime_minmax_boundary),
        ("bench_datetime_ptp_boundary", bench_datetime_ptp_boundary),
        ("bench_datetime_argextreme_boundary", bench_datetime_argextreme_boundary),
        ("bench_argextreme_f32_axis_boundary", bench_argextreme_f32_axis_boundary),
        ("bench_nanarg_nonlast_boundary", bench_nanarg_nonlast_boundary),
        ("bench_nanarg_lastaxis_boundary", bench_nanarg_lastaxis_boundary),
        ("bench_lexsort_boundary", bench_lexsort_boundary),
        ("bench_unique_rows_boundary", bench_unique_rows_boundary),
        ("bench_unique_cols_boundary", bench_unique_cols_boundary),
        ("bench_unique_rows_full_boundary", bench_unique_rows_full_boundary),
    ]);
}
