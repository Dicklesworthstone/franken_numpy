#![feature(portable_simd)]

//! statistics / variance / nan-reduction / cumulative criterion benches —
//! statistics, cov, std and var across axes, nanvar/nansum/nanextreme (f32),
//! var axis-0, sum/prod/cumsum last-axis, cumsum-flat, and accumulate-extremum —
//! split out of the monolithic `criterion_python_surface.rs` into their own
//! per-domain bench binary. See bead deadlock-audit-x7nnf.

#[path = "common/mod.rs"]
mod common;

use common::ensure_numpy_available;
use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyDict, PyModule, PyTuple};
use rayon::prelude::*;
use std::collections::VecDeque;
use std::hint::black_box;
use std::simd::Simd;
use std::time::{Duration, Instant};

#[inline(never)]
fn cov_gram_schedule(
    centered: &[f64],
    n_vars: usize,
    n_obs: usize,
    paired: bool,
    result: &mut [f64],
) {
    type Lanes = Simd<f64, 8>;
    const MR: usize = 4;

    let inv_fact = 1.0_f64 / (n_obs - 1) as f64;
    let n_chunks = n_obs / 8;
    let tail = n_chunks * 8;
    let dot8 = |ci: &[f64], cj: &[f64]| -> f64 {
        let mut acc = [0.0_f64; 8];
        let (ca_chunks, ca_remainder) = ci.as_chunks::<8>();
        let (cb_chunks, cb_remainder) = cj.as_chunks::<8>();
        let mut ca = ca_chunks.iter();
        let mut cb = cb_chunks.iter();
        for (ga, gb) in ca.by_ref().zip(cb.by_ref()) {
            for lane in 0..8 {
                acc[lane] += ga[lane] * gb[lane];
            }
        }
        let mut sum =
            ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        for (&a, &b) in ca_remainder.iter().zip(cb_remainder) {
            sum += a * b;
        }
        sum
    };
    let finish = |acc: Lanes, tail_a: &[f64], tail_b: &[f64]| -> f64 {
        let lanes = acc.to_array();
        let mut sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        for (&a, &b) in tail_a.iter().zip(tail_b) {
            sum += a * b;
        }
        sum
    };
    let block = |i0: usize, rows: &mut [f64]| {
        let mr = rows.len() / n_vars;
        let row = |i: usize| &centered[i * n_obs..(i + 1) * n_obs];
        for j in 0..=i0 {
            let j_row = row(j);
            let mut acc = [Lanes::splat(0.0); MR];
            if mr == MR {
                let (r0, r1, r2, r3) = (row(i0), row(i0 + 1), row(i0 + 2), row(i0 + 3));
                for ((((b, a0), a1), a2), a3) in j_row[..tail]
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .zip(r0[..tail].as_chunks::<8>().0.iter())
                    .zip(r1[..tail].as_chunks::<8>().0.iter())
                    .zip(r2[..tail].as_chunks::<8>().0.iter())
                    .zip(r3[..tail].as_chunks::<8>().0.iter())
                {
                    let b = Lanes::from_slice(b);
                    acc[0] += Lanes::from_slice(a0) * b;
                    acc[1] += Lanes::from_slice(a1) * b;
                    acc[2] += Lanes::from_slice(a2) * b;
                    acc[3] += Lanes::from_slice(a3) * b;
                }
            } else {
                for chunk in 0..n_chunks {
                    let b = Lanes::from_slice(&j_row[chunk * 8..]);
                    for (m, lane_acc) in acc.iter_mut().enumerate().take(mr) {
                        *lane_acc +=
                            Lanes::from_slice(&centered[(i0 + m) * n_obs + chunk * 8..]) * b;
                    }
                }
            }
            for (m, lane_acc) in acc.iter().enumerate().take(mr) {
                let i = i0 + m;
                rows[m * n_vars + j] = finish(
                    *lane_acc,
                    &centered[i * n_obs + tail..(i + 1) * n_obs],
                    &j_row[tail..],
                ) * inv_fact;
            }
        }
        for m in 1..mr {
            let i = i0 + m;
            let i_row = row(i);
            for j in (i0 + 1)..=i {
                rows[m * n_vars + j] = dot8(i_row, row(j)) * inv_fact;
            }
        }
    };

    if paired {
        let mut blocks: VecDeque<(usize, &mut [f64])> =
            result.chunks_mut(n_vars * MR).enumerate().collect();
        let mut tasks = Vec::with_capacity(blocks.len().div_ceil(2));
        while let Some(heavy) = blocks.pop_back() {
            tasks.push((heavy, blocks.pop_front()));
        }
        tasks.into_par_iter().for_each(|((heavy, rows), light)| {
            block(heavy * MR, rows);
            if let Some((light, rows)) = light {
                block(light * MR, rows);
            }
        });
    } else {
        result
            .par_chunks_mut(n_vars * MR)
            .enumerate()
            .for_each(|(block_index, rows)| block(block_index * MR, rows));
    }
    for i in 0..n_vars {
        for j in (i + 1)..n_vars {
            result[i * n_vars + j] = result[j * n_vars + i];
        }
    }
}

fn bench_cov_gram_pairing_contract(c: &mut Criterion) {
    // Official ledger-resurrection rank 4. This reconstructs the archived
    // candidate exactly at the scheduler seam while retaining today's MR=4
    // strip kernel. No production dispatch changes: a decisive loss remains a
    // durable no-ship and a win would only authorize a separate source patch.
    const N_VARS: usize = 1_000;
    const N_OBS: usize = 1_000;

    let centered = (0..N_VARS * N_OBS)
        .map(|index| {
            let phase = (index % 8_191) as f64 * 0.000_976_562_5;
            phase.sin() - 0.25 * phase.cos()
        })
        .collect::<Vec<_>>();
    let mut former_result = vec![0.0; N_VARS * N_VARS];
    let mut candidate_result = vec![0.0; N_VARS * N_VARS];
    cov_gram_schedule(&centered, N_VARS, N_OBS, false, &mut former_result);
    cov_gram_schedule(&centered, N_VARS, N_OBS, true, &mut candidate_result);
    assert_eq!(
        former_result, candidate_result,
        "paired scheduling must preserve every covariance bit"
    );
    println!(
        "PARITY row=python_cov_gram_paired_schedule kind=f64_to_bits_exact cells={}",
        N_VARS * N_VARS
    );

    let time_former = || {
        let started = Instant::now();
        cov_gram_schedule(&centered, N_VARS, N_OBS, false, &mut former_result);
        let elapsed = started.elapsed();
        let checksum = former_result[0].to_bits()
            ^ former_result[N_VARS * N_VARS / 2].to_bits().rotate_left(17)
            ^ former_result[N_VARS * N_VARS - 1].to_bits().rotate_left(37);
        black_box(&former_result);
        common::ContractObservation { elapsed, checksum }
    };
    let time_candidate = || {
        let started = Instant::now();
        cov_gram_schedule(&centered, N_VARS, N_OBS, true, &mut candidate_result);
        let elapsed = started.elapsed();
        let checksum = candidate_result[0].to_bits()
            ^ candidate_result[N_VARS * N_VARS / 2]
                .to_bits()
                .rotate_left(17)
            ^ candidate_result[N_VARS * N_VARS - 1]
                .to_bits()
                .rotate_left(37);
        black_box(&candidate_result);
        common::ContractObservation { elapsed, checksum }
    };
    let _ = common::run_median_ci_contract(
        "python_cov_gram_paired_schedule",
        time_former,
        time_candidate,
    );

    let mut group = c.benchmark_group("python_cov_gram_pairing_contract");
    group.sample_size(10);
    group.measurement_time(Duration::from_millis(750));
    group.warm_up_time(Duration::from_millis(250));
    group.bench_function("former_contiguous_blocks_1000x1000", |bench| {
        bench.iter(|| {
            cov_gram_schedule(&centered, N_VARS, N_OBS, false, &mut former_result);
            black_box(former_result[N_VARS * N_VARS - 1]);
        });
    });
    group.bench_function("candidate_paired_blocks_1000x1000", |bench| {
        bench.iter(|| {
            cov_gram_schedule(&centered, N_VARS, N_OBS, true, &mut candidate_result);
            black_box(candidate_result[N_VARS * N_VARS - 1]);
        });
    });
    group.finish();
}

fn bench_statistics_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_statistics_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_cov = module.getattr("cov").expect("fnp_python.cov");
        let numpy_cov = numpy.getattr("cov").expect("numpy.cov");
        let fnp_corrcoef = module.getattr("corrcoef").expect("fnp_python.corrcoef");
        let numpy_corrcoef = numpy.getattr("corrcoef").expect("numpy.corrcoef");

        let make_input = |rows: usize, cols: usize| {
            let total = rows * cols;
            numpy
                .call_method1("linspace", (-2.0_f64, 3.0_f64, total))
                .expect("cov f64 input")
                .call_method1("reshape", ((rows, cols),))
                .expect("2-D cov input")
        };
        let inputs = [
            ("50x1000", make_input(50, 1000)),
            ("200x500", make_input(200, 500)),
            ("500x500", make_input(500, 500)),
            ("50x10000", make_input(50, 10_000)),
        ];

        for (shape, input) in inputs {
            group.bench_function(format!("fnp_cov_rowvar_f64_{shape}"), |bench| {
                bench.iter(|| {
                    let result = fnp_cov.call1((&input,)).expect("fnp cov benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("numpy_cov_rowvar_f64_{shape}"), |bench| {
                bench.iter(|| {
                    let result = numpy_cov
                        .call1((&input,))
                        .expect("numpy cov benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("fnp_corrcoef_rowvar_f64_{shape}"), |bench| {
                bench.iter(|| {
                    let result = fnp_corrcoef
                        .call1((&input,))
                        .expect("fnp corrcoef benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("numpy_corrcoef_rowvar_f64_{shape}"), |bench| {
                bench.iter(|| {
                    let result = numpy_corrcoef
                        .call1((&input,))
                        .expect("numpy corrcoef benchmark call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

// Large-n_vars / large-n_obs cov+corrcoef: the Gram-path shapes where the register
// tile and its output stages dominate. bench_statistics_boundary tops out at 500x500,
// which never leaves the small-shape gates, so CI was blind both to Gram-kernel
// regressions and to the fault-storm allocation mode documented in
// docs/NEGATIVE_EVIDENCE.md 2026-07-10 (30.5 MiB result buffers refaulted per call in
// unlucky builds -- these rows make that mode visible as an fnp-vs-numpy ratio shift).
// Conformance is embedded: the group panics if fnp and numpy diverge beyond 1e-12.
fn bench_cov_large_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_cov_large_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_cov = module.getattr("cov").expect("fnp_python.cov");
        let numpy_cov = numpy.getattr("cov").expect("numpy.cov");
        let fnp_corrcoef = module.getattr("corrcoef").expect("fnp_python.corrcoef");
        let numpy_corrcoef = numpy.getattr("corrcoef").expect("numpy.corrcoef");
        let np_allclose = numpy.getattr("allclose").expect("np.allclose");

        let rng = numpy
            .getattr("random")
            .expect("np.random")
            .call_method1("default_rng", (0_u64,))
            .expect("default_rng");
        let make_input = |rows: usize, cols: usize| {
            rng.call_method1("standard_normal", ((rows, cols),))
                .expect("standard_normal input")
        };
        let inputs = [
            ("2000x500", make_input(2000, 500)),
            ("1000x1000", make_input(1000, 1000)),
            ("500x5000", make_input(500, 5000)),
        ];

        let tol = PyDict::new(py);
        tol.set_item("rtol", 1e-12_f64).expect("rtol");
        tol.set_item("atol", 1e-14_f64).expect("atol");

        for (shape, input) in inputs {
            for (opname, fnp_op, numpy_op) in [
                ("cov", &fnp_cov, &numpy_cov),
                ("corrcoef", &fnp_corrcoef, &numpy_corrcoef),
            ] {
                let ours = fnp_op.call1((&input,)).expect("fnp result");
                let oracle = numpy_op.call1((&input,)).expect("numpy result");
                let close: bool = np_allclose
                    .call((&ours, &oracle), Some(&tol))
                    .expect("allclose call")
                    .extract()
                    .expect("allclose bool");
                assert!(close, "fnp.{opname} diverges from numpy at {shape}");

                group.bench_function(format!("fnp_{opname}_rowvar_f64_{shape}"), |bench| {
                    bench.iter(|| {
                        black_box(fnp_op.call1((&input,)).expect("fnp benchmark call"));
                    });
                });
                group.bench_function(format!("numpy_{opname}_rowvar_f64_{shape}"), |bench| {
                    bench.iter(|| {
                        black_box(numpy_op.call1((&input,)).expect("numpy benchmark call"));
                    });
                });
            }
        }
    });

    group.finish();
}

fn bench_std_var_axis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_std_var_axis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_std = module.getattr("std").expect("fnp_python.std");
        let numpy_std = numpy.getattr("std").expect("numpy.std");
        let fnp_var = module.getattr("var").expect("fnp_python.var");
        let numpy_var = numpy.getattr("var").expect("numpy.var");

        for (label, rows, cols) in [
            ("4096x512", 4096_i64, 512_i64),
            ("8192x1024", 8192_i64, 1024_i64),
        ] {
            let size = rows * cols;
            let input = numpy
                .call_method1("linspace", (-4.0_f64, 6.0_f64, size))
                .expect("std/var axis f64 input")
                .call_method1("reshape", ((rows, cols),))
                .expect("std/var axis 2-D shape");

            group.bench_function(format!("fnp_var_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_var
                        .call1((&input, -1_i64))
                        .expect("fnp var axis benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("numpy_var_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_var
                        .call1((&input, -1_i64))
                        .expect("numpy var axis benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("fnp_std_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_std
                        .call1((&input, -1_i64))
                        .expect("fnp std axis benchmark call");
                    black_box(result);
                });
            });

            group.bench_function(format!("numpy_std_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_std
                        .call1((&input, -1_i64))
                        .expect("numpy std axis benchmark call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

fn bench_var_multiaxis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_var_multiaxis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_var = module.getattr("var").expect("fnp_python.var");
        let numpy_var = numpy.getattr("var").expect("numpy.var");
        let fnp_std = module.getattr("std").expect("fnp_python.std");
        let numpy_std = numpy.getattr("std").expect("numpy.std");
        let fnp_nanvar = module.getattr("nanvar").expect("fnp_python.nanvar");
        let numpy_nanvar = numpy.getattr("nanvar").expect("numpy.nanvar");
        let fnp_nanstd = module.getattr("nanstd").expect("fnp_python.nanstd");
        let numpy_nanstd = numpy.getattr("nanstd").expect("numpy.nanstd");

        for (label, b, m, n) in [
            ("4096x16x16", 4096_i64, 16_i64, 16_i64),
            ("2048x32x32", 2048_i64, 32_i64, 32_i64),
        ] {
            let size = b * m * n;
            let input = numpy
                .call_method1("linspace", (-4.0_f64, 6.0_f64, size))
                .expect("var multiaxis f64 input")
                .call_method1("reshape", ((b, m, n),))
                .expect("var multiaxis 3-D shape");

            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs
                .set_item("axis", (-2_i64, -1_i64))
                .expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs
                .set_item("axis", (-2_i64, -1_i64))
                .expect("numpy axis kwarg");

            group.bench_function(format!("fnp_var_f64_axis_m2m1_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_var
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp var multiaxis call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_var_f64_axis_m2m1_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_var
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy var multiaxis call");
                    black_box(result);
                });
            });
            group.bench_function(format!("fnp_std_f64_axis_m2m1_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_std
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp std multiaxis call");
                    black_box(result);
                });
            });
            group.bench_function(format!("fnp_nanvar_f64_axis_m2m1_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_nanvar
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp nanvar multiaxis call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_nanvar_f64_axis_m2m1_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_nanvar
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy nanvar multiaxis call");
                    black_box(result);
                });
            });
            group.bench_function(format!("fnp_nanstd_f64_axis_m2m1_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_nanstd
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp nanstd multiaxis call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_nanstd_f64_axis_m2m1_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_nanstd
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy nanstd multiaxis call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_std_f64_axis_m2m1_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_std
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy std multiaxis call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

// var/std along a MIDDLE axis (0 < ax < ndim-1) of a 3-D f64 stack. numpy reduces a
// non-last axis with a strided, two-temp-materializing pass; the native block-parallel
// streaming two-pass (try_zerocopy_f64_var_nonlast_axis) is bit-exact and much faster.
fn bench_var_midaxis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_var_midaxis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_var = module.getattr("var").expect("fnp_python.var");
        let numpy_var = numpy.getattr("var").expect("numpy.var");
        let fnp_std = module.getattr("std").expect("fnp_python.std");
        let numpy_std = numpy.getattr("std").expect("numpy.std");

        for (label, d0, d1, d2) in [
            ("256x256x64", 256_usize, 256_usize, 64_usize),
            ("128x512x64", 128_usize, 512_usize, 64_usize),
        ] {
            let size = (d0 * d1 * d2) as i64;
            let input = numpy
                .call_method1("linspace", (-4.0_f64, 6.0_f64, size))
                .expect("var midaxis f64 input")
                .call_method1("reshape", ((d0, d1, d2),))
                .expect("var midaxis 3-D shape");
            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("axis", 1_i64).expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs
                .set_item("axis", 1_i64)
                .expect("numpy axis kwarg");

            group.bench_function(format!("fnp_var_f64_axis1_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_var
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp var axis1 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_var_f64_axis1_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_var
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy var axis1 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("fnp_std_f64_axis1_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_std
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp std axis1 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_std_f64_axis1_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_std
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy std axis1 call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

// FLOAT32 var/std along a non-last axis (middle ax=1 + axis 0). numpy keeps the float32
// accumulator and on a non-last axis reduces SEQUENTIALLY while materializing the (a-mean)
// and (a-mean)^2 whole-array f32 temps (~28ms@8M middle); try_zerocopy_f32_var_nonlast_axis
// runs a per-block sequential f32 two-pass (block-parallel for a middle axis, serial for
// axis 0) with no temp -> bit-identical and 3-33x faster. The f32 sibling of the f64
// midaxis/axis0 paths above.
fn bench_var_f32_axis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_var_f32_axis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let f32_dtype = numpy.getattr("float32").expect("numpy.float32");
        let fnp_var = module.getattr("var").expect("fnp_python.var");
        let numpy_var = numpy.getattr("var").expect("numpy.var");
        let fnp_std = module.getattr("std").expect("fnp_python.std");
        let numpy_std = numpy.getattr("std").expect("numpy.std");

        // (label, shape, reduce-axis): a middle axis (block-parallel) and axis 0 (serial).
        let mid = numpy
            .call_method1("linspace", (-4.0_f64, 6.0_f64, 256_i64 * 128 * 256))
            .expect("f32 mid source")
            .call_method1("reshape", ((256_usize, 128_usize, 256_usize),))
            .expect("256x128x256 reshape")
            .call_method1("astype", (&f32_dtype,))
            .expect("astype f32");
        let ax0 = numpy
            .call_method1("linspace", (-4.0_f64, 6.0_f64, 4000_i64 * 2000))
            .expect("f32 ax0 source")
            .call_method1("reshape", ((4000_usize, 2000_usize),))
            .expect("4000x2000 reshape")
            .call_method1("astype", (&f32_dtype,))
            .expect("astype f32");

        for (label, input, axis) in [
            ("mid_256x128x256", &mid, 1_i64),
            ("axis0_4000x2000", &ax0, 0_i64),
        ] {
            let fkw = PyDict::new(py);
            fkw.set_item("axis", axis).expect("axis");
            let nkw = PyDict::new(py);
            nkw.set_item("axis", axis).expect("axis");
            group.bench_function(format!("fnp_var_f32_{label}"), |bench| {
                bench.iter(|| black_box(fnp_var.call((input,), Some(&fkw)).expect("fnp var f32")));
            });
            group.bench_function(format!("numpy_var_f32_{label}"), |bench| {
                bench.iter(|| {
                    black_box(numpy_var.call((input,), Some(&nkw)).expect("numpy var f32"))
                });
            });
            group.bench_function(format!("fnp_std_f32_{label}"), |bench| {
                bench.iter(|| black_box(fnp_std.call((input,), Some(&fkw)).expect("fnp std f32")));
            });
            group.bench_function(format!("numpy_std_f32_{label}"), |bench| {
                bench.iter(|| {
                    black_box(numpy_std.call((input,), Some(&nkw)).expect("numpy std f32"))
                });
            });
        }
    });

    group.finish();
}

// FLOAT32 nanvar/nanstd along a non-last axis (middle ax=1 + axis 0) of a 3-D/2-D stack
// with ~10% NaN. numpy.nanvar on float32 keeps the f32 accumulator and materializes a
// NaN->0 copy, an isnan mask, a count, and the (a-mean)/squared f32 temps before two
// sequential strided reduces (~70-77ms@8M middle); try_zerocopy_f32_nanvar_nonlast_axis
// runs a per-block sequential f32 NaN-skip two-pass (block-parallel middle / serial axis0)
// with no temp -> bit-identical and 5-35x faster. f32 sibling of the f64 nanvar paths.
// np.nanmax/nanmin(f32, axis): f32 had no nanextreme-axis kernel (only f64+f16), so with NaN
// present it delegated to numpy which materializes a temp (~80ms@16M). The f32 twin (scalar
// f32::max/min skip-NaN fold, parallel) wins ~18x.
fn bench_nanextreme_f32_axis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_nanextreme_f32_axis_boundary");
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
a[a > 2.0] = np.nan\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("nanextreme f32 setup");
        let a = ns.get_item("a").expect("a");
        for name in ["nanmax", "nanmin"] {
            let fnp_fn = module.getattr(name).expect("fnp fn");
            let numpy_fn = numpy.getattr(name).expect("numpy fn");
            let kw = PyDict::new(py);
            kw.set_item("axis", 1_i64).unwrap();
            let kw2 = kw.clone();
            group.bench_function(format!("fnp_{name}_f32_mid"), |b| {
                b.iter(|| black_box(fnp_fn.call((&a,), Some(&kw)).expect("fnp nanext")));
            });
            group.bench_function(format!("numpy_{name}_f32_mid"), |b| {
                b.iter(|| black_box(numpy_fn.call((&a,), Some(&kw2)).expect("np nanext")));
            });
        }
    });

    group.finish();
}

// np.nansum/nanprod(f32, non-last axis): f32 delegated to numpy's temp-materializing nansum
// (copy + isnan + reduce, ~34ms@8M) while f64 had a kernel. The f32 twin (sequential per-block,
// parallel over outer blocks) avoids the temp AND parallelizes -> ~53x.
fn bench_nansum_f32_axis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_nansum_f32_axis_boundary");
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
a = rng.standard_normal((512, 512, 32)).astype(np.float32)\n\
a[a > 2.0] = np.nan\n",
            )
            .unwrap()
            .as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("nansum f32 setup");
        let a = ns.get_item("a").expect("a");
        let a64 = a.call_method1("astype", ("float64",)).expect("a64");
        for name in ["nansum", "nanprod"] {
            let fnp_fn = module.getattr(name).expect("fnp fn");
            let numpy_fn = numpy.getattr(name).expect("numpy fn");
            let kw = PyDict::new(py);
            kw.set_item("axis", 1_i64).unwrap();
            let kw2 = kw.clone();
            group.bench_function(format!("fnp_{name}_f32_mid"), |b| {
                b.iter(|| black_box(fnp_fn.call((&a,), Some(&kw)).expect("fnp nan")));
            });
            group.bench_function(format!("numpy_{name}_f32_mid"), |b| {
                b.iter(|| black_box(numpy_fn.call((&a,), Some(&kw2)).expect("np nan")));
            });
        }
        // f64 nansum non-last: the SERIAL branch was parallelized (its sibling nanprod was already
        // parallel) -> ~12x (temp-avoidance) becomes ~40x.
        let fnp_nansum = module.getattr("nansum").expect("fnp nansum");
        let numpy_nansum = numpy.getattr("nansum").expect("numpy nansum");
        let kw = PyDict::new(py);
        kw.set_item("axis", 1_i64).unwrap();
        let kw2 = kw.clone();
        group.bench_function("fnp_nansum_f64_mid", |b| {
            b.iter(|| black_box(fnp_nansum.call((&a64,), Some(&kw)).expect("fnp nansum f64")));
        });
        group.bench_function("numpy_nansum_f64_mid", |b| {
            b.iter(|| {
                black_box(
                    numpy_nansum
                        .call((&a64,), Some(&kw2))
                        .expect("np nansum f64"),
                )
            });
        });
        // f64 nansum LAST axis: per-lane pairwise (now bit-exact, was sequential) + parallel.
        let lax = PyDict::new(py);
        lax.set_item("axis", 2_i64).unwrap();
        let lax2 = lax.clone();
        group.bench_function("fnp_nansum_f64_last", |b| {
            b.iter(|| {
                black_box(
                    fnp_nansum
                        .call((&a64,), Some(&lax))
                        .expect("fnp nansum f64 last"),
                )
            });
        });
        group.bench_function("numpy_nansum_f64_last", |b| {
            b.iter(|| {
                black_box(
                    numpy_nansum
                        .call((&a64,), Some(&lax2))
                        .expect("np nansum f64 last"),
                )
            });
        });
    });

    group.finish();
}

fn bench_nanvar_f32_axis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_nanvar_f32_axis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let f32_dtype = numpy.getattr("float32").expect("numpy.float32");
        let nan = numpy.getattr("nan").expect("np.nan");
        let fnp_nanvar = module.getattr("nanvar").expect("fnp_python.nanvar");
        let numpy_nanvar = numpy.getattr("nanvar").expect("numpy.nanvar");
        let fnp_nanstd = module.getattr("nanstd").expect("fnp_python.nanstd");
        let numpy_nanstd = numpy.getattr("nanstd").expect("numpy.nanstd");
        let fnp_nanmean = module.getattr("nanmean").expect("fnp_python.nanmean");
        let numpy_nanmean = numpy.getattr("nanmean").expect("numpy.nanmean");

        // Build an f32 array with ~10% NaN (deterministic stride), reshape to target.
        let build = |dims: &[usize], total: i64| {
            let arr = numpy
                .call_method1("linspace", (-4.0_f64, 6.0_f64, total))
                .expect("f32 nan source")
                .call_method1("astype", (&f32_dtype,))
                .expect("astype f32");
            let idx = numpy
                .call_method1("arange", (0_i64, total, 10_i64))
                .expect("nan stride");
            arr.call_method1("__setitem__", (idx, &nan))
                .expect("inject NaN");
            arr.call_method1(
                "reshape",
                (PyTuple::new(py, dims.iter().copied()).unwrap(),),
            )
            .expect("reshape")
        };
        let mid = build(&[256, 128, 256], 256 * 128 * 256);
        let ax0 = build(&[4000, 2000], 4000 * 2000);

        for (label, input, axis) in [
            ("mid_256x128x256", &mid, 1_i64),
            ("axis0_4000x2000", &ax0, 0_i64),
        ] {
            let fkw = PyDict::new(py);
            fkw.set_item("axis", axis).expect("axis");
            let nkw = PyDict::new(py);
            nkw.set_item("axis", axis).expect("axis");
            group.bench_function(format!("fnp_nanvar_f32_{label}"), |b| {
                b.iter(|| {
                    black_box(
                        fnp_nanvar
                            .call((input,), Some(&fkw))
                            .expect("fnp nanvar f32"),
                    )
                });
            });
            group.bench_function(format!("numpy_nanvar_f32_{label}"), |b| {
                b.iter(|| {
                    black_box(
                        numpy_nanvar
                            .call((input,), Some(&nkw))
                            .expect("numpy nanvar f32"),
                    )
                });
            });
            group.bench_function(format!("fnp_nanstd_f32_{label}"), |b| {
                b.iter(|| {
                    black_box(
                        fnp_nanstd
                            .call((input,), Some(&fkw))
                            .expect("fnp nanstd f32"),
                    )
                });
            });
            group.bench_function(format!("numpy_nanstd_f32_{label}"), |b| {
                b.iter(|| {
                    black_box(
                        numpy_nanstd
                            .call((input,), Some(&nkw))
                            .expect("numpy nanstd f32"),
                    )
                });
            });
            group.bench_function(format!("fnp_nanmean_f32_{label}"), |b| {
                b.iter(|| {
                    black_box(
                        fnp_nanmean
                            .call((input,), Some(&fkw))
                            .expect("fnp nanmean f32"),
                    )
                });
            });
            group.bench_function(format!("numpy_nanmean_f32_{label}"), |b| {
                b.iter(|| {
                    black_box(
                        numpy_nanmean
                            .call((input,), Some(&nkw))
                            .expect("numpy nanmean f32"),
                    )
                });
            });
        }
    });

    group.finish();
}

// nanvar/nanstd/nanmean along the CONTIGUOUS LAST axis (and a trailing tuple) of an f32
// stack with ~10% NaN. numpy.nanmean/nanvar on float32 materializes a NaN->0 copy + isnan
// mask then PAIRWISE-reduces the last axis (~3-7ms/2M unloaded, far worse loaded); the
// native per-lane bit-exact f32 pairwise paths (try_zerocopy_f32_nanmean_last_axis /
// try_zerocopy_f32_nanvar_last_axis) parallelize across the independent lanes.
fn bench_nanvar_f32_last_axis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_nanvar_f32_last_axis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let f32_dtype = numpy.getattr("float32").expect("numpy.float32");
        let nan = numpy.getattr("nan").expect("np.nan");
        let fnp_nanvar = module.getattr("nanvar").expect("fnp_python.nanvar");
        let numpy_nanvar = numpy.getattr("nanvar").expect("numpy.nanvar");
        let fnp_nanstd = module.getattr("nanstd").expect("fnp_python.nanstd");
        let numpy_nanstd = numpy.getattr("nanstd").expect("numpy.nanstd");
        let fnp_nanmean = module.getattr("nanmean").expect("fnp_python.nanmean");
        let numpy_nanmean = numpy.getattr("nanmean").expect("numpy.nanmean");

        let build = |dims: &[usize], total: i64| {
            let arr = numpy
                .call_method1("linspace", (-4.0_f64, 6.0_f64, total))
                .expect("f32 nan source")
                .call_method1("astype", (&f32_dtype,))
                .expect("astype f32");
            let idx = numpy
                .call_method1("arange", (0_i64, total, 10_i64))
                .expect("nan stride");
            arr.call_method1("__setitem__", (idx, &nan))
                .expect("inject NaN");
            arr.call_method1(
                "reshape",
                (PyTuple::new(py, dims.iter().copied()).unwrap(),),
            )
            .expect("reshape")
        };
        let last2d = build(&[1000, 2048], 1000 * 2048);
        let trail3d = build(&[512, 64, 64], 512 * 64 * 64);

        // Per-case kwargs dicts (axis=-1 for the single last axis; axis=(-2,-1) for the
        // contiguous trailing tuple). Built up front so the case loop is homogeneous.
        let fkw_last = PyDict::new(py);
        fkw_last.set_item("axis", -1_i64).expect("axis");
        let fkw_trail = PyDict::new(py);
        fkw_trail.set_item("axis", (-2_i64, -1_i64)).expect("axis");
        let cases = [
            ("last_1000x2048", &last2d, &fkw_last),
            ("trail_512x64x64", &trail3d, &fkw_trail),
        ];
        for (label, input, kw) in cases {
            // Owned handles so the `Some(&fkw)` call sites below are a single
            // borrow (kw is already `&Bound`); clone is a refcount bump in
            // setup, outside every timed `b.iter` closure.
            let fkw = kw.clone();
            let nkw = kw.clone();
            group.bench_function(format!("fnp_nanvar_f32_{label}"), |b| {
                b.iter(|| {
                    black_box(
                        fnp_nanvar
                            .call((input,), Some(&fkw))
                            .expect("fnp nanvar f32"),
                    )
                });
            });
            group.bench_function(format!("numpy_nanvar_f32_{label}"), |b| {
                b.iter(|| {
                    black_box(
                        numpy_nanvar
                            .call((input,), Some(&nkw))
                            .expect("numpy nanvar f32"),
                    )
                });
            });
            group.bench_function(format!("fnp_nanstd_f32_{label}"), |b| {
                b.iter(|| {
                    black_box(
                        fnp_nanstd
                            .call((input,), Some(&fkw))
                            .expect("fnp nanstd f32"),
                    )
                });
            });
            group.bench_function(format!("numpy_nanstd_f32_{label}"), |b| {
                b.iter(|| {
                    black_box(
                        numpy_nanstd
                            .call((input,), Some(&nkw))
                            .expect("numpy nanstd f32"),
                    )
                });
            });
            group.bench_function(format!("fnp_nanmean_f32_{label}"), |b| {
                b.iter(|| {
                    black_box(
                        fnp_nanmean
                            .call((input,), Some(&fkw))
                            .expect("fnp nanmean f32"),
                    )
                });
            });
            group.bench_function(format!("numpy_nanmean_f32_{label}"), |b| {
                b.iter(|| {
                    black_box(
                        numpy_nanmean
                            .call((input,), Some(&nkw))
                            .expect("numpy nanmean f32"),
                    )
                });
            });
        }
    });

    group.finish();
}

// nanvar/nanstd along a MIDDLE axis (0 < ax < ndim-1) of a 3-D f64 stack with scattered
// NaN. numpy.nanvar on a non-last axis materializes a NaN->0 copy, an isnan mask, a count,
// and the (a-mean)/squared temps then strided-reduces; the native block-parallel NaN-skip
// two-pass (try_zerocopy_f64_nanvar_nonlast_axis) is bit-exact and far faster.
fn bench_nanvar_midaxis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_nanvar_midaxis_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_nanvar = module.getattr("nanvar").expect("fnp_python.nanvar");
        let numpy_nanvar = numpy.getattr("nanvar").expect("numpy.nanvar");
        let fnp_nanstd = module.getattr("nanstd").expect("fnp_python.nanstd");
        let numpy_nanstd = numpy.getattr("nanstd").expect("numpy.nanstd");
        let fnp_nanmean = module.getattr("nanmean").expect("fnp_python.nanmean");
        let numpy_nanmean = numpy.getattr("nanmean").expect("numpy.nanmean");

        for (label, d0, d1, d2) in [
            ("256x256x64", 256_usize, 256_usize, 64_usize),
            ("128x512x64", 128_usize, 512_usize, 64_usize),
        ] {
            let size = (d0 * d1 * d2) as i64;
            // Build a 3-D f64 array, then poke ~10% NaN into it (deterministic stride).
            let input = numpy
                .call_method1("linspace", (-4.0_f64, 6.0_f64, size))
                .expect("nanvar midaxis f64 input")
                .call_method1("reshape", ((d0, d1, d2),))
                .expect("nanvar midaxis 3-D shape");
            let flat = input
                .call_method1("reshape", ((size,),))
                .expect("flat view");
            let idx = numpy
                .call_method1("arange", (0_i64, size, 10_i64))
                .expect("nan index stride");
            let nan = numpy.getattr("nan").expect("np.nan");
            flat.call_method1("__setitem__", (idx, nan))
                .expect("inject NaN");

            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("axis", 1_i64).expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs
                .set_item("axis", 1_i64)
                .expect("numpy axis kwarg");

            group.bench_function(format!("fnp_nanvar_f64_axis1_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_nanvar
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp nanvar axis1 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_nanvar_f64_axis1_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_nanvar
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy nanvar axis1 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("fnp_nanstd_f64_axis1_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_nanstd
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp nanstd axis1 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_nanstd_f64_axis1_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_nanstd
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy nanstd axis1 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("fnp_nanmean_f64_axis1_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_nanmean
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp nanmean axis1 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_nanmean_f64_axis1_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_nanmean
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy nanmean axis1 call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

fn bench_var_axis0_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_var_axis0_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_var = module.getattr("var").expect("fnp_python.var");
        let numpy_var = numpy.getattr("var").expect("numpy.var");
        let fnp_std = module.getattr("std").expect("fnp_python.std");
        let numpy_std = numpy.getattr("std").expect("numpy.std");
        let fnp_nanvar = module.getattr("nanvar").expect("fnp_python.nanvar");
        let numpy_nanvar = numpy.getattr("nanvar").expect("numpy.nanvar");
        let fnp_nanstd = module.getattr("nanstd").expect("fnp_python.nanstd");
        let numpy_nanstd = numpy.getattr("nanstd").expect("numpy.nanstd");
        let fnp_nanmean = module.getattr("nanmean").expect("fnp_python.nanmean");
        let numpy_nanmean = numpy.getattr("nanmean").expect("numpy.nanmean");

        for (label, rows, cols) in [
            ("4096x512", 4096_i64, 512_i64),
            ("50000x64", 50000_i64, 64_i64),
        ] {
            let size = rows * cols;
            let input = numpy
                .call_method1("linspace", (-4.0_f64, 6.0_f64, size))
                .expect("var axis0 f64 input")
                .call_method1("reshape", ((rows, cols),))
                .expect("var axis0 2-D shape");

            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("axis", 0_i64).expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs
                .set_item("axis", 0_i64)
                .expect("numpy axis kwarg");

            group.bench_function(format!("fnp_var_f64_axis0_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_var
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp var axis0 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_var_f64_axis0_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_var
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy var axis0 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("fnp_std_f64_axis0_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_std
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp std axis0 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_std_f64_axis0_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_std
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy std axis0 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("fnp_nanvar_f64_axis0_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_nanvar
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp nanvar axis0 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_nanvar_f64_axis0_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_nanvar
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy nanvar axis0 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("fnp_nanstd_f64_axis0_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_nanstd
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp nanstd axis0 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_nanstd_f64_axis0_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_nanstd
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy nanstd axis0 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("fnp_nanmean_f64_axis0_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_nanmean
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp nanmean axis0 call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_nanmean_f64_axis0_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_nanmean
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy nanmean axis0 call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

fn bench_sum_lastaxis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_sum_lastaxis_boundary");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_sum = module.getattr("sum").expect("fnp_python.sum");
        let numpy_sum = numpy.getattr("sum").expect("numpy.sum");

        for (label, rows, cols) in [
            ("8192x1024", 8192_i64, 1024_i64),
            ("65536x256", 65536_i64, 256_i64),
        ] {
            let size = rows * cols;
            let input = numpy
                .call_method1("linspace", (-2.0_f64, 3.0_f64, size))
                .expect("sum input")
                .call_method1("reshape", ((rows, cols),))
                .expect("sum 2-D shape");
            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("axis", -1_i64).expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs
                .set_item("axis", -1_i64)
                .expect("numpy axis kwarg");

            group.bench_function(format!("fnp_sum_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_sum
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp sum call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_sum_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_sum
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy sum call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

fn bench_prod_lastaxis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_prod_lastaxis_boundary");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_prod = module.getattr("prod").expect("fnp_python.prod");
        let numpy_prod = numpy.getattr("prod").expect("numpy.prod");

        for (label, rows, cols) in [
            ("8192x1024", 8192_i64, 1024_i64),
            ("65536x256", 65536_i64, 256_i64),
        ] {
            let size = rows * cols;
            // values near 1.0 so the product stays finite across the axis.
            let input = numpy
                .call_method1("linspace", (0.9999_f64, 1.0001_f64, size))
                .expect("prod input")
                .call_method1("reshape", ((rows, cols),))
                .expect("prod 2-D shape");
            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("axis", -1_i64).expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs
                .set_item("axis", -1_i64)
                .expect("numpy axis kwarg");

            group.bench_function(format!("fnp_prod_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_prod
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp prod call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_prod_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_prod
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy prod call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

fn bench_cumsum_lastaxis_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_cumsum_lastaxis_boundary");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_cumsum = module.getattr("cumsum").expect("fnp_python.cumsum");
        let numpy_cumsum = numpy.getattr("cumsum").expect("numpy.cumsum");

        for (label, rows, cols) in [
            ("8192x1024", 8192_i64, 1024_i64),
            ("65536x256", 65536_i64, 256_i64),
        ] {
            let size = rows * cols;
            let input = numpy
                .call_method1("linspace", (-1.0_f64, 1.0_f64, size))
                .expect("cumsum input")
                .call_method1("reshape", ((rows, cols),))
                .expect("cumsum 2-D shape");
            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs.set_item("axis", -1_i64).expect("fnp axis kwarg");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs
                .set_item("axis", -1_i64)
                .expect("numpy axis kwarg");

            group.bench_function(format!("fnp_cumsum_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = fnp_cumsum
                        .call((&input,), Some(&fnp_kwargs))
                        .expect("fnp cumsum call");
                    black_box(result);
                });
            });
            group.bench_function(format!("numpy_cumsum_f64_axis_last_{label}"), |bench| {
                bench.iter(|| {
                    let result = numpy_cumsum
                        .call((&input,), Some(&numpy_kwargs))
                        .expect("numpy cumsum call");
                    black_box(result);
                });
            });
        }
    });

    group.finish();
}

fn bench_cumsum_flat_boundary(c: &mut Criterion) {
    // FLAT 1-D integer np.cumsum(8M) — a single-lane prefix sum. numpy's 1-D cumsum is
    // a serial dependency chain; the native two-pass block scan breaks it across cores
    // (bit-exact for wrapping integer add). Was serial (parity with numpy).
    let mut group = c.benchmark_group("python_cumsum_flat_boundary");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bench").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let fnp_cumsum = module.getattr("cumsum").expect("fnp cumsum");
        let numpy_cumsum = numpy.getattr("cumsum").expect("numpy cumsum");
        let i64_in = numpy
            .call_method1("arange", (8_000_000_i64,))
            .expect("8M i64");
        let i32_in = i64_in.call_method1("astype", ("int32",)).expect("i32");
        group.bench_function("fnp_cumsum_i64_flat_8m", |b| {
            b.iter(|| black_box(fnp_cumsum.call1((&i64_in,)).expect("fnp cumsum i64")));
        });
        group.bench_function("numpy_cumsum_i64_flat_8m", |b| {
            b.iter(|| black_box(numpy_cumsum.call1((&i64_in,)).expect("numpy cumsum i64")));
        });
        group.bench_function("fnp_cumsum_i32_flat_8m", |b| {
            b.iter(|| black_box(fnp_cumsum.call1((&i32_in,)).expect("fnp cumsum i32")));
        });
        group.bench_function("numpy_cumsum_i32_flat_8m", |b| {
            b.iter(|| black_box(numpy_cumsum.call1((&i32_in,)).expect("numpy cumsum i32")));
        });
    });

    group.finish();
}

fn bench_accumulate_extremum_boundary(c: &mut Criterion) {
    // FLAT 1-D f64 np.maximum.accumulate(8M) — running max. numpy delegates to a serial
    // prefix scan (dependency chain); the native two-pass parallel prefix breaks it.
    let mut group = c.benchmark_group("python_accumulate_extremum_boundary");
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
x = rng.standard_normal(8_000_000)\n";
        let ns = PyDict::new(py);
        py.run(
            std::ffi::CString::new(setup).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )
        .expect("accumulate setup");
        let x = ns.get_item("x").expect("x");
        let fnp_max = module.getattr("maximum").expect("fnp maximum");
        let numpy_max = numpy.getattr("maximum").expect("numpy maximum");
        group.bench_function("fnp_maximum_accumulate_f64_8m", |b| {
            b.iter(|| {
                black_box(
                    fnp_max
                        .call_method1("accumulate", (&x,))
                        .expect("fnp max.accum"),
                )
            });
        });
        group.bench_function("numpy_maximum_accumulate_f64_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_max
                        .call_method1("accumulate", (&x,))
                        .expect("numpy max.accum"),
                )
            });
        });

        // f32 + i64 running max share the generic two-pass (bit-exact: max/min
        // associative for float, no NaN/promotion for int).
        let x32 = x.call_method1("astype", ("float32",)).expect("x32");
        let xi = numpy
            .call_method1("arange", (8_000_000_i64,))
            .expect("8M i64 base")
            .call_method1("__mod__", (1_000_003_i64,))
            .expect("xi");
        group.bench_function("fnp_maximum_accumulate_f32_8m", |b| {
            b.iter(|| {
                black_box(
                    fnp_max
                        .call_method1("accumulate", (&x32,))
                        .expect("fnp max.accum f32"),
                )
            });
        });
        group.bench_function("numpy_maximum_accumulate_f32_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_max
                        .call_method1("accumulate", (&x32,))
                        .expect("np max.accum f32"),
                )
            });
        });
        group.bench_function("fnp_maximum_accumulate_i64_8m", |b| {
            b.iter(|| {
                black_box(
                    fnp_max
                        .call_method1("accumulate", (&xi,))
                        .expect("fnp max.accum i64"),
                )
            });
        });
        group.bench_function("numpy_maximum_accumulate_i64_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_max
                        .call_method1("accumulate", (&xi,))
                        .expect("np max.accum i64"),
                )
            });
        });

        // add.accumulate(int) routes to the parallel cumsum path (== np.cumsum); a win
        // here proves the routing engages (vs the prior full delegation to numpy serial).
        let fnp_add = module.getattr("add").expect("fnp add");
        let numpy_add = numpy.getattr("add").expect("numpy add");
        let xa = numpy
            .call_method1("arange", (8_000_000_i64,))
            .expect("8M i64 arange");
        group.bench_function("fnp_add_accumulate_i64_8m", |b| {
            b.iter(|| {
                black_box(
                    fnp_add
                        .call_method1("accumulate", (&xa,))
                        .expect("fnp add.accum i64"),
                )
            });
        });
        group.bench_function("numpy_add_accumulate_i64_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_add
                        .call_method1("accumulate", (&xa,))
                        .expect("np add.accum i64"),
                )
            });
        });

        // bitwise_or.accumulate(int) native two-pass prefix vs numpy serial.
        let fnp_or = module.getattr("bitwise_or").expect("fnp bitwise_or");
        let numpy_or = numpy.getattr("bitwise_or").expect("numpy bitwise_or");
        group.bench_function("fnp_bitwise_or_accumulate_i64_8m", |b| {
            b.iter(|| {
                black_box(
                    fnp_or
                        .call_method1("accumulate", (&xi,))
                        .expect("fnp or.accum i64"),
                )
            });
        });
        group.bench_function("numpy_bitwise_or_accumulate_i64_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_or
                        .call_method1("accumulate", (&xi,))
                        .expect("np or.accum i64"),
                )
            });
        });

        // logical_and/or/xor.accumulate(bool): numpy runs a serial dependency-chain scan
        // (~40ms/16M). Bool logical == bitwise (0/1 values), routed to the proven two-pass
        // bitwise prefix. Mask is ~86% True (realistic, avoids AND collapsing to all-False).
        let xb = numpy
            .call_method1("arange", (8_000_000_i64,))
            .expect("8M i64 arange for bool")
            .call_method1("__mod__", (7_i64,))
            .expect("mod 7")
            .call_method1("__ne__", (0_i64,))
            .expect("bool mask");
        let fnp_land = module.getattr("logical_and").expect("fnp logical_and");
        let numpy_land = numpy.getattr("logical_and").expect("numpy logical_and");
        group.bench_function("fnp_logical_and_accumulate_bool_8m", |b| {
            b.iter(|| {
                black_box(
                    fnp_land
                        .call_method1("accumulate", (&xb,))
                        .expect("fnp land.accum bool"),
                )
            });
        });
        group.bench_function("numpy_logical_and_accumulate_bool_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_land
                        .call_method1("accumulate", (&xb,))
                        .expect("np land.accum bool"),
                )
            });
        });
        let fnp_lor = module.getattr("logical_or").expect("fnp logical_or");
        let numpy_lor = numpy.getattr("logical_or").expect("numpy logical_or");
        group.bench_function("fnp_logical_or_accumulate_bool_8m", |b| {
            b.iter(|| {
                black_box(
                    fnp_lor
                        .call_method1("accumulate", (&xb,))
                        .expect("fnp lor.accum bool"),
                )
            });
        });
        group.bench_function("numpy_logical_or_accumulate_bool_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_lor
                        .call_method1("accumulate", (&xb,))
                        .expect("np lor.accum bool"),
                )
            });
        });
        let fnp_lxor = module.getattr("logical_xor").expect("fnp logical_xor");
        let numpy_lxor = numpy.getattr("logical_xor").expect("numpy logical_xor");
        group.bench_function("fnp_logical_xor_accumulate_bool_8m", |b| {
            b.iter(|| {
                black_box(
                    fnp_lxor
                        .call_method1("accumulate", (&xb,))
                        .expect("fnp lxor.accum bool"),
                )
            });
        });
        group.bench_function("numpy_logical_xor_accumulate_bool_8m", |b| {
            b.iter(|| {
                black_box(
                    numpy_lxor
                        .call_method1("accumulate", (&xb,))
                        .expect("np lxor.accum bool"),
                )
            });
        });
    });

    group.finish();
}

/// `fnp.masked_sum(a, mask)` against the only NumPy spelling of the same intent,
/// `a[mask].sum()`, live in the same invocation.
///
/// NumPy's API forces the compacted array to exist before it can be reduced:
/// there is no fused spelling, so the gather allocates a k-element temporary and
/// walks it once, then `sum` walks it again. The candidate never materialises it
/// — it evaluates NumPy's own pairwise tree over the selected elements straight
/// from a cursor, so the result is BYTE-identical rather than merely close, and
/// that byte-equality is asserted here before any timing.
///
/// This arm exists because the 2026-08-02 ledger row banking this win had no
/// committed harness, and therefore no incumbent artifact hash or shared
/// invocation id (bead deadlock-audit-5jk8g). A claim whose harness cannot be
/// re-run is not re-checkable.
///
/// ENGAGEMENT: byte-equality cannot prove the route ran here, because
/// `masked_sum`'s own fallback IS `a[mask].sum()` — a declined route would match
/// bits while measuring NumPy against NumPy and would read ~1.0x. The
/// observed-thread-activity report is the engagement evidence: the candidate
/// spreads across the pinned pool where the incumbent stays at one thread.
fn bench_masked_sum_f64_median_gate(_c: &mut Criterion) {
    const REQUIRED_BUILD_PROFILE: &str = "release-perf";
    const CONTRACT_ROUNDS: usize = 21;
    const CONTRACT_MIN_OF: usize = 1;
    const THREAD_ACTIVITY_REPETITIONS: usize = 3;

    assert_eq!(
        std::env::var("FNP_BENCH_PROFILE").as_deref(),
        Ok(REQUIRED_BUILD_PROFILE),
        "ship-grade masked-sum evidence requires FNP_BENCH_PROFILE=release-perf"
    );
    let build_worker =
        std::env::var("FNP_BUILD_WORKER").expect("FNP_BUILD_WORKER records the build origin");
    assert!(
        !build_worker.trim().is_empty(),
        "FNP_BUILD_WORKER must be set"
    );
    let threads = std::env::var("RAYON_NUM_THREADS")
        .expect("RAYON_NUM_THREADS must be explicitly pinned before masked-sum timing");
    for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok("1"),
            "{variable} must be one: neither a boolean gather nor a sum calls BLAS"
        );
    }
    let threads: usize = threads.parse().expect("thread count is numeric");
    assert_eq!(
        rayon::current_num_threads(),
        threads,
        "Rayon pool width does not match the pinned masked-sum configuration"
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_masked_sum").expect("masked-sum bench module");
        fnp_python(&module).expect("initialize fnp_python masked-sum module");
        let numpy = py.import("numpy").expect("numpy incumbent");

        common::report_incumbent_topology("fnp.masked_sum", "numpy.ndarray.sum");
        println!("NUMPY_BUILD_CONFIG_BEGIN workload=masked_sum_f64");
        numpy
            .getattr("show_config")
            .expect("numpy.show_config")
            .call0()
            .expect("report NumPy build configuration");
        println!("NUMPY_BUILD_CONFIG_END workload=masked_sum_f64");
        println!(
            "BLAS_RELEVANCE workload=masked_sum_f64 numpy_uses_blas=false \
             candidate_uses_blas=false blas_threads_pinned=1 \
             reason=boolean_gather_and_pairwise_sum_have_no_gemm"
        );

        let numpy_sum = numpy
            .getattr("ndarray")
            .expect("numpy.ndarray")
            .getattr("sum")
            .expect("numpy.ndarray.sum");
        common::report_numpy_incumbent_identity(py, "ndarray.sum", &numpy_sum);
        let fnp_masked_sum = module.getattr("masked_sum").expect("fnp.masked_sum");
        assert!(
            !fnp_masked_sum.is(&numpy_sum),
            "dispatch trap: fnp.masked_sum resolved to a NumPy callable"
        );

        let rng = numpy
            .getattr("random")
            .expect("numpy.random")
            .call_method1("default_rng", (20260802_u64,))
            .expect("seeded generator");

        for elements in [16_777_216_usize, 67_108_864] {
            let values = rng
                .call_method1("standard_normal", (elements,))
                .expect("f64 corpus");
            // Band selection, the spelling the ledger row profiled. The mask is
            // built OUTSIDE the timer: the measured scope is the gather and the
            // reduction, which is where the incumbent's forced materialisation
            // lives, not the two comparisons that produce the mask.
            let mask = values
                .call_method1("__gt__", (-0.5_f64,))
                .expect("lower band")
                .call_method1(
                    "__and__",
                    (values
                        .call_method1("__lt__", (0.5_f64,))
                        .expect("upper band"),),
                )
                .expect("band mask");
            let selected: usize = mask
                .call_method0("sum")
                .expect("selected count")
                .extract()
                .expect("selected count value");

            let run_incumbent = || {
                numpy
                    .call_method1("asarray", (black_box(&values),))
                    .expect("incumbent asarray")
                    .get_item(black_box(&mask))
                    .expect("incumbent gather")
                    .call_method0("sum")
                    .expect("incumbent sum")
            };
            let run_candidate = || {
                fnp_masked_sum
                    .call1((black_box(&values), black_box(&mask)))
                    .expect("candidate masked_sum")
            };

            let ours: f64 = run_candidate().extract().expect("candidate scalar");
            let theirs: f64 = run_incumbent().extract().expect("incumbent scalar");
            assert_eq!(
                ours.to_bits(),
                theirs.to_bits(),
                "masked_sum at {elements} elements is not BYTE-identical to a[mask].sum()"
            );

            let row = format!("python_masked_sum_f64_{elements}_vs_numpy");
            println!(
                "PARITY row={row} exact_bits=passed elements={elements} \
                 selected={selected} input_bytes={} \
                 incumbent_temporary_bytes={} checksum={:016x}",
                elements * 8,
                selected * 8,
                theirs.to_bits()
            );
            println!(
                "ROUTE_PRECONDITIONS row={row} dtype=float64 mask_dtype=bool \
                 exact_ndarray=true c_contiguous=true ndim=1 elements={elements} \
                 selected={selected} pinned_threads={threads} host_avx2={} \
                 host_avx512f={} candidate_route=try_zerocopy_f64_masked_sum",
                std::arch::is_x86_feature_detected!("avx2"),
                std::arch::is_x86_feature_detected!("avx512f"),
            );
            println!(
                "COUNTED_MECHANISM row={row} class=materialization_elimination_and_parallelism \
                 incumbent_algorithm=numpy_boolean_gather_to_k_element_temporary_then_pairwise_sum \
                 candidate_algorithm=streamed_pairwise_tree_over_selected_elements_no_temporary \
                 incumbent_allocated_bytes={} candidate_allocated_bytes=0 \
                 incumbent_passes_over_selected=2 candidate_passes_over_selected=1 \
                 incumbent_expected_threads=1 candidate_pinned_threads={threads} \
                 shared_input=true",
                selected * 8
            );

            common::report_observed_thread_activity(
                &row,
                "numpy",
                THREAD_ACTIVITY_REPETITIONS,
                || {
                    black_box(run_incumbent().extract::<f64>().expect("incumbent scalar"));
                },
            );
            common::report_observed_thread_activity(
                &row,
                "fnp",
                THREAD_ACTIVITY_REPETITIONS,
                || {
                    black_box(run_candidate().extract::<f64>().expect("candidate scalar"));
                },
            );

            let mut observe_incumbent = || {
                let started = Instant::now();
                let result = run_incumbent();
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: result.extract::<f64>().expect("incumbent scalar").to_bits(),
                }
            };
            let mut observe_candidate = || {
                let started = Instant::now();
                let result = run_candidate();
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: result.extract::<f64>().expect("candidate scalar").to_bits(),
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
                "MASKED_SUM_RESULT row={row} elements={elements} selected={selected} \
                 threads={threads} verdict={verdict} incumbent_median_ms={:.6} \
                 candidate_median_ms={:.6} ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
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
        }
    });
}

fn main() {
    common::gated_main(&[
        (
            "bench_masked_sum_f64_median_gate",
            bench_masked_sum_f64_median_gate,
        ),
        (
            "bench_cov_gram_pairing_contract",
            bench_cov_gram_pairing_contract,
        ),
        ("bench_statistics_boundary", bench_statistics_boundary),
        ("bench_cov_large_boundary", bench_cov_large_boundary),
        ("bench_std_var_axis_boundary", bench_std_var_axis_boundary),
        ("bench_var_multiaxis_boundary", bench_var_multiaxis_boundary),
        ("bench_var_midaxis_boundary", bench_var_midaxis_boundary),
        ("bench_var_f32_axis_boundary", bench_var_f32_axis_boundary),
        (
            "bench_nanextreme_f32_axis_boundary",
            bench_nanextreme_f32_axis_boundary,
        ),
        (
            "bench_nansum_f32_axis_boundary",
            bench_nansum_f32_axis_boundary,
        ),
        (
            "bench_nanvar_f32_axis_boundary",
            bench_nanvar_f32_axis_boundary,
        ),
        (
            "bench_nanvar_f32_last_axis_boundary",
            bench_nanvar_f32_last_axis_boundary,
        ),
        (
            "bench_nanvar_midaxis_boundary",
            bench_nanvar_midaxis_boundary,
        ),
        ("bench_var_axis0_boundary", bench_var_axis0_boundary),
        ("bench_sum_lastaxis_boundary", bench_sum_lastaxis_boundary),
        ("bench_prod_lastaxis_boundary", bench_prod_lastaxis_boundary),
        (
            "bench_cumsum_lastaxis_boundary",
            bench_cumsum_lastaxis_boundary,
        ),
        ("bench_cumsum_flat_boundary", bench_cumsum_flat_boundary),
        (
            "bench_accumulate_extremum_boundary",
            bench_accumulate_extremum_boundary,
        ),
    ]);
}
