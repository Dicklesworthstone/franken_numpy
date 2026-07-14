//! batch_solve A/B: serial per-lane solve_nxn loop vs the shipped parallel
//! batch_solve. This is the win that makes wiring fnp's own batch_solve into
//! np.linalg.solve beat numpy's serial-C per-lane LU.
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_linalg::{batch_solve, lu_factor_nxn, solve_nxn};
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Duration;

fn old_batch(a: &[f64], b: &[f64], batch: usize, n: usize) -> Vec<f64> {
    let ms = n * n;
    let mut out = Vec::with_capacity(batch * n);
    for k in 0..batch {
        let x = solve_nxn(&a[k * ms..(k + 1) * ms], &b[k * n..(k + 1) * n], n).unwrap();
        out.extend_from_slice(&x);
    }
    out
}

fn old_batch_broadcast_a(a: &[f64], b: &[f64], batch: usize, n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(batch * n);
    for k in 0..batch {
        let x = solve_nxn(a, &b[k * n..(k + 1) * n], n).unwrap();
        out.extend_from_slice(&x);
    }
    out
}

fn old_factor_once_matrix(
    a: &[f64],
    b: &[f64],
    batch: usize,
    n: usize,
    rhs_cols: usize,
) -> Vec<f64> {
    let (lu, perm, _) = lu_factor_nxn(a, n).unwrap();
    let rhs_width = n * rhs_cols;
    let mut result = vec![0.0; batch * rhs_width];
    result
        .par_chunks_mut(rhs_width)
        .enumerate()
        .for_each(|(idx, out)| {
            let b_sub = &b[idx * rhs_width..(idx + 1) * rhs_width];
            for i in 0..n {
                let p_i = perm[i];
                for col in 0..rhs_cols {
                    out[i * rhs_cols + col] = b_sub[p_i * rhs_cols + col];
                }
            }
            for i in 1..n {
                for j in 0..i {
                    let l_ij = lu[i * n + j];
                    for col in 0..rhs_cols {
                        out[i * rhs_cols + col] -= l_ij * out[j * rhs_cols + col];
                    }
                }
            }
            for i in (0..n).rev() {
                for j in (i + 1)..n {
                    let u_ij = lu[i * n + j];
                    for col in 0..rhs_cols {
                        out[i * rhs_cols + col] -= u_ij * out[j * rhs_cols + col];
                    }
                }
                let u_ii = lu[i * n + i];
                for col in 0..rhs_cols {
                    out[i * rhs_cols + col] /= u_ii;
                }
            }
        });
    result
}

fn make(batch: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let ms = n * n;
    let mut s = 0x2545u64;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let mut a = vec![0.0f64; batch * ms];
    // diagonally dominant -> well-conditioned, non-singular
    for k in 0..batch {
        for i in 0..n {
            for j in 0..n {
                a[k * ms + i * n + j] = rnd();
            }
            a[k * ms + i * n + i] += n as f64 * 2.0;
        }
    }
    let b: Vec<f64> = (0..batch * n).map(|_| rnd()).collect();
    (a, b)
}

fn make_broadcast_a(batch: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let ms = n * n;
    let mut s = 0x5eed_f00du64;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let mut a = vec![0.0f64; ms];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = rnd();
        }
        a[i * n + i] += n as f64 * 2.0;
    }
    let b: Vec<f64> = (0..batch * n).map(|_| rnd()).collect();
    (a, b)
}

fn bench(c: &mut Criterion) {
    for &(batch, n) in &[(8192usize, 4usize), (4096, 8), (2048, 16)] {
        let (a, b) = make(batch, n);
        let a_shape = vec![batch, n, n];
        let b_shape = vec![batch, n];
        let mut g = c.benchmark_group(format!("batch_solve_b{batch}_n{n}"));
        g.sample_size(20);
        g.bench_with_input(BenchmarkId::new("serial", batch), &batch, |bb, _| {
            bb.iter(|| black_box(old_batch(black_box(&a), black_box(&b), batch, n)))
        });
        g.bench_with_input(BenchmarkId::new("parallel", batch), &batch, |bb, _| {
            bb.iter(|| {
                black_box(
                    batch_solve(black_box(&a), &a_shape, black_box(&b), &b_shape, true).unwrap(),
                )
            })
        });
        g.finish();
    }

    for &(batch, n) in &[(8192usize, 16usize), (2048, 32), (512, 64)] {
        let (a, b) = make_broadcast_a(batch, n);
        let a_shape = vec![n, n];
        let b_shape = vec![batch, n];
        let mut g = c.benchmark_group(format!("batch_solve_broadcast_a_b{batch}_n{n}"));
        g.sample_size(10);
        g.bench_with_input(
            BenchmarkId::new("refactor_per_lane_serial", batch),
            &batch,
            |bb, _| {
                bb.iter(|| {
                    black_box(old_batch_broadcast_a(
                        black_box(&a),
                        black_box(&b),
                        batch,
                        n,
                    ))
                })
            },
        );
        g.bench_with_input(BenchmarkId::new("batch_solve", batch), &batch, |bb, _| {
            bb.iter(|| {
                black_box(
                    batch_solve(black_box(&a), &a_shape, black_box(&b), &b_shape, true).unwrap(),
                )
            })
        });
        g.finish();
    }

    // A literal 2-D A and an equivalent singleton-batch A have identical solve
    // semantics. Keep the latter as a stable per-lane-refactor control while the
    // former prices factor-once reuse for batched matrix right-hand sides.
    {
        let (batch, n, rhs_cols) = (128usize, 128usize, 4usize);
        let (a, b) = make_broadcast_a(batch * rhs_cols, n);
        let literal_a_shape = [n, n];
        let control_a_shape = [1, n, n];
        let b_shape = [batch, n, rhs_cols];
        let literal = batch_solve(&a, &literal_a_shape, &b, &b_shape, false).unwrap();
        let control = batch_solve(&a, &control_a_shape, &b, &b_shape, false).unwrap();
        let reload_store = old_factor_once_matrix(&a, &b, batch, n, rhs_cols);
        assert_eq!(literal.len(), control.len());
        for (idx, (lhs, rhs)) in literal.iter().zip(&control).enumerate() {
            assert_eq!(lhs.to_bits(), rhs.to_bits(), "flat output {idx} diverged");
        }
        for (idx, (lhs, rhs)) in literal.iter().zip(&reload_store).enumerate() {
            assert_eq!(lhs.to_bits(), rhs.to_bits(), "reload/store output {idx} diverged");
        }

        let mut g = c.benchmark_group(format!(
            "batch_solve_broadcast_a_matrix_b{batch}_n{n}_m{rhs_cols}"
        ));
        g.sample_size(10);
        g.warm_up_time(Duration::from_millis(250));
        g.measurement_time(Duration::from_secs(1));
        g.bench_function("singleton_batch_control", |bb| {
            bb.iter(|| {
                black_box(
                    batch_solve(
                        black_box(&a),
                        &control_a_shape,
                        black_box(&b),
                        &b_shape,
                        false,
                    )
                    .unwrap(),
                )
            })
        });
        g.bench_function("literal_2d_candidate", |bb| {
            bb.iter(|| {
                black_box(
                    batch_solve(
                        black_box(&a),
                        &literal_a_shape,
                        black_box(&b),
                        &b_shape,
                        false,
                    )
                    .unwrap(),
                )
            })
        });
        g.bench_function("reload_store_control", |bb| {
            bb.iter(|| {
                black_box(old_factor_once_matrix(
                    black_box(&a),
                    black_box(&b),
                    batch,
                    n,
                    rhs_cols,
                ))
            })
        });
        g.finish();
    }
}
criterion_group!(benches, bench);
criterion_main!(benches);
