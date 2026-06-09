//! batch_solve A/B: serial per-lane solve_nxn loop vs the shipped parallel
//! batch_solve. This is the win that makes wiring fnp's own batch_solve into
//! np.linalg.solve beat numpy's serial-C per-lane LU.
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_linalg::{batch_solve, solve_nxn};
use std::hint::black_box;

fn old_batch(a: &[f64], b: &[f64], batch: usize, n: usize) -> Vec<f64> {
    let ms = n * n;
    let mut out = Vec::with_capacity(batch * n);
    for k in 0..batch {
        let x = solve_nxn(&a[k * ms..(k + 1) * ms], &b[k * n..(k + 1) * n], n).unwrap();
        out.extend_from_slice(&x);
    }
    out
}

fn make(batch: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let ms = n * n;
    let mut s = 0x2545u64;
    let mut rnd = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0 };
    let mut a = vec![0.0f64; batch * ms];
    // diagonally dominant -> well-conditioned, non-singular
    for k in 0..batch {
        for i in 0..n {
            for j in 0..n { a[k * ms + i * n + j] = rnd(); }
            a[k * ms + i * n + i] += n as f64 * 2.0;
        }
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
            bb.iter(|| black_box(batch_solve(black_box(&a), &a_shape, black_box(&b), &b_shape, true).unwrap()))
        });
        g.finish();
    }
}
criterion_group!(benches, bench);
criterion_main!(benches);
