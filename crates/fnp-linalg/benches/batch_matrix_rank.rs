//! batch_matrix_rank A/B: serial per-lane matrix_rank_nxn loop vs the shipped
//! parallel batch_matrix_rank. This is the win that makes wiring fnp's own
//! batch_matrix_rank into np.linalg.matrix_rank beat numpy's serial-C per-lane SVD.
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_linalg::{batch_matrix_rank, matrix_rank_nxn};
use std::hint::black_box;

fn old_batch(data: &[f64], batch: usize, n: usize, rcond: f64) -> Vec<usize> {
    let ms = n * n;
    (0..batch)
        .map(|b| matrix_rank_nxn(&data[b * ms..(b + 1) * ms], n, rcond).unwrap())
        .collect()
}

fn make_stack(batch: usize, n: usize) -> Vec<f64> {
    let ms = n * n;
    let mut s = 0x2545u64;
    (0..batch * ms)
        .map(|_| {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let rcond = 1e-12;
    for &(batch, n) in &[(4096usize, 4usize), (2048, 8), (1024, 16)] {
        let data = make_stack(batch, n);
        let shape = vec![batch, n, n];
        let mut g = c.benchmark_group(format!("batch_matrix_rank_b{batch}_n{n}"));
        g.sample_size(20);
        g.bench_with_input(BenchmarkId::new("serial", batch), &batch, |bb, _| {
            bb.iter(|| black_box(old_batch(black_box(&data), batch, n, rcond)))
        });
        g.bench_with_input(BenchmarkId::new("parallel", batch), &batch, |bb, _| {
            bb.iter(|| black_box(batch_matrix_rank(black_box(&data), &shape, rcond).unwrap()))
        });
        g.finish();
    }
}
criterion_group!(benches, bench);
criterion_main!(benches);
