//! batch_eigh A/B: serial per-lane eigh_nxn loop vs the shipped parallel batch_eigh.
//! "new" calls the real `batch_eigh` (production, parallel across lanes). "old"
//! replicates a serial loop over lanes. This is the win that makes wiring fnp's
//! own batch_eigh into np.linalg.eigh beat numpy's serial-C batched LAPACK loop.
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_linalg::{batch_eigh, eigh_nxn};
use std::hint::black_box;

fn old_batch_eigh(data: &[f64], batch: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let ms = n * n;
    let mut evals = Vec::with_capacity(batch * n);
    let mut evecs = Vec::with_capacity(batch * ms);
    for b in 0..batch {
        let (vl, vc) = eigh_nxn(&data[b * ms..(b + 1) * ms], n).unwrap();
        evals.extend_from_slice(&vl);
        evecs.extend_from_slice(&vc);
    }
    (evals, evecs)
}

fn make_sym_stack(batch: usize, n: usize) -> Vec<f64> {
    let ms = n * n;
    let mut s = 0x2545u64;
    let mut data = vec![0.0f64; batch * ms];
    for b in 0..batch {
        let base = b * ms;
        for i in 0..n {
            for j in i..n {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                let v = (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
                data[base + i * n + j] = v;
                data[base + j * n + i] = v;
            }
        }
    }
    data
}

fn bench(c: &mut Criterion) {
    // Stacks of small matrices (the common batched-eigh regime).
    for &(batch, n) in &[(4096usize, 4usize), (2048, 8), (1024, 16)] {
        let data = make_sym_stack(batch, n);
        let shape = vec![batch, n, n];
        let mut g = c.benchmark_group(format!("batch_eigh_b{batch}_n{n}"));
        g.sample_size(20);
        g.bench_with_input(BenchmarkId::new("serial", batch), &batch, |bb, _| {
            bb.iter(|| black_box(old_batch_eigh(black_box(&data), batch, n)))
        });
        g.bench_with_input(BenchmarkId::new("parallel", batch), &batch, |bb, _| {
            bb.iter(|| black_box(batch_eigh(black_box(&data), &shape).unwrap()))
        });
        g.finish();
    }
}
criterion_group!(benches, bench);
criterion_main!(benches);
