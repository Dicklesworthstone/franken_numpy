//! kron A/B: old serial nested-loop fill vs the shipped parallel row fill.
//!
//! "new" calls the real `kron_nxn` (production path). "old" replicates the
//! previous serial nested loops. Both produce bit-identical output.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_linalg::kron_nxn;
use std::hint::black_box;

fn old_kron(a: &[f64], m: usize, n: usize, b: &[f64], p: usize, q: usize) -> Vec<f64> {
    let out_cols = n * q;
    let mut out = vec![0.0f64; m * p * out_cols];
    for i in 0..m {
        for j in 0..n {
            let av = a[i * n + j];
            for k in 0..p {
                for l in 0..q {
                    out[(i * p + k) * out_cols + (j * q + l)] = av * b[k * q + l];
                }
            }
        }
    }
    out
}

fn bench_kron(c: &mut Criterion) {
    // (m, n, p, q) — large outputs where parallelism pays off.
    let cases: &[(usize, usize, usize, usize)] = &[
        (64, 64, 64, 64),  // 16.7M output
        (256, 256, 16, 16),
        (16, 16, 256, 256),
    ];
    for &(m, n, p, q) in cases {
        let a: Vec<f64> = (0..m * n).map(|i| (i as f64) * 0.5 - 1.0).collect();
        let b: Vec<f64> = (0..p * q).map(|i| (i as f64) * 0.25 + 0.3).collect();
        let got = kron_nxn(&a, m, n, &b, p, q).unwrap();
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            old_kron(&a, m, n, &b, p, q)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let out_n = got.len();
        let mut group = c.benchmark_group(format!("kron_{m}x{n}_{p}x{q}"));
        group.bench_with_input(BenchmarkId::new("old_serial", out_n), &out_n, |bb, _| {
            bb.iter(|| black_box(old_kron(black_box(&a), m, n, black_box(&b), p, q)))
        });
        group.bench_with_input(BenchmarkId::new("row_par", out_n), &out_n, |bb, _| {
            bb.iter(|| black_box(kron_nxn(black_box(&a), m, n, black_box(&b), p, q).unwrap()))
        });
        group.finish();
    }
}

criterion_group!(benches, bench_kron);
criterion_main!(benches);
