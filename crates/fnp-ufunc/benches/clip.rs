//! clip A/B: old serial iter().map() vs the shipped parallel chunk map.
//!
//! "new" calls the real `UFuncArray::clip` (production path). "old" replicates the
//! previous serial per-element map. Both produce bit-identical output.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;
use std::hint::black_box;

fn old_clip(values: &[f64], lo: f64, hi: f64) -> Vec<f64> {
    values
        .iter()
        .map(|&v| {
            if lo.is_nan() || hi.is_nan() || v.is_nan() {
                f64::NAN
            } else {
                let tmp = if v < lo { lo } else { v };
                if tmp > hi { hi } else { tmp }
            }
        })
        .collect()
}

fn bench_clip(c: &mut Criterion) {
    for &n in &[4_000_000usize, 16_777_216] {
        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001 - 5000.0).collect();
        let arr = UFuncArray::new(vec![n], data.clone(), DType::F64).unwrap();
        let got = arr.clip(-100.0, 100.0);
        assert_eq!(
            got.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            old_clip(&data, -100.0, 100.0)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let mut group = c.benchmark_group(format!("clip_{n}"));
        group.bench_with_input(BenchmarkId::new("old_serial", n), &n, |b, _| {
            b.iter(|| black_box(old_clip(black_box(&data), -100.0, 100.0)))
        });
        group.bench_with_input(BenchmarkId::new("chunk_par", n), &n, |b, _| {
            b.iter(|| black_box(arr.clip(black_box(-100.0), black_box(100.0))))
        });
        group.finish();
    }
}

criterion_group!(benches, bench_clip);
criterion_main!(benches);
