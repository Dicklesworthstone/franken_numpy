//! histogram_full A/B: old fully-serial bin+scatter vs the shipped parallel
//! path (privatized fold for unweighted, parallel bin-lookup + ordered serial
//! accumulate for weighted). "new" calls the real `UFuncArray::histogram_full`.
//! "old" replicates the previous serial loop. Both are bit-identical.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;
use std::hint::black_box;

fn old_histogram_full(
    data: &[f64],
    bins: usize,
    range: Option<(f64, f64)>,
    weights: Option<&[f64]>,
) -> Vec<f64> {
    let (mut lo, mut hi) = match range {
        Some((a, b)) => (a, b),
        None => {
            let mut mn = f64::INFINITY;
            let mut mx = f64::NEG_INFINITY;
            for &v in data {
                if v < mn {
                    mn = v;
                }
                if v > mx {
                    mx = v;
                }
            }
            (mn, mx)
        }
    };
    if (hi - lo).abs() < f64::EPSILON {
        lo -= 0.5;
        hi += 0.5;
    }
    let width = (hi - lo) / bins as f64;
    let mut edges = Vec::with_capacity(bins + 1);
    for i in 0..=bins {
        edges.push(lo + i as f64 * width);
    }
    let mut counts = vec![0.0f64; bins];
    for (i, &v) in data.iter().enumerate() {
        if !(v >= lo && v <= hi) {
            continue;
        }
        let count_le = edges.partition_point(|e| *e <= v);
        let idx = (if count_le == 0 { 0 } else { count_le - 1 }).min(bins - 1);
        counts[idx] += weights.map_or(1.0, |w| w[i]);
    }
    counts
}

fn bench_histogram_full(c: &mut Criterion) {
    let n = 1 << 24; // 16.7M
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let data: Vec<f64> = (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 * 100.0
        })
        .collect();
    let weights: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let arr = UFuncArray::new(vec![n], data.clone(), DType::F64).unwrap();
    let warr = UFuncArray::new(vec![n], weights.clone(), DType::F64).unwrap();

    for &bins in &[64usize, 256] {
        // Unweighted (privatized-fold path).
        let got = arr.histogram_full(bins, Some((0.0, 100.0)), None, false).unwrap();
        assert_eq!(
            got.0.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            old_histogram_full(&data, bins, Some((0.0, 100.0)), None)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let mut g = c.benchmark_group(format!("histogram_full_unweighted_bins{bins}"));
        g.sample_size(20);
        g.bench_with_input(BenchmarkId::new("old_serial", n), &n, |b, _| {
            b.iter(|| black_box(old_histogram_full(black_box(&data), bins, Some((0.0, 100.0)), None)))
        });
        g.bench_with_input(BenchmarkId::new("par_fold", n), &n, |b, _| {
            b.iter(|| black_box(arr.histogram_full(bins, Some((0.0, 100.0)), None, false).unwrap()))
        });
        g.finish();
    }

    // Weighted (parallel bin-lookup + ordered serial accumulate).
    let bins = 64usize;
    let mut g = c.benchmark_group("histogram_full_weighted_bins64");
    g.sample_size(20);
    g.bench_with_input(BenchmarkId::new("old_serial", n), &n, |b, _| {
        b.iter(|| {
            black_box(old_histogram_full(
                black_box(&data),
                bins,
                Some((0.0, 100.0)),
                Some(&weights),
            ))
        })
    });
    g.bench_with_input(BenchmarkId::new("par_lookup", n), &n, |b, _| {
        b.iter(|| black_box(arr.histogram_full(bins, Some((0.0, 100.0)), Some(&warr), false).unwrap()))
    });
    g.finish();
}

criterion_group!(benches, bench_histogram_full);
criterion_main!(benches);
