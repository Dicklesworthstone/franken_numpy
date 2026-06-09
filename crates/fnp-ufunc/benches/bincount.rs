use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::UFuncArray;
use std::hint::black_box;

fn old_bincount(vals: &[f64], len: usize) -> Vec<f64> {
    let mut counts = vec![0.0f64; len];
    for &v in vals {
        counts[v as usize] += 1.0;
    }
    counts
}

fn old_bincount_i64(vals: &[i64], len: usize) -> Vec<f64> {
    let mut counts = vec![0.0f64; len];
    for &v in vals {
        counts[v as usize] += 1.0;
    }
    counts
}

fn bench(c: &mut Criterion) {
    let n = 1 << 24; // 16.7M
    let mut s = 0x1234_5678_9abc_def0u64;
    for &bins in &[256usize, 4096, 65536] {
        let data_i64: Vec<i64> = (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) % bins as u64) as i64
            })
            .collect();
        let data: Vec<f64> = data_i64.iter().map(|&v| v as f64).collect();
        let arr = UFuncArray::new(vec![n], data.clone(), DType::F64).unwrap();
        let arr_i64 =
            UFuncArray::from_storage(vec![n], ArrayStorage::I64(data_i64.clone())).unwrap();
        let got = arr.bincount().unwrap();
        assert_eq!(
            got.values()[..bins.min(got.values().len())]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            old_bincount(&data, bins)[..bins.min(got.values().len())]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let got_i64 = arr_i64.bincount().unwrap();
        assert_eq!(
            got_i64.values()[..bins.min(got_i64.values().len())]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            old_bincount_i64(&data_i64, bins)[..bins.min(got_i64.values().len())]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let mut g = c.benchmark_group(format!("bincount_bins{bins}"));
        g.sample_size(20);
        g.bench_with_input(BenchmarkId::new("old_serial", n), &n, |b, _| {
            b.iter(|| black_box(old_bincount(black_box(&data), bins)))
        });
        g.bench_with_input(BenchmarkId::new("par_fold", n), &n, |b, _| {
            b.iter(|| black_box(arr.bincount().unwrap()))
        });
        g.finish();

        let mut g = c.benchmark_group(format!("bincount_i64_storage_bins{bins}"));
        g.sample_size(20);
        g.bench_with_input(BenchmarkId::new("old_serial", n), &n, |b, _| {
            b.iter(|| black_box(old_bincount_i64(black_box(&data_i64), bins)))
        });
        g.bench_with_input(BenchmarkId::new("current", n), &n, |b, _| {
            b.iter(|| black_box(arr_i64.bincount().unwrap()))
        });
        g.finish();
    }
}
criterion_group!(benches, bench);
criterion_main!(benches);
