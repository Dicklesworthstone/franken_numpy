use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;
use std::hint::black_box;

fn old_bincount(vals: &[f64], len: usize) -> Vec<f64> {
    let mut counts = vec![0.0f64; len];
    for &v in vals { counts[v as usize] += 1.0; }
    counts
}

fn bench(c: &mut Criterion) {
    let n = 1 << 24; // 16.7M
    let mut s = 0x1234_5678_9abc_def0u64;
    for &bins in &[256usize, 4096, 65536] {
        let data: Vec<f64> = (0..n).map(|_| {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            ((s >> 11) % bins as u64) as f64
        }).collect();
        let arr = UFuncArray::new(vec![n], data.clone(), DType::F64).unwrap();
        let got = arr.bincount().unwrap();
        assert_eq!(got.values()[..bins.min(got.values().len())].iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
                   old_bincount(&data, bins)[..bins.min(got.values().len())].iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        let mut g = c.benchmark_group(format!("bincount_bins{bins}"));
        g.sample_size(20);
        g.bench_with_input(BenchmarkId::new("old_serial", n), &n, |b,_| b.iter(|| black_box(old_bincount(black_box(&data), bins))));
        g.bench_with_input(BenchmarkId::new("par_fold", n), &n, |b,_| b.iter(|| black_box(arr.bincount().unwrap())));
        g.finish();
    }
}
criterion_group!(benches, bench);
criterion_main!(benches);
