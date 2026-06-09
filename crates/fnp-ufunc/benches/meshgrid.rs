use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;
use std::hint::black_box;

fn old_meshgrid(arrays: &[UFuncArray]) -> Vec<Vec<f64>> {
    let ndim = arrays.len();
    let out_shape: Vec<usize> = arrays.iter().map(|a| a.shape()[0]).collect();
    let out_count: usize = out_shape.iter().product();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() { strides[i] = strides[i+1]*out_shape[i+1]; }
    let mut out = Vec::new();
    for (dim, arr) in arrays.iter().enumerate() {
        let axis = dim;
        let mut v = Vec::with_capacity(out_count);
        for flat in 0..out_count { v.push(arr.values()[(flat/strides[axis])%out_shape[axis]]); }
        out.push(v);
    }
    out
}

fn bench(c: &mut Criterion) {
    for &n in &[2048usize, 4096] {
        let x = UFuncArray::new(vec![n], (0..n).map(|i| i as f64).collect(), DType::F64).unwrap();
        let y = UFuncArray::new(vec![n], (0..n).map(|i| i as f64 * 2.0).collect(), DType::F64).unwrap();
        let arrs = vec![x.clone(), y.clone()];
        let _ = UFuncArray::meshgrid_advanced(&arrs, "ij", false).unwrap();
        let mut g = c.benchmark_group(format!("meshgrid_{n}x{n}"));
        g.sample_size(20);
        g.bench_with_input(BenchmarkId::new("old_serial", n), &n, |b,_| b.iter(|| black_box(old_meshgrid(black_box(&arrs)))));
        g.bench_with_input(BenchmarkId::new("par_map", n), &n, |b,_| b.iter(|| black_box(UFuncArray::meshgrid_advanced(black_box(&arrs), "ij", false).unwrap())));
        g.finish();
    }
}
criterion_group!(benches, bench);
criterion_main!(benches);
