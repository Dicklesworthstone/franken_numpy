//! 2-D transpose A/B: old per-element coordinate-decomposition gather vs the
//! shipped cache-tiled parallel data move (`transpose_2d_par`).
//!
//! "new" calls the real `UFuncArray::transpose` (exercises the production fast
//! path). "old" replicates the previous behaviour: a flat scan that decomposes
//! every output index into a multi-index with `ndim` integer divisions and then
//! gathers strided from the source. Both produce bit-identical output.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;
use std::hint::black_box;

/// Previous kernel: per-element index decomposition + strided gather (2-D).
fn old_transpose(values: &[f64], r: usize, c: usize) -> Vec<f64> {
    // new shape is [c, r]; new C-strides are [r, 1]; old C-strides are [c, 1].
    let total = r * c;
    let mut out = vec![0.0f64; total];
    let new_strides = [r, 1usize];
    let old_strides = [c, 1usize];
    let perm = [1usize, 0usize];
    for (flat_new, slot) in out.iter_mut().enumerate() {
        let mut remainder = flat_new;
        let mut flat_old = 0usize;
        for (new_axis, &ns) in new_strides.iter().enumerate() {
            let idx = remainder / ns;
            remainder %= ns;
            flat_old += idx * old_strides[perm[new_axis]];
        }
        *slot = values[flat_old];
    }
    out
}

fn bench_transpose(c: &mut Criterion) {
    for &(r, cc) in &[(4096usize, 4096usize), (2048, 2048), (1024, 4096)] {
        let n = r * cc;
        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 - 1.0).collect();
        let arr = UFuncArray::new(vec![r, cc], data.clone(), DType::F64).unwrap();
        // Correctness: production matches the old per-element gather bit-for-bit.
        let got = arr.transpose(None).unwrap();
        assert_eq!(
            got.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            old_transpose(&data, r, cc)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );

        let mut group = c.benchmark_group(format!("transpose_{r}x{cc}"));
        group.bench_with_input(BenchmarkId::new("old_gather", n), &n, |b, _| {
            b.iter(|| black_box(old_transpose(black_box(&data), r, cc)))
        });
        group.bench_with_input(BenchmarkId::new("tiled_par", n), &n, |b, _| {
            b.iter(|| black_box(arr.transpose(black_box(None)).unwrap()))
        });
        group.finish();
    }
}

criterion_group!(benches, bench_transpose);
criterion_main!(benches);
