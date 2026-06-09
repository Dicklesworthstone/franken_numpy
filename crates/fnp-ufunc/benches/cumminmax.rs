//! cummin non-last-axis A/B: old serial stride-`inner` column scan vs the shipped
//! row-wise contiguous (+ parallel across outer blocks) rewrite in `cumulative_op`.
//!
//! "new" calls the real `UFuncArray::cummin`, so it exercises the exact production
//! path (including its parallel-gating decision); "old" replicates the previous
//! stride-`inner` serial column scan. Both are bit-identical (same per-column
//! ascending fold order). Two regimes:
//!   * 2-D axis 0  — single outer block => production runs serial row-wise; the win
//!     is purely the contiguous (cache-streaming) access vs the strided column walk.
//!   * 3-D axis 1  — many outer blocks => production also parallelizes across them.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;
use std::hint::black_box;

#[inline]
fn cmin(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a < b {
        a
    } else {
        b
    }
}

/// Previous kernel: serial, stride-`inner` per-column scan.
fn strided_serial(values: &[f64], shape: &[usize], axis: usize) -> Vec<f64> {
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let mut out = vec![0.0f64; values.len()];
    for o in 0..outer {
        let base = o * axis_len * inner;
        for ii in 0..inner {
            let mut acc = values[base + ii];
            out[base + ii] = acc;
            let mut off = base + ii + inner;
            for _ in 1..axis_len {
                acc = cmin(acc, values[off]);
                out[off] = acc;
                off += inner;
            }
        }
    }
    out
}

fn data_for(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| ((i as u64).wrapping_mul(2654435761) % 100000) as f64 / 1000.0 - 50.0)
        .collect()
}

fn bench_cumminmax(c: &mut Criterion) {
    let configs: &[(Vec<usize>, usize)] = &[
        (vec![4096, 4096], 0),   // single outer block: cache win only
        (vec![64, 256, 256], 1), // many outer blocks: cache + parallel
    ];
    for (shape, axis) in configs {
        let n: usize = shape.iter().product();
        let data = data_for(n);
        let arr = UFuncArray::new(shape.clone(), data.clone(), DType::F64).unwrap();
        // Correctness: production matches the old strided scan bit-for-bit.
        let want = strided_serial(&data, shape, *axis);
        let got = arr.cummin(Some(*axis as isize)).unwrap();
        assert_eq!(
            got.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        let label = format!("cummin_{shape:?}_ax{axis}");
        let mut group = c.benchmark_group(label);
        group.bench_with_input(BenchmarkId::new("strided_serial", n), &n, |b, _| {
            b.iter(|| black_box(strided_serial(black_box(&data), shape, *axis)))
        });
        group.bench_with_input(BenchmarkId::new("rowwise_prod", n), &n, |b, _| {
            b.iter(|| black_box(arr.cummin(black_box(Some(*axis as isize))).unwrap()))
        });
        group.finish();
    }
}

criterion_group!(benches, bench_cumminmax);
criterion_main!(benches);
