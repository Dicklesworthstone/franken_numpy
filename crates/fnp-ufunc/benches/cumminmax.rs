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
use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::UFuncArray;
use rayon::prelude::*;
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

/// Former `cumulate_axis` work for `[N, 1]`, axis 1: zero-fill the output,
/// then dispatch one-element prefix lanes across Rayon.
fn former_cumsum_singleton(values: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0f64; values.len()];
    let scan_lane = |(out_lane, in_lane): (&mut [f64], &[f64])| {
        let mut acc = 0.0;
        for (out_slot, &value) in out_lane.iter_mut().zip(in_lane.iter()) {
            acc += value;
            *out_slot = acc;
        }
    };
    if values.len() >= (1 << 15) && rayon::current_num_threads() >= 2 {
        out.par_chunks_mut(1)
            .zip(values.par_chunks(1))
            .for_each(scan_lane);
    } else {
        out.chunks_mut(1)
            .zip(values.chunks(1))
            .for_each(scan_lane);
    }
    out
}

fn bench_cumsum_singleton_axis(c: &mut Criterion) {
    let n = 1usize << 18;
    let data: Vec<f64> = (0..n)
        .map(|i| match i % 257 {
            0 => -0.0,
            1 => f64::from_bits(0x7ff8_0000_0000_0042),
            2 => f64::INFINITY,
            3 => f64::NEG_INFINITY,
            _ => (i as f64) * 0.25 - 17.0,
        })
        .collect();
    let arr = UFuncArray::new(vec![n, 1], data.clone(), DType::F64).unwrap();
    let former = former_cumsum_singleton(&data);
    let candidate = arr.cumsum(Some(1)).unwrap();
    assert_eq!(candidate.shape(), &[n, 1]);
    assert_eq!(candidate.dtype(), DType::F64);
    assert_eq!(
        candidate
            .values()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        former.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );

    let mut group = c.benchmark_group("cumsum_singleton_axis");
    group.bench_function("former_zero_fill_rayon", |b| {
        b.iter(|| black_box(former_cumsum_singleton(black_box(&data))))
    });
    group.bench_function("direct_map", |b| {
        b.iter(|| black_box(arr.cumsum(black_box(Some(1))).unwrap()))
    });
    group.finish();
}

/// Former exact-I64 `cumsum` work for `[N, 1]`, axis 1: clone the source
/// sidecar, zero-fill the result, dispatch one-element lanes across Rayon, then
/// materialize the f64 bridge carried by `UFuncArray`.
fn former_i64_cumsum_singleton(values: &[i64]) -> (Vec<f64>, Vec<i64>) {
    let values = values.to_vec();
    let mut out = vec![0i64; values.len()];
    let scan_lane = |(out_lane, in_lane): (&mut [i64], &[i64])| {
        out_lane[0] = 0i64.wrapping_add(in_lane[0]);
    };
    if values.len() >= (1 << 15) && rayon::current_num_threads() >= 2 {
        out.par_chunks_mut(1)
            .zip(values.par_chunks(1))
            .for_each(scan_lane);
    } else {
        out.chunks_mut(1).zip(values.chunks(1)).for_each(scan_lane);
    }
    let bridge = out.iter().map(|&value| value as f64).collect();
    (bridge, out)
}

fn bench_i64_cumsum_singleton_axis(c: &mut Criterion) {
    let n = 1usize << 18;
    let data: Vec<i64> = (0..n)
        .map(|i| match i % 257 {
            0 => i64::MIN,
            1 => i64::MAX,
            2 => (1_i64 << 53) + 7,
            3 => -((1_i64 << 53) + 7),
            _ => (i as i64).wrapping_mul(2_654_435_761) - 50_000,
        })
        .collect();
    let arr = UFuncArray::from_storage(vec![n, 1], ArrayStorage::I64(data.clone())).unwrap();
    let (former_bridge, former_sidecar) = former_i64_cumsum_singleton(&data);
    let candidate = arr.cumsum(Some(1)).unwrap();
    assert_eq!(candidate.shape(), &[n, 1]);
    assert_eq!(candidate.dtype(), DType::I64);
    assert_eq!(
        candidate
            .values()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        former_bridge
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        candidate.to_storage().unwrap(),
        ArrayStorage::I64(former_sidecar)
    );

    let mut group = c.benchmark_group("i64_cumsum_singleton_axis");
    group.bench_function("former_zero_fill_rayon", |b| {
        b.iter(|| black_box(former_i64_cumsum_singleton(black_box(&data))))
    });
    group.bench_function("direct_exact_fold_map", |b| {
        b.iter(|| black_box(arr.cumsum(black_box(Some(1))).unwrap()))
    });
    group.finish();
}

/// Former `cumulative_op` work for `[N, 1]`, axis 1: zero-fill the output,
/// then dispatch one-element extrema lanes across Rayon.
fn former_cummin_singleton(values: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0f64; values.len()];
    let scan_lane = |(out_lane, in_lane): (&mut [f64], &[f64])| {
        let acc = in_lane[0];
        out_lane[0] = acc;
    };
    if values.len() >= (1 << 15) && rayon::current_num_threads() >= 2 {
        out.par_chunks_mut(1)
            .zip(values.par_chunks(1))
            .for_each(scan_lane);
    } else {
        out.chunks_mut(1).zip(values.chunks(1)).for_each(scan_lane);
    }
    out
}

fn bench_cummin_singleton_axis(c: &mut Criterion) {
    let n = 1usize << 18;
    let data: Vec<f64> = (0..n)
        .map(|i| match i % 257 {
            0 => -0.0,
            1 => f64::from_bits(0x7ff8_0000_0000_0042),
            2 => f64::INFINITY,
            3 => f64::NEG_INFINITY,
            _ => (i as f64) * 0.25 - 17.0,
        })
        .collect();
    let arr = UFuncArray::new(vec![n, 1], data.clone(), DType::F64).unwrap();
    let former = former_cummin_singleton(&data);
    let candidate = arr.cummin(Some(1)).unwrap();
    assert_eq!(candidate.shape(), &[n, 1]);
    assert_eq!(candidate.dtype(), DType::F64);
    assert_eq!(
        candidate
            .values()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        former.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );

    let mut group = c.benchmark_group("cummin_singleton_axis");
    group.bench_function("former_zero_fill_rayon", |b| {
        b.iter(|| black_box(former_cummin_singleton(black_box(&data))))
    });
    group.bench_function("direct_values_clone", |b| {
        b.iter(|| black_box(arr.cummin(black_box(Some(1))).unwrap()))
    });
    group.finish();
}

fn bench_cumminmax(c: &mut Criterion) {
    bench_i64_cumsum_singleton_axis(c);
    bench_cumsum_singleton_axis(c);
    bench_cummin_singleton_axis(c);
    if std::env::args().any(|arg| {
        matches!(
            arg.as_str(),
            "i64_cumsum_singleton_axis" | "cumsum_singleton_axis" | "cummin_singleton_axis"
        )
    }) {
        return;
    }
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
