//! triu/tril A/B: old per-element branch loop (writes every cell, copy-or-zero)
//! vs the shipped zero-fill + per-row copy_from_slice build (`triangle_build`),
//! parallel across row bands above the L3 cache cliff.
//!
//! "new" calls the real `UFuncArray::triu` / `::tril` (production path). "old"
//! replicates the previous behaviour: a fresh buffer filled one element at a
//! time with a per-cell mask branch. Both produce bit-identical output.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;
use std::hint::black_box;

fn old_triu(values: &[f64], rows: usize, cols: usize, k: i64) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            if c as i64 >= (r as i64).saturating_add(k) {
                out[idx] = values[idx];
            } else {
                out[idx] = 0.0;
            }
        }
    }
    out
}

fn old_tril(values: &[f64], rows: usize, cols: usize, k: i64) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            if c as i64 <= (r as i64).saturating_add(k) {
                out[idx] = values[idx];
            } else {
                out[idx] = 0.0;
            }
        }
    }
    out
}

/// Former `triangle_build` work for an offset whose mask retains every cell:
/// zero-fill the destination, then overwrite each complete row.
fn former_full_keep(values: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        let start = r * cols;
        out[start..start + cols].copy_from_slice(&values[start..start + cols]);
    }
    out
}

fn bench_triangle_full_keep(c: &mut Criterion) {
    let (rows, cols) = (1024usize, 1024usize);
    let n = rows * cols;
    let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 - 1.0).collect();
    let arr = UFuncArray::new(vec![rows, cols], data.clone(), DType::F64).unwrap();
    let triu_k = 1 - rows as i64;
    let tril_k = cols as i64 - 1;
    let former = former_full_keep(&data, rows, cols);
    let upper = arr.triu(triu_k).unwrap();
    let lower = arr.tril(tril_k).unwrap();
    let former_bits: Vec<u64> = former.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        upper.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        former_bits
    );
    assert_eq!(
        lower.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        former_bits
    );

    let mut group = c.benchmark_group("triangle_full_keep");
    group.bench_function("former_zero_then_copy", |b| {
        b.iter(|| black_box(former_full_keep(black_box(&data), rows, cols)))
    });
    group.bench_function("clone_direct", |b| {
        b.iter(|| black_box(arr.triu(black_box(triu_k)).unwrap()))
    });
    group.finish();
}

fn bench_triangle(c: &mut Criterion) {
    bench_triangle_full_keep(c);
    // Keep focused proof runs cheap: the historical triangle groups allocate
    // several 128 MiB matrices before Criterion can filter their measurements.
    if std::env::args().any(|arg| arg == "triangle_full_keep") {
        return;
    }
    let cases: &[(usize, usize)] = &[(4096, 4096), (8192, 2048), (2048, 8192)];
    for &(rows, cols) in cases {
        let n = rows * cols;
        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 - 1.0).collect();
        let arr = UFuncArray::new(vec![rows, cols], data.clone(), DType::F64).unwrap();
        // Parity check before benching.
        let got_u = arr.triu(0).unwrap();
        let got_l = arr.tril(0).unwrap();
        assert_eq!(
            got_u
                .values()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            old_triu(&data, rows, cols, 0)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            got_l
                .values()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            old_tril(&data, rows, cols, 0)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );

        let mut group = c.benchmark_group(format!("triu_{rows}x{cols}"));
        group.bench_with_input(BenchmarkId::new("old_branch", n), &n, |b, _| {
            b.iter(|| black_box(old_triu(black_box(&data), rows, cols, 0)))
        });
        group.bench_with_input(BenchmarkId::new("copyslice_par", n), &n, |b, _| {
            b.iter(|| black_box(arr.triu(black_box(0)).unwrap()))
        });
        group.finish();

        let mut group = c.benchmark_group(format!("tril_{rows}x{cols}"));
        group.bench_with_input(BenchmarkId::new("old_branch", n), &n, |b, _| {
            b.iter(|| black_box(old_tril(black_box(&data), rows, cols, 0)))
        });
        group.bench_with_input(BenchmarkId::new("copyslice_par", n), &n, |b, _| {
            b.iter(|| black_box(arr.tril(black_box(0)).unwrap()))
        });
        group.finish();
    }
}

criterion_group!(benches, bench_triangle);
criterion_main!(benches);
