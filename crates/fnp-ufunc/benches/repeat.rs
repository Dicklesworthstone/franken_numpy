//! repeat-axis A/B: old element-by-element scatter vs the shipped
//! copy_from_slice + parallel row fill.
//!
//! "new" calls the real `UFuncArray::repeat` (production path). "old" replicates
//! the previous behaviour: four nested loops copying each output element
//! individually. Both produce bit-identical output.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;
use std::hint::black_box;

fn old_repeat(values: &[f64], shape: &[usize], ax: usize, reps: usize) -> Vec<f64> {
    let inner: usize = shape[ax + 1..].iter().product();
    let outer: usize = shape[..ax].iter().product();
    let axis_len = shape[ax];
    let new_axis_len = axis_len * reps;
    let total = outer * new_axis_len * inner;
    let mut out = vec![0.0f64; total];
    for o in 0..outer {
        for k in 0..axis_len {
            for r in 0..reps {
                for i in 0..inner {
                    let src = o * axis_len * inner + k * inner + i;
                    let dst = o * new_axis_len * inner + (k * reps + r) * inner + i;
                    out[dst] = values[src];
                }
            }
        }
    }
    out
}

fn bench_repeat(c: &mut Criterion) {
    // (shape, axis, repeats) — axis 0 (outer==1, inner large), middle axis, and
    // last axis (inner==1).
    let cases: &[(Vec<usize>, usize, usize)] = &[
        (vec![2048, 2048], 0, 4),
        (vec![64, 2048, 64], 1, 3),
        (vec![1024, 1024], 1, 8),
    ];
    for (shape, ax, reps) in cases {
        let n: usize = shape.iter().product();
        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 - 1.0).collect();
        let arr = UFuncArray::new(shape.clone(), data.clone(), DType::F64).unwrap();
        let got = arr.repeat(*reps, Some(*ax as isize)).unwrap();
        assert_eq!(
            got.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            old_repeat(&data, shape, *ax, *reps)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let out_n: usize = got.shape().iter().product();
        let mut group = c.benchmark_group(format!("repeat_{shape:?}_ax{ax}_x{reps}"));
        group.bench_with_input(BenchmarkId::new("old_scatter", out_n), &out_n, |b, _| {
            b.iter(|| black_box(old_repeat(black_box(&data), shape, *ax, *reps)))
        });
        group.bench_with_input(BenchmarkId::new("copyslice_par", out_n), &out_n, |b, _| {
            b.iter(|| black_box(arr.repeat(*reps, black_box(Some(*ax as isize))).unwrap()))
        });
        group.finish();
    }
}

criterion_group!(benches, bench_repeat);
criterion_main!(benches);
