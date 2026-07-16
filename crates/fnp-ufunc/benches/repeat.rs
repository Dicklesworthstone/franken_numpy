//! repeat-axis A/B: old element-by-element scatter vs the shipped
//! copy_from_slice + parallel row fill.
//!
//! "new" calls the real `UFuncArray::repeat` (production path). "old" replicates
//! the previous behaviour: four nested loops copying each output element
//! individually. Both produce bit-identical output.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;
use std::{hint::black_box, time::Duration};

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

fn resize_modulo_control(array: &UFuncArray, output_len: usize) -> UFuncArray {
    let values = (0..output_len)
        .map(|index| array.values()[index % array.values().len()])
        .collect();
    let source_indices: Vec<usize> = (0..output_len)
        .map(|index| index % array.values().len())
        .collect();
    black_box(source_indices);
    UFuncArray::new(vec![output_len], values, array.dtype()).unwrap()
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

fn former_repeat_once_flat(values: &[f64]) -> (Vec<f64>, Vec<usize>) {
    let repeated = values
        .iter()
        .flat_map(|&value| std::iter::repeat_n(value, 1))
        .collect();
    let mut source_indices = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        source_indices.push(i);
    }
    (repeated, source_indices)
}

fn bench_repeat_once_identity(c: &mut Criterion) {
    let shape = vec![256, 512];
    let n: usize = shape.iter().product();
    let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.25 - 1.0).collect();
    let arr = UFuncArray::new(shape, data.clone(), DType::F64).unwrap();

    let (control, source_indices) = former_repeat_once_flat(&data);
    let candidate = arr.repeat(1, None).unwrap();
    assert_eq!(candidate.shape(), &[n]);
    assert!(
        candidate
            .values()
            .iter()
            .zip(&control)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );
    assert!(source_indices.iter().copied().eq(0..n));

    let mut group = c.benchmark_group("repeat_once_identity");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_secs(1));
    group.bench_function("former_values_plus_indices_256x512", |bench| {
        bench.iter(|| black_box(former_repeat_once_flat(black_box(&data))))
    });
    group.bench_function("clone_flatten_256x512", |bench| {
        bench.iter(|| black_box(arr.repeat(1, None).unwrap()))
    });
    group.finish();
}

fn bench_resize_repeat(c: &mut Criterion) {
    const SOURCE_LEN: usize = 1_024;
    const OUTPUT_LEN: usize = 8_388_608;

    let proof_values = vec![
        0.0,
        -0.0,
        f64::from_bits(0x7ff8_0000_0000_1234),
        f64::INFINITY,
        f64::NEG_INFINITY,
        1.25,
        -3.5,
    ];
    let proof_array = UFuncArray::new(vec![proof_values.len()], proof_values, DType::F64).unwrap();
    for output_len in [0, 1, 2, 5, 7, 8, 19, 64] {
        let former = resize_modulo_control(&proof_array, output_len);
        let candidate = proof_array.resize(&[output_len]).unwrap();
        assert_eq!(candidate.shape(), former.shape());
        assert_eq!(candidate.dtype(), former.dtype());
        assert_eq!(
            candidate.has_integer_sidecar(),
            former.has_integer_sidecar()
        );
        assert_eq!(
            candidate
                .values()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            former
                .values()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "resize bit mismatch at output length {output_len}"
        );
    }

    let values: Vec<f64> = (0..SOURCE_LEN)
        .map(|index| index as f64 * 0.25 - 17.0)
        .collect();
    let array = UFuncArray::new(vec![SOURCE_LEN], values, DType::F64).unwrap();

    let mut group = c.benchmark_group("resize_repeat");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.bench_function("former_values_plus_indices_modulo", |bench| {
        bench.iter(|| black_box(resize_modulo_control(black_box(&array), OUTPUT_LEN)))
    });
    group.bench_function("candidate_seed_and_double", |bench| {
        bench.iter(|| black_box(array.resize(black_box(&[OUTPUT_LEN])).unwrap()))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_repeat,
    bench_repeat_once_identity,
    bench_resize_repeat
);
criterion_main!(benches);
