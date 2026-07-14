use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::{IntegerSidecar, UFuncArray};
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Duration;

fn old_meshgrid(arrays: &[UFuncArray]) -> Vec<Vec<f64>> {
    let ndim = arrays.len();
    let out_shape: Vec<usize> = arrays.iter().map(|a| a.shape()[0]).collect();
    let out_count: usize = out_shape.iter().product();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * out_shape[i + 1];
    }
    let mut out = Vec::new();
    for (dim, arr) in arrays.iter().enumerate() {
        let axis = dim;
        let mut v = Vec::with_capacity(out_count);
        for flat in 0..out_count {
            v.push(arr.values()[(flat / strides[axis]) % out_shape[axis]]);
        }
        out.push(v);
    }
    out
}

fn full_divmod_parallel_meshgrid(arrays: &[UFuncArray]) -> Vec<Vec<f64>> {
    let ndim = arrays.len();
    let out_shape: Vec<usize> = arrays.iter().map(|a| a.shape()[0]).collect();
    let out_count: usize = out_shape.iter().product();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * out_shape[i + 1];
    }
    arrays
        .iter()
        .enumerate()
        .map(|(axis, arr)| {
            (0..out_count)
                .into_par_iter()
                .map(|flat| arr.values()[(flat / strides[axis]) % out_shape[axis]])
                .collect()
        })
        .collect()
}

fn partial_strength_reduce_parallel_meshgrid(arrays: &[UFuncArray]) -> Vec<Vec<f64>> {
    let ndim = arrays.len();
    let out_shape: Vec<usize> = arrays.iter().map(|a| a.shape()[0]).collect();
    let out_count: usize = out_shape.iter().product();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * out_shape[i + 1];
    }
    arrays
        .iter()
        .enumerate()
        .map(|(axis, arr)| {
            let stride = strides[axis];
            let alen = out_shape[axis];
            match (stride.is_power_of_two(), alen.is_power_of_two()) {
                (true, true) => {
                    let shift = stride.trailing_zeros();
                    let mask = alen - 1;
                    (0..out_count)
                        .into_par_iter()
                        .map(|flat| arr.values()[(flat >> shift) & mask])
                        .collect()
                }
                (true, false) => {
                    let shift = stride.trailing_zeros();
                    (0..out_count)
                        .into_par_iter()
                        .map(|flat| arr.values()[(flat >> shift) % alen])
                        .collect()
                }
                (false, true) => {
                    let mask = alen - 1;
                    (0..out_count)
                        .into_par_iter()
                        .map(|flat| arr.values()[(flat / stride) & mask])
                        .collect()
                }
                (false, false) => (0..out_count)
                    .into_par_iter()
                    .map(|flat| arr.values()[(flat / stride) % alen])
                    .collect(),
            }
        })
        .collect()
}

fn per_cell_i64_sidecar_meshgrid(arrays: &[UFuncArray]) -> Vec<(Vec<f64>, Vec<i64>)> {
    let ndim = arrays.len();
    let out_shape: Vec<usize> = arrays.iter().map(|a| a.shape()[0]).collect();
    let out_count: usize = out_shape.iter().product();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * out_shape[i + 1];
    }
    arrays
        .iter()
        .enumerate()
        .map(|(axis, arr)| {
            let mut source_indices = vec![0usize; out_count];
            source_indices
                .par_iter_mut()
                .enumerate()
                .for_each(|(flat, source)| {
                    *source = (flat / strides[axis]) % out_shape[axis]
                });
            let values = source_indices
                .iter()
                .map(|&source| arr.values()[source])
                .collect();
            let source_exact = match arr.integer_sidecar() {
                Some(IntegerSidecar::I64(exact)) => exact,
                _ => panic!("benchmark requires an i64 sidecar"),
            };
            let exact = source_indices
                .iter()
                .map(|&source| source_exact[source])
                .collect();
            (values, exact)
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    for &n in &[2048usize, 4096] {
        let x = UFuncArray::new(vec![n], (0..n).map(|i| i as f64).collect(), DType::F64).unwrap();
        let y = UFuncArray::new(
            vec![n],
            (0..n).map(|i| i as f64 * 2.0).collect(),
            DType::F64,
        )
        .unwrap();
        let arrs = vec![x.clone(), y.clone()];
        let _ = UFuncArray::meshgrid_advanced(&arrs, "ij", false).unwrap();
        let mut g = c.benchmark_group(format!("meshgrid_{n}x{n}"));
        g.sample_size(20);
        g.bench_with_input(BenchmarkId::new("old_serial", n), &n, |b, _| {
            b.iter(|| black_box(old_meshgrid(black_box(&arrs))))
        });
        g.bench_with_input(BenchmarkId::new("par_map", n), &n, |b, _| {
            b.iter(|| {
                black_box(UFuncArray::meshgrid_advanced(black_box(&arrs), "ij", false).unwrap())
            })
        });
        g.finish();
    }

    {
        let (nx, ny) = (2048usize, 2000usize);
        let x = UFuncArray::new(
            vec![nx],
            (0..nx).map(|i| i as f64).collect(),
            DType::F64,
        )
        .unwrap();
        let y = UFuncArray::new(
            vec![ny],
            (0..ny).map(|i| -(i as f64)).collect(),
            DType::F64,
        )
        .unwrap();
        let arrs = vec![x, y];
        let control = full_divmod_parallel_meshgrid(&arrs);
        let candidate = UFuncArray::meshgrid_advanced(&arrs, "ij", false).unwrap();
        for (grid, reference) in candidate.iter().zip(&control) {
            assert_eq!(grid.values().len(), reference.len());
            for (actual, expected) in grid.values().iter().zip(reference) {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }

        let mut g = c.benchmark_group(format!("meshgrid_mixed_pow2_{nx}x{ny}"));
        g.sample_size(10);
        g.warm_up_time(Duration::from_millis(250));
        g.measurement_time(Duration::from_secs(1));
        g.bench_function("full_divmod_control", |b| {
            b.iter(|| black_box(full_divmod_parallel_meshgrid(black_box(&arrs))))
        });
        g.bench_function("partial_strength_reduce", |b| {
            b.iter(|| {
                black_box(UFuncArray::meshgrid_advanced(
                    black_box(&arrs),
                    "ij",
                    false,
                )
                .unwrap())
            })
        });
        g.finish();
    }

    {
        let (nx, ny) = (2048usize, 2000usize);
        let x = UFuncArray::new(
            vec![nx],
            (0..nx).map(|i| i as f64).collect(),
            DType::F64,
        )
        .unwrap();
        let y = UFuncArray::new(
            vec![ny],
            (0..ny).map(|i| -((i as f64) + 0.5)).collect(),
            DType::F64,
        )
        .unwrap();
        let arrs = vec![x, y];
        let control = partial_strength_reduce_parallel_meshgrid(&arrs);
        let candidate = UFuncArray::meshgrid_advanced(&arrs, "ij", false).unwrap();
        for (grid, reference) in candidate.iter().zip(&control) {
            assert_eq!(grid.values().len(), reference.len());
            for (actual, expected) in grid.values().iter().zip(reference) {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }

        let mut g = c.benchmark_group(format!("meshgrid_block_fill_{nx}x{ny}"));
        g.sample_size(10);
        g.warm_up_time(Duration::from_millis(250));
        g.measurement_time(Duration::from_secs(1));
        g.bench_function("partial_strength_reduce_control", |b| {
            b.iter(|| {
                black_box(partial_strength_reduce_parallel_meshgrid(black_box(
                    &arrs,
                )))
            })
        });
        g.bench_function("block_fill", |b| {
            b.iter(|| {
                black_box(UFuncArray::meshgrid_advanced(
                    black_box(&arrs),
                    "ij",
                    false,
                )
                .unwrap())
            })
        });
        g.finish();
    }

    {
        let (nx, ny) = (2048usize, 2000usize);
        let base = 9_007_199_254_740_992_i64;
        let x = UFuncArray::from_storage(
            vec![nx],
            ArrayStorage::I64((0..nx).map(|i| base + i as i64).collect()),
        )
        .unwrap();
        let y = UFuncArray::from_storage(
            vec![ny],
            ArrayStorage::I64((0..ny).map(|i| base + 10_000 + i as i64).collect()),
        )
        .unwrap();
        let arrs = vec![x, y];
        let control = per_cell_i64_sidecar_meshgrid(&arrs);
        let candidate = UFuncArray::meshgrid_advanced(&arrs, "ij", false).unwrap();
        for (grid, (reference_values, reference_exact)) in candidate.iter().zip(&control) {
            assert_eq!(grid.values().len(), reference_values.len());
            for (actual, expected) in grid.values().iter().zip(reference_values) {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
            match grid.integer_sidecar() {
                Some(IntegerSidecar::I64(exact)) => assert_eq!(exact, reference_exact),
                _ => panic!("candidate lost its i64 sidecar"),
            }
        }

        let mut g = c.benchmark_group(format!("meshgrid_sidecar_block_fill_{nx}x{ny}"));
        g.sample_size(10);
        g.warm_up_time(Duration::from_millis(250));
        g.measurement_time(Duration::from_secs(1));
        g.bench_function("per_cell_index_control", |b| {
            b.iter(|| black_box(per_cell_i64_sidecar_meshgrid(black_box(&arrs))))
        });
        g.bench_function("sidecar_block_fill", |b| {
            b.iter(|| {
                black_box(UFuncArray::meshgrid_advanced(
                    black_box(&arrs),
                    "ij",
                    false,
                )
                .unwrap())
            })
        });
        g.finish();
    }
}
criterion_group!(benches, bench);
criterion_main!(benches);
