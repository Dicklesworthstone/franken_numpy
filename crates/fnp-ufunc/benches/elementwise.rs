use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::{MaskedArray, UFuncArray, UnaryOp, add, divide, multiply, subtract, where_nonzero};
use std::hint::black_box;

fn make_array(n: usize) -> UFuncArray {
    let values: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 + 1.0).collect();
    UFuncArray::new(vec![n], values, DType::F64).unwrap()
}

fn make_sign_array(n: usize) -> UFuncArray {
    let values: Vec<f64> = (0..n)
        .map(|i| match i % 8 {
            0 => -((i as f64) + 1.0),
            1 => (i as f64) + 1.0,
            2 => -0.0,
            3 => 0.0,
            4 => f64::NAN,
            5 => -f64::INFINITY,
            6 => f64::INFINITY,
            _ => (i as f64) - 3.0,
        })
        .collect();
    UFuncArray::new(vec![n], values, DType::F64).unwrap()
}

fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_add");
    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let a = make_array(*size);
        let b = make_array(*size);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| add(black_box(&a), black_box(&b)).unwrap())
        });
    }
    group.finish();
}

fn bench_subtract(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_subtract");
    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let a = make_array(*size);
        let b = make_array(*size);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| subtract(black_box(&a), black_box(&b)).unwrap())
        });
    }
    group.finish();
}

fn bench_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_multiply");
    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let a = make_array(*size);
        let b = make_array(*size);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| multiply(black_box(&a), black_box(&b)).unwrap())
        });
    }
    group.finish();
}

fn bench_divide(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_divide");
    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let a = make_array(*size);
        let b = make_array(*size);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| divide(black_box(&a), black_box(&b)).unwrap())
        });
    }
    group.finish();
}

fn bench_chained_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("chained_ops");
    for size in [1_000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let a = make_array(*size);
        let b = make_array(*size);
        let c_arr = make_array(*size);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let t1 = add(black_box(&a), black_box(&b)).unwrap();
                let t2 = multiply(black_box(&t1), black_box(&c_arr)).unwrap();
                subtract(black_box(&t2), black_box(&a)).unwrap()
            })
        });
    }
    group.finish();
}

fn bench_from_storage_f64_move(c: &mut Criterion) {
    let mut group = c.benchmark_group("from_storage_f64_move");
    for size in [100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let template: Vec<f64> = (0..*size).map(|i| (i as f64) * 0.25).collect();
        let shape = vec![*size];
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter_batched(
                || ArrayStorage::F64(template.clone()),
                |storage| UFuncArray::from_storage(black_box(shape.clone()), black_box(storage)),
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

fn bench_sign(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_sign");
    for size in [100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let a = make_sign_array(*size);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| black_box(&a).elementwise_unary(UnaryOp::Sign))
        });
    }
    group.finish();
}

fn bench_boolean_set_f64_masked(c: &mut Criterion) {
    let mut group = c.benchmark_group("boolean_set_f64_masked");
    for size in [100_000usize, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let dst_template = make_array(*size);
        let mask = UFuncArray::new(
            vec![*size],
            (0..*size)
                .map(|i| {
                    if matches!(i % 17, 0 | 4 | 9 | 15) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            DType::Bool,
        )
        .unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter_batched(
                || dst_template.clone(),
                |mut dst| {
                    dst.boolean_set(black_box(&mask), black_box(-0.0)).unwrap();
                    dst
                },
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

fn bench_extract_f64_masked(c: &mut Criterion) {
    let mut group = c.benchmark_group("extract_f64_masked");
    for size in [100_000usize, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let arr = make_sign_array(*size);
        let condition = UFuncArray::new(
            vec![*size],
            (0..*size)
                .map(|i| {
                    if matches!(i % 19, 0 | 3 | 7 | 14) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            DType::Bool,
        )
        .unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| UFuncArray::extract(black_box(&condition), black_box(&arr)).unwrap())
        });
    }
    group.finish();
}

fn bench_compress_f64_bool_flat_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress_f64_bool_flat_sparse");
    for size in [100_000usize, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let arr = make_sign_array(*size);
        let condition: Vec<bool> = (0..*size)
            .map(|i| (i % 181 == 0) || matches!((i * 41 + 17) % 23, 0 | 3 | 8 | 13 | 21))
            .collect();
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                black_box(&arr)
                    .compress(black_box(&condition), None)
                    .unwrap()
            })
        });
    }
    group.finish();
}

fn bench_boolean_index_f64_masked_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("boolean_index_f64_masked_sparse");
    for size in [100_000usize, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let arr = make_sign_array(*size);
        let mask = UFuncArray::new(
            vec![*size],
            (0..*size)
                .map(|i| {
                    if i % 197 == 0 {
                        f64::NAN
                    } else if i % 173 == 0 {
                        -0.0
                    } else if matches!((i * 43 + 11) % 29, 0 | 5 | 9 | 17 | 23) {
                        2.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            DType::Bool,
        )
        .unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| black_box(&arr).boolean_index(black_box(&mask)).unwrap())
        });
    }
    group.finish();
}

fn bench_delete_flat_f64_sparse_indices(c: &mut Criterion) {
    let mut group = c.benchmark_group("delete_flat_f64_sparse_indices");
    for size in [100_000usize, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let arr = make_sign_array(*size);
        let mut indices: Vec<usize> = (0..*size)
            .rev()
            .filter(|&i| i % 251 == 0 || matches!((i * 41 + 19) % 113, 0 | 7 | 31))
            .collect();
        indices.extend_from_slice(&[*size / 2, 0, *size - 1, 251, 251]);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| black_box(&arr).delete(black_box(&indices), None).unwrap())
        });
    }
    group.finish();
}

fn bench_insert_flat_f64_midpoint_many(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_flat_f64_midpoint_many");
    for size in [100_000usize, 1_000_000].iter() {
        group.throughput(Throughput::Elements((*size + 313) as u64));
        let arr = make_sign_array(*size);
        let insert_values = make_sign_array(313);
        let index = *size / 2 + 17;
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                black_box(&arr)
                    .insert(black_box(index), black_box(&insert_values), None)
                    .unwrap()
            })
        });
    }
    group.finish();
}

fn bench_flatnonzero_f64_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("flatnonzero_f64_sparse");
    for size in [100_000usize, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let arr = UFuncArray::new(
            vec![*size],
            (0..*size)
                .map(|i| {
                    if i % 173 == 0 {
                        f64::NAN
                    } else if i % 149 == 0 {
                        -0.0
                    } else if i % 131 == 0 {
                        f64::INFINITY
                    } else if matches!((i * 31 + 7) % 23, 0 | 5 | 11 | 19) {
                        ((i * 37 + 11) % 2003) as f64 - 1001.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            DType::F64,
        )
        .unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| black_box(&arr).flatnonzero())
        });
    }
    group.finish();
}

fn bench_count_nonzero_flat_f64_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("count_nonzero_flat_f64_sparse");
    for size in [100_000usize, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let arr = UFuncArray::new(
            vec![*size],
            (0..*size)
                .map(|i| {
                    if i % 181 == 0 {
                        f64::NAN
                    } else if i % 163 == 0 {
                        -0.0
                    } else if i % 127 == 0 {
                        f64::NEG_INFINITY
                    } else if matches!((i * 41 + 13) % 29, 0 | 3 | 7 | 17 | 23) {
                        ((i * 43 + 19) % 3001) as f64 + 0.5
                    } else {
                        0.0
                    }
                })
                .collect(),
            DType::F64,
        )
        .unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| black_box(&arr).count_nonzero(None, false).unwrap())
        });
    }
    group.finish();
}

fn bench_where_nonzero_f64_2d_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("where_nonzero_f64_2d_sparse");
    for rows in [512usize, 1024].iter() {
        let cols = *rows;
        let size = *rows * cols;
        group.throughput(Throughput::Elements(size as u64));
        let arr = UFuncArray::new(
            vec![*rows, cols],
            (0..size)
                .map(|i| {
                    if i % 191 == 0 {
                        f64::NAN
                    } else if i % 167 == 0 {
                        -0.0
                    } else if i % 137 == 0 {
                        f64::NEG_INFINITY
                    } else if matches!((i * 43 + 17) % 31, 0 | 5 | 11 | 19 | 23) {
                        ((i * 47 + 29) % 4001) as f64 + 0.25
                    } else {
                        0.0
                    }
                })
                .collect(),
            DType::F64,
        )
        .unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bench, _| {
            bench.iter(|| where_nonzero(black_box(&arr)).unwrap())
        });
    }
    group.finish();
}

fn bench_argwhere_f64_2d_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("argwhere_f64_2d_sparse");
    for rows in [512usize, 1024].iter() {
        let cols = *rows;
        let size = *rows * cols;
        group.throughput(Throughput::Elements(size as u64));
        let arr = UFuncArray::new(
            vec![*rows, cols],
            (0..size)
                .map(|i| {
                    if i % 193 == 0 {
                        f64::NAN
                    } else if i % 151 == 0 {
                        -0.0
                    } else if i % 139 == 0 {
                        f64::INFINITY
                    } else if matches!((i * 37 + 23) % 29, 0 | 4 | 9 | 16 | 21) {
                        ((i * 53 + 31) % 5003) as f64 - 2501.5
                    } else {
                        0.0
                    }
                })
                .collect(),
            DType::F64,
        )
        .unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bench, _| {
            bench.iter(|| black_box(&arr).argwhere())
        });
    }
    group.finish();
}

fn bench_copyto_equal_shape_masked(c: &mut Criterion) {
    let mut group = c.benchmark_group("copyto_equal_shape_masked");
    for size in [100_000usize, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let dst_template = make_array(*size);
        let src = make_sign_array(*size);
        let mask = UFuncArray::new(
            vec![*size],
            (0..*size)
                .map(|i| {
                    if matches!(i % 11, 0 | 3 | 7) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            DType::Bool,
        )
        .unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter_batched(
                || dst_template.clone(),
                |mut dst| {
                    dst.copyto(black_box(&src), Some(black_box(&mask)), None)
                        .unwrap();
                    dst
                },
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

fn bench_putmask_f64_masked(c: &mut Criterion) {
    let mut group = c.benchmark_group("putmask_f64_masked");
    for size in [100_000usize, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let dst_template = make_array(*size);
        let values = make_sign_array(257);
        let mask = UFuncArray::new(
            vec![*size],
            (0..*size)
                .map(|i| {
                    if matches!(i % 13, 0 | 2 | 8) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            DType::Bool,
        )
        .unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter_batched(
                || dst_template.clone(),
                |mut dst| {
                    dst.putmask(black_box(&mask), black_box(&values)).unwrap();
                    dst
                },
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

fn bench_place_f64_masked_cycling(c: &mut Criterion) {
    let mut group = c.benchmark_group("place_f64_masked_cycling");
    for size in [100_000usize, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let dst_template = make_array(*size);
        let values = make_sign_array(257);
        let mask = UFuncArray::new(
            vec![*size],
            (0..*size)
                .map(|i| {
                    if matches!(i % 19, 3 | 7 | 11) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            DType::Bool,
        )
        .unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter_batched(
                || dst_template.clone(),
                |mut dst| {
                    dst.place(black_box(&mask), black_box(&values)).unwrap();
                    dst
                },
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

fn bench_put_mask_f64_masked_cycling(c: &mut Criterion) {
    let mut group = c.benchmark_group("put_mask_f64_masked_cycling");
    for size in [100_000usize, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let dst_template = make_array(*size);
        let values: Vec<f64> = make_sign_array(193).values().to_vec();
        let mask = UFuncArray::new(
            vec![*size],
            (0..*size)
                .map(|i| {
                    if matches!((i * 37 + 5) % 41, 0 | 3 | 11 | 17 | 29) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            DType::Bool,
        )
        .unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter_batched(
                || dst_template.clone(),
                |mut dst| {
                    dst.put_mask(black_box(&mask), black_box(&values)).unwrap();
                    dst
                },
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

#[inline(never)]
fn polyval_degree_zero_former(coeffs: &UFuncArray, x: &UFuncArray) -> UFuncArray {
    let values = x
        .values()
        .iter()
        .map(|&xi| {
            let mut result = 0.0;
            for &coefficient in coeffs.values() {
                result = result * xi + coefficient;
            }
            result
        })
        .collect();
    UFuncArray::new(x.shape().to_vec(), values, DType::F64).unwrap()
}

fn bench_polyval_degree_zero(c: &mut Criterion) {
    let mut group = c.benchmark_group("polyval_degree_zero");
    let size = 1_000_000usize;
    group.throughput(Throughput::Elements(size as u64));
    let x = make_sign_array(size);
    let coeffs = UFuncArray::new(vec![1], vec![-0.0], DType::F64).unwrap();

    group.bench_function("former_dynamic_horner", |bench| {
        bench.iter(|| polyval_degree_zero_former(black_box(&coeffs), black_box(&x)))
    });
    group.bench_function("specialized", |bench| {
        bench.iter(|| UFuncArray::polyval(black_box(&coeffs), black_box(&x)).unwrap())
    });
    group.finish();
}

#[inline(never)]
fn masked_count_axis_no_mask_former(data: &UFuncArray, axis: isize) -> UFuncArray {
    UFuncArray::ones(data.shape().to_vec(), DType::F64)
        .unwrap()
        .reduce_sum(Some(axis), false)
        .unwrap()
}

fn assert_ufunc_array_bits_eq(lhs: &UFuncArray, rhs: &UFuncArray) {
    assert_eq!(lhs.shape(), rhs.shape());
    assert_eq!(lhs.dtype(), rhs.dtype());
    assert_eq!(lhs.integer_sidecar(), rhs.integer_sidecar());
    assert_eq!(lhs.values().len(), rhs.values().len());
    assert!(
        lhs.values()
            .iter()
            .zip(rhs.values())
            .all(|(&left, &right)| left.to_bits() == right.to_bits())
    );
}

fn bench_masked_count_axis_no_mask(c: &mut Criterion) {
    let mut group = c.benchmark_group("masked_count_axis_no_mask");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_millis(250));
    group.measurement_time(std::time::Duration::from_millis(750));

    let shape = vec![4096, 1024];
    let data =
        UFuncArray::new(shape.clone(), vec![7.0; shape.iter().product()], DType::F64).unwrap();
    let masked = MaskedArray::new(data.clone(), None, None).unwrap();
    let candidate = masked.count(Some(1)).unwrap();
    let former = masked_count_axis_no_mask_former(&data, 1);
    assert_ufunc_array_bits_eq(&candidate, &former);

    group.bench_function("former_exact_ones_reduce", |bench| {
        bench.iter(|| masked_count_axis_no_mask_former(black_box(&data), black_box(1)))
    });
    group.bench_function("shape_metadata", |bench| {
        bench.iter(|| black_box(&masked).count(black_box(Some(1))).unwrap())
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_add,
    bench_subtract,
    bench_multiply,
    bench_divide,
    bench_chained_ops,
    bench_from_storage_f64_move,
    bench_sign,
    bench_boolean_set_f64_masked,
    bench_extract_f64_masked,
    bench_compress_f64_bool_flat_sparse,
    bench_boolean_index_f64_masked_sparse,
    bench_delete_flat_f64_sparse_indices,
    bench_insert_flat_f64_midpoint_many,
    bench_flatnonzero_f64_sparse,
    bench_count_nonzero_flat_f64_sparse,
    bench_where_nonzero_f64_2d_sparse,
    bench_argwhere_f64_2d_sparse,
    bench_copyto_equal_shape_masked,
    bench_putmask_f64_masked,
    bench_place_f64_masked_cycling,
    bench_put_mask_f64_masked_cycling,
    bench_polyval_degree_zero,
    bench_masked_count_axis_no_mask
);
criterion_main!(benches);
