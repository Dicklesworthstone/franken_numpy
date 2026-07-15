//! Performance benchmarks for fnp-dtype core operations.
//!
//! These benchmarks establish baselines for dtype promotion, casting checks,
//! and type inference - all hot-path operations in array computations.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_dtype::{
    ArrayStorage, DType, can_cast, common_type, min_scalar_type, promote, result_type,
};
use std::hint::black_box;
use std::time::Duration;

fn bench_result_type(c: &mut Criterion) {
    let mut group = c.benchmark_group("result_type");

    let pairs = [
        ("i32_f64", vec![DType::I32, DType::F64]),
        ("u8_i16", vec![DType::U8, DType::I16]),
        ("f32_complex128", vec![DType::F32, DType::Complex128]),
        ("bool_u64", vec![DType::Bool, DType::U64]),
    ];

    for (name, dtypes) in &pairs {
        group.bench_with_input(BenchmarkId::new("pair", name), dtypes, |b, dtypes| {
            b.iter(|| result_type(black_box(dtypes)))
        });
        group.bench_with_input(
            BenchmarkId::new("bool_seed_control", name),
            dtypes,
            |b, dtypes| {
                b.iter(|| {
                    black_box(dtypes)
                        .iter()
                        .copied()
                        .fold(DType::Bool, promote)
                })
            },
        );
    }

    let triple = vec![DType::I32, DType::F32, DType::U16];
    group.bench_function("triple", |b| b.iter(|| result_type(black_box(&triple))));

    let many: Vec<DType> = vec![
        DType::Bool,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
        DType::F32,
    ];
    group.bench_function("ten_types", |b| b.iter(|| result_type(black_box(&many))));

    group.finish();
}

fn bench_can_cast(c: &mut Criterion) {
    let mut group = c.benchmark_group("can_cast");

    let cases = [
        ("i32_to_f64_safe", DType::I32, DType::F64, "safe"),
        ("f64_to_i32_unsafe", DType::F64, DType::I32, "unsafe"),
        ("u8_to_i16_safe", DType::U8, DType::I16, "safe"),
        (
            "complex_to_f64_same_kind",
            DType::Complex128,
            DType::F64,
            "same_kind",
        ),
        ("bool_to_i64_safe", DType::Bool, DType::I64, "safe"),
    ];

    for (name, from, to, casting) in &cases {
        group.bench_with_input(
            BenchmarkId::new("check", name),
            &(*from, *to, *casting),
            |b, (from, to, casting)| {
                b.iter(|| can_cast(black_box(*from), black_box(*to), black_box(casting)))
            },
        );
    }

    group.finish();
}

fn bench_min_scalar_type(c: &mut Criterion) {
    let mut group = c.benchmark_group("min_scalar_type");

    let values = [
        ("zero", 0.0),
        ("small_int", 42.0),
        ("large_int", 1e15),
        ("small_float", 0.001),
        ("large_float", 1e300),
        ("negative", -123.456),
    ];

    for (name, val) in &values {
        group.bench_with_input(BenchmarkId::new("infer", name), val, |b, val| {
            b.iter(|| min_scalar_type(black_box(*val)))
        });
    }

    group.finish();
}

fn common_type_control(dtypes: &[DType]) -> DType {
    let mut iter = dtypes.iter().copied();
    let Some(first) = iter.next() else {
        return DType::F64;
    };
    let mut result = if first.is_float() || first.is_complex() {
        first
    } else {
        DType::F64
    };
    for dt in iter {
        let as_float = if dt.is_float() || dt.is_complex() {
            dt
        } else {
            DType::F64
        };
        result = promote(result, as_float);
    }
    result
}

fn bench_common_type(c: &mut Criterion) {
    let mut group = c.benchmark_group("common_type");

    let pairs = [
        ("f32_f64", vec![DType::F32, DType::F64]),
        ("i32_f32", vec![DType::I32, DType::F32]),
        (
            "complex64_complex128",
            vec![DType::Complex64, DType::Complex128],
        ),
    ];

    for (name, dtypes) in &pairs {
        group.bench_with_input(BenchmarkId::new("pair", name), dtypes, |b, dtypes| {
            b.iter(|| common_type(black_box(dtypes)))
        });
        group.bench_with_input(BenchmarkId::new("control", name), dtypes, |b, dtypes| {
            b.iter(|| common_type_control(black_box(dtypes)))
        });
    }

    let complex128_head = vec![
        DType::Complex128,
        DType::I32,
        DType::F64,
        DType::U64,
        DType::Complex64,
        DType::F32,
        DType::I64,
        DType::Bool,
    ];
    group.bench_function("complex128_head", |b| {
        b.iter(|| common_type(black_box(&complex128_head)))
    });
    group.bench_function("complex128_head_control", |b| {
        b.iter(|| common_type_control(black_box(&complex128_head)))
    });

    group.finish();
}

fn bench_dtype_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtype_parse");

    let names = [
        ("float64", "float64"),
        ("int32", "int32"),
        ("f4", "f4"),
        ("i8", "i8"),
        ("complex128", "complex128"),
        ("bool", "bool"),
    ];

    for (label, name) in &names {
        group.bench_with_input(BenchmarkId::new("parse", label), name, |b, name| {
            b.iter(|| DType::parse(black_box(name)))
        });
    }

    group.finish();
}

fn bench_array_storage_cast(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_storage_cast");

    let sizes = [100, 1000, 10000, 100_000];

    for &size in &sizes {
        let i32_data: Vec<i32> = (0..size).collect();
        let storage = ArrayStorage::I32(i32_data);

        group.bench_with_input(
            BenchmarkId::new("i32_to_f64", size),
            &storage,
            |b, storage| b.iter(|| storage.cast_to(black_box(DType::F64))),
        );
    }

    for &size in &sizes {
        let f64_data: Vec<f64> = (0..size).map(|x| x as f64 * 0.1).collect();
        let storage = ArrayStorage::F64(f64_data);

        group.bench_with_input(
            BenchmarkId::new("f64_to_i32", size),
            &storage,
            |b, storage| b.iter(|| storage.cast_to(black_box(DType::I32))),
        );
    }

    // Integer-to-integer path (goes through the i128 intermediary).
    for &size in &sizes {
        let i64_data: Vec<i64> = (0..size as i64).collect();
        let storage = ArrayStorage::I64(i64_data);

        group.bench_with_input(
            BenchmarkId::new("i64_to_i32", size),
            &storage,
            |b, storage| b.iter(|| storage.cast_to(black_box(DType::I32))),
        );
    }

    group.finish();
}

fn former_i64_to_i32_cast(values: &[i64]) -> ArrayStorage {
    let mut out = vec![0i32; values.len()];
    let staged: Vec<i128> = values.iter().map(|&value| i128::from(value)).collect();
    out.iter_mut()
        .zip(&staged)
        .for_each(|(slot, &value)| *slot = value as i32);
    ArrayStorage::I32(out)
}

fn bench_i64_to_i32_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_storage_cast_i64_to_i32_direct");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_secs(1));

    let values: Vec<i64> = (0..100_000i64)
        .map(|index| match index % 4 {
            0 => i64::MIN.wrapping_add(index),
            1 => i64::MAX.wrapping_sub(index),
            2 => index.wrapping_mul(1_000_003),
            _ => index.wrapping_mul(-1_000_003),
        })
        .collect();
    let storage = ArrayStorage::I64(values.clone());
    let former = former_i64_to_i32_cast(&values);
    let direct = storage.cast_to(DType::I32).unwrap();
    assert_eq!(direct, former, "direct cast changed i64-to-i32 wrapping");

    group.bench_function("former_i128_staging", |b| {
        b.iter(|| former_i64_to_i32_cast(black_box(&values)))
    });
    group.bench_function("direct_typed_collect", |b| {
        b.iter(|| black_box(&storage).cast_to(black_box(DType::I32)).unwrap())
    });

    group.finish();
}

fn former_u64_to_u32_cast(values: &[u64]) -> ArrayStorage {
    let mut out = vec![0u32; values.len()];
    let staged: Vec<i128> = values.iter().map(|&value| i128::from(value)).collect();
    out.iter_mut()
        .zip(&staged)
        .for_each(|(slot, &value)| *slot = value as u32);
    ArrayStorage::U32(out)
}

fn bench_u64_to_u32_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_storage_cast_u64_to_u32_direct");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));

    let values: Vec<u64> = (0..100_000u64)
        .map(|index| match index % 4 {
            0 => index,
            1 => u64::MAX.wrapping_sub(index),
            2 => u64::from(u32::MAX).wrapping_add(index),
            _ => index.wrapping_mul(0x1_0000_0001),
        })
        .collect();
    let storage = ArrayStorage::U64(values.clone());
    let former = former_u64_to_u32_cast(&values);
    let direct = storage.cast_to(DType::U32).unwrap();
    assert_eq!(direct, former, "direct cast changed u64-to-u32 wrapping");

    group.bench_function("former_i128_staging", |b| {
        b.iter(|| former_u64_to_u32_cast(black_box(&values)))
    });
    group.bench_function("direct_typed_collect", |b| {
        b.iter(|| black_box(&storage).cast_to(black_box(DType::U32)).unwrap())
    });

    group.finish();
}

fn bench_to_f64_vec(c: &mut Criterion) {
    let mut group = c.benchmark_group("to_f64_vec");

    let size = 100_000usize;

    let f64_storage = ArrayStorage::F64((0..size).map(|x| x as f64 * 0.5).collect());
    group.bench_function("f64", |b| b.iter(|| black_box(&f64_storage).to_f64_vec()));

    let f32_storage = ArrayStorage::F32((0..size).map(|x| x as f32 * 0.5).collect());
    group.bench_function("f32", |b| b.iter(|| black_box(&f32_storage).to_f64_vec()));

    let i32_storage = ArrayStorage::I32((0..size as i32).collect());
    group.bench_function("i32", |b| b.iter(|| black_box(&i32_storage).to_f64_vec()));

    let i64_storage = ArrayStorage::I64((0..size as i64).collect());
    group.bench_function("i64", |b| b.iter(|| black_box(&i64_storage).to_f64_vec()));

    group.finish();
}

fn bench_to_complex128_vec(c: &mut Criterion) {
    let mut group = c.benchmark_group("to_complex128_vec");

    let size = 100_000usize;

    let f64_storage = ArrayStorage::F64((0..size).map(|x| x as f64 * 0.5).collect());
    group.bench_function("f64", |b| {
        b.iter(|| black_box(&f64_storage).to_complex128_vec())
    });

    let f32_storage = ArrayStorage::F32((0..size).map(|x| x as f32 * 0.5).collect());
    group.bench_function("f32", |b| {
        b.iter(|| black_box(&f32_storage).to_complex128_vec())
    });

    let i32_storage = ArrayStorage::I32((0..size as i32).collect());
    group.bench_function("i32", |b| {
        b.iter(|| black_box(&i32_storage).to_complex128_vec())
    });

    let i64_storage = ArrayStorage::I64((0..size as i64).collect());
    group.bench_function("i64", |b| {
        b.iter(|| black_box(&i64_storage).to_complex128_vec())
    });

    let complex128_storage =
        ArrayStorage::Complex128((0..size).map(|x| (x as f64, -(x as f64))).collect());
    group.bench_function("complex128", |b| {
        b.iter(|| black_box(&complex128_storage).to_complex128_vec())
    });

    group.finish();
}

fn former_complex128_sum(storage: &ArrayStorage) -> (f64, f64) {
    let pairs = storage.to_complex128_vec();
    pairs
        .iter()
        .fold((0.0, 0.0), |(sr, si), &(r, i)| (sr + r, si + i))
}

fn bench_complex128_sum_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex128_sum_direct");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));

    let storage = ArrayStorage::Complex128(
        (0..100_000)
            .map(|index| {
                let real = f64::from(index) * 0.25 - 12_500.0;
                (real, -real * 0.5)
            })
            .collect(),
    );
    let former = former_complex128_sum(&storage);
    let direct = storage.complex_sum();
    assert_eq!(direct.0.to_bits(), former.0.to_bits());
    assert_eq!(direct.1.to_bits(), former.1.to_bits());

    group.bench_function("former_clone_then_fold", |b| {
        b.iter(|| former_complex128_sum(black_box(&storage)))
    });
    group.bench_function("direct_borrowed_fold", |b| {
        b.iter(|| black_box(&storage).complex_sum())
    });

    group.finish();
}

fn former_complex128_prod(storage: &ArrayStorage) -> (f64, f64) {
    let pairs = storage.to_complex128_vec();
    pairs.iter().fold((1.0, 0.0), |(pr, pi), &(r, i)| {
        (pr * r - pi * i, pr * i + pi * r)
    })
}

fn bench_complex128_prod_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex128_prod_direct");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));

    let storage = ArrayStorage::Complex128(
        (0..100_000)
            .map(|index| {
                let offset = f64::from(index % 7) - 3.0;
                (1.0 + offset * 1e-8, offset * 1e-7)
            })
            .collect(),
    );
    let former = former_complex128_prod(&storage);
    let direct = storage.complex_prod();
    assert_eq!(direct.0.to_bits(), former.0.to_bits());
    assert_eq!(direct.1.to_bits(), former.1.to_bits());

    group.bench_function("former_clone_then_fold", |b| {
        b.iter(|| former_complex128_prod(black_box(&storage)))
    });
    group.bench_function("direct_borrowed_fold", |b| {
        b.iter(|| black_box(&storage).complex_prod())
    });

    group.finish();
}

fn bench_array_storage_get_set(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_storage_access");

    let size = 10000;
    let f64_data: Vec<f64> = (0..size).map(|x| x as f64).collect();
    let storage = ArrayStorage::F64(f64_data);

    group.bench_function("get_f64_sequential", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for i in 0..100 {
                sum += storage.get_f64(black_box(i)).unwrap_or(0.0);
            }
            sum
        })
    });

    let i64_data: Vec<i64> = (0..size as i64).collect();
    let storage_i64 = ArrayStorage::I64(i64_data);

    group.bench_function("get_i128_sequential", |b| {
        b.iter(|| {
            let mut sum: i128 = 0;
            for i in 0..100 {
                sum += storage_i64.get_i128(black_box(i)).unwrap_or(0);
            }
            sum
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_result_type,
    bench_can_cast,
    bench_min_scalar_type,
    bench_common_type,
    bench_dtype_parse,
    bench_array_storage_cast,
    bench_i64_to_i32_direct,
    bench_u64_to_u32_direct,
    bench_to_f64_vec,
    bench_to_complex128_vec,
    bench_complex128_sum_direct,
    bench_complex128_prod_direct,
    bench_array_storage_get_set,
);

criterion_main!(benches);
