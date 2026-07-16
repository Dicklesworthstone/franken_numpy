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

fn former_complex128_add(lhs: &ArrayStorage, rhs: &ArrayStorage) -> ArrayStorage {
    let lhs = lhs.to_complex128_vec();
    let rhs = rhs.to_complex128_vec();
    ArrayStorage::Complex128(
        lhs.iter()
            .zip(&rhs)
            .map(|(&(ar, ai), &(br, bi))| (ar + br, ai + bi))
            .collect(),
    )
}

fn former_complex64_add(lhs: &ArrayStorage, rhs: &ArrayStorage) -> ArrayStorage {
    let lhs = lhs.to_complex128_vec();
    let rhs = rhs.to_complex128_vec();
    ArrayStorage::Complex128(
        lhs.iter()
            .zip(&rhs)
            .map(|(&(ar, ai), &(br, bi))| (ar + br, ai + bi))
            .collect(),
    )
}

/// Faithful replica of the CURRENT Complex64 `complex_sub` path: both inputs
/// materialized via `to_complex128_vec` before subtracting (the shape .343
/// removed for `complex_add`).
fn former_complex64_sub(lhs: &ArrayStorage, rhs: &ArrayStorage) -> ArrayStorage {
    let lhs = lhs.to_complex128_vec();
    let rhs = rhs.to_complex128_vec();
    ArrayStorage::Complex128(
        lhs.iter()
            .zip(&rhs)
            .map(|(&(ar, ai), &(br, bi))| (ar - br, ai - bi))
            .collect(),
    )
}

fn bench_complex64_sub_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex64_sub_direct");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    let lhs = ArrayStorage::Complex64(
        (0..100_000)
            .map(|index| {
                let value = index as f32 * 0.25 - 12_500.0;
                (value, -value * 0.5)
            })
            .collect(),
    );
    let rhs = ArrayStorage::Complex64(
        (0..100_000)
            .map(|index| {
                let value = index as f32 * 0.125 - 6_250.0;
                (-value, value * 0.75)
            })
            .collect(),
    );
    let former = former_complex64_sub(&lhs, &rhs).to_complex128_vec();
    let public = lhs.complex_sub(&rhs).unwrap().to_complex128_vec();
    assert_eq!(public.len(), former.len());
    for (actual, expected) in public.iter().zip(&former) {
        assert_eq!(actual.0.to_bits(), expected.0.to_bits());
        assert_eq!(actual.1.to_bits(), expected.1.to_bits());
    }

    group.bench_function("former_convert_both_inputs", |b| {
        b.iter(|| former_complex64_sub(black_box(&lhs), black_box(&rhs)))
    });
    group.bench_function("direct_inline_widening", |b| {
        b.iter(|| black_box(&lhs).complex_sub(black_box(&rhs)).unwrap())
    });

    group.finish();
}

/// Faithful replica of the CURRENT Complex64 `complex_mul` path: both inputs
/// materialized via `to_complex128_vec` before multiplying (the shape
/// .343/.353 removed for add and sub).
fn former_complex64_mul(lhs: &ArrayStorage, rhs: &ArrayStorage) -> ArrayStorage {
    let lhs = lhs.to_complex128_vec();
    let rhs = rhs.to_complex128_vec();
    ArrayStorage::Complex128(
        lhs.iter()
            .zip(&rhs)
            .map(|(&(ar, ai), &(br, bi))| (ar * br - ai * bi, ar * bi + ai * br))
            .collect(),
    )
}

fn bench_complex64_mul_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex64_mul_direct");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    let lhs = ArrayStorage::Complex64(
        (0..100_000)
            .map(|index| {
                let value = index as f32 * 0.25 - 12_500.0;
                (value, -value * 0.5)
            })
            .collect(),
    );
    let rhs = ArrayStorage::Complex64(
        (0..100_000)
            .map(|index| {
                let value = index as f32 * 0.125 - 6_250.0;
                (-value, value * 0.75)
            })
            .collect(),
    );
    let former = former_complex64_mul(&lhs, &rhs).to_complex128_vec();
    let public = lhs.complex_mul(&rhs).unwrap().to_complex128_vec();
    assert_eq!(public.len(), former.len());
    for (actual, expected) in public.iter().zip(&former) {
        assert_eq!(actual.0.to_bits(), expected.0.to_bits());
        assert_eq!(actual.1.to_bits(), expected.1.to_bits());
    }

    group.bench_function("former_convert_both_inputs", |b| {
        b.iter(|| former_complex64_mul(black_box(&lhs), black_box(&rhs)))
    });
    group.bench_function("direct_inline_widening", |b| {
        b.iter(|| black_box(&lhs).complex_mul(black_box(&rhs)).unwrap())
    });

    group.finish();
}

/// Faithful replica of the CURRENT Complex64 `complex_div` path: both inputs
/// materialized via `to_complex128_vec` before dividing (the last leaf of the
/// .343/.353/.354 convert-both family).
fn former_complex64_div(lhs: &ArrayStorage, rhs: &ArrayStorage) -> ArrayStorage {
    let lhs = lhs.to_complex128_vec();
    let rhs = rhs.to_complex128_vec();
    ArrayStorage::Complex128(
        lhs.iter()
            .zip(&rhs)
            .map(|(&(ar, ai), &(br, bi))| {
                let denom = br * br + bi * bi;
                if denom == 0.0 {
                    (f64::NAN, f64::NAN)
                } else {
                    ((ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom)
                }
            })
            .collect(),
    )
}

fn bench_complex64_div_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex64_div_direct");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    let lhs = ArrayStorage::Complex64(
        (0..100_000)
            .map(|index| {
                let value = index as f32 * 0.25 - 12_500.0;
                (value, -value * 0.5)
            })
            .collect(),
    );
    let rhs = ArrayStorage::Complex64(
        (0..100_000)
            .map(|index| {
                // Includes one exact zero divisor to exercise the NaN arm.
                let value = index as f32 * 0.125 - 6_250.0;
                if index == 77 { (0.0, 0.0) } else { (-value, value * 0.75) }
            })
            .collect(),
    );
    let former = former_complex64_div(&lhs, &rhs).to_complex128_vec();
    let public = lhs.complex_div(&rhs).unwrap().to_complex128_vec();
    assert_eq!(public.len(), former.len());
    for (actual, expected) in public.iter().zip(&former) {
        assert_eq!(actual.0.to_bits(), expected.0.to_bits());
        assert_eq!(actual.1.to_bits(), expected.1.to_bits());
    }

    group.bench_function("former_convert_both_inputs", |b| {
        b.iter(|| former_complex64_div(black_box(&lhs), black_box(&rhs)))
    });
    group.bench_function("direct_inline_widening", |b| {
        b.iter(|| black_box(&lhs).complex_div(black_box(&rhs)).unwrap())
    });

    group.finish();
}

/// Faithful replica of the CURRENT mixed Complex64+Complex128 `complex_add`
/// path: BOTH inputs materialized via `to_complex128_vec` before adding.
fn former_mixed_add(lhs: &ArrayStorage, rhs: &ArrayStorage) -> ArrayStorage {
    let lhs = lhs.to_complex128_vec();
    let rhs = rhs.to_complex128_vec();
    ArrayStorage::Complex128(
        lhs.iter()
            .zip(&rhs)
            .map(|(&(ar, ai), &(br, bi))| (ar + br, ai + bi))
            .collect(),
    )
}

fn bench_complex_mixed_add_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_mixed_add_direct");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    let c64 = ArrayStorage::Complex64(
        (0..100_000)
            .map(|index| {
                let value = index as f32 * 0.25 - 12_500.0;
                (value, -value * 0.5)
            })
            .collect(),
    );
    let c128 = ArrayStorage::Complex128(
        (0..100_000)
            .map(|index| {
                let value = f64::from(index) * 0.125 - 6_250.0;
                (-value, value * 0.75)
            })
            .collect(),
    );
    for (label, lhs, rhs) in [("c64_lhs", &c64, &c128), ("c128_lhs", &c128, &c64)] {
        let former = former_mixed_add(lhs, rhs).to_complex128_vec();
        let public = lhs.complex_add(rhs).unwrap().to_complex128_vec();
        assert_eq!(public.len(), former.len());
        for (actual, expected) in public.iter().zip(&former) {
            assert_eq!(actual.0.to_bits(), expected.0.to_bits());
            assert_eq!(actual.1.to_bits(), expected.1.to_bits());
        }
        group.bench_function(format!("former_convert_both_{label}"), |b| {
            b.iter(|| former_mixed_add(black_box(lhs), black_box(rhs)))
        });
        group.bench_function(format!("direct_borrow_widen_{label}"), |b| {
            b.iter(|| black_box(lhs).complex_add(black_box(rhs)).unwrap())
        });
    }

    group.finish();
}

/// Faithful replicas of the CURRENT mixed-pair sub/mul/div paths: both
/// inputs materialized via `to_complex128_vec` before the op kernel.
fn former_mixed_op(
    lhs: &ArrayStorage,
    rhs: &ArrayStorage,
    kernel: fn((f64, f64), (f64, f64)) -> (f64, f64),
) -> ArrayStorage {
    let lhs = lhs.to_complex128_vec();
    let rhs = rhs.to_complex128_vec();
    ArrayStorage::Complex128(lhs.iter().zip(&rhs).map(|(&a, &b)| kernel(a, b)).collect())
}

fn bench_complex_mixed_ops_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_mixed_ops_direct");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    let c64 = ArrayStorage::Complex64(
        (0..100_000)
            .map(|index| {
                let value = index as f32 * 0.25 - 12_500.0;
                (value, -value * 0.5)
            })
            .collect(),
    );
    let c128 = ArrayStorage::Complex128(
        (0..100_000)
            .map(|index| {
                let value = f64::from(index) * 0.125 - 6_250.0;
                if index == 77 { (0.0, 0.0) } else { (-value, value * 0.75) }
            })
            .collect(),
    );
    type Kernel = fn((f64, f64), (f64, f64)) -> (f64, f64);
    let sub: Kernel = |(ar, ai), (br, bi)| (ar - br, ai - bi);
    let mul: Kernel = |(ar, ai), (br, bi)| (ar * br - ai * bi, ar * bi + ai * br);
    let div: Kernel = |(ar, ai), (br, bi)| {
        let denom = br * br + bi * bi;
        if denom == 0.0 {
            (f64::NAN, f64::NAN)
        } else {
            ((ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom)
        }
    };
    let ops: [(&str, Kernel); 3] = [("sub", sub), ("mul", mul), ("div", div)];
    for (name, kernel) in ops {
        let former = former_mixed_op(&c64, &c128, kernel).to_complex128_vec();
        let public = match name {
            "sub" => c64.complex_sub(&c128),
            "mul" => c64.complex_mul(&c128),
            _ => c64.complex_div(&c128),
        }
        .unwrap()
        .to_complex128_vec();
        assert_eq!(public.len(), former.len());
        for (actual, expected) in public.iter().zip(&former) {
            assert_eq!(actual.0.to_bits(), expected.0.to_bits());
            assert_eq!(actual.1.to_bits(), expected.1.to_bits());
        }
        group.bench_function(format!("former_convert_both_{name}"), |b| {
            b.iter(|| former_mixed_op(black_box(&c64), black_box(&c128), kernel))
        });
        group.bench_function(format!("direct_borrow_widen_{name}"), |b| {
            b.iter(|| {
                match name {
                    "sub" => black_box(&c64).complex_sub(black_box(&c128)),
                    "mul" => black_box(&c64).complex_mul(black_box(&c128)),
                    _ => black_box(&c64).complex_div(black_box(&c128)),
                }
                .unwrap()
            })
        });
    }

    group.finish();
}

/// Faithful replicas of the CURRENT complex unary paths: the input fully
/// materialized via `to_complex128_vec` (a whole-vector copy even for native
/// Complex128) before the transcendental kernel.
fn former_unary_op(
    input: &ArrayStorage,
    kernel: fn((f64, f64)) -> (f64, f64),
) -> ArrayStorage {
    let pairs = input.to_complex128_vec();
    ArrayStorage::Complex128(pairs.iter().map(|&z| kernel(z)).collect())
}

fn bench_complex_unary_borrow(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_unary_borrow");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    let c128 = ArrayStorage::Complex128(
        (0..100_000)
            .map(|index| {
                let value = f64::from(index) * 0.000_125 - 6.25;
                (-value, value * 0.75)
            })
            .collect(),
    );
    let c64 = ArrayStorage::Complex64(
        (0..100_000)
            .map(|index| {
                let value = index as f32 * 0.000_25 - 12.5;
                (value, -value * 0.5)
            })
            .collect(),
    );
    type Kernel = fn((f64, f64)) -> (f64, f64);
    let exp_k: Kernel = |(r, i)| {
        let ea = r.exp();
        (ea * i.cos(), ea * i.sin())
    };
    let log_k: Kernel = |(r, i)| {
        let mag = (r * r + i * i).sqrt();
        let ang = i.atan2(r);
        (mag.ln(), ang)
    };
    let sqrt_k: Kernel = |(r, i)| {
        let mag = (r * r + i * i).sqrt();
        let re = f64::midpoint(mag, r).sqrt();
        let im = f64::midpoint(mag, -r).sqrt();
        (re, if i >= 0.0 { im } else { -im })
    };
    let rows: [(&str, &ArrayStorage, Kernel); 4] = [
        ("exp_c128", &c128, exp_k),
        ("log_c128", &c128, log_k),
        ("sqrt_c128", &c128, sqrt_k),
        ("sqrt_c64", &c64, sqrt_k),
    ];
    for (name, input, kernel) in rows {
        let former = former_unary_op(input, kernel).to_complex128_vec();
        let public = match name {
            n if n.starts_with("exp") => input.complex_exp(),
            n if n.starts_with("log") => input.complex_log(),
            _ => input.complex_sqrt(),
        }
        .to_complex128_vec();
        assert_eq!(public.len(), former.len());
        for (actual, expected) in public.iter().zip(&former) {
            assert_eq!(actual.0.to_bits(), expected.0.to_bits());
            assert_eq!(actual.1.to_bits(), expected.1.to_bits());
        }
        group.bench_function(format!("former_convert_{name}"), |b| {
            b.iter(|| former_unary_op(black_box(input), kernel))
        });
        group.bench_function(format!("direct_borrow_{name}"), |b| {
            b.iter(|| {
                match name {
                    n if n.starts_with("exp") => black_box(input).complex_exp(),
                    n if n.starts_with("log") => black_box(input).complex_log(),
                    _ => black_box(input).complex_sqrt(),
                }
            })
        });
    }

    group.finish();
}

/// Faithful replica of the CURRENT convert-both `complex_pow` path.
fn former_complex_pow(lhs: &ArrayStorage, rhs: &ArrayStorage) -> ArrayStorage {
    let bases = lhs.to_complex128_vec();
    let exps = rhs.to_complex128_vec();
    let n = bases.len().min(exps.len());
    ArrayStorage::Complex128(
        (0..n)
            .map(|idx| {
                let (zr, zi) = bases[idx];
                let (wr, wi) = exps[idx];
                let mag = (zr * zr + zi * zi).sqrt();
                let ang = zi.atan2(zr);
                let log_r = mag.ln();
                let log_i = ang;
                let prod_r = wr * log_r - wi * log_i;
                let prod_i = wr * log_i + wi * log_r;
                let ea = prod_r.exp();
                (ea * prod_i.cos(), ea * prod_i.sin())
            })
            .collect(),
    )
}

fn bench_complex_pow_borrow(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_pow_borrow");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    let c128 = ArrayStorage::Complex128(
        (0..100_000)
            .map(|index| {
                let value = f64::from(index) * 0.000_05 - 2.5;
                (1.5 + value * 0.1, value * 0.25)
            })
            .collect(),
    );
    let c64 = ArrayStorage::Complex64(
        (0..100_000)
            .map(|index| {
                let value = index as f32 * 0.000_1 - 5.0;
                (1.25 - value * 0.05, value * 0.5)
            })
            .collect(),
    );
    for (name, lhs, rhs) in [("c128_c128", &c128, &c128), ("c64_c64", &c64, &c64)] {
        let former = former_complex_pow(lhs, rhs).to_complex128_vec();
        let public = lhs.complex_pow(rhs).to_complex128_vec();
        assert_eq!(public.len(), former.len());
        for (actual, expected) in public.iter().zip(&former) {
            assert_eq!(actual.0.to_bits(), expected.0.to_bits());
            assert_eq!(actual.1.to_bits(), expected.1.to_bits());
        }
        group.bench_function(format!("former_convert_{name}"), |b| {
            b.iter(|| former_complex_pow(black_box(lhs), black_box(rhs)))
        });
        group.bench_function(format!("direct_borrow_{name}"), |b| {
            b.iter(|| black_box(lhs).complex_pow(black_box(rhs)))
        });
    }

    group.finish();
}

fn bench_complex64_add_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex64_add_direct");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));

    let lhs = ArrayStorage::Complex64(
        (0..100_000)
            .map(|index| {
                let value = index as f32 * 0.25 - 12_500.0;
                (value, -value * 0.5)
            })
            .collect(),
    );
    let rhs = ArrayStorage::Complex64(
        (0..100_000)
            .map(|index| {
                let value = index as f32 * 0.125 - 6_250.0;
                (-value, value * 0.75)
            })
            .collect(),
    );
    let former = former_complex64_add(&lhs, &rhs).to_complex128_vec();
    let public = lhs.complex_add(&rhs).unwrap().to_complex128_vec();
    assert_eq!(public.len(), former.len());
    for (actual, expected) in public.iter().zip(&former) {
        assert_eq!(actual.0.to_bits(), expected.0.to_bits());
        assert_eq!(actual.1.to_bits(), expected.1.to_bits());
    }

    group.bench_function("former_generic_control", |b| {
        b.iter(|| former_complex64_add(black_box(&lhs), black_box(&rhs)))
    });
    group.bench_function("public_current_or_direct", |b| {
        b.iter(|| black_box(&lhs).complex_add(black_box(&rhs)).unwrap())
    });

    group.finish();
}

fn bench_complex128_add_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex128_add_direct");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));

    let lhs = ArrayStorage::Complex128(
        (0..100_000)
            .map(|index| {
                let value = f64::from(index) * 0.25 - 12_500.0;
                (value, -value * 0.5)
            })
            .collect(),
    );
    let rhs = ArrayStorage::Complex128(
        (0..100_000)
            .map(|index| {
                let value = f64::from(index) * 0.125 - 6_250.0;
                (-value, value * 0.75)
            })
            .collect(),
    );
    let former = former_complex128_add(&lhs, &rhs).to_complex128_vec();
    let direct = lhs.complex_add(&rhs).unwrap().to_complex128_vec();
    assert_eq!(direct.len(), former.len());
    for (actual, expected) in direct.iter().zip(&former) {
        assert_eq!(actual.0.to_bits(), expected.0.to_bits());
        assert_eq!(actual.1.to_bits(), expected.1.to_bits());
    }

    group.bench_function("former_clone_both_inputs", |b| {
        b.iter(|| former_complex128_add(black_box(&lhs), black_box(&rhs)))
    });
    group.bench_function("direct_borrow_both_inputs", |b| {
        b.iter(|| black_box(&lhs).complex_add(black_box(&rhs)).unwrap())
    });

    group.finish();
}

fn former_complex128_sub(lhs: &ArrayStorage, rhs: &ArrayStorage) -> ArrayStorage {
    let lhs = lhs.to_complex128_vec();
    let rhs = rhs.to_complex128_vec();
    ArrayStorage::Complex128(
        lhs.iter()
            .zip(&rhs)
            .map(|(&(ar, ai), &(br, bi))| (ar - br, ai - bi))
            .collect(),
    )
}

fn bench_complex128_sub_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex128_sub_direct");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));

    let lhs = ArrayStorage::Complex128(
        (0..100_000)
            .map(|index| {
                let value = f64::from(index) * 0.25 - 12_500.0;
                (value, -value * 0.5)
            })
            .collect(),
    );
    let rhs = ArrayStorage::Complex128(
        (0..100_000)
            .map(|index| {
                let value = f64::from(index) * 0.125 - 6_250.0;
                (-value, value * 0.75)
            })
            .collect(),
    );
    let former = former_complex128_sub(&lhs, &rhs).to_complex128_vec();
    let direct = lhs.complex_sub(&rhs).unwrap().to_complex128_vec();
    assert_eq!(direct.len(), former.len());
    for (actual, expected) in direct.iter().zip(&former) {
        assert_eq!(actual.0.to_bits(), expected.0.to_bits());
        assert_eq!(actual.1.to_bits(), expected.1.to_bits());
    }

    group.bench_function("former_clone_both_inputs", |b| {
        b.iter(|| former_complex128_sub(black_box(&lhs), black_box(&rhs)))
    });
    group.bench_function("direct_borrow_both_inputs", |b| {
        b.iter(|| black_box(&lhs).complex_sub(black_box(&rhs)).unwrap())
    });

    group.finish();
}

fn former_complex128_mul(lhs: &ArrayStorage, rhs: &ArrayStorage) -> ArrayStorage {
    let lhs = lhs.to_complex128_vec();
    let rhs = rhs.to_complex128_vec();
    ArrayStorage::Complex128(
        lhs.iter()
            .zip(&rhs)
            .map(|(&(ar, ai), &(br, bi))| (ar * br - ai * bi, ar * bi + ai * br))
            .collect(),
    )
}

fn bench_complex128_mul_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex128_mul_direct");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));

    let lhs = ArrayStorage::Complex128(
        (0..100_000)
            .map(|index| {
                let value = f64::from(index) * 0.25 - 12_500.0;
                (value, -value * 0.5)
            })
            .collect(),
    );
    let rhs = ArrayStorage::Complex128(
        (0..100_000)
            .map(|index| {
                let value = f64::from(index) * 0.125 - 6_250.0;
                (-value, value * 0.75)
            })
            .collect(),
    );
    let former = former_complex128_mul(&lhs, &rhs).to_complex128_vec();
    let direct = lhs.complex_mul(&rhs).unwrap().to_complex128_vec();
    assert_eq!(direct.len(), former.len());
    for (actual, expected) in direct.iter().zip(&former) {
        assert_eq!(actual.0.to_bits(), expected.0.to_bits());
        assert_eq!(actual.1.to_bits(), expected.1.to_bits());
    }

    group.bench_function("former_clone_both_inputs", |b| {
        b.iter(|| former_complex128_mul(black_box(&lhs), black_box(&rhs)))
    });
    group.bench_function("direct_borrow_both_inputs", |b| {
        b.iter(|| black_box(&lhs).complex_mul(black_box(&rhs)).unwrap())
    });

    group.finish();
}

fn former_complex128_div(lhs: &ArrayStorage, rhs: &ArrayStorage) -> ArrayStorage {
    let lhs = lhs.to_complex128_vec();
    let rhs = rhs.to_complex128_vec();
    ArrayStorage::Complex128(
        lhs.iter()
            .zip(&rhs)
            .map(|(&(ar, ai), &(br, bi))| {
                let denom = br * br + bi * bi;
                if denom == 0.0 {
                    (f64::NAN, f64::NAN)
                } else {
                    ((ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom)
                }
            })
            .collect(),
    )
}

fn bench_complex128_div_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex128_div_direct");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));

    let lhs = ArrayStorage::Complex128(
        (0..100_000)
            .map(|index| {
                let value = f64::from(index) * 0.25 - 12_500.0;
                (value, -value * 0.5)
            })
            .collect(),
    );
    let rhs = ArrayStorage::Complex128(
        (0..100_000)
            .map(|index| {
                let value = f64::from(index) * 0.125 - 6_250.0;
                (-value, value * 0.75)
            })
            .collect(),
    );
    let former = former_complex128_div(&lhs, &rhs).to_complex128_vec();
    let direct = lhs.complex_div(&rhs).unwrap().to_complex128_vec();
    assert_eq!(direct.len(), former.len());
    for (actual, expected) in direct.iter().zip(&former) {
        assert_eq!(actual.0.to_bits(), expected.0.to_bits());
        assert_eq!(actual.1.to_bits(), expected.1.to_bits());
    }

    group.bench_function("former_clone_both_inputs", |b| {
        b.iter(|| former_complex128_div(black_box(&lhs), black_box(&rhs)))
    });
    group.bench_function("direct_borrow_both_inputs", |b| {
        b.iter(|| black_box(&lhs).complex_div(black_box(&rhs)).unwrap())
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
    bench_complex64_add_direct,
    bench_complex64_sub_direct,
    bench_complex64_mul_direct,
    bench_complex64_div_direct,
    bench_complex_mixed_add_direct,
    bench_complex_mixed_ops_direct,
    bench_complex_unary_borrow,
    bench_complex_pow_borrow,
    bench_complex128_add_direct,
    bench_complex128_sub_direct,
    bench_complex128_mul_direct,
    bench_complex128_div_direct,
    bench_complex128_sum_direct,
    bench_complex128_prod_direct,
    bench_array_storage_get_set,
);

criterion_main!(benches);
