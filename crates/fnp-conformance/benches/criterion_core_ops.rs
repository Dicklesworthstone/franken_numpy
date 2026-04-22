#![forbid(unsafe_code)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_io::{IOSupportedDType, load, save};
use fnp_ufunc::{BinaryOp, UFuncArray};
use std::hint::black_box;
use std::time::Duration;

fn build_matrix_values(dim: usize, step: usize, modulo: usize) -> Vec<f64> {
    (0..(dim * dim))
        .map(|i| f64::from(((i * step) % modulo) as u32))
        .collect()
}

fn bench_core_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("core_ops");

    let add_dim = 1024usize;
    let add_elements = add_dim * add_dim;
    let add_lhs = UFuncArray::new(
        vec![add_dim, add_dim],
        build_matrix_values(add_dim, 7, 257),
        DType::F64,
    )
    .expect("broadcast lhs setup must succeed");
    let add_rhs = UFuncArray::new(
        vec![add_dim],
        (0..add_dim).map(|i| f64::from((i % 29) as u32)).collect(),
        DType::F64,
    )
    .expect("broadcast rhs setup must succeed");
    group.bench_with_input(
        BenchmarkId::new("ufunc_add_broadcast", "1024x1024_by_1024"),
        &add_elements,
        |b, _| {
            b.iter(|| {
                let out = add_lhs
                    .elementwise_binary(&add_rhs, BinaryOp::Add)
                    .expect("broadcast add must succeed");
                black_box(out.values()[0]);
            });
        },
    );

    let reduce_dim = 1024usize;
    let reduce_in = UFuncArray::new(
        vec![reduce_dim, reduce_dim],
        build_matrix_values(reduce_dim, 17, 509),
        DType::F64,
    )
    .expect("reduction setup must succeed");
    group.bench_function("reduce_sum_axis1_1024x1024", |b| {
        b.iter(|| {
            let out = reduce_in
                .reduce_sum(Some(1), false)
                .expect("axis reduction must succeed");
            black_box(out.values()[0]);
        });
    });

    let matmul_dim = 256usize;
    let matmul_lhs = UFuncArray::new(
        vec![matmul_dim, matmul_dim],
        build_matrix_values(matmul_dim, 13, 997),
        DType::F64,
    )
    .expect("matmul lhs setup must succeed");
    let matmul_rhs = UFuncArray::new(
        vec![matmul_dim, matmul_dim],
        build_matrix_values(matmul_dim, 19, 991),
        DType::F64,
    )
    .expect("matmul rhs setup must succeed");
    group.bench_function("matmul_256x256_by_256x256", |b| {
        b.iter(|| {
            let out = matmul_lhs.matmul(&matmul_rhs).expect("matmul must succeed");
            black_box(out.values()[0]);
        });
    });

    let sort_len = 1_000_000usize;
    let sort_in = UFuncArray::new(
        vec![sort_len],
        (0..sort_len)
            .map(|i| f64::from(((i * 48_271) % sort_len) as u32))
            .collect(),
        DType::F64,
    )
    .expect("sort setup must succeed");
    group.bench_function("sort_quicksort_1m", |b| {
        b.iter(|| {
            let out = sort_in
                .sort(None, Some("quicksort"))
                .expect("sort must succeed");
            black_box(out.values()[0]);
        });
    });

    let fft_len = 65_536usize;
    let fft_in = UFuncArray::new(
        vec![fft_len],
        (0..fft_len)
            .map(|i| {
                let t = i as f64 / fft_len as f64;
                (std::f64::consts::TAU * 5.0 * t).sin()
                    + 0.5 * (std::f64::consts::TAU * 13.0 * t).cos()
            })
            .collect(),
        DType::F64,
    )
    .expect("fft setup must succeed");
    group.bench_function("fft_65536", |b| {
        b.iter(|| {
            let out = fft_in.fft(None).expect("fft must succeed");
            black_box(out.values()[0]);
        });
    });

    let astype_dim = 1024usize;
    let astype_elements = astype_dim * astype_dim;
    let astype_in = UFuncArray::new(
        vec![astype_dim, astype_dim],
        build_matrix_values(astype_dim, 23, 10_003),
        DType::F64,
    )
    .expect("astype setup must succeed");
    group.bench_with_input(
        BenchmarkId::new("astype_f64_to_i32", "1024x1024"),
        &astype_elements,
        |b, _| {
            b.iter(|| {
                let out = astype_in.astype(DType::I32);
                black_box(out.values()[0]);
            });
        },
    );

    group.bench_function("reshape_1024x1024_to_2048x512", |b| {
        b.iter(|| {
            let out = astype_in
                .reshape(&[2048, 512])
                .expect("reshape must succeed");
            black_box(out.shape()[0]);
        });
    });

    let io_dim = 512usize;
    let io_values: Vec<f64> = (0..(io_dim * io_dim))
        .map(|i| f64::from(((i * 29) % 65_537) as u32) / 11.0)
        .collect();
    group.bench_function("io_npy_save_load_512x512_f64", |b| {
        b.iter(|| {
            let payload = save(&[io_dim, io_dim], &io_values, IOSupportedDType::F64).expect("save");
            let (shape, values, dtype) = load(&payload).expect("load");
            black_box(shape);
            black_box(values[0]);
            black_box(dtype);
        });
    });

    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .measurement_time(Duration::from_secs(8))
        .warm_up_time(Duration::from_secs(2))
        .sample_size(12)
        .configure_from_args()
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_core_ops
}
criterion_main!(benches);
