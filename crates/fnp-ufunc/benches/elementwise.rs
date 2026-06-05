use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fnp_dtype::{ArrayStorage, DType};
use fnp_ufunc::{UFuncArray, UnaryOp, add, divide, multiply, subtract};
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

criterion_group!(
    benches,
    bench_add,
    bench_subtract,
    bench_multiply,
    bench_divide,
    bench_chained_ops,
    bench_from_storage_f64_move,
    bench_sign
);
criterion_main!(benches);
