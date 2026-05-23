use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_ufunc::{UFuncArray, UnaryOp};
use std::hint::black_box;

fn make_array(n: usize) -> UFuncArray {
    let values: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01 + 0.1).collect();
    UFuncArray::new(vec![n], values, DType::F64).unwrap()
}

fn make_trig_array(n: usize) -> UFuncArray {
    let values: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
    UFuncArray::new(vec![n], values, DType::F64).unwrap()
}

fn make_domain_array(n: usize) -> UFuncArray {
    let values: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64) * 0.999 - 0.4995).collect();
    UFuncArray::new(vec![n], values, DType::F64).unwrap()
}

fn bench_exp(c: &mut Criterion) {
    let mut group = c.benchmark_group("transcendental_exp");
    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let a = make_array(*size);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| black_box(&a).elementwise_unary(UnaryOp::Exp))
        });
    }
    group.finish();
}

fn bench_log(c: &mut Criterion) {
    let mut group = c.benchmark_group("transcendental_log");
    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let a = make_array(*size);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| black_box(&a).elementwise_unary(UnaryOp::Log))
        });
    }
    group.finish();
}

fn bench_sin(c: &mut Criterion) {
    let mut group = c.benchmark_group("transcendental_sin");
    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let a = make_trig_array(*size);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| black_box(&a).elementwise_unary(UnaryOp::Sin))
        });
    }
    group.finish();
}

fn bench_cos(c: &mut Criterion) {
    let mut group = c.benchmark_group("transcendental_cos");
    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let a = make_trig_array(*size);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| black_box(&a).elementwise_unary(UnaryOp::Cos))
        });
    }
    group.finish();
}

fn bench_sqrt(c: &mut Criterion) {
    let mut group = c.benchmark_group("transcendental_sqrt");
    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let a = make_array(*size);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| black_box(&a).elementwise_unary(UnaryOp::Sqrt))
        });
    }
    group.finish();
}

fn bench_arcsin(c: &mut Criterion) {
    let mut group = c.benchmark_group("transcendental_arcsin");
    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let a = make_domain_array(*size);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| black_box(&a).elementwise_unary(UnaryOp::Arcsin))
        });
    }
    group.finish();
}

fn bench_expm1(c: &mut Criterion) {
    let mut group = c.benchmark_group("transcendental_expm1");
    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let a = make_array(*size);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| black_box(&a).elementwise_unary(UnaryOp::Expm1))
        });
    }
    group.finish();
}

fn bench_log1p(c: &mut Criterion) {
    let mut group = c.benchmark_group("transcendental_log1p");
    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        let a = make_array(*size);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| black_box(&a).elementwise_unary(UnaryOp::Log1p))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_exp,
    bench_log,
    bench_sin,
    bench_cos,
    bench_sqrt,
    bench_arcsin,
    bench_expm1,
    bench_log1p,
);
criterion_main!(benches);
