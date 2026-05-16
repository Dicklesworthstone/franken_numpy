use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use fnp_dtype::DType;
use fnp_ufunc::{UFuncArray, add, divide, multiply, subtract};

fn make_array(n: usize) -> UFuncArray {
    let values: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 + 1.0).collect();
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

criterion_group!(
    benches,
    bench_add,
    bench_subtract,
    bench_multiply,
    bench_divide,
    bench_chained_ops
);
criterion_main!(benches);
