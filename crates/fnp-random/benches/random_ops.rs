//! Performance benchmarks for fnp-random core operations.
//!
//! Establishes baselines for:
//! - Bit generator raw throughput (next_u64, next_f64)
//! - Distribution sampling (normal, uniform, exponential, integers)
//! - Array generation at various sizes

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fnp_random::{
    BitGenerator, BitGeneratorKind, Generator, Pcg64DxsmRng, Pcg64Rng, PhiloxRng, SeedSequence,
    Sfc64Rng,
};
use std::hint::black_box;

fn seed_sequence() -> SeedSequence {
    SeedSequence::new(&[42]).unwrap()
}

// ─────────────────────────────────────────────────────────────────────────────
// Bit generator raw throughput
// ─────────────────────────────────────────────────────────────────────────────

fn bench_pcg64_next_u64(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcg64_raw");
    group.throughput(Throughput::Elements(1));

    let mut rng = Pcg64Rng::from_seed_sequence(&seed_sequence()).unwrap();

    group.bench_function("next_u64", |b| b.iter(|| black_box(rng.next_u64())));

    group.bench_function("next_f64", |b| b.iter(|| black_box(rng.next_f64())));

    group.finish();
}

fn bench_pcg64dxsm_next_u64(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcg64dxsm_raw");
    group.throughput(Throughput::Elements(1));

    let mut rng = Pcg64DxsmRng::from_seed_sequence(&seed_sequence()).unwrap();

    group.bench_function("next_u64", |b| b.iter(|| black_box(rng.next_u64())));

    group.bench_function("next_f64", |b| b.iter(|| black_box(rng.next_f64())));

    group.finish();
}

fn bench_philox_next_u64(c: &mut Criterion) {
    let mut group = c.benchmark_group("philox_raw");
    group.throughput(Throughput::Elements(1));

    let mut rng = PhiloxRng::from_seed_sequence(&seed_sequence()).unwrap();

    group.bench_function("next_u64", |b| b.iter(|| black_box(rng.next_u64())));

    group.bench_function("next_f64", |b| b.iter(|| black_box(rng.next_f64())));

    group.finish();
}

fn bench_sfc64_next_u64(c: &mut Criterion) {
    let mut group = c.benchmark_group("sfc64_raw");
    group.throughput(Throughput::Elements(1));

    let mut rng = Sfc64Rng::from_seed_sequence(&seed_sequence()).unwrap();

    group.bench_function("next_u64", |b| b.iter(|| black_box(rng.next_u64())));

    group.bench_function("next_f64", |b| b.iter(|| black_box(rng.next_f64())));

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Generator distribution sampling
// ─────────────────────────────────────────────────────────────────────────────

fn bench_generator_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("generator_random");

    let sizes = [100, 1000, 10000];

    for &size in &sizes {
        let ss = seed_sequence();
        let bg = BitGenerator::from_seed_sequence(BitGeneratorKind::Pcg64, &ss).unwrap();
        let mut generator = Generator::from_bit_generator(bg);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("f64", size), &size, |b, &size| {
            b.iter(|| black_box(generator.random(size)))
        });
    }

    group.finish();
}

fn bench_generator_standard_normal(c: &mut Criterion) {
    let mut group = c.benchmark_group("generator_standard_normal");

    let sizes = [100, 1000, 10000];

    for &size in &sizes {
        let ss = seed_sequence();
        let bg = BitGenerator::from_seed_sequence(BitGeneratorKind::Pcg64, &ss).unwrap();
        let mut generator = Generator::from_bit_generator(bg);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("ziggurat", size), &size, |b, &size| {
            b.iter(|| black_box(generator.standard_normal(size)))
        });
    }

    group.finish();
}

fn bench_generator_normal(c: &mut Criterion) {
    let mut group = c.benchmark_group("generator_normal");

    let sizes = [100, 1000, 10000];

    for &size in &sizes {
        let ss = seed_sequence();
        let bg = BitGenerator::from_seed_sequence(BitGeneratorKind::Pcg64, &ss).unwrap();
        let mut generator = Generator::from_bit_generator(bg);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("loc_scale", size), &size, |b, &size| {
            b.iter(|| black_box(generator.normal(0.0, 1.0, size).unwrap()))
        });
    }

    group.finish();
}

fn bench_generator_uniform(c: &mut Criterion) {
    let mut group = c.benchmark_group("generator_uniform");

    let sizes = [100, 1000, 10000];

    for &size in &sizes {
        let ss = seed_sequence();
        let bg = BitGenerator::from_seed_sequence(BitGeneratorKind::Pcg64, &ss).unwrap();
        let mut generator = Generator::from_bit_generator(bg);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("f64", size), &size, |b, &size| {
            b.iter(|| black_box(generator.uniform(0.0, 1.0, size).unwrap()))
        });
    }

    group.finish();
}

fn bench_generator_integers(c: &mut Criterion) {
    let mut group = c.benchmark_group("generator_integers");

    let sizes = [100, 1000, 10000];

    for &size in &sizes {
        let ss = seed_sequence();
        let bg = BitGenerator::from_seed_sequence(BitGeneratorKind::Pcg64, &ss).unwrap();
        let mut generator = Generator::from_bit_generator(bg);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("i64", size), &size, |b, &size| {
            b.iter(|| black_box(generator.integers(0, 100, size).unwrap()))
        });
    }

    group.finish();
}

fn bench_generator_exponential(c: &mut Criterion) {
    let mut group = c.benchmark_group("generator_exponential");

    let sizes = [100, 1000, 10000];

    for &size in &sizes {
        let ss = seed_sequence();
        let bg = BitGenerator::from_seed_sequence(BitGeneratorKind::Pcg64, &ss).unwrap();
        let mut generator = Generator::from_bit_generator(bg);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("scale_1", size), &size, |b, &size| {
            b.iter(|| black_box(generator.exponential(1.0, size).unwrap()))
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Bit generator comparison (same operation across all generators)
// ─────────────────────────────────────────────────────────────────────────────

fn bench_bitgen_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitgen_fill_1000");
    group.throughput(Throughput::Elements(1000));

    let ss = seed_sequence();

    group.bench_function("pcg64", |b| {
        let mut rng = Pcg64Rng::from_seed_sequence(&ss).unwrap();
        b.iter(|| black_box(rng.fill_u64(1000)))
    });

    group.bench_function("pcg64dxsm", |b| {
        let mut rng = Pcg64DxsmRng::from_seed_sequence(&ss).unwrap();
        b.iter(|| black_box(rng.fill_u64(1000)))
    });

    group.bench_function("philox", |b| {
        let mut rng = PhiloxRng::from_seed_sequence(&ss).unwrap();
        b.iter(|| {
            let mut sum = 0u64;
            for _ in 0..1000 {
                sum = sum.wrapping_add(rng.next_u64());
            }
            black_box(sum)
        })
    });

    group.bench_function("sfc64", |b| {
        let mut rng = Sfc64Rng::from_seed_sequence(&ss).unwrap();
        b.iter(|| {
            let mut sum = 0u64;
            for _ in 0..1000 {
                sum = sum.wrapping_add(rng.next_u64());
            }
            black_box(sum)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_pcg64_next_u64,
    bench_pcg64dxsm_next_u64,
    bench_philox_next_u64,
    bench_sfc64_next_u64,
    bench_generator_random,
    bench_generator_standard_normal,
    bench_generator_normal,
    bench_generator_uniform,
    bench_generator_integers,
    bench_generator_exponential,
    bench_bitgen_comparison,
);

criterion_main!(benches);
