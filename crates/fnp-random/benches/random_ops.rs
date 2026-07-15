//! Performance benchmarks for fnp-random core operations.
//!
//! Establishes baselines for:
//! - Bit generator raw throughput (next_u64, next_f64)
//! - Distribution sampling (normal, uniform, exponential, integers)
//! - Array generation at various sizes

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fnp_random::{
    BitGenerator, BitGeneratorKind, Generator, Pcg64DxsmRng, Pcg64Rng, PhiloxRng, RandomError,
    SeedSequence, Sfc64Rng,
};
use std::hint::black_box;

fn seed_sequence() -> SeedSequence {
    SeedSequence::new(&[42]).unwrap()
}

fn pcg64_generator() -> Generator {
    let bit_generator =
        BitGenerator::from_seed_sequence(BitGeneratorKind::Pcg64, &seed_sequence()).unwrap();
    Generator::from_bit_generator(bit_generator)
}

#[inline(never)]
fn former_choice_weighted_replace_one(
    generator: &mut Generator,
    values: &[f64],
    probabilities: &[f64],
) -> Result<Vec<f64>, RandomError> {
    let n = values.len();
    if probabilities.len() != n {
        return Err(RandomError::InvalidUpperBound);
    }
    let sum_tolerance = f64::EPSILON.sqrt();
    let sum: f64 = probabilities.iter().sum();
    if !sum.is_finite()
        || (sum - 1.0).abs() > sum_tolerance
        || probabilities
            .iter()
            .any(|&probability| !probability.is_finite() || probability < 0.0)
    {
        return Err(RandomError::InvalidUpperBound);
    }

    let mut cdf = Vec::with_capacity(n);
    let mut cumulative = 0.0;
    for &probability in probabilities {
        cumulative += probability;
        cdf.push(cumulative);
    }
    let draw = generator.next_f64();
    let index = cdf.partition_point(|&value| value <= draw).min(n - 1);
    Ok(vec![values[index]])
}

fn bench_choice_weighted_singleton(c: &mut Criterion) {
    const LEN: usize = 131_071;
    let mut values: Vec<f64> = (0..LEN).map(|index| index as f64 * 0.25 - 7.0).collect();
    values[0] = f64::from_bits(0x7ff8_0000_0000_0042);
    let mut probabilities = vec![0.0; LEN];
    probabilities[0] = 1.0;

    let mut former_proof = pcg64_generator();
    let mut candidate_proof = former_proof.clone();
    for _ in 0..64 {
        let former =
            former_choice_weighted_replace_one(&mut former_proof, &values, &probabilities).unwrap();
        let candidate = candidate_proof
            .choice_weighted(&values, 1, true, &probabilities)
            .unwrap();
        assert_eq!(former[0].to_bits(), candidate[0].to_bits());
    }
    assert_eq!(former_proof, candidate_proof);

    let mut former_generator = pcg64_generator();
    let mut candidate_generator = pcg64_generator();
    let mut group = c.benchmark_group("choice_weighted_singleton");
    group.throughput(Throughput::Elements(LEN as u64));
    group.bench_function("former_materialized_cdf", |bench| {
        bench.iter(|| {
            black_box(
                former_choice_weighted_replace_one(
                    black_box(&mut former_generator),
                    black_box(&values),
                    black_box(&probabilities),
                )
                .unwrap(),
            )
        })
    });
    group.bench_function("direct_cumulative_scan", |bench| {
        bench.iter(|| {
            black_box(
                candidate_generator
                    .choice_weighted(black_box(&values), 1, true, black_box(&probabilities))
                    .unwrap(),
            )
        })
    });
    group.finish();
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

    let sizes = [100, 1000, 10000, 100000];

    for &size in &sizes {
        let ss = seed_sequence();
        let bg = BitGenerator::from_seed_sequence(BitGeneratorKind::Pcg64, &ss).unwrap();
        let mut generator = Generator::from_bit_generator(bg);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("f64", size), &size, |b, &size| {
            b.iter(|| black_box(generator.random(size)))
        });

        let ss = seed_sequence();
        let bg = BitGenerator::from_seed_sequence(BitGeneratorKind::Pcg64, &ss).unwrap();
        let mut generator = Generator::from_bit_generator(bg);
        group.bench_with_input(BenchmarkId::new("f32", size), &size, |b, &size| {
            b.iter(|| black_box(generator.random_f32(size)))
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

    let sizes = [100, 1000, 10000, 100000];

    for &size in &sizes {
        let ss = seed_sequence();
        let bg = BitGenerator::from_seed_sequence(BitGeneratorKind::Pcg64, &ss).unwrap();
        let mut generator = Generator::from_bit_generator(bg);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("scale_1", size), &size, |b, &size| {
            b.iter(|| black_box(generator.exponential(1.0, size).unwrap()))
        });

        let ss = seed_sequence();
        let bg = BitGenerator::from_seed_sequence(BitGeneratorKind::Pcg64, &ss).unwrap();
        let mut generator = Generator::from_bit_generator(bg);
        group.bench_with_input(BenchmarkId::new("standard_inv", size), &size, |b, &size| {
            b.iter(|| black_box(generator.standard_exponential_inv(size)))
        });
    }

    group.finish();
}

fn bench_generator_logistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("generator_logistic");

    let sizes = [100, 1000, 10000, 100000];

    for &size in &sizes {
        let ss = seed_sequence();
        let bg = BitGenerator::from_seed_sequence(BitGeneratorKind::Pcg64, &ss).unwrap();
        let mut generator = Generator::from_bit_generator(bg);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("loc_scale", size), &size, |b, &size| {
            b.iter(|| black_box(generator.logistic(-2.25, 3.5, size).unwrap()))
        });
    }

    group.finish();
}

fn bench_generator_gumbel(c: &mut Criterion) {
    let mut group = c.benchmark_group("generator_gumbel");

    let sizes = [100, 1000, 10000, 100000];

    for &size in &sizes {
        let ss = seed_sequence();
        let bg = BitGenerator::from_seed_sequence(BitGeneratorKind::Pcg64, &ss).unwrap();
        let mut generator = Generator::from_bit_generator(bg);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("loc_scale", size), &size, |b, &size| {
            b.iter(|| black_box(generator.gumbel(-1.25, 2.75, size).unwrap()))
        });
    }

    group.finish();
}

fn bench_generator_laplace(c: &mut Criterion) {
    let mut group = c.benchmark_group("generator_laplace");

    let sizes = [100, 1000, 10000, 100000];

    for &size in &sizes {
        let ss = seed_sequence();
        let bg = BitGenerator::from_seed_sequence(BitGeneratorKind::Pcg64, &ss).unwrap();
        let mut generator = Generator::from_bit_generator(bg);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("loc_scale", size), &size, |b, &size| {
            b.iter(|| black_box(generator.laplace(1.5, 2.25, size).unwrap()))
        });
    }

    group.finish();
}

fn bench_generator_triangular(c: &mut Criterion) {
    let mut group = c.benchmark_group("generator_triangular");

    let sizes = [100, 1000, 10000, 100000];

    for &size in &sizes {
        let ss = seed_sequence();
        let bg = BitGenerator::from_seed_sequence(BitGeneratorKind::Pcg64, &ss).unwrap();
        let mut generator = Generator::from_bit_generator(bg);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("skewed", size), &size, |b, &size| {
            b.iter(|| black_box(generator.triangular(-3.0, 0.25, 5.5, size).unwrap()))
        });
    }

    group.finish();
}

fn bench_generator_vonmises(c: &mut Criterion) {
    let mut group = c.benchmark_group("generator_vonmises");

    let sizes = [100, 1000, 10000, 100000];

    for &size in &sizes {
        let ss = seed_sequence();
        let bg = BitGenerator::from_seed_sequence(BitGeneratorKind::Pcg64, &ss).unwrap();
        let mut generator = Generator::from_bit_generator(bg);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("near_uniform", size), &size, |b, &size| {
            b.iter(|| black_box(generator.vonmises(1.75, 0.0, size).unwrap()))
        });
    }

    group.finish();
}

fn bench_generator_bytes(c: &mut Criterion) {
    let mut group = c.benchmark_group("generator_bytes");
    let sizes = [1000usize, 100_000, 1_000_000];

    for &size in &sizes {
        let ss = seed_sequence();
        let bg = BitGenerator::from_seed_sequence(BitGeneratorKind::Pcg64, &ss).unwrap();
        let mut generator = Generator::from_bit_generator(bg);

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("pcg64", size), &size, |b, &size| {
            b.iter(|| black_box(generator.bytes(size)))
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

fn bench_pcg_fill_u64_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcg_fill_u64_large");
    let sizes = [1000usize, 100_000, 1_000_000];

    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));

        let ss = seed_sequence();
        let mut rng = Pcg64Rng::from_seed_sequence(&ss).unwrap();
        group.bench_with_input(BenchmarkId::new("pcg64", size), &size, |b, &size| {
            b.iter(|| black_box(rng.fill_u64(size)))
        });

        let ss = seed_sequence();
        let mut rng = Pcg64DxsmRng::from_seed_sequence(&ss).unwrap();
        group.bench_with_input(BenchmarkId::new("pcg64dxsm", size), &size, |b, &size| {
            b.iter(|| black_box(rng.fill_u64(size)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_choice_weighted_singleton,
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
    bench_generator_logistic,
    bench_generator_gumbel,
    bench_generator_laplace,
    bench_generator_triangular,
    bench_generator_vonmises,
    bench_generator_bytes,
    bench_bitgen_comparison,
    bench_pcg_fill_u64_large,
);

criterion_main!(benches);
