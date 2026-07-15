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

#[allow(clippy::excessive_precision)]
fn former_random_loggam(x: f64) -> f64 {
    const A: [f64; 10] = [
        8.333333333333333e-02,
        -2.777777777777778e-03,
        7.936507936507937e-04,
        -5.952380952380952e-04,
        8.417508417508418e-04,
        -1.917526917526918e-03,
        6.410256410256410e-03,
        -2.955065359477124e-02,
        1.796443723688307e-01,
        -1.39243221690590e+00,
    ];

    if x == 1.0 || x == 2.0 {
        return 0.0;
    }

    let n = if x < 7.0 { (7.0 - x) as i64 } else { 0 };
    let mut x0 = x + n as f64;
    let x2 = (1.0 / x0) * (1.0 / x0);
    let mut gl0 = A[9];
    for k in (0..=8).rev() {
        gl0 = gl0 * x2 + A[k];
    }
    let mut gl = gl0 / x0 + 0.5 * 1.8378770664093453 + (x0 - 0.5) * x0.ln() - x0;
    if x < 7.0 {
        for _ in 0..n {
            gl -= (x0 - 1.0).ln();
            x0 -= 1.0;
        }
    }
    gl
}

#[inline(never)]
fn former_poisson_ptrs_one(rng: &mut Pcg64Rng, lam: f64) -> u64 {
    let slam = lam.sqrt();
    let loglam = lam.ln();
    let b = 0.931 + 2.53 * slam;
    let a = -0.059 + 0.02483 * b;
    let invalpha = 1.1239 + 1.1328 / (b - 3.4);
    let vr = 0.9277 - 3.6224 / (b - 2.0);
    loop {
        let u = rng.next_f64() - 0.5;
        let v = rng.next_f64();
        let us = 0.5 - u.abs();
        let k = ((2.0 * a / us + b) * u + lam + 0.43).floor() as i64;
        if us >= 0.07 && v <= vr {
            return k as u64;
        }
        if k < 0 || (us < 0.013 && v > us) {
            continue;
        }
        if v.ln() + invalpha.ln() - (a / (us * us) + b).ln()
            <= -lam + (k as f64) * loglam - former_random_loggam((k + 1) as f64)
        {
            return k as u64;
        }
    }
}

fn former_poisson_ptrs(rng: &mut Pcg64Rng, lam: f64, size: usize) -> Vec<u64> {
    (0..size)
        .map(|_| former_poisson_ptrs_one(rng, lam))
        .collect()
}

fn bench_poisson_ptrs_cache(c: &mut Criterion) {
    const SIZE: usize = 100_000;
    const LAM: f64 = 20.0;

    let mut former_proof = Pcg64Rng::from_seed_sequence(&seed_sequence()).unwrap();
    let mut candidate_proof = pcg64_generator();
    let former = former_poisson_ptrs(&mut former_proof, LAM, SIZE);
    let candidate = candidate_proof.poisson(LAM, SIZE).unwrap();
    assert_eq!(former, candidate);
    assert_eq!(former_proof.next_u64(), candidate_proof.next_u64());

    let mut former_rng = Pcg64Rng::from_seed_sequence(&seed_sequence()).unwrap();
    let mut candidate_generator = pcg64_generator();
    let mut group = c.benchmark_group("poisson_ptrs_parameter_cache");
    group.throughput(Throughput::Elements(SIZE as u64));
    group.bench_function("former_recompute_per_sample", |bench| {
        bench.iter(|| {
            black_box(former_poisson_ptrs(
                black_box(&mut former_rng),
                black_box(LAM),
                black_box(SIZE),
            ))
        })
    });
    group.bench_function("candidate_cache_per_batch", |bench| {
        bench.iter(|| {
            black_box(
                candidate_generator
                    .poisson(black_box(LAM), black_box(SIZE))
                    .unwrap(),
            )
        })
    });
    group.finish();
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

#[derive(Clone)]
struct BufferedPcg64 {
    rng: Pcg64Rng,
    buffered: u32,
    ready: bool,
}

impl BufferedPcg64 {
    fn from_seed_sequence(seed: &SeedSequence) -> Self {
        Self {
            rng: Pcg64Rng::from_seed_sequence(seed).unwrap(),
            buffered: 0,
            ready: false,
        }
    }

    #[expect(clippy::cast_possible_truncation)]
    fn next_u32(&mut self) -> u32 {
        if self.ready {
            self.ready = false;
            self.buffered
        } else {
            let value = self.rng.next_u64();
            self.buffered = (value >> 32) as u32;
            self.ready = true;
            value as u32
        }
    }

    fn random_interval(&mut self, max: usize) -> usize {
        let mask = (max + 1).next_power_of_two() - 1;
        loop {
            let value = self.next_u32() as usize & mask;
            if value <= max {
                return value;
            }
        }
    }

    fn next_f64(&mut self) -> f64 {
        self.ready = false;
        self.rng.next_f64()
    }
}

#[inline(never)]
fn former_permuted_last_axis(
    rng: &mut BufferedPcg64,
    input: &[f64],
    rows: usize,
    axis_len: usize,
) -> Vec<f64> {
    let mut result = input.to_vec();
    let shape = [rows, axis_len];
    let strides = [axis_len, 1];
    for slice_index in 0..rows {
        let mut multi_index = vec![0usize; 2];
        let mut remainder = slice_index;
        for dimension in (0..2).rev() {
            if dimension == 1 {
                continue;
            }
            multi_index[dimension] = remainder % shape[dimension];
            remainder /= shape[dimension];
        }
        let base_offset = multi_index[0] * strides[0];
        let indices: Vec<usize> = (0..axis_len)
            .map(|index| base_offset + index * strides[1])
            .collect();
        for index in (1..axis_len).rev() {
            let swap_index = rng.random_interval(index);
            result.swap(indices[index], indices[swap_index]);
        }
    }
    result
}

#[inline(never)]
fn direct_permuted_last_axis(rng: &mut BufferedPcg64, input: &[f64], axis_len: usize) -> Vec<f64> {
    let mut result = input.to_vec();
    for lane in result.chunks_exact_mut(axis_len) {
        for index in (1..axis_len).rev() {
            let swap_index = rng.random_interval(index);
            lane.swap(index, swap_index);
        }
    }
    result
}

fn bench_permuted_last_axis_allocations(c: &mut Criterion) {
    const ROWS: usize = 32_768;
    const AXIS_LEN: usize = 8;
    let input: Vec<f64> = (0..ROWS * AXIS_LEN).map(|index| index as f64).collect();
    let seed = seed_sequence();

    let mut former_proof = BufferedPcg64::from_seed_sequence(&seed);
    let mut direct_proof = BufferedPcg64::from_seed_sequence(&seed);
    let mut public_proof = pcg64_generator();
    let former = former_permuted_last_axis(&mut former_proof, &input, ROWS, AXIS_LEN);
    let direct = direct_permuted_last_axis(&mut direct_proof, &input, AXIS_LEN);
    let public = public_proof
        .permuted(&input, &[ROWS, AXIS_LEN], Some(1))
        .unwrap();
    assert_eq!(direct, former);
    assert_eq!(public, former);
    let former_after = former_proof.next_f64().to_bits();
    let direct_after = direct_proof.next_f64().to_bits();
    let public_after = public_proof.random(1)[0].to_bits();
    assert_eq!(direct_after, former_after);
    assert_eq!(public_after, former_after);

    let mut former_rng = BufferedPcg64::from_seed_sequence(&seed);
    let mut direct_rng = BufferedPcg64::from_seed_sequence(&seed);
    let mut group = c.benchmark_group("permuted_last_axis_allocations");
    group.throughput(Throughput::Elements((ROWS * AXIS_LEN) as u64));
    group.bench_function("former_per_row_vectors", |bench| {
        bench.iter(|| {
            black_box(former_permuted_last_axis(
                black_box(&mut former_rng),
                black_box(&input),
                ROWS,
                AXIS_LEN,
            ))
        })
    });
    group.bench_function("direct_contiguous_lanes", |bench| {
        bench.iter(|| {
            black_box(direct_permuted_last_axis(
                black_box(&mut direct_rng),
                black_box(&input),
                AXIS_LEN,
            ))
        })
    });
    group.finish();
}

#[inline(never)]
fn former_permuted_strided_axis(
    rng: &mut BufferedPcg64,
    input: &[f64],
    shape: [usize; 3],
    axis: usize,
) -> Vec<f64> {
    let mut result = input.to_vec();
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for index in (0..ndim - 1).rev() {
        strides[index] = strides[index + 1] * shape[index + 1];
    }
    let axis_len = shape[axis];
    let axis_stride = strides[axis];
    let n_slices = result.len() / axis_len;
    for slice_index in 0..n_slices {
        let mut multi_index = vec![0usize; ndim];
        let mut remainder = slice_index;
        for dimension in (0..ndim).rev() {
            if dimension == axis {
                continue;
            }
            multi_index[dimension] = remainder % shape[dimension];
            remainder /= shape[dimension];
        }
        let mut base_offset = 0;
        for dimension in 0..ndim {
            if dimension != axis {
                base_offset += multi_index[dimension] * strides[dimension];
            }
        }
        let indices: Vec<usize> = (0..axis_len)
            .map(|index| base_offset + index * axis_stride)
            .collect();
        for index in (1..axis_len).rev() {
            let swap_index = rng.random_interval(index);
            result.swap(indices[index], indices[swap_index]);
        }
    }
    result
}

#[inline(never)]
fn direct_permuted_strided_axis(
    rng: &mut BufferedPcg64,
    input: &[f64],
    shape: [usize; 3],
    axis: usize,
) -> Vec<f64> {
    let mut result = input.to_vec();
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for index in (0..ndim - 1).rev() {
        strides[index] = strides[index + 1] * shape[index + 1];
    }
    let axis_len = shape[axis];
    let axis_stride = strides[axis];
    let n_slices = result.len() / axis_len;
    for slice_index in 0..n_slices {
        let mut remainder = slice_index;
        let mut base_offset = 0;
        for dimension in (0..ndim).rev() {
            if dimension == axis {
                continue;
            }
            let coordinate = remainder % shape[dimension];
            remainder /= shape[dimension];
            base_offset += coordinate * strides[dimension];
        }
        for index in (1..axis_len).rev() {
            let swap_index = rng.random_interval(index);
            result.swap(
                base_offset + index * axis_stride,
                base_offset + swap_index * axis_stride,
            );
        }
    }
    result
}

fn bench_permuted_strided_axis_allocations(c: &mut Criterion) {
    const SHAPE: [usize; 3] = [256, 8, 128];
    const AXIS: usize = 1;
    const LEN: usize = SHAPE[0] * SHAPE[1] * SHAPE[2];
    let input: Vec<f64> = (0..LEN).map(|index| index as f64).collect();
    let seed = seed_sequence();

    let mut former_proof = BufferedPcg64::from_seed_sequence(&seed);
    let mut direct_proof = BufferedPcg64::from_seed_sequence(&seed);
    let mut public_proof = pcg64_generator();
    let former = former_permuted_strided_axis(&mut former_proof, &input, SHAPE, AXIS);
    let direct = direct_permuted_strided_axis(&mut direct_proof, &input, SHAPE, AXIS);
    let public = public_proof.permuted(&input, &SHAPE, Some(AXIS)).unwrap();
    assert_eq!(direct, former);
    assert_eq!(public, former);
    let former_after = former_proof.next_f64().to_bits();
    let direct_after = direct_proof.next_f64().to_bits();
    let public_after = public_proof.random(1)[0].to_bits();
    assert_eq!(direct_after, former_after);
    assert_eq!(public_after, former_after);

    let mut former_rng = BufferedPcg64::from_seed_sequence(&seed);
    let mut direct_rng = BufferedPcg64::from_seed_sequence(&seed);
    let mut group = c.benchmark_group("permuted_strided_axis_allocations");
    group.throughput(Throughput::Elements(LEN as u64));
    group.bench_function("former_per_slice_vectors", |bench| {
        bench.iter(|| {
            black_box(former_permuted_strided_axis(
                black_box(&mut former_rng),
                black_box(&input),
                SHAPE,
                AXIS,
            ))
        })
    });
    group.bench_function("direct_strided_addresses", |bench| {
        bench.iter(|| {
            black_box(direct_permuted_strided_axis(
                black_box(&mut direct_rng),
                black_box(&input),
                SHAPE,
                AXIS,
            ))
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
    bench_poisson_ptrs_cache,
    bench_choice_weighted_singleton,
    bench_permuted_last_axis_allocations,
    bench_permuted_strided_axis_allocations,
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
