//! Performance benchmarks for fnp-random core operations.
//!
//! Establishes baselines for:
//! - Bit generator raw throughput (next_u64, next_f64)
//! - Distribution sampling (normal, uniform, exponential, integers)
//! - Array generation at various sizes

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fnp_random::{
    BitGenerator, BitGeneratorKind, Generator, Pcg64DxsmRng, Pcg64Rng, PhiloxRng, RandomError,
    RandomState, SeedMaterial, SeedSequence, Sfc64Rng,
};
use std::cell::{Cell, RefCell};
use std::hint::black_box;
use std::time::{Duration, Instant};

fn seed_sequence() -> SeedSequence {
    SeedSequence::new(&[42]).unwrap()
}

fn pcg64_generator() -> Generator {
    let bit_generator =
        BitGenerator::from_seed_sequence(BitGeneratorKind::Pcg64, &seed_sequence()).unwrap();
    Generator::from_bit_generator(bit_generator)
}

fn ledger_tail_stats(samples: &RefCell<Vec<f64>>) -> (usize, f64, f64) {
    let samples = samples.borrow();
    let count = samples.len().min(10);
    assert!(count >= 2, "Criterion must retain paired samples");
    let tail = &samples[samples.len() - count..];
    let mean = tail.iter().sum::<f64>() / count as f64;
    let variance = tail
        .iter()
        .map(|sample| {
            let delta = sample - mean;
            delta * delta
        })
        .sum::<f64>()
        / (count - 1) as f64;
    (count, mean, variance.sqrt() * 100.0 / mean)
}

fn report_ledger_pair(
    row: &str,
    candidate_samples: &RefCell<Vec<f64>>,
    former_samples: &RefCell<Vec<f64>>,
) {
    if candidate_samples.borrow().len() < 2 || former_samples.borrow().len() < 2 {
        return;
    }
    let (candidate_n, candidate_ns, candidate_cv) = ledger_tail_stats(candidate_samples);
    let (former_n, former_ns, former_cv) = ledger_tail_stats(former_samples);
    assert_eq!(candidate_n, former_n);
    println!(
        "LEDGER_AUDIT row={row} samples={candidate_n} candidate_mean_ms={:.6} \
         candidate_cv_pct={candidate_cv:.3} orig_mean_ms={:.6} orig_cv_pct={former_cv:.3} \
         orig_over_candidate={:.4}",
        candidate_ns / 1_000_000.0,
        former_ns / 1_000_000.0,
        former_ns / candidate_ns,
    );
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

/// The public batched gamma path PLUS exactly the parameter-only work the
/// per-batch cache removes: one Marsaglia-Tsang `d`/`c` recomputation per
/// sample. The removed terms consume no RNG draws, so this additive model is
/// operation-exact (same machinery in both arms — see the `.333` bench-arm
/// rule: bare-loop former arms understate batch levers).
fn former_model_standard_gamma(generator: &mut Generator, shape: f64, size: usize) -> Vec<f64> {
    let out = generator.standard_gamma(shape, size).unwrap();
    for _ in 0..size {
        let s = black_box(shape);
        let d = s - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        black_box((d, c));
    }
    out
}

fn bench_gamma_shape_cache(c: &mut Criterion) {
    const SIZE: usize = 100_000;
    const SHAPE: f64 = 5.0;

    // Batch-vs-singleton stream equivalence: the cache must not change what a
    // batched draw produces relative to repeated single draws, and both must
    // leave the raw stream in the same place.
    let mut batch_proof = pcg64_generator();
    let mut single_proof = pcg64_generator();
    for &shape in &[0.5f64, 1.0, 5.0, 20.0] {
        let batch = batch_proof.standard_gamma(shape, 64).unwrap();
        let singles: Vec<f64> = (0..64)
            .map(|_| single_proof.standard_gamma(shape, 1).unwrap()[0])
            .collect();
        assert_eq!(
            batch.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            singles.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "gamma batch/singleton divergence at shape {shape}"
        );
    }
    assert_eq!(batch_proof.next_u64(), single_proof.next_u64());

    let mut former_generator = pcg64_generator();
    let mut candidate_generator = pcg64_generator();
    let mut group = c.benchmark_group("gamma_shape_parameter_cache");
    group.throughput(Throughput::Elements(SIZE as u64));
    group.bench_function("former_model_recompute_terms", |bench| {
        bench.iter(|| {
            black_box(former_model_standard_gamma(
                black_box(&mut former_generator),
                black_box(SHAPE),
                black_box(SIZE),
            ))
        })
    });
    group.bench_function("candidate_batch_cache", |bench| {
        bench.iter(|| {
            black_box(
                candidate_generator
                    .standard_gamma(black_box(SHAPE), black_box(SIZE))
                    .unwrap(),
            )
        })
    });
    group.finish();
}

/// Replica of one per-sample `GammaShapeCache::new` (the exact branch chain
/// and expressions), used to model the work the fixed-shape hoist removes.
fn gamma_cache_build_model(shape: f64) -> (f64, f64) {
    if shape == 1.0 || shape == 0.0 || !shape.is_finite() {
        (0.0, 0.0)
    } else if shape < 1.0 {
        (1.0 - shape, 1.0 / shape)
    } else {
        let d = shape - 1.0 / 3.0;
        (d, 1.0 / (9.0 * d).sqrt())
    }
}

/// The public batched beta path PLUS exactly the parameter-only work the
/// fixed-shape hoist removes: TWO gamma cache builds per sample (one per
/// shape). Same machinery in both arms (`.333` bench-arm rule); the removed
/// terms consume no RNG draws, so the additive model is operation-exact.
fn former_model_beta(generator: &mut Generator, a: f64, b: f64, size: usize) -> Vec<f64> {
    let out = generator.beta(a, b, size).unwrap();
    for _ in 0..size {
        black_box(gamma_cache_build_model(black_box(a)));
        black_box(gamma_cache_build_model(black_box(b)));
    }
    out
}

fn bench_beta_gamma_shape_cache(c: &mut Criterion) {
    const SIZE: usize = 100_000;
    const A: f64 = 2.5;
    const B: f64 = 4.0;

    // Batch-vs-singleton stream equivalence for every hoisted loop.
    let mut batch_proof = pcg64_generator();
    let mut single_proof = pcg64_generator();
    for &(a, b) in &[(2.5f64, 4.0f64), (0.5, 3.0), (1.5, 0.75)] {
        let batch = batch_proof.beta(a, b, 32).unwrap();
        let singles: Vec<f64> = (0..32)
            .map(|_| single_proof.beta(a, b, 1).unwrap()[0])
            .collect();
        assert_eq!(
            batch.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            singles.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "beta batch/singleton divergence at ({a}, {b})"
        );
    }
    assert_eq!(batch_proof.next_u64(), single_proof.next_u64());

    let mut former_generator = pcg64_generator();
    let mut candidate_generator = pcg64_generator();
    let mut group = c.benchmark_group("beta_gamma_shape_cache");
    group.throughput(Throughput::Elements(SIZE as u64));
    group.bench_function("former_model_recompute_caches", |bench| {
        bench.iter(|| {
            black_box(former_model_beta(
                black_box(&mut former_generator),
                black_box(A),
                black_box(B),
                black_box(SIZE),
            ))
        })
    });
    group.bench_function("candidate_hoisted_caches", |bench| {
        bench.iter(|| {
            black_box(
                candidate_generator
                    .beta(black_box(A), black_box(B), black_box(SIZE))
                    .unwrap(),
            )
        })
    });
    group.finish();
}

/// The public batched dirichlet path PLUS exactly the parameter-only work the
/// per-alpha cache vector removes: one gamma cache build per component per
/// draw. Additive model per the `.333` rule; per the `.336` method note it
/// UNDERSTATES dependency-chained removed work (a faithful replica is not
/// possible here - the gamma body's ziggurat cores are private), so the
/// measured ratio is a conservative floor.
fn former_model_dirichlet(generator: &mut Generator, alpha: &[f64], size: usize) -> Vec<Vec<f64>> {
    let out = generator.dirichlet(alpha, size).unwrap();
    for _ in 0..size {
        for &a in alpha {
            black_box(gamma_cache_build_model(black_box(a)));
        }
    }
    out
}

fn bench_dirichlet_gamma_shape_cache(c: &mut Criterion) {
    const SIZE: usize = 20_000;
    let alpha = [2.5f64, 4.0, 1.5, 0.5, 3.0];

    // Batch-vs-singleton stream equivalence: the per-alpha cache vector must
    // not change what a batched draw produces relative to repeated single
    // draws, including zero and sub-1 components.
    let mut batch_proof = pcg64_generator();
    let mut single_proof = pcg64_generator();
    for alphas in [&alpha[..], &[0.5f64, 0.0, 2.0][..]] {
        let batch = batch_proof.dirichlet(alphas, 16).unwrap();
        let singles: Vec<Vec<f64>> = (0..16)
            .map(|_| {
                single_proof
                    .dirichlet(alphas, 1)
                    .unwrap()
                    .into_iter()
                    .next()
                    .unwrap()
            })
            .collect();
        for (draw_index, (b, s)) in batch.iter().zip(singles.iter()).enumerate() {
            assert_eq!(
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                s.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "dirichlet batch/singleton divergence at draw {draw_index} for {alphas:?}"
            );
        }
    }
    assert_eq!(batch_proof.next_u64(), single_proof.next_u64());

    let mut former_generator = pcg64_generator();
    let mut candidate_generator = pcg64_generator();
    let mut group = c.benchmark_group("dirichlet_gamma_shape_cache");
    group.throughput(Throughput::Elements((SIZE * alpha.len()) as u64));
    group.bench_function("former_model_recompute_caches", |bench| {
        bench.iter(|| {
            black_box(former_model_dirichlet(
                black_box(&mut former_generator),
                black_box(&alpha),
                black_box(SIZE),
            ))
        })
    });
    group.bench_function("candidate_cache_vector", |bench| {
        bench.iter(|| {
            black_box(
                candidate_generator
                    .dirichlet(black_box(&alpha), black_box(SIZE))
                    .unwrap(),
            )
        })
    });
    group.finish();
}

/// Faithful replica of the FORMER Best-Fisher vonmises batch loop over the
/// public `Generator::next_f64` (the same method the production loop draws
/// from): the kappa-only `s` is recomputed per sample in its true dependency
/// position, unlike an additive model loop whose independent iterations
/// pipeline the recomputation away. Valid for 1e-5 <= kappa <= 1e6 (the
/// rejection regime, which consumes only `next_f64` draws).
#[inline(never)]
fn former_vonmises_best_fisher(
    generator: &mut Generator,
    mu: f64,
    kappa: f64,
    size: usize,
) -> Vec<f64> {
    fn wrap_angle_to_pi(angle: f64) -> f64 {
        (angle + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
    }
    (0..size)
        .map(|_| {
            let s = if kappa < 1e-5 {
                1.0 / kappa + kappa
            } else {
                let r = 1.0 + (1.0 + 4.0 * kappa * kappa).sqrt();
                let rho = (r - (2.0 * r).sqrt()) / (2.0 * kappa);
                (1.0 + rho * rho) / (2.0 * rho)
            };
            loop {
                let u1 = generator.next_f64();
                let z = (std::f64::consts::PI * u1).cos();
                let w = (1.0 + s * z) / (s + z);
                let y = kappa * (s - w);
                let u2 = generator.next_f64();
                if y * (2.0 - y) - u2 >= 0.0 || (y / u2).ln() + 1.0 - y >= 0.0 {
                    let u3 = generator.next_f64();
                    let theta = if u3 < 0.5 { -w.acos() } else { w.acos() };
                    return wrap_angle_to_pi(mu + theta);
                }
            }
        })
        .collect()
}

fn bench_vonmises_kappa_cache(c: &mut Criterion) {
    const SIZE: usize = 100_000;
    const MU: f64 = 1.25;
    const KAPPA: f64 = 2.5;

    // Batch-vs-singleton stream equivalence across the three touched kappa
    // regimes (small-s series, Best-Fisher rejection, large-kappa normal
    // approximation).
    let mut batch_proof = pcg64_generator();
    let mut single_proof = pcg64_generator();
    for &kappa in &[5e-6f64, 2.5, 1e7] {
        let batch = batch_proof.vonmises(MU, kappa, 32).unwrap();
        let singles: Vec<f64> = (0..32)
            .map(|_| single_proof.vonmises(MU, kappa, 1).unwrap()[0])
            .collect();
        assert_eq!(
            batch.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            singles.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "vonmises batch/singleton divergence at kappa {kappa}"
        );
    }
    assert_eq!(batch_proof.next_u64(), single_proof.next_u64());

    // The faithful former replica must reproduce the public path bit-for-bit
    // (it recomputes s per sample; the hoist must not change any draw).
    let mut replica_proof = pcg64_generator();
    let mut public_proof = pcg64_generator();
    let replica = former_vonmises_best_fisher(&mut replica_proof, MU, KAPPA, 4096);
    let public = public_proof.vonmises(MU, KAPPA, 4096).unwrap();
    assert_eq!(
        replica.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        public.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(replica_proof.next_u64(), public_proof.next_u64());

    let mut former_generator = pcg64_generator();
    let mut candidate_generator = pcg64_generator();
    let mut group = c.benchmark_group("vonmises_kappa_cache");
    group.throughput(Throughput::Elements(SIZE as u64));
    group.bench_function("former_full_replica", |bench| {
        bench.iter(|| {
            black_box(former_vonmises_best_fisher(
                black_box(&mut former_generator),
                black_box(MU),
                black_box(KAPPA),
                black_box(SIZE),
            ))
        })
    });
    group.bench_function("candidate_hoisted_terms", |bench| {
        bench.iter(|| {
            black_box(
                candidate_generator
                    .vonmises(black_box(MU), black_box(KAPPA), black_box(SIZE))
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
        let mut multi_index = [0usize; 2];
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

fn bench_noncentral_f_fixed_shape_cache(c: &mut Criterion) {
    const SIZE: usize = 100_000;
    const DFNUM: f64 = 5.0;
    const DFDEN: f64 = 20.0;
    const NONC: f64 = 2.0;

    let mut generator = pcg64_generator();
    let mut group = c.benchmark_group("noncentral_f_fixed_shape_cache");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Elements(SIZE as u64));
    group.bench_function("hoisted_shape_terms", |bench| {
        bench.iter(|| {
            black_box(
                generator
                    .noncentral_f(
                        black_box(DFNUM),
                        black_box(DFDEN),
                        black_box(NONC),
                        black_box(SIZE),
                    )
                    .unwrap(),
            )
        })
    });
    group.finish();
}

fn bench_hypergeometric_hrua_cache(c: &mut Criterion) {
    const SIZE: usize = 100_000;
    const NGOOD: u64 = 20_000;
    const NBAD: u64 = 30_000;
    const NSAMPLE: u64 = 10_000;

    let mut generator = pcg64_generator();
    let mut group = c.benchmark_group("hypergeometric_hrua_parameter_cache");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Elements(SIZE as u64));
    group.bench_function("cached_plan_per_batch", |bench| {
        bench.iter(|| {
            black_box(
                generator
                    .hypergeometric(
                        black_box(NGOOD),
                        black_box(NBAD),
                        black_box(NSAMPLE),
                        black_box(SIZE),
                    )
                    .unwrap(),
            )
        })
    });
    group.finish();
}

#[inline(never)]
fn former_zipf_single(generator: &mut Generator, a: f64) -> i64 {
    if a >= 1025.0 {
        return 1;
    }
    let am1 = a - 1.0;
    let b = 2.0_f64.powf(am1);
    let umin = (i64::MAX as f64).powf(-am1);

    loop {
        let u01 = generator.next_f64();
        let u = u01 * umin + (1.0 - u01);
        let v = generator.next_f64();
        let x = u.powf(-1.0 / am1).floor();
        if x > i64::MAX as f64 || x < 1.0 {
            continue;
        }
        let t = (1.0 + 1.0 / x).powf(am1);
        if v * x * (t - 1.0) / (b - 1.0) <= t / b {
            return x as i64;
        }
    }
}

fn former_zipf(generator: &mut Generator, a: f64, size: usize) -> Result<Vec<f64>, RandomError> {
    if a.is_nan() || a <= 1.0 {
        return Err(RandomError::InvalidParameter);
    }
    Ok((0..size)
        .map(|_| former_zipf_single(generator, a) as f64)
        .collect())
}

#[inline(never)]
fn rejected_cached_zipf_single(
    generator: &mut Generator,
    am1: f64,
    b: f64,
    umin: f64,
    exponent: f64,
) -> i64 {
    loop {
        let u01 = generator.next_f64();
        let u = u01 * umin + (1.0 - u01);
        let v = generator.next_f64();
        let x = u.powf(exponent).floor();
        if x > i64::MAX as f64 || x < 1.0 {
            continue;
        }
        let t = (1.0 + 1.0 / x).powf(am1);
        if v * x * (t - 1.0) / (b - 1.0) <= t / b {
            return x as i64;
        }
    }
}

fn rejected_cached_zipf(
    generator: &mut Generator,
    a: f64,
    size: usize,
) -> Result<Vec<f64>, RandomError> {
    if a.is_nan() || a <= 1.0 {
        return Err(RandomError::InvalidParameter);
    }
    if size == 0 {
        return Ok(Vec::new());
    }
    if a >= 1025.0 {
        return Ok(vec![1.0; size]);
    }
    let am1 = a - 1.0;
    let b = 2.0_f64.powf(am1);
    let umin = (i64::MAX as f64).powf(-am1);
    let exponent = -1.0 / am1;
    Ok((0..size)
        .map(|_| rejected_cached_zipf_single(generator, am1, b, umin, exponent) as f64)
        .collect())
}

fn bench_zipf_parameter_cache(c: &mut Criterion) {
    const SIZE: usize = 100_000;
    const A: f64 = 2.5;

    let mut former_proof = pcg64_generator();
    let mut candidate_proof = pcg64_generator();
    let former = former_zipf(&mut former_proof, A, 4096).unwrap();
    let candidate = rejected_cached_zipf(&mut candidate_proof, A, 4096).unwrap();
    assert_eq!(
        former
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        candidate
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(former_proof.next_u64(), candidate_proof.next_u64());

    let mut group = c.benchmark_group("zipf_parameter_cache");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Elements(SIZE as u64));

    let mut former_generator = pcg64_generator();
    let mut candidate_generator = pcg64_generator();
    let candidate_samples = RefCell::new(Vec::new());
    let former_samples = RefCell::new(Vec::new());
    let effect_order = Cell::new(0u64);
    group.bench_function("paired_former_candidate", |bench| {
        bench.iter_custom(|iterations| {
            let mut candidate_total = Duration::ZERO;
            let mut former_total = Duration::ZERO;
            let time_candidate = |generator: &mut Generator| {
                let started = Instant::now();
                black_box(rejected_cached_zipf(generator, black_box(A), black_box(SIZE)).unwrap());
                started.elapsed()
            };
            let time_former = |generator: &mut Generator| {
                let started = Instant::now();
                black_box(former_zipf(generator, black_box(A), black_box(SIZE)).unwrap());
                started.elapsed()
            };
            for _ in 0..iterations {
                if effect_order.get() & 1 == 0 {
                    former_total += time_former(&mut former_generator);
                    candidate_total += time_candidate(&mut candidate_generator);
                    candidate_total += time_candidate(&mut candidate_generator);
                    former_total += time_former(&mut former_generator);
                } else {
                    candidate_total += time_candidate(&mut candidate_generator);
                    former_total += time_former(&mut former_generator);
                    former_total += time_former(&mut former_generator);
                    candidate_total += time_candidate(&mut candidate_generator);
                }
                effect_order.set(effect_order.get().wrapping_add(1));
            }
            candidate_samples
                .borrow_mut()
                .push(candidate_total.as_secs_f64() * 0.5e9 / iterations as f64);
            former_samples
                .borrow_mut()
                .push(former_total.as_secs_f64() * 0.5e9 / iterations as f64);
            candidate_total + former_total
        })
    });
    report_ledger_pair(
        "zipf_parameter_cache_effect",
        &candidate_samples,
        &former_samples,
    );

    let mut null_lhs_generator = pcg64_generator();
    let mut null_rhs_generator = pcg64_generator();
    let null_lhs_samples = RefCell::new(Vec::new());
    let null_rhs_samples = RefCell::new(Vec::new());
    let null_order = Cell::new(0u64);
    group.bench_function("null_candidate_aa", |bench| {
        bench.iter_custom(|iterations| {
            let mut lhs_total = Duration::ZERO;
            let mut rhs_total = Duration::ZERO;
            let time_candidate = |generator: &mut Generator| {
                let started = Instant::now();
                black_box(rejected_cached_zipf(generator, black_box(A), black_box(SIZE)).unwrap());
                started.elapsed()
            };
            for _ in 0..iterations {
                if null_order.get() & 1 == 0 {
                    lhs_total += time_candidate(&mut null_lhs_generator);
                    rhs_total += time_candidate(&mut null_rhs_generator);
                    rhs_total += time_candidate(&mut null_rhs_generator);
                    lhs_total += time_candidate(&mut null_lhs_generator);
                } else {
                    rhs_total += time_candidate(&mut null_rhs_generator);
                    lhs_total += time_candidate(&mut null_lhs_generator);
                    lhs_total += time_candidate(&mut null_lhs_generator);
                    rhs_total += time_candidate(&mut null_rhs_generator);
                }
                null_order.set(null_order.get().wrapping_add(1));
            }
            null_lhs_samples
                .borrow_mut()
                .push(lhs_total.as_secs_f64() * 0.5e9 / iterations as f64);
            null_rhs_samples
                .borrow_mut()
                .push(rhs_total.as_secs_f64() * 0.5e9 / iterations as f64);
            lhs_total + rhs_total
        })
    });
    report_ledger_pair(
        "zipf_parameter_cache_null",
        &null_lhs_samples,
        &null_rhs_samples,
    );

    group.finish();
}

#[inline(never)]
fn former_random_state_zipf_single(random_state: &mut RandomState, a: f64) -> i64 {
    if a >= 1025.0 {
        return 1;
    }
    let am1 = a - 1.0;
    let b = 2.0_f64.powf(am1);
    let umin = (i64::MAX as f64).powf(-am1);

    loop {
        let u01 = random_state.next_f64();
        let u = u01 * umin + (1.0 - u01);
        let v = random_state.next_f64();
        let x = u.powf(-1.0 / am1).floor();
        if x > i64::MAX as f64 || x < 1.0 {
            continue;
        }
        let t = (1.0 + 1.0 / x).powf(am1);
        if v * x * (t - 1.0) / (b - 1.0) <= t / b {
            return x as i64;
        }
    }
}

fn former_random_state_zipf(
    random_state: &mut RandomState,
    a: f64,
    size: usize,
) -> Result<Vec<u64>, RandomError> {
    if a.is_nan() || a <= 1.0 {
        return Err(RandomError::InvalidParameter);
    }
    Ok((0..size)
        .map(|_| former_random_state_zipf_single(random_state, a) as u64)
        .collect())
}

#[inline(never)]
fn rejected_cached_random_state_zipf_single(
    random_state: &mut RandomState,
    am1: f64,
    b: f64,
    umin: f64,
    exponent: f64,
) -> i64 {
    loop {
        let u01 = random_state.next_f64();
        let u = u01 * umin + (1.0 - u01);
        let v = random_state.next_f64();
        let x = u.powf(exponent).floor();
        if x > i64::MAX as f64 || x < 1.0 {
            continue;
        }
        let t = (1.0 + 1.0 / x).powf(am1);
        if v * x * (t - 1.0) / (b - 1.0) <= t / b {
            return x as i64;
        }
    }
}

fn rejected_cached_random_state_zipf(
    random_state: &mut RandomState,
    a: f64,
    size: usize,
) -> Result<Vec<u64>, RandomError> {
    if a.is_nan() || a <= 1.0 {
        return Err(RandomError::InvalidParameter);
    }
    if size == 0 {
        return Ok(Vec::new());
    }
    if a >= 1025.0 {
        return Ok(vec![1; size]);
    }
    let am1 = a - 1.0;
    let b = 2.0_f64.powf(am1);
    let umin = (i64::MAX as f64).powf(-am1);
    let exponent = -1.0 / am1;
    Ok((0..size)
        .map(|_| {
            rejected_cached_random_state_zipf_single(random_state, am1, b, umin, exponent) as u64
        })
        .collect())
}

fn bench_random_state_zipf_parameter_cache(c: &mut Criterion) {
    const SIZE: usize = 100_000;
    const REPEATS: usize = 6;
    const A: f64 = 2.5;

    let mut former_proof = RandomState::new(SeedMaterial::U64(42)).unwrap();
    let mut candidate_proof = RandomState::new(SeedMaterial::U64(42)).unwrap();
    assert_eq!(
        former_random_state_zipf(&mut former_proof, A, 4096).unwrap(),
        rejected_cached_random_state_zipf(&mut candidate_proof, A, 4096).unwrap()
    );
    assert_eq!(former_proof.next_u64(), candidate_proof.next_u64());

    let mut group = c.benchmark_group("random_state_zipf_parameter_cache");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Elements((SIZE * REPEATS) as u64));

    let mut random_state = RandomState::new(SeedMaterial::U64(42)).unwrap();
    group.bench_function("baseline", |bench| {
        bench.iter(|| black_box(random_state.zipf(black_box(A), black_box(SIZE)).unwrap()))
    });

    let candidate_samples = RefCell::new(Vec::new());
    let former_samples = RefCell::new(Vec::new());
    let effect_order = Cell::new(0u64);
    group.bench_function("fixed_trace_paired_former_candidate", |bench| {
        bench.iter_custom(|iterations| {
            let mut candidate_total = Duration::ZERO;
            let mut former_total = Duration::ZERO;
            let time_candidate = || {
                let mut random_state = RandomState::new(SeedMaterial::U64(42)).unwrap();
                let started = Instant::now();
                for _ in 0..REPEATS {
                    black_box(
                        rejected_cached_random_state_zipf(
                            &mut random_state,
                            black_box(A),
                            black_box(SIZE),
                        )
                        .unwrap(),
                    );
                }
                started.elapsed()
            };
            let time_former = || {
                let mut random_state = RandomState::new(SeedMaterial::U64(42)).unwrap();
                let started = Instant::now();
                for _ in 0..REPEATS {
                    black_box(
                        former_random_state_zipf(&mut random_state, black_box(A), black_box(SIZE))
                            .unwrap(),
                    );
                }
                started.elapsed()
            };
            for _ in 0..iterations {
                if effect_order.get() & 1 == 0 {
                    former_total += time_former();
                    candidate_total += time_candidate();
                    candidate_total += time_candidate();
                    former_total += time_former();
                } else {
                    candidate_total += time_candidate();
                    former_total += time_former();
                    former_total += time_former();
                    candidate_total += time_candidate();
                }
                effect_order.set(effect_order.get().wrapping_add(1));
            }
            candidate_samples
                .borrow_mut()
                .push(candidate_total.as_secs_f64() * 0.5e9 / iterations as f64);
            former_samples
                .borrow_mut()
                .push(former_total.as_secs_f64() * 0.5e9 / iterations as f64);
            candidate_total + former_total
        })
    });
    report_ledger_pair(
        "random_state_zipf_parameter_cache_effect",
        &candidate_samples,
        &former_samples,
    );

    let null_lhs_samples = RefCell::new(Vec::new());
    let null_rhs_samples = RefCell::new(Vec::new());
    let null_order = Cell::new(0u64);
    group.bench_function("fixed_trace_null_candidate_aa", |bench| {
        bench.iter_custom(|iterations| {
            let mut lhs_total = Duration::ZERO;
            let mut rhs_total = Duration::ZERO;
            let time_candidate = || {
                let mut random_state = RandomState::new(SeedMaterial::U64(42)).unwrap();
                let started = Instant::now();
                for _ in 0..REPEATS {
                    black_box(
                        rejected_cached_random_state_zipf(
                            &mut random_state,
                            black_box(A),
                            black_box(SIZE),
                        )
                        .unwrap(),
                    );
                }
                started.elapsed()
            };
            for _ in 0..iterations {
                if null_order.get() & 1 == 0 {
                    lhs_total += time_candidate();
                    rhs_total += time_candidate();
                    rhs_total += time_candidate();
                    lhs_total += time_candidate();
                } else {
                    rhs_total += time_candidate();
                    lhs_total += time_candidate();
                    lhs_total += time_candidate();
                    rhs_total += time_candidate();
                }
                null_order.set(null_order.get().wrapping_add(1));
            }
            null_lhs_samples
                .borrow_mut()
                .push(lhs_total.as_secs_f64() * 0.5e9 / iterations as f64);
            null_rhs_samples
                .borrow_mut()
                .push(rhs_total.as_secs_f64() * 0.5e9 / iterations as f64);
            lhs_total + rhs_total
        })
    });
    report_ledger_pair(
        "random_state_zipf_parameter_cache_null",
        &null_lhs_samples,
        &null_rhs_samples,
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_poisson_ptrs_cache,
    bench_gamma_shape_cache,
    bench_beta_gamma_shape_cache,
    bench_dirichlet_gamma_shape_cache,
    bench_vonmises_kappa_cache,
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
    bench_noncentral_f_fixed_shape_cache,
    bench_hypergeometric_hrua_cache,
    bench_zipf_parameter_cache,
    bench_random_state_zipf_parameter_cache,
);

criterion_main!(benches);
