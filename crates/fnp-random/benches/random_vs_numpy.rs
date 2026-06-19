//! Head-to-head PCG throughput benchmarks against original NumPy.
//!
//! The NumPy side keeps one Python process per Criterion sample and times the
//! inner loop with `time.perf_counter()`, avoiding per-iteration spawn cost.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use fnp_random::{BitGenerator, BitGeneratorKind, Generator, Pcg64Rng, SeedSequence};
use std::hint::black_box;
use std::process::Command;
use std::time::{Duration, Instant};

const SEED: u32 = 42;
const RAW_U64_SIZES: [usize; 2] = [100_000, 1_000_000];
const BYTE_SIZES: [usize; 2] = [100_000, 1_000_000];

const NUMPY_RANDOM_RAW_SCRIPT: &str = r#"
import sys, time
import numpy as np

iters = int(sys.argv[1])
size = int(sys.argv[2])
rng = np.random.Generator(np.random.PCG64(42))
checksum = 0

start = time.perf_counter()
for _ in range(iters):
    values = rng.bit_generator.random_raw(size)
    checksum ^= int(values[0])
    checksum ^= int(values[-1])
elapsed = time.perf_counter() - start

print(f"{elapsed:.12f} {checksum}")
"#;

const NUMPY_BYTES_SCRIPT: &str = r#"
import sys, time
import numpy as np

iters = int(sys.argv[1])
size = int(sys.argv[2])
rng = np.random.Generator(np.random.PCG64(42))
checksum = 0

start = time.perf_counter()
for _ in range(iters):
    payload = rng.bytes(size)
    checksum ^= payload[0]
    checksum ^= payload[-1]
elapsed = time.perf_counter() - start

print(f"{elapsed:.12f} {checksum}")
"#;

fn seed_sequence() -> SeedSequence {
    SeedSequence::new(&[SEED]).expect("fixed seed should build")
}

fn generator_from_pcg64() -> Generator {
    let bit_generator = BitGenerator::from_seed_sequence(BitGeneratorKind::Pcg64, &seed_sequence())
        .expect("PCG64 generator should build");
    Generator::from_bit_generator(bit_generator)
}

fn run_numpy_timed(script: &str, iterations: u64, size: usize) -> Duration {
    let python = std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let output = Command::new(&python)
        .arg("-c")
        .arg(script)
        .arg(iterations.to_string())
        .arg(size.to_string())
        .output()
        .expect("failed to run NumPy benchmark command");

    assert!(
        output.status.success(),
        "NumPy benchmark command failed: status={:?}\nstdout={}\nstderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("NumPy stdout should be UTF-8");
    let seconds = stdout
        .split_whitespace()
        .next()
        .expect("NumPy timing output should include elapsed seconds")
        .parse::<f64>()
        .expect("NumPy elapsed seconds should parse");
    Duration::from_secs_f64(seconds)
}

fn bench_pcg64_random_raw_vs_numpy(c: &mut Criterion) {
    let mut group = c.benchmark_group("vs_numpy_pcg64_random_raw");

    for &size in &RAW_U64_SIZES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("franken_fill_u64", size),
            &size,
            |b, &size| {
                b.iter_custom(|iterations| {
                    let mut rng = Pcg64Rng::from_seed_sequence(&seed_sequence())
                        .expect("PCG64 generator should build");
                    let start = Instant::now();
                    for _ in 0..iterations {
                        black_box(rng.fill_u64(size));
                    }
                    start.elapsed()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("numpy_random_raw", size),
            &size,
            |b, &size| {
                b.iter_custom(|iterations| {
                    run_numpy_timed(NUMPY_RANDOM_RAW_SCRIPT, iterations, size)
                });
            },
        );
    }

    group.finish();
}

fn bench_pcg64_bytes_vs_numpy(c: &mut Criterion) {
    let mut group = c.benchmark_group("vs_numpy_pcg64_bytes");

    for &size in &BYTE_SIZES {
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("franken_bytes", size),
            &size,
            |b, &size| {
                b.iter_custom(|iterations| {
                    let mut generator = generator_from_pcg64();
                    let start = Instant::now();
                    for _ in 0..iterations {
                        black_box(generator.bytes(size));
                    }
                    start.elapsed()
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("numpy_bytes", size), &size, |b, &size| {
            b.iter_custom(|iterations| run_numpy_timed(NUMPY_BYTES_SCRIPT, iterations, size));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_pcg64_random_raw_vs_numpy,
    bench_pcg64_bytes_vs_numpy,
);
criterion_main!(benches);
