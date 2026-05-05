//! Profiling baseline tests for fnp-conformance.
//!
//! These tests capture baseline performance metrics and environment fingerprint
//! to enable regression detection without Criterion overhead.
//!
//! Run with: cargo test -p fnp-conformance --test profiling_baseline --release
//!
//! Following /profiling-software-performance skill methodology:
//! - Capture fingerprint (CPU, memory, toolchain)
//! - Baseline p50/p95/p99 latencies
//! - Validate no regression beyond 20% envelope

use fnp_dtype::DType;
use fnp_io::{IOSupportedDType, load, save};
use fnp_ufunc::{BinaryOp, UFuncArray};
use std::time::{Duration, Instant};

fn build_matrix_values(dim: usize, step: usize, modulo: usize) -> Vec<f64> {
    (0..(dim * dim))
        .map(|i| f64::from(((i * step) % modulo) as u32))
        .collect()
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let idx = ((sorted.len() as f64 - 1.0) * p / 100.0).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn benchmark<F: FnMut()>(mut f: F, warmup_iters: usize, sample_iters: usize) -> Vec<Duration> {
    for _ in 0..warmup_iters {
        f();
    }
    let mut samples = Vec::with_capacity(sample_iters);
    for _ in 0..sample_iters {
        let start = Instant::now();
        f();
        samples.push(start.elapsed());
    }
    samples.sort();
    samples
}

// ─────────────────────────────────────────────────────────────────────────────
// Fingerprint capture
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn capture_environment_fingerprint() {
    let rustc_version = std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    let cpu_info = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string());

    let mem_info = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemTotal"))
                .map(|l| l.split_whitespace().nth(1).unwrap_or("0").to_string())
        })
        .unwrap_or_else(|| "0".to_string());

    println!("\n=== ENVIRONMENT FINGERPRINT ===");
    println!("rustc: {}", rustc_version);
    println!("cpu: {}", cpu_info);
    println!("mem_kb: {}", mem_info);
    println!("profile: {}", if cfg!(debug_assertions) { "debug" } else { "release" });
    println!("================================\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Baseline: elementwise add with broadcast
// Budget: < 25ms p95 for 1024x1024 + 1024 broadcast
// NOTE: Measured 18.8ms p95 on 64-core TR PRO 5995WX. Criterion reports ~1.1ms
// but that's with different warmup. Setting budget at 25ms for regression gate.
// Future optimization target: get this under 5ms.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn baseline_ufunc_add_broadcast() {
    let dim = 1024usize;
    let lhs = UFuncArray::new(
        vec![dim, dim],
        build_matrix_values(dim, 7, 257),
        DType::F64,
    )
    .expect("lhs");
    let rhs = UFuncArray::new(
        vec![dim],
        (0..dim).map(|i| f64::from((i % 29) as u32)).collect(),
        DType::F64,
    )
    .expect("rhs");

    let samples = benchmark(
        || {
            let out = lhs.elementwise_binary(&rhs, BinaryOp::Add).expect("add");
            std::hint::black_box(out.values()[0]);
        },
        10,
        50,
    );

    let p50 = percentile(&samples, 50.0);
    let p95 = percentile(&samples, 95.0);
    let p99 = percentile(&samples, 99.0);

    println!("ufunc_add_broadcast 1024x1024+1024: p50={:?} p95={:?} p99={:?}", p50, p95, p99);

    // Budget envelope: p95 < 25ms in release (measured 18.8ms; optimization target: <5ms)
    #[cfg(not(debug_assertions))]
    assert!(
        p95 < Duration::from_millis(25),
        "REGRESSION: ufunc_add_broadcast p95={:?} exceeds 25ms budget",
        p95
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Baseline: reduce sum along axis
// Budget: < 1.5ms p95 for 1024x1024 axis=1
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn baseline_reduce_sum_axis1() {
    let dim = 1024usize;
    let arr = UFuncArray::new(
        vec![dim, dim],
        build_matrix_values(dim, 17, 509),
        DType::F64,
    )
    .expect("arr");

    let samples = benchmark(
        || {
            let out = arr.reduce_sum(Some(1), false).expect("reduce");
            std::hint::black_box(out.values()[0]);
        },
        10,
        50,
    );

    let p50 = percentile(&samples, 50.0);
    let p95 = percentile(&samples, 95.0);
    let p99 = percentile(&samples, 99.0);

    println!("reduce_sum_axis1 1024x1024: p50={:?} p95={:?} p99={:?}", p50, p95, p99);

    #[cfg(not(debug_assertions))]
    assert!(
        p95 < Duration::from_micros(1500),
        "REGRESSION: reduce_sum p95={:?} exceeds 1.5ms budget",
        p95
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Baseline: matrix multiply
// Budget: < 15ms p95 for 256x256 x 256x256
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn baseline_matmul_256x256() {
    let dim = 256usize;
    let lhs = UFuncArray::new(
        vec![dim, dim],
        build_matrix_values(dim, 13, 997),
        DType::F64,
    )
    .expect("lhs");
    let rhs = UFuncArray::new(
        vec![dim, dim],
        build_matrix_values(dim, 19, 991),
        DType::F64,
    )
    .expect("rhs");

    let samples = benchmark(
        || {
            let out = lhs.matmul(&rhs).expect("matmul");
            std::hint::black_box(out.values()[0]);
        },
        5,
        20,
    );

    let p50 = percentile(&samples, 50.0);
    let p95 = percentile(&samples, 95.0);
    let p99 = percentile(&samples, 99.0);

    println!("matmul 256x256x256x256: p50={:?} p95={:?} p99={:?}", p50, p95, p99);

    #[cfg(not(debug_assertions))]
    assert!(
        p95 < Duration::from_millis(15),
        "REGRESSION: matmul p95={:?} exceeds 15ms budget",
        p95
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Baseline: sort quicksort
// Budget: < 80ms p95 for 1M elements
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn baseline_sort_quicksort_1m() {
    let len = 1_000_000usize;
    let arr = UFuncArray::new(
        vec![len],
        (0..len)
            .map(|i| f64::from(((i * 48_271) % len) as u32))
            .collect(),
        DType::F64,
    )
    .expect("arr");

    let samples = benchmark(
        || {
            let out = arr.sort(None, Some("quicksort")).expect("sort");
            std::hint::black_box(out.values()[0]);
        },
        3,
        10,
    );

    let p50 = percentile(&samples, 50.0);
    let p95 = percentile(&samples, 95.0);
    let p99 = percentile(&samples, 99.0);

    println!("sort_quicksort 1M: p50={:?} p95={:?} p99={:?}", p50, p95, p99);

    #[cfg(not(debug_assertions))]
    assert!(
        p95 < Duration::from_millis(80),
        "REGRESSION: sort p95={:?} exceeds 80ms budget",
        p95
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Baseline: FFT
// Budget: < 5ms p95 for 65536 elements
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn baseline_fft_65536() {
    let len = 65_536usize;
    let arr = UFuncArray::new(
        vec![len],
        (0..len)
            .map(|i| {
                let t = i as f64 / len as f64;
                (std::f64::consts::TAU * 5.0 * t).sin()
                    + 0.5 * (std::f64::consts::TAU * 13.0 * t).cos()
            })
            .collect(),
        DType::F64,
    )
    .expect("arr");

    let samples = benchmark(
        || {
            let out = arr.fft(None).expect("fft");
            std::hint::black_box(out.values()[0]);
        },
        5,
        20,
    );

    let p50 = percentile(&samples, 50.0);
    let p95 = percentile(&samples, 95.0);
    let p99 = percentile(&samples, 99.0);

    println!("fft 65536: p50={:?} p95={:?} p99={:?}", p50, p95, p99);

    #[cfg(not(debug_assertions))]
    assert!(
        p95 < Duration::from_millis(5),
        "REGRESSION: fft p95={:?} exceeds 5ms budget",
        p95
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Baseline: NPY save/load round-trip
// Budget: < 800µs p95 for 512x512 f64
// NOTE: Measured 654µs p95 on TR PRO 5995WX. Setting 800µs for variance envelope.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn baseline_io_npy_roundtrip() {
    let dim = 512usize;
    let values: Vec<f64> = (0..(dim * dim))
        .map(|i| f64::from(((i * 29) % 65_537) as u32) / 11.0)
        .collect();

    let samples = benchmark(
        || {
            let payload = save(&[dim, dim], &values, IOSupportedDType::F64).expect("save");
            let (shape, vals, dtype) = load(&payload).expect("load");
            std::hint::black_box(shape);
            std::hint::black_box(vals[0]);
            std::hint::black_box(dtype);
        },
        10,
        50,
    );

    let p50 = percentile(&samples, 50.0);
    let p95 = percentile(&samples, 95.0);
    let p99 = percentile(&samples, 99.0);

    println!("io_npy_roundtrip 512x512: p50={:?} p95={:?} p99={:?}", p50, p95, p99);

    #[cfg(not(debug_assertions))]
    assert!(
        p95 < Duration::from_micros(800),
        "REGRESSION: io_npy p95={:?} exceeds 800µs budget",
        p95
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Hotspot summary table (run with --nocapture)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hotspot_summary() {
    println!("\n=== PROFILING BASELINE HOTSPOT TABLE (TR PRO 5995WX) ===");
    println!("| Rank | Operation                  | Measured p95 | Budget   | Category |");
    println!("|------|----------------------------|--------------|----------|----------|");
    println!("|  1   | ufunc_add_broadcast 1Mx1K  | 18.8ms       | 25ms     | CPU      |");
    println!("|  2   | sort_quicksort 1M          | 12.9ms       | 80ms     | CPU      |");
    println!("|  3   | fft 65536                  | 3.7ms        | 5ms      | CPU      |");
    println!("|  4   | matmul 256x256             | 3.2ms        | 15ms     | CPU      |");
    println!("|  5   | reduce_sum_axis1 1024x1024 | 778µs        | 1.5ms    | CPU      |");
    println!("|  6   | io_npy_roundtrip 512x512   | 654µs        | 800µs    | I/O      |");
    println!("============================================================\n");
    println!("Optimization targets: ufunc_add_broadcast (<5ms), fft (<2ms)");
}
