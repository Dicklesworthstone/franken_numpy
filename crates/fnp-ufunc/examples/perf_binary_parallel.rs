//! Profiling-only golden + timing harness for the parallel transcendental binary
//! ufunc lever (bead franken_numpy-* binary parallel).
//!
//! Compares the live `elementwise_binary` (now rayon-parallel for transcendental
//! binary ops over large arrays) against an inlined *serial* `op.apply` map.
//! Asserts the outputs are bit-for-bit identical (per-element libm results are
//! unchanged), prints an FNV-1a checksum, and reports the speedup.
//!
//! Run: `cargo run --release --example perf_binary_parallel -p fnp-ufunc -- 4000000`

use std::time::Instant;

use fnp_dtype::DType;
use fnp_ufunc::{BinaryOp, UFuncArray};

fn fnv1a(values: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for v in values {
        for byte in v.to_bits().to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn time_median_ms<T, F: FnMut() -> T>(iters: usize, mut run: F) -> f64 {
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        std::hint::black_box(run());
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    median(samples)
}

fn main() {
    let n = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(4_000_000usize);

    let a: Vec<f64> = (0..n).map(|i| ((i % 1000) as f64) * 0.001 + 0.5).collect();
    let b: Vec<f64> = (0..n).map(|i| ((i % 777) as f64) * 0.002 + 0.5).collect();

    for op in [BinaryOp::Power, BinaryOp::Hypot, BinaryOp::Arctan2] {
        let lhs = UFuncArray::new(vec![n], a.clone(), DType::F64).unwrap();
        let rhs = UFuncArray::new(vec![n], b.clone(), DType::F64).unwrap();
        let parallel = lhs.elementwise_binary(&rhs, op).expect("binary");
        let serial: Vec<f64> = a.iter().zip(&b).map(|(&x, &y)| op.apply(x, y)).collect();
        assert_eq!(
            parallel.values(),
            serial.as_slice(),
            "{op:?}: parallel != serial"
        );
        let checksum = fnv1a(parallel.values());

        for _ in 0..2 {
            std::hint::black_box(lhs.elementwise_binary(&rhs, op).unwrap());
        }
        let par_ms = time_median_ms(11, || lhs.elementwise_binary(&rhs, op).unwrap());
        let ser_ms = time_median_ms(11, || {
            a.iter()
                .zip(&b)
                .map(|(&x, &y)| op.apply(x, y))
                .collect::<Vec<f64>>()
        });
        let speedup = ser_ms / par_ms;
        let melem_s = (n as f64) / (par_ms * 1e-3) / 1e6;
        println!(
            "binary_{op:?}_n{n} fnv1a=0x{checksum:016x} serial_ms={ser_ms:.4} parallel_ms={par_ms:.4} speedup={speedup:.2}x throughput_Melem_s={melem_s:.1}"
        );
    }
}
