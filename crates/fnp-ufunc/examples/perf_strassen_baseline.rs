//! Baseline timing for the production GEMM path (`UFuncArray::matmul` ->
//! `matmul_accumulate`) at large square sizes, to scope a Strassen-Winograd
//! recursive-GEMM lever (bead franken_numpy-3q3pe follow-on). Prints an FNV-1a
//! checksum (pins the input/baseline) + median wall time per size.
//!
//! Run: `cargo run --release --example perf_strassen_baseline -p fnp-ufunc`

use std::time::Instant;

use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;

fn build(dim: usize, seed: u64) -> UFuncArray {
    // Deterministic pseudo-random-ish finite f64 in [-1, 1).
    let mut s = seed;
    let values: Vec<f64> = (0..dim * dim)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect();
    UFuncArray::new(vec![dim, dim], values, DType::F64).unwrap()
}

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

fn main() {
    println!("threads = {}", rayon::current_num_threads());
    for &n in &[512usize, 768, 1024, 1536, 2048] {
        let a = build(n, 0x1234_5678);
        let b = build(n, 0x9abc_def0);
        // warm
        let c0 = a.matmul(&b).unwrap();
        let checksum = fnv1a(c0.values());
        let iters = if n <= 768 { 5 } else { 3 };
        let mut times = Vec::new();
        for _ in 0..iters {
            let t = Instant::now();
            let c = a.matmul(&b).unwrap();
            std::hint::black_box(&c);
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let ms = median(times);
        let gflops = 2.0 * (n as f64).powi(3) / (ms * 1e-3) / 1e9;
        println!("n={n:5}  {ms:9.3} ms   {gflops:7.2} GFLOP/s   checksum=0x{checksum:016x}");
    }
}
