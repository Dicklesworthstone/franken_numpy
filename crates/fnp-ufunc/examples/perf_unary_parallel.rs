//! Profiling-only golden + timing harness for large unary ufunc maps.
//!
//! Compares the live `elementwise_unary` against an inlined *serial* `op.apply`
//! map. Asserts the outputs are bit-for-bit identical, prints a SHA-256 golden
//! digest, and reports the median wall-clock of each plus the speedup.
//!
//! Run: `cargo run --release --example perf_unary_parallel -p fnp-ufunc -- 4000000`

use std::time::Instant;

use fnp_dtype::DType;
use fnp_ufunc::{UFuncArray, UnaryOp};
use sha2::{Digest, Sha256};

fn sha256_hex(values: &[f64]) -> String {
    let mut digest = Sha256::new();
    for v in values {
        digest.update(v.to_bits().to_le_bytes());
    }
    digest
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
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

    // Spread of finite inputs so libm does real work (no fast NaN/inf paths).
    let data: Vec<f64> = (0..n).map(|i| ((i % 1000) as f64) * 0.001 + 0.5).collect();

    for op in [
        UnaryOp::Abs,
        UnaryOp::Square,
        UnaryOp::Floor,
        UnaryOp::Rint,
        UnaryOp::Reciprocal,
        UnaryOp::Exp,
        UnaryOp::Sin,
        UnaryOp::Log,
    ] {
        let arr = UFuncArray::new(vec![n], data.clone(), DType::F64).unwrap();
        let parallel = arr.elementwise_unary(op);
        let serial: Vec<f64> = data.iter().map(|&v| op.apply(v)).collect();
        assert_eq!(
            parallel.values(),
            serial.as_slice(),
            "{:?}: parallel != serial",
            op
        );
        let sha256 = sha256_hex(parallel.values());

        for _ in 0..2 {
            std::hint::black_box(arr.elementwise_unary(op));
        }
        let par_ms = time_median_ms(11, || arr.elementwise_unary(op));
        let ser_ms = time_median_ms(11, || {
            data.iter().map(|&v| op.apply(v)).collect::<Vec<f64>>()
        });
        let speedup = ser_ms / par_ms;
        let melem_s = (n as f64) / (par_ms * 1e-3) / 1e6;
        println!(
            "unary_{:?}_n{n} sha256={sha256} serial_ms={ser_ms:.4} parallel_ms={par_ms:.4} speedup={speedup:.2}x throughput_Melem_s={melem_s:.1}",
            op
        );
    }
}
