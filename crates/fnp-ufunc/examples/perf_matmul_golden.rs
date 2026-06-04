//! Profiling-only golden + timing harness for the matmul row-blocking
//! optimization (bead franken_numpy-evqs4). Reproduces the exact `core_ops`
//! matmul fixture (256×256 · 256×256, f64), prints an FNV-1a checksum of the
//! output bit-patterns (the bit-exactness proof — must be identical before and
//! after the optimization) and the median wall-clock time.
//!
//! Run: `cargo run --release --example perf_matmul_golden -p fnp-ufunc`

use std::time::Instant;

use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;

// Mirrors build_matrix_values() in crates/fnp-conformance/benches/criterion_core_ops.rs.
fn build_matrix_values(dim: usize, step: usize, modulo: usize) -> Vec<f64> {
    (0..(dim * dim))
        .map(|i| f64::from(((i * step) % modulo) as u32))
        .collect()
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

fn matmul_naive_reference(lhs: &[f64], rhs: &[f64], dim: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; dim * dim];
    for (lhs_row, out_row) in lhs.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        for (&a_val, rhs_row) in lhs_row.iter().zip(rhs.chunks_exact(dim)) {
            for (slot, &rhs_value) in out_row.iter_mut().zip(rhs_row.iter()) {
                *slot += a_val * rhs_value;
            }
        }
    }
    out
}

// Prior committed kernel: row-blocked (ROW_BLOCK=8) i-k-j. Kept here as a
// same-process, same-worker baseline to measure the register-tile lever against.
fn matmul_row_block_reference(lhs: &[f64], rhs: &[f64], dim: usize) -> Vec<f64> {
    const ROW_BLOCK: usize = 8;
    let (m, k, n) = (dim, dim, dim);
    let mut out = vec![0.0f64; m * n];
    let mut row_start = 0usize;
    while row_start < m {
        let row_end = (row_start + ROW_BLOCK).min(m);
        for (kk, rhs_row) in rhs.chunks_exact(n).enumerate() {
            for row in row_start..row_end {
                let a_val = lhs[row * k + kk];
                let out_row = &mut out[row * n..(row + 1) * n];
                for (slot, &rhs_value) in out_row.iter_mut().zip(rhs_row.iter()) {
                    *slot += a_val * rhs_value;
                }
            }
        }
        row_start = row_end;
    }
    out
}

fn time_median_ms<T, F>(iters: usize, mut run: F) -> f64
where
    F: FnMut() -> T,
{
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        std::hint::black_box(run());
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    median(samples)
}

fn main() {
    let dim = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(256usize);
    let lhs = UFuncArray::new(
        vec![dim, dim],
        build_matrix_values(dim, 13, 997),
        DType::F64,
    )
    .unwrap();
    let rhs = UFuncArray::new(
        vec![dim, dim],
        build_matrix_values(dim, 19, 991),
        DType::F64,
    )
    .unwrap();

    let out = lhs.matmul(&rhs).expect("matmul");
    let reference_out = matmul_naive_reference(lhs.values(), rhs.values(), dim);
    assert_eq!(out.values(), reference_out.as_slice());

    let checksum = fnv1a(out.values());
    let reference_checksum = fnv1a(&reference_out);

    let iters = if dim >= 1024 { 7 } else { 80 };
    for _ in 0..3 {
        std::hint::black_box(lhs.matmul(&rhs).unwrap());
    }

    // Bit-exactness vs row-block too (must be identical).
    let row_block_out = matmul_row_block_reference(lhs.values(), rhs.values(), dim);
    assert_eq!(
        out.values(),
        row_block_out.as_slice(),
        "register-tile != row-block"
    );

    let current_median = time_median_ms(iters, || lhs.matmul(&rhs).unwrap());
    let reference_median = time_median_ms(iters.max(3), || {
        matmul_naive_reference(lhs.values(), rhs.values(), dim)
    });
    let row_block_median = time_median_ms(iters.max(3), || {
        matmul_row_block_reference(lhs.values(), rhs.values(), dim)
    });

    let gflops = 2.0 * (dim as f64).powi(3) / (current_median * 1e-3) / 1e9;
    let speedup_vs_rowblock = row_block_median / current_median;
    println!(
        "matmul_{dim}x{dim} current_fnv1a=0x{checksum:016x} reference_fnv1a=0x{reference_checksum:016x}"
    );
    println!(
        "matmul_{dim}x{dim} naive_ms={reference_median:.4} rowblock_ms={row_block_median:.4} regtile_ms={current_median:.4} regtile_gflops={gflops:.2} speedup_vs_rowblock={speedup_vs_rowblock:.2}x"
    );
    println!(
        "out[0]={}  out[{}]={}",
        out.values()[0],
        dim * dim - 1,
        out.values()[dim * dim - 1]
    );
}
