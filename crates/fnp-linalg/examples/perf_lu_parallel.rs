//! Profiling-only golden + timing harness for the parallel LU trailing-update
//! optimization (bead franken_numpy-* — parallel `lu_decompose_inner`).
//!
//! Compares the live `lu_factor_nxn` (now parallel across trailing rows) against
//! an inlined *serial* reference that reproduces the exact pre-optimization
//! right-looking LU. Asserts the packed LU factors, the permutation, and the sign
//! are bit-for-bit identical (the isomorphism proof — partial pivoting, ordering,
//! and floating-point rounding must be unchanged), prints an FNV-1a checksum of
//! the LU bit-patterns, and reports the median wall-clock of each plus the
//! speedup and GFLOP/s.
//!
//! Run: `cargo run --release --example perf_lu_parallel -p fnp-linalg -- 1024`

use std::time::Instant;

use fnp_linalg::lu_factor_nxn;

// Mirrors generate_invertible_matrix() in benches/criterion_linalg.rs: strongly
// diagonally dominant, so LU with partial pivoting never trips the singularity
// guard and never permutes (stable, deterministic factorization).
fn generate_invertible_matrix(n: usize) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = if i == j {
                (n * 2) as f64
            } else {
                ((i + j) % 5) as f64 * 0.1
            };
        }
    }
    a
}

// Exact pre-optimization serial right-looking LU (partial pivoting). This is the
// behavioral reference; the parallel kernel must reproduce its bits exactly.
fn serial_lu(a: &[f64], n: usize) -> (Vec<f64>, Vec<usize>, f64) {
    let matrix_max_abs = a.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let singularity_threshold = if matrix_max_abs.is_finite() {
        (n as f64) * f64::EPSILON * matrix_max_abs
    } else {
        0.0
    };
    let mut lu = a.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();
    let mut sign = 1.0_f64;
    for k in 0..n {
        let mut max_val = lu[k * n + k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let val = lu[i * n + k].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }
        if max_val <= singularity_threshold {
            panic!("serial_lu: unexpected singularity at k={k}");
        }
        if max_row != k {
            for j in 0..n {
                lu.swap(k * n + j, max_row * n + j);
            }
            perm.swap(k, max_row);
            sign = -sign;
        }
        let pivot = lu[k * n + k];
        if pivot.is_nan() {
            continue;
        }
        for i in (k + 1)..n {
            let factor = lu[i * n + k] / pivot;
            lu[i * n + k] = factor;
            for j in (k + 1)..n {
                let u_val = lu[k * n + j];
                lu[i * n + j] -= factor * u_val;
            }
        }
    }
    (lu, perm, sign)
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
    let dim = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1024usize);

    let a = generate_invertible_matrix(dim);

    let (par_lu, par_perm, par_sign) = lu_factor_nxn(&a, dim).expect("parallel lu");
    let (ref_lu, ref_perm, ref_sign) = serial_lu(&a, dim);

    // Bit-exact isomorphism proof.
    assert_eq!(
        par_lu.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        ref_lu.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "LU factors diverge bit-wise"
    );
    assert_eq!(par_perm, ref_perm, "permutation diverges");
    assert_eq!(par_sign.to_bits(), ref_sign.to_bits(), "sign diverges");

    let checksum = fnv1a(&par_lu);
    let ref_checksum = fnv1a(&ref_lu);

    let iters = if dim >= 1024 { 7 } else { 31 };
    for _ in 0..3 {
        std::hint::black_box(lu_factor_nxn(&a, dim).unwrap());
    }
    let current_median = time_median_ms(iters, || lu_factor_nxn(&a, dim).unwrap());
    let reference_median = time_median_ms(iters, || serial_lu(&a, dim));

    let gflop = 2.0 / 3.0 * (dim as f64).powi(3) / 1e9;
    let speedup = reference_median / current_median;
    println!("lu_{dim}x{dim} current_fnv1a=0x{checksum:016x} reference_fnv1a=0x{ref_checksum:016x}");
    println!(
        "lu_{dim}x{dim} serial_median_ms={reference_median:.4} parallel_median_ms={current_median:.4} speedup={speedup:.2}x parallel_gflops={:.2} serial_gflops={:.2}",
        gflop / (current_median * 1e-3),
        gflop / (reference_median * 1e-3)
    );
}
