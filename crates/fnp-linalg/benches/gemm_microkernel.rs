//! Crate-local probe: does an explicit `std::simd` register tile beat the
//! autovectorized scalar tile in the packed f64 GEMM microkernel?
//!
//! `fnp-linalg`'s production `packed_gemm_serial` accumulates an
//! `[[f64; PACKED_NR]; PACKED_MR]` register tile with plain scalar loops and
//! relies on LLVM to vectorize the NR-wide inner row. This bench reproduces
//! that kernel exactly, adds a variant whose tile is `[Simd<f64, PACKED_NR>;
//! PACKED_MR]`, and measures them head to head on the same data.
//!
//! BIT-IDENTITY: the SIMD variant vectorizes across the NR output *columns*,
//! never across `k`. Lane `j` still accumulates `sum_k a[i,k]*b[k,j]` in
//! ascending `k`, exactly as the scalar tile does, and Rust does not contract
//! `a * b + c` into an FMA on its own. Both kernels must therefore produce
//! byte-identical output, which `assert_bit_identical` enforces at setup — the
//! production kernel is locked bit-identical by golden-sha256 tests, so a
//! variant that changed a single bit would be unshippable regardless of speed.
//!
//! Run one arm only (the whole point of the FNP_BENCH_GROUPS work):
//!   cargo bench -p fnp-linalg --profile bench-fast --bench gemm_microkernel

#![feature(portable_simd)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::simd::Simd;

const PACKED_MR: usize = 4;
const PACKED_NR: usize = 8;

/// Deterministic, reproducible operand fill (no RNG dependency, no clock).
fn fill(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            // SplitMix64 step, mapped into [-1, 1) so products stay well scaled.
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        })
        .collect()
}

/// Column-panel size matching the production kernel's ~256 KiB L2 target.
fn panel_cols(k: usize) -> usize {
    let cols = (256 * 1024) / (k.max(1) * core::mem::size_of::<f64>());
    (cols / PACKED_NR).max(1) * PACKED_NR
}

/// Verbatim structure of the shipped `packed_gemm_serial` tile loop.
fn gemm_scalar(a: &[f64], b: &[f64], m: usize, k: usize, n: usize, out: &mut [f64]) {
    let m_full = m - m % PACKED_MR;
    let n_full = n - n % PACKED_NR;
    let nc = panel_cols(k);
    let mut bp = vec![0.0f64; k * PACKED_NR];
    let mut jc = 0;
    while jc < n_full {
        let jc_end = (jc + nc).min(n_full);
        let mut j0 = jc;
        while j0 < jc_end {
            for kk in 0..k {
                bp[kk * PACKED_NR..kk * PACKED_NR + PACKED_NR]
                    .copy_from_slice(&b[kk * n + j0..kk * n + j0 + PACKED_NR]);
            }
            let mut i0 = 0;
            while i0 < m_full {
                let mut acc = [[0.0f64; PACKED_NR]; PACKED_MR];
                for kk in 0..k {
                    let brow = &bp[kk * PACKED_NR..kk * PACKED_NR + PACKED_NR];
                    for (ii, row) in acc.iter_mut().enumerate() {
                        let av = a[(i0 + ii) * k + kk];
                        for (slot, &bv) in row.iter_mut().zip(brow) {
                            *slot += av * bv;
                        }
                    }
                }
                for (ii, row) in acc.iter().enumerate() {
                    let base = (i0 + ii) * n + j0;
                    for (slot, &v) in out[base..base + PACKED_NR].iter_mut().zip(row) {
                        *slot += v;
                    }
                }
                i0 += PACKED_MR;
            }
            j0 += PACKED_NR;
        }
        jc += nc;
    }
}

/// Same loop nest, but the register tile is `PACKED_MR` explicit SIMD vectors.
fn gemm_simd(a: &[f64], b: &[f64], m: usize, k: usize, n: usize, out: &mut [f64]) {
    type Lane = Simd<f64, PACKED_NR>;
    let m_full = m - m % PACKED_MR;
    let n_full = n - n % PACKED_NR;
    let nc = panel_cols(k);
    let mut bp = vec![0.0f64; k * PACKED_NR];
    let mut jc = 0;
    while jc < n_full {
        let jc_end = (jc + nc).min(n_full);
        let mut j0 = jc;
        while j0 < jc_end {
            for kk in 0..k {
                bp[kk * PACKED_NR..kk * PACKED_NR + PACKED_NR]
                    .copy_from_slice(&b[kk * n + j0..kk * n + j0 + PACKED_NR]);
            }
            let mut i0 = 0;
            while i0 < m_full {
                let mut acc = [Lane::splat(0.0); PACKED_MR];
                for kk in 0..k {
                    let bv = Lane::from_slice(&bp[kk * PACKED_NR..kk * PACKED_NR + PACKED_NR]);
                    for (ii, slot) in acc.iter_mut().enumerate() {
                        *slot += Lane::splat(a[(i0 + ii) * k + kk]) * bv;
                    }
                }
                for (ii, lane) in acc.iter().enumerate() {
                    let base = (i0 + ii) * n + j0;
                    let prev = Lane::from_slice(&out[base..base + PACKED_NR]);
                    (prev + lane).copy_to_slice(&mut out[base..base + PACKED_NR]);
                }
                i0 += PACKED_MR;
            }
            j0 += PACKED_NR;
        }
        jc += nc;
    }
}

/// The ship gate: a single differing bit makes the SIMD variant unshippable.
fn assert_bit_identical(m: usize, k: usize, n: usize) {
    let a = fill(m * k, 0x5EED_1234);
    let b = fill(k * n, 0x0FF1_CE55);
    let mut s = vec![0.0f64; m * n];
    let mut v = vec![0.0f64; m * n];
    gemm_scalar(&a, &b, m, k, n, &mut s);
    gemm_simd(&a, &b, m, k, n, &mut v);
    for (idx, (x, y)) in s.iter().zip(v.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "simd tile diverged from scalar tile at {idx} ({m}x{k}x{n})"
        );
    }
}

fn bench_gemm_microkernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_microkernel");
    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_secs(2));
    group.measurement_time(std::time::Duration::from_secs(5));

    for n in [256usize, 512, 1024] {
        assert_bit_identical(n, n, n);
        let a = fill(n * n, 0x5EED_1234);
        let b = fill(n * n, 0x0FF1_CE55);
        let mut out = vec![0.0f64; n * n];

        // Interleaved A/B plus an A/A null control: the control is the same
        // scalar kernel benched twice, so its spread bounds what counts as a
        // real difference between the two arms on this host.
        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| {
                out.fill(0.0);
                gemm_scalar(black_box(&a), black_box(&b), n, n, n, &mut out);
                black_box(out[0]);
            });
        });
        group.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| {
                out.fill(0.0);
                gemm_simd(black_box(&a), black_box(&b), n, n, n, &mut out);
                black_box(out[0]);
            });
        });
        group.bench_with_input(BenchmarkId::new("scalar_null_control", n), &n, |bench, _| {
            bench.iter(|| {
                out.fill(0.0);
                gemm_scalar(black_box(&a), black_box(&b), n, n, n, &mut out);
                black_box(out[0]);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_gemm_microkernel);
criterion_main!(benches);
