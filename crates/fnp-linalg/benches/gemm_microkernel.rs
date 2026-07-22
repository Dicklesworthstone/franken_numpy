//! Crate-local probe for the packed f64 GEMM register tile.
//!
//! Two questions, both about the shipped `packed_gemm_serial` microkernel:
//!
//! 1. Does an explicit `std::simd` tile beat the autovectorized scalar tile at
//!    the production shape (MR=4, NR=8)? **Answered: no** - every delta landed
//!    inside the A/A null-control spread (ledger, 2026-07-22 REJECT). The two
//!    arms are retained as the regression baseline.
//! 2. Is the production tile *geometry* itself optimal? The shipped MR=4 x NR=8
//!    predates any measured sweep. On AVX2 (16 vector registers, 4 f64 each)
//!    that tile needs 8 accumulator registers; on AVX-512 (32 registers, 8 f64
//!    each) it needs only 4, leaving most of the register file idle. This is
//!    the retry predicate the REJECT row recorded.
//!
//! BIT-IDENTITY: changing MR/NR regroups which output elements share a tile but
//! never reorders any single element's k-sum, which stays ascending. Every
//! shape must therefore be byte-identical to the production (4,8) baseline, and
//! `assert_all_shapes_bit_identical` enforces exactly that - a shape that
//! differed by one bit would be unshippable against the golden-sha256 locks
//! regardless of speed.
//!
//! Run: cargo bench -p fnp-linalg --profile bench-fast --bench gemm_microkernel

#![feature(portable_simd)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::simd::Simd;

/// Production tile geometry, mirrored from `fnp-linalg`'s `packed_gemm_serial`.
const PROD_MR: usize = 4;
const PROD_NR: usize = 8;

/// Deterministic operand fill (SplitMix64; no RNG dep, no clock).
fn fill(count: usize, seed: u64) -> Vec<f64> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        })
        .collect()
}

/// ~256 KiB L2-resident column panel, rounded down to whole register tiles.
fn panel_cols(k: usize, nr: usize) -> usize {
    let cols = (256 * 1024) / (k.max(1) * core::mem::size_of::<f64>());
    (cols / nr).max(1) * nr
}

/// out[i, j0..n] += sum_k a[i,k]*b[k,j], ascending k (bit-exact tail).
fn row_tail(a: &[f64], b: &[f64], out: &mut [f64], i: usize, k: usize, n: usize, j0: usize) {
    let a_base = i * k;
    let o_base = i * n;
    for j in j0..n {
        let mut s = 0.0f64;
        for kk in 0..k {
            s += a[a_base + kk] * b[kk * n + j];
        }
        out[o_base + j] += s;
    }
}

/// The shipped kernel, generic over tile geometry. `MR`/`NR` change only which
/// output elements share a register tile; each element still sums k ascending.
fn gemm_tiled<const MR: usize, const NR: usize>(
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f64],
) {
    let m_full = m - m % MR;
    let n_full = n - n % NR;
    let nc = panel_cols(k, NR);
    let mut bp = vec![0.0f64; k * NR];
    let mut jc = 0;
    while jc < n_full {
        let jc_end = (jc + nc).min(n_full);
        let mut j0 = jc;
        while j0 < jc_end {
            for kk in 0..k {
                bp[kk * NR..kk * NR + NR].copy_from_slice(&b[kk * n + j0..kk * n + j0 + NR]);
            }
            let mut i0 = 0;
            while i0 < m_full {
                let mut acc = [[0.0f64; NR]; MR];
                for kk in 0..k {
                    let brow = &bp[kk * NR..kk * NR + NR];
                    for (ii, row) in acc.iter_mut().enumerate() {
                        let av = a[(i0 + ii) * k + kk];
                        for (slot, &bv) in row.iter_mut().zip(brow) {
                            *slot += av * bv;
                        }
                    }
                }
                for (ii, row) in acc.iter().enumerate() {
                    let base = (i0 + ii) * n + j0;
                    for (slot, &v) in out[base..base + NR].iter_mut().zip(row) {
                        *slot += v;
                    }
                }
                i0 += MR;
            }
            j0 += NR;
        }
        jc += nc;
    }
    for i in 0..m_full {
        row_tail(a, b, out, i, k, n, n_full);
    }
    for i in m_full..m {
        row_tail(a, b, out, i, k, n, 0);
    }
}

/// Production geometry with an explicit SIMD tile (the answered question).
fn gemm_simd(a: &[f64], b: &[f64], m: usize, k: usize, n: usize, out: &mut [f64]) {
    type Lane = Simd<f64, PROD_NR>;
    let m_full = m - m % PROD_MR;
    let n_full = n - n % PROD_NR;
    let nc = panel_cols(k, PROD_NR);
    let mut bp = vec![0.0f64; k * PROD_NR];
    let mut jc = 0;
    while jc < n_full {
        let jc_end = (jc + nc).min(n_full);
        let mut j0 = jc;
        while j0 < jc_end {
            for kk in 0..k {
                bp[kk * PROD_NR..kk * PROD_NR + PROD_NR]
                    .copy_from_slice(&b[kk * n + j0..kk * n + j0 + PROD_NR]);
            }
            let mut i0 = 0;
            while i0 < m_full {
                let mut acc = [Lane::splat(0.0); PROD_MR];
                for kk in 0..k {
                    let bv = Lane::from_slice(&bp[kk * PROD_NR..kk * PROD_NR + PROD_NR]);
                    for (ii, slot) in acc.iter_mut().enumerate() {
                        *slot += Lane::splat(a[(i0 + ii) * k + kk]) * bv;
                    }
                }
                for (ii, lane) in acc.iter().enumerate() {
                    let base = (i0 + ii) * n + j0;
                    let prev = Lane::from_slice(&out[base..base + PROD_NR]);
                    (prev + lane).copy_to_slice(&mut out[base..base + PROD_NR]);
                }
                i0 += PROD_MR;
            }
            j0 += PROD_NR;
        }
        jc += nc;
    }
    for i in 0..m_full {
        row_tail(a, b, out, i, k, n, n_full);
    }
    for i in m_full..m {
        row_tail(a, b, out, i, k, n, 0);
    }
}

fn run<const MR: usize, const NR: usize>(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    gemm_tiled::<MR, NR>(a, b, n, n, n, &mut out);
    out
}

/// Ship gate: every candidate geometry must be byte-identical to production.
fn assert_all_shapes_bit_identical(n: usize) {
    let a = fill(n * n, 0x5EED_1234);
    let b = fill(n * n, 0x0FF1_CE55);
    let reference = run::<PROD_MR, PROD_NR>(&a, &b, n);

    let mut candidates: Vec<(&str, Vec<f64>)> = vec![
        ("2x8", run::<2, 8>(&a, &b, n)),
        ("6x8", run::<6, 8>(&a, &b, n)),
        ("8x8", run::<8, 8>(&a, &b, n)),
        ("4x16", run::<4, 16>(&a, &b, n)),
        ("6x16", run::<6, 16>(&a, &b, n)),
        ("8x16", run::<8, 16>(&a, &b, n)),
    ];
    let mut simd_out = vec![0.0f64; n * n];
    gemm_simd(&a, &b, n, n, n, &mut simd_out);
    candidates.push(("simd_4x8", simd_out));

    for (label, candidate) in &candidates {
        for (idx, (x, y)) in reference.iter().zip(candidate.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "tile {label} diverged from production 4x8 at element {idx} (n={n})"
            );
        }
    }
}

fn bench_gemm_microkernel(c: &mut Criterion) {
    assert_all_shapes_bit_identical(256);

    let mut group = c.benchmark_group("gemm_microkernel");
    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_secs(2));
    group.measurement_time(std::time::Duration::from_secs(5));

    for n in [512usize, 1024] {
        let a = fill(n * n, 0x5EED_1234);
        let b = fill(n * n, 0x0FF1_CE55);
        let mut out = vec![0.0f64; n * n];

        macro_rules! arm {
            ($label:expr, $mr:expr, $nr:expr) => {
                group.bench_with_input(BenchmarkId::new($label, n), &n, |bench, _| {
                    bench.iter(|| {
                        out.fill(0.0);
                        gemm_tiled::<$mr, $nr>(black_box(&a), black_box(&b), n, n, n, &mut out);
                        black_box(out[0]);
                    });
                });
            };
        }

        // Production geometry first, then the sweep, then an A/A repeat of
        // production as the null control bounding what counts as a real delta.
        arm!("prod_4x8", 4, 8);
        arm!("tile_2x8", 2, 8);
        arm!("tile_6x8", 6, 8);
        arm!("tile_8x8", 8, 8);
        arm!("tile_4x16", 4, 16);
        arm!("tile_6x16", 6, 16);
        arm!("tile_8x16", 8, 16);
        group.bench_with_input(BenchmarkId::new("simd_4x8", n), &n, |bench, _| {
            bench.iter(|| {
                out.fill(0.0);
                gemm_simd(black_box(&a), black_box(&b), n, n, n, &mut out);
                black_box(out[0]);
            });
        });
        arm!("null_control_4x8", 4, 8);
    }

    group.finish();
}

criterion_group!(benches, bench_gemm_microkernel);
criterion_main!(benches);
