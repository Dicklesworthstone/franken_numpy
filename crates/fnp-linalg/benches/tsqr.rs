//! Tall-skinny QR (TSQR) probe: `tsqr_r` vs the shapes where LAPACK's
//! `dgeqrf` falls into its unblocked BLAS-2 path.
//!
//! There is deliberately no in-tree baseline arm here: `qr_mxn` forms an
//! explicit m×m Q, which at m = 1e6 is 1e12 elements, so the shipped native QR
//! cannot run these shapes at all. The comparator is `numpy.linalg.qr`
//! captured on the same host (see the ledger row), and this bench supplies the
//! fnp side plus the correctness gates.
//!
//! Run: cargo bench -p fnp-linalg --profile bench-fast --bench tsqr

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_linalg::{qr_mxn, tsqr_r};
use std::hint::black_box;

/// Deterministic operand fill (SplitMix64, no RNG dep, no clock).
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

/// Gate 1: R must be upper triangular.
fn assert_upper_triangular(r: &[f64], n: usize) {
    for i in 0..n {
        for j in 0..i {
            assert_eq!(r[i * n + j], 0.0, "R not upper triangular at ({i},{j})");
        }
    }
}

/// Gate 2: RᵀR == AᵀA. For full-rank A this determines R up to row signs, and
/// it never forms Q, so it is a genuinely independent check of the whole tree.
fn assert_gram_identity(a: &[f64], r: &[f64], m: usize, n: usize) {
    let mut ata = vec![0.0f64; n * n];
    for row in 0..m {
        for i in 0..n {
            let av = a[row * n + i];
            for j in 0..n {
                ata[i * n + j] += av * a[row * n + j];
            }
        }
    }
    let mut rtr = vec![0.0f64; n * n];
    for row in 0..n {
        for i in 0..n {
            let rv = r[row * n + i];
            for j in 0..n {
                rtr[i * n + j] += rv * r[row * n + j];
            }
        }
    }
    let scale = ata.iter().fold(0.0f64, |acc, v| acc.max(v.abs())).max(1.0);
    for idx in 0..n * n {
        let diff = (ata[idx] - rtr[idx]).abs();
        assert!(
            diff <= 1e-9 * scale,
            "Gram mismatch at {idx}: AtA={} RtR={} (tol {})",
            ata[idx],
            rtr[idx],
            1e-9 * scale
        );
    }
}

/// Gate 3: agreement with the shipped `qr_mxn` at a size where forming Q is
/// affordable. Compared up to row sign, which neither LAPACK nor Householder
/// pins down.
fn assert_matches_qr_mxn(m: usize, n: usize) {
    let a = fill(m * n, 0xA11C_E501);
    let (_, r_ref) = qr_mxn(&a, m, n).expect("qr_mxn reference");
    let r = tsqr_r(&a, m, n).expect("tsqr_r");
    let scale = r_ref
        .iter()
        .take(n * n)
        .fold(0.0f64, |acc, v| acc.max(v.abs()))
        .max(1.0);
    for i in 0..n {
        // qr_mxn returns R as m×n; its leading n×n block is the comparator.
        let flip = if r_ref[i * n + i].signum() == r[i * n + i].signum() {
            1.0
        } else {
            -1.0
        };
        for j in 0..n {
            let diff = (r_ref[i * n + j] - flip * r[i * n + j]).abs();
            assert!(
                diff <= 1e-8 * scale,
                "tsqr_r disagrees with qr_mxn at ({i},{j}): {} vs {}",
                r_ref[i * n + j],
                flip * r[i * n + j]
            );
        }
    }
}

fn bench_tsqr(c: &mut Criterion) {
    // Correctness gates run before any timing.
    assert_matches_qr_mxn(256, 16);
    assert_matches_qr_mxn(512, 8);
    {
        let (m, n) = (8192usize, 16usize);
        let a = fill(m * n, 0xBEEF_0001);
        let r = tsqr_r(&a, m, n).expect("tsqr_r");
        assert_upper_triangular(&r, n);
        assert_gram_identity(&a, &r, m, n);
    }

    let mut group = c.benchmark_group("tsqr");
    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_secs(2));
    group.measurement_time(std::time::Duration::from_secs(5));

    // The shapes where dgeqrf degenerates: m/n >= ~1e4.
    for (m, n) in [(1_000_000usize, 16usize), (1_000_000, 8), (2_000_000, 8)] {
        let a = fill(m * n, 0xBEEF_0002);
        group.bench_with_input(
            BenchmarkId::new("tsqr_r", format!("{m}x{n}")),
            &(m, n),
            |bench, &(m, n)| {
                bench.iter(|| {
                    let r = tsqr_r(black_box(&a), m, n).expect("tsqr_r");
                    black_box(r[0]);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_tsqr);
criterion_main!(benches);
