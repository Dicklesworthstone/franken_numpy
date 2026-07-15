#![allow(clippy::needless_range_loop)]

use criterion::{Criterion, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_ufunc::{UFuncArray, convolve_gather_fill};
use rayon::prelude::*;
use std::hint::black_box;

// np.correlate/convolve route 1-D f64 through UFuncArray::convolve_mode, which picks
// between a direct vectorized SAXPY scatter and an FFT-based linear convolution via a
// cost-model gate. For inputs that take the DIRECT parallel scatter with a LARGE
// kernel (m comparable to full_len), the block size matters: the old `.max(m)` floor
// collapsed the split to a few blocks (idle cores); the new `.max(64)` floor keeps
// ~2 blocks/thread. This bench A/Bs the two block-sizing strategies on the same data
// in the same run (so the comparison is load-robust), and also covers the small/medium
// regime where the two are identical.
fn mk(n: usize, seed: u64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(seed);
            (h >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect()
}

// Direct parallel scatter with a configurable block floor — mirrors the production
// kernel so we can A/B the `.max(m)` (old) vs `.max(64)` (new) block sizing.
fn scatter(a: &[f64], k: &[f64], floor_is_m: bool) -> Vec<f64> {
    let n = a.len();
    let m = k.len();
    let full_len = n + m - 1;
    let mut full = vec![0.0f64; full_len];
    let threads = rayon::current_num_threads();
    let chunk = if floor_is_m {
        (full_len / (threads * 2)).max(m).max(1)
    } else {
        (full_len / (threads * 2)).max(64)
    };
    full.par_chunks_mut(chunk).enumerate().for_each(|(c, out)| {
        let lo = c * chunk;
        let hi = lo + out.len();
        let i_start = lo.saturating_sub(m - 1);
        let i_end = hi.min(n);
        for i in i_start..i_end {
            let ai = a[i];
            let j0 = lo.saturating_sub(i);
            let j1 = m.min(hi - i);
            if j0 >= j1 {
                continue;
            }
            let out_off = i + j0 - lo;
            for (d, &kv) in out[out_off..out_off + (j1 - j0)]
                .iter_mut()
                .zip(k[j0..j1].iter())
            {
                *d += ai * kv;
            }
        }
    });
    full
}

fn former_scalar_kernel(a: &[f64], scalar: f64) -> Vec<f64> {
    let reversed_kernel = vec![scalar];
    let mut full = vec![0.0f64; a.len()];
    convolve_gather_fill(a, &reversed_kernel, a.len(), 1, &mut full, 0);
    full
}

fn bench_scalar_kernel(c: &mut Criterion) {
    let n = 1usize << 18;
    let mut data = mk(n, 23);
    data[0] = -0.0;
    data[1] = f64::from_bits(0x7ff8_0000_0000_0042);
    data[2] = f64::INFINITY;
    data[3] = f64::NEG_INFINITY;
    let scalar = -2.0;
    let a = UFuncArray::new(vec![n], data.clone(), DType::F64).unwrap();
    let k = UFuncArray::new(vec![1], vec![scalar], DType::F64).unwrap();

    let former = former_scalar_kernel(&data, scalar);
    let candidate = a.convolve_mode(&k, "full").unwrap();
    assert_eq!(candidate.shape(), &[n]);
    assert_eq!(candidate.dtype(), DType::F64);
    assert_eq!(
        candidate
            .values()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        former
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let mut group = c.benchmark_group("convolve_scalar_kernel");
    group.bench_function("former_gather_setup_zero_fill", |b| {
        b.iter(|| black_box(former_scalar_kernel(black_box(&data), scalar)))
    });
    group.bench_function("direct_single_pass_collect", |b| {
        b.iter(|| black_box(a.convolve_mode(black_box(&k), "full").unwrap()))
    });
    group.finish();
}

fn bench(c: &mut Criterion) {
    bench_scalar_kernel(c);
    if std::env::args().any(|arg| arg == "convolve_scalar_kernel") {
        return;
    }
    // (n, m): direct+large-m cases where block sizing matters, plus a small-m control.
    let shapes = [
        (5_000usize, 20_000usize), // kernel >> signal: old -> ~1 block, new -> ~32
        (20_000, 12_000),          // kernel ~ signal
        (200_000, 1_000),          // small-m control: identical block size
    ];
    for &(n, m) in &shapes {
        let a = mk(n, 1);
        let k = mk(m, 7);
        // sanity: both strategies agree bit-for-bit (chunk count never changes result)
        assert_eq!(scatter(&a, &k, true), scatter(&a, &k, false));
        let _ = UFuncArray::new(vec![n], a.clone(), DType::F64).unwrap();
        let mut g = c.benchmark_group(format!("convchunk_{n}x{m}"));
        g.sample_size(20);
        g.bench_function("old_max_m", |b| {
            b.iter(|| black_box(scatter(black_box(&a), black_box(&k), true)))
        });
        g.bench_function("new_floor64", |b| {
            b.iter(|| black_box(scatter(black_box(&a), black_box(&k), false)))
        });
        g.finish();
    }

    let n = 1_000_000usize;
    let m = 8usize;
    let a = UFuncArray::new(vec![n], mk(n, 11), DType::F64).unwrap();
    let k = UFuncArray::new(vec![m], mk(m, 13), DType::F64).unwrap();
    let mut g = c.benchmark_group("convprod_short_tail_1000000x8");
    g.sample_size(10);
    g.bench_function("prod_convolve", |b| {
        b.iter(|| black_box(a.convolve_mode(black_box(&k), "full").unwrap()));
    });
    g.finish();
}
criterion_group!(benches, bench);
criterion_main!(benches);
