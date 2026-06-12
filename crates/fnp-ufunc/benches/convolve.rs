use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;
use std::hint::black_box;

// np.correlate/convolve route 1-D f64 through UFuncArray::convolve_mode, which picks
// between a direct vectorized SAXPY scatter and an FFT-based linear convolution. The
// FFT gate (n.min(m) >= 64 && n*m >= 2^18) is suspected too aggressive: it sends
// medium problems to the ~1.7ms-fixed-cost FFT path when the direct SAXPY is far
// faster. This bench exercises convolve_mode across the regime so the gate value can
// be A/B'd (run with the current gate, then with the gate raised).
fn mk(n: usize, seed: u64) -> UFuncArray {
    let v: Vec<f64> = (0..n)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(seed);
            (h >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect();
    UFuncArray::new(vec![n], v, DType::F64).unwrap()
}

fn bench(c: &mut Criterion) {
    // (n, m): a-length, kernel-length. Spans the regime the FFT gate covers.
    let shapes = [
        (5_000usize, 200usize), // was FFT (slow); now direct
        (50_000, 500),          // was FFT (slow); now direct
        (2_000, 2_000),         // was FFT (slow); now direct
        (100_000, 1_000),       // was FFT (slow); now direct
        (200_000, 4_000),       // was FFT; now direct (FFT loses here)
        (100_000, 20_000),      // huge kernel: FFT genuinely wins, stays FFT
    ];
    for &(n, m) in &shapes {
        let a = mk(n, 1);
        let k = mk(m, 7);
        let _ = a.convolve_mode(&k, "full").unwrap();
        let mut g = c.benchmark_group(format!("convolve_{n}x{m}"));
        g.sample_size(20);
        g.bench_with_input(BenchmarkId::new("full", format!("{n}x{m}")), &(), |b, _| {
            b.iter(|| black_box(a.convolve_mode(black_box(&k), "full").unwrap()))
        });
        g.finish();
    }
}
criterion_group!(benches, bench);
criterion_main!(benches);
