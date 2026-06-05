//! Profiling-only golden + timing harness for `franken_numpy-g9jvo`.
//!
//! Compares the current public `UFuncArray::fft` path against a reference copy
//! of the prior radix-2 bit-reversal loop on the exact `criterion_core_ops`
//! `fft_65536` fixture.

use std::time::Instant;

use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;

fn fft_fixture(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| {
            let t = i as f64 / len as f64;
            (std::f64::consts::TAU * 5.0 * t).sin() + 0.5 * (std::f64::consts::TAU * 13.0 * t).cos()
        })
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

fn fft_mul(a: f64, b: f64) -> f64 {
    const EPS: f64 = 1e-14;
    if (a.abs() < EPS && !b.is_finite()) || (b.abs() < EPS && !a.is_finite()) {
        0.0
    } else {
        a * b
    }
}

fn fft_pow2_old_bitrev(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j: usize = 0;
    for i in 0..n {
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
        let mut m = n >> 1;
        while m >= 1 && j >= m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle_step = -std::f64::consts::TAU / len as f64;
        let wn_re = angle_step.cos();
        let wn_im = angle_step.sin();
        let mut start = 0;
        while start < n {
            let mut w_re = 1.0;
            let mut w_im = 0.0;
            for k in 0..half {
                let even = start + k;
                let odd = start + k + half;
                let tr = fft_mul(w_re, re[odd]) - fft_mul(w_im, im[odd]);
                let ti = fft_mul(w_re, im[odd]) + fft_mul(w_im, re[odd]);
                re[odd] = re[even] - tr;
                im[odd] = im[even] - ti;
                re[even] += tr;
                im[even] += ti;
                let new_w_re = w_re * wn_re - w_im * wn_im;
                let new_w_im = w_re * wn_im + w_im * wn_re;
                w_re = new_w_re;
                w_im = new_w_im;
            }
            start += len;
        }
        len <<= 1;
    }
}

fn fft_reference(values: &[f64]) -> Vec<f64> {
    let mut re = values.to_vec();
    let mut im = vec![0.0; values.len()];
    fft_pow2_old_bitrev(&mut re, &mut im);
    let mut out = vec![0.0; values.len() * 2];
    for ((pair, &real), &imag) in out.chunks_exact_mut(2).zip(re.iter()).zip(im.iter()) {
        pair[0] = real;
        pair[1] = imag;
    }
    out
}

fn current_fft(input: &UFuncArray) -> Vec<f64> {
    input.fft(None).expect("fft").values().to_vec()
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
    let len = 65_536usize;
    let values = fft_fixture(len);
    let input = UFuncArray::new(vec![len], values.clone(), DType::F64).expect("input");
    let reference = fft_reference(&values);
    let current = current_fft(&input);
    assert_eq!(current, reference);

    let iters = 80;
    for _ in 0..10 {
        std::hint::black_box(fft_reference(&values));
        std::hint::black_box(current_fft(&input));
    }

    let reference_median = time_median_ms(iters, || fft_reference(&values));
    let current_median = time_median_ms(iters, || current_fft(&input));

    println!(
        "fft_65536 current_fnv1a=0x{:016x} reference_fnv1a=0x{:016x}",
        fnv1a(&current),
        fnv1a(&reference)
    );
    println!(
        "fft_65536 reference_median_ms={reference_median:.4} current_median_ms={current_median:.4}"
    );
}
