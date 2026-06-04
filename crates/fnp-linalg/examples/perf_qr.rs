//! Times qr_nxn (Householder QR behind np.linalg.qr / lstsq). Serial vs parallel
//! via RAYON_NUM_THREADS:
//!   RAYON_NUM_THREADS=1 cargo run --release --example perf_qr -p fnp-linalg
//!   cargo run --release --example perf_qr -p fnp-linalg

use std::time::Instant;

use fnp_linalg::qr_nxn;

fn gen_mat(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    (0..n * n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn main() {
    println!("threads={}", rayon::current_num_threads());
    for &n in &[256usize, 512, 1024] {
        let a = gen_mat(n, 0x1234);
        let iters = if n <= 512 { 7 } else { 4 };
        let mut times = Vec::new();
        let mut checksum = 0u64;
        for _ in 0..iters {
            let t = Instant::now();
            let (q, r) = qr_nxn(&a, n).unwrap();
            times.push(t.elapsed().as_secs_f64() * 1e3);
            checksum = q.iter().chain(r.iter()).fold(0xcbf29ce484222325u64, |h, x| {
                x.to_bits()
                    .to_le_bytes()
                    .iter()
                    .fold(h, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3))
            });
        }
        let ms = median(times);
        println!("n={n:5} {ms:9.3} ms  checksum=0x{checksum:016x}");
    }
}
