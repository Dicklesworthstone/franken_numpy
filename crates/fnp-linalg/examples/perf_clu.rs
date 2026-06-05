//! Times complex_solve_nxn (interleaved complex LU). Serial vs parallel via
//! RAYON_NUM_THREADS=1 vs default.
use std::time::Instant;
use fnp_linalg::complex_det_nxn;
fn build(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    let mut v: Vec<f64> = (0..2 * n * n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }).collect();
    for i in 0..n { v[2 * (i * n + i)] += 2.0 * n as f64; }
    v
}
fn median(mut xs: Vec<f64>) -> f64 { xs.sort_by(|a,b| a.partial_cmp(b).unwrap()); xs[xs.len()/2] }
fn main() {
    println!("threads={}", rayon::current_num_threads());
    for &n in &[256usize, 512, 1024] {
        let a = build(n, 0x1234);
        
        let it = if n <= 512 { 7 } else { 4 };
        let mut ts = Vec::new(); let mut cs = 0u64;
        for _ in 0..it {
            let t = Instant::now();
            let (dr, di) = complex_det_nxn(&a, n).unwrap();
            ts.push(t.elapsed().as_secs_f64() * 1e3);
            cs = dr.to_bits() ^ di.to_bits();
        }
        println!("n={n:5} {:9.3} ms checksum=0x{cs:016x}", median(ts));
    }
}
