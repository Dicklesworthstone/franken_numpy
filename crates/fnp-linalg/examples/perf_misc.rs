use std::time::Instant;
use fnp_linalg::matrix_power_nxn;
fn gm(n: usize, seed: u64) -> Vec<f64> { let mut s=seed|1; (0..n*n).map(|_| { s=s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((s>>11) as f64/(1u64<<53) as f64)*0.1 }).collect() }
fn med(mut xs: Vec<f64>) -> f64 { xs.sort_by(|a,b| a.partial_cmp(b).unwrap()); xs[xs.len()/2] }
fn main() {
    for &n in &[512usize] {
        let mut a = gm(n, 1); for i in 0..n { a[i*n+i] += 1.0; }
        // warmup
        let _ = matrix_power_nxn(&a, n, 2);
        for &p in &[2i64, 5, 16] {
            let mut ts = Vec::new();
            for _ in 0..4 { let t = Instant::now(); let _ = matrix_power_nxn(&a, n, p); ts.push(t.elapsed().as_secs_f64()*1e3); }
            println!("matrix_power^{p} n={n} {:.1} ms (warmed median)", med(ts));
        }
    }
}
