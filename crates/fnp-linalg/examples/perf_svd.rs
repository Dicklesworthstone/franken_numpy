use std::time::Instant;
use fnp_linalg::{svd_mxn, svd_mxn_full};
fn gen_m(n: usize, seed: u64) -> Vec<f64> { let mut s=seed|1; (0..n*n).map(|_| { s=s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((s>>11) as f64/(1u64<<53) as f64)*2.0-1.0 }).collect() }
fn main() {
    for &n in &[256usize, 512] {
        let a = gen_m(n, 0x1234);
        let t = Instant::now(); let _ = svd_mxn(&a, n, n).unwrap(); println!("svd_values n={n} {:.1} ms", t.elapsed().as_secs_f64()*1e3);
        let t = Instant::now(); let _ = svd_mxn_full(&a, n, n).unwrap(); println!("svd_FULL   n={n} {:.1} ms", t.elapsed().as_secs_f64()*1e3);
    }
}
