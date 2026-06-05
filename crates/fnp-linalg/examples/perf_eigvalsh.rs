use std::time::Instant;
use fnp_linalg::eigvalsh_nxn;
fn sym(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    let b: Vec<f64> = (0..n*n).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((s>>11) as f64/(1u64<<53) as f64)*2.0-1.0 }).collect();
    let mut a = vec![0.0; n*n];
    for i in 0..n { for j in 0..n { let mut t=0.0; for k in 0..n { t+=b[i*n+k]*b[j*n+k]; } a[i*n+j]=t; } }
    a
}
fn main() {
    for &n in &[512usize, 1024, 2048] {
        let a = sym(n, 0x1234);
        let t = Instant::now();
        let _ev = eigvalsh_nxn(&a, n).unwrap();
        println!("eigvalsh n={n} {:.1} ms", t.elapsed().as_secs_f64()*1e3);
    }
}
