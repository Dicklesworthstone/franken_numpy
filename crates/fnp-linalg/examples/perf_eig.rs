use std::time::Instant;
use fnp_linalg::{eigh_nxn, eig_nxn};
fn sym(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    let b: Vec<f64> = (0..n*n).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((s>>11) as f64/(1u64<<53) as f64)*2.0-1.0 }).collect();
    let mut a = vec![0.0; n*n];
    for i in 0..n { for j in 0..n { let mut t=0.0; for k in 0..n { t+=b[i*n+k]*b[j*n+k]; } a[i*n+j]=t; } }
    a
}
fn gen_mat(n: usize, seed: u64) -> Vec<f64> { let mut s=seed|1; (0..n*n).map(|_| { s=s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((s>>11) as f64/(1u64<<53) as f64)*2.0-1.0 }).collect() }
fn main() {
    for &n in &[512usize, 1024] {
        let a = sym(n, 0x1234);
        let t = Instant::now(); let _ = eigh_nxn(&a, n).unwrap(); println!("eigh(vectors) n={n} {:.1} ms", t.elapsed().as_secs_f64()*1e3);
        let g = gen_mat(n, 0x55);
        let t = Instant::now(); let _ = eig_nxn(&g, n); println!("eig(nonsym) n={n} {:.1} ms", t.elapsed().as_secs_f64()*1e3);
    }
}
