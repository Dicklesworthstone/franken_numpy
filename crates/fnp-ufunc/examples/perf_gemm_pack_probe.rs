//! Isolated probe: does BLIS-style A+B panel packing beat the current
//! stride-n register-tiled GEMM inner loop? Both kernels here mirror the
//! production `matmul_accumulate_serial` structure (MR=4, NR=8, NC column
//! panel, increasing-k accumulation) so the ONLY difference is memory layout:
//! `gemm_base` reads B with stride n (as production does), `gemm_packed`
//! pre-packs the A i0-block and the B jc-panel into contiguous kk-major
//! micropanels. Packing reorders memory access only — never the arithmetic —
//! so the outputs must be BIT-IDENTICAL (verified via FNV-1a here).
//!
//! Single-threaded on purpose: packing is a per-core locality lever, orthogonal
//! to the row-band parallelism in the production driver.
//!
//! Run: `cargo run --release --example perf_gemm_pack_probe -p fnp-ufunc`

use std::time::Instant;

const MR: usize = 4;
const NR: usize = 8;

fn gen_mat(rows: usize, cols: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    (0..rows * cols)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

// Baseline: exact mirror of matmul_accumulate_serial's hot path (sizes chosen
// divisible by MR and NR, so no tail handling needed).
fn gemm_base(a: &[f64], b: &[f64], m: usize, k: usize, n: usize, c: &mut [f64]) {
    let nc = {
        const PANEL_BYTES: usize = 256 * 1024;
        let cols = PANEL_BYTES / (k.max(1) * 8);
        (cols / NR).max(1) * NR
    };
    let mut jc = 0;
    while jc < n {
        let jc_end = (jc + nc).min(n);
        let mut i0 = 0;
        while i0 < m {
            let mut j0 = jc;
            while j0 < jc_end {
                let mut acc = [[0.0f64; NR]; MR];
                for kk in 0..k {
                    let bs = &b[kk * n + j0..kk * n + j0 + NR];
                    for (ii, row) in acc.iter_mut().enumerate() {
                        let av = a[(i0 + ii) * k + kk];
                        for (slot, &bv) in row.iter_mut().zip(bs) {
                            *slot += av * bv;
                        }
                    }
                }
                for (ii, row) in acc.iter().enumerate() {
                    let base = (i0 + ii) * n + j0;
                    for (slot, &v) in c[base..base + NR].iter_mut().zip(row) {
                        *slot += v;
                    }
                }
                j0 += NR;
            }
            i0 += MR;
        }
        jc += nc;
    }
}

// Packed: pack each B jc-panel into NR-wide kk-major micropanels (contiguous),
// and each A i0-block into MR-wide kk-major micropanels, so the inner loop reads
// both operands sequentially. Identical accumulation order => bit-identical.
fn gemm_packed(a: &[f64], b: &[f64], m: usize, k: usize, n: usize, c: &mut [f64]) {
    let nc = {
        const PANEL_BYTES: usize = 256 * 1024;
        let cols = PANEL_BYTES / (k.max(1) * 8);
        (cols / NR).max(1) * NR
    };
    let mut bp = vec![0.0f64; k * NR]; // one B micropanel at a time
    let mut ap = vec![0.0f64; k * MR]; // one A micropanel at a time
    let mut jc = 0;
    while jc < n {
        let jc_end = (jc + nc).min(n);
        let mut j0 = jc;
        while j0 < jc_end {
            // Pack B[:, j0..j0+NR] -> bp[kk*NR + s] (contiguous per kk).
            for kk in 0..k {
                bp[kk * NR..kk * NR + NR].copy_from_slice(&b[kk * n + j0..kk * n + j0 + NR]);
            }
            let mut i0 = 0;
            while i0 < m {
                // Pack A[i0..i0+MR, :] -> ap[kk*MR + ii].
                for kk in 0..k {
                    for ii in 0..MR {
                        ap[kk * MR + ii] = a[(i0 + ii) * k + kk];
                    }
                }
                let mut acc = [[0.0f64; NR]; MR];
                for kk in 0..k {
                    let bs = &bp[kk * NR..kk * NR + NR];
                    let as_ = &ap[kk * MR..kk * MR + MR];
                    for (ii, row) in acc.iter_mut().enumerate() {
                        let av = as_[ii];
                        for (slot, &bv) in row.iter_mut().zip(bs) {
                            *slot += av * bv;
                        }
                    }
                }
                for (ii, row) in acc.iter().enumerate() {
                    let base = (i0 + ii) * n + j0;
                    for (slot, &v) in c[base..base + NR].iter_mut().zip(row) {
                        *slot += v;
                    }
                }
                i0 += MR;
            }
            j0 += NR;
        }
        jc += nc;
    }
}

// Packed-B-only: pack just the B panel once per jc-panel (shared across all i0
// blocks), leave A on the original contiguous layout. Cheaper packing, isolates
// the B-stride effect (which is the suspected bottleneck).
fn gemm_packed_b(a: &[f64], b: &[f64], m: usize, k: usize, n: usize, c: &mut [f64]) {
    let nc = {
        const PANEL_BYTES: usize = 256 * 1024;
        let cols = PANEL_BYTES / (k.max(1) * 8);
        (cols / NR).max(1) * NR
    };
    let mut bp = vec![0.0f64; k * NR];
    let mut jc = 0;
    while jc < n {
        let jc_end = (jc + nc).min(n);
        let mut j0 = jc;
        while j0 < jc_end {
            for kk in 0..k {
                bp[kk * NR..kk * NR + NR].copy_from_slice(&b[kk * n + j0..kk * n + j0 + NR]);
            }
            let mut i0 = 0;
            while i0 < m {
                let mut acc = [[0.0f64; NR]; MR];
                for kk in 0..k {
                    let bs = &bp[kk * NR..kk * NR + NR];
                    for (ii, row) in acc.iter_mut().enumerate() {
                        let av = a[(i0 + ii) * k + kk];
                        for (slot, &bv) in row.iter_mut().zip(bs) {
                            *slot += av * bv;
                        }
                    }
                }
                for (ii, row) in acc.iter().enumerate() {
                    let base = (i0 + ii) * n + j0;
                    for (slot, &v) in c[base..base + NR].iter_mut().zip(row) {
                        *slot += v;
                    }
                }
                i0 += MR;
            }
            j0 += NR;
        }
        jc += nc;
    }
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

fn time<F: FnMut(&mut [f64])>(m: usize, n: usize, iters: usize, mut f: F) -> (f64, u64) {
    let mut last = 0u64;
    let mut ts = Vec::new();
    for _ in 0..iters {
        let mut c = vec![0.0f64; m * n];
        let t = Instant::now();
        f(&mut c);
        ts.push(t.elapsed().as_secs_f64() * 1e3);
        last = fnv1a(&c);
    }
    (median(ts), last)
}

fn main() {
    for &n in &[512usize, 1024, 2048] {
        let (m, k) = (n, n);
        let a = gen_mat(m, k, 0x1234);
        let b = gen_mat(k, n, 0x9abc);
        let iters = if n <= 1024 { 5 } else { 3 };
        let (tb, hb) = time(m, n, iters, |c| gemm_base(&a, &b, m, k, n, c));
        let (tp, hp) = time(m, n, iters, |c| gemm_packed(&a, &b, m, k, n, c));
        let (tpb, hpb) = time(m, n, iters, |c| gemm_packed_b(&a, &b, m, k, n, c));
        let g = |ms: f64| 2.0 * (n as f64).powi(3) / (ms * 1e-3) / 1e9;
        println!("n={n:5}");
        println!("  base     {tb:9.3} ms  {:6.1} GF  hash=0x{hb:016x}", g(tb));
        println!(
            "  packedAB {tp:9.3} ms  {:6.1} GF  {:.2}x  bitmatch={}",
            g(tp),
            tb / tp,
            hp == hb
        );
        println!(
            "  packedB  {tpb:9.3} ms  {:6.1} GF  {:.2}x  bitmatch={}",
            g(tpb),
            tb / tpb,
            hpb == hb
        );
    }
}
