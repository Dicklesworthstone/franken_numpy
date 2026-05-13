#![no_main]

use libfuzzer_sys::fuzz_target;

// Fuzz QR on rectangular m×n matrices. The decomposition uses
// Householder reflections; non-finite or pathological inputs must
// return Err, not panic.
fuzz_target!(|data: (u8, u8, &[u8])| {
    let (m_byte, n_byte, bytes) = data;
    // Cap m,n at 16 so the per-iteration matmul stays under budget.
    let m = (m_byte as usize) % 17;
    let n = (n_byte as usize) % 17;
    if m == 0 || n == 0 {
        return;
    }
    let need = m * n * 8;
    if bytes.len() < need {
        return;
    }
    let mut a = Vec::with_capacity(m * n);
    for i in 0..(m * n) {
        let chunk = &bytes[i * 8..(i + 1) * 8];
        let bits = u64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3],
            chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        a.push(f64::from_bits(bits));
    }
    let _ = fnp_linalg::qr_mxn(&a, m, n);
});
