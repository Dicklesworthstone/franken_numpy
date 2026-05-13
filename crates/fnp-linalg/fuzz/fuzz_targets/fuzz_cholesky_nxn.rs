#![no_main]

use libfuzzer_sys::fuzz_target;

// Feed arbitrary f64 matrices into Cholesky. The decomposition rejects
// non-positive-definite inputs with a structured error; the contract is
// "always Ok or Err, never panic." Fuzzing covers shape/size mismatch,
// non-finite entries, and pathological values like subnormals.
fuzz_target!(|data: (u8, &[u8])| {
    let (n_byte, bytes) = data;
    // Cap n at 32 so each iteration stays under the 10s libfuzzer budget.
    let n = (n_byte as usize) % 33;
    if n == 0 {
        return;
    }
    let need = n * n * 8;
    if bytes.len() < need {
        return;
    }
    let mut a = Vec::with_capacity(n * n);
    for i in 0..(n * n) {
        let chunk = &bytes[i * 8..(i + 1) * 8];
        let bits = u64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3],
            chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        a.push(f64::from_bits(bits));
    }
    let _ = fnp_linalg::cholesky_nxn(&a, n);
});
