#![no_main]

use libfuzzer_sys::fuzz_target;

// Feed arbitrary entropy slices into SeedSequence::new. The constructor
// rejects malformed inputs with SeedSequenceError; the contract is that
// it must always return Ok(_) or Err(_) — never panic.
fuzz_target!(|data: &[u8]| {
    // Cap input size so each iteration stays under the libfuzzer 10s budget.
    if data.len() > 4096 {
        return;
    }
    // Reinterpret the byte slice as a u32 entropy buffer (round down to
    // u32 boundary). Empty entropy is a legitimate edge case for the parser.
    let entropy: Vec<u32> = data
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let _ = fnp_random::SeedSequence::new(&entropy);
});
