#![no_main]

use libfuzzer_sys::fuzz_target;

// Verify that the u64 → SeedSequence → BitGenerator pipeline accepts every
// representable u64 seed without panicking. Pcg64Rng and Pcg64DxsmRng share
// the path; covering both catches divergent error handling.
fuzz_target!(|seed: u64| {
    let _ = fnp_random::Pcg64Rng::from_u64_seed(seed);
    let _ = fnp_random::Pcg64DxsmRng::from_u64_seed(seed);
});
