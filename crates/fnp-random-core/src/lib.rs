//! Dependency-free NumPy-compatible random primitives.
//!
//! This crate deliberately contains only deterministic state transforms. It
//! performs no I/O, reads no host entropy, starts no threads, and allocates
//! only caller-requested seed/output storage. Higher-level distributions and
//! optional acceleration remain in `fnp-random`.

#![forbid(unsafe_code)]

const XSHIFT: u32 = 16;
const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const POOL_SIZE: usize = 4;
const CHEAP_MULTIPLIER: u64 = 0xda94_2042_e4dd_58b5;
const DEFAULT_MULTIPLIER: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

/// NumPy's default `SeedSequence` pool width in 32-bit words.
pub const DEFAULT_SEED_SEQUENCE_POOL_SIZE: usize = POOL_SIZE;

/// NumPy-compatible `SeedSequence` specialized to the canonical four-word pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeedSequence {
    pool: [u32; POOL_SIZE],
}

impl SeedSequence {
    /// Construct `SeedSequence(seed)` from an unsigned 64-bit integer.
    #[must_use]
    pub fn from_seed(seed: u64) -> Self {
        Self::assemble(&entropy_words(seed), &[])
    }

    /// Construct `SeedSequence(entropy=seed, spawn_key=spawn_key)`.
    #[must_use]
    pub fn with_spawn_key(seed: u64, spawn_key: &[u32]) -> Self {
        Self::assemble(&entropy_words(seed), spawn_key)
    }

    /// Construct from NumPy-style little-endian entropy words.
    ///
    /// An empty entropy slice is normalized to the single word zero, matching
    /// integer seed zero rather than creating an unseeded host-entropy path.
    #[must_use]
    pub fn from_entropy_words(entropy: &[u32], spawn_key: &[u32]) -> Self {
        if entropy.is_empty() {
            Self::assemble(&[0], spawn_key)
        } else {
            Self::assemble(entropy, spawn_key)
        }
    }

    fn assemble(entropy: &[u32], spawn_key: &[u32]) -> Self {
        let mut assembled = entropy.to_vec();
        if !spawn_key.is_empty() && assembled.len() < POOL_SIZE {
            assembled.resize(POOL_SIZE, 0);
        }
        assembled.extend_from_slice(spawn_key);

        let mut pool = [0_u32; POOL_SIZE];
        let mut hash_const = INIT_A;
        for (index, slot) in pool.iter_mut().enumerate() {
            *slot = hashmix(assembled.get(index).copied().unwrap_or(0), &mut hash_const);
        }
        for source in 0..POOL_SIZE {
            for destination in 0..POOL_SIZE {
                if source != destination {
                    let hashed = hashmix(pool[source], &mut hash_const);
                    pool[destination] = mix(pool[destination], hashed);
                }
            }
        }
        for &extra in assembled.iter().skip(POOL_SIZE) {
            for slot in &mut pool {
                let hashed = hashmix(extra, &mut hash_const);
                *slot = mix(*slot, hashed);
            }
        }
        Self { pool }
    }

    /// Generate NumPy-compatible 32-bit state words.
    #[must_use]
    pub fn generate_state(&self, word_count: usize) -> Vec<u32> {
        let mut output = Vec::with_capacity(word_count);
        let mut hash_const = INIT_B;
        for index in 0..word_count {
            let mut value = self.pool[index % POOL_SIZE];
            value ^= hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            value = value.wrapping_mul(hash_const);
            value ^= value >> XSHIFT;
            output.push(value);
        }
        output
    }

    /// Generate NumPy-compatible 64-bit state words, pairing low word first.
    #[must_use]
    pub fn generate_state_u64(&self, word_count: usize) -> Vec<u64> {
        let state_word_count = word_count
            .checked_mul(2)
            .expect("SeedSequence u64 output word count overflow");
        let words = self.generate_state(state_word_count);
        words
            .chunks_exact(2)
            .map(|pair| u64::from(pair[0]) | (u64::from(pair[1]) << 32))
            .collect()
    }
}

/// NumPy-compatible `PCG64DXSM` bit generator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pcg64Dxsm {
    state: u128,
    increment: u128,
}

/// Compatibility spelling used by the full `fnp-random` crate.
pub type Pcg64DxsmRng = Pcg64Dxsm;

impl Pcg64Dxsm {
    /// Seed from a NumPy-compatible `SeedSequence`.
    #[must_use]
    pub fn from_seed_sequence(sequence: &SeedSequence) -> Self {
        let words = sequence.generate_state_u64(4);
        let initial_state = (u128::from(words[0]) << 64) | u128::from(words[1]);
        let initial_sequence = (u128::from(words[2]) << 64) | u128::from(words[3]);
        Self::seed(initial_state, initial_sequence)
    }

    /// Construct exactly as `numpy.random.PCG64DXSM(seed)`.
    #[must_use]
    pub fn from_seed(seed: u64) -> Self {
        Self::from_seed_sequence(&SeedSequence::from_seed(seed))
    }

    /// Apply the standard PCG set-sequence seeding dance.
    #[must_use]
    pub fn seed(initial_state: u128, initial_sequence: u128) -> Self {
        let increment = (initial_sequence << 1) | 1;
        let mut generator = Self {
            state: 0,
            increment,
        };
        generator.seed_step();
        generator.state = generator.state.wrapping_add(initial_state);
        generator.seed_step();
        generator
    }

    /// Restore an exact raw state without running the seeding transform.
    #[must_use]
    pub const fn from_raw_state(state: u128, increment: u128) -> Self {
        Self { state, increment }
    }

    /// Return the exact `(state, increment)` pair.
    #[must_use]
    pub const fn raw_state(&self) -> (u128, u128) {
        (self.state, self.increment)
    }

    /// Return the split state form used by portable journals and manifests.
    #[must_use]
    pub const fn split_state(&self) -> ([u64; 2], [u64; 2]) {
        (
            [(self.state >> 64) as u64, self.state as u64],
            [
                (self.increment >> 64) as u64,
                self.increment as u64,
            ],
        )
    }

    /// Restore a state emitted by [`Self::split_state`].
    #[must_use]
    pub const fn from_split_state(state: [u64; 2], increment: [u64; 2]) -> Self {
        Self {
            state: (state[0] as u128) << 64 | state[1] as u128,
            increment: (increment[0] as u128) << 64 | increment[1] as u128,
        }
    }

    fn seed_step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(DEFAULT_MULTIPLIER)
            .wrapping_add(self.increment);
    }

    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(u128::from(CHEAP_MULTIPLIER))
            .wrapping_add(self.increment);
    }

    /// Generate the next raw 64-bit DXSM word, then advance the state.
    pub fn next_u64(&mut self) -> u64 {
        let low = (self.state as u64) | 1;
        let mut high = (self.state >> 64) as u64;
        high ^= high >> 32;
        high = high.wrapping_mul(CHEAP_MULTIPLIER);
        high ^= high >> 48;
        high = high.wrapping_mul(low);
        self.step();
        high
    }

    /// Generate NumPy's 53-bit `random()` value in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

fn entropy_words(seed: u64) -> Vec<u32> {
    if seed == 0 {
        return vec![0];
    }
    let mut words = Vec::with_capacity(2);
    let mut value = seed;
    while value != 0 {
        words.push(value as u32);
        value >>= 32;
    }
    words
}

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut mixed = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    mixed = mixed.wrapping_mul(*hash_const);
    mixed ^ (mixed >> XSHIFT)
}

fn mix(left: u32, right: u32) -> u32 {
    let mut mixed = left
        .wrapping_mul(MIX_MULT_L)
        .wrapping_sub(right.wrapping_mul(MIX_MULT_R));
    mixed ^= mixed >> XSHIFT;
    mixed
}

#[cfg(test)]
mod tests {
    use super::{Pcg64Dxsm, SeedSequence};

    #[test]
    fn numpy_seed_sequence_vectors_are_exact() {
        assert_eq!(
            SeedSequence::from_seed(0).generate_state(8),
            [
                0xb0f4_78be,
                0xdb2c_d7e7,
                0x2c71_ba49,
                0xabf4_641a,
                0x9d7b_8d41,
                0x20c6_ed6d,
                0x223c_39d4,
                0x2c40_99de,
            ]
        );
        assert_eq!(
            SeedSequence::from_seed(42).generate_state(8),
            [
                0xcd54_0ab7,
                0x9f1e_2e6d,
                0x79fb_94b6,
                0xd578_73dc,
                0x64d4_20b7,
                0x7d28_2a1b,
                0x4692_d5ff,
                0x3365_7971,
            ]
        );
    }

    #[test]
    fn numpy_pcg64dxsm_vectors_are_exact() {
        let mut generator = Pcg64Dxsm::from_seed(42);
        let actual: Vec<u64> = (0..8).map(|_| generator.next_u64()).collect();
        assert_eq!(
            actual,
            [
                0xab1c_5033_8e63_481d,
                0x01bd_f91d_548d_1872,
                0xa872_905d_0418_d0a1,
                0x5f0a_8427_0b80_eabc,
                0x34e8_2505_4db5_f685,
                0x319f_f93c_b20c_b433,
                0xc24f_b90e_b5d6_26af,
                0xf1c7_6bf8_e2e9_99a6,
            ]
        );
    }

    #[test]
    fn split_state_round_trip_preserves_the_next_draw() {
        let mut generator = Pcg64Dxsm::from_seed(u64::MAX);
        for _ in 0..257 {
            let _ = generator.next_u64();
        }
        let (state, increment) = generator.split_state();
        let mut restored = Pcg64Dxsm::from_split_state(state, increment);
        assert_eq!(generator.next_u64(), restored.next_u64());
        assert_eq!(generator.raw_state(), restored.raw_state());
    }

    #[test]
    fn spawn_keys_are_position_sensitive_and_repeatable() {
        let left = SeedSequence::with_spawn_key(7, &[1, 2, 3]).generate_state(16);
        let repeated = SeedSequence::with_spawn_key(7, &[1, 2, 3]).generate_state(16);
        let reordered = SeedSequence::with_spawn_key(7, &[3, 2, 1]).generate_state(16);
        assert_eq!(left, repeated);
        assert_ne!(left, reordered);
    }

    #[test]
    fn empty_entropy_words_mean_explicit_zero_not_host_entropy() {
        assert_eq!(
            SeedSequence::from_entropy_words(&[], &[]),
            SeedSequence::from_seed(0)
        );
    }
}
