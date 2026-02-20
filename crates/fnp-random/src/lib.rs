#![forbid(unsafe_code)]

const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX_CONST1: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX_CONST2: u64 = 0x94D0_49BB_1331_11EB;
pub const DEFAULT_RNG_SEED: u64 = 0xC0DE_CAFE_F00D_BAAD;
pub const DEFAULT_SEED_SEQUENCE_POOL_SIZE: usize = 4;
pub const MAX_SEED_SEQUENCE_POOL_SIZE: usize = 256;
pub const MAX_SEED_SEQUENCE_CHILDREN: usize = 4096;
pub const MAX_SEED_SEQUENCE_WORDS: usize = 1_048_576;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RandomRuntimeMode {
    Strict,
    Hardened,
}

impl RandomRuntimeMode {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Hardened => "hardened",
        }
    }
}

pub const RANDOM_PACKET_REASON_CODES: [&str; 10] = [
    "random_seed_determinism_contract",
    "random_stream_divergence_contract",
    "random_upper_bound_rejected",
    "random_bounded_output_contract",
    "random_float_range_contract",
    "random_jump_ahead_contract",
    "random_state_restore_contract",
    "random_fill_length_contract",
    "random_structured_log_contract",
    "random_replay_artifact_contract",
];

pub const RNG_CORE_REASON_CODES: [&str; 10] = [
    "rng_constructor_seed_invalid",
    "rng_generator_binding_invalid",
    "rng_seedsequence_generate_state_failed",
    "rng_seedsequence_spawn_contract_violation",
    "rng_bitgenerator_init_failed",
    "rng_jump_contract_violation",
    "rng_state_schema_invalid",
    "rng_pickle_state_mismatch",
    "rng_policy_unknown_metadata",
    "rng_reproducibility_witness_failed",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeedMaterial {
    None,
    U64(u64),
    U32Words(Vec<u32>),
    State { seed: u64, counter: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RngConstructorError {
    SeedMetadataInvalid,
}

impl RngConstructorError {
    #[must_use]
    pub const fn reason_code(self) -> &'static str {
        match self {
            Self::SeedMetadataInvalid => "rng_constructor_seed_invalid",
        }
    }
}

impl std::fmt::Display for RngConstructorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SeedMetadataInvalid => {
                write!(f, "seed material is invalid for deterministic constructor")
            }
        }
    }
}

impl std::error::Error for RngConstructorError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RandomError {
    InvalidUpperBound,
}

impl RandomError {
    #[must_use]
    pub const fn reason_code(self) -> &'static str {
        match self {
            Self::InvalidUpperBound => "random_upper_bound_rejected",
        }
    }
}

impl std::fmt::Display for RandomError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidUpperBound => write!(f, "upper_bound must be > 0"),
        }
    }
}

impl std::error::Error for RandomError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedSequenceError {
    GenerateStateContractViolation,
    SpawnContractViolation,
}

impl SeedSequenceError {
    #[must_use]
    pub const fn reason_code(self) -> &'static str {
        match self {
            Self::GenerateStateContractViolation => "rng_seedsequence_generate_state_failed",
            Self::SpawnContractViolation => "rng_seedsequence_spawn_contract_violation",
        }
    }
}

impl std::fmt::Display for SeedSequenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GenerateStateContractViolation => {
                write!(f, "seed sequence state generation contract violated")
            }
            Self::SpawnContractViolation => write!(f, "seed sequence spawn contract violated"),
        }
    }
}

impl std::error::Error for SeedSequenceError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeterministicRng {
    stream_seed: u64,
    counter: u64,
}

impl DeterministicRng {
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self {
            stream_seed: seed,
            counter: 0,
        }
    }

    #[must_use]
    pub const fn from_state(seed: u64, counter: u64) -> Self {
        Self {
            stream_seed: seed,
            counter,
        }
    }

    #[must_use]
    pub const fn state(self) -> (u64, u64) {
        (self.stream_seed, self.counter)
    }

    pub fn jump_ahead(&mut self, steps: u64) {
        self.counter = self.counter.wrapping_add(steps);
    }

    #[must_use]
    pub fn next_u64(&mut self) -> u64 {
        self.counter = self.counter.wrapping_add(1);
        splitmix64(
            self.stream_seed
                .wrapping_add(self.counter.wrapping_mul(GOLDEN_GAMMA)),
        )
    }

    #[must_use]
    pub fn next_f64(&mut self) -> f64 {
        // Sample the high 53 bits for IEEE754 mantissa precision in [0, 1).
        let sample = self.next_u64() >> 11;
        sample as f64 / (1u64 << 53) as f64
    }

    pub fn bounded_u64(&mut self, upper_bound: u64) -> Result<u64, RandomError> {
        if upper_bound == 0 {
            return Err(RandomError::InvalidUpperBound);
        }

        let threshold = u64::MAX - u64::MAX % upper_bound;

        loop {
            let candidate = self.next_u64();
            if candidate < threshold {
                return Ok(candidate % upper_bound);
            }
        }
    }

    #[must_use]
    pub fn fill_u64(&mut self, len: usize) -> Vec<u64> {
        (0..len).map(|_| self.next_u64()).collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeedSequence {
    entropy: Vec<u32>,
    spawn_key: Vec<u32>,
    pool_size: usize,
    spawn_counter: u64,
}

impl SeedSequence {
    pub fn new(entropy: &[u32]) -> Result<Self, SeedSequenceError> {
        Self::with_spawn_key(entropy, &[], DEFAULT_SEED_SEQUENCE_POOL_SIZE)
    }

    pub fn with_spawn_key(
        entropy: &[u32],
        spawn_key: &[u32],
        pool_size: usize,
    ) -> Result<Self, SeedSequenceError> {
        if entropy.is_empty() || pool_size == 0 || pool_size > MAX_SEED_SEQUENCE_POOL_SIZE {
            return Err(SeedSequenceError::GenerateStateContractViolation);
        }

        Ok(Self {
            entropy: entropy.to_vec(),
            spawn_key: spawn_key.to_vec(),
            pool_size,
            spawn_counter: 0,
        })
    }

    #[must_use]
    pub fn entropy(&self) -> &[u32] {
        &self.entropy
    }

    #[must_use]
    pub fn spawn_key(&self) -> &[u32] {
        &self.spawn_key
    }

    #[must_use]
    pub const fn pool_size(&self) -> usize {
        self.pool_size
    }

    #[must_use]
    pub const fn spawn_counter(&self) -> u64 {
        self.spawn_counter
    }

    pub fn generate_state_u32(&self, words: usize) -> Result<Vec<u32>, SeedSequenceError> {
        if words > MAX_SEED_SEQUENCE_WORDS {
            return Err(SeedSequenceError::GenerateStateContractViolation);
        }
        if words == 0 {
            return Ok(Vec::new());
        }

        let mut state = seed_material_to_u64(&self.entropy);
        state ^= splitmix64(seed_material_to_u64(&self.spawn_key));
        let pool_size_u64 = u64::try_from(self.pool_size)
            .map_err(|_| SeedSequenceError::GenerateStateContractViolation)?;
        state ^= pool_size_u64.wrapping_mul(GOLDEN_GAMMA);

        let mut generated = Vec::with_capacity(words);
        for idx in 0..words {
            let idx_u64 =
                u64::try_from(idx).map_err(|_| SeedSequenceError::GenerateStateContractViolation)?;
            state = splitmix64(state.wrapping_add((idx_u64 + 1).wrapping_mul(GOLDEN_GAMMA)));
            let bytes = state.to_le_bytes();
            generated.push(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
        }

        Ok(generated)
    }

    pub fn spawn(&mut self, n_children: usize) -> Result<Vec<Self>, SeedSequenceError> {
        if n_children == 0 || n_children > MAX_SEED_SEQUENCE_CHILDREN {
            return Err(SeedSequenceError::SpawnContractViolation);
        }

        let n_children_u64 =
            u64::try_from(n_children).map_err(|_| SeedSequenceError::SpawnContractViolation)?;
        let end = self
            .spawn_counter
            .checked_add(n_children_u64)
            .ok_or(SeedSequenceError::SpawnContractViolation)?;

        let mut children = Vec::with_capacity(n_children);
        for child_counter in self.spawn_counter..end {
            let mut child_spawn_key = self.spawn_key.clone();
            let bytes = child_counter.to_le_bytes();
            child_spawn_key.push(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
            child_spawn_key.push(u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]));
            children.push(Self {
                entropy: self.entropy.clone(),
                spawn_key: child_spawn_key,
                pool_size: self.pool_size,
                spawn_counter: 0,
            });
        }

        self.spawn_counter = end;
        Ok(children)
    }
}

fn seed_material_to_u64(words: &[u32]) -> u64 {
    let mut mixed = splitmix64(DEFAULT_RNG_SEED);
    for (idx, word) in words.iter().copied().enumerate() {
        let idx_u64 = match u64::try_from(idx) {
            Ok(value) => value,
            Err(_) => u64::MAX,
        };
        let contribution = u64::from(word).wrapping_add((idx_u64 + 1).wrapping_mul(GOLDEN_GAMMA));
        mixed = splitmix64(mixed ^ contribution);
    }
    mixed
}

pub fn default_rng(seed: SeedMaterial) -> Result<DeterministicRng, RngConstructorError> {
    match seed {
        SeedMaterial::None => Ok(DeterministicRng::new(DEFAULT_RNG_SEED)),
        SeedMaterial::U64(value) => Ok(DeterministicRng::new(value)),
        SeedMaterial::U32Words(words) => {
            if words.is_empty() {
                return Err(RngConstructorError::SeedMetadataInvalid);
            }
            Ok(DeterministicRng::new(seed_material_to_u64(&words)))
        }
        SeedMaterial::State { seed, counter } => Ok(DeterministicRng::from_state(seed, counter)),
    }
}

pub fn generator_from_seed_sequence(
    seed_sequence: &SeedSequence,
) -> Result<DeterministicRng, SeedSequenceError> {
    let words = seed_sequence.generate_state_u32(2)?;
    let seed = u64::from(words[0]) | (u64::from(words[1]) << 32);
    Ok(DeterministicRng::new(seed))
}

#[must_use]
fn splitmix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(MIX_CONST1);
    x ^= x >> 27;
    x = x.wrapping_mul(MIX_CONST2);
    x ^ (x >> 31)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RandomLogRecord {
    pub fixture_id: String,
    pub seed: u64,
    pub mode: RandomRuntimeMode,
    pub env_fingerprint: String,
    pub artifact_refs: Vec<String>,
    pub reason_code: String,
    pub passed: bool,
}

impl RandomLogRecord {
    #[must_use]
    pub fn is_replay_complete(&self) -> bool {
        !self.fixture_id.trim().is_empty()
            && !self.mode.as_str().is_empty()
            && !self.env_fingerprint.trim().is_empty()
            && !self.reason_code.trim().is_empty()
            && !self.artifact_refs.is_empty()
            && self
                .artifact_refs
                .iter()
                .all(|artifact| !artifact.trim().is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_RNG_SEED, DeterministicRng, MAX_SEED_SEQUENCE_CHILDREN, RANDOM_PACKET_REASON_CODES,
        RNG_CORE_REASON_CODES, RandomError, RandomLogRecord, RandomRuntimeMode, SeedMaterial,
        SeedSequence, SeedSequenceError, default_rng, generator_from_seed_sequence,
    };

    fn packet007_artifacts() -> Vec<String> {
        vec![
            "artifacts/phase2c/FNP-P2C-007/fixture_manifest.json".to_string(),
            "artifacts/phase2c/FNP-P2C-007/parity_gate.yaml".to_string(),
        ]
    }

    #[test]
    fn reason_code_registry_matches_packet_contract() {
        assert_eq!(
            RANDOM_PACKET_REASON_CODES,
            [
                "random_seed_determinism_contract",
                "random_stream_divergence_contract",
                "random_upper_bound_rejected",
                "random_bounded_output_contract",
                "random_float_range_contract",
                "random_jump_ahead_contract",
                "random_state_restore_contract",
                "random_fill_length_contract",
                "random_structured_log_contract",
                "random_replay_artifact_contract",
            ]
        );
    }

    #[test]
    fn rng_core_reason_code_registry_matches_packet_contract() {
        assert_eq!(
            RNG_CORE_REASON_CODES,
            [
                "rng_constructor_seed_invalid",
                "rng_generator_binding_invalid",
                "rng_seedsequence_generate_state_failed",
                "rng_seedsequence_spawn_contract_violation",
                "rng_bitgenerator_init_failed",
                "rng_jump_contract_violation",
                "rng_state_schema_invalid",
                "rng_pickle_state_mismatch",
                "rng_policy_unknown_metadata",
                "rng_reproducibility_witness_failed",
            ]
        );
    }

    #[test]
    fn default_rng_constructor_normalizes_seed_material() {
        let mut from_none = default_rng(SeedMaterial::None).expect("default constructor");
        let mut from_default_seed =
            default_rng(SeedMaterial::U64(DEFAULT_RNG_SEED)).expect("explicit default seed");
        for _ in 0..64 {
            assert_eq!(from_none.next_u64(), from_default_seed.next_u64());
        }

        let words = vec![0x1234_5678, 0x90AB_CDEF, 0x4444_9999];
        let mut from_words_first =
            default_rng(SeedMaterial::U32Words(words.clone())).expect("word-seeded");
        let mut from_words_second =
            default_rng(SeedMaterial::U32Words(words)).expect("word-seeded");
        for _ in 0..64 {
            assert_eq!(from_words_first.next_u64(), from_words_second.next_u64());
        }

        let err = default_rng(SeedMaterial::U32Words(Vec::new()))
            .expect_err("empty seed words must fail closed");
        assert_eq!(err.reason_code(), "rng_constructor_seed_invalid");
    }

    #[test]
    fn seed_sequence_generate_state_is_deterministic() {
        let sequence = SeedSequence::new(&[1, 2, 3, 4]).expect("seed sequence");
        let first = sequence.generate_state_u32(32).expect("state words");
        let second = sequence.generate_state_u32(32).expect("state words");
        assert_eq!(first, second);
        assert_eq!(sequence.pool_size(), super::DEFAULT_SEED_SEQUENCE_POOL_SIZE);
    }

    #[test]
    fn seed_sequence_spawn_lineage_is_monotonic() {
        let mut root = SeedSequence::with_spawn_key(&[11, 22, 33], &[7], 8).expect("root");
        let first_children = root.spawn(2).expect("first spawn");
        let second_children = root.spawn(1).expect("second spawn");

        assert_eq!(root.spawn_counter(), 3);
        assert_eq!(first_children.len(), 2);
        assert_eq!(second_children.len(), 1);
        assert_ne!(first_children[0].spawn_key(), first_children[1].spawn_key());
        assert_ne!(first_children[1].spawn_key(), second_children[0].spawn_key());
    }

    #[test]
    fn seed_sequence_spawn_rejects_invalid_requests() {
        let mut root = SeedSequence::new(&[5, 8, 13]).expect("root");
        let zero = root.spawn(0).expect_err("zero-child spawn invalid");
        assert_eq!(
            zero.reason_code(),
            "rng_seedsequence_spawn_contract_violation"
        );

        let too_many = root
            .spawn(MAX_SEED_SEQUENCE_CHILDREN + 1)
            .expect_err("spawn budget exceeded");
        assert_eq!(
            too_many.reason_code(),
            "rng_seedsequence_spawn_contract_violation"
        );
    }

    #[test]
    fn generator_from_seed_sequence_produces_deterministic_stream() {
        let sequence = SeedSequence::new(&[144, 233, 377]).expect("seed sequence");
        let mut first = generator_from_seed_sequence(&sequence).expect("first generator");
        let mut second = generator_from_seed_sequence(&sequence).expect("second generator");
        for _ in 0..64 {
            assert_eq!(first.next_u64(), second.next_u64());
        }
    }

    #[test]
    fn default_rng_state_material_replays_counter_position() {
        let mut source = default_rng(SeedMaterial::U64(42)).expect("source");
        for _ in 0..9 {
            let _ = source.next_u64();
        }
        let (seed, counter) = source.state();
        let mut restored =
            default_rng(SeedMaterial::State { seed, counter }).expect("state constructor");
        for _ in 0..32 {
            assert_eq!(source.next_u64(), restored.next_u64());
        }
    }

    #[test]
    fn same_seed_stream_is_deterministic() {
        let mut lhs = DeterministicRng::new(0xDEAD_BEEF_u64);
        let mut rhs = DeterministicRng::new(0xDEAD_BEEF_u64);

        for _ in 0..128 {
            assert_eq!(lhs.next_u64(), rhs.next_u64());
        }
    }

    #[test]
    fn distinct_seeds_diverge_in_stream() {
        let mut lhs = DeterministicRng::new(11);
        let mut rhs = DeterministicRng::new(17);

        let diverged = (0..64).any(|_| lhs.next_u64() != rhs.next_u64());
        assert!(diverged);
    }

    #[test]
    fn jump_ahead_matches_repeated_advancement() {
        let steps = 1024u64;
        let mut jumped = DeterministicRng::new(42);
        let mut stepped = DeterministicRng::new(42);

        jumped.jump_ahead(steps);
        for _ in 0..steps {
            let _ = stepped.next_u64();
        }

        assert_eq!(jumped.next_u64(), stepped.next_u64());
    }

    #[test]
    fn state_restore_replays_identical_sequence() {
        let mut source = DeterministicRng::new(99);
        let mut prefix = Vec::new();
        for _ in 0..16 {
            prefix.push(source.next_u64());
        }

        let (seed, counter) = source.state();
        let mut restored = DeterministicRng::from_state(seed, counter);
        for _ in 0..32 {
            assert_eq!(source.next_u64(), restored.next_u64());
        }

        assert_eq!(prefix.len(), 16);
    }

    #[test]
    fn next_f64_stays_in_unit_interval() {
        let mut rng = DeterministicRng::new(123456);
        for _ in 0..4096 {
            let sample = rng.next_f64();
            assert!((0.0..1.0).contains(&sample), "sample={sample}");
        }
    }

    #[test]
    fn bounded_u64_rejects_zero_upper_bound() {
        let mut rng = DeterministicRng::new(123);
        let err = rng
            .bounded_u64(0)
            .expect_err("upper bound zero must be rejected");
        assert_eq!(err, RandomError::InvalidUpperBound);
        assert_eq!(err.reason_code(), "random_upper_bound_rejected");
    }

    #[test]
    fn seed_sequence_errors_map_to_contract_reason_codes() {
        let generate_err = SeedSequence::new(&[]).expect_err("empty entropy must fail");
        assert_eq!(
            generate_err,
            SeedSequenceError::GenerateStateContractViolation
        );
        assert_eq!(
            generate_err.reason_code(),
            "rng_seedsequence_generate_state_failed"
        );
    }

    #[test]
    fn bounded_u64_respects_upper_bound_grid() {
        let mut rng = DeterministicRng::new(314159);
        let bounds = [1u64, 2, 3, 5, 7, 16, 257, 1024];

        for upper_bound in bounds {
            for _ in 0..512 {
                let value = rng.bounded_u64(upper_bound).expect("upper bound is valid");
                assert!(value < upper_bound);
            }
        }
    }

    #[test]
    fn fill_u64_respects_requested_length_and_determinism() {
        let mut first = DeterministicRng::new(777);
        let mut second = DeterministicRng::new(777);
        let lhs = first.fill_u64(64);
        let rhs = second.fill_u64(64);
        assert_eq!(lhs.len(), 64);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn packet007_log_record_is_replay_complete() {
        let record = RandomLogRecord {
            fixture_id: "UP-007-seed-determinism".to_string(),
            seed: 7001,
            mode: RandomRuntimeMode::Strict,
            env_fingerprint: "fnp-random-tests".to_string(),
            artifact_refs: packet007_artifacts(),
            reason_code: "random_seed_determinism_contract".to_string(),
            passed: true,
        };
        assert!(record.is_replay_complete());
    }

    #[test]
    fn packet007_log_record_rejects_missing_fields() {
        let record = RandomLogRecord {
            fixture_id: String::new(),
            seed: 7002,
            mode: RandomRuntimeMode::Hardened,
            env_fingerprint: String::new(),
            artifact_refs: Vec::new(),
            reason_code: String::new(),
            passed: false,
        };
        assert!(!record.is_replay_complete());
    }

    #[test]
    fn packet007_reason_codes_round_trip_into_logs() {
        for (idx, reason_code) in RANDOM_PACKET_REASON_CODES.iter().enumerate() {
            let record = RandomLogRecord {
                fixture_id: format!("UP-007-{idx}"),
                seed: 8000 + u64::try_from(idx).expect("small index"),
                mode: RandomRuntimeMode::Strict,
                env_fingerprint: "fnp-random-tests".to_string(),
                artifact_refs: packet007_artifacts(),
                reason_code: (*reason_code).to_string(),
                passed: true,
            };
            assert!(record.is_replay_complete());
            assert_eq!(record.reason_code, *reason_code);
        }
    }
}
