#![forbid(unsafe_code)]

const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX_CONST1: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX_CONST2: u64 = 0x94D0_49BB_1331_11EB;
pub const DEFAULT_RNG_SEED: u64 = 0xC0DE_CAFE_F00D_BAAD;
pub const DEFAULT_SEED_SEQUENCE_POOL_SIZE: usize = 4;
pub const MAX_SEED_SEQUENCE_POOL_SIZE: usize = 256;
pub const MAX_SEED_SEQUENCE_CHILDREN: usize = 4096;
pub const MAX_SEED_SEQUENCE_WORDS: usize = 1_048_576;
pub const MAX_RNG_JUMP_OPERATIONS: u64 = 1024;
pub const MAX_RNG_STATE_SCHEMA_FIELDS: usize = 4096;
pub const BIT_GENERATOR_STATE_SCHEMA_VERSION: u32 = 1;

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
pub enum BitGeneratorKind {
    Mt19937,
    Pcg64,
    Philox,
    Sfc64,
}

impl BitGeneratorKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mt19937 => "mt19937",
            Self::Pcg64 => "pcg64",
            Self::Philox => "philox",
            Self::Sfc64 => "sfc64",
        }
    }

    #[must_use]
    const fn stream_tag(self) -> u64 {
        match self {
            Self::Mt19937 => 0x4D54_3139_3937_u64,
            Self::Pcg64 => 0x5043_4736_3400_u64,
            Self::Philox => 0x5048_494C_4F58_u64,
            Self::Sfc64 => 0x5346_4336_3400_u64,
        }
    }

    #[must_use]
    const fn jump_stride(self) -> u64 {
        match self {
            Self::Mt19937 => 128,
            Self::Pcg64 => 256,
            Self::Philox => 512,
            Self::Sfc64 => 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitGeneratorError {
    GeneratorBindingInvalid(&'static str),
    InitFailed(&'static str),
    JumpContractViolation(&'static str),
    StateSchemaInvalid(&'static str),
    PickleStateMismatch(&'static str),
}

impl BitGeneratorError {
    #[must_use]
    pub const fn reason_code(self) -> &'static str {
        match self {
            Self::GeneratorBindingInvalid(_) => "rng_generator_binding_invalid",
            Self::InitFailed(_) => "rng_bitgenerator_init_failed",
            Self::JumpContractViolation(_) => "rng_jump_contract_violation",
            Self::StateSchemaInvalid(_) => "rng_state_schema_invalid",
            Self::PickleStateMismatch(_) => "rng_pickle_state_mismatch",
        }
    }
}

impl std::fmt::Display for BitGeneratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GeneratorBindingInvalid(msg)
            | Self::InitFailed(msg)
            | Self::JumpContractViolation(msg)
            | Self::StateSchemaInvalid(msg)
            | Self::PickleStateMismatch(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for BitGeneratorError {}

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
        Self::with_spawn_key_and_counter(entropy, spawn_key, pool_size, 0)
    }

    fn with_spawn_key_and_counter(
        entropy: &[u32],
        spawn_key: &[u32],
        pool_size: usize,
        spawn_counter: u64,
    ) -> Result<Self, SeedSequenceError> {
        if entropy.is_empty() || pool_size == 0 || pool_size > MAX_SEED_SEQUENCE_POOL_SIZE {
            return Err(SeedSequenceError::GenerateStateContractViolation);
        }

        Ok(Self {
            entropy: entropy.to_vec(),
            spawn_key: spawn_key.to_vec(),
            pool_size,
            spawn_counter,
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

    #[must_use]
    pub fn snapshot(&self) -> SeedSequenceSnapshot {
        SeedSequenceSnapshot {
            entropy: self.entropy.clone(),
            spawn_key: self.spawn_key.clone(),
            pool_size: self.pool_size,
            spawn_counter: self.spawn_counter,
        }
    }

    pub fn from_snapshot(snapshot: &SeedSequenceSnapshot) -> Result<Self, SeedSequenceError> {
        Self::with_spawn_key_and_counter(
            &snapshot.entropy,
            &snapshot.spawn_key,
            snapshot.pool_size,
            snapshot.spawn_counter,
        )
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
            let idx_u64 = u64::try_from(idx)
                .map_err(|_| SeedSequenceError::GenerateStateContractViolation)?;
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
            children.push(Self::with_spawn_key_and_counter(
                &self.entropy,
                &child_spawn_key,
                self.pool_size,
                0,
            )?);
        }

        self.spawn_counter = end;
        Ok(children)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeedSequenceSnapshot {
    pub entropy: Vec<u32>,
    pub spawn_key: Vec<u32>,
    pub pool_size: usize,
    pub spawn_counter: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitGeneratorState {
    pub kind: BitGeneratorKind,
    pub schema_version: u32,
    pub seed: u64,
    pub counter: u64,
    pub schema_entries: Vec<(String, u64)>,
}

impl BitGeneratorState {
    fn validate(&self) -> Result<(), BitGeneratorError> {
        if self.schema_version != BIT_GENERATOR_STATE_SCHEMA_VERSION {
            return Err(BitGeneratorError::StateSchemaInvalid(
                "bit-generator state schema version is unsupported",
            ));
        }

        if self.schema_entries.is_empty() || self.schema_entries.len() > MAX_RNG_STATE_SCHEMA_FIELDS
        {
            return Err(BitGeneratorError::StateSchemaInvalid(
                "bit-generator state schema entry count is invalid",
            ));
        }

        let mut seen = std::collections::BTreeSet::new();
        let mut stream_seed = None;
        let mut counter = None;
        let mut algorithm_tag = None;
        let mut schema_version = None;

        for (key, value) in &self.schema_entries {
            let normalized = key.trim();
            if normalized.is_empty() {
                return Err(BitGeneratorError::StateSchemaInvalid(
                    "bit-generator state schema key must not be empty",
                ));
            }
            if !seen.insert(normalized.to_string()) {
                return Err(BitGeneratorError::StateSchemaInvalid(
                    "bit-generator state schema keys must be unique",
                ));
            }

            match normalized {
                "stream_seed" => stream_seed = Some(*value),
                "counter" => counter = Some(*value),
                "algorithm_tag" => algorithm_tag = Some(*value),
                "schema_version" => schema_version = Some(*value),
                _ => {}
            }
        }

        if stream_seed != Some(self.seed) || counter != Some(self.counter) {
            return Err(BitGeneratorError::StateSchemaInvalid(
                "bit-generator state schema values do not match encoded state",
            ));
        }

        if algorithm_tag != Some(self.kind.stream_tag()) {
            return Err(BitGeneratorError::StateSchemaInvalid(
                "bit-generator state algorithm tag mismatch",
            ));
        }

        if schema_version != Some(u64::from(self.schema_version)) {
            return Err(BitGeneratorError::StateSchemaInvalid(
                "bit-generator state schema-version metadata mismatch",
            ));
        }

        Ok(())
    }
}

fn default_state_schema_entries(
    kind: BitGeneratorKind,
    seed: u64,
    counter: u64,
) -> Vec<(String, u64)> {
    vec![
        ("stream_seed".to_string(), seed),
        ("counter".to_string(), counter),
        ("algorithm_tag".to_string(), kind.stream_tag()),
        (
            "schema_version".to_string(),
            u64::from(BIT_GENERATOR_STATE_SCHEMA_VERSION),
        ),
    ]
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratorPicklePayload {
    pub bit_generator_state: BitGeneratorState,
    pub seed_sequence: Option<SeedSequenceSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitGenerator {
    kind: BitGeneratorKind,
    rng: DeterministicRng,
}

impl BitGenerator {
    pub fn new(kind: BitGeneratorKind, seed: SeedMaterial) -> Result<Self, BitGeneratorError> {
        let (stream_seed, counter) = match seed {
            SeedMaterial::State { seed, counter } => (seed, counter),
            material => {
                let rng = default_rng(material).map_err(|_| {
                    BitGeneratorError::InitFailed(
                        "bit-generator constructor rejected seed material",
                    )
                })?;
                let (base_seed, base_counter) = rng.state();
                (splitmix64(base_seed ^ kind.stream_tag()), base_counter)
            }
        };
        Ok(Self {
            kind,
            rng: DeterministicRng::from_state(stream_seed, counter),
        })
    }

    pub fn from_seed_sequence(
        kind: BitGeneratorKind,
        seed_sequence: &SeedSequence,
    ) -> Result<Self, BitGeneratorError> {
        let rng = generator_from_seed_sequence(seed_sequence).map_err(|_| {
            BitGeneratorError::InitFailed("bit-generator constructor rejected SeedSequence state")
        })?;
        let (seed, counter) = rng.state();
        Ok(Self {
            kind,
            rng: DeterministicRng::from_state(splitmix64(seed ^ kind.stream_tag()), counter),
        })
    }

    #[must_use]
    pub const fn kind(&self) -> BitGeneratorKind {
        self.kind
    }

    fn raw_state(&self) -> (u64, u64) {
        self.rng.state()
    }

    #[must_use]
    pub fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }

    #[must_use]
    pub fn next_f64(&mut self) -> f64 {
        self.rng.next_f64()
    }

    pub fn bounded_u64(&mut self, upper_bound: u64) -> Result<u64, RandomError> {
        self.rng.bounded_u64(upper_bound)
    }

    #[must_use]
    pub fn fill_u64(&mut self, len: usize) -> Vec<u64> {
        self.rng.fill_u64(len)
    }

    pub fn jump_in_place(&mut self, jumps: u64) -> Result<(), BitGeneratorError> {
        if jumps == 0 || jumps > MAX_RNG_JUMP_OPERATIONS {
            return Err(BitGeneratorError::JumpContractViolation(
                "jump count exceeded bounded packet-007 policy limits",
            ));
        }
        let stride = self.kind.jump_stride();
        let steps = jumps
            .checked_mul(stride)
            .ok_or(BitGeneratorError::JumpContractViolation(
                "jump count overflowed deterministic step budget",
            ))?;
        self.rng.jump_ahead(steps);
        Ok(())
    }

    pub fn jumped(&self, jumps: u64) -> Result<Self, BitGeneratorError> {
        let mut jumped = self.clone();
        jumped.jump_in_place(jumps)?;
        Ok(jumped)
    }

    #[must_use]
    pub fn state(&self) -> BitGeneratorState {
        let (seed, counter) = self.raw_state();
        BitGeneratorState {
            kind: self.kind,
            schema_version: BIT_GENERATOR_STATE_SCHEMA_VERSION,
            seed,
            counter,
            schema_entries: default_state_schema_entries(self.kind, seed, counter),
        }
    }

    pub fn set_state(&mut self, state: &BitGeneratorState) -> Result<(), BitGeneratorError> {
        if state.kind != self.kind {
            return Err(BitGeneratorError::StateSchemaInvalid(
                "bit-generator state kind does not match target algorithm",
            ));
        }
        state.validate()?;
        self.rng = DeterministicRng::from_state(state.seed, state.counter);
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratorFacade {
    bit_generator: BitGenerator,
    seed_sequence: Option<SeedSequence>,
}

impl GeneratorFacade {
    #[must_use]
    pub fn from_bit_generator(bit_generator: BitGenerator) -> Self {
        Self {
            bit_generator,
            seed_sequence: None,
        }
    }

    pub fn from_seed_sequence(
        kind: BitGeneratorKind,
        seed_sequence: &SeedSequence,
    ) -> Result<Self, BitGeneratorError> {
        let bit_generator = BitGenerator::from_seed_sequence(kind, seed_sequence)?;
        Ok(Self {
            bit_generator,
            seed_sequence: Some(seed_sequence.clone()),
        })
    }

    pub fn bind_seed_sequence(
        bit_generator: BitGenerator,
        seed_sequence: &SeedSequence,
    ) -> Result<Self, BitGeneratorError> {
        let expected = BitGenerator::from_seed_sequence(bit_generator.kind, seed_sequence)
            .map_err(|_| {
                BitGeneratorError::GeneratorBindingInvalid(
                    "failed to derive deterministic bit-generator state for binding",
                )
            })?;
        if expected.raw_state().0 != bit_generator.raw_state().0 {
            return Err(BitGeneratorError::GeneratorBindingInvalid(
                "bit-generator and seed-sequence deterministic identities diverge",
            ));
        }
        Ok(Self {
            bit_generator,
            seed_sequence: Some(seed_sequence.clone()),
        })
    }

    #[must_use]
    pub fn bit_generator(&self) -> &BitGenerator {
        &self.bit_generator
    }

    #[must_use]
    pub fn next_u64(&mut self) -> u64 {
        self.bit_generator.next_u64()
    }

    pub fn jump_in_place(&mut self, jumps: u64) -> Result<(), BitGeneratorError> {
        self.bit_generator.jump_in_place(jumps)
    }

    pub fn jumped(&self, jumps: u64) -> Result<Self, BitGeneratorError> {
        Ok(Self {
            bit_generator: self.bit_generator.jumped(jumps)?,
            seed_sequence: self.seed_sequence.clone(),
        })
    }

    #[must_use]
    pub fn state(&self) -> BitGeneratorState {
        self.bit_generator.state()
    }

    pub fn set_state(&mut self, state: &BitGeneratorState) -> Result<(), BitGeneratorError> {
        self.bit_generator.set_state(state)
    }

    #[must_use]
    pub fn to_pickle_payload(&self) -> GeneratorPicklePayload {
        GeneratorPicklePayload {
            bit_generator_state: self.state(),
            seed_sequence: self.seed_sequence.as_ref().map(SeedSequence::snapshot),
        }
    }

    pub fn from_pickle_payload(payload: GeneratorPicklePayload) -> Result<Self, BitGeneratorError> {
        let mut bit_generator = BitGenerator::new(
            payload.bit_generator_state.kind,
            SeedMaterial::State {
                seed: payload.bit_generator_state.seed,
                counter: payload.bit_generator_state.counter,
            },
        )
        .map_err(|_| {
            BitGeneratorError::PickleStateMismatch(
                "pickle payload could not initialize bit-generator state",
            )
        })?;
        bit_generator
            .set_state(&payload.bit_generator_state)
            .map_err(|_| {
                BitGeneratorError::PickleStateMismatch(
                    "pickle payload bit-generator state schema was invalid",
                )
            })?;

        let seed_sequence = match payload.seed_sequence {
            Some(snapshot) => Some(SeedSequence::from_snapshot(&snapshot).map_err(|_| {
                BitGeneratorError::PickleStateMismatch(
                    "pickle payload seed-sequence snapshot was invalid",
                )
            })?),
            None => None,
        };

        if let Some(ref sequence) = seed_sequence {
            return Self::bind_seed_sequence(bit_generator, sequence).map_err(|_| {
                BitGeneratorError::PickleStateMismatch(
                    "pickle payload seed-sequence binding was inconsistent",
                )
            });
        }

        Ok(Self {
            bit_generator,
            seed_sequence,
        })
    }
}

fn seed_material_to_u64(words: &[u32]) -> u64 {
    let mut mixed = splitmix64(DEFAULT_RNG_SEED);
    for (idx, word) in words.iter().copied().enumerate() {
        let idx_u64 = u64::try_from(idx).unwrap_or(u64::MAX);
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
        BIT_GENERATOR_STATE_SCHEMA_VERSION, BitGenerator, BitGeneratorError, BitGeneratorKind,
        DEFAULT_RNG_SEED, DeterministicRng, GeneratorFacade, GeneratorPicklePayload,
        MAX_RNG_JUMP_OPERATIONS, MAX_SEED_SEQUENCE_CHILDREN, RANDOM_PACKET_REASON_CODES,
        RNG_CORE_REASON_CODES, RandomError, RandomLogRecord, RandomRuntimeMode, SeedMaterial,
        SeedSequence, SeedSequenceError, SeedSequenceSnapshot, default_rng,
        generator_from_seed_sequence,
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
        assert_ne!(
            first_children[1].spawn_key(),
            second_children[0].spawn_key()
        );
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
    fn bit_generator_constructor_is_deterministic_per_algorithm() {
        let kinds = [
            BitGeneratorKind::Mt19937,
            BitGeneratorKind::Pcg64,
            BitGeneratorKind::Philox,
            BitGeneratorKind::Sfc64,
        ];

        for kind in kinds {
            let mut lhs = BitGenerator::new(kind, SeedMaterial::U64(0xABCD_0123_u64)).expect("lhs");
            let mut rhs = BitGenerator::new(kind, SeedMaterial::U64(0xABCD_0123_u64)).expect("rhs");
            for _ in 0..64 {
                assert_eq!(lhs.next_u64(), rhs.next_u64());
            }
        }
    }

    #[test]
    fn bit_generator_kinds_do_not_alias_streams_for_same_seed() {
        let mut mt = BitGenerator::new(BitGeneratorKind::Mt19937, SeedMaterial::U64(44))
            .expect("mt generator");
        let mut pcg = BitGenerator::new(BitGeneratorKind::Pcg64, SeedMaterial::U64(44))
            .expect("pcg generator");
        let diverged = (0..32).any(|_| mt.next_u64() != pcg.next_u64());
        assert!(diverged);
    }

    #[test]
    fn bit_generator_jump_and_state_contracts_hold() {
        let source = BitGenerator::new(BitGeneratorKind::Philox, SeedMaterial::U64(71))
            .expect("source generator");
        let mut jumped = source.jumped(17).expect("jumped generator");
        let mut stepped = source.clone();
        stepped.jump_in_place(17).expect("stepped jump");
        for _ in 0..32 {
            assert_eq!(jumped.next_u64(), stepped.next_u64());
        }

        let zero_err = source
            .jumped(0)
            .expect_err("zero jump count must fail contract");
        assert_eq!(zero_err.reason_code(), "rng_jump_contract_violation");
        let too_many = source
            .jumped(MAX_RNG_JUMP_OPERATIONS + 1)
            .expect_err("jump count above packet cap must fail");
        assert_eq!(too_many.reason_code(), "rng_jump_contract_violation");

        let mut advanced = BitGenerator::new(BitGeneratorKind::Sfc64, SeedMaterial::U64(1234))
            .expect("advanced generator");
        for _ in 0..9 {
            let _ = advanced.next_u64();
        }
        let state = advanced.state();
        let mut restored =
            BitGenerator::new(BitGeneratorKind::Sfc64, SeedMaterial::U64(1)).expect("restored");
        restored.set_state(&state).expect("state roundtrip");
        for _ in 0..32 {
            assert_eq!(advanced.next_u64(), restored.next_u64());
        }

        let mut invalid = state.clone();
        invalid.schema_entries.push(("".to_string(), 1));
        let state_err = restored
            .set_state(&invalid)
            .expect_err("empty schema key must be rejected");
        assert_eq!(state_err.reason_code(), "rng_state_schema_invalid");
        assert_eq!(invalid.schema_version, BIT_GENERATOR_STATE_SCHEMA_VERSION);
    }

    #[test]
    fn generator_facade_binding_and_pickle_roundtrip_contracts_hold() {
        let sequence = SeedSequence::with_spawn_key(&[2, 4, 6], &[9], 8).expect("seed sequence");
        let bit_generator = BitGenerator::new(BitGeneratorKind::Mt19937, SeedMaterial::U64(5))
            .expect("mt generator");
        let binding_err = GeneratorFacade::bind_seed_sequence(bit_generator, &sequence)
            .expect_err("non-derived generator binding must fail closed");
        assert_eq!(binding_err.reason_code(), "rng_generator_binding_invalid");

        let mut facade = GeneratorFacade::from_seed_sequence(BitGeneratorKind::Philox, &sequence)
            .expect("facade from seed-sequence");
        facade.jump_in_place(3).expect("in-place jump");
        for _ in 0..11 {
            let _ = facade.next_u64();
        }

        let payload = facade.to_pickle_payload();
        let mut restored =
            GeneratorFacade::from_pickle_payload(payload).expect("pickle payload roundtrip");
        for _ in 0..32 {
            assert_eq!(facade.next_u64(), restored.next_u64());
        }

        let snapshot_state = restored.state();
        let invalid_payload = GeneratorPicklePayload {
            bit_generator_state: snapshot_state,
            seed_sequence: Some(SeedSequenceSnapshot {
                entropy: Vec::new(),
                spawn_key: vec![1],
                pool_size: 4,
                spawn_counter: 0,
            }),
        };
        let pickle_err = GeneratorFacade::from_pickle_payload(invalid_payload)
            .expect_err("invalid seed-sequence snapshot must fail closed");
        assert_eq!(pickle_err.reason_code(), "rng_pickle_state_mismatch");

        let kind_label = restored.bit_generator().kind().as_str();
        assert_eq!(kind_label, "philox");
    }

    #[test]
    fn seed_sequence_snapshot_roundtrip_preserves_spawn_counter() {
        let mut root = SeedSequence::new(&[10, 20, 30]).expect("seed sequence");
        let _ = root.spawn(3).expect("spawn children");
        let snapshot = root.snapshot();
        let restored = SeedSequence::from_snapshot(&snapshot).expect("snapshot restore");
        assert_eq!(restored.spawn_counter(), root.spawn_counter());
        assert_eq!(restored.entropy(), root.entropy());
        assert_eq!(restored.spawn_key(), root.spawn_key());
    }

    #[test]
    fn bit_generator_error_reason_codes_match_contract_rows() {
        let init_err = BitGenerator::new(
            BitGeneratorKind::Mt19937,
            SeedMaterial::U32Words(Vec::new()),
        )
        .expect_err("empty seed material must fail constructor");
        assert_eq!(init_err.reason_code(), "rng_bitgenerator_init_failed");

        let state_err = BitGeneratorError::StateSchemaInvalid("schema mismatch");
        assert_eq!(state_err.reason_code(), "rng_state_schema_invalid");
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
