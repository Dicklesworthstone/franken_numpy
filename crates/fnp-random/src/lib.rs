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
    SeedSequence(SeedSequence),
    BitGenerator(BitGenerator),
    Generator(Generator),
    RandomState(RandomState),
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
pub enum RandomPolicyError {
    UnknownMetadata,
}

impl RandomPolicyError {
    #[must_use]
    pub const fn reason_code(self) -> &'static str {
        match self {
            Self::UnknownMetadata => "rng_policy_unknown_metadata",
        }
    }
}

impl std::fmt::Display for RandomPolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownMetadata => {
                write!(f, "unknown mode/class metadata rejected fail-closed")
            }
        }
    }
}

impl std::error::Error for RandomPolicyError {}

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
    SpawnContractViolation(&'static str),
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
            Self::SpawnContractViolation(_) => "rng_seedsequence_spawn_contract_violation",
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
            | Self::SpawnContractViolation(msg)
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

    pub fn generate_state_u64(&self, words: usize) -> Result<Vec<u64>, SeedSequenceError> {
        let doubled_words = words
            .checked_mul(2)
            .ok_or(SeedSequenceError::GenerateStateContractViolation)?;
        if doubled_words > MAX_SEED_SEQUENCE_WORDS {
            return Err(SeedSequenceError::GenerateStateContractViolation);
        }
        if words == 0 {
            return Ok(Vec::new());
        }

        let u32_words = self.generate_state_u32(doubled_words)?;
        let mut generated = Vec::with_capacity(words);
        for pair in u32_words.chunks_exact(2) {
            generated.push(u64::from(pair[0]) | (u64::from(pair[1]) << 32));
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
        let mut algorithm_state = None;
        let expected_algorithm_key = algorithm_state_schema_key(self.kind);

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
            if normalized == expected_algorithm_key {
                algorithm_state = Some(*value);
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

        let expected_algorithm_state =
            algorithm_state_schema_value(self.kind, self.seed, self.counter);
        if algorithm_state != Some(expected_algorithm_state) {
            return Err(BitGeneratorError::StateSchemaInvalid(
                "bit-generator algorithm-specific state metadata mismatch",
            ));
        }

        Ok(())
    }
}

fn algorithm_state_schema_key(kind: BitGeneratorKind) -> &'static str {
    match kind {
        BitGeneratorKind::Mt19937 => "mt19937_index",
        BitGeneratorKind::Pcg64 => "pcg64_stream",
        BitGeneratorKind::Philox => "philox_key",
        BitGeneratorKind::Sfc64 => "sfc64_carry",
    }
}

fn algorithm_state_schema_value(kind: BitGeneratorKind, seed: u64, counter: u64) -> u64 {
    match kind {
        BitGeneratorKind::Mt19937 => counter % 624,
        BitGeneratorKind::Pcg64 => splitmix64(seed ^ 0x5043_4736_3400_0001_u64),
        BitGeneratorKind::Philox => seed.rotate_left(17) ^ counter.rotate_right(7),
        BitGeneratorKind::Sfc64 => splitmix64(counter ^ seed ^ 0x5346_4336_3400_0001_u64),
    }
}

fn default_state_schema_entries(
    kind: BitGeneratorKind,
    seed: u64,
    counter: u64,
) -> Vec<(String, u64)> {
    let algorithm_key = algorithm_state_schema_key(kind);
    let algorithm_value = algorithm_state_schema_value(kind, seed, counter);
    vec![
        ("stream_seed".to_string(), seed),
        ("counter".to_string(), counter),
        ("algorithm_tag".to_string(), kind.stream_tag()),
        (
            "schema_version".to_string(),
            u64::from(BIT_GENERATOR_STATE_SCHEMA_VERSION),
        ),
        (algorithm_key.to_string(), algorithm_value),
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
                let rng = deterministic_rng_from_seed_material(material).map_err(|_| {
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
        let rng = rng_from_seed_sequence(seed_sequence).map_err(|_| {
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

    pub fn spawn(&mut self, n_children: usize) -> Result<Vec<Self>, BitGeneratorError> {
        if n_children == 0 || n_children > MAX_SEED_SEQUENCE_CHILDREN {
            return Err(BitGeneratorError::SpawnContractViolation(
                "bit-generator spawn request violated packet-007 child bounds",
            ));
        }

        let n_children_u64 = u64::try_from(n_children).map_err(|_| {
            BitGeneratorError::SpawnContractViolation(
                "bit-generator spawn request exceeded deterministic child budget",
            )
        })?;
        let (parent_seed, parent_counter) = self.raw_state();

        let mut children = Vec::with_capacity(n_children);
        for child_index in 0..n_children {
            let child_index_u64 = u64::try_from(child_index).map_err(|_| {
                BitGeneratorError::SpawnContractViolation(
                    "bit-generator child index exceeded deterministic budget",
                )
            })?;
            let child_ordinal = child_index_u64 + 1;
            let child_seed = splitmix64(
                parent_seed ^ self.kind.stream_tag() ^ child_ordinal.wrapping_mul(GOLDEN_GAMMA),
            );
            let child_counter = parent_counter.wrapping_add(
                child_ordinal.checked_mul(self.kind.jump_stride()).ok_or(
                    BitGeneratorError::SpawnContractViolation(
                        "bit-generator child counter overflowed deterministic budget",
                    ),
                )?,
            );
            children.push(Self {
                kind: self.kind,
                rng: DeterministicRng::from_state(child_seed, child_counter),
            });
        }

        self.rng.jump_ahead(n_children_u64);
        Ok(children)
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

macro_rules! define_algorithm_adapter {
    ($name:ident, $kind:path) => {
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct $name {
            inner: BitGenerator,
        }

        impl $name {
            pub fn new(seed: SeedMaterial) -> Result<Self, BitGeneratorError> {
                Ok(Self {
                    inner: BitGenerator::new($kind, seed)?,
                })
            }

            pub fn from_seed_sequence(
                seed_sequence: &SeedSequence,
            ) -> Result<Self, BitGeneratorError> {
                Ok(Self {
                    inner: BitGenerator::from_seed_sequence($kind, seed_sequence)?,
                })
            }

            #[must_use]
            pub fn as_bit_generator(&self) -> &BitGenerator {
                &self.inner
            }

            #[must_use]
            pub fn as_bit_generator_mut(&mut self) -> &mut BitGenerator {
                &mut self.inner
            }

            #[must_use]
            pub fn into_bit_generator(self) -> BitGenerator {
                self.inner
            }

            #[must_use]
            pub fn next_u64(&mut self) -> u64 {
                self.inner.next_u64()
            }

            #[must_use]
            pub fn next_f64(&mut self) -> f64 {
                self.inner.next_f64()
            }

            pub fn bounded_u64(&mut self, upper_bound: u64) -> Result<u64, RandomError> {
                self.inner.bounded_u64(upper_bound)
            }

            #[must_use]
            pub fn fill_u64(&mut self, len: usize) -> Vec<u64> {
                self.inner.fill_u64(len)
            }

            pub fn jump_in_place(&mut self, jumps: u64) -> Result<(), BitGeneratorError> {
                self.inner.jump_in_place(jumps)
            }

            pub fn jumped(&self, jumps: u64) -> Result<Self, BitGeneratorError> {
                Ok(Self {
                    inner: self.inner.jumped(jumps)?,
                })
            }

            pub fn spawn(&mut self, n_children: usize) -> Result<Vec<Self>, BitGeneratorError> {
                let children = self.inner.spawn(n_children)?;
                Ok(children
                    .into_iter()
                    .map(|inner| Self { inner })
                    .collect::<Vec<_>>())
            }

            #[must_use]
            pub fn state(&self) -> BitGeneratorState {
                self.inner.state()
            }

            pub fn set_state(
                &mut self,
                state: &BitGeneratorState,
            ) -> Result<(), BitGeneratorError> {
                self.inner.set_state(state)
            }
        }
    };
}

define_algorithm_adapter!(Mt19937, BitGeneratorKind::Mt19937);
define_algorithm_adapter!(Pcg64, BitGeneratorKind::Pcg64);
define_algorithm_adapter!(Philox, BitGeneratorKind::Philox);
define_algorithm_adapter!(Sfc64, BitGeneratorKind::Sfc64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RandomState {
    bit_generator: BitGenerator,
}

impl RandomState {
    pub fn new(seed: SeedMaterial) -> Result<Self, BitGeneratorError> {
        let bit_generator = BitGenerator::new(BitGeneratorKind::Mt19937, seed)?;
        Ok(Self { bit_generator })
    }

    #[must_use]
    pub fn from_bit_generator(bit_generator: BitGenerator) -> Self {
        Self { bit_generator }
    }

    #[must_use]
    pub fn bit_generator(&self) -> &BitGenerator {
        &self.bit_generator
    }

    #[must_use]
    pub fn next_u64(&mut self) -> u64 {
        self.bit_generator.next_u64()
    }

    #[must_use]
    pub fn next_f64(&mut self) -> f64 {
        self.bit_generator.next_f64()
    }

    pub fn bounded_u64(&mut self, upper_bound: u64) -> Result<u64, RandomError> {
        self.bit_generator.bounded_u64(upper_bound)
    }

    #[must_use]
    pub fn fill_u64(&mut self, len: usize) -> Vec<u64> {
        self.bit_generator.fill_u64(len)
    }

    pub fn jump_in_place(&mut self, jumps: u64) -> Result<(), BitGeneratorError> {
        self.bit_generator.jump_in_place(jumps)
    }

    pub fn jumped(&self, jumps: u64) -> Result<Self, BitGeneratorError> {
        Ok(Self {
            bit_generator: self.bit_generator.jumped(jumps)?,
        })
    }

    #[must_use]
    pub fn state(&self) -> BitGeneratorState {
        self.bit_generator.state()
    }

    pub fn set_state(&mut self, state: &BitGeneratorState) -> Result<(), BitGeneratorError> {
        self.bit_generator.set_state(state)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generator {
    bit_generator: BitGenerator,
    seed_sequence: Option<SeedSequence>,
}

impl Generator {
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

    #[must_use]
    pub fn next_f64(&mut self) -> f64 {
        self.bit_generator.next_f64()
    }

    pub fn bounded_u64(&mut self, upper_bound: u64) -> Result<u64, RandomError> {
        self.bit_generator.bounded_u64(upper_bound)
    }

    #[must_use]
    pub fn fill_u64(&mut self, len: usize) -> Vec<u64> {
        self.bit_generator.fill_u64(len)
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

    pub fn spawn(&mut self, n_children: usize) -> Result<Vec<Self>, BitGeneratorError> {
        if n_children == 0 || n_children > MAX_SEED_SEQUENCE_CHILDREN {
            return Err(BitGeneratorError::SpawnContractViolation(
                "generator spawn request violated packet-007 child bounds",
            ));
        }

        if let Some(seed_sequence) = self.seed_sequence.as_mut() {
            let child_sequences = seed_sequence.spawn(n_children).map_err(|_| {
                BitGeneratorError::SpawnContractViolation(
                    "generator seed-sequence spawn failed deterministic contract",
                )
            })?;

            let mut children = Vec::with_capacity(child_sequences.len());
            for child_sequence in child_sequences {
                let bit_generator =
                    BitGenerator::from_seed_sequence(self.bit_generator.kind(), &child_sequence)
                        .map_err(|_| {
                            BitGeneratorError::SpawnContractViolation(
                                "generator child bit-generator derivation failed",
                            )
                        })?;
                children.push(Self {
                    bit_generator,
                    seed_sequence: Some(child_sequence),
                });
            }
            return Ok(children);
        }

        let children = self.bit_generator.spawn(n_children)?;
        Ok(children
            .into_iter()
            .map(Self::from_bit_generator)
            .collect::<Vec<_>>())
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

    // ── Distribution sampling methods ───────────────────────────────────

    /// Generate an array of uniform random floats in `[0.0, 1.0)`.
    ///
    /// Mimics `rng.random(size)`.
    #[must_use]
    pub fn random(&mut self, size: usize) -> Vec<f64> {
        (0..size).map(|_| self.next_f64()).collect()
    }

    /// Generate uniform random floats in `[low, high)`.
    ///
    /// Mimics `rng.uniform(low, high, size)`.
    #[must_use]
    pub fn uniform(&mut self, low: f64, high: f64, size: usize) -> Vec<f64> {
        let range = high - low;
        (0..size).map(|_| low + self.next_f64() * range).collect()
    }

    /// Generate random integers in `[low, high)`.
    ///
    /// Mimics `rng.integers(low, high, size)`.
    pub fn integers(&mut self, low: i64, high: i64, size: usize) -> Result<Vec<i64>, RandomError> {
        if high <= low {
            return Err(RandomError::InvalidUpperBound);
        }
        let range = (high as u64).wrapping_sub(low as u64);
        let mut result = Vec::with_capacity(size);
        for _ in 0..size {
            let val = (self.bounded_u64(range)? as i64).wrapping_add(low);
            result.push(val);
        }
        Ok(result)
    }

    /// Generate standard normal (Gaussian, mean=0, std=1) samples using Box-Muller.
    ///
    /// Mimics `rng.standard_normal(size)`.
    #[must_use]
    pub fn standard_normal(&mut self, size: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(size);
        let mut spare: Option<f64> = None;
        for _ in 0..size {
            if let Some(val) = spare.take() {
                result.push(val);
            } else {
                // Box-Muller transform
                let u1 = self.next_f64().max(f64::MIN_POSITIVE); // avoid log(0)
                let u2 = self.next_f64();
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f64::consts::PI * u2;
                result.push(r * theta.cos());
                spare = Some(r * theta.sin());
            }
        }
        result
    }

    /// Generate normal (Gaussian) samples with given mean and standard deviation.
    ///
    /// Mimics `rng.normal(loc, scale, size)`.
    #[must_use]
    pub fn normal(&mut self, loc: f64, scale: f64, size: usize) -> Vec<f64> {
        self.standard_normal(size)
            .into_iter()
            .map(|z| loc + scale * z)
            .collect()
    }

    /// Generate exponentially distributed samples.
    ///
    /// Mimics `rng.exponential(scale, size)`.
    #[must_use]
    pub fn exponential(&mut self, scale: f64, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| {
                let u = self.next_f64().max(f64::MIN_POSITIVE);
                -scale * u.ln()
            })
            .collect()
    }

    /// Generate Poisson-distributed samples using the inverse transform method
    /// for small lambda, Knuth's algorithm for moderate lambda.
    ///
    /// Mimics `rng.poisson(lam, size)`.
    #[must_use]
    pub fn poisson(&mut self, lam: f64, size: usize) -> Vec<u64> {
        (0..size)
            .map(|_| {
                // Knuth's algorithm
                let l = (-lam).exp();
                let mut k = 0u64;
                let mut p = 1.0;
                loop {
                    k += 1;
                    p *= self.next_f64();
                    if p <= l {
                        break;
                    }
                }
                k - 1
            })
            .collect()
    }

    /// Generate binomially distributed samples.
    ///
    /// Mimics `rng.binomial(n, p, size)`.
    #[must_use]
    pub fn binomial(&mut self, n: u64, p: f64, size: usize) -> Vec<u64> {
        (0..size)
            .map(|_| {
                let mut successes = 0u64;
                for _ in 0..n {
                    if self.next_f64() < p {
                        successes += 1;
                    }
                }
                successes
            })
            .collect()
    }

    /// Randomly choose elements from a 1-D array, with or without replacement.
    ///
    /// Mimics `rng.choice(a, size, replace)`.
    pub fn choice(
        &mut self,
        a: &[f64],
        size: usize,
        replace: bool,
    ) -> Result<Vec<f64>, RandomError> {
        let n = a.len();
        if !replace && size > n {
            return Err(RandomError::InvalidUpperBound);
        }
        if replace {
            let mut result = Vec::with_capacity(size);
            for _ in 0..size {
                let idx = self.bounded_u64(n as u64)? as usize;
                result.push(a[idx]);
            }
            Ok(result)
        } else {
            // Fisher-Yates shuffle on a copy, then take first `size` elements
            let mut pool = a.to_vec();
            for i in (1..n).rev() {
                let j = self.bounded_u64((i + 1) as u64)? as usize;
                pool.swap(i, j);
            }
            Ok(pool[..size].to_vec())
        }
    }

    /// Shuffle a mutable slice in-place.
    ///
    /// Mimics `rng.shuffle(x)`.
    pub fn shuffle(&mut self, x: &mut [f64]) -> Result<(), RandomError> {
        let n = x.len();
        for i in (1..n).rev() {
            let j = self.bounded_u64((i + 1) as u64)? as usize;
            x.swap(i, j);
        }
        Ok(())
    }

    /// Return a shuffled copy of the input (or a random permutation of integers).
    ///
    /// Mimics `rng.permutation(x)`.
    pub fn permutation(&mut self, x: &[f64]) -> Result<Vec<f64>, RandomError> {
        let mut result = x.to_vec();
        self.shuffle(&mut result)?;
        Ok(result)
    }

    /// Generate a random permutation of integers `[0, n)`.
    ///
    /// Mimics `rng.permutation(n)`.
    pub fn permutation_range(&mut self, n: usize) -> Result<Vec<u64>, RandomError> {
        let mut result: Vec<u64> = (0..n as u64).collect();
        for i in (1..n).rev() {
            let j = self.bounded_u64((i + 1) as u64)? as usize;
            result.swap(i, j);
        }
        Ok(result)
    }

    // ── additional distributions ────────

    /// Gamma distribution using Marsaglia and Tsang's method.
    pub fn gamma(&mut self, shape_param: f64, scale: f64, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| self.sample_gamma(shape_param) * scale)
            .collect()
    }

    fn sample_gamma(&mut self, shape_param: f64) -> f64 {
        if shape_param < 1.0 {
            // For shape < 1, use the fact that gamma(a) = gamma(a+1) * U^(1/a)
            let g = self.sample_gamma(shape_param + 1.0);
            let u = self.next_f64();
            return g * u.powf(1.0 / shape_param);
        }
        // Marsaglia and Tsang's method for shape >= 1
        let d = shape_param - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.sample_standard_normal_single();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.next_f64();
            if u < 1.0 - 0.0331 * x.powi(4) {
                return d * v;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    fn sample_standard_normal_single(&mut self) -> f64 {
        // Box-Muller transform for a single value
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Beta distribution via gamma sampling.
    pub fn beta(&mut self, a: f64, b: f64, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| {
                let x = self.sample_gamma(a);
                let y = self.sample_gamma(b);
                x / (x + y)
            })
            .collect()
    }

    /// Geometric distribution: number of trials until first success.
    pub fn geometric(&mut self, p: f64, size: usize) -> Vec<u64> {
        (0..size)
            .map(|_| {
                let u = self.next_f64();
                (u.ln() / (1.0 - p).ln()).ceil() as u64
            })
            .collect()
    }

    /// Log-normal distribution.
    pub fn lognormal(&mut self, mean: f64, sigma: f64, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| {
                let z = self.sample_standard_normal_single();
                (mean + sigma * z).exp()
            })
            .collect()
    }

    /// Chi-squared distribution with df degrees of freedom.
    pub fn chisquare(&mut self, df: f64, size: usize) -> Vec<f64> {
        // Chi-squared is gamma(df/2, 2)
        self.gamma(df / 2.0, 2.0, size)
    }

    /// Standard Cauchy distribution.
    pub fn standard_cauchy(&mut self, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| {
                let u = self.next_f64();
                (std::f64::consts::PI * (u - 0.5)).tan()
            })
            .collect()
    }

    /// Triangular distribution.
    pub fn triangular(&mut self, left: f64, mode: f64, right: f64, size: usize) -> Vec<f64> {
        let fc = (mode - left) / (right - left);
        (0..size)
            .map(|_| {
                let u = self.next_f64();
                if u < fc {
                    left + ((right - left) * (mode - left) * u).sqrt()
                } else {
                    right - ((right - left) * (right - mode) * (1.0 - u)).sqrt()
                }
            })
            .collect()
    }

    /// Laplace (double exponential) distribution.
    pub fn laplace(&mut self, loc: f64, scale: f64, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| {
                let u = self.next_f64() - 0.5;
                loc - scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
            })
            .collect()
    }

    /// Gumbel distribution.
    pub fn gumbel(&mut self, loc: f64, scale: f64, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| {
                let u = self.next_f64();
                loc - scale * (-(-u.ln()).ln())
            })
            .collect()
    }

    /// Weibull distribution.
    pub fn weibull(&mut self, a: f64, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| {
                let u = self.next_f64();
                (-(-u).ln_1p()).powf(1.0 / a)
            })
            .collect()
    }

    // ── multivariate distributions ────────

    /// Multinomial distribution (np.random.multinomial).
    /// Returns `size` samples, each a vector of length `pvals.len()`.
    /// `n` is the number of trials, `pvals` are probabilities (must sum to ~1).
    pub fn multinomial(&mut self, n: u64, pvals: &[f64], size: usize) -> Vec<Vec<u64>> {
        (0..size)
            .map(|_| {
                let mut result = vec![0u64; pvals.len()];
                let mut remaining = n;
                let mut p_remaining = 1.0;
                for (i, &p) in pvals.iter().enumerate() {
                    if remaining == 0 || i == pvals.len() - 1 {
                        result[i] = remaining;
                        break;
                    }
                    let draws = self.sample_binomial_single(remaining, p / p_remaining);
                    result[i] = draws;
                    remaining -= draws;
                    p_remaining -= p;
                    if p_remaining <= 0.0 {
                        p_remaining = 1e-15;
                    }
                }
                result
            })
            .collect()
    }

    fn sample_binomial_single(&mut self, n: u64, p: f64) -> u64 {
        let p_clamped = p.clamp(0.0, 1.0);
        let mut count = 0u64;
        for _ in 0..n {
            if self.next_f64() < p_clamped {
                count += 1;
            }
        }
        count
    }

    /// Dirichlet distribution (np.random.dirichlet).
    /// Returns `size` samples, each a vector of length `alpha.len()`.
    pub fn dirichlet(&mut self, alpha: &[f64], size: usize) -> Vec<Vec<f64>> {
        (0..size)
            .map(|_| {
                let gamma_samples: Vec<f64> = alpha.iter().map(|&a| self.sample_gamma(a)).collect();
                let sum: f64 = gamma_samples.iter().sum();
                gamma_samples.into_iter().map(|g| g / sum).collect()
            })
            .collect()
    }

    /// Multivariate normal distribution (np.random.multivariate_normal).
    /// Simplified version using diagonal covariance only.
    /// `mean` is the mean vector, `cov_diag` is the diagonal of the covariance matrix.
    pub fn multivariate_normal_diag(
        &mut self,
        mean: &[f64],
        cov_diag: &[f64],
        size: usize,
    ) -> Vec<Vec<f64>> {
        (0..size)
            .map(|_| {
                mean.iter()
                    .zip(cov_diag)
                    .map(|(&m, &v)| m + self.sample_standard_normal_single() * v.sqrt())
                    .collect()
            })
            .collect()
    }

    /// Negative binomial distribution (np.random.negative_binomial).
    /// Number of failures before `n` successes, with success probability `p`.
    pub fn negative_binomial(&mut self, n: f64, p: f64, size: usize) -> Vec<u64> {
        // Gamma-Poisson mixture
        (0..size)
            .map(|_| {
                let gamma_val = self.sample_gamma(n) * (1.0 - p) / p;
                // Poisson with rate gamma_val
                let l = (-gamma_val).exp();
                let mut k = 0u64;
                let mut pp = 1.0;
                loop {
                    k += 1;
                    pp *= self.next_f64();
                    if pp <= l {
                        break;
                    }
                }
                k - 1
            })
            .collect()
    }

    /// F-distribution (Fisher-Snedecor).  Ratio of two scaled chi-squared
    /// variates: (X1/dfnum) / (X2/dfden).
    pub fn f_distribution(&mut self, dfnum: f64, dfden: f64, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| {
                let x1 = self.sample_gamma(dfnum / 2.0) * 2.0 / dfnum;
                let x2 = self.sample_gamma(dfden / 2.0) * 2.0 / dfden;
                x1 / x2
            })
            .collect()
    }

    /// Student's t-distribution with `df` degrees of freedom.
    pub fn standard_t(&mut self, df: f64, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| {
                let z = self.sample_standard_normal_single();
                let chi2 = self.sample_gamma(df / 2.0) * 2.0;
                z / (chi2 / df).sqrt()
            })
            .collect()
    }

    /// Power distribution on [0, 1).  CDF = x^a, inverse-CDF = u^(1/a).
    pub fn power(&mut self, a: f64, size: usize) -> Vec<f64> {
        (0..size).map(|_| self.next_f64().powf(1.0 / a)).collect()
    }

    /// Von Mises circular distribution (Best-Fisher algorithm).
    pub fn vonmises(&mut self, mu: f64, kappa: f64, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| {
                if kappa < 1e-6 {
                    return mu + std::f64::consts::PI * (2.0 * self.next_f64() - 1.0);
                }
                let tau = 1.0 + (1.0 + 4.0 * kappa * kappa).sqrt();
                let rho = (tau - (2.0 * tau).sqrt()) / (2.0 * kappa);
                let r = (1.0 + rho * rho) / (2.0 * rho);
                loop {
                    let u1 = self.next_f64();
                    let z = (std::f64::consts::PI * u1).cos();
                    let f = (1.0 + r * z) / (r + z);
                    let c = kappa * (r - f);
                    let u2 = self.next_f64();
                    if u2 < c * (2.0 - c) || u2 <= c * (-c).exp() {
                        let u3 = self.next_f64();
                        let theta = if u3 > 0.5 { f.acos() } else { -f.acos() };
                        return mu + theta;
                    }
                }
            })
            .collect()
    }

    /// Rayleigh distribution.
    pub fn rayleigh(&mut self, scale: f64, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| {
                let u = self.next_f64();
                scale * (-2.0 * (1.0 - u).ln()).sqrt()
            })
            .collect()
    }

    /// Pareto distribution (Lomax).  CDF = 1 - (1+x)^(-a) for x >= 0.
    pub fn pareto(&mut self, a: f64, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| {
                let u = self.next_f64();
                (1.0 - u).powf(-1.0 / a) - 1.0
            })
            .collect()
    }

    /// Logistic distribution via inverse-CDF.
    pub fn logistic(&mut self, loc: f64, scale: f64, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| {
                let u = self.next_f64();
                loc + scale * (u / (1.0 - u)).ln()
            })
            .collect()
    }

    /// Hypergeometric distribution: draws from a population of `ngood + nbad`
    /// containing `ngood` success states, taking `nsample` draws without replacement.
    /// Uses the direct counting algorithm (draw balls one-by-one).
    pub fn hypergeometric(&mut self, ngood: u64, nbad: u64, nsample: u64, size: usize) -> Vec<u64> {
        (0..size)
            .map(|_| {
                let mut good_remaining = ngood;
                let mut total_remaining = ngood + nbad;
                let mut successes: u64 = 0;
                for _ in 0..nsample {
                    if total_remaining == 0 {
                        break;
                    }
                    let u = self.next_f64();
                    if u < (good_remaining as f64) / (total_remaining as f64) {
                        successes += 1;
                        good_remaining -= 1;
                    }
                    total_remaining -= 1;
                }
                successes
            })
            .collect()
    }

    /// Zipf (Zipfian) distribution with parameter `a > 1`.
    /// Uses rejection method based on Luc Devroye's algorithm.
    pub fn zipf(&mut self, a: f64, size: usize) -> Vec<f64> {
        let b = 2.0_f64.powf(a - 1.0);
        (0..size)
            .map(|_| {
                loop {
                    let u = 1.0 - self.next_f64();
                    let v = self.next_f64();
                    let x = (u.powf(-1.0 / (a - 1.0))).floor();
                    let t = ((1.0 + 1.0 / x).powf(a - 1.0)) / b;
                    if v * x * (t - 1.0) / (b - 1.0) <= t / b {
                        return x;
                    }
                }
            })
            .collect()
    }

    /// Wald (inverse Gaussian) distribution with given `mean` and `scale`.
    /// Uses the Michael-Schucany-Haas algorithm.
    pub fn wald(&mut self, mean: f64, scale: f64, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| {
                let z = self.sample_standard_normal_single();
                let v = z * z;
                let mu = mean;
                let lam = scale;
                let x = mu + (mu * mu * v) / (2.0 * lam)
                    - (mu / (2.0 * lam)) * (4.0 * mu * lam * v + mu * mu * v * v).sqrt();
                let u = self.next_f64();
                if u <= mu / (mu + x) { x } else { mu * mu / x }
            })
            .collect()
    }

    /// Logarithmic (log-series) distribution with parameter `p` in (0, 1).
    /// Uses Kemp's algorithm.
    pub fn logseries(&mut self, p: f64, size: usize) -> Vec<u64> {
        let log_q = (1.0 - p).ln();
        (0..size)
            .map(|_| {
                let u = self.next_f64();
                let v = self.next_f64();
                if u < -p / ((1.0 - p) * log_q) {
                    1
                } else {
                    let y = 1.0 - (v.ln() / log_q).exp().min(1.0);
                    if y <= 0.0 {
                        1
                    } else {
                        1 + (v.ln() / y.ln()).floor() as u64
                    }
                }
            })
            .collect()
    }

    /// Full multivariate normal with arbitrary covariance matrix.
    /// Uses Cholesky decomposition: if Sigma = L L^T, then
    /// X = mean + L * z where z ~ N(0, I).
    /// `cov` is a flat n*n covariance matrix (row-major), must be symmetric positive definite.
    pub fn multivariate_normal(&mut self, mean: &[f64], cov: &[f64], size: usize) -> Vec<Vec<f64>> {
        let n = mean.len();
        let l = cholesky_flat(cov, n);
        (0..size)
            .map(|_| {
                let z: Vec<f64> = (0..n)
                    .map(|_| self.sample_standard_normal_single())
                    .collect();
                let mut sample = mean.to_vec();
                for i in 0..n {
                    for j in 0..=i {
                        sample[i] += l[i * n + j] * z[j];
                    }
                }
                sample
            })
            .collect()
    }
}

/// Cholesky decomposition for multivariate_normal: A = L L^T.
/// Returns L as flat n*n row-major. Assumes A is symmetric PD.
fn cholesky_flat(a: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let diag = a[i * n + i] - sum;
                l[i * n + j] = if diag > 0.0 { diag.sqrt() } else { 0.0 };
            } else {
                let denom = l[j * n + j];
                l[i * n + j] = if denom != 0.0 {
                    (a[i * n + j] - sum) / denom
                } else {
                    0.0
                };
            }
        }
    }
    l
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

fn deterministic_rng_from_seed_material(
    seed: SeedMaterial,
) -> Result<DeterministicRng, RngConstructorError> {
    match seed {
        SeedMaterial::None => Ok(DeterministicRng::new(DEFAULT_RNG_SEED)),
        SeedMaterial::U64(value) => Ok(DeterministicRng::new(value)),
        SeedMaterial::U32Words(words) => {
            if words.is_empty() {
                return Err(RngConstructorError::SeedMetadataInvalid);
            }
            Ok(DeterministicRng::new(seed_material_to_u64(&words)))
        }
        SeedMaterial::SeedSequence(seed_sequence) => rng_from_seed_sequence(&seed_sequence)
            .map_err(|_| RngConstructorError::SeedMetadataInvalid),
        SeedMaterial::BitGenerator(bit_generator) => {
            let (seed, counter) = bit_generator.raw_state();
            Ok(DeterministicRng::from_state(seed, counter))
        }
        SeedMaterial::Generator(generator) => {
            let (seed, counter) = generator.bit_generator().raw_state();
            Ok(DeterministicRng::from_state(seed, counter))
        }
        SeedMaterial::RandomState(random_state) => {
            let (seed, counter) = random_state.bit_generator().raw_state();
            Ok(DeterministicRng::from_state(seed, counter))
        }
        SeedMaterial::State { seed, counter } => Ok(DeterministicRng::from_state(seed, counter)),
    }
}

fn rng_from_seed_sequence(
    seed_sequence: &SeedSequence,
) -> Result<DeterministicRng, SeedSequenceError> {
    let words = seed_sequence.generate_state_u32(2)?;
    let seed = u64::from(words[0]) | (u64::from(words[1]) << 32);
    Ok(DeterministicRng::new(seed))
}

pub fn default_rng(seed: SeedMaterial) -> Result<Generator, RngConstructorError> {
    match seed {
        SeedMaterial::Generator(generator) => Ok(generator),
        SeedMaterial::BitGenerator(bit_generator) => {
            Ok(Generator::from_bit_generator(bit_generator))
        }
        SeedMaterial::RandomState(random_state) => {
            Ok(Generator::from_bit_generator(random_state.bit_generator))
        }
        SeedMaterial::SeedSequence(seed_sequence) => {
            Generator::from_seed_sequence(BitGeneratorKind::Pcg64, &seed_sequence)
                .map_err(|_| RngConstructorError::SeedMetadataInvalid)
        }
        material => {
            let bit_generator = BitGenerator::new(BitGeneratorKind::Pcg64, material)
                .map_err(|_| RngConstructorError::SeedMetadataInvalid)?;
            Ok(Generator::from_bit_generator(bit_generator))
        }
    }
}

pub fn generator_from_seed_sequence(
    seed_sequence: &SeedSequence,
) -> Result<Generator, SeedSequenceError> {
    Generator::from_seed_sequence(BitGeneratorKind::Pcg64, seed_sequence)
        .map_err(|_| SeedSequenceError::GenerateStateContractViolation)
}

pub fn validate_rng_policy_metadata(mode: &str, class: &str) -> Result<(), RandomPolicyError> {
    let known_mode = mode == "strict" || mode == "hardened";
    let known_class = class == "known_compatible_low_risk"
        || class == "known_compatible_high_risk"
        || class == "known_incompatible_semantics"
        || class == "unknown_semantics";

    if !known_mode || !known_class {
        return Err(RandomPolicyError::UnknownMetadata);
    }

    Ok(())
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
        DEFAULT_RNG_SEED, DeterministicRng, Generator, GeneratorPicklePayload,
        MAX_RNG_JUMP_OPERATIONS, MAX_SEED_SEQUENCE_CHILDREN, MAX_SEED_SEQUENCE_WORDS, Mt19937,
        Pcg64, Philox, RANDOM_PACKET_REASON_CODES, RNG_CORE_REASON_CODES, RandomError,
        RandomLogRecord, RandomPolicyError, RandomRuntimeMode, RandomState, SeedMaterial,
        SeedSequence, SeedSequenceError, SeedSequenceSnapshot, Sfc64, default_rng,
        generator_from_seed_sequence, validate_rng_policy_metadata,
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

        let sequence = SeedSequence::new(&[3, 1, 4, 1, 5]).expect("seed sequence");
        let mut from_sequence_material =
            default_rng(SeedMaterial::SeedSequence(sequence.clone())).expect("seed-sequence seed");
        let mut from_sequence_expected =
            generator_from_seed_sequence(&sequence).expect("seed-sequence expectation");
        for _ in 0..64 {
            assert_eq!(
                from_sequence_material.next_u64(),
                from_sequence_expected.next_u64()
            );
        }

        let mut seeded_bit_generator =
            BitGenerator::new(BitGeneratorKind::Pcg64, SeedMaterial::U64(77)).expect("pcg64");
        for _ in 0..13 {
            let _ = seeded_bit_generator.next_u64();
        }
        let mut from_bit_generator_material =
            default_rng(SeedMaterial::BitGenerator(seeded_bit_generator.clone()))
                .expect("bit-generator seed");
        let mut from_bit_generator_expected =
            Generator::from_bit_generator(seeded_bit_generator.clone());
        for _ in 0..64 {
            assert_eq!(
                from_bit_generator_material.next_u64(),
                from_bit_generator_expected.next_u64()
            );
        }

        let mut generator = Generator::from_bit_generator(seeded_bit_generator.clone());
        generator.jump_in_place(2).expect("jump");
        for _ in 0..7 {
            let _ = generator.next_u64();
        }
        let mut from_generator_material =
            default_rng(SeedMaterial::Generator(generator.clone())).expect("generator seed");
        let mut from_generator_expected = generator.clone();
        for _ in 0..64 {
            assert_eq!(
                from_generator_material.next_u64(),
                from_generator_expected.next_u64()
            );
        }

        let mut random_state =
            RandomState::new(SeedMaterial::U64(12345)).expect("random-state constructor");
        for _ in 0..11 {
            let _ = random_state.next_u64();
        }
        let mut from_random_state_material =
            default_rng(SeedMaterial::RandomState(random_state.clone())).expect("random-state");
        let mut from_random_state_expected =
            Generator::from_bit_generator(random_state.bit_generator().clone());
        for _ in 0..64 {
            assert_eq!(
                from_random_state_material.next_u64(),
                from_random_state_expected.next_u64()
            );
        }
    }

    #[test]
    fn generator_passthrough_methods_match_bit_generator_contracts() {
        let mut bit_generator =
            BitGenerator::new(BitGeneratorKind::Philox, SeedMaterial::U64(0xDEAD_BEEF_u64))
                .expect("bit-generator");
        let mut generator = Generator::from_bit_generator(bit_generator.clone());

        for _ in 0..32 {
            assert_eq!(generator.next_u64(), bit_generator.next_u64());
        }

        assert_eq!(generator.next_f64(), bit_generator.next_f64());
        assert_eq!(
            generator.bounded_u64(97).expect("bounded draw"),
            bit_generator.bounded_u64(97).expect("bounded draw"),
        );
        assert_eq!(generator.fill_u64(16), bit_generator.fill_u64(16));

        let err = generator
            .bounded_u64(0)
            .expect_err("zero upper-bound must fail");
        assert_eq!(err.reason_code(), "random_upper_bound_rejected");
    }

    #[test]
    fn random_state_passthrough_and_state_roundtrip_hold() {
        let mut lhs = RandomState::new(SeedMaterial::U64(2024)).expect("lhs");
        let mut rhs = RandomState::new(SeedMaterial::U64(2024)).expect("rhs");

        for _ in 0..32 {
            assert_eq!(lhs.next_u64(), rhs.next_u64());
        }
        assert_eq!(lhs.next_f64(), rhs.next_f64());
        assert_eq!(
            lhs.bounded_u64(53).expect("bounded draw"),
            rhs.bounded_u64(53).expect("bounded draw"),
        );
        assert_eq!(lhs.fill_u64(9), rhs.fill_u64(9));

        let jumped = lhs.jumped(7).expect("jumped");
        let mut stepped = lhs.clone();
        stepped.jump_in_place(7).expect("stepped");
        let mut jumped = jumped;
        for _ in 0..16 {
            assert_eq!(jumped.next_u64(), stepped.next_u64());
        }

        let state = lhs.state();
        let mut restored = RandomState::new(SeedMaterial::U64(1)).expect("restored");
        restored.set_state(&state).expect("state set");
        for _ in 0..32 {
            assert_eq!(lhs.next_u64(), restored.next_u64());
        }

        let err = restored
            .bounded_u64(0)
            .expect_err("zero upper-bound must fail");
        assert_eq!(err.reason_code(), "random_upper_bound_rejected");
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
    fn seed_sequence_generate_state_u64_matches_u32_pairing_and_bounds() {
        let sequence = SeedSequence::new(&[1, 2, 3, 4]).expect("seed sequence");

        let first = sequence.generate_state_u64(16).expect("u64 words");
        let second = sequence.generate_state_u64(16).expect("u64 words");
        assert_eq!(first, second);

        let via_u32 = sequence.generate_state_u32(32).expect("paired u32 words");
        for (idx, value) in first.iter().copied().enumerate() {
            let lower = u64::from(via_u32[idx * 2]);
            let upper = u64::from(via_u32[idx * 2 + 1]) << 32;
            assert_eq!(value, lower | upper);
        }

        let empty = sequence.generate_state_u64(0).expect("empty generation");
        assert!(empty.is_empty());

        let too_many_words = (MAX_SEED_SEQUENCE_WORDS / 2).saturating_add(1);
        let err = sequence
            .generate_state_u64(too_many_words)
            .expect_err("oversized u64 generation must fail");
        assert_eq!(err.reason_code(), "rng_seedsequence_generate_state_failed");
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
        let state = source.state();
        let mut restored = default_rng(SeedMaterial::State {
            seed: state.seed,
            counter: state.counter,
        })
        .expect("state constructor");
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
    fn typed_algorithm_adapters_expose_deterministic_streams() {
        let mut mt_lhs = Mt19937::new(SeedMaterial::U64(9123)).expect("mt lhs");
        let mut mt_rhs = Mt19937::new(SeedMaterial::U64(9123)).expect("mt rhs");
        for _ in 0..64 {
            assert_eq!(mt_lhs.next_u64(), mt_rhs.next_u64());
        }

        let mut pcg_lhs = Pcg64::new(SeedMaterial::U64(9123)).expect("pcg lhs");
        let mut pcg_rhs = Pcg64::new(SeedMaterial::U64(9123)).expect("pcg rhs");
        for _ in 0..64 {
            assert_eq!(pcg_lhs.next_u64(), pcg_rhs.next_u64());
        }

        let mut philox = Philox::new(SeedMaterial::U64(9123)).expect("philox");
        let mut sfc64 = Sfc64::new(SeedMaterial::U64(9123)).expect("sfc64");
        let diverged = (0..64).any(|_| philox.next_u64() != sfc64.next_u64());
        assert!(diverged);
    }

    #[test]
    fn typed_algorithm_adapter_jump_and_state_roundtrip_hold() {
        let source = Philox::new(SeedMaterial::U64(55)).expect("source");
        let mut jumped = source.jumped(9).expect("jumped");
        let mut stepped = source.clone();
        stepped.jump_in_place(9).expect("stepped");
        for _ in 0..32 {
            assert_eq!(jumped.next_u64(), stepped.next_u64());
        }

        let mut restored = Sfc64::new(SeedMaterial::U64(88)).expect("restored");
        for _ in 0..7 {
            let _ = restored.next_u64();
        }
        let state = restored.state();
        let mut replay = Sfc64::new(SeedMaterial::U64(1)).expect("replay");
        replay.set_state(&state).expect("state set");
        for _ in 0..32 {
            assert_eq!(restored.next_u64(), replay.next_u64());
        }
    }

    #[test]
    fn typed_algorithm_seed_sequence_constructor_matches_kind_constructor() {
        let sequence = SeedSequence::new(&[101, 202, 303]).expect("seed sequence");
        let from_sequence = Mt19937::from_seed_sequence(&sequence).expect("from sequence");
        let from_kind = BitGenerator::from_seed_sequence(BitGeneratorKind::Mt19937, &sequence)
            .expect("from kind");
        assert_eq!(from_sequence.as_bit_generator().state(), from_kind.state());
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
    fn bit_generator_state_schema_rejects_algorithm_specific_mismatches() {
        let mut generator = BitGenerator::new(BitGeneratorKind::Mt19937, SeedMaterial::U64(222))
            .expect("generator");
        let mut state = generator.state();
        for (key, value) in &mut state.schema_entries {
            if key == "mt19937_index" {
                *value = value.wrapping_add(1);
            }
        }
        let mismatch = generator
            .set_state(&state)
            .expect_err("algorithm-specific mismatch must fail");
        assert_eq!(mismatch.reason_code(), "rng_state_schema_invalid");

        let mut missing = generator.state();
        missing
            .schema_entries
            .retain(|(key, _)| key != "mt19937_index");
        let missing_err = generator
            .set_state(&missing)
            .expect_err("missing algorithm-specific key must fail");
        assert_eq!(missing_err.reason_code(), "rng_state_schema_invalid");
    }

    #[test]
    fn bit_generator_spawn_is_deterministic_and_bounded() {
        let mut lhs =
            BitGenerator::new(BitGeneratorKind::Pcg64, SeedMaterial::U64(909)).expect("lhs parent");
        let mut rhs =
            BitGenerator::new(BitGeneratorKind::Pcg64, SeedMaterial::U64(909)).expect("rhs parent");
        let lhs_children = lhs.spawn(3).expect("lhs spawn");
        let rhs_children = rhs.spawn(3).expect("rhs spawn");

        assert_eq!(lhs_children.len(), 3);
        assert_eq!(rhs_children.len(), 3);
        for (lhs_child, rhs_child) in lhs_children.iter().zip(rhs_children.iter()) {
            assert_eq!(lhs_child.state(), rhs_child.state());
        }

        let zero = lhs
            .spawn(0)
            .expect_err("zero-child spawn must fail contract");
        assert_eq!(
            zero.reason_code(),
            "rng_seedsequence_spawn_contract_violation"
        );
        let too_many = lhs
            .spawn(MAX_SEED_SEQUENCE_CHILDREN + 1)
            .expect_err("over-budget spawn must fail contract");
        assert_eq!(
            too_many.reason_code(),
            "rng_seedsequence_spawn_contract_violation"
        );
    }

    #[test]
    fn generator_spawn_preserves_seed_sequence_lineage() {
        let root = SeedSequence::with_spawn_key(&[8, 13, 21], &[34], 8).expect("root");
        let mut lhs =
            Generator::from_seed_sequence(BitGeneratorKind::Mt19937, &root).expect("lhs parent");
        let mut rhs =
            Generator::from_seed_sequence(BitGeneratorKind::Mt19937, &root).expect("rhs parent");

        let lhs_children = lhs.spawn(2).expect("lhs spawn");
        let rhs_children = rhs.spawn(2).expect("rhs spawn");
        for (lhs_child, rhs_child) in lhs_children.iter().zip(rhs_children.iter()) {
            assert_eq!(lhs_child.state(), rhs_child.state());
        }

        let next_batch = lhs.spawn(1).expect("next batch");
        assert_ne!(lhs_children[0].state(), next_batch[0].state());
    }

    #[test]
    fn generator_binding_and_pickle_roundtrip_contracts_hold() {
        let sequence = SeedSequence::with_spawn_key(&[2, 4, 6], &[9], 8).expect("seed sequence");
        let bit_generator = BitGenerator::new(BitGeneratorKind::Mt19937, SeedMaterial::U64(5))
            .expect("mt generator");
        let binding_err = Generator::bind_seed_sequence(bit_generator, &sequence)
            .expect_err("non-derived generator binding must fail closed");
        assert_eq!(binding_err.reason_code(), "rng_generator_binding_invalid");

        let mut facade = Generator::from_seed_sequence(BitGeneratorKind::Philox, &sequence)
            .expect("facade from seed-sequence");
        facade.jump_in_place(3).expect("in-place jump");
        for _ in 0..11 {
            let _ = facade.next_u64();
        }

        let payload = facade.to_pickle_payload();
        let mut restored =
            Generator::from_pickle_payload(payload).expect("pickle payload roundtrip");
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
        let pickle_err = Generator::from_pickle_payload(invalid_payload)
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

        let spawn_err = BitGeneratorError::SpawnContractViolation("spawn mismatch");
        assert_eq!(
            spawn_err.reason_code(),
            "rng_seedsequence_spawn_contract_violation"
        );
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
    fn rng_policy_metadata_validation_is_fail_closed() {
        validate_rng_policy_metadata("strict", "known_compatible_low_risk")
            .expect("known strict metadata");
        validate_rng_policy_metadata("hardened", "unknown_semantics")
            .expect("known hardened metadata");

        let unknown_mode = validate_rng_policy_metadata("mystery", "known_compatible_low_risk")
            .expect_err("unknown mode must fail closed");
        assert_eq!(unknown_mode, RandomPolicyError::UnknownMetadata);
        assert_eq!(unknown_mode.reason_code(), "rng_policy_unknown_metadata");

        let unknown_class = validate_rng_policy_metadata("strict", "alien_class")
            .expect_err("unknown class must fail closed");
        assert_eq!(unknown_class, RandomPolicyError::UnknownMetadata);
        assert_eq!(unknown_class.reason_code(), "rng_policy_unknown_metadata");
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

    // ── Distribution sampling tests ─────────────────────────────────────

    fn test_generator() -> Generator {
        Generator::from_bit_generator(
            BitGenerator::new(BitGeneratorKind::Pcg64, SeedMaterial::U64(42)).unwrap(),
        )
    }

    #[test]
    fn random_in_unit_interval() {
        let mut rng = test_generator();
        let vals = rng.random(100);
        assert_eq!(vals.len(), 100);
        assert!(vals.iter().all(|&v| (0.0..1.0).contains(&v)));
    }

    #[test]
    fn uniform_in_range() {
        let mut rng = test_generator();
        let vals = rng.uniform(2.0, 5.0, 100);
        assert_eq!(vals.len(), 100);
        assert!(vals.iter().all(|&v| (2.0..5.0).contains(&v)));
    }

    #[test]
    fn integers_in_range() {
        let mut rng = test_generator();
        let vals = rng.integers(0, 10, 100).unwrap();
        assert_eq!(vals.len(), 100);
        assert!(vals.iter().all(|&v| (0..10).contains(&v)));
    }

    #[test]
    fn integers_high_le_low_error() {
        let mut rng = test_generator();
        assert!(rng.integers(5, 5, 10).is_err());
    }

    #[test]
    fn standard_normal_basic_stats() {
        let mut rng = test_generator();
        let vals = rng.standard_normal(10000);
        assert_eq!(vals.len(), 10000);
        let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
        // Mean should be close to 0 with 10000 samples
        assert!(mean.abs() < 0.1, "mean was {mean}");
    }

    #[test]
    fn normal_shifted() {
        let mut rng = test_generator();
        let vals = rng.normal(100.0, 1.0, 10000);
        let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
        assert!((mean - 100.0).abs() < 0.5, "mean was {mean}");
    }

    #[test]
    fn exponential_positive() {
        let mut rng = test_generator();
        let vals = rng.exponential(1.0, 100);
        assert_eq!(vals.len(), 100);
        assert!(vals.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn poisson_nonnegative() {
        let mut rng = test_generator();
        let vals = rng.poisson(3.0, 100);
        assert_eq!(vals.len(), 100);
        // Mean should be approximately lambda=3
        let mean: f64 = vals.iter().map(|&v| v as f64).sum::<f64>() / 100.0;
        assert!((mean - 3.0).abs() < 1.0, "mean was {mean}");
    }

    #[test]
    fn binomial_bounded() {
        let mut rng = test_generator();
        let vals = rng.binomial(10, 0.5, 100);
        assert_eq!(vals.len(), 100);
        assert!(vals.iter().all(|&v| v <= 10));
    }

    #[test]
    fn choice_with_replacement() {
        let mut rng = test_generator();
        let pool = [1.0, 2.0, 3.0];
        let vals = rng.choice(&pool, 10, true).unwrap();
        assert_eq!(vals.len(), 10);
        assert!(vals.iter().all(|v| pool.contains(v)));
    }

    #[test]
    fn choice_without_replacement() {
        let mut rng = test_generator();
        let pool = [1.0, 2.0, 3.0, 4.0, 5.0];
        let vals = rng.choice(&pool, 3, false).unwrap();
        assert_eq!(vals.len(), 3);
        // All unique (since no replacement)
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        sorted.dedup();
        assert_eq!(sorted.len(), 3);
    }

    #[test]
    fn choice_without_replacement_too_many() {
        let mut rng = test_generator();
        let pool = [1.0, 2.0];
        assert!(rng.choice(&pool, 5, false).is_err());
    }

    #[test]
    fn shuffle_preserves_elements() {
        let mut rng = test_generator();
        let mut vals = [1.0, 2.0, 3.0, 4.0, 5.0];
        rng.shuffle(&mut vals).unwrap();
        let mut sorted = vals;
        sorted.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(sorted, [1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn permutation_preserves_elements() {
        let mut rng = test_generator();
        let vals = [1.0, 2.0, 3.0, 4.0, 5.0];
        let perm = rng.permutation(&vals).unwrap();
        let mut sorted = perm;
        sorted.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(sorted, [1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn permutation_range_basic() {
        let mut rng = test_generator();
        let perm = rng.permutation_range(5).unwrap();
        assert_eq!(perm.len(), 5);
        let mut sorted = perm;
        sorted.sort();
        assert_eq!(sorted, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn deterministic_same_seed() {
        let mut rng1 = test_generator();
        let mut rng2 = test_generator();
        assert_eq!(rng1.random(10), rng2.random(10));
        assert_eq!(
            rng1.integers(0, 100, 10).unwrap(),
            rng2.integers(0, 100, 10).unwrap()
        );
    }

    // ── additional distribution tests ────────

    #[test]
    fn gamma_basic() {
        let mut rng = test_generator();
        let samples = rng.gamma(2.0, 1.0, 1000);
        assert_eq!(samples.len(), 1000);
        assert!(samples.iter().all(|&v| v > 0.0));
        let mean: f64 = samples.iter().sum::<f64>() / 1000.0;
        assert!((mean - 2.0).abs() < 0.5); // shape * scale = 2
    }

    #[test]
    fn gamma_small_shape() {
        let mut rng = test_generator();
        let samples = rng.gamma(0.5, 1.0, 100);
        assert!(samples.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn beta_basic() {
        let mut rng = test_generator();
        let samples = rng.beta(2.0, 5.0, 1000);
        assert_eq!(samples.len(), 1000);
        assert!(samples.iter().all(|&v| (0.0..=1.0).contains(&v)));
        let mean: f64 = samples.iter().sum::<f64>() / 1000.0;
        // E[Beta(a,b)] = a/(a+b) = 2/7 ≈ 0.286
        assert!((mean - 2.0 / 7.0).abs() < 0.1);
    }

    #[test]
    fn geometric_basic() {
        let mut rng = test_generator();
        let samples = rng.geometric(0.5, 1000);
        assert_eq!(samples.len(), 1000);
        assert!(samples.iter().all(|&v| v >= 1));
    }

    #[test]
    fn lognormal_basic() {
        let mut rng = test_generator();
        let samples = rng.lognormal(0.0, 1.0, 1000);
        assert_eq!(samples.len(), 1000);
        assert!(samples.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn chisquare_basic() {
        let mut rng = test_generator();
        let samples = rng.chisquare(3.0, 1000);
        assert!(samples.iter().all(|&v| v > 0.0));
        let mean: f64 = samples.iter().sum::<f64>() / 1000.0;
        assert!((mean - 3.0).abs() < 0.5); // E[chi2(df)] = df
    }

    #[test]
    fn cauchy_basic() {
        let mut rng = test_generator();
        let samples = rng.standard_cauchy(100);
        assert_eq!(samples.len(), 100);
        // Cauchy has no finite mean, just check we get values
    }

    #[test]
    fn triangular_basic() {
        let mut rng = test_generator();
        let samples = rng.triangular(0.0, 5.0, 10.0, 1000);
        assert!(samples.iter().all(|&v| (0.0..=10.0).contains(&v)));
        let mean: f64 = samples.iter().sum::<f64>() / 1000.0;
        // E[tri(a,c,b)] = (a+b+c)/3 = (0+5+10)/3 = 5.0
        assert!((mean - 5.0).abs() < 0.5);
    }

    #[test]
    fn laplace_basic() {
        let mut rng = test_generator();
        let samples = rng.laplace(0.0, 1.0, 1000);
        let mean: f64 = samples.iter().sum::<f64>() / 1000.0;
        assert!(mean.abs() < 0.3);
    }

    #[test]
    fn gumbel_basic() {
        let mut rng = test_generator();
        let samples = rng.gumbel(0.0, 1.0, 100);
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn weibull_basic() {
        let mut rng = test_generator();
        let samples = rng.weibull(1.5, 1000);
        assert!(samples.iter().all(|&v| v >= 0.0));
    }

    // ── multivariate distribution tests ────────

    #[test]
    fn multinomial_basic() {
        let mut rng = test_generator();
        let samples = rng.multinomial(10, &[0.2, 0.3, 0.5], 5);
        assert_eq!(samples.len(), 5);
        for sample in &samples {
            assert_eq!(sample.len(), 3);
            assert_eq!(sample.iter().sum::<u64>(), 10); // Must sum to n
        }
    }

    #[test]
    fn dirichlet_basic() {
        let mut rng = test_generator();
        let samples = rng.dirichlet(&[1.0, 2.0, 3.0], 10);
        assert_eq!(samples.len(), 10);
        for sample in &samples {
            assert_eq!(sample.len(), 3);
            let sum: f64 = sample.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10); // Must sum to 1
            assert!(sample.iter().all(|&v| v >= 0.0));
        }
    }

    #[test]
    fn multivariate_normal_diag_basic() {
        let mut rng = test_generator();
        let samples = rng.multivariate_normal_diag(&[0.0, 5.0], &[1.0, 4.0], 1000);
        assert_eq!(samples.len(), 1000);
        assert_eq!(samples[0].len(), 2);
        // Check mean is approximately right
        let mean0: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / 1000.0;
        let mean1: f64 = samples.iter().map(|s| s[1]).sum::<f64>() / 1000.0;
        assert!(mean0.abs() < 0.3);
        assert!((mean1 - 5.0).abs() < 0.5);
    }

    #[test]
    fn negative_binomial_basic() {
        let mut rng = test_generator();
        let samples = rng.negative_binomial(5.0, 0.5, 100);
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn f_distribution_positive_and_reasonable_mean() {
        let mut rng = test_generator();
        let samples = rng.f_distribution(5.0, 10.0, 5000);
        assert!(samples.iter().all(|&v| v > 0.0));
        let mean: f64 = samples.iter().sum::<f64>() / 5000.0;
        // theoretical mean = dfden / (dfden - 2) = 10/8 = 1.25
        assert!((mean - 1.25).abs() < 0.3, "f mean={mean}");
    }

    #[test]
    fn standard_t_symmetric_around_zero() {
        let mut rng = test_generator();
        let samples = rng.standard_t(10.0, 5000);
        let mean: f64 = samples.iter().sum::<f64>() / 5000.0;
        assert!(mean.abs() < 0.15, "t mean={mean}");
    }

    #[test]
    fn power_values_in_unit_interval() {
        let mut rng = test_generator();
        let samples = rng.power(2.0, 1000);
        assert!(samples.iter().all(|&v| (0.0..1.0).contains(&v)));
    }

    #[test]
    fn vonmises_centered_on_mu() {
        let mut rng = test_generator();
        let mu = 1.5;
        let samples = rng.vonmises(mu, 5.0, 5000);
        let mean: f64 = samples.iter().sum::<f64>() / 5000.0;
        assert!((mean - mu).abs() < 0.2, "vonmises mean={mean}");
    }

    #[test]
    fn rayleigh_positive_and_expected_mean() {
        let mut rng = test_generator();
        let scale = 2.0;
        let samples = rng.rayleigh(scale, 5000);
        assert!(samples.iter().all(|&v| v > 0.0));
        let mean: f64 = samples.iter().sum::<f64>() / 5000.0;
        let expected = scale * (std::f64::consts::FRAC_PI_2).sqrt();
        assert!(
            (mean - expected).abs() < 0.2,
            "rayleigh mean={mean}, expected={expected}"
        );
    }

    #[test]
    fn pareto_non_negative() {
        let mut rng = test_generator();
        let samples = rng.pareto(3.0, 1000);
        assert!(samples.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn logistic_centered_on_loc() {
        let mut rng = test_generator();
        let loc = 3.0;
        let samples = rng.logistic(loc, 1.0, 5000);
        let mean: f64 = samples.iter().sum::<f64>() / 5000.0;
        assert!((mean - loc).abs() < 0.2, "logistic mean={mean}");
    }

    #[test]
    fn hypergeometric_bounded_by_ngood_and_nsample() {
        let mut rng = test_generator();
        let samples = rng.hypergeometric(10, 20, 8, 5000);
        for &s in &samples {
            assert!(s <= 8, "can't draw more successes than nsample");
            assert!(s <= 10, "can't draw more successes than ngood");
        }
        // Expected mean = nsample * ngood / (ngood+nbad) = 8 * 10/30 ≈ 2.667
        let mean = samples.iter().sum::<u64>() as f64 / 5000.0;
        assert!((mean - 2.667).abs() < 0.2, "hypergeometric mean={mean}");
    }

    #[test]
    fn zipf_values_are_positive_integers() {
        let mut rng = test_generator();
        let samples = rng.zipf(2.0, 5000);
        for &s in &samples {
            assert!(s >= 1.0, "zipf values must be >= 1");
            assert!(
                (s - s.floor()).abs() < 1e-10,
                "zipf values must be integers"
            );
        }
        // Mode should be 1; majority of values should be 1 or 2
        let count_one = samples.iter().filter(|&&x| x == 1.0).count();
        assert!(
            count_one > 1500,
            "zipf(a=2) should produce many 1s, got {count_one}"
        );
    }

    #[test]
    fn wald_positive_and_expected_mean() {
        let mut rng = test_generator();
        let mu = 2.0;
        let samples = rng.wald(mu, 5.0, 5000);
        for &s in &samples {
            assert!(s > 0.0, "wald values must be positive");
        }
        let mean = samples.iter().sum::<f64>() / 5000.0;
        assert!((mean - mu).abs() < 0.3, "wald mean={mean}, expected ~{mu}");
    }

    #[test]
    fn logseries_values_are_positive_integers() {
        let mut rng = test_generator();
        let samples = rng.logseries(0.5, 5000);
        for &s in &samples {
            assert!(s >= 1, "logseries values must be >= 1");
        }
        // With p=0.5, most values should be small (1 or 2)
        let count_one = samples.iter().filter(|&&x| x == 1).count();
        assert!(
            count_one > 1000,
            "logseries(p=0.5) should produce many 1s, got {count_one}"
        );
    }

    #[test]
    fn multivariate_normal_mean_and_covariance() {
        let mut rng = test_generator();
        let mean = [1.0, 2.0];
        // Covariance: [[1, 0.5], [0.5, 1]]
        let cov = [1.0, 0.5, 0.5, 1.0];
        let samples = rng.multivariate_normal(&mean, &cov, 5000);
        assert_eq!(samples.len(), 5000);
        assert_eq!(samples[0].len(), 2);

        // Check empirical mean is close to true mean
        let mean0: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / 5000.0;
        let mean1: f64 = samples.iter().map(|s| s[1]).sum::<f64>() / 5000.0;
        assert!((mean0 - 1.0).abs() < 0.1, "mean0={mean0}");
        assert!((mean1 - 2.0).abs() < 0.1, "mean1={mean1}");

        // Check correlation is positive (due to 0.5 covariance)
        let cov_emp: f64 = samples
            .iter()
            .map(|s| (s[0] - mean0) * (s[1] - mean1))
            .sum::<f64>()
            / 5000.0;
        assert!(cov_emp > 0.2, "empirical cov={cov_emp}, expected positive");
    }
}
