#![forbid(unsafe_code)]

mod ziggurat;

const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX_CONST1: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX_CONST2: u64 = 0x94D0_49BB_1331_11EB;

fn wrap_angle_to_pi(angle: f64) -> f64 {
    (angle + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}

/// Precomputed log(k!) for k = 0..125, matching NumPy's `logfactorial.c`.
/// Values are verbatim from NumPy's lookup table for bit-exact parity.
#[allow(clippy::excessive_precision, clippy::approx_constant)]
const LOGFACT: [f64; 126] = [
    0.0,
    0.0,
    0.69314718055994529,
    1.791759469228055,
    3.1780538303479458,
    4.7874917427820458,
    6.5792512120101012,
    8.5251613610654147,
    10.604602902745251,
    12.801827480081469,
    15.104412573075516,
    17.502307845873887,
    19.987214495661885,
    22.552163853123425,
    25.19122118273868,
    27.89927138384089,
    30.671860106080672,
    33.505073450136891,
    36.395445208033053,
    39.339884187199495,
    42.335616460753485,
    45.380138898476908,
    48.471181351835227,
    51.606675567764377,
    54.784729398112319,
    58.003605222980518,
    61.261701761002001,
    64.557538627006338,
    67.88974313718154,
    71.257038967168015,
    74.658236348830158,
    78.092223553315307,
    81.557959456115043,
    85.054467017581516,
    88.580827542197682,
    92.136175603687093,
    95.719694542143202,
    99.330612454787428,
    102.96819861451381,
    106.63176026064346,
    110.32063971475739,
    114.03421178146171,
    117.77188139974507,
    121.53308151543864,
    125.3172711493569,
    129.12393363912722,
    132.95257503561632,
    136.80272263732635,
    140.67392364823425,
    144.5657439463449,
    148.47776695177302,
    152.40959258449735,
    156.3608363030788,
    160.3311282166309,
    164.32011226319517,
    168.32744544842765,
    172.35279713916279,
    176.39584840699735,
    180.45629141754378,
    184.53382886144948,
    188.6281734236716,
    192.7390472878449,
    196.86618167289001,
    201.00931639928152,
    205.1681994826412,
    209.34258675253685,
    213.53224149456327,
    217.73693411395422,
    221.95644181913033,
    226.1905483237276,
    230.43904356577696,
    234.70172344281826,
    238.97838956183432,
    243.26884900298271,
    247.57291409618688,
    251.89040220972319,
    256.22113555000954,
    260.56494097186322,
    264.92164979855278,
    269.29109765101981,
    273.67312428569369,
    278.06757344036612,
    282.4742926876304,
    286.89313329542699,
    291.32395009427029,
    295.76660135076065,
    300.22094864701415,
    304.68685676566872,
    309.1641935801469,
    313.65282994987905,
    318.1526396202093,
    322.66349912672615,
    327.1852877037752,
    331.71788719692847,
    336.26118197919845,
    340.81505887079902,
    345.37940706226686,
    349.95411804077025,
    354.53908551944079,
    359.1342053695754,
    363.73937555556347,
    368.35449607240474,
    372.97946888568902,
    377.61419787391867,
    382.25858877306001,
    386.91254912321756,
    391.57598821732961,
    396.24881705179155,
    400.93094827891576,
    405.6222961611449,
    410.32277652693733,
    415.03230672824964,
    419.75080559954472,
    424.47819341825709,
    429.21439186665157,
    433.95932399501481,
    438.71291418612117,
    443.47508812091894,
    448.24577274538461,
    453.02489623849613,
    457.81238798127816,
    462.60817852687489,
    467.4121995716082,
    472.22438392698058,
    477.04466549258564,
    481.87297922988796,
];

/// Compute log(k!) matching NumPy's `logfactorial()` in `logfactorial.c`.
/// Uses a lookup table for k <= 125, Stirling series for larger k.
fn logfactorial(k: i64) -> f64 {
    debug_assert!(k >= 0, "logfactorial called with negative k={k}");
    if let Ok(idx) = usize::try_from(k)
        && idx < LOGFACT.len()
    {
        return LOGFACT[idx];
    }
    let kf = k as f64;
    let halfln2pi = 0.9189385332046728;
    (kf + 0.5) * kf.ln() - kf + (halfln2pi + (1.0 / kf) * (1.0 / 12.0 - 1.0 / (360.0 * kf * kf)))
}

/// Log-gamma function matching NumPy's `random_loggam()` in `distributions.c`.
/// Uses the algorithm from SPECFUN by Zhang and Jin (1996).
#[allow(clippy::excessive_precision)]
fn random_loggam(x: f64) -> f64 {
    const A: [f64; 10] = [
        8.333333333333333e-02,
        -2.777777777777778e-03,
        7.936507936507937e-04,
        -5.952380952380952e-04,
        8.417508417508418e-04,
        -1.917526917526918e-03,
        6.410256410256410e-03,
        -2.955065359477124e-02,
        1.796443723688307e-01,
        -1.39243221690590e+00,
    ];

    if x == 1.0 || x == 2.0 {
        return 0.0;
    }

    let n = if x < 7.0 { (7.0 - x) as i64 } else { 0 };
    let mut x0 = x + (n as f64);
    let x2 = (1.0 / x0) * (1.0 / x0);
    let lg2pi = 1.8378770664093453;

    let mut gl0 = A[9];
    for k in (0..=8).rev() {
        gl0 = gl0 * x2 + A[k];
    }
    let mut gl = gl0 / x0 + 0.5 * lg2pi + (x0 - 0.5) * x0.ln() - x0;

    if x < 7.0 {
        for _ in 0..n {
            gl -= (x0 - 1.0).ln();
            x0 -= 1.0;
        }
    }
    gl
}

pub const DEFAULT_RNG_SEED: u64 = 0xC0DE_CAFE_F00D_BAAD;
pub const DEFAULT_SEED_SEQUENCE_POOL_SIZE: usize = 4;
/// Maximum entropy pool size for SeedSequence. NumPy default is 4 words; 256 is generous
/// upper bound to prevent resource exhaustion from malformed seed material.
pub const MAX_SEED_SEQUENCE_POOL_SIZE: usize = 256;
/// Maximum child SeedSequences that can be spawned from a single parent. Prevents
/// unbounded tree expansion in hierarchical seeding workflows.
pub const MAX_SEED_SEQUENCE_CHILDREN: usize = 4096;
/// Maximum words requestable from generate_state(). At 4 bytes per word, this caps a
/// single state initialization at ~4 MB. Defends against malformed state restoration.
pub const MAX_SEED_SEQUENCE_WORDS: usize = 1_048_576;
/// Maximum consecutive jump-ahead operations on a generator. Each jump advances the
/// state by 2^128 steps (PCG64) which is computationally cheap but logically irreversible.
pub const MAX_RNG_JUMP_OPERATIONS: u64 = 1024;
/// Maximum fields in a serialized RNG state schema. Defends against deserialization of
/// malformed state blobs with millions of spurious fields.
pub const MAX_RNG_STATE_SCHEMA_FIELDS: usize = 4096;
/// Schema version for bit-generator state serialization. Incremented when the state
/// layout changes in a backward-incompatible way.
pub const BIT_GENERATOR_STATE_SCHEMA_VERSION: u32 = 1;
/// Maximum trials for direct binomial sampling (before switching to approximation).
/// Set to i64::MAX to match the u64-to-i64 boundary in NumPy's implementation.
const MAX_BINOMIAL_DIRECT_TRIALS: u64 = i64::MAX as u64;

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
    State { seed: u128, counter: u128 },
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
    InvalidParameter,
}

impl RandomError {
    #[must_use]
    pub const fn reason_code(self) -> &'static str {
        match self {
            Self::InvalidUpperBound => "random_upper_bound_rejected",
            Self::InvalidParameter => "random_invalid_parameter",
        }
    }
}

impl std::fmt::Display for RandomError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidUpperBound => write!(f, "upper_bound must be > 0"),
            Self::InvalidParameter => write!(
                f,
                "parameter is out of valid bounds (e.g. negative variance or invalid probability)"
            ),
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
    pub const fn from_state(seed: u128, counter: u128) -> Self {
        Self {
            stream_seed: seed as u64,
            counter: counter as u64,
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

// PCG64-DXSM constants (from Melissa O'Neill's PCG family)
// DEFAULT_MULTIPLIER_128: used during seeding (pcg_setseq_128_srandom_r)
const PCG_DEFAULT_MULTIPLIER_128: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;
// CHEAP_MULTIPLIER: used during generation (pcg_cm_step_r), only 64-bit
const PCG_CHEAP_MULTIPLIER: u64 = 0xda94_2042_e4dd_58b5;

/// NumPy-compatible PCG64-DXSM bit generator.
///
/// Implements the PCG-XSL-DXSM-128/64 variant used by `numpy.random.PCG64DXSM`
/// and `numpy.random.PCG64` (same state, different output for the XSL-RR variant).
/// State is 128-bit with a 128-bit increment (stream selector).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pcg64DxsmRng {
    state: u128,
    inc: u128,
}

impl Pcg64DxsmRng {
    /// Create from a SeedSequence (NumPy-compatible initialization).
    ///
    /// Draws 4 u64 words from the SeedSequence:
    /// - words[0..2] → initstate (big-endian: words[0] is high)
    /// - words[2..4] → initseq (big-endian: words[2] is high)
    ///
    /// Then applies the standard PCG seeding:
    /// ```text
    /// state = 0
    /// inc = (initseq << 1) | 1
    /// state = state * DEFAULT_MULT + inc
    /// state += initstate
    /// state = state * DEFAULT_MULT + inc
    /// ```
    pub fn from_seed_sequence(ss: &SeedSequence) -> Result<Self, SeedSequenceError> {
        let words = ss.generate_state_u64(4)?;
        let initstate = (u128::from(words[0]) << 64) | u128::from(words[1]);
        let initseq = (u128::from(words[2]) << 64) | u128::from(words[3]);
        Ok(Self::seed(initstate, initseq))
    }

    /// Create from a u64 seed by constructing a SeedSequence internally.
    pub fn from_u64_seed(seed: u64) -> Result<Self, SeedSequenceError> {
        // NumPy converts seed to u32 words for SeedSequence
        let entropy = if seed <= u64::from(u32::MAX) {
            vec![seed as u32]
        } else {
            vec![seed as u32, (seed >> 32) as u32]
        };
        let ss = SeedSequence::new(&entropy)?;
        Self::from_seed_sequence(&ss)
    }

    /// Create from explicit 128-bit state and initseq.
    /// Applies the standard PCG seeding procedure.
    #[must_use]
    pub fn seed(initstate: u128, initseq: u128) -> Self {
        let inc = (initseq << 1) | 1;
        let mut state: u128 = 0;
        // First step with DEFAULT_MULTIPLIER
        state = state
            .wrapping_mul(PCG_DEFAULT_MULTIPLIER_128)
            .wrapping_add(inc);
        // Add initstate
        state = state.wrapping_add(initstate);
        // Second step with DEFAULT_MULTIPLIER
        state = state
            .wrapping_mul(PCG_DEFAULT_MULTIPLIER_128)
            .wrapping_add(inc);
        Self { state, inc }
    }

    /// Create from raw state and increment (no seeding, direct construction).
    /// Use this for restoring a previously saved state.
    #[must_use]
    pub const fn from_raw_state(state: u128, inc: u128) -> Self {
        Self { state, inc }
    }

    /// Get the current raw state as (state, inc).
    #[must_use]
    pub fn raw_state(&self) -> (u128, u128) {
        (self.state, self.inc)
    }

    pub fn to_state_entries(&self) -> Vec<(String, u64)> {
        vec![
            ("pcg64_state_hi".to_string(), (self.state >> 64) as u64),
            ("pcg64_state_lo".to_string(), self.state as u64),
            ("pcg64_inc_hi".to_string(), (self.inc >> 64) as u64),
            ("pcg64_inc_lo".to_string(), self.inc as u64),
        ]
    }

    pub fn from_state_entries(entries: &[(String, u64)]) -> Option<Self> {
        let mut hi = None;
        let mut lo = None;
        let mut ihi = None;
        let mut ilo = None;
        for (k, v) in entries {
            match k.trim() {
                "pcg64_state_hi" => hi = Some(*v),
                "pcg64_state_lo" => lo = Some(*v),
                "pcg64_inc_hi" => ihi = Some(*v),
                "pcg64_inc_lo" => ilo = Some(*v),
                _ => {}
            }
        }
        let state = (u128::from(hi?) << 64) | u128::from(lo?);
        let inc = (u128::from(ihi?) << 64) | u128::from(ilo?);
        Some(Self { state, inc })
    }

    /// PCG-CM step: advance state using the cheap multiplier.
    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(u128::from(PCG_CHEAP_MULTIPLIER))
            .wrapping_add(self.inc);
    }

    /// DXSM output function applied to the current state.
    #[must_use]
    fn dxsm_output(&self) -> u64 {
        let mut hi = (self.state >> 64) as u64;
        let lo = (self.state as u64) | 1; // lo |= 1
        hi ^= hi >> 32;
        hi = hi.wrapping_mul(PCG_CHEAP_MULTIPLIER);
        hi ^= hi >> 48;
        hi = hi.wrapping_mul(lo);
        hi
    }

    /// Generate the next random u64.
    /// Outputs DXSM of current state, then advances.
    #[must_use]
    pub fn next_u64(&mut self) -> u64 {
        let output = self.dxsm_output();
        self.step();
        output
    }

    /// Generate the next random f64 in [0, 1).
    /// Uses the high 53 bits for IEEE754 mantissa precision.
    #[must_use]
    pub fn next_f64(&mut self) -> f64 {
        let sample = self.next_u64() >> 11;
        sample as f64 / (1u64 << 53) as f64
    }

    /// Generate a bounded random u64 in [0, upper_bound) using rejection sampling.
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

    /// Fill a vector with `len` random u64 values.
    #[must_use]
    pub fn fill_u64(&mut self, len: usize) -> Vec<u64> {
        (0..len).map(|_| self.next_u64()).collect()
    }

    /// Advance the state by `delta` steps (jump-ahead).
    /// Uses the standard PCG advance formula: O(log(delta)) multiplications.
    pub fn advance(&mut self, delta: u128) {
        self.state = pcg_advance_128(
            self.state,
            delta,
            u128::from(PCG_CHEAP_MULTIPLIER),
            self.inc,
        );
    }
}

/// PCG advance formula: compute state after `delta` steps.
/// state_{n+delta} = mult^delta * state_n + (mult^delta - 1) / (mult - 1) * inc
/// Implemented via repeated squaring in O(log(delta)) time.
fn pcg_advance_128(state: u128, mut delta: u128, mult: u128, inc: u128) -> u128 {
    let mut cur_mult: u128 = mult;
    let mut cur_plus: u128 = inc;
    let mut acc_mult: u128 = 1;
    let mut acc_plus: u128 = 0;

    while delta > 0 {
        if delta & 1 != 0 {
            acc_mult = acc_mult.wrapping_mul(cur_mult);
            acc_plus = acc_plus.wrapping_mul(cur_mult).wrapping_add(cur_plus);
        }
        delta >>= 1;
        if delta > 0 {
            cur_plus = cur_mult.wrapping_add(1).wrapping_mul(cur_plus);
            cur_mult = cur_mult.wrapping_mul(cur_mult);
        }
    }
    acc_mult.wrapping_mul(state).wrapping_add(acc_plus)
}

// ── MT19937 Mersenne Twister ─────────────────────────────────────────────

const MT_N: usize = 624;
const MT_M: usize = 397;
const MT_MATRIX_A: u32 = 0x9908_b0df;
const MT_UPPER_MASK: u32 = 0x8000_0000;
const MT_LOWER_MASK: u32 = 0x7fff_ffff;
const MT_INIT_MULT: u32 = 1_812_433_253;

/// NumPy-compatible Philox 4x64 counter-based PRNG.
///
/// Philox is a cryptographically-secure PRNG that uses 10 rounds of the Philox
/// permutation. It produces the same u64 stream as NumPy's `Philox` generator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhiloxRng {
    ctr: [u64; 4],
    key: [u64; 2],
    buffer: [u64; 4],
    buffer_pos: usize,
}

impl PhiloxRng {
    pub fn from_seed_sequence(ss: &SeedSequence) -> Result<Self, SeedSequenceError> {
        let words = ss.generate_state_u64(2)?;
        let key = [words[0], words[1]];
        Ok(Self::new(key, [0; 4]))
    }

    #[must_use]
    pub fn new(key: [u64; 2], ctr: [u64; 4]) -> Self {
        let mut rng = Self {
            ctr,
            key,
            buffer: [0; 4],
            buffer_pos: 4,
        };
        rng.fill_buffer();
        rng
    }

    fn fill_buffer(&mut self) {
        let mut ctr = self.ctr;
        let mut key = self.key;

        for _ in 0..10 {
            let (hi0, lo0) = mulhilo64(0xD2E7_470E_E14C_6C93, ctr[0]);
            let (hi1, lo1) = mulhilo64(0xCA5A_8263_9512_1157, ctr[2]);
            let next_ctr = [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0];
            ctr = next_ctr;
            key[0] = key[0].wrapping_add(0x9E37_79B9_7F4A_7C15);
            key[1] = key[1].wrapping_add(0xBB67_AE85_84CA_A73B);
        }
        self.buffer = ctr;
        self.buffer_pos = 0;
    }

    #[must_use]
    pub fn next_u64(&mut self) -> u64 {
        if self.buffer_pos >= 4 {
            self.ctr[0] = self.ctr[0].wrapping_add(1);
            if self.ctr[0] == 0 {
                self.ctr[1] = self.ctr[1].wrapping_add(1);
                if self.ctr[1] == 0 {
                    self.ctr[2] = self.ctr[2].wrapping_add(1);
                    if self.ctr[2] == 0 {
                        self.ctr[3] = self.ctr[3].wrapping_add(1);
                    }
                }
            }
            self.fill_buffer();
        }
        let out = self.buffer[self.buffer_pos];
        self.buffer_pos += 1;
        out
    }

    #[must_use]
    pub fn next_f64(&mut self) -> f64 {
        let sample = self.next_u64() >> 11;
        sample as f64 / (1u64 << 53) as f64
    }

    pub fn jump_ahead(&mut self, steps: u64) {
        // Philox jump-ahead simply increments the counter
        let (sum, overflow) = self.ctr[0].overflowing_add(steps);
        self.ctr[0] = sum;
        if overflow {
            self.ctr[1] = self.ctr[1].wrapping_add(1);
            if self.ctr[1] == 0 {
                self.ctr[2] = self.ctr[2].wrapping_add(1);
                if self.ctr[2] == 0 {
                    self.ctr[3] = self.ctr[3].wrapping_add(1);
                }
            }
        }
        self.buffer_pos = 4; // Force refill
    }

    pub fn to_state_entries(&self) -> Vec<(String, u64)> {
        vec![
            ("philox_ctr0".to_string(), self.ctr[0]),
            ("philox_ctr1".to_string(), self.ctr[1]),
            ("philox_ctr2".to_string(), self.ctr[2]),
            ("philox_ctr3".to_string(), self.ctr[3]),
            ("philox_key0".to_string(), self.key[0]),
            ("philox_key1".to_string(), self.key[1]),
            ("philox_buf0".to_string(), self.buffer[0]),
            ("philox_buf1".to_string(), self.buffer[1]),
            ("philox_buf2".to_string(), self.buffer[2]),
            ("philox_buf3".to_string(), self.buffer[3]),
            ("philox_pos".to_string(), self.buffer_pos as u64),
        ]
    }

    pub fn from_state_entries(entries: &[(String, u64)]) -> Option<Self> {
        let mut ctr0 = None;
        let mut ctr1 = None;
        let mut ctr2 = None;
        let mut ctr3 = None;
        let mut key0 = None;
        let mut key1 = None;
        let mut buf0 = None;
        let mut buf1 = None;
        let mut buf2 = None;
        let mut buf3 = None;
        let mut buffer_pos = None;
        for (k, v) in entries {
            match k.trim() {
                "philox_ctr0" => ctr0 = Some(*v),
                "philox_ctr1" => ctr1 = Some(*v),
                "philox_ctr2" => ctr2 = Some(*v),
                "philox_ctr3" => ctr3 = Some(*v),
                "philox_key0" => key0 = Some(*v),
                "philox_key1" => key1 = Some(*v),
                "philox_buf0" => buf0 = Some(*v),
                "philox_buf1" => buf1 = Some(*v),
                "philox_buf2" => buf2 = Some(*v),
                "philox_buf3" => buf3 = Some(*v),
                "philox_pos" => {
                    let pos = usize::try_from(*v).ok()?;
                    if pos > 4 {
                        return None;
                    }
                    buffer_pos = Some(pos);
                }
                _ => {}
            }
        }
        Some(Self {
            ctr: [ctr0?, ctr1?, ctr2?, ctr3?],
            key: [key0?, key1?],
            buffer: [buf0?, buf1?, buf2?, buf3?],
            buffer_pos: buffer_pos?,
        })
    }
}

fn mulhilo64(a: u64, b: u64) -> (u64, u64) {
    let product = u128::from(a) * u128::from(b);
    ((product >> 64) as u64, product as u64)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sfc64Rng {
    s: [u64; 4],
}

impl Sfc64Rng {
    pub fn from_seed_sequence(ss: &SeedSequence) -> Result<Self, SeedSequenceError> {
        let words = ss.generate_state_u64(3)?;
        Ok(Self::seed([words[0], words[1], words[2]]))
    }

    #[must_use]
    pub fn seed(seed: [u64; 3]) -> Self {
        let s = [seed[0], seed[1], seed[2], 1];
        let mut rng = Self { s };
        for _ in 0..12 {
            let _ = rng.next_u64();
        }
        rng
    }

    #[must_use]
    pub fn next_u64(&mut self) -> u64 {
        let tmp = self.s[0].wrapping_add(self.s[1]).wrapping_add(self.s[3]);
        self.s[3] = self.s[3].wrapping_add(1);
        self.s[0] = self.s[1] ^ (self.s[1] >> 11);
        self.s[1] = self.s[2].wrapping_add(self.s[2] << 3);
        self.s[2] = self.s[2].rotate_left(24).wrapping_add(tmp);
        tmp
    }

    #[must_use]
    pub fn next_f64(&mut self) -> f64 {
        let sample = self.next_u64() >> 11;
        sample as f64 / (1u64 << 53) as f64
    }

    pub fn to_state_entries(&self) -> Vec<(String, u64)> {
        vec![
            ("sfc64_s0".to_string(), self.s[0]),
            ("sfc64_s1".to_string(), self.s[1]),
            ("sfc64_s2".to_string(), self.s[2]),
            ("sfc64_s3".to_string(), self.s[3]),
        ]
    }

    pub fn from_state_entries(entries: &[(String, u64)]) -> Option<Self> {
        let mut s0 = None;
        let mut s1 = None;
        let mut s2 = None;
        let mut s3 = None;
        for (k, v) in entries {
            match k.trim() {
                "sfc64_s0" => s0 = Some(*v),
                "sfc64_s1" => s1 = Some(*v),
                "sfc64_s2" => s2 = Some(*v),
                "sfc64_s3" => s3 = Some(*v),
                _ => {}
            }
        }
        Some(Self {
            s: [s0?, s1?, s2?, s3?],
        })
    }
}

/// NumPy-compatible MT19937 (Mersenne Twister) PRNG.
///
/// Supports two seeding modes:
/// - **Classic** (`from_u32_seed`): NumPy's `RandomState(seed)` — `init_genrand(seed)`
/// - **Modern** (`from_seed_sequence`): NumPy's `MT19937(seed)` — SeedSequence-based
///
/// Produces the same u32 stream as NumPy for both modes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mt19937Rng {
    mt: Vec<u32>,
    pos: usize,
}

impl Mt19937Rng {
    /// Classic `init_genrand` seeding (NumPy's `RandomState(seed)`).
    ///
    /// After seeding, `pos = N` (624) meaning a twist is required before first output.
    #[must_use]
    pub fn from_u32_seed(seed: u32) -> Self {
        let mut mt = vec![0u32; MT_N];
        mt[0] = seed;
        for i in 1..MT_N {
            mt[i] = MT_INIT_MULT
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { mt, pos: MT_N }
    }

    /// Modern SeedSequence-based seeding (NumPy's `MT19937(seed)`).
    ///
    /// Fills the 624-element state from `SeedSequence.generate_state(624, u32)`,
    /// sets `mt[0] = UPPER_MASK`, and `pos = N-1` (623).
    pub fn from_seed_sequence(ss: &SeedSequence) -> Result<Self, SeedSequenceError> {
        let mut mt = ss.generate_state_u32(MT_N)?;
        mt[0] = MT_UPPER_MASK;
        Ok(Self { mt, pos: MT_N - 1 })
    }

    /// Create from a u64 seed using classic seeding (low 32 bits).
    #[must_use]
    pub fn from_u64_seed_classic(seed: u64) -> Self {
        Self::from_u32_seed(seed as u32)
    }

    /// Create from explicit state and position (for state restoration).
    pub fn from_raw_state(mt: &[u32], pos: usize) -> Result<Self, BitGeneratorError> {
        if mt.len() != MT_N {
            return Err(BitGeneratorError::StateSchemaInvalid(
                "MT19937 state vector must have exactly 624 elements",
            ));
        }
        if pos > MT_N {
            return Err(BitGeneratorError::StateSchemaInvalid(
                "MT19937 position must be <= 624",
            ));
        }
        Ok(Self {
            mt: mt.to_vec(),
            pos,
        })
    }

    /// Get the raw state as (state_vector, position).
    #[must_use]
    pub fn raw_state(&self) -> (&[u32], usize) {
        (&self.mt, self.pos)
    }

    /// The Mersenne Twister twist operation.
    /// Regenerates the entire 624-element state vector.
    fn twist(&mut self) {
        for kk in 0..(MT_N - MT_M) {
            let y = (self.mt[kk] & MT_UPPER_MASK) | (self.mt[kk + 1] & MT_LOWER_MASK);
            self.mt[kk] = self.mt[kk + MT_M] ^ (y >> 1) ^ if y & 1 != 0 { MT_MATRIX_A } else { 0 };
        }
        for kk in (MT_N - MT_M)..(MT_N - 1) {
            let y = (self.mt[kk] & MT_UPPER_MASK) | (self.mt[kk + 1] & MT_LOWER_MASK);
            self.mt[kk] =
                self.mt[kk + MT_M - MT_N] ^ (y >> 1) ^ if y & 1 != 0 { MT_MATRIX_A } else { 0 };
        }
        let y = (self.mt[MT_N - 1] & MT_UPPER_MASK) | (self.mt[0] & MT_LOWER_MASK);
        self.mt[MT_N - 1] = self.mt[MT_M - 1] ^ (y >> 1) ^ if y & 1 != 0 { MT_MATRIX_A } else { 0 };
    }

    /// The MT19937 tempering transformation.
    #[must_use]
    fn temper(mut y: u32) -> u32 {
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C_5680;
        y ^= (y << 15) & 0xEFC6_0000;
        y ^= y >> 18;
        y
    }

    /// Generate the next random u32.
    #[must_use]
    pub fn next_u32(&mut self) -> u32 {
        if self.pos >= MT_N {
            self.twist();
            self.pos = 0;
        }
        let y = self.mt[self.pos];
        self.pos += 1;
        Self::temper(y)
    }

    /// Generate the next random u64 by combining two u32 values.
    #[must_use]
    pub fn next_u64(&mut self) -> u64 {
        let hi = u64::from(self.next_u32());
        let lo = u64::from(self.next_u32());
        (hi << 32) | lo
    }

    /// Generate a random f64 in [0, 1) using NumPy's `genrand_res53` algorithm.
    ///
    /// `a = u32 >> 5` (27 bits), `b = u32 >> 6` (26 bits) → 53-bit mantissa.
    /// `result = (a * 2^26 + b) / 2^53`
    #[must_use]
    pub fn next_f64(&mut self) -> f64 {
        let a = f64::from(self.next_u32() >> 5);
        let b = f64::from(self.next_u32() >> 6);
        (a * 67_108_864.0 + b) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Generate a bounded random u32 in [0, upper_bound) using rejection sampling.
    pub fn bounded_u32(&mut self, upper_bound: u32) -> Result<u32, RandomError> {
        if upper_bound == 0 {
            return Err(RandomError::InvalidUpperBound);
        }
        let threshold = u32::MAX - u32::MAX % upper_bound;
        loop {
            let candidate = self.next_u32();
            if candidate < threshold {
                return Ok(candidate % upper_bound);
            }
        }
    }

    /// Fill a vector with `len` random u32 values.
    #[must_use]
    pub fn fill_u32(&mut self, len: usize) -> Vec<u32> {
        (0..len).map(|_| self.next_u32()).collect()
    }

    pub fn to_state_entries(&self) -> Vec<(String, u64)> {
        let mut entries = Vec::with_capacity(MT_N + 1);
        entries.push(("mt19937_pos".to_string(), self.pos as u64));
        for (i, &val) in self.mt.iter().enumerate() {
            entries.push((format!("mt19937_s{i}"), u64::from(val)));
        }
        entries
    }

    pub fn from_state_entries(entries: &[(String, u64)]) -> Option<Self> {
        let mut pos = 0usize;
        let mut mt = vec![0u32; MT_N];
        let mut found_count = 0;
        for (k, v) in entries {
            let normalized = k.trim();
            if normalized == "mt19937_pos" {
                pos = usize::try_from(*v).ok()?;
                if pos > MT_N {
                    return None;
                }
            } else if let Some(suffix) = normalized.strip_prefix("mt19937_s")
                && let Ok(idx) = suffix.parse::<usize>()
                && idx < MT_N
            {
                mt[idx] = u32::try_from(*v).ok()?;
                found_count += 1;
            }
        }
        if found_count == MT_N {
            Some(Self { mt, pos })
        } else {
            None
        }
    }
}

// NumPy-compatible SeedSequence constants (from Melissa O'Neill's seed_seq design)
const SS_INIT_A: u32 = 0x43b0_d7e5;
const SS_MULT_A: u32 = 0x931e_8875;
const SS_INIT_B: u32 = 0x8b51_f9dd;
const SS_MULT_B: u32 = 0x58f3_8ded;
const SS_MIX_MULT_L: u32 = 0xca01_f9dd;
const SS_MIX_MULT_R: u32 = 0x4973_f715;
const SS_XSHIFT: u32 = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeedSequence {
    entropy: Vec<u32>,
    spawn_key: Vec<u32>,
    pool_size: usize,
    pool: Vec<u32>,
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
        if entropy.is_empty()
            || !(DEFAULT_SEED_SEQUENCE_POOL_SIZE..=MAX_SEED_SEQUENCE_POOL_SIZE).contains(&pool_size)
        {
            return Err(SeedSequenceError::GenerateStateContractViolation);
        }

        // Assemble entropy: run_entropy + spawn_key
        // NumPy (since 1.19): when spawn_key is present and run_entropy < pool_size,
        // pad run_entropy with zeros to pool_size to avoid position conflicts.
        let mut assembled: Vec<u32> = entropy.to_vec();
        if !spawn_key.is_empty() && assembled.len() < pool_size {
            assembled.resize(pool_size, 0);
        }
        assembled.extend_from_slice(spawn_key);

        // Mix entropy into pool (NumPy algorithm)
        let pool = mix_entropy_pool(&assembled, pool_size);

        Ok(Self {
            entropy: entropy.to_vec(),
            spawn_key: spawn_key.to_vec(),
            pool_size,
            pool,
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
    pub fn pool(&self) -> &[u32] {
        &self.pool
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

    /// Generate `words` u32 values from the pool (NumPy algorithm).
    /// Cycles through pool elements applying hash transformation.
    pub fn generate_state_u32(&self, words: usize) -> Result<Vec<u32>, SeedSequenceError> {
        if words > MAX_SEED_SEQUENCE_WORDS {
            return Err(SeedSequenceError::GenerateStateContractViolation);
        }
        if words == 0 {
            return Ok(Vec::new());
        }

        let mut hash_const: u32 = SS_INIT_B;
        let mut state = vec![0u32; words];
        let pool_len = self.pool.len();

        for (i, slot) in state.iter_mut().enumerate() {
            let mut data_val: u32 = self.pool[i % pool_len];
            data_val ^= hash_const;
            hash_const = hash_const.wrapping_mul(SS_MULT_B);
            data_val = data_val.wrapping_mul(hash_const);
            data_val ^= data_val >> SS_XSHIFT;
            *slot = data_val;
        }

        Ok(state)
    }

    /// Generate `words` u64 values by drawing 2*words u32 values and pairing them.
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

    /// Spawn child SeedSequences by extending the spawn_key.
    /// Matches NumPy: child gets spawn_key + (child_index,).
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
        for i in self.spawn_counter..end {
            let mut child_spawn_key = self.spawn_key.clone();
            // NumPy appends child index as a single u32 value
            let child_idx =
                u32::try_from(i).map_err(|_| SeedSequenceError::SpawnContractViolation)?;
            child_spawn_key.push(child_idx);
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

/// NumPy-compatible entropy mixing into a pool of u32 words.
/// Implements the hash-pool algorithm from Melissa O'Neill's seed_seq.
fn mix_entropy_pool(entropy: &[u32], pool_size: usize) -> Vec<u32> {
    let mut hash_const: u32 = SS_INIT_A;
    let mut pool = vec![0u32; pool_size];

    /// Hash a value using the evolving hash constant.
    fn ss_hash(value: u32, hash_const: &mut u32) -> u32 {
        let mut v = value ^ *hash_const;
        *hash_const = hash_const.wrapping_mul(SS_MULT_A);
        v = v.wrapping_mul(*hash_const);
        v ^= v >> SS_XSHIFT;
        v
    }

    /// Mix two values using the L/R multiply-XOR-shift pattern.
    fn ss_mix(x: u32, y: u32) -> u32 {
        let result = SS_MIX_MULT_L
            .wrapping_mul(x)
            .wrapping_sub(SS_MIX_MULT_R.wrapping_mul(y));
        result ^ (result >> SS_XSHIFT)
    }

    // Step 1: Hash entropy into pool positions (up to pool_size).
    for i in 0..pool_size {
        if i < entropy.len() {
            pool[i] = ss_hash(entropy[i], &mut hash_const);
        } else {
            pool[i] = ss_hash(0, &mut hash_const);
        }
    }

    // Step 2: Mix all bits together so late bits can affect earlier bits.
    for i_src in 0..pool_size {
        for i_dst in 0..pool_size {
            if i_src != i_dst {
                pool[i_dst] = ss_mix(pool[i_dst], ss_hash(pool[i_src], &mut hash_const));
            }
        }
    }

    // Step 3: Add any remaining entropy, mixing each word with each pool word.
    if entropy.len() > pool_size {
        for &ent_val in &entropy[pool_size..] {
            for slot in pool.iter_mut() {
                *slot = ss_mix(*slot, ss_hash(ent_val, &mut hash_const));
            }
        }
    }

    pool
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
    pub seed: u128,
    pub counter: u128,
    pub schema_entries: Vec<(String, u64)>,
}

impl BitGeneratorState {
    fn algorithm_state_schema_value(
        kind: BitGeneratorKind,
        seed: u64,
        counter: u64,
    ) -> Option<u64> {
        match kind {
            BitGeneratorKind::Mt19937 => Some(counter % 624),
            BitGeneratorKind::Pcg64 => Some(splitmix64(seed ^ 0x5043_4736_3400_0001_u64)),
            BitGeneratorKind::Philox => Some(seed.rotate_left(17) ^ counter.rotate_right(7)),
            BitGeneratorKind::Sfc64 => Some(splitmix64(counter ^ seed ^ 0x5346_4336_3400_0001_u64)),
        }
    }

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

        if stream_seed != Some(self.seed as u64) || counter != Some(self.counter as u64) {
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

        // For real algorithms, we relax the check if they use custom state entries
        let is_real_prng = matches!(
            self.kind,
            BitGeneratorKind::Mt19937
                | BitGeneratorKind::Philox
                | BitGeneratorKind::Sfc64
                | BitGeneratorKind::Pcg64
        );
        let expected_algorithm_key = algorithm_state_schema_key(self.kind);
        let has_unknown_entries = self.schema_entries.iter().any(|(k, _)| {
            let normalized = k.trim();
            !matches!(
                normalized,
                "stream_seed" | "counter" | "algorithm_tag" | "schema_version"
            ) && normalized != expected_algorithm_key
                && !normalized.starts_with("mt19937_")
                && !normalized.starts_with("pcg64_")
                && !normalized.starts_with("philox_")
                && !normalized.starts_with("sfc64_")
        });

        if (!is_real_prng || !has_unknown_entries)
            && let Some(expected_algorithm_state) =
                Self::algorithm_state_schema_value(self.kind, self.seed as u64, self.counter as u64)
            && algorithm_state != Some(expected_algorithm_state)
        {
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

fn default_state_schema_entries(
    kind: BitGeneratorKind,
    seed: u128,
    counter: u128,
) -> Vec<(String, u64)> {
    let mut entries = vec![
        ("stream_seed".to_string(), seed as u64),
        ("counter".to_string(), counter as u64),
        ("algorithm_tag".to_string(), kind.stream_tag()),
        (
            "schema_version".to_string(),
            u64::from(BIT_GENERATOR_STATE_SCHEMA_VERSION),
        ),
    ];
    let algorithm_key = algorithm_state_schema_key(kind);
    if let Some(algorithm_value) =
        BitGeneratorState::algorithm_state_schema_value(kind, seed as u64, counter as u64)
    {
        entries.push((algorithm_key.to_string(), algorithm_value));
    }
    entries
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratorPicklePayload {
    pub bit_generator_state: BitGeneratorState,
    pub seed_sequence: Option<SeedSequenceSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RngBackend {
    Deterministic(DeterministicRng),
    Pcg64Dxsm(Pcg64DxsmRng),
    Mt19937(Mt19937Rng),
    Philox(PhiloxRng),
    Sfc64(Sfc64Rng),
}

impl RngBackend {
    fn next_u64(&mut self) -> u64 {
        match self {
            Self::Deterministic(rng) => rng.next_u64(),
            Self::Pcg64Dxsm(rng) => rng.next_u64(),
            Self::Mt19937(rng) => rng.next_u64(),
            Self::Philox(rng) => rng.next_u64(),
            Self::Sfc64(rng) => rng.next_u64(),
        }
    }

    fn next_f64(&mut self) -> f64 {
        match self {
            Self::Deterministic(rng) => rng.next_f64(),
            Self::Pcg64Dxsm(rng) => rng.next_f64(),
            Self::Mt19937(rng) => rng.next_f64(),
            Self::Philox(rng) => rng.next_f64(),
            Self::Sfc64(rng) => rng.next_f64(),
        }
    }

    fn bounded_u64(&mut self, upper_bound: u64) -> Result<u64, RandomError> {
        if upper_bound == 0 {
            return Err(RandomError::InvalidUpperBound);
        }
        match self {
            Self::Deterministic(rng) => rng.bounded_u64(upper_bound),
            Self::Pcg64Dxsm(rng) => rng.bounded_u64(upper_bound),
            Self::Mt19937(rng) => {
                let threshold = u64::MAX - u64::MAX % upper_bound;
                loop {
                    let candidate = rng.next_u64();
                    if candidate < threshold {
                        return Ok(candidate % upper_bound);
                    }
                }
            }
            Self::Philox(rng) => {
                let threshold = u64::MAX - u64::MAX % upper_bound;
                loop {
                    let candidate = rng.next_u64();
                    if candidate < threshold {
                        return Ok(candidate % upper_bound);
                    }
                }
            }
            Self::Sfc64(rng) => {
                let threshold = u64::MAX - u64::MAX % upper_bound;
                loop {
                    let candidate = rng.next_u64();
                    if candidate < threshold {
                        return Ok(candidate % upper_bound);
                    }
                }
            }
        }
    }

    fn fill_u64(&mut self, len: usize) -> Vec<u64> {
        match self {
            Self::Deterministic(rng) => rng.fill_u64(len),
            Self::Pcg64Dxsm(rng) => (0..len).map(|_| rng.next_u64()).collect(),
            Self::Mt19937(rng) => (0..len).map(|_| rng.next_u64()).collect(),
            Self::Philox(rng) => (0..len).map(|_| rng.next_u64()).collect(),
            Self::Sfc64(rng) => (0..len).map(|_| rng.next_u64()).collect(),
        }
    }

    fn jump_ahead(&mut self, steps: u64) {
        match self {
            Self::Deterministic(rng) => rng.jump_ahead(steps),
            Self::Pcg64Dxsm(rng) => rng.advance(u128::from(steps)),
            Self::Philox(rng) => rng.jump_ahead(steps),
            _ => {} // Mt19937 and Sfc64 don't have built-in jump-ahead in standard NumPy
        }
    }

    fn to_state_entries(&self) -> Vec<(String, u64)> {
        match self {
            Self::Deterministic(_) => Vec::new(),
            Self::Pcg64Dxsm(rng) => rng.to_state_entries(),
            Self::Mt19937(rng) => rng.to_state_entries(),
            Self::Philox(rng) => rng.to_state_entries(),
            Self::Sfc64(rng) => rng.to_state_entries(),
        }
    }

    fn state(&self) -> (u128, u128) {
        match self {
            Self::Deterministic(rng) => {
                let (s, c) = rng.state();
                (u128::from(s), u128::from(c))
            }
            Self::Pcg64Dxsm(rng) => rng.raw_state(),
            _ => (0, 0), // Other algorithms have larger states handled via schema_entries
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitGenerator {
    kind: BitGeneratorKind,
    rng: RngBackend,
}

impl BitGenerator {
    pub fn new(kind: BitGeneratorKind, seed: SeedMaterial) -> Result<Self, BitGeneratorError> {
        let rng = match seed {
            SeedMaterial::None => match kind {
                BitGeneratorKind::Pcg64 => {
                    RngBackend::Pcg64Dxsm(Pcg64DxsmRng::from_u64_seed(DEFAULT_RNG_SEED).map_err(
                        |_| BitGeneratorError::InitFailed("PCG64 initialization failed"),
                    )?)
                }
                BitGeneratorKind::Mt19937 => {
                    RngBackend::Mt19937(Mt19937Rng::from_u32_seed(DEFAULT_RNG_SEED as u32))
                }
                BitGeneratorKind::Philox => {
                    let ss = SeedSequence::new(&[
                        DEFAULT_RNG_SEED as u32,
                        (DEFAULT_RNG_SEED >> 32) as u32,
                    ])
                    .map_err(|_| BitGeneratorError::InitFailed("Philox initialization failed"))?;
                    RngBackend::Philox(PhiloxRng::from_seed_sequence(&ss).map_err(|_| {
                        BitGeneratorError::InitFailed("Philox initialization failed")
                    })?)
                }
                BitGeneratorKind::Sfc64 => {
                    let ss = SeedSequence::new(&[
                        DEFAULT_RNG_SEED as u32,
                        (DEFAULT_RNG_SEED >> 32) as u32,
                    ])
                    .map_err(|_| BitGeneratorError::InitFailed("SFC64 initialization failed"))?;
                    RngBackend::Sfc64(Sfc64Rng::from_seed_sequence(&ss).map_err(|_| {
                        BitGeneratorError::InitFailed("SFC64 initialization failed")
                    })?)
                }
            },
            SeedMaterial::U64(s) => match kind {
                BitGeneratorKind::Pcg64 => {
                    RngBackend::Pcg64Dxsm(Pcg64DxsmRng::from_u64_seed(s).map_err(|_| {
                        BitGeneratorError::InitFailed("PCG64 initialization failed")
                    })?)
                }
                BitGeneratorKind::Mt19937 => {
                    RngBackend::Mt19937(Mt19937Rng::from_u32_seed(s as u32))
                }
                BitGeneratorKind::Philox => {
                    let ss = SeedSequence::new(&[s as u32, (s >> 32) as u32]).map_err(|_| {
                        BitGeneratorError::InitFailed("Philox initialization failed")
                    })?;
                    RngBackend::Philox(PhiloxRng::from_seed_sequence(&ss).map_err(|_| {
                        BitGeneratorError::InitFailed("Philox initialization failed")
                    })?)
                }
                BitGeneratorKind::Sfc64 => {
                    let ss = SeedSequence::new(&[s as u32, (s >> 32) as u32]).map_err(|_| {
                        BitGeneratorError::InitFailed("SFC64 initialization failed")
                    })?;
                    RngBackend::Sfc64(Sfc64Rng::from_seed_sequence(&ss).map_err(|_| {
                        BitGeneratorError::InitFailed("SFC64 initialization failed")
                    })?)
                }
            },
            SeedMaterial::State { seed, counter } => match kind {
                BitGeneratorKind::Pcg64 => {
                    RngBackend::Pcg64Dxsm(Pcg64DxsmRng::from_raw_state(seed, counter))
                }
                _ => RngBackend::Deterministic(DeterministicRng::from_state(seed, counter)),
            },
            material => {
                let rng = deterministic_rng_from_seed_material(material).map_err(|_| {
                    BitGeneratorError::InitFailed(
                        "bit-generator constructor rejected seed material",
                    )
                })?;
                RngBackend::Deterministic(rng)
            }
        };
        Ok(Self { kind, rng })
    }

    pub fn from_seed_sequence(
        kind: BitGeneratorKind,
        seed_sequence: &SeedSequence,
    ) -> Result<Self, BitGeneratorError> {
        let backend = match kind {
            BitGeneratorKind::Pcg64 => RngBackend::Pcg64Dxsm(
                Pcg64DxsmRng::from_seed_sequence(seed_sequence)
                    .map_err(|_| BitGeneratorError::InitFailed("PCG64 SeedSequence init failed"))?,
            ),
            BitGeneratorKind::Mt19937 => {
                RngBackend::Mt19937(Mt19937Rng::from_seed_sequence(seed_sequence).map_err(
                    |_| BitGeneratorError::InitFailed("MT19937 SeedSequence init failed"),
                )?)
            }
            BitGeneratorKind::Philox => {
                RngBackend::Philox(PhiloxRng::from_seed_sequence(seed_sequence).map_err(|_| {
                    BitGeneratorError::InitFailed("Philox SeedSequence init failed")
                })?)
            }
            BitGeneratorKind::Sfc64 => RngBackend::Sfc64(
                Sfc64Rng::from_seed_sequence(seed_sequence)
                    .map_err(|_| BitGeneratorError::InitFailed("SFC64 SeedSequence init failed"))?,
            ),
        };
        Ok(Self { kind, rng: backend })
    }

    /// Create a BitGenerator backed by a real PCG64-DXSM PRNG.
    /// Distribution methods on Generator will produce NumPy-identical sequences.
    pub fn from_pcg64_dxsm(pcg: Pcg64DxsmRng) -> Self {
        Self {
            kind: BitGeneratorKind::Pcg64,
            rng: RngBackend::Pcg64Dxsm(pcg),
        }
    }

    #[must_use]
    pub const fn kind(&self) -> BitGeneratorKind {
        self.kind
    }

    fn raw_state(&self) -> (u128, u128) {
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
                (parent_seed as u64)
                    ^ self.kind.stream_tag()
                    ^ child_ordinal.wrapping_mul(GOLDEN_GAMMA),
            );
            let child_counter = (parent_counter as u64).wrapping_add(
                child_ordinal.checked_mul(self.kind.jump_stride()).ok_or(
                    BitGeneratorError::SpawnContractViolation(
                        "bit-generator child counter overflowed deterministic budget",
                    ),
                )?,
            );
            children.push(Self {
                kind: self.kind,
                rng: RngBackend::Deterministic(DeterministicRng::from_state(
                    u128::from(child_seed),
                    u128::from(child_counter),
                )),
            });
        }

        self.rng.jump_ahead(n_children_u64);
        Ok(children)
    }

    #[must_use]
    pub fn state(&self) -> BitGeneratorState {
        let (seed, counter) = self.raw_state();
        let mut schema_entries = default_state_schema_entries(self.kind, seed, counter);

        // Merge algorithm-specific state entries
        for entry in self.rng.to_state_entries() {
            // Only add if not already present (default_state_schema_entries might have added some)
            if !schema_entries.iter().any(|(k, _)| k == &entry.0) {
                schema_entries.push(entry);
            }
        }

        BitGeneratorState {
            kind: self.kind,
            schema_version: BIT_GENERATOR_STATE_SCHEMA_VERSION,
            seed,
            counter,
            schema_entries,
        }
    }

    pub fn set_state(&mut self, state: &BitGeneratorState) -> Result<(), BitGeneratorError> {
        if state.kind != self.kind {
            return Err(BitGeneratorError::StateSchemaInvalid(
                "bit-generator state kind does not match target algorithm",
            ));
        }
        state.validate()?;

        self.rng = match state.kind {
            BitGeneratorKind::Pcg64 => {
                let pcg = Pcg64DxsmRng::from_state_entries(&state.schema_entries).ok_or(
                    BitGeneratorError::StateSchemaInvalid("failed to restore PCG64 state"),
                )?;
                RngBackend::Pcg64Dxsm(pcg)
            }
            BitGeneratorKind::Mt19937 => {
                let mt = Mt19937Rng::from_state_entries(&state.schema_entries).ok_or(
                    BitGeneratorError::StateSchemaInvalid("failed to restore MT19937 state"),
                )?;
                RngBackend::Mt19937(mt)
            }
            BitGeneratorKind::Philox => {
                let philox = PhiloxRng::from_state_entries(&state.schema_entries).ok_or(
                    BitGeneratorError::StateSchemaInvalid("failed to restore Philox state"),
                )?;
                RngBackend::Philox(philox)
            }
            BitGeneratorKind::Sfc64 => {
                let sfc = Sfc64Rng::from_state_entries(&state.schema_entries).ok_or(
                    BitGeneratorError::StateSchemaInvalid("failed to restore SFC64 state"),
                )?;
                RngBackend::Sfc64(sfc)
            }
        };

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

/// Cached precomputed parameters for NumPy-compatible binomial sampling.
/// Avoids recomputation when the same `(n, p)` pair is used repeatedly.
/// Matches `binomial_t` in NumPy's distributions.h.
#[derive(Debug, Clone)]
struct BinomialCache {
    has_binomial: bool,
    psave: f64,
    nsave: i64,
    r: f64,
    q: f64,
    fm: f64,
    m: i64,
    p1: f64,
    xm: f64,
    xl: f64,
    xr: f64,
    c: f64,
    laml: f64,
    lamr: f64,
    p2: f64,
    p3: f64,
    p4: f64,
}

impl BinomialCache {
    fn new() -> Self {
        Self {
            has_binomial: false,
            psave: 0.0,
            nsave: 0,
            r: 0.0,
            q: 0.0,
            fm: 0.0,
            m: 0,
            p1: 0.0,
            xm: 0.0,
            xl: 0.0,
            xr: 0.0,
            c: 0.0,
            laml: 0.0,
            lamr: 0.0,
            p2: 0.0,
            p3: 0.0,
            p4: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generator {
    bit_generator: BitGenerator,
    seed_sequence: Option<SeedSequence>,
    /// Internal buffer for NumPy-compatible `next_uint32` buffering.
    /// Each u64 produces two u32 values (low first, then high).
    u32_buf: u32,
    u32_buf_ready: bool,
}

impl Generator {
    #[must_use]
    pub fn from_bit_generator(bit_generator: BitGenerator) -> Self {
        Self {
            bit_generator,
            seed_sequence: None,
            u32_buf: 0,
            u32_buf_ready: false,
        }
    }

    /// Create a Generator backed by PCG64-DXSM, matching `numpy.random.Generator(PCG64DXSM(seed))`.
    pub fn from_pcg64_dxsm(seed: u64) -> Result<Self, SeedSequenceError> {
        let pcg = Pcg64DxsmRng::from_u64_seed(seed)?;
        Ok(Self {
            bit_generator: BitGenerator::from_pcg64_dxsm(pcg),
            seed_sequence: None,
            u32_buf: 0,
            u32_buf_ready: false,
        })
    }

    pub fn from_seed_sequence(
        kind: BitGeneratorKind,
        seed_sequence: &SeedSequence,
    ) -> Result<Self, BitGeneratorError> {
        let bit_generator = BitGenerator::from_seed_sequence(kind, seed_sequence)?;
        Ok(Self {
            bit_generator,
            seed_sequence: Some(seed_sequence.clone()),
            u32_buf: 0,
            u32_buf_ready: false,
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
        if expected.state() != bit_generator.state() {
            return Err(BitGeneratorError::GeneratorBindingInvalid(
                "bit-generator and seed-sequence deterministic identities diverge",
            ));
        }
        Ok(Self {
            bit_generator,
            seed_sequence: Some(seed_sequence.clone()),
            u32_buf: 0,
            u32_buf_ready: false,
        })
    }

    #[must_use]
    pub fn bit_generator(&self) -> &BitGenerator {
        &self.bit_generator
    }

    #[must_use]
    pub fn next_u64(&mut self) -> u64 {
        self.u32_buf_ready = false;
        self.bit_generator.next_u64()
    }

    #[must_use]
    pub fn next_f64(&mut self) -> f64 {
        self.u32_buf_ready = false;
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
            u32_buf: 0,
            u32_buf_ready: false,
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
                    u32_buf: 0,
                    u32_buf_ready: false,
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

        Ok(Self {
            bit_generator,
            seed_sequence,
            u32_buf: 0,
            u32_buf_ready: false,
        })
    }

    // ── NumPy-compatible bounded integer primitives ──────────────────────

    /// Generate a 32-bit random integer using NumPy's buffering strategy.
    ///
    /// Each u64 from the underlying bit generator is split into two u32s:
    /// the low 32 bits are returned first, and the high 32 bits are buffered
    /// for the next call.  Matches `next_uint32()` in NumPy's `pcg64.c`.
    fn next_uint32(&mut self) -> u32 {
        if self.u32_buf_ready {
            self.u32_buf_ready = false;
            self.u32_buf
        } else {
            let val = self.bit_generator.next_u64();
            self.u32_buf = (val >> 32) as u32;
            self.u32_buf_ready = true;
            (val & 0xFFFF_FFFF) as u32
        }
    }

    /// 32-bit Lemire's method for bounded integers in `[0, rng]`.
    ///
    /// Matches `buffered_bounded_lemire_uint32()` in NumPy's `distributions.c`.
    /// Caller must ensure `rng < 0xFFFF_FFFF`.
    fn bounded_lemire_uint32(&mut self, rng: u32) -> u32 {
        let rng_excl = u64::from(rng) + 1;

        let mut m = u64::from(self.next_uint32()) * rng_excl;
        let mut leftover = m as u32;

        if u64::from(leftover) < rng_excl {
            let threshold = (u32::MAX - rng) % (rng + 1);
            while leftover < threshold {
                m = u64::from(self.next_uint32()) * rng_excl;
                leftover = m as u32;
            }
        }

        (m >> 32) as u32
    }

    /// 64-bit Lemire's method for bounded integers in `[0, rng]`.
    ///
    /// Matches `bounded_lemire_uint64()` in NumPy's `distributions.c`.
    /// Caller must ensure `rng < 0xFFFF_FFFF_FFFF_FFFF`.
    fn bounded_lemire_uint64(&mut self, rng: u64) -> u64 {
        let rng_excl = u128::from(rng) + 1;

        let mut m = u128::from(self.bit_generator.next_u64()) * rng_excl;
        let mut leftover = m as u64;

        if u128::from(leftover) < rng_excl {
            let threshold = (u64::MAX - rng) % (rng + 1);
            while leftover < threshold {
                m = u128::from(self.bit_generator.next_u64()) * rng_excl;
                leftover = m as u64;
            }
        }

        (m >> 64) as u64
    }

    /// NumPy-compatible bounded uint64 with automatic 32/64-bit dispatch.
    ///
    /// Returns a value in `[0, rng]` (inclusive).
    /// Matches `random_bounded_uint64()` in NumPy's `distributions.c`:
    /// - `rng == 0` → returns 0
    /// - `rng <= 0xFFFF_FFFE` → 32-bit Lemire via `next_uint32()`
    /// - `rng == 0xFFFF_FFFF` → raw `next_uint32()`
    /// - `rng < u64::MAX` → 64-bit Lemire via `next_u64()`
    /// - `rng == u64::MAX` → raw `next_u64()`
    fn numpy_bounded_uint64(&mut self, rng: u64) -> u64 {
        if rng == 0 {
            return 0;
        }
        if rng <= 0xFFFF_FFFF {
            if rng == 0xFFFF_FFFF {
                return u64::from(self.next_uint32());
            }
            #[expect(clippy::cast_possible_truncation)]
            return u64::from(self.bounded_lemire_uint32(rng as u32));
        }
        if rng == u64::MAX {
            return self.bit_generator.next_u64();
        }
        self.bounded_lemire_uint64(rng)
    }

    /// Masked rejection sampling for a random integer in `[0, max]`.
    ///
    /// Matches `random_interval()` in NumPy's `distributions.c`.
    /// Used by shuffle/permutation.  Uses 32-bit path when `max <= 0xFFFF_FFFF`.
    fn random_interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }

        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        mask |= mask >> 32;

        if max <= 0xFFFF_FFFF {
            #[expect(clippy::cast_possible_truncation)]
            let mask32 = mask as u32;
            loop {
                let value = u64::from(self.next_uint32() & mask32);
                if value <= max {
                    return value;
                }
            }
        } else {
            loop {
                let value = self.bit_generator.next_u64() & mask;
                if value <= max {
                    return value;
                }
            }
        }
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
    ///
    /// NumPy requires `high >= low` and both to be finite.
    pub fn uniform(&mut self, low: f64, high: f64, size: usize) -> Result<Vec<f64>, RandomError> {
        if !low.is_finite() || !high.is_finite() {
            return Err(RandomError::InvalidParameter);
        }
        if high < low {
            return Err(RandomError::InvalidParameter);
        }
        let range = high - low;
        Ok((0..size).map(|_| low + self.next_f64() * range).collect())
    }

    /// Generate random integers in `[low, high)`.
    ///
    /// Mimics `rng.integers(low, high, size)`.
    /// Uses NumPy-compatible Lemire bounded integer algorithm with
    /// automatic 32/64-bit dispatch.
    pub fn integers(&mut self, low: i64, high: i64, size: usize) -> Result<Vec<i64>, RandomError> {
        if size == 0 {
            return Ok(Vec::new());
        }
        if high <= low {
            return Err(RandomError::InvalidUpperBound);
        }
        let off = low as u64;
        // rng is the inclusive upper bound relative to off
        let rng = (high as u64).wrapping_sub(low as u64) - 1;
        let mut result = Vec::with_capacity(size);
        for _ in 0..size {
            let val = (off.wrapping_add(self.numpy_bounded_uint64(rng))) as i64;
            result.push(val);
        }
        Ok(result)
    }

    /// Generate integers in `[low, high]` (inclusive on both ends).
    ///
    /// Mimics `rng.integers(low, high, size, endpoint=True)`.
    /// Uses NumPy-compatible Lemire bounded integer algorithm.
    pub fn integers_endpoint(
        &mut self,
        low: i64,
        high: i64,
        size: usize,
    ) -> Result<Vec<i64>, RandomError> {
        if size == 0 {
            return Ok(Vec::new());
        }
        if high < low {
            return Err(RandomError::InvalidUpperBound);
        }
        let off = low as u64;
        // rng is the inclusive upper bound relative to off
        let rng = (high as u64).wrapping_sub(low as u64);
        let mut result = Vec::with_capacity(size);
        for _ in 0..size {
            let val = if rng == u64::MAX {
                // Full u64 range — just use raw
                self.next_u64() as i64
            } else {
                (off.wrapping_add(self.numpy_bounded_uint64(rng))) as i64
            };
            result.push(val);
        }
        Ok(result)
    }

    /// Generate standard normal (Gaussian, mean=0, std=1) samples using
    /// the Ziggurat method (matching NumPy's Generator.standard_normal).
    ///
    /// Mimics `rng.standard_normal(size)`.
    #[must_use]
    pub fn standard_normal(&mut self, size: usize) -> Vec<f64> {
        (0..size).map(|_| self.sample_ziggurat_normal()).collect()
    }

    /// Generate normal (Gaussian) samples with given mean and standard deviation.
    ///
    /// Mimics `rng.normal(loc, scale, size)`.
    ///
    /// NumPy requires `scale >= 0`.
    pub fn normal(&mut self, loc: f64, scale: f64, size: usize) -> Result<Vec<f64>, RandomError> {
        if scale < 0.0 || scale.is_nan() {
            return Err(RandomError::InvalidParameter);
        }
        Ok(self
            .standard_normal(size)
            .into_iter()
            .map(|z| loc + scale * z)
            .collect())
    }

    /// Generate exponentially distributed samples.
    ///
    /// Mimics `rng.exponential(scale, size)`.
    ///
    /// NumPy requires `scale >= 0`.
    pub fn exponential(&mut self, scale: f64, size: usize) -> Result<Vec<f64>, RandomError> {
        if scale < 0.0 || scale.is_nan() {
            return Err(RandomError::InvalidParameter);
        }
        Ok((0..size)
            .map(|_| scale * self.sample_ziggurat_exponential())
            .collect())
    }

    /// Generate standard exponential samples (scale=1).
    ///
    /// Mimics `rng.standard_exponential(size)`.
    #[must_use]
    pub fn standard_exponential(&mut self, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| self.sample_ziggurat_exponential())
            .collect()
    }

    /// Generate standard gamma samples (scale=1).
    ///
    /// Mimics `rng.standard_gamma(shape, size)`.
    pub fn standard_gamma(
        &mut self,
        shape_param: f64,
        size: usize,
    ) -> Result<Vec<f64>, RandomError> {
        if shape_param <= 0.0 {
            return Err(RandomError::InvalidParameter);
        }
        Ok((0..size).map(|_| self.sample_gamma(shape_param)).collect())
    }

    /// Generate random bytes.
    ///
    /// Mimics `rng.bytes(length)`. Returns `length` random bytes.
    #[must_use]
    pub fn bytes(&mut self, length: usize) -> Vec<u8> {
        let mut result = Vec::with_capacity(length);
        let mut remaining = length;
        while remaining > 0 {
            let word = self.next_u64();
            let bytes = word.to_le_bytes();
            let take = remaining.min(8);
            result.extend_from_slice(&bytes[..take]);
            remaining -= take;
        }
        result
    }

    /// Generate Poisson-distributed samples.
    ///
    /// Mimics `rng.poisson(lam, size)`.
    /// Uses NumPy's exact algorithms: multiplicative method for lam < 10,
    /// PTRS (Hörmann 1993) for lam >= 10.
    pub fn poisson(&mut self, lam: f64, size: usize) -> Result<Vec<u64>, RandomError> {
        if !lam.is_finite() || lam < 0.0 {
            return Err(RandomError::InvalidParameter);
        }
        Ok((0..size).map(|_| self.sample_poisson_single(lam)).collect())
    }

    /// Single Poisson sample matching NumPy's `random_poisson` dispatcher.
    fn sample_poisson_single(&mut self, lam: f64) -> u64 {
        if lam >= 10.0 {
            self.poisson_ptrs(lam) as u64
        } else if lam == 0.0 {
            0
        } else {
            self.poisson_mult(lam) as u64
        }
    }

    /// Multiplicative (Knuth) method for small lambda.
    /// Matches `random_poisson_mult()` in NumPy's distributions.c.
    fn poisson_mult(&mut self, lam: f64) -> i64 {
        let enlam = (-lam).exp();
        let mut x: i64 = 0;
        let mut prod = 1.0;
        loop {
            let u = self.next_f64();
            prod *= u;
            if prod > enlam {
                x += 1;
            } else {
                return x;
            }
        }
    }

    /// Transformed rejection method (PTRS) for large lambda.
    /// Matches `random_poisson_ptrs()` in NumPy's distributions.c.
    /// W. Hörmann, "The transformed rejection method for generating
    /// Poisson random variables", Insurance: Mathematics and Economics 12, 39-45 (1993).
    fn poisson_ptrs(&mut self, lam: f64) -> i64 {
        let slam = lam.sqrt();
        let loglam = lam.ln();
        let b = 0.931 + 2.53 * slam;
        let a = -0.059 + 0.02483 * b;
        let invalpha = 1.1239 + 1.1328 / (b - 3.4);
        let vr = 0.9277 - 3.6224 / (b - 2.0);

        loop {
            let u = self.next_f64() - 0.5;
            let v = self.next_f64();
            let us = 0.5 - u.abs();
            let k = ((2.0 * a / us + b) * u + lam + 0.43).floor() as i64;

            if us >= 0.07 && v <= vr {
                return k;
            }
            if k < 0 || (us < 0.013 && v > us) {
                continue;
            }
            if v.ln() + invalpha.ln() - (a / (us * us) + b).ln()
                <= -lam + (k as f64) * loglam - random_loggam((k + 1) as f64)
            {
                return k;
            }
        }
    }

    /// Generate binomially distributed samples.
    ///
    /// Mimics `rng.binomial(n, p, size)`.
    /// Uses NumPy's exact algorithm: BTPE (Kachitvichyanukul & Schmeiser 1988)
    /// for large `min(p,q)*n`, inversion for small `min(p,q)*n <= 30`.
    pub fn binomial(&mut self, n: u64, p: f64, size: usize) -> Result<Vec<u64>, RandomError> {
        if !(0.0..=1.0).contains(&p) || p.is_nan() {
            return Err(RandomError::InvalidParameter);
        }
        // Precompute cached BTPE/inversion parameters (shared across all samples)
        let mut cache = BinomialCache::new();
        Ok((0..size)
            .map(|_| self.sample_binomial_single(n, p, &mut cache))
            .collect())
    }

    /// Single binomial sample matching NumPy's `random_binomial` dispatcher.
    fn sample_binomial_single(&mut self, n: u64, p: f64, cache: &mut BinomialCache) -> u64 {
        if n == 0 || p == 0.0 {
            return 0;
        }

        let mut remaining = n;
        let mut total = 0_u64;
        while remaining > MAX_BINOMIAL_DIRECT_TRIALS {
            total += self.sample_binomial_single_i64(i64::MAX, p, cache) as u64;
            remaining -= MAX_BINOMIAL_DIRECT_TRIALS;
        }
        total + self.sample_binomial_single_i64(remaining as i64, p, cache) as u64
    }

    /// Single binomial sample matching NumPy's `random_binomial` dispatcher
    /// for the direct `i64` kernel range used by BTPE and inversion.
    fn sample_binomial_single_i64(&mut self, n: i64, p: f64, cache: &mut BinomialCache) -> i64 {
        if n == 0 || p == 0.0 {
            return 0;
        }

        if p <= 0.5 {
            if p * (n as f64) <= 30.0 {
                self.binomial_inversion(n, p, cache)
            } else {
                self.binomial_btpe(n, p, cache)
            }
        } else {
            let q = 1.0 - p;
            if q * (n as f64) <= 30.0 {
                n - self.binomial_inversion(n, q, cache)
            } else {
                n - self.binomial_btpe(n, q, cache)
            }
        }
    }

    /// BTPE algorithm for binomial sampling (Kachitvichyanukul & Schmeiser 1988).
    /// Matches `random_binomial_btpe()` in NumPy's distributions.c.
    #[expect(clippy::many_single_char_names)]
    fn binomial_btpe(&mut self, n: i64, p: f64, cache: &mut BinomialCache) -> i64 {
        if !cache.has_binomial || cache.nsave != n || cache.psave != p {
            cache.nsave = n;
            cache.psave = p;
            cache.has_binomial = true;
            cache.r = p.min(1.0 - p);
            cache.q = 1.0 - cache.r;
            cache.fm = (n as f64 + 1.0) * cache.r;
            cache.m = cache.fm.floor() as i64;
            cache.p1 =
                (2.195 * ((n as f64) * cache.r * cache.q).sqrt() - 4.6 * cache.q).floor() + 0.5;
            cache.xm = (cache.m as f64) + 0.5;
            cache.xl = cache.xm - cache.p1;
            cache.xr = cache.xm + cache.p1;
            cache.c = 0.134 + 20.5 / (15.3 + cache.m as f64);
            let a1 = (cache.fm - cache.xl) / (cache.fm - cache.xl * cache.r);
            cache.laml = a1 * (1.0 + a1 / 2.0);
            let a2 = (cache.xr - cache.fm) / (cache.xr * cache.q);
            cache.lamr = a2 * (1.0 + a2 / 2.0);
            cache.p2 = cache.p1 * (1.0 + 2.0 * cache.c);
            cache.p3 = cache.p2 + cache.c / cache.laml;
            cache.p4 = cache.p3 + cache.c / cache.lamr;
        }

        let r = cache.r;
        let q = cache.q;
        let m = cache.m;
        let p1 = cache.p1;
        let xm = cache.xm;
        let xl = cache.xl;
        let xr = cache.xr;
        let c = cache.c;
        let laml = cache.laml;
        let lamr = cache.lamr;
        let p2 = cache.p2;
        let p3 = cache.p3;
        let p4 = cache.p4;

        let nrq = (n as f64) * r * q;
        let s = r / q;
        let a = s * ((n + 1) as f64);

        loop {
            let u = self.next_f64() * p4;
            let mut v = self.next_f64();

            let y = if u <= p1 {
                (xm - p1 * v + u).floor() as i64
            } else if u <= p2 {
                let x = xl + (u - p1) / c;
                v = v * c + 1.0 - (m as f64 - x + 0.5).abs() / p1;
                if v > 1.0 {
                    continue;
                }
                x.floor() as i64
            } else if u <= p3 {
                let y_val = (xl + v.ln() / laml).floor() as i64;
                if y_val < 0 || v == 0.0 {
                    continue;
                }
                v *= (u - p2) * laml;
                y_val
            } else {
                let y_val = (xr - v.ln() / lamr).floor() as i64;
                if y_val > n || v == 0.0 {
                    continue;
                }
                v *= (u - p3) * lamr;
                y_val
            };

            let k = (y - m).unsigned_abs();
            if k <= 20 && (k as f64) <= nrq / 2.0 - 1.0 {
                // Direct ratio test for small k
                let mut f_val = 1.0;
                if m < y {
                    for i in (m + 1)..=y {
                        f_val *= a / (i as f64) - s;
                    }
                } else if m > y {
                    for i in (y + 1)..=m {
                        f_val /= a / (i as f64) - s;
                    }
                }
                if v <= f_val {
                    return if p <= 0.5 { y } else { n - y };
                }
            } else {
                // Stirling approximation for large k
                let log_v = v.ln();
                let rho = (k as f64 / nrq)
                    * ((k as f64 * (k as f64 / 3.0 + 0.625) + 0.16666666666666666) / nrq + 0.5);
                let t = -(k as f64 * k as f64) / (2.0 * nrq);
                if log_v < t - rho {
                    return if p <= 0.5 { y } else { n - y };
                }
                if log_v > t + rho {
                    continue;
                }

                // Final Stirling check
                let x1 = y as f64 + 1.0;
                let f1 = m as f64 + 1.0;
                let z = (n - m) as f64 + 1.0;
                let w = (n - y) as f64 + 1.0;
                if log_v
                    <= (m as f64 + 0.5) * (f1 / x1).ln()
                        + (n as f64 - m as f64 + 0.5) * (z / w).ln()
                        + (y as f64 - m as f64) * (w * r / (x1 * q)).ln()
                        + (1.0 / 12.0) * (1.0 / f1 + 1.0 / z - 1.0 / x1 - 1.0 / w)
                {
                    return if p <= 0.5 { y } else { n - y };
                }
            }
        }
    }

    /// Inversion algorithm for binomial sampling (small n*p).
    /// Matches `random_binomial_inversion()` in NumPy's distributions.c.
    fn binomial_inversion(&mut self, n: i64, p: f64, cache: &mut BinomialCache) -> i64 {
        if !cache.has_binomial || cache.nsave != n || cache.psave != p {
            cache.nsave = n;
            cache.psave = p;
            cache.has_binomial = true;
            cache.q = 1.0 - p;
            // qn = (1-p)^n computed via log1p for numerical stability
            cache.r = ((n as f64) * (-p).ln_1p()).exp();
            let np = (n as f64) * p;
            cache.m = (n as f64).min(np + 10.0 * (np * cache.q + 1.0).sqrt()) as i64;
        }

        let q = cache.q;
        let qn = cache.r;
        let bound = cache.m;

        let mut x: i64 = 0;
        let mut px = qn;
        let mut u = self.next_f64();
        loop {
            if u <= px {
                return x;
            }
            x += 1;
            if x > bound {
                x = 0;
                px = qn;
                u = self.next_f64();
            } else {
                u -= px;
                px *= ((n - x + 1) as f64) * p / ((x as f64) * q);
            }
        }
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
        if n == 0 && size > 0 {
            return Err(RandomError::InvalidUpperBound);
        }
        if !replace && size > n {
            return Err(RandomError::InvalidUpperBound);
        }
        if replace {
            let mut result = Vec::with_capacity(size);
            for _ in 0..size {
                // NumPy choice(replace=True) calls integers(0, n) → Lemire
                let idx = self.numpy_bounded_uint64(n as u64 - 1) as usize;
                result.push(a[idx]);
            }
            Ok(result)
        } else {
            // Fisher-Yates shuffle on a copy, then take first `size` elements.
            // NumPy shuffles an int64 index array via _shuffle_int → Lemire.
            let mut pool = a.to_vec();
            for i in (1..n).rev() {
                let j = self.numpy_bounded_uint64(i as u64) as usize;
                pool.swap(i, j);
            }
            Ok(pool[..size].to_vec())
        }
    }

    /// Choose random elements from an array with probability weights.
    ///
    /// Mimics `rng.choice(a, size, replace, p=weights)`. The `p` array
    /// must sum to 1.0 (within tolerance) and have the same length as `a`.
    ///
    /// For `replace=true`, uses the inverse-CDF method.
    /// For `replace=false`, uses sequential weighted sampling without replacement.
    pub fn choice_weighted(
        &mut self,
        a: &[f64],
        size: usize,
        replace: bool,
        p: &[f64],
    ) -> Result<Vec<f64>, RandomError> {
        let n = a.len();
        if p.len() != n {
            return Err(RandomError::InvalidUpperBound);
        }
        if !replace && size > n {
            return Err(RandomError::InvalidUpperBound);
        }
        // Validate probabilities are non-negative and sum to ~1.0
        let sum: f64 = p.iter().sum();
        if !sum.is_finite()
            || (sum - 1.0).abs() > 1e-8
            || p.iter().any(|&v| !v.is_finite() || v < 0.0)
        {
            return Err(RandomError::InvalidUpperBound);
        }
        if !replace && p.iter().filter(|&&weight| weight > 0.0).count() < size {
            return Err(RandomError::InvalidParameter);
        }

        if replace {
            // Inverse-CDF sampling
            let mut cdf = Vec::with_capacity(n);
            let mut cumulative = 0.0;
            for &prob in p {
                cumulative += prob;
                cdf.push(cumulative);
            }
            let mut result = Vec::with_capacity(size);
            for _ in 0..size {
                let u = self.next_f64();
                let idx = cdf.partition_point(|&c| c <= u).min(n - 1);
                result.push(a[idx]);
            }
            Ok(result)
        } else {
            // Weighted sampling without replacement
            let mut weights = p.to_vec();
            let mut indices: Vec<usize> = (0..n).collect();
            let mut result = Vec::with_capacity(size);
            for _ in 0..size {
                let total: f64 = weights.iter().sum();
                if total <= 0.0 {
                    break;
                }
                let u = self.next_f64() * total;
                let mut cumulative = 0.0;
                let mut chosen = indices.len() - 1;
                for (i, &w) in weights.iter().enumerate() {
                    cumulative += w;
                    if cumulative > u {
                        chosen = i;
                        break;
                    }
                }
                result.push(a[indices[chosen]]);
                indices.swap_remove(chosen);
                weights.swap_remove(chosen);
            }
            Ok(result)
        }
    }

    /// Shuffle a mutable slice in-place.
    ///
    /// Mimics `rng.shuffle(x)`.
    /// Uses NumPy's `random_interval()` (masked rejection with 32-bit path
    /// for small indices), matching `_shuffle_raw` in `_generator.pyx`.
    pub fn shuffle(&mut self, x: &mut [f64]) -> Result<(), RandomError> {
        let n = x.len();
        for i in (1..n).rev() {
            let j = self.random_interval(i as u64) as usize;
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
    /// Uses NumPy's `random_interval()` for Fisher-Yates shuffle.
    pub fn permutation_range(&mut self, n: usize) -> Result<Vec<u64>, RandomError> {
        let mut result: Vec<u64> = (0..n as u64).collect();
        for i in (1..n).rev() {
            let j = self.random_interval(i as u64) as usize;
            result.swap(i, j);
        }
        Ok(result)
    }

    // ── additional distributions ────────

    /// Gamma distribution using Marsaglia and Tsang's method.
    pub fn gamma(
        &mut self,
        shape_param: f64,
        scale: f64,
        size: usize,
    ) -> Result<Vec<f64>, RandomError> {
        if shape_param <= 0.0 || shape_param.is_nan() {
            return Err(RandomError::InvalidParameter);
        }
        if scale <= 0.0 || scale.is_nan() {
            return Err(RandomError::InvalidParameter);
        }
        Ok((0..size)
            .map(|_| self.sample_gamma(shape_param) * scale)
            .collect())
    }

    fn sample_gamma(&mut self, shape_param: f64) -> f64 {
        if shape_param == 1.0 {
            // Special case: gamma(1) = exponential(1)
            return self.sample_ziggurat_exponential();
        }
        if shape_param == 0.0 {
            return 0.0;
        }
        if shape_param < 1.0 {
            // NumPy's exact algorithm for shape < 1 from distributions.c:
            // Uses uniform + exponential rejection.
            loop {
                let u = self.next_f64();
                let v = self.sample_ziggurat_exponential();
                if u <= 1.0 - shape_param {
                    let x = u.powf(1.0 / shape_param);
                    if x <= v {
                        return x;
                    }
                } else {
                    let y = -((1.0 - u) / shape_param).ln();
                    let x = (1.0 - shape_param + shape_param * y).powf(1.0 / shape_param);
                    if x <= v + y {
                        return x;
                    }
                }
            }
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
        self.sample_ziggurat_normal()
    }

    /// Ziggurat method for standard normal, matching NumPy's
    /// `random_standard_normal` in `distributions.c` exactly.
    ///
    /// Bit layout from a single u64: bits 0..7 = rectangle index,
    /// bit 8 = sign, bits 9..60 = rectangle position (rabs, 52 bits).
    fn sample_ziggurat_normal(&mut self) -> f64 {
        use crate::ziggurat::*;
        loop {
            let mut r = self.next_u64();
            let idx = (r & 0xFF) as usize;
            r >>= 8;
            let sign = r & 0x1;
            let rabs = (r >> 1) & 0x000F_FFFF_FFFF_FFFF;
            let mut x = rabs as f64 * ZIGGURAT_NOR_W[idx];
            if (sign & 0x1) != 0 {
                x = -x;
            }
            if rabs < ZIGGURAT_NOR_K[idx] {
                return x;
            }
            if idx == 0 {
                // Tail: Marsaglia's method using log1p(-U) to avoid log(0)
                loop {
                    let xx = -ZIGGURAT_NOR_INV_R * (-self.next_f64()).ln_1p();
                    let yy = -(-self.next_f64()).ln_1p();
                    if yy + yy > xx * xx {
                        return if (rabs >> 8) & 0x1 != 0 {
                            -(ZIGGURAT_NOR_R + xx)
                        } else {
                            ZIGGURAT_NOR_R + xx
                        };
                    }
                }
            } else {
                let f_idx = ZIGGURAT_NOR_F[idx];
                let f_prev = ZIGGURAT_NOR_F[idx - 1];
                if (f_prev - f_idx) * self.next_f64() + f_idx < (-0.5 * x * x).exp() {
                    return x;
                }
            }
        }
    }

    /// Ziggurat method for standard exponential, matching NumPy's
    /// `random_standard_exponential` in distributions.c exactly.
    fn sample_ziggurat_exponential(&mut self) -> f64 {
        use crate::ziggurat::*;
        loop {
            let mut ri = self.next_u64();
            ri >>= 3;
            let idx = (ri & 0xFF) as usize;
            ri >>= 8;
            let x = ri as f64 * ZIGGURAT_EXP_W[idx];
            if ri < ZIGGURAT_EXP_K[idx] {
                return x;
            }
            // Tail: use log1p(-U) to avoid log(0), matching NumPy GH-13361
            if idx == 0 {
                return ZIGGURAT_EXP_R - (-self.next_f64()).ln_1p();
            }
            // Wedge test: (F[idx-1] - F[idx]) * U + F[idx] < exp(-x)
            let f_idx = ZIGGURAT_EXP_F[idx];
            let f_prev = ZIGGURAT_EXP_F[idx - 1];
            if (f_prev - f_idx) * self.next_f64() + f_idx < (-x).exp() {
                return x;
            }
        }
    }

    /// Beta distribution via gamma sampling.
    pub fn beta(&mut self, a: f64, b: f64, size: usize) -> Result<Vec<f64>, RandomError> {
        if a <= 0.0 || b <= 0.0 {
            return Err(RandomError::InvalidParameter);
        }
        Ok((0..size)
            .map(|_| {
                let x = self.sample_gamma(a);
                let y = self.sample_gamma(b);
                if x == 0.0 && y == 0.0 {
                    0.0
                } else {
                    x / (x + y)
                }
            })
            .collect())
    }

    /// Geometric distribution: number of trials until first success.
    /// For p >= 1/3, uses search method; otherwise uses inversion via
    /// standard_exponential (matching NumPy's algorithm).
    pub fn geometric(&mut self, p: f64, size: usize) -> Result<Vec<u64>, RandomError> {
        if p <= 0.0 || p > 1.0 || p.is_nan() {
            return Err(RandomError::InvalidParameter);
        }
        Ok((0..size)
            .map(|_| {
                if p >= 1.0 / 3.0 {
                    // Search method
                    let q = 1.0 - p;
                    let u = self.next_f64();
                    let mut x = 1u64;
                    let mut sum = p;
                    let mut prod = p;
                    while u > sum {
                        prod *= q;
                        sum += prod;
                        x += 1;
                    }
                    x
                } else {
                    // Inversion via standard exponential
                    let z = (-self.sample_ziggurat_exponential() / (-p).ln_1p()).ceil();
                    if z >= 9.223_372_036_854_776e18 {
                        i64::MAX as u64
                    } else {
                        z as u64
                    }
                }
            })
            .collect())
    }

    /// Log-normal distribution.
    pub fn lognormal(
        &mut self,
        mean: f64,
        sigma: f64,
        size: usize,
    ) -> Result<Vec<f64>, RandomError> {
        if sigma < 0.0 || sigma.is_nan() {
            return Err(RandomError::InvalidParameter);
        }
        Ok((0..size)
            .map(|_| {
                let z = self.sample_standard_normal_single();
                (mean + sigma * z).exp()
            })
            .collect())
    }

    /// Chi-squared distribution with df degrees of freedom.
    pub fn chisquare(&mut self, df: f64, size: usize) -> Result<Vec<f64>, RandomError> {
        // Chi-squared is gamma(df/2, 2)
        self.gamma(df / 2.0, 2.0, size)
    }

    /// Standard Cauchy distribution (matching NumPy: ratio of two normals).
    pub fn standard_cauchy(&mut self, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| {
                let n1 = self.sample_ziggurat_normal();
                let n2 = self.sample_ziggurat_normal();
                n1 / n2
            })
            .collect()
    }

    /// Triangular distribution (matching NumPy's exact formula).
    ///
    /// NumPy requires `left <= mode <= right` and `left < right`.
    pub fn triangular(
        &mut self,
        left: f64,
        mode: f64,
        right: f64,
        size: usize,
    ) -> Result<Vec<f64>, RandomError> {
        if left > mode || mode > right || left >= right {
            return Err(RandomError::InvalidParameter);
        }
        if !left.is_finite() || !mode.is_finite() || !right.is_finite() {
            return Err(RandomError::InvalidParameter);
        }
        let base = right - left;
        let leftbase = mode - left;
        let ratio = leftbase / base;
        let leftprod = leftbase * base;
        let rightprod = (right - mode) * base;
        Ok((0..size)
            .map(|_| {
                let u = self.next_f64();
                if u <= ratio {
                    left + (u * leftprod).sqrt()
                } else {
                    right - ((1.0 - u) * rightprod).sqrt()
                }
            })
            .collect())
    }

    /// Laplace (double exponential) distribution (matching NumPy's algorithm).
    ///
    /// NumPy requires `scale >= 0`.
    pub fn laplace(&mut self, loc: f64, scale: f64, size: usize) -> Result<Vec<f64>, RandomError> {
        if scale < 0.0 || scale.is_nan() {
            return Err(RandomError::InvalidParameter);
        }
        Ok((0..size)
            .map(|_| {
                loop {
                    let u = self.next_f64();
                    if u >= 0.5 {
                        return loc - scale * (2.0 - u - u).ln();
                    } else if u > 0.0 {
                        return loc + scale * (u + u).ln();
                    }
                }
            })
            .collect())
    }

    /// Gumbel distribution (matching NumPy: uses 1-u to avoid log(0)).
    ///
    /// NumPy requires `scale >= 0`.
    pub fn gumbel(&mut self, loc: f64, scale: f64, size: usize) -> Result<Vec<f64>, RandomError> {
        if scale < 0.0 || scale.is_nan() {
            return Err(RandomError::InvalidParameter);
        }
        Ok((0..size)
            .map(|_| {
                loop {
                    let u = 1.0 - self.next_f64();
                    if u < 1.0 {
                        return loc - scale * (-u.ln()).ln();
                    }
                }
            })
            .collect())
    }

    /// Weibull distribution (matching NumPy: uses standard_exponential).
    pub fn weibull(&mut self, a: f64, size: usize) -> Result<Vec<f64>, RandomError> {
        if a <= 0.0 {
            return Err(RandomError::InvalidParameter);
        }
        Ok((0..size)
            .map(|_| self.sample_ziggurat_exponential().powf(1.0 / a))
            .collect())
    }

    // ── multivariate distributions ────────

    /// Multinomial distribution (np.random.multinomial).
    /// Returns `size` samples, each a vector of length `pvals.len()`.
    /// `n` is the number of trials, `pvals` are probabilities (must sum to ~1).
    pub fn multinomial(&mut self, n: u64, pvals: &[f64], size: usize) -> Vec<Vec<u64>> {
        let mut cache = BinomialCache::new();
        (0..size)
            .map(|_| {
                let mut result = vec![0u64; pvals.len()];
                let mut remaining = n;
                let mut p_remaining = 1.0;
                for (i, &p) in pvals.iter().enumerate() {
                    if remaining == 0 {
                        break;
                    }
                    if i == pvals.len() - 1 {
                        result[i] = remaining;
                        break;
                    }
                    let p_cond = (p / p_remaining).clamp(0.0, 1.0);
                    let draws = self.sample_binomial_single(remaining, p_cond, &mut cache);
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

    /// Dirichlet distribution (np.random.dirichlet).
    /// Returns `size` samples, each a vector of length `alpha.len()`.
    pub fn dirichlet(&mut self, alpha: &[f64], size: usize) -> Result<Vec<Vec<f64>>, RandomError> {
        if alpha.iter().any(|&a| a.is_nan() || a < 0.0) {
            return Err(RandomError::InvalidParameter);
        }
        Ok((0..size)
            .map(|_| {
                let gamma_samples: Vec<f64> = alpha.iter().map(|&a| self.sample_gamma(a)).collect();
                let sum: f64 = gamma_samples.iter().sum();
                if sum == 0.0 {
                    vec![0.0; gamma_samples.len()]
                } else {
                    gamma_samples.into_iter().map(|g| g / sum).collect()
                }
            })
            .collect())
    }

    /// Multivariate normal distribution (np.random.multivariate_normal).
    /// Simplified version using diagonal covariance only.
    /// `mean` is the mean vector, `cov_diag` is the diagonal of the covariance matrix.
    pub fn multivariate_normal_diag(
        &mut self,
        mean: &[f64],
        cov_diag: &[f64],
        size: usize,
    ) -> Result<Vec<Vec<f64>>, RandomError> {
        if cov_diag.iter().any(|&v| v < 0.0) {
            return Err(RandomError::InvalidParameter);
        }
        Ok((0..size)
            .map(|_| {
                mean.iter()
                    .zip(cov_diag)
                    .map(|(&m, &v)| m + self.sample_standard_normal_single() * v.sqrt())
                    .collect()
            })
            .collect())
    }

    /// Negative binomial distribution (np.random.negative_binomial).
    /// Number of failures before `n` successes, with success probability `p`.
    ///
    /// Uses gamma-Poisson mixture matching NumPy's `random_negative_binomial`:
    /// `Y = gamma(n, (1-p)/p); return poisson(Y)`.
    /// Negative binomial distribution.
    ///
    /// NumPy requires `n > 0` and `0 < p <= 1`.
    pub fn negative_binomial(
        &mut self,
        n: f64,
        p: f64,
        size: usize,
    ) -> Result<Vec<u64>, RandomError> {
        if n <= 0.0 || n.is_nan() || p.is_nan() || p <= 0.0 || p > 1.0 {
            return Err(RandomError::InvalidParameter);
        }
        Ok((0..size)
            .map(|_| {
                let y = self.sample_gamma(n) * (1.0 - p) / p;
                self.sample_poisson_single(y)
            })
            .collect())
    }

    /// F-distribution (Fisher-Snedecor).  Ratio of two scaled chi-squared
    /// variates: (X1/dfnum) / (X2/dfden).
    pub fn f_distribution(
        &mut self,
        dfnum: f64,
        dfden: f64,
        size: usize,
    ) -> Result<Vec<f64>, RandomError> {
        if dfnum <= 0.0 || dfden <= 0.0 {
            return Err(RandomError::InvalidParameter);
        }
        Ok((0..size)
            .map(|_| {
                let x1 = self.sample_gamma(dfnum / 2.0) * 2.0 / dfnum;
                let x2 = self.sample_gamma(dfden / 2.0) * 2.0 / dfden;
                x1 / x2
            })
            .collect())
    }

    /// Alias for `f_distribution` matching NumPy's `rng.f(dfnum, dfden, size)`.
    pub fn f(&mut self, dfnum: f64, dfden: f64, size: usize) -> Result<Vec<f64>, RandomError> {
        self.f_distribution(dfnum, dfden, size)
    }

    /// Randomly permute elements of `x` along a given axis (np.random.Generator.permuted).
    ///
    /// With `axis=None`, this matches NumPy's flattened-array behavior. With
    /// `Some(axis)`, it permutes each 1-D slice along that axis independently.
    /// `x` is a flat row-major array with the given `shape`.
    pub fn permuted(
        &mut self,
        x: &[f64],
        shape: &[usize],
        axis: Option<usize>,
    ) -> Result<Vec<f64>, RandomError> {
        if shape.is_empty() {
            return Err(RandomError::InvalidUpperBound);
        }
        let total: usize =
            fnp_ndarray::element_count(shape).map_err(|_| RandomError::InvalidParameter)?;
        if x.len() != total {
            return Err(RandomError::InvalidUpperBound);
        }

        let mut result = x.to_vec();
        if total <= 1 {
            return Ok(result);
        }

        if axis.is_none() {
            for i in (1..total).rev() {
                let j = self.random_interval(i as u64) as usize;
                result.swap(i, j);
            }
            return Ok(result);
        }
        let axis = axis.expect("checked above");
        if axis >= shape.len() {
            return Err(RandomError::InvalidUpperBound);
        }

        // Compute strides for row-major layout
        let ndim = shape.len();
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        let axis_len = shape[axis];
        if axis_len <= 1 {
            return Ok(result);
        }
        let axis_stride = strides[axis];

        // Number of independent 1-D slices to shuffle
        let n_slices = total / axis_len;

        // For each slice along the axis, perform Fisher-Yates shuffle
        for slice_idx in 0..n_slices {
            // Compute multi-index excluding axis dimension
            let mut multi_idx = vec![0usize; ndim];
            let mut rem = slice_idx;
            for d in (0..ndim).rev() {
                if d == axis {
                    continue;
                }
                multi_idx[d] = rem % shape[d];
                rem /= shape[d];
            }
            let mut base_offset = 0;
            for d in 0..ndim {
                if d != axis {
                    base_offset += multi_idx[d] * strides[d];
                }
            }

            // Gather indices along the axis
            let indices: Vec<usize> = (0..axis_len)
                .map(|k| base_offset + k * axis_stride)
                .collect();

            // Fisher-Yates shuffle on these indices (random_interval for generic arrays)
            for i in (1..axis_len).rev() {
                let j = self.random_interval(i as u64) as usize;
                result.swap(indices[i], indices[j]);
            }
        }

        Ok(result)
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

    /// Non-central chi-squared distribution (scipy.stats.ncx2).
    ///
    /// Generated as the sum of `df` independent standard normals shifted by `nonc/df`,
    /// or equivalently: Poisson(nonc/2) mixture of chi-squared variates.
    /// Uses the simpler additive method: sum of (Z + sqrt(nonc/df))² for each df.
    pub fn noncentral_chisquare(
        &mut self,
        df: f64,
        nonc: f64,
        size: usize,
    ) -> Result<Vec<f64>, RandomError> {
        if df <= 0.0 || nonc < 0.0 {
            return Err(RandomError::InvalidParameter);
        }
        Ok((0..size)
            .map(|_| {
                if df > 1.0 {
                    // X ~ chi²(df-1) + (Z + sqrt(nonc))²
                    let chi2_part = self.sample_gamma((df - 1.0) / 2.0) * 2.0;
                    let z = self.sample_standard_normal_single() + nonc.sqrt();
                    chi2_part + z * z
                } else {
                    // df <= 1.0: use Poisson mixture
                    let i = self.sample_poisson_single(nonc / 2.0);
                    self.sample_gamma(df / 2.0 + i as f64) * 2.0
                }
            })
            .collect())
    }

    /// Non-central F-distribution (scipy.stats.ncf).
    ///
    /// Ratio of non-central chi-squared to central chi-squared:
    /// `(X1/dfnum) / (X2/dfden)` where X1 ~ ncχ²(dfnum, nonc), X2 ~ χ²(dfden).
    pub fn noncentral_f(
        &mut self,
        dfnum: f64,
        dfden: f64,
        nonc: f64,
        size: usize,
    ) -> Result<Vec<f64>, RandomError> {
        if dfnum <= 0.0 || dfden <= 0.0 || nonc < 0.0 {
            return Err(RandomError::InvalidParameter);
        }
        Ok((0..size)
            .map(|_| {
                let nc_chi2 = {
                    if dfnum > 1.0 {
                        let chi2_part = self.sample_gamma((dfnum - 1.0) / 2.0) * 2.0;
                        let z = self.sample_standard_normal_single() + nonc.sqrt();
                        chi2_part + z * z
                    } else {
                        let i = self.sample_poisson_single(nonc / 2.0);
                        self.sample_gamma(dfnum / 2.0 + i as f64) * 2.0
                    }
                };
                let chi2 = self.sample_gamma(dfden / 2.0) * 2.0;
                (nc_chi2 / dfnum) / (chi2 / dfden)
            })
            .collect())
    }

    /// Power distribution (matching NumPy: uses standard_exponential).
    ///
    /// NumPy requires `a > 0`.
    pub fn power(&mut self, a: f64, size: usize) -> Result<Vec<f64>, RandomError> {
        if a <= 0.0 || a.is_nan() {
            return Err(RandomError::InvalidParameter);
        }
        Ok((0..size)
            .map(|_| (-(-self.sample_ziggurat_exponential()).exp_m1()).powf(1.0 / a))
            .collect())
    }

    /// Von Mises circular distribution (Best-Fisher algorithm).
    ///
    /// NumPy requires `kappa >= 0`.
    pub fn vonmises(&mut self, mu: f64, kappa: f64, size: usize) -> Result<Vec<f64>, RandomError> {
        if kappa < 0.0 || kappa.is_nan() {
            return Err(RandomError::InvalidParameter);
        }
        Ok((0..size)
            .map(|_| {
                if kappa < 1e-6 {
                    return std::f64::consts::PI * (2.0 * self.next_f64() - 1.0);
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
                        return wrap_angle_to_pi(mu + theta);
                    }
                }
            })
            .collect())
    }

    /// Rayleigh distribution (matching NumPy: uses standard_exponential).
    ///
    /// NumPy requires `scale >= 0`.
    pub fn rayleigh(&mut self, scale: f64, size: usize) -> Result<Vec<f64>, RandomError> {
        if scale < 0.0 || scale.is_nan() {
            return Err(RandomError::InvalidParameter);
        }
        Ok((0..size)
            .map(|_| scale * (2.0 * self.sample_ziggurat_exponential()).sqrt())
            .collect())
    }

    /// Pareto distribution (matching NumPy: uses expm1 of standard_exponential).
    pub fn pareto(&mut self, a: f64, size: usize) -> Result<Vec<f64>, RandomError> {
        if a <= 0.0 {
            return Err(RandomError::InvalidParameter);
        }
        Ok((0..size)
            .map(|_| (self.sample_ziggurat_exponential() / a).exp_m1())
            .collect())
    }

    /// Logistic distribution via inverse-CDF.
    pub fn logistic(&mut self, loc: f64, scale: f64, size: usize) -> Result<Vec<f64>, RandomError> {
        if scale < 0.0 {
            return Err(RandomError::InvalidParameter);
        }
        Ok((0..size)
            .map(|_| {
                if scale == 0.0 {
                    return loc;
                }
                let u = self.next_f64();
                loc + scale * (u / (1.0 - u)).ln()
            })
            .collect())
    }

    /// Hypergeometric distribution: draws from a population of `ngood + nbad`
    /// containing `ngood` success states, taking `nsample` draws without replacement.
    ///
    /// Uses NumPy's exact algorithm: HRUA (Stadlober 1989) for large samples,
    /// direct counting via `random_interval()` for small samples.
    ///
    /// # Errors
    /// Returns `InvalidParameter` if ngood or nbad >= 10^9, or if nsample > ngood + nbad.
    pub fn hypergeometric(
        &mut self,
        ngood: u64,
        nbad: u64,
        nsample: u64,
        size: usize,
    ) -> Result<Vec<u64>, RandomError> {
        const HYPERGEOMETRIC_MAX: u64 = 1_000_000_000;
        if ngood >= HYPERGEOMETRIC_MAX {
            return Err(RandomError::InvalidParameter);
        }
        if nbad >= HYPERGEOMETRIC_MAX {
            return Err(RandomError::InvalidParameter);
        }
        let total = ngood + nbad;
        if nsample > total {
            return Err(RandomError::InvalidParameter);
        }
        let good = ngood as i64;
        let bad = nbad as i64;
        let sample = nsample as i64;
        Ok((0..size)
            .map(|_| self.sample_hypergeometric(good, bad, sample) as u64)
            .collect())
    }

    /// Single hypergeometric sample matching NumPy's `random_hypergeometric`.
    fn sample_hypergeometric(&mut self, good: i64, bad: i64, sample: i64) -> i64 {
        let total = good + bad;
        if sample >= 10 && sample <= total - 10 {
            self.hypergeometric_hrua(good, bad, sample)
        } else {
            self.hypergeometric_sample(good, bad, sample)
        }
    }

    /// Simple hypergeometric sampling via `random_interval()`.
    /// Matches `hypergeometric_sample()` in NumPy's `random_hypergeometric.c`.
    fn hypergeometric_sample(&mut self, good: i64, bad: i64, sample: i64) -> i64 {
        let total = good + bad;
        let computed_sample = if sample > total / 2 {
            total - sample
        } else {
            sample
        };

        let mut remaining_total = total;
        let mut remaining_good = good;
        let mut cs = computed_sample;

        while cs > 0 && remaining_good > 0 && remaining_total > remaining_good {
            remaining_total -= 1;
            if (self.random_interval(remaining_total as u64) as i64) < remaining_good {
                remaining_good -= 1;
            }
            cs -= 1;
        }

        if remaining_total == remaining_good {
            remaining_good -= cs;
        }

        if sample > total / 2 {
            remaining_good
        } else {
            good - remaining_good
        }
    }

    /// HRUA (ratio-of-uniforms) hypergeometric sampling.
    /// Matches `hypergeometric_hrua()` in NumPy's `random_hypergeometric.c`.
    #[expect(clippy::many_single_char_names)]
    fn hypergeometric_hrua(&mut self, good: i64, bad: i64, sample: i64) -> i64 {
        // D1 = 2*sqrt(2/e), D2 = 3 - 2*sqrt(3/e)
        const D1: f64 = 1.7155277699214135;
        const D2: f64 = 0.8989161620588988;

        let popsize = good + bad;
        let computed_sample = sample.min(popsize - sample);
        let mingoodbad = good.min(bad);
        let maxgoodbad = good.max(bad);

        let p = (mingoodbad as f64) / (popsize as f64);
        let q = (maxgoodbad as f64) / (popsize as f64);

        let mu = (computed_sample as f64) * p;
        let a = mu + 0.5;

        let var = ((popsize - computed_sample) as f64) * (computed_sample as f64) * p * q
            / ((popsize - 1) as f64);
        let c = (var + 0.5).sqrt();
        let h = D1 * c + D2;

        let m = (((computed_sample + 1) as f64) * ((mingoodbad + 1) as f64)
            / ((popsize + 2) as f64))
            .floor() as i64;

        let g = logfactorial(m)
            + logfactorial(mingoodbad - m)
            + logfactorial(computed_sample - m)
            + logfactorial(maxgoodbad - computed_sample + m);

        let b = (((computed_sample.min(mingoodbad) + 1) as f64).min(a + 16.0 * c)).floor();

        let mut k;
        loop {
            let u = self.next_f64();
            let v = self.next_f64();
            let x = a + h * (v - 0.5) / u;

            if x < 0.0 || x >= b {
                continue;
            }

            k = x.floor() as i64;

            let gp = logfactorial(k)
                + logfactorial(mingoodbad - k)
                + logfactorial(computed_sample - k)
                + logfactorial(maxgoodbad - computed_sample + k);

            let t = g - gp;

            if u * (4.0 - u) - 3.0 <= t {
                break;
            }
            if u * (u - t) >= 1.0 {
                continue;
            }
            if 2.0 * u.ln() <= t {
                break;
            }
        }

        if good > bad {
            k = computed_sample - k;
        }
        if computed_sample < sample {
            k = good - k;
        }

        k
    }

    /// Zipf (Zipfian) distribution with parameter `a > 1`.
    ///
    /// Matches `random_zipf()` in NumPy's `distributions.c`.
    /// Returns integer-valued floats (zipf values are positive integers).
    pub fn zipf(&mut self, a: f64, size: usize) -> Result<Vec<f64>, RandomError> {
        if a.is_nan() || a <= 1.0 {
            return Err(RandomError::InvalidParameter);
        }
        Ok((0..size)
            .map(|_| self.sample_zipf_single(a) as f64)
            .collect())
    }

    fn sample_zipf_single(&mut self, a: f64) -> i64 {
        if a >= 1025.0 {
            return 1;
        }
        let am1 = a - 1.0;
        let b = 2.0_f64.powf(am1);
        let umin = (i64::MAX as f64).powf(-am1);

        loop {
            let u01 = self.next_f64();
            let u = u01 * umin + (1.0 - u01); // U in (Umin, 1]
            let v = self.next_f64();
            let x = u.powf(-1.0 / am1).floor();

            if x > (i64::MAX as f64) || x < 1.0 {
                continue;
            }

            let t = (1.0 + 1.0 / x).powf(am1);
            if v * x * (t - 1.0) / (b - 1.0) <= t / b {
                return x as i64;
            }
        }
    }

    /// Wald (inverse Gaussian) distribution (matching NumPy's algorithm).
    pub fn wald(&mut self, mean: f64, scale: f64, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| {
                let y = self.sample_ziggurat_normal();
                let y = mean * y * y;
                let d = 1.0 + (1.0 + 4.0 * scale / y).sqrt();
                let x = mean * (1.0 - 2.0 / d);
                let u = self.next_f64();
                if u <= mean / (mean + x) {
                    x
                } else {
                    mean * mean / x
                }
            })
            .collect()
    }

    /// Logarithmic (log-series) distribution (matching NumPy's algorithm).
    pub fn logseries(&mut self, p: f64, size: usize) -> Result<Vec<u64>, RandomError> {
        if p <= 0.0 || p >= 1.0 {
            return Err(RandomError::InvalidParameter);
        }
        let r = (-p).ln_1p(); // log1p(-p) = ln(1 - p)
        Ok((0..size)
            .map(|_| {
                loop {
                    let v = self.next_f64();
                    if v >= p {
                        return 1;
                    }
                    let u = self.next_f64();
                    let q = -(r * u).exp_m1(); // -expm1(r * u)
                    if v <= q * q {
                        let result = (1.0 + v.ln() / q.ln()).floor() as i64;
                        if result < 1 || v == 0.0 {
                            continue;
                        }
                        return result as u64;
                    }
                    if v >= q {
                        return 1;
                    }
                    return 2;
                }
            })
            .collect())
    }

    /// Maxwell distribution (np.random.Generator.maxwell).
    /// The Maxwell distribution with scale parameter `scale`.
    /// PDF: sqrt(2/pi) * x^2 * exp(-x^2 / (2*scale^2)) / scale^3
    /// Generated via: scale * sqrt(chi2(3)) = scale * sqrt(X1^2 + X2^2 + X3^2)
    /// where X1, X2, X3 are independent standard normals.
    pub fn maxwell(&mut self, scale: f64, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| {
                let x1 = self.sample_standard_normal_single();
                let x2 = self.sample_standard_normal_single();
                let x3 = self.sample_standard_normal_single();
                scale * (x1 * x1 + x2 * x2 + x3 * x3).sqrt()
            })
            .collect()
    }

    /// Multivariate hypergeometric distribution.
    /// Draw `nsample` items without replacement from a population of groups
    /// where `colors[i]` is the number of items in group `i`.
    /// Returns a vector of vectors, each of length `colors.len()`.
    ///
    /// # Errors
    /// Returns `InvalidParameter` if sum(colors) >= 10^9 or nsample > sum(colors).
    pub fn multivariate_hypergeometric(
        &mut self,
        colors: &[u64],
        nsample: u64,
        size: usize,
    ) -> Result<Vec<Vec<u64>>, RandomError> {
        const HYPERGEOMETRIC_MAX: u64 = 1_000_000_000;
        let total: u64 = colors.iter().sum();
        if total >= HYPERGEOMETRIC_MAX {
            return Err(RandomError::InvalidParameter);
        }
        if nsample > total {
            return Err(RandomError::InvalidParameter);
        }
        Ok((0..size)
            .map(|_| {
                let mut remaining = total;
                let mut draws_left = nsample;
                let mut result = Vec::with_capacity(colors.len());
                for &color_count in colors {
                    if remaining == 0 || draws_left == 0 {
                        result.push(0);
                        continue;
                    }
                    // Draw from hypergeometric(color_count, remaining - color_count, draws_left)
                    let ngood = color_count;
                    let nbad = remaining - color_count;
                    let n = draws_left;
                    let drawn =
                        self.sample_hypergeometric(ngood as i64, nbad as i64, n as i64) as u64;
                    result.push(drawn);
                    remaining -= color_count;
                    draws_left -= drawn;
                }
                result
            })
            .collect())
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

    /// Half-normal distribution: |X| where X ~ N(0, sigma^2).
    /// Equivalent to the folded normal with mean 0.
    pub fn halfnormal(&mut self, sigma: f64, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| (self.sample_standard_normal_single() * sigma).abs())
            .collect()
    }

    /// Truncated normal distribution on `[low, high]` using rejection sampling.
    /// Draws from N(loc, scale^2) conditioned on `low <= X <= high`.
    pub fn truncated_normal(
        &mut self,
        loc: f64,
        scale: f64,
        low: f64,
        high: f64,
        size: usize,
    ) -> Vec<f64> {
        let mut result = Vec::with_capacity(size);
        for _ in 0..size {
            loop {
                let x = loc + self.sample_standard_normal_single() * scale;
                if x >= low && x <= high {
                    result.push(x);
                    break;
                }
            }
        }
        result
    }

    /// Lomax (Pareto Type II) distribution with shape `c` and scale 1.
    /// If X ~ Pareto(c), then X - 1 ~ Lomax(c).
    /// PDF: c / (1 + x)^(c+1) for x >= 0.
    pub fn lomax(&mut self, c: f64, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| {
                let u = self.next_f64();
                (1.0 - u).powf(-1.0 / c) - 1.0
            })
            .collect()
    }

    /// Levy distribution with location `loc` and scale `c`.
    /// Uses the inverse CDF method: X = loc + c / (Phi^{-1}(1-U/2))^2
    /// where Phi^{-1} is the standard normal quantile.
    pub fn levy(&mut self, loc: f64, c: f64, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| {
                let z = self.sample_standard_normal_single().abs();
                if z < 1e-15 {
                    loc + c * 1e30 // avoid division by zero
                } else {
                    loc + c / (z * z)
                }
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
    let mode = mode.trim();
    let class = class.trim();
    let known_mode = mode == "strict" || mode == "hardened";
    let known_class = class == "known_compatible"
        || class == "known_compatible_low_risk"
        || class == "known_compatible_high_risk"
        || class == "known_incompatible"
        || class == "known_incompatible_semantics"
        || class == "unknown"
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
    use std::process::Command;

    use super::{
        BIT_GENERATOR_STATE_SCHEMA_VERSION, BitGenerator, BitGeneratorError, BitGeneratorKind,
        DEFAULT_RNG_SEED, DeterministicRng, Generator, GeneratorPicklePayload,
        MAX_RNG_JUMP_OPERATIONS, MAX_SEED_SEQUENCE_CHILDREN, MAX_SEED_SEQUENCE_WORDS, Mt19937,
        Mt19937Rng, Pcg64, Pcg64DxsmRng, Philox, RANDOM_PACKET_REASON_CODES, RNG_CORE_REASON_CODES,
        RandomError, RandomLogRecord, RandomPolicyError, RandomRuntimeMode, RandomState,
        SeedMaterial, SeedSequence, SeedSequenceError, SeedSequenceSnapshot, Sfc64, default_rng,
        generator_from_seed_sequence, validate_rng_policy_metadata,
    };

    fn packet007_artifacts() -> Vec<String> {
        vec![
            "artifacts/phase2c/FNP-P2C-007/fixture_manifest.json".to_string(),
            "artifacts/phase2c/FNP-P2C-007/parity_gate.yaml".to_string(),
        ]
    }

    fn oracle_python_bin() -> String {
        std::env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".to_string())
    }

    fn numpy_oracle_available() -> bool {
        Command::new(oracle_python_bin())
            .args(["-c", "import numpy"])
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    fn numpy_oracle_permuted(shape: &[usize], axis: Option<usize>) -> Vec<f64> {
        let shape_arg = format!(
            "[{}]",
            shape
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        );
        let axis_arg = axis
            .map(|value| value.to_string())
            .unwrap_or_else(|| "__none__".to_string());
        let script = r#"
import json
import sys
import numpy as np

shape = tuple(json.loads(sys.argv[1]))
axis = sys.argv[2]
rng = np.random.Generator(np.random.PCG64DXSM(12345))
arr = np.arange(np.prod(shape), dtype=float).reshape(shape)
if axis == "__none__":
    out = rng.permuted(arr, axis=None)
else:
    out = rng.permuted(arr, axis=int(axis))
print(",".join(str(float(v)) for v in out.reshape(-1)))
"#;
        let output = Command::new(oracle_python_bin())
            .arg("-c")
            .arg(script)
            .arg(shape_arg)
            .arg(axis_arg)
            .output()
            .expect("python oracle should launch");
        assert!(
            output.status.success(),
            "NumPy permuted oracle must succeed"
        );
        String::from_utf8(output.stdout)
            .expect("oracle stdout must be utf-8")
            .trim()
            .split(',')
            .filter(|token| !token.is_empty())
            .map(|token| token.parse::<f64>().expect("oracle float"))
            .collect()
    }

    fn numpy_oracle_dirichlet(alpha: &[f64], size: usize) -> Vec<Vec<f64>> {
        let alpha_arg = format!(
            "[{}]",
            alpha
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let script = r#"
import json
import sys
import numpy as np

alpha = json.loads(sys.argv[1])
size = int(sys.argv[2])
rng = np.random.Generator(np.random.PCG64DXSM(12345))
out = rng.dirichlet(alpha, size=size)
for row in out:
    print(",".join(str(float(value)) for value in row.tolist()))
"#;
        let output = Command::new(oracle_python_bin())
            .arg("-c")
            .arg(script)
            .arg(alpha_arg)
            .arg(size.to_string())
            .output()
            .expect("python oracle should launch");
        assert!(
            output.status.success(),
            "NumPy dirichlet oracle must succeed"
        );
        String::from_utf8(output.stdout)
            .expect("oracle stdout must be utf-8")
            .lines()
            .map(|line| {
                line.split(',')
                    .filter(|token| !token.is_empty())
                    .map(|token| token.parse::<f64>().expect("oracle float"))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn numpy_oracle_zipf_outcome(a: f64, size: usize) -> String {
        let a_arg = if a.is_nan() {
            "__nan__".to_string()
        } else if a.is_infinite() && a.is_sign_positive() {
            "__inf__".to_string()
        } else if a.is_infinite() && a.is_sign_negative() {
            "__ninf__".to_string()
        } else {
            a.to_string()
        };
        let script = r#"
import sys
import numpy as np

a = sys.argv[1]
if a == "__nan__":
    a = float("nan")
elif a == "__inf__":
    a = float("inf")
elif a == "__ninf__":
    a = float("-inf")
else:
    a = float(a)
size = int(sys.argv[2])
rng = np.random.default_rng(12345)
try:
    out = rng.zipf(a, size)
    print("ok:" + ",".join(str(int(value)) for value in out.tolist()))
except Exception as exc:
    print(f"err:{type(exc).__name__}:{exc}")
"#;
        let output = Command::new(oracle_python_bin())
            .arg("-c")
            .arg(script)
            .arg(a_arg)
            .arg(size.to_string())
            .output()
            .expect("python oracle should launch");
        assert!(output.status.success(), "NumPy zipf oracle must succeed");
        String::from_utf8(output.stdout)
            .expect("oracle stdout must be utf-8")
            .trim()
            .to_string()
    }

    fn numpy_oracle_integers_outcome(low: i64, high: i64, size: usize, endpoint: bool) -> String {
        let script = r#"
import sys
import numpy as np

low = int(sys.argv[1])
high = int(sys.argv[2])
size = int(sys.argv[3])
endpoint = sys.argv[4] == "true"
rng = np.random.Generator(np.random.PCG64DXSM(12345))
try:
    out = rng.integers(low, high, size=size, endpoint=endpoint)
    print("ok:" + ",".join(str(int(value)) for value in out.tolist()))
except Exception as exc:
    print(f"err:{type(exc).__name__}:{exc}")
"#;
        let output = Command::new(oracle_python_bin())
            .arg("-c")
            .arg(script)
            .arg(low.to_string())
            .arg(high.to_string())
            .arg(size.to_string())
            .arg(if endpoint { "true" } else { "false" })
            .output()
            .expect("python oracle should launch");
        assert!(
            output.status.success(),
            "NumPy integers oracle must succeed"
        );
        String::from_utf8(output.stdout)
            .expect("oracle stdout must be utf-8")
            .trim()
            .to_string()
    }

    fn numpy_oracle_vonmises(mu: f64, kappa: f64, size: usize) -> Vec<f64> {
        let script = r#"
import sys
import numpy as np

mu = float(sys.argv[1])
kappa = float(sys.argv[2])
size = int(sys.argv[3])
rng = np.random.Generator(np.random.PCG64DXSM(12345))
out = rng.vonmises(mu, kappa, size)
print(",".join(str(float(value)) for value in out.tolist()))
"#;
        let output = Command::new(oracle_python_bin())
            .arg("-c")
            .arg(script)
            .arg(mu.to_string())
            .arg(kappa.to_string())
            .arg(size.to_string())
            .output()
            .expect("python oracle should launch");
        assert!(
            output.status.success(),
            "NumPy vonmises oracle must succeed"
        );
        String::from_utf8(output.stdout)
            .expect("oracle stdout must be utf-8")
            .trim()
            .split(',')
            .filter(|token| !token.is_empty())
            .map(|token| token.parse::<f64>().expect("oracle float"))
            .collect()
    }

    fn numpy_oracle_seed_sequence_spawn_words(
        entropy: &[u32],
        first_batch: usize,
        second_batch: usize,
    ) -> Vec<(Vec<u32>, Vec<u32>)> {
        let entropy_arg = entropy
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let script = r#"
import sys
import numpy as np

entropy = [int(part) for part in sys.argv[1].split(",") if part]
first_batch = int(sys.argv[2])
second_batch = int(sys.argv[3])
seed = entropy[0] if len(entropy) == 1 else entropy
ss = np.random.SeedSequence(seed)
children = ss.spawn(first_batch)
children.extend(ss.spawn(second_batch))
for child in children:
    spawn_key = ",".join(str(value) for value in child.spawn_key)
    state = ",".join(str(int(value)) for value in child.generate_state(8).tolist())
    print(f"{spawn_key}|{state}")
"#;
        let output = Command::new(oracle_python_bin())
            .arg("-c")
            .arg(script)
            .arg(entropy_arg)
            .arg(first_batch.to_string())
            .arg(second_batch.to_string())
            .output()
            .expect("python oracle should launch");
        assert!(
            output.status.success(),
            "NumPy SeedSequence.spawn oracle must succeed"
        );
        String::from_utf8(output.stdout)
            .expect("oracle stdout must be utf-8")
            .lines()
            .map(|line| {
                let (spawn_key, state) = line
                    .split_once('|')
                    .expect("oracle line must contain spawn key and state");
                let spawn_key = if spawn_key.is_empty() {
                    Vec::new()
                } else {
                    spawn_key
                        .split(',')
                        .map(|value| value.parse::<u32>().expect("oracle spawn key"))
                        .collect()
                };
                let state = if state.is_empty() {
                    Vec::new()
                } else {
                    state
                        .split(',')
                        .map(|value| value.parse::<u32>().expect("oracle state"))
                        .collect()
                };
                (spawn_key, state)
            })
            .collect()
    }

    fn numpy_oracle_generator_spawn_raw(
        entropy: &[u32],
        n_children: usize,
        words: usize,
    ) -> Vec<Vec<u64>> {
        let entropy_arg = entropy
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let script = r#"
import sys
import numpy as np

entropy = [int(part) for part in sys.argv[1].split(",") if part]
n_children = int(sys.argv[2])
words = int(sys.argv[3])
seed = entropy[0] if len(entropy) == 1 else entropy
ss = np.random.SeedSequence(seed)
rng = np.random.Generator(np.random.PCG64DXSM(ss))
for child in rng.spawn(n_children):
    print(",".join(str(int(value)) for value in child.bit_generator.random_raw(words).tolist()))
"#;
        let output = Command::new(oracle_python_bin())
            .arg("-c")
            .arg(script)
            .arg(entropy_arg)
            .arg(n_children.to_string())
            .arg(words.to_string())
            .output()
            .expect("python oracle should launch");
        assert!(
            output.status.success(),
            "NumPy Generator.spawn oracle must succeed"
        );
        String::from_utf8(output.stdout)
            .expect("oracle stdout must be utf-8")
            .lines()
            .map(|line| {
                line.split(',')
                    .filter(|token| !token.is_empty())
                    .map(|token| token.parse::<u64>().expect("oracle raw word"))
                    .collect()
            })
            .collect()
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
    fn bit_generator_algorithms_diverge() {
        let seed = SeedMaterial::U64(42);
        let mut mt = BitGenerator::new(BitGeneratorKind::Mt19937, seed.clone()).unwrap();
        let mut pcg = BitGenerator::new(BitGeneratorKind::Pcg64, seed.clone()).unwrap();
        let mut philox = BitGenerator::new(BitGeneratorKind::Philox, seed.clone()).unwrap();
        let mut sfc64 = BitGenerator::new(BitGeneratorKind::Sfc64, seed.clone()).unwrap();

        let mt_val = mt.next_u64();
        let pcg_val = pcg.next_u64();
        let philox_val = philox.next_u64();
        let sfc64_val = sfc64.next_u64();

        // They should all be different if they use different algorithms
        assert_ne!(mt_val, pcg_val, "MT19937 and PCG64 should differ");
        assert_ne!(pcg_val, philox_val, "PCG64 and Philox should differ");
        assert_ne!(philox_val, sfc64_val, "Philox and SFC64 should differ");
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
    fn seed_sequence_spawn_matches_numpy_seed_12345_reference() {
        let mut root = SeedSequence::new(&[12345]).expect("root");
        let first_children = root.spawn(3).expect("first spawn");
        let second_children = root.spawn(2).expect("second spawn");

        let expected_first: [(&[u32], &[u32]); 3] = [
            (
                &[0],
                &[
                    959_183_449,
                    3_196_577_012,
                    2_719_720_162,
                    1_792_540_688,
                    4_210_643_451,
                    161_689_488,
                    2_223_833_396,
                    3_139_330_310,
                ],
            ),
            (
                &[1],
                &[
                    1_457_248_422,
                    358_904_087,
                    711_457_119,
                    482_272_698,
                    2_884_697_037,
                    68_710_443,
                    77_169_265,
                    2_657_238_038,
                ],
            ),
            (
                &[2],
                &[
                    642_571_064,
                    3_843_934_530,
                    1_770_119_126,
                    1_685_408_148,
                    2_016_457_908,
                    3_630_565_817,
                    935_809_840,
                    858_553_581,
                ],
            ),
        ];
        let expected_second: [(&[u32], &[u32]); 2] = [
            (
                &[3],
                &[
                    3_609_844_797,
                    1_102_929_138,
                    1_660_579_046,
                    2_056_729_905,
                    575_306_158,
                    2_670_191_432,
                    1_184_629_392,
                    2_825_097_045,
                ],
            ),
            (
                &[4],
                &[
                    1_067_841_243,
                    1_074_700_622,
                    4_277_799_263,
                    207_906_991,
                    2_646_164_919,
                    2_269_553_654,
                    598_561_128,
                    1_970_099_910,
                ],
            ),
        ];

        assert_eq!(root.spawn_counter(), 5);
        for (child, (spawn_key, state)) in first_children.iter().zip(expected_first.iter()) {
            assert_eq!(child.spawn_key(), *spawn_key);
            assert_eq!(child.generate_state_u32(8).unwrap(), *state);
        }
        for (child, (spawn_key, state)) in second_children.iter().zip(expected_second.iter()) {
            assert_eq!(child.spawn_key(), *spawn_key);
            assert_eq!(child.generate_state_u32(8).unwrap(), *state);
        }
    }

    #[test]
    fn seed_sequence_spawn_matches_numpy_list_entropy_reference() {
        let mut root = SeedSequence::new(&[1, 2, 3]).expect("root");
        let children = root.spawn(2).expect("spawn");
        let expected: [(&[u32], &[u32]); 2] = [
            (
                &[0],
                &[
                    231_967_115,
                    299_383_237,
                    2_987_783_097,
                    2_908_150_571,
                    834_795_113,
                    3_625_391_018,
                    250_114_644,
                    206_593_069,
                ],
            ),
            (
                &[1],
                &[
                    956_539_761,
                    3_600_538_202,
                    3_398_694_662,
                    2_982_641_888,
                    2_000_443_198,
                    2_257_681_808,
                    3_810_005_645,
                    1_242_537_320,
                ],
            ),
        ];

        for (child, (spawn_key, state)) in children.iter().zip(expected.iter()) {
            assert_eq!(child.spawn_key(), *spawn_key);
            assert_eq!(child.generate_state_u32(8).unwrap(), *state);
        }
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
        invalid.schema_entries.push((String::new(), 1));
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
    fn bit_generator_state_schema_normalizes_whitespace_padded_keys() {
        let mut source =
            BitGenerator::new(BitGeneratorKind::Pcg64, SeedMaterial::U64(1234)).expect("source");
        for _ in 0..16 {
            let _ = source.next_u64();
        }

        let mut state = source.state();
        for (key, _) in &mut state.schema_entries {
            *key = format!(" {key} ");
        }

        let mut restored =
            BitGenerator::new(BitGeneratorKind::Pcg64, SeedMaterial::U64(1)).expect("restored");
        restored
            .set_state(&state)
            .expect("whitespace-padded keys should normalize");

        for _ in 0..32 {
            assert_eq!(source.next_u64(), restored.next_u64());
        }
    }

    #[test]
    fn bit_generator_state_schema_rejects_truncated_philox_restore_state() {
        let mut generator =
            BitGenerator::new(BitGeneratorKind::Philox, SeedMaterial::U64(77)).expect("generator");
        let mut state = generator.state();
        state
            .schema_entries
            .retain(|(key, _)| key != "philox_ctr0" && key != "philox_buf3");

        let err = generator
            .set_state(&state)
            .expect_err("missing philox state entries must fail closed");
        assert_eq!(err.reason_code(), "rng_state_schema_invalid");
    }

    #[test]
    fn bit_generator_state_schema_rejects_invalid_philox_buffer_position() {
        let mut generator =
            BitGenerator::new(BitGeneratorKind::Philox, SeedMaterial::U64(91)).expect("generator");
        let mut state = generator.state();
        for (key, value) in &mut state.schema_entries {
            if key == "philox_pos" {
                *value = 9;
            }
        }

        let err = generator
            .set_state(&state)
            .expect_err("out-of-range philox buffer position must fail closed");
        assert_eq!(err.reason_code(), "rng_state_schema_invalid");
    }

    #[test]
    fn bit_generator_state_schema_rejects_truncated_sfc64_restore_state() {
        let mut generator =
            BitGenerator::new(BitGeneratorKind::Sfc64, SeedMaterial::U64(123)).expect("generator");
        let mut state = generator.state();
        state.schema_entries.retain(|(key, _)| key != "sfc64_s2");

        let err = generator
            .set_state(&state)
            .expect_err("missing sfc64 state entries must fail closed");
        assert_eq!(err.reason_code(), "rng_state_schema_invalid");
    }

    #[test]
    fn bit_generator_state_schema_rejects_invalid_mt19937_position() {
        let mut generator =
            BitGenerator::new(BitGeneratorKind::Mt19937, SeedMaterial::U64(55)).expect("generator");
        let mut state = generator.state();
        for (key, value) in &mut state.schema_entries {
            if key == "mt19937_pos" {
                *value = u64::try_from(super::MT_N + 1).expect("small overflowed position");
            }
        }

        let err = generator
            .set_state(&state)
            .expect_err("out-of-range mt19937 position must fail closed");
        assert_eq!(err.reason_code(), "rng_state_schema_invalid");
    }

    #[test]
    fn bit_generator_state_schema_rejects_oversized_mt19937_state_word() {
        let mut generator =
            BitGenerator::new(BitGeneratorKind::Mt19937, SeedMaterial::U64(77)).expect("generator");
        let mut state = generator.state();
        for (key, value) in &mut state.schema_entries {
            if key == "mt19937_s0" {
                *value = u64::from(u32::MAX) + 1;
                break;
            }
        }

        let err = generator
            .set_state(&state)
            .expect_err("oversized mt19937 word must fail closed");
        assert_eq!(err.reason_code(), "rng_state_schema_invalid");
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
    fn generator_spawn_matches_numpy_pcg64dxsm_reference() {
        let root = SeedSequence::new(&[12345]).expect("seed sequence");
        let mut generator =
            Generator::from_seed_sequence(BitGeneratorKind::Pcg64, &root).expect("generator");
        let mut children = generator.spawn(3).expect("spawn children");
        let expected: [&[u64]; 3] = [
            &[
                2_200_516_532_192_980_437,
                4_161_482_641_097_408_010,
                8_252_466_299_466_180_484,
                13_346_021_809_819_476_580,
                7_797_442_349_405_549_206,
                12_404_377_654_512_563_436,
            ],
            &[
                15_973_472_848_157_220_562,
                6_595_761_104_610_863_534,
                4_781_051_709_214_724_155,
                4_650_127_122_996_825_720,
                607_321_442_220_489_178,
                11_424_307_440_023_841_594,
            ],
            &[
                3_630_127_626_566_264_255,
                14_678_413_081_440_072_679,
                17_487_494_085_757_640_946,
                13_609_397_124_369_525_435,
                17_404_614_185_626_143_740,
                18_263_028_108_381_265_677,
            ],
        ];

        for (index, (child, expected_words)) in children.iter_mut().zip(expected.iter()).enumerate()
        {
            let actual = (0..expected_words.len())
                .map(|_| child.next_u64())
                .collect::<Vec<_>>();
            assert_u64_seq(
                &format!("generator_spawn_child_{index}"),
                &actual,
                expected_words,
            );
        }
    }

    #[test]
    fn seed_sequence_spawn_matches_live_numpy_oracle_when_available() {
        if !numpy_oracle_available() {
            return;
        }

        let mut root = SeedSequence::new(&[12345]).expect("root");
        let mut actual = root
            .spawn(3)
            .expect("first spawn")
            .into_iter()
            .map(|child| {
                (
                    child.spawn_key().to_vec(),
                    child.generate_state_u32(8).expect("state words"),
                )
            })
            .collect::<Vec<_>>();
        actual.extend(
            root.spawn(2)
                .expect("second spawn")
                .into_iter()
                .map(|child| {
                    (
                        child.spawn_key().to_vec(),
                        child.generate_state_u32(8).expect("state words"),
                    )
                }),
        );

        let expected = numpy_oracle_seed_sequence_spawn_words(&[12345], 3, 2);
        assert_eq!(actual, expected);
    }

    #[test]
    fn generator_spawn_matches_live_numpy_oracle_when_available() {
        if !numpy_oracle_available() {
            return;
        }

        let root = SeedSequence::new(&[12345]).expect("seed sequence");
        let mut generator =
            Generator::from_seed_sequence(BitGeneratorKind::Pcg64, &root).expect("generator");
        let mut actual = Vec::new();
        for mut child in generator.spawn(3).expect("spawn children") {
            actual.push((0..6).map(|_| child.next_u64()).collect::<Vec<_>>());
        }

        let expected = numpy_oracle_generator_spawn_raw(&[12345], 3, 6);
        assert_eq!(actual, expected);
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
        let mut restored = DeterministicRng::from_state(u128::from(seed), u128::from(counter));
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
        validate_rng_policy_metadata("strict", "known_compatible")
            .expect("legacy known compatible metadata");
        validate_rng_policy_metadata("hardened", "known_incompatible")
            .expect("legacy known incompatible metadata");
        validate_rng_policy_metadata("strict", "unknown")
            .expect("legacy unknown semantics metadata");
        validate_rng_policy_metadata(" strict ", " known_compatible_low_risk ")
            .expect("whitespace-padded metadata should normalize");

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
        let vals = rng.uniform(2.0, 5.0, 100).unwrap();
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
    fn integers_zero_size_allows_degenerate_bounds() {
        let mut rng = test_generator();
        assert_eq!(rng.integers(5, 5, 0).unwrap(), Vec::<i64>::new());
        assert_eq!(rng.integers(5, 4, 0).unwrap(), Vec::<i64>::new());
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
        let vals = rng.normal(100.0, 1.0, 10000).unwrap();
        let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
        assert!((mean - 100.0).abs() < 0.5, "mean was {mean}");
    }

    #[test]
    fn exponential_positive() {
        let mut rng = test_generator();
        let vals = rng.exponential(1.0, 100).unwrap();
        assert_eq!(vals.len(), 100);
        assert!(vals.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn poisson_nonnegative() {
        let mut rng = test_generator();
        let vals = rng.poisson(3.0, 100).unwrap();
        assert_eq!(vals.len(), 100);
        // Mean should be approximately lambda=3
        let mean: f64 = vals.iter().map(|&v| v as f64).sum::<f64>() / 100.0;
        assert!((mean - 3.0).abs() < 1.0, "mean was {mean}");
    }

    #[test]
    fn poisson_rejects_non_finite_lambda() {
        let mut rng = test_generator();
        assert_eq!(rng.poisson(f64::NAN, 1), Err(RandomError::InvalidParameter));
        assert_eq!(
            rng.poisson(f64::INFINITY, 1),
            Err(RandomError::InvalidParameter)
        );
        assert_eq!(
            rng.poisson(f64::NEG_INFINITY, 1),
            Err(RandomError::InvalidParameter)
        );
    }

    #[test]
    fn binomial_bounded() {
        let mut rng = test_generator();
        let vals = rng.binomial(10, 0.5, 100).unwrap();
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
        sorted.sort_by(f64::total_cmp);
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
        sorted.sort_by(f64::total_cmp);
        assert_eq!(sorted, [1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn permutation_preserves_elements() {
        let mut rng = test_generator();
        let vals = [1.0, 2.0, 3.0, 4.0, 5.0];
        let perm = rng.permutation(&vals).unwrap();
        let mut sorted = perm;
        sorted.sort_by(f64::total_cmp);
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
        let samples = rng.gamma(2.0, 1.0, 1000).unwrap();
        assert_eq!(samples.len(), 1000);
        assert!(samples.iter().all(|&v| v > 0.0));
        let mean: f64 = samples.iter().sum::<f64>() / 1000.0;
        assert!((mean - 2.0).abs() < 0.5); // shape * scale = 2
    }

    #[test]
    fn gamma_small_shape() {
        let mut rng = test_generator();
        let samples = rng.gamma(0.5, 1.0, 100).unwrap();
        assert!(samples.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn beta_basic() {
        let mut rng = test_generator();
        let samples = rng.beta(2.0, 5.0, 1000).unwrap();
        assert_eq!(samples.len(), 1000);
        assert!(samples.iter().all(|&v| (0.0..=1.0).contains(&v)));
        let mean: f64 = samples.iter().sum::<f64>() / 1000.0;
        // E[Beta(a,b)] = a/(a+b) = 2/7 ≈ 0.286
        assert!((mean - 2.0 / 7.0).abs() < 0.1);
    }

    #[test]
    fn geometric_basic() {
        let mut rng = test_generator();
        let samples = rng.geometric(0.5, 1000).unwrap();
        assert_eq!(samples.len(), 1000);
        assert!(samples.iter().all(|&v| v >= 1));
    }

    #[test]
    fn lognormal_basic() {
        let mut rng = test_generator();
        let samples = rng.lognormal(0.0, 1.0, 1000).unwrap();
        assert_eq!(samples.len(), 1000);
        assert!(samples.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn chisquare_basic() {
        let mut rng = test_generator();
        let samples = rng.chisquare(3.0, 1000).unwrap();
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
        let samples = rng.triangular(0.0, 5.0, 10.0, 1000).unwrap();
        assert!(samples.iter().all(|&v| (0.0..=10.0).contains(&v)));
        let mean: f64 = samples.iter().sum::<f64>() / 1000.0;
        // E[tri(a,c,b)] = (a+b+c)/3 = (0+5+10)/3 = 5.0
        assert!((mean - 5.0).abs() < 0.5);
    }

    #[test]
    fn laplace_basic() {
        let mut rng = test_generator();
        let samples = rng.laplace(0.0, 1.0, 1000).unwrap();
        let mean: f64 = samples.iter().sum::<f64>() / 1000.0;
        assert!(mean.abs() < 0.3);
    }

    #[test]
    fn gumbel_basic() {
        let mut rng = test_generator();
        let samples = rng.gumbel(0.0, 1.0, 100).unwrap();
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn weibull_basic() {
        let mut rng = test_generator();
        let samples = rng.weibull(1.5, 1000).unwrap();
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
    fn binomial_supports_trials_above_i64_max() {
        let mut rng = test_generator();
        let n = i64::MAX as u64 + 17;
        let samples = rng.binomial(n, 1.0, 3).unwrap();
        assert_eq!(samples, vec![n, n, n]);
    }

    #[test]
    fn multinomial_supports_trials_above_i64_max() {
        let mut rng = test_generator();
        let n = i64::MAX as u64 + 17;
        let samples = rng.multinomial(n, &[1.0, 0.0], 2);
        assert_eq!(samples, vec![vec![n, 0], vec![n, 0]]);
    }

    #[test]
    fn dirichlet_basic() {
        let mut rng = test_generator();
        let samples = rng.dirichlet(&[1.0, 2.0, 3.0], 10).unwrap();
        assert_eq!(samples.len(), 10);
        for sample in &samples {
            assert_eq!(sample.len(), 3);
            let sum: f64 = sample.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10); // Must sum to 1
            assert!(sample.iter().all(|&v| v >= 0.0));
        }
    }

    #[test]
    fn dirichlet_zero_alpha_preserves_zero_coordinates() {
        let mut rng = test_generator();
        let samples = rng.dirichlet(&[0.5, 0.0, 2.0], 10).unwrap();
        assert_eq!(samples.len(), 10);
        for sample in &samples {
            assert_eq!(sample.len(), 3);
            assert_eq!(sample[1], 0.0);
            let sum: f64 = sample.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn dirichlet_all_zero_alpha_returns_all_zero_samples() {
        let mut rng = test_generator();
        let samples = rng.dirichlet(&[0.0, 0.0], 4).unwrap();
        assert_eq!(samples, vec![vec![0.0, 0.0]; 4]);
    }

    #[test]
    fn multivariate_normal_diag_basic() {
        let mut rng = test_generator();
        let samples = rng
            .multivariate_normal_diag(&[0.0, 5.0], &[1.0, 4.0], 1000)
            .unwrap();
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
        let samples = rng.negative_binomial(5.0, 0.5, 100).unwrap();
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn f_distribution_positive_and_reasonable_mean() {
        let mut rng = test_generator();
        let samples = rng.f_distribution(5.0, 10.0, 5000).unwrap();
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
        let samples = rng.power(2.0, 1000).unwrap();
        assert!(samples.iter().all(|&v| (0.0..1.0).contains(&v)));
    }

    #[test]
    fn vonmises_centered_on_mu() {
        let mut rng = test_generator();
        let mu = 1.5;
        let samples = rng.vonmises(mu, 5.0, 5000).unwrap();
        let mean: f64 = samples.iter().sum::<f64>() / 5000.0;
        assert!((mean - mu).abs() < 0.2, "vonmises mean={mean}");
    }

    #[test]
    fn vonmises_zero_kappa_stays_within_wrapped_range() {
        let mut rng = test_generator();
        let samples = rng.vonmises(10.0, 0.0, 1000).unwrap();
        assert!(
            samples
                .iter()
                .all(|&x| (-std::f64::consts::PI..=std::f64::consts::PI).contains(&x))
        );
    }

    #[test]
    fn rayleigh_positive_and_expected_mean() {
        let mut rng = test_generator();
        let scale = 2.0;
        let samples = rng.rayleigh(scale, 5000).unwrap();
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
        let samples = rng.pareto(3.0, 1000).unwrap();
        assert!(samples.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn logistic_centered_on_loc() {
        let mut rng = test_generator();
        let loc = 3.0;
        let samples = rng.logistic(loc, 1.0, 5000).unwrap();
        let mean: f64 = samples.iter().sum::<f64>() / 5000.0;
        assert!((mean - loc).abs() < 0.2, "logistic mean={mean}");
    }

    #[test]
    fn hypergeometric_bounded_by_ngood_and_nsample() {
        let mut rng = test_generator();
        let samples = rng.hypergeometric(10, 20, 8, 5000).unwrap();
        for &s in &samples {
            assert!(s <= 8, "can't draw more successes than nsample");
            assert!(s <= 10, "can't draw more successes than ngood");
        }
        // Expected mean = nsample * ngood / (ngood+nbad) = 8 * 10/30 ≈ 2.667
        let mean = samples.iter().sum::<u64>() as f64 / 5000.0;
        assert!((mean - 2.667).abs() < 0.2, "hypergeometric mean={mean}");
    }

    #[test]
    fn hypergeometric_rejects_too_large_ngood_nbad() {
        let mut rng = test_generator();
        // ngood >= 10^9 should fail
        assert!(rng.hypergeometric(1_000_000_000, 10, 5, 1).is_err());
        // nbad >= 10^9 should fail
        assert!(rng.hypergeometric(10, 1_000_000_000, 5, 1).is_err());
        // nsample > total should fail
        assert!(rng.hypergeometric(10, 10, 25, 1).is_err());
        // valid parameters should succeed
        assert!(rng.hypergeometric(100, 100, 50, 1).is_ok());
    }

    #[test]
    fn zipf_values_are_positive_integers() {
        let mut rng = test_generator();
        let samples = rng.zipf(2.0, 5000).unwrap();
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
    fn zipf_rejects_nan_and_boundary_parameters() {
        let mut rng = test_generator();
        for invalid in [f64::NAN, 1.0, 0.999_999_999, 0.0, -1.0] {
            assert_eq!(rng.zipf(invalid, 3), Err(RandomError::InvalidParameter));
        }
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
        let samples = rng.logseries(0.5, 5000).unwrap();
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

    #[test]
    fn noncentral_chisquare_positive_values() {
        let mut rng = test_generator();
        let samples = rng.noncentral_chisquare(4.0, 1.0, 1000).unwrap();
        assert_eq!(samples.len(), 1000);
        // All values should be positive
        assert!(samples.iter().all(|&v| v > 0.0));
        // Mean should be df + nonc = 5.0
        let mean: f64 = samples.iter().sum::<f64>() / 1000.0;
        assert!((mean - 5.0).abs() < 1.0, "ncx2 mean={mean}, expected ~5.0");
    }

    #[test]
    fn noncentral_chisquare_reduces_to_central() {
        // With nonc=0, should behave like regular chi-squared
        let mut rng = test_generator();
        let samples = rng.noncentral_chisquare(5.0, 0.0, 2000).unwrap();
        let mean: f64 = samples.iter().sum::<f64>() / 2000.0;
        // Mean of chi²(5) = 5
        assert!((mean - 5.0).abs() < 0.5, "chi2 mean={mean}, expected ~5.0");
    }

    #[test]
    fn noncentral_f_positive_values() {
        let mut rng = test_generator();
        let samples = rng.noncentral_f(5.0, 10.0, 1.0, 1000).unwrap();
        assert_eq!(samples.len(), 1000);
        assert!(samples.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn noncentral_f_mean_approximation() {
        // Mean of ncF(d1, d2, lambda) ≈ d2*(d1+lambda) / (d1*(d2-2)) when d2 > 2
        let mut rng = test_generator();
        let d1 = 5.0;
        let d2 = 20.0;
        let lam = 2.0;
        let samples = rng.noncentral_f(d1, d2, lam, 5000).unwrap();
        let mean: f64 = samples.iter().sum::<f64>() / 5000.0;
        let expected_mean = d2 * (d1 + lam) / (d1 * (d2 - 2.0));
        assert!(
            (mean - expected_mean).abs() < 0.5,
            "ncf mean={mean}, expected ~{expected_mean}"
        );
    }

    #[test]
    fn standard_exponential_mean() {
        let mut rng = test_generator();
        let samples = rng.standard_exponential(5000);
        assert_eq!(samples.len(), 5000);
        assert!(samples.iter().all(|&v| v >= 0.0));
        let mean: f64 = samples.iter().sum::<f64>() / 5000.0;
        assert!(
            (mean - 1.0).abs() < 0.1,
            "standard_exponential mean={mean}, expected ~1.0"
        );
    }

    #[test]
    fn standard_gamma_mean() {
        let mut rng = test_generator();
        let shape_param = 3.0;
        let samples = rng.standard_gamma(shape_param, 5000).unwrap();
        assert_eq!(samples.len(), 5000);
        assert!(samples.iter().all(|&v| v >= 0.0));
        let mean: f64 = samples.iter().sum::<f64>() / 5000.0;
        assert!(
            (mean - shape_param).abs() < 0.3,
            "standard_gamma mean={mean}, expected ~{shape_param}"
        );
    }

    #[test]
    fn bytes_correct_length() {
        let mut rng = test_generator();
        let data = rng.bytes(100);
        assert_eq!(data.len(), 100);
        // At least some variance (not all zeros)
        let distinct: std::collections::HashSet<u8> = data.iter().copied().collect();
        assert!(distinct.len() > 1, "expected diverse byte values");
    }

    #[test]
    fn bytes_empty() {
        let mut rng = test_generator();
        let data = rng.bytes(0);
        assert!(data.is_empty());
    }

    #[test]
    fn bytes_partial_word() {
        let mut rng = test_generator();
        let data = rng.bytes(3);
        assert_eq!(data.len(), 3);
    }

    #[test]
    fn choice_weighted_replace() {
        let mut rng = test_generator();
        let a = [10.0, 20.0, 30.0];
        let p = [0.7, 0.2, 0.1];
        let samples = rng.choice_weighted(&a, 1000, true, &p).unwrap();
        assert_eq!(samples.len(), 1000);
        // Most picks should be 10.0 (p=0.7)
        let count_10 = samples.iter().filter(|&&v| v == 10.0).count();
        assert!(count_10 > 500, "expected majority 10.0, got {count_10}");
    }

    #[test]
    fn choice_weighted_no_replace() {
        let mut rng = test_generator();
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let p = [0.4, 0.3, 0.2, 0.05, 0.05];
        let samples = rng.choice_weighted(&a, 3, false, &p).unwrap();
        assert_eq!(samples.len(), 3);
        // All values should be from the original array
        for &v in &samples {
            assert!(a.contains(&v), "unexpected value {v}");
        }
    }

    #[test]
    fn choice_weighted_rejects_bad_probabilities() {
        let mut rng = test_generator();
        let a = [1.0, 2.0, 3.0];
        // Probabilities don't sum to 1
        let p = [0.5, 0.2, 0.1];
        assert!(rng.choice_weighted(&a, 1, true, &p).is_err());
        // Negative probability
        let p2 = [0.5, 0.7, -0.2];
        assert!(rng.choice_weighted(&a, 1, true, &p2).is_err());
        // Non-finite probabilities must fail closed.
        let p3 = [0.5, f64::NAN, 0.5];
        assert!(rng.choice_weighted(&a, 1, true, &p3).is_err());
        let p4 = [0.5, f64::INFINITY, 0.5];
        assert!(rng.choice_weighted(&a, 1, true, &p4).is_err());
    }

    #[test]
    fn choice_weighted_no_replace_rejects_request_larger_than_positive_support() {
        let mut rng = test_generator();
        let a = [1.0, 2.0, 3.0];
        let p = [1.0, 0.0, 0.0];

        let err = rng
            .choice_weighted(&a, 2, false, &p)
            .expect_err("sampling past non-zero support should fail closed");

        assert_eq!(err, RandomError::InvalidParameter);
    }

    // ── maxwell distribution tests ──

    #[test]
    fn maxwell_all_positive() {
        let mut rng = test_generator();
        let samples = rng.maxwell(1.0, 1000);
        assert_eq!(samples.len(), 1000);
        for &v in &samples {
            assert!(v >= 0.0, "Maxwell samples must be non-negative, got {v}");
        }
    }

    #[test]
    fn maxwell_mean_approx() {
        // E[X] = 2 * scale * sqrt(2/pi) ≈ 1.5958 * scale for scale=1
        let mut rng = test_generator();
        let n = 50_000;
        let samples = rng.maxwell(1.0, n);
        let mean: f64 = samples.iter().sum::<f64>() / n as f64;
        let expected = 2.0 * (2.0 / std::f64::consts::PI).sqrt();
        assert!(
            (mean - expected).abs() < 0.05,
            "Maxwell mean {mean}, expected ~{expected}"
        );
    }

    #[test]
    fn maxwell_scale_parameter() {
        let mut rng1 = test_generator();
        let mut rng2 = test_generator();
        let samples1 = rng1.maxwell(1.0, 1000);
        let samples2 = rng2.maxwell(3.0, 1000);
        let mean1: f64 = samples1.iter().sum::<f64>() / 1000.0;
        let mean2: f64 = samples2.iter().sum::<f64>() / 1000.0;
        // Mean should scale linearly with scale parameter
        assert!(
            (mean2 / mean1 - 3.0).abs() < 0.3,
            "Scale ratio: {}, expected ~3.0",
            mean2 / mean1
        );
    }

    // ── multivariate hypergeometric distribution tests ──

    #[test]
    fn multivariate_hypergeometric_sum_equals_nsample() {
        let mut rng = test_generator();
        let colors = [10, 20, 30];
        let nsample = 15;
        let results = rng
            .multivariate_hypergeometric(&colors, nsample, 100)
            .unwrap();
        assert_eq!(results.len(), 100);
        for sample in &results {
            assert_eq!(sample.len(), 3);
            let total: u64 = sample.iter().sum();
            assert_eq!(
                total, nsample,
                "Each draw must sum to nsample={nsample}, got {total}"
            );
        }
    }

    #[test]
    fn multivariate_hypergeometric_respects_bounds() {
        let mut rng = test_generator();
        let colors = [5, 10, 3];
        let nsample = 8;
        let results = rng
            .multivariate_hypergeometric(&colors, nsample, 200)
            .unwrap();
        for sample in &results {
            for (i, &drawn) in sample.iter().enumerate() {
                assert!(
                    drawn <= colors[i],
                    "Drew {drawn} from group {i} with only {} items",
                    colors[i]
                );
            }
        }
    }

    #[test]
    fn multivariate_hypergeometric_exhaustive() {
        // If nsample >= total, draw all items
        let mut rng = test_generator();
        let colors = [3, 2, 1];
        let nsample = 6; // = total
        let results = rng
            .multivariate_hypergeometric(&colors, nsample, 10)
            .unwrap();
        for sample in &results {
            assert_eq!(sample, &[3, 2, 1]);
        }
    }

    #[test]
    fn multivariate_hypergeometric_rejects_large_total() {
        let mut rng = test_generator();
        // sum(colors) >= 10^9 should fail
        assert!(
            rng.multivariate_hypergeometric(&[500_000_000, 500_000_000], 5, 1)
                .is_err()
        );
        // nsample > total should fail
        assert!(rng.multivariate_hypergeometric(&[5, 10], 20, 1).is_err());
        // valid parameters should succeed
        assert!(rng.multivariate_hypergeometric(&[100, 200], 50, 1).is_ok());
    }

    #[test]
    fn halfnormal_all_nonnegative() {
        let mut rng = test_generator();
        let samples = rng.halfnormal(2.0, 1000);
        assert_eq!(samples.len(), 1000);
        assert!(samples.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn truncated_normal_within_bounds() {
        let mut rng = test_generator();
        let samples = rng.truncated_normal(0.0, 1.0, -1.0, 1.0, 500);
        assert_eq!(samples.len(), 500);
        assert!(samples.iter().all(|&x| (-1.0..=1.0).contains(&x)));
    }

    #[test]
    fn lomax_all_nonnegative() {
        let mut rng = test_generator();
        let samples = rng.lomax(2.0, 1000);
        assert_eq!(samples.len(), 1000);
        assert!(samples.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn levy_all_at_least_loc() {
        let mut rng = test_generator();
        let loc = 1.0;
        let samples = rng.levy(loc, 2.0, 1000);
        assert_eq!(samples.len(), 1000);
        assert!(samples.iter().all(|&x| x >= loc));
    }

    #[test]
    fn integers_endpoint_inclusive() {
        let mut rng = test_generator();
        let samples = rng.integers_endpoint(0, 5, 1000).unwrap();
        assert_eq!(samples.len(), 1000);
        assert!(samples.iter().all(|&x| (0..=5).contains(&x)));
        // With endpoint=True, 5 should be reachable
        assert!(samples.contains(&5));
    }

    #[test]
    fn integers_endpoint_single_value() {
        let mut rng = test_generator();
        let samples = rng.integers_endpoint(3, 3, 100).unwrap();
        assert!(samples.iter().all(|&x| x == 3));
    }

    #[test]
    fn integers_endpoint_zero_size_allows_inverted_bounds() {
        let mut rng = test_generator();
        assert_eq!(rng.integers_endpoint(5, 5, 0).unwrap(), Vec::<i64>::new());
        assert_eq!(rng.integers_endpoint(5, 4, 0).unwrap(), Vec::<i64>::new());
    }

    #[test]
    fn f_alias_matches_f_distribution() {
        let mut rng1 = test_generator();
        let mut rng2 = test_generator();
        let a = rng1.f(2.0, 5.0, 100).unwrap();
        let b = rng2.f_distribution(2.0, 5.0, 100).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn permuted_1d() {
        let mut rng = test_generator();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rng.permuted(&x, &[5], Some(0)).unwrap();
        assert_eq!(result.len(), 5);
        // Same elements, possibly reordered
        let mut sorted = result.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(sorted, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn permuted_invalid_axis() {
        let mut rng = test_generator();
        let x = vec![1.0, 2.0, 3.0];
        assert!(rng.permuted(&x, &[3], Some(1)).is_err());
    }

    // ── NumPy-oracle SeedSequence validation tests ──

    #[test]
    fn seed_sequence_pool_matches_numpy_seed_42() {
        // np.random.SeedSequence(42).pool
        let ss = SeedSequence::new(&[42]).expect("seed");
        assert_eq!(
            ss.pool(),
            &[1_662_858_758, 128_880_814, 1_875_164_712, 753_753_205]
        );
    }

    #[test]
    fn seed_sequence_generate_state_matches_numpy_seed_42() {
        // np.random.SeedSequence(42).generate_state(4, np.uint32)
        let ss = SeedSequence::new(&[42]).expect("seed");
        let state = ss.generate_state_u32(4).expect("state");
        assert_eq!(
            state,
            vec![3_444_837_047, 2_669_555_309, 2_046_530_742, 3_581_440_988]
        );
    }

    #[test]
    fn seed_sequence_generate_state_u64_matches_numpy_seed_42() {
        // np.random.SeedSequence(42).generate_state(4, np.uint64)
        let ss = SeedSequence::new(&[42]).expect("seed");
        let state = ss.generate_state_u64(4).expect("state");
        assert_eq!(
            state,
            vec![
                11_465_652_750_463_011_511,
                15_382_171_918_060_459_190,
                9_018_504_550_953_525_431,
                3_703_499_796_004_394_495
            ]
        );
    }

    #[test]
    fn seed_sequence_pool_matches_numpy_seed_0() {
        // np.random.SeedSequence(0).pool
        let ss = SeedSequence::new(&[0]).expect("seed");
        assert_eq!(
            ss.pool(),
            &[4_265_667_335, 1_328_953_910, 1_320_288_413, 3_365_567_143]
        );
    }

    #[test]
    fn seed_sequence_generate_state_matches_numpy_seed_0() {
        // np.random.SeedSequence(0).generate_state(4, np.uint32)
        let ss = SeedSequence::new(&[0]).expect("seed");
        let state = ss.generate_state_u32(4).expect("state");
        assert_eq!(
            state,
            vec![2_968_811_710, 3_677_149_159, 745_650_761, 2_884_920_346]
        );
    }

    #[test]
    fn seed_sequence_spawn_child0_matches_numpy() {
        // np.random.SeedSequence(42).spawn(2)[0].generate_state(4, np.uint32)
        let mut ss = SeedSequence::new(&[42]).expect("seed");
        let children = ss.spawn(2).expect("spawn");
        let state = children[0].generate_state_u32(4).expect("state");
        assert_eq!(
            state,
            vec![2_684_470_948, 3_757_501_821, 1_691_896_351, 1_126_406_280]
        );
    }

    #[test]
    fn seed_sequence_spawn_child1_matches_numpy() {
        // np.random.SeedSequence(42).spawn(2)[1].generate_state(4, np.uint32)
        let mut ss = SeedSequence::new(&[42]).expect("seed");
        let children = ss.spawn(2).expect("spawn");
        let state = children[1].generate_state_u32(4).expect("state");
        assert_eq!(
            state,
            vec![4_091_952_314, 31_242_083, 366_899_054, 1_794_014_678]
        );
    }

    #[test]
    fn seed_sequence_large_seed_matches_numpy() {
        // np.random.SeedSequence(12345678901234567890)
        // int_to_uint32: 12345678901234567890 >> 0x AB54_A98C_EB1F_0AD2
        // as u32 words: [0xEB1F0AD2, 0xAB54A98C, 2] = [3944651474, 2874329484, 2]
        let big_seed: u128 = 12_345_678_901_234_567_890;
        let w0 = (big_seed & 0xFFFF_FFFF) as u32;
        let w1 = ((big_seed >> 32) & 0xFFFF_FFFF) as u32;
        let w2 = ((big_seed >> 64) & 0xFFFF_FFFF) as u32;
        let mut words = vec![w0, w1];
        if w2 > 0 {
            words.push(w2);
        }
        let ss = SeedSequence::new(&words).expect("seed");
        assert_eq!(
            ss.pool(),
            &[3_436_620_908, 2_934_369_251, 723_718_028, 663_245_011]
        );
        let state = ss.generate_state_u32(4).expect("state");
        assert_eq!(
            state,
            vec![879_039_660, 987_554_174, 2_298_115_603, 1_508_490_435]
        );
    }

    // ── PCG64-DXSM oracle tests ──────────────────────────────────────────

    #[test]
    fn pcg64_dxsm_seed_42_state_matches_numpy() {
        // np.random.PCG64DXSM(42).state
        let rng = Pcg64DxsmRng::from_u64_seed(42).expect("seed");
        let (state, inc) = rng.raw_state();
        assert_eq!(state, 274_674_114_334_540_486_603_088_602_300_644_985_544);
        assert_eq!(inc, 332_724_090_758_049_132_448_979_897_138_935_081_983);
    }

    #[test]
    fn pcg64_dxsm_seed_42_first_10_raw_u64_match_numpy() {
        // np.random.PCG64DXSM(42): raw_u64 values
        let mut rng = Pcg64DxsmRng::from_u64_seed(42).expect("seed");
        let expected: [u64; 10] = [
            12_329_818_062_196_000_797,
            125_530_269_004_142_706,
            12_137_922_674_892_001_441,
            6_848_431_486_601_849_532,
            3_812_337_789_277_959_813,
            3_575_850_668_235_994_163,
            14_001_613_239_349_552_815,
            17_422_012_400_209_074_598,
            18_351_825_975_941_195_139,
            4_178_940_177_906_521_234,
        ];
        for (i, &exp) in expected.iter().enumerate() {
            let got = rng.next_u64();
            assert_eq!(got, exp, "mismatch at index {i}");
        }
    }

    #[test]
    fn pcg64_dxsm_seed_42_state_after_10_matches_numpy() {
        // State after drawing 10 values
        let mut rng = Pcg64DxsmRng::from_u64_seed(42).expect("seed");
        for _ in 0..10 {
            let _ = rng.next_u64();
        }
        let (state, _) = rng.raw_state();
        assert_eq!(state, 288_129_388_986_518_204_794_732_611_361_733_560_794);
    }

    #[test]
    fn pcg64_dxsm_seed_0_state_matches_numpy() {
        // np.random.PCG64DXSM(0).state
        let rng = Pcg64DxsmRng::from_u64_seed(0).expect("seed");
        let (state, inc) = rng.raw_state();
        assert_eq!(state, 35_399_562_948_360_463_058_890_781_895_381_311_971);
        assert_eq!(inc, 87_136_372_517_582_989_555_478_159_403_783_844_777);
    }

    #[test]
    fn pcg64_dxsm_seed_0_first_10_raw_u64_match_numpy() {
        let mut rng = Pcg64DxsmRng::from_u64_seed(0).expect("seed");
        let expected: [u64; 10] = [
            15_672_045_205_194_312_304,
            10_230_625_629_676_741_203,
            1_393_141_542_142_426_128,
            6_186_804_329_743_392_408,
            11_731_200_791_580_184_074,
            593_908_079_603_723_752,
            16_906_657_996_318_114_573,
            4_942_292_156_362_687_787,
            6_282_811_449_723_952_844,
            11_452_040_163_060_472_522,
        ];
        for (i, &exp) in expected.iter().enumerate() {
            let got = rng.next_u64();
            assert_eq!(got, exp, "mismatch at index {i}");
        }
    }

    #[test]
    fn pcg64_dxsm_seed_12345_state_matches_numpy() {
        // np.random.PCG64DXSM(12345).state
        let rng = Pcg64DxsmRng::from_u64_seed(12345).expect("seed");
        let (state, inc) = rng.raw_state();
        assert_eq!(state, 33_261_208_707_367_790_463_622_745_601_869_196_757);
        assert_eq!(inc, 268_209_174_141_567_072_605_526_753_992_732_310_247);
    }

    #[test]
    fn pcg64_dxsm_seed_12345_first_10_raw_u64_match_numpy() {
        let mut rng = Pcg64DxsmRng::from_u64_seed(12345).expect("seed");
        let expected: [u64; 10] = [
            17_193_872_397_121_361_007,
            6_225_879_447_261_284_483,
            4_002_610_872_796_635_837,
            6_506_281_922_641_356_830,
            10_147_648_032_342_742_849,
            16_291_563_099_318_681_940,
            16_386_928_929_934_799_094,
            5_595_252_491_649_564_630,
            8_117_175_315_690_381_675,
            6_073_746_230_066_033_075,
        ];
        for (i, &exp) in expected.iter().enumerate() {
            let got = rng.next_u64();
            assert_eq!(got, exp, "mismatch at index {i}");
        }
    }

    #[test]
    fn pcg64_dxsm_from_seed_sequence_matches_from_u64_seed() {
        // Both paths should produce identical state
        let rng1 = Pcg64DxsmRng::from_u64_seed(42).expect("seed");
        let ss = SeedSequence::new(&[42]).expect("ss");
        let rng2 = Pcg64DxsmRng::from_seed_sequence(&ss).expect("ss_seed");
        assert_eq!(rng1.raw_state(), rng2.raw_state());
    }

    #[test]
    fn pcg64_dxsm_from_raw_state_roundtrip() {
        let mut rng = Pcg64DxsmRng::from_u64_seed(42).expect("seed");
        let (state, inc) = rng.raw_state();
        let v1 = rng.next_u64();
        // Restore and get same value
        let mut rng2 = Pcg64DxsmRng::from_raw_state(state, inc);
        let v2 = rng2.next_u64();
        assert_eq!(v1, v2);
    }

    #[test]
    fn pcg64_dxsm_advance_matches_sequential() {
        let mut rng1 = Pcg64DxsmRng::from_u64_seed(42).expect("seed");
        let mut rng2 = Pcg64DxsmRng::from_u64_seed(42).expect("seed");
        // Advance rng1 by 1000 steps sequentially
        for _ in 0..1000 {
            let _ = rng1.next_u64();
        }
        // Advance rng2 by 1000 steps via jump
        rng2.advance(1000);
        // Both should now produce the same output
        assert_eq!(rng1.next_u64(), rng2.next_u64());
        assert_eq!(rng1.raw_state(), rng2.raw_state());
    }

    #[test]
    fn pcg64_dxsm_fill_u64_matches_sequential() {
        let mut rng1 = Pcg64DxsmRng::from_u64_seed(42).expect("seed");
        let mut rng2 = Pcg64DxsmRng::from_u64_seed(42).expect("seed");
        let filled = rng1.fill_u64(5);
        let sequential: Vec<u64> = (0..5).map(|_| rng2.next_u64()).collect();
        assert_eq!(filled, sequential);
    }

    #[test]
    fn pcg64_dxsm_next_f64_in_unit_interval() {
        let mut rng = Pcg64DxsmRng::from_u64_seed(42).expect("seed");
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v), "f64 out of [0,1): {v}");
        }
    }

    #[test]
    fn pcg64_dxsm_bounded_u64_within_range() {
        let mut rng = Pcg64DxsmRng::from_u64_seed(42).expect("seed");
        for bound in [1, 2, 10, 100, 1000, u64::MAX] {
            let val = rng.bounded_u64(bound).expect("bounded");
            assert!(val < bound, "bounded({bound}) gave {val}");
        }
    }

    #[test]
    fn pcg64_dxsm_bounded_u64_rejects_zero() {
        let mut rng = Pcg64DxsmRng::from_u64_seed(42).expect("seed");
        assert!(rng.bounded_u64(0).is_err());
    }

    // ── MT19937 oracle tests ─────────────────────────────────────────────

    #[test]
    fn mt19937_classic_seed_42_first_20_u32_match_numpy() {
        // np.random.RandomState(42): first 20 randint(0, 2^32)
        let mut rng = Mt19937Rng::from_u32_seed(42);
        let expected: [u32; 20] = [
            1_608_637_542,
            3_421_126_067,
            4_083_286_876,
            787_846_414,
            3_143_890_026,
            3_348_747_335,
            2_571_218_620,
            2_563_451_924,
            670_094_950,
            1_914_837_113,
            669_991_378,
            429_389_014,
            249_467_210,
            1_972_458_954,
            3_720_198_231,
            1_433_267_572,
            2_581_769_315,
            613_608_295,
            3_041_148_567,
            2_795_544_706,
        ];
        for (i, &exp) in expected.iter().enumerate() {
            let got = rng.next_u32();
            assert_eq!(got, exp, "classic seed=42 mismatch at index {i}");
        }
    }

    #[test]
    fn mt19937_classic_seed_42_rand_10_match_numpy() {
        // np.random.RandomState(42).rand(10) uses genrand_res53
        let mut rng = Mt19937Rng::from_u32_seed(42);
        let expected: [f64; 10] = [
            0.374_540_118_847_362_5,
            0.950_714_306_409_916_2,
            0.731_993_941_811_405_1,
            0.598_658_484_197_036_6,
            0.156_018_640_442_436_52,
            0.155_994_520_336_202_65,
            0.058_083_612_168_199_46,
            0.866_176_145_774_935_2,
            0.601_115_011_743_208_8,
            0.708_072_577_796_045_5,
        ];
        for (i, &exp) in expected.iter().enumerate() {
            let got = rng.next_f64();
            assert!(
                (got - exp).abs() < 1e-15,
                "classic seed=42 rand mismatch at {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn mt19937_modern_seed_42_first_20_u32_match_numpy() {
        // np.random.MT19937(42).random_raw() x 20
        let ss = SeedSequence::new(&[42]).expect("ss");
        let mut rng = Mt19937Rng::from_seed_sequence(&ss).expect("seed");
        let expected: [u32; 20] = [
            2_327_846_034,
            3_904_886_566,
            2_661_450_408,
            1_733_955_692,
            246_401_338,
            3_249_250_296,
            3_487_099_619,
            1_498_159_427,
            3_694_075_699,
            3_512_848_995,
            2_695_531_450,
            296_751_540,
            2_928_881_423,
            1_121_898_314,
            2_900_273_415,
            3_780_017_243,
            2_064_865_906,
            852_217_197,
            3_155_620_539,
            3_977_451_023,
        ];
        for (i, &exp) in expected.iter().enumerate() {
            let got = rng.next_u32();
            assert_eq!(got, exp, "modern seed=42 mismatch at index {i}");
        }
    }

    #[test]
    fn mt19937_modern_seed_0_first_20_u32_match_numpy() {
        // np.random.MT19937(0).random_raw() x 20
        let ss = SeedSequence::new(&[0]).expect("ss");
        let mut rng = Mt19937Rng::from_seed_sequence(&ss).expect("seed");
        let expected: [u32; 20] = [
            2_058_676_884,
            2_606_108_953,
            1_230_491_694,
            2_111_045_058,
            95_419_988,
            1_510_044_700,
            2_342_423_583,
            3_663_458_586,
            5_280_294,
            1_692_592_663,
            2_194_070_197,
            3_828_360_469,
            2_174_269_555,
            3_287_063_170,
            2_191_392_552,
            1_383_654_479,
            2_571_795_756,
            4_056_454_040,
            2_939_149_804,
            581_682_926,
        ];
        for (i, &exp) in expected.iter().enumerate() {
            let got = rng.next_u32();
            assert_eq!(got, exp, "modern seed=0 mismatch at index {i}");
        }
    }

    #[test]
    fn mt19937_raw_state_roundtrip() {
        let mut rng = Mt19937Rng::from_u32_seed(42);
        // Advance a bit
        for _ in 0..100 {
            let _ = rng.next_u32();
        }
        let (state, pos) = rng.raw_state();
        let state_owned = state.to_vec();
        let v1 = rng.next_u32();
        // Restore
        let mut rng2 = Mt19937Rng::from_raw_state(&state_owned, pos).expect("restore");
        let v2 = rng2.next_u32();
        assert_eq!(v1, v2);
    }

    #[test]
    fn mt19937_next_f64_in_unit_interval() {
        let mut rng = Mt19937Rng::from_u32_seed(42);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v), "f64 out of [0,1): {v}");
        }
    }

    #[test]
    fn mt19937_fill_u32_matches_sequential() {
        let mut rng1 = Mt19937Rng::from_u32_seed(42);
        let mut rng2 = Mt19937Rng::from_u32_seed(42);
        let filled = rng1.fill_u32(100);
        let sequential: Vec<u32> = (0..100).map(|_| rng2.next_u32()).collect();
        assert_eq!(filled, sequential);
    }

    #[test]
    fn mt19937_bounded_u32_within_range() {
        let mut rng = Mt19937Rng::from_u32_seed(42);
        for bound in [1, 2, 10, 100, 1000, u32::MAX] {
            let val = rng.bounded_u32(bound).expect("bounded");
            assert!(val < bound, "bounded({bound}) gave {val}");
        }
    }

    #[test]
    fn mt19937_bounded_u32_rejects_zero() {
        let mut rng = Mt19937Rng::from_u32_seed(42);
        assert!(rng.bounded_u32(0).is_err());
    }

    #[test]
    fn bit_generator_mt19937_bounded_u64_preserves_large_upper_bounds() {
        let mut rng = BitGenerator::new(BitGeneratorKind::Mt19937, SeedMaterial::U64(987654321))
            .expect("mt19937 bit generator");
        let upper_bound = (1_u64 << 40) + 17;
        let mut saw_value_above_u32 = false;

        for _ in 0..64 {
            let value = rng.bounded_u64(upper_bound).expect("bounded_u64");
            assert!(
                value < upper_bound,
                "value {value} must stay below {upper_bound}"
            );
            if value > u64::from(u32::MAX) {
                saw_value_above_u32 = true;
                break;
            }
        }

        assert!(
            saw_value_above_u32,
            "large upper bounds must not be truncated to 32-bit space"
        );
    }

    #[test]
    fn mt19937_from_raw_state_invalid_length_rejected() {
        let short_state = vec![0u32; 100];
        assert!(Mt19937Rng::from_raw_state(&short_state, 0).is_err());
    }

    #[test]
    fn mt19937_from_raw_state_invalid_pos_rejected() {
        let state = vec![0u32; 624];
        assert!(Mt19937Rng::from_raw_state(&state, 625).is_err());
    }

    // ── Distribution Oracle Tests (NumPy PCG64DXSM(12345) reference) ─────

    /// Helper: assert f64 slices match to full precision (bit-identical).
    fn assert_f64_seq(label: &str, got: &[f64], expected: &[f64]) {
        assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-12,
                "{label}[{i}]: got {g}, expected {e}, diff {}",
                (g - e).abs()
            );
        }
    }

    fn assert_u64_seq(label: &str, got: &[u64], expected: &[u64]) {
        assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(g, e, "{label}[{i}]: got {g}, expected {e}");
        }
    }

    fn oracle_gen() -> Generator {
        Generator::from_pcg64_dxsm(12345).expect("seed")
    }

    #[test]
    fn oracle_random() {
        let mut g = oracle_gen();
        let vals = g.random(10);
        let expected = [
            0.9320816903198763,
            0.3375056011176768,
            0.21698197019501064,
            0.3527062497665462,
            0.5501051021142127,
            0.8831674052732996,
            0.8883371973100436,
            0.3033192453525694,
            0.4400329555858611,
            0.32925844288816175,
        ];
        assert_f64_seq("random", &vals, &expected);
    }

    #[test]
    fn oracle_uniform() {
        let mut g = oracle_gen();
        let vals = g.uniform(2.0, 5.0, 10).unwrap();
        let expected = [
            4.796245070959629,
            3.0125168033530305,
            2.650945910585032,
            3.0581187492996387,
            3.6503153063426383,
            4.649502215819899,
            4.6650115919301305,
            2.909957736057708,
            3.3200988667575833,
            2.987775328664485,
        ];
        assert_f64_seq("uniform", &vals, &expected);
    }

    #[test]
    fn oracle_standard_normal() {
        let mut g = oracle_gen();
        let vals = g.standard_normal(10);
        let expected = [
            0.648970595720087,
            1.089083499221345,
            1.4703372914093853,
            0.6605483574733667,
            -0.44234454827889136,
            -0.08094864935117092,
            0.3087957082495461,
            -0.9638571574827081,
            -0.7254609073102689,
            -1.2118074194203396,
        ];
        assert_f64_seq("standard_normal", &vals, &expected);
    }

    #[test]
    fn oracle_normal() {
        let mut g = oracle_gen();
        let vals = g.normal(5.0, 2.0, 10).unwrap();
        let expected = [
            6.297941191440174,
            7.1781669984426895,
            7.940674582818771,
            6.321096714946734,
            4.115310903442217,
            4.838102701297658,
            5.617591416499092,
            3.0722856850345837,
            3.549078185379462,
            2.5763851611593207,
        ];
        assert_f64_seq("normal", &vals, &expected);
    }

    #[test]
    fn oracle_standard_exponential() {
        let mut g = oracle_gen();
        let vals = g.standard_exponential(10);
        let expected = [
            1.714034816202212,
            0.1304857330479185,
            0.2596812762587198,
            0.04842832490161937,
            2.1864510761713536,
            0.6275885179904133,
            0.5074780068833896,
            1.7187312558502839,
            0.32741607613953083,
            0.27740364655423955,
        ];
        assert_f64_seq("standard_exponential", &vals, &expected);
    }

    #[test]
    fn oracle_exponential() {
        let mut g = oracle_gen();
        let vals = g.exponential(2.5, 10).unwrap();
        let expected = [
            4.28508704050553,
            0.32621433261979627,
            0.6492031906467994,
            0.12107081225404842,
            5.466127690428384,
            1.5689712949760333,
            1.268695017208474,
            4.29682813962571,
            0.8185401903488271,
            0.6935091163855989,
        ];
        assert_f64_seq("exponential", &vals, &expected);
    }

    #[test]
    fn oracle_integers() {
        let mut g = oracle_gen();
        let vals = g.integers(0, 100, 10).expect("integers");
        let expected: Vec<i64> = vec![12, 93, 1, 33, 80, 21, 84, 35, 94, 55];
        assert_eq!(vals, expected, "integers oracle mismatch");
    }

    #[test]
    fn oracle_integers_endpoint() {
        let mut g = oracle_gen();
        let vals = g
            .integers_endpoint(0, 5, 10)
            .expect("integers endpoint should succeed");
        let expected: Vec<i64> = vec![0, 5, 0, 2, 4, 1, 5, 2, 5, 3];
        assert_eq!(vals, expected, "integers endpoint oracle mismatch");
    }

    #[test]
    fn oracle_laplace() {
        let mut g = oracle_gen();
        let vals = g.laplace(0.0, 1.0, 10).unwrap();
        let expected = [
            1.9963024436527586,
            -0.39302599234309016,
            -0.834793834992557,
            -0.3489725415567814,
            0.10559410319107608,
            1.4538660025183756,
            1.4991244586404344,
            -0.4998222325505851,
            -0.12775847525590414,
            -0.41776511533892585,
        ];
        assert_f64_seq("laplace", &vals, &expected);
    }

    #[test]
    fn oracle_gumbel() {
        let mut g = oracle_gen();
        let vals = g.gumbel(0.0, 1.0, 10).unwrap();
        let expected = [
            -0.9893365720157541,
            0.8873554840149915,
            1.408132868136375,
            0.832512543783072,
            0.2247181857117213,
            -0.7640776591098938,
            -0.7849382844008232,
            1.0176924248162385,
            0.5449386697962436,
            0.9178635255883896,
        ];
        assert_f64_seq("gumbel", &vals, &expected);
    }

    #[test]
    fn oracle_logistic() {
        let mut g = oracle_gen();
        let vals = g.logistic(0.0, 1.0, 10).unwrap();
        let expected = [
            2.6191148066328793,
            -0.6744299972282881,
            -1.2833414588669285,
            -0.6071646535080572,
            0.2010953594922383,
            2.0227726736767893,
            2.073867757832975,
            -0.8315414121966367,
            -0.2410283095707554,
            -0.711540918907822,
        ];
        assert_f64_seq("logistic", &vals, &expected);
    }

    #[test]
    fn oracle_lognormal() {
        let mut g = oracle_gen();
        let vals = g.lognormal(0.0, 1.0, 10).unwrap();
        let expected = [
            1.9135699776616077,
            2.9715493968340625,
            4.350702348137479,
            1.9358535831832708,
            0.6425282153220304,
            0.9222410479048919,
            1.3617841408185067,
            0.38141885243216656,
            0.48410139165696314,
            0.2976587986528511,
        ];
        assert_f64_seq("lognormal", &vals, &expected);
    }

    #[test]
    fn oracle_weibull() {
        let mut g = oracle_gen();
        let vals = g.weibull(2.0, 10).unwrap();
        let expected = [
            1.3092115246216756,
            0.3612280900593398,
            0.5095893211780638,
            0.2200643653607266,
            1.4786653022815386,
            0.7922048459776129,
            0.7123749061297637,
            1.3110039114549903,
            0.5722028277975659,
            0.5266912250590848,
        ];
        assert_f64_seq("weibull", &vals, &expected);
    }

    #[test]
    fn oracle_rayleigh() {
        let mut g = oracle_gen();
        let vals = g.rayleigh(1.0, 10).unwrap();
        let expected = [
            1.8515046941351307,
            0.5108536640720481,
            0.7206681292505168,
            0.3112180100881675,
            2.091148524697064,
            1.120346837359229,
            1.0074502537429724,
            1.8540395119038233,
            0.8092169994995543,
            0.7448538736614579,
        ];
        assert_f64_seq("rayleigh", &vals, &expected);
    }

    #[test]
    fn oracle_pareto() {
        let mut g = oracle_gen();
        let vals = g.pareto(3.0, 10).unwrap();
        let expected = [
            0.7706468622745504,
            0.044455027236940385,
            0.09041725464725865,
            0.016273773503058895,
            1.072627291394015,
            0.23268679426427932,
            0.18430882648161737,
            0.773420945135703,
            0.11531702521584737,
            0.09687791167243072,
        ];
        assert_f64_seq("pareto", &vals, &expected);
    }

    #[test]
    fn oracle_power() {
        let mut g = oracle_gen();
        let vals = g.power(5.0, 10).unwrap();
        let expected = [
            0.9610549152494955,
            0.6569121505771804,
            0.7444820463934245,
            0.5431567259656599,
            0.9764540513247876,
            0.8584202987597479,
            0.8317139954376589,
            0.9612527086874139,
            0.7747973537703587,
            0.7531010773400103,
        ];
        assert_f64_seq("power", &vals, &expected);
    }

    #[test]
    fn oracle_triangular() {
        let mut g = oracle_gen();
        let vals = g.triangular(-1.0, 0.0, 1.0, 10).unwrap();
        let expected = [
            0.6314398022571518,
            -0.1784093463072054,
            -0.3412406050840555,
            -0.1601116148361782,
            0.05142749577505956,
            0.5166107267911288,
            0.5274266137625683,
            -0.22112999113771314,
            -0.06188171792053743,
            -0.18850946661324297,
        ];
        assert_f64_seq("triangular", &vals, &expected);
    }

    #[test]
    fn oracle_standard_cauchy() {
        let mut g = oracle_gen();
        let vals = g.standard_cauchy(10);
        let expected = [
            0.5958869050757608,
            2.2259343691860884,
            5.464508077953408,
            -0.3203749703493653,
            0.5986602290793768,
            0.1310774619296665,
            0.01082150363099386,
            2.6511792597164083,
            -14.184214453507638,
            -0.023788035645022024,
        ];
        assert_f64_seq("standard_cauchy", &vals, &expected);
    }

    #[test]
    fn oracle_geometric() {
        let mut g = oracle_gen();
        let vals = g.geometric(0.5, 10).unwrap();
        let expected: Vec<u64> = vec![4, 1, 1, 1, 2, 4, 4, 1, 1, 1];
        assert_u64_seq("geometric", &vals, &expected);
    }

    #[test]
    fn oracle_standard_gamma() {
        let mut g = oracle_gen();
        let vals = g.standard_gamma(3.0, 10).unwrap();
        let expected = [
            3.8730178986382064,
            5.860442600894906,
            2.007580882434013,
            3.203380706017545,
            1.648765619968363,
            2.72481060071336,
            2.661129359850849,
            3.887845119149194,
            13.61058714233652,
            2.6281160439916977,
        ];
        assert_f64_seq("standard_gamma", &vals, &expected);
    }

    #[test]
    fn oracle_gamma() {
        let mut g = oracle_gen();
        let vals = g.gamma(2.0, 3.0, 10).unwrap();
        let expected = [
            7.958138981245406,
            13.130063396284479,
            3.475026335965074,
            6.293849659854425,
            2.6837349405954196,
            5.138163583416871,
            4.986869533913612,
            7.995528829118353,
            35.00081178275254,
            4.908686593272669,
        ];
        assert_f64_seq("gamma", &vals, &expected);
    }

    #[test]
    fn oracle_chisquare() {
        let mut g = oracle_gen();
        let vals = g.chisquare(5.0, 10).unwrap();
        let expected = [
            6.538380773318506,
            10.26311385719701,
            3.1571964002088624,
            5.307454924305685,
            2.5292828476704683,
            4.438236370161119,
            4.323351568092394,
            6.565802771078493,
            25.301773085230973,
            4.263872078966991,
        ];
        assert_f64_seq("chisquare", &vals, &expected);
    }

    #[test]
    fn oracle_beta() {
        let mut g = oracle_gen();
        let vals = g.beta(2.0, 5.0, 10).unwrap();
        let expected = [
            0.23536154859916344,
            0.17754074164377456,
            0.1586686788060632,
            0.21056377129012013,
            0.7165302874074536,
            0.17181153381629066,
            0.31765174585585376,
            0.1379429581403801,
            0.2718552779844278,
            0.10894170824615793,
        ];
        assert_f64_seq("beta", &vals, &expected);
    }

    #[test]
    fn oracle_wald() {
        let mut g = oracle_gen();
        let vals = g.wald(3.0, 2.0, 10);
        let expected = [
            1.3817490656886038,
            0.5946945995263678,
            5.1241047841309015,
            2.0598532428152385,
            1.267450207084615,
            2.87289685181097,
            2.987558305897547,
            6.566074359042968,
            0.14489404392407723,
            3.088435618658114,
        ];
        assert_f64_seq("wald", &vals, &expected);
    }

    #[test]
    fn oracle_logseries() {
        let mut g = oracle_gen();
        let vals = g.logseries(0.6, 10).unwrap();
        let expected: Vec<u64> = vec![1, 1, 2, 1, 1, 2, 1, 2, 2, 1];
        assert_u64_seq("logseries", &vals, &expected);
    }

    #[test]
    fn oracle_permutation() {
        let mut g = oracle_gen();
        let arr: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let perm = g.permutation(&arr).expect("permutation");
        let expected = [4.0, 8.0, 0.0, 2.0, 6.0, 1.0, 5.0, 7.0, 3.0, 9.0];
        assert_f64_seq("permutation", &perm, &expected);
    }

    #[test]
    fn oracle_binomial() {
        let mut g = oracle_gen();
        let vals = g.binomial(10, 0.5, 10).unwrap();
        let expected: Vec<u64> = vec![7, 4, 4, 4, 5, 7, 7, 4, 5, 4];
        assert_u64_seq("binomial", &vals, &expected);
    }

    #[test]
    fn oracle_poisson() {
        let mut g = oracle_gen();
        let vals = g.poisson(3.0, 10).unwrap();
        let expected: Vec<u64> = vec![3, 5, 3, 2, 6, 3, 2, 2, 5, 0];
        assert_u64_seq("poisson", &vals, &expected);
    }

    #[test]
    fn oracle_poisson_large_lambda() {
        let mut g = oracle_gen();
        let vals = g.poisson(20.0, 10).unwrap();
        let expected: Vec<u64> = vec![28, 16, 21, 26, 19, 18, 26, 18, 22, 16];
        assert_u64_seq("poisson_large", &vals, &expected);
    }

    #[test]
    fn oracle_f_distribution() {
        let mut g = oracle_gen();
        let vals = g.f(5.0, 10.0, 10).unwrap();
        let expected = [
            0.7586812017109351,
            0.5883671125201785,
            0.5332160107805164,
            0.6937139034552943,
            5.481784811626646,
            0.5607429508652583,
            1.1690388610620435,
            0.43695159278479795,
            0.9652319482294194,
            0.34215986276681587,
        ];
        assert_f64_seq("f_distribution", &vals, &expected);
    }

    #[test]
    fn oracle_standard_t() {
        let mut g = oracle_gen();
        let vals = g.standard_t(5.0, 10);
        let expected = [
            0.5008333204796186,
            0.8312637071182254,
            0.4799108786097328,
            -1.2862144825001673,
            -0.004070721482134564,
            0.11000227570376948,
            -0.018775083733685572,
            -0.2506624111588833,
            0.3810922270159009,
            -0.8725449195712555,
        ];
        assert_f64_seq("standard_t", &vals, &expected);
    }

    #[test]
    fn oracle_vonmises() {
        let mut g = oracle_gen();
        let vals = g.vonmises(0.0, 4.0, 10).unwrap();
        let expected = [
            0.1765341139961678,
            -0.25614490477469687,
            0.3346666062077488,
            0.036444763154689586,
            0.31733041245381255,
            0.21844747943523402,
            0.1624001118964289,
            -0.17308441650004802,
            0.09840307599485199,
            0.10288417897762203,
        ];
        assert_f64_seq("vonmises", &vals, &expected);
    }

    #[test]
    fn oracle_vonmises_zero_kappa_with_nonzero_mu() {
        let mut g = oracle_gen();
        let vals = g.vonmises(1.5, 0.0, 10).unwrap();
        let expected = [
            2.714849328119167,
            -1.0209824195563921,
            -1.7782547265376234,
            -0.9254739273062167,
            0.3148196414187541,
            2.4075118110033222,
            2.439994572369766,
            -1.2357816278057292,
            -0.3767840523779033,
            -1.0728008429700662,
        ];
        assert_f64_seq("vonmises_zero_kappa", &vals, &expected);
    }

    #[test]
    fn oracle_zipf() {
        let mut g = oracle_gen();
        let vals = g.zipf(2.0, 10).unwrap();
        // NumPy returns integers: [14, 1, 8, 1, 1, 8, 1, 1, 3, 1]
        let expected = [14.0, 1.0, 8.0, 1.0, 1.0, 8.0, 1.0, 1.0, 3.0, 1.0];
        assert_f64_seq("zipf", &vals, &expected);
    }

    #[test]
    fn zipf_parameter_ranges_match_live_numpy_oracle_when_available() {
        if !numpy_oracle_available() {
            return;
        }

        let error_cases = [f64::NAN, 1.0, 0.999_999_999, 0.0, -1.0];
        for a in error_cases {
            let expected = numpy_oracle_zipf_outcome(a, 3);
            assert_eq!(expected, "err:ValueError:a <= 1 or a is NaN");
            let mut g = oracle_gen();
            assert_eq!(g.zipf(a, 3), Err(RandomError::InvalidParameter));
        }

        let expected = numpy_oracle_zipf_outcome(f64::INFINITY, 3);
        assert_eq!(expected, "ok:1,1,1");
        let mut g = oracle_gen();
        assert_eq!(g.zipf(f64::INFINITY, 3).unwrap(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn integers_zero_size_range_validation_matches_live_numpy_oracle_when_available() {
        if !numpy_oracle_available() {
            return;
        }

        assert_eq!(numpy_oracle_integers_outcome(5, 5, 0, false), "ok:");
        assert_eq!(numpy_oracle_integers_outcome(5, 4, 0, false), "ok:");
        assert_eq!(numpy_oracle_integers_outcome(5, 5, 0, true), "ok:");
        assert_eq!(numpy_oracle_integers_outcome(5, 4, 0, true), "ok:");

        let mut g = oracle_gen();
        assert_eq!(g.integers(5, 5, 0).unwrap(), Vec::<i64>::new());
        assert_eq!(g.integers(5, 4, 0).unwrap(), Vec::<i64>::new());
        assert_eq!(g.integers_endpoint(5, 5, 0).unwrap(), Vec::<i64>::new());
        assert_eq!(g.integers_endpoint(5, 4, 0).unwrap(), Vec::<i64>::new());
    }

    #[test]
    fn vonmises_zero_kappa_matches_live_numpy_oracle_when_available() {
        if !numpy_oracle_available() {
            return;
        }

        let expected = numpy_oracle_vonmises(1.5, 0.0, 10);
        let mut g = oracle_gen();
        let actual = g.vonmises(1.5, 0.0, 10).unwrap();
        assert_f64_seq("vonmises_zero_kappa_live", &actual, &expected);
    }

    #[test]
    fn oracle_negative_binomial() {
        let mut g = oracle_gen();
        let vals = g.negative_binomial(5.0, 0.5, 10).unwrap();
        let expected: Vec<u64> = vec![7, 2, 9, 2, 8, 8, 2, 2, 8, 3];
        assert_u64_seq("negative_binomial", &vals, &expected);
    }

    #[test]
    fn oracle_negative_binomial_large_n() {
        // Large n produces gamma values that trigger PTRS Poisson path (lam >= 10)
        let mut g = oracle_gen();
        let vals = g.negative_binomial(50.0, 0.3, 5).unwrap();
        let expected: Vec<u64> = vec![117, 124, 101, 100, 191];
        assert_u64_seq("negative_binomial_large_n", &vals, &expected);
    }

    #[test]
    fn oracle_noncentral_chisquare() {
        let mut g = oracle_gen();
        let vals = g.noncentral_chisquare(5.0, 1.0, 10).unwrap();
        let expected = [
            11.407992320824796,
            6.1909369599500135,
            4.271271486651185,
            2.693449092495373,
            6.067052771714831,
            4.577526920646614,
            3.6679090442005395,
            2.9555669311382315,
            5.611663834516616,
            3.6265665029968828,
        ];
        assert_f64_seq("noncentral_chisquare", &vals, &expected);
    }

    #[test]
    fn oracle_noncentral_f() {
        let mut g = oracle_gen();
        let vals = g.noncentral_f(5.0, 10.0, 1.0, 10).unwrap();
        let expected = [
            1.8270336158811435,
            0.9576523667963563,
            1.0987515131035,
            0.9917493712351674,
            1.2956632147394853,
            1.195433209484817,
            2.7167671099912774,
            0.48625503929084185,
            1.1466715845139002,
            0.39845145064667553,
        ];
        assert_f64_seq("noncentral_f", &vals, &expected);
    }

    #[test]
    fn oracle_hypergeometric() {
        let mut g = oracle_gen();
        let vals = g.hypergeometric(20, 30, 10, 10).unwrap();
        let expected: Vec<u64> = vec![2, 6, 3, 3, 3, 4, 3, 7, 3, 2];
        assert_u64_seq("hypergeometric", &vals, &expected);
    }

    #[test]
    fn oracle_multinomial() {
        let mut g = oracle_gen();
        let vals = g.multinomial(20, &[0.3, 0.5, 0.2], 3);
        let expected: Vec<Vec<u64>> = vec![vec![9, 9, 2], vec![4, 12, 4], vec![6, 8, 6]];
        assert_eq!(
            vals.len(),
            expected.len(),
            "multinomial: sample count mismatch"
        );
        for (i, (got, exp)) in vals.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, exp, "multinomial[{i}]: got {got:?}, expected {exp:?}");
        }
    }

    // NOTE: NumPy defaults to method='svd' for multivariate_normal. Our
    // implementation uses Cholesky. Adding SVD support requires fnp-linalg
    // dependency. This test validates Cholesky-based output; a separate
    // oracle_multivariate_normal_svd test should be added once SVD method is
    // implemented.
    #[test]
    fn oracle_multivariate_normal() {
        let mut g = oracle_gen();
        let cov = [1.0, 0.5, 0.5, 1.0]; // flattened 2x2
        let vals = g.multivariate_normal(&[0.0, 0.0], &cov, 3);
        // Expected values from: rng.multivariate_normal(..., method='cholesky')
        let expected = [
            vec![0.648970595720087, 1.267659275028178],
            vec![1.4703372914093853, 1.3072203037047128],
            vec![-0.44234454827889136, -0.2912758608795984],
        ];
        assert_eq!(vals.len(), expected.len(), "mvn: sample count mismatch");
        for (i, (got, exp)) in vals.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got.len(), exp.len(), "mvn[{i}]: dim mismatch");
            for (j, (&g_val, &e_val)) in got.iter().zip(exp.iter()).enumerate() {
                assert!(
                    (g_val - e_val).abs() < 1e-12,
                    "mvn[{i}][{j}]: got {g_val}, expected {e_val}, diff {}",
                    (g_val - e_val).abs()
                );
            }
        }
    }

    #[test]
    fn oracle_dirichlet() {
        let mut g = oracle_gen();
        let vals = g.dirichlet(&[1.0, 2.0, 3.0], 3).unwrap();
        let expected = [
            vec![
                0.18801570709043375,
                0.38448038254055683,
                0.42750391036900925,
            ],
            vec![0.14347162759248622, 0.47960769869466385, 0.37692067371285],
            vec![0.2047024041741115, 0.38381281192790556, 0.4114847838979829],
        ];
        assert_eq!(
            vals.len(),
            expected.len(),
            "dirichlet: sample count mismatch"
        );
        for (i, (got, exp)) in vals.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got.len(), exp.len(), "dirichlet[{i}]: dim mismatch");
            for (j, (&g_val, &e_val)) in got.iter().zip(exp.iter()).enumerate() {
                assert!(
                    (g_val - e_val).abs() < 1e-12,
                    "dirichlet[{i}][{j}]: got {g_val}, expected {e_val}, diff {}",
                    (g_val - e_val).abs()
                );
            }
        }
    }

    #[test]
    fn oracle_dirichlet_zero_alpha_edges() {
        let mut g = oracle_gen();

        let alpha_zero = g.dirichlet(&[0.0, 1.0], 3).unwrap();
        assert_eq!(alpha_zero, vec![vec![0.0, 1.0]; 3]);

        let trailing_zero = g.dirichlet(&[1.0, 0.0], 3).unwrap();
        assert_eq!(trailing_zero, vec![vec![1.0, 0.0]; 3]);

        let all_zero = g.dirichlet(&[0.0, 0.0], 3).unwrap();
        assert_eq!(all_zero, vec![vec![0.0, 0.0]; 3]);

        let mixed = g.dirichlet(&[0.5, 0.0, 2.0], 3).unwrap();
        let expected = [
            vec![0.6357556543741558, 0.0, 0.36424434562584407],
            vec![0.34837779130894964, 0.0, 0.6516222086910504],
            vec![0.07453613306355736, 0.0, 0.9254638669364426],
        ];
        assert_eq!(
            mixed.len(),
            expected.len(),
            "dirichlet zero-alpha: sample count mismatch"
        );
        for (i, (got, exp)) in mixed.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                got.len(),
                exp.len(),
                "dirichlet zero-alpha[{i}]: dim mismatch"
            );
            for (j, (&g_val, &e_val)) in got.iter().zip(exp.iter()).enumerate() {
                assert!(
                    (g_val - e_val).abs() < 1e-12,
                    "dirichlet zero-alpha[{i}][{j}]: got {g_val}, expected {e_val}, diff {}",
                    (g_val - e_val).abs()
                );
            }
        }
    }

    #[test]
    fn dirichlet_zero_alpha_matches_live_numpy_oracle_when_available() {
        if !numpy_oracle_available() {
            return;
        }

        let cases: [&[f64]; 4] = [&[0.0, 1.0], &[1.0, 0.0], &[0.0, 0.0], &[0.5, 0.0, 2.0]];
        for alpha in cases {
            let mut g = oracle_gen();
            let actual = g.dirichlet(alpha, 3).expect("dirichlet should succeed");
            let expected = numpy_oracle_dirichlet(alpha, 3);
            assert_eq!(actual.len(), expected.len());
            for (row_idx, (got_row, exp_row)) in actual.iter().zip(expected.iter()).enumerate() {
                assert_eq!(
                    got_row.len(),
                    exp_row.len(),
                    "dirichlet live oracle row {row_idx}: dim mismatch"
                );
                for (col_idx, (&got, &exp)) in got_row.iter().zip(exp_row.iter()).enumerate() {
                    assert!(
                        (got - exp).abs() < 1e-12,
                        "dirichlet live oracle[{row_idx}][{col_idx}]: got {got}, expected {exp}"
                    );
                }
            }
        }
    }

    // ── permuted tests ──────────────────────────────────────────────────

    #[test]
    fn permuted_2d_axis0() {
        let mut rng = Generator::from_pcg64_dxsm(42).unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = rng.permuted(&data, &[3, 2], Some(0)).unwrap();
        assert_eq!(result.len(), 6);
        let col0: Vec<f64> = vec![result[0], result[2], result[4]];
        let mut sorted0 = col0.clone();
        sorted0.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(sorted0, vec![1.0, 3.0, 5.0]);
        let col1: Vec<f64> = vec![result[1], result[3], result[5]];
        let mut sorted1 = col1.clone();
        sorted1.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(sorted1, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn permuted_axis_out_of_bounds() {
        let mut rng = Generator::from_pcg64_dxsm(42).unwrap();
        let data = vec![1.0, 2.0, 3.0];
        assert!(rng.permuted(&data, &[3], Some(1)).is_err());
    }

    #[test]
    fn permuted_single_element_axis() {
        let mut rng = Generator::from_pcg64_dxsm(42).unwrap();
        let data = vec![1.0, 2.0, 3.0];
        let result = rng.permuted(&data, &[1, 3], Some(0)).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn permuted_deterministic() {
        let mut rng1 = Generator::from_pcg64_dxsm(999).unwrap();
        let mut rng2 = Generator::from_pcg64_dxsm(999).unwrap();
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let r1 = rng1.permuted(&data, &[5], Some(0)).unwrap();
        let r2 = rng2.permuted(&data, &[5], Some(0)).unwrap();
        assert_eq!(r1, r2, "Same seed should produce identical permutations");
    }

    #[test]
    fn permuted_shape_mismatch() {
        let mut rng = Generator::from_pcg64_dxsm(42).unwrap();
        let data = vec![1.0, 2.0, 3.0];
        assert!(rng.permuted(&data, &[4], Some(0)).is_err());
    }

    #[test]
    fn permuted_axis_none_flattens_before_shuffling() {
        let mut rng = Generator::from_pcg64_dxsm(12345).unwrap();
        let data: Vec<f64> = (0..12).map(|value| value as f64).collect();
        let result = rng.permuted(&data, &[3, 4], None).unwrap();
        let expected = vec![0.0, 10.0, 8.0, 7.0, 5.0, 2.0, 4.0, 6.0, 11.0, 1.0, 3.0, 9.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn permuted_matches_numpy_oracle_for_axis_none_and_named_axes() {
        if !numpy_oracle_available() {
            return;
        }

        let data: Vec<f64> = (0..12).map(|value| value as f64).collect();
        for axis in [None, Some(0), Some(1)] {
            let mut rng = Generator::from_pcg64_dxsm(12345).unwrap();
            let actual = rng.permuted(&data, &[3, 4], axis).unwrap();
            let expected = numpy_oracle_permuted(&[3, 4], axis);
            assert_eq!(actual, expected, "axis={axis:?}");
        }
    }
}
