#![forbid(unsafe_code)]

const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX_CONST1: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX_CONST2: u64 = 0x94D0_49BB_1331_11EB;

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
        DeterministicRng, RANDOM_PACKET_REASON_CODES, RandomError, RandomLogRecord,
        RandomRuntimeMode,
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
