//! Shared helpers for the split per-domain criterion bench binaries.
//!
//! Each per-domain bench binary pulls this in with
//! `#[path = "common/mod.rs"] mod common;`. The helpers are extracted verbatim
//! from the former monolithic `criterion_python_surface.rs` so that each
//! per-domain binary compiles only its own bench functions instead of forcing
//! the whole 200-plus-function monolith to compile just to run one group.
//!
//! `benches/common/mod.rs` is not itself a bench target: Cargo auto-discovers
//! `benches/*.rs` and `benches/*/main.rs`, and this is neither.
//!
//! Each per-domain bench binary that `#[path]`-includes this module uses only a
//! subset of the shared helpers, so items unused by a given binary are expected;
//! `#![allow(dead_code)]` keeps those honest, cross-binary-unused helpers from
//! tripping the `-D warnings` gate in the binaries that do not call them.
#![allow(dead_code)]

use criterion::Criterion;
use pyo3::types::PyAnyMethods;
use pyo3::{Bound, Py, PyAny, PyResult, Python};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::hint::black_box;
use std::path::Path;
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const CONTRACT_ROUNDS: usize = 41;
const CONTRACT_MIN_OF: usize = 3;
const CONTRACT_BOOTSTRAP_RESAMPLES: usize = 4_096;
/// Sanctioned busy-host schedule, ported from franken_networkx's
/// `balanced_square_ab.py`: every round interleaves the two arms as
/// `A B B A A B B A`. Each arm therefore occupies symmetric positions in both
/// halves of the round, so shared-host drift affects both arms equally.
const BALANCED_SQUARE: [bool; 8] = [true, false, false, true, true, false, false, true];
const DUAL_NULL_WARMUP_ROUNDS: usize = 4;

/// One phase of the dual-null schedule. `selects_incumbent` maps a
/// balanced-square slot to the arm that runs in it; the runner drives every
/// phase through this predicate and the slot accounting below counts the same
/// predicate, so the executed schedule and the lifecycle expectation cannot
/// drift apart.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DualNullPhase {
    IncumbentNull,
    CandidateNull,
    Effect,
}

impl DualNullPhase {
    fn selects_incumbent(self, slot_is_a: bool) -> bool {
        match self {
            Self::IncumbentNull => true,
            Self::CandidateNull => false,
            Self::Effect => slot_is_a,
        }
    }

    /// Slots per round this phase spends on one arm.
    fn arm_slots(self, incumbent: bool) -> usize {
        BALANCED_SQUARE
            .iter()
            .filter(|&&slot_is_a| self.selects_incumbent(slot_is_a) == incumbent)
            .count()
    }

    /// Short name for provenance lines.
    fn label(self) -> &'static str {
        match self {
            Self::IncumbentNull => "incumbent_null",
            Self::CandidateNull => "candidate_null",
            Self::Effect => "effect",
        }
    }

    fn checksum_mismatch_message(self) -> &'static str {
        match self {
            Self::IncumbentNull => {
                "incumbent/incumbent null arms produced different output checksums"
            }
            Self::CandidateNull => {
                "candidate/candidate null arms produced different output checksums"
            }
            Self::Effect => "incumbent/candidate arms produced different output checksums",
        }
    }

    /// Suffix appended to the caller's row name when this phase reports. The
    /// published row names are load-bearing evidence, so they stay attached to
    /// the phase that produces them rather than to a call site.
    fn row_suffix(self) -> &'static str {
        match self {
            Self::IncumbentNull => "_null_incumbent_aa",
            Self::CandidateNull => "_null_candidate_aa",
            Self::Effect => "_effect_incumbent_over_candidate",
        }
    }
}

/// The dual-null schedule in execution order: an incumbent A/A, a candidate
/// A/A, then the interleaved effect.
const DUAL_NULL_PHASES: [DualNullPhase; 3] = [
    DualNullPhase::IncumbentNull,
    DualNullPhase::CandidateNull,
    DualNullPhase::Effect,
];

/// Slots one arm occupies in a single pass over every dual-null phase. An arm
/// runs all eight slots of *its own* A/A phase, none of the other arm's, and
/// four of the eight interleaved effect slots.
fn dual_null_arm_slots_per_round(incumbent: bool) -> usize {
    DUAL_NULL_PHASES
        .iter()
        .map(|phase| phase.arm_slots(incumbent))
        .sum()
}

/// Number of arm-local materializations made by the dual-null schedule.
/// Lifecycle probes use this to prove their counters cover their own A/A phase
/// and their half of the effect phase, including balanced warm-up rounds.
pub fn dual_null_observation_count_per_arm(rounds: usize, min_of: usize) -> usize {
    let incumbent_slots = dual_null_arm_slots_per_round(true);
    assert_eq!(
        incumbent_slots,
        dual_null_arm_slots_per_round(false),
        "the dual-null schedule must spend the same number of slots on each arm",
    );
    incumbent_slots * (DUAL_NULL_WARMUP_ROUNDS + rounds) * min_of
}

#[derive(Clone, Copy)]
pub struct ContractPairStats {
    pub ratio_median: f64,
    pub ratio_ci_low: f64,
    pub ratio_ci_high: f64,
    pub ratio_cv_pct: f64,
    pub ratio_mad: f64,
    pub arm_a_median_ns: f64,
    pub arm_b_median_ns: f64,
    pub checksum: u64,
}

#[derive(Clone, Copy)]
pub struct ContractObservation {
    pub elapsed: Duration,
    pub checksum: u64,
}

/// Time both arms in one balanced-square round and reduce each arm's four
/// slots to its median. The checksum check is deliberately per-arm: an A/A
/// null runs the same closure in both arms, while an effect pair has its
/// cross-arm check at the caller where equivalent results are required.
/// Which CPU is this thread running on right now? (`deadlock-audit-ei9jz`)
///
/// Field 39 of `/proc/self/stat` is the last CPU the task executed on. The `comm`
/// field can itself contain spaces and parentheses, so the parse starts after the
/// LAST `)` - splitting the whole line on whitespace is the classic bug here.
///
/// Safe `std::fs` only: no `libc`, no `sched_getcpu`, no new dependency, and nothing
/// that would need `unsafe`.
fn observed_cpu() -> Option<u32> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let tail = &stat[stat.rfind(')')? + 1..];
    // After `comm`, the first field is `state`, which is field 3. So the CPU (field
    // 39) is the 37th whitespace-separated token of this tail, i.e. index 36.
    tail.split_whitespace().nth(36)?.parse::<u32>().ok()
}

/// The CURRENT clock of one specific core, in MHz.
///
/// This is the field that matters: cores on this 64-core part run at DIFFERENT clocks
/// SIMULTANEOUSLY - 4089 MHz on cpu0/cpu5 against 2733 MHz on cpu63, read in the same
/// instant. A machine-wide mean is therefore not a property of the timed arm, and a
/// ratio whose two arms sat on cores at different clocks is partly a FREQUENCY ratio.
fn cpu_mhz(cpu: u32) -> Option<f64> {
    let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq");
    let khz = std::fs::read_to_string(path).ok()?;
    Some(khz.trim().parse::<f64>().ok()? / 1000.0)
}

/// Per-ARM frequency provenance for one contract phase.
///
/// `same_core` is the field that decides whether a ratio is trustworthy: if the two
/// arms ran on cores at different clocks, the ratio carries a frequency component that
/// no amount of interleaving removes.
#[derive(Clone, Copy, Default)]
pub struct ArmCpuWitness {
    pub a_cpu: Option<u32>,
    pub b_cpu: Option<u32>,
    pub a_mhz_mean: f64,
    pub b_mhz_mean: f64,
    pub a_samples: usize,
    pub b_samples: usize,
}

impl ArmCpuWitness {
    /// True when both arms were observed on the SAME core, so they necessarily shared
    /// a clock domain and the ratio cannot be a frequency artefact.
    pub fn same_core(&self) -> bool {
        match (self.a_cpu, self.b_cpu) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }

    /// How far apart the two arms' observed clocks sat, as a ratio >= 1.0. A row whose
    /// arms differ here is reporting a frequency ratio in disguise to that extent.
    pub fn mhz_spread(&self) -> f64 {
        if self.a_mhz_mean <= 0.0 || self.b_mhz_mean <= 0.0 {
            return 1.0;
        }
        let (lo, hi) = if self.a_mhz_mean < self.b_mhz_mean {
            (self.a_mhz_mean, self.b_mhz_mean)
        } else {
            (self.b_mhz_mean, self.a_mhz_mean)
        };
        hi / lo
    }

    /// Combine another round's witness into this phase-level one, weighting each arm's
    /// mean clock by how many slots contributed to it.
    fn merge(&mut self, other: &ArmCpuWitness) {
        if other.a_samples > 0 {
            let total = self.a_samples + other.a_samples;
            self.a_mhz_mean = (self.a_mhz_mean * self.a_samples as f64
                + other.a_mhz_mean * other.a_samples as f64)
                / total as f64;
            self.a_samples = total;
            self.a_cpu = other.a_cpu;
        }
        if other.b_samples > 0 {
            let total = self.b_samples + other.b_samples;
            self.b_mhz_mean = (self.b_mhz_mean * self.b_samples as f64
                + other.b_mhz_mean * other.b_samples as f64)
                / total as f64;
            self.b_samples = total;
            self.b_cpu = other.b_cpu;
        }
    }

    fn record(&mut self, is_a: bool) {
        let Some(cpu) = observed_cpu() else {
            return;
        };
        let mhz = cpu_mhz(cpu).unwrap_or(0.0);
        if is_a {
            self.a_cpu = Some(cpu);
            self.a_mhz_mean =
                (self.a_mhz_mean * self.a_samples as f64 + mhz) / (self.a_samples as f64 + 1.0);
            self.a_samples += 1;
        } else {
            self.b_cpu = Some(cpu);
            self.b_mhz_mean =
                (self.b_mhz_mean * self.b_samples as f64 + mhz) / (self.b_samples as f64 + 1.0);
            self.b_samples += 1;
        }
    }
}

fn balanced_square_round_with<F>(observe: F) -> (ContractObservation, ContractObservation)
where
    F: FnMut(bool) -> ContractObservation,
{
    let (a, b, _witness) = balanced_square_round_witnessed(observe);
    (a, b)
}

/// The same balanced square, additionally witnessing WHICH CORE each arm ran on and at
/// WHAT CLOCK (`deadlock-audit-ei9jz`).
///
/// The frequency is sampled immediately after each slot's timed work, so it reflects the
/// core that just executed that arm rather than a machine-wide average. Sampling is two
/// small `/proc` and `/sys` reads and happens OUTSIDE the timed region.
fn balanced_square_round_witnessed<F>(
    mut observe: F,
) -> (ContractObservation, ContractObservation, ArmCpuWitness)
where
    F: FnMut(bool) -> ContractObservation,
{
    let mut a_slots = Vec::with_capacity(BALANCED_SQUARE.len() / 2);
    let mut b_slots = Vec::with_capacity(BALANCED_SQUARE.len() / 2);
    let mut witness = ArmCpuWitness::default();
    for is_a in BALANCED_SQUARE {
        let observation = observe(is_a);
        witness.record(is_a);
        if is_a {
            a_slots.push(observation);
        } else {
            b_slots.push(observation);
        }
    }

    let reduce = |slots: &[ContractObservation], arm: &str| {
        let checksum = slots[0].checksum;
        assert!(
            slots
                .iter()
                .all(|observation| observation.checksum == checksum),
            "balanced-square {arm} slots produced different output checksums"
        );
        let mut elapsed_ns = slots
            .iter()
            .map(|observation| observation.elapsed.as_secs_f64() * 1.0e9)
            .collect::<Vec<_>>();
        ContractObservation {
            elapsed: Duration::from_secs_f64(median(&mut elapsed_ns) / 1.0e9),
            checksum,
        }
    };
    (reduce(&a_slots, "A"), reduce(&b_slots, "B"), witness)
}

#[derive(Clone, Copy)]
struct CpuTicks {
    total: u64,
    idle: u64,
}

fn file_identity(path: &Path) -> Option<(String, usize)> {
    let Ok(bytes) = std::fs::read(path) else {
        return None;
    };
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let digest = hasher.finalize();
    let mut hash = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut hash, "{byte:02x}").expect("writing to String cannot fail");
    }
    Some((hash, bytes.len()))
}

fn self_identity() -> String {
    let Ok(path) = std::env::current_exe() else {
        return "unavailable".to_string();
    };
    let Some((hash, byte_len)) = file_identity(&path) else {
        return "unavailable".to_string();
    };
    format!("{} ({} bytes) {}", hash, byte_len, path.display())
}

fn read_cpu_ticks() -> Result<BTreeMap<usize, CpuTicks>, String> {
    let content = std::fs::read_to_string("/proc/stat")
        .map_err(|error| format!("read /proc/stat: {error}"))?;
    let mut cpus = BTreeMap::new();
    for line in content.lines() {
        let mut fields = line.split_ascii_whitespace();
        let Some(label) = fields.next() else {
            continue;
        };
        let Some(suffix) = label.strip_prefix("cpu") else {
            continue;
        };
        if suffix.is_empty() || !suffix.bytes().all(|byte| byte.is_ascii_digit()) {
            continue;
        }
        let cpu = suffix
            .parse::<usize>()
            .map_err(|error| format!("parse CPU index {suffix}: {error}"))?;
        let ticks = fields
            .map(str::parse::<u64>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("parse /proc/stat ticks for cpu{cpu}: {error}"))?;
        if ticks.len() < 5 {
            return Err(format!("cpu{cpu} /proc/stat row is too short"));
        }
        cpus.insert(
            cpu,
            CpuTicks {
                total: ticks.iter().copied().sum(),
                idle: ticks[3].saturating_add(ticks[4]),
            },
        );
    }
    if cpus.is_empty() {
        return Err("no per-CPU rows in /proc/stat".to_owned());
    }
    Ok(cpus)
}

fn parse_cpu_list(value: &str) -> Result<BTreeSet<usize>, String> {
    let mut cpus = BTreeSet::new();
    for range in value.trim().split(',').filter(|part| !part.is_empty()) {
        if let Some((start, end)) = range.split_once('-') {
            let start = start
                .parse::<usize>()
                .map_err(|error| format!("parse CPU range start {start}: {error}"))?;
            let end = end
                .parse::<usize>()
                .map_err(|error| format!("parse CPU range end {end}: {error}"))?;
            if start > end {
                return Err(format!("descending CPU range: {range}"));
            }
            cpus.extend(start..=end);
        } else {
            cpus.insert(
                range
                    .parse::<usize>()
                    .map_err(|error| format!("parse CPU index {range}: {error}"))?,
            );
        }
    }
    if cpus.is_empty() {
        return Err("CPU list is empty".to_owned());
    }
    Ok(cpus)
}

fn self_allowed_cpus() -> Result<BTreeSet<usize>, String> {
    let status = std::fs::read_to_string("/proc/self/status")
        .map_err(|error| format!("read /proc/self/status: {error}"))?;
    let allowed = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:").map(str::trim))
        .ok_or_else(|| "Cpus_allowed_list missing from /proc/self/status".to_owned())?;
    parse_cpu_list(allowed)
}

fn format_cpu_set(cpus: &BTreeSet<usize>) -> String {
    cpus.iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(":")
}

fn read_trimmed(path: impl AsRef<Path>) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn sanitize_provenance_field(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn host_name() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(|| read_trimmed("/etc/hostname"))
        .unwrap_or_else(|| "unavailable".to_owned())
}

fn cpu_model_name() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|contents| {
            contents.lines().find_map(|line| {
                let (label, value) = line.split_once(':')?;
                matches!(label.trim(), "model name" | "Hardware").then(|| value.trim().to_owned())
            })
        })
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unavailable".to_owned())
}

fn physical_core_count(online: &BTreeSet<usize>) -> Option<usize> {
    let mut cores = BTreeSet::new();
    for cpu in online {
        let topology = format!("/sys/devices/system/cpu/cpu{cpu}/topology");
        let package = read_trimmed(Path::new(&topology).join("physical_package_id"))?;
        let core = read_trimmed(Path::new(&topology).join("core_id"))?;
        cores.insert((package, core));
    }
    (!cores.is_empty()).then_some(cores.len())
}

fn cpu_governors(online: &BTreeSet<usize>) -> String {
    let governors = online
        .iter()
        .filter_map(|cpu| {
            read_trimmed(format!(
                "/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
            ))
        })
        .collect::<BTreeSet<_>>();
    if governors.is_empty() {
        "unavailable".to_owned()
    } else {
        governors.into_iter().collect::<Vec<_>>().join(":")
    }
}

fn report_host_execution_provenance() {
    let online = read_cpu_ticks()
        .expect("host topology requires readable per-CPU /proc/stat rows")
        .into_keys()
        .collect::<BTreeSet<_>>();
    let allowed = self_allowed_cpus().expect("host topology requires process affinity");
    let physical_cores = physical_core_count(&online)
        .map_or_else(|| "unavailable".to_owned(), |count| count.to_string());
    println!(
        "HOST_BASELINE host={} cpu_model={} physical_cores={physical_cores} \
         logical_threads={} online_cpus={} allowed_logical_threads={} allowed_cpus={} \
         governor={}",
        sanitize_provenance_field(&host_name()),
        sanitize_provenance_field(&cpu_model_name()),
        online.len(),
        format_cpu_set(&online),
        allowed.len(),
        format_cpu_set(&allowed),
        sanitize_provenance_field(&cpu_governors(&online)),
    );
    println!(
        "THREAD_CONFIGURATION rayon_pool_threads={} RAYON_NUM_THREADS={} \
         OPENBLAS_NUM_THREADS={} OMP_NUM_THREADS={} MKL_NUM_THREADS={}",
        rayon::current_num_threads(),
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".to_owned()),
        std::env::var("OPENBLAS_NUM_THREADS").unwrap_or_else(|_| "unset".to_owned()),
        std::env::var("OMP_NUM_THREADS").unwrap_or_else(|_| "unset".to_owned()),
        std::env::var("MKL_NUM_THREADS").unwrap_or_else(|_| "unset".to_owned()),
    );
    #[cfg(target_arch = "x86_64")]
    println!(
        "ISA_BASELINE target_arch=x86_64 compile_sse2={} compile_avx2={} \
         runtime_sse2={} runtime_avx={} runtime_avx2={} runtime_f16c={} \
         runtime_fma={} runtime_avx512f={} runtime_avx512bw={}",
        cfg!(target_feature = "sse2"),
        cfg!(target_feature = "avx2"),
        std::arch::is_x86_feature_detected!("sse2"),
        std::arch::is_x86_feature_detected!("avx"),
        std::arch::is_x86_feature_detected!("avx2"),
        std::arch::is_x86_feature_detected!("f16c"),
        std::arch::is_x86_feature_detected!("fma"),
        std::arch::is_x86_feature_detected!("avx512f"),
        std::arch::is_x86_feature_detected!("avx512bw"),
    );
    #[cfg(not(target_arch = "x86_64"))]
    println!(
        "ISA_BASELINE target_arch={} runtime_features=not_x86_64",
        std::env::consts::ARCH,
    );
    std::io::Write::flush(&mut std::io::stdout()).expect("flushing stdout cannot fail");
}

#[derive(Clone)]
struct ThreadCpuTicks {
    name: String,
    ticks: u64,
}

fn process_thread_cpu_ticks() -> Result<BTreeMap<u32, ThreadCpuTicks>, String> {
    let entries = std::fs::read_dir("/proc/self/task")
        .map_err(|error| format!("read /proc/self/task: {error}"))?;
    let mut threads = BTreeMap::new();
    for entry in entries {
        let entry = entry.map_err(|error| format!("read /proc/self/task entry: {error}"))?;
        let Some(tid) = entry
            .file_name()
            .to_str()
            .and_then(|value| value.parse::<u32>().ok())
        else {
            continue;
        };
        let stat = match std::fs::read_to_string(entry.path().join("stat")) {
            Ok(stat) => stat,
            Err(_) => continue,
        };
        let Some(open) = stat.find('(') else {
            continue;
        };
        let Some(close) = stat.rfind(')') else {
            continue;
        };
        if close <= open {
            continue;
        }
        let fields = stat[close + 1..]
            .split_ascii_whitespace()
            .collect::<Vec<_>>();
        let Some(user_ticks) = fields.get(11).and_then(|value| value.parse::<u64>().ok()) else {
            continue;
        };
        let Some(system_ticks) = fields.get(12).and_then(|value| value.parse::<u64>().ok()) else {
            continue;
        };
        threads.insert(
            tid,
            ThreadCpuTicks {
                name: stat[open + 1..close].to_owned(),
                ticks: user_ticks.saturating_add(system_ticks),
            },
        );
    }
    if threads.is_empty() {
        Err("/proc/self/task exposed no readable thread CPU counters".to_owned())
    } else {
        Ok(threads)
    }
}

/// Count the OS threads that actually accrued CPU time while repeatedly
/// executing one arm outside the timed contract. Configured pool width is
/// useful context, but it is not evidence that every configured worker ran.
pub fn report_observed_thread_activity<F>(
    row: &str,
    arm: &str,
    repetitions: usize,
    mut operation: F,
) where
    F: FnMut(),
{
    assert!(
        repetitions > 0,
        "thread-activity repetitions must be non-zero"
    );
    operation();
    let before =
        process_thread_cpu_ticks().expect("thread-activity baseline requires /proc task counters");
    for _ in 0..repetitions {
        operation();
    }
    let after =
        process_thread_cpu_ticks().expect("thread-activity result requires /proc task counters");
    let mut active = Vec::new();
    let mut total_ticks = 0_u64;
    for (tid, final_ticks) in &after {
        let initial_ticks = before.get(tid).map_or(0, |initial| initial.ticks);
        let delta = final_ticks.ticks.saturating_sub(initial_ticks);
        if delta > 0 {
            total_ticks = total_ticks.saturating_add(delta);
            active.push(format!(
                "{tid}:{}:{delta}",
                sanitize_provenance_field(&final_ticks.name)
            ));
        }
    }
    assert!(
        !active.is_empty(),
        "{row} {arm}: no thread accrued a scheduler CPU tick across {repetitions} repetitions"
    );
    println!(
        "OBSERVED_THREAD_ACTIVITY row={row} arm={arm} repetitions={repetitions} \
         threads_actually_used={} total_cpu_ticks={total_ticks} \
         tid_name_delta_ticks={}",
        active.len(),
        active.join(","),
    );
    std::io::Write::flush(&mut std::io::stdout()).expect("flushing stdout cannot fail");
}

pub fn bench_invocation_id() -> &'static str {
    static INVOCATION_ID: OnceLock<String> = OnceLock::new();
    INVOCATION_ID.get_or_init(|| {
        let unix_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("bench clock must be after Unix epoch")
            .as_nanos();
        format!("{unix_nanos:032x}-{:08x}", std::process::id())
    })
}

/// Prove that a Python benchmark's incumbent arm is the named live NumPy
/// callable and bind it to NumPy's executing compiled core. `callable_name`
/// may be a dotted public path such as `char.upper`.
///
/// The hash is computed by the executing bench process, not by an adjacent
/// shell step, and the invocation id is shared with every row in this process.
/// Checking object identity against `numpy.<callable_name>` also covers ufuncs,
/// which do not consistently expose a Python `__module__` attribute.
pub fn report_numpy_incumbent_identity(
    py: Python<'_>,
    callable_name: &str,
    numpy_callable: &Bound<'_, PyAny>,
) {
    let numpy = py.import("numpy").expect("numpy incumbent");
    let numpy_name = numpy
        .getattr("__name__")
        .expect("numpy __name__")
        .extract::<String>()
        .expect("numpy __name__ string");
    assert_eq!(numpy_name, "numpy", "incumbent module is not numpy");
    let numpy_version = numpy
        .getattr("__version__")
        .expect("numpy __version__")
        .extract::<String>()
        .expect("numpy __version__ string");
    let mut expected_callable = numpy.clone().into_any();
    for component in callable_name.split('.') {
        assert!(
            !component.is_empty(),
            "NumPy incumbent callable path contains an empty component"
        );
        expected_callable = expected_callable
            .getattr(component)
            .unwrap_or_else(|_| panic!("named NumPy incumbent callable numpy.{callable_name}"));
    }
    assert!(
        expected_callable.is(numpy_callable),
        "incumbent callable is not numpy.{callable_name}"
    );
    let callable_module = numpy_callable
        .getattr("__module__")
        .and_then(|module| module.extract::<String>())
        .unwrap_or_else(|_| "numpy.<compiled-ufunc>".to_owned());
    assert!(
        callable_module.starts_with("numpy"),
        "incumbent callable is not defined under numpy: {callable_module}"
    );

    // Public NumPy dispatchers and ufuncs ultimately execute through this
    // compiled core. Hash the ELF shared object rather than a package
    // __init__.py or Python wrapper source.
    let core_module = py
        .import("numpy._core._multiarray_umath")
        .expect("numpy compiled core module");
    let artifact_path = core_module
        .getattr("__file__")
        .expect("numpy core __file__")
        .extract::<String>()
        .expect("numpy core path string");
    let artifact_path = Path::new(&artifact_path);
    let (artifact_sha256, artifact_bytes) =
        file_identity(artifact_path).expect("hash numpy core artifact");
    println!(
        "INCUMBENT_IDENTITY arm=numpy.{callable_name} numpy_version={numpy_version} \
         callable_module={callable_module} invocation_id={} \
         artifact_sha256={artifact_sha256} artifact_bytes={artifact_bytes} \
         artifact_path={} dispatch_assert=passed",
        bench_invocation_id(),
        artifact_path.display(),
    );
    std::io::Write::flush(&mut std::io::stdout()).expect("flushing stdout cannot fail");
}

/// Pin a bound method on an exact live `numpy.random.Generator` instance to the
/// compiled `_generator` extension that implements it. A bound C method is not
/// object-identical to the descriptor stored on `Generator`, so this checks its
/// receiver, name, qualified name, module, and descriptor rebind rather than
/// weakening the ordinary incumbent identity helper.
pub fn report_numpy_generator_method_identity(
    py: Python<'_>,
    method_name: &str,
    generator: &Bound<'_, PyAny>,
    bound_method: &Bound<'_, PyAny>,
) {
    assert!(
        !method_name.is_empty() && !method_name.contains('.'),
        "Generator method identity requires one non-empty attribute name"
    );
    let numpy = py.import("numpy").expect("numpy incumbent");
    let numpy_name = numpy
        .getattr("__name__")
        .expect("numpy __name__")
        .extract::<String>()
        .expect("numpy __name__ string");
    assert_eq!(numpy_name, "numpy", "incumbent module is not numpy");
    let numpy_version = numpy
        .getattr("__version__")
        .expect("numpy __version__")
        .extract::<String>()
        .expect("numpy __version__ string");
    let numpy_random = numpy.getattr("random").expect("numpy.random");
    let generator_type = numpy_random
        .getattr("Generator")
        .expect("numpy.random.Generator");
    assert!(
        generator.is_exact_instance(&generator_type),
        "incumbent receiver is not an exact numpy.random.Generator"
    );

    let receiver = bound_method
        .getattr("__self__")
        .expect("bound Generator method __self__");
    assert!(
        receiver.is(generator),
        "bound Generator method receiver differs from the measured incumbent"
    );
    let reported_name = bound_method
        .getattr("__name__")
        .expect("bound Generator method __name__")
        .extract::<String>()
        .expect("bound Generator method name string");
    assert_eq!(
        reported_name, method_name,
        "bound Generator method name differs from the requested incumbent"
    );
    let callable_module = bound_method
        .getattr("__module__")
        .expect("bound Generator method __module__")
        .extract::<String>()
        .expect("bound Generator method module string");
    assert_eq!(
        callable_module, "numpy.random._generator",
        "bound incumbent method is not implemented by numpy.random._generator"
    );
    let qualified_name = bound_method
        .getattr("__qualname__")
        .expect("bound Generator method __qualname__")
        .extract::<String>()
        .expect("bound Generator method qualified-name string");
    assert_eq!(
        qualified_name,
        format!("Generator.{method_name}"),
        "bound Generator method qualified name differs"
    );

    let descriptor = generator_type
        .getattr(method_name)
        .unwrap_or_else(|_| panic!("numpy.random.Generator.{method_name} descriptor"));
    let rebound = descriptor
        .call_method1("__get__", (generator, &generator_type))
        .expect("rebind numpy.random.Generator descriptor");
    assert!(
        rebound
            .eq(bound_method)
            .expect("compare rebound numpy.random.Generator method"),
        "measured bound method does not match the public Generator descriptor"
    );

    let generator_module = py
        .import("numpy.random._generator")
        .expect("numpy.random compiled Generator module");
    let artifact_path = generator_module
        .getattr("__file__")
        .expect("numpy.random._generator __file__")
        .extract::<String>()
        .expect("numpy.random._generator path string");
    let artifact_path = Path::new(&artifact_path);
    let (artifact_sha256, artifact_bytes) =
        file_identity(artifact_path).expect("hash numpy.random Generator artifact");
    println!(
        "INCUMBENT_IDENTITY arm=numpy.random.Generator.{method_name} \
         numpy_version={numpy_version} callable_module={callable_module} \
         qualified_name={qualified_name} invocation_id={} \
         artifact_sha256={artifact_sha256} artifact_bytes={artifact_bytes} \
         artifact_path={} receiver_assert=passed descriptor_rebind_assert=passed",
        bench_invocation_id(),
        artifact_path.display(),
    );
    std::io::Write::flush(&mut std::io::stdout()).expect("flushing stdout cannot fail");
}

/// Loadtxt-specific compatibility wrapper retained for the existing benches.
pub fn report_numpy_loadtxt_incumbent_identity(py: Python<'_>, numpy_loadtxt: &Bound<'_, PyAny>) {
    report_numpy_incumbent_identity(py, "loadtxt", numpy_loadtxt);
}

/// State the measured topology before an end-to-end incumbent comparison.
/// Both call paths must be independently complete; a shared timed component is
/// a maintenance self-comparison, not campaign output.
pub fn report_incumbent_topology(candidate: &str, incumbent: &str) {
    assert!(
        candidate.starts_with("fnp.") && incumbent.starts_with("numpy."),
        "incumbent topology must name public fnp and numpy entry points"
    );
    assert_ne!(
        candidate, incumbent,
        "candidate and incumbent entry points must be distinct"
    );
    println!(
        "INCUMBENT_TOPOLOGY candidate={candidate} incumbent={incumbent} \
         shared_timed_component=none"
    );
    std::io::Write::flush(&mut std::io::stdout()).expect("flushing stdout cannot fail");
}

/// State the measured topology when the candidate arm is known to route one or
/// more of its public stages back into the incumbent's compiled code. The
/// shared component must be named: an end-to-end job comparison stays honest
/// only if a delegated stage is disclosed rather than implied absent. A shared
/// stage cannot inflate the candidate's ratio (identical code runs in both
/// arms), so a disclosed-delegation row is conservative, not flattering.
pub fn report_incumbent_topology_with_shared_component(
    candidate: &str,
    incumbent: &str,
    shared_timed_component: &str,
) {
    assert!(
        candidate.starts_with("fnp.") && incumbent.starts_with("numpy."),
        "incumbent topology must name public fnp and numpy entry points"
    );
    assert_ne!(
        candidate, incumbent,
        "candidate and incumbent entry points must be distinct"
    );
    assert!(
        !shared_timed_component.is_empty() && shared_timed_component != "none",
        "this helper exists to name a real shared component; \
         use report_incumbent_topology for a fully independent pair"
    );
    println!(
        "INCUMBENT_TOPOLOGY candidate={candidate} incumbent={incumbent} \
         shared_timed_component={shared_timed_component} \
         shared_component_direction=conservative_for_candidate"
    );
    std::io::Write::flush(&mut std::io::stdout()).expect("flushing stdout cannot fail");
}

/// This module's source, embedded by the compiler from the file it actually
/// compiled. Hashing it at runtime is the one link from a running binary back to
/// the SOURCE it was built from — `bench_elf_sha256` identifies the binary, and
/// a binary built from stale source has a perfectly good, entirely fresh-looking
/// sha of its own.
///
/// Measured 2026-08-15 (`deadlock-audit-rko39`): two runs of one command against
/// one unchanged working tree produced binaries `59423d99...` on vmi1227854 and
/// `522ce318...` on vmi1153651, and the second printed the PRE-CHANGE row format
/// after genuinely recompiling for 18m41s. Every provenance field it printed —
/// ELF sha, host, threads, ISA, invocation id — was present and self-consistent.
/// Nothing said it had built the wrong source. This field says it.
///
/// COVERAGE, stated because a half-covering check invites false confidence: this
/// fingerprints the shared contract module only — the balanced square, the
/// median-CI gate, the null accounting, every published row's arithmetic. It
/// does NOT cover a change confined to an individual bench file, which is what
/// the observed incident actually was. Extending it needs a build script that
/// hashes the whole `benches/` tree; per-file `include_str!` would rebuild every
/// bench binary whenever any bench source changed, which on a fleet where these
/// builds run 10-20 minutes is a worse trade than the gap it closes.
const HARNESS_CONTRACT_SOURCE: &str = include_str!("mod.rs");

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut hash = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut hash, "{byte:02x}").expect("writing to String cannot fail");
    }
    hash
}

fn harness_contract_source_sha256() -> String {
    sha256_hex(HARNESS_CONTRACT_SOURCE.as_bytes())
}

/// Whether the contract source COMPILED INTO this binary is the contract source
/// sitting in the tree the binary is about to be measured in. `embedded` comes
/// from `include_str!` at compile time and `on_disk` is read at run time, so a
/// binary built from stale source reports `false` here while every other
/// provenance field it prints still looks perfectly fresh.
fn harness_source_matches(embedded: &str, on_disk: &str) -> bool {
    embedded == on_disk
}

/// `Some(true|false)` when the contract source can be read back from the tree,
/// `None` when it cannot (the source may legitimately be absent beside a copied
/// ELF, which is a different situation from a mismatch and must not be reported
/// as one).
fn harness_contract_source_matches_disk() -> Option<bool> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/benches/common/mod.rs");
    let on_disk = std::fs::read_to_string(path).ok()?;
    Some(harness_source_matches(HARNESS_CONTRACT_SOURCE, &on_disk))
}

/// `bench_source` is the calling bench binary's OWN source, embedded by the
/// compiler at its call site via `include_str!("<that file>.rs")`. Passing it
/// closes the gap the contract-module fingerprint alone leaves: a stale remote
/// build of a per-bench file is otherwise invisible, because `common/mod.rs`
/// hashes identical while the bench body differs (`deadlock-audit-cvnmf`).
///
/// A file that includes ITSELF is tracked by cargo as a dependency of that binary
/// only, so this does NOT couple the 28 bench binaries to each other — editing one
/// bench rebuilds that bench. (A build script hashing the whole `benches/` tree
/// would have coupled them, which is why that route was rejected.)
fn report_bench_identity(bench_source: Option<&str>) {
    println!("bench_elf_sha256={}", self_identity());
    println!(
        "harness_contract_source_sha256={} harness_contract_source_bytes={} \
         harness_source_matches_disk={} covers={}",
        harness_contract_source_sha256(),
        HARNESS_CONTRACT_SOURCE.len(),
        match harness_contract_source_matches_disk() {
            Some(true) => "true",
            Some(false) => "false_BINARY_BUILT_FROM_DIFFERENT_SOURCE",
            None => "unknown_source_not_readable",
        },
        if bench_source.is_some() {
            "common/mod.rs+bench_file"
        } else {
            "common/mod.rs_only"
        },
    );
    match bench_source {
        Some(source) => println!(
            "bench_file_source_sha256={} bench_file_source_bytes={}",
            sha256_hex(source.as_bytes()),
            source.len(),
        ),
        // Not silent: a binary that declines to fingerprint its own body is
        // exactly the case this field exists to make visible.
        None => println!(
            "bench_file_source_sha256=unreported \
             bench_file_source_bytes=0 \
             stale_bench_body_would_be_undetectable=true"
        ),
    }
    println!("bench_invocation_id={}", bench_invocation_id());
    report_host_execution_provenance();
    std::io::Write::flush(&mut std::io::stdout()).expect("flushing stdout cannot fail");
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() & 1 == 0 {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}

fn bootstrap_median_ci(values: &[f64]) -> (f64, f64) {
    let mut state = 0x4d59_5df4_d0f3_3173u64 ^ values.len() as u64;
    let mut resample = vec![0.0; values.len()];
    let mut medians = Vec::with_capacity(CONTRACT_BOOTSTRAP_RESAMPLES);
    for _ in 0..CONTRACT_BOOTSTRAP_RESAMPLES {
        for slot in &mut resample {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *slot = values[(state as usize) % values.len()];
        }
        medians.push(median(&mut resample));
    }
    medians.sort_by(f64::total_cmp);
    let low = CONTRACT_BOOTSTRAP_RESAMPLES * 25 / 1_000;
    let high = (CONTRACT_BOOTSTRAP_RESAMPLES * 975 / 1_000).min(CONTRACT_BOOTSTRAP_RESAMPLES - 1);
    (medians[low], medians[high])
}

fn mix_checksum(state: u64, value: u64) -> u64 {
    state.rotate_left(11) ^ value.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0xa076_1d64_78bd_642f
}

fn min_observation<F>(operation: &mut F) -> ContractObservation
where
    F: FnMut() -> ContractObservation,
{
    min_observation_with(operation, CONTRACT_MIN_OF)
}

fn min_observation_with<F>(operation: &mut F, min_of: usize) -> ContractObservation
where
    F: FnMut() -> ContractObservation,
{
    assert!(min_of >= 1, "contract min-of count must be non-zero");
    let mut best = operation();
    let mut checksum = best.checksum;
    for _ in 1..min_of {
        let observation = operation();
        checksum = mix_checksum(checksum, observation.checksum);
        if observation.elapsed < best.elapsed {
            best.elapsed = observation.elapsed;
        }
    }
    ContractObservation {
        elapsed: best.elapsed,
        checksum,
    }
}

fn contract_pair_stats(arm_a: &[f64], arm_b: &[f64], checksum: u64) -> ContractPairStats {
    let ratios = arm_a
        .iter()
        .zip(arm_b)
        .map(|(a, b)| a / b)
        .collect::<Vec<_>>();
    let mut arm_a_sorted = arm_a.to_vec();
    let mut arm_b_sorted = arm_b.to_vec();
    let mut ratio_sorted = ratios.clone();
    let arm_a_median_ns = median(&mut arm_a_sorted);
    let arm_b_median_ns = median(&mut arm_b_sorted);
    let ratio_median = median(&mut ratio_sorted);
    let (ratio_ci_low, ratio_ci_high) = bootstrap_median_ci(&ratios);
    let ratio_mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
    let ratio_variance = ratios
        .iter()
        .map(|ratio| {
            let delta = ratio - ratio_mean;
            delta * delta
        })
        .sum::<f64>()
        / (ratios.len() - 1) as f64;
    let mut deviations = ratios
        .iter()
        .map(|ratio| (ratio - ratio_median).abs())
        .collect::<Vec<_>>();

    ContractPairStats {
        ratio_median,
        ratio_ci_low,
        ratio_ci_high,
        ratio_cv_pct: ratio_variance.sqrt() * 100.0 / ratio_mean,
        ratio_mad: median(&mut deviations),
        arm_a_median_ns,
        arm_b_median_ns,
        checksum,
    }
}

fn report_contract_pair(row: &str, stats: ContractPairStats) {
    report_contract_pair_with_sampling(row, stats, CONTRACT_ROUNDS, CONTRACT_MIN_OF);
}

fn report_contract_pair_with_sampling(
    row: &str,
    stats: ContractPairStats,
    rounds: usize,
    min_of: usize,
) {
    println!(
        "PAIRED row={row} rounds={rounds} min_of={min_of} \
         arm_a_median_ms={:.6} arm_b_median_ms={:.6} ratio_median={:.6} \
         ratio_median_ci95=[{:.6},{:.6}] ratio_cv_pct={:.3} ratio_mad={:.6} checksum={:016x}",
        stats.arm_a_median_ns / 1_000_000.0,
        stats.arm_b_median_ns / 1_000_000.0,
        stats.ratio_median,
        stats.ratio_ci_low,
        stats.ratio_ci_high,
        stats.ratio_cv_pct,
        stats.ratio_mad,
        stats.checksum,
    );
}

fn contract_gate_verdict(
    effect: ContractPairStats,
    null_ci_low: f64,
    null_ci_high: f64,
    controlling_half_width: f64,
) -> &'static str {
    let required_delta = (2.0 * controlling_half_width).max(0.01);
    let effect_delta = effect.ratio_median - 1.0;
    let effect_ci_above_one = effect.ratio_ci_low > 1.0;
    let effect_ci_below_one = effect.ratio_ci_high < 1.0;
    let above_null_envelope = effect.ratio_median > null_ci_high;
    let below_null_envelope = effect.ratio_median < null_ci_low;
    if effect_ci_above_one && above_null_envelope && effect_delta >= required_delta {
        "DECIDABLE_WIN"
    } else if effect_ci_below_one && below_null_envelope && effect_delta <= -required_delta {
        "DECIDABLE_REGRESSION"
    } else {
        "UNDECIDED"
    }
}

fn report_contract_gate(row: &str, effect: ContractPairStats, null: ContractPairStats) {
    let null_half_width = (null.ratio_ci_low - 1.0)
        .abs()
        .max((null.ratio_ci_high - 1.0).abs());
    let required_delta = (2.0 * null_half_width).max(0.01);
    let verdict = contract_gate_verdict(
        effect,
        null.ratio_ci_low,
        null.ratio_ci_high,
        null_half_width,
    );
    println!(
        "MEDIAN_CI_GATE row={row} verdict={verdict} effect_ratio={:.6} \
         effect_ci95=[{:.6},{:.6}] effect_ci_excludes_one={} \
         null_ci95=[{:.6},{:.6}] null_half_width={:.6} required_2x_delta={required_delta:.6} \
         null_straddles_unity={} null_bias={:.6} \
         cv_is_provenance_only=true",
        effect.ratio_median,
        effect.ratio_ci_low,
        effect.ratio_ci_high,
        effect.ratio_ci_low > 1.0 || effect.ratio_ci_high < 1.0,
        null.ratio_ci_low,
        null.ratio_ci_high,
        null_half_width,
        null_straddles_unity(null),
        null_bias(null),
    );
}

fn null_half_width(null: ContractPairStats) -> f64 {
    (null.ratio_ci_low - 1.0)
        .abs()
        .max((null.ratio_ci_high - 1.0).abs())
}

/// Does this A/A null's 95% CI actually contain 1.0? (`deadlock-audit-7xcq2`)
///
/// An A/A null runs the SAME arm against ITSELF, so its ratio must be 1.0. A null
/// whose CI EXCLUDES unity means the harness is systematically favouring one
/// position in the schedule, and the effect measured in that same position inherits
/// the bias.
///
/// WHAT THIS IS *NOT* FOR, because I first justified it with a mechanism that turned
/// out to be false and the correction belongs where the next reader will see it: I
/// claimed the gate was BLIND to bias, on the theory that `contract_gate_verdict`
/// scales its threshold by the null's half-width and so a biased-but-tight null would
/// slip through with a small threshold. That is wrong. `null_half_width` measures the
/// distance from **1.0** to the furthest CI bound — not the width of the interval — so
/// an offset null produces a LARGER half-width and a STRICTER threshold. The observed
/// lookup-ceiling row proves it arithmetically: half_width 0.034364 gave
/// required_2x_delta 0.068729, exactly 2x, and that is larger than a centred-but-wider
/// null would have produced. The gate already charges bias into the threshold.
///
/// WHAT IT IS FOR: VISIBILITY. A biased null is a real anomaly — the harness favoured
/// one position in the schedule — and today nothing in the emitted line says so. You
/// can only notice by reading the CI bounds yourself and doing the comparison in your
/// head, which is exactly how both of today's instances got banked with hand-written
/// prose caveats instead of a machine-readable flag. It also makes the population
/// auditable: across the whole ledger there are 116 prose `ci95=` mentions but ZERO
/// pasted null rows, so nobody can currently count how often this happens.
///
/// Two instances were observed on thinkstation1 on 2026-08-16, both of which passed
/// the gate legitimately (their effects cleared the inflated threshold): the
/// lookup-ceiling `empty` pair at 1.030928 ci95=[1.030405,1.034364] (3.1% bias), and
/// the parallel divide incumbent null at 1.006625 ci95=[1.001361,1.029159] (0.66%
/// bias, under an effect clearing its threshold by only 1.2x — the thin one).
fn null_straddles_unity(null: ContractPairStats) -> bool {
    null.ratio_ci_low <= 1.0 && null.ratio_ci_high >= 1.0
}

/// How far the null's CENTRE sits from unity — the quantity the half-width misses.
fn null_bias(null: ContractPairStats) -> f64 {
    (null.ratio_median - 1.0).abs()
}

pub fn dual_null_contract_verdict(
    effect: ContractPairStats,
    incumbent_null: ContractPairStats,
    candidate_null: ContractPairStats,
) -> &'static str {
    let controlling_half_width =
        null_half_width(incumbent_null).max(null_half_width(candidate_null));
    let null_ci_low = incumbent_null.ratio_ci_low.min(candidate_null.ratio_ci_low);
    let null_ci_high = incumbent_null
        .ratio_ci_high
        .max(candidate_null.ratio_ci_high);
    contract_gate_verdict(effect, null_ci_low, null_ci_high, controlling_half_width)
}

fn report_dual_null_contract_gate(
    row: &str,
    effect: ContractPairStats,
    incumbent_null: ContractPairStats,
    candidate_null: ContractPairStats,
) {
    let incumbent_half_width = null_half_width(incumbent_null);
    let candidate_half_width = null_half_width(candidate_null);
    let controlling_half_width = incumbent_half_width.max(candidate_half_width);
    let required_delta = (2.0 * controlling_half_width).max(0.01);
    let verdict = dual_null_contract_verdict(effect, incumbent_null, candidate_null);
    println!(
        "MEDIAN_CI_GATE row={row} verdict={verdict} effect_ratio={:.6} \
         effect_ci95=[{:.6},{:.6}] effect_ci_excludes_one={} \
         incumbent_null_ci95=[{:.6},{:.6}] candidate_null_ci95=[{:.6},{:.6}] \
         incumbent_null_half_width={incumbent_half_width:.6} \
         candidate_null_half_width={candidate_half_width:.6} \
         controlling_null_half_width={controlling_half_width:.6} \
         required_2x_delta={required_delta:.6} both_nulls=true \
         incumbent_null_straddles_unity={} candidate_null_straddles_unity={} \
         incumbent_null_bias={:.6} candidate_null_bias={:.6} \
         cv_is_provenance_only=true",
        effect.ratio_median,
        effect.ratio_ci_low,
        effect.ratio_ci_high,
        effect.ratio_ci_low > 1.0 || effect.ratio_ci_high < 1.0,
        incumbent_null.ratio_ci_low,
        incumbent_null.ratio_ci_high,
        candidate_null.ratio_ci_low,
        candidate_null.ratio_ci_high,
        null_straddles_unity(incumbent_null),
        null_straddles_unity(candidate_null),
        null_bias(incumbent_null),
        null_bias(candidate_null),
    );
}

/// Proves the stale-source detector actually discriminates, using a pair that
/// differs the way the observed incident differed: one line of the harness
/// changed, everything else identical. A detector that compared lengths, or the
/// first N bytes, or nothing at all, passes every "does it return a bool" check
/// and reports `true` here — which is precisely how a stale-source run gets
/// banked as a good one (`deadlock-audit-rko39`).
fn verify_stale_source_detector() {
    let shipped = "share = kernel_ns / end_to_end_median;\n";
    let stale = "share = kernel_ns / end_to_end_bestof;\n";
    assert!(
        harness_source_matches(shipped, shipped),
        "identical source must report as matching"
    );
    assert!(
        !harness_source_matches(shipped, stale),
        "a one-token difference in the middle of the source must report as a MISMATCH"
    );
    // The pair is deliberately the SAME LENGTH: a detector that compared sizes
    // — which is exactly what rsync-style staleness checks do — reports these as
    // identical. Keep them equal-length or this stops testing the hard case.
    assert_eq!(
        shipped.len(),
        stale.len(),
        "the fixture pair must stay the same length or it stops testing the hard case"
    );
    assert!(
        !harness_source_matches("", shipped),
        "an empty read must not be mistaken for a match"
    );
    // The digest the identity line prints must be a function of the content.
    assert_ne!(sha256_hex(shipped.as_bytes()), sha256_hex(stale.as_bytes()));
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "sha256_hex must be SHA-256, not some other digest with the same shape"
    );
}

/// Proves the straddle detector DISCRIMINATES, and proves the inversion it exists to
/// expose is real (`deadlock-audit-7xcq2`).
///
/// This lives in the startup self-check rather than a `#[test]`, because these are
/// benchmark binaries: `cargo test` does not run their `#[test]`s, so a unit test here
/// would be dormant and prove nothing on any real run. `verify_contract_gate_semantics`
/// executes on every bench invocation, before Criterion is constructed.
fn verify_null_straddle_detector() {
    // Every field is written out: `ContractPairStats` does not derive `Default`, and
    // adding a derive to a shared public struct to shorten a self-check would be a
    // worse trade than four extra lines here.
    let stats = |median: f64, low: f64, high: f64| ContractPairStats {
        ratio_median: median,
        ratio_ci_low: low,
        ratio_ci_high: high,
        ratio_cv_pct: 0.0,
        ratio_mad: 0.0,
        arm_a_median_ns: 0.0,
        arm_b_median_ns: 0.0,
        checksum: 0,
    };

    // POSITIVE: a healthy null contains unity and reports ~zero bias.
    let healthy = stats(1.000332, 0.998696, 1.000817);
    assert!(
        null_straddles_unity(healthy),
        "a null whose CI contains 1.0 must report straddles_unity=true",
    );
    assert!(
        null_bias(healthy) < 0.001,
        "a healthy null's bias must be near zero, got {:.6}",
        null_bias(healthy),
    );

    // NEGATIVE CASE — the whole point of this check, and the shape a naive
    // implementation gets wrong. Both of these were MEASURED on thinkstation1 on
    // 2026-08-16 and both PASSED the live gate. A detector that tested the null's
    // half-width, or its median alone, or that compared against a tolerance instead
    // of against the interval, reports these as fine.
    let biased_tight = stats(1.030928, 1.030405, 1.034364);
    assert!(
        !null_straddles_unity(biased_tight),
        "the observed lookup-ceiling null [1.030405,1.034364] EXCLUDES 1.0 and must be \
         reported as not straddling — this is the case the live gate misses",
    );
    let biased_wide = stats(1.006625, 1.001361, 1.029159);
    assert!(
        !null_straddles_unity(biased_wide),
        "the observed parallel-divide incumbent null [1.001361,1.029159] EXCLUDES 1.0 \
         and must be reported as not straddling",
    );

    // PINS THE CORRECTION, so the false mechanism cannot come back. `null_half_width`
    // is the distance from 1.0 to the furthest bound, NOT the interval's width, so a
    // biased null yields a LARGER half-width and therefore a STRICTER threshold — the
    // gate charges bias rather than missing it. If someone later "fixes" half-width to
    // mean interval width, this assertion fails and they are forced to re-read why
    // these fields exist, instead of silently recreating the hole I wrongly claimed
    // was already there.
    let healthy_wide = stats(0.999575, 0.976336, 1.032376);
    assert!(
        null_half_width(biased_tight) > null_half_width(healthy_wide),
        "expected the BIASED null to have the LARGER half-width ({:.6} vs {:.6}): \
         half_width is measured from 1.0, so offset inflates it and the gate becomes \
         stricter, not more permissive",
        null_half_width(biased_tight),
        null_half_width(healthy_wide),
    );
    // And the reason a separate bias field still earns its place: half-width conflates
    // offset with spread, so these two nulls have nearly equal half-widths for
    // completely different reasons — one is centred and noisy, the other is tight and
    // displaced. Only `null_bias` tells them apart in the emitted line.
    assert!(
        null_bias(biased_tight) > null_bias(healthy_wide),
        "expected the biased-tight null to have the LARGER bias ({:.6} vs {:.6})",
        null_bias(biased_tight),
        null_bias(healthy_wide),
    );

    // Boundary: a CI that touches unity exactly still straddles. Using a strict
    // comparison here would flag every null that happens to land on the boundary,
    // which is a false alarm rather than a finding.
    assert!(
        null_straddles_unity(stats(1.0005, 1.000000, 1.001000)),
        "a CI whose bound is exactly 1.0 must count as straddling",
    );
}

fn verify_contract_gate_semantics() {
    verify_stale_source_detector();
    verify_null_straddle_detector();
    assert_eq!(
        BALANCED_SQUARE,
        [true, false, false, true, true, false, false, true]
    );
    assert_eq!(
        BALANCED_SQUARE.iter().filter(|&&is_a| is_a).count(),
        BALANCED_SQUARE.len() / 2,
        "balanced square must give each arm four slots",
    );
    let mut linear_slot_ns = 0_u64;
    let (drift_a, drift_b) = balanced_square_round_with(|_| {
        linear_slot_ns += 10;
        ContractObservation {
            elapsed: Duration::from_nanos(linear_slot_ns),
            checksum: 0x5a17,
        }
    });
    assert_eq!(
        drift_a.elapsed,
        Duration::from_nanos(45),
        "A's ABBAABBA slots must center linear drift at the round midpoint",
    );
    assert_eq!(
        drift_b.elapsed,
        Duration::from_nanos(45),
        "B's ABBAABBA slots must center linear drift at the round midpoint",
    );
    let mut one_shot_calls = 0_usize;
    let one_shot = min_observation_with(
        &mut || {
            one_shot_calls += 1;
            ContractObservation {
                elapsed: Duration::from_nanos(17),
                checksum: 0x51_0f,
            }
        },
        1,
    );
    assert_eq!(
        one_shot_calls, 1,
        "a lifecycle probe with min_of=1 must make one materialization per timed slot"
    );
    assert_eq!(one_shot.elapsed, Duration::from_nanos(17));
    // Lifecycle accounting counts BALANCED_SQUARE entries, so pin that a round
    // visits each entry exactly once, in order. Without this link the phase
    // slot arithmetic below could describe a schedule the runner never runs.
    let mut visited_slots = Vec::with_capacity(BALANCED_SQUARE.len());
    balanced_square_round_with(|slot_is_a| {
        visited_slots.push(slot_is_a);
        ContractObservation {
            elapsed: Duration::from_nanos(1),
            checksum: 0x5107,
        }
    });
    assert_eq!(
        visited_slots.as_slice(),
        BALANCED_SQUARE.as_slice(),
        "a balanced-square round must visit every slot exactly once in order",
    );
    assert_eq!(
        DUAL_NULL_PHASES.map(|phase| phase.arm_slots(true)),
        [BALANCED_SQUARE.len(), 0, BALANCED_SQUARE.len() / 2],
        "the incumbent runs its own A/A in full, none of the candidate's, and half the effect",
    );
    assert_eq!(
        DUAL_NULL_PHASES.map(|phase| phase.arm_slots(false)),
        [0, BALANCED_SQUARE.len(), BALANCED_SQUARE.len() / 2],
        "the candidate runs its own A/A in full, none of the incumbent's, and half the effect",
    );
    // End-to-end count: drive the real phase runner and tally how many times
    // each arm is actually materialized. The lifecycle probes in the workload
    // benches assert their sample vectors against
    // `dual_null_observation_count_per_arm`, and that assertion fires only
    // after a multi-minute measurement, so the schedule and the formula are
    // reconciled here instead. A formula that counts the peer arm's A/A phase,
    // drops the warm-up rounds, or ignores `min_of` fails this before any
    // bench spends a second timing.
    let probe_rounds = 9;
    let probe_min_of = 2;
    let mut incumbent_materializations = 0_usize;
    let mut candidate_materializations = 0_usize;
    {
        let mut incumbent_probe = || {
            incumbent_materializations += 1;
            ContractObservation {
                elapsed: Duration::from_nanos(11),
                checksum: 0x5107_5107,
            }
        };
        let mut candidate_probe = || {
            candidate_materializations += 1;
            ContractObservation {
                elapsed: Duration::from_nanos(7),
                checksum: 0x5107_5107,
            }
        };
        for phase in DUAL_NULL_PHASES {
            run_dual_null_phase(
                phase,
                &mut incumbent_probe,
                &mut candidate_probe,
                probe_rounds,
                probe_min_of,
            );
        }
    }
    let expected_materializations = dual_null_observation_count_per_arm(probe_rounds, probe_min_of);
    assert_eq!(
        incumbent_materializations, expected_materializations,
        "the incumbent arm must be materialized once per slot the accounting counts",
    );
    assert_eq!(
        candidate_materializations, expected_materializations,
        "the candidate arm must be materialized once per slot the accounting counts",
    );
    assert_eq!(
        expected_materializations, 312,
        "nine timed rounds plus four warm-up rounds, twelve arm slots each, min-of-two",
    );
    assert_eq!(
        dual_null_observation_count_per_arm(21, 1),
        300,
        "an arm sees its own eight A/A slots plus four effect slots in each of 25 rounds"
    );
    assert_eq!(
        dual_null_observation_count_per_arm(21, 2),
        600,
        "min-of trials must each be represented in lifecycle accounting"
    );
    let stats = |median: f64, low: f64, high: f64| ContractPairStats {
        ratio_median: median,
        ratio_ci_low: low,
        ratio_ci_high: high,
        ratio_cv_pct: 0.0,
        ratio_mad: 0.0,
        arm_a_median_ns: 1.0,
        arm_b_median_ns: 1.0,
        checksum: 0,
    };

    assert_eq!(
        contract_gate_verdict(stats(1.30, 0.98, 1.40), 0.98, 1.02, 0.02),
        "UNDECIDED",
        "an effect CI that crosses 1.0 must never score a win"
    );
    assert_eq!(
        contract_gate_verdict(stats(0.70, 0.60, 1.01), 0.98, 1.02, 0.02),
        "UNDECIDED",
        "an effect CI that crosses 1.0 must never score a regression"
    );
    assert_eq!(
        contract_gate_verdict(stats(1.30, 1.20, 1.40), 0.98, 1.02, 0.02),
        "DECIDABLE_WIN",
        "a separated effect CI that clears the 2x null margin must score a win"
    );
    assert_eq!(
        contract_gate_verdict(stats(0.70, 0.60, 0.80), 0.98, 1.02, 0.02),
        "DECIDABLE_REGRESSION",
        "a separated effect CI that clears the 2x null margin must score a regression"
    );
    assert_eq!(
        contract_gate_verdict(stats(1.03, 1.01, 1.05), 0.98, 1.02, 0.02),
        "UNDECIDED",
        "an effect CI excluding 1.0 is insufficient without the 2x null margin"
    );
    assert_eq!(
        contract_gate_verdict(stats(1.30, 1.20, 1.40), 0.96, 0.99, 0.04),
        "DECIDABLE_WIN",
        "a precise null must not be vetoed merely because its CI excludes 1.0"
    );
}

/// Run a base/base null first, then an interleaved base/candidate effect in one
/// process. Callers prove output parity before entering this timer.
pub fn run_median_ci_contract<A, B>(
    row: &str,
    mut former: A,
    mut candidate: B,
) -> (ContractPairStats, ContractPairStats)
where
    A: FnMut() -> ContractObservation,
    B: FnMut() -> ContractObservation,
{
    for _ in 0..4 {
        black_box(balanced_square_round_with(|_| min_observation(&mut former)));
    }

    // Contract order is deliberate: establish the base/base null before the
    // candidate can perturb caches, allocator state, or worker scheduling.
    let mut null_a_samples = Vec::with_capacity(CONTRACT_ROUNDS);
    let mut null_b_samples = Vec::with_capacity(CONTRACT_ROUNDS);
    let mut null_checksum = 0_u64;
    for _ in 0..CONTRACT_ROUNDS {
        let (a, b) = balanced_square_round_with(|_| min_observation(&mut former));
        assert_eq!(
            a.checksum, b.checksum,
            "base/base null arms produced different output checksums"
        );
        null_a_samples.push(a.elapsed.as_secs_f64() * 1.0e9);
        null_b_samples.push(b.elapsed.as_secs_f64() * 1.0e9);
        null_checksum = mix_checksum(null_checksum, a.checksum);
    }
    let null = contract_pair_stats(&null_a_samples, &null_b_samples, null_checksum);
    report_contract_pair("null_base_aa", null);

    for _ in 0..4 {
        black_box(balanced_square_round_with(|is_former| {
            if is_former {
                min_observation(&mut former)
            } else {
                min_observation(&mut candidate)
            }
        }));
    }
    let mut former_samples = Vec::with_capacity(CONTRACT_ROUNDS);
    let mut candidate_samples = Vec::with_capacity(CONTRACT_ROUNDS);
    let mut effect_checksum = 0_u64;
    for _ in 0..CONTRACT_ROUNDS {
        let (former_elapsed, candidate_elapsed) = balanced_square_round_with(|is_former| {
            if is_former {
                min_observation(&mut former)
            } else {
                min_observation(&mut candidate)
            }
        });
        assert_eq!(
            former_elapsed.checksum, candidate_elapsed.checksum,
            "base/candidate arms produced different output checksums"
        );
        former_samples.push(former_elapsed.elapsed.as_secs_f64() * 1.0e9);
        candidate_samples.push(candidate_elapsed.elapsed.as_secs_f64() * 1.0e9);
        effect_checksum = mix_checksum(effect_checksum, former_elapsed.checksum);
    }
    let effect = contract_pair_stats(&former_samples, &candidate_samples, effect_checksum);
    report_contract_pair("effect_former_over_candidate", effect);
    report_contract_gate(row, effect, null);
    (effect, null)
}

/// Run one dual-null phase: balanced warm-up rounds followed by `rounds` timed
/// rounds, with every slot routed to an arm by `phase.selects_incumbent`. The
/// A/A phases therefore reach exactly one arm and the effect phase splits its
/// slots, which is the distribution `dual_null_observation_count_per_arm`
/// counts.
fn run_dual_null_phase<A, B>(
    phase: DualNullPhase,
    incumbent: &mut A,
    candidate: &mut B,
    rounds: usize,
    min_of: usize,
) -> ContractPairStats
where
    A: FnMut() -> ContractObservation,
    B: FnMut() -> ContractObservation,
{
    for _ in 0..DUAL_NULL_WARMUP_ROUNDS {
        black_box(balanced_square_round_with(|slot_is_a| {
            if phase.selects_incumbent(slot_is_a) {
                min_observation_with(incumbent, min_of)
            } else {
                min_observation_with(candidate, min_of)
            }
        }));
    }

    let mut phase_witness = ArmCpuWitness::default();
    let mut arm_a = Vec::with_capacity(rounds);
    let mut arm_b = Vec::with_capacity(rounds);
    let mut checksum = 0_u64;
    for _ in 0..rounds {
        let (a, b, round_witness) = balanced_square_round_witnessed(|slot_is_a| {
            if phase.selects_incumbent(slot_is_a) {
                min_observation_with(incumbent, min_of)
            } else {
                min_observation_with(candidate, min_of)
            }
        });
        assert_eq!(
            a.checksum,
            b.checksum,
            "{}",
            phase.checksum_mismatch_message()
        );
        phase_witness.merge(&round_witness);
        arm_a.push(a.elapsed.as_secs_f64() * 1.0e9);
        arm_b.push(b.elapsed.as_secs_f64() * 1.0e9);
        checksum = mix_checksum(checksum, a.checksum);
    }
    // Per-ARM frequency provenance. Emitted rather than folded into
    // `ContractPairStats` because that struct is constructed by literal in several
    // places, so adding fields would ripple into every one of them.
    println!(
        "CPU_WITNESS phase={} arm_a_cpu={} arm_b_cpu={} \
         arm_a_mhz_mean={:.1} arm_b_mhz_mean={:.1} \
         same_core={} arm_mhz_spread={:.4} \
         sampled_after_each_slot_outside_the_timed_region=true",
        phase.label(),
        phase_witness
            .a_cpu
            .map_or_else(|| "unknown".to_string(), |c| c.to_string()),
        phase_witness
            .b_cpu
            .map_or_else(|| "unknown".to_string(), |c| c.to_string()),
        phase_witness.a_mhz_mean,
        phase_witness.b_mhz_mean,
        phase_witness.same_core(),
        phase_witness.mhz_spread(),
    );
    contract_pair_stats(&arm_a, &arm_b, checksum)
}

/// Run incumbent/incumbent and candidate/candidate nulls before the interleaved
/// incumbent/candidate effect. This is the realistic-workload contract: both
/// arms occupy symmetric slots on a contended host, and the wider A/A interval
/// controls the median-CI verdict.
///
/// # WHAT THESE NULLS CANNOT DETECT
///
/// A null proves each arm is INTERNALLY REPRODUCIBLE. It can never show that the two
/// arms are COMPARABLE. Two arms measuring different things are both perfectly stable,
/// so both nulls sit on unity while the effect between them measures something other
/// than the difference under test. Passing this gate is necessary, not sufficient.
///
/// `deadlock-audit-48by6` is the worked example: `bench_maximum_arms_vs_numpy` timed
/// Rust replicas writing into `Vec`s allocated ONCE outside the loop against a NumPy
/// call allocating a fresh 32 MB output every iteration. The candidate was handed for
/// free the most expensive thing the incumbent did. Both nulls were clean throughout,
/// and the group read 2.430654x where the shipped route read 0.907848x for the same op
/// at the same n — a 2.7x error that survived long enough for a REJECT to be built on
/// it.
///
/// So before trusting a row from this contract, check by hand what the nulls cannot:
///
/// - **Allocation symmetry.** Does exactly one side allocate its output per iteration?
///   Say which side allocates in the row rather than assuming symmetry — and note that
///   `numpy.empty` and a Rust `Vec` do not have the same first-touch behaviour, so
///   "make both allocate" is not automatically a fix.
/// - **Route or replica.** Is the candidate the shipped path, or a bench-local replica
///   that skips work the real route pays? If a replica, the row must say so and must
///   never be quoted as a vs-incumbent number.
/// - **Parallel against serial.** If the candidate is parallel and the incumbent
///   serial, host load biases the ratio DIRECTIONALLY against the candidate rather than
///   adding symmetric noise. The ratio of the two null half-widths is the cheapest
///   in-band detector; record the load endpoints (`deadlock-audit-322j4`).
/// - **Operand parity.** Do both arms compute the same values, checked by checksum
///   BEFORE timing?
pub fn run_dual_null_median_ci_contract<A, B>(
    row: &str,
    incumbent: A,
    candidate: B,
) -> (ContractPairStats, ContractPairStats, ContractPairStats)
where
    A: FnMut() -> ContractObservation,
    B: FnMut() -> ContractObservation,
{
    run_dual_null_median_ci_contract_with_sampling(
        row,
        incumbent,
        candidate,
        CONTRACT_ROUNDS,
        CONTRACT_MIN_OF,
    )
}

/// Run the dual-null contract with an explicit sampling budget for realistic
/// workloads whose multi-second incumbent arm cannot fit the microbenchmark
/// default inside the worker's execution ceiling. Sampling remains odd-sized,
/// interleaved, bootstrap-median-CI gated, and controlled by the wider A/A
/// envelope.
pub fn run_dual_null_median_ci_contract_with_sampling<A, B>(
    row: &str,
    mut incumbent: A,
    mut candidate: B,
    rounds: usize,
    min_of: usize,
) -> (ContractPairStats, ContractPairStats, ContractPairStats)
where
    A: FnMut() -> ContractObservation,
    B: FnMut() -> ContractObservation,
{
    assert!(
        rounds >= 9 && rounds & 1 == 1,
        "contract rounds must be an odd count of at least nine"
    );
    assert!(min_of >= 1, "contract min-of count must be non-zero");

    // Walk the schedule constant itself, in order, so the phases that run are
    // exactly the phases `dual_null_observation_count_per_arm` counts. Writing
    // the three calls out by hand is what let the accounting drift before:
    // a phase can only be added to the schedule by adding it here too.
    let mut incumbent_null = None;
    let mut candidate_null = None;
    let mut effect = None;
    for phase in DUAL_NULL_PHASES {
        let stats = run_dual_null_phase(phase, &mut incumbent, &mut candidate, rounds, min_of);
        report_contract_pair_with_sampling(
            &format!("{row}{}", phase.row_suffix()),
            stats,
            rounds,
            min_of,
        );
        // Exhaustive on purpose: a phase added to the schedule cannot compile
        // until it is given a destination here, which is the drift this whole
        // accounting exists to prevent.
        match phase {
            DualNullPhase::IncumbentNull => incumbent_null = Some(stats),
            DualNullPhase::CandidateNull => candidate_null = Some(stats),
            DualNullPhase::Effect => effect = Some(stats),
        }
    }
    let incumbent_null = incumbent_null.expect("the schedule must run an incumbent A/A phase");
    let candidate_null = candidate_null.expect("the schedule must run a candidate A/A phase");
    let effect = effect.expect("the schedule must run the interleaved effect phase");

    report_dual_null_contract_gate(row, effect, incumbent_null, candidate_null);
    (effect, incumbent_null, candidate_null)
}

/// Import numpy on the interpreter, mapping the module handle away; every bench
/// group calls this before allocating its inputs so a missing numpy fails loud.
pub fn ensure_numpy_available(py: Python<'_>) -> PyResult<()> {
    py.import("numpy").map(drop)
}

/// Mean and coefficient-of-variation (percent) over the last <=10 retained
/// paired samples. Panics below two samples, which Criterion always retains.
pub fn ledger_tail_stats(samples: &RefCell<Vec<f64>>) -> (usize, f64, f64) {
    let samples = samples.borrow();
    let count = samples.len().min(10);
    assert!(
        count >= 2,
        "Criterion must retain at least two paired samples"
    );
    let tail = &samples[samples.len() - count..];
    let mean = tail.iter().sum::<f64>() / count as f64;
    let variance = tail
        .iter()
        .map(|sample| {
            let delta = sample - mean;
            delta * delta
        })
        .sum::<f64>()
        / (count - 1) as f64;
    (count, mean, variance.sqrt() * 100.0 / mean)
}

/// Emit the paired `LEDGER_AUDIT` line the negative-evidence flow parses:
/// candidate/orig means (ms), CVs (%), and the orig/candidate ratio.
pub fn report_ledger_pair(
    row: &str,
    candidate_samples: &RefCell<Vec<f64>>,
    orig_samples: &RefCell<Vec<f64>>,
) {
    if candidate_samples.borrow().is_empty() && orig_samples.borrow().is_empty() {
        return;
    }
    let (candidate_n, candidate_ns, candidate_cv) = ledger_tail_stats(candidate_samples);
    let (orig_n, orig_ns, orig_cv) = ledger_tail_stats(orig_samples);
    assert_eq!(candidate_n, orig_n);
    println!(
        "LEDGER_AUDIT row={row} samples={candidate_n} candidate_mean_ms={:.6} \
         candidate_cv_pct={candidate_cv:.3} orig_mean_ms={:.6} orig_cv_pct={orig_cv:.3} \
         orig_over_candidate={:.4}",
        candidate_ns / 1_000_000.0,
        orig_ns / 1_000_000.0,
        orig_ns / candidate_ns,
    );
}

/// `FNP_BENCH_GROUPS`, when set to a comma-separated substring list, restricts a
/// run to the group functions whose names contain one of the tokens; unset
/// preserves run-everything behavior. A `fnp-group=<same list>` positional
/// Criterion filter is the fail-closed remote equivalent: RCH intentionally
/// does not forward arbitrary caller environment variables, while the argument
/// is part of the exact remotely executed command.
fn group_selection_spec() -> Option<String> {
    // The explicit command-line selector is the remote, auditable source of
    // truth. It must override any inherited worker environment (including an
    // accidentally present empty FNP_BENCH_GROUPS), otherwise a correctly
    // filtered RCH invocation can build the ELF and silently execute no group.
    std::env::args()
        .skip(1)
        .find_map(|argument| argument.strip_prefix("fnp-group=").map(str::to_owned))
        .or_else(|| std::env::var("FNP_BENCH_GROUPS").ok())
}

fn group_enabled_with_spec(group_fn_name: &str, spec: Option<&str>) -> bool {
    let Some(spec) = spec else {
        return true;
    };
    spec.split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .any(|token| group_fn_name.contains(token))
}

fn selected_group_count<'a>(
    group_names: impl IntoIterator<Item = &'a str>,
    spec: Option<&str>,
) -> usize {
    group_names
        .into_iter()
        .filter(|name| group_enabled_with_spec(name, spec))
        .count()
}

fn verify_group_selection_contract() {
    let sessionization = "bench_realistic_clickstream_sessionization_vs_numpy_median_gate";
    let groups = [sessionization, "bench_loadtxt_selected_bool_median_gate"];
    assert!(group_enabled_with_spec(sessionization, None));
    assert!(group_enabled_with_spec(
        sessionization,
        Some("realistic_clickstream_sessionization"),
    ));
    assert!(group_enabled_with_spec(
        sessionization,
        Some("other_group, clickstream_sessionization"),
    ));
    assert!(!group_enabled_with_spec(
        sessionization,
        Some("loadtxt_selected_bool")
    ));
    assert!(!group_enabled_with_spec(sessionization, Some(" , ")));
    assert_eq!(selected_group_count(groups, None), 2);
    assert_eq!(
        selected_group_count(groups, Some("clickstream_sessionization")),
        1
    );
    assert_eq!(selected_group_count(groups, Some("unmatched_group")), 0);
}

/// A named bench group: the group function's name (for `FNP_BENCH_GROUPS`
/// gating) paired with the function itself.
pub type BenchGroup = (&'static str, fn(&mut Criterion));

/// Drive the selected bench group functions under one `Criterion`, then emit the
/// final summary. Mirrors the former `gated_benches!` macro's `main`: each entry
/// is `(group_fn_name, group_fn)`, gated by [`group_enabled`].
/// Prefer [`gated_main_with_source`], which additionally fingerprints the calling
/// bench's own body. This entry point fingerprints only the shared contract module,
/// and says so on its own identity line (`covers=common/mod.rs_only`,
/// `stale_bench_body_would_be_undetectable=true`).
pub fn gated_main(targets: &[BenchGroup]) {
    gated_main_inner(None, targets);
}

/// `bench_source` must be the caller's OWN source: `include_str!("<this file>.rs")`
/// at the call site, which resolves relative to the invoking file's directory and
/// so needs no path plumbing. See [`report_bench_identity`] for why this is the
/// only field that ties a running bench binary to the body it was compiled from.
pub fn gated_main_with_source(bench_source: &'static str, targets: &[BenchGroup]) {
    gated_main_inner(Some(bench_source), targets);
}

fn gated_main_inner(bench_source: Option<&str>, targets: &[BenchGroup]) {
    // Line one, before Criterion is constructed: Criterion may print its
    // backend notice during construction.
    report_bench_identity(bench_source);
    verify_contract_gate_semantics();
    verify_group_selection_contract();
    println!(
        "BALANCED_SQUARE_ADMISSION schedule=ABBAABBA host_quiescence=not_required \
         null_controls=required incumbent_same_invocation=required"
    );
    std::io::Write::flush(&mut std::io::stdout()).expect("flushing stdout cannot fail");
    let selected_spec = group_selection_spec();
    let selected_groups = selected_group_count(
        targets.iter().map(|(name, _)| *name),
        selected_spec.as_deref(),
    );
    if let Some(spec) = selected_spec.as_deref() {
        assert!(
            selected_groups > 0,
            "fnp-group selector {spec:?} matched no benchmark groups"
        );
        // This must precede Criterion construction and target setup: remote
        // workers can spend minutes compiling or importing Python, and a
        // captured prefix still needs to prove which exact group was admitted.
        println!("BENCH_GROUP_SELECTION selector={spec:?} selected_groups={selected_groups}");
        std::io::Write::flush(&mut std::io::stdout()).expect("flushing stdout cannot fail");
    }
    let mut criterion = Criterion::default().configure_from_args();
    for (name, target) in targets {
        if group_enabled_with_spec(name, selected_spec.as_deref()) {
            target(&mut criterion);
        }
    }
    criterion.final_summary();
}

// Ledger-integrity retries for three historical REJECT rows. These helpers live only in the
// benchmark binary: production dispatch is deliberately untouched. `inline(never)` gives perf
// an exact execution marker for each reconstructed candidate and each NumPy ORIG reference.
#[inline]
pub fn ledger_f64_sortable_key(value: f64) -> u64 {
    let bits = if value == 0.0 { 0 } else { value.to_bits() };
    bits ^ ((((bits as i64) >> 63) as u64) | 0x8000_0000_0000_0000)
}

#[inline]
pub fn ledger_f64_from_sortable_key(key: u64) -> f64 {
    let bits = if key & 0x8000_0000_0000_0000 != 0 {
        key ^ 0x8000_0000_0000_0000
    } else {
        !key
    };
    f64::from_bits(bits)
}

#[inline(never)]
pub fn ledger_radix_select_key(mut current: Vec<u64>, mut rank: usize, start_byte: i32) -> u64 {
    let mut byte = start_byte;
    loop {
        let len = current.len();
        if len <= 1 || byte < 0 {
            return current[rank];
        }
        let shift = (byte as u64) * 8;
        let histogram: [usize; 256] = if len > (1 << 16) {
            let chunk_size = (len / (rayon::current_num_threads() * 4).max(1)).max(1);
            current
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut local = [0usize; 256];
                    for &key in chunk {
                        local[((key >> shift) & 0xff) as usize] += 1;
                    }
                    local
                })
                .reduce(
                    || [0usize; 256],
                    |mut left, right| {
                        for digit in 0..256 {
                            left[digit] += right[digit];
                        }
                        left
                    },
                )
        } else {
            let mut local = [0usize; 256];
            for &key in &current {
                local[((key >> shift) & 0xff) as usize] += 1;
            }
            local
        };
        let mut prefix = 0usize;
        let mut selected = 255usize;
        for (digit, &count) in histogram.iter().enumerate() {
            if prefix + count > rank {
                selected = digit;
                break;
            }
            prefix += count;
        }
        current = if len > (1 << 16) {
            current
                .par_iter()
                .copied()
                .filter(|&key| ((key >> shift) & 0xff) as usize == selected)
                .collect()
        } else {
            current
                .iter()
                .copied()
                .filter(|&key| ((key >> shift) & 0xff) as usize == selected)
                .collect()
        };
        rank -= prefix;
        byte -= 1;
    }
}

#[inline(never)]
pub fn ledger_radix_median_f64(data: &[f64]) -> f64 {
    assert!(!data.par_iter().any(|value| value.is_nan()));
    let keys: Vec<u64> = data
        .par_iter()
        .map(|&value| ledger_f64_sortable_key(value))
        .collect();
    let n = keys.len();
    if n % 2 == 1 {
        ledger_f64_from_sortable_key(ledger_radix_select_key(keys, n / 2, 7))
    } else {
        let low = ledger_f64_from_sortable_key(ledger_radix_select_key(keys.clone(), n / 2 - 1, 7));
        let high = ledger_f64_from_sortable_key(ledger_radix_select_key(keys, n / 2, 7));
        (low + high) / 2.0
    }
}

#[inline(never)]
pub fn ledger_orig_median_reference(
    numpy_median: &Bound<'_, PyAny>,
    input: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    numpy_median.call1((input,))?.extract()
}

#[inline(never)]
pub fn ledger_try_native_f16_sort(
    numpy_sort: &Bound<'_, PyAny>,
    input: &Bound<'_, PyAny>,
    input_bits: &[u16],
) -> PyResult<Py<PyAny>> {
    let must_defer = input_bits
        .par_iter()
        .any(|&bits| bits == 0x8000 || ((bits & 0x7c00) == 0x7c00 && (bits & 0x03ff) != 0));
    assert!(
        !must_defer,
        "finite positive f16 audit input must stay on candidate route"
    );
    let widened = input.call_method1("astype", ("float32",))?;
    let sorted = numpy_sort.call1((&widened,))?;
    Ok(sorted.call_method1("astype", ("float16",))?.unbind())
}

#[inline(never)]
pub fn ledger_orig_f16_sort_reference(
    numpy_sort: &Bound<'_, PyAny>,
    input: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    Ok(numpy_sort.call1((input,))?.unbind())
}

#[inline(never)]
pub fn ledger_f32_tie_argsort_candidate(
    fnp_argsort: &Bound<'_, PyAny>,
    input: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    Ok(fnp_argsort.call1((input,))?.unbind())
}

#[inline(never)]
pub fn ledger_orig_f32_argsort_reference(
    numpy_argsort: &Bound<'_, PyAny>,
    input: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    Ok(numpy_argsort.call1((input,))?.unbind())
}

pub fn report_substrate_v2_pair(
    row: &str,
    candidate_samples: &RefCell<Vec<f64>>,
    orig_samples: &RefCell<Vec<f64>>,
) {
    if candidate_samples.borrow().is_empty() && orig_samples.borrow().is_empty() {
        return;
    }
    let (candidate_n, candidate_ns, candidate_cv) = ledger_tail_stats(candidate_samples);
    let (orig_n, orig_ns, orig_cv) = ledger_tail_stats(orig_samples);
    assert_eq!(candidate_n, orig_n);
    println!(
        "SUBSTRATE_V2 row={row} samples={candidate_n} candidate_mean_ms={:.6} \
         candidate_cv_pct={candidate_cv:.3} orig_mean_ms={:.6} orig_cv_pct={orig_cv:.3} \
         orig_over_candidate={:.4}",
        candidate_ns / 1_000_000.0,
        orig_ns / 1_000_000.0,
        orig_ns / candidate_ns,
    );
}

pub const MEDIAN_GATE_FINAL_BATCHES: usize = 10;
pub const MEDIAN_GATE_OBSERVATIONS_PER_BATCH: usize = 2;

#[derive(Clone, Copy)]
pub struct MedianGateDistribution {
    median: f64,
    p10: f64,
    p90: f64,
    low: f64,
    high: f64,
    cv_pct: f64,
    above_one: usize,
}

pub fn median_gate_quantile(sorted: &[f64], quantile: f64) -> f64 {
    assert!(!sorted.is_empty());
    let position = quantile * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = position - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

pub fn median_gate_distribution(samples: &[f64]) -> MedianGateDistribution {
    assert!(samples.len() >= 2);
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples
        .iter()
        .map(|sample| {
            let delta = sample - mean;
            delta * delta
        })
        .sum::<f64>()
        / (samples.len() - 1) as f64;
    MedianGateDistribution {
        median: median_gate_quantile(&sorted, 0.5),
        p10: median_gate_quantile(&sorted, 0.1),
        p90: median_gate_quantile(&sorted, 0.9),
        low: sorted[0],
        high: sorted[sorted.len() - 1],
        cv_pct: variance.sqrt() * 100.0 / mean,
        above_one: samples.iter().filter(|&&ratio| ratio > 1.0).count(),
    }
}

pub fn median_gate_tail(samples: &RefCell<Vec<f64>>) -> Vec<f64> {
    let samples = samples.borrow();
    let retained = MEDIAN_GATE_FINAL_BATCHES * MEDIAN_GATE_OBSERVATIONS_PER_BATCH;
    assert!(
        samples.len() >= retained,
        "Criterion must retain {retained} median-gate observations"
    );
    samples[samples.len() - retained..].to_vec()
}

pub fn report_median_gate_pair(
    row: &str,
    null_base_ns: &RefCell<Vec<f64>>,
    null_peer_ns: &RefCell<Vec<f64>>,
    null_ratios: &RefCell<Vec<f64>>,
    base_ns: &RefCell<Vec<f64>>,
    candidate_ns: &RefCell<Vec<f64>>,
    effect_ratios: &RefCell<Vec<f64>>,
) {
    if effect_ratios.borrow().is_empty() {
        return;
    }
    let null_base = median_gate_tail(null_base_ns);
    let null_peer = median_gate_tail(null_peer_ns);
    let null = median_gate_distribution(&median_gate_tail(null_ratios));
    let base = median_gate_distribution(&median_gate_tail(base_ns));
    let candidate = median_gate_distribution(&median_gate_tail(candidate_ns));
    let effect = median_gate_distribution(&median_gate_tail(effect_ratios));
    // LATENT GATE DEFECT — DORMANT HERE, DO NOT COPY THIS CLAUSE INTO A NEW GATE.
    //
    // `null_brackets_one` vetoes before the effect is even consulted, and it
    // keys on the null's PRECISION: the tighter (better) the null, the narrower
    // p10..p90, the more likely it excludes 1.0, the more likely it vetoes a
    // real effect. frankenlibc quantified the same clause elsewhere in the fleet
    // suppressing 130-265% effects on null intervals that missed 1.0 by
    // 0.04-0.5%. The current dual-null contract
    // (`report_dual_null_contract_gate`) deliberately does NOT have this clause:
    // it requires the EFFECT bootstrap CI to exclude 1.0, compares the effect
    // median against the null CI bounds, and folds null bias into
    // `required_delta`, so null precision cannot veto a row.
    //
    // Audited 2026-07-30 on the identical release-perf ELF
    // a27630778eff50a49987056bc8d5fc5025758a686d0b2f6dac54232b3ec6ac53, two
    // passes on a drained host-exclusive worker: isclose_f64_8m,
    // maximum_accumulate_f64_8m, and int_convolve_i64_200k_256 reproduced their
    // effects within 1.6-8.9% and returned WIN 6/6. No veto fired, because
    // these microbench nulls are WIDE (p10..p90 spans 4-24%) and so bracket 1.0
    // comfortably. The defect is therefore dormant on this surface, and the
    // fleet rule is to leave a verdict-stable gate alone — hence no behaviour
    // change here.
    //
    // It stops being dormant the moment anyone tightens these arms (more
    // batches, pinned warm iterations, a quieter host). If you do that, replace
    // this clause with the corrected rule rather than debugging your data: gate
    // on the effect CI excluding 1.0 AND the effect deviation exceeding twice
    // the larger null half-width, and report null bias as telemetry. Bound the
    // bias against its OWN uncertainty if you must gate on it at all; a bare
    // "median within 2%" test was measured on this repo's workload surface to
    // move which size point it rejects between identical-ELF runs.
    let null_brackets_one = null.p10 <= 1.0 && null.p90 >= 1.0;
    let verdict = if !null_brackets_one {
        "BIASED_NULL"
    } else if effect.median > null.p90 {
        "WIN"
    } else if effect.median < null.p10 {
        "PROFILE_REQUIRED"
    } else {
        "UNDECIDED"
    };
    let null_base_cv = median_gate_distribution(&null_base).cv_pct;
    let null_peer_cv = median_gate_distribution(&null_peer).cv_pct;
    println!(
        "NULL_MEDIAN_GATE row={row} observations={} base_median_ms={:.6} \
         candidate_median_ms={:.6} base_cv_pct={:.3} candidate_cv_pct={:.3} \
         effect_median={:.6} effect_p10={:.6} effect_p90={:.6} \
         effect_low={:.6} effect_high={:.6} effect_cv_pct={:.3} effect_above_one={} \
         null_median={:.6} null_p10={:.6} null_p90={:.6} null_low={:.6} \
         null_high={:.6} null_cv_pct={:.3} null_base_cv_pct={:.3} \
         null_peer_cv_pct={:.3} null_corrected_median={:.6} verdict={verdict}",
        effect_ratios
            .borrow()
            .len()
            .min(MEDIAN_GATE_FINAL_BATCHES * MEDIAN_GATE_OBSERVATIONS_PER_BATCH),
        base.median / 1_000_000.0,
        candidate.median / 1_000_000.0,
        base.cv_pct,
        candidate.cv_pct,
        effect.median,
        effect.p10,
        effect.p90,
        effect.low,
        effect.high,
        effect.cv_pct,
        effect.above_one,
        null.median,
        null.p10,
        null.p90,
        null.low,
        null.high,
        null.cv_pct,
        null_base_cv,
        null_peer_cv,
        effect.median / null.median,
    );
}

pub fn time_python_binary_call<'py>(
    function: &Bound<'py, PyAny>,
    lhs: &Bound<'py, PyAny>,
    rhs: &Bound<'py, PyAny>,
) -> Duration {
    let start = Instant::now();
    let function = black_box(function);
    let lhs = black_box(lhs);
    let rhs = black_box(rhs);
    let result = function
        .call1((lhs, rhs))
        .expect("median-gate binary Python call");
    drop(black_box(result));
    start.elapsed()
}

pub fn time_python_unary_call<'py>(
    function: &Bound<'py, PyAny>,
    input: &Bound<'py, PyAny>,
) -> Duration {
    let start = Instant::now();
    let function = black_box(function);
    let input = black_box(input);
    let result = function
        .call1((input,))
        .expect("median-gate unary Python call");
    drop(black_box(result));
    start.elapsed()
}
