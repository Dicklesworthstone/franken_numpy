//! median-gate / substrate-v2 variance-protocol criterion benches — the paired
//! `iter_custom` A/B harness with median-gate reporting (interleaved AB/BA,
//! null control, tail stats) plus its local timing helpers. Split out of the
//! monolithic `criterion_python_surface.rs` into their own per-domain bench
//! binary; this is the single largest domain (~3900 lines / 21 fns), so pulling
//! it out is the biggest compile-volume reduction of the split. See bead
//! deadlock-audit-x7nnf.

#[path = "common/mod.rs"]
mod common;

use common::*;
use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::buffer::PyBuffer;
use pyo3::types::{PyAnyMethods, PyDict, PyModule, PyTuple};
use pyo3::{Bound, PyAny, Python};
use rayon::prelude::*;
use std::cell::{Cell, RefCell};
use std::fmt::Write as _;
use std::hint::black_box;
use std::time::{Duration, Instant};

fn workload_checksum<const N: usize>(
    numpy: &Bound<'_, PyModule>,
    outputs: &[Bound<'_, PyAny>; N],
) -> u64 {
    fn mix_bytes(mut state: u64, bytes: &[u8]) -> u64 {
        for byte in (bytes.len() as u64)
            .to_le_bytes()
            .iter()
            .chain(bytes.iter())
        {
            state = (state ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
        state
    }

    let mut state = 0xcbf2_9ce4_8422_2325_u64;
    for output in outputs {
        let array = numpy
            .call_method1("asarray", (output,))
            .expect("workload output converts to ndarray");
        let dtype = array
            .getattr("dtype")
            .expect("workload output dtype")
            .str()
            .expect("workload output dtype string")
            .to_string();
        let shape = array
            .getattr("shape")
            .expect("workload output shape")
            .str()
            .expect("workload output shape string")
            .to_string();
        let bytes = array
            .call_method0("tobytes")
            .expect("workload output bytes")
            .extract::<Vec<u8>>()
            .expect("workload output byte vector");
        state = mix_bytes(state, dtype.as_bytes());
        state = mix_bytes(state, shape.as_bytes());
        state = mix_bytes(state, &bytes);
    }
    state
}

fn assert_workload_outputs_equal<const N: usize>(
    numpy: &Bound<'_, PyModule>,
    row: &str,
    candidate: &[Bound<'_, PyAny>; N],
    incumbent: &[Bound<'_, PyAny>; N],
) {
    for (index, (candidate_output, incumbent_output)) in candidate.iter().zip(incumbent).enumerate()
    {
        let candidate_array = numpy
            .call_method1("asarray", (candidate_output,))
            .expect("candidate workload output converts to ndarray");
        let incumbent_array = numpy
            .call_method1("asarray", (incumbent_output,))
            .expect("incumbent workload output converts to ndarray");
        let candidate_dtype = candidate_array
            .getattr("dtype")
            .expect("candidate output dtype")
            .str()
            .expect("candidate output dtype string")
            .to_string();
        let incumbent_dtype = incumbent_array
            .getattr("dtype")
            .expect("incumbent output dtype")
            .str()
            .expect("incumbent output dtype string")
            .to_string();
        assert_eq!(
            candidate_dtype, incumbent_dtype,
            "{row}: output {index} dtype differs"
        );
        let candidate_shape = candidate_array
            .getattr("shape")
            .expect("candidate output shape")
            .extract::<Vec<usize>>()
            .expect("candidate output shape vector");
        let incumbent_shape = incumbent_array
            .getattr("shape")
            .expect("incumbent output shape")
            .extract::<Vec<usize>>()
            .expect("incumbent output shape vector");
        assert_eq!(
            candidate_shape, incumbent_shape,
            "{row}: output {index} shape differs"
        );
        let candidate_bytes = candidate_array
            .call_method0("tobytes")
            .expect("candidate output bytes")
            .extract::<Vec<u8>>()
            .expect("candidate output byte vector");
        let incumbent_bytes = incumbent_array
            .call_method0("tobytes")
            .expect("incumbent output bytes")
            .extract::<Vec<u8>>()
            .expect("incumbent output byte vector");
        assert_eq!(
            candidate_bytes, incumbent_bytes,
            "{row}: output {index} bytes differ"
        );
    }
    assert_eq!(
        workload_checksum(numpy, candidate),
        workload_checksum(numpy, incumbent),
        "{row}: aggregate output checksum differs"
    );
}

/// Linux process resources sampled around one materialized Python result.  The
/// benchmark deliberately keeps this outside the timed interval: these are
/// mechanism counters for the result-buffer lifecycle, not a timing shortcut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ProcessResourceSnapshot {
    minor_faults: u64,
    major_faults: u64,
    rss_kib: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResultBufferLifecycle {
    minor_faults: u64,
    major_faults: u64,
    rss_while_live_kib: i64,
    rss_after_release_kib: i64,
}

fn parse_proc_stat_faults(stat: &str) -> Result<(u64, u64), String> {
    // `/proc/self/stat` wraps the potentially space-containing command name in
    // parentheses.  Fields after its final `)` start at process state (#3), so
    // minflt (#10) and majflt (#12) are indexes 7 and 9 respectively.
    let (_, fields) = stat
        .rsplit_once(')')
        .ok_or_else(|| "missing closing process-name delimiter in /proc/self/stat".to_owned())?;
    let fields = fields.split_ascii_whitespace().collect::<Vec<_>>();
    let parse = |index: usize, name: &str| {
        fields
            .get(index)
            .ok_or_else(|| format!("missing {name} in /proc/self/stat"))?
            .parse::<u64>()
            .map_err(|error| format!("invalid {name} in /proc/self/stat: {error}"))
    };
    Ok((parse(7, "minflt")?, parse(9, "majflt")?))
}

fn parse_proc_status_rss_kib(status: &str) -> Result<u64, String> {
    let rss = status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .ok_or_else(|| "missing VmRSS in /proc/self/status".to_owned())?;
    rss.split_ascii_whitespace()
        .next()
        .ok_or_else(|| "missing VmRSS value in /proc/self/status".to_owned())?
        .parse::<u64>()
        .map_err(|error| format!("invalid VmRSS in /proc/self/status: {error}"))
}

fn process_resource_snapshot() -> ProcessResourceSnapshot {
    let stat = std::fs::read_to_string("/proc/self/stat")
        .expect("read Linux process fault counters from /proc/self/stat");
    let status = std::fs::read_to_string("/proc/self/status")
        .expect("read Linux process RSS from /proc/self/status");
    let (minor_faults, major_faults) =
        parse_proc_stat_faults(&stat).expect("parse Linux process fault counters");
    let rss_kib = parse_proc_status_rss_kib(&status).expect("parse Linux process RSS");
    ProcessResourceSnapshot {
        minor_faults,
        major_faults,
        rss_kib,
    }
}

fn result_buffer_lifecycle(
    before: ProcessResourceSnapshot,
    while_live: ProcessResourceSnapshot,
    after_release: ProcessResourceSnapshot,
) -> ResultBufferLifecycle {
    assert!(
        while_live.minor_faults >= before.minor_faults,
        "minor fault counter regressed while materializing a result"
    );
    assert!(
        while_live.major_faults >= before.major_faults,
        "major fault counter regressed while materializing a result"
    );
    ResultBufferLifecycle {
        minor_faults: while_live.minor_faults - before.minor_faults,
        major_faults: while_live.major_faults - before.major_faults,
        rss_while_live_kib: while_live.rss_kib as i64 - before.rss_kib as i64,
        rss_after_release_kib: after_release.rss_kib as i64 - before.rss_kib as i64,
    }
}

fn median_result_buffer_lifecycle(samples: &[ResultBufferLifecycle]) -> ResultBufferLifecycle {
    assert!(
        !samples.is_empty(),
        "result-buffer lifecycle measurement needs at least one observation"
    );
    let median_u64 = |values: Vec<u64>| {
        let mut values = values;
        values.sort_unstable();
        values[values.len() / 2]
    };
    let median_i64 = |values: Vec<i64>| {
        let mut values = values;
        values.sort_unstable();
        values[values.len() / 2]
    };
    ResultBufferLifecycle {
        minor_faults: median_u64(samples.iter().map(|sample| sample.minor_faults).collect()),
        major_faults: median_u64(samples.iter().map(|sample| sample.major_faults).collect()),
        rss_while_live_kib: median_i64(
            samples
                .iter()
                .map(|sample| sample.rss_while_live_kib)
                .collect(),
        ),
        rss_after_release_kib: median_i64(
            samples
                .iter()
                .map(|sample| sample.rss_after_release_kib)
                .collect(),
        ),
    }
}

fn verify_process_resource_snapshot_parser() {
    let stat = "4242 (criterion python worker) R 1 2 3 4 5 6 7 8 9 10 11";
    assert_eq!(parse_proc_stat_faults(stat), Ok((7, 9)));
    assert_eq!(
        parse_proc_status_rss_kib("Name:\tcriterion\nVmRSS:\t4242 kB\n"),
        Ok(4242)
    );
}

struct EventAttributionArm<'py> {
    add_at: Bound<'py, PyAny>,
    maximum_at: Bound<'py, PyAny>,
    minimum_at: Bound<'py, PyAny>,
    spend_state: Bound<'py, PyAny>,
    first_seen_state: Bound<'py, PyAny>,
    last_seen_state: Bound<'py, PyAny>,
}

impl<'py> EventAttributionArm<'py> {
    fn reset(&self) {
        self.spend_state
            .call_method1("fill", (0_i64,))
            .expect("reset spend state");
        self.first_seen_state
            .call_method1("fill", (i64::MAX,))
            .expect("reset first-seen state");
        self.last_seen_state
            .call_method1("fill", (i64::MIN,))
            .expect("reset last-seen state");
    }

    fn run(
        &self,
        account_ids: &Bound<'py, PyAny>,
        spend_deltas: &Bound<'py, PyAny>,
        event_timestamps: &Bound<'py, PyAny>,
    ) -> (Duration, [Bound<'py, PyAny>; 3]) {
        // Reset is harness preparation, not part of the persistent-state update
        // job. Keeping it outside the interval also prevents a shared ndarray
        // fill from contaminating the independent public FNP/NumPy call graphs.
        self.reset();
        let started = Instant::now();
        self.add_at
            .call1((
                black_box(&self.spend_state),
                black_box(account_ids),
                black_box(spend_deltas),
            ))
            .expect("event spend scatter-add");
        self.maximum_at
            .call1((
                black_box(&self.last_seen_state),
                black_box(account_ids),
                black_box(event_timestamps),
            ))
            .expect("event last-seen scatter-maximum");
        self.minimum_at
            .call1((
                black_box(&self.first_seen_state),
                black_box(account_ids),
                black_box(event_timestamps),
            ))
            .expect("event first-seen scatter-minimum");
        let elapsed = started.elapsed();
        (
            elapsed,
            [
                self.spend_state.clone(),
                self.first_seen_state.clone(),
                self.last_seen_state.clone(),
            ],
        )
    }

    fn profile(
        &self,
        account_ids: &Bound<'py, PyAny>,
        spend_deltas: &Bound<'py, PyAny>,
        event_timestamps: &Bound<'py, PyAny>,
    ) -> [f64; 3] {
        const PROFILE_ROUNDS: usize = 7;
        let mut add_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut maximum_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut minimum_ms = Vec::with_capacity(PROFILE_ROUNDS);
        for _ in 0..PROFILE_ROUNDS {
            self.reset();
            let started = Instant::now();
            self.add_at
                .call1((&self.spend_state, account_ids, spend_deltas))
                .expect("profile event spend scatter-add");
            add_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            self.maximum_at
                .call1((&self.last_seen_state, account_ids, event_timestamps))
                .expect("profile event last-seen scatter-maximum");
            maximum_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            self.minimum_at
                .call1((&self.first_seen_state, account_ids, event_timestamps))
                .expect("profile event first-seen scatter-minimum");
            minimum_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
        let median = |samples: &mut Vec<f64>| {
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2]
        };
        [
            median(&mut add_ms),
            median(&mut maximum_ms),
            median(&mut minimum_ms),
        ]
    }
}

struct EntitlementReconciliationArm<'py> {
    unique: Bound<'py, PyAny>,
    intersect1d: Bound<'py, PyAny>,
    setdiff1d: Bound<'py, PyAny>,
    unique_kwargs: Bound<'py, PyDict>,
}

impl<'py> EntitlementReconciliationArm<'py> {
    fn run(
        &self,
        previous: &Bound<'py, PyAny>,
        current: &Bound<'py, PyAny>,
    ) -> (Duration, [Bound<'py, PyAny>; 5]) {
        let started = Instant::now();
        let canonical = self
            .unique
            .call((black_box(current),), Some(&self.unique_kwargs))
            .expect("canonicalize current entitlement grants");
        let current_unique = canonical
            .get_item(0)
            .expect("canonical current entitlement keys");
        let current_counts = canonical
            .get_item(1)
            .expect("current entitlement duplicate counts");
        let unchanged = self
            .intersect1d
            .call1((black_box(previous), black_box(current)))
            .expect("unchanged entitlement grants");
        let added = self
            .setdiff1d
            .call1((black_box(current), black_box(previous)))
            .expect("added entitlement grants");
        let revoked = self
            .setdiff1d
            .call1((black_box(previous), black_box(current)))
            .expect("revoked entitlement grants");
        let elapsed = started.elapsed();
        (
            elapsed,
            [current_unique, current_counts, unchanged, added, revoked],
        )
    }

    fn profile(&self, previous: &Bound<'py, PyAny>, current: &Bound<'py, PyAny>) -> [f64; 4] {
        const PROFILE_ROUNDS: usize = 7;
        let mut canonicalize_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut unchanged_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut added_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut revoked_ms = Vec::with_capacity(PROFILE_ROUNDS);
        for _ in 0..PROFILE_ROUNDS {
            let started = Instant::now();
            black_box(
                self.unique
                    .call((current,), Some(&self.unique_kwargs))
                    .expect("profile canonical entitlement grants"),
            );
            canonicalize_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            black_box(
                self.intersect1d
                    .call1((previous, current))
                    .expect("profile unchanged entitlement grants"),
            );
            unchanged_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            black_box(
                self.setdiff1d
                    .call1((current, previous))
                    .expect("profile added entitlement grants"),
            );
            added_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            black_box(
                self.setdiff1d
                    .call1((previous, current))
                    .expect("profile revoked entitlement grants"),
            );
            revoked_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
        let median = |samples: &mut Vec<f64>| {
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2]
        };
        [
            median(&mut canonicalize_ms),
            median(&mut unchanged_ms),
            median(&mut added_ms),
            median(&mut revoked_ms),
        ]
    }
}

struct TelemetryDistributionArm<'py> {
    histogram: Bound<'py, PyAny>,
    cumsum: Bound<'py, PyAny>,
    argmax: Bound<'py, PyAny>,
}

impl<'py> TelemetryDistributionArm<'py> {
    fn run(&self, samples: &Bound<'py, PyAny>) -> (Duration, [Bound<'py, PyAny>; 4]) {
        let started = Instant::now();
        let histogram = self
            .histogram
            .call1((black_box(samples), 256_i64))
            .expect("telemetry latency histogram");
        let counts = histogram.get_item(0).expect("telemetry histogram counts");
        let edges = histogram.get_item(1).expect("telemetry histogram edges");
        let cumulative = self
            .cumsum
            .call1((black_box(&counts),))
            .expect("telemetry cumulative counts");
        let peak_bin = self
            .argmax
            .call1((black_box(&counts),))
            .expect("telemetry peak bin");
        let elapsed = started.elapsed();
        (elapsed, [counts, edges, cumulative, peak_bin])
    }

    fn profile(&self, samples: &Bound<'py, PyAny>) -> [f64; 3] {
        const PROFILE_ROUNDS: usize = 7;
        let mut histogram_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut cumulative_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut peak_ms = Vec::with_capacity(PROFILE_ROUNDS);
        for _ in 0..PROFILE_ROUNDS {
            let started = Instant::now();
            let histogram = self
                .histogram
                .call1((samples, 256_i64))
                .expect("profile telemetry histogram");
            histogram_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
            let counts = histogram
                .get_item(0)
                .expect("profile telemetry histogram counts");

            let started = Instant::now();
            black_box(
                self.cumsum
                    .call1((&counts,))
                    .expect("profile telemetry cumulative counts"),
            );
            cumulative_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            black_box(
                self.argmax
                    .call1((&counts,))
                    .expect("profile telemetry peak bin"),
            );
            peak_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
        let median = |samples: &mut Vec<f64>| {
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2]
        };
        [
            median(&mut histogram_ms),
            median(&mut cumulative_ms),
            median(&mut peak_ms),
        ]
    }
}

struct DynamicRangeAuditArm<'py> {
    frexp: Bound<'py, PyAny>,
    bincount: Bound<'py, PyAny>,
    ldexp: Bound<'py, PyAny>,
    bincount_kwargs: Bound<'py, PyDict>,
}

impl<'py> DynamicRangeAuditArm<'py> {
    fn run(&self, samples: &Bound<'py, PyAny>) -> (Duration, [Bound<'py, PyAny>; 4]) {
        let started = Instant::now();
        let decomposition = self
            .frexp
            .call1((black_box(samples),))
            .expect("dynamic-range frexp decomposition");
        let mantissa = decomposition.get_item(0).expect("dynamic-range mantissa");
        let exponent = decomposition.get_item(1).expect("dynamic-range exponent");
        let exponent_counts = self
            .bincount
            .call((black_box(&exponent),), Some(&self.bincount_kwargs))
            .expect("dynamic-range exponent distribution");
        let reconstructed = self
            .ldexp
            .call1((black_box(&mantissa), black_box(&exponent)))
            .expect("dynamic-range exact reconstruction");
        let elapsed = started.elapsed();
        (
            elapsed,
            [mantissa, exponent, exponent_counts, reconstructed],
        )
    }

    fn profile(&self, samples: &Bound<'py, PyAny>) -> [f64; 3] {
        const PROFILE_ROUNDS: usize = 7;
        let mut decompose_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut distribution_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut reconstruct_ms = Vec::with_capacity(PROFILE_ROUNDS);
        for _ in 0..PROFILE_ROUNDS {
            let started = Instant::now();
            let decomposition = self
                .frexp
                .call1((samples,))
                .expect("profile dynamic-range decomposition");
            decompose_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
            let mantissa = decomposition
                .get_item(0)
                .expect("profile dynamic-range mantissa");
            let exponent = decomposition
                .get_item(1)
                .expect("profile dynamic-range exponent");

            let started = Instant::now();
            black_box(
                self.bincount
                    .call((&exponent,), Some(&self.bincount_kwargs))
                    .expect("profile dynamic-range exponent distribution"),
            );
            distribution_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            black_box(
                self.ldexp
                    .call1((&mantissa, &exponent))
                    .expect("profile dynamic-range reconstruction"),
            );
            reconstruct_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
        let median = |samples: &mut Vec<f64>| {
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2]
        };
        [
            median(&mut decompose_ms),
            median(&mut distribution_ms),
            median(&mut reconstruct_ms),
        ]
    }
}

struct VectorFieldPolarArm<'py> {
    hypot: Bound<'py, PyAny>,
    arctan2: Bound<'py, PyAny>,
    average: Bound<'py, PyAny>,
    maximum: Bound<'py, PyAny>,
}

impl<'py> VectorFieldPolarArm<'py> {
    fn run(
        &self,
        velocity_x: &Bound<'py, PyAny>,
        velocity_y: &Bound<'py, PyAny>,
    ) -> (Duration, [Bound<'py, PyAny>; 4]) {
        let started = Instant::now();
        let magnitude = self
            .hypot
            .call1((black_box(velocity_x), black_box(velocity_y)))
            .expect("vector-field magnitude");
        let heading = self
            .arctan2
            .call1((black_box(velocity_y), black_box(velocity_x)))
            .expect("vector-field heading");
        let mean_magnitude = self
            .average
            .call1((black_box(&magnitude), 1_i64))
            .expect("per-frame mean magnitude");
        let peak_magnitude = self
            .maximum
            .call1((black_box(&magnitude), 1_i64))
            .expect("per-frame peak magnitude");
        let elapsed = started.elapsed();
        (
            elapsed,
            [magnitude, heading, mean_magnitude, peak_magnitude],
        )
    }

    fn profile(&self, velocity_x: &Bound<'py, PyAny>, velocity_y: &Bound<'py, PyAny>) -> [f64; 4] {
        const PROFILE_ROUNDS: usize = 7;
        let mut hypot_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut arctan2_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut average_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut maximum_ms = Vec::with_capacity(PROFILE_ROUNDS);
        for _ in 0..PROFILE_ROUNDS {
            let started = Instant::now();
            let magnitude = self
                .hypot
                .call1((velocity_x, velocity_y))
                .expect("profile vector-field magnitude");
            hypot_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            black_box(
                self.arctan2
                    .call1((velocity_y, velocity_x))
                    .expect("profile vector-field heading"),
            );
            arctan2_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            black_box(
                self.average
                    .call1((&magnitude, 1_i64))
                    .expect("profile per-frame mean magnitude"),
            );
            average_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            black_box(
                self.maximum
                    .call1((&magnitude, 1_i64))
                    .expect("profile per-frame peak magnitude"),
            );
            maximum_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
        let median = |samples: &mut Vec<f64>| {
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2]
        };
        [
            median(&mut hypot_ms),
            median(&mut arctan2_ms),
            median(&mut average_ms),
            median(&mut maximum_ms),
        ]
    }
}

struct AccessControlExposureArm<'py> {
    matmul: Bound<'py, PyAny>,
    sum: Bound<'py, PyAny>,
    maximum: Bound<'py, PyAny>,
}

impl<'py> AccessControlExposureArm<'py> {
    fn run(
        &self,
        account_roles: &Bound<'py, PyAny>,
        role_permissions: &Bound<'py, PyAny>,
    ) -> (Duration, [Bound<'py, PyAny>; 4]) {
        let started = Instant::now();
        let exposure = self
            .matmul
            .call1((black_box(account_roles), black_box(role_permissions)))
            .expect("access-control exposure propagation");
        let account_total = self
            .sum
            .call1((black_box(&exposure), 1_i64))
            .expect("per-account total exposure");
        let account_peak = self
            .maximum
            .call1((black_box(&exposure), 1_i64))
            .expect("per-account peak exposure");
        let fleet_total = self
            .sum
            .call1((black_box(&account_total),))
            .expect("fleet total exposure");
        let elapsed = started.elapsed();
        (
            elapsed,
            [exposure, account_total, account_peak, fleet_total],
        )
    }

    fn profile(
        &self,
        account_roles: &Bound<'py, PyAny>,
        role_permissions: &Bound<'py, PyAny>,
    ) -> [f64; 4] {
        const PROFILE_ROUNDS: usize = 7;
        let mut matmul_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut account_sum_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut account_max_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut fleet_sum_ms = Vec::with_capacity(PROFILE_ROUNDS);
        for _ in 0..PROFILE_ROUNDS {
            let started = Instant::now();
            let exposure = self
                .matmul
                .call1((account_roles, role_permissions))
                .expect("profile access-control exposure propagation");
            matmul_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            let account_total = self
                .sum
                .call1((&exposure, 1_i64))
                .expect("profile per-account total exposure");
            account_sum_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            black_box(
                self.maximum
                    .call1((&exposure, 1_i64))
                    .expect("profile per-account peak exposure"),
            );
            account_max_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            black_box(
                self.sum
                    .call1((&account_total,))
                    .expect("profile fleet total exposure"),
            );
            fleet_sum_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
        let median = |samples: &mut Vec<f64>| {
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2]
        };
        [
            median(&mut matmul_ms),
            median(&mut account_sum_ms),
            median(&mut account_max_ms),
            median(&mut fleet_sum_ms),
        ]
    }
}

struct CriticalAccessExposureArm<'py> {
    matmul: Bound<'py, PyAny>,
    maximum: Bound<'py, PyAny>,
    argmax: Bound<'py, PyAny>,
    ptp: Bound<'py, PyAny>,
}

impl<'py> CriticalAccessExposureArm<'py> {
    fn run(
        &self,
        account_roles: &Bound<'py, PyAny>,
        role_permissions: &Bound<'py, PyAny>,
    ) -> (Duration, [Bound<'py, PyAny>; 4]) {
        let started = Instant::now();
        let exposure = self
            .matmul
            .call1((black_box(account_roles), black_box(role_permissions)))
            .expect("critical-access exposure propagation");
        let account_peak = self
            .maximum
            .call1((black_box(&exposure), 1_i64))
            .expect("per-account peak exposure");
        let critical_permission = self
            .argmax
            .call1((black_box(&exposure), 1_i64))
            .expect("per-account critical permission");
        let fleet_peak_spread = self
            .ptp
            .call1((black_box(&account_peak),))
            .expect("fleet peak-exposure spread");
        let elapsed = started.elapsed();
        (
            elapsed,
            [
                exposure,
                account_peak,
                critical_permission,
                fleet_peak_spread,
            ],
        )
    }

    fn profile(
        &self,
        account_roles: &Bound<'py, PyAny>,
        role_permissions: &Bound<'py, PyAny>,
    ) -> [f64; 4] {
        const PROFILE_ROUNDS: usize = 7;
        let mut matmul_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut account_max_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut account_argmax_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut fleet_ptp_ms = Vec::with_capacity(PROFILE_ROUNDS);
        for _ in 0..PROFILE_ROUNDS {
            let started = Instant::now();
            let exposure = self
                .matmul
                .call1((account_roles, role_permissions))
                .expect("profile critical-access exposure propagation");
            matmul_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            let account_peak = self
                .maximum
                .call1((&exposure, 1_i64))
                .expect("profile per-account peak exposure");
            account_max_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            black_box(
                self.argmax
                    .call1((&exposure, 1_i64))
                    .expect("profile per-account critical permission"),
            );
            account_argmax_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            black_box(
                self.ptp
                    .call1((&account_peak,))
                    .expect("profile fleet peak-exposure spread"),
            );
            fleet_ptp_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
        let median = |samples: &mut Vec<f64>| {
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2]
        };
        [
            median(&mut matmul_ms),
            median(&mut account_max_ms),
            median(&mut account_argmax_ms),
            median(&mut fleet_ptp_ms),
        ]
    }
}

struct RollingLoadSaturationArm<'py> {
    convolve: Bound<'py, PyAny>,
    maximum_accumulate: Bound<'py, PyAny>,
    ptp: Bound<'py, PyAny>,
}

impl<'py> RollingLoadSaturationArm<'py> {
    fn run(
        &self,
        requests_per_second: &Bound<'py, PyAny>,
        rolling_window: &Bound<'py, PyAny>,
    ) -> (Duration, [Bound<'py, PyAny>; 3]) {
        let started = Instant::now();
        let rolling_load = self
            .convolve
            .call1((
                black_box(requests_per_second),
                black_box(rolling_window),
                "valid",
            ))
            .expect("rolling 60-second request totals");
        let high_water_envelope = self
            .maximum_accumulate
            .call1((black_box(&rolling_load),))
            .expect("running rolling-load high-water envelope");
        let rolling_spread = self
            .ptp
            .call1((black_box(&rolling_load),))
            .expect("rolling-load peak-to-peak spread");
        let elapsed = started.elapsed();
        (elapsed, [rolling_load, high_water_envelope, rolling_spread])
    }

    fn profile(
        &self,
        requests_per_second: &Bound<'py, PyAny>,
        rolling_window: &Bound<'py, PyAny>,
    ) -> [f64; 3] {
        const PROFILE_ROUNDS: usize = 7;
        let mut convolve_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut high_water_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut spread_ms = Vec::with_capacity(PROFILE_ROUNDS);
        for _ in 0..PROFILE_ROUNDS {
            let started = Instant::now();
            let rolling_load = self
                .convolve
                .call1((requests_per_second, rolling_window, "valid"))
                .expect("profile rolling 60-second request totals");
            convolve_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            black_box(
                self.maximum_accumulate
                    .call1((&rolling_load,))
                    .expect("profile rolling-load high-water envelope"),
            );
            high_water_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            black_box(
                self.ptp
                    .call1((&rolling_load,))
                    .expect("profile rolling-load spread"),
            );
            spread_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
        let median = |samples: &mut Vec<f64>| {
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2]
        };
        [
            median(&mut convolve_ms),
            median(&mut high_water_ms),
            median(&mut spread_ms),
        ]
    }
}

/// One arm of the clickstream sessionization job. Both arms hold their own
/// module's `lexsort`, `take`, `diff`, and `count_nonzero` and execute the
/// identical six public calls over the identical read-only inputs.
struct ClickstreamSessionizationArm<'py> {
    lexsort: Bound<'py, PyAny>,
    take: Bound<'py, PyAny>,
    diff: Bound<'py, PyAny>,
    count_nonzero: Bound<'py, PyAny>,
}

impl<'py> ClickstreamSessionizationArm<'py> {
    /// `sort_keys` is the pre-built `(event_time, user_id)` key tuple — NumPy's
    /// LAST key is primary, so this orders events by user, then by time. The
    /// tuple is constructed once outside the timer so neither arm pays tuple
    /// allocation inside the measured region.
    fn run(
        &self,
        sort_keys: &Bound<'py, PyAny>,
        user_ids: &Bound<'py, PyAny>,
        event_times: &Bound<'py, PyAny>,
    ) -> (Duration, [Bound<'py, PyAny>; 6]) {
        let started = Instant::now();
        let session_order = self
            .lexsort
            .call1((black_box(sort_keys),))
            .expect("session-order permutation");
        let ordered_users = self
            .take
            .call1((black_box(user_ids), &session_order))
            .expect("user column in session order");
        let ordered_times = self
            .take
            .call1((black_box(event_times), &session_order))
            .expect("timestamp column in session order");
        let user_boundary = self
            .diff
            .call1((&ordered_users,))
            .expect("user-boundary column");
        let inter_event_gap = self
            .diff
            .call1((&ordered_times,))
            .expect("inter-event gap column");
        let user_transitions = self
            .count_nonzero
            .call1((&user_boundary,))
            .expect("user-transition count");
        let elapsed = started.elapsed();
        (
            elapsed,
            [
                session_order,
                ordered_users,
                ordered_times,
                user_boundary,
                inter_event_gap,
                user_transitions,
            ],
        )
    }

    fn profile(
        &self,
        sort_keys: &Bound<'py, PyAny>,
        user_ids: &Bound<'py, PyAny>,
        event_times: &Bound<'py, PyAny>,
    ) -> [f64; 6] {
        const PROFILE_ROUNDS: usize = 7;
        let mut order_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut gather_users_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut gather_times_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut boundary_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut gap_ms = Vec::with_capacity(PROFILE_ROUNDS);
        let mut transitions_ms = Vec::with_capacity(PROFILE_ROUNDS);
        for _ in 0..PROFILE_ROUNDS {
            let started = Instant::now();
            let session_order = self
                .lexsort
                .call1((sort_keys,))
                .expect("profile session-order permutation");
            order_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            let ordered_users = self
                .take
                .call1((user_ids, &session_order))
                .expect("profile user column in session order");
            gather_users_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            let ordered_times = self
                .take
                .call1((event_times, &session_order))
                .expect("profile timestamp column in session order");
            gather_times_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            let user_boundary = self
                .diff
                .call1((&ordered_users,))
                .expect("profile user-boundary column");
            boundary_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            black_box(
                self.diff
                    .call1((&ordered_times,))
                    .expect("profile inter-event gap column"),
            );
            gap_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

            let started = Instant::now();
            black_box(
                self.count_nonzero
                    .call1((&user_boundary,))
                    .expect("profile user-transition count"),
            );
            transitions_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
        let median = |samples: &mut Vec<f64>| {
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2]
        };
        [
            median(&mut order_ms),
            median(&mut gather_users_ms),
            median(&mut gather_times_ms),
            median(&mut boundary_ms),
            median(&mut gap_ms),
            median(&mut transitions_ms),
        ]
    }
}

fn observe_inter_event_gap_result_lifecycle(
    numpy: &Bound<'_, PyModule>,
    diff: &Bound<'_, PyAny>,
    ordered_times: &Bound<'_, PyAny>,
    lifecycles: &RefCell<Vec<ResultBufferLifecycle>>,
) -> common::ContractObservation {
    let before = process_resource_snapshot();
    let started = Instant::now();
    let output = diff
        .call1((black_box(ordered_times),))
        .expect("materialize inter-event gap result");
    let elapsed = started.elapsed();
    let while_live = process_resource_snapshot();
    let checksum = workload_checksum(numpy, std::array::from_ref(&output));
    drop(output);
    let after_release = process_resource_snapshot();
    lifecycles
        .borrow_mut()
        .push(result_buffer_lifecycle(before, while_live, after_release));
    common::ContractObservation { elapsed, checksum }
}

fn bench_median_gate_python_binary<'py>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    bench_name: &'static str,
    row: &'static str,
    base: &Bound<'py, PyAny>,
    candidate: &Bound<'py, PyAny>,
    lhs: &Bound<'py, PyAny>,
    rhs: &Bound<'py, PyAny>,
) {
    let null_base_ns = RefCell::new(Vec::new());
    let null_peer_ns = RefCell::new(Vec::new());
    let null_ratios = RefCell::new(Vec::new());
    let base_ns = RefCell::new(Vec::new());
    let candidate_ns = RefCell::new(Vec::new());
    let effect_ratios = RefCell::new(Vec::new());
    group.bench_function(bench_name, |bench| {
        bench.iter_custom(|iterations| {
            let mut combined = Duration::ZERO;
            for _ in 0..iterations {
                // The exact-function A/A null is always measured first. The two observations
                // are ABBA then BAAB, balancing call position before the effect is observed.
                for observation in 0..MEDIAN_GATE_OBSERVATIONS_PER_BATCH {
                    let outer_base = observation & 1 == 0;
                    let (base_total, peer_total) = if outer_base {
                        let a1 = time_python_binary_call(base, lhs, rhs);
                        let b1 = time_python_binary_call(base, lhs, rhs);
                        let b2 = time_python_binary_call(base, lhs, rhs);
                        let a2 = time_python_binary_call(base, lhs, rhs);
                        (a1 + a2, b1 + b2)
                    } else {
                        let b1 = time_python_binary_call(base, lhs, rhs);
                        let a1 = time_python_binary_call(base, lhs, rhs);
                        let a2 = time_python_binary_call(base, lhs, rhs);
                        let b2 = time_python_binary_call(base, lhs, rhs);
                        (a1 + a2, b1 + b2)
                    };
                    let base_average = base_total.as_secs_f64() * 0.5e9;
                    let peer_average = peer_total.as_secs_f64() * 0.5e9;
                    null_base_ns.borrow_mut().push(base_average);
                    null_peer_ns.borrow_mut().push(peer_average);
                    null_ratios.borrow_mut().push(base_average / peer_average);
                    combined += base_total + peer_total;
                }
                for observation in 0..MEDIAN_GATE_OBSERVATIONS_PER_BATCH {
                    let outer_base = observation & 1 == 0;
                    let (base_total, candidate_total) = if outer_base {
                        let a1 = time_python_binary_call(base, lhs, rhs);
                        let b1 = time_python_binary_call(candidate, lhs, rhs);
                        let b2 = time_python_binary_call(candidate, lhs, rhs);
                        let a2 = time_python_binary_call(base, lhs, rhs);
                        (a1 + a2, b1 + b2)
                    } else {
                        let b1 = time_python_binary_call(candidate, lhs, rhs);
                        let a1 = time_python_binary_call(base, lhs, rhs);
                        let a2 = time_python_binary_call(base, lhs, rhs);
                        let b2 = time_python_binary_call(candidate, lhs, rhs);
                        (a1 + a2, b1 + b2)
                    };
                    let base_average = base_total.as_secs_f64() * 0.5e9;
                    let candidate_average = candidate_total.as_secs_f64() * 0.5e9;
                    base_ns.borrow_mut().push(base_average);
                    candidate_ns.borrow_mut().push(candidate_average);
                    effect_ratios
                        .borrow_mut()
                        .push(base_average / candidate_average);
                    combined += base_total + candidate_total;
                }
            }
            combined
        });
    });
    report_median_gate_pair(
        row,
        &null_base_ns,
        &null_peer_ns,
        &null_ratios,
        &base_ns,
        &candidate_ns,
        &effect_ratios,
    );
}

fn bench_median_gate_python_unary<'py>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    bench_name: &'static str,
    row: &'static str,
    base: &Bound<'py, PyAny>,
    candidate: &Bound<'py, PyAny>,
    input: &Bound<'py, PyAny>,
) {
    let null_base_ns = RefCell::new(Vec::new());
    let null_peer_ns = RefCell::new(Vec::new());
    let null_ratios = RefCell::new(Vec::new());
    let base_ns = RefCell::new(Vec::new());
    let candidate_ns = RefCell::new(Vec::new());
    let effect_ratios = RefCell::new(Vec::new());
    group.bench_function(bench_name, |bench| {
        bench.iter_custom(|iterations| {
            let mut combined = Duration::ZERO;
            for _ in 0..iterations {
                for observation in 0..MEDIAN_GATE_OBSERVATIONS_PER_BATCH {
                    let outer_base = observation & 1 == 0;
                    let (base_total, peer_total) = if outer_base {
                        let a1 = time_python_unary_call(base, input);
                        let b1 = time_python_unary_call(base, input);
                        let b2 = time_python_unary_call(base, input);
                        let a2 = time_python_unary_call(base, input);
                        (a1 + a2, b1 + b2)
                    } else {
                        let b1 = time_python_unary_call(base, input);
                        let a1 = time_python_unary_call(base, input);
                        let a2 = time_python_unary_call(base, input);
                        let b2 = time_python_unary_call(base, input);
                        (a1 + a2, b1 + b2)
                    };
                    let base_average = base_total.as_secs_f64() * 0.5e9;
                    let peer_average = peer_total.as_secs_f64() * 0.5e9;
                    null_base_ns.borrow_mut().push(base_average);
                    null_peer_ns.borrow_mut().push(peer_average);
                    null_ratios.borrow_mut().push(base_average / peer_average);
                    combined += base_total + peer_total;
                }
                for observation in 0..MEDIAN_GATE_OBSERVATIONS_PER_BATCH {
                    let outer_base = observation & 1 == 0;
                    let (base_total, candidate_total) = if outer_base {
                        let a1 = time_python_unary_call(base, input);
                        let b1 = time_python_unary_call(candidate, input);
                        let b2 = time_python_unary_call(candidate, input);
                        let a2 = time_python_unary_call(base, input);
                        (a1 + a2, b1 + b2)
                    } else {
                        let b1 = time_python_unary_call(candidate, input);
                        let a1 = time_python_unary_call(base, input);
                        let a2 = time_python_unary_call(base, input);
                        let b2 = time_python_unary_call(candidate, input);
                        (a1 + a2, b1 + b2)
                    };
                    let base_average = base_total.as_secs_f64() * 0.5e9;
                    let candidate_average = candidate_total.as_secs_f64() * 0.5e9;
                    base_ns.borrow_mut().push(base_average);
                    candidate_ns.borrow_mut().push(candidate_average);
                    effect_ratios
                        .borrow_mut()
                        .push(base_average / candidate_average);
                    combined += base_total + candidate_total;
                }
            }
            combined
        });
    });
    report_median_gate_pair(
        row,
        &null_base_ns,
        &null_peer_ns,
        &null_ratios,
        &base_ns,
        &candidate_ns,
        &effect_ratios,
    );
}

fn bench_substrate_v2_python_binary_pair<'py>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    bench_name: &'static str,
    row: &'static str,
    candidate: &Bound<'py, PyAny>,
    orig: &Bound<'py, PyAny>,
    lhs: &Bound<'py, PyAny>,
    rhs: &Bound<'py, PyAny>,
) {
    let candidate_samples = RefCell::new(Vec::new());
    let orig_samples = RefCell::new(Vec::new());
    let order = Cell::new(0_u64);
    group.bench_function(bench_name, |bench| {
        bench.iter_custom(|iterations| {
            // Slow Python surface rows otherwise collapse to one A/B pair per
            // Criterion sample.  Keep each sample order-balanced and average
            // enough interleaved pairs to make worker jitter visible instead
            // of letting a single interruption decide the row.
            let measured_iterations = iterations.max(4);
            let measured_iterations = measured_iterations + (measured_iterations & 1);
            let mut candidate_total = Duration::ZERO;
            let mut orig_total = Duration::ZERO;
            for _ in 0..measured_iterations {
                let orig_first = order.get() & 1 == 1;
                order.set(order.get().wrapping_add(1));
                let time_call = |function: &Bound<'py, PyAny>| {
                    let start = Instant::now();
                    let lhs = black_box(lhs);
                    let rhs = black_box(rhs);
                    let result = function
                        .call1((lhs, rhs))
                        .expect("paired binary Python call");
                    black_box(result);
                    start.elapsed()
                };
                if orig_first {
                    orig_total += time_call(orig);
                    candidate_total += time_call(candidate);
                } else {
                    candidate_total += time_call(candidate);
                    orig_total += time_call(orig);
                }
            }
            candidate_samples
                .borrow_mut()
                .push(candidate_total.as_secs_f64() * 1e9 / measured_iterations as f64);
            orig_samples
                .borrow_mut()
                .push(orig_total.as_secs_f64() * 1e9 / measured_iterations as f64);
            (candidate_total + orig_total).mul_f64(iterations as f64 / measured_iterations as f64)
        });
    });
    report_substrate_v2_pair(row, &candidate_samples, &orig_samples);
}

fn bench_substrate_v2_python_unary_pair<'py>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    bench_name: &'static str,
    row: &'static str,
    candidate: &Bound<'py, PyAny>,
    orig: &Bound<'py, PyAny>,
    input: &Bound<'py, PyAny>,
) {
    let candidate_samples = RefCell::new(Vec::new());
    let orig_samples = RefCell::new(Vec::new());
    let order = Cell::new(0_u64);
    group.bench_function(bench_name, |bench| {
        bench.iter_custom(|iterations| {
            let measured_iterations = iterations.max(4);
            let measured_iterations = measured_iterations + (measured_iterations & 1);
            let mut candidate_total = Duration::ZERO;
            let mut orig_total = Duration::ZERO;
            for _ in 0..measured_iterations {
                let orig_first = order.get() & 1 == 1;
                order.set(order.get().wrapping_add(1));
                let time_call = |function: &Bound<'py, PyAny>| {
                    let start = Instant::now();
                    let input = black_box(input);
                    let result = function.call1((input,)).expect("paired unary Python call");
                    black_box(result);
                    start.elapsed()
                };
                if orig_first {
                    orig_total += time_call(orig);
                    candidate_total += time_call(candidate);
                } else {
                    candidate_total += time_call(candidate);
                    orig_total += time_call(orig);
                }
            }
            candidate_samples
                .borrow_mut()
                .push(candidate_total.as_secs_f64() * 1e9 / measured_iterations as f64);
            orig_samples
                .borrow_mut()
                .push(orig_total.as_secs_f64() * 1e9 / measured_iterations as f64);
            (candidate_total + orig_total).mul_f64(iterations as f64 / measured_iterations as f64)
        });
    });
    report_substrate_v2_pair(row, &candidate_samples, &orig_samples);
}

fn bench_completion_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_completion_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        println!(
            "ISA_PROVENANCE target_arch={} avx2={} sse2={}",
            std::env::consts::ARCH,
            cfg!(target_feature = "avx2"),
            cfg!(target_feature = "sse2"),
        );
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_completion_median_gate")
            .expect("completion bench module");
        fnp_python(&module).expect("initialize fnp_python completion bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy version")
            .extract::<String>()
            .expect("numpy version string");
        let numpy_simd = numpy
            .getattr("__config__")
            .expect("numpy config")
            .getattr("CONFIG")
            .expect("numpy CONFIG")
            .get_item("SIMD Extensions")
            .expect("numpy SIMD Extensions")
            .str()
            .expect("numpy SIMD str")
            .extract::<String>()
            .expect("numpy SIMD string value");
        let numpy_cpu_features = numpy
            .getattr("_core")
            .expect("numpy core")
            .getattr("_multiarray_umath")
            .expect("numpy multiarray umath")
            .getattr("__cpu_features__")
            .expect("numpy runtime CPU features")
            .str()
            .expect("numpy runtime CPU feature str")
            .extract::<String>()
            .expect("numpy runtime CPU feature string value");
        println!(
            "NUMPY_PROVENANCE version={numpy_version} build_simd={numpy_simd} \
             runtime_cpu_features={numpy_cpu_features}"
        );
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 powers = np.power(np.uint64(26), np.arange(5, dtype=np.uint64))\n\
                 u_a_ids = np.arange(0, 1_000_000, dtype=np.uint64)\n\
                 u_a_words = np.zeros((1_000_000, 16), dtype=np.uint32)\n\
                 u_a_words[:, :5] = (97 + (u_a_ids[:, None] // powers) % 26).astype(np.uint32)\n\
                 u_a = u_a_words.reshape(-1).view('U16')\n\
                 u_fresh_ids = np.arange(1_000_000, 1_500_000, dtype=np.uint64)\n\
                 u_fresh_words = np.zeros((500_000, 16), dtype=np.uint32)\n\
                 u_fresh_words[:, :5] = (97 + (u_fresh_ids[:, None] // powers) % 26).astype(np.uint32)\n\
                 u_fresh = u_fresh_words.reshape(-1).view('U16')\n\
                 u_b = np.concatenate([u_a[:500_000], u_fresh])\n\
                 u_union_ids = np.arange(2_000_000, 3_000_000, dtype=np.uint64)\n\
                 u_union_words = np.zeros((1_000_000, 16), dtype=np.uint32)\n\
                 u_union_words[:, :5] = (97 + (u_union_ids[:, None] // powers) % 26).astype(np.uint32)\n\
                 u_union_b = u_union_words.reshape(-1).view('U16')\n",
            )
            .expect("completion setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("completion setup");
        let u_a = namespace.get_item("u_a").expect("u_a present");
        let u_b = namespace.get_item("u_b").expect("u_b present");
        let u_union_b = namespace.get_item("u_union_b").expect("u_union_b present");
        let array_equal = numpy.getattr("array_equal").expect("numpy.array_equal");

        let fnp_unique = module.getattr("unique").expect("fnp unique");
        let np_unique = numpy.getattr("unique").expect("numpy unique");
        let fnp_union = module.getattr("union1d").expect("fnp union1d");
        let np_union = numpy.getattr("union1d").expect("numpy union1d");
        let fnp_setxor = module.getattr("setxor1d").expect("fnp setxor1d");
        let np_setxor = numpy.getattr("setxor1d").expect("numpy setxor1d");

        for (label, candidate, base) in [
            (
                "U16 unique",
                fnp_unique.call1((&u_a,)).expect("fnp unique parity"),
                np_unique.call1((&u_a,)).expect("numpy unique parity"),
            ),
            (
                "U16 disjoint union",
                fnp_union
                    .call1((&u_a, &u_union_b))
                    .expect("fnp union parity"),
                np_union
                    .call1((&u_a, &u_union_b))
                    .expect("numpy union parity"),
            ),
            (
                "U16 50% overlap setxor",
                fnp_setxor.call1((&u_a, &u_b)).expect("fnp setxor parity"),
                np_setxor.call1((&u_a, &u_b)).expect("numpy setxor parity"),
            ),
        ] {
            let candidate_dtype = candidate.getattr("dtype").expect("candidate dtype");
            let base_dtype = base.getattr("dtype").expect("base dtype");
            assert_eq!(
                candidate_dtype
                    .getattr("str")
                    .expect("candidate dtype str")
                    .extract::<String>()
                    .expect("candidate dtype str value"),
                base_dtype
                    .getattr("str")
                    .expect("base dtype str")
                    .extract::<String>()
                    .expect("base dtype str value"),
                "{label} dtype string parity",
            );
            assert!(
                candidate_dtype
                    .getattr("metadata")
                    .expect("candidate dtype metadata")
                    .eq(base_dtype.getattr("metadata").expect("base dtype metadata"))
                    .expect("dtype metadata equality"),
                "{label} dtype metadata parity",
            );
            assert!(
                array_equal
                    .call1((&candidate, &base))
                    .expect("completion array_equal")
                    .extract::<bool>()
                    .expect("completion array_equal bool"),
                "{label} value parity",
            );
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "{label} byte parity",
            );
        }

        bench_median_gate_python_unary(
            &mut group,
            "u16_unique_1m_null_then_effect",
            "u16_unique_1m",
            &np_unique,
            &fnp_unique,
            &u_a,
        );
        bench_median_gate_python_binary(
            &mut group,
            "u16_union_disjoint_1m_null_then_effect",
            "u16_union_disjoint_1m",
            &np_union,
            &fnp_union,
            &u_a,
            &u_union_b,
        );
        bench_median_gate_python_binary(
            &mut group,
            "u16_setxor_1m_null_then_effect",
            "u16_setxor_1m",
            &np_setxor,
            &fnp_setxor,
            &u_a,
            &u_b,
        );
    });

    group.finish();
}

fn bench_f64_transcendental_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_f64_transcendental_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        println!(
            "ISA_PROVENANCE target_arch={} avx2={} sse2={}",
            std::env::consts::ARCH,
            cfg!(target_feature = "avx2"),
            cfg!(target_feature = "sse2"),
        );
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_f64_transcendental_median_gate")
            .expect("transcendental bench module");
        fnp_python(&module).expect("initialize fnp_python transcendental bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy version")
            .extract::<String>()
            .expect("numpy version string");
        let numpy_cpu_features = numpy
            .getattr("_core")
            .expect("numpy core")
            .getattr("_multiarray_umath")
            .expect("numpy multiarray umath")
            .getattr("__cpu_features__")
            .expect("numpy runtime CPU features")
            .str()
            .expect("numpy runtime CPU feature str")
            .extract::<String>()
            .expect("numpy runtime CPU feature string value");
        println!(
            "NUMPY_PROVENANCE version={numpy_version} runtime_cpu_features={numpy_cpu_features}"
        );
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260710)\n\
                 t_262k = rng.standard_normal(262_144)\n\
                 t_1m = rng.standard_normal(1_048_576)\n\
                 t_4m = rng.standard_normal(4_194_304)\n",
            )
            .expect("transcendental setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("transcendental setup");
        let t_262k = namespace.get_item("t_262k").expect("t_262k present");
        let t_1m = namespace.get_item("t_1m").expect("t_1m present");
        let t_4m = namespace.get_item("t_4m").expect("t_4m present");

        // Diagnostic parity probe (print, not assert): fnp's native f64 route is
        // scalar system libm; numpy may dispatch a SIMD kernel on some workers.
        // Byte-level agreement per worker is itself evidence for the transcendental
        // lane (see the 2026-07-10 ISA addendum), so record it instead of dying.
        for name in ["sin", "cos", "tan", "tanh", "expm1"] {
            let fnp_fn = module.getattr(name).expect("fnp transcendental fn");
            let np_fn = numpy.getattr(name).expect("numpy transcendental fn");
            for (label, input) in [("262k", &t_262k), ("1m", &t_1m), ("4m", &t_4m)] {
                let candidate = fnp_fn.call1((input,)).expect("fnp parity call");
                let base = np_fn.call1((input,)).expect("numpy parity call");
                let candidate_bytes: Vec<u8> = candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract()
                    .expect("candidate byte Vec");
                let base_bytes: Vec<u8> = base
                    .call_method0("tobytes")
                    .expect("base bytes")
                    .extract()
                    .expect("base byte Vec");
                let first_diff = candidate_bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .zip(base_bytes.as_chunks::<8>().0.iter())
                    .position(|(a, b)| a != b);
                let diff_count = candidate_bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .zip(base_bytes.as_chunks::<8>().0.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                println!(
                    "TRANSCENDENTAL_PARITY op={name} n={label} byte_equal={} \
                     diff_elems={diff_count} first_diff_elem={:?}",
                    candidate_bytes == base_bytes,
                    first_diff,
                );
            }
        }

        let fnp_sin = module.getattr("sin").expect("fnp sin");
        let np_sin = numpy.getattr("sin").expect("numpy sin");
        let fnp_cos = module.getattr("cos").expect("fnp cos");
        let np_cos = numpy.getattr("cos").expect("numpy cos");
        let fnp_tan = module.getattr("tan").expect("fnp tan");
        let np_tan = numpy.getattr("tan").expect("numpy tan");
        let fnp_tanh = module.getattr("tanh").expect("fnp tanh");
        let np_tanh = numpy.getattr("tanh").expect("numpy tanh");
        let fnp_expm1 = module.getattr("expm1").expect("fnp expm1");
        let np_expm1 = numpy.getattr("expm1").expect("numpy expm1");

        bench_median_gate_python_unary(
            &mut group,
            "f64_sin_262k_null_then_effect",
            "f64_sin_262k",
            &np_sin,
            &fnp_sin,
            &t_262k,
        );
        bench_median_gate_python_unary(
            &mut group,
            "f64_sin_1m_null_then_effect",
            "f64_sin_1m",
            &np_sin,
            &fnp_sin,
            &t_1m,
        );
        bench_median_gate_python_unary(
            &mut group,
            "f64_sin_4m_null_then_effect",
            "f64_sin_4m",
            &np_sin,
            &fnp_sin,
            &t_4m,
        );
        bench_median_gate_python_unary(
            &mut group,
            "f64_cos_1m_null_then_effect",
            "f64_cos_1m",
            &np_cos,
            &fnp_cos,
            &t_1m,
        );
        bench_median_gate_python_unary(
            &mut group,
            "f64_tan_1m_null_then_effect",
            "f64_tan_1m",
            &np_tan,
            &fnp_tan,
            &t_1m,
        );
        bench_median_gate_python_unary(
            &mut group,
            "f64_tanh_1m_null_then_effect",
            "f64_tanh_1m",
            &np_tanh,
            &fnp_tanh,
            &t_1m,
        );
        bench_median_gate_python_unary(
            &mut group,
            "f64_expm1_262k_null_then_effect",
            "f64_expm1_262k",
            &np_expm1,
            &fnp_expm1,
            &t_262k,
        );
        bench_median_gate_python_unary(
            &mut group,
            "f64_expm1_1m_null_then_effect",
            "f64_expm1_1m",
            &np_expm1,
            &fnp_expm1,
            &t_1m,
        );
    });

    group.finish();
}

fn bench_f64_exp_log_probe(c: &mut Criterion) {
    // Probe for bead deadlock-audit-gkznn (reopen of the stale 2026-06-09
    // exp/log passthrough decision): (1) BYTE PROBE — does numpy's f64
    // exp/log/log2/log10 output match Rust scalar system-libm bit-for-bit on
    // this worker? (2) TIMING — does a rayon parallel scalar-libm map (with a
    // deliberate vec![0.0; n] zero-init handicap the real zero-copy path would
    // not pay) beat numpy's kernel? Both must hold before any production
    // rewiring; the probe writes evidence only.
    let mut group = c.benchmark_group("python_f64_exp_log_probe");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let numpy = py.import("numpy").expect("numpy oracle");
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy version")
            .extract::<String>()
            .expect("numpy version string");
        println!("EXP_LOG_PROBE_NUMPY version={numpy_version}");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 e_1m = rng.standard_normal(1_048_576)\n\
                 e_4m = rng.standard_normal(4_194_304)\n\
                 l_1m = np.abs(rng.standard_normal(1_048_576)) + 0.5\n\
                 l_4m = np.abs(rng.standard_normal(4_194_304)) + 0.5\n",
            )
            .expect("probe setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("probe setup");
        let e_1m = namespace.get_item("e_1m").expect("e_1m present");
        let e_4m = namespace.get_item("e_4m").expect("e_4m present");
        let l_1m = namespace.get_item("l_1m").expect("l_1m present");
        let l_4m = namespace.get_item("l_4m").expect("l_4m present");

        let to_vec = |arr: &Bound<'_, PyAny>| -> Vec<f64> {
            let raw: Vec<u8> = arr
                .call_method0("tobytes")
                .expect("probe input bytes")
                .extract()
                .expect("probe input byte Vec");
            raw.as_chunks::<8>()
                .0
                .iter()
                .map(|chunk| f64::from_ne_bytes(*chunk))
                .collect()
        };

        // (1) BYTE PROBE: numpy output vs Rust scalar libm, element-exact.
        for (name, rust_fn, input) in [
            ("exp", f64::exp as fn(f64) -> f64, &e_1m),
            ("log", f64::ln as fn(f64) -> f64, &l_1m),
            ("log2", f64::log2 as fn(f64) -> f64, &l_1m),
            ("log10", f64::log10 as fn(f64) -> f64, &l_1m),
        ] {
            let data = to_vec(input);
            let np_bytes: Vec<u8> = numpy
                .getattr(name)
                .expect("numpy probe fn")
                .call1((input,))
                .expect("numpy probe call")
                .call_method0("tobytes")
                .expect("numpy probe bytes")
                .extract()
                .expect("numpy probe byte Vec");
            let mut diff_elems = 0usize;
            let mut first_diff = None;
            let mut max_bitdiff: u64 = 0;
            for (index, (np_chunk, &value)) in np_bytes
                .as_chunks::<8>()
                .0
                .iter()
                .zip(data.iter())
                .enumerate()
            {
                let np_bits = u64::from_ne_bytes(*np_chunk);
                let mine_bits = rust_fn(value).to_bits();
                if np_bits != mine_bits {
                    diff_elems += 1;
                    if first_diff.is_none() {
                        first_diff = Some(index);
                    }
                    max_bitdiff = max_bitdiff.max(np_bits.abs_diff(mine_bits));
                }
            }
            println!(
                "EXP_LOG_PROBE op={name} n=1m byte_equal={} diff_elems={diff_elems} \
                 first_diff_elem={first_diff:?} max_bitdiff={max_bitdiff}",
                diff_elems == 0,
            );
        }

        // (2) TIMING: ledger-pair ABBA — candidate = parallel scalar-libm map
        // (zero-init handicap), orig = the numpy call. Plus numpy A/A nulls.
        for (row, name, rust_fn, input) in [
            (
                "exp_log_probe_exp_1m",
                "exp",
                f64::exp as fn(f64) -> f64,
                &e_1m,
            ),
            (
                "exp_log_probe_exp_4m",
                "exp",
                f64::exp as fn(f64) -> f64,
                &e_4m,
            ),
            (
                "exp_log_probe_log_1m",
                "log",
                f64::ln as fn(f64) -> f64,
                &l_1m,
            ),
            (
                "exp_log_probe_log_4m",
                "log",
                f64::ln as fn(f64) -> f64,
                &l_4m,
            ),
        ] {
            let data = to_vec(input);
            let np_fn = numpy.getattr(name).expect("numpy timing fn");
            let run_candidate = || {
                let n = data.len();
                let mut out = vec![0.0f64; n];
                let chunk = n.div_ceil(rayon::current_num_threads().max(1));
                out.par_chunks_mut(chunk)
                    .zip(data.par_chunks(chunk))
                    .for_each(|(o, i)| {
                        for (slot, &value) in o.iter_mut().zip(i.iter()) {
                            *slot = rust_fn(value);
                        }
                    });
                out
            };
            let candidate_samples = RefCell::new(Vec::new());
            let orig_samples = RefCell::new(Vec::new());
            let order = Cell::new(0u64);
            group.bench_function(format!("{row}_paired"), |bench| {
                bench.iter_custom(|iterations| {
                    let mut candidate_total = Duration::ZERO;
                    let mut orig_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let orig_first = order.get() & 1 == 1;
                        order.set(order.get().wrapping_add(1));
                        if orig_first {
                            let start = Instant::now();
                            black_box(np_fn.call1((input,)).expect("np timing call"));
                            orig_total += start.elapsed();
                            let start = Instant::now();
                            black_box(run_candidate());
                            candidate_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(run_candidate());
                            candidate_total += start.elapsed();
                            let start = Instant::now();
                            black_box(np_fn.call1((input,)).expect("np timing call"));
                            orig_total += start.elapsed();
                        }
                    }
                    candidate_samples
                        .borrow_mut()
                        .push(candidate_total.as_secs_f64() * 1e9 / iterations as f64);
                    orig_samples
                        .borrow_mut()
                        .push(orig_total.as_secs_f64() * 1e9 / iterations as f64);
                    candidate_total + orig_total
                });
            });
            report_ledger_pair(row, &candidate_samples, &orig_samples);

            let null_a = RefCell::new(Vec::new());
            let null_b = RefCell::new(Vec::new());
            let null_order = Cell::new(0u64);
            group.bench_function(format!("{row}_null_aa"), |bench| {
                bench.iter_custom(|iterations| {
                    let mut a_total = Duration::ZERO;
                    let mut b_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let b_first = null_order.get() & 1 == 1;
                        null_order.set(null_order.get().wrapping_add(1));
                        if b_first {
                            let start = Instant::now();
                            black_box(np_fn.call1((input,)).expect("np null call"));
                            b_total += start.elapsed();
                            let start = Instant::now();
                            black_box(np_fn.call1((input,)).expect("np null call"));
                            a_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(np_fn.call1((input,)).expect("np null call"));
                            a_total += start.elapsed();
                            let start = Instant::now();
                            black_box(np_fn.call1((input,)).expect("np null call"));
                            b_total += start.elapsed();
                        }
                    }
                    null_a
                        .borrow_mut()
                        .push(a_total.as_secs_f64() * 1e9 / iterations as f64);
                    null_b
                        .borrow_mut()
                        .push(b_total.as_secs_f64() * 1e9 / iterations as f64);
                    a_total + b_total
                });
            });
            report_ledger_pair(&format!("{row}_null"), &null_a, &null_b);
        }
    });

    group.finish();
}

fn bench_f64_exp_log_median_gate(c: &mut Criterion) {
    // SHIP rows for bead deadlock-audit-gkznn: the ACTUAL wired route
    // (fnp.exp/log/log2/log10 -> try_zerocopy_f64_unary parallel scalar-libm
    // map on non-AVX-512 hosts) vs numpy, with pre-timing byte parity asserts.
    // On an avx512f worker the ISA gate routes these to the numpy passthrough
    // and the rows read ~1.0x by construction; the probe group's
    // EXP_LOG_PROBE byte rows identify the worker class in the same run.
    let mut group = c.benchmark_group("python_f64_exp_log_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module =
            PyModule::new(py, "fnp_python_exp_log_median_gate").expect("exp/log bench module");
        fnp_python(&module).expect("initialize fnp_python exp/log bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        // Worker-class provenance: the rows below are only expected to beat
        // numpy where the ISA gate enables the native route (x86-64 with
        // avx512f=false); elsewhere they measure passthrough-vs-numpy ~1.0x.
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy version")
            .extract::<String>()
            .expect("numpy version string");
        #[cfg(target_arch = "x86_64")]
        let native_route = !std::arch::is_x86_feature_detected!("avx512f");
        #[cfg(not(target_arch = "x86_64"))]
        let native_route = false;
        println!("EXP_LOG_GATE_WORKER numpy={numpy_version} native_route={native_route}");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 e_1m = rng.standard_normal(1_048_576)\n\
                 e_4m = rng.standard_normal(4_194_304)\n\
                 l_1m = np.abs(rng.standard_normal(1_048_576)) + 0.5\n\
                 l_4m = np.abs(rng.standard_normal(4_194_304)) + 0.5\n",
            )
            .expect("exp/log gate setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("exp/log gate setup");
        let e_1m = namespace.get_item("e_1m").expect("e_1m present");
        let e_4m = namespace.get_item("e_4m").expect("e_4m present");
        let l_1m = namespace.get_item("l_1m").expect("l_1m present");
        let l_4m = namespace.get_item("l_4m").expect("l_4m present");

        let rows = [
            (
                "explog_exp_1m_null_then_effect",
                "explog_exp_1m",
                "exp",
                &e_1m,
            ),
            (
                "explog_exp_4m_null_then_effect",
                "explog_exp_4m",
                "exp",
                &e_4m,
            ),
            (
                "explog_exp2_4m_null_then_effect",
                "explog_exp2_4m",
                "exp2",
                &e_4m,
            ),
            (
                "explog_log_1m_null_then_effect",
                "explog_log_1m",
                "log",
                &l_1m,
            ),
            (
                "explog_log_4m_null_then_effect",
                "explog_log_4m",
                "log",
                &l_4m,
            ),
            (
                "explog_log2_4m_null_then_effect",
                "explog_log2_4m",
                "log2",
                &l_4m,
            ),
            (
                "explog_log10_4m_null_then_effect",
                "explog_log10_4m",
                "log10",
                &l_4m,
            ),
        ];
        for (bench_name, row, op, input) in rows {
            let fnp_fn = module.getattr(op).expect("fnp exp/log fn");
            let np_fn = numpy.getattr(op).expect("numpy exp/log fn");
            let candidate = fnp_fn.call1((input,)).expect("fnp exp/log parity call");
            let base = np_fn.call1((input,)).expect("numpy exp/log parity call");
            assert_eq!(
                candidate
                    .getattr("dtype")
                    .expect("candidate dtype")
                    .str()
                    .expect("candidate dtype str")
                    .to_string(),
                base.getattr("dtype")
                    .expect("base dtype")
                    .str()
                    .expect("base dtype str")
                    .to_string(),
                "exp/log {row} dtype parity",
            );
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "exp/log {row} byte parity",
            );
            bench_median_gate_python_unary(&mut group, bench_name, row, &np_fn, &fnp_fn, input);
        }
    });

    group.finish();
}

fn bench_bool_sort_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_bool_sort_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module =
            PyModule::new(py, "fnp_python_bool_sort_median_gate").expect("bool sort bench module");
        fnp_python(&module).expect("initialize fnp_python bool sort bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 b_8m = rng.integers(0, 2, 8_000_000).astype(bool)\n\
                 b_2m = rng.integers(0, 2, 2_000_000).astype(bool)\n",
            )
            .expect("bool sort setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("bool sort setup");
        let b_8m = namespace.get_item("b_8m").expect("b_8m present");
        let b_2m = namespace.get_item("b_2m").expect("b_2m present");

        let fnp_sort = module.getattr("sort").expect("fnp sort");
        let np_sort = numpy.getattr("sort").expect("numpy sort");
        for (label, input) in [("8m", &b_8m), ("2m", &b_2m)] {
            let candidate = fnp_sort.call1((input,)).expect("fnp bool sort parity");
            let base = np_sort.call1((input,)).expect("numpy bool sort parity");
            assert_eq!(
                candidate
                    .getattr("dtype")
                    .expect("candidate dtype")
                    .str()
                    .expect("candidate dtype str")
                    .to_string(),
                base.getattr("dtype")
                    .expect("base dtype")
                    .str()
                    .expect("base dtype str")
                    .to_string(),
                "bool sort {label} dtype parity",
            );
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "bool sort {label} byte parity",
            );
        }

        bench_median_gate_python_unary(
            &mut group,
            "bool_sort_8m_null_then_effect",
            "bool_sort_8m",
            &np_sort,
            &fnp_sort,
            &b_8m,
        );
        bench_median_gate_python_unary(
            &mut group,
            "bool_sort_2m_null_then_effect",
            "bool_sort_2m",
            &np_sort,
            &fnp_sort,
            &b_2m,
        );
    });

    group.finish();
}

fn bench_wide_string_sort_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_wide_string_sort_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_wide_string_sort_median_gate")
            .expect("wide string sort bench module");
        fnp_python(&module).expect("initialize fnp_python wide string sort bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 u9 = rng.integers(97, 123, (1_000_000, 9), dtype=np.uint32).reshape(-1).view('U9')\n\
                 u16 = rng.integers(97, 123, (1_000_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
                 s9 = rng.integers(97, 123, (1_000_000, 9), dtype=np.uint8).reshape(-1).view('S9')\n\
                 s16 = rng.integers(97, 123, (1_000_000, 16), dtype=np.uint8).reshape(-1).view('S16')\n",
            )
            .expect("wide string sort setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("wide string sort setup");
        let u9_input = namespace.get_item("u9").expect("u9 present");
        let u16_input = namespace.get_item("u16").expect("u16 present");
        let s9_input = namespace.get_item("s9").expect("s9 present");
        let s16_input = namespace.get_item("s16").expect("s16 present");
        let fnp_sort = module.getattr("sort").expect("fnp sort");
        let numpy_sort = numpy.getattr("sort").expect("numpy sort");

        for (label, input) in [
            ("U9", &u9_input),
            ("U16", &u16_input),
            ("S9", &s9_input),
            ("S16", &s16_input),
        ] {
            let candidate = fnp_sort
                .call1((input,))
                .expect("fnp wide string sort parity");
            let base = numpy_sort
                .call1((input,))
                .expect("numpy wide string sort parity");
            assert_eq!(
                candidate
                    .getattr("dtype")
                    .expect("candidate dtype")
                    .str()
                    .expect("candidate dtype str")
                    .to_string(),
                base.getattr("dtype")
                    .expect("base dtype")
                    .str()
                    .expect("base dtype str")
                    .to_string(),
                "wide string sort {label} dtype parity",
            );
            assert_eq!(
                candidate
                    .getattr("shape")
                    .expect("candidate shape")
                    .extract::<Vec<usize>>()
                    .expect("candidate shape Vec"),
                base.getattr("shape")
                    .expect("base shape")
                    .extract::<Vec<usize>>()
                    .expect("base shape Vec"),
                "wide string sort {label} shape parity",
            );
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "wide string sort {label} byte parity",
            );
            assert_eq!(
                candidate
                    .getattr("flags")
                    .expect("candidate flags")
                    .getattr("owndata")
                    .expect("candidate owndata")
                    .extract::<bool>()
                    .expect("candidate owndata bool"),
                base.getattr("flags")
                    .expect("base flags")
                    .getattr("owndata")
                    .expect("base owndata")
                    .extract::<bool>()
                    .expect("base owndata bool"),
                "wide string sort {label} ownership parity",
            );
        }

        group.bench_function("wide_string_sort_u16_1m_fnp_profile", |bench| {
            bench.iter(|| {
                black_box(
                    fnp_sort
                        .call1((black_box(&u16_input),))
                        .expect("profile fnp U16 sort"),
                )
            });
        });
        bench_median_gate_python_unary(
            &mut group,
            "wide_string_sort_u9_1m_null_then_effect",
            "wide_string_sort_u9_1m",
            &numpy_sort,
            &fnp_sort,
            &u9_input,
        );
        bench_median_gate_python_unary(
            &mut group,
            "wide_string_sort_u16_1m_null_then_effect",
            "wide_string_sort_u16_1m",
            &numpy_sort,
            &fnp_sort,
            &u16_input,
        );
        bench_median_gate_python_unary(
            &mut group,
            "wide_string_sort_s9_1m_null_then_effect",
            "wide_string_sort_s9_1m",
            &numpy_sort,
            &fnp_sort,
            &s9_input,
        );
        bench_median_gate_python_unary(
            &mut group,
            "wide_string_sort_s16_1m_null_then_effect",
            "wide_string_sort_s16_1m",
            &numpy_sort,
            &fnp_sort,
            &s16_input,
        );
    });

    group.finish();
    if std::env::var_os("FNP_WIDE_STRING_SORT_BENCH_ONLY").is_some() {
        use std::io::Write;
        let _ = std::io::stdout().flush();
        std::process::exit(0);
    }
}

fn bench_accumulate_extremum_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_accumulate_extremum_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_accumulate_extremum_median_gate")
            .expect("accumulate extremum bench module");
        fnp_python(&module).expect("initialize fnp_python accumulate extremum bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 x = rng.standard_normal(8_000_000).astype(np.float64)\n",
            )
            .expect("accumulate extremum setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("accumulate extremum setup");
        let input = namespace.get_item("x").expect("x present");
        let fnp_accumulate = module
            .getattr("maximum")
            .expect("fnp maximum")
            .getattr("accumulate")
            .expect("fnp maximum.accumulate");
        let numpy_accumulate = numpy
            .getattr("maximum")
            .expect("numpy maximum")
            .getattr("accumulate")
            .expect("numpy maximum.accumulate");

        let candidate = fnp_accumulate
            .call1((&input,))
            .expect("fnp maximum.accumulate parity");
        let base = numpy_accumulate
            .call1((&input,))
            .expect("numpy maximum.accumulate parity");
        assert_eq!(
            candidate
                .getattr("dtype")
                .expect("candidate dtype")
                .str()
                .expect("candidate dtype str")
                .to_string(),
            base.getattr("dtype")
                .expect("base dtype")
                .str()
                .expect("base dtype str")
                .to_string(),
            "maximum.accumulate dtype parity",
        );
        assert_eq!(
            candidate
                .getattr("shape")
                .expect("candidate shape")
                .extract::<Vec<usize>>()
                .expect("candidate shape Vec"),
            base.getattr("shape")
                .expect("base shape")
                .extract::<Vec<usize>>()
                .expect("base shape Vec"),
            "maximum.accumulate shape parity",
        );
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "maximum.accumulate byte parity",
        );

        group.bench_function("maximum_accumulate_f64_8m_fnp_profile", |bench| {
            bench.iter(|| {
                black_box(
                    fnp_accumulate
                        .call1((black_box(&input),))
                        .expect("profile fnp maximum.accumulate"),
                )
            });
        });
        bench_median_gate_python_unary(
            &mut group,
            "maximum_accumulate_f64_8m_null_then_effect",
            "maximum_accumulate_f64_8m",
            &numpy_accumulate,
            &fnp_accumulate,
            &input,
        );
    });

    group.finish();
    if std::env::var_os("FNP_ACCUMULATE_EXTREMUM_BENCH_ONLY").is_some() {
        use std::io::Write;
        let _ = std::io::stdout().flush();
        std::process::exit(0);
    }
}

fn bench_int_convolve_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_int_convolve_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_int_convolve_median_gate")
            .expect("int convolve bench module");
        fnp_python(&module).expect("initialize fnp_python int convolve bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 a = rng.integers(-2**31, 2**31, 200_000).astype(np.int64)\n\
                 v = rng.integers(-2**31, 2**31, 256).astype(np.int64)\n",
            )
            .expect("int convolve setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("int convolve setup");
        let a = namespace.get_item("a").expect("a present");
        let v = namespace.get_item("v").expect("v present");
        let fnp_convolve = module.getattr("convolve").expect("fnp convolve");
        let numpy_convolve = numpy.getattr("convolve").expect("numpy convolve");

        let candidate = fnp_convolve
            .call1((&a, &v))
            .expect("fnp int convolve parity");
        let base = numpy_convolve
            .call1((&a, &v))
            .expect("numpy int convolve parity");
        assert_eq!(
            candidate
                .getattr("dtype")
                .expect("candidate dtype")
                .str()
                .expect("candidate dtype str")
                .to_string(),
            base.getattr("dtype")
                .expect("base dtype")
                .str()
                .expect("base dtype str")
                .to_string(),
            "int convolve dtype parity",
        );
        assert_eq!(
            candidate
                .getattr("shape")
                .expect("candidate shape")
                .extract::<Vec<usize>>()
                .expect("candidate shape Vec"),
            base.getattr("shape")
                .expect("base shape")
                .extract::<Vec<usize>>()
                .expect("base shape Vec"),
            "int convolve shape parity",
        );
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "int convolve byte parity",
        );

        group.bench_function("int_convolve_i64_200k_256_fnp_profile", |bench| {
            bench.iter(|| {
                black_box(
                    fnp_convolve
                        .call1((black_box(&a), black_box(&v)))
                        .expect("profile fnp int convolve"),
                )
            });
        });
        bench_median_gate_python_binary(
            &mut group,
            "int_convolve_i64_200k_256_null_then_effect",
            "int_convolve_i64_200k_256",
            &numpy_convolve,
            &fnp_convolve,
            &a,
            &v,
        );
    });

    group.finish();
    if std::env::var_os("FNP_INT_CONVOLVE_BENCH_ONLY").is_some() {
        use std::io::Write;
        let _ = std::io::stdout().flush();
        std::process::exit(0);
    }
}

fn bench_int_matmul_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_int_matmul_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_int_matmul_median_gate")
            .expect("int matmul bench module");
        fnp_python(&module).expect("initialize fnp_python int matmul bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 a64 = rng.integers(-2**31, 2**31, (512, 512)).astype(np.int64)\n\
                 b64 = rng.integers(-2**31, 2**31, (512, 512)).astype(np.int64)\n\
                 a32 = rng.integers(-2**15, 2**15, (512, 512)).astype(np.int32)\n\
                 b32 = rng.integers(-2**15, 2**15, (512, 512)).astype(np.int32)\n\
                 ab64 = rng.integers(-2**31, 2**31, (64, 128, 128)).astype(np.int64)\n\
                 bb64 = rng.integers(-2**31, 2**31, (64, 128, 128)).astype(np.int64)\n\
                 mp64 = rng.integers(-2**31, 2**31, (256, 256)).astype(np.int64)\n\
                 p5 = 5\n",
            )
            .expect("int matmul setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("int matmul setup");
        let a64 = namespace.get_item("a64").expect("a64 present");
        let b64 = namespace.get_item("b64").expect("b64 present");
        let a32 = namespace.get_item("a32").expect("a32 present");
        let b32 = namespace.get_item("b32").expect("b32 present");

        let fnp_matmul = module.getattr("matmul").expect("fnp matmul");
        let np_matmul = numpy.getattr("matmul").expect("numpy matmul");
        for (label, x, y) in [("i64_512", &a64, &b64), ("i32_512", &a32, &b32)] {
            let candidate = fnp_matmul.call1((x, y)).expect("fnp int matmul parity");
            let base = np_matmul.call1((x, y)).expect("numpy int matmul parity");
            assert_eq!(
                candidate
                    .getattr("dtype")
                    .expect("candidate dtype")
                    .str()
                    .expect("candidate dtype str")
                    .to_string(),
                base.getattr("dtype")
                    .expect("base dtype")
                    .str()
                    .expect("base dtype str")
                    .to_string(),
                "int matmul {label} dtype parity",
            );
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "int matmul {label} byte parity",
            );
        }

        bench_median_gate_python_binary(
            &mut group,
            "int_matmul_i64_512_null_then_effect",
            "int_matmul_i64_512",
            &np_matmul,
            &fnp_matmul,
            &a64,
            &b64,
        );
        bench_median_gate_python_binary(
            &mut group,
            "int_matmul_i32_512_null_then_effect",
            "int_matmul_i32_512",
            &np_matmul,
            &fnp_matmul,
            &a32,
            &b32,
        );

        let ab64 = namespace.get_item("ab64").expect("ab64 present");
        let bb64 = namespace.get_item("bb64").expect("bb64 present");
        let mp64 = namespace.get_item("mp64").expect("mp64 present");
        let p5 = namespace.get_item("p5").expect("p5 present");
        let fnp_matrix_power = module
            .getattr("linalg")
            .expect("fnp linalg")
            .getattr("matrix_power")
            .expect("fnp matrix_power");
        let np_matrix_power = numpy
            .getattr("linalg")
            .expect("numpy linalg")
            .getattr("matrix_power")
            .expect("numpy matrix_power");
        for (label, f_c, f_b, x, y) in [
            ("i64_batched", &fnp_matmul, &np_matmul, &ab64, &bb64),
            (
                "i64_matpow5",
                &fnp_matrix_power,
                &np_matrix_power,
                &mp64,
                &p5,
            ),
        ] {
            let candidate = f_c.call1((x, y)).expect("fnp candidate parity");
            let base = f_b.call1((x, y)).expect("numpy base parity");
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "int {label} byte parity",
            );
        }
        bench_median_gate_python_binary(
            &mut group,
            "int_matmul_i64_batched_null_then_effect",
            "int_matmul_i64_batched",
            &np_matmul,
            &fnp_matmul,
            &ab64,
            &bb64,
        );
        bench_median_gate_python_binary(
            &mut group,
            "int_matrix_power_i64_256_p5_null_then_effect",
            "int_matrix_power_i64_256_p5",
            &np_matrix_power,
            &fnp_matrix_power,
            &mp64,
            &p5,
        );
    });

    group.finish();
}

fn bench_f16_matmul_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_f16_matmul_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_f16_matmul_median_gate")
            .expect("f16 matmul bench module");
        fnp_python(&module).expect("initialize fnp_python f16 matmul bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 h_a = rng.standard_normal((512, 512)).astype(np.float16)\n\
                 h_b = rng.standard_normal((512, 512)).astype(np.float16)\n\
                 hb_a = rng.standard_normal((8, 256, 256)).astype(np.float16)\n\
                 hb_b = rng.standard_normal((8, 256, 256)).astype(np.float16)\n\
                 hbc_a = rng.standard_normal((32, 128, 128)).astype(np.float16)\n\
                 hbc_b = rng.standard_normal((128, 96)).astype(np.float16)\n",
            )
            .expect("f16 matmul setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("f16 matmul setup");
        let h_a = namespace.get_item("h_a").expect("h_a present");
        let h_b = namespace.get_item("h_b").expect("h_b present");

        let fnp_matmul = module.getattr("matmul").expect("fnp matmul");
        let np_matmul = numpy.getattr("matmul").expect("numpy matmul");
        let candidate = fnp_matmul
            .call1((&h_a, &h_b))
            .expect("fnp f16 matmul parity");
        let base = np_matmul
            .call1((&h_a, &h_b))
            .expect("numpy f16 matmul parity");
        assert_eq!(
            candidate
                .getattr("dtype")
                .expect("candidate dtype")
                .str()
                .expect("candidate dtype str")
                .to_string(),
            base.getattr("dtype")
                .expect("base dtype")
                .str()
                .expect("base dtype str")
                .to_string(),
            "f16 matmul dtype parity",
        );
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 matmul byte parity",
        );

        bench_median_gate_python_binary(
            &mut group,
            "f16_matmul_512_null_then_effect",
            "f16_matmul_512",
            &np_matmul,
            &fnp_matmul,
            &h_a,
            &h_b,
        );

        let hb_a = namespace.get_item("hb_a").expect("hb_a present");
        let hb_b = namespace.get_item("hb_b").expect("hb_b present");
        let candidate = fnp_matmul
            .call1((&hb_a, &hb_b))
            .expect("fnp f16 batched matmul parity");
        let base = np_matmul
            .call1((&hb_a, &hb_b))
            .expect("numpy f16 batched matmul parity");
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 batched matmul byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_matmul_batched_8x256_null_then_effect",
            "f16_matmul_batched_8x256",
            &np_matmul,
            &fnp_matmul,
            &hb_a,
            &hb_b,
        );

        let hbc_a = namespace.get_item("hbc_a").expect("hbc_a present");
        let hbc_b = namespace.get_item("hbc_b").expect("hbc_b present");
        let candidate = fnp_matmul
            .call1((&hbc_a, &hbc_b))
            .expect("fnp f16 broadcast matmul parity");
        let base = np_matmul
            .call1((&hbc_a, &hbc_b))
            .expect("numpy f16 broadcast matmul parity");
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 broadcast matmul byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_matmul_broadcast_32x128_null_then_effect",
            "f16_matmul_broadcast_32x128",
            &np_matmul,
            &fnp_matmul,
            &hbc_a,
            &hbc_b,
        );
    });

    group.finish();
}

fn bench_f16_unique_median_gate(c: &mut Criterion) {
    // f16 unique at 8M: presence-table walk vs numpy's ~600ms-class sort.
    let mut group = c.benchmark_group("python_f16_unique_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module =
            PyModule::new(py, "fnp_python_f16_unique_median_gate").expect("unique bench module");
        fnp_python(&module).expect("initialize fnp_python unique bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260716)\n\
                 uq16 = (rng.standard_normal(8_000_000) * 2).astype(np.float16)\n",
            )
            .expect("unique setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("unique setup");
        let uq16 = namespace.get_item("uq16").expect("uq16 present");
        let fnp_unique = module.getattr("unique").expect("fnp unique");
        let np_unique = numpy.getattr("unique").expect("numpy unique");
        let candidate = fnp_unique.call1((&uq16,)).expect("fnp unique parity");
        let base = np_unique.call1((&uq16,)).expect("numpy unique parity");
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 unique byte parity",
        );
        bench_median_gate_python_unary(
            &mut group,
            "f16_unique_8m_null_then_effect",
            "f16_unique_8m",
            &np_unique,
            &fnp_unique,
            &uq16,
        );

        // f16 isin at 8M/1k: presence-bitmap membership vs numpy's ~1.2s sort path.
        py.run(
            std::ffi::CString::new("iq16 = (rng.standard_normal(1000) * 2).astype(np.float16)\n")
                .expect("isin setup CString")
                .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("isin setup");
        let iq16 = namespace.get_item("iq16").expect("iq16 present");
        let fnp_isin = module.getattr("isin").expect("fnp isin");
        let np_isin = numpy.getattr("isin").expect("numpy isin");
        let candidate_i = fnp_isin.call1((&uq16, &iq16)).expect("fnp isin parity");
        let base_i = np_isin.call1((&uq16, &iq16)).expect("numpy isin parity");
        assert_eq!(
            candidate_i
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_i
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 isin byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_isin_8m_null_then_effect",
            "f16_isin_8m",
            &np_isin,
            &fnp_isin,
            &uq16,
            &iq16,
        );
    });

    group.finish();
}

fn bench_f16_around_median_gate(c: &mut Criterion) {
    // f16 round(a, 2) at 8M: the per-step-narrow chain kernel vs numpy's serial
    // half multiply->rint->divide loops (~90ms class on hz1).
    let mut group = c.benchmark_group("python_f16_around_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module =
            PyModule::new(py, "fnp_python_f16_around_median_gate").expect("around bench module");
        fnp_python(&module).expect("initialize fnp_python around bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260718)\n\
                 ra16 = (rng.standard_normal(8_000_000) * 2).astype(np.float16)\n\
                 dec2 = 2\n",
            )
            .expect("around setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("around setup");
        let ra16 = namespace.get_item("ra16").expect("ra16 present");
        let dec2 = namespace.get_item("dec2").expect("dec2 present");
        let fnp_round = module.getattr("round").expect("fnp round");
        let np_round = numpy.getattr("round").expect("numpy round");
        let candidate = fnp_round.call1((&ra16, &dec2)).expect("fnp round parity");
        let base = np_round.call1((&ra16, &dec2)).expect("numpy round parity");
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 around byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_around_8m_null_then_effect",
            "f16_around_8m",
            &np_round,
            &fnp_round,
            &ra16,
            &dec2,
        );
    });

    group.finish();
}

fn bench_isclose_median_gate(c: &mut Criterion) {
    // f64 array-array isclose at 8M: the parallelized zero-copy predicate vs
    // numpy's temp-heavy ufunc chain (~180ms class on hz1).
    let mut group = c.benchmark_group("python_isclose_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module =
            PyModule::new(py, "fnp_python_isclose_median_gate").expect("isclose bench module");
        fnp_python(&module).expect("initialize fnp_python isclose bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260716)\n\
                 ic_a = rng.standard_normal(8_000_000)\n\
                 ic_b = ic_a + rng.standard_normal(8_000_000) * 1e-7\n",
            )
            .expect("isclose setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("isclose setup");
        let ic_a = namespace.get_item("ic_a").expect("ic_a present");
        let ic_b = namespace.get_item("ic_b").expect("ic_b present");
        let fnp_isclose = module.getattr("isclose").expect("fnp isclose");
        let np_isclose = numpy.getattr("isclose").expect("numpy isclose");
        let candidate = fnp_isclose
            .call1((&ic_a, &ic_b))
            .expect("fnp isclose parity");
        let base = np_isclose
            .call1((&ic_a, &ic_b))
            .expect("numpy isclose parity");
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "isclose byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "isclose_f64_8m_null_then_effect",
            "isclose_f64_8m",
            &np_isclose,
            &fnp_isclose,
            &ic_a,
            &ic_b,
        );
    });

    group.finish();
}

fn bench_multidot_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_multidot_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module =
            PyModule::new(py, "fnp_python_multidot_median_gate").expect("multidot bench module");
        fnp_python(&module).expect("initialize fnp_python multidot bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 md_args = [rng.standard_normal((512, 512)) for _ in range(3)]\n",
            )
            .expect("multidot setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("multidot setup");
        let md_args = namespace.get_item("md_args").expect("md_args present");

        let fnp_multidot = module
            .getattr("linalg")
            .expect("fnp linalg")
            .getattr("multi_dot")
            .expect("fnp multi_dot");
        let np_multidot = numpy
            .getattr("linalg")
            .expect("numpy linalg")
            .getattr("multi_dot")
            .expect("numpy multi_dot");
        let candidate = fnp_multidot
            .call1((&md_args,))
            .expect("fnp multi_dot parity");
        let base = np_multidot
            .call1((&md_args,))
            .expect("numpy multi_dot parity");
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "multi_dot byte parity",
        );

        bench_median_gate_python_unary(
            &mut group,
            "multidot_3x512_null_then_effect",
            "multidot_3x512",
            &np_multidot,
            &fnp_multidot,
            &md_args,
        );

        // f16 3-chain: numpy's pairs are the naive ~245x f16 loops; fnp
        // routes both pairs through the shipped byte-matched f16 matmul
        // kernel per the replicated _multi_dot_three order rule.
        py.run(
            std::ffi::CString::new(
                "md16_args = [(rng.standard_normal((256, 256)) * 0.3).astype(np.float16) for _ in range(3)]\n",
            )
            .expect("multidot f16 setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("multidot f16 setup");
        let md16_args = namespace.get_item("md16_args").expect("md16_args present");
        let candidate16 = fnp_multidot
            .call1((&md16_args,))
            .expect("fnp f16 multi_dot parity");
        let base16 = np_multidot
            .call1((&md16_args,))
            .expect("numpy f16 multi_dot parity");
        assert_eq!(
            candidate16
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base16
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 multi_dot byte parity",
        );
        bench_median_gate_python_unary(
            &mut group,
            "multidot_f16_3x256_null_then_effect",
            "multidot_f16_3x256",
            &np_multidot,
            &fnp_multidot,
            &md16_args,
        );
    });

    group.finish();
}

fn bench_f16_einsum_median_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_f16_einsum_median_gate");
    group.sample_size(MEDIAN_GATE_FINAL_BATCHES);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_f16_einsum_median_gate")
            .expect("f16 einsum bench module");
        fnp_python(&module).expect("initialize fnp_python f16 einsum bench module");
        let _numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        namespace
            .set_item("fnp_mod", &module)
            .expect("expose fnp module");
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260711)\n\
                 es_a = (rng.standard_normal((512, 512)) * 0.3).astype(np.float16)\n\
                 es_b = (rng.standard_normal((512, 512)) * 0.3).astype(np.float16)\n\
                 fnp_es = lambda a, b: fnp_mod.einsum('ij,jk->ik', a, b)\n\
                 np_es = lambda a, b: np.einsum('ij,jk->ik', a, b)\n\
                 fnp_es_t = lambda a, b: fnp_mod.einsum('ij,lj->il', a, b)\n\
                 np_es_t = lambda a, b: np.einsum('ij,lj->il', a, b)\n\
                 fnp_es_g = lambda a, b: fnp_mod.einsum('ji,jl->il', a, b)\n\
                 np_es_g = lambda a, b: np.einsum('ji,jl->il', a, b)\n\
                 fnp_es_ts = lambda a, b: fnp_mod.einsum('ij,lj->li', a, b)\n\
                 np_es_ts = lambda a, b: np.einsum('ij,lj->li', a, b)\n\
                 fnp_es_gs = lambda a, b: fnp_mod.einsum('ji,jl->li', a, b)\n\
                 np_es_gs = lambda a, b: np.einsum('ji,jl->li', a, b)\n\
                 dot_a = (rng.standard_normal(8_388_608) * 0.3).astype(np.float16)\n\
                 dot_b = (rng.standard_normal(8_388_608) * 0.3).astype(np.float16)\n\
                 fnp_es_d = lambda a, b: fnp_mod.einsum('j,j->', a, b)\n\
                 np_es_d = lambda a, b: np.einsum('j,j->', a, b)\n\
                 fc_a = (rng.standard_normal((2896, 2896)) * 0.3).astype(np.float16)\n\
                 fc_b = (rng.standard_normal((2896, 2896)) * 0.3).astype(np.float16)\n\
                 fnp_es_fc = lambda a, b: fnp_mod.einsum('ij,ij->', a, b)\n\
                 np_es_fc = lambda a, b: np.einsum('ij,ij->', a, b)\n\
                 fnp_es_ew = lambda a, b: fnp_mod.einsum('j,j->j', a, b)\n\
                 np_es_ew = lambda a, b: np.einsum('j,j->j', a, b)\n\
                 ew64_a = rng.standard_normal(8_388_608)\n\
                 ew64_b = rng.standard_normal(8_388_608)\n\
                 bc64_full = rng.standard_normal((2896, 2896))\n\
                 bc64_vec = rng.standard_normal(2896)\n\
                 red16 = (rng.standard_normal((2896, 2896)) * 0.3).astype(np.float16)\n\
                 red64 = rng.standard_normal((2896, 2896))\n\
                 red32 = rng.standard_normal((2896, 2896)).astype(np.float32)\n\
                 ch_a = (rng.standard_normal((512, 512)) * 0.3).astype(np.float16)\n\
                 ch_b = (rng.standard_normal((512, 512)) * 0.3).astype(np.float16)\n\
                 ch_c = (rng.standard_normal((512, 512)) * 0.3).astype(np.float16)\n\
                 fnp_es_ch = lambda a, b: fnp_mod.einsum('ij,jk,kl->il', a, b, ch_c, optimize=True)\n\
                 np_es_ch = lambda a, b: np.einsum('ij,jk,kl->il', a, b, ch_c, optimize=True)\n\
                 fnp_es_rj = lambda a: fnp_mod.einsum('ij->j', a)\n\
                 np_es_rj = lambda a: np.einsum('ij->j', a)\n\
                 fnp_es_ri = lambda a: fnp_mod.einsum('ij->i', a)\n\
                 np_es_ri = lambda a: np.einsum('ij->i', a)\n\
                 fnp_es_bc = lambda a, b: fnp_mod.einsum('ij,j->ij', a, b)\n\
                 np_es_bc = lambda a, b: np.einsum('ij,j->ij', a, b)\n\
                 bat_a = (rng.standard_normal((8, 256, 256)) * 0.3).astype(np.float16)\n\
                 bat_b = (rng.standard_normal((8, 256, 256)) * 0.3).astype(np.float16)\n\
                 fnp_es_b = lambda a, b: fnp_mod.einsum('bij,bjk->bik', a, b)\n\
                 np_es_b = lambda a, b: np.einsum('bij,bjk->bik', a, b)\n\
                 fnp_es_bt = lambda a, b: fnp_mod.einsum('bij,blj->bil', a, b)\n\
                 np_es_bt = lambda a, b: np.einsum('bij,blj->bil', a, b)\n\
                 fnp_es_bg = lambda a, b: fnp_mod.einsum('bji,bjl->bil', a, b)\n\
                 np_es_bg = lambda a, b: np.einsum('bji,bjl->bil', a, b)\n\
                 fnp_es_bts = lambda a, b: fnp_mod.einsum('bij,blj->bli', a, b)\n\
                 np_es_bts = lambda a, b: np.einsum('bij,blj->bli', a, b)\n\
                 fnp_es_bgs = lambda a, b: fnp_mod.einsum('bji,bjl->bli', a, b)\n\
                 np_es_bgs = lambda a, b: np.einsum('bji,bjl->bli', a, b)\n",
            )
            .expect("f16 einsum setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("f16 einsum setup");
        let es_a = namespace.get_item("es_a").expect("es_a present");
        let es_b = namespace.get_item("es_b").expect("es_b present");
        let fnp_es = namespace.get_item("fnp_es").expect("fnp_es present");
        let np_es = namespace.get_item("np_es").expect("np_es present");

        let candidate = fnp_es.call1((&es_a, &es_b)).expect("fnp f16 einsum parity");
        let base = np_es
            .call1((&es_a, &es_b))
            .expect("numpy f16 einsum parity");
        assert_eq!(
            candidate
                .getattr("dtype")
                .expect("candidate dtype")
                .str()
                .expect("candidate dtype str")
                .to_string(),
            base.getattr("dtype")
                .expect("base dtype")
                .str()
                .expect("base dtype str")
                .to_string(),
            "f16 einsum dtype parity",
        );
        assert_eq!(
            candidate
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base.call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum byte parity",
        );

        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_matmul_512_null_then_effect",
            "f16_einsum_matmul_512",
            &np_es,
            &fnp_es,
            &es_a,
            &es_b,
        );

        // Transposed spec ('ij,lj->il', the a@b.T idiom): a different numpy
        // contract class (wide-accumulate-once blocked-4) with its own kernel.
        let fnp_es_t = namespace.get_item("fnp_es_t").expect("fnp_es_t present");
        let np_es_t = namespace.get_item("np_es_t").expect("np_es_t present");
        let candidate_t = fnp_es_t
            .call1((&es_a, &es_b))
            .expect("fnp f16 einsum transposed parity");
        let base_t = np_es_t
            .call1((&es_a, &es_b))
            .expect("numpy f16 einsum transposed parity");
        assert_eq!(
            candidate_t
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_t
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum transposed byte parity",
        );

        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_transposed_512_null_then_effect",
            "f16_einsum_transposed_512",
            &np_es_t,
            &fnp_es_t,
            &es_a,
            &es_b,
        );

        // Gram spec ('ji,jl->il', the a.T@b idiom): the third numpy contract
        // class (per-step-narrow muladd rows, stride0_contig_outcontig) with
        // its own kernel. Same 512^2 operands (k = leading axis).
        let fnp_es_g = namespace.get_item("fnp_es_g").expect("fnp_es_g present");
        let np_es_g = namespace.get_item("np_es_g").expect("np_es_g present");
        let candidate_g = fnp_es_g
            .call1((&es_a, &es_b))
            .expect("fnp f16 einsum gram parity");
        let base_g = np_es_g
            .call1((&es_a, &es_b))
            .expect("numpy f16 einsum gram parity");
        assert_eq!(
            candidate_g
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_g
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum gram byte parity",
        );

        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_gram_512_null_then_effect",
            "f16_einsum_gram_512",
            &np_es_g,
            &fnp_es_g,
            &es_a,
            &es_b,
        );

        // Output-transposed variants: operand-swap arms of the transposed and
        // gram kernels ('ij,lj->li' / 'ji,jl->li'). Rows prove the swapped
        // dispatch engages the native route (effect >> 1, not numpy ~1.0x).
        for (bench_name, row, fnp_key, np_key) in [
            (
                "f16_einsum_transposed_swapped_512_null_then_effect",
                "f16_einsum_transposed_swapped_512",
                "fnp_es_ts",
                "np_es_ts",
            ),
            (
                "f16_einsum_gram_swapped_512_null_then_effect",
                "f16_einsum_gram_swapped_512",
                "fnp_es_gs",
                "np_es_gs",
            ),
        ] {
            let fnp_fn = namespace.get_item(fnp_key).expect("fnp swapped fn");
            let np_fn = namespace.get_item(np_key).expect("np swapped fn");
            let candidate = fnp_fn
                .call1((&es_a, &es_b))
                .expect("fnp swapped einsum parity");
            let base = np_fn
                .call1((&es_a, &es_b))
                .expect("numpy swapped einsum parity");
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "f16 einsum swapped-output byte parity ({row})",
            );
            bench_median_gate_python_binary(
                &mut group, bench_name, row, &np_fn, &fnp_fn, &es_a, &es_b,
            );
        }

        // 1-D dot ('j,j->') at 8M: per-8192-buffer trees in parallel, serial
        // f16 fold. Scalar output - parity assert via float16 byte equality.
        let dot_a = namespace.get_item("dot_a").expect("dot_a present");
        let dot_b = namespace.get_item("dot_b").expect("dot_b present");
        let fnp_es_d = namespace.get_item("fnp_es_d").expect("fnp_es_d present");
        let np_es_d = namespace.get_item("np_es_d").expect("np_es_d present");
        let candidate_d = fnp_es_d
            .call1((&dot_a, &dot_b))
            .expect("fnp f16 einsum dot parity");
        let base_d = np_es_d
            .call1((&dot_a, &dot_b))
            .expect("numpy f16 einsum dot parity");
        assert_eq!(
            candidate_d
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_d
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum 1-D dot byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_dot1d_8m_null_then_effect",
            "f16_einsum_dot1d_8m",
            &np_es_d,
            &fnp_es_d,
            &dot_a,
            &dot_b,
        );

        // 2-D full contraction ('ij,ij->') at 2896^2 ~ 8.4M: the coalesced
        // chunk-fold route through the generalized full-contraction parser.
        let fc_a = namespace.get_item("fc_a").expect("fc_a present");
        let fc_b = namespace.get_item("fc_b").expect("fc_b present");
        let fnp_es_fc = namespace.get_item("fnp_es_fc").expect("fnp_es_fc present");
        let np_es_fc = namespace.get_item("np_es_fc").expect("np_es_fc present");
        let candidate_fc = fnp_es_fc
            .call1((&fc_a, &fc_b))
            .expect("fnp f16 einsum full-contraction parity");
        let base_fc = np_es_fc
            .call1((&fc_a, &fc_b))
            .expect("numpy f16 einsum full-contraction parity");
        assert_eq!(
            candidate_fc
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_fc
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum full-contraction byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_fullc_2d_8m_null_then_effect",
            "f16_einsum_fullc_2d_8m",
            &np_es_fc,
            &fnp_es_fc,
            &fc_a,
            &fc_b,
        );

        // Elementwise product ('j,j->j') at 8M: zero-seeded parallel flat map.
        let fnp_es_ew = namespace.get_item("fnp_es_ew").expect("fnp_es_ew present");
        let np_es_ew = namespace.get_item("np_es_ew").expect("np_es_ew present");
        let candidate_ew = fnp_es_ew
            .call1((&dot_a, &dot_b))
            .expect("fnp f16 einsum elementwise parity");
        let base_ew = np_es_ew
            .call1((&dot_a, &dot_b))
            .expect("numpy f16 einsum elementwise parity");
        assert_eq!(
            candidate_ew
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_ew
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum elementwise byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_elemwise_8m_null_then_effect",
            "f16_einsum_elemwise_8m",
            &np_es_ew,
            &fnp_es_ew,
            &dot_a,
            &dot_b,
        );

        // f64 elementwise ('j,j->j') at 8M: the f64/f32 zero-seeded kernel.
        let ew64_a = namespace.get_item("ew64_a").expect("ew64_a present");
        let ew64_b = namespace.get_item("ew64_b").expect("ew64_b present");
        let candidate_e64 = fnp_es_ew
            .call1((&ew64_a, &ew64_b))
            .expect("fnp f64 einsum elementwise parity");
        let base_e64 = np_es_ew
            .call1((&ew64_a, &ew64_b))
            .expect("numpy f64 einsum elementwise parity");
        assert_eq!(
            candidate_e64
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_e64
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f64 einsum elementwise byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f64_einsum_elemwise_8m_null_then_effect",
            "f64_einsum_elemwise_8m",
            &np_es_ew,
            &fnp_es_ew,
            &ew64_a,
            &ew64_b,
        );

        // f64 broadcast form ('ij,j->ij') at 2896^2: the broadcast kernel.
        let bc64_full = namespace.get_item("bc64_full").expect("bc64_full present");
        let bc64_vec = namespace.get_item("bc64_vec").expect("bc64_vec present");
        let fnp_es_bc = namespace.get_item("fnp_es_bc").expect("fnp_es_bc present");
        let np_es_bc = namespace.get_item("np_es_bc").expect("np_es_bc present");
        let candidate_bc = fnp_es_bc
            .call1((&bc64_full, &bc64_vec))
            .expect("fnp f64 einsum broadcast parity");
        let base_bc = np_es_bc
            .call1((&bc64_full, &bc64_vec))
            .expect("numpy f64 einsum broadcast parity");
        assert_eq!(
            candidate_bc
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_bc
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f64 einsum broadcast byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f64_einsum_bcast_8m_null_then_effect",
            "f64_einsum_bcast_8m",
            &np_es_bc,
            &fnp_es_bc,
            &bc64_full,
            &bc64_vec,
        );

        // f16 3-op chain 512^3 with optimize=True: plan + shipped matmul
        // kernel per pair (c operand captured in the lambda closures).
        let ch_a = namespace.get_item("ch_a").expect("ch_a present");
        let ch_b = namespace.get_item("ch_b").expect("ch_b present");
        let fnp_es_ch = namespace.get_item("fnp_es_ch").expect("fnp_es_ch present");
        let np_es_ch = namespace.get_item("np_es_ch").expect("np_es_ch present");
        let candidate_ch = fnp_es_ch
            .call1((&ch_a, &ch_b))
            .expect("fnp f16 chain parity");
        let base_ch = np_es_ch
            .call1((&ch_a, &ch_b))
            .expect("numpy f16 chain parity");
        assert_eq!(
            candidate_ch
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_ch
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum chain3 byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_chain3_512_null_then_effect",
            "f16_einsum_chain3_512",
            &np_es_ch,
            &fnp_es_ch,
            &ch_a,
            &ch_b,
        );

        // f16 reduction specs at 2896^2: col-sum (the 27.9ms strided rank)
        // and row-sum.
        let red16 = namespace.get_item("red16").expect("red16 present");
        let red64 = namespace.get_item("red64").expect("red64 present");
        let red32 = namespace.get_item("red32").expect("red32 present");
        for (bench_name, row, fnp_key, np_key, input) in [
            (
                "f16_einsum_colsum_8m_null_then_effect",
                "f16_einsum_colsum_8m",
                "fnp_es_rj",
                "np_es_rj",
                &red16,
            ),
            (
                "f16_einsum_rowsum_8m_null_then_effect",
                "f16_einsum_rowsum_8m",
                "fnp_es_ri",
                "np_es_ri",
                &red16,
            ),
            (
                "f64_einsum_colsum_8m_null_then_effect",
                "f64_einsum_colsum_8m",
                "fnp_es_rj",
                "np_es_rj",
                &red64,
            ),
            (
                "f64_einsum_rowsum_8m_null_then_effect",
                "f64_einsum_rowsum_8m",
                "fnp_es_ri",
                "np_es_ri",
                &red64,
            ),
            (
                "f32_einsum_colsum_8m_null_then_effect",
                "f32_einsum_colsum_8m",
                "fnp_es_rj",
                "np_es_rj",
                &red32,
            ),
            (
                "f32_einsum_rowsum_8m_null_then_effect",
                "f32_einsum_rowsum_8m",
                "fnp_es_ri",
                "np_es_ri",
                &red32,
            ),
        ] {
            let fnp_fn = namespace.get_item(fnp_key).expect("fnp reduce fn");
            let np_fn = namespace.get_item(np_key).expect("np reduce fn");
            let candidate = fnp_fn.call1((input,)).expect("fnp reduce parity");
            let base = np_fn.call1((input,)).expect("numpy reduce parity");
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "f16 einsum reduction byte parity ({row})",
            );
            bench_median_gate_python_unary(&mut group, bench_name, row, &np_fn, &fnp_fn, input);
        }

        // Batched matmul spec ('bij,bjk->bik') at (8,256,256)@(8,256,256):
        // the plain per-step chain per batch, parallel across batches.
        let bat_a = namespace.get_item("bat_a").expect("bat_a present");
        let bat_b = namespace.get_item("bat_b").expect("bat_b present");
        let fnp_es_b = namespace.get_item("fnp_es_b").expect("fnp_es_b present");
        let np_es_b = namespace.get_item("np_es_b").expect("np_es_b present");
        let candidate_b = fnp_es_b
            .call1((&bat_a, &bat_b))
            .expect("fnp f16 einsum batched parity");
        let base_b = np_es_b
            .call1((&bat_a, &bat_b))
            .expect("numpy f16 einsum batched parity");
        assert_eq!(
            candidate_b
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_b
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum batched byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_batched_8x256_null_then_effect",
            "f16_einsum_batched_8x256",
            &np_es_b,
            &fnp_es_b,
            &bat_a,
            &bat_b,
        );

        // Batched transposed spec ('bij,blj->bil'): buffered chunk-fold wide
        // trees per element, parallel across batches + row blocks.
        let fnp_es_bt = namespace.get_item("fnp_es_bt").expect("fnp_es_bt present");
        let np_es_bt = namespace.get_item("np_es_bt").expect("np_es_bt present");
        let candidate_bt = fnp_es_bt
            .call1((&bat_a, &bat_b))
            .expect("fnp f16 einsum batched-t parity");
        let base_bt = np_es_bt
            .call1((&bat_a, &bat_b))
            .expect("numpy f16 einsum batched-t parity");
        assert_eq!(
            candidate_bt
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_bt
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum batched transposed byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_batched_t_8x256_null_then_effect",
            "f16_einsum_batched_t_8x256",
            &np_es_bt,
            &fnp_es_bt,
            &bat_a,
            &bat_b,
        );

        // Batched gram spec ('bji,bjl->bil'): per-step muladd rows per batch
        // (chunk-immune per-step class). Same (8,256,256) operands.
        let fnp_es_bg = namespace.get_item("fnp_es_bg").expect("fnp_es_bg present");
        let np_es_bg = namespace.get_item("np_es_bg").expect("np_es_bg present");
        let candidate_bg = fnp_es_bg
            .call1((&bat_a, &bat_b))
            .expect("fnp f16 einsum batched-gram parity");
        let base_bg = np_es_bg
            .call1((&bat_a, &bat_b))
            .expect("numpy f16 einsum batched-gram parity");
        assert_eq!(
            candidate_bg
                .call_method0("tobytes")
                .expect("candidate bytes")
                .extract::<Vec<u8>>()
                .expect("candidate byte Vec"),
            base_bg
                .call_method0("tobytes")
                .expect("base bytes")
                .extract::<Vec<u8>>()
                .expect("base byte Vec"),
            "f16 einsum batched gram byte parity",
        );
        bench_median_gate_python_binary(
            &mut group,
            "f16_einsum_batched_g_8x256_null_then_effect",
            "f16_einsum_batched_g_8x256",
            &np_es_bg,
            &fnp_es_bg,
            &bat_a,
            &bat_b,
        );

        // Output-swapped batched forms: operand-swap arms of the batched
        // transposed/gram kernels. Rows prove the swapped dispatch engages.
        for (bench_name, row, fnp_key, np_key) in [
            (
                "f16_einsum_batched_ts_8x256_null_then_effect",
                "f16_einsum_batched_ts_8x256",
                "fnp_es_bts",
                "np_es_bts",
            ),
            (
                "f16_einsum_batched_gs_8x256_null_then_effect",
                "f16_einsum_batched_gs_8x256",
                "fnp_es_bgs",
                "np_es_bgs",
            ),
        ] {
            let fnp_fn = namespace.get_item(fnp_key).expect("fnp batched-swap fn");
            let np_fn = namespace.get_item(np_key).expect("np batched-swap fn");
            let candidate = fnp_fn
                .call1((&bat_a, &bat_b))
                .expect("fnp batched-swap parity");
            let base = np_fn
                .call1((&bat_a, &bat_b))
                .expect("numpy batched-swap parity");
            assert_eq!(
                candidate
                    .call_method0("tobytes")
                    .expect("candidate bytes")
                    .extract::<Vec<u8>>()
                    .expect("candidate byte Vec"),
                base.call_method0("tobytes")
                    .expect("base bytes")
                    .extract::<Vec<u8>>()
                    .expect("base byte Vec"),
                "f16 einsum batched swapped byte parity ({row})",
            );
            bench_median_gate_python_binary(
                &mut group, bench_name, row, &np_fn, &fnp_fn, &bat_a, &bat_b,
            );
        }
    });

    group.finish();
}

fn bench_wide_string_substrate_v2(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_wide_string_substrate_v2");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_wide_string_v2").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(611)\n\
                 u_a = rng.integers(97, 123, (1_000_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
                 u_fresh = rng.integers(97, 123, (500_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
                 u_b = np.concatenate([u_a[:500_000], u_fresh])\n\
                 u_union_b = rng.integers(97, 123, (1_000_000, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
                 s_a = rng.integers(0, 256, (1_000_000, 16), dtype=np.uint8).view('S16').reshape(-1)\n\
                 s_fresh = rng.integers(0, 256, (500_000, 16), dtype=np.uint8).view('S16').reshape(-1)\n\
                 s_b = np.concatenate([s_a[:500_000], s_fresh])\n",
            )
            .expect("wide string setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("wide string setup");
        let u_a = namespace.get_item("u_a").expect("u_a present");
        let u_b = namespace.get_item("u_b").expect("u_b present");
        let u_union_b = namespace.get_item("u_union_b").expect("u_union_b present");
        let s_a = namespace.get_item("s_a").expect("s_a present");
        let s_b = namespace.get_item("s_b").expect("s_b present");
        let array_equal = numpy.getattr("array_equal").expect("numpy.array_equal");

        for (lhs, rhs) in [(&u_a, &u_b), (&s_a, &s_b)] {
            for op in ["unique", "union1d", "intersect1d", "setxor1d"] {
                let candidate_fn = module.getattr(op).expect("fnp op");
                let orig_fn = numpy.getattr(op).expect("numpy op");
                let candidate = if op == "unique" {
                    candidate_fn.call1((lhs,)).expect("fnp parity call")
                } else {
                    candidate_fn.call1((lhs, rhs)).expect("fnp parity call")
                };
                let orig = if op == "unique" {
                    orig_fn.call1((lhs,)).expect("numpy parity call")
                } else {
                    orig_fn.call1((lhs, rhs)).expect("numpy parity call")
                };
                assert!(
                    array_equal
                        .call1((&candidate, &orig))
                        .expect("array_equal")
                        .extract::<bool>()
                        .expect("array_equal bool"),
                    "wide string {op} parity",
                );
            }
        }
        let fnp_union_parity = module
            .getattr("union1d")
            .expect("fnp union1d parity function")
            .call1((&u_a, &u_union_b))
            .expect("fnp union1d parity call");
        let numpy_union_parity = numpy
            .getattr("union1d")
            .expect("numpy union1d parity function")
            .call1((&u_a, &u_union_b))
            .expect("numpy union1d parity call");
        assert!(
            array_equal
                .call1((&fnp_union_parity, &numpy_union_parity))
                .expect("union array_equal")
                .extract::<bool>()
                .expect("union array_equal bool"),
            "wide string disjoint union parity",
        );

        let fnp_unique = module.getattr("unique").expect("fnp unique");
        let np_unique = numpy.getattr("unique").expect("numpy unique");
        let fnp_union = module.getattr("union1d").expect("fnp union1d");
        let np_union = numpy.getattr("union1d").expect("numpy union1d");
        let fnp_intersect = module.getattr("intersect1d").expect("fnp intersect1d");
        let np_intersect = numpy.getattr("intersect1d").expect("numpy intersect1d");
        let fnp_setxor = module.getattr("setxor1d").expect("fnp setxor1d");
        let np_setxor = numpy.getattr("setxor1d").expect("numpy setxor1d");

        bench_substrate_v2_python_unary_pair(
            &mut group,
            "u16_unique_1m_paired",
            "u16_unique_1m",
            &fnp_unique,
            &np_unique,
            &u_a,
        );
        bench_substrate_v2_python_binary_pair(
            &mut group,
            "u16_union_disjoint_1m_paired",
            "u16_union_disjoint_1m",
            &fnp_union,
            &np_union,
            &u_a,
            &u_union_b,
        );
        bench_substrate_v2_python_binary_pair(
            &mut group,
            "u16_setxor_1m_paired",
            "u16_setxor_1m",
            &fnp_setxor,
            &np_setxor,
            &u_a,
            &u_b,
        );
        bench_substrate_v2_python_unary_pair(
            &mut group,
            "s16_unique_1m_paired",
            "s16_unique_1m",
            &fnp_unique,
            &np_unique,
            &s_a,
        );
        bench_substrate_v2_python_binary_pair(
            &mut group,
            "s16_intersect_1m_paired",
            "s16_intersect_1m",
            &fnp_intersect,
            &np_intersect,
            &s_a,
            &s_b,
        );
        bench_substrate_v2_python_binary_pair(
            &mut group,
            "s16_setxor_1m_paired",
            "s16_setxor_1m",
            &fnp_setxor,
            &np_setxor,
            &s_a,
            &s_b,
        );
    });

    group.finish();
}

fn bench_ledger_integrity_rejects(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_ledger_integrity_rejects");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(6));
    group.warm_up_time(Duration::from_secs(2));

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_ledger_audit").expect("bench module");
        fnp_python(&module).expect("initialize fnp_python bench module");
        let numpy = py.import("numpy").expect("numpy oracle");

        {
            let namespace = PyDict::new(py);
            py.run(
                std::ffi::CString::new(
                    "import numpy as np\n\
                     rng = np.random.default_rng(0)\n\
                     median_input = rng.standard_normal(16_000_000).astype(np.float64)\n",
                )
                .expect("median setup CString")
                .as_c_str(),
                Some(&namespace),
                Some(&namespace),
            )
            .expect("median setup");
            let input = namespace
                .get_item("median_input")
                .expect("median input present");
            let raw: Vec<u8> = input
                .call_method0("tobytes")
                .expect("median bytes")
                .extract()
                .expect("extract median bytes");
            let data: Vec<f64> = raw
                .as_chunks::<8>()
                .0
                .iter()
                .map(|chunk| f64::from_ne_bytes(*chunk))
                .collect();
            assert_eq!(data.len(), 16_000_000);
            let numpy_median = numpy.getattr("median").expect("numpy.median");
            let candidate = ledger_radix_median_f64(&data);
            let orig = ledger_orig_median_reference(&numpy_median, &input)
                .expect("NumPy median parity reference");
            assert_eq!(candidate.to_bits(), orig.to_bits(), "radix median parity");

            let candidate_samples = RefCell::new(Vec::new());
            let orig_samples = RefCell::new(Vec::new());
            let order = Cell::new(0u64);
            group.bench_function("radix_median_f64_normal_16m_paired", |bench| {
                bench.iter_custom(|iterations| {
                    let mut candidate_total = Duration::ZERO;
                    let mut orig_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let orig_first = order.get() & 1 == 1;
                        order.set(order.get().wrapping_add(1));
                        if orig_first {
                            let start = Instant::now();
                            black_box(
                                ledger_orig_median_reference(&numpy_median, &input)
                                    .expect("NumPy median audit call"),
                            );
                            orig_total += start.elapsed();
                            let start = Instant::now();
                            black_box(ledger_radix_median_f64(&data));
                            candidate_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(ledger_radix_median_f64(&data));
                            candidate_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                ledger_orig_median_reference(&numpy_median, &input)
                                    .expect("NumPy median audit call"),
                            );
                            orig_total += start.elapsed();
                        }
                    }
                    candidate_samples
                        .borrow_mut()
                        .push(candidate_total.as_secs_f64() * 1e9 / iterations as f64);
                    orig_samples
                        .borrow_mut()
                        .push(orig_total.as_secs_f64() * 1e9 / iterations as f64);
                    candidate_total + orig_total
                });
            });
            report_ledger_pair(
                "radix_median_f64_normal_16m",
                &candidate_samples,
                &orig_samples,
            );
        }

        {
            let namespace = PyDict::new(py);
            py.run(
                std::ffi::CString::new(
                    "import numpy as np\n\
                     rng = np.random.default_rng(0)\n\
                     f16_input = (rng.integers(1, 4000, 4_000_000) / 7).astype(np.float16)\n",
                )
                .expect("f16 setup CString")
                .as_c_str(),
                Some(&namespace),
                Some(&namespace),
            )
            .expect("f16 setup");
            let input = namespace.get_item("f16_input").expect("f16 input present");
            let bit_bytes: Vec<u8> = input
                .call_method1("view", ("uint16",))
                .expect("f16 uint16 view")
                .call_method0("tobytes")
                .expect("f16 bit bytes")
                .extract()
                .expect("extract f16 bit bytes");
            let input_bits: Vec<u16> = bit_bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| u16::from_ne_bytes(*chunk))
                .collect();
            assert_eq!(input_bits.len(), 4_000_000);
            let numpy_sort = numpy.getattr("sort").expect("numpy.sort");
            let equal = numpy.getattr("array_equal").expect("numpy.array_equal");
            let candidate = ledger_try_native_f16_sort(&numpy_sort, &input, &input_bits)
                .expect("f16 widening candidate parity call");
            let orig =
                ledger_orig_f16_sort_reference(&numpy_sort, &input).expect("f16 ORIG parity call");
            assert!(
                equal
                    .call1((candidate.bind(py), orig.bind(py)))
                    .expect("f16 array_equal")
                    .extract::<bool>()
                    .expect("f16 equality bool"),
                "f16 widening-sort parity",
            );

            let candidate_samples = RefCell::new(Vec::new());
            let orig_samples = RefCell::new(Vec::new());
            let order = Cell::new(0u64);
            group.bench_function("f16_sort_via_f32_widening_4m_paired", |bench| {
                bench.iter_custom(|iterations| {
                    let mut candidate_total = Duration::ZERO;
                    let mut orig_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let orig_first = order.get() & 1 == 1;
                        order.set(order.get().wrapping_add(1));
                        if orig_first {
                            let start = Instant::now();
                            black_box(
                                ledger_orig_f16_sort_reference(&numpy_sort, &input)
                                    .expect("f16 ORIG audit call"),
                            );
                            orig_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                ledger_try_native_f16_sort(&numpy_sort, &input, &input_bits)
                                    .expect("f16 widening audit call"),
                            );
                            candidate_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(
                                ledger_try_native_f16_sort(&numpy_sort, &input, &input_bits)
                                    .expect("f16 widening audit call"),
                            );
                            candidate_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                ledger_orig_f16_sort_reference(&numpy_sort, &input)
                                    .expect("f16 ORIG audit call"),
                            );
                            orig_total += start.elapsed();
                        }
                    }
                    candidate_samples
                        .borrow_mut()
                        .push(candidate_total.as_secs_f64() * 1e9 / iterations as f64);
                    orig_samples
                        .borrow_mut()
                        .push(orig_total.as_secs_f64() * 1e9 / iterations as f64);
                    candidate_total + orig_total
                });
            });
            report_ledger_pair(
                "f16_sort_via_f32_widening_4m",
                &candidate_samples,
                &orig_samples,
            );

            // PRODUCTION arm (bead deadlock-audit-98chw): fnp.sort(f16) now routes through
            // try_native_f16_sort_flat for this input; paired vs numpy.sort in the same
            // interleaved routine, plus an A/A null-control row (per-function noise floor).
            let fnp_sort = module.getattr("sort").expect("fnp sort");
            let prod = fnp_sort.call1((&input,)).expect("fnp f16 sort parity call");
            let prod_bytes: Vec<u8> = prod
                .call_method0("tobytes")
                .expect("prod bytes")
                .extract()
                .expect("extract prod bytes");
            let orig_bytes: Vec<u8> = orig
                .bind(py)
                .call_method0("tobytes")
                .expect("orig bytes")
                .extract()
                .expect("extract orig bytes");
            assert_eq!(
                prod_bytes, orig_bytes,
                "production f16 sort parity (tobytes)"
            );

            let prod_samples = RefCell::new(Vec::new());
            let prod_orig_samples = RefCell::new(Vec::new());
            let prod_order = Cell::new(0u64);
            group.bench_function("f16_sort_production_4m_paired", |bench| {
                bench.iter_custom(|iterations| {
                    let mut cand_total = Duration::ZERO;
                    let mut orig_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let orig_first = prod_order.get() & 1 == 1;
                        prod_order.set(prod_order.get().wrapping_add(1));
                        if orig_first {
                            let start = Instant::now();
                            black_box(numpy_sort.call1((&input,)).expect("numpy f16 sort"));
                            orig_total += start.elapsed();
                            let start = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("fnp f16 sort"));
                            cand_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("fnp f16 sort"));
                            cand_total += start.elapsed();
                            let start = Instant::now();
                            black_box(numpy_sort.call1((&input,)).expect("numpy f16 sort"));
                            orig_total += start.elapsed();
                        }
                    }
                    prod_samples
                        .borrow_mut()
                        .push(cand_total.as_secs_f64() * 1e9 / iterations as f64);
                    prod_orig_samples
                        .borrow_mut()
                        .push(orig_total.as_secs_f64() * 1e9 / iterations as f64);
                    cand_total + orig_total
                });
            });
            report_ledger_pair("f16_sort_production_4m", &prod_samples, &prod_orig_samples);

            let null_a = RefCell::new(Vec::new());
            let null_b = RefCell::new(Vec::new());
            let null_order = Cell::new(0u64);
            group.bench_function("f16_sort_production_4m_null_control", |bench| {
                bench.iter_custom(|iterations| {
                    let mut a_total = Duration::ZERO;
                    let mut b_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let b_first = null_order.get() & 1 == 1;
                        null_order.set(null_order.get().wrapping_add(1));
                        if b_first {
                            let start = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("null b"));
                            b_total += start.elapsed();
                            let start = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("null a"));
                            a_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("null a"));
                            a_total += start.elapsed();
                            let start = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("null b"));
                            b_total += start.elapsed();
                        }
                    }
                    null_a
                        .borrow_mut()
                        .push(a_total.as_secs_f64() * 1e9 / iterations as f64);
                    null_b
                        .borrow_mut()
                        .push(b_total.as_secs_f64() * 1e9 / iterations as f64);
                    a_total + b_total
                });
            });
            report_ledger_pair("f16_sort_production_null_AA", &null_a, &null_b);

            // f16 STABLE ARGSORT production arm (widened stable radix; sibling lever): the
            // same 4M f16 input is tie-dense by construction (63k distinct values), the case
            // where stability is load-bearing. Paired vs numpy + A/A null control.
            let fnp_argsort = module.getattr("argsort").expect("fnp argsort");
            let numpy_argsort = numpy.getattr("argsort").expect("numpy argsort");
            let stable_kw = pyo3::types::PyDict::new(py);
            stable_kw.set_item("kind", "stable").expect("kind kwarg");
            let ag_prod = fnp_argsort
                .call((&input,), Some(&stable_kw))
                .expect("fnp f16 stable argsort parity call");
            let ag_orig = numpy_argsort
                .call((&input,), Some(&stable_kw))
                .expect("numpy f16 stable argsort parity call");
            assert!(
                equal
                    .call1((&ag_prod, &ag_orig))
                    .expect("f16 argsort array_equal")
                    .extract::<bool>()
                    .expect("f16 argsort equality bool"),
                "production f16 stable argsort parity",
            );

            let ag_samples = RefCell::new(Vec::new());
            let ag_orig_samples = RefCell::new(Vec::new());
            let ag_order = Cell::new(0u64);
            group.bench_function("f16_argsort_stable_production_4m_paired", |bench| {
                bench.iter_custom(|iterations| {
                    let mut cand_total = Duration::ZERO;
                    let mut orig_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let orig_first = ag_order.get() & 1 == 1;
                        ag_order.set(ag_order.get().wrapping_add(1));
                        if orig_first {
                            let start = Instant::now();
                            black_box(
                                numpy_argsort
                                    .call((&input,), Some(&stable_kw))
                                    .expect("numpy f16 argsort"),
                            );
                            orig_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                fnp_argsort
                                    .call((&input,), Some(&stable_kw))
                                    .expect("fnp f16 argsort"),
                            );
                            cand_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(
                                fnp_argsort
                                    .call((&input,), Some(&stable_kw))
                                    .expect("fnp f16 argsort"),
                            );
                            cand_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                numpy_argsort
                                    .call((&input,), Some(&stable_kw))
                                    .expect("numpy f16 argsort"),
                            );
                            orig_total += start.elapsed();
                        }
                    }
                    ag_samples
                        .borrow_mut()
                        .push(cand_total.as_secs_f64() * 1e9 / iterations as f64);
                    ag_orig_samples
                        .borrow_mut()
                        .push(orig_total.as_secs_f64() * 1e9 / iterations as f64);
                    cand_total + orig_total
                });
            });
            report_ledger_pair(
                "f16_argsort_stable_production_4m",
                &ag_samples,
                &ag_orig_samples,
            );

            let ag_null_a = RefCell::new(Vec::new());
            let ag_null_b = RefCell::new(Vec::new());
            let ag_null_order = Cell::new(0u64);
            group.bench_function("f16_argsort_stable_production_4m_null_control", |bench| {
                bench.iter_custom(|iterations| {
                    let mut a_total = Duration::ZERO;
                    let mut b_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let b_first = ag_null_order.get() & 1 == 1;
                        ag_null_order.set(ag_null_order.get().wrapping_add(1));
                        if b_first {
                            let start = Instant::now();
                            black_box(
                                fnp_argsort
                                    .call((&input,), Some(&stable_kw))
                                    .expect("null b"),
                            );
                            b_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                fnp_argsort
                                    .call((&input,), Some(&stable_kw))
                                    .expect("null a"),
                            );
                            a_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(
                                fnp_argsort
                                    .call((&input,), Some(&stable_kw))
                                    .expect("null a"),
                            );
                            a_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                fnp_argsort
                                    .call((&input,), Some(&stable_kw))
                                    .expect("null b"),
                            );
                            b_total += start.elapsed();
                        }
                    }
                    ag_null_a
                        .borrow_mut()
                        .push(a_total.as_secs_f64() * 1e9 / iterations as f64);
                    ag_null_b
                        .borrow_mut()
                        .push(b_total.as_secs_f64() * 1e9 / iterations as f64);
                    a_total + b_total
                });
            });
            report_ledger_pair("f16_argsort_stable_null_AA", &ag_null_a, &ag_null_b);

            // LAST-AXIS siblings (2000x2000 view of the same 4M input): per-lane widened
            // value sort + per-lane widened stable argsort, paired with A/A null controls.
            let input2d = input
                .call_method1("reshape", ((2000, 2000),))
                .expect("reshape 2000x2000");
            let axis_kw = pyo3::types::PyDict::new(py);
            axis_kw.set_item("axis", -1).expect("axis kwarg");
            let fnp_sort2 = module.getattr("sort").expect("fnp sort");
            let numpy_sort2 = numpy.getattr("sort").expect("numpy sort");
            let s2_f = fnp_sort2
                .call((&input2d,), Some(&axis_kw))
                .expect("fnp f16 lastaxis sort parity");
            let s2_n = numpy_sort2
                .call((&input2d,), Some(&axis_kw))
                .expect("numpy f16 lastaxis sort parity");
            let s2_fb: Vec<u8> = s2_f
                .call_method0("tobytes")
                .expect("bytes")
                .extract()
                .expect("extract");
            let s2_nb: Vec<u8> = s2_n
                .call_method0("tobytes")
                .expect("bytes")
                .extract()
                .expect("extract");
            assert_eq!(s2_fb, s2_nb, "f16 lastaxis sort parity (tobytes)");
            let stable_axis_kw = pyo3::types::PyDict::new(py);
            stable_axis_kw.set_item("axis", -1).expect("axis kwarg");
            stable_axis_kw
                .set_item("kind", "stable")
                .expect("kind kwarg");
            let a2_f = fnp_argsort
                .call((&input2d,), Some(&stable_axis_kw))
                .expect("fnp f16 lastaxis argsort parity");
            let a2_n = numpy_argsort
                .call((&input2d,), Some(&stable_axis_kw))
                .expect("numpy f16 lastaxis argsort parity");
            assert!(
                equal
                    .call1((&a2_f, &a2_n))
                    .expect("array_equal")
                    .extract::<bool>()
                    .expect("bool"),
                "f16 lastaxis stable argsort parity",
            );

            for (label, fnp_fn, numpy_fn, kw) in [
                (
                    "f16_sort_lastaxis_2000x2000",
                    &fnp_sort2,
                    &numpy_sort2,
                    &axis_kw,
                ),
                (
                    "f16_argsort_stable_lastaxis_2000x2000",
                    &fnp_argsort,
                    &numpy_argsort,
                    &stable_axis_kw,
                ),
            ] {
                let cand = RefCell::new(Vec::new());
                let orig = RefCell::new(Vec::new());
                let ord = Cell::new(0u64);
                group.bench_function(format!("{label}_paired"), |bench| {
                    bench.iter_custom(|iterations| {
                        let mut ct = Duration::ZERO;
                        let mut ot = Duration::ZERO;
                        for _ in 0..iterations {
                            let of = ord.get() & 1 == 1;
                            ord.set(ord.get().wrapping_add(1));
                            if of {
                                let s = Instant::now();
                                black_box(numpy_fn.call((&input2d,), Some(kw)).expect("orig"));
                                ot += s.elapsed();
                                let s = Instant::now();
                                black_box(fnp_fn.call((&input2d,), Some(kw)).expect("cand"));
                                ct += s.elapsed();
                            } else {
                                let s = Instant::now();
                                black_box(fnp_fn.call((&input2d,), Some(kw)).expect("cand"));
                                ct += s.elapsed();
                                let s = Instant::now();
                                black_box(numpy_fn.call((&input2d,), Some(kw)).expect("orig"));
                                ot += s.elapsed();
                            }
                        }
                        cand.borrow_mut()
                            .push(ct.as_secs_f64() * 1e9 / iterations as f64);
                        orig.borrow_mut()
                            .push(ot.as_secs_f64() * 1e9 / iterations as f64);
                        ct + ot
                    });
                });
                report_ledger_pair(label, &cand, &orig);

                let na = RefCell::new(Vec::new());
                let nb = RefCell::new(Vec::new());
                let nord = Cell::new(0u64);
                group.bench_function(format!("{label}_null_control"), |bench| {
                    bench.iter_custom(|iterations| {
                        let mut at = Duration::ZERO;
                        let mut bt = Duration::ZERO;
                        for _ in 0..iterations {
                            let bf = nord.get() & 1 == 1;
                            nord.set(nord.get().wrapping_add(1));
                            if bf {
                                let s = Instant::now();
                                black_box(fnp_fn.call((&input2d,), Some(kw)).expect("nb"));
                                bt += s.elapsed();
                                let s = Instant::now();
                                black_box(fnp_fn.call((&input2d,), Some(kw)).expect("na"));
                                at += s.elapsed();
                            } else {
                                let s = Instant::now();
                                black_box(fnp_fn.call((&input2d,), Some(kw)).expect("na"));
                                at += s.elapsed();
                                let s = Instant::now();
                                black_box(fnp_fn.call((&input2d,), Some(kw)).expect("nb"));
                                bt += s.elapsed();
                            }
                        }
                        na.borrow_mut()
                            .push(at.as_secs_f64() * 1e9 / iterations as f64);
                        nb.borrow_mut()
                            .push(bt.as_secs_f64() * 1e9 / iterations as f64);
                        at + bt
                    });
                });
                report_ledger_pair(&format!("{label}_null_AA"), &na, &nb);
            }
        }

        {
            // Narrow-int counting sort: i16 8M full-range (the 2-byte case is where numpy's
            // serial radix is slowest). Paired vs numpy + A/A null control.
            let namespace = PyDict::new(py);
            py.run(
                std::ffi::CString::new(
                    "import numpy as np\n\
                     rng = np.random.default_rng(5)\n\
                     i16_input = rng.integers(-32768, 32768, 8_000_000, dtype=np.int16)\n",
                )
                .expect("i16 setup CString")
                .as_c_str(),
                Some(&namespace),
                Some(&namespace),
            )
            .expect("i16 setup");
            let input = namespace.get_item("i16_input").expect("i16 input");
            let fnp_sort = module.getattr("sort").expect("fnp sort");
            let numpy_sort = numpy.getattr("sort").expect("numpy sort");
            let equal = numpy.getattr("array_equal").expect("array_equal");
            let f = fnp_sort.call1((&input,)).expect("fnp i16 sort parity");
            let nres = numpy_sort.call1((&input,)).expect("numpy i16 sort parity");
            assert!(
                equal
                    .call1((&f, &nres))
                    .expect("array_equal")
                    .extract::<bool>()
                    .expect("bool"),
                "narrow-int i16 sort parity",
            );

            let cand = RefCell::new(Vec::new());
            let orig = RefCell::new(Vec::new());
            let ord = Cell::new(0u64);
            group.bench_function("narrow_int_i16_sort_8m_paired", |bench| {
                bench.iter_custom(|iterations| {
                    let mut ct = Duration::ZERO;
                    let mut ot = Duration::ZERO;
                    for _ in 0..iterations {
                        let of = ord.get() & 1 == 1;
                        ord.set(ord.get().wrapping_add(1));
                        if of {
                            let s = Instant::now();
                            black_box(numpy_sort.call1((&input,)).expect("orig"));
                            ot += s.elapsed();
                            let s = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("cand"));
                            ct += s.elapsed();
                        } else {
                            let s = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("cand"));
                            ct += s.elapsed();
                            let s = Instant::now();
                            black_box(numpy_sort.call1((&input,)).expect("orig"));
                            ot += s.elapsed();
                        }
                    }
                    cand.borrow_mut()
                        .push(ct.as_secs_f64() * 1e9 / iterations as f64);
                    orig.borrow_mut()
                        .push(ot.as_secs_f64() * 1e9 / iterations as f64);
                    ct + ot
                });
            });
            report_ledger_pair("narrow_int_i16_sort_8m", &cand, &orig);

            let na = RefCell::new(Vec::new());
            let nb = RefCell::new(Vec::new());
            let nord = Cell::new(0u64);
            group.bench_function("narrow_int_i16_sort_8m_null_control", |bench| {
                bench.iter_custom(|iterations| {
                    let mut at = Duration::ZERO;
                    let mut bt = Duration::ZERO;
                    for _ in 0..iterations {
                        let bf = nord.get() & 1 == 1;
                        nord.set(nord.get().wrapping_add(1));
                        if bf {
                            let s = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("nb"));
                            bt += s.elapsed();
                            let s = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("na"));
                            at += s.elapsed();
                        } else {
                            let s = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("na"));
                            at += s.elapsed();
                            let s = Instant::now();
                            black_box(fnp_sort.call1((&input,)).expect("nb"));
                            bt += s.elapsed();
                        }
                    }
                    na.borrow_mut()
                        .push(at.as_secs_f64() * 1e9 / iterations as f64);
                    nb.borrow_mut()
                        .push(bt.as_secs_f64() * 1e9 / iterations as f64);
                    at + bt
                });
            });
            report_ledger_pair("narrow_int_i16_sort_null_AA", &na, &nb);

            // Stable ARGSORT sibling on the same 8M i16 input (dense ties by construction;
            // routes to the parallel counting-prefix stable argsort). Paired + A/A null.
            let fnp_argsort_n = module.getattr("argsort").expect("fnp argsort");
            let numpy_argsort_n = numpy.getattr("argsort").expect("numpy argsort");
            let skw = pyo3::types::PyDict::new(py);
            skw.set_item("kind", "stable").expect("kind kwarg");
            let af = fnp_argsort_n
                .call((&input,), Some(&skw))
                .expect("fnp i16 stable argsort parity");
            let an = numpy_argsort_n
                .call((&input,), Some(&skw))
                .expect("numpy i16 stable argsort parity");
            assert!(
                equal
                    .call1((&af, &an))
                    .expect("array_equal")
                    .extract::<bool>()
                    .expect("bool"),
                "narrow-int i16 stable argsort parity",
            );
            let cand2 = RefCell::new(Vec::new());
            let orig2 = RefCell::new(Vec::new());
            let ord2 = Cell::new(0u64);
            group.bench_function("narrow_int_i16_argsort_stable_8m_paired", |bench| {
                bench.iter_custom(|iterations| {
                    let mut ct = Duration::ZERO;
                    let mut ot = Duration::ZERO;
                    for _ in 0..iterations {
                        let of = ord2.get() & 1 == 1;
                        ord2.set(ord2.get().wrapping_add(1));
                        if of {
                            let s = Instant::now();
                            black_box(numpy_argsort_n.call((&input,), Some(&skw)).expect("orig"));
                            ot += s.elapsed();
                            let s = Instant::now();
                            black_box(fnp_argsort_n.call((&input,), Some(&skw)).expect("cand"));
                            ct += s.elapsed();
                        } else {
                            let s = Instant::now();
                            black_box(fnp_argsort_n.call((&input,), Some(&skw)).expect("cand"));
                            ct += s.elapsed();
                            let s = Instant::now();
                            black_box(numpy_argsort_n.call((&input,), Some(&skw)).expect("orig"));
                            ot += s.elapsed();
                        }
                    }
                    cand2
                        .borrow_mut()
                        .push(ct.as_secs_f64() * 1e9 / iterations as f64);
                    orig2
                        .borrow_mut()
                        .push(ot.as_secs_f64() * 1e9 / iterations as f64);
                    ct + ot
                });
            });
            report_ledger_pair("narrow_int_i16_argsort_stable_8m", &cand2, &orig2);

            let na2 = RefCell::new(Vec::new());
            let nb2 = RefCell::new(Vec::new());
            let nord2 = Cell::new(0u64);
            group.bench_function("narrow_int_i16_argsort_stable_8m_null_control", |bench| {
                bench.iter_custom(|iterations| {
                    let mut at = Duration::ZERO;
                    let mut bt = Duration::ZERO;
                    for _ in 0..iterations {
                        let bf = nord2.get() & 1 == 1;
                        nord2.set(nord2.get().wrapping_add(1));
                        if bf {
                            let s = Instant::now();
                            black_box(fnp_argsort_n.call((&input,), Some(&skw)).expect("nb"));
                            bt += s.elapsed();
                            let s = Instant::now();
                            black_box(fnp_argsort_n.call((&input,), Some(&skw)).expect("na"));
                            at += s.elapsed();
                        } else {
                            let s = Instant::now();
                            black_box(fnp_argsort_n.call((&input,), Some(&skw)).expect("na"));
                            at += s.elapsed();
                            let s = Instant::now();
                            black_box(fnp_argsort_n.call((&input,), Some(&skw)).expect("nb"));
                            bt += s.elapsed();
                        }
                    }
                    na2.borrow_mut()
                        .push(at.as_secs_f64() * 1e9 / iterations as f64);
                    nb2.borrow_mut()
                        .push(bt.as_secs_f64() * 1e9 / iterations as f64);
                    at + bt
                });
            });
            report_ledger_pair("narrow_int_i16_argsort_stable_null_AA", &na2, &nb2);

            // LAST-AXIS siblings on a 4000x2000 view of the same 8M i16 input: per-lane sort
            // + per-lane stable argsort, paired with A/A null controls.
            let input2d = input
                .call_method1("reshape", ((4000, 2000),))
                .expect("reshape 4000x2000");
            let axkw = pyo3::types::PyDict::new(py);
            axkw.set_item("axis", -1).expect("axis kwarg");
            let stax_kw = pyo3::types::PyDict::new(py);
            stax_kw.set_item("axis", -1).expect("axis kwarg");
            stax_kw.set_item("kind", "stable").expect("kind kwarg");
            let sf = fnp_sort
                .call((&input2d,), Some(&axkw))
                .expect("fnp lastaxis parity");
            let sn = numpy_sort
                .call((&input2d,), Some(&axkw))
                .expect("numpy lastaxis parity");
            let sfb: Vec<u8> = sf.call_method0("tobytes").expect("b").extract().expect("e");
            let snb: Vec<u8> = sn.call_method0("tobytes").expect("b").extract().expect("e");
            assert_eq!(sfb, snb, "narrow-int i16 lastaxis sort parity");
            let gf = fnp_argsort_n
                .call((&input2d,), Some(&stax_kw))
                .expect("fnp lastaxis argsort parity");
            let gn = numpy_argsort_n
                .call((&input2d,), Some(&stax_kw))
                .expect("numpy lastaxis argsort parity");
            assert!(
                equal
                    .call1((&gf, &gn))
                    .expect("array_equal")
                    .extract::<bool>()
                    .expect("bool"),
                "narrow-int i16 lastaxis stable argsort parity",
            );
            for (label, ffn, nfn, kw) in [
                (
                    "narrow_int_i16_sort_lastaxis_4000x2000",
                    &fnp_sort,
                    &numpy_sort,
                    &axkw,
                ),
                (
                    "narrow_int_i16_argsort_stable_lastaxis_4000x2000",
                    &fnp_argsort_n,
                    &numpy_argsort_n,
                    &stax_kw,
                ),
            ] {
                let cand = RefCell::new(Vec::new());
                let orig = RefCell::new(Vec::new());
                let ord = Cell::new(0u64);
                group.bench_function(format!("{label}_paired"), |bench| {
                    bench.iter_custom(|iterations| {
                        let mut ct = Duration::ZERO;
                        let mut ot = Duration::ZERO;
                        for _ in 0..iterations {
                            let of = ord.get() & 1 == 1;
                            ord.set(ord.get().wrapping_add(1));
                            if of {
                                let s = Instant::now();
                                black_box(nfn.call((&input2d,), Some(kw)).expect("orig"));
                                ot += s.elapsed();
                                let s = Instant::now();
                                black_box(ffn.call((&input2d,), Some(kw)).expect("cand"));
                                ct += s.elapsed();
                            } else {
                                let s = Instant::now();
                                black_box(ffn.call((&input2d,), Some(kw)).expect("cand"));
                                ct += s.elapsed();
                                let s = Instant::now();
                                black_box(nfn.call((&input2d,), Some(kw)).expect("orig"));
                                ot += s.elapsed();
                            }
                        }
                        cand.borrow_mut()
                            .push(ct.as_secs_f64() * 1e9 / iterations as f64);
                        orig.borrow_mut()
                            .push(ot.as_secs_f64() * 1e9 / iterations as f64);
                        ct + ot
                    });
                });
                report_ledger_pair(label, &cand, &orig);
                let na = RefCell::new(Vec::new());
                let nb = RefCell::new(Vec::new());
                let nord = Cell::new(0u64);
                group.bench_function(format!("{label}_null_control"), |bench| {
                    bench.iter_custom(|iterations| {
                        let mut at = Duration::ZERO;
                        let mut bt = Duration::ZERO;
                        for _ in 0..iterations {
                            let bf = nord.get() & 1 == 1;
                            nord.set(nord.get().wrapping_add(1));
                            if bf {
                                let s = Instant::now();
                                black_box(ffn.call((&input2d,), Some(kw)).expect("nb"));
                                bt += s.elapsed();
                                let s = Instant::now();
                                black_box(ffn.call((&input2d,), Some(kw)).expect("na"));
                                at += s.elapsed();
                            } else {
                                let s = Instant::now();
                                black_box(ffn.call((&input2d,), Some(kw)).expect("na"));
                                at += s.elapsed();
                                let s = Instant::now();
                                black_box(ffn.call((&input2d,), Some(kw)).expect("nb"));
                                bt += s.elapsed();
                            }
                        }
                        na.borrow_mut()
                            .push(at.as_secs_f64() * 1e9 / iterations as f64);
                        nb.borrow_mut()
                            .push(bt.as_secs_f64() * 1e9 / iterations as f64);
                        at + bt
                    });
                });
                report_ledger_pair(&format!("{label}_null_AA"), &na, &nb);
            }
        }

        {
            let namespace = PyDict::new(py);
            py.run(
                std::ffi::CString::new(
                    "import numpy as np\n\
                     rng = np.random.default_rng(0)\n\
                     f32_ties = np.round(rng.standard_normal(2_000_000), 2).astype(np.float32)\n",
                )
                .expect("f32 argsort setup CString")
                .as_c_str(),
                Some(&namespace),
                Some(&namespace),
            )
            .expect("f32 argsort setup");
            let input = namespace
                .get_item("f32_ties")
                .expect("f32 tie input present");
            let fnp_argsort = module.getattr("argsort").expect("fnp argsort");
            let numpy_argsort = numpy.getattr("argsort").expect("numpy.argsort");
            let equal = numpy.getattr("array_equal").expect("numpy.array_equal");
            let candidate = ledger_f32_tie_argsort_candidate(&fnp_argsort, &input)
                .expect("f32 tied candidate parity call");
            let orig = ledger_orig_f32_argsort_reference(&numpy_argsort, &input)
                .expect("f32 tied ORIG parity call");
            assert!(
                equal
                    .call1((candidate.bind(py), orig.bind(py)))
                    .expect("f32 argsort array_equal")
                    .extract::<bool>()
                    .expect("f32 argsort equality bool"),
                "tied f32 argsort parity",
            );

            let candidate_samples = RefCell::new(Vec::new());
            let orig_samples = RefCell::new(Vec::new());
            let order = Cell::new(0u64);
            group.bench_function("f32_argsort_rounded_ties_2m_paired", |bench| {
                bench.iter_custom(|iterations| {
                    let mut candidate_total = Duration::ZERO;
                    let mut orig_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let orig_first = order.get() & 1 == 1;
                        order.set(order.get().wrapping_add(1));
                        if orig_first {
                            let start = Instant::now();
                            black_box(
                                ledger_orig_f32_argsort_reference(&numpy_argsort, &input)
                                    .expect("f32 argsort ORIG audit call"),
                            );
                            orig_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                ledger_f32_tie_argsort_candidate(&fnp_argsort, &input)
                                    .expect("f32 argsort candidate audit call"),
                            );
                            candidate_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(
                                ledger_f32_tie_argsort_candidate(&fnp_argsort, &input)
                                    .expect("f32 argsort candidate audit call"),
                            );
                            candidate_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                ledger_orig_f32_argsort_reference(&numpy_argsort, &input)
                                    .expect("f32 argsort ORIG audit call"),
                            );
                            orig_total += start.elapsed();
                        }
                    }
                    candidate_samples
                        .borrow_mut()
                        .push(candidate_total.as_secs_f64() * 1e9 / iterations as f64);
                    orig_samples
                        .borrow_mut()
                        .push(orig_total.as_secs_f64() * 1e9 / iterations as f64);
                    candidate_total + orig_total
                });
            });
            report_ledger_pair(
                "f32_argsort_rounded_ties_2m",
                &candidate_samples,
                &orig_samples,
            );

            // NULL CONTROL (A/A): the candidate arm registered twice in the same interleaved
            // routine. Its ratio and cv are the harness noise floor - any lever effect below
            // this floor is undecidable on this harness (franken_whisper null-control rule).
            let null_a = RefCell::new(Vec::new());
            let null_b = RefCell::new(Vec::new());
            let null_order = Cell::new(0u64);
            group.bench_function("f32_argsort_rounded_ties_2m_null_control", |bench| {
                bench.iter_custom(|iterations| {
                    let mut a_total = Duration::ZERO;
                    let mut b_total = Duration::ZERO;
                    for _ in 0..iterations {
                        let b_first = null_order.get() & 1 == 1;
                        null_order.set(null_order.get().wrapping_add(1));
                        if b_first {
                            let start = Instant::now();
                            black_box(
                                ledger_f32_tie_argsort_candidate(&fnp_argsort, &input)
                                    .expect("null-control arm b"),
                            );
                            b_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                ledger_f32_tie_argsort_candidate(&fnp_argsort, &input)
                                    .expect("null-control arm a"),
                            );
                            a_total += start.elapsed();
                        } else {
                            let start = Instant::now();
                            black_box(
                                ledger_f32_tie_argsort_candidate(&fnp_argsort, &input)
                                    .expect("null-control arm a"),
                            );
                            a_total += start.elapsed();
                            let start = Instant::now();
                            black_box(
                                ledger_f32_tie_argsort_candidate(&fnp_argsort, &input)
                                    .expect("null-control arm b"),
                            );
                            b_total += start.elapsed();
                        }
                    }
                    null_a
                        .borrow_mut()
                        .push(a_total.as_secs_f64() * 1e9 / iterations as f64);
                    null_b
                        .borrow_mut()
                        .push(b_total.as_secs_f64() * 1e9 / iterations as f64);
                    a_total + b_total
                });
            });
            report_ledger_pair("f32_argsort_null_control_AA", &null_a, &null_b);

            // Self-time of the pre-check unit the dispatch dedupe removes: ONE full parallel
            // NaN scan + ONE 65,536-sample strided tie oracle over the same 2M f32 buffer
            // (bench-local reconstruction of the dispatch's NaN scan + argsort_sample_has_tie;
            // before the fix, dense-tie input paid this unit TWICE - radix candidate then
            // comparison candidate - before delegation).
            let raw: Vec<u8> = input
                .call_method0("tobytes")
                .expect("f32 tie bytes")
                .extract()
                .expect("extract f32 tie bytes");
            let data: Vec<f32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_ne_bytes(*chunk))
                .collect();
            group.bench_function("f32_argsort_tie_precheck_selftime_2m", |bench| {
                bench.iter(|| {
                    use rayon::prelude::*;
                    let d = black_box(&data);
                    let nan = d.par_iter().any(|v| v.is_nan());
                    const TIE_SAMPLE: usize = 1 << 16;
                    let n = d.len();
                    let k = n.min(TIE_SAMPLE);
                    let stride = (n / k).max(1);
                    let mut sample: Vec<f32> = (0..k).map(|i| d[i * stride]).collect();
                    sample.sort_unstable_by(|a, b| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let tie = (1..sample.len()).any(|i| sample[i] == sample[i - 1]);
                    black_box((nan, tie))
                });
            });
        }
    });

    group.finish();
}

/// Resurrection of `franken_numpy-ixs5y.377` (ledger row `NEGATIVE_EVIDENCE.md:300`,
/// audit rank 1). The original measured 3.09-3.69x across five runs against A/A
/// nulls of 0.988-1.045 and was rejected anyway, because a predeclared gate
/// required all four arms to clear `cv < 5%` — a threshold campaign §2.3 shows is
/// unreachable on this hardware. Re-decided here on the bootstrapped median-CI
/// contract: the effect must clear twice the A/A CI half-width, and `cv` is
/// provenance only.
///
/// SAME-BINARY MAINTENANCE CONTROL: both sides are
/// `fnp.loadtxt` on the identical file selecting the identical columns. The base
/// uses NEGATIVE `usecols`, which the direct path deliberately declines (negative
/// indices resolve against the row width, which the borrowed view does not
/// hoist), so it walks the former `Vec<Vec<String>>` owned-token path. The
/// candidate uses the equivalent non-negative indices and takes the new path.
/// One ELF, one file, one selection — the only difference is the code path, so
/// this isolates the lever. Under the campaign's incumbent-only win policy that
/// first ratio is `maintenance-self-speedup`; the second contract below compares
/// the candidate with the actual `numpy.loadtxt` incumbent.
///
/// The corpus poisons an UNSELECTED column with a non-bool token. NumPy never
/// parses unselected columns and neither may the direct path; if that ever
/// regresses, this bench fails its parity assert before it times anything.
fn bench_loadtxt_selected_bool_median_gate(_c: &mut Criterion) {
    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_loadtxt_selected_bool")
            .expect("loadtxt selected-bool bench module");
        fnp_python(&module).expect("initialize fnp_python loadtxt bench module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np, os, tempfile\n\
                 rng = np.random.default_rng(20260726)\n\
                 rows, cols = 8192, 16\n\
                 vals = rng.integers(0, 2, size=(rows, cols))\n\
                 lines = []\n\
                 for r in range(rows):\n\
                 \x20   cells = [str(v) for v in vals[r]]\n\
                 \x20   cells[7] = 'not_a_bool'\n\
                 \x20   lines.append(','.join(cells))\n\
                 lt_path = os.path.join(tempfile.gettempdir(),\n\
                 \x20   'fnp_loadtxt_selected_bool_%d.csv' % os.getpid())\n\
                 with open(lt_path, 'w') as fh:\n\
                 \x20   fh.write('\\n'.join(lines) + '\\n')\n",
            )
            .expect("loadtxt corpus CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("loadtxt corpus setup");
        let lt_path = namespace.get_item("lt_path").expect("lt_path present");

        let functools = py.import("functools").expect("functools");
        let partial = functools.getattr("partial").expect("functools.partial");
        let fnp_loadtxt = module.getattr("loadtxt").expect("fnp loadtxt");
        let numpy_loadtxt = numpy.getattr("loadtxt").expect("numpy loadtxt");
        assert!(
            !numpy_loadtxt.is(&fnp_loadtxt),
            "dispatch trap: incumbent loadtxt resolved to the FNP callable"
        );
        common::report_numpy_loadtxt_incumbent_identity(py, &numpy_loadtxt);
        let bool_dtype = numpy.getattr("bool_").expect("numpy bool_");

        // 16 columns: -16 == 0, -15 == 1, -13 == 3, -12 == 4. Same four columns,
        // and column 7 (the poisoned one) is selected by neither.
        let make_fnp_arm = |cols: Vec<i64>| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("delimiter", ",").expect("delimiter kwarg");
            kwargs.set_item("dtype", &bool_dtype).expect("dtype kwarg");
            kwargs.set_item("usecols", cols).expect("usecols kwarg");
            partial
                .call((&fnp_loadtxt,), Some(&kwargs))
                .expect("partial-bound loadtxt arm")
        };
        let former = make_fnp_arm(vec![-16, -15, -13, -12]);
        let candidate = make_fnp_arm(vec![0, 1, 3, 4]);
        let incumbent_kwargs = PyDict::new(py);
        incumbent_kwargs
            .set_item("delimiter", ",")
            .expect("incumbent delimiter");
        incumbent_kwargs
            .set_item("dtype", &bool_dtype)
            .expect("incumbent dtype");
        incumbent_kwargs
            .set_item("usecols", vec![0_i64, 1, 3, 4])
            .expect("incumbent usecols");
        let incumbent = partial
            .call((&numpy_loadtxt,), Some(&incumbent_kwargs))
            .expect("partial-bound numpy loadtxt arm");

        // BEHAVIOR BEFORE TIMING: the two paths must be byte-identical, and both
        // must match the live NumPy oracle.
        let former_out = former.call1((&lt_path,)).expect("former path parity");
        let candidate_out = candidate.call1((&lt_path,)).expect("direct path parity");
        let oracle = incumbent.call1((&lt_path,)).expect("numpy oracle parity");
        let bytes_of = |value: &Bound<'_, PyAny>| {
            value
                .call_method0("tobytes")
                .expect("tobytes")
                .extract::<Vec<u8>>()
                .expect("byte Vec")
        };
        assert_eq!(
            bytes_of(&former_out),
            bytes_of(&candidate_out),
            "selected-bool direct path must be byte-identical to the former owned-token path",
        );
        assert_eq!(
            bytes_of(&candidate_out),
            bytes_of(&oracle),
            "selected-bool direct path must be byte-identical to numpy",
        );

        // Prove this valid compatible workload cannot be silently delegated to
        // the incumbent. Restore NumPy before any timed call.
        let poison = PyModule::from_code(
            py,
            pyo3::ffi::c_str!(
                "def fail(*args, **kwargs):\n    raise RuntimeError('numpy.loadtxt passthrough called')\n"
            ),
            pyo3::ffi::c_str!("poison_loadtxt_bench.py"),
            pyo3::ffi::c_str!("poison_loadtxt_bench"),
        )
        .expect("loadtxt poison module");
        numpy
            .setattr("loadtxt", poison.getattr("fail").expect("poison callable"))
            .expect("install loadtxt poison");
        let native_out = candidate
            .call1((&lt_path,))
            .expect("selected-bool path must not delegate");
        numpy
            .setattr("loadtxt", &numpy_loadtxt)
            .expect("restore numpy.loadtxt");
        assert_eq!(
            bytes_of(&native_out),
            bytes_of(&oracle),
            "native selected-bool path must retain NumPy bytes",
        );
        println!(
            "DISPATCH_PROOF row=python_loadtxt_selected_bool_8192x16_vs_numpy \
             fnp_callable=fnp_python.loadtxt delegated_to_numpy=false"
        );

        let time_arm = |function: &Bound<'_, PyAny>| {
            let started = Instant::now();
            let output = function
                .call1((&lt_path,))
                .expect("selected-bool median-CI arm");
            let elapsed = started.elapsed();
            let checksum = bytes_of(&output)
                .into_iter()
                .fold(0xcbf2_9ce4_8422_2325_u64, |state, byte| {
                    (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                });
            black_box(output);
            common::ContractObservation { elapsed, checksum }
        };
        let _ = common::run_median_ci_contract(
            "loadtxt_selected_bool_8192x16",
            || time_arm(&former),
            || time_arm(&candidate),
        );
        // The incumbent decision needs both sides' A/A controls: NumPy/NumPy
        // establishes incumbent stability and FNP/FNP catches candidate-only
        // variance before the same-process NumPy/FNP effect is classified.
        let _ = common::run_dual_null_median_ci_contract(
            "python_loadtxt_selected_bool_8192x16_vs_numpy",
            || time_arm(&incumbent),
            || time_arm(&candidate),
        );
    });
}

/// End-to-end incumbent decision for the bounded all-negative `usecols`
/// tail-ring. Both arms call the compatible Python surface with the identical
/// path, dtype, delimiter, and duplicate/order-preserving selection. The FNP
/// binding routes this measured f64 shape into `fnp_io::loadtxt_usecols_signed`;
/// mixed, positive, empty, oversized, and unpacked cases stay on their former
/// paths and are outside this row.
fn bench_loadtxt_negative_tail_vs_numpy_median_gate(_c: &mut Criterion) {
    const ROWS: usize = 8_192;
    const COLS: usize = 64;
    const USECOLS: [i64; 4] = [-1, -8, -32, -1];

    let mut text = String::new();
    for row in 0..ROWS {
        for column in 0..COLS {
            if column > 0 {
                text.push(',');
            }
            text.push_str(&format!("{}.{}", row % 977, column));
        }
        text.push('\n');
    }
    let path = std::env::temp_dir().join(format!(
        "fnp_loadtxt_negative_tail_bench_{}.csv",
        std::process::id()
    ));
    std::fs::write(&path, text).expect("write negative-tail bench corpus");
    let path = path.to_str().expect("temporary path is UTF-8").to_owned();

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_loadtxt_negative_tail")
            .expect("negative-tail bench module");
        fnp_python(&module).expect("initialize fnp_python loadtxt bench module");
        let numpy = py.import("numpy").expect("numpy incumbent");
        let fnp_loadtxt = module.getattr("loadtxt").expect("fnp loadtxt");
        let numpy_loadtxt = numpy.getattr("loadtxt").expect("numpy loadtxt");
        assert!(
            !numpy_loadtxt.is(&fnp_loadtxt),
            "dispatch trap: incumbent loadtxt resolved to the FNP callable"
        );
        common::report_numpy_loadtxt_incumbent_identity(py, &numpy_loadtxt);

        let float64 = numpy.getattr("float64").expect("numpy float64");
        let functools = py.import("functools").expect("functools");
        let partial = functools.getattr("partial").expect("functools.partial");
        let kwargs = PyDict::new(py);
        kwargs.set_item("delimiter", ",").expect("delimiter kwarg");
        kwargs.set_item("dtype", &float64).expect("dtype kwarg");
        kwargs
            .set_item("usecols", USECOLS)
            .expect("negative usecols kwarg");
        let candidate = partial
            .call((&fnp_loadtxt,), Some(&kwargs))
            .expect("partial-bound FNP loadtxt arm");
        let incumbent = partial
            .call((&numpy_loadtxt,), Some(&kwargs))
            .expect("partial-bound NumPy loadtxt arm");

        let bytes_of = |value: &Bound<'_, PyAny>| {
            value
                .call_method0("tobytes")
                .expect("tobytes")
                .extract::<Vec<u8>>()
                .expect("byte Vec")
        };
        let oracle = incumbent
            .call1((&path,))
            .expect("negative-tail NumPy parity");
        let candidate_out = candidate.call1((&path,)).expect("negative-tail FNP parity");
        assert_eq!(
            bytes_of(&candidate_out),
            bytes_of(&oracle),
            "negative-tail FNP path must be byte-identical to NumPy",
        );
        assert_eq!(
            candidate_out
                .getattr("shape")
                .expect("candidate shape")
                .extract::<Vec<usize>>()
                .expect("candidate shape vector"),
            [ROWS, USECOLS.len()],
        );

        // A valid all-negative bounded selection must stay native. If the
        // binding accidentally falls back to NumPy, this call fails loudly.
        let poison = PyModule::from_code(
            py,
            pyo3::ffi::c_str!(
                "def fail(*args, **kwargs):\n    raise RuntimeError('numpy.loadtxt passthrough called')\n"
            ),
            pyo3::ffi::c_str!("poison_loadtxt_tail_bench.py"),
            pyo3::ffi::c_str!("poison_loadtxt_tail_bench"),
        )
        .expect("loadtxt poison module");
        numpy
            .setattr("loadtxt", poison.getattr("fail").expect("poison callable"))
            .expect("install loadtxt poison");
        let native_out = candidate
            .call1((&path,))
            .expect("negative-tail path must not delegate");
        numpy
            .setattr("loadtxt", &numpy_loadtxt)
            .expect("restore numpy.loadtxt");
        assert_eq!(
            bytes_of(&native_out),
            bytes_of(&oracle),
            "native negative-tail path must retain NumPy bytes",
        );
        println!(
            "DISPATCH_PROOF row=python_loadtxt_negative_tail_8192x64_vs_numpy \
             fnp_callable=fnp_python.loadtxt \
             native_route=fnp_io::loadtxt_usecols_signed delegated_to_numpy=false"
        );

        let time_arm = |function: &Bound<'_, PyAny>| {
            let started = Instant::now();
            let output = function
                .call1((black_box(&path),))
                .expect("negative-tail median-CI arm");
            let elapsed = started.elapsed();
            let checksum = bytes_of(&output)
                .into_iter()
                .fold(0xcbf2_9ce4_8422_2325_u64, |state, byte| {
                    (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                });
            black_box(output);
            common::ContractObservation { elapsed, checksum }
        };
        let _ = common::run_median_ci_contract(
            "python_loadtxt_negative_tail_8192x64_vs_numpy",
            || time_arm(&incumbent),
            || time_arm(&candidate),
        );
    });
}

/// Isolates the two byte-producing shapes in `build_numpy_array_from_storage`'s
/// `ArrayStorage::Bool` materialization arm. This is a common funnel for owned
/// bool storage, but direct and passthrough bool-returning paths can bypass it.
///
/// The former arm collected a second full-size `Vec<u8>` from the `Vec<bool>`
/// before copying it into the NumPy buffer. Rust guarantees `bool` is size 1,
/// align 1, and only ever the bit patterns 0x00/0x01 — exactly what
/// `u8::from(b)` produced — so that collect walked every element to rebuild
/// bytes it already had. The 8,000,000-element measured configuration showed
/// allocation-sensitive variance, but this harness did not count allocator
/// calls or faults. Glibc's 128 KiB mmap threshold is only its initial/default
/// setting and may adapt, so it is not a universal boundary.
///
/// Both arms below perform the *identical* NumPy tail — `numpy.empty(n, uint8)`
/// then `PyBuffer::copy_from_slice` — so the only difference measured is the
/// redundant allocation and pass. This is not a replica of production logic;
/// it is production's exact two lines with a shared, real tail.
///
/// `run_median_ci_contract` asserts the two arms' output checksums are equal on
/// every round, so byte-identity is enforced continuously during timing rather
/// than once beforehand.
/// VS-INCUMBENT, end-to-end: `fnp.isin` against `numpy.isin` on f64.
///
/// This is a **missing-capability** surface, which is where our real margins
/// live. NumPy's `isin` has a fast `table` method that is integer-only; float
/// input falls back to a sort-based path. We do not share that path, so nothing
/// expensive is common to the two arms — the whole of our call is measured
/// against the whole of theirs.
///
/// All six fleet traps are guarded here, in order:
///
/// 1. DISPATCH — the incumbent's identity is asserted **at runtime inside this
///    binary**: the module is genuinely `numpy`, the callable's `__module__` is
///    under `numpy`, and it is not the same object as ours. franken_networkx
///    published a 2.6x whose baseline was already dispatched to their own code
///    while genuine NetworkX was 1.88x SLOWER.
/// 2. UNMATCHED CONFIG — both arms receive the identical two array objects; no
///    dtype, order, or option differs.
/// 3. NON-INTERLEAVED — `run_median_ci_contract` interleaves both arms inside one
///    measured routine with the order alternating per round.
/// 4. CORE CONTENTION — the contract establishes a base/base A/A null in the same
///    invocation before the effect. If that null is not near unity the window is
///    void regardless of the effect.
/// 5. CLIENT-BOUND — the timed region is the call itself, and both arms marshal
///    the same pre-built Python objects, so harness cost is common and small
///    relative to a 1M-element scan.
/// 6. SHARED COMPONENT — none. `fnp.isin` allocates and computes its own result;
///    `numpy.isin` allocates and computes its own. This is the trap the
///    bool-return arm fell into and the reason this arm exists.
fn bench_isin_f64_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module =
            PyModule::new(py, "fnp_python_isin_vs_numpy").expect("isin vs-numpy bench module");
        fnp_python(&module).expect("initialize fnp_python isin bench module");
        let numpy = py.import("numpy").expect("numpy oracle");

        // TRAP 1 — prove the incumbent is genuinely NumPy, at runtime, here.
        let numpy_name = numpy
            .getattr("__name__")
            .expect("numpy __name__")
            .extract::<String>()
            .expect("numpy __name__ str");
        assert_eq!(numpy_name, "numpy", "incumbent module is not numpy");
        let numpy_version = numpy
            .getattr("__version__")
            .expect("numpy __version__")
            .extract::<String>()
            .expect("numpy __version__ str");
        let np_isin = numpy.getattr("isin").expect("numpy isin");
        let fnp_isin = module.getattr("isin").expect("fnp isin");
        common::report_numpy_incumbent_identity(py, "isin", &np_isin);
        let incumbent_module = np_isin
            .getattr("__module__")
            .expect("numpy isin __module__")
            .extract::<String>()
            .expect("numpy isin __module__ str");
        assert!(
            incumbent_module.starts_with("numpy"),
            "incumbent isin is not defined under numpy: {incumbent_module}"
        );
        assert!(
            !np_isin.is(&fnp_isin),
            "dispatch trap: the incumbent arm resolved to our own callable"
        );
        println!(
            "INCUMBENT_IDENTITY arm=numpy.isin numpy.__version__={numpy_version} \
             callable_module={incumbent_module} dispatch_assert=passed"
        );
        common::report_incumbent_topology("fnp.isin", "numpy.isin");

        // TRAP 2 — one pair of inputs, handed to both arms unchanged.
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260727)\n\
                 isin_a = rng.standard_normal(1_000_000)\n\
                 isin_b = rng.standard_normal(1_000)\n\
                 isin_b[:200] = isin_a[:200]\n",
            )
            .expect("isin setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("isin corpus setup");
        let isin_a = namespace.get_item("isin_a").expect("isin_a present");
        let isin_b = namespace.get_item("isin_b").expect("isin_b present");

        let checksum_of = |array: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
            let bytes = array
                .call_method0("tobytes")
                .expect("tobytes")
                .extract::<Vec<u8>>()
                .expect("byte Vec");
            bytes
                .iter()
                .fold(0xcbf2_9ce4_8422_2325_u64, |state, &byte| {
                    (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                })
        };

        // Behavior before timing: our whole result must equal theirs, byte for
        // byte. `run_median_ci_contract` re-asserts this every round.
        let ours = fnp_isin.call1((&isin_a, &isin_b)).expect("fnp isin");
        let theirs = np_isin.call1((&isin_a, &isin_b)).expect("numpy isin");
        assert_eq!(
            checksum_of(&ours),
            checksum_of(&theirs),
            "fnp.isin and numpy.isin disagree — a perf comparison of divergent \
             results is meaningless",
        );

        let mut time_incumbent = || {
            let started = Instant::now();
            let result = np_isin
                .call1((black_box(&isin_a), black_box(&isin_b)))
                .expect("numpy isin arm");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: checksum_of(&result),
            }
        };
        let mut time_ours = || {
            let started = Instant::now();
            let result = fnp_isin
                .call1((black_box(&isin_a), black_box(&isin_b)))
                .expect("fnp isin arm");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: checksum_of(&result),
            }
        };

        // Base arm is the INCUMBENT, so the reported ratio reads
        // numpy-over-fnp: above 1.0 means we are faster.
        let _ = common::run_median_ci_contract(
            "python_isin_f64_1m_vs_numpy",
            &mut time_incumbent,
            &mut time_ours,
        );
    });
}

/// Convert the single most load-bearing unconverted number in the README:
/// `README.md:1868` publishes "`isin` hashed-set up to 530x (16M f64)" under a
/// heading that promises "Every ratio below is a measured vs-NumPy speedup".
/// The 530x row (ledger 2026-07-02) is a bare same-worker A/B: no A/A null, no
/// executing-ELF identity, no incumbent artifact identity, no observed threads,
/// no median-CI gate. The successor incumbent-class row measured 23.882236x at
/// 1,000,000 elements and explicitly disclaims generalizing to this shape, so
/// the README's headline number has never been measured under the corrected
/// contract. This arm measures the README's own regime, whichever way it falls.
fn bench_isin_f64_readme_16m_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    const ROW: &str = "python_isin_f64_16m_readme_headline_vs_numpy";
    const HAYSTACK: usize = 16_000_000;
    const NEEDLE: usize = 65_536;
    const PLANTED_HITS: usize = 16_384;
    // NumPy's sort-based f64 fallback runs in seconds at this size, so the
    // 41-round min-of-3 microbenchmark default would not fit the worker's
    // execution ceiling. Sampling stays odd, interleaved, and dual-null gated.
    const CONTRACT_ROUNDS: usize = 21;
    const CONTRACT_MIN_OF: usize = 1;
    const THREAD_ACTIVITY_REPETITIONS: usize = 3;
    const THREADS: &str = "4";
    const REQUIRED_BUILD_PROFILE: &str = "release-perf";

    assert_eq!(
        std::env::var("FNP_BENCH_PROFILE").as_deref(),
        Ok(REQUIRED_BUILD_PROFILE),
        "ship-grade isin evidence requires FNP_BENCH_PROFILE=release-perf"
    );
    let build_worker =
        std::env::var("FNP_BUILD_WORKER").expect("FNP_BUILD_WORKER records the RCH build origin");
    assert!(
        !build_worker.trim().is_empty(),
        "FNP_BUILD_WORKER must name the RCH build worker"
    );
    for variable in [
        "RAYON_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(THREADS),
            "{variable} must be explicitly pinned before isin timing"
        );
    }
    assert_eq!(
        rayon::current_num_threads(),
        THREADS
            .parse::<usize>()
            .expect("thread count constant is numeric"),
        "Rayon pool width does not match the pinned isin configuration"
    );
    let host = std::env::var("HOSTNAME").unwrap_or_else(|_| "unavailable".to_owned());
    println!(
        "WORKLOAD_RUNTIME workload=isin_f64_16m_readme_headline host={host} \
         build_worker={build_worker} build_profile={REQUIRED_BUILD_PROFILE} \
         rayon_threads={} RAYON_NUM_THREADS={} OPENBLAS_NUM_THREADS={} \
         OMP_NUM_THREADS={} MKL_NUM_THREADS={} contract_rounds={CONTRACT_ROUNDS} \
         contract_min_of={CONTRACT_MIN_OF}",
        rayon::current_num_threads(),
        THREADS,
        THREADS,
        THREADS,
        THREADS,
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_isin_readme_16m")
            .expect("isin README-conversion bench module");
        fnp_python(&module).expect("initialize fnp_python isin README bench module");
        let numpy = py.import("numpy").expect("numpy incumbent");

        let np_isin = numpy.getattr("isin").expect("numpy isin");
        let fnp_isin = module.getattr("isin").expect("fnp isin");
        common::report_numpy_incumbent_identity(py, "isin", &np_isin);
        let incumbent_module = np_isin
            .getattr("__module__")
            .expect("numpy isin __module__")
            .extract::<String>()
            .expect("numpy isin __module__ str");
        assert!(
            incumbent_module.starts_with("numpy"),
            "incumbent isin is not defined under numpy: {incumbent_module}"
        );
        assert!(
            !np_isin.is(&fnp_isin),
            "dispatch trap: the incumbent arm resolved to our own callable"
        );
        common::report_incumbent_topology("fnp.isin", "numpy.isin");

        // One pair of inputs, built once, handed unchanged to both arms. The
        // needle plants real hits: a needle with zero members is a degenerate
        // membership job and is not what a user runs.
        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260731)\n\
                 isin_a = rng.standard_normal(16_000_000)\n\
                 isin_b = rng.standard_normal(65_536)\n\
                 isin_b[:16_384] = isin_a[rng.choice(16_000_000, 16_384, replace=False)]\n",
            )
            .expect("isin README corpus CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("isin README corpus setup");
        let isin_a = namespace.get_item("isin_a").expect("isin_a present");
        let isin_b = namespace.get_item("isin_b").expect("isin_b present");

        for (name, array, expected) in
            [("haystack", &isin_a, HAYSTACK), ("needle", &isin_b, NEEDLE)]
        {
            let shape = array
                .getattr("shape")
                .expect("isin operand shape")
                .extract::<Vec<usize>>()
                .expect("isin operand shape vector");
            let dtype = array
                .getattr("dtype")
                .expect("isin operand dtype")
                .str()
                .expect("isin operand dtype string")
                .to_string();
            let c_contiguous = array
                .getattr("flags")
                .expect("isin operand flags")
                .getattr("c_contiguous")
                .expect("isin operand C-contiguous flag")
                .extract::<bool>()
                .expect("isin operand C-contiguous bool");
            assert_eq!(shape, vec![expected], "{name} shape");
            assert_eq!(dtype, "float64", "{name} dtype");
            assert!(c_contiguous, "{name} must be C-contiguous");
        }

        let checksum_of = |array: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
            let bytes = array
                .call_method0("tobytes")
                .expect("tobytes")
                .extract::<Vec<u8>>()
                .expect("byte Vec");
            bytes
                .len()
                .to_le_bytes()
                .iter()
                .chain(bytes.iter())
                .fold(0xcbf2_9ce4_8422_2325_u64, |state, &byte| {
                    (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                })
        };

        // Fail closed on route attribution before timing. The measured
        // C-contiguous f64 regime must survive a poisoned NumPy entry point;
        // otherwise the candidate and incumbent would share the expensive
        // membership implementation.
        let route_namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "def poison_isin(*args, **kwargs):\n\
                 \x20   raise AssertionError('fnp.isin delegated to numpy.isin')\n",
            )
            .expect("isin route-attribution CString")
            .as_c_str(),
            Some(&route_namespace),
            Some(&route_namespace),
        )
        .expect("isin route-attribution helper");
        let poison_isin = route_namespace
            .get_item("poison_isin")
            .expect("poison_isin present");
        numpy
            .setattr("isin", &poison_isin)
            .expect("poison numpy.isin");
        let ours = fnp_isin
            .call1((&isin_a, &isin_b))
            .expect("native fnp.isin under poisoned numpy.isin");
        numpy.setattr("isin", &np_isin).expect("restore numpy.isin");

        // Behavior before timing: the whole boolean result must be byte-equal.
        let theirs = np_isin.call1((&isin_a, &isin_b)).expect("numpy isin");
        let ours_bytes = ours
            .call_method0("tobytes")
            .expect("fnp isin tobytes")
            .extract::<Vec<u8>>()
            .expect("fnp isin byte Vec");
        let theirs_bytes = theirs
            .call_method0("tobytes")
            .expect("numpy isin tobytes")
            .extract::<Vec<u8>>()
            .expect("numpy isin byte Vec");
        assert_eq!(
            ours_bytes, theirs_bytes,
            "fnp.isin and numpy.isin disagree - a perf comparison of divergent \
             results is meaningless"
        );
        let ours_dtype = ours
            .getattr("dtype")
            .expect("fnp isin dtype")
            .str()
            .expect("fnp isin dtype string")
            .to_string();
        let theirs_dtype = theirs
            .getattr("dtype")
            .expect("numpy isin dtype")
            .str()
            .expect("numpy isin dtype string")
            .to_string();
        assert_eq!(ours_dtype, theirs_dtype, "isin result dtype differs");
        let observed_hits = theirs_bytes.iter().filter(|byte| **byte != 0).count();
        assert!(
            observed_hits >= PLANTED_HITS,
            "needle planting failed: {observed_hits} members observed"
        );
        println!(
            "PARITY row={ROW} exact_bytes=passed result_dtype={ours_dtype} \
             result_elements={} member_elements={observed_hits} checksum={:016x}",
            theirs_bytes.len(),
            checksum_of(&theirs)
        );
        println!(
            "ROUTE_PRECONDITIONS row={ROW} haystack_dtype=float64 \
             haystack_elements={HAYSTACK} needle_dtype=float64 \
             needle_elements={NEEDLE} planted_members={PLANTED_HITS} \
             both_c_contiguous=true assume_unique=default invert=default \
             candidate_route=try_zerocopy_float_isin \
             readme_claim=README.md:1868_isin_hashed_set_up_to_530x_16M_f64"
        );
        println!(
            "ROUTE_PROOF row={ROW} numpy_isin_poison_survived=true \
             candidate_route=try_zerocopy_float_isin \
             candidate_stages_all_native=true shared_timed_component=none"
        );
        println!(
            "WORK_ACCOUNTING row={ROW} candidate_public_calls=1 \
             incumbent_public_calls=1 candidate_haystack_elements={HAYSTACK} \
             incumbent_haystack_elements={HAYSTACK} \
             candidate_needle_elements={NEEDLE} incumbent_needle_elements={NEEDLE} \
             candidate_output_elements={HAYSTACK} incumbent_output_elements={HAYSTACK} \
             shared_inputs=true equal_work=true"
        );
        println!(
            "COUNTED_MECHANISM row={ROW} class=one_fewer_algorithmic_class \
             incumbent_algorithm=sort_of_concatenated_haystack_and_needle \
             incumbent_sorted_elements={} \
             candidate_algorithm=hash_set_build_plus_membership_probe \
             candidate_sorted_elements=0 \
             incumbent_reason=numpy_kind_table_admits_integer_and_boolean_only \
             incumbent_source=legacy_numpy_code/numpy/numpy/lib/_arraysetops_impl.py:in1d",
            HAYSTACK + NEEDLE
        );

        let run_incumbent = || {
            np_isin
                .call1((black_box(&isin_a), black_box(&isin_b)))
                .expect("numpy isin arm")
        };
        let run_candidate = || {
            fnp_isin
                .call1((black_box(&isin_a), black_box(&isin_b)))
                .expect("fnp isin arm")
        };

        common::report_observed_thread_activity(ROW, "numpy", THREAD_ACTIVITY_REPETITIONS, || {
            black_box(checksum_of(&run_incumbent()));
        });
        common::report_observed_thread_activity(ROW, "fnp", THREAD_ACTIVITY_REPETITIONS, || {
            black_box(checksum_of(&run_candidate()));
        });

        let mut observe_incumbent = || {
            let started = Instant::now();
            let result = run_incumbent();
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: checksum_of(&result),
            }
        };
        let mut observe_candidate = || {
            let started = Instant::now();
            let result = run_candidate();
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: checksum_of(&result),
            }
        };
        let (effect, incumbent_null, candidate_null) =
            common::run_dual_null_median_ci_contract_with_sampling(
                ROW,
                &mut observe_incumbent,
                &mut observe_candidate,
                CONTRACT_ROUNDS,
                CONTRACT_MIN_OF,
            );
        let verdict = common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
        println!(
            "README_CLAIM_CONVERSION row={ROW} verdict={verdict} \
             claimed_readme_ratio=530 claimed_readme_source=README.md:1868 \
             claimed_ledger_heading=2026-07-02_float_isin_530x \
             incumbent_median_ms={:.6} candidate_median_ms={:.6} \
             ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
             incumbent_null_ratio={:.6} incumbent_null_ci95=[{:.6},{:.6}] \
             candidate_null_ratio={:.6} candidate_null_ci95=[{:.6},{:.6}] \
             effect_ci_excludes_one={} corrected_dual_null_gate=true \
             median_clause=true actual_threads_reported=true \
             incumbent=numpy_live_same_invocation",
            effect.arm_a_median_ns / 1_000_000.0,
            effect.arm_b_median_ns / 1_000_000.0,
            effect.ratio_median,
            effect.ratio_ci_low,
            effect.ratio_ci_high,
            incumbent_null.ratio_median,
            incumbent_null.ratio_ci_low,
            incumbent_null.ratio_ci_high,
            candidate_null.ratio_median,
            candidate_null.ratio_ci_low,
            candidate_null.ratio_ci_high,
            effect.ratio_ci_low > 1.0 || effect.ratio_ci_high < 1.0,
        );
        let (decision, reason) = match verdict {
            "DECIDABLE_WIN" => ("choose_fnp", "corrected_dual_null_incumbent_win"),
            "DECIDABLE_REGRESSION" => ("choose_numpy", "corrected_dual_null_regression"),
            _ => (
                "choose_neither_on_performance",
                "effect_not_separated_from_dual_null_envelope",
            ),
        };
        println!(
            "CHOOSER_STATEMENT workload=isin_f64_16m_readme_headline \
             decision={decision} reason={reason} \
             incumbent=numpy_live_same_invocation \
             measured_scope=float64_16m_haystack_65536_needle_16384_planted_members \
             shared_timed_component=none \
             outside_scope=run_same_contract_before_choosing"
        );
    });
}

/// FUSED ELEMENT-WISE CHAIN, whole job vs live NumPy.
///
/// The job is "compute `a * b + c` over three large `float64` arrays". NumPy's
/// only public way to do it is two ufunc calls that materialize a full
/// intermediate array; ours is one fused parallel pass. This is a structural
/// gap, not a tuning gap: NumPy cannot fuse an expression without a JIT.
fn bench_fused_multiply_add_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    const ROW: &str = "python_fused_multiply_add_f64_8m_vs_numpy";
    const ELEMENTS: usize = 8_000_000;
    const CONTRACT_ROUNDS: usize = 41;
    const CONTRACT_MIN_OF: usize = 3;
    const THREAD_ACTIVITY_REPETITIONS: usize = 3;
    const REQUIRED_BUILD_PROFILE: &str = "release-perf";

    assert_eq!(
        std::env::var("FNP_BENCH_PROFILE").as_deref(),
        Ok(REQUIRED_BUILD_PROFILE),
        "ship-grade fusion evidence requires FNP_BENCH_PROFILE=release-perf"
    );
    let build_worker =
        std::env::var("FNP_BUILD_WORKER").expect("FNP_BUILD_WORKER records the RCH build origin");
    assert!(
        !build_worker.trim().is_empty(),
        "FNP_BUILD_WORKER must be set"
    );
    // The pool width is a MEASURED VARIABLE for this lever, not a fixed 4: the
    // whole point is that NumPy is single-threaded here while we are not, so the
    // row must be reproducible at any pinned width. Every pool is still pinned
    // to the SAME explicit value and the actual Rayon width is asserted, so a
    // silently-defaulted pool can never be mistaken for a pinned one.
    let threads = std::env::var("RAYON_NUM_THREADS")
        .expect("RAYON_NUM_THREADS must be explicitly pinned before fusion timing");
    for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(threads.as_str()),
            "{variable} must be pinned to the same width as RAYON_NUM_THREADS"
        );
    }
    let threads: usize = threads.parse().expect("thread count is numeric");
    assert!(threads >= 1, "pinned thread count must be positive");
    assert_eq!(
        rayon::current_num_threads(),
        threads,
        "Rayon pool width does not match the pinned fusion configuration"
    );
    println!(
        "WORKLOAD_RUNTIME workload=fused_multiply_add build_worker={build_worker} \
         build_profile={REQUIRED_BUILD_PROFILE} pinned_threads={threads} \
         rayon_threads={} contract_rounds={CONTRACT_ROUNDS} \
         contract_min_of={CONTRACT_MIN_OF}",
        rayon::current_num_threads()
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_fused_multiply_add")
            .expect("fused multiply-add bench module");
        fnp_python(&module).expect("initialize fnp_python fused multiply-add module");
        let numpy = py.import("numpy").expect("numpy incumbent");

        let np_multiply = numpy.getattr("multiply").expect("numpy multiply");
        let np_add = numpy.getattr("add").expect("numpy add");
        let fnp_multiply_add = module.getattr("multiply_add").expect("fnp multiply_add");
        common::report_numpy_incumbent_identity(py, "multiply", &np_multiply);
        for (name, callable) in [("multiply", &np_multiply), ("add", &np_add)] {
            let owner = callable
                .getattr("__module__")
                .ok()
                .and_then(|value| value.extract::<String>().ok())
                .unwrap_or_else(|| "numpy".to_owned());
            assert!(
                owner.starts_with("numpy"),
                "incumbent {name} is not defined under numpy: {owner}"
            );
            assert!(
                !callable.is(&fnp_multiply_add),
                "dispatch trap: incumbent {name} resolved to the candidate callable"
            );
        }
        common::report_incumbent_topology("fnp.multiply_add", "numpy.multiply+numpy.add");

        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260731)\n\
                 fma_a = rng.standard_normal(8_000_000)\n\
                 fma_b = rng.standard_normal(8_000_000)\n\
                 fma_c = rng.standard_normal(8_000_000)\n",
            )
            .expect("fused multiply-add corpus CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("fused multiply-add corpus setup");
        let fma_a = namespace.get_item("fma_a").expect("fma_a present");
        let fma_b = namespace.get_item("fma_b").expect("fma_b present");
        let fma_c = namespace.get_item("fma_c").expect("fma_c present");

        let checksum_of = |array: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
            let bytes = array
                .call_method0("tobytes")
                .expect("tobytes")
                .extract::<Vec<u8>>()
                .expect("byte Vec");
            bytes
                .len()
                .to_le_bytes()
                .iter()
                .chain(bytes.iter())
                .fold(0xcbf2_9ce4_8422_2325_u64, |state, &byte| {
                    (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                })
        };

        // The incumbent arm is exactly what a NumPy user writes for this job.
        let run_incumbent = || {
            let product = np_multiply
                .call1((black_box(&fma_a), black_box(&fma_b)))
                .expect("numpy multiply arm");
            np_add
                .call1((product, black_box(&fma_c)))
                .expect("numpy add arm")
        };
        let run_candidate = || {
            fnp_multiply_add
                .call1((black_box(&fma_a), black_box(&fma_b), black_box(&fma_c)))
                .expect("fnp multiply_add arm")
        };

        // BIT-EXACTNESS before timing: fusion must not change a single bit.
        let ours = run_candidate();
        let theirs = run_incumbent();
        let ours_bytes = ours
            .call_method0("tobytes")
            .expect("fnp tobytes")
            .extract::<Vec<u8>>()
            .expect("fnp byte Vec");
        let theirs_bytes = theirs
            .call_method0("tobytes")
            .expect("numpy tobytes")
            .extract::<Vec<u8>>()
            .expect("numpy byte Vec");
        assert_eq!(
            ours_bytes, theirs_bytes,
            "fused multiply_add is not byte-identical to numpy multiply-then-add"
        );
        let ours_dtype = ours
            .getattr("dtype")
            .expect("fnp dtype")
            .str()
            .expect("fnp dtype str")
            .to_string();
        let theirs_dtype = theirs
            .getattr("dtype")
            .expect("numpy dtype")
            .str()
            .expect("numpy dtype str")
            .to_string();
        assert_eq!(ours_dtype, theirs_dtype, "result dtype differs");
        println!(
            "PARITY row={ROW} exact_bytes=passed result_dtype={ours_dtype} \
             result_elements={ELEMENTS} checksum={:016x} \
             fma_contraction=absent_two_roundings_match_numpy",
            checksum_of(&theirs)
        );
        println!(
            "ROUTE_PRECONDITIONS row={ROW} dtype=float64 elements={ELEMENTS} \
             operands=3 equal_shapes=true all_c_contiguous=true \
             candidate_route=zerocopy_multiply_add_typed \
             parallel_min=65536 chunk_elements=16384"
        );
        println!(
            "WORK_ACCOUNTING row={ROW} candidate_public_calls=1 incumbent_public_calls=2 \
             candidate_elements={ELEMENTS} incumbent_elements={ELEMENTS} \
             candidate_output_elements={ELEMENTS} incumbent_output_elements={ELEMENTS} \
             candidate_intermediate_arrays=0 incumbent_intermediate_arrays=1 \
             candidate_intermediate_bytes=0 incumbent_intermediate_bytes={} \
             shared_inputs=true equal_work=true",
            ELEMENTS * 8
        );
        println!(
            "COUNTED_MECHANISM row={ROW} class=materialization_and_pass_elimination \
             incumbent_element_stream_touches=6 candidate_element_stream_touches=4 \
             incumbent_output_allocations=2 candidate_output_allocations=1 \
             incumbent_eliminated_temporary_bytes={} \
             incumbent_reason=numpy_evaluates_each_ufunc_separately_and_cannot_fuse_without_a_jit",
            ELEMENTS * 8
        );

        common::report_observed_thread_activity(ROW, "numpy", THREAD_ACTIVITY_REPETITIONS, || {
            black_box(checksum_of(&run_incumbent()));
        });
        common::report_observed_thread_activity(ROW, "fnp", THREAD_ACTIVITY_REPETITIONS, || {
            black_box(checksum_of(&run_candidate()));
        });

        let mut observe_incumbent = || {
            let started = Instant::now();
            let result = run_incumbent();
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: checksum_of(&result),
            }
        };
        let mut observe_candidate = || {
            let started = Instant::now();
            let result = run_candidate();
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: checksum_of(&result),
            }
        };
        let (effect, incumbent_null, candidate_null) =
            common::run_dual_null_median_ci_contract_with_sampling(
                ROW,
                &mut observe_incumbent,
                &mut observe_candidate,
                CONTRACT_ROUNDS,
                CONTRACT_MIN_OF,
            );
        let verdict = common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
        println!(
            "FUSION_RESULT row={ROW} verdict={verdict} \
             incumbent_median_ms={:.6} candidate_median_ms={:.6} \
             ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
             incumbent_null_ratio={:.6} incumbent_null_ci95=[{:.6},{:.6}] \
             candidate_null_ratio={:.6} candidate_null_ci95=[{:.6},{:.6}] \
             effect_ci_excludes_one={} corrected_dual_null_gate=true \
             median_clause=true actual_threads_reported=true \
             incumbent=numpy_live_same_invocation",
            effect.arm_a_median_ns / 1_000_000.0,
            effect.arm_b_median_ns / 1_000_000.0,
            effect.ratio_median,
            effect.ratio_ci_low,
            effect.ratio_ci_high,
            incumbent_null.ratio_median,
            incumbent_null.ratio_ci_low,
            incumbent_null.ratio_ci_high,
            candidate_null.ratio_median,
            candidate_null.ratio_ci_low,
            candidate_null.ratio_ci_high,
            effect.ratio_ci_low > 1.0 || effect.ratio_ci_high < 1.0,
        );
        let (decision, reason) = match verdict {
            "DECIDABLE_WIN" => ("choose_fnp", "corrected_dual_null_incumbent_win"),
            "DECIDABLE_REGRESSION" => ("choose_numpy", "corrected_dual_null_regression"),
            _ => (
                "choose_numpy",
                "effect_not_separated_from_dual_null_envelope",
            ),
        };
        println!(
            "CHOOSER_STATEMENT workload=fused_multiply_add_f64_8m \
             decision={decision} reason={reason} \
             incumbent=numpy_live_same_invocation \
             measured_scope=float64_8m_three_equal_shape_c_contiguous_operands_at_{threads}_pinned_threads \
             outside_scope=run_same_contract_before_choosing"
        );
    });
}

/// Two-product element-wise fusion, broad dtype matrix vs live NumPy.
///
/// NumPy evaluates `a * b + c * d` as two multiply ufuncs followed by add,
/// materializing both full-size products. The candidate performs the same
/// operations and roundings while streaming four inputs into one output in a
/// single cache-banded parallel pass.
fn bench_fused_pairwise_multiply_add_matrix_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    const TARGET_BYTES_PER_OPERAND: usize = 32 * 1024 * 1024;
    const CONTRACT_ROUNDS: usize = 21;
    const CONTRACT_MIN_OF: usize = 1;
    const REQUIRED_BUILD_PROFILE: &str = "release-perf";

    assert_eq!(
        std::env::var("FNP_BENCH_PROFILE").as_deref(),
        Ok(REQUIRED_BUILD_PROFILE),
        "ship-grade fusion evidence requires FNP_BENCH_PROFILE=release-perf"
    );
    let build_worker =
        std::env::var("FNP_BUILD_WORKER").expect("FNP_BUILD_WORKER records the RCH build origin");
    assert!(
        !build_worker.trim().is_empty(),
        "FNP_BUILD_WORKER must be set"
    );
    let threads = std::env::var("RAYON_NUM_THREADS")
        .expect("RAYON_NUM_THREADS must be explicitly pinned before fusion timing");
    for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(threads.as_str()),
            "{variable} must match RAYON_NUM_THREADS"
        );
    }
    let threads: usize = threads.parse().expect("thread count is numeric");
    assert_eq!(
        rayon::current_num_threads(),
        threads,
        "Rayon pool width does not match the pinned fusion configuration"
    );
    println!(
        "WORKLOAD_RUNTIME workload=fused_pairwise_multiply_add_matrix \
         build_worker={build_worker} build_profile={REQUIRED_BUILD_PROFILE} \
         pinned_threads={threads} rayon_threads={} contract_rounds={CONTRACT_ROUNDS} \
         contract_min_of={CONTRACT_MIN_OF} bytes_per_operand={TARGET_BYTES_PER_OPERAND}",
        rayon::current_num_threads()
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_fused_pairwise_multiply_add")
            .expect("pairwise fusion bench module");
        fnp_python(&module).expect("initialize fnp_python pairwise fusion module");
        let numpy = py.import("numpy").expect("numpy incumbent");

        let np_multiply = numpy.getattr("multiply").expect("numpy multiply");
        let np_add = numpy.getattr("add").expect("numpy add");
        let fnp_chain = module
            .getattr("pairwise_multiply_add")
            .expect("fnp pairwise_multiply_add");
        common::report_numpy_incumbent_identity(py, "multiply", &np_multiply);
        for (name, callable) in [("multiply", &np_multiply), ("add", &np_add)] {
            let owner = callable
                .getattr("__module__")
                .ok()
                .and_then(|value| value.extract::<String>().ok())
                .unwrap_or_else(|| "numpy".to_owned());
            assert!(
                owner.starts_with("numpy"),
                "incumbent {name} is not defined under numpy: {owner}"
            );
            assert!(
                !callable.is(&fnp_chain),
                "dispatch trap: incumbent {name} resolved to the candidate callable"
            );
        }
        common::report_incumbent_topology(
            "fnp.pairwise_multiply_add",
            "numpy.multiply+numpy.multiply+numpy.add",
        );

        let checksum_of = |array: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
            let bytes = array
                .call_method0("tobytes")
                .expect("tobytes")
                .extract::<Vec<u8>>()
                .expect("byte Vec");
            bytes
                .len()
                .to_le_bytes()
                .iter()
                .chain(bytes.iter())
                .fold(0xcbf2_9ce4_8422_2325_u64, |state, &byte| {
                    (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                })
        };

        let cases = [
            ("float64", 8_usize, false),
            ("float32", 4, false),
            ("int8", 1, true),
            ("uint8", 1, true),
            ("int16", 2, true),
            ("uint16", 2, true),
            ("int32", 4, true),
            ("uint32", 4, true),
            ("int64", 8, true),
            ("uint64", 8, true),
        ];
        for (case_index, (dtype, item_size, is_integer)) in cases.into_iter().enumerate() {
            let elements = TARGET_BYTES_PER_OPERAND / item_size;
            let namespace = PyDict::new(py);
            let setup = if is_integer {
                format!(
                    "import numpy as np\n\
                     rng = np.random.default_rng({})\n\
                     dt = np.dtype('{}')\n\
                     n = {}\n\
                     pair_a = np.frombuffer(rng.bytes(n * dt.itemsize), dtype=dt).copy()\n\
                     pair_b = np.frombuffer(rng.bytes(n * dt.itemsize), dtype=dt).copy()\n\
                     pair_c = np.frombuffer(rng.bytes(n * dt.itemsize), dtype=dt).copy()\n\
                     pair_d = np.frombuffer(rng.bytes(n * dt.itemsize), dtype=dt).copy()\n",
                    20260802 + case_index,
                    dtype,
                    elements,
                )
            } else {
                format!(
                    "import numpy as np\n\
                     rng = np.random.default_rng({})\n\
                     dt = np.dtype('{}')\n\
                     n = {}\n\
                     pair_a = rng.standard_normal(n).astype(dt)\n\
                     pair_b = rng.standard_normal(n).astype(dt)\n\
                     pair_c = rng.standard_normal(n).astype(dt)\n\
                     pair_d = rng.standard_normal(n).astype(dt)\n",
                    20260802 + case_index,
                    dtype,
                    elements,
                )
            };
            py.run(
                std::ffi::CString::new(setup)
                    .expect("pairwise fusion corpus CString")
                    .as_c_str(),
                Some(&namespace),
                Some(&namespace),
            )
            .expect("pairwise fusion corpus setup");
            let a = namespace.get_item("pair_a").expect("pair_a present");
            let b = namespace.get_item("pair_b").expect("pair_b present");
            let c = namespace.get_item("pair_c").expect("pair_c present");
            let d = namespace.get_item("pair_d").expect("pair_d present");
            let row = format!("python_fused_pairwise_multiply_add_{dtype}_32mib_vs_numpy");

            let run_incumbent = || {
                let first_product = np_multiply
                    .call1((black_box(&a), black_box(&b)))
                    .expect("first numpy multiply arm");
                let second_product = np_multiply
                    .call1((black_box(&c), black_box(&d)))
                    .expect("second numpy multiply arm");
                np_add
                    .call1((first_product, second_product))
                    .expect("numpy add arm")
            };
            let run_candidate = || {
                fnp_chain
                    .call1((black_box(&a), black_box(&b), black_box(&c), black_box(&d)))
                    .expect("fnp pairwise_multiply_add arm")
            };

            let ours = run_candidate();
            let theirs = run_incumbent();
            assert_eq!(
                ours.call_method0("tobytes")
                    .expect("fnp tobytes")
                    .extract::<Vec<u8>>()
                    .expect("fnp bytes"),
                theirs
                    .call_method0("tobytes")
                    .expect("numpy tobytes")
                    .extract::<Vec<u8>>()
                    .expect("numpy bytes"),
                "pairwise fused {dtype} result is not byte-identical to NumPy"
            );
            assert_eq!(
                ours.getattr("dtype")
                    .expect("fnp dtype")
                    .str()
                    .expect("fnp dtype str")
                    .to_string(),
                dtype,
                "pairwise fused dtype drifted"
            );
            println!(
                "PARITY row={row} exact_bytes=passed result_dtype={dtype} \
                 result_elements={elements} result_bytes={TARGET_BYTES_PER_OPERAND} \
                 checksum={:016x} operation_order=multiply_multiply_add",
                checksum_of(&theirs)
            );
            println!(
                "ROUTE_PRECONDITIONS row={row} dtype={dtype} elements={elements} \
                 operands=4 equal_shapes=true all_c_contiguous=true \
                 candidate_route=zerocopy_pairwise_multiply_add_typed \
                 parallel_min=65536 chunk_elements=8192"
            );
            println!(
                "COUNTED_MECHANISM row={row} class=materialization_and_pass_elimination \
                 incumbent_element_stream_touches=9 candidate_element_stream_touches=5 \
                 incumbent_output_allocations=3 candidate_output_allocations=1 \
                 incumbent_eliminated_temporary_bytes={} candidate_intermediate_bytes=0 \
                 arithmetic_ops_per_element=3 shared_inputs=true equal_work=true",
                TARGET_BYTES_PER_OPERAND * 2
            );

            common::report_observed_thread_activity(&row, "numpy", 1, || {
                black_box(checksum_of(&run_incumbent()));
            });
            common::report_observed_thread_activity(&row, "fnp", 1, || {
                black_box(checksum_of(&run_candidate()));
            });

            let mut observe_incumbent = || {
                let started = Instant::now();
                let result = run_incumbent();
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let mut observe_candidate = || {
                let started = Instant::now();
                let result = run_candidate();
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract_with_sampling(
                    &row,
                    &mut observe_incumbent,
                    &mut observe_candidate,
                    CONTRACT_ROUNDS,
                    CONTRACT_MIN_OF,
                );
            let verdict =
                common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
            println!(
                "FUSION_PAIR_RESULT row={row} dtype={dtype} verdict={verdict} \
                 incumbent_median_ms={:.6} candidate_median_ms={:.6} \
                 ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
                 incumbent_null_ratio={:.6} incumbent_null_ci95=[{:.6},{:.6}] \
                 candidate_null_ratio={:.6} candidate_null_ci95=[{:.6},{:.6}] \
                 corrected_dual_null_gate=true incumbent=numpy_live_same_invocation",
                effect.arm_a_median_ns / 1_000_000.0,
                effect.arm_b_median_ns / 1_000_000.0,
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                incumbent_null.ratio_median,
                incumbent_null.ratio_ci_low,
                incumbent_null.ratio_ci_high,
                candidate_null.ratio_median,
                candidate_null.ratio_ci_low,
                candidate_null.ratio_ci_high,
            );
            let decision = if verdict == "DECIDABLE_WIN" {
                "choose_fnp"
            } else {
                "choose_numpy"
            };
            println!(
                "CHOOSER_STATEMENT workload=fused_pairwise_multiply_add_{dtype}_32mib \
                 decision={decision} verdict={verdict} incumbent=numpy_live_same_invocation \
                 measured_scope={elements}_equal_shape_c_contiguous_elements_at_{threads}_pinned_threads \
                 outside_scope=run_same_contract_before_choosing"
            );
        }
    });
}

/// LONGER FUSED ELEMENT-WISE CHAIN, broad dtype matrix vs live NumPy.
///
/// NumPy evaluates `(a - b) * c + d` as three independent binary ufuncs,
/// materializing two full-size intermediates. The candidate performs the same
/// three arithmetic operations, in the same order, while streaming four inputs
/// into one output in a single cache-banded parallel pass.
fn bench_fused_subtract_multiply_add_matrix_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    const TARGET_BYTES_PER_OPERAND: usize = 32 * 1024 * 1024;
    const CONTRACT_ROUNDS: usize = 21;
    const CONTRACT_MIN_OF: usize = 1;
    const REQUIRED_BUILD_PROFILE: &str = "release-perf";

    assert_eq!(
        std::env::var("FNP_BENCH_PROFILE").as_deref(),
        Ok(REQUIRED_BUILD_PROFILE),
        "ship-grade fusion evidence requires FNP_BENCH_PROFILE=release-perf"
    );
    let build_worker =
        std::env::var("FNP_BUILD_WORKER").expect("FNP_BUILD_WORKER records the RCH build origin");
    assert!(
        !build_worker.trim().is_empty(),
        "FNP_BUILD_WORKER must be set"
    );
    let threads = std::env::var("RAYON_NUM_THREADS")
        .expect("RAYON_NUM_THREADS must be explicitly pinned before fusion timing");
    for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(threads.as_str()),
            "{variable} must match RAYON_NUM_THREADS"
        );
    }
    let threads: usize = threads.parse().expect("thread count is numeric");
    assert_eq!(
        rayon::current_num_threads(),
        threads,
        "Rayon pool width does not match the pinned fusion configuration"
    );
    println!(
        "WORKLOAD_RUNTIME workload=fused_subtract_multiply_add_matrix \
         build_worker={build_worker} build_profile={REQUIRED_BUILD_PROFILE} \
         pinned_threads={threads} rayon_threads={} contract_rounds={CONTRACT_ROUNDS} \
         contract_min_of={CONTRACT_MIN_OF} bytes_per_operand={TARGET_BYTES_PER_OPERAND}",
        rayon::current_num_threads()
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_fused_subtract_multiply_add")
            .expect("long fusion bench module");
        fnp_python(&module).expect("initialize fnp_python long fusion module");
        let numpy = py.import("numpy").expect("numpy incumbent");

        let np_subtract = numpy.getattr("subtract").expect("numpy subtract");
        let np_multiply = numpy.getattr("multiply").expect("numpy multiply");
        let np_add = numpy.getattr("add").expect("numpy add");
        let fnp_chain = module
            .getattr("subtract_multiply_add")
            .expect("fnp subtract_multiply_add");
        common::report_numpy_incumbent_identity(py, "subtract", &np_subtract);
        for (name, callable) in [
            ("subtract", &np_subtract),
            ("multiply", &np_multiply),
            ("add", &np_add),
        ] {
            let owner = callable
                .getattr("__module__")
                .ok()
                .and_then(|value| value.extract::<String>().ok())
                .unwrap_or_else(|| "numpy".to_owned());
            assert!(
                owner.starts_with("numpy"),
                "incumbent {name} is not defined under numpy: {owner}"
            );
            assert!(
                !callable.is(&fnp_chain),
                "dispatch trap: incumbent {name} resolved to the candidate callable"
            );
        }
        common::report_incumbent_topology(
            "fnp.subtract_multiply_add",
            "numpy.subtract+numpy.multiply+numpy.add",
        );

        let checksum_of = |array: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
            let bytes = array
                .call_method0("tobytes")
                .expect("tobytes")
                .extract::<Vec<u8>>()
                .expect("byte Vec");
            bytes
                .len()
                .to_le_bytes()
                .iter()
                .chain(bytes.iter())
                .fold(0xcbf2_9ce4_8422_2325_u64, |state, &byte| {
                    (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                })
        };

        let cases = [
            ("float64", 8_usize, false),
            ("float32", 4, false),
            ("int8", 1, true),
            ("uint8", 1, true),
            ("int16", 2, true),
            ("uint16", 2, true),
            ("int32", 4, true),
            ("uint32", 4, true),
            ("int64", 8, true),
            ("uint64", 8, true),
        ];
        for (case_index, (dtype, item_size, is_integer)) in cases.into_iter().enumerate() {
            let elements = TARGET_BYTES_PER_OPERAND / item_size;
            let namespace = PyDict::new(py);
            let setup = if is_integer {
                format!(
                    "import numpy as np\n\
                     rng = np.random.default_rng({})\n\
                     dt = np.dtype('{}')\n\
                     n = {}\n\
                     chain_a = np.frombuffer(rng.bytes(n * dt.itemsize), dtype=dt).copy()\n\
                     chain_b = np.frombuffer(rng.bytes(n * dt.itemsize), dtype=dt).copy()\n\
                     chain_c = np.frombuffer(rng.bytes(n * dt.itemsize), dtype=dt).copy()\n\
                     chain_d = np.frombuffer(rng.bytes(n * dt.itemsize), dtype=dt).copy()\n",
                    20260731 + case_index,
                    dtype,
                    elements,
                )
            } else {
                format!(
                    "import numpy as np\n\
                     rng = np.random.default_rng({})\n\
                     dt = np.dtype('{}')\n\
                     n = {}\n\
                     chain_a = rng.standard_normal(n).astype(dt)\n\
                     chain_b = rng.standard_normal(n).astype(dt)\n\
                     chain_c = rng.standard_normal(n).astype(dt)\n\
                     chain_d = rng.standard_normal(n).astype(dt)\n",
                    20260731 + case_index,
                    dtype,
                    elements,
                )
            };
            py.run(
                std::ffi::CString::new(setup)
                    .expect("long fusion corpus CString")
                    .as_c_str(),
                Some(&namespace),
                Some(&namespace),
            )
            .expect("long fusion corpus setup");
            let a = namespace.get_item("chain_a").expect("chain_a present");
            let b = namespace.get_item("chain_b").expect("chain_b present");
            let c = namespace.get_item("chain_c").expect("chain_c present");
            let d = namespace.get_item("chain_d").expect("chain_d present");
            let row = format!("python_fused_subtract_multiply_add_{dtype}_32mib_vs_numpy");

            let run_incumbent = || {
                let difference = np_subtract
                    .call1((black_box(&a), black_box(&b)))
                    .expect("numpy subtract arm");
                let product = np_multiply
                    .call1((difference, black_box(&c)))
                    .expect("numpy multiply arm");
                np_add
                    .call1((product, black_box(&d)))
                    .expect("numpy add arm")
            };
            let run_candidate = || {
                fnp_chain
                    .call1((black_box(&a), black_box(&b), black_box(&c), black_box(&d)))
                    .expect("fnp subtract_multiply_add arm")
            };

            let ours = run_candidate();
            let theirs = run_incumbent();
            assert_eq!(
                ours.call_method0("tobytes")
                    .expect("fnp tobytes")
                    .extract::<Vec<u8>>()
                    .expect("fnp bytes"),
                theirs
                    .call_method0("tobytes")
                    .expect("numpy tobytes")
                    .extract::<Vec<u8>>()
                    .expect("numpy bytes"),
                "long fused {dtype} result is not byte-identical to NumPy"
            );
            assert_eq!(
                ours.getattr("dtype")
                    .expect("fnp dtype")
                    .str()
                    .expect("fnp dtype str")
                    .to_string(),
                dtype,
                "long fused dtype drifted"
            );
            println!(
                "PARITY row={row} exact_bytes=passed result_dtype={dtype} \
                 result_elements={elements} result_bytes={TARGET_BYTES_PER_OPERAND} \
                 checksum={:016x} operation_order=subtract_multiply_add",
                checksum_of(&theirs)
            );
            println!(
                "ROUTE_PRECONDITIONS row={row} dtype={dtype} elements={elements} \
                 operands=4 equal_shapes=true all_c_contiguous=true \
                 candidate_route=zerocopy_subtract_multiply_add_typed \
                 parallel_min=65536 chunk_elements=8192"
            );
            println!(
                "COUNTED_MECHANISM row={row} class=materialization_and_pass_elimination \
                 incumbent_element_stream_touches=9 candidate_element_stream_touches=5 \
                 incumbent_output_allocations=3 candidate_output_allocations=1 \
                 incumbent_eliminated_temporary_bytes={} candidate_intermediate_bytes=0 \
                 arithmetic_ops_per_element=3 shared_inputs=true equal_work=true",
                TARGET_BYTES_PER_OPERAND * 2
            );

            common::report_observed_thread_activity(&row, "numpy", 1, || {
                black_box(checksum_of(&run_incumbent()));
            });
            common::report_observed_thread_activity(&row, "fnp", 1, || {
                black_box(checksum_of(&run_candidate()));
            });

            let mut observe_incumbent = || {
                let started = Instant::now();
                let result = run_incumbent();
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let mut observe_candidate = || {
                let started = Instant::now();
                let result = run_candidate();
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract_with_sampling(
                    &row,
                    &mut observe_incumbent,
                    &mut observe_candidate,
                    CONTRACT_ROUNDS,
                    CONTRACT_MIN_OF,
                );
            let verdict =
                common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
            println!(
                "FUSION_CHAIN_RESULT row={row} dtype={dtype} verdict={verdict} \
                 incumbent_median_ms={:.6} candidate_median_ms={:.6} \
                 ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
                 incumbent_null_ratio={:.6} incumbent_null_ci95=[{:.6},{:.6}] \
                 candidate_null_ratio={:.6} candidate_null_ci95=[{:.6},{:.6}] \
                 corrected_dual_null_gate=true incumbent=numpy_live_same_invocation",
                effect.arm_a_median_ns / 1_000_000.0,
                effect.arm_b_median_ns / 1_000_000.0,
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                incumbent_null.ratio_median,
                incumbent_null.ratio_ci_low,
                incumbent_null.ratio_ci_high,
                candidate_null.ratio_median,
                candidate_null.ratio_ci_low,
                candidate_null.ratio_ci_high,
            );
            let decision = if verdict == "DECIDABLE_WIN" {
                "choose_fnp"
            } else {
                "choose_numpy"
            };
            println!(
                "CHOOSER_STATEMENT workload=fused_subtract_multiply_add_{dtype}_32mib \
                 decision={decision} verdict={verdict} incumbent=numpy_live_same_invocation \
                 measured_scope={elements}_equal_shape_c_contiguous_elements_at_{threads}_pinned_threads \
                 outside_scope=run_same_contract_before_choosing"
            );
        }
    });
}

/// Parallel worst-case fixed-width truth reductions against live NumPy.
///
/// Each row forces a complete 64 MiB scan: `any` sees only zeros and `all` sees
/// only ones. The candidate first checks one 8 KiB prefix for the common
/// early-exit case, then distributes independent 1 MiB bands across the pinned
/// Rayon pool. Truth OR/AND are associative, so scheduling cannot change the
/// scalar result. `FNP_TRUTH_REDUCTION_DTYPES=numeric` selects every native
/// integer width plus float32/float64; a single NumPy dtype name selects one row
/// family, and the default remains bool for a cheap smoke run.
fn bench_flat_all_any_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    const TARGET_INPUT_BYTES: usize = 64 * 1024 * 1024;
    const PREFIX_BYTES: usize = 8192;
    const PARALLEL_CHUNK_BYTES: usize = 1 << 20;
    const CONTRACT_ROUNDS: usize = 21;
    const CONTRACT_MIN_OF: usize = 1;
    const INCUMBENT_THREAD_ACTIVITY_REPETITIONS: usize = 16;
    const CANDIDATE_THREAD_ACTIVITY_REPETITIONS: usize = 128;
    const REQUIRED_BUILD_PROFILE: &str = "release-perf";

    let dtype_mode =
        std::env::var("FNP_TRUTH_REDUCTION_DTYPES").unwrap_or_else(|_| "bool".to_owned());
    let dtype_names = if dtype_mode == "numeric" {
        vec![
            "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float32",
            "float64",
        ]
    } else {
        assert!(
            [
                "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
                "float32", "float64",
            ]
            .contains(&dtype_mode.as_str()),
            "FNP_TRUTH_REDUCTION_DTYPES must be numeric or one native fixed-width dtype"
        );
        vec![dtype_mode.as_str()]
    };

    assert_eq!(
        std::env::var("FNP_BENCH_PROFILE").as_deref(),
        Ok(REQUIRED_BUILD_PROFILE),
        "ship-grade truth-reduction evidence requires FNP_BENCH_PROFILE=release-perf"
    );
    let build_worker =
        std::env::var("FNP_BUILD_WORKER").expect("FNP_BUILD_WORKER records the build origin");
    assert!(
        !build_worker.trim().is_empty(),
        "FNP_BUILD_WORKER must be set"
    );
    let threads = std::env::var("RAYON_NUM_THREADS")
        .expect("RAYON_NUM_THREADS must be explicitly pinned before truth-reduction timing");
    for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok("1"),
            "{variable} must be one: neither truth-reduction arm calls BLAS"
        );
    }
    let threads: usize = threads.parse().expect("thread count is numeric");
    assert_eq!(
        rayon::current_num_threads(),
        threads,
        "Rayon pool width does not match the pinned truth-reduction configuration"
    );
    println!(
        "WORKLOAD_RUNTIME workload=fixed_width_flat_all_any_matrix dtype_mode={dtype_mode} \
         build_worker={build_worker} build_profile={REQUIRED_BUILD_PROFILE} \
         pinned_threads={threads} rayon_threads={} contract_rounds={CONTRACT_ROUNDS} \
         contract_min_of={CONTRACT_MIN_OF} input_bytes={TARGET_INPUT_BYTES} dtype_count={}",
        rayon::current_num_threads(),
        dtype_names.len()
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_fixed_width_flat_all_any_matrix")
            .expect("fixed-width flat-reduction bench module");
        fnp_python(&module).expect("initialize fixed-width flat-reduction module");
        let numpy = py.import("numpy").expect("numpy incumbent");
        common::report_incumbent_topology("fnp.all/fnp.any", "numpy.all/numpy.any");
        println!("NUMPY_BUILD_CONFIG_BEGIN workload=fixed_width_flat_all_any_matrix");
        numpy
            .getattr("show_config")
            .expect("numpy.show_config")
            .call0()
            .expect("report NumPy build configuration");
        println!("NUMPY_BUILD_CONFIG_END workload=fixed_width_flat_all_any_matrix");
        println!(
            "BLAS_RELEVANCE workload=fixed_width_flat_all_any_matrix \
             numpy_reduction_uses_blas=false candidate_uses_blas=false \
             blas_threads_pinned=1 reason=native_truth_ufunc_reduction"
        );

        let checksum_of = |scalar: &Bound<'_, PyAny>| -> u64 {
            let dtype = scalar
                .getattr("dtype")
                .expect("bool reduction scalar dtype")
                .str()
                .expect("bool reduction scalar dtype string")
                .to_string();
            let bytes = scalar
                .call_method0("tobytes")
                .expect("bool reduction scalar tobytes")
                .extract::<Vec<u8>>()
                .expect("bool reduction scalar byte Vec");
            dtype
                .as_bytes()
                .iter()
                .chain(bytes.iter())
                .fold(0xcbf2_9ce4_8422_2325_u64, |state, &byte| {
                    (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                })
        };

        let cases = [("any_false", "any", "zeros"), ("all_true", "all", "ones")];
        for dtype in dtype_names {
            let numpy_dtype_name = if dtype == "bool" { "bool_" } else { dtype };
            let dtype_object = numpy
                .getattr(numpy_dtype_name)
                .expect("native fixed-width NumPy dtype");
            let dtype_spec = numpy
                .call_method1("dtype", (&dtype_object,))
                .expect("normalize NumPy dtype");
            let item_bytes = dtype_spec
                .getattr("itemsize")
                .expect("dtype itemsize")
                .extract::<usize>()
                .expect("numeric dtype itemsize");
            assert_eq!(
                TARGET_INPUT_BYTES % item_bytes,
                0,
                "target byte count must divide the native dtype width"
            );
            let elements = TARGET_INPUT_BYTES / item_bytes;

            for &(case, operation, constructor) in &cases {
                let row = format!("python_{dtype}_flat_{case}_64mib_vs_numpy");
                let np_reduction = numpy
                    .getattr(operation)
                    .expect("NumPy fixed-width truth reduction");
                let fnp_reduction = module
                    .getattr(operation)
                    .expect("FrankenNumPy fixed-width truth reduction");
                assert!(
                    !fnp_reduction.is(&np_reduction),
                    "dispatch trap: fnp.{operation} resolved to the NumPy callable"
                );
                common::report_numpy_incumbent_identity(py, operation, &np_reduction);

                let kwargs = PyDict::new(py);
                kwargs
                    .set_item("dtype", &dtype_object)
                    .expect("set fixed-width input dtype");
                let input = numpy
                    .call_method(constructor, (elements,), Some(&kwargs))
                    .expect("construct worst-case fixed-width truth input");
                assert_eq!(
                    input
                        .getattr("nbytes")
                        .expect("truth input nbytes")
                        .extract::<usize>()
                        .expect("truth input nbytes value"),
                    TARGET_INPUT_BYTES
                );
                assert!(
                    input
                        .getattr("flags")
                        .expect("truth input flags")
                        .getattr("c_contiguous")
                        .expect("truth input C-contiguous flag")
                        .extract::<bool>()
                        .expect("truth input C-contiguous value")
                );

                let run_incumbent = || {
                    np_reduction
                        .call1((black_box(&input),))
                        .expect("NumPy fixed-width flat truth-reduction arm")
                };
                let run_candidate = || {
                    fnp_reduction
                        .call1((black_box(&input),))
                        .expect("FrankenNumPy fixed-width flat truth-reduction arm")
                };

                let ours = run_candidate();
                let theirs = run_incumbent();
                assert!(
                    ours.get_type().is(theirs.get_type()),
                    "{dtype} flat-{operation} scalar type differs from NumPy"
                );
                assert_eq!(
                    ours.call_method0("tobytes")
                        .expect("FrankenNumPy truth reduction tobytes")
                        .extract::<Vec<u8>>()
                        .expect("FrankenNumPy truth reduction bytes"),
                    theirs
                        .call_method0("tobytes")
                        .expect("NumPy truth reduction tobytes")
                        .extract::<Vec<u8>>()
                        .expect("NumPy truth reduction bytes"),
                    "{dtype} flat-{operation} is not byte-exact"
                );
                println!(
                    "PARITY row={row} exact_bytes=passed exact_scalar_type=passed \
                     input_dtype={dtype} result_dtype=bool input_elements={elements} \
                     input_bytes={TARGET_INPUT_BYTES} corpus={case} checksum={:016x}",
                    checksum_of(&theirs)
                );
                let candidate_route = if dtype == "bool" {
                    if operation == "any" {
                        "block_any_u8"
                    } else {
                        "block_all_u8"
                    }
                } else {
                    "block_any_all_native_numeric"
                };
                println!(
                    "ROUTE_PRECONDITIONS row={row} axis=none dtype={dtype} exact_ndarray=true \
                     c_contiguous=true input_bytes={TARGET_INPUT_BYTES} parallel_min_bytes=16777216 \
                     prefix_bytes={PREFIX_BYTES} candidate_route={candidate_route}"
                );
                println!(
                    "COUNTED_MECHANISM row={row} class=parallel_associative_truth_reduction \
                     incumbent_input_sweeps=1 candidate_input_sweeps=1 \
                     incumbent_input_bytes={TARGET_INPUT_BYTES} candidate_input_bytes={TARGET_INPUT_BYTES} \
                     incumbent_expected_threads=1 candidate_pinned_threads={threads} \
                     candidate_prefix_bytes={PREFIX_BYTES} candidate_parallel_chunk_bytes={PARALLEL_CHUNK_BYTES} \
                     candidate_parallel_chunks={} truth_operation_associative=true \
                     candidate_intermediate_bool_buffer=false shared_input=true",
                    (TARGET_INPUT_BYTES - PREFIX_BYTES).div_ceil(PARALLEL_CHUNK_BYTES)
                );

                common::report_observed_thread_activity(
                    &row,
                    "numpy",
                    INCUMBENT_THREAD_ACTIVITY_REPETITIONS,
                    || {
                        black_box(checksum_of(&run_incumbent()));
                    },
                );
                common::report_observed_thread_activity(
                    &row,
                    "fnp",
                    CANDIDATE_THREAD_ACTIVITY_REPETITIONS,
                    || {
                        black_box(checksum_of(&run_candidate()));
                    },
                );

                let mut observe_incumbent = || {
                    let started = Instant::now();
                    let result = run_incumbent();
                    let elapsed = started.elapsed();
                    common::ContractObservation {
                        elapsed,
                        checksum: checksum_of(&result),
                    }
                };
                let mut observe_candidate = || {
                    let started = Instant::now();
                    let result = run_candidate();
                    let elapsed = started.elapsed();
                    common::ContractObservation {
                        elapsed,
                        checksum: checksum_of(&result),
                    }
                };
                let (effect, incumbent_null, candidate_null) =
                    common::run_dual_null_median_ci_contract_with_sampling(
                        &row,
                        &mut observe_incumbent,
                        &mut observe_candidate,
                        CONTRACT_ROUNDS,
                        CONTRACT_MIN_OF,
                    );
                let verdict =
                    common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
                println!(
                    "TRUTH_REDUCTION_RESULT row={row} dtype={dtype} operation={operation} \
                     corpus={case} verdict={verdict} \
                     incumbent_median_ms={:.6} candidate_median_ms={:.6} \
                     ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
                     incumbent_null_ratio={:.6} incumbent_null_ci95=[{:.6},{:.6}] \
                     candidate_null_ratio={:.6} candidate_null_ci95=[{:.6},{:.6}] \
                     corrected_dual_null_gate=true incumbent=numpy_live_same_invocation",
                    effect.arm_a_median_ns / 1_000_000.0,
                    effect.arm_b_median_ns / 1_000_000.0,
                    effect.ratio_median,
                    effect.ratio_ci_low,
                    effect.ratio_ci_high,
                    incumbent_null.ratio_median,
                    incumbent_null.ratio_ci_low,
                    incumbent_null.ratio_ci_high,
                    candidate_null.ratio_median,
                    candidate_null.ratio_ci_low,
                    candidate_null.ratio_ci_high,
                );
                let decision = if verdict == "DECIDABLE_WIN" {
                    "choose_fnp"
                } else {
                    "choose_numpy"
                };
                println!(
                    "CHOOSER_STATEMENT workload={dtype}_flat_{case}_64mib \
                     decision={decision} verdict={verdict} incumbent=numpy_live_same_invocation \
                     measured_scope={elements}_c_contiguous_{dtype}_elements_{TARGET_INPUT_BYTES}_bytes_at_{threads}_pinned_threads \
                     outside_scope=run_same_contract_before_choosing"
                );
            }
        }
    });
}

/// Exact-tree parallel float32/float64 flat sums against live NumPy.
///
/// Both arms reduce the same 64 MiB C-contiguous input with NumPy's exact
/// pairwise arithmetic tree.  The incumbent evaluates that tree on one thread;
/// the candidate schedules independent 65,536-element subtrees over the pinned
/// Rayon pool and preserves every parent combine edge.
fn bench_float_flat_sum_matrix_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    const TARGET_INPUT_BYTES: usize = 64 * 1024 * 1024;
    const PARALLEL_LEAF_ELEMENTS: usize = 1 << 16;
    const CONTRACT_ROUNDS: usize = 21;
    const CONTRACT_MIN_OF: usize = 1;
    const THREAD_ACTIVITY_REPETITIONS: usize = 64;
    const REQUIRED_BUILD_PROFILE: &str = "release-perf";

    let operation = std::env::var("FNP_FLOAT_FLAT_REDUCTION").unwrap_or_else(|_| "sum".to_owned());
    assert!(
        operation == "sum" || operation == "mean",
        "FNP_FLOAT_FLAT_REDUCTION must be sum or mean"
    );

    assert_eq!(
        std::env::var("FNP_BENCH_PROFILE").as_deref(),
        Ok(REQUIRED_BUILD_PROFILE),
        "ship-grade float-reduction evidence requires FNP_BENCH_PROFILE=release-perf"
    );
    let build_worker =
        std::env::var("FNP_BUILD_WORKER").expect("FNP_BUILD_WORKER records the build origin");
    assert!(
        !build_worker.trim().is_empty(),
        "FNP_BUILD_WORKER must be set"
    );
    let threads = std::env::var("RAYON_NUM_THREADS")
        .expect("RAYON_NUM_THREADS must be explicitly pinned before float-reduction timing");
    for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok("1"),
            "{variable} must be one: neither reduction arm calls BLAS"
        );
    }
    let threads: usize = threads.parse().expect("thread count is numeric");
    assert_eq!(
        rayon::current_num_threads(),
        threads,
        "Rayon pool width does not match the pinned float-reduction configuration"
    );
    println!(
        "WORKLOAD_RUNTIME workload=float_flat_{operation}_matrix operation={operation} \
         build_worker={build_worker} build_profile={REQUIRED_BUILD_PROFILE} \
         pinned_threads={threads} rayon_threads={} contract_rounds={CONTRACT_ROUNDS} \
         contract_min_of={CONTRACT_MIN_OF} input_bytes={TARGET_INPUT_BYTES}",
        rayon::current_num_threads()
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_float_flat_reduction_matrix")
            .expect("float flat-reduction bench module");
        fnp_python(&module).expect("initialize float flat-reduction module");
        let numpy = py.import("numpy").expect("numpy incumbent");
        let np_reduction = numpy
            .getattr(operation.as_str())
            .expect("numpy float reduction");
        let fnp_reduction = module
            .getattr(operation.as_str())
            .expect("fnp float reduction");
        assert!(
            !fnp_reduction.is(&np_reduction),
            "dispatch trap: fnp.{operation} resolved to the NumPy callable"
        );
        common::report_numpy_incumbent_identity(py, operation.as_str(), &np_reduction);
        common::report_incumbent_topology(
            &format!("fnp.{operation}"),
            &format!("numpy.{operation}"),
        );
        println!("NUMPY_BUILD_CONFIG_BEGIN workload=float_flat_{operation}_matrix");
        numpy
            .getattr("show_config")
            .expect("numpy.show_config")
            .call0()
            .expect("report NumPy build configuration");
        println!("NUMPY_BUILD_CONFIG_END workload=float_flat_{operation}_matrix");
        println!(
            "BLAS_RELEVANCE workload=float_flat_{operation}_matrix numpy_reduction_uses_blas=false \
             candidate_uses_blas=false blas_threads_pinned=1 reason=floating_ufunc_reduction"
        );

        let checksum_of = |scalar: &Bound<'_, PyAny>| -> u64 {
            let dtype = scalar
                .getattr("dtype")
                .expect("reduction scalar dtype")
                .str()
                .expect("reduction scalar dtype string")
                .to_string();
            let bytes = scalar
                .call_method0("tobytes")
                .expect("reduction scalar tobytes")
                .extract::<Vec<u8>>()
                .expect("reduction scalar byte Vec");
            dtype
                .as_bytes()
                .iter()
                .chain(bytes.iter())
                .fold(0xcbf2_9ce4_8422_2325_u64, |state, &byte| {
                    (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                })
        };

        let cases = [("float32", 4_usize), ("float64", 8_usize)];
        for (case_index, (dtype, item_size)) in cases.into_iter().enumerate() {
            let elements = TARGET_INPUT_BYTES / item_size;
            let exact_tree_parallel_leaves = elements.div_ceil(PARALLEL_LEAF_ELEMENTS);
            let row = format!("python_float_flat_{operation}_{dtype}_64mib_vs_numpy");
            let namespace = PyDict::new(py);
            let setup = format!(
                "import numpy as np\n\
                 rng = np.random.default_rng({})\n\
                 float_reduction_input = rng.standard_normal({elements}, dtype=np.{dtype})\n\
                 assert float_reduction_input.nbytes == {TARGET_INPUT_BYTES}\n\
                 assert float_reduction_input.flags.c_contiguous\n\
                 assert float_reduction_input.dtype.isnative\n\
                 assert np.isfinite(float_reduction_input).all()\n",
                20260801 + case_index,
            );
            py.run(
                std::ffi::CString::new(setup)
                    .expect("float flat-reduction corpus CString")
                    .as_c_str(),
                Some(&namespace),
                Some(&namespace),
            )
            .expect("float flat-reduction corpus setup");
            let input = namespace
                .get_item("float_reduction_input")
                .expect("float reduction input present");

            let run_incumbent = || {
                np_reduction
                    .call1((black_box(&input),))
                    .expect("numpy float flat reduction arm")
            };
            let run_candidate = || {
                fnp_reduction
                    .call1((black_box(&input),))
                    .expect("fnp float flat reduction arm")
            };

            let ours = run_candidate();
            let theirs = run_incumbent();
            assert!(
                ours.get_type().is(theirs.get_type()),
                "float flat-{operation} {dtype} scalar type differs from NumPy"
            );
            assert_eq!(
                ours.getattr("dtype")
                    .expect("fnp reduction dtype")
                    .str()
                    .expect("fnp reduction dtype string")
                    .to_string(),
                theirs
                    .getattr("dtype")
                    .expect("numpy reduction dtype")
                    .str()
                    .expect("numpy reduction dtype string")
                    .to_string(),
                "candidate dtype differs from NumPy"
            );
            assert_eq!(
                ours.call_method0("tobytes")
                    .expect("fnp reduction tobytes")
                    .extract::<Vec<u8>>()
                    .expect("fnp reduction bytes"),
                theirs
                    .call_method0("tobytes")
                    .expect("numpy reduction tobytes")
                    .extract::<Vec<u8>>()
                    .expect("numpy reduction bytes"),
                "float flat-{operation} {dtype} is not bit-exact"
            );
            println!(
                "PARITY row={row} exact_bytes=passed exact_scalar_type=passed \
                 input_dtype={dtype} result_dtype={dtype} input_elements={elements} \
                 input_bytes={TARGET_INPUT_BYTES} checksum={:016x}",
                checksum_of(&theirs)
            );
            println!(
                "ROUTE_PRECONDITIONS row={row} axis=none dtype={dtype} item_size={item_size} \
                 exact_ndarray=true c_contiguous=true native_endian=true finite=true \
                 input_bytes={TARGET_INPUT_BYTES} parallel_min_bytes=16777216 \
                 candidate_route=try_zerocopy_float_{operation}_flat"
            );
            println!(
                "COUNTED_MECHANISM row={row} class=parallel_exact_pairwise_tree \
                 incumbent_input_sweeps=1 candidate_input_sweeps=1 \
                 incumbent_input_bytes={TARGET_INPUT_BYTES} candidate_input_bytes={TARGET_INPUT_BYTES} \
                 incumbent_arithmetic_add_edges={} candidate_arithmetic_add_edges={} \
                 pairwise_base_elements=128 candidate_parallel_leaf_elements={PARALLEL_LEAF_ELEMENTS} \
                 candidate_parallel_leaves={exact_tree_parallel_leaves} arithmetic_tree_identical=true \
                 incumbent_expected_threads=1 candidate_pinned_threads={threads} \
                 incumbent_output_scalars=1 candidate_output_scalars=1 shared_input=true",
                elements - 1,
                elements - 1,
            );

            common::report_observed_thread_activity(
                &row,
                "numpy",
                THREAD_ACTIVITY_REPETITIONS,
                || {
                    black_box(checksum_of(&run_incumbent()));
                },
            );
            common::report_observed_thread_activity(
                &row,
                "fnp",
                THREAD_ACTIVITY_REPETITIONS,
                || {
                    black_box(checksum_of(&run_candidate()));
                },
            );

            let mut observe_incumbent = || {
                let started = Instant::now();
                let result = run_incumbent();
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let mut observe_candidate = || {
                let started = Instant::now();
                let result = run_candidate();
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract_with_sampling(
                    &row,
                    &mut observe_incumbent,
                    &mut observe_candidate,
                    CONTRACT_ROUNDS,
                    CONTRACT_MIN_OF,
                );
            let verdict =
                common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
            println!(
                "FLOAT_REDUCTION_RESULT row={row} operation={operation} dtype={dtype} verdict={verdict} \
                 incumbent_median_ms={:.6} candidate_median_ms={:.6} \
                 ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
                 incumbent_null_ratio={:.6} incumbent_null_ci95=[{:.6},{:.6}] \
                 candidate_null_ratio={:.6} candidate_null_ci95=[{:.6},{:.6}] \
                 corrected_dual_null_gate=true incumbent=numpy_live_same_invocation",
                effect.arm_a_median_ns / 1_000_000.0,
                effect.arm_b_median_ns / 1_000_000.0,
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                incumbent_null.ratio_median,
                incumbent_null.ratio_ci_low,
                incumbent_null.ratio_ci_high,
                candidate_null.ratio_median,
                candidate_null.ratio_ci_low,
                candidate_null.ratio_ci_high,
            );
            let decision = if verdict == "DECIDABLE_WIN" {
                "choose_fnp"
            } else {
                "choose_numpy"
            };
            println!(
                "CHOOSER_STATEMENT workload=float_flat_{operation}_{dtype}_64mib \
                 decision={decision} verdict={verdict} incumbent=numpy_live_same_invocation \
                 measured_scope={elements}_c_contiguous_native_endian_elements_at_{threads}_pinned_threads \
                 outside_scope=run_same_contract_before_choosing"
            );
        }
    });
}

/// Broad fixed-width integer flat-sum matrix against live NumPy.
///
/// Each row reduces the same 64 MiB C-contiguous input to the exact promoted
/// scalar. The candidate distributes cache-sized wrapping-add bands across the
/// pinned Rayon pool; NumPy's integer ufunc reduction remains single-threaded.
fn bench_integer_flat_sum_matrix_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    const TARGET_INPUT_BYTES: usize = 64 * 1024 * 1024;
    const CACHE_BAND_BYTES: usize = 256 * 1024;
    const CONTRACT_ROUNDS: usize = 21;
    const CONTRACT_MIN_OF: usize = 1;
    const THREAD_ACTIVITY_REPETITIONS: usize = 64;
    const REQUIRED_BUILD_PROFILE: &str = "release-perf";

    assert_eq!(
        std::env::var("FNP_BENCH_PROFILE").as_deref(),
        Ok(REQUIRED_BUILD_PROFILE),
        "ship-grade integer-sum evidence requires FNP_BENCH_PROFILE=release-perf"
    );
    let build_worker =
        std::env::var("FNP_BUILD_WORKER").expect("FNP_BUILD_WORKER records the build origin");
    assert!(
        !build_worker.trim().is_empty(),
        "FNP_BUILD_WORKER must be set"
    );
    let threads = std::env::var("RAYON_NUM_THREADS")
        .expect("RAYON_NUM_THREADS must be explicitly pinned before integer-sum timing");
    for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(threads.as_str()),
            "{variable} must match RAYON_NUM_THREADS"
        );
    }
    let threads: usize = threads.parse().expect("thread count is numeric");
    assert_eq!(
        rayon::current_num_threads(),
        threads,
        "Rayon pool width does not match the pinned integer-sum configuration"
    );
    println!(
        "WORKLOAD_RUNTIME workload=integer_flat_sum_matrix \
         build_worker={build_worker} build_profile={REQUIRED_BUILD_PROFILE} \
         pinned_threads={threads} rayon_threads={} contract_rounds={CONTRACT_ROUNDS} \
         contract_min_of={CONTRACT_MIN_OF} input_bytes={TARGET_INPUT_BYTES}",
        rayon::current_num_threads()
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_integer_flat_sum_matrix")
            .expect("integer flat-sum bench module");
        fnp_python(&module).expect("initialize integer flat-sum module");
        let numpy = py.import("numpy").expect("numpy incumbent");
        let np_sum = numpy.getattr("sum").expect("numpy sum");
        let fnp_sum = module.getattr("sum").expect("fnp sum");
        assert!(
            !fnp_sum.is(&np_sum),
            "dispatch trap: fnp.sum resolved to the NumPy callable"
        );
        common::report_numpy_incumbent_identity(py, "sum", &np_sum);
        common::report_incumbent_topology("fnp.sum", "numpy.sum");
        println!("NUMPY_BUILD_CONFIG_BEGIN workload=integer_flat_sum_matrix");
        numpy
            .getattr("show_config")
            .expect("numpy.show_config")
            .call0()
            .expect("report NumPy build configuration");
        println!("NUMPY_BUILD_CONFIG_END workload=integer_flat_sum_matrix");
        println!(
            "BLAS_RELEVANCE workload=integer_flat_sum_matrix numpy_sum_uses_blas=false \
             candidate_uses_blas=false reason=fixed_width_integer_ufunc_reduction"
        );

        let checksum_of = |scalar: &Bound<'_, PyAny>| -> u64 {
            let dtype = scalar
                .getattr("dtype")
                .expect("sum scalar dtype")
                .str()
                .expect("sum scalar dtype string")
                .to_string();
            let bytes = scalar
                .call_method0("tobytes")
                .expect("sum scalar tobytes")
                .extract::<Vec<u8>>()
                .expect("sum scalar byte Vec");
            dtype
                .as_bytes()
                .iter()
                .chain(bytes.iter())
                .fold(0xcbf2_9ce4_8422_2325_u64, |state, &byte| {
                    (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                })
        };

        let cases = [
            ("int8", 1_usize, "int64"),
            ("uint8", 1, "uint64"),
            ("int16", 2, "int64"),
            ("uint16", 2, "uint64"),
            ("int32", 4, "int64"),
            ("uint32", 4, "uint64"),
            ("int64", 8, "int64"),
            ("uint64", 8, "uint64"),
        ];
        let selected_dtypes = std::env::var("FNP_INTEGER_SUM_DTYPES").ok().map(|spec| {
            spec.split(',')
                .map(str::trim)
                .filter(|dtype| !dtype.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>()
        });
        println!(
            "INTEGER_SUM_DTYPE_FILTER selected={}",
            selected_dtypes
                .as_ref()
                .map_or_else(|| "all".to_owned(), |dtypes| dtypes.join(":"))
        );

        for (case_index, (dtype, item_size, result_dtype)) in cases.into_iter().enumerate() {
            if selected_dtypes
                .as_ref()
                .is_some_and(|selected| !selected.iter().any(|candidate| candidate == dtype))
            {
                continue;
            }
            let elements = TARGET_INPUT_BYTES / item_size;
            let cache_bands = TARGET_INPUT_BYTES.div_ceil(CACHE_BAND_BYTES);
            let row = format!("python_integer_flat_sum_{dtype}_64mib_vs_numpy");
            let namespace = PyDict::new(py);
            let setup = format!(
                "import numpy as np\n\
                 rng = np.random.default_rng({})\n\
                 dt = np.dtype('{}')\n\
                 integer_sum_input = np.frombuffer(\
                     rng.bytes({TARGET_INPUT_BYTES}), dtype=dt).copy()\n\
                 assert integer_sum_input.size == {elements}\n\
                 assert integer_sum_input.flags.c_contiguous\n\
                 assert integer_sum_input.dtype.isnative\n",
                20260742 + case_index,
                dtype,
            );
            py.run(
                std::ffi::CString::new(setup)
                    .expect("integer flat-sum corpus CString")
                    .as_c_str(),
                Some(&namespace),
                Some(&namespace),
            )
            .expect("integer flat-sum corpus setup");
            let input = namespace
                .get_item("integer_sum_input")
                .expect("integer sum input present");

            let run_incumbent = || {
                np_sum
                    .call1((black_box(&input),))
                    .expect("numpy integer flat sum arm")
            };
            let run_candidate = || {
                fnp_sum
                    .call1((black_box(&input),))
                    .expect("fnp integer flat sum arm")
            };

            let ours = run_candidate();
            let theirs = run_incumbent();
            assert!(
                ours.get_type().is(theirs.get_type()),
                "integer flat-sum {dtype} scalar type differs from NumPy"
            );
            let ours_dtype = ours
                .getattr("dtype")
                .expect("fnp sum dtype")
                .str()
                .expect("fnp sum dtype string")
                .to_string();
            let theirs_dtype = theirs
                .getattr("dtype")
                .expect("numpy sum dtype")
                .str()
                .expect("numpy sum dtype string")
                .to_string();
            assert_eq!(ours_dtype, result_dtype, "candidate promotion drifted");
            assert_eq!(
                ours_dtype, theirs_dtype,
                "candidate dtype differs from NumPy"
            );
            assert_eq!(
                ours.call_method0("tobytes")
                    .expect("fnp sum tobytes")
                    .extract::<Vec<u8>>()
                    .expect("fnp sum bytes"),
                theirs
                    .call_method0("tobytes")
                    .expect("numpy sum tobytes")
                    .extract::<Vec<u8>>()
                    .expect("numpy sum bytes"),
                "integer flat-sum {dtype} is not bit-exact"
            );
            println!(
                "PARITY row={row} exact_bytes=passed exact_scalar_type=passed \
                 input_dtype={dtype} result_dtype={result_dtype} input_elements={elements} \
                 input_bytes={TARGET_INPUT_BYTES} checksum={:016x}",
                checksum_of(&theirs)
            );
            println!(
                "ROUTE_PRECONDITIONS row={row} axis=none dtype={dtype} item_size={item_size} \
                 exact_ndarray=true c_contiguous=true native_endian=true \
                 input_bytes={TARGET_INPUT_BYTES} parallel_min_bytes=8388608 \
                 candidate_route=try_zerocopy_integer_sum_flat"
            );
            println!(
                "COUNTED_MECHANISM row={row} class=parallel_cache_banded_reduction \
                 incumbent_input_sweeps=1 candidate_input_sweeps=1 \
                 incumbent_input_bytes={TARGET_INPUT_BYTES} candidate_input_bytes={TARGET_INPUT_BYTES} \
                 incumbent_element_additions={} candidate_block_element_additions={elements} \
                 candidate_partial_combine_additions=scheduler_dependent \
                 candidate_cache_band_bytes={CACHE_BAND_BYTES} candidate_cache_bands={cache_bands} \
                 incumbent_expected_threads=1 candidate_pinned_threads={threads} \
                 incumbent_output_scalars=1 candidate_output_scalars=1 shared_input=true",
                elements - 1,
            );

            common::report_observed_thread_activity(
                &row,
                "numpy",
                THREAD_ACTIVITY_REPETITIONS,
                || {
                    black_box(checksum_of(&run_incumbent()));
                },
            );
            common::report_observed_thread_activity(
                &row,
                "fnp",
                THREAD_ACTIVITY_REPETITIONS,
                || {
                    black_box(checksum_of(&run_candidate()));
                },
            );

            let mut observe_incumbent = || {
                let started = Instant::now();
                let result = run_incumbent();
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let mut observe_candidate = || {
                let started = Instant::now();
                let result = run_candidate();
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract_with_sampling(
                    &row,
                    &mut observe_incumbent,
                    &mut observe_candidate,
                    CONTRACT_ROUNDS,
                    CONTRACT_MIN_OF,
                );
            let verdict =
                common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
            println!(
                "INTEGER_SUM_RESULT row={row} dtype={dtype} verdict={verdict} \
                 incumbent_median_ms={:.6} candidate_median_ms={:.6} \
                 ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
                 incumbent_null_ratio={:.6} incumbent_null_ci95=[{:.6},{:.6}] \
                 candidate_null_ratio={:.6} candidate_null_ci95=[{:.6},{:.6}] \
                 corrected_dual_null_gate=true incumbent=numpy_live_same_invocation",
                effect.arm_a_median_ns / 1_000_000.0,
                effect.arm_b_median_ns / 1_000_000.0,
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                incumbent_null.ratio_median,
                incumbent_null.ratio_ci_low,
                incumbent_null.ratio_ci_high,
                candidate_null.ratio_median,
                candidate_null.ratio_ci_low,
                candidate_null.ratio_ci_high,
            );
            let decision = if verdict == "DECIDABLE_WIN" {
                "choose_fnp"
            } else {
                "choose_numpy"
            };
            println!(
                "CHOOSER_STATEMENT workload=integer_flat_sum_{dtype}_64mib \
                 decision={decision} verdict={verdict} incumbent=numpy_live_same_invocation \
                 measured_scope={elements}_c_contiguous_native_endian_elements_at_{threads}_pinned_threads \
                 outside_scope=run_same_contract_before_choosing"
            );
        }
    });
}

/// PREALLOCATED LONG-CHAIN FUSION, broad dtype matrix vs live NumPy.
///
/// Both arms receive caller-owned output arrays, so neither allocates inside
/// the timed job. NumPy still has to stream that output through subtract,
/// multiply, and add separately; the candidate writes it exactly once.
fn bench_fused_subtract_multiply_add_out_matrix_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    const TARGET_BYTES_PER_OPERAND: usize = 32 * 1024 * 1024;
    const CONTRACT_ROUNDS: usize = 21;
    const CONTRACT_MIN_OF: usize = 1;
    const REQUIRED_BUILD_PROFILE: &str = "release-perf";

    assert_eq!(
        std::env::var("FNP_BENCH_PROFILE").as_deref(),
        Ok(REQUIRED_BUILD_PROFILE),
        "ship-grade fusion evidence requires FNP_BENCH_PROFILE=release-perf"
    );
    let build_worker =
        std::env::var("FNP_BUILD_WORKER").expect("FNP_BUILD_WORKER records the RCH build origin");
    assert!(
        !build_worker.trim().is_empty(),
        "FNP_BUILD_WORKER must be set"
    );
    let threads = std::env::var("RAYON_NUM_THREADS")
        .expect("RAYON_NUM_THREADS must be explicitly pinned before fusion timing");
    for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(threads.as_str()),
            "{variable} must match RAYON_NUM_THREADS"
        );
    }
    let threads: usize = threads.parse().expect("thread count is numeric");
    assert_eq!(
        rayon::current_num_threads(),
        threads,
        "Rayon pool width does not match the pinned fusion configuration"
    );
    println!(
        "WORKLOAD_RUNTIME workload=fused_subtract_multiply_add_out_matrix \
         build_worker={build_worker} build_profile={REQUIRED_BUILD_PROFILE} \
         pinned_threads={threads} rayon_threads={} contract_rounds={CONTRACT_ROUNDS} \
         contract_min_of={CONTRACT_MIN_OF} bytes_per_operand={TARGET_BYTES_PER_OPERAND}",
        rayon::current_num_threads()
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_fused_subtract_multiply_add_out")
            .expect("preallocated long fusion bench module");
        fnp_python(&module).expect("initialize preallocated long fusion module");
        let numpy = py.import("numpy").expect("numpy incumbent");

        let np_subtract = numpy.getattr("subtract").expect("numpy subtract");
        let np_multiply = numpy.getattr("multiply").expect("numpy multiply");
        let np_add = numpy.getattr("add").expect("numpy add");
        let fnp_chain = module
            .getattr("subtract_multiply_add")
            .expect("fnp subtract_multiply_add");
        common::report_numpy_incumbent_identity(py, "subtract", &np_subtract);
        println!("NUMPY_BUILD_CONFIG_BEGIN workload=fused_subtract_multiply_add_out_matrix");
        numpy
            .getattr("show_config")
            .expect("numpy.show_config")
            .call0()
            .expect("report NumPy build configuration");
        println!("NUMPY_BUILD_CONFIG_END workload=fused_subtract_multiply_add_out_matrix");
        for (name, callable) in [
            ("subtract", &np_subtract),
            ("multiply", &np_multiply),
            ("add", &np_add),
        ] {
            let owner = callable
                .getattr("__module__")
                .ok()
                .and_then(|value| value.extract::<String>().ok())
                .unwrap_or_else(|| "numpy".to_owned());
            assert!(
                owner.starts_with("numpy"),
                "incumbent {name} is not defined under numpy: {owner}"
            );
            assert!(
                !callable.is(&fnp_chain),
                "dispatch trap: incumbent {name} resolved to the candidate callable"
            );
        }
        common::report_incumbent_topology(
            "fnp.subtract_multiply_add(out=)",
            "numpy.subtract(out=)+numpy.multiply(out=)+numpy.add(out=)",
        );

        let checksum_of = |array: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
            let bytes = array
                .call_method0("tobytes")
                .expect("tobytes")
                .extract::<Vec<u8>>()
                .expect("byte Vec");
            bytes
                .len()
                .to_le_bytes()
                .iter()
                .chain(bytes.iter())
                .fold(0xcbf2_9ce4_8422_2325_u64, |state, &byte| {
                    (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                })
        };

        let cases = [
            ("float64", 8_usize, false),
            ("float32", 4, false),
            ("int8", 1, true),
            ("uint8", 1, true),
            ("int16", 2, true),
            ("uint16", 2, true),
            ("int32", 4, true),
            ("uint32", 4, true),
            ("int64", 8, true),
            ("uint64", 8, true),
        ];
        let selected_dtypes = std::env::var("FNP_FUSION_OUT_DTYPES").ok().map(|spec| {
            spec.split(',')
                .map(str::trim)
                .filter(|dtype| !dtype.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>()
        });
        println!(
            "FUSION_OUT_DTYPE_FILTER selected={}",
            selected_dtypes
                .as_ref()
                .map_or_else(|| "all".to_owned(), |dtypes| dtypes.join(":"))
        );
        for (case_index, (dtype, item_size, is_integer)) in cases.into_iter().enumerate() {
            if selected_dtypes
                .as_ref()
                .is_some_and(|selected| !selected.iter().any(|candidate| candidate == dtype))
            {
                continue;
            }
            let elements = TARGET_BYTES_PER_OPERAND / item_size;
            let namespace = PyDict::new(py);
            let setup = if is_integer {
                format!(
                    "import numpy as np\n\
                     rng = np.random.default_rng({})\n\
                     dt = np.dtype('{}')\n\
                     n = {}\n\
                     chain_a = np.frombuffer(rng.bytes(n * dt.itemsize), dtype=dt).copy()\n\
                     chain_b = np.frombuffer(rng.bytes(n * dt.itemsize), dtype=dt).copy()\n\
                     chain_c = np.frombuffer(rng.bytes(n * dt.itemsize), dtype=dt).copy()\n\
                     chain_d = np.frombuffer(rng.bytes(n * dt.itemsize), dtype=dt).copy()\n\
                     numpy_out = np.empty_like(chain_a)\n\
                     fnp_out = np.empty_like(chain_a)\n",
                    20260741 + case_index,
                    dtype,
                    elements,
                )
            } else {
                format!(
                    "import numpy as np\n\
                     rng = np.random.default_rng({})\n\
                     dt = np.dtype('{}')\n\
                     n = {}\n\
                     chain_a = rng.standard_normal(n).astype(dt)\n\
                     chain_b = rng.standard_normal(n).astype(dt)\n\
                     chain_c = rng.standard_normal(n).astype(dt)\n\
                     chain_d = rng.standard_normal(n).astype(dt)\n\
                     numpy_out = np.empty_like(chain_a)\n\
                     fnp_out = np.empty_like(chain_a)\n",
                    20260741 + case_index,
                    dtype,
                    elements,
                )
            };
            py.run(
                std::ffi::CString::new(setup)
                    .expect("preallocated long fusion corpus CString")
                    .as_c_str(),
                Some(&namespace),
                Some(&namespace),
            )
            .expect("preallocated long fusion corpus setup");
            let a = namespace.get_item("chain_a").expect("chain_a present");
            let b = namespace.get_item("chain_b").expect("chain_b present");
            let c = namespace.get_item("chain_c").expect("chain_c present");
            let d = namespace.get_item("chain_d").expect("chain_d present");
            let numpy_out = namespace.get_item("numpy_out").expect("numpy_out present");
            let fnp_out = namespace.get_item("fnp_out").expect("fnp_out present");
            let numpy_kwargs = PyDict::new(py);
            numpy_kwargs
                .set_item("out", &numpy_out)
                .expect("set NumPy output");
            let fnp_kwargs = PyDict::new(py);
            fnp_kwargs
                .set_item("out", &fnp_out)
                .expect("set candidate output");
            let row = format!("python_fused_subtract_multiply_add_out_{dtype}_32mib_vs_numpy");

            let run_incumbent = || {
                let difference = np_subtract
                    .call((black_box(&a), black_box(&b)), Some(&numpy_kwargs))
                    .expect("numpy subtract out arm");
                let product = np_multiply
                    .call((difference, black_box(&c)), Some(&numpy_kwargs))
                    .expect("numpy multiply out arm");
                np_add
                    .call((product, black_box(&d)), Some(&numpy_kwargs))
                    .expect("numpy add out arm")
            };
            let run_candidate = || {
                fnp_chain
                    .call(
                        (black_box(&a), black_box(&b), black_box(&c), black_box(&d)),
                        Some(&fnp_kwargs),
                    )
                    .expect("fnp subtract_multiply_add out arm")
            };

            let ours = run_candidate();
            let theirs = run_incumbent();
            assert!(ours.is(&fnp_out), "candidate did not return its output");
            assert!(theirs.is(&numpy_out), "NumPy did not return its output");
            assert_eq!(
                ours.call_method0("tobytes")
                    .expect("fnp tobytes")
                    .extract::<Vec<u8>>()
                    .expect("fnp bytes"),
                theirs
                    .call_method0("tobytes")
                    .expect("numpy tobytes")
                    .extract::<Vec<u8>>()
                    .expect("numpy bytes"),
                "preallocated long fused {dtype} result is not byte-identical to NumPy"
            );
            println!(
                "PARITY row={row} exact_bytes=passed result_dtype={dtype} \
                 result_elements={elements} result_bytes={TARGET_BYTES_PER_OPERAND} \
                 output_identity=passed checksum={:016x} operation_order=subtract_multiply_add",
                checksum_of(&theirs)
            );
            println!(
                "ROUTE_PRECONDITIONS row={row} dtype={dtype} elements={elements} \
                 operands=4 equal_shapes=true all_c_contiguous=true output_c_contiguous=true \
                 output_writable=true output_disjoint=true \
                 candidate_route=zerocopy_subtract_multiply_add_out_typed \
                 parallel_min=65536 chunk_elements=8192"
            );
            println!(
                "COUNTED_MECHANISM row={row} class=full_pass_elimination \
                 incumbent_element_stream_touches=9 candidate_element_stream_touches=5 \
                 incumbent_timed_allocations=0 candidate_timed_allocations=0 \
                 incumbent_full_array_sweeps=3 candidate_full_array_sweeps=1 \
                 eliminated_full_array_sweeps=2 output_storage=preallocated_both_arms \
                 arithmetic_ops_per_element=3 shared_inputs=true equal_work=true"
            );

            common::report_observed_thread_activity(&row, "numpy", 4, || {
                black_box(checksum_of(&run_incumbent()));
            });
            common::report_observed_thread_activity(&row, "fnp", 32, || {
                black_box(checksum_of(&run_candidate()));
            });

            let mut observe_incumbent = || {
                let started = Instant::now();
                let result = run_incumbent();
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let mut observe_candidate = || {
                let started = Instant::now();
                let result = run_candidate();
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract_with_sampling(
                    &row,
                    &mut observe_incumbent,
                    &mut observe_candidate,
                    CONTRACT_ROUNDS,
                    CONTRACT_MIN_OF,
                );
            let verdict =
                common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
            println!(
                "FUSION_OUT_RESULT row={row} dtype={dtype} verdict={verdict} \
                 incumbent_median_ms={:.6} candidate_median_ms={:.6} \
                 ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
                 incumbent_null_ratio={:.6} incumbent_null_ci95=[{:.6},{:.6}] \
                 candidate_null_ratio={:.6} candidate_null_ci95=[{:.6},{:.6}] \
                 corrected_dual_null_gate=true incumbent=numpy_live_same_invocation",
                effect.arm_a_median_ns / 1_000_000.0,
                effect.arm_b_median_ns / 1_000_000.0,
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                incumbent_null.ratio_median,
                incumbent_null.ratio_ci_low,
                incumbent_null.ratio_ci_high,
                candidate_null.ratio_median,
                candidate_null.ratio_ci_low,
                candidate_null.ratio_ci_high,
            );
            let decision = if verdict == "DECIDABLE_WIN" {
                "choose_fnp"
            } else {
                "choose_numpy"
            };
            println!(
                "CHOOSER_STATEMENT workload=fused_subtract_multiply_add_out_{dtype}_32mib \
                 decision={decision} verdict={verdict} incumbent=numpy_live_same_invocation \
                 measured_scope={elements}_equal_shape_c_contiguous_elements_preallocated_output_at_{threads}_pinned_threads \
                 outside_scope=run_same_contract_before_choosing"
            );
        }
    });
}

/// Broad integer-width proof for the fused `a * b + c` implementation.
///
/// Every row processes 32 MiB per operand so the working set is beyond cache
/// for narrow as well as wide dtypes. The incumbent is the live two-ufunc
/// NumPy expression in the same invocation; the candidate eliminates its full
/// intermediate and performs wrapping integer arithmetic in one parallel pass.
fn bench_fused_multiply_add_integer_matrix_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    const TARGET_BYTES_PER_OPERAND: usize = 32 * 1024 * 1024;
    const CONTRACT_ROUNDS: usize = 21;
    const CONTRACT_MIN_OF: usize = 1;
    const REQUIRED_BUILD_PROFILE: &str = "release-perf";

    assert_eq!(
        std::env::var("FNP_BENCH_PROFILE").as_deref(),
        Ok(REQUIRED_BUILD_PROFILE),
        "ship-grade fusion evidence requires FNP_BENCH_PROFILE=release-perf"
    );
    let build_worker =
        std::env::var("FNP_BUILD_WORKER").expect("FNP_BUILD_WORKER records the RCH build origin");
    assert!(
        !build_worker.trim().is_empty(),
        "FNP_BUILD_WORKER must be set"
    );
    let threads = std::env::var("RAYON_NUM_THREADS")
        .expect("RAYON_NUM_THREADS must be explicitly pinned before fusion timing");
    for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(threads.as_str()),
            "{variable} must be pinned to the same width as RAYON_NUM_THREADS"
        );
    }
    let threads: usize = threads.parse().expect("thread count is numeric");
    assert_eq!(
        rayon::current_num_threads(),
        threads,
        "Rayon pool width does not match the pinned fusion configuration"
    );
    println!(
        "WORKLOAD_RUNTIME workload=fused_multiply_add_integer_matrix \
         build_worker={build_worker} build_profile={REQUIRED_BUILD_PROFILE} \
         pinned_threads={threads} rayon_threads={} contract_rounds={CONTRACT_ROUNDS} \
         contract_min_of={CONTRACT_MIN_OF} bytes_per_operand={TARGET_BYTES_PER_OPERAND}",
        rayon::current_num_threads()
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_fused_multiply_add_integer_matrix")
            .expect("integer fusion bench module");
        fnp_python(&module).expect("initialize fnp_python integer fusion module");
        let numpy = py.import("numpy").expect("numpy incumbent");
        let np_multiply = numpy.getattr("multiply").expect("numpy multiply");
        let np_add = numpy.getattr("add").expect("numpy add");
        let fnp_multiply_add = module.getattr("multiply_add").expect("fnp multiply_add");
        common::report_numpy_incumbent_identity(py, "multiply", &np_multiply);
        common::report_incumbent_topology(
            "fnp.multiply_add_integer_matrix",
            "numpy.multiply+numpy.add",
        );

        let checksum_of = |array: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
            let bytes = array
                .call_method0("tobytes")
                .expect("tobytes")
                .extract::<Vec<u8>>()
                .expect("byte Vec");
            bytes
                .len()
                .to_le_bytes()
                .iter()
                .chain(bytes.iter())
                .fold(0xcbf2_9ce4_8422_2325_u64, |state, &byte| {
                    (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                })
        };

        let dtypes = [
            ("int8", 1usize, -120i64, 120i64),
            ("uint8", 1, 0, 240),
            ("int16", 2, -30_000, 30_000),
            ("uint16", 2, 0, 60_000),
            ("int32", 4, -1_000_000, 1_000_000),
            ("uint32", 4, 0, 2_000_000),
            ("int64", 8, -1_000_000, 1_000_000),
            ("uint64", 8, 0, 2_000_000),
        ];

        for (dtype, item_size, low, high) in dtypes {
            let elements = TARGET_BYTES_PER_OPERAND / item_size;
            let row = format!("python_fused_multiply_add_{dtype}_32mib_vs_numpy");
            let namespace = PyDict::new(py);
            let corpus = format!(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260731)\n\
                 fma_a = rng.integers({low}, {high}, {elements}, dtype=np.{dtype})\n\
                 fma_b = rng.integers({low}, {high}, {elements}, dtype=np.{dtype})\n\
                 fma_c = rng.integers({low}, {high}, {elements}, dtype=np.{dtype})\n"
            );
            py.run(
                std::ffi::CString::new(corpus)
                    .expect("integer fusion corpus CString")
                    .as_c_str(),
                Some(&namespace),
                Some(&namespace),
            )
            .expect("integer fusion corpus setup");
            let fma_a = namespace.get_item("fma_a").expect("fma_a present");
            let fma_b = namespace.get_item("fma_b").expect("fma_b present");
            let fma_c = namespace.get_item("fma_c").expect("fma_c present");

            let run_incumbent = || {
                let product = np_multiply
                    .call1((black_box(&fma_a), black_box(&fma_b)))
                    .expect("numpy multiply arm");
                np_add
                    .call1((product, black_box(&fma_c)))
                    .expect("numpy add arm")
            };
            let run_candidate = || {
                fnp_multiply_add
                    .call1((black_box(&fma_a), black_box(&fma_b), black_box(&fma_c)))
                    .expect("fnp multiply_add arm")
            };

            let ours = run_candidate();
            let theirs = run_incumbent();
            assert_eq!(
                ours.call_method0("tobytes")
                    .expect("fnp tobytes")
                    .extract::<Vec<u8>>()
                    .expect("fnp bytes"),
                theirs
                    .call_method0("tobytes")
                    .expect("numpy tobytes")
                    .extract::<Vec<u8>>()
                    .expect("numpy bytes"),
                "fused {dtype} result is not byte-identical to NumPy"
            );
            assert_eq!(
                ours.getattr("dtype")
                    .expect("fnp dtype")
                    .str()
                    .expect("fnp dtype str")
                    .to_string(),
                dtype,
                "fused integer dtype drifted"
            );
            println!(
                "PARITY row={row} exact_bytes=passed result_dtype={dtype} \
                 result_elements={elements} result_bytes={TARGET_BYTES_PER_OPERAND} \
                 checksum={:016x} wrapping_overflow=explicit",
                checksum_of(&theirs)
            );
            println!(
                "COUNTED_MECHANISM row={row} class=materialization_and_pass_elimination \
                 incumbent_element_stream_touches=6 candidate_element_stream_touches=4 \
                 incumbent_output_allocations=2 candidate_output_allocations=1 \
                 incumbent_eliminated_temporary_bytes={TARGET_BYTES_PER_OPERAND} \
                 shared_inputs=true equal_work=true"
            );

            common::report_observed_thread_activity(&row, "numpy", 1, || {
                black_box(checksum_of(&run_incumbent()));
            });
            common::report_observed_thread_activity(&row, "fnp", 1, || {
                black_box(checksum_of(&run_candidate()));
            });

            let mut observe_incumbent = || {
                let started = Instant::now();
                let result = run_incumbent();
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let mut observe_candidate = || {
                let started = Instant::now();
                let result = run_candidate();
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract_with_sampling(
                    &row,
                    &mut observe_incumbent,
                    &mut observe_candidate,
                    CONTRACT_ROUNDS,
                    CONTRACT_MIN_OF,
                );
            let verdict =
                common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
            println!(
                "FUSION_DTYPE_RESULT row={row} dtype={dtype} verdict={verdict} \
                 incumbent_median_ms={:.6} candidate_median_ms={:.6} \
                 ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
                 incumbent_null_ci95=[{:.6},{:.6}] candidate_null_ci95=[{:.6},{:.6}] \
                 corrected_dual_null_gate=true incumbent=numpy_live_same_invocation",
                effect.arm_a_median_ns / 1_000_000.0,
                effect.arm_b_median_ns / 1_000_000.0,
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                incumbent_null.ratio_ci_low,
                incumbent_null.ratio_ci_high,
                candidate_null.ratio_ci_low,
                candidate_null.ratio_ci_high,
            );
            let decision = if verdict == "DECIDABLE_WIN" {
                "choose_fnp"
            } else {
                "choose_numpy"
            };
            println!(
                "CHOOSER_STATEMENT workload=fused_multiply_add_{dtype}_32mib \
                 decision={decision} verdict={verdict} \
                 incumbent=numpy_live_same_invocation measured_scope={elements}_equal_shape_c_contiguous_elements_at_{threads}_pinned_threads \
                 outside_scope=run_same_contract_before_choosing"
            );
        }
    });
}

/// Fusion dtype matrix for the two families NumPy handles WORST, measured whole
/// against live NumPy in the same invocation.
///
/// `float16`: NumPy has no f16 ALU at all, so `multiply` widens every half to
/// `float`, multiplies, and narrows back — then `add` pays the same round trip
/// again, on top of materializing the intermediate.
///
/// `complex64`/`complex128`: the intermediate is the widest in the whole family
/// (32 MiB per operand here, same as the inputs), and NumPy's complex loops are
/// single-threaded like every other ufunc.
///
/// The complex rows also pin the CONTRACTION asymmetry: NumPy's complex multiply
/// is compiled with FP contraction, so byte-exactness REQUIRES our kernel to
/// fuse, where the real `float64` row above requires it not to. Both directions
/// are asserted before any timing happens.
/// Preallocated `out=` fusion for `a * b + c`, measured whole against live NumPy.
///
/// NEITHER ARM ALLOCATES. The incumbent is the strongest explicit NumPy spelling
/// of this contract — `numpy.multiply(a, b, out=out)` then
/// `numpy.add(out, c, out=out)` — which needs no temporary at all. Verified
/// byte-identical to `numpy.add(numpy.multiply(a, b), c, out=out)` before timing,
/// so this is a fair incumbent rather than a weakened one.
///
/// That removes the allocator from both sides, so the entire remaining gap is
/// pass elimination (six element-stream touches down to four) plus parallelism.
/// These ratios are therefore expected to sit BELOW the allocating rows, and the
/// comparison is the honest one for callers who already reuse buffers.
/// Flat `float64` min/max against live NumPy.
///
/// NumPy runs BOTH single-threaded — measured `cpu/wall = 1.00x` at 64M
/// `float64`, the same structural gap as every reduction it owns; it has no
/// threading layer for them at all. Unlike the sum route there is no
/// accumulation tree to reproduce, because selection is order-independent.
///
/// The corpus is deliberately drawn from a continuous distribution so the
/// extremum is not a zero: mixed-sign-zero extrema DEFER (NumPy's tie between
/// `+0.0` and `-0.0` follows its internal blocking, not index order), and a
/// corpus that tripped that guard would silently measure NumPy against NumPy.
fn bench_f64_flat_min_max_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    const CONTRACT_ROUNDS: usize = 21;
    const CONTRACT_MIN_OF: usize = 1;
    const REQUIRED_BUILD_PROFILE: &str = "release-perf";

    assert_eq!(
        std::env::var("FNP_BENCH_PROFILE").as_deref(),
        Ok(REQUIRED_BUILD_PROFILE),
        "ship-grade min/max evidence requires FNP_BENCH_PROFILE=release-perf"
    );
    let build_worker =
        std::env::var("FNP_BUILD_WORKER").expect("FNP_BUILD_WORKER records the RCH build origin");
    assert!(
        !build_worker.trim().is_empty(),
        "FNP_BUILD_WORKER must be set"
    );
    let threads = std::env::var("RAYON_NUM_THREADS")
        .expect("RAYON_NUM_THREADS must be explicitly pinned before min/max timing");
    for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(threads.as_str()),
            "{variable} must be pinned to the same width as RAYON_NUM_THREADS"
        );
    }
    let threads: usize = threads.parse().expect("thread count is numeric");
    assert_eq!(
        rayon::current_num_threads(),
        threads,
        "Rayon pool width does not match the pinned min/max configuration"
    );
    println!(
        "WORKLOAD_RUNTIME workload=f64_flat_min_max build_worker={build_worker} \
         build_profile={REQUIRED_BUILD_PROFILE} pinned_threads={threads} \
         rayon_threads={} contract_rounds={CONTRACT_ROUNDS} contract_min_of={CONTRACT_MIN_OF}",
        rayon::current_num_threads()
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_f64_min_max").expect("min/max bench module");
        fnp_python(&module).expect("initialize fnp_python min/max module");
        let numpy = py.import("numpy").expect("numpy incumbent");
        let np_min = numpy.getattr("min").expect("numpy min");
        let np_max = numpy.getattr("max").expect("numpy max");
        let fnp_min = module.getattr("min").expect("fnp min");
        let fnp_max = module.getattr("max").expect("fnp max");
        let np_argmin = numpy.getattr("argmin").expect("numpy argmin");
        let np_argmax = numpy.getattr("argmax").expect("numpy argmax");
        let fnp_argmin = module.getattr("argmin").expect("fnp argmin");
        let fnp_argmax = module.getattr("argmax").expect("fnp argmax");
        common::report_numpy_incumbent_identity(py, "min", &np_min);
        common::report_incumbent_topology("fnp.min_max_f64_flat", "numpy.min+numpy.max");

        let bits_of = |value: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
            value.extract::<f64>().expect("float64 scalar").to_bits()
        };
        // arg* return an INDEX, not a float, so they need their own extraction.
        let index_of = |value: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
            value.extract::<u64>().expect("index")
        };

        for &elements in &[1usize << 22, 1 << 26] {
            let namespace = PyDict::new(py);
            let corpus = format!(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260731)\n\
                 mm_a = rng.standard_normal({elements})\n"
            );
            py.run(
                std::ffi::CString::new(corpus)
                    .expect("min/max corpus CString")
                    .as_c_str(),
                Some(&namespace),
                Some(&namespace),
            )
            .expect("min/max corpus setup");
            let mm_a = namespace.get_item("mm_a").expect("mm_a present");

            for (label, np_fn, fnp_fn) in [("min", &np_min, &fnp_min), ("max", &np_max, &fnp_max)] {
                let row = format!("python_f64_flat_{label}_{elements}_vs_numpy");
                let run_incumbent = || np_fn.call1((black_box(&mm_a),)).expect("numpy arm");
                let run_candidate = || fnp_fn.call1((black_box(&mm_a),)).expect("fnp arm");

                let ours = run_candidate();
                let theirs = run_incumbent();
                assert_eq!(
                    bits_of(&ours),
                    bits_of(&theirs),
                    "f64 flat {label} at {elements} is not BIT-identical to NumPy"
                );
                // A zero extremum would have taken the mixed-sign deferral guard
                // and this row would be measuring NumPy against NumPy.
                assert!(
                    bits_of(&theirs) != 0u64 && bits_of(&theirs) != (1u64 << 63),
                    "corpus extremum is a zero; this row would measure the deferral path"
                );
                println!(
                    "PARITY row={row} exact_bits=passed op={label} result_elements={elements} \
                     result_bits={:016x} tie_regime=non_zero_extremum",
                    bits_of(&theirs)
                );
                println!(
                    "COUNTED_MECHANISM row={row} class=serial_scan_parallelised \
                     incumbent_threads=1 candidate_threads={threads} \
                     order_independent_selection=true shared_inputs=true equal_work=true"
                );

                let reps = ((1usize << 27) / elements).max(1);
                common::report_observed_thread_activity(&row, "numpy", reps, || {
                    black_box(bits_of(&run_incumbent()));
                });
                common::report_observed_thread_activity(&row, "fnp", reps, || {
                    black_box(bits_of(&run_candidate()));
                });

                let mut observe_incumbent = || {
                    let started = Instant::now();
                    let result = run_incumbent();
                    let elapsed = started.elapsed();
                    common::ContractObservation {
                        elapsed,
                        checksum: bits_of(&result),
                    }
                };
                let mut observe_candidate = || {
                    let started = Instant::now();
                    let result = run_candidate();
                    let elapsed = started.elapsed();
                    common::ContractObservation {
                        elapsed,
                        checksum: bits_of(&result),
                    }
                };
                let (effect, incumbent_null, candidate_null) =
                    common::run_dual_null_median_ci_contract_with_sampling(
                        &row,
                        &mut observe_incumbent,
                        &mut observe_candidate,
                        CONTRACT_ROUNDS,
                        CONTRACT_MIN_OF,
                    );
                let verdict =
                    common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
                println!(
                    "MINMAX_RESULT row={row} op={label} elements={elements} verdict={verdict} \
                     incumbent_median_ms={:.6} candidate_median_ms={:.6} \
                     ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
                     incumbent_null_ci95=[{:.6},{:.6}] candidate_null_ci95=[{:.6},{:.6}] \
                     corrected_dual_null_gate=true incumbent=numpy_live_same_invocation",
                    effect.arm_a_median_ns / 1_000_000.0,
                    effect.arm_b_median_ns / 1_000_000.0,
                    effect.ratio_median,
                    effect.ratio_ci_low,
                    effect.ratio_ci_high,
                    incumbent_null.ratio_ci_low,
                    incumbent_null.ratio_ci_high,
                    candidate_null.ratio_ci_low,
                    candidate_null.ratio_ci_high,
                );
                let decision = if verdict == "DECIDABLE_WIN" {
                    "choose_fnp"
                } else {
                    "choose_numpy"
                };
                println!(
                    "CHOOSER_STATEMENT workload=f64_flat_{label}_{elements} decision={decision} \
                     verdict={verdict} incumbent=numpy_live_same_invocation \
                     measured_scope={elements}_c_contiguous_float64_elements_at_{threads}_pinned_threads \
                     outside_scope=run_same_contract_before_choosing"
                );
            }

            // arg* rows. Same corpus, but the result is an index, and unlike the
            // value routes these have NO deferral regimes — ties resolve to the
            // first index and NaN resolves to the first NaN's index, both
            // exactly, so any corpus is measurable.
            for (label, np_fn, fnp_fn) in [
                ("argmin", &np_argmin, &fnp_argmin),
                ("argmax", &np_argmax, &fnp_argmax),
            ] {
                let row = format!("python_f64_flat_{label}_{elements}_vs_numpy");
                let run_incumbent = || np_fn.call1((black_box(&mm_a),)).expect("numpy arg arm");
                let run_candidate = || fnp_fn.call1((black_box(&mm_a),)).expect("fnp arg arm");

                let ours = run_candidate();
                let theirs = run_incumbent();
                assert_eq!(
                    index_of(&ours),
                    index_of(&theirs),
                    "f64 flat {label} at {elements} does not match NumPy's index"
                );
                println!(
                    "PARITY row={row} exact_index=passed op={label} result_elements={elements} \
                     result_index={} tie_regime=first_index",
                    index_of(&theirs)
                );
                println!(
                    "COUNTED_MECHANISM row={row} class=serial_scan_parallelised_and_vectorised \
                     incumbent_threads=1 candidate_threads={threads} \
                     order_independent_selection=true shared_inputs=true equal_work=true"
                );

                let reps = ((1usize << 27) / elements).max(1);
                common::report_observed_thread_activity(&row, "numpy", reps, || {
                    black_box(index_of(&run_incumbent()));
                });
                common::report_observed_thread_activity(&row, "fnp", reps, || {
                    black_box(index_of(&run_candidate()));
                });

                let mut observe_incumbent = || {
                    let started = Instant::now();
                    let result = run_incumbent();
                    let elapsed = started.elapsed();
                    common::ContractObservation {
                        elapsed,
                        checksum: index_of(&result),
                    }
                };
                let mut observe_candidate = || {
                    let started = Instant::now();
                    let result = run_candidate();
                    let elapsed = started.elapsed();
                    common::ContractObservation {
                        elapsed,
                        checksum: index_of(&result),
                    }
                };
                let (effect, incumbent_null, candidate_null) =
                    common::run_dual_null_median_ci_contract_with_sampling(
                        &row,
                        &mut observe_incumbent,
                        &mut observe_candidate,
                        CONTRACT_ROUNDS,
                        CONTRACT_MIN_OF,
                    );
                let verdict =
                    common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
                println!(
                    "MINMAX_RESULT row={row} op={label} elements={elements} verdict={verdict} \
                     incumbent_median_ms={:.6} candidate_median_ms={:.6} \
                     ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
                     incumbent_null_ci95=[{:.6},{:.6}] candidate_null_ci95=[{:.6},{:.6}] \
                     corrected_dual_null_gate=true incumbent=numpy_live_same_invocation",
                    effect.arm_a_median_ns / 1_000_000.0,
                    effect.arm_b_median_ns / 1_000_000.0,
                    effect.ratio_median,
                    effect.ratio_ci_low,
                    effect.ratio_ci_high,
                    incumbent_null.ratio_ci_low,
                    incumbent_null.ratio_ci_high,
                    candidate_null.ratio_ci_low,
                    candidate_null.ratio_ci_high,
                );
                let decision = if verdict == "DECIDABLE_WIN" {
                    "choose_fnp"
                } else {
                    "choose_numpy"
                };
                println!(
                    "CHOOSER_STATEMENT workload=f64_flat_{label}_{elements} decision={decision} \
                     verdict={verdict} incumbent=numpy_live_same_invocation \
                     measured_scope={elements}_c_contiguous_float64_elements_at_{threads}_pinned_threads \
                     outside_scope=run_same_contract_before_choosing"
                );
            }
        }
    });
}

fn bench_fused_multiply_add_out_matrix_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    const TARGET_BYTES_PER_OPERAND: usize = 32 * 1024 * 1024;
    const CONTRACT_ROUNDS: usize = 21;
    const CONTRACT_MIN_OF: usize = 1;
    const REQUIRED_BUILD_PROFILE: &str = "release-perf";

    assert_eq!(
        std::env::var("FNP_BENCH_PROFILE").as_deref(),
        Ok(REQUIRED_BUILD_PROFILE),
        "ship-grade fusion evidence requires FNP_BENCH_PROFILE=release-perf"
    );
    let build_worker =
        std::env::var("FNP_BUILD_WORKER").expect("FNP_BUILD_WORKER records the RCH build origin");
    assert!(
        !build_worker.trim().is_empty(),
        "FNP_BUILD_WORKER must be set"
    );
    let threads = std::env::var("RAYON_NUM_THREADS")
        .expect("RAYON_NUM_THREADS must be explicitly pinned before fusion timing");
    for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(threads.as_str()),
            "{variable} must be pinned to the same width as RAYON_NUM_THREADS"
        );
    }
    let threads: usize = threads.parse().expect("thread count is numeric");
    assert_eq!(
        rayon::current_num_threads(),
        threads,
        "Rayon pool width does not match the pinned fusion configuration"
    );
    println!(
        "WORKLOAD_RUNTIME workload=fused_multiply_add_out_matrix \
         build_worker={build_worker} build_profile={REQUIRED_BUILD_PROFILE} \
         pinned_threads={threads} rayon_threads={} contract_rounds={CONTRACT_ROUNDS} \
         contract_min_of={CONTRACT_MIN_OF} bytes_per_operand={TARGET_BYTES_PER_OPERAND} \
         preallocated_output=true allocations_in_either_arm=0",
        rayon::current_num_threads()
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_fused_multiply_add_out")
            .expect("out= fusion bench module");
        fnp_python(&module).expect("initialize fnp_python out= fusion module");
        let numpy = py.import("numpy").expect("numpy incumbent");
        let np_multiply = numpy.getattr("multiply").expect("numpy multiply");
        let np_add = numpy.getattr("add").expect("numpy add");
        let fnp_multiply_add = module.getattr("multiply_add").expect("fnp multiply_add");
        common::report_numpy_incumbent_identity(py, "multiply", &np_multiply);
        common::report_incumbent_topology(
            "fnp.multiply_add_out_matrix",
            "numpy.multiply+numpy.add",
        );

        let checksum_of = |array: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
            let bytes = array
                .call_method0("tobytes")
                .expect("tobytes")
                .extract::<Vec<u8>>()
                .expect("byte Vec");
            bytes
                .len()
                .to_le_bytes()
                .iter()
                .chain(bytes.iter())
                .fold(0xcbf2_9ce4_8422_2325_u64, |state, &byte| {
                    (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                })
        };

        let dtypes = [
            ("float64", 8usize),
            ("float32", 4),
            ("float16", 2),
            ("int64", 8),
            ("complex128", 16),
            ("complex64", 8),
        ];

        for (dtype, item_size) in dtypes {
            let elements = TARGET_BYTES_PER_OPERAND / item_size;
            let row = format!("python_fused_multiply_add_out_{dtype}_32mib_vs_numpy");
            let namespace = PyDict::new(py);
            let draw = if dtype.starts_with("complex") {
                format!(
                    "(rng.standard_normal({elements}) \
                     + 1j * rng.standard_normal({elements})).astype(np.{dtype})"
                )
            } else if dtype.starts_with("float") {
                format!("(rng.standard_normal({elements}) * 4).astype(np.{dtype})")
            } else {
                format!("rng.integers(-1000000, 1000000, {elements}, dtype=np.{dtype})")
            };
            // Both output buffers are allocated ONCE here, outside every timed
            // region, so neither arm pays an allocation during measurement.
            let corpus = format!(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260731)\n\
                 fma_a = {draw}\n\
                 fma_b = {draw}\n\
                 fma_c = {draw}\n\
                 out_numpy = np.empty({elements}, dtype=np.{dtype})\n\
                 out_fnp = np.empty({elements}, dtype=np.{dtype})\n"
            );
            py.run(
                std::ffi::CString::new(corpus)
                    .expect("out= fusion corpus CString")
                    .as_c_str(),
                Some(&namespace),
                Some(&namespace),
            )
            .expect("out= fusion corpus setup");
            let fma_a = namespace.get_item("fma_a").expect("fma_a present");
            let fma_b = namespace.get_item("fma_b").expect("fma_b present");
            let fma_c = namespace.get_item("fma_c").expect("fma_c present");
            let out_numpy = namespace.get_item("out_numpy").expect("out_numpy present");
            let out_fnp = namespace.get_item("out_fnp").expect("out_fnp present");

            // Chained through out=, so the incumbent needs no temporary either.
            let run_incumbent = || {
                np_multiply
                    .call1((black_box(&fma_a), black_box(&fma_b), black_box(&out_numpy)))
                    .expect("numpy multiply out= arm");
                np_add
                    .call1((
                        black_box(&out_numpy),
                        black_box(&fma_c),
                        black_box(&out_numpy),
                    ))
                    .expect("numpy add out= arm")
            };
            let run_candidate = || {
                let kwargs = PyDict::new(py);
                kwargs.set_item("out", black_box(&out_fnp)).expect("out kw");
                fnp_multiply_add
                    .call(
                        (black_box(&fma_a), black_box(&fma_b), black_box(&fma_c)),
                        Some(&kwargs),
                    )
                    .expect("fnp multiply_add out= arm")
            };

            let ours = run_candidate();
            let theirs = run_incumbent();
            assert_eq!(
                ours.call_method0("tobytes")
                    .expect("fnp tobytes")
                    .extract::<Vec<u8>>()
                    .expect("fnp bytes"),
                theirs
                    .call_method0("tobytes")
                    .expect("numpy tobytes")
                    .extract::<Vec<u8>>()
                    .expect("numpy bytes"),
                "fused out= {dtype} result is not byte-identical to NumPy"
            );
            // The routed contract RETURNS the caller's array; if it ever stopped
            // writing in place the ratio would be measuring the wrong thing.
            assert!(
                ours.is(&out_fnp),
                "fnp.multiply_add(out=) must return the caller's output array"
            );
            println!(
                "PARITY row={row} exact_bytes=passed result_dtype={dtype} \
                 result_elements={elements} result_bytes={TARGET_BYTES_PER_OPERAND} \
                 checksum={:016x} preallocated_output=true returned_output_identity=passed",
                checksum_of(&theirs)
            );
            println!(
                "COUNTED_MECHANISM row={row} class=pass_elimination_only \
                 incumbent_element_stream_touches=6 candidate_element_stream_touches=4 \
                 incumbent_output_allocations=0 candidate_output_allocations=0 \
                 incumbent_eliminated_temporary_bytes=0 \
                 shared_inputs=true equal_work=true"
            );

            common::report_observed_thread_activity(&row, "numpy", 1, || {
                black_box(checksum_of(&run_incumbent()));
            });
            common::report_observed_thread_activity(&row, "fnp", 1, || {
                black_box(checksum_of(&run_candidate()));
            });

            let mut observe_incumbent = || {
                let started = Instant::now();
                let result = run_incumbent();
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let mut observe_candidate = || {
                let started = Instant::now();
                let result = run_candidate();
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract_with_sampling(
                    &row,
                    &mut observe_incumbent,
                    &mut observe_candidate,
                    CONTRACT_ROUNDS,
                    CONTRACT_MIN_OF,
                );
            let verdict =
                common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
            println!(
                "FUSION_OUT_RESULT row={row} dtype={dtype} verdict={verdict} \
                 incumbent_median_ms={:.6} candidate_median_ms={:.6} \
                 ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
                 incumbent_null_ci95=[{:.6},{:.6}] candidate_null_ci95=[{:.6},{:.6}] \
                 corrected_dual_null_gate=true incumbent=numpy_live_same_invocation",
                effect.arm_a_median_ns / 1_000_000.0,
                effect.arm_b_median_ns / 1_000_000.0,
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                incumbent_null.ratio_ci_low,
                incumbent_null.ratio_ci_high,
                candidate_null.ratio_ci_low,
                candidate_null.ratio_ci_high,
            );
            let decision = if verdict == "DECIDABLE_WIN" {
                "choose_fnp"
            } else {
                "choose_numpy"
            };
            println!(
                "CHOOSER_STATEMENT workload=fused_multiply_add_out_{dtype}_32mib \
                 decision={decision} verdict={verdict} \
                 incumbent=numpy_live_same_invocation measured_scope={elements}_equal_shape_c_contiguous_elements_preallocated_out_at_{threads}_pinned_threads \
                 outside_scope=run_same_contract_before_choosing"
            );
        }
    });
}

fn bench_fused_multiply_add_narrow_and_complex_matrix_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    const TARGET_BYTES_PER_OPERAND: usize = 32 * 1024 * 1024;
    const CONTRACT_ROUNDS: usize = 21;
    const CONTRACT_MIN_OF: usize = 1;
    const REQUIRED_BUILD_PROFILE: &str = "release-perf";

    assert_eq!(
        std::env::var("FNP_BENCH_PROFILE").as_deref(),
        Ok(REQUIRED_BUILD_PROFILE),
        "ship-grade fusion evidence requires FNP_BENCH_PROFILE=release-perf"
    );
    let build_worker =
        std::env::var("FNP_BUILD_WORKER").expect("FNP_BUILD_WORKER records the RCH build origin");
    assert!(
        !build_worker.trim().is_empty(),
        "FNP_BUILD_WORKER must be set"
    );
    let threads = std::env::var("RAYON_NUM_THREADS")
        .expect("RAYON_NUM_THREADS must be explicitly pinned before fusion timing");
    for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(threads.as_str()),
            "{variable} must be pinned to the same width as RAYON_NUM_THREADS"
        );
    }
    let threads: usize = threads.parse().expect("thread count is numeric");
    assert_eq!(
        rayon::current_num_threads(),
        threads,
        "Rayon pool width does not match the pinned fusion configuration"
    );
    println!(
        "WORKLOAD_RUNTIME workload=fused_multiply_add_narrow_and_complex_matrix \
         build_worker={build_worker} build_profile={REQUIRED_BUILD_PROFILE} \
         pinned_threads={threads} rayon_threads={} contract_rounds={CONTRACT_ROUNDS} \
         contract_min_of={CONTRACT_MIN_OF} bytes_per_operand={TARGET_BYTES_PER_OPERAND}",
        rayon::current_num_threads()
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_fused_multiply_add_narrow_complex")
            .expect("narrow/complex fusion bench module");
        fnp_python(&module).expect("initialize fnp_python narrow/complex fusion module");
        let numpy = py.import("numpy").expect("numpy incumbent");
        let np_multiply = numpy.getattr("multiply").expect("numpy multiply");
        let np_add = numpy.getattr("add").expect("numpy add");
        let fnp_multiply_add = module.getattr("multiply_add").expect("fnp multiply_add");
        common::report_numpy_incumbent_identity(py, "multiply", &np_multiply);
        common::report_incumbent_topology(
            "fnp.multiply_add_narrow_and_complex_matrix",
            "numpy.multiply+numpy.add",
        );

        let checksum_of = |array: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
            let bytes = array
                .call_method0("tobytes")
                .expect("tobytes")
                .extract::<Vec<u8>>()
                .expect("byte Vec");
            bytes
                .len()
                .to_le_bytes()
                .iter()
                .chain(bytes.iter())
                .fold(0xcbf2_9ce4_8422_2325_u64, |state, &byte| {
                    (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                })
        };

        // (dtype, itemsize, contraction class asserted before timing)
        let dtypes = [
            ("float16", 2usize, "must_not_contract_narrow_per_ufunc"),
            ("complex64", 8, "must_contract_like_numpy"),
            ("complex128", 16, "must_contract_like_numpy"),
        ];

        for (dtype, item_size, contraction_class) in dtypes {
            let elements = TARGET_BYTES_PER_OPERAND / item_size;
            let row = format!("python_fused_multiply_add_{dtype}_32mib_vs_numpy");
            let namespace = PyDict::new(py);
            let draw = if dtype == "float16" {
                format!("(rng.standard_normal({elements}) * 4).astype(np.{dtype})")
            } else {
                format!(
                    "(rng.standard_normal({elements}) \
                     + 1j * rng.standard_normal({elements})).astype(np.{dtype})"
                )
            };
            let corpus = format!(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260731)\n\
                 fma_a = {draw}\n\
                 fma_b = {draw}\n\
                 fma_c = {draw}\n"
            );
            py.run(
                std::ffi::CString::new(corpus)
                    .expect("narrow/complex fusion corpus CString")
                    .as_c_str(),
                Some(&namespace),
                Some(&namespace),
            )
            .expect("narrow/complex fusion corpus setup");
            let fma_a = namespace.get_item("fma_a").expect("fma_a present");
            let fma_b = namespace.get_item("fma_b").expect("fma_b present");
            let fma_c = namespace.get_item("fma_c").expect("fma_c present");

            let run_incumbent = || {
                let product = np_multiply
                    .call1((black_box(&fma_a), black_box(&fma_b)))
                    .expect("numpy multiply arm");
                np_add
                    .call1((product, black_box(&fma_c)))
                    .expect("numpy add arm")
            };
            let run_candidate = || {
                fnp_multiply_add
                    .call1((black_box(&fma_a), black_box(&fma_b), black_box(&fma_c)))
                    .expect("fnp multiply_add arm")
            };

            let ours = run_candidate();
            let theirs = run_incumbent();
            assert_eq!(
                ours.call_method0("tobytes")
                    .expect("fnp tobytes")
                    .extract::<Vec<u8>>()
                    .expect("fnp bytes"),
                theirs
                    .call_method0("tobytes")
                    .expect("numpy tobytes")
                    .extract::<Vec<u8>>()
                    .expect("numpy bytes"),
                "fused {dtype} result is not byte-identical to NumPy"
            );
            assert_eq!(
                ours.getattr("dtype")
                    .expect("fnp dtype")
                    .str()
                    .expect("fnp dtype str")
                    .to_string(),
                dtype,
                "fused narrow/complex dtype drifted"
            );
            // The route must actually engage. If it silently deferred, the bytes
            // would still match and the row would measure numpy against numpy.
            assert!(
                !fnp_multiply_add.is(&np_multiply),
                "fnp.multiply_add resolved to numpy.multiply"
            );
            println!(
                "PARITY row={row} exact_bytes=passed result_dtype={dtype} \
                 result_elements={elements} result_bytes={TARGET_BYTES_PER_OPERAND} \
                 checksum={:016x} contraction_class={contraction_class}",
                checksum_of(&theirs)
            );
            println!(
                "COUNTED_MECHANISM row={row} class=materialization_and_pass_elimination \
                 incumbent_element_stream_touches=6 candidate_element_stream_touches=4 \
                 incumbent_output_allocations=2 candidate_output_allocations=1 \
                 incumbent_eliminated_temporary_bytes={TARGET_BYTES_PER_OPERAND} \
                 shared_inputs=true equal_work=true"
            );

            common::report_observed_thread_activity(&row, "numpy", 1, || {
                black_box(checksum_of(&run_incumbent()));
            });
            common::report_observed_thread_activity(&row, "fnp", 1, || {
                black_box(checksum_of(&run_candidate()));
            });

            let mut observe_incumbent = || {
                let started = Instant::now();
                let result = run_incumbent();
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let mut observe_candidate = || {
                let started = Instant::now();
                let result = run_candidate();
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract_with_sampling(
                    &row,
                    &mut observe_incumbent,
                    &mut observe_candidate,
                    CONTRACT_ROUNDS,
                    CONTRACT_MIN_OF,
                );
            let verdict =
                common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
            println!(
                "FUSION_DTYPE_RESULT row={row} dtype={dtype} verdict={verdict} \
                 incumbent_median_ms={:.6} candidate_median_ms={:.6} \
                 ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
                 incumbent_null_ci95=[{:.6},{:.6}] candidate_null_ci95=[{:.6},{:.6}] \
                 corrected_dual_null_gate=true incumbent=numpy_live_same_invocation",
                effect.arm_a_median_ns / 1_000_000.0,
                effect.arm_b_median_ns / 1_000_000.0,
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                incumbent_null.ratio_ci_low,
                incumbent_null.ratio_ci_high,
                candidate_null.ratio_ci_low,
                candidate_null.ratio_ci_high,
            );
            let decision = if verdict == "DECIDABLE_WIN" {
                "choose_fnp"
            } else {
                "choose_numpy"
            };
            println!(
                "CHOOSER_STATEMENT workload=fused_multiply_add_{dtype}_32mib \
                 decision={decision} verdict={verdict} \
                 incumbent=numpy_live_same_invocation measured_scope={elements}_equal_shape_c_contiguous_elements_at_{threads}_pinned_threads \
                 outside_scope=run_same_contract_before_choosing"
            );
        }
    });
}

/// End-to-end redecision for the public bool-return surface fed by the
/// `ArrayStorage::Bool` materialization funnel. The internal funnel remains a
/// maintenance-only result; this row times `fnp.greater` as a whole against
/// `numpy.greater` as a whole, with no shared measured component.
/// CLASS-3 MISSING CAPABILITY, end-to-end vs NumPy. Two surfaces NumPy has no
/// fast path for at all, which is where every historical monster in this repo
/// came from. Deliberately not square f64 GEMM.
///
/// 1. `matmul` on int64 — **NumPy has no integer BLAS**, so it falls to a
///    generic loop while we route to the native tiled integer kernel.
/// 2. `char.upper` on a fixed-width ASCII array — **NumPy has no vectorized
///    string case kernel**; it loops per element through the Python-level
///    `numpy.char` layer.
///
/// Both are exactly reproducible (integer arithmetic; byte-wise ASCII mapping),
/// so the arms are byte-identical by construction and the contract's per-round
/// checksum equality holds without tolerance.
///
/// This function also restores measurement for the int64 matmul row: a
/// concurrent edit dropped the original arm, which left a banked incumbent-win
/// whose evidence could not be regenerated from the tree. An unreproducible
/// claim is the frankensearch failure mode in miniature.
fn bench_class3_missing_capability_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module =
            PyModule::new(py, "fnp_python_class3_gaps").expect("class3 missing-capability module");
        fnp_python(&module).expect("initialize fnp_python class3 module");
        let numpy = py.import("numpy").expect("numpy oracle");

        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260728)\n\
                 mm_a = rng.integers(-64, 64, size=(256, 256), dtype=np.int64)\n\
                 mm_b = rng.integers(-64, 64, size=(256, 256), dtype=np.int64)\n\
                 alphabet = np.array(list('abcdefghijklmnopqrstuvwxyz'))\n\
                 idx = rng.integers(0, 26, size=(400_000, 16))\n\
                 up_a = np.array([''.join(row) for row in alphabet[idx]], dtype='U16')\n",
            )
            .expect("class3 setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("class3 corpus setup");

        let checksum_of = |array: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
            let bytes = array
                .call_method0("tobytes")
                .expect("tobytes")
                .extract::<Vec<u8>>()
                .expect("byte Vec");
            bytes
                .iter()
                .fold(0xcbf2_9ce4_8422_2325_u64, |state, &byte| {
                    (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                })
        };

        // --- surface 1: int64 matmul, no integer BLAS ---
        {
            let np_matmul = numpy.getattr("matmul").expect("numpy matmul");
            let fnp_matmul = module.getattr("matmul").expect("fnp matmul");
            common::report_numpy_incumbent_identity(py, "matmul", &np_matmul);
            assert!(
                !np_matmul.is(&fnp_matmul),
                "dispatch trap: incumbent matmul resolved to our callable"
            );
            common::report_incumbent_topology("fnp.matmul", "numpy.matmul");

            let lhs = namespace.get_item("mm_a").expect("mm_a present");
            let rhs = namespace.get_item("mm_b").expect("mm_b present");
            let ours = fnp_matmul.call1((&lhs, &rhs)).expect("fnp matmul parity");
            let theirs = np_matmul.call1((&lhs, &rhs)).expect("numpy matmul parity");
            assert_eq!(
                checksum_of(&ours),
                checksum_of(&theirs),
                "int64 matmul: fnp and numpy disagree"
            );

            let mut time_incumbent = || {
                let started = Instant::now();
                let result = np_matmul
                    .call1((black_box(&lhs), black_box(&rhs)))
                    .expect("numpy matmul arm");
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let mut time_ours = || {
                let started = Instant::now();
                let result = fnp_matmul
                    .call1((black_box(&lhs), black_box(&rhs)))
                    .expect("fnp matmul arm");
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let _ = common::run_median_ci_contract(
                "python_int64_matmul_256_vs_numpy",
                &mut time_incumbent,
                &mut time_ours,
            );
        }

        // --- surface 2: ASCII case mapping, no vectorized string kernel ---
        {
            let np_char = numpy.getattr("char").expect("numpy char namespace");
            let np_upper = np_char.getattr("upper").expect("numpy char.upper");
            let fnp_upper = module
                .getattr("char")
                .expect("fnp char namespace")
                .getattr("upper")
                .expect("fnp char.upper");
            assert!(
                !np_upper.is(&fnp_upper),
                "dispatch trap: incumbent char.upper resolved to our callable"
            );
            common::report_incumbent_topology("fnp.char.upper", "numpy.char.upper");

            let text = namespace.get_item("up_a").expect("up_a present");
            let ours = fnp_upper.call1((&text,)).expect("fnp char.upper parity");
            let theirs = np_upper.call1((&text,)).expect("numpy char.upper parity");
            assert_eq!(
                checksum_of(&ours),
                checksum_of(&theirs),
                "char.upper: fnp and numpy disagree"
            );

            let mut time_incumbent = || {
                let started = Instant::now();
                let result = np_upper
                    .call1((black_box(&text),))
                    .expect("numpy char.upper arm");
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let mut time_ours = || {
                let started = Instant::now();
                let result = fnp_upper
                    .call1((black_box(&text),))
                    .expect("fnp char.upper arm");
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let _ = common::run_median_ci_contract(
                "python_char_upper_ascii_400k_vs_numpy",
                &mut time_incumbent,
                &mut time_ours,
            );
        }
    });
}

/// Re-decide the 2026-07-28 `char.upper` HOLD row under the current campaign
/// contract. The historical invocation had only a NumPy/NumPy null and omitted
/// nested-callable, host, ISA, and observed-thread provenance.
fn bench_char_upper_hold_redecision_median_gate(c: &mut Criterion) {
    let _ = c;
    const ELEMENTS: usize = 400_000;
    const CODEPOINTS_PER_ELEMENT: usize = 16;
    const THREAD_ACTIVITY_REPETITIONS: usize = 11;
    const ROW: &str = "python_char_upper_ascii_u16_400k_hold_redecision";

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_char_upper_redecision")
            .expect("char.upper redecision module");
        fnp_python(&module).expect("initialize fnp_python char.upper redecision module");
        let numpy = py.import("numpy").expect("numpy oracle");
        assert_eq!(
            rayon::current_num_threads(),
            4,
            "char.upper redecision requires the declared four-worker Rayon pool"
        );

        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260728)\n\
                 alphabet = np.array(list('abcdefghijklmnopqrstuvwxyz'))\n\
                 idx = rng.integers(0, 26, size=(400_000, 16))\n\
                 up_a = np.array([''.join(row) for row in alphabet[idx]], dtype='U16')\n",
            )
            .expect("char.upper corpus CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("char.upper corpus setup");

        let np_char = numpy.getattr("char").expect("numpy char namespace");
        let np_upper = np_char.getattr("upper").expect("numpy char.upper");
        let fnp_upper = module
            .getattr("char")
            .expect("fnp char namespace")
            .getattr("upper")
            .expect("fnp char.upper");
        common::report_numpy_incumbent_identity(py, "char.upper", &np_upper);
        assert!(
            !np_upper.is(&fnp_upper),
            "dispatch trap: incumbent char.upper resolved to the candidate callable"
        );
        common::report_incumbent_topology("fnp.char.upper", "numpy.char.upper");
        println!(
            "INCUMBENT_PIPELINE workload=char_upper_ascii_u16 \
             candidate=fnp.char.upper incumbent=numpy.char.upper \
             shared_timed_component=none inputs_shared_read_only=true"
        );

        let text = namespace.get_item("up_a").expect("up_a present");
        let shape = text
            .getattr("shape")
            .expect("char.upper input shape")
            .extract::<Vec<usize>>()
            .expect("char.upper input shape vector");
        let dtype = text
            .getattr("dtype")
            .expect("char.upper input dtype")
            .str()
            .expect("char.upper input dtype string")
            .to_string();
        let c_contiguous = text
            .getattr("flags")
            .expect("char.upper input flags")
            .getattr("c_contiguous")
            .expect("char.upper input C-contiguous flag")
            .extract::<bool>()
            .expect("char.upper input C-contiguous bool");
        assert_eq!(shape, vec![ELEMENTS]);
        assert_eq!(dtype, "<U16");
        assert!(c_contiguous);
        println!(
            "ROUTE_PRECONDITIONS row={ROW} dtype=U16 shape={ELEMENTS} \
             c_contiguous=true ascii_only=true total_codepoints={} \
             native_parallel_gate={} rayon_pool_threads=4 \
             candidate_route=try_zerocopy_unicode_ascii_case",
            ELEMENTS * CODEPOINTS_PER_ELEMENT,
            1 << 20,
        );
        println!(
            "WORK_ACCOUNTING row={ROW} candidate_public_calls=1 incumbent_public_calls=1 \
             candidate_elements={ELEMENTS} incumbent_elements={ELEMENTS} \
             candidate_codepoint_slots={} incumbent_codepoint_slots={} \
             candidate_codepoint_passes=2 incumbent_internal_passes=unmeasured \
             more_work_verdict=no_less_work_claim",
            ELEMENTS * CODEPOINTS_PER_ELEMENT,
            ELEMENTS * CODEPOINTS_PER_ELEMENT,
        );
        println!(
            "COUNTED_MECHANISM row={ROW} class=algorithmic \
             incumbent=numpy_char_per_element_string_case \
             candidate=parallel_ascii_codepoint_scan_plus_map"
        );

        let ours = fnp_upper.call1((&text,)).expect("fnp char.upper parity");
        let theirs = np_upper.call1((&text,)).expect("numpy char.upper parity");
        let candidate_outputs = [ours.clone()];
        let incumbent_outputs = [theirs.clone()];
        assert_workload_outputs_equal(&numpy, ROW, &candidate_outputs, &incumbent_outputs);
        println!(
            "PARITY row={ROW} outputs=1 dtype_shape_byte_identity=passed checksum={:016x}",
            workload_checksum(&numpy, &candidate_outputs),
        );

        common::report_observed_thread_activity(ROW, "numpy", THREAD_ACTIVITY_REPETITIONS, || {
            let output = np_upper
                .call1((black_box(&text),))
                .expect("numpy char.upper thread probe");
            black_box(output);
        });
        common::report_observed_thread_activity(ROW, "fnp", THREAD_ACTIVITY_REPETITIONS, || {
            let output = fnp_upper
                .call1((black_box(&text),))
                .expect("fnp char.upper thread probe");
            black_box(output);
        });

        let mut observe_incumbent = || {
            let started = Instant::now();
            let output = np_upper
                .call1((black_box(&text),))
                .expect("numpy char.upper arm");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: workload_checksum(&numpy, &[output]),
            }
        };
        let mut observe_candidate = || {
            let started = Instant::now();
            let output = fnp_upper
                .call1((black_box(&text),))
                .expect("fnp char.upper arm");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: workload_checksum(&numpy, &[output]),
            }
        };
        let (effect, incumbent_null, candidate_null) = common::run_dual_null_median_ci_contract(
            ROW,
            &mut observe_incumbent,
            &mut observe_candidate,
        );
        let verdict = common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
        println!(
            "HOLD_REDECISION row={ROW} verdict={verdict} \
             incumbent_median_ms={:.6} candidate_median_ms={:.6} \
             ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
             incumbent_null_ratio={:.6} candidate_null_ratio={:.6} \
             effect_ci_excludes_one={} corrected_dual_null_gate=true \
             median_clause=true",
            effect.arm_a_median_ns / 1_000_000.0,
            effect.arm_b_median_ns / 1_000_000.0,
            effect.ratio_median,
            effect.ratio_ci_low,
            effect.ratio_ci_high,
            incumbent_null.ratio_median,
            candidate_null.ratio_median,
            effect.ratio_ci_low > 1.0 || effect.ratio_ci_high < 1.0,
        );
    });
}

/// Re-decide the exact 256x256 int64 `matmul` HOLD under the corrected
/// dual-null contract. The historical row had a NumPy/NumPy null, but no
/// FNP/FNP null; three independent ELFs consequently reported a stable NumPy
/// arm and a candidate arm that moved from about 0.83 ms to 1.37 ms.
fn bench_int64_matmul_hold_redecision_median_gate(c: &mut Criterion) {
    let _ = c;
    const DIMENSION: usize = 256;
    const THREAD_ACTIVITY_REPETITIONS: usize = 101;
    const THREADS: &str = "4";
    const REQUIRED_BUILD_PROFILE: &str = "release-perf";
    const ROW: &str = "python_int64_matmul_256_hold_redecision";

    assert_eq!(
        std::env::var("FNP_BENCH_PROFILE").as_deref(),
        Ok(REQUIRED_BUILD_PROFILE),
        "ship-grade int64 matmul evidence requires FNP_BENCH_PROFILE=release-perf"
    );
    for variable in [
        "RAYON_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(THREADS),
            "{variable} must be explicitly pinned before int64 matmul timing"
        );
    }
    assert_eq!(
        rayon::current_num_threads(),
        THREADS
            .parse::<usize>()
            .expect("thread count constant is numeric"),
        "Rayon pool width does not match the pinned int64 matmul configuration"
    );
    println!(
        "WORKLOAD_RUNTIME host={} build_profile={REQUIRED_BUILD_PROFILE} \
         rayon_threads={} RAYON_NUM_THREADS={THREADS} OPENBLAS_NUM_THREADS={THREADS} \
         OMP_NUM_THREADS={THREADS} MKL_NUM_THREADS={THREADS}",
        std::env::var("HOSTNAME").unwrap_or_else(|_| "unavailable".to_owned()),
        rayon::current_num_threads(),
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_int64_matmul_redecision")
            .expect("int64 matmul redecision module");
        fnp_python(&module).expect("initialize fnp_python int64 matmul redecision module");
        let numpy = py.import("numpy").expect("numpy incumbent");

        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260727)\n\
                 mm_a = rng.integers(-64, 64, size=(256, 256), dtype=np.int64)\n\
                 mm_b = rng.integers(-64, 64, size=(256, 256), dtype=np.int64)\n",
            )
            .expect("int64 matmul corpus CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("int64 matmul corpus setup");

        let np_matmul = numpy.getattr("matmul").expect("numpy matmul");
        let fnp_matmul = module.getattr("matmul").expect("fnp matmul");
        common::report_numpy_incumbent_identity(py, "matmul", &np_matmul);
        assert!(
            !np_matmul.is(&fnp_matmul),
            "dispatch trap: incumbent matmul resolved to the candidate callable"
        );
        common::report_incumbent_topology("fnp.matmul", "numpy.matmul");
        println!(
            "INCUMBENT_PIPELINE workload=int64_matmul_256 \
             candidate=fnp.matmul incumbent=numpy.matmul \
             shared_timed_component=none inputs_shared_read_only=true"
        );

        let lhs = namespace.get_item("mm_a").expect("mm_a present");
        let rhs = namespace.get_item("mm_b").expect("mm_b present");
        for (name, operand) in [("lhs", &lhs), ("rhs", &rhs)] {
            let shape = operand
                .getattr("shape")
                .expect("matmul operand shape")
                .extract::<Vec<usize>>()
                .expect("matmul operand shape vector");
            let dtype = operand
                .getattr("dtype")
                .expect("matmul operand dtype")
                .str()
                .expect("matmul operand dtype string")
                .to_string();
            let c_contiguous = operand
                .getattr("flags")
                .expect("matmul operand flags")
                .getattr("c_contiguous")
                .expect("matmul operand C-contiguous flag")
                .extract::<bool>()
                .expect("matmul operand C-contiguous bool");
            assert_eq!(shape, vec![DIMENSION, DIMENSION], "{name} shape");
            assert_eq!(dtype, "int64", "{name} dtype");
            assert!(c_contiguous, "{name} must be C-contiguous");
        }
        println!(
            "ROUTE_PRECONDITIONS row={ROW} dtype=int64 lhs_shape=256x256 \
             rhs_shape=256x256 c_contiguous=true same_dtype=true \
             matching_inner_dim={DIMENSION} matmul_work={} int_matmul_min_work={} \
             rayon_threads_min=2 candidate_route=native_int64_tiled_gemm \
             source_pin=fnp-python/src/lib.rs:try_native_int_matmul",
            DIMENSION * DIMENSION * DIMENSION,
            1 << 18,
        );
        println!(
            "WORK_ACCOUNTING row={ROW} candidate_public_calls=1 incumbent_public_calls=1 \
             candidate_scalar_multiply_accumulates={} \
             incumbent_scalar_multiply_accumulates={} \
             candidate_output_elements={} incumbent_output_elements={} \
             integer_bound_max_abs_product_sum={} overflow_headroom=proven",
            DIMENSION * DIMENSION * DIMENSION,
            DIMENSION * DIMENSION * DIMENSION,
            DIMENSION * DIMENSION,
            DIMENSION * DIMENSION,
            DIMENSION * 64 * 64,
        );
        println!(
            "COUNTED_MECHANISM row={ROW} class=algorithmic \
             incumbent=generic_integer_matrix_multiplication \
             candidate=tiled_register_blocked_integer_matrix_multiplication"
        );

        let ours = fnp_matmul
            .call1((&lhs, &rhs))
            .expect("fnp int64 matmul parity");
        let theirs = np_matmul
            .call1((&lhs, &rhs))
            .expect("numpy int64 matmul parity");
        let candidate_outputs = [ours.clone()];
        let incumbent_outputs = [theirs.clone()];
        assert_workload_outputs_equal(&numpy, ROW, &candidate_outputs, &incumbent_outputs);
        println!(
            "PARITY row={ROW} outputs=1 dtype_shape_byte_identity=passed checksum={:016x}",
            workload_checksum(&numpy, &candidate_outputs),
        );

        common::report_observed_thread_activity(ROW, "numpy", THREAD_ACTIVITY_REPETITIONS, || {
            let output = np_matmul
                .call1((black_box(&lhs), black_box(&rhs)))
                .expect("numpy int64 matmul thread probe");
            black_box(output);
        });
        common::report_observed_thread_activity(ROW, "fnp", THREAD_ACTIVITY_REPETITIONS, || {
            let output = fnp_matmul
                .call1((black_box(&lhs), black_box(&rhs)))
                .expect("fnp int64 matmul thread probe");
            black_box(output);
        });

        let mut observe_incumbent = || {
            let started = Instant::now();
            let output = np_matmul
                .call1((black_box(&lhs), black_box(&rhs)))
                .expect("numpy int64 matmul arm");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: workload_checksum(&numpy, &[output]),
            }
        };
        let mut observe_candidate = || {
            let started = Instant::now();
            let output = fnp_matmul
                .call1((black_box(&lhs), black_box(&rhs)))
                .expect("fnp int64 matmul arm");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: workload_checksum(&numpy, &[output]),
            }
        };
        let (effect, incumbent_null, candidate_null) = common::run_dual_null_median_ci_contract(
            ROW,
            &mut observe_incumbent,
            &mut observe_candidate,
        );
        let verdict = common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
        println!(
            "HOLD_REDECISION row={ROW} verdict={verdict} \
             incumbent_median_ms={:.6} candidate_median_ms={:.6} \
             ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
             incumbent_null_ratio={:.6} incumbent_null_ci95=[{:.6},{:.6}] \
             candidate_null_ratio={:.6} candidate_null_ci95=[{:.6},{:.6}] \
             effect_ci_excludes_one={} corrected_dual_null_gate=true \
             median_clause=true actual_threads_reported=true",
            effect.arm_a_median_ns / 1_000_000.0,
            effect.arm_b_median_ns / 1_000_000.0,
            effect.ratio_median,
            effect.ratio_ci_low,
            effect.ratio_ci_high,
            incumbent_null.ratio_median,
            incumbent_null.ratio_ci_low,
            incumbent_null.ratio_ci_high,
            candidate_null.ratio_median,
            candidate_null.ratio_ci_low,
            candidate_null.ratio_ci_high,
            effect.ratio_ci_low > 1.0 || effect.ratio_ci_high < 1.0,
        );
        let (decision, reason) = match verdict {
            "DECIDABLE_WIN" => ("choose_fnp", "corrected_dual_null_incumbent_win"),
            "DECIDABLE_REGRESSION" => ("choose_numpy", "corrected_dual_null_regression"),
            _ => (
                "choose_numpy",
                "effect_not_separated_from_dual_null_envelope",
            ),
        };
        println!(
            "CHOOSER_STATEMENT workload=int64_matmul_256x256 decision={decision} \
             reason={reason} incumbent=numpy_live_same_invocation \
             measured_scope=int64_c_contiguous_256x256_at_four_configured_threads \
             outside_scope=run_same_contract_before_choosing"
        );
    });
}

/// Convert the banked `fnp_io::tofile_text` maintenance win into a complete
/// public-job comparison. A million-element delimited integer snapshot is a
/// recognizable NumPy workload: one call opens a path, formats every value,
/// writes the complete interoperable text payload, and closes the file.
fn bench_int64_tofile_text_snapshot_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;
    const ELEMENTS: usize = 1_000_000;
    const THREAD_ACTIVITY_REPETITIONS: usize = 3;
    const THREADS: &str = "4";
    const REQUIRED_BUILD_PROFILE: &str = "release-perf";
    const ROW: &str = "python_int64_tofile_text_snapshot_1m_vs_numpy";

    assert_eq!(
        std::env::var("FNP_BENCH_PROFILE").as_deref(),
        Ok(REQUIRED_BUILD_PROFILE),
        "ship-grade tofile evidence requires FNP_BENCH_PROFILE=release-perf"
    );
    for variable in [
        "RAYON_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(THREADS),
            "{variable} must be explicitly pinned before tofile timing"
        );
    }
    assert_eq!(
        rayon::current_num_threads(),
        THREADS
            .parse::<usize>()
            .expect("thread count constant is numeric"),
        "Rayon pool width does not match the pinned tofile configuration"
    );

    fn file_checksum(path: &str) -> u64 {
        let bytes = std::fs::read(path).expect("read completed tofile payload");
        bytes
            .len()
            .to_le_bytes()
            .iter()
            .chain(bytes.iter())
            .fold(0xcbf2_9ce4_8422_2325_u64, |state, &byte| {
                (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
            })
    }

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_int64_tofile_snapshot")
            .expect("int64 tofile snapshot module");
        fnp_python(&module).expect("initialize fnp_python int64 tofile snapshot module");
        let numpy = py.import("numpy").expect("numpy incumbent");

        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 tofile_values = ((np.arange(1_000_000, dtype=np.int64) * 1_000_003) \
                     % 1_999_999_999) - 999_999_999\n",
            )
            .expect("int64 tofile corpus CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("int64 tofile corpus setup");
        let values = namespace
            .get_item("tofile_values")
            .expect("tofile_values present");
        let shape = values
            .getattr("shape")
            .expect("tofile values shape")
            .extract::<Vec<usize>>()
            .expect("tofile values shape vector");
        let dtype = values
            .getattr("dtype")
            .expect("tofile values dtype")
            .str()
            .expect("tofile values dtype string")
            .to_string();
        let c_contiguous = values
            .getattr("flags")
            .expect("tofile values flags")
            .getattr("c_contiguous")
            .expect("tofile values C-contiguous flag")
            .extract::<bool>()
            .expect("tofile values C-contiguous bool");
        let minimum = values
            .call_method0("min")
            .expect("tofile values minimum")
            .extract::<i64>()
            .expect("tofile values minimum i64");
        let maximum = values
            .call_method0("max")
            .expect("tofile values maximum")
            .extract::<i64>()
            .expect("tofile values maximum i64");
        assert_eq!(shape, vec![ELEMENTS]);
        assert_eq!(dtype, "int64");
        assert!(c_contiguous);
        assert!(
            minimum.unsigned_abs() < 1_000_000_000_000_000
                && maximum.unsigned_abs() < 1_000_000_000_000_000,
            "tofile fixture must stay inside the exact native integer guard"
        );

        let ndarray_type = numpy.getattr("ndarray").expect("numpy.ndarray");
        let np_tofile_descriptor = ndarray_type
            .getattr("tofile")
            .expect("numpy.ndarray.tofile descriptor");
        common::report_numpy_incumbent_identity(py, "ndarray.tofile", &np_tofile_descriptor);
        let np_tofile = values
            .getattr("tofile")
            .expect("bound numpy ndarray.tofile");
        let receiver = np_tofile
            .getattr("__self__")
            .expect("bound numpy ndarray.tofile receiver");
        assert!(
            receiver.is(&values),
            "measured incumbent method is bound to a different ndarray"
        );
        let rebound = np_tofile_descriptor
            .call_method1("__get__", (&values, &ndarray_type))
            .expect("rebind numpy.ndarray.tofile descriptor");
        assert!(
            rebound
                .eq(&np_tofile)
                .expect("compare rebound numpy ndarray.tofile"),
            "measured incumbent method differs from numpy.ndarray.tofile descriptor"
        );
        let fnp_tofile = module.getattr("tofile").expect("fnp tofile");
        assert!(
            !np_tofile_descriptor.is(&fnp_tofile),
            "dispatch trap: incumbent tofile descriptor resolved to the candidate callable"
        );
        common::report_incumbent_topology("fnp.tofile", "numpy.ndarray.tofile");
        println!(
            "INCUMBENT_PIPELINE workload=int64_tofile_text_snapshot \
             candidate=fnp.tofile incumbent=numpy.ndarray.tofile \
             shared_timed_component=none inputs_shared_read_only=true \
             destination_contract=separate_same_filesystem_paths"
        );

        let invocation_id = common::bench_invocation_id();
        let incumbent_path = format!("/data/tmp/fnp-tofile-{invocation_id}-numpy-incumbent.txt");
        let candidate_path = format!("/data/tmp/fnp-tofile-{invocation_id}-candidate.txt");
        let kwargs = PyDict::new(py);
        kwargs.set_item("sep", ",").expect("set tofile separator");

        let run_incumbent = || {
            np_tofile
                .call((incumbent_path.as_str(),), Some(&kwargs))
                .expect("numpy int64 tofile snapshot");
        };
        let run_candidate = || {
            fnp_tofile
                .call((&values, candidate_path.as_str()), Some(&kwargs))
                .expect("fnp int64 tofile snapshot");
        };

        run_incumbent();
        run_candidate();
        let incumbent_bytes =
            std::fs::read(&incumbent_path).expect("read NumPy tofile parity payload");
        let candidate_bytes =
            std::fs::read(&candidate_path).expect("read FNP tofile parity payload");
        assert_eq!(
            candidate_bytes, incumbent_bytes,
            "public int64 tofile payload differs from live NumPy"
        );
        let output_bytes = candidate_bytes.len();
        let parity_checksum = file_checksum(&candidate_path);
        println!(
            "PARITY row={ROW} outputs=1 exact_file_bytes=passed \
             output_bytes={output_bytes} checksum={parity_checksum:016x}"
        );
        println!(
            "ROUTE_PRECONDITIONS row={ROW} dtype=int64 shape={ELEMENTS} \
             c_contiguous=true separator=comma format=percent_s \
             min_value={minimum} max_value={maximum} \
             exact_integer_limit=1000000000000000 destination=unicode_path \
             candidate_route=fnp_io::tofile_text \
             source_pin=fnp-python/src/lib.rs:tofile"
        );
        println!(
            "WORK_ACCOUNTING row={ROW} candidate_public_calls=1 incumbent_public_calls=1 \
             candidate_elements={ELEMENTS} incumbent_elements={ELEMENTS} \
             candidate_output_bytes={output_bytes} incumbent_output_bytes={output_bytes} \
             candidate_open_truncate_close=1 incumbent_open_truncate_close=1 \
             candidate_write_all_calls=1 incumbent_value_fwrite_calls={ELEMENTS} \
             incumbent_separator_writes={} equal_payload=true",
            ELEMENTS - 1,
        );
        println!(
            "COUNTED_MECHANISM row={ROW} class=object_and_write_call_elimination \
             incumbent_per_element_scalar_objects={ELEMENTS} \
             incumbent_per_element_string_objects={ELEMENTS} \
             incumbent_per_element_ascii_byte_objects={ELEMENTS} \
             incumbent_fwrite_calls={} candidate_python_scalar_objects=0 \
             candidate_python_string_objects=0 candidate_python_byte_objects=0 \
             candidate_write_all_calls=1 kernel_syscall_count=not_measured \
             incumbent_source=legacy_numpy_code/numpy/numpy/_core/src/multiarray/convert.c:PyArray_ToFile",
            ELEMENTS * 2 - 1,
        );
        println!(
            "WORKLOAD_REPRESENTATIVENESS row={ROW} \
             job=delimited_integer_snapshot_export \
             rationale=users_persist_ids_counters_and_event_snapshots_as_interoperable_text \
             whole_job=open_plus_format_all_values_plus_write_payload_plus_close"
        );
        println!(
            "WORKLOAD_SCRATCH row={ROW} incumbent_path={incumbent_path} \
             candidate_path={candidate_path} cleanup_owner=BlackThrush"
        );

        common::report_observed_thread_activity(ROW, "numpy", THREAD_ACTIVITY_REPETITIONS, || {
            run_incumbent();
            black_box(file_checksum(&incumbent_path));
        });
        common::report_observed_thread_activity(ROW, "fnp", THREAD_ACTIVITY_REPETITIONS, || {
            run_candidate();
            black_box(file_checksum(&candidate_path));
        });

        let mut observe_incumbent = || {
            let started = Instant::now();
            run_incumbent();
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: file_checksum(&incumbent_path),
            }
        };
        let mut observe_candidate = || {
            let started = Instant::now();
            run_candidate();
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: file_checksum(&candidate_path),
            }
        };
        let (effect, incumbent_null, candidate_null) = common::run_dual_null_median_ci_contract(
            ROW,
            &mut observe_incumbent,
            &mut observe_candidate,
        );
        let verdict = common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
        println!(
            "WHOLE_JOB_RESULT row={ROW} verdict={verdict} \
             incumbent_median_ms={:.6} candidate_median_ms={:.6} \
             ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
             incumbent_null_ratio={:.6} incumbent_null_ci95=[{:.6},{:.6}] \
             candidate_null_ratio={:.6} candidate_null_ci95=[{:.6},{:.6}] \
             effect_ci_excludes_one={} corrected_dual_null_gate=true \
             median_clause=true actual_threads_reported=true",
            effect.arm_a_median_ns / 1_000_000.0,
            effect.arm_b_median_ns / 1_000_000.0,
            effect.ratio_median,
            effect.ratio_ci_low,
            effect.ratio_ci_high,
            incumbent_null.ratio_median,
            incumbent_null.ratio_ci_low,
            incumbent_null.ratio_ci_high,
            candidate_null.ratio_median,
            candidate_null.ratio_ci_low,
            candidate_null.ratio_ci_high,
            effect.ratio_ci_low > 1.0 || effect.ratio_ci_high < 1.0,
        );
        let (decision, reason) = match verdict {
            "DECIDABLE_WIN" => ("choose_fnp", "corrected_dual_null_incumbent_win"),
            "DECIDABLE_REGRESSION" => ("choose_numpy", "corrected_dual_null_regression"),
            _ => (
                "choose_numpy",
                "effect_not_separated_from_dual_null_envelope",
            ),
        };
        println!(
            "CHOOSER_STATEMENT workload=int64_tofile_text_snapshot_1m \
             decision={decision} reason={reason} \
             incumbent=numpy_live_same_invocation \
             measured_scope=int64_c_contiguous_one_million_values_comma_sep_percent_s_path \
             outside_scope=run_same_contract_before_choosing"
        );
    });
}

fn bench_bool_public_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_bool_public").expect("bool-public bench module");
        fnp_python(&module).expect("initialize fnp_python bool-public module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let np_greater = numpy.getattr("greater").expect("numpy greater");
        let fnp_greater = module.getattr("greater").expect("fnp greater");
        common::report_numpy_incumbent_identity(py, "greater", &np_greater);
        assert!(
            !np_greater.is(&fnp_greater),
            "dispatch trap: incumbent greater resolved to our callable"
        );
        common::report_incumbent_topology("fnp.greater", "numpy.greater");

        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260727)\n\
                 cmp_a = rng.standard_normal(8_000_000)\n\
                 cmp_b = rng.standard_normal(8_000_000)\n",
            )
            .expect("bool-public setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("bool-public corpus setup");

        let checksum_of = |array: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
            let bytes = array
                .call_method0("tobytes")
                .expect("tobytes")
                .extract::<Vec<u8>>()
                .expect("byte Vec");
            bytes
                .iter()
                .fold(0xcbf2_9ce4_8422_2325_u64, |state, &byte| {
                    (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                })
        };

        let lhs = namespace.get_item("cmp_a").expect("cmp_a present");
        let rhs = namespace.get_item("cmp_b").expect("cmp_b present");
        let ours = fnp_greater.call1((&lhs, &rhs)).expect("fnp greater parity");
        let theirs = np_greater
            .call1((&lhs, &rhs))
            .expect("numpy greater parity");
        assert_eq!(
            checksum_of(&ours),
            checksum_of(&theirs),
            "fnp.greater and numpy.greater disagree",
        );

        let mut time_incumbent = || {
            let started = Instant::now();
            let result = np_greater
                .call1((black_box(&lhs), black_box(&rhs)))
                .expect("numpy greater arm");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: checksum_of(&result),
            }
        };
        let mut time_ours = || {
            let started = Instant::now();
            let result = fnp_greater
                .call1((black_box(&lhs), black_box(&rhs)))
                .expect("fnp greater arm");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: checksum_of(&result),
            }
        };
        let _ = common::run_median_ci_contract(
            "python_greater_f64_8m_bool_out_vs_numpy",
            &mut time_incumbent,
            &mut time_ours,
        );
    });
}

/// Class-3 missing-capability candidate: NumPy has no f16 ALU. For a large
/// finite C-contiguous last-axis reduction, its unweighted `average` is one f32
/// pairwise sum and division per lane. FrankenNumPy reproduces that exact tree
/// while parallelizing across independent lanes.
fn bench_average_f16_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_average_f16").expect("average-f16 bench module");
        fnp_python(&module).expect("initialize fnp_python average-f16 module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let np_average = numpy.getattr("average").expect("numpy average");
        let fnp_average = module.getattr("average").expect("fnp average");
        common::report_numpy_incumbent_identity(py, "average", &np_average);
        assert!(
            !np_average.is(&fnp_average),
            "dispatch trap: incumbent average resolved to our callable"
        );
        common::report_incumbent_topology("fnp.average", "numpy.average");

        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260727)\n\
                 avg_f16 = rng.uniform(-3.0, 3.0, size=(2048, 4096)).astype(np.float16)\n",
            )
            .expect("average-f16 setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("average-f16 corpus setup");
        let input = namespace.get_item("avg_f16").expect("avg_f16 present");

        // Profile the actual incumbent workload before timing the proposed
        // routing change. This names the Python frames and supplies the
        // cumulative-time denominator used for the Amdahl ceiling; the C ufunc
        // body is charged to NumPy's `_methods._mean` call.
        let profile_namespace = PyDict::new(py);
        profile_namespace
            .set_item("np_average", &np_average)
            .expect("profile average callable");
        profile_namespace
            .set_item("avg_f16", &input)
            .expect("profile average input");
        py.run(
            std::ffi::CString::new(
                "import cProfile, io, pstats\n\
                 profiler = cProfile.Profile()\n\
                 profiler.enable()\n\
                 profile_results = [np_average(avg_f16, axis=-1) for _ in range(8)]\n\
                 profiler.disable()\n\
                 profile_stream = io.StringIO()\n\
                 pstats.Stats(profiler, stream=profile_stream).sort_stats('cumulative').print_stats(8)\n\
                 profile_report = profile_stream.getvalue()\n",
            )
            .expect("average-f16 profile CString")
            .as_c_str(),
            Some(&profile_namespace),
            Some(&profile_namespace),
        )
        .expect("profile NumPy average");
        println!(
            "PROFILE_ATTRIBUTION surface=numpy.average(float16,2048x4096,axis=-1) repeats=8\n{}",
            profile_namespace
                .get_item("profile_report")
                .expect("profile report present")
                .extract::<String>()
                .expect("profile report string")
        );

        let checksum_of = |value: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
            let bytes = value
                .call_method0("tobytes")
                .expect("tobytes")
                .extract::<Vec<u8>>()
                .expect("byte Vec");
            bytes
                .iter()
                .fold(0xcbf2_9ce4_8422_2325_u64, |state, &byte| {
                    (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                })
        };

        let ours = fnp_average
            .call1((&input, -1_i64))
            .expect("fnp average parity");
        let theirs = np_average
            .call1((&input, -1_i64))
            .expect("numpy average parity");
        assert_eq!(
            checksum_of(&ours),
            checksum_of(&theirs),
            "fnp.average and numpy.average disagree",
        );

        let mut time_incumbent = || {
            let started = Instant::now();
            let result = np_average
                .call1((black_box(&input), -1_i64))
                .expect("numpy average arm");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: checksum_of(&result),
            }
        };
        let mut time_ours = || {
            let started = Instant::now();
            let result = fnp_average
                .call1((black_box(&input), -1_i64))
                .expect("fnp average arm");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: checksum_of(&result),
            }
        };
        let _ = common::run_median_ci_contract(
            "python_average_f16_2048x4096_axis_last_vs_numpy",
            &mut time_incumbent,
            &mut time_ours,
        );
    });
}

/// Class-3 missing-capability gate for the bounded float16 domain. NumPy has no
/// half-precision order-statistics kernel: its exact public arm copies the input
/// and partitions it, while the candidate uses one 65,536-bin histogram.
fn bench_quantile_f16_histogram_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module =
            PyModule::new(py, "fnp_python_quantile_f16").expect("quantile-f16 bench module");
        fnp_python(&module).expect("initialize fnp_python quantile-f16 module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let np_quantile = numpy.getattr("quantile").expect("numpy quantile");
        let fnp_quantile = module.getattr("quantile").expect("fnp quantile");
        common::report_numpy_incumbent_identity(py, "quantile", &np_quantile);
        assert!(
            !np_quantile.is(&fnp_quantile),
            "dispatch trap: incumbent quantile resolved to our callable"
        );
        common::report_incumbent_topology("fnp.quantile", "numpy.quantile");

        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "import numpy as np\n\
                 rng = np.random.default_rng(20260728)\n\
                 quantile_f16 = rng.uniform(-200.0, 200.0, size=8_000_000).astype(np.float16)\n\
                 quantile_qs = np.array([0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999], dtype=np.float64)\n",
            )
            .expect("quantile-f16 setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("quantile-f16 corpus setup");
        let input = namespace
            .get_item("quantile_f16")
            .expect("quantile_f16 present");
        let qs = namespace
            .get_item("quantile_qs")
            .expect("quantile_qs present");

        let profile_namespace = PyDict::new(py);
        profile_namespace
            .set_item("np_quantile", &np_quantile)
            .expect("profile quantile callable");
        profile_namespace
            .set_item("quantile_f16", &input)
            .expect("profile quantile input");
        profile_namespace
            .set_item("quantile_qs", &qs)
            .expect("profile quantiles");
        py.run(
            std::ffi::CString::new(
                "import cProfile, io, pstats\n\
                 profiler = cProfile.Profile()\n\
                 profiler.enable()\n\
                 profile_results = [np_quantile(quantile_f16, quantile_qs) for _ in range(5)]\n\
                 profiler.disable()\n\
                 profile_stream = io.StringIO()\n\
                 pstats.Stats(profiler, stream=profile_stream).strip_dirs().sort_stats('tottime').print_stats(8)\n\
                 profile_report = profile_stream.getvalue()\n",
            )
            .expect("quantile-f16 profile CString")
            .as_c_str(),
            Some(&profile_namespace),
            Some(&profile_namespace),
        )
        .expect("profile NumPy quantile");
        println!(
            "PROFILE_ATTRIBUTION surface=numpy.quantile(float16[8m],float64[9]) repeats=5 \
             target_frame=ndarray.partition self_time_fraction=profile_report \
             amdahl_ceiling=total_time/(total_time-partition_self_time)\n{}",
            profile_namespace
                .get_item("profile_report")
                .expect("profile report present")
                .extract::<String>()
                .expect("profile report string")
        );

        let checksum_of = |value: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
            let bytes = value
                .call_method0("tobytes")
                .expect("tobytes")
                .extract::<Vec<u8>>()
                .expect("byte Vec");
            bytes
                .iter()
                .fold(0xcbf2_9ce4_8422_2325_u64, |state, &byte| {
                    (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                })
        };

        let ours = fnp_quantile
            .call1((&input, &qs))
            .expect("fnp quantile parity");
        let theirs = np_quantile
            .call1((&input, &qs))
            .expect("numpy quantile parity");
        assert_eq!(
            ours.getattr("dtype")
                .expect("fnp dtype")
                .str()
                .expect("fnp dtype string")
                .to_string(),
            theirs
                .getattr("dtype")
                .expect("numpy dtype")
                .str()
                .expect("numpy dtype string")
                .to_string(),
            "f16 multi-quantile dtype mismatch"
        );
        assert_eq!(
            checksum_of(&ours),
            checksum_of(&theirs),
            "f16 multi-quantile byte mismatch"
        );

        let mut time_incumbent = || {
            let started = Instant::now();
            let result = np_quantile
                .call1((black_box(&input), black_box(&qs)))
                .expect("numpy quantile arm");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: checksum_of(&result),
            }
        };
        let mut time_ours = || {
            let started = Instant::now();
            let result = fnp_quantile
                .call1((black_box(&input), black_box(&qs)))
                .expect("fnp quantile arm");
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: checksum_of(&result),
            }
        };
        let _ = common::run_median_ci_contract(
            "python_quantile_f16_8m_q9_vs_numpy",
            &mut time_incumbent,
            &mut time_ours,
        );
    });
}

/// Re-decide the 25k-row wide-CSV whole-job parity result at the exact
/// 100k-row threshold in its retry predicate. This is the complete job a
/// sensor-data user runs: parse eight tail columns from a wide on-disk batch,
/// build a global histogram, and report per-channel standard deviations.
fn bench_wide_csv_sensor_etl_retry_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    const ROW: &str = "workload_wide_csv_sensor_etl_retry_100000x48_use8";
    const CSV_ROWS: usize = 100_000;
    const CSV_COLUMNS: usize = 48;
    const CSV_USECOLS: [i64; 8] = [-1, -3, -5, -7, -9, -11, -13, -15];
    const HISTOGRAM_BINS: usize = 64;
    const THREAD_ACTIVITY_REPETITIONS: usize = 3;
    const THREADS: &str = "4";
    const REQUIRED_BUILD_PROFILE: &str = "release-perf";

    assert_eq!(
        std::env::var("FNP_BENCH_PROFILE").as_deref(),
        Ok(REQUIRED_BUILD_PROFILE),
        "ship-grade wide-CSV evidence requires FNP_BENCH_PROFILE=release-perf"
    );
    let build_worker =
        std::env::var("FNP_BUILD_WORKER").expect("FNP_BUILD_WORKER records the RCH build origin");
    assert!(
        !build_worker.trim().is_empty(),
        "FNP_BUILD_WORKER must name the RCH build worker"
    );
    for variable in [
        "RAYON_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(THREADS),
            "{variable} must be explicitly pinned before wide-CSV timing"
        );
    }
    assert_eq!(
        rayon::current_num_threads(),
        THREADS
            .parse::<usize>()
            .expect("thread count constant is numeric"),
        "Rayon pool width does not match the pinned wide-CSV configuration"
    );
    let host = std::env::var("HOSTNAME").unwrap_or_else(|_| "unavailable".to_owned());
    println!(
        "WORKLOAD_RUNTIME workload=wide_csv_sensor_etl_retry host={host} \
         build_worker={build_worker} build_profile={REQUIRED_BUILD_PROFILE} \
         rayon_threads={} RAYON_NUM_THREADS={} OPENBLAS_NUM_THREADS={} \
         OMP_NUM_THREADS={} MKL_NUM_THREADS={}",
        rayon::current_num_threads(),
        THREADS,
        THREADS,
        THREADS,
        THREADS,
    );

    let csv_path = format!(
        "/data/tmp/fnp-wide-csv-etl-retry-{}.csv",
        common::bench_invocation_id()
    );
    let mut csv = String::with_capacity(CSV_ROWS * CSV_COLUMNS * 10);
    for row in 0..CSV_ROWS {
        for column in 0..CSV_COLUMNS {
            if column > 0 {
                csv.push(',');
            }
            let raw = ((row * 7_919 + column * 104_729) % 1_000_000) as i64 - 500_000;
            write!(&mut csv, "{:.3}", raw as f64 / 1_000.0)
                .expect("writing the wide-CSV corpus to a String cannot fail");
        }
        csv.push('\n');
    }
    let csv_bytes = csv.len();
    let csv_checksum = csv
        .as_bytes()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |state, &byte| {
            (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        });
    std::fs::write(&csv_path, csv).expect("write 100k-row wide-CSV corpus");
    let warm_page_cache = std::fs::read(&csv_path).expect("warm wide-CSV page cache");
    assert_eq!(warm_page_cache.len(), csv_bytes);
    black_box(warm_page_cache.len());
    drop(warm_page_cache);
    println!(
        "WORKLOAD_INPUT row={ROW} path={csv_path} rows={CSV_ROWS} \
         columns={CSV_COLUMNS} selected_columns={} selected_fraction={:.6} \
         csv_bytes={csv_bytes} csv_checksum={csv_checksum:016x} \
         same_path_per_pair=true page_cache=prewarmed",
        CSV_USECOLS.len(),
        CSV_USECOLS.len() as f64 / CSV_COLUMNS as f64,
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_wide_csv_sensor_etl_retry")
            .expect("wide-CSV retry module");
        fnp_python(&module).expect("initialize fnp_python wide-CSV retry module");
        let numpy = py.import("numpy").expect("numpy incumbent");

        let fnp_loadtxt = module.getattr("loadtxt").expect("fnp loadtxt");
        let np_loadtxt = numpy.getattr("loadtxt").expect("numpy loadtxt");
        let fnp_histogram = module.getattr("histogram").expect("fnp histogram");
        let np_histogram = numpy.getattr("histogram").expect("numpy histogram");
        let fnp_std = module.getattr("std").expect("fnp std");
        let np_std = numpy.getattr("std").expect("numpy std");
        for (candidate, incumbent, surface) in [
            (&fnp_loadtxt, &np_loadtxt, "loadtxt"),
            (&fnp_histogram, &np_histogram, "histogram"),
            (&fnp_std, &np_std, "std"),
        ] {
            assert!(
                !candidate.is(incumbent),
                "dispatch trap: {surface} incumbent resolved to the FNP callable"
            );
        }
        common::report_numpy_loadtxt_incumbent_identity(py, &np_loadtxt);
        common::report_numpy_incumbent_identity(py, "histogram", &np_histogram);
        common::report_numpy_incumbent_identity(py, "std", &np_std);
        common::report_incumbent_topology(
            "fnp.workload.wide_csv_sensor_etl_retry",
            "numpy.workload.wide_csv_sensor_etl_retry",
        );
        println!(
            "INCUMBENT_ISOLATION_PROOF workload=wide_csv_sensor_etl_retry \
             candidate=fnp.loadtxt+fnp.histogram+fnp.std \
             incumbent=numpy.loadtxt+numpy.histogram+numpy.std \
             callable_identity_distinct=passed \
             shared_timed_component=none candidate_stages_all_native=true \
             inputs_shared_read_only=true same_path=true result_cache=none"
        );

        let loadtxt_kwargs = PyDict::new(py);
        loadtxt_kwargs
            .set_item("delimiter", ",")
            .expect("CSV delimiter kwarg");
        loadtxt_kwargs
            .set_item("dtype", numpy.getattr("float64").expect("numpy float64"))
            .expect("CSV dtype kwarg");
        loadtxt_kwargs
            .set_item("usecols", CSV_USECOLS)
            .expect("CSV usecols kwarg");

        // Fail closed on route attribution before timing. The compatible
        // negative-usecols parse and axis-0 std must survive poisoned NumPy
        // entry points. The C-contiguous 2-D histogram must likewise avoid the
        // counted NumPy wrapper.
        let route_namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                "histogram_calls = [0]\n\
                 def poison_loadtxt(*args, **kwargs):\n\
                 \x20   raise AssertionError('fnp.loadtxt delegated')\n\
                 def poison_std(*args, **kwargs):\n\
                 \x20   raise AssertionError('fnp.std delegated')\n\
                 def counted_histogram(*args, **kwargs):\n\
                 \x20   histogram_calls[0] += 1\n\
                 \x20   return original_histogram(*args, **kwargs)\n",
            )
            .expect("route-attribution CString")
            .as_c_str(),
            Some(&route_namespace),
            Some(&route_namespace),
        )
        .expect("route-attribution helpers");
        route_namespace
            .set_item("original_histogram", &np_histogram)
            .expect("publish original histogram");
        let poison_loadtxt = route_namespace
            .get_item("poison_loadtxt")
            .expect("poison_loadtxt present");
        numpy
            .setattr("loadtxt", &poison_loadtxt)
            .expect("poison numpy.loadtxt");
        let native_parsed = fnp_loadtxt
            .call((csv_path.as_str(),), Some(&loadtxt_kwargs))
            .expect("native FNP wide-CSV parse under poisoned numpy.loadtxt");
        numpy
            .setattr("loadtxt", &np_loadtxt)
            .expect("restore numpy.loadtxt");

        let poison_std = route_namespace
            .get_item("poison_std")
            .expect("poison_std present");
        numpy.setattr("std", &poison_std).expect("poison numpy.std");
        fnp_std
            .call1((&native_parsed, 0_i64))
            .expect("native FNP axis-0 std under poisoned numpy.std");
        numpy.setattr("std", &np_std).expect("restore numpy.std");

        let counted_histogram = route_namespace
            .get_item("counted_histogram")
            .expect("counted_histogram present");
        numpy
            .setattr("histogram", &counted_histogram)
            .expect("install counted numpy.histogram");
        fnp_histogram
            .call1((&native_parsed, HISTOGRAM_BINS))
            .expect("FNP 2-D histogram through counted NumPy fallback");
        numpy
            .setattr("histogram", &np_histogram)
            .expect("restore numpy.histogram");
        let histogram_calls = route_namespace
            .get_item("histogram_calls")
            .expect("histogram_calls present")
            .get_item(0)
            .expect("histogram call counter")
            .extract::<usize>()
            .expect("histogram call counter integer");
        assert_eq!(
            histogram_calls, 0,
            "FNP's native 2-D histogram route delegated to NumPy"
        );
        println!(
            "ROUTE_PROOF row={ROW} loadtxt_numpy_poison_survived=true \
             std_numpy_poison_survived=true histogram_counted_numpy_calls={histogram_calls} \
             loadtxt_route=fnp_io::loadtxt_usecols_signed \
             loadtxt_source_pin=fnp-python/src/lib.rs:loadtxt_bounded_negative_usecols \
             histogram_route=try_zerocopy_histogram_2d_c_contiguous_ravel \
             histogram_source_pin=fnp-python/src/lib.rs:try_zerocopy_histogram \
             std_route=try_zerocopy_f64_var_axis0 \
             std_source_pin=fnp-python/src/lib.rs:py_std_axis0"
        );
        println!(
            "RETRY_PREDICATE_PROOF row={ROW} required_rows_min=100000 actual_rows={CSV_ROWS} \
             required_columns_min=48 actual_columns={CSV_COLUMNS} \
             selected_columns_max_fraction=0.25 actual_selected_fraction={:.6} satisfied=true",
            CSV_USECOLS.len() as f64 / CSV_COLUMNS as f64,
        );
        println!(
            "WORKLOAD_CONFIG row={ROW} user_job=wide_csv_sensor_etl \
             input=csv[{CSV_ROWS},{CSV_COLUMNS}]_selected_tail_columns[{}] \
             distribution=bounded_signed_decimal \
             stages=loadtxt_negative_usecols,histogram_64,std_axis0 \
             output=global_histogram_edges_plus_per_column_std \
             matched_config=same_process_same_path_warm_page_cache \
             target_user=sensor_batch_ETL",
            CSV_USECOLS.len(),
        );

        let run_incumbent = || {
            let started = Instant::now();
            let parsed = np_loadtxt
                .call((black_box(csv_path.as_str()),), Some(&loadtxt_kwargs))
                .expect("numpy wide-CSV parse");
            let histogram = np_histogram
                .call1((black_box(&parsed), HISTOGRAM_BINS))
                .expect("numpy wide-CSV histogram");
            let std = np_std
                .call1((black_box(&parsed), 0_i64))
                .expect("numpy wide-CSV std");
            let elapsed = started.elapsed();
            let counts = histogram.get_item(0).expect("numpy histogram counts");
            let edges = histogram.get_item(1).expect("numpy histogram edges");
            (elapsed, [counts, edges, std])
        };
        let run_candidate = || {
            let started = Instant::now();
            let parsed = fnp_loadtxt
                .call((black_box(csv_path.as_str()),), Some(&loadtxt_kwargs))
                .expect("fnp wide-CSV parse");
            let histogram = fnp_histogram
                .call1((black_box(&parsed), HISTOGRAM_BINS))
                .expect("fnp wide-CSV histogram");
            let std = fnp_std
                .call1((black_box(&parsed), 0_i64))
                .expect("fnp wide-CSV std");
            let elapsed = started.elapsed();
            let counts = histogram.get_item(0).expect("fnp histogram counts");
            let edges = histogram.get_item(1).expect("fnp histogram edges");
            (elapsed, [counts, edges, std])
        };

        let (_, incumbent_output) = run_incumbent();
        let (_, candidate_output) = run_candidate();
        assert_workload_outputs_equal(&numpy, ROW, &candidate_output, &incumbent_output);
        let output_checksum = workload_checksum(&numpy, &candidate_output);
        println!(
            "PARITY row={ROW} outputs=3 dtype_shape_every_output_byte=identical \
             output_checksum={output_checksum:016x}"
        );
        println!(
            "WORK_ACCOUNTING row={ROW} candidate_public_calls=3 incumbent_public_calls=3 \
             candidate_source_rows={CSV_ROWS} incumbent_source_rows={CSV_ROWS} \
             candidate_source_fields={} incumbent_source_fields={} \
             candidate_selected_values={} incumbent_selected_values={} \
             candidate_histogram_values={} incumbent_histogram_values={} \
             candidate_histogram_bins={HISTOGRAM_BINS} \
             incumbent_histogram_bins={HISTOGRAM_BINS} \
             candidate_std_columns={} incumbent_std_columns={} \
             candidate_output_elements={} incumbent_output_elements={} \
             same_input_bytes={csv_bytes} less_work_claim=false",
            CSV_ROWS * CSV_COLUMNS,
            CSV_ROWS * CSV_COLUMNS,
            CSV_ROWS * CSV_USECOLS.len(),
            CSV_ROWS * CSV_USECOLS.len(),
            CSV_ROWS * CSV_USECOLS.len(),
            CSV_ROWS * CSV_USECOLS.len(),
            CSV_USECOLS.len(),
            CSV_USECOLS.len(),
            HISTOGRAM_BINS + (HISTOGRAM_BINS + 1) + CSV_USECOLS.len(),
            HISTOGRAM_BINS + (HISTOGRAM_BINS + 1) + CSV_USECOLS.len(),
        );

        common::report_observed_thread_activity(ROW, "numpy", THREAD_ACTIVITY_REPETITIONS, || {
            let (_, outputs) = run_incumbent();
            black_box(outputs);
        });
        common::report_observed_thread_activity(ROW, "fnp", THREAD_ACTIVITY_REPETITIONS, || {
            let (_, outputs) = run_candidate();
            black_box(outputs);
        });

        let mut observe_incumbent = || {
            let (elapsed, outputs) = run_incumbent();
            common::ContractObservation {
                elapsed,
                checksum: workload_checksum(&numpy, &outputs),
            }
        };
        let mut observe_candidate = || {
            let (elapsed, outputs) = run_candidate();
            common::ContractObservation {
                elapsed,
                checksum: workload_checksum(&numpy, &outputs),
            }
        };
        let (effect, incumbent_null, candidate_null) = common::run_dual_null_median_ci_contract(
            ROW,
            &mut observe_incumbent,
            &mut observe_candidate,
        );
        let verdict = common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
        println!(
            "WHOLE_JOB_RESULT row={ROW} verdict={verdict} \
             incumbent_median_ms={:.6} candidate_median_ms={:.6} \
             ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
             incumbent_null_ratio={:.6} incumbent_null_ci95=[{:.6},{:.6}] \
             candidate_null_ratio={:.6} candidate_null_ci95=[{:.6},{:.6}] \
             effect_ci_excludes_one={} corrected_dual_null_gate=true \
             median_clause=true actual_threads_reported=true",
            effect.arm_a_median_ns / 1_000_000.0,
            effect.arm_b_median_ns / 1_000_000.0,
            effect.ratio_median,
            effect.ratio_ci_low,
            effect.ratio_ci_high,
            incumbent_null.ratio_median,
            incumbent_null.ratio_ci_low,
            incumbent_null.ratio_ci_high,
            candidate_null.ratio_median,
            candidate_null.ratio_ci_low,
            candidate_null.ratio_ci_high,
            effect.ratio_ci_low > 1.0 || effect.ratio_ci_high < 1.0,
        );
        let (decision, reason) = match verdict {
            "DECIDABLE_WIN" => ("choose_fnp", "corrected_dual_null_incumbent_win"),
            "DECIDABLE_REGRESSION" => ("choose_numpy", "corrected_dual_null_regression"),
            _ => (
                "choose_neither_on_performance",
                "effect_not_separated_from_dual_null_envelope",
            ),
        };
        println!(
            "CHOOSER_STATEMENT workload=wide_csv_sensor_etl_retry decision={decision} \
             reason={reason} \
             measured_scope=100000x48_bounded_decimal_csv_select_8_negative_tail_columns_histogram64_std_axis0 \
             incumbent=numpy_live_same_invocation \
             shared_component=none \
             outside_scope=run_same_contract_before_choosing"
        );
    });

    std::fs::remove_file(&csv_path).expect("remove owned wide-CSV retry corpus");
    println!("WORKLOAD_SCRATCH_CLEANUP path={csv_path} removed=true");
}

/// Phase-2 incumbent suite: complete jobs rather than accessor-level kernels.
/// Every job consumes realistic, skewed data; crosses multiple public
/// subsystems; returns the report a user asked for; and runs the live NumPy job
/// side-by-side in the same process. Output hashing is deliberately outside the
/// timed interval, so the timed arms share no checksum or serialization tail.
fn bench_realistic_end_to_end_workloads_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    const CSV_ROWS: usize = 25_000;
    const CSV_COLUMNS: usize = 48;
    const CSV_USECOLS: [i64; 8] = [-1, -3, -5, -7, -9, -11, -13, -15];
    const THREADS: &str = "4";

    for variable in [
        "RAYON_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(THREADS),
            "{variable} must be explicitly pinned before workload timing"
        );
    }
    assert_eq!(
        rayon::current_num_threads(),
        THREADS
            .parse::<usize>()
            .expect("thread count constant is numeric"),
        "Rayon pool width does not match the pinned workload configuration"
    );
    println!(
        "WORKLOAD_RUNTIME host={} rayon_threads={} RAYON_NUM_THREADS={} \
         OPENBLAS_NUM_THREADS={} OMP_NUM_THREADS={} MKL_NUM_THREADS={}",
        std::env::var("HOSTNAME").unwrap_or_else(|_| "unavailable".to_owned()),
        rayon::current_num_threads(),
        THREADS,
        THREADS,
        THREADS,
        THREADS,
    );

    let csv_path = std::env::temp_dir().join(format!(
        "fnp_phase2_wide_sensor_batch_{}.csv",
        std::process::id()
    ));
    let mut csv = String::with_capacity(CSV_ROWS * CSV_COLUMNS * 10);
    for row in 0..CSV_ROWS {
        for column in 0..CSV_COLUMNS {
            if column > 0 {
                csv.push(',');
            }
            let raw = ((row * 7_919 + column * 104_729) % 1_000_000) as i64 - 500_000;
            write!(&mut csv, "{:.3}", raw as f64 / 1_000.0)
                .expect("writing the CSV corpus to a String cannot fail");
        }
        csv.push('\n');
    }
    std::fs::write(&csv_path, csv).expect("write Phase-2 wide CSV corpus");
    let csv_path = csv_path
        .to_str()
        .expect("temporary CSV path is UTF-8")
        .to_owned();

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_realistic_workloads")
            .expect("realistic-workload bench module");
        fnp_python(&module).expect("initialize fnp_python realistic-workload module");
        let numpy = py.import("numpy").expect("numpy incumbent");

        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(
                r#"
import numpy as np
rng = np.random.default_rng(20260728)

# 4,096 normalized sensors, each reporting a 512-sample finite half-precision window.
device_level = rng.lognormal(mean=0.5, sigma=0.45, size=(4096, 1))
phase = rng.uniform(0.0, 2.0 * np.pi, size=(4096, 1))
t = np.linspace(0.0, 4.0 * np.pi, 512, dtype=np.float32)[None, :]
seasonal = 0.08 * device_level * np.sin(t + phase)
drift = rng.normal(0.0, 0.0015, size=(4096, 1)) * np.arange(512)[None, :]
noise = rng.normal(0.0, 0.15, size=(4096, 512))
telemetry = np.clip(device_level + seasonal + drift + noise, -8.0, 8.0).astype(np.float16)
telemetry_q = np.array([0.01, 0.10, 0.50, 0.90, 0.99], dtype=np.float64)

# Two million transactions with the heavy-tailed merchant skew seen in fraud logs.
merchant_ids = ((rng.zipf(1.25, size=2_000_000) - 1) % 250_000).astype(np.int64)
watchlist = np.unique(rng.integers(0, 250_000, size=50_000, dtype=np.int64))
channel_ids = rng.integers(0, 64, size=2_000_000, dtype=np.int16)

# A zero-inflated quantized inference batch with rectangular integer GEMM.
activations = rng.integers(-8, 9, size=(4096, 256), dtype=np.int64)
activations[rng.random(size=activations.shape) < 0.72] = 0
weights = rng.integers(-6, 7, size=(256, 64), dtype=np.int64)

# Five hundred thousand service tags drawn from a Zipfian 4,096-tag vocabulary.
tag_vocabulary = np.array(
    [f"svc-{index:04d}-ERR" for index in range(4096)],
    dtype="U16",
)
tag_indices = ((rng.zipf(1.22, size=500_000) - 1) % len(tag_vocabulary)).astype(np.int64)
log_tags = tag_vocabulary[tag_indices]
translate_table = str.maketrans("-ERR", "_err")
"#,
            )
            .expect("realistic-workload setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("realistic-workload corpus setup");

        let telemetry = namespace.get_item("telemetry").expect("telemetry present");
        let telemetry_q = namespace
            .get_item("telemetry_q")
            .expect("telemetry quantiles present");
        let merchant_ids = namespace
            .get_item("merchant_ids")
            .expect("merchant ids present");
        let watchlist = namespace.get_item("watchlist").expect("watchlist present");
        let channel_ids = namespace
            .get_item("channel_ids")
            .expect("channel ids present");
        let activations = namespace
            .get_item("activations")
            .expect("activations present");
        let weights = namespace.get_item("weights").expect("weights present");
        let log_tags = namespace.get_item("log_tags").expect("log tags present");
        let translate_table = namespace
            .get_item("translate_table")
            .expect("translation table present");

        let fnp_average = module.getattr("average").expect("fnp average");
        let np_average = numpy.getattr("average").expect("numpy average");
        let fnp_std = module.getattr("std").expect("fnp std");
        let np_std = numpy.getattr("std").expect("numpy std");
        let fnp_quantile = module.getattr("quantile").expect("fnp quantile");
        let np_quantile = numpy.getattr("quantile").expect("numpy quantile");
        let fnp_isin = module.getattr("isin").expect("fnp isin");
        let np_isin = numpy.getattr("isin").expect("numpy isin");
        let fnp_count_nonzero = module.getattr("count_nonzero").expect("fnp count_nonzero");
        let np_count_nonzero = numpy.getattr("count_nonzero").expect("numpy count_nonzero");
        let fnp_unique = module.getattr("unique").expect("fnp unique");
        let np_unique = numpy.getattr("unique").expect("numpy unique");
        let fnp_bincount = module.getattr("bincount").expect("fnp bincount");
        let np_bincount = numpy.getattr("bincount").expect("numpy bincount");
        let fnp_matmul = module.getattr("matmul").expect("fnp matmul");
        let np_matmul = numpy.getattr("matmul").expect("numpy matmul");
        let fnp_argmax = module.getattr("argmax").expect("fnp argmax");
        let np_argmax = numpy.getattr("argmax").expect("numpy argmax");
        let fnp_loadtxt = module.getattr("loadtxt").expect("fnp loadtxt");
        let np_loadtxt = numpy.getattr("loadtxt").expect("numpy loadtxt");
        let fnp_histogram = module.getattr("histogram").expect("fnp histogram");
        let np_histogram = numpy.getattr("histogram").expect("numpy histogram");
        let fnp_char_translate = module
            .getattr("char")
            .expect("fnp char namespace")
            .getattr("translate")
            .expect("fnp char.translate");
        let np_char_translate = numpy
            .getattr("char")
            .expect("numpy char namespace")
            .getattr("translate")
            .expect("numpy char.translate");

        for (candidate, incumbent, surface) in [
            (&fnp_average, &np_average, "average"),
            (&fnp_std, &np_std, "std"),
            (&fnp_quantile, &np_quantile, "quantile"),
            (&fnp_isin, &np_isin, "isin"),
            (&fnp_count_nonzero, &np_count_nonzero, "count_nonzero"),
            (&fnp_unique, &np_unique, "unique"),
            (&fnp_bincount, &np_bincount, "bincount"),
            (&fnp_matmul, &np_matmul, "matmul"),
            (&fnp_argmax, &np_argmax, "argmax"),
            (&fnp_loadtxt, &np_loadtxt, "loadtxt"),
            (&fnp_histogram, &np_histogram, "histogram"),
            (&fnp_char_translate, &np_char_translate, "char.translate"),
        ] {
            assert!(
                !candidate.is(incumbent),
                "dispatch trap: {surface} incumbent resolved to the FNP callable"
            );
        }

        // Workload 1: a device-health report over an in-memory telemetry window.
        {
            const ROW: &str = "workload_telemetry_health_report_f16_4096x512";
            common::report_numpy_incumbent_identity(py, "average", &np_average);
            common::report_incumbent_topology(
                "fnp.workload.telemetry_health_report",
                "numpy.workload.telemetry_health_report",
            );
            println!(
                "WORKLOAD_CONFIG row={ROW} user_job=device_health_report \
                 input=float16[4096,512] distribution=lognormal_baseline_plus_seasonality_drift_noise \
                 stages=average_axis_last,std_axis_last,quantile_q5 \
                 output=per_device_mean_std_plus_fleet_quantiles matched_config=same_process_same_inputs"
            );

            let run_incumbent = || {
                let started = Instant::now();
                let mean = np_average
                    .call1((black_box(&telemetry), -1_i64))
                    .expect("numpy telemetry average");
                let std = np_std
                    .call1((black_box(&telemetry), -1_i64))
                    .expect("numpy telemetry std");
                let quantiles = np_quantile
                    .call1((black_box(&telemetry), black_box(&telemetry_q)))
                    .expect("numpy telemetry quantiles");
                let elapsed = started.elapsed();
                (elapsed, [mean, std, quantiles])
            };
            let run_candidate = || {
                let started = Instant::now();
                let mean = fnp_average
                    .call1((black_box(&telemetry), -1_i64))
                    .expect("fnp telemetry average");
                let std = fnp_std
                    .call1((black_box(&telemetry), -1_i64))
                    .expect("fnp telemetry std");
                let quantiles = fnp_quantile
                    .call1((black_box(&telemetry), black_box(&telemetry_q)))
                    .expect("fnp telemetry quantiles");
                let elapsed = started.elapsed();
                (elapsed, [mean, std, quantiles])
            };

            let (_, incumbent_output) = run_incumbent();
            let (_, candidate_output) = run_candidate();
            let isfinite = numpy.getattr("isfinite").expect("numpy isfinite");
            for (arm, outputs) in [
                ("candidate", &candidate_output),
                ("incumbent", &incumbent_output),
            ] {
                for (index, output) in outputs.iter().enumerate() {
                    let finite = isfinite
                        .call1((output,))
                        .expect("telemetry output finiteness")
                        .call_method0("all")
                        .expect("all telemetry outputs finite")
                        .extract::<bool>()
                        .expect("telemetry finite verdict");
                    assert!(
                        finite,
                        "{ROW}: {arm} output {index} must be finite for a useful report"
                    );
                }
            }
            assert_workload_outputs_equal(&numpy, ROW, &candidate_output, &incumbent_output);
            let mut observe_incumbent = || {
                let (elapsed, outputs) = run_incumbent();
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let mut observe_candidate = || {
                let (elapsed, outputs) = run_candidate();
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let _ = common::run_dual_null_median_ci_contract(
                ROW,
                &mut observe_incumbent,
                &mut observe_candidate,
            );
        }

        // Workload 2: transaction-risk screening and a compact categorical report.
        {
            const ROW: &str = "workload_transaction_risk_report_zipf_2m";
            common::report_numpy_incumbent_identity(py, "isin", &np_isin);
            common::report_incumbent_topology(
                "fnp.workload.transaction_risk_report",
                "numpy.workload.transaction_risk_report",
            );
            println!(
                "WORKLOAD_CONFIG row={ROW} user_job=transaction_risk_report \
                 input=int64_transactions[2000000]_watchlist[~45000]_int16_channels[2000000] \
                 distribution=zipf_merchants_plus_uniform_channels \
                 stages=isin,count_nonzero,unique_return_counts,bincount \
                 output=risk_hits_merchant_frequency_channel_frequency \
                 matched_config=same_process_same_inputs"
            );
            let unique_kwargs = PyDict::new(py);
            unique_kwargs
                .set_item("return_counts", true)
                .expect("unique return_counts kwarg");

            let run_incumbent = || {
                let started = Instant::now();
                let risk_mask = np_isin
                    .call1((black_box(&merchant_ids), black_box(&watchlist)))
                    .expect("numpy transaction isin");
                let risk_hits = np_count_nonzero
                    .call1((black_box(&risk_mask),))
                    .expect("numpy risk hit count");
                let merchant_frequency = np_unique
                    .call((black_box(&merchant_ids),), Some(&unique_kwargs))
                    .expect("numpy merchant frequency");
                let channel_frequency = np_bincount
                    .call1((black_box(&channel_ids),))
                    .expect("numpy channel frequency");
                let elapsed = started.elapsed();
                let merchant_values = merchant_frequency
                    .get_item(0)
                    .expect("numpy merchant values");
                let merchant_counts = merchant_frequency
                    .get_item(1)
                    .expect("numpy merchant counts");
                (
                    elapsed,
                    [
                        risk_hits,
                        merchant_values,
                        merchant_counts,
                        channel_frequency,
                    ],
                )
            };
            let run_candidate = || {
                let started = Instant::now();
                let risk_mask = fnp_isin
                    .call1((black_box(&merchant_ids), black_box(&watchlist)))
                    .expect("fnp transaction isin");
                let risk_hits = fnp_count_nonzero
                    .call1((black_box(&risk_mask),))
                    .expect("fnp risk hit count");
                let merchant_frequency = fnp_unique
                    .call((black_box(&merchant_ids),), Some(&unique_kwargs))
                    .expect("fnp merchant frequency");
                let channel_frequency = fnp_bincount
                    .call1((black_box(&channel_ids),))
                    .expect("fnp channel frequency");
                let elapsed = started.elapsed();
                let merchant_values = merchant_frequency.get_item(0).expect("fnp merchant values");
                let merchant_counts = merchant_frequency.get_item(1).expect("fnp merchant counts");
                (
                    elapsed,
                    [
                        risk_hits,
                        merchant_values,
                        merchant_counts,
                        channel_frequency,
                    ],
                )
            };

            let (_, incumbent_output) = run_incumbent();
            let (_, candidate_output) = run_candidate();
            assert_workload_outputs_equal(&numpy, ROW, &candidate_output, &incumbent_output);
            let mut observe_incumbent = || {
                let (elapsed, outputs) = run_incumbent();
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let mut observe_candidate = || {
                let (elapsed, outputs) = run_candidate();
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let _ = common::run_dual_null_median_ci_contract(
                ROW,
                &mut observe_incumbent,
                &mut observe_candidate,
            );
        }

        // Workload 3: rectangular integer inference followed by prediction tallying.
        {
            const ROW: &str = "workload_quantized_batch_inference_i64_4096x256x64";
            common::report_numpy_incumbent_identity(py, "matmul", &np_matmul);
            common::report_incumbent_topology(
                "fnp.workload.quantized_batch_inference",
                "numpy.workload.quantized_batch_inference",
            );
            println!(
                "WORKLOAD_CONFIG row={ROW} user_job=quantized_batch_inference \
                 input=int64_activations[4096,256]_weights[256,64] \
                 distribution=72pct_zero_inflated_small_integers \
                 stages=rectangular_matmul,argmax_axis1,bincount \
                 output=predicted_class_per_row_plus_class_frequency \
                 matched_config=same_process_same_inputs no_square_float_gemm=true"
            );

            let run_incumbent = || {
                let started = Instant::now();
                let scores = np_matmul
                    .call1((black_box(&activations), black_box(&weights)))
                    .expect("numpy inference matmul");
                let predictions = np_argmax
                    .call1((black_box(&scores), 1_i64))
                    .expect("numpy inference argmax");
                let class_frequency = np_bincount
                    .call1((black_box(&predictions),))
                    .expect("numpy inference bincount");
                let elapsed = started.elapsed();
                (elapsed, [predictions, class_frequency])
            };
            let run_candidate = || {
                let started = Instant::now();
                let scores = fnp_matmul
                    .call1((black_box(&activations), black_box(&weights)))
                    .expect("fnp inference matmul");
                let predictions = fnp_argmax
                    .call1((black_box(&scores), 1_i64))
                    .expect("fnp inference argmax");
                let class_frequency = fnp_bincount
                    .call1((black_box(&predictions),))
                    .expect("fnp inference bincount");
                let elapsed = started.elapsed();
                (elapsed, [predictions, class_frequency])
            };

            let (_, incumbent_output) = run_incumbent();
            let (_, candidate_output) = run_candidate();
            assert_workload_outputs_equal(&numpy, ROW, &candidate_output, &incumbent_output);
            let mut observe_incumbent = || {
                let (elapsed, outputs) = run_incumbent();
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let mut observe_candidate = || {
                let (elapsed, outputs) = run_candidate();
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let _ = common::run_dual_null_median_ci_contract(
                ROW,
                &mut observe_incumbent,
                &mut observe_candidate,
            );
        }

        // Workload 4: parse a wide on-disk sensor batch and immediately summarize it.
        {
            const ROW: &str = "workload_wide_csv_sensor_etl_25000x48_use8";
            common::report_numpy_loadtxt_incumbent_identity(py, &np_loadtxt);
            common::report_incumbent_topology(
                "fnp.workload.wide_csv_sensor_etl",
                "numpy.workload.wide_csv_sensor_etl",
            );
            println!(
                "WORKLOAD_CONFIG row={ROW} user_job=wide_csv_sensor_etl \
                 input=csv[25000,48]_selected_tail_columns[8] distribution=bounded_signed_decimal \
                 stages=loadtxt_negative_usecols,histogram_64,std_axis0 \
                 output=global_histogram_edges_plus_per_column_std \
                 matched_config=same_process_same_path_warm_page_cache"
            );
            let loadtxt_kwargs = PyDict::new(py);
            loadtxt_kwargs
                .set_item("delimiter", ",")
                .expect("CSV delimiter kwarg");
            loadtxt_kwargs
                .set_item("dtype", numpy.getattr("float64").expect("numpy float64"))
                .expect("CSV dtype kwarg");
            loadtxt_kwargs
                .set_item("usecols", CSV_USECOLS)
                .expect("CSV usecols kwarg");

            let run_incumbent = || {
                let started = Instant::now();
                let parsed = np_loadtxt
                    .call((black_box(&csv_path),), Some(&loadtxt_kwargs))
                    .expect("numpy CSV parse");
                let histogram = np_histogram
                    .call1((black_box(&parsed), 64_i64))
                    .expect("numpy CSV histogram");
                let std = np_std
                    .call1((black_box(&parsed), 0_i64))
                    .expect("numpy CSV std");
                let elapsed = started.elapsed();
                let counts = histogram.get_item(0).expect("numpy histogram counts");
                let edges = histogram.get_item(1).expect("numpy histogram edges");
                (elapsed, [counts, edges, std])
            };
            let run_candidate = || {
                let started = Instant::now();
                let parsed = fnp_loadtxt
                    .call((black_box(&csv_path),), Some(&loadtxt_kwargs))
                    .expect("fnp CSV parse");
                let histogram = fnp_histogram
                    .call1((black_box(&parsed), 64_i64))
                    .expect("fnp CSV histogram");
                let std = fnp_std
                    .call1((black_box(&parsed), 0_i64))
                    .expect("fnp CSV std");
                let elapsed = started.elapsed();
                let counts = histogram.get_item(0).expect("fnp histogram counts");
                let edges = histogram.get_item(1).expect("fnp histogram edges");
                (elapsed, [counts, edges, std])
            };

            let (_, incumbent_output) = run_incumbent();
            let (_, candidate_output) = run_candidate();
            assert_workload_outputs_equal(&numpy, ROW, &candidate_output, &incumbent_output);
            let mut observe_incumbent = || {
                let (elapsed, outputs) = run_incumbent();
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let mut observe_candidate = || {
                let (elapsed, outputs) = run_candidate();
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let _ = common::run_dual_null_median_ci_contract(
                ROW,
                &mut observe_incumbent,
                &mut observe_candidate,
            );
        }

        // Workload 5: normalize fixed-width ASCII service tags, then aggregate them.
        {
            const ROW: &str = "workload_ascii_log_normalization_zipf_500k";
            common::report_numpy_incumbent_identity(py, "unique", &np_unique);
            common::report_incumbent_topology(
                "fnp.workload.ascii_log_normalization",
                "numpy.workload.ascii_log_normalization",
            );
            println!(
                "WORKLOAD_CONFIG row={ROW} user_job=ascii_log_normalization \
                 input=unicode16_tags[500000]_vocabulary[4096] distribution=zipf \
                 stages=char_translate,unique_return_counts \
                 output=normalized_tag_dictionary_plus_frequency \
                 matched_config=same_process_same_inputs"
            );
            let unique_kwargs = PyDict::new(py);
            unique_kwargs
                .set_item("return_counts", true)
                .expect("log unique return_counts kwarg");

            let run_incumbent = || {
                let started = Instant::now();
                let normalized = np_char_translate
                    .call1((black_box(&log_tags), black_box(&translate_table)))
                    .expect("numpy log normalization");
                let frequency = np_unique
                    .call((black_box(&normalized),), Some(&unique_kwargs))
                    .expect("numpy normalized tag frequency");
                let elapsed = started.elapsed();
                let values = frequency.get_item(0).expect("numpy normalized tags");
                let counts = frequency.get_item(1).expect("numpy normalized counts");
                (elapsed, [values, counts])
            };
            let run_candidate = || {
                let started = Instant::now();
                let normalized = fnp_char_translate
                    .call1((black_box(&log_tags), black_box(&translate_table)))
                    .expect("fnp log normalization");
                let frequency = fnp_unique
                    .call((black_box(&normalized),), Some(&unique_kwargs))
                    .expect("fnp normalized tag frequency");
                let elapsed = started.elapsed();
                let values = frequency.get_item(0).expect("fnp normalized tags");
                let counts = frequency.get_item(1).expect("fnp normalized counts");
                (elapsed, [values, counts])
            };

            let (_, incumbent_output) = run_incumbent();
            let (_, candidate_output) = run_candidate();
            assert_workload_outputs_equal(&numpy, ROW, &candidate_output, &incumbent_output);
            let mut observe_incumbent = || {
                let (elapsed, outputs) = run_incumbent();
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let mut observe_candidate = || {
                let (elapsed, outputs) = run_candidate();
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let _ = common::run_dual_null_median_ci_contract(
                ROW,
                &mut observe_incumbent,
                &mut observe_candidate,
            );
        }
    });
}

/// Phase-2 Class-3 workload: apply a batch of financial events to a
/// large-cardinality account state. NumPy owns the public `ufunc.at` API but
/// has no order-free parallel scatter engine for its cache-exceeding target
/// regime; FrankenNumPy's integer add/min/max arms use atomic RMWs.
///
/// The mixed uniform/Zipf account distribution supplies both cold random
/// targets and duplicate hot accounts. Three event counts distinguish a flat
/// per-event gap from fixed overhead or coordination effects. The 3,000,000
/// account target deliberately excludes the ledger-rejected 1,024-bin
/// histogram regime where NumPy's own specialized loop is already saturated.
fn bench_realistic_event_attribution_scatter_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    const ACCOUNT_COUNT: usize = 3_000_000;
    const EVENT_SIZES: [usize; 3] = [2_200_000, 4_400_000, 8_800_000];
    const THREADS: &str = "4";

    for variable in [
        "RAYON_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(THREADS),
            "{variable} must be explicitly pinned before workload timing"
        );
    }
    assert_eq!(
        rayon::current_num_threads(),
        THREADS
            .parse::<usize>()
            .expect("thread count constant is numeric"),
        "Rayon pool width does not match the pinned workload configuration"
    );
    println!(
        "WORKLOAD_RUNTIME host={} rayon_threads={} RAYON_NUM_THREADS={} \
         OPENBLAS_NUM_THREADS={} OMP_NUM_THREADS={} MKL_NUM_THREADS={} trj_used=false",
        std::env::var("HOSTNAME").unwrap_or_else(|_| "unavailable".to_owned()),
        rayon::current_num_threads(),
        THREADS,
        THREADS,
        THREADS,
        THREADS,
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_event_attribution_workload")
            .expect("event-attribution module");
        fnp_python(&module).expect("initialize fnp_python event-attribution module");
        let numpy = py.import("numpy").expect("numpy incumbent");

        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(format!(
                r#"
import numpy as np
rng = np.random.default_rng(20260729)
account_count = {ACCOUNT_COUNT}
event_sizes = {EVENT_SIZES:?}
max_events = event_sizes[-1]

# Most events touch the long tail, while 35% land on a 100k-account hot set.
# This models payment/accounting streams without collapsing into the rejected
# tiny-histogram regime.
event_accounts = rng.integers(0, account_count, size=max_events, dtype=np.int64)
hot_mask = rng.random(max_events) < 0.35
hot_count = int(hot_mask.sum())
event_accounts[hot_mask] = (
    (rng.zipf(1.18, size=hot_count) - 1) % 100_000
).astype(np.int64)
spend_deltas = rng.integers(-25_000, 50_001, size=max_events, dtype=np.int64)
event_timestamps = (
    np.arange(max_events, dtype=np.int64) + np.int64(1_800_000_000_000_000)
)

event_accounts_by_size = [
    np.ascontiguousarray(event_accounts[:size]) for size in event_sizes
]
spend_deltas_by_size = [
    np.ascontiguousarray(spend_deltas[:size]) for size in event_sizes
]
event_timestamps_by_size = [
    np.ascontiguousarray(event_timestamps[:size]) for size in event_sizes
]

np_spend_state = np.empty(account_count, dtype=np.int64)
np_first_seen_state = np.empty(account_count, dtype=np.int64)
np_last_seen_state = np.empty(account_count, dtype=np.int64)
fnp_spend_state = np.empty(account_count, dtype=np.int64)
fnp_first_seen_state = np.empty(account_count, dtype=np.int64)
fnp_last_seen_state = np.empty(account_count, dtype=np.int64)
"#
            ))
            .expect("event-attribution setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("event-attribution corpus setup");

        let fnp_add = module.getattr("add").expect("fnp add");
        let np_add = numpy.getattr("add").expect("numpy add");
        let fnp_maximum = module.getattr("maximum").expect("fnp maximum");
        let np_maximum = numpy.getattr("maximum").expect("numpy maximum");
        let fnp_minimum = module.getattr("minimum").expect("fnp minimum");
        let np_minimum = numpy.getattr("minimum").expect("numpy minimum");
        for (candidate, incumbent, surface) in [
            (&fnp_add, &np_add, "add"),
            (&fnp_maximum, &np_maximum, "maximum"),
            (&fnp_minimum, &np_minimum, "minimum"),
        ] {
            assert!(
                !candidate.is(incumbent),
                "{surface}: candidate callable aliases NumPy"
            );
            common::report_numpy_incumbent_identity(py, surface, incumbent);
        }
        common::report_incumbent_topology(
            "fnp.workload.event_attribution_scatter",
            "numpy.workload.event_attribution_scatter",
        );
        println!(
            "INCUMBENT_PIPELINE workload=event_attribution_scatter \
             candidate=fnp.add.at+fnp.maximum.at+fnp.minimum.at \
             incumbent=numpy.add.at+numpy.maximum.at+numpy.minimum.at \
             shared_timed_component=none reset_outside_timed_region=true \
             inputs_shared_read_only=true"
        );

        let incumbent = EventAttributionArm {
            add_at: np_add.getattr("at").expect("numpy add.at"),
            maximum_at: np_maximum.getattr("at").expect("numpy maximum.at"),
            minimum_at: np_minimum.getattr("at").expect("numpy minimum.at"),
            spend_state: namespace
                .get_item("np_spend_state")
                .expect("numpy spend state"),
            first_seen_state: namespace
                .get_item("np_first_seen_state")
                .expect("numpy first-seen state"),
            last_seen_state: namespace
                .get_item("np_last_seen_state")
                .expect("numpy last-seen state"),
        };
        let candidate = EventAttributionArm {
            add_at: fnp_add.getattr("at").expect("fnp add.at"),
            maximum_at: fnp_maximum.getattr("at").expect("fnp maximum.at"),
            minimum_at: fnp_minimum.getattr("at").expect("fnp minimum.at"),
            spend_state: namespace
                .get_item("fnp_spend_state")
                .expect("fnp spend state"),
            first_seen_state: namespace
                .get_item("fnp_first_seen_state")
                .expect("fnp first-seen state"),
            last_seen_state: namespace
                .get_item("fnp_last_seen_state")
                .expect("fnp last-seen state"),
        };

        let account_ids_by_size = namespace
            .get_item("event_accounts_by_size")
            .expect("event accounts by size");
        let spend_deltas_by_size = namespace
            .get_item("spend_deltas_by_size")
            .expect("spend deltas by size");
        let timestamps_by_size = namespace
            .get_item("event_timestamps_by_size")
            .expect("event timestamps by size");

        let mut scaling = Vec::with_capacity(EVENT_SIZES.len());
        for (size_index, size) in EVENT_SIZES.into_iter().enumerate() {
            let row = format!("workload_event_attribution_scatter_i64_{size}");
            println!(
                "WORKLOAD_CONFIG row={row} user_job=event_attribution_state_update \
                 accounts={ACCOUNT_COUNT} events={size} \
                 distribution=65pct_uniform_plus_35pct_zipf_hot100k \
                 stages=add_at_spend,maximum_at_last_seen,minimum_at_first_seen \
                 output=int64_spend_first_seen_last_seen_state \
                 matched_config=same_process_same_inputs target_regime=large \
                 rejected_histogram_regime_excluded=true"
            );
            let account_ids = account_ids_by_size
                .get_item(size_index)
                .expect("event account slice");
            let spend_deltas = spend_deltas_by_size
                .get_item(size_index)
                .expect("event spend slice");
            let event_timestamps = timestamps_by_size
                .get_item(size_index)
                .expect("event timestamp slice");

            let (_, incumbent_output) =
                incumbent.run(&account_ids, &spend_deltas, &event_timestamps);
            let (_, candidate_output) =
                candidate.run(&account_ids, &spend_deltas, &event_timestamps);
            assert_workload_outputs_equal(&numpy, &row, &candidate_output, &incumbent_output);
            println!(
                "PARITY row={row} dtype=int64 shape=[{ACCOUNT_COUNT}] outputs=3 \
                 byte_identity=passed checksum={:016x}",
                workload_checksum(&numpy, &candidate_output),
            );

            let incumbent_profile =
                incumbent.profile(&account_ids, &spend_deltas, &event_timestamps);
            let candidate_profile =
                candidate.profile(&account_ids, &spend_deltas, &event_timestamps);
            let candidate_total = candidate_profile.iter().sum::<f64>();
            let (target_stage, target_ms) = [
                ("add_at_spend", candidate_profile[0]),
                ("maximum_at_last_seen", candidate_profile[1]),
                ("minimum_at_first_seen", candidate_profile[2]),
            ]
            .into_iter()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .expect("three profile stages");
            for (stage, incumbent_ms, candidate_ms) in [
                ("add_at_spend", incumbent_profile[0], candidate_profile[0]),
                (
                    "maximum_at_last_seen",
                    incumbent_profile[1],
                    candidate_profile[1],
                ),
                (
                    "minimum_at_first_seen",
                    incumbent_profile[2],
                    candidate_profile[2],
                ),
            ] {
                println!(
                    "PROFILE_STAGE row={row} arm_pair=numpy_fnp stage={stage} \
                     incumbent_median_ms={incumbent_ms:.6} \
                     candidate_median_ms={candidate_ms:.6}"
                );
            }
            let target_fraction = target_ms / candidate_total;
            println!(
                "PROFILE_SUMMARY row={row} target_stage={target_stage} \
                 target_candidate_ms={target_ms:.6} candidate_stage_sum_ms={candidate_total:.6} \
                 target_self_fraction_pct={:.3} amdahl_remove_ceiling={:.6}",
                target_fraction * 100.0,
                1.0 / (1.0 - target_fraction),
            );

            let mut observe_incumbent = || {
                let (elapsed, outputs) =
                    incumbent.run(&account_ids, &spend_deltas, &event_timestamps);
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let mut observe_candidate = || {
                let (elapsed, outputs) =
                    candidate.run(&account_ids, &spend_deltas, &event_timestamps);
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let (effect, incumbent_null, candidate_null) = common::run_dual_null_median_ci_contract(
                &row,
                &mut observe_incumbent,
                &mut observe_candidate,
            );
            println!(
                "WORKLOAD_SIZE_POINT workload=event_attribution_scatter size={size} \
                 incumbent_median_ms={:.6} candidate_median_ms={:.6} \
                 ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
                 incumbent_null_ratio={:.6} candidate_null_ratio={:.6} \
                 incumbent_ns_per_event={:.3} candidate_ns_per_event={:.3}",
                effect.arm_a_median_ns / 1_000_000.0,
                effect.arm_b_median_ns / 1_000_000.0,
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                incumbent_null.ratio_median,
                candidate_null.ratio_median,
                effect.arm_a_median_ns / size as f64,
                effect.arm_b_median_ns / size as f64,
            );
            scaling.push(effect);
        }

        let first_ratio = scaling.first().expect("first scaling point").ratio_median;
        let last_ratio = scaling.last().expect("last scaling point").ratio_median;
        let ratio_min = scaling
            .iter()
            .map(|stats| stats.ratio_median)
            .fold(f64::INFINITY, f64::min);
        let ratio_max = scaling
            .iter()
            .map(|stats| stats.ratio_median)
            .fold(f64::NEG_INFINITY, f64::max);
        let ratio_spread = ratio_max / ratio_min - 1.0;
        let monotonic_up = scaling
            .windows(2)
            .all(|points| points[1].ratio_median >= points[0].ratio_median);
        let monotonic_down = scaling
            .windows(2)
            .all(|points| points[1].ratio_median <= points[0].ratio_median);
        let shape = if ratio_spread <= 0.15 {
            "FLAT_PER_EVENT_COST"
        } else if monotonic_up && last_ratio >= first_ratio * 1.15 {
            "WIDENING_WITH_BATCH"
        } else if monotonic_down && last_ratio <= first_ratio * 0.85 {
            "NARROWING_WITH_BATCH"
        } else {
            "MIXED_OR_NOISE"
        };
        println!(
            "SCALING_SHAPE workload=event_attribution_scatter dimension=batch_size \
             sizes=[{},{},{}] ratios=[{:.6},{:.6},{:.6}] \
             ratio_spread={ratio_spread:.6} classification={shape}",
            EVENT_SIZES[0],
            EVENT_SIZES[1],
            EVENT_SIZES[2],
            scaling[0].ratio_median,
            scaling[1].ratio_median,
            scaling[2].ratio_median,
        );
    });
}

/// Phase-2 Class-3 workload: reconcile two entitlement snapshots whose keys
/// are packed `(tenant, principal, entitlement)` records. NumPy has no hashed
/// structured-record set algebra; its public operations sort through the
/// generic per-field void comparator. FrankenNumPy value-lex canonicalizes the
/// records and uses fixed-record hash membership for intersection/difference.
///
/// Three snapshot sizes distinguish fixed setup from per-record comparator
/// cost. Each current snapshot deliberately contains repeated carry-forward
/// grants plus genuinely new keys, producing meaningful canonical counts and
/// unchanged/added/revoked outputs.
fn bench_realistic_entitlement_reconciliation_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    // Keep all three points on the >=65,536-record structured-hash route while
    // fitting the complete explicit dual-null sweep inside RCH's 1,800-second
    // remote-execution ceiling.
    const SNAPSHOT_SIZES: [usize; 3] = [65_536, 72_000, 80_000];
    const WORKLOAD_CONTRACT_ROUNDS: usize = 21;
    const WORKLOAD_CONTRACT_MIN_OF: usize = 2;
    const THREADS: &str = "4";

    for variable in [
        "RAYON_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(THREADS),
            "{variable} must be explicitly pinned before workload timing"
        );
    }
    assert_eq!(
        rayon::current_num_threads(),
        THREADS
            .parse::<usize>()
            .expect("thread count constant is numeric"),
        "Rayon pool width does not match the pinned workload configuration"
    );
    println!(
        "WORKLOAD_RUNTIME host={} rayon_threads={} RAYON_NUM_THREADS={} \
         OPENBLAS_NUM_THREADS={} OMP_NUM_THREADS={} MKL_NUM_THREADS={} trj_used=false",
        std::env::var("HOSTNAME").unwrap_or_else(|_| "unavailable".to_owned()),
        rayon::current_num_threads(),
        THREADS,
        THREADS,
        THREADS,
        THREADS,
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_entitlement_reconciliation_workload")
            .expect("entitlement-reconciliation module");
        fnp_python(&module).expect("initialize fnp_python entitlement-reconciliation module");
        let numpy = py.import("numpy").expect("numpy incumbent");

        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(format!(
                r#"
import numpy as np
rng = np.random.default_rng(20260729)
snapshot_sizes = {SNAPSHOT_SIZES:?}
grant_dtype = np.dtype([
    ("tenant_id", "<i8"),
    ("principal_id", "<i8"),
    ("entitlement_id", "<i8"),
], align=False)

previous_by_size = []
current_by_size = []
for size in snapshot_sizes:
    previous = np.empty(size, dtype=grant_dtype)
    previous["tenant_id"] = (rng.zipf(1.19, size=size) - 1) % 2_048
    previous["principal_id"] = (
        previous["tenant_id"] * np.int64(1_000_000)
        + rng.integers(0, 200_000, size=size, dtype=np.int64)
    )
    previous["entitlement_id"] = (
        (rng.zipf(1.31, size=size) - 1) % 512
    ).astype(np.int64)

    carry_count = int(size * 0.72)
    carry = previous[
        rng.integers(0, size, size=carry_count, dtype=np.int64)
    ].copy()
    new_count = size - carry_count
    new = np.empty(new_count, dtype=grant_dtype)
    new["tenant_id"] = (rng.zipf(1.19, size=new_count) - 1) % 2_048
    new["principal_id"] = (
        new["tenant_id"] * np.int64(1_000_000)
        + rng.integers(200_000, 450_000, size=new_count, dtype=np.int64)
    )
    new["entitlement_id"] = (
        (rng.zipf(1.31, size=new_count) - 1) % 512
    ).astype(np.int64)

    current = np.concatenate((carry, new))
    rng.shuffle(current)
    previous_by_size.append(np.ascontiguousarray(previous))
    current_by_size.append(np.ascontiguousarray(current))

assert grant_dtype.names == (
    "tenant_id", "principal_id", "entitlement_id",
)
assert grant_dtype.itemsize == 24
for snapshot in previous_by_size + current_by_size:
    assert snapshot.ndim == 1
    assert snapshot.flags.c_contiguous
    assert snapshot.dtype == grant_dtype
"#
            ))
            .expect("entitlement-reconciliation setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("entitlement-reconciliation corpus setup");

        let fnp_unique = module.getattr("unique").expect("fnp unique");
        let np_unique = numpy.getattr("unique").expect("numpy unique");
        let fnp_intersect1d = module.getattr("intersect1d").expect("fnp intersect1d");
        let np_intersect1d = numpy.getattr("intersect1d").expect("numpy intersect1d");
        let fnp_setdiff1d = module.getattr("setdiff1d").expect("fnp setdiff1d");
        let np_setdiff1d = numpy.getattr("setdiff1d").expect("numpy setdiff1d");
        for (candidate, incumbent, surface) in [
            (&fnp_unique, &np_unique, "unique"),
            (&fnp_intersect1d, &np_intersect1d, "intersect1d"),
            (&fnp_setdiff1d, &np_setdiff1d, "setdiff1d"),
        ] {
            assert!(
                !candidate.is(incumbent),
                "{surface}: candidate callable aliases NumPy"
            );
            common::report_numpy_incumbent_identity(py, surface, incumbent);
        }
        common::report_incumbent_topology(
            "fnp.workload.entitlement_reconciliation",
            "numpy.workload.entitlement_reconciliation",
        );
        println!(
            "INCUMBENT_PIPELINE workload=entitlement_reconciliation \
             candidate=fnp.unique+fnp.intersect1d+fnp.setdiff1d \
             incumbent=numpy.unique+numpy.intersect1d+numpy.setdiff1d \
             shared_timed_component=none inputs_shared_read_only=true"
        );
        println!(
            "ROUTE_PRECONDITIONS workload=entitlement_reconciliation \
             dtype=packed_native_3xi64 itemsize=24 ndim=1 c_contiguous=true \
             no_padding=true no_float_fields=true min_snapshot_size={} \
             structured_hash_min=65536 route_gates_satisfied=true \
             contract_rounds={WORKLOAD_CONTRACT_ROUNDS} \
             contract_min_of={WORKLOAD_CONTRACT_MIN_OF}",
            SNAPSHOT_SIZES[0],
        );

        let fnp_unique_kwargs = PyDict::new(py);
        fnp_unique_kwargs
            .set_item("return_counts", true)
            .expect("fnp unique return_counts kwarg");
        let np_unique_kwargs = PyDict::new(py);
        np_unique_kwargs
            .set_item("return_counts", true)
            .expect("numpy unique return_counts kwarg");
        let candidate = EntitlementReconciliationArm {
            unique: fnp_unique,
            intersect1d: fnp_intersect1d,
            setdiff1d: fnp_setdiff1d,
            unique_kwargs: fnp_unique_kwargs,
        };
        let incumbent = EntitlementReconciliationArm {
            unique: np_unique,
            intersect1d: np_intersect1d,
            setdiff1d: np_setdiff1d,
            unique_kwargs: np_unique_kwargs,
        };

        let previous_by_size = namespace
            .get_item("previous_by_size")
            .expect("previous entitlement snapshots");
        let current_by_size = namespace
            .get_item("current_by_size")
            .expect("current entitlement snapshots");

        let mut scaling = Vec::with_capacity(SNAPSHOT_SIZES.len());
        for (size_index, size) in SNAPSHOT_SIZES.into_iter().enumerate() {
            let row = format!("workload_entitlement_reconciliation_struct_3xi64_{size}");
            println!(
                "WORKLOAD_CONFIG row={row} user_job=entitlement_snapshot_reconciliation \
                 previous_records={size} current_records={size} \
                 distribution=zipf_tenants_and_entitlements_72pct_carry_forward \
                 stages=unique_return_counts_current,intersect_unchanged,\
                 setdiff_added,setdiff_revoked \
                 output=canonical_counts_unchanged_added_revoked \
                 matched_config=same_process_same_inputs target_regime=large_structured"
            );
            let previous = previous_by_size
                .get_item(size_index)
                .expect("previous entitlement snapshot");
            let current = current_by_size
                .get_item(size_index)
                .expect("current entitlement snapshot");

            let (_, incumbent_output) = incumbent.run(&previous, &current);
            let (_, candidate_output) = candidate.run(&previous, &current);
            assert_workload_outputs_equal(&numpy, &row, &candidate_output, &incumbent_output);
            println!(
                "PARITY row={row} dtype=packed_struct_3xi64 input_shape=[{size}] outputs=5 \
                 byte_identity=passed checksum={:016x}",
                workload_checksum(&numpy, &candidate_output),
            );

            let incumbent_profile = incumbent.profile(&previous, &current);
            let candidate_profile = candidate.profile(&previous, &current);
            let candidate_total = candidate_profile.iter().sum::<f64>();
            let (target_stage, target_ms) = [
                ("unique_return_counts_current", candidate_profile[0]),
                ("intersect_unchanged", candidate_profile[1]),
                ("setdiff_added", candidate_profile[2]),
                ("setdiff_revoked", candidate_profile[3]),
            ]
            .into_iter()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .expect("four profile stages");
            for (stage, incumbent_ms, candidate_ms) in [
                (
                    "unique_return_counts_current",
                    incumbent_profile[0],
                    candidate_profile[0],
                ),
                (
                    "intersect_unchanged",
                    incumbent_profile[1],
                    candidate_profile[1],
                ),
                ("setdiff_added", incumbent_profile[2], candidate_profile[2]),
                (
                    "setdiff_revoked",
                    incumbent_profile[3],
                    candidate_profile[3],
                ),
            ] {
                println!(
                    "PROFILE_STAGE row={row} arm_pair=numpy_fnp stage={stage} \
                     incumbent_median_ms={incumbent_ms:.6} \
                     candidate_median_ms={candidate_ms:.6}"
                );
            }
            let target_fraction = target_ms / candidate_total;
            println!(
                "PROFILE_SUMMARY row={row} target_stage={target_stage} \
                 target_candidate_ms={target_ms:.6} candidate_stage_sum_ms={candidate_total:.6} \
                 target_self_fraction_pct={:.3} amdahl_remove_ceiling={:.6}",
                target_fraction * 100.0,
                1.0 / (1.0 - target_fraction),
            );

            let mut observe_incumbent = || {
                let (elapsed, outputs) = incumbent.run(&previous, &current);
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let mut observe_candidate = || {
                let (elapsed, outputs) = candidate.run(&previous, &current);
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract_with_sampling(
                    &row,
                    &mut observe_incumbent,
                    &mut observe_candidate,
                    WORKLOAD_CONTRACT_ROUNDS,
                    WORKLOAD_CONTRACT_MIN_OF,
                );
            println!(
                "WORKLOAD_SIZE_POINT workload=entitlement_reconciliation size={size} \
                 incumbent_median_ms={:.6} candidate_median_ms={:.6} \
                 ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
                 incumbent_null_ratio={:.6} candidate_null_ratio={:.6} \
                 incumbent_ns_per_input_record={:.3} \
                 candidate_ns_per_input_record={:.3}",
                effect.arm_a_median_ns / 1_000_000.0,
                effect.arm_b_median_ns / 1_000_000.0,
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                incumbent_null.ratio_median,
                candidate_null.ratio_median,
                effect.arm_a_median_ns / (2 * size) as f64,
                effect.arm_b_median_ns / (2 * size) as f64,
            );
            scaling.push(effect);
        }

        let first_ratio = scaling.first().expect("first scaling point").ratio_median;
        let last_ratio = scaling.last().expect("last scaling point").ratio_median;
        let ratio_min = scaling
            .iter()
            .map(|stats| stats.ratio_median)
            .fold(f64::INFINITY, f64::min);
        let ratio_max = scaling
            .iter()
            .map(|stats| stats.ratio_median)
            .fold(f64::NEG_INFINITY, f64::max);
        let ratio_spread = ratio_max / ratio_min - 1.0;
        let monotonic_up = scaling
            .windows(2)
            .all(|points| points[1].ratio_median >= points[0].ratio_median);
        let monotonic_down = scaling
            .windows(2)
            .all(|points| points[1].ratio_median <= points[0].ratio_median);
        let shape = if ratio_spread <= 0.15 {
            "FLAT_PER_RECORD_COST"
        } else if monotonic_up && last_ratio >= first_ratio * 1.15 {
            "WIDENING_WITH_SNAPSHOT"
        } else if monotonic_down && last_ratio <= first_ratio * 0.85 {
            "NARROWING_WITH_SNAPSHOT"
        } else {
            "MIXED_OR_NOISE"
        };
        println!(
            "SCALING_SHAPE workload=entitlement_reconciliation dimension=snapshot_size \
             sizes=[{},{},{}] ratios=[{:.6},{:.6},{:.6}] \
             ratio_spread={ratio_spread:.6} classification={shape}",
            SNAPSHOT_SIZES[0],
            SNAPSHOT_SIZES[1],
            SNAPSHOT_SIZES[2],
            scaling[0].ratio_median,
            scaling[1].ratio_median,
            scaling[2].ratio_median,
        );
    });
}

/// Phase-2 Class-3 workload: turn a large half-precision latency stream into
/// the distribution report used by telemetry dashboards. NumPy has no native
/// f16 histogram kernel; its equal-width path performs half arithmetic over
/// every sample. FrankenNumPy counts the bounded 65,536-pattern half domain,
/// then classifies occupied patterns against byte-identical f16 edges.
///
/// The job returns the histogram, cumulative counts, and modal bin. Three
/// stream sizes distinguish fixed bounded-domain setup from per-sample work.
fn bench_realistic_f16_telemetry_distribution_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    const SAMPLE_SIZES: [usize; 3] = [2_000_000, 4_000_000, 8_000_000];
    const WORKLOAD_CONTRACT_ROUNDS: usize = 21;
    const WORKLOAD_CONTRACT_MIN_OF: usize = 2;
    const THREADS: &str = "4";

    for variable in [
        "RAYON_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(THREADS),
            "{variable} must be explicitly pinned before workload timing"
        );
    }
    assert_eq!(
        rayon::current_num_threads(),
        THREADS
            .parse::<usize>()
            .expect("thread count constant is numeric"),
        "Rayon pool width does not match the pinned workload configuration"
    );
    println!(
        "WORKLOAD_RUNTIME host={} rayon_threads={} RAYON_NUM_THREADS={} \
         OPENBLAS_NUM_THREADS={} OMP_NUM_THREADS={} MKL_NUM_THREADS={} trj_used=false",
        std::env::var("HOSTNAME").unwrap_or_else(|_| "unavailable".to_owned()),
        rayon::current_num_threads(),
        THREADS,
        THREADS,
        THREADS,
        THREADS,
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_f16_telemetry_distribution_workload")
            .expect("f16 telemetry-distribution module");
        fnp_python(&module).expect("initialize fnp_python telemetry-distribution module");
        let numpy = py.import("numpy").expect("numpy incumbent");

        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(format!(
                r#"
import numpy as np
rng = np.random.default_rng(20260729)
sample_sizes = {SAMPLE_SIZES:?}
max_samples = sample_sizes[-1]

# Positive request latencies: a lognormal body plus a 4% burst tail. Values
# are clipped to the practical dashboard range before f16 storage.
latency_ms = rng.lognormal(mean=3.1, sigma=0.62, size=max_samples)
burst_mask = rng.random(max_samples) < 0.04
latency_ms[burst_mask] *= rng.uniform(2.0, 7.0, size=int(burst_mask.sum()))
latency_ms = np.clip(latency_ms, 0.0625, 4096.0).astype(np.float16)
latency_by_size = [
    np.ascontiguousarray(latency_ms[:size]) for size in sample_sizes
]

assert latency_ms.dtype == np.float16
assert latency_ms.flags.c_contiguous
assert np.isfinite(latency_ms).all()
assert (latency_ms > np.float16(0.0)).all()
"#
            ))
            .expect("f16 telemetry-distribution setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("f16 telemetry-distribution corpus setup");

        let fnp_histogram = module.getattr("histogram").expect("fnp histogram");
        let np_histogram = numpy.getattr("histogram").expect("numpy histogram");
        let fnp_cumsum = module.getattr("cumsum").expect("fnp cumsum");
        let np_cumsum = numpy.getattr("cumsum").expect("numpy cumsum");
        let fnp_argmax = module.getattr("argmax").expect("fnp argmax");
        let np_argmax = numpy.getattr("argmax").expect("numpy argmax");
        for (candidate, incumbent, surface) in [
            (&fnp_histogram, &np_histogram, "histogram"),
            (&fnp_cumsum, &np_cumsum, "cumsum"),
            (&fnp_argmax, &np_argmax, "argmax"),
        ] {
            assert!(
                !candidate.is(incumbent),
                "{surface}: candidate callable aliases NumPy"
            );
            common::report_numpy_incumbent_identity(py, surface, incumbent);
        }
        common::report_incumbent_topology(
            "fnp.workload.f16_telemetry_distribution",
            "numpy.workload.f16_telemetry_distribution",
        );
        println!(
            "INCUMBENT_PIPELINE workload=f16_telemetry_distribution \
             candidate=fnp.histogram+fnp.cumsum+fnp.argmax \
             incumbent=numpy.histogram+numpy.cumsum+numpy.argmax \
             shared_timed_component=none inputs_shared_read_only=true"
        );
        println!(
            "ROUTE_PRECONDITIONS workload=f16_telemetry_distribution \
             dtype=float16 ndim=1 c_contiguous=true native_endian=true \
             finite=true negative_zero=false bins=256 range=none weights=none \
             density=none bounded_pattern_domain=65536 \
             contract_rounds={WORKLOAD_CONTRACT_ROUNDS} \
             contract_min_of={WORKLOAD_CONTRACT_MIN_OF}"
        );

        let incumbent = TelemetryDistributionArm {
            histogram: np_histogram,
            cumsum: np_cumsum,
            argmax: np_argmax,
        };
        let candidate = TelemetryDistributionArm {
            histogram: fnp_histogram,
            cumsum: fnp_cumsum,
            argmax: fnp_argmax,
        };
        let latency_by_size = namespace
            .get_item("latency_by_size")
            .expect("latency streams by size");

        let mut scaling = Vec::with_capacity(SAMPLE_SIZES.len());
        for (size_index, size) in SAMPLE_SIZES.into_iter().enumerate() {
            let row = format!("workload_f16_telemetry_distribution_256bins_{size}");
            println!(
                "WORKLOAD_CONFIG row={row} user_job=latency_distribution_report \
                 samples={size} dtype=float16 bins=256 \
                 distribution=lognormal_body_plus_4pct_burst_tail_clipped_positive \
                 stages=histogram_uniform_256,cumulative_counts,modal_bin \
                 output=counts_edges_cdf_peak_bin \
                 matched_config=same_process_same_inputs target_regime=large_finite_f16 \
                 square_gemm_excluded=true"
            );
            let samples = latency_by_size
                .get_item(size_index)
                .expect("latency stream for size");

            let (_, incumbent_output) = incumbent.run(&samples);
            let (_, candidate_output) = candidate.run(&samples);
            assert_workload_outputs_equal(&numpy, &row, &candidate_output, &incumbent_output);
            println!(
                "PARITY row={row} dtype=float16_input outputs=4 \
                 byte_identity=passed checksum={:016x}",
                workload_checksum(&numpy, &candidate_output),
            );

            let incumbent_profile = incumbent.profile(&samples);
            let candidate_profile = candidate.profile(&samples);
            let candidate_total = candidate_profile.iter().sum::<f64>();
            let (target_stage, target_ms) = [
                ("histogram_uniform_256", candidate_profile[0]),
                ("cumulative_counts", candidate_profile[1]),
                ("modal_bin", candidate_profile[2]),
            ]
            .into_iter()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .expect("three telemetry profile stages");
            for (stage, incumbent_ms, candidate_ms) in [
                (
                    "histogram_uniform_256",
                    incumbent_profile[0],
                    candidate_profile[0],
                ),
                (
                    "cumulative_counts",
                    incumbent_profile[1],
                    candidate_profile[1],
                ),
                ("modal_bin", incumbent_profile[2], candidate_profile[2]),
            ] {
                println!(
                    "PROFILE_STAGE row={row} arm_pair=numpy_fnp stage={stage} \
                     incumbent_median_ms={incumbent_ms:.6} \
                     candidate_median_ms={candidate_ms:.6}"
                );
            }
            let target_fraction = target_ms / candidate_total;
            println!(
                "PROFILE_SUMMARY row={row} target_stage={target_stage} \
                 target_candidate_ms={target_ms:.6} candidate_stage_sum_ms={candidate_total:.6} \
                 target_self_fraction_pct={:.3} amdahl_remove_ceiling={:.6}",
                target_fraction * 100.0,
                1.0 / (1.0 - target_fraction),
            );

            let mut observe_incumbent = || {
                let (elapsed, outputs) = incumbent.run(&samples);
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let mut observe_candidate = || {
                let (elapsed, outputs) = candidate.run(&samples);
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract_with_sampling(
                    &row,
                    &mut observe_incumbent,
                    &mut observe_candidate,
                    WORKLOAD_CONTRACT_ROUNDS,
                    WORKLOAD_CONTRACT_MIN_OF,
                );
            println!(
                "WORKLOAD_SIZE_POINT workload=f16_telemetry_distribution size={size} \
                 incumbent_median_ms={:.6} candidate_median_ms={:.6} \
                 ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
                 incumbent_null_ratio={:.6} candidate_null_ratio={:.6} \
                 incumbent_ns_per_sample={:.3} candidate_ns_per_sample={:.3}",
                effect.arm_a_median_ns / 1_000_000.0,
                effect.arm_b_median_ns / 1_000_000.0,
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                incumbent_null.ratio_median,
                candidate_null.ratio_median,
                effect.arm_a_median_ns / size as f64,
                effect.arm_b_median_ns / size as f64,
            );
            scaling.push(effect);
        }

        let first_ratio = scaling.first().expect("first scaling point").ratio_median;
        let last_ratio = scaling.last().expect("last scaling point").ratio_median;
        let ratio_min = scaling
            .iter()
            .map(|stats| stats.ratio_median)
            .fold(f64::INFINITY, f64::min);
        let ratio_max = scaling
            .iter()
            .map(|stats| stats.ratio_median)
            .fold(f64::NEG_INFINITY, f64::max);
        let ratio_spread = ratio_max / ratio_min - 1.0;
        let monotonic_up = scaling
            .windows(2)
            .all(|points| points[1].ratio_median >= points[0].ratio_median);
        let monotonic_down = scaling
            .windows(2)
            .all(|points| points[1].ratio_median <= points[0].ratio_median);
        let shape = if ratio_spread <= 0.15 {
            "FLAT_PER_SAMPLE_COST"
        } else if monotonic_up && last_ratio >= first_ratio * 1.15 {
            "WIDENING_WITH_STREAM"
        } else if monotonic_down && last_ratio <= first_ratio * 0.85 {
            "NARROWING_WITH_STREAM"
        } else {
            "MIXED_OR_NOISE"
        };
        println!(
            "SCALING_SHAPE workload=f16_telemetry_distribution dimension=sample_count \
             sizes=[{},{},{}] ratios=[{:.6},{:.6},{:.6}] \
             ratio_spread={ratio_spread:.6} classification={shape}",
            SAMPLE_SIZES[0],
            SAMPLE_SIZES[1],
            SAMPLE_SIZES[2],
            scaling[0].ratio_median,
            scaling[1].ratio_median,
            scaling[2].ratio_median,
        );
    });
}

/// Phase-2 Class-3 workload: audit the dynamic range of a large f16 sensor
/// stream, retain its exact mantissa/exponent representation, count exponent
/// occupancy, and reconstruct the original stream byte for byte. NumPy has no
/// f16 ALU, so both frexp and ldexp widen every element through a scalar loop;
/// FrankenNumPy uses the landed exact parallel half-domain paths.
fn bench_realistic_f16_dynamic_range_audit_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    const SAMPLE_SIZES: [usize; 3] = [2_000_000, 4_000_000, 8_000_000];
    const EXPONENT_BUCKETS: i64 = 17;
    const WORKLOAD_CONTRACT_ROUNDS: usize = 21;
    const WORKLOAD_CONTRACT_MIN_OF: usize = 2;
    const THREADS: &str = "4";

    for variable in [
        "RAYON_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(THREADS),
            "{variable} must be explicitly pinned before workload timing"
        );
    }
    assert_eq!(
        rayon::current_num_threads(),
        THREADS
            .parse::<usize>()
            .expect("thread count constant is numeric"),
        "Rayon pool width does not match the pinned workload configuration"
    );
    println!(
        "WORKLOAD_RUNTIME host={} rayon_threads={} RAYON_NUM_THREADS={} \
         OPENBLAS_NUM_THREADS={} OMP_NUM_THREADS={} MKL_NUM_THREADS={} trj_used=false",
        std::env::var("HOSTNAME").unwrap_or_else(|_| "unavailable".to_owned()),
        rayon::current_num_threads(),
        THREADS,
        THREADS,
        THREADS,
        THREADS,
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_f16_dynamic_range_audit_workload")
            .expect("f16 dynamic-range module");
        fnp_python(&module).expect("initialize fnp_python dynamic-range module");
        let numpy = py.import("numpy").expect("numpy incumbent");

        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(format!(
                r#"
import numpy as np
rng = np.random.default_rng(20260730)
sample_sizes = {SAMPLE_SIZES:?}
max_samples = sample_sizes[-1]

# Signed vibration/power amplitudes: a lognormal body, a 2.5% burst tail,
# symmetric polarity, and 0.8% sensor dropouts including negative zero.
magnitude = rng.lognormal(mean=4.1, sigma=1.35, size=max_samples)
burst_mask = rng.random(max_samples) < 0.025
magnitude[burst_mask] *= rng.uniform(4.0, 16.0, size=int(burst_mask.sum()))
magnitude = np.clip(magnitude, 0.5, 32768.0)
polarity = np.where(rng.random(max_samples) < 0.5, -1.0, 1.0)
signal = (magnitude * polarity).astype(np.float16)
dropout = rng.random(max_samples) < 0.008
signal[dropout] = np.float16(0.0)
negative_zero_indices = np.flatnonzero(dropout)[::2]
signal[negative_zero_indices] = np.float16(-0.0)
signal_by_size = [
    np.ascontiguousarray(signal[:size]) for size in sample_sizes
]

assert signal.dtype == np.float16
assert signal.flags.c_contiguous
assert np.isfinite(signal).all()
_, route_exponents = np.frexp(signal)
assert route_exponents.min() >= 0
assert route_exponents.max() < {EXPONENT_BUCKETS}
"#
            ))
            .expect("f16 dynamic-range setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("f16 dynamic-range corpus setup");

        let fnp_frexp = module.getattr("frexp").expect("fnp frexp");
        let np_frexp = numpy.getattr("frexp").expect("numpy frexp");
        let fnp_bincount = module.getattr("bincount").expect("fnp bincount");
        let np_bincount = numpy.getattr("bincount").expect("numpy bincount");
        let fnp_ldexp = module.getattr("ldexp").expect("fnp ldexp");
        let np_ldexp = numpy.getattr("ldexp").expect("numpy ldexp");
        for (candidate, incumbent, surface) in [
            (&fnp_frexp, &np_frexp, "frexp"),
            (&fnp_bincount, &np_bincount, "bincount"),
            (&fnp_ldexp, &np_ldexp, "ldexp"),
        ] {
            assert!(
                !candidate.is(incumbent),
                "{surface}: candidate callable aliases NumPy"
            );
            common::report_numpy_incumbent_identity(py, surface, incumbent);
        }
        common::report_incumbent_topology(
            "fnp.workload.f16_dynamic_range_audit",
            "numpy.workload.f16_dynamic_range_audit",
        );
        println!(
            "INCUMBENT_PIPELINE workload=f16_dynamic_range_audit \
             candidate=fnp.frexp+fnp.bincount+fnp.ldexp \
             incumbent=numpy.frexp+numpy.bincount+numpy.ldexp \
             shared_timed_component=none inputs_shared_read_only=true"
        );
        println!(
            "ROUTE_PRECONDITIONS workload=f16_dynamic_range_audit \
             dtype=float16 ndim=1 c_contiguous=true native_endian=true \
             finite=true elements_min={} exponent_dtype=int32 exponent_range=0..{} \
             bincount_weights=none bincount_minlength={EXPONENT_BUCKETS} \
             contract_rounds={WORKLOAD_CONTRACT_ROUNDS} \
             contract_min_of={WORKLOAD_CONTRACT_MIN_OF}",
            1 << 20,
            EXPONENT_BUCKETS - 1,
        );

        let incumbent_bincount_kwargs = PyDict::new(py);
        incumbent_bincount_kwargs
            .set_item("minlength", EXPONENT_BUCKETS)
            .expect("incumbent exponent minlength");
        let candidate_bincount_kwargs = PyDict::new(py);
        candidate_bincount_kwargs
            .set_item("minlength", EXPONENT_BUCKETS)
            .expect("candidate exponent minlength");
        let incumbent = DynamicRangeAuditArm {
            frexp: np_frexp,
            bincount: np_bincount,
            ldexp: np_ldexp,
            bincount_kwargs: incumbent_bincount_kwargs,
        };
        let candidate = DynamicRangeAuditArm {
            frexp: fnp_frexp,
            bincount: fnp_bincount,
            ldexp: fnp_ldexp,
            bincount_kwargs: candidate_bincount_kwargs,
        };
        let signal_by_size = namespace
            .get_item("signal_by_size")
            .expect("dynamic-range streams by size");

        let mut scaling = Vec::with_capacity(SAMPLE_SIZES.len());
        for (size_index, size) in SAMPLE_SIZES.into_iter().enumerate() {
            let row = format!("workload_f16_dynamic_range_audit_{size}");
            println!(
                "WORKLOAD_CONFIG row={row} user_job=f16_dynamic_range_audit \
                 samples={size} dtype=float16 exponent_buckets={EXPONENT_BUCKETS} \
                 distribution=signed_lognormal_plus_2_5pct_bursts_and_0_8pct_dropouts \
                 stages=frexp_decompose,bincount_exponents,ldexp_reconstruct \
                 output=mantissa_exponent_distribution_reconstruction \
                 matched_config=same_process_same_inputs target_regime=large_finite_f16 \
                 square_gemm_excluded=true"
            );
            let samples = signal_by_size
                .get_item(size_index)
                .expect("dynamic-range stream for size");

            let (_, incumbent_output) = incumbent.run(&samples);
            let (_, candidate_output) = candidate.run(&samples);
            assert_workload_outputs_equal(&numpy, &row, &candidate_output, &incumbent_output);
            let input_bytes = samples
                .call_method0("tobytes")
                .expect("dynamic-range input bytes")
                .extract::<Vec<u8>>()
                .expect("dynamic-range input byte vector");
            for (arm, output) in [
                ("numpy", &incumbent_output[3]),
                ("fnp", &candidate_output[3]),
            ] {
                assert_eq!(
                    output
                        .call_method0("tobytes")
                        .expect("dynamic-range reconstruction bytes")
                        .extract::<Vec<u8>>()
                        .expect("dynamic-range reconstruction byte vector"),
                    input_bytes,
                    "{row}: {arm} reconstruction must preserve every f16 bit"
                );
            }
            println!(
                "PARITY row={row} dtype=float16_input outputs=4 \
                 byte_identity=passed reconstruction_identity=passed checksum={:016x}",
                workload_checksum(&numpy, &candidate_output),
            );

            let incumbent_profile = incumbent.profile(&samples);
            let candidate_profile = candidate.profile(&samples);
            let candidate_total = candidate_profile.iter().sum::<f64>();
            let (target_stage, target_ms) = [
                ("frexp_decompose", candidate_profile[0]),
                ("bincount_exponents", candidate_profile[1]),
                ("ldexp_reconstruct", candidate_profile[2]),
            ]
            .into_iter()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .expect("three dynamic-range profile stages");
            for (stage, incumbent_ms, candidate_ms) in [
                (
                    "frexp_decompose",
                    incumbent_profile[0],
                    candidate_profile[0],
                ),
                (
                    "bincount_exponents",
                    incumbent_profile[1],
                    candidate_profile[1],
                ),
                (
                    "ldexp_reconstruct",
                    incumbent_profile[2],
                    candidate_profile[2],
                ),
            ] {
                println!(
                    "PROFILE_STAGE row={row} arm_pair=numpy_fnp stage={stage} \
                     incumbent_median_ms={incumbent_ms:.6} \
                     candidate_median_ms={candidate_ms:.6}"
                );
            }
            let target_fraction = target_ms / candidate_total;
            println!(
                "PROFILE_SUMMARY row={row} target_stage={target_stage} \
                 target_candidate_ms={target_ms:.6} candidate_stage_sum_ms={candidate_total:.6} \
                 target_self_fraction_pct={:.3} amdahl_remove_ceiling={:.6}",
                target_fraction * 100.0,
                1.0 / (1.0 - target_fraction),
            );

            let mut observe_incumbent = || {
                let (elapsed, outputs) = incumbent.run(&samples);
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let mut observe_candidate = || {
                let (elapsed, outputs) = candidate.run(&samples);
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract_with_sampling(
                    &row,
                    &mut observe_incumbent,
                    &mut observe_candidate,
                    WORKLOAD_CONTRACT_ROUNDS,
                    WORKLOAD_CONTRACT_MIN_OF,
                );
            println!(
                "WORKLOAD_SIZE_POINT workload=f16_dynamic_range_audit size={size} \
                 incumbent_median_ms={:.6} candidate_median_ms={:.6} \
                 ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
                 incumbent_null_ratio={:.6} candidate_null_ratio={:.6} \
                 incumbent_ns_per_sample={:.3} candidate_ns_per_sample={:.3}",
                effect.arm_a_median_ns / 1_000_000.0,
                effect.arm_b_median_ns / 1_000_000.0,
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                incumbent_null.ratio_median,
                candidate_null.ratio_median,
                effect.arm_a_median_ns / size as f64,
                effect.arm_b_median_ns / size as f64,
            );
            scaling.push(effect);
        }

        let first_ratio = scaling.first().expect("first scaling point").ratio_median;
        let last_ratio = scaling.last().expect("last scaling point").ratio_median;
        let ratio_min = scaling
            .iter()
            .map(|stats| stats.ratio_median)
            .fold(f64::INFINITY, f64::min);
        let ratio_max = scaling
            .iter()
            .map(|stats| stats.ratio_median)
            .fold(f64::NEG_INFINITY, f64::max);
        let ratio_spread = ratio_max / ratio_min - 1.0;
        let monotonic_up = scaling
            .windows(2)
            .all(|points| points[1].ratio_median >= points[0].ratio_median);
        let monotonic_down = scaling
            .windows(2)
            .all(|points| points[1].ratio_median <= points[0].ratio_median);
        let shape = if ratio_spread <= 0.15 {
            "FLAT_PER_SAMPLE_COST"
        } else if monotonic_up && last_ratio >= first_ratio * 1.15 {
            "WIDENING_WITH_STREAM"
        } else if monotonic_down && last_ratio <= first_ratio * 0.85 {
            "NARROWING_WITH_STREAM"
        } else {
            "MIXED_OR_NOISE"
        };
        println!(
            "SCALING_SHAPE workload=f16_dynamic_range_audit dimension=sample_count \
             sizes=[{},{},{}] ratios=[{:.6},{:.6},{:.6}] \
             ratio_spread={ratio_spread:.6} classification={shape}",
            SAMPLE_SIZES[0],
            SAMPLE_SIZES[1],
            SAMPLE_SIZES[2],
            scaling[0].ratio_median,
            scaling[1].ratio_median,
            scaling[2].ratio_median,
        );
    });
}

/// Class-3 integration workload: convert a realistic half-precision planar
/// vector field to magnitude/heading, then emit per-frame mean and peak
/// magnitude. Both arms perform the same four public calls over the same
/// inputs. NumPy has no f16 ALU for the two binary transcendentals, while the
/// FrankenNumPy routes are exact parallel widen/operate/narrow kernels.
fn bench_realistic_f16_vector_field_polar_report_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    const FRAME_COUNT: usize = 4_096;
    const SAMPLES_PER_FRAME: [usize; 3] = [512, 1_024, 2_048];
    const WORKLOAD_CONTRACT_ROUNDS: usize = 21;
    const WORKLOAD_CONTRACT_MIN_OF: usize = 2;
    const THREAD_ACTIVITY_REPETITIONS: usize = 5;
    const THREADS: &str = "4";

    for variable in [
        "RAYON_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(THREADS),
            "{variable} must be explicitly pinned before workload timing"
        );
    }
    assert_eq!(
        rayon::current_num_threads(),
        THREADS
            .parse::<usize>()
            .expect("thread count constant is numeric"),
        "Rayon pool width does not match the pinned workload configuration"
    );
    println!(
        "WORKLOAD_RUNTIME host={} rayon_threads={} RAYON_NUM_THREADS={} \
         OPENBLAS_NUM_THREADS={} OMP_NUM_THREADS={} MKL_NUM_THREADS={} trj_used=false",
        std::env::var("HOSTNAME").unwrap_or_else(|_| "unavailable".to_owned()),
        rayon::current_num_threads(),
        THREADS,
        THREADS,
        THREADS,
        THREADS,
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_f16_vector_field_polar_workload")
            .expect("f16 vector-field module");
        fnp_python(&module).expect("initialize fnp_python vector-field module");
        let numpy = py.import("numpy").expect("numpy incumbent");

        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(format!(
                r#"
import numpy as np
rng = np.random.default_rng(20260730)
frame_count = {FRAME_COUNT}
samples_per_frame = {SAMPLES_PER_FRAME:?}
max_width = samples_per_frame[-1]

# A rotating planar flow field with frame-level speed/phase variation,
# low-frequency gusts, and independent component noise.
frame_phase = rng.uniform(-np.pi, np.pi, size=(frame_count, 1))
frame_speed = rng.lognormal(mean=2.2, sigma=0.55, size=(frame_count, 1))
sample_phase = np.linspace(0.0, 8.0 * np.pi, max_width, dtype=np.float32)[None, :]
gust = 1.0 + 0.22 * np.sin(0.37 * sample_phase + frame_phase)
turn = frame_phase + 0.31 * np.sin(0.19 * sample_phase + 0.5 * frame_phase)
speed = frame_speed * gust
noise_x = rng.normal(0.0, 0.18, size=(frame_count, max_width))
noise_y = rng.normal(0.0, 0.18, size=(frame_count, max_width))
velocity_x = np.clip(speed * np.cos(turn) + noise_x, -2048.0, 2048.0).astype(np.float16)
velocity_y = np.clip(speed * np.sin(turn) + noise_y, -2048.0, 2048.0).astype(np.float16)
velocity_x_by_width = [
    np.ascontiguousarray(velocity_x[:, :width]) for width in samples_per_frame
]
velocity_y_by_width = [
    np.ascontiguousarray(velocity_y[:, :width]) for width in samples_per_frame
]

assert velocity_x.dtype == np.float16
assert velocity_y.dtype == np.float16
assert velocity_x.flags.c_contiguous
assert velocity_y.flags.c_contiguous
assert np.isfinite(velocity_x).all()
assert np.isfinite(velocity_y).all()
assert np.max(np.abs(velocity_x)) < np.float16(32768.0)
assert np.max(np.abs(velocity_y)) < np.float16(32768.0)
"#
            ))
            .expect("f16 vector-field setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("f16 vector-field corpus setup");

        let fnp_hypot = module.getattr("hypot").expect("fnp hypot");
        let np_hypot = numpy.getattr("hypot").expect("numpy hypot");
        let fnp_arctan2 = module.getattr("arctan2").expect("fnp arctan2");
        let np_arctan2 = numpy.getattr("arctan2").expect("numpy arctan2");
        let fnp_average = module.getattr("average").expect("fnp average");
        let np_average = numpy.getattr("average").expect("numpy average");
        let fnp_maximum = module.getattr("max").expect("fnp max");
        let np_maximum = numpy.getattr("max").expect("numpy max");
        for (candidate, incumbent, surface) in [
            (&fnp_hypot, &np_hypot, "hypot"),
            (&fnp_arctan2, &np_arctan2, "arctan2"),
            (&fnp_average, &np_average, "average"),
            (&fnp_maximum, &np_maximum, "max"),
        ] {
            assert!(
                !candidate.is(incumbent),
                "{surface}: candidate callable aliases NumPy"
            );
            common::report_numpy_incumbent_identity(py, surface, incumbent);
        }
        common::report_incumbent_topology(
            "fnp.workload.f16_vector_field_polar_report",
            "numpy.workload.f16_vector_field_polar_report",
        );
        println!(
            "INCUMBENT_PIPELINE workload=f16_vector_field_polar_report \
             candidate=fnp.hypot+fnp.arctan2+fnp.average+fnp.max \
             incumbent=numpy.hypot+numpy.arctan2+numpy.average+numpy.max \
             shared_timed_component=none inputs_shared_read_only=true"
        );
        println!(
            "ROUTE_PRECONDITIONS workload=f16_vector_field_polar_report \
             dtype=float16 ndim=2 c_contiguous=true native_endian=true finite=true \
             hypot_abs_input_lt=32768 last_axis=true elements_min={} \
             average_weights=none average_returned=false max_keepdims=false \
             contract_rounds={WORKLOAD_CONTRACT_ROUNDS} \
             contract_min_of={WORKLOAD_CONTRACT_MIN_OF}",
            FRAME_COUNT * SAMPLES_PER_FRAME[0],
        );

        let incumbent = VectorFieldPolarArm {
            hypot: np_hypot,
            arctan2: np_arctan2,
            average: np_average,
            maximum: np_maximum,
        };
        let candidate = VectorFieldPolarArm {
            hypot: fnp_hypot,
            arctan2: fnp_arctan2,
            average: fnp_average,
            maximum: fnp_maximum,
        };
        let velocity_x_by_width = namespace
            .get_item("velocity_x_by_width")
            .expect("vector-field x arrays by width");
        let velocity_y_by_width = namespace
            .get_item("velocity_y_by_width")
            .expect("vector-field y arrays by width");

        let mut scaling = Vec::with_capacity(SAMPLES_PER_FRAME.len());
        for (size_index, width) in SAMPLES_PER_FRAME.into_iter().enumerate() {
            let elements = FRAME_COUNT * width;
            let row = format!("workload_f16_vector_field_polar_{FRAME_COUNT}x{width}");
            println!(
                "WORKLOAD_CONFIG row={row} user_job=vector_field_polar_report \
                 frames={FRAME_COUNT} samples_per_frame={width} elements={elements} \
                 dtype=float16 distribution=rotating_lognormal_flow_plus_gusts_and_noise \
                 stages=hypot_magnitude,arctan2_heading,average_magnitude_axis1,max_magnitude_axis1 \
                 output=magnitude_heading_per_frame_mean_per_frame_peak \
                 matched_config=same_process_same_inputs_same_axis square_gemm_excluded=true"
            );
            println!(
                "WORK_ACCOUNTING row={row} candidate_public_calls=4 incumbent_public_calls=4 \
                 candidate_hypot_pairs={elements} incumbent_hypot_pairs={elements} \
                 candidate_arctan2_pairs={elements} incumbent_arctan2_pairs={elements} \
                 candidate_average_elements={elements} incumbent_average_elements={elements} \
                 candidate_max_elements={elements} incumbent_max_elements={elements} \
                 candidate_logical_element_visits={} incumbent_logical_element_visits={} \
                 candidate_input_element_reads={} incumbent_input_element_reads={} \
                 candidate_output_elements={} incumbent_output_elements={} \
                 more_work_verdict=equal_public_work",
                4 * elements,
                4 * elements,
                6 * elements,
                6 * elements,
                2 * elements + 2 * FRAME_COUNT,
                2 * elements + 2 * FRAME_COUNT,
            );
            let velocity_x = velocity_x_by_width
                .get_item(size_index)
                .expect("vector-field x array for width");
            let velocity_y = velocity_y_by_width
                .get_item(size_index)
                .expect("vector-field y array for width");

            let (_, incumbent_output) = incumbent.run(&velocity_x, &velocity_y);
            let (_, candidate_output) = candidate.run(&velocity_x, &velocity_y);
            assert_workload_outputs_equal(&numpy, &row, &candidate_output, &incumbent_output);
            println!(
                "PARITY row={row} outputs=4 dtype_shape_byte_identity=passed checksum={:016x}",
                workload_checksum(&numpy, &candidate_output),
            );

            common::report_observed_thread_activity(
                &row,
                "numpy",
                THREAD_ACTIVITY_REPETITIONS,
                || {
                    let (_, outputs) = incumbent.run(&velocity_x, &velocity_y);
                    black_box(outputs);
                },
            );
            common::report_observed_thread_activity(
                &row,
                "fnp",
                THREAD_ACTIVITY_REPETITIONS,
                || {
                    let (_, outputs) = candidate.run(&velocity_x, &velocity_y);
                    black_box(outputs);
                },
            );

            let incumbent_profile = incumbent.profile(&velocity_x, &velocity_y);
            let candidate_profile = candidate.profile(&velocity_x, &velocity_y);
            let candidate_total = candidate_profile.iter().sum::<f64>();
            let (target_stage, target_ms) = [
                ("hypot_magnitude", candidate_profile[0]),
                ("arctan2_heading", candidate_profile[1]),
                ("average_magnitude_axis1", candidate_profile[2]),
                ("max_magnitude_axis1", candidate_profile[3]),
            ]
            .into_iter()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .expect("four vector-field profile stages");
            for (stage, incumbent_ms, candidate_ms) in [
                (
                    "hypot_magnitude",
                    incumbent_profile[0],
                    candidate_profile[0],
                ),
                (
                    "arctan2_heading",
                    incumbent_profile[1],
                    candidate_profile[1],
                ),
                (
                    "average_magnitude_axis1",
                    incumbent_profile[2],
                    candidate_profile[2],
                ),
                (
                    "max_magnitude_axis1",
                    incumbent_profile[3],
                    candidate_profile[3],
                ),
            ] {
                println!(
                    "PROFILE_STAGE row={row} arm_pair=numpy_fnp stage={stage} \
                     incumbent_median_ms={incumbent_ms:.6} \
                     candidate_median_ms={candidate_ms:.6}"
                );
            }
            let target_fraction = target_ms / candidate_total;
            println!(
                "PROFILE_SUMMARY row={row} target_stage={target_stage} \
                 target_candidate_ms={target_ms:.6} candidate_stage_sum_ms={candidate_total:.6} \
                 target_self_fraction_pct={:.3} amdahl_remove_ceiling={:.6}",
                target_fraction * 100.0,
                1.0 / (1.0 - target_fraction),
            );

            let mut observe_incumbent = || {
                let (elapsed, outputs) = incumbent.run(&velocity_x, &velocity_y);
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let mut observe_candidate = || {
                let (elapsed, outputs) = candidate.run(&velocity_x, &velocity_y);
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract_with_sampling(
                    &row,
                    &mut observe_incumbent,
                    &mut observe_candidate,
                    WORKLOAD_CONTRACT_ROUNDS,
                    WORKLOAD_CONTRACT_MIN_OF,
                );
            println!(
                "WORKLOAD_SIZE_POINT workload=f16_vector_field_polar_report \
                 frames={FRAME_COUNT} samples_per_frame={width} elements={elements} \
                 incumbent_median_ms={:.6} candidate_median_ms={:.6} \
                 ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
                 incumbent_null_ratio={:.6} candidate_null_ratio={:.6} \
                 incumbent_ns_per_element={:.3} candidate_ns_per_element={:.3}",
                effect.arm_a_median_ns / 1_000_000.0,
                effect.arm_b_median_ns / 1_000_000.0,
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                incumbent_null.ratio_median,
                candidate_null.ratio_median,
                effect.arm_a_median_ns / elements as f64,
                effect.arm_b_median_ns / elements as f64,
            );
            scaling.push(effect);
        }

        let first_ratio = scaling.first().expect("first scaling point").ratio_median;
        let last_ratio = scaling.last().expect("last scaling point").ratio_median;
        let ratio_min = scaling
            .iter()
            .map(|stats| stats.ratio_median)
            .fold(f64::INFINITY, f64::min);
        let ratio_max = scaling
            .iter()
            .map(|stats| stats.ratio_median)
            .fold(f64::NEG_INFINITY, f64::max);
        let ratio_spread = ratio_max / ratio_min - 1.0;
        let monotonic_up = scaling
            .windows(2)
            .all(|points| points[1].ratio_median >= points[0].ratio_median);
        let monotonic_down = scaling
            .windows(2)
            .all(|points| points[1].ratio_median <= points[0].ratio_median);
        let shape = if ratio_spread <= 0.15 {
            "FLAT_PER_ELEMENT_COST"
        } else if monotonic_up && last_ratio >= first_ratio * 1.15 {
            "WIDENING_WITH_FIELD"
        } else if monotonic_down && last_ratio <= first_ratio * 0.85 {
            "NARROWING_WITH_FIELD"
        } else {
            "MIXED_OR_NOISE"
        };
        println!(
            "SCALING_SHAPE workload=f16_vector_field_polar_report \
             dimension=samples_per_frame frames={FRAME_COUNT} \
             widths=[{},{},{}] elements=[{},{},{}] \
             ratios=[{:.6},{:.6},{:.6}] ratio_spread={ratio_spread:.6} \
             classification={shape}",
            SAMPLES_PER_FRAME[0],
            SAMPLES_PER_FRAME[1],
            SAMPLES_PER_FRAME[2],
            FRAME_COUNT * SAMPLES_PER_FRAME[0],
            FRAME_COUNT * SAMPLES_PER_FRAME[1],
            FRAME_COUNT * SAMPLES_PER_FRAME[2],
            scaling[0].ratio_median,
            scaling[1].ratio_median,
            scaling[2].ratio_median,
        );
    });
}

/// Class-3 integration workload: propagate sparse account-role assignments
/// through a role-permission risk matrix, then emit per-account total and peak
/// exposure plus a fleet total. Both arms execute the same four public calls
/// and the same mathematical operation counts. NumPy has no integer BLAS, so
/// its int64 matrix product uses the generic integer loop while FrankenNumPy
/// takes the existing safe-Rust tiled kernel.
fn bench_realistic_access_control_exposure_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    const ACCOUNT_COUNTS: [usize; 3] = [2_048, 4_096, 8_192];
    const ROLE_COUNT: usize = 128;
    const PERMISSION_COUNT: usize = 256;
    const WORKLOAD_CONTRACT_ROUNDS: usize = 21;
    const WORKLOAD_CONTRACT_MIN_OF: usize = 2;
    const THREAD_ACTIVITY_REPETITIONS: usize = 11;
    const THREADS: &str = "4";

    for variable in [
        "RAYON_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(THREADS),
            "{variable} must be explicitly pinned before workload timing"
        );
    }
    assert_eq!(
        rayon::current_num_threads(),
        THREADS
            .parse::<usize>()
            .expect("thread count constant is numeric"),
        "Rayon pool width does not match the pinned workload configuration"
    );
    println!(
        "WORKLOAD_RUNTIME host={} rayon_threads={} RAYON_NUM_THREADS={} \
         OPENBLAS_NUM_THREADS={} OMP_NUM_THREADS={} MKL_NUM_THREADS={} trj_used=false",
        std::env::var("HOSTNAME").unwrap_or_else(|_| "unavailable".to_owned()),
        rayon::current_num_threads(),
        THREADS,
        THREADS,
        THREADS,
        THREADS,
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_access_control_exposure_workload")
            .expect("access-control exposure module");
        fnp_python(&module).expect("initialize fnp_python access-control exposure module");
        let numpy = py.import("numpy").expect("numpy incumbent");

        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(format!(
                r#"
import numpy as np
rng = np.random.default_rng(20260730)
account_counts = {ACCOUNT_COUNTS:?}
role_count = {ROLE_COUNT}
permission_count = {PERMISSION_COUNT}

# Fixed integer risk weights for every role/permission pair. Zero weights model
# permissions irrelevant to a role; positive weights are bounded severity
# points, not floating estimates.
role_permissions = rng.integers(
    0,
    32,
    size=(role_count, permission_count),
    dtype=np.int64,
)

account_roles_by_size = []
for account_count in account_counts:
    # Most accounts have no assignment to most roles. A deterministic hot-role
    # increment guarantees at least one assigned role per account while
    # retaining a realistic sparse count matrix.
    account_roles = rng.binomial(
        3,
        0.08,
        size=(account_count, role_count),
    ).astype(np.int64, copy=False)
    hot_role = rng.integers(
        0,
        role_count,
        size=account_count,
        dtype=np.int64,
    )
    account_roles[np.arange(account_count), hot_role] += np.int64(1)
    account_roles_by_size.append(np.ascontiguousarray(account_roles))

assert role_permissions.dtype == np.int64
assert role_permissions.ndim == 2
assert role_permissions.flags.c_contiguous
assert int(role_permissions.min()) >= 0
assert int(role_permissions.max()) <= 31
for account_roles in account_roles_by_size:
    assert account_roles.dtype == np.int64
    assert account_roles.ndim == 2
    assert account_roles.shape[1] == role_count
    assert account_roles.flags.c_contiguous
    assert int(account_roles.min()) >= 0
    assert int(account_roles.max()) <= 4

# The largest possible intermediate and aggregation remain far below int64
# overflow, so exact parity is not relying on wraparound.
assert role_count * 4 * 31 < np.iinfo(np.int64).max
assert (
    max(account_counts) * permission_count * role_count * 4 * 31
    < np.iinfo(np.int64).max
)
"#
            ))
            .expect("access-control exposure setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("access-control exposure corpus setup");

        let fnp_matmul = module.getattr("matmul").expect("fnp matmul");
        let np_matmul = numpy.getattr("matmul").expect("numpy matmul");
        let fnp_sum = module.getattr("sum").expect("fnp sum");
        let np_sum = numpy.getattr("sum").expect("numpy sum");
        let fnp_maximum = module.getattr("max").expect("fnp max");
        let np_maximum = numpy.getattr("max").expect("numpy max");
        for (candidate, incumbent, surface) in [
            (&fnp_matmul, &np_matmul, "matmul"),
            (&fnp_sum, &np_sum, "sum"),
            (&fnp_maximum, &np_maximum, "max"),
        ] {
            assert!(
                !candidate.is(incumbent),
                "{surface}: candidate callable aliases NumPy"
            );
            common::report_numpy_incumbent_identity(py, surface, incumbent);
        }
        // Truthful isolation record. Two of the candidate's four public stages
        // are source-pinned NumPy passthroughs, so this pair is NOT
        // shared_timed_component=none and must not borrow that claim.
        common::report_incumbent_topology_with_shared_component(
            "fnp.workload.access_control_exposure_report",
            "numpy.workload.access_control_exposure_report",
            "numpy.sum_int64_axis1,numpy.sum_int64_flat",
        );
        println!(
            "INCUMBENT_PIPELINE workload=access_control_exposure_report \
             candidate=fnp.matmul+fnp.sum+fnp.max+fnp.sum \
             incumbent=numpy.matmul+numpy.sum+numpy.max+numpy.sum \
             shared_timed_component=numpy.sum_int64_axis1,numpy.sum_int64_flat \
             native_candidate_stages=matmul_exposure,account_max_axis1 \
             delegated_candidate_stages=account_sum_axis1,fleet_sum \
             inputs_shared_read_only=true"
        );
        // Source-pinned routing for every stage the candidate executes. These
        // pins are read from the shipped dispatch, not inferred from timings;
        // the per-stage profile below is the independent empirical corroboration
        // (a delegated stage must land near parity with the incumbent).
        for (stage, route, pin) in [
            (
                "matmul_exposure",
                "native_int64_tiled_gemm",
                "fnp-python/src/lib.rs:try_native_int_matmul(int64,2d,2d,contiguous,work>=1<<18,threads>=2)",
            ),
            (
                "account_sum_axis1",
                "delegated_to_numpy_sum",
                "fnp-python/src/lib.rs:sum(native routes cover f64 last-axis and f16 only; int64 axis falls through to numpy.sum)",
            ),
            (
                "account_max_axis1",
                "native_int64_zerocopy_minmax",
                "fnp-python/src/lib.rs:py_max->try_zerocopy_int_minmax(int64,contiguous,integer axis)",
            ),
            (
                "fleet_sum",
                "delegated_to_numpy_sum",
                "fnp-python/src/lib.rs:sum(axis=None,int64 -> numpy.sum passthrough)",
            ),
        ] {
            println!(
                "ROUTE_DISCLOSURE workload=access_control_exposure_report stage={stage} \
                 candidate_route={route} source_pin={pin}"
            );
        }
        println!(
            "ROUTE_PRECONDITIONS workload=access_control_exposure_report \
             dtype=int64 lhs_ndim=2 rhs_ndim=2 c_contiguous=true \
             same_dtype=true matching_inner_dim={ROLE_COUNT} \
             min_matmul_work={} int_matmul_min_work={} \
             rayon_threads_min=2 axis_reductions=1 fleet_sum_axis=none \
             overflow_headroom=proven contract_rounds={WORKLOAD_CONTRACT_ROUNDS} \
             contract_min_of={WORKLOAD_CONTRACT_MIN_OF}",
            ACCOUNT_COUNTS[0] * ROLE_COUNT * PERMISSION_COUNT,
            1 << 18,
        );

        let incumbent = AccessControlExposureArm {
            matmul: np_matmul,
            sum: np_sum,
            maximum: np_maximum,
        };
        let candidate = AccessControlExposureArm {
            matmul: fnp_matmul,
            sum: fnp_sum,
            maximum: fnp_maximum,
        };
        let account_roles_by_size = namespace
            .get_item("account_roles_by_size")
            .expect("account-role matrices by size");
        let role_permissions = namespace
            .get_item("role_permissions")
            .expect("role-permission risk weights");

        let mut scaling = Vec::with_capacity(ACCOUNT_COUNTS.len());
        for (size_index, account_count) in ACCOUNT_COUNTS.into_iter().enumerate() {
            let matmul_outputs = account_count * PERMISSION_COUNT;
            let matmul_multiplications = matmul_outputs * ROLE_COUNT;
            let matmul_additions = matmul_outputs * (ROLE_COUNT - 1);
            let account_sum_additions = account_count * (PERMISSION_COUNT - 1);
            let account_max_comparisons = account_count * (PERMISSION_COUNT - 1);
            let fleet_sum_additions = account_count - 1;
            let output_elements = matmul_outputs + 2 * account_count + 1;
            let row = format!(
                "workload_access_control_exposure_{account_count}x{ROLE_COUNT}x{PERMISSION_COUNT}"
            );
            println!(
                "WORKLOAD_CONFIG row={row} user_job=access_control_exposure_report \
                 accounts={account_count} roles={ROLE_COUNT} \
                 permissions={PERMISSION_COUNT} dtype=int64 \
                 distribution=sparse_binomial_role_counts_plus_one_hot_role \
                 stages=matmul_exposure,account_sum_axis1,account_max_axis1,fleet_sum \
                 output=exposure_matrix_per_account_total_per_account_peak_fleet_total \
                 matched_config=same_process_same_inputs_same_axes \
                 target_user=access_control_risk_analytics"
            );
            println!(
                "WORK_ACCOUNTING row={row} candidate_public_calls=4 incumbent_public_calls=4 \
                 candidate_matmul_multiplications={matmul_multiplications} \
                 incumbent_matmul_multiplications={matmul_multiplications} \
                 candidate_matmul_additions={matmul_additions} \
                 incumbent_matmul_additions={matmul_additions} \
                 candidate_account_sum_additions={account_sum_additions} \
                 incumbent_account_sum_additions={account_sum_additions} \
                 candidate_account_max_comparisons={account_max_comparisons} \
                 incumbent_account_max_comparisons={account_max_comparisons} \
                 candidate_fleet_sum_additions={fleet_sum_additions} \
                 incumbent_fleet_sum_additions={fleet_sum_additions} \
                 candidate_output_elements={output_elements} \
                 incumbent_output_elements={output_elements} \
                 more_work_verdict=equal_mathematical_work"
            );
            let account_roles = account_roles_by_size
                .get_item(size_index)
                .expect("account-role matrix for size");

            let (_, incumbent_output) = incumbent.run(&account_roles, &role_permissions);
            let (_, candidate_output) = candidate.run(&account_roles, &role_permissions);
            assert_workload_outputs_equal(&numpy, &row, &candidate_output, &incumbent_output);
            println!(
                "PARITY row={row} outputs=4 dtype_shape_byte_identity=passed \
                 no_overflow_precondition=passed checksum={:016x}",
                workload_checksum(&numpy, &candidate_output),
            );

            common::report_observed_thread_activity(
                &row,
                "numpy",
                THREAD_ACTIVITY_REPETITIONS,
                || {
                    let (_, outputs) = incumbent.run(&account_roles, &role_permissions);
                    black_box(outputs);
                },
            );
            common::report_observed_thread_activity(
                &row,
                "fnp",
                THREAD_ACTIVITY_REPETITIONS,
                || {
                    let (_, outputs) = candidate.run(&account_roles, &role_permissions);
                    black_box(outputs);
                },
            );

            let incumbent_profile = incumbent.profile(&account_roles, &role_permissions);
            let candidate_profile = candidate.profile(&account_roles, &role_permissions);
            let candidate_total = candidate_profile.iter().sum::<f64>();
            let (target_stage, target_ms) = [
                ("matmul_exposure", candidate_profile[0]),
                ("account_sum_axis1", candidate_profile[1]),
                ("account_max_axis1", candidate_profile[2]),
                ("fleet_sum", candidate_profile[3]),
            ]
            .into_iter()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .expect("four access-control profile stages");
            for (stage, incumbent_ms, candidate_ms) in [
                (
                    "matmul_exposure",
                    incumbent_profile[0],
                    candidate_profile[0],
                ),
                (
                    "account_sum_axis1",
                    incumbent_profile[1],
                    candidate_profile[1],
                ),
                (
                    "account_max_axis1",
                    incumbent_profile[2],
                    candidate_profile[2],
                ),
                ("fleet_sum", incumbent_profile[3], candidate_profile[3]),
            ] {
                println!(
                    "PROFILE_STAGE row={row} arm_pair=numpy_fnp stage={stage} \
                     incumbent_median_ms={incumbent_ms:.6} \
                     candidate_median_ms={candidate_ms:.6} \
                     stage_ratio_numpy_over_fnp={:.6}",
                    incumbent_ms / candidate_ms,
                );
            }
            // Quantify how much of the candidate's own job time is NumPy code
            // running inside the candidate arm. This bounds how much of the
            // end-to-end ratio can be credited to FrankenNumPy at all, and the
            // delegated stages double as an in-arm parity check: their
            // numpy/fnp stage ratios must sit near unity.
            let delegated_ms = candidate_profile[1] + candidate_profile[3];
            let native_ms = candidate_profile[0] + candidate_profile[2];
            println!(
                "DELEGATION_SHARE row={row} \
                 delegated_stages=account_sum_axis1,fleet_sum \
                 delegated_candidate_ms={delegated_ms:.6} \
                 native_candidate_ms={native_ms:.6} \
                 delegated_share_of_candidate_pct={:.3} \
                 delegated_stage_ratio_account_sum={:.6} \
                 delegated_stage_ratio_fleet_sum={:.6} \
                 expectation=delegated_ratios_near_unity",
                delegated_ms / candidate_total * 100.0,
                incumbent_profile[1] / candidate_profile[1],
                incumbent_profile[3] / candidate_profile[3],
            );
            let target_fraction = target_ms / candidate_total;
            println!(
                "PROFILE_SUMMARY row={row} target_stage={target_stage} \
                 target_candidate_ms={target_ms:.6} candidate_stage_sum_ms={candidate_total:.6} \
                 target_self_fraction_pct={:.3} amdahl_remove_ceiling={:.6}",
                target_fraction * 100.0,
                1.0 / (1.0 - target_fraction),
            );

            let mut observe_incumbent = || {
                let (elapsed, outputs) = incumbent.run(&account_roles, &role_permissions);
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let mut observe_candidate = || {
                let (elapsed, outputs) = candidate.run(&account_roles, &role_permissions);
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract_with_sampling(
                    &row,
                    &mut observe_incumbent,
                    &mut observe_candidate,
                    WORKLOAD_CONTRACT_ROUNDS,
                    WORKLOAD_CONTRACT_MIN_OF,
                );
            println!(
                "WORKLOAD_SIZE_POINT workload=access_control_exposure_report \
                 accounts={account_count} roles={ROLE_COUNT} \
                 permissions={PERMISSION_COUNT} \
                 incumbent_median_ms={:.6} candidate_median_ms={:.6} \
                 ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
                 incumbent_null_ratio={:.6} candidate_null_ratio={:.6} \
                 incumbent_ns_per_matmul_multiplication={:.6} \
                 candidate_ns_per_matmul_multiplication={:.6}",
                effect.arm_a_median_ns / 1_000_000.0,
                effect.arm_b_median_ns / 1_000_000.0,
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                incumbent_null.ratio_median,
                candidate_null.ratio_median,
                effect.arm_a_median_ns / matmul_multiplications as f64,
                effect.arm_b_median_ns / matmul_multiplications as f64,
            );
            scaling.push(effect);
        }

        let first_ratio = scaling.first().expect("first scaling point").ratio_median;
        let last_ratio = scaling.last().expect("last scaling point").ratio_median;
        let ratio_min = scaling
            .iter()
            .map(|stats| stats.ratio_median)
            .fold(f64::INFINITY, f64::min);
        let ratio_max = scaling
            .iter()
            .map(|stats| stats.ratio_median)
            .fold(f64::NEG_INFINITY, f64::max);
        let ratio_spread = ratio_max / ratio_min - 1.0;
        let monotonic_up = scaling
            .windows(2)
            .all(|points| points[1].ratio_median >= points[0].ratio_median);
        let monotonic_down = scaling
            .windows(2)
            .all(|points| points[1].ratio_median <= points[0].ratio_median);
        let shape = if ratio_spread <= 0.15 {
            "FLAT_PER_ACCOUNT_COST"
        } else if monotonic_up && last_ratio >= first_ratio * 1.15 {
            "WIDENING_WITH_ACCOUNTS"
        } else if monotonic_down && last_ratio <= first_ratio * 0.85 {
            "NARROWING_WITH_ACCOUNTS"
        } else {
            "MIXED_OR_NOISE"
        };
        println!(
            "SCALING_SHAPE workload=access_control_exposure_report \
             dimension=account_count roles={ROLE_COUNT} permissions={PERMISSION_COUNT} \
             sizes=[{},{},{}] ratios=[{:.6},{:.6},{:.6}] \
             ratio_spread={ratio_spread:.6} classification={shape}",
            ACCOUNT_COUNTS[0],
            ACCOUNT_COUNTS[1],
            ACCOUNT_COUNTS[2],
            scaling[0].ratio_median,
            scaling[1].ratio_median,
            scaling[2].ratio_median,
        );
        println!(
            "CHOOSER_SCOPE workload=access_control_exposure_report \
             target_user=access_control_risk_analytics dtype=int64 \
             account_range={}..={} roles={ROLE_COUNT} permissions={PERMISSION_COUNT} \
             thread_topology=recorded_above decision_requires_all_three_effects_and_both_nulls \
             generalization_beyond_measured_shape=false",
            ACCOUNT_COUNTS[0], ACCOUNT_COUNTS[2],
        );
    });
}

/// Fully isolated Class-3 integration workload: propagate account-role counts
/// through a role-permission risk matrix, identify each account's peak exposure
/// and the permission responsible for it, then report the fleet-wide spread of
/// those peaks. Every candidate stage has a source-pinned native route; the
/// exposure matrix is deliberately at least 2^23 elements so integer max does
/// not cross its small-array delegation gate.
fn bench_realistic_critical_access_exposure_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    const ACCOUNT_COUNTS: [usize; 3] = [4_096, 8_192, 16_384];
    const ROLE_COUNT: usize = 8;
    const PERMISSION_COUNT: usize = 2_048;
    const WORKLOAD_CONTRACT_ROUNDS: usize = 21;
    const WORKLOAD_CONTRACT_MIN_OF: usize = 2;
    const THREAD_ACTIVITY_REPETITIONS: usize = 11;
    const THREADS: &str = "4";
    const REQUIRED_BUILD_PROFILE: &str = "release-perf";

    assert_eq!(
        std::env::var("FNP_BENCH_PROFILE").as_deref(),
        Ok(REQUIRED_BUILD_PROFILE),
        "ship-grade critical-access evidence requires FNP_BENCH_PROFILE=release-perf"
    );
    for variable in [
        "RAYON_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(THREADS),
            "{variable} must be explicitly pinned before workload timing"
        );
    }
    assert_eq!(
        rayon::current_num_threads(),
        THREADS
            .parse::<usize>()
            .expect("thread count constant is numeric"),
        "Rayon pool width does not match the pinned workload configuration"
    );
    println!(
        "WORKLOAD_RUNTIME host={} build_profile={REQUIRED_BUILD_PROFILE} \
         rayon_threads={} RAYON_NUM_THREADS={} OPENBLAS_NUM_THREADS={} \
         OMP_NUM_THREADS={} MKL_NUM_THREADS={} trj_used=false",
        std::env::var("HOSTNAME").unwrap_or_else(|_| "unavailable".to_owned()),
        rayon::current_num_threads(),
        THREADS,
        THREADS,
        THREADS,
        THREADS,
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_critical_access_exposure_workload")
            .expect("critical-access exposure module");
        fnp_python(&module).expect("initialize fnp_python critical-access exposure module");
        let numpy = py.import("numpy").expect("numpy incumbent");

        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(format!(
                r#"
import numpy as np
rng = np.random.default_rng(20260730)
account_counts = {ACCOUNT_COUNTS:?}
role_count = {ROLE_COUNT}
permission_count = {PERMISSION_COUNT}

# Eight coarse organizational roles map onto a broad enterprise entitlement
# catalog. Integer values are auditable count/severity points, not estimates.
role_permissions = rng.integers(
    0,
    32,
    size=(role_count, permission_count),
    dtype=np.int64,
)

account_roles_by_size = []
for account_count in account_counts:
    account_roles = rng.binomial(
        3,
        0.08,
        size=(account_count, role_count),
    ).astype(np.int64, copy=False)
    hot_role = rng.integers(
        0,
        role_count,
        size=account_count,
        dtype=np.int64,
    )
    account_roles[np.arange(account_count), hot_role] += np.int64(1)
    account_roles_by_size.append(np.ascontiguousarray(account_roles))

assert min(account_counts) * permission_count >= (1 << 23)
assert role_permissions.dtype == np.int64
assert role_permissions.ndim == 2
assert role_permissions.flags.c_contiguous
assert int(role_permissions.min()) >= 0
assert int(role_permissions.max()) <= 31
for account_roles in account_roles_by_size:
    assert account_roles.dtype == np.int64
    assert account_roles.ndim == 2
    assert account_roles.shape[1] == role_count
    assert account_roles.flags.c_contiguous
    assert int(account_roles.min()) >= 0
    assert int(account_roles.max()) <= 4

# Every exposure element is bounded by 8 * 4 * 31 = 992. The matrix product,
# max, argmax, and peak-to-peak stages therefore avoid overflow and exceptional
# values; parity does not rely on wrapping or undefined tie behavior.
assert role_count * 4 * 31 < np.iinfo(np.int64).max
"#
            ))
            .expect("critical-access exposure setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("critical-access exposure corpus setup");

        let fnp_matmul = module.getattr("matmul").expect("fnp matmul");
        let np_matmul = numpy.getattr("matmul").expect("numpy matmul");
        let fnp_maximum = module.getattr("max").expect("fnp max");
        let np_maximum = numpy.getattr("max").expect("numpy max");
        let fnp_argmax = module.getattr("argmax").expect("fnp argmax");
        let np_argmax = numpy.getattr("argmax").expect("numpy argmax");
        let fnp_ptp = module.getattr("ptp").expect("fnp ptp");
        let np_ptp = numpy.getattr("ptp").expect("numpy ptp");
        for (candidate, incumbent, surface) in [
            (&fnp_matmul, &np_matmul, "matmul"),
            (&fnp_maximum, &np_maximum, "max"),
            (&fnp_argmax, &np_argmax, "argmax"),
            (&fnp_ptp, &np_ptp, "ptp"),
        ] {
            assert!(
                !candidate.is(incumbent),
                "{surface}: candidate callable aliases NumPy"
            );
            common::report_numpy_incumbent_identity(py, surface, incumbent);
        }
        common::report_incumbent_topology(
            "fnp.workload.critical_access_exposure_report",
            "numpy.workload.critical_access_exposure_report",
        );
        println!(
            "INCUMBENT_PIPELINE workload=critical_access_exposure_report \
             candidate=fnp.matmul+fnp.max+fnp.argmax+fnp.ptp \
             incumbent=numpy.matmul+numpy.max+numpy.argmax+numpy.ptp \
             shared_timed_component=none candidate_stages_all_native=true \
             inputs_shared_read_only=true"
        );
        for (stage, route, pin) in [
            (
                "matmul_exposure",
                "native_int64_tiled_gemm",
                "fnp-python/src/lib.rs:try_native_int_matmul(int64,2d,2d,same-dtype,work>=1<<18,threads>=2)",
            ),
            (
                "account_max_axis1",
                "native_int64_zerocopy_minmax",
                "fnp-python/src/lib.rs:try_zerocopy_int_minmax(int64,size>=1<<23,axis=1)",
            ),
            (
                "account_argmax_axis1",
                "native_int64_lastaxis_argextreme",
                "fnp-python/src/lib.rs:try_zerocopy_lastaxis_argextreme(int64,c-contiguous,last-axis,total>=1<<20)",
            ),
            (
                "fleet_peak_ptp",
                "native_int64_zerocopy_ptp",
                "fnp-python/src/lib.rs:try_zerocopy_int_ptp(int64,axis=None,nonempty-contiguous)",
            ),
        ] {
            println!(
                "ROUTE_DISCLOSURE workload=critical_access_exposure_report stage={stage} \
                 candidate_route={route} source_pin={pin}"
            );
        }
        println!(
            "ROUTE_PRECONDITIONS workload=critical_access_exposure_report \
             dtype=int64 lhs_ndim=2 rhs_ndim=2 c_contiguous=true \
             same_dtype=true matching_inner_dim={ROLE_COUNT} \
             min_matmul_work={} int_matmul_min_work={} \
             min_exposure_elements={} int_max_native_min={} \
             argmax_lastaxis_parallel_min={} rayon_threads_min=2 \
             reductions=max_axis1,argmax_axis1,ptp_axis_none \
             overflow_headroom=proven contract_rounds={WORKLOAD_CONTRACT_ROUNDS} \
             contract_min_of={WORKLOAD_CONTRACT_MIN_OF}",
            ACCOUNT_COUNTS[0] * ROLE_COUNT * PERMISSION_COUNT,
            1 << 18,
            ACCOUNT_COUNTS[0] * PERMISSION_COUNT,
            1 << 23,
            1 << 20,
        );

        let incumbent = CriticalAccessExposureArm {
            matmul: np_matmul,
            maximum: np_maximum,
            argmax: np_argmax,
            ptp: np_ptp,
        };
        let candidate = CriticalAccessExposureArm {
            matmul: fnp_matmul,
            maximum: fnp_maximum,
            argmax: fnp_argmax,
            ptp: fnp_ptp,
        };
        let account_roles_by_size = namespace
            .get_item("account_roles_by_size")
            .expect("account-role matrices by size");
        let role_permissions = namespace
            .get_item("role_permissions")
            .expect("role-permission risk weights");

        let mut scaling = Vec::with_capacity(ACCOUNT_COUNTS.len());
        for (size_index, account_count) in ACCOUNT_COUNTS.into_iter().enumerate() {
            let matmul_outputs = account_count * PERMISSION_COUNT;
            let matmul_multiplications = matmul_outputs * ROLE_COUNT;
            let matmul_additions = matmul_outputs * (ROLE_COUNT - 1);
            let account_max_comparisons = account_count * (PERMISSION_COUNT - 1);
            let account_argmax_extreme_comparisons = account_count * (PERMISSION_COUNT - 1);
            let fleet_ptp_comparisons = 2 * (account_count - 1);
            let output_elements = matmul_outputs + 2 * account_count + 1;
            let row = format!(
                "workload_critical_access_exposure_{account_count}x{ROLE_COUNT}x{PERMISSION_COUNT}"
            );
            println!(
                "WORKLOAD_CONFIG row={row} user_job=critical_access_exposure_report \
                 accounts={account_count} roles={ROLE_COUNT} \
                 permissions={PERMISSION_COUNT} dtype=int64 \
                 distribution=sparse_binomial_role_counts_plus_one_hot_role \
                 stages=matmul_exposure,account_max_axis1,account_argmax_axis1,fleet_peak_ptp \
                 output=exposure_matrix_per_account_peak_critical_permission_fleet_peak_spread \
                 matched_config=same_process_same_inputs_same_axes \
                 target_user=enterprise_access_review"
            );
            let account_roles = account_roles_by_size
                .get_item(size_index)
                .expect("account-role matrix for size");

            let (_, incumbent_output) = incumbent.run(&account_roles, &role_permissions);
            let (_, candidate_output) = candidate.run(&account_roles, &role_permissions);
            assert_workload_outputs_equal(&numpy, &row, &candidate_output, &incumbent_output);
            let critical_indices = candidate_output[2]
                .call_method0("tolist")
                .expect("critical-permission indices convert to list")
                .extract::<Vec<usize>>()
                .expect("critical-permission index list");
            let candidate_argmax_equality_checks = critical_indices
                .iter()
                .map(|index| *index + 1)
                .sum::<usize>();
            println!(
                "WORK_ACCOUNTING row={row} candidate_public_calls=4 incumbent_public_calls=4 \
                 candidate_matmul_multiplications={matmul_multiplications} \
                 incumbent_matmul_multiplications={matmul_multiplications} \
                 candidate_matmul_additions={matmul_additions} \
                 incumbent_matmul_additions={matmul_additions} \
                 candidate_account_max_comparisons={account_max_comparisons} \
                 incumbent_account_max_comparisons={account_max_comparisons} \
                 candidate_account_argmax_extreme_comparisons={account_argmax_extreme_comparisons} \
                 incumbent_account_argmax_extreme_comparisons={account_argmax_extreme_comparisons} \
                 candidate_argmax_first_index_equality_checks={candidate_argmax_equality_checks} \
                 candidate_fleet_ptp_comparisons={fleet_ptp_comparisons} \
                 incumbent_fleet_ptp_comparisons={fleet_ptp_comparisons} \
                 candidate_output_elements={output_elements} \
                 incumbent_output_elements={output_elements} \
                 more_work_verdict=candidate_does_more_counted_argmax_work \
                 extra_candidate_comparisons={candidate_argmax_equality_checks}"
            );
            println!(
                "PARITY row={row} outputs=4 dtype_shape_byte_identity=passed \
                 no_overflow_precondition=passed checksum={:016x}",
                workload_checksum(&numpy, &candidate_output),
            );

            common::report_observed_thread_activity(
                &row,
                "numpy",
                THREAD_ACTIVITY_REPETITIONS,
                || {
                    let (_, outputs) = incumbent.run(&account_roles, &role_permissions);
                    black_box(outputs);
                },
            );
            common::report_observed_thread_activity(
                &row,
                "fnp",
                THREAD_ACTIVITY_REPETITIONS,
                || {
                    let (_, outputs) = candidate.run(&account_roles, &role_permissions);
                    black_box(outputs);
                },
            );

            let incumbent_profile = incumbent.profile(&account_roles, &role_permissions);
            let candidate_profile = candidate.profile(&account_roles, &role_permissions);
            let candidate_total = candidate_profile.iter().sum::<f64>();
            let (target_stage, target_ms) = [
                ("matmul_exposure", candidate_profile[0]),
                ("account_max_axis1", candidate_profile[1]),
                ("account_argmax_axis1", candidate_profile[2]),
                ("fleet_peak_ptp", candidate_profile[3]),
            ]
            .into_iter()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .expect("four critical-access profile stages");
            for (stage, incumbent_ms, candidate_ms) in [
                (
                    "matmul_exposure",
                    incumbent_profile[0],
                    candidate_profile[0],
                ),
                (
                    "account_max_axis1",
                    incumbent_profile[1],
                    candidate_profile[1],
                ),
                (
                    "account_argmax_axis1",
                    incumbent_profile[2],
                    candidate_profile[2],
                ),
                ("fleet_peak_ptp", incumbent_profile[3], candidate_profile[3]),
            ] {
                println!(
                    "PROFILE_STAGE row={row} arm_pair=numpy_fnp stage={stage} \
                     incumbent_median_ms={incumbent_ms:.6} \
                     candidate_median_ms={candidate_ms:.6} \
                     stage_ratio_numpy_over_fnp={:.6}",
                    incumbent_ms / candidate_ms,
                );
            }
            let target_fraction = target_ms / candidate_total;
            println!(
                "PROFILE_SUMMARY row={row} target_stage={target_stage} \
                 target_candidate_ms={target_ms:.6} candidate_stage_sum_ms={candidate_total:.6} \
                 target_self_fraction_pct={:.3} amdahl_remove_ceiling={:.6} \
                 shared_timed_component=none",
                target_fraction * 100.0,
                1.0 / (1.0 - target_fraction),
            );

            let mut observe_incumbent = || {
                let (elapsed, outputs) = incumbent.run(&account_roles, &role_permissions);
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let mut observe_candidate = || {
                let (elapsed, outputs) = candidate.run(&account_roles, &role_permissions);
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract_with_sampling(
                    &row,
                    &mut observe_incumbent,
                    &mut observe_candidate,
                    WORKLOAD_CONTRACT_ROUNDS,
                    WORKLOAD_CONTRACT_MIN_OF,
                );
            println!(
                "WORKLOAD_SIZE_POINT workload=critical_access_exposure_report \
                 accounts={account_count} roles={ROLE_COUNT} \
                 permissions={PERMISSION_COUNT} \
                 incumbent_median_ms={:.6} candidate_median_ms={:.6} \
                 ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
                 incumbent_null_ratio={:.6} candidate_null_ratio={:.6} \
                 incumbent_ns_per_matmul_multiplication={:.6} \
                 candidate_ns_per_matmul_multiplication={:.6}",
                effect.arm_a_median_ns / 1_000_000.0,
                effect.arm_b_median_ns / 1_000_000.0,
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                incumbent_null.ratio_median,
                candidate_null.ratio_median,
                effect.arm_a_median_ns / matmul_multiplications as f64,
                effect.arm_b_median_ns / matmul_multiplications as f64,
            );
            scaling.push(effect);
        }

        let first_ratio = scaling.first().expect("first scaling point").ratio_median;
        let last_ratio = scaling.last().expect("last scaling point").ratio_median;
        let ratio_min = scaling
            .iter()
            .map(|stats| stats.ratio_median)
            .fold(f64::INFINITY, f64::min);
        let ratio_max = scaling
            .iter()
            .map(|stats| stats.ratio_median)
            .fold(f64::NEG_INFINITY, f64::max);
        let ratio_spread = ratio_max / ratio_min - 1.0;
        let monotonic_up = scaling
            .windows(2)
            .all(|points| points[1].ratio_median >= points[0].ratio_median);
        let monotonic_down = scaling
            .windows(2)
            .all(|points| points[1].ratio_median <= points[0].ratio_median);
        let shape = if ratio_spread <= 0.15 {
            "FLAT_PER_ACCOUNT_COST"
        } else if monotonic_up && last_ratio >= first_ratio * 1.15 {
            "WIDENING_WITH_ACCOUNTS"
        } else if monotonic_down && last_ratio <= first_ratio * 0.85 {
            "NARROWING_WITH_ACCOUNTS"
        } else {
            "MIXED_OR_NOISE"
        };
        println!(
            "SCALING_SHAPE workload=critical_access_exposure_report \
             dimension=account_count roles={ROLE_COUNT} permissions={PERMISSION_COUNT} \
             sizes=[{},{},{}] ratios=[{:.6},{:.6},{:.6}] \
             ratio_spread={ratio_spread:.6} classification={shape}",
            ACCOUNT_COUNTS[0],
            ACCOUNT_COUNTS[1],
            ACCOUNT_COUNTS[2],
            scaling[0].ratio_median,
            scaling[1].ratio_median,
            scaling[2].ratio_median,
        );
        println!(
            "CHOOSER_SCOPE workload=critical_access_exposure_report \
             target_user=enterprise_access_review dtype=int64 \
             account_range={}..={} roles={ROLE_COUNT} permissions={PERMISSION_COUNT} \
             candidate_profile={REQUIRED_BUILD_PROFILE} \
             thread_topology=recorded_above decision_requires_all_three_effects_and_both_nulls \
             generalization_beyond_measured_shape=false",
            ACCOUNT_COUNTS[0], ACCOUNT_COUNTS[2],
        );
    });
}

/// Phase-2 Class-3 workload: turn a production request-count stream into the
/// rolling capacity report an SRE uses for saturation review. The 60-second
/// valid convolution produces per-minute rolling totals, maximum.accumulate
/// records the running high-water envelope, and ptp reports demand spread.
///
/// All three public candidate stages stay on independently implemented native
/// int64 paths at every size. In particular, this workload deliberately avoids
/// flat int64 argmax: that route delegates large arrays to NumPy and would make
/// a purported incumbent-isolated job share a timed component.
fn bench_realistic_rolling_load_saturation_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    const SAMPLE_SIZES: [usize; 3] = [2_200_000, 4_400_000, 8_800_000];
    const WINDOW_SECONDS: usize = 60;
    const WORKLOAD_CONTRACT_ROUNDS: usize = 21;
    const WORKLOAD_CONTRACT_MIN_OF: usize = 2;
    const THREAD_ACTIVITY_REPETITIONS: usize = 5;
    const THREADS: &str = "4";
    const REQUIRED_BUILD_PROFILE: &str = "release-perf";

    assert_eq!(
        std::env::var("FNP_BENCH_PROFILE").as_deref(),
        Ok(REQUIRED_BUILD_PROFILE),
        "ship-grade rolling-load evidence requires FNP_BENCH_PROFILE=release-perf"
    );
    let build_worker =
        std::env::var("FNP_BUILD_WORKER").expect("FNP_BUILD_WORKER records the RCH build origin");
    assert!(
        !build_worker.trim().is_empty(),
        "FNP_BUILD_WORKER must name the RCH build worker"
    );
    for variable in [
        "RAYON_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(THREADS),
            "{variable} must be explicitly pinned before workload timing"
        );
    }
    assert_eq!(
        rayon::current_num_threads(),
        THREADS
            .parse::<usize>()
            .expect("thread count constant is numeric"),
        "Rayon pool width does not match the pinned workload configuration"
    );
    let host = std::env::var("HOSTNAME").unwrap_or_else(|_| "unavailable".to_owned());
    println!(
        "WORKLOAD_RUNTIME workload=rolling_load_saturation_report host={host} \
         build_worker={build_worker} build_profile={REQUIRED_BUILD_PROFILE} \
         rayon_threads={} RAYON_NUM_THREADS={} OPENBLAS_NUM_THREADS={} \
         OMP_NUM_THREADS={} MKL_NUM_THREADS={} trj_used=false",
        rayon::current_num_threads(),
        THREADS,
        THREADS,
        THREADS,
        THREADS,
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_rolling_load_saturation_workload")
            .expect("rolling-load saturation module");
        fnp_python(&module).expect("initialize fnp_python rolling-load saturation module");
        let numpy = py.import("numpy").expect("numpy incumbent");

        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(format!(
                r#"
import numpy as np
rng = np.random.default_rng(20260730)
sample_sizes = {SAMPLE_SIZES:?}
window_seconds = {WINDOW_SECONDS}
max_samples = sample_sizes[-1]

# A stable service baseline with periodic incident bursts. Each element is the
# observed request count for one second; the timed job starts from this already
# collected stream, as a normal capacity-analysis notebook would.
request_counts = rng.poisson(900, size=max_samples).astype(np.int64, copy=False)
for start in range(200_000, max_samples, 600_000):
    stop = min(start + 3_600, max_samples)
    request_counts[start:stop] += rng.poisson(
        2_200,
        size=stop - start,
    ).astype(np.int64, copy=False)

requests_by_size = [
    np.ascontiguousarray(request_counts[:sample_count])
    for sample_count in sample_sizes
]
rolling_window = np.ones(window_seconds, dtype=np.int64)

max_request_count = int(request_counts.max())
assert max_request_count >= 0
assert max_request_count * window_seconds < np.iinfo(np.int64).max
assert rolling_window.dtype == np.int64
assert rolling_window.ndim == 1
assert rolling_window.flags.c_contiguous
for sample_count, requests in zip(sample_sizes, requests_by_size):
    assert requests.dtype == np.int64
    assert requests.ndim == 1
    assert requests.flags.c_contiguous
    assert requests.size == sample_count
    assert requests.size - window_seconds + 1 >= (1 << 21)
    requests.flags.writeable = False
rolling_window.flags.writeable = False
"#
            ))
            .expect("rolling-load setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("rolling-load corpus setup");

        let fnp_convolve = module.getattr("convolve").expect("fnp convolve");
        let np_convolve = numpy.getattr("convolve").expect("numpy convolve");
        let fnp_maximum = module.getattr("maximum").expect("fnp maximum");
        let np_maximum = numpy.getattr("maximum").expect("numpy maximum");
        let fnp_maximum_accumulate = fnp_maximum
            .getattr("accumulate")
            .expect("fnp maximum.accumulate");
        let np_maximum_accumulate = np_maximum
            .getattr("accumulate")
            .expect("numpy maximum.accumulate");
        let fnp_ptp = module.getattr("ptp").expect("fnp ptp");
        let np_ptp = numpy.getattr("ptp").expect("numpy ptp");
        for (candidate, incumbent, surface) in [
            (&fnp_convolve, &np_convolve, "convolve"),
            (&fnp_maximum, &np_maximum, "maximum"),
            (&fnp_ptp, &np_ptp, "ptp"),
        ] {
            assert!(
                !candidate.is(incumbent),
                "{surface}: candidate callable aliases NumPy"
            );
            common::report_numpy_incumbent_identity(py, surface, incumbent);
        }
        let numpy_cpu_features = numpy
            .getattr("_core")
            .expect("numpy core")
            .getattr("_multiarray_umath")
            .expect("numpy multiarray umath")
            .getattr("__cpu_features__")
            .expect("numpy runtime CPU features")
            .str()
            .expect("numpy runtime CPU feature str")
            .extract::<String>()
            .expect("numpy runtime CPU feature string");
        println!(
            "NUMPY_RUNTIME_ISA workload=rolling_load_saturation_report \
             runtime_cpu_features={numpy_cpu_features}"
        );
        common::report_incumbent_topology(
            "fnp.workload.rolling_load_saturation_report",
            "numpy.workload.rolling_load_saturation_report",
        );
        println!(
            "INCUMBENT_ISOLATION_PROOF workload=rolling_load_saturation_report \
             candidate=fnp.convolve+fnp.maximum.accumulate+fnp.ptp \
             incumbent=numpy.convolve+numpy.maximum.accumulate+numpy.ptp \
             candidate_callable_identity_distinct=passed \
             shared_timed_component=none candidate_stages_all_native=true \
             inputs_shared_read_only=true numpy_live_calls_per_observation=3 \
             numpy_result_cache=none callable_handles_bound_once=true"
        );
        for (stage, route, pin) in [
            (
                "rolling_total_60s",
                "native_int64_parallel_direct_convolve_valid",
                "fnp-python/src/lib.rs:try_native_int_convolve(int64,1d,same-dtype,c-contiguous,mode=valid,n*m>=1<<16,threads>=2)",
            ),
            (
                "running_high_water",
                "native_int64_parallel_two_pass_maximum_accumulate",
                "fnp-python/src/lib.rs:try_zerocopy_accumulate_extremum(int64,1d,c-contiguous,axis=0,dtype/out=None,n>=1<<21,threads>=2)",
            ),
            (
                "rolling_spread",
                "native_int64_zerocopy_ptp",
                "fnp-python/src/lib.rs:try_zerocopy_int_ptp(int64,axis=None,nonempty-exact-ndarray)",
            ),
        ] {
            println!(
                "ROUTE_DISCLOSURE workload=rolling_load_saturation_report stage={stage} \
                 candidate_route={route} source_pin={pin}"
            );
        }
        println!(
            "ROUTE_PRECONDITIONS workload=rolling_load_saturation_report \
             dtype=int64 ndim=1 c_contiguous=true mode=valid window_seconds={WINDOW_SECONDS} \
             min_convolve_work={} int_convolve_min_work={} \
             min_rolling_elements={} accumulate_parallel_min={} \
             rayon_threads_min=2 overflow_headroom=proven \
             contract_rounds={WORKLOAD_CONTRACT_ROUNDS} \
             contract_min_of={WORKLOAD_CONTRACT_MIN_OF}",
            SAMPLE_SIZES[0] * WINDOW_SECONDS,
            1 << 16,
            SAMPLE_SIZES[0] - WINDOW_SECONDS + 1,
            1 << 21,
        );
        println!(
            "RUNTIME_ISA_BINDING workload=rolling_load_saturation_report \
             candidate_process_features=ISA_BASELINE_above \
             incumbent_dispatch_features=NUMPY_RUNTIME_ISA_above \
             same_process_same_host_all_rows=true"
        );

        let incumbent = RollingLoadSaturationArm {
            convolve: np_convolve,
            maximum_accumulate: np_maximum_accumulate,
            ptp: np_ptp,
        };
        let candidate = RollingLoadSaturationArm {
            convolve: fnp_convolve,
            maximum_accumulate: fnp_maximum_accumulate,
            ptp: fnp_ptp,
        };
        let requests_by_size = namespace
            .get_item("requests_by_size")
            .expect("request streams by size");
        let rolling_window = namespace
            .get_item("rolling_window")
            .expect("60-second rolling window");

        let mut scaling = Vec::with_capacity(SAMPLE_SIZES.len());
        for (size_index, sample_count) in SAMPLE_SIZES.into_iter().enumerate() {
            let rolling_elements = sample_count - WINDOW_SECONDS + 1;
            let convolve_terms = rolling_elements * WINDOW_SECONDS;
            let incumbent_accumulate_comparisons = rolling_elements - 1;
            let candidate_accumulate_comparisons = 2 * rolling_elements - 2;
            let ptp_comparisons = 2 * (rolling_elements - 1);
            let output_elements = 2 * rolling_elements + 1;
            let row =
                format!("workload_rolling_load_saturation_{sample_count}s_window{WINDOW_SECONDS}");
            println!(
                "WORKLOAD_CONFIG row={row} user_job=rolling_load_saturation_report \
                 host={host} build_worker={build_worker} \
                 invocation_id={} configured_threads={THREADS} \
                 samples={sample_count} cadence=one_second window_seconds={WINDOW_SECONDS} \
                 dtype=int64 distribution=poisson_service_baseline_plus_periodic_incident_bursts \
                 stages=convolve_valid_rolling_total,maximum_accumulate_high_water,ptp_spread \
                 output=rolling_totals_running_high_water_envelope_peak_to_peak_spread \
                 matched_config=same_process_same_inputs_same_mode_no_result_cache \
                 target_user=site_reliability_capacity_review",
                common::bench_invocation_id(),
            );
            let requests = requests_by_size
                .get_item(size_index)
                .expect("request stream for size");

            let (_, incumbent_output) = incumbent.run(&requests, &rolling_window);
            let (_, candidate_output) = candidate.run(&requests, &rolling_window);
            assert_workload_outputs_equal(&numpy, &row, &candidate_output, &incumbent_output);
            println!(
                "WORK_ACCOUNTING row={row} candidate_public_calls=3 incumbent_public_calls=3 \
                 candidate_convolve_terms={convolve_terms} \
                 incumbent_convolve_terms={convolve_terms} \
                 candidate_accumulate_comparisons={candidate_accumulate_comparisons} \
                 incumbent_accumulate_comparisons={incumbent_accumulate_comparisons} \
                 candidate_ptp_comparisons={ptp_comparisons} \
                 incumbent_ptp_comparisons={ptp_comparisons} \
                 candidate_output_elements={output_elements} \
                 incumbent_output_elements={output_elements} \
                 more_work_verdict=candidate_does_more_counted_accumulate_work \
                 extra_candidate_comparisons={incumbent_accumulate_comparisons}"
            );
            println!(
                "PARITY row={row} outputs=3 dtype_shape_byte_identity=passed \
                 no_overflow_precondition=passed checksum={:016x}",
                workload_checksum(&numpy, &candidate_output),
            );

            common::report_observed_thread_activity(
                &row,
                "numpy",
                THREAD_ACTIVITY_REPETITIONS,
                || {
                    let (_, outputs) = incumbent.run(&requests, &rolling_window);
                    black_box(outputs);
                },
            );
            common::report_observed_thread_activity(
                &row,
                "fnp",
                THREAD_ACTIVITY_REPETITIONS,
                || {
                    let (_, outputs) = candidate.run(&requests, &rolling_window);
                    black_box(outputs);
                },
            );

            let incumbent_profile = incumbent.profile(&requests, &rolling_window);
            let candidate_profile = candidate.profile(&requests, &rolling_window);
            let candidate_total = candidate_profile.iter().sum::<f64>();
            let (target_stage, target_ms) = [
                ("rolling_total_60s", candidate_profile[0]),
                ("running_high_water", candidate_profile[1]),
                ("rolling_spread", candidate_profile[2]),
            ]
            .into_iter()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .expect("three rolling-load profile stages");
            for (stage, incumbent_ms, candidate_ms) in [
                (
                    "rolling_total_60s",
                    incumbent_profile[0],
                    candidate_profile[0],
                ),
                (
                    "running_high_water",
                    incumbent_profile[1],
                    candidate_profile[1],
                ),
                ("rolling_spread", incumbent_profile[2], candidate_profile[2]),
            ] {
                println!(
                    "PROFILE_STAGE row={row} arm_pair=numpy_fnp stage={stage} \
                     incumbent_median_ms={incumbent_ms:.6} \
                     candidate_median_ms={candidate_ms:.6} \
                     stage_ratio_numpy_over_fnp={:.6}",
                    incumbent_ms / candidate_ms,
                );
            }
            let target_fraction = target_ms / candidate_total;
            println!(
                "PROFILE_SUMMARY row={row} target_stage={target_stage} \
                 target_candidate_ms={target_ms:.6} candidate_stage_sum_ms={candidate_total:.6} \
                 target_self_fraction_pct={:.3} amdahl_remove_ceiling={:.6} \
                 shared_timed_component=none",
                target_fraction * 100.0,
                1.0 / (1.0 - target_fraction),
            );

            let mut observe_incumbent = || {
                let (elapsed, outputs) = incumbent.run(&requests, &rolling_window);
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let mut observe_candidate = || {
                let (elapsed, outputs) = candidate.run(&requests, &rolling_window);
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract_with_sampling(
                    &row,
                    &mut observe_incumbent,
                    &mut observe_candidate,
                    WORKLOAD_CONTRACT_ROUNDS,
                    WORKLOAD_CONTRACT_MIN_OF,
                );
            let verdict =
                common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
            println!(
                "WORKLOAD_SIZE_POINT workload=rolling_load_saturation_report row={row} \
                 host={host} build_worker={build_worker} \
                 invocation_id={} configured_threads={THREADS} \
                 samples={sample_count} rolling_elements={rolling_elements} \
                 incumbent_median_ms={:.6} candidate_median_ms={:.6} \
                 ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
                 incumbent_null_ratio={:.6} candidate_null_ratio={:.6} \
                 verdict={verdict} gate=effect_bootstrap_ci_plus_2x_controlling_null_margin \
                 cv_gate=false incumbent_ns_per_input_sample={:.6} \
                 candidate_ns_per_input_sample={:.6}",
                common::bench_invocation_id(),
                effect.arm_a_median_ns / 1_000_000.0,
                effect.arm_b_median_ns / 1_000_000.0,
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                incumbent_null.ratio_median,
                candidate_null.ratio_median,
                effect.arm_a_median_ns / sample_count as f64,
                effect.arm_b_median_ns / sample_count as f64,
            );
            scaling.push((effect, incumbent_null, candidate_null));
        }

        let first_ratio = scaling.first().expect("first scaling point").0.ratio_median;
        let last_ratio = scaling.last().expect("last scaling point").0.ratio_median;
        let ratio_min = scaling
            .iter()
            .map(|(effect, _, _)| effect.ratio_median)
            .fold(f64::INFINITY, f64::min);
        let ratio_max = scaling
            .iter()
            .map(|(effect, _, _)| effect.ratio_median)
            .fold(f64::NEG_INFINITY, f64::max);
        let ratio_spread = ratio_max / ratio_min - 1.0;
        let monotonic_up = scaling
            .windows(2)
            .all(|points| points[1].0.ratio_median >= points[0].0.ratio_median);
        let monotonic_down = scaling
            .windows(2)
            .all(|points| points[1].0.ratio_median <= points[0].0.ratio_median);
        let shape = if ratio_spread <= 0.15 {
            "FLAT_PER_SAMPLE_COST"
        } else if monotonic_up && last_ratio >= first_ratio * 1.15 {
            "WIDENING_WITH_SAMPLES"
        } else if monotonic_down && last_ratio <= first_ratio * 0.85 {
            "NARROWING_WITH_SAMPLES"
        } else {
            "MIXED_OR_NOISE"
        };
        println!(
            "SCALING_SHAPE workload=rolling_load_saturation_report \
             dimension=sample_count window_seconds={WINDOW_SECONDS} \
             sizes=[{},{},{}] ratios=[{:.6},{:.6},{:.6}] \
             ratio_spread={ratio_spread:.6} classification={shape}",
            SAMPLE_SIZES[0],
            SAMPLE_SIZES[1],
            SAMPLE_SIZES[2],
            scaling[0].0.ratio_median,
            scaling[1].0.ratio_median,
            scaling[2].0.ratio_median,
        );
        let all_decidable_wins = scaling
            .iter()
            .all(|(effect, incumbent_null, candidate_null)| {
                common::dual_null_contract_verdict(*effect, *incumbent_null, *candidate_null)
                    == "DECIDABLE_WIN"
            });
        let any_decidable_regression =
            scaling
                .iter()
                .any(|(effect, incumbent_null, candidate_null)| {
                    common::dual_null_contract_verdict(*effect, *incumbent_null, *candidate_null)
                        == "DECIDABLE_REGRESSION"
                });
        let (decision, reason) = if all_decidable_wins {
            (
                "choose_fnp",
                "all_three_effect_CIs_and_2x_null_margins_clear",
            )
        } else if any_decidable_regression {
            (
                "choose_numpy",
                "at_least_one_size_is_a_decidable_regression",
            )
        } else {
            (
                "choose_numpy_pending_remeasure",
                "at_least_one_size_is_statistically_undecided",
            )
        };
        println!(
            "CHOOSER_SCOPE workload=rolling_load_saturation_report \
             target_user=site_reliability_capacity_review dtype=int64 \
             sample_range={}..={} cadence=one_second window_seconds={WINDOW_SECONDS} \
             candidate_profile={REQUIRED_BUILD_PROFILE} build_worker={build_worker} \
             measurement_host={host} configured_threads={THREADS} \
             thread_activity_recorded_per_row=true \
             decision_requires_all_three_effects_and_both_nulls \
             generalization_beyond_measured_shape=false",
            SAMPLE_SIZES[0], SAMPLE_SIZES[2],
        );
        println!(
            "CHOOSER_STATEMENT workload=rolling_load_saturation_report decision={decision} \
             reason={reason} measured_scope=int64_one_second_request_streams_60s_valid_window \
             incumbent=numpy_live_same_invocation_no_result_cache \
             outside_scope=run_same_contract_before_choosing"
        );
    });
}

/// Phase-2 Class-3 workload: turn a raw fortnight of clickstream events into the
/// session-ordered event table an analyst cuts sessions from. This is the
/// **permutation** class of whole job — order by a compound key, carry the
/// payload columns through that order, then difference adjacent rows — which no
/// other banked realistic row exercises (the f16 dynamic-range job is
/// elementwise numeric, the access-control jobs are linear-algebra, the
/// rolling-load job is a 1-D scan, and the entitlement job is set logic).
///
/// All six public candidate stages stay on independently implemented native
/// int64 paths at every size. The corpus deliberately pins the packed composite
/// span into `(1<<24, u64::MAX]` so the lexsort takes the parallel
/// packed-composite pair sort — not the small-range counting sort and not
/// NumPy's radix fallback — identically at all three sizes.
fn bench_realistic_clickstream_sessionization_vs_numpy_median_gate(c: &mut Criterion) {
    let _ = c;

    const EVENT_COUNTS: [usize; 3] = [2_400_000, 4_800_000, 9_600_000];
    const USER_COUNT: usize = 250_000;
    const HORIZON_SECONDS: usize = 14 * 86_400;
    const POWER_USER_COUNT: usize = 4_096;
    const WORKLOAD_CONTRACT_ROUNDS: usize = 21;
    const WORKLOAD_CONTRACT_MIN_OF: usize = 2;
    const THREAD_ACTIVITY_REPETITIONS: usize = 5;
    const THREADS: &str = "4";
    const REQUIRED_BUILD_PROFILE: &str = "release-perf";

    verify_process_resource_snapshot_parser();

    assert_eq!(
        std::env::var("FNP_BENCH_PROFILE").as_deref(),
        Ok(REQUIRED_BUILD_PROFILE),
        "ship-grade sessionization evidence requires FNP_BENCH_PROFILE=release-perf"
    );
    let build_worker =
        std::env::var("FNP_BUILD_WORKER").expect("FNP_BUILD_WORKER records the RCH build origin");
    assert!(
        !build_worker.trim().is_empty(),
        "FNP_BUILD_WORKER must name the RCH build worker"
    );
    for variable in [
        "RAYON_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(THREADS),
            "{variable} must be explicitly pinned before workload timing"
        );
    }
    assert_eq!(
        rayon::current_num_threads(),
        THREADS
            .parse::<usize>()
            .expect("thread count constant is numeric"),
        "Rayon pool width does not match the pinned workload configuration"
    );
    let host = std::env::var("HOSTNAME").unwrap_or_else(|_| "unavailable".to_owned());
    println!(
        "WORKLOAD_RUNTIME workload=clickstream_sessionization_report host={host} \
         build_worker={build_worker} build_profile={REQUIRED_BUILD_PROFILE} \
         rayon_threads={} RAYON_NUM_THREADS={} OPENBLAS_NUM_THREADS={} \
         OMP_NUM_THREADS={} MKL_NUM_THREADS={} trj_used=false",
        rayon::current_num_threads(),
        THREADS,
        THREADS,
        THREADS,
        THREADS,
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_clickstream_sessionization_workload")
            .expect("clickstream sessionization module");
        fnp_python(&module).expect("initialize fnp_python clickstream sessionization module");
        let numpy = py.import("numpy").expect("numpy incumbent");

        let namespace = PyDict::new(py);
        py.run(
            std::ffi::CString::new(format!(
                r#"
import numpy as np
rng = np.random.default_rng(20260731)
event_counts = {EVENT_COUNTS:?}
user_count = {USER_COUNT}
horizon_seconds = {HORIZON_SECONDS}
power_user_count = {POWER_USER_COUNT}
max_events = event_counts[-1]

# A fortnight of one-second-resolution product clickstream. Traffic is
# heavy-tailed across users the way real telemetry is: a small set of power
# users emits a large share of the events while the long tail emits a handful
# each. The timed job starts from this already collected, arrival-ordered
# stream, exactly as a sessionization notebook would.
user_ids = rng.integers(0, user_count, size=max_events, dtype=np.int64)
power_user_events = max_events // 5
user_ids[:power_user_events] = rng.integers(
    0,
    power_user_count,
    size=power_user_events,
    dtype=np.int64,
)
rng.shuffle(user_ids)
event_times = rng.integers(0, horizon_seconds, size=max_events, dtype=np.int64)

# Pin the per-key spans so every size point takes the identical source-pinned
# route: positions 0 and 1 carry both extremes of both keys, so every prefix
# slice has exactly the same span and the same packed composite range.
user_ids[0] = 0
user_ids[1] = user_count - 1
event_times[0] = 0
event_times[1] = horizon_seconds - 1

events_by_size = []
corpus_structure = []
for event_count in event_counts:
    users = np.ascontiguousarray(user_ids[:event_count])
    times = np.ascontiguousarray(event_times[:event_count])
    assert users.dtype == np.int64
    assert times.dtype == np.int64
    assert users.ndim == 1 and times.ndim == 1
    assert users.flags.c_contiguous and times.flags.c_contiguous
    assert users.size == event_count and times.size == event_count
    user_span = int(users.max()) - int(users.min()) + 1
    time_span = int(times.max()) - int(times.min()) + 1
    assert user_span == user_count
    assert time_span == horizon_seconds
    composite_range = user_span * time_span
    # Above the packed counting-sort ceiling and inside the u64 packing ceiling:
    # together these pin the parallel composite pair-sort route at every size.
    assert composite_range > (1 << 24)
    assert composite_range <= (1 << 64) - 1
    # Both the gather and the first-difference parallel gates need the output to
    # clear 1<<21 elements.
    assert event_count >= (1 << 21)
    assert event_count - 1 >= (1 << 21)
    # How many events share an EXACT (user, time) key. These are the rows whose
    # order is decided purely by the stable tie-break, so a non-zero count is what
    # makes the byte-identity assertion below a real test of that contract rather
    # than a vacuous one.
    packed = (times - times.min()) + (users - users.min()) * time_span
    _, key_counts = np.unique(packed, return_counts=True)
    exact_key_collisions = int((key_counts - 1).sum())
    distinct_users_present = int(np.unique(users).size)
    corpus_structure.append((exact_key_collisions, distinct_users_present))
    users.flags.writeable = False
    times.flags.writeable = False
    # NumPy's LAST lexsort key is primary: this orders by user, then by time.
    events_by_size.append((users, times, (times, users)))
"#
            ))
            .expect("clickstream sessionization setup CString")
            .as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .expect("clickstream sessionization corpus setup");

        let fnp_lexsort = module.getattr("lexsort").expect("fnp lexsort");
        let np_lexsort = numpy.getattr("lexsort").expect("numpy lexsort");
        let fnp_take = module.getattr("take").expect("fnp take");
        let np_take = numpy.getattr("take").expect("numpy take");
        let fnp_diff = module.getattr("diff").expect("fnp diff");
        let np_diff = numpy.getattr("diff").expect("numpy diff");
        let fnp_count_nonzero = module.getattr("count_nonzero").expect("fnp count_nonzero");
        let np_count_nonzero = numpy.getattr("count_nonzero").expect("numpy count_nonzero");
        for (candidate, incumbent, surface) in [
            (&fnp_lexsort, &np_lexsort, "lexsort"),
            (&fnp_take, &np_take, "take"),
            (&fnp_diff, &np_diff, "diff"),
            (&fnp_count_nonzero, &np_count_nonzero, "count_nonzero"),
        ] {
            assert!(
                !candidate.is(incumbent),
                "{surface}: candidate callable aliases NumPy"
            );
            common::report_numpy_incumbent_identity(py, surface, incumbent);
        }
        let numpy_cpu_features = numpy
            .getattr("_core")
            .expect("numpy core")
            .getattr("_multiarray_umath")
            .expect("numpy multiarray umath")
            .getattr("__cpu_features__")
            .expect("numpy runtime CPU features")
            .str()
            .expect("numpy runtime CPU feature str")
            .extract::<String>()
            .expect("numpy runtime CPU feature string");
        println!(
            "NUMPY_RUNTIME_ISA workload=clickstream_sessionization_report \
             runtime_cpu_features={numpy_cpu_features}"
        );
        common::report_incumbent_topology(
            "fnp.workload.clickstream_sessionization_report",
            "numpy.workload.clickstream_sessionization_report",
        );
        println!(
            "INCUMBENT_ISOLATION_PROOF workload=clickstream_sessionization_report \
             candidate=fnp.lexsort+fnp.take+fnp.take+fnp.diff+fnp.diff+fnp.count_nonzero \
             incumbent=numpy.lexsort+numpy.take+numpy.take+numpy.diff+numpy.diff+numpy.count_nonzero \
             candidate_callable_identity_distinct=passed \
             shared_timed_component=none candidate_stages_all_native=true \
             inputs_shared_read_only=true numpy_live_calls_per_observation=6 \
             numpy_result_cache=none callable_handles_bound_once=true \
             sort_key_tuple_built_once_outside_timer=true"
        );
        for (stage, route, pin) in [
            (
                "session_order",
                "native_int64_parallel_packed_composite_lexsort",
                "fnp-python/src/lib.rs:lexsort->try_native_lexsort_composite(tuple-of-2 1-D int64 keys,axis=-1,n>=1<<18,threads>=2,packed-span>1<<24,packed-span<=u64::MAX)",
            ),
            (
                "ordered_users",
                "native_int64_parallel_zerocopy_byte_gather",
                "fnp-python/src/lib.rs:take->try_zerocopy_int_take(int64 source,int64 indices,axis=None,mode=raise,n>=1<<21,threads>=2)",
            ),
            (
                "ordered_times",
                "native_int64_parallel_zerocopy_byte_gather",
                "fnp-python/src/lib.rs:take->try_zerocopy_int_take(int64 source,int64 indices,axis=None,mode=raise,n>=1<<21,threads>=2)",
            ),
            (
                "user_boundary",
                "native_int64_parallel_first_difference",
                "fnp-python/src/lib.rs:diff->try_zerocopy_int_diff(int64,1-D,n=1,axis=-1)->diff_typed(out>=1<<21,threads>=2)",
            ),
            (
                "inter_event_gap",
                "native_int64_parallel_first_difference",
                "fnp-python/src/lib.rs:diff->try_zerocopy_int_diff(int64,1-D,n=1,axis=-1)->diff_typed(out>=1<<21,threads>=2)",
            ),
            (
                "user_transitions",
                "native_int64_zerocopy_count_nonzero_SERIAL",
                "fnp-python/src/lib.rs:count_nonzero->try_zerocopy_count_nonzero(int64 exact ndarray,axis=None,keepdims=false)->count_nonzero_typed(serial scalar loop)",
            ),
        ] {
            println!(
                "ROUTE_DISCLOSURE workload=clickstream_sessionization_report stage={stage} \
                 candidate_route={route} source_pin={pin}"
            );
        }
        // Stated up front rather than discovered in the profile: the sixth stage
        // has no parallel arm. It is native, but it is not expected to win, and
        // the whole-job ratio is reported with that stage inside the timer.
        println!(
            "ROUTE_CAVEAT workload=clickstream_sessionization_report \
             stage=user_transitions candidate_arm=serial_scalar_loop \
             incumbent_arm=numpy_simd_count expected_stage_direction=parity_or_loss \
             counted_in_candidate_total=true"
        );
        println!(
            "ROUTE_PRECONDITIONS workload=clickstream_sessionization_report \
             dtype=int64 ndim=1 c_contiguous=true keys=2 lexsort_axis=-1 \
             primary_key=user_id secondary_key=event_time_seconds \
             user_span={USER_COUNT} time_span={HORIZON_SECONDS} \
             packed_composite_range={} lexsort_counting_sort_ceiling={} \
             min_lexsort_elements={} lexsort_composite_min={} \
             min_gather_elements={} take_parallel_min={} \
             min_difference_outputs={} diff_parallel_min={} \
             rayon_threads_min=2 index_dtype=intp_int64 \
             candidate_auxiliary_buffers=composite_u64+pair_u64_u32 \
             contract_rounds={WORKLOAD_CONTRACT_ROUNDS} \
             contract_min_of={WORKLOAD_CONTRACT_MIN_OF}",
            USER_COUNT * HORIZON_SECONDS,
            1u64 << 24,
            EVENT_COUNTS[0],
            1 << 18,
            EVENT_COUNTS[0],
            1 << 21,
            EVENT_COUNTS[0] - 1,
            1 << 21,
        );
        println!(
            "RUNTIME_ISA_BINDING workload=clickstream_sessionization_report \
             candidate_process_features=ISA_BASELINE_above \
             incumbent_dispatch_features=NUMPY_RUNTIME_ISA_above \
             same_process_same_host_all_rows=true"
        );

        let incumbent = ClickstreamSessionizationArm {
            lexsort: np_lexsort,
            take: np_take,
            diff: np_diff,
            count_nonzero: np_count_nonzero,
        };
        let candidate = ClickstreamSessionizationArm {
            lexsort: fnp_lexsort,
            take: fnp_take,
            diff: fnp_diff,
            count_nonzero: fnp_count_nonzero,
        };
        let events_by_size = namespace
            .get_item("events_by_size")
            .expect("clickstream event columns by size");
        let corpus_structure = namespace
            .get_item("corpus_structure")
            .expect("clickstream corpus tie structure by size");

        let mut scaling = Vec::with_capacity(EVENT_COUNTS.len());
        for (size_index, event_count) in EVENT_COUNTS.into_iter().enumerate() {
            let permutation_elements = event_count;
            let gathered_elements = 2 * event_count;
            let differenced_elements = 2 * (event_count - 1);
            let counted_elements = event_count - 1;
            let output_elements = 3 * event_count + 2 * (event_count - 1) + 1;
            let candidate_auxiliary_elements = 2 * event_count;
            let row = format!("workload_clickstream_sessionization_{event_count}events");
            println!(
                "WORKLOAD_CONFIG row={row} user_job=clickstream_sessionization_report \
                 host={host} build_worker={build_worker} \
                 invocation_id={} configured_threads={THREADS} \
                 events={event_count} cadence=one_second horizon_days=14 \
                 users={USER_COUNT} power_users={POWER_USER_COUNT} dtype=int64 \
                 distribution=heavy_tailed_power_user_clickstream \
                 stages=lexsort_session_order,take_user_column,take_time_column,\
diff_user_boundary,diff_inter_event_gap,count_nonzero_user_transitions \
                 output=session_ordered_event_table_boundary_column_gap_column_transition_count \
                 matched_config=same_process_same_inputs_same_keys_no_result_cache \
                 target_user=product_analytics_sessionization",
                common::bench_invocation_id(),
            );
            let size_entry = events_by_size
                .get_item(size_index)
                .expect("event columns for size");
            let user_ids = size_entry.get_item(0).expect("user id column");
            let event_times = size_entry.get_item(1).expect("event time column");
            let sort_keys = size_entry.get_item(2).expect("lexsort key tuple");

            let (_, incumbent_output) = incumbent.run(&sort_keys, &user_ids, &event_times);
            let (_, candidate_output) = candidate.run(&sort_keys, &user_ids, &event_times);
            assert_workload_outputs_equal(&numpy, &row, &candidate_output, &incumbent_output);
            println!(
                "WORK_ACCOUNTING row={row} candidate_public_calls=6 incumbent_public_calls=6 \
                 candidate_permutation_elements={permutation_elements} \
                 incumbent_permutation_elements={permutation_elements} \
                 candidate_gathered_elements={gathered_elements} \
                 incumbent_gathered_elements={gathered_elements} \
                 candidate_differenced_elements={differenced_elements} \
                 incumbent_differenced_elements={differenced_elements} \
                 candidate_counted_elements={counted_elements} \
                 incumbent_counted_elements={counted_elements} \
                 candidate_output_elements={output_elements} \
                 incumbent_output_elements={output_elements} \
                 candidate_auxiliary_elements={candidate_auxiliary_elements} \
                 incumbent_auxiliary_elements=0 \
                 more_work_verdict=candidate_materializes_two_extra_full_length_buffers \
                 extra_candidate_elements={candidate_auxiliary_elements}"
            );
            let structure = corpus_structure
                .get_item(size_index)
                .expect("corpus tie structure for size");
            let exact_key_collisions = structure
                .get_item(0)
                .expect("exact key collision count")
                .extract::<usize>()
                .expect("collision count is an integer");
            let distinct_users_present = structure
                .get_item(1)
                .expect("distinct user count")
                .extract::<usize>()
                .expect("distinct user count is an integer");
            println!(
                "CORPUS_TIE_STRUCTURE row={row} exact_user_time_key_collisions={exact_key_collisions} \
                 distinct_users_present={distinct_users_present} \
                 tie_break_contract_exercised={}",
                exact_key_collisions > 0,
            );
            println!(
                "PARITY row={row} outputs=6 dtype_shape_byte_identity=passed \
                 stable_tie_order_matches_numpy_lexsort=passed checksum={:016x}",
                workload_checksum(&numpy, &candidate_output),
            );
            let incumbent_ordered_times = incumbent_output[2].clone();
            let candidate_ordered_times = candidate_output[2].clone();
            drop(incumbent_output);
            drop(candidate_output);

            // This is the exact `diff(ordered_times)` result path from the
            // whole job above, not a synthetic allocation microbenchmark.  Its
            // input is produced independently by the candidate and incumbent
            // arms and was byte-compared just above.  Keep both arms live in
            // this invocation, including their A/A controls, so the fault/RSS
            // evidence remains tied to the same timing contract.
            let gap_row = format!("{row}_inter_event_gap_result_buffer");
            let incumbent_gap_lifecycles = RefCell::new(Vec::new());
            let candidate_gap_lifecycles = RefCell::new(Vec::new());
            let mut observe_incumbent_gap = || {
                observe_inter_event_gap_result_lifecycle(
                    &numpy,
                    &incumbent.diff,
                    &incumbent_ordered_times,
                    &incumbent_gap_lifecycles,
                )
            };
            let mut observe_candidate_gap = || {
                observe_inter_event_gap_result_lifecycle(
                    &numpy,
                    &candidate.diff,
                    &candidate_ordered_times,
                    &candidate_gap_lifecycles,
                )
            };
            let (gap_effect, gap_incumbent_null, gap_candidate_null) =
                common::run_dual_null_median_ci_contract_with_sampling(
                    &gap_row,
                    &mut observe_incumbent_gap,
                    &mut observe_candidate_gap,
                    WORKLOAD_CONTRACT_ROUNDS,
                    WORKLOAD_CONTRACT_MIN_OF,
                );
            let gap_verdict = common::dual_null_contract_verdict(
                gap_effect,
                gap_incumbent_null,
                gap_candidate_null,
            );
            let incumbent_gap_lifecycle =
                median_result_buffer_lifecycle(&incumbent_gap_lifecycles.borrow());
            let candidate_gap_lifecycle =
                median_result_buffer_lifecycle(&candidate_gap_lifecycles.borrow());
            println!(
                "RESULT_BUFFER_LIFECYCLE row={gap_row} stage=inter_event_gap \
                 source=linux_proc_self_stat_and_status allocator=rust_default_system \
                 input=independently_materialized_ordered_times byte_parity=passed \
                 incumbent_live_same_invocation=true aa_nulls=true timing_ratio_median={:.6} \
                 timing_ratio_ci95=[{:.6},{:.6}] timing_verdict={gap_verdict} \
                 incumbent_samples={} candidate_samples={} \
                 incumbent_minor_faults_median={} candidate_minor_faults_median={} \
                 incumbent_major_faults_median={} candidate_major_faults_median={} \
                 incumbent_rss_while_live_kib_median={} candidate_rss_while_live_kib_median={} \
                 incumbent_rss_after_release_kib_median={} candidate_rss_after_release_kib_median={}",
                gap_effect.ratio_median,
                gap_effect.ratio_ci_low,
                gap_effect.ratio_ci_high,
                incumbent_gap_lifecycles.borrow().len(),
                candidate_gap_lifecycles.borrow().len(),
                incumbent_gap_lifecycle.minor_faults,
                candidate_gap_lifecycle.minor_faults,
                incumbent_gap_lifecycle.major_faults,
                candidate_gap_lifecycle.major_faults,
                incumbent_gap_lifecycle.rss_while_live_kib,
                candidate_gap_lifecycle.rss_while_live_kib,
                incumbent_gap_lifecycle.rss_after_release_kib,
                candidate_gap_lifecycle.rss_after_release_kib,
            );

            common::report_observed_thread_activity(
                &row,
                "numpy",
                THREAD_ACTIVITY_REPETITIONS,
                || {
                    let (_, outputs) = incumbent.run(&sort_keys, &user_ids, &event_times);
                    black_box(outputs);
                },
            );
            common::report_observed_thread_activity(
                &row,
                "fnp",
                THREAD_ACTIVITY_REPETITIONS,
                || {
                    let (_, outputs) = candidate.run(&sort_keys, &user_ids, &event_times);
                    black_box(outputs);
                },
            );

            let incumbent_profile = incumbent.profile(&sort_keys, &user_ids, &event_times);
            let candidate_profile = candidate.profile(&sort_keys, &user_ids, &event_times);
            let candidate_total = candidate_profile.iter().sum::<f64>();
            const STAGE_NAMES: [&str; 6] = [
                "session_order",
                "ordered_users",
                "ordered_times",
                "user_boundary",
                "inter_event_gap",
                "user_transitions",
            ];
            let (target_stage, target_ms) = STAGE_NAMES
                .into_iter()
                .zip(candidate_profile)
                .max_by(|left, right| left.1.total_cmp(&right.1))
                .expect("six sessionization profile stages");
            for ((stage, incumbent_ms), candidate_ms) in STAGE_NAMES
                .into_iter()
                .zip(incumbent_profile)
                .zip(candidate_profile)
            {
                println!(
                    "PROFILE_STAGE row={row} arm_pair=numpy_fnp stage={stage} \
                     incumbent_median_ms={incumbent_ms:.6} \
                     candidate_median_ms={candidate_ms:.6} \
                     stage_ratio_numpy_over_fnp={:.6}",
                    incumbent_ms / candidate_ms,
                );
            }
            let target_fraction = target_ms / candidate_total;
            println!(
                "PROFILE_SUMMARY row={row} target_stage={target_stage} \
                 target_candidate_ms={target_ms:.6} candidate_stage_sum_ms={candidate_total:.6} \
                 target_self_fraction_pct={:.3} amdahl_remove_ceiling={:.6} \
                 shared_timed_component=none",
                target_fraction * 100.0,
                1.0 / (1.0 - target_fraction),
            );

            let mut observe_incumbent = || {
                let (elapsed, outputs) = incumbent.run(&sort_keys, &user_ids, &event_times);
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let mut observe_candidate = || {
                let (elapsed, outputs) = candidate.run(&sort_keys, &user_ids, &event_times);
                common::ContractObservation {
                    elapsed,
                    checksum: workload_checksum(&numpy, &outputs),
                }
            };
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract_with_sampling(
                    &row,
                    &mut observe_incumbent,
                    &mut observe_candidate,
                    WORKLOAD_CONTRACT_ROUNDS,
                    WORKLOAD_CONTRACT_MIN_OF,
                );
            let verdict =
                common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
            println!(
                "WORKLOAD_SIZE_POINT workload=clickstream_sessionization_report row={row} \
                 host={host} build_worker={build_worker} \
                 invocation_id={} configured_threads={THREADS} \
                 events={event_count} users={USER_COUNT} \
                 incumbent_median_ms={:.6} candidate_median_ms={:.6} \
                 ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
                 incumbent_null_ratio={:.6} candidate_null_ratio={:.6} \
                 verdict={verdict} gate=effect_bootstrap_ci_plus_2x_controlling_null_margin \
                 cv_gate=false incumbent_ns_per_event={:.6} candidate_ns_per_event={:.6}",
                common::bench_invocation_id(),
                effect.arm_a_median_ns / 1_000_000.0,
                effect.arm_b_median_ns / 1_000_000.0,
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                incumbent_null.ratio_median,
                candidate_null.ratio_median,
                effect.arm_a_median_ns / event_count as f64,
                effect.arm_b_median_ns / event_count as f64,
            );
            scaling.push((effect, incumbent_null, candidate_null));
        }

        let first_ratio = scaling.first().expect("first scaling point").0.ratio_median;
        let last_ratio = scaling.last().expect("last scaling point").0.ratio_median;
        let ratio_min = scaling
            .iter()
            .map(|(effect, _, _)| effect.ratio_median)
            .fold(f64::INFINITY, f64::min);
        let ratio_max = scaling
            .iter()
            .map(|(effect, _, _)| effect.ratio_median)
            .fold(f64::NEG_INFINITY, f64::max);
        let ratio_spread = ratio_max / ratio_min - 1.0;
        let monotonic_up = scaling
            .windows(2)
            .all(|points| points[1].0.ratio_median >= points[0].0.ratio_median);
        let monotonic_down = scaling
            .windows(2)
            .all(|points| points[1].0.ratio_median <= points[0].0.ratio_median);
        let shape = if ratio_spread <= 0.15 {
            "FLAT_PER_EVENT_COST"
        } else if monotonic_up && last_ratio >= first_ratio * 1.15 {
            "WIDENING_WITH_EVENTS"
        } else if monotonic_down && last_ratio <= first_ratio * 0.85 {
            "NARROWING_WITH_EVENTS"
        } else {
            "MIXED_OR_NOISE"
        };
        println!(
            "SCALING_SHAPE workload=clickstream_sessionization_report \
             dimension=event_count users={USER_COUNT} horizon_seconds={HORIZON_SECONDS} \
             sizes=[{},{},{}] ratios=[{:.6},{:.6},{:.6}] \
             ratio_spread={ratio_spread:.6} classification={shape}",
            EVENT_COUNTS[0],
            EVENT_COUNTS[1],
            EVENT_COUNTS[2],
            scaling[0].0.ratio_median,
            scaling[1].0.ratio_median,
            scaling[2].0.ratio_median,
        );
        let all_decidable_wins = scaling
            .iter()
            .all(|(effect, incumbent_null, candidate_null)| {
                common::dual_null_contract_verdict(*effect, *incumbent_null, *candidate_null)
                    == "DECIDABLE_WIN"
            });
        let any_decidable_regression =
            scaling
                .iter()
                .any(|(effect, incumbent_null, candidate_null)| {
                    common::dual_null_contract_verdict(*effect, *incumbent_null, *candidate_null)
                        == "DECIDABLE_REGRESSION"
                });
        let (decision, reason) = if all_decidable_wins {
            (
                "choose_fnp",
                "all_three_effect_CIs_and_2x_null_margins_clear",
            )
        } else if any_decidable_regression {
            (
                "choose_numpy",
                "at_least_one_size_is_a_decidable_regression",
            )
        } else {
            (
                "choose_numpy_pending_remeasure",
                "at_least_one_size_is_statistically_undecided",
            )
        };
        println!(
            "CHOOSER_SCOPE workload=clickstream_sessionization_report \
             target_user=product_analytics_sessionization dtype=int64 \
             event_range={}..={} users={USER_COUNT} horizon_seconds={HORIZON_SECONDS} \
             keys=2 primary=user_id secondary=event_time_seconds \
             candidate_profile={REQUIRED_BUILD_PROFILE} build_worker={build_worker} \
             measurement_host={host} configured_threads={THREADS} \
             thread_activity_recorded_per_row=true \
             decision_requires_all_three_effects_and_both_nulls \
             generalization_beyond_measured_shape=false",
            EVENT_COUNTS[0], EVENT_COUNTS[2],
        );
        println!(
            "CHOOSER_STATEMENT workload=clickstream_sessionization_report decision={decision} \
             reason={reason} \
             measured_scope=int64_two_key_clickstream_sessionization_2.4M_to_9.6M_events \
             incumbent=numpy_live_same_invocation_no_result_cache \
             outside_scope=run_same_contract_before_choosing"
        );
    });
}

#[derive(Clone, Copy)]
enum GeneratorIncumbentDistribution {
    Zipf,
    NoncentralChisquare,
}

#[derive(Clone, Copy)]
struct GeneratorIncumbentSpec {
    row: &'static str,
    method_name: &'static str,
    seed: u64,
    distribution: GeneratorIncumbentDistribution,
}

fn seeded_pcg64_generator<'py>(random: &Bound<'py, PyAny>, seed: u64) -> Bound<'py, PyAny> {
    let bit_generator = random
        .getattr("PCG64")
        .expect("random.PCG64")
        .call1((seed,))
        .expect("construct seeded PCG64");
    random
        .getattr("Generator")
        .expect("random.Generator")
        .call1((bit_generator,))
        .expect("construct Generator(PCG64)")
}

fn call_generator_incumbent_distribution<'py>(
    method: &Bound<'py, PyAny>,
    spec: GeneratorIncumbentSpec,
) -> Bound<'py, PyAny> {
    const SIZE: usize = 100_000;
    match spec.distribution {
        GeneratorIncumbentDistribution::Zipf => method
            .call1((black_box(2.5_f64), black_box(SIZE)))
            .expect("Generator.zipf"),
        GeneratorIncumbentDistribution::NoncentralChisquare => method
            .call1((black_box(5.0_f64), black_box(1.0_f64), black_box(SIZE)))
            .expect("Generator.noncentral_chisquare"),
    }
}

fn observe_generator_incumbent_distribution(
    numpy: &Bound<'_, PyModule>,
    random: &Bound<'_, PyAny>,
    spec: GeneratorIncumbentSpec,
) -> common::ContractObservation {
    // A real user keeps a Generator and times the distribution draw, not its
    // one-time constructor. Recreate the fixed seed outside each timed interval
    // so every null/effect observation sees the identical stream and output.
    let generator = seeded_pcg64_generator(random, spec.seed);
    let method = generator
        .getattr(spec.method_name)
        .expect("bind Generator distribution method");
    let started = Instant::now();
    let output = call_generator_incumbent_distribution(&method, spec);
    let elapsed = started.elapsed();
    common::ContractObservation {
        elapsed,
        checksum: workload_checksum(numpy, &[output]),
    }
}

fn prove_generator_incumbent_distribution_stream(
    numpy: &Bound<'_, PyModule>,
    candidate_random: &Bound<'_, PyAny>,
    incumbent_random: &Bound<'_, PyAny>,
    spec: GeneratorIncumbentSpec,
) -> u64 {
    let candidate_generator = seeded_pcg64_generator(candidate_random, spec.seed);
    let incumbent_generator = seeded_pcg64_generator(incumbent_random, spec.seed);
    let candidate_method = candidate_generator
        .getattr(spec.method_name)
        .expect("bind candidate Generator method");
    let incumbent_method = incumbent_generator
        .getattr(spec.method_name)
        .expect("bind incumbent Generator method");
    let candidate_output = call_generator_incumbent_distribution(&candidate_method, spec);
    let incumbent_output = call_generator_incumbent_distribution(&incumbent_method, spec);
    let candidate_next = candidate_generator
        .call_method1("random", (32_usize,))
        .expect("candidate post-distribution stream probe");
    let incumbent_next = incumbent_generator
        .call_method1("random", (32_usize,))
        .expect("incumbent post-distribution stream probe");
    let candidate_outputs = [candidate_output, candidate_next];
    let incumbent_outputs = [incumbent_output, incumbent_next];
    assert_workload_outputs_equal(numpy, spec.row, &candidate_outputs, &incumbent_outputs);
    workload_checksum(numpy, &candidate_outputs)
}

fn observe_numpy_output_array_bridge<T>(
    py: Python<'_>,
    numpy: &Bound<'_, PyModule>,
    values: &[T],
    dtype_name: &str,
) -> common::ContractObservation
where
    T: pyo3::buffer::Element + Copy,
{
    let started = Instant::now();
    let live_numpy = py.import("numpy").expect("re-import numpy output bridge");
    assert!(
        live_numpy.is(numpy),
        "candidate output bridge resolved a different NumPy module"
    );
    let kwargs = PyDict::new(py);
    kwargs
        .set_item("dtype", dtype_name)
        .expect("output bridge dtype");
    let array = live_numpy
        .call_method("empty", (values.len(),), Some(&kwargs))
        .expect("numpy.empty output bridge");
    if !values.is_empty() {
        let buffer = PyBuffer::<T>::get(&array).expect("typed NumPy output buffer");
        buffer
            .copy_from_slice(py, values)
            .expect("copy output bridge bytes");
    }
    let output_shape = PyTuple::new(py, [values.len()]).expect("output bridge shape");
    let output = array
        .call_method1("reshape", (&output_shape,))
        .expect("reshape output bridge");
    let elapsed = started.elapsed();
    common::ContractObservation {
        elapsed,
        checksum: workload_checksum(numpy, &[output]),
    }
}

fn component_min_of_three_median_ns<F>(mut observe: F) -> f64
where
    F: FnMut() -> common::ContractObservation,
{
    const ROUNDS: usize = 41;
    const MIN_OF: usize = 3;
    for _ in 0..4 {
        black_box(observe());
    }
    let mut samples = Vec::with_capacity(ROUNDS);
    let mut expected_checksum = None;
    for _ in 0..ROUNDS {
        let mut best = None;
        for _ in 0..MIN_OF {
            let observation = observe();
            if let Some(expected) = expected_checksum {
                assert_eq!(
                    observation.checksum, expected,
                    "shared output bridge checksum changed between observations"
                );
            } else {
                expected_checksum = Some(observation.checksum);
            }
            best = Some(best.map_or(observation.elapsed, |elapsed: Duration| {
                elapsed.min(observation.elapsed)
            }));
        }
        samples.push(
            best.expect("min-of-three output bridge observation")
                .as_secs_f64()
                * 1.0e9,
        );
    }
    samples.sort_by(f64::total_cmp);
    samples[ROUNDS / 2]
}

/// Convert the two banked Generator maintenance wins to public, same-invocation
/// NumPy ratios. These are distribution-batch jobs users recognize directly:
/// draw 100,000 variates from a long-lived seeded Generator. The parameters and
/// PCG64 seed match the historical self-speedup regimes exactly.
fn bench_generator_distribution_incumbent_ratios_median_gate(c: &mut Criterion) {
    let _ = c;
    const SIZE: usize = 100_000;
    const THREAD_ACTIVITY_REPETITIONS: usize = 5;
    const RAYON_THREADS: &str = "4";
    const NATIVE_LIBRARY_THREADS: &str = "1";
    const REQUIRED_BUILD_PROFILE: &str = "release-perf";
    const SHARED_COMPONENT: &str = "numpy.empty_reshape_buffer_bridge";
    const SPECS: [GeneratorIncumbentSpec; 2] = [
        GeneratorIncumbentSpec {
            row: "python_generator_zipf_pcg64_a2p5_100k_vs_numpy",
            method_name: "zipf",
            seed: 42,
            distribution: GeneratorIncumbentDistribution::Zipf,
        },
        GeneratorIncumbentSpec {
            row: "python_generator_noncentral_chisquare_pcg64_df5_nonc1_100k_vs_numpy",
            method_name: "noncentral_chisquare",
            seed: 42,
            distribution: GeneratorIncumbentDistribution::NoncentralChisquare,
        },
    ];

    assert_eq!(
        std::env::var("FNP_BENCH_PROFILE").as_deref(),
        Ok(REQUIRED_BUILD_PROFILE),
        "Generator incumbent evidence requires the ship-grade release-perf profile"
    );
    let build_worker =
        std::env::var("FNP_BUILD_WORKER").expect("FNP_BUILD_WORKER records the RCH build origin");
    assert!(
        !build_worker.trim().is_empty(),
        "FNP_BUILD_WORKER must name the RCH build worker"
    );
    assert_eq!(
        std::env::var("RAYON_NUM_THREADS").as_deref(),
        Ok(RAYON_THREADS),
        "Rayon configuration must be explicit"
    );
    for variable in [
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(NATIVE_LIBRARY_THREADS),
            "{variable} must be pinned to one thread"
        );
    }
    assert_eq!(
        rayon::current_num_threads(),
        4,
        "Rayon pool width differs from the declared configuration"
    );
    let host = std::env::var("HOSTNAME").unwrap_or_else(|_| "unavailable".to_owned());
    println!(
        "GENERATOR_RUNTIME host={host} build_worker={build_worker} \
         build_profile={REQUIRED_BUILD_PROFILE} rayon_pool_threads=4 \
         native_library_threads=1 actual_threads_reported_per_row=true"
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_generator_incumbent_ratios")
            .expect("Generator incumbent-ratio module");
        fnp_python(&module).expect("initialize fnp_python Generator incumbent-ratio module");
        let numpy = py.import("numpy").expect("numpy incumbent");
        let candidate_random = module.getattr("random").expect("fnp.random");
        let incumbent_random = numpy.getattr("random").expect("numpy.random");
        let numpy_generator_type = incumbent_random
            .getattr("Generator")
            .expect("numpy.random.Generator");

        let mut chooser_rows = Vec::with_capacity(SPECS.len());
        for spec in SPECS {
            let candidate_generator = seeded_pcg64_generator(&candidate_random, spec.seed);
            let incumbent_generator = seeded_pcg64_generator(&incumbent_random, spec.seed);
            assert!(
                !candidate_generator.is_exact_instance(&numpy_generator_type),
                "{}: candidate Generator resolved to the incumbent type",
                spec.row
            );
            let incumbent_method = incumbent_generator
                .getattr(spec.method_name)
                .expect("bind live NumPy Generator method");
            common::report_numpy_generator_method_identity(
                py,
                spec.method_name,
                &incumbent_generator,
                &incumbent_method,
            );
            let candidate_name = format!("fnp.random.Generator.{}", spec.method_name);
            let incumbent_name = format!("numpy.random.Generator.{}", spec.method_name);
            common::report_incumbent_topology_with_shared_component(
                &candidate_name,
                &incumbent_name,
                SHARED_COMPONENT,
            );

            let parameters = match spec.distribution {
                GeneratorIncumbentDistribution::Zipf => "a=2.5",
                GeneratorIncumbentDistribution::NoncentralChisquare => "df=5.0,nonc=1.0",
            };
            println!(
                "GENERATOR_REGIME row={} host={host} build_worker={build_worker} \
                 bit_generator=PCG64 seed={} samples={SIZE} parameters={} \
                 candidate_public_calls=1 incumbent_public_calls=1 \
                 generator_construction_outside_timed_region=true \
                 bound_method_lookup_outside_timed_region=true \
                 output_materialization_inside_timed_region=true \
                 shared_timed_component={SHARED_COMPONENT}",
                spec.row, spec.seed, parameters,
            );
            println!(
                "WORK_ACCOUNTING row={} candidate_samples={SIZE} \
                 incumbent_samples={SIZE} variable_rejection_draws=true \
                 exact_draw_count_not_instrumented=true less_work_claim=false",
                spec.row,
            );

            let parity_checksum = prove_generator_incumbent_distribution_stream(
                &numpy,
                &candidate_random,
                &incumbent_random,
                spec,
            );
            println!(
                "PARITY row={} generator_api=true at_scale=true samples={SIZE} \
                 dtype_shape_every_output_byte=identical \
                 next_stream_f64_values=32 next_stream_every_byte=identical \
                 aggregate_checksum={parity_checksum:016x}",
                spec.row,
            );

            common::report_observed_thread_activity(
                spec.row,
                "numpy",
                THREAD_ACTIVITY_REPETITIONS,
                || {
                    let generator = seeded_pcg64_generator(&incumbent_random, spec.seed);
                    let method = generator
                        .getattr(spec.method_name)
                        .expect("bind NumPy Generator thread probe");
                    black_box(call_generator_incumbent_distribution(&method, spec));
                },
            );
            common::report_observed_thread_activity(
                spec.row,
                "fnp",
                THREAD_ACTIVITY_REPETITIONS,
                || {
                    let generator = seeded_pcg64_generator(&candidate_random, spec.seed);
                    let method = generator
                        .getattr(spec.method_name)
                        .expect("bind FNP Generator thread probe");
                    black_box(call_generator_incumbent_distribution(&method, spec));
                },
            );

            let mut observe_incumbent =
                || observe_generator_incumbent_distribution(&numpy, &incumbent_random, spec);
            let mut observe_candidate =
                || observe_generator_incumbent_distribution(&numpy, &candidate_random, spec);
            let (effect, incumbent_null, candidate_null) = common::run_dual_null_median_ci_contract(
                spec.row,
                &mut observe_incumbent,
                &mut observe_candidate,
            );
            let statistical_verdict =
                common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);

            let output_bridge_median_ns = match spec.distribution {
                GeneratorIncumbentDistribution::Zipf => {
                    let values = vec![0_i64; SIZE];
                    component_min_of_three_median_ns(|| {
                        observe_numpy_output_array_bridge(py, &numpy, &values, "int64")
                    })
                }
                GeneratorIncumbentDistribution::NoncentralChisquare => {
                    let values = vec![0.0_f64; SIZE];
                    component_min_of_three_median_ns(|| {
                        observe_numpy_output_array_bridge(py, &numpy, &values, "float64")
                    })
                }
            };
            let shared_component_pct = output_bridge_median_ns / effect.arm_b_median_ns * 100.0;
            println!(
                "SHARED_COMPONENT_TELEMETRY row={} component={SHARED_COMPONENT} \
                 component_median_ms={:.6} candidate_median_ms={:.6} \
                 share_of_candidate_pct={shared_component_pct:.6} \
                 rounds=41 min_of=3 direction=conservative_for_candidate",
                spec.row,
                output_bridge_median_ns / 1_000_000.0,
                effect.arm_b_median_ns / 1_000_000.0,
            );

            let shared_component_admissible = shared_component_pct < 50.0;
            let (campaign_verdict, decision, reason) = match statistical_verdict {
                "DECIDABLE_WIN" if shared_component_admissible => (
                    "INCUMBENT_WIN_ELIGIBLE",
                    "choose_fnp",
                    "effect_CI_and_2x_null_margin_clear_and_shared_component_is_below_50pct",
                ),
                "DECIDABLE_WIN" => (
                    "MAINTENANCE_ONLY_SHARED_COMPONENT_DOMINATES",
                    "choose_numpy",
                    "candidate_routes_at_least_half_its_time_through_incumbent_code",
                ),
                "DECIDABLE_REGRESSION" => (
                    "INCUMBENT_REGRESSION",
                    "choose_numpy",
                    "effect_CI_and_2x_null_margin_show_numpy_faster",
                ),
                _ => (
                    "STATISTICALLY_UNDECIDED",
                    "choose_numpy_pending_remeasure",
                    "effect_does_not_clear_corrected_dual_null_gate",
                ),
            };
            println!(
                "INCUMBENT_RATIO row={} host={host} build_worker={build_worker} \
                 invocation_id={} statistical_verdict={statistical_verdict} \
                 campaign_verdict={campaign_verdict} \
                 incumbent_median_ms={:.6} candidate_median_ms={:.6} \
                 ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
                 incumbent_null_ratio={:.6} incumbent_null_ci95=[{:.6},{:.6}] \
                 candidate_null_ratio={:.6} candidate_null_ci95=[{:.6},{:.6}] \
                 effect_ci_excludes_one={} corrected_dual_null_gate=true \
                 median_clause=true actual_threads_reported=true \
                 shared_component_pct={shared_component_pct:.6}",
                spec.row,
                common::bench_invocation_id(),
                effect.arm_a_median_ns / 1_000_000.0,
                effect.arm_b_median_ns / 1_000_000.0,
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                incumbent_null.ratio_median,
                incumbent_null.ratio_ci_low,
                incumbent_null.ratio_ci_high,
                candidate_null.ratio_median,
                candidate_null.ratio_ci_low,
                candidate_null.ratio_ci_high,
                effect.ratio_ci_low > 1.0 || effect.ratio_ci_high < 1.0,
            );
            println!(
                "CHOOSER_STATEMENT workload={} decision={decision} reason={reason} \
                 measured_scope=PCG64_seed42_100000_samples_parameters_{} \
                 incumbent=numpy_live_same_invocation_no_result_cache \
                 outside_scope=run_same_contract_before_choosing",
                spec.method_name,
                parameters.replace([',', '='], "_"),
            );
            chooser_rows.push((spec.method_name, decision));
        }

        println!(
            "CHOOSER_STATEMENT workload=generator_distribution_batches \
             decision=method_specific choices={}:{};{}:{} \
             representative_job=draw_100000_variates_from_a_long_lived_seeded_Generator \
             incumbent=numpy_live_same_invocation \
             outside_measured_regimes=run_same_contract_before_choosing",
            chooser_rows[0].0, chooser_rows[0].1, chooser_rows[1].0, chooser_rows[1].1,
        );
    });
}

fn bench_bool_storage_bytes_median_gate(c: &mut Criterion) {
    let _ = c;
    const N: usize = 8_000_000;

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let numpy = py.import("numpy").expect("numpy oracle");

        // Deterministic mixed pattern; the byte content is what gets copied, so
        // an all-true or all-false buffer would not be representative.
        let values: Vec<bool> = (0..N).map(|index| index % 3 == 0).collect();

        let checksum_of = |array: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
            let bytes = array
                .call_method0("tobytes")
                .expect("tobytes")
                .extract::<Vec<u8>>()
                .expect("byte Vec");
            bytes
                .iter()
                .fold(0xcbf2_9ce4_8422_2325_u64, |state, &byte| {
                    (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                })
        };

        let build_from = |bytes: &[u8]| -> pyo3::Bound<'_, pyo3::PyAny> {
            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", "uint8").expect("dtype kwarg");
            let array = numpy
                .call_method("empty", (bytes.len(),), Some(&kwargs))
                .expect("numpy.empty");
            let buffer = pyo3::buffer::PyBuffer::<u8>::get(&array).expect("uint8 buffer");
            buffer.copy_from_slice(py, bytes).expect("buffer copy");
            array
        };

        let mut time_former = || {
            let started = Instant::now();
            let bytes: Vec<u8> = values.iter().map(|&b| u8::from(b)).collect();
            let array = build_from(black_box(&bytes));
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: checksum_of(&array),
            }
        };
        let mut time_candidate = || {
            let started = Instant::now();
            // SAFETY: `bool` and `u8` share size 1 and align 1, and every valid
            // `bool` is a valid `u8`, so the `Vec<bool>` allocation is a valid
            // `[u8]` of the same length for as long as `values` lives.
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len()) };
            let array = build_from(black_box(bytes));
            let elapsed = started.elapsed();
            common::ContractObservation {
                elapsed,
                checksum: checksum_of(&array),
            }
        };

        // Byte-identity before any timing, over and above the per-round checksum
        // equality the contract harness enforces.
        assert_eq!(
            time_former().checksum,
            time_candidate().checksum,
            "bool storage arms must produce byte-identical NumPy output",
        );

        let _ = common::run_median_ci_contract(
            "python_bool_storage_bytes_8m",
            &mut time_former,
            &mut time_candidate,
        );
    });
}

fn main() {
    common::gated_main(&[
        (
            "bench_average_f16_vs_numpy_median_gate",
            bench_average_f16_vs_numpy_median_gate,
        ),
        (
            "bench_bool_public_vs_numpy_median_gate",
            bench_bool_public_vs_numpy_median_gate,
        ),
        (
            "bench_flat_all_any_vs_numpy_median_gate",
            bench_flat_all_any_vs_numpy_median_gate,
        ),
        (
            "bench_class3_missing_capability_vs_numpy_median_gate",
            bench_class3_missing_capability_vs_numpy_median_gate,
        ),
        (
            "bench_char_upper_hold_redecision_median_gate",
            bench_char_upper_hold_redecision_median_gate,
        ),
        (
            "bench_generator_distribution_incumbent_ratios_median_gate",
            bench_generator_distribution_incumbent_ratios_median_gate,
        ),
        (
            "bench_fused_multiply_add_vs_numpy_median_gate",
            bench_fused_multiply_add_vs_numpy_median_gate,
        ),
        (
            "bench_fused_multiply_add_integer_matrix_vs_numpy_median_gate",
            bench_fused_multiply_add_integer_matrix_vs_numpy_median_gate,
        ),
        (
            "bench_fused_pairwise_multiply_add_matrix_vs_numpy_median_gate",
            bench_fused_pairwise_multiply_add_matrix_vs_numpy_median_gate,
        ),
        (
            "bench_fused_subtract_multiply_add_matrix_vs_numpy_median_gate",
            bench_fused_subtract_multiply_add_matrix_vs_numpy_median_gate,
        ),
        (
            "bench_fused_subtract_multiply_add_out_matrix_vs_numpy_median_gate",
            bench_fused_subtract_multiply_add_out_matrix_vs_numpy_median_gate,
        ),
        (
            "bench_fused_multiply_add_narrow_and_complex_matrix_vs_numpy_median_gate",
            bench_fused_multiply_add_narrow_and_complex_matrix_vs_numpy_median_gate,
        ),
        (
            "bench_fused_multiply_add_out_matrix_vs_numpy_median_gate",
            bench_fused_multiply_add_out_matrix_vs_numpy_median_gate,
        ),
        (
            "bench_f64_flat_min_max_vs_numpy_median_gate",
            bench_f64_flat_min_max_vs_numpy_median_gate,
        ),
        (
            "bench_isin_f64_vs_numpy_median_gate",
            bench_isin_f64_vs_numpy_median_gate,
        ),
        (
            "bench_isin_f64_readme_16m_vs_numpy_median_gate",
            bench_isin_f64_readme_16m_vs_numpy_median_gate,
        ),
        (
            "bench_int64_matmul_hold_redecision_median_gate",
            bench_int64_matmul_hold_redecision_median_gate,
        ),
        (
            "bench_float_flat_sum_matrix_vs_numpy_median_gate",
            bench_float_flat_sum_matrix_vs_numpy_median_gate,
        ),
        (
            "bench_float_flat_mean_matrix_vs_numpy_median_gate",
            bench_float_flat_sum_matrix_vs_numpy_median_gate,
        ),
        (
            "bench_integer_flat_sum_matrix_vs_numpy_median_gate",
            bench_integer_flat_sum_matrix_vs_numpy_median_gate,
        ),
        (
            "bench_int64_tofile_text_snapshot_vs_numpy_median_gate",
            bench_int64_tofile_text_snapshot_vs_numpy_median_gate,
        ),
        (
            "bench_quantile_f16_histogram_vs_numpy_median_gate",
            bench_quantile_f16_histogram_vs_numpy_median_gate,
        ),
        (
            "bench_wide_csv_sensor_etl_retry_vs_numpy_median_gate",
            bench_wide_csv_sensor_etl_retry_vs_numpy_median_gate,
        ),
        (
            "bench_realistic_end_to_end_workloads_vs_numpy_median_gate",
            bench_realistic_end_to_end_workloads_vs_numpy_median_gate,
        ),
        (
            "bench_realistic_access_control_exposure_vs_numpy_median_gate",
            bench_realistic_access_control_exposure_vs_numpy_median_gate,
        ),
        (
            "bench_realistic_critical_access_exposure_vs_numpy_median_gate",
            bench_realistic_critical_access_exposure_vs_numpy_median_gate,
        ),
        (
            "bench_realistic_rolling_load_saturation_vs_numpy_median_gate",
            bench_realistic_rolling_load_saturation_vs_numpy_median_gate,
        ),
        (
            "bench_realistic_clickstream_sessionization_vs_numpy_median_gate",
            bench_realistic_clickstream_sessionization_vs_numpy_median_gate,
        ),
        (
            "bench_realistic_event_attribution_scatter_vs_numpy_median_gate",
            bench_realistic_event_attribution_scatter_vs_numpy_median_gate,
        ),
        (
            "bench_realistic_entitlement_reconciliation_vs_numpy_median_gate",
            bench_realistic_entitlement_reconciliation_vs_numpy_median_gate,
        ),
        (
            "bench_realistic_f16_telemetry_distribution_vs_numpy_median_gate",
            bench_realistic_f16_telemetry_distribution_vs_numpy_median_gate,
        ),
        (
            "bench_realistic_f16_dynamic_range_audit_vs_numpy_median_gate",
            bench_realistic_f16_dynamic_range_audit_vs_numpy_median_gate,
        ),
        (
            "bench_realistic_f16_vector_field_polar_report_vs_numpy_median_gate",
            bench_realistic_f16_vector_field_polar_report_vs_numpy_median_gate,
        ),
        (
            "bench_bool_storage_bytes_median_gate",
            bench_bool_storage_bytes_median_gate,
        ),
        (
            "bench_loadtxt_selected_bool_median_gate",
            bench_loadtxt_selected_bool_median_gate,
        ),
        (
            "bench_loadtxt_negative_tail_vs_numpy_median_gate",
            bench_loadtxt_negative_tail_vs_numpy_median_gate,
        ),
        (
            "bench_wide_string_sort_median_gate",
            bench_wide_string_sort_median_gate,
        ),
        (
            "bench_accumulate_extremum_median_gate",
            bench_accumulate_extremum_median_gate,
        ),
        (
            "bench_int_convolve_median_gate",
            bench_int_convolve_median_gate,
        ),
        ("bench_completion_median_gate", bench_completion_median_gate),
        (
            "bench_f64_transcendental_median_gate",
            bench_f64_transcendental_median_gate,
        ),
        ("bench_f64_exp_log_probe", bench_f64_exp_log_probe),
        (
            "bench_f64_exp_log_median_gate",
            bench_f64_exp_log_median_gate,
        ),
        ("bench_bool_sort_median_gate", bench_bool_sort_median_gate),
        ("bench_int_matmul_median_gate", bench_int_matmul_median_gate),
        ("bench_f16_matmul_median_gate", bench_f16_matmul_median_gate),
        ("bench_multidot_median_gate", bench_multidot_median_gate),
        ("bench_isclose_median_gate", bench_isclose_median_gate),
        ("bench_f16_unique_median_gate", bench_f16_unique_median_gate),
        ("bench_f16_around_median_gate", bench_f16_around_median_gate),
        ("bench_f16_einsum_median_gate", bench_f16_einsum_median_gate),
        (
            "bench_wide_string_substrate_v2",
            bench_wide_string_substrate_v2,
        ),
        (
            "bench_ledger_integrity_rejects",
            bench_ledger_integrity_rejects,
        ),
    ]);
}
