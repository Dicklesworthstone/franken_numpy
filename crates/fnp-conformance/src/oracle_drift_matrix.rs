use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::io::Write as _;
use std::path::Path;
use std::process::{Command, Output, Stdio};

pub const DEFAULT_REPORT_PATH: &str = "target/oracle_drift_matrix.json";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneSpec {
    pub id: String,
    pub python: String,
    pub required: bool,
}

impl LaneSpec {
    pub fn new(id: impl Into<String>, python: impl Into<String>, required: bool) -> Self {
        Self {
            id: id.into(),
            python: python.into(),
            required,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LaneStatus {
    Available,
    Missing,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureStatus {
    Stable,
    Divergent,
    Unavailable,
}

#[derive(Debug, Clone, Serialize)]
pub struct LaneReport {
    pub id: String,
    pub python: String,
    pub required: bool,
    pub status: LaneStatus,
    pub python_version: Option<String>,
    pub numpy_version: Option<String>,
    pub observations: BTreeMap<String, Value>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FixtureObservation {
    pub lane_id: String,
    pub value_hash: String,
    pub value: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct FixtureMatrix {
    pub fixture_id: String,
    pub status: FixtureStatus,
    pub observations: Vec<FixtureObservation>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DriftSummary {
    pub available_lanes: usize,
    pub missing_optional_lanes: Vec<String>,
    pub unavailable_required_lanes: Vec<String>,
    pub stable_fixtures: usize,
    pub divergent_fixtures: usize,
    pub unavailable_fixtures: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct DriftReport {
    pub schema_version: u32,
    pub fixtures: Vec<String>,
    pub lanes: Vec<LaneReport>,
    pub matrix: Vec<FixtureMatrix>,
    pub summary: DriftSummary,
}

impl DriftReport {
    pub fn has_required_lane_failure(&self) -> bool {
        !self.summary.unavailable_required_lanes.is_empty()
    }

    pub fn write_pretty(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .map_err(|error| format!("create {}: {error}", parent.display()))?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|error| format!("serialize drift report: {error}"))?;
        fs::write(path, format!("{json}\n"))
            .map_err(|error| format!("write {}: {error}", path.display()))
    }
}

pub fn default_fixture_ids() -> Vec<String> {
    vec![
        "dtype_promotion_int_float".to_string(),
        "divide_by_zero_warning".to_string(),
    ]
}

pub fn default_lane_specs() -> Vec<LaneSpec> {
    let mut lanes = vec![LaneSpec::new(
        "repo-default",
        env::var("FNP_ORACLE_PYTHON").unwrap_or_else(|_| "python3".to_string()),
        true,
    )];
    for (env_key, lane_id) in [
        ("FNP_ORACLE_PYTHON_NUMPY1", "numpy1"),
        ("FNP_ORACLE_PYTHON_NUMPY2", "numpy2"),
        ("FNP_ORACLE_PYTHON_NIGHTLY", "numpy-nightly"),
    ] {
        if let Ok(python) = env::var(env_key) {
            lanes.push(LaneSpec::new(lane_id, python, false));
        }
    }
    lanes
}

pub fn parse_lane_spec(raw: &str, required: bool) -> Result<LaneSpec, String> {
    let (id, python) = raw
        .split_once('=')
        .ok_or_else(|| format!("lane spec must be id=python, got {raw:?}"))?;
    let id = id.trim();
    let python = python.trim();
    if id.is_empty() || python.is_empty() {
        return Err(format!(
            "lane spec must include non-empty id and python: {raw:?}"
        ));
    }
    Ok(LaneSpec::new(id, python, required))
}

pub fn run_report(lanes: &[LaneSpec], fixture_ids: &[String]) -> DriftReport {
    let lane_reports = lanes
        .iter()
        .map(|lane| run_lane(lane, fixture_ids))
        .collect::<Vec<_>>();
    build_report(fixture_ids.to_vec(), lane_reports)
}

pub fn run_lane(lane: &LaneSpec, fixture_ids: &[String]) -> LaneReport {
    let mut report = probe_lane(lane);
    if report.status != LaneStatus::Available {
        return report;
    }
    for fixture_id in fixture_ids {
        match run_fixture(&lane.python, fixture_id) {
            Ok(value) => {
                report.observations.insert(fixture_id.clone(), value);
            }
            Err(error) => {
                report.status = LaneStatus::Failed;
                report.error = Some(error);
                report.observations.clear();
                break;
            }
        }
    }
    report
}

pub fn probe_lane(lane: &LaneSpec) -> LaneReport {
    let output = run_python_script(
        &lane.python,
        r#"import json, sys
import numpy as np
print(json.dumps({"python_version": sys.version.split()[0], "numpy_version": np.__version__}, sort_keys=True))
"#,
        &[],
    );
    let output = match output {
        Ok(output) => output,
        Err(error) => {
            return LaneReport {
                id: lane.id.clone(),
                python: lane.python.clone(),
                required: lane.required,
                status: LaneStatus::Missing,
                python_version: None,
                numpy_version: None,
                observations: BTreeMap::new(),
                error: Some(error.to_string()),
            };
        }
    };
    if !output.status.success() {
        return LaneReport {
            id: lane.id.clone(),
            python: lane.python.clone(),
            required: lane.required,
            status: LaneStatus::Failed,
            python_version: None,
            numpy_version: None,
            observations: BTreeMap::new(),
            error: Some(String::from_utf8_lossy(&output.stderr).trim().to_string()),
        };
    }
    let value = match serde_json::from_slice::<Value>(&output.stdout) {
        Ok(value) => value,
        Err(error) => {
            return LaneReport {
                id: lane.id.clone(),
                python: lane.python.clone(),
                required: lane.required,
                status: LaneStatus::Failed,
                python_version: None,
                numpy_version: None,
                observations: BTreeMap::new(),
                error: Some(format!("parse probe json: {error}")),
            };
        }
    };
    LaneReport {
        id: lane.id.clone(),
        python: lane.python.clone(),
        required: lane.required,
        status: LaneStatus::Available,
        python_version: value
            .get("python_version")
            .and_then(Value::as_str)
            .map(str::to_string),
        numpy_version: value
            .get("numpy_version")
            .and_then(Value::as_str)
            .map(str::to_string),
        observations: BTreeMap::new(),
        error: None,
    }
}

pub fn run_fixture(python: &str, fixture_id: &str) -> Result<Value, String> {
    let output = run_python_script(
        python,
        r#"import json, sys, warnings
import numpy as np
fixture = sys.argv[1]
if fixture == "dtype_promotion_int_float":
    value = {
        "result_dtype": str(np.result_type(np.array([1], dtype=np.int32), np.array([1.5], dtype=np.float32)))
    }
elif fixture == "divide_by_zero_warning":
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = np.divide(np.array([1.0]), np.array([0.0]))
    value = {
        "result": [str(item) for item in result.tolist()],
        "warnings": [type(item.message).__name__ + ":" + str(item.message) for item in caught],
    }
else:
    raise SystemExit("unknown fixture: " + fixture)
print(json.dumps(value, sort_keys=True, separators=(",", ":")))
"#,
        &[fixture_id],
    )
    .map_err(|error| format!("launch {python}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "fixture {fixture_id} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("parse fixture {fixture_id} json: {error}"))
}

fn run_python_script(python: &str, script: &str, args: &[&str]) -> Result<Output, String> {
    let mut child = Command::new(python)
        .arg("-")
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| error.to_string())?;
    let stdin = child
        .stdin
        .as_mut()
        .ok_or_else(|| "python stdin should be piped".to_string())?;
    stdin
        .write_all(script.as_bytes())
        .map_err(|error| format!("write python script to stdin: {error}"))?;
    child.wait_with_output().map_err(|error| error.to_string())
}

pub fn build_report(fixtures: Vec<String>, lanes: Vec<LaneReport>) -> DriftReport {
    let matrix = build_matrix(&fixtures, &lanes);
    let missing_optional_lanes = lanes
        .iter()
        .filter(|lane| !lane.required && lane.status != LaneStatus::Available)
        .map(|lane| lane.id.clone())
        .collect::<Vec<_>>();
    let unavailable_required_lanes = lanes
        .iter()
        .filter(|lane| lane.required && lane.status != LaneStatus::Available)
        .map(|lane| lane.id.clone())
        .collect::<Vec<_>>();
    let summary = DriftSummary {
        available_lanes: lanes
            .iter()
            .filter(|lane| lane.status == LaneStatus::Available)
            .count(),
        missing_optional_lanes,
        unavailable_required_lanes,
        stable_fixtures: matrix
            .iter()
            .filter(|fixture| fixture.status == FixtureStatus::Stable)
            .count(),
        divergent_fixtures: matrix
            .iter()
            .filter(|fixture| fixture.status == FixtureStatus::Divergent)
            .count(),
        unavailable_fixtures: matrix
            .iter()
            .filter(|fixture| fixture.status == FixtureStatus::Unavailable)
            .count(),
    };
    DriftReport {
        schema_version: 1,
        fixtures,
        lanes,
        matrix,
        summary,
    }
}

pub fn build_matrix(fixtures: &[String], lanes: &[LaneReport]) -> Vec<FixtureMatrix> {
    fixtures
        .iter()
        .map(|fixture_id| {
            let observations = lanes
                .iter()
                .filter_map(|lane| {
                    lane.observations
                        .get(fixture_id)
                        .map(|value| FixtureObservation {
                            lane_id: lane.id.clone(),
                            value_hash: value_hash(value),
                            value: value.clone(),
                        })
                })
                .collect::<Vec<_>>();
            let unique_hashes = observations
                .iter()
                .map(|observation| observation.value_hash.clone())
                .collect::<BTreeSet<_>>();
            let status = match unique_hashes.len() {
                0 => FixtureStatus::Unavailable,
                1 => FixtureStatus::Stable,
                _ => FixtureStatus::Divergent,
            };
            FixtureMatrix {
                fixture_id: fixture_id.clone(),
                status,
                observations,
            }
        })
        .collect()
}

pub fn value_hash(value: &Value) -> String {
    let encoded = serde_json::to_vec(value).expect("serde_json::Value should encode");
    let digest = Sha256::digest(encoded);
    let mut encoded_digest = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut encoded_digest, "{byte:02x}").expect("writing to a String should not fail");
    }
    encoded_digest
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn lane_with_observation(id: &str, fixture_id: &str, value: Value) -> LaneReport {
        let mut observations = BTreeMap::new();
        observations.insert(fixture_id.to_string(), value);
        LaneReport {
            id: id.to_string(),
            python: format!("/python/{id}"),
            required: true,
            status: LaneStatus::Available,
            python_version: Some("3.14.0".to_string()),
            numpy_version: Some("2.4.0".to_string()),
            observations,
            error: None,
        }
    }

    #[test]
    fn oracle_drift_matrix_merge_marks_stable_fixture() {
        let fixtures = vec!["dtype_promotion_int_float".to_string()];
        let lanes = vec![
            lane_with_observation("numpy2", &fixtures[0], json!({"dtype": "float64"})),
            lane_with_observation("numpy-dev", &fixtures[0], json!({"dtype": "float64"})),
        ];

        let report = build_report(fixtures, lanes);

        assert_eq!(report.summary.stable_fixtures, 1);
        assert_eq!(report.summary.divergent_fixtures, 0);
        assert_eq!(report.matrix[0].status, FixtureStatus::Stable);
    }

    #[test]
    fn oracle_drift_matrix_merge_marks_version_divergence() {
        let fixtures = vec!["divide_by_zero_warning".to_string()];
        let lanes = vec![
            lane_with_observation("numpy1", &fixtures[0], json!({"warning": "RuntimeWarning"})),
            lane_with_observation(
                "numpy2",
                &fixtures[0],
                json!({"warning": "FloatingPointWarning"}),
            ),
        ];

        let report = build_report(fixtures, lanes);

        assert_eq!(report.summary.stable_fixtures, 0);
        assert_eq!(report.summary.divergent_fixtures, 1);
        assert_eq!(report.matrix[0].status, FixtureStatus::Divergent);
    }

    #[test]
    fn oracle_drift_matrix_missing_optional_lane_degrades() {
        let lane = LaneSpec::new(
            "missing-optional",
            "/definitely/missing/franken-numpy-python",
            false,
        );

        let report = build_report(
            vec!["dtype_promotion_int_float".to_string()],
            vec![run_lane(&lane, &default_fixture_ids())],
        );

        assert_eq!(report.summary.missing_optional_lanes, ["missing-optional"]);
        assert!(!report.has_required_lane_failure());
        assert_eq!(report.summary.unavailable_fixtures, 1);
    }

    #[test]
    fn oracle_drift_matrix_missing_required_lane_fails_report() {
        let lane = LaneSpec::new(
            "missing-required",
            "/definitely/missing/franken-numpy-python",
            true,
        );

        let report = build_report(
            vec!["dtype_promotion_int_float".to_string()],
            vec![run_lane(&lane, &default_fixture_ids())],
        );

        assert_eq!(
            report.summary.unavailable_required_lanes,
            ["missing-required"]
        );
        assert!(report.has_required_lane_failure());
    }

    #[test]
    fn oracle_drift_matrix_parse_lane_spec_requires_id_and_python() {
        let parsed = parse_lane_spec("numpy2=/opt/numpy2/bin/python", false).unwrap();
        assert_eq!(parsed.id, "numpy2");
        assert_eq!(parsed.python, "/opt/numpy2/bin/python");
        assert!(!parsed.required);
        assert!(parse_lane_spec("numpy2", false).is_err());
        assert!(parse_lane_spec("=/opt/python", false).is_err());
    }
}
