#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const SWARM_HANDOFF_SCHEMA_VERSION: &str = "fnp-swarm-handoff-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SwarmHandoffReport {
    pub schema_version: String,
    pub generated_by: String,
    pub sources: SwarmHandoffSources,
    pub summary: SwarmHandoffSummary,
    pub idea_wizard_graph: IdeaWizardGraph,
    pub beads: Vec<SwarmHandoffBead>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SwarmHandoffSources {
    pub issues_path: String,
    pub bv_source: String,
    pub bv_robot_command: Vec<String>,
    pub conformance_manifest_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SwarmHandoffSummary {
    pub open_count: usize,
    pub ready_count: usize,
    pub blocked_count: usize,
    pub closed_sample_count: usize,
    pub conformance_suite_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdeaWizardGraph {
    pub epic_ids: Vec<String>,
    pub blocker_ids: Vec<String>,
    pub labels: Vec<String>,
    pub note: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SwarmReadiness {
    Ready,
    Blocked,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SwarmHandoffBead {
    pub id: String,
    pub title: String,
    pub status: String,
    pub priority: u8,
    pub issue_type: String,
    pub labels: Vec<String>,
    pub readiness: SwarmReadiness,
    pub bv_score: Option<f64>,
    pub bv_reasons: Vec<String>,
    pub blocked_by: Vec<String>,
    pub unblocks: Vec<String>,
    pub suggested_reservation_globs: Vec<String>,
    pub rch_validation_commands: Vec<String>,
    pub related_conformance_shards: Vec<ConformanceShardHint>,
    pub likely_artifact_outputs: Vec<String>,
    pub proof_expectations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformanceShardHint {
    pub suite_name: String,
    pub suite_type: String,
    pub fixture: String,
    pub ci_gate: String,
    pub match_reason: String,
}

#[derive(Debug, Clone, Deserialize)]
struct IssueRecord {
    id: String,
    title: String,
    #[serde(default)]
    description: String,
    status: String,
    #[serde(default)]
    priority: Option<u8>,
    #[serde(default)]
    issue_type: Option<String>,
    #[serde(default)]
    labels: Vec<String>,
    #[serde(default)]
    dependencies: Vec<IssueDependency>,
}

#[derive(Debug, Clone, Deserialize)]
struct IssueDependency {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    depends_on_id: Option<String>,
    #[serde(default)]
    status: Option<String>,
}

impl IssueDependency {
    fn dependency_id(&self) -> Option<&str> {
        self.depends_on_id.as_deref().or(self.id.as_deref())
    }
}

#[derive(Debug, Clone, Deserialize)]
struct BvRobotEnvelope {
    triage: BvTriage,
}

#[derive(Debug, Clone, Deserialize)]
struct BvTriage {
    #[serde(default)]
    quick_ref: Option<BvQuickRef>,
    #[serde(default)]
    recommendations: Vec<BvRecommendation>,
    #[serde(default)]
    blockers_to_clear: Vec<BvBlocker>,
}

#[derive(Debug, Clone, Deserialize)]
struct BvQuickRef {
    #[serde(default)]
    open_count: usize,
    #[serde(default)]
    blocked_count: usize,
    #[serde(default)]
    top_picks: Vec<BvTopPick>,
}

#[derive(Debug, Clone, Deserialize)]
struct BvTopPick {
    id: String,
    #[serde(default)]
    reasons: Vec<String>,
    #[serde(default)]
    unblocks: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct BvRecommendation {
    id: String,
    #[serde(default)]
    score: Option<f64>,
    #[serde(default)]
    reasons: Vec<String>,
    #[serde(default)]
    blocked_by: Vec<String>,
    #[serde(default)]
    unblocks_ids: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct BvBlocker {
    id: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ConformanceManifest {
    suites: Vec<ConformanceManifestSuite>,
}

#[derive(Debug, Clone, Deserialize)]
struct ConformanceManifestSuite {
    name: String,
    #[serde(rename = "type")]
    suite_type: String,
    #[serde(default)]
    fixture: Option<String>,
    #[serde(default)]
    ci_gate: Option<String>,
}

pub fn default_issues_path(repo_root: &Path) -> PathBuf {
    repo_root.join(".beads/issues.jsonl")
}

pub fn default_conformance_manifest_path(repo_root: &Path) -> PathBuf {
    repo_root.join("artifacts/contracts/conformance_suite_manifest_v1.json")
}

pub fn default_output_path(repo_root: &Path) -> PathBuf {
    repo_root.join("target/swarm_handoff.json")
}

pub fn bv_robot_command() -> Vec<String> {
    vec!["bv".to_string(), "--robot-triage".to_string()]
}

pub fn read_bv_robot_triage(repo_root: &Path) -> Result<String, String> {
    let output = Command::new("bv")
        .arg("--robot-triage")
        .current_dir(repo_root)
        .output()
        .map_err(|err| format!("run bv --robot-triage: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "bv --robot-triage failed with status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8(output.stdout).map_err(|err| format!("bv output was not utf-8: {err}"))
}

pub fn build_swarm_handoff_report_from_paths(
    repo_root: &Path,
    issues_path: &Path,
    bv_source: &str,
    bv_json: &str,
    conformance_manifest_path: &Path,
) -> Result<SwarmHandoffReport, String> {
    let issues_jsonl = fs::read_to_string(issues_path)
        .map_err(|err| format!("read {}: {err}", issues_path.display()))?;
    let manifest_json = fs::read_to_string(conformance_manifest_path)
        .map_err(|err| format!("read {}: {err}", conformance_manifest_path.display()))?;
    build_swarm_handoff_report_from_inputs(
        repo_root,
        &issues_jsonl,
        bv_source,
        bv_json,
        &manifest_json,
        issues_path,
        conformance_manifest_path,
    )
}

pub fn build_swarm_handoff_report_from_inputs(
    repo_root: &Path,
    issues_jsonl: &str,
    bv_source: &str,
    bv_json: &str,
    manifest_json: &str,
    issues_path: &Path,
    conformance_manifest_path: &Path,
) -> Result<SwarmHandoffReport, String> {
    let issues = parse_issues(issues_jsonl)?;
    let bv: BvRobotEnvelope =
        serde_json::from_str(bv_json).map_err(|err| format!("parse bv robot json: {err}"))?;
    let manifest: ConformanceManifest =
        serde_json::from_str(manifest_json).map_err(|err| format!("parse manifest json: {err}"))?;
    let status_by_id = issues
        .iter()
        .map(|issue| (issue.id.clone(), issue.status.clone()))
        .collect::<BTreeMap<_, _>>();
    let bv_by_id = bv
        .triage
        .recommendations
        .iter()
        .map(|rec| (rec.id.clone(), rec))
        .collect::<BTreeMap<_, _>>();
    let top_pick_by_id = bv
        .triage
        .quick_ref
        .as_ref()
        .into_iter()
        .flat_map(|quick| quick.top_picks.iter())
        .map(|pick| (pick.id.clone(), pick))
        .collect::<BTreeMap<_, _>>();

    let mut open_count = 0usize;
    let mut closed_sample_count = 0usize;
    let mut rows = Vec::new();

    for issue in &issues {
        if issue.status == "closed" {
            closed_sample_count += 1;
            continue;
        }
        if !matches!(issue.status.as_str(), "open" | "in_progress") {
            continue;
        }
        open_count += 1;
        let recommendation = bv_by_id.get(&issue.id).copied();
        let top_pick = top_pick_by_id.get(&issue.id).copied();
        let mut blocked_by = recommendation
            .map(|rec| rec.blocked_by.clone())
            .unwrap_or_default();
        for dependency in &issue.dependencies {
            let Some(dep_id) = dependency.dependency_id() else {
                continue;
            };
            let dep_status = dependency
                .status
                .as_deref()
                .or_else(|| status_by_id.get(dep_id).map(String::as_str));
            if dep_status != Some("closed") && !blocked_by.iter().any(|id| id == dep_id) {
                blocked_by.push(dep_id.to_string());
            }
        }
        blocked_by.sort();
        blocked_by.dedup();
        let readiness = if blocked_by.is_empty() {
            SwarmReadiness::Ready
        } else {
            SwarmReadiness::Blocked
        };

        let labels = sorted(issue.labels.clone());
        let related_conformance_shards = conformance_shards_for_issue(issue, &manifest);
        rows.push(SwarmHandoffBead {
            id: issue.id.clone(),
            title: issue.title.clone(),
            status: issue.status.clone(),
            priority: issue.priority.unwrap_or(2),
            issue_type: issue
                .issue_type
                .clone()
                .unwrap_or_else(|| "task".to_string()),
            labels: labels.clone(),
            readiness,
            bv_score: recommendation.and_then(|rec| rec.score),
            bv_reasons: bv_reasons(recommendation, top_pick),
            blocked_by,
            unblocks: recommendation
                .map(|rec| sorted(rec.unblocks_ids.clone()))
                .unwrap_or_default(),
            suggested_reservation_globs: reservation_globs_for_issue(issue),
            rch_validation_commands: rch_commands_for_issue(issue, &related_conformance_shards),
            related_conformance_shards,
            likely_artifact_outputs: artifact_outputs_for_issue(issue),
            proof_expectations: proof_expectations_for_issue(issue),
        });
    }

    rows.sort_by(|lhs, rhs| {
        readiness_rank(lhs.readiness)
            .cmp(&readiness_rank(rhs.readiness))
            .then_with(|| {
                rhs.bv_score
                    .partial_cmp(&lhs.bv_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| lhs.id.cmp(&rhs.id))
    });

    let ready_count = rows
        .iter()
        .filter(|row| row.readiness == SwarmReadiness::Ready)
        .count();
    let blocked_count = rows.len().saturating_sub(ready_count);

    Ok(SwarmHandoffReport {
        schema_version: SWARM_HANDOFF_SCHEMA_VERSION.to_string(),
        generated_by: "fnp-conformance".to_string(),
        sources: SwarmHandoffSources {
            issues_path: relative_display(repo_root, issues_path),
            bv_source: bv_source.to_string(),
            bv_robot_command: bv_robot_command(),
            conformance_manifest_path: relative_display(repo_root, conformance_manifest_path),
        },
        summary: SwarmHandoffSummary {
            open_count: bv
                .triage
                .quick_ref
                .as_ref()
                .map_or(open_count, |quick| quick.open_count),
            ready_count,
            blocked_count: bv
                .triage
                .quick_ref
                .as_ref()
                .map_or(blocked_count, |quick| quick.blocked_count),
            closed_sample_count,
            conformance_suite_count: manifest.suites.len(),
        },
        idea_wizard_graph: idea_wizard_graph(&issues, &bv),
        beads: rows,
    })
}

pub fn write_report_json(report: &SwarmHandoffReport, output_path: &Path) -> Result<(), String> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|err| format!("create {}: {err}", parent.display()))?;
    }
    let raw = serde_json::to_string_pretty(report)
        .map_err(|err| format!("serialize swarm handoff report: {err}"))?;
    fs::write(output_path, format!("{raw}\n"))
        .map_err(|err| format!("write {}: {err}", output_path.display()))
}

pub fn render_terminal_report(report: &SwarmHandoffReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "swarm_handoff open={} ready={} blocked={} suites={}\n",
        report.summary.open_count,
        report.summary.ready_count,
        report.summary.blocked_count,
        report.summary.conformance_suite_count
    ));
    for bead in report
        .beads
        .iter()
        .filter(|bead| bead.readiness == SwarmReadiness::Ready)
        .take(10)
    {
        out.push_str(&format!(
            "- {} P{} {} shards={} commands={}\n",
            bead.id,
            bead.priority,
            bead.title,
            bead.related_conformance_shards.len(),
            bead.rch_validation_commands.len()
        ));
    }
    out
}

fn parse_issues(raw: &str) -> Result<Vec<IssueRecord>, String> {
    let mut issues = Vec::new();
    for (idx, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        issues.push(
            serde_json::from_str::<IssueRecord>(line)
                .map_err(|err| format!("parse issue jsonl line {}: {err}", idx + 1))?,
        );
    }
    Ok(issues)
}

fn bv_reasons(
    recommendation: Option<&BvRecommendation>,
    top_pick: Option<&BvTopPick>,
) -> Vec<String> {
    let mut reasons = BTreeSet::new();
    if let Some(rec) = recommendation {
        reasons.extend(rec.reasons.iter().cloned());
    }
    if let Some(pick) = top_pick {
        reasons.extend(pick.reasons.iter().cloned());
        if pick.unblocks > 0 {
            reasons.insert(format!("unblocks {} downstream item(s)", pick.unblocks));
        }
    }
    reasons.into_iter().collect()
}

fn conformance_shards_for_issue(
    issue: &IssueRecord,
    manifest: &ConformanceManifest,
) -> Vec<ConformanceShardHint> {
    let families = matched_families(issue);
    let mut hints = Vec::new();
    for suite in &manifest.suites {
        let haystack = format!(
            "{} {} {}",
            suite.name.to_lowercase(),
            suite.fixture.as_deref().unwrap_or_default().to_lowercase(),
            suite.suite_type.to_lowercase()
        );
        let Some(family) = families.iter().find(|family| haystack.contains(*family)) else {
            continue;
        };
        hints.push(ConformanceShardHint {
            suite_name: suite.name.clone(),
            suite_type: suite.suite_type.clone(),
            fixture: suite.fixture.clone().unwrap_or_default(),
            ci_gate: suite.ci_gate.clone().unwrap_or_default(),
            match_reason: format!("matched {family} keyword from bead metadata"),
        });
    }
    hints.sort_by(|lhs, rhs| lhs.suite_name.cmp(&rhs.suite_name));
    hints.truncate(8);
    hints
}

fn matched_families(issue: &IssueRecord) -> Vec<&'static str> {
    let text = issue_text(issue);
    let mut families = BTreeSet::new();
    for (keyword, family) in [
        ("dtype", "dtype"),
        ("cast", "dtype"),
        ("shape", "shape"),
        ("stride", "shape"),
        ("broadcast", "shape"),
        ("matrix", "linalg"),
        ("linalg", "linalg"),
        ("masked", "masked"),
        ("mask", "masked"),
        ("random", "rng"),
        ("rng", "rng"),
        ("fft", "fft"),
        ("string", "string"),
        ("datetime", "datetime"),
        ("polynomial", "polynomial"),
        ("signal", "signal"),
        ("ufunc", "ufunc"),
        ("runtime", "runtime"),
        ("iter", "iter"),
        ("io", "io"),
        ("npy", "io"),
        ("npz", "io"),
        ("raptorq", "raptorq"),
    ] {
        if text.contains(keyword) {
            families.insert(family);
        }
    }
    families.into_iter().collect()
}

fn reservation_globs_for_issue(issue: &IssueRecord) -> Vec<String> {
    let text = issue_text(issue);
    let mut globs = BTreeSet::new();
    if text.contains("python") {
        globs.insert("crates/fnp-python/src/lib.rs".to_string());
        globs.insert("crates/fnp-python/tests/conformance_*.rs".to_string());
        globs.insert("crates/fnp-conformance/src/fnp_python_api_coverage.rs".to_string());
    }
    if text.contains("dtype") || text.contains("cast") {
        globs.insert("crates/fnp-dtype/src/lib.rs".to_string());
        globs.insert("crates/fnp-python/tests/conformance_dtype*.rs".to_string());
        globs.insert("crates/fnp-conformance/fixtures/*dtype*".to_string());
    }
    if text.contains("shape") || text.contains("stride") || text.contains("broadcast") {
        globs.insert("crates/fnp-ndarray/src/lib.rs".to_string());
        globs.insert("crates/fnp-python/tests/conformance_shape*.rs".to_string());
    }
    if text.contains("masked") || text.contains("mask") {
        globs.insert("crates/fnp-python/tests/conformance_ma*.rs".to_string());
        globs.insert("crates/fnp-conformance/fixtures/masked_*".to_string());
    }
    if text.contains("raptorq") {
        globs.insert("crates/fnp-conformance/src/raptorq_artifacts.rs".to_string());
        globs.insert("crates/fnp-conformance/src/bin/run_raptorq_gate.rs".to_string());
        globs.insert("scripts/e2e/run_raptorq*_gate.sh".to_string());
    }
    if text.contains("swarm") || text.contains("proof backlog") || text.contains("tooling") {
        globs.insert("crates/fnp-conformance/src/swarm_handoff.rs".to_string());
        globs.insert("crates/fnp-conformance/src/bin/generate_swarm_handoff_report.rs".to_string());
    }
    if globs.is_empty() {
        globs.insert("crates/fnp-conformance/src/**".to_string());
    }
    globs.insert(".beads/issues.jsonl".to_string());
    globs.into_iter().collect()
}

fn rch_commands_for_issue(issue: &IssueRecord, shards: &[ConformanceShardHint]) -> Vec<String> {
    let text = issue_text(issue);
    let mut commands = BTreeSet::new();
    for shard in shards.iter().take(3) {
        let filter = shard
            .suite_name
            .strip_prefix("run_")
            .unwrap_or(&shard.suite_name)
            .strip_suffix("_suite")
            .unwrap_or(shard.suite_name.as_str());
        commands.insert(format!(
            "rch exec -- cargo test -p fnp-conformance {filter} -- --nocapture"
        ));
    }
    if text.contains("python") {
        commands.insert("rch exec -- cargo run -p fnp-conformance --bin run_fnp_python_api_coverage -- --report-path target/fnp_python_api_coverage.json".to_string());
    }
    if text.contains("raptorq") {
        commands.insert("rch exec -- cargo run -p fnp-conformance --bin run_raptorq_gate -- --report-path target/raptorq_stress_report.json --coverage-floor 1.0".to_string());
    }
    commands.insert("rch exec -- cargo check -p fnp-conformance --all-targets".to_string());
    commands.into_iter().collect()
}

fn artifact_outputs_for_issue(issue: &IssueRecord) -> Vec<String> {
    let text = issue_text(issue);
    let mut outputs = BTreeSet::new();
    outputs.insert(format!("target/swarm_handoff/{}.json", issue.id));
    if text.contains("python") {
        outputs.insert("target/fnp_python_api_coverage.json".to_string());
    }
    if text.contains("raptorq") {
        outputs.insert("target/raptorq_stress_report.json".to_string());
    }
    if text.contains("performance") || text.contains("many-core") {
        outputs.insert("target/parallel_budget_report.json".to_string());
    }
    outputs.into_iter().collect()
}

fn proof_expectations_for_issue(issue: &IssueRecord) -> Vec<String> {
    let mut expectations = BTreeSet::new();
    expectations.insert("reserve the listed globs before editing".to_string());
    expectations.insert("run validation through rch, not local cargo".to_string());
    expectations.insert(
        "record generated reports under target/ unless a bead asks for committed artifacts"
            .to_string(),
    );
    expectations.insert("do not create or mutate beads from this report generator".to_string());
    if issue_text(issue).contains("idea-wizard") {
        expectations.insert("preserve linkage to the idea-wizard many-core graph".to_string());
    }
    expectations.into_iter().collect()
}

fn idea_wizard_graph(issues: &[IssueRecord], bv: &BvRobotEnvelope) -> IdeaWizardGraph {
    let mut epic_ids = BTreeSet::new();
    let mut blocker_ids = BTreeSet::new();
    for issue in issues {
        if issue.status == "closed" {
            continue;
        }
        if issue.labels.iter().any(|label| label == "idea-wizard")
            && (issue.issue_type.as_deref() == Some("epic") || issue.title.contains("idea-wizard"))
        {
            epic_ids.insert(issue.id.clone());
        }
    }
    for blocker in &bv.triage.blockers_to_clear {
        blocker_ids.insert(blocker.id.clone());
    }
    IdeaWizardGraph {
        epic_ids: epic_ids.into_iter().collect(),
        blocker_ids: blocker_ids.into_iter().collect(),
        labels: vec![
            "idea-wizard".to_string(),
            "many-core".to_string(),
            "swarm".to_string(),
        ],
        note: "Report-only view of the idea-wizard many-core graph; it never creates or mutates beads.".to_string(),
    }
}

fn issue_text(issue: &IssueRecord) -> String {
    format!(
        "{} {} {} {}",
        issue.title,
        issue.description,
        issue.issue_type.as_deref().unwrap_or_default(),
        issue.labels.join(" ")
    )
    .to_lowercase()
}

fn sorted(mut values: Vec<String>) -> Vec<String> {
    values.sort();
    values.dedup();
    values
}

fn readiness_rank(readiness: SwarmReadiness) -> u8 {
    match readiness {
        SwarmReadiness::Ready => 0,
        SwarmReadiness::Blocked => 1,
    }
}

fn relative_display(repo_root: &Path, path: &Path) -> String {
    path.strip_prefix(repo_root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::{
        SWARM_HANDOFF_SCHEMA_VERSION, SwarmReadiness, build_swarm_handoff_report_from_inputs,
        bv_robot_command,
    };
    use std::path::Path;

    const ISSUES: &str = r#"{"id":"ready-dtype","title":"Add dtype API coverage","description":"dtype conformance proof","status":"open","priority":2,"issue_type":"task","labels":["python","dtype","conformance"],"dependencies":[]}
{"id":"blocked-shape","title":"Shape helper coverage","description":"shape helpers","status":"open","priority":2,"issue_type":"task","labels":["shape","conformance"],"dependencies":[{"depends_on_id":"closed-parent"}]}
{"id":"closed-parent","title":"Closed parent","description":"","status":"closed","priority":2,"issue_type":"task","labels":[],"dependencies":[]}
{"id":"idea-epic","title":"[idea-wizard] Many-core conformance and performance wave","description":"","status":"open","priority":1,"issue_type":"epic","labels":["idea-wizard","many-core","swarm"],"dependencies":[{"depends_on_id":"ready-dtype"}]}"#;

    const BV: &str = r#"{
  "triage": {
    "quick_ref": {
      "open_count": 3,
      "blocked_count": 2,
      "top_picks": [
        {"id": "ready-dtype", "reasons": ["available"], "unblocks": 1}
      ]
    },
    "recommendations": [
      {"id": "ready-dtype", "score": 0.9, "reasons": ["available"], "blocked_by": [], "unblocks_ids": ["idea-epic"]},
      {"id": "blocked-shape", "score": 0.4, "reasons": ["blocked"], "blocked_by": ["closed-parent"], "unblocks_ids": []},
      {"id": "idea-epic", "score": 0.8, "reasons": ["blocked"], "blocked_by": ["ready-dtype"], "unblocks_ids": []}
    ],
    "blockers_to_clear": [{"id": "ready-dtype"}]
  }
}"#;

    const MANIFEST: &str = r#"{
  "suites": [
    {"name":"run_dtype_differential_suite","type":"differential","fixture":"dtype_differential_cases.json","ci_gate":"G5"},
    {"name":"run_shape_stride_suite","type":"basic","fixture":"shape_stride_cases.json","ci_gate":"G5"},
    {"name":"run_crash_signature_regression_suite","type":"regression","fixture":null,"ci_gate":null}
  ]
}"#;

    #[test]
    fn swarm_handoff_report_generation_covers_ready_blocked_and_closed_samples() {
        let repo_root = Path::new("/repo");
        let report = build_swarm_handoff_report_from_inputs(
            repo_root,
            ISSUES,
            "fixture",
            BV,
            MANIFEST,
            Path::new("/repo/.beads/issues.jsonl"),
            Path::new("/repo/artifacts/contracts/conformance_suite_manifest_v1.json"),
        )
        .expect("report");

        assert_eq!(report.schema_version, SWARM_HANDOFF_SCHEMA_VERSION);
        assert_eq!(report.summary.open_count, 3);
        assert_eq!(report.summary.ready_count, 1);
        assert_eq!(report.summary.blocked_count, 2);
        assert_eq!(report.summary.closed_sample_count, 1);
        assert_eq!(report.summary.conformance_suite_count, 3);
        assert_eq!(report.beads[0].id, "ready-dtype");
        assert_eq!(report.beads[0].readiness, SwarmReadiness::Ready);
        assert!(
            report.beads[0]
                .suggested_reservation_globs
                .iter()
                .any(|glob| glob.contains("fnp-dtype"))
        );
        assert!(
            report.beads[0]
                .rch_validation_commands
                .iter()
                .any(|command| command.contains("cargo test -p fnp-conformance dtype"))
        );
        assert_eq!(
            report.beads[0].related_conformance_shards[0].suite_name,
            "run_dtype_differential_suite"
        );
        assert!(!report.beads.iter().any(|bead| bead.id == "closed-parent"));
    }

    #[test]
    fn swarm_handoff_report_points_at_idea_wizard_graph_without_mutation() {
        let report = build_swarm_handoff_report_from_inputs(
            Path::new("/repo"),
            ISSUES,
            "fixture",
            BV,
            MANIFEST,
            Path::new("/repo/.beads/issues.jsonl"),
            Path::new("/repo/artifacts/contracts/conformance_suite_manifest_v1.json"),
        )
        .expect("report");

        assert_eq!(report.idea_wizard_graph.epic_ids, vec!["idea-epic"]);
        assert_eq!(report.idea_wizard_graph.blocker_ids, vec!["ready-dtype"]);
        assert!(
            report
                .idea_wizard_graph
                .note
                .contains("never creates or mutates beads")
        );
    }

    #[test]
    fn swarm_handoff_report_uses_robot_mode_bv_command() {
        assert_eq!(
            bv_robot_command(),
            vec!["bv".to_string(), "--robot-triage".to_string()]
        );
    }
}
