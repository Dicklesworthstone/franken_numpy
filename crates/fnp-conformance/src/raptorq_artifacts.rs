#![forbid(unsafe_code)]

use crate::{HarnessConfig, SuiteReport};
use asupersync::config::EncodingConfig;
use asupersync::decoding::{DecodingConfig, DecodingPipeline};
use asupersync::encoding::EncodingPipeline;
use asupersync::security::{AuthenticatedSymbol, AuthenticationTag};
use asupersync::types::resource::{PoolConfig, SymbolPool};
use asupersync::types::{ObjectId, ObjectParams, Symbol, SymbolId, SymbolKind};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub const MAX_RAPTORQ_PARALLELISM: usize = 64;
pub const RAPTORQ_STRESS_REPORT_SCHEMA_VERSION: u8 = 1;
pub const RAPTORQ_STRESS_SMOKE_BYTES: usize = 64 * 1024;
pub const RAPTORQ_STRESS_LOCAL_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RaptorQParallelismConfig {
    pub worker_count: usize,
}

impl RaptorQParallelismConfig {
    pub const fn serial() -> Self {
        Self { worker_count: 1 }
    }

    pub fn from_worker_count(worker_count: usize) -> Result<Self, String> {
        if worker_count == 0 {
            return Err("raptorq parallelism must be at least 1".to_string());
        }
        if worker_count > MAX_RAPTORQ_PARALLELISM {
            return Err(format!(
                "raptorq parallelism {worker_count} exceeds max {MAX_RAPTORQ_PARALLELISM}"
            ));
        }
        Ok(Self { worker_count })
    }

    pub fn available() -> Self {
        let available = thread::available_parallelism().map_or(1, usize::from);
        Self {
            worker_count: available.clamp(1, MAX_RAPTORQ_PARALLELISM),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleFilePayload {
    pub path: String,
    pub sha256: String,
    pub size: usize,
    pub bytes_b64: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundlePayload {
    pub schema_version: u8,
    pub bundle_id: String,
    pub generated_at_unix_ms: u128,
    pub files: Vec<BundleFilePayload>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaptorQSymbolRecord {
    pub sbn: u8,
    pub esi: u32,
    pub kind: String,
    pub data_b64: String,
    pub data_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaptorQSidecar {
    pub schema_version: u8,
    pub bundle_id: String,
    pub generated_at_unix_ms: u128,
    pub source_hash: String,
    pub source_size: usize,
    pub object_id_u128: u128,
    pub symbol_size: u16,
    pub max_block_size: usize,
    pub repair_overhead: f64,
    pub source_blocks: u8,
    pub source_symbols: u16,
    pub repair_symbols: usize,
    pub total_symbols: usize,
    pub encoding_parallelism: usize,
    pub decoding_parallelism: usize,
    pub symbols: Vec<RaptorQSymbolRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrubReport {
    pub schema_version: u8,
    pub bundle_id: String,
    pub generated_at_unix_ms: u128,
    pub status: String,
    pub expected_hash: String,
    pub decoded_hash: String,
    pub full_decode_match: bool,
    pub recovery_decode_match: bool,
    pub symbols_total: usize,
    pub symbols_used_full: usize,
    pub symbols_used_recovery: usize,
    pub decoding_parallelism: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeProofArtifact {
    pub schema_version: u8,
    pub bundle_id: String,
    pub generated_at_unix_ms: u128,
    pub dropped_symbol: Option<String>,
    pub recovery_symbols_used: usize,
    pub recovery_success: bool,
    pub expected_hash: String,
    pub recovered_hash: Option<String>,
    pub error: Option<String>,
    pub decoding_parallelism: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RaptorQStressMode {
    Smoke,
    Local,
}

impl RaptorQStressMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Local => "local",
        }
    }

    pub const fn default_source_bytes(self) -> usize {
        match self {
            Self::Smoke => RAPTORQ_STRESS_SMOKE_BYTES,
            Self::Local => RAPTORQ_STRESS_LOCAL_BYTES,
        }
    }

    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "smoke" => Ok(Self::Smoke),
            "local" => Ok(Self::Local),
            other => Err(format!(
                "invalid raptorq stress mode '{other}', expected smoke or local"
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RaptorQStressGateConfig {
    pub repo_root: PathBuf,
    pub output_dir: PathBuf,
    pub mode: RaptorQStressMode,
    pub parallelism: RaptorQParallelismConfig,
    pub source_bytes: usize,
    pub replay_command: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaptorQStressReport {
    pub schema_version: u8,
    pub bundle_id: String,
    pub mode: RaptorQStressMode,
    pub status: String,
    pub worker_count: usize,
    pub output_dir: String,
    pub input_files: Vec<String>,
    pub sidecar_path: String,
    pub scrub_report_path: String,
    pub decode_proof_path: String,
    pub input_hash: String,
    pub recovered_hash: String,
    pub elapsed_ms: u128,
    pub source_size: usize,
    pub source_symbols: u16,
    pub repair_symbols: usize,
    pub total_symbols: usize,
    pub dropped_symbol_scenario: Option<String>,
    pub recovery_symbols_used: usize,
    pub replay_command: String,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct RaptorQBundleReportPaths<'a> {
    pub sidecar_path: &'a Path,
    pub scrub_report_path: &'a Path,
    pub decode_proof_path: &'a Path,
}

impl<'a> RaptorQBundleReportPaths<'a> {
    pub const fn new(
        sidecar_path: &'a Path,
        scrub_report_path: &'a Path,
        decode_proof_path: &'a Path,
    ) -> Self {
        Self {
            sidecar_path,
            scrub_report_path,
            decode_proof_path,
        }
    }
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

pub fn build_bundle_payload(
    bundle_id: &str,
    repo_root: &Path,
    files: &[PathBuf],
) -> Result<Vec<u8>, String> {
    let canonical_repo_root =
        fs::canonicalize(repo_root).unwrap_or_else(|_| repo_root.to_path_buf());
    let mut sorted_files = files.to_vec();
    sorted_files.sort_by(|a, b| a.as_os_str().cmp(b.as_os_str()));

    let mut payload_files = Vec::with_capacity(sorted_files.len());

    for file_path in sorted_files {
        let canonical_file_path = fs::canonicalize(&file_path).unwrap_or(file_path);
        let bytes = fs::read(&canonical_file_path)
            .map_err(|err| format!("failed reading {}: {err}", canonical_file_path.display()))?;

        let rel = canonical_file_path
            .strip_prefix(&canonical_repo_root)
            .unwrap_or(canonical_file_path.as_path())
            .to_string_lossy()
            .to_string();

        payload_files.push(BundleFilePayload {
            path: rel,
            sha256: sha256_hex(&bytes),
            size: bytes.len(),
            bytes_b64: BASE64.encode(bytes),
        });
    }

    let payload = BundlePayload {
        schema_version: 1,
        bundle_id: bundle_id.to_string(),
        // Keep bundle payload hashing deterministic across regeneration/verification.
        generated_at_unix_ms: 0,
        files: payload_files,
    };

    serde_json::to_vec(&payload).map_err(|err| format!("failed serializing bundle payload: {err}"))
}

pub fn generate_sidecar_from_payload(
    bundle_id: &str,
    payload: &[u8],
    sidecar_path: &Path,
    object_seed: u64,
) -> Result<RaptorQSidecar, String> {
    generate_sidecar_from_payload_with_config(
        bundle_id,
        payload,
        sidecar_path,
        object_seed,
        RaptorQParallelismConfig::serial(),
    )
}

pub fn generate_sidecar_from_payload_with_config(
    bundle_id: &str,
    payload: &[u8],
    sidecar_path: &Path,
    object_seed: u64,
    parallelism: RaptorQParallelismConfig,
) -> Result<RaptorQSidecar, String> {
    let symbol_size = 256u16;
    let max_block_size = payload.len().max(usize::from(symbol_size));

    let config = EncodingConfig {
        repair_overhead: 1.25,
        max_block_size,
        symbol_size,
        encoding_parallelism: parallelism.worker_count,
        decoding_parallelism: parallelism.worker_count,
    };

    let source_symbol_count = payload.len().div_ceil(usize::from(symbol_size)).max(1);
    let repair_count = source_symbol_count.div_ceil(4).max(2);

    let pool = SymbolPool::new(PoolConfig {
        symbol_size,
        initial_size: 0,
        max_size: source_symbol_count + repair_count + 16,
        allow_growth: true,
        growth_increment: 16,
    });

    let object_id = ObjectId::new_for_test(object_seed);
    let mut pipeline = EncodingPipeline::new(config.clone(), pool);

    let mut symbol_records = Vec::new();

    let iterator = pipeline.encode_with_repair(object_id, payload, repair_count);
    for item in iterator {
        let encoded = item.map_err(|err| format!("encoding failed: {err}"))?;
        let symbol = encoded.into_symbol();

        symbol_records.push(RaptorQSymbolRecord {
            sbn: symbol.sbn(),
            esi: symbol.esi(),
            kind: match symbol.kind() {
                SymbolKind::Source => "source".to_string(),
                SymbolKind::Repair => "repair".to_string(),
            },
            data_sha256: sha256_hex(symbol.data()),
            data_b64: BASE64.encode(symbol.data()),
        });
    }
    symbol_records.sort_by_key(|record| (record.sbn, record.esi, symbol_kind_rank(&record.kind)));

    let stats = pipeline.stats();
    let source_blocks = u8::try_from(stats.blocks)
        .map_err(|_| format!("too many source blocks: {}", stats.blocks))?;
    let source_symbols = u16::try_from(stats.source_symbols)
        .map_err(|_| format!("too many source symbols: {}", stats.source_symbols))?;

    let sidecar = RaptorQSidecar {
        schema_version: 1,
        bundle_id: bundle_id.to_string(),
        generated_at_unix_ms: now_unix_ms(),
        source_hash: sha256_hex(payload),
        source_size: payload.len(),
        object_id_u128: object_id.as_u128(),
        symbol_size,
        max_block_size,
        repair_overhead: config.repair_overhead,
        source_blocks,
        source_symbols,
        repair_symbols: stats.repair_symbols,
        total_symbols: symbol_records.len(),
        encoding_parallelism: config.encoding_parallelism,
        decoding_parallelism: config.decoding_parallelism,
        symbols: symbol_records,
    };

    if let Some(parent) = sidecar_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }

    let raw = serde_json::to_string_pretty(&sidecar)
        .map_err(|err| format!("failed serializing sidecar: {err}"))?;
    fs::write(sidecar_path, raw)
        .map_err(|err| format!("failed writing {}: {err}", sidecar_path.display()))?;

    Ok(sidecar)
}

fn symbol_kind_rank(kind: &str) -> u8 {
    match kind {
        "source" => 0,
        "repair" => 1,
        _ => u8::MAX,
    }
}

fn decode_payload_from_records(
    sidecar: &RaptorQSidecar,
    records: &[RaptorQSymbolRecord],
) -> Result<Vec<u8>, String> {
    let mut decoder = DecodingPipeline::new(DecodingConfig {
        symbol_size: sidecar.symbol_size,
        max_block_size: sidecar.max_block_size,
        // For scrub/recovery drills we decode with the minimal admissible overhead
        // to validate recoverability even when some symbols are intentionally missing.
        repair_overhead: 1.0,
        min_overhead: 0,
        max_buffered_symbols: 0,
        block_timeout: std::time::Duration::from_secs(10),
        verify_auth: false,
    });

    let object_id = ObjectId::from_u128(sidecar.object_id_u128);
    let params = ObjectParams::new(
        object_id,
        sidecar.source_size as u64,
        sidecar.symbol_size,
        sidecar.source_blocks.into(),
        sidecar.source_symbols,
    );
    decoder
        .set_object_params(params)
        .map_err(|err| format!("set_object_params failed: {err}"))?;

    for record in records {
        let kind = match record.kind.as_str() {
            "source" => SymbolKind::Source,
            "repair" => SymbolKind::Repair,
            other => return Err(format!("invalid symbol kind: {other}")),
        };

        let data = BASE64
            .decode(&record.data_b64)
            .map_err(|err| format!("base64 decode failed: {err}"))?;
        let actual_hash = sha256_hex(&data);
        if actual_hash != record.data_sha256 {
            return Err(format!(
                "symbol hash mismatch sbn={} esi={} kind={} expected={} actual={}",
                record.sbn, record.esi, record.kind, record.data_sha256, actual_hash
            ));
        }

        let symbol = Symbol::new(SymbolId::new(object_id, record.sbn, record.esi), data, kind);
        let auth = AuthenticatedSymbol::new_verified(symbol, AuthenticationTag::zero());
        let _ = decoder
            .feed(auth)
            .map_err(|err| format!("decoder feed failed: {err}"))?;
    }

    decoder
        .into_data()
        .map_err(|err| format!("decoder finalize failed: {err}"))
}

pub fn scrub_and_write_reports(
    sidecar_path: &Path,
    scrub_report_path: &Path,
    decode_proof_path: &Path,
) -> Result<(ScrubReport, DecodeProofArtifact), String> {
    scrub_and_write_reports_with_config(
        sidecar_path,
        scrub_report_path,
        decode_proof_path,
        RaptorQParallelismConfig::serial(),
    )
}

pub fn scrub_and_write_reports_with_config(
    sidecar_path: &Path,
    scrub_report_path: &Path,
    decode_proof_path: &Path,
    parallelism: RaptorQParallelismConfig,
) -> Result<(ScrubReport, DecodeProofArtifact), String> {
    let raw = fs::read_to_string(sidecar_path)
        .map_err(|err| format!("failed reading {}: {err}", sidecar_path.display()))?;
    let sidecar: RaptorQSidecar = serde_json::from_str(&raw)
        .map_err(|err| format!("invalid sidecar {}: {err}", sidecar_path.display()))?;

    let decoded = decode_payload_from_records(&sidecar, &sidecar.symbols)?;
    let decoded_hash = sha256_hex(&decoded);
    let full_match = decoded_hash == sidecar.source_hash;

    let mut candidate_indexes: Vec<usize> = sidecar
        .symbols
        .iter()
        .enumerate()
        .filter_map(|(idx, record)| (record.kind == "source").then_some(idx))
        .collect();
    if candidate_indexes.is_empty() && !sidecar.symbols.is_empty() {
        candidate_indexes.push(0);
    }

    let mut selected_drop: Option<(usize, Vec<RaptorQSymbolRecord>)> = None;
    let mut selected_recovery: Option<(bool, Option<String>, Option<String>)> = None;

    for idx in candidate_indexes {
        let mut records = sidecar.symbols.clone();
        let _removed = records.remove(idx);
        let recovery = decode_payload_from_records(&sidecar, &records);
        let result = match recovery {
            Ok(bytes) => {
                let hash = sha256_hex(&bytes);
                (hash == sidecar.source_hash, Some(hash), None)
            }
            Err(err) => (false, None, Some(err)),
        };

        if selected_drop.is_none() {
            selected_drop = Some((idx, records.clone()));
            selected_recovery = Some(result.clone());
        }

        if result.0 {
            selected_drop = Some((idx, records));
            selected_recovery = Some(result);
            break;
        }
    }

    let (drop_index, recovery_records, (recovery_success, recovered_hash, recovery_error)) =
        match (selected_drop, selected_recovery) {
            (Some((idx, records)), Some(recovery)) => (Some(idx), records, recovery),
            _ => (None, sidecar.symbols.clone(), (false, None, None)),
        };

    let dropped_symbol = drop_index.map(|idx| {
        let rec = &sidecar.symbols[idx];
        format!("sbn={} esi={} kind={}", rec.sbn, rec.esi, rec.kind)
    });

    let status = if full_match && recovery_success {
        "ok".to_string()
    } else {
        "failed".to_string()
    };

    let scrub_report = ScrubReport {
        schema_version: 1,
        bundle_id: sidecar.bundle_id.clone(),
        generated_at_unix_ms: now_unix_ms(),
        status,
        expected_hash: sidecar.source_hash.clone(),
        decoded_hash,
        full_decode_match: full_match,
        recovery_decode_match: recovery_success,
        symbols_total: sidecar.total_symbols,
        symbols_used_full: sidecar.symbols.len(),
        symbols_used_recovery: recovery_records.len(),
        decoding_parallelism: parallelism.worker_count,
    };

    let decode_proof = DecodeProofArtifact {
        schema_version: 1,
        bundle_id: sidecar.bundle_id,
        generated_at_unix_ms: now_unix_ms(),
        dropped_symbol,
        recovery_symbols_used: recovery_records.len(),
        recovery_success,
        expected_hash: sidecar.source_hash,
        recovered_hash,
        error: recovery_error,
        decoding_parallelism: parallelism.worker_count,
    };

    if let Some(parent) = scrub_report_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }

    let scrub_raw = serde_json::to_string_pretty(&scrub_report)
        .map_err(|err| format!("failed serializing scrub report: {err}"))?;
    fs::write(scrub_report_path, scrub_raw)
        .map_err(|err| format!("failed writing {}: {err}", scrub_report_path.display()))?;

    let proof_raw = serde_json::to_string_pretty(&decode_proof)
        .map_err(|err| format!("failed serializing decode proof: {err}"))?;
    fs::write(decode_proof_path, proof_raw)
        .map_err(|err| format!("failed writing {}: {err}", decode_proof_path.display()))?;

    Ok((scrub_report, decode_proof))
}

pub fn generate_bundle_sidecar_and_reports(
    bundle_id: &str,
    repo_root: &Path,
    files: &[PathBuf],
    sidecar_path: &Path,
    scrub_report_path: &Path,
    decode_proof_path: &Path,
    object_seed: u64,
) -> Result<(), String> {
    generate_bundle_sidecar_and_reports_with_config(
        bundle_id,
        repo_root,
        files,
        RaptorQBundleReportPaths::new(sidecar_path, scrub_report_path, decode_proof_path),
        object_seed,
        RaptorQParallelismConfig::serial(),
    )
}

pub fn generate_bundle_sidecar_and_reports_with_config(
    bundle_id: &str,
    repo_root: &Path,
    files: &[PathBuf],
    report_paths: RaptorQBundleReportPaths<'_>,
    object_seed: u64,
    parallelism: RaptorQParallelismConfig,
) -> Result<(), String> {
    let payload = build_bundle_payload(bundle_id, repo_root, files)?;
    let _sidecar = generate_sidecar_from_payload_with_config(
        bundle_id,
        &payload,
        report_paths.sidecar_path,
        object_seed,
        parallelism,
    )?;
    let _ = scrub_and_write_reports_with_config(
        report_paths.sidecar_path,
        report_paths.scrub_report_path,
        report_paths.decode_proof_path,
        parallelism,
    )?;
    Ok(())
}

#[derive(Debug, Clone)]
struct BundleArtifactSpec {
    bundle_id: &'static str,
    source_files: Vec<PathBuf>,
    sidecar_path: PathBuf,
    scrub_report_path: PathBuf,
    decode_proof_path: PathBuf,
}

fn default_bundle_specs(repo_root: &Path) -> Vec<BundleArtifactSpec> {
    let fixture_root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fnp-conformance/fixtures");

    vec![
        BundleArtifactSpec {
            bundle_id: "conformance_bundle_v1",
            source_files: vec![
                fixture_root.join("ufunc_input_cases.json"),
                fixture_root.join("workflow_scenario_corpus.json"),
                fixture_root.join("oracle_outputs/ufunc_oracle_output.json"),
                fixture_root.join("oracle_outputs/ufunc_differential_report.json"),
            ],
            sidecar_path: repo_root.join("artifacts/raptorq/conformance_bundle_v1.sidecar.json"),
            scrub_report_path: repo_root
                .join("artifacts/raptorq/conformance_bundle_v1.scrub_report.json"),
            decode_proof_path: repo_root
                .join("artifacts/raptorq/conformance_bundle_v1.decode_proof.json"),
        },
        BundleArtifactSpec {
            bundle_id: "benchmark_bundle_v1",
            source_files: vec![repo_root.join("artifacts/baselines/ufunc_benchmark_baseline.json")],
            sidecar_path: repo_root.join("artifacts/raptorq/benchmark_bundle_v1.sidecar.json"),
            scrub_report_path: repo_root
                .join("artifacts/raptorq/benchmark_bundle_v1.scrub_report.json"),
            decode_proof_path: repo_root
                .join("artifacts/raptorq/benchmark_bundle_v1.decode_proof.json"),
        },
    ]
}

pub fn generate_default_bundle_sidecars_and_reports(
    repo_root: &Path,
    parallelism: RaptorQParallelismConfig,
) -> Result<(), String> {
    let specs = default_bundle_specs(repo_root)
        .into_iter()
        .enumerate()
        .collect::<Vec<_>>();
    if parallelism.worker_count == 1 || specs.len() <= 1 {
        for (idx, spec) in &specs {
            generate_bundle_sidecar_and_reports_with_config(
                spec.bundle_id,
                repo_root,
                &spec.source_files,
                RaptorQBundleReportPaths::new(
                    &spec.sidecar_path,
                    &spec.scrub_report_path,
                    &spec.decode_proof_path,
                ),
                1001 + *idx as u64,
                parallelism,
            )?;
        }
        return Ok(());
    }

    let workers = parallelism.worker_count.min(specs.len());
    let mut pending = specs.into_iter();
    loop {
        let batch = pending.by_ref().take(workers).collect::<Vec<_>>();
        if batch.is_empty() {
            break;
        }

        thread::scope(|scope| {
            let handles = batch
                .into_iter()
                .map(|(idx, spec)| {
                    scope.spawn(move || {
                        generate_bundle_sidecar_and_reports_with_config(
                            spec.bundle_id,
                            repo_root,
                            &spec.source_files,
                            RaptorQBundleReportPaths::new(
                                &spec.sidecar_path,
                                &spec.scrub_report_path,
                                &spec.decode_proof_path,
                            ),
                            1001 + idx as u64,
                            parallelism,
                        )
                    })
                })
                .collect::<Vec<_>>();

            for handle in handles {
                handle
                    .join()
                    .map_err(|_| "raptorq worker thread panicked".to_string())??;
            }

            Ok::<(), String>(())
        })?;
    }

    Ok(())
}

fn deterministic_stress_bytes(size: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(size);
    for idx in 0..size {
        let word = (idx as u64)
            .wrapping_mul(0x9E37_79B1)
            .wrapping_add((idx as u64) >> 7)
            .wrapping_add(0xA5);
        bytes.push((word & 0xFF) as u8);
    }
    bytes
}

fn write_stress_bundle_inputs(config: &RaptorQStressGateConfig) -> Result<Vec<PathBuf>, String> {
    if config.source_bytes == 0 {
        return Err("raptorq stress source bytes must be greater than zero".to_string());
    }

    let input_dir = config.output_dir.join("input");
    fs::create_dir_all(&input_dir)
        .map_err(|err| format!("failed creating {}: {err}", input_dir.display()))?;

    let payload_path = input_dir.join("deterministic_payload.bin");
    fs::write(
        &payload_path,
        deterministic_stress_bytes(config.source_bytes),
    )
    .map_err(|err| format!("failed writing {}: {err}", payload_path.display()))?;

    let manifest_path = input_dir.join("manifest.json");
    let manifest = serde_json::json!({
        "schema_version": 1,
        "mode": config.mode.as_str(),
        "source_bytes": config.source_bytes,
        "worker_count": config.parallelism.worker_count,
        "payload_path": "deterministic_payload.bin",
    });
    let manifest_raw = serde_json::to_string_pretty(&manifest)
        .map_err(|err| format!("failed serializing stress manifest: {err}"))?;
    fs::write(&manifest_path, manifest_raw)
        .map_err(|err| format!("failed writing {}: {err}", manifest_path.display()))?;

    Ok(vec![manifest_path, payload_path])
}

pub fn run_raptorq_stress_gate(
    config: &RaptorQStressGateConfig,
) -> Result<RaptorQStressReport, String> {
    let start = Instant::now();
    let bundle_id = format!("raptorq_stress_{}", config.mode.as_str());
    fs::create_dir_all(&config.output_dir)
        .map_err(|err| format!("failed creating {}: {err}", config.output_dir.display()))?;

    let input_files = write_stress_bundle_inputs(config)?;
    let sidecar_path = config.output_dir.join("stress_bundle.sidecar.json");
    let scrub_report_path = config.output_dir.join("stress_bundle.scrub_report.json");
    let decode_proof_path = config.output_dir.join("stress_bundle.decode_proof.json");

    let payload = build_bundle_payload(&bundle_id, &config.repo_root, &input_files)?;
    let input_hash = sha256_hex(&payload);
    let sidecar = generate_sidecar_from_payload_with_config(
        &bundle_id,
        &payload,
        &sidecar_path,
        0x5EED_8A77,
        config.parallelism,
    )?;
    let (scrub, proof) = scrub_and_write_reports_with_config(
        &sidecar_path,
        &scrub_report_path,
        &decode_proof_path,
        config.parallelism,
    )?;

    let recovered_hash = proof.recovered_hash.clone().unwrap_or_default();
    let mut diagnostics = Vec::new();
    if sidecar.source_hash != input_hash {
        diagnostics
            .push("sidecar source hash did not match deterministic bundle input".to_string());
    }
    if scrub.status != "ok" || !scrub.full_decode_match || !scrub.recovery_decode_match {
        diagnostics.push("scrub report did not prove full and recovery decode match".to_string());
    }
    if !proof.recovery_success {
        diagnostics.push("decode proof did not recover after bounded symbol loss".to_string());
    }
    if proof.expected_hash != input_hash
        || proof.recovered_hash.as_deref() != Some(input_hash.as_str())
    {
        diagnostics.push("decode proof hashes did not match deterministic input hash".to_string());
    }

    let status = if diagnostics.is_empty() {
        "pass"
    } else {
        "fail"
    }
    .to_string();
    let report = RaptorQStressReport {
        schema_version: RAPTORQ_STRESS_REPORT_SCHEMA_VERSION,
        bundle_id,
        mode: config.mode,
        status,
        worker_count: config.parallelism.worker_count,
        output_dir: config.output_dir.display().to_string(),
        input_files: input_files
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        sidecar_path: sidecar_path.display().to_string(),
        scrub_report_path: scrub_report_path.display().to_string(),
        decode_proof_path: decode_proof_path.display().to_string(),
        input_hash,
        recovered_hash,
        elapsed_ms: start.elapsed().as_millis(),
        source_size: sidecar.source_size,
        source_symbols: sidecar.source_symbols,
        repair_symbols: sidecar.repair_symbols,
        total_symbols: sidecar.total_symbols,
        dropped_symbol_scenario: proof.dropped_symbol,
        recovery_symbols_used: proof.recovery_symbols_used,
        replay_command: config.replay_command.clone(),
        diagnostics,
    };

    validate_raptorq_stress_report(&report)?;
    Ok(report)
}

pub fn validate_raptorq_stress_report(report: &RaptorQStressReport) -> Result<(), String> {
    if report.schema_version != RAPTORQ_STRESS_REPORT_SCHEMA_VERSION {
        return Err(format!(
            "raptorq stress report schema_version must be {}, got {}",
            RAPTORQ_STRESS_REPORT_SCHEMA_VERSION, report.schema_version
        ));
    }
    if report.status != "pass" {
        return Err(format!(
            "raptorq stress report status must be pass, got {}: {:?}",
            report.status, report.diagnostics
        ));
    }
    RaptorQParallelismConfig::from_worker_count(report.worker_count)?;
    if report.input_files.is_empty() {
        return Err("raptorq stress report must list input files".to_string());
    }
    for path in &report.input_files {
        if !Path::new(path).is_file() {
            return Err(format!("raptorq stress input file missing: {path}"));
        }
    }
    for path in [
        &report.sidecar_path,
        &report.scrub_report_path,
        &report.decode_proof_path,
    ] {
        if !Path::new(path).is_file() {
            return Err(format!("raptorq stress artifact missing: {path}"));
        }
    }
    if report.input_hash.len() != 64
        || !report
            .input_hash
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("raptorq stress report input_hash must be a sha256 hex digest".to_string());
    }
    if report.recovered_hash != report.input_hash {
        return Err(format!(
            "raptorq stress recovered hash mismatch expected={} actual={}",
            report.input_hash, report.recovered_hash
        ));
    }
    if report.source_size == 0 {
        return Err("raptorq stress source_size must be non-zero".to_string());
    }
    if report.total_symbols != usize::from(report.source_symbols) + report.repair_symbols {
        return Err(format!(
            "raptorq stress symbol counts mismatch total={} source={} repair={}",
            report.total_symbols, report.source_symbols, report.repair_symbols
        ));
    }
    if report.dropped_symbol_scenario.is_none() {
        return Err("raptorq stress report must record a dropped-symbol scenario".to_string());
    }
    if report.recovery_symbols_used == 0 {
        return Err("raptorq stress report must record recovery symbol usage".to_string());
    }
    if !report.replay_command.contains("run_raptorq_gate") {
        return Err(
            "raptorq stress report replay command must reference run_raptorq_gate".to_string(),
        );
    }
    Ok(())
}

fn modified_unix_ms(path: &Path) -> Result<u128, String> {
    let metadata =
        fs::metadata(path).map_err(|err| format!("failed stat {}: {err}", path.display()))?;
    let modified = metadata
        .modified()
        .map_err(|err| format!("failed reading mtime {}: {err}", path.display()))?;
    let millis = modified
        .duration_since(UNIX_EPOCH)
        .map_err(|err| format!("mtime before unix epoch {}: {err}", path.display()))?
        .as_millis();
    Ok(millis)
}

fn latest_mtime(paths: &[PathBuf]) -> Result<u128, String> {
    let mut latest = 0u128;
    for path in paths {
        let ts = modified_unix_ms(path)?;
        latest = latest.max(ts);
    }
    Ok(latest)
}

fn record_check(report: &mut SuiteReport, passed: bool, failure: String) {
    report.case_count += 1;
    if passed {
        report.pass_count += 1;
    } else {
        report.failures.push(failure);
    }
}

pub fn run_raptorq_artifact_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    run_raptorq_artifact_suite_with_parallelism(config, None)
}

pub fn run_raptorq_artifact_suite_with_parallelism(
    config: &HarnessConfig,
    expected_parallelism: Option<RaptorQParallelismConfig>,
) -> Result<SuiteReport, String> {
    let repo_root = config
        .contract_root
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| {
            format!(
                "unable to derive repo root from contract_root {}",
                config.contract_root.display()
            )
        })?;
    let specs = default_bundle_specs(&repo_root);

    let mut report = SuiteReport {
        suite: "raptorq_artifacts",
        case_count: 0,
        pass_count: 0,
        failures: Vec::new(),
    };

    for spec in &specs {
        for source in &spec.source_files {
            record_check(
                &mut report,
                source.is_file(),
                format!(
                    "{}: source file missing {}",
                    spec.bundle_id,
                    source.display()
                ),
            );
        }

        let sidecar_exists = spec.sidecar_path.is_file();
        record_check(
            &mut report,
            sidecar_exists,
            format!(
                "{}: sidecar missing {}",
                spec.bundle_id,
                spec.sidecar_path.display()
            ),
        );

        let scrub_exists = spec.scrub_report_path.is_file();
        record_check(
            &mut report,
            scrub_exists,
            format!(
                "{}: scrub report missing {}",
                spec.bundle_id,
                spec.scrub_report_path.display()
            ),
        );

        let proof_exists = spec.decode_proof_path.is_file();
        record_check(
            &mut report,
            proof_exists,
            format!(
                "{}: decode proof missing {}",
                spec.bundle_id,
                spec.decode_proof_path.display()
            ),
        );

        if !sidecar_exists || !scrub_exists || !proof_exists {
            continue;
        }

        let sidecar_raw = match fs::read_to_string(&spec.sidecar_path) {
            Ok(raw) => raw,
            Err(err) => {
                record_check(
                    &mut report,
                    false,
                    format!(
                        "{}: failed reading sidecar {}: {err}",
                        spec.bundle_id,
                        spec.sidecar_path.display()
                    ),
                );
                continue;
            }
        };
        let sidecar: RaptorQSidecar = match serde_json::from_str(&sidecar_raw) {
            Ok(parsed) => parsed,
            Err(err) => {
                record_check(
                    &mut report,
                    false,
                    format!(
                        "{}: invalid sidecar json {}: {err}",
                        spec.bundle_id,
                        spec.sidecar_path.display()
                    ),
                );
                continue;
            }
        };

        let scrub_raw = match fs::read_to_string(&spec.scrub_report_path) {
            Ok(raw) => raw,
            Err(err) => {
                record_check(
                    &mut report,
                    false,
                    format!(
                        "{}: failed reading scrub report {}: {err}",
                        spec.bundle_id,
                        spec.scrub_report_path.display()
                    ),
                );
                continue;
            }
        };
        let scrub: ScrubReport = match serde_json::from_str(&scrub_raw) {
            Ok(parsed) => parsed,
            Err(err) => {
                record_check(
                    &mut report,
                    false,
                    format!(
                        "{}: invalid scrub json {}: {err}",
                        spec.bundle_id,
                        spec.scrub_report_path.display()
                    ),
                );
                continue;
            }
        };

        let proof_raw = match fs::read_to_string(&spec.decode_proof_path) {
            Ok(raw) => raw,
            Err(err) => {
                record_check(
                    &mut report,
                    false,
                    format!(
                        "{}: failed reading decode proof {}: {err}",
                        spec.bundle_id,
                        spec.decode_proof_path.display()
                    ),
                );
                continue;
            }
        };
        let proof: DecodeProofArtifact = match serde_json::from_str(&proof_raw) {
            Ok(parsed) => parsed,
            Err(err) => {
                record_check(
                    &mut report,
                    false,
                    format!(
                        "{}: invalid decode proof json {}: {err}",
                        spec.bundle_id,
                        spec.decode_proof_path.display()
                    ),
                );
                continue;
            }
        };

        let payload = match build_bundle_payload(spec.bundle_id, &repo_root, &spec.source_files) {
            Ok(bytes) => bytes,
            Err(err) => {
                record_check(
                    &mut report,
                    false,
                    format!(
                        "{}: failed rebuilding bundle payload: {err}",
                        spec.bundle_id
                    ),
                );
                continue;
            }
        };
        let expected_hash = sha256_hex(&payload);

        record_check(
            &mut report,
            sidecar.schema_version == 1,
            format!("{}: sidecar schema_version must be 1", spec.bundle_id),
        );
        record_check(
            &mut report,
            scrub.schema_version == 1,
            format!("{}: scrub schema_version must be 1", spec.bundle_id),
        );
        record_check(
            &mut report,
            proof.schema_version == 1,
            format!("{}: decode proof schema_version must be 1", spec.bundle_id),
        );
        record_check(
            &mut report,
            RaptorQParallelismConfig::from_worker_count(sidecar.encoding_parallelism).is_ok(),
            format!(
                "{}: sidecar encoding_parallelism out of bounds: {}",
                spec.bundle_id, sidecar.encoding_parallelism
            ),
        );
        record_check(
            &mut report,
            RaptorQParallelismConfig::from_worker_count(sidecar.decoding_parallelism).is_ok(),
            format!(
                "{}: sidecar decoding_parallelism out of bounds: {}",
                spec.bundle_id, sidecar.decoding_parallelism
            ),
        );
        record_check(
            &mut report,
            sidecar.decoding_parallelism == scrub.decoding_parallelism
                && scrub.decoding_parallelism == proof.decoding_parallelism,
            format!(
                "{}: decoding parallelism metadata mismatch sidecar={} scrub={} proof={}",
                spec.bundle_id,
                sidecar.decoding_parallelism,
                scrub.decoding_parallelism,
                proof.decoding_parallelism
            ),
        );
        if let Some(expected) = expected_parallelism {
            record_check(
                &mut report,
                sidecar.encoding_parallelism == expected.worker_count
                    && sidecar.decoding_parallelism == expected.worker_count,
                format!(
                    "{}: sidecar parallelism metadata does not match expected {}",
                    spec.bundle_id, expected.worker_count
                ),
            );
        }
        record_check(
            &mut report,
            sidecar.bundle_id == spec.bundle_id,
            format!(
                "{}: sidecar bundle_id mismatch actual={}",
                spec.bundle_id, sidecar.bundle_id
            ),
        );
        record_check(
            &mut report,
            scrub.bundle_id == spec.bundle_id,
            format!(
                "{}: scrub bundle_id mismatch actual={}",
                spec.bundle_id, scrub.bundle_id
            ),
        );
        record_check(
            &mut report,
            proof.bundle_id == spec.bundle_id,
            format!(
                "{}: decode proof bundle_id mismatch actual={}",
                spec.bundle_id, proof.bundle_id
            ),
        );
        record_check(
            &mut report,
            sidecar.total_symbols == sidecar.symbols.len(),
            format!(
                "{}: total_symbols mismatch declared={} actual={}",
                spec.bundle_id,
                sidecar.total_symbols,
                sidecar.symbols.len()
            ),
        );
        let source_hash_matches = sidecar.source_hash == expected_hash;
        record_check(
            &mut report,
            source_hash_matches,
            format!(
                "{}: sidecar source_hash does not match source payload",
                spec.bundle_id
            ),
        );
        record_check(
            &mut report,
            sidecar.source_size == payload.len(),
            format!(
                "{}: sidecar source_size mismatch declared={} actual={}",
                spec.bundle_id,
                sidecar.source_size,
                payload.len()
            ),
        );

        let latest_source_mtime = latest_mtime(&spec.source_files).unwrap_or(0);
        let sidecar_mtime = modified_unix_ms(&spec.sidecar_path).unwrap_or(0);
        let scrub_mtime = modified_unix_ms(&spec.scrub_report_path).unwrap_or(0);
        let proof_mtime = modified_unix_ms(&spec.decode_proof_path).unwrap_or(0);
        let stale = latest_source_mtime > sidecar_mtime
            || latest_source_mtime > scrub_mtime
            || latest_source_mtime > proof_mtime;

        record_check(
            &mut report,
            !stale || source_hash_matches,
            format!(
                "{}: stale durability artifacts latest_source_mtime={} sidecar_mtime={} scrub_mtime={} proof_mtime={}",
                spec.bundle_id, latest_source_mtime, sidecar_mtime, scrub_mtime, proof_mtime
            ),
        );
        record_check(
            &mut report,
            (sidecar.generated_at_unix_ms >= latest_source_mtime
                && scrub.generated_at_unix_ms >= latest_source_mtime
                && proof.generated_at_unix_ms >= latest_source_mtime)
                || source_hash_matches,
            format!(
                "{}: generated_at_unix_ms indicates stale artifacts relative to source mtimes",
                spec.bundle_id
            ),
        );

        record_check(
            &mut report,
            scrub.status == "ok",
            format!(
                "{}: scrub status must be ok, got {}",
                spec.bundle_id, scrub.status
            ),
        );
        record_check(
            &mut report,
            scrub.full_decode_match && scrub.recovery_decode_match,
            format!(
                "{}: scrub decode match flags must both be true",
                spec.bundle_id
            ),
        );
        record_check(
            &mut report,
            scrub.expected_hash == expected_hash && scrub.decoded_hash == expected_hash,
            format!(
                "{}: scrub expected/decoded hashes must match bundle source hash",
                spec.bundle_id
            ),
        );
        record_check(
            &mut report,
            proof.recovery_success,
            format!(
                "{}: decode proof recovery_success must be true",
                spec.bundle_id
            ),
        );
        record_check(
            &mut report,
            proof.expected_hash == expected_hash
                && proof.recovered_hash.as_deref() == Some(expected_hash.as_str()),
            format!(
                "{}: decode proof expected/recovered hashes must match source hash",
                spec.bundle_id
            ),
        );
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::{
        DecodeProofArtifact, MAX_RAPTORQ_PARALLELISM, RaptorQBundleReportPaths,
        RaptorQParallelismConfig, RaptorQSidecar, RaptorQStressGateConfig, RaptorQStressMode,
        RaptorQStressReport, ScrubReport, generate_bundle_sidecar_and_reports,
        generate_bundle_sidecar_and_reports_with_config, generate_sidecar_from_payload,
        generate_sidecar_from_payload_with_config, run_raptorq_stress_gate,
        scrub_and_write_reports, validate_raptorq_stress_report,
    };
    use base64::Engine as _;
    use std::fs;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        std::env::temp_dir().join(format!("fnp_{name}_{ts}"))
    }

    #[test]
    fn parallelism_config_rejects_invalid_worker_counts() {
        assert!(RaptorQParallelismConfig::from_worker_count(0).is_err());
        assert!(RaptorQParallelismConfig::from_worker_count(MAX_RAPTORQ_PARALLELISM + 1).is_err());
        assert_eq!(
            RaptorQParallelismConfig::from_worker_count(MAX_RAPTORQ_PARALLELISM)
                .expect("max worker count should be accepted")
                .worker_count,
            MAX_RAPTORQ_PARALLELISM
        );
    }

    #[test]
    fn sidecar_roundtrip_scrub_is_ok() {
        let repo_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let file_a = temp_path("bundle_a.txt");
        let file_b = temp_path("bundle_b.txt");

        fs::write(&file_a, "alpha").expect("write file_a");
        fs::write(&file_b, "beta").expect("write file_b");

        let sidecar = temp_path("bundle_sidecar.json");
        let scrub = temp_path("bundle_scrub.json");
        let proof = temp_path("bundle_proof.json");

        generate_bundle_sidecar_and_reports(
            "test_bundle",
            &repo_root,
            &[file_a.clone(), file_b.clone()],
            &sidecar,
            &scrub,
            &proof,
            42,
        )
        .expect("sidecar generation should succeed");

        let scrub_raw = fs::read_to_string(&scrub).expect("read scrub");
        assert!(scrub_raw.contains("\"status\": \"ok\""));

        let _ = fs::remove_file(file_a);
        let _ = fs::remove_file(file_b);
        let _ = fs::remove_file(sidecar);
        let _ = fs::remove_file(scrub);
        let _ = fs::remove_file(proof);
    }

    #[test]
    fn sidecar_records_are_stable_across_worker_counts() {
        let payload = b"parallelism-stability-payload-".repeat(128);
        let serial_path = temp_path("serial_sidecar.json");
        let parallel_path = temp_path("parallel_sidecar.json");
        let parallelism =
            RaptorQParallelismConfig::from_worker_count(4).expect("parallelism config");

        let serial = generate_sidecar_from_payload_with_config(
            "stable_bundle",
            &payload,
            &serial_path,
            77,
            RaptorQParallelismConfig::serial(),
        )
        .expect("serial sidecar generation should succeed");
        let parallel = generate_sidecar_from_payload_with_config(
            "stable_bundle",
            &payload,
            &parallel_path,
            77,
            parallelism,
        )
        .expect("parallel sidecar generation should succeed");

        assert_eq!(serial.source_hash, parallel.source_hash);
        assert_eq!(serial.source_size, parallel.source_size);
        assert_eq!(serial.source_blocks, parallel.source_blocks);
        assert_eq!(serial.source_symbols, parallel.source_symbols);
        assert_eq!(serial.repair_symbols, parallel.repair_symbols);
        assert_eq!(serial.total_symbols, parallel.total_symbols);
        assert_eq!(serial.symbols.len(), parallel.symbols.len());
        assert_eq!(
            serial
                .symbols
                .iter()
                .map(|record| (
                    record.sbn,
                    record.esi,
                    record.kind.as_str(),
                    record.data_sha256.as_str()
                ))
                .collect::<Vec<_>>(),
            parallel
                .symbols
                .iter()
                .map(|record| (
                    record.sbn,
                    record.esi,
                    record.kind.as_str(),
                    record.data_sha256.as_str()
                ))
                .collect::<Vec<_>>()
        );
        assert_eq!(serial.encoding_parallelism, 1);
        assert_eq!(serial.decoding_parallelism, 1);
        assert_eq!(parallel.encoding_parallelism, 4);
        assert_eq!(parallel.decoding_parallelism, 4);
    }

    #[test]
    fn scrub_reports_record_configured_decode_parallelism() {
        let repo_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let file = temp_path("parallel_report_bundle.txt");
        fs::write(&file, "parallel-report").expect("write fixture");

        let sidecar = temp_path("parallel_report_sidecar.json");
        let scrub = temp_path("parallel_report_scrub.json");
        let proof = temp_path("parallel_report_proof.json");
        let parallelism =
            RaptorQParallelismConfig::from_worker_count(3).expect("parallelism config");

        generate_bundle_sidecar_and_reports_with_config(
            "parallel_report_bundle",
            &repo_root,
            &[file],
            RaptorQBundleReportPaths::new(&sidecar, &scrub, &proof),
            123,
            parallelism,
        )
        .expect("parallel sidecar generation should succeed");

        let scrub_raw = fs::read_to_string(&scrub).expect("read scrub");
        let proof_raw = fs::read_to_string(&proof).expect("read proof");
        let scrub: ScrubReport = serde_json::from_str(&scrub_raw).expect("parse scrub");
        let proof: DecodeProofArtifact = serde_json::from_str(&proof_raw).expect("parse proof");

        assert_eq!(scrub.status, "ok");
        assert_eq!(scrub.decoding_parallelism, 3);
        assert_eq!(proof.decoding_parallelism, 3);
        assert!(proof.recovery_success);
    }

    #[test]
    fn tampered_sidecar_fails_scrub() {
        let payload = b"tamper-me-payload";
        let sidecar_path = temp_path("tamper_sidecar.json");
        let scrub_path = temp_path("tamper_scrub.json");
        let proof_path = temp_path("tamper_proof.json");

        let _ = generate_sidecar_from_payload("tamper_bundle", payload, &sidecar_path, 99)
            .expect("sidecar create");

        let raw = fs::read_to_string(&sidecar_path).expect("read sidecar");
        let mut parsed: RaptorQSidecar = serde_json::from_str(&raw).expect("parse sidecar");

        if let Some(first) = parsed.symbols.first_mut() {
            let mut bytes = base64::engine::general_purpose::STANDARD
                .decode(&first.data_b64)
                .expect("decode symbol bytes");
            if let Some(byte) = bytes.first_mut() {
                *byte ^= 0xFF;
            }
            first.data_b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
        }

        let tampered_raw =
            serde_json::to_string_pretty(&parsed).expect("serialize tampered sidecar");
        fs::write(&sidecar_path, tampered_raw).expect("write tampered sidecar");

        let err = scrub_and_write_reports(&sidecar_path, &scrub_path, &proof_path)
            .expect_err("tampered symbol payload should fail closed");
        assert!(err.contains("symbol hash mismatch"));

        let _ = fs::remove_file(sidecar_path);
        let _ = fs::remove_file(scrub_path);
        let _ = fs::remove_file(proof_path);
    }

    fn small_stress_report(name: &str) -> RaptorQStressReport {
        let repo_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let output_dir = temp_path(name);
        run_raptorq_stress_gate(&RaptorQStressGateConfig {
            repo_root,
            output_dir,
            mode: RaptorQStressMode::Smoke,
            parallelism: RaptorQParallelismConfig::serial(),
            source_bytes: 2048,
            replay_command:
                "cargo run -p fnp-conformance --bin run_raptorq_gate -- --stress-mode smoke"
                    .to_string(),
        })
        .expect("stress report should pass")
    }

    #[test]
    fn run_raptorq_gate_stress_report_records_required_fields() {
        let report = small_stress_report("stress_required_fields");

        assert_eq!(report.status, "pass");
        assert_eq!(report.mode, RaptorQStressMode::Smoke);
        assert_eq!(report.worker_count, 1);
        assert_eq!(report.input_hash, report.recovered_hash);
        assert_eq!(
            report.total_symbols,
            usize::from(report.source_symbols) + report.repair_symbols
        );
        assert!(report.dropped_symbol_scenario.is_some());
        assert!(report.recovery_symbols_used > 0);
        assert!(report.replay_command.contains("run_raptorq_gate"));
        validate_raptorq_stress_report(&report).expect("stress report validates");
    }

    #[test]
    fn run_raptorq_gate_stress_report_validation_fails_closed_on_hash_mismatch() {
        let mut report = small_stress_report("stress_hash_mismatch");
        report.recovered_hash = "0".repeat(64);

        let err =
            validate_raptorq_stress_report(&report).expect_err("hash mismatch should fail closed");
        assert!(err.contains("recovered hash mismatch"));
    }

    #[test]
    fn run_raptorq_gate_stress_report_validation_fails_closed_on_missing_decode_proof() {
        let mut report = small_stress_report("stress_missing_proof");
        report.decode_proof_path = temp_path("missing_decode_proof").display().to_string();

        let err = validate_raptorq_stress_report(&report)
            .expect_err("missing decode proof should fail closed");
        assert!(err.contains("raptorq stress artifact missing"));
    }
}
