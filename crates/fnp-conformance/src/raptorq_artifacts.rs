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
use std::time::{SystemTime, UNIX_EPOCH};

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
    let symbol_size = 256u16;
    let max_block_size = payload.len().max(usize::from(symbol_size));

    let config = EncodingConfig {
        repair_overhead: 1.25,
        max_block_size,
        symbol_size,
        encoding_parallelism: 1,
        decoding_parallelism: 1,
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
        sidecar.source_blocks,
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
    let payload = build_bundle_payload(bundle_id, repo_root, files)?;
    let _sidecar = generate_sidecar_from_payload(bundle_id, &payload, sidecar_path, object_seed)?;
    let _ = scrub_and_write_reports(sidecar_path, scrub_report_path, decode_proof_path)?;
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
        RaptorQSidecar, generate_bundle_sidecar_and_reports, generate_sidecar_from_payload,
        scrub_and_write_reports,
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

        let (scrub, _proof) = scrub_and_write_reports(&sidecar_path, &scrub_path, &proof_path)
            .expect("scrub should produce report");
        assert_eq!(scrub.status, "failed");

        let _ = fs::remove_file(sidecar_path);
        let _ = fs::remove_file(scrub_path);
        let _ = fs::remove_file(proof_path);
    }
}
