#!/bin/bash
set -euo pipefail
cd /data/projects/franken_numpy
# Run the ufunc differential first so the report file is current
# before computing the RaptorQ sidecar hash over bundle source files.
cargo run -p fnp-conformance --bin run_ufunc_differential
cargo run -p fnp-conformance --bin generate_raptorq_sidecars
# Copy to target dir so rch artifact retrieval picks them up
mkdir -p /data/tmp/cargo-target/raptorq_out
cp artifacts/raptorq/conformance_bundle_v1.sidecar.json /data/tmp/cargo-target/raptorq_out/
cp artifacts/raptorq/conformance_bundle_v1.scrub_report.json /data/tmp/cargo-target/raptorq_out/
cp artifacts/raptorq/conformance_bundle_v1.decode_proof.json /data/tmp/cargo-target/raptorq_out/
cp artifacts/raptorq/benchmark_bundle_v1.sidecar.json /data/tmp/cargo-target/raptorq_out/
cp artifacts/raptorq/benchmark_bundle_v1.scrub_report.json /data/tmp/cargo-target/raptorq_out/
cp artifacts/raptorq/benchmark_bundle_v1.decode_proof.json /data/tmp/cargo-target/raptorq_out/
# Also copy the differential report and mismatch report so rch retrieves them
cp crates/fnp-conformance/fixtures/oracle_outputs/ufunc_differential_report.json /data/tmp/cargo-target/raptorq_out/
cp crates/fnp-conformance/fixtures/oracle_outputs/ufunc_differential_mismatch_report.json /data/tmp/cargo-target/raptorq_out/ 2>/dev/null || true
echo "DONE: sidecars generated and copied to target"
