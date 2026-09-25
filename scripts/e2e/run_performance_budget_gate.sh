#!/usr/bin/env bash
# G7: same-job A/B performance budget.
#
# Builds generate_benchmark_baseline (release profile) at a REFERENCE commit and at the candidate
# (HEAD) on THIS host, runs the two binaries for FNP_PERF_AB_ROUNDS rounds in alternating order,
# and gates every budgeted workload on the per-round candidate/reference median ratio: a bootstrap
# CI plus each arm's own round-to-round A/A null (run_performance_budget_gate --ab-*). Both arms
# share the host, the toolchain, the profile and the job, so the ratio measures code.
#
# It used to compare artifacts/baselines/ufunc_benchmark_baseline.json - captured on a 128-core
# host in April - with a DEBUG build measured on the CI runner, gating on a p99 of 6-20 samples;
# its verdicts moved between runs on unchanged code (bead deadlock-audit-rc0923-epic-71qy3.28).
#
#   FNP_PERF_AB_REFERENCE   reference commit-ish (default HEAD^; CI passes a push's `before` or a
#                           PR's base; HEAD itself makes an A/A run of the gate)
#   FNP_PERF_AB_ROUNDS      rounds per arm (default 9)
#   FNP_PERF_MAX_MEDIAN_REGRESSION_RATIO  (default 0.07)   FNP_PERF_COVERAGE_FLOOR (default 1.0)
#   FNP_PERF_REPORT_DIR     per-round JSON, both binaries, their SHA-256 and report.json
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT="${FNP_PERF_REPORT_DIR:-$ROOT_DIR/artifacts/logs/perf_ab_$(date +%s)}"
ROUNDS="${FNP_PERF_AB_ROUNDS:-9}"
MAX_MEDIAN_REGRESSION_RATIO="${FNP_PERF_MAX_MEDIAN_REGRESSION_RATIO:-0.07}"
COVERAGE_FLOOR="${FNP_PERF_COVERAGE_FLOOR:-1.0}"
REFERENCE_SPEC="${FNP_PERF_AB_REFERENCE:-}"
# A push's `before` is all zeros for a new branch; any other unknown commit falls back too.
if [[ -z "$REFERENCE_SPEC" || "$REFERENCE_SPEC" =~ ^0+$ ]] \
  || ! git cat-file -e "${REFERENCE_SPEC}^{commit}" 2>/dev/null; then
  REFERENCE_SPEC="HEAD^"
fi
REFERENCE="$(git rev-parse "${REFERENCE_SPEC}^{commit}")"
CANDIDATE="$(git rev-parse HEAD)"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT_DIR/target}"
mkdir -p "$OUT"

echo "[performance-budget-gate] mode=same_job_ab reference=$REFERENCE candidate=$CANDIDATE rounds=$ROUNDS"
echo "[performance-budget-gate] max_median_regression_ratio=$MAX_MEDIAN_REGRESSION_RATIO coverage_floor=$COVERAGE_FLOOR out=$OUT"
echo "[performance-budget-gate] host=$(uname -n) nproc=$(nproc) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs) rustc=$(rustc --version)"

# Candidate first, copied out before the reference build reuses the same target directory (which
# keeps every third-party dependency warm for the second build).
cargo build --release -p fnp-conformance \
  --bin generate_benchmark_baseline --bin run_performance_budget_gate
cp "$TARGET_DIR/release/generate_benchmark_baseline" "$OUT/candidate_generate_benchmark_baseline"
cp "$TARGET_DIR/release/run_performance_budget_gate" "$OUT/run_performance_budget_gate"

if [[ ! -d "$OUT/reference-src" ]]; then
  git worktree add --detach "$OUT/reference-src" "$REFERENCE"
fi
(cd "$OUT/reference-src" && CARGO_TARGET_DIR="$TARGET_DIR" cargo build --release -p fnp-conformance \
  --bin generate_benchmark_baseline)
cp "$TARGET_DIR/release/generate_benchmark_baseline" "$OUT/reference_generate_benchmark_baseline"
(cd "$OUT" && sha256sum reference_generate_benchmark_baseline candidate_generate_benchmark_baseline \
  | tee binaries.sha256)

GATE_ARGS=()
for ((round = 0; round < ROUNDS; round++)); do
  if ((round % 2 == 0)); then arms=(reference candidate); else arms=(candidate reference); fi
  for arm in "${arms[@]}"; do
    "$OUT/${arm}_generate_benchmark_baseline" --output-path "$OUT/${arm}_round_${round}.json" >/dev/null
  done
  GATE_ARGS+=(--ab-reference-run "$OUT/reference_round_${round}.json")
  GATE_ARGS+=(--ab-candidate-run "$OUT/candidate_round_${round}.json")
  echo "[performance-budget-gate] round $((round + 1))/$ROUNDS order=${arms[*]}"
done

"$OUT/run_performance_budget_gate" "${GATE_ARGS[@]}" \
  --report-path "$OUT/report.json" \
  --max-median-regression-ratio "$MAX_MEDIAN_REGRESSION_RATIO" \
  --coverage-floor "$COVERAGE_FLOOR"

echo "[performance-budget-gate] completed"
