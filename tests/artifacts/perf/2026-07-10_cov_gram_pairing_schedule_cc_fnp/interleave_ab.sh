#!/usr/bin/env bash
# Interleaved A/B: alternate base/candidate worker PROCESSES for N rounds so both
# sides sample the same background load. malloc knobs pin the allocation mode
# (fault-storm lottery, see ledger 2026-07-10 FINDING A); fault_probe.py records
# each binary's native mode alongside. Usage: interleave_ab.sh <rounds> <outfile>
set -u
ROUNDS=${1:-7}
OUT=${2:-/tmp/covbal_ab.txt}
A=tests/artifacts/perf/2026-07-10_cov_gram_pairing_schedule_cc_fnp
BASE=/data/projects/.scratch/fnp-covbal-probe/base
CAND=/data/projects/.scratch/fnp-covbal-probe/cand
export MALLOC_MMAP_THRESHOLD_=134217728 MALLOC_TRIM_THRESHOLD_=134217728
: > "$OUT"
echo "== native fault modes (no knobs) ==" >> "$OUT"
env -u MALLOC_MMAP_THRESHOLD_ -u MALLOC_TRIM_THRESHOLD_ python3 $A/fault_probe.py base "$BASE" >> "$OUT" 2>&1
env -u MALLOC_MMAP_THRESHOLD_ -u MALLOC_TRIM_THRESHOLD_ python3 $A/fault_probe.py cand "$CAND" >> "$OUT" 2>&1
for r in $(seq 1 "$ROUNDS"); do
  echo "== round $r base ==" >> "$OUT"
  python3 $A/cov_perf_worker.py fnp-base "$BASE" >> "$OUT" 2>&1
  echo "== round $r cand ==" >> "$OUT"
  python3 $A/cov_perf_worker.py fnp-cand "$CAND" >> "$OUT" 2>&1
  echo "== round $r numpy control ==" >> "$OUT"
  python3 $A/cov_perf_worker.py np >> "$OUT" 2>&1
done
echo done
