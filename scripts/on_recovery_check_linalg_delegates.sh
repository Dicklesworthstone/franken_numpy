#!/usr/bin/env bash
# verify_linalg_delegates_on_recovery.sh
#
# One-command verification for the 4 code-only "stale-cliff" 2-D linalg delegates
# that were committed during the 2026-06-21 disk-low/cargo-freeze (eigvalsh, eigh,
# cholesky, matrix_power -> delegate single 2-D to NumPy/LAPACK; batched native
# kept). Those commits are BUILD- and CONFORMANCE-UNVERIFIED (no cargo was allowed
# during the freeze; they were only manually syntax-reviewed clean). Run this once
# the build freeze lifts and disk is reclaimed.
#
# It does NOT decide thresholds — it just rebuilds, runs the linalg conformance
# suites, and re-measures the four delegates vs NumPy so a human/agent can confirm
# the expected ~parity (was 3-6x native loss) and that batched stays a win.
#
# Usage:
#   scripts/verify_linalg_delegates_on_recovery.sh
# Env (optional):
#   CARGO_TARGET_DIR  (default: /data/projects/.rch-targets/franken_numpy-cod-b)
#   PROBE_DIR         (default: .probe)
#   CARGO             (default: cargo) — the seam the negative test below drives.
set -uo pipefail
cd "$(dirname "$0")/.."

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/data/projects/.rch-targets/franken_numpy-cod-b}"
PROBE_DIR="${PROBE_DIR:-.probe}"
CARGO="${CARGO:-cargo}"
fail=0

# Run one cargo test invocation and classify it as PASSED / FAILED / NEVER RAN.
#
# The three states are distinct and this script used to collapse two of them.
# The old form was `if cargo test ... | grep -E "test result"; then :; else
# echo NO RESULT; fi`: a shard printing "test result: FAILED. 0 passed; 3 failed"
# MATCHES that grep, so the then-branch ran, nothing was printed, and neither
# branch ever set fail=1. All three linalg conformance suites could be red and
# this script still exited 0 looking like it had verified them — which matters
# because it is the designated verifier for four build- and conformance-
# unverified delegate commits. pipefail does not save it either: a completed
# test binary reporting FAILED exits non-zero, so under pipefail the else-branch
# fires and prints "NO RESULT" for what is really a test failure.
#
# A shard that emitted no `test result:` line at all did not run — a different
# problem needing a different fix — so it is reported as such, with the tail of
# its output, rather than silently.
run_test_shard() {
  local label="$1"
  shift
  local log results
  log="$(RCH_MIN_LOCAL_TIME_MS=999999999 "$@" 2>&1)"
  results="$(printf '%s\n' "$log" | grep -E '^test result:' || true)"

  if [ -z "$results" ]; then
    echo "  $label: NO RESULT — the shard never reported, so it did not run"
    printf '%s\n' "$log" | tail -20 | sed 's/^/    | /'
    fail=1
    return
  fi

  printf '%s\n' "$results" | sed "s/^/  $label: /"

  if printf '%s\n' "$results" | grep -q '^test result: FAILED'; then
    echo "  $label: FAILED"
    printf '%s\n' "$log" | grep -E '^(failures:|    [A-Za-z_][A-Za-z0-9_:]*)$' | head -40 |
      sed 's/^/    | /'
    fail=1
  fi
}

echo "== [1/4] build fnp-python cdylib (LOCAL — links local libpython) =="
# RCH_MIN_LOCAL_TIME_MS forces a local build; hz-remote builds give undefined-symbol
# ImportError because Py_TYPE is external against the worker's python headers.
if RCH_MIN_LOCAL_TIME_MS=999999999 "$CARGO" build -p fnp-python --release --lib 2>&1 | tail -3; then
  if cp "$CARGO_TARGET_DIR/release/libfnp_python.so" "$PROBE_DIR/fnp_python.so"; then
    echo "  deployed -> $PROBE_DIR/fnp_python.so"
  else
    # Every step below imports that .so. Without this, a failed deploy left the
    # stale copy in place and steps 2-4 silently measured the PREVIOUS build.
    echo "  DEPLOY FAILED — $PROBE_DIR/fnp_python.so is stale or missing"; fail=1
  fi
else
  echo "  BUILD FAILED"; fail=1
fi

echo "== [2/4] linalg conformance suites =="
for t in conformance_linalg conformance_linalg_advanced conformance_linalg_decomp; do
  run_test_shard "$t" "$CARGO" test -p fnp-python --release --test "$t"
done

echo "== [3/4] fnp-ufunc + fnp-linalg crate tests (kernels behind the delegates) =="
run_test_shard fnp-linalg "$CARGO" test -p fnp-linalg --release

echo "== [4/4] re-measure the 4 delegated ops vs NumPy (expect ~parity 2-D; batched WIN) =="
if ! OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PROBE_DIR" python3 - <<'PY'
import numpy as np, fnp_python as f, time
def b(fn, n=7):
    for _ in range(2): fn()
    r=[]
    for _ in range(n):
        s=time.perf_counter(); fn(); r.append(time.perf_counter()-s)
    return sorted(r)[n//2]
rng=np.random.default_rng(7)
def spd(d): A=rng.standard_normal((d,d)); return A@A.T + d*np.eye(d)
print(f"{'op':18} {'d':>5} {'ratio':>7}  verdict")
def row(nm, d, nf, ff, want_lo=0.85, want_hi=1.15):
    tn=b(nf); tf=b(ff); r=tf/tn
    v='parity' if want_lo<=r<=want_hi else ('WIN' if r<want_lo else 'LOSS(check)')
    ok=np.allclose(np.asarray(nf()), np.asarray(ff()), atol=1e-6) if nm not in ('eigh','eigvalsh') else True
    print(f"{nm:18} {d:>5} {r:7.2f}  {v}")
for d in (200, 800):
    S=spd(d); M=rng.standard_normal((d,d))
    row('eigvalsh', d, lambda S=S:np.linalg.eigvalsh(S), lambda S=S:f.eigvalsh(S))
    row('eigh',     d, lambda S=S:np.linalg.eigh(S)[0],  lambda S=S:np.asarray(f.eigh(S)[0]))
    row('cholesky', d, lambda S=S:np.linalg.cholesky(S), lambda S=S:f.cholesky(S))
    row('matrix_power0', d, lambda M=M:np.linalg.matrix_power(M,0), lambda M=M:f.matrix_power(M,0))
# batched (should stay a native WIN)
A3=rng.standard_normal((200,16,16)); S3=np.einsum('...ij,...kj->...ik',A3,A3)+16*np.eye(16)
print("-- batched (expect WIN) --")
tn=b(lambda:np.linalg.eigvalsh(S3)); tf=b(lambda:f.eigvalsh(S3)); print(f"{'eigvalsh_batch':18} {'16':>5} {tf/tn:7.2f}  {'WIN' if tf<tn else 'check'}")
PY
then
  # This block imports the freshly deployed .so. An ImportError or a raised
  # exception here used to leave fail at 0, so the script reported success for a
  # module it could not even load.
  echo "  MEASUREMENT BLOCK FAILED — see the traceback above"; fail=1
fi

echo "== done (fail=$fail) =="
exit $fail
