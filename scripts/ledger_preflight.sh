#!/usr/bin/env bash
# ledger_preflight.sh — refuse a perf lever that the ledger has already settled.
#
# Run this BEFORE mutating any source for a performance candidate.
#
#   scripts/ledger_preflight.sh <keyword> [keyword ...]
#
# Exit codes (stable; scriptable):
#   0  CLEAR    — no prior rejection matched, or every match is VOID (the prior
#                 measurement could not have detected the lever, so it is a
#                 resurrection candidate rather than a closed question).
#   2  BLOCKED  — a prior rejection matched AND it is SOUND: it recorded an A/A
#                 null control, or refuted the lever on a counted mechanism, or
#                 the blocker is behavioral. Do not re-derive it.
#   3  USAGE
#
# Why not block on every prior rejection: the 2026-07-25 fleet audit measured
# 67.8% of this repo's rejected levers as VOID — rejected by a harness that
# could not have seen the effect. Blocking on those would entomb them. The
# taxonomy is in docs/LEDGER_RESURRECTION.md.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEDGER="${REPO_ROOT}/docs/NEGATIVE_EVIDENCE.md"

if [[ $# -eq 0 ]]; then
    cat >&2 <<'USAGE'
usage: scripts/ledger_preflight.sh <keyword> [keyword ...]

Searches docs/NEGATIVE_EVIDENCE.md for prior rejections of a candidate lever and
classifies each match as SOUND (blocking) or VOID (resurrection candidate).

  exit 0  CLEAR    no prior rejection, or all matches VOID
  exit 2  BLOCKED  a SOUND prior rejection exists — do not re-derive
  exit 3  usage

example:
  scripts/ledger_preflight.sh loadtxt usecols bool
USAGE
    exit 3
fi

if [[ ! -r "$LEDGER" ]]; then
    echo "preflight: cannot read $LEDGER" >&2
    exit 3
fi

if ! command -v rg >/dev/null 2>&1; then
    echo "preflight: ripgrep (rg) is required — apt-get install ripgrep" >&2
    exit 3
fi

# Build a case-insensitive AND query across headings: a row must mention every
# keyword to be considered a match for this candidate.
mapfile -t HEADING_LINES < <(rg -n '^## ' "$LEDGER" || true)

blocked=0
void_matches=0
sound_matches=0
declare -a REPORT=()

total="${#HEADING_LINES[@]}"
for idx in "${!HEADING_LINES[@]}"; do
    entry="${HEADING_LINES[$idx]}"
    lineno="${entry%%:*}"
    heading="${entry#*:}"
    heading="${heading###\# }"

    # every keyword must appear in the heading
    hay="$(printf '%s' "$heading" | tr '[:upper:]' '[:lower:]')"
    matched=1
    for kw in "$@"; do
        needle="$(printf '%s' "$kw" | tr '[:upper:]' '[:lower:]')"
        [[ "$hay" == *"$needle"* ]] || { matched=0; break; }
    done
    [[ $matched -eq 1 ]] || continue

    # verdict token = text before the first colon
    label="$(printf '%s' "${heading%%:*}" | tr '[:lower:]' '[:upper:]')"
    case "$label" in
        *WIN*|*KEEP*|*LANDED*) continue ;;  # a prior win is not a blocker
    esac
    case "$label" in
        *REJECT*|*NO-SHIP*|*NO\ SHIP*|*NOSHIP*|*REVERT*|*HOLD*|*BLOCKER*|*DROPPED*|*BENCH-BLOCKED*) ;;
        *) continue ;;
    esac

    # body = this heading up to the next one
    if (( idx + 1 < total )); then
        next="${HEADING_LINES[$((idx + 1))]}"
        end=$(( ${next%%:*} - 1 ))
    else
        end="$(wc -l < "$LEDGER")"
    fi
    body="$(sed -n "${lineno},${end}p" "$LEDGER" | tr '[:upper:]' '[:lower:]')"

    verdict="VOID"
    reason="no A/A null and no counted mechanism recorded — resurrection candidate"

    # Precedence matters and is load-bearing.
    #
    # (1) A row whose measured effect sits INSIDE its own A/A null is VALID-AB and
    #     is sound no matter what else it says about cv. Checked first, because
    #     such rows often discuss cv in passing and would otherwise be misread as
    #     cv-killed (the shipped GEMM register-tile rejection is exactly this).
    # (2) Only THEN VOID-CV: a row with a DIRECTIONAL effect that was rejected on
    #     the cv<5% gate campaign section 2.3 proves unreachable on this hardware.
    #     The highest-value resurrection here is that shape — effect 3.09-3.69x
    #     against nulls of 0.988-1.045, rejected because not all four arms cleared
    #     cv. Treating a recorded null as automatically SOUND would entomb it.
    if grep -qE 'inside the (a/a )?null|within the (a/a )?null|sits inside|every delta inside|no effect|indistinguishable from the null' <<<"$body"; then
        verdict="SOUND"; reason="measured effect sits inside its own A/A null — decided, not undecidable"
    elif grep -qE 'never clear|no run cleared|not clear[a-z]* (both )?(the )?cv|cv below 5%|below 5% cv|5% cv|cv (ceiling|gate)|variance gate|mandatory (5%|cv)|predeclared requirement that .{0,60}cv' <<<"$body"; then
        verdict="VOID"; reason="killed by the cv<5% gate, which campaign 2.3 proves unreachable here — resurrection candidate"
    elif grep -qE 'a/a|null control|null arm|null floor|null ratio|null median' <<<"$body"; then
        verdict="SOUND"; reason="recorded an A/A null control"
    elif grep -qE 'cycles|instruction|perf stat|ru_minflt|fault|syscall|allocation count|gb/s|gflop|bandwidth-bound|bandwidth-saturat|cache miss' <<<"$body"; then
        verdict="SOUND"; reason="refuted on a counted mechanism"
    elif grep -qE 'bit-exact|byte-exact|not bit-reproducible|not byte-reproducib|non-deterministic|last-ulp|signed-zero|order-unspecified|observable output|does not canonicalize|parity-fatal|sign ambiguit|parity minefield' <<<"$body"; then
        verdict="SOUND"; reason="behavioral blocker — no measurement can overturn it"
    fi

    if [[ "$verdict" == "SOUND" ]]; then
        blocked=1
        sound_matches=$((sound_matches + 1))
    else
        void_matches=$((void_matches + 1))
    fi
    REPORT+=("  [$verdict] docs/NEGATIVE_EVIDENCE.md:${lineno}
      ${heading}
      -> ${reason}")
done

printf 'ledger preflight: %s\n' "$*"
if [[ ${#REPORT[@]} -eq 0 ]]; then
    echo "  no prior rejection matched all keywords."
else
    printf '%s\n' "${REPORT[@]}"
fi
printf '  sound=%d void=%d\n' "$sound_matches" "$void_matches"

if [[ $blocked -eq 1 ]]; then
    cat <<'BLOCKED'

BLOCKED (exit 2): a prior rejection of this lever is SOUND — it recorded an A/A
null control, refuted the lever on a counted mechanism, or is behaviorally
blocked. Do not re-derive it. If you believe the rejection is stale, satisfy its
retry predicate first and say so in the new row.
BLOCKED
    exit 2
fi

if (( void_matches > 0 )); then
    cat <<'CLEAR'

CLEAR (exit 0): prior rejections exist but every one is VOID — the measurement
could not have detected the lever. This is a RESURRECTION, not a fresh idea.
Re-run under the median-CI gate (common::report_median_gate_pair), record an A/A
null in the same invocation, and cite the void row you are overturning.
CLEAR
else
    echo
    echo "CLEAR (exit 0): no prior rejection. Record an A/A null control or a counted"
    echo "mechanism in whatever row you write, or CI will reject it (ledger_hygiene.rs)."
fi
exit 0
