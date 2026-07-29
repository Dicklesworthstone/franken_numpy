#!/usr/bin/env python3
"""Audit docs/NEGATIVE_EVIDENCE.md per campaign Meta-Lever #1 (Ledger Resurrection).

Read-only: parses the ledger into entries and classifies REJECT rows as VOID or SOUND.
Emits JSON for downstream table generation. Touches no source files.
"""
import json
import re
import sys
from pathlib import Path

LEDGER = Path("/data/projects/franken_numpy/docs/NEGATIVE_EVIDENCE.md")

text = LEDGER.read_text(encoding="utf-8", errors="replace")
lines = text.splitlines()

# Split on level-2 headings.
idxs = [i for i, l in enumerate(lines) if l.startswith("## ")]
entries = []
for n, i in enumerate(idxs):
    end = idxs[n + 1] if n + 1 < len(idxs) else len(lines)
    head = lines[i][3:].strip()
    body = "\n".join(lines[i:end])
    entries.append({"heading": head, "body": body, "line": i + 1, "nlines": end - i})

# Negative verdicts: a lever was considered and did not ship.
REJECT_RE = re.compile(
    r"\b(REJECT|REJECTED|NO-SHIP|NO SHIP|NOSHIP|REVERT|REVERTED|HOLD|"
    r"BENCH-BLOCKED|LOSS|DROPPED|BLOCKER|NOT SHIPPED|ABANDON)\b",
    re.I,
)
WIN_RE = re.compile(r"\b(WIN|SHIP|SHIPPED|KEEP|LANDED|FIX)\b", re.I)


def verdict_of(h):
    # Heading form: "YYYY-MM-DD - VERDICT: title"
    m = re.match(r"^\s*[\d-]{8,10}\s*[-–]\s*(.*)$", h)
    tail = m.group(1) if m else h
    label = tail.split(":", 1)[0]
    # The label (text before the first colon) is the authoritative verdict token.
    if REJECT_RE.search(label):
        return "REJECT"
    if WIN_RE.search(label):
        return "WIN"
    return "OTHER"


RATIO_RE = re.compile(r"(\d+\.\d+)\s*[x×]")
# A/A null-control evidence
NULL_RE = re.compile(r"\bnull\b|\bA/A\b|\bAA\b|null control|null arm|null floor", re.I)
# binary provenance
SHA_RE = re.compile(r"sha256|sha-256|\bsha\b[^a-z]", re.I)
# self-time profile attribution
SELF_RE = re.compile(r"self[- ]?time|% self|self%|\bsamples\b|perf record|profil", re.I)
SELFPCT_RE = re.compile(r"(\d+\.\d+)\s*%\s*self|self[^.\n]{0,20}?(\d+\.\d+)\s*%", re.I)
CV_RE = re.compile(r"\bCV\b|coefficient of variation", re.I)
CVGATE_RE = re.compile(
    r"(CV|cv)[^.\n]{0,60}(ceiling|gate|exceed|above|over|below|threshold|5\s*%|5%)"
    r"|(exceed|above|over|failed|fails|missed)[^.\n]{0,40}(CV|cv)"
    r"|excluded[^.\n]{0,60}(CV|cv)",
)
PINNED_RE = re.compile(r"pinned to CPU|taskset|pinned|isolated worker|exclusive worker", re.I)

# ratio phrasing that indicates the *measured* effect of the candidate
MEASURED_RE = re.compile(
    r"(?:ratio|lhs/rhs|former/candidate|speedup|measured)[^.\n]{0,80}?(\d+\.\d+)\s*[x×]", re.I
)

results = []
for e in entries:
    v = verdict_of(e["heading"])
    body = e["body"]
    ratios = [float(x) for x in RATIO_RE.findall(body)]
    has_null = bool(NULL_RE.search(body))
    has_sha = bool(SHA_RE.search(body))
    has_self = bool(SELF_RE.search(body))
    has_cv = bool(CV_RE.search(body))
    cv_gate_kill = bool(CVGATE_RE.search(body))
    pinned = bool(PINNED_RE.search(body))

    # self-time percentages mentioned
    selfpcts = []
    for m in SELFPCT_RE.finditer(body):
        g = m.group(1) or m.group(2)
        if g:
            try:
                selfpcts.append(float(g))
            except ValueError:
                pass
    max_self = max(selfpcts) if selfpcts else None

    # candidate "effect" ratio: prefer explicitly-measured phrasing, else max distance from 1
    meas = [float(x) for x in MEASURED_RE.findall(body)]
    if meas:
        eff = max(meas, key=lambda r: abs(r - 1.0))
    elif ratios:
        eff = max(ratios, key=lambda r: abs(r - 1.0))
    else:
        eff = None

    results.append(
        {
            "heading": e["heading"],
            "line": e["line"],
            "nlines": e["nlines"],
            "verdict": v,
            "n_ratios": len(ratios),
            "eff_ratio": eff,
            "min_ratio": min(ratios) if ratios else None,
            "max_ratio": max(ratios) if ratios else None,
            "has_null": has_null,
            "has_sha": has_sha,
            "has_self": has_self,
            "has_cv": has_cv,
            "cv_gate_kill": cv_gate_kill,
            "pinned": pinned,
            "max_self_pct": max_self,
        }
    )

# ---------------------------------------------------------------- VOID rules
# Fleet null band (frankenlibc bd-3ollh0, adopted fleet-wide): a ratio inside
# 0.905-1.105 is indistinguishable from an A/A null on this hardware unless the
# entry recorded its own, tighter, measured null floor.
NULL_LO, NULL_HI = 0.905, 1.105

BLOCKED_RE = re.compile(r"BENCH-BLOCKED|did not compile|never (?:ran|executed|measured)|"
                        r"could not (?:run|measure|obtain)|unmeasured|UNTIMED|"
                        r"workers unavailable|queue_timeout", re.I)
DEADFRAME_RE = re.compile(r"0\.000+\s*%|0\.000000s|zero self|no self-time|"
                          r"never routed|did not route|not on the (?:hot )?path", re.I)


def classify(r):
    """Return (verdict, reason). VOID = the measurement could not have detected the lever."""
    body_blocked = r["blocked"]
    if body_blocked:
        return "VOID", "never measured (bench-blocked / harness or worker failure)"
    if r["dead_frame"]:
        return "VOID", "target frame ~0% self-time in the profile the bench exercised"
    if r["cv_gate_kill"] and not r["pinned"]:
        return "VOID", "killed by a cv<5% gate on an unpinned shared worker (unreachable gate)"
    eff = r["eff_ratio"]
    if eff is None:
        return "VOID", "no ratio recorded at all"
    if NULL_LO <= eff <= NULL_HI:
        if not r["has_null"]:
            return "VOID", f"claimed {eff:.3f}x sits inside the 0.905-1.105 null band, no null control"
        return "VOID-WEAK", f"claimed {eff:.3f}x sits inside the fleet null band (own null recorded)"
    if not r["has_null"] and not r["has_sha"]:
        return "SOUND-WEAK", f"{eff:.3f}x is outside the null band but has neither null control nor binary sha"
    return "SOUND", f"{eff:.3f}x is decidably outside the null band"


for r in results:
    r["blocked"] = bool(BLOCKED_RE.search(r["heading"]))
    r["dead_frame"] = bool(DEADFRAME_RE.search(r["heading"]))

# body-level checks need the body again; re-scan
for r, e in zip(results, entries):
    if not r["blocked"]:
        r["blocked"] = bool(BLOCKED_RE.search(e["body"][:1200]))
    if not r["dead_frame"]:
        r["dead_frame"] = bool(DEADFRAME_RE.search(e["body"]))

# ------------------------------------------------- hand adjudication overrides
# Every regex VOID candidate was read in full; these are the rows where the
# hand read disagrees with the automated verdict, with the reason.
HAND = {
    10440: ("SOUND", "66.1% self-time frame + bit-identical proof + 7 interleaved rounds + "
                     "profile-confirmed mechanism (idle cycles are drain-tail, off the critical path)"),
    1160: ("SOUND", "clean measurement; all six tile alternatives lose 2.3-39%; campaign marks it SETTLED"),
    1370: ("SKIP", "WIN row with an embedded REJECT clause - heading misparse, not a reject"),
    14250: ("SOUND", "behavior: fused narrow-float transcendentals are not bit-exact"),
    27096: ("SOUND", "behavior: numpy hypotf path is not bit-reproducible"),
    27108: ("SOUND", "behavior: numpy signed-zero sort order is non-deterministic"),
    28562: ("SOUND", "behavior: unique_values is order-unspecified"),
    29615: ("SOUND", "behavior: complex128 sqrt last-ULP divergence"),
    19644: ("SOUND", "strictly slower than numpy's C loop - decidable margin"),
    23865: ("SOUND", "1.5-3x SLOWER - decidable margin"),
    14217: ("SOUND", "mechanism: bandwidth-saturated, numpy already at the memory wall"),
    27578: ("SOUND", "mechanism: bandwidth-saturated"),
    27925: ("SOUND", "mechanism: bandwidth-saturated"),
    15609: ("SOUND", "mechanism: serial unary regime is a measured kernel wall"),
    10313: ("SOUND", "this row IS a prior ledger-integrity revalidation, not a lever reject"),
}

for r in results:
    if r["verdict"] == "REJECT":
        v, why = classify(r)
        if r["line"] in HAND:
            v, why = HAND[r["line"]]
            why = "HAND: " + why
        r["audit"] = v
        r["audit_reason"] = why
    else:
        r["audit"] = None
        r["audit_reason"] = None

Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ledger_audit.json").write_text(
    json.dumps(results, indent=1)
)

rej2 = [r for r in results if r["verdict"] == "REJECT"]
from collections import Counter
print("\n--- AUDIT VERDICTS (REJECT population) ---")
for k, n in Counter(r["audit"] for r in rej2).most_common():
    print(f"  {k:12s} {n:4d}  ({100*n/len(rej2):.1f}%)")

tot = len(results)
rej = [r for r in results if r["verdict"] == "REJECT"]
print(f"entries={tot} REJECT={len(rej)} WIN={sum(1 for r in results if r['verdict']=='WIN')} "
      f"OTHER={sum(1 for r in results if r['verdict']=='OTHER')}")
print(f"REJECT with no null control: {sum(1 for r in rej if not r['has_null'])}")
print(f"REJECT with no sha:          {sum(1 for r in rej if not r['has_sha'])}")
print(f"REJECT with no self-time:    {sum(1 for r in rej if not r['has_self'])}")
print(f"REJECT killed by CV gate:    {sum(1 for r in rej if r['cv_gate_kill'])}")
print(f"REJECT unpinned + CV gate:   {sum(1 for r in rej if r['cv_gate_kill'] and not r['pinned'])}")
