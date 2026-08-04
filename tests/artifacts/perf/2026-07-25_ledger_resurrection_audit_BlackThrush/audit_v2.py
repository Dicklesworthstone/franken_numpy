#!/usr/bin/env python3
"""Ledger resurrection audit v2 — frankenfs six-class taxonomy, applied to franken_numpy.

Read-only over docs/NEGATIVE_EVIDENCE.md. The screen is TRIAGE, not a verdict;
every queued row is then read and adjudicated by hand (HAND table at the bottom).

Classes (frankenfs, verbatim):
  VALID-PROFILE   rejected before any source edit, named frame w/ non-zero self-time
                  + a computed Amdahl ceiling
  VALID-MECHANISM no A/A null, but refuted on a COUNTED mechanism (instructions,
                  cycles, syscalls, allocations, faults unchanged)
  VALID-AB        A/B with a recorded A/A null, effect sits inside it
  VOID-CV         killed ONLY by a cv<5% gate
  VOID-ZEROSELF   target frame ~0% self-time in the profile the bench actually ran
  VOID-NONULL     near-1.0 ratio, no null, no counted mechanism

Two additions, declared rather than smuggled (frankenfs added VOID-ISA the same way):
  VOID-UNMEASURED no timing was ever obtained (bench/control/worker died first).
                  frankenfs treated this as "the strongest form of void" for their
                  rank 1 but did not name it; naming it keeps it out of VOID-NONULL,
                  which is defined around a ratio that these rows do not have.
  VALID-MARGIN    a regression whose magnitude is far outside any A/A null this
                  hardware has ever produced. VOID-NONULL is defined on a NEAR-1.0
                  ratio; a 0.33x or 3x-slower result is not near 1.0 and a missing
                  null cannot manufacture it. Threshold is set from THIS repo's own
                  observed worst A/A null spread (0.78-1.10), widened to [0.70, 1.43]
                  -- i.e. a row must be >=1.43x or <=0.70x to qualify.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Default to the committed HEAD snapshot so appendix line numbers match git and do
# not drift with uncommitted rows in the working tree.
LEDGER = Path(sys.argv[2] if len(sys.argv) > 2
              else "/data/projects/franken_numpy/docs/NEGATIVE_EVIDENCE.md")
lines = LEDGER.read_text(encoding="utf-8", errors="replace").splitlines()

idxs = [i for i, l in enumerate(lines) if l.startswith("## ")]
entries = []
for n, i in enumerate(idxs):
    end = idxs[n + 1] if n + 1 < len(idxs) else len(lines)
    entries.append({"heading": lines[i][3:].strip(), "body": "\n".join(lines[i:end]),
                    "line": i + 1})

NEG = re.compile(r"\b(REJECT|REJECTED|NO-SHIP|NO SHIP|NOSHIP|REVERT|REVERTED|HOLD|"
                 r"BENCH-BLOCKED|LOSS|DROPPED|BLOCKER|NOT SHIPPED|ABANDON)\b", re.I)
POS = re.compile(r"\b(WIN|SHIP|SHIPPED|KEEP|LANDED|FIX)\b", re.I)
# Rows that are measurements/recon, not rejected levers - excluded like frankenfs's SURVEY.
SURVEY = re.compile(r"\b(SURVEY|SURFACE|RECON|PROBE|DATA|ADDENDUM|CORRECTION|"
                    r"RE-DECISION|CROSS-VALIDATED|NEUTRAL|CONVERGENCE|SWEEP COMPLETE|"
                    r"MECHANISM PROOF)\b", re.I)


def verdict_of(h):
    m = re.match(r"^\s*[\d-]{8,10}\s*[-–]\s*(.*)$", h)
    label = (m.group(1) if m else h).split(":", 1)[0]
    if SURVEY.search(label) and not NEG.search(label):
        return "SURVEY"
    if NEG.search(label):
        return "REJECT"
    if POS.search(label):
        return "KEEP"
    return "UNKNOWN"


RATIO = re.compile(r"(\d+\.\d+)\s*[x×]")
MEASURED = re.compile(r"(?:ratio|lhs/rhs|former/candidate|speedup|measured|effect)"
                      r"[^.\n]{0,80}?(\d+\.\d+)\s*[x×]", re.I)
NULL = re.compile(r"\bA/A\b|null control|null arm|null floor|null ratio|"
                  r"candidate A/A|\bnull\b", re.I)
# A COUNTED mechanism: a hardware/OS counter, not a wall clock.
COUNTED = re.compile(r"\bcycles\b|\binstructions?\b|perf stat|ru_minflt|"
                     r"\bfaults?\b|\bsyscalls?\b|allocation count|"
                     r"\bGB/s\b|GFLOP|bandwidth-bound|bandwidth-saturat|"
                     r"instruction count|\bIPC\b|cache miss", re.I)
AMDAHL = re.compile(r"Amdahl|ceiling|upper bound|at most .{0,20}%|"
                    r"removable|frame % ≠|FRAME % ", re.I)
UNTOUCHED = re.compile(r"production untouched|no source (?:was )?edit|"
                       r"before any (?:source )?edit|no production code was edited|"
                       r"benchmark-only|read-only", re.I)
SELFPCT = re.compile(r"(\d+\.\d+)\s*%\s*self|self[^.\n]{0,24}?(\d+\.\d+)\s*%", re.I)
SHA = re.compile(r"sha256|sha-256", re.I)
CVKILL = re.compile(r"(?:CV|cv)[^.\n]{0,70}(?:ceiling|gate|exceed|above|below|"
                    r"threshold|5\s*%|5%)|(?:exceed|above|failed|fails|missed|"
                    r"never clear|not clear)[^.\n]{0,50}(?:CV|cv)|"
                    r"excluded[^.\n]{0,70}(?:CV|cv)|variance gate", re.I)
UNMEAS = re.compile(r"BENCH-BLOCKED|did not compile|never (?:ran|executed|measured)|"
                    r"could not (?:run|measure|obtain|compile)|unmeasured|UNTIMED|"
                    r"workers unavailable|queue_timeout|bench-cost prohibitive|"
                    r"no (?:second )?benchmark[^.\n]{0,30}ran|prohibitive", re.I)
ZEROSELF = re.compile(r"0\.000+\s*%|0\.000000s|zero self|no self-time|"
                      r"never routed|did not route|not on the (?:hot )?path|"
                      r"profile blocker", re.I)
BEHAVIOR = re.compile(r"bit-exact|byte-exact|not bit-reproducible|non-deterministic|"
                      r"last-ULP|ULP|signed-zero|order-unspecified|"
                      r"not byte-reproducib|parity minefield", re.I)
# The row's OWN words about its outcome beat any scraped number. Hand-checking the
# first screen showed VALID-MARGIN was contaminated by ratios belonging to a sibling
# lever or to the vs-numpy residual gap, on rows whose headings said "flat" outright.
NEAR_ONE = re.compile(r"\bis flat\b|\bflat\b(?![- ]sum)|~0[- ]gain|0-gain|no gain|"
                      r"\bneutral\b|~?parity\b|do(?:es)? not move|did not move|"
                      r"within noise|no measurable|no measured (?:win|gain)|"
                      r"wall-NEUTRAL|indistinguishable", re.I)
# Dropped because another path already covers it - a dedup, not a measurement verdict.
SUPERSEDED = re.compile(r"supersed|already contain|duplicate source|dropped during rebase|"
                        r"covered by the (?:broader|upstream)|already landed|"
                        r"PAID-BY-PEER|already an ancestor", re.I)

# This repo's own observed A/A null extremes (from the .370/.377 rows): 0.78-1.10.
# Widened for the decidable-margin boundary.
MARGIN_LO, MARGIN_HI = 0.70, 1.43

rows = []
for e in entries:
    v = verdict_of(e["heading"])
    b = e["body"]
    ratios = [float(x) for x in RATIO.findall(b)]
    meas = [float(x) for x in MEASURED.findall(b)]
    eff = (max(meas, key=lambda r: abs(r - 1.0)) if meas
           else (max(ratios, key=lambda r: abs(r - 1.0)) if ratios else None))
    sp = []
    for m in SELFPCT.finditer(b):
        g = m.group(1) or m.group(2)
        if g:
            try:
                sp.append(float(g))
            except ValueError:
                pass
    rows.append({
        "heading": e["heading"], "line": e["line"], "verdict": v, "eff": eff,
        "has_null": bool(NULL.search(b)), "counted": bool(COUNTED.search(b)),
        "amdahl": bool(AMDAHL.search(b)), "untouched": bool(UNTOUCHED.search(b)),
        "self_pct": max(sp) if sp else None, "has_sha": bool(SHA.search(b)),
        "cvkill": bool(CVKILL.search(b)), "unmeas": bool(UNMEAS.search(b)),
        "zeroself": bool(ZEROSELF.search(b)), "behavior": bool(BEHAVIOR.search(b)),
        # attributable = the ratio was phrased as this candidate's effect, not
        # merely present somewhere in the prose.
        "attributable": bool(meas),
        "near_one_words": bool(NEAR_ONE.search(e["heading"]) or NEAR_ONE.search(b[:900])),
        "superseded": bool(SUPERSEDED.search(b)),
    })


def screen(r):
    """Mechanical triage in the taxonomy's own precedence order."""
    if r["behavior"] and r["eff"] is None:
        return "EXCLUDED-BEHAVIOR", "bit/byte-exactness blocker, not a perf rejection"
    if r["superseded"] and not r["near_one_words"]:
        return "EXCLUDED-SUPERSEDED", "dropped as duplicate/covered elsewhere, not a measurement verdict"
    # The row's own outcome words override any scraped ratio.
    if r["near_one_words"]:
        if r["has_null"]:
            return "VALID-AB", "self-described flat/neutral with a recorded A/A null"
        if r["counted"]:
            return "VALID-MECHANISM", "self-described flat/neutral, refuted on a counted mechanism"
        return "VOID-NONULL", "self-described flat/neutral, no A/A null, no counted mechanism"
    if r["unmeas"]:
        return "VOID-UNMEASURED", "no timing was ever obtained"
    if r["zeroself"]:
        return "VOID-ZEROSELF", "target frame ~0% self-time in the profile the bench ran"
    if r["cvkill"] and not r["has_null"]:
        return "VOID-CV", "killed only by a cv<5% gate"
    if r["untouched"] and r["self_pct"] and r["amdahl"]:
        return "VALID-PROFILE", (f"rejected pre-edit on a named frame at "
                                 f"{r['self_pct']:.2f}% self with a computed ceiling")
    if r["eff"] is None:
        if r["counted"]:
            return "VALID-MECHANISM", "no ratio, but refuted on a counted mechanism"
        return "VOID-NONULL", "no ratio and no counted mechanism recorded"
    if r["has_null"]:
        if r["cvkill"] and not (MARGIN_LO <= r["eff"] <= MARGIN_HI):
            return "VOID-CV", (f"{r['eff']:.4f}x with a recorded null, killed on the cv gate")
        return "VALID-AB", f"{r['eff']:.4f}x decided against a recorded A/A null"
    if not (MARGIN_LO <= r["eff"] <= MARGIN_HI):
        if r["counted"]:
            return "VALID-MECHANISM", (f"{r['eff']:.4f}x, no null, but a counted "
                                       f"mechanism is recorded")
        # VALID-MARGIN requires the ratio to be ATTRIBUTABLE to this candidate.
        # A bare number scraped from prose is not evidence: hand-checking the first
        # screen found sibling-lever and residual-gap ratios rescuing rows that the
        # row itself called flat. Unattributable => fall through to VOID-NONULL.
        if r["attributable"]:
            return "VALID-MARGIN", (f"{r['eff']:.4f}x (attributed) is outside any A/A "
                                    f"null this hardware has produced")
        return "VOID-NONULL", (f"only an unattributable {r['eff']:.4f}x in prose, "
                               f"no A/A null, no counted mechanism")
    if r["counted"]:
        return "VALID-MECHANISM", (f"near-1.0 ({r['eff']:.4f}x) but refuted on a "
                                   f"counted mechanism")
    return "VOID-NONULL", (f"near-1.0 ({r['eff']:.4f}x), no A/A null, no counted mechanism")


# ------------------------------------------------------- hand adjudication
# Line numbers are against the COMMITTED HEAD snapshot of NEGATIVE_EVIDENCE.md.
# Every row here was read in full; the screen's verdict is recorded next to the
# correction so the triage error itself is auditable.
HAND = {
    806: ("VOID-CV",
          "screen said VOID-UNMEASURED. It WAS measured - four runs, 1.82-1.98x - and "
          "the row's own heading says 'no run cleared the CV/null gate'. Killed by the "
          "cv gate, not by absence of a measurement."),
    1121: ("EXCLUDED-BEHAVIOR",
           "screen said VOID-NONULL. TSQR behind fnp.linalg.qr is a CORRECTNESS blocker - "
           "R's row signs are observable output and numpy does not canonicalize them - and "
           "it was filed BEFORE any dispatch code was written. Not a perf rejection."),
    1301: ("VALID-AB",
           "screen said VOID-CV. The heading states 'every delta inside the A/A null-control "
           "spread': a null WAS recorded and the effect sits inside it. Textbook VALID-AB. "
           "The campaign's own dead-end list marks this settled; voiding it would be wrong."),
    1370: ("EXCLUDED-PARSE",
           "a WIN (SHIP) row with an embedded REJECT clause - heading misparse, not a lever."),
    1562: ("EXCLUDED-SURVEY",
           "screen said VOID-NONULL. 'LOSS BASELINE (LEDGERED) ... production untouched' is a "
           "baseline measurement opening a frontier, not a rejected lever. It was subsequently "
           "FIXED by .367 (1.445x shipped)."),
    6427: ("VOID-UNMEASURED",
           "screen said VALID-MECHANISM. The requested Rayon pool 'was not admitted', so the "
           "arm never ran under the configuration it claimed. The bench could not have "
           "detected the lever."),
}

for r in rows:
    if r["verdict"] == "REJECT":
        r["cls"], r["why"] = screen(r)
        r["screen_cls"] = r["cls"]
        if r["line"] in HAND:
            r["cls"], r["why"] = HAND[r["line"]]
            r["why"] = "HAND: " + r["why"]
    else:
        r["cls"] = r["why"] = r["screen_cls"] = None

Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/audit_v2.json").write_text(
    json.dumps(rows, indent=1))

print("verdict distribution:", dict(Counter(r["verdict"] for r in rows)))
rej = [r for r in rows if r["verdict"] == "REJECT"]
print(f"\nREJECT audited: {len(rej)}")
for k, n in Counter(r["cls"] for r in rej).most_common():
    print(f"  {k:20s} {n:4d}  {100*n/len(rej):5.1f}%")
void = [r for r in rej if r["cls"].startswith("VOID")]
print(f"\nVOID total: {len(void)}/{len(rej)} = {100*len(void)/len(rej):.1f}%")
print(f"rows with binary sha256: {sum(1 for r in rej if r['has_sha'])}/{len(rej)} = "
      f"{100*sum(1 for r in rej if r['has_sha'])/len(rej):.1f}%")
