#!/usr/bin/env python3
"""Offline replica of crates/fnp-conformance/tests/ledger_hygiene.rs predicates.
Analysis only - reads the ledger, enumerates offenders per gate. Changes nothing.
"""
import re
import sys

LEDGER = "/data/projects/franken_numpy/docs/NEGATIVE_EVIDENCE.md"
ENFORCEMENT_DATE = "2026-07-26"
WORKER_DATE = "2026-08-15"
HARNESS_DATE = "2026-08-15"
HIST_VOID_BUDGET = 87
HIST_UNWORKERED_BUDGET = 39


def parse_entries():
    with open(LEDGER, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    starts = [i for i, l in enumerate(lines) if l.startswith("## ")]
    out = []
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        heading = lines[start][3:].strip()
        first = heading.split()[0] if heading.split() else ""
        date = first if (len(first) == 10 and first.count("-") == 2) else ""
        out.append({
            "heading": heading, "date": date,
            "body": "\n".join(lines[start:end]), "line": start + 1,
        })
    return out


def is_reject(heading):
    tail = heading.split(" - ", 1)[1] if " - " in heading else heading
    label = tail.split(":")[0].upper()
    negative = ["REJECT", "NO-SHIP", "NO SHIP", "NOSHIP", "REVERT", "HOLD",
                "BENCH-BLOCKED", "DROPPED", "BLOCKER"]
    positive = ["WIN", "SHIP,", "KEEP", "LANDED"]
    if any(p in label for p in positive) and "REJECT" not in label:
        return False
    return any(n in label for n in negative)


def is_keep(heading):
    tail = heading.split(" - ", 1)[1] if " - " in heading else heading
    label = tail.split(":")[0].upper()
    tokens = re.split(r"[^a-zA-Z0-9]+", label)

    def has(word):
        return word in tokens

    ship_negated = has("NO")
    return has("KEEP") or has("WIN") or (has("SHIP") and not ship_negated)


def line_has_decimal_measurement(line):
    low = line.lower()
    has_decimal = bool(re.search(r"\d\.\d", line))
    has_shape = ("null_base_aa" in low or "ratio" in low or "median=" in low
                 or "median:" in low or ("[" in low and "]" in low)
                 or (":" in low and any(c.isdigit() for c in low.split(":", 1)[1])))
    return has_decimal and has_shape


def records_null_control(body):
    for line in body.split("\n"):
        low = line.lower()
        pos = "null_base_aa" in low or "a/a" in low or "null control" in low
        neg = ("no a/a" in low or "without a/a" in low or "lacks a/a" in low
               or "no null" in low or "without null" in low or "lacks null" in low)
        if pos and not neg and line_has_decimal_measurement(line):
            return True
    return False


def records_counted_mechanism(body):
    for line in body.split("\n"):
        if ":" in line:
            label, evidence = line.split(":", 1)
            if label.strip().lower() == "counted_mechanism" and any(c.isdigit() for c in evidence):
                return True
    return False


def contains_sha256(text):
    return bool(re.search(r"[0-9a-fA-F]{64}", text))


def records_executing_elf_sha256(body):
    for line in body.split("\n"):
        low = line.lower()
        marker = ("bench_elf_sha256" in low or "executing elf sha" in low
                  or "executing-elf sha" in low or "executing binary sha" in low
                  or "executable sha" in low or "executable hash" in low)
        if marker and "unavailable" not in low and contains_sha256(line):
            return True
    return False


BEHAVIORAL = ["bit-exact", "byte-exact", "not bit-reproducible", "not byte-reproducib",
              "non-deterministic", "last-ulp", "signed-zero", "order-unspecified",
              "observable output", "does not canonicalize", "parity-fatal", "parity minefield"]


def is_behavioral_blocker(body):
    low = body.lower()
    return any(n in low for n in BEHAVIORAL)


UNMEASURED = ["bench-blocked", "did not compile", "never ran", "never measured",
              "could not run", "could not measure", "could not obtain", "unmeasured",
              "workers unavailable", "queue_timeout", "bench-cost prohibitive"]


def is_unmeasured(body):
    low = body.lower()
    return any(n in low for n in UNMEASURED)


PLACEHOLDERS = ["unavailable", "unknown", "n/a", "none", "tbd"]


def records_measuring_worker(body):
    for line in body.split("\n"):
        low = line.lower()
        for marker in ("host=", "worker="):
            if marker in low:
                rest = low.split(marker, 1)[1]
                name = re.split(r"[\s,)]", rest)[0].strip("`*\"")
                if name and name not in PLACEHOLDERS:
                    return True
    return False


def records_measuring_harness(body):
    for line in body.split("\n"):
        low = line.lower()
        if "harness=" in low:
            rest = low.split("harness=", 1)[1]
            name = re.split(r"[\s,)]", rest)[0].strip("`*\"")
            if name and name not in PLACEHOLDERS:
                return True
    return False


RESULT_CLASS_MARKER = "**Campaign result class:**"
NULL_MARKER = "**A/A null control (same invocation):**"
INCUMBENT_ARM_MARKER = "**Legacy incumbent arm (same invocation):**"
ISOLATION_MARKER = "**Incumbent isolation proof:**"
DISCLOSURE_MARKER = "**Shared timed component disclosure:**"


def marker_value(line, marker):
    stripped = line.strip()
    return stripped[len(marker):].strip() if stripped.startswith(marker) else None


def marker_entry_value(body, marker):
    for line in body.split("\n"):
        value = marker_value(line, marker)
        if value is not None:
            return value
    return None


def token_value(value, token):
    fields = [f.strip("`,;") for f in value.split() if f.startswith(token)]
    if len(fields) != 1:
        return None
    return fields[0][len(token):]


def first_sha256(text):
    match = re.search(r"[0-9a-fA-F]{64}", text)
    return match.group(0) if match else None


def executing_elf_sha256(body):
    for line in body.split("\n"):
        low = line.lower()
        marker = ("bench_elf_sha256" in low or "executing elf sha" in low
                  or "executing-elf sha" in low or "executing binary sha" in low
                  or "executable sha" in low or "executable hash" in low)
        if marker and "unavailable" not in low:
            sha = first_sha256(line)
            if sha:
                return sha
    return None


def result_class(body):
    values = [v for v in (marker_value(l, RESULT_CLASS_MARKER) for l in body.split("\n"))
              if v is not None]
    if not values:
        return "Missing"
    if len(values) > 1:
        return "Ambiguous"
    v = values[0].strip("`")
    if v == "maintenance-self-speedup":
        return "MaintenanceSelfSpeedup"
    if v == "incumbent-win":
        return "IncumbentWin"
    return "Invalid"


def is_lowercase_sha256(value):
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def disclosed_shared_component_is_admissible(body, declared):
    disclosure = marker_entry_value(body, DISCLOSURE_MARKER)
    if disclosure is None:
        return False
    if token_value(disclosure, "components=") != declared:
        return False
    comps = [c.strip() for c in declared.split(",")]
    if not all(c.startswith("numpy.") and len(c) > 6 for c in comps):
        return False
    if token_value(disclosure, "direction=") != "conservative_for_candidate":
        return False
    shares_raw = token_value(disclosure, "share_of_candidate_pct=")
    if shares_raw is None:
        return False
    try:
        shares = [float(s.strip()) for s in shares_raw.split(",")]
    except ValueError:
        return False
    if not shares or not all(s > 0.0 and s < 50.0 for s in shares):
        return False
    return True


def has_incumbent_win_contract(body):
    null_line = marker_entry_value(body, NULL_MARKER)
    measured_null = null_line is not None and line_has_decimal_measurement(null_line)
    incumbent = marker_entry_value(body, INCUMBENT_ARM_MARKER)
    if incumbent is None:
        return False
    name = token_value(incumbent, "name=")
    actual_numpy = name is not None and name.lower() == "numpy"
    version = token_value(incumbent, "version=")
    pinned_version = (version is not None and version != ""
                      and version.lower() != "unavailable"
                      and all(c.isalnum() or c in ".+_" or c == "-" for c in version))
    artifact = token_value(incumbent, "artifact_sha256=")
    pinned_artifact = False
    if artifact is not None and is_lowercase_sha256(artifact):
        cand = executing_elf_sha256(body)
        pinned_artifact = cand is None or cand.lower() != artifact.lower()
    inv = token_value(incumbent, "invocation_id=")
    shared_invocation = (inv is not None and inv != ""
                         and all(c.isalnum() or c in "._:-" for c in inv))
    ratio_tok = token_value(incumbent, "measured_ratio=")
    measured_ratio = False
    if ratio_tok is not None and ratio_tok.endswith("x"):
        try:
            r = float(ratio_tok[:-1])
            measured_ratio = r > 0.0
        except ValueError:
            pass
    isolation = marker_entry_value(body, ISOLATION_MARKER)
    independent = False
    if isolation is not None:
        candidate = token_value(isolation, "candidate=")
        inc = token_value(isolation, "incumbent=")
        shared = token_value(isolation, "shared_timed_component=")
        if shared == "none":
            isolated = True
        elif shared:
            isolated = disclosed_shared_component_is_admissible(body, shared)
        else:
            isolated = False
        independent = (candidate is not None and candidate.startswith("fnp.") and len(candidate) > 4
                       and inc is not None and inc.startswith("numpy.") and len(inc) > 6
                       and candidate != inc and isolated)
    return (measured_null and actual_numpy and pinned_version and pinned_artifact
            and shared_invocation and measured_ratio and independent)


entries = parse_entries()

gates = {}


def add(name, offs):
    gates[name] = offs


# Gate 1: new REJECT rows need null/mechanism
add("new_reject_rows_record_a_null_control_or_counted_mechanism",
    [e for e in entries if e["date"] and e["date"] >= ENFORCEMENT_DATE
     and is_reject(e["heading"])
     and not records_null_control(e["body"])
     and not records_counted_mechanism(e["body"])
     and not is_behavioral_blocker(e["body"])
     and not is_unmeasured(e["body"])])

# Gate 2: historical void-nonull budget
hist_void = [e for e in entries if (not e["date"] or e["date"] < ENFORCEMENT_DATE)
             and is_reject(e["heading"])
             and not records_null_control(e["body"])
             and not records_counted_mechanism(e["body"])
             and not is_behavioral_blocker(e["body"])
             and not is_unmeasured(e["body"])]
add("historical_void_nonull_debt_does_not_grow (budget<=87)", hist_void)

# Gate 3: KEEP rows need executing ELF
add("new_keep_rows_carry_an_executing_elf_sha256",
    [e for e in entries if e["date"] and e["date"] >= ENFORCEMENT_DATE
     and is_keep(e["heading"]) and not records_executing_elf_sha256(e["body"])])

# Gate 4: KEEP rows need one campaign class
add("new_keep_rows_declare_one_campaign_result_class",
    [e for e in entries if e["date"] and e["date"] >= ENFORCEMENT_DATE
     and is_keep(e["heading"])
     and result_class(e["body"]) not in ("MaintenanceSelfSpeedup", "IncumbentWin")])

# Gate 5: incumbent-win full contract
add("incumbent_win_rows_carry_the_complete_same_invocation_contract",
    [e for e in entries if e["date"] and e["date"] >= ENFORCEMENT_DATE
     and result_class(e["body"]) == "IncumbentWin"
     and not has_incumbent_win_contract(e["body"])])

# Gate 6: REJECT rows need retry predicate
add("new_reject_rows_carry_a_retry_predicate",
    [e for e in entries if e["date"] and e["date"] >= ENFORCEMENT_DATE
     and is_reject(e["heading"])
     and not any(w in e["body"].lower() for w in ("retry", "reopen"))])

# Gate 7: duplicate reject headings
seen = set()
dups = []
for e in entries:
    if is_reject(e["heading"]):
        if e["heading"] in seen:
            dups.append(e)
        seen.add(e["heading"])
add("reject_headings_are_unique", dups)

# Gate 8: measured rows name worker
add("new_measured_rows_name_their_worker",
    [e for e in entries if e["date"] and e["date"] >= WORKER_DATE
     and not is_unmeasured(e["body"]) and not is_behavioral_blocker(e["body"])
     and not records_measuring_worker(e["body"])])

# Gate 9: historical unworkered incumbent-win == 39 exactly
hist_unworkered = [e for e in entries if (not e["date"] or e["date"] < WORKER_DATE)
                   and result_class(e["body"]) == "IncumbentWin"
                   and not records_measuring_worker(e["body"])]
add(f"unworkered_incumbent_win_budget_is_not_stale (== {HIST_UNWORKERED_BUDGET})",
    hist_unworkered)

# Gate 10: measured rows name harness
add("new_measured_rows_name_their_harness",
    [e for e in entries if e["date"] and e["date"] >= HARNESS_DATE
     and not is_unmeasured(e["body"]) and not is_behavioral_blocker(e["body"])
     and not records_measuring_harness(e["body"])])

total_offending_rows = set()
print("=" * 72)
for name, offs in gates.items():
    print(f"\n### {name}: {len(offs)} offender(s)")
    for e in offs[:6]:
        print(f"  L{e['line']}: {e['heading'][:110]}")
    if len(offs) > 6:
        print(f"  ... and {len(offs) - 6} more")
    if "budget" not in name:
        total_offending_rows.update(id(e) for e in offs)

print("\n" + "=" * 72)
print(f"TOTAL distinct offending rows (non-budget gates): {len(total_offending_rows)}")

# Union of rows dated 2026-08-16..31 among offenders, per README's claim
window = [e for e in entries if "2026-08-16" <= e["date"] <= "2026-08-31"]
window_off = [e for e in window if any(e is g for g in sum([v for k, v in gates.items() if "budget" not in k], []))]
print(f"window rows 08-16..08-31 total: {len(window)}; offending among them: {len(window_off)}")
