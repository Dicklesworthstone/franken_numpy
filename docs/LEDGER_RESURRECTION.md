# Ledger Resurrection Audit — `docs/NEGATIVE_EVIDENCE.md`

Campaign: FrankenSuite Performance Domination, 2026-07-25, Meta-Lever #1.
Auditor: `BlackThrush` (cc / Lane M). Ledger snapshot: 1,005 `##` entries at
commit `38f8acf3`.

**Taxonomy: frankenfs's six classes, adopted verbatim per the 2026-07-25 fleet
broadcast.** This document is a rewrite. The first version of this audit, landed
in `a65315a5`, used a taxonomy of my own and reported **16.6% void**. That number
was wrong, and it was wrong in the flattering direction. §3.1 explains exactly
how, because the mistake is more instructive than the corrected number.

A REJECT row is **VOID** when the measurement *could not have detected the
lever* — as opposed to detecting it and finding it absent.

---

## 1. Method

Entries split on `##` headings; the verdict token is the text before the first
colon. `KEEP`/`WIN`/`SHIP`/`LANDED` rows and `SURVEY`/`RECON`/`PROBE` rows are
excluded — the latter are measurements, not rejected levers, following
frankenfs's exclusion of the same class.

The screen is **triage, not a verdict**. Its output is in
`tests/artifacts/perf/2026-07-25_ledger_resurrection_audit_BlackThrush/`
(`audit_v2.py` + `audit_v2.json` + the per-row appendix). Every row in the §4
queue and every row where the screen's class was contestable was read in full
and adjudicated by hand; the `HAND` table in `audit_v2.py` records each override
next to the screen's original verdict, so the triage error itself is auditable.

### Classes

| Class | Meaning | Sound? |
|---|---|:--:|
| `VALID-PROFILE` | Rejected before any source edit, on a named frame with non-zero self-time and a computed Amdahl ceiling. | ✅ |
| `VALID-MECHANISM` | No A/A null, but refuted on a **counted** mechanism — instructions, cycles, syscalls, allocations, faults, bandwidth. A null cannot change the fact that no work was removed. | ✅ |
| `VALID-AB` | A/B with a recorded A/A null; the effect sits inside it. | ✅ |
| `VOID-CV` | An A/B ran and was killed **only** by a `cv<5%` gate. | ❌ |
| `VOID-ZEROSELF` | Target frame ~0% self-time in the profile the bench actually ran. | ❌ |
| `VOID-NONULL` | A/B rejected on a near-1.0 ratio, with **no** A/A null and no counted mechanism. Cannot distinguish lever from harness. | ❌ |

Two additions, **declared rather than smuggled** (frankenfs added `VOID-ISA` the
same way):

- **`VOID-UNMEASURED`** — no timing was ever obtained; the bench, control, or
  worker died first. frankenfs treated this as "the strongest form of void" for
  their rank 1 but did not name it. It does not belong in `VOID-NONULL`, which is
  defined around a ratio these rows never produced.
- **`VALID-MARGIN`** — a regression whose magnitude is far outside any A/A null
  this hardware has produced. `VOID-NONULL` is defined on a **near-1.0** ratio; a
  0.33× is not near 1.0 and a missing null cannot manufacture one. Threshold set
  from this repo's own observed worst A/A null spread (0.78–1.10), widened to
  [0.70, 1.43]. **A row qualifies only if the ratio is attributable** — see §3.1.

Three exclusions carry rows out of the audited population rather than into a
verdict: `EXCLUDED-BEHAVIOR` (bit-exactness or observable-output blockers, which
no measurement can overturn), `EXCLUDED-SUPERSEDED` (dropped as a duplicate of a
path that already landed), and `EXCLUDED-PARSE`/`-SURVEY`.

---

## 2. Counts

| Metric | Count |
|---|---:|
| Ledger entries parsed | 1,005 |
| — KEEP verdict | 713 |
| — SURVEY / recon | 38 |
| — UNKNOWN (unparsable verdict token) | 108 |
| **REJECT verdict — audited** | **146** |
| `VOID-NONULL` | **81** |
| `VOID-UNMEASURED` | 15 |
| `VOID-CV` | 2 |
| `VOID-ZEROSELF` | 1 |
| **VOID total** | **99 / 146 = 67.8%** |
| `VALID-MECHANISM` | 18 |
| `VALID-AB` | 12 |
| `VALID-MARGIN` | 4 |
| `EXCLUDED-*` | 13 |
| Rows carrying a binary sha256 | 21 / 146 = **14.4%** |

**Read this honestly, as frankenfs insists.** 67.8% void is *not* 99 buried
wins. `VOID-NONULL` overwhelmingly means "the row measured ~1.0× and never wrote
down what ~1.0× means on that bench" — most of those levers really are dead, and
the class exists because the row cannot *prove* it. The actionable yield is a
small head, ranked in §4.

The fleet-wide prediction holds here: **`VOID-CV` is 2 rows; the epidemic is
`VOID-NONULL` at 81.**

---

## 3. What the first version of this audit got wrong

### 3.1 The `SOUND-WEAK` rationalization, and the ratio-attribution bug

The first version had a class called `SOUND-WEAK`: 90 rows with a decidable
margin but no null control **and** no binary sha. I called them sound on the
reasoning that a large ratio is self-evidently outside any null, so the missing
null was a provenance blemish rather than a verdict problem. That is precisely
the `VOID-NONULL` epidemic, renamed so it read as acceptable.

Re-screening under the six classes exposed a second, mechanical error that had
inflated the same class. The first mechanical pass under the new taxonomy put
**71 rows** in `VALID-MARGIN`. Hand-reading the sample found the ratio regex was
scraping numbers that were not the candidate's effect at all:

| Row | Scraped | What that number actually was |
|---|---|---|
| `element_count` rank specialization | 1.50× | nothing to do with the candidate — the heading says "**is flat**" |
| f64 `frexp` overhead trims | 2.703× | the *vs-numpy residual gap* (1.187×), not the candidate's effect; the row says the trims "do not move" it |
| narrow ASCII `U8` union helper | 96.3× | the **upstream primitive's** win, which superseded this candidate as a duplicate |

Two corrections followed. First, **the row's own outcome words override any
scraped number** — a heading that says "flat", "~0-gain", "neutral", "parity" or
"do not move" is a near-1.0 row regardless of what digits appear in its prose.
Second, **`VALID-MARGIN` now requires the ratio to be *attributable*** — phrased
as this candidate's measured effect, not merely present somewhere in the body.
Unattributable ratios fall through to `VOID-NONULL`, the conservative direction.

That moved 67 of the 71 out of `VALID-MARGIN`, taking the void rate from 28.8%
to 67.8%. **Any screen that rescues rows on a bare scraped ratio is under-counting
void the same way.**

### 3.2 Hand adjudication overturned the screen in both directions

Six rows, recorded in `audit_v2.py`'s `HAND` table with the screen's original
verdict beside each:

| Line | Screen said | Hand verdict | Why |
|---:|---|---|---|
| 1301 | `VOID-CV` | **`VALID-AB`** | Heading: "every delta inside the A/A null-control spread". A null *was* recorded and the effect sits inside it. Voiding it would have reopened the `std::simd` GEMM register tile — a lane the campaign's own dead-end list marks settled. |
| 806 | `VOID-UNMEASURED` | **`VOID-CV`** | It *was* measured — four runs, 1.82–1.98× — and the heading says "no run cleared the CV/null gate". |
| 6427 | `VALID-MECHANISM` | **`VOID-UNMEASURED`** | The requested Rayon pool "was not admitted", so the arm never ran under the configuration it claimed. |
| 1121 | `VOID-NONULL` | **`EXCLUDED-BEHAVIOR`** | TSQR behind `fnp.linalg.qr`: R's row signs are observable output NumPy does not canonicalize. Filed *before* any dispatch code. A correctness blocker, not a perf rejection. |
| 1562 | `VOID-NONULL` | **`EXCLUDED-SURVEY`** | "LOSS BASELINE (LEDGERED) … production untouched" is a baseline opening a frontier. Subsequently fixed by `.367` (1.445× shipped). |
| 1370 | — | **`EXCLUDED-PARSE`** | A WIN (SHIP) row with an embedded REJECT clause. |

`VALID-MECHANISM` was audited in the rescuing direction too, as the broadcast
requires. The `packbits` rejection (1.040×, no null) records "~17 GB/s at 64M
bools, memory-bound" — a counted bandwidth measurement establishing saturation.
That refutes the lever regardless of any null: you cannot beat a memory-bound
kernel with threads. Kept `VALID-MECHANISM`.

---

## 4. Ranked rehabilitation queue

Ranked by target-frame self-time, per the campaign rule.

| # | Entry | Line | Class | Self-time | Effect on record | Status |
|---:|---|---:|---|---|---|---|
| 1 | selected-bool `loadtxt(usecols)` direct parse | 300 | `VOID-CV` | 8.14% `cfree` + 6.92% `malloc` + 5.80% owned-`Vec<String>` collect | **3.09–3.69×**, A/A nulls **0.988–1.045** | patch ready (`deadlock-audit-8mrfx`) |
| 2 | all-negative usecols bounded tail ring | 806 | `VOID-CV` | 10.67% full-row `Vec<&str>` collect | **1.82–1.98×** | **re-won** by the cod lane at 1.661938×, CI [1.632, 1.716], null CI [0.956, 1.023] |
| 3 | complex `nancumprod` axis-0 | 6375 | `VOID-UNMEASURED` | never profiled | — | cod lane |
| 4 | cov/corrcoef paired triangular Gram | 30156 | `VOID-UNMEASURED` | 66.1% kernel closure (sibling row) | — | cod lane |
| 5 | direct C-order `nditer` chunk range | 5666 | `VOID-UNMEASURED` | never profiled | — | open; wants a dedicated `fnp-iter` target |

Ranks 1–2 are the two levers the allocation addendum named for this repo. Both
are `VOID-CV`, and neither needed a quieter host so much as a correct gate:
**rank 1's nulls were already at unity while it was being rejected.**

Five of the 15 `VOID-UNMEASURED` rows were blocked by the monolithic
`criterion_python_surface` bench binary's compile cost — the failure mode closed
by `deadlock-audit-x7nnf` (17 per-domain binaries; 19.6k lines / 205 fns → 3.2k /
28). **The instrument that voided them has since been repaired.**

---

## 5. Yield

| Metric | Count |
|---|---:|
| Entries parsed | 1,005 |
| REJECT audited | 146 |
| VOID | 99 (67.8%) |
| Re-run under the corrected harness | 5 |
| **Re-won** | **4** |
| Best re-won ratio | 167.610955× |

Scoreboard line: `franken_numpy | 1005 | 146 | 99 | 67.8 | 5 | 4 | 167.610955x`

**Provenance caveat:** the reruns are the cod lane's (`VioletOwl`) and were
measured but not yet pushed to `origin/main` at the time of writing — Generator
Zipf 1.296191×, RandomState Zipf 1.249485×, the tail ring 1.661938×, and rank 5
at 167.610955×. They are relayed, not verified against a landed commit.

---

## 6. Institutionalized — ledger integrity decays

The fleet result is unambiguous: the repo that audited once *and enforced the
check* sits at 1.7% void; repos that never did sit at 25–91%. Banking the wins is
not the deliverable. Two gates landed with this audit.

### 6.1 `crates/fnp-conformance/tests/ledger_hygiene.rs` — CI-enforced

Runs in G2 alongside the existing hygiene suite.

- `new_reject_rows_record_a_null_control_or_counted_mechanism` — a REJECT row
  dated on or after **2026-07-26** must record an A/A null **or** a counted
  mechanism, else CI fails with the offending `file:line`. Writing a bare
  near-1.0 rejection is now impossible, not merely discouraged.
- `historical_void_nonull_debt_does_not_grow` — the 81 historical rows are
  grandfathered behind an explicit budget that may shrink but never grow, so
  backdating a row to dodge the gate above trips this one instead.
- `new_reject_rows_carry_a_retry_predicate` — a rejection nobody can reopen is
  how a void row becomes permanent.
- `reject_headings_are_unique` — two agents cannot silently write the same
  rejection twice.

### 6.2 `scripts/ledger_preflight.sh <keywords>` — exit 2 = BLOCKED

frankensqlite's preflight, with one deliberate difference: it does **not** block
on every prior rejection. It classifies each match and blocks only on `SOUND`
ones. Blocking on all of them would entomb the 99 void rows. On a void match it
exits 0 and tells the agent it is running a *resurrection*, naming the row to
cite.

Building it caught a bug in itself worth recording: v1 returned BLOCKED for the
rank-1 bool-`loadtxt(usecols)` row, because that row *does* record a clean A/A
null and was killed by the cv gate anyway. Precedence is now (1) effect sits
inside its own null → SOUND, (2) cv-killed → VOID, (3) null recorded → SOUND.

### 6.3 ELF sha-256 self-report

`benches/common/mod.rs::self_identity()`, called from `gated_main`, which fronts
**all 17** fnp-python bench binaries. Printed as line one **before `Criterion` is
constructed** — Criterion emits a backend notice on construction, and printing
after it buries the provenance line and costs a rerun.

This repo was the only one in the fleet at zero benches emitting an executing-ELF
hash. A hash computed by a shell step next to the run proves nothing: `rch`
compiles into an opaque per-worker pool target dir the caller cannot predict, and
concurrent agents in this fleet have edited crates mid-benchmark at least three
times.

### 6.4 Standing rules this audit adds

1. **A near-1.0 rejection without a null or a counted mechanism is void.** Not
   "weakly sound", not "provenance debt". Void.
2. **A ratio in the prose is not the candidate's effect** unless it is phrased as
   such. Screens that rescue rows on scraped numbers under-count void.
3. **The row's own outcome words beat any number in it.** "Flat" means flat.
4. **A behavioral blocker is permanently sound** — bit-exactness and
   observable-output rejections are not resurrection candidates at any gate.
5. **A bench-blocked row is not negative evidence about its lever.** File it
   against the instrument and re-queue when the instrument is fixed.

---

## 7. Environment blocker, still open

rch worker `ovh-b` **SIGILLs on the workspace `+avx2` baseline** — the
`zerocopy v0.8.48` build script dies with `signal: 4` under
`cargo clippy --workspace`. This is an independent franken_numpy reproduction of
frankenscipy `hhr7j` and franken_networkx's global-AVX2 rejection (campaign
§3b). Any franken_numpy measurement that landed on `ovh-b` is suspect; pin away
from it for anything ISA-sensitive.
