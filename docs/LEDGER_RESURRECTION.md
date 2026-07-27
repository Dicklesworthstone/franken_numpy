# Ledger Resurrection Audit — `docs/NEGATIVE_EVIDENCE.md`

## Corrected six-class hand audit (2026-07-27 model-integrity review)

This section supersedes both the earlier VioletOwl map and every preliminary
count, queue, taxonomy extension, and rerun table retained below it. The audit follows
`/data/projects/frankenfs/docs/LEDGER_RESURRECTION.md` and the fleet broadcast
verbatim: there are exactly six verdict classes. Regexes only built the review
queue; they did not assign a final verdict.

Snapshot `f4d70a5e` contains 1,005 `##` entries. The mechanical screen selected
146 possible negative rows. L1370 was a KEEP heading containing the word
REJECT, leaving 145 rows; every one of those 145 rows was then read in full by
hand. Thirty-six were surveys, behavior-only blockers, superseded reports,
or other non-lever rows and are listed separately rather than forced into a
verdict. The remaining 109 rejected levers received one of the six allowed
classes.

### Taxonomy

| Class | Verbatim decision rule |
|---|---|
| `VALID-PROFILE` | Rejected before any source edit on a named frame with non-zero self-time plus a computed Amdahl ceiling. |
| `VALID-MECHANISM` | No A/A null, but refuted on a counted mechanism: instructions, cycles, syscalls, allocations, faults, or equivalent counted work unchanged. |
| `VALID-AB` | A/B with a recorded A/A null, with the effect inside that null. |
| `VOID-CV` | Killed only by a `cv < 5%` gate. |
| `VOID-ZEROSELF` | The target frame had approximately zero self-time in the profile the benchmark actually ran. |
| `VOID-NONULL` | Near-1.0 A/B, no A/A null, and no counted mechanism. |

No `VALID-MARGIN`, `VOID-UNMEASURED`, `SOUND`, or `SOUND-WEAK` class is used.
An interrupted or never-admitted run cannot establish negative lever evidence;
where such a report nevertheless carried a lever rejection, it is
conservatively `VOID-NONULL` unless a counted mechanism independently proves
the rejection.

### Counts

| Metric | Count |
|---|---:|
| Ledger entries parsed | 1,005 |
| Mechanical negative candidates | 146 |
| Heading parse artifact | 1 |
| Rows read and adjudicated by hand | 145 |
| Non-lever exclusions | 36 |
| **Rejected levers classified** | **109** |
| `VALID-PROFILE` | 0 |
| `VALID-MECHANISM` | 35 |
| `VALID-AB` | 3 |
| `VOID-CV` | 5 |
| `VOID-ZEROSELF` | 0 |
| `VOID-NONULL` | 66 |
| **VOID total** | **71 / 109 = 65.1%** |
| Rejected levers carrying a self-reported executing-ELF SHA-256 | **0 / 109 = 0.0%** |

The epidemic is therefore `VOID-NONULL`, not the CV gate: 66 of the 71 VOID
rows are undecidable near-unity A/B reports with neither a null nor a counted
mechanism. `VALID-MECHANISM` was applied in both directions and rescued 35
rows; a null cannot change a counter proving that the proposed work was not
removed.

The previous `21 / 122` binary-provenance count was not an executing-ELF
count. Its regex accepted source hashes, benchmark-source hashes, truncated
runner notes, and hashes in mixed KEEP/blocker reports. In the classified
rejected-lever population, no row self-reported the hash of the ELF that
actually executed. This distinction is now enforced by the preflight on one
line: an ELF marker cannot borrow an unrelated 64-hex source hash elsewhere in
the row.

### Complete hand-verdict map

Line identifiers refer to the `f4d70a5e` ledger snapshot. This is the compact,
auditable record of all 145 hand decisions.

```text
VALID-PROFILE (0): none

VALID-MECHANISM (35):
L3451 L6916 L6935 L7161 L10440 L10664 L10788 L11361 L11516
L12055 L12415 L12594 L13009 L13322 L14217 L14554 L14778 L15292
L15734 L17360 L18606 L19644 L20171 L20622 L22419 L22899 L23298
L23495 L23601 L23611 L23865 L24771 L26269 L27851 L30207

VALID-AB (3):
L1160 L1301 L8980

VOID-CV (5):
L193 L395 L477 L806 L10313

VOID-ZEROSELF (0): none

VOID-NONULL (66):
L2110 L2248 L2346 L2392 L2447 L3158 L3937 L4002 L4509 L5256
L5559 L5666 L5706 L5888 L5959 L6190 L6319 L6375 L6427 L6985
L7353 L7375 L7395 L7415 L7510 L7569 L7624 L7664 L7766 L7851
L7892 L8108 L9292 L12395 L14671 L14708 L15124 L15609 L15929
L16074 L16118 L17190 L18719 L19307 L19462 L20012 L20285 L20571
L20761 L20868 L21279 L22496 L22553 L22773 L23630 L23648 L24670
L24706 L25156 L25433 L25481 L27567 L27578 L27925 L28081 L30156

NON-LEVER EXCLUSIONS (36):
L1121 L1562 L5752 L7247 L7876 L9201 L9986 L11286 L13168 L14250
L15055 L15527 L15564 L15888 L16842 L16952 L17137 L17561 L19718
L19853 L24916 L25406 L27096 L27108 L27190 L28175 L28428 L28455
L28486 L28509 L28524 L28542 L28562 L28634 L29615 L29731
```

### Resurrection reruns and the corrected rank gap

VOID rows were ranked by the target frame's self-time in the profile attached
to that row, not by a scraped ratio. The prior map wrongly treated L2392's
42%-profiled manual integer formatter as `VALID-MECHANISM`, even though it
recorded neither an A/A null nor a counted removal mechanism. That row is the
true rank 2. Four of the corrected top five were rerun on quiet `vmi1227854`,
pinned to CPU 6, with all eight worker slots held; the assigned negative-usecols
directional lever, now rank 6, was also rerun. Each completed invocation printed
the executing ELF hash first, ran base/base before base/candidate in the same
process, and gated only on the bootstrapped 95% median-ratio CI. CV was
provenance only.

| Corrected rank | Row and profiled target | Executing ELF SHA-256 | A/A median, CI95 | Effect median, CI95 | Re-audit status |
|---:|---|---|---|---|---|
| 1 | L2447 C-order element-step odometer; decode = 44.0% | `2d8f85747cc703b899e54ec4fcb17125d07a3db9f1efb93f327701f74598ee8e` | 1.022770 [1.006650, 1.058581] | 1.016927 [1.002447, 1.051011] | `VALID-AB`; repeat below also inside null |
| 2 | L2392 `tofile_text` integer formatting; named fmt frames approximately 42% | — | prior row had no A/A | prior A/B 1.00-1.10x overlapping | **PENDING: still `VOID-NONULL`** |
| 3 | L2346 usecols scatter; fn + allocator approximately 21% | `05ca263ea39f01bdd3d68d62088555861a54060fc31273e4650c9a23c051cb47` | 1.002036 [0.980107, 1.019245] | 1.034835 [1.025527, 1.039039] | `VALID-AB`; below required 1.039787 |
| 4 | L477 Generator Zipf invariant terms; 12.93% | `f25d6a15885cc81796958247664798850be17702934a2f44c77d04a98e29f9eb3` | 0.993360 [0.986830, 0.997768] | **1.364282 [1.348971, 1.396232]** | **KEEP** |
| 5 | L2110 noncentral chi-square fixed gamma cache; 12.11% | `08b0ceca45d80e88d19f170e0b76ef4de80d57b3e07b6ace2430c8ba8fb55508` | 0.989887 [0.978433, 1.007559] | **1.108215 [1.076852, 1.132333]** | **KEEP** |
| 6, assigned directional lever | L806 negative-usecols tail ring; collect = 10.67% | `5b3e2e3cd3079ac8bcba841e01e4d4003cdf95d55c00442ad8f7e43f5f762a13` | 1.014925 [0.983995, 1.056524] | **1.171830 [1.111937, 1.234068]** | **KEEP** |

Rank 1 was repeated from a freshly built second ELF,
`12cf201485beb3152be2da628d1653532b5f4149e74ec3ad34af58005f222009`:
A/A 0.990316 [0.960183, 1.043890], effect 1.020997
[0.987467, 1.051449]. The direction flipped within the broad null between the
two runs, so the production trial was reverted. Exact checksum on both runs:
`52a9418a7fd74354`.

Rank 3 exact checksum was `a2e078ba6c280f6d`; no production source was
changed. Rank 4 checksum was `7530e1225533b088`, rank 5
`983fa2c568e726e7`, and rank 6 `fe47f366bfb808d2`.

Corrected top-five status: **2 KEEP, 2 VALID-AB, 1 pending**. The fifth
completed rerun was rank 6 and remains a valid directional KEEP; it cannot be
counted as completion of the top-five assignment.

Concrete retry predicates:

- L2447: retry only after counters on the exact production `Nditer::next`
  candidate prove at least 10% fewer instructions or cycles and a pinned A/A
  half-width at most 1%; do not retry from wall direction alone.
- L2392: reconstruct the same manual integer formatter only in a harness that
  self-reports its executing ELF and runs an A/A null before the effect in the
  same invocation. Keep it closed unless the median-CI effect clears twice the
  null half-width; frame percentage alone is not a removal count.
- L2346: retry the exact retained scatter vehicle only when the A/A half-width
  is below 1.74% or the effect reaches 1.039787; a different vehicle first
  needs a fresh profile and counted work-removal proof.
- L477: closed for fixed-parameter Generator Zipf. Reopen only a different
  Zipf regime with at least 5% newly profiled self-time.
- L2110: closed for the measured fixed-shape regime. Reopen only a distinct
  shape or dtype with at least 5% newly profiled self-time and exact stream
  proof.
- L806: closed for the measured negative-usecols scanner. Reopen only a
  separately profiled scanner primitive, not another full-row/tail-ring
  comparison.

The allocation addendum's other named directional lever, selected-bool
`loadtxt(usecols)` (L193), was independently re-won on the same pinned worker.
The final production path reported ELF
`2377aeca302c2dcdc2cfa167fb85032c2d8b015606ca8247ede0d437e2143a55`
(47,322,576 bytes): A/A 1.034931 [0.983311, 1.096850], effect
**3.496057 [3.405198, 3.715384]**, checksum `30fb6a0b5c0da785`.
The required two-null-width delta was 0.193701, so this is a decisive KEEP
under the bootstrapped median-CI gate despite effect CV 17.179%. The old
CV-only rejection is superseded.

### Institutional gate

Ledger integrity is now enforced twice. The Rust preflight can query a proposed
surface and prints any prior retry predicate; staged audit exits 2 for a new
REJECT lacking either A/A evidence or a counted mechanism, and for a KEEP
lacking an executing-ELF SHA-256. The tracked pre-commit hook runs that staged
audit. The independent conformance test and shell preflight merged from the
peer lane provide a second CI path.

---

## Superseded preliminary screen retained as provenance

The material below records earlier mechanical screens and provisional queues.
It is intentionally retained for auditability, but its added classes, counts,
rankings, and yield are not authoritative.

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

> ## ⚠ SUPERSEDED — do not quote the counts below
>
> Everything from here down is earlier audit provenance. **The corrected audit
> at the top of this file is the only count/map to cite: 71 VOID / 109
> classified levers = 65.1%.** Both the intermediate 68/122 map and the
> preliminary 99/146 screen are superseded.
>
> The 2026-07-27 review corrected three judgment errors:
>
> 1. Behavior-only and survey/blocker reports are exclusions, not performance
>    mechanism evidence.
> 2. A near-unity A/B with mechanism prose but no counted work and no A/A null
>    remains `VOID-NONULL`.
> 3. A source or benchmark hash is not an executing-ELF self-report.
>
> The sections below are retained as the working record of how the numbers were
> derived, including §3.1's screen-bug account. Shipped measurements carrying
> their own A/A, executing-ELF hash, and identity proof remain valid; only the
> classification and queue claims above were corrected.

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

### Superseded hand-screen appendix from the preliminary audit

## 4. Per-row audit appendix

`not recorded` means the original row supplied no admissible value for that
field. `recorded` means the row supplied the artifact but the compact appendix
defers the full value to the source ledger entry. The line numbers identify the
audited `f4d70a5e` snapshot.

| Entry | Ratio claimed | Null floor at the time | Target-frame self-time | Binary sha? | Verdict |
|---|---:|---|---:|:---:|---|
| L193 — selected bool `loadtxt(usecols)` direct parse | 3.09–3.69× | 0.988–1.045 | 5.80% (+15.06% alloc) | no | **VOID** |
| L395 — RandomState legacy Zipf batch constants | 1.27× | recorded | 11.58% max recorded | yes | **SOUND** |
| L477 — Zipf batch-invariant rejection terms | 1.26–1.31× | recorded | 12.93% max recorded | yes | **SOUND** |
| L806 — all-negative signed `loadtxt` usecols tail ring | 1.82–1.98× | 0.780–1.087 | 10.67% (+26.76% search) | no | **VOID** |
| L1121 — TSQR behind `fnp.linalg.qr` | not recorded | not recorded | not recorded | no | **SOUND** |
| L1160 — MR=4 × NR=8 GEMM tile geometry | 2.3–39% loss | 1.0002 | not recorded | yes | **SOUND** |
| L1301 — explicit-SIMD packed-f64 GEMM register tile | no effect | recorded | profile recorded | yes | **SOUND** |
| L1562 — Python-surface `loadtxt` baseline | 0.574× | not recorded | profile recorded | yes | **SOUND** |
| L2110 — `noncentral_chisquare` gamma-shape cache | 1.085× | not recorded | profile recorded | yes | **VOID** |
| L2248 — legacy `RandomState` gamma-shape cache | 0.973× | not recorded | profile recorded | yes | **VOID** |
| L2346 — direct-scatter `usecols` rows | 0.86–0.95× | recorded | profile recorded | yes | **SOUND** |
| L2392 — `tofile_text` manual integer formatting | 1.00–1.10× | recorded | profile recorded | yes | **VOID** |
| L2447 — C-order element-step multi-index odometer | 1.009× | not recorded | profile recorded | no | **VOID** |
| L3158 — von Mises kappa-only term hoist | 1.02–1.04× | recorded | profile recorded | yes | **SOUND** |
| L3451 — block-rotation transpose tiling | 0.33–1.21× | recorded | profile recorded | yes | **SOUND** |
| L3937 — singleton-axis `flip` clone | 1.18× | not recorded | profile recorded | no | **VOID** |
| L4002 — Exact-U64 singleton cumulative fold | 1.03× | not recorded | profile recorded | no | **VOID** |
| L4509 — I8 `fromfile` direct byte-slice decode | 1.17% slower | not recorded | profile recorded | no | **VOID** |
| L5256 — one-pass slice `choice(replace=true)` gather | 1.35× slower | not recorded | profile recorded | no | **SOUND-WEAK** |
| L5559 — duplicate-active-stride overlap proof | not recorded | not recorded | profile recorded | no | **VOID** |
| L5666 — direct C-order `nditer` chunk range | not recorded | not recorded | profile recorded | yes | **VOID** |
| L5706 — hoisted `loadtxt(usecols)` row plan | 1.10× | not recorded | profile recorded | yes | **VOID** |
| L5752 — exact-pentadiagonal Cholesky recurrence | not recorded | not recorded | profile recorded | yes | **SOUND** |
| L5888 — direct-collect serial PCG64 fill | 1.16× slower | not recorded | profile recorded | yes | **SOUND** |
| L5959 — F-order identity broadcast overlap proof | not recorded | not recorded | profile recorded | yes | **VOID** |
| L6190 — `standard_exponential` backend dispatch hoist | 1.79× slower | not recorded | profile recorded | no | **SOUND-WEAK** |
| L6319 — singleton-source multi-axis `broadcast_to` fill | not recorded | not recorded | profile recorded | no | **VOID** |
| L6375 — complex `nancumprod` axis-0 proof | unmeasured | not recorded | not recorded | no | **VOID** |
| L6427 — int32 flat-sort small-pool SIMD regate | not recorded | not recorded | not recorded | no | **VOID** |
| L6916 — edges-array histogram per-element search | not recorded | not recorded | not recorded | no | **SOUND-WEAK** |
| L6935 — complex `np.select` arms | not recorded | not recorded | profile recorded | no | **SOUND-WEAK** |
| L6985 — `ufunc.reduceat` probe | not recorded | not recorded | not recorded | no | **SOUND-WEAK** |
| L7161 — native parallel `packbits` | 1.040× | not recorded | not recorded | no | **SOUND-WEAK** |
| L7247 — uniform-NaN f64 value-sort arms | 0.97–1.13× | not recorded | not recorded | no | **SOUND-WEAK** |
| L7353 — exact unit-interval `uniform` route | not recorded | not recorded | profile recorded | no | **SOUND-WEAK** |
| L7375 — single-dispatch `ArrayStorage::get_f64` | not recorded | not recorded | profile recorded | no | **SOUND-WEAK** |
| L7395 — single-input `broadcast_shapes` identity path | not recorded | not recorded | profile recorded | no | **SOUND-WEAK** |
| L7415 — later-Complex128 `common_type` terminal | not recorded | not recorded | profile recorded | no | **SOUND-WEAK** |
| L7510 — rank-0-to-4 `element_count` specialization | not recorded | not recorded | profile recorded | no | **SOUND-WEAK** |
| L7569 — scalar identity `broadcast_shape` path | 0.76× | not recorded | profile recorded | no | **SOUND-WEAK** |
| L7624 — direct I64-to-I32 cast construction | unmeasured | not recorded | profile recorded | yes | **VOID** |
| L7664 — sub-threshold PCG uniform one-pass route | −6.6% to +7.3% | not recorded | profile recorded | yes | **SOUND** |
| L7766 — direct F64-to-I32 cast construction | 1.09–1.24× | not recorded | not recorded | no | **SOUND-WEAK** |
| L7851 — single-pass C-contiguity predicate | 1.14–1.25× | not recorded | not recorded | no | **SOUND-WEAK** |
| L7876 — integer array-q percentile/quantile stale reject | 1.80× | not recorded | not recorded | no | **SOUND-WEAK** |
| L7892 — direct `isize` contiguous-stride buffer | 0.97–1.05× | recorded | profile recorded | no | **SOUND** |
| L8108 — flat multi-q keepdims native unlock | 0.852× | not recorded | not recorded | no | **SOUND-WEAK** |
| L8980 — `accumulate_extremum_typed` static dispatch | +25.6% regression | recorded | 51.27% | no | **VOID** |
| L9201 — core `reduce_fold` last-axis row bands | 1.33–1.47× | recorded | not recorded | no | **SOUND** |
| L9292 — `UFuncArray` last-axis product row bands | 2.08× cross-worker | recorded | profile recorded | no | **SOUND** |
| L9986 — f16 union/setxor profile-blocked arm | 0.736× | recorded | profile recorded | yes | **VOID** |
| L10313 — legacy “irreducible wall” integrity audit | not recorded | not recorded | 0.89% max recorded | no | **SOUND** |
| L10440 — cov Gram pairing schedule | 0.91× | not recorded | 66.1% | yes | **SOUND** |
| L10664 — cov/corrcoef zero-copy Gram and fused mirror | not recorded | not recorded | not recorded | no | **SOUND-WEAK** |
| L10788 — fused symmetric mirror | 1.27× kernel / 1.51× pipeline loss | not recorded | not recorded | no | **SOUND-WEAK** |
| L11286 — narrow ASCII U8 union helper | not recorded | not recorded | profile recorded | no | **SOUND-WEAK** |
| L11361 — dense-domain row-unique occupancy table | not recorded | not recorded | profile recorded | no | **SOUND-WEAK** |
| L11516 — float median radix-select | 0.68× | not recorded | not recorded | no | **SOUND-WEAK** |
| L12055 — float16 sort via f32 widening | 0.75× | not recorded | not recorded | no | **SOUND-WEAK** |
| L12395 — 2-D string-row `unique` memcmp sort | 1.06× | not recorded | not recorded | no | **SOUND-WEAK** |
| L12415 — 2-D per-lane string sort | 1.00× | not recorded | not recorded | no | **SOUND-WEAK** |
| L12594 — packed-u64 narrow string keys | 1.70× vs NumPy / 4.3× slower than shipped | not recorded | profile recorded | no | **SOUND-WEAK** |
| L13009 — parallel bordered-row `pad` copy | 0.87× f64 / 1.42× i32 | not recorded | not recorded | no | **SOUND-WEAK** |
| L13168 — local Python 3.14 tooling blocker | unmeasured | not recorded | not recorded | no | **SOUND-WEAK** |
| L13322 — parallel float set-op sort/dedup/merge | 0.70× | not recorded | not recorded | no | **SOUND-WEAK** |
| L14217 — `strings.startswith`/`endswith` | parity | recorded | not recorded | no | **SOUND** |
| L14250 — f32/f16 `sinc` | not bit-exact | not recorded | not recorded | no | **SOUND** |
| L14554 — hashed float `unique` metadata | not recorded | not recorded | not recorded | no | **SOUND-WEAK** |
| L14671 — flat `compress`/`extract` density gate | not recorded | not recorded | profile recorded | no | **SOUND-WEAK** |
| L14708 — sparse-identity inverse extension | not recorded | not recorded | profile recorded | no | **SOUND-WEAK** |
| L14778 — native GEMM window 1536→1024 | not recorded | not recorded | profile recorded | no | **SOUND-WEAK** |
| L15055 — BLAS-backed-op gates | unmeasured | not recorded | not recorded | no | **VOID** |
| L15124 — 2-D `tensordot(axes=1)` gate | not recorded | not recorded | profile recorded | no | **SOUND-WEAK** |
| L15292 — packed-GEMM A-panel packing | not recorded | not recorded | profile recorded | no | **SOUND-WEAK** |
| L15527 — moderate-n `batch_inv` FLOP-count wall | not recorded | not recorded | not recorded | no | **SOUND-WEAK** |
| L15564 — dtype/FFT/indexing/linalg convergence survey | 1.27× residual | not recorded | not recorded | no | **SOUND-WEAK** |
| L15609 — f64 unary serial-regime rewrite | ~1.1× | not recorded | not recorded | no | **SOUND** |
| L15734 — `TRIDIAG_MATVEC_PAR_MIN` 1024→256 | 1.41× regression | not recorded | profile recorded | no | **SOUND-WEAK** |
| L15888 — `diag_indices` repro | not recorded | not recorded | profile recorded | no | **SOUND-WEAK** |
| L15929 — f64 `frexp` Python-overhead trims | not recorded | not recorded | not recorded | no | **SOUND-WEAK** |
| L16074 — duplicate-heavy f64 `unique` HashSet | not recorded | not recorded | profile recorded | no | **SOUND-WEAK** |
| L16118 — f64 `unique` output-buffer dedup | not recorded | not recorded | profile recorded | no | **SOUND-WEAK** |
| L16842 — einsum matmul-shaped contraction verify | not recorded | not recorded | not recorded | no | **SOUND-WEAK** |
| L16952 — post-nanargmax broad sweep | not recorded | not recorded | not recorded | no | **SOUND-WEAK** |
| L17137 — SBR stage-1-only `eigvalsh` | not recorded | not recorded | not recorded | no | **SOUND-WEAK** |
| L17190 — `eigvalsh`/`cond` 128 values-only route | not recorded | not recorded | not recorded | no | **SOUND-WEAK** |
| L17360 — `eigvalsh(128)` Sturm bisection | not recorded | not recorded | not recorded | no | **SOUND-WEAK** |
| L17561 — fused-parallel f64 sqrt stale reject | up to 8× | not recorded | not recorded | no | **SOUND-WEAK** |
| L18606 — `sqrt` zero-init-tax diagnosis | 1.5× loss | not recorded | not recorded | no | **SOUND-WEAK** |
| L18719 — moderate-batch `batch_inv` | 1.4–1.9× loss | not recorded | not recorded | no | **SOUND-WEAK** |
| L19307 — `eigvalsh(128)` tail-local reducer | not recorded | not recorded | not recorded | yes | **SOUND** |
| L19462 — symmetric `cond_nxn` scan/sort elision | neutral | not recorded | not recorded | no | **SOUND-WEAK** |
| L19644 — zero-copy short-kernel convolve/correlate | regression | not recorded | not recorded | no | **SOUND** |
| L19718 — short-kernel large-N convolve/correlate | not recorded | not recorded | profile recorded | no | **SOUND-WEAK** |
| L19853 — symmetric spectral-gap sweep | not recorded | not recorded | profile recorded | no | **SOUND-WEAK** |
| L20012 — linalg spectral small-lever sweep | not recorded | not recorded | profile recorded | yes | **SOUND** |
| L20171 — generated 8-lane SoA batch-Cholesky | not recorded | not recorded | profile recorded | no | **SOUND-WEAK** |
| L20285 — medium Cholesky update threshold | not recorded | not recorded | not recorded | no | **SOUND-WEAK** |
| L20571 — `cholesky_nxn` const specialization | not recorded | not recorded | not recorded | no | **SOUND-WEAK** |
| L20622 — batch-Cholesky finite-validation hoist | not recorded | not recorded | not recorded | no | **SOUND-WEAK** |
| L20761 — batch-Cholesky f64x4 across-lanes SIMD | regression | not recorded | not recorded | no | **SOUND-WEAK** |
| L20868 — batch-Cholesky allocation elimination | 5–8× loss | not recorded | not recorded | no | **SOUND-WEAK** |
| L21279 — fnp-python f64 einsum diagonal shortcut | not recorded | not recorded | not recorded | no | **SOUND-WEAK** |
| L22419 — matrix-column norm NaN prefilter | not recorded | not recorded | not recorded | no | **SOUND-WEAK** |
| L22496 — matrix-column norm 8-column strip mine | not recorded | not recorded | not recorded | no | **SOUND-WEAK** |
| L22553 — batch-Cholesky blocked ordered-dot helper | inside null band | not recorded | not recorded | no | **VOID** |
| L22773 — terminal 2×2 `eigvalsh` QR deflation | regression | not recorded | not recorded | no | **SOUND-WEAK** |
| L22899 — diagonal `eigvalsh` QR skip | regression | not recorded | not recorded | no | **SOUND-WEAK** |
| L23298 — char/string strip+pad+concat | not recorded | not recorded | not recorded | no | **SOUND-WEAK** |
| L23495 — flat `nanargmax`/`nanargmin` | not recorded | not recorded | not recorded | no | **SOUND-WEAK** |
| L23601 — int64 `ptp` axis parallel-gate retune | ~1.09× | not recorded | not recorded | no | **VOID** |
| L23611 — int max/min axis parallel-gate retune | ~0 gain | not recorded | not recorded | no | **SOUND-WEAK** |
| L23630 — f64 cheap-unary raw-slice rewrite | ~0 gain | not recorded | not recorded | no | **SOUND-WEAK** |
| L23648 — f64 1-D `diff` raw-slice rewrite | regression | not recorded | not recorded | no | **SOUND-WEAK** |
| L23865 — batched-SIMD-SoA Cholesky | 1.5–3× slower | not recorded | not recorded | no | **SOUND** |
| L24670 — stacked-Cholesky 8×8 wrapper hoist | not recorded | not recorded | profile recorded | no | **SOUND-WEAK** |
| L24706 — parallel f64 `where(cond, arr, scalar)` | 1.14× | not recorded | not recorded | no | **SOUND-WEAK** |
| L24771 — parallel f64/i64 `repeat` | 1.5× f64 loss | not recorded | not recorded | no | **SOUND-WEAK** |
| L24916 — post-`searchsorted` boundary sweep | near noise | not recorded | profile recorded | no | **SOUND-WEAK** |
| L25156 — parallel `log2`/`log10`/`exp2` | loss | not recorded | not recorded | no | **SOUND-WEAK** |
| L25406 — `sort_complex` comparator-only route | loss | not recorded | profile recorded | no | **SOUND-WEAK** |
| L25433 — 2-D native-matmul cap 1024→2048 | loss | not recorded | not recorded | no | **SOUND-WEAK** |
| L25481 — native `char.add`/`strings.add` | inside null band | recorded | not recorded | no | **VOID** |
| L26269 — parallel `choose` typed gather | ~0 gain | not recorded | not recorded | no | **SOUND-WEAK** |
| L27096 — float16 `hypot` | not bit-reproducible | not recorded | not recorded | no | **SOUND** |
| L27108 — float16 sort | signed-zero order nondeterministic | not recorded | not recorded | no | **SOUND** |
| L27190 — float32 `vander`/`sinc` | libm divergence | not recorded | not recorded | no | **SOUND-WEAK** |
| L27567 — `char.count` | loss | not recorded | not recorded | no | **SOUND-WEAK** |
| L27578 — `char.ljust`/`rjust` | ~0 gain | not recorded | not recorded | no | **SOUND** |
| L27851 — native parallel `unpackbits` | 0.44× | not recorded | not recorded | no | **SOUND-WEAK** |
| L27925 — native parallel float16 `spacing` | 1.03× | not recorded | not recorded | no | **SOUND** |
| L28081 — complex last-axis max/min accumulate | not recorded | not recorded | not recorded | no | **SOUND-WEAK** |
| L28175 — complex non-last reductions | bit-exactness not tractable | not recorded | not recorded | no | **SOUND** |
| L28428 — percentile/quantile array-q with axis | 0.64× loss→parity | not recorded | not recorded | no | **SOUND-WEAK** |
| L28455 — native f64 2-D GEMM regime gate | 0.32–0.58× loss removed | not recorded | not recorded | no | **SOUND-WEAK** |
| L28486 — serial-BLAS gate extension | 0.32–0.63× loss removed | not recorded | not recorded | no | **SOUND-WEAK** |
| L28509 — nanpercentile/nanquantile extract tax | 0.88–0.95×→parity | not recorded | not recorded | no | **SOUND-WEAK** |
| L28524 — non-last-axis weighted average | 0.09–0.15×→parity | not recorded | not recorded | no | **SOUND-WEAK** |
| L28542 — tuple-axis extract tax | 0.66–0.84×→parity | not recorded | not recorded | no | **SOUND-WEAK** |
| L28562 — `unique_values` | not byte-reproducible | not recorded | not recorded | no | **SOUND** |
| L28634 — large-N `logspace`/`geomspace` | 0.84–0.85×→parity | not recorded | not recorded | no | **SOUND-WEAK** |
| L29615 — native complex128 `sqrt` | last-ULP divergence | not recorded | not recorded | no | **SOUND** |
| L29731 — tie-heavy flat-f32 `argsort` | 1.05–1.2× loss | not recorded | not recorded | no | **SOUND-WEAK** |
| L30156 — paired cov/corrcoef triangular Gram blocks | unmeasured | not recorded | profile recorded | no | **VOID** |
| L30207 — native TSQR `lstsq` Python surface | no win | not recorded | not recorded | no | **SOUND-WEAK** |

Appendix totals: **24 VOID**, **31 SOUND**, and **90 SOUND-WEAK**; 145
adjudicated negative rows after excluding the L1370 heading parse artifact.

---

## 5. Ranked re-run queue

Ranked by target-frame self-time, per the campaign rule.

| # | Entry | Line | Class | Self-time | Effect on record | Status |
|---:|---|---:|---|---|---|---|
| 1 | selected-bool `loadtxt(usecols)` direct parse | 300 | `VOID-CV` | 8.14% `cfree` + 6.92% `malloc` + 5.80% owned-`Vec<String>` collect | **3.09–3.69×**, A/A nulls **0.988–1.045** | **RE-WON at 3.636795×, shipped `c828e871`** — see §5.1 |
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

## 6. Yield

| Metric | Count |
|---|---:|
| Entries parsed | 1,005 |
| REJECT audited | 146 |
| VOID | 99 (67.8%) |
| Re-run under the corrected harness | 6 |
| **Re-won** | **5** |
| Best re-won ratio | 167.610955× |

Scoreboard line: `franken_numpy | 1005 | 146 | 99 | 67.8 | 6 | 5 | 167.610955x`

**Provenance caveat:** four of the five re-wins are the cod lane's
(`VioletOwl`) and were measured but not yet pushed to `origin/main` at the time
of writing — Generator Zipf 1.296191×, RandomState Zipf 1.249485×, the tail
ring 1.661938×, and rank 5 at 167.610955×. Those are relayed, not verified
against a landed commit. Rank 1 below is landed and verifiable.

### 5.1 Rank 1 — RE-WON and shipped (`c828e871`)

> **MODEL-INTEGRITY CORRECTION (2026-07-27):** the `c828e871` p10/p90 decision
> below was not a bootstrapped median-CI gate. The production route and byte
> proof stand; the decisive bank is the later executed-ELF invocation
> `4527aadbd8e543310622ac9f737bef8963273b57ed1cb1877a8c82fb3284b4fd`,
> with A/A 1.001104 [0.994382, 1.002883] and effect
> 3.465137 [3.255616, 3.666974]. The allocator-composability claim below is also
> withdrawn: no exact FrankenNumPy allocation/fault counts were recorded.

Pinned worker `vmi1227854`, release with LTO off, 20 retained observations,
arms interleaved ABBA/BAAB, A/A null and A/B in the same invocation. The binary
self-reported its own SHA-256 —
`1050fcc0abc52b7f6221cbfd82751a8235581d0bc21a819e7b59b90f1ad78135`
(47,241,304 bytes) — via the `gated_main` hook from §6.3, on its first real use.

| arm | median | p10 | p90 | cv |
|---|---:|---:|---:|---:|
| former (negative `usecols`, owned-token path) | 8.227393 ms | | | 15.690% |
| candidate (direct path) | 2.317728 ms | | | 16.525% |
| **A/B effect** | **3.636795×** | 3.235359 | 4.205890 | 11.241% |
| A/A null | 0.979394 | 0.882273 | **1.102141** | 9.600% |

**`verdict=WIN`** — `effect.median` clears `null.p90` by 3.3×, far outside the
2× margin; `effect_above_one = 20/20`; the null brackets unity, so it is not a
biased null.

**Independently corroborated.** The cod lane implemented this lever separately
and measured the production path at **3.496057× [3.405198, 3.715384]** (ELF
`2377aeca…`). Each median sits inside the other's interval — two
implementations, two harnesses, two runs on the same pinned worker, converging.
The merge kept one live code path; the duplicate implementation is in history
at `e4cad9a4` and is not in the tree.

Do not confuse either figure with the **1.918783×** in the canonical section's
Lane M table: that arm is a pure-Rust replica of the parse loop in
`fnp-io/benches/criterion_io.rs`, which measures the kernel in isolation. The
3.5–3.6× figures measure the real `fnp.loadtxt` public entry point. Different
quantities, both sound; quote the one whose base matches the claim being made.

**Every arm's `cv` is 9.6–16.5%. Under the predicate that originally rejected
this row, it would have been rejected a second time.** That is the cleanest
demonstration in this repo that the gate was the problem rather than the lever.
The measured 3.6368× sits inside the original row's 3.09–3.69× range, so this
reproduces the earlier observation rather than replacing it.

Same-binary control: both arms are `fnp.loadtxt` on one 8192×16 file under one
ELF selecting identical columns — the base via negative `usecols`, which the
direct path deliberately declines. Column 7 is poisoned with a non-bool token
selected by neither arm, which also pins the correctness requirement that NumPy
never parses unselected columns. Byte-identity former == candidate == NumPy was
asserted before any timing.

**Mechanism correction to the original row:** the win is *not* skipping the
parse of unselected tokens — the former path never parsed them either, since
its dtype loop walked only the already-narrowed selection. The win is ~131k
`String` allocations never made per call, matching the profile's ~22.8% of
self-time in malloc/cfree/collect/clone. That is sub-128 KiB free-list churn,
so it is independent of and composable with the >128 KiB `mmap` tax recorded in
`deadlock-audit-tztko`.

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

### Superseded provisional re-run summary

| Entries audited | 1,005 |
| Negative rows in scope | 146 (145 after one parse-artifact exclusion) |
| VOID (re-measurable) | 24 (16.6%) |
| — high-EV: decidable effect already on record, killed by the CV gate | 2 |
| — unmeasured, and the blocking instrument has since been fixed | 5 |
| — unmeasured, other blocker | 5 |
| — sub-noise micro-levers (void but low-EV) | 12 |
| SOUND-WEAK: decidable margin, provenance incomplete | 90 |
| Re-run under the corrected harness | see §7 |
| Re-won | see §7 |

---

## 7. Superseded re-run status

> The “complete top five” table below used the preliminary queue. The corrected
> queue has L2392 pending at rank 2 and the tail ring at rank 6; corrected
> top-five status is 2 KEEP, 2 VALID-AB, 1 pending. This table is retained only
> as provenance for the runs it actually lists.

Complete for the ranked top five. The active campaign targets self-report the
executing ELF SHA-256, run base/base before base/candidate in the same
invocation, and decide from a bootstrapped median-ratio CI with a 2x A/A
margin. CV remains provenance only.

| Rank | Result | Effect ratio / CI95 | A/A ratio / CI95 | Disposition |
|---:|---|---|---|---|
| 1 | selected-bool `loadtxt` direct parse | 1.918783 / [1.805047, 2.016407] | 0.998879 / [0.996542, 1.004521] | **KEEP** |
| 2 | all-negative signed-`usecols` tail ring | 1.171830 / [1.111937, 1.234068] | 1.014925 / [0.983995, 1.056524] | **KEEP** |
| 3 | complex `nancumprod` axis 0 | 1.563101 / [1.419484, 1.913478] | 0.998477 / [0.997407, 1.000482] | **KEEP** |
| 4 | paired cov/corrcoef Gram blocks | 0.996469 / [0.986097, 1.005434] | 0.999706 / [0.987268, 1.010215] | **REJECT / UNDECIDED** |
| 5 | direct C-order `nditer` chunk | 208.071281 / [206.335116, 210.587810] | 0.987308 / [0.979785, 1.001212] | **KEEP** |

Ranks 1-2 were re-decided under the allocation addendum's Lane M contract on
the same quiet, CPU-pinned worker: `vmi1227854`, all eight RCH slots reserved
and zero remaining. Their executable hashes were respectively
`1951dabb76d3a0215e22150a3c5144f495827e76062c5870ab0721e74d4c8fb1`
and
`5b3e2e3cd3079ac8bcba841e01e4d4003cdf95d55c00442ad8f7e43f5f762a13`.
The full per-row evidence, checksums, and retry predicates are in
`docs/NEGATIVE_EVIDENCE.md`.

**Yield: 4 of 5 ranked VOID rows re-won.** The fifth produced a real null,
not another harness verdict.

rch worker `ovh-b` **SIGILLs on the workspace `+avx2` baseline** — the
`zerocopy v0.8.48` build script dies with `signal: 4` under
`cargo clippy --workspace`. This is an independent franken_numpy reproduction of
frankenscipy `hhr7j` and franken_networkx's global-AVX2 rejection (campaign
§3b). Any franken_numpy measurement that landed on `ovh-b` is suspect; pin away
from it for anything ISA-sensitive.

### Superseded standing-rule draft

`zerocopy v0.8.48` build script died with `signal: 4, SIGILL` under
`cargo clippy --workspace`. This is an independent franken_numpy reproduction
of frankenscipy `hhr7j` and franken_networkx's global-AVX2 rejection. Any
franken_numpy measurement that landed on `ovh-b` is suspect, and every
ISA-sensitive run must pin away from it until the fleet fixes worker
capability detection.

---

## 8. Standing rules this audit adds

1. **A ratio inside 0.905–1.105 is void unless the entry supplies a mechanism.**
   Bandwidth saturation, critical-path analysis, and bit-exactness blockers are
   mechanisms; "we measured 1.03× and moved on" is not.
2. **Behavior/bit-exactness rejections are permanently sound.** Do not queue
   them for resurrection; no gate change can rescue a non-reproducible result.
3. **A bench-blocked row is not negative evidence about its lever.** File it
   against the instrument, and re-queue it the moment the instrument is fixed.
4. **Record the null and the binary sha or the row is unfalsifiable later.**
   93.8% of this ledger's negative rows are missing at least one.
