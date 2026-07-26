# Ledger Resurrection Audit — `docs/NEGATIVE_EVIDENCE.md`

Campaign: FrankenSuite Performance Domination, 2026-07-25, Meta-Lever #1.
Auditor: `BlackThrush` (cc / STRUCTURAL lane). Ledger snapshot: 30,235 lines,
1,005 `##` entries, as of commit `f4d70a5e`.

A REJECT row is **VOID** when the measurement *could not have detected the
lever* — not when the lever was measured and lost. A void row is a harness
verdict wearing a lever's clothes; the design work behind it is already paid
for and only needs a re-measurement.

---

## 1. Method

Entries were split on `##` headings and the verdict token read from the text
before the first colon (`REJECT`, `NO-SHIP`, `LOSS (DROPPED)`, `REVERTED`,
`BENCH-BLOCKED`, `BLOCKER`, `HOLD` → negative; `WIN`, `SHIP`, `KEEP`,
`LANDED`, `FIX` → positive). Regex classification produced a candidate void
set; **every candidate was then read by hand and re-adjudicated**, because the
first pass mis-ranked several rows in both directions (see §4).

Void criteria, applied in order (campaign §1):

1. Never measured — the bench, control, or worker failed before any timing ran.
2. Target frame ~0% self-time in the profile the bench actually exercised.
3. Killed by a `cv < 5%` gate on an unpinned shared worker — a gate the
   fleet's own calibration (frankenmermaid) proves is unreachable on this
   hardware, so rejects on it carry no information.
4. Claimed ratio lies inside the fleet A/A null band (0.905–1.105) with no
   null control of its own recorded.
5. No ratio recorded at all.

A row is **SOUND** when the effect is decidably outside the null band, or when
the entry supplies a *mechanism* for the loss (profile-confirmed), or when the
blocker is behavioral rather than statistical.

---

## 2. Population

| Class | Count | Share |
|---|---:|---:|
| Total ledger entries | 1,005 | — |
| Positive verdicts (WIN / SHIP / KEEP / LANDED / FIX) | 718 | 71.4% |
| Negative verdicts (the audited population) | 146 | 14.5% |
| Neutral / survey / recon | 141 | 14.0% |

Provenance hygiene across the 146 negative rows:

| Property recorded | Count | Share |
|---|---:|---:|
| A/A null control of any kind | 17 | 11.6% |
| Binary sha256 | 22 | 15.1% |
| Both null control **and** binary sha | 9 | 6.2% |
| Profile / self-time attribution | 61 | 41.8% |
| Pinned or reserved worker | 17 | 11.6% |

**6.2% of rejects carry the two provenance artifacts the campaign's §2 harness
contract makes mandatory.** That is the headline number: for 93.8% of this
repo's negative evidence, we cannot currently distinguish "the lever lost"
from "the harness could not see it".

---

## 3. Audit result

One heading (line 1370) is a WIN row with an embedded REJECT clause and is
excluded as a parse artifact, leaving **145 adjudicated negative rows**.

| Verdict | Count | Share of 145 |
|---|---:|---:|
| **VOID** — measurement could not have detected the lever | **24** | **16.6%** |
| SOUND — decidable margin, mechanism-confirmed, or behavior-blocked | 31 | 21.4% |
| SOUND-WEAK — decidable margin, but no null control *and* no binary sha | 90 | 62.1% |

**Resurrection yield: 24 of 145 negative rows (16.6%) are re-measurable.**
Lower than frankenlibc's 39/93 (41.9%), and the reason is worth recording: this
ledger's rejects skew toward *large* margins (0.5×, 0.7×, 1.5–3× slower) where
the verdict survives a bad gate. The exposure here is not the void rate but the
**SOUND-WEAK** class: 90 rows whose effect is decidable but whose provenance
would not survive a hostile re-reading.

### 3.1 VOID — never measured (9 rows)

These rows record a *harness or worker failure*, and several say so in their
own verdict line. Nothing about the lever was learned.

| Line | Entry | Failure |
|---:|---|---|
| 5559 | duplicate-active-stride overlap proof | control did not compile |
| 5666 | direct C-order `nditer` chunk range | cold conformance target hit the 5-min cap |
| 5959 | F-order identity broadcast overlap proof | bench-blocked |
| 6319 | singleton-source multi-axis `broadcast_to` direct fill | compile failure |
| 6375 | complex `nancumprod` axis-0 proof | bench-cost prohibitive |
| 30156 | cov/corrcoef paired triangular Gram blocks | bench-cost prohibitive |
| 15055 | BLAS-backed-op gates | rch workers unavailable |
| 7624 | direct I64→I32 cast construction | unpaired one-shot budget |
| 8980 | `accumulate_extremum_typed` static dispatch | rch degradation mid-run |

Five of these nine were blocked by *the monolithic `criterion_python_surface`
bench binary's compile cost* — the exact failure mode closed by bead
`deadlock-audit-x7nnf` (17 per-domain bench binaries; the monolith went from
19.6k lines / 205 bench fns to 3.2k / 28). **The instrument that voided them
has since been fixed**, which makes this sub-group cheap to re-run.

One more, line 9986 (f16 union/setxor), is void under criterion 2 — a profile
blocker meant the target frame's self-time was never established.

### 3.2 VOID — killed by an unreachable CV gate (2 rows, both high-value)

| Line | Entry | Effect | A/A null | Why void |
|---:|---|---|---|---|
| 193 | selected bool `loadtxt(usecols=…)` direct token parse | **3.09–3.69×**, 5 independent runs | **0.988, 0.998, 1.001, 1.003, 1.045** | rejected only because not all four arms cleared `cv<5%` simultaneously |
| 806 | bounded tail ring, all-negative signed `loadtxt` usecols | **1.82–1.98×**, 4 independent runs | 1.029, 1.079, 1.087, 0.780 | same dual-CV gate; nulls noisy, effect never was |

Row 193 is the clearest void in the ledger. The effect is **3.5×** and the A/A
nulls sit within 0.2–4.5% of unity across five runs on three workers. Under the
campaign §2.3 median-CI gate — decidable iff the claimed ratio lies outside the
arm's A/A null 95% CI with 2× margin — this clears by more than an order of
magnitude. It was rejected by a gate that measures the hardware, not the lever.
Its own profile is sound: 8.14% `cfree`, 6.92% `malloc`, 5.80% owned
`Vec<String>` collection, 5.67% `fnp_python::loadtxt` over 4,560 samples.

Row 806 has the same shape with a smaller margin and noisier nulls (one null
arm at 0.78 is itself undecidable), so it ranks below 193 but well above the
rest.

### 3.3 VOID — inside the null band with no null control (12 rows)

Lines 2110, 2248, 2392, 2447, 3937, 4002, 4509, 5706, 6427, 22553, 23601,
25481. All claim 0.91×–1.18× with no A/A control recorded:
`noncentral_chisquare` gamma-shape cache (1.085×), legacy `RandomState` gamma
cache (0.973×), `tofile_text` integer formatting (1.00–1.10×), C-order odometer
(1.009×), singleton-axis `flip` clone (1.18×), Exact-U64 singleton cumulative
fold (1.03×), I8 `fromfile` byte-slice decode (−1.17%), hoisted
`loadtxt(usecols)` row plan (1.10×), int32 flat-sort small-pool regate
(1.017×), batch-Cholesky ordered-dot helper (0.914×), `ptp` int64 gate retune
(1.09×), `np.char.add` UCS4 concat (1.10×).

These are **not** promising resurrections. Each is a micro-lever whose entire
claimed effect is smaller than this hardware's noise floor; re-running them
under a correct gate will most likely produce an honest *undecidable*, not a
win. They are void in the strict sense — the verdict carries no information —
but their EV is low and they are ranked last.

### 3.4 SOUND — corrections the hand pass made against the regex

The automated first pass is recorded here because its errors are instructive.

| Line | Regex said | Hand verdict | Why |
|---:|---|---|---|
| 10440 | VOID (0.91× inside band) | **SOUND** | 66.1% self-time frame, bit-identical proof, 7 interleaved rounds, and a *profile-confirmed mechanism*: pairing equalized task cost but the crossbeam steal share barely moved (8.29%→7.64%), because the idle cycles are drain-tail off the critical path. A mechanism that explains the loss is stronger evidence than a ratio. |
| 27096 | VOID (no ratio) | **SOUND (behavior)** | f16 `hypot` — numpy's `hypotf` path is not bit-reproducible. No harness fixes this. |
| 27108 | VOID (no ratio) | **SOUND (behavior)** | f16 sort — numpy's signed-zero order is non-deterministic. |
| 28562 | VOID (1.0×) | **SOUND (behavior)** | `unique_values` is order-unspecified; not byte-reproducible. |
| 29615 | VOID (no ratio) | **SOUND (behavior)** | complex128 `sqrt` — last-ULP divergence. |
| 14250 | VOID (no ratio) | **SOUND (behavior)** | f32/f16 `sinc` — fused narrow transcendentals are not bit-exact. |
| 1160 | VOID (CV language) | **SOUND** | GEMM tile geometry sweep is a *clean* measurement; all six alternatives lose 2.3–39%. The campaign brief independently marks it SETTLED. |
| 14217, 27578, 27925 | VOID (inside band) | **SOUND (mechanism)** | bandwidth-saturation is a mechanism: numpy already saturates memory bandwidth on cheap fixed-affix string compares and f16 `spacing`, so parity is the physical answer, not a harness artifact. |

**Rule extracted:** a ratio inside the null band is void *unless the entry
supplies a mechanism*. Five behavior/bit-exactness rejections and four
bandwidth-bound rejections are permanently sound regardless of harness quality,
because no gate change can make a non-reproducible result reproducible.

---

## 4. Ranked re-run queue

Ranked by target-frame self-time in the profile the bench actually exercised,
per campaign §1.

| # | Entry | Line | Target frame self-time | Effect on record | Re-run cost |
|---:|---|---:|---|---|---|
| 1 | selected bool `loadtxt(usecols)` direct parse | 193 | 5.80% owned-`Vec<String>` collect + 15.06% malloc/cfree | 3.09–3.69× | low — prototype recorded, `fnp-io` bench |
| 2 | all-negative signed `loadtxt` usecols tail ring | 806 | 10.67% full-row `Vec<&str>` collect (+26.76% `CharSearcher::next_match`) | 1.82–1.98× | low — same crate, prototype recorded |
| 3 | complex `nancumprod` axis-0 | 6375 | never profiled | unmeasured | now low — the bench-binary split removed the blocker |
| 4 | cov/corrcoef paired triangular Gram blocks | 30156 | 66.1% kernel closure (from the sibling row) | unmeasured | now low — same reason |
| 5 | direct C-order `nditer` chunk range | 5666 | never profiled | unmeasured | low — wants a dedicated `fnp-iter` target, not `fnp-conformance` |

Ranks 1–2 are the campaign brief's own named candidates for this repo
("two of your levers are real effects that only the gate is blocking… you need
a quiet host, not a new idea"). This audit confirms that framing from the
ledger side and adds the decisive detail: **row 193's nulls are already at
unity.** It does not need a quieter host so much as a correct gate.

Ranks 3–5 are the bench-blocked cluster whose instrument has since been
repaired by `deadlock-audit-x7nnf`.

---

## 5. Yield

| Metric | Value |
|---|---:|
| Entries audited | 1,005 |
| Negative rows in scope | 146 (145 after one parse-artifact exclusion) |
| VOID (re-measurable) | 24 (16.6%) |
| — high-EV: decidable effect already on record, killed by the CV gate | 2 |
| — unmeasured, and the blocking instrument has since been fixed | 5 |
| — unmeasured, other blocker | 5 |
| — sub-noise micro-levers (void but low-EV) | 12 |
| SOUND-WEAK: decidable margin, provenance incomplete | 90 |
| Re-run under the corrected harness | see §6 |
| Re-won | see §6 |

---

## 6. Re-run status

Pending. Ranks 1–2 require the §2 harness contract (self-reporting ELF sha,
paired A/A in the same invocation, median-CI gate) in `fnp-io`'s Criterion
target, and a worker that is not `ovh-b`.

**Blocker surfaced during this audit (campaign §3b, ISA-heterogeneity tax):**
rch worker `ovh-b` **SIGILLs on the workspace `+avx2` baseline** — the
`zerocopy v0.8.48` build script died with `signal: 4, SIGILL` under
`cargo clippy --workspace`. This is an independent franken_numpy reproduction
of frankenscipy `hhr7j` and franken_networkx's global-AVX2 rejection. Any
franken_numpy measurement that landed on `ovh-b` is suspect, and every
ISA-sensitive run must pin away from it until the fleet fixes worker
capability detection.

---

## 7. Standing rules this audit adds

1. **A ratio inside 0.905–1.105 is void unless the entry supplies a mechanism.**
   Bandwidth saturation, critical-path analysis, and bit-exactness blockers are
   mechanisms; "we measured 1.03× and moved on" is not.
2. **Behavior/bit-exactness rejections are permanently sound.** Do not queue
   them for resurrection; no gate change can rescue a non-reproducible result.
3. **A bench-blocked row is not negative evidence about its lever.** File it
   against the instrument, and re-queue it the moment the instrument is fixed.
4. **Record the null and the binary sha or the row is unfalsifiable later.**
   93.8% of this ledger's negative rows are missing at least one.
