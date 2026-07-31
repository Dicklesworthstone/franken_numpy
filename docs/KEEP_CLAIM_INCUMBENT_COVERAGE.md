# KEEP-Claim Incumbent-Coverage Audit

Fleet audit, 2026-07-31, `BlackThrush`. Run unprompted-by-content after
frankenfs published that 67 of its 186 KEEP claims carry no vs-incumbent ratio.
Same question, asked of this repo. **Inventory only — no claim is deleted,
weakened, or re-worded by this document.**

## The number

**Of 732 KEEP claims in `docs/NEGATIVE_EVIDENCE.md`, 18 (2.5%) carry an
authoritative recorded vs-incumbent classification, 4 (0.5%) are recorded
self-speedups, and 710 (97.0%) cannot be classified from the record at all
because the campaign-result-class marker did not exist when they were
written.**

That is a worse position than frankenfs's, and differently shaped. Theirs is
*known-unsupported*: they can name the 67. Ours is *unknown*: for 710 rows the
repo cannot demonstrate from its own record whether the incumbent ever ran.

| bucket | count | share |
|---|---:|---:|
| KEEP claims total | 732 | 100% |
| declared `incumbent-win` (authoritative) | 18 | 2.5% |
| declared `maintenance-self-speedup` (authoritative) | 4 | 0.5% |
| taxonomy-era rows missing a class (gate leak) | 0 | 0% |
| pre-taxonomy, class never recorded → **UNKNOWN** | 710 | 97.0% |

`**Campaign result class:**` became mandatory on 2026-07-26
(`WIN_CLASS_ENFORCEMENT_DATE`). Only 22 KEEP rows postdate it. The gate has
zero leaks since — every taxonomy-era row declares a class — so the problem is
entirely historical debt, not ongoing.

## Method, and what it cannot tell you

Entry parsing and the KEEP predicate replicate
`crates/fnp-conformance/tests/ledger_hygiene.rs` `parse_entries()` / `is_keep()`
so the denominator is the repo's own definition.

**Two defects found in the first pass, both corrected, one of them in the
repo's own gate:**

1. `is_keep()` matches `SHIP` as a substring of `NO-SHIP`, so rows headed
   `REJECT (NO-SHIP)` are counted as KEEPs by the shipped hygiene test. That
   inflated the first count from 732 to 838. This audit excludes rows carrying
   any negative verdict token. **The bug is in `ledger_hygiene.rs`, not only in
   this audit, and is filed separately — it is not fixed here.**
2. A keyword classifier was tried for the 710 unknown rows and abandoned as
   unsound. Markers like `vs numpy` in prose do not establish that the
   incumbent executed *live in the same invocation*, which is the actual fleet
   criterion; and rows plainly reporting vs-NumPy ratios ("3.8-4x") often carry
   none of the modern marker vocabulary. It produced 44% "unsupported" with a
   sampled error rate high enough in both directions that the number was not
   publishable. **The honest answer for those 710 is UNKNOWN, not a number.**

Resolving the 710 requires re-reading each row. That is a bounded, mechanical
task; it is not done here.

## Ranked conversion queue

Ordered by how load-bearing the claim is for someone who might act on it.
Ledger-only rows rank below anything a user reads.

### Tier 1 — README headline table, 28 numeric vs-NumPy claims (highest exposure)

`README.md:1864`, introduced as *"Every ratio below is a measured vs-NumPy
speedup from the negative-evidence ledger."* This is the top-of-README public
claim surface.

| # | capability row | numeric claims | backed by an `incumbent-win` row? |
|---:|---|---:|---|
| 1 | `float16` (no f16 ALU/BLAS in NumPy) | 7 | partially — f16 quantile/average/telemetry rows exist, but not for the quoted `nan_to_num 91×`, `clip 39×`, `isnan/isinf 27–33×`, `floor/ceil/rint 37–40×` |
| 2 | Sort / argsort / unique / set-ops | 5 | no — includes the `isin … 530×` claim, see below |
| 3 | Reductions / scans / stats | 4 | no |
| 4 | Integer / GEMM (no integer BLAS in NumPy) | 3 | no |
| 5 | Complex / temporal dtypes | 3 | no |
| 6 | Strings (`np.strings` / `np.char`) | 3 | no — the ASCII-log row is a *workload*, not `translate 183×` |
| 7 | Array construction / manipulation | 3 | no |

**Highest-priority single item: `isin` hashed-set "up to 530× (16M f64)".** The
only authoritative incumbent-class measurement of this surface is **23.882236x**
at 1M f64 (ledger L2177), and that row states in its own scope paragraph:
*"The historical 530x row uses another shape and does not generalize to this
one."* The README publishes the 530× headline without that caveat. The claim is
not necessarily false — it is an "up to" at a different shape — but the number a
reader is most likely to quote is the one our own successor row disclaims.

### Tier 2 — README prose claims

- `README.md:1920` / `:2383` — "Average ratio 1.06×", 13-workload cross-engine
  snapshot. Explicitly dated 2026-05-25 and labelled as predating the campaign,
  so it is honestly scoped already; it needs re-measurement, not reclassification.
- `README.md:1894` — contiguous reduction kernel "~56% p50 faster", commit
  `d9cfe90`. Internal before/after, i.e. a **self-speedup presented in public
  docs without that label**.
- `README.md:1855` — "roughly 1,230 landed `perf(...)` commits". A commit count,
  not a performance claim; no conversion needed.

### Tier 3 — CHANGELOG per-capability breakdown

`README.md:1862` points users to `CHANGELOG.md` for the per-capability
breakdown. Not audited here; it inherits Tier 1's exposure and should be
audited in the same sweep.

### Tier 4 — ledger-only rows

The residual of the 710. Real debt, lowest urgency: no user acts on them
directly.

## Claims that CANNOT be converted — a different problem

These are not "nobody got around to measuring". There is **no incumbent arm to
measure against**, so no amount of bench time produces a ratio. Each needs a
wiring decision, not a measurement.

1. **`fnp_io::tofile_text` (ledger L2237, 1.215448x).** Both public text-output
   surfaces are pure NumPy passthroughs — `savetxt` calls `numpy.savetxt`
   (`lib.rs:56695`) and `tofile` calls `numpy.asarray(...).tofile(...)`
   (`lib.rs:56728`). The win lives in a Rust helper unreachable from the Python
   API. **UNWIRED-HELPER class.** Tracked in `franken_numpy-ixs5y.397`.
2. **Any surface where FrankenNumPy delegates to NumPy by design.** A delegated
   surface measures ~1.0 by construction; a "win" there would indicate a
   measurement error, not a result. These must be labelled `delegated`, never
   converted.
3. **Structural/missing-capability wins measured against a NumPy fallback**
   (e.g. f64 `isin`, where NumPy's `table` method is integer-only). These CAN be
   converted and several already are — listed here only to keep them out of
   category 1, which they superficially resemble.

## What this audit deliberately does not do

No claim is deleted, softened, or removed from the README in this pass.
Inventory first. Re-measuring 710 rows is not proposed either — the proposal is
to re-read and classify them, then re-measure only what is both unsupported and
load-bearing.

## Follow-ups filed

- `ledger_hygiene.rs` `is_keep()` `NO-SHIP`/`SHIP` substring bug.
- Tier-1 README conversion queue, `isin 530×` first.
- Classification sweep of the 710 pre-taxonomy KEEP rows.
