# KEEP-Claim Incumbent-Coverage Audit

Fleet audit, 2026-07-31, `BlackThrush`. Run after frankenfs published that 67 of
its 186 KEEP claims carry no vs-incumbent ratio. Same question, asked of this
repo. **Inventory only — no claim is deleted, weakened, or re-worded by this
document.**

Superseded the first pass of this file (`caf38d36`), which reported 732 KEEP
claims and 710 unclassifiable. Both numbers were wrong, in opposite directions,
and both corrections are recorded below.

## The three numbers

| | count | share |
|---|---:|---:|
| **KEEP claims held** | **751** | 100% |
| **carry a vs-incumbent ratio with the incumbent live in the same invocation** | **22** | 2.9% |
| **do not** | **729** | 97.1% |

"Live in the same invocation" is the schema-enforced form: a
`**Legacy incumbent arm (same invocation):**` line naming the incumbent, its
version, its compiled-artifact SHA-256, the invocation ID, and the measured
ratio, produced by the same process that printed its own executing-ELF hash.
All 22 also carry `**Campaign result class:** incumbent-win` and an A/A null.
That marker became mandatory on 2026-07-26; only 28 KEEP rows postdate it, and
the gate has leaked zero times since, so the shortfall is historical debt rather
than an open hole.

### The 729 are not one problem, and "unknown" was the wrong answer

The previous pass declared 710 rows unclassifiable and stopped. That was
over-cautious: it tried a keyword classifier, found it unsound, and concluded
nothing could be said. What can be said is what a hand-read of a random sample
says. A seeded random sample (`seed=20260731`) of **30** of the 723 unclassified
rows was read in full and classified:

| what the row actually contains | n / 30 | est. share | 95% CI (Wilson) |
|---|---:|---:|---|
| a vs-NumPy ratio from a NumPy arm in the same bench binary | 23 | 76.7% | 59.1–88.2% |
| no live NumPy arm at all (self-speedup, or cross-invocation) | 7 | 23.3% | 11.8–40.9% |

Extrapolated to the 723: roughly **554 rows (CI 427–638)** do compare against a
live NumPy arm and simply predate the schema; roughly **169 rows (CI 85–296)**
have no live incumbent comparison at all. Those two need different work — the
first needs re-recording under the current schema, the second needs measuring.

Of the 7 with no live NumPy arm:

- 6 are candidate-vs-candidate self-speedups measured in-tree (`complex_add`
  borrow-and-widen, `complex_div`, `complex_sum`, masked `count` from shape
  metadata, the `sliding_window_view` overlap detector, the core `UFuncArray`
  last-axis sum, which says outright it is "not a new Python-surface or
  NumPy-relative claim").
- 1 (`L24808`, `fnp-linalg kron` identity-RHS) measured NumPy **in a separate
  invocation over `ssh hz2`** while timing the candidate locally. That is the
  two-invocation form the campaign rejects; it is the single clearest
  unsupported row the sample turned up.

The sample is 30 rows; the intervals above are wide on purpose. Anyone wanting a
point answer must read all 723.

## Claims that CANNOT be converted — a genuinely different problem

Zero of the 30 sampled rows fall here, and the strict class is small enough to
enumerate rather than estimate. A claim cannot be converted when **no incumbent
arm exists for the surface**, so no amount of bench time yields a ratio:

1. **Unwired helpers.** A win in a Rust helper unreachable from the Python API.
   The only identified member, `fnp_io::tofile_text` (L2739), **stopped being a
   member on 2026-07-31**: `franken_numpy-ixs5y.400` wired it into public
   `fnp.tofile` for the exact int64 regime and measured 6.762491x against live
   `numpy.ndarray.tofile`. The correct response to this class is wiring, and it
   worked.
2. **Surfaces FrankenNumPy delegates to NumPy by design.** A delegated surface
   measures ~1.0 by construction; a "win" there indicates a measurement error,
   not a result. These must be labelled `delegated`, never converted.
3. **Rows that are not perf claims.** Correctness, parity, and gate-hygiene
   rows headed KEEP have no ratio to convert.

**Structural/missing-capability wins measured against a NumPy fallback** (f64
`isin`, integer `matmul`, `char.upper`) are NOT in this class even though they
superficially resemble it. NumPy runs the job, just by a worse algorithm, so a
live incumbent arm exists and several are already converted.

## Two defects corrected

1. `ledger_hygiene.rs` `is_keep()` matched `SHIP` inside `NO-SHIP`, counting
   REJECT rows as KEEPs. **Fixed in `b4d58f9b`** (word-boundary verdicts,
   `NO-SHIP` excluded, regression test added). Filed as `ixs5y.398`.
2. The first pass then over-corrected by hand, excluding any row whose label
   contained *any* negative token. That drops legitimate KEEPs whose heading
   records the prior state — `HOLD REDECISION / KEEP (INCUMBENT-WIN)` is a KEEP.
   The 732 in the first pass was too low; **751** is the count under the repo's
   own fixed predicate, and this audit no longer uses an ad-hoc filter.

## Ranked conversion queue

Ordered by how load-bearing the claim is for someone who might act on it, not by
how easy it is to measure. Ledger-only rows rank below anything a user reads.

### Tier 1 — README headline table, 28 numeric vs-NumPy claims

`README.md:1864`, introduced as *"Every ratio below is a measured vs-NumPy
speedup from the negative-evidence ledger."* Top-of-README public claim surface.

| # | capability row | numeric claims | backed by an `incumbent-win` row? |
|---:|---|---:|---|
| 1 | `float16` (no f16 ALU/BLAS in NumPy) | 7 | partially — f16 quantile/average/telemetry rows exist, not the quoted `nan_to_num 91x`, `clip 39x`, `isnan/isinf 27-33x`, `floor/ceil/rint 37-40x` |
| 2 | Sort / argsort / unique / set-ops | 5 | **`isin ... 530x` CONVERTED 2026-07-31 — see below**; the other 4 no |
| 3 | Reductions / scans / stats | 4 | no |
| 4 | Integer / GEMM (no integer BLAS in NumPy) | 3 | partially — exact 256x256 int64 `matmul` 19.999278x (L142) |
| 5 | Complex / temporal dtypes | 3 | no |
| 6 | Strings (`np.strings` / `np.char`) | 3 | partially — ASCII `char.upper` 20.000211x (L1636) |
| 7 | Array construction / manipulation | 3 | no |

### Tier 2 — README prose claims

- `README.md:1920` / `:2383` — "Average ratio 1.06x", 13-workload cross-engine
  snapshot. Dated 2026-05-25 and labelled as predating the campaign, so it is
  honestly scoped already; it needs re-measurement, not reclassification.
- `README.md:1894` — contiguous reduction kernel "~56% p50 faster", commit
  `d9cfe90`. An internal before/after, i.e. a **self-speedup presented in public
  docs without that label**.
- `README.md:1855` — "roughly 1,230 landed `perf(...)` commits". A commit count,
  not a performance claim; no conversion needed.

### Tier 3 — CHANGELOG per-capability breakdown

`README.md:1862` points users there. Not audited; inherits Tier 1's exposure.

### Tier 4 — ledger-only rows

The residual ~554 + ~169. Real debt, lowest urgency: no user acts on them
directly.

## Conversions completed against this queue

| date | claim | where it was published | converted result |
|---|---|---|---|
| 2026-07-31 | `isin` hashed-set "up to 530x (16M f64)" | `README.md:1868` (Tier 1, rank 1) | see `docs/NEGATIVE_EVIDENCE.md` — measured at the README's own 16M f64 shape against live NumPy in one invocation under the corrected dual-null gate |
| 2026-07-31 | int64 `matmul` | Tier 1 row 4 | 19.999278x vs live NumPy 2.4.6 (L142) |
| 2026-07-31 | ASCII `char.upper` | Tier 1 row 6 | 20.000211x vs live NumPy (L1636) |
| 2026-07-31 | `tofile_text` (was CANNOT-convert) | ledger L2739 | wired to public `fnp.tofile`; 6.762491x vs live `numpy.ndarray.tofile` (L7) |

## What this audit deliberately does not do

No claim is deleted, softened, or removed from the README in this pass.
Re-measuring 723 rows is not proposed either: the proposal is to re-read and
classify them, then re-measure only what is both unsupported and load-bearing.

## Follow-ups

- Classification sweep of the 723 pre-taxonomy KEEP rows (bounded, mechanical,
  ~554 expected to be re-recordings rather than re-measurements).
- `L24808` `kron` identity-RHS: the sampled row with a cross-invocation NumPy
  reference. Convert or scope it.
- Tier-1 README rows 1, 3, 5, 7 remain unconverted.

## Reproducing the counts

`parse_entries` / `is_keep` replicate
`crates/fnp-conformance/tests/ledger_hygiene.rs` exactly, so the denominator is
the repo's own definition; the audit script lives with this campaign's
scratch artifacts and the sample is reproducible from `seed=20260731` over the
sorted list of unclassified KEEP row line numbers.
