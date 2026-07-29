# Flat wide-string value-sort packed-key lever

Date: 2026-07-11
Agent: `cod_fnp`
Bead: `deadlock-audit-jlqsi`
Crate: `fnp-python`
Decision: **SHIP**

## Negative-ledger and BV routing

The preceding `accumulate_extremum_typed` rejection named flat U9..U16/S9..S16
value sort as the next untried cod-lane frontier. Earlier statements that the
wide-key family was complete refer specifically to the unique/setops family
(unique, intersect, set difference, set xor, and union), not `np.sort`.

`bv --robot-triage` completed before profiling with data hash
`ea1d3735d1d12821`: 2,421 total issues, 146 actionable, no dependency cycles.
The only open unassigned perf beads were unrelated exp/log and historical
accumulate work, so this new sort bead was created and claimed. The production
edit touches neither linalg nor setops logic.

## Profile-first evidence

The pre-edit million-element U16 path ran under strict-remote RCH on effective
worker `vmi1149989`, with `RAYON_NUM_THREADS=4`. `perf record -F 199 -e cycles:u`
captured 7,371 `cycles:u` samples with zero lost:

| frame/class | samples |
|---|---:|
| unresolved libc region consistent with record compare/memcmp | 50.27% |
| Rayon sort recursion comparator | 14.79% |
| required Unicode high-byte eligibility scan | 5.47% |
| insertion-sort comparator | 2.74% |
| resolved output gather | about 0.27% |

This directly profiles the current flat U16 value-sort route after warmup. The
comparison frames, rather than output gathering, dominate the sampled path.

Profiled benchmark binary SHA-256:
`fd1802aacf8d8fe03faf6741b1be1c4ee6275bd9fac4cc50ffb78a4ff23aeb83`.
`perf.data` SHA-256:
`50f5e7401d8e4db0c328c0f3146c7b5bfb06b5b79ae00a4e97b7741d10f9bc0a`.

The profile was dispatched with the mandatory fail-closed prefix:

```text
RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- cargo bench -p fnp-python --bench criterion_python_surface --config <remote perf runner with RAYON_NUM_THREADS=4 and FNP_WIDE_STRING_SORT_BENCH_ONLY=1> -- python_wide_string_sort_median_gate/wide_string_sort_u16_1m_fnp_profile --profile-time 10 --noplot
```

## One production lever

`try_native_string_sort_flat` already packs U1..U8/S1..S8 records into one
ordered `u64`, but U9..U16/S9..S16 fell back to sorting record indices with
cache-cold full-record comparisons. The sole production change reuses the
existing `PackedWideStringKey { high, low }` encoder for those widths, sorts
`(key, original_index)` pairs, and passes the derived permutation to the
unchanged gather.

The encoder is a complete order-preserving image of every eligible record:
codepoints/bytes 0..7 fill the big-endian `high` word, and the remaining 1..8
fill the left-aligned big-endian `low` word. Unicode inputs retain the existing
Latin-1/native-layout eligibility check. Equal keys are therefore identical
records, so the index tie-break introduced by the tuple cannot alter sorted
bytes. All narrow, non-Latin-1, non-native-layout, wider-than-16, non-flat, and
below-size-gate routes are unchanged.

The packed route has a deliberate memory tradeoff: while deriving the final
permutation, it holds approximately 16 bytes/record of keys, 24 bytes/record of
`(key,index)` pairs (including alignment), and 4 bytes/record of permutation.
That is roughly 40 MB more transient scratch than the old permutation-only sort
at n=1M, before the common output gather. Peak RSS was not measured, so this
lever makes no memory-improvement claim.

## Same-worker median gate

The final pre/post comparisons used the same effective worker `vmi1264463`
before/after each pair, six reserved slots (leaving no room for a normal
co-tenant), and `RAYON_NUM_THREADS=4`. Each read was a single binary/process.
The harness made 20 ABBA/BAAB NumPy/FNP observations plus 20 NumPy A/A null
observations and asserted dtype, shape, ownership, and raw-byte parity before
timing. U16/S16 were the original profiled representatives. Independent review
then identified S9 as the riskiest compact-record crossover, so a second strict
pair gated both U9 and S9 rather than extrapolating across the entire range.

| fixture/read | NumPy median | FNP median | effect median [p10,p90] | null median [p10,p90] | FNP CV | wins |
|---|---:|---:|---:|---:|---:|---:|
| U9 HEAD | 214.896406 ms | 188.102154 ms | 1.165035 [1.053336,1.231738] | 1.000436 [0.953715,1.048084] | 7.313% | 19/20 |
| U9 candidate | 222.250620 ms | **80.664659 ms** | **2.837996 [2.224104,3.024474]** | 1.002569 [0.967072,1.048340] | 17.217% | **20/20** |
| U16 HEAD | 233.459528 ms | 241.193212 ms | 0.965347 [0.886156,1.026138] | 1.000939 [0.955022,1.160165] | 9.176% | - |
| U16 candidate | 228.278575 ms | **91.610434 ms** | **2.518444 [2.302069,2.888821]** | 1.007224 [0.968822,1.051889] | 15.766% | **20/20** |
| S9 HEAD | 200.076134 ms | 154.940143 ms | 1.317287 [1.211232,1.705213] | 1.008200 [0.953716,1.060584] | 14.919% | 20/20 |
| S9 candidate | 207.965203 ms | **59.335614 ms** | **3.625924 [3.253424,3.974680]** | 0.995497 [0.956869,1.092434] | 10.318% | **20/20** |
| S16 HEAD | 195.053952 ms | 169.427677 ms | 1.161963 [1.063404,1.307552] | 0.997552 [0.950113,1.043965] | 10.333% | 18/20 |
| S16 candidate | 204.288826 ms | **59.025391 ms** | **3.432197 [3.096605,3.764361]** | 1.022743 [0.943594,1.073951] | 7.792% | **20/20** |

Cross-version FNP self-time improves **2.331903x / 57.1166%** for U9,
**2.632814x / 62.0178%** for U16, **2.611250x / 61.7042%** for S9, and
**2.870420x / 65.1619%** for S16. Every candidate effect p10 exceeds its null
p90 and all candidate rows win 20/20 pairs. Criterion's combined NumPy+FNP
estimates also improved, but the keep decision uses the direct FNP medians. The
harness's printed `WIN` is an FNP-versus-NumPy judgment, not the pre/post gate.

Baseline benchmark binary SHA-256:
`ea74ea3a133fedfcc92e388ba50fac186433964aee0c74b6d4d10667c8f02ec1`.
Candidate benchmark binary SHA-256:
`48d06ac84d95449db8f49167358bcb90c441b97135b4d0a5bd8ed8a3349eab6f`.
The lower-bound crossover baseline/candidate binary SHA-256 values were
`86ab023a0ec67621658f2c4239cf0b0cb9ab3915fb06c9ea827424aa368afd55`
and `10012e0cef1058bccc6bcab0618ac9d88bd82de85f49249113e511458b056ecc`.

The baseline and candidate used the same strict-remote command shape around the
single production edit:

```text
RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR RCH_BUILD_SLOTS=6 RCH_TEST_SLOTS=6 rch exec -- cargo bench -p fnp-python --bench criterion_python_surface --config <remote runner with RAYON_NUM_THREADS=4 and FNP_WIDE_STRING_SORT_BENCH_ONLY=1> -- python_wide_string_sort_median_gate --sample-size 10 --warm-up-time 1 --measurement-time 5 --noplot
```

## Exact-output proof

The focused conformance test uses n=300,000 U9, U16, S9, and S16 arrays so all
four cases cross the native-route size gate. Values span the full 0..255 range,
including embedded NULs, and repeated assignments create dense ties. It compares
NumPy and FrankenNumPy dtype, shape, ownership, and `tobytes()` exactly. The final
strict-remote run passed 1/1 on `vmi1149989` in 2.59 seconds:

```text
RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- cargo test -p fnp-python --test conformance_sorting sort_string_packed_wide_latin1_matches_numpy -- --exact --nocapture
```

The result is byte-identical, which is stronger than a per-operation ULP bound.

## Remote-only and degradation record

Every Cargo command used `RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec --`;
no Cargo command ran locally.

- An initial six-slot profile request found no admissible remote worker. Strict
  mode refused local fallback; the four-slot retry ran remotely.
- Installed RCH 1.0.47 treats `RCH_WORKER`/`RCH_WORKERS` as preferences rather
  than hard pins. Mismatched or loaded placements were inspected and cancelled
  before candidate execution. The decisive pair verified the runner host itself.
- The first conformance attempt reached `ovh-b` but third-party `zerocopy`'s
  build script SIGILLed before any project test ran. A later project-test attempt
  exposed and fixed an indentation error in the new Python fixture. The final
  strict-remote retry passed on `vmi1149989`.
- A post-edit `cargo check` attempt selected `vmi1149989` but was cancelled
  while crates.io repeatedly returned HTTP 503/timeouts; it had not reached
  project code. The final candidate `cargo bench` then compiled the exact
  production and benchmark tree remotely on `vmi1264463` and exited 0.
- RCH evicted the worker target between lower-bound reads, so both width-9
  binaries paid independent full release builds. This increased wall time but
  did not change the same-worker, same-slot, same-Rayon timing conditions.

## Final validation

The final strict-remote candidate invocation compiled the exact production and
benchmark tree in the optimized bench profile on `vmi1264463`, ran both U9/S9
median/null rows, and exited 0. The earlier focused conformance command passed
the exact-output test 1/1 on `vmi1149989`. `git diff --check` passed. UBS ran
with Cargo subchecks disabled: categories 2..24 passed for the benchmark and
conformance test; production `lib.rs` passed categories 2..7 and 9..24, while
category 8 terminated in the scanner's own `BrokenPipeError` without emitting a
code finding. Category 1 is intentionally macro-heavy across these giant
existing files; the changed hunks were reviewed directly and add no production
panic/unwrap/unsafe site (the evidence-only benchmark/test use their existing
`expect`/assert conventions).

## Decision

SHIP. The one production lever removes the profiled record-comparison chase,
preserves the complete record order and every fallback boundary, is byte-exact
on adversarial eligible data, and clears same-worker median/null gates by large
margins for both Unicode and byte-string representatives.
