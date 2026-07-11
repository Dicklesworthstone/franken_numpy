# Accumulate-extremum static-dispatch audit

Date: 2026-07-11
Agent: `cod_fnp`
Bead: `deadlock-audit-wmxzr` (remaining accumulate half)
Crate: `fnp-python`
Decision: **REJECT; production source restored to HEAD**

## Scope

The only production candidate replaced `accumulate_extremum_typed`'s erased
`combine: fn(T, T) -> T` parameter with `impl Fn(T, T) -> T + Sync`. Its callers
branched once before the scan and passed the existing max, min, or bitwise function
item directly. The two-pass algorithm, chunking, Rayon ownership, allocations,
eligibility gate, argument order, float tie/NaN semantics, bitwise operation mapping,
and bool u8-view routing were unchanged.

The final tree does **not** contain that source change. It keeps only a finite f64[8M]
profile/median harness with pre-timing dtype, shape, and raw-byte parity assertions.

## Profile-first evidence

The pre-edit profile executed remotely on effective worker `vmi1149989`. It captured
4,972 `cycles:u` samples with zero lost:

| frame | samples |
|---|---:|
| `fnp_python::np_fmax` | 51.27% |
| block-total Rayon closure | 22.72% |
| output-rescan Rayon closure | 21.05% |

Profiled benchmark binary SHA-256:
`a3aa068f514f4d133072c076eae15d820579b7853a173208dd3b6b72020aa0a8`.
`perf.data` SHA-256:
`952bf1451c17976cac5c4d3a2a86ddd5e19b2146ee61cd9578da9e2d374e4399`.

The profile used a first-draft edge fixture containing a sticky NaN. Review caught
that this was not representative of the existing finite-random frontier row, so the
decision benchmark was corrected to finite standard-normal data before the final
HEAD/candidate pair. The profile remains valid evidence that the exact candidate seam
was exercised; its percentage is not treated as proof that the semantic work inside
`np_fmax` was removable overhead.

Exact profile command:

```text
RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR RCH_WORKER=vmi1149989 RCH_WORKERS=vmi1149989 rch exec -- cargo bench -p fnp-python --bench criterion_python_surface --config 'target.x86_64-unknown-linux-gnu.runner=["sh","-c","printf RUNNER_HOST=; hostname; printf BINARY_SHA256=; sha256sum \"$1\"; RAYON_NUM_THREADS=4 FNP_ACCUMULATE_EXTREMUM_BENCH_ONLY=1 perf record -F 199 -e cycles:u -g -o /tmp/fnp_accumulate_extremum_fnptr_baseline.data -- \"$@\"; rc=$?; printf PERF_DATA_SHA256=; sha256sum /tmp/fnp_accumulate_extremum_fnptr_baseline.data; perf report --stdio --no-children --percent-limit 0.10 --sort overhead,symbol --call-graph none -i /tmp/fnp_accumulate_extremum_fnptr_baseline.data; exit $rc","fnp-perf"]' -- python_accumulate_extremum_median_gate/maximum_accumulate_f64_8m_fnp_profile --profile-time 10 --noplot
```

## Median gate

The final comparison used finite `rng.standard_normal(8_000_000).astype(float64)`
input. Each read ran as one binary/process with ABBA/BAAB effect pairs, NumPy A/A null
pairs, `black_box`, and a pre-timing raw-byte assertion. Both reads used effective
worker `vmi1149989` with six of its eight slots reserved; the queue was empty before
each invocation, leaving too few slots for a normal four-slot co-tenant.

| read | NumPy median | FNP median | NumPy/FNP effect median [p10,p90] | null median [p10,p90] | FNP CV |
|---|---:|---:|---:|---:|---:|
| HEAD finite | 32.732995 ms | 14.256978 ms | 2.277776 [1.598710,2.809840] | 1.008737 [0.933592,1.061708] | 30.669% |
| candidate finite | 33.769022 ms | **17.902274 ms** | 1.915649 [1.041172,3.050349] | 0.989480 [0.902876,1.190389] | 72.343% |

The candidate regressed FNP self-time by **25.57%**. Criterion's combined paired row
reported a **21.481% regression**, confidence interval +9.7469%..+36.224%, p=0.00.
Baseline binary SHA-256:
`1721254833d03acb9967fb26a33bda177844f7d87776c3044033ffff68bcdc73`.
Candidate binary SHA-256:
`4730ae5af99a329b78e52d6b5b2f372c6e7ae95e3c6ba1bb66f902d598f4c490`.

An earlier same-worker sticky-NaN cross-version pair pointed the same direction:
11.899382 ms at HEAD versus 12.962021 ms candidate (**8.93% regression**), while its
combined Criterion row regressed 7.435%. It is recorded only as corroborating edge
evidence, not the representative decision row.

The identical final command around the one source edit was:

```text
RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR RCH_WORKER=vmi1149989 RCH_WORKERS=vmi1149989 RCH_BUILD_SLOTS=6 RCH_TEST_SLOTS=6 rch exec -- cargo bench -p fnp-python --bench criterion_python_surface --config 'target.x86_64-unknown-linux-gnu.runner=["sh","-c","printf RUNNER_HOST=; hostname; printf BINARY_SHA256=; sha256sum \"$1\"; RAYON_NUM_THREADS=4 FNP_ACCUMULATE_EXTREMUM_BENCH_ONLY=1 exec \"$@\"","fnp-runner"]' -- python_accumulate_extremum_median_gate/maximum_accumulate_f64_8m_null_then_effect --sample-size 10 --warm-up-time 1 --measurement-time 5 --noplot
```

The harness prints `verdict=WIN` when FNP beats NumPy beyond the NumPy A/A null. That
was already true at HEAD, so this label was not used to decide the source change. The
cross-version FNP medians and Criterion change estimate are the relevant gate.

## Remote-only and degradation record

Every Cargo command used `RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- ...`;
no Cargo command ran locally. RCH remained degraded at 9/12 healthy workers.

- A requested eight-slot and six-slot run on `hz2` was refused with `no admissible
  workers`; strict mode refused local fallback.
- `RCH_WORKER` and `RCH_WORKERS` are advisory in installed RCH 1.0.47. An attempted
  `hz2` read was routed to `vmi1149989`; its effective-worker mismatch was surfaced and
  cancelled before compilation.
- A finite baseline attempt on `vmi1149989` was cancelled after queue inspection found
  a four-slot co-tenant. It produced no timing result.
- The final pair waited for the worker to clear, then reserved six slots for each read
  and verified `RUNNER_HOST=vmi1149989` in the runner itself.

## Validation record

The focused gate passed remotely both before and after integration onto the released
tree. The post-integration run completed on `vmi1227854` with exit 0:

```text
RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- cargo check -p fnp-python --lib --bench criterion_python_surface
```

The workspace-wide remote checks were also attempted. `cargo check --workspace
--all-targets` stopped in the third-party `zerocopy` build script with `SIGILL` on
the selected worker before FrankenNumPy code was checked. `cargo clippy --workspace
--all-targets -- -D warnings` reached project code and stopped on the two existing
constant-assert lints in `crates/fnp-runtime/tests/worker_isa_probe.rs`; neither
failure is in this bundle's edited paths. Strict RCH declined `cargo fmt --check` as
a non-compilation request (`RCH-E301`), so no local Cargo fallback was used.
`git diff --check` passed. UBS's standalone no-Cargo Rust categories 2 through 24
returned exit 0 (category 24 also printed the scanner's own `BrokenPipeError`). The
monolithic all-category scan terminated inside category 1 on this 15k-line benchmark's
pre-existing panic-macro volume before emitting a summary; the new benchmark hunk was
reviewed directly and uses only the file's existing assertion/`expect` conventions.

## Decision and retry boundary

REJECT. Static dispatch was bit-isomorphic and compiled cleanly, but it failed two
same-worker median comparisons and made the representative finite row materially
slower and noisier. The production file is restored byte-for-byte to HEAD.

Do not retry this exact transfer unless a future compiler/codegen change or a
lower-level profile separates function-call overhead from the required `np_fmax`
semantic work, followed by the same cross-version median/null proof.

The next untried cod-lane frontier, deliberately not attempted in this one-lever run,
is reusing the existing `PackedWideStringKey` representation for flat U9..U16 and
S9..S16 value sort.
