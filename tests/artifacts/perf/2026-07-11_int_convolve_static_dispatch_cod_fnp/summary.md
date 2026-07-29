# Integer convolution inner-loop static dispatch

Date: 2026-07-11
Agent: `cod_fnp`
Bead: `deadlock-audit-wmxzr` (integer-convolution half only; the separate
`accumulate_extremum_typed` sibling remains open)
Crate: `fnp-python`

## Scope and lever

The production change is one dispatch lever in `int_convolve_typed`: replace the
`fn(T, T) -> T` multiply and add parameters with statically dispatched
`impl Fn(T, T) -> T + Sync` parameters. Call sites, eligibility gates,
allocation, parallel ownership, output slicing, tap order, wrapping operations,
and every loop bound are unchanged. This lets LLVM inline the existing wrapping
multiply and add closures inside the O(n*m) tap loop.

The benchmark and conformance edits are evidence only. No linalg source or row
was touched.

## Profile-first evidence

The baseline profile ran on remote worker `vmi1149989` before the production
edit. `perf record` captured 7,509 `cycles:u` samples with zero samples lost.
The exact convolution path accounted for 98.20% of samples:

| frame | samples |
|---|---:|
| `int_convolve_typed::<i64>` Rayon closure `call_mut` | 52.95% |
| wrapping multiply closure `call_once` | 26.26% |
| wrapping add closure `call_once` | 18.99% |

Profiled benchmark binary SHA-256:
`221f718dda1ccbbf00ad8410bba65a6c1394c354a0058117f6f6d30a770f651b`.
`perf.data` SHA-256:
`66631e13bea663785d6fbac338ba0a4c38d211acd2df254b197da03a9a6f058f`.

Exact profile command:

```text
RCH_WORKER=vmi1149989 RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- cargo bench -p fnp-python --bench criterion_python_surface --config 'target.x86_64-unknown-linux-gnu.runner=["sh","-c","printf RUNNER_HOST=; hostname; printf BINARY_SHA256=; sha256sum \"$1\"; RAYON_NUM_THREADS=4 FNP_INT_CONVOLVE_BENCH_ONLY=1 perf record -F 199 -e cycles:u -g -o /tmp/fnp_int_convolve_fnptr_baseline.data -- \"$@\"; rc=$?; printf PERF_DATA_SHA256=; sha256sum /tmp/fnp_int_convolve_fnptr_baseline.data; perf report --stdio --no-children --percent-limit 0.10 --sort overhead,symbol --call-graph none -i /tmp/fnp_int_convolve_fnptr_baseline.data; exit $rc","fnp-perf"]' -- python_int_convolve_median_gate/int_convolve_i64_200k_256_fnp_profile --profile-time 10 --noplot
```

## Median gate

The decision row is `int_convolve_i64_200k_256_null_then_effect`. It performs
pre-timing dtype, shape, and byte-parity assertions, then uses ABBA/BAAB paired
samples with a NumPy A/A null control. Both before and after reads ran through
strict remote RCH on worker `vmi1149989`; no Cargo command ran locally.

| read | NumPy median | FNP median | paired effect median (NumPy/FNP) | effect p10..p90 | null median | null p10..p90 | paired wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 44.324550 ms | 94.942347 ms | 0.460631 | 0.284995..0.547550 | 0.887478 | 0.518653..1.104887 | 0/20 |
| candidate before rebase | 26.388243 ms | 12.115056 ms | **2.246560** | **1.900383..2.624485** | 1.006761 | 0.935578..1.077594 | **20/20** |
| integrated candidate atop `9b024cad` | 27.130505 ms | **10.765410 ms** | **2.531046** | **2.112046..2.806347** | 0.993796 | 0.924075..1.100653 | **20/20** |

The FNP medians improve by **7.84x**. The candidate effect p10 (1.900383)
exceeds the null p90 (1.077594), so the median win clears both noise and the A/A
control. The null-corrected candidate effect is 2.231474. Baseline benchmark
binary SHA-256 was
`67dcb5c8ca8e688c7870cd2445d9e22817baa4770fc0085c7f5dc6685802d110`;
candidate binary SHA-256 was
`b7d64ea7528361b928743f382f97e7bdb54e46c3f83086510f2f0aaa7819cddb`.
After rebasing over cc's linalg commit, the exact same strict-remote command
replicated the win on the integrated tree: effect p10 2.112046 remained above
null p90 1.100653, null-corrected effect was 2.546847, and all 20 pairs won.
That integrated benchmark binary SHA-256 was
`d671048df6ece78bf5dcf4926d1e36cf40af529a60950156424bd1618dc187d1`.

The baseline and candidate used this identical command around the one source
edit:

```text
RCH_WORKER=vmi1149989 RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- cargo bench -p fnp-python --bench criterion_python_surface --config 'target.x86_64-unknown-linux-gnu.runner=["sh","-c","printf RUNNER_HOST=; hostname; printf BINARY_SHA256=; sha256sum \"$1\"; RAYON_NUM_THREADS=4 FNP_INT_CONVOLVE_BENCH_ONLY=1 exec \"$@\"","fnp-runner"]' -- python_int_convolve_median_gate/int_convolve_i64_200k_256_null_then_effect --sample-size 10 --warm-up-time 1 --measurement-time 5 --noplot
```

## Bit-identity and validation

Static dispatch changes only how the same closures are called. Each output
still visits exactly the same `jlo..=jhi` taps in the same order and performs the
same wrapping multiply followed by wrapping add. There is no floating-point,
tie-breaking, RNG, or output-order surface in this route.

The strict-remote focused conformance test passed 1/1. Its main shape compares
dtype, shape, and raw bytes against NumPy for signed and unsigned 8/16/32/64-bit
dtypes, `full`/`same`/`valid`, and both convolve and correlate. A second
all-dtype/all-mode convolve case reverses operand lengths, and a separate
int64/full/convolve case forces overflow. The uint16 row was added because it
was the only native integer dtype absent from the existing battery.
The exact command was repeated after rebasing over `9b024cad`; it again passed
1/1 on `vmi1149989` (1.15 s test time).

```text
RCH_WORKER=vmi1149989 RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- cargo test -p fnp-python --test conformance_convolution int_convolve_correlate_native_parallel_bit_exact_matches_numpy -- --exact --nocapture
```

Workspace-wide strict-remote check compiled the changed library and benchmark,
then failed in three pre-existing `where_py` lib-test calls at `lib.rs:99624`,
`:99649`, and `:99675`. Full strict-remote clippy reached an unrelated existing
`fnp-runtime/tests/worker_isa_probe.rs` constant-assert warning under
`-D warnings`; another clippy worker lacked `cargo-clippy`, and an earlier check
worker SIGILLed in zerocopy's build script. RCH also ignored requested-worker
placement for the final check and selected `vmi1264463`. Every event remained
fail-closed/remote; none triggered local Cargo fallback. A strict-remote clippy
retry scoped to the changed lib/bench/test targets compiled into `fnp-python`
and then stopped on 105 existing default clippy lints across the 137K-line
library. The only report on this function was the pre-existing
`too_many_arguments` shape (ten parameters before and after); no report points
at either changed parameter line.

UBS's Rust scanner ran on each changed Rust file with cargo-based subchecks
disabled so the remote-only build rule could not be bypassed. The benchmark and
test files reported zero critical findings. The monolithic library scan exited
nonzero on its existing whole-file inventory (1,155 critical, 6,291 warnings,
6,698 informational); the changed production hunk adds no unwrap, unsafe,
allocation, arithmetic, indexing, or control flow, and no scanner sample points
to either modified parameter line. This is surfaced as a non-baselined scanner
result, not called a clean UBS pass.

Direct rustfmt checks pass for the conformance file and show no formatting diff
in either changed source/benchmark hunk. Whole-file checks remain red on broad
pre-existing formatting drift elsewhere in the 15K-line benchmark and 137K-line
library. `git diff --check` passes.

## Decision

SHIP. The sole production lever is bit-identical, removes the profiled indirect
inner-loop calls, and clears the paired median/null gate decisively. Do not infer
completion of the bead's untouched `accumulate_extremum_typed` sibling.
