# fnp-io NPZ STORE clone profile - deadlock-audit-perf-npz-store-clone-45e20

## Scenario

- Subsystem: `fnp-io`
- Workload: Criterion `criterion_io`
- Command:
  `AGENT_NAME=BlackThrush RCH_FORCE_REMOTE=true CARGO_TARGET_DIR=/data/tmp/rch_target_franken_numpy_blackthrush_io rch exec -- cargo bench -p fnp-io --bench criterion_io -- --sample-size 12 --measurement-time 5 --warm-up-time 2`
- Worker: `vmi1153651`
- Git head before change: `cbd2f54c`
- Date: `2026-06-02`
- Metric: Criterion center estimate and confidence interval

## Ranked hotspot table

| Rank | Workload | Center | Confidence interval | Category | Evidence |
|------|----------|--------|---------------------|----------|----------|
| 1 | `write_npz_bytes/num_arrays/20` | `7.8612 ms` | `[5.2329 ms, 10.059 ms]` | CPU/memory copy | rch Criterion output |
| 2 | `read_npz_bytes/num_arrays/20` | `4.8938 ms` | `[4.8183 ms, 5.0009 ms]` | CPU/ZIP parse | rch Criterion output |
| 3 | `write_npz_bytes/num_arrays/10` | `2.5924 ms` | `[2.4778 ms, 2.7401 ms]` | CPU/memory copy | rch Criterion output |
| 4 | `read_npz_bytes/num_arrays/10` | `2.5590 ms` | `[2.4179 ms, 2.7629 ms]` | CPU/ZIP parse | rch Criterion output |
| 5 | `write_npz_bytes/num_arrays/5` | `1.5587 ms` | `[1.4226 ms, 1.7099 ms]` | CPU/memory copy | rch Criterion output |

Context: `npy_roundtrip/elements/100000` centered at `57.618 us`, so NPZ
STORE archive construction is the dominant measured `fnp-io` workload.

## Hypothesis ledger

- STORE payload clone: supports. `write_npz_bytes_with_compression` owns
  `npy_data` from `write_npy_bytes`, then clones it for `NpzCompression::Store`
  before extending the ZIP buffer.
- DEFLATE compression: rejects for this bead. The profiled `write_npz_bytes`
  path uses STORE, not DEFLATE.
- CRC required by ZIP: rejects as first lever. CRC must be preserved and is
  user-visible through ZIP metadata validation.
- Header synthesis: unclear. It contributes per entry, but the visible
  removable copy is lower risk and preserves exact bytes.

## Opportunity matrix

| Target | Impact | Confidence | Effort | Score |
|--------|--------|------------|--------|-------|
| Avoid STORE `npy_data.clone()` in `write_npz_bytes_with_compression` | 2 | 4 | 1 | 8.0 |

## Isomorphism requirements

- Entry order must be unchanged.
- ZIP local and central directory bytes must remain identical for STORE output.
- CRC-32 values, compressed size, uncompressed size, member names, and method 0
  must remain unchanged.
- NPY payload bytes must remain unchanged.
- Floating-point values are not interpreted by the change.
- RNG is not involved.

## Lever

Use `Cow<[u8]>` inside `write_npz_bytes_with_compression` so STORE entries
borrow the already-owned `npy_data` buffer instead of cloning it. DEFLATE still
owns the compressed output from `DeflateEncoder::finish`.

This is one lever: it removes one redundant payload copy on the STORE path and
does not change header synthesis, CRC calculation, ZIP ordering, compression
method selection, or NPY serialization.

## After benchmark

Command:

`AGENT_NAME=BlackThrush RCH_FORCE_REMOTE=true CARGO_TARGET_DIR=/data/tmp/rch_target_franken_numpy_blackthrush_io rch exec -- cargo bench -p fnp-io --bench criterion_io -- write_npz_bytes/num_arrays/20 --sample-size 12 --measurement-time 5 --warm-up-time 2`

`rch exec` does not expose a worker pin. Baseline was collected on
`vmi1153651`; post-change targeted reruns landed on `vmi1293453` and
`vmi1156319`. Both post-change center estimates were lower than the baseline.

| Workload | Worker | Center | Confidence interval | Delta vs baseline center |
|----------|--------|--------|---------------------|--------------------------|
| Before `write_npz_bytes/num_arrays/20` | `vmi1153651` | `7.8612 ms` | `[5.2329 ms, 10.059 ms]` | baseline |
| After targeted rerun 1 | `vmi1293453` | `3.4934 ms` | `[3.4568 ms, 3.5338 ms]` | `2.25x` center speedup, `55.6%` lower center time |
| After targeted rerun 2, final code | `vmi1156319` | `5.6255 ms` | `[5.3286 ms, 5.9181 ms]` | `1.40x` center speedup, `28.4%` lower center time |

The final comparison keeps the worker caveat explicit because remote selection
shifted. The lower final center estimate, the narrower final interval, and the
mechanical removal of one full STORE payload allocation/copy are sufficient for
the score threshold on this low-effort lever.

## Isomorphism proof

- Ordering and tie-breaking: archive member order is unchanged because the loop,
  `central_entries` push order, and central directory emission are unchanged.
- ZIP/NPY bytes: `npz_store_writer_matches_independent_store_zip_builder`
  compares the STORE writer output against an independent STORE ZIP builder.
- Golden output: the fixed STORE NPZ byte stream has SHA-256
  `2112e8eb6aa3e6d2fcbdb7ccd75d21f99c162f38977fdcbed12d698f875523f0`.
- Floating point: the change moves bytes only; it never decodes or reorders f64
  values.
- RNG: no RNG state or sampling path is involved.

## Validation

- `rch exec -- cargo test -p fnp-io --lib npz -- --nocapture` on
  `vmi1153651`: passed, 25 tests.
- `rch exec -- cargo test -p fnp-io --test metamorphic_io npz -- --nocapture`
  on `vmi1153651`: passed, 4 tests.
- `rch exec -- cargo test -p fnp-io --test npy_npz_diagnostic npz -- --nocapture`
  on `vmi1153651`: passed, 2 tests.
- `rch exec -- cargo check -p fnp-io --all-targets` on `vmi1227854`: passed.
- `rch exec -- cargo clippy -p fnp-io --all-targets -- -D warnings` on
  `vmi1149989`: passed.
- `cargo fmt --check -p fnp-io`: passed.

Rejected validation:

- `rch exec -- cargo test -p fnp-io npz -- --nocapture` included the
  NumPy-oracle integration shard and failed only because the remote worker did
  not have `FNP_ORACLE_PYTHON`, repo `.venv-numpy314`, or Python NumPy
  configured. The non-oracle NPZ lib, metamorphic, and diagnostic shards in
  that same run passed and were rerun separately above.

## Final score

Impact `2` x confidence `3` / effort `1` = `6.0`. The change clears the
required `>= 2.0` bar and is kept.
