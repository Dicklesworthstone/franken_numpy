# String Searchsorted Packed Merge Evidence

Date: 2026-07-10
Agent: cod_fnp
Bead: deadlock-audit-l3lcj

## Lever

`try_native_string_searchsorted` now routes narrow fixed-width strings (`S1..8` and Latin-1 `U1..8`) through the existing order-preserving packed `u64` key helper. The sorted haystack keys are scanned once after sorting query keys, then insertion indices are scattered back to the original query order. Wider or non-Latin-1 records fall through to the existing memcmp binary-search route.

## Criterion Rows

Pre-edit supporting baseline, worker `vmi1227854`:

| row | median |
|---|---:|
| `fnp_searchsorted_U8_left_2m` | 133.88 ms |
| `numpy_searchsorted_U8_left_2m` | 2957.8 ms |

Candidate keep row, worker `ovh-a`:

| row | median |
|---|---:|
| `fnp_searchsorted_U8_left_2m` | 53.751 ms |
| `numpy_searchsorted_U8_left_2m` | 1466.1 ms |

Ratios:

| comparison | ratio |
|---|---:|
| candidate vs same-worker NumPy | 27.3x |
| candidate vs pre-edit FNP baseline | 2.49x cross-worker |

## Correctness

`rch exec -- cargo test -p fnp-python --test conformance_sort_search searchsorted_packed_latin1_string_records_match_numpy`

Result: passed, 1 test; covers U8/S8 Latin-1 records, NUL bytes, duplicates, out-of-range probes, left/right sides, and small all-identical U4/S4 cases.

## Gate Notes

`rch exec -- cargo check -p fnp-python --lib`: passed.

`rch exec -- cargo check -p fnp-python --tests`: fails on pre-existing `where_py` lib-test call signatures at `crates/fnp-python/src/lib.rs` 98523, 98548, 98574.

`rch exec -- cargo clippy -p fnp-python --lib -- -D warnings`: blocked before this crate by pre-existing `fnp-ufunc::UFuncArray::nan_filtered` dead code under `-D warnings`.

`cargo fmt --check -p fnp-python`: fails due broad existing crate formatting drift.

`perf stat` and `cargo flamegraph`: attempted, but RCH refused to offload wrapper/non-compilation commands and local execution hit the shared cargo build lock. See `perf_stat_candidate.txt` and `flamegraph_attempt.txt`.
