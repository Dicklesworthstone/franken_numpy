# np.histogram — parallel privatized counting (BlackThrush, 2026-06-19)

## Gap targeted (we LOST)
`np.histogram(a, bins=int)` for large 1-D f64/integer arrays was **1.61x SLOWER**
than NumPy. The fnp-python zero-copy path (`histogram_typed`) ran a serial
per-element affine-index + edge-correction loop. NumPy runs histogram
single-threaded, so a parallel privatized tally beats it outright.

## Lever
`extreme-software-optimization` / `alien-graveyard` privatized-histogram pattern
(same family as the radix-select median win). For `n >= 1<<16` and
`rayon::current_num_threads() >= 2`:
- Read the read-only contiguous PyBuffer cells as a plain `&[T]` (sound under the
  GIL; `ReadOnlyCell<T>` is `repr(transparent)` over `T`; T is a POD Sync numeric).
  Zero copy — `par_iter` reads the buffer directly.
- Parallel reduce → `(min, max, all_supported, all_finite)`.
- Edges via `numpy.linspace` (bit-identical floats, unchanged).
- Privatized parallel fold: each rayon task tallies into its own `vec![0i64; nbins]`,
  reduced by elementwise add. Bin chosen by `partition_point` over the edges —
  the same rule as `UFuncArray::histogram_full` (franken_numpy-40n4u), bit-identical
  to NumPy and immune to the f64 internal-edge rounding drop. Integer +1 tallies
  merge order-independently → exact regardless of fold grouping.

f32 left serial (separate `histogram_f32`, already 0.86–0.91x = winning; its
float32-precision binning has delicate bail paths — not worth the risk).

## MEASURED (4M f64, load ~3.7, 64 cores)
| bins | NumPy us | fnp us | fnp/np |
|------|----------|--------|--------|
| 10   | 26021    | 3237   | **0.124** (8.0x) |
| 50   | 25701    | 4572   | **0.178** (5.6x) |
| 100  | 26759    | 4358   | **0.163** (6.1x) |
| 256  | 27236    | 4591   | **0.169** (5.9x) |
| 1000 | 26644    | 6917   | **0.260** (3.9x) |

Before: 1.61x SLOWER. After: 3.9–8.0x FASTER.

## Parity
240/240 differential cases bit-identical (counts via `array_equal`, edges via
`array_equal`) across 10 dtypes {f64,f32,i8/16/32/64,u8/16/32/64} × 6 bin counts
{1,2,10,50,100,257} × 4 distributions {normal,uniform,dup,small}.

Conformance `conformance_histogram_bincount`: all 30 histogram/bincount/digitize
counting tests GREEN. 2 unrelated failures (`*_python_container_keyword_surfaces_*`)
are a **pre-existing** test-harness bug: `outcome_body` builds its Python oracle
with `"...\n\"` line-continuations that strip the next line's indentation, yielding
`def outcome(op):\ntry:` → IndentationError. The test file is unmodified by this
change (diff touches only `histogram_typed`), so these fail on committed `main`.

## NEGATIVE-EVIDENCE LEDGER (this session)
- WIN: histogram f64/int — 1.61x slow → 3.9–8.0x fast (shipped).
- NEUTRAL/no-ship: histogram f32 already winning (0.86–0.91x), left serial.
- DEFER: corrcoef 1.57x / cov 1.51x (n_vars=50) — gram family, heavily
  multi-agent-contended; investigate separately.
- NEUTRAL: flipud 3.62x but absolute 1.5us (pure call overhead — not a real gap).
