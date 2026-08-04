# fnp-ufunc compress bool-mask BOLD-VERIFY scorecard

Bead: `franken_numpy-ixs5y.263`

Lever kept: `UFuncArray::compress(condition, axis=None)` now uses a sidecar-free
F64 fast path that decodes the bool mask once in 8-lane bitmask chunks and
writes selected values directly into one output buffer. It does not reuse the
rejected `.249` per-chunk `Vec<Vec<f64>>` parallel gather.

## Decision rows

| Row | FrankenNumPy | NumPy | Ratio | Verdict |
|---|---:|---:|---:|---|
| 100k same-host local | 44.374 us | 52.056 us | 0.852x | keep |
| 1M same-host local | 410.823 us | 503.993 us | 0.815x | keep |
| 100k remote `hz2` routing | 33.082 us | 52.056 us local | 0.635x | routing confirmation |
| 1M remote `hz2` routing | 339.188 us | 503.993 us local | 0.673x | routing confirmation |

Final same-host win/loss/neutral vs NumPy: `2/0/0`.

Superseded candidates:
- Two-pass exact allocation: 100k lost 1.63x, 1M lost 1.03x.
- Single-pass full input capacity: 100k lost 1.01x, 1M lost 1.01x.

Baseline gap on current `origin/main` before the kept path:
- 100k: FNP 113.207 us vs NumPy 52.056 us, 2.18x slower.
- 1M: FNP 1.232777 ms vs NumPy 503.993 us, 2.45x slower.

## Validation

- `rch exec -- cargo test -p fnp-ufunc compress -- --nocapture`: pass.
- `rch exec -- cargo check -p fnp-ufunc --all-targets`: pass.
- `rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings`: pass.
- `git diff --check`: pass.
- `cargo fmt --check -p fnp-ufunc`: fails on pre-existing unrelated fnp-ufunc
  formatting drift; see `cargo_fmt_check_fnp_ufunc.txt`.
- `ubs` on changed files: exits nonzero on broad pre-existing fnp-ufunc
  inventory (489 critical, 14639 warnings); sampled findings are outside the
  new compress helper/path. See `ubs_changed_files.txt`.

Same-worker NumPy note: direct SSH to the `hz2` worker selected by the final
remote candidate failed with `Permission denied (publickey,password)`, so the
keep decision uses the local FNP Criterion confirmation against the local NumPy
probe and keeps the `hz2` row as routing confirmation.
