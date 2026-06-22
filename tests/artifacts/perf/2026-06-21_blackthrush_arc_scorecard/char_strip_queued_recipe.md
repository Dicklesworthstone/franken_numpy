# QUEUED (disk-critical, NO builds yet): char/strings strip family win-vein (BlackThrush)

Build-free assessment 2026-06-22 (against existing .probe/.so). Per the refined char win-rule
(parity-delegate is a win-vein ONLY when numpy is also slow), measured numpy ABSOLUTE ns/element
to classify the char search/trim ops:

| op | numpy ns/el | verdict |
|----|-------------|---------|
| char.find / index / rfind / count | 9-14 ns | numpy C-fast -> NO vein (skip; refined rule confirmed) |
| char.strip / lstrip / rstrip | 32-34 ns | **WIN-VEIN** (numpy slow, tractable) |
| char.replace | 80 ns | win-vein but LENGTH-CHANGING (output width widens, e.g. <U3 a->bb => <U6) -> harder, defer |
| char.split | 457 ns | win-vein but OBJECT-array output (list per elem) -> skip |
| char.encode | 231 ns | win-vein but BYTES output + encoding -> skip |

## Ready-to-implement: strip/lstrip/rstrip (default whitespace, no chars arg)

VERIFIED numpy semantics (build-free):
- Output PRESERVES field width: `<U6` in -> `<U6` out (stripped content left-justified, NUL-padded).
  So NO re-pack / width recompute needed -> same shape as the case ops.
- ASCII whitespace set (Python str.strip / isspace): {0x09-0x0D (\t\n\v\f\r), 0x1C-0x1F, 0x20}.
- empty / all-whitespace -> '' (all NUL).

Design (mirror try_zerocopy_unicode_ascii_cap_title — per fixed-width slot, w = itemsize/4):
- ASCII gate: if any codepoint > 0x7f -> Ok(None) delegate (numpy handles unicode whitespace).
- Only the no-chars-arg form fast-paths; a `chars=` argument -> delegate.
- Per slot (codepoints, trailing NUL already meaningless padding):
  - logical len L = last non-NUL + 1.
  - lstrip: start = first index in 0..L that is not whitespace (or L if all ws).
  - rstrip: end = last index in 0..L that is not whitespace, +1 (or 0).
  - strip: both.
  - copy slot[start..end] to out[0..end-start], NUL-fill out[end-start..w].
- Output: numpy.empty_like(codepoints view) then view back as dtype (same as case ops).
- Parallelize over slots if N*w large (rayon par_chunks) — lanes independent.

Pyfunctions: char_strip_ascii/lstrip/rstrip + strings_* (6) routing to a
unicode_ascii_strip_or_numpy(py, a, namespace, method, mode) wrapper that calls
try_zerocopy_unicode_ascii_strip(py, a, mode) else delegates. Register 6.
Signature MUST allow an optional chars arg (default None) and delegate when chars is not None
(numpy signature: char.strip(a, chars=None)).

## Verify before commit (when disk recovers)
- conformance_strings_namespace (currently 9).
- bit-exact vs numpy: strip/lstrip/rstrip on ASCII incl leading/trailing/both/all-ws/empty/no-ws/
  embedded-ws, multiple widths (U1/U6/U20), N-D, + chars-arg-delegates + non-ASCII-delegates.
- timing: expect ~0.1-0.3x (numpy 33ns/el is slow Python-ish; cf swapcase 0.12x).
- GOTCHA: helper fns ABOVE #[pyfunction] attrs (E0433). Bit-exact width preservation is the key risk.

Expected: 6 ops (strip/lstrip/rstrip x char/strings) at ~0.1-0.3x = real wins (numpy-slow confirmed).
