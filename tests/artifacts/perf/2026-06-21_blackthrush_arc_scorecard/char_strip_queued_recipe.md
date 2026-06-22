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

---

## ADDITIONAL char win-veins (build-free assessment 2026-06-22, padding/binary ops)

| op | numpy ns/el | output width | verdict |
|----|-------------|--------------|---------|
| char.ljust / rjust / center | 37-38 ns | = pad width (KNOWN) | WIN-VEIN, tractable (justify + fillchar pad) |
| char.add | 36 ns | = w1 + w2 (KNOWN) | WIN-VEIN, tractable + ALL-UNICODE (concat, no ASCII gate) |
| char.multiply | 53 ns | = w * n (KNOWN) | WIN-VEIN, tractable + all-unicode (repeat) |
| char.expandtabs | 63 ns | variable | win-vein but variable width -> harder, defer |
| char.zfill | 10 ns | (width) | numpy-C-fast -> NO vein (skip) |
| char.mod | -- | -- | fnp ALREADY WINS 0.06x (16x) -> done |

PRIORITY when disk recovers (batch all in ONE build):
1. char/strings.strip/lstrip/rstrip (same-width, ASCII, see above) — cleanest.
2. char/strings.add (concat -> w1+w2, ALL-UNICODE no ASCII gate, common) — clean + broad.
3. char/strings.multiply (repeat -> w*n, all-unicode).
4. char/strings.ljust/rjust/center (justify to pad width, fillchar default ' ', ASCII fast / non-ASCII
   fillchar or content delegate). Output dtype <U(max(w, width)).

add/multiply are ALL-UNICODE (pure copy/repeat, dtype-independent) -> no ASCII gate, widest win.
ljust/center need fillchar handling (default space; non-default ASCII ok; verify numpy padding side
for center with odd remainder = extra pad on RIGHT). zfill numpy-C-fast (skip). expandtabs deferred.
All width-changing ops: build numpy.empty(shape, dtype='<U{outw}'), view uint32, per-slot fill, no
re-pack (output width is known/fixed). Verify bit-exact + conformance_strings_namespace before commit.
