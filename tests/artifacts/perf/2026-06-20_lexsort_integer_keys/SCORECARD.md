# np.lexsort integer keys — delegate to numpy radix (BlackThrush, 2026-06-20)

## Gap (we LOST)
np.lexsort on INTEGER keys was 1.5-20x SLOWER than numpy (int8 4-key worst): numpy
uses a radix sort for integer/bool keys while fnp's native UFuncArray::lexsort is a
comparison-based stable sort. Float keys were mixed (native wins multi-key, loses 1-key).

## Lever
Peek the keys' asarray dtype kind up front; if integer/unsigned/bool, delegate to
numpy (its radix lexsort). Float keys keep the native path, which BEATS numpy for
multi-key float (3-key f64 0.36x). Mixed int/float promotes to float -> native.

## MEASURED
2key_int 1.54x->1.00x; 4key_int8 20.12x->1.00x; 2key_manydup 1.83x->0.99x;
3key_f64 0.36x (native, unchanged win). LOSS -> parity, float win preserved.

## Parity
int/float/mixed/tuple/2-D/many-dup keys 0 fails (array_equal); conformance_lexsort
16/16 green.
