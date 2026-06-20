# linalg.inv — small-n shape-peek delegate (BlackThrush, 2026-06-20)

## Gap (we LOST)
np.linalg.inv of small 2-D systems (n<100) was 1.5-2.0x SLOWER than numpy: native
inv_nxn loses to OpenBLAS getrf+getri for small n, and inv extracted the operand into
a UFuncArray before any size check (the copy dominates a ~3us inv).

## Lever
Shape-peek before extraction (the solve/det getrf-cliff pattern): a 2-D square ndarray
with n<100 delegates to numpy WITHOUT extracting. numpy's getri cliffs sharply at n~100
(n<=96 ~1.7x faster than native; n>=100 native wins at LOW load). Bit-identical
(numpy is the oracle for the delegated path; LinAlgError identity preserved).

## MEASURED (under load, small-n is load-robust)
n=16 1.5x->1.20x, n=48 ->1.17x, n=80 ->1.07x, n=96 1.78x->1.01x (parity, delegated).
Small-n loss eliminated.

## SCOPE / NEGATIVE EVIDENCE
The large-n native path (n>=100) is UNCHANGED by this commit. inv at n>=256 is the
ERRATIC multithreaded-BLAS class (numpy getri swings 9.7-16.8ms at n=512 under load
while native inv_nxn is stable ~22ms) -- load-dependent, no clean threshold, NOT
addressed here (matches the matrix_power/pinv erratic-BLAS no-ship family). This commit
only removes the clean, load-robust small-n loss.

## Parity
n in {2,16,64,96,99,100,128,256} + singular(LinAlgError) + list + batched 0 fails;
conformance_linalg green.
