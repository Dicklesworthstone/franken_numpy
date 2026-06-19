# 2026-06-19 Random vs NumPy PCG Gauntlet

Scenario: compare recent `fnp-random` PCG backlog against original NumPy using Criterion.

Subject:
- FrankenNumPy `fnp-random`
- Random subject commit before measured commit: `e32d58ea`
- Integration base before this commit: `70bae5da`; intervening changes did not touch `fnp-random`
- Target dir: `/data/projects/.rch-targets/franken_numpy-cod-a`

Oracle:
- Original NumPy API: `np.random.Generator(np.random.PCG64(42))`
- Local `python3` identity observed before the run: `/usr/bin/python3`, NumPy `2.4.3`

Artifacts:
- `criterion_random_vs_numpy_prerevert.txt`: measured `.255` raw-fill keep plus `.257` bytes word-fill candidate before the revert.
- `criterion_random_vs_numpy_post_revert.txt`: measured final code after removing the losing `.257` bytes word-fill production path.
- `criterion_random_vs_numpy.txt`: initial capture, preserved by copying to the pre-revert artifact before rerun.

Outcome:
- `.255` raw PCG64 `fill_u64`: kept.
- `.257` PCG bytes word-fill: rejected and reverted.
