# fnp-ufunc Public API Goldens

`public_api.golden` snapshots the current observable output for commonly used
`fnp_ufunc::UFuncArray` entry points: creation, broadcasting, elementwise
ufuncs, reductions, slicing, scalar indexing, shape transforms, joins,
selection/order operations, set/index helpers, contractions, cumulative ops,
and comparison summaries.

Refresh intentionally with:

```bash
UPDATE_GOLDENS=1 cargo test -p fnp-ufunc public_api_output_matches_golden
```

Then inspect the diff before committing. These goldens are regression fixtures;
they should change only when a deliberate public behavior change is backed by
NumPy parity evidence or a tracked compatibility bead.
