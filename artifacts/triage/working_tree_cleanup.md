# Working Tree Cleanup Triage (franken_numpy-rosd)

Generated from `git status --porcelain` on 2026-04-10. This is a review-only manifest. **No deletions or moves have been performed.**

Categories:
- `project-artifact`: deliverables that should be tracked (keep/add to git)
- `tooling-output`: transient logs or tool output (candidate .gitignore)
- `fixed-bug-repro`: one-off repro files from past bugs (delete only after approval)
- `useful-script`: potentially reusable helper scripts (promote to `scripts/dev/` after approval)
- `compiled-binary`: extensionless compiled test binaries (delete after approval; add ignore patterns)
- `unclear`: needs human review

| File | Category | Disposition | Rationale |
| --- | --- | --- | --- |
| `.ntm/` | tooling-output | .gitignore | NTM session artifacts, not source |
| `apply_fixes.py` | useful-script | promote (pending approval) | One-off agent automation; keep only if still useful |
| `artifacts/baselines/cross_engine_benchmark_v1.decode_proof.json` | project-artifact | keep/add to git | Cross-engine benchmark deliverable (RaptorQ decode proof) |
| `artifacts/baselines/cross_engine_benchmark_v1.json` | project-artifact | keep/add to git | Cross-engine benchmark baseline (azql) |
| `artifacts/baselines/cross_engine_benchmark_v1.raptorq.json` | project-artifact | keep/add to git | Cross-engine benchmark sidecar |
| `artifacts/baselines/cross_engine_benchmark_v1.report.md` | project-artifact | keep/add to git | Cross-engine benchmark markdown report |
| `artifacts/baselines/cross_engine_benchmark_v1.scrub_report.json` | project-artifact | keep/add to git | Cross-engine benchmark scrub report |
| `docs/adr/ADR-001-parity-pivot.md` | unclear | keep/add to git (pending owner review) | Untracked ADR draft; confirm ownership and completeness |
| `clippy_cast.txt` | tooling-output | .gitignore | Transient lint capture |
| `clippy_casts_all.txt` | tooling-output | .gitignore | Transient lint capture |
| `clippy_conformance.txt` | tooling-output | .gitignore | Transient lint capture |
| `clippy_deep.txt` | tooling-output | .gitignore | Transient lint capture |
| `clippy_indexing.txt` | tooling-output | .gitignore | Transient lint capture |
| `clippy_indexing_new.txt` | tooling-output | .gitignore | Transient lint capture |
| `clippy_issues.txt` | tooling-output | .gitignore | Transient lint capture |
| `clippy_json.txt` | tooling-output | .gitignore | Transient lint capture |
| `clippy_new.txt` | tooling-output | .gitignore | Transient lint capture |
| `clippy_out.txt` | tooling-output | .gitignore | Transient lint capture |
| `clippy_out2.txt` | tooling-output | .gitignore | Transient lint capture |
| `clippy_out3.txt` | tooling-output | .gitignore | Transient lint capture |
| `clippy_out4.txt` | tooling-output | .gitignore | Transient lint capture |
| `clippy_pedantic.txt` | tooling-output | .gitignore | Transient lint capture |
| `clippy_pedantic_2.txt` | tooling-output | .gitignore | Transient lint capture |
| `clippy_pedantic_all.txt` | tooling-output | .gitignore | Transient lint capture |
| `clippy_pedantic_new.txt` | tooling-output | .gitignore | Transient lint capture |
| `clippy_ptr.txt` | tooling-output | .gitignore | Transient lint capture |
| `clippy_suspicious.txt` | tooling-output | .gitignore | Transient lint capture |
| `clippy_unwrap.txt` | tooling-output | .gitignore | Transient lint capture |
| `deep_clippy.txt` | tooling-output | .gitignore | Transient lint capture |
| `fix_clippy.py` | useful-script | promote (pending approval) | One-off agent automation; keep only if still useful |
| `fix_product.py` | useful-script | promote (pending approval) | One-off agent automation; keep only if still useful |
| `get_context.py` | useful-script | promote (pending approval) | One-off agent automation; keep only if still useful |
| `patch_oracle` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `patch_oracle.rs` | fixed-bug-repro | delete (pending approval) | One-off oracle patch experiment |
| `rust-bug-scan.txt` | tooling-output | .gitignore | Transient tool output |
| `test_arange` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `test_cast_str.rs` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_contiguous` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `test_cow_bug.rs` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_digitize` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `test_digitize.py` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_digitize_nan` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `test_div` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `test_f16.rs` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_fnp_digitize.rs` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_idx` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `test_idx.rs` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_max` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `test_max_rows` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `test_max_rows.rs` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_min` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `test_multiple` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `test_multiple.rs` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_nan_bin` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `test_nan_cast` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `test_nan_max` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `test_nan_max.rs` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_nan_min` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `test_nat` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `test_ncx2.py` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_ncx2_fnp.rs` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_nearest.py` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_nearest_method.rs` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_neg_zero` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `test_npz.rs` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_overlap` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `test_pad` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `test_parse_json.rs` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_parse_tab` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `test_parse_tab.rs` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_polyder.py` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_polydiv.py` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_polydiv2.py` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_polydiv3.py` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_ptrs.c` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_put_along.py` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_put_along.rs` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_rem_panic.rs` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_rem_panic2` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `test_rem_panic2.rs` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_reshape.py` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_roots.rs` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_shift` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `test_shift2` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `test_skip.csv` | fixed-bug-repro | delete (pending approval) | One-off repro data |
| `test_skip.py` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_skip_fnp.rs` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `test_solve` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `test_split` | compiled-binary | delete + .gitignore (pending approval) | Extensionless compiled artifact |
| `test_trace.py` | fixed-bug-repro | delete (pending approval) | One-off repro |
| `ubs_out.txt` | tooling-output | .gitignore | Transient tool output |
