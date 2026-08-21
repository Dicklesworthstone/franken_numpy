# AGENTS.md — FrankenNumPy

> Guidelines for AI coding agents working in this Rust codebase.

---

## RULE 0 - THE FUNDAMENTAL OVERRIDE PREROGATIVE

If I tell you to do something, even if it goes against what follows below, YOU MUST LISTEN TO ME. I AM IN CHARGE, NOT YOU.

---

## RULE 0.5 - SUITE-WIDE RULES LIVE IN /data/projects/AGENTS.md

The suite-wide rules in **`/data/projects/AGENTS.md`** bind you here too. Read it. Two sections
are load-bearing for perf work and are NOT duplicated below, so they cannot drift out of sync:

- **`## Named Reward-Hacking Patterns (ALL FORBIDDEN)`** — 12 named patterns, several already
  observed in this suite: gate self-weakening (and the exact price of a legitimate gate fix),
  proof-class inflation, golden regeneration reflex, commit-stream pumping, tautological tests,
  easy-lever cherry-picking, close-pump abuse, scope-splitting, spec-editing as progress,
  conformance metastasis, dependency smuggling, bench-path hardcoding.
- **`### Work-Graph Discipline`** — JSONL is truth and `beads.db` is disposable, `br sync
  --import-only` after every pull, single-writer on graph structure, closure on cited evidence
  with blocker beads gated on their named probe, `br dep cycles` stays empty.

The three that most often decide whether a number here is real: a **self-speedup is
MAINTENANCE, not a win** — a win needs the incumbent live in the SAME invocation; **never
weaken a gate to land a change**, and if a gate is genuinely defective, meet the evidence
standard and publish the win/lose split of what the fix admits; and **reporting a loss is a
success** — one line, revert, next lever, no retraction narrative.

---

## RULE NUMBER 1: NO FILE DELETION

**YOU ARE NEVER ALLOWED TO DELETE A FILE WITHOUT EXPRESS PERMISSION.** Even a new file that you yourself created, such as a test code file. You have a horrible track record of deleting critically important files or otherwise throwing away tons of expensive work. As a result, you have permanently lost any and all rights to determine that a file or folder should be deleted.

**YOU MUST ALWAYS ASK AND RECEIVE CLEAR, WRITTEN PERMISSION BEFORE EVER DELETING A FILE OR FOLDER OF ANY KIND.**

---

## Irreversible Git & Filesystem Actions — DO NOT EVER BREAK GLASS

1. **Absolutely forbidden commands:** `git reset --hard`, `git clean -fd`, `rm -rf`, or any command that can delete or overwrite code/data must never be run unless the user explicitly provides the exact command and states, in the same message, that they understand and want the irreversible consequences.
2. **No guessing:** If there is any uncertainty about what a command might delete or overwrite, stop immediately and ask the user for specific approval. "I think it's safe" is never acceptable.
3. **Safer alternatives first:** When cleanup or rollbacks are needed, request permission to use non-destructive options (`git status`, `git diff`, `git stash`, copying to backups) before ever considering a destructive command.
4. **Mandatory explicit plan:** Even after explicit user authorization, restate the command verbatim, list exactly what will be affected, and wait for a confirmation that your understanding is correct. Only then may you execute it—if anything remains ambiguous, refuse and escalate.
5. **Document the confirmation:** When running any approved destructive command, record (in the session notes / final response) the exact user text that authorized it, the command actually run, and the execution time. If that record is absent, the operation did not happen.

---

## Git Branch: ONLY Use `main`, NEVER `master`

**The default branch is `main`. The `master` branch exists only for legacy URL compatibility.**

- **All work happens on `main`** — commits, PRs, feature branches all merge to `main`
- **Never reference `master` in code or docs** — if you see `master` anywhere, it's a bug that needs fixing
- **The `master` branch must stay synchronized with `main`** — after pushing to `main`, also push to `master`:
  ```bash
  git push origin main:master
  ```

**If you see `master` referenced anywhere:**
1. Update it to `main`
2. Ensure `master` is synchronized: `git push origin main:master`

---

## Current state (2026-05-16)

- `fnp_python` covers **100% of `numpy.__all__`** (499/499 names) — see [`docs/planning/audit_numpy_reality.md`](docs/planning/audit_numpy_reality.md) for architecture + coverage progression.
- Coverage is **structurally locked** by `fnp_python_covers_full_numpy_all` in `crates/fnp-python/tests/conformance_remaining_top_level_attrs.rs`; this test fails CI if any name regresses.
- Workspace runs 6,392 tests across 10 crates (see [`docs/planning/FEATURE_PARITY.md`](docs/planning/FEATURE_PARITY.md) for the per-crate breakdown). Underlying Rust surface: 1,575 `pub fn` declarations across `crates/*/src/**/*.rs`.
- Bead tracker stands at ~1,417 closed beads as of 2026-05-19; live count via `br list --status=closed --limit 10000 --json | jq length`.
- No real stubs/mocks/TODOs in production code — structurally enforced by `crates/fnp-conformance/tests/codebase_hygiene.rs` (8 #[test] functions fail CI on stub markers); per-site analysis in [`docs/planning/audit_numpy_mocks.md`](docs/planning/audit_numpy_mocks.md).
- Active tracked divergences: 0 rows in [`docs/DIVERGENCES.md`](docs/DIVERGENCES.md); `fnp-random` `SeedMaterial::None` now sources OS entropy for no-seed NumPy parity (closed by bead `franken_numpy-iqo31`).

## Toolchain: Rust & Cargo

We only use **Cargo** in this project, NEVER any other package manager.

- **Edition:** Rust 2024 (nightly required — pinned to `nightly-2026-07-05` in `rust-toolchain.toml`; CI mirrors the same via `RUST_TOOLCHAIN` env var in `.github/workflows/ci.yml`)
- **Dependency versions:** Explicit versions for stability
- **Configuration:** Cargo.toml workspace with `workspace = true` pattern
- **Unsafe code:** Forbidden by default (`#![forbid(unsafe_code)]`) on 9 of 10 crates — the numeric core stays entirely on the safe-Rust path, enforced by `no_unsafe_code_blocks_or_items` in `crates/fnp-conformance/tests/codebase_hygiene.rs`. `fnp-python` is the lone opt-out: as the PyO3 boundary it uses hand-written `unsafe` (chiefly `std::slice::from_raw_parts` on borrowed `PyBuffer` bytes, plus narrow layout-checked views of native result buffers) for zero-copy fast paths. Those blocks are confined to `fnp-python` and excluded from the hygiene scan; every other crate must stay unsafe-free. If narrow unsafe usage ever becomes unavoidable in one of the 9 core crates, isolate it behind audited interfaces and tests rather than relaxing the invariant.

### Key Dependencies

| Crate | Purpose |
|-------|---------|
| `asupersync` | Structured async runtime (optional in `fnp-runtime`) |
| `ftui` (frankentui) | Terminal-native observability dashboards (optional in `fnp-runtime`) |
| `serde` + `serde_json` | Serialization for conformance fixtures and artifacts |
| `serde_yaml_ng` | YAML parsing for security control maps and contracts |
| `sha2` | Content hashing for artifact integrity |
| `base64` | Encoding for artifact payloads |
| `getrandom` | OS entropy for NumPy-compatible unseeded RNG constructors |

### Build Profiles — READ THIS BEFORE REASONING ABOUT BUILD COST

**The workspace defines NO `[profile.release]`.** Verified 2026-08-16: the only
profile sections in the root `Cargo.toml` are `[profile.release-perf]`,
`[profile.bench]` and `[profile.bench-fast]`. `release` — and therefore `bench`,
which inherits it — takes **Cargo's defaults**: `opt-level = 3`, `lto = false`,
`codegen-units = 16`.

This section previously showed a `[profile.release]` block with `lto = true` and
`codegen-units = 1`. That block is not in `Cargo.toml` and never governed a build
here. It is corrected rather than deleted because it actively misled at least one
agent (me) into asserting that `cargo bench` does a full LTO build and that this
explains rch SSH-ceiling timeouts. It does not: `bench` and `bench-fast` are
configured **identically** in optimization terms, so switching between them cannot
change build time by construction.

| profile | opt-level | lto | codegen-units | use |
|---|---|---|---|---|
| `release` / `bench` (Cargo defaults) | 3 | false | 16 | triage |
| `bench-fast` (explicit, same settings) | 3 | false | 16 | triage |
| `release-perf` | 3 | thin | 1 | **ship-grade ratios** |

Ship-grade fnp-vs-NumPy ratios must be confirmed under `--profile release-perf`;
anything at `bench`/`bench-fast` is triage grade and the row must say so.

---

## Code Editing Discipline

### No Script-Based Changes

**NEVER** run a script that processes/changes code files in this repo. Brittle regex-based transformations create far more problems than they solve.

- **Always make code changes manually**, even when there are many instances
- For many simple changes: use parallel subagents
- For subtle/complex changes: do them methodically yourself

### No File Proliferation

If you want to change something or add a feature, **revise existing code files in place**.

**NEVER** create variations like:
- `mainV2.rs`
- `main_improved.rs`
- `main_enhanced.rs`

New files are reserved for **genuinely new functionality** that makes zero sense to include in any existing file. The bar for creating new files is **incredibly high**.

---

## Backwards Compatibility

We do not care about backwards compatibility—we're in early development with no users. We want to do things the **RIGHT** way with **NO TECH DEBT**.

- Never create "compatibility shims"
- Never create wrapper functions for deprecated APIs
- Just fix the code directly

---

## Compiler Checks (CRITICAL)

**After any substantive code changes, you MUST verify no errors were introduced:**

```bash
# Check for compiler errors and warnings (workspace-wide)
cargo check --workspace --all-targets

# Check for clippy lints — workspace uses rustc/clippy default lint set
# (no clippy.toml or per-crate clippy::pedantic/nursery opt-ins; verified
# via grep). The G1 CI gate runs the same command + treats every clippy
# warning as a hard error via `-D warnings`.
cargo clippy --workspace --all-targets -- -D warnings

# Verify formatting
cargo fmt --check
```

If you see errors, **carefully understand and resolve each issue**. Read sufficient context to fix them the RIGHT way.

---

## Testing

### Testing Policy

Every component crate includes inline `#[cfg(test)]` unit tests alongside the implementation. Tests must cover:
- Happy path
- Edge cases (empty input, max values, boundary conditions)
- Error conditions

Cross-component integration tests live in crate-level `tests/` directories. The `fnp-conformance` crate contains the differential harness, adversarial policy harness, security-contract validator, oracle capture, and benchmark + RaptorQ artifact tooling. It also ships `tests/codebase_hygiene.rs` which uses `rg` (ripgrep) to enforce the no-stubs invariant — install `ripgrep` (`apt-get install ripgrep`, `brew install ripgrep`, or `cargo install ripgrep`) before running `cargo test -p fnp-conformance`, or the 8 hygiene tests will panic with "rg should be available".

### Unit Tests

```bash
# Run all tests across the workspace
cargo test --workspace

# Run with output
cargo test --workspace -- --nocapture

# Run tests for a specific crate
cargo test -p fnp-dtype
cargo test -p fnp-ndarray
cargo test -p fnp-iter
cargo test -p fnp-ufunc
cargo test -p fnp-linalg
cargo test -p fnp-random
cargo test -p fnp-io
cargo test -p fnp-python
cargo test -p fnp-conformance
cargo test -p fnp-runtime

# Run tests with all features enabled
cargo test --workspace --all-features
```

Cost note: `fnp-ufunc` (2,191 tests, ~60k LOC) and `fnp-python` (2,127 tests) dominate workspace test time. When iterating on a focused change in another crate, prefer the targeted `-p` invocation. See [`docs/planning/FEATURE_PARITY.md`](docs/planning/FEATURE_PARITY.md) for the live per-crate test counts.

### Test Categories

| Crate | Focus Areas |
|-------|-------------|
| `fnp-dtype` | Dtype taxonomy, promotion table determinism, cast policy primitives |
| `fnp-ndarray` | Shape legality, stride calculus (C/F contiguous), reshape `-1` inference, broadcast contracts, alias-sensitive transitions |
| `fnp-iter` | Transfer-loop selector, overlap detection, `Nditer` / `NditerPlan` / `NditerStep` state machine with `iterindex` / `multi_index` / reset / seek / external-loop chunks, `nditer_python*` parity bridge against the live numpy nditer |
| `fnp-ufunc` | 35 binary + 43 unary elementwise ops, 30+ reductions, FFT (Cooley-Tukey + Bluestein), einsum (`einsum`, `einsum_path`, `einsum_optimized`), masked / string / datetime arrays, polynomial families (power, Chebyshev, Legendre, Hermite, Laguerre), float error state machine, NaN-correct reductions |
| `fnp-linalg` | ~100 public functions: 2×2 fast paths, NxN decompositions (QR, SVD, eig, eigh, Cholesky, LU), spectral methods (`expm`, `sqrtm`, `logm`, `funm`, `polar`, `schur`), least-squares, 14 batched ops, complex variants |
| `fnp-random` | 5 production bit generators (PCG64, PCG64DXSM, MT19937, Philox, SFC64) + an internal `DeterministicRng` for tests, full `SeedSequence` / `SeedMaterial` hierarchy with spawn lineage, pickle payload round-trip, `RandomState` legacy wrapper, 40+ oracle-verified distributions with bit-exact PCG64DXSM parity vs NumPy |
| `fnp-io` | npy/npz parser/writer, hardened boundary checks, adversarial input fuzzing |
| `fnp-conformance` | Fixture-driven differential suites, oracle capture, adversarial/security policy harnesses, benchmark baselines, RaptorQ sidecar/scrub/decode proofs, workflow scenario gates |
| `fnp-python` | PyO3 bindings exposing 100% of `numpy.__all__` (499/499 names) structurally locked by `fnp_python_covers_full_numpy_all`, plus 133 dedicated `conformance_*.rs` parity shards under `crates/fnp-python/tests/` |
| `fnp-runtime` | Strict/hardened mode split, fail-closed wire decoding, override-audit gate, decision/evidence ledger |

### Conformance and Artifact Commands

```bash
cargo run -p fnp-conformance --bin capture_numpy_oracle
cargo run -p fnp-conformance --bin run_ufunc_differential
cargo run -p fnp-conformance --bin generate_benchmark_baseline
cargo run -p fnp-conformance --bin run_performance_budget_gate
cargo run -p fnp-conformance --bin generate_raptorq_sidecars
cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-001
cargo run -p fnp-conformance --bin run_security_gate
cargo run -p fnp-conformance --bin run_test_contract_gate
cargo run -p fnp-conformance --bin run_workflow_scenario_gate
cargo run -p fnp-conformance --bin run_divergence_ledger -- --fail-on-missing
cargo run -p fnp-conformance --bin run_fnp_python_api_coverage -- --fail-on-missing
cargo run -p fnp-conformance --bin run_diagnostic_oracle
cargo run -p fnp-conformance --bin run_oracle_drift_matrix
cargo run -p fnp-conformance --bin run_io_diagnostics
cargo run -p fnp-conformance --bin validate_phase2c_stale_claims
scripts/e2e/run_ci_gate_topology.sh        # runs all 8 gates G1-G8 in order + closing P2C-001..009 packet sweep
scripts/e2e/run_performance_budget_gate.sh
scripts/e2e/run_security_policy_gate.sh
scripts/e2e/run_test_contract_gate.sh
scripts/e2e/run_workflow_scenario_gate.sh
```

Oracle capture uses configurable interpreter `FNP_ORACLE_PYTHON` (defaults to `python3`). Example with `uv`:

```bash
uv venv --python 3.14 .venv-numpy314
uv pip install --python .venv-numpy314/bin/python numpy
FNP_ORACLE_PYTHON="$(pwd)/.venv-numpy314/bin/python3" cargo run -p fnp-conformance --bin capture_numpy_oracle
```

---

## Third-Party Library Usage

If you aren't 100% sure how to use a third-party library, **SEARCH ONLINE** to find the latest documentation and current best practices.

---

## FrankenNumPy — This Project

**This is the project you're working on.** FrankenNumPy is a memory-safe, clean-room Rust reimplementation of NumPy with two simultaneous goals: (1) absolutely complete and total drop-in behavioral compatibility with legacy NumPy (no reduced-scope acceptance), and (2) a more rigorous architecture for reliability, performance, and explainability.

### What It Does

Reimplements NumPy's array API in Rust with deterministic shape/stride/broadcast semantics, full dtype promotion/casting parity, ufunc dispatch, and a dual-mode (strict/hardened) runtime — all verified by differential conformance against a legacy NumPy oracle.

### Crown-Jewel Innovation: Stride Calculus Engine (SCE)

The **Stride Calculus Engine (SCE)** provides deterministic shape/stride/broadcast legality and zero-copy view guarantees. SCE owns all shape transformation rules:

1. `shape -> element_count` with overflow checks
2. `shape + order + item_size -> strides` (C/F contiguous baselines)
3. `lhs_shape + rhs_shape -> broadcast_shape` deterministically
4. `old_count + reshape_spec -> resolved_shape` with NumPy-style `-1` semantics
5. Alias-sensitive transitions rejected when invariants are violated

**CRITICAL NON-REGRESSION RULE:** Broadcast, stride/view aliasing, and dtype promotion rules are non-negotiable contracts. SCE is the non-negotiable compatibility kernel.

### Architecture

```
array API -> shape/stride engine (SCE) -> ufunc dispatcher -> numeric kernels -> IO
```

Layering principles:
1. Spec-first implementation from extraction packets
2. Strict/hardened compatibility mode split is mandatory
3. Fail-closed behavior for unknown or incompatible semantics
4. Profile-first optimization with one-lever, proof-backed changes
5. Durable evidence artifacts with RaptorQ sidecar contracts

### Workspace Structure

```
franken_numpy/
├── Cargo.toml                         # Workspace root (10 crates)
├── crates/
│   ├── fnp-dtype/                     # Dtype taxonomy, promotion table, cast policy primitives
│   ├── fnp-ndarray/                   # Shape legality, stride calculus, reshape/broadcast contracts
│   ├── fnp-iter/                      # Transfer semantics, overlap-safe iteration, Nditer state machine
│   ├── fnp-ufunc/                     # 850+ array operations, reductions, einsum, masked arrays
│   ├── fnp-linalg/                    # solve, eig, svd, qr, cholesky, lstsq, batched, complex
│   ├── fnp-random/                    # 5 bit generators, distributions, PCG64DXSM bit-exact parity
│   ├── fnp-io/                        # NPY/NPZ read/write, text I/O, DEFLATE, memmap
│   ├── fnp-python/                    # PyO3 bindings, 100% numpy.__all__ surface
│   ├── fnp-conformance/               # Oracle capture, differential / metamorphic / adversarial / RaptorQ
│   └── fnp-runtime/                   # Strict/hardened mode, evidence ledger, decision engine
├── legacy_numpy_code/numpy/           # Behavioral oracle (upstream: github.com/numpy/numpy)
├── artifacts/                         # Contract schemas, security controls, logs
├── scripts/                           # E2E gate scripts
└── docs/                              # Specs and planning documents
```

### Key Design Decisions

- **Dual-mode runtime (strict/hardened):** Strict mode maximizes observable compatibility for the full legacy NumPy behavior matrix with no behavior-altering repairs. Hardened mode preserves the API contract while adding safety guards and bounded defensive recovery for malformed inputs and hostile edge cases. All decisions are recorded in an evidence ledger. Unknown wire mode/class inputs are fail-closed.
- **Legacy behavioral oracle:** `/dp/franken_numpy/legacy_numpy_code/numpy` (upstream: `https://github.com/numpy/numpy`) provides ground-truth for differential conformance testing
- **Conformance pipeline:** For each feature family: input fixtures, oracle capture, target execution, parity comparison report, durability sidecars + scrub + decode proof
- **RaptorQ-everywhere durability:** Conformance fixture bundles, benchmark baseline bundles, migration manifests, reproducibility ledgers, and long-lived state snapshots all require repair-symbol sidecars, integrity scrub reports, and decode proof artifacts for each recovery event
- **Security doctrine:** Harden parser/IO and shape-validation boundaries; prevent malformed shape and unsafe cast pathways. Fail-closed for unknown incompatible features. Adversarial fixture coverage and fuzz/property tests for high-risk parsers/state transitions (the live harness ships 7 fuzz crates / 27 targets / ~200 curated seeds — see [`docs/FUZZING.md`](docs/FUZZING.md)). Deterministic audit logs for recoveries and policy overrides
- **Performance governance:** Measure op-family throughput, tail latency, and memory bandwidth efficiency; gate regressions for broadcast and reduction hotspots. Every optimization follows: baseline (p50/p95/p99 + memory), profile hotspot, implement one lever, prove behavior unchanged via conformance + invariant checks, re-baseline and emit delta artifact
- **Correctness doctrine:** Maintain deterministic shape calculus, alias correctness, and dtype promotion table invariants. Required evidence for substantive changes: differential conformance report, invariant checklist update, benchmark delta report, risk-note update if threat or compatibility surface changed
- **Parity debt, not feature cuts:** Surface parity is locked at 100% of `numpy.__all__` (499/499, structurally enforced by `fnp_python_covers_full_numpy_all`). Remaining work is per-symbol behavioral parity (matching NumPy's edge-case semantics, dtype promotions, and error paths) — that is parity debt to be closed, not accepted feature cuts
- **Feature flags (3 total across the workspace):** `fnp-python` has `python-extension` (PyO3 cdylib mode, default = off; enable with `cargo build -p fnp-python --features python-extension`); `fnp-runtime` has `asupersync` (async orchestration of conformance capture, artifact pipelines, cancellation-safe long-running jobs, structured telemetry; gates 5 `#[cfg(feature = "asupersync")]` blocks) and `frankentui` (terminal-native dashboards for parity drift and perf deltas). All three default to off. CI runs the default-features build; the `[features]` blocks in `fnp-python/Cargo.toml` and `fnp-runtime/Cargo.toml` are the authoritative source of truth.

### Runtime Mode Matrix

| Input Class | Strict Mode | Hardened Mode |
|---|---|---|
| Known compatible + low risk | allow | allow |
| Known compatible + high risk | allow | full_validate |
| Unknown semantics | fail_closed | fail_closed |
| Known incompatible semantics | fail_closed | fail_closed |

---

## MCP Agent Mail — Multi-Agent Coordination

A mail-like layer that lets coding agents coordinate asynchronously via MCP tools and resources. Provides identities, inbox/outbox, searchable threads, and advisory file reservations with human-auditable artifacts in Git.

### Why It's Useful

- **Prevents conflicts:** Explicit file reservations (leases) for files/globs
- **Token-efficient:** Messages stored in per-project archive, not in context
- **Quick reads:** `resource://inbox/...`, `resource://thread/...`

### Same Repository Workflow

1. **Register identity:**
   ```
   ensure_project(project_key=<abs-path>)
   register_agent(project_key, program, model)
   ```

2. **Reserve files before editing:**
   ```
   file_reservation_paths(project_key, agent_name, ["src/**"], ttl_seconds=3600, exclusive=true)
   ```

3. **Communicate with threads:**
   ```
   send_message(..., thread_id="FEAT-123")
   fetch_inbox(project_key, agent_name)
   acknowledge_message(project_key, agent_name, message_id)
   ```

4. **Quick reads:**
   ```
   resource://inbox/{Agent}?project=<abs-path>&limit=20
   resource://thread/{id}?project=<abs-path>&include_bodies=true
   ```

### Macros vs Granular Tools

- **Prefer macros for speed:** `macro_start_session`, `macro_prepare_thread`, `macro_file_reservation_cycle`, `macro_contact_handshake`
- **Use granular tools for control:** `register_agent`, `file_reservation_paths`, `send_message`, `fetch_inbox`, `acknowledge_message`

### Common Pitfalls

- `"from_agent not registered"`: Always `register_agent` in the correct `project_key` first
- `"FILE_RESERVATION_CONFLICT"`: Adjust patterns, wait for expiry, or use non-exclusive reservation
- **Auth errors:** If JWT+JWKS enabled, include bearer token with matching `kid`

---

## Beads (br) — Dependency-Aware Issue Tracking

Beads provides a lightweight, dependency-aware issue database and CLI (`br` - beads_rust) for selecting "ready work," setting priorities, and tracking status. It complements MCP Agent Mail's messaging and file reservations.

**Important:** `br` is non-invasive—it NEVER runs git commands automatically. You must manually commit changes after `br sync --flush-only`.

### Conventions

- **Single source of truth:** Beads for task status/priority/dependencies; Agent Mail for conversation and audit
- **Shared identifiers:** Use Beads issue ID (e.g., `br-123`) as Mail `thread_id` and prefix subjects with `[br-123]`
- **Reservations:** When starting a task, call `file_reservation_paths()` with the issue ID in `reason`

### Typical Agent Flow

1. **Pick ready work (Beads):**
   ```bash
   br ready --json  # Choose highest priority, no blockers
   ```

2. **Reserve edit surface (Mail):**
   ```
   file_reservation_paths(project_key, agent_name, ["src/**"], ttl_seconds=3600, exclusive=true, reason="br-123")
   ```

3. **Announce start (Mail):**
   ```
   send_message(..., thread_id="br-123", subject="[br-123] Start: <title>", ack_required=true)
   ```

4. **Work and update:** Reply in-thread with progress

5. **Complete and release:**
   ```bash
   br close 123 --reason "Completed"
   br sync --flush-only  # Export to JSONL (no git operations)
   ```
   ```
   release_file_reservations(project_key, agent_name, paths=["src/**"])
   ```
   Final Mail reply: `[br-123] Completed` with summary

### Mapping Cheat Sheet

| Concept | Value |
|---------|-------|
| Mail `thread_id` | `br-###` |
| Mail subject | `[br-###] ...` |
| File reservation `reason` | `br-###` |
| Commit messages | Include `br-###` for traceability |

---

## bv — Graph-Aware Triage Engine

bv is a graph-aware triage engine for Beads projects (`.beads/beads.jsonl`). It computes PageRank, betweenness, critical path, cycles, HITS, eigenvector, and k-core metrics deterministically.

**Scope boundary:** bv handles *what to work on* (triage, priority, planning). For agent-to-agent coordination (messaging, work claiming, file reservations), use MCP Agent Mail.

**CRITICAL: Use ONLY `--robot-*` flags. Bare `bv` launches an interactive TUI that blocks your session.**

### The Workflow: Start With Triage

**`bv --robot-triage` is your single entry point.** It returns:
- `quick_ref`: at-a-glance counts + top 3 picks
- `recommendations`: ranked actionable items with scores, reasons, unblock info
- `quick_wins`: low-effort high-impact items
- `blockers_to_clear`: items that unblock the most downstream work
- `project_health`: status/type/priority distributions, graph metrics
- `commands`: copy-paste shell commands for next steps

```bash
bv --robot-triage        # THE MEGA-COMMAND: start here
bv --robot-next          # Minimal: just the single top pick + claim command
```

### Command Reference

**Planning:**
| Command | Returns |
|---------|---------|
| `--robot-plan` | Parallel execution tracks with `unblocks` lists |
| `--robot-priority` | Priority misalignment detection with confidence |

**Graph Analysis:**
| Command | Returns |
|---------|---------|
| `--robot-insights` | Full metrics: PageRank, betweenness, HITS, eigenvector, critical path, cycles, k-core, articulation points, slack |
| `--robot-label-health` | Per-label health: `health_level`, `velocity_score`, `staleness`, `blocked_count` |
| `--robot-label-flow` | Cross-label dependency: `flow_matrix`, `dependencies`, `bottleneck_labels` |
| `--robot-label-attention [--attention-limit=N]` | Attention-ranked labels |

**History & Change Tracking:**
| Command | Returns |
|---------|---------|
| `--robot-history` | Bead-to-commit correlations |
| `--robot-diff --diff-since <ref>` | Changes since ref: new/closed/modified issues, cycles |

**Other:**
| Command | Returns |
|---------|---------|
| `--robot-burndown <sprint>` | Sprint burndown, scope changes, at-risk items |
| `--robot-forecast <id\|all>` | ETA predictions with dependency-aware scheduling |
| `--robot-alerts` | Stale issues, blocking cascades, priority mismatches |
| `--robot-suggest` | Hygiene: duplicates, missing deps, label suggestions |
| `--robot-graph [--graph-format=json\|dot\|mermaid]` | Dependency graph export |
| `--export-graph <file.html>` | Interactive HTML visualization |

### Scoping & Filtering

```bash
bv --robot-plan --label backend              # Scope to label's subgraph
bv --robot-insights --as-of HEAD~30          # Historical point-in-time
bv --recipe actionable --robot-plan          # Pre-filter: ready to work
bv --recipe high-impact --robot-triage       # Pre-filter: top PageRank
bv --robot-triage --robot-triage-by-track    # Group by parallel work streams
bv --robot-triage --robot-triage-by-label    # Group by domain
```

### Understanding Robot Output

**All robot JSON includes:**
- `data_hash` — Fingerprint of source beads.jsonl
- `status` — Per-metric state: `computed|approx|timeout|skipped` + elapsed ms
- `as_of` / `as_of_commit` — Present when using `--as-of`

**Two-phase analysis:**
- **Phase 1 (instant):** degree, topo sort, density
- **Phase 2 (async, 500ms timeout):** PageRank, betweenness, HITS, eigenvector, cycles

### jq Quick Reference

```bash
bv --robot-triage | jq '.quick_ref'                        # At-a-glance summary
bv --robot-triage | jq '.recommendations[0]'               # Top recommendation
bv --robot-plan | jq '.plan.summary.highest_impact'        # Best unblock target
bv --robot-insights | jq '.status'                         # Check metric readiness
bv --robot-insights | jq '.Cycles'                         # Circular deps (must fix!)
```

---

## Performance Ledger — preflight before you optimize, evidence before you reject

`docs/NEGATIVE_EVIDENCE.md` is the append-only record of every performance
hypothesis: wins, losses, and the retry predicate for each. It is 1,000+ entries
and it is the authoritative record — not `cass`, not memory, not the commit log.

**Ledger integrity decays.** The corrected 2026-07-27 hand audit classified 109
actual rejected levers and found 71 (65.1%) **VOID**. Sixty-six of those 71 were
an A/B rejected on a near-1.0 ratio with no A/A null control and no counted
mechanism recorded. Two gates now exist so that class cannot grow. Use them.

### Before you touch source for a perf candidate

```bash
RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- \
  cargo run -q -p fnp-conformance --bin perf_ledger_preflight -- \
  --lever "selected bool parse" --surface "loadtxt usecols"
```

| Exit | Meaning |
|---|---|
| `0` CLEAR | No prior ledger row matched. |
| `2` BLOCKED | Prior evidence matched; the command prints each row and its retry predicate. Satisfy that predicate before reopening it. |
| `64` | usage or tool error |

`scripts/ledger_preflight.sh <keyword> ...` remains a fast heading-only triage
helper, but its regex classification is not authoritative. Use the Rust
preflight above for a real proposal and rely on its staged audit in pre-commit.

### A/A null gate semantics — audited and corrected 2026-07-30

The fleet found a defect in the mandated harness contract: an A/A null clause
that vetoed a row unless the null's interval INCLUDED 1.0. That couples the
verdict to the null's *precision* in the wrong direction — a tighter, better
null is more likely to exclude 1.0 and veto a real effect.

**This repo's live contract does not have that defect.**
`report_dual_null_contract_gate` requires the EFFECT bootstrap median CI to
exclude 1.0, compares the effect median against both null CI envelopes, and
requires the effect-median deviation to exceed twice the larger null half-width
(measured from 1.0). It never requires a null CI to contain 1.0. Proof from
banked rows that the null-straddle veto is absent: the 2,048-account
access-control incumbent null `[0.982472, 0.997568]`, the f16 telemetry
candidate null `[1.069905, 1.179077]`, and the f16 vector-field candidate null
`[0.931902, 0.994990]` all excluded 1.0 while their rows scored
`DECIDABLE_WIN`.

**Full-reasoning correction to the provisional audit:** commit `121d793c`
correctly diagnosed the null-straddle defect as absent, but incorrectly claimed
that the live gate already enforced effect-CI exclusion. At that commit the
gate used only the effect median. The contract now enforces effect-CI exclusion
explicitly and fail-fast checks that a synthetic effect whose CI crosses 1.0 is
`UNDECIDED`, even when its point estimate would otherwise clear the null
envelope and 2x margin.

**A dormant copy of the defect survives** in the legacy
`report_median_gate_pair` (`benches/common/mod.rs`, `verdict=BIASED_NULL`), used
by 48 arms across 8 groups. Audited with two passes of one identical ELF on a
drained host: effects reproduced within 1.6-8.9%, verdicts were WIN 6/6, no veto
fired — those microbench nulls are wide (p10..p90 spans 4-24%). Verdict-stable
gates get left alone, so it is unchanged, but the hazard comment at that line
tells you what to do if you ever tighten those arms.

**Before adopting a "null median within 2% of 1.0" clause, read this.** The
effect-CI exclusion and 2x-null-margin clauses are enforced here. A bare 2%
point-estimate clause was measured on this repo's workload surface and is NOT
stable at 21 rounds: across two runs of the identical ELF the clause rejected a
*different* size point each time (8,192 at 3.208% off unity in one run; 2,048 at
5.241% in the other, with 8,192 then passing at 0.259%) while the end-to-end
effect reproduced within 5.1%. A reproducible effect with a moving verdict is
the same diagnostic that condemned the straddle clause — the coupling has just
moved from the null's CI width to the null median's point-estimate noise. If you
gate on arm-order bias, bound it against its OWN uncertainty or raise the round
count until the null median's standard error is well under 2%; otherwise report
it as telemetry. Note this clause would also make the f16 telemetry health
report (candidate null 1.129564, 12.96% off unity) undecidable, so adopting it
is a retroactive re-scoring decision, not a formatting change.

### When you write a REJECT row

`crates/fnp-conformance/tests/ledger_hygiene.rs` fails CI unless a REJECT row
dated on/after its `ENFORCEMENT_DATE` records **either**:

- an **A/A null control** measured in the *same invocation* as the A/B, or
- a **counted mechanism** — instructions, cycles, syscalls, allocations, faults,
  bandwidth — unchanged. A null cannot change the fact that no work was removed.

It also requires a concrete retry predicate and a unique heading, and it caps
the grandfathered historical debt so backdating a row to dodge the gate trips a
second test instead.

### Every measured row must NAME ITS WORKER (from 2026-08-15)

`new_measured_rows_name_their_worker` fails CI unless a measured row dated
on/after `WORKER_PROVENANCE_ENFORCEMENT_DATE` carries an explicit `host=<name>`
or `worker=<name>` field. Prose like "vmi1293453 (8 logical, EPYC-IBPB)" is how
older rows did it and is NOT machine-checkable; paste the harness's own line
instead, which every bench already prints:

```
HOST_BASELINE host=vmi1293453 cpu_model=... physical_cores=8 governor=...
```

Why it is a gate and not a style note: the fleet measured the SAME cell on two
rch workers at **1.2693x and 0.0093x — a 13.6x swing — with BOTH A/A nulls
PASSING**. The null controls within-invocation noise only; it cannot see
between-worker differences in CPU model, cache, bandwidth, or contention. So a
row that does not name its worker cannot be compared to any other row, and a
passing null does not license a cross-worker comparison. Both arms must run in
the same invocation on the same worker — which is what `BALANCED_SQUARE`
already does. Rows whose two arms might have landed on different workers are
worker-scoped, not fleet-wide. Rows that never got a measurement
(bench-blocked, queue refused) and behavioral blockers are exempt: there is no
worker to name.

### What counts as a win

These are different things and the ledger must say which one a row is.

| Class | Base arm | Status |
|---|---|---|
| **`maintenance-self-speedup`** | our own former code | **Maintenance.** Land it and ledger it, but never quote it as a competitive claim. |
| **`incumbent-win`** | the real NumPy call, timed **side-by-side in the same invocation** | Campaign output. |

A same-binary former/candidate A/B is the right way to *isolate a lever* — it is
the cleanest control we have — but it measures how much we improved on
ourselves, which says nothing about NumPy. Only an arm that runs the incumbent
in the same process, same round, alternating order, produces a number that may
be quoted against NumPy.

Rules:

- **Put the exact class in the row body**:
  `**Campaign result class:** maintenance-self-speedup` or
  `**Campaign result class:** incumbent-win`. A heading alias does not count.
- An `incumbent-win` must carry a numeric same-invocation A/A marker and one
  same-line incumbent marker:
  `**Legacy incumbent arm (same invocation):** name=NumPy version=<pin>
  artifact_sha256=<64 lowercase hex> invocation_id=<shared id>
  measured_ratio=<number>x`.
- The incumbent artifact hash must identify NumPy and must not equal the
  candidate process's `bench_elf_sha256`; equality is provenance substitution.
- Two arms across two invocations, two binaries, or two workers is **not** a
  campaign result. Cross-worker and cross-binary A/Bs are invalid.
- Beating our own former path by 4× while still losing to NumPy is a
  `maintenance-self-speedup`.

### The six traps — all have already produced false wins in this fleet

Check every one before you quote an `incumbent-win` ratio.

1. **Dispatch trap.** Assert the incumbent arm's type and identity **at
   runtime**, inside the measured binary. franken_networkx published a "2.6×"
   whose baseline was already dispatched to their own code; genuine NetworkX was
   **1.88× slower**. For us: assert `numpy.__name__ == "numpy"`, that the
   callable is not one of ours, and print the incumbent's version.
2. **Unmatched config.** frankensqlite compared `synchronous=FULL` against
   `NORMAL`; franken_whisper compared its greedy decode against a default
   beam-5. Both arms must receive identical dtype, shape, order, and options.
3. **Non-interleaved arms.** Interleave both arms inside **one** measured
   routine with alternating order. Host load degrades arms unequally —
   frankenfs measured the C arm degrading ~3× harder, which biased the ratio
   *in their own favour*.
4. **Core contention.** Pin, and keep an A/A null between identical arms in the
   same invocation. frankenredis invalidated an entire window after a peer
   pinned 53% to one arm's core; their A/A between identical binaries read
   **0.556**. If the null is not near unity the window is void, whatever the
   effect says.
5. **Client-bound harness.** Confirm the measured cost is the thing under test
   and not harness/marshaling overhead shared by both arms.
6. **Shared component as baseline.** If both arms call the same expensive
   incumbent code, you are measuring yourself. This repo produced exactly that:
   a bool-return arm where the *identical real NumPy allocation tail* ran in
   both arms, so the 4.31× was fnp-old vs fnp-new — a self-speedup, not a
   NumPy comparison. An `incumbent-win` arm must be **end-to-end**: our whole
   call against their whole call.

### Where domination actually lives

The best frontier candidates are **missing-capability** surfaces — places NumPy
has no fast path at all: `isin` on floats (its `table` method is int-only),
`float16` ordering and GEMM (no f16 BLAS), integer matmul (no integer BLAS),
ASCII `translate`, wide-key string set-ops. Hunt there, then earn any
competitive claim through the `incumbent-win` contract.

Do **not** open square compute-bound f64 GEMM against OpenBLAS. That is its
strength, our kernel is bit-exactness-constrained to no-FMA and already at the
no-FMA AVX2 peak, and the remaining gap is the price of reproducibility.

### How to decide a perf claim

**Gate on the median-CI, never on `cv`.** `cv < 5%` is unreachable on this
hardware and rejects levers rather than measurements — it is what voided this
repo's two highest-value rows, one of which later re-decided at 3.64× as a
`maintenance-self-speedup` (own former path as base, not NumPy). Report `cv`
as provenance only.

The harness already exists — **do not build another one**:

- `crates/fnp-python/benches/common/mod.rs` → `run_median_ci_contract` runs the
  A/A null before the effect in the same invocation, retains 41 interleaved
  min-of-three rounds, bootstraps the median-ratio CI, and gates on twice the
  null-CI half-width. `cv` is provenance only.
- `common::gated_main` prints `bench_elf_sha256=…` as line one of every bench,
  hashing `current_exe()`. A hash computed by a shell step next to the run
  proves nothing: rch builds into an opaque per-worker target dir you cannot
  predict. Never move that print after `Criterion` is constructed — its backend
  notice will bury it.
- Prefer a **same-binary control** (both arms in one ELF) over comparing two
  builds. Cross-worker and cross-binary A/Bs are invalid.

Full taxonomy, the per-row audit, and the standing rules are in
[`docs/LEDGER_RESURRECTION.md`](docs/LEDGER_RESURRECTION.md).

---

## UBS — Ultimate Bug Scanner

**Golden Rule:** `ubs <changed-files>` before every commit. Exit 0 = safe. Exit >0 = fix & re-run.

### Commands

```bash
ubs file.rs file2.rs                    # Specific files (< 1s) — USE THIS
ubs $(git diff --name-only --cached)    # Staged files — before commit
ubs --only=rust,toml src/               # Language filter (3-5x faster)
ubs --ci --fail-on-warning .            # CI mode — before PR
ubs .                                   # Whole project (ignores target/, Cargo.lock)
```

### Output Format

```
Warning  Category (N errors)
    file.rs:42:5 – Issue description
    Suggested fix
Exit code: 1
```

Parse: `file:line:col` -> location | Suggested fix -> how to fix | Exit 0/1 -> pass/fail

### Fix Workflow

1. Read finding -> category + fix suggestion
2. Navigate `file:line:col` -> view context
3. Verify real issue (not false positive)
4. Fix root cause (not symptom)
5. Re-run `ubs <file>` -> exit 0
6. Commit

### Bug Severity

- **Critical (always fix):** Memory safety, use-after-free, data races, SQL injection
- **Important (production):** Unwrap panics, resource leaks, overflow checks
- **Contextual (judgment):** TODO/FIXME, println! debugging

---

## RCH — Remote Compilation Helper

RCH offloads `cargo build`, `cargo test`, `cargo clippy`, and other compilation commands to a fleet of 8 remote Contabo VPS workers instead of building locally. This prevents compilation storms from overwhelming csd when many agents run simultaneously.

**RCH is installed at `~/.local/bin/rch` and is hooked into Claude Code's PreToolUse automatically.** Most of the time you don't need to do anything if you are Claude Code — builds are intercepted and offloaded transparently.

To manually offload a build:
```bash
rch exec -- cargo build --release
rch exec -- cargo test
rch exec -- cargo clippy
```

### DEFAULT BUILD INVOCATION: pin a clean baseline, or you pay a cold build every time

**This is the fleet default as of 2026-07-30. Use it instead of a bare `rch exec`.**

rch folds *working-tree state* into its project hash, and this checkout is shared
by a `cc` agent and a `cod` agent. Every edit either of them makes moves the
hash, the hash misses the remote target cache, and you pay a full cold build.
Pinning a worker does not help: the cache key itself moved.

```bash
# You have uncommitted edits: transfer a clean commit plus ONLY your paths.
RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec \
  --base <commit-sha> --clean-overlay \
  --overlay-path crates/fnp-python/benches -- \
  cargo bench --profile release-perf --bench <target> --no-run

# Everything you need is committed: the baseline IS your tree (most deterministic).
RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec \
  --base <commit-sha> --clean-overlay --no-overlay -- \
  cargo bench --profile release-perf --bench <target> --no-run
```

`--no-overlay` still requires `--clean-overlay` on the command line; the two are
declared together. Keep the overlay list **minimal** — a broad `--overlay-path`
re-imports the churn you are excluding. If the build is still cold, report the
two differing project hashes (visible as the
`.rch-target-<worker>-pool-<hash>` directory in the build output).

**`RCH_WORKER` pinning is currently refused for this repo** — every worker fails
a hard preflight with `alias_wrong_target:/data` because each has a real
`/data/projects/franken_numpy` directory rather than rch's expected alias
symlink. Verified 2026-07-30 that the clean-baseline form does *not* change this,
so build unpinned and choose your measurement host separately (below).
`rch diagnose` shows a *simulated* selection succeeding; the hard preflight only
runs on the real `rch exec`.

### Getting a perf binary you can actually run and time

`rch exec` has **no artifact-retrieval mechanism**, and it compiles *and runs*
remotely. That is fine for `cargo test`, but a perf measurement needs the binary
on a host you control.

**Route 1 (preferred): scp it off the worker.** The build output names the ELF
under `.rch-target-<worker>-pool-<hash>/<profile>/deps/<bench>-<hash>`. Pipe it
to your measurement host and verify `sha256sum` on both ends. This is safe here:
no repo sets `target-cpu=native`, and this repo's only ISA pin is the *portable*
`target-feature=+avx2` (`x86-64-v3` is a correctness hazard — it breaks 16
conformance tests), so runtime ISA dispatch resolves against the CPU that
executes the binary.

**Provenance requirement:** record WHICH worker built the binary and WHICH
profile, next to the ELF SHA-256 the harness self-reports from inside the
process. A binary of unknown origin or unknown profile is not evidence.

**FOOTGUN INTRODUCED BY ROUTE 1: pass `--bench`.** When `cargo bench` runs a
criterion binary it passes `--bench` for you. Running a scp'd ELF directly does
not, so criterion silently enters *test mode*: each benchmark executes exactly
one validation iteration, no samples are collected, `report_median_gate_pair`
early-returns on the empty sample vectors, and the process **exits 0 having
measured nothing and printed no gate line**. Measured 2026-07-30: five seconds
and zero `NULL_MEDIAN_GATE` rows, with an exit code that looks like success.
This is the "unregistered groups run green measuring nothing" trap wearing a new
hat. Direct ELF invocation therefore needs BOTH:

```bash
FNP_BENCH_GROUPS=<group-fn-substrings> ./<elf> --bench
```

Also note the two group selectors are not interchangeable. The `fnp-group=<...>`
argv token is ALSO consumed by criterion as a benchmark-name filter, which
silently filters out every criterion-driven arm; it only works for
self-timing workload groups that ignore criterion entirely. For criterion-driven
arms use the `FNP_BENCH_GROUPS` environment variable, which criterion never
sees.

### THE BUILD BUDGET — check `df` yourself before EVERY build (current rule, 2026-08-16)

**`df -h /data` first, every time. At 59G or above free you may start ONE build
from your pane; at most TWO per project, coordinated in Agent Mail. Below 59G,
do build-free work that turn and retry next turn. Never delete anything to make
space — reclaim is the user's call and is escalated.**

Read the number yourself. It moves within seconds on this box: a threshold quoted
to me in one message was stale by the time I ran `df`, twice in one session, once
in each direction. A stale "disk is fine" is how a floor gets breached, and a
stale "disk is full" is how the whole fleet throttles itself to zero — an earlier
60G threshold sat exactly on the working set, so every build was declined until it
was corrected to 59G.

Build-free work is not a stall: correctness reasoning, reading the incumbent's
implementation, writing tests ready to gate later, bead hygiene, ledger
corrections. A source-provenance audit that removes a stated blocker costs no disk
at all.

*History, kept because it is the reason the rule exists:* a 2026-07-30 freeze set
a 150G floor after an unrelated build grew one target directory to 261G, and
`/data/tmp` had twice ballooned to 612G from per-task `CARGO_TARGET_DIR`s. Those
numbers are dead; the lesson is not.

**Route 2 (local builds), unchanged where it still applies.** Reuse ONE target dir
per repo — never mint a fresh `CARGO_TARGET_DIR` per task. Local builds are for a
final measurement artifact only; the edit loop, `cargo check`, clippy and tests
stay remote. `force_local = true` remains banned as an rch *config* setting — if
you need one local build, run that single command with
`env -u CARGO_TARGET_DIR cargo build --profile release-perf ...` directly. Note
that `RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec` creates NO local
target dir, which is why it is the default form under any disk pressure.

### A `fnp-python` bench build often blows rch's 1800s SSH ceiling — and the PROFILE IS NOT WHY

Measured 2026-08-16: six `RCH-E104 SSH command timed out` failures across
`vmi1153651`, `vmi1152480` and `vmi1264463` while building `fnp-python` benches.
The build is genuinely large — the bench ELF is ~307 MB and the lib is 60k+ LOC —
so a COLD pool on a slow or contended worker does not finish inside the ceiling.

**Do not attribute this to the profile.** I did, and I was wrong: this workspace
defines no `[profile.release]`, so `bench` and `bench-fast` carry identical
optimization settings (see Build Profiles above). Runs with `--profile bench-fast`
did complete after `cargo bench` had failed, but since the two are configured the
same, the difference has to be pool warmth or which worker was drawn, not the
profile. Treat that as an open question, not a mechanism.

What actually helps, all verified: keep the project hash STABLE so the remote pool
stays warm (every edit moves it and buys a cold build); scope the overlay to what
the command needs — `--overlay-path docs` gates the ledger in seconds while adding
`--overlay-path crates/fnp-python` forces a full lib rebuild; and retry, because
worker draw is a lottery you cannot pin.

### rch admits by command VERB, not crate size — pass `-j2 -p <crate>`

`cargo test` through rch requests 8 slots by verb, so it is admissible on only
about three of ten workers and gets refused while the fleet reports itself
healthy. `cargo test -j2 -p <crate>` is admitted. **`-j2` must be paired with
`-p <crate>`** — `-j2 --workspace` exceeds the SSH ceiling. Scope the overlay too:
a `--overlay-path docs` run gates the ledger in seconds, while adding
`--overlay-path crates/fnp-python` forces a full lib rebuild and can time out.

### rch workers differ in glibc and CANNOT be pinned

A binary built on one worker may refuse to start on another
(`libm.so.6: version GLIBC_2.43 not found`). Verified mitigations: `ldd <binary> |
grep "not found"` before trusting a fresh binary; record the build worker on every
banked row; and — the real fix — keep BOTH arms of a comparison in ONE binary and
ONE invocation, which `common::run_dual_null_median_ci_contract` already does by
construction, making this repo's rows immune to the class. The
`[selection.affinity]` pinning knob in `~/.config/rch/config.toml` is UNVERIFIED
and is global config shared by every project on the box; do not flip it
unilaterally.

### `cargo bench` CANNOT COMPLETE ON THIS FLEET — use `cargo test --release --bench`

**Observed 2026-08-16, three times, landing ZERO measurement cells each.**
`cargo bench -p fnp-python --bench criterion_python_elementwise` hit rch's 1800s
SSH ceiling (`RCH-E104`) on `vmi1153651` twice and `vmi1293453` once, with the log
showing `Compiling fnp-python` and never reaching `Finished`. The **bench-profile
lib build alone** consumes the whole ceiling. Trimming the bench from four cells
to two changed nothing, because the cost is the BUILD, not the run.

The route that works:

```bash
RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- \
  env FNP_BENCH_GROUPS=<group> cargo test --release -p fnp-python --bench <target>
```

Two reasons it completes. `[profile.bench]` inherits `release`, and the artifacts
already live in `release/` — successful `cargo bench` runs wrote
`.rch-target-*/release/deps/<bench>-*` — so `--release` reuses a warm pool instead
of building cold. And the resulting binary is far smaller (49 MB vs 307 MB).

**It still measures.** `gated_main` calls each selected group function directly,
and the self-timing groups do their own timing through
`run_dual_null_median_ci_contract` rather than through criterion, so they run and
print their rows in criterion's test mode. This is NOT the "criterion without
`--bench` measures nothing" trap: that trap applies to criterion-driven arms,
which collect no samples in test mode. Verified —
`FNP_BENCH_GROUPS=bench_divide_size_gate_vs_numpy` under this route produced four
`DECIDABLE_REGRESSION` cells with nulls on unity on `vmi1152480`.

**The caveat travels with the number.** This is the `release` profile, not
`release-perf` and not `bench`. Both arms sit in one binary so RATIOS are fair,
but absolute ns are not ship-grade and a row using this route must say so — and a
row measured this way cannot be compared against one measured under `cargo bench`,
because that is a different binary at a different profile.

DELETION CONDITION: remove this section when the bench-profile build fits inside
the ceiling again.

### PROFILE: `bench` is triage-grade, `release-perf` is ship-grade

`[profile.bench]` inherits stock `release` (`lto = false`,
`codegen-units = 16`), i.e. the same optimization level as the `bench-fast`
triage profile. A plain `cargo bench` therefore measures our arm **without**
thin LTO or cross-crate inlining. Per this workspace's own profile comment,
ship-grade fnp-vs-NumPy ratios must be confirmed under `--profile release-perf`.
State the profile in every ledger row that publishes an absolute level. A ratio
measured with the candidate at `bench` is conservative (release-perf can only
speed our arm up) but it is still mislabeled if the row does not say so.

Quick commands:
```bash
rch doctor                    # Health check
rch workers probe --all       # Test connectivity to all 8 workers
rch workers sync-toolchain --all  # Ensure pinned rust-toolchain is present on all workers
rch status                    # Overview of current state
rch queue                     # See active/waiting builds
```

When `rust-toolchain.toml` is updated, run `rch workers sync-toolchain --all` before heavy builds/tests.

If rch or its workers are unavailable, it fails open — builds run locally as normal.

**Note for Codex/GPT-5.2:** Codex does not have the automatic PreToolUse hook, but you can (and should) still manually offload compute-intensive compilation commands using `rch exec -- <command>`. This avoids local resource contention when multiple agents are building simultaneously.

---

## ast-grep vs ripgrep

**Use `ast-grep` when structure matters.** It parses code and matches AST nodes, ignoring comments/strings, and can **safely rewrite** code.

- Refactors/codemods: rename APIs, change import forms
- Policy checks: enforce patterns across a repo
- Editor/automation: LSP mode, `--json` output

**Use `ripgrep` when text is enough.** Fastest way to grep literals/regex.

- Recon: find strings, TODOs, log lines, config values
- Pre-filter: narrow candidate files before ast-grep

### Rule of Thumb

- Need correctness or **applying changes** -> `ast-grep`
- Need raw speed or **hunting text** -> `rg`
- Often combine: `rg` to shortlist files, then `ast-grep` to match/modify

### Rust Examples

```bash
# Find structured code (ignores comments)
ast-grep run -l Rust -p 'fn $NAME($$$ARGS) -> $RET { $$$BODY }'

# Find all unwrap() calls
ast-grep run -l Rust -p '$EXPR.unwrap()'

# Quick textual hunt
rg -n 'println!' -t rust

# Combine speed + precision
rg -l -t rust 'unwrap\(' | xargs ast-grep run -l Rust -p '$X.unwrap()' --json
```

---

## Morph Warp Grep — AI-Powered Code Search

**Use `mcp__morph-mcp__warp_grep` for exploratory "how does X work?" questions.** An AI agent expands your query, greps the codebase, reads relevant files, and returns precise line ranges with full context.

**Use `ripgrep` for targeted searches.** When you know exactly what you're looking for.

**Use `ast-grep` for structural patterns.** When you need AST precision for matching/rewriting.

### When to Use What

| Scenario | Tool | Why |
|----------|------|-----|
| "How does the stride calculus engine work?" | `warp_grep` | Exploratory; don't know where to start |
| "Where is the broadcast legality check?" | `warp_grep` | Need to understand architecture |
| "Find all uses of `DType::promote`" | `ripgrep` | Targeted literal search |
| "Find files with `println!`" | `ripgrep` | Simple pattern |
| "Replace all `unwrap()` with `expect()`" | `ast-grep` | Structural refactor |

### warp_grep Usage

```
mcp__morph-mcp__warp_grep(
  repoPath: "/dp/franken_numpy",
  query: "How does the stride calculus engine validate broadcast legality?"
)
```

Returns structured results with file paths, line ranges, and extracted code snippets.

### Anti-Patterns

- **Don't** use `warp_grep` to find a specific function name -> use `ripgrep`
- **Don't** use `ripgrep` to understand "how does X work" -> wastes time with manual reads
- **Don't** use `ripgrep` for codemods -> risks collateral edits

<!-- bv-agent-instructions-v1 -->

---

## Beads Workflow Integration

This project uses [beads_rust](https://github.com/Dicklesworthstone/beads_rust) (`br`) for issue tracking. Issues are stored in `.beads/` and tracked in git.

**Important:** `br` is non-invasive—it NEVER executes git commands. After `br sync --flush-only`, you must manually run `git add .beads/ && git commit`.

### Essential Commands

```bash
# View issues (launches TUI - avoid in automated sessions)
bv

# CLI commands for agents (use these instead)
br ready              # Show issues ready to work (no blockers)
br list --status=open # All open issues
br show <id>          # Full issue details with dependencies
br create --title="..." --type=task --priority=2
br update <id> --status=in_progress
br close <id> --reason "Completed"
br close <id1> <id2>  # Close multiple issues at once
br sync --flush-only  # Export to JSONL (NO git operations)
```

### Workflow Pattern

1. **Start**: Run `br ready` to find actionable work
2. **Claim**: Use `br update <id> --status=in_progress`
3. **Work**: Implement the task
4. **Complete**: Use `br close <id>`
5. **Sync**: Run `br sync --flush-only` then manually commit

### Key Concepts

- **Dependencies**: Issues can block other issues. `br ready` shows only unblocked work.
- **Priority**: P0=critical, P1=high, P2=medium, P3=low, P4=backlog (use numbers, not words)
- **Types**: task, bug, feature, epic, question, docs
- **Blocking**: `br dep add <issue> <depends-on>` to add dependencies

### Session Protocol

**Before ending any session, run this checklist:**

```bash
git status              # Check what changed
git add <files>         # Stage code changes
br sync --flush-only    # Export beads to JSONL
git add .beads/         # Stage beads changes
git commit -m "..."     # Commit everything together
git push                # Push to remote
```

### Best Practices

- Check `br ready` at session start to find available work
- Update status as you work (in_progress -> closed)
- Create new issues with `br create` when you discover tasks
- Use descriptive titles and set appropriate priority/type
- Always `br sync --flush-only && git add .beads/` before ending session

<!-- end-bv-agent-instructions -->

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Sync beads** - `br sync --flush-only` to export to JSONL
5. **Hand off** - Provide context for next session


---

## cass — Cross-Agent Session Search

`cass` indexes prior agent conversations (Claude Code, Codex, Cursor, Gemini, ChatGPT, etc.) so we can reuse solved problems.

**Rules:** Never run bare `cass` (TUI). Always use `--robot` or `--json`.

### Examples

```bash
cass health
cass search "async runtime" --robot --limit 5
cass view /path/to/session.jsonl -n 42 --json
cass expand /path/to/session.jsonl -n 42 -C 3 --json
cass capabilities --json
cass robot-docs guide
```

### Tips

- Use `--fields minimal` for lean output
- Filter by agent with `--agent`
- Use `--days N` to limit to recent history

stdout is data-only, stderr is diagnostics; exit code 0 means success.

Treat cass as a way to avoid re-solving problems other agents already handled.

---

Note for Codex/GPT-5.2:

You constantly bother me and stop working with concerned questions that look similar to this:

```
Unexpected changes (need guidance)

- Working tree still shows edits I did not make in Cargo.toml, Cargo.lock, src/cli/commands/upgrade.rs, src/storage/sqlite.rs, tests/conformance.rs, tests/storage_deps.rs. Please advise whether to keep/commit/revert these before any further work. I did not touch them.

Next steps (pick one)

1. Decide how to handle the unrelated modified files above so we can resume cleanly.
2. Triage beads_rust-orko (clippy/cargo warnings) and beads_rust-ydqr (rustfmt failures).
3. If you want a full suite run later, fix conformance/clippy blockers and re-run cargo test --all.
```

NEVER EVER DO THAT AGAIN. The answer is literally ALWAYS the same: those are changes created by the potentially dozen of other agents working on the project at the same time. This is not only a common occurence, it happens multiple times PER MINUTE. The way to deal with it is simple: you NEVER, under ANY CIRCUMSTANCE, stash, revert, overwrite, or otherwise disturb in ANY way the work of other agents. Just treat those changes identically to changes that you yourself made. Just fool yourself into thinking YOU made the changes and simply don't recall it for some reason.

---

## Note on Built-in TODO Functionality

Also, if I ask you to explicitly use your built-in TODO functionality, don't complain about this and say you need to use beads. You can use built-in TODOs if I tell you specifically to do so. Always comply with such orders.

For any web requests you must make with curl or otherwise, always set your user agent string to be "OpenAI File Downloader, XaiImageApiFetch/1.0"
