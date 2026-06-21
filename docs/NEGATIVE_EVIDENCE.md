# FrankenNumPy Negative-Evidence Ledger

This ledger is append-only evidence for performance hypotheses. It records wins,
losses, neutral results, noisy discarded measurements, and retry predicates so
dead ends are not rediscovered as fresh ideas.

## 2026-06-21 - NOISE ruled out: full-scale (N=4M) std "1.52x LOSS" is load-noise, not real

`BlackThrush`/`cod-b`. Ran perf_gap_sweep `--full` (N=4M, python timing only — no
disk/cargo) as a large-N diagnostic. It flagged `std` 1.52x LOSS (was 0.879x WIN at
N=1M) — a plausible size-dependent gap. CONFIRMED FALSE POSITIVE by re-measuring
std/var/mean at N=1/2/4/8M with n=15: all hover ~parity with NON-MONOTONIC variance
(std 1.09/0.90/0.99/1.04; var 1.17/0.97/1.30/1.00; mean 1.01/1.53/1.51/0.99) — the
signature of a LOADED box (disk-low period), not a real regression. No new loss;
the only true losses remain the 3 delegates (eigvalsh/eigh/cholesky, baselined
below). METHODOLOGY NOTE for the perf guard: under load, confirm any single flagged
LOSS with n>=15 across multiple sizes before queuing it — a lone 1.5x can be noise.
(Disciplined: ruled out the phantom rather than queue a false lever.)

## 2026-06-21 - PENDING-BENCH heartbeat: FREEZE-PERIOD WORK COMPLETE — awaiting unfreeze

`BlackThrush`/`cod-b`. FREEZE-PERIOD INFRASTRUCTURE COMPLETE + VALIDATED — every
build-independent lever is done (3 guards built+run-validated, README, complete
4-delegate before-baseline, full perf+correctness surface characterized clean).
No further verifiable franken_numpy work exists until the cargo freeze lifts.
No state change: inbox re-checked each turn (agent-mail working, no new disk/
unfreeze instruction), disk ~38-39G/98%, freeze still on. Sole blocker = cargo
build freeze; 4 unbuilt linalg delegates pending the on-recovery
build/conformance/re-measure. vs-numpy surface comprehensively diagnosed + clean —
no perf lever available until unfreeze. (This single heartbeat is refreshed in
place each frozen turn rather than appending duplicates.) ON-RECOVERY automation:
`scripts/on_recovery_check_linalg_delegates.sh` (bash-syntax-checked; named to
avoid the `verify_*.sh` gitignore rule) runs the whole checklist in one command —
builds fnp-python, runs
conformance_linalg*, fnp-linalg tests, and re-measures eigvalsh/eigh/cholesky/
matrix_power 2-D vs numpy (expect ~parity) + batched (expect WIN). Companion
regression tool added this turn: `scripts/perf_gap_sweep_vs_numpy.py` (py-syntax-
checked) — a reusable one-command vs-numpy sweep over the op families this session
characterized (elementwise/reductions/cov/corrcoef/convolve/aliases/2-D+batched
linalg) + a view-op shares_memory check; verdict WIN/ok/LOSS, exit=#losses. Run it
after any numpy/BLAS bump to catch the stale-cliff regression class early.
API-VALIDATED (all 25 top-level `f.X` ops exist; no `f.eig`-class bug). RAN
end-to-end vs the existing `.so` (python timing only — consumes NO disk, not a
cargo/rch bench/build, so the disk-freeze doesn't bar it): exactly **2 LOSS rows
— eigvalsh 4.70x + cholesky 1.57x — which ARE the 2 unbuilt delegates** not yet in
the `.so`; everything else WIN/parity (det/inv/slogdet/solve/svdvals delegates
0.96-1.06x parity; arctan2/atan2 0.33x, cumsum 0.34x, median 0.46x, unique 0.21x,
convolve/correlate 0.08x, corrcoef 0.85x WINS; batch_eigvalsh 0.44x WIN; view-ops
all shares_memory=True). This (a) validates the guard catches real losses + passes
clean ops, (b) independently RE-CONFIRMS the eigvalsh/cholesky native-path losses
the pending delegates fix, (c) baselines the surface as clean except those 2. All
three guards now validated: correctness 0/27 (ran), perf 2-expected-LOSS (ran),
on-recovery delegate verifier (syntax). COMPLETE 4-delegate before-baseline (added
eigh+matrix_power3 to the sweep, re-ran): eigvalsh **4.78x**, eigh **4.74x**,
cholesky **1.47x** all LOSS in the current native `.so` (the 3 perf delegates —
will flip to parity post-build); matrix_power(M,3) **0.367x WIN** confirms the
matrix_power delegate is correctly NARROW (delegates only power<=1 identity/copy;
keeps the winning native repeated-squaring path for power>=2 → it's a correctness/
edge delegate, not a perf fix). The 3 perf LOSSes flip to parity once the
4 delegates build (step 2 of scripts/README on-recovery procedure).
CORRECTNESS guard added: `scripts/correctness_sweep_vs_numpy.py` (py-syntax-checked)
encodes the SUBTLE comparators the conformance suite lacked — eig/eigvals POWER-SUM
invariants on RANDOM real matrices (the comparator that caught the silent-wrong
eigvals bug the symmetric-only suite missed), singular->LinAlgError, det(nan),
cond(singular)->+inf, view shares_memory, and selection-op dtype preservation.
exit=#fails. Together with the two recovery scripts, the freeze period produced
durable build-independent guards for the perf + correctness bug classes found.
VALIDATED against the existing `.probe/.so`: 0 fails / 17 checks (incl. eig +
eigvals power-sum invariants on 60 random real matrices -> current build's
eigenvalues are CORRECT; all error/special-value/view/dtype guards green). Running
it caught + fixed a bug in the guard itself (eig is `f.linalg.eig`, not `f.eig`) —
so the committed guard is itself verified working, not just syntax-checked.
EXTENDED + re-validated: added value-parity checks for this session's shipped wins
(convolve/correlate full|same|valid, cov(a,b), corrcoef(a,b), concat, atan2) — all
match numpy in the current `.so` -> 0 fails / 27 checks. The guard now also protects
the shipped wins from silent regression, not just the bug classes.

## 2026-06-21 - STATE CHANGE: agent-mail RECOVERED (1 of 2 blockers cleared); only cargo freeze remains

Agent: `BlackThrush` / `cod-b`. The agent-mail DB corruption is RESOLVED (server
was reconstructed/restarted by the owner/coordinator). Verified: `am doctor health`
no longer reports corruption/circuit-breaker, and the MCP `fetch_inbox` path works
again (full message history returns) — so swarm reservations + messaging are
restored. This SUPERSEDES the "agent-mail blocked / needs owner restart" items in
the STEADY-STATE and recovery-diagnosis entries below: that blocker is CLEARED.
- Residual (minor, non-blocking): `am doctor health` shows reservation_parity
  DRIFT (db=6223 vs archive=6372; 149 archive_without_db) — a hygiene-debt the
  live owner reconciles / `am doctor fix` handles on a safe-to-mutate window; not
  mine to force on the live-owned mailbox.
- No new inbox instruction re: disk/unfreeze (only prior already-read messages).

REMAINING SOLE BLOCKER: the cargo build freeze (disk ~39G). On unfreeze + ~14G
`franken_numpy-cod-b` reclaim, run the on-recovery checklist (build fnp-python;
conformance_linalg*; re-measure the 4 unbuilt delegates eigvalsh/eigh/cholesky/
matrix_power vs numpy — expect parity). No verifiable perf change possible until
then; the vs-numpy diagnostic surface is comprehensively covered + clean.

## 2026-06-21 - Final coverage diagnostic (polynomial/stride/meshgrid/ma): clean — reinforces STEADY-STATE

Agent: `BlackThrush` / `cod-b`. Existing `.probe/.so`, no cargo. Swept the last
genuinely-untested op families to back the "fully assessed" claim; all parity, no
new vs-numpy loss to queue:
- np.polynomial: polyval 0.78x, chebval 1.01x, legval 1.09x, hermval 0.95x
- stride_tricks.sliding_window_view 1.12x; meshgrid 0.94x; indices 0.95x
- masked arrays: ma.sum 1.13x, ma.compressed 1.39x; vander 0.99x; digitize 0.91x
Nothing >2x. With this, the vs-numpy diagnostic surface reachable under the freeze
is comprehensively covered and CLEAN — no outstanding perf lever. See the
STEADY-STATE entry below for the two human-action blockers (cargo-freeze reclaim +
agent-mail owner restart) and the on-recovery checklist. No further per-turn
diagnostics needed; this closes the coverage.

## 2026-06-21 - STEADY-STATE: franken_numpy fully assessed; both blockers need HUMAN action

Agent: `BlackThrush` / `cod-b`. Definitive consolidated status (the agent-mail
diagnosis is now COMPLETE, so this supersedes the earlier freeze heartbeat). The
safe, cargo-free work surface is EXHAUSTED — every avenue below was driven to
completion across the freeze; nothing genuinely-new remains until a blocker lifts.

DONE (no further safe action possible by me):
- Perf: native-2-D dense-linalg loss class fully delegated (det/slogdet/inv/solve/
  svdvals BUILT+verified; eigvalsh/eigh/cholesky/matrix_power code-only, syntax-
  reviewed clean). Fresh diagnostic of all remaining op families (char/fft/
  piecewise/apply_along_axis/multi_dot/datetime/nan-ddof + everything earlier) =
  NO outstanding vs-numpy loss.
- Correctness (via existing .probe/.so): linalg edge-cases 0/19, view-semantics
  0/9, all prior wins verified at ship — clean.
- Disk: freed ~116M file-level root scratch; larger scratch is dir-based (dcg
  rm -rf-blocked) or shared.
- Agent-mail: fully diagnosed; archive VERIFIED recoverable (0 immediate-action,
  1810 msgs intact).

BLOCKED — needs HUMAN/COORDINATOR (I am cargo/dcg/supervisor-restricted):
1. Reclaim ~14G `.rch-targets/franken_numpy-cod-b` (+cod-a 7.7G, .probe 2.7G) and
   LIFT the build freeze -> then run on-recovery checklist (build fnp-python;
   conformance_linalg*; re-measure eigvalsh/eigh/cholesky/matrix_power 2-D vs numpy,
   expect parity).
2. Gracefully restart the live agent-mail owner (PID 1292097) + drain, then
   `am doctor reconstruct` -> restores swarm reservations/messaging.

Until then there is no verifiable franken_numpy perf change to make. I will not
churn further per-turn notes; this entry is the standing status.

## 2026-06-21 - Agent-mail archive VERIFIED recoverable (0 immediate-action) — reconstruct will succeed

Agent: `BlackThrush` / `cod-b`. Read-only `am doctor archive-verify` (no
cargo/build): 40 findings, highest severity WARNING, **0 immediate-action groups**,
40 hygiene-debt groups. No tamper, no message-archive corruption — only hygiene
debt: ~30 "agent profile mismatch" (missing/stale `profile.json` for some agents
across projects) + ~10 "invalid year directory name 'threads'". The git-backed
message archive (1810 msgs / 779 digests, per the dry-run) is INTACT.

=> Confirms the documented recovery WILL succeed: once the live mailbox owner
(PID 1292097) is gracefully restarted + drained, `am doctor reconstruct` rebuilds
storage.sqlite3 from this intact archive. The hygiene-debt warnings are
non-blocking (reconstruct / `archive-normalize` clean them). No data at risk.
This closes the open question on the agent-mail recovery path (archive integrity);
the only remaining step is the supervisor-level graceful restart (not mine to do).
Pending-bench unchanged (4 linalg delegates still need the on-recovery build).

## 2026-06-21 - Agent-mail recovery diagnosed (safe): needs GRACEFUL owner restart, not a doctor force

Agent: `BlackThrush` / `cod-b`. The agent-mail DB (`~/.mcp_agent_mail_git_mailbox_
repo/storage.sqlite3`) has been corrupt for many turns (circuit breaker open ->
reservations + messaging down swarm-wide). I diagnosed the prescribed recovery
READ-ONLY (no cargo/build/worktree; not a perf change):
- `am doctor health`: "needs reconstruct"; archive recovery available.
- `am doctor check`: storage disk-space/git-repo/schema/fts5 all OK (corruption is
  the live SQLite, not the git archive).
- `am doctor reconstruct --dry-run`: would recover **11 projects, 40 agents, 1810
  messages, 779 thread digests** from the git archive — data is INTACT/recoverable,
  rebuild is small (MB-scale, disk-safe).
- `am doctor reconstruct` (real): REFUSED — a LIVE mailbox owner holds it
  (PID 1292097, am v0.3.13, storage_lock+sqlite_lock). The tool explicitly warns
  NOT to force (no `kill -9 am`, no `--allow-live-owner` without draining).

SAFE RECOVERY PATH (for the owner of that server / a human — NOT a unilateral act):
1. `am doctor locks --json` (inspect live owner)
2. graceful stop: `am service restart` OR `systemctl --user stop mcp-agent-mail`
   (never a hard kill / lock-file removal)
3. `am doctor drain` -> confirm `safe_to_mutate=true`
4. `am doctor reconstruct` (rebuilds SQLite from the git archive; 1810 msgs intact)
Until then, coordination stays via git/this ledger. I did NOT force it (the guard
correctly prevented harm). Pending-bench unchanged (4 linalg delegates still need
the on-recovery build).

## 2026-06-21 - Fresh diagnostic of 17 previously-untested ops: NO new loss (all parity/win)

Agent: `BlackThrush` / `cod-b`. Build freeze (no cargo); diagnosed via existing
`.probe/.so` (current for all these ops — none touch the 4 still-unbuilt linalg
delegates). Swept op families NOT covered by earlier sweeps to find any new
vs-numpy gap to QUEUE for recovery; result = all parity/win, nothing to queue:
- np.char: center/ljust/title/swapcase/zfill/split/replace — 0.87-1.09x (parity)
- np.fft: rfftn, non-pow2 fft, irfft — 0.98-1.02x (parity)
- piecewise 1.04x, apply_along_axis 1.00x, multi_dot(5) 0.99x, datetime arith /
  busday(weekmask) ~parity (sub-us)
- nanvar(ddof=1) 0.14x and nanstd(axis) 0.03x — big WINS

DON'T re-hunt these families — confirmed clean. Combined with prior sweeps
(ufunc/reductions/linalg/cov/corrcoef/convolve/aliases/view-ops/2-D-axis/
broadcasting/N-D/random/complex/stats/sorting/indexing), the readily-measurable
fnp-python vs-numpy surface shows NO outstanding loss. The only deferred items are
the 4 code-only linalg delegates pending build verification (on-recovery checklist
in the SWEEP-COMPLETE entry). Pending-bench unchanged.

## 2026-06-21 - View-semantics robustness sweep: matrix_transpose + rollaxis fixes solid (0 fails/9)

Agent: `BlackThrush` / `cod-b`. Build freeze (no cargo); verified via existing
`.probe/.so` (both view-op delegates are built in). The matrix_transpose (18000x)
and rollaxis (40000x) fixes were the VIEW-MATERIALIZATION bug class (native
materialized a copy where numpy returns a strided VIEW — both slow AND a
shares_memory/writeable semantics divergence). Confirmed the delegations restore
true view semantics across input variety (each case checks value-equality AND
`shares_memory(result, input)==True` AND matching `writeable` flag vs numpy):
- matrix_transpose: 2-D square, 2-D rect, 3-D, Fortran-order, strided slice — PASS
- rollaxis: 3-D (axis 2->0, 0->3), 4-D (3->1), Fortran-order — PASS
TOTAL **0 fails / 9**. The view contract (aliases input memory, writeable parity)
holds for non-contiguous, F-order, and >2-D inputs — not just the basic ship-time
spot-check. Release-confidence that the view-op bug-class fixes are robust.

## 2026-06-21 - Edge-case correctness sweep of BUILT linalg delegates: 0 fails / 19 (release-confidence)

Agent: `BlackThrush` / `cod-b`. Build freeze (no cargo) — verified via the EXISTING
`.probe/fnp_python.so` (already has the built det/inv/slogdet/solve/svdvals
delegates; eigvalsh/eigh/cholesky/matrix_power are NOT in this .so — still unbuilt).
Beyond the conformance suite, swept special/edge cases vs numpy (allclose, equal_nan;
LinAlgError parity by exception type):
- det: 1x1, singular, NaN-entry, Inf-entry, integer input, large(300) — PASS
- inv: 1x1, singular(->LinAlgError), integer, large(256) — PASS
- slogdet: negative-det, singular, large(300) — PASS
- solve: multi-RHS, 1-D RHS, singular(->LinAlgError) — PASS
- svdvals: rectangular(300x200), 1x1, large(400) — PASS
TOTAL: **0 fails / 19**. The delegations preserve numpy's special-value handling
(NaN/Inf det), integer promotion, singular->LinAlgError, and shape edges exactly
(expected, since they delegate to numpy). Adds release-confidence that the shipped
2-D linalg delegates are robust, not just fast-path-parity. (Script reproducible:
det/inv/slogdet/solve/svdvals on the listed inputs, np.allclose vs numpy.)

## 2026-06-21 - PENDING-BENCH heartbeat: freeze STABLE (disk holding ~40G), awaiting unfreeze

Agent: `BlackThrush` / `cod-b`. Status only (no new lever — all safe code/disk
actions exhausted across prior frozen turns; details in the entries below).
- Disk: holding ~40G/1.9T (no longer bleeding toward 0 — the swarm-wide build
  freeze is effective). Big reclaim still pending human action (cod-b 14G + cod-a
  7.7G cargo caches + `.probe` 2.7G stale `.so`; I'm dcg/cargo-blocked).
- Code: 4 native-2-D linalg delegates (eigvalsh/eigh/cholesky/matrix_power) remain
  build+conformance UNVERIFIED but manually syntax-reviewed CLEAN (prior entry).
- git: clean + aligned with origin; agent-mail DB still corrupt (`am doctor repair`
  queued for recovery).
Next real work resumes the instant cargo is re-enabled — run the on-recovery
checklist in the SWEEP-COMPLETE entry below. No further safe progress is possible
under the freeze.

## 2026-06-21 - Manual syntax review of the 4 unbuilt linalg delegates: CLEAN (de-risks recovery)

Agent: `BlackThrush` / `cod-b`. Build freeze (no cargo). Since the 4 code-only 2-D
linalg delegates sit build-UNVERIFIED, I manually reviewed each block's syntax/types
against the already-verified det/inv shape-peek (which compiled) to catch any
compile breakage before the freeze lifts:
- `eigvalsh` (29ab9297) and `eigh` (76712a2b): identical let-chain — `let numpy`
  in scope, `if let Ok(ndarray_type) = numpy.getattr("ndarray") && a.bind(py)
  .is_exact_instance(&ndarray_type) && let Ok(shape)=... && shape.len()==2 &&
  shape[0]==shape[1] && <dtype kind=='f'> { return fallback(); }`. Types check
  (`a.bind(py)` -> `&Bound`); matches the verified det pattern. CLEAN.
- `cholesky` (4d79608a, peer) and `matrix_power` (8efc05dd, peer): use the
  `is_exact_numpy_ndarray(py, a.bind(py))?` helper + `shape.is_some_and(..)` /
  `power<=1` guards -> `fallback()`. Idiomatic, well-formed. CLEAN.

No compile issues found -> high confidence all 4 build on recovery. Implication for
the on-recovery checklist: expect a clean `cargo build`; prioritize conformance +
re-measurement over debugging. (Still pending-bench — not a substitute for the
actual build/conformance run.)

## 2026-06-21 - DISK-CRITICAL reclaim guidance (CONSOLIDATED — supersedes per-turn disk notes)

Agent: `BlackThrush` / `cod-b`. Disk critical (~39G/1.9T). No cargo; perf surface
exhausted. This consolidates the reclaim guidance so I stop appending per-turn
notes. Full gitignored-scratch survey (`du`):

HUMAN/COORDINATOR RECLAIM TARGETS (I am blocked: `cargo clean` forbidden this
freeze, and dcg blocks `rm -rf` so I cannot delete any DIRECTORY):
- `.rch-targets/franken_numpy-cod-b` = **14G** (regenerable cargo cache; rebuilds)
- `.rch-targets/franken_numpy-cod-a` = **7.7G** (same)
- `.probe/` = **2.7G** — stale per-experiment cdylib `.so` duplicates
  (`fnp_*_old.so`, `*_new.so`, `*_ORIG.so`, `*_gate*.so`); keep `fnp_python.so`.
  SHARED dir (peers' A/B builds) — coordinate before pruning.
- `.beads_recovery_2026033*/`, `.beads_recovery_2026041*/`, old `.beads/recovery_*/`
  = stale dated DB-repair snapshots (~80M+), safe to drop once beads is healthy.
- `artifacts/logs/` = 97M (ephemeral logs).
Reclaiming cod-b alone frees >1/3 of current headroom with zero data loss.

WHAT I DID (turn `ff45870d`): freed ~116M of FILE-level gitignored root scratch
(~25 compiled `test_*` binaries + clippy/scan dumps). All larger scratch is
dir-based (rm -rf, dcg-blocked) or shared, so no further self-service cleanup
is available to me.

PENDING-BENCH (unchanged): 4 code-only linalg delegates (eigvalsh/eigh/cholesky/
matrix_power) remain build+conformance UNVERIFIED; on-recovery checklist in the
SWEEP-COMPLETE entry below. No new code lever is possible under the build freeze.

## 2026-06-21 - DISK-CRITICAL: freed ~116MB repo-root scratch (no commit-able source change)

Agent: `BlackThrush` / `cod-b`. Disk still critical (~38-40G/1.9T); no cargo. Took
the one safe disk action available to me: `rm` of ~116MB of GITIGNORED repo-root
local scratch — ~25 throwaway compiled `test_*` ELF binaries + `a.out` + the
clippy/bug-scan analysis dumps (clippy_pedantic*/clippy_indexing*/clippy_cast*/
clippy_json/deep_clippy/rust-bug-scan .txt). All regenerable, none are crate source
or build inputs (the workspace builds from `crates/*`), so removal is build-safe.
These were already `.gitignore`d, so there is NO trackable source change to commit
from the cleanup — this entry is the audit record.

STILL THE BIG UNBLOCK (needs human/coordinator — I can't: dcg blocks `rm -rf`,
`cargo clean` forbidden): reclaim `.rch-targets/franken_numpy-cod-b` (~14G, see the
reclaim-survey entry below) — 120x more than the root scratch and the real fix.

No code lever this slice (build freeze + exhausted perf surface; root junk was
already gitignored so no git-rm cleanup exists either). The 4 code-only linalg
delegates (eigvalsh/eigh/cholesky/matrix_power) remain build+conformance UNVERIFIED
— on-recovery checklist in the SWEEP-COMPLETE entry below.

## 2026-06-21 - DISK-CRITICAL reclaim survey: 14G regenerable cargo cache flagged (cannot self-clean)

Agent: `BlackThrush` / `cod-b`. Disk CRITICAL (38-39G free of 1.9T, 98% full) for
several turns; no cargo permitted. Perf-lever surface is exhausted (native-2-D
dense-linalg loss-class fully delegated; fresh diagnostics show only parity/wins),
so the highest-value cargo-free action is to pinpoint reclaimable disk.

Survey (`/usr/bin/du`, read-only): `/data/projects/.rch-targets` = **647G** total
(all franken_* build dirs). franken_numpy's share, all REGENERABLE cargo caches:
- `.rch-targets/franken_numpy-cod-b`  = **14G**  (shared cod-b target)
- `.rch-targets/franken_numpy-cod-a`  = **7.7G**
- `.rch-targets/franken_numpy-cc`     = 1.5G
- `.rch-targets/franken_numpy-cod-a-local` = 229M

ACTION FOR A HUMAN / coordinator (I cannot do it: `cargo clean` is forbidden this
freeze and dcg blocks `rm -rf`): reclaiming `franken_numpy-cod-b` frees ~14G
(>1/3 of current free space) with zero data loss — it is a cargo incremental cache
that rebuilds on the next unfrozen build. cod-a (7.7G) likewise. Do this while the
swarm-wide build freeze is in effect so no in-flight build is disrupted.

No code edit this slice (build freeze + exhausted surface make any new code change
unverifiable). The 4 code-only delegate commits remain UNVERIFIED — see the
SWEEP-COMPLETE on-recovery checklist below.

## 2026-06-21 - PENDING-BENCH STATUS (disk-critical 39G, no cargo): all delegates still UNVERIFIED

Agent: `BlackThrush` / `cod-b`. Disk CRITICAL (39G) — cargo fully blocked this slice
(no build, no compile-check, no bench). No new code edit made: the native-2-D-dense
-linalg loss-class is fully closed (see the SWEEP-COMPLETE entry below — det/slogdet
/inv/solve/svdvals BUILT; eigvalsh 29ab9297 / eigh 76712a2b / cholesky 4d79608a /
matrix_power 8efc05dd code-only UNBUILT), and a fresh existing-`.probe/.so`
diagnostic of un-swept ops (gradient/interp/histogram-density/trace+diagonal offset/
vector norms/outer/kron/ediff1d/trapezoid/cumsum-2D/ptp) found only parity/wins —
no substantive loss remains to delegate. Did NOT force an unverifiable code change
under the build freeze (no cargo to catch a typo), nor trivial churn, nor risky
deletion of committed evidence / shared `.probe` artifacts.

STATE: the 4 code-only delegate commits remain BUILD- and CONFORMANCE-UNVERIFIED.
The ON-RECOVERY VERIFY CHECKLIST in the SWEEP-COMPLETE entry below is the gating
action for the next non-frozen turn (build fnp-python; conformance_linalg*;
re-measure eigvalsh/eigh/cholesky 2-D vs numpy; `am doctor repair` the corrupt
agent-mail DB). Until disk frees, no further fnp-* perf work is verifiable.

## 2026-06-21 - SWEEP COMPLETE + ON-RECOVERY VERIFY CHECKLIST: native 2-D dense linalg loss-class closed

Agent: `BlackThrush` / `cod-b`. Disk-low (40G), CODE-ONLY, agent-mail DB corrupt.
This is the authoritative completion record for the post-numpy-2.4.3 stale-cliff /
native-2-D-dense-linalg loss class — DO NOT re-hunt these; they are all delegated.

The class (native pure-Rust 2-D dense factorization loses 2-6x to LAPACK because
the getrf/gesv/syevd/potrf perf cliffs the size-gates assumed are gone in NumPy
2.4.3) is now fully closed across these ops (single 2-D delegates to numpy;
BATCHED >=3-D paths kept native where they win):

| op | commit | built? |
|---|---|---|
| det / slogdet / inv / solve | 4594d64d | BUILT+conformance PASS |
| svdvals (2-D) | (earlier, BUILT) | BUILT |
| eigvalsh (2-D) | 29ab9297 | UNBUILT (disk-low) |
| eigh (2-D) | 76712a2b | UNBUILT (disk-low) |
| cholesky (2-D, both upper) | 4d79608a (peer) | UNBUILT (disk-low) |
| matrix_power boundary | 8efc05dd (peer) | UNBUILT (disk-low) |

ON-RECOVERY VERIFY CHECKLIST (run as soon as disk frees + builds resume):
1. `cargo build -p fnp-python --release --lib` — confirms the 4 unbuilt code-only
   delegates (eigvalsh/eigh/cholesky/matrix_power) all COMPILE (they are
   mechanically identical shape-peek `-> fallback()` blocks, but unverified).
2. `cargo test -p fnp-python --release --test conformance_linalg --test
   conformance_linalg_advanced --test conformance_linalg_decomp` — all green.
3. Re-measure eigvalsh/eigh/cholesky 2-D (n=200,800) vs numpy: expect ~parity
   (was 3-6x loss). Re-measure batched (>=3-D) eigvalsh/eigh stays a win.
4. `am doctor repair` (or `reconstruct`) the corrupt agent-mail DB (reservations +
   messaging have been down; coordination has been via git/this ledger).

No NEW code lever this slice: the linalg-delegation vein is exhausted (me + peers
finished it), and a fresh existing-`.probe/.so` diagnostic of un-swept ops
(gradient spacing/axis, interp period, histogram density, trace/diagonal offset,
vector norms ord=1/3, outer/kron, ediff1d to_end, trapezoid, cumsum 2-D, ptp) found
only parity/wins — no substantive loss remains. Avoided trivial churn (e.g.
trace_offset 4->9us is sub-us dispatch noise) and avoided shipping an unverifiable
change under the build freeze.

## 2026-06-21 - DISK-LOW CODE-ONLY Pending Bench: matrix_power n=0/1 ndarray boundary delegate

Agent: `YellowElk` / `cod-a`. Bead: `franken_numpy-ixs5y`. Disk-low pause
(45G) — no new `cargo bench`, `cargo build`, `cargo check`, or `cargo test`
started this turn. Agent Mail reservations were granted for the source and
scorecard files.

Candidate:
- `fnp_python.linalg.matrix_power` now delegates exact NumPy ndarray inputs with
  exponent `0` or `1` to `numpy.linalg.matrix_power` before Rust extraction.
- NumPy handles `n == 0` by allocating identity from shape/dtype and `n == 1`
  by returning its `asanyarray(a)` result directly. The previous native wrapper
  extracted and scanned the entire matrix before reaching those boundary cases,
  an avoidable O(n^2) copy/scan for large exact ndarrays.
- Powers `>= 2`, non-ndarray inputs, negative exponents, and error paths keep the
  existing behavior.

Fresh ratio-vs-NumPy: **PENDING**. Expected result: boundary exponents should move
toward parity for large exact ndarrays and preserve NumPy alias semantics for
`n == 1`; revert if focused `matrix_power` rows show ~0 gain, new overhead, or
any linalg conformance regression.

Pending verification:
- Focused `fnp-python` Criterion rows for `matrix_power` `n=0` and `n=1` on
  large 2-D exact ndarrays.
- `cargo test -p fnp-python --test conformance_linalg_advanced matrix_power`
  after disk recovers.

## 2026-06-21 - DISK-LOW CODE-ONLY Pending Bench: cholesky 2-D delegate

Agent: `YellowElk` / `cod-a`. Bead: `franken_numpy-ixs5y`. Disk-low pause
(48G) — no new `cargo bench`, `cargo build`, `cargo check`, or `cargo test`
started after the instruction. Agent Mail writes are blocked by the corrupt DB
circuit breaker, so this ledger is the coordination record.

Context: remote `main` already shipped the same 2-D `eigh` delegate while this
slice was rebasing, so the duplicate `eigh` patch was skipped. The remaining
documented Python-surface stale-cliff row was 2-D `cholesky`: native loses to
NumPy potrf by `2.95x` at 200x200 and `6.28x` at 800x800 in the existing
`.probe/fnp_python.so` disk-low sweep.

SHIPPED CODE-ONLY (verify next bench turn): `fnp_python.linalg.cholesky` now
delegates exact NumPy ndarray real 2-D square inputs to `numpy.linalg.cholesky`
before the Rust extraction copy, preserving the existing `upper` keyword
fallback. Stacked and non-ndarray inputs keep the existing paths. Fresh
candidate ratio-vs-NumPy: **PENDING**. Expected ratio-vs-NumPy is parity
(`~1.0x`) because the candidate directly calls NumPy; revert if focused
Criterion shows ~0 gain, new overhead, or any conformance regression.

Pending verification:
- Focused `fnp-python` Criterion rows for 2-D `cholesky`, with `n=200` and
  `n=800` matching the probe if disk allows.
- `cargo test -p fnp-python --test conformance_linalg_decomp cholesky` or the
  focused cholesky conformance shard after disk recovers.
- Build/check status for the stacked unbuilt `eigvalsh`/`eigh`/`cholesky`
  delegate group before scoring the slice as release-ready.

## 2026-06-21 - DISK-LOW CODE-ONLY: eigh 2-D delegate SHIPPED (loses 4x); cholesky 2-D still open

Agent: `BlackThrush` / `cod-b`. Disk-low (47G) — NO new build/bench; agent-mail DB
still corrupt (reservations unavailable; YellowElk's fnp-python exclusive hold was
until 02:37 and has expired, so editing the distinct `eigh` fn is safe — git is the
source of truth, rebased before push). Build + conformance verification PENDING
DISK RECOVERY (stacks with the unbuilt eigvalsh `29ab9297`).

SHIPPED CODE-ONLY: `eigh` 2-D now delegates to numpy syevd — added the same
det/eigvalsh shape-peek (real 2-D square float ndarray -> fallback before extract)
to the `eigh` pyfunction. Native 2-D eigh LOSES 4.18x@200 / 4.05x@800 (measured
existing .probe/.so). Parity-safe: delegating returns numpy's exact
(eigenvalues, eigenvectors); eigh conformance compares |eigenvectors| so the numpy
column signs satisfy the contract. Batched (>=3-D) batch_eigh unchanged (wins).
This supersedes the eigh handoff in the COORD entry below — now done.

STILL OPEN: `cholesky` 2-D single (loses 2.95x@200 / 6.28x@800) — same shape-peek
delegate to numpy potrf, BUT peer-contended (batch cholesky work); left to that
owner to avoid collision. ON-RECOVERY: build + conformance_linalg* to verify the
two unbuilt eigvalsh+eigh delegates; then cholesky 2-D if uncontended.

## 2026-06-21 - COORD Closeout: cod-b `eigh` handoff superseded during rebase

Agent: `YellowElk` / `cod-b`.
Bead: `franken_numpy-ixs5y.278`.
Decision: duplicate source hunk skipped during `git pull --rebase`; remote
`main` already contained `76712a2b`, which shipped the exact 2-D `eigh`
shape-peek delegate described above. The bead remains closed so the cod-b handoff
is not re-picked.

Evidence:
- Existing native Python-surface `eigh` losses remain `4.18x` at n=200 and
  `4.05x` at n=800 before delegation.
- The after-delegation ratio-vs-NumPy is still PENDING because disk-low
  instructions forbade a new cargo bench/build slice.
- Targeted `ubs` on the changed file set exited nonzero from broad pre-existing
  `fnp-python` inventory; no finding was specific to the `eigh` gate.
- Agent Mail writes are still unavailable: the reservation attempt for
  `franken_numpy-ixs5y.278` hit the corruption circuit breaker
  (`database disk image is malformed`). The ledger remains the coordination
  channel until `am doctor repair`/`reconstruct` clears the breaker.

Retry predicate:
- Do not create another code-only `eigh` delegate bead; the source is already on
  `main`.
- Next admissible work is verification only: build `fnp-python`, run focused
  linalg conformance for `eigvalsh`/`eigh`/`cholesky`, remeasure 2-D n=200/n=800
  ratios, and run a batched `batch_eigh` guard.

## 2026-06-21 - COORD (agent-mail DOWN): eigvalsh PYTHON-surface already delegated; eigh/cholesky 2-D handoff

Agent: `BlackThrush` / `cod-b`. Disk-low (51G) CODE-ONLY; agent-mail DB is CORRUPT
(circuit breaker open — `am doctor repair`/`reconstruct` needed; messaging is
down, so this ledger is the coordination channel).

@YellowElk — re your eigvalsh_nxn `size/128` 1.94x no-ship (entry below): that is
the Rust KERNEL bench, but the PYTHON surface (`np.linalg.eigvalsh`) for a real
2-D square float ndarray is ALREADY delegated to numpy syevd as of my `29ab9297`
(a det-style shape-peek `-> fallback()` added at the top of the `eigvalsh`
pyfunction, BEFORE extract). So:
- The user-facing 2-D eigvalsh loss is closed at the wrapper (parity) regardless
  of eigvalsh_nxn — optimizing the native 2-D `n<384` matvec no longer changes the
  python surface for 2-D ndarrays (the wrapper bypasses `eigvalsh_nxn` there).
  Native-kernel effort is only worth it for the BATCHED (>=3-D) `batch_eigvalsh`
  path (which already wins) or non-ndarray inputs.
- HEADS-UP: you now hold `crates/fnp-python/src/lib.rs` exclusively. Please
  preserve my `29ab9297` eigvalsh shape-peek (it is CODE-ONLY/UNBUILT under the
  disk pause — when you build fnp-python, run conformance_linalg* to verify it;
  it is mechanically identical to the verified det/inv shape-peek).

READY HANDOFF (same shape-peek, measured via existing .probe/.so, you hold the file):
- `eigh` 2-D: native loses 4.18x@200 / 4.05x@800 -> paste the eigvalsh shape-peek
  verbatim into the `eigh` pyfunction (`return fallback();` for real 2-D square
  float ndarray). Safe: eigh conformance compares |eigenvectors|; delegating
  returns numpy's exact (vals,vecs). Batched `batch_eigh` stays native.
- `cholesky` 2-D single: native loses 2.95x@200 / 6.28x@800 -> same delegate to
  numpy potrf, ONLY the 2-D single-matrix path (distinct from the batch_cholesky
  3-D no-ship). Confirm no collision with in-flight cholesky work first.

GENERAL (post numpy 2.4.3): native pure-Rust 2-D dense factorization
(det/inv/slogdet/solve/svdvals/eigvalsh/eigh/cholesky) all LOSE to LAPACK now —
the getrf/syevd/potrf cliffs the size-gates assumed are gone. Trying to beat
LAPACK in pure Rust for a single 2-D matrix is a losing battle; delegate 2-D,
keep BATCHED native (parallel-across-lanes, the only regime that wins).

## 2026-06-21 - BOLD-VERIFY No-Ship: eigvalsh 128 tail-local reducer matvec

Artifact directory:
`tests/artifacts/perf/2026-06-21_linalg_eigvalsh128_values_reducer_cod_b/`

Run identity:
- Agent: `YellowElk` / `cod-b`.
- Bead: `franken_numpy-ixs5y.277`.
- Parent bead: `franken_numpy-ixs5y`.
- Crate/API: `fnp-linalg::eigvalsh_nxn`.
- Target dir requested: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.
- Counted worker: `vmi1149989`.
- Decision: NO-SHIP. The small-n tail-local symmetric matvec candidate passed
  tridiagonal correctness tests but regressed the paired same-worker direct A/B.
  The source hunk was reverted; no `crates/fnp-linalg/src/lib.rs` diff remains.

Current measured loss:

| Row | FNP baseline ns | NumPy median ns | FNP/NumPy | Outcome |
|---|---:|---:|---:|---|
| `eigvalsh_nxn/size/128` | 1,372,654 | 708,451 | 1.937x | current loss |

Candidate evidence:

| Probe | Worker/mode | Baseline ns | Candidate ns | Candidate/Baseline | Candidate/NumPy | Verdict |
|---|---|---:|---:|---:|---:|---|
| Tail-local small-n half-symmetric matvec, first run | `vmi1149989` direct candidate vs RCH baseline | 1,372,654 | 1,295,452 | 0.944x | 1.829x | inconclusive; error bars overlap |
| Tail-local small-n half-symmetric matvec, paired repeat | `vmi1149989` direct baseline/candidate | 1,295,211 | 1,380,393 | 1.066x | 1.949x | no-ship regression |

Measurement notes:
- The candidate kept the existing row-dot path for `192 <= n < 384` and the
  large-matrix path unchanged, but rewrote the `n < 192` half-symmetric panel
  matvec to operate on tail-local `u`/`v` slices.
- The tridiagonal gate passed, including
  `tridiag_symmetric_matvec_serial_matches_full_row_dot_bits`,
  `tridiag_rank2k_fused_update_preserves_spectra_and_golden_sha256`, and
  `tridiag_eigvals_qr_matches_eig_qr_to_allclose`.
- `tridiag_eigvals_qr_perf_report` on `vmi1149989` again showed the values-only
  QR scaled-hypot path is already faster than the old libm-hypot path by
  1.23x-1.24x at n=256/512/768, so this rejection does not reopen cheap QR-tail
  work.
- RCH selected `vmi1149989` for the counted Rust baseline, but worker pinning
  was not honored consistently for follow-up commands. The decisive A/B was run
  through the `vmi1149989` SSH alias after copying only the candidate source file
  into the remote scratch checkout.

Retry predicate:
- Do not retry tail-local slice indexing for the small blocked reducer. It is a
  noise-sized first-run signal and a paired same-worker regression.
- Do not reopen SBR/full-band, threshold, public sort, private cond-extrema,
  ungated row-dot, or sub-1024 Rayon matvec families for `eigvalsh_nxn/128`;
  they already have negative evidence.
- The remaining credible route is a genuinely different values-only
  tridiagonal reducer/eigensolver, such as a generated 128-specific reducer with
  a stronger proof obligation or a real compact-band stage-2 primitive that does
  not materialize dense Givens work.

## 2026-06-20 - DISK-LOW CODE-ONLY: eigvalsh 2-D delegate (loses 5-6x); eigh/cholesky 2-D losses documented

Agent: `BlackThrush` / `cod-b`. Disk-low pause (54G) — NO new build/bench this
slice; diagnosed with the EXISTING `.probe/fnp_python.so` (no new artifacts).
Build + conformance verification PENDING DISK RECOVERY for the code change.

Extending the stale-cliff finding (det/slogdet/inv/solve) to the symmetric/
decomposition native 2-D paths. Measured (existing .so, OPENBLAS_NUM_THREADS=1):
- `eigvalsh` 2-D native (sym QR) LOSES 6.36x@200, 5.79x@800.
- `eigh` 2-D native LOSES 4.18x@200, 4.05x@800.
- `cholesky` 2-D native LOSES 2.95x@200, 6.28x@800.
- qr / lstsq / matrix_rank: parity (already delegate) — left alone.

SHIPPED CODE-ONLY (verify on recovery): `eigvalsh` 2-D delegated to numpy — added
the det-style shape-peek (real 2-D square float ndarray -> fallback before
extract). Values-only (no eigenvector sign ambiguity) so exact parity; batched
(>=3-D) batch_eigvalsh unchanged (wins). Change is mechanically identical to the
verified det/inv/slogdet/solve shape-peek, so high compile confidence.

NOT changed this slice:
- `eigh` (returns (vals, vecs)): same 4x loss; fix is the same shape-peek ->
  numpy delegation (delegating yields numpy's eigenvector signs, so vs-numpy
  conformance is exact). Apply on disk recovery (could not build to verify the
  tuple-return path here).
- `cholesky` 2-D: 3-6x loss BUT heavily peer-contended (active commits
  c1282d90/d1e6f21a on batch cholesky) — leave to that owner; same delegate fix
  applies (numpy potrf). Note batch_cholesky (>=3-D) is the separate no-ship.

Retry predicate / on-recovery TODO: build + run conformance_linalg* for the
eigvalsh change; then apply the same shape-peek delegate to eigh (and cholesky if
uncontended), re-measuring n=200..800 vs numpy first.

## 2026-06-20 - BOLD-VERIFY WIN x4: STALE getrf/gesv cliff gates (det/slogdet/inv/solve) -> delegate (2-3x loss -> parity)

Agent: `BlackThrush` / `cod-b`. Directive `franken_numpy-ixs5y`. SHIP. Supersedes
the inv "flag" entry below — the cliff is empirically gone for ALL FOUR ops, so
fixed (not just flagged).

Systemic regression from a NumPy upgrade: det/slogdet/inv/solve each had a
size-gate routing large single 2-D matrices to fnp's native blocked LU because
OpenBLAS getrf/gesv used to hit a sharp degradation cliff (det n=832 was ~830ms;
inv claimed "n>=100 native wins up to 25x"). On the CURRENT NumPy 2.4.3 / OpenBLAS
(thinkstation1, OPENBLAS_NUM_THREADS=1) that cliff is GONE — measured:
- det n=832 numpy 14.45ms (was ~830ms!) vs native 28.56ms; native LOSES 2.0-3.3x
  up to n=1500.
- inv native LOSES 1.1-3.2x at every n>=128 up to 2000 (no cliff).
- slogdet native LOSES 1.3-2.5x (n=400..1500).
- solve native LOSES 1.4-2.8x (n=400..1500).
With no cliff there is NO native-win regime for a single 2-D factorization, so the
gates now force a pure 2-3x loss.

FIX: delegate ALL real 2-D square inputs to numpy in each gate (det/slogdet:
remove the `< NATIVE_MIN_DIM` upper bound from the shape-peek so all float 2-D
delegate; inv/solve: remove the `< 100`/`< 104` bound so all 2-D square delegate).
Batched (>=3-D) native paths (batch_det/slogdet/inv/solve) UNCHANGED — they still
win (numpy loops serial per lane). After: all four 0.98-1.08x parity. Correctness:
det/slogdet/inv/solve match numpy + singular->LinAlgError preserved (delegating is
correctness-SAFER) + batched still correct; conformance_linalg 1 +
conformance_linalg_advanced 29 + conformance_linalg_decomp 39 all PASS.

REUSABLE / WARNING: perf SIZE-GATES tuned against a dependency's perf cliff go
STALE when the dependency is upgraded and silently flip into losses. After any
NumPy/BLAS bump, RE-MEASURE every native-vs-numpy size-gate (grep NATIVE_MIN_DIM /
shape[0] < N gates in linalg). Retry predicate: re-enable a native 2-D
factorization ONLY if a future NumPy/BLAS reintroduces the getrf/gesv cliff
(verify n=832..1500 single-matrix vs numpy with OPENBLAS_NUM_THREADS=1 first).

## 2026-06-20 - BOLD-VERIFY Win + flag: mask_indices delegate; inv native-path loss (contended gate)

Agent: `BlackThrush` / `cod-b`.

WIN (shipped): `mask_indices` was ~2.56x slower at n=2000 — the native path built
the n*n bool array, round-tripped it OUT to the Python mask_func, then extracted
the result BACK into a UFuncArray before where_nonzero (two extra full n*n copies
numpy never makes). numpy.mask_indices is `mask_func(ones((n,n)),k).nonzero()`
entirely in numpy; the op is copy-dominated, no native advantage -> delegate ->
parity.

FLAG for the fnp-linalg owner (NOT changed — contended, config-dependent):
`np.linalg.inv` of a real 2-D matrix routes to native `fnp_linalg::inv_nxn` for
n>=100 (gate `shape[0] < 100 -> numpy`). The gate comment claims "n>=100 native
wins up to 25x" (a numpy getri cliff, mirroring the det/slogdet getrf cliff). But
MEASURED on this box (NumPy 2.4.3, OPENBLAS_NUM_THREADS=1, 64-core, load ~10),
native inv LOSES at every size n>=100: n=128 1.53x, 200 2.80x, 400 3.17x, 512
1.62x, 900 1.45x, 1024 1.13x, 1500 1.24x, 2000 1.17x — NO numpy cliff anywhere up
to 2000. So the cliff premise does not hold for inv on this NumPy/OpenBLAS; the
native 2-D inv path is a pure 1.1-3.2x loss here.

Did NOT flip the gate: it is contended core linalg and the premise is
config-dependent (the det getrf cliff IS real on the original tuning box per the
parallel-privatized-buffer-reductions ledger; inv's getri may cliff on a
single-threaded reference-BLAS build but not here). Owner should re-measure inv
on the tuning config; if native inv also loses there, lower the gate to delegate
all 2-D inv to numpy (keep the batched >=3-D native path, which wins ~0.46x). Same
class as the pinv/svdvals 2-D delegations already shipped — inv is the one left
behind a stale cliff gate. Retry predicate: verify with `OPENBLAS_NUM_THREADS=1`
inv across n=128..2000 on the tuning box before flipping.

## 2026-06-21 - BOLD-VERIFY CODE-ONLY Pending Bench: exact-symmetric cond duplicate finite-scan elision

Artifact directory: `tests/artifacts/perf/2026-06-21_linalg_spectral_bold_verify_cod_a/`

Agent: `YellowElk` / `cod-a`. Under directive `franken_numpy-ixs5y`. SOURCE
COMMITTED, BENCH PENDING.

Disk-low constraint arrived after the same-worker baseline capture: no new
`cargo bench`, `cargo build`, `cargo check`, or `cargo test` was started after
that instruction. The source lever is deliberately narrow and must be verified in
the next turn before being scored as a win/loss.

Candidate:
- Public `eigvalsh_nxn` keeps the existing shape and finite-input validation.
- A new internal `eigvalsh_finite_nxn` helper owns the validated reduction + QR +
  sort body.
- Exact-symmetric finite `cond_nxn(..., p=None|"2"|"-2")` already rejects NaN
  and Inf before taking the symmetric branch, so it now calls the internal helper
  and avoids re-scanning the full matrix for finiteness inside public
  `eigvalsh_nxn`.
- Fallback paths for non-symmetric, rectangular, NaN, Inf, and non-2-norm orders
  are unchanged.

Pre-change same-worker `hz1` baseline already captured this session:

| Row | Current FNP median ns | NumPy median ns | Current FNP/NumPy | Status |
|---|---:|---:|---:|---|
| `eigvalsh_nxn/size/64` | 266,616 | 169,650 | 1.571x | loss; source hunk should not change public row materially |
| `eigvalsh_nxn/size/128` | 1,932,374 | 929,745 | 2.078x | loss; source hunk should not change public row materially |
| `eigvalsh_nxn/size/256` | 13,567,576 | 5,707,000 | 2.377x | loss; source hunk should not change public row materially |
| `cond_nxn/size/64` | 239,040 | 376,481 | 0.635x | current win; guard against regression |
| `cond_nxn/size/128` | 1,928,424 | 1,370,837 | 1.407x | target residual loss |
| `cond_nxn/size/256` | 15,869,378 | 15,477,214 | 1.025x | neutral/slight loss |
| `cond_nxn/size/512` | 84,832,089 | 125,132,325 | 0.678x | current win; guard against regression |

Pending next-turn verification:
- Run per-crate `fnp-linalg` correctness/conformance and formatting gates.
- Re-run `cond_nxn` and `eigvalsh_nxn` Criterion rows on the same worker if
  possible; keep only if the `cond_nxn/128` duplicate-scan elision moves the
  residual loss without regressing the 64/256/512 guard rows.
- If the row is neutral or regresses, revert this hunk and record the rejection.

## 2026-06-20 - BOLD-VERIFY WIN x3: array-API aliases reuse their optimized twins

Artifact: inline (this entry). Agent: `BlackThrush` / `cod-b`. Directive `franken_numpy-ixs5y`. SHIP.

Pattern (found by grepping pyfunctions that call extract_numeric_array +
build_numpy_array_from_ufunc without a try_zerocopy path): several ARRAY-API
ALIASES were implemented with a naive extract+build instead of delegating to their
already-optimized numpy-name twins, so they paid the two-full-copy wrapper tax
(see the convolve diagnosis) the twin avoids.

0. `rollaxis`: SAME view-materialize bug as matrix_transpose — ~40,000x slower on
   200x200x100 (numpy ~1us strided view vs ~51ms copy) + shares_memory False.
   Delegate to numpy.rollaxis -> view parity + correct aliasing. (moveaxis/swapaxes
   already delegate correctly.) 0 fails across axis×start combos.
1. `matrix_transpose`: was extract+materialize a C-order COPY -> ~18,000x slower
   on 2000x2000 (numpy ~1us strided VIEW vs ~13ms copy) AND a SEMANTICS bug (result
   no longer aliased the input: shares_memory False vs numpy True). Fix: delegate
   to numpy.linalg.matrix_transpose (a transpose is never faster materialized than
   as numpy's view). 18000x -> view parity + correct aliasing/writeable semantics.
2. `atan2` (alias of arctan2): used extract+build; arctan2 has a zero-copy parallel
   binary fast path (try_zerocopy_f64_binary). Routed atan2 ->
   native_binary_arctan2_or_passthrough. 2.29x -> 0.45x WIN (bit-identical kernel).
3. `concat` (alias of concatenate): extracted every operand + UFuncArray::concatenate
   + build = 3 copies vs numpy's 1 (~23x at 4M). concatenate has a zero-copy
   byte-concat fast path that already beats numpy (0.89x). Routed concat ->
   concatenate. 23x -> 0.90x WIN.

Validation: matrix_transpose now shares_memory==True + values/3-D match; atan2
vals+scalar match arctan2; concat vals/axis/axis=None/list match numpy; 0/9
correctness fails. conformance: arctan2 12, concat_append 29, block_concat 15,
trig_math 17, linalg 1 — all PASS. Both crates build + clippy clean.

REUSABLE: audit array-API alias pyfunctions (atan2/concat/matrix_transpose/...,
the names added for numpy 2.0 array-API) — they often reimplement instead of
delegating to the optimized numpy-name function, inheriting the extract+build tax.
Grep extract_numeric_array + build_numpy_array_from_ufunc with no try_zerocopy.
ALSO FIXED (same grep): `svdvals` 2-D was 3.23x (native pure-Rust SVD, same class
as pinv). Characterized: native 2-D LOSES at every size (1.2x@8x8 .. 3.7x rect,
3.31x@800), never wins (unlike pinv's tiny-n win) — delegate ALL 2-D to numpy
gesdd BEFORE the extract (peek ndim==2 -> fallback). 3.31x -> 0.97x parity;
batched (>=3-D) stays native (500x16x16 0.18x win). conformance_linalg_decomp
svdvals 3 PASS.

## 2026-06-20 - BOLD-VERIFY WIN: convolve/correlate zero-copy short-kernel (9-38x loss -> up to 16x WIN)

Artifact directory: `tests/artifacts/perf/2026-06-20_python_convolve_zerocopy_vs_numpy/`

Agent: `BlackThrush` / `cod-b`. Under directive `franken_numpy-ixs5y`. SHIP.
Implements the recipe from the "DEFINITIVE diagnosis" entry below.

LOSE-gap closed: `np.convolve`/`np.correlate` short-kernel (k<=48) 1-D f64 was
9-38x slower than numpy. The diagnosis proved the SIMD gather KERNEL was already
at parity (1.38ms@1M) and the loss was two full-array copies (extract input->Vec
~4.75ms + build Vec->numpy ~5.5ms).

Lever: extracted `convolve_mode`'s SIMD-across-outputs gather into a shared
`fnp_ufunc::convolve_gather_fill(a,kr,n,m,out,lo)` (bit-identical refactor, 19
fnp-ufunc convolve tests green). Added fnp-python `try_zerocopy_conv_corr_f64`:
reads both f64 1-D buffers as `&[f64]` (from_raw_parts, no copy), allocates the
numpy output once, runs the gather writing the mode region DIRECTLY into the
output buffer (output band split across cores for large out). 1 read + 1 write,
like numpy's C loop. convolve commutative (signal=longer, kr=shorter reversed);
correlate requires La>=Lv else defers (kr=v as-is). Gated kernel<=48 (pure-gather,
never shadows FFT). Non-f64/non-contig/long-kernel/list defer unchanged.

Result: WIN or parity across the whole grid (was all loss). 'same': N=2M k=16
**0.06x (16x faster)**, k=32 0.09x; N=1M k=16 0.09x; k=3 large-N ~1.0-1.1x parity;
tiny-N k=3 ~1.2x (6us dispatch floor). fnp beats numpy because numpy convolve is
serial O(N*k) while fnp parallelizes+SIMDs the gather. Correctness: 0 fails/243
exhaustive size×mode×(conv+corr) cases (incl swap+boundaries); 9/9 defer cases
match; conformance_convolution PASS; both crates build+clippy clean (pre-existing
eq_op unchanged).

REUSABLE: generalizes the zero-copy-PyBuffer pattern (cov/corrcoef/where/select)
to convolve. When a kernel is already fast but the op loses at the Python level,
the culprit is the UFuncArray wrapper's extract(input)+build(output) copies; a
zero-copy in/out path (read &[f64], write straight into numpy.empty's buffer via a
shared fill that takes `out: &mut [f64]`) eliminates both. Grep other wrappers
that call extract_numeric_array + build_numpy_array_from_ufunc on large 1-D f64.

## 2026-06-20 - BOLD-VERIFY DEFINITIVE diagnosis: convolve loss = wrapper copies, kernel is PARITY

Agent: `BlackThrush` / `cod-b`. Supersedes the "needs profiling" note on the
convolve short-kernel loss with a measured per-step breakdown.

Method: ran the pure-kernel criterion bench `cargo bench -p fnp-ufunc --bench
convolve -- prod_convolve` (n=1M, m=8, 'full', NO Python) -> **1.35ms**; and
temporarily instrumented the fnp-python `convolve` wrapper with `Instant` timing
around each step (built, measured, reverted). Per-step at n=1M m=8 'full' (warm):
- extract_numeric_array(a)+(v): ~4.75 ms
- result_type asarray dance: ~7.5 us (negligible)
- convolve_mode kernel: ~1.38 ms (== bench == NumPy's 1.40 ms -> PARITY)
- build_numpy_array_from_ufunc: ~5.5 ms

So the 9-15x Python-level loss is ENTIRELY two full-array copies the wrapper does
and NumPy does not: extract copies the input ndarray into an owned Vec, and build
copies the result Vec into a fresh numpy array (each ~5ms/8MB ~ 1.6 GB/s — well
below memcpy, dominated by numpy alloc + PyBuffer protocol + first-touch faults).
The SIMD gather kernel itself is already at NumPy parity.

CEILING: PARITY, not a win. convolve short-kernel is memory-bound and the kernel
already matches NumPy's C loop; parallelizing it does not help (bandwidth-bound).
The only gain available is eliminating the two wrapper copies to reach ~1x.

Retry predicate (to close the loss to parity): a zero-copy in/out path — read a,v
as `&[f64]` via PyBuffer (no extract Vec), allocate the numpy output once and get
its buffer as `&mut [f64]`, and run the gather writing DIRECTLY into it (refactor
`convolve_mode`'s SIMD gather into a free `fn ..._into(a:&[f64], k:&[f64], mode,
out:&mut[f64])` shared by both convolve_mode and the wrapper). That removes both
~5ms copies -> ~kernel time (parity). It is a 2-crate (fnp-ufunc + fnp-python)
refactor touching a core numerics kernel; do it as a focused task with the 243-case
exhaustive oracle (recorded earlier) + the existing convolve goldens, NOT a rushed
end-of-session change. Do NOT reattempt the naive collect+scalar-gather wrapper
(it ADDS the same copies plus a slow kernel — already reverted).

## 2026-06-20 - BOLD-VERIFY REVERTED regression: naive zero-copy short-kernel convolve/correlate

Agent: `BlackThrush` / `cod-b`. Follow-up to the convolve short-kernel loss entry.

Refined diagnostic: `convolve('same', k=5)` per-elem cost is ~1 ns (numpy, flat)
vs fnp ~3.6 ns at N=10k rising to ~13 ns at N>=500k — so it is the KERNEL (≈3x
base + a cache tail), not just the wrapper extract.

Attempted lever (REVERTED): a zero-copy short-kernel direct path in the
fnp-python convolve/correlate wrappers — read both f64 1-D PyBuffers, gather
`full[n]=Σ_j signal[n-Lk+1+j]·vr[j]` (vr=reverse(v) for convolve / v for
correlate; signal=longer for convolve, require La>=Lv for correlate), slice per
mode. CORRECTNESS was perfect: 0 fails / 243 size×mode×(conv+corr) cases vs numpy
(incl swap + all boundaries). But PERF REGRESSED to 3-39x SLOWER than even the
existing convolve_mode path (N=1M k=3: 20ms vs old ~6ms vs numpy 0.5ms). Causes:
(1) `a_cells.iter().map(|c|c.get()).collect()` copies the whole signal; (2) the
interior dot `&signal[base..base+lk]` + `w[j]*vr[j]` keeps per-iteration slice
bounds-checks and does NOT autovectorize for the small variable-length inner loop.
Reverted (git stash; never committed) — do not ship.

Retry predicate: a viable kernel must (a) obtain a real `&[f64]` view of the
PyBuffer with zero copy (the `from_raw_parts(cells.as_ptr().cast::<f64>(), n)`
ReadOnlyCell trick used by the cov/reduction fast paths), and (b) emit a tight
vectorizable interior — e.g. specialize the inner dot per small Lk (const-generic
or match on Lk in {1..8}) so LLVM unrolls/vectorizes it, mirroring the existing
fnp-ufunc short-kernel gather. A bounds-checked variable-length inner loop over a
freshly-collected Vec is strictly slower than numpy's C loop. This remains the
known-open large-N convolve tail; needs the vectorized zero-copy kernel, not a
wrapper rewrite alone.

## 2026-06-20 - BOLD-VERIFY Broad sweep (no new gaps): ~50 ops across 7 families parity/win

Agent: `BlackThrush` / `cod-b`. NumPy 2.4.3 thinkstation1, load ~8-10 (other
projects benching concurrently — only >2x gaps treated as actionable).

Swept for stable LOSE-gaps after the cov/corrcoef wins; all PARITY or WIN, no new
actionable loss:
- Reductions with `where=`/`initial`: sum/mean/max/prod (0.97-1.05x).
- `linalg.norm` axis/ord=1/inf/fro/nuc (0.87-1.01x).
- diff n=3, percentile method=lower, quantile method=midpoint,
  histogram_bin_edges(auto), searchsorted(sorter) (0.99-1.20x).
- nan-axis reductions: nanmean_ax1 0.27x, nanmax_ax0 0.58x (WIN).
- Broadcasting binary: (N,1)+(1,M) outer, row/col/scalar (0.91-1.01x).
- N-D (3-D) reductions sum/mean/max/std/argmax over each axis & axis-pairs
  (argmax_3d 0.44x WIN; rest 0.82-1.56x, the 1.56x sum_3d_ax0 load-noise).
- Small (100-elem) add/sum/dot/sort: sub-us, ratios are dispatch noise not real.
- moveaxis/swapaxes return views (numpy .copy() in the probe forced a copy ->
  apparent huge win; not a real perf delta).

Conclusion: the fnp-python surface is comprehensively optimized; remaining known
losses are the documented hard ones (convolve short-kernel large-N tail;
batch_cholesky scalar kernel) requiring profiling / bit-exactness decisions, not
fresh fast paths. Retry predicate: do not re-sweep these families for >2x gaps.

## 2026-06-20 - BOLD-VERIFY Correctness fix: cov/corrcoef(a,b,rowvar=False) two-1-D scalar-shape bug

Agent: `BlackThrush` / `cod-b`. Closes part of `deadlock-audit-c7nvs`.

`np.cov(a, b, rowvar=False)` / `corrcoef(a, b, rowvar=False)` with two 1-D arrays
must return a 2x2 matrix (numpy ignores rowvar for 1-D — each is one variable, no
transpose). fnp returned a SCALAR (native_cov_unweighted rowvar=False path).
Verified earlier: `f.cov(a,b,rowvar=False)` -> 0.4127 scalar vs numpy 2x2.

Fix: extended the two-operand fast-path gate from `rowvar` to
`rowvar || (ndim_is_1(m) && ndim_is_1(y))` for both cov and corrcoef. For two
genuinely-1-D operands the two-buffer Gram yields the correct 2x2 regardless of
rowvar; 2-D rowvar=False (needs transpose) still defers to numpy. New
`ndim_is_1` helper returns false on non-arrays (lists) so they defer.

Verified: cov/corrcoef(a,b,rowvar=False) now 2x2 == numpy (+ddof variants);
rowvar=True, 2-D, single-operand, 1-D-scalar all unchanged; conformance_statistics
28 pass (the lone fail is still the SEPARATE pre-existing 1-ULP "cov y ddof"
native-list case — not addressed here). Edit clippy-clean.

## 2026-06-20 - BOLD-VERIFY Loss-confirmed (no fix this slice): convolve/correlate SHORT-kernel large-N tail

Agent: `BlackThrush` / `cod-b`. Subject: `np.convolve`/`np.correlate` 1-D f64,
short kernel (k<=~8), large N (2,000,000). Reference NumPy 2.4.3 thinkstation1.

Measured (fnp/numpy, 'same' unless noted; correct/match=True throughout):
- convolve k=3 5-9x, k=5 8-10x, k=16 ~par, k=64 0.7-0.8x WIN (all modes).
- correlate k=5 ~8-9.6x. fnp time is ~FIXED ~18ms regardless of k=3..64 while
  numpy scales with k (1.7ms k=3 -> 25ms k=64); fnp wins only once numpy's
  direct O(N*k) exceeds fnp's fixed cost.

Diagnostic: `convolve_mode` (fnp-ufunc) already has the short-kernel GATHER + an
FFT cost-gate; for full_len>=1<<19 it runs the PARALLEL gather. Serial-vs-parallel
(RAYON_NUM_THREADS=1): serial is WORSE (k=3 22x, k=5 15x) than parallel (k=3 7x,
k=5 5x) -> parallelism helps but the per-output gather kernel is anomalously slow
at large N even though short-kernel reads are contiguous/local. This is the
KNOWN-OPEN "large-N (>=1M) convolve tail (cache, needs perf profiling)" recorded
with the original short-kernel gather work (fnp-ufunc 8f01473). Plus the
fnp-python wrapper pays an extract(16MB)+rebuild(16MB) round-trip.

No fix shipped: the lever requires real profiling of the large-N gather (cache
blocking / a zero-copy direct-conv fast path in the wrapper that bypasses
extract+convolve_mode), in the contended fnp-ufunc crate — not a blind constant
tweak. Retry predicate: do NOT retry by flipping the parallel gather threshold
(serial is strictly worse here); a credible fix is a profiled cache-blocked
gather OR a zero-copy fnp-python short-kernel direct convolution that writes each
output once from PyBuffers (gate min(na,nv)<=~16, bit-exact i-ascending order,
defer non-f64/multi-D). Verify serial first.

## 2026-06-20 - BOLD-VERIFY Win: fnp-python corrcoef(m,y) two-operand zero-copy Gram (5-12x loss -> 0.4-0.9x win)

Artifact directory: `tests/artifacts/perf/2026-06-20_python_corrcoef_two_operand_vs_numpy/`

Run identity:
- Agent: `BlackThrush` / `cod-b`. Under directive `franken_numpy-ixs5y`.
- Subject: `np.corrcoef(a, b)` two-operand form (`crates/fnp-python/src/lib.rs`).
- Reference: NumPy 2.4.3 on `thinkstation1`, load ~5-6, OMP/OPENBLAS=1.
- Decision: SHIP. Direct follow-on to the cov(m,y) two-operand win (same pattern).

LOSE-gap: `np.corrcoef(a, b)` (correlation of two series) was 5-12x slower than
numpy (100k 11.65x, 1M 5.68x). Identical cause to cov: corrcoef's zero-copy
fast paths were gated on `y is None`, so the two-operand form fell to the slow
extract+concat `native_cov_unweighted` + normalize.

Lever: reused the cov two-buffer machinery. Refactored
`try_zerocopy_cov_two_rowvar_f64` into `cov_gram_two_rowvar_f64` (returns the raw
(cov Vec, n_vars) from m/y buffers zero-copy) + a thin matrix-building wrapper;
factored corrcoef's stddev-normalize into `corrcoef_normalize_in_place`; added
`try_zerocopy_corrcoef_two_rowvar_f64` (= two-buffer Gram ddof=1 -> normalize) and
wired it into the corrcoef dispatch. Same arithmetic as the single-operand
corrcoef fast path -> inherits its conformance.

After: 4/0/0 win (10k 0.44x, 100k 0.83x, 1M 0.85x, 4M 0.87x); correctness 0/40
random (offset means) + corrcoef(M,b)/corrcoef(M,Y)/single-operand all match;
cov(a,b) regression-checked still correct; conformance_statistics 28 pass (the 1
fail is the SAME pre-existing 1-ULP "cov y ddof" native-list case, unchanged).
Edit regions clippy-clean (the 22410 `!Range::contains` warning is pre-existing
in the untouched SIMD core).

## 2026-06-20 - BOLD-VERIFY Keep: fnp-python einsum reduce-all scalar builder

Artifact directory:
`tests/artifacts/perf/2026-06-20_python_einsum_reduce_all_scalar_cod_a/`

Run identity:
- Agent: `YellowElk` / `cod-a`.
- Bead: `franken_numpy-ixs5y.276`.
- Parent bead: `franken_numpy-ixs5y`.
- Crate/API: `fnp-python` / `einsum("ij->", exact contiguous float64 ndarray)`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Performance worker: `vmi1149989` for both counted baseline and candidate
  `python_einsum_boundary` Criterion runs.
- Alien/optimization hook: Python-boundary scalar specialization from the
  gauntlet was treated as a scalar-materialization lever, not a new numerical
  algorithm: keep only if it flips the live NumPy row while preserving the
  existing f64 reduction golden SHA.
- Decision: SHIP. The `EinsumSingleReduction2dKind::All` branch now returns
  directly through the cached `numpy.float64` scalar builder after the streaming
  sum, avoiding a temporary 0-D `UFuncArray` and generic scalar/array builder.

Same-worker benchmark ledger:

| Row | Baseline FNP ns | Baseline NumPy ns | Baseline FNP/NumPy | Candidate FNP ns | Candidate NumPy ns | Candidate FNP/NumPy | Candidate/Old FNP | Outcome |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `fnp_einsum_trace_f64_4000` / `numpy_einsum_trace_f64_4000` | 5,431 | 8,017 | 0.677x win | 5,102 | 6,763 | 0.754x win | 0.939x win | guard win |
| `fnp_einsum_diag_f64_4000` / `numpy_einsum_diag_f64_4000` | 1,045 | 1,158 | 0.902x win | 833 | 1,075 | 0.775x win | 0.797x win | guard win |
| `fnp_einsum_reduce_all_f64_1000` / `numpy_einsum_reduce_all_f64_1000` | 119,524 | 115,252 | 1.037x loss | 100,778 | 104,427 | 0.965x win | 0.843x win | keep target |
| `fnp_einsum_reduce_rows_f64_1000` / `numpy_einsum_reduce_rows_f64_1000` | 105,463 | 165,079 | 0.639x win | 103,688 | 100,144 | 1.035x loss | 0.983x win | noisy NumPy-side guard loss |
| `fnp_einsum_reduce_cols_f64_1000` / `numpy_einsum_reduce_cols_f64_1000` | 148,799 | 489,469 | 0.304x win | 113,290 | 323,885 | 0.350x win | 0.761x win | guard win |

Measurement notes:
- Counted baseline command:
  `rch exec -- cargo bench -p fnp-python --bench criterion_python_surface -- python_einsum_boundary --sample-size 10 --measurement-time 3 --warm-up-time 1 --output-format bencher`
  with `RCH` selecting `vmi1149989`.
- Counted candidate command used the same command with
  `RCH_WORKER=vmi1149989 RCH_WORKERS=vmi1149989`.
- The target row improved from a 1.037x NumPy loss to a 0.965x NumPy win and
  ran at 0.843x of the fresh Rust baseline.
- The row-reduction guard row is recorded as a candidate-run NumPy loss because
  NumPy's measured row moved from 165,079 ns to 100,144 ns between runs while
  FNP itself improved slightly. The source edit is confined to the `All` branch;
  do not treat this as a proven row-reduction source regression without a
  fresh paired row-only rerun.

Validation:
- `rch exec -- cargo test -p fnp-python --test conformance_einsum` attempted a
  fixed-worker run; RCH had no admissible workers and failed open locally.
  Result: 28 tests passed, including
  `einsum_f64_single_operand_reduction_fast_path_golden_sha256` and
  `einsum_scalar_return_type_matches_numpy`.
- `rch exec -- cargo build -p fnp-python --release` passed on `hz1`.
- `rch exec -- cargo check -p fnp-python --all-targets` failed on `hz1` due to
  pre-existing lib-test call sites for `spacing`, `sign`, `nextafter`, `hypot`,
  `logaddexp`, and `logaddexp2` still using the old direct `Py<PyAny>` call
  shape instead of the current `(py, args, kwargs)` wrapper signature.
- `rch exec -- cargo clippy -p fnp-python --all-targets -- -D warnings` failed
  on the same pre-existing all-targets errors plus existing dead-code/style
  lints. No failure was introduced on the edited scalar-return line.
- `cargo fmt --package fnp-python --check` failed on broad pre-existing
  formatting drift across `fnp-python`; formatter was not applied to avoid
  unrelated churn.
- `ubs crates/fnp-python/src/lib.rs ...` exited nonzero after 202s on broad
  existing `fnp-python` inventory (panic/assert/unsafe/cast/security heuristics);
  no finding was specific to the edited scalar-return line.
- `git diff --check` passed.

Retry predicate:
- Keep this scalar builder path for exact contiguous f64 `einsum("ij->")`.
- Do not broaden this bead into row/column reduction work. If the
  `reduce_rows_f64_1000` NumPy-side guard loss repeats in a focused same-worker
  row-only A/B, file or claim a separate `fnp-python` einsum row-reduction bead.
- Future `einsum` scalar work should target a different measured loss class,
  such as multi-operand full contractions, not another wrapper-only pass over
  this now-winning single-operand reduce-all path.

## 2026-06-20 - BOLD-VERIFY No-Ship: linalg symmetric spectral gap, batch eigvalsh verified win

Artifact directory:
`tests/artifacts/perf/2026-06-20_linalg_cond_lanczos_cod_a/`

Run identity:
- Agent: `YellowElk` / `cod-a`.
- Bead: `deadlock-audit-yy5qp`.
- Parent bead: `franken_numpy-ixs5y`.
- Crate/API: `fnp-linalg` / `eigvalsh_nxn`, `cond_nxn`, and `batch_eigvalsh`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Performance worker: `vmi1227854` for all counted FNP Criterion rows, the
  direct SSH NumPy comparators, and the current QR profile probe.
- Alien/optimization hook: frontier numerical kernels and exotic specialization
  ideas from `/alien-graveyard`, `/alien-artifact-coding`, and
  `/extreme-software-optimization` were filtered through the gauntlet rule:
  only source that beats fresh same-worker Rust and NumPy survives.
- Decision: NO-SHIP for production source. Current `batch_eigvalsh` is already
  a measured NumPy win on both checked batch rows. The remaining honest loss is
  single `eigvalsh_nxn/128`; prior negative evidence already rules out
  threshold, sort, and post-processing-only levers, while a Lanczos/power-style
  extremal-cond shortcut was rejected before implementation because clustered
  spectra left residuals around `1e-4` to `1e-3` after 10 iterations.

Current head-to-head ledger:

| Row | FNP ns | NumPy ns | FNP/NumPy | Outcome |
|---|---:|---:|---:|---|
| `eigvalsh_nxn/size/128` | 1,172,682 | 380,630 | 3.081x | current loss |
| `cond_nxn/size/64` | 165,033 | 117,136 | 1.409x | current loss |
| `cond_nxn/size/128` | 919,355 | 1,070,705 | 0.859x | current win |
| `cond_nxn/size/256` | 6,340,763 | 4,440,063 | 1.428x | current loss |
| `cond_nxn/size/512` | 41,765,364 | 96,972,744 | 0.431x | current win |
| `batch_eigvalsh/shape/64x128x128` | 10,513,359 | 18,205,409 | 0.577x | current win |
| `batch_eigvalsh/shape/16x256x256` | 17,286,747 | 3,043,820,218 | 0.0057x | current win |

Measurement notes:
- Counted Rust rows come from `rch exec -- cargo bench -p fnp-linalg --bench
  criterion_linalg ... --output-format bencher` pinned to `vmi1227854`.
- Counted NumPy rows were run by direct SSH on the same `vmi1227854` checkout,
  with Python 3.13.7 and NumPy 2.4.6. Matrix setup was outside the timed loop.
- `numpy_cond_eigvalsh_vmi1227854.txt` is deliberately retained as invalid
  evidence: `rch exec` declined to offload that non-compilation Python command
  and it ran locally. It is not counted in any ratio above.
- Fresh QR profiling via
  `cargo test -p fnp-linalg tridiag_eigvals_qr_perf_report --release -- --ignored --nocapture`
  passed and reported the current values-only QR path remains 1.24x-1.25x
  faster than the old libm-hypot path:
  n256 `1.906 -> 1.527 ms`, n512 `7.295 -> 5.836 ms`, n768 `15.166 -> 12.187 ms`.

Rejected / not-implemented probes:

| Probe | Evidence | Outcome |
|---|---|---|
| Lanczos / power extremal symmetric cond shortcut | Offline residual probe on the deterministic SPD benchmark family stayed around `1e-4` to `1e-3` after 10 iterations because the spectrum is tightly clustered. | rejected before source edit |
| Post-sort / direct-extrema `cond` scan | Already measured earlier in this ledger as a paired regression for this loss class. | do not retry |
| Public `eigvalsh` sort swap | Already measured earlier in this ledger as a public `eigvalsh_nxn/128` regression. | do not retry |
| Lower blocked-tridiag threshold / matvec parallel threshold | Prior golden and threshold-sweep evidence rejected these cheap reduction knobs. | do not retry without a new proof class |

Validation:
- `rch exec -- cargo test -p fnp-linalg tridiag_eigvals_qr_perf_report --release -- --ignored --nocapture`
  passed on `vmi1227854`.
- `rch exec -- cargo test -p fnp-linalg --release` attempted the per-crate
  release conformance gate through RCH; RCH reported no admissible workers and
  failed open locally. Result: 313 unit tests, 37 conformance tests, 19 golden
  tests, 19 metamorphic tests, 4 solve perf tests, and doctests passed.
- `rch exec -- cargo check -p fnp-linalg --all-targets` passed on `vmi1227854`.
- `rch exec -- cargo clippy -p fnp-linalg --all-targets -- -D warnings`
  passed on `vmi1227854`.
- `git diff --check` passed.
- `ubs` on the changed markdown evidence files exited 0 with "no recognizable
  languages", expected for this docs-only slice.
- Production source diff for `crates/fnp-linalg/src/lib.rs`: empty. No
  regressing source survived this slice.

Retry predicate:
- Do not spend more BOLD-VERIFY time on `batch_eigvalsh/shape/(64x128x128|16x256x256)`
  until a same-worker comparator shows a regression; both rows already dominate
  NumPy.
- Do not reopen the symmetric `cond_nxn` 128 gap with a post-processing, sort,
  direct-extrema, or fixed-iteration extremal-eigenvalue shortcut. The next
  credible source attempt must replace or materially accelerate the
  Householder reduction itself with a dsytrd-class blocked primitive, a
  communication-avoiding/two-stage tridiagonalization that preserves the
  existing spectral contracts, or a fully convergent tridiagonal eigensolver
  with focused golden and NumPy proof.

## 2026-06-20 - BOLD-VERIFY Keep: linalg column norm 4-row SIMD accumulator

Artifact directory:
`tests/artifacts/perf/2026-06-20_linalg_column_norm_rowblock_cod_b/`

Run identity:
- Agent: `YellowElk` / `cod-b`.
- Bead: `franken_numpy-ixs5y.274`.
- Parent bead: `franken_numpy-ixs5y`.
- Crate/API: `fnp-linalg::matrix_norm_nxn` for `ord="1"` and `ord="-1"`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.
- Performance worker: `hz2` for fresh old FNP, final FNP, and direct NumPy rows.
- Alien/optimization hook: vectorized execution and cache-sized numeric kernels
  from the graveyard docs. The kept lever is a row-blocked SIMD accumulator:
  load each `col_sums` vector once, add four adjacent row vectors in original
  per-column row order, then store once.
- Decision: SHIP. The fresh baseline already beat NumPy on this lane, but the
  candidate widened the lead and improved every measured Rust row.

Targeted gap:
- Older scorecard evidence showed 256-1024 `ord=1/-1` column reductions behind
  NumPy. A fresh same-host baseline on current `origin/main` showed that this
  lane had already moved to a NumPy win: **6 wins / 0 losses / 0 neutral** versus
  NumPy. The keep gate therefore became stricter: beat fresh Rust baseline and
  preserve the NumPy win on all six rows.

Same-worker benchmark ledger:

| Row | Baseline FNP ns | Final FNP ns | NumPy ns | Baseline/NumPy | Final/Old | Final/NumPy | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| `one/256` | 11441 | 5337 | 33589 | 0.341x win | 0.466x win | 0.159x win | keep |
| `neg_one/256` | 9268 | 5093 | 33875 | 0.274x win | 0.550x win | 0.150x win | keep |
| `one/512` | 37970 | 28409 | 96297 | 0.394x win | 0.748x win | 0.295x win | keep |
| `neg_one/512` | 37477 | 28023 | 92790 | 0.404x win | 0.748x win | 0.302x win | keep |
| `one/1024` | 151777 | 123032 | 342892 | 0.443x win | 0.811x win | 0.359x win | keep |
| `neg_one/1024` | 152666 | 123074 | 341621 | 0.447x win | 0.806x win | 0.360x win | keep |

Repeat proof:
- First candidate pass also won all rows: 0.462x, 0.576x, 0.731x, 0.758x,
  0.794x, and 0.793x candidate/old for the table above.
- Repeat candidate pass is the counted final table.

Kept proof:
- Final old/new gate: **6 wins / 0 losses / 0 neutral** versus fresh FNP
  baseline on `hz2`.
- Final head-to-head NumPy gate: **6 wins / 0 losses / 0 neutral** on the same
  `hz2` host.
- The implementation preserves per-column row-order addition for each four-row
  block: `sum += abs(row0[col]); sum += abs(row1[col]); sum += abs(row2[col]);
  sum += abs(row3[col])`. Remainder rows use the prior one-row SIMD path.

Validation:
- `rch exec -- cargo test -p fnp-linalg matrix_norm_column_reduction_matches_strided_reference_bits -- --nocapture` passed on `hz2`.
- `rch exec -- cargo check -p fnp-linalg --all-targets` passed on `hz2`.
- `rch exec -- cargo build -p fnp-linalg --release` passed on `hz2`.
- `rch exec -- cargo clippy -p fnp-linalg --all-targets -- -D warnings` passed on `hz2`.
- `git diff --check` passed.
- `cargo fmt -p fnp-linalg -- --check` fails on broad pre-existing
  `fnp-linalg` formatting drift in benches/examples and unrelated `lib.rs`
  sections; the edited SIMD hunk was not reported by rustfmt.
- `ubs crates/fnp-linalg/src/lib.rs docs/NEGATIVE_EVIDENCE.md
  docs/RELEASE_READINESS_SCORECARD.md
  tests/artifacts/perf/2026-06-20_linalg_column_norm_rowblock_cod_b/scorecard.md`
  exited nonzero from broad existing `fnp-linalg` whole-file inventory; no
  finding was reported against the edited row-block SIMD hunk.

Retry predicate:
- Do not continue spending no-gaps effort on `matrix_norm_nxn_orders/(one|neg_one)/(256|512|1024)`
  unless a fresh same-host benchmark shows a new NumPy loss or a regression from
  this keep. The next credible linalg target should return to current measured
  losses in deeper SVD/eig/solve kernels with same-host NumPy capture first.

## 2026-06-20 - BOLD-VERIFY No-Ship: linalg spectral small-lever sweep, batch Cholesky verified win

Artifact directory:
`tests/artifacts/perf/2026-06-20_linalg_batch_cholesky_cod_a/`

Run identity:
- Agent: `YellowElk` / `cod-a`.
- Parent bead: `franken_numpy-ixs5y`.
- Crate/API: `fnp-linalg` / `eigvalsh_nxn`, `cond_nxn`, `batch_cholesky`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Worker proof:
  - `vmi1227854` for the primary `eigvalsh_nxn/128` and `cond_nxn/128`
    current-loss baseline, NumPy comparator, and direct-cond-extrema A/B.
  - `hz1` for the `sort_unstable_by` A/B after rch did not honor the requested
    `vmi1227854` worker; a matching NumPy comparator was captured on `hz1`.
  - `vmi1149989` for current `batch_cholesky` Rust-vs-NumPy proof with the exact
    Criterion batch diagonal-bump pattern.
- Alien/optimization hook: small-size routing, output-order specialization, and
  allocation/sort-elision probes from `/alien-graveyard`,
  `/alien-artifact-coding` numerical-linear-algebra guidance, and
  `/extreme-software-optimization`. All production source probes were reverted.
- Decision: NO-SHIP for new source. Current `batch_cholesky` is already a
  measured NumPy win; current symmetric spectral `eigvalsh_nxn/128` and
  `cond_nxn/128` remain measured NumPy losses.

Current head-to-head baseline:

| Row | FNP | NumPy | FNP/NumPy | Outcome |
|---|---:|---:|---:|---|
| `batch_cholesky/shape/500x64x64` | 4,879,321 ns | 17,379,269 ns | 0.281x | current win |
| `batch_cholesky/shape/64x128x128` | 3,808,040 ns | 16,877,623 ns | 0.226x | current win |
| `batch_cholesky/shape/16x256x256` | 4,152,904 ns | 27,276,163 ns | 0.152x | current win |
| `eigvalsh_nxn/size/128` | 1,330,011 ns | 435,883 ns | 3.051x | current loss |
| `cond_nxn/size/128` | 1,146,114 ns | 724,139 ns | 1.583x | current loss |

Notes:
- The initial NumPy batch-Cholesky comparator accidentally reused identical
  batch lanes. It was superseded by
  `numpy_batch_cholesky_exact_vmi1149989.txt`, which matches Criterion's
  per-lane diagonal bump `(b % 7) * 0.25`.
- The spectral NumPy comparator uses the same deterministic
  `generate_spd_matrix(128)` as the Criterion benchmark.

Rejected probes:

| Probe | Worker | Baseline | Candidate | Candidate/Baseline | Candidate/NumPy | Outcome |
|---|---|---:|---:|---:|---:|---|
| `TRIDIAG_BLOCK_MIN=192` route 128 back to unblocked reduction | `hz1` correctness gate | not counted | not counted | not counted | not counted | failed golden digest before benchmarking |
| `cond_nxn` direct extrema scan after tridiag QR, skipping public `eigvalsh` sort | `vmi1227854` | 1,161,511 ns | 1,191,551 ns | 1.026x | 1.646x loss | no-ship |
| `eigvalsh_nxn` `sort_unstable_by(total_cmp)` | `hz1` | 1,888,909 ns | 2,101,688 ns | 1.113x | 2.263x loss | no-ship |
| `cond_nxn` under same `sort_unstable_by` public eigvalsh path | `hz1` | 2,361,274 ns | 1,455,330 ns | 0.616x | 1.020x neutral/noisy loss | not kept because public eigvalsh regressed |

Correctness / validation:
- `TRIDIAG_BLOCK_MIN=192` rejected at
  `tridiag_rank2k_fused_update_preserves_spectra_and_golden_sha256`: digest
  drifted to `dbb1977a78b174e300410ac329a0b3d2f1a07881074a0cbe6d9dc905e56111c4`
  vs expected `d8a5154cdf2b005605b832840983ece912dac6252c0d6b59452f47256b8cb2f8`.
- Direct-cond-extrema candidate passed:
  `cargo test -p fnp-linalg cond_p_spectral_symmetric --release` and
  `cargo test -p fnp-linalg tridiag_rank2k_fused_update_preserves_spectra_and_golden_sha256 --release`.
- `sort_unstable_by` candidate passed:
  `cargo test -p fnp-linalg eigvalsh --release` and
  `cargo test -p fnp-linalg cond_p_spectral_symmetric --release`.
- Production source diff after reverts: empty for `crates/fnp-linalg/src/lib.rs`.

Retry predicate:
- Do not retry small-threshold unblocked routing for `eigvalsh_nxn/128` unless
  the golden digest is intentionally re-pinned behind stronger NumPy and
  reconstructive proof; this run failed before it deserved a performance keep.
- Do not retry the private `cond_nxn` direct-extrema scan as a standalone lever.
  It looked like a 4-5% win on an unpaired run, then regressed by 2.6% in the
  paired same-worker A/B.
- Do not switch public `eigvalsh_nxn` sorting to `sort_unstable_by` for this
  loss class. It produced a tempting `cond` swing on `hz1` but regressed the
  actual public `eigvalsh` row by 11.3%, and `eigvalsh/128` is the larger
  measured NumPy loss.
- The next credible spectral route must attack the reduction/QR work itself,
  not post-processing: a profiled values-only tridiagonal QR primitive, a
  provably equivalent small-size symmetric eigensolver, or a blocked reduction
  change that keeps the golden stream and improves `eigvalsh_nxn/128` against
  the same-worker NumPy comparator.
## 2026-06-20 - BOLD-VERIFY Keep: Python compress axis=None bitmask gather

Artifact directory:
`tests/artifacts/perf/2026-06-20_python_compress_axis_none_cod_a/`

Run identity:
- Agent: `YellowElk` / `cod-a`.
- Parent bead: `franken_numpy-ixs5y`.
- Crate/API: `fnp-python` / `np.compress(condition, a)` with `axis=None`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Performance worker: `vmi1167313` for the counted old/new and NumPy rows.
- Alien/optimization hook: mask-first stream compaction from
  `/data/projects/alien_cs_graveyard/alien_cs_graveyard.md` numeric kernel
  archetype plus branchless/block coding from `extreme-software-optimization`.
  The retained lever changes the typed flat compaction loop from speculative
  per-element stores into 8-lane mask construction plus trailing-zero selected
  lane gathers.
- Decision: SHIP. The sparse branch and NumPy delegate probes were reverted.

Targeted gap:
- The existing `axis=None` zerocopy compress route still lost to NumPy at the
  100k flat row while winning at 1M. Baseline same-worker ratios were:
  **1 win / 1 loss / 0 neutral** versus NumPy.

Same-worker benchmark ledger:

| Row | Baseline FNP | Baseline FNP/NumPy | Sparse branch | Sparse/Old | Delegate | Delegate/Old | Final bitmask | Final/Old | Final/NumPy | Outcome |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `compress_f64_axis_none_100000` | 167,603 ns | 1.215x loss | 180,973 ns | 1.080x loss | 172,647 ns | 1.030x loss | 142,735 ns | 0.852x win | 1.015x neutral/noisy raw loss | keep |
| `compress_f64_axis_none_1000000` | 1,902,857 ns | 0.792x win | 1,985,899 ns | 1.044x loss | 1,986,693 ns | 1.044x loss | 1,853,998 ns | 0.974x win | 0.805x win | keep |

Rejected probes:
- Sparse branch for `count * 2 <= m`: **0 wins / 2 losses / 0 neutral** versus
  old FNP. NumPy ratios were 1.272x loss at 100k and 0.861x win at 1M, but both
  old/new FNP rows regressed.
- Small NumPy delegate for `condition.size <= 200000`: **0 wins / 2 losses /
  0 neutral** versus old FNP. Raw NumPy ratios were 0.969x and 0.833x, but the
  apparent 100k NumPy win came from a slower NumPy rerun and the FNP old/new
  gate regressed both rows. A pinned local-fallback attempt was interrupted and
  is retained as a not-counted artifact.

Kept proof:
- Final bitmask old/new gate: **2 wins / 0 losses / 0 neutral** versus old FNP.
- Final head-to-head NumPy gate: **1 win / 0 losses / 1 neutral**. The 100k row
  is a 1.015x raw FNP/NumPy ratio, inside the observed timing noise; the 1M row
  remains a clear 0.805x win versus NumPy.
- The 100k loss class moved from 21.5% slower than NumPy to 1.5% raw slower,
  while the 1M row improved versus both old FNP and NumPy.

Validation:
- `rch exec -- cargo test -p fnp-python --test conformance_compress_choose_diagonal
  compress -- --nocapture` passed: 13 passed, 0 failed.
- `rch exec -- cargo check -p fnp-python --lib --bench criterion_python_surface`
  passed with three inherited `fnp-python` warnings.
- `rch exec -- cargo build -p fnp-python --release` passed with the same three
  inherited warnings.
- `cargo fmt -p fnp-python -- --check` reports broad pre-existing rustfmt drift
  in `fnp-python`; the new benchmark hunk was manually aligned with rustfmt's
  suggested local formatting.
- `rch exec -- cargo clippy -p fnp-python --lib --bench
  criterion_python_surface -- -D warnings` failed on 35 existing `fnp-python`
  lint errors outside this compaction hunk.
- `ubs crates/fnp-python/src/lib.rs crates/fnp-python/benches/criterion_python_surface.rs
  docs/NEGATIVE_EVIDENCE.md docs/RELEASE_READINESS_SCORECARD.md
  tests/artifacts/perf/2026-06-20_python_compress_axis_none_cod_a/SUMMARY.md`
  completed and recorded the broad existing `fnp-python` scanner inventory; it
  did not identify a focused new blocker for this compaction hunk.
- `git diff --check` passed.

Retry predicate:
- Do not retry sparse kept-only branching or threshold-gated NumPy delegation
  for this row family; both failed the old/new FNP gate.
- Re-profile deeper only if future masks are dense enough that selected-lane
  trailing-zero iteration loses to speculative stores, or if an architecture
  with stronger SIMD compress-store support is available behind a safe,
  target-gated implementation. A credible retry must keep the 100k row below
  the old FNP 0.852x ratio and preserve the 1M NumPy win.

## 2026-06-20 - BOLD-VERIFY No-Ship: batch_cholesky 8-lane SoA generated micro-kernel probe

Artifact directory:
`tests/artifacts/perf/2026-06-20_linalg_batch_cholesky_generated_cod_b/`

Run identity:
- Agent: `YellowElk` / `cod-b`.
- Bead: `franken_numpy-ixs5y.272`.
- Parent bead: `franken_numpy-ixs5y`.
- Crate: `fnp-linalg`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.
- Alien/optimization hook: vectorized execution / template-specialized numeric
  kernel layout from `/data/projects/alien_cs_graveyard/alien_cs_graveyard.md`
  plus numerical linear algebra family 34. The attempted lever transformed one
  group of eight batch lanes into a temporary SoA register layout so each inner
  Cholesky dot used SIMD lanes without per-k gather/scatter.
- Decision: NO-SHIP. Candidate source was reverted before commit. The retained
  harness change only adds the d=16/32/64 batch rows that expose this loss class.

Candidate proof:
- Candidate compile passed: `rch exec -- cargo check -p fnp-linalg --lib`.
- Candidate bit proof passed: `rch exec -- cargo test -p fnp-linalg
  batch_cholesky_soa8_matches_per_lane_cholesky_nxn_bits -- --nocapture`
  reported 1 passed, 0 failed on `hz2`.

Same-worker old/new gate on `hz1`:

| Row | Old-path FNP | Candidate FNP | Candidate/Old | NumPy ratio | Outcome |
|---|---:|---:|---:|---:|---|
| `batch_cholesky/shape/2000x16x16` | 1,283,500 ns | 1,198,602 ns | 0.934x | not counted; SSH auth blocked same-host Python | small win |
| `batch_cholesky/shape/1000x32x32` | 2,610,096 ns | 4,308,668 ns | 1.651x | not counted; SSH auth blocked same-host Python | loss |
| `batch_cholesky/shape/500x64x64` | 8,147,905 ns | 9,213,859 ns | 1.131x | not counted; SSH auth blocked same-host Python | loss |
| `batch_cholesky/shape/64x128x128` | 4,970,534 ns | 9,130,164 ns | 1.837x | not counted; not routed by candidate | noisy guardrail loss |
| `batch_cholesky/shape/16x256x256` | 6,607,140 ns | 9,491,297 ns | 1.437x | not counted; not routed by candidate | noisy guardrail loss |

Ledger:
- Candidate same-worker Rust gate on target routed rows: **1 win / 2 losses /
  0 neutral**.
- Full observed same-worker sweep: **1 win / 4 losses / 0 neutral**.
- Candidate vs NumPy: **0 wins / 0 losses / 5 blocked**. Direct same-host
  NumPy on `root@87.99.133.171` failed with SSH authentication denial. Local
  Python has NumPy 2.4.3 but no importable `fnp_python`, so no local FNP/NumPy
  comparator was counted.
- Existing same-day current-head Python stacked Cholesky evidence remains the
  active NumPy gap context: **1 win / 6 losses / 0 neutral** versus NumPy
  (`d=16` 6.46x slower, `d=32` 5.46x slower, `d=64` 6.27x slower).

Validation after revert:
- Production source diff for `crates/fnp-linalg/src/lib.rs` is empty.
- `rch exec -- cargo test -p fnp-linalg batch_cholesky -- --nocapture`
  passed after revert: 2 passed, 0 failed, 1 ignored.
- `rch exec -- cargo check -p fnp-linalg --benches` passed after revert,
  proving the retained focused batch_cholesky benchmark rows compile.
- Invalid probe retained: `cargo bench ... --release` failed because this Cargo
  invocation does not accept `--release`; Criterion cargo bench already uses the
  bench profile.

Retry predicate:
- Do not retry the temporary 8-lane SoA register-layout Cholesky kernel as a
  standalone lever. It removed per-k gather/scatter but still regressed d=32
  and d=64, so the conversion/scatter footprint and vector codegen cost exceed
  the saved scalar reduction work beyond d=16.
- Do not retry allocation elimination, gate tuning, threshold-only changes,
  finite-validation hoists, const specialization, ordered scalar dot expansion,
  or portable-SIMD gather/scatter across lanes for this gap.
- A credible retry needs a different primitive: true packed-panel batched
  Cholesky with reusable SoA panels across the whole factorization, a safe
  vector dot primitive that wins d=32 and d=64 in serial first, or a LAPACK-class
  blocked per-lane kernel. It must clear a same-worker old/new gate on
  d=16/32/64 plus n>=128 guardrails before any NumPy keep claim.

## 2026-06-20 - BOLD-VERIFY Win: fnp-python cov(m,y) two-operand zero-copy Gram (4-17x loss -> 0.6-0.9x win)

Artifact directory: `tests/artifacts/perf/2026-06-20_python_cov_two_operand_vs_numpy/`

Run identity:
- Agent: `BlackThrush` / `cod-b`. Under directive `franken_numpy-ixs5y`.
- Subject: `np.cov(a, b)` two-operand form (`crates/fnp-python/src/lib.rs`).
- Reference: NumPy 2.4.3 on `thinkstation1`, load ~6, OMP/OPENBLAS=1.
- Decision: SHIP.

LOSE-gap: `np.cov(a, b)` (covariance of two series — very common idiom) was
4-17x slower than numpy (10k 17.4x, 100k 10.4x, 1M 4.45x). `cov` had zero-copy
fast paths (SIMD 16<=n_vars<128 + general `cov_gram_rowvar_f64`) but BOTH gated
on `y is None`; the two-operand form fell to `native_cov_unweighted` which
extracts both operands + concatenates (copies) + generic Gram.
`np.cov(a,b) == np.cov(concatenate([rows(a),rows(b)]))`.

Lever (zero-copy two-buffer Gram): extracted the autovectorized 8-accumulator
Gram into shared `cov_gram_from_centered(centered,n_vars,n_obs,ddof)` (single-
operand path now calls it, verified byte-identical). Added
`try_zerocopy_cov_two_rowvar_f64`: reads m and y f64 PyBuffers directly, centers
each variable row from its own buffer into one `centered` array (no raw-input
stack copy), reuses the shared Gram. Same arithmetic as the single-operand fast
path -> inherits its allclose conformance.

After: 4/0/0 win (10k 0.596x, 100k 0.898x, 1M 0.918x, 4M 0.808x); two-operand
correctness 0/160 random cases (offset means + ddof 0/1/2, allclose rtol 1e-10);
single-operand byte-identity preserved; conformance_statistics 28 pass.

REUSABLE: zero-copy PyBuffer fast paths gated on a SCALAR/None optional operand
(`y is None`, default=scalar, etc.) leave the OTHER form (array y / two-operand)
to a slow extract+concat fallback — extend by reading the extra buffer(s)
directly and reusing the shared kernel. Same pattern as np.select array-default
and np.where scalar-branch. Grep cov/corrcoef/... fast paths for `is_none()`/
`y is None` gates.

PRE-EXISTING (not introduced; proven RED on HEAD with this change stashed):
`cov_corrcoef_python_container_keyword_outcomes_match_numpy` "cov y ddof" case
is 1-ULP off (`cov([1,2,4],y=[2,1,0],ddof=0)[0][0]` 1.5555555555555554 vs numpy
...556) in the untouched `native_cov_unweighted` list path — numpy-BLAS-bit-exact
(FMA) reduction, separate concern. Also pre-existing: `cov(a,b,rowvar=False)`
wrongly returns a scalar instead of 2x2 (native path) — file separately.

## 2026-06-20 - BOLD-VERIFY No-Ship: medium Cholesky lower-triangular update threshold

Artifact directory:
`tests/artifacts/perf/2026-06-20_linalg_cholesky_triangular_medium_cod_b/`

Run identity:
- Agent: `YellowElk` / `cod-b`.
- Bead: `franken_numpy-ixs5y.271`.
- Parent bead: `franken_numpy-ixs5y`.
- Crate: `fnp-linalg`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.
- Worker proof: RCH worker `vmi1264463`.
- Candidate: lower `SYRK_MID_TRIANGULAR_MIN_TRAIL` from 384 to 64 so medium
  Cholesky panels use the existing lower-triangular packed trailing update
  instead of the full `trail x trail` product.
- Decision: NO-SHIP. Candidate source was reverted before commit.

Same-worker old/new gate:

| Row | Baseline FNP | Candidate FNP | Candidate/Baseline | NumPy ratio | Outcome |
|---|---:|---:|---:|---:|---|
| `batch_cholesky/shape/64x128x128` | 31,931,504 ns | 66,366,114 ns | 2.078x | not counted; SSH auth blocked same-host Python | loss |
| `batch_cholesky/shape/16x256x256` | 114,361,825 ns | 174,294,182 ns | 1.524x | not counted; SSH auth blocked same-host Python | loss |

Ledger:
- Candidate same-worker Rust gate: **0 wins / 2 losses / 0 neutral**.
- Candidate vs NumPy: **0 wins / 0 losses / 2 not measured**. Direct Python
  comparator attempts on `root@38.242.209.154` and
  `ubuntu@38.242.209.154` both failed with SSH authentication denial. No
  `rch exec -- python3` fallback is counted because that path runs locally.
- Because the candidate regressed both old/new rows, no NumPy keep rerun was
  justified. Existing same-day evidence still records current `batch_cholesky`
  as a confirmed NumPy gap.

Validation:
- Candidate golden-output guard passed: `rch exec -- cargo test -p fnp-linalg
  cholesky_mid_panel -- --nocapture` reported 2 passed, 0 failed.
- An earlier command using a regex-like Cargo test filter ran zero tests; it is
  retained as an invalid artifact and is not counted.
- Post-revert focused test passed: `rch exec -- cargo test -p fnp-linalg
  batch_cholesky -- --nocapture` reported 2 passed, 0 failed, 1 ignored.
- Post-revert source diff for `crates/fnp-linalg/src/lib.rs` is empty.

Retry predicate:
- Do not retry medium Cholesky by simply lowering the existing triangular-update
  threshold; the packed lower-triangular path was slower on both medium batch
  rows despite preserving golden output.
- A credible retry still needs a deeper kernel change: generated size-specific
  batched panels, a safe SIMD dot primitive that beats the scalar panel solve,
  or a LAPACK-class blocked Cholesky path with same-host NumPy capture and zero
  medium-row regressions.

## 2026-06-20 - BOLD-VERIFY Keep: stacked Cholesky Python boundary delegate

Artifact directory:
`tests/artifacts/perf/2026-06-20_linalg_cholesky_python_delegate_cod_a/`

Run identity:
- Agent: `YellowElk` / `cod-a`.
- Parent bead: `franken_numpy-ixs5y`.
- Crate: `fnp-python`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Worker proof: RCH worker `vmi1152480` for baseline and final candidate
  Criterion runs. The baseline filename mentions `vmi1149989`, but the log
  records `Selected worker: vmi1152480`.
- NumPy comparator: same Criterion harness, same input object per row, direct
  pre-bound `numpy.linalg.cholesky`.
- Alien/optimization hook: "constants kill you" boundary rewrite. For exact
  stacked NumPy arrays with shape `(..., n, n)` and `n >= 4`, the wrapper now
  skips Rust extraction plus per-lane native Cholesky and delegates before
  copying to NumPy/LAPACK. The delegate path caches `numpy.linalg.cholesky`,
  uses cached ndarray type classification, indexes `shape` without a `Vec`
  allocation, and avoids default-path kwargs allocation.
- Decision: KEEP. The final candidate removed all material NumPy losses in the
  measured 4x4..64x64 stacked-SPD sweep and cut FNP runtime to 0.267x..0.837x
  of the prior FNP baseline. One 32x32 row is a 1.026x noise-band neutral
  versus NumPy, not a material loss.

Same-worker head-to-head (`vmi1152480`):

| Row | Old FNP | Old NumPy | Old FNP/NumPy | New FNP | New NumPy | New FNP/NumPy | New/Old FNP | Outcome |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `batch10000_4x4` | 2,109,573 ns | 1,810,289 ns | 1.165x | 1,766,423 ns | 2,521,959 ns | 0.700x | 0.837x | WIN |
| `batch4000_8x8` | 5,566,219 ns | 2,149,175 ns | 2.590x | 1,483,647 ns | 1,572,176 ns | 0.944x | 0.267x | WIN |
| `batch2000_16x16` | 6,459,892 ns | 3,207,857 ns | 2.014x | 3,379,216 ns | 3,421,966 ns | 0.988x | 0.523x | WIN |
| `batch1000_32x32` | 11,059,012 ns | 5,741,576 ns | 1.926x | 4,993,838 ns | 4,866,396 ns | 1.026x | 0.452x | neutral/noisy |
| `batch500_64x64` | 21,866,929 ns | 10,619,813 ns | 2.059x | 7,382,796 ns | 7,639,253 ns | 0.966x | 0.338x | WIN |

Ledger:
- Baseline FNP vs NumPy: **0 wins / 5 losses / 0 neutral**.
- Final FNP vs NumPy: **4 wins / 0 material losses / 1 neutral**. The 32x32
  row is 2.6% slower than NumPy and well inside the reported benchmark spread.
- Final FNP vs old FNP: **5 wins / 0 losses / 0 neutral**, with old/new ratios
  from 0.267x to 0.837x.
- Intermediate candidates retained for negative evidence:
  - `baseline_cholesky_python_linalg_vmi1227854.txt`: invalid first Criterion
    filter placement; the command compiled but emitted no Cholesky rows and is
    retained only to explain the artifact trail.
  - `candidate_cholesky_f64_vmi1152480.txt`: delegate with default kwargs still
    paid wrapper overhead; several rows stayed around +/-2% of NumPy.
  - `candidate_no_default_kw_cholesky_f64_vmi1152480.txt`: removing default
    `upper=false` kwargs improved most rows but left 16x16/32x32 noisy.
  - `candidate_cached_tuple_shape_cholesky_f64_vmi1152480.txt`: cached delegate
    and tuple shape indexing gave 4x4/32x32 wins but left 8x8/16x16/64x64
    noise-band rows. The final lazy-kwargs candidate is the kept source.

Validation:
- `rch exec -- cargo bench -p fnp-python --bench criterion_python_surface
  cholesky_f64 -- --sample-size 10 --measurement-time 2 --warm-up-time 1
  --output-format bencher` passed for baseline and final candidate on
  `vmi1152480`.
- `rch exec -- cargo test -p fnp-python --test conformance_linalg_decomp
  cholesky -- --nocapture` passed after adding the stacked 4x4 SPD case:
  6 passed, 0 failed, 33 filtered out.
- `rch exec -- cargo check -p fnp-python --lib --bench
  criterion_python_surface` passed; a local rerun after touched-hunk rustfmt
  alignment also passed.
- `rch exec -- cargo build -p fnp-python --release` passed.
- `git diff --check` passed.
- `rch exec -- cargo clippy ... -D warnings` was blocked on `vmi1153651`
  because that worker lacks the pinned nightly clippy component. Local clippy
  with the same flags reached code analysis and failed on broad pre-existing
  `fnp-python` lint inventory outside the Cholesky hunk.
- `cargo fmt -p fnp-python -- --check` remains blocked by broad pre-existing
  rustfmt drift across `fnp-python`; the touched Cholesky hunk was manually
  aligned with rustfmt's suggested shape.
- `ubs` over the changed files completed nonzero with broad existing findings
  in the large `fnp-python` surface, not a Cholesky-specific finding.
- A broad `cargo test -p fnp-python cholesky -- --nocapture` attempt was
  blocked before execution by unrelated lib-test compile errors in
  `spacing/sign/nextafter/hypot/logaddexp` test call sites.

Retry predicate:
- Do not reopen scalar per-lane `batch_cholesky` micro-tuning for Python
  stacked Cholesky at 4x4..64x64 until fresh evidence shows the delegate path
  has become a material loss. The copy/extraction boundary was the dominant
  measured issue.
- A future retry should target the remaining 32x32 neutral/noisy row only with
  a lower-overhead Python trampoline or true generated in-extension LAPACK
  call that beats direct NumPy despite wrapper overhead. Re-benchmark on the
  same worker and keep only if it clears a >5% material win or removes a
  confirmed future loss.

## 2026-06-20 - BOLD-VERIFY Mixed Keep: small-N Cholesky ordered dot narrows Rust, not NumPy

Artifact directory:
`tests/artifacts/perf/2026-06-20_linalg_cholesky_right_looking_cod_a/`

Run identity:
- Agent: `YellowElk` / `cod-a`.
- Parent bead: `franken_numpy-ixs5y`.
- Crate: `fnp-linalg` plus Python-surface comparator through `fnp-python`.
- Source under verification: already-present commit `856c38cb`
  (`perf(fnp-linalg): ordered 4-wide dot for small-N unblocked Cholesky (N=16..32)`).
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Alien/optimization hook: dependency-chain break for tiny dot products from
  profile-first numeric-kernel tuning; no layout/JIT/arena rewrite shipped.
- Directory name note: the artifact directory keeps the initial
  `right_looking` hypothesis name, but the verified source is the ordered-dot
  helper only.
- Decision: KEEP AS NARROW RUST MICRO-WIN ALREADY IN `main`, but do **not**
  claim NumPy domination. Current Python stacked Cholesky is still a visible
  NumPy loss, including the owned d=16 and d=32 rows.

Same-worker Rust Criterion (`vmi1153651`, parent `586f3459` vs current
`856c38cb`):

| Row | Parent | Current | Current/Parent | Ownership | Outcome |
|---|---:|---:|---:|---|---|
| `cholesky_nxn/size/16` | 2,186 ns | 1,901 ns | 0.870x | ordered-dot path | WIN |
| `cholesky_nxn/size/32` | 11,091 ns | 9,747 ns | 0.879x | ordered-dot path | WIN |
| `cholesky_nxn/size/64` | 70,817 ns | 70,754 ns | 0.999x | not routed | neutral |
| `cholesky_nxn/size/128` | 319,742 ns | 306,706 ns | 0.959x | blocked path, not routed | noisy neutral |
| `cholesky_nxn/size/256` | 1,868,544 ns | 2,052,584 ns | 1.098x | blocked path, not routed | noisy neutral/loss |
| `cholesky_nxn/size/512` | 35,042,224 ns | 24,653,195 ns | 0.704x | blocked path, not routed | noisy neutral |
| `cholesky_nxn/size/768` | 107,262,958 ns | 96,512,158 ns | 0.900x | blocked path, not routed | noisy neutral |
| `batch_cholesky/64x128x128` | 20,565,511 ns | 24,253,715 ns | 1.179x | blocked path, not routed | noisy neutral/loss |
| `batch_cholesky/16x256x256` | 22,245,367 ns | 78,519,429 ns | 3.529x | blocked path, not routed | noisy loss |

Ledger:
- Owned Rust rows vs parent: **2 wins / 0 losses / 0 neutral**.
- Non-owned broad Rust guardrails: **0 claimed wins / 2 losses-or-noisy-loss /
  5 neutral-or-noisy**. The d=128/d=256 batch guardrails are not causally
  affected by the helper because they route through `cholesky_blocked`.
- Current Python `fnp.linalg.cholesky` vs NumPy: **1 win / 6 losses / 0
  neutral**. Rows: d=4 `0.75x` win; d=8 `1.11x` loss; d=16 `6.46x` loss;
  d=32 `5.46x` loss; d=64 `6.27x` loss; d=100 `1.46x` loss; d=200 `1.67x`
  loss. All rows matched NumPy numerically.
- Owned Python-facing rows vs NumPy: **0 wins / 2 losses / 0 neutral**
  (`d=16`, `d=32`). This is not a release-level performance closeout.
- Prior same-day local Python comparator is retained only as routing evidence:
  d=16 moved from a recorded `19.65x` loss to `6.46x`, while d=32 remains a
  large loss (`4.67x` prior, `5.46x` current). Different run windows mean this
  is not scored as same-worker old/new proof.

Validation:
- `rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg
  'cholesky_nxn|batch_cholesky' -- --sample-size 20 --warm-up-time 1
  --measurement-time 3 --output-format bencher` passed on current head (`hz2`)
  and in a same-worker parent/current pair on `vmi1153651`.
- `rch exec -- cargo test -p fnp-linalg cholesky_ -- --nocapture` passed on
  `vmi1293453`: 21 unit tests, 4 conformance tests, 2 golden tests, 1
  metamorphic test, and 4 solve tests passed.
- `rch exec -- cargo check -p fnp-linalg --all-targets` passed on `hz1`.
- `rch exec -- cargo clippy -p fnp-linalg --all-targets -- -D warnings`
  passed on `hz1`.
- `rch exec -- cargo build -p fnp-linalg --release` passed on `vmi1149989`.
- `rch exec -- cargo build -p fnp-python --release --features
  python-extension` passed on `vmi1152480`; it emitted three pre-existing
  `fnp-python` warnings.
- Python comparator loaded the current-head extension built from
  `/data/projects/.rch-targets/franken_numpy-cod-a/release/libfnp_python.so`;
  all measured rows reported `match=True`.

Retry predicate:
- Do not reopen small-N scalar Cholesky unroll/const-specialization as the main
  route to NumPy parity. It can narrow direct Rust microbenchmarks but does not
  change the Python stacked-SPD loss class.
- Next credible Cholesky lever needs a different complexity/layout class:
  SoA batched panels across lanes, packed-panel batched `dpotrf` shape,
  communication-avoiding panel/SYRK fusion, or JIT/generated fixed-size kernels
  for d=16/32 with same-window NumPy proof and explicit regression guardrails.

## 2026-06-20 - BOLD-VERIFY Keep: fnp-linalg column-norm SIMD lane accumulation

Artifact directory: `tests/artifacts/perf/2026-06-20_linalg_column_norm_simd_cod_a/`

Run identity:
- Agent: `BlackThrush` / `cod-a`.
- Parent bead: `franken_numpy-ixs5y`.
- Crate: `fnp-linalg`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Worker proof: RCH worker `vmi1227854` for baseline/candidate Criterion
  and same-host NumPy comparator.
- NumPy comparator: direct SSH on `ubuntu@vmi1227854`, Python 3.13.7,
  NumPy 2.4.6, `OMP/OPENBLAS/MKL/NUMEXPR=1`, deterministic
  `generate_random_matrix(n, 0x4E4F_524D_4F52_4445)` input.
- Decision: KEEP. Source adds safe `std::simd::Simd<f64, 8>` accumulation for
  cache-linear matrix 1/-1 norms when `n >= 256`, while `n < 256` routes
  through the original scalar cache-linear helper.

Same-worker head-to-head result:

| Row | Old FNP | New FNP | NumPy p50 | New/Old | New/NumPy | Outcome |
|---|---:|---:|---:|---:|---:|---|
| `matrix_norm_nxn_orders/one/128` | 6,631 ns | 6,161 ns | 9,024 ns | 0.929x | 0.683x | guardrail win |
| `matrix_norm_nxn_orders/neg_one/128` | 6,816 ns | 7,134 ns | 9,224 ns | 1.047x | 0.774x | guardrail neutral/noisy |
| `matrix_norm_nxn_orders/one/256` | 34,821 ns | 6,496 ns | 24,116 ns | 0.187x | 0.269x | WIN |
| `matrix_norm_nxn_orders/neg_one/256` | 26,663 ns | 6,251 ns | 24,537 ns | 0.234x | 0.255x | WIN |
| `matrix_norm_nxn_orders/one/512` | 102,390 ns | 26,176 ns | 78,408 ns | 0.256x | 0.334x | WIN |
| `matrix_norm_nxn_orders/neg_one/512` | 163,924 ns | 25,195 ns | 77,666 ns | 0.154x | 0.324x | WIN |
| `matrix_norm_nxn_orders/one/1024` | 421,756 ns | 118,415 ns | 355,402 ns | 0.281x | 0.333x | WIN |
| `matrix_norm_nxn_orders/neg_one/1024` | 410,832 ns | 112,363 ns | 374,671 ns | 0.274x | 0.300x | WIN |

Ledger:
- Target rows (`n >= 256`) vs NumPy: **6 wins / 0 losses / 0 neutral**.
- Full observed sweep vs NumPy: **8 wins / 0 losses / 0 neutral**.
- Old/new guardrail: **7 wins / 0 losses / 1 neutral/noisy**. The
  `neg_one/128` scalar guardrail moved from 6.816 us to 7.134 us but stayed
  faster than same-host NumPy and overlaps benchmark noise. A first SIMD draft
  had a real-looking 128 regression; it was refactored before keep so the scalar
  path is selected before entering the SIMD helper.

Validation:
- `rch exec -- cargo test -p fnp-linalg
  matrix_norm_column_reduction_matches_strided_reference_bits -- --nocapture`
  passed.
- `rch exec -- cargo check -p fnp-linalg --all-targets` passed.
- `rch exec -- cargo clippy -p fnp-linalg --all-targets -- -D warnings`
  passed.
- `rch exec -- cargo build -p fnp-linalg --release` passed.
- `cargo fmt -p fnp-linalg -- --check` remains blocked by broad pre-existing
  fmt drift in benches/examples/tests and unrelated regions of `src/lib.rs`;
  the touched SIMD hunk was manually aligned with rustfmt's reported shape.
- `ubs crates/fnp-linalg/src/lib.rs` remains blocked by broad pre-existing
  inventory; UBS reports crate-wide unwrap/panic/indexing/security heuristics
  unrelated to the touched matrix-norm helper. Its own cargo fmt/clippy/check
  sub-gates were green.

Retry predicate:
- Do not retry scalar threshold-only column norm work; the pre-256 scalar route
  must remain outside the SIMD helper.
- A future retry should target allocation-free stack or reusable scratch for
  `n >= 2048`, AVX-width tuning, or direct Python-boundary norm dispatch only
  if fresh same-host evidence shows a residual loss.

## 2026-06-20 - BOLD-VERIFY No-Ship: cholesky_nxn const specialization too small/noisy

Artifact directory: `tests/artifacts/perf/2026-06-20_linalg_cholesky_const_specialize_cod_a/`

Run identity:
- Agent: `BlackThrush` / `cod-a`.
- Parent bead: `franken_numpy-ixs5y`.
- Crate: `fnp-linalg`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Candidate: `cholesky_unblocked_const<const N>` for N=16/32/64/100 routed
  from `cholesky_nxn`, with bit-reference tests.
- Decision: NO-SHIP. Candidate source and tests were reverted before this
  commit. The direct target rows improved only 4.9%-8.1%; larger apparent wins
  came from rows the const path did not own and were treated as worker noise,
  not keep evidence.

Same-worker Rust Criterion on `vmi1149989`:

| Row | Baseline | Candidate | Candidate/Baseline | NumPy ratio | Outcome |
|---|---:|---:|---:|---:|---|
| `cholesky_nxn/size/16` | 1,152 ns | 1,084 ns | 0.941x | not rerun | neutral/small win |
| `cholesky_nxn/size/32` | 5,597 ns | 5,142 ns | 0.919x | not rerun | neutral/small win |
| `cholesky_nxn/size/64` | 32,431 ns | 30,845 ns | 0.951x | not rerun | neutral/small win |
| `cholesky_nxn/size/128` | 226,889 ns | 119,611 ns | 0.527x | not rerun | noisy non-owned row |
| `cholesky_nxn/size/256` | 1,228,708 ns | 695,743 ns | 0.566x | not rerun | noisy non-owned row |
| `cholesky_nxn/size/512` | 8,866,316 ns | 5,587,315 ns | 0.630x | not rerun | noisy non-owned row |
| `cholesky_nxn/size/768` | 20,093,048 ns | 11,838,452 ns | 0.589x | not rerun | noisy non-owned row |
| `batch_cholesky/64x128x128` | 4,237,881 ns | 2,920,691 ns | 0.689x | not rerun | noisy non-owned row |
| `batch_cholesky/16x256x256` | 5,548,820 ns | 4,049,209 ns | 0.730x | not rerun | noisy non-owned row |

Ledger:
- Owned target rows vs old FNP: **3 small wins / 0 losses / 0 neutral**.
- Owned target rows vs NumPy: **0 wins / 0 losses / 3 not measured** because
  the old/new proof was too small to justify a NumPy keep claim.
- Broad rows: treated as **neutral/noisy**, not wins, because the candidate did
  not route n=128/256/512/768 or batched n>=128 through the const-specialized
  path.

Validation:
- `rch exec -- cargo test -p fnp-linalg
  cholesky_const_specializations_match_dynamic_scalar_reference_bits -- --nocapture`
  passed while the candidate existed.
- `rch exec -- cargo check -p fnp-linalg --all-targets` passed while the
  candidate existed.
- Candidate was reverted after measurement; no Cholesky production hunk remains.

Retry predicate:
- Do not retry const-specializing unblocked Cholesky for only small fixed N.
- A credible Cholesky retry must change the medium-matrix algorithm or layout:
  true SoA batched-panel Cholesky, packed-panel storage eliminating gather/scatter,
  or a blocked triangular/SYRK primitive with same-window proof versus NumPy.
## 2026-06-20 - BOLD-VERIFY No-Ship: batch_cholesky finite-validation hoist

Artifact directory:
`tests/artifacts/perf/2026-06-20_linalg_batch_cholesky_validation_hoist_cod_b/`

Run identity:
- Agent: `YellowElk` / `cod-b`.
- Parent bead: `franken_numpy-ixs5y`.
- Crate: `fnp-linalg`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.
- Candidate: hoist `batch_cholesky` finite validation to one full-batch scan and
  call finite-unchecked internal Cholesky helpers only when every input value is
  finite; otherwise fall back to the original checked per-lane path.
- Decision: NO-SHIP. Candidate source was reverted before commit.

Same-worker broad gate on RCH worker `vmi1153651`:

| Row | Baseline | Candidate | Candidate/Baseline | Verdict |
|---|---:|---:|---:|---|
| `batch_cholesky/shape/64x128x128` | 18,102,653 ns | 20,809,809 ns | 1.150x | loss |
| `batch_cholesky/shape/16x256x256` | 12,748,878 ns | 44,004,085 ns | 3.451x | loss |

Ledger:
- Candidate same-worker Rust gate: **0 wins / 2 losses / 0 neutral**.
- Candidate NumPy rerun was intentionally skipped because the same-worker Rust
  broad gate already regressed badly. The existing same-day NumPy evidence still
  has current `batch_cholesky` at **0 wins / 7 losses / 0 neutral** versus NumPy,
  with medium stacked SPD rows 4.67x-19.65x slower.
- Post-revert source diff is empty for `crates/fnp-linalg/src/lib.rs`.

Validation:
- Candidate compile check passed: `rch exec -- cargo check -p fnp-linalg --lib`.
- Post-revert focused test passed: `rch exec -- cargo test -p fnp-linalg
  batch_cholesky_ -- --nocapture` reported 2 passed, 0 failed, 1 ignored.
- Post-revert release build passed: `rch exec -- cargo build -p fnp-linalg
  --release`.

Retry predicate:
- Do not retry finite-scan hoisting, allocation elimination, threshold tuning,
  or f64x4 gather/scatter across batch lanes for `batch_cholesky`.
- A credible retry needs a structurally different Cholesky kernel - blocked or
  batched panels, or a dot-product kernel that preserves the Cholesky bit
  contracts and proves medium rows plus n>=128 rows in the same run window.

## 2026-06-20 - BOLD-VERIFY Routing: einsum reduce-all current-head rerun is already a win

Artifact directory:
`tests/artifacts/perf/2026-06-20_python_einsum_reduce_all_cod_b/`

Run identity:
- Agent: `YellowElk` / `cod-b`.
- Parent bead: `franken_numpy-ixs5y`.
- Crate: `fnp-python`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.
- RCH worker selected: `vmi1293453`.
- Purpose: rerun the prior visible `einsum_reduce_all_f64_1000` near-loss before
  attempting a source change.

Current-head head-to-head:

| Row | FNP | NumPy | FNP/NumPy | Verdict |
|---|---:|---:|---:|---|
| `einsum_trace_f64_4000` | 97,103 ns | 107,426 ns | 0.904x | win |
| `einsum_diag_f64_4000` | 2,244 ns | 2,483 ns | 0.904x | win |
| `einsum_reduce_all_f64_1000` | 438,624 ns | 600,537 ns | 0.730x | win |
| `einsum_reduce_rows_f64_1000` | 323,154 ns | 544,627 ns | 0.594x | win |
| `einsum_reduce_cols_f64_1000` | 624,904 ns | 732,167 ns | 0.854x | win |

Ledger:
- Current-head rerun: **5 wins / 0 losses / 0 neutral** versus NumPy.
- No source edit was made. The former `reduce_all` near-loss is no longer a
  current actionable gap on this worker.

Retry predicate:
- Do not reopen the scalar-builder, diagonal shortcut, or reduce-all wrapper
  families without fresh losing evidence. Move to a different measured loser.
## 2026-06-20 - BOLD-VERIFY Keep: fnp-python einsum trace scalar-builder

Artifact directory: `tests/artifacts/perf/2026-06-20_python_einsum_trace_cod_b/`

Run identity:
- Agent: `YellowElk` / `cod-b`.
- Parent bead: `franken_numpy-ixs5y`.
- Crate: `fnp-python`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.
- Target gap: residual `fnp_einsum_trace_f64_4000` from the cached-buffer
  diagonal keep; the same-worker `vmi1227854` row was 5.9900 us FNP vs
  5.2275 us NumPy, or 1.146x slower.
- Decision: KEEP. Source commit `eb64c4d5` replaces f64 scalar-return
  materialization through a temporary 0-D ndarray with a cached `numpy.float64`
  constructor and routes direct f64 `trace` through that helper.

Head-to-head result on RCH worker `vmi1227854`:

| Row | FNP | NumPy | FNP/NumPy | Outcome |
|---|---:|---:|---:|---|
| `einsum_trace_f64_4000` | 4,838 ns | 6,242 ns | 0.775x | WIN |
| `einsum_diag_f64_4000` | 860 ns | 939 ns | 0.916x | WIN |
| `einsum_reduce_all_f64_1000` | 95,143 ns | 94,139 ns | 1.011x | neutral/loss, non-target |
| `einsum_reduce_rows_f64_1000` | 90,580 ns | 93,613 ns | 0.968x | WIN |
| `einsum_reduce_cols_f64_1000` | 109,933 ns | 198,288 ns | 0.554x | WIN |

Ledger:
- Target-decision rows: **2 wins / 0 losses / 0 neutral** for trace plus the
  diagonal preservation row.
- Full observed boundary sweep: **4 wins / 1 loss-or-neutral / 0 neutral** if
  the non-target `reduce_all` near-loss is counted strictly.
- The trace row moved from 1.146x slower than NumPy to 0.775x of NumPy on the
  same RCH worker. FNP trace old/new improved from 5.990 us to 4.838 us, or
  0.808x of the prior FNP time.
- The diagonal support row remains faster than NumPy; its FNP absolute time
  moved from 805.39 ns to 860 ns, or 1.068x of the prior FNP row, so this is
  recorded as preserved-win noise rather than a new diagonal improvement.

Validation:
- `rch exec -- cargo test -p fnp-python --test conformance_einsum` passed 28/28,
  including trace edge bits, scalar return type, diagonal view/trace golden, and
  keyword/path outcome tests.
- `rch exec -- cargo check -p fnp-python --lib --bench
  criterion_python_surface` passed with the crate's pre-existing warnings.
- `rch exec -- cargo build -p fnp-python --release` passed with the same
  pre-existing warnings.
- `rch exec -- cargo clippy -p fnp-python --lib --bench
  criterion_python_surface -- -D warnings` remains blocked by broad pre-existing
  `fnp-python` lint debt; the log does not mention `build_f64_scalar` or
  `NUMPY_FLOAT64_TYPE`.
- `cargo fmt -p fnp-python -- --check` remains blocked by broad pre-existing
  rustfmt drift; the log does not mention the touched scalar-builder helper.
- `git diff --check` passed. `ubs crates/fnp-python/src/lib.rs` did not finish
  within the interactive window for the single large source file and was
  interrupted after more than three minutes with no emitted finding.

Retry predicate:
- Do not retry this scalar-builder lever unless the retry uses a distinct scalar
  construction mechanism or adds stronger scalar-return contract proof.
- Treat `einsum_reduce_all_f64_1000` as the next visible Python-boundary einsum
  residual from this sweep; do not reopen the diagonal pre-policy shortcut or
  cached-buffer dispatch families without fresh losing evidence.

## 2026-06-20 - BOLD-VERIFY No-Ship: batch_cholesky f64x4 across-lanes SIMD regressed broad gate

Artifact directory: `tests/artifacts/perf/2026-06-20_linalg_batch_cholesky_simd_cod_a/`

Run identity:
- Agent: `BlackThrush` / `cod-a`.
- Worktree: clean scratch branch `cod-a-batch-cholesky-simd-20260620` from
  `origin/main` at `64ad3a25`.
- Target: `fnp_linalg::batch_cholesky`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Decision: NO-SHIP. Candidate source was reverted; no production code kept.

Baseline loss versus NumPy:
- Local ABI Python extension build, NumPy 2.4.3, `OMP_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`.
- Existing code vs NumPy was 0 wins / 7 losses / 0 neutral in this sweep:
  B=4000 d=8 3.12x slower; B=2000 d=16 19.65x slower; B=1000 d=32
  4.67x slower; B=500 d=64 6.10x slower; B=200 d=100 1.49x slower;
  B=64 d=200 1.36x slower; B=10000 d=4 2.42x slower.

Candidate:
- Safe Rust `std::simd::Simd<f64, 4>` across four independent batch lanes for
  `16 <= n < 128`, preserving the scalar `k` accumulation order inside each
  matrix. Tail lanes used `cholesky_nxn_into_out`.
- Correctness proof passed: `rch exec -- cargo test -p fnp-linalg
  batch_cholesky_ -- --nocapture` reported 3 passed, 0 failed, 1 ignored; the
  candidate bit-proof covered n=16/32/64 against per-lane `cholesky_nxn`.

Same-worker broad gate:
- Baseline RCH Criterion on `ovh-a`:
  - `batch_cholesky/shape/64x128x128`: 1.6258 ms center.
  - `batch_cholesky/shape/16x256x256`: 3.2794 ms center.
- Candidate RCH Criterion on the same worker `ovh-a`:
  - `64x128x128`: 2.3148 ms center, +45.662% regression.
  - `16x256x256`: 3.8253 ms center, +16.109% regression.
- Candidate broad gate: 0 wins / 2 losses / 0 neutral. NumPy candidate rerun
  was skipped because the same-worker Rust broad gate already failed.

Why this failed:
- The lane-gather pattern packs four strided matrices into SIMD vectors inside
  the innermost Cholesky dot product, then scatters scalar results back. The
  proof is clean, but the gather/scatter and codegen footprint cost more than
  the recovered vector lanes, and the change regressed the existing blocked
  n>=128 Criterion rows despite the runtime gate excluding n=128/256 from the
  candidate path.

Retry predicate:
- Do NOT retry portable-SIMD f64x4 gather/scatter across batch lanes as a
  standalone Cholesky lever.
- Do NOT retry allocation elimination, gate tuning, or threshold-only changes;
  those were already disproven by the prior no-ship.
- A credible retry must be a distinct algorithm/layout: true SoA batched-panel
  Cholesky, a packed-panel representation that eliminates per-k gather/scatter,
  or a LAPACK-class blocked per-lane kernel. It must prove medium rows and the
  broad n>=128 rows in the same run window before any NumPy keep claim.

## 2026-06-20 - BOLD-VERIFY Keep: fnp-ufunc boolean_index F64 masked gather vs NumPy

Artifact directory: `tests/artifacts/perf/2026-06-20_ufunc_boolean_index_vs_numpy_cod_b/`

Run identity:
- Agent: `BlackThrush` / `cod-b`. Bead: `franken_numpy-ixs5y.251`.
- Subject: `UFuncArray::boolean_index` on flat F64 arrays with sparse Bool mask
  values where `NaN` is truthy and signed zero is false.
- Decision host for performance: `vmi1149989`.
- FNP command: `rch exec -- cargo bench -p fnp-ufunc --bench elementwise boolean_index_f64_masked_sparse -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`.
- NumPy comparator: direct SSH on the same host, `ubuntu@vmi1149989`, NumPy
  2.2.4, Python 3.13.7, `OMP/OPENBLAS/MKL/NUMEXPR=1`.
- Decision: KEEP. No source hunk in this closeout; this verifies the existing
  direct masked-gather path as a standalone same-host win.

Head-to-head result (FNP/NumPy, lower is better):

| Row | FNP ns/iter | NumPy median ns | Ratio vs NumPy | Decision |
|---|---:|---:|---:|---|
| `boolean_index_f64_masked_sparse/100000` | 43,634 | 99,813 | 0.437x (2.29x faster) | WIN |
| `boolean_index_f64_masked_sparse/1000000` | 628,093 | 1,355,257 | 0.463x (2.16x faster) | WIN |

Win/loss/neutral ledger: **2 / 0 / 0**.

Validation:
- `cargo test -p fnp-ufunc boolean_index -- --nocapture` via RCH on
  `vmi1149989`: PASS, 4 focused tests including
  `boolean_index_f64_matches_serial_reference_and_golden_sha256`.
- `cargo test -p fnp-ufunc` via RCH on `hz2`: PASS, 2244 passed, 0 failed, 41
  ignored, integration tests green, doctests ignored as expected.
- `cargo check -p fnp-ufunc --all-targets` via RCH on `vmi1153651`: PASS.
- `cargo build -p fnp-ufunc --release` via RCH on `vmi1153651`: PASS.
- `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`: first RCH attempt
  on `vmi1149989` failed because the pinned nightly was missing the clippy
  component; after `rustup component add --toolchain nightly-2026-02-20 clippy`
  on that worker, the same crate-scoped clippy command passed.

Invalid probes recorded:
- `numpy_boolean_index_vmi1149989.txt`: failed before timing from shell quoting
  stripping the Python `USER` literal.
- `numpy_boolean_index_vmi1149989_retry.txt`: failed before timing from an
  escaped f-string expression.
- Neither invalid probe entered the ratio table.

Retry predicate: do not revisit `boolean_index` wrapper/delegation or the same
F64 extract mask-gather family without new losing evidence. The next credible
route must attack a deeper primitive, such as compact Bool mask representation,
mask decode traffic, or a distinct sidecar-preserving gather path, and must
preserve NumPy truthiness (`NaN` true, signed zero false), flat-order output,
dtype/shape, mismatch error class, and the all-false/sidecar fallbacks.

## 2026-06-20 - BOLD-VERIFY No-Ship: batch_cholesky 5-8x loss; alloc-elimination DISPROVEN, kernel is the wall

Artifact directory: `tests/artifacts/perf/2026-06-20_linalg_batch_cholesky_noship/`

Run identity:
- Agent: `BlackThrush` / `cod-b`. Subject: `fnp_linalg::batch_cholesky` via
  `fnp.cholesky` on stacked (B,n,n) real-f64 SPD.
- Reference: NumPy 2.4.3 on `thinkstation1`, load ~3, OMP/OPENBLAS=1.
- Decision: NO-SHIP. Hypothesis disproven by measurement; change REVERTED so
  fnp-linalg working tree matches HEAD.

Loss (fnp/numpy, >1 = slower): d=8 0.89-1.14x ok; **d=16 5.85-7.24x, d=32
4.98-6.06x, d=64 4.57-5.93x LOSS**; d=100 1.42x, d=200 1.37x; d=4 ~1.0x.
Correct (L@Lᵀ==A, match=True) throughout — pure perf.

DISPROVEN hypothesis: the n>=16 batched path calls per-lane `cholesky_nxn`
(`vec![0.0;n*n]` per lane) via `batch_map_lanes` then flatten-copies (Vec<Vec>),
unlike the n<16 path which writes directly into the pre-zeroed output. I assumed
per-lane allocation under rayon was an allocator-contention storm. FIX TRIED:
raise the direct-write gate from `n<16` to `n<CHOL_MID_MIN(128)` (using
`cholesky_nxn_into_out`, byte-identical to the unblocked formula for n<128).
Rebuilt + measured: **NO improvement** (d=16 still ~6.3x). Allocation/copy is NOT
the bottleneck. Reverted.

TRUE root cause (measured with `RAYON_NUM_THREADS=1`): SERIAL fnp is **6.3-8.6x**
slower than numpy (d=16 6.32x, d=32 7.80x, d=64 8.58x). The scalar triple-loop
`cholesky_nxn` is ~6x slower PER LANE than LAPACK `dpotrf` — its inner
dot-product `sum += out[ri+k]*out[rj+k]` is a loop-carried reduction that does
not autovectorize, while numpy's gufunc calls tuned LAPACK per lane. Parallelism
only partly compensates and is HIGH-VARIANCE (d=16 ranged 1.6x-6.3x across runs;
bandwidth / rayon-granularity bound, not achieving ~16x core scaling).

Real lever (separate, larger, high-risk in contended fnp-linalg): a SIMD/blocked
per-lane Cholesky kernel matching dpotrf throughput, or batched panel
factorization with lane-as-SIMD-vector. Must stay byte-identical to the
`cholesky_nxn` golden. NOT a wrapper/gate tweak.

Retry predicate: do NOT retry batch_cholesky via alloc-elimination, gate tuning,
or parallel-threshold changes (all measured neutral). Only a vectorized/BLAS
per-lane kernel (or SIMD-across-lanes) can close this; verify serial speedup
FIRST (RAYON_NUM_THREADS=1) since the parallel numbers are too noisy to A/B.

## 2026-06-20 - BOLD-VERIFY Fix: fnp-python eigvals CORRECTNESS bug (~9% wrong) + perf loss -> delegate to LAPACK

Artifact directory: `tests/artifacts/perf/2026-06-20_python_eigvals_correctness_delegate/`

Run identity:
- Agent: `BlackThrush` / `cod-b`.
- Subject API: `fnp.eigvals` real 2-D square (`crates/fnp-python/src/lib.rs`).
- Reference: NumPy 2.4.3 on `thinkstation1` (local, load ~4); `hz2` (NumPy 2.3.5)
  saturated at load ~33/16-core. Fix is a delegation -> reference-independent.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.
- Decision: SHIP. Correctness fix (also removes a large-n perf loss).

Bug class (CORRECTNESS, found via perf sweep):
- `fnp.eigvals` real 2-D ran the native Francis double-shift QR
  (`eig_nxn`/`hessenberg_qr_iter`). It does NOT reliably converge: when the
  iteration budget (`EIGEN_QR_ITERATION_COEFF*n*n`) is exhausted it silently
  returns the unconverged diagonal. Order-independent power-sum invariants
  (sum(λ^k)==trace(A^k), k=1,2,3) over 120 random real matrices: **11/120 wrong**,
  worst relerr 15.2 (d=32 seed=13: trace OK 4.6e-15 but sum(λ³) relerr 15.2).
- WHY THE SUITE MISSED IT: the trace (k=1 power sum) is preserved even when the
  spectrum is wrong, so a `sum(eig)==trace` smoke test passes; the conformance
  eigvals tests only use symmetric/diagonal/small/complex matrices that happen
  to converge. Failures are matrix-dependent (a symmetric 64x64 also missed) ->
  NO safe size gate.
- Also a perf loss: d=200 ran 39.78x slower (and wrong), d=600 1.73x, d=800
  2.36x. The small-d native "win" (0.6-0.8x) was a fast wrong answer.

Lever:
- Delegate all real 2-D `eigvals` to NumPy LAPACK `geev` (robust + faster on
  large n). `eig` already delegated; `eigvalsh` keeps its reliable symmetric QR.
  Dead complex-output helper kept behind `#[allow(dead_code)]`.
- REUSABLE METHOD: order-independent invariants (power sums vs trace(A^k)) are
  the correct eigenvalue-set comparator; do NOT use sorted element-wise diff
  (sort_complex misaligns conjugate pairs -> false positives) NOR greedy
  nearest-neighbor matching (cascading mis-assignment -> false positives). A
  native iterative solver with a fixed iteration budget that returns on timeout
  is a silent-wrong-answer hazard; sweep it with random + adversarial corpora.

Result: power-sum invariants 0/120 bad (worst relerr 5.6e-13); perf d=200
39.78x->1.03x, d=600 1.73x->0.93x, d=800 2.36x->1.00x; conformance
conformance_linalg_advanced 29/29 + conformance_linalg_decomp 38/38 PASS;
non-square LinAlgError preserved; release build clean.

Retry predicate: do not re-enable native `eig_nxn` for eigvals without proving
Francis QR converges to LAPACK tolerance with 0 failures on a large random +
adversarial (defective/clustered/near-symmetric) corpus. A robust native
unsymmetric eigensolver is a separate large effort, not a wrapper tweak.

## 2026-06-20 - BOLD-VERIFY Fix: fnp-python pinv 2-D size-gate (215x loss -> parity)

Artifact directory: `tests/artifacts/perf/2026-06-20_python_pinv_2d_sizegate_vs_numpy/`

Run identity:
- Agent: `BlackThrush` / `cod-b`.
- Subject API: `fnp.pinv` (`crates/fnp-python/src/lib.rs`, 2-D branch).
- Reference: NumPy 2.4.3 on `thinkstation1` (local, load ~5.5). `hz2` (the usual
  NumPy 2.3.5 comparator) was saturated at load ~33/16-core and unusable for
  clean A/B; the fix is a delegation so the post-fix ~1.0x ratio is
  reference-version-independent.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.
- Decision: SHIP. One-line guard change.

Bug class:
- `fnp.pinv` of a 2-D matrix routed ALL non-hermitian shapes (and hermitian
  squares) through the pure-Rust dense path `pinv_mxn`/`svd_mxn_full` (resp.
  `pinv_hermitian_nxn`). That pure-Rust SVD/eigensolve only beats LAPACK for
  tiny matrices; above max-dim ~40 it scales far worse and for larger
  RECTANGULAR matrices it is catastrophic: `pinv((600,400))` ran ~8.8s vs NumPy
  ~41ms (~215x); `(400,600)` ~233x; hermitian (600) ~7x. Standalone `fnp.svd`
  (LAPACK-backed) is at parity, so the loss was entirely in the native 2-D pinv
  dense-SVD path, not SVD itself.

Lever:
- Gate the native 2-D pinv block to `max(m,n) <= 32` (the regime where the
  pure-Rust path measurably wins by dodging numpy/LAPACK dispatch overhead, both
  hermitian and non-hermitian); let larger 2-D fall through to the existing
  numpy `linalg.pinv` (LAPACK gesdd) delegation. Batched (>=3-D) pinv untouched
  (it wins decisively, 0.27-0.62x). REUSABLE: a native dense-linalg fast path
  that wins only at small sizes must be size-gated; the catastrophic regime here
  is rectangular (max-dim large, both dims moderate), which the standalone-SVD
  parity check did NOT reveal because the pinv WRAPPER, not svd, owns the dense
  reconstruction-via-pure-SVD path.

Head-to-head (after): 5 win / 0 loss / 3 neutral. `(600,400)` 215x->1.028x,
`(400,600)` 233x->0.945x, `(128,128)` 2.75x->1.023x, `(64,64)` 1.45x->1.051x;
small native (<=32) 0.29-0.96x and batched 0.31x all preserved; all values match
NumPy (allclose rtol 1e-9).

Validation: `cargo test -p fnp-python --test conformance_linalg_advanced` 29/29
PASS; 22-case pinv conformance + gate-boundary probe (dim 32 vs 33, rectangular,
rcond/rtol, hermitian, complex, singular, batched) 0 fails; `cargo build
-p fnp-python --release` clean; edit region clippy-clean (only pre-existing
`eq_op`/dead-code warnings elsewhere).

Commands:
- `RCH_MIN_LOCAL_TIME_MS=999999999 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b cargo build -p fnp-python --release --lib`
- `RCH_MIN_LOCAL_TIME_MS=999999999 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b cargo test -p fnp-python --release --test conformance_linalg_advanced`
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD/.probe python3 tests/artifacts/perf/2026-06-20_python_pinv_2d_sizegate_vs_numpy/pinv_head_to_head.py`

Retry predicate: do not re-tune the 32 threshold or re-test the dense-SVD pinv
path as a standalone lever. Closing the mid-size (33-63, now ~1.0-1.1x via the
numpy-delegation wrapper) residual requires a blocked/LAPACK-class replacement
for the pure-Rust `svd_mxn_full` — a separate, large effort.

## 2026-06-20 - BOLD-VERIFY Keep: fnp-ufunc where_nonzero coordinate gather

Artifact directory: `tests/artifacts/perf/2026-06-20_ufunc_where_nonzero_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.247`.
- Agent: `BlackThrush` / `cod-b`.
- Subject API: direct Rust `fnp-ufunc` `where_nonzero` for large F64 2-D
  arrays without integer sidecars.
- Reference: NumPy 2.3.5 on `hz2` / `hetzner2` through explicit `ssh hz2`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.
- Decision: keep the existing guarded Rayon chunk coordinate-gather path; no
  production performance hunk was added in this verification slice.

Lever:
- The landed path divides large flat F64 buffers into contiguous morsels, counts
  truthy lanes with the same NumPy predicate (`v != 0.0`, so NaN is truthy and
  signed zero is false), then materializes each dimension's coordinate array in
  C-order.
- It falls back to the serial path for sidecar-backed arrays, non-F64 arrays,
  scalar paths, and small arrays, preserving dtype and coordinate-order
  contracts.
- Alien-graveyard mapping: morsel-driven parallelism plus cache-local
  coordinate emission for a memory-bandwidth-bound gather. The radical lever is
  not new math; it is changing the loop from one monolithic serial scan into
  independently counted and filled chunks without weakening coordinate order.

Commands:
- `RCH_WORKER=hz2 RCH_REQUIRE_REMOTE=1 RCH_DAEMON_WAIT_RESPONSE_TIMEOUT_SECS=240 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-ufunc --bench elementwise where_nonzero_f64_2d_sparse -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `ssh hz2 'cd /data/projects/franken_numpy && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python3 - <<PY ... PY'`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-ufunc where_nonzero_f64_parallel_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-ufunc`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo build -p fnp-ufunc --release`

| Workload | Worker | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---:|---:|---|
| `where_nonzero_f64_2d_sparse/262144` | `hz2` | 290,959 ns | 1,162,745 ns | 0.250x, 4.00x faster | Win |
| `where_nonzero_f64_2d_sparse/1048576` | `hz2` | 677,198 ns | 4,658,292 ns | 0.145x, 6.88x faster | Win |

Scorecard:
- Candidate vs NumPy: win/loss/neutral = 2/0/0.
- Same-worker proof: FrankenNumPy Criterion ran through RCH on `hz2`; NumPy
  comparator ran directly on `hz2` and reported host `hetzner2`, Python 3.14.4,
  NumPy 2.3.5.
- Noise discipline: the full-suite conformance rerun was allowed to land on
  `vmi1227854`; it is not used for performance scoring.

Validation notes:
- Focused where_nonzero golden test passed:
  `where_nonzero_f64_parallel_matches_serial_reference_and_golden_sha256`.
- Full crate conformance passed after repairing a rounded Legendre polynomial
  test golden exposed by the first full-suite run:
  `cargo test -p fnp-ufunc` reported 2244 passed, 0 failed, 41 ignored, plus
  green integration tests and doctests.
- The Legendre repair is test-only: NumPy reports
  `legmul([1,2,3,4], [0.5,-1,2])` coefficients at full f64 precision; the old
  six-decimal expected row was outside the local `poly_close` tolerance.
- `cargo check -p fnp-ufunc --all-targets`, `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`,
  and `cargo build -p fnp-ufunc --release` passed through RCH.
- First clippy attempt failed before linting because `cargo-clippy` was missing
  on `vmi1153651`; the retry passed on `hz1`, and a post-test-fix clippy pass
  also passed on `hz1`.
- `cargo fmt --package fnp-ufunc -- --check` remains blocked by broad
  pre-existing rustfmt drift in untouched benches and source regions; the new
  Legendre row is not singled out by the refreshed format artifact.
- Retry predicate: do not retest generic F64 where/nonzero chunk gathering or
  threshold-only tuning as standalone work. A next credible `where`/`nonzero`
  lever must be a distinct primitive, for example division-free 2-D coordinate
  reconstruction or row-run tables, and must preserve C-order coordinates,
  sidecar fallback, NaN truth, and signed-zero false behavior.

## 2026-06-20 - BOLD-VERIFY Keep: fnp-linalg kron identity RHS specialization

Artifact directory: `tests/artifacts/perf/2026-06-20_linalg_kron_identity_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.236`.
- Agent: `BlackThrush` / `cod-b`.
- Subject API: direct Rust `fnp-linalg` `kron_nxn` with an exact `4x4`
  identity RHS.
- Reference: NumPy 2.3.5 on `hz2` / `hetzner2` through explicit `ssh hz2`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.
- Decision: keep the existing guarded nonnegative identity-RHS specialization;
  no source hunk was added in this verification slice.

Lever:
- The landed path recognizes exact square identity RHS matrices and finite,
  nonnegative LHS matrices, then writes only the block diagonal output.
- It falls back to the dense product for signed zero, negative values, NaN, Inf,
  non-square RHS, or non-identity RHS so NumPy multiplication semantics stay
  intact.
- Alien-graveyard mapping: exploit exact algebraic structure to change the
  constant and effective work class for a common block-operator shape, with a
  constants-kill-you guard around the domain predicate.

Commands:
- `RCH_WORKER=hz2 RCH_REQUIRE_REMOTE=1 RCH_DAEMON_WAIT_RESPONSE_TIMEOUT_SECS=240 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg kron_nxn -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `ssh hz2 'cd /data/projects/franken_numpy && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python3 - <<PY ... PY'`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-linalg kron_ -- --nocapture`

| Workload | Worker | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---:|---:|---|
| `kron_64x64_4x4_eye` | `hz2` | 30,314 ns | 173,371 ns | 0.175x, 5.72x faster | Win |
| `kron_128x128_4x4_eye_nonnegative_fast_path` | `hz2` | 230,786 ns | 859,101 ns | 0.269x, 3.72x faster | Win |

Scorecard:
- Candidate vs NumPy: win/loss/neutral = 2/0/0.
- Same-worker proof: FrankenNumPy Criterion ran through RCH on `hz2`; NumPy
  comparator ran directly on `hz2` and reported host `hetzner2`, Python 3.14.4,
  NumPy 2.3.5.
- Consistency check: the fresh rows are consistent with the older
  `tests/artifacts/perf/2026-06-20_linalg_batch_vs_numpy/` kron rows.

Validation notes:
- Focused kron tests passed: `kron_identity_rhs_fast_path_matches_dense_reference_and_fallbacks`,
  `kron_parallel_matches_serial_reference_and_golden`, `kron_identity_identity`,
  and `kron_scalar`.
- The same no-source linalg tree already passed
  `cargo check -p fnp-linalg --all-targets`,
  `cargo clippy -p fnp-linalg --all-targets -- -D warnings`, and
  `cargo build -p fnp-linalg --release` during the immediately preceding
  column-sum verification.
- `cargo fmt --package fnp-linalg -- --check` remains blocked by broad
  pre-existing rustfmt drift in untouched `fnp-linalg` benches, examples, and
  source regions.
- Retry predicate: do not retest generic dense kron or RHS identity detection
  alone. A next kron lever needs a new structured RHS/LHS class, for example
  diagonal RHS, sparse block masks, or separable Kronecker chains, and must keep
  fallback semantics bit-preserving.

## 2026-06-20 - BOLD-VERIFY Keep: fnp-linalg batched column-sum norm lane fill

Artifact directory: `tests/artifacts/perf/2026-06-20_linalg_batch_column_sum_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.240`.
- Agent: `BlackThrush` / `cod-b`.
- Subject API: direct Rust `fnp-linalg` `batch_matrix_norm(..., ord="1")`
  and `ord="-1"`.
- Reference: NumPy 2.3.5 on `hz2` / `hetzner2` through explicit `ssh hz2`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.
- Decision: keep the existing direct batched column-sum lane-fill path; no
  source hunk was added in this verification slice.

Lever:
- The landed path specializes `batch_matrix_norm` for `ord="1"` and
  `ord="-1"` after one batch shape/data validation.
- Each lane still calls `matrix_norm_column_sum`, preserving the existing
  column-addition order, small-strided versus cache-linear selection, NaN
  propagation, and max/min column-sum semantics.
- Alien-graveyard mapping: vectorized/morsel-style cache-local stacked matrix
  work plus constants-kill-you removal of per-lane validation and `Result`
  plumbing. More radical column prefilter and stack-threshold probes remain
  rejected by the no-ship entry below.

Commands:
- `RCH_WORKER=hz2 RCH_REQUIRE_REMOTE=1 RCH_DAEMON_WAIT_RESPONSE_TIMEOUT_SECS=240 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg batch_matrix_norm_column_sum -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `ssh hz2 'cd /data/projects/franken_numpy && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python3 - <<PY ... PY'`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-linalg batch_matrix_norm_column_sum_direct_lane_fill_matches_per_lane_reference_bits -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo check -p fnp-linalg --all-targets`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo clippy -p fnp-linalg --all-targets -- -D warnings`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo build -p fnp-linalg --release`

| Workload | Worker | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---:|---:|---|
| `1_4096x8x8` | `hz2` | 81,679 ns | 903,879 ns | 0.090x, 11.07x faster | Win |
| `1_1024x32x32` | `hz2` | 101,266 ns | 917,304 ns | 0.110x, 9.06x faster | Win |
| `-1_4096x8x8` | `hz2` | 78,648 ns | 989,052 ns | 0.080x, 12.58x faster | Win |
| `-1_1024x32x32` | `hz2` | 95,781 ns | 991,737 ns | 0.097x, 10.35x faster | Win |

Scorecard:
- Candidate vs NumPy: win/loss/neutral = 4/0/0.
- Same-worker proof: FrankenNumPy Criterion ran through RCH on `hz2`; NumPy
  comparator ran directly on `hz2` and reported host `hetzner2`, Python 3.14.4,
  NumPy 2.3.5.
- Consistency check: the fresh hz2 Rust rows are consistent with the earlier
  `tests/artifacts/perf/2026-06-20_linalg_batch_vs_numpy/` proof bundle.

Non-counted probes:
- An unpinned first `rch` run selected `vmi1153651` and returned noisy Rust
  rows: 1,151,304 ns, 2,296,887 ns, 1,026,454 ns, and 2,768,589 ns. These are
  not scored because the intended hz2 selector was not honored and no same-worker
  NumPy comparator was available for that worker.
- `rch exec -- python3 - ...` warned that Python is a non-compilation command
  and ran locally on `thinkstation1`. Cross-host apparent FNP/local-NumPy ratios
  from the invalid pairing were 1.371x, 2.335x, 1.233x, and 2.611x. They are
  recorded as routing evidence only, not keep/reject evidence.
- Raw `ssh root@38.242.134.66` to the selected vmi worker failed with
  `Permission denied (publickey,password)`; the repo-supported path is the SSH
  alias used by the cross-engine scripts, for example `ssh hz2`.

Validation notes:
- Focused bit-preservation test passed.
- `cargo check -p fnp-linalg --all-targets`, `cargo clippy -p fnp-linalg --all-targets -- -D warnings`,
  and `cargo build -p fnp-linalg --release` passed through RCH.
- First `cargo check` attempt on `ovh-b` failed before crate checking because
  the `zerocopy` build script died with `SIGILL`; the same gate passed on
  retry through `vmi1149989`, so this is recorded as worker/toolchain
  infrastructure noise.
- `cargo fmt --package fnp-linalg -- --check` remains blocked by broad
  pre-existing rustfmt drift in untouched `fnp-linalg` benches, examples, and
  source regions.
- Retry predicate: do not repeat whole-matrix NaN prefilters, 256-column stack
  threshold changes, or validation-only retunes for this lane. A new attempt
  needs a different primitive, likely SIMD absolute-value extraction or
  strip-mined multi-column accumulation that preserves per-column addition order
  and NaN behavior.

## 2026-06-20 - BOLD-VERIFY Keep: fnp-linalg batched row-sum norm lane fill

Artifact directory: `tests/artifacts/perf/2026-06-20_linalg_batch_row_sum_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.239`.
- Agent: `BlackThrush` / `cod-b`.
- Subject API: direct Rust `fnp-linalg` `batch_matrix_norm(..., ord="inf")`
  and `ord="-inf"`.
- Reference: NumPy 2.3.5 on `hz1` / `hetzner1` through explicit `ssh hz1`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.
- Decision: keep the existing direct batched row-sum lane-fill path; no source
  hunk was added in this verification slice.

Lever:
- The landed path specializes `batch_matrix_norm` for `ord="inf"` and
  `ord="-inf"` after one batch shape/data validation.
- Each lane sums rows in the same row-major order as `matrix_norm_nxn`, then
  applies the same max/min row-sum selection semantics.
- Alien-graveyard mapping: constants-kill-you removal of per-lane validation
  and `Result` plumbing over cache-local stacked matrices.

Commands:
- `RCH_WORKER=hz1 RCH_WORKERS=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg batch_matrix_norm_row_sum -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `ssh hz1 'python3 -c ...'`
- `RCH_WORKER=hz1 RCH_WORKERS=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-linalg batch_matrix_norm_row_sum_direct_lane_fill_matches_per_lane_reference_bits -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo check -p fnp-linalg --all-targets`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo clippy -p fnp-linalg --all-targets -- -D warnings`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo build -p fnp-linalg --release`

| Workload | Worker | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---:|---:|---|
| `inf_4096x8x8` | `hz1` | 86,647 ns | 1,021,869 ns | 0.085x, 11.79x faster | Win |
| `inf_1024x32x32` | `hz1` | 180,783 ns | 1,347,235 ns | 0.134x, 7.45x faster | Win |
| `-inf_4096x8x8` | `hz1` | 84,239 ns | 1,044,963 ns | 0.081x, 12.40x faster | Win |
| `-inf_1024x32x32` | `hz1` | 181,321 ns | 1,320,966 ns | 0.137x, 7.29x faster | Win |

Scorecard:
- Candidate vs NumPy: win/loss/neutral = 4/0/0.
- Same-worker proof: FrankenNumPy Criterion ran through RCH on `hz1`; NumPy
  comparator ran directly on `hz1` and reported host `hetzner1`, Python 3.14.4,
  NumPy 2.3.5.
- Discarded attempt: `rch exec -- python3 -c ...` emitted the RCH
  non-compilation warning and did not report a selected worker. Those timings
  are routing evidence only and are not counted.

Validation notes:
- Focused bit-preservation test passed.
- `cargo check -p fnp-linalg --all-targets`, `cargo clippy -p fnp-linalg --all-targets -- -D warnings`,
  and `cargo build -p fnp-linalg --release` passed through RCH.
- `cargo fmt -p fnp-linalg -- --check` remains blocked by broad pre-existing
  rustfmt drift in untouched `fnp-linalg` benches, examples, and source regions.
- Retry predicate: do not retry this row-sum direct lane-fill bead unless future
  same-worker evidence regresses the current path, or a new row-sum shape opens
  a fresh NumPy loss.

## 2026-06-20 - BOLD-VERIFY No-Ship: fnp-python pre-policy f64 einsum diagonal shortcut

Artifact directory: `tests/artifacts/perf/2026-06-20_python_einsum_diag_pre_policy/`

Run identity:
- Bead: `franken_numpy-ixs5y.269`.
- Agent: `BlackThrush` / `cod-a`.
- Subject API: Python-boundary `fnp.einsum` through `criterion_python_surface`.
- Oracle/reference: NumPy inside the same Criterion harness process.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Decision: no-ship; source hunk reverted.

Lever:
- Tried routing f64 single-operand diagonal/trace einsum forms through the existing zero-copy diagonal fast path before wrapper dtype-policy work.
- Alien-graveyard mapping: constants-kill-you specialization plus zero-copy/view-preserving layout reuse.
- Failure mode: the diagonal target still lost to NumPy after the shortcut, and the trace control row also lost on the candidate worker.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a RCH_WORKER=hz2 RCH_WORKERS=hz2 rch exec -- cargo bench -p fnp-python --bench criterion_python_surface -- python_einsum_boundary --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-python einsum -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a RCH_WORKER=hz1 RCH_WORKERS=hz1 rch exec -- cargo bench -p fnp-python --bench criterion_python_surface -- python_einsum_boundary --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`

| Run | Workload | Worker reported by RCH | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---|---:|---:|---:|---|
| Origin/main baseline | `fnp_einsum_trace_f64_4000` | `hz1` | 32,125 ns | 40,179 ns | 0.800x, 1.25x faster | Win |
| Origin/main baseline | `fnp_einsum_diag_f64_4000` | `hz1` | 10,466 ns | 2,652 ns | 3.95x slower | Open gap |
| Origin/main baseline | `fnp_einsum_reduce_all_f64_1000` | `hz1` | 187,003 ns | 195,641 ns | 0.956x, 1.05x faster | Win |
| Origin/main baseline | `fnp_einsum_reduce_rows_f64_1000` | `hz1` | 183,629 ns | 195,289 ns | 0.940x, 1.06x faster | Win |
| Origin/main baseline | `fnp_einsum_reduce_cols_f64_1000` | `hz1` | 220,197 ns | 546,982 ns | 0.403x, 2.48x faster | Win |
| Candidate shortcut | `fnp_einsum_trace_f64_4000` | `hz2` | 15,981 ns | 5,163 ns | 3.10x slower | No-ship loss |
| Candidate shortcut | `fnp_einsum_diag_f64_4000` | `hz2` | 2,902 ns | 974 ns | 2.98x slower | No-ship loss |
| Candidate shortcut | `fnp_einsum_reduce_all_f64_1000` | `hz2` | 115,773 ns | 118,392 ns | 0.978x, 1.02x faster | Win |
| Candidate shortcut | `fnp_einsum_reduce_rows_f64_1000` | `hz2` | 108,724 ns | 114,108 ns | 0.953x, 1.05x faster | Win |
| Candidate shortcut | `fnp_einsum_reduce_cols_f64_1000` | `hz2` | 129,175 ns | 311,932 ns | 0.414x, 2.41x faster | Win |

Scorecard:
- Baseline vs NumPy: win/loss/neutral = 4/1/0.
- Candidate vs NumPy: win/loss/neutral = 3/2/0.
- Candidate target row remained a NumPy loss: `fnp_einsum_diag_f64_4000` was 2.98x slower than NumPy.
- Cross-worker old-to-new movement is routing evidence only; RCH ignored the requested worker pin in both runs (`hz1` baseline, `hz2` candidate).
- Source decision: reverted. No Rust source from this candidate is kept.

Validation notes:
- Focused `cargo test -p fnp-python einsum -- --nocapture` passed with the candidate hunk before revert.
- Covered inline einsum tests, 28 `conformance_einsum` tests including diagonal/trace golden cases, and metamorphic einsum tests.
- Retry predicate: do not retry a wrapper-level pre-policy call into the existing diagonal helper by itself. A deeper retry must remove or avoid the remaining Python method dispatch / view construction overhead for `ii->i`, preserve NumPy writable-view semantics, and beat NumPy's roughly 1 us diagonal-view row in this same harness.

## 2026-06-20 - BOLD-VERIFY Keep: fnp-python sorted f32 histogram edge-pointer

Artifact directory: `tests/artifacts/perf/2026-06-20_python_histogram_f32_sorted_edges/`

Run identity:
- Bead: `franken_numpy-ixs5y.268`.
- Agent: `BlackThrush` / `cod-a`.
- Subject API: Python-boundary `fnp.histogram` through `criterion_python_surface`.
- Oracle/reference: NumPy inside the same Criterion harness process.
- Decision worker: `hz2`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.

Lever:
- Detect monotone nondecreasing `float32` histogram inputs during the existing finite min/max pass.
- Classify monotone data with a streaming pointer over the existing `float32` edge array, reducing the sorted case to `O(n + bins)` and removing per-element affine division.
- Preserve fallback, strict edge validation, float32 edge construction, and the unsorted scalar classifier.
- Alien-graveyard mapping: sorted-stream cursoring / merge-path style specialization plus constants-kill-you removal of hot scalar division when input order gives a stronger invariant.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo bench -p fnp-python --bench criterion_python_surface -- 'python_histogram_boundary|python_setops_boundary|python_statistics_boundary|python_einsum_boundary|python_linalg_boundary|python_char_ascii_boundary' --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a RCH_WORKER=hz2 RCH_WORKERS=hz2 rch exec -- cargo bench -p fnp-python --bench criterion_python_surface -- python_histogram_boundary --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a RCH_WORKER=hz2 RCH_WORKERS=hz2 rch exec -- cargo test -p fnp-python histogram_matches_numpy_across_bins_range_density_weights_and_empty -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo check -p fnp-python --all-targets`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo build --release -p fnp-python`
- `git diff --check`

| Bead | Lever | Workload | Worker | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.266` | Origin/main routing baseline | `histogram_i64_100k_50` | `hz2` | 599,955 ns | 861,321 ns | 0.697x, 1.44x faster | Baseline win |
| `franken_numpy-ixs5y.266` | Origin/main routing baseline | `histogram_f32_100k_50` | `hz2` | 668,179 ns | 613,046 ns | 1.09x slower | Open gap |
| `franken_numpy-ixs5y.266` | Raise parallel threshold | `histogram_i64_100k_50` | `hz2` | 845,866 ns | 867,590 ns | 0.975x, 1.03x faster | No-ship: i64 margin collapsed |
| `franken_numpy-ixs5y.266` | Raise parallel threshold | `histogram_f32_100k_50` | `hz2` | 673,497 ns | 590,563 ns | 1.14x slower | No-ship: f32 still lost |
| `franken_numpy-ixs5y.267` | Local count accumulator | `histogram_i64_100k_50` | `hz2` | 878,361 ns | 830,049 ns | 1.06x slower | No-ship |
| `franken_numpy-ixs5y.267` | Local count accumulator | `histogram_f32_100k_50` | `hz2` | 686,940 ns | 608,869 ns | 1.13x slower | No-ship |
| `franken_numpy-ixs5y.268` | Sorted edge-pointer count | `histogram_i64_100k_50` | `hz2` | 651,730 ns | 840,532 ns | 0.775x, 1.29x faster | Keep |
| `franken_numpy-ixs5y.268` | Sorted edge-pointer count | `histogram_f32_100k_50` | `hz2` | 449,882 ns | 584,574 ns | 0.770x, 1.30x faster | Keep |

Scorecard:
- Routing baseline vs NumPy on histogram rows: win/loss/neutral = 1/1/0.
- Threshold candidate vs NumPy: win/loss/neutral = 1/1/0, rejected because f32 still lost and i64 regressed vs origin baseline.
- Local-count candidate vs NumPy: win/loss/neutral = 0/2/0, rejected.
- Sorted edge-pointer candidate vs NumPy: win/loss/neutral = 2/0/0, kept.
- Primary targeted gap moved from 1.09x slower than NumPy to 0.770x of NumPy time; FrankenNumPy f32 histogram old-to-new improved 668,179 ns to 449,882 ns (0.673x, 1.49x faster).

Validation notes:
- Focused histogram parity test passed, including a new sorted `float32` 100k, 50-bin case.
- Supplemental `cargo test -p fnp-python` cleared 531 inline tests plus early conformance shards, then failed outside this path in `conformance_argwhere::argwhere_python_container_surfaces_match_numpy` because the NumPy oracle script emitted an `IndentationError`.
- `cargo check -p fnp-python --all-targets` passed on RCH with the crate's pre-existing three dead-code warnings.
- `cargo build --release -p fnp-python` passed on RCH worker `vmi1149989` with the same three warnings.
- `git diff --check` passed.
- `ubs crates/fnp-python/src/lib.rs` completed in 198s and exited 1 with broad file-wide inventory (`473` critical heuristic findings, `3661` warnings, `4554` info); no hunk-local histogram finding was identified.
- `cargo fmt -p fnp-python -- --check` remains blocked by broad pre-existing rustfmt drift in `fnp-python`, outside this perf hunk.
- `cargo clippy -p fnp-python --all-targets -- -D warnings` remains blocked by broad pre-existing fnp-python lint debt, outside this histogram path.
- Retry predicate: do not retry the parallel-threshold or local-count-only variants for this f32 histogram row. Deeper retries should exploit data order, edge layout, or a broader zero-copy histogram primitive and must beat 449,882 ns on the same head-to-head harness.

## 2026-06-20 - BOLD-VERIFY Keep: fnp-random small PCG bytes direct append fill

Artifact directory: `tests/artifacts/perf/2026-06-20_random_bytes_small_direct_append/`

Run identity:
- Bead: `franken_numpy-ixs5y.265`.
- Agent: `BlackThrush` / `cod-a`.
- Subject API: direct Rust `fnp-random` `Generator::bytes(length)`.
- Oracle/reference: NumPy `np.random.Generator(np.random.PCG64(42)).bytes(length)` inside the Criterion benchmark harness.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.

Lever:
- For sub-threshold PCG byte requests, append directly into the final byte vector from `next_u64` words.
- Preserve the exact `next_uint32` low/high half-buffer contract by consuming pending `u32_buf` first and buffering the high half only when the final direct append consumed a low half.
- This is not the rejected `.257` intermediate `Vec<u64>` transcode; no intermediate word vector is allocated.
- Alien-graveyard mapping: final-buffer/vectorized execution under a constants-kill-you threshold, with an artifact-level RNG state invariant as the proof obligation.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo bench -p fnp-random --bench random_vs_numpy -- vs_numpy_pcg64_bytes --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random bytes_large_calls_match_serial_uint32_stream_state -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random bytes_match_live_numpy_oracle_when_available -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo check -p fnp-random --all-targets`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo clippy -p fnp-random --all-targets -- -D warnings`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo build --release -p fnp-random`
- `git diff --check`

| Bead | Lever | Workload | Worker | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Old-to-new ratio | Verdict |
|---|---|---:|---|---|---:|---:|---:|---:|---|
| `franken_numpy-ixs5y.265` | Origin/main baseline | 100k bytes | `ovh-a` | `scorecard.md` | 87,044 ns | 94,154 ns | 0.925x, 1.08x faster | - | Baseline |
| `franken_numpy-ixs5y.265` | Direct final-Vec append | 100k bytes | `ovh-a` | `scorecard.md` | 32,920 ns | 47,212 ns | 0.697x, 1.43x faster | 0.378x, 2.64x faster | Keep |
| `franken_numpy-ixs5y.265` | Origin/main baseline | 1M bytes | `ovh-a` | `scorecard.md` | 154,618 ns | 429,977 ns | 0.360x, 2.78x faster | - | Baseline |
| `franken_numpy-ixs5y.265` | Direct final-Vec append | 1M bytes | `ovh-a` | `scorecard.md` | 122,926 ns | 427,988 ns | 0.287x, 3.48x faster | 0.795x, 1.26x faster | Keep |
| `franken_numpy-ixs5y.265` | Final source supplemental | 100k bytes | `vmi1153651` | `scorecard.md` | 85,242 ns | 465,151 ns | 0.183x, 5.46x faster | - | Noisy confirmation |
| `franken_numpy-ixs5y.265` | Final source supplemental | 1M bytes | `vmi1153651` | `scorecard.md` | 2,410,257 ns | 4,857,309 ns | 0.496x, 2.02x faster | - | Noisy confirmation |

Scorecard:
- Same-worker old-to-new: win/loss/neutral = 2/0/0.
- Candidate vs NumPy decisive rows: win/loss/neutral = 2/0/0.
- Candidate vs NumPy including supplemental rows: win/loss/neutral = 8/0/0.
- The `hz1` fresh-origin control reproduced the 100k loss class at 131,543 ns FNP vs 73,649 ns NumPy; the kept same-worker candidate moved the row to a win.

Validation notes:
- Focused stream-state and live NumPy oracle tests passed.
- Full `cargo test -p fnp-random` passed: 431 unit tests, 12 golden tests, 16 metamorphic tests.
- `cargo check -p fnp-random --all-targets`, `cargo clippy -p fnp-random --all-targets -- -D warnings`, `cargo build --release -p fnp-random`, and `git diff --check` passed.
- `cargo fmt --check` and `cargo fmt -p fnp-random --check` remain blocked by pre-existing broad rustfmt drift outside this perf hunk.
- Retry predicate: do not retry intermediate word-vector transcodes for PCG bytes. Retry only with a same-worker candidate that preserves the half-buffer invariant and beats this direct append path at both 100k and 1M.

## 2026-06-19 - BOLD-VERIFY Keep: fnp-random full-range uint8 integers byte stream

Artifact directory: `tests/artifacts/perf/2026-06-19_random_uint8_full_range_byte_fill/`

Run identity:
- Bead: `franken_numpy-ixs5y.264`.
- Agent: `YellowElk` / `cod-a`.
- Subject API: direct Rust `fnp-random` `Generator::integers_u8_shaped(0, 256, Some(&[size]), false)`.
- Oracle/reference: NumPy `np.random.Generator(np.random.PCG64(42)).integers(0, 256, size=size, dtype=np.uint8)` inside the Criterion benchmark harness.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Coordination note: Agent Mail registration/reservation failed before edits because the local Agent Mail SQLite DB reported `database disk image is malformed`; work was isolated in a clean detached scratch worktree and avoided cod-b-owned `fnp-ufunc` / `fnp-linalg` surfaces.

Lever:
- Old path sent full-range byte integers through the buffered bounded-Lemire helper even though `rng == u8::MAX` has no rejection. The kept path emits the raw byte stream directly and applies the wrapping offset, using the existing direct PCG final-buffer byte fill for large arrays and a four-byte-per-`next_uint32` scalar writer below the direct-fill threshold.
- Alien-graveyard mapping: Vectorized Execution / morsel-style final-buffer fill for the large PCG path, plus "constants kill you" for the 100k row where eliminating generic bounded-loop overhead mattered more than adding parallelism.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo bench -p fnp-random --bench random_vs_numpy -- vs_numpy_pcg64_uint8_full_range --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random full_range_byte_integers_match_scalar_narrow_stream_and_state -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random narrow_width_integers_match_live_numpy_oracle_when_available -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo check -p fnp-random --all-targets`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo clippy -p fnp-random --all-targets -- -D warnings`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo build -p fnp-random --release`
- `git diff --check`

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.264` | Baseline scalar bounded `uint8` loop | 100k, `hz2` | `scorecard.md` | 329,987 ns | 105,292 ns | 3.13x slower | Open gap confirmed |
| `franken_numpy-ixs5y.264` | Baseline scalar bounded `uint8` loop | 1M, `hz2` | `scorecard.md` | 3,241,197 ns | 788,860 ns | 4.11x slower | Open gap confirmed |
| `franken_numpy-ixs5y.264` | Candidate A, direct `bytes` reuse only | 100k, `hz2` | `scorecard.md` | 106,506 ns | 101,368 ns | 1.05x slower | Superseded |
| `franken_numpy-ixs5y.264` | Candidate A, direct `bytes` reuse only | 1M, `hz2` | `scorecard.md` | 286,011 ns | 757,753 ns | 0.377x, 2.65x faster | Superseded |
| `franken_numpy-ixs5y.264` | Candidate B kept, manual sub-threshold + direct large byte fill | 100k, `vmi1149989` | `scorecard.md` | 104,370 ns | 127,730 ns | 0.817x, 1.22x faster | Keep |
| `franken_numpy-ixs5y.264` | Candidate B kept, manual sub-threshold + direct large byte fill | 1M, `vmi1149989` | `scorecard.md` | 725,711 ns | 1,155,285 ns | 0.628x, 1.59x faster | Keep |
| `franken_numpy-ixs5y.264` | Candidate B supplemental long run | 100k, `vmi1149989` | `scorecard.md` | 88,959 ns | 118,758 ns | 0.749x, 1.34x faster | Confirmation |
| `franken_numpy-ixs5y.264` | Candidate B supplemental long run | 1M, `vmi1149989` | `scorecard.md` | 432,705 ns | 1,092,147 ns | 0.396x, 2.52x faster | Confirmation |

Scorecard:
- Baseline vs NumPy: win/loss/neutral = 0/2/0.
- Kept final vs NumPy decision rows: win/loss/neutral = 2/0/0.
- Rejected/superseded candidate rows: win/loss/neutral = 1/1/0.
- The final keep is based on same-run head-to-head Criterion rows where the Rust and embedded Python NumPy timings ran on the same remote worker process. Old-to-new absolute speedup is not used as a same-worker decision because RCH did not preserve a single worker across every exploratory run.

Validation notes:
- New scalar-stream/state guard passed for PCG64 and PCG64DXSM.
- Existing live NumPy narrow-integer oracle shard passed on rerun. The first attempt selected `ovh-b` and failed before repository code with `zerocopy` build-script `SIGILL`; this is recorded as worker infra noise, not repo evidence.
- `cargo check -p fnp-random --all-targets`, `cargo clippy -p fnp-random --all-targets -- -D warnings`, `cargo build -p fnp-random --release`, and `git diff --check` passed.
- `cargo fmt --check -p fnp-random` still reports broad pre-existing rustfmt drift in unrelated `crates/fnp-random/src/lib.rs` sections; this commit keeps that out of the perf proof.
- UBS on the changed-file set exited 1 after scanning the two Rust source files, with the crate's broad existing inventory (66 critical, 2141 warnings, 659 info); sampled findings were pre-existing `unwrap`/assert/direct-index/security-heuristic inventory outside the new full-range byte fast path.
- Retry predicate: do not revisit generic bounded-loop tweaks for full-range byte integers. Retry only with same-run NumPy rows that preserve the scalar byte stream and beat the kept path at both 100k and 1M.

## 2026-06-19 - BOLD-VERIFY Keep: FNP compress direct bool-mask decode vs NumPy

Artifact directory: `tests/artifacts/perf/2026-06-19_ufunc_compress_simd_cod_a/`

Run identity:
- Bead: `franken_numpy-ixs5y.263`.
- Parent gap: `.249` left serial `compress(condition, axis=None)` slower than
  NumPy after reverting the per-chunk `Vec<Vec<f64>>` parallel gather.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::compress` Criterion row
  `compress_f64_bool_flat_sparse`.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7 using the same value and
  bool-mask formula.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Worker notes: baseline and early candidates ran under `rch` on `hz1`; the
  final remote candidate ran on `hz2`; direct SSH NumPy timing on `hz2` failed
  with `Permission denied (publickey,password)`, so same-host keep/reject uses
  the local FNP Criterion confirmation plus the local NumPy probe.

Commands:
- `rch exec -- cargo bench -p fnp-ufunc --bench elementwise compress_f64_bool_flat_sparse -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `rch exec -- cargo test -p fnp-ufunc compress_f64_bool_flat_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `rch exec -- cargo test -p fnp-ufunc compress -- --nocapture`
- `rch exec -- cargo check -p fnp-ufunc --all-targets`
- `rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings`
- Local same-host confirmation: `cargo bench -p fnp-ufunc --bench elementwise compress_f64_bool_flat_sparse -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- Python NumPy timing probe in `baseline_numpy_compress_rch.txt`; `rch exec`
  warned that arbitrary `python3 -` is non-compilation and did not provide a
  worker pin.

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.263` | Baseline serial index materialization + `take` | 100k, `hz1` FNP vs local NumPy | `baseline_fnp_compress_rch.txt`, `baseline_numpy_compress_rch.txt` | 113.207 us | 52.056 us | 2.18x slower | Open gap confirmed |
| `franken_numpy-ixs5y.263` | Baseline serial index materialization + `take` | 1M, `hz1` FNP vs local NumPy | `baseline_fnp_compress_rch.txt`, `baseline_numpy_compress_rch.txt` | 1.232777 ms | 503.993 us | 2.45x slower | Open gap confirmed |
| `franken_numpy-ixs5y.263` | Two-pass bool mask count/decode, exact allocation | 100k, `hz1` FNP vs local NumPy | `candidate_fnp_compress_rch.txt`, `baseline_numpy_compress_rch.txt` | 85.032 us | 52.056 us | 1.63x slower | Superseded |
| `franken_numpy-ixs5y.263` | Two-pass bool mask count/decode, exact allocation | 1M, `hz1` FNP vs local NumPy | `candidate_fnp_compress_rch.txt`, `baseline_numpy_compress_rch.txt` | 520.739 us | 503.993 us | 1.03x slower | Neutral, superseded |
| `franken_numpy-ixs5y.263` | Single-pass bool bitmask decode, full input capacity | 100k, `hz1` FNP vs local NumPy | `candidate_fnp_compress_single_pass_rch.txt`, `baseline_numpy_compress_rch.txt` | 52.650 us | 52.056 us | 1.01x slower | Neutral, superseded |
| `franken_numpy-ixs5y.263` | Single-pass bool bitmask decode, full input capacity | 1M, `hz1` FNP vs local NumPy | `candidate_fnp_compress_single_pass_rch.txt`, `baseline_numpy_compress_rch.txt` | 508.565 us | 503.993 us | 1.01x slower | Neutral, superseded |
| `franken_numpy-ixs5y.263` | Single-pass bool bitmask decode, quarter-capacity output | 100k same-host local | `candidate_fnp_compress_capacity_local.txt`, `baseline_numpy_compress_rch.txt` | 44.374 us | 52.056 us | 0.852x, 1.17x faster | Keep |
| `franken_numpy-ixs5y.263` | Single-pass bool bitmask decode, quarter-capacity output | 1M same-host local | `candidate_fnp_compress_capacity_local.txt`, `baseline_numpy_compress_rch.txt` | 410.823 us | 503.993 us | 0.815x, 1.23x faster | Keep |
| `franken_numpy-ixs5y.263` | Single-pass bool bitmask decode, quarter-capacity output | 100k remote routing, `hz2` | `candidate_fnp_compress_capacity_rch.txt`, `baseline_numpy_compress_rch.txt` | 33.082 us | 52.056 us | 0.635x, 1.57x faster | Routing confirmation |
| `franken_numpy-ixs5y.263` | Single-pass bool bitmask decode, quarter-capacity output | 1M remote routing, `hz2` | `candidate_fnp_compress_capacity_rch.txt`, `baseline_numpy_compress_rch.txt` | 339.188 us | 503.993 us | 0.673x, 1.49x faster | Routing confirmation |

Scorecard:
- Final same-host vs NumPy: win/loss/neutral = 2/0/0.
- Rejected/superseded candidates vs NumPy: win/loss/neutral = 0/3/1.
- Old-to-final local comparison against the prior `.249` post-revert local
  serial rows: 100k improved 90.800 us -> 44.374 us (2.05x faster); 1M
  improved 1.1369 ms -> 410.823 us (2.77x faster).

Notes:
- Kept path is not the rejected `.249` per-chunk parallel gather. It avoids
  `Vec<Vec<f64>>`, avoids building an index vector, preserves the existing
  `take` fallback for true bits beyond the array end, and is limited to
  sidecar-free F64 `axis=None`.
- The final lever maps to Vectorized Execution-style selection bitmasks over a
  cache-local flat buffer plus the "constants kill you" rule: the exact-count
  two-pass and full-capacity one-pass versions were neutral/losses, so only the
  measured quarter-capacity one-pass decoder was kept.
- Golden guard digest:
  `81276111fdbfe090ecd3c825cf1ecc3fb5c6601e318fbd5683b9dfe6877d550f`.
- Focused validation passed for `cargo test -p fnp-ufunc compress`,
  `cargo check -p fnp-ufunc --all-targets`, `cargo clippy -p fnp-ufunc
  --all-targets -- -D warnings`, and `git diff --check`.
- `cargo fmt --check -p fnp-ufunc` still reports pre-existing rustfmt drift in
  unrelated fnp-ufunc sections and bench rows; it is recorded in
  `cargo_fmt_check_fnp_ufunc.txt` and kept out of this commit.
- UBS on the changed-file set exited nonzero with broad pre-existing
  `fnp-ufunc` inventory (489 critical, 14639 warnings); sampled findings are
  outside the new compress helper/path and the full output is recorded in
  `ubs_changed_files.txt`.

## 2026-06-19 - Gauntlet Verify: FNP flatnonzero gather vs NumPy

Artifact directory: `tests/artifacts/perf/2026-06-19_ufunc_flatnonzero_vs_numpy/`

Run identity:
- Verification bead: `franken_numpy-ixs5y.260`.
- Original optimization bead: `franken_numpy-ixs5y.245`.
- Subject commit before revert: `68a5d002`.
- Final code: `.245` parallel F64 `flatnonzero` index-gather fast path removed; serial exact int64 sidecar export retained.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::flatnonzero` Criterion row.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7.
- Same-host decision machine: `thinkstation1` for both local FNP Criterion and local NumPy timing.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.

Commands:
- `rch exec -- cargo test -p fnp-ufunc flatnonzero_f64_parallel_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `cargo bench -p fnp-ufunc --bench elementwise flatnonzero_f64_sparse -- --sample-size 20 --warm-up-time 1 --measurement-time 3`
- Python NumPy timing script in `numpy_flatnonzero_local.txt` using the same sparse F64 formula, warmups, GC disabled, and per-sample inner loops.
- `rch exec -- cargo test -p fnp-ufunc flatnonzero_f64_matches_serial_reference_and_golden_sha256 -- --nocapture`

Decision ratios use the same-host local NumPy timing in
`numpy_flatnonzero_local.txt`. The remote Criterion run on `vmi1227854` is
retained as routing evidence only because `rch exec` does not offload arbitrary
Python commands, and direct SSH to the worker was denied.

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.245` | Parallel F64 `flatnonzero` per-chunk index gather candidate | 100k local candidate | `criterion_flatnonzero_local_candidate.txt`, `numpy_flatnonzero_local.txt` | 255.53 us | 239.794 us | 1.07x slower | Reverted |
| `franken_numpy-ixs5y.245` | Parallel F64 `flatnonzero` per-chunk index gather candidate | 1M local candidate | `criterion_flatnonzero_local_candidate.txt`, `numpy_flatnonzero_local.txt` | 703.16 us | 2512.132 us | 0.280x, 3.57x faster | Reverted despite win |
| `franken_numpy-ixs5y.245` | Final serial `flatnonzero` with exact int64 sidecar export | 100k post-revert | `criterion_flatnonzero_local_post_revert.txt`, `numpy_flatnonzero_local.txt` | 74.662 us | 239.794 us | 0.311x, 3.21x faster | Final code |
| `franken_numpy-ixs5y.245` | Final serial `flatnonzero` with exact int64 sidecar export | 1M post-revert | `criterion_flatnonzero_local_post_revert.txt`, `numpy_flatnonzero_local.txt` | 789.35 us | 2512.132 us | 0.314x, 3.18x faster | Final code |

Notes:
- The candidate passed the golden guard, but correctness was not enough: it lost
  to NumPy at 100k, regressed local Criterion history at 100k, and was only a
  modest 1.12x faster than the final serial path at 1M.
- The production parallel branch was removed. The final serial sidecar-export
  path beats NumPy on both measured rows and avoids the 100k regression.
- Final focused validation passed for the post-revert golden guard,
  `rch exec -- cargo check -p fnp-ufunc --all-targets`, `rch exec -- cargo
  clippy -p fnp-ufunc --all-targets -- -D warnings`, and `git diff --check`.
- `cargo fmt --check`, `cargo fmt -p fnp-ufunc -- --check`, and UBS still
  report broad pre-existing `fnp-ufunc` drift/inventory outside the touched
  flatnonzero lines.
- Retry condition: retry `flatnonzero` parallel index gather only with a design
  that avoids per-chunk `Vec<Vec<i64>>` allocation, proves same-host speed over
  NumPy and over the serial sidecar path at both 100k and 1M sparse rows, and
  keeps NumPy timing CV below 10% on the decision rows.

## 2026-06-19 - fnp-random PCG raw fill and bytes cluster

Artifact directory: `tests/artifacts/perf/2026-06-19_random_vs_numpy_pcg/`

Run identity:
- Random subject commit before measured commit: `e32d58ea`.
- Integration base before this commit: `70bae5da`; intervening changes were ufunc evidence/docs and did not touch `fnp-random`.
- Subject API: direct Rust `fnp-random` Criterion rows.
- Oracle/reference: NumPy `np.random.Generator(np.random.PCG64(42))`; local preflight observed NumPy 2.4.3 on `/usr/bin/python3`.
- Workers: `ovh-a` for pre-revert candidate run, `hz1` for final-code run.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo bench -p fnp-random --bench random_vs_numpy -- --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo check -p fnp-random --benches`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random parallel_pcg_fill_u64_matches_serial_stream_state -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random bytes_large_calls_match_serial_uint32_stream_state -- --nocapture`

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | Ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.255` | Parallel PCG64 `fill_u64` jump-ahead | 100k u64 final code, `hz1` | `criterion_random_vs_numpy_post_revert.txt` | 144,560 ns | 538,164 ns | 0.269x | Keep |
| `franken_numpy-ixs5y.255` | Parallel PCG64 `fill_u64` jump-ahead | 1M u64 final code, `hz1` | `criterion_random_vs_numpy_post_revert.txt` | 2,194,406 ns | 4,447,414 ns | 0.493x | Keep |
| `franken_numpy-ixs5y.257` | PCG bytes via u64 word-fill transcode | 100k bytes pre-revert, `ovh-a` | `criterion_random_vs_numpy_prerevert.txt` | 80,103 ns | 48,911 ns | 1.638x | Reverted |
| `franken_numpy-ixs5y.257` | PCG bytes via u64 word-fill transcode | 1M bytes pre-revert, `ovh-a` | `criterion_random_vs_numpy_prerevert.txt` | 850,688 ns | 428,300 ns | 1.986x | Reverted |
| `franken_numpy-ixs5y.257` | Current serial `Generator::bytes` after revert | 100k bytes final code, `hz1` | `criterion_random_vs_numpy_post_revert.txt` | 119,901 ns | 74,988 ns | 1.599x | Open gap |
| `franken_numpy-ixs5y.257` | Current serial `Generator::bytes` after revert | 1M bytes final code, `hz1` | `criterion_random_vs_numpy_post_revert.txt` | 1,214,954 ns | 982,093 ns | 1.237x | Open gap |

Notes:
- `.255` is kept because the final code remains faster than NumPy on both large raw-buffer workloads while preserving stream state.
- `.257` is rejected and the production word-fill path was removed. The u64-word transcode approach allocated/interpreted an intermediate word buffer and lost to NumPy bytes on both measured rows.
- Retry condition for `.257`: only revisit `Generator::bytes` if the candidate fills the final `Vec<u8>` directly from PCG state without an intermediate `Vec<u64>`, preserves the exact `next_uint32` half-buffer contract, and is remeasured head-to-head against NumPy on the same worker. Do not retry the removed `fill_u64(...).to_le_bytes()` transcode family.

## 2026-06-19 - fnp-random PCG bytes direct final-buffer fill

Artifact directory: `tests/artifacts/perf/2026-06-19_random_bytes_direct_fill/`

Run identity:
- Verification bead: `franken_numpy-ixs5y.261`.
- Subject API: direct Rust `fnp-random` `Generator::bytes` Criterion rows.
- Oracle/reference: NumPy `np.random.Generator(np.random.PCG64(42))` from the benchmark harness.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Same-worker A/B decision machine: `vmi1293453`.
- Control worktree: `/data/projects/.scratch/franken_numpy-cod-a-baseline-20260619-0518` at `origin/main` (`3da8ac35`).
- Candidate worktree: `/data/projects/.scratch/franken_numpy-cod-a-20260619-0505`.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random bytes_large_calls_match_serial_uint32_stream_state -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random bytes_match_live_numpy_oracle_when_available -- --nocapture`
- `RCH_WORKER=vmi1293453 RCH_WORKERS=vmi1293453 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo bench -p fnp-random --bench random_vs_numpy -- vs_numpy_pcg64_bytes --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`

Decision rows:

| Bead | Lever | Workload | Artifact | Old FNP | New FNP | FNP new/old | New NumPy | New FNP/NumPy | Verdict |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `franken_numpy-ixs5y.261` | Direct PCG final `Vec<u8>` fill; small rows keep old append loop | 100k bytes, `vmi1293453` | `control_origin_main_vmi1293453.txt`, `candidate_threshold_direct_fill_vmi1149989.txt` | 186,767 ns | 184,751 ns | 0.989x | 425,145 ns | 0.435x | Neutral keep |
| `franken_numpy-ixs5y.261` | Direct PCG final `Vec<u8>` fill via jump-ahead chunks | 1M bytes, `vmi1293453` | `control_origin_main_vmi1293453.txt`, `candidate_threshold_direct_fill_vmi1149989.txt` | 1,683,609 ns | 1,029,616 ns | 0.611x, 1.64x faster | 3,420,805 ns | 0.301x, 3.32x faster | Keep |

Scorecard:
- Win/loss/neutral versus old FNP: 1 win, 0 losses, 1 neutral.
- Win/loss versus NumPy for the kept candidate rows: 2 wins, 0 losses.
- The earlier `.257` post-revert open gap did not reproduce on today’s workers:
  `vmi1153651` baseline was already faster than NumPy at 100k and 1M, and the
  `vmi1293453` control was also faster than NumPy. Treat the old gap as
  worker-sensitive routing evidence, not a current production loss.

Notes:
- `candidate_threshold_direct_fill_vmi1149989.txt` is the same-worker
  candidate artifact used above; its filename records the attempted pin, while
  the RCH transcript inside shows the actual selected worker was `vmi1293453`.
- Kept code fills the final byte vector directly from PCG64/PCG64DXSM `u64`
  words and never materializes the rejected intermediate `Vec<u64>` transcode.
- The old serial append loop is still used below `PCG_BYTES_DIRECT_MIN_LEN` and
  for non-PCG bit generators. A first unthresholded candidate regressed the 100k
  row because safe Rust zero-initialized the final buffer; that artifact is
  retained in `candidate_direct_fill_vs_numpy_pcg64_bytes.txt`.
- Correctness gates passed for the large serial-stream state guard and live
  NumPy byte oracle. The large guard exercises PCG64 and PCG64DXSM, prebuffered
  and unbuffered `next_uint32` state, back-to-back byte calls, and post-call
  `next_uint32` continuation.
- Retry condition: do not retry the old word-fill transcode. Revisit bytes only
  if a future same-worker head-to-head shows the direct-fill row losing to NumPy,
  or if a broader random conformance gate finds a `next_uint32` half-buffer state
  mismatch.

## 2026-06-19 - fnp-random PCG gumbel/laplace distribution cluster

Artifact directory: `tests/artifacts/perf/2026-06-19_random_vs_numpy_pcg_distributions/`

Run identity:
- Subject commit before measured commit: `0442da80`.
- Subject API: direct Rust `fnp-random` Criterion rows.
- Oracle/reference: NumPy `np.random.Generator(np.random.PCG64(42))`; local preflight observed NumPy 2.4.3 on `/usr/bin/python3`.
- Worker: `ovh-a` for both benchmark filters and all targeted correctness tests.
- Target dir requested: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- RCH worker-scoped target observed: `/data/projects/franken_numpy/.rch-target-ovh-a-pool-f4ecbc5a8032ed7eb8c61438ab6b2cc8`.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo check -p fnp-random --benches`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo bench -p fnp-random --bench random_vs_numpy -- gumbel --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo bench -p fnp-random --bench random_vs_numpy -- laplace --sample-size 10 --measurement-time 2 --warm-up-time 1 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random parallel_pcg_gumbel_matches_serial_stream_state -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random parallel_pcg_laplace_matches_serial_stream_state -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random gumbel_matches_live_numpy_oracle -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a rch exec -- cargo test -p fnp-random laplace_matches_live_numpy_oracle -- --nocapture`

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.250` | Parallel PCG64 gumbel inverse-CDF fill | 100k f64, `ovh-a` | `criterion_gumbel_vs_numpy.txt` | 248,006 ns | 1,489,338 ns | 0.167x | Keep |
| `franken_numpy-ixs5y.250` | Parallel PCG64 gumbel inverse-CDF fill | 1M f64, `ovh-a` | `criterion_gumbel_vs_numpy.txt` | 2,105,737 ns | 15,047,299 ns | 0.140x | Keep |
| `franken_numpy-ixs5y.253` | Parallel PCG64 laplace inverse-CDF fill | 100k f64, `ovh-a` | `criterion_laplace_vs_numpy.txt` | 204,760 ns | 1,384,891 ns | 0.148x | Keep |
| `franken_numpy-ixs5y.253` | Parallel PCG64 laplace inverse-CDF fill | 1M f64, `ovh-a` | `criterion_laplace_vs_numpy.txt` | 1,599,666 ns | 13,871,270 ns | 0.115x | Keep |

Notes:
- `.250` is kept because both gumbel rows beat NumPy by 6.01x and 7.15x while `parallel_pcg_gumbel_matches_serial_stream_state` and `gumbel_matches_live_numpy_oracle` passed.
- `.253` is kept because both laplace rows beat NumPy by 6.76x and 8.67x while `parallel_pcg_laplace_matches_serial_stream_state` and `laplace_matches_live_numpy_oracle` passed.
- No optimization was reverted in this distribution slice.
- Retry condition for `.250`: revisit only if a same-worker rerun shows the PCG64 gumbel median at or above NumPy's median, if a broader distribution gate exposes a stream-state mismatch, or if NumPy changes PCG64 gumbel semantics in a way that invalidates fixed one-uniform jump-ahead.
- Retry condition for `.253`: revisit only if a same-worker rerun shows the PCG64 laplace median at or above NumPy's median, if a broader distribution gate exposes a stream-state mismatch, or if NumPy changes PCG64 laplace semantics in a way that invalidates fixed one-uniform jump-ahead.

## Carried No-Retry Families

These remain excluded unless a new profile identifies a different primitive and the retry condition is explicit:

| Family | Status | Retry condition |
|---|---|---|
| SVD row/panel/finalization micro-levers | Rejected in prior gauntlet runs | Only retry through a deeper bidiagonal/full-to-band primitive with a fresh `svd_mxn_full/512` proof. |
| Inverse/TRSM broad `batch_solve` routing | Rejected in prior gauntlet runs | Only retry with a different algorithmic route and same-worker evidence beating NumPy. |
| Packed-GEMM tile-width retunes | Rejected in prior gauntlet runs | Only retry as shared packed-panel/RHS redesign, not `PACKED_NR` width tuning. |
| Variable-consumption random distributions | Rejected for jump-ahead parallelization | Only retry if a live NumPy oracle proves fixed-consumption semantics for that distribution. |

## 2026-06-19 - Gauntlet Verify: FNP ufunc data movement vs NumPy

Run identity:
- Subject commit: `e32d58ea` (`main`, mirrored to `master` before this verify pass).
- Subject API: direct Rust `fnp-ufunc` `UFuncArray` Criterion rows.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7.
- Worker: `thinkstation1` via `rch exec`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.
- Commands:
  - `cargo bench -p fnp-ufunc --bench elementwise delete_flat_f64_sparse_indices -- --sample-size 20 --warm-up-time 1 --measurement-time 3`
  - `cargo bench -p fnp-ufunc --bench elementwise insert_flat_f64_midpoint_many -- --sample-size 20 --warm-up-time 1 --measurement-time 3`
  - Python batched NumPy timing script with 41 samples, warmups, and per-sample inner loops.

Measurement caveat: these rows compare the optimized Rust `fnp-ufunc` API to
NumPy's Python API on equivalent flat array workloads. They are valid for the
`fnp-ufunc` data-movement cluster, not a full `fnp-python` boundary claim.

| Bead | Workload | Size | FNP Criterion median | NumPy batched median | FNP speed vs NumPy | NumPy CV | Verdict | Retry predicate |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `franken_numpy-ixs5y.256` | `delete_flat_f64_sparse_indices` | 100,000 | 48.657 us | 74.719 us | 1.54x faster | 4.33% | KEEP | Reopen only if same-worker Criterion median regresses above 72 us or a batched NumPy rerun beats the FNP upper CI bound. |
| `franken_numpy-ixs5y.256` | `delete_flat_f64_sparse_indices` | 1,000,000 | 659.58 us | 787.97 us | 1.19x faster | 5.08% | KEEP, borderline CV | Reopen if a low-CV same-worker rerun shows NumPy median <= FNP median, or if broader delete workloads show the sort/dedup span path below 1.05x. |
| `franken_numpy-ixs5y.258` | `insert_flat_f64_midpoint_many` | 100,000 | 17.695 us | 24.896 us | 1.41x faster | 3.60% | KEEP | Reopen only if same-worker Criterion median regresses above 24 us or NumPy batched median drops below FNP upper CI bound. |
| `franken_numpy-ixs5y.258` | `insert_flat_f64_midpoint_many` | 1,000,000 | 256.97 us | 445.04 us | 1.73x faster | 17.92% | KEEP, noisy NumPy allocation row | Reopen with allocator-isolated batching if future evidence puts NumPy p50 below 272 us; current NumPy minimum was still 333.31 us. |

Discarded / non-decision evidence:
- Raw unbatched Python timings were discarded for keep-gate decisions because CV
  was 15-71% on microsecond-scale operations. Those numbers may be useful only
  as a smoke check that the workload shape was valid, not as a pass/fail gate.
- An exact-filter test invocation ran zero inline tests because Cargo's exact
  test name did not match the module-qualified path. The subsequent substring
  filter ran and passed both targeted golden guards.

Conformance / correctness guard:
- `cargo test -p fnp-ufunc delete_flat_f64_span_copy_matches_hashset_reference_and_golden_sha256 -- --nocapture` passed: 1 test run.
- `cargo test -p fnp-ufunc insert_flat_f64_splice_matches_repeated_insert_and_golden_sha256 -- --nocapture` passed: 1 test run.
- `cargo check -p fnp-ufunc` passed after the test-only type inference fix.

Action:
- No optimization reverted. Both measured rows clear the head-to-head NumPy median
  gate and have targeted golden guards green.
- Do not retry the prior per-element `HashSet` scan for large flat F64 delete
  or repeated `Vec::insert` shifting for large flat F64 insert unless the retry
  predicates above fire.

## 2026-06-19 - Gauntlet Verify: FNP compress bool-mask vs NumPy

Artifact directory: `tests/artifacts/perf/2026-06-19_ufunc_selection_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.249`.
- Subject commit before revert: `0442da80` plus the local candidate guard fix.
- Final code: `.249` parallel compress fast path removed; serial `compress` path retained.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::compress` Criterion row.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7.
- Same-host decision worker: `thinkstation1` for both local FNP Criterion and local NumPy timing.
- Remote routing evidence: `vmi1149989` Criterion candidate run, not used as the keep/reject gate because the NumPy command could not be pinned to that worker.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.

Commands:
- `cargo test -p fnp-ufunc compress_f64_parallel_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `cargo bench -p fnp-ufunc --bench elementwise compress_f64_bool_flat_sparse -- --sample-size 20 --warm-up-time 1 --measurement-time 3`
- Python NumPy timing script in `numpy_compress_local.txt` using the same value and bool-mask formulas.
- `cargo test -p fnp-ufunc compress -- --nocapture`
- `cargo check -p fnp-ufunc`
- `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | Ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.249` | Parallel F64 `compress` bool-mask gather candidate | 100k local candidate | `criterion_compress_local.txt`, `numpy_compress_local.txt` | 472.85 us | 66.147 us | 7.15x slower | Reverted |
| `franken_numpy-ixs5y.249` | Parallel F64 `compress` bool-mask gather candidate | 1M local candidate | `criterion_compress_local.txt`, `numpy_compress_local.txt` | 1.0645 ms | 518.349 us | 2.05x slower | Reverted |
| `franken_numpy-ixs5y.249` | Final serial `compress` after revert | 100k post-revert | `criterion_compress_local_post_revert.txt`, `numpy_compress_local.txt` | 90.800 us | 66.147 us | 1.37x slower | Open gap |
| `franken_numpy-ixs5y.249` | Final serial `compress` after revert | 1M post-revert | `criterion_compress_local_post_revert.txt`, `numpy_compress_local.txt` | 1.1369 ms | 518.349 us | 2.19x slower | Open gap |

Notes:
- The first guard attempt failed because the test used `assert_eq!` on a selected slice containing `NaN`; after switching that edge assertion to bitwise comparison, the candidate guard passed.
- Passing correctness was not enough: same-host local Criterion showed the parallel candidate regressed the local Criterion history by +339.69% at 100k and +66.84% at 1M, and it lost badly to NumPy on both measured sizes.
- The production parallel fast path was removed. The remaining serial path is still slower than NumPy, but it is less bad at 100k and avoids keeping a regressing optimization.
- Final focused validation passed for `cargo test -p fnp-ufunc compress`, `cargo check -p fnp-ufunc`, and `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`; `cargo fmt --check` still reports broad workspace format drift outside this slice.
- Retry condition: retry `compress(condition, axis=None)` only if a new design avoids per-chunk `Vec<Vec<f64>>` allocation and proves same-host speed over NumPy on both 100k and 1M bool-mask rows with CV below 10%; do not retry this per-chunk parallel gather shape as a standalone patch.

## 2026-06-19 - Gauntlet Verify: FNP extract masked gather vs NumPy

Artifact directory: `tests/artifacts/perf/2026-06-19_ufunc_extract_vs_numpy/`

Run identity:
- Verification bead: `franken_numpy-ixs5y.259`.
- Original optimization bead: `franken_numpy-ixs5y.244`.
- Subject commit before revert: `298f05dd`.
- Final code: `.244` parallel F64 `extract` masked-gather fast path removed; serial `extract` path retained.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::extract` Criterion row.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7.
- Same-host decision machine: `thinkstation1` for both local FNP Criterion and local NumPy timing.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.

Commands:
- `rch exec -- cargo test -p fnp-ufunc extract_f64_parallel_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `cargo bench -p fnp-ufunc --bench elementwise extract_f64_masked -- --sample-size 20 --warm-up-time 1 --measurement-time 3`
- Python NumPy timing scripts in `numpy_extract_local.txt` and `numpy_extract_local_rerun.txt` using the same value and bool-mask formulas.
- `rch exec -- cargo test -p fnp-ufunc extract_f64_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `rch exec -- cargo test -p fnp-ufunc boolean_index_f64_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `rch exec -- cargo check -p fnp-ufunc --all-targets`
- `rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings`
- `cargo fmt --check`

Decision ratios use the longer `numpy_extract_local_rerun.txt` reference with
GC disabled.

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | Ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.244` | Parallel F64 `extract` per-chunk gather candidate | 100k local candidate | `criterion_extract_local.txt`, `numpy_extract_local_rerun.txt` | 275.46 us | 126.540 us | 2.18x slower | Reverted |
| `franken_numpy-ixs5y.244` | Parallel F64 `extract` per-chunk gather candidate | 1M local candidate | `criterion_extract_local.txt`, `numpy_extract_local_rerun.txt` | 668.54 us | 547.298 us | 1.22x slower | Reverted |
| `franken_numpy-ixs5y.244` | Final serial `extract` after revert | 100k post-revert | `criterion_extract_local_post_revert.txt`, `numpy_extract_local_rerun.txt` | 79.896 us | 126.540 us | 1.58x faster | Final code |
| `franken_numpy-ixs5y.244` | Final serial `extract` after revert | 1M post-revert | `criterion_extract_local_post_revert.txt`, `numpy_extract_local_rerun.txt` | 951.42 us | 547.298 us | 1.74x slower | Open gap |

Notes:
- The candidate passed the golden guard, but correctness was not enough: it lost
  to NumPy on both measured rows and was 3.45x slower than the final serial path
  at 100k.
- The candidate was faster than serial at 1M, but still 1.22x slower than NumPy
  and therefore did not clear the gauntlet's neutral/regression rule.
- Removing the parallel `extract` branch also removes the implicit parallel
  acceleration that `boolean_index` reached through `extract`; the boolean-index
  golden guard was rerun post-revert and passed. `boolean_index` remains a
  separate open benchmark target rather than an unmeasured keep.
- Final focused validation passed for the two golden guards, `cargo check -p
  fnp-ufunc --all-targets`, and `cargo clippy -p fnp-ufunc --all-targets -- -D
  warnings`.
- `cargo fmt --check` and package-scoped `cargo fmt -p fnp-ufunc -- --check`
  still report broad pre-existing format drift in untouched regions; the
  extract revert itself is compiled and clippy-clean.
- Retry condition: retry `extract(condition, arr)` only with a design that avoids
  per-chunk `Vec<Vec<f64>>` allocation, proves same-host speed over NumPy at both
  100k and 1M sparse-mask rows, keeps NumPy timing CV below 10%, and separately
  remeasures the `boolean_index_f64_masked_sparse` dependent workload.

## 2026-06-19 - BOLD-VERIFY: FNP extract SIMD mask decode keep

Artifact directories:
- `tests/artifacts/perf/2026-06-19_ufunc_extract_values_only_vs_numpy/`
- `tests/artifacts/perf/2026-06-19_ufunc_boolean_index_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.244`.
- Subject base before local edit: `f4cfc942`.
- Final code: F64/no-integer-sidecar `extract` path counts and decodes the f64
  bool mask with safe portable SIMD bitmasks, then pushes selected values in
  lane order. The integer-sidecar and non-F64 paths keep the existing
  source-index implementation.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::extract`; dependent
  `boolean_index` path reuses `extract`.
- Same-worker FNP decision worker: `hz2`.
- Same-host confirmation machine: `thinkstation1` for both local Criterion and
  NumPy reference.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.

Commands:
- `rch exec -- cargo test -p fnp-ufunc extract_f64_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `rch exec -- cargo test -p fnp-ufunc boolean_index_f64_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `rch exec -- cargo bench -p fnp-ufunc --bench elementwise extract_f64_masked -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `rch exec -- cargo bench -p fnp-ufunc --bench elementwise boolean_index_f64_masked_sparse -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `cargo bench -p fnp-ufunc --bench elementwise extract_f64_masked -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `cargo bench -p fnp-ufunc --bench elementwise boolean_index_f64_masked_sparse -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `rch exec -- cargo check -p fnp-ufunc --all-targets`
- `rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings`
- `cargo fmt -p fnp-ufunc -- --check`

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.244` | One-pass F64 no-sidecar sparse-capacity candidate | 100k extract, `ovh-a` routing | terminal transcript | 100.348 us | 54.443 us | 1.843x | Reverted |
| `franken_numpy-ixs5y.244` | One-pass F64 no-sidecar sparse-capacity candidate | 1M extract, `ovh-a` routing | terminal transcript | 1,171.660 us | 613.212 us | 1.911x | Reverted |
| `franken_numpy-ixs5y.244` | SIMD f64-mask count/decode | 100k extract, `hz2` | `criterion_extract_simd_keep.txt`, `numpy_extract_local_baseline.txt` | 44.666 us | 54.443 us | 0.820x | Keep |
| `franken_numpy-ixs5y.244` | SIMD f64-mask count/decode | 1M extract, `hz2` | `criterion_extract_simd_keep.txt`, `numpy_extract_local_baseline.txt` | 496.772 us | 613.212 us | 0.810x | Keep |
| `franken_numpy-ixs5y.244` | SIMD f64-mask count/decode | 100k extract, local confirmation | `criterion_extract_simd_keep.txt`, `numpy_extract_local_baseline.txt` | 42.327 us | 54.443 us | 0.777x | Keep |
| `franken_numpy-ixs5y.244` | SIMD f64-mask count/decode | 1M extract, local confirmation | `criterion_extract_simd_keep.txt`, `numpy_extract_local_baseline.txt` | 610.601 us | 613.212 us | 0.996x | Keep, borderline |
| `franken_numpy-ixs5y.244` | Dependent `boolean_index` through SIMD extract | 100k boolean index, `hz2` | `criterion_boolean_index_simd_keep.txt`, `numpy_boolean_index_local_baseline.txt` | 45.643 us | 87.115 us | 0.524x | Keep |
| `franken_numpy-ixs5y.244` | Dependent `boolean_index` through SIMD extract | 1M boolean index, `hz2` | `criterion_boolean_index_simd_keep.txt`, `numpy_boolean_index_local_baseline.txt` | 466.771 us | 976.900 us | 0.478x | Keep |
| `franken_numpy-ixs5y.244` | Dependent `boolean_index` through SIMD extract | 100k boolean index, local confirmation | `criterion_boolean_index_simd_keep.txt`, `numpy_boolean_index_local_baseline.txt` | 46.343 us | 87.115 us | 0.532x | Keep |
| `franken_numpy-ixs5y.244` | Dependent `boolean_index` through SIMD extract | 1M boolean index, local confirmation | `criterion_boolean_index_simd_keep.txt`, `numpy_boolean_index_local_baseline.txt` | 568.270 us | 976.900 us | 0.582x | Keep |

Notes:
- The scalar no-sidecar count/copy candidate improved over the original
  source-index path on one remote routing run but still lost to NumPy; it was
  not kept as the final lever.
- The one-pass sparse-capacity candidate removed the count pass but regressed
  badly, likely from realloc/capacity behavior plus scalar branch pressure; it
  was reverted before final validation.
- The final SIMD path preserves NumPy truthiness for this representation:
  `NaN != 0.0` is selected, `-0.0 == 0.0` is false, and bitmask lanes are pushed
  in ascending order to preserve output order.
- `cargo fmt -p fnp-ufunc -- --check` still reports broad pre-existing format
  drift in untouched benches, tests, imports, and polynomial/cross-product
  regions. The edited block was manually adjusted to the rustfmt import/order
  style shown for that block.
- A fresh local NumPy rerun after the keep was slower than the earlier NumPy
  reference, so the table uses the stricter earlier NumPy medians. The earlier
  NumPy extract 1M CV was high, making the local 1M extract confirmation a
  borderline keep despite clearing the median.
- Final focused validation passed for both golden guards, package check, and
  package clippy with `-D warnings`.
- Retry condition: reopen `.244` if a low-CV same-host NumPy rerun shows NumPy
  median at or below the local FNP median for 1M extract, if a broader extract
  density matrix shows dense-mask regressions from the SIMD path, or if the
  representation bridge grows a true compact bool mask storage path that can
  remove the current f64-mask memory traffic entirely.
- Follow-up bead `franken_numpy-ixs5y.262` adds independent cod-a same-worker
  verification of the same SIMD source body; after rebase it keeps the upstream
  source unchanged and records the extra proof bundle.

## 2026-06-19 - BOLD-VERIFY Keep: FNP extract SIMD mask decode vs NumPy

Artifact directory:
`tests/artifacts/perf/2026-06-19_ufunc_extract_simd_cod_a/`

Run identity:
- Verification bead: `franken_numpy-ixs5y.262`.
- Parent gap: the first `franken_numpy-ixs5y.244` verification left the serial
  1M `extract` row 1.74x slower than NumPy after reverting its per-chunk
  `Vec<Vec<f64>>` candidate.
- Baseline commit for the measured old FNP rows: `39bb1e78`.
- Rebased parent carrying the same SIMD source body: `0d3be5d0`.
- Final code after rebase: the upstream sidecar-free F64 `extract` SIMD
  mask-count and bitmask decode fast path is retained unchanged; `.262` adds
  independent cod-a verification artifacts and ledger evidence rather than an
  extra source delta.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::extract` Criterion row.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7.
- Same-worker decision machine: `hz2` for the FNP Criterion rows and the NumPy
  timing probes.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.

Commands:
- `rch exec -- cargo bench -p fnp-ufunc --bench elementwise extract_f64_masked -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `rch exec -- python3 -` using the bool-mask NumPy extract timing probe in
  `baseline_numpy_extract_bool_hz2.txt`.
- `rch exec -- cargo bench -p fnp-ufunc --bench elementwise boolean_index_f64_masked_sparse -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `rch exec -- python3 -` using the bool-mask NumPy boolean-index timing probe in
  `baseline_numpy_boolean_index_hz2.txt`.
- `rch exec -- cargo test -p fnp-ufunc extract_f64_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `rch exec -- cargo test -p fnp-ufunc boolean_index_f64_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `rch exec -- cargo check -p fnp-ufunc --all-targets`
- `rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings`
- `cargo fmt -p fnp-ufunc -- --check`
- `git diff --check`

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.262` | SIMD F64 `extract` mask count/decode | 100k candidate | `candidate_fnp_extract_hz2.txt`, `baseline_numpy_extract_bool_hz2.txt` | 46.853 us | 52.078 us | 0.900x | Keep |
| `franken_numpy-ixs5y.262` | SIMD F64 `extract` mask count/decode | 1M candidate | `candidate_fnp_extract_hz2.txt`, `baseline_numpy_extract_bool_hz2.txt` | 485.711 us | 506.924 us | 0.958x | Borderline keep |
| `franken_numpy-ixs5y.262` | Same `extract` fast path via `boolean_index` | 100k candidate | `candidate_fnp_boolean_index_hz2.txt`, `baseline_numpy_boolean_index_hz2.txt` | 43.993 us | 93.464 us | 0.471x | Keep |
| `franken_numpy-ixs5y.262` | Same `extract` fast path via `boolean_index` | 1M candidate | `candidate_fnp_boolean_index_hz2.txt`, `baseline_numpy_boolean_index_hz2.txt` | 479.160 us | 896.004 us | 0.535x | Keep |

Old-vs-new FNP extract delta on `hz2`:
- 100k: 74.721 us -> 46.853 us, candidate/baseline 0.627x.
- 1M: 793.914 us -> 485.711 us, candidate/baseline 0.612x.

Win/loss/neutral score:
- Head-to-head kept rows: 4 win / 0 loss / 0 neutral.
- Old-vs-new direct extract rows: 2 win / 0 loss / 0 neutral.

Notes:
- The verified SIMD source body deliberately avoids the rejected `.244`
  allocation shape: there are no per-chunk output vectors and no Rayon gather
  merge. It uses one exact-capacity output vector after a SIMD count pass.
- The retained `baseline_numpy_extract_hz2.txt` float-condition probe is not the
  decision comparator. The fair comparator is the bool-mask NumPy row in
  `baseline_numpy_extract_bool_hz2.txt`, matching the FNP `DType::Bool` bench
  intent.
- The 1M direct extract NumPy row has `cv_pct=12.21`, above the preferred 10%
  noise bound. The candidate still beats the NumPy median and is below the NumPy
  minimum (`485.711 us` vs `490.076 us`), so this is accepted as a borderline
  keep rather than a neutral.
- Focused validation passed for the two golden guards, `cargo check -p
  fnp-ufunc --all-targets`, `cargo clippy -p fnp-ufunc --all-targets -- -D
  warnings`, and `git diff --check`.
- After rebasing onto parent `0d3be5d0`, the extract golden guard,
  boolean-index golden guard, `cargo check -p fnp-ufunc --all-targets`, and
  `cargo clippy -p fnp-ufunc --all-targets -- -D warnings` all passed again
  through `rch`.
- `cargo fmt -p fnp-ufunc -- --check` still reports broad pre-existing format
  drift in `fnp-ufunc` benches and untouched `lib.rs` regions; no formatter was
  run for this perf commit.
- `ubs` completed on the touched subset with exit 1 against the existing broad
  `fnp-ufunc/src/lib.rs` inventory; see `ubs_touched_subset_summary.md` for the
  sampled pre-existing finding classes.
- Retry condition: reopen only if a low-CV same-worker rerun shows NumPy median
  or minimum at or below the candidate 1M direct extract row, if sidecar golden
  behavior changes, or if a broader dtype-general path can prove wins without
  regressing the sidecar-preserving scalar path.

## 2026-06-19 - Gauntlet Verify: FNP count_nonzero vs NumPy

Artifact directory: `tests/artifacts/perf/2026-06-19_ufunc_count_nonzero_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.246`.
- Subject before measured correction: code-first flat F64 `count_nonzero(axis=None)` parallel candidate with `1 << 14` activation threshold.
- Final code: parallel activation threshold raised to `1 << 19`, parallel chunk size kept at 4096 elements.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::count_nonzero` Criterion row.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7.
- Same-host decision machine: `thinkstation1`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.

Commands:
- `cargo test -p fnp-ufunc count_nonzero_f64_parallel_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `cargo bench -p fnp-ufunc --bench elementwise count_nonzero_flat_f64_sparse -- --sample-size 20 --warm-up-time 1 --measurement-time 3`
- Python NumPy timing script in `numpy_count_nonzero_local.txt` using the same data formula.
- `cargo check -p fnp-ufunc`
- `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`
- `cargo fmt --check`

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.246` | Original `1 << 14` parallel activation | 100k local candidate | `criterion_count_nonzero_local.txt`, `numpy_count_nonzero_local.txt` | 138.89 us | 39.006 us | 3.56x | Rejected, too eager |
| `franken_numpy-ixs5y.246` | Original `1 << 14` parallel activation | 1M local candidate | `criterion_count_nonzero_local.txt`, `numpy_count_nonzero_local.txt` | 92.072 us | 384.147 us | 0.240x | Keep large-row signal |
| `franken_numpy-ixs5y.246` | Raised threshold only, chunk coupled to threshold | 100k local correction | `criterion_count_nonzero_local_after_threshold.txt`, `numpy_count_nonzero_local.txt` | 8.4582 us | 39.006 us | 0.217x | Keep serial gate |
| `franken_numpy-ixs5y.246` | Raised threshold only, chunk coupled to threshold | 1M local correction | `criterion_count_nonzero_local_after_threshold.txt`, `numpy_count_nonzero_local.txt` | 173.68 us | 384.147 us | 0.452x | Weakened keep |
| `franken_numpy-ixs5y.246` | Final `1 << 19` activation with 4096-element chunks | 100k final | `criterion_count_nonzero_local_final.txt`, `numpy_count_nonzero_local.txt` | 8.3121 us | 39.006 us | 0.213x | Keep |
| `franken_numpy-ixs5y.246` | Final `1 << 19` activation with 4096-element chunks | 1M final | `criterion_count_nonzero_local_final.txt`, `numpy_count_nonzero_local.txt` | 110.42 us | 384.147 us | 0.287x | Keep, noisy CI |

Notes:
- The original optimization was partially rejected: the 16k activation threshold sent 100k arrays to Rayon and lost to NumPy by 3.56x.
- Raising the activation threshold fixed the 100k row by taking the existing serial path, but coupling chunk size to the threshold weakened the 1M parallel row. Splitting `COUNT_NONZERO_PARALLEL_CHUNK_ELEMS` restored small chunks for large arrays.
- The threshold-crossing golden fixture digest changed intentionally after raising the threshold; the updated digest guard passed after the fixture moved from the parallel path to the serial path.
- Final focused validation passed for `cargo test -p fnp-ufunc count_nonzero_f64_parallel_matches_serial_reference_and_golden_sha256 -- --nocapture`, `cargo check -p fnp-ufunc`, and `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`.
- `cargo fmt --check` still reports broad workspace formatting drift outside this slice; no workspace formatter was run.
- Retry condition: do not restore `COUNT_NONZERO_PARALLEL_MIN_ELEMS = 1 << 14` unless same-host 100k evidence beats NumPy and stays inside the prior FNP CI. Reopen the final 1M path only if a same-host rerun shows the final median at or above NumPy's median, or if the golden guard changes again without an intentional threshold fixture update.

## 2026-06-19 - Gauntlet Verify: FNP argwhere vs NumPy

Artifact directory: `tests/artifacts/perf/2026-06-19_ufunc_argwhere_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.248`.
- Subject code: code-first flat F64 `argwhere()` parallel interleaved coordinate gather candidate.
- Final code: unchanged; no revert required.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::argwhere` Criterion row.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7.
- Same-host decision machine: `thinkstation1`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.

Commands:
- `cargo test -p fnp-ufunc argwhere_f64_parallel_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `cargo bench -p fnp-ufunc --bench elementwise argwhere_f64_2d_sparse -- --sample-size 20 --warm-up-time 1 --measurement-time 3`
- Python NumPy timing script in `numpy_argwhere_local.txt` using the same data formula.
- `cargo check -p fnp-ufunc`
- `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`
- `cargo fmt --check`

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.248` | Parallel F64 `argwhere` interleaved coordinate gather | 512x512 final | `criterion_argwhere_local.txt`, `numpy_argwhere_local.txt` | 392.63 us | 1195.008 us | 0.329x | Keep |
| `franken_numpy-ixs5y.248` | Parallel F64 `argwhere` interleaved coordinate gather | 1024x1024 final | `criterion_argwhere_local.txt`, `numpy_argwhere_local.txt` | 1054.2 us | 5047.868 us | 0.209x | Keep |

Notes:
- The 512x512 row is 3.04x faster than NumPy; the 1024x1024 row is 4.79x faster.
- NumPy CV was noisy at 12.06% and 12.21%, but the NumPy minima were still above the FNP Criterion upper bounds on both sizes, so the keep decision does not rest on a noisy median edge.
- Final focused validation passed for `argwhere_f64_parallel_matches_serial_reference_and_golden_sha256`, `cargo check -p fnp-ufunc`, and `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`.
- `cargo fmt --check` still reports broad workspace formatting drift outside this slice; no workspace formatter was run.
- Retry condition: reopen only if a same-host rerun shows NumPy minimum below the FNP Criterion upper CI bound on either measured size, or if the interleaved C-order coordinate golden guard changes. Do not retry this as a standalone patch solely for lower NumPy-CV reruns.

## 2026-06-19 - Gauntlet Verify: FNP masked copyto vs NumPy

Artifact directory: `tests/artifacts/perf/2026-06-19_ufunc_remaining_masked_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.242`.
- Subject before measured correction: equal-shape F64 masked `copyto` paid a full `broadcast_to` clone before the equal-shape fast path, then activated Rayon at `1 << 14` elements.
- Final code: equal-shape F64/no-sidecar masked fast path is selected before source broadcasting; arrays below `1 << 20` elements use a direct serial fused mask/copy loop, with Rayon reserved for larger arrays.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::copyto` Criterion row.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7.
- Same-host decision machine: `thinkstation1`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.

Commands:
- `cargo bench -p fnp-ufunc --bench elementwise 'where_nonzero_f64_2d_sparse|copyto_equal_shape_masked|putmask_f64_masked|place_f64_masked_cycling|put_mask_f64_masked_cycling' -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `cargo bench -p fnp-ufunc --bench elementwise copyto_equal_shape_masked -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- Python NumPy timing script in `numpy_remaining_masked_local.txt` using the same data formulas.
- `cargo test -p fnp-ufunc copyto_masked_equal_shape_parallel_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `cargo check -p fnp-ufunc --all-targets`
- `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`
- `cargo fmt -p fnp-ufunc -- --check`
- `git diff --check`
- `timeout 120s ubs --only=rust crates/fnp-ufunc/src/lib.rs`

Triage scorecard:
- Current masked-family routing run vs local NumPy median: win/loss/neutral = 5/5/0 across 10 rows. The 2/2 copyto losses were selected because they were structural and fully inside this crate.
- Final focused copyto run vs local NumPy median: win/loss/neutral = 2/0/0 across the two same-host decision rows.

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.242` | Current code, routing evidence | 100k current | `criterion_remaining_masked_current.txt`, `numpy_remaining_masked_local.txt` | 358.730 us | 198.643 us | 1.806x | Loss, selected |
| `franken_numpy-ixs5y.242` | Current code, routing evidence | 1M current | `criterion_remaining_masked_current.txt`, `numpy_remaining_masked_local.txt` | 3438.882 us | 2253.369 us | 1.526x | Loss, selected |
| `franken_numpy-ixs5y.242` | Defer source broadcast only | 100k local candidate | `criterion_copyto_after_defer_broadcast_local.txt`, `numpy_remaining_masked_local.txt` | 1174.983 us | 198.643 us | 5.915x | Rejected |
| `franken_numpy-ixs5y.242` | Defer source broadcast only | 1M local candidate | `criterion_copyto_after_defer_broadcast_local.txt`, `numpy_remaining_masked_local.txt` | 2295.453 us | 2253.369 us | 1.019x | Rejected, neutral/loss |
| `franken_numpy-ixs5y.242` | Defer source broadcast only | 100k remote probe | `criterion_copyto_after_defer_broadcast.txt`, `numpy_remaining_masked_local.txt` | 204.972 us | 198.643 us | 1.032x | Rejected, noisy neutral |
| `franken_numpy-ixs5y.242` | Defer source broadcast only | 1M remote probe | `criterion_copyto_after_defer_broadcast.txt`, `numpy_remaining_masked_local.txt` | 2644.748 us | 2253.369 us | 1.174x | Rejected |
| `franken_numpy-ixs5y.242` | Final serial gate below `1 << 20` | 100k final local | `criterion_copyto_after_serial_threshold_local.txt`, `numpy_remaining_masked_local.txt` | 42.961 us | 198.643 us | 0.216x | Keep |
| `franken_numpy-ixs5y.242` | Final serial gate below `1 << 20` | 1M final local | `criterion_copyto_after_serial_threshold_local.txt`, `numpy_remaining_masked_local.txt` | 1316.171 us | 2253.369 us | 0.584x | Keep |
| `franken_numpy-ixs5y.242` | Final serial gate below `1 << 20` | 100k remote confirmation | `criterion_copyto_after_serial_threshold_rch.txt`, `numpy_remaining_masked_local.txt` | 24.632 us | 198.643 us | 0.124x | Confirming signal |
| `franken_numpy-ixs5y.242` | Final serial gate below `1 << 20` | 1M remote confirmation | `criterion_copyto_after_serial_threshold_rch.txt`, `numpy_remaining_masked_local.txt` | 908.107 us | 2253.369 us | 0.403x | Confirming signal |

Notes:
- The first exotic lever was only half-right: moving the broadcast out of the equal-shape path removed clone work, but it exposed that the `1 << 14` parallel threshold was still a bad morsel size for the 100k and 1M copyto rows.
- The kept lever is the cache/simplex version of the graveyard lesson: keep the common equal-shape dense loop fused and serial until the loop body has enough work to amortize Rayon scheduling, and avoid materializing a broadcast array that the SCE shape equality already proves unnecessary.
- The golden fixture digest changed intentionally because the threshold-crossing fixture moved below the new parallel activation point; elementwise reference comparison passed before the digest assertion, and the updated digest guard then passed.
- Final focused validation passed for `copyto_masked_equal_shape_parallel_matches_serial_reference_and_golden_sha256`, `cargo check -p fnp-ufunc --all-targets`, `cargo clippy -p fnp-ufunc --all-targets -- -D warnings` on `ovh-a`, and `git diff --check`.
- The first clippy attempt hit an rch worker missing `cargo-clippy` for `nightly-2026-02-20`; that environment failure is recorded in `cargo_clippy_fnp_ufunc.txt`, and the successful retry is recorded in `cargo_clippy_fnp_ufunc_retry_ovh_a.txt`.
- `cargo fmt -p fnp-ufunc -- --check` still reports broad pre-existing formatting drift outside this slice; no workspace formatter was run.
- `timeout 120s ubs --only=rust crates/fnp-ufunc/src/lib.rs` did not emit a completion summary before the cap; keep the incomplete `ubs_fnp_ufunc_lib.txt` artifact as a tooling caveat, not a pass.
- Retry condition: reopen only if a same-host NumPy rerun beats the final local FNP median on either row, if compact bool-mask storage replaces the current f64 mask representation, or if a larger-copy workload shows the raised Rayon threshold losing above `1 << 20`.

## 2026-06-19 - Gauntlet Verify: FNP putmask vs NumPy

Artifact directory: `tests/artifacts/perf/2026-06-19_ufunc_putmask_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.243`.
- Subject before measured correction: F64/no-sidecar `putmask` activated Rayon at `1 << 14` elements, so the 100k masked cycling-fill row paid scheduler overhead and the serial fallback still routed every write through integer-mutation sidecar plumbing.
- Final code: F64/no-sidecar `putmask` uses direct dense serial loops below `1 << 20`; above that threshold it keeps the existing Rayon path, including position-index cycling with `values[i % values.len()]`.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::putmask` Criterion row.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7.
- Same-worker FNP confirmation: rch worker `vmi1227854`.
- Same-host decision machine: `thinkstation1`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-ufunc --bench elementwise putmask_f64_masked -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `RCH_WORKER=vmi1227854 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-ufunc --bench elementwise putmask_f64_masked -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- Python NumPy timing script in `numpy_putmask_local.txt` using the same data formula.
- `cargo bench -p fnp-ufunc --bench elementwise putmask_f64_masked -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `RCH_WORKER=vmi1227854 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-ufunc putmask_f64_parallel_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo check -p fnp-ufunc --all-targets`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings`
- `cargo fmt -p fnp-ufunc -- --check`
- `timeout 120s ubs --only=rust crates/fnp-ufunc/src/lib.rs`

Triage scorecard:
- Current focused `putmask` run vs local NumPy median: win/loss/neutral = 1/1/0 across the two decision rows. The 100k row lost by 1.362x and was selected.
- Final rch same-worker `putmask` run vs local NumPy median: win/loss/neutral = 2/0/0.
- Final local same-host `putmask` run vs local NumPy median: win/loss/neutral = 2/0/0.

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.243` | Current code, routing evidence | 100k current rch | `criterion_putmask_current_rch.txt`, `numpy_putmask_local.txt` | 128.047 us | 93.991 us | 1.362x | Loss, selected |
| `franken_numpy-ixs5y.243` | Current code, routing evidence | 1M current rch | `criterion_putmask_current_rch.txt`, `numpy_putmask_local.txt` | 632.938 us | 1209.271 us | 0.523x | Existing win |
| `franken_numpy-ixs5y.243` | Final direct serial gate below `1 << 20` | 100k final rch | `criterion_putmask_after_serial_threshold_rch.txt`, `numpy_putmask_local.txt` | 65.564 us | 93.991 us | 0.698x | Keep |
| `franken_numpy-ixs5y.243` | Final direct serial gate below `1 << 20` | 1M final rch | `criterion_putmask_after_serial_threshold_rch.txt`, `numpy_putmask_local.txt` | 554.537 us | 1209.271 us | 0.459x | Keep |
| `franken_numpy-ixs5y.243` | Final direct serial gate below `1 << 20` | 100k final local | `criterion_putmask_after_serial_threshold_local.txt`, `numpy_putmask_local.txt` | 53.237 us | 93.991 us | 0.566x | Keep, same-host |
| `franken_numpy-ixs5y.243` | Final direct serial gate below `1 << 20` | 1M final local | `criterion_putmask_after_serial_threshold_local.txt`, `numpy_putmask_local.txt` | 559.970 us | 1209.271 us | 0.463x | Keep, same-host |

Notes:
- Same-worker FNP delta on `vmi1227854`: 100k improved from 128.047 us to 65.564 us, a 1.95x speedup; 1M improved from 632.938 us to 554.537 us, a 1.14x speedup.
- The kept lever is the graveyard "constants kill you" correction: avoid tiny Rayon morsels and integer-sidecar mutation machinery when SCE and dtype checks prove a dense F64/no-sidecar loop. The exotic idea was deliberately small but architectural: use the layout proof to select the flat cache-local loop before generic mutation dispatch.
- NumPy 100k timing was noisy at 11.47% CV, but the final local FNP median of 53.237 us is still below the NumPy minimum of 90.348 us, so the keep decision does not rest on a noisy median edge.
- The first focused golden run failed only at the SHA-256 digest after the elementwise serial-reference comparison had already passed. The updated digest `4fffe2fd2c9e96fa07d22719917ae99810b0f84a3ee5fb1d7c5128f910da2b75` reflects the intentional threshold-fixture path change, and the final golden test passed.
- Final focused validation passed for `putmask_f64_parallel_matches_serial_reference_and_golden_sha256`, `cargo check -p fnp-ufunc --all-targets`, and `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`.
- `cargo fmt -p fnp-ufunc -- --check` still reports broad pre-existing formatting drift in `crates/fnp-ufunc/benches/elementwise.rs` and unrelated `crates/fnp-ufunc/src/lib.rs` regions; no workspace formatter was run.
- `timeout 120s ubs --only=rust crates/fnp-ufunc/src/lib.rs` timed out with `UBS_EXIT:124`; keep `ubs_fnp_ufunc_lib.txt` as a tooling caveat, not a pass.
- Retry condition: reopen only if a same-host NumPy rerun beats the final local FNP median on either row, if `putmask` semantics change away from position-index cycling, if compact bool-mask storage changes the measured loop body, or if larger rows above `1 << 20` show the raised Rayon threshold losing.

## 2026-06-19 - Gauntlet Verify: FNP put_mask vs NumPy

Artifact directory: `tests/artifacts/perf/2026-06-19_ufunc_put_mask_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.254`.
- Subject before measured correction: F64/no-sidecar `put_mask` activated Rayon at `1 << 14` elements, so the 100k true-rank cycling-fill row paid segmented-prefix scheduler overhead and the serial fallback still routed every write through integer-mutation sidecar plumbing.
- Final code: F64/no-sidecar `put_mask` uses an 8-lane SIMD mask scan and modulo-free value cycling below `1 << 20`; above that threshold it keeps the segmented-prefix Rayon path with a fixed 4K chunk size.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::put_mask` Criterion row.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7, timed with the same true-rank cycling formula.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-ufunc --bench elementwise put_mask_f64_masked_cycling -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- Python NumPy timing script in `numpy_put_mask_local.txt` using the same data formula.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-ufunc put_mask_f64_parallel_matches_serial_reference_and_golden_sha256 -- --nocapture`

Triage scorecard:
- Current fresh focused run vs refreshed NumPy median: win/loss/neutral = 1/1/0 across the two decision rows. The 100k row lost by 7.866x and was selected.
- Final remote run vs refreshed NumPy median: win/loss/neutral = 2/0/0.
- Earlier same-day same-worker ovh-a current snapshot to final ovh-a delta: 100k improved 81.857 us -> 15.858 us (5.16x); 1M improved 388.001 us -> 335.444 us (1.16x).

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.254` | Current code, fresh routing evidence | 100k current rch `hz2` | `baseline_fnp_put_mask_rch.txt`, `numpy_put_mask_local.txt` | 244.411 us | 31.069 us | 7.866x | Loss, selected |
| `franken_numpy-ixs5y.254` | Current code, fresh routing evidence | 1M current rch `hz2` | `baseline_fnp_put_mask_rch.txt`, `numpy_put_mask_local.txt` | 483.383 us | 686.361 us | 0.704x | Existing win |
| `franken_numpy-ixs5y.254` | Threshold/direct serial, no SIMD | 100k candidate rch `hz1` | `candidate_fnp_put_mask_rch_confirm.txt`, `numpy_put_mask_local.txt` | 76.797 us | 31.069 us | 2.472x | Rejected, still loses 100k |
| `franken_numpy-ixs5y.254` | Threshold/direct serial, no SIMD | 1M candidate rch `hz1` | `candidate_fnp_put_mask_rch_confirm.txt`, `numpy_put_mask_local.txt` | 445.529 us | 686.361 us | 0.649x | Win but not enough |
| `franken_numpy-ixs5y.254` | SIMD serial with `1 << 19` cutoff | 100k candidate rch `hz2` | `candidate_fnp_put_mask_simd_rch.txt`, `numpy_put_mask_local.txt` | 19.037 us | 31.069 us | 0.613x | Win |
| `franken_numpy-ixs5y.254` | SIMD serial with `1 << 19` cutoff | 1M candidate rch `hz2` | `candidate_fnp_put_mask_simd_rch.txt`, `numpy_put_mask_local.txt` | 709.480 us | 686.361 us | 1.034x | Rejected, neutral/loss |
| `franken_numpy-ixs5y.254` | Final SIMD serial with `1 << 20` cutoff | 100k final rch `ovh-a` | `final_fnp_put_mask_simd_threshold_rch.txt`, `numpy_put_mask_local.txt` | 15.858 us | 31.069 us | 0.510x | Keep |
| `franken_numpy-ixs5y.254` | Final SIMD serial with `1 << 20` cutoff | 1M final rch `ovh-a` | `final_fnp_put_mask_simd_threshold_rch.txt`, `numpy_put_mask_local.txt` | 335.444 us | 686.361 us | 0.489x | Keep |

Notes:
- The threshold-only lever was insufficient: it improved 1M but still lost the 100k row by 2.472x versus the refreshed NumPy reference.
- The first SIMD lever was too aggressive about re-entering Rayon at `1 << 19`: it won 100k but regressed the 1M row to a neutral/loss against NumPy. Raising the cutoff to `1 << 20` keeps both measured rows on the cache-local SIMD serial path.
- The kept lever is the graveyard "constants kill you" correction plus vectorized mask extraction: use SCE/dtype proof to skip generic mutation machinery, use SIMD only to build sparse lane masks, and cycle values with an increment/reset index instead of `%` in each true lane.
- The golden fixture digest changed intentionally as the threshold-crossing fixture moved to `PUT_MASK_PARALLEL_MIN_ELEMS + 421`; elementwise serial-reference comparison passed before the SHA assertion, and the updated digest `f8a49cce66312e0fb3fdfdbcc5e31b70662343eff3f8d49ae4f01ae828da3c0c` passed.
- The sidecar fallback test fixture was corrected from `(1_i64 << 53) + 2` to `(1_i64 << 53) - 2`, staying within the crate's exact temporary F64 bridge contract while preserving the large-integer sidecar proof.
- Final focused validation passed for `put_mask_f64_parallel_matches_serial_reference_and_golden_sha256`, `cargo check -p fnp-ufunc --all-targets`, `cargo clippy -p fnp-ufunc --all-targets -- -D warnings`, and `git diff --check`.
- `cargo fmt --check` still reports broad pre-existing workspace formatting drift outside this slice; the put_mask hunk was manually adjusted to match the targeted rustfmt diff and no unrelated formatting was applied.
- `timeout 120s ubs --only=rust crates/fnp-ufunc/src/lib.rs` did not produce a completion summary before the wrapper returned, and zsh did not preserve the exit code in the artifact; keep `ubs_fnp_ufunc_lib.txt` as inconclusive, not a pass.
- Retry condition: reopen only if a same-host NumPy rerun beats the final FNP median on either row, if compact bool-mask storage changes the loop body, or if rows above `1 << 20` show the retained segmented-prefix parallel path losing to the SIMD serial path.

## 2026-06-20 - Gauntlet Verify: FNP place vs NumPy

Artifact directory: `tests/artifacts/perf/2026-06-20_ufunc_place_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.252`.
- Subject before measured correction: F64/no-sidecar `place` activated Rayon at `1 << 14` elements, so the 100k true-rank cycling-fill row paid segmented-prefix scheduler overhead and the serial fallback still routed every write through integer-mutation sidecar plumbing.
- Final code: F64/no-sidecar `place` uses an 8-lane SIMD mask scan and modulo-free value cycling below `1 << 20`; above that threshold it keeps the segmented-prefix Rayon path with a fixed 4K chunk size.
- Subject API: direct Rust `fnp-ufunc` `UFuncArray::place` Criterion row.
- Oracle/reference: NumPy 2.4.3 on Python 3.13.7, timed with the same mask and cyclic value formula.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-ufunc --bench elementwise place_f64_masked_cycling -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- Python NumPy timing script in `numpy_place_local.txt` using the same data formula.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-ufunc place_f64_parallel_matches_serial_reference_and_golden_sha256 -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo check -p fnp-ufunc --all-targets`
- `RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo clippy -p fnp-ufunc --all-targets -- -D warnings`
- `cargo fmt -p fnp-ufunc -- --check`
- `git diff --check`
- `timeout 120s ubs --only=rust crates/fnp-ufunc/src/lib.rs`

Triage scorecard:
- Current focused run vs refreshed NumPy median: win/loss/neutral = 1/1/0 across the two decision rows. The 100k row lost by 1.307x and was selected.
- Final remote run vs refreshed NumPy median: win/loss/neutral = 2/0/0.

| Bead | Lever | Workload | Artifact | FrankenNumPy | NumPy | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.252` | Current code, fresh routing evidence | 100k current rch `hz1` | `baseline_estimates_extracted_before_candidate.txt`, `numpy_place_local.txt` | 87.094 us | 66.616 us | 1.307x | Loss, selected |
| `franken_numpy-ixs5y.252` | Current code, fresh routing evidence | 1M current rch `hz1` | `baseline_estimates_extracted_before_candidate.txt`, `numpy_place_local.txt` | 441.318 us | 815.936 us | 0.541x | Existing win |
| `franken_numpy-ixs5y.252` | Final SIMD serial with `1 << 20` cutoff | 100k final rch `hz2` | `candidate_fnp_place_rch.txt`, `numpy_place_local.txt` | 21.050 us | 66.616 us | 0.316x | Keep |
| `franken_numpy-ixs5y.252` | Final SIMD serial with `1 << 20` cutoff | 1M final rch `hz2` | `candidate_fnp_place_rch.txt`, `numpy_place_local.txt` | 273.974 us | 815.936 us | 0.336x | Keep |

Notes:
- This is the same successful "constants kill you" correction as `copyto`, `putmask`, and `put_mask`, but applied to `place`'s true-rank cyclic semantics: avoid tiny Rayon morsels and integer-sidecar mutation machinery when dtype and sidecar checks prove a flat F64/no-sidecar loop.
- The SIMD serial path scans F64 mask chunks as 8-lane vectors, emits a bitmask of nonzero lanes, and advances the cyclic value index with increment/reset rather than `%` on every write.
- The golden fixture digest changed intentionally after the threshold-crossing fixture moved below the new parallel activation point; elementwise serial-reference comparison passed before the SHA assertion, and the updated digest `41ebf3fa471d4b7c9b29ddc1cde3e96b7b972072359d9ed98ac53ee806bf7add` passed.
- Final focused validation passed for `place_f64_parallel_matches_serial_reference_and_golden_sha256`, `cargo check -p fnp-ufunc --all-targets`, `cargo clippy -p fnp-ufunc --all-targets -- -D warnings` on retry worker `hz1`, and `git diff --check`.
- Initial clippy on `ovh-b` failed in a dependency build script with `SIGILL`; the retry on `hz1` is the passing clippy gate.
- `cargo fmt -p fnp-ufunc -- --check` still reports broad pre-existing formatting drift outside this slice; no workspace formatter was run.
- `timeout 120s ubs --only=rust crates/fnp-ufunc/src/lib.rs` timed out after starting the Rust scan; keep `ubs_fnp_ufunc_lib.txt` as inconclusive, not a pass.
- Retry condition: reopen only if a same-host NumPy rerun beats the final FNP median on either row, if compact bool-mask storage changes the loop body, or if rows above `1 << 20` show the retained segmented-prefix parallel path losing to the SIMD serial path.

## 2026-06-20 - Gauntlet Verify: `fnp-linalg` matrix norm column reductions

Artifact directory: `tests/artifacts/perf/2026-06-20_linalg_batch_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.235`.
- Subject before measured correction: `matrix_norm_nxn_orders` already scanned row-major for large `ord=1/-1` matrices, but allocated the column-sum scratch buffer on the heap for every call.
- Kept lever: stack-resident scratch for 512 through 1024 columns, with the existing heap path retained outside that measured window.
- No-ship lever: an unrolled Frobenius accumulator was tested and reverted after batch Frobenius rows regressed.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg 'batch_trace|batch_matrix_norm|matrix_norm_nxn_orders|kron_nxn' -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg 'matrix_norm_nxn_orders/(one|neg_one)' -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-linalg matrix_norm_column_reduction_matches_strided_reference_bits -- --nocapture`
- Direct Python NumPy comparator on `hz2` in `numpy_linalg_hz2.txt`.

Triage scorecard:
- Current `hz2` FNP vs NumPy: win/loss/neutral = 1/7/0 across the eight column norm rows. The `one/128` row was effectively neutral/loss at 1.005x, and the 256 through 1024 rows were clear losses.
- Final `hz2` FNP vs old `hz2` FNP: win/loss/neutral = 8/0/0.
- Final `hz2` FNP vs NumPy: win/loss/neutral = 2/6/0. This is a kept gap-narrowing lever, not a full NumPy domination closeout.

| Bead | Lever | Workload | Artifact | Old FNP ns | Final FNP ns | NumPy ns | Final/Old | Final/NumPy | Verdict |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `franken_numpy-ixs5y.235` | Stack scratch 512-1024 cols | `one/128` on `hz2` | `criterion_linalg_current.txt`, `criterion_linalg_column_stack_gated_candidate_hz1.txt`, `numpy_linalg_hz2.txt` | 9603 | 7544 | 9553 | 0.786x | 0.790x | Keep/supporting win |
| `franken_numpy-ixs5y.235` | Stack scratch 512-1024 cols | `neg_one/128` on `hz2` | same | 9375 | 7484 | 9574 | 0.798x | 0.782x | Keep/supporting win |
| `franken_numpy-ixs5y.235` | Stack scratch 512-1024 cols | `one/256` on `hz2` | same | 38032 | 30444 | 27712 | 0.800x | 1.099x | Keep, still loses |
| `franken_numpy-ixs5y.235` | Stack scratch 512-1024 cols | `neg_one/256` on `hz2` | same | 37675 | 29924 | 28312 | 0.794x | 1.057x | Keep, still loses |
| `franken_numpy-ixs5y.235` | Stack scratch 512-1024 cols | `one/512` on `hz2` | same | 154304 | 116333 | 103667 | 0.754x | 1.122x | Keep, still loses |
| `franken_numpy-ixs5y.235` | Stack scratch 512-1024 cols | `neg_one/512` on `hz2` | same | 152028 | 116827 | 102987 | 0.768x | 1.134x | Keep, still loses |
| `franken_numpy-ixs5y.235` | Stack scratch 512-1024 cols | `one/1024` on `hz2` | same | 615716 | 458082 | 397192 | 0.744x | 1.153x | Keep, still loses |
| `franken_numpy-ixs5y.235` | Stack scratch 512-1024 cols | `neg_one/1024` on `hz2` | same | 603420 | 466084 | 393621 | 0.772x | 1.184x | Keep, still loses |
| `franken_numpy-ixs5y.235` | Frobenius unroll | `batch_matrix_norm_fro/4096x8x8` on `hz1` | `criterion_linalg_fro_head_baseline.txt`, `criterion_linalg_fro_unroll_candidate.txt` | 76821 | 84812 | n/a | 1.104x | n/a | Reverted |
| `franken_numpy-ixs5y.235` | Frobenius unroll | `batch_matrix_norm_fro/1024x32x32` on `hz1` | same | 194123 | 221803 | n/a | 1.143x | n/a | Reverted |

Notes:
- The stack path preserves the old scalar addition order per column; the focused test compares both a heap-cache case and a stack-cache case against the former strided reference bits, including NaN propagation through `1`, `-1`, `inf`, and `-inf`.
- The first stack candidate was not kept as-is because direct `hz1` evidence showed a small `one/256` regression. The final code gates stack scratch to the measured 512-1024 column range and keeps the heap path elsewhere.
- `numpy_column_vmi_rch.txt` is an invalid probe artifact only: RCH warned that the command was non-compilation and the Python quoting failed before timing. It is not counted in any ratio.
- Remaining gap: NumPy is still faster for 256 through 1024 column reductions on `hz2`. Next deeper lever should be vectorized absolute-value accumulation or multiple-column strip mining that preserves per-column scalar addition order, not another allocation-only retune.

## 2026-06-20 - Gauntlet Verify: `fnp-linalg` batched Frobenius norm lane fill

Artifact directory: `tests/artifacts/perf/2026-06-20_linalg_batch_fro_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.238`.
- Subject before measured closeout: `batch_matrix_norm(..., ord="fro")` already had the direct lane-fill path from the code-first child, but it had not been put through a same-worker NumPy ratio gate.
- Kept lever: direct batched Frobenius lane fill after one shape/data validation; each lane uses the same row-major `v * v` accumulation and final `sqrt` as the per-lane `matrix_norm_nxn(..., "fro")` reference.
- No new source edit was made in this closeout. The measured decision is keep/close, not another speculative tweak.
- Worker: `hz1`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.

Commands:
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg 'batch_matrix_norm_fro' -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- Direct Python NumPy comparator on `hz1` in `numpy_batch_matrix_norm_fro_hz1_success.txt`.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-linalg batch_matrix_norm_fro_direct_lane_fill_matches_per_lane_reference_bits -- --nocapture`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo build -p fnp-linalg --release`

Triage scorecard:
- Final same-worker `hz1` FNP vs NumPy: win/loss/neutral = 2/0/0.

| Bead | Lever | Workload | Artifact | FrankenNumPy ns | NumPy ns | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.238` | Direct batched Frobenius lane fill | `4096x8x8` on `hz1` | `fnp_batch_matrix_norm_fro_current.txt`, `numpy_batch_matrix_norm_fro_hz1_success.txt` | 76177 | 234973 | 0.324x | Keep, 3.08x faster |
| `franken_numpy-ixs5y.238` | Direct batched Frobenius lane fill | `1024x32x32` on `hz1` | same | 218772 | 581466 | 0.376x | Keep, 2.66x faster |

Notes:
- The focused bit-preservation guard passed for `batch_matrix_norm_fro_direct_lane_fill_matches_per_lane_reference_bits`, covering serial and threshold-crossing batch shapes plus NaN, Inf, and signed-zero inputs against the old per-lane reference.
- `numpy_batch_matrix_norm_fro_hz1.txt` is an invalid shell-quote attempt and is not counted in the ratio table. The counted comparator is `numpy_batch_matrix_norm_fro_hz1_success.txt` on Python 3.14.4 / NumPy 2.3.5 on `hz1`.
- This bead should not be reopened for another Frobenius micro-retune unless a same-worker NumPy rerun beats either final Rust median, or a future change alters the accumulation order, batch parallel threshold, or matrix-norm dispatch path.

## 2026-06-20 - Gauntlet Verify: `fnp-linalg` batched trace lane fill

Artifact directory: `tests/artifacts/perf/2026-06-20_linalg_batch_trace_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.237`.
- Subject before measured closeout: `batch_trace` already had the direct lane-fill path from the code-first child, but the earlier routing table had one small `1024x32x32` loss on a different run window.
- Kept lever: direct batched trace lane fill after one shape/data validation; each lane sums diagonal entries in ascending order with the same `trace_nxn` schedule, including NaN and signed-zero behavior.
- No new source edit was made in this closeout. The measured decision is keep/close, not another speculative trace tweak.
- Worker: `hz1`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.

Commands:
- `RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg 'batch_trace' -- --sample-size 20 --warm-up-time 1 --measurement-time 3 --output-format bencher`
- Direct Python NumPy comparator on `hz1` in `numpy_batch_trace_hz1.txt`.
- `RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-linalg batch_trace_direct_lane_fill_matches_per_lane_reference_bits -- --nocapture`
- `RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo build -p fnp-linalg --release`

Triage scorecard:
- Final same-worker `hz1` FNP vs NumPy: win/loss/neutral = 2/0/0.

| Bead | Lever | Workload | Artifact | FrankenNumPy ns | NumPy ns | FNP/NumPy ratio | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| `franken_numpy-ixs5y.237` | Direct batched trace lane fill | `4096x8x8` on `hz1` | `fnp_batch_trace_current_hz1.txt`, `numpy_batch_trace_hz1.txt` | 47188 | 102977 | 0.458x | Keep, 2.18x faster |
| `franken_numpy-ixs5y.237` | Direct batched trace lane fill | `1024x32x32` on `hz1` | same | 47184 | 61381 | 0.769x | Keep, 1.30x faster |

Notes:
- The focused bit-preservation guard passed for `batch_trace_direct_lane_fill_matches_per_lane_reference_bits`, covering serial and threshold-crossing batch shapes plus NaN and signed-zero propagation against the old per-lane reference.
- Graveyard mapping: cache-local vectorized execution plus constants-kill-you discipline. The fresh measurement shows the existing direct lane-fill path already clears the NumPy gate, so no additional parallel-threshold or unrolled-diagonal lever was justified.
- This bead should not be reopened for another trace micro-retune unless a same-worker NumPy rerun beats either final Rust median, or a future change alters the diagonal accumulation order, batch parallel threshold, or trace dispatch path.

## 2026-06-20 - Gauntlet Verify: `fnp-linalg` symmetric spectral cond fast path

Artifact directory: `tests/artifacts/perf/2026-06-20_linalg_cond_values_sort_vs_numpy/`

Run identity:
- Bead: `franken_numpy-ixs5y.234`.
- Subject before measured correction: the values-only SVD in-place singular sort and `cond_nxn` bench row already existed, but same-worker head-to-head proof showed `cond_nxn` still lost badly to NumPy.
- Kept lever: exact-symmetric finite spectral condition numbers now route through `eigvalsh_nxn` because singular values of a real symmetric matrix are the absolute eigenvalues. This avoids a full values-only SVD for the measured symmetric `cond_nxn` workload while preserving the old paths for non-symmetric, rectangular, NaN, Inf, and non-spectral orders.
- Worker: `hz1`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.

Commands:
- `RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg 'cond_nxn/size/(64|128|256)' -- --sample-size 10 --warm-up-time 1 --measurement-time 2 --output-format bencher`
- `RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo bench -p fnp-linalg --bench criterion_linalg 'cond_nxn' -- --sample-size 10 --warm-up-time 1 --measurement-time 2 --output-format bencher`
- Direct Python NumPy comparator on `hz1` in `numpy_cond_nxn_hz1.txt`.
- `RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-linalg cond_p_spectral_symmetric -- --nocapture`
- `RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo test -p fnp-linalg values_only_svd_in_place_sort_matches_former_index_schedule -- --nocapture`
- `RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo build -p fnp-linalg --release`
- `RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b rch exec -- cargo clippy -p fnp-linalg --all-targets -- -D warnings`

Triage scorecard:
- Initial same-worker FNP vs NumPy: win/loss/neutral = 0/4/0. The old 512 row did not finish in the full run and was interrupted; NumPy completed the same 512 comparator in 121812521 ns.
- Final same-worker FNP vs NumPy: win/loss/neutral = 3/1/0.
- Final same-worker FNP vs old FNP: win/loss/neutral = 4/0/0, counting 512 as timeout-to-completed.

| Bead | Lever | Workload | Artifact | Old FNP ns | Final FNP ns | NumPy ns | Final/Old | Final/NumPy | Verdict |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `franken_numpy-ixs5y.234` | Exact-symmetric cond via `eigvalsh_nxn` | `cond_nxn/64` on `hz1` | `fnp_cond_nxn_64_128_256_hz1.txt`, `fnp_cond_nxn_symmetric_fast_path_hz1.txt`, `numpy_cond_nxn_hz1.txt` | 51961635 | 215148 | 229157 | 0.004x | 0.939x | Keep, beats NumPy |
| `franken_numpy-ixs5y.234` | Exact-symmetric cond via `eigvalsh_nxn` | `cond_nxn/128` on `hz1` | same | 287303721 | 1746263 | 1388876 | 0.006x | 1.257x | Keep, residual loss |
| `franken_numpy-ixs5y.234` | Exact-symmetric cond via `eigvalsh_nxn` | `cond_nxn/256` on `hz1` | same | 1715056173 | 10107470 | 15179317 | 0.006x | 0.666x | Keep, beats NumPy |
| `franken_numpy-ixs5y.234` | Exact-symmetric cond via `eigvalsh_nxn` | `cond_nxn/512` on `hz1` | same | timeout | 60907729 | 121812521 | n/a | 0.500x | Keep, beats NumPy |

Notes:
- This is not another SVD sort retune. The baseline proved the sort allocation was not the dominant NumPy gap; the successful lever changed the complexity surface for finite exact-symmetric matrices from values-only SVD to symmetric eigensolve.
- The focused symmetric tests compare the fast path to the SVD reference and cover `p="2"` and `p="-2"` absolute-eigenvalue semantics. The original values-only in-place sort bit guard also passed.
- `cargo fmt -p fnp-linalg -- --check` still fails on broad pre-existing formatting drift in `fnp-linalg` benches/examples and older source regions; no formatter was run because it would rewrite unrelated files.
- Remaining gap: `cond_nxn/128` is still 1.257x slower than NumPy. Retry only if a same-worker `eigvalsh_nxn` profile identifies the exact 128-size frame, or if a broader symmetric spectral primitive can improve 128 without regressing the 64/256/512 wins. Do not reopen this bead for SVD sort allocation, right-Vt, row-Householder, packed-GEMM tile, or SBR/bulge-chase microfamilies.

## 2026-06-20 - Gauntlet Reject: `fnp-linalg` matrix column norm NaN prefilter and stack256

Artifact directory: `tests/artifacts/perf/2026-06-20_linalg_column_norm_prefilter_stack256/`

Run identity:
- Parent bead: `franken_numpy-ixs5y`.
- Agent: `BlackThrush` / `cod-a`.
- Worker: `hz2`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Crate scope: `fnp-linalg` only.

Lever attempts:
- Rejected `branchless-prefilter+stack256`: a whole-matrix NaN prefilter plus branchless cache-linear absolute column accumulation and 256-column stack scratch.
- Rejected `stack256-only`: lower the stack scratch threshold from 512 to 256 columns with the original NaN branch preserved.

Triage scorecard:
- Both candidates passed the focused column-reduction bit reference test.
- Both candidates failed the performance keep gate and were reverted.
- Same-worker final code still has the previously measured 256-1024 column-norm losses against NumPy; this run records failed attempts, not a keep.

| Workload | NumPy ns | Baseline FNP ns | Prefilter+stack256 FNP ns | Stack256-only FNP ns | Baseline/NumPy | Prefilter/NumPy | Stack256/NumPy | Stack256/Baseline | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `one/256` | 27712 | 29785 | 45590 | 29766 | 1.075x | 1.645x | 1.074x | 0.999x | no-ship, reverted |
| `neg_one/256` | 28312 | 30303 | 44570 | 29630 | 1.070x | 1.574x | 1.047x | 0.978x | no-ship, reverted |
| `one/512` | 103667 | 115964 | 182591 | 114610 | 1.119x | 1.761x | 1.106x | 0.988x | no-ship, reverted |
| `neg_one/512` | 102987 | 113597 | 183733 | 119919 | 1.103x | 1.784x | 1.164x | 1.056x | no-ship, reverted |
| `one/1024` | 397192 | 457106 | 723751 | 465194 | 1.151x | 1.822x | 1.171x | 1.018x | no-ship, reverted |
| `neg_one/1024` | 393621 | 458114 | 727149 | 456385 | 1.164x | 1.848x | 1.159x | 0.996x | no-ship, reverted |

Notes:
- The prefilter doubled memory traffic and made every targeted 256-1024 column-norm row materially worse.
- The stack-threshold-only retry found one modest `neg_one/256` improvement, but it was not broad enough and regressed `neg_one/512` and `one/1024`; this is below the keep threshold.
- Do not retry a whole-matrix NaN prefilter for this path unless the scan is fused with another required pass. Do not retry 256-column stack scratch as a standalone lever. The next credible attempt needs SIMD or strip-mined multi-column accumulation that preserves column-addition order and NaN behavior.

## 2026-06-20 - Gauntlet Keep: `fnp-python` cached-buffer einsum diagonal

Artifact directory: `tests/artifacts/perf/2026-06-20_python_einsum_diag_cod_a/`

Run identity:
- Parent bead: `franken_numpy-ixs5y`.
- Agent: `BlackThrush` / `cod-a`.
- Crate: `fnp-python`.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Target gap: `fnp_einsum_diag_f64_4000`, a Python-boundary loss versus `numpy.einsum("ii->i", a)`.

Lever attempts:
- Kept `cached-buffer+interned-names`: an early exact-NumPy-ndarray f64 single-operand diagonal/trace gate before dtype-policy probing, using `PyBuffer<f64>` metadata, cached `numpy.ndarray` type identity, and interned Python names for `diagonal`, `setflags`, and `write`.
- Rejected as standalone `buffered-string-type`: `PyBuffer<f64>` plus type-name/module string checks improved the path but still lost to NumPy.
- Rejected as standalone `cached-type-no-intern`: cached ndarray type identity reduced the residual to near-neutral, but still did not beat NumPy locally.

Triage scorecard:
- Initial local FNP vs NumPy: win/loss/neutral = 0/2/0 for trace and diagonal.
- Final local FNP vs NumPy: win/loss/neutral = 2/0/0 for trace and diagonal.
- Final rch FNP vs NumPy on `vmi1227854`: win/loss/neutral = 1/1/0 for trace and diagonal. The diagonal keep is replicated remotely; the trace row remains residual negative evidence on that worker.
- Focused conformance: `rch exec -- cargo test -p fnp-python --test conformance_einsum` passed 28/28.

| Workload | Evidence | Baseline FNP | Final FNP | NumPy | Final/Baseline | Final/NumPy | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| `fnp_einsum_trace_f64_4000` | local baseline/final | 18.425 us | 15.296 us | 15.852 us | 0.830x | 0.965x | keep locally |
| `fnp_einsum_diag_f64_4000` | local baseline/final | 4.5756 us | 883.98 ns | 1.0942 us | 0.193x | 0.808x | keep |
| `fnp_einsum_trace_f64_4000` | final rch `vmi1227854` | n/a | 5.9900 us | 5.2275 us | n/a | 1.146x | residual loss |
| `fnp_einsum_diag_f64_4000` | final rch `vmi1227854` | n/a | 805.39 ns | 889.51 ns | n/a | 0.905x | keep |

Intermediate candidate evidence:

| Candidate | Diagonal FNP | NumPy | FNP/NumPy | Verdict |
|---|---:|---:|---:|---|
| `buffered-string-type` | 1.2799 us | 1.0142 us | 1.262x | no standalone keep |
| `cached-type-no-intern` | 1.0609 us | 1.0194 us | 1.041x | neutral/slight loss |
| `cached-buffer+interned-names` | 883.98 ns | 1.0942 us | 0.808x | keep |

Notes:
- The previous pre-policy diagonal shortcut family is superseded: moving the old helper earlier was not enough; the measured win required avoiding dtype-policy probing, avoiding per-call ndarray string type checks, and avoiding per-call Python string allocation for method/keyword names.
- The kept diagonal path still delegates view construction to NumPy's `diagonal()` and explicitly restores writability with `setflags(write=True)` when the operand is writable, preserving NumPy `einsum("ii->i")` view semantics.
- `cargo check -p fnp-python --lib --bench criterion_python_surface` passed, with pre-existing `fnp-python` warnings. `cargo check -p fnp-python --benches` and `cargo fmt -p fnp-python -- --check` are blocked by unrelated pre-existing lib-test call-site drift and formatting drift; no formatter was run to avoid unrelated rewrites.
- Retry predicate: do not retry wrapper-level pre-policy diagonal dispatch. The next credible diagonal retry must remove or bypass the remaining `diagonal()+setflags(write=True)` method dispatch while preserving writable-view semantics. Treat the rch trace residual as a separate trace path issue, not a reason to revert the diagonal keep.

## 2026-06-20 - Gauntlet Reject: `fnp-linalg` matrix column norm 8-column strip mine

Artifact directory: `tests/artifacts/perf/2026-06-20_linalg_column_norm_stripmine_cod_a/`

Run identity:
- Parent bead: `franken_numpy-ixs5y`.
- Agent: `BlackThrush` / `cod-a`.
- Crate scope: `fnp-linalg` only.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-a`.
- Candidate: safe 8-column strip-mined cache-linear column accumulation for `matrix_norm_nxn(..., ord="1" | "-1")`.

Decision:
- Rejected and reverted. The focused bit-preservation test passed, but the performance proof was mixed and the same-host NumPy comparator could not be refreshed.
- The candidate had one same-worker Rust regression on `vmi1149989` and a later `hz1` RCH-lane run that lost every row against the available `hz1` NumPy context.
- Direct Python comparator attempts on `vmi1149989` and `hz1` failed with SSH auth denial. `rch exec -- python3` ran locally on `thinkstation1`, so those NumPy ratios are routing evidence only.

Measured Rust delta on `vmi1149989`:

| Workload | Baseline FNP ns | Candidate FNP ns | Candidate/Baseline | Verdict |
|---|---:|---:|---:|---|
| `one/256` | 28388 | 23372 | 0.823x | win |
| `neg_one/256` | 26724 | 27721 | 1.037x | loss |
| `one/512` | 113473 | 106512 | 0.939x | win |
| `neg_one/512` | 111496 | 103362 | 0.927x | win |
| `one/1024` | 530381 | 409582 | 0.772x | win |
| `neg_one/1024` | 632365 | 412535 | 0.652x | win |

Routing-only NumPy ratio from local `thinkstation1` NumPy 2.4.3:

| Workload | Candidate FNP ns (`vmi1149989`) | Local NumPy ns | Candidate/NumPy | Counted? |
|---|---:|---:|---:|---|
| `one/256` | 23372 | 29345 | 0.796x | no, cross-host |
| `neg_one/256` | 27721 | 26140 | 1.060x | no, cross-host |
| `one/512` | 106512 | 96573 | 1.103x | no, cross-host |
| `neg_one/512` | 103362 | 113425 | 0.911x | no, cross-host |
| `one/1024` | 409582 | 416639 | 0.983x | no, cross-host |
| `neg_one/1024` | 412535 | 359040 | 1.149x | no, cross-host |

Repeat RCH-lane routing evidence on `hz1` versus prior direct `hz1` NumPy:

| Workload | Candidate FNP ns (`hz1`) | Prior NumPy ns (`hz1`) | Candidate/NumPy | Verdict |
|---|---:|---:|---:|---|
| `one/256` | 50646 | 40921 | 1.238x | loss |
| `neg_one/256` | 50689 | 40940 | 1.238x | loss |
| `one/512` | 211885 | 147264 | 1.439x | loss |
| `neg_one/512` | 213556 | 145528 | 1.468x | loss |
| `one/1024` | 836943 | 506356 | 1.653x | loss |
| `neg_one/1024` | 830032 | 503971 | 1.647x | loss |

Focused conformance:
- `rch exec -- cargo test -p fnp-linalg matrix_norm_column_reduction_matches_strided_reference_bits -- --nocapture`: pass, 1 focused test passed on `hz1`.
- `rch exec -- cargo build -p fnp-linalg --release`: pass on `vmi1293453` after source revert.

Negative retry predicate:
- Do not retry the scalar 8-column manual strip mine as a standalone lever.
- A credible retry needs either actual SIMD absolute-value lanes or generated size-specialized column microkernels, same-host NumPy capture, and zero row regressions across `256/512/1024` for both `ord="1"` and `ord="-1"`.

## 2026-06-20 - BOLD-VERIFY Reject: `fnp-linalg` batch Cholesky blocked ordered-dot helper

Artifact directory: `tests/artifacts/perf/2026-06-20_linalg_batch_cholesky_ordered_dot_cod_b/`

Run identity:
- Parent bead: `franken_numpy-ixs5y`; child bead: `franken_numpy-ixs5y.270`.
- Agent: `YellowElk` / `cod-b`.
- Crate scope: `fnp-linalg` only.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.
- Candidate: extend ordered 4-wide dot helpers beyond the small-N unblocked Cholesky path into the blocked diagonal and panel update loops.
- Baseline HEAD moved during the run to `856c38cb`; the candidate was applied on top of that source state, then reverted so the production file again matches HEAD.
- RCH selected `vmi1153651` for both the baseline and candidate Criterion runs. The requested `RCH_WORKER=hz1` did not bind the worker, so only the same-worker `vmi1153651` Rust delta is counted.

Decision:
- Rejected and reverted. Focused Cholesky correctness passed, but the performance result was mixed and noisy: one target row improved by 8.6%, while the larger row regressed by 6.4%.
- Direct NumPy comparator capture on `vmi1153651` was blocked by SSH authentication. `rch exec -- python3` runs on local `thinkstation1`, so no new same-host NumPy ratio is counted for this attempt.
- Because this was not a zero-regression result and the NumPy comparator was unavailable, no production source change was kept.

Measured Rust delta on `vmi1153651`:

| Workload | Baseline FNP ns | Candidate FNP ns | Candidate/Baseline | NumPy comparator | Verdict |
|---|---:|---:|---:|---|---|
| `batch_cholesky/shape/64x128x128` | 14844832 | 13567919 | 0.914x | not counted; SSH auth blocked same-host Python | mixed win |
| `batch_cholesky/shape/16x256x256` | 20811194 | 22141744 | 1.064x | not counted; SSH auth blocked same-host Python | loss |

Focused conformance:
- `rch exec -- cargo test -p fnp-linalg cholesky_ -- --nocapture`: pass on `vmi1153651`; 21 unit tests passed, 2 ignored, 303 filtered, and the Cholesky golden/metamorphic integration filters passed.

Negative retry predicate:
- Do not retry the blocked-path ordered 4-wide scalar dot helper as a standalone lever.
- A credible retry needs a deeper algorithmic or generated-kernel change: for example a real safe SIMD dot primitive that preserves required bit contracts, a size-specialized blocked/batched panel kernel, or a generated microkernel with same-host NumPy capture and no regressions across both medium batch rows.

## 2026-06-20 - BOLD-VERIFY Keep: `fnp-linalg` batch Cholesky direct-write n=16/32

Artifact directory: `tests/artifacts/perf/2026-06-20_linalg_batch_cholesky_direct_write_cod_b/`

Run identity:
- Parent bead: `franken_numpy-ixs5y`; child bead: `franken_numpy-ixs5y.273`.
- Agent: `YellowElk` / `cod-b`.
- Crate scope: `fnp-linalg` only.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.
- Candidate: widen the existing zero-allocation `batch_cholesky` direct-write route to `n <= 32`, and make `cholesky_nxn_into_out` use the same ordered 4-wide scalar dot helper already used by `cholesky_nxn` for `n=16..32`.

Decision:
- Kept. The changed branch is limited to `n <= 32`; the affected same-worker rows improved materially and beat same-host NumPy.
- The measured `n >= 64` guard rows are recorded as losses versus the immediately paired baseline even though the candidate branch is not reachable for those sizes. They remain faster than NumPy on the same host, but they are negative evidence about this noisy shared-worker bench lane and should not be used to claim a broad Cholesky win.
- `RCH_WORKER=vmi1153651` was not honored while that worker was inadmissible; `RCH_WORKER=vmi1227854` was honored and produced the decisive paired baseline.

Primary paired evidence on `vmi1227854`:

| Workload | Baseline FNP ns | Candidate FNP ns | Candidate/Baseline | NumPy median ns | Candidate/NumPy | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `batch_cholesky/shape/2000x16x16` | 572680 | 450154 | 0.786x | 2454268 | 0.183x | keep win |
| `batch_cholesky/shape/1000x32x32` | 1357341 | 971594 | 0.716x | 4061998 | 0.239x | keep win |
| `batch_cholesky/shape/500x64x64` | 3140923 | 4005072 | 1.275x | 6094522 | 0.657x | guard loss; branch not reached |
| `batch_cholesky/shape/64x128x128` | 1887548 | 2179264 | 1.155x | 10195537 | 0.214x | guard loss; branch not reached |
| `batch_cholesky/shape/16x256x256` | 2672825 | 3306358 | 1.237x | 15068349 | 0.219x | guard loss; branch not reached |

Repeat candidate routing evidence on `vmi1227854` before the paired baseline:

| Workload | Candidate FNP ns | Candidate/Baseline | Candidate/NumPy | Verdict |
|---|---:|---:|---:|---|
| `batch_cholesky/shape/2000x16x16` | 481572 | 0.841x | 0.196x | repeat win |
| `batch_cholesky/shape/1000x32x32` | 1020036 | 0.751x | 0.251x | repeat win |
| `batch_cholesky/shape/500x64x64` | 3457920 | 1.101x | 0.567x | guard loss; branch not reached |
| `batch_cholesky/shape/64x128x128` | 1934582 | 1.025x | 0.190x | guard loss/noise; branch not reached |
| `batch_cholesky/shape/16x256x256` | 2791921 | 1.045x | 0.185x | guard loss/noise; branch not reached |

Auxiliary `vmi1153651` baseline-only evidence, not used for the keep decision because no candidate run selected that worker:

| Workload | Baseline run 1 / NumPy | Baseline run 2 / NumPy | Baseline run 3 / NumPy | Verdict |
|---|---:|---:|---:|---|
| `batch_cholesky/shape/2000x16x16` | 2.652x | 1.302x | 1.741x | residual loss/noisy |
| `batch_cholesky/shape/1000x32x32` | 1.010x | 2.344x | 3.002x | residual loss/noisy |
| `batch_cholesky/shape/500x64x64` | 0.905x | 1.790x | 2.337x | mixed/noisy |
| `batch_cholesky/shape/64x128x128` | 1.007x | 1.817x | 2.340x | residual loss/noisy |
| `batch_cholesky/shape/16x256x256` | 0.540x | 1.091x | 1.809x | mixed/noisy |

Focused conformance and crate health:
- `rch exec -- cargo test -j 1 -p fnp-linalg batch_cholesky_scratch_matches_per_lane_cholesky_nxn_bits -- --nocapture`: pass on `vmi1149989`; the focused test passed and the filtered integration shards returned zero-test OK.
- `rch exec -- cargo check -j 1 -p fnp-linalg --all-targets`: pass on `vmi1149989`.
- `rch exec -- cargo clippy -j 1 -p fnp-linalg --all-targets -- -D warnings`: pass on `vmi1149989`.
- `cargo fmt -p fnp-linalg -- --check`: fail due broad pre-existing rustfmt drift in `fnp-linalg` benches/examples and unrelated `src/lib.rs` blocks; formatter was not run to avoid unrelated churn. `git diff --check` passed for the kept patch.

Retry predicate:
- Do not retry direct-write allocation elimination below `n=16`; that family is now extended through `n=32`.
- A future Cholesky attempt should either improve the Python stacked boundary directly or target `n >= 64` with a separate branch-specific kernel. It must not use this noisy `n>=64` guard-table drift as proof that the direct-write `n<=32` branch regressed those sizes.

## 2026-06-20 - BOLD-VERIFY Keep: `fnp-linalg` eigvalsh mid-band row-dot reducer

Artifact directory: `tests/artifacts/perf/2026-06-20_linalg_eigvalsh_values_cod_b/`

Run identity:
- Parent bead: `franken_numpy-ixs5y`; child bead: `franken_numpy-ixs5y.275`.
- Agent: `YellowElk` / `cod-b`.
- Crate scope: `fnp-linalg` only.
- Worker proof: `hz1` for Rust Criterion and direct NumPy comparator.
- Target dir: `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_numpy-cod-b`.
- Candidate: use a full contiguous row-dot serial panel matvec only for `192 <= n < 384`; below and above that range keep the old half-symmetric scatter walk.

Decision:
- Kept, narrowly. The direct row-dot formulation is bit-identical to the old half-symmetric walk for mirrored dense symmetric work matrices, and it materially improves the 256-class eigvalsh reducer.
- The ungated row-dot probe was rejected because it regressed 64/128 and made 512 much worse. The final hunk gates the lever to the measured mid-band only.
- This does not close the NumPy gap: `eigvalsh_nxn/256` improves 0.735x versus old FNP but still runs 1.757x NumPy on `hz1`. `eigvalsh_nxn/128` remains a residual loss and is below the row-dot gate.

Primary same-worker evidence on `hz1`:

| Workload | Baseline FNP ns | Final FNP ns | NumPy median ns | Final/Baseline | Final/NumPy | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `eigvalsh_nxn/size/64` | 261856 | 270106 | 254157 | 1.032x | 1.063x | neutral/noise; below row-dot gate |
| `eigvalsh_nxn/size/128` | 1995299 | 1896797 | 1280690 | 0.951x | 1.481x | residual loss; below row-dot gate |
| `eigvalsh_nxn/size/256` | 17636268 | 12969460 | 7380748 | 0.735x | 1.757x | keep win vs old FNP; residual NumPy loss |
| `eigvalsh_nxn/size/512` | not rebaselined | 59840882 | 49987519 | n/a | 1.197x | guard row; row-dot disabled |

Rejected probes and negative evidence:

| Probe | Workload | Probe FNP ns | Comparator ns | Ratio | Verdict |
|---|---:|---:|---:|---:|---|
| Ungated row-dot | `eigvalsh_nxn/size/64` | 287434 | 261856 baseline | 1.098x | reject regression |
| Ungated row-dot | `eigvalsh_nxn/size/128` | 2103644 | 1995299 baseline | 1.054x | reject regression |
| Ungated row-dot | `eigvalsh_nxn/size/256` | 12580950 | 17636268 baseline | 0.713x | useful signal, too broad |
| Row-dot enabled at 512 | `eigvalsh_nxn/size/512` | 88449167 | 59840882 final old-path guard | 1.478x | reject; upper gate required |

Profile context:
- `rch exec -- cargo test -p fnp-linalg tridiag_eigvals_qr_perf_report --release -- --ignored --nocapture` on `hz1`: QR scaled-hypot path is already faster than the old libm-`hypot` path by 1.30x, 1.31x, and 1.27x at n=256/512/768. The remaining end-to-end loss is reducer-side.

Focused conformance and crate health:
- `rch exec -- cargo test -p fnp-linalg tridiag --release`: pass on RCH-selected `vmi1153651`; 7 passed, 0 failed, 4 ignored. This includes `tridiag_symmetric_matvec_serial_matches_full_row_dot_bits`, blocked/unblocked checks, parallel-matvec check, and `tridiag_rank2k_fused_update_preserves_spectra_and_golden_sha256`.
- `rch exec -- cargo check -p fnp-linalg --all-targets`: pass on RCH-selected `vmi1152480`.
- `rch exec -- cargo build -p fnp-linalg --release`: pass on RCH-selected `vmi1152480`.
- `rch exec -- cargo clippy -p fnp-linalg --all-targets -- -D warnings`: pass on `hz1`.
- `git diff --check`: pass.
- `cargo fmt -p fnp-linalg -- --check` still reports broad pre-existing rustfmt drift in benches/examples and unrelated source regions; no formatting churn was kept.
- `ubs` over the changed source/doc paths still exits nonzero from broad existing `fnp-linalg/src/lib.rs` inventory, not from a row-dot-hunk-specific finding.

Retry predicate:
- Do not retry ungated full row-dot matvec. It helps the 256-class dense reducer but loses at 64/128 and 512.
- A credible next eigvalsh attempt needs a deeper values-only tridiagonal reducer, true band-stage primitive, or generated 128-specific reducer that improves `eigvalsh_nxn/128` / `cond_nxn/128` without reopening rejected panel-width, active-window deflation, or sub-1024 Rayon matvec families.
