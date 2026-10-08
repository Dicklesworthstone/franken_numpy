//! Python-surface contract of the strict/hardened runtime switch (`FNP_RUNTIME_MODE`,
//! `set_runtime_mode`, the decision ledger).
//!
//! Two defects this shard locks out (deadlock-audit-rc0923-epic-71qy3.9):
//! 1. An unknown `FNP_RUNTIME_MODE` was discarded with `let _ =` at import, so a typo such as
//!    `hardend` silently ran Strict. The runtime mode matrix says unknown wire modes fail closed.
//! 2. The process-global ledger was an unbounded `Vec`: 200,000 hardened `clip` calls grew
//!    ~93 MB. It is now bounded and evictions are counted.

mod support;

fn run_python(body: String) -> Result<String, String> {
    let script = support::fnp_script(body);
    let output = support::python_command()
        .args(["-c", &script])
        .output()
        .map_err(|error| format!("python should be available: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "python failed: {}\nScript: {script}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[test]
fn fnp_runtime_mode_env_var_fails_closed_on_unknown_value() -> Result<(), String> {
    let result = run_python(
        r#"
import os, subprocess, sys
loader = (
    "import importlib.util, sys\n"
    "spec = importlib.util.spec_from_file_location('fnp_python', sys.argv[1])\n"
    "m = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(m)\n"
    "print(m.get_runtime_mode())\n"
)
def run(value):
    env = dict(os.environ)
    env.pop("FNP_RUNTIME_MODE", None)
    if value is not None:
        env["FNP_RUNTIME_MODE"] = value
    p = subprocess.run([sys.executable, "-c", loader, spec.origin], env=env,
                       capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr
for value in (None, "", "   ", "strict", "hardened", " hardened "):
    rc, out, err = run(value)
    print(f"{value!r} rc={rc} mode={out}")
for value in ("hardend", "HARDENED", "0"):
    rc, out, err = run(value)
    print(f"{value!r} rc={rc} named={'FNP_RUNTIME_MODE' in err}")
"#
        .into(),
    )?;
    let expected = "\
None rc=0 mode=strict
'' rc=0 mode=strict
'   ' rc=0 mode=strict
'strict' rc=0 mode=strict
'hardened' rc=0 mode=hardened
' hardened ' rc=0 mode=hardened
'hardend' rc=1 named=True
'HARDENED' rc=1 named=True
'0' rc=1 named=True";
    assert_eq!(result, expected, "FNP_RUNTIME_MODE handling drifted");
    Ok(())
}

#[test]
fn fnp_hardened_decision_ledger_is_bounded_and_counts_evictions() -> Result<(), String> {
    let result = run_python(
        r#"
x = np.arange(16.0)
fnp.set_runtime_mode("hardened")
fnp.clear_runtime_decisions()
calls = 10000
for _ in range(calls):
    fnp.clip(x, 2.0, 5.0)
retained = fnp.get_runtime_decision_count()
dropped = fnp.get_runtime_decisions_dropped()
print(retained > 0, retained <= 4096, retained + dropped == calls)
fnp.clear_runtime_decisions()
print(fnp.get_runtime_decision_count(), fnp.get_runtime_decisions_dropped())
fnp.set_runtime_mode("strict")
"#
        .into(),
    )?;
    assert_eq!(
        result, "True True True\n0 0",
        "hardened ledger must stay bounded and account for every eviction"
    );
    Ok(())
}

/// deadlock-audit-3ltbd.5: `get_runtime_decisions()` exports the audit each decision carries -
/// the posterior incompatibility probability, the three expected losses, the loss of the action
/// taken, and the evidence terms - not just the action. The selected loss must be the recorded
/// action's own, and the posterior must move with the evidence: a clean `clip` (risk 0.1) and a
/// non-finite linalg operand (risk 1.0) cannot share one posterior.
#[test]
fn runtime_decisions_export_the_posterior_expected_losses_and_evidence_terms() -> Result<(), String>
{
    let result = run_python(
        r#"
fnp.set_runtime_mode("hardened")
fnp.clear_runtime_decisions()
fnp.clip(np.arange(4.0), 1.0, 2.0)
try:
    fnp.linalg.inv(np.array([[1.0, np.nan], [0.0, 1.0]]))
except Exception as exc:
    print(type(exc).__name__)
events = fnp.get_runtime_decisions()
fnp.set_runtime_mode("strict")
by_reason = {e["reason_code"]: e for e in events}
clip, inv = by_reason["clip_operation"], by_reason["linalg_nonfinite_operand"]
for e in (clip, inv):
    loss = {"allow": e["expected_loss_allow"], "full_validate": e["expected_loss_full_validate"],
            "fail_closed": e["expected_loss_fail_closed"]}[e["action"]]
    names = [name for name, _ in e["evidence_terms"]]
    print(e["action"], 0.0 < e["posterior_incompatible"] < 1.0, loss == e["selected_expected_loss"],
          names == ["prior_class_log_odds", "risk_vs_threshold_llr"], e["ts_millis"] > 0)
print(inv["posterior_incompatible"] > clip["posterior_incompatible"])
"#
        .into(),
    )?;
    assert_eq!(
        result, "LinAlgError\nallow True True True True\nfull_validate True True True True\nTrue",
        "the decision export must carry the audit fields, consistent with the action taken"
    );
    Ok(())
}

/// Bead rc0923 .10 acceptance (first guard): Hardened mode ACTS on the decision engine. A linalg
/// decomposition or solve on an operand carrying inf or NaN is known-compatible input at high
/// risk, which the runtime mode matrix sends to `full_validate`; the validation fails, so the
/// call raises LinAlgError and the ledger holds a `linalg_nonfinite_operand` / `full_validate`
/// event. Strict mode must stay NumPy's outcome exactly (value or exception type) on the same
/// hostile operand, and a finite operand must behave identically in both modes with no guard
/// event. Negative case: with recording only (the pre-fix state), the Hardened raise is absent
/// and this fails.
#[test]
fn hardened_mode_rejects_nonfinite_linalg_operands_strict_matches_numpy() -> Result<(), String> {
    let result = run_python(
        r#"
import numpy.linalg as npl
def outcome(fn):
    try:
        r = fn()
    except Exception as exc:
        return ("raised", type(exc).__name__)
    parts = list(r) if isinstance(r, tuple) else [r]
    return ("ok", [(np.asarray(p).shape, np.asarray(p).dtype.str) for p in parts],
            [np.asarray(p) for p in parts])
def same(a, b):
    if a[0] != b[0] or a[1] != b[1]:
        return False
    return a[0] == "raised" or all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a[2], b[2]))
hostile = np.array([[1.0, np.nan], [np.inf, 2.0]])
finite = np.array([[4.0, 1.0], [1.0, 3.0]])
rhs = np.array([1.0, 2.0])
ops = {
    "svd": lambda m, a: m.linalg.svd(a), "qr": lambda m, a: m.linalg.qr(a),
    "cholesky": lambda m, a: m.linalg.cholesky(a),
    "lstsq": lambda m, a: m.linalg.lstsq(a, rhs, rcond=None),
    "solve": lambda m, a: m.linalg.solve(a, rhs), "inv": lambda m, a: m.linalg.inv(a),
    "det": lambda m, a: m.linalg.det(a), "slogdet": lambda m, a: m.linalg.slogdet(a),
    "eigh": lambda m, a: m.linalg.eigh(a), "eigvals": lambda m, a: m.linalg.eigvals(a),
    "eigvalsh": lambda m, a: m.linalg.eigvalsh(a), "eig": lambda m, a: m.linalg.eig(a),
    "pinv": lambda m, a: m.linalg.pinv(a),
    "matrix_rank": lambda m, a: m.linalg.matrix_rank(a),
}
bad = []
for name, op in ops.items():
    fnp.set_runtime_mode("strict")
    if not same(outcome(lambda: op(fnp, hostile)), outcome(lambda: op(np, hostile))):
        bad.append(f"{name}: strict differs from numpy on a non-finite operand")
    # Finite-operand parity with numpy belongs to each op's own shard (native pinv/lstsq are
    # tolerance-checked there); here the guard's contract is only that Hardened leaves a
    # finite operand's result exactly as Strict computes it.
    strict_finite = outcome(lambda: op(fnp, finite))
    fnp.set_runtime_mode("hardened")
    fnp.clear_runtime_decisions()
    try:
        op(fnp, hostile)
        bad.append(f"{name}: hardened did not raise")
    except Exception as exc:
        if not isinstance(exc, npl.LinAlgError) or "hardened" not in str(exc):
            bad.append(f"{name}: hardened raised {type(exc).__name__}, not the guard: {exc}")
    events = [e for e in fnp.get_runtime_decisions()
              if e["reason_code"] == "linalg_nonfinite_operand"]
    if not events or events[-1]["action"] != "full_validate" or events[-1]["mode"] != "hardened":
        bad.append(f"{name}: no full_validate ledger event")
    fnp.clear_runtime_decisions()
    if not same(outcome(lambda: op(fnp, finite)), strict_finite):
        bad.append(f"{name}: hardened changed a finite operand's result")
    if any(e["reason_code"] == "linalg_nonfinite_operand" for e in fnp.get_runtime_decisions()):
        bad.append(f"{name}: guard fired on a finite operand")
fnp.set_runtime_mode("strict")
fnp.clear_runtime_decisions()
print("HARDENED_LINALG_VERDICT", bad if bad else True, flush=True)
"#
        .into(),
    )?;
    // Tagged, not "the last line": numpy's LAPACK writes " ** On entry to DLASCL parameter ..."
    // diagnostics to stdout for NaN operands on some hosts (worker vmi1227854), after the verdict.
    let verdict = result
        .lines()
        .find_map(|line| line.strip_prefix("HARDENED_LINALG_VERDICT "))
        .unwrap_or("");
    assert_eq!(
        verdict, "True",
        "hardened linalg guard / strict parity: {result}"
    );
    Ok(())
}

/// Hardened ADMISSION CAP (spec section 16, "Shape bomb": strict executes if in limit, hardened
/// enforces stricter admission caps; bead rc0923 .10). With `FNP_HARDENED_MAX_ARRAY_BYTES` at
/// 1 MiB, every guarded creation routine is asked for just over 1 MiB: Strict answers exactly
/// what NumPy answers, Hardened raises MemoryError before allocating and records an
/// `admission_cap_exceeded` / `full_validate` decision. A request of exactly the cap and small
/// requests pass Hardened unchanged with no event. With no override the default 4 GiB cap
/// refuses `zeros(2**29 + 1)` without allocating it, and a malformed override fails the import.
/// Negative case: with the guard removed, every Hardened over-cap call returns an array and
/// this fails.
#[test]
fn hardened_mode_caps_shape_bomb_allocations_strict_matches_numpy() -> Result<(), String> {
    let result = run_python(
        r#"
import os, subprocess, sys
child = r'''
import importlib.util, sys
import numpy as np
spec = importlib.util.spec_from_file_location("fnp_python", sys.argv[1])
fnp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fnp)
if sys.argv[2] == "default":
    fnp.set_runtime_mode("hardened")
    try:
        fnp.zeros(2**29 + 1)
        print("DEFAULT_CAP_VERDICT", "no raise")
    except MemoryError as exc:
        print("DEFAULT_CAP_VERDICT", "hardened admission cap" in str(exc))
    raise SystemExit
CAP = 1 << 20
N8 = CAP // 8
over = {
    "zeros": lambda m: m.zeros(N8 + 1),
    "ones": lambda m: m.ones((N8 + 1,)),
    "empty": lambda m: m.empty(shape=(2, N8 // 2 + 1)),
    "full": lambda m: m.full(N8 + 1, 1.5),
    "full_int8": lambda m: m.full(CAP + 1, 3, dtype=np.int8),
    "zeros_like": lambda m: m.zeros_like(np.ones(3), shape=(N8 + 1,)),
    "ones_like": lambda m: m.ones_like(np.ones(3, np.float32), shape=2 * N8 + 1),
    "full_like": lambda m: m.full_like(np.ones(3), 2.0, shape=(N8 + 1,)),
    "eye": lambda m: m.eye(363),
    "identity": lambda m: m.identity(363),
    "tri": lambda m: m.tri(363, 363),
    "arange": lambda m: m.arange(N8 + 1),
    "arange_float": lambda m: m.arange(0.0, float(N8 + 1), 1.0),
    "linspace": lambda m: m.linspace(0, 1, N8 + 1),
    "logspace": lambda m: m.logspace(0, 1, N8 + 1),
    "geomspace": lambda m: m.geomspace(1, 2, N8 + 1),
    "indices": lambda m: m.indices((256, 257)),
    "repeat": lambda m: m.repeat(np.ones(4), N8 // 4 + 1),
    "tile": lambda m: m.tile(np.ones(4), N8 // 4 + 1),
    "resize": lambda m: m.resize(np.ones(4), N8 + 1),
}
admitted = {
    "zeros_at_cap": lambda m: m.zeros(N8),
    "eye_under_cap": lambda m: m.eye(362),
    "zeros_small": lambda m: m.zeros((3, 4), np.int16),
    "arange_small": lambda m: m.arange(2, 20, 3),
    "repeat_small": lambda m: m.repeat(np.arange(4), [1, 2, 3, 4]),
    "tile_small": lambda m: m.tile(np.arange(3), (2, 2)),
    "indices_small": lambda m: m.indices((2, 3), sparse=True),
    "zeros_like_prototype": lambda m: m.zeros_like(np.ones((5, 5))),
}
def outcome(fn, content=True):
    try:
        r = fn()
    except Exception as exc:
        return ("raised", type(exc).__name__)
    parts = r if isinstance(r, tuple) else (r,)
    return ("ok", [(np.asarray(p).dtype.str, np.asarray(p).shape,
                    np.asarray(p).tobytes() if content else None) for p in parts])
def cap_events():
    return [e for e in fnp.get_runtime_decisions() if e["reason_code"] == "admission_cap_exceeded"]
bad = []
for name, call in over.items():
    fnp.set_runtime_mode("strict")
    fnp.clear_runtime_decisions()
    # `empty`'s content is unspecified, so only its dtype and shape can be compared.
    content = not name.startswith("empty")
    if outcome(lambda: call(fnp), content) != outcome(lambda: call(np), content):
        bad.append(f"{name}: strict differs from numpy")
    if cap_events():
        bad.append(f"{name}: strict recorded an admission event")
    fnp.set_runtime_mode("hardened")
    fnp.clear_runtime_decisions()
    try:
        call(fnp)
        bad.append(f"{name}: hardened admitted an over-cap request")
    except MemoryError as exc:
        if "hardened admission cap" not in str(exc):
            bad.append(f"{name}: MemoryError without the guard's message: {exc}")
    except Exception as exc:
        bad.append(f"{name}: hardened raised {type(exc).__name__}: {exc}")
    events = cap_events()
    if not events or events[-1]["action"] != "full_validate" or events[-1]["mode"] != "hardened":
        bad.append(f"{name}: no full_validate admission event")
for name, call in admitted.items():
    fnp.set_runtime_mode("hardened")
    fnp.clear_runtime_decisions()
    if outcome(lambda: call(fnp)) != outcome(lambda: call(np)):
        bad.append(f"{name}: hardened changed an admitted request")
    if cap_events():
        bad.append(f"{name}: admission event on an admitted request")
fnp.set_runtime_mode("strict")
print("ADMISSION_CAP_VERDICT", bad if bad else True, len(over), len(admitted))
'''
def run(cap, mode_arg):
    env = dict(os.environ)
    env.pop("FNP_RUNTIME_MODE", None)
    env.pop("FNP_HARDENED_MAX_ARRAY_BYTES", None)
    if cap is not None:
        env["FNP_HARDENED_MAX_ARRAY_BYTES"] = cap
    return subprocess.run([sys.executable, "-c", child, spec.origin, mode_arg], env=env,
                          capture_output=True, text=True)
p = run(str(1 << 20), "capped")
print(p.stdout.strip() or p.stderr.strip()[-600:])
p = run(None, "default")
print(p.stdout.strip() or p.stderr.strip()[-600:])
for value in ("0", "-5", "1GB", "1.5"):
    p = run(value, "capped")
    print(f"BAD_CAP {value!r} rc={p.returncode} named={'FNP_HARDENED_MAX_ARRAY_BYTES' in p.stderr}")
"#
        .into(),
    )?;
    let tagged = |tag: &str| {
        result
            .lines()
            .find_map(|line| line.strip_prefix(tag))
            .unwrap_or("")
            .to_string()
    };
    assert_eq!(
        tagged("ADMISSION_CAP_VERDICT "),
        "True 20 8",
        "hardened admission cap / strict parity: {result}"
    );
    assert_eq!(
        tagged("DEFAULT_CAP_VERDICT "),
        "True",
        "the default cap must refuse 4 GiB + 8 bytes: {result}"
    );
    for value in ["'0'", "'-5'", "'1GB'", "'1.5'"] {
        assert!(
            result.contains(&format!("BAD_CAP {value} rc=1 named=True")),
            "a malformed FNP_HARDENED_MAX_ARRAY_BYTES={value} must fail the import: {result}"
        );
    }
    Ok(())
}

/// Hardened mode keeps the packet-007 spawn budget ("Explicit bounded caps (hardened policy
/// path)" in its risk note) that strict mode dropped for numpy parity (deadlock-audit-r8eqg):
/// `SeedSequence.spawn` above 4096 children per call raises ValueError and records a
/// `rng_seedsequence_spawn_contract_violation` / `full_validate` decision - for the seed sequence
/// directly and through a bit generator's `spawn` - while a budget-sized call is unchanged and
/// records nothing. Strict mode, in the same process, spawns the 5000 children numpy does.
#[test]
fn hardened_mode_keeps_the_seed_sequence_spawn_budget() -> Result<(), String> {
    let result = run_python(
        r#"
def budget_events():
    return [e for e in fnp.get_runtime_decisions()
            if e["reason_code"] == "rng_seedsequence_spawn_contract_violation"]

bad = []
fnp.set_runtime_mode("strict")
if len(fnp.random.SeedSequence(1).spawn(5000)) != 5000:
    bad.append("strict spawn(5000) did not return 5000 children")
fnp.set_runtime_mode("hardened")
for label, call in (("SeedSequence", lambda: fnp.random.SeedSequence(1).spawn(5000)),
                    ("PCG64", lambda: fnp.random.PCG64(1).spawn(5000))):
    fnp.clear_runtime_decisions()
    try:
        call()
        bad.append(f"{label}: hardened spawn(5000) was admitted")
    except ValueError as exc:
        if "hardened spawn budget" not in str(exc):
            bad.append(f"{label}: ValueError without the budget message: {exc}")
    events = budget_events()
    if not events or events[-1]["action"] != "full_validate" or events[-1]["mode"] != "hardened":
        bad.append(f"{label}: no full_validate spawn contract event")
fnp.clear_runtime_decisions()
ours = [tuple(k.spawn_key) for k in fnp.random.SeedSequence(1).spawn(4096)[::1000]]
theirs = [tuple(k.spawn_key) for k in np.random.SeedSequence(1).spawn(4096)[::1000]]
if ours != theirs:
    bad.append("hardened spawn(4096) differs from numpy")
if budget_events():
    bad.append("a budget-sized spawn recorded a budget event")
fnp.set_runtime_mode("strict")
print("SPAWN_BUDGET_VERDICT", bad if bad else True)
"#
        .into(),
    )?;
    let verdict = result
        .lines()
        .find_map(|line| line.strip_prefix("SPAWN_BUDGET_VERDICT "))
        .unwrap_or("");
    assert_eq!(
        verdict, "True",
        "hardened spawn budget / strict parity: {result}"
    );
    Ok(())
}

/// NON-NATIVE BYTE ORDER (runtime mode matrix, bead rc0923 .10). pyo3 accepts a big-endian
/// buffer as a native `T`, so the zero-copy routes once read `>f8` bytes raw and returned silent
/// wrong values (15 of 65 ops, 2026-09-02). The crate's `PyBuffer` refuses such a buffer and the
/// route declines to NumPy - in BOTH modes, since NumPy computes the right answer - and Hardened
/// records the decision (`non_native_byte_order_decline`, full_validate: a known-compatible
/// high-risk input, validated by handing it to the implementation that owns it). Strict records
/// nothing. Negative case, measured on a build with the refusal removed: Hardened records no
/// decision and this fails. The sixteen values below still matched NumPy on that build - the
/// routes' own byte-order checks now cover them - so the refusal is the backstop for the sites
/// that lack one, and the value half of this test guards those checks.
#[test]
fn non_native_byte_order_operands_match_numpy_in_both_modes_and_hardened_audits_them()
-> Result<(), String> {
    let result = run_python(
        r#"
rng = np.random.default_rng(12)
f = rng.standard_normal(1 << 16).astype(">f8")
i = rng.integers(-1000, 1000, 1 << 16).astype(">i8")
m = rng.standard_normal((256, 256)).astype(">f8")
ops = {
    "sum": lambda mod: mod.sum(f), "cumsum": lambda mod: mod.cumsum(f), "max": lambda mod: mod.max(f),
    "argmax": lambda mod: mod.argmax(f), "std": lambda mod: mod.std(f), "sort": lambda mod: mod.sort(f),
    "add": lambda mod: mod.add(f, f), "multiply": lambda mod: mod.multiply(f, 2.0),
    "clip": lambda mod: mod.clip(f, -0.5, 0.5), "sort i8": lambda mod: mod.sort(i),
    "unique i8": lambda mod: mod.unique(i), "isin i8": lambda mod: mod.isin(i, i[:100]),
    "searchsorted": lambda mod: mod.searchsorted(np.sort(f), f[:512]),
    "nan_to_num": lambda mod: mod.nan_to_num(f), "matmul": lambda mod: mod.matmul(m, m),
    "sum axis 0": lambda mod: mod.sum(m, axis=0),
}
def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
def audits():
    return [e for e in fnp.get_runtime_decisions() if e["reason_code"] == "non_native_byte_order_decline"]
bad = []
for mode in ("strict", "hardened"):
    fnp.set_runtime_mode(mode)
    fnp.clear_runtime_decisions()
    for name, op in ops.items():
        if not same(op(fnp), op(np)):
            bad.append(f"{mode} {name}: differs from numpy on a big-endian operand")
    events = audits()
    if mode == "strict" and events:
        bad.append("strict recorded a byte-order decision")
    if mode == "hardened" and not any(e["mode"] == "hardened" and e["action"] == "full_validate" for e in events):
        bad.append(f"hardened recorded no full_validate byte-order decision: {events[:1]}")
fnp.set_runtime_mode("strict")
fnp.clear_runtime_decisions()
print("BYTE_ORDER_VERDICT", bad if bad else True)
"#
        .into(),
    )?;
    let verdict = result
        .lines()
        .find_map(|line| line.strip_prefix("BYTE_ORDER_VERDICT "))
        .unwrap_or("");
    assert_eq!(
        verdict, "True",
        "non-native byte order parity / hardened audit: {result}"
    );
    Ok(())
}
