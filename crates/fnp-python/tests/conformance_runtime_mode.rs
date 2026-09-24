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
print(bad if bad else True)
"#
        .into(),
    )?;
    assert_eq!(
        result.lines().last().unwrap_or(""),
        "True",
        "hardened linalg guard / strict parity: {result}"
    );
    Ok(())
}
