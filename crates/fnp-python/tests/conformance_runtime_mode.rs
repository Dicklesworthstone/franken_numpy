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
