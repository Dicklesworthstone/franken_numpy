//! Conformance tests for `numpy.extract` on NON-ndarray public surfaces.
//!
//! `conformance_extract_put.rs` covers the ndarray paths. This shard covers the
//! container-coercion path the public API exposes (bead `deadlock-audit-qkt72`):
//! a Python list condition over a list payload, tuple condition and tuple
//! payload, nested-list flattening, short-condition truncation and the
//! over-long-condition error, string payloads, and a scalar payload.
//!
//! `extract` is NOT `compress` with the axis dropped. It flattens BOTH operands
//! and always returns a 1-D result, so a nested condition over a flat payload
//! and a flat condition over a nested payload both work and agree. Its condition
//! is also evaluated by TRUTHINESS, so a float `0.0` is a reject and `2.5` is a
//! select — a `!= 0` integer coercion and a bool cast disagree on that, which is
//! why a float condition is pinned here.

use std::process::Command;

fn numpy_oracle(script: &str) -> Result<String, String> {
    let output = Command::new("python3")
        .args(["-c", script])
        .output()
        .map_err(|error| format!("python3 should be available: {error}\nScript: {script}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("NumPy oracle failed: {stderr}\nScript: {script}"));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn fnp_script(body: String) -> String {
    let library_name = format!(
        "{}fnp_python{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    );
    let module_path = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.join(&library_name)))
        .unwrap_or_else(|| library_name.into());
    let module_literal = format!("{module_path:?}");
    format!(
        "import importlib.util\n\
         import numpy as np\n\
         spec = importlib.util.spec_from_file_location('fnp_python', {module_literal})\n\
         fnp = importlib.util.module_from_spec(spec)\n\
         spec.loader.exec_module(fnp)\n\
         {body}"
    )
}

/// Capture the full observable outcome of one `extract` call.
///
/// A `\` line-continuation eats the SOURCE indentation, so the Python indent is
/// injected through `{I4}`/`{I8}` placeholders rather than written as literal
/// spaces; literal spaces would emit a flat script and the oracle would raise
/// `IndentationError` (the harness bug already recorded in
/// `conformance_flatnonzero`).
fn extract_outcome_body(function_expr: &str, args_expr: &str) -> String {
    format!(
        "def outcome(fn):\n\
         {I4}try:\n\
         {I8}value = fn({args_expr})\n\
         {I8}arr = np.asarray(value)\n\
         {I8}print('ok')\n\
         {I8}print(type(value).__name__)\n\
         {I8}print(str(arr.dtype))\n\
         {I8}print(tuple(arr.shape))\n\
         {I8}print(arr.tolist())\n\
         {I4}except Exception as exc:\n\
         {I8}print('err')\n\
         {I8}print(type(exc).__name__)\n\
         {I8}print(str(exc))\n\
         outcome({function_expr})",
        I4 = "    ",
        I8 = "        ",
    )
}

fn numpy_outcome_script(args_expr: &str) -> String {
    format!(
        "import numpy as np\n{}",
        extract_outcome_body("np.extract", args_expr)
    )
}

fn fnp_outcome_script(args_expr: &str) -> String {
    fnp_script(extract_outcome_body("fnp.extract", args_expr))
}

#[test]
fn extract_python_container_surfaces_match_numpy() -> Result<(), String> {
    let cases = [
        ("list condition list payload", "[0, 1, 1, 0], [1, 2, 3, 4]"),
        ("tuple condition", "(1, 0, 1), [10, 20, 30]"),
        ("tuple condition tuple payload", "(1, 0, 1), (10, 20, 30)"),
        ("bool list condition", "[True, False, True], [1, 2, 3]"),
        // Both operands flatten, so these three agree on the same answer.
        (
            "nested condition nested payload",
            "[[0, 1], [1, 0]], [[1, 2], [3, 4]]",
        ),
        (
            "nested condition flat payload",
            "[[0, 1], [1, 0]], [1, 2, 3, 4]",
        ),
        (
            "flat condition nested payload",
            "[0, 1, 1, 0], [[1, 2], [3, 4]]",
        ),
        // Short truncates, long raises — the same asymmetry compress has.
        ("shorter condition truncates", "[1, 1], [1, 2, 3, 4]"),
        ("single element condition", "[1], [1, 2, 3]"),
        ("empty condition", "[], [1, 2, 3]"),
        ("all false selects nothing", "[0, 0], [1, 2]"),
        ("all true selects everything", "[1, 1, 1], [1, 2, 3]"),
        // Condition is TRUTHINESS, not != 0 on an integer coercion.
        ("float condition truthiness", "[0.0, 2.5, 0.0], [1, 2, 3]"),
        ("negative condition is truthy", "[-1, 0, -2], [1, 2, 3]"),
        // dtype must survive the container round trip, not decay to object.
        (
            "string payload preserves dtype",
            "[1, 0, 1], [\"ant\", \"bee\", \"cat\"]",
        ),
        ("float payload", "[1, 0, 1], [1.5, 2.5, 3.5]"),
        ("mixed int float payload", "[1, 0, 1], [1, 2.5, 3]"),
        // A scalar payload is legal and comes back as shape (1,), not 0-d.
        ("scalar payload", "1, 5"),
        ("scalar payload false condition", "0, 5"),
        // Negative case: an over-long condition MUST raise.
        ("longer condition is an IndexError", "[1, 1, 1, 1], [1, 2]"),
    ];

    for (label, args_expr) in cases {
        let numpy_result = numpy_oracle(&numpy_outcome_script(args_expr))?;
        let rust_result = numpy_oracle(&fnp_outcome_script(args_expr))?;

        assert_eq!(
            numpy_result, rust_result,
            "extract Python-container surface mismatch for {label} (args {args_expr})"
        );
    }

    Ok(())
}

/// Guard the guard. The comparison above still passes if BOTH arms degrade the
/// same way, so pin the oracle's own contract: the flatten-both rule, the
/// truthiness condition, the scalar-payload shape, and that the over-long
/// condition really raises.
#[test]
fn extract_oracle_shape_of_the_contract_is_what_we_think() -> Result<(), String> {
    // Nested condition over a nested payload flattens to 1-D, not a 2-D select.
    let nested = numpy_oracle(&numpy_outcome_script("[[0, 1], [1, 0]], [[1, 2], [3, 4]]"))?;
    let nested: Vec<&str> = nested.lines().collect();
    assert_eq!(nested.first(), Some(&"ok"));
    assert_eq!(
        nested.get(3),
        Some(&"(2,)"),
        "extract must flatten both operands and return 1-D"
    );
    assert_eq!(nested.get(4), Some(&"[2, 3]"));

    // A float condition selects on truthiness: 0.0 rejects, 2.5 selects.
    let truthy = numpy_oracle(&numpy_outcome_script("[0.0, 2.5, 0.0], [1, 2, 3]"))?;
    let truthy: Vec<&str> = truthy.lines().collect();
    assert_eq!(truthy.first(), Some(&"ok"));
    assert_eq!(
        truthy.get(4),
        Some(&"[2]"),
        "a float condition must select on truthiness"
    );

    // A scalar payload comes back as shape (1,), not 0-d.
    let scalar = numpy_oracle(&numpy_outcome_script("1, 5"))?;
    let scalar: Vec<&str> = scalar.lines().collect();
    assert_eq!(scalar.first(), Some(&"ok"));
    assert_eq!(
        scalar.get(3),
        Some(&"(1,)"),
        "a scalar payload must extract to shape (1,)"
    );

    let outcome = numpy_oracle(&numpy_outcome_script("[1, 1, 1, 1], [1, 2]"))?;
    let mut lines = outcome.lines();
    assert_eq!(
        lines.next(),
        Some("err"),
        "an over-long condition was expected to raise"
    );
    assert_eq!(lines.next(), Some("IndexError"));

    Ok(())
}
