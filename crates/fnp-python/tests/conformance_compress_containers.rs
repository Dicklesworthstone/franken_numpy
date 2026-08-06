//! Conformance tests for `numpy.compress` on NON-ndarray public surfaces.
//!
//! `conformance_compress_choose_diagonal.rs` covers the ndarray paths. This shard
//! covers the container-coercion path the public API exposes (bead
//! `deadlock-audit-g70zt`): a Python list condition over a list payload, a tuple
//! condition, a SHORT condition that truncates, nested-list axis selection,
//! string-payload dtype preservation, and the mismatched-axis error surface.
//!
//! Two behaviours here are easy to get subtly wrong and are pinned explicitly:
//! a condition SHORTER than the axis silently selects only as far as it reaches,
//! while a condition LONGER than the axis is an IndexError — so truncation is
//! not symmetric. And with no `axis` the payload is flattened first, which is a
//! different result from `axis=0` on the same nested input.

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

/// Capture the full observable outcome of one `compress` call.
///
/// A `\` line-continuation eats the SOURCE indentation, so the Python indent is
/// injected through `{I4}`/`{I8}` placeholders rather than written as literal
/// spaces; literal spaces would emit a flat script and the oracle would raise
/// `IndentationError` (the harness bug already recorded in
/// `conformance_flatnonzero`).
fn compress_outcome_body(function_expr: &str, args_expr: &str) -> String {
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
        compress_outcome_body("np.compress", args_expr)
    )
}

fn fnp_outcome_script(args_expr: &str) -> String {
    fnp_script(compress_outcome_body("fnp.compress", args_expr))
}

#[test]
fn compress_python_container_surfaces_match_numpy() -> Result<(), String> {
    let cases = [
        ("list condition list payload", "[0, 1, 1, 0], [1, 2, 3, 4]"),
        ("tuple condition", "(1, 0, 1), [10, 20, 30]"),
        ("tuple condition tuple payload", "(1, 0, 1), (10, 20, 30)"),
        ("bool list condition", "[True, False, True], [1, 2, 3]"),
        // Asymmetric: short truncates, long raises. Both pinned.
        ("shorter condition truncates", "[1, 1], [1, 2, 3, 4]"),
        ("single element condition", "[1], [1, 2, 3]"),
        ("empty condition", "[], [1, 2, 3]"),
        ("all false selects nothing", "[0, 0], [1, 2]"),
        ("all true selects everything", "[1, 1, 1], [1, 2, 3]"),
        // No axis flattens first; axis=0 does not. Different results, same input.
        (
            "nested list no axis flattens",
            "[0, 1, 1, 0], [[1, 2], [3, 4]]",
        ),
        ("nested list axis 0", "[0, 1], [[1, 2], [3, 4]], axis=0"),
        ("nested list axis 1", "[0, 1], [[1, 2], [3, 4]], axis=1"),
        (
            "nested list negative axis",
            "[1, 0], [[1, 2], [3, 4]], axis=-1",
        ),
        ("nested tuple axis 0", "(0, 1), ((1, 2), (3, 4)), axis=0"),
        // dtype must survive the container round trip, not decay to object.
        (
            "string payload preserves dtype",
            "[1, 0, 1], [\"ant\", \"bee\", \"cat\"]",
        ),
        ("float payload", "[1, 0, 1], [1.5, 2.5, 3.5]"),
        ("mixed int float payload", "[1, 0, 1], [1, 2.5, 3]"),
        // Negative cases: three distinct error classes, all MUST raise.
        (
            "longer condition is an IndexError",
            "[1, 1, 1, 1, 1], [1, 2]",
        ),
        ("axis out of bounds", "[1, 0], [1, 2], axis=1"),
        (
            "two-dimensional condition",
            "[[1, 0], [0, 1]], [[1, 2], [3, 4]], axis=0",
        ),
    ];

    for (label, args_expr) in cases {
        let numpy_result = numpy_oracle(&numpy_outcome_script(args_expr))?;
        let rust_result = numpy_oracle(&fnp_outcome_script(args_expr))?;

        assert_eq!(
            numpy_result, rust_result,
            "compress Python-container surface mismatch for {label} (args {args_expr})"
        );
    }

    Ok(())
}

/// Guard the guard. The comparison above still passes if BOTH arms degrade the
/// same way, so pin the oracle's own contract: the short/long asymmetry, the
/// flatten-without-axis rule, and that all three negative cases really raise
/// with the exception type this shard claims.
#[test]
fn compress_oracle_shape_of_the_contract_is_what_we_think() -> Result<(), String> {
    // A SHORT condition truncates silently: 2 flags over a length-4 payload.
    let short = numpy_oracle(&numpy_outcome_script("[1, 1], [1, 2, 3, 4]"))?;
    let short: Vec<&str> = short.lines().collect();
    assert_eq!(short.first(), Some(&"ok"), "short condition must not raise");
    assert_eq!(
        short.get(4),
        Some(&"[1, 2]"),
        "short condition must select only as far as it reaches"
    );

    // No axis flattens, so the same flags give a 1-D answer, not a row select.
    let flat = numpy_oracle(&numpy_outcome_script("[0, 1, 1, 0], [[1, 2], [3, 4]]"))?;
    let flat: Vec<&str> = flat.lines().collect();
    assert_eq!(flat.first(), Some(&"ok"));
    assert_eq!(
        flat.get(3),
        Some(&"(2,)"),
        "compress without axis must flatten the payload first"
    );

    for (args_expr, expected_exception) in [
        ("[1, 1, 1, 1, 1], [1, 2]", "IndexError"),
        ("[1, 0], [1, 2], axis=1", "AxisError"),
        ("[[1, 0], [0, 1]], [[1, 2], [3, 4]], axis=0", "ValueError"),
    ] {
        let outcome = numpy_oracle(&numpy_outcome_script(args_expr))?;
        let mut lines = outcome.lines();
        assert_eq!(
            lines.next(),
            Some("err"),
            "numpy.compress({args_expr}) was expected to raise"
        );
        assert_eq!(
            lines.next(),
            Some(expected_exception),
            "numpy.compress({args_expr}) raised an unexpected exception type"
        );
    }

    Ok(())
}
