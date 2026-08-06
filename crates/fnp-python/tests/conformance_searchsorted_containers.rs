//! Conformance tests for `numpy.searchsorted` on NON-ndarray public surfaces.
//!
//! Covers the container-coercion path the public API exposes (bead
//! `deadlock-audit-81ft6`): a list haystack with list probes, tuple probes with
//! `side="right"`, a Python scalar and a 0-D ndarray query, `sorter` supplied as
//! a Python list, and string-list delegation through NumPy.
//!
//! THE RETURN TYPE VARIES WITH THE QUERY, not with the haystack: a scalar probe
//! yields a `numpy.int64` SCALAR and an array-like probe yields an `ndarray`.
//! Both spellings compare equal on value, so the outcome capture records the
//! type name, dtype, shape, and value — a wrapper that always returned an array
//! (or always an int) would pass a value-only check.

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

/// Capture the full observable outcome of one `searchsorted` call.
///
/// `args_expr` is the complete argument list so a case can carry `side=` or
/// `sorter=` without a second harness shape.
///
/// A `\` line-continuation eats the SOURCE indentation, so the Python indent is
/// injected through `{I4}`/`{I8}` placeholders rather than written as literal
/// spaces; literal spaces would emit a flat script and the oracle would raise
/// `IndentationError` (the harness bug already recorded in
/// `conformance_flatnonzero`).
fn searchsorted_outcome_body(function_expr: &str, args_expr: &str) -> String {
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
        searchsorted_outcome_body("np.searchsorted", args_expr)
    )
}

fn fnp_outcome_script(args_expr: &str) -> String {
    fnp_script(searchsorted_outcome_body("fnp.searchsorted", args_expr))
}

#[test]
fn searchsorted_python_container_surfaces_match_numpy() -> Result<(), String> {
    let cases = [
        ("list haystack scalar probe", "[1, 3, 5, 7], 4"),
        ("list haystack list probes", "[1, 3, 5, 7], [2, 6]"),
        ("tuple haystack list probes", "(1, 3, 5, 7), [2, 6]"),
        (
            "tuple haystack tuple probes side right",
            "(1, 3, 5, 7), (2, 6), side=\"right\"",
        ),
        // side changes the answer exactly on a tie, which is the only place the
        // two sides differ — probing 3 against a haystack containing 3.
        ("exact tie side left", "[1, 3, 5, 7], 3, side=\"left\""),
        ("exact tie side right", "[1, 3, 5, 7], 3, side=\"right\""),
        ("python scalar query", "[1, 3, 5, 7], 4"),
        ("zero-d ndarray query", "[1, 3, 5, 7], np.array(4)"),
        ("python bool query", "[0, 1, 2], True"),
        ("float query into int haystack", "[1, 3, 5, 7], 4.5"),
        ("nan query sorts last", "[1, 3, 5, 7], float('nan')"),
        ("empty probes list", "[1, 3, 5, 7], []"),
        ("empty haystack", "[], 3"),
        ("below and above range", "[1, 3, 5, 7], [0, 99]"),
        (
            "sorter as python list",
            "[1, 3, 5, 7], 4, sorter=[0, 1, 2, 3]",
        ),
        (
            "sorter reorders an unsorted haystack",
            "[5, 3, 1, 7], 4, sorter=[2, 1, 0, 3]",
        ),
        // String haystacks delegate to NumPy's own comparison.
        (
            "string list scalar probe",
            "[\"ant\", \"cat\", \"dog\"], \"bee\"",
        ),
        (
            "string list array probes",
            "[\"ant\", \"cat\", \"dog\"], [\"bee\", \"emu\"]",
        ),
        // Negative cases: these MUST raise with NumPy's exact type and message.
        ("nested haystack is too deep", "[[1, 2], [3, 4]], 3"),
        ("invalid side", "[1, 3, 5, 7], 4, side=\"bogus\""),
        (
            "sorter length mismatch",
            "[1, 3, 5, 7], 4, sorter=[0, 1, 2]",
        ),
    ];

    for (label, args_expr) in cases {
        let numpy_result = numpy_oracle(&numpy_outcome_script(args_expr))?;
        let rust_result = numpy_oracle(&fnp_outcome_script(args_expr))?;

        assert_eq!(
            numpy_result, rust_result,
            "searchsorted Python-container surface mismatch for {label} (args {args_expr})"
        );
    }

    Ok(())
}

/// Guard the guard. The comparison above passes if BOTH arms degrade the same
/// way, so pin the oracle's own contract: scalar probe -> numpy scalar, array
/// probe -> ndarray, and the three negative cases really raise.
#[test]
fn searchsorted_oracle_shape_of_the_contract_is_what_we_think() -> Result<(), String> {
    let scalar = numpy_oracle(&numpy_outcome_script("[1, 3, 5, 7], 4"))?;
    let mut lines = scalar.lines();
    assert_eq!(lines.next(), Some("ok"));
    assert_eq!(
        lines.next(),
        Some("int64"),
        "a scalar probe must return a numpy int64 scalar"
    );

    let arrayed = numpy_oracle(&numpy_outcome_script("[1, 3, 5, 7], [2, 6]"))?;
    let mut lines = arrayed.lines();
    assert_eq!(lines.next(), Some("ok"));
    assert_eq!(
        lines.next(),
        Some("ndarray"),
        "an array-like probe must return an ndarray"
    );

    for (args_expr, expected_exception) in [
        ("[[1, 2], [3, 4]], 3", "ValueError"),
        ("[1, 3, 5, 7], 4, side=\"bogus\"", "ValueError"),
        ("[1, 3, 5, 7], 4, sorter=[0, 1, 2]", "ValueError"),
    ] {
        let outcome = numpy_oracle(&numpy_outcome_script(args_expr))?;
        let mut lines = outcome.lines();
        assert_eq!(
            lines.next(),
            Some("err"),
            "numpy.searchsorted({args_expr}) was expected to raise"
        );
        assert_eq!(
            lines.next(),
            Some(expected_exception),
            "numpy.searchsorted({args_expr}) raised an unexpected exception type"
        );
    }

    Ok(())
}
