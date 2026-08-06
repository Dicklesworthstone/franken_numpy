//! Conformance tests for `numpy.count_nonzero` on NON-ndarray public surfaces.
//!
//! `conformance_count_nonzero_zerocopy.rs` covers the ndarray fast path. This
//! shard covers the container-coercion path the public API exposes — Python
//! lists, tuples, nested sequences, scalars, object arrays, and ragged input —
//! together with the `axis` / `keepdims` keywords (bead
//! `deadlock-audit-vyyk3`).
//!
//! THE RETURN TYPE IS PART OF THE CONTRACT, and it is not uniform: with no axis
//! `count_nonzero` returns a `numpy.int64` SCALAR, while `axis` or
//! `keepdims=True` make it return an `ndarray`. A wrapper handing back a plain
//! Python `int` compares equal on value and still breaks any caller that reads
//! `.dtype` or `.shape`, so the outcome capture records the type name, dtype,
//! shape, and value rather than the value alone.

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

/// Capture the full observable outcome of one `count_nonzero` call.
///
/// `args_expr` is the complete argument list, so a case can carry keywords
/// (`axis=0`, `keepdims=True`) without a second harness shape.
///
/// A `\` line-continuation eats the SOURCE indentation, so the Python indent is
/// injected through `{I4}`/`{I8}` placeholders rather than written as literal
/// spaces; literal spaces would emit a flat script and the oracle would raise
/// `IndentationError` (the harness bug already recorded in
/// `conformance_flatnonzero`).
fn count_nonzero_outcome_body(function_expr: &str, args_expr: &str) -> String {
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
        count_nonzero_outcome_body("np.count_nonzero", args_expr)
    )
}

fn fnp_outcome_script(args_expr: &str) -> String {
    fnp_script(count_nonzero_outcome_body("fnp.count_nonzero", args_expr))
}

/// Every non-ndarray surface the bead enumerates, compared outcome-for-outcome.
#[test]
fn count_nonzero_python_container_surfaces_match_numpy() -> Result<(), String> {
    let cases = [
        ("list fallback", "[0, 2, 0, 3]"),
        ("tuple fallback", "(0, 0, 5, 0)"),
        ("bool list", "[False, True, False, True]"),
        ("all-zero list", "[0, 0, 0]"),
        ("empty list", "[]"),
        ("nested list count", "[[0, 1], [2, 0]]"),
        ("nested tuple count", "((0, 1), (2, 0))"),
        ("three-dimensional nested list", "[[[0, 1]], [[2, 0]]]"),
        ("mixed int float list", "[0, 1.5, 0.0, -2]"),
        ("signed zero and nan list", "[-0.0, float('nan'), 2.5, 0.0]"),
        // NaN is truthy and -0.0 is falsy; a naive `!= 0` on the raw objects and
        // a bool-cast disagree here, so this case separates them.
        (
            "object truthiness",
            "np.array(['', 'x', '0'], dtype=object)",
        ),
        // Scalars COUNT (unlike numpy.nonzero, which rejects 0-d input).
        ("scalar nonzero", "7"),
        ("scalar zero", "0"),
        ("zero-d ndarray", "np.array(7)"),
        ("python bool scalar", "True"),
        // axis / keepdims change the RETURN TYPE from a numpy scalar to an
        // ndarray, and keepdims alone still reduces to a 1x1.
        ("axis 0 on nested list", "[[0, 1], [2, 0]], axis=0"),
        ("axis 1 on nested list", "[[0, 1], [2, 0]], axis=1"),
        ("negative axis on nested list", "[[0, 1], [2, 0]], axis=-1"),
        ("axis tuple on nested list", "[[0, 1], [2, 0]], axis=(0, 1)"),
        ("keepdims on nested list", "[[0, 1], [2, 0]], keepdims=True"),
        (
            "axis 0 with keepdims",
            "[[0, 1], [2, 0]], axis=0, keepdims=True",
        ),
        ("keepdims on flat list", "[0, 2, 0, 3], keepdims=True"),
        // Negative cases: these MUST raise, with NumPy's exact type and message.
        ("ragged list error", "[[1], [0, 2]]"),
        ("axis out of bounds", "[0, 1], axis=1"),
    ];

    for (label, args_expr) in cases {
        let numpy_result = numpy_oracle(&numpy_outcome_script(args_expr))?;
        let rust_result = numpy_oracle(&fnp_outcome_script(args_expr))?;

        assert_eq!(
            numpy_result, rust_result,
            "count_nonzero Python-container surface mismatch for {label} (args {args_expr})"
        );
    }

    Ok(())
}

/// Guard the guard. The comparison above would still pass if BOTH arms stopped
/// raising, or if the no-axis return silently became a plain Python `int` in
/// both. Pin the oracle's own behaviour directly so those cannot pass unnoticed.
#[test]
fn count_nonzero_oracle_shape_of_the_contract_is_what_we_think() -> Result<(), String> {
    // No axis returns a numpy scalar named int64, NOT a Python int.
    let scalar = numpy_oracle(&numpy_outcome_script("[0, 2, 0, 3]"))?;
    let mut lines = scalar.lines();
    assert_eq!(lines.next(), Some("ok"));
    assert_eq!(
        lines.next(),
        Some("int64"),
        "no-axis count_nonzero must return a numpy int64 scalar"
    );

    // axis makes it an ndarray.
    let axed = numpy_oracle(&numpy_outcome_script("[[0, 1], [2, 0]], axis=0"))?;
    let mut lines = axed.lines();
    assert_eq!(lines.next(), Some("ok"));
    assert_eq!(
        lines.next(),
        Some("ndarray"),
        "count_nonzero with axis must return an ndarray"
    );

    // Both negative cases really raise.
    for (args_expr, expected_exception) in [
        ("[[1], [0, 2]]", "ValueError"),
        ("[0, 1], axis=1", "AxisError"),
    ] {
        let outcome = numpy_oracle(&numpy_outcome_script(args_expr))?;
        let mut lines = outcome.lines();
        assert_eq!(
            lines.next(),
            Some("err"),
            "numpy.count_nonzero({args_expr}) was expected to raise"
        );
        assert_eq!(
            lines.next(),
            Some(expected_exception),
            "numpy.count_nonzero({args_expr}) raised an unexpected exception type"
        );
    }

    Ok(())
}
