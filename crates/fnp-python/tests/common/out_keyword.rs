//! Canonical `out=` conformance probe for the subprocess-based shards.
//!
//! Five shards check the same contract — that a ufunc wrapper forwards `out=`
//! to numpy, returns the very object it was handed, fills it with the right
//! values, and raises the same exception type on a bad `out` shape. Each of them
//! hand-rolled the probe AND its comparison, and they did not agree:
//!
//! | shard                     | comparison                    | verdict |
//! |---------------------------|-------------------------------|---------|
//! | `conformance_isinf_signed`| `actual != expected`          | strict  |
//! | `conformance_fp_classify` | `actual != expected`          | strict  |
//! | `conformance_trig`        | indices 0,1 only              | FIXED   |
//! | `conformance_rounding`    | indices 0,1 only              | FIXED   |
//! | `conformance_unary_ops`   | indices 0,1 only              | FIXED   |
//!
//! In the weakened form the probe still collected `out.tolist()` — it was
//! gathered and then thrown away, so a wrapper that returned the right object
//! filled with the wrong numbers passed. The values are the *mutation* half of
//! the contract; the aliasing flag alone is not the contract.
//!
//! Emitting the probe from one place makes strictness a property of this file
//! rather than a per-shard decision. Two traps are baked in deliberately:
//!
//! * **Signed zeros.** `-0.0 == 0.0` in Python, so comparing `out.tolist()`
//!   cannot see the sign of a zero — and `negative(-0.0)` is exactly `+0.0`.
//!   The probe carries an explicit signbit tuple so that case is observable.
//! * **Exception text.** numpy rewords its messages between releases and the
//!   supported floor spans several, so the probe reports the exception *type*
//!   only. Pinning text buys no parity and makes a shard version dependent.
//!
//! Tracked by `deadlock-audit-au3z4`.

/// Emit the shared `out_outcome` probe plus the `run_out_cases` driver.
///
/// `x_expr` is the Python expression for the input array, `out_dtype` the dtype
/// the `out` buffer is allocated with, and `out_len` its correct length (the
/// bad-shape case deliberately allocates one element short of it).
///
/// The caller supplies only its `cases` list and then calls
/// `run_out_cases(cases)`, which prints `True` when every case agrees.
pub fn out_keyword_probe_py(x_expr: &str, out_dtype: &str, out_len: usize) -> String {
    let bad_len = out_len.saturating_sub(1).max(1);
    format!(
        r#"
def out_outcome(module, name, positional=False, bad_shape=False):
    fn = getattr(module, name)
    try:
        x = {x_expr}
        out = np.empty(({bad_len},), dtype={out_dtype}) if bad_shape else np.empty({out_len}, dtype={out_dtype})
        if positional:
            result = fn(x, out)
        else:
            result = fn(x, out=out)
        return (
            "ok",
            result is out,
            out.dtype.str,
            tuple(out.shape),
            out.tolist(),
            tuple(bool(b) for b in np.signbit(out)) if out.dtype.kind == "f" else None,
        )
    except Exception as exc:
        return ("err", type(exc).__name__)

def run_out_cases(cases):
    ok = True
    for label, name, positional, bad_shape in cases:
        actual = out_outcome(fnp, name, positional, bad_shape)
        expected = out_outcome(np, name, positional, bad_shape)
        if actual != expected:
            print(label)
            print(actual)
            print(expected)
            ok = False
    print(ok)
"#
    )
}
