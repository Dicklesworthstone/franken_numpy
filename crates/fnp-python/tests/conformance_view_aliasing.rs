//! VIEW-vs-COPY parity: does the result alias its input the way numpy's does?
//!
//! This is a different question from the sweeps in `conformance_return_types.rs`
//! (what comes back). A copy where numpy returns a view is SILENT on every axis
//! those check - same values, same dtype, same shape, same return type, same
//! exception class. The only observable difference is that writing through the
//! result stops updating the base, so user code doing
//! `v = np.reshape(a, ...); v[0] = 9` quietly stops mutating `a`.
//!
//! It is a known class here, with a recorded past fix. reshape's own comment:
//! "np.reshape returns a VIEW when the new shape is stride-compatible (a copy
//! otherwise) - pure metadata. The old native path always materialized a copy
//! (slow, a view-semantics divergence, and it widened narrow dtypes)." And
//! trim_zeros': "Zero-copy slice-view ... returns a view like numpy." The
//! project has already shipped and fixed this bug once; nothing swept for the
//! next one.
//!
//! The converse matters too: `np.broadcast_to` returns a READ-ONLY view, so a
//! wrapper handing back a writeable copy lets callers mutate what numpy froze.

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

#[test]
fn view_aliasing_and_writeability_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import platform
import warnings

warnings.simplefilter("ignore")

# MUTATION PROPAGATION is the assertion that actually catches a copy: `.base`
# alone can be non-None on a copy that happens to own a temporary, and it is
# None on `asarray(a)` which returns `a` ITSELF and therefore does alias.
#
# The write must be layout-safe. An earlier draft wrote through
# `r.reshape(-1)[0]`, which COPIES for a transposed view - it reported
# "does not propagate" for transpose/swapaxes/moveaxis, which was an artifact of
# the probe, not of numpy. Indexing the first element with a tuple of zeros
# writes to the view itself whatever its strides.
def observe(module, call):
    a = np.arange(6, dtype=np.float64)
    try:
        r = call(module, a)
    except Exception as exc:
        return ("err", type(exc).__name__)
    same_object = r is a
    writeable = bool(r.flags.writeable)
    propagates = None
    if writeable and r.size:
        before = a.copy()
        r[tuple([0] * r.ndim)] = 99.0
        propagates = not np.array_equal(a, before)
    # `.base` is compared for BOTH groups. It was briefly exempted for the copy
    # group because fnp's concatenate filled a flat uintN buffer and returned
    # `.view(dtype).reshape(shape)`, leaving `.base` pointing at a private
    # temporary where numpy's is None. That is now fixed - the mover allocates at
    # the final dtype/shape and fills through a view of the output
    # (deadlock-audit-concatenate-base-attribute-st00f) - so the exemption is
    # withdrawn and the assertion is back at full strength.
    return ("ok", same_object, r.base is not None, writeable, propagates)

view_ops = [
    ("reshape", lambda m, a: m.reshape(a, (3, 2))),
    ("ravel", lambda m, a: m.ravel(a)),
    ("transpose", lambda m, a: m.transpose(m.reshape(a, (2, 3)))),
    ("swapaxes", lambda m, a: m.swapaxes(m.reshape(a, (2, 3)), 0, 1)),
    ("moveaxis", lambda m, a: m.moveaxis(m.reshape(a, (2, 3)), 0, 1)),
    ("squeeze", lambda m, a: m.squeeze(m.reshape(a, (1, 6)))),
    ("expand_dims", lambda m, a: m.expand_dims(a, 0)),
    ("flip", lambda m, a: m.flip(a)),
    ("diagonal", lambda m, a: m.diagonal(m.reshape(a, (2, 3)))),
    ("broadcast_to", lambda m, a: m.broadcast_to(a, (2, 6))),
    ("real", lambda m, a: m.real(a)),
    ("atleast_1d", lambda m, a: m.atleast_1d(a)),
    ("atleast_2d", lambda m, a: m.atleast_2d(a)),
    ("asarray same dtype", lambda m, a: m.asarray(a)),
    ("asanyarray same dtype", lambda m, a: m.asanyarray(a)),
    ("ascontiguousarray", lambda m, a: m.ascontiguousarray(a)),
    ("trim_zeros", lambda m, a: m.trim_zeros(a)),
]

# MUST COPY. Without this group the sweep could pass by asserting everything
# aliases, which is the opposite error and just as wrong.
copy_ops = [
    ("np.array copy", lambda m, a: m.array(a)),
    ("astype same dtype", lambda m, a: m.asarray(a).astype(np.float64)),
    ("copy", lambda m, a: m.copy(a)),
    ("concatenate", lambda m, a: m.concatenate([a, a])),
    ("sort", lambda m, a: m.sort(a)),
    ("cumsum", lambda m, a: m.cumsum(a)),
]

ok = True
for group, cases in (("view", view_ops), ("copy", copy_ops)):
    for label, call in cases:
        actual = observe(fnp, call)
        expected = observe(np, call)
        if actual != expected:
            print(f"{group}: {label}")
            print(f"  fnp   {actual}")
            print(f"  numpy {expected}")
            ok = False

# Preconditions. The two groups must genuinely differ on numpy, or "matches
# numpy" is satisfied by a build that copies (or aliases) everywhere. Every
# copy-group member must still fail to write through on numpy, and reshape must
# still write through, or the groups are no longer testing opposite things.
#
# observe returns ("ok", same_object, base_is_not_none, writeable, propagates),
# so propagation is index 4 and writeability is index 3. Naming them here
# because an earlier edit shifted this tuple and left these lookups reading
# `writeable` where they meant `propagates`.
PROPAGATES, WRITEABLE = 4, 3
if observe(np, lambda m, a: m.reshape(a, (3, 2)))[PROPAGATES] is not True:
    print("PRECONDITION LOST: numpy's reshape no longer writes through to the base")
    ok = False
for label, call in copy_ops:
    if observe(np, call)[PROPAGATES] is not False:
        print(f"PRECONDITION LOST: numpy's {label} now writes through to the base")
        ok = False
if observe(np, lambda m, a: m.broadcast_to(a, (2, 6)))[WRITEABLE] is not False:
    print("PRECONDITION LOST: numpy's broadcast_to result is no longer read-only")
    ok = False

print(ok)
print("oracle", platform.node(), np.__version__)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.trim().lines().rev();
    let provenance = lines.next().unwrap_or("").trim();
    let verdict = lines.next().unwrap_or("").trim();
    assert_eq!(
        verdict, "True",
        "view/copy aliasing should match numpy ({provenance}): {result}"
    );
    Ok(())
}
