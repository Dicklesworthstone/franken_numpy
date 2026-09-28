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

mod support;
use support::fnp_script;

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

/// `out=` that OVERLAPS an input at another offset: numpy computes the result as if `out`
/// aliased nothing (it buffers the overlapping operand). The float64 binary zero-copy route
/// read x[i + k] after writing out[i], so `maximum(a[:-1], a[1:], out=a[1:])` answered a
/// running maximum and divide/minimum/power/arctan2/remainder fed their own results back in
/// (51 of the 1,368 cells of the full sweep, every size from 17 up). Exactly in place
/// (`out=a`) is safe and stays native; that control is swept too.
#[test]
fn out_overlapping_an_operand_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
rng = np.random.default_rng(3)
BINARY = ["add", "subtract", "multiply", "divide", "maximum", "minimum", "power", "hypot",
          "arctan2", "fmod", "copysign", "logaddexp", "floor_divide", "remainder", "fmax", "fmin"]
def outcome(run, m):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = run(m)
        except Exception as ex:
            return (type(ex).__name__, str(ex)[:60])
    return (r.dtype.str, r.shape, r.tobytes())
bad, cells = [], 0
for n in (17, 5000):
    for dt in ("f8", "f4", "i8"):
        base = (rng.random(n + 8) * 10 + 1).astype(dt)
        for name in BINARY:
            for shift in (1, 3, 7, "same"):
                def run(m, name=name, shift=shift):
                    a = base.copy()
                    x, y = a[:n], a[1:n + 1]
                    out = x if shift == "same" else a[shift:shift + n]
                    getattr(m, name)(x, y, out=out)
                    return a
                cells += 1
                if outcome(run, fnp) != outcome(run, np):
                    bad.append(f"{name} {dt} n={n} shift={shift}")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim().lines().last().unwrap_or(""),
        "384 []",
        "an out= overlapping an operand must give numpy's answer: {result}"
    );
    Ok(())
}

/// STRIDED VIEW OPERANDS - `x[::2]`, `x[::-1]`, a column, and a 2-D `a[:, ::2]` numpy flattens
/// without a copy - reach the extract routes as non-contiguous buffers, which are made contiguous
/// before they are read (bincount, unique, isin, searchsorted and 1-D float64 nan_to_num copy them
/// contiguous for their flat kernels, a Fortran-ordered 2-D included). A reader that took such a
/// buffer as contiguous memory would
/// return other elements, so every cell compares bytes with numpy; big-endian strided views take
/// the value cast.
#[test]
fn strided_view_operands_through_extract_routes_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(927)
bad, cells = [], 0
def same(label, ours, theirs):
    global cells
    cells += 1
    xs = ours if isinstance(ours, tuple) else (ours,)
    ys = theirs if isinstance(theirs, tuple) else (theirs,)
    for x, y in zip(xs, ys):
        if type(x) is not type(y):
            bad.append(label)
            return
        x, y = np.asarray(x), np.asarray(y)
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            bad.append(label)
            return
for n in (77, 4096, 300_001):
    base = rng.standard_normal(2 * n)
    ints = rng.integers(0, 1000, 2 * n)
    grid = rng.standard_normal((n // 64 + 1, 128))
    igrid = rng.integers(0, 1000, grid.shape)
    views = [
        ("x[::2]", base[::2], ints[::2]),
        ("x[::-1]", base[n:][::-1], ints[n:][::-1]),
        ("column", base.reshape(n, 2)[:, 1], ints.reshape(n, 2)[:, 1]),
        ("2-D [:, ::2]", grid[:, ::2], igrid[:, ::2]),
        ("2-D Fortran", np.asfortranarray(grid), np.asfortranarray(igrid)),
        ("big-endian x[::2]", base.astype(">f8")[::2], ints.astype(">i8")[::2]),
    ]
    for vname, v, iv in views:
        assert not v.flags["C_CONTIGUOUS"] and not iv.flags["C_CONTIGUOUS"]
        work = [
            ("median", lambda m: m.median(v)),
            ("median axis -1", lambda m: m.median(v, axis=-1)),
            ("percentile", lambda m: m.percentile(v, 30)),
            ("nanmedian", lambda m: m.nanmedian(v)),
            ("ptp", lambda m: m.ptp(v)),
            ("cumsum", lambda m: m.cumsum(v)),
            ("sort", lambda m: m.sort(v, axis=None)),
            ("unique ints", lambda m: m.unique(iv)),
            ("histogram", lambda m: m.histogram(v, bins=32)),
            ("isin", lambda m: m.isin(iv, np.arange(0, 1000, 7))),
            ("isin strided test", lambda m: m.isin(np.arange(0, 1000), iv)),
            ("searchsorted", lambda m: m.searchsorted(np.sort(base), v)),
        ]
        if iv.ndim == 1:
            work.append(("bincount", lambda m: m.bincount(iv)))
        for name, fn in work:
            same(f"{name} {vname} n={n}", fn(fnp), fn(np))
    # nan_to_num maps a 1-D strided float64 operand natively; its result layout must be numpy's.
    basen = base.copy()
    basen[::5] = np.nan
    basen[3::11] = np.inf
    basen[7::13] = -np.inf
    gridn = np.asfortranarray(grid)
    gridn[::3, ::5] = np.nan
    for vname, v in (("x[::2]", basen[::2]), ("x[::-1]", basen[n:][::-1]),
                     ("column", basen.reshape(n, 2)[:, 1]), ("2-D Fortran", gridn)):
        ours, theirs = fnp.nan_to_num(v), np.nan_to_num(v)
        same(f"nan_to_num {vname} n={n}", ours, theirs)
        if ours.strides != theirs.strides:
            bad.append(f"nan_to_num strides {vname} n={n}")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim().lines().last().unwrap_or(""),
        "240 []",
        "strided view operands must give numpy's bytes: {result}"
    );
    Ok(())
}

/// NON-C 2-D OPERANDS: a transpose, `a[::2]` and `a[:, ::2]` under dot / matmul (a non-C float64
/// operand goes to numpy's BLAS, which takes it as it is), isin across mixed numeric dtypes (numpy
/// promotes; nothing native copies first) and unique (the flattened copy is what numpy sorts). Bytes,
/// dtype, shape and result type must be numpy's in every layout, the C-ordered controls included.
#[test]
fn non_c_two_dimensional_operands_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(928)
bad, cells = [], 0
def same(label, ours, theirs):
    global cells
    cells += 1
    if type(ours) is not type(theirs):
        bad.append(label)
        return
    x, y = np.asarray(ours), np.asarray(theirs)
    if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
        bad.append(label)
base = rng.standard_normal((256, 256))
layouts = {
    "C": base,
    "F": np.asfortranarray(base),
    "a[::2]": rng.standard_normal((512, 256))[::2],
    "a[:, ::2]": rng.standard_normal((256, 512))[:, ::2],
}
right = rng.standard_normal((256, 48))
for name, a in layouts.items():
    grid = np.floor(a * 4)
    for label, fn in [
        ("dot", lambda m: m.dot(a, right)),
        ("dot by a transpose", lambda m: m.dot(a, a[:48].T)),
        ("matmul", lambda m: m.matmul(a, right)),
        ("matmul transposed left", lambda m: m.matmul(a.T, right)),
        ("isin float vs int", lambda m: m.isin(grid, np.arange(-5, 5))),
        ("isin int32 vs int64", lambda m: m.isin(grid.astype(np.int32), np.arange(-5, 5))),
        ("isin float vs float", lambda m: m.isin(grid, np.arange(-5.0, 5.0))),
        ("unique float", lambda m: m.unique(grid)),
        ("unique int", lambda m: m.unique(grid.astype(np.int64))),
    ]:
        same(f"{label} {name}", fn(fnp), fn(np))
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim().lines().last().unwrap_or(""),
        "36 []",
        "non-C 2-D operands must give numpy's bytes: {result}"
    );
    Ok(())
}
