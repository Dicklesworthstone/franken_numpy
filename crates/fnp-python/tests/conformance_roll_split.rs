//! Exact roll parity for the split Python benchmark target.

use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyModule};

#[test]
fn roll_flat_and_each_2d_axis_match_numpy_exactly() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "fnp_python_roll_test").expect("test module");
        fnp_python(&module).expect("initialize fnp_python test module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let input = numpy
            .call_method1("arange", (24_i64,))
            .expect("roll input")
            .call_method1("reshape", ((4_i64, 6_i64),))
            .expect("roll input shape");
        let fnp_roll = module.getattr("roll").expect("fnp roll");
        let numpy_roll = numpy.getattr("roll").expect("numpy roll");
        let array_equal = numpy.getattr("array_equal").expect("numpy array_equal");

        for (shift, axis, label) in [
            (5_i64, None, "flat"),
            (-1_i64, Some(0_i64), "axis0"),
            (7_i64, Some(1_i64), "axis1"),
        ] {
            let actual = match axis {
                Some(axis) => fnp_roll.call1((&input, shift, axis)),
                None => fnp_roll.call1((&input, shift)),
            }
            .expect("fnp roll");
            let expected = match axis {
                Some(axis) => numpy_roll.call1((&input, shift, axis)),
                None => numpy_roll.call1((&input, shift)),
            }
            .expect("numpy roll");
            let equal: bool = array_equal
                .call1((&actual, &expected))
                .expect("roll comparison")
                .extract()
                .expect("comparison boolean");
            assert!(equal, "{label} roll should match NumPy exactly");
        }
    });
}

fn numpy_oracle(script: &str) -> Result<String, String> {
    let output = std::process::Command::new("python3")
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

/// The per-lane / per-row roll routes batch whole lanes into >= 2 MiB tasks and fan out only from
/// 16 MiB (bead `deadlock-audit-vc4p4`; they went parallel from 2^16 elements, one item per lane).
/// A batch that mapped a lane to the wrong output slot, or a shift applied per task instead of per
/// lane, shows up as bytes that differ from numpy - checked below and above the floor, float64 and
/// byte-rotated dtypes, per-axis and 2-D multi-axis, negative and oversized shifts.
#[test]
fn roll_lanes_match_numpy_across_the_parallel_floor() -> Result<(), String> {
    let script = support::fnp_script(
        r#"
rng = np.random.default_rng(927)
bad = []
for rows, cols in ((512, 511), (2048, 1100), (4096, 1031)):
    f8 = rng.standard_normal((rows, cols))
    f8[::13, ::7] = np.nan
    f4 = f8.astype(np.float32)
    i2 = rng.integers(-300, 300, (rows, cols)).astype(np.int16)
    t3 = f8[: rows // 4 * 4].reshape(4, rows // 4, cols)
    for label, fn in [
        ("f8 ax1", lambda m: m.roll(f8, 7, axis=1)),
        ("f8 ax1 neg", lambda m: m.roll(f8, -3 * cols - 5, axis=1)),
        ("f8 ax0", lambda m: m.roll(f8, 11, axis=0)),
        ("f4 ax1", lambda m: m.roll(f4, 5, axis=1)),
        ("i2 ax-1", lambda m: m.roll(i2, cols + 2, axis=-1)),
        ("3d ax2", lambda m: m.roll(t3, 3, axis=2)),
        ("3d ax1", lambda m: m.roll(t3, -2, axis=1)),
        ("f8 2d", lambda m: m.roll(f8, (3, 5), axis=(0, 1))),
        ("f4 2d", lambda m: m.roll(f4, (-7, 2 * cols + 1), axis=(0, 1))),
        ("i2 2d", lambda m: m.roll(i2, (1, -1), axis=(0, 1))),
    ]:
        ours, theirs = fn(fnp), fn(np)
        if ours.dtype != theirs.dtype or ours.shape != theirs.shape or ours.tobytes() != theirs.tobytes():
            bad.append(f"{label} {rows}x{cols}")
print(bad if bad else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "roll lanes across the floor must match numpy bytes: {result}"
    );
    Ok(())
}
