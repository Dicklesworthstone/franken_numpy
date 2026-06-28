//! Conformance tests for ufunc edge cases (empty arrays, identity values).
//!
//! Tests verify PyUFunc reduce/accumulate/outer behaviors match NumPy on
//! edge cases like empty arrays and identity element semantics.

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
fn add_reduce_empty_returns_zero() {
    let fnp_result = numpy_oracle(&fnp_script("print(float(fnp.add.reduce([])))".to_string()))
        .expect("fnp add.reduce");
    let np_result =
        numpy_oracle("import numpy as np; print(float(np.add.reduce([])))").expect("np");
    assert_eq!(fnp_result, np_result, "add.reduce([]) should return 0");
    assert_eq!(fnp_result.trim(), "0.0", "identity is 0");
}

#[test]
fn multiply_reduce_empty_returns_one() {
    let fnp_result = numpy_oracle(&fnp_script(
        "print(float(fnp.multiply.reduce([])))".to_string(),
    ))
    .expect("fnp multiply.reduce");
    let np_result =
        numpy_oracle("import numpy as np; print(float(np.multiply.reduce([])))").expect("np");
    assert_eq!(fnp_result, np_result, "multiply.reduce([]) should return 1");
    assert_eq!(fnp_result.trim(), "1.0", "identity is 1");
}

#[test]
fn add_accumulate_empty_returns_empty() {
    let fnp_result = numpy_oracle(&fnp_script(
        "print(len(fnp.add.accumulate([])))".to_string(),
    ))
    .expect("fnp add.accumulate");
    let np_result =
        numpy_oracle("import numpy as np; print(len(np.add.accumulate([])))").expect("np");
    assert_eq!(
        fnp_result, np_result,
        "add.accumulate([]).shape should match"
    );
    assert_eq!(fnp_result.trim(), "0", "empty array");
}

#[test]
fn add_outer_empty_first() {
    let fnp_result = numpy_oracle(&fnp_script(
        "print(fnp.add.outer([], [1,2,3]).shape)".to_string(),
    ))
    .expect("fnp add.outer");
    let np_result =
        numpy_oracle("import numpy as np; print(np.add.outer([], [1,2,3]).shape)").expect("np");
    assert_eq!(
        fnp_result, np_result,
        "add.outer([],arr).shape should match"
    );
}

#[test]
fn add_outer_empty_second() {
    let fnp_result = numpy_oracle(&fnp_script(
        "print(fnp.add.outer([1,2,3], []).shape)".to_string(),
    ))
    .expect("fnp add.outer");
    let np_result =
        numpy_oracle("import numpy as np; print(np.add.outer([1,2,3], []).shape)").expect("np");
    assert_eq!(
        fnp_result, np_result,
        "add.outer(arr,[]).shape should match"
    );
}

#[test]
fn add_reduce_keepdims_empty() {
    let fnp_result = numpy_oracle(&fnp_script(
        "arr = np.array([]).reshape(0,3); print(fnp.add.reduce(arr, axis=0, keepdims=True).shape)"
            .to_string(),
    ))
    .expect("fnp");
    let np_result = numpy_oracle(
        "import numpy as np; arr = np.array([]).reshape(0,3); print(np.add.reduce(arr, axis=0, keepdims=True).shape)",
    )
    .expect("np");
    assert_eq!(fnp_result, np_result, "reduce keepdims on empty");
}

#[test]
fn add_identity_is_zero() {
    let fnp_result = numpy_oracle(&fnp_script("print(fnp.add.identity)".to_string())).expect("fnp");
    let np_result = numpy_oracle("import numpy as np; print(np.add.identity)").expect("np");
    assert_eq!(fnp_result, np_result, "add.identity should be 0");
}

#[test]
fn multiply_identity_is_one() {
    let fnp_result =
        numpy_oracle(&fnp_script("print(fnp.multiply.identity)".to_string())).expect("fnp");
    let np_result = numpy_oracle("import numpy as np; print(np.multiply.identity)").expect("np");
    assert_eq!(fnp_result, np_result, "multiply.identity should be 1");
}

#[test]
fn maximum_identity_is_none() {
    let fnp_result =
        numpy_oracle(&fnp_script("print(fnp.maximum.identity)".to_string())).expect("fnp");
    let np_result = numpy_oracle("import numpy as np; print(np.maximum.identity)").expect("np");
    assert_eq!(fnp_result, np_result, "maximum.identity should be None");
}

#[test]
fn add_nin_is_two() {
    let fnp_result = numpy_oracle(&fnp_script("print(fnp.add.nin)".to_string())).expect("fnp");
    let np_result = numpy_oracle("import numpy as np; print(np.add.nin)").expect("np");
    assert_eq!(fnp_result, np_result, "add.nin should be 2");
}

#[test]
fn add_nout_is_one() {
    let fnp_result = numpy_oracle(&fnp_script("print(fnp.add.nout)".to_string())).expect("fnp");
    let np_result = numpy_oracle("import numpy as np; print(np.add.nout)").expect("np");
    assert_eq!(fnp_result, np_result, "add.nout should be 1");
}

#[test]
fn add_nargs_is_three() {
    let fnp_result = numpy_oracle(&fnp_script("print(fnp.add.nargs)".to_string())).expect("fnp");
    let np_result = numpy_oracle("import numpy as np; print(np.add.nargs)").expect("np");
    assert_eq!(fnp_result, np_result, "add.nargs should be 3");
}

#[test]
fn ufunc_signature_has_x1_x2() {
    let fnp_result = numpy_oracle(&fnp_script(
        "import inspect; sig = inspect.signature(fnp.add); print('x1' in sig.parameters and 'x2' in sig.parameters)".to_string(),
    ))
    .expect("fnp");
    let np_result = numpy_oracle(
        "import numpy as np; import inspect; sig = inspect.signature(np.add); print('x1' in sig.parameters and 'x2' in sig.parameters)",
    )
    .expect("np");
    assert_eq!(fnp_result, np_result, "signature should have x1,x2");
    assert_eq!(fnp_result.trim(), "True");
}

#[test]
fn add_reduce_with_initial() {
    let fnp_result = numpy_oracle(&fnp_script(
        "print(float(fnp.add.reduce([1,2,3], initial=10)))".to_string(),
    ))
    .expect("fnp");
    let np_result =
        numpy_oracle("import numpy as np; print(float(np.add.reduce([1,2,3], initial=10)))")
            .expect("np");
    assert_eq!(fnp_result, np_result, "reduce with initial");
    assert_eq!(fnp_result.trim(), "16.0");
}

#[test]
fn add_reduce_empty_with_initial() {
    let fnp_result = numpy_oracle(&fnp_script(
        "print(float(fnp.add.reduce([], initial=42)))".to_string(),
    ))
    .expect("fnp");
    let np_result = numpy_oracle("import numpy as np; print(float(np.add.reduce([], initial=42)))")
        .expect("np");
    assert_eq!(fnp_result, np_result, "reduce empty with initial");
    assert_eq!(fnp_result.trim(), "42.0");
}

#[test]
fn maximum_minimum_accumulate_parallel_large_bit_exact_matches_numpy() -> Result<(), String> {
    // Above the 1<<21 gate the native flat f64 maximum/minimum.accumulate runs the
    // two-pass parallel prefix. max/min are associative and np_fmax/np_fmin replicate
    // numpy's tie rule (return SECOND arg) + NaN propagation, so it must be byte-exact
    // to numpy's serial accumulate — incl. signed-zero ties and a propagating NaN.
    let script = fnp_script(
        r#"
n = (1 << 21) + 65
rng = np.random.default_rng(0)

# random data with signed zeros + a NaN that must propagate forward
x = rng.standard_normal(n)
x[10] = -0.0; x[11] = 0.0; x[12] = -0.0; x[13] = 0.0
x[1234] = np.nan
ok = True
for ufm, npf in [(fnp.maximum, np.maximum), (fnp.minimum, np.minimum)]:
    a = ufm.accumulate(x); e = npf.accumulate(x)
    ok = ok and a.dtype == e.dtype and a.shape == e.shape and a.tobytes() == e.tobytes()

# signed-zero-heavy, no NaN (stresses the tie bit-pattern across block boundaries)
y = np.full(n, -0.0)
y[1::3] = 1.5
y[2::5] = -2.5
y[::7] = 0.0
for ufm, npf in [(fnp.maximum, np.maximum), (fnp.minimum, np.minimum)]:
    a = ufm.accumulate(y); e = npf.accumulate(y)
    ok = ok and a.tobytes() == e.tobytes()

print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "large parallel maximum/minimum.accumulate must be bit-identical to numpy"
    );
    Ok(())
}

#[test]
fn accumulate_extremum_f32_int_parallel_large_bit_exact_matches_numpy() -> Result<(), String> {
    // f32 + integer maximum/minimum.accumulate share the generic two-pass prefix.
    // f32: same NaN/signed-zero rules as f64. Integer: no NaN/promotion, output dtype
    // == input dtype. Must be byte-identical to numpy's serial accumulate above the gate.
    let script = fnp_script(
        r#"
n = (1 << 21) + 65
rng = np.random.default_rng(1)
ok = True

# float32 with signed zeros + propagating NaN
xf = rng.standard_normal(n).astype(np.float32)
xf[20] = np.float32(-0.0); xf[21] = np.float32(0.0); xf[22] = np.float32(-0.0)
xf[9999] = np.float32(np.nan)
for ufm, npf in [(fnp.maximum, np.maximum), (fnp.minimum, np.minimum)]:
    a = ufm.accumulate(xf); e = npf.accumulate(xf)
    ok = ok and a.dtype == e.dtype and a.shape == e.shape and a.tobytes() == e.tobytes()

# integer running max/min across several widths (output dtype preserved)
for dt in [np.int64, np.int32, np.int16, np.uint32, np.uint8]:
    xi = (rng.integers(-50, 50, n)).astype(dt)
    for ufm, npf in [(fnp.maximum, np.maximum), (fnp.minimum, np.minimum)]:
        a = ufm.accumulate(xi); e = npf.accumulate(xi)
        ok = ok and a.dtype == e.dtype and a.tobytes() == e.tobytes()

print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "large f32/int maximum/minimum.accumulate must be bit-identical to numpy"
    );
    Ok(())
}
