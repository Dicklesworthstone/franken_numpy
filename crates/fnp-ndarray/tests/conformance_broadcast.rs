//! Conformance tests for fnp-ndarray broadcast/shape operations against NumPy oracle.
//!
//! These tests verify that fnp-ndarray's shape manipulation matches NumPy exactly.

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

// ─────────────────────────────────────────────────────────────────────────────
// broadcast_shapes conformance
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn conformance_broadcast_shapes_same() -> Result<(), String> {
    let numpy_result =
        numpy_oracle("import numpy as np; print(np.broadcast_shapes((3, 4), (3, 4)))")?;
    let fnp_result =
        fnp_ndarray::broadcast_shape(&[3, 4], &[3, 4]).map_err(|e| format!("{e:?}"))?;
    assert_eq!(
        format!("{fnp_result:?}"),
        numpy_result.replace("(", "[").replace(")", "]"),
        "broadcast_shapes((3,4), (3,4)) should match numpy"
    );
    Ok(())
}

#[test]
fn conformance_broadcast_shapes_scalar() -> Result<(), String> {
    let numpy_result = numpy_oracle("import numpy as np; print(np.broadcast_shapes((3, 4), ()))")?;
    let fnp_result = fnp_ndarray::broadcast_shape(&[3, 4], &[]).map_err(|e| format!("{e:?}"))?;
    assert_eq!(
        fnp_result,
        vec![3, 4],
        "broadcast with scalar should preserve shape"
    );
    assert_eq!(numpy_result, "(3, 4)");
    Ok(())
}

#[test]
fn conformance_broadcast_shapes_1d_expand() -> Result<(), String> {
    let numpy_result =
        numpy_oracle("import numpy as np; print(np.broadcast_shapes((3, 1), (1, 4)))")?;
    let fnp_result =
        fnp_ndarray::broadcast_shape(&[3, 1], &[1, 4]).map_err(|e| format!("{e:?}"))?;
    assert_eq!(fnp_result, vec![3, 4]);
    assert_eq!(numpy_result, "(3, 4)");
    Ok(())
}

#[test]
fn conformance_broadcast_shapes_different_ndim() -> Result<(), String> {
    let numpy_result =
        numpy_oracle("import numpy as np; print(np.broadcast_shapes((2, 3, 4), (4,)))")?;
    let fnp_result =
        fnp_ndarray::broadcast_shape(&[2, 3, 4], &[4]).map_err(|e| format!("{e:?}"))?;
    assert_eq!(fnp_result, vec![2, 3, 4]);
    assert_eq!(numpy_result, "(2, 3, 4)");
    Ok(())
}

#[test]
fn conformance_broadcast_shapes_incompatible() -> Result<(), String> {
    let numpy_result = numpy_oracle(
        r#"
import numpy as np
try:
    np.broadcast_shapes((3,), (4,))
    print("OK")
except ValueError:
    print("ERROR")
"#,
    )?;
    let fnp_result = fnp_ndarray::broadcast_shape(&[3], &[4]);
    assert!(fnp_result.is_err(), "incompatible shapes should error");
    assert_eq!(numpy_result, "ERROR");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// can_broadcast conformance
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn conformance_can_broadcast_true() -> Result<(), String> {
    let numpy_result = numpy_oracle(
        r#"
import numpy as np
try:
    np.broadcast_shapes((3, 1), (1, 4))
    print("True")
except ValueError:
    print("False")
"#,
    )?;
    let fnp_result = fnp_ndarray::can_broadcast(&[3, 1], &[1, 4]);
    assert!(fnp_result);
    assert_eq!(numpy_result, "True");
    Ok(())
}

#[test]
fn conformance_can_broadcast_false() -> Result<(), String> {
    let numpy_result = numpy_oracle(
        r#"
import numpy as np
try:
    np.broadcast_shapes((3,), (4,))
    print("True")
except ValueError:
    print("False")
"#,
    )?;
    let fnp_result = fnp_ndarray::can_broadcast(&[3], &[4]);
    assert!(!fnp_result);
    assert_eq!(numpy_result, "False");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// element_count conformance (np.prod on shape)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn conformance_element_count_1d() -> Result<(), String> {
    let numpy_result = numpy_oracle("import numpy as np; print(np.prod([5]))")?;
    let fnp_result = fnp_ndarray::element_count(&[5]).map_err(|e| format!("{e:?}"))?;
    assert_eq!(fnp_result, 5);
    assert_eq!(numpy_result, "5");
    Ok(())
}

#[test]
fn conformance_element_count_2d() -> Result<(), String> {
    let numpy_result = numpy_oracle("import numpy as np; print(np.prod([3, 4]))")?;
    let fnp_result = fnp_ndarray::element_count(&[3, 4]).map_err(|e| format!("{e:?}"))?;
    assert_eq!(fnp_result, 12);
    assert_eq!(numpy_result, "12");
    Ok(())
}

#[test]
fn conformance_element_count_3d() -> Result<(), String> {
    let numpy_result = numpy_oracle("import numpy as np; print(np.prod([2, 3, 4]))")?;
    let fnp_result = fnp_ndarray::element_count(&[2, 3, 4]).map_err(|e| format!("{e:?}"))?;
    assert_eq!(fnp_result, 24);
    assert_eq!(numpy_result, "24");
    Ok(())
}

#[test]
fn conformance_element_count_empty() -> Result<(), String> {
    let numpy_result = numpy_oracle("import numpy as np; print(np.prod([]))")?;
    let fnp_result = fnp_ndarray::element_count(&[]).map_err(|e| format!("{e:?}"))?;
    assert_eq!(fnp_result, 1);
    assert_eq!(numpy_result, "1.0");
    Ok(())
}

#[test]
fn conformance_element_count_with_zero() -> Result<(), String> {
    let numpy_result = numpy_oracle("import numpy as np; print(np.prod([3, 0, 4]))")?;
    let fnp_result = fnp_ndarray::element_count(&[3, 0, 4]).map_err(|e| format!("{e:?}"))?;
    assert_eq!(fnp_result, 0);
    assert_eq!(numpy_result, "0");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// fix_unknown_dimension conformance (reshape with -1)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn conformance_fix_unknown_dim_simple() -> Result<(), String> {
    let numpy_result =
        numpy_oracle("import numpy as np; a = np.arange(12); print(a.reshape(3, -1).shape)")?;
    let fnp_result =
        fnp_ndarray::fix_unknown_dimension(&[3, -1], 12).map_err(|e| format!("{e:?}"))?;
    assert_eq!(fnp_result, vec![3, 4]);
    assert_eq!(numpy_result, "(3, 4)");
    Ok(())
}

#[test]
fn conformance_fix_unknown_dim_first() -> Result<(), String> {
    let numpy_result =
        numpy_oracle("import numpy as np; a = np.arange(20); print(a.reshape(-1, 5).shape)")?;
    let fnp_result =
        fnp_ndarray::fix_unknown_dimension(&[-1, 5], 20).map_err(|e| format!("{e:?}"))?;
    assert_eq!(fnp_result, vec![4, 5]);
    assert_eq!(numpy_result, "(4, 5)");
    Ok(())
}

#[test]
fn conformance_fix_unknown_dim_3d() -> Result<(), String> {
    let numpy_result =
        numpy_oracle("import numpy as np; a = np.arange(24); print(a.reshape(2, -1, 4).shape)")?;
    let fnp_result =
        fnp_ndarray::fix_unknown_dimension(&[2, -1, 4], 24).map_err(|e| format!("{e:?}"))?;
    assert_eq!(fnp_result, vec![2, 3, 4]);
    assert_eq!(numpy_result, "(2, 3, 4)");
    Ok(())
}

#[test]
fn conformance_fix_unknown_dim_no_unknown() -> Result<(), String> {
    let fnp_result =
        fnp_ndarray::fix_unknown_dimension(&[3, 4], 12).map_err(|e| format!("{e:?}"))?;
    assert_eq!(fnp_result, vec![3, 4]);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// contiguous_strides conformance
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn conformance_contiguous_strides_c_order() -> Result<(), String> {
    use fnp_ndarray::MemoryOrder;
    let numpy_result = numpy_oracle(
        "import numpy as np; a = np.zeros((2, 3, 4), dtype=np.float64); print(a.strides)",
    )?;
    let fnp_result = fnp_ndarray::contiguous_strides(&[2, 3, 4], 8, MemoryOrder::C)
        .map_err(|e| format!("{e:?}"))?;
    assert_eq!(fnp_result, vec![96, 32, 8]);
    assert_eq!(numpy_result, "(96, 32, 8)");
    Ok(())
}

#[test]
fn conformance_contiguous_strides_f_order() -> Result<(), String> {
    use fnp_ndarray::MemoryOrder;
    let numpy_result = numpy_oracle(
        "import numpy as np; a = np.zeros((2, 3, 4), dtype=np.float64, order='F'); print(a.strides)",
    )?;
    let fnp_result = fnp_ndarray::contiguous_strides(&[2, 3, 4], 8, MemoryOrder::F)
        .map_err(|e| format!("{e:?}"))?;
    assert_eq!(fnp_result, vec![8, 16, 48]);
    assert_eq!(numpy_result, "(8, 16, 48)");
    Ok(())
}

#[test]
fn conformance_contiguous_strides_1d() -> Result<(), String> {
    use fnp_ndarray::MemoryOrder;
    let numpy_result = numpy_oracle(
        "import numpy as np; a = np.zeros((10,), dtype=np.float64); print(a.strides)",
    )?;
    let fnp_result =
        fnp_ndarray::contiguous_strides(&[10], 8, MemoryOrder::C).map_err(|e| format!("{e:?}"))?;
    assert_eq!(fnp_result, vec![8]);
    assert_eq!(numpy_result, "(8,)");
    Ok(())
}

#[test]
fn conformance_contiguous_strides_scalar() -> Result<(), String> {
    use fnp_ndarray::MemoryOrder;
    let fnp_result =
        fnp_ndarray::contiguous_strides(&[], 8, MemoryOrder::C).map_err(|e| format!("{e:?}"))?;
    assert!(fnp_result.is_empty());
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// broadcast_shapes (multi-input) conformance
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn conformance_broadcast_shapes_three() -> Result<(), String> {
    let numpy_result = numpy_oracle(
        "import numpy as np; print(np.broadcast_shapes((2, 1, 3), (1, 4, 1), (5, 1, 1, 1)))",
    )?;
    let fnp_result = fnp_ndarray::broadcast_shapes(&[&[2, 1, 3], &[1, 4, 1], &[5, 1, 1, 1]])
        .map_err(|e| format!("{e:?}"))?;
    assert_eq!(fnp_result, vec![5, 2, 4, 3]);
    assert_eq!(numpy_result, "(5, 2, 4, 3)");
    Ok(())
}

#[test]
fn conformance_broadcast_shapes_single() -> Result<(), String> {
    let numpy_result = numpy_oracle("import numpy as np; print(np.broadcast_shapes((3, 4)))")?;
    let fnp_result = fnp_ndarray::broadcast_shapes(&[&[3, 4]]).map_err(|e| format!("{e:?}"))?;
    assert_eq!(fnp_result, vec![3, 4]);
    assert_eq!(numpy_result, "(3, 4)");
    Ok(())
}

#[test]
fn conformance_broadcast_shapes_empty_list() -> Result<(), String> {
    let fnp_result = fnp_ndarray::broadcast_shapes(&[]).map_err(|e| format!("{e:?}"))?;
    assert!(fnp_result.is_empty());
    Ok(())
}
