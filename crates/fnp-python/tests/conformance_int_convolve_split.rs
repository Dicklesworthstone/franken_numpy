//! Integer convolution parity for the split Python benchmark target.

use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyModule};

#[test]
fn int64_convolve_and_correlate_all_modes_match_numpy_bytes() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "fnp_python_int_convolve_test").expect("test module");
        fnp_python(&module).expect("initialize fnp_python test module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let a = numpy
            .call_method1("array", (vec![7_i64, -3, 4, 0, -8, 2],))
            .expect("convolve lhs");
        let v = numpy
            .call_method1("array", (vec![-2_i64, 5, 1],))
            .expect("convolve rhs");

        for operation in ["convolve", "correlate"] {
            for mode in ["full", "same", "valid"] {
                let actual = module
                    .getattr(operation)
                    .expect("fnp operation")
                    .call1((&a, &v, mode))
                    .expect("fnp integer operation");
                let expected = numpy
                    .getattr(operation)
                    .expect("numpy operation")
                    .call1((&a, &v, mode))
                    .expect("numpy integer operation");
                let actual_bytes: Vec<u8> = actual
                    .call_method0("tobytes")
                    .expect("fnp bytes")
                    .extract()
                    .expect("fnp byte vector");
                let expected_bytes: Vec<u8> = expected
                    .call_method0("tobytes")
                    .expect("numpy bytes")
                    .extract()
                    .expect("numpy byte vector");
                assert_eq!(
                    actual_bytes, expected_bytes,
                    "{operation} {mode} should match NumPy bytes"
                );
            }
        }
    });
}
