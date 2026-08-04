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
