//! Float64 convolution parity for the split Python benchmark target.

use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyModule};

#[test]
fn float64_convolve_and_correlate_modes_match_numpy() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "fnp_python_convolve_test").expect("test module");
        fnp_python(&module).expect("initialize fnp_python test module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let a = numpy
            .call_method1("array", (vec![1.0_f64, -2.5, 3.0, 4.5, -1.0],))
            .expect("convolve lhs");
        let v = numpy
            .call_method1("array", (vec![0.5_f64, -1.0, 2.0],))
            .expect("convolve rhs");
        let array_equal = numpy.getattr("array_equal").expect("numpy array_equal");

        for (name, mode) in [("convolve", "same"), ("correlate", "valid")] {
            let actual = module
                .getattr(name)
                .expect("fnp operation")
                .call1((&a, &v, mode))
                .expect("fnp convolve operation");
            let expected = numpy
                .getattr(name)
                .expect("numpy operation")
                .call1((&a, &v, mode))
                .expect("numpy convolve operation");
            let equal: bool = array_equal
                .call1((&actual, &expected))
                .expect("convolve comparison")
                .extract()
                .expect("comparison boolean");
            assert!(equal, "{name} {mode} should match NumPy exactly");
        }
    });
}
