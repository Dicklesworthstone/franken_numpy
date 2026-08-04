//! Float16 matmul coverage for the split Python benchmark target.

use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyDict, PyModule};

#[test]
fn float16_matrix_batch_and_broadcast_matmul_match_numpy() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "fnp_python_f16_matmul_test").expect("test module");
        fnp_python(&module).expect("initialize fnp_python test module");
        let numpy = py.import("numpy").expect("numpy oracle");
        let array = numpy.getattr("array").expect("numpy array");
        let dtype = numpy.getattr("float16").expect("numpy float16");
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", dtype).expect("float16 dtype keyword");

        let matrix_a = array
            .call((vec![vec![1.0_f64, -2.0, 3.0], vec![4.0, 5.0, -6.0]],), Some(&kwargs))
            .expect("float16 matrix lhs");
        let matrix_b = array
            .call(
                (
                    vec![
                        vec![1.0_f64, 2.0],
                        vec![-3.0, 4.0],
                        vec![5.0, -6.0],
                    ],
                ),
                Some(&kwargs),
            )
            .expect("float16 matrix rhs");
        let batch_a = array
            .call(
                (
                    vec![
                        vec![vec![1.0_f64, -2.0, 3.0], vec![4.0, 5.0, -6.0]],
                        vec![vec![-1.0, 2.0, 0.5], vec![3.0, -4.0, 6.0]],
                    ],
                ),
                Some(&kwargs),
            )
            .expect("float16 batch lhs");
        let batch_b = array
            .call(
                (
                    vec![
                        vec![vec![1.0_f64, 2.0], vec![-3.0, 4.0], vec![5.0, -6.0]],
                        vec![vec![-2.0, 1.0], vec![3.0, -4.0], vec![0.5, 6.0]],
                    ],
                ),
                Some(&kwargs),
            )
            .expect("float16 batch rhs");
        let fnp_matmul = module.getattr("matmul").expect("fnp matmul");
        let numpy_matmul = numpy.getattr("matmul").expect("numpy matmul");
        let array_equal = numpy.getattr("array_equal").expect("numpy array_equal");

        for (lhs, rhs, label) in [
            (&matrix_a, &matrix_b, "matrix"),
            (&batch_a, &batch_b, "batch"),
            (&batch_a, &matrix_b, "broadcast"),
        ] {
            let actual = fnp_matmul.call1((lhs, rhs)).expect("fnp float16 matmul");
            let expected = numpy_matmul
                .call1((lhs, rhs))
                .expect("numpy float16 matmul");
            let equal: bool = array_equal
                .call1((&actual, &expected))
                .expect("float16 matmul comparison")
                .extract()
                .expect("comparison boolean");
            assert!(equal, "{label} float16 matmul should match NumPy exactly");
        }
    });
}
