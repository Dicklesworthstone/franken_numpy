use fnp_iter::{Nditer, NditerOptions, NditerOrder};
use fnp_ndarray::{broadcast_shapes, element_count};
use pyo3::Bound;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule, PyTuple};
use pyo3::wrap_pyfunction;

#[pyclass(name = "NditerStep", get_all, unsendable)]
#[derive(Clone)]
pub struct PyNditerStep {
    pub iterindex: usize,
    pub multi_index: Vec<usize>,
    pub linear_indices: Vec<usize>,
}

#[pyclass(name = "Nditer", unsendable)]
pub struct PyNditer {
    inner: Nditer,
}

#[pyclass(name = "FromPyFunc", unsendable)]
pub struct PyFromPyFunc {
    callable: Py<PyAny>,
    nin: usize,
    nout: usize,
}

enum VectorizeArgSlot {
    Fixed(Py<PyAny>),
    PendingArray(Py<PyAny>),
    Broadcast(Vec<Py<PyAny>>),
}

#[pyclass(name = "Vectorize", unsendable)]
pub struct PyVectorize {
    callable: Py<PyAny>,
    excluded: Vec<usize>,
}

impl PyFromPyFunc {
    fn new_checked(callable: Py<PyAny>, nin: usize, nout: usize, py: Python<'_>) -> PyResult<Self> {
        if nout == 0 {
            return Err(PyValueError::new_err(
                "frompyfunc: nout must be greater than zero",
            ));
        }

        if !callable.bind(py).is_callable() {
            return Err(PyTypeError::new_err(
                "frompyfunc: callable_obj must be callable",
            ));
        }

        Ok(Self {
            callable,
            nin,
            nout,
        })
    }

    fn call_bound(&self, py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
        if args.len() != self.nin {
            return Err(PyValueError::new_err(format!(
                "frompyfunc: expected {} input arrays, got {}",
                self.nin,
                args.len()
            )));
        }

        let numpy = py.import("numpy")?;
        let builtins = py.import("builtins")?;
        let object_dtype = builtins.getattr("object")?;

        let mut input_shapes = Vec::with_capacity(args.len());
        let mut input_arrays = Vec::with_capacity(args.len());

        for arg in args.iter() {
            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", &object_dtype)?;

            let array = numpy.call_method("asarray", (arg,), Some(&kwargs))?;
            let shape = array.getattr("shape")?.extract::<Vec<usize>>()?;

            input_shapes.push(shape);
            input_arrays.push(array.unbind());
        }

        let shape_refs: Vec<&[usize]> = input_shapes.iter().map(Vec::as_slice).collect();
        let out_shape =
            broadcast_shapes(&shape_refs).map_err(|err| PyValueError::new_err(err.to_string()))?;
        let flat_len =
            element_count(&out_shape).map_err(|err| PyValueError::new_err(err.to_string()))?;
        let output_shape = PyTuple::new(py, out_shape.iter().copied())?;
        let mut broadcasted_inputs = Vec::with_capacity(input_arrays.len());

        for array in input_arrays {
            let broadcasted =
                numpy.call_method1("broadcast_to", (array.bind(py), output_shape.clone()))?;
            let flattened = broadcasted.call_method1("reshape", (-1,))?;
            let list = flattened.call_method0("tolist")?;
            broadcasted_inputs.push(list.extract::<Vec<Py<PyAny>>>()?);
        }

        let mut outputs: Vec<Vec<Py<PyAny>>> = (0..self.nout)
            .map(|_| Vec::with_capacity(flat_len))
            .collect();

        for element_idx in 0..flat_len {
            let call_args = PyTuple::new(
                py,
                broadcasted_inputs
                    .iter()
                    .map(|values| values[element_idx].bind(py)),
            )?;
            let result = self.callable.bind(py).call1(call_args)?;

            if self.nout == 1 {
                outputs[0].push(result.unbind());
                continue;
            }

            if let Ok(values) = result.downcast::<PyTuple>() {
                if values.len() != self.nout {
                    return Err(PyValueError::new_err(format!(
                        "frompyfunc: expected {} outputs, got {}",
                        self.nout,
                        values.len()
                    )));
                }

                for (output, value) in outputs.iter_mut().zip(values.iter()) {
                    output.push(value.unbind());
                }
                continue;
            }

            if let Ok(values) = result.downcast::<PyList>() {
                if values.len() != self.nout {
                    return Err(PyValueError::new_err(format!(
                        "frompyfunc: expected {} outputs, got {}",
                        self.nout,
                        values.len()
                    )));
                }

                for (output, value) in outputs.iter_mut().zip(values.iter()) {
                    output.push(value.unbind());
                }
                continue;
            }

            return Err(PyTypeError::new_err(format!(
                "frompyfunc: callable must return a tuple or list with {} outputs",
                self.nout
            )));
        }

        let mut arrays = Vec::with_capacity(self.nout);
        for values in outputs {
            let list = PyList::new(py, values.iter().map(|value| value.bind(py)))?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", &object_dtype)?;
            let array = numpy.call_method("array", (list,), Some(&kwargs))?;
            let reshaped = array.call_method1("reshape", (&output_shape,))?;
            arrays.push(reshaped.unbind());
        }

        if self.nout == 1 {
            Ok(arrays.remove(0))
        } else {
            Ok(PyTuple::new(py, arrays.iter().map(|array| array.bind(py)))?
                .into_any()
                .unbind())
        }
    }
}

impl PyVectorize {
    fn new_checked(
        callable: Py<PyAny>,
        excluded: Option<Vec<usize>>,
        py: Python<'_>,
    ) -> PyResult<Self> {
        if !callable.bind(py).is_callable() {
            return Err(PyTypeError::new_err(
                "vectorize: callable_obj must be callable",
            ));
        }

        let mut excluded = excluded.unwrap_or_default();
        excluded.sort_unstable();
        excluded.dedup();

        Ok(Self { callable, excluded })
    }

    fn infer_output_dtype(
        py: Python<'_>,
        numpy: &Bound<'_, PyModule>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let probe = PyList::new(py, [value])?;
        Ok(numpy
            .call_method1("array", (probe,))?
            .getattr("dtype")?
            .unbind())
    }

    fn call_bound(&self, py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
        if args.is_empty() {
            return Err(PyValueError::new_err(
                "vectorize: need at least one input array",
            ));
        }

        let numpy = py.import("numpy")?;
        let mut vectorized_shapes = Vec::new();
        let mut slots = Vec::with_capacity(args.len());

        for (idx, arg) in args.iter().enumerate() {
            if self.excluded.binary_search(&idx).is_ok() {
                slots.push(VectorizeArgSlot::Fixed(arg.unbind()));
                continue;
            }

            let array = numpy.call_method1("asarray", (arg,))?;
            let shape = array.getattr("shape")?.extract::<Vec<usize>>()?;
            vectorized_shapes.push(shape);
            slots.push(VectorizeArgSlot::PendingArray(array.unbind()));
        }

        let out_shape = if vectorized_shapes.is_empty() {
            Vec::new()
        } else {
            let shape_refs: Vec<&[usize]> = vectorized_shapes.iter().map(Vec::as_slice).collect();
            broadcast_shapes(&shape_refs).map_err(|err| PyValueError::new_err(err.to_string()))?
        };
        let flat_len = if out_shape.is_empty() {
            1
        } else {
            element_count(&out_shape).map_err(|err| PyValueError::new_err(err.to_string()))?
        };
        let output_shape = PyTuple::new(py, out_shape.iter().copied())?;

        let mut prepared_slots = Vec::with_capacity(slots.len());
        for slot in slots {
            match slot {
                VectorizeArgSlot::Fixed(value) => {
                    prepared_slots.push(VectorizeArgSlot::Fixed(value))
                }
                VectorizeArgSlot::PendingArray(array) => {
                    let broadcasted = numpy
                        .call_method1("broadcast_to", (array.bind(py), output_shape.clone()))?;
                    let flattened = broadcasted.call_method1("reshape", (-1,))?;
                    let values = flattened
                        .call_method0("tolist")?
                        .extract::<Vec<Py<PyAny>>>()?;
                    prepared_slots.push(VectorizeArgSlot::Broadcast(values));
                }
                VectorizeArgSlot::Broadcast(_) => unreachable!("pending slots are prepared once"),
            }
        }

        let mut outputs: Vec<Vec<Py<PyAny>>> = Vec::new();
        let mut output_dtypes: Vec<Py<PyAny>> = Vec::new();

        for element_idx in 0..flat_len {
            let call_args = PyTuple::new(
                py,
                prepared_slots.iter().map(|slot| match slot {
                    VectorizeArgSlot::Fixed(value) => value.bind(py),
                    VectorizeArgSlot::Broadcast(values) => values[element_idx].bind(py),
                    VectorizeArgSlot::PendingArray(_) => unreachable!("pending slots are prepared"),
                }),
            )?;
            let result = self.callable.bind(py).call1(call_args)?;
            let values = if let Ok(tuple) = result.downcast::<PyTuple>() {
                tuple.iter().map(|value| value.unbind()).collect::<Vec<_>>()
            } else {
                vec![result.unbind()]
            };

            if outputs.is_empty() {
                outputs = (0..values.len())
                    .map(|_| Vec::with_capacity(flat_len))
                    .collect();
            } else if values.len() != outputs.len() {
                return Err(PyValueError::new_err(format!(
                    "vectorize: output arity changed from {} to {}",
                    outputs.len(),
                    values.len()
                )));
            }

            for (output_idx, value) in values.into_iter().enumerate() {
                if output_dtypes.len() == output_idx {
                    output_dtypes.push(Self::infer_output_dtype(py, &numpy, value.bind(py))?);
                }
                outputs[output_idx].push(value);
            }
        }

        let mut arrays = Vec::with_capacity(outputs.len());
        for (values, dtype) in outputs.into_iter().zip(output_dtypes) {
            let list = PyList::new(py, values.iter().map(|value| value.bind(py)))?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", dtype.bind(py))?;
            let array = numpy.call_method("array", (list,), Some(&kwargs))?;
            let reshaped = array.call_method1("reshape", (&output_shape,))?;
            arrays.push(reshaped.unbind());
        }

        if arrays.len() == 1 {
            Ok(arrays.remove(0))
        } else {
            Ok(PyTuple::new(py, arrays.iter().map(|array| array.bind(py)))?
                .into_any()
                .unbind())
        }
    }
}

#[pymethods]
impl PyNditer {
    #[new]
    #[pyo3(signature = (shape, item_size=8, order="C", external_loop=false))]
    fn new(
        shape: Vec<usize>,
        item_size: usize,
        order: &str,
        external_loop: bool,
    ) -> PyResult<Self> {
        let nditer_order = match order {
            "C" => NditerOrder::C,
            "F" => NditerOrder::F,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "order must be 'C' or 'F'",
                ));
            }
        };

        let options = NditerOptions {
            order: nditer_order,
            external_loop,
        };

        let inner = Nditer::new(shape, item_size, options)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{:?}", e)))?;

        Ok(PyNditer { inner })
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<PyNditerStep>> {
        if let Some(step) = slf.inner.next() {
            Ok(Some(PyNditerStep {
                iterindex: step.iterindex,
                multi_index: step.multi_index,
                linear_indices: step.linear_indices,
            }))
        } else {
            Ok(None)
        }
    }
}

#[pymethods]
impl PyFromPyFunc {
    #[new]
    fn new(callable_obj: Py<PyAny>, nin: usize, nout: usize, py: Python<'_>) -> PyResult<Self> {
        Self::new_checked(callable_obj, nin, nout, py)
    }

    #[getter]
    fn nin(&self) -> usize {
        self.nin
    }

    #[getter]
    fn nout(&self) -> usize {
        self.nout
    }

    fn __call__(&self, py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
        self.call_bound(py, args)
    }

    fn __repr__(&self) -> String {
        format!("FromPyFunc(nin={}, nout={})", self.nin, self.nout)
    }
}

#[pymethods]
impl PyVectorize {
    #[new]
    #[pyo3(signature = (callable_obj, excluded=None))]
    fn new(
        callable_obj: Py<PyAny>,
        excluded: Option<Vec<usize>>,
        py: Python<'_>,
    ) -> PyResult<Self> {
        Self::new_checked(callable_obj, excluded, py)
    }

    #[getter]
    fn excluded(&self) -> Vec<usize> {
        self.excluded.clone()
    }

    fn __call__(&self, py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
        self.call_bound(py, args)
    }

    fn __repr__(&self) -> String {
        format!("Vectorize(excluded={:?})", self.excluded)
    }
}

#[pyfunction]
fn frompyfunc(
    py: Python<'_>,
    callable_obj: Py<PyAny>,
    nin: usize,
    nout: usize,
) -> PyResult<PyFromPyFunc> {
    PyFromPyFunc::new_checked(callable_obj, nin, nout, py)
}

#[pyfunction]
#[pyo3(signature = (callable_obj, excluded=None))]
fn vectorize(
    py: Python<'_>,
    callable_obj: Py<PyAny>,
    excluded: Option<Vec<usize>>,
) -> PyResult<PyVectorize> {
    PyVectorize::new_checked(callable_obj, excluded, py)
}

#[pymodule]
fn fnp_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNditerStep>()?;
    m.add_class::<PyNditer>()?;
    m.add_class::<PyFromPyFunc>()?;
    m.add_class::<PyVectorize>()?;
    m.add_function(wrap_pyfunction!(frompyfunc, m)?)?;
    m.add_function(wrap_pyfunction!(vectorize, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{PyFromPyFunc, PyVectorize, fnp_python};
    use pyo3::IntoPyObject;
    use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyModule, PyTuple};
    use pyo3::{PyResult, Python};

    fn with_python(test: impl FnOnce(Python<'_>) -> PyResult<()>) {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            test(py).unwrap();
        });
    }

    fn numpy_available(py: Python<'_>) -> bool {
        py.import("numpy").is_ok()
    }

    fn object_dtype<'py>(py: Python<'py>) -> pyo3::Bound<'py, pyo3::types::PyAny> {
        py.import("builtins")
            .unwrap()
            .getattr("object")
            .expect("builtins.object should exist")
    }

    fn object_array<'py>(
        py: Python<'py>,
        values: impl IntoPyObject<'py>,
    ) -> pyo3::Bound<'py, pyo3::types::PyAny> {
        let numpy = py.import("numpy").unwrap();
        let kwargs = PyDict::new(py);
        kwargs
            .set_item("dtype", object_dtype(py))
            .expect("dtype should be accepted");
        numpy
            .call_method("array", (values,), Some(&kwargs))
            .expect("np.array should work")
    }

    fn repr_string(value: &pyo3::Bound<'_, pyo3::types::PyAny>) -> String {
        value.repr().unwrap().extract::<String>().unwrap()
    }

    #[test]
    fn module_exports_python_surface() {
        with_python(|py| {
            let module = PyModule::new(py, "fnp_python_test")?;
            fnp_python(&module)?;

            assert!(module.getattr("frompyfunc").is_ok());
            assert!(module.getattr("FromPyFunc").is_ok());
            assert!(module.getattr("vectorize").is_ok());
            assert!(module.getattr("Vectorize").is_ok());
            assert!(module.getattr("Nditer").is_ok());
            Ok(())
        });
    }

    #[test]
    fn frompyfunc_live_callable_matches_numpy_single_output() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let functools = py.import("functools")?;
            let operator = py.import("operator")?;
            let callable = functools
                .getattr("partial")?
                .call1((operator.getattr("mul")?, "x"))?
                .unbind();
            let ufunc = PyFromPyFunc::new_checked(callable.clone_ref(py), 1, 1, py)?;

            let values = object_array(py, vec![1, 2, 3]);
            let args = PyTuple::new(py, [values.clone()])?;

            let actual = ufunc.call_bound(py, &args)?;
            let numpy = py.import("numpy")?;
            let expected_args = PyTuple::new(py, [values])?;
            let expected = numpy
                .getattr("frompyfunc")?
                .call1((callable.bind(py), 1, 1))?
                .call1(expected_args)?;

            assert_eq!(
                actual.bind(py).getattr("shape")?.extract::<Vec<usize>>()?,
                vec![3]
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn frompyfunc_live_callable_matches_numpy_multi_output() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let builtins = py.import("builtins")?;
            let callable = builtins.getattr("divmod")?.unbind();
            let ufunc = PyFromPyFunc::new_checked(callable.clone_ref(py), 2, 2, py)?;

            let lhs = object_array(py, vec![10, 11, 12]);
            let rhs = object_array(py, vec![3]);
            let args = PyTuple::new(py, [lhs.clone(), rhs.clone()])?;

            let actual = ufunc.call_bound(py, &args)?;
            let numpy = py.import("numpy")?;
            let expected_args = PyTuple::new(py, [lhs, rhs])?;
            let expected = numpy
                .getattr("frompyfunc")?
                .call1((callable.bind(py), 2, 2))?
                .call1(expected_args)?;

            let actual_tuple = actual.bind(py).downcast::<PyTuple>()?;
            let expected_tuple = expected.downcast::<PyTuple>()?;
            assert_eq!(actual_tuple.len()?, 2);
            assert_eq!(expected_tuple.len()?, 2);

            for (actual_item, expected_item) in
                actual_tuple.try_iter()?.zip(expected_tuple.try_iter()?)
            {
                let actual_item = actual_item?;
                let expected_item = expected_item?;
                assert_eq!(
                    actual_item.getattr("shape")?.extract::<Vec<usize>>()?,
                    vec![3]
                );
                assert_eq!(
                    repr_string(&actual_item.call_method0("tolist")?),
                    repr_string(&expected_item.call_method0("tolist")?)
                );
            }
            Ok(())
        });
    }

    #[test]
    fn vectorize_live_callable_matches_numpy_single_output() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let functools = py.import("functools")?;
            let operator = py.import("operator")?;
            let callable = functools
                .getattr("partial")?
                .call1((operator.getattr("add")?, 10))?
                .unbind();
            let vectorized = PyVectorize::new_checked(callable.clone_ref(py), None, py)?;

            let values = object_array(py, vec![1, 2, 3]);
            let args = PyTuple::new(py, [values.clone()])?;

            let actual = vectorized.call_bound(py, &args)?;
            let numpy = py.import("numpy")?;
            let expected_args = PyTuple::new(py, [values])?;
            let expected = numpy
                .getattr("vectorize")?
                .call1((callable.bind(py),))?
                .call1(expected_args)?;

            assert_eq!(
                actual.bind(py).getattr("shape")?.extract::<Vec<usize>>()?,
                vec![3]
            );
            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }

    #[test]
    fn vectorize_live_callable_matches_numpy_multi_output() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let builtins = py.import("builtins")?;
            let callable = builtins.getattr("divmod")?.unbind();
            let vectorized = PyVectorize::new_checked(callable.clone_ref(py), None, py)?;

            let lhs = object_array(py, vec![10, 11, 12]);
            let rhs = object_array(py, vec![3]);
            let args = PyTuple::new(py, [lhs.clone(), rhs.clone()])?;

            let actual = vectorized.call_bound(py, &args)?;
            let numpy = py.import("numpy")?;
            let expected_args = PyTuple::new(py, [lhs, rhs])?;
            let expected = numpy
                .getattr("vectorize")?
                .call1((callable.bind(py),))?
                .call1(expected_args)?;

            let actual_tuple = actual.bind(py).downcast::<PyTuple>()?;
            let expected_tuple = expected.downcast::<PyTuple>()?;
            assert_eq!(actual_tuple.len()?, 2);
            assert_eq!(expected_tuple.len()?, 2);

            for (actual_item, expected_item) in
                actual_tuple.try_iter()?.zip(expected_tuple.try_iter()?)
            {
                let actual_item = actual_item?;
                let expected_item = expected_item?;
                assert_eq!(
                    actual_item.getattr("shape")?.extract::<Vec<usize>>()?,
                    vec![3]
                );
                assert_eq!(
                    repr_string(&actual_item.call_method0("tolist")?),
                    repr_string(&expected_item.call_method0("tolist")?)
                );
            }
            Ok(())
        });
    }

    #[test]
    fn vectorize_excluded_argument_matches_numpy() {
        with_python(|py| {
            if !numpy_available(py) {
                return Ok(());
            }

            let operator = py.import("operator")?;
            let callable = operator.getattr("add")?.unbind();
            let vectorized = PyVectorize::new_checked(callable.clone_ref(py), Some(vec![1]), py)?;

            let lhs = object_array(py, vec![1, 2, 3]);
            let scalar = 10i32.into_pyobject(py)?.unbind();
            let args = PyTuple::new(py, vec![lhs.clone().unbind(), scalar.clone_ref(py).into()])?;

            let actual = vectorized.call_bound(py, &args)?;
            let numpy = py.import("numpy")?;
            let expected_args = PyTuple::new(py, vec![lhs.unbind(), scalar.clone_ref(py).into()])?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("excluded", vec![1])?;
            let expected = numpy
                .call_method("vectorize", (callable.bind(py),), Some(&kwargs))?
                .call1(expected_args)?;

            assert_eq!(
                repr_string(&actual.bind(py).call_method0("tolist")?),
                repr_string(&expected.call_method0("tolist")?)
            );
            Ok(())
        });
    }
}
